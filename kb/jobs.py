"""
Jobs — ingestão assíncrona em subprocesso.

Duas razões para não rodar a ingestão dentro do servidor MCP, e a segunda é a
que manda:

1. Converter um livro leva minutos. Uma chamada de tool não pode bloquear tanto.
2. O Docling carrega modelos de layout e de tabela na GPU. Se rodasse no mesmo
   processo do servidor, dividiria VRAM com o embedder e o reranker — que é
   exatamente o que a regra de "um modelo por vez" proíbe.

Em processo separado, ao terminar o processo morre e a VRAM volta limpa sem
depender de `empty_cache()`. O progresso trafega pela tabela `jobs` no SQLite.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from kb.config import ALLOWED_INGEST_ROOTS, PROJECT_ROOT
from kb.db import get_conn, transaction

logger = logging.getLogger(__name__)


class PathNotAllowed(ValueError):
    """Caminho fora das raízes autorizadas para ingestão."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_path(raw: str | Path) -> Path:
    """
    Confere que o caminho está sob uma raiz autorizada.

    A ingestão é disparada por um agente, que pode ser induzido pelo conteúdo de
    um documento a pedir a indexação de qualquer arquivo do disco. `resolve()`
    normaliza antes da comparação, de modo que `..` não escapa da raiz.
    """
    path = Path(raw).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

    for root in ALLOWED_INGEST_ROOTS:
        if path == root or root in path.parents:
            return path

    allowed = ", ".join(str(root) for root in ALLOWED_INGEST_ROOTS)
    raise PathNotAllowed(
        f"'{path}' está fora das raízes permitidas. Autorizadas: {allowed}. "
        "Ajuste KB_INGEST_ROOTS para incluir outra pasta."
    )


def create_job(
    kind: str,
    target: str,
    collection_id: str | None = None,
    total: int = 0,
    conn=None,
) -> str:
    conn = conn or get_conn()
    job_id = uuid.uuid4().hex[:12]

    with transaction(conn):
        conn.execute(
            "INSERT INTO jobs(job_id, kind, status, target, collection_id, total, created_at)"
            " VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (job_id, kind, target, collection_id, total, _now()),
        )

    return job_id


def update_job(job_id: str, conn=None, **fields: Any) -> None:
    if not fields:
        return

    conn = conn or get_conn()
    assignments = ", ".join(f"{key} = ?" for key in fields)

    with transaction(conn):
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?",
            [*fields.values(), job_id],
        )


def get_job(job_id: str, conn=None) -> dict[str, Any] | None:
    conn = conn or get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if not row:
        return None

    job = dict(row)
    job["running"] = job["status"] == "running" and _pid_alive(job.get("pid"))

    # Um job "running" cujo processo sumiu morreu sem conseguir se marcar.
    if job["status"] == "running" and not job["running"]:
        update_job(
            job_id,
            status="failed",
            error="processo terminou sem reportar conclusão",
            finished_at=_now(),
            conn=conn,
        )
        job["status"] = "failed"

    return job


def list_jobs(limit: int = 10, conn=None) -> list[dict[str, Any]]:
    conn = conn or get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()

    return [dict(row) for row in rows]


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False

    try:
        if sys.platform == "win32":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True

        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def cancel_job(job_id: str, conn=None) -> bool:
    job = get_job(job_id, conn=conn)

    if not job or job["status"] not in {"queued", "running"}:
        return False

    pid = job.get("pid")
    if pid and _pid_alive(pid):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                os.kill(int(pid), 15)
        except Exception:
            logger.exception("Falha ao encerrar o processo %s", pid)

    update_job(job_id, status="cancelled", finished_at=_now(), conn=conn)
    return True


def spawn_ingest(
    paths: Sequence[str | Path],
    collection_id: str | None = None,
    force: bool = False,
    conn=None,
) -> dict[str, Any]:
    """
    Valida os caminhos, cria o job e dispara o worker. Retorna imediatamente.

    O que volta já diz o que será feito (quantos arquivos, quantas páginas) para
    o agente poder responder ao usuário sem esperar a ingestão terminar.
    """
    conn = conn or get_conn()

    validated = [validate_path(path) for path in paths]

    # Import tardio: `kb.ingest` puxa docling, que é caro. O servidor MCP não
    # deve pagar esse custo só para responder uma busca.
    from kb.ingest.embed import count_pending
    from kb.ingest.pipeline import expand_paths, is_indexed
    from kb.utils import file_sha256, get_pdf_page_count

    files = expand_paths(validated)

    if not files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {', '.join(str(p) for p in validated)}"
        )

    pending, skipped = [], []
    doc_ids = []

    for path in files:
        doc_id = file_sha256(path)
        doc_ids.append(doc_id)

        if not force and is_indexed(doc_id, conn):
            skipped.append(path.name)
        else:
            pending.append(path)

    # Um documento parseado mas com vetores faltando (Pass 2 interrompido) conta
    # como "indexado" pelo critério estrutural. Sem esta checagem ele nunca
    # seria reembedado por este caminho.
    pending_vectors = count_pending(doc_ids, conn) if doc_ids else 0

    n_pages = 0
    for path in pending:
        try:
            n_pages += get_pdf_page_count(path)
        except Exception:
            logger.warning("Não consegui contar páginas de %s", path.name)

    job_id = create_job(
        kind="ingest",
        target="; ".join(str(path) for path in validated)[:500],
        collection_id=collection_id,
        total=len(pending),
        conn=conn,
    )

    if not pending and not pending_vectors:
        update_job(
            job_id,
            status="done",
            message="tudo já indexado",
            finished_at=_now(),
            conn=conn,
        )
        return {
            "job_id": job_id,
            "status": "done",
            "n_files": len(files),
            "n_pending": 0,
            "n_pages": 0,
            "skipped": skipped,
        }

    # Mesmo sem arquivo novo, o worker precisa rodar para completar embeddings
    # pendentes; ele pula o parse do que já está em cache.
    worker_paths = pending or files

    command = [
        sys.executable,
        "-m",
        "kb.ingest.worker",
        "--job-id",
        job_id,
        *(["--collection", collection_id] if collection_id else []),
        *(["--force"] if force else []),
        "--",
        *[str(path) for path in worker_paths],
    ]

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )

    update_job(job_id, pid=process.pid, conn=conn)

    logger.info("Job %s disparado (pid=%s) para %s arquivos", job_id, process.pid, len(pending))

    return {
        "job_id": job_id,
        "status": "queued",
        "n_files": len(files),
        "n_pending": len(pending),
        "n_pages": n_pages,
        "pending_vectors": pending_vectors,
        "skipped": skipped,
        "pid": process.pid,
    }
