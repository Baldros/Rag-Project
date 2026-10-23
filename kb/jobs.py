"""Persistent job records for operations in the explicit ingestion session."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from kb.config import ALLOWED_INGEST_ROOTS
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


def active_jobs(
    collection_ids: Sequence[str] | None = None,
    conn=None,
) -> list[dict[str, Any]]:
    """
    Jobs de ingestão ainda em andamento, opcionalmente filtrados por collection.

    Existe para a busca poder avisar quando o escopo consultado está sendo
    ingerido naquele momento: os resultados sairiam parciais — só os chunks já
    embedados — e o agente não teria como perceber.
    """
    conn = conn or get_conn()

    rows = conn.execute(
        "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
    ).fetchall()

    active: list[dict[str, Any]] = []

    for row in rows:
        job = dict(row)

        # Um job cujo processo sumiu não está mais em andamento.
        if job["status"] == "running" and not _pid_alive(job.get("pid")):
            continue

        if collection_ids and job["collection_id"] not in collection_ids:
            continue

        active.append(job)

    return active


def list_jobs(limit: int = 10, conn=None, offset: int = 0) -> list[dict[str, Any]]:
    conn = conn or get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ? OFFSET ?",
        (limit, offset),
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
