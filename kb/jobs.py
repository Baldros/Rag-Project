"""Persistent job records for operations in the explicit ingestion session."""

from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from kb.config import ALLOWED_INGEST_ROOTS
from kb.db import get_conn, transaction
from kb import processes

logger = logging.getLogger(__name__)
ACTIVE = ("queued", "running", "cancelling", "finalizing")
TERMINAL = ("done", "partial", "failed", "cancelled")


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
    pid, started = processes.current()

    with transaction(conn):
        conn.execute(
            "INSERT INTO jobs(job_id, kind, status, target, collection_id, total, created_at, pid, owner_started)"
            " VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)",
            (job_id, kind, target, collection_id, total, _now(), pid, started),
        )

    return job_id


def update_job(job_id: str, conn=None, **fields: Any) -> bool:
    if not fields:
        return False

    if fields.get("status") in TERMINAL:
        status = fields.pop("status")
        return finish_job(job_id, status, conn=conn, **fields)

    conn = conn or get_conn()
    if "result" in fields:
        fields["result_json"] = json.dumps(fields.pop("result"), ensure_ascii=False)
    assignments = ", ".join(f"{key} = ?" for key in fields)

    with transaction(conn):
        cursor = conn.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ? "
            "AND status IN ('queued', 'running', 'cancelling', 'finalizing') "
            "AND cancel_requested_at IS NULL",
            [*fields.values(), job_id],
        )
    return cursor.rowcount > 0


def finish_job(job_id: str, status: str, conn=None, **fields: Any) -> bool:
    """Publish a terminal outcome once; late workers cannot overwrite it."""
    if status not in TERMINAL:
        raise ValueError(f"Invalid terminal job status: {status}")
    conn = conn or get_conn()
    fields.setdefault("finished_at", _now())
    fields.setdefault("cleanup_status", "unconfirmed")
    fields.setdefault("pass_name", None)
    if "result" in fields:
        fields["result_json"] = json.dumps(fields.pop("result"), ensure_ascii=False)
    assignments = ", ".join(f"{key} = ?" for key in fields)
    empty_result = ", result_json = COALESCE(result_json, '{}')" if "result_json" not in fields else ""
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE jobs SET status = CASE WHEN cancel_requested_at IS NOT NULL "
            "THEN 'cancelled' ELSE ? END, " + assignments + empty_result + " "
            "WHERE job_id = ? AND status IN ('queued', 'running', 'cancelling', 'finalizing')",
            [status, *fields.values(), job_id],
        )
    return cursor.rowcount > 0


def request_cancel(job_id: str, cancel_pid: int, cancel_owner_started: str, conn=None) -> bool:
    conn = conn or get_conn()
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE jobs SET status = 'cancelling', cancel_requested_at = ?, "
            "cancel_pid = ?, cancel_owner_started = ? WHERE job_id = ? "
            "AND status IN ('queued', 'running', 'finalizing') AND cancel_requested_at IS NULL",
            (_now(), cancel_pid, cancel_owner_started, job_id),
        )
    return cursor.rowcount > 0


def reconcile(conn=None) -> None:
    """Recover only proven orphans. Reads and runtime startup share this path.

    The cancellation coordinator owns external cleanup after killing the worker.
    If both processes disappear, preserve data and mark cleanup unconfirmed;
    recovery never unloads a shared model or resumes ingestion implicitly.
    """
    conn = conn or get_conn()
    # Serialize the snapshot with cancel intent and terminal publication. A
    # concurrent controller must not be mistaken for an abandoned cancellation.
    with transaction(conn, immediate=True):
        _reconcile_locked(conn)


def _reconcile_locked(conn) -> None:
    rows = conn.execute("SELECT * FROM jobs WHERE status IN "
                        "('queued', 'running', 'cancelling', 'finalizing')").fetchall()
    for row in rows:
        if processes.alive(row["pid"], row["owner_started"]):
            continue
        if row["cancel_requested_at"] and processes.alive(row["cancel_pid"], row["cancel_owner_started"]):
            continue
        finish_job(
            row["job_id"], "cancelled" if row["cancel_requested_at"] else "failed", conn=conn,
            error="Processo terminou sem concluir o job; dados já gravados foram preservados",
            cleanup_status="unconfirmed",
            cleanup_error="Processo encerrado; limpeza de recursos externos não confirmada",
        )


def get_job(job_id: str, conn=None) -> dict[str, Any] | None:
    conn = conn or get_conn()
    reconcile(conn)
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    if not row:
        return None

    return _decode_job(row)


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
    reconcile(conn)

    rows = conn.execute(
        "SELECT * FROM jobs WHERE status IN ('queued', 'running', 'cancelling', 'finalizing') ORDER BY created_at"
    ).fetchall()

    active: list[dict[str, Any]] = []

    for row in rows:
        job = _decode_job(row)

        if collection_ids and job["collection_id"] not in collection_ids:
            continue

        active.append(job)

    return active


def list_jobs(limit: int = 10, conn=None, offset: int = 0) -> list[dict[str, Any]]:
    conn = conn or get_conn()
    reconcile(conn)
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()

    return [_decode_job(row, compact=True) for row in rows]


def _decode_job(row, compact: bool = False) -> dict:
    job = dict(row)
    raw = job.pop("result_json", None)
    result = json.loads(raw) if raw else None
    if compact and result:
        result = {key: len(result.get(key, [])) for key in
                  ("indexed", "skipped", "failed", "pending", "pending_inputs")}
    job["result"] = result
    job["running"] = job["status"] in ACTIVE
    return job


def result_status(result: dict) -> str:
    available = [*result.get("indexed", []), *result.get("skipped", [])]
    if not available:
        return "failed"
    enrichment = result.get("enrichment", {})
    if (result.get("failed") or result.get("pending") or enrichment.get("failed") or enrichment.get("skipped") or
            any(item.get("quality", {}).get("status") == "partial" for item in available)):
        return "partial"
    return "done"


def _pid_alive(pid: int | None) -> bool:
    return processes.alive(pid, None)
