"""
Meta-condição da base.

Alimenta a tool `status`, para o agente conseguir se orientar sozinho: o que
existe, se está íntegro e se alguma coisa precisa de atenção.

A checagem que mais importa é a de consistência entre o SQLite e o índice
vetorial. É a falha silenciosa clássica deste tipo de sistema: a base parece
inteira, as buscas retornam, e o que falta simplesmente nunca aparece nos
resultados — sem erro nenhum.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from kb.config import (
    ARTIFACTS_DIR,
    CHUNK_MAX_TOKENS,
    EMBED_MODEL,
    RERANK_ENABLED,
    RERANK_MODEL,
    STORE_DIR,
    parse_version,
)
from kb.db import get_conn

logger = logging.getLogger(__name__)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return round(total / 1024**2, 1)


def counts(conn: sqlite3.Connection | None = None) -> dict[str, int]:
    conn = conn or get_conn()

    row = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM collections)                      AS collections,
               (SELECT COUNT(*) FROM documents)                        AS documents,
               (SELECT COUNT(*) FROM sections)                         AS sections,
               (SELECT COUNT(*) FROM chunks)                           AS chunks,
               (SELECT COUNT(*) FROM chunks WHERE embedded = 1)        AS embedded,
               (SELECT COUNT(*) FROM chunks WHERE embedded = 0)        AS pending_embed,
               (SELECT COUNT(*) FROM documents WHERE status='failed')  AS failed_documents,
               (SELECT COUNT(*) FROM documents WHERE summary IS NULL)  AS pending_doc_summary,
               (SELECT COUNT(*) FROM sections  WHERE summary IS NULL)  AS pending_section_summary
        """
    ).fetchone()

    return {key: int(row[key]) for key in row.keys()}


def chunk_stats(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    conn = conn or get_conn()

    row = conn.execute(
        """
        SELECT ROUND(AVG(n_tokens), 1) AS avg_tokens,
               MAX(n_tokens)           AS max_tokens,
               SUM(content_type='table') AS tables,
               SUM(page_start IS NULL) AS without_page
          FROM chunks
        """
    ).fetchone()

    return {
        "avg_tokens": row["avg_tokens"] or 0,
        "max_tokens": row["max_tokens"] or 0,
        "tables": int(row["tables"] or 0),
        "chunks_without_page": int(row["without_page"] or 0),
    }


def consistency(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Compara o que o SQLite diz estar indexado com o que o índice realmente tem."""
    conn = conn or get_conn()

    expected = int(
        conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
    )

    try:
        from kb.vectors import get_index

        actual = get_index().count()
    except Exception as exc:
        return {"ok": False, "error": f"índice vetorial inacessível: {exc}"}

    drift = actual - expected

    result: dict[str, Any] = {
        "ok": drift == 0,
        "sqlite_embedded": expected,
        "vector_index": actual,
        "drift": drift,
    }

    if drift < 0:
        result["hint"] = (
            "faltam vetores no índice; rode `reindex` para recompor "
            "(não reprocessa nenhum PDF)"
        )
    elif drift > 0:
        result["hint"] = (
            "há vetores órfãos no índice, de documentos removidos; "
            "`reindex --reset` limpa"
        )

    return result


def orphan_artifacts() -> list[str]:
    """Artefatos de parse sem documento correspondente no SQLite."""
    if not ARTIFACTS_DIR.exists():
        return []

    conn = get_conn()
    known = {
        row["doc_id"] for row in conn.execute("SELECT doc_id FROM documents").fetchall()
    }

    return [
        item.name
        for item in ARTIFACTS_DIR.iterdir()
        if item.is_dir() and item.name not in known
    ]


def status(
    collection_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Retrato completo da base, para o agente se auto-orientar."""
    conn = conn or get_conn()

    from kb.collection import list_all
    from kb.jobs import list_jobs
    from kb.models import models_info, readiness, vram_info

    ready = readiness()

    report: dict[str, Any] = {
        # Primeiro campo de propósito: é a pergunta que mais importa logo depois
        # de subir o servidor — "já posso buscar?".
        "retrieval": {
            **ready,
            "hint": (
                None
                if ready["ready"]
                else "modelos ainda carregando; `search` responderá com warming_up até ficarem prontos"
                if ready["warming_up"]
                else "modelos não carregados; a primeira busca vai carregá-los sob demanda"
            ),
        },
        "counts": counts(conn),
        "chunks": chunk_stats(conn),
        # A checagem de consistência importa o chromadb, e durante o aquecimento
        # esse import fica serializado atrás do `import torch` da thread de
        # warmup — o lock de import do Python é global. Pular aqui é o que faz
        # `status` responder na hora justamente quando é mais útil: para saber
        # se já dá para buscar.
        "consistency": (
            {"deferred": "checagem adiada durante o aquecimento dos modelos"}
            if ready["warming_up"]
            else consistency(conn)
        ),
        "models": {
            "embedding": EMBED_MODEL,
            "reranker": RERANK_MODEL if RERANK_ENABLED else None,
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "parse_version": parse_version(),
            "loaded": models_info(),
        },
        "gpu": vram_info(),
        "disk_mb": {
            "total": _dir_size_mb(STORE_DIR),
            "artifacts": _dir_size_mb(ARTIFACTS_DIR),
        },
        "collections": list_all(conn),
        "recent_jobs": [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "pass": job["pass_name"],
                "progress": f"{job['progress']}/{job['total']}",
                "message": job["message"],
            }
            for job in list_jobs(5, conn)
        ],
    }

    if collection_id:
        report["collection"] = _collection_detail(collection_id, conn)

    report["warnings"] = _warnings(report)

    return report


def _collection_detail(collection_id: str, conn: sqlite3.Connection) -> dict[str, Any]:
    from kb.collection import documents_in, get

    meta = get(collection_id, conn=conn)
    if not meta:
        return {"error": f"collection não encontrada: {collection_id}"}

    documents = documents_in(collection_id, conn=conn)

    return {
        **meta,
        "n_documents": len(documents),
        "documents": [
            {
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "n_pages": document["n_pages"],
                "n_chunks": document["n_chunks"],
                "status": document["status"],
                "has_summary": bool(document["summary"]),
            }
            for document in documents
        ],
    }


def _warnings(report: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    counts_ = report["counts"]

    consistency_ = report["consistency"]
    if not consistency_.get("deferred") and not consistency_.get("ok"):
        messages.append(
            f"índice vetorial fora de sincronia ({consistency_.get('hint', '')})"
        )

    if counts_["pending_embed"]:
        messages.append(f"{counts_['pending_embed']} chunks sem vetor")

    if counts_["failed_documents"]:
        messages.append(f"{counts_['failed_documents']} documento(s) com falha de ingestão")

    if report["chunks"]["chunks_without_page"]:
        messages.append(
            f"{report['chunks']['chunks_without_page']} chunks sem página "
            "(citação fica sem âncora)"
        )

    if counts_["documents"] == 0:
        messages.append("base vazia — use `ingest` para adicionar documentos")

    return messages
