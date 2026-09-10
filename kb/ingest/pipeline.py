"""
Orquestração da ingestão.

Os dois passes rodam em sequência, nunca sobrepostos, porque cada um carrega o
seu próprio modelo na GPU:

    Pass 1  PARSE + SEGMENT   Docling  → artifacts/ e SQLite
            (descarrega)
    Pass 2  EMBED             bge-m3   → índice vetorial
            (descarrega)

Entre eles a comunicação é por disco. É isso que torna o processo retomável: um
kill no meio do Pass 2 não perde o Pass 1, e trocar de modelo de embedding não
reprocessa nenhum PDF.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from kb import collection as collections_api
from kb.config import parse_version
from kb.db import get_conn, transaction
from kb.ingest.embed import embed_pending
from kb.ingest.parse import (
    export_markdown,
    is_parsed,
    parse_document,
    release_converter,
)
from kb.ingest.segment import (
    ChunkRecord,
    SectionRecord,
    release_chunker,
    segment_document,
    summarize,
)
from kb.utils import file_sha256, get_pdf_page_count, list_documents

logger = logging.getLogger(__name__)

# (fase, feito, total, mensagem)
ProgressFn = Callable[[str, int, int, str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expand_paths(paths: Sequence[Path | str]) -> list[Path]:
    """Aceita arquivo, pasta ou mistura; devolve PDFs únicos e ordenados."""
    found: list[Path] = []

    for raw in paths:
        found.extend(list_documents(Path(raw)))

    unique: dict[str, Path] = {str(path.resolve()): path for path in found}
    return sorted(unique.values())


def is_indexed(doc_id: str, conn: sqlite3.Connection | None = None) -> bool:
    """Documento já parseado, segmentado e com a assinatura de parse atual."""
    conn = conn or get_conn()

    row = conn.execute(
        "SELECT parse_version, n_chunks FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()

    if not row or not row["n_chunks"]:
        return False

    if row["parse_version"] != parse_version():
        return False

    return is_parsed(doc_id)


def register_document(
    doc_id: str,
    pdf_path: Path,
    n_pages: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or get_conn()

    with transaction(conn):
        conn.execute(
            """
            INSERT INTO documents(doc_id, filename, source_path, n_pages, status, created_at)
                 VALUES (?, ?, ?, ?, 'pending', ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                 source_path = excluded.source_path,
                 n_pages     = COALESCE(excluded.n_pages, documents.n_pages)
            """,
            (doc_id, pdf_path.name, str(pdf_path), n_pages, _now()),
        )


def persist_segments(
    doc_id: str,
    sections: list[SectionRecord],
    chunks: list[ChunkRecord],
    title: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """
    Grava sections e chunks numa transação só.

    Apaga o que existia antes do mesmo documento: re-segmentar é uma operação
    de substituição, não de acúmulo. Os triggers do FTS5 acompanham.
    """
    conn = conn or get_conn()

    with transaction(conn):
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))

        conn.executemany(
            """
            INSERT INTO sections(section_id, doc_id, parent_id, ord, level, heading,
                                 heading_path, page_start, page_end, n_chars, text)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.section_id, s.doc_id, s.parent_id, s.ord, s.level, s.heading,
                    s.heading_path, s.page_start, s.page_end, s.n_chars, s.text,
                )
                for s in sections
            ],
        )

        conn.executemany(
            """
            INSERT INTO chunks(chunk_id, doc_id, section_id, ord, page_start, page_end,
                               n_tokens, heading_path, text, embed_text, content_type, embedded)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            [
                (
                    c.chunk_id, c.doc_id, c.section_id, c.ord, c.page_start, c.page_end,
                    c.n_tokens, c.heading_path, c.text, c.embed_text, c.content_type,
                )
                for c in chunks
            ],
        )

        conn.execute(
            """
            UPDATE documents
               SET n_sections = ?, n_chunks = ?, parse_version = ?,
                   title = COALESCE(?, title), status = 'parsed', error = NULL
             WHERE doc_id = ?
            """,
            (len(sections), len(chunks), parse_version(), title, doc_id),
        )


def ingest_paths(
    paths: Sequence[Path | str],
    collection_id: str | None = None,
    on_progress: ProgressFn | None = None,
    force: bool = False,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Executa a ingestão completa e devolve o resumo do que aconteceu."""
    conn = conn or get_conn()
    files = expand_paths(paths)

    result: dict[str, Any] = {
        "n_files": len(files),
        "indexed": [],
        "skipped": [],
        "failed": [],
        "n_chunks": 0,
        "n_sections": 0,
    }

    if not files:
        logger.warning("Nenhum PDF encontrado em: %s", paths)
        return result

    touched: list[str] = []

    # ---------- Pass 1: Docling ----------
    try:
        for index, pdf_path in enumerate(files, start=1):
            doc_id = file_sha256(pdf_path)

            if on_progress:
                on_progress("parse", index - 1, len(files), pdf_path.name)

            if not force and is_indexed(doc_id, conn):
                logger.info("skip (já indexado): %s", pdf_path.name)
                result["skipped"].append({"doc_id": doc_id, "filename": pdf_path.name})
                touched.append(doc_id)
                _associate(collection_id, doc_id, conn)
                continue

            try:
                register_document(
                    doc_id, pdf_path, get_pdf_page_count(pdf_path), conn=conn
                )

                parse_document(
                    pdf_path,
                    doc_id,
                    on_progress=lambda done, total, msg: (
                        on_progress("parse", done, total, msg) if on_progress else None
                    ),
                    force=force,
                )

                sections, chunks = segment_document(doc_id)

                if not chunks:
                    raise RuntimeError("parse não produziu nenhum chunk")

                persist_segments(doc_id, sections, chunks, conn=conn)
                export_markdown(doc_id)
                _associate(collection_id, doc_id, conn)

                stats = summarize(sections, chunks)
                result["n_sections"] += stats["n_sections"]
                result["n_chunks"] += stats["n_chunks"]
                result["indexed"].append(
                    {"doc_id": doc_id, "filename": pdf_path.name, **stats}
                )
                touched.append(doc_id)

            except Exception as exc:
                logger.exception("Falha ingerindo %s", pdf_path.name)
                _mark_failed(doc_id, str(exc), conn)
                result["failed"].append(
                    {"doc_id": doc_id, "filename": pdf_path.name, "error": str(exc)}
                )
    finally:
        # Libera a GPU antes do Pass 2: a regra é um modelo por vez.
        release_converter()
        release_chunker()

    # ---------- Pass 2: bge-m3 ----------
    if touched:
        result["n_embedded"] = embed_pending(
            doc_ids=touched,
            on_progress=lambda done, total, msg: (
                on_progress("embed", done, total, msg) if on_progress else None
            ),
            conn=conn,
        )

    return result


def _associate(
    collection_id: str | None,
    doc_id: str,
    conn: sqlite3.Connection,
) -> None:
    if collection_id:
        collections_api.add_documents(collection_id, [doc_id], conn=conn)


def _mark_failed(doc_id: str, error: str, conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET status = 'failed', error = ? WHERE doc_id = ?",
            (error[:2000], doc_id),
        )
