"""
Pass 2 — embedding.

Roda depois que todo o parse terminou e o Docling já saiu da GPU. Lê os chunks
pendentes do SQLite, embeda em lote e grava no índice vetorial.

O pass é separado de propósito: trocar de modelo de embedding no futuro custa
re-rodar só este arquivo, sem tocar em nenhum PDF. Basta zerar `embedded` e
apagar o índice — ambos reconstruíveis a partir do SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable

from kb.config import EMBED_BATCH_SIZE, EMBED_MODEL
from kb.db import get_conn, transaction
from kb.models import embed_texts, unload_all
from kb.vectors import get_index

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


def count_pending(
    doc_ids: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    conn = conn or get_conn()

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        row = conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE embedded = 0 AND doc_id IN ({placeholders})",
            doc_ids,
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()

    return int(row[0])


def _fetch_batch(
    conn: sqlite3.Connection,
    doc_ids: list[str] | None,
    limit: int,
) -> list[sqlite3.Row]:
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        return conn.execute(
            "SELECT chunk_id, doc_id, embed_text FROM chunks"
            f" WHERE embedded = 0 AND doc_id IN ({placeholders})"
            " ORDER BY doc_id, ord LIMIT ?",
            [*doc_ids, limit],
        ).fetchall()

    return conn.execute(
        "SELECT chunk_id, doc_id, embed_text FROM chunks"
        " WHERE embedded = 0 ORDER BY doc_id, ord LIMIT ?",
        (limit,),
    ).fetchall()


def embed_pending(
    doc_ids: list[str] | None = None,
    on_progress: ProgressFn | None = None,
    conn: sqlite3.Connection | None = None,
    unload_when_done: bool = True,
) -> int:
    """
    Embeda todos os chunks pendentes. Devolve quantos foram processados.

    Retomável: o flag `embedded` só vira 1 depois que o vetor está no índice,
    então uma interrupção no meio nunca deixa chunk marcado sem vetor.
    """
    conn = conn or get_conn()
    index = get_index()

    total = count_pending(doc_ids, conn)
    if total == 0:
        logger.info("Nenhum chunk pendente de embedding.")
        return 0

    logger.info("Embedding de %s chunks com %s", total, EMBED_MODEL)

    done = 0

    try:
        while True:
            rows = _fetch_batch(conn, doc_ids, EMBED_BATCH_SIZE)
            if not rows:
                break

            vectors = embed_texts(
                [row["embed_text"] for row in rows],
                batch_size=EMBED_BATCH_SIZE,
            )

            # Vetor primeiro, flag depois: se cair aqui no meio, o chunk volta
            # na próxima rodada em vez de ficar marcado sem estar indexado.
            index.upsert(
                ids=[row["chunk_id"] for row in rows],
                embeddings=vectors,
                doc_ids=[row["doc_id"] for row in rows],
            )

            with transaction(conn):
                conn.executemany(
                    "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?",
                    [(row["chunk_id"],) for row in rows],
                )

            done += len(rows)

            if on_progress:
                on_progress(done, total, f"{done}/{total} chunks embedados")

        _mark_documents_indexed(conn, doc_ids)

    finally:
        if unload_when_done:
            unload_all()

    logger.info("Embedding concluído: %s chunks", done)
    return done


def _mark_documents_indexed(
    conn: sqlite3.Connection,
    doc_ids: list[str] | None,
) -> None:
    """Documento vira 'indexed' quando não sobrou nenhum chunk pendente."""
    params: list[str] = []
    clause = ""

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        clause = f" AND doc_id IN ({placeholders})"
        params = list(doc_ids)

    with transaction(conn):
        conn.execute(
            f"""
            UPDATE documents
               SET status = 'indexed',
                   embed_model = ?,
                   indexed_at = datetime('now')
             WHERE status != 'indexed'{clause}
               AND NOT EXISTS (
                   SELECT 1 FROM chunks
                    WHERE chunks.doc_id = documents.doc_id
                      AND chunks.embedded = 0
               )
            """,
            [EMBED_MODEL, *params],
        )


def reset_embeddings(
    doc_ids: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """
    Invalida os vetores para reindexar (troca de modelo, por exemplo).

    Só mexe na camada descartável: o texto e a estrutura ficam intactos, então
    reindexar não relê nenhum PDF.
    """
    conn = conn or get_conn()
    index = get_index()

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        with transaction(conn):
            cursor = conn.execute(
                f"UPDATE chunks SET embedded = 0 WHERE doc_id IN ({placeholders})",
                doc_ids,
            )
            conn.execute(
                f"UPDATE documents SET status = 'parsed' WHERE doc_id IN ({placeholders})",
                doc_ids,
            )
        index.delete_docs(doc_ids)
        return cursor.rowcount

    with transaction(conn):
        cursor = conn.execute("UPDATE chunks SET embedded = 0")
        conn.execute("UPDATE documents SET status = 'parsed'")

    index.reset()
    return cursor.rowcount
