"""
Collections — recortes de conhecimento.

O usuário pode querer perguntar uma coisa de um lugar e outra de outro sem
carregar a biblioteca inteira. Um documento pertence a quantas collections
quiser sem duplicar vetor nem texto: a associação é uma linha em
`document_collections`.

Este módulo é também onde o escopo de busca é resolvido — collections viram
`doc_ids` no SQLite *antes* de qualquer chamada ao índice vetorial.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence

from kb.db import get_conn, transaction
from kb.utils import slugify

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(
    name: str,
    description: str | None = None,
    collection_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Cria uma collection. Se já existir com o mesmo id, devolve a existente."""
    conn = conn or get_conn()
    collection_id = collection_id or slugify(name)

    existing = get(collection_id, conn=conn)
    if existing:
        return existing

    with transaction(conn):
        conn.execute(
            "INSERT INTO collections(collection_id, name, description, created_at)"
            " VALUES (?, ?, ?, ?)",
            (collection_id, name, description, _now()),
        )

    logger.info("Collection criada: %s", collection_id)
    return get(collection_id, conn=conn)  # type: ignore[return-value]


def get(
    collection_id: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    conn = conn or get_conn()
    row = conn.execute(
        "SELECT * FROM collections WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()

    return dict(row) if row else None


def list_all(conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
    """Collections com contagens, para o agente saber o que existe."""
    conn = conn or get_conn()

    rows = conn.execute(
        """
        SELECT c.collection_id,
               c.name,
               c.description,
               c.created_at,
               COUNT(DISTINCT dc.doc_id)    AS n_documents,
               COALESCE(SUM(d.n_chunks), 0) AS n_chunks,
               COALESCE(SUM(d.n_pages), 0)  AS n_pages
        FROM collections c
        LEFT JOIN document_collections dc ON dc.collection_id = c.collection_id
        LEFT JOIN documents d             ON d.doc_id = dc.doc_id
        GROUP BY c.collection_id
        ORDER BY c.name
        """
    ).fetchall()

    return [dict(row) for row in rows]


def update(
    collection_id: str,
    name: str | None = None,
    description: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    conn = conn or get_conn()

    if name is None and description is None:
        return get(collection_id, conn=conn)

    fields, values = [], []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if description is not None:
        fields.append("description = ?")
        values.append(description)

    values.append(collection_id)

    with transaction(conn):
        conn.execute(
            f"UPDATE collections SET {', '.join(fields)} WHERE collection_id = ?",
            values,
        )

    return get(collection_id, conn=conn)


def delete(
    collection_id: str,
    delete_documents: bool = False,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Remove a collection.

    Por padrão só desfaz as associações — os documentos continuam na base e
    podem estar em outras collections. `delete_documents=True` apaga os que
    ficariam órfãos, inclusive seus vetores.
    """
    conn = conn or get_conn()
    orphans: list[str] = []

    if delete_documents:
        orphans = [
            row["doc_id"]
            for row in conn.execute(
                """
                SELECT dc.doc_id
                FROM document_collections dc
                WHERE dc.collection_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM document_collections other
                      WHERE other.doc_id = dc.doc_id
                        AND other.collection_id != dc.collection_id
                  )
                """,
                (collection_id,),
            ).fetchall()
        ]

    with transaction(conn):
        conn.execute(
            "DELETE FROM collections WHERE collection_id = ?",
            (collection_id,),
        )

        if orphans:
            placeholders = ",".join("?" * len(orphans))
            conn.execute(
                f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
                orphans,
            )

    if orphans:
        from kb.vectors import get_index

        get_index().delete_docs(orphans)

    return {"collection_id": collection_id, "deleted_documents": orphans}


def add_documents(
    collection_id: str,
    doc_ids: Sequence[str],
    conn: sqlite3.Connection | None = None,
) -> int:
    conn = conn or get_conn()

    if not doc_ids:
        return 0

    now = _now()
    with transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO document_collections(doc_id, collection_id, added_at)"
            " VALUES (?, ?, ?)",
            [(doc_id, collection_id, now) for doc_id in doc_ids],
        )

    return len(doc_ids)


def remove_documents(
    collection_id: str,
    doc_ids: Sequence[str],
    conn: sqlite3.Connection | None = None,
) -> int:
    conn = conn or get_conn()

    if not doc_ids:
        return 0

    placeholders = ",".join("?" * len(doc_ids))
    with transaction(conn):
        cursor = conn.execute(
            "DELETE FROM document_collections"
            f" WHERE collection_id = ? AND doc_id IN ({placeholders})",
            [collection_id, *doc_ids],
        )

    return cursor.rowcount


def resolve_scope(
    collections: Sequence[str] | None = None,
    doc_ids: Sequence[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[str] | None:
    """
    Traduz o escopo do usuário em `doc_ids`.

    `None` significa "toda a base" — o índice vetorial roda sem filtro, que é o
    caminho mais rápido. Lista vazia significa "escopo válido, porém sem nenhum
    documento", e a busca deve retornar vazio em vez de cair para a base toda.
    """
    conn = conn or get_conn()

    if not collections and not doc_ids:
        return None

    resolved: list[str] = list(doc_ids or [])

    if collections:
        placeholders = ",".join("?" * len(collections))
        rows = conn.execute(
            "SELECT DISTINCT doc_id FROM document_collections"
            f" WHERE collection_id IN ({placeholders})",
            list(collections),
        ).fetchall()
        resolved.extend(row["doc_id"] for row in rows)

    return sorted(set(resolved))


def delete_document(
    doc_id: str,
    purge_artifacts: bool = False,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Remove um documento da base inteira.

    Diferente de `remove_documents`, que só desfaz a associação com uma
    collection e deixa o documento vivo — ainda aparecendo em busca sem escopo.
    Aqui somem as linhas (sections e chunks caem por CASCADE, e os triggers do
    FTS5 acompanham) e os vetores.

    `purge_artifacts` também apaga o resultado do parse em disco. Por padrão ele
    fica: é a parte cara de reconstruir, e mantê-lo torna uma reingestão futura
    quase instantânea.
    """
    conn = conn or get_conn()

    row = conn.execute(
        "SELECT filename, n_chunks FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()

    if not row:
        return {"ok": False, "error": f"documento não encontrado: {doc_id}"}

    with transaction(conn):
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    from kb.vectors import get_index

    get_index().delete_docs([doc_id])

    artifacts_removed = False
    if purge_artifacts:
        import shutil

        from kb.ingest.parse import artifact_dir

        target = artifact_dir(doc_id)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            artifacts_removed = True

    logger.info("Documento removido: %s (%s chunks)", row["filename"], row["n_chunks"])

    return {
        "ok": True,
        "doc_id": doc_id,
        "filename": row["filename"],
        "chunks_removed": row["n_chunks"],
        "artifacts_removed": artifacts_removed,
    }


def documents_in(
    collection_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Catálogo de documentos, opcionalmente filtrado por collection."""
    conn = conn or get_conn()

    if collection_id:
        rows = conn.execute(
            """
            SELECT d.*
            FROM documents d
            JOIN document_collections dc ON dc.doc_id = d.doc_id
            WHERE dc.collection_id = ?
            ORDER BY d.filename
            """,
            (collection_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY filename").fetchall()

    return [dict(row) for row in rows]
