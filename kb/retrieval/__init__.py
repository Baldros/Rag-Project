"""
API pública de retrieval.

Um ponto de entrada só, usado tanto pelo CLI quanto pelas tools do MCP.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Sequence

from kb.config import FINAL_TOP_K
from kb.db import get_conn
from kb.retrieval.expand import Passage, expand_to_sections, to_chunk_passages
from kb.retrieval.search import Hit, search_chunks

logger = logging.getLogger(__name__)

__all__ = ["Hit", "Passage", "search", "fetch_section", "fetch_pages", "get_outline"]


def search(
    query: str,
    collections: Sequence[str] | None = None,
    doc_ids: Sequence[str] | None = None,
    top_k: int = FINAL_TOP_K,
    expand: str = "section",
    use_rerank: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    Busca híbrida com rerank e expansão small-to-big.

    `expand="section"` devolve a seção-pai de cada acerto (o padrão, e o que faz
    a resposta ser utilizável); `expand="chunk"` devolve o trecho cru, útil para
    inspecionar a qualidade da recuperação.
    """
    conn = conn or get_conn()

    hits = search_chunks(
        query,
        collections=collections,
        doc_ids=doc_ids,
        top_k=top_k,
        use_rerank=use_rerank,
        conn=conn,
    )

    if not hits:
        return []

    passages = (
        expand_to_sections(hits, conn=conn)
        if expand == "section"
        else to_chunk_passages(hits)
    )

    return [passage.to_dict() for passage in passages]


def fetch_section(
    section_id: str,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Lê uma seção inteira pelo id devolvido numa citação."""
    conn = conn or get_conn()

    row = conn.execute(
        """
        SELECT s.*, d.filename
          FROM sections s
          JOIN documents d ON d.doc_id = s.doc_id
         WHERE s.section_id = ?
        """,
        (section_id,),
    ).fetchone()

    return dict(row) if row else None


def fetch_pages(
    doc_id: str,
    page_start: int,
    page_end: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Lê um intervalo de páginas.

    Serve para o agente conferir uma citação ou ler adiante depois de uma busca,
    sem precisar de nova query semântica.
    """
    conn = conn or get_conn()
    page_end = page_end or page_start

    rows = conn.execute(
        """
        SELECT chunk_id, heading_path, page_start, page_end, text, content_type
          FROM chunks
         WHERE doc_id = ?
           AND page_start <= ?
           AND COALESCE(page_end, page_start) >= ?
         ORDER BY ord
        """,
        (doc_id, page_end, page_start),
    ).fetchall()

    document = conn.execute(
        "SELECT filename, title, n_pages FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()

    return {
        "doc_id": doc_id,
        "filename": document["filename"] if document else None,
        "pages": f"{page_start}-{page_end}" if page_end != page_start else str(page_start),
        "n_chunks": len(rows),
        "text": "\n\n".join(row["text"] for row in rows),
        "headings": list(
            dict.fromkeys(row["heading_path"] for row in rows if row["heading_path"])
        ),
    }


def get_outline(
    doc_id: str,
    max_depth: int = 3,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Sumário navegável do documento.

    Vem da hierarquia de headings que o Docling extraiu do layout — sem LLM
    nenhum, sem custo de geração e sem risco de alucinar uma seção que não
    existe.
    """
    conn = conn or get_conn()

    document = conn.execute(
        "SELECT filename, title, n_pages, n_sections, summary, key_topics"
        " FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()

    if not document:
        return {"error": f"documento não encontrado: {doc_id}"}

    rows = conn.execute(
        """
        SELECT section_id, parent_id, ord, level, heading, heading_path,
               page_start, page_end, n_chars,
               summary IS NOT NULL AS has_summary
          FROM sections
         WHERE doc_id = ? AND level <= ?
         ORDER BY ord
        """,
        (doc_id, max_depth),
    ).fetchall()

    return {
        "doc_id": doc_id,
        "filename": document["filename"],
        "title": document["title"],
        "n_pages": document["n_pages"],
        "n_sections": document["n_sections"],
        "summary": document["summary"],
        "sections": [
            {
                "section_id": row["section_id"],
                "level": row["level"],
                "heading": row["heading"] or "(sem título)",
                "pages": f"{row['page_start']}-{row['page_end']}"
                if row["page_start"] != row["page_end"]
                else str(row["page_start"]),
                "n_chars": row["n_chars"],
                "has_summary": bool(row["has_summary"]),
            }
            for row in rows
        ],
    }
