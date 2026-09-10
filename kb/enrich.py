"""
Enriquecimento delegado ao agente.

Nenhum LLM roda nesta máquina. Quando um resumo é útil, quem gera é o agente que
consome o MCP — ele já tem um modelo bem melhor do que caberia num 3060, e assim
a GPU local fica inteira para recuperação.

O mecanismo é *pull*, não *push*: o servidor expõe a fila de pendências como
tools comuns e o agente roda o laço. O caminho "oficial" do MCP para isso seria
Sampling (`sampling/createMessage`), mas ele foi deprecado na revisão 2026-07-28
da spec e o Claude Code não o implementa — pull funciona em qualquer cliente.

Escopo deliberadamente pequeno: resumo por **documento** é barato (dezenas de
chamadas) e alimenta o catálogo. Resumo por **seção** é caro (centenas) e só é
gerado sob demanda, porque a navegação já sai de graça do `get_outline()`, que
usa os headings extraídos pelo Docling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Literal

from kb.db import get_conn, transaction
from kb.utils import truncate

logger = logging.getLogger(__name__)

Kind = Literal["document", "section"]

# Teto do material enviado ao agente por item. Um livro inteiro não cabe (nem
# deve caber) no contexto de uma chamada de tool.
MAX_DOCUMENT_CONTEXT = 24000
MAX_SECTION_CONTEXT = 12000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_pending(
    kind: Kind = "document",
    limit: int = 1,
    collection_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    Devolve os próximos itens sem resumo, já com o texto necessário.

    O agente lê, resume e devolve por `submit`. Como o estado vive no SQLite, o
    laço pode ser interrompido e retomado a qualquer momento sem perder trabalho.
    """
    conn = conn or get_conn()

    if kind == "document":
        return _pending_documents(limit, collection_id, conn)

    return _pending_sections(limit, collection_id, conn)


def _pending_documents(
    limit: int,
    collection_id: str | None,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    sql = """
        SELECT d.doc_id, d.filename, d.title, d.n_pages, d.n_sections
          FROM documents d
         WHERE d.summary IS NULL AND d.status = 'indexed'
    """
    params: list[Any] = []

    if collection_id:
        sql += """
           AND EXISTS (SELECT 1 FROM document_collections dc
                        WHERE dc.doc_id = d.doc_id AND dc.collection_id = ?)
        """
        params.append(collection_id)

    sql += " ORDER BY d.filename LIMIT ?"
    params.append(limit)

    items = []
    for row in conn.execute(sql, params).fetchall():
        items.append(
            {
                "kind": "document",
                "target_id": row["doc_id"],
                "filename": row["filename"],
                "n_pages": row["n_pages"],
                "n_sections": row["n_sections"],
                # O sumário estrutural, não o texto integral: resumir um livro a
                # partir dos seus títulos é barato e costuma bastar.
                "outline": _outline_text(row["doc_id"], conn),
                "sample": _document_sample(row["doc_id"], conn),
                "instructions": (
                    "Resuma este documento em 3 a 6 frases, dizendo do que trata e "
                    "que tipo de pergunta ele responde. Depois liste de 5 a 12 "
                    "tópicos-chave. Baseie-se apenas no material fornecido; se não "
                    "for suficiente, diga isso em vez de inventar."
                ),
            }
        )

    return items


def _pending_sections(
    limit: int,
    collection_id: str | None,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    sql = """
        SELECT s.section_id, s.doc_id, s.heading_path, s.page_start, s.page_end,
               s.text, d.filename
          FROM sections s
          JOIN documents d ON d.doc_id = s.doc_id
         WHERE s.summary IS NULL AND s.n_chars > 400
    """
    params: list[Any] = []

    if collection_id:
        sql += """
           AND EXISTS (SELECT 1 FROM document_collections dc
                        WHERE dc.doc_id = s.doc_id AND dc.collection_id = ?)
        """
        params.append(collection_id)

    # Seções maiores primeiro: são as que mais ganham com um resumo.
    sql += " ORDER BY s.n_chars DESC LIMIT ?"
    params.append(limit)

    return [
        {
            "kind": "section",
            "target_id": row["section_id"],
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "heading_path": row["heading_path"],
            "pages": f"{row['page_start']}-{row['page_end']}",
            "text": truncate(row["text"] or "", MAX_SECTION_CONTEXT),
            "instructions": (
                "Resuma esta seção em 2 a 4 frases, preservando definições, "
                "fórmulas e resultados. Não acrescente nada que não esteja no texto."
            ),
        }
        for row in conn.execute(sql, params).fetchall()
    ]


def _outline_text(doc_id: str, conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT level, heading, page_start FROM sections"
        " WHERE doc_id = ? AND heading IS NOT NULL ORDER BY ord",
        (doc_id,),
    ).fetchall()

    return "\n".join(
        f"{'  ' * max(row['level'] - 1, 0)}- {row['heading']} (p. {row['page_start']})"
        for row in rows
    )


def _document_sample(doc_id: str, conn: sqlite3.Connection) -> str:
    """Amostra do começo do documento, para dar tom e vocabulário ao resumo."""
    rows = conn.execute(
        "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ord LIMIT 6",
        (doc_id,),
    ).fetchall()

    return truncate("\n\n".join(row["text"] for row in rows), MAX_DOCUMENT_CONTEXT)


def submit(
    kind: Kind,
    target_id: str,
    summary: str,
    key_topics: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Grava o resumo produzido pelo agente. Idempotente: sobrescreve."""
    conn = conn or get_conn()
    summary = (summary or "").strip()

    if not summary:
        return {"ok": False, "error": "resumo vazio"}

    if kind == "document":
        with transaction(conn):
            cursor = conn.execute(
                "UPDATE documents SET summary = ?, key_topics = ?, enriched_at = ?"
                " WHERE doc_id = ?",
                (
                    summary,
                    json.dumps(key_topics, ensure_ascii=False) if key_topics else None,
                    _now(),
                    target_id,
                ),
            )
    else:
        with transaction(conn):
            cursor = conn.execute(
                "UPDATE sections SET summary = ?, enriched_at = ? WHERE section_id = ?",
                (summary, _now(), target_id),
            )

    if cursor.rowcount == 0:
        return {"ok": False, "error": f"{kind} não encontrado: {target_id}"}

    logger.info("Enriquecimento gravado: %s %s", kind, target_id[:12])

    return {"ok": True, "kind": kind, "target_id": target_id}


def pending_counts(conn: sqlite3.Connection | None = None) -> dict[str, int]:
    conn = conn or get_conn()

    row = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM documents
                 WHERE summary IS NULL AND status = 'indexed')      AS documents,
               (SELECT COUNT(*) FROM sections
                 WHERE summary IS NULL AND n_chars > 400)           AS sections
        """
    ).fetchone()

    return {"documents": int(row["documents"]), "sections": int(row["sections"])}
