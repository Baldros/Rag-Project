"""
Pass 3 — enriquecimento com LLM local.

Resumo é processamento de texto e pertence ao **sistema de ingestão**. Uma
versão anterior deste módulo delegava isso ao agente que consome o MCP, via fila
pull; estava errado, porque fazia a ingestão depender da camada de consumo — uma
ingestão rodada por CLI, sem nenhum agente conectado, sairia sem resumo nenhum.

Aqui o LLM roda local, sozinho na GPU, depois que o Docling e o embedder já
saíram. É o pass mais pesado e o mais dispensável: se o Ollama estiver fora do
ar, os Passes 1 e 2 continuam valendo e o enriquecimento fica pendente.

Escopo por padrão: **documento sim, seção não**. O resumo de documento custa
dezenas de chamadas e alimenta o catálogo; o de seção custa centenas e a
navegação já sai de graça do `get_outline()`, que usa os headings extraídos do
layout pelo Docling.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence

from kb.config import ENRICH_MIN_SECTION_CHARS, ENRICH_SECTIONS
from kb.db import get_conn, transaction
from kb.utils import truncate

logger = logging.getLogger(__name__)

Kind = Literal["document", "section"]
ProgressFn = Callable[[int, int, str], None]

MAX_DOCUMENT_CONTEXT = 12000
MAX_SECTION_CONTEXT = 8000


SYSTEM_PROMPT = (
    "Você resume material técnico para um catálogo de busca. "
    "Escreva em português, de forma direta e factual. "
    "Baseie-se exclusivamente no material fornecido: não complete lacunas com "
    "conhecimento próprio e não afirme nada que o texto não sustente. "
    "Não use marcadores nem títulos; escreva em prosa corrida."
)

DOCUMENT_PROMPT = """Abaixo estão o sumário estrutural e um trecho inicial de um documento.

SUMÁRIO:
{outline}

TRECHO:
{sample}

Escreva um resumo de 3 a 6 frases dizendo do que o documento trata e que tipo de
pergunta ele responde. Em seguida, numa última linha isolada, escreva:

TÓPICOS: tópico um; tópico dois; tópico três

com 5 a 12 tópicos-chave separados por ponto e vírgula."""

SECTION_PROMPT = """Seção "{heading}" (páginas {pages}) do documento {filename}:

{text}

Resuma em 2 a 4 frases, preservando definições, fórmulas e resultados."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- seleção


def _pending_documents(
    limit: int,
    doc_ids: Sequence[str] | None,
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:
    sql = """
        SELECT doc_id, filename, title, n_pages, n_sections
          FROM documents
         WHERE summary IS NULL AND status = 'indexed'
    """
    params: list[Any] = []

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        sql += f" AND doc_id IN ({placeholders})"
        params.extend(doc_ids)

    sql += " ORDER BY filename LIMIT ?"
    params.append(limit)

    return conn.execute(sql, params).fetchall()


def _pending_sections(
    limit: int,
    doc_ids: Sequence[str] | None,
    conn: sqlite3.Connection,
) -> list[sqlite3.Row]:
    sql = """
        SELECT s.section_id, s.doc_id, s.heading, s.heading_path,
               s.page_start, s.page_end, s.text, d.filename
          FROM sections s
          JOIN documents d ON d.doc_id = s.doc_id
         WHERE s.summary IS NULL AND s.n_chars >= ?
    """
    params: list[Any] = [ENRICH_MIN_SECTION_CHARS]

    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        sql += f" AND s.doc_id IN ({placeholders})"
        params.extend(doc_ids)

    # Seções maiores primeiro: são as que mais ganham com um resumo.
    sql += " ORDER BY s.n_chars DESC LIMIT ?"
    params.append(limit)

    return conn.execute(sql, params).fetchall()


def _outline_text(doc_id: str, conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT level, heading, page_start FROM sections"
        " WHERE doc_id = ? AND heading IS NOT NULL ORDER BY ord",
        (doc_id,),
    ).fetchall()

    return "\n".join(
        f"{'  ' * max(row['level'] - 1, 0)}- {row['heading']} (p. {row['page_start']})"
        for row in rows
    ) or "(sem títulos detectados)"


def _document_sample(doc_id: str, conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ord LIMIT 6",
        (doc_id,),
    ).fetchall()

    return truncate("\n\n".join(row["text"] for row in rows), MAX_DOCUMENT_CONTEXT)


def pending_counts(conn: sqlite3.Connection | None = None) -> dict[str, int]:
    conn = conn or get_conn()

    row = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM documents
                 WHERE summary IS NULL AND status = 'indexed')  AS documents,
               (SELECT COUNT(*) FROM sections
                 WHERE summary IS NULL AND n_chars >= ?)        AS sections
        """,
        (ENRICH_MIN_SECTION_CHARS,),
    ).fetchone()

    return {"documents": int(row["documents"]), "sections": int(row["sections"])}


# ---------------------------------------------------------------- escrita


def _write(
    kind: Kind,
    target_id: str,
    summary: str,
    key_topics: list[str] | None,
    conn: sqlite3.Connection,
) -> bool:
    summary = (summary or "").strip()
    if not summary:
        return False

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

    return cursor.rowcount > 0


def parse_response(text: str) -> tuple[str, list[str]]:
    """
    Separa o resumo da linha de tópicos.

    Formato delimitado em vez de JSON: um modelo pequeno erra chave e vírgula com
    frequência, e uma resposta malformada aqui custaria o item inteiro. Sem a
    linha `TÓPICOS:`, o texto todo vira resumo e os tópicos ficam vazios — perda
    parcial em vez de total.
    """
    match = re.search(r"^\s*T[ÓO]PICOS\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)

    if not match:
        return text.strip(), []

    summary = text[: match.start()].strip()
    topics = [
        topic.strip(" .;·-")
        for topic in re.split(r"[;\n]", match.group(1))
        if topic.strip(" .;·-")
    ]

    return summary, topics[:12]


# ---------------------------------------------------------------- o pass


def run_pass(
    doc_ids: Sequence[str] | None = None,
    sections: bool | None = None,
    limit: int = 500,
    on_progress: ProgressFn | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Pass 3 completo: documentos e, opcionalmente, seções.

    Nunca levanta por causa do LLM. Se o Ollama estiver fora, devolve o que
    conseguiu e reporta `skipped`, deixando o enriquecimento pendente para uma
    próxima rodada — uma ingestão não deve falhar por causa de um resumo.
    """
    conn = conn or get_conn()

    from kb import llm

    result: dict[str, Any] = {"documents": 0, "sections": 0, "failed": 0, "skipped": None}

    if not llm.is_available():
        result["skipped"] = "LLM indisponível; enriquecimento adiado"
        logger.warning("Pass 3 pulado: %s", result["skipped"])
        return result

    do_sections = ENRICH_SECTIONS if sections is None else sections

    try:
        result["documents"] = _enrich_documents(doc_ids, limit, on_progress, conn, llm, result)

        if do_sections:
            result["sections"] = _enrich_sections(doc_ids, limit, on_progress, conn, llm, result)
    finally:
        # Descarrega antes de devolver o controle: o pass acabou, a VRAM não
        # deve ficar ocupada pelo modelo mais pesado do sistema.
        llm.unload()

    logger.info("Pass 3 concluído: %s", result)
    return result


def _enrich_documents(
    doc_ids: Sequence[str] | None,
    limit: int,
    on_progress: ProgressFn | None,
    conn: sqlite3.Connection,
    llm,
    result: dict[str, Any],
) -> int:
    rows = _pending_documents(limit, doc_ids, conn)
    done = 0

    for index, row in enumerate(rows, start=1):
        prompt = DOCUMENT_PROMPT.format(
            outline=_outline_text(row["doc_id"], conn),
            sample=_document_sample(row["doc_id"], conn),
        )

        try:
            raw = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=700)
            summary, topics = parse_response(raw)

            if _write("document", row["doc_id"], summary, topics, conn):
                done += 1
        except Exception:
            logger.exception("Falha resumindo documento %s", row["filename"])
            result["failed"] += 1

        if on_progress:
            on_progress(index, len(rows), f"resumo: {row['filename']}")

    return done


def _enrich_sections(
    doc_ids: Sequence[str] | None,
    limit: int,
    on_progress: ProgressFn | None,
    conn: sqlite3.Connection,
    llm,
    result: dict[str, Any],
) -> int:
    rows = _pending_sections(limit, doc_ids, conn)
    done = 0

    for index, row in enumerate(rows, start=1):
        prompt = SECTION_PROMPT.format(
            heading=row["heading_path"] or "(sem título)",
            pages=f"{row['page_start']}-{row['page_end']}",
            filename=row["filename"],
            text=truncate(row["text"] or "", MAX_SECTION_CONTEXT),
        )

        try:
            raw = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=400)
            summary, _ = parse_response(raw)

            if _write("section", row["section_id"], summary, None, conn):
                done += 1
        except Exception:
            logger.exception("Falha resumindo seção %s", row["section_id"][:12])
            result["failed"] += 1

        if on_progress:
            on_progress(index, len(rows), f"seção: {(row['heading'] or '')[:40]}")

    return done
