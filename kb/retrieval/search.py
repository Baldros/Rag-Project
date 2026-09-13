"""
Busca híbrida — vetorial + BM25, fundidos por RRF.

Os dois canais erram de formas diferentes: o vetorial acha paráfrase mas perde
termo técnico exato; o BM25 acha o termo exato mas não entende sinônimo. Fundir
os dois cobre as duas falhas, e a medição da Anthropic sobre esse arranjo mostra
queda de 49% na taxa de falha de recuperação — 67% com rerank.

O escopo é sempre resolvido no SQLite *antes* de tocar no índice vetorial: quem
decide o que é uma collection é a camada estrutural, não o Chroma.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from kb.collection import resolve_scope
from kb.config import FINAL_TOP_K, FUSION_TOP_K, RECALL_TOP_K, RRF_K
from kb.db import get_conn
from kb.utils import ascii_fold

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    section_id: str | None
    text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    content_type: str
    filename: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def citation(self) -> str:
        pages = _page_label(self.page_start, self.page_end)
        return f"{self.filename} | p. {pages}"


def _page_label(start: int | None, end: int | None) -> str:
    if start is None:
        return "?"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def clean_query(question: str) -> str:
    """
    Remove instruções de formato antes de buscar.

    "O que é X? Responda em 3 frases" deve recuperar por "O que é X?", não pelo
    pedido de formatação — que não existe em nenhum documento.
    """
    text = " ".join(question.strip().split())

    if "?" in text:
        return text.split("?", 1)[0].strip() + "?"

    patterns = [
        r"\b(responda|explique|resuma|cite)\b.*$",
        r"\b(answer|explain|summarize|cite)\b.*$",
        r"\bem\s+at[eé]\s+\d+\s+(frases?|linhas?|par[aá]grafos?).*$",
        r"\bin\s+(up\s+to\s+)?\d+\s+(sentences?|lines?|paragraphs?).*$",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;:")

    return cleaned or text


def fts_query(question: str) -> str:
    """
    Converte a pergunta em expressão MATCH do FTS5.

    Cada termo vai entre aspas porque a sintaxe do FTS5 trata `-`, `*`, `:`,
    `(`, `)` e `AND`/`OR`/`NOT` como operadores: passar a pergunta crua dispara
    erro de sintaxe ou, pior, uma busca silenciosamente errada.
    """
    terms = [
        term
        for term in re.findall(r"\w+", ascii_fold(question).lower())
        if len(term) > 2
    ]

    if not terms:
        return ""

    return " OR ".join(f'"{term}"' for term in terms)


def vector_search(
    query: str,
    top_k: int,
    doc_ids: Sequence[str] | None,
) -> list[tuple[str, float]]:
    from kb.models import embed_query
    from kb.vectors import get_index

    embedding = embed_query(query)
    return get_index().query(embedding, top_k=top_k, doc_ids=doc_ids)


def lexical_search(
    query: str,
    top_k: int,
    doc_ids: Sequence[str] | None,
    conn: sqlite3.Connection | None = None,
) -> list[tuple[str, float]]:
    """BM25 sobre o FTS5. Score menor = melhor, como o SQLite devolve."""
    conn = conn or get_conn()
    match = fts_query(query)

    if not match:
        return []

    sql = """
        SELECT c.chunk_id, bm25(chunks_fts, 1.0, 0.5) AS score
          FROM chunks_fts
          JOIN chunks c ON c.rowid = chunks_fts.rowid
         WHERE chunks_fts MATCH ?
    """
    params: list[Any] = [match]

    if doc_ids is not None:
        if not doc_ids:
            return []
        placeholders = ",".join("?" * len(doc_ids))
        sql += f" AND c.doc_id IN ({placeholders})"
        params.extend(doc_ids)

    sql += " ORDER BY score LIMIT ?"
    params.append(top_k)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("Busca léxica falhou (%s); seguindo só com a vetorial.", exc)
        return []

    return [(row["chunk_id"], float(row["score"])) for row in rows]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.

    Combina por *posição*, não por score: distância de cosseno e BM25 vivem em
    escalas incomparáveis, e normalizá-las seria arbitrário.
    """
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def load_chunks(
    chunk_ids: Sequence[str],
    conn: sqlite3.Connection | None = None,
) -> dict[str, Hit]:
    if not chunk_ids:
        return {}

    conn = conn or get_conn()
    placeholders = ",".join("?" * len(chunk_ids))

    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.doc_id, c.section_id, c.text, c.heading_path,
               c.page_start, c.page_end, c.content_type, d.filename
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
         WHERE c.chunk_id IN ({placeholders})
        """,
        list(chunk_ids),
    ).fetchall()

    return {
        row["chunk_id"]: Hit(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            text=row["text"],
            heading_path=row["heading_path"] or "",
            page_start=row["page_start"],
            page_end=row["page_end"],
            content_type=row["content_type"],
            filename=row["filename"],
        )
        for row in rows
    }


def search_chunks(
    query: str,
    collections: Sequence[str] | None = None,
    doc_ids: Sequence[str] | None = None,
    top_k: int = FINAL_TOP_K,
    use_rerank: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[Hit]:
    """Pipeline completo até os chunks finais, já reordenados."""
    conn = conn or get_conn()
    cleaned = clean_query(query)

    scope = resolve_scope(collections, doc_ids, conn=conn)
    if scope is not None and not scope:
        logger.info("Escopo vazio: nenhuma busca executada.")
        return []

    vector_hits = vector_search(cleaned, RECALL_TOP_K, scope)
    lexical_hits = lexical_search(cleaned, RECALL_TOP_K, scope, conn=conn)

    fused = rrf_fuse(
        [
            [chunk_id for chunk_id, _ in vector_hits],
            [chunk_id for chunk_id, _ in lexical_hits],
        ]
    )[:FUSION_TOP_K]

    if not fused:
        return []

    by_id = load_chunks([chunk_id for chunk_id, _ in fused], conn=conn)

    vector_scores = dict(vector_hits)
    lexical_scores = dict(lexical_hits)

    candidates: list[Hit] = []
    for chunk_id, rrf_score in fused:
        hit = by_id.get(chunk_id)
        if hit is None:
            continue

        hit.scores = {"rrf": round(rrf_score, 5)}
        if chunk_id in vector_scores:
            hit.scores["vector_distance"] = round(vector_scores[chunk_id], 4)
        if chunk_id in lexical_scores:
            hit.scores["bm25"] = round(lexical_scores[chunk_id], 3)

        candidates.append(hit)

    if use_rerank:
        from kb.retrieval.rerank import rerank_hits

        candidates = rerank_hits(cleaned, candidates)

    return candidates[:top_k]
