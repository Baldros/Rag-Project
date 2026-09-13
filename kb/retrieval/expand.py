"""
Expansao small-to-big.

Recupera-se pelo chunk (pequeno, preciso para achar) e entrega-se a secao-pai
(grande, coerente para raciocinar). E o que corrige a fragmentacao do pipeline
antigo sem mexer em nada da indexacao.

Varios chunks costumam cair na mesma secao; a deduplicacao por section_id evita
mandar o mesmo trecho tres vezes ao agente.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kb.config import MAX_SECTION_CHARS
from kb.db import get_conn
from kb.utils import truncate

if TYPE_CHECKING:
    from kb.retrieval.search import Hit

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    """Unidade entregue ao agente: texto grande com citacao verificavel."""

    doc_id: str
    filename: str
    section_id: str | None
    heading_path: str
    page_start: int | None
    page_end: int | None
    text: str
    content_type: str
    scores: dict[str, float]
    matched_chunks: int = 1

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return "?"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "section_id": self.section_id,
            "heading_path": self.heading_path,
            "pages": self.pages,
            "content_type": self.content_type,
            "matched_chunks": self.matched_chunks,
            "scores": self.scores,
            "text": self.text,
        }


def expand_to_sections(
    hits: list["Hit"],
    max_chars: int = MAX_SECTION_CHARS,
    conn: sqlite3.Connection | None = None,
) -> list[Passage]:
    """Troca cada chunk pela sua secao-pai, preservando a ordem do rerank."""
    conn = conn or get_conn()

    passages: list[Passage] = []
    seen: dict[str, Passage] = {}

    for hit in hits:
        if hit.section_id and hit.section_id in seen:
            seen[hit.section_id].matched_chunks += 1
            continue

        section = _load_section(hit.section_id, conn) if hit.section_id else None

        if section is None:
            # Sem secao (documento antigo ou chunk orfao): devolve o chunk.
            passage = Passage(
                doc_id=hit.doc_id,
                filename=hit.filename,
                section_id=None,
                heading_path=hit.heading_path,
                page_start=hit.page_start,
                page_end=hit.page_end,
                text=hit.text,
                content_type=hit.content_type,
                scores=hit.scores,
            )
        else:
            passage = Passage(
                doc_id=hit.doc_id,
                filename=hit.filename,
                section_id=section["section_id"],
                heading_path=section["heading_path"] or hit.heading_path,
                page_start=section["page_start"],
                page_end=section["page_end"],
                text=truncate(section["text"] or hit.text, max_chars),
                content_type=hit.content_type,
                scores=hit.scores,
            )
            seen[section["section_id"]] = passage

        passages.append(passage)

    return passages


def _load_section(section_id: str, conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT section_id, heading_path, page_start, page_end, text, summary"
        " FROM sections WHERE section_id = ?",
        (section_id,),
    ).fetchone()


def to_chunk_passages(hits: list["Hit"]) -> list[Passage]:
    """Modo sem expansao: devolve os chunks como estao."""
    return [
        Passage(
            doc_id=hit.doc_id,
            filename=hit.filename,
            section_id=hit.section_id,
            heading_path=hit.heading_path,
            page_start=hit.page_start,
            page_end=hit.page_end,
            text=hit.text,
            content_type=hit.content_type,
            scores=hit.scores,
        )
        for hit in hits
    ]
