"""
Segmentação — DoclingDocument em sections e chunks.

Duas unidades com papéis distintos, que é o que resolve a fragmentação medida no
pipeline antigo (chunk médio de 587 chars, ~150 tokens):

- **chunk**: pequeno, é por ele que se *encontra*. Vai para o índice vetorial e
  para o BM25.
- **section**: grande e coerente, é o que se *entrega* ao agente. Vive só no
  SQLite e é alcançada pela expansão small-to-big.

As sections são derivadas do `heading_path` que o próprio Docling anexa a cada
chunk, então a hierarquia vem do layout real do documento em vez de ser
reconstruída por heurística de texto.

Nota: o `HybridChunker` do docling 2.9x não expõe `overlap_tokens`. A falta de
sobreposição não prejudica aqui — o contexto que ela daria vem da seção-pai — e
ainda deixa o texto da seção livre de duplicação.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from kb.config import CHUNK_MAX_TOKENS, EMBED_MODEL
from kb.ingest.parse import iter_blocks
from kb.utils import stable_id

logger = logging.getLogger(__name__)

HEADING_SEP = " > "
NO_HEADING = "(sem seção)"

_chunker: Any = None
_tokenizer: Any = None


def get_chunker():
    """
    HybridChunker calibrado pelo tokenizer do modelo de embedding.

    Usar o tokenizer do bge-m3 (e não uma contagem por caracteres) é o que
    garante que nenhum chunk estoure a janela do modelo na hora de embedar.
    """
    global _chunker, _tokenizer

    if _chunker is None:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        _tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=EMBED_MODEL,
            max_tokens=CHUNK_MAX_TOKENS,
        )
        _chunker = HybridChunker(tokenizer=_tokenizer, merge_peers=True)

    return _chunker


def release_chunker() -> None:
    global _chunker, _tokenizer

    _chunker = None
    _tokenizer = None


@dataclass
class SectionRecord:
    section_id: str
    doc_id: str
    parent_id: str | None
    ord: int
    level: int
    heading: str | None
    heading_path: str
    page_start: int | None = None
    page_end: int | None = None
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    section_id: str
    ord: int
    page_start: int | None
    page_end: int | None
    n_tokens: int
    heading_path: str
    text: str
    embed_text: str
    content_type: str


def _pages_of(chunk: Any) -> tuple[int | None, int | None]:
    """Extrai o intervalo de páginas do `prov` do Docling, como inteiros.

    Fazer isso aqui, na ingestão, é o que evita o `json.loads` por resultado que
    o pipeline antigo pagava em toda query.
    """
    pages: list[int] = []

    for item in getattr(chunk.meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                pages.append(page_no)

    if not pages:
        return None, None

    return min(pages), max(pages)


def _content_type_of(chunk: Any) -> str:
    for item in getattr(chunk.meta, "doc_items", None) or []:
        label = getattr(getattr(item, "label", None), "value", None)
        if label == "table":
            return "table"

    return "text"


def _heading_path_of(chunk: Any) -> tuple[str, list[str]]:
    headings = [
        heading.strip()
        for heading in (getattr(chunk.meta, "headings", None) or [])
        if heading and heading.strip()
    ]

    if not headings:
        return NO_HEADING, []

    return HEADING_SEP.join(headings), headings


def _widen(
    current: tuple[int | None, int | None],
    new: tuple[int | None, int | None],
) -> tuple[int | None, int | None]:
    starts = [value for value in (current[0], new[0]) if value is not None]
    ends = [value for value in (current[1], new[1]) if value is not None]

    return (min(starts) if starts else None, max(ends) if ends else None)


def segment_document(doc_id: str) -> tuple[list[SectionRecord], list[ChunkRecord]]:
    """
    Lê o artefato do parse e produz sections + chunks.

    Não toca no PDF nem no SQLite: recebe artefato, devolve registros. Isso é o
    que permite re-segmentar a base inteira (mudança de `max_tokens`, por
    exemplo) sem reprocessar nenhum documento.
    """
    chunker = get_chunker()

    sections: list[SectionRecord] = []
    chunks: list[ChunkRecord] = []
    chunk_ord = 0
    section: SectionRecord | None = None

    for _, _, document in iter_blocks(doc_id):
        for chunk in chunker.chunk(dl_doc=document):
            text = (getattr(chunk, "text", "") or "").strip()
            if not text:
                continue

            embed_text = chunker.contextualize(chunk).strip() or text
            heading_path, headings = _heading_path_of(chunk)
            pages = _pages_of(chunk)

            # Seção nova a cada mudança de heading, e não por texto de heading:
            # num livro-texto "SOLUÇÃO" se repete depois de cada exemplo, e
            # agrupar por string funde passagens que não têm relação nenhuma.
            if section is None or section.heading_path != heading_path:
                section = SectionRecord(
                    section_id=stable_id(doc_id, "section", len(sections), heading_path),
                    doc_id=doc_id,
                    parent_id=_parent_id(doc_id, headings),
                    ord=len(sections),
                    level=len(headings),
                    heading=headings[-1] if headings else None,
                    heading_path=heading_path,
                )
                sections.append(section)

            section.parts.append(text)
            section.page_start, section.page_end = _widen(
                (section.page_start, section.page_end), pages
            )

            chunks.append(
                ChunkRecord(
                    chunk_id=stable_id(doc_id, "chunk", chunk_ord, heading_path),
                    doc_id=doc_id,
                    section_id=section.section_id,
                    ord=chunk_ord,
                    page_start=pages[0],
                    page_end=pages[1],
                    n_tokens=_count_tokens(embed_text),
                    heading_path=heading_path,
                    text=text,
                    embed_text=embed_text,
                    content_type=_content_type_of(chunk),
                )
            )
            chunk_ord += 1

    _resolve_parents(sections)

    logger.info(
        "segmentado %s | sections=%s chunks=%s",
        doc_id[:12],
        len(sections),
        len(chunks),
    )

    return sections, chunks


def _parent_id(doc_id: str, headings: list[str]) -> str | None:
    """Placeholder; a ligação real é feita em `_resolve_parents`."""
    return None


def _resolve_parents(sections: list[SectionRecord]) -> None:
    """
    Liga cada seção à ancestral mais próxima.

    Precisa ser um segundo passe porque a identidade da seção depende da
    posição: "A > B" pode ocorrer várias vezes no documento, e o pai correto é
    sempre a ocorrência anterior mais próxima, não a primeira.
    """
    stack: list[SectionRecord] = []

    for section in sections:
        while stack and stack[-1].level >= section.level:
            stack.pop()

        section.parent_id = stack[-1].section_id if stack else None
        stack.append(section)


def _count_tokens(text: str) -> int:
    if _tokenizer is None:
        return 0

    try:
        return _tokenizer.count_tokens(text)
    except Exception:  # pragma: no cover
        return 0


def summarize(sections: Iterable[SectionRecord], chunks: Iterable[ChunkRecord]) -> dict:
    """Estatísticas do resultado, para log e para o job."""
    chunk_list = list(chunks)
    token_counts = [chunk.n_tokens for chunk in chunk_list if chunk.n_tokens]

    return {
        "n_sections": len(list(sections)),
        "n_chunks": len(chunk_list),
        "n_tables": sum(1 for c in chunk_list if c.content_type == "table"),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1)
        if token_counts
        else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
    }
