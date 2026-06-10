"""
Small helpers for querying the existing Chroma collection.

This module does not re-index documents. It retrieves chunks, cleans noisy
metadata, and builds a grounded prompt for the LLM.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any

from processing.config import RAG_MAX_CONTEXT_CHARS, RAG_TOP_K
from processing.storage import create_collection


class AnswerMode(str, Enum):
    """Controls how much the assistant should develop the answer."""

    OBJECTIVE = "objective"
    EXPLANATORY = "explanatory"
    CONVERSATIONAL = "conversational"


RAG_SYSTEM_PROMPT = """You are a physics study assistant that answers using a document base.

Your goal is to be accurate, helpful, and conversational.

Use the retrieved context as the primary source for specific claims such as
definitions, formulas, authors, dates, page references, and document-specific
statements.

Do not invent sources, pages, authors, dates, formulas, citations, or claims
that are not supported by the retrieved context.

You may enrich the answer with didactic explanations, reformulations,
intuitions, simple examples, and connections between concepts, as long as you do
not present those additions as if they were quoted from the source material.

If the retrieved context is insufficient, say what can be answered from it and
what is missing. Do not pretend that the document base supports something that it
does not support.

Write in English.
Do not write a separate sources section; the system will append sources after
the answer."""


ANSWER_MODE_INSTRUCTIONS = {
    AnswerMode.OBJECTIVE: """Answer directly and briefly.
Stay close to the retrieved context.
Use this mode for factual lookups, definitions, formulas, or page-specific questions.""",
    AnswerMode.EXPLANATORY: """Start with the direct answer, then develop the explanation.
Use the retrieved context as evidence, but also explain the intuition behind the idea.
When useful, add a simple example, contrast, or conceptual connection.
Make the answer rich enough for study, not just a short extraction.""",
    AnswerMode.CONVERSATIONAL: """Answer as a tutor in a dialogue.
Start with the direct answer, then unpack the idea step by step.
Use analogies, examples, conceptual bridges, and follow-up directions when useful.
Do not stop just because one retrieved passage contains a short answer.
End with one natural next-step question or suggestion when it would help the conversation continue.""",
}


@dataclass
class RetrievedChunk:
    document: str
    metadata: dict[str, Any]
    distance: float | None


def retrieve_chunks(query: str, n_results: int = RAG_TOP_K) -> list[RetrievedChunk]:
    """Run semantic search against the persisted Chroma collection."""
    collection, _ = create_collection(create=False)
    search_query = retrieval_query(query)

    if collection.count() == 0:
        raise RuntimeError("The Chroma collection is empty. Run indexing before querying.")

    results = collection.query(
        query_texts=[search_query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                document=(doc or "").strip(),
                metadata=meta or {},
                distance=distance,
            )
        )

    return chunks


def retrieval_query(question: str) -> str:
    """
    Remove answer-style instructions before embedding the query.

    Example: "What does X mean? Answer in 3 sentences and cite the source."
    should retrieve by "What does X mean?", not by "cite the source".
    """
    text = " ".join(question.strip().split())
    folded_text = ascii_fold(text)

    if "?" in text:
        return text.split("?", 1)[0].strip() + "?"

    cleanup_patterns = [
        r"\b(answer|respond|explain|reply)\b.*$",
        r"\b(responda|responder|explique)\b.*$",
        r"\b(cite|citar)\s+(the\s+)?(source|fonte).*$",
        r"\bin\s+up\s+to\s+\d+\s+sentences?.*$",
        r"\bem\s+ate\s+\d+\s+frases?.*$",
    ]

    cleaned = folded_text
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;")

    return cleaned or text


def ascii_fold(text: str) -> str:
    """Remove accents for simple rule-based query cleanup."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def page_label(metadata: dict[str, Any]) -> str:
    """Return Docling's real page number when available; otherwise page block."""
    pages: list[int] = []
    raw_docling_meta = metadata.get("docling_meta")

    if isinstance(raw_docling_meta, str):
        try:
            docling_meta = json.loads(raw_docling_meta)
        except json.JSONDecodeError:
            docling_meta = {}

        for item in docling_meta.get("doc_items", []):
            for prov in item.get("prov", []):
                page_no = prov.get("page_no")
                if isinstance(page_no, int):
                    pages.append(page_no)

    if pages:
        first = min(pages)
        last = max(pages)
        return str(first) if first == last else f"{first}-{last}"

    start = metadata.get("page_block_start")
    end = metadata.get("page_block_end")
    if start and end:
        return str(start) if start == end else f"{start}-{end}"

    return "unknown"


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata that is useful for answer citation."""
    return {
        "filename": metadata.get("filename", "unknown source"),
        "pages": page_label(metadata),
        "chunk": metadata.get("chunk_index_in_block"),
    }


def format_context(
    chunks: list[RetrievedChunk],
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> str:
    """Build compact context without Docling JSON or hashes."""
    parts: list[str] = []
    used_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.document:
            continue

        meta = clean_metadata(chunk.metadata)
        distance = ""
        if chunk.distance is not None:
            distance = f" | distance: {chunk.distance:.3f}"

        header = (
            f"[Source {index}: {meta['filename']} | pages: {meta['pages']} "
            f"| chunk: {meta['chunk']}{distance}]"
        )
        text = f"{header}\n{chunk.document}"

        remaining = max_chars - used_chars
        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0].rstrip()

        parts.append(text)
        used_chars += len(text) + 2

    return "\n\n".join(parts)


def infer_answer_mode(question: str) -> AnswerMode:
    """Infer whether the user wants a short, explanatory, or conversational answer."""
    folded_question = ascii_fold(question).lower()

    conversational_patterns = [
        r"\b(converse|talk|dialogue|discuss|walk me through)\b",
        r"\b(vamos conversar|conversa|dialogar|bate papo)\b",
    ]
    explanatory_patterns = [
        r"\b(explain|teach|develop|expand|intuition|example|why|how)\b",
        r"\b(explique|ensina|desenvolva|aprofund|intuicao|exemplo|por que|como)\b",
        r"\b(entender|understand)\b",
    ]
    objective_patterns = [
        r"\b(define|formula|page|cite|source|who|when)\b",
        r"\b(defina|formula|pagina|fonte|quem|quando)\b",
    ]

    if any(re.search(pattern, folded_question) for pattern in conversational_patterns):
        return AnswerMode.CONVERSATIONAL
    if any(re.search(pattern, folded_question) for pattern in explanatory_patterns):
        return AnswerMode.EXPLANATORY
    if any(re.search(pattern, folded_question) for pattern in objective_patterns):
        return AnswerMode.OBJECTIVE

    return AnswerMode.EXPLANATORY


def build_user_prompt(
    question: str,
    context: str,
    answer_mode: AnswerMode | str | None = None,
) -> str:
    mode = AnswerMode(answer_mode) if answer_mode else infer_answer_mode(question)
    mode_instruction = ANSWER_MODE_INSTRUCTIONS[mode]

    return f"""Retrieved context:
{context}

User question:
{question}

Answer mode: {mode.value}
Mode instructions:
{mode_instruction}

Final instructions:
- Answer the user's question first.
- Then enrich the answer only as much as the selected answer mode calls for.
- Separate document-grounded claims from general didactic explanation when needed.
- If the retrieved context is weak or incomplete, explicitly say what is missing.
- Write the full answer in English.

Answer:"""


def finalize_answer(
    raw_answer: str,
    question: str,
    chunks: list[RetrievedChunk],
    max_sources: int = 3,
) -> str:
    """Remove source sections generated by the model and append reliable sources."""
    answer = strip_source_section(raw_answer).strip()

    if not answer:
        answer = "I could not find an answer that is sufficiently supported by the retrieved context."

    sources = format_sources(chunks, max_sources=max_sources)
    if sources:
        return f"{answer}\n\nSources: {sources}"

    return answer


def strip_source_section(answer: str) -> str:
    return re.split(r"\b(sources?|fontes?)\s*:", answer, maxsplit=1, flags=re.IGNORECASE)[0]


def format_sources(chunks: list[RetrievedChunk], max_sources: int = 3) -> str:
    sources: list[str] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        metadata = clean_metadata(chunk.metadata)
        key = (str(metadata["filename"]), str(metadata["pages"]))
        if key in seen:
            continue

        seen.add(key)
        sources.append(f"[{metadata['filename']} | pages: {metadata['pages']}]")

        if len(sources) >= max_sources:
            break

    return " ".join(sources)
