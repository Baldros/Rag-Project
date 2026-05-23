"""
Helpers pequenos para consulta RAG sobre a collection Chroma existente.

Este modulo nao reindexa documentos. Ele so recupera chunks, limpa metadados
barulhentos e monta um contexto curto para o LLM.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from processing.config import RAG_MAX_CONTEXT_CHARS, RAG_TOP_K
from processing.storage import create_collection


RAG_SYSTEM_PROMPT = """Voce e um assistente de fisica.
Responda em portugues, de forma direta, usando apenas o contexto fornecido.
Se o contexto nao trouxer evidencia suficiente, diga que nao encontrou a resposta na base.
Nao invente autores, datas, formulas ou referencias.
Nao acrescente contexto historico, autoria ou datas a menos que a pergunta peca isso explicitamente.
Nao escreva secao de fontes; o sistema adicionara as fontes depois."""


@dataclass
class RetrievedChunk:
    document: str
    metadata: dict[str, Any]
    distance: float | None


def retrieve_chunks(query: str, n_results: int = RAG_TOP_K) -> list[RetrievedChunk]:
    """Busca semantica na collection persistida do Chroma."""
    collection, _ = create_collection(create=False)
    search_query = retrieval_query(query)

    if collection.count() == 0:
        raise RuntimeError("A collection Chroma esta vazia. Rode a indexacao antes de consultar.")

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
    Remove instrucoes de resposta antes do embedding.

    Ex.: "O que diz X? Responda em 3 frases e cite a fonte."
    deve recuperar por "O que diz X?", nao por "cite a fonte".
    """
    text = " ".join(question.strip().split())
    folded_text = ascii_fold(text)

    if "?" in text:
        return text.split("?", 1)[0].strip() + "?"

    cleanup_patterns = [
        r"\b(responda|responder)\b.*$",
        r"\b(cite|citar)\s+(a\s+)?fonte.*$",
        r"\bem\s+ate\s+\d+\s+frases?.*$",
    ]

    cleaned = folded_text
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;")

    return cleaned or text


def ascii_fold(text: str) -> str:
    """Remove acentos para regras simples de limpeza da pergunta."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def page_label(metadata: dict[str, Any]) -> str:
    """Retorna pagina real do Docling quando disponivel; senao, bloco de paginas."""
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

    return "desconhecida"


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Mantem so metadados uteis para resposta/citacao."""
    return {
        "filename": metadata.get("filename", "fonte desconhecida"),
        "pages": page_label(metadata),
        "chunk": metadata.get("chunk_index_in_block"),
    }


def format_context(
    chunks: list[RetrievedChunk],
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> str:
    """Monta contexto compacto, sem JSON do Docling nem hashes."""
    parts: list[str] = []
    used_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.document:
            continue

        meta = clean_metadata(chunk.metadata)
        distance = ""
        if chunk.distance is not None:
            distance = f" | distancia: {chunk.distance:.3f}"

        header = (
            f"[Fonte {index}: {meta['filename']} | paginas: {meta['pages']} "
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


def build_user_prompt(question: str, context: str) -> str:
    return f"""Contexto recuperado:
{context}

Pergunta:
{question}

Instrucao final:
Responda so ao que foi perguntado. Nao use conhecimento externo.
Se um trecho do contexto responder diretamente, prefira uma sintese curta desse trecho e pare.

Resposta:"""


def finalize_answer(
    raw_answer: str,
    question: str,
    chunks: list[RetrievedChunk],
    max_sources: int = 2,
) -> str:
    """Remove trechos obviamente nao solicitados e adiciona fontes confiaveis."""
    answer = strip_source_section(raw_answer).strip()
    answer = remove_unasked_provenance(answer, question).strip()

    if not answer:
        answer = "Nao encontrei uma resposta suficientemente sustentada no contexto recuperado."

    sources = format_sources(chunks, max_sources=max_sources)
    if sources:
        return f"{answer}\n\nFontes: {sources}"

    return answer


def strip_source_section(answer: str) -> str:
    return re.split(r"\bfontes?\s*:", answer, maxsplit=1, flags=re.IGNORECASE)[0]


def remove_unasked_provenance(answer: str, question: str) -> str:
    if asks_for_provenance(question):
        return answer

    provenance_pattern = re.compile(
        r"(descobert[ao]s?|formulad[ao]s?|estabelecid[ao]s?)\s+por|"
        r"\bBenjamin\s+Franklin\b|"
        r"\bCharles-Augustin\b|"
        r"\bautor(?:es)?\b|"
        r"\bdata\b|\bano\b",
        flags=re.IGNORECASE,
    )

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    kept = [sentence for sentence in sentences if not provenance_pattern.search(sentence)]

    return " ".join(kept)


def asks_for_provenance(question: str) -> bool:
    folded_question = ascii_fold(question)
    return bool(
        re.search(
            r"\b(quem|quando|autor(?:es)?|ano|data|historia|descobriu|formulou)\b",
            folded_question,
            flags=re.IGNORECASE,
        )
    )


def format_sources(chunks: list[RetrievedChunk], max_sources: int = 2) -> str:
    sources: list[str] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        metadata = clean_metadata(chunk.metadata)
        key = (str(metadata["filename"]), str(metadata["pages"]))
        if key in seen:
            continue

        seen.add(key)
        sources.append(f"[{metadata['filename']} | paginas: {metadata['pages']}]")

        if len(sources) >= max_sources:
            break

    return " ".join(sources)
