"""
Camada de persistência — ChromaDB.

Responsável por criar/recuperar coleções e inserir chunks com embeddings.
Este módulo NÃO conhece o Docling nem o formato dos PDFs.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from tqdm.auto import tqdm

from processing.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    OLLAMA_URL,
)
from processing.utils import batched

logger = logging.getLogger(__name__)


def create_collection():
    """
    Cria ou recupera coleção persistente do Chroma
    com a embedding function do Ollama já configurada.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    ollama_ef = OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={
            "description": "Base vetorial de livros processados com Docling + Ollama",
            "embedding_model": EMBED_MODEL,
        },
    )

    return collection, ollama_ef


def insert_chunks(
    *,
    records: list[dict],
    collection,
    ollama_ef,
    label: str = "",
) -> int:
    """
    Gera embeddings e insere uma lista de records no Chroma.

    Cada record deve ter: {"id": str, "document": str, "metadata": dict}.

    Retorna a quantidade de chunks inseridos.
    """
    if not records:
        return 0

    inserted = 0

    total_batches = (len(records) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch in tqdm(
        batched(records, EMBED_BATCH_SIZE),
        total=total_batches,
        desc=f"Embedding/upload {label}".strip(),
        leave=False,
    ):
        ids = [r["id"] for r in batch]
        documents = [r["document"] for r in batch]
        metadatas = [r["metadata"] for r in batch]

        embeddings = ollama_ef(documents)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        inserted += len(batch)

    return inserted
