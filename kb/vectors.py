"""
Índice vetorial — camada descartável.

O Chroma aqui é deliberadamente burro: guarda `chunk_id`, o vetor e o `doc_id`
para filtro. Nada de texto (isso vive no SQLite) e nenhuma embedding function
acoplada (os vetores chegam prontos de `kb.models`). Assim a camada inteira pode
ser apagada e reconstruída a partir do SQLite, e trocada por LanceDB sem que o
retrieval perceba.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from kb.config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """Contrato mínimo. Implementar isto é tudo que um backend novo precisa."""

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        doc_ids: Sequence[str],
    ) -> None: ...

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]: ...

    def delete_docs(self, doc_ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class ChromaIndex:
    """Implementação sobre ChromaDB persistente."""

    def __init__(self, path: str | None = None, collection: str | None = None) -> None:
        import chromadb

        self._path = str(path or CHROMA_DIR)
        self._name = collection or CHROMA_COLLECTION

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._path)

        # Sem embedding_function: os vetores são calculados em kb.models, com
        # controle explícito de carga e descarga da VRAM.
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={
                "embedding_model": EMBED_MODEL,
                "hnsw:space": "cosine",
            },
        )

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        doc_ids: Sequence[str],
    ) -> None:
        if not ids:
            return

        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[{"doc_id": doc_id} for doc_id in doc_ids],
        )

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Devolve [(chunk_id, distância)], menor distância primeiro."""
        where: dict[str, Any] | None = None

        if doc_ids is not None:
            if not doc_ids:
                return []
            where = {"doc_id": {"$in": list(doc_ids)}}

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=where,
            include=["distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return list(zip(ids, distances))

    def delete_docs(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return

        self._collection.delete(where={"doc_id": {"$in": list(doc_ids)}})

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Apaga o índice inteiro. Seguro: é reconstruível a partir do SQLite."""
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"embedding_model": EMBED_MODEL, "hnsw:space": "cosine"},
        )


_index: ChromaIndex | None = None


def get_index() -> ChromaIndex:
    """Singleton preguiçoso — abrir o Chroma custa I/O."""
    global _index

    if _index is None:
        _index = ChromaIndex()

    return _index
