"""
Interface de consulta semântica.

Módulo independente para fazer queries na base vetorial.
Pode ser importado de notebooks, scripts, APIs, etc.
"""

from __future__ import annotations

from processing.storage import create_collection


def query_base(query: str, n_results: int = 5) -> None:
    """
    Consulta semântica ao ChromaDB.

    Como a collection foi criada com OllamaEmbeddingFunction,
    o Chroma consegue embedar a query automaticamente.
    """
    collection, _ = create_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print("=" * 80)
        print(f"Resultado {i} | distância: {distance}")
        print(f"Arquivo: {meta.get('filename')}")
        print(f"Páginas do bloco: {meta.get('page_block_start')} - {meta.get('page_block_end')}")
        print()
        print(doc[:1500])
