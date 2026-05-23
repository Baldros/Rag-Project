"""
Interface de consulta semântica.

Módulo independente para fazer queries na base vetorial.
Pode ser importado de notebooks, scripts, APIs, etc.
"""

from __future__ import annotations

from processing.rag import clean_metadata, retrieval_query
from processing.storage import create_collection
from tqdm import tqdm


def query_base(query: str, n_results: int = 5) -> None:
    """
    Consulta semântica ao ChromaDB.

    Como a collection foi criada com OllamaEmbeddingFunction,
    o Chroma consegue embedar a query automaticamente.
    """
    collection, _ = create_collection(create=False)

    results = collection.query(
        query_texts=[retrieval_query(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, distance) in tqdm(enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ), desc="Processando resultados..."):
        print("=" * 80)
        print(f"Resultado {i} | distância: {distance}")
        clean_meta = clean_metadata(meta)
        print(f"Arquivo: {clean_meta.get('filename')}")
        print(f"Páginas: {clean_meta.get('pages')}")
        print()
        print(doc[:1500])
