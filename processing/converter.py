"""
Conversão de PDF — Docling.

Responsável por converter blocos de páginas de um PDF
em chunks estruturados prontos para armazenamento.

Este módulo NÃO conhece o ChromaDB. Ele recebe um PDF
e retorna chunks. A separação permite trocar o Docling
por outro backend no futuro sem tocar no storage.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from processing.utils import (
    force_gc,
    normalize_metadata,
    stable_chunk_id,
)

logger = logging.getLogger(__name__)


def convert_pdf_block(
    *,
    pdf_path: Path,
    chunker: HybridChunker,
    file_hash: str,
    page_start: int,
    page_end: int,
) -> list[dict]:
    """
    Converte um bloco de páginas de um PDF em records prontos
    para inserção no storage.

    Retorna lista de dicts com: {"id": str, "document": str, "metadata": dict}.
    Lista vazia se a conversão falhar ou não gerar chunks.
    """
    # Cria um converter novo a cada bloco para garantir que
    # a memória interna dos modelos ONNX seja liberada.
    converter = DocumentConverter()

    try:
        result = converter.convert(
            pdf_path,
            page_range=(page_start, page_end),
            raises_on_error=False,
        )
    except Exception:
        del converter
        force_gc()
        raise

    if result.document is None:
        logger.warning(
            "Falha ao converter %s páginas %s-%s",
            pdf_path.name,
            page_start,
            page_end,
        )
        del converter
        force_gc()
        return []

    doc = result.document
    chunks = list(chunker.chunk(dl_doc=doc))

    # Libera o resultado do Docling e o converter o mais cedo possível
    del result
    del converter
    force_gc()

    if not chunks:
        logger.warning(
            "Nenhum chunk gerado para %s páginas %s-%s",
            pdf_path.name,
            page_start,
            page_end,
        )
        return []

    records = []

    for chunk_index, chunk in enumerate(chunks):
        contextualized_text = chunker.contextualize(chunk).strip()

        if not contextualized_text:
            continue

        raw_text = getattr(chunk, "text", "") or ""

        chunk_meta = {}
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "export_json_dict"):
            chunk_meta = chunk.meta.export_json_dict()

        doc_id = stable_chunk_id(
            file_hash=file_hash,
            page_start=page_start,
            page_end=page_end,
            chunk_index=chunk_index,
        )

        metadata = normalize_metadata({
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "file_hash": file_hash,
            "page_block_start": page_start,
            "page_block_end": page_end,
            "chunk_index_in_block": chunk_index,
            "raw_text": raw_text,
            "docling_meta": chunk_meta,
        })

        records.append({
            "id": doc_id,
            "document": contextualized_text,
            "metadata": metadata,
        })

    return records
