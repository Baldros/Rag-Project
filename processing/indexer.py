"""
Orquestração do pipeline de indexação.

Conecta converter → storage, gerencia iteração sobre
blocos de páginas e arquivos PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docling.chunking import HybridChunker
from tqdm.auto import tqdm

from processing.config import PAGE_BLOCK_SIZE
from processing.converter import convert_pdf_block
from processing.storage import create_collection, insert_chunks
from processing.utils import (
    file_sha256,
    force_gc,
    get_pdf_page_count,
    list_files,
)

logger = logging.getLogger(__name__)


def index_pdf(
    *,
    pdf_path: Path,
    chunker: HybridChunker,
    collection,
    ollama_ef,
) -> int:
    """
    Indexa um PDF inteiro em blocos de páginas.
    Divide o PDF, converte cada bloco e insere no Chroma.
    """
    total_pages = get_pdf_page_count(pdf_path)
    file_hash = file_sha256(pdf_path)

    logger.info("Indexando %s | páginas=%s", pdf_path.name, total_pages)

    inserted_total = 0

    page_ranges = list(
        range(1, total_pages + 1, PAGE_BLOCK_SIZE)
    )

    for page_start in tqdm(
        page_ranges,
        desc=f"Blocos de páginas: {pdf_path.name}",
    ):
        page_end = min(page_start + PAGE_BLOCK_SIZE - 1, total_pages)

        try:
            # 1. Converter PDF → chunks (Docling)
            records = convert_pdf_block(
                pdf_path=pdf_path,
                chunker=chunker,
                file_hash=file_hash,
                page_start=page_start,
                page_end=page_end,
            )

            # 2. Gerar embeddings e inserir no Chroma
            inserted = insert_chunks(
                records=records,
                collection=collection,
                ollama_ef=ollama_ef,
                label=f"{pdf_path.name} p.{page_start}-{page_end}",
            )

            inserted_total += inserted

        except Exception as exc:
            logger.exception(
                "Erro ao processar %s páginas %s-%s: %s",
                pdf_path.name,
                page_start,
                page_end,
                exc,
            )

        # Força GC entre blocos independentemente de sucesso/falha
        force_gc()

    logger.info(
        "Finalizado %s | chunks inseridos=%s",
        pdf_path.name,
        inserted_total,
    )

    return inserted_total


def index_folder(path: Path | str) -> None:
    """
    Indexa todos os PDFs de uma pasta.
    """
    pdf_files = list_files(path)

    if not pdf_files:
        logger.warning("Nenhum PDF encontrado em: %s", path)
        return

    collection, ollama_ef = create_collection()

    # HybridChunker preserva melhor a estrutura do documento do que um split
    # puramente por caracteres. Depois usamos contextualize(chunk) para embedar
    # texto com contexto estrutural.
    chunker = HybridChunker()

    total_inserted = 0

    for pdf_path in tqdm(pdf_files, desc="Arquivos PDF"):
        inserted = index_pdf(
            pdf_path=pdf_path,
            chunker=chunker,
            collection=collection,
            ollama_ef=ollama_ef,
        )

        total_inserted += inserted

    logger.info("Indexação concluída. Total de chunks inseridos: %s", total_inserted)
