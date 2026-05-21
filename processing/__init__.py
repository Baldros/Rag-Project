"""
Pacote de processamento de PDFs e geração de embeddings.

Uso típico:
    from processing import index_folder
    from processing.config import PDF_DIR

    index_folder(PDF_DIR)
"""

from processing.indexer import index_folder, index_pdf

__all__ = ["index_folder", "index_pdf"]
