"""
Configuração centralizada do pipeline de processamento.

Todas as constantes e parâmetros ficam aqui para facilitar
ajustes sem mexer na lógica dos outros módulos.
"""

import logging
from pathlib import Path


# =========================
# Caminhos
# =========================

PDF_DIR = Path(r"E:\Estudo\Fisica\Fisica III")
CHROMA_DIR = Path("./chroma_db")


# =========================
# ChromaDB / Embeddings
# =========================

COLLECTION_NAME = "livros_fisica"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"


# =========================
# Parâmetros de processamento
# =========================

PAGE_BLOCK_SIZE = 15
EMBED_BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = {".pdf"}


# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
