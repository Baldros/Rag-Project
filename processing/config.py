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
CHROMA_DIR = Path("./chroma_db")


# =========================
# ChromaDB / Embeddings
# =========================

COLLECTION_NAME = "livros_fisica"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "qwen3.5:0.8b"

RAG_TOP_K = 6
RAG_MAX_CONTEXT_CHARS = 4500
LLM_REQUEST_TIMEOUT = 120


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
