"""
Central configuration for the document processing and RAG pipeline.

Keep tunable constants here so behavior can be adjusted without touching
pipeline logic.
"""

import logging
from pathlib import Path


# =========================
# Paths
# =========================
CHROMA_DIR = Path("./chroma_db")


# =========================
# ChromaDB / Embeddings / LLM
# =========================

COLLECTION_NAME = "livros_fisica"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "qwen3.5:0.8b"

# Retrieve more candidates than the final answer strictly needs. This gives the
# model enough surrounding evidence to explain instead of only quoting a tiny
# fragment. Add a reranker later before increasing this much further.
RAG_TOP_K = 10
RAG_MAX_CONTEXT_CHARS = 9000

# Keep the generation budget aligned with the richer prompt style.
LLM_CONTEXT_WINDOW = 8192
LLM_MAX_TOKENS = 1024
LLM_REQUEST_TIMEOUT = 120


# =========================
# Processing parameters
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
