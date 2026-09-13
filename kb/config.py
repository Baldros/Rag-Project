"""
Configuração central do knowledge base.

Todos os parâmetros ajustáveis vivem aqui para que o comportamento possa ser
alterado sem tocar na lógica do pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORE_DIR = Path(os.getenv("KB_STORE_DIR", PROJECT_ROOT / "store"))
DB_PATH = STORE_DIR / "kb.sqlite3"
ARTIFACTS_DIR = STORE_DIR / "artifacts"
CHROMA_DIR = STORE_DIR / "chroma"


def _parse_roots(raw: str | None) -> tuple[Path, ...]:
    """Interpreta uma lista de diretórios separados por os.pathsep."""
    if not raw:
        return ()
    return tuple(
        Path(part).expanduser().resolve()
        for part in raw.split(os.pathsep)
        if part.strip()
    )


# Diretórios de onde a ingestão pode ler. A tool `ingest` recusa qualquer caminho
# fora daqui: sem essa guarda, um agente induzido pelo conteúdo de um documento
# poderia indexar arquivos arbitrários do disco.
ALLOWED_INGEST_ROOTS: tuple[Path, ...] = _parse_roots(
    os.getenv("KB_INGEST_ROOTS")
) or (
    Path("E:/Estudo").resolve(),
    PROJECT_ROOT,
)


# =========================
# Modelos
# =========================

# Ambos rodam via sentence-transformers/torch. Runtime único dá controle
# explícito de carga e descarga da VRAM, que é o requisito central deste projeto.
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192

RERANK_MODEL = os.getenv("KB_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_ENABLED = os.getenv("KB_RERANK", "1") != "0"
RERANK_MAX_LENGTH = 1024

DEVICE = os.getenv("KB_DEVICE", "cuda")
TORCH_DTYPE = os.getenv("KB_DTYPE", "float16")

# O ciclo de vida dos modelos é o do servidor: sobem no startup e caem no
# shutdown. Descarregar por ociosidade sairia caro — nesta máquina o custo de
# subir de novo é dominado pelos ~150s de `import torch` + `sentence_transformers`,
# não pela leitura dos pesos — e não compraria nada: a pilha de retrieval ocupa
# 2,2 GB e não disputa VRAM com mais nada. 0 desliga.
MODEL_IDLE_TIMEOUT = int(os.getenv("KB_MODEL_IDLE_TIMEOUT", "0"))


# =========================
# LLM de ingestao (Pass 3)
# =========================

# Usado somente para processamento de texto na ingestao: resumir documento e
# secao. Nao precisa de tool calling nem de contexto longo. Roda sozinho na GPU,
# depois que o Docling e o embedder ja sairam.
#
# O modelo em si ainda nao foi escolhido a serio — merece uma sessao de teste
# comparativa. qwen3:8b e o ponto de partida; trocar e uma linha.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("KB_LLM_MODEL", "qwen3:8b")
LLM_ENABLED = os.getenv("KB_LLM", "1") != "0"
LLM_TEMPERATURE = float(os.getenv("KB_LLM_TEMPERATURE", "0.2"))
LLM_NUM_CTX = int(os.getenv("KB_LLM_NUM_CTX", "8192"))
LLM_REQUEST_TIMEOUT = float(os.getenv("KB_LLM_TIMEOUT", "180"))

# Resumo por secao multiplica o custo por centenas. O de documento e barato e
# alimenta o catalogo, entao vem ligado; o de secao fica opcional.
ENRICH_SECTIONS = os.getenv("KB_ENRICH_SECTIONS", "0") != "0"
ENRICH_MIN_SECTION_CHARS = int(os.getenv("KB_ENRICH_MIN_SECTION_CHARS", "1500"))


# =========================
# Vector index
# =========================

# Nome interno e fixo da collection do Chroma. Não confundir com as collections
# do usuário, que são um conceito do SQLite (tabela `collections`).
CHROMA_COLLECTION = "kb_chunks"


# =========================
# Ingestão
# =========================

# O Docling converte o PDF em blocos de páginas em vez do arquivo inteiro, para
# limitar o pico de memória em livros grandes.
PAGE_BLOCK_SIZE = int(os.getenv("KB_PAGE_BLOCK_SIZE", "15"))

CHUNK_MAX_TOKENS = int(os.getenv("KB_CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("KB_CHUNK_OVERLAP_TOKENS", "64"))

EMBED_BATCH_SIZE = int(os.getenv("KB_EMBED_BATCH_SIZE", "16"))

SUPPORTED_EXTENSIONS = {".pdf"}

# Invalida artefatos de parse quando a forma de converter muda. Bump manual ao
# alterar o pipeline do Docling.
PARSE_SCHEMA_VERSION = "1"


def parse_version() -> str:
    """Assinatura do parse; artefato com assinatura diferente é reprocessado."""
    try:
        from importlib.metadata import version as _pkg_version

        docling_version = _pkg_version("docling")
    except Exception:  # pragma: no cover - docling sempre presente em runtime
        docling_version = "unknown"

    return f"docling{docling_version}-blk{PAGE_BLOCK_SIZE}-v{PARSE_SCHEMA_VERSION}"


# =========================
# Retrieval
# =========================

RECALL_TOP_K = int(os.getenv("KB_RECALL_TOP_K", "40"))
FUSION_TOP_K = int(os.getenv("KB_FUSION_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("KB_FINAL_TOP_K", "5"))

# Constante do Reciprocal Rank Fusion. 60 é o valor do paper original.
RRF_K = 60

# Teto de caracteres por seção devolvida na expansão small-to-big.
MAX_SECTION_CHARS = int(os.getenv("KB_MAX_SECTION_CHARS", "12000"))


# =========================
# Logging
# =========================

logging.basicConfig(
    level=os.getenv("KB_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
