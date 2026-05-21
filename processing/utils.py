"""
Utilitários puros — sem dependências de domínio.

Funções auxiliares usadas por vários módulos do pacote.
Nenhuma função aqui conhece Docling, ChromaDB ou Ollama.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from processing.config import SUPPORTED_EXTENSIONS


def list_files(path: Path | str) -> list[Path]:
    """
    Lista arquivos suportados diretamente dentro de uma pasta.
    Não é recursivo.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"O caminho não é uma pasta: {path}")

    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def batched(items: list, batch_size: int) -> Iterable[list]:
    """
    Divide uma lista em batches de tamanho fixo.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """
    Gera hash SHA-256 do arquivo para criar IDs estáveis
    e detectar duplicatas.
    """
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(block_size):
            digest.update(chunk)

    return digest.hexdigest()


def stable_chunk_id(
    *,
    file_hash: str,
    page_start: int,
    page_end: int,
    chunk_index: int,
) -> str:
    """
    ID determinístico para evitar duplicação quando o script roda de novo.
    """
    raw = f"{file_hash}:{page_start}:{page_end}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_pdf_page_count(path: Path) -> int:
    """
    Conta páginas do PDF sem carregar o documento inteiro no Docling.
    """
    reader = PdfReader(str(path))
    return len(reader.pages)


def normalize_metadata(metadata: dict) -> dict:
    """
    Normaliza metadados para o formato aceito pelo Chroma.

    Valores simples (str, int, float, bool) passam direto.
    Estruturas complexas são serializadas como JSON.
    """
    normalized = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)

    return normalized


def force_gc():
    """Força coleta de lixo para liberar memória entre blocos."""
    gc.collect()
