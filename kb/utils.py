"""Utilitários puros, sem dependência de domínio."""

from __future__ import annotations

import gc
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

from pypdf import PdfReader

from kb.config import SUPPORTED_EXTENSIONS

T = TypeVar("T")


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash do arquivo: identidade do documento e chave de deduplicação."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)

    return digest.hexdigest()


def stable_id(*parts: object) -> str:
    """ID determinístico — reprocessar não duplica linha."""
    raw = ":".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def get_pdf_page_count(path: Path) -> int:
    """Conta páginas sem passar o arquivo pelo Docling."""
    return len(PdfReader(str(path)).pages)


def list_documents(path: Path | str, recursive: bool = True) -> list[Path]:
    """Lista arquivos suportados sob um caminho (arquivo ou pasta)."""
    path = Path(path)

    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if not path.is_dir():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        item
        for item in path.glob(pattern)
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def force_gc() -> None:
    """Coleta entre blocos pesados do Docling."""
    gc.collect()


def ascii_fold(text: str) -> str:
    """Remove acentos, para regras textuais simples."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def slugify(text: str, max_length: int = 64) -> str:
    """Converte um nome legível em identificador de collection."""
    folded = ascii_fold(text).lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return (slug or "collection")[:max_length]


def truncate(text: str, max_chars: int) -> str:
    """Corta em fronteira de palavra e sinaliza o corte."""
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + " […]"


def iter_unique(items: Iterable[T]) -> Iterator[T]:
    seen: set[T] = set()

    for item in items:
        if item not in seen:
            seen.add(item)
            yield item
