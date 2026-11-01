"""Quality metadata and conservative normalization of lexical search text."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

NORMALIZATION_VERSION = "1"
UNKNOWN = {"status": "unknown", "warnings": ["Qualidade da extração ainda não verificada."]}

# Protect explicit mathematics and code. Never apply this to the source/citation.
_PROTECTED = re.compile(r"(```[\s\S]*?```|`[^`]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))")
_MARKS = {"´": "\u0301", "`": "\u0300", "ˆ": "\u0302", "˜": "\u0303", "¨": "\u0308"}


def normalize_lexical(text: str | None) -> str:
    """Repair detached TeX accents inside Latin words, without rewriting sources."""
    def prose(value: str) -> str:
        value = re.sub(r"¸\s*([Cc])", lambda m: unicodedata.normalize("NFC", m[1] + "\u0327"), value)
        value = re.sub(r"([Cc])\s*¸\s*", lambda m: unicodedata.normalize("NFC", m[1] + "\u0327"), value)
        def accent(match: re.Match) -> str:
            letter = match[3].replace("ı", "i").replace("ȷ", "j")
            return match[1] + unicodedata.normalize("NFC", letter + _MARKS[match[2]])
        value = re.sub(r"([A-Za-zÀ-ÿ])([´`ˆ˜¨])\s*([AEIOUYaeiouyıȷ])(?=[A-Za-zÀ-ÿ])", accent, value)
        # Standalone accents at word starts, followed by a multi-letter word.
        value = re.sub(r"(?<!\w)([´`ˆ˜¨])\s*([AEIOUYaeiouyı])(?=[A-Za-zÀ-ÿ]{2})",
                       lambda m: unicodedata.normalize("NFC", m[2].replace("ı", "i") + _MARKS[m[1]]), value)
        return unicodedata.normalize("NFC", value)
    return "".join(part if i % 2 else prose(part) for i, part in enumerate(_PROTECTED.split(text or "")))


def decode_quality(raw: str | dict | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if raw else None
        return value if isinstance(value, dict) else dict(UNKNOWN)
    except (ValueError, TypeError):
        return dict(UNKNOWN)


def for_pages(raw: str | dict | None, start: int | None, end: int | None) -> dict:
    quality = decode_quality(raw)
    if quality.get("status") == "unknown":
        return quality
    issues = [i for i in quality.get("issues", [])
              if start is None or any(start <= p <= (end or start) for p in i.get("pages", []))]
    errors = quality.get("errors", [])
    partial = bool(issues or errors)
    warnings = ["Há fórmulas sem extração confiável neste trecho ou falhas sem localização precisa no documento."] if partial else []
    verification = quality.get("formula_verification", "unknown")
    if verification == "automated_unverified":
        warnings.append("Fórmulas transcritas por modelo local; confira no PDF antes de usá-las em cálculos.")
    return {"status": "partial" if partial else "complete", "issues": issues,
            "formula_verification": verification, "warnings": warnings}


def document_metadata(row: dict) -> dict:
    row = dict(row)
    row["quality"] = decode_quality(row.pop("quality_json", None))
    return row
