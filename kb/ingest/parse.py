"""
Pass 1 — parse.

Converte o PDF em um artefato canônico e imutável no disco. É o passo mais caro
do sistema inteiro, então roda uma vez só: qualquer mudança futura de chunking,
de modelo de embedding ou de schema relê o artefato em vez de reprocessar o PDF.

Dois detalhes que vieram da auditoria do pipeline antigo:

1. O `DocumentConverter` é criado **uma vez** e reusado entre blocos. A versão
   anterior instanciava um por bloco de 15 páginas, o que reinicializava o
   pipeline de modelos ONNX 67 vezes num livro de 1000 páginas.
2. Cada bloco é gravado assim que converte. Matar o processo no meio e rodar de
   novo retoma de onde parou, em vez de recomeçar o livro.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from kb.config import ARTIFACTS_DIR, PAGE_BLOCK_SIZE, parse_version
from kb.utils import file_sha256, force_gc, get_pdf_page_count

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]

_converter: Any = None


def get_converter():
    """
    Singleton do DocumentConverter.

    Construir um custa o carregamento dos modelos de layout e de tabela; reusar
    entre blocos e entre arquivos é o que torna a ingestão de uma pasta viável.
    """
    global _converter

    if _converter is None:
        from docling.document_converter import DocumentConverter

        logger.info("Inicializando DocumentConverter (carrega modelos de layout)...")
        _converter = DocumentConverter()

    return _converter


def release_converter() -> None:
    """Libera o converter e seus modelos ao fim do pass."""
    global _converter

    if _converter is not None:
        del _converter
        _converter = None

    force_gc()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def artifact_dir(doc_id: str) -> Path:
    return ARTIFACTS_DIR / doc_id


def manifest_path(doc_id: str) -> Path:
    return artifact_dir(doc_id) / "manifest.json"


def read_manifest(doc_id: str) -> dict[str, Any] | None:
    path = manifest_path(doc_id)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Manifest ilegível para %s; será reprocessado.", doc_id)
        return None


def _write_manifest(doc_id: str, manifest: dict[str, Any]) -> None:
    path = manifest_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Escrita atômica: um manifest truncado por interrupção invalidaria o
    # artefato inteiro e forçaria reprocessar o livro.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_parsed(doc_id: str, version: str | None = None) -> bool:
    """True se existe artefato completo e com a assinatura de parse esperada."""
    manifest = read_manifest(doc_id)

    if not manifest:
        return False

    if manifest.get("parse_version") != (version or parse_version()):
        return False

    return manifest.get("complete") is True


def _block_ranges(n_pages: int, block_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + block_size - 1, n_pages))
        for start in range(1, n_pages + 1, block_size)
    ]


def parse_document(
    pdf_path: Path | str,
    doc_id: str | None = None,
    on_progress: ProgressFn | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Converte um PDF em artefato. Devolve o manifest.

    Idempotente: se o artefato já existe com a mesma `parse_version`, retorna
    imediatamente sem tocar no Docling.
    """
    pdf_path = Path(pdf_path).resolve()
    doc_id = doc_id or file_sha256(pdf_path)
    version = parse_version()

    if not force and is_parsed(doc_id, version):
        logger.info("skip parse (já indexado): %s", pdf_path.name)
        manifest = read_manifest(doc_id)
        assert manifest is not None
        manifest["skipped"] = True
        return manifest

    n_pages = get_pdf_page_count(pdf_path)
    ranges = _block_ranges(n_pages, PAGE_BLOCK_SIZE)

    blocks_dir = artifact_dir(doc_id) / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    previous = read_manifest(doc_id) or {}
    reusable = (
        previous.get("parse_version") == version and not force
    )
    done_blocks: dict[str, dict[str, Any]] = (
        {block["file"]: block for block in previous.get("blocks", [])}
        if reusable
        else {}
    )

    manifest: dict[str, Any] = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "source_path": str(pdf_path),
        "parse_version": version,
        "n_pages": n_pages,
        "block_size": PAGE_BLOCK_SIZE,
        "blocks": [],
        "complete": False,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    converter = get_converter()
    total = len(ranges)

    for index, (page_start, page_end) in enumerate(ranges, start=1):
        name = f"{page_start:05d}-{page_end:05d}.json"
        target = blocks_dir / name

        if name in done_blocks and target.exists():
            manifest["blocks"].append(done_blocks[name])
            if on_progress:
                on_progress(index, total, f"{pdf_path.name} p.{page_start}-{page_end} (cache)")
            continue

        entry: dict[str, Any] = {
            "file": name,
            "page_start": page_start,
            "page_end": page_end,
            "ok": False,
        }

        try:
            result = converter.convert(
                pdf_path,
                page_range=(page_start, page_end),
                raises_on_error=False,
            )

            if result.document is None:
                entry["error"] = "documento vazio"
                logger.warning(
                    "Sem documento em %s p.%s-%s", pdf_path.name, page_start, page_end
                )
            else:
                result.document.save_as_json(target)
                entry["ok"] = True

            del result

        except Exception as exc:
            entry["error"] = str(exc)
            logger.exception(
                "Falha convertendo %s p.%s-%s", pdf_path.name, page_start, page_end
            )

        manifest["blocks"].append(entry)

        # Grava o manifest a cada bloco: é o que torna o pass retomável.
        _write_manifest(doc_id, manifest)
        force_gc()

        if on_progress:
            on_progress(index, total, f"{pdf_path.name} p.{page_start}-{page_end}")

    ok_blocks = [block for block in manifest["blocks"] if block.get("ok")]
    manifest["complete"] = bool(ok_blocks)
    manifest["n_blocks_ok"] = len(ok_blocks)
    manifest["n_blocks_failed"] = len(manifest["blocks"]) - len(ok_blocks)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _write_manifest(doc_id, manifest)

    logger.info(
        "parse concluído: %s | páginas=%s | blocos ok=%s falhos=%s",
        pdf_path.name,
        n_pages,
        manifest["n_blocks_ok"],
        manifest["n_blocks_failed"],
    )

    return manifest


def iter_blocks(doc_id: str) -> Iterator[tuple[int, int, Any]]:
    """
    Percorre os blocos do artefato em ordem de página.

    Devolve (page_start, page_end, DoclingDocument). Blocos que falharam no
    parse são pulados silenciosamente — já foram registrados no manifest.
    """
    from docling_core.types.doc import DoclingDocument

    manifest = read_manifest(doc_id)
    if not manifest:
        return

    blocks_dir = artifact_dir(doc_id) / "blocks"

    for block in sorted(manifest["blocks"], key=lambda b: b["page_start"]):
        if not block.get("ok"):
            continue

        path = blocks_dir / block["file"]
        if not path.exists():
            logger.warning("Bloco ausente: %s", path)
            continue

        yield block["page_start"], block["page_end"], DoclingDocument.load_from_json(path)


def export_markdown(doc_id: str) -> Path:
    """Concatena os blocos num markdown legível, para inspeção humana."""
    parts: list[str] = []

    for _, _, document in iter_blocks(doc_id):
        parts.append(document.export_to_markdown())

    target = artifact_dir(doc_id) / "document.md"
    target.write_text("\n\n".join(parts), encoding="utf-8")

    return target
