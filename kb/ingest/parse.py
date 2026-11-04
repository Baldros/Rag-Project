"""Resumable Docling conversion with immutable, validated generations."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator

from kb import config
from kb.config import ARTIFACTS_DIR, PAGE_BLOCK_SIZE, formula_version, parse_version
from kb.utils import file_sha256, force_gc, get_pdf_page_count

logger = logging.getLogger(__name__)
ProgressFn = Callable[[int, int, str], None]
_converter: Any = None
_converter_ocr: str | None = None
_formula_model: Any = None
# A corrected formula must still sit where the person saw it after a new conversion.
MIN_CORRECTION_OVERLAP = 0.5


class ParseError(RuntimeError):
    def __init__(self, message: str, manifest: dict):
        super().__init__(message)
        self.manifest = manifest


def get_converter(ocr: str | None = None):
    global _converter, _converter_ocr
    ocr = ocr or config.OCR_MODE
    if ocr not in {"auto", "off", "full"}:
        raise ValueError("OCR deve ser auto, off ou full")
    if _converter is not None and _converter_ocr != ocr:
        release_converter()
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions, CodeFormulaVlmOptions, OcrMode, PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(
            do_ocr=ocr != "off", do_table_structure=True,
            do_formula_enrichment=True, do_code_enrichment=False,
            accelerator_options=AcceleratorOptions(device=config.DEVICE),
            code_formula_options=CodeFormulaVlmOptions.from_preset(
                config.FORMULA_PRESET, expansion_factor=config.FORMULA_MARGIN,
                stop_on_repetition=True,
            ),
        )
        if ocr == "full":
            options.ocr_options.mode = OcrMode.FULL_PAGE
        _converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
        )
        _converter_ocr = ocr
    return _converter


def release_converter() -> None:
    global _converter, _converter_ocr, _formula_model
    _converter = _converter_ocr = None
    _formula_model = None
    force_gc()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def artifact_dir(doc_id: str) -> Path:
    return ARTIFACTS_DIR / doc_id


def manifest_path(doc_id: str) -> Path:
    return artifact_dir(doc_id) / "manifest.json"


def _path(doc_id: str, relative: str) -> Path:
    root = artifact_dir(doc_id).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Caminho de artefato fora do documento")
    return path


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def read_manifest(doc_id: str, relative: str | None = None) -> dict | None:
    return _read_json(_path(doc_id, relative or "manifest.json"))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_manifest(doc_id: str, manifest: dict) -> None:
    _atomic_json(_path(doc_id, manifest["manifest_file"]), manifest)
    _atomic_json(_path(doc_id, "candidate.json"), manifest)


def activate_manifest(doc_id: str, manifest: dict) -> None:
    # SQLite stores the authoritative generation. This pointer is for inspection.
    _atomic_json(manifest_path(doc_id), manifest)
    # A retry reuses the candidate's verdict on each block, so it must follow what was published.
    _atomic_json(_path(doc_id, "candidate.json"), manifest)


def _block_ranges(n_pages: int, block_size: int) -> list[tuple[int, int]]:
    if block_size < 1:
        raise ValueError("Tamanho de bloco deve ser positivo")
    return [(p, min(p + block_size - 1, n_pages)) for p in range(1, n_pages + 1, block_size)]


def _block_valid(doc_id: str, block: dict) -> bool:
    try:
        path = _path(doc_id, block["file"])
        if not block.get("ok") or file_sha256(path) != block.get("sha256"):
            return False
        data = _read_json(path)
        return bool(data and data.get("schema_name") == "DoclingDocument" and
                    set(map(int, data.get("pages", {}))) ==
                    set(range(block["page_start"], block["page_end"] + 1)))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_manifest(doc_id: str, manifest: dict | None) -> bool:
    try:
        if not manifest or not manifest.get("complete") or manifest["n_pages"] < 1:
            return False
        blocks = manifest["blocks"]
        return ([(b["page_start"], b["page_end"]) for b in blocks] ==
                _block_ranges(manifest["n_pages"], manifest["block_size"]) and
                all(_block_valid(doc_id, b) for b in blocks))
    except (KeyError, TypeError, ValueError):
        return False


def is_parsed(doc_id: str, version: str | None = None, manifest: dict | None = None) -> bool:
    manifest = manifest if manifest is not None else read_manifest(doc_id)
    return bool(manifest and manifest.get("parse_version") == (version or parse_version())
                and manifest.get("formula_version") == formula_version()
                and validate_manifest(doc_id, manifest))


def _formulas(document):
    return [item for item, _ in document.iterate_items() if item.label.value == "formula"]


def _missing(item) -> bool:
    return not item.text.strip() or "formula-not-decoded" in item.text


def _formula_problem(item) -> str | None:
    if _missing(item):
        return "Fórmula sem LaTeX após uma nova tentativa."
    depth = 0
    for brace in re.findall(r"(?<!\\)[{}]", item.text):
        depth += 1 if brace == "{" else -1
        if depth < 0:
            break
    if depth != 0:
        return "LaTeX com chaves desequilibradas; confira a fórmula no PDF."
    return None


@contextmanager
def _formula_errors():
    errors: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                errors.append(record.getMessage())

    log = logging.getLogger("docling.models.stages.code_formula.code_formula_vlm_model")
    handler = Capture()
    log.addHandler(handler)
    try:
        yield errors
    finally:
        log.removeHandler(handler)


def _enrich_existing(document, source: bytes, converter, refs: set[str] | None = None) -> list[str]:
    """Use Docling's own formula model on cached layout and PDF crops only."""
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
    from docling.datamodel.document import InputDocument
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel

    global _formula_model
    model = next((m for p in converter.initialized_pipelines.values()
                  for m in p.enrichment_pipe if isinstance(m, CodeFormulaVlmModel)), None)
    if model is None:
        # A formula-only retry must not load the layout/table/OCR models.
        if _formula_model is None:
            from docling.datamodel.pipeline_options import AcceleratorOptions, CodeFormulaVlmOptions
            _formula_model = CodeFormulaVlmModel(
                enabled=True, enable_remote_services=False, artifacts_path=None,
                accelerator_options=AcceleratorOptions(device=config.DEVICE),
                options=CodeFormulaVlmOptions.from_preset(
                    config.FORMULA_PRESET, expansion_factor=config.FORMULA_MARGIN,
                    stop_on_repetition=True, extract_code=False, extract_formulas=True,
                ),
            )
        model = _formula_model
    inp = InputDocument(path_or_stream=BytesIO(source), format=InputFormat.PDF,
                        backend=PyPdfiumDocumentBackend, filename="source.pdf")
    try:
        with _formula_errors() as errors:
            batch = []
            for item in _formulas(document):
                if refs is not None and item.self_ref not in refs:
                    continue
                prov = item.prov[0]
                page = inp._backend.load_page(prov.page_no - 1)
                try:
                    bbox = prov.bbox.expand_by_scale(model.expansion_factor, model.expansion_factor)
                    crop = page.get_page_image(scale=model.images_scale, cropbox=bbox)
                finally:
                    page.unload()
                batch.append(ItemAndImageEnrichmentElement(item=item, image=crop))
                if len(batch) >= model.elements_batch_size:
                    list(model(doc=document, element_batch=batch))
                    batch = []
            if batch:
                list(model(doc=document, element_batch=batch))
        return errors
    finally:
        inp.release()


def _location(item) -> tuple[int, list[float]]:
    prov = item.prov[0]
    return prov.page_no, [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]


def _overlap(a: list[float], b: list[float]) -> float:
    """Intersection over union of two boxes sharing a coordinate origin."""
    width = min(max(a[0], a[2]), max(b[0], b[2])) - max(min(a[0], a[2]), min(b[0], b[2]))
    height = min(max(a[1], a[3]), max(b[1], b[3])) - max(min(a[1], a[3]), min(b[1], b[3]))
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / (abs(a[2] - a[0]) * abs(a[3] - a[1]) + abs(b[2] - b[0]) * abs(b[3] - b[1]) - shared)


def _block_corrections(corrections: list[dict], start: int, end: int) -> list[dict]:
    return [c for c in corrections if start <= c["page"] <= end]


def _overlay_corrections(document, corrections: list[dict]) -> tuple[set[str], list[dict]]:
    """Put manual LaTeX on a loaded block. The block file keeps what the model wrote.

    A correction whose formula is no longer in place is returned instead of being
    written over a different formula.
    """
    if not corrections:
        return set(), []
    formulas = {item.self_ref: item for item in _formulas(document)}
    corrected: set[str] = set()
    orphaned: list[dict] = []
    for correction in corrections:
        item = formulas.get(correction["ref"])
        if item is not None:
            page, box = _location(item)
            if page == correction["page"] and _overlap(box, correction["bbox"]) >= MIN_CORRECTION_OVERLAP:
                item.text = correction["text"]
                corrected.add(item.self_ref)
                continue
        orphaned.append(correction)
    return corrected, orphaned


def _quality(document, errors: list[str], corrected: set[str] | None = None,
             orphaned: list[dict] | None = None) -> dict:
    formulas = _formulas(document)
    corrected = corrected or set()
    # A person checked a corrected formula, so the automatic rule no longer judges it.
    issues = [{"code": "FORMULA_EXTRACTION", "ref": item.self_ref,
               "pages": sorted({p.page_no for p in item.prov}),
               "message": _formula_problem(item)}
              for item in formulas if item.self_ref not in corrected and _formula_problem(item)]
    issues += [{"code": "FORMULA_CORRECTION_ORPHANED", "pages": [c["page"]],
                "message": f"Correção manual não reaplicada: a fórmula mudou em uma nova conversão. LaTeX guardado: {c['text']}"}
               for c in orphaned or []]
    return {"status": "partial" if issues or errors else "complete",
            "formula_count": len(formulas), "formula_missing": sum(_missing(i) for i in formulas),
            "formula_flagged": len(issues), "formula_corrected": len(corrected),
            "formula_verification": "automated_unverified" if formulas else "not_applicable",
            "issues": issues, "errors": errors,
            "warnings": (["Há fórmulas ausentes ou cuja extração falhou."] if issues or errors else []) +
                        (["Fórmulas transcritas por modelo local; a correção matemática não foi verificada."] if formulas else [])}


def _aggregate_quality(manifest: dict) -> dict:
    qualities = [b.get("quality", {}) for b in manifest["blocks"]]
    quality = {
        "status": "complete" if manifest["complete"] and all(q.get("status") == "complete" for q in qualities) else "partial",
        "formula_count": sum(q.get("formula_count", 0) for q in qualities),
        "formula_missing": sum(q.get("formula_missing", 0) for q in qualities),
        "formula_flagged": sum(q.get("formula_flagged", 0) for q in qualities),
        "formula_corrected": sum(q.get("formula_corrected", 0) for q in qualities),
        "formula_verification": "automated_unverified" if any(q.get("formula_count", 0) for q in qualities) else "not_applicable",
        "issues": [{**i, "block": b["file"]} for b in manifest["blocks"] for i in b.get("quality", {}).get("issues", [])],
        "errors": [e for q in qualities for e in q.get("errors", [])],
        "warnings": [],
    }
    if quality["status"] != "complete":
        quality["warnings"] = ["Há falhas de conversão ou fórmulas ausentes; consulte os detalhes de qualidade."]
    if quality["formula_count"]:
        quality["warnings"].append("Fórmulas transcritas por modelo local; a correção matemática não foi verificada.")
    return quality


def correct_formulas(doc_id: str, manifest: dict, changes: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Return a manifest with the manual LaTeX, plus the applied and the rejected changes.

    Each change has `ref`, `page`, `text` (None restores the model's output) and
    `expected`, the LaTeX the person was looking at. The new generation reuses the
    block files, so their hashes stay valid, and takes effect only when the caller
    publishes it together with its chunks. With nothing to change, `manifest` itself
    comes back: applying the same text twice is not a conflict.
    """
    from docling_core.types.doc import DoclingDocument

    if not validate_manifest(doc_id, manifest):
        raise RuntimeError("Artefato incompleto, ausente ou corrompido; execute novamente a ingestão")
    corrections = {(c["page"], c["ref"]): c for c in manifest.get("corrections") or []}
    applied: list[dict] = []
    rejected: list[dict] = []
    touched: set[int] = set()
    for change in changes:
        page, ref, text = change["page"], change["ref"], change.get("text")
        prior = corrections.get((page, ref))
        index = next((i for i, b in enumerate(manifest["blocks"]) if b["page_start"] <= page <= b["page_end"]), None)
        if index is None:
            rejected.append({**change, "reason": f"A página {page} não existe neste documento."})
            continue
        if text is None:
            if prior is None:
                rejected.append({**change, "reason": "Esta fórmula não tem correção para restaurar."})
                continue
            del corrections[(page, ref)]
            touched.add(index)
            applied.append(change)
            continue
        if not text.strip():
            rejected.append({**change, "reason": "O LaTeX está vazio."})
            continue
        document = DoclingDocument.load_from_json(_path(doc_id, manifest["blocks"][index]["file"]))
        item = next((i for i in _formulas(document) if i.self_ref == ref), None)
        if item is None or _location(item)[0] != page:
            rejected.append({**change, "reason": "Esta fórmula não existe mais neste documento."})
            continue
        current = prior["text"] if prior else item.text
        if text == current:
            applied.append(change)
            continue
        if change.get("expected") != current:
            rejected.append({**change, "reason": "A fórmula mudou depois da edição; abra de novo e revise."})
            continue
        corrections[(page, ref)] = {"page": page, "ref": ref, "bbox": _location(item)[1], "original": item.text,
                                    "text": text, "corrected_at": datetime.now(timezone.utc).isoformat()}
        touched.add(index)
        applied.append(change)
    if not touched:
        return manifest, applied, rejected

    current_corrections = list(corrections.values())
    blocks = [dict(block) for block in manifest["blocks"]]
    for index in touched:
        block = blocks[index]
        document = DoclingDocument.load_from_json(_path(doc_id, block["file"]))
        corrected, orphaned = _overlay_corrections(
            document, _block_corrections(current_corrections, block["page_start"], block["page_end"]))
        block["quality"] = _quality(document, block.get("quality", {}).get("errors", []), corrected, orphaned)
    updated = {**manifest, "manifest_file": f"generations/{uuid.uuid4().hex}/manifest.json", "blocks": blocks,
               "corrections": current_corrections}
    updated["quality"] = _aggregate_quality(updated)
    _atomic_json(_path(doc_id, updated["manifest_file"]), updated)
    return updated, applied, rejected


def parse_document(pdf_path: Path | str, doc_id: str | None = None,
                   on_progress: ProgressFn | None = None, force: bool = False,
                   ocr: str | None = None, previous: dict | None = None) -> dict:
    global _formula_model
    pdf_path = Path(pdf_path).resolve()
    doc_id = doc_id or file_sha256(pdf_path)
    ocr = ocr or config.OCR_MODE
    version, formulas_version = parse_version(ocr), formula_version()
    previous = previous if previous is not None else read_manifest(doc_id)
    if (not force and is_parsed(doc_id, version, previous) and
            previous.get("quality", {}).get("status") == "complete"):
        return previous

    candidate = read_manifest(doc_id, "candidate.json")
    reusable = candidate if candidate and candidate.get("parse_version") == version else previous
    old_blocks = ({(b["page_start"], b["page_end"]): b for b in reusable.get("blocks", [])}
                  if not force and reusable and reusable.get("parse_version") == version else {})
    n_pages = get_pdf_page_count(pdf_path)
    ranges = _block_ranges(n_pages, PAGE_BLOCK_SIZE)
    generation = f"generations/{uuid.uuid4().hex}"
    manifest = {"doc_id": doc_id, "filename": pdf_path.name, "source_path": str(pdf_path),
                "parse_version": version, "formula_version": formulas_version, "ocr": ocr,
                "manifest_file": f"{generation}/manifest.json", "n_pages": n_pages,
                "block_size": PAGE_BLOCK_SIZE, "blocks": [], "complete": False,
                "started_at": datetime.now(timezone.utc).isoformat()}
    # Corrections come from the published manifest and outlive any new conversion, even with `force`.
    corrections = (previous or {}).get("corrections") or []
    if corrections:
        manifest["corrections"] = corrections
    source = pdf_path.read_bytes()  # Unicode paths never reach a native PDF backend.
    started = time.monotonic()
    from docling.datamodel.base_models import DocumentStream, ConversionStatus
    from docling_core.types.doc import DoclingDocument

    for index, (start, end) in enumerate(ranges, 1):
        if on_progress:
            on_progress(index - 1, len(ranges), f"{pdf_path.name} p.{start}-{end} (convertendo)")
        old = old_blocks.get((start, end), {})
        valid_old = _block_valid(doc_id, old)
        if (valid_old and old.get("formula_version") == formulas_version and
                old.get("quality", {}).get("status") == "complete"):
            manifest["blocks"].append(old)
            _write_manifest(doc_id, manifest)
            if on_progress:
                on_progress(index, len(ranges), f"{pdf_path.name} p.{start}-{end} (cache)")
            continue
        entry = {"file": f"{generation}/blocks/{start:05d}-{end:05d}.json",
                 "page_start": start, "page_end": end, "ok": False,
                 "formula_version": formulas_version}
        block_started = time.monotonic()
        try:
            converter = get_converter(ocr)
            if valid_old:
                document = DoclingDocument.load_from_json(_path(doc_id, old["file"]))
                refs = (None if old.get("formula_version") != formulas_version or
                        old.get("quality", {}).get("errors") else
                        {i["ref"] for i in old.get("quality", {}).get("issues", []) if i.get("ref")})
                try:
                    errors = _enrich_existing(document, source, converter, refs)
                except Exception as exc:
                    errors = [str(exc)]
                entry["layout_reused"] = True
            else:
                # Drop a standalone retry model before initializing a full pipeline.
                if _formula_model is not None:
                    _formula_model = None
                    force_gc()
                with _formula_errors() as errors:
                    result = converter.convert(
                        DocumentStream(name=pdf_path.name, stream=BytesIO(source)),
                        page_range=(start, end), raises_on_error=False,
                    )
                entry["conversion_status"] = result.status.value
                if result.status != ConversionStatus.SUCCESS:
                    entry["conversion_errors"] = [e.model_dump(mode="json") for e in result.errors]
                    raise RuntimeError(f"Conversão {result.status.value}: {entry['conversion_errors']}")
                document = result.document
                missing = {i.self_ref for i in _formulas(document) if _formula_problem(i)}
                if missing or errors:
                    try:
                        errors = _enrich_existing(document, source, converter, None if errors else missing)
                    except Exception as exc:
                        errors = [str(exc)]
                    entry["formula_retried"] = True
            if set(document.pages) != set(range(start, end + 1)):
                raise RuntimeError(f"Cobertura de páginas inválida: esperado {start}-{end}, obtido {sorted(document.pages)}")
            target = _path(doc_id, entry["file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            document.save_as_json(tmp)
            tmp.replace(target)
            # After saving, so the block keeps what the model wrote.
            corrected, orphaned = _overlay_corrections(document, _block_corrections(corrections, start, end))
            entry["quality"] = _quality(document, errors, corrected, orphaned)
            entry.update(ok=True, sha256=file_sha256(target))
        except Exception as exc:
            entry["error"] = str(exc)
            logger.exception("Falha convertendo %s p.%s-%s", pdf_path.name, start, end)
        entry["elapsed_seconds"] = round(time.monotonic() - block_started, 3)
        manifest["blocks"].append(entry)
        _write_manifest(doc_id, manifest)
        force_gc()
        if on_progress:
            on_progress(index, len(ranges), f"{pdf_path.name} p.{start}-{end}")

    manifest["n_blocks_ok"] = sum(bool(b.get("ok")) for b in manifest["blocks"])
    manifest["n_blocks_failed"] = len(ranges) - manifest["n_blocks_ok"]
    manifest["complete"] = bool(ranges) and manifest["n_blocks_failed"] == 0
    manifest["quality"] = _aggregate_quality(manifest)
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(doc_id, manifest)
    if not manifest["complete"]:
        raise ParseError(f"Conversão incompleta: {manifest['n_blocks_failed']} de {len(ranges)} blocos falharam", manifest)
    return manifest


def iter_blocks(doc_id: str, manifest: dict | None = None) -> Iterator[tuple[int, int, Any]]:
    from docling_core.types.doc import DoclingDocument
    manifest = manifest if manifest is not None else read_manifest(doc_id)
    if not validate_manifest(doc_id, manifest):
        raise RuntimeError("Artefato incompleto, ausente ou corrompido; execute novamente a ingestão")
    corrections = manifest.get("corrections") or []
    for block in manifest["blocks"]:
        document = DoclingDocument.load_from_json(_path(doc_id, block["file"]))
        _overlay_corrections(document, _block_corrections(corrections, block["page_start"], block["page_end"]))
        yield block["page_start"], block["page_end"], document


def export_markdown(doc_id: str, manifest: dict | None = None) -> Path:
    manifest = manifest if manifest is not None else read_manifest(doc_id)
    parts = [doc.export_to_markdown() for _, _, doc in iter_blocks(doc_id, manifest)]
    target = _path(doc_id, str(Path(manifest["manifest_file"]).parent / "document.md"))
    target.write_text("\n\n".join(parts), encoding="utf-8")
    return target
