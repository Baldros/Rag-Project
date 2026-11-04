"""
Orquestração da ingestão.

Os passes rodam em sequência, sobre os modelos que a ativação da sessão deixou
carregados; quem os descarrega é o usuário, ao desligar a sessão:

    Pass 1  PARSE + SEGMENT   Docling  → artifacts/ e SQLite
    Pass 2  EMBED             bge-m3   → índice vetorial
    Pass 3  RESUMOS           Ollama   → entra e sai só neste passe

Entre eles a comunicação é por disco. É isso que torna o processo retomável: um
kill no meio do Pass 2 não perde o Pass 1, e trocar de modelo de embedding não
reprocessa nenhum PDF.
"""

from __future__ import annotations

import logging
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from kb import collection as collections_api
from kb.config import parse_version, formula_version, segment_version
from kb.quality import decode_quality
from kb.db import get_conn, transaction
from kb.ingest.embed import embed_pending
from kb.ingest.parse import (
    correct_formulas,
    export_markdown,
    is_parsed,
    parse_document,
    read_manifest,
    activate_manifest,
)
from kb.ingest.segment import (
    ChunkRecord,
    SectionRecord,
    segment_document,
    summarize,
)
from kb.utils import file_sha256, get_pdf_page_count, list_documents

logger = logging.getLogger(__name__)

# (fase, feito, total, mensagem)
ProgressFn = Callable[[str, int, int, str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def expand_paths(paths: Sequence[Path | str]) -> list[Path]:
    """Aceita arquivo, pasta ou mistura; devolve PDFs únicos e ordenados."""
    found: list[Path] = []

    for raw in paths:
        found.extend(list_documents(Path(raw)))

    unique: dict[str, Path] = {str(path.resolve()): path for path in found}
    return sorted(unique.values())


def is_indexed(doc_id: str, conn: sqlite3.Connection | None = None, ocr: str | None = None) -> bool:
    """Documento já parseado, segmentado e com a assinatura de parse atual."""
    conn = conn or get_conn()

    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()

    if not row or not row["n_chunks"]:
        return False

    if (row["parse_version"] != parse_version(ocr) or row["formula_version"] != formula_version()
            or row["segment_version"] != segment_version()
            or decode_quality(row["quality_json"]).get("status") != "complete"):
        return False

    return is_parsed(doc_id, parse_version(ocr), read_manifest(doc_id, row["artifact_manifest"]))


def register_document(
    doc_id: str,
    pdf_path: Path,
    n_pages: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    conn = conn or get_conn()

    with transaction(conn):
        conn.execute(
            """
            INSERT INTO documents(doc_id, filename, source_path, n_pages, status, created_at)
                 VALUES (?, ?, ?, ?, 'pending', ?)
            ON CONFLICT(doc_id) DO NOTHING
            """,
            (doc_id, pdf_path.name, str(pdf_path), n_pages, _now()),
        )


def persist_segments(
    doc_id: str,
    sections: list[SectionRecord],
    chunks: list[ChunkRecord],
    title: str | None = None,
    conn: sqlite3.Connection | None = None,
    manifest: dict | None = None,
    keep_summary: bool = False,
) -> None:
    """
    Grava sections e chunks numa transação só.

    Apaga o que existia antes do mesmo documento: re-segmentar é uma operação
    de substituição, não de acúmulo. Os triggers do FTS5 acompanham.

    `keep_summary` mantém o resumo do documento mesmo com o texto alterado: quem
    chama decidiu que a mudança não o invalida.
    """
    conn = conn or get_conn()

    with transaction(conn):
        old_content = [tuple(row) for row in conn.execute(
            "SELECT text, heading_path FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,))]
        changed = old_content != [(c.text, c.heading_path) for c in chunks]
        old_summaries = {(r["text"], r["heading_path"]): (r["summary"], r["enriched_at"])
                         for r in conn.execute("SELECT * FROM sections WHERE doc_id = ?", (doc_id,))}
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc_id,))

        conn.executemany(
            """
            INSERT INTO sections(section_id, doc_id, parent_id, ord, level, heading,
                                 heading_path, page_start, page_end, n_chars, text)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.section_id, s.doc_id, s.parent_id, s.ord, s.level, s.heading,
                    s.heading_path, s.page_start, s.page_end, s.n_chars, s.text,
                )
                for s in sections
            ],
        )

        conn.executemany(
            """
            INSERT INTO chunks(chunk_id, doc_id, section_id, ord, page_start, page_end,
                               n_tokens, heading_path, text, embed_text, content_type, embedded)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            [
                (
                    c.chunk_id, c.doc_id, c.section_id, c.ord, c.page_start, c.page_end,
                    c.n_tokens, c.heading_path, c.text, c.embed_text, c.content_type,
                )
                for c in chunks
            ],
        )

        conn.execute(
            """
            UPDATE documents
               SET n_sections = ?, n_chunks = ?, parse_version = ?,
                   formula_version = ?, segment_version = ?, vector_reset_pending = 1,
                   title = COALESCE(?, title), status = 'parsed', error = NULL
             WHERE doc_id = ?
            """,
            (len(sections), len(chunks), (manifest or {}).get("parse_version", parse_version()),
             formula_version(), segment_version(), title, doc_id),
        )
        for section in sections:
            prior = old_summaries.get((section.text, section.heading_path))
            if prior:
                conn.execute("UPDATE sections SET summary = ?, enriched_at = ? WHERE section_id = ?",
                             (*prior, section.section_id))
        if changed and not keep_summary:
            conn.execute("UPDATE documents SET summary = NULL, key_topics = NULL, enriched_at = NULL WHERE doc_id = ?", (doc_id,))
        if manifest:
            conn.execute("UPDATE documents SET artifact_manifest = ?, quality_json = ?, filename = ?, source_path = ?, n_pages = ? WHERE doc_id = ?",
                         (manifest["manifest_file"], json.dumps(manifest["quality"], ensure_ascii=False),
                          manifest["filename"], manifest["source_path"], manifest["n_pages"], doc_id))


def ingest_paths(
    paths: Sequence[Path | str],
    collection_id: str | None = None,
    on_progress: ProgressFn | None = None,
    force: bool = False,
    enrich: bool = True,
    conn: sqlite3.Connection | None = None,
    ocr: str | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Executa a ingestão completa e devolve o resumo do que aconteceu.

    Três passes sequenciais: Docling, embedder e LLM. Os dois primeiros ficam
    carregados até o usuário desligar a sessão; o LLM sai ao fim do seu passe.

    O processo de sessão de ingestão chama esta função após ativação manual.
    """
    conn = conn or get_conn()
    files = expand_paths(paths)

    result: dict[str, Any] = {
        "n_files": len(files),
        "indexed": [],
        "skipped": [],
        "failed": [],
        "n_chunks": 0,
        "n_sections": 0,
        "prepared": [],
        "pending_inputs": [str(path) for path in files],
        "enrichment_requested": enrich,
    }

    if not files:
        logger.warning("Nenhum PDF encontrado em: %s", paths)
        return result

    touched: list[str] = []
    prepared: list[dict] = result["prepared"]
    started = time.monotonic()
    result["timings"] = {}

    def checkpoint() -> None:
        result["pending"] = pending_work(result, conn)
        if on_checkpoint:
            on_checkpoint(result)

    def progress(phase: str, done: int, total: int, message: str) -> None:
        checkpoint()
        if on_progress:
            on_progress(phase, done, total, message)

    checkpoint()

    # ---------- Pass 1: Docling ----------
    for index, pdf_path in enumerate(files, start=1):
        progress("parse", index - 1, len(files), pdf_path.name)
        doc_id = None
        try:
            doc_id = file_sha256(pdf_path)
            prior = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not force and is_indexed(doc_id, conn, ocr):
                with transaction(conn):
                    conn.execute("UPDATE documents SET filename = ?, source_path = ? WHERE doc_id = ?",
                                 (pdf_path.name, str(pdf_path), doc_id))
                result["skipped"].append({"doc_id": doc_id, "filename": pdf_path.name,
                                          "quality": decode_quality(prior["quality_json"])})
                touched.append(doc_id)
                _associate(collection_id, doc_id, conn)
                continue
            register_document(
                doc_id, pdf_path, get_pdf_page_count(pdf_path), conn=conn
            )

            manifest = parse_document(
                pdf_path,
                doc_id,
                on_progress=lambda done, total, msg: (
                    on_progress("parse", done, total, msg) if on_progress else None
                ),
                force=force,
                ocr=ocr,
                previous=read_manifest(doc_id, prior["artifact_manifest"]) if prior else None,
            )

            sections, chunks = segment_document(doc_id, manifest)

            if not chunks:
                raise RuntimeError("parse não produziu nenhum chunk")

            export_markdown(doc_id, manifest)
            # The artifact and its chunks become visible in one transaction.
            with transaction(conn):
                persist_segments(doc_id, sections, chunks, conn=conn, manifest=manifest)
                _associate(collection_id, doc_id, conn)
            try:
                activate_manifest(doc_id, manifest)
            except OSError:
                logger.exception("Não foi possível atualizar o atalho do manifest; a geração está no SQLite")

            stats = summarize(sections, chunks)
            result["n_sections"] += stats["n_sections"]
            result["n_chunks"] += stats["n_chunks"]
            prepared.append(
                {"doc_id": doc_id, "filename": pdf_path.name, **stats,
                 "quality": manifest["quality"], "manifest": manifest["manifest_file"]}
            )
            touched.append(doc_id)

        except Exception as exc:
            logger.exception("Falha ingerindo %s", pdf_path.name)
            if doc_id:
                _mark_failed(doc_id, str(exc), conn)
            result["failed"].append(
                {"doc_id": doc_id, "filename": pdf_path.name, "phase": "parse",
                 "error": str(exc), **({"manifest": exc.manifest} if hasattr(exc, "manifest") else {})}
            )
        finally:
            result["pending_inputs"].remove(str(pdf_path))
            checkpoint()
    result["timings"]["parse_segment"] = round(time.monotonic() - started, 3)
    checkpoint()

    # ---------- Pass 2: bge-m3 ----------
    if touched:
        embed_started = time.monotonic()
        embed_error = None
        try:
            result["n_embedded"] = embed_pending(
                doc_ids=touched,
                on_progress=lambda done, total, msg: (
                    progress("embed", done, total, msg)
                ), conn=conn, unload_when_done=False,
            )
        except Exception as exc:
            embed_error = str(exc)
            logger.exception("Falha no pass de embedding")
        result["timings"]["embed"] = round(time.monotonic() - embed_started, 3)
        successful = []
        skipped_ids = {item["doc_id"] for item in result["skipped"]}
        for item in [*prepared, *result["skipped"]]:
            row = conn.execute("SELECT status, vector_reset_pending FROM documents WHERE doc_id = ?", (item["doc_id"],)).fetchone()
            pending = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND embedded = 0", (item["doc_id"],)).fetchone()[0]
            if row["status"] == "indexed" and not pending and not row["vector_reset_pending"]:
                successful.append(item["doc_id"])
                if item["doc_id"] not in skipped_ids:
                    result["indexed"].append(item)
            else:
                result["failed"].append({**item, "phase": "embed", "error": embed_error or "Embedding incompleto"})
        result["skipped"] = [i for i in result["skipped"] if i["doc_id"] in successful]
        touched = successful
        checkpoint()

    # ---------- Pass 3: LLM local ----------
    if touched and enrich:
        from kb.enrich import run_pass

        progress("enrich", 0, 1, "iniciando resumos")
        enrich_started = time.monotonic()
        try:
            result["enrichment"] = run_pass(
                doc_ids=touched,
                on_progress=lambda done, total, msg: (
                    progress("enrich", done, total, msg)
                ), conn=conn,
            )
        except Exception as exc:
            result["enrichment"] = {"failed": 1, "error": str(exc)}
        result["timings"]["enrich"] = round(time.monotonic() - enrich_started, 3)

    result["timings"]["total"] = round(time.monotonic() - started, 3)
    checkpoint()
    return result


def apply_formula_corrections(
    doc_id: str,
    changes: list[dict],
    on_progress: ProgressFn | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """
    Publica LaTeX corrigido à mão e refaz o que deriva dele.

    Roda sobre o tokenizer e o embedder que a ativação deixou carregados. O resumo
    fica como está, e `summary_affected` diz se o trecho que ele leu mudou. Vale
    repetir depois de uma falha: correção já publicada não é conflito, e os vetores
    pendentes são retomados.
    """
    from kb.enrich import document_sample

    conn = conn or get_conn()
    started = time.monotonic()
    row = conn.execute("SELECT filename, artifact_manifest FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if not row or not row["artifact_manifest"]:
        raise ValueError(f"Documento sem artefatos publicados: {doc_id}")
    manifest = read_manifest(doc_id, row["artifact_manifest"])

    def progress(phase: str, done: int, total: int, message: str) -> None:
        if on_progress:
            on_progress(phase, done, total, message)

    progress("segment", 0, 1, f"{row['filename']}: aplicando correções")
    updated, applied, rejected = correct_formulas(doc_id, manifest, changes)
    summary_affected = False
    if updated is not manifest:
        sample = document_sample(doc_id, conn)
        sections, chunks = segment_document(doc_id, updated)
        if not chunks:
            raise RuntimeError("as correções não produziram nenhum chunk")
        export_markdown(doc_id, updated)
        persist_segments(doc_id, sections, chunks, conn=conn, manifest=updated, keep_summary=True)
        try:
            activate_manifest(doc_id, updated)
        except OSError:
            logger.exception("Não foi possível atualizar o atalho do manifest; a geração está no SQLite")
        summary_affected = document_sample(doc_id, conn) != sample
    progress("segment", 1, 1, f"{row['filename']}: correções aplicadas")
    embedded = embed_pending(
        doc_ids=[doc_id],
        on_progress=lambda done, total, msg: progress("embed", done, total, msg),
        conn=conn, unload_when_done=False,
    )
    return {"doc_id": doc_id, "filename": row["filename"], "applied": applied, "rejected": rejected,
            "summary_affected": summary_affected, "n_embedded": embedded, "quality": updated["quality"],
            "manifest": updated["manifest_file"], "timings": {"total": round(time.monotonic() - started, 3)}}


def pending_work(result: dict, conn: sqlite3.Connection) -> list[dict]:
    """Describe missing work without confusing catalog availability with success."""
    from kb.config import ENRICH_SECTIONS, ENRICH_MIN_SECTION_CHARS

    pending = []
    doc_ids = {item.get("doc_id") for key in ("prepared", "indexed", "skipped", "failed")
               for item in result.get(key, []) if item.get("doc_id")}
    for doc_id in sorted(doc_ids):
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if not row:
            continue
        quality = decode_quality(row["quality_json"])
        if quality.get("status") == "partial":
            pending.append({"doc_id": doc_id, "phase": "extraction", "quality": quality})
        chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND embedded = 0",
                              (doc_id,)).fetchone()[0]
        if chunks or row["vector_reset_pending"]:
            pending.append({"doc_id": doc_id, "phase": "embed", "chunks": chunks,
                            "vector_reset_pending": bool(row["vector_reset_pending"])})
        if result.get("enrichment_requested"):
            sections = conn.execute("SELECT COUNT(*) FROM sections WHERE doc_id = ? "
                                    "AND summary IS NULL AND n_chars >= ?",
                                    (doc_id, ENRICH_MIN_SECTION_CHARS)).fetchone()[0] if ENRICH_SECTIONS else 0
            if row["summary"] is None or sections:
                pending.append({"doc_id": doc_id, "phase": "enrich",
                                "document_summary": row["summary"] is None, "sections": sections})
    return pending


def _associate(
    collection_id: str | None,
    doc_id: str,
    conn: sqlite3.Connection,
) -> None:
    if collection_id:
        collections_api.add_documents(collection_id, [doc_id], conn=conn)


def _mark_failed(doc_id: str, error: str, conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET status = CASE WHEN COALESCE(n_chunks, 0) > 0 THEN status ELSE 'failed' END, error = ? WHERE doc_id = ?",
            (error[:2000], doc_id),
        )
