"""
Worker de ingestão — entrypoint do subprocesso.

Roda isolado do servidor MCP para que os modelos do Docling nunca dividam VRAM
com o embedder e o reranker. Quando termina, o processo morre e a GPU volta
limpa por construção, sem depender de `empty_cache()`.

Reporta progresso pela tabela `jobs`, que é o canal de leitura do servidor.

Uso:
    python -m kb.ingest.worker --job-id abc123 --collection fisica-3 -- a.pdf b.pdf
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from datetime import datetime, timezone

from kb.db import init_db
from kb.jobs import update_job

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb.ingest.worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-enrich", action="store_true")
    parser.add_argument("paths", nargs="+")

    args = parser.parse_args(argv)

    conn = init_db()
    job_id = args.job_id

    # Regrava o pid a partir do próprio processo: é a fonte confiável para o
    # servidor checar se o job ainda está vivo.
    update_job(
        job_id,
        status="running",
        pid=os.getpid(),
        started_at=_now(),
        pass_name="parse",
        conn=conn,
    )

    def on_progress(phase: str, done: int, total: int, message: str) -> None:
        update_job(
            job_id,
            pass_name=phase,
            progress=done,
            total=max(total, 1),
            message=message[:300],
            conn=conn,
        )

    try:
        from kb.ingest.pipeline import ingest_paths

        result = ingest_paths(
            args.paths,
            collection_id=args.collection,
            on_progress=on_progress,
            force=args.force,
            enrich=not args.no_enrich,
            conn=conn,
        )

        summary = (
            f"{len(result['indexed'])} indexado(s), "
            f"{len(result['skipped'])} pulado(s), "
            f"{len(result['failed'])} falho(s); "
            f"{result['n_chunks']} chunks"
        )

        update_job(
            job_id,
            status="failed" if result["failed"] and not result["indexed"] else "done",
            pass_name=None,
            message=summary,
            error="; ".join(
                f"{item['filename']}: {item['error']}" for item in result["failed"]
            )[:2000]
            or None,
            finished_at=_now(),
            conn=conn,
        )

        logger.info("Job %s concluído: %s", job_id, summary)
        return 0

    except Exception as exc:
        logger.exception("Job %s falhou", job_id)
        update_job(
            job_id,
            status="failed",
            error=f"{exc}\n{traceback.format_exc()}"[:2000],
            finished_at=_now(),
            conn=conn,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
