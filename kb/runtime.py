"""Private process owning the models of one manually selected RAG mode."""

from __future__ import annotations

import argparse
import os
import threading
import traceback
from datetime import datetime, timezone
from multiprocessing.connection import Listener
from typing import Any

from kb.db import init_db
from kb.session import MODES, SessionError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Runtime:
    def __init__(self, mode: str, nonce: str) -> None:
        self.mode = mode
        self.nonce = nonce
        self.running = True
        self.job_id: str | None = None
        self.job_thread: threading.Thread | None = None
        self.pass_name: str | None = None

    def prepare(self) -> None:
        from kb.providers import validate_config

        validate_config()
        init_db().close()
        if self.mode == "interaction":
            from kb.config import RERANK_ENABLED
            from kb.models import preload
            from kb.vectors import get_index

            get_index()  # Refuse an incompatible embedding before loading GPU models.
            preload(rerank=RERANK_ENABLED)
        else:
            self._prepare_ingestion()

    @staticmethod
    def _prepare_ingestion() -> None:
        from kb.ingest.parse import get_converter
        from kb.ingest.segment import get_chunker

        get_converter()
        get_chunker()

    @staticmethod
    def _release_ingestion() -> None:
        from kb.ingest.parse import release_converter
        from kb.ingest.segment import release_chunker

        release_converter()
        release_chunker()

    def ping(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "mode": self.mode,
            "ready": True,
            "job_id": self.job_id,
            "pass_name": self.pass_name,
        }

    def _run_job(self, job_id: str, action: str, payload: dict[str, Any]) -> None:
        from kb.jobs import update_job

        update_job(job_id, status="running", pid=os.getpid(), started_at=_now())

        def progress(phase: str, done: int, total: int, message: str) -> None:
            self.pass_name = phase
            update_job(
                job_id, pass_name=phase, progress=done, total=total,
                message=message[:300],
            )

        try:
            if action == "ingest":
                from kb.ingest.pipeline import ingest_paths

                result = ingest_paths(
                    payload["paths"], collection_id=payload.get("collection"),
                    on_progress=progress, force=payload.get("force", False),
                    enrich=payload.get("enrich", True),
                )
            else:
                self._release_ingestion()
                from kb.collection import resolve_scope
                from kb.ingest.embed import embed_pending, reset_embeddings
                from kb.models import EMBEDDER

                collection = payload.get("collection")
                doc_ids = resolve_scope([collection]) if collection else None
                if collection and not doc_ids:
                    raise ValueError(f"Collection sem documentos: {collection}")
                reset = payload.get("reset", False)
                if reset:
                    EMBEDDER.load()  # Validate load before replacing derived vectors.
                invalidated = reset_embeddings(doc_ids) if reset else 0
                embedded = embed_pending(
                    doc_ids=doc_ids,
                    on_progress=lambda done, total, message: progress(
                        "embed", done, total, message
                    ),
                )
                from kb.models import unload_all

                unload_all()  # Also covers an empty reindex after preloading.
                result = {"invalidated": invalidated, "embedded": embedded}
            self._prepare_ingestion()
            failed = action == "ingest" and bool(result["failed"]) and not result["indexed"]
            update_job(
                job_id, status="failed" if failed else "done", finished_at=_now(),
                pass_name=None, message=str(result)[:300],
                error=str(result["failed"])[:2000] if failed else None,
            )
        except Exception as exc:
            from kb.models import unload_all

            unload_all()
            update_job(
                job_id, status="failed", finished_at=_now(), pass_name=None,
                error=f"{exc}\n{traceback.format_exc()}"[:2000],
            )
            try:
                self._prepare_ingestion()
            except Exception:
                # A session whose ready models cannot be restored must end.
                self.running = False
        finally:
            self.pass_name = None
            self.job_id = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        payload = request.get("payload") or {}
        if action == "ping":
            return self.ping()
        if action == "shutdown":
            if self.job_id:
                raise SessionError("BUSY", "Job em andamento; aguarde ou cancele")
            self.running = False
            return {"active": False}
        if self.mode == "interaction" and action == "search":
            from kb.retrieval import search

            return {"results": search(
                payload["query"], collections=payload.get("collections"),
                doc_ids=payload.get("doc_ids"), top_k=payload.get("top_k", 5),
                expand=payload.get("expand", "section"),
                use_rerank=payload.get("rerank", True),
            )}
        if self.mode == "ingestion" and action in {"ingest", "reindex"}:
            if self.job_id:
                raise SessionError("BUSY", f"Job {self.job_id} em andamento")
            from kb.jobs import create_job, validate_path

            if action == "ingest":
                payload["paths"] = [str(validate_path(path)) for path in payload["paths"]]
                from kb.ingest.pipeline import expand_paths

                if not expand_paths(payload["paths"]):
                    raise FileNotFoundError("Nenhum PDF encontrado nos caminhos informados")
                if payload.get("collection"):
                    from kb.collection import create

                    create(payload.get("collection_name") or payload["collection"],
                           collection_id=payload["collection"])
            job_id = create_job(
                kind=action,
                target="; ".join(payload.get("paths", [])) if action == "ingest"
                else payload.get("collection") or "toda a base",
                collection_id=payload.get("collection"),
            )
            self.job_id = job_id
            self.job_thread = threading.Thread(
                target=self._run_job, args=(job_id, action, payload),
                name=f"kb-{action}-{job_id}", daemon=True,
            )
            self.job_thread.start()
            return {"job_id": job_id, "status": "queued"}
        raise SessionError("WRONG_MODE", f"Operação {action} indisponível em {self.mode}")

    def close(self) -> None:
        if self.mode == "interaction":
            from kb.models import unload_all

            unload_all()
        else:
            self._release_ingestion()
            from kb import llm

            llm.unload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--address", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    key = os.environ.pop("KB_SESSION_AUTHKEY", None)
    if not key:
        raise RuntimeError("Chave de sessão ausente")
    runtime = Runtime(args.mode, args.nonce)
    try:
        runtime.prepare()
        with Listener(
            args.address,
            family="AF_PIPE" if os.name == "nt" else "AF_UNIX",
            authkey=bytes.fromhex(key),
        ) as listener:
            while runtime.running:
                with listener.accept() as connection:
                    try:
                        data = runtime.handle(connection.recv())
                        reply = {"ok": True, "data": data}
                    except SessionError as exc:
                        reply = {"ok": False, "error": {
                            "code": exc.code, "message": str(exc), "hint": exc.hint,
                        }}
                    except Exception as exc:
                        reply = {"ok": False, "error": {
                            "code": "RUNTIME_ERROR", "message": str(exc),
                        }}
                    connection.send({"nonce": runtime.nonce, **reply})
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
