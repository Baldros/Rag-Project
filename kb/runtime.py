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
from kb.startup_log import StartupLog


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _JobCancelled(BaseException):
    """Stop at a phase boundary without being swallowed as a document error."""


class Runtime:
    def __init__(self, mode: str, nonce: str, startup: StartupLog | None = None) -> None:
        self.mode = mode
        self.nonce = nonce
        self.startup = startup or StartupLog(None, "runtime")
        self.running = True
        self.job_id: str | None = None
        self.job_thread: threading.Thread | None = None
        self.pass_name: str | None = None
        self._job_lock = threading.RLock()
        self._cancelling = False

    def prepare(self) -> None:
        """Activation: everything a mode needs, loaded once, until the user turns it off."""
        with self.startup.stage("validate_config"):
            from kb.providers import validate_config

            validate_config()
        with self.startup.stage("database"):
            from kb.jobs import reconcile

            conn = init_db()
            try:
                reconcile(conn)
            finally:
                conn.close()
        if self.mode == "interaction":
            self._prepare_interaction()
        else:
            self._prepare_ingestion()

    def _prepare_interaction(self) -> None:
        with self.startup.stage("vector_index"):
            from kb.vectors import get_index

            get_index()  # Refuse an incompatible embedding before loading GPU models.
        with self.startup.stage("models_load"):
            from kb.config import RERANK_ENABLED
            from kb.models import preload

            preload(rerank=RERANK_ENABLED)

    def _prepare_ingestion(self) -> None:
        # Imports are separate stages because they are the slow part on a cold
        # disk, and the first place a broken installation shows up.
        with self.startup.stage("vector_index"):
            from kb.config import EMBED_ID
            from kb.vectors import get_index

            # A mismatch is what `reindex --reset` repairs, so it cannot refuse activation.
            identity = get_index(allow_mismatch=True).embedding_identity()
            if identity != EMBED_ID:
                self.startup.warning("embedding", f"Índice criado com {identity!r}, configuração atual "
                                                  f"{EMBED_ID!r}. Execute a reindexação com reset.")
        with self.startup.stage("parser_import"):
            import docling.document_converter  # noqa: F401
            from docling.datamodel.base_models import InputFormat
            from kb.ingest.parse import get_converter
        with self.startup.stage("parser_load"):
            get_converter().initialize_pipeline(InputFormat.PDF)
        with self.startup.stage("chunker_import"):
            import docling.chunking  # noqa: F401
            from kb.ingest.segment import get_chunker
        with self.startup.stage("tokenizer_load"):
            get_chunker()
        with self.startup.stage("embedder_load"):
            from kb.models import EMBEDDER

            EMBEDDER.load()
        with self.startup.stage("checks"):
            self._report_checks()

    def _report_checks(self) -> None:
        """Irregularities that do not stop the session, visible from the start."""
        import torch
        from kb import llm
        from kb.config import DEVICE, LLM_ENABLED, LLM_MODEL, OLLAMA_URL

        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            self.startup.warning("gpu", "KB_DEVICE pede CUDA, mas ele não está disponível: a conversão vai para a CPU.")
        if LLM_ENABLED and not llm.is_available():
            self.startup.warning("ollama", f"Ollama ou o modelo {LLM_MODEL} indisponível em {OLLAMA_URL}: "
                                           "os resumos ficarão pendentes.")

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
        from kb.jobs import update_job, result_status, finish_job

        def progress(phase: str, done: int, total: int, message: str) -> None:
            with self._job_lock:
                if self._cancelling or not update_job(
                    job_id, pass_name=phase, progress=done, total=total,
                    message=message[:300],
                ):
                    raise _JobCancelled()
                self.pass_name = phase

        result = None
        status = "failed"
        error = None
        try:
            if not update_job(job_id, status="running", started_at=_now()):
                return
            if action == "ingest":
                from kb.ingest.pipeline import ingest_paths

                result = ingest_paths(
                    payload["paths"], collection_id=payload.get("collection"),
                    on_progress=progress, force=payload.get("force", False),
                    enrich=payload.get("enrich", True),
                    ocr=payload.get("ocr"),
                    on_checkpoint=lambda snapshot: update_job(job_id, result=snapshot),
                )
            else:
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
                result = {"invalidated": invalidated, "embedded": 0}
                update_job(job_id, result=result)

                def reindex_progress(done: int, total: int, message: str) -> None:
                    result["embedded"] = done
                    update_job(job_id, result=result)
                    progress("embed", done, total, message)

                result["embedded"] = embed_pending(doc_ids=doc_ids, on_progress=reindex_progress,
                                                   unload_when_done=False)
            status = result_status(result) if action == "ingest" else "done"
            if status == "failed":
                error = "Ingestão incompleta; consulte result"
        except BaseException as exc:
            error = f"{exc}\n{traceback.format_exc()}"[:2000]
        finally:
            try:
                update_job(job_id, status="finalizing", error=error,
                           **({"result": result} if result is not None else {}))
                cleanup_errors = self._cleanup_job()
                with self._job_lock:
                    # Once cancellation is accepted the controller owns finalization.
                    if not self._cancelling:
                        finish_job(
                            job_id, "failed" if cleanup_errors else status,
                            cleanup_status="failed" if cleanup_errors else "done",
                            cleanup_error="; ".join(cleanup_errors) or None,
                            error=error or ("Falha na limpeza de recursos" if cleanup_errors else None),
                            message="Job finalizado; detalhes no campo result",
                        )
                        self.pass_name = None
                        self.job_id = None
            except BaseException:
                # A failed durable write must never leave an apparently idle
                # runtime accepting more jobs. Exit so recovery can reconcile it.
                traceback.print_exc()
                os._exit(1)

    def _cleanup_job(self) -> list[str]:
        """Parser, chunker and embedder stay loaded until the user turns the session off."""
        from kb import llm
        from kb.models import free_vram

        errors = []
        try:
            # The generation model belongs to Ollama and only enters for the summary pass.
            if llm.unload() is False:
                errors.append("generation: descarga não confirmada")
        except Exception as exc:
            errors.append(f"generation: {exc}")
        free_vram()
        return errors

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._job_lock:
            return self._handle(request)

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        payload = request.get("payload") or {}
        if action == "ping":
            return self.ping()
        if action == "cancel":
            from kb.jobs import request_cancel

            job_id = payload["job_id"]
            if self.job_id != job_id or not request_cancel(
                job_id, payload["cancel_pid"], payload["cancel_owner_started"],
            ):
                raise SessionError("JOB_NOT_ACTIVE", f"Job {job_id} não está ativo")
            self._cancelling = True
            self.running = False  # Refuse subsequent work even if the controller disconnects.
            return {"job_id": job_id}
        if not self.running:
            raise SessionError("SESSION_STOPPING", "Sessão está encerrando")
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
            try:
                self.job_thread = threading.Thread(
                    target=self._run_job, args=(job_id, action, payload),
                    name=f"kb-{action}-{job_id}", daemon=True,
                )
                self.job_thread.start()
            except BaseException as exc:
                from kb.jobs import finish_job

                finish_job(job_id, "failed", error=f"Falha ao iniciar worker: {exc}",
                           cleanup_status="done")
                self.job_id = None
                raise
            return {"job_id": job_id, "status": "queued"}
        raise SessionError("WRONG_MODE", f"Operação {action} indisponível em {self.mode}")

    def close(self) -> None:
        if self.mode == "interaction":
            from kb.models import unload_all

            unload_all()
        else:
            self._release_ingestion()
            from kb import llm
            from kb.models import unload_all

            unload_all()
            llm.unload()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--address", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--startup-id")
    args = parser.parse_args()
    key = os.environ.pop("KB_SESSION_AUTHKEY", None)
    if not key:
        raise RuntimeError("Chave de sessão ausente")
    startup = StartupLog(args.startup_id, "runtime")
    startup.emit("stage.finished", stage="process_start", pid=os.getpid())
    runtime = Runtime(args.mode, args.nonce, startup)
    try:
        runtime.prepare()
        with startup.stage("listener_bind"):
            server = Listener(
                args.address,
                family="AF_PIPE" if os.name == "nt" else "AF_UNIX",
                authkey=bytes.fromhex(key),
            )
        with server as listener:
            startup.emit("ready", pid=os.getpid())
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
                    try:
                        connection.send({"nonce": runtime.nonce, **reply})
                    except (BrokenPipeError, EOFError, OSError):
                        # The accepted job belongs to the runtime, not its caller.
                        pass
    finally:
        if runtime._cancelling:
            # No cleanup may race the still-running ingestion thread. The
            # cancellation controller waits for exit before external cleanup.
            os._exit(0)
        runtime.close()


if __name__ == "__main__":
    main()
