"""Public CLI for the standalone RAG system. JSON is the default output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from kb.session import SessionError


class JsonParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(json.dumps({"ok": False, "error": {"code": "USAGE_ERROR", "message": message}}))
        raise SystemExit(2)


def _missing(kind: str, identifier: str):
    raise SessionError("NOT_FOUND", f"{kind} não encontrado: {identifier}")


def _wait_job(job_id: str) -> dict[str, Any]:
    from kb.jobs import get_job

    previous = None
    while True:
        job = get_job(job_id)
        if job is None:
            return _missing("job", job_id)
        if job["status"] in {"done", "failed", "cancelled"}:
            if job["status"] != "done":
                raise SessionError("JOB_FAILED", job.get("error") or job["status"])
            return job
        progress = (job.get("pass_name"), job.get("progress"), job.get("total"))
        if sys.stderr.isatty() and progress != previous:
            print(f"{job_id}: {progress[0]} {progress[1]}/{progress[2]}", file=sys.stderr)
            previous = progress
        time.sleep(0.5)


def _run(args: argparse.Namespace) -> Any:
    from kb import session

    cmd = args.command
    if cmd == "on":
        return session.start(args.mode)
    if cmd == "off":
        return session.stop(cancel=args.cancel, interactive=not args.wait)
    if cmd == "switch":
        return session.switch(args.mode, cancel=args.cancel, interactive=not args.wait)
    if cmd == "session":
        return session.status()
    if cmd == "search":
        return session.call("interaction", "search", query=args.query,
                            collections=args.collection, doc_ids=args.doc_id,
                            top_k=args.top_k, expand=args.expand,
                            rerank=not args.no_rerank)
    if cmd in {"ingest", "reindex"}:
        payload = (
            {"paths": args.paths, "collection": args.collection,
             "collection_name": args.collection_name, "force": args.force,
             "enrich": not args.no_enrich}
            if cmd == "ingest" else
            {"collection": args.collection, "reset": args.reset}
        )
        result = session.call("ingestion", cmd, **payload)
        return result if args.background else _wait_job(result["job_id"])

    if cmd == "fetch":
        from kb.retrieval import fetch_pages, fetch_section

        if args.section_id:
            return fetch_section(args.section_id) or _missing("seção", args.section_id)
        if not args.doc_id or args.page_start is None:
            raise SessionError("USAGE_ERROR", "Informe --section-id ou --doc-id e --page-start")
        return fetch_pages(args.doc_id, args.page_start, args.page_end)
    if cmd == "outline":
        from kb.retrieval import get_outline

        return get_outline(args.doc_id, max_depth=args.max_depth)
    if cmd == "status":
        from kb.health import status

        return status(args.collection)
    if cmd == "doctor":
        from kb.health import consistency, orphan_artifacts

        return {"consistency": consistency(), "orphan_artifacts": orphan_artifacts()}
    if cmd == "models":
        from kb import config

        return {
            "embedding": {"provider": config.EMBED_PROVIDER, "model": config.EMBED_MODEL},
            "reranker": {"provider": config.RERANK_PROVIDER, "model": config.RERANK_MODEL,
                         "enabled": config.RERANK_ENABLED},
            "generation": {"provider": config.LLM_PROVIDER, "model": config.LLM_MODEL,
                           "enabled": config.LLM_ENABLED},
            "session": session.status(),
        }
    if cmd == "collections":
        from kb import collection as api

        action = args.action
        if action == "list":
            return api.list_all()
        if action == "show":
            return api.get(args.collection) or _missing("collection", args.collection)
        if action == "create":
            return api.create(args.name or args.collection, args.description,
                              collection_id=args.collection)
        if action == "update":
            return api.update(args.collection, args.name, args.description) or _missing(
                "collection", args.collection)
        if action == "delete":
            return api.delete(args.collection, delete_documents=args.delete_documents)
        if action == "add-docs":
            return {"added": api.add_documents(args.collection, args.doc_ids)}
        if action == "remove-docs":
            return {"removed": api.remove_documents(args.collection, args.doc_ids)}
    if cmd == "documents":
        from kb import collection as api

        if args.action == "list":
            return api.documents_in(args.collection)
        if args.action == "show":
            from kb.db import get_conn

            row = get_conn().execute("SELECT * FROM documents WHERE doc_id = ?",
                                     (args.doc_id,)).fetchone()
            return dict(row) if row else _missing("documento", args.doc_id)
        if args.action == "delete":
            return api.delete_document(args.doc_id, purge_artifacts=args.purge_artifacts)
    if cmd == "jobs":
        from kb.jobs import get_job, list_jobs

        if args.action == "list":
            return list_jobs(args.limit)
        if args.action == "show":
            return get_job(args.job_id) or _missing("job", args.job_id)
        if args.action == "cancel":
            active = session.status()
            if active.get("job_id") != args.job_id:
                raise SessionError("JOB_NOT_ACTIVE", f"Job {args.job_id} não está ativo")
            return session.stop(cancel=True)
    raise SessionError("USAGE_ERROR", f"Comando desconhecido: {cmd}")


def build_parser() -> JsonParser:
    parser = JsonParser(prog="kb", description="RAG local — ingestão e recuperação")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=JsonParser)
    p = sub.add_parser("on", help="liga sessão e carrega modelos")
    p.add_argument("--mode", choices=["interaction", "ingestion"], required=True)
    p = sub.add_parser("off", help="desliga sessão e libera recursos")
    p.add_argument("--cancel", action="store_true")
    p.add_argument("--wait", action="store_true")
    p = sub.add_parser("switch", help="troca explicitamente de modo")
    p.add_argument("--mode", choices=["interaction", "ingestion"], required=True)
    p.add_argument("--cancel", action="store_true")
    p.add_argument("--wait", action="store_true")
    sub.add_parser("session", help="estado da sessão sem carregar modelos")
    p = sub.add_parser("search", help="busca híbrida em modo interaction")
    p.add_argument("query")
    p.add_argument("--collection", action="append")
    p.add_argument("--doc-id", action="append")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--expand", choices=["section", "chunk"], default="section")
    p.add_argument("--no-rerank", action="store_true")
    p = sub.add_parser("ingest", help="indexa PDFs em modo ingestion")
    p.add_argument("paths", nargs="+")
    p.add_argument("--collection")
    p.add_argument("--collection-name")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-enrich", action="store_true")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser("reindex", help="reconstrói vetores em modo ingestion")
    p.add_argument("--collection")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser("fetch", help="lê seção ou páginas")
    p.add_argument("--section-id")
    p.add_argument("--doc-id")
    p.add_argument("--page-start", type=int)
    p.add_argument("--page-end", type=int)
    p = sub.add_parser("outline", help="sumário navegável")
    p.add_argument("doc_id")
    p.add_argument("--max-depth", type=int, default=3)
    p = sub.add_parser("status", help="estado da base sem carregar modelos")
    p.add_argument("--collection")
    sub.add_parser("doctor", help="confere índice e artefatos")
    sub.add_parser("models", help="modelos e provedores configurados")
    p = sub.add_parser("collections", help="gerencia collections")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    actions.add_parser("list")
    for action in ("show", "create", "update", "delete", "add-docs", "remove-docs"):
        item = actions.add_parser(action)
        item.add_argument("collection")
        if action in {"create", "update"}:
            item.add_argument("--name")
            item.add_argument("--description")
        if action == "delete":
            item.add_argument("--delete-documents", action="store_true")
        if action in {"add-docs", "remove-docs"}:
            item.add_argument("doc_ids", nargs="+")
    p = sub.add_parser("documents", help="lista e gerencia documentos")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    actions.add_parser("list").add_argument("--collection")
    actions.add_parser("show").add_argument("doc_id")
    item = actions.add_parser("delete")
    item.add_argument("doc_id")
    item.add_argument("--purge-artifacts", action="store_true")
    p = sub.add_parser("jobs", help="progresso e cancelamento")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    actions.add_parser("show").add_argument("job_id")
    actions.add_parser("cancel").add_argument("job_id")
    actions.add_parser("list").add_argument("--limit", type=int, default=10)
    return parser


def _human(data: Any) -> str:
    if isinstance(data, dict) and "results" in data:
        return "\n\n".join(
            f"{item['filename']} | p. {item['pages']}\n{item['heading_path']}\n{item['text']}"
            for item in data["results"]
        ) or "Nenhum resultado"
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    human = "--human" in argv or "-H" in argv
    argv = [arg for arg in argv if arg not in {"--human", "-H", "--json"}]
    args = build_parser().parse_args(argv)
    try:
        data = _run(args)
    except SessionError as exc:
        error = {"code": exc.code, "message": str(exc)}
        if exc.hint:
            error["hint"] = exc.hint
        print(json.dumps({"ok": False, "error": error}))
        return 1
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(json.dumps({"ok": False, "error": {
            "code": "OPERATION_FAILED", "message": str(exc),
        }}))
        return 1
    print(_human(data) if human else json.dumps({"ok": True, "data": data}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
