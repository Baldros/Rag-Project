"""Public CLI for the standalone RAG system. JSON is the default output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from kb.config import FINAL_TOP_K
from kb.session import SessionError


class JsonParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(json.dumps({"ok": False, "error": {"code": "USAGE_ERROR", "message": message}}))
        raise SystemExit(2)


def _missing(kind: str, identifier: str):
    raise SessionError("NOT_FOUND", f"{kind} não encontrado: {identifier}")


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Informe um inteiro maior que zero") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("Informe um inteiro maior que zero")
    return value


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Informe um inteiro não negativo") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("Informe um inteiro não negativo")
    return value


def _add_paging(parser: argparse.ArgumentParser, limit_default: int | None = None) -> None:
    """Paginação das listagens, descrita num lugar só."""
    parser.add_argument("--limit", type=_positive_int, default=limit_default, metavar="N",
                        help="itens por página")
    parser.add_argument("--offset", type=_nonnegative_int, default=0, metavar="N",
                        help="itens a pular antes do primeiro desta página")


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
        return session.stop(cancel=args.cancel, interactive=not (args.wait or args.no_wait),
                            fail_if_busy=args.no_wait)
    if cmd == "switch":
        return session.switch(args.mode, cancel=args.cancel,
                              interactive=not (args.wait or args.no_wait),
                              fail_if_busy=args.no_wait)
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
        result = fetch_pages(args.doc_id, args.page_start, args.page_end)
        return result if result["filename"] is not None else _missing("documento", args.doc_id)
    if cmd == "outline":
        from kb.retrieval import get_outline

        result = get_outline(args.doc_id, max_depth=args.max_depth)
        return result if "error" not in result else _missing("documento", args.doc_id)
    if cmd == "status":
        from kb.health import status

        result = status(args.collection)
        if args.collection and "error" in result.get("collection", {}):
            return _missing("collection", args.collection)
        return result
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
            return api.list_all(limit=args.limit, offset=args.offset)
        if action == "show":
            return api.get(args.collection) or _missing("collection", args.collection)
        if action == "create":
            return api.create(args.name or args.collection, args.description,
                              collection_id=args.collection)
        if action == "update":
            return api.update(args.collection, args.name, args.description) or _missing(
                "collection", args.collection)
        if action == "delete":
            if not api.get(args.collection):
                return _missing("collection", args.collection)
            return api.delete(args.collection, delete_documents=args.delete_documents)
        if action in {"add-docs", "remove-docs"}:
            # INSERT OR IGNORE e DELETE não reclamam de collection inexistente:
            # sem esta checagem a escrita reporta sucesso sem ter feito nada.
            if not api.get(args.collection):
                return _missing("collection", args.collection)
            if action == "add-docs":
                return {"added": api.add_documents(args.collection, args.doc_ids)}
            return {"removed": api.remove_documents(args.collection, args.doc_ids)}
    if cmd == "documents":
        from kb import collection as api

        if args.action == "list":
            return api.documents_in(args.collection, limit=args.limit, offset=args.offset)
        if args.action == "show":
            from kb.db import get_conn

            row = get_conn().execute("SELECT * FROM documents WHERE doc_id = ?",
                                     (args.doc_id,)).fetchone()
            return dict(row) if row else _missing("documento", args.doc_id)
        if args.action == "delete":
            result = api.delete_document(args.doc_id, purge_artifacts=args.purge_artifacts)
            return result if result["ok"] else _missing("documento", args.doc_id)
    if cmd == "jobs":
        from kb.jobs import get_job, list_jobs

        if args.action == "list":
            return list_jobs(args.limit, offset=args.offset)
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
    p.add_argument("--mode", choices=["interaction", "ingestion"], required=True,
                   help="interaction carrega embedding e reranker; ingestion, o parser")
    p = sub.add_parser("off", help="desliga sessão e libera recursos")
    p.add_argument("--cancel", action="store_true",
                   help="cancela o job ativo em vez de aguardá-lo")
    waiting = p.add_mutually_exclusive_group()
    waiting.add_argument("--wait", action="store_true",
                         help="aguarda o job ativo sem perguntar")
    waiting.add_argument("--no-wait", action="store_true", help="retorna BUSY se houver job ativo")
    p = sub.add_parser("switch", help="troca explicitamente de modo")
    p.add_argument("--mode", choices=["interaction", "ingestion"], required=True,
                   help="modo de destino")
    p.add_argument("--cancel", action="store_true",
                   help="cancela o job ativo em vez de aguardá-lo")
    waiting = p.add_mutually_exclusive_group()
    waiting.add_argument("--wait", action="store_true",
                         help="aguarda o job ativo sem perguntar")
    waiting.add_argument("--no-wait", action="store_true", help="retorna BUSY se houver job ativo")
    sub.add_parser("session", help="estado da sessão sem carregar modelos")
    p = sub.add_parser("search", help="busca híbrida em modo interaction")
    p.add_argument("query", help="pergunta em linguagem natural")
    p.add_argument("--collection", action="append", metavar="ID",
                   help="id vindo de `collections list`, repetível; nome não serve")
    p.add_argument("--doc-id", action="append", metavar="ID",
                   help="id vindo de `documents list`, repetível")
    p.add_argument("--top-k", type=_positive_int, default=FINAL_TOP_K, metavar="N",
                   help=f"quantas passagens devolver (padrão {FINAL_TOP_K})")
    p.add_argument("--expand", choices=["section", "chunk"], default="section",
                   help="section devolve a seção-pai do acerto; chunk, o trecho cru")
    p.add_argument("--no-rerank", action="store_true",
                   help="pula o reranker: mais rápido e menos preciso")
    p = sub.add_parser("ingest", help="indexa PDFs em modo ingestion")
    p.add_argument("paths", nargs="+",
                   help="arquivos ou pastas de PDF, dentro das raízes permitidas")
    p.add_argument("--collection", metavar="ID",
                   help="collection de destino, criada se não existir")
    p.add_argument("--collection-name", metavar="NOME",
                   help="nome legível da collection, quando ela for criada")
    p.add_argument("--force", action="store_true", help="reprocessa documento já indexado")
    p.add_argument("--no-enrich", action="store_true", help="não gera resumos com o LLM")
    p.add_argument("--background", action="store_true",
                   help="devolve job_id em vez de aguardar o fim")
    p = sub.add_parser("reindex", help="reconstrói vetores em modo ingestion")
    p.add_argument("--collection", metavar="ID", help="limita a reindexação a uma collection")
    p.add_argument("--reset", action="store_true",
                   help="invalida os vetores existentes antes de reembedar")
    p.add_argument("--background", action="store_true",
                   help="devolve job_id em vez de aguardar o fim")
    p = sub.add_parser("fetch", help="lê seção ou páginas")
    p.add_argument("--section-id", metavar="ID", help="id de seção devolvido numa citação")
    p.add_argument("--doc-id", metavar="ID", help="documento a ler por páginas")
    p.add_argument("--page-start", type=_positive_int, metavar="N",
                   help="primeira página do intervalo, a partir de 1")
    p.add_argument("--page-end", type=_positive_int, metavar="N",
                   help="última página do intervalo (padrão: a inicial)")
    p = sub.add_parser("outline", help="sumário navegável")
    p.add_argument("doc_id", help="id do documento")
    p.add_argument("--max-depth", type=_positive_int, default=3, metavar="N",
                   help="nível máximo de heading no sumário (padrão 3)")
    p = sub.add_parser("status", help="estado da base sem carregar modelos")
    p.add_argument("--collection", metavar="ID", help="detalha uma collection pelo id")
    sub.add_parser("doctor", help="confere índice e artefatos")
    sub.add_parser("models", help="modelos e provedores configurados")
    p = sub.add_parser("collections", help="gerencia collections")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    _add_paging(actions.add_parser("list", help="lista collections e suas contagens"))
    for action, summary in (
        ("show", "mostra uma collection"),
        ("create", "cria uma collection"),
        ("update", "altera nome ou descrição"),
        ("delete", "exclui uma collection"),
        ("add-docs", "associa documentos a uma collection"),
        ("remove-docs", "desfaz associações, sem excluir documentos"),
    ):
        item = actions.add_parser(action, help=summary)
        item.add_argument("collection", help="id da collection")
        if action in {"create", "update"}:
            item.add_argument("--name", metavar="NOME", help="nome legível")
            item.add_argument("--description", metavar="TEXTO", help="descrição livre")
        if action == "delete":
            item.add_argument("--delete-documents", action="store_true",
                              help="também apaga documentos que ficariam sem collection")
        if action in {"add-docs", "remove-docs"}:
            item.add_argument("doc_ids", nargs="+", help="ids de documentos")
    p = sub.add_parser("documents", help="lista e gerencia documentos")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    item = actions.add_parser("list", help="lista metadados de documentos")
    item.add_argument("--collection", metavar="ID", help="filtra por collection")
    _add_paging(item)
    actions.add_parser("show", help="mostra um documento").add_argument(
        "doc_id", help="id do documento")
    item = actions.add_parser("delete", help="exclui documento, chunks e vetores")
    item.add_argument("doc_id", help="id do documento")
    item.add_argument("--purge-artifacts", action="store_true",
                      help="também apaga os artefatos de parse em disco")
    p = sub.add_parser("jobs", help="progresso e cancelamento")
    actions = p.add_subparsers(dest="action", required=True, parser_class=JsonParser)
    actions.add_parser("show", help="mostra um job").add_argument("job_id", help="id do job")
    actions.add_parser("cancel", help="cancela o job ativo").add_argument(
        "job_id", help="id do job")
    _add_paging(actions.add_parser("list", help="lista jobs, inclusive concluídos"),
                limit_default=10)
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
