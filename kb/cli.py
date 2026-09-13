"""
CLI do knowledge base.

Mesmas funções que o servidor MCP expõe, para uso desatendido e para depurar sem
precisar de um agente no meio.

    python -m kb.cli ingest "E:/Estudo/Fisica" --collection fisica-3
    python -m kb.cli search "campo elétrico de um dipolo" --collection fisica-3
    python -m kb.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kb.db import init_db


def _print(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def cmd_ingest(args) -> int:
    from kb.ingest.pipeline import ingest_paths
    from kb.jobs import spawn_ingest

    if args.collection:
        from kb import collection as collections_api

        collections_api.create(args.collection_name or args.collection, collection_id=args.collection)

    if args.background:
        _print(spawn_ingest(args.paths, args.collection, force=args.force))
        return 0

    started = time.perf_counter()

    def on_progress(phase: str, done: int, total: int, message: str) -> None:
        print(f"  [{phase}] {done}/{total} {message}", file=sys.stderr, flush=True)

    result = ingest_paths(
        args.paths,
        collection_id=args.collection,
        on_progress=on_progress,
        force=args.force,
        enrich=not args.no_enrich,
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 1)

    _print(result)
    return 1 if result["failed"] and not result["indexed"] else 0


def cmd_search(args) -> int:
    from kb.retrieval import search

    started = time.perf_counter()
    results = search(
        args.query,
        collections=[args.collection] if args.collection else None,
        top_k=args.top_k,
        expand="chunk" if args.chunks else "section",
        use_rerank=not args.no_rerank,
    )
    elapsed = time.perf_counter() - started

    if args.json:
        _print(results)
        return 0

    print(f"\n{len(results)} resultado(s) em {elapsed * 1000:.0f}ms\n")

    for index, item in enumerate(results, start=1):
        print("=" * 78)
        print(f"[{index}] {item['filename']} | p. {item['pages']}")
        print(f"    {item['heading_path']}")
        print(f"    scores: {item['scores']}")
        print()
        text = item["text"]
        print(text if args.full else text[:800] + ("…" if len(text) > 800 else ""))
        print()

    return 0


def cmd_status(args) -> int:
    from kb.health import status

    _print(status(args.collection))
    return 0


def cmd_collections(args) -> int:
    from kb.collection import list_all

    _print(list_all())
    return 0


def cmd_documents(args) -> int:
    from kb.collection import documents_in

    _print(
        [
            {
                "doc_id": document["doc_id"][:16],
                "filename": document["filename"],
                "pages": document["n_pages"],
                "chunks": document["n_chunks"],
                "status": document["status"],
                "summary": bool(document["summary"]),
            }
            for document in documents_in(args.collection)
        ]
    )
    return 0


def cmd_outline(args) -> int:
    from kb.retrieval import get_outline

    _print(get_outline(args.doc_id, max_depth=args.max_depth))
    return 0


def cmd_reindex(args) -> int:
    from kb.ingest.embed import embed_pending, reset_embeddings

    if args.reset:
        removed = reset_embeddings()
        print(f"invalidados {removed} chunks", file=sys.stderr)

    count = embed_pending(
        on_progress=lambda done, total, msg: print(
            f"  [embed] {done}/{total}", file=sys.stderr, flush=True
        )
    )
    _print({"embedded": count})
    return 0


def cmd_jobs(args) -> int:
    from kb.jobs import get_job, list_jobs

    _print(get_job(args.job_id) if args.job_id else list_jobs(args.limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="Knowledge base — ETL e retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="indexa PDFs (arquivo ou pasta)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--collection", help="slug da collection de destino")
    p.add_argument("--collection-name", help="nome legível, se for criar")
    p.add_argument("--force", action="store_true", help="reprocessa mesmo se já indexado")
    p.add_argument("--background", action="store_true", help="dispara subprocesso e retorna")
    p.add_argument(
        "--no-enrich",
        action="store_true",
        help="pula o Pass 3 (resumos com LLM local)",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="busca híbrida")
    p.add_argument("query")
    p.add_argument("--collection")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--chunks", action="store_true", help="sem expansão para seção")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--full", action="store_true", help="texto completo")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="meta-condição da base")
    p.add_argument("--collection")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collections", help="lista collections")
    p.set_defaults(func=cmd_collections)

    p = sub.add_parser("documents", help="lista documentos")
    p.add_argument("--collection")
    p.set_defaults(func=cmd_documents)

    p = sub.add_parser("outline", help="sumário de um documento")
    p.add_argument("doc_id")
    p.add_argument("--max-depth", type=int, default=3)
    p.set_defaults(func=cmd_outline)

    p = sub.add_parser("reindex", help="recompõe o índice vetorial a partir do SQLite")
    p.add_argument("--reset", action="store_true", help="invalida tudo antes")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("jobs", help="jobs de ingestão")
    p.add_argument("job_id", nargs="?")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_jobs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
