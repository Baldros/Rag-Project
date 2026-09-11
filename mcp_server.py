"""
Servidor MCP do knowledge base.

Expõe a base como capacidade para qualquer agente: buscar, ler, navegar,
ingerir e inspecionar. Nada de geração de texto — quem raciocina é o agente do
outro lado; aqui só existe recuperação e gerência.

Registro no Claude Code:

    claude mcp add knowledge-base -- \\
        E:/Rag-Project/.venv/Scripts/python.exe E:/Rag-Project/mcp_server.py

As docstrings destas funções são o que o agente lê para escolher a tool certa,
então elas são parte da interface, não comentário.
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server import MCPServer

from kb.db import init_db

# stderr: stdout é o canal do protocolo MCP e qualquer print o corrompe.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("kb.mcp")


def _warm_models() -> None:
    """Sobe os modelos de recuperação fora da thread do protocolo."""
    try:
        from kb.models import preload

        preload()
    except Exception:
        # Falhar aqui não pode derrubar o servidor: as tools que não dependem de
        # modelo (listagens, outline, fetch, status) continuam servindo, e a
        # busca tenta carregar sob demanda.
        logger.exception("Pré-carregamento dos modelos falhou")


@asynccontextmanager
async def lifespan(_server: MCPServer):
    """
    O ciclo de vida dos modelos é o do servidor.

    Sobe: dispara o carregamento em background, para o handshake responder na
    hora enquanto os ~150s de import e carga correm em paralelo com a
    inicialização do cliente. Uma busca que chegue antes do fim simplesmente
    espera no mesmo lock, em vez de disparar uma segunda carga.

    Cai: devolve a VRAM explicitamente, sem depender do fim do processo.
    """
    init_db()

    warmer = threading.Thread(target=_warm_models, name="kb-warmup", daemon=True)
    warmer.start()
    logger.info("knowledge-base MCP pronto (stdio); modelos carregando em background")

    try:
        yield {}
    finally:
        from kb.models import unload_all

        released = unload_all()
        logger.info("Servidor encerrado; VRAM liberada: %s", released or "nada carregado")


server = MCPServer(
    lifespan=lifespan,
    name="knowledge-base",
    instructions=(
        "Base de conhecimento sobre documentos PDF indexados, organizada em "
        "collections (recortes temáticos independentes).\n\n"
        "Seu papel aqui é **consumir** conhecimento já processado. Fluxo usual: "
        "`list_collections` para ver o que existe, `search` para encontrar "
        "passagens (devolve a seção inteira, com arquivo e páginas para citar), "
        "`fetch` para ler adiante e `get_outline` para navegar a estrutura de um "
        "documento.\n\n"
        "A ingestão é um sistema à parte, que roda sozinho e não depende de você: "
        "converte o PDF, indexa e resume com um modelo local. A tool `ingest` é "
        "só um atalho para dispará-lo sem você precisar montar a linha de "
        "comando. Ela devolve um job_id na hora e o processamento segue em "
        "background — acompanhe com `job_status`. Não escreva resumos nem "
        "conteúdo para a base; isso é trabalho da ingestão.\n\n"
        "Se uma busca voltar com `incomplete: true`, há ingestão em andamento "
        "naquele escopo e os resultados cobrem só o que já foi indexado.\n\n"
        "Toda citação devolvida por `search` traz nome do arquivo e intervalo de "
        "páginas reais. Use-os; não invente referência."
    ),
)


# =========================================================================
# Busca e leitura
# =========================================================================


@server.tool()
def search(
    query: str,
    collections: list[str] | None = None,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    expand: Literal["section", "chunk"] = "section",
    rerank: bool = True,
) -> dict[str, Any]:
    """Busca passagens relevantes nos documentos indexados.

    Combina busca semântica e busca por palavra-chave (BM25), funde os dois
    rankings e reordena com um cross-encoder. Por padrão devolve a **seção
    inteira** em que o trecho relevante foi encontrado, não o fragmento isolado
    — é o que dá contexto suficiente para responder.

    Args:
        query: A pergunta ou tema, em linguagem natural.
        collections: Restringe a busca a estas collections. Omita para buscar
            em toda a base. Use `list_collections` para ver as disponíveis.
        doc_ids: Restringe a documentos específicos.
        top_k: Quantas passagens devolver (padrão 5).
        expand: "section" devolve a seção-pai (padrão); "chunk" devolve só o
            trecho exato, útil para avaliar a precisão da recuperação.
        rerank: Reordenação por cross-encoder. Desligue apenas para comparar.

    Returns:
        Passagens com texto, arquivo de origem, intervalo de páginas e o caminho
        de títulos até a seção.
    """
    from kb.jobs import active_jobs
    from kb.retrieval import search as run_search

    results = run_search(
        query,
        collections=collections,
        doc_ids=doc_ids,
        top_k=top_k,
        expand=expand,
        use_rerank=rerank,
    )

    payload: dict[str, Any] = {
        "query": query,
        "scope": collections or doc_ids or "toda a base",
        "n_results": len(results),
        "results": results,
    }

    # Buscar numa collection que está sendo ingerida devolve só o que já foi
    # embedado. Sem este aviso a resposta parece completa e não é.
    running = active_jobs(collections)
    if running:
        payload["incomplete"] = True
        payload["warning"] = (
            f"{len(running)} ingestão(ões) em andamento neste escopo — estes "
            "resultados cobrem apenas o que já foi indexado. Acompanhe com "
            "job_status e repita a busca ao final se a resposta parecer incompleta."
        )
        payload["jobs_running"] = [
            {"job_id": job["job_id"], "progress": f"{job['progress']}/{job['total']}"}
            for job in running
        ]

    if not results:
        payload["note"] = (
            "Nenhuma passagem encontrada. Verifique o escopo com list_collections."
        )

    return payload


@server.tool()
def fetch(
    section_id: str | None = None,
    doc_id: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    """Lê um trecho específico da base, sem busca semântica.

    Use depois de um `search` para ler adiante, ou para conferir uma citação.
    Informe `section_id` (vindo de um resultado de busca) **ou** `doc_id` com o
    intervalo de páginas.

    Args:
        section_id: Id da seção, como devolvido por `search` ou `get_outline`.
        doc_id: Id do documento, para leitura por página.
        page_start: Primeira página (obrigatório com doc_id).
        page_end: Última página; omita para ler uma página só.
    """
    from kb.retrieval import fetch_pages, fetch_section

    if section_id:
        section = fetch_section(section_id)
        return section or {"error": f"seção não encontrada: {section_id}"}

    if doc_id and page_start:
        return fetch_pages(doc_id, page_start, page_end)

    return {"error": "informe section_id, ou doc_id junto com page_start"}


@server.tool()
def get_outline(doc_id: str, max_depth: int = 3) -> dict[str, Any]:
    """Devolve o sumário navegável de um documento.

    A estrutura vem dos títulos que o parser extraiu do layout do PDF, com as
    páginas de cada seção. Serve para localizar o assunto antes de buscar, ou
    para escolher o que ler com `fetch`.

    Args:
        doc_id: Id do documento (veja `list_documents`).
        max_depth: Profundidade máxima de títulos aninhados.
    """
    from kb.retrieval import get_outline as run_outline

    return run_outline(doc_id, max_depth=max_depth)


# =========================================================================
# Descoberta
# =========================================================================


@server.tool()
def list_collections() -> dict[str, Any]:
    """Lista as collections existentes, com quantos documentos e chunks cada uma tem.

    Uma collection é um recorte independente de conhecimento. Chame isto antes
    de buscar quando não souber onde o assunto está.
    """
    from kb.collection import list_all

    collections = list_all()

    return {
        "n_collections": len(collections),
        "collections": collections,
        "note": None if collections else "Nenhuma collection ainda. Crie uma com manage_collection.",
    }


@server.tool()
def list_documents(collection: str | None = None) -> dict[str, Any]:
    """Lista os documentos indexados, com páginas, status e resumo quando houver.

    Args:
        collection: Filtra por collection. Omita para listar tudo.
    """
    from kb.collection import documents_in

    documents = documents_in(collection)

    return {
        "collection": collection,
        "n_documents": len(documents),
        "documents": [
            {
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "n_pages": document["n_pages"],
                "n_chunks": document["n_chunks"],
                "status": document["status"],
                "title": document["title"],
                "summary": document["summary"],
                "key_topics": document["key_topics"],
            }
            for document in documents
        ],
    }


# =========================================================================
# Auto-gerenciamento
# =========================================================================


@server.tool()
def ingest(
    paths: list[str],
    collection: str | None = None,
    collection_name: str | None = None,
    force: bool = False,
    enrich: bool = True,
) -> dict[str, Any]:
    """Indexa PDFs a partir de um caminho de arquivo ou pasta.

    Retorna imediatamente com um `job_id`: o processamento roda em background e
    pode levar minutos por livro. Acompanhe com `job_status`.

    Documentos já indexados são pulados automaticamente, então re-executar sobre
    a mesma pasta é barato e seguro.

    Por segurança, só aceita caminhos dentro das pastas autorizadas na
    configuração do servidor; um caminho fora disso é recusado.

    Args:
        paths: Arquivos .pdf ou pastas contendo PDFs.
        collection: Slug da collection de destino. Criada se não existir.
        collection_name: Nome legível, usado apenas na criação.
        force: Reprocessa mesmo o que já está indexado.
        enrich: Gera resumos com o LLM local ao final. Desligue para uma
            ingestão mais rápida; os resumos podem ser feitos depois.
    """
    from kb import collection as collections_api
    from kb.jobs import PathNotAllowed, spawn_ingest

    try:
        if collection:
            collections_api.create(
                collection_name or collection,
                collection_id=collection,
            )

        return spawn_ingest(paths, collection, force=force, enrich=enrich)

    except PathNotAllowed as exc:
        return {"error": "caminho não autorizado", "detail": str(exc)}
    except FileNotFoundError as exc:
        return {"error": "caminho não encontrado", "detail": str(exc)}


@server.tool()
def job_status(job_id: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Consulta o progresso de uma ingestão.

    Args:
        job_id: Id devolvido por `ingest`. Omita para listar os jobs recentes.
        limit: Quantos jobs listar quando job_id não for informado.
    """
    from kb.jobs import get_job, list_jobs

    if job_id:
        job = get_job(job_id)
        return job or {"error": f"job não encontrado: {job_id}"}

    return {"jobs": list_jobs(limit)}


@server.tool()
def manage_collection(
    action: Literal["create", "update", "delete", "add_docs", "remove_docs"],
    collection: str,
    name: str | None = None,
    description: str | None = None,
    doc_ids: list[str] | None = None,
    delete_documents: bool = False,
) -> dict[str, Any]:
    """Cria, edita ou reorganiza collections.

    Um documento pode pertencer a várias collections ao mesmo tempo, sem
    duplicar armazenamento — mover ou compartilhar é barato.

    Args:
        action: A operação desejada.
        collection: Slug da collection.
        name: Nome legível (create/update).
        description: Descrição (create/update).
        doc_ids: Documentos afetados (add_docs/remove_docs).
        delete_documents: Em `delete`, também apaga os documentos que ficariam
            sem nenhuma collection. Por padrão eles são preservados.
    """
    from kb import collection as api

    if action == "create":
        return api.create(name or collection, description, collection_id=collection)

    if action == "update":
        result = api.update(collection, name, description)
        return result or {"error": f"collection não encontrada: {collection}"}

    if action == "delete":
        return api.delete(collection, delete_documents=delete_documents)

    if not doc_ids:
        return {"error": f"{action} exige doc_ids"}

    if action == "add_docs":
        return {"added": api.add_documents(collection, doc_ids)}

    return {"removed": api.remove_documents(collection, doc_ids)}


@server.tool()
def status(collection: str | None = None) -> dict[str, Any]:
    """Retrato da base: quanto tem, se está íntegra e o que precisa de atenção.

    Reporta contagens, tamanho em disco, modelos em uso, estado da GPU, jobs
    recentes e — o mais importante — se o índice vetorial está em sincronia com
    o banco estrutural. Divergência ali é a falha silenciosa típica: a busca
    continua respondendo, mas o que falta nunca aparece.

    Args:
        collection: Detalha uma collection específica além do panorama geral.
    """
    from kb.health import status as run_status

    return run_status(collection)


# =========================================================================
# Enriquecimento delegado
# =========================================================================


@server.tool()
def delete_document(doc_id: str, purge_artifacts: bool = False) -> dict[str, Any]:
    """Remove um documento da base inteira.

    Diferente de `manage_collection` com `remove_docs`, que apenas tira o
    documento de uma collection e o deixa na base — ainda aparecendo em buscas
    sem escopo. Aqui somem o texto, a estrutura e os vetores.

    Args:
        doc_id: Id do documento (veja `list_documents`).
        purge_artifacts: Também apaga o resultado do parse guardado em disco.
            Por padrão ele é preservado, porque é a parte cara de reconstruir e
            torna uma reingestão futura quase instantânea.
    """
    from kb.collection import delete_document as run_delete

    return run_delete(doc_id, purge_artifacts=purge_artifacts)


@server.tool()
def reindex(collection: str | None = None, reset: bool = False) -> dict[str, Any]:
    """Recompõe o índice vetorial a partir do banco estrutural.

    Use quando `status` apontar divergência entre o banco e o índice. Não relê
    nenhum PDF: o texto já está guardado, só os vetores são recalculados.

    Args:
        collection: Limita a uma collection. Omita para a base toda.
        reset: Invalida todos os vetores antes de recomeçar. Necessário ao
            trocar de modelo de embedding; desnecessário para apenas preencher
            o que falta.
    """
    from kb.collection import resolve_scope
    from kb.ingest.embed import embed_pending, reset_embeddings

    doc_ids = resolve_scope([collection] if collection else None)

    if doc_ids is not None and not doc_ids:
        return {"error": f"collection sem documentos: {collection}"}

    invalidated = reset_embeddings(doc_ids) if reset else 0
    embedded = embed_pending(doc_ids=doc_ids)

    from kb.health import consistency

    return {
        "scope": collection or "toda a base",
        "invalidated": invalidated,
        "embedded": embedded,
        "consistency": consistency(),
    }


def main() -> None:
    # init_db, carga e descarga dos modelos ficam no `lifespan`.
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
