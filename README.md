# Knowledge Base — ETL e retrieval para agentes

Sistema de ampliação de conhecimento estruturado. Ingere PDFs grandes, armazena
estrutura real (documento → seções → chunks) e expõe busca híbrida como
**servidor MCP**, para que qualquer agente — Claude Code, Codex, scripts
próprios — consulte a base sem saber nada de Docling, embeddings ou vetores.

Não há LLM nesta máquina. Recuperação roda local na GPU; a geração é do agente.

## Arquitetura

Três camadas, com papéis separados de propósito:

```
store/
├── kb.sqlite3            ESTRUTURA — a verdade do sistema. Documentos, seções,
│                         chunks, collections, jobs, e BM25 via FTS5.
├── artifacts/{doc_id}/   ARTEFATOS — resultado imutável do parse.
│   ├── manifest.json         Reindexar nunca mais roda o Docling.
│   ├── blocks/*.json
│   └── document.md
└── chroma/               ÍNDICE VETORIAL — descartável, reconstruível a
                          partir do SQLite a qualquer momento.
```

O índice vetorial é deliberadamente burro: guarda `chunk_id`, vetor e `doc_id`.
Todo escopo (collections, filtros) é resolvido no SQLite **antes** da busca
vetorial. É isso que permite trocar o Chroma por outro backend sem tocar no
retrieval, e que faz um documento pertencer a várias collections sem duplicar
nada.

### Small-to-big

Recupera-se pelo **chunk** (~500 tokens, preciso para achar) e entrega-se a
**seção-pai** (grande e coerente, adequada para raciocinar). A fragmentação que
torna RAG inútil em livro-texto some sem custo extra de indexação.

## Os dois pipelines

```
INGESTÃO  (subprocesso, um modelo por vez na VRAM, retomável)
  Pass 1  PARSE   Docling  → artifacts/          descarrega ao fim
  Pass 2  EMBED   bge-m3   → índice vetorial     descarrega ao fim

RETRIEVAL (no servidor MCP, por query)
  1. Escopo    collections → doc_ids             SQLite       sem modelo
  2. Recall    ├─ vetorial  top-40               Chroma       bge-m3
               └─ léxico    top-40 bm25()        FTS5         sem modelo
  3. Fusão     RRF → top-20                      memória      sem modelo
  4. Rerank    top-20 → top-5                    cross-encoder
  5. Expansão  chunk → seção-pai                 SQLite       sem modelo
  6. Citação   {arquivo, títulos, páginas}       SQLite       sem modelo
```

A ingestão roda em **processo separado** porque o Docling carrega modelos de
layout na GPU: no mesmo processo, disputaria VRAM com o embedder e o reranker.
Ao terminar, o processo morre e a GPU volta limpa por construção.

## Modelos

| Papel | Modelo | VRAM (fp16) | Quando |
|---|---|---|---|
| Parse | Docling (layout + TableFormer) | ~1–2 GB | subprocesso, Pass 1 |
| Embedding | `BAAI/bge-m3` — 1024d, 8192 ctx | ~1,2 GB | Pass 2 e query |
| Rerank | `BAAI/bge-reranker-v2-m3` | ~1,2 GB | query |

O servidor carrega modelo sob demanda e descarrega por inatividade
(`KB_MODEL_IDLE_TIMEOUT`, padrão 600 s).

## Instalação

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Os modelos baixam sozinhos no primeiro uso (~6 GB em `~/.cache/huggingface`).

## Uso — CLI

```bash
python -m kb.cli ingest "E:/Estudo/Fisica" --collection fisica-3
python -m kb.cli search "campo elétrico de um dipolo" --collection fisica-3
python -m kb.cli status
python -m kb.cli collections
python -m kb.cli outline <doc_id>
python -m kb.cli reindex          # recompõe vetores sem reprocessar PDF
```

## Uso — MCP

```bash
claude mcp add knowledge-base -- E:/Rag-Project/.venv/Scripts/python.exe E:/Rag-Project/mcp_server.py
```

Onze tools, agrupadas por intenção:

| Grupo | Tools |
|---|---|
| Busca e leitura | `search` · `fetch` · `get_outline` |
| Descoberta | `list_collections` · `list_documents` |
| Auto-gerenciamento | `ingest` · `job_status` · `manage_collection` · `status` |
| Enriquecimento | `get_pending_enrichment` · `submit_enrichment` |

O agente se vira sozinho: descobre o que existe, dispara ingestão a partir de um
caminho que o usuário mencionou na conversa, acompanha o progresso e inspeciona
a saúde da base — tudo sem sair do chat.

### Collections

Recortes independentes de conhecimento. Um documento pode estar em várias sem
duplicar armazenamento. `search` sem escopo busca em tudo; com
`collections=["fisica-3"]` isola a prateleira.

### Enriquecimento delegado

Resumos são escritos pelo **agente**, não por um LLM local: `get_pending_enrichment`
devolve o material, `submit_enrichment` grava o resultado, e o estado fica no
SQLite — o laço pode parar e retomar. A navegação (`get_outline`) não depende
disso: sai dos títulos que o Docling extraiu do layout, de graça.

## Configuração

Tudo em `kb/config.py`, sobrescrevível por variável de ambiente:

| Variável | Padrão | Para quê |
|---|---|---|
| `KB_INGEST_ROOTS` | `E:\Estudo`, raiz do projeto | Pastas de onde a ingestão pode ler |
| `KB_EMBED_MODEL` | `BAAI/bge-m3` | Trocar exige `reindex --reset` |
| `KB_RERANK` | `1` | `0` desliga o rerank |
| `KB_CHUNK_MAX_TOKENS` | `512` | Teto do chunk |
| `KB_MODEL_IDLE_TIMEOUT` | `600` | Segundos até descarregar da VRAM |
| `KB_DEVICE` | `cuda` | `cpu` para rodar sem GPU |

`KB_INGEST_ROOTS` é uma guarda de segurança: a ingestão é disparada por um
agente, que pode ser induzido pelo conteúdo de um documento a indexar arquivos
arbitrários do disco. Caminhos fora das raízes são recusados.

## Segurança e limites conhecidos

- O filtro por `doc_id` no Chroma usa `$in`; uma collection com milhares de
  documentos degrada a consulta. É o gatilho para migrar o índice para LanceDB —
  a interface `VectorIndex` em `kb/vectors.py` já isola essa troca.
- `status` reporta divergência entre SQLite e índice vetorial. É a falha
  silenciosa típica: a busca continua respondendo e o que falta simplesmente
  nunca aparece.
