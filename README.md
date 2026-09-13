# Knowledge Base — ETL e retrieval para agentes

Sistema de ampliação de conhecimento estruturado. Ingere PDFs grandes, armazena
estrutura real (documento → seções → chunks) e expõe busca híbrida como
**servidor MCP**, para que qualquer agente — Claude Code, Codex, scripts
próprios — consulte a base sem saber nada de Docling, embeddings ou vetores.

## Dois sistemas independentes

Esta é a distinção que organiza o projeto inteiro:

| | **Ingestão** | **Retrieval** |
|---|---|---|
| O que faz | PDF → base de conhecimento | base → passagens citáveis |
| Modelos | Docling → bge-m3 → LLM local | bge-m3 + reranker |
| Superfície | **CLI** | **MCP** |
| Depende de agente? | **nunca** | sim, o agente é o consumidor |
| VRAM | um modelo por vez, sequencial | os dois residentes, ~2,2 GB |

**A ingestão é completa em si.** Roda por linha de comando, sem servidor MCP no
ar e sem nenhum agente conectado, e produz a base pronta — inclusive os resumos,
feitos por um LLM local. Nada nela depende da camada de consumo.

**O MCP é onde o retrieval acontece**, e é só isso que ele precisa ser. Ele
também expõe uma tool `ingest`, mas por conveniência: o agente dispara o mesmo
processo que você dispararia no terminal, sem gastar contexto lendo o
repositório para descobrir como. Capacidade nenhuma nova, só previsibilidade.

A ordem de uso é sequencial: primeiro o material entra e é processado; só então
a base está disponível para consulta. Uma busca feita enquanto a ingestão daquele
escopo ainda roda volta marcada com `incomplete: true`.

## Arquitetura de armazenamento

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
Todo escopo é resolvido no SQLite **antes** da busca vetorial. É isso que permite
trocar o Chroma por outro backend sem tocar no retrieval, e que faz um documento
pertencer a várias collections sem duplicar nada.

**Small-to-big:** recupera-se pelo chunk (~500 tokens, preciso para achar) e
entrega-se a seção-pai (grande e coerente, adequada para raciocinar).

## Os pipelines

```
INGESTÃO  (CLI ou subprocesso; um modelo por vez, retomável)
  Pass 1  PARSE    Docling   → artifacts/          descarrega
  Pass 2  EMBED    bge-m3    → índice vetorial     descarrega
  Pass 3  ENRICH   LLM local → resumos no SQLite   descarrega

RETRIEVAL (no servidor MCP, por query)
  1. Escopo    collections → doc_ids             SQLite       sem modelo
  2. Recall    ├─ vetorial  top-40               Chroma       bge-m3
               └─ léxico    top-40 bm25()        FTS5         sem modelo
  3. Fusão     RRF → top-20                      memória      sem modelo
  4. Rerank    top-20 → top-5                    cross-encoder
  5. Expansão  chunk → seção-pai                 SQLite       sem modelo
  6. Citação   {arquivo, títulos, páginas}       SQLite       sem modelo
```

Cada pass da ingestão carrega seu modelo, faz todo o trabalho do lote e
descarrega antes do próximo — nunca dois na GPU ao mesmo tempo. No retrieval os
dois modelos são pequenos e coexistem sem disputa.

## Modelos

| Papel | Modelo | VRAM | Quando |
|---|---|---|---|
| Parse | Docling (layout + TableFormer) | ~1–2 GB | ingestão, Pass 1 |
| Embedding | `BAAI/bge-m3` — 1024d, 8192 ctx | ~1,2 GB | Pass 2 e query |
| Resumo | `qwen3:8b` via Ollama | ~5 GB | ingestão, Pass 3 |
| Rerank | `BAAI/bge-reranker-v2-m3` | ~1,2 GB | query |

O LLM faz apenas processamento de texto — não precisa de tool calling nem de
contexto longo. A escolha ainda não está fechada; `KB_LLM_MODEL` troca em uma
linha. No servidor MCP, embedding e reranker sobem no startup e caem no
shutdown.

## Instalação

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
ollama pull qwen3:8b
```

Os modelos de recuperação baixam sozinhos no primeiro uso (~6 GB em
`~/.cache/huggingface`). O Ollama é um serviço externo, necessário só para o
Pass 3: sem ele a ingestão conclui os Passes 1 e 2 e deixa os resumos pendentes,
sem falhar.

## Uso — CLI (o sistema de ingestão)

```bash
python -m kb.cli ingest "E:/Estudo/Fisica" --collection fisica-3
python -m kb.cli ingest <pdf> --collection x --no-enrich   # sem o Pass 3
python -m kb.cli search "campo elétrico de um dipolo" --collection fisica-3
python -m kb.cli status
python -m kb.cli reindex          # recompõe vetores sem reprocessar PDF
```

## Uso — MCP (o sistema de retrieval)

```bash
claude mcp add knowledge-base -- E:/Rag-Project/.venv/Scripts/python.exe E:/Rag-Project/mcp_server.py
```

| Grupo | Tools |
|---|---|
| Busca e leitura | `search` · `fetch` · `get_outline` |
| Descoberta | `list_collections` · `list_documents` |
| Auto-gerenciamento | `ingest` · `job_status` · `manage_collection` · `delete_document` · `reindex` · `status` |

As tools de auto-gerenciamento existem para o agente não precisar ler o
repositório nem montar linhas de comando — economizam contexto e tornam o
comportamento previsível.

### Collections

Recortes independentes de conhecimento. Um documento pode estar em várias sem
duplicar armazenamento. `search` sem escopo busca em tudo; com
`collections=["fisica-3"]` isola a prateleira.

## Configuração

Tudo em `kb/config.py`, sobrescrevível por variável de ambiente:

| Variável | Padrão | Para quê |
|---|---|---|
| `KB_INGEST_ROOTS` | `E:\Estudo`, raiz do projeto | Pastas de onde a ingestão pode ler |
| `KB_EMBED_MODEL` | `BAAI/bge-m3` | Trocar exige `reindex --reset` |
| `KB_LLM_MODEL` | `qwen3:8b` | LLM do Pass 3 |
| `KB_LLM` | `1` | `0` desliga o Pass 3 |
| `KB_ENRICH_SECTIONS` | `0` | `1` também resume seções (custa centenas de chamadas) |
| `KB_RERANK` | `1` | `0` desliga o rerank |
| `KB_CHUNK_MAX_TOKENS` | `512` | Teto do chunk |
| `KB_DEVICE` | `cuda` | `cpu` para rodar sem GPU |

`KB_INGEST_ROOTS` é uma guarda de segurança: a ingestão pode ser disparada por um
agente, que pode ser induzido pelo conteúdo de um documento a indexar arquivos
arbitrários do disco. Caminhos fora das raízes são recusados.

## Limites conhecidos

- **PDF escaneado degrada os títulos.** O OCR produz headings colados
  (`Potencialdeduplacamada`), o que afeta `heading_path` e `get_outline`.
- **Import lento nesta máquina.** `import torch` + `sentence_transformers` levam
  ~150 s por processo, o que domina o custo de subir qualquer coisa. O servidor
  MCP contorna carregando no startup; a ingestão paga uma vez por job. Vale
  investigar exclusão do antivírus para o `.venv`.
- **O filtro por `doc_id` no Chroma usa `$in`**; uma collection com milhares de
  documentos degrada a consulta. É o gatilho para migrar para LanceDB — a
  interface `VectorIndex` em `kb/vectors.py` já isola essa troca.
- **`status` reporta divergência** entre SQLite e índice vetorial. É a falha
  silenciosa típica: a busca continua respondendo e o que falta nunca aparece.
