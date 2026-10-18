# RAG local

Este projeto ingere PDFs e recupera passagens citáveis. **Recuperação** é a busca
de trechos relevantes na base, seguida da leitura do contexto e da origem
(arquivo, seção e páginas). Ela não envolve conversar com um modelo de IA: uma
pessoa, um script ou outra aplicação pode usar a mesma CLI.

O SQLite guarda documentos, seções, chunks, collections, jobs e busca lexical
(FTS5). O Chroma guarda somente vetores derivados. Artefatos de parse ficam em
`store/artifacts/` e permitem retomar uma ingestão sem converter o PDF de novo.
`store/` não é versionado.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:8b
```

O Ollama só é usado na geração opcional de resumos. Sem ele, os passes de parse e
embedding ainda concluem e os resumos ficam pendentes. Os pesos de embedding e
reranking são baixados no primeiro `on` correspondente.

## Sessões e recursos

Modelos só carregam após o comando manual `on` ou durante um job iniciado nessa
sessão. Há **uma sessão local por vez**:

```powershell
python -m kb.cli on --mode interaction  # carrega embedding e reranker
python -m kb.cli search "campo elétrico" --collection fisica-3
python -m kb.cli off                    # libera os modelos e encerra a sessão

python -m kb.cli on --mode ingestion    # prepara o parser
python -m kb.cli ingest "E:/Estudo/Fisica" --collection fisica-3
python -m kb.cli off
```

`switch --mode interaction|ingestion` troca explicitamente de modo. Busca requer
`interaction`; ingestão e reindexação requerem `ingestion`. Um comando no modo
errado ou com a sessão desligada devolve erro JSON com instrução para ligar ou
trocar de modo. `on` no modo já ativo é idempotente. A comunicação com o processo
local é privada e autenticada; não há servidor de rede público.

Uma ingestão executa parse, embedding e enriquecimento em sequência, liberando
um modelo antes de carregar o próximo. O parser volta a ficar pronto ao fim do
job. `ingest` espera o resultado por padrão; `--background` devolve um `job_id`.
`off` e `switch` aguardam o job ativo. Em um terminal interativo, oferecem
esperar ou cancelar; `--wait` dispensa a pergunta e `--cancel` cancela
imediatamente. Cancelar encerra a sessão; a ingestão pode ser retomada depois.
Para chamadas automatizadas, `off --no-wait` e `switch --no-wait` retornam
`BUSY` se houver job ativo, sem interrompê-lo nem aguardar sua conclusão.

## CLI

Saída padrão: JSON com `ok` e `data` ou `error` (`code`, `message`, `hint`).
`--human` ou `-H` exibe uma versão legível. `-h` mostra a ajuda. Consultas leves
funcionam mesmo com a sessão desligada e não importam PyTorch.

| Grupo | Comandos |
|---|---|
| Sessão | `on`, `off`, `switch`, `session` |
| Recuperação | `search`, `fetch`, `outline` |
| Ingestão | `ingest`, `reindex`, `jobs list/show/cancel` |
| Catálogo | `collections list/show/create/update/delete/add-docs/remove-docs`, `documents list/show/delete` |
| Diagnóstico | `status`, `doctor`, `models` |

Exemplos:

```powershell
python -m kb.cli collections list
python -m kb.cli documents list --collection fisica-3
python -m kb.cli fetch --section-id SECAO_ID
python -m kb.cli fetch --doc-id DOCUMENTO_ID --page-start 12 --page-end 14
python -m kb.cli jobs show JOB_ID
python -m kb.cli status
python -m kb.cli doctor
python -m kb.cli models
python -m kb.cli collections list --limit 50 --offset 0
python -m kb.cli documents list --limit 50 --offset 50
python -m kb.cli jobs list --limit 10 --offset 0
```

`status` consulta o driver da GPU sem carregar modelos. `doctor` faz a checagem
mais profunda entre SQLite e índice vetorial. A base e os artefatos existentes
permanecem no mesmo formato.
As listagens aceitam `--limit` e `--offset` para paginação. Sem essas opções,
collections e documentos continuam retornando a lista completa; jobs continuam
limitados aos 10 mais recentes. Itens inexistentes retornam `NOT_FOUND` no
envelope JSON da CLI.

## Adapters de modelos

O pipeline usa três contratos separados em `kb/providers.py`: `EmbeddingProvider`
(texto → vetores), `RerankProvider` (pergunta + trechos → scores) e
`GenerationProvider` (prompt → resumo, disponibilidade e descarga). Os adapters
atuais usam `sentence-transformers` para embedding/reranking e Ollama para
geração. O núcleo não conhece as chamadas específicas desses provedores.

| Variável | Padrão | Papel |
|---|---|---|
| `KB_EMBED_PROVIDER` | `sentence_transformers` | Adapter de embedding |
| `KB_EMBED_MODEL` | `BAAI/bge-m3` | Modelo de embedding |
| `KB_RERANK_PROVIDER` | `sentence_transformers` | Adapter de reranking |
| `KB_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Modelo de reranking |
| `KB_LLM_PROVIDER` | `ollama` | Adapter de geração |
| `KB_LLM_MODEL` | `qwen3:8b` | Modelo de resumos |
| `OLLAMA_URL` | `http://localhost:11434` | Serviço Ollama |
| `KB_DEVICE` | `cuda` | Dispositivo dos adapters locais |
| `KB_CHUNK_TOKENIZER_MODEL` | `BAAI/bge-m3` | Tokenizer usado no corte dos documentos |

Um provider não implementado produz erro explícito; nunca há fallback silencioso
para outro serviço. O próximo adapter pode falar com `llama.cpp` sem alterar
ingestão ou recuperação. Trocar o modelo ou provider de geração não exige
reindexação. Trocar o embedding exige `reindex --reset` no modo de ingestão,
pois vetores produzidos por modelos diferentes não são intercambiáveis.
O tokenizer de chunking é configurado à parte; se mudar o corte dos documentos,
é preciso reingerir os PDFs afetados para gerar novos chunks.

`KB_INGEST_ROOTS` limita as pastas aceitas por `ingest` (separadas pelo
separador de caminhos do sistema). Por padrão, `E:/Estudo` e a raiz do projeto
são permitidas. Consulte `kb/config.py` para os demais ajustes de chunking e
busca.
