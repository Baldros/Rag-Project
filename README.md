# RAG local

Este projeto ingere PDFs e recupera passagens citáveis. **Recuperação** é a busca
de trechos relevantes na base, seguida da leitura do contexto e da origem
(arquivo, seção e páginas). Ela não envolve conversar com um modelo de IA: uma
pessoa, um script ou outra aplicação pode usar a mesma CLI.

O SQLite guarda documentos, seções, chunks, collections, jobs e busca lexical
(FTS5). O Chroma guarda somente vetores derivados. Artefatos de parse ficam em
`store/artifacts/` e permitem retomar uma ingestão sem converter o PDF de novo.
`store/` não é versionado.

O esquema do SQLite é versionado em `PRAGMA user_version`. Abrir o banco migra
uma versão anterior e recusa, com `SCHEMA_TOO_NEW`, um banco gravado por código
mais novo — em vez de rebaixar a marca e tratar como antigo um banco que já tem
as mudanças.

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

## Extração e qualidade

O parser usa Docling 2.145.0, com fórmulas sempre habilitadas pelo modelo local
CodeFormulaV2. `--no-enrich` desliga somente os resumos do LLM. O parâmetro
`ingest --ocr auto|off|full` controla OCR: automático por padrão, desligado para
PDFs com texto confiável, ou aplicado a todas as páginas. `KB_OCR` define o
padrão quando a opção é omitida. Caminhos Unicode são abertos pelo Python e
entregues ao Docling como `DocumentStream`, sem copiar ou renomear o PDF.

A [documentação oficial de enriquecimento](https://docling-project.github.io/docling/usage/enrichments/)
descreve o custo adicional de fórmulas e a configuração da margem dos recortes.
Usamos margem de 8%, para reduzir a inclusão de texto vizinho, e interrupção de
repetições. O [CodeFormulaV2](https://huggingface.co/docling-project/CodeFormulaV2)
trabalha com imagens a 120 DPI. Falhas detectáveis recebem uma nova tentativa
usando os recortes e o layout salvo, sem reconverter o PDF inteiro.

`quality.status` informa `complete`, `partial` ou `unknown`. `complete` confirma
a cobertura da extração e a ausência de falhas detectadas; **não certifica a
correção matemática** da transcrição. Fórmulas geradas têm
`formula_verification: automated_unverified`. Avisos, páginas e referências dos
problemas aparecem no catálogo, `outline`, `search` e `fetch`. Os documentos
antigos continuam consultáveis e têm qualidade `unknown` até nova ingestão.

Artefatos são salvos em gerações imutáveis com hashes e cobertura exata de
páginas. Todos os blocos precisam converter com sucesso; uma falha mantém a
versão anterior publicada e deixa um checkpoint para retomada. A geração usada
pelo documento é registrada no SQLite junto com os novos chunks. Limpeza de
vetores antigos fica marcada no banco até terminar, antes dos novos embeddings.

Conversão, fórmulas e segmentação têm assinaturas separadas. Alterar o tamanho
de chunk reutiliza o parse; mudar a configuração de fórmulas reutiliza o layout
compatível. O chunker recebe o documento concatenado para preservar os títulos
entre blocos de 15 páginas. A busca lexical usa uma cópia normalizada dos acentos
separados pelo TeX; o texto original e as fórmulas das citações são preservados.

Jobs terminam como `done`, `partial`, `failed` ou `cancelled`. `partial` significa
que há dados disponíveis com falhas detectadas na extração, ou resumos solicitados
que ficaram pendentes. `jobs show JOB_ID` devolve o resultado estruturado completo,
incluindo erros e tempos por pass. Na espera síncrona, `partial` retorna
`JOB_PARTIAL` com instrução para consultar o job.

O esquema SQLite 2 migra o FTS numa transação, mantendo fontes, resumos,
collections e marcações de embedding. Antes de atualizar uma base existente,
desligue a sessão e faça backup de SQLite, Chroma e artefatos.

## CLI

Saída padrão: JSON com `ok` e `data` ou `error` (`code`, `message`, `hint`).
As listagens acrescentam `page` (`limit`, `offset`, `has_more`) ao lado de
`data`. `--human` ou `-H` exibe uma versão legível. `-h` descreve cada
argumento. Consultas leves funcionam mesmo com a sessão desligada e não
importam PyTorch.

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
mais profunda: compara os IDs dos vetores com os chunks, verifica limpezas
pendentes e valida hashes e cobertura dos artefatos. Artefatos antigos aparecem
como `legacy`, sem serem apagados ou reprocessados automaticamente.
As listagens aceitam `--limit` e `--offset` para paginação, e `page.has_more`
diz se a página esgotou a lista. Sem essas opções, collections e documentos
continuam retornando a lista completa; jobs continuam limitados aos 10 mais
recentes. Item inexistente retorna `NOT_FOUND` no envelope JSON da CLI, e isso
vale para a collection pedida em `search`, `documents list` e
`collections add-docs`: id errado é erro, não resultado vazio.

Resposta cortada diz que foi cortada. A passagem devolvida por `search` traz
`truncated` quando a seção passou de `KB_MAX_SECTION_CHARS` — leia-a inteira
com `fetch --section-id` —, e `fetch` por páginas traz `truncated` e
`next_page_start` quando o intervalo passou de `KB_FETCH_MAX_CHARS`. Esse corte
só acontece ao virar de página, então ler em sequência sempre avança.

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
