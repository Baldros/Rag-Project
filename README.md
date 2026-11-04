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

Modelos só carregam após o comando manual `on` e só saem com `off` ou `switch`.
Há **uma sessão local por vez**:

```powershell
python -m kb.cli on --mode interaction  # carrega embedding e reranker
python -m kb.cli search "campo elétrico" --collection fisica-3
python -m kb.cli off                    # libera os modelos e encerra a sessão

python -m kb.cli on --mode ingestion    # carrega parser, tokenizer e embedding
python -m kb.cli ingest "E:/Estudo/Fisica" --collection fisica-3
python -m kb.cli off
```

`switch --mode interaction|ingestion` troca explicitamente de modo. Busca requer
`interaction`; ingestão e reindexação requerem `ingestion`. Um comando no modo
errado ou com a sessão desligada devolve erro JSON com instrução para ligar ou
trocar de modo. `on` no modo já ativo é idempotente. A comunicação com o processo
local é privada e autenticada; não há servidor de rede público.

`on` e `switch` não têm prazo: a ativação termina quando o processo responde,
quando morre ou quando `off` a cancela (`START_CANCELLED`). Em disco lento ela pode
levar vários minutos. Cada tentativa grava suas etapas em
`store/startup/<id>.jsonl`, e `session` devolve `starting` com esse id enquanto ela
estiver em andamento. Uma instalação quebrada aparece aqui, como `START_FAILED`
com a etapa e o erro. Avisos que não impedem o uso, como Ollama fora do ar, ficam no
mesmo arquivo. As últimas 20 tentativas são mantidas.

Uma ingestão executa parse, embedding e enriquecimento em sequência, sobre os
modelos que a ativação carregou. O job não os descarrega: **liberar a GPU exige
`off`**. A exceção é o modelo de resumos, que roda no Ollama, entra só no passe de
enriquecimento e sai ao fim dele. Um lote com OCR diferente do carregado recria o
parser. A limpeza do job termina antes de publicar o estado terminal. A sessão
permanece no modo escolhido; não há troca automática, desligamento automático ou
prazo de inatividade.
Em `interaction`, embedding e reranker continuam carregados entre perguntas,
até `off` ou `switch` explícito. `ingest` espera o resultado por padrão;
`--background` devolve um `job_id` e o processo continua independente do cliente.
`off` e `switch` aguardam o job ativo. Em um terminal interativo, oferecem
esperar ou cancelar; `--wait` dispensa a pergunta e `--cancel` cancela
imediatamente. Cancelar encerra a sessão; a ingestão pode ser retomada depois.
Para chamadas automatizadas, `off --no-wait` e `switch --no-wait` retornam
`BUSY` se houver job ativo, sem interrompê-lo nem aguardar sua conclusão.
Admissão de trabalho e transições de modo usam o mesmo lock entre clientes.
`jobs cancel JOB_ID` cancela somente esse job: se já terminou ou foi substituído,
retorna `JOB_NOT_ACTIVE`. A intenção de cancelar é gravada antes de encerrar o
processo, e o estado `cancelled` é publicado após a tentativa de limpeza externa.

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

### Corrigir uma fórmula à mão

`kb formulas apply --doc-id ID --file correcoes.json` aplica LaTeX escrito por uma
pessoa, em modo de ingestão e sem job em andamento. O arquivo é uma lista de
objetos com `ref`, `page`, `expected` (o LaTeX que a pessoa estava vendo) e `text`;
`text: null` restaura o que o modelo escreveu. Uma fórmula que mudou desde a edição
é recusada e listada em `rejected`, e o job termina como `partial`.

As correções ficam no manifest (`corrections`), sem alterar o JSON do bloco, que
guarda o que o modelo escreveu. A aplicação publica uma nova geração junto com os
novos chunks e refaz os vetores do documento. Uma nova conversão, mesmo com
`--force`, reaplica as correções. Se a fórmula não estiver mais no mesmo lugar, a
correção não é aplicada e vira um problema de qualidade, com o texto preservado. A
fórmula corrigida deixa de ser sinalizada e entra em `formula_corrected`.

O resumo do documento é mantido, porque lê só o sumário e os seis primeiros chunks.
`summary_affected` diz se a correção mudou esse trecho, e `kb summarize --doc-id ID`
refaz o resumo. O anterior só é substituído quando o novo é gerado.

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

`indexed` no catálogo significa que os vetores estão disponíveis; o enriquecimento
ainda pode estar em andamento. O fim da operação é dado pelo **job**, não pelo
catálogo. `running`, `cancelling` e `finalizing` ainda são estados ativos.
`result.pending` detalha pendências de extração, vetores e resumos solicitados;
`prepared` identifica documentos já publicados e `pending_inputs` lista arquivos
ainda não processados no último checkpoint. Checkpoints são gravados por documento
e entre fases; após queda, podem anteceder as últimas gravações no catálogo.

Reingerir o mesmo PDF de qualidade `partial`, **mesmo sem `--force`**, mantém o
comportamento de tentar melhorar a extração: reutiliza blocos compatíveis, tenta
fórmulas pendentes e pode refazer segmentação e embeddings. Isso pode ser demorado.
`--force` também ignora o cache de parse. `--no-enrich` só pula os resumos, não as
tentativas de extração. Para apenas usar o material já disponível, consulte-o;
uma nova ingestão é uma nova execução explícita. Falhas de parse antes da publicação
preservam a versão anterior e seus resumos. Qualidade parcial continua visível nos resultados.

Jobs novos registram PID e identidade de criação do processo antes de retornar
`queued`, protegendo contra reutilização de PID no Windows e Linux. Na inicialização
do runtime e nas leituras de jobs, uma única rotina recupera órfãos como `failed`,
ou `cancelled` quando a intenção de cancelar foi persistida. Não há retomada
automática. `jobs list`, `jobs show`, `status` e a indicação de trabalho ativo usam
essa recuperação. Falha de comunicação ou permissão não prova que o processo morreu.
Registros antigos sem identidade de criação só permitem detectar a ausência do PID.

`finished_at` e o resultado terminal são duráveis e não podem ser sobrescritos por
um worker atrasado. `cleanup_status` informa `done`, `failed` ou `unconfirmed`;
`pending` indica que a finalização ainda não ocorreu. Falha de limpeza em execução
normal termina o job como `failed`, preservando seu resultado. Após queda, recursos
externos como o Ollama podem continuar residentes: a recuperação registra
`unconfirmed` e não descarrega modelos de outra sessão. O controle do Ollama não
isola aplicações externas que usem o mesmo modelo. Não foi adicionado mecanismo de
notificação; o resultado continua consultável pelo `job_id`.

O esquema SQLite 3 acrescenta identidade, cancelamento e limpeza dos jobs; a
migração da versão 1 também atualiza o FTS numa transação, mantendo fontes, resumos,
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
| Ingestão | `ingest`, `reindex`, `formulas apply`, `summarize`, `jobs list/show/cancel` |
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
