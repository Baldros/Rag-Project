# Melhorias inspiradas no WhatsApp Reader

> **Outro projeto, usado só como referência.** O WhatsApp Reader (`E:\whatsapp-reader`) é um
> projeto separado: importa exports de conversas do WhatsApp para SQLite e os expõe por CLI,
> chat e MCP. Nada dele é dependência deste RAG, e este documento não propõe copiar código.
> Os trechos citados mostram de onde veio cada ideia e como ela ficou lá; a adaptação precisa
> seguir a arquitetura daqui.

Análise feita em 2026-10-23 sobre o branch `dev` deste repositório (commit `72f3777`) e sobre o
`E:\MCPs\RAG-MCP` (commit `da7d406`), que expõe esta CLI como servidor MCP. Nenhum código foi
alterado; os pontos marcados como "conferido" foram reproduzidos num store temporário.

## O que mudou no WhatsApp Reader

Commits do branch `windows-version` daquele repositório:

| Commit | Mudança |
| --- | --- |
| `e4c6012` | Mensagens idênticas enviadas no mesmo minuto eram descartadas na importação. O fingerprint passou a incluir o chat e a posição de cada repetição, com migração do banco versionada por `PRAGMA user_version`. |
| `71ba47f` | O MCP HTTP passou a responder 405 a `GET`/`DELETE`, os logs internos do SDK foram para WARNING e o chat passou a iniciar o MCP via `stdio`, em vez de HTTP numa porta aleatória. |
| `2bc8b7d` | As consultas de mensagens ganharam `order` e `has_more`, todos os parâmetros ganharam descrição no schema e os horários passaram a aceitar ISO 8601 com fuso. |
| `7901908` | A busca de texto passou a usar FTS5 com `unicode61 remove_diacritics 2`, como esquema 2 do banco. |

Uma decisão também entra na comparação: lá se avaliou e se descartou exigir token bearer no
MCP HTTP local. O custo de configuração não compensava numa máquina de um usuário só, em que
qualquer processo do usuário já consegue ler o arquivo SQLite diretamente.

## Resumo

| Ideia | Situação aqui | Recomendação |
| --- | --- | --- |
| Versionamento de esquema com migrações | Ausente; `init_db` regrava a versão como 1 | **Adotar.** Prioridade alta, esforço baixo |
| Palavras curtas na busca de texto | O índice guarda termos curtos, mas a consulta descarta os de até 2 caracteres | **Adotar.** Prioridade alta, esforço baixo |
| Avisar quando o resultado está incompleto | Listagens paginadas sem indicação de que há mais; `fetch` por páginas sem limite; corte de seção marcado só no texto | **Adotar.** Prioridade média, esforço baixo a médio |
| Contrato dos parâmetros | Collection inexistente devolve busca vazia com sucesso; `--top-k` aceita valor negativo; parâmetros do MCP sem descrição | **Adotar.** Prioridade média, esforço baixo; parte é no RAG-MCP |
| Posição no ID para não colapsar repetições | Já resolvido | Nada a fazer |
| FTS5 que ignora acentos e maiúsculas | Já existe, com a mesma configuração | Nada a fazer |
| `stdio` em vez de porta aleatória | Já resolvido com pipe nomeado e MCP via `stdio` | Nada a fazer |
| 405, logs do SDK, token bearer | Não há servidor HTTP aqui | Não se aplica |
| Horários ISO com fuso | Nenhum comando filtra por data | Não se aplica |

## 1. Versionamento do esquema com migrações

**Situação atual.** [`kb/db.py`](../kb/db.py) declara `SCHEMA_VERSION = 1` e, a cada
`init_db`, executa o `SCHEMA` com `CREATE ... IF NOT EXISTS` e grava
`PRAGMA user_version = 1` sem ler o valor anterior (linha 171). Não existe caminho de migração.

**Por que importa.**

- `CREATE TABLE IF NOT EXISTS` não altera uma tabela que já existe. A primeira coluna nova não
  aparece nos bancos já criados: o código falha com `no such column`, ou a saída é apagar
  `store/` e reprocessar todos os PDFs. É justamente o trabalho caro (Docling e embeddings) que
  a ingestão em passes existe para preservar.
- A versão é sobrescrita, não lida. **Conferido:** um banco marcado como versão 2 volta a ser 1
  quando este código o abre. Quando existir uma versão 2, qualquer processo com código antigo
  (uma sessão aberta antes da atualização, outro checkout) rebaixa a marca, e o código novo
  passa a enxergar um banco "antigo" que já tem as mudanças.

**De onde veio.** No WhatsApp Reader, corrigir a deduplicação mudou o formato de um dado já
gravado (o fingerprint), e isso obrigou a criar o mecanismo (`e4c6012`). A busca com FTS5
(`7901908`) entrou depois como esquema 2, no mesmo caminho. Trecho de
`E:\whatsapp-reader\whatsapp_reader\db.py`:

```python
# Stored in PRAGMA user_version.
# 1: fingerprints include the chat and the occurrence of identical messages.
# 2: messages_fts, a full-text index of message text that ignores case and accents.
SCHEMA_VERSION = 2


def _migrate_schema(connection: sqlite3.Connection) -> None:
    if _schema_version(connection) >= SCHEMA_VERSION:
        return

    connection.execute("BEGIN IMMEDIATE")
    try:
        # Read again under the write lock: another process may have just migrated.
        version = _schema_version(connection)
        if version < 1:
            _rebuild_fingerprints(connection)
        if version < 2:
            _create_text_index(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
```

**Como adaptar aqui.**

- Ler `user_version` antes de gravar e só subir o número, nunca descer.
- Um banco mais novo que o código deveria gerar um erro claro no envelope da CLI (por exemplo
  `SCHEMA_TOO_NEW`) em vez de ser aberto. Lá esse caso é apenas ignorado; aqui há uma sessão de
  vida longa e jobs gravando no banco, o que torna o risco maior.
- O `BEGIN IMMEDIATE` com nova leitura da versão importa também aqui: a sessão (`kb.runtime`) e
  a CLI abrem o mesmo banco em processos diferentes.
- A conexão daqui roda em autocommit (`isolation_level=None`), então o `BEGIN IMMEDIATE`
  explícito funciona diretamente. Só não use `executescript` no meio da migração: ele faz
  `COMMIT` da transação aberta antes de executar.
- O esquema de hoje passa a ser a versão 1. Nenhuma migração é necessária agora; o ganho é ter
  o mecanismo pronto antes da primeira mudança de schema.
- Testes no estilo dos que já existem: abrir um banco marcado com versão maior e esperar o erro;
  abrir um banco da versão anterior e conferir o resultado da migração.

## 2. Palavras curtas na busca léxica

**Situação atual.** `fts_query`, em [`kb/retrieval/search.py`](../kb/retrieval/search.py),
monta a consulta BM25 apenas com termos de mais de 2 caracteres (linha 92). O índice
`chunks_fts` guarda esses termos normalmente; quem os descarta é a montagem da consulta.
**Conferido:**

| Pergunta | Consulta BM25 gerada |
| --- | --- |
| `pH da solução tampão` | `"solucao" OR "tampao"` |
| `lei de Ohm em CA e CC` | `"lei" OR "ohm"` |
| `raios UV e luz IR` | `"raios" OR "luz"` |
| `5G vs 4G` | vazia, e o canal BM25 não roda |

Num banco de teste, um chunk com "O pH mede a acidez" é encontrado por `MATCH '"ph"'`, mas
`fts_query("pH")` devolve uma string vazia.

**Por que importa.** O próprio módulo explica por que o BM25 está na busca: "o vetorial acha
paráfrase mas perde termo técnico exato". Siglas e símbolos curtos (pH, UV, IR, CA, CC, 5G) são
exatamente esses termos, e aparecem o tempo todo em material de física e química. Hoje eles
dependem só do canal vetorial.

**De onde veio.** No WhatsApp Reader, a escolha do tokenizador FTS5 girou em torno de palavras
curtas: o `trigram` foi descartado porque não encontra consultas com menos de 3 caracteres, e
conversas têm muitos termos assim ("ok", "vc"). Da mensagem do commit `7901908`:

> unicode61 was chosen over the trigram tokenizer because trigram does not match queries
> shorter than three characters ("ok", "vc") and only folds accents from SQLite 3.45.

Aqui o tokenizador já é o `unicode61`, com a mesma configuração. O índice está certo; falta a
consulta usar o que ele tem.

**Como adaptar aqui.** O corte por tamanho parece existir para tirar palavras vazias ("de",
"da", "e", "em"). Duas opções, da mais simples para a mais completa:

- Manter os termos curtos que, na pergunta original, tenham letra maiúscula ou dígito (pH, UV,
  CA, 5G) e continuar descartando os demais. Pega siglas sem nenhuma lista, mas perde uma sigla
  escrita toda em minúsculas.
- Trocar o corte por uma lista curta de palavras vazias em português e inglês. É mais
  previsível, mas é mais uma lista para manter.

Como a consulta usa `OR` e o BM25 dá peso pela raridade do termo, incluir um termo curto custa
pouco. As perguntas da tabela acima servem de teste.

## 3. Avisar quando o resultado está incompleto

**De onde veio.** No WhatsApp Reader, `get_messages` devolvia no máximo `limit` mensagens sem
avisar que havia mais. Num dia movimentado, o modelo resumia metade do dia como se fosse o dia
inteiro. A correção (`2bc8b7d`) busca uma linha além do limite e devolve `has_more`. Trecho de
`E:\whatsapp-reader\whatsapp_reader\conversation_tools.py`:

```python
def _page_payload(messages: list[StoredMessage], limit: int) -> str:
    """Payload for a query run with limit + 1: a surplus row means more messages matched."""
    return _messages_payload(messages[:limit], limit, has_more=len(messages) > limit)
```

**Onde se aplica aqui.**

- **Listagens paginadas.** `collections list`, `documents list` e `jobs list`
  ([`kb/collection.py`](../kb/collection.py) linhas 68 e 309, [`kb/jobs.py`](../kb/jobs.py)
  linha 143) devolvem só a lista. `jobs list` mostra os 10 mais recentes por padrão, e o
  RAG-MCP passa `limit` 50 nas outras duas; nada no resultado diz se havia mais itens.
  - Para não quebrar o contrato (`data` é uma lista, e
    `tests/test_cli_bridge_contract.py` depende disso), a informação pode ir ao lado de `data`
    no envelope: `{"ok": true, "data": [...], "page": {"limit": 50, "offset": 0, "has_more": true}}`.
  - O RAG-MCP repassa o objeto que a CLI imprime sem reconstruí-lo
    (`E:\MCPs\RAG-MCP\src\bridge.ts`), então o campo chega ao client sem mudança lá. Basta
    declará-lo no tipo `CliResult`.
- **`fetch` por páginas.** `fetch_pages` ([`kb/retrieval/__init__.py`](../kb/retrieval/__init__.py)
  linha 82) devolve todo o texto do intervalo, sem limite. Um intervalo grande vira uma resposta
  enorme para o modelo; o bridge aceita até 32 MB. Um teto de caracteres, com `has_more` e a
  próxima página a pedir, mantém a leitura em partes.
- **Seções cortadas na busca.** `expand_to_sections` corta seções acima de
  `KB_MAX_SECTION_CHARS` e marca o corte só no texto, com `[…]`
  ([`kb/retrieval/expand.py`](../kb/retrieval/expand.py) linha 105). Um campo
  `truncated: true` deixaria explícito para o client que vale chamar `fetch --section-id`. O
  `section_id` já vem no resultado.

## 4. Contrato dos parâmetros

**De onde veio.** No WhatsApp Reader, nenhum parâmetro das ferramentas MCP tinha descrição, e um
filtro com nome inexato (`sender="Italo"` para "Italo Silva") devolvia zero resultados sem erro.
A correção (`2bc8b7d`) descreveu cada parâmetro num só lugar, usado tanto pelo MCP quanto pelo
agent. Trecho de `E:\whatsapp-reader\whatsapp_reader\tool_params.py`:

```python
Sender = Annotated[
    str | None,
    Field(
        description=(
            "Exact sender name or phone number as returned by list_senders. "
            "Partial names match nothing."
        )
    ),
]
```

**Onde se aplica aqui.**

- **Collection inexistente vira busca vazia com sucesso.** `resolve_scope`
  ([`kb/collection.py`](../kb/collection.py) linha 222) filtra por `collection_id`.
  **Conferido:** com uma collection de ID `fisica-3` e nome "Física 3", `resolve_scope(["fisica-3"])`
  encontra o documento, mas `["Física 3"]` e `["fisica3"]` devolvem lista vazia. A busca então
  responde `ok` com zero resultados, o que é indistinguível de "nada relevante".
  - A lista vazia é intencional para uma collection que existe mas não tem documentos (está
    documentado em `resolve_scope`).
  - O caso a separar é a collection que não existe: ela poderia devolver `NOT_FOUND`, como
    `collections show` já faz.
- **Validação na CLI.** `--top-k` e `--max-depth` usam `type=int` ([`kb/cli.py`](../kb/cli.py)
  linhas 202 e 223), sem o `_positive_int` que as listagens já usam. Com `--top-k -1`,
  `candidates[:top_k]` (em `kb/retrieval/search.py`) devolve todos os candidatos menos o último.
  O RAG-MCP valida esses valores antes, mas a CLI é a interface pública.
- **Ajuda da CLI.** Esses argumentos também não têm `help`. Para uma pessoa ou um script, o `-h`
  é o contrato.
- **Descrições no RAG-MCP.** As ferramentas em `E:\MCPs\RAG-MCP\src\tools.ts` têm boa descrição
  geral e annotations, mas nenhum parâmetro tem `.describe()`. O modelo não sabe que
  `collections` espera IDs (os de `rag_collections_list`) e não nomes, nem o que muda com
  `expand: "chunk"`. Essa parte é mudança no RAG-MCP, não neste repositório.

## O que não se aplica ou já está resolvido

| Ideia | Por que não precisa aqui |
| --- | --- |
| Posição no ID para não colapsar repetições | `chunk_id` e `section_id` já incluem a posição (`stable_id(doc_id, "chunk", chunk_ord, heading_path)` em `kb/ingest/segment.py`), e `doc_id` é o hash do arquivo. O bug corrigido lá, em que conteúdo idêntico virava uma linha só, não ocorre aqui. |
| FTS5 que ignora acentos e maiúsculas | `chunks_fts` já usa `unicode61 remove_diacritics 2`, conteúdo externo e triggers: a mesma solução que o WhatsApp Reader adotou depois. |
| `stdio` em vez de HTTP numa porta aleatória | A CLI fala com a sessão por pipe nomeado (`AF_PIPE`/`AF_UNIX`) com nome aleatório e `authkey`, sem porta; o RAG-MCP usa `stdio`. Os problemas que motivaram a mudança lá (corrida pela porta, porta aberta, erros jogados em `DEVNULL`) não existem aqui: a sessão escreve em `store/session.log`, e `_last_log_error` leva o erro até a CLI. |
| 405 para métodos não suportados e logs do SDK | Não há servidor HTTP neste projeto. |
| Recusa do token bearer | O `authkey` daqui é gerado a cada sessão e fica invisível ao usuário. Não tem o custo de configuração que motivou a recusa lá. |
| Horários ISO com fuso | Nenhum comando filtra por data. |

## O caminho inverso

Alguns pontos deste projeto e do RAG-MCP estão à frente do WhatsApp Reader e podem servir de
referência para ele:

- O RAG-MCP marca as ferramentas com `readOnlyHint` e `destructiveHint` e devolve
  `isError: true` em falhas. Lá, os erros ainda voltam como sucesso.
- O envelope da CLI daqui, com `code`, `message` e `hint`, é um contrato de erro mais claro que
  o `{"error": ...}` de lá.
- `WAL` e `busy_timeout` já estão configurados aqui; lá esse ajuste ficou como baixa prioridade.

## Ordem sugerida

1. Palavras curtas na busca léxica: mudança pequena, num ponto só, com efeito direto na
   recuperação.
2. Versionamento do esquema: preventivo, e melhor feito antes da primeira mudança de schema.
3. `NOT_FOUND` para collection inexistente e validação de `--top-k` e `--max-depth`.
4. `page.has_more` nas listagens, `truncated` nas seções e teto em `fetch_pages`.
5. Descrições de parâmetros no RAG-MCP (outro repositório).

## Como os pontos foram conferidos

Tudo rodou com `KB_STORE_DIR` apontando para uma pasta temporária fora do projeto, sem tocar em
`store/`:

```powershell
$env:KB_STORE_DIR = "$env:TEMP\rag-store-teste"
.\.venv\Scripts\python.exe -c "from kb.retrieval.search import fts_query; print(fts_query('pH da solução tampão'))"
```

- **Versão rebaixada:** `init_db()`, depois `PRAGMA user_version = 2`, depois um novo
  `init_db()`, que devolve a versão 1.
- **Termos curtos:** `fts_query` com as perguntas da seção 2, e `MATCH '"ph"'` direto no
  `chunks_fts` com um chunk de teste.
- **Escopo:** `collections create` com ID e nome diferentes, seguido de `resolve_scope` com o
  ID, com o nome e com um ID digitado errado.
