# Revisão da ingestão — novembro de 2026

## Correções implementadas

| Problema observado | Correção |
| --- | --- |
| Caminhos Unicode chegavam diretamente ao backend PDF nativo. | Python abre o arquivo e entrega um `DocumentStream` em memória ao Docling. |
| Um objeto de documento vazio podia ser tratado como conversão bem-sucedida; bastava um bloco válido para marcar o manifest como completo. | Exigir `ConversionStatus.SUCCESS`, cobertura exata de páginas e todos os blocos válidos. |
| O parser padrão descartava as fórmulas em bloco. | Habilitar CodeFormulaV2 em todos os PDFs, margem de 8%, interrupção de repetição e uma nova tentativa sobre o layout salvo para falhas detectáveis. |
| O contexto dos títulos reiniciava a cada bloco de 15 páginas. | Concatenar os documentos com a API do Docling antes do chunking e verificar a preservação das páginas. |
| Alterar chunking não tinha uma assinatura própria de cache. | Separar assinaturas de conversão, fórmulas e segmentação. |
| Substituir chunks podia deixar vetores obsoletos no Chroma. | Publicar uma marca persistente de limpeza com os chunks; excluir vetores antigos antes de inserir os novos e retomar em caso de interrupção. |
| Jobs perdiam o resultado detalhado e representavam mal sucessos parciais. | Guardar `result_json`, classificar `done/partial/failed` e só declarar documentos indexados após confirmar os embeddings. |
| Acentos separados do TeX prejudicavam o FTS. | Manter campos lexicais normalizados, sem modificar o texto das citações; reconstruir o FTS numa transação. |

A publicação usa gerações de artefatos com hashes. O SQLite referencia a geração
ativa e a troca de chunks, seções, qualidade e marca de limpeza é transacional.
Uma conversão malsucedida não substitui a versão anteriormente indexada.

`doctor` compara os IDs dos vetores, além das contagens, e verifica artefatos e
limpezas pendentes. Metadados de qualidade chegam a catálogo, sumário, busca e
leitura. No MCP, a opção de OCR é tipada e o pipe preserva caracteres UTF-8
divididos entre blocos de saída.

## Validação

- 58 testes Python, incluindo migração real do esquema 1, rollback, retomada,
  falha de embedding, substituição de vetores, Unicode, FTS e jobs parciais.
- 7 testes MCP, incluindo transporte stdio e chamadas reais à CLI.
- Integração com as classes reais do Docling: título preservado da página 15
  para a 16, incluindo uma página em branco no bloco seguinte.
- `pip check` sem dependências quebradas; PyTorch 2.12.0+cu126 preservado.
- Migração da base existente: 4 documentos, 213 seções e 455 chunks preservados,
  com `PRAGMA integrity_check = ok`, antes da reingestão selecionada.
- Conversão real das páginas 22–23 usando o caminho original com acentos:
  15 fórmulas recuperadas. Conferência visual mostrou que reduzir a margem de
  18% para 8% corrigiu erros na densidade normal e na esperança do escore.

O backup anterior às mudanças fica em `store-backups/20261031-232304`.
Os scripts e resultados da validação local ficam em `.testdata/`, ignorados pelo
Git para não publicar o conteúdo dos livros.

## Reingestão do Bolfarine

O job `4b8a9d709b96` reingeriu o PDF original, com acentos no caminho, na collection
`inferencia-estatistica`. Terminou como `partial`, com o documento indexado e o
resumo atualizado. Os três outros documentos, seus chunks e suas seções foram
comparados com o backup e permaneceram iguais.

| Medida | Resultado |
| --- | --- |
| Páginas | 126, com cobertura e hashes dos 9 blocos válidos |
| Seções / chunks / tabelas | 60 / 306 / 9 |
| Fórmulas detectadas | 577, nenhuma vazia após a extração |
| Fórmulas sinalizadas após a nova tentativa | 5, nas páginas PDF 27, 74, 111, 115 e 116 |
| Conversão e segmentação | 1.941,985 s |
| Embeddings | 45,390 s |
| Resumo | 30,971 s |
| Total | 2.018,363 s, aproximadamente 33min38s |

As seções passaram de 66 para 60 após preservar o contexto dos títulos entre
blocos; os chunks passaram de 191 para 306 com a inclusão das fórmulas. A única
seção sem título agora é a página inicial.

A base inteira terminou com 570 vetores e exatamente os mesmos IDs de chunks
marcados como indexados no SQLite: nenhuma ausência, sobra ou limpeza pendente.
`integrity_check` retornou `ok` e `foreign_key_check` não retornou erros. O
diagnóstico também identifica um diretório de artefatos órfão de setembro, já
presente no backup, que foi preservado.

A consulta lexical direta por “estatística” passou de 0 para 65 chunks do
Bolfarine; “estatísticas”, de 0 para 53; “introdução”, de 0 para 5. Essas contagens
medem recuperação lexical, não precisão matemática. A extração atual pode
detectar itens diferentes da versão anterior do Docling; a cobertura integral
das páginas não é garantia de detecção de todas as fórmulas do PDF.

Na consulta “Qual é a definição de estatística suficiente?”, os três primeiros
resultados pertencem à seção 2.2, nas páginas PDF 26, 26 e 27, com scores do
reranker de 0,9323, 0,8652 e 0,8252. O primeiro contém a Definição 2.2.1. A leitura
do exemplo de Bernoulli preserva `t = 0, . . . , n`. O resultado da página 27
inclui o aviso de fórmula sinalizada; os demais também informam que a correção
matemática não foi verificada. A sessão foi desligada ao terminar a validação.

## Limites observados

Transcrever uma equação para LaTeX não garante que os símbolos estejam corretos.
A inspeção da seção 2.2 encontrou erros em fórmulas complexas mesmo quando o
resultado tinha conteúdo e chaves balanceadas. A ordem de algumas equações na
saída do Docling também pode diferir da página original. Esses limites exigem
conferência no PDF antes de usar as equações em cálculos.

`quality.status = complete` significa cobertura completa e ausência de falhas
detectadas pela validação automática. `formula_verification = automated_unverified`
explicita o limite da transcrição. Fórmulas vazias ou com chaves desequilibradas
após uma tentativa adicional deixam a qualidade como `partial`; seus números de
página e referências são preservados. Essa verificação não detecta todo erro
matemático possível.

O processamento continua local e sequencial: Docling, embeddings, resumos
opcionais. Quando os resumos solicitados já existem, o pass não exige que o LLM
esteja disponível nem produz um aviso falso de pendência. A extração de centenas
de fórmulas aumenta o tempo do primeiro pass;
mudanças posteriores de chunking reutilizam o parse e alterações de fórmulas
reutilizam o layout compatível.

## Referências consultadas

- [Enriquecimento e configuração de recortes no Docling](https://docling-project.github.io/docling/usage/enrichments/).
- [Modelo oficial CodeFormulaV2](https://huggingface.co/docling-project/CodeFormulaV2).
- [Enriquecimento de um DoclingDocument já convertido](https://docling-project.github.io/docling/_generated/examples/enrich_doclingdocument/).
