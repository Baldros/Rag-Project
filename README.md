# Multi-Framework RAG System

Este é um sistema de **Retrieval-Augmented Generation (RAG)** modular e extensível, desenvolvido para processar documentos PDF e fornecer respostas precisas utilizando LLMs locais. O sistema oferece suporte a diferentes implementações, permitindo escolher entre **LangChain**, **LlamaIndex** ou uma abordagem **Vanilla LLM**.

## 🚀 Funcionalidades

- **Multi-Framework:** Implementações prontas para LangChain e LlamaIndex.
- **Processamento Avançado:** Conversão de PDFs de alta qualidade utilizando [Docling](https://github.com/DS4SD/docling).
- **Banco Vetorial Persistente:** Armazenamento e recuperação de chunks com [ChromaDB](https://www.trychroma.com/).
- **Privacidade e Localidade:** Integração total com [Ollama](https://ollama.ai/) para embeddings e geração de texto local.
- **Interface Interativa:** CLI de chat para interagir com os documentos indexados.
- **Código Modular:** Lógica de negócio separada no pacote `processing/` para fácil manutenção e expansão.

## 🏗️ Estrutura do Projeto

```text
E:\Rag-Project\
├── chat.py                # Interface principal de chat (CLI)
├── Processing.py          # Script para indexação de novos documentos
├── langchain-rag-system.py # Implementação do agente RAG via LangChain
├── llama-rag-system.py     # Implementação do agente RAG via LlamaIndex
├── vanilla-system.py       # Chat direto com LLM (sem RAG)
├── requirements.txt        # Dependências do projeto
├── chroma_db/              # Diretório de persistência do banco vetorial
└── processing/             # Pacote principal com a lógica do pipeline
    ├── config.py           # Configurações centralizadas (Modelos, Paths, etc)
    ├── converter.py        # Conversão de documentos via Docling
    ├── indexer.py          # Lógica de chunking e indexação no ChromaDB
    ├── query.py            # Funções de busca e recuperação
    ├── rag.py              # Lógica de integração RAG e prompts
    └── storage.py          # Gerenciamento do cliente ChromaDB
```

## 📋 Pré-requisitos

1.  **Python 3.10 ou superior.**
2.  **Ollama instalado e rodando.**
3.  **Modelos do Ollama baixados:**
    ```bash
    ollama pull qwen3.5:0.8b    # Modelo de linguagem (ajustável no config.py)
    ollama pull embeddinggemma  # Modelo de embeddings (ajustável no config.py)
    ```
4.  **PyTorch com suporte a CUDA (Opcional, mas recomendado):**
    O `docling` e o processamento de embeddings se beneficiam muito de aceleração por GPU.

## 🛠️ Instalação

1.  Clone o repositório ou baixe os arquivos.
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    # No Windows:
    .venv\Scripts\activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Como Usar

### 1. Indexação de Documentos
Antes de conversar, você precisa indexar seus arquivos PDF.
```bash
python Processing.py
```
O script solicitará o caminho da pasta contendo os PDFs. O sistema irá processar, converter para Markdown via Docling, dividir em blocos e armazenar no ChromaDB.

### 2. Chat Interativo
Inicie a interface de chat para interagir com os documentos:
```bash
python chat.py
```
Ao iniciar, você poderá escolher entre:
1.  **LlamaIndex (RAG):** Recuperação e geração via LlamaIndex.
2.  **LangChain (RAG):** Recuperação e geração via LangChain.
3.  **Vanilla LLM:** Conversa direta com o modelo (sem contexto dos documentos).

## ⚙️ Configuração

Ajuste os parâmetros do sistema no arquivo `processing/config.py`. Lá você pode alterar:
- Nomes dos modelos (`LLM_MODEL`, `EMBED_MODEL`).
- URL do Ollama.
- Parâmetros de RAG (`RAG_TOP_K`, `RAG_MAX_CONTEXT_CHARS`).
- Tamanho dos blocos de processamento (`PAGE_BLOCK_SIZE`).

## 🛠️ Tecnologias Utilizadas

- **Docling:** Para parsing robusto de PDFs.
- **ChromaDB:** Banco de dados vetorial para busca semântica.
- **LangChain & LlamaIndex:** Frameworks de orquestração de IA.
- **Ollama:** Execução local de modelos de linguagem e embeddings.
- **Python:** Linguagem base do projeto.
