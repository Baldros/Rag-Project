"""
RAG simples com LlamaIndex LLM + ChromaDB.

O Chroma e consultado como base vetorial persistida; o LLM so recebe
um contexto curto e limpo.
"""

from __future__ import annotations

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from processing.config import (
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    OLLAMA_URL,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_K,
)
from processing.rag import (
    RAG_SYSTEM_PROMPT,
    RetrievedChunk,
    build_user_prompt,
    finalize_answer,
    format_context,
    retrieve_chunks,
)


class SimpleChatResponse:
    def __init__(self, response: str, source_chunks: list[RetrievedChunk]) -> None:
        self.response = response
        self.source_chunks = source_chunks

    def __str__(self) -> str:
        return self.response


class LlamaRAGChatEngine:
    def __init__(self, *, top_k: int = RAG_TOP_K) -> None:
        self.top_k = top_k
        self.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            request_timeout=LLM_REQUEST_TIMEOUT,
            temperature=0,
            context_window=4096,
            thinking=False,
            keep_alive="5m",
        )

    def chat(self, message: str) -> SimpleChatResponse:
        chunks = retrieve_chunks(message, n_results=self.top_k)
        context = format_context(chunks, max_chars=RAG_MAX_CONTEXT_CHARS)
        response = self.llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=RAG_SYSTEM_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=build_user_prompt(message, context),
                ),
            ]
        )

        answer = finalize_answer(response.message.content or "", message, chunks)
        return SimpleChatResponse(answer, chunks)


def get_chat_engine(top_k: int = RAG_TOP_K):
    return LlamaRAGChatEngine(top_k=top_k)
