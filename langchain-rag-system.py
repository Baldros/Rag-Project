"""
Simple RAG agent with LangChain + ChromaDB.

Chroma is used as a vector base, and the LLM receives a clean, grounded context
with response-mode-aware instructions.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from processing.config import (
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    OLLAMA_URL,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOP_K,
)
from processing.rag import (
    RAG_SYSTEM_PROMPT,
    AnswerMode,
    build_user_prompt,
    finalize_answer,
    format_context,
    infer_answer_mode,
    retrieve_chunks,
)


class LangChainRAGAgent:
    def __init__(self, *, top_k: int = RAG_TOP_K) -> None:
        self.top_k = top_k
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            reasoning=False,
            temperature=0,
            num_ctx=LLM_CONTEXT_WINDOW,
            num_predict=LLM_MAX_TOKENS,
            keep_alive="5m",
            sync_client_kwargs={"timeout": LLM_REQUEST_TIMEOUT},
        )

    def invoke(self, payload: dict) -> dict:
        question = _latest_user_message(payload)
        answer_mode = _answer_mode(payload, question)
        chunks = retrieve_chunks(question, n_results=self.top_k)
        context = format_context(chunks, max_chars=RAG_MAX_CONTEXT_CHARS)

        response = self.llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(question, context, answer_mode=answer_mode)),
            ]
        )
        answer = finalize_answer(response.content, question, chunks)

        return {"messages": [AIMessage(content=answer)]}


def _latest_user_message(payload: dict) -> str:
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", "")).strip()

        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role in {"human", "user"}:
            return str(getattr(message, "content", "")).strip()

    text = payload.get("input") or payload.get("query") or payload.get("question")
    if text:
        return str(text).strip()

    raise ValueError("No user message found for the RAG query.")


def _answer_mode(payload: dict, question: str) -> AnswerMode:
    requested_mode = payload.get("answer_mode") or payload.get("mode")
    if requested_mode:
        return AnswerMode(str(requested_mode))
    return infer_answer_mode(question)


def get_chat_agent():
    return LangChainRAGAgent()
