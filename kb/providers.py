"""Contracts and adapters for the three independent model roles.

The RAG pipeline depends on these contracts, never on a provider's HTTP or
Python API. Adapters are created only during an explicit CLI session or job.
"""

from __future__ import annotations

from typing import Protocol

from kb import config


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]: ...


class RerankProvider(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class GenerationProvider(Protocol):
    def is_available(self) -> bool: ...
    def generate(self, prompt: str, system: str | None, max_tokens: int) -> str: ...
    def unload(self) -> bool: ...


def _device() -> str:
    import torch

    if config.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return config.DEVICE


def _dtype(device: str):
    import torch

    return torch.float32 if device == "cpu" else getattr(torch, config.TORCH_DTYPE)


class SentenceTransformersEmbedding:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        device = _device()
        self.model = SentenceTransformer(
            config.EMBED_MODEL,
            device=device,
            model_kwargs={"torch_dtype": _dtype(device)},
        )

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32").tolist()


class SentenceTransformersRerank:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        device = _device()
        self.model = CrossEncoder(
            config.RERANK_MODEL,
            device=device,
            max_length=config.RERANK_MAX_LENGTH,
            model_kwargs={"torch_dtype": _dtype(device)},
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores = self.model.predict(
            [(query, document) for document in documents],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]


def embedding_adapter() -> EmbeddingProvider:
    if config.EMBED_PROVIDER == "sentence_transformers":
        return SentenceTransformersEmbedding()
    raise ValueError(f"Adapter de embedding desconhecido: {config.EMBED_PROVIDER}")


def rerank_adapter() -> RerankProvider:
    if config.RERANK_PROVIDER == "sentence_transformers":
        return SentenceTransformersRerank()
    raise ValueError(f"Adapter de rerank desconhecido: {config.RERANK_PROVIDER}")


def generation_adapter() -> GenerationProvider:
    if config.LLM_PROVIDER == "ollama":
        from kb.llm import OllamaAdapter

        return OllamaAdapter()
    raise ValueError(f"Adapter de geração desconhecido: {config.LLM_PROVIDER}")


def validate_config() -> None:
    """Reject unsupported adapters before any model or index is changed."""
    for role, selected, supported, enabled in (
        ("embedding", config.EMBED_PROVIDER, {"sentence_transformers"}, True),
        ("rerank", config.RERANK_PROVIDER, {"sentence_transformers"}, config.RERANK_ENABLED),
        ("geração", config.LLM_PROVIDER, {"ollama"}, config.LLM_ENABLED),
    ):
        if enabled and selected not in supported:
            raise ValueError(f"Adapter de {role} desconhecido: {selected}")
