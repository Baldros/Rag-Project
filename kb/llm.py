"""Geração de resumos por adapter configurável.

Usado apenas pelo Pass 3 da ingestão, para processamento de texto: resumir um
documento ou uma seção. Não precisa de tool calling, nem de contexto longo, nem
de raciocínio elaborado. É o membro mais pesado do sistema e por isso o que mais
se beneficia da descarga explícita.

O Ollama é o adapter inicial; outros provedores implementam o mesmo contrato.
O gerador só é usado no passo de enriquecimento da ingestão.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from kb.config import (
    LLM_ENABLED,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_URL,
)

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """O provedor de geração não está pronto."""


def _ollama_available() -> bool:
    """Checagem barata, para decidir se vale rodar o Pass 3."""
    if not LLM_ENABLED:
        return False

    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Ollama indisponível em %s: %s", OLLAMA_URL, exc)
        return False

    models = {
        model.get("name", "") for model in response.json().get("models", [])
    }

    # O Ollama trata "qwen3:8b" e "qwen3:8b-..." como nomes distintos; aceitar o
    # prefixo evita falso negativo com tags de quantização.
    if any(name == LLM_MODEL or name.startswith(f"{LLM_MODEL}-") for name in models):
        return True

    logger.warning(
        "Modelo %s não encontrado no Ollama. Disponíveis: %s",
        LLM_MODEL,
        ", ".join(sorted(models)) or "nenhum",
    )
    return False


def _ollama_generate(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 700,
    keep_alive: str | int = "5m",
) -> str:
    """
    Gera texto. `keep_alive=0` descarrega o modelo assim que a resposta sai.

    Durante um lote vale manter carregado entre as chamadas; ao fim do pass,
    `unload()` devolve a VRAM.
    """
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": max_tokens,
        },
    }

    if system:
        payload["system"] = system

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=LLM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except Exception as exc:
        raise LLMUnavailable(f"Falha ao chamar {LLM_MODEL}: {exc}") from exc

    return strip_reasoning(response.json().get("response", "")).strip()


def strip_reasoning(text: str) -> str:
    """
    Remove blocos de raciocínio dos modelos que pensam antes de responder.

    O qwen3 emite `<think>...</think>` por padrão. Guardar isso como resumo
    encheria o catálogo de monólogo interno em vez do conteúdo pedido.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _ollama_unload() -> bool:
    """
    Tira o modelo da VRAM imediatamente.

    Chamado ao fim do Pass 3. `keep_alive: 0` é o primitivo que o Ollama expõe
    para isso; sem ele o modelo fica residente por 5 minutos, atravessando o
    fim da ingestão.
    """
    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "keep_alive": 0},
            timeout=30.0,
        )
        response.raise_for_status()
        logger.info("LLM %s descarregado da VRAM", LLM_MODEL)
        return True
    except Exception as exc:
        logger.warning("Não consegui descarregar %s: %s", LLM_MODEL, exc)
        return False


class OllamaAdapter:
    def is_available(self) -> bool:
        return _ollama_available()

    def generate(self, prompt: str, system: str | None, max_tokens: int) -> str:
        return _ollama_generate(prompt, system, max_tokens)

    def unload(self) -> bool:
        return _ollama_unload()


_adapter = None


def _provider():
    global _adapter
    if _adapter is None:
        from kb.providers import generation_adapter

        _adapter = generation_adapter()
    return _adapter


def is_available() -> bool:
    return _provider().is_available()


def generate(prompt: str, system: str | None = None, max_tokens: int = 700) -> str:
    return _provider().generate(prompt, system, max_tokens)


def unload(force: bool = False) -> bool:
    """Unload a model used by this process, or by a cancelled runtime."""
    if _adapter is None and not force:
        return False
    return _provider().unload()
