"""
Gerência de modelos e de VRAM.

Só modelos de **recuperação** vivem aqui — embedding e rerank. O LLM da
ingestão é outro sistema e mora em `kb/llm.py`.

Só uma sessão iniciada explicitamente carrega modelos. Operações de busca nunca
os carregam por conta própria: sem sessão pronta, recebem erro claro.

`readiness()` e `ModelSlot.info()` são deliberadamente livres de lock: quem os
chama está tentando descobrir se a carga terminou, e não pode ficar preso atrás
dela para saber.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable

from kb.config import (
    RERANK_ENABLED,
)

logger = logging.getLogger(__name__)

class ModelNotLoaded(RuntimeError):
    """A sessão necessária ainda não carregou este modelo."""


def free_vram() -> None:
    """Devolve a memória liberada ao driver. Sem isso o cache do torch a retém."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        logger.debug("empty_cache falhou", exc_info=True)


class ModelSlot:
    """Um modelo carregado e descarregado explicitamente pela sessão."""

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self.name = name
        self._loader = loader
        self._model: Any = None
        self._last_used = 0.0
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        with self._lock:
            if self._model is None:
                started = time.perf_counter()
                logger.info("Carregando %s (%s)...", self.name, self._loader.__name__)
                self._model = self._loader()
                logger.info(
                    "%s carregado em %.1fs",
                    self.name,
                    time.perf_counter() - started,
                )

            self._last_used = time.monotonic()
            return self._model

    def get(self) -> Any:
        model = self._model
        if model is None:
            raise ModelNotLoaded(
                f"{self.name} não carregado; execute `kb on --mode interaction`"
            )
        self._last_used = time.monotonic()
        return model

    def unload(self) -> bool:
        with self._lock:
            if self._model is None:
                return False

            logger.info("Descarregando %s", self.name)
            del self._model
            self._model = None

        free_vram()
        return True

    def info(self) -> dict[str, Any]:
        """
        Estado do slot, **sem** pegar o lock.

        De propósito: o lock fica retido durante a carga inteira, que aqui leva
        dezenas de segundos. Se este método esperasse por ele, a única tool capaz
        de responder "já está pronto?" ficaria presa exatamente enquanto a
        resposta seria "ainda não". Ler uma referência é atômico o bastante para
        um relatório de estado.
        """
        model = self._model
        idle = time.monotonic() - self._last_used if model is not None else None

        return {
            "name": self.name,
            "loaded": model is not None,
            "idle_seconds": round(idle, 1) if idle is not None else None,
        }


def _load_embedder():
    from kb.providers import embedding_adapter

    return embedding_adapter()


def _load_reranker():
    from kb.providers import rerank_adapter

    return rerank_adapter()


EMBEDDER = ModelSlot("embedder", _load_embedder)
RERANKER = ModelSlot("reranker", _load_reranker)

_SLOTS = (EMBEDDER, RERANKER)

def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embeda documentos. Vetores normalizados: similaridade vira produto escalar."""
    if not texts:
        return []

    model = EMBEDDER.get()
    return model.embed(texts, batch_size)


def embed_query(query: str) -> list[float]:
    """
    Embeda a pergunta.

    O bge-m3 não usa prefixo de instrução em query — ao contrário dos BGE
    anteriores, que exigiam "Represent this sentence...".
    """
    return embed_texts([query], batch_size=1)[0]


def rerank(query: str, documents: list[str]) -> list[float]:
    """Pontua cada documento contra a query com atenção cruzada (cross-encoder)."""
    if not documents:
        return []

    model = RERANKER.get()
    return model.score(query, documents)


# Estado do aquecimento, escrito pela thread de warmup e lido pelas tools.
# Chaves independentes e escritas atômicas: não precisa de lock, e não pode
# precisar — quem lê está justamente tentando descobrir se a carga terminou.
_warmup: dict[str, Any] = {"started_at": None, "finished_at": None, "error": None}


def preload(rerank: bool = True) -> dict[str, float]:
    """
    Sobe os modelos de recuperação de uma vez.

    Chamado exclusivamente pelo comando manual que inicia a sessão de interação.
    """
    timings: dict[str, float] = {}

    _warmup["started_at"] = time.monotonic()
    _warmup["finished_at"] = None
    _warmup["error"] = None

    try:
        started = time.perf_counter()
        EMBEDDER.load()
        timings["embedder"] = round(time.perf_counter() - started, 1)

        if rerank:
            started = time.perf_counter()
            RERANKER.load()
            timings["reranker"] = round(time.perf_counter() - started, 1)
    except Exception as exc:
        _warmup["error"] = str(exc)
        raise
    finally:
        _warmup["finished_at"] = time.monotonic()

    logger.info("Modelos prontos: %s", timings)
    return timings


def readiness() -> dict[str, Any]:
    """
    Se a busca pode ser atendida agora, sem bloquear para descobrir.

    `warming_up` distingue "ainda carregando" de "sessão desligada".
    """
    ready = EMBEDDER.loaded and (RERANKER.loaded or not RERANK_ENABLED)

    started = _warmup["started_at"]
    finished = _warmup["finished_at"]
    warming = started is not None and finished is None and not ready

    state: dict[str, Any] = {
        "ready": ready,
        "warming_up": warming,
        "embedder_loaded": EMBEDDER.loaded,
        "reranker_loaded": RERANKER.loaded,
    }

    if started is not None:
        end = finished if finished is not None else time.monotonic()
        state["elapsed_s"] = round(end - started, 1)

    if _warmup["error"]:
        state["error"] = _warmup["error"]

    return state


def unload_all() -> list[str]:
    """
    Descarrega tudo e devolve a VRAM.

    Usado ao desligar a sessão e ao fim de um pass de ingestão.
    """
    return [slot.name for slot in _SLOTS if slot.unload()]


def models_info() -> list[dict[str, Any]]:
    return [slot.info() for slot in _SLOTS]


def vram_info() -> dict[str, Any]:
    """Estado da GPU, para a tool `status`."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "total_mb": round(total / 1024**2),
            "free_mb": round(free / 1024**2),
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
        }
    except Exception:
        return {"available": False}
