"""
Gerência de modelos e de VRAM.

Regra do projeto: um modelo por vez na GPU. Aqui isso vira carga preguiçosa
(nada sobe até ser usado) mais descarga por inatividade (o que subiu desce
sozinho). Numa ingestão longa e desatendida, a pilha de retrieval já se
descarregou antes do worker precisar da GPU.

O LLM não aparece em lugar nenhum: geração é responsabilidade do agente que
consome o MCP. Aqui só existem modelos de recuperação.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable

from kb.config import (
    DEVICE,
    EMBED_MODEL,
    MODEL_IDLE_TIMEOUT,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    TORCH_DTYPE,
)

logger = logging.getLogger(__name__)

_REAPER_INTERVAL = 30.0


def resolve_device() -> str:
    """Devolve o device realmente utilizável, caindo para CPU se não houver CUDA."""
    import torch

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA indisponível; usando CPU.")
        return "cpu"

    return DEVICE


def _dtype(device: str):
    import torch

    if device == "cpu":
        return torch.float32

    return getattr(torch, TORCH_DTYPE, torch.float16)


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
    """Um modelo com carga preguiçosa, uso registrado e descarga por ociosidade."""

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self.name = name
        self._loader = loader
        self._model: Any = None
        self._last_used = 0.0
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> Any:
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
                _ensure_reaper()

            self._last_used = time.monotonic()
            return self._model

    def unload(self) -> bool:
        with self._lock:
            if self._model is None:
                return False

            logger.info("Descarregando %s", self.name)
            del self._model
            self._model = None

        free_vram()
        return True

    def unload_if_idle(self, timeout: float) -> bool:
        with self._lock:
            if self._model is None:
                return False
            if time.monotonic() - self._last_used < timeout:
                return False

        return self.unload()

    def info(self) -> dict[str, Any]:
        with self._lock:
            idle = time.monotonic() - self._last_used if self._model else None
            return {
                "name": self.name,
                "loaded": self._model is not None,
                "idle_seconds": round(idle, 1) if idle is not None else None,
            }


def _load_embedder():
    from sentence_transformers import SentenceTransformer

    device = resolve_device()
    return SentenceTransformer(
        EMBED_MODEL,
        device=device,
        model_kwargs={"torch_dtype": _dtype(device)},
    )


def _load_reranker():
    from sentence_transformers import CrossEncoder

    device = resolve_device()
    return CrossEncoder(
        RERANK_MODEL,
        device=device,
        max_length=RERANK_MAX_LENGTH,
        model_kwargs={"torch_dtype": _dtype(device)},
    )


EMBEDDER = ModelSlot("embedder", _load_embedder)
RERANKER = ModelSlot("reranker", _load_reranker)

_SLOTS = (EMBEDDER, RERANKER)

_reaper: threading.Thread | None = None
_reaper_lock = threading.Lock()


def _reap_loop() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL)

        if MODEL_IDLE_TIMEOUT <= 0:
            continue

        for slot in _SLOTS:
            try:
                slot.unload_if_idle(MODEL_IDLE_TIMEOUT)
            except Exception:  # pragma: no cover
                logger.exception("Falha ao descarregar %s", slot.name)


def _ensure_reaper() -> None:
    global _reaper

    if MODEL_IDLE_TIMEOUT <= 0:
        return

    with _reaper_lock:
        if _reaper is None or not _reaper.is_alive():
            _reaper = threading.Thread(
                target=_reap_loop,
                name="kb-model-reaper",
                daemon=True,
            )
            _reaper.start()


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embeda documentos. Vetores normalizados: similaridade vira produto escalar."""
    if not texts:
        return []

    model = EMBEDDER.get()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return vectors.astype("float32").tolist()


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
    scores = model.predict(
        [(query, doc) for doc in documents],
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return [float(score) for score in scores]


def unload_all() -> list[str]:
    """Descarrega tudo. Usado ao fim de um pass de ingestão."""
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
