"""
Rerank — cross-encoder.

A busca vetorial usa bi-encoder: chunk e query viram vetores separadamente,
antes de um saber da existencia do outro. E barato e varre a base inteira, mas e
cego para *qual parte* do chunk importa para *esta* pergunta.

O cross-encoder le os dois juntos, na mesma passada, com atencao cruzada. O
score e muito melhor e o custo e uma passada por candidato: inviavel sobre a
base toda, trivial sobre os 20 que a fusao entregou.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kb.config import RERANK_ENABLED

if TYPE_CHECKING:
    from kb.retrieval.search import Hit

logger = logging.getLogger(__name__)


def rerank_hits(query: str, hits: list["Hit"]) -> list["Hit"]:
    """
    Reordena os candidatos pelo score do cross-encoder.

    Falha aberta: se o modelo nao carregar, devolve a ordem da fusao em vez de
    derrubar a busca. Perder qualidade e aceitavel; perder a busca nao e.
    """
    if not hits or not RERANK_ENABLED:
        return hits

    try:
        from kb.models import rerank

        scores = rerank(query, [_passage(hit) for hit in hits])
    except Exception:
        logger.exception("Rerank falhou; mantendo a ordem do RRF.")
        return hits

    for hit, score in zip(hits, scores):
        hit.scores["rerank"] = round(float(score), 4)

    return sorted(hits, key=lambda hit: hit.scores.get("rerank", 0.0), reverse=True)


def _passage(hit: "Hit") -> str:
    """
    Texto apresentado ao cross-encoder.

    Inclui o heading_path porque ele carrega o contexto estrutural que o chunk
    isolado perdeu: "SOLUCAO" sozinho nao diz de que exemplo se trata.
    """
    if hit.heading_path:
        return f"{hit.heading_path}\n{hit.text}"

    return hit.text
