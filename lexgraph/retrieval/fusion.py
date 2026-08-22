"""Reciprocal Rank Fusion.

Merging a dense ranking with a BM25 ranking needs the two score scales
reconciled. Cosine similarity sits in roughly [0, 1]; BM25 is unbounded and
corpus-dependent. Normalising either one introduces a tuning knob and breaks
whenever the corpus changes.

RRF sidesteps that by discarding scores and using only rank:

    score(d) = sum over rankings of  1 / (k + rank(d))

``k`` (conventionally 60) damps the top of each list so a single ranker cannot
dominate on its first result alone. Nothing needs normalising, and adding a
third retriever later needs no re-tuning.
"""

from __future__ import annotations

from .base import Hit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    """Fuse several ranked hit lists into one.

    ``rankings`` are ordered best-first. ``weights`` scales each ranking's
    contribution and defaults to equal weighting. Hits are matched by
    ``chunk_id``; the first occurrence supplies the returned text and metadata.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights ({len(weights)}) must match rankings ({len(rankings)})"
        )

    fused: dict[str, Hit] = {}
    scores: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            contribution = weight / (k + rank)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + contribution

            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    title=hit.title,
                    year=hit.year,
                    para_label=hit.para_label,
                    components=dict(hit.components),
                )
            fused[hit.chunk_id].components.update(hit.components)

    for chunk_id, score in scores.items():
        fused[chunk_id].score = score

    ranked = sorted(fused.values(), key=lambda hit: (-hit.score, hit.chunk_id))
    return ranked[:top_k] if top_k else ranked
