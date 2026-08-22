"""Rank-based retrieval metrics with bootstrap confidence intervals.

These need no LLM. They are deterministic, run in milliseconds, and turn a
two-hour judged evaluation cycle into one you can run after every change --
which is the only reason iterating on retrieval is practical at all.

Ground truth in the gold set is document-level, while retrievers return
chunks. Chunks are therefore collapsed to their parent documents, preserving
rank order and keeping each document's best position. "@k" means the top k
distinct documents.

With binary relevance the metrics are:

    Recall@k     fraction of relevant documents that appear in the top k
    Precision@k  fraction of the top k that are relevant
    MRR          1 / rank of the first relevant document
    nDCG@k       DCG@k / ideal DCG@k, discounting by log2(rank + 1)
"""

from __future__ import annotations

import math
import random


def to_document_ranking(retrieved_doc_ids: list[str]) -> list[str]:
    """Collapse a chunk-level ranking to distinct documents, best rank first."""
    seen, ordered = set(), []
    for doc_id in retrieved_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document; 0 if none was retrieved."""
    for position, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranking[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def score_query(
    retrieved_doc_ids: list[str],
    relevant_docs: list[str] | set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """All metrics for a single query."""
    ranking = to_document_ranking(retrieved_doc_ids)
    relevant = set(relevant_docs)

    scores = {"mrr": reciprocal_rank(ranking, relevant)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(ranking, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranking, relevant, k)
    return scores


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    iterations: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Reported alongside every headline number because these evaluations run on
    a few dozen questions. A win rate quoted from n=25 without an interval
    implies a precision the sample size does not support.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * iterations)]
    high = means[min(int((1.0 - tail) * iterations), iterations - 1)]
    return (low, high)


def abstention_rate(answered: list[bool]) -> float:
    """Fraction of questions the system declined to answer."""
    if not answered:
        return 0.0
    return sum(1 for did_answer in answered if not did_answer) / len(answered)
