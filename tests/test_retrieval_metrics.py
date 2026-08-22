import math

import pytest

from lexgraph.eval.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
    to_document_ranking,
)


def test_chunk_ranking_collapses_to_documents_keeping_best_rank():
    chunks = ["a.txt", "a.txt", "b.txt", "a.txt", "c.txt"]
    assert to_document_ranking(chunks) == ["a.txt", "b.txt", "c.txt"]


def test_recall_counts_relevant_documents_found():
    ranking = ["a", "b", "c", "d"]
    assert recall_at_k(ranking, {"a", "d"}, k=4) == 1.0
    assert recall_at_k(ranking, {"a", "d"}, k=2) == 0.5
    assert recall_at_k(ranking, {"z"}, k=4) == 0.0


def test_recall_with_no_relevant_documents_is_zero_not_an_error():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "b"}, k=4) == 0.5
    assert precision_at_k(["a", "b", "c", "d"], {"a", "b"}, k=2) == 1.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_penalises_relevant_documents_ranked_lower():
    top = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    bottom = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert top == pytest.approx(1.0)
    assert bottom < top


def test_ndcg_matches_hand_computed_value():
    # One relevant document at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_score_query_returns_all_metrics_at_each_k():
    scores = score_query(["a", "b"], ["a"], ks=(1, 3))
    assert set(scores) == {"mrr", "recall@1", "precision@1", "ndcg@1",
                           "recall@3", "precision@3", "ndcg@3"}
    assert scores["recall@1"] == 1.0
    assert scores["mrr"] == 1.0


def test_score_query_deduplicates_chunks_before_scoring():
    # Five chunks from one document must not count as five retrieved documents.
    scores = score_query(["a", "a", "a", "a", "a"], ["a", "b"], ks=(5,))
    assert scores["recall@5"] == 0.5
    assert scores["precision@5"] == pytest.approx(1 / 5)


def test_aggregate_averages_across_queries():
    per_query = [{"recall@5": 1.0}, {"recall@5": 0.0}, {"recall@5": 0.5}]
    assert aggregate(per_query)["recall@5"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}


def test_bootstrap_ci_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0] * 6
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values, iterations=2000)
    assert low <= mean <= high


def test_bootstrap_ci_is_narrower_with_more_data():
    few = bootstrap_ci([0.0, 1.0] * 5, iterations=2000)
    many = bootstrap_ci([0.0, 1.0] * 100, iterations=2000)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    values = [0.1, 0.7, 0.3, 0.9]
    assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)


def test_bootstrap_ci_of_a_constant_has_zero_width():
    low, high = bootstrap_ci([0.5] * 20, iterations=500)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_bootstrap_ci_handles_empty_and_single_values():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
