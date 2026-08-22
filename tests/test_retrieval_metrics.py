import math

import pytest

from lexgraph.eval.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    ndcg_at_k,
    paired_bootstrap,
    paragraph_recall_at_k,
    parse_para_span,
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


# --- paragraph-level ground truth --------------------------------------------

def test_parse_para_span_reads_both_label_shapes():
    assert parse_para_span("para 12-15") == (12, 15)
    assert parse_para_span("para 7") == (7, 7)
    assert parse_para_span("") is None
    assert parse_para_span("chunk 4") is None


def test_paragraph_recall_needs_the_right_passage_not_just_the_right_case():
    gold = {"a.txt": [12]}
    # The correct judgment, at a paragraph that does not carry the answer.
    assert paragraph_recall_at_k([("a.txt", "para 3")], gold, 5) == 0.0
    # Document-level recall would have scored this 1.0.
    assert paragraph_recall_at_k([("a.txt", "para 11-14")], gold, 5) == 1.0


def test_paragraph_recall_counts_each_document_once():
    gold = {"a.txt": [2], "b.txt": [9]}
    hits = [("a.txt", "para 1-3"), ("a.txt", "para 2"), ("c.txt", "para 9")]
    assert paragraph_recall_at_k(hits, gold, 5) == 0.5


def test_paragraph_recall_respects_the_cutoff():
    gold = {"a.txt": [8]}
    hits = [("z.txt", "para 1"), ("y.txt", "para 1"), ("a.txt", "para 8")]
    assert paragraph_recall_at_k(hits, gold, 2) == 0.0
    assert paragraph_recall_at_k(hits, gold, 3) == 1.0


def test_unlabelled_query_is_skipped_not_scored_zero():
    # Six judgments carry no usable numbering. Scoring their questions zero
    # would report a retrieval failure that never happened.
    assert paragraph_recall_at_k([("a.txt", "para 1")], {}, 5) is None


def test_aggregate_skips_missing_metrics_rather_than_failing():
    rows = [{"recall@5": 1.0, "paragraph_recall@5": 0.5}, {"recall@5": 0.0}]
    means = aggregate(rows)
    assert means["recall@5"] == 0.5
    assert means["paragraph_recall@5"] == 0.5, "averaged over the row that has it"


# --- paired comparison -------------------------------------------------------

def test_paired_bootstrap_detects_a_consistent_small_gain():
    # A gain that is small but present on nearly every question. Comparing two
    # separate confidence intervals would call this a tie, because the spread
    # across questions dwarfs the shift between configurations.
    baseline = [0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.5]
    candidate = [b + 0.05 for b in baseline]

    stats = paired_bootstrap(baseline, candidate)
    assert stats["mean_difference"] == pytest.approx(0.05)
    assert stats["significant"], "a consistent gain must be distinguishable"
    assert stats["wins"] == 10 and stats["losses"] == 0

    wide_a = bootstrap_ci(baseline)
    wide_b = bootstrap_ci(candidate)
    assert wide_a[1] > wide_b[0], "the separate intervals do overlap; that is the point"


def test_paired_bootstrap_calls_noise_a_tie():
    baseline = [0.2, 0.9, 0.4, 0.7, 0.1, 0.8, 0.3, 0.6]
    candidate = [0.9, 0.2, 0.7, 0.4, 0.8, 0.1, 0.6, 0.3]
    assert not paired_bootstrap(baseline, candidate)["significant"]


def test_paired_bootstrap_counts_wins_losses_and_ties():
    stats = paired_bootstrap([0.5, 0.5, 0.5], [0.9, 0.1, 0.5])
    assert (stats["wins"], stats["losses"], stats["ties"]) == (1, 1, 1)


def test_paired_bootstrap_requires_matched_lengths():
    # Unequal lengths mean the scores are not from the same questions, and
    # pairing them would silently compare question 3 against question 4.
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_paired_bootstrap_handles_no_questions():
    assert paired_bootstrap([], [])["mean_difference"] == 0.0

