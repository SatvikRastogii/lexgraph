import pytest

from lexgraph.guardrails.abstention import (
    assess,
    calibrate_threshold,
    retrieval_confidence,
    separation,
)
from lexgraph.guardrails.citations import (
    extract_citations,
    verify_citations,
)
from lexgraph.retrieval.base import Hit

# --- citation extraction -----------------------------------------------------

def test_extracts_case_names_articles_and_sections():
    answer = (
        "In Maneka Gandhi vs Union of India the Court read Article 21 to require "
        "a fair procedure. Section 167(2) of the CrPC was considered separately."
    )
    found = extract_citations(answer)

    assert any("Maneka Gandhi" in c for c in found["cases"])
    assert "Article 21" in found["articles"]
    assert "Section 167(2)" in found["sections"]


def test_extracts_reported_citations():
    found = extract_citations("See AIR 1978 SC 597 and (2006) 4 SCC 1 for the position.")
    joined = " ".join(found["reported"])
    assert "AIR 1978 SC 597" in joined
    assert "(2006) 4 SCC 1" in joined


def test_ignores_pseudo_case_names_made_of_noise_words():
    found = extract_citations("The State vs The Court is not a real citation.")
    assert found["cases"] == []


def test_deduplicates_repeated_references():
    found = extract_citations("Article 21 matters. Article 21 again. Article 21 once more.")
    assert found["articles"] == ["Article 21"]


# --- citation verification ---------------------------------------------------

def test_citation_present_in_context_is_supported():
    answer = "The Court relied on Article 32 to issue the writ."
    contexts = ["Under Article 32 of the Constitution the Court may issue writs."]
    report = verify_citations(answer, contexts)

    assert report.supported == ["Article 32"]
    assert not report.has_unsupported
    assert report.accuracy == 1.0


def test_fabricated_case_is_flagged():
    answer = "This follows from Puttaswamy vs Union of India, which settled the point."
    contexts = ["A curative petition may be entertained in rare cases."]
    report = verify_citations(answer, contexts)

    assert report.has_unsupported
    assert any("Puttaswamy" in c for c in report.unsupported)
    assert report.accuracy == 0.0


def test_wrong_article_number_is_flagged():
    # The characteristic legal hallucination: right shape, wrong number.
    answer = "The right arises under Article 19."
    contexts = ["Article 21 protects life and personal liberty."]
    report = verify_citations(answer, contexts)
    assert "Article 19" in report.unsupported


def test_case_is_supported_by_a_single_distinctive_party_name():
    # Judgments routinely shorten the full case name, so requiring both sides
    # would flag correct citations.
    answer = "As held in Sunil Batra vs Delhi Administration, prisoners retain rights."
    contexts = ["In Sunil Batra the Court held that prisoners retain fundamental rights."]
    report = verify_citations(answer, contexts)
    assert not report.has_unsupported


def test_answer_without_citations_is_not_penalised():
    report = verify_citations("The petition was allowed.", ["Some judgment text."])
    assert report.total == 0
    assert report.accuracy == 1.0
    assert not report.has_unsupported


def test_warning_names_the_unsupported_citations():
    report = verify_citations("See Vishaka vs State of Rajasthan.", ["Unrelated text."])
    assert "Vishaka" in report.warning()


def test_mixed_answer_reports_partial_accuracy():
    answer = "Article 21 was applied, following Kesavananda Bharati vs State of Kerala."
    contexts = ["Article 21 protects life and personal liberty."]
    report = verify_citations(answer, contexts)
    assert report.supported == ["Article 21"]
    assert len(report.unsupported) == 1
    assert report.accuracy == pytest.approx(0.5)


# --- abstention --------------------------------------------------------------

def _hits(*scores):
    return [Hit(chunk_id=str(i), doc_id="d.txt", text="t", score=s)
            for i, s in enumerate(scores)]


def test_confidence_is_zero_without_hits():
    assert retrieval_confidence([]) == 0.0


def test_confidence_weights_the_best_hit_most():
    peaked = retrieval_confidence(_hits(0.9, 0.1, 0.1))
    flat = retrieval_confidence(_hits(0.4, 0.4, 0.4))
    assert peaked > flat


def test_assess_refuses_below_threshold_and_answers_above():
    assert assess(_hits(0.2, 0.1), threshold=0.5).should_answer is False
    assert assess(_hits(0.9, 0.8), threshold=0.5).should_answer is True


def test_assess_refuses_when_nothing_was_retrieved():
    decision = assess([], threshold=0.1)
    assert decision.should_answer is False
    assert "no documents" in decision.reason


def test_refusal_names_the_threshold_it_failed():
    decision = assess(_hits(0.2), threshold=0.6)
    assert "0.6" in decision.reason


def test_calibration_finds_a_separating_threshold():
    answerable = [0.80, 0.85, 0.90, 0.95]
    unanswerable = [0.10, 0.20, 0.30, 0.40]
    threshold, stats = calibrate_threshold(answerable, unanswerable)

    assert 0.40 < threshold < 0.80
    assert stats["answered_when_answerable"] == 1.0
    assert stats["refused_when_unanswerable"] == 1.0
    assert stats["youden_j"] == pytest.approx(1.0)


def test_calibration_degrades_honestly_on_overlapping_signal():
    # When the two populations overlap completely no threshold can separate
    # them, and Youden's J must reflect that rather than claiming success.
    overlapping = [0.5, 0.5, 0.5, 0.5]
    threshold, stats = calibrate_threshold(overlapping, list(overlapping))
    assert stats["youden_j"] < 0.5


def test_calibration_requires_both_populations():
    with pytest.raises(ValueError):
        calibrate_threshold([0.8], [])


def test_separation_is_positive_when_answerable_scores_higher():
    assert separation([0.9, 0.8], [0.2, 0.1]) == pytest.approx(0.7)


def test_separation_is_zero_for_identical_populations():
    assert separation([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_operating_point_keeps_answering_real_questions():
    # Populations that overlap the way the hard tier made them overlap: some
    # genuine questions score below some out-of-corpus ones.
    answerable = [0.30, 0.45, 0.55, 0.60, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    unanswerable = [0.20, 0.25, 0.35, 0.50, 0.65]

    constrained, stats = calibrate_threshold(answerable, unanswerable)
    assert stats["answered_when_answerable"] >= 0.80

    unconstrained, j_stats = calibrate_threshold(
        answerable, unanswerable, min_answer_rate=None
    )
    # Maximising Youden's J alone is free to refuse real questions to buy
    # specificity; the constrained point must not sit above it.
    assert constrained <= unconstrained
    assert j_stats["criterion"] == "youden_j"


def test_operating_point_reports_the_alternative_it_passed_over():
    answerable = [0.30, 0.60, 0.80, 0.90, 0.95]
    unanswerable = [0.20, 0.40, 0.70]
    _, stats = calibrate_threshold(answerable, unanswerable)
    assert "youden_j_alternative" in stats, "the trade-off must stay visible"


def test_unmeetable_constraint_is_flagged_not_silently_dropped():
    # An out-of-corpus question scoring between the two real ones: every
    # candidate threshold sits above one of them, so no operating point can
    # answer 80%. Falling back to Youden's J without saying so would hide that
    # the requested floor was never reached.
    answerable = [0.10, 0.90]
    unanswerable = [0.50]
    _, stats = calibrate_threshold(answerable, unanswerable, min_answer_rate=0.8)
    assert stats.get("constraint_unmet") == 0.8
