import pytest

from lexgraph.eval.judge import (
    SelfJudgingError,
    parse_score,
    require_independent_judge,
    score_pipeline,
)
from lexgraph.llm import judge_is_independent


class FakeJudge:
    """Returns canned responses so judge logic is testable without a network."""

    def __init__(self, response='{"score": 4, "reason": "well grounded"}'):
        self.response = response
        self.prompts = []

    def chat(self, prompt, max_tokens=200, temperature=0.0):
        self.prompts.append(prompt)
        return self.response


# --- score parsing -----------------------------------------------------------

def test_parses_clean_json():
    result = parse_score('{"score": 4, "reason": "mostly supported"}')
    assert result.score == 4
    assert result.reason == "mostly supported"
    assert result.parsed


def test_parses_json_wrapped_in_prose():
    result = parse_score('Sure!\n```json\n{"score": 5, "reason": "exact"}\n```\nDone.')
    assert result.score == 5


def test_clamps_out_of_range_scores():
    assert parse_score('{"score": 9}').score == 5
    assert parse_score('{"score": -3}').score == 1


def test_accepts_float_scores():
    assert parse_score('{"score": 4.0}').score == 4


def test_falls_back_to_a_bare_digit():
    result = parse_score("I would rate this a 3 out of 5.")
    assert result.score == 3


def test_unparseable_response_is_flagged_not_defaulted():
    # Silently returning 3 would drag every average toward the centre and
    # hide judge failures completely.
    result = parse_score("I cannot evaluate this content.")
    assert result.parsed is False
    assert result.score == 0.0


def test_malformed_json_does_not_raise():
    result = parse_score('{"score": }')
    assert result.parsed in (True, False)


# --- judge independence ------------------------------------------------------

def test_same_model_is_not_independent():
    assert not judge_is_independent("ollama:llama3.1", "ollama:llama3.1")
    assert not judge_is_independent("ollama:llama3.1", "OLLAMA:LLAMA3.1")


def test_different_models_are_independent():
    assert judge_is_independent("ollama:llama3.1", "gemini:gemini-2.5-flash")


def test_self_judging_run_is_refused():
    with pytest.raises(SelfJudgingError):
        require_independent_judge("ollama:llama3.1", "ollama:llama3.1")


def test_independent_judge_is_allowed():
    require_independent_judge("ollama:llama3.1", "gemini:gemini-2.5-flash")


# --- scoring -----------------------------------------------------------------

def test_score_pipeline_records_answer_length():
    scores = score_pipeline(
        "naive", "What is a curative petition?",
        "A curative petition is an extraordinary remedy.",
        ["A curative petition may be entertained."],
        FakeJudge(), metrics=["faithfulness"],
    )
    assert scores.answer_words == 7
    assert scores.answer_chars == len("A curative petition is an extraordinary remedy.")


def test_citation_accuracy_is_measured_not_judged():
    judge = FakeJudge('{"score": 5, "reason": "perfect"}')
    scores = score_pipeline(
        "naive", "Which article applies?",
        "The answer relies on Article 99, which is decisive.",
        ["Article 21 protects life and personal liberty."],
        judge, metrics=["faithfulness"],
    )
    # The judge said 5 for its metric, but Article 99 is absent from context,
    # so the deterministic citation score must be the floor regardless.
    assert scores.metrics["citation_accuracy"].score == 1.0
    assert "Article 99" in scores.metrics["citation_accuracy"].reason


def test_perfect_citations_score_top_of_scale():
    scores = score_pipeline(
        "naive", "Which article applies?",
        "Article 21 applies here.",
        ["Article 21 protects life and personal liberty."],
        FakeJudge(), metrics=["faithfulness"],
    )
    assert scores.metrics["citation_accuracy"].score == 5.0


def test_context_precision_is_skipped_without_context():
    judge = FakeJudge()
    scores = score_pipeline(
        "graph", "A question?", "An answer.", [], judge,
        metrics=["faithfulness", "context_precision"],
    )
    assert "context_precision" not in scores.metrics
    assert "faithfulness" in scores.metrics


def test_context_is_only_sent_for_metrics_that_need_it():
    judge = FakeJudge()
    score_pipeline(
        "naive", "Q?", "A.", ["SECRET_CONTEXT_MARKER"], judge,
        metrics=["answer_relevancy"],
    )
    assert "SECRET_CONTEXT_MARKER" not in judge.prompts[0]


def test_completeness_rubric_warns_the_judge_against_length_bias():
    judge = FakeJudge()
    score_pipeline("naive", "Q?", "A.", [], judge, metrics=["completeness"])
    assert "NOT length" in judge.prompts[0]


def test_hallucination_rubric_states_its_direction():
    judge = FakeJudge()
    score_pipeline("naive", "Q?", "A.", ["ctx"], judge, metrics=["hallucination"])
    assert "HIGHER IS BETTER" in judge.prompts[0]


def test_mean_ignores_nothing_and_matches_hand_computation():
    judge = FakeJudge('{"score": 4, "reason": "ok"}')
    scores = score_pipeline(
        "naive", "Q?", "Article 21 applies.",
        ["Article 21 protects life."], judge,
        metrics=["faithfulness", "coherence"],
    )
    # faithfulness 4, coherence 4, citation_accuracy 5 -> mean 13/3
    assert scores.mean() == pytest.approx(13 / 3)
