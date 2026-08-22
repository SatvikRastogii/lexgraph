import pytest

from lexgraph.eval.judge import (
    SelfJudgingError,
    parse_score,
    require_independent_judge,
    score_pipeline,
)
from lexgraph.llm import judge_is_independent

ALL_METRICS = [
    "faithfulness", "answer_relevancy", "context_precision",
    "completeness", "hallucination", "coherence", "legal_reasoning",
]


def batch_response(score=4, metrics=None):
    """A well-formed batched judge response."""
    body = ", ".join(
        f'"{m}": {{"score": {score}, "reason": "ok"}}' for m in (metrics or ALL_METRICS)
    )
    return f"{{{body}}}"


class FakeJudge:
    """Returns canned responses so judge logic is testable without a network."""

    def __init__(self, response=None):
        self.response = response if response is not None else batch_response()
        self.prompts = []

    def chat(self, prompt, max_tokens=1600, temperature=0.0):
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
    judge = FakeJudge(batch_response(score=5, metrics=["faithfulness"]))
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


def test_all_metrics_are_scored_in_one_call():
    # Seven calls per answer put a two-generator sweep past any free-tier
    # daily quota; one call does the same work.
    judge = FakeJudge()
    scores = score_pipeline("naive", "Q?", "A.", ["ctx"], judge)
    assert len(judge.prompts) == 1
    assert set(ALL_METRICS) <= set(scores.metrics)


def test_one_malformed_metric_does_not_lose_the_others():
    judge = FakeJudge(
        '{"faithfulness": {"score": 4, "reason": "ok"}, "coherence": "broken"}'
    )
    scores = score_pipeline(
        "naive", "Q?", "A.", ["ctx"], judge, metrics=["faithfulness", "coherence"]
    )
    assert scores.metrics["faithfulness"].score == 4
    assert scores.metrics["coherence"].parsed is False


def test_completeness_rubric_warns_the_judge_against_length_bias():
    judge = FakeJudge()
    score_pipeline("naive", "Q?", "A.", [], judge, metrics=["completeness"])
    assert "NOT length" in judge.prompts[0]


def test_hallucination_rubric_states_its_direction():
    judge = FakeJudge()
    score_pipeline("naive", "Q?", "A.", ["ctx"], judge, metrics=["hallucination"])
    assert "HIGHER IS BETTER" in judge.prompts[0]


def test_mean_ignores_nothing_and_matches_hand_computation():
    judge = FakeJudge(batch_response(score=4, metrics=["faithfulness", "coherence"]))
    scores = score_pipeline(
        "naive", "Q?", "Article 21 applies.",
        ["Article 21 protects life."], judge,
        metrics=["faithfulness", "coherence"],
    )
    # faithfulness 4, coherence 4, citation_accuracy 5 -> mean 13/3
    assert scores.mean() == pytest.approx(13 / 3)


def test_truncated_response_salvages_the_metrics_that_survived():
    # A response cut off at the token limit has no closing brace, so whole
    # object parsing finds nothing. The metrics before the cut must survive.
    truncated = (
        '{"faithfulness": {"score": 4, "reason": "grounded"}, '
        '"coherence": {"score": 5, "reason": "clear"}, '
        '"completeness": {"score": 3, "reas'
    )
    judge = FakeJudge(truncated)
    scores = score_pipeline(
        "naive", "Q?", "A.", ["ctx"], judge,
        metrics=["faithfulness", "coherence", "completeness"],
    )
    assert scores.metrics["faithfulness"].score == 4
    assert scores.metrics["coherence"].score == 5
    # completeness was cut mid-object but its score came first, so it survives.
    assert scores.metrics["completeness"].score == 3


def test_response_truncated_before_any_score_is_flagged():
    judge = FakeJudge('{"faithfulness": {"sco')
    scores = score_pipeline("naive", "Q?", "A.", ["ctx"], judge, metrics=["faithfulness"])
    assert scores.metrics["faithfulness"].parsed is False


# --- quota handling ----------------------------------------------------------

def test_daily_quota_is_distinguished_from_a_per_minute_limit():
    from lexgraph.llm import is_daily_quota_error

    daily = Exception(
        '429: {"error":{"details":[{"violations":'
        '[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}'
    )
    per_minute = Exception(
        '429: {"error":{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel"}}'
    )
    server = Exception("500: internal error")

    # Waiting fixes a per-minute limit and never fixes a per-day one, so only
    # the daily case should short-circuit the retry loop.
    assert is_daily_quota_error(daily) is True
    assert is_daily_quota_error(per_minute) is False
    assert is_daily_quota_error(server) is False


def test_fallback_retires_a_model_only_on_quota():
    from lexgraph.llm import FallbackClient

    class Stub:
        def __init__(self, label, error=None):
            self.label, self.error, self.calls = label, error, 0
            self.model = label

        def chat(self, prompt, max_tokens=512, temperature=0.0):
            self.calls += 1
            if self.error:
                raise self.error
            return "ok"

    exhausted = Stub("a", RuntimeError("429 quota exceeded"))
    flaky = Stub("b", RuntimeError("500 server error"))
    healthy = Stub("c")

    client = FallbackClient([exhausted, flaky, healthy])
    assert client.chat("q") == "ok"
    assert "a" in client.exhausted, "a quota failure retires the model"
    assert "b" not in client.exhausted, "a transient failure must not retire it"
