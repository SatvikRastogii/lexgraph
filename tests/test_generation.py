import pytest

from lexgraph.generation import answer_question
from lexgraph.retrieval.base import Hit

CONTEXT = (
    "Sunil Batra vs Delhi Administration. The Court held under Article 21 that "
    "a prisoner retains his fundamental rights during confinement."
)


class FakeRetriever:
    def search(self, query, top_k=5):
        return [Hit(chunk_id="c1", doc_id="judgment_0126.txt", text=CONTEXT, score=0.9)]


class ScriptedGenerator:
    """Returns each scripted answer in turn, recording the prompts it saw."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def chat(self, prompt, max_tokens=512, temperature=0.0):
        self.prompts.append(prompt)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


GROUNDED = "Sunil Batra vs Delhi Administration held that Article 21 applies."
FABRICATED = "In Ramesh Kumar vs State of Bihar the Court held under Article 47 that..."


def test_warn_returns_the_answer_and_reports_the_problem():
    generator = ScriptedGenerator(FABRICATED)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="warn")

    assert result.answer == FABRICATED, "warn must not alter the answer"
    assert result.citations.has_unsupported
    assert not result.abstained
    assert len(generator.prompts) == 1, "warn must not spend a second call"


def test_retry_regenerates_and_keeps_the_cleaner_answer():
    generator = ScriptedGenerator(FABRICATED, GROUNDED)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="retry")

    assert result.answer == GROUNDED
    assert result.citation_retry
    assert not result.citations.has_unsupported


def test_retry_names_the_offending_citations_in_its_prompt():
    # "Only cite the extracts" is the instruction the model already broke.
    generator = ScriptedGenerator(FABRICATED, GROUNDED)
    answer_question("q", FakeRetriever(), generator, citation_policy="retry")
    assert "Ramesh Kumar" in generator.prompts[1]


def test_retry_keeps_the_original_when_the_rewrite_is_no_better():
    # A retry is free to make things worse; accepting it unconditionally would
    # let the guardrail lower citation accuracy while appearing to enforce it.
    worse = "In A vs B and C vs D and E vs F the Court held under Article 99..."
    generator = ScriptedGenerator(FABRICATED, worse)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="retry")
    assert result.answer == FABRICATED


def test_retry_is_not_triggered_by_a_clean_answer():
    generator = ScriptedGenerator(GROUNDED)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="retry")
    assert len(generator.prompts) == 1
    assert not result.citation_retry


def test_refuse_withholds_an_answer_with_fabricated_citations():
    generator = ScriptedGenerator(FABRICATED)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="refuse")

    assert result.abstained
    assert FABRICATED not in result.answer
    # The report survives the refusal, so the reason stays inspectable.
    assert result.citations.has_unsupported


def test_refuse_lets_a_grounded_answer_through():
    generator = ScriptedGenerator(GROUNDED)
    result = answer_question("q", FakeRetriever(), generator, citation_policy="refuse")
    assert not result.abstained
    assert result.answer == GROUNDED


def test_unknown_policy_is_rejected_before_any_generation():
    generator = ScriptedGenerator(GROUNDED)
    with pytest.raises(ValueError):
        answer_question("q", FakeRetriever(), generator, citation_policy="ignore")
    assert not generator.prompts
