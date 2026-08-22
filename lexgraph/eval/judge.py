"""LLM-as-judge scoring, arranged so the scores mean something.

Four changes from the original evaluation, each addressing a specific way the
old numbers were inflated:

1. The judge is a different model from the generator. Scoring llama3.1's
   answers with llama3.1 is self-evaluation. ``require_independent_judge``
   makes a self-judged run fail rather than quietly report better numbers.

2. Citation accuracy is measured, not judged. It is a question about whether
   strings in the answer appear in the context, which a regex answers exactly
   and an LLM answers approximately. Asking a model to grade it added noise
   and, worse, let a fabricated citation be scored generously.

3. Answer length travels with every score. LLM judges reward verbosity, and
   the two pipelines produce very differently sized answers, so a score gap
   that is really a length gap must be visible in the output.

4. Scores carry the judge's stated reason. A metric nobody can audit is a
   metric nobody should trust.

Prompts ask for a single JSON object and describe each point on the scale, so
parsing is deterministic and the judge is anchored rather than inventing its
own rubric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..guardrails.citations import verify_citations
from ..llm import BaseClient, judge_is_independent

SCALE_MIN, SCALE_MAX = 1, 5


@dataclass
class MetricScore:
    score: float
    reason: str = ""
    parsed: bool = True

    def as_dict(self) -> dict:
        return {"score": self.score, "reason": self.reason, "parsed": self.parsed}


@dataclass
class PipelineScores:
    label: str
    question: str
    answer_chars: int
    answer_words: int
    metrics: dict[str, MetricScore] = field(default_factory=dict)

    def mean(self) -> float:
        scores = [m.score for m in self.metrics.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "question": self.question,
            "answer_chars": self.answer_chars,
            "answer_words": self.answer_words,
            "mean": self.mean(),
            "metrics": {name: m.as_dict() for name, m in self.metrics.items()},
        }


class SelfJudgingError(RuntimeError):
    """Raised when the judge and the generator are the same model."""


def require_independent_judge(generator_spec: str, judge_spec: str) -> None:
    if not judge_is_independent(generator_spec, judge_spec):
        raise SelfJudgingError(
            f"judge ({judge_spec}) is the same model as the generator "
            f"({generator_spec}). Scores from a self-judged run are not "
            f"comparable; choose a different judge model."
        )


def parse_score(response: str) -> MetricScore:
    """Extract ``{\"score\": n, \"reason\": \"...\"}`` from a judge response.

    Falls back to the first standalone digit in range. When nothing parses the
    result is flagged rather than silently defaulting to a middling 3, which
    would quietly pull every average toward the centre.
    """
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            raw = payload.get("score")
            if raw is not None:
                score = max(SCALE_MIN, min(SCALE_MAX, int(float(raw))))
                return MetricScore(score, str(payload.get("reason", "")).strip())
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    digit = re.search(r"\b([1-5])\b", response)
    if digit:
        return MetricScore(int(digit.group(1)), response[:160].strip())

    return MetricScore(0.0, f"unparseable judge response: {response[:120]}", parsed=False)


_RUBRIC = {
    "faithfulness": (
        "FAITHFULNESS: is every claim in the answer supported by the context?",
        "5 every claim is directly supported | 4 mostly supported, minor unsupported additions | "
        "3 significant portions ungrounded | 2 few claims supported | 1 unsupported or contradicts context",
        True,
    ),
    "answer_relevancy": (
        "ANSWER RELEVANCY: does the answer address the question asked?",
        "5 directly and completely | 4 addresses it, misses minor aspects | 3 partially | "
        "2 tangential | 1 irrelevant",
        False,
    ),
    "context_precision": (
        "CONTEXT PRECISION: are the retrieved passages relevant to the question?",
        "5 all directly relevant | 4 mostly relevant, minor noise | 3 about half | "
        "2 mostly noise | 1 none relevant",
        True,
    ),
    "completeness": (
        "COMPLETENESS: how fully does the answer cover the question?",
        "5 covers all aspects with specifics | 4 covers most | 3 misses notable aspects | "
        "2 superficial | 1 minimal or empty. Judge coverage, NOT length: a concise "
        "answer that covers everything scores 5.",
        False,
    ),
    "hallucination": (
        "HALLUCINATION: does the answer invent cases, dates, holdings or provisions?",
        "5 nothing fabricated | 4 one trivial unsupported detail | 3 some invented specifics | "
        "2 several fabrications | 1 largely fabricated. HIGHER IS BETTER (5 = no hallucination).",
        True,
    ),
    "coherence": (
        "COHERENCE: is the answer logically ordered and readable?",
        "5 clear structure, follows through | 4 mostly clear | 3 loose but followable | "
        "2 disjointed | 1 incoherent",
        False,
    ),
    "legal_reasoning": (
        "LEGAL REASONING: does the answer reason like a lawyer -- identifying the "
        "holding, distinguishing authority, applying law to facts?",
        "5 sound legal analysis | 4 competent with gaps | 3 describes without analysing | "
        "2 superficial | 1 no legal reasoning",
        False,
    ),
}


def parse_batch_scores(response: str, metrics: list[str]) -> dict[str, MetricScore]:
    """Parse one JSON object holding a score per metric.

    Each metric is recovered independently, so one malformed entry costs that
    metric and not the whole question.
    """
    scores: dict[str, MetricScore] = {}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    payload = {}
    if match:
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            payload = {}

    # Whole-object parsing fails outright when the response is truncated at the
    # token limit: there is no closing brace, so the greedy match above finds
    # nothing and every metric is lost at once. Recovering each metric's own
    # object individually means a cut-off response costs only the metrics that
    # were actually cut off. Measured at ~12% of questions before this.
    if not payload:
        payload = _salvage_per_metric(response, metrics)

    for metric in metrics:
        entry = payload.get(metric)
        if isinstance(entry, dict) and entry.get("score") is not None:
            try:
                value = max(SCALE_MIN, min(SCALE_MAX, int(float(entry["score"]))))
                scores[metric] = MetricScore(value, str(entry.get("reason", "")).strip())
                continue
            except (ValueError, TypeError):
                pass
        if isinstance(entry, (int, float)):
            scores[metric] = MetricScore(max(SCALE_MIN, min(SCALE_MAX, int(entry))))
            continue
        scores[metric] = MetricScore(
            0.0, f"missing or unparseable in judge response: {response[:100]}", parsed=False
        )
    return scores


def _salvage_per_metric(response: str, metrics: list[str]) -> dict:
    """Recover individual metric objects from a malformed or truncated response."""
    recovered = {}
    for metric in metrics:
        entry = re.search(
            rf'"{re.escape(metric)}"\s*:\s*(\{{[^{{}}]*\}})', response, re.DOTALL
        )
        if entry:
            try:
                recovered[metric] = json.loads(entry.group(1))
                continue
            except json.JSONDecodeError:
                pass
        # Even the object may be cut mid-string; the score comes first, so it
        # usually survives when the reason does not.
        bare = re.search(rf'"{re.escape(metric)}"\s*:\s*\{{\s*"score"\s*:\s*([1-5])', response)
        if bare:
            recovered[metric] = {"score": int(bare.group(1)), "reason": "(reason truncated)"}
    return recovered


def _build_batch_prompt(
    question: str, answer: str, contexts: list[str], metrics: list[str]
) -> str:
    """One prompt scoring every metric.

    Scoring each metric in its own request meant seven calls per answer, or
    490 for a two-generator sweep over the gold set -- past any free-tier
    daily quota, and seven times the latency for no measurable benefit. The
    rubric is stated in full either way, so the judge is anchored the same.
    """
    rubric_lines = []
    for metric in metrics:
        title, scale, _ = _RUBRIC[metric]
        rubric_lines.append(f"{metric}\n  {title}\n  {scale}")

    joined = "\n\n---\n\n".join(contexts[:4]) if contexts else "(no context retrieved)"
    schema = ", ".join(f'"{m}": {{"score": <1-5>, "reason": "<one sentence>"}}' for m in metrics)

    return (
        "You are an impartial evaluation judge for a legal question-answering "
        "system. Score the answer below on each metric independently, on a 1-5 "
        "scale. Judge each metric on its own terms; do not let one score pull "
        "the others.\n\n"
        f"METRICS:\n{chr(10).join(rubric_lines)}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT PROVIDED TO THE SYSTEM:\n{joined[:6000]}\n\n"
        f"ANSWER BEING EVALUATED:\n{answer[:4000]}\n\n"
        f"Respond with ONLY a JSON object of exactly this shape:\n{{{schema}}}"
    )


def score_pipeline(
    label: str,
    question: str,
    answer: str,
    contexts: list[str],
    judge: BaseClient,
    metrics: list[str] | None = None,
) -> PipelineScores:
    """Score one pipeline's answer across every metric in a single judge call."""
    result = PipelineScores(
        label=label,
        question=question,
        answer_chars=len(answer),
        answer_words=len(answer.split()),
    )

    requested = list(metrics or _RUBRIC)
    if not contexts:
        requested = [m for m in requested if m != "context_precision"]

    if requested:
        response = judge.chat(
            _build_batch_prompt(question, answer, contexts, requested),
            max_tokens=3000,
            temperature=0.0,
        )
        result.metrics.update(parse_batch_scores(response, requested))

    # Deterministic, not judged.
    report = verify_citations(answer, contexts)
    result.metrics["citation_accuracy"] = MetricScore(
        score=round(SCALE_MIN + report.accuracy * (SCALE_MAX - SCALE_MIN), 2),
        reason=(
            f"{len(report.supported)}/{report.total} citations found in the retrieved "
            f"context" + (f"; unsupported: {', '.join(report.unsupported[:3])}"
                          if report.unsupported else "")
        ),
    )
    return result
