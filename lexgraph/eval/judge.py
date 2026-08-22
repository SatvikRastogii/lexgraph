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


def _build_prompt(metric: str, question: str, answer: str, contexts: list[str]) -> str:
    title, scale, needs_context = _RUBRIC[metric]
    parts = [
        "You are an impartial evaluation judge for a legal question-answering system.",
        f"Score the following on a 1-5 scale.\n\n{title}\n{scale}",
        f"\nQUESTION:\n{question}",
    ]
    if needs_context:
        joined = "\n\n---\n\n".join(contexts[:4]) if contexts else "(no context retrieved)"
        parts.append(f"\nCONTEXT PROVIDED TO THE SYSTEM:\n{joined[:6000]}")
    parts.append(f"\nANSWER BEING EVALUATED:\n{answer[:4000]}")
    parts.append(
        '\nRespond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}'
    )
    return "\n".join(parts)


def score_pipeline(
    label: str,
    question: str,
    answer: str,
    contexts: list[str],
    judge: BaseClient,
    metrics: list[str] | None = None,
) -> PipelineScores:
    """Score one pipeline's answer across every metric."""
    result = PipelineScores(
        label=label,
        question=question,
        answer_chars=len(answer),
        answer_words=len(answer.split()),
    )

    for metric in metrics or list(_RUBRIC):
        if metric == "context_precision" and not contexts:
            continue
        response = judge.chat(
            _build_prompt(metric, question, answer, contexts),
            max_tokens=200,
            temperature=0.0,
        )
        result.metrics[metric] = parse_score(response)

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
