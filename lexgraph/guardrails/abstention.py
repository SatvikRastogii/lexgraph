"""Refuse to answer when retrieval did not find support.

The pipeline already computed a retrieval confidence and then answered anyway,
printing the warning beside the answer. That is not a guardrail: on a question
the corpus cannot answer, the generator still produces fluent text, and a
caveat under it does not stop anyone reading the answer.

Here the decision is made before generation, and the threshold is calibrated
against the gold set's out-of-corpus questions rather than picked by feel.
Because score scales differ by retriever (cosine sits near 0-1, BM25 is
unbounded, cross-encoder scores are their own thing), a threshold is only
meaningful for the configuration it was calibrated on.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..retrieval.base import Hit


@dataclass
class AbstentionDecision:
    should_answer: bool
    confidence: float
    reason: str = ""

    @property
    def refusal_message(self) -> str:
        return (
            "The indexed judgments do not contain material that answers this "
            "question, so no answer is given. Answering from the model's general "
            "knowledge would not be grounded in the corpus."
        )


def retrieval_confidence(hits: list[Hit], top_n: int = 3) -> float:
    """Blend the best score with the mean of the top few.

    The maximum alone rewards a single lucky match; the mean alone is dragged
    down by the long tail of a deep shortlist. Weighting toward the maximum
    reflects that one strongly relevant passage is often enough to answer.
    """
    if not hits:
        return 0.0
    scores = [hit.score for hit in hits[:top_n]]
    return 0.6 * max(scores) + 0.4 * (sum(scores) / len(scores))


def assess(hits: list[Hit], threshold: float, top_n: int = 3) -> AbstentionDecision:
    """Decide whether retrieval is strong enough to answer from."""
    if not hits:
        return AbstentionDecision(False, 0.0, "no documents retrieved")

    confidence = retrieval_confidence(hits, top_n)
    if confidence < threshold:
        return AbstentionDecision(
            False,
            confidence,
            f"retrieval confidence {confidence:.3f} is below the calibrated "
            f"threshold {threshold:.3f}",
        )
    return AbstentionDecision(True, confidence, "")


MIN_ANSWER_RATE = 0.80


def _sweep(answerable_confidences, unanswerable_confidences):
    """Every candidate threshold with the rates it achieves."""
    observed = sorted(set(answerable_confidences + unanswerable_confidences))
    candidates = [
        (observed[i] + observed[i + 1]) / 2 for i in range(len(observed) - 1)
    ] or [observed[0]]

    points = []
    for threshold in candidates:
        answered = sum(1 for c in answerable_confidences if c >= threshold)
        refused = sum(1 for c in unanswerable_confidences if c < threshold)
        sensitivity = answered / len(answerable_confidences)
        specificity = refused / len(unanswerable_confidences)
        points.append({
            "threshold": threshold,
            "answered_when_answerable": sensitivity,
            "refused_when_unanswerable": specificity,
            "youden_j": sensitivity + specificity - 1,
        })
    return points


def calibrate_threshold(
    answerable_confidences: list[float],
    unanswerable_confidences: list[float],
    min_answer_rate: float | None = MIN_ANSWER_RATE,
) -> tuple[float, dict]:
    """Pick an operating point on the abstention signal.

    Sweeps every midpoint between observed confidences. Which point to take is
    a choice about cost, and this is where the first version of this function
    was quietly wrong: it maximised Youden's J, which weights both errors
    equally, and nothing in the system says they are.

    They are not. A refused question the corpus can answer is a visible failure
    the user sees immediately. An answered out-of-corpus question is caught
    downstream -- citations are verified against the retrieved context, so a
    fabricated authority is already detected. Equal weighting therefore buys
    specificity that is partly redundant with a cheaper guardrail, and pays for
    it in the failure that is not.

    That mattered little while the gold set was easy: J-optimal answered every
    answerable question. On the paraphrase tier a genuine question phrased in
    lay language scores much like an out-of-corpus one, the two populations
    overlap, and J-optimal collapses to answering 44% of real questions. The
    signal did not get worse -- the questions got harder and exposed what the
    criterion was really choosing.

    So ``min_answer_rate`` fixes the floor on answering real questions, and
    among the thresholds that clear it the most specific one wins. Pass None
    for the unconstrained J-optimal point, which is reported alongside either
    way so the trade-off stays visible rather than baked in.
    """
    if not answerable_confidences or not unanswerable_confidences:
        raise ValueError("both answerable and unanswerable confidences are required")

    points = _sweep(answerable_confidences, unanswerable_confidences)
    youden = max(points, key=lambda p: p["youden_j"])

    if min_answer_rate is None:
        return youden["threshold"], {**youden, "criterion": "youden_j"}

    eligible = [
        p for p in points if p["answered_when_answerable"] >= min_answer_rate
    ]
    if not eligible:
        # No threshold answers enough real questions. Reporting the J-optimal
        # point here would hide that; the caller needs to know the constraint
        # could not be met on this signal.
        return youden["threshold"], {
            **youden,
            "criterion": "youden_j",
            "constraint_unmet": min_answer_rate,
        }

    chosen = max(
        eligible,
        key=lambda p: (p["refused_when_unanswerable"], p["answered_when_answerable"]),
    )
    return chosen["threshold"], {
        **chosen,
        "criterion": f"most specific threshold answering >= {min_answer_rate:.0%}",
        "youden_j_alternative": {
            "threshold": youden["threshold"],
            "answered_when_answerable": youden["answered_when_answerable"],
            "refused_when_unanswerable": youden["refused_when_unanswerable"],
            "youden_j": youden["youden_j"],
        },
    }


def separation(
    answerable_confidences: list[float],
    unanswerable_confidences: list[float],
) -> float:
    """Gap between the mean confidence of answerable and unanswerable questions.

    A gap near zero means the retriever gives out-of-corpus questions the same
    confidence as real ones, so no threshold can separate them and abstention
    cannot work on this signal at all. Worth reporting alongside the threshold,
    because a calibrated threshold on an unseparated signal is theatre.
    """
    if not answerable_confidences or not unanswerable_confidences:
        return 0.0
    mean_answerable = sum(answerable_confidences) / len(answerable_confidences)
    mean_unanswerable = sum(unanswerable_confidences) / len(unanswerable_confidences)
    return mean_answerable - mean_unanswerable
