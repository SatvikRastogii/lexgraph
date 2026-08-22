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


def calibrate_threshold(
    answerable_confidences: list[float],
    unanswerable_confidences: list[float],
) -> tuple[float, dict]:
    """Pick the threshold that best separates answerable from unanswerable.

    Sweeps every midpoint between observed confidences and maximises Youden's J
    (sensitivity + specificity - 1), which weights both errors equally --
    answering an unanswerable question and refusing an answerable one are both
    failures here.

    Returns the threshold and the statistics it achieves.
    """
    if not answerable_confidences or not unanswerable_confidences:
        raise ValueError("both answerable and unanswerable confidences are required")

    observed = sorted(set(answerable_confidences + unanswerable_confidences))
    candidates = [
        (observed[i] + observed[i + 1]) / 2 for i in range(len(observed) - 1)
    ] or [observed[0]]

    best_threshold, best_j, best_stats = candidates[0], -1.0, {}
    for threshold in candidates:
        answered = sum(1 for c in answerable_confidences if c >= threshold)
        refused = sum(1 for c in unanswerable_confidences if c < threshold)
        sensitivity = answered / len(answerable_confidences)
        specificity = refused / len(unanswerable_confidences)
        youden_j = sensitivity + specificity - 1
        if youden_j > best_j:
            best_j = youden_j
            best_threshold = threshold
            best_stats = {
                "threshold": threshold,
                "answered_when_answerable": sensitivity,
                "refused_when_unanswerable": specificity,
                "youden_j": youden_j,
            }

    return best_threshold, best_stats


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
