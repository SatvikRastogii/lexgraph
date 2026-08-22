"""Scheduled regression check on retrieval and answer quality.

Two signals, deliberately different in cost:

Retrieval drift is free -- Recall@5 and nDCG@10 against the gold set need no
LLM at all, so the whole set runs every time and any regression in the
retriever is caught immediately.

Answer drift needs the judge, so it re-scores a small fixed sample. The sample
is fixed rather than sampled: comparability between runs matters more than
coverage when the question is "did anything change".

Scores are logged to logs/monitoring.db and compared against a rolling mean of
recent runs. Intended for a cron entry, not interactive use:

    docker compose exec app python scripts/drift_check.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telemetry
from lexgraph.eval.judge import score_pipeline
from lexgraph.eval.retrieval_metrics import aggregate, score_query
from lexgraph.generation import answer_question
from lexgraph.llm import DEFAULT_GENERATOR, build_judge, get_client
from lexgraph.retrieval.pipeline import build_retriever

GOLDSET_PATH = os.path.join("data", "goldset.json")
ROLLING_WINDOW = 5
DRIFT_ABS_THRESHOLD = 0.5   # points on the 1-5 scale, or 0.5 of a 0-1 metric
DRIFT_REL_THRESHOLD = 0.15  # or a 15% relative drop
SAMPLE_PER_CATEGORY = 1


def fixed_sample(questions, per_category=SAMPLE_PER_CATEGORY):
    """The first N questions of each category, so every run scores the same set."""
    seen, sample = {}, []
    for question in questions:
        category = question["category"]
        if seen.get(category, 0) < per_category:
            seen[category] = seen.get(category, 0) + 1
            sample.append(question)
    return sample


def rolling_baseline(metric, pipeline):
    history = telemetry.fetch_ragas_history(metric=metric, pipeline=pipeline, limit=10000)
    recent = history[-ROLLING_WINDOW:]
    if not recent:
        return None
    return round(sum(row["score"] for row in recent) / len(recent), 3)


def check(metric, pipeline, score):
    """Log one score and report whether it has drifted below the baseline."""
    baseline = rolling_baseline(metric, pipeline)
    drifted = False
    if baseline is not None and baseline > 0:
        drifted = (
            baseline - score >= DRIFT_ABS_THRESHOLD
            or (baseline - score) / baseline >= DRIFT_REL_THRESHOLD
        )
    telemetry.log_ragas_drift(metric, pipeline, score, baseline=baseline, drift_detected=drifted)
    if drifted:
        print(f"  DRIFT: {pipeline}/{metric} at {score:.3f}, baseline {baseline:.3f}")
    return drifted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="hybrid-rerank")
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--judge", default=None)
    parser.add_argument("--skip-answers", action="store_true",
                        help="retrieval drift only; no judge calls")
    args = parser.parse_args()

    telemetry.init_db()
    with open(GOLDSET_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    answerable = [q for q in questions if q["answerable"]]

    retriever = build_retriever(args.config)
    drifted = False

    # Retrieval: the whole gold set, no LLM.
    print(f"Retrieval drift over {len(answerable)} questions ({args.config})")
    per_question = [
        score_query(
            [hit.doc_id for hit in retriever.search(q["question"], 10)],
            q["relevant_docs"],
        )
        for q in answerable
    ]
    means = aggregate(per_question)
    for metric in ("recall@5", "ndcg@10", "mrr"):
        print(f"  {metric:<10} {means[metric]:.3f}")
        drifted |= check(metric, args.config, means[metric])

    # Answers: a fixed sample, because each question costs a judge call.
    if not args.skip_answers:
        sample = fixed_sample(answerable)
        print(f"\nAnswer drift over {len(sample)} fixed sample question(s)")
        generator = get_client(args.generator)
        judge = build_judge(args.judge)

        totals: dict[str, list[float]] = {}
        for question in sample:
            generated = answer_question(question["question"], retriever, generator)
            scores = score_pipeline(
                args.config, question["question"], generated.answer,
                generated.contexts, judge,
            )
            for name, metric in scores.metrics.items():
                if metric.parsed:
                    totals.setdefault(name, []).append(metric.score)

        for name, values in sorted(totals.items()):
            mean = sum(values) / len(values)
            print(f"  {name:<20} {mean:.2f}")
            drifted |= check(name, args.config, mean)

    print("\nDrift detected." if drifted else "\nNo drift against the rolling baseline.")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
