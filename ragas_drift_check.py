"""
LexGraph — RAGAS Drift Check
Re-scores a small, fixed sample of benchmark questions on a schedule (e.g. a
daily cron entry) and flags when scores drop meaningfully vs. a rolling
baseline. This is the lightweight, always-on regression signal -- not a
replacement for the full 21-question sweep in ragas_evaluation.py.
"""

import json
import os
from datetime import datetime

import telemetry
from ragas_evaluation import run_evaluation, METRIC_NAMES, BENCHMARK_FILE

# Fixed index per category, not random -- comparability across runs matters
# more than variety when the goal is detecting drift over time.
SAMPLE_INDEX = 0
ROLLING_WINDOW = 5
DRIFT_ABS_THRESHOLD = 0.5    # points on the 1-5 scale
DRIFT_REL_THRESHOLD = 0.15   # 15% relative drop


def build_sample():
    """One fixed question per category from benchmark_questions.json."""
    with open(BENCHMARK_FILE, "r") as f:
        benchmarks = json.load(f)
    return {cat: qs[SAMPLE_INDEX:SAMPLE_INDEX + 1] for cat, qs in benchmarks.items() if qs}


def aggregate(results):
    """Per-metric average score for each pipeline across this run's questions."""
    agg = {m: {"naive": [], "graphrag": []} for m in METRIC_NAMES}
    for entry in results:
        for m in METRIC_NAMES:
            n_score = entry["naive"][m]["score"]
            g_score = entry["graphrag"][m]["score"]
            if n_score > 0:
                agg[m]["naive"].append(n_score)
            if g_score > 0:
                agg[m]["graphrag"].append(g_score)
    return {
        m: {
            "naive": round(sum(v["naive"]) / len(v["naive"]), 2) if v["naive"] else None,
            "graphrag": round(sum(v["graphrag"]) / len(v["graphrag"]), 2) if v["graphrag"] else None,
        }
        for m, v in agg.items()
    }


def rolling_baseline(metric, pipeline):
    """Mean of the last ROLLING_WINDOW historical scores for this metric/pipeline."""
    history = telemetry.fetch_ragas_history(metric=metric, pipeline=pipeline, limit=10000)
    recent = history[-ROLLING_WINDOW:]
    if not recent:
        return None
    return round(sum(r["score"] for r in recent) / len(recent), 2)


def main():
    telemetry.init_db()
    sample = build_sample()
    n_questions = sum(len(qs) for qs in sample.values())
    print(f"Running RAGAS drift check on {n_questions} fixed sample question(s)...")

    today = datetime.now().strftime("%Y-%m-%d")
    drift_dir = os.path.join("logs", "ragas_drift_history")
    os.makedirs(drift_dir, exist_ok=True)

    results = run_evaluation(
        questions=sample,
        results_file=os.path.join(drift_dir, f"{today}.json"),
        report_file=None,
    )
    if not results:
        print("Drift check produced no results (evaluation failed) -- see output above.")
        return

    scores = aggregate(results)

    any_drift = False
    for metric, pipelines in scores.items():
        for pipeline, score in pipelines.items():
            if score is None:
                continue
            baseline = rolling_baseline(metric, pipeline)
            drift_detected = False
            if baseline is not None:
                abs_drop = baseline - score
                rel_drop = (abs_drop / baseline) if baseline else 0
                drift_detected = abs_drop >= DRIFT_ABS_THRESHOLD or rel_drop >= DRIFT_REL_THRESHOLD
            telemetry.log_ragas_drift(metric, pipeline, score, baseline=baseline, drift_detected=drift_detected)
            if drift_detected:
                any_drift = True
                print(f"⚠ DRIFT DETECTED: {pipeline}/{metric} dropped to {score} (baseline {baseline})")

    if not any_drift:
        print("No drift detected -- scores within normal range of the rolling baseline.")


if __name__ == "__main__":
    main()
