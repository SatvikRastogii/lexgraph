"""Render the measured results into a single markdown report.

Reads whatever exists in reports/ and writes reports/EVALUATION.md. Kept
separate from the evaluation scripts so a report can be regenerated without
re-running anything expensive.

    python scripts/make_report.py
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORTS_DIR = "reports"
OUTPUT = os.path.join(REPORTS_DIR, "EVALUATION.md")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def retrieval_section(ablation):
    if not ablation:
        return ["_No retrieval ablation yet. Run `python scripts/eval_retrieval.py`._"]

    lines = [
        "## Retrieval",
        "",
        "Document-level ground truth, no LLM in the loop.",
        "",
        "| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, payload in ablation.items():
        m = payload["metrics"]
        low, high = payload["recall@5_ci95"]
        lines.append(
            f"| `{name}` | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
            f"[{low:.2f}, {high:.2f}] | {m['ndcg@10']:.3f} | {m['mrr']:.3f} | "
            f"{payload['latency_ms']['p50']:.0f}ms |"
        )

    tiers = sorted({t for p in ablation.values() for t in p.get("by_difficulty", {})})
    if len(tiers) > 1:
        lines += ["", "### By difficulty tier", ""]
        header = "| configuration | " + " | ".join(
            f"{t} R@5 | {t} nDCG@10" for t in tiers
        ) + " |"
        lines.append(header)
        lines.append("|---" * (1 + 2 * len(tiers)) + "|")
        for name, payload in ablation.items():
            cells = []
            for tier in tiers:
                stats = payload.get("by_difficulty", {}).get(tier)
                cells.append(
                    f"{stats['recall@5']:.3f} | {stats['ndcg@10']:.3f}" if stats else "- | -"
                )
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        counts = next(iter(ablation.values())).get("by_difficulty", {})
        lines += [
            "",
            "n per tier: "
            + ", ".join(f"{t} = {counts[t]['n']}" for t in tiers if t in counts)
            + ".",
        ]

    lines += [
        "",
        "Intervals are percentile bootstrap over per-question scores. Where they "
        "overlap, the difference between configurations is not established at this "
        "sample size.",
    ]
    return lines


def answers_section(quality_files):
    if not quality_files:
        return ["## Answer quality", "", "_No judged run yet._"]

    lines = ["## Answer quality", ""]
    for path in quality_files:
        payload = _load(path)
        if not payload:
            continue
        for config, summary in payload["configs"].items():
            metrics = summary.get("metrics", {})
            if not metrics:
                continue
            complete = "" if summary.get("complete", True) else " _(partial run)_"
            lines += [
                f"### `{payload['generator']}` · `{config}`{complete}",
                "",
                f"Judge: `{payload['judge']}` — a different model family from the "
                f"generator.",
                "",
                "| metric | mean | 95% CI | n |",
                "|---|---|---|---|",
            ]
            for name, stats in sorted(metrics.items()):
                low, high = stats["ci95"]
                lines.append(
                    f"| {name.replace('_', ' ')} | {stats['mean']:.2f} | "
                    f"[{low:.2f}, {high:.2f}] | {stats['n']} |"
                )
            lines += [
                "",
                f"Median answer length: {summary['answer_words']['median']:.0f} words. "
                f"Median latency: {summary['latency_ms_median'] / 1000:.1f}s.",
            ]
            if summary.get("unparsed_judge_responses"):
                lines.append(
                    f"{summary['unparsed_judge_responses']} judge response(s) could not "
                    f"be parsed and are excluded from the means above."
                )
            if "abstention" in summary:
                stats = summary["abstention"]
                lines += [
                    "",
                    f"Abstention: refused {stats['correctly_refused']}/"
                    f"{stats['out_of_corpus_questions']} out-of-corpus questions; "
                    f"wrongly refused {stats['wrongly_refused_answerable']} answerable.",
                ]
            lines.append("")
    return lines


def calibration_section(calibration):
    if not calibration:
        return []
    lines = [
        "## Abstention calibration",
        "",
        "| retriever | answerable | out-of-corpus | separation | threshold | answers | refuses |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, payload in calibration.items():
        lines.append(
            f"| `{name}` | {payload['answerable_mean']:.3f} | "
            f"{payload['unanswerable_mean']:.3f} | {payload['separation']:+.3f} | "
            f"{payload['threshold']:.3f} | "
            f"{payload['answered_when_answerable']:.0%} | "
            f"{payload['refused_when_unanswerable']:.0%} |"
        )
    lines += [
        "",
        "Thresholds maximise Youden's J against the gold set's out-of-corpus "
        "questions. Separation is the gap between the two populations' mean "
        "confidence; a retriever whose separation is near zero cannot support "
        "abstention at any threshold.",
        "",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    ablation = _load(os.path.join(REPORTS_DIR, "retrieval_ablation.json"))
    calibration = _load(os.path.join(REPORTS_DIR, "abstention_calibration.json"))
    quality_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "answer_quality_*.json")))

    goldset = _load(os.path.join("data", "goldset.json")) or {"questions": []}
    answerable = sum(1 for q in goldset["questions"] if q["answerable"])
    unanswerable = len(goldset["questions"]) - answerable

    lines = [
        "# LexGraph — evaluation results",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Gold set: {answerable} answerable, {unanswerable} out-of-corpus, over 40 "
        "judgments (38 unique cases).",
        "",
        "Regenerate with `python scripts/make_report.py`.",
        "",
    ]
    lines += retrieval_section(ablation)
    lines += ["", *calibration_section(calibration)]
    lines += answers_section(quality_files)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
