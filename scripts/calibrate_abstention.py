"""Calibrate the abstention threshold against the gold set.

Retrieval confidence is measured for every question, split by whether the
corpus can actually answer it, and the threshold that best separates the two
populations is chosen by maximising Youden's J.

The separation figure matters more than the threshold. If out-of-corpus
questions score about as confidently as answerable ones, no threshold can
divide them and abstention on this signal is decoration -- better to report
that honestly than to ship a guardrail that never fires or always does.

    python scripts/calibrate_abstention.py
    python scripts/calibrate_abstention.py --configs dense hybrid-rerank
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.guardrails.abstention import calibrate_threshold, retrieval_confidence, separation
from lexgraph.retrieval.pipeline import build_retriever

GOLDSET_PATH = os.path.join("data", "goldset.json")
OUTPUT_PATH = os.path.join("reports", "abstention_calibration.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--configs", nargs="*", default=["dense", "bm25", "hybrid-rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    with open(args.goldset, "r", encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    print(f"{len(answerable)} answerable, {len(unanswerable)} out-of-corpus\n")

    results = {}
    for name in args.configs:
        retriever = build_retriever(name)
        yes = [retrieval_confidence(retriever.search(q["question"], args.top_k))
               for q in answerable]
        no = [retrieval_confidence(retriever.search(q["question"], args.top_k))
              for q in unanswerable]

        threshold, stats = calibrate_threshold(yes, no)
        gap = separation(yes, no)
        results[name] = {
            "threshold": threshold,
            "separation": gap,
            "answerable_mean": statistics.mean(yes),
            "unanswerable_mean": statistics.mean(no),
            **stats,
        }

        print(f"[{name}]")
        print(f"  confidence   answerable {statistics.mean(yes):.3f} | "
              f"out-of-corpus {statistics.mean(no):.3f} | separation {gap:+.3f}")
        print(f"  threshold    {threshold:.3f}")
        print(f"  would answer {stats['answered_when_answerable']:.0%} of answerable, "
              f"refuse {stats['refused_when_unanswerable']:.0%} of out-of-corpus "
              f"(Youden J {stats['youden_j']:.2f})")
        if gap < 0.05:
            print("  WARNING: the two populations barely separate; abstention on this "
                  "signal is not meaningful")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
