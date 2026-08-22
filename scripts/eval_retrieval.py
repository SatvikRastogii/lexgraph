"""Measure every retrieval configuration against the gold set.

No LLM is involved, so the whole sweep runs in about a minute and can be
re-run after any retrieval change. This is the loop that makes iterating on
retrieval practical; the judged evaluation is for answer quality, not for
deciding whether a retriever got better.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --configs dense hybrid --k 5
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.eval.retrieval_metrics import aggregate, bootstrap_ci, score_query
from lexgraph.retrieval.pipeline import CONFIGURATIONS, build_retriever, timed_search

GOLDSET_PATH = os.path.join("data", "goldset.json")
RESULTS_PATH = os.path.join("reports", "retrieval_ablation.json")
HEADLINE_KS = (1, 3, 5, 10)


def evaluate(retriever, questions, top_k, ks):
    """Score one retriever over the answerable gold-set questions."""
    per_question, latencies = [], []
    for question in questions:
        hits, latency_ms = timed_search(retriever, question["question"], top_k)
        latencies.append(latency_ms)
        scores = score_query(
            [hit.doc_id for hit in hits], question["relevant_docs"], ks=ks
        )
        scores["_id"] = question["id"]
        scores["_category"] = question["category"]
        per_question.append(scores)
    return per_question, latencies


def probe_out_of_corpus(retriever, questions, top_k):
    """Top-1 score for questions with no supporting document.

    Used to calibrate the abstention threshold: a retriever that returns
    confident scores for questions the corpus cannot answer has no usable
    signal to refuse on.
    """
    tops = []
    for question in questions:
        hits = retriever.search(question["question"], top_k)
        tops.append(hits[0].score if hits else 0.0)
    return tops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGURATIONS))
    parser.add_argument("--k", type=int, default=10, help="hits retrieved per query")
    parser.add_argument("--results-file", default=RESULTS_PATH)
    args = parser.parse_args()

    with open(args.goldset, encoding="utf-8") as handle:
        goldset = json.load(handle)

    answerable = [q for q in goldset["questions"] if q["answerable"]]
    unanswerable = [q for q in goldset["questions"] if not q["answerable"]]

    print(f"Gold set: {len(answerable)} answerable, {len(unanswerable)} out-of-corpus")
    print(f"Retrieving top {args.k} per query\n")

    results = {}
    for name in args.configs:
        print(f"[{name}] {CONFIGURATIONS[name]}")
        started = time.perf_counter()
        retriever = build_retriever(name)
        build_seconds = time.perf_counter() - started

        per_question, latencies = evaluate(retriever, answerable, args.k, HEADLINE_KS)
        means = aggregate([{k: v for k, v in q.items() if not k.startswith("_")}
                           for q in per_question])
        recall5 = [q["recall@5"] for q in per_question]
        low, high = bootstrap_ci(recall5)

        results[name] = {
            "description": CONFIGURATIONS[name],
            "metrics": means,
            "recall@5_ci95": [low, high],
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "max": max(latencies),
            },
            "build_seconds": build_seconds,
            "per_question": per_question,
            "out_of_corpus_top_score": probe_out_of_corpus(retriever, unanswerable, args.k),
        }
        print(
            f"  recall@5 {means['recall@5']:.3f} [{low:.3f}, {high:.3f}]  "
            f"ndcg@10 {means['ndcg@10']:.3f}  mrr {means['mrr']:.3f}  "
            f"p50 {statistics.median(latencies):.0f}ms\n"
        )

    _print_table(results)

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {args.results_file}")


def _print_table(results):
    header = f"{'configuration':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'nDCG@10':>9}{'MRR':>7}{'p50 ms':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, payload in results.items():
        metrics = payload["metrics"]
        print(
            f"{name:<16}"
            f"{metrics['recall@1']:>7.3f}{metrics['recall@5']:>7.3f}"
            f"{metrics['recall@10']:>7.3f}{metrics['ndcg@10']:>9.3f}"
            f"{metrics['mrr']:>7.3f}{payload['latency_ms']['p50']:>9.0f}"
        )


if __name__ == "__main__":
    main()
