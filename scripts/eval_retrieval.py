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

from lexgraph.eval.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    paragraph_recall_at_k,
    score_query,
)
from lexgraph.retrieval.pipeline import CONFIGURATIONS, build_retriever, timed_search

GOLDSET_PATH = os.path.join("data", "goldset.json")
RESULTS_PATH = os.path.join("reports", "retrieval_ablation.json")
HEADLINE_KS = (1, 3, 5, 10)


def evaluate(retriever, questions, top_k, ks):
    """Score one retriever over the answerable gold-set questions.

    Returns the per-question scores, the latencies, and whether this
    configuration emits paragraph labels at all -- see ``_strip_paragraph``.
    """
    per_question, latencies = [], []
    labelled_hits = 0
    for question in questions:
        hits, latency_ms = timed_search(retriever, question["question"], top_k)
        latencies.append(latency_ms)
        scores = score_query(
            [hit.doc_id for hit in hits], question["relevant_docs"], ks=ks
        )

        # Document-level recall gives full credit for any chunk of the right
        # judgment, including the one listing who appeared for whom. This asks
        # the stricter question: did the retriever land on the passage that
        # actually carries the answer? Queries whose supporting documents have
        # no usable numbering score None and drop out of the mean.
        gold_paragraphs = {
            name: [int(n) for n in paragraphs]
            for name, paragraphs in question.get("relevant_paragraphs", {}).items()
        }
        chunk_ranking = [(hit.doc_id, hit.para_label) for hit in hits]
        labelled_hits += sum(1 for _, label in chunk_ranking if label)
        for k in ks:
            scores[f"paragraph_recall@{k}"] = paragraph_recall_at_k(
                chunk_ranking, gold_paragraphs, k
            )

        scores["_id"] = question["id"]
        scores["_category"] = question["category"]
        scores["_difficulty"] = question.get("difficulty", "standard")
        per_question.append(scores)
    return per_question, latencies, labelled_hits > 0


def _strip_paragraph(per_question):
    """Drop paragraph metrics from a configuration that cannot carry them.

    ``dense-fixed`` chunks on 500-word windows, so no chunk knows which
    paragraph it came from and every hit scores zero by construction. Printing
    that as 0.000 would read as a retriever that never finds the right passage,
    when the truth is that this configuration cannot say which passage it
    found. Not applicable and zero are different claims, and only one of them
    is true here.
    """
    for scores in per_question:
        for key in [k for k in scores if k.startswith("paragraph_recall@")]:
            del scores[key]
    return per_question


def by_difficulty(per_question):
    """Split the means by tier.

    The standard questions turned out to be near-saturated -- MRR above 0.93
    for every configuration -- so an aggregate number hides whether a change
    helped. The hard tier is deliberately phrased to share little vocabulary
    with the judgments it should retrieve, which is where the configurations
    have room to differ.
    """
    tiers = {}
    for tier in sorted({q["_difficulty"] for q in per_question}):
        rows = [
            {k: v for k, v in q.items() if not k.startswith("_")}
            for q in per_question
            if q["_difficulty"] == tier
        ]
        if rows:
            tiers[tier] = {"n": len(rows), **aggregate(rows)}
    return tiers


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

        per_question, latencies, has_paragraphs = evaluate(
            retriever, answerable, args.k, HEADLINE_KS
        )
        if not has_paragraphs:
            per_question = _strip_paragraph(per_question)
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
            "by_difficulty": by_difficulty(per_question),
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
    header = (f"{'configuration':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'nDCG@10':>9}"
              f"{'MRR':>7}{'ParaR@5':>9}{'p50 ms':>9}")
    print("\n" + header)
    print("-" * len(header))
    for name, payload in results.items():
        metrics = payload["metrics"]
        print(
            f"{name:<16}"
            f"{metrics['recall@1']:>7.3f}{metrics['recall@5']:>7.3f}"
            f"{metrics['recall@10']:>7.3f}{metrics['ndcg@10']:>9.3f}"
            f"{metrics['mrr']:>7.3f}"
            f"{_para(metrics.get('paragraph_recall@5')):>9}"
            f"{payload['latency_ms']['p50']:>9.0f}"
        )
    print("\nParaR@5 is recall@5 restricted to landing on a paragraph that "
          "carries the answer.\nThe gap against R@5 is how often the right "
          "case was found at the wrong passage.\nn/a means the configuration's "
          "chunks carry no paragraph labels to score.")

    tiers = sorted({t for p in results.values() for t in p["by_difficulty"]})
    if len(tiers) > 1:
        print(f"\nBy difficulty tier ({', '.join(tiers)}):")
        sub = f"{'configuration':<16}" + "".join(
            f"{t + ' R@5':>14}{t + ' nDCG':>15}" for t in tiers
        )
        print(sub)
        print("-" * len(sub))
        for name, payload in results.items():
            row = f"{name:<16}"
            for tier in tiers:
                stats = payload["by_difficulty"].get(tier)
                row += (
                    f"{stats['recall@5']:>14.3f}{stats['ndcg@10']:>15.3f}"
                    if stats else f"{'-':>14}{'-':>15}"
                )
            print(row)


def _para(value):
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
