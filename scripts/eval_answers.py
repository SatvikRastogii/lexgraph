"""Judged answer-quality evaluation, with an independent judge.

Differs from the evaluation it replaces in four ways that all push the numbers
toward being trustworthy rather than flattering:

  * the judge is a different model from the generator, enforced, not assumed
  * every headline score carries a bootstrap confidence interval, because
    n is a few dozen questions and a bare mean implies precision that does
    not exist
  * answer length is reported next to every score, since LLM judges reward
    verbosity and the configurations produce different-length answers
  * citation accuracy is measured by string-checking the answer against the
    retrieved context, not asked of the judge

    python scripts/eval_answers.py --configs dense hybrid-rerank
    python scripts/eval_answers.py --limit 5          # smoke test
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.eval.judge import require_independent_judge, score_pipeline
from lexgraph.eval.retrieval_metrics import bootstrap_ci
from lexgraph.generation import answer_question
from lexgraph.llm import DEFAULT_GENERATOR, DEFAULT_JUDGE, build_judge, get_client
from lexgraph.retrieval.pipeline import build_retriever

GOLDSET_PATH = os.path.join("data", "goldset.json")
RESULTS_PATH = os.path.join("reports", "answer_quality.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--configs", nargs="*", default=["dense", "hybrid-rerank"])
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--abstention-threshold", type=float, default=None)
    parser.add_argument("--results-file", default=RESULTS_PATH)
    args = parser.parse_args()

    # Refuse a self-judged run before spending an hour producing numbers that
    # would have to be thrown away.
    require_independent_judge(args.generator, args.judge)

    with open(args.goldset, encoding="utf-8") as handle:
        goldset = json.load(handle)
    questions = goldset["questions"]
    if args.limit:
        questions = questions[: args.limit]

    generator = get_client(args.generator)
    judge = build_judge(args.judge)

    print(f"Generator: {generator.label}")
    print(f"Judge:     {judge.label}   (independent)")
    print(f"Questions: {len(questions)}  |  configs: {', '.join(args.configs)}\n")

    results = {"generator": generator.label, "judge": judge.label, "configs": {}}

    for config in args.configs:
        print(f"=== {config} ===")
        retriever = build_retriever(config)
        records = []

        for index, question in enumerate(questions, start=1):
            started = time.perf_counter()
            generated = answer_question(
                question["question"], retriever, generator,
                top_k=args.top_k, abstention_threshold=args.abstention_threshold,
            )
            try:
                scores = score_pipeline(
                    config, question["question"], generated.answer,
                    generated.contexts, judge,
                )
            except Exception as error:  # noqa: BLE001
                # Generation is the expensive half and it already succeeded.
                # Losing the whole run because the judge ran out of daily quota
                # at question 30 would throw away an hour of local inference,
                # so the partial results are written out and the run stops.
                print(f"\n  judge unavailable at {question['id']}: "
                      f"{str(error)[:160]}")
                print(f"  keeping {len(records)} completed question(s)")
                break

            records.append({
                "id": question["id"],
                "category": question["category"],
                "answerable": question["answerable"],
                "generation": generated.as_dict(),
                "scores": scores.as_dict(),
                "wall_seconds": time.perf_counter() - started,
            })
            print(f"  [{index}/{len(questions)}] {question['id']} "
                  f"mean {scores.mean():.2f}  {generated.latency['total_ms']:.0f}ms")

            _write(args.results_file, results, config, records)

        if not records:
            print(f"  no questions completed for {config}")
            continue
        results["configs"][config] = _summarise(records)
        results["configs"][config]["records"] = records
        _print_summary(config, results["configs"][config])
        _write(args.results_file, results, config, records)

    print(f"\nWrote {args.results_file}")


def _write(path, results, config, records):
    """Persist progress after every question so a mid-run failure keeps its work."""
    if not records:
        return
    snapshot = dict(results)
    snapshot["configs"] = dict(results["configs"])
    summary = _summarise(records)
    summary["records"] = records
    summary["complete"] = False
    snapshot["configs"][config] = summary
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)


def _summarise(records):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    metric_names = sorted({m for r in answerable for m in r["scores"]["metrics"]})
    metrics = {}
    for name in metric_names:
        values = [
            r["scores"]["metrics"][name]["score"]
            for r in answerable
            if name in r["scores"]["metrics"]
            and r["scores"]["metrics"][name].get("parsed", True)
        ]
        if not values:
            continue
        low, high = bootstrap_ci(values)
        metrics[name] = {
            "mean": statistics.mean(values),
            "ci95": [low, high],
            "n": len(values),
        }

    lengths = [r["scores"]["answer_words"] for r in answerable]
    unparsed = sum(
        1 for r in records for m in r["scores"]["metrics"].values()
        if not m.get("parsed", True)
    )

    summary = {
        "metrics": metrics,
        "answer_words": {
            "mean": statistics.mean(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
        },
        "unparsed_judge_responses": unparsed,
        "latency_ms_median": statistics.median(
            [r["generation"]["latency"]["total_ms"] for r in records]
        ) if records else 0,
    }

    if unanswerable:
        refused = sum(1 for r in unanswerable if r["generation"]["abstained"])
        summary["abstention"] = {
            "out_of_corpus_questions": len(unanswerable),
            "correctly_refused": refused,
            "refusal_rate": refused / len(unanswerable),
        }
        wrongly_refused = sum(1 for r in answerable if r["generation"]["abstained"])
        summary["abstention"]["wrongly_refused_answerable"] = wrongly_refused

    return summary


def _print_summary(config, summary):
    print(f"\n  {config} summary")
    for name, payload in summary["metrics"].items():
        low, high = payload["ci95"]
        print(f"    {name:<20} {payload['mean']:.2f}  [{low:.2f}, {high:.2f}]  n={payload['n']}")
    print(f"    {'answer words (median)':<20} {summary['answer_words']['median']:.0f}")
    if summary["unparsed_judge_responses"]:
        print(f"    unparsed judge responses: {summary['unparsed_judge_responses']}")
    if "abstention" in summary:
        stats = summary["abstention"]
        print(f"    refused {stats['correctly_refused']}/{stats['out_of_corpus_questions']} "
              f"out-of-corpus, wrongly refused {stats['wrongly_refused_answerable']} answerable")
    print()


if __name__ == "__main__":
    main()
