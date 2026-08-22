"""Test whether the judge's scores mean anything.

Every number in reports/answer_quality_*.json comes from one model's opinion.
Nothing so far has checked that the opinion tracks the property it claims to
measure, which makes the judged table the least defensible thing in this
repository. Two checks are run here, and neither is a substitute for human
labels -- what each can and cannot support is stated with its result.

  probes
      Answer/context pairs whose correct score is fixed by construction: an
      answer copied out of the context is faithful; one that contradicts it is
      not; one citing an Article absent from the context is fabricating. The
      judge is asked to score both variants of each pair and only has to rank
      them correctly. Direction rather than absolute value, because a judge
      that is uniformly harsh is still useful and a judge that cannot tell
      grounded from fabricated is not, at any calibration.

      This is an absolute check: failing it means the scores are noise. Passing
      it means the judge separates the cases a human would separate most
      obviously, which is a floor on validity, not a ceiling.

  agreement
      Re-scores answers already in a results file with a second, unrelated
      judge model and reports Spearman rho and mean absolute difference per
      metric. High agreement means the metric is a property of the answer
      rather than of the model reading it. It cannot show both judges are not
      wrong in the same direction -- two language models share far more
      training data with each other than either does with a lawyer.

    python scripts/validate_judge.py --probes
    python scripts/validate_judge.py --agreement reports/answer_quality_llama3.1.json
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.eval.judge import score_pipeline
from lexgraph.llm import DEFAULT_JUDGE, build_judge, get_client

RESULTS_PATH = os.path.join("reports", "judge_validation.json")

# Real corpus text, so the judge is reading the register it will meet in the
# evaluation rather than invented prose it might treat differently.
CONTEXT = [
    "Sunil Batra vs Delhi Administration (1979). The Court held that a "
    "prisoner does not lose his fundamental rights on conviction, and that "
    "Article 21 reaches the conditions of confinement. A letter from a "
    "prisoner may be treated as a petition under Article 32.",
    "Rupa Ashok Hurra vs Ashok Hurra (2002). A curative petition may be "
    "entertained after dismissal of a review petition, to prevent abuse of "
    "process and cure gross miscarriage of justice. It must be certified by a "
    "senior advocate and is ordinarily decided without oral hearing.",
]

QUESTION = "Can a convicted prisoner invoke fundamental rights over conditions inside prison?"

# (metric, label, better answer, worse answer). The judge must score the
# better one at least as high; a strict inequality is what counts as a win.
PROBES = [
    (
        "faithfulness",
        "grounded restatement vs contradiction",
        "A prisoner does not lose his fundamental rights on conviction, and "
        "Article 21 reaches the conditions of confinement.",
        "A prisoner forfeits all fundamental rights on conviction, and "
        "Article 21 stops at the prison gate.",
    ),
    (
        "hallucination",
        "cited from context vs fabricated authority",
        "Sunil Batra vs Delhi Administration (1979) held that Article 21 "
        "reaches conditions of confinement.",
        "In Ramesh Kumar vs State of Bihar (1987) the Court held under "
        "Article 47 that prisoners may petition the Chief Justice directly.",
    ),
    (
        "answer_relevancy",
        "answers the question vs answers a different one",
        "Yes. A convicted prisoner retains his fundamental rights, and "
        "Article 21 extends to the conditions of his confinement.",
        "A curative petition must be certified by a senior advocate and is "
        "ordinarily decided without an oral hearing.",
    ),
    (
        "completeness",
        "full answer vs bare assertion",
        "Yes. Conviction does not strip a prisoner of fundamental rights; "
        "Article 21 governs the conditions of confinement, and a letter from "
        "a prisoner may itself be treated as an Article 32 petition.",
        "Yes.",
    ),
    (
        "coherence",
        "ordered argument vs shuffled fragments",
        "Conviction does not remove fundamental rights. Article 21 therefore "
        "governs the conditions of confinement. A prisoner may raise those "
        "conditions before the Court, and a letter may serve as the petition.",
        "Letter petition Article 32. Conditions confinement the. Rights not "
        "removed conviction does. Article 21 therefore.",
    ),
]

METRIC_ORDER = [
    "faithfulness", "answer_relevancy", "context_precision",
    "completeness", "hallucination", "coherence", "legal_reasoning",
]


def run_probes(judge):
    """Score both variants of each probe and report which way the judge ranked."""
    rows = []
    for metric, label, better, worse in PROBES:
        scores = {}
        for variant, answer in (("better", better), ("worse", worse)):
            result = score_pipeline(
                "probe", QUESTION, answer, CONTEXT, judge, metrics=[metric]
            )
            score = result.metrics.get(metric)
            scores[variant] = score.score if score and score.parsed else None

        high, low = scores["better"], scores["worse"]
        verdict = "unparsed" if high is None or low is None else (
            "pass" if high > low else "tie" if high == low else "FAIL"
        )
        rows.append({
            "metric": metric, "probe": label,
            "better": high, "worse": low, "verdict": verdict,
        })
        print(f"  {metric:<18} {label:<44} "
              f"{_fmt(high)} vs {_fmt(low)}  {verdict}")
    return rows


def _fmt(value):
    return "  -" if value is None else f"{value:3.0f}"


def spearman(xs, ys):
    """Rank correlation, with the average-rank tie correction.

    Judge scores are a five-point scale, so ties are the common case, not an
    edge case; ranking without the correction would invent an ordering the
    judge never expressed.
    """
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0 or dy == 0:
        return None  # one judge gave every answer the same score
    return numerator / (dx * dy) ** 0.5


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def run_agreement(judge, results_path, config, limit):
    """Re-score stored answers with a second judge and compare."""
    with open(results_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    configs = payload.get("configs", {})
    name = config or next(iter(configs))
    records = [r for r in configs[name]["records"] if r["answerable"]][:limit]
    print(f"  re-scoring {len(records)} answers from {name} "
          f"(generated by {payload.get('generator')})\n")

    original, second = {}, {}
    for index, record in enumerate(records, start=1):
        generation = record["generation"]
        rescored = score_pipeline(
            name, generation["question"], generation["answer"],
            generation["contexts"], judge,
        )
        for metric, stored in record["scores"]["metrics"].items():
            fresh = rescored.metrics.get(metric)
            if not stored.get("parsed", True) or fresh is None or not fresh.parsed:
                continue
            original.setdefault(metric, []).append(stored["score"])
            second.setdefault(metric, []).append(fresh.score)
        print(f"  [{index}/{len(records)}] {record['id']}")

    rows = []
    for metric in METRIC_ORDER:
        xs, ys = original.get(metric), second.get(metric)
        if not xs or len(xs) < 3:
            continue
        rho = spearman(xs, ys)
        rows.append({
            "metric": metric,
            "n": len(xs),
            "spearman": rho,
            "mean_abs_diff": statistics.mean(abs(a - b) for a, b in zip(xs, ys, strict=True)),
            "exact_agreement": sum(1 for a, b in zip(xs, ys, strict=True) if a == b) / len(xs),
            "within_one": sum(1 for a, b in zip(xs, ys, strict=True) if abs(a - b) <= 1) / len(xs),
            "mean_first": statistics.mean(xs),
            "mean_second": statistics.mean(ys),
        })
    return rows


def _print_agreement(rows):
    header = (f"\n  {'metric':<18}{'n':>4}{'rho':>8}{'|diff|':>8}"
              f"{'exact':>8}{'within 1':>10}{'mean A':>9}{'mean B':>9}")
    print(header)
    print("  " + "-" * (len(header) - 3))
    for row in rows:
        rho = "    -" if row["spearman"] is None else f"{row['spearman']:5.2f}"
        print(f"  {row['metric']:<18}{row['n']:>4}{rho:>8}"
              f"{row['mean_abs_diff']:>8.2f}{row['exact_agreement']:>8.0%}"
              f"{row['within_one']:>10.0%}{row['mean_first']:>9.2f}"
              f"{row['mean_second']:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--probes", action="store_true", help="run the known-answer probes")
    parser.add_argument("--agreement", metavar="RESULTS_FILE",
                        help="re-score this results file with a second judge")
    parser.add_argument("--second-judge", default=None,
                        help="the second judge for --agreement; must differ from the first")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--results-file", default=RESULTS_PATH)
    args = parser.parse_args()

    if not args.probes and not args.agreement:
        parser.error("pass --probes, --agreement, or both")

    output = {}

    if args.probes:
        judge = build_judge(args.judge)
        print(f"Probes  judge: {judge.label}\n")
        rows = run_probes(judge)
        passed = sum(1 for r in rows if r["verdict"] == "pass")
        print(f"\n  {passed}/{len(rows)} probes ranked correctly")
        if passed < len(rows):
            print("  A probe the judge cannot rank is a metric its scores "
                  "should not be quoted for.")
        output["probes"] = {"judge": judge.label, "passed": passed,
                            "total": len(rows), "rows": rows}

    if args.agreement:
        if not args.second_judge:
            parser.error("--agreement needs --second-judge")
        second = get_client(args.second_judge)
        print(f"\nAgreement  second judge: {second.label}\n")
        rows = run_agreement(second, args.agreement, args.config, args.limit)
        _print_agreement(rows)
        print("\n  Agreement is reproducibility, not correctness: two language "
              "models\n  share far more with each other than either does with "
              "a lawyer.")
        output["agreement"] = {
            "second_judge": second.label,
            "source": args.agreement,
            "rows": rows,
        }

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nWrote {args.results_file}")


if __name__ == "__main__":
    main()
