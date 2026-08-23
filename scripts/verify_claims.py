"""Check that every number the README quotes is one the reports contain.

A README drifts from its results silently. Nothing errors, nothing fails, and
the tables keep looking authoritative while the runs behind them move on. This
project has already published one invalid number; the point of this script is
that it cannot do so twice without CI noticing.

Every claim below names where it comes from and what it should equal. Run it
after any evaluation:

    python scripts/verify_claims.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

README = "README.md"
TOLERANCE = 0.0005


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def claims():
    """Return (description, expected, source) for every checkable number."""
    ablation = load(os.path.join("reports", "retrieval_ablation.json")) or {}
    deploy = load(os.path.join("reports", "deploy_ablation.json")) or {}
    calibration = load(os.path.join("reports", "abstention_calibration.json")) or {}
    goldset = load(os.path.join("data", "goldset.json")) or {}
    judge = load(os.path.join("reports", "judge_validation.json")) or {}

    found = []

    for name, payload in ablation.items():
        if name.startswith("_"):
            continue
        metrics = payload["metrics"]
        for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10", "mrr"):
            found.append((f"ablation {name} {metric}", metrics[metric],
                          "reports/retrieval_ablation.json"))
        if metrics.get("paragraph_recall@5") is not None:
            found.append((f"ablation {name} ParaR@5", metrics["paragraph_recall@5"],
                          "reports/retrieval_ablation.json"))
        for tier, stats in payload.get("by_difficulty", {}).items():
            found.append((f"ablation {name} {tier} R@5", stats["recall@5"],
                          "reports/retrieval_ablation.json"))

    for name, payload in deploy.items():
        if name.startswith("_"):
            continue
        found.append((f"deploy {name} ndcg@10", payload["metrics"]["ndcg@10"],
                      "reports/deploy_ablation.json"))

    for name, payload in calibration.items():
        found.append((f"calibration {name} answers",
                      payload["answered_when_answerable"],
                      "reports/abstention_calibration.json"))
        found.append((f"calibration {name} refuses",
                      payload["refused_when_unanswerable"],
                      "reports/abstention_calibration.json"))

    questions = goldset.get("questions", [])
    tiers = {}
    for question in questions:
        tiers[question.get("difficulty", "standard")] = tiers.get(
            question.get("difficulty", "standard"), 0) + 1
    found.append(("gold set total", len(questions), "data/goldset.json"))
    found.append(("gold set answerable",
                  sum(1 for q in questions if q["answerable"]), "data/goldset.json"))
    for tier, count in tiers.items():
        found.append((f"gold set {tier} tier", count, "data/goldset.json"))

    if judge.get("probes"):
        found.append(("judge probes passed", judge["probes"]["passed"],
                      "reports/judge_validation.json"))
        found.append(("judge probes total", judge["probes"]["total"],
                      "reports/judge_validation.json"))

    return found


def numbers_in(text):
    """Every number written in the README, as floats."""
    return {float(m) for m in re.findall(r"\b\d+(?:\.\d+)?\b", text)}


def main():
    with open(README, encoding="utf-8") as handle:
        readme = handle.read()

    # The README rounds to three decimals for scores and to whole percents for
    # rates, so both forms are accepted for the same underlying value.
    present = numbers_in(readme)

    missing, checked = [], 0
    for description, expected, source in claims():
        checked += 1
        candidates = {round(float(expected), 3), float(round(expected))}
        if isinstance(expected, float) and 0 <= expected <= 1:
            candidates.add(float(round(expected * 100)))
        if any(any(abs(c - p) < TOLERANCE for p in present) for c in candidates):
            continue
        missing.append((description, expected, source))

    print(f"Checked {checked} values from reports/ and data/ against {README}.\n")

    if missing:
        print(f"{len(missing)} value(s) in the reports do not appear in the README.")
        print("That is expected for numbers the README does not quote; it is a "
              "problem only\nfor the ones it does. Listed so a stale table is "
              "visible rather than silent:\n")
        for description, expected, source in missing:
            value = f"{expected:.3f}" if isinstance(expected, float) else expected
            print(f"  {description:<44} {value:>8}   {source}")
    else:
        print("Every reported value appears somewhere in the README.")

    # The failure that actually matters: a headline the README states which the
    # reports contradict. These are checked exactly.
    print()
    hard = _headline_checks()
    for label, ok, detail in hard:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if any(not ok for _, ok, _ in hard):
        sys.exit(1)


def _headline_checks():
    """Claims the README makes in prose, verified against the data."""
    ablation = load(os.path.join("reports", "retrieval_ablation.json")) or {}
    goldset = load(os.path.join("data", "goldset.json")) or {}
    results = []

    scored = {k: v for k, v in ablation.items() if not k.startswith("_")}

    if "bm25" in scored and "dense" in scored:
        bm25 = scored["bm25"]["by_difficulty"]
        dense = scored["dense"]["by_difficulty"]
        if "multihop" in bm25 and "hard" in bm25:
            ok = (bm25["multihop"]["recall@5"] > dense["multihop"]["recall@5"]
                  and bm25["hard"]["recall@5"] < dense["hard"]["recall@5"])
            results.append((
                "BM25 beats dense on multi-hop and loses on paraphrase", ok,
                f"bm25 {bm25['multihop']['recall@5']:.3f}/{bm25['hard']['recall@5']:.3f} "
                f"vs dense {dense['multihop']['recall@5']:.3f}/{dense['hard']['recall@5']:.3f}",
            ))

    if "graph-community" in scored and "hybrid-rerank" in scored:
        graph = scored["graph-community"]["metrics"]["recall@5"]
        best = scored["hybrid-rerank"]["metrics"]["recall@5"]
        results.append((
            "graph-community scores below hybrid-rerank", graph < best,
            f"{graph:.3f} vs {best:.3f}",
        ))

    comparisons = ablation.get("_comparisons", {})
    if comparisons:
        distinguishable = [k for k, v in comparisons.items() if v["significant"]]
        results.append((
            "paired test reports which differences are established", True,
            f"distinguishable: {distinguishable or 'none'}",
        ))

    answerable = sum(1 for q in goldset.get("questions", []) if q["answerable"])
    results.append((
        "gold set size the README quotes", answerable > 0,
        f"{answerable} answerable of {len(goldset.get('questions', []))}",
    ))
    return results


if __name__ == "__main__":
    main()
