"""Pre-compute answers to the gold-set questions for the public demo.

A public URL on a free-tier key is a quota that one visitor can drain, after
which the demo shows 429s to everyone who follows. Replay removes that: the 65
gold-set questions ship with their answers already computed, so the common path
costs nothing and returns instantly, and the live path stays available behind a
per-session budget for anyone who wants to prove it really runs.

These are real answers from the real pipeline, not fixtures. Every one is
produced by the same retriever, prompt, abstention threshold and citation check
the live path uses, and each record carries the retrieved sources and the
citation verification so the demo can show its work rather than a bare string.

    python scripts/build_replay.py
    python scripts/build_replay.py --limit 5      # smoke test first

Run it with the deployment's own configuration, or the cached answers will
describe a pipeline the deployment is not running:

    LEXGRAPH_DENSE_BACKEND=numpy LEXGRAPH_EMBEDDER=fastembed \\
      python scripts/build_replay.py --generator gemini:gemini-3-flash-preview
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.generation import answer_question
from lexgraph.llm import build_generator
from lexgraph.retrieval.pipeline import build_retriever

GOLDSET_PATH = os.path.join("data", "goldset.json")
REPLAY_PATH = os.path.join("data", "replay.json")
LOCAL_CALIBRATION = os.path.join("reports", "abstention_calibration.json")
DEPLOY_CALIBRATION = os.path.join("reports", "abstention_deploy.json")


def default_calibration():
    """Match the calibration file to the embedder actually in use.

    The two are on different score scales -- hybrid-rerank calibrates to 0.045
    under the deployment embedder and 0.027 locally -- and picking the wrong
    one produces a plausible number that refuses the wrong questions. The first
    run of this script did exactly that.
    """
    backend = os.getenv("LEXGRAPH_DENSE_BACKEND", "chroma").lower()
    return DEPLOY_CALIBRATION if backend == "numpy" else LOCAL_CALIBRATION


def abstention_threshold(config, path):
    """The calibrated threshold, or None.

    Read rather than hardcoded: it is per-configuration and per-score-scale, so
    a constant here would go stale the moment the gold set or embedder changes
    and would keep looking plausible while it did.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)[config]["threshold"]
    except (OSError, KeyError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--config", default=os.getenv("LEXGRAPH_RETRIEVER", "hybrid-rerank"))
    parser.add_argument("--generator", default=os.getenv("LEXGRAPH_GENERATOR", "ollama:qwen2.5:3b"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=REPLAY_PATH)
    parser.add_argument("--citation-policy", default="warn")
    parser.add_argument("--calibration", default=None,
                        help="calibration file; defaults to matching the embedder")
    parser.add_argument(
        "--rotate", action="store_true",
        help="rotate across Gemini models when one exhausts its daily quota",
    )
    args = parser.parse_args()

    with open(args.goldset, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    if args.limit:
        questions = questions[: args.limit]

    retriever = build_retriever(args.config)
    generator = build_generator(args.generator, rotate=args.rotate)
    calibration = args.calibration or default_calibration()
    threshold = abstention_threshold(args.config, calibration)

    print(f"config     {args.config}")
    print(f"generator  {generator.label}")
    print(f"threshold  {threshold if threshold is not None else 'off'} "
          f"(from {calibration})")
    print(f"questions  {len(questions)}\n")

    # Resume, because this is one API call per question against a metered free
    # tier and losing 40 completed answers to a rate limit at question 41 is
    # avoidable.
    existing = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as handle:
                payload = json.load(handle)
            # Keyed on the requested spec, not the client's label. Turning on
            # rotation changes the label to a chain of six models, and matching
            # on that would discard every answer computed before it and spend
            # the quota again to recreate them.
            if payload.get("config") == args.config and payload.get("spec") == args.generator:
                existing = payload.get("answers", {})
                print(f"resuming: {len(existing)} already computed\n")
        except (OSError, ValueError):
            pass

    answers = dict(existing)
    for index, question in enumerate(questions, start=1):
        text = question["question"]
        if text in answers:
            continue
        try:
            result = answer_question(
                text, retriever, generator,
                top_k=args.top_k,
                abstention_threshold=threshold,
                citation_policy=args.citation_policy,
            )
        except Exception as error:  # noqa: BLE001
            print(f"  [{index}/{len(questions)}] {question['id']} failed: "
                  f"{str(error)[:120]}")
            print(f"  keeping {len(answers)} computed answer(s)")
            break

        answers[text] = {
            "id": question["id"],
            "answer": result.answer,
            "abstained": result.abstained,
            "sources": result.sources,
            "contexts": result.contexts,
            "citations": {
                "supported": result.citations.supported if result.citations else [],
                "unsupported": result.citations.unsupported if result.citations else [],
                "accuracy": result.citations.accuracy if result.citations else 1.0,
            },
            "confidence": result.abstention.confidence if result.abstention else None,
            "latency_ms": result.latency.get("total_ms", 0.0),
            "answerable": question["answerable"],
            "difficulty": question.get("difficulty", "standard"),
        }
        _write(args.output, args.config, args.generator, generator.label, threshold, answers)
        state = "refused" if result.abstained else f"{len(result.sources)} sources"
        print(f"  [{index}/{len(questions)}] {question['id']}  {state}  "
              f"{result.latency.get('total_ms', 0):.0f}ms")

    refused = sum(1 for a in answers.values() if a["abstained"])
    print(f"\n{len(answers)} answers, {refused} refusals")
    print(f"Wrote {args.output} "
          f"({os.path.getsize(args.output) / 1024:.0f} KB)")


def _write(path, config, spec, generator, threshold, answers):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config,
                "spec": spec,
                "generator": generator,
                "abstention_threshold": threshold,
                "generated_at": time.strftime("%Y-%m-%d"),
                "answers": answers,
            },
            handle,
            indent=1,
        )


if __name__ == "__main__":
    main()


