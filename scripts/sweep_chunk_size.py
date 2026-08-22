"""Is 500 words the right chunk size, or just the one settings.yaml had?

DEFAULT_CHUNK_WORDS was inherited from GraphRAG's config so the two indexes
would stay comparable. That is a good reason to keep one strategy pinned to it
and no reason at all to believe it is optimal, and it has never been measured.

The sweep runs on BM25 because BM25 builds its index in memory from `input/` in
about a second, so the whole curve costs no embedding calls and no ChromaDB
rebuild. That is the trade: it answers "how does chunk size move retrieval"
cheaply and does not answer it for the dense retriever, whose 500-word setting
would need one four-minute index build per point.

Read the result as a direction to test, not a conclusion to ship. Chunk size
interacts with the retriever -- a cross-encoder reranker cares about how much
context each candidate carries in a way BM25 does not -- so a size that wins
here is a hypothesis about the dense pipeline, not a finding about it.

    python scripts/sweep_chunk_size.py
    python scripts/sweep_chunk_size.py --sizes 200 300 500 800
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.chunking import chunk_corpus
from lexgraph.corpus import load_corpus
from lexgraph.eval.retrieval_metrics import aggregate, bootstrap_ci, score_query
from lexgraph.retrieval.pipeline import SparseRetriever

GOLDSET_PATH = os.path.join("data", "goldset.json")
RESULTS_PATH = os.path.join("reports", "chunk_size_sweep.json")
DEFAULT_SIZES = (150, 250, 350, 500, 750, 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed"])
    parser.add_argument("--overlap-ratio", type=float, default=0.1,
                        help="overlap as a fraction of chunk size, held constant")
    parser.add_argument("--results-file", default=RESULTS_PATH)
    args = parser.parse_args()

    with open(args.goldset, encoding="utf-8") as handle:
        questions = [q for q in json.load(handle)["questions"] if q["answerable"]]

    documents = load_corpus("input")
    print(f"{len(documents)} documents, {len(questions)} answerable questions, "
          f"{args.strategy} chunking\n")

    header = (f"{'words':>7}{'chunks':>9}{'R@1':>8}{'R@5':>8}{'nDCG@10':>10}"
              f"{'MRR':>8}{'R@5 95% CI':>18}{'build':>9}")
    print(header)
    print("-" * len(header))

    results = {}
    for size in args.sizes:
        # Overlap scales with size rather than staying at 50 words, otherwise
        # the sweep varies two things at once and neither can be attributed.
        overlap = int(size * args.overlap_ratio)

        started = time.perf_counter()
        chunks = chunk_corpus(documents, args.strategy, size, overlap)
        retriever = SparseRetriever(chunks)
        build_seconds = time.perf_counter() - started

        per_question = []
        for question in questions:
            hits = retriever.search(question["question"], 10)
            per_question.append(
                score_query([h.doc_id for h in hits], question["relevant_docs"])
            )

        means = aggregate(per_question)
        low, high = bootstrap_ci([q["recall@5"] for q in per_question])
        results[size] = {
            "overlap": overlap,
            "chunks": len(chunks),
            "mean_chunk_words": statistics.mean(len(c.text.split()) for c in chunks),
            "metrics": means,
            "recall@5_ci95": [low, high],
            "build_seconds": build_seconds,
        }
        print(f"{size:>7}{len(chunks):>9}{means['recall@1']:>8.3f}"
              f"{means['recall@5']:>8.3f}{means['ndcg@10']:>10.3f}{means['mrr']:>8.3f}"
              f"{f'[{low:.2f}, {high:.2f}]':>18}{build_seconds:>8.1f}s")

    best = max(results, key=lambda s: results[s]["metrics"]["ndcg@10"])
    current = 500
    if current in results:
        delta = results[best]["metrics"]["ndcg@10"] - results[current]["metrics"]["ndcg@10"]
        print(f"\nBest nDCG@10 at {best} words, {delta:+.3f} against the "
              f"{current}-word default.")
        if abs(delta) < 0.02:
            print("That is inside the noise this gold set can resolve; the "
                  "default is not costing anything measurable.")

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {args.results_file}")


if __name__ == "__main__":
    main()
