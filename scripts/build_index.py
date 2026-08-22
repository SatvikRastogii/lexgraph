"""Build the vector store for the naive/hybrid retrieval pipelines.

Indexes ``input/`` -- the same 40 judgments GraphRAG indexes -- so the two
pipelines are finally comparable. One collection is written per chunking
strategy, which is what lets the ablation isolate the effect of chunking.

    python scripts/build_index.py                  # both strategies
    python scripts/build_index.py --strategy fixed
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.chunking import chunk_corpus
from lexgraph.corpus import load_corpus
from lexgraph.retrieval.dense import DEFAULT_CHROMA_DIR, build_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph", "both"],
        default="both",
        help="chunking strategy to index (default: both, for the ablation)",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    documents = load_corpus(args.input_dir)
    if not documents:
        sys.exit(f"No documents found in {args.input_dir!r}")

    print(f"Corpus: {len(documents)} judgments from {args.input_dir}/")
    unique_titles = {d.short_title.lower() for d in documents}
    if len(unique_titles) < len(documents):
        print(
            f"  note: {len(documents) - len(unique_titles)} duplicate case(s) present; "
            "kept so this index matches the GraphRAG index exactly"
        )

    strategies = ["fixed", "paragraph"] if args.strategy == "both" else [args.strategy]

    for strategy in strategies:
        chunks = chunk_corpus(documents, strategy, args.chunk_size, args.overlap)
        print(f"\n[{strategy}] {len(chunks)} chunks")
        started = time.perf_counter()
        build_collection(chunks, strategy, chroma_dir=args.chroma_dir)
        print(f"[{strategy}] done in {time.perf_counter() - started:.1f}s")

    print("\nIndex build complete.")


if __name__ == "__main__":
    main()
