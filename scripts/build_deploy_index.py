"""Build the committed vector store the deployment queries.

The local index is 84MB of ChromaDB built by an embedder that needs a GPU, and
no free host has one. This writes the committed alternative: a single dense
matrix small enough to live in the repository.

The default embedder is ONNX on CPU. Gemini was the obvious choice and is not
viable: its free tier allows 1,000 embed requests per day and counts a batch of
32 as 32 requests, so a 1,384-chunk index does not fit inside one day's quota,
and every live query would afterwards compete with rebuilds for what was left.

    python scripts/build_deploy_index.py                 # fastembed, paragraph
    python scripts/build_deploy_index.py --dry-run       # size it, spend nothing

``--dry-run`` reports what would be embedded and how large the store would be
without making a single request.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.chunking import chunk_corpus
from lexgraph.corpus import load_corpus
from lexgraph.embeddings import build_embedder
from lexgraph.retrieval.vectors import DEFAULT_STORE, save_store, store_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedder", default="fastembed",
        choices=["fastembed", "gemini", "ollama"],
    )
    parser.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed"])
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--store-dir", default=DEFAULT_STORE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    documents = load_corpus(args.input_dir)
    chunks = chunk_corpus(documents, args.strategy)
    print(f"{len(documents)} documents -> {len(chunks)} {args.strategy} chunks")

    if args.dry_run:
        words = sum(len(c.text.split()) for c in chunks)
        print(f"would embed ~{words:,} words with {args.embedder}")
        for dims in (768, 3072):
            print(f"  store at {dims} dims, float16: "
                  f"{len(chunks) * dims * 2 / 1024 / 1024:.1f} MB")
        return

    embedder = build_embedder(args.embedder)
    embedder.health_check()
    dimensions = getattr(embedder, "dimensions", None)

    started = time.perf_counter()
    vectors = embedder.embed(
        # The same indexed_text() the local pipeline embeds: case name, year
        # and paragraph prepended. Deployment must not quietly index different
        # text from the one the reported numbers were measured on.
        [chunk.indexed_text() for chunk in chunks],
        progress=lambda done, total: (
            done % 128 == 0 or done == total
        ) and print(f"  embedded {done}/{total}"),
    )
    elapsed = time.perf_counter() - started

    if dimensions is None:
        dimensions = len(vectors[0])

    path = store_path(args.store_dir, args.strategy)
    save_store(path, chunks, vectors, embedder.model, dimensions)

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\nWrote {path}")
    print(f"  {len(chunks)} chunks x {dimensions} dims, {size_mb:.1f} MB, "
          f"embedded in {elapsed:.0f}s")
    print(f"  embedder: {embedder.model}")
    print("\nThis file is committed. The abstention threshold is calibrated per "
          "score scale,\nso re-run scripts/calibrate_abstention.py before "
          "trusting the guardrail on it.")


if __name__ == "__main__":
    main()
