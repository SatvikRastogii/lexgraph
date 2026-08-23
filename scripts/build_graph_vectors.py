"""Embed GraphRAG's community reports and entities for the committed store.

The graph is the expensive artefact in this project -- 3,750 entities, 1,505
relationships and 184 community reports, all of it hours of LLM extraction --
and until now the deployment could not use any of it. GraphRAG's own query
engine needs roughly 500MB of dependencies and drives dozens of sequential LLM
calls per question, which is minutes per query and more memory than a free host
has.

What it does not need is that engine. The artefacts are on disk and committed;
embedding them once turns the graph into something a free host can search in
milliseconds. One generation call then reads the result, instead of a
map-reduce over 180 reports.

That is a real difference and the UI says so: this is retrieval over what
GraphRAG produced, not GraphRAG's own global and local search.

    python scripts/build_graph_vectors.py                 # fastembed, both
    python scripts/build_graph_vectors.py --methods global
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.embeddings import build_embedder
from lexgraph.retrieval.graph import GRAPH_METHODS, VECTOR_CACHE, GraphRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", default="fastembed",
                        choices=["fastembed", "gemini", "ollama"])
    parser.add_argument("--methods", nargs="*", default=list(GRAPH_METHODS))
    parser.add_argument("--store-dir", default=VECTOR_CACHE)
    args = parser.parse_args()

    embedder = build_embedder(args.embedder)
    embedder.health_check()
    print(f"embedder: {embedder.model}\n")

    for method in args.methods:
        kind, description = GRAPH_METHODS[method]
        started = time.perf_counter()
        # Constructing it does the work: the store is written on a miss and
        # reused on a hit, so this is also the way to check one is current.
        retriever = GraphRetriever(kind=kind, embedder=embedder,
                                   cache_dir=args.store_dir)
        elapsed = time.perf_counter() - started

        safe = embedder.model.replace("/", "_")
        path = os.path.join(args.store_dir, f"{kind}_{safe}.npz")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[{method}] {kind}: {len(retriever.unit_ids)} units, "
              f"{retriever.breadth:.2f} documents each")
        print(f"  {description}")
        print(f"  {size_mb:.1f} MB in {elapsed:.0f}s -> {path}\n")

    print("These files are committed. The deployment reads them directly and "
          "never re-embeds.")


if __name__ == "__main__":
    main()
