"""Composable retrievers, and the named configurations the ablation compares.

Each retriever exposes the same ``search(query, top_k) -> list[Hit]``, so a
configuration is just a composition. Keeping them interchangeable is what lets
the ablation attribute a change in Recall@5 to one specific component rather
than to "the new pipeline".
"""

from __future__ import annotations

import os
import time

from ..chunking import chunk_corpus
from ..corpus import load_corpus
from .base import Hit
from .bm25 import BM25
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker

# Retrieve a deeper shortlist than we return, so fusion and reranking have
# something to work with. Too shallow and the reranker can only reorder what
# dense retrieval already got right.
DEFAULT_CANDIDATES = 20


class SparseRetriever:
    """BM25 over the same chunks the dense index holds."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = BM25([chunk.indexed_text() for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        hits = []
        for position, score in self.index.search(query, top_k):
            chunk = self.chunks[position]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.indexed_text(),
                    score=score,
                    title=chunk.title,
                    year=chunk.year,
                    para_label=chunk.para_label,
                    components={"bm25": score},
                )
            )
        return hits


class HybridRetriever:
    """Dense and sparse rankings fused with RRF."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        candidates: int = DEFAULT_CANDIDATES,
        weights: tuple[float, float] = (1.0, 1.0),
    ):
        self.dense = dense
        self.sparse = sparse
        self.candidates = candidates
        self.weights = weights

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        depth = max(self.candidates, top_k)
        return reciprocal_rank_fusion(
            [self.dense.search(query, depth), self.sparse.search(query, depth)],
            weights=list(self.weights),
            top_k=top_k,
        )


class RerankedRetriever:
    """Wraps any retriever, deepening its shortlist and reordering it."""

    def __init__(self, base, reranker: CrossEncoderReranker, candidates: int = DEFAULT_CANDIDATES):
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        shortlist = self.base.search(query, max(self.candidates, top_k))
        return self.reranker.rerank(query, shortlist, top_k=top_k)


# The configurations the ablation reports. Each isolates one change from the
# one above it, so a difference in the numbers has a single cause.
CONFIGURATIONS = {
    "dense-fixed": "Dense only, 500-word fixed chunks (the original baseline)",
    "dense": "Dense only, paragraph-aware chunks",
    "bm25": "BM25 only, paragraph-aware chunks",
    "hybrid": "Dense + BM25 fused with RRF",
    "hybrid-rerank": "Dense + BM25 fused, then cross-encoder reranked",
    "graph-units": "GraphRAG's own text units, same embedder (graph-free control)",
    "graph-community": "GraphRAG community reports, same embedder",
    "hyde-hybrid": "Hybrid, searched with a generated hypothetical answer",
    "hyde-rerank": "HyDE shortlist, reranked against the original question",
}

# Cost one generation per query, so they are excluded from the default sweep.
HYDE_CONFIGURATIONS = ("hyde-hybrid", "hyde-rerank")

# Need output/*.parquet from `graphrag index`, which is not committed. They are
# skipped rather than failed when the index is absent, so the ablation still
# runs on a clean checkout.
GRAPH_CONFIGURATIONS = ("graph-units", "graph-community")


def _dense(strategy: str, chroma_dir: str):
    """The dense half, from whichever backend this environment has.

    ``chroma`` is the local default: incremental, two collections, built by
    nomic-embed-text on the GPU. ``numpy`` is the deployed one: a 2MB committed
    matrix and a hosted query embedder, because no free host has a GPU to run
    the local model on.

    Deliberately not a separate named configuration. The deployment must run
    the same `hybrid-rerank` the benchmark reports, or the numbers on the
    dashboard stop describing the thing serving the answers.
    """
    backend = os.getenv("LEXGRAPH_DENSE_BACKEND", "chroma").lower()
    if backend == "chroma":
        return DenseRetriever(strategy=strategy, chroma_dir=chroma_dir)
    if backend == "numpy":
        from ..embeddings import build_embedder
        from .vectors import DEFAULT_STORE, NumpyDenseRetriever, store_path

        directory = os.getenv("LEXGRAPH_VECTOR_STORE", DEFAULT_STORE)
        return NumpyDenseRetriever(store_path(directory, strategy), build_embedder())
    raise ValueError(
        f"unknown dense backend {backend!r}; expected 'chroma' or 'numpy'"
    )


def build_retriever(
    name: str,
    chroma_dir: str = "chroma_db",
    input_dir: str = "input",
    hyde_generator: str | None = None,
):
    """Construct one named configuration.

    Shared pieces are built per call rather than cached globally; the ablation
    builds each configuration once and reuses it across all questions.
    """
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name!r}; expected one of {list(CONFIGURATIONS)}")

    if name in GRAPH_CONFIGURATIONS:
        from .graph import GraphRetriever

        return GraphRetriever(kind=name.split("-", 1)[1])

    strategy = "fixed" if name == "dense-fixed" else "paragraph"

    if name in ("dense", "dense-fixed"):
        return _dense(strategy, chroma_dir)

    chunks = chunk_corpus(load_corpus(input_dir), strategy)

    if name == "bm25":
        return SparseRetriever(chunks)

    hybrid = HybridRetriever(
        _dense(strategy, chroma_dir),
        SparseRetriever(chunks),
    )
    if name == "hybrid":
        return hybrid

    if name in HYDE_CONFIGURATIONS:
        from ..llm import DEFAULT_HYDE_GENERATOR, get_client
        from .hyde import HyDEReranked, HyDERetriever

        hyde = HyDERetriever(hybrid, get_client(hyde_generator or DEFAULT_HYDE_GENERATOR))
        if name == "hyde-hybrid":
            return hyde
        return HyDEReranked(hyde, CrossEncoderReranker())

    return RerankedRetriever(hybrid, CrossEncoderReranker())


def timed_search(retriever, query: str, top_k: int = 5) -> tuple[list[Hit], float]:
    """``(hits, latency_ms)`` for one query."""
    started = time.perf_counter()
    hits = retriever.search(query, top_k)
    return hits, (time.perf_counter() - started) * 1000

