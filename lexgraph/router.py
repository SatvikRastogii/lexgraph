"""Semantic router: send a query to vector retrieval or to the graph.

Single-hop factual questions are answered well and cheaply by vector
retrieval. Multi-hop and corpus-wide questions are what a graph index is for,
and it costs orders of magnitude more per query. Choosing between them is
worth doing, and worth doing without an LLM call.

Two banks of prototype questions are embedded once at startup. An incoming
query is embedded and compared against both; the closer bank wins. That is one
embedding call rather than a generation, and on this hardware a generation
costs seconds.

This module previously existed twice -- as hybrid_router.py and inlined again
in app.py -- with a comment warning that changes had to be mirrored by hand.
Both copies now live here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .embeddings import OllamaEmbedder

SIMPLE_PROTOTYPES = [
    "What does Article 21 of the Indian Constitution guarantee?",
    "What is the right to equality under Article 14?",
    "What year was this case decided?",
    "Who was the petitioner in this case?",
    "What did the court hold regarding preventive detention?",
    "What freedoms are protected under Article 19?",
    "What remedies does Article 32 provide?",
    "What is a curative petition?",
    "Define reasonable restrictions under Article 19(2).",
    "What are the grounds for granting default bail?",
]

COMPLEX_PROTOTYPES = [
    "How are Articles 14, 19, and 21 interconnected across these judgments?",
    "How has the interpretation of Article 21 evolved over time?",
    "Which legal principles from earlier cases were expanded in later ones?",
    "What is the relationship between these three cases?",
    "How has the court balanced individual rights against state power across decades?",
    "Which cases form the foundational lineage of this doctrine?",
    "Compare the approaches of different benches to the scope of personal liberty.",
    "What patterns exist in how the court interprets reasonable restrictions?",
    "In which cases did dissenting opinions later become the majority view?",
    "What themes recur across the whole corpus of judgments?",
]

# The margin between banks on real (non-prototype) queries runs to roughly 0.3.
# Confidence is scaled against that so a healthy margin does not read as
# uncertain. A ratio-of-sums formula was tried and compressed every margin
# toward zero, because both banks score high for any in-domain legal query.
MARGIN_SCALE = 0.3


@dataclass
class RoutingDecision:
    route: str
    confidence: float
    simple_score: float
    complex_score: float
    latency_ms: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


class SemanticRouter:
    """Routes a query to NAIVE or GRAPH by prototype similarity."""

    def __init__(self, embedder: OllamaEmbedder | None = None):
        self.embedder = embedder or OllamaEmbedder()
        self.simple = np.array(self.embedder.embed(SIMPLE_PROTOTYPES))
        self.complex = np.array(self.embedder.embed(COMPLEX_PROTOTYPES))

    def classify(self, query: str) -> RoutingDecision:
        started = time.perf_counter()
        embedded = np.array(self.embedder.embed_one(query))

        # Best match per bank, not an average over the top few. Averaging let a
        # query that resembles several mediocre complex prototypes outvote one
        # excellent simple-bank match; measured 21/22 against 20/22 on the
        # benchmark questions, with no regressions.
        simple_score = max(cosine_similarity(embedded, p) for p in self.simple)
        complex_score = max(cosine_similarity(embedded, p) for p in self.complex)

        margin = abs(simple_score - complex_score)
        return RoutingDecision(
            route="NAIVE" if simple_score > complex_score else "GRAPH",
            confidence=round(min(1.0, margin / MARGIN_SCALE), 4),
            simple_score=round(simple_score, 4),
            complex_score=round(complex_score, 4),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
