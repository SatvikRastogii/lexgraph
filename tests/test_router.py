import numpy as np

from lexgraph.router import (
    COMPLEX_PROTOTYPES,
    SIMPLE_PROTOTYPES,
    SemanticRouter,
    cosine_similarity,
)


class FakeEmbedder:
    """Maps texts to fixed vectors so routing is testable without Ollama."""

    def __init__(self, mapping, default=(0.0, 0.0, 1.0)):
        self.mapping = mapping
        self.default = default

    def _vector(self, text):
        return list(self.mapping.get(text, self.default))

    def embed(self, texts, progress=None):
        return [self._vector(t) for t in texts]

    def embed_one(self, text):
        return self._vector(text)


def _router(query_vector):
    """A router whose simple bank points at x and complex bank at y."""
    mapping = dict.fromkeys(SIMPLE_PROTOTYPES, (1.0, 0.0, 0.0))
    mapping.update(dict.fromkeys(COMPLEX_PROTOTYPES, (0.0, 1.0, 0.0)))
    mapping["Q"] = query_vector
    return SemanticRouter(embedder=FakeEmbedder(mapping))


def test_cosine_similarity_basics():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_query_close_to_simple_bank_routes_naive():
    decision = _router((1.0, 0.0, 0.0)).classify("Q")
    assert decision.route == "NAIVE"
    assert decision.simple_score > decision.complex_score


def test_query_close_to_complex_bank_routes_graph():
    decision = _router((0.0, 1.0, 0.0)).classify("Q")
    assert decision.route == "GRAPH"
    assert decision.complex_score > decision.simple_score


def test_an_exactly_ambiguous_query_does_not_crash():
    decision = _router((1.0, 1.0, 0.0)).classify("Q")
    assert decision.route in {"NAIVE", "GRAPH"}
    assert decision.confidence == 0.0


def test_confidence_grows_with_the_margin():
    clear = _router((1.0, 0.0, 0.0)).classify("Q").confidence
    ambiguous = _router((1.0, 0.9, 0.0)).classify("Q").confidence
    assert clear > ambiguous


def test_confidence_is_bounded_to_one():
    assert _router((1.0, 0.0, 0.0)).classify("Q").confidence <= 1.0


def test_decision_reports_latency():
    assert _router((1.0, 0.0, 0.0)).classify("Q").latency_ms >= 0


def test_prototypes_avoid_cases_absent_from_the_corpus():
    # The old banks were anchored on Maneka Gandhi, Kesavananda Bharati and
    # Puttaswamy, none of which are in input/ -- the router was being tuned on
    # cases it could never retrieve.
    absent = ("maneka gandhi", "kesavananda", "puttaswamy")
    joined = " ".join(SIMPLE_PROTOTYPES + COMPLEX_PROTOTYPES).lower()
    for name in absent:
        assert name not in joined
