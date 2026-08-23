"""Tests for the GraphRAG retrieval arms.

The parquet-backed parts skip when output/ is absent, since the GraphRAG index
is an hours-long offline step and is not committed. The document expansion
logic is tested without it, because that is where the measurement hazard is.
"""

import os

import numpy as np
import pytest

from lexgraph.retrieval.graph import GraphRetriever, load_units

GRAPH_ROOT = "output"

def _graph_artefacts_readable():
    """Both halves: the parquets have to exist *and* be readable.

    They were gitignored when these tests were written, so a missing directory
    was the only way to be unable to read them. Committing them flipped that:
    the files are now always present, the skip stopped firing, and the tests
    began failing on an environment that simply has no pandas rather than
    skipping as they were meant to.
    """
    if not os.path.exists(os.path.join(GRAPH_ROOT, "text_units.parquet")):
        return False
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


requires_graph = pytest.mark.skipif(
    not _graph_artefacts_readable(),
    reason="GraphRAG parquets not present, or pandas not installed",
)


class FakeEmbedder:
    """Embeds by keyword presence, so similarity is predictable."""

    model = "fake"
    VOCAB = ["prison", "curative", "reservation"]

    def embed(self, texts, progress=None):
        return [self._vector(t) for t in texts]

    def embed_one(self, text):
        return self._vector(text)

    def _vector(self, text):
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in self.VOCAB] + [0.01]


def _retriever(tmp_path, documents, texts):
    retriever = GraphRetriever.__new__(GraphRetriever)
    retriever.kind = "community"
    retriever.embedder = FakeEmbedder()
    retriever.unit_ids = [f"u{i}" for i in range(len(texts))]
    retriever.texts = texts
    retriever.documents = documents
    matrix = np.asarray(retriever.embedder.embed(texts), dtype=np.float32)
    retriever.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return retriever


def test_a_broad_unit_expands_to_every_document_it_covers(tmp_path):
    # A community report is evidence for every judgment it was built from.
    # Returning only the first would understate what global search surfaces.
    retriever = _retriever(
        tmp_path,
        documents=[["a.txt", "b.txt", "c.txt"], ["d.txt"]],
        texts=["prison conditions", "reservation policy"],
    )
    hits = retriever.search("prison", top_k=10)
    assert [h.doc_id for h in hits[:3]] == ["a.txt", "b.txt", "c.txt"]


def test_expansion_consumes_the_top_k_budget(tmp_path):
    # The cost has to be visible: a report spanning many judgments really does
    # fill many slots, and counting it as one would hide the inflation.
    retriever = _retriever(
        tmp_path,
        documents=[["a.txt", "b.txt", "c.txt"], ["d.txt"]],
        texts=["prison conditions", "prison discipline"],
    )
    hits = retriever.search("prison", top_k=2)
    assert len(hits) == 2
    assert "d.txt" not in [h.doc_id for h in hits]


def test_a_document_is_never_returned_twice(tmp_path):
    retriever = _retriever(
        tmp_path,
        documents=[["a.txt", "b.txt"], ["a.txt"], ["c.txt"]],
        texts=["prison one", "prison two", "curative"],
    )
    doc_ids = [h.doc_id for h in retriever.search("prison", top_k=10)]
    assert len(doc_ids) == len(set(doc_ids))


def test_breadth_reports_mean_documents_per_unit(tmp_path):
    retriever = _retriever(
        tmp_path,
        documents=[["a.txt", "b.txt", "c.txt"], ["d.txt"]],
        texts=["prison", "curative"],
    )
    assert retriever.breadth == 2.0


def test_missing_index_raises_a_useful_error():
    with pytest.raises(FileNotFoundError, match="graphrag index"):
        load_units("units", root="no-such-directory")


def test_unknown_unit_kind_is_rejected():
    with pytest.raises(ValueError):
        load_units("entities")


@requires_graph
def test_every_text_unit_maps_to_a_corpus_filename():
    rows = load_units("units", GRAPH_ROOT)
    assert rows, "the index has text units"
    # Filenames, not GraphRAG's content-hash ids: the gold set is annotated
    # against input/, and a translation table on one side of the comparison is
    # exactly where a mismatch would hide.
    assert all(
        name.startswith("judgment_") and name.endswith(".txt")
        for _, _, names in rows
        for name in names
    )


@requires_graph
def test_text_units_belong_to_exactly_one_document():
    rows = load_units("units", GRAPH_ROOT)
    assert all(len(names) == 1 for _, _, names in rows)


@requires_graph
def test_community_reports_span_several_documents():
    # If they did not, they would just be chunks, and the community arm would
    # not be measuring anything the units arm does not.
    rows = load_units("community", GRAPH_ROOT)
    assert max(len(names) for _, _, names in rows) > 1
