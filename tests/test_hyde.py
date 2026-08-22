from lexgraph.retrieval.base import Hit
from lexgraph.retrieval.hyde import HyDEReranked, HyDERetriever

HITS = [Hit(chunk_id="c1", doc_id="judgment_0001.txt", text="history sheet", score=0.9)]


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        return HITS


class StubGenerator:
    def __init__(self, response="The maintenance of a history sheet under Standing Order 601."):
        self.response = response
        self.calls = 0

    def chat(self, prompt, max_tokens=512, temperature=0.0):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class RecordingReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, hits, top_k=5):
        self.queries.append(query)
        return hits[:top_k]


def test_search_uses_question_plus_hypothesis():
    base, generator = RecordingRetriever(), StubGenerator()
    HyDERetriever(base, generator).search("can police keep a watch file?")

    searched = base.queries[0]
    assert "can police keep a watch file?" in searched, "the question must anchor the search"
    assert "history sheet" in searched, "the hypothetical must reach the retriever"


def test_hypothesis_is_generated_once_per_question():
    base, generator = RecordingRetriever(), StubGenerator()
    hyde = HyDERetriever(base, generator)
    hyde.search("q")
    hyde.search("q")
    assert generator.calls == 1


def test_a_dead_generator_degrades_to_plain_retrieval():
    # HyDE is an enhancement. A generator that is down or rate limited must
    # cost the enhancement, not the query.
    base = RecordingRetriever()
    hyde = HyDERetriever(base, StubGenerator(RuntimeError("connection refused")))
    hits = hyde.search("what is a curative petition?")

    assert hits == HITS
    assert base.queries == ["what is a curative petition?"]


def test_empty_hypothesis_falls_back_to_the_question():
    base = RecordingRetriever()
    HyDERetriever(base, StubGenerator("   ")).search("q")
    assert base.queries == ["q"]


def test_reranking_scores_against_the_original_question():
    # Ranking judgments by how much they resemble a passage the model invented
    # would optimise for the wrong target; the hypothetical is a bridge to the
    # corpus vocabulary, not a statement of what was asked.
    base, reranker = RecordingRetriever(), RecordingReranker()
    hyde = HyDERetriever(base, StubGenerator())
    HyDEReranked(hyde, reranker).search("can police keep a watch file?")

    assert reranker.queries == ["can police keep a watch file?"]
    assert "history sheet" in base.queries[0], "but the shortlist still uses HyDE"
