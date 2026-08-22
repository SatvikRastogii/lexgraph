import pytest

from lexgraph.retrieval.base import Hit
from lexgraph.retrieval.bm25 import BM25, tokenize
from lexgraph.retrieval.fusion import reciprocal_rank_fusion


# --- tokenizer ---------------------------------------------------------------

def test_tokenizer_folds_article_references():
    assert "article21" in tokenize("The scope of Article 21 was widened.")
    assert "article19" in tokenize("Article 19 protects speech.")


def test_subclause_reference_also_matches_its_parent_article():
    tokens = tokenize("Article 19(1)(a) protects speech.")
    assert "article19" in tokens, "a search for Article 19 must find 19(1)(a)"
    assert "article191a" in tokens, "the precise sub-clause must stay searchable"


def test_parent_article_query_does_not_gain_subclause_precision():
    # "Article 19" alone must not manufacture a 19(1)(a) token.
    assert "article191a" not in tokenize("Article 19 protects speech.")


def test_article_folding_keeps_articles_distinct():
    # The whole point: Article 21 and Article 19 must not collide, which is
    # exactly what dense embeddings tend to do.
    assert set(tokenize("Article 21")) & set(tokenize("Article 19")) == set()


def test_tokenizer_drops_stopwords_but_keeps_legal_content():
    tokens = tokenize("the right to life and personal liberty")
    assert "the" not in tokens and "to" not in tokens and "and" not in tokens
    assert {"right", "life", "personal", "liberty"} <= set(tokens)


# --- BM25 --------------------------------------------------------------------

CORPUS = [
    "The right to life under Article 21 includes the right to livelihood.",
    "Article 14 guarantees equality before the law to every person.",
    "Preventive detention is governed by Article 22 of the Constitution.",
    "The right to livelihood was recognised in a pavement dwellers case.",
]


def test_bm25_ranks_exact_term_match_first():
    index = BM25(CORPUS)
    top = index.search("preventive detention", top_k=1)
    assert top[0][0] == 2


def test_bm25_article_query_hits_the_right_document():
    index = BM25(CORPUS)
    assert index.search("Article 14", top_k=1)[0][0] == 1
    assert index.search("Article 22", top_k=1)[0][0] == 2


def test_bm25_rare_terms_outweigh_common_ones():
    index = BM25(CORPUS)
    # "detention" is in one document, "livelihood" in two, "right" in two.
    assert index.idf["detention"] > index.idf["livelihood"]
    assert index.idf["livelihood"] == pytest.approx(index.idf["right"])


def test_bm25_unknown_query_returns_nothing():
    index = BM25(CORPUS)
    assert index.search("maritime admiralty jurisdiction") == []


def test_bm25_empty_query_returns_nothing():
    assert BM25(CORPUS).search("") == []


def test_bm25_respects_top_k():
    index = BM25(CORPUS)
    assert len(index.search("right", top_k=2)) == 2


def test_bm25_scores_are_descending():
    scores = [s for _, s in BM25(CORPUS).search("right to life livelihood", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_bm25_length_normalisation_prefers_the_concise_document():
    short = "Article 32 remedies."
    padded = "Article 32 remedies. " + " ".join(["filler"] * 200)
    index = BM25([padded, short])
    # Same term frequency, so the shorter document must score higher.
    assert index.search("Article 32", top_k=1)[0][0] == 1


def test_bm25_handles_empty_corpus():
    assert BM25([]).search("anything") == []


# --- reciprocal rank fusion --------------------------------------------------

def _hit(chunk_id):
    return Hit(chunk_id=chunk_id, doc_id="d.txt", text=chunk_id)


def test_rrf_matches_hand_computed_scores():
    a = [_hit("x"), _hit("y"), _hit("z")]
    b = [_hit("y"), _hit("z"), _hit("x")]
    fused = reciprocal_rank_fusion([a, b], k=60)

    # x: 1/61 + 1/63,  y: 1/62 + 1/61,  z: 1/63 + 1/62  ->  y > x > z
    assert [h.chunk_id for h in fused] == ["y", "x", "z"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_prefers_consensus_over_a_single_first_place():
    # "a" is rank 1 in one list and absent from the other; "b" is rank 2 in
    # both. Agreement should win.
    a = [_hit("a"), _hit("b")]
    b = [_hit("c"), _hit("b")]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert fused[0].chunk_id == "b"


def test_rrf_weights_shift_the_ranking():
    dense = [_hit("d1"), _hit("d2")]
    sparse = [_hit("d2"), _hit("d1")]

    assert reciprocal_rank_fusion([dense, sparse], weights=[3.0, 1.0])[0].chunk_id == "d1"
    assert reciprocal_rank_fusion([dense, sparse], weights=[1.0, 3.0])[0].chunk_id == "d2"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_hit("a")], [_hit("b")]], weights=[1.0])


def test_rrf_deduplicates_by_chunk_id():
    a = [_hit("a"), _hit("b")]
    b = [_hit("a"), _hit("b")]
    fused = reciprocal_rank_fusion([a, b])
    assert len(fused) == 2


def test_rrf_handles_a_single_ranking():
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b")]])
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_respects_top_k():
    a = [_hit(c) for c in "abcdef"]
    assert len(reciprocal_rank_fusion([a], top_k=3)) == 3


def test_rrf_preserves_metadata_from_first_occurrence():
    rich = Hit(chunk_id="a", doc_id="j.txt", text="body", title="Olga Tellis", year="1985")
    bare = Hit(chunk_id="a", doc_id="j.txt", text="body")
    fused = reciprocal_rank_fusion([[rich], [bare]])
    assert fused[0].title == "Olga Tellis"
    assert fused[0].year == "1985"


def test_hit_citation_formats_case_year_and_paragraph():
    hit = Hit(
        chunk_id="c", doc_id="judgment_0126.txt", text="t",
        title="Sunil Batra vs Delhi Administration", year="1979", para_label="para 12-15",
    )
    assert hit.citation == "Sunil Batra vs Delhi Administration (1979), para 12-15"


# --- embedding cache ---------------------------------------------------------

def test_query_embeddings_are_cached_per_process():
    from lexgraph.embeddings import OllamaEmbedder

    embedder = OllamaEmbedder()
    calls = []

    def fake_post(inputs):
        calls.append(list(inputs))
        return [[0.1, 0.2, 0.3] for _ in inputs]

    embedder._post = fake_post

    first = embedder.embed_one("What is a curative petition?")
    second = embedder.embed_one("What is a curative petition?")

    assert first == second
    assert len(calls) == 1, "the second identical query must not hit Ollama again"

    embedder.embed_one("A different question?")
    assert len(calls) == 2
