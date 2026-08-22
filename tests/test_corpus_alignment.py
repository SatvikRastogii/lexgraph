"""Guards against the bug that invalidated the original benchmark.

GraphRAG indexed input/ while the vector store was built from a
keyword-filtered slice of legal_corpus/. The two shared no documents at all,
so every comparison between them measured two different corpora rather than
two retrieval strategies -- and nothing failed, because both pipelines
answered fluently the whole time.

That is the failure mode worth a permanent test: silent, plausible, and
invisible from inside either pipeline.

The integration checks skip when the index has not been built, so CI stays
free of ChromaDB and Ollama.
"""

import os

import pytest

from lexgraph.corpus import load_corpus

INPUT_DIR = "input"
CHROMA_DIR = "chroma_db"


def _corpus_available():
    return os.path.isdir(INPUT_DIR) and any(
        f.endswith(".txt") for f in os.listdir(INPUT_DIR)
    )


requires_corpus = pytest.mark.skipif(
    not _corpus_available(), reason="input/ corpus not present"
)


@requires_corpus
def test_load_corpus_reads_the_directory_graphrag_indexes():
    documents = load_corpus(INPUT_DIR)
    on_disk = {f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")}
    loaded = {d.filename for d in documents}
    assert loaded == on_disk, "load_corpus must read every judgment in input/"


@requires_corpus
def test_every_document_parses_its_header():
    # A document whose header fails to parse loses its case name and year, and
    # those are what the contextual chunk prefix is built from.
    unparsed = [d.filename for d in load_corpus(INPUT_DIR) if d.title == "Unknown Case"]
    assert not unparsed, f"header parse failed for {unparsed}"


@requires_corpus
def test_settings_yaml_chunking_matches_the_fixed_strategy():
    """The fixed chunker exists to mirror GraphRAG's chunking.

    If settings.yaml drifts, `dense-fixed` stops being a like-for-like
    comparison point against the graph index and quietly becomes a different
    experiment.
    """
    import re

    from lexgraph.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS

    if not os.path.exists("settings.yaml"):
        pytest.skip("settings.yaml not present")
    text = open("settings.yaml", encoding="utf-8").read()
    chunking = re.search(r"chunking:(.*?)(?:\n\w|\Z)", text, re.DOTALL)
    assert chunking, "settings.yaml has no chunking block"
    size = re.search(r"size:\s*(\d+)", chunking.group(1))
    overlap = re.search(r"overlap:\s*(\d+)", chunking.group(1))
    assert size and int(size.group(1)) == DEFAULT_CHUNK_WORDS
    assert overlap and int(overlap.group(1)) == DEFAULT_OVERLAP_WORDS


def _chroma_client():
    """A ChromaDB client, or a skip.

    Skips on two conditions, not one: the index may not be built, and chromadb
    may not be installed at all. CI installs neither, and skipping only on the
    missing directory left the import to fail on any machine that had the
    directory but not the package.
    """
    if not os.path.isdir(CHROMA_DIR):
        pytest.skip("chroma_db/ not built; run scripts/build_index.py")
    chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _chroma_collection(name):
    client = _chroma_client()
    if name not in {c.name for c in client.list_collections()}:
        pytest.skip(f"collection {name} not built; run scripts/build_index.py")
    return client.get_collection(name)


@requires_corpus
@pytest.mark.parametrize("collection_name", ["judgments_fixed", "judgments_paragraph"])
def test_vector_store_indexes_exactly_the_graphrag_corpus(collection_name):
    collection = _chroma_collection(collection_name)
    stored = collection.get(include=["metadatas"], limit=1_000_000)
    indexed = {m["doc_id"] for m in stored["metadatas"]}
    expected = {f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")}

    missing = expected - indexed
    extra = indexed - expected
    assert not missing, f"{collection_name} is missing {len(missing)} document(s): {sorted(missing)[:5]}"
    assert not extra, f"{collection_name} indexes {len(extra)} document(s) not in input/: {sorted(extra)[:5]}"


@requires_corpus
def test_no_collection_is_built_from_the_raw_scrape():
    """legal_corpus/ holds 228 documents; only the curated 40 may be indexed."""
    client = _chroma_client()
    expected = {f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")}

    for descriptor in client.list_collections():
        collection = client.get_collection(descriptor.name)
        stored = collection.get(include=["metadatas"], limit=1_000_000)
        if not stored["metadatas"]:
            continue
        key = "doc_id" if "doc_id" in stored["metadatas"][0] else "source"
        indexed = {m.get(key) for m in stored["metadatas"]}
        assert indexed <= expected, (
            f"collection {descriptor.name!r} contains documents outside input/: "
            f"{sorted(indexed - expected)[:5]} -- this is the disjoint-corpus bug"
        )
