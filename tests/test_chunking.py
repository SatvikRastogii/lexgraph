import pytest

from lexgraph.chunking import (
    Chunk,
    chunk_document,
    chunk_words,
    split_numbered_paragraphs,
)
from lexgraph.corpus import Document, parse_document


def test_chunk_words_respects_size_and_overlap():
    text = " ".join(str(i) for i in range(250))
    chunks = chunk_words(text, chunk_size=100, overlap=20)

    assert all(len(c.split()) <= 100 for c in chunks)
    # Step is 80, so windows start at 0, 80, 160, 240.
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "80"


def test_chunk_words_overlap_actually_overlaps():
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_words(text, chunk_size=100, overlap=20)
    tail = chunks[0].split()[-20:]
    head = chunks[1].split()[:20]
    assert tail == head


def test_chunk_words_covers_every_word():
    text = " ".join(str(i) for i in range(333))
    joined = " ".join(chunk_words(text, chunk_size=50, overlap=10))
    for word in text.split():
        assert word in joined.split()


def test_chunk_words_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError):
        chunk_words("a b c", chunk_size=10, overlap=10)


def test_chunk_words_handles_empty_text():
    assert chunk_words("") == []


def test_split_numbered_paragraphs_extracts_numbers():
    text = (
        "Preamble text before numbering.\n"
        "1. The first paragraph of the judgment.\n"
        "2. The second paragraph.\n"
        "3. The third paragraph.\n"
        "4. The fourth paragraph.\n"
        "5. The fifth paragraph.\n"
    )
    segments = split_numbered_paragraphs(text)
    numbers = [n for n, _ in segments]

    assert numbers[0] is None, "preamble should be kept with no paragraph number"
    assert numbers[1:] == [1, 2, 3, 4, 5]


def test_split_numbered_paragraphs_falls_back_without_numbering():
    text = "First block of prose.\n\nSecond block of prose.\n\nThird block."
    segments = split_numbered_paragraphs(text)
    assert [n for n, _ in segments] == [None, None, None]
    assert len(segments) == 3


def test_paragraph_chunks_stay_within_budget():
    paragraphs = "\n".join(f"{i}. " + " ".join(["word"] * 60) for i in range(1, 21))
    document = Document(filename="d.txt", body=paragraphs, title="Case", year="1999")
    chunks = chunk_document(document, strategy="paragraph", chunk_size=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text.split()) <= 200


def test_paragraph_chunks_record_their_span():
    paragraphs = "\n".join(f"{i}. " + " ".join(["word"] * 60) for i in range(1, 21))
    document = Document(filename="d.txt", body=paragraphs, title="Case", year="1999")
    chunks = chunk_document(document, strategy="paragraph", chunk_size=200)

    first = chunks[0]
    assert first.para_start == 1
    assert first.para_end >= first.para_start
    assert first.para_label.startswith("para ")


def test_oversized_paragraph_is_windowed_not_dropped():
    body = "1. " + " ".join(["word"] * 900)
    document = Document(filename="d.txt", body=body, title="Case")
    chunks = chunk_document(document, strategy="paragraph", chunk_size=300, overlap=0)

    assert len(chunks) >= 3
    assert all(len(c.text.split()) <= 300 for c in chunks)
    total = sum(len(c.text.split()) for c in chunks)
    assert total >= 900, "no text may be silently dropped"


def test_indexed_text_carries_case_context():
    chunk = Chunk(
        doc_id="judgment_0126.txt",
        chunk_index=7,
        text="The prisoner retains fundamental rights.",
        title="Sunil Batra vs Delhi Administration",
        year="1979",
        article_focus="Article 32",
        para_start=12,
        para_end=15,
    )
    indexed = chunk.indexed_text()

    assert "Sunil Batra" in indexed
    assert "1979" in indexed
    assert "para 12-15" in indexed
    assert chunk.text in indexed


def test_chunk_ids_are_unique_and_stable():
    paragraphs = "\n".join(f"{i}. " + " ".join(["word"] * 60) for i in range(1, 31))
    document = Document(filename="d.txt", body=paragraphs)
    chunks = chunk_document(document, strategy="paragraph", chunk_size=150)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_document(Document(filename="d.txt", body="text"), strategy="semantic")


def test_parse_document_reads_scraper_header():
    raw = (
        "CASE TITLE: Sunil Batra vs Delhi Administration on 20 December, 1979\n"
        "PRIMARY ARTICLE: Article 32\n"
        "SOURCE: https://indiankanoon.org/doc/1263942/\n"
        "YEAR: 1979\n"
        "ARTICLES CITED: Article 32, Article 21, Article 14\n"
        "BENCH TYPE: Constitution Bench (5)\n"
        "HAS DISSENT: Yes\n"
        "======================================================================\n"
        "The body of the judgment begins here.\n"
    )
    document = parse_document("judgment_0126.txt", raw)

    assert document.year == "1979"
    assert document.short_title == "Sunil Batra vs Delhi Administration"
    assert document.articles_cited == ["Article 32", "Article 21", "Article 14"]
    assert document.has_dissent is True
    assert document.body.startswith("The body of the judgment")
    assert "CASE TITLE" not in document.body


def test_parse_document_without_header_keeps_full_text():
    document = parse_document("x.txt", "Just text, no header separator.")
    assert document.body == "Just text, no header separator."
    assert document.title == "Unknown Case"


def test_slivers_are_merged_not_indexed():
    # A one-line paragraph immediately before an oversized one is exactly the
    # shape that used to emit a 2-word chunk.
    body = "1. Short.\n2. " + " ".join(["word"] * 700) + "\n3. Also short.\n"
    document = Document(filename="d.txt", body=body, title="Case")
    chunks = chunk_document(document, strategy="paragraph", chunk_size=300, overlap=0)

    from lexgraph.chunking import MIN_CHUNK_WORDS

    assert all(len(c.text.split()) >= MIN_CHUNK_WORDS for c in chunks)
    assert "Short." in " ".join(c.text for c in chunks), "sliver text must survive the merge"
    assert "Also short." in " ".join(c.text for c in chunks)


def test_single_short_document_is_not_discarded():
    document = Document(filename="d.txt", body="Only five words here total.", title="Case")
    chunks = chunk_document(document, strategy="paragraph", chunk_size=300)
    assert len(chunks) == 1
    assert "Only five words" in chunks[0].text
