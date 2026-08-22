"""Loading and parsing of the judgment corpus.

Every file in ``input/`` carries a header block written by ``scraper.py``::

    CASE TITLE: Rupa Ashok Hurra vs Ashok Hurra & Anr on 10 April, 2002
    YEAR: 2002
    ARTICLES CITED: Article 32, Article 142, ...
    ======================================================================
    <judgment text>

The header is worth parsing rather than discarding: the case title and year
are exactly the context a retrieved chunk usually lacks, and the retrieval
pipeline prepends them to each chunk before embedding.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

HEADER_SEPARATOR = re.compile(r"^={10,}\s*$", re.MULTILINE)

# Article references appear as "Article 21", "Article 19(1)(a)", "Article 21-A".
ARTICLE_PATTERN = re.compile(r"\bArticle\s+(\d{1,3}[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*)")


@dataclass
class Document:
    """One judgment, with its scraper-written header parsed out."""

    filename: str
    body: str
    title: str = "Unknown Case"
    year: str = "unknown"
    article_focus: str = ""
    url: str = ""
    articles_cited: list[str] = field(default_factory=list)
    bench_type: str = ""
    has_dissent: bool = False

    @property
    def short_title(self) -> str:
        """Case name without the trailing ``on <date>`` that Indian Kanoon appends."""
        return re.split(r"\s+on\s+\d", self.title)[0].strip() or self.title

    @property
    def citation_label(self) -> str:
        """Human-facing label used in answers and citation lists."""
        return f"{self.short_title} ({self.year})" if self.year != "unknown" else self.short_title


def _parse_header(header: str) -> dict:
    """Turn the ``KEY: value`` header block into a dict of known fields."""
    fields = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().upper()] = value.strip()

    articles = [
        a.strip()
        for a in fields.get("ARTICLES CITED", "").split(",")
        if a.strip() and a.strip().lower() != "none"
    ]

    return {
        "title": fields.get("CASE TITLE") or "Unknown Case",
        "year": fields.get("YEAR") or "unknown",
        "article_focus": fields.get("PRIMARY ARTICLE", ""),
        "url": fields.get("SOURCE", ""),
        "articles_cited": articles,
        "bench_type": fields.get("BENCH TYPE", ""),
        "has_dissent": fields.get("HAS DISSENT", "").strip().lower() == "yes",
    }


def parse_document(filename: str, text: str) -> Document:
    """Split a raw judgment file into header metadata and body text.

    Files without a header separator are still usable; they just carry
    default metadata and the whole file as the body.
    """
    parts = HEADER_SEPARATOR.split(text, maxsplit=1)
    if len(parts) == 2:
        header, body = parts
        return Document(filename=filename, body=body.strip(), **_parse_header(header))
    return Document(filename=filename, body=text.strip())


def load_corpus(input_dir: str = "input", min_chars: int = 500) -> list[Document]:
    """Load every ``.txt`` judgment in ``input_dir``, sorted by filename.

    ``input/`` is the authoritative corpus for BOTH pipelines. GraphRAG indexes
    this directory directly, so the vector store must be built from it too --
    reading from anywhere else silently makes the two pipelines incomparable.
    """
    documents = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(input_dir, filename)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        if len(text) < min_chars:
            continue
        documents.append(parse_document(filename, text))
    return documents


def articles_in(text: str) -> list[str]:
    """Unique article references in ``text``, in order of first appearance."""
    seen, ordered = set(), []
    for match in ARTICLE_PATTERN.findall(text):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered
