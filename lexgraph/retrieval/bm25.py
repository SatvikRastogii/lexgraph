"""Okapi BM25, implemented directly rather than pulled in as a dependency.

Dense retrieval is weak at exactly the tokens legal queries turn on: article
numbers, section numbers, case names and reported-citation strings. An
embedding places "Article 21" and "Article 19" in near-identical space; BM25
does not. The two are complementary, which is why the pipeline fuses them.

Scoring is standard Okapi BM25:

    score(D, Q) = sum_i  IDF(q_i) * (f_i * (k1 + 1))
                          / (f_i + k1 * (1 - b + b * |D| / avgdl))

    IDF(q)      = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

with the standard k1=1.5, b=0.75. The IDF form above is the one that stays
positive for terms appearing in more than half the corpus, which matters here
because words like "court" and "article" are in almost every chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Deliberately short. Aggressive stopword removal hurts legal search, where
# phrases like "right to life" and "in re" carry meaning.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were
    be been being it its this that these those which who whom whose what
    """.split()
)

# "Article 21", "article 19(1)(a)" -> a single token, so an article reference
# survives as one high-IDF term instead of decomposing into "article" (in
# nearly every chunk) plus a bare number.
ARTICLE_REFERENCE = re.compile(r"\barticles?\s+(\d{1,3}[a-z]?(?:\(\d+\))*(?:\([a-z]\))*)")


ARTICLE_PARENT = re.compile(r"\d{1,3}[a-z]?")


def _fold_article(match: re.Match) -> str:
    """Expand an article reference into its parent and its full form.

    ``Article 19(1)(a)`` yields both ``article19`` and ``article191a``. Keeping
    only the full form would mean a search for "Article 19" missed every
    sub-clause citation of it, which is the common case in judgment text;
    keeping only the parent would lose the precision of the sub-clause.
    """
    reference = match.group(1)
    parent = ARTICLE_PARENT.match(reference).group(0)
    full = reference.replace("(", "").replace(")", "")
    return f" article{parent} article{full} " if full != parent else f" article{parent} "


def tokenize(text: str) -> list[str]:
    """Lowercase, fold article references, split on non-alphanumerics."""
    folded = ARTICLE_REFERENCE.sub(_fold_article, text.lower())
    return [t for t in TOKEN_PATTERN.findall(folded) if t not in STOPWORDS]


class BM25:
    """A BM25 index over a fixed list of documents.

    ``search`` returns ``(document_index, score)`` pairs, highest first. Only
    documents sharing at least one query term are scored, so an unrelated
    query is cheap rather than a full scan.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)

        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 0.0
        )

        # term -> {doc_index: term_frequency}
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, tokens in enumerate(self.doc_tokens):
            for term, frequency in Counter(tokens).items():
                self.postings[term][index] = frequency

        self.idf = {
            term: math.log(
                1 + (self.corpus_size - len(docs) + 0.5) / (len(docs) + 0.5)
            )
            for term, docs in self.postings.items()
        }

    def score(self, query: str) -> dict[int, float]:
        """Raw BM25 score per matching document index."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for index, frequency in postings.items():
                norm = 1 - self.b + self.b * (self.doc_lengths[index] / self.average_length)
                scores[index] += idf * (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * norm
                )
        return dict(scores)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Top-``top_k`` ``(document_index, score)`` pairs, highest score first."""
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]
