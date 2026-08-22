"""Verify that the citations in an answer appear in the retrieved context.

The failure mode that matters in legal RAG is not a clumsy answer, it is a
confident one citing a case that was never retrieved -- or does not exist.
Lawyers have been sanctioned for filing submissions containing hallucinated
citations, so this is the domain's characteristic risk rather than a generic
one, and it is worth checking directly instead of hoping a faithfulness score
catches it.

Every case name, article reference, section reference and reported citation in
the answer is extracted and looked for in the context the generator was
actually given. Anything missing is reported as unsupported.

This is deliberately a lexical check, not a semantic one: it should flag "the
answer named something the context does not contain", which is exactly a
string question. Paraphrase is the generator's business, not the verifier's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "Maneka Gandhi vs Union of India", "R. Rajagopal v. Jayalalitha".
# Deliberately case-SENSITIVE: [A-Z] is doing real work here by requiring
# proper nouns. Under re.IGNORECASE it would match any letter and the pattern
# would happily read "the case vs the court" as a citation.
CASE_NAME = re.compile(
    r"\b([A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,5})"
    r"\s+(?:[vV][sS]?\.?|[vV]ersus)\s+"
    r"([A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,5})"
)
ARTICLE_REFERENCE = re.compile(r"\bArticle\s+(\d{1,3}[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*)", re.I)
SECTION_REFERENCE = re.compile(r"\bSection\s+(\d{1,4}[A-Z]?(?:\(\d+\))*)", re.I)
# "AIR 1978 SC 597", "(2006) 4 SCC 1", "1981 Cri LJ 477".
# No leading \b: a boundary can never match before "(", which silently hid
# every SCC-style citation.
REPORTED_CITATION = re.compile(
    r"(?:\bAIR\s+\d{4}\s+[A-Z]{2,4}\s+\d+"
    r"|\(\d{4}\)\s*\d+\s*[A-Z]{2,4}\s*\d+"
    r"|\b\d{4}\s+Cri\s+LJ\s+\d+)",
    re.I,
)

# Words that survive the case-name regex but never identify a case.
CASE_NAME_NOISE = frozenset(
    {"the", "this", "that", "court", "state", "union", "india", "petitioner",
     "respondent", "appellant", "case", "cases", "judgment", "act", "section"}
)


@dataclass
class CitationReport:
    """Which of an answer's citations the context supports."""

    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.supported) + len(self.unsupported)

    @property
    def accuracy(self) -> float:
        """Fraction of citations found in context; 1.0 when none were made."""
        return len(self.supported) / self.total if self.total else 1.0

    @property
    def has_unsupported(self) -> bool:
        return bool(self.unsupported)

    def warning(self) -> str:
        if not self.unsupported:
            return ""
        listed = ", ".join(self.unsupported[:4])
        more = f" (and {len(self.unsupported) - 4} more)" if len(self.unsupported) > 4 else ""
        return (
            f"{len(self.unsupported)} citation(s) in this answer do not appear in the "
            f"retrieved judgments and may be fabricated: {listed}{more}"
        )


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _party_tokens(name: str) -> list[str]:
    """Distinctive words from one side of a case name."""
    return [
        token
        for token in _normalise(name).split()
        if len(token) > 2 and token not in CASE_NAME_NOISE
    ]


def extract_citations(answer: str) -> dict[str, list[str]]:
    """Pull every citation-like reference out of an answer."""
    cases = []
    for left, right in CASE_NAME.findall(answer):
        # One distinctive side is enough. Requiring both would discard the
        # commonest citation form in this corpus -- "X vs Union of India",
        # "Y vs State of Kerala" -- whose right-hand side is always a noise
        # word. A match where NEITHER side is distinctive ("The State vs The
        # Court") is not a citation and is dropped.
        if _party_tokens(left) or _party_tokens(right):
            cases.append(f"{left.strip()} v. {right.strip()}")

    return {
        "cases": _unique(cases),
        "articles": _unique(f"Article {a}" for a in ARTICLE_REFERENCE.findall(answer)),
        "sections": _unique(f"Section {s}" for s in SECTION_REFERENCE.findall(answer)),
        "reported": _unique(REPORTED_CITATION.findall(answer)),
    }


def _unique(items) -> list[str]:
    seen, ordered = set(), []
    for item in items:
        key = _normalise(item)
        if key and key not in seen:
            seen.add(key)
            ordered.append(item.strip() if isinstance(item, str) else item)
    return ordered


def verify_citations(answer: str, contexts: list[str]) -> CitationReport:
    """Check each citation in ``answer`` against the retrieved ``contexts``."""
    haystack = _normalise(" ".join(contexts))
    report = CitationReport()
    citations = extract_citations(answer)

    for case in citations["cases"]:
        left, _, right = case.partition(" v. ")
        # A case counts as supported when a distinctive party name is present.
        # Requiring both sides is too strict: judgments routinely shorten
        # "Maneka Gandhi vs Union of India" to "Maneka Gandhi".
        tokens = _party_tokens(left) or _party_tokens(right)
        if tokens and all(token in haystack for token in tokens):
            report.supported.append(case)
        else:
            report.unsupported.append(case)

    for reference in citations["articles"] + citations["sections"] + citations["reported"]:
        (report.supported if _normalise(reference) in haystack else report.unsupported).append(
            reference
        )

    return report
