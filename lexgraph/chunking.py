"""Chunking strategies for judgment text.

Two strategies are implemented so the ablation can measure the difference
rather than assume it:

``fixed``
    500-word windows with 50-word overlap. Matches ``settings.yaml``'s
    chunking, so it is the strategy to use when comparing against the
    existing GraphRAG index.

``paragraph``
    Splits on the numbered paragraphs that Indian judgments already carry
    ("1.", "2." ...), then packs whole paragraphs up to a word budget. Chunks
    stop mid-argument far less often, and each one knows which paragraphs it
    covers -- so a citation can say "Sunil Batra (1979) para 12-15" instead of
    "judgment_0126 chunk 7".

36 of the 40 corpus documents carry usable paragraph numbering; the rest fall
back to blank-line paragraphs, and any single paragraph over the budget is
split with the fixed windower.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

DEFAULT_CHUNK_WORDS = 500
DEFAULT_OVERLAP_WORDS = 50

# Packing whole paragraphs occasionally flushes a near-empty buffer (a one-line
# paragraph sitting just before an oversized one). Chunks that short carry no
# retrievable signal but still take a slot in top-k, so they are merged into a
# neighbour rather than indexed.
MIN_CHUNK_WORDS = 25

# "12." or "12)" at the start of a line, followed by real text.
NUMBERED_PARAGRAPH = re.compile(r"^[ \t]*(\d{1,3})[.)]\s+(?=\S)", re.MULTILINE)


@dataclass
class Chunk:
    """A retrievable unit of text plus the provenance needed to cite it."""

    doc_id: str
    chunk_index: int
    text: str
    title: str = ""
    year: str = "unknown"
    article_focus: str = ""
    para_start: int | None = None
    para_end: int | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}__{self.chunk_index:04d}"

    @property
    def para_label(self) -> str:
        """``para 12-15``, ``para 12``, or empty when paragraphs are unknown."""
        if self.para_start is None:
            return ""
        if self.para_end is None or self.para_end == self.para_start:
            return f"para {self.para_start}"
        return f"para {self.para_start}-{self.para_end}"

    def indexed_text(self) -> str:
        """Text as embedded and searched.

        The case name and year are prepended because a mid-judgment chunk
        usually never repeats them -- without this, a chunk about the right
        principle is unfindable by the case it belongs to, and the generator
        cannot attribute it. This is the cheap form of contextual retrieval:
        document-level metadata rather than an LLM-written per-chunk summary.
        """
        context = f"Case: {self.title or self.doc_id}"
        if self.year and self.year != "unknown":
            context += f" ({self.year})"
        if self.article_focus:
            context += f" | {self.article_focus}"
        if self.para_label:
            context += f" | {self.para_label}"
        return f"[{context}]\n{self.text}"


def chunk_words(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping fixed-size word windows.

    Word boundaries are preserved. ``overlap`` must be smaller than
    ``chunk_size``, otherwise the window would never advance.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def split_numbered_paragraphs(text: str) -> list[tuple[int | None, str]]:
    """Split judgment text on its own paragraph numbering.

    Returns ``(paragraph_number, text)`` pairs. Any preamble before the first
    numbered paragraph is returned with a number of ``None``. Falls back to
    blank-line splitting when the document carries no usable numbering.
    """
    matches = list(NUMBERED_PARAGRAPH.finditer(text))
    if len(matches) < 5:
        return [(None, block.strip()) for block in re.split(r"\n\s*\n", text) if block.strip()]

    segments: list[tuple[int | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append((None, preamble))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            segments.append((int(match.group(1)), body))
    return segments


def chunk_document(
    document: Document,
    strategy: str = "paragraph",
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk one document with the named strategy."""
    if strategy == "fixed":
        pieces = [(None, None, body) for body in chunk_words(document.body, chunk_size, overlap)]
    elif strategy == "paragraph":
        pieces = _pack_paragraphs(document.body, chunk_size, overlap)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")

    return [
        Chunk(
            doc_id=document.filename,
            chunk_index=index,
            text=body,
            title=document.short_title,
            year=document.year,
            article_focus=document.article_focus,
            para_start=start,
            para_end=end,
        )
        for index, (start, end, body) in enumerate(pieces)
    ]


def _pack_paragraphs(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[int | None, int | None, str]]:
    """Greedily pack whole paragraphs into chunks of at most ``chunk_size`` words."""
    packed: list[tuple[int | None, int | None, str]] = []
    buffer: list[str] = []
    buffer_words = 0
    first_para: int | None = None
    last_para: int | None = None

    def flush():
        nonlocal buffer, buffer_words, first_para, last_para
        if buffer:
            packed.append((first_para, last_para, "\n\n".join(buffer)))
        buffer, buffer_words, first_para, last_para = [], 0, None, None

    for number, body in split_numbered_paragraphs(text):
        words = len(body.split())

        # A single oversized paragraph cannot be packed; window it on its own
        # so the chunk budget still holds.
        if words > chunk_size:
            flush()
            for window in chunk_words(body, chunk_size, overlap):
                packed.append((number, number, window))
            continue

        if buffer_words + words > chunk_size:
            flush()

        if first_para is None:
            first_para = number
        if number is not None:
            last_para = number
        buffer.append(body)
        buffer_words += words

    flush()
    return _merge_slivers(packed, chunk_size)


def _merge_slivers(
    packed: list[tuple[int | None, int | None, str]], chunk_size: int
) -> list[tuple[int | None, int | None, str]]:
    """Fold chunks below ``MIN_CHUNK_WORDS`` into an adjacent chunk.

    Merges backwards when the previous chunk has room, otherwise forwards, so
    no text is dropped. A sliver that fits nowhere (a single very short
    document) is kept as-is rather than discarded.
    """
    merged: list[list] = []
    for start, end, body in packed:
        is_sliver = len(body.split()) < MIN_CHUNK_WORDS
        if is_sliver and merged:
            previous = merged[-1]
            if len(previous[2].split()) + len(body.split()) <= chunk_size:
                previous[2] = f"{previous[2]}\n\n{body}"
                if previous[0] is None:
                    previous[0] = start
                previous[1] = end if end is not None else previous[1]
                continue
        merged.append([start, end, body])

    # A sliver at position 0 has no previous chunk; fold it into the next one.
    if len(merged) > 1 and len(merged[0][2].split()) < MIN_CHUNK_WORDS:
        head = merged.pop(0)
        merged[0][2] = f"{head[2]}\n\n{merged[0][2]}"
        if head[0] is not None:
            merged[0][0] = head[0]

    return [(start, end, body) for start, end, body in merged]


def chunk_corpus(
    documents: list[Document],
    strategy: str = "paragraph",
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk every document in the corpus with one strategy."""
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy, chunk_size, overlap))
    return chunks
