"""Search with a hypothetical answer instead of the question.

The hard tier exists because a question phrased in lay language shares almost
no vocabulary with the judgment that answers it. *"Can the police keep a
permanent watch file on a man who finished his sentence?"* and *"the
maintenance of a history sheet under the Police Standing Orders"* are the same
question in two registers, and BM25 loses ten points on that gap while dense
retrieval loses three.

HyDE closes the register rather than the meaning: the generator writes a short
passage in the *style of a judgment* answering the question, and that passage
is what gets embedded and searched. The hypothetical is usually wrong on the
facts -- it will invent a case name and a year -- and that does not matter,
because it is never shown to anyone and never enters the context. Only its
vocabulary and register are used.

Two decisions worth stating.

The original question is prepended to the hypothetical rather than replaced by
it. A hypothetical that drifts off topic would otherwise take the search with
it, and keeping the question anchors the embedding to what was actually asked.

The reranker, where one is used, still scores against the *original* question.
Reranking documents for their similarity to a passage the model invented would
optimise for the wrong target -- the hypothetical is a bridge to the corpus
vocabulary, not a statement of what the user wants.

The cost is one generation per query, which is why this lives behind its own
configuration rather than in the default path: it roughly doubles query
latency, and `lexgraph.cache` makes it free only on a repeat.
"""

from __future__ import annotations

from .base import Hit

HYDE_PROMPT = """You are drafting a short passage in the style of an Indian
court judgment, to be used only for document retrieval.

Write 2-3 sentences that would plausibly appear in a judgment answering the
question below. Use the formal legal register: name the constitutional
provisions and statutory sections that would be involved, and the standard
terminology a judge would use.

Do not hedge, do not explain what you are doing, and do not say the answer is
uncertain. Accuracy does not matter -- only that the vocabulary matches how a
judgment would phrase it.

Question: {question}

Passage:"""

MAX_HYPOTHESIS_TOKENS = 160


class HyDERetriever:
    """Wraps any retriever, searching with question + hypothetical answer."""

    def __init__(self, base, generator, max_tokens: int = MAX_HYPOTHESIS_TOKENS):
        self.base = base
        self.generator = generator
        self.max_tokens = max_tokens
        self.hypotheses: dict[str, str] = {}

    def expand(self, query: str) -> str:
        """``question + hypothetical passage``, or the question if generation fails.

        A generator that is down or rate limited must degrade to ordinary
        retrieval rather than fail the query. Silently returning nothing would
        be worse than either.
        """
        if query in self.hypotheses:
            return self.hypotheses[query]

        try:
            hypothesis = self.generator.chat(
                HYDE_PROMPT.format(question=query), max_tokens=self.max_tokens
            ).strip()
        except Exception:  # noqa: BLE001 - retrieval must survive a dead generator
            hypothesis = ""

        expanded = f"{query}\n\n{hypothesis}" if hypothesis else query
        self.hypotheses[query] = expanded
        return expanded

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return self.base.search(self.expand(query), top_k)


class HyDEReranked:
    """HyDE for the shortlist, the original question for the reranking.

    Kept separate from wrapping a ``RerankedRetriever`` in ``HyDERetriever``,
    which would hand the cross-encoder the hypothetical and have it rank
    judgments by their resemblance to something the model made up.
    """

    def __init__(self, hyde: HyDERetriever, reranker, candidates: int = 20):
        self.hyde = hyde
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        shortlist = self.hyde.search(query, max(self.candidates, top_k))
        return self.reranker.rerank(query, shortlist, top_k=top_k)
