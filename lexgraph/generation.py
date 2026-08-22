"""Answer generation over retrieved context, with abstention wired in.

One generator serves every retrieval configuration so that a difference in
answer quality is attributable to retrieval rather than to prompt drift
between pipelines. The previous code base had three separate copies of this
prompt -- in naive_rag.py, inlined in app.py, and again in the evaluation
script -- which quietly made the benchmark compare prompts as well as
pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .guardrails.abstention import AbstentionDecision, assess
from .guardrails.citations import CitationReport, verify_citations
from .llm import BaseClient
from .retrieval.base import Hit

SYSTEM_PROMPT = """You are a legal research assistant working with Indian case law.

Answer the question using ONLY the judgment extracts provided below. Rules:
- Cite the case name for every proposition you state.
- If the extracts do not settle the question, say so explicitly rather than
  filling the gap from general knowledge.
- Do not cite any case, article or section that does not appear in the extracts.
- Be precise about what was actually held, as opposed to what was argued.

QUESTION: {question}

JUDGMENT EXTRACTS:
{context}

ANSWER:"""

# Naming the offending citations is the point. "Please only cite the extracts"
# is the instruction the model already had and already broke; quoting back the
# exact strings that failed verification gives it something to act on.
CITATION_RETRY_PROMPT = """Your previous answer cited material that does not
appear in the judgment extracts you were given:

{unsupported}

Those citations could not be found and may be fabricated. Rewrite the answer
using only what the extracts below actually contain. If a proposition cannot be
supported by them, drop it or say the extracts do not settle it. Do not
substitute a different case you are unsure of.

QUESTION: {question}

JUDGMENT EXTRACTS:
{context}

CORRECTED ANSWER:"""

# What to do when an answer cites something the context does not contain.
#   warn    attach the report and return the answer (what the numbers so far measure)
#   retry   regenerate once, naming the unsupported citations, then warn
#   refuse  withhold the answer entirely below min_citation_accuracy
CITATION_POLICIES = ("warn", "retry", "refuse")


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    abstained: bool = False
    abstention: AbstentionDecision | None = None
    citations: CitationReport | None = None
    latency: dict[str, float] = field(default_factory=dict)
    citation_retry: bool = False

    @property
    def contexts(self) -> list[str]:
        return [hit.text for hit in self.hits]

    @property
    def sources(self) -> list[str]:
        seen, ordered = set(), []
        for hit in self.hits:
            if hit.doc_id not in seen:
                seen.add(hit.doc_id)
                ordered.append(hit.doc_id)
        return ordered

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstention_reason": self.abstention.reason if self.abstention else "",
            "confidence": self.abstention.confidence if self.abstention else 0.0,
            "sources": self.sources,
            "citations": {
                "supported": self.citations.supported if self.citations else [],
                "unsupported": self.citations.unsupported if self.citations else [],
                "accuracy": self.citations.accuracy if self.citations else 1.0,
            },
            "citation_retry": self.citation_retry,
            "latency": self.latency,
        }


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered, attributed extracts."""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] {hit.citation}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    retriever,
    generator: BaseClient,
    top_k: int = 5,
    abstention_threshold: float | None = None,
    max_tokens: int = 800,
    citation_policy: str = "warn",
    min_citation_accuracy: float = 1.0,
) -> GeneratedAnswer:
    """Retrieve, decide whether to answer, generate, then act on the citations.

    When ``abstention_threshold`` is None the guardrail is off, which is how
    the ablation measures what abstention actually buys. ``citation_policy``
    defaults to ``warn`` for the same reason: it is the behaviour every number
    reported so far was measured under, and changing the default would move
    those numbers without a run to attribute the change to.
    """
    if citation_policy not in CITATION_POLICIES:
        raise ValueError(
            f"unknown citation policy {citation_policy!r}; "
            f"expected one of {list(CITATION_POLICIES)}"
        )
    latency = {}

    started = time.perf_counter()
    hits = retriever.search(question, top_k)
    latency["retrieval_ms"] = (time.perf_counter() - started) * 1000

    decision = None
    if abstention_threshold is not None:
        decision = assess(hits, abstention_threshold)
        if not decision.should_answer:
            latency["total_ms"] = latency["retrieval_ms"]
            return GeneratedAnswer(
                question=question,
                answer=decision.refusal_message,
                hits=hits,
                abstained=True,
                abstention=decision,
                citations=CitationReport(),
                latency=latency,
            )

    context = format_context(hits)
    texts = [hit.text for hit in hits]

    started = time.perf_counter()
    answer = generator.chat(
        SYSTEM_PROMPT.format(question=question, context=context),
        max_tokens=max_tokens,
        temperature=0.0,
    )
    citations = verify_citations(answer, texts)
    retried = False

    # The verifier has been computing this all along and nothing acted on it:
    # an answer citing a case the context never contained was measured, scored,
    # and returned anyway. That is a report, not a guardrail.
    if citation_policy == "retry" and citations.has_unsupported:
        retried = True
        corrected = generator.chat(
            CITATION_RETRY_PROMPT.format(
                unsupported="\n".join(f"- {c}" for c in citations.unsupported),
                question=question,
                context=context,
            ),
            max_tokens=max_tokens,
            temperature=0.0,
        )
        corrected_citations = verify_citations(corrected, texts)
        # Keep the rewrite only if it is actually cleaner. A retry is free to
        # make things worse, and accepting it unconditionally would let the
        # guardrail lower citation accuracy while appearing to enforce it.
        if corrected_citations.accuracy > citations.accuracy:
            answer, citations = corrected, corrected_citations

    latency["generation_ms"] = (time.perf_counter() - started) * 1000
    latency["total_ms"] = latency["retrieval_ms"] + latency["generation_ms"]

    if citation_policy == "refuse" and citations.accuracy < min_citation_accuracy:
        return GeneratedAnswer(
            question=question,
            answer=(
                "This answer was withheld: it cited material that does not appear "
                "in the retrieved judgments, which in legal research is the failure "
                "that matters most."
            ),
            hits=hits,
            abstained=True,
            abstention=decision,
            citations=citations,
            latency=latency,
            citation_retry=retried,
        )

    return GeneratedAnswer(
        question=question,
        answer=answer,
        hits=hits,
        abstained=False,
        abstention=decision,
        citations=citations,
        latency=latency,
        citation_retry=retried,
    )
