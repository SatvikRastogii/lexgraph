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


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    abstained: bool = False
    abstention: AbstentionDecision | None = None
    citations: CitationReport | None = None
    latency: dict[str, float] = field(default_factory=dict)

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
) -> GeneratedAnswer:
    """Retrieve, decide whether to answer, generate, then verify citations.

    When ``abstention_threshold`` is None the guardrail is off, which is how
    the ablation measures what abstention actually buys.
    """
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

    started = time.perf_counter()
    answer = generator.chat(
        SYSTEM_PROMPT.format(question=question, context=format_context(hits)),
        max_tokens=max_tokens,
        temperature=0.0,
    )
    latency["generation_ms"] = (time.perf_counter() - started) * 1000
    latency["total_ms"] = latency["retrieval_ms"] + latency["generation_ms"]

    return GeneratedAnswer(
        question=question,
        answer=answer,
        hits=hits,
        abstained=False,
        abstention=decision,
        citations=verify_citations(answer, [hit.text for hit in hits]),
        latency=latency,
    )
