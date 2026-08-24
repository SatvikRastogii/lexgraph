"""Score the pipeline with RAGAS, as a second opinion on the built-in judge.

`lexgraph.eval.judge` implements its own scoring because it needed things RAGAS
does not give: an enforced-independent judge, bootstrap intervals, salvage
parsing for truncated responses, and citation accuracy measured by regex rather
than asked of a model. None of that makes it *right*, and a metric nobody else
can reproduce is a metric nobody else should trust.

So RAGAS runs over the same answers. Where the two agree, the number is a
property of the answer rather than of my prompt. Where they disagree, that gap
is the honest error bar on the judged table, and it is worth more than either
score alone.

Eight metrics:

    faithfulness ........................ every claim traceable to the context
    answer_relevancy .................... does the answer address the question
    context_precision ................... are the retrieved passages relevant
    context_utilization ................. did the answer actually use them
    context_relevance ................... is the context relevant to the question
    response_groundedness ............... is the response supported by context
    attributes_to_a_named_case .......... legal-domain check, via AspectCritic
    admits_when_context_is_insufficient . legal-domain check, via AspectCritic

The last two exist because RAGAS's generic metrics cannot see the failures that
matter in legal answers: an unattributed proposition, and a confident answer
built on context that does not support one.

RAGAS's two non-LLM context metrics were tried and dropped. They compare
reference against retrieved by Levenshtein similarity, which between a gold
paragraph and a correctly retrieved chunk is near zero -- they scored 0.05 and
0.25 on answers the LLM metrics rated 1.00 faithful. That is a units artifact,
not a measurement, and Recall@k already covers the question properly.

Runs in its own virtualenv. RAGAS pins langchain-core below 0.3, which
conflicts with everything else installed here:

    python -m venv .ragasenv
    .ragasenv/Scripts/pip install "ragas==0.2.15" "langchain-core<0.3" \\
        "langchain-community<0.3" "langchain-openai<0.2" "openai<2" \\
        fastembed flashrank pandas pyarrow
    .ragasenv/Scripts/python scripts/eval_ragas.py --limit 5
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLDSET_PATH = os.path.join("data", "goldset.json")
RESULTS_PATH = os.path.join("reports", "ragas_evaluation.json")

# Groq, reached through its OpenAI-compatible endpoint. Not Gemini: that is the
# built-in judge, and a second opinion from the same provider is not a second
# opinion.
GROQ_BASE = "https://api.groq.com/openai/v1"

# The smaller model by default. RAGAS sends the retrieved context once per
# metric per question, so eight metrics over five 500-word passages is roughly
# 20k tokens a question -- and Groq's free tier meters 200,000 tokens per day
# per model. Scoring all 67 answerable questions would need about 1.4M, so the
# full set is a paid-tier run; what fits free is a stratified sample.
DEFAULT_SCORER = "openai/gpt-oss-20b"

# Roughly what one question costs to score, used only to warn before spending.
TOKENS_PER_QUESTION = 20_000
FREE_TIER_DAILY_TOKENS = 200_000


class FastEmbedWrapper:
    """LangChain embedding interface over fastembed.

    RAGAS wants an embeddings object for response_relevancy and semantic
    similarity. This reuses the model the deployment already runs -- ONNX on
    CPU, no key, no quota -- rather than adding a third provider to the
    evaluation just to measure it.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(list(texts))]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def reference_contexts(question, documents, limit=5):
    """Gold passages, at the same granularity as what retrieval returns.

    The non-LLM context metrics compare reference against retrieved by string
    similarity. Handing them whole judgments -- ten to twenty thousand words --
    against 500-word chunks scores 0.000 for every question, which looks like
    total retrieval failure and is really a units mismatch.

    So the reference is the paragraph-level ground truth derived by
    scripts/derive_paragraph_labels.py: the specific paragraphs that carry the
    answer. Capped at ``limit`` to match top-k, otherwise a question with
    thirty gold paragraphs is unrecallable by a retriever returning five.

    Questions without paragraph labels fall back to the opening of each gold
    document, which is weaker and is why those questions are reported
    separately rather than averaged in silently.
    """
    from lexgraph.chunking import split_numbered_paragraphs

    labels = question.get("relevant_paragraphs") or {}
    passages = []
    for name, numbers in labels.items():
        body = documents.get(name)
        if not body:
            continue
        wanted = set(numbers)
        for number, text in split_numbered_paragraphs(body):
            if number in wanted and text.strip():
                passages.append(text.strip())

    if not passages:
        for name in question["relevant_docs"]:
            body = documents.get(name)
            if body:
                passages.append(" ".join(body.split()[:400]))

    return passages[:limit] or [""]


def stratify(questions, per_tier):
    """An equal number of questions from each difficulty tier.

    Taking the first N would take them in id order, which is tier order -- a
    sample of 20 would be entirely 'hard' and the RAGAS numbers would describe
    the paraphrase tier while claiming to describe the pipeline.
    """
    if not per_tier:
        return questions
    buckets = {}
    for question in questions:
        buckets.setdefault(question.get("difficulty", "standard"), []).append(question)
    sampled = [q for tier in sorted(buckets) for q in buckets[tier][:per_tier]]
    return sorted(sampled, key=lambda q: q["id"])


def build_samples(questions, config, generator_spec, top_k, threshold, documents):
    """Generate an answer per question and shape it the way RAGAS expects."""
    from lexgraph.generation import answer_question
    from lexgraph.llm import build_generator
    from lexgraph.retrieval.pipeline import build_retriever

    retriever = build_retriever(config)
    generator = build_generator(generator_spec, rotate=True)
    samples = []

    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        generated = answer_question(
            question["question"], retriever, generator,
            top_k=top_k, abstention_threshold=threshold,
        )
        reference = reference_contexts(question, documents)
        samples.append({
            "id": question["id"],
            "difficulty": question.get("difficulty", "standard"),
            "user_input": question["question"],
            "response": generated.answer,
            "retrieved_contexts": generated.contexts or [""],
            "reference_contexts": reference or [""],
            "abstained": generated.abstained,
            "seconds": time.perf_counter() - started,
        })
        print(f"  [{index}/{len(questions)}] {question['id']} "
              f"{'refused' if generated.abstained else 'answered'} "
              f"{samples[-1]['seconds']:.1f}s")
    return samples


def score(samples, scorer_model, timeout, workers):
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AspectCritic,
        ContextRelevance,
        ContextUtilization,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseGroundedness,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=scorer_model, base_url=GROQ_BASE, api_key=key,
        temperature=0, timeout=timeout, max_retries=6,
        # RAGAS parses structured output, and gpt-oss spends its budget on a
        # private reasoning pass before writing any -- the first run raised
        # LLMDidNotFinishException on every metric. A larger budget plus less
        # reasoning is the same fix applied to the generator, and to Gemini's
        # thinkingBudget before that.
        max_tokens=3000,
        model_kwargs=(
            {"reasoning_effort": "low"} if "gpt-oss" in scorer_model else {}
        ),
    ))
    embeddings = LangchainEmbeddingsWrapper(FastEmbedWrapper())

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        ContextUtilization(llm=llm),
        ContextRelevance(llm=llm),
        ResponseGroundedness(llm=llm),
        # Domain checks. RAGAS's generic metrics cannot see what actually goes
        # wrong in legal answers, and AspectCritic is the supported way to ask
        # a yes/no question of every answer.
        AspectCritic(
            name="attributes_to_a_named_case",
            definition=(
                "Does the answer attribute its propositions to a case named in "
                "the retrieved context, rather than stating them unattributed?"
            ),
            llm=llm,
        ),
        AspectCritic(
            name="admits_when_context_is_insufficient",
            definition=(
                "If the retrieved context does not settle the question, does "
                "the answer say so, instead of filling the gap with general "
                "knowledge? Answer yes if the context is sufficient and the "
                "answer stays within it."
            ),
            llm=llm,
        ),
    ]

    dataset = EvaluationDataset.from_list([
        {k: s[k] for k in
         ("user_input", "response", "retrieved_contexts", "reference_contexts")}
        for s in samples
    ])

    # RAGAS fans out one job per metric per sample and runs them concurrently.
    # Against a free tier that is a burst of 429s, and every LLM metric came
    # back empty on the first run while the two non-LLM ones -- the only jobs
    # not making requests -- completed fine. Throttling here is what makes the
    # numbers exist at all.
    config = RunConfig(max_workers=workers, timeout=timeout, max_retries=6)
    return evaluate(dataset=dataset, metrics=metrics, llm=llm,
                    embeddings=embeddings, run_config=config)


# A metric scored on a handful of the sample is not a measurement of the
# sample. RAGAS assigns NaN to a job that ran out of quota and carries on, so
# without this a run where 53 of 72 jobs died still prints a confident 0.981
# and writes it to a results file that reads exactly like a complete one.
MIN_COVERAGE = 0.6


def summarise(result, samples):
    """Per-metric mean, plus a split by question tier."""
    frame = result.to_pandas()
    columns = [c for c in frame.columns
               if c not in ("user_input", "response", "retrieved_contexts",
                            "reference_contexts")]

    overall, by_tier, coverage = {}, {}, {}
    for column in columns:
        values = [v for v in frame[column].tolist() if v == v]  # drop NaN
        coverage[column] = len(values) / len(samples) if samples else 0.0
        if values and coverage[column] >= MIN_COVERAGE:
            overall[column] = statistics.mean(values)

    tiers = {s["difficulty"] for s in samples}
    for tier in sorted(tiers):
        rows = [i for i, s in enumerate(samples) if s["difficulty"] == tier]
        scores = {}
        for column in columns:
            values = [frame[column].iloc[i] for i in rows]
            values = [v for v in values if v == v]
            if values:
                scores[column] = statistics.mean(values)
        by_tier[tier] = {"n": len(rows), **scores}

    return overall, by_tier, frame, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--config", default="hybrid-rerank")
    parser.add_argument("--generator", default="groq:openai/gpt-oss-120b")
    parser.add_argument("--scorer", default=DEFAULT_SCORER,
                        help="the model RAGAS scores with")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-tier", type=int, default=None,
                        help="sample N questions from each difficulty tier")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1,
                        help=("concurrent RAGAS jobs. gpt-oss-20b allows 8,000 "
                              "tokens per minute and one job costs ~2.5k, so "
                              "anything above 1 spends the run on 429s"))
    parser.add_argument("--answerable-only", action="store_true", default=True)
    parser.add_argument("--results-file", default=RESULTS_PATH)
    args = parser.parse_args()

    from lexgraph.corpus import load_corpus

    with open(args.goldset, encoding="utf-8") as handle:
        questions = [q for q in json.load(handle)["questions"] if q["answerable"]]
    questions = stratify(questions, args.per_tier)
    if args.limit:
        questions = questions[: args.limit]

    documents = {d.filename: d.body for d in load_corpus("input")}

    print(f"config    {args.config}")
    print(f"generator {args.generator}")
    print(f"scorer    groq:{args.scorer}")
    print(f"questions {len(questions)}\n")

    # Abstention off. RAGAS scores the answer, and a refusal is not an answer;
    # leaving the guardrail on would score the refusal message for
    # faithfulness and quietly drag every mean down.
    samples = build_samples(questions, args.config, args.generator,
                            args.top_k, None, documents)

    print("\nscoring with RAGAS...")
    result = score(samples, args.scorer, args.timeout, args.workers)
    overall, by_tier, frame, coverage = summarise(result, samples)

    print(f"\n{'metric':<38}{'score':>8}{'scored':>10}")
    print("-" * 56)
    for name in sorted(coverage):
        scored = f"{int(round(coverage[name] * len(samples)))}/{len(samples)}"
        if name in overall:
            print(f"{name:<38}{overall[name]:>8.3f}{scored:>10}")
        else:
            print(f"{name:<38}{'--':>8}{scored:>10}   below {MIN_COVERAGE:.0%}")

    withheld = [n for n in coverage if n not in overall]
    if withheld:
        print(f"\n{len(withheld)} metric(s) scored too few questions to report.")
        print("RAGAS assigns NaN when a job exhausts its quota and carries on, so")
        print("a thin run still produces confident-looking means. Those are")
        print("withheld rather than published.")

    if overall:
        print(f"\n{'metric':<38}" + "".join(f"{t:>12}" for t in by_tier))
        print("-" * (38 + 12 * len(by_tier)))
        for name in sorted(overall):
            row = f"{name:<38}"
            for tier in by_tier:
                value = by_tier[tier].get(name)
                row += f"{value:>12.3f}" if value is not None else f"{'-':>12}"
            print(row)

    payload = {
        "config": args.config,
        "generator": args.generator,
        "scorer": f"groq:{args.scorer}",
        "ragas_version": __import__("ragas").__version__,
        "n": len(samples),
        "complete": not [n for n in coverage if n not in overall],
        "coverage": coverage,
        "min_coverage": MIN_COVERAGE,
        "metrics": overall,
        "by_difficulty": by_tier,
        "per_question": [
            {"id": s["id"], "difficulty": s["difficulty"],
             **{c: (None if frame[c].iloc[i] != frame[c].iloc[i] else float(frame[c].iloc[i]))
                for c in frame.columns
                if c not in ("user_input", "response", "retrieved_contexts",
                             "reference_contexts")}}
            for i, s in enumerate(samples)
        ],
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {args.results_file}")


if __name__ == "__main__":
    main()
