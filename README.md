<h1 align="center">LexGraph</h1>

<p align="center">
  <em>An evaluation harness for retrieval systems, built on a corpus of Indian case law.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/tests-110%20passing-3B6650" />
  <img src="https://img.shields.io/badge/retrieval-BM25%20%2B%20dense%20%2B%20rerank-8F6620" />
  <img src="https://img.shields.io/badge/inference-local%20(Ollama)-3F5670" />
</p>

---

## What this is

Most RAG projects can show you an answer. Far fewer can tell you whether the
answer was any good, or whether a change made it better.

LexGraph is the second thing. It indexes 40 Indian court judgments and puts
five retrieval configurations, two generator models and a graph index through
the same measurement: a hand-annotated gold set with document-level ground
truth, rank metrics that need no LLM, and answer scoring by a judge model from
a different family than the generator.

Everything below is measured on an RTX 4050 with 6GB of VRAM. Retrieval and
generation run locally through Ollama. Only the judge is remote, deliberately.

---

## Findings

### BM25 collapses on paraphrase; dense does not

The gold set has two tiers. Standard questions use the vocabulary of the
judgments. Hard questions deliberately do not — *"if someone is being held in
jail, can they complain about how they are treated inside"* instead of *"can a
prisoner invoke writ jurisdiction"*. Splitting the metrics by tier is what
makes the retrievers' characters visible:

| configuration | hard R@5 | standard R@5 | drop | hard nDCG@10 |
|---|---|---|---|---|
| `bm25` | 0.717 | 0.857 | **−0.140** | 0.671 |
| `dense-fixed` | 0.833 | 0.889 | −0.056 | 0.742 |
| `dense` | 0.867 | 0.900 | −0.033 | 0.783 |
| `hybrid` | 0.867 | 0.885 | −0.018 | **0.795** |
| `hybrid-rerank` | 0.867 | **0.902** | −0.035 | 0.791 |

Lexical matching loses 14 points the moment the question stops sharing words
with the document. Dense retrieval loses 3. That gap is the whole argument for
fusing them, and it is the reason the aggregate numbers below are worth less
than this table.

### The aggregate picture

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense-fixed` | 0.610 | 0.873 | [0.78, 0.95] | 0.859 | 0.914 | 2152ms |
| `dense` | 0.586 | 0.890 | [0.80, 0.96] | 0.861 | 0.895 | 2184ms |
| `bm25` | 0.578 | 0.817 | [0.71, 0.92] | 0.803 | 0.858 | **2ms** |
| `hybrid` | 0.621 | 0.880 | [0.79, 0.96] | 0.867 | 0.914 | 2188ms |
| **`hybrid-rerank`** | **0.664** | **0.892** | [0.81, 0.96] | **0.890** | **0.924** | 4353ms |

The cross-encoder helps where a cross-encoder should — at the top of the
ranking. R@1 rises from 0.586 to 0.664 and nDCG@10 from 0.861 to 0.890.

Every Recall@5 interval still overlaps every other one. At 35 answerable
questions this gold set **cannot** establish those differences, and the honest
reading is that it does not. The intervals are printed for that reason rather
than hidden.

### Cosine similarity is a poor signal for refusing to answer

Ten gold-set questions have no supporting document anywhere in the corpus. The
right behaviour is refusal. Whether that is achievable depends entirely on
whether the retriever scores those questions differently from real ones:

| retriever | answerable | out-of-corpus | separation | Youden's J |
|---|---|---|---|---|
| dense | 0.718 | 0.698 | **+0.020** | 0.24 |
| bm25 | 24.80 | 14.48 | +10.32 | 0.57 |
| **hybrid-rerank** | 0.736 | 0.243 | **+0.492** | **0.63** |

Dense similarity barely moves between the two populations. Catching 90% of
out-of-corpus questions on that signal costs refusing 66% of answerable ones —
which is why a confidence score built on cosine distance can only ever be a
warning label beside an answer, not a decision to withhold one.

The cross-encoder score separates cleanly. At a threshold of 0.375 it answers
83% of answerable questions and refuses 80% of out-of-corpus ones. That
threshold is chosen by maximising Youden's J against the gold set, not by feel,
and the dashboard reads it from the calibration file rather than hardcoding it
— adding the hard tier moved it from 0.423 to 0.375, and a constant would have
gone stale silently.

Measured end to end during the judged run: **8 of 10 out-of-corpus questions
correctly refused, at the cost of wrongly refusing 6 of 35 answerable ones.**

That second number is the honest half of the result. On the easier question
set the same guardrail refused nothing it should have answered; adding the
paraphrase tier pushed genuinely answerable questions down toward the refusal
threshold, because a question sharing little vocabulary with its source scores
lower under the cross-encoder whether or not the corpus can answer it. A
guardrail tuned on easy questions looks free. Tuned on hard ones, it has a
price, and the price is visible here rather than averaged away.

So reranking pays for its latency twice: once in ranking quality, and again by
producing a score calibrated enough for a guardrail to work at all.

### The model did not fit the card

```
llama3.1:latest   7.2 GB required   43%/57% CPU/GPU   12.0 tok/s   ~100s cold start
qwen2.5:3b        2.7 GB required   100% GPU          58.8 tok/s
```

`llama3.1` needs 7.2GB resident on a 6GB card, so Ollama spills 43% of it to
system RAM across PCIe. Worse, it and `nomic-embed-text` evict each other, and
every eviction pays the cold start again. This is the actual cause of the
system feeling slow — not the corpus, not the pipeline.

A related measurement: an Ollama embed request costs ~2.15s of **fixed**
overhead on this hardware regardless of payload. Batches of 1, 2 and 8 take
the same wall time; a batch of 32 amortises to ~94ms per item. The ChromaDB
query itself is 3ms. Single-query dense retrieval is dominated by a constant,
which is why BM25 at 2ms is not a rounding difference.

### What the smaller model actually costs you

Both generators, same retriever, same questions, scored by the same
independent judge:

Both generators, same retriever, same 45 questions, same judge, n=35
answerable, zero unparsed judge responses on either run:

| metric | llama3.1 (8B) | qwen2.5:3b | Δ |
|---|---|---|---|
| answer relevancy | **4.54** | 3.86 | +0.69 |
| completeness | **3.83** | 3.11 | +0.71 |
| legal reasoning | **3.57** | 2.94 | +0.63 |
| hallucination ↑ | 3.26 | **3.86** | −0.60 |
| faithfulness | 3.23 | 3.03 | +0.20 |
| coherence | 4.91 | 4.80 | +0.11 |
| citation accuracy | 4.90 | 4.92 | −0.02 |
| *context precision* | *4.20* | *4.57* | *−0.37* |
| median answer length | 245 words | 221 words | |
| **median latency** | **52.5s** | **12.0s** | |

**Read that table with a noise floor.** `context precision` scores only the
question and the retrieved passages — and those are byte-identical between the
two runs, because it is the same retriever answering the same questions. Its
true difference is therefore exactly zero. The judge returned **−0.37**.

That number is a free, built-in estimate of how much this judge disagrees with
itself, and it costs nothing to collect because the metric was already there.
Applying it: relevancy (+0.69), completeness (+0.71), legal reasoning (+0.63)
and hallucination (−0.60) all clear the floor and are worth believing.
Faithfulness (+0.20) and coherence (+0.11) do not, and should not be read as
differences at all. Citation accuracy is deterministic, and its −0.02 is the
sanity check that the harness is wired correctly.

Part of that 0.37 is scoring all metrics in one call: the judge sees the answer
while rating the context, so a metric that should be answer-independent is not
quite. That was the price of fitting a sweep inside a free-tier daily quota,
and it is a real cost rather than a free optimisation.

So: llama3.1 is the better writer and the clearly better legal reasoner, at
4.4× the latency for an answer of similar length — and it **hallucinates more**.
That is the interesting one. A smaller model has less parametric knowledge to
substitute for what retrieval failed to supply, so it stays nearer the text; a
larger one is more willing to fill the gap from memory. In a grounded legal
tool that willingness is a liability, not a feature.

Neither column is the "right" answer. The point is that the tradeoff is a
measurement.

---

## How the evaluation is kept honest

**The judge is never the generator.** Scoring llama3.1's answers with
llama3.1 is self-evaluation. `require_independent_judge()` raises rather than
letting a self-judged run quietly report better numbers.

**Ground truth is machine-verified.** `scripts/validate_goldset.py` checks that
every supporting document exists and contains the terms its annotation claims,
that no out-of-corpus question is secretly answerable, and that dissent labels
match the document headers. It runs in CI. It caught four bad labels on its
first run, including three questions marked out-of-corpus that the corpus
actually cites.

**Citation accuracy is measured, not judged.** Case names, article and section
references and reported citations are extracted from the answer and checked
against the retrieved text. Asking a model to grade this lets a fabricated
citation be scored generously; a regex cannot be talked round.

It earns its place. Asked what the courts held about Parliament's power to
amend fundamental rights, the generator cited **Article 368 and Article 13(2)**
— the legally correct provisions, and neither of them anywhere in the retrieved
judgments. A judge scoring citation quality would likely have rewarded them.
The verifier flags them, because being right from memory is still not being
grounded in the corpus, and in a legal tool that distinction is the product.

**Answer length travels with every score.** LLM judges reward verbosity, so a
score gap that is really a length gap stays visible in the output.

**Every mean carries a bootstrap interval.** At a few dozen questions, a bare
mean implies a precision the sample size does not support.

**Out-of-corpus probes are near-misses.** Shreya Singhal, Sabarimala, Navtej
Singh Johar, triple talaq — all absent from the corpus, all certainly known to
the generator from pre-training. A fluent answer to one of them is a retrieval
failure wearing the costume of a success. Cases the corpus merely *cites*
(Maneka Gandhi appears in 10 documents, Kesavananda in 4) were rejected as
probes for exactly that reason.

---

## Being straight about the corpus

This matters more than any number above, so it goes in the README rather than a
footnote.

- **40 files, 38 unique cases.** Two are duplicate pairs. They are kept because
  the GraphRAG index was built over all 40, and removing them from `input/`
  without re-indexing would make the two pipelines incomparable again.
- **17 Supreme Court, 22 High Court, 1 unclear.** An earlier version of this
  README described all 40 as Supreme Court judgments. That was wrong.
- **These are not landmark constitutional cases.** They are keyword-scrape
  results that mention Articles 14, 19, 21 or 32 — service law, criminal
  procedure, prohibition, reservation. *Maneka Gandhi*, *Kesavananda Bharati*,
  *Puttaswamy* and *Vishaka* are **not in the corpus**, though several
  judgments cite them.
- **The published benchmark that preceded this one was invalid.** GraphRAG
  indexed `input/` (40 documents) while the vector pipeline indexed a
  keyword-filtered slice of `legal_corpus/` (9 documents). The overlap was
  **zero**. The "GraphRAG wins 81%" result compared two disjoint corpora, with
  llama3.1 scoring its own answers, on questions about cases neither pipeline
  had ever seen. Both the result and the code that produced it have been
  removed rather than corrected.

---

## Layout

```
lexgraph/
  corpus.py          parse the scraper's header block; load from input/
  chunking.py        fixed-window and paragraph-aware strategies
  embeddings.py      batched Ollama embedding with retry and a query cache
  llm.py             provider dispatch: Ollama, Gemini, OpenAI-compatible
  router.py          NAIVE vs GRAPH by prototype similarity, no LLM call
  generation.py      prompt, generate, verify citations
  retrieval/         bm25 · dense · fusion (RRF) · rerank · pipeline
  eval/              retrieval_metrics · judge
  guardrails/        abstention · citations
scripts/
  build_index.py         index input/ into ChromaDB, one collection per strategy
  validate_goldset.py    verify every annotation against the corpus text
  eval_retrieval.py      the ablation table; no LLM, runs in about a minute
  eval_answers.py        judged answer quality with an independent judge
  calibrate_abstention.py  choose the refusal threshold from data
  drift_check.py         scheduled regression check
  list_judge_models.py   which judge models a given key can actually reach
data/goldset.json    35 questions, 25 answerable, 10 out-of-corpus
tests/               110 tests, no network or GPU required
```

---

## Running it

```bash
pip install -r requirements.txt
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5:3b            # fits 6GB fully; llama3.1 does not

cp .env.example .env              # add a judge key: aistudio.google.com/apikey

python scripts/build_index.py       # ~4 min, both chunking strategies
python scripts/validate_goldset.py  # verify the annotations against the corpus
python scripts/eval_retrieval.py    # the ablation table; ~1 min, no LLM
python scripts/calibrate_abstention.py
python scripts/eval_answers.py --generator ollama:qwen2.5:3b
python scripts/make_report.py       # renders reports/EVALUATION.md

streamlit run app.py
```

`eval_answers.py` is resumable. It writes after every question and, on
relaunch, scores only the questions missing from the results file — which
matters because the judge's free tier is metered per day and llama3.1 needs
about a minute per answer on this hardware. `--restart` forces a clean run.

If the judge returns 404 or runs out of quota, `python
scripts/list_judge_models.py --probe` reports which models the key can
currently reach; the judge rotates through several because the quota is
per-model.

The GraphRAG index (`output/*.parquet`) is a separate, hours-long offline step
and is not committed:

```bash
graphrag index --root .
```

Full stack with Prometheus and Grafana:

```bash
docker compose up --build
```

---

## Limitations

- 35 answerable questions is a small gold set. It is large enough to detect the
  reranking effect on R@1 and nDCG and BM25's collapse on paraphrase, and
  demonstrably too small to separate configurations on R@5.
- The judge has not been validated against human labels. Independence from the
  generator removes the worst bias, not all of it. The `context precision`
  control above bounds its self-disagreement at roughly 0.37 points, but that
  measures consistency, not correctness — a judge can be perfectly consistent
  and perfectly wrong. Hand-scoring 20 answers and reporting Spearman ρ is the
  obvious next step.
- Scoring every metric in one judge call lets the answer leak into metrics that
  should not see it, which is part of that 0.37. Separate calls would remove
  it at seven times the quota, which the free tier does not support.
- The hard tier is 10 questions. It separates the configurations far better
  than the standard tier, which argues for making it much larger.
- GraphRAG is compared on the answer side only. Its retrieval is not yet scored
  on the same footing, though `text_unit_ids` in its output would allow it.
- **GraphRAG's `local` search method crashes** on this index — a vector-dimension
  mismatch between the embeddings LanceDB stored at index time and the live
  `nomic-embed-text` configuration. Only `global` search works, and fixing it
  requires a full re-index. `global` is also the expensive method (it fans out
  across every community report), which is most of why GraphRAG queries take
  minutes rather than seconds.
- Document-level ground truth, not paragraph-level. Paragraph spans are tracked
  in chunk metadata but not annotated in the gold set.
