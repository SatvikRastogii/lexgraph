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

### Reranking earns its cost twice

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense-fixed` — 500-word chunks | 0.681 | 0.889 | [0.79, 0.97] | 0.906 | 0.980 | 2190ms |
| `dense` — paragraph chunks | 0.628 | 0.902 | [0.82, 0.97] | 0.897 | 0.940 | 2193ms |
| `bm25` | 0.636 | 0.857 | [0.74, 0.96] | 0.857 | 0.933 | **1ms** |
| `hybrid` — RRF fusion | 0.676 | 0.885 | [0.78, 0.97] | 0.900 | 0.960 | 2195ms |
| **`hybrid-rerank`** | **0.716** | **0.905** | [0.82, 0.97] | **0.933** | 0.980 | 4168ms |

The cross-encoder helps where a cross-encoder should — at the top of the
ranking. R@1 rises from 0.628 to 0.716 and nDCG@10 from 0.897 to 0.933.

The Recall@5 column is a different story. Every interval overlaps every other
interval. At 25 answerable questions this gold set **cannot** separate these
configurations on R@5, and the honest reading is that it does not. The
intervals are printed for that reason rather than hidden.

### Cosine similarity is a poor signal for refusing to answer

Ten gold-set questions have no supporting document anywhere in the corpus. The
right behaviour is refusal. Whether that is achievable depends entirely on
whether the retriever scores those questions differently from real ones:

| retriever | answerable | out-of-corpus | separation | Youden's J |
|---|---|---|---|---|
| dense | 0.737 | 0.698 | **+0.039** | 0.38 |
| bm25 | 27.75 | 14.48 | +13.27 | 0.72 |
| **hybrid-rerank** | 0.893 | 0.243 | **+0.649** | **0.80** |

Dense similarity barely moves between the two populations. Catching 90% of
out-of-corpus questions on that signal costs refusing 52% of answerable ones —
which is why a confidence score built on cosine distance can only ever be a
warning label beside an answer, not a decision to withhold one.

The cross-encoder score separates cleanly. At a threshold of 0.423 it answers
100% of answerable questions and refuses 80% of out-of-corpus ones. That
threshold is chosen by maximising Youden's J against the gold set, not by feel.

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
which is why BM25 at 1ms is not a rounding difference.

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

python scripts/build_index.py     # ~4 min, both chunking strategies
python scripts/validate_goldset.py
python scripts/eval_retrieval.py  # the ablation table
python scripts/eval_answers.py --generator ollama:qwen2.5:3b

streamlit run app.py
```

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

- 25 answerable questions is a small gold set. It is large enough to detect the
  reranking effect on R@1 and nDCG, and demonstrably too small to separate
  configurations on R@5.
- The gold set is close to saturated: MRR sits between 0.93 and 0.98 across
  every configuration, meaning a relevant document is almost always at rank 1
  already. A harder question tier would give the harness more room.
- The judge has not been validated against human labels. Independence from the
  generator removes the worst bias, not all of it.
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
