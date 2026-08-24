<h1 align="center">LexGraph</h1>

<p align="center">
  <em>An evaluation harness for retrieval systems, built on a corpus of Indian case law.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/tests-168%20passing-3B6650" />
  <img src="https://img.shields.io/badge/retrieval-BM25%20%2B%20dense%20%2B%20rerank-8F6620" />
  <img src="https://img.shields.io/badge/inference-local%20(Ollama)-3F5670" />
</p>

---

## What this is

Most RAG projects can show you an answer. Far fewer can tell you whether the
answer was any good, or whether a change made it better.

LexGraph is the second thing. It indexes 40 Indian court judgments and puts
six retrieval configurations, two generator models and a knowledge graph
through the same measurement: a machine-verified gold set with document- and
paragraph-level ground truth, rank metrics that need no LLM, paired
significance testing, and answer scoring by both an independent judge model and
RAGAS.

The numbers below are measured locally on an RTX 4050 with 6GB of VRAM, where
retrieval and generation run through Ollama and only the judge is remote. The
deployed build swaps both for CPU inference and a hosted generator — and is
re-measured rather than assumed equivalent.

---

## Findings

### No single retriever is good at both kinds of hard question

The gold set has three tiers. **Standard** questions use the vocabulary of the
judgments. **Hard** questions deliberately do not — *"if someone is being held
in jail, can they complain about how they are treated inside"* instead of *"can
a prisoner invoke writ jurisdiction"*. **Multi-hop** questions are joins that no
single judgment answers — *"which judgments rely on both Maneka Gandhi and Sunil
Batra?"*

Splitting by tier is what makes the retrievers' characters visible:

| configuration | standard R@5 | hard R@5 | multi-hop R@5 |
|---|---|---|---|
| `dense` | **0.900** | 0.872 | 0.494 |
| `bm25` | 0.857 | 0.756 | 0.722 |
| `hybrid` | 0.885 | 0.872 | 0.681 |
| **`hybrid-rerank`** | **0.902** | 0.872 | **0.760** |
| `graph-units` | 0.890 | **0.889** | 0.467 |
| `graph-community` | 0.530 | 0.428 | 0.374 |

n = 25 / 30 / 12.

**The two hard tiers fail in opposite directions, and that is the point.**
BM25 is the *worst* retriever on paraphrase (0.756) and nearly the *best* on
multi-hop (0.722). Dense is the reverse: strong on paraphrase, and it collapses
to 0.494 on multi-hop.

The reason is visible in the questions. The paraphrase tier is written to share
no vocabulary with the judgment, which is where lexical matching has nothing to
grip. The multi-hop tier is mostly citation joins — *"which judgments rely on
both Maneka Gandhi and Sunil Batra?"* — where the answer hinges on exact proper
nouns that a 384-dimensional embedding blurs together.

Neither retriever is good at both. That is the argument for fusing them, made
with data rather than asserted, and it is why `hybrid-rerank` is the only
configuration that stays near the top of all three tiers.

### Right case, wrong paragraph

Recall counts a document as found if any of its chunks is retrieved, including
the chunk listing which counsel appeared for whom. `paragraph_recall@5` asks
the stricter question: did the retriever land on a paragraph that actually
carries the answer?

| configuration | R@5 | ParaR@5 | gap |
|---|---|---|---|
| `dense` | 0.815 | 0.677 | −0.138 |
| `hybrid` | 0.843 | 0.717 | −0.125 |
| `hybrid-rerank` | **0.863** | **0.751** | **−0.112** |
| `bm25` | 0.788 | 0.694 | −0.093 |
| `graph-units` | 0.814 | n/a | — |

Roughly eight to ten points of every configuration's recall is the right
judgment at the wrong passage, and the cross-encoder closes the least of it.
`graph-units` reads *n/a*, not zero: GraphRAG's own chunks carry no paragraph
labels, so that configuration cannot say which passage it found. Not applicable
and zero are different claims and only one of them is true.

Paragraph labels are derived by locating each question's already-verified
`must_contain` terms inside the judgment's own numbering, then re-checked in
CI — see `scripts/derive_paragraph_labels.py`. They cover 49 of 67 answerable
questions. A term-bearing paragraph is where an answer is *stated*, which is
not always the whole of where it is *reasoned*, so read this as a floor on
passage quality rather than a full account of it.

### The aggregate picture

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | ParaR@5 | p50 |
|---|---|---|---|---|---|---|---|
| `dense` | 0.507 | 0.815 | [0.73, 0.89] | 0.777 | 0.823 | 0.677 | 2212ms |
| `bm25` | 0.522 | 0.788 | [0.70, 0.87] | 0.770 | 0.822 | 0.694 | **2ms** |
| `hybrid` | 0.558 | 0.843 | [0.77, 0.91] | 0.827 | 0.878 | 0.717 | 2211ms |
| **`hybrid-rerank`** | **0.581** | **0.863** | [0.79, 0.93] | **0.848** | **0.880** | **0.751** | 4288ms |
| `graph-units` | 0.522 | 0.814 | [0.73, 0.89] | 0.806 | 0.830 | n/a | 2188ms |
| `graph-community` | 0.206 | 0.456 | [0.35, 0.56] | 0.433 | 0.432 | n/a | 2182ms |

These numbers are lower than the ones this table carried at 55 questions. The
retrievers did not get worse; twelve multi-hop questions were added, and they
are the hardest tier in the set.

### What the gold set can and cannot establish

Reading the table above as a ranking is the mistake this section exists to
prevent. Each configuration answers the *same* questions, so the right test is
to resample the per-question **difference**, not to check whether two
independently computed error bars happen not to touch. Shared question
difficulty cancels in the difference and inflates both intervals separately,
which makes the overlap test badly conservative.

Paired against `hybrid-rerank` on nDCG@10, n=67:

| configuration | Δ mean | 95% CI | W/L/T | verdict |
|---|---|---|---|---|
| `hybrid` | −0.021 | [−0.050, +0.008] | 4/14/49 | not distinguishable |
| `graph-units` | −0.042 | [−0.109, +0.019] | 14/17/36 | not distinguishable |
| **`dense`** | **−0.071** | **[−0.139, −0.007]** | **8/19/40** | **distinguishable** |
| **`bm25`** | **−0.078** | **[−0.121, −0.040]** | **4/18/45** | **distinguishable** |
| **`graph-community`** | **−0.415** | **[−0.508, −0.320]** | **5/49/13** | **distinguishable** |

Three differences are now established. `hybrid-rerank` is distinguishably better
than **either component alone** — which is the claim a hybrid pipeline exists to
make, and which the 55-question set could not support. Adding the multi-hop tier
is what made it visible: it is the tier where dense and BM25 fail differently,
so fusion has the most to gain.

`hybrid` and `graph-units` remain indistinguishable from the full pipeline, so
the cross-encoder's contribution is still a direction rather than a result.

### Where the graph is actually better

"The graph is worse" is too blunt, and the per-question data says where it is
not. Two places, both real:

| configuration | R@1 | R@5 | **R@10** | hard R@5 |
|---|---|---|---|---|
| `hybrid-rerank` | **0.581** | **0.863** | 0.876 | 0.872 |
| `graph-units` | 0.522 | 0.814 | **0.894** | **0.889** |
| `graph-rerank` | 0.497 | 0.733 | **0.899** | 0.711 |
| `hybrid-graph` | 0.558 | 0.837 | 0.851 | **0.900** |

**GraphRAG's chunking has the best Recall@10 in the entire ablation.** It
surfaces the right judgment *somewhere* more often than anything else — it just
ranks it worse. That is the profile of a candidate generator, not a ranker, and
it is worth knowing before dismissing it.

It is also marginally better on the paraphrase tier (0.889 against 0.872), and
fusing it in as a third retrieval source gives the **best hard-tier recall of
any configuration measured** (0.900).

Both advantages come from unit size rather than from the graph. GraphRAG's text
units are roughly four times longer than the paragraph chunks used elsewhere, so
each one covers more ground: better recall, worse precision. `graph-units`
contains no graph structure at all — it is GraphRAG's chunker, nothing more.

The candidate-generator idea was then tested and did not survive. Reranking
those units (`graph-rerank`) pushes Recall@10 to 0.899, the highest here, while
dropping Recall@5 to 0.733 — the cross-encoder handles long passages worse, and
this is the one graph arm the paired test calls *distinguishably worse* than the
baseline. Fusing all three sources (`hybrid-graph-rerank`) lands at 0.860
against 0.863, indistinguishable, for 6.7 seconds a query against 4.2.

So: nothing involving the graph beats `hybrid-rerank` overall, and two
configurations are measurably worse. But the corpus-level claim "the graph is
worse" hides a real strength at k=10 and on paraphrase, and this table is here
so that strength is not hidden.

### Where the graph hurts

This is the question the project is named after, and until now it had only been
asked on the answer side. GraphRAG's retrieval is now scored on the same gold
set, with the same metrics and deliberately the **same embedder** as `dense`,
so the only thing that varies is what a retrieval unit is:

- **`graph-units`** — GraphRAG's own 1,556 text units. No graph is involved;
  this is the control. It is competitive with everything else and has the best
  R@10 in the table (0.894), which is worth noting: GraphRAG's *chunking* is
  fine. It is the graph layer on top that costs.
- **`graph-community`** — the 184 community reports, LLM-written summaries of
  clustered entities and relationships. This is where the graph actually lives,
  and it is what `global` search fans out across.

The graph roughly halves retrieval quality, and the gap is wide enough that the
gold set establishes it easily — −0.415 nDCG against `hybrid-rerank`, losing 49
of 67 questions.

The direction survives an advantage handed to it. A community report stands for
every judgment it was built from, so retrieving one fills several document
slots — 3.84 on average, 28 out of 40 at worst. That **inflates** its recall
rather than depressing it, and it still scores half. `documents_per_hit` is
recorded beside the metrics so the inflation stays visible.

The explanation is not that GraphRAG is broken. Community reports abstract away
case-specific detail, which is the correct thing for a corpus-level summary to
do and the wrong shape for finding a particular judgment. Graph structure buys
corpus-wide synthesis; it is not a retrieval index, and this corpus's questions
want retrieval.

None of this needed a re-index. The expensive phase — 3,750 entities, 1,505
relationships, 184 reports — was already on disk; only the embeddings were
missing, which is also why `local` search fails.

### What was supposed to help and didn't

Two changes were made because there was a clear argument for them. Both were
measured and neither survived. They are reported because a repository that only
shows what worked is not showing you its method.

**HyDE.** The hard tier exists precisely because a lay-phrased question shares
no vocabulary with the judgment. HyDE is the standard answer: have the model
write a passage in the judgment's register and search with that instead. It
should have been the biggest win available.

| configuration | hard R@5 | nDCG@10 | Δ vs base (paired) | p50 |
|---|---|---|---|---|
| `hybrid` | 0.872 | 0.847 | — | 2199ms |
| `hyde-hybrid` | **0.794** | 0.818 | −0.028 [−0.078, +0.019] | 6033ms |
| `hybrid-rerank` | 0.872 | 0.860 | — | 4167ms |
| `hyde-rerank` | 0.844 | 0.858 | +0.011 [−0.030, +0.049] | 4532ms |

It made the hard tier **worse** — the one place it was supposed to help —
while roughly tripling latency. Here is the actual hypothetical generated for
*"Can the police keep a permanent watch file on a man who finished serving his
sentence years ago?"*:

> In accordance with Article 21 of the Constitution, the right to privacy is
> fundamental […] pursuant to Section 509(1) of the Indian Penal Code […]

Section 509 is about insulting a woman's modesty. More importantly, the passage
never says **"history sheet"** — the term the answering judgment actually uses.
HyDE supplied *generic* constitutional register and a fabricated section number,
which pulled the query toward all eight Article 21 documents instead of the one.
The failure mode is specific: HyDE bridges register when the target vocabulary
is predictable from the question, and this corpus's hard questions are hard
exactly because it is not.

`hyde-rerank` does move R@1 from 0.659 to 0.688 and MRR from 0.879 to 0.897
while losing R@5 — it trades recall for precision at the top. That is a real
trade and still not a distinguishable difference at n=67.

**Chunk size.** 500 words was inherited from `settings.yaml` and never checked.
Sweeping 150 to 1000 words moves nDCG@10 between 0.743 and 0.783, with every
interval overlapping; the best point beats the default by +0.008. Chunk size is
not where the wins are in this corpus, and the inherited default costs nothing
measurable.

Two further ideas were dropped without being built, on the same reasoning that
justified building the rest. **Metadata pre-filtering** on `article_focus` would
only help questions that name their Article — the standard tier, which is
already saturated at 0.90 R@5 — and would risk the hard tier, which deliberately
avoids legal vocabulary. **Structured JSON generation** would add enforcement
the citation policy below already provides, at the cost of format adherence from
a 3B model.

### Cosine similarity is a poor signal for refusing to answer

Ten gold-set questions have no supporting document anywhere in the corpus. The
right behaviour is refusal. Whether that is achievable depends entirely on
whether the retriever scores those questions differently from real ones:

| retriever | answerable | out-of-corpus | separation |
|---|---|---|---|
| dense | 0.699 | 0.695 | **+0.004** |
| bm25 | 22.13 | 14.47 | +7.67 |
| **hybrid-rerank** | 0.576 | 0.249 | **+0.328** |

Dense similarity does not move between the two populations at all. A threshold
on a signal with four thousandths of separation is theatre, and the calibration
script prints a warning when it sees one — a cosine score can be a warning
label beside an answer, never a decision to withhold one.

The cross-encoder does separate. What the hard tier revealed is that
separating and *thresholding usefully* are different things:

| retriever | operating point | J-optimal point |
|---|---|---|
| dense | answers 91%, refuses 20% | answers 13%, refuses 100% |
| bm25 | answers 82%, refuses **70%** | answers 73%, refuses 80% |
| hybrid-rerank | answers **89%**, refuses 50% | answers 44%, refuses 100% |

The right-hand column is what this project used to report. Maximising Youden's
J weights both errors equally, and on the easier question set that cost
nothing — the J-optimal threshold answered every answerable question. On the
paraphrase tier it collapses to answering 44% of genuine questions, because a
real question phrased in lay language scores much like an out-of-corpus one.
The signal did not get worse. The questions got harder and exposed what the
criterion had been choosing all along.

Equal weighting is also the wrong loss here. A refused answerable question is a
visible failure. An answered out-of-corpus question is already caught
downstream, because citations are verified against the retrieved context. So
the threshold is now chosen as *the most specific point that still answers at
least 80% of answerable questions*, and the J-optimal alternative is recorded
beside it so the trade-off stays legible rather than baked in.

One reversal worth stating: **on the harder gold set BM25's raw score is a
better abstention signal than the cross-encoder's** — 82%/70% against 89%/50%.
The earlier version of this README concluded that reranking "pays for its
latency twice", once in ranking and again in calibration. On 25 easy questions
that was what the data showed. On 55 it is not, and the sentence has been
removed rather than kept because it read well.

The dashboard reads the threshold from the calibration file rather than
hardcoding it, which is why expanding the gold set moved it instead of
silently leaving a stale constant in the source.

### The second guardrail: citations that were checked but never acted on

`verify_citations` extracts every case name, Article, Section and reported
citation from an answer and looks for it in the context the generator was
actually given. It has run on every answer since it was written, and for most
of that time **nothing acted on the result**. An answer citing a case the
context never contained was checked, scored, recorded, and returned anyway.
That is a report, not a guardrail — and fabricated citations are the failure
this domain actually sanctions people for.

Three policies now:

| policy | behaviour |
|---|---|
| `warn` | attach the report, return the answer — **the default**, and what every judged number here was measured under |
| `retry` | regenerate once with the offending citations quoted back, keep the rewrite only if it verifies better |
| `refuse` | withhold the answer below a citation-accuracy floor |

`retry` quotes the failed citations verbatim rather than repeating "only cite
the extracts" — the model already had that instruction and broke it. It keeps
the second attempt only when the second attempt verifies better, because a
retry is free to make things worse and accepting it unconditionally would let a
guardrail lower citation accuracy while appearing to enforce it.

`warn` stays the default deliberately. Changing it would move the judged
numbers with no run to attribute the change to, and the judged evaluation of
these policies has not been run yet.

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

Both generators, same retriever, same questions, same independent judge, zero
unparsed judge responses on either run.

> These numbers are from the **45-question** gold set, before the hard tier was
> expanded to 30. The retrieval tables above are current at 77 questions; this
> one is not. Re-running it costs an hour of local generation per generator
> plus a day of judge quota, so it is labelled rather than quietly refreshed.
> n=35 answerable.

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

**RAGAS runs as a second opinion.** The built-in judge exists because RAGAS
does not enforce judge independence, report bootstrap intervals, salvage
truncated responses, or measure citation accuracy by regex — but a metric only
one implementation can reproduce is a metric to be suspicious of. So
`scripts/eval_ragas.py` scores the same answers with eight RAGAS metrics:
faithfulness, answer relevancy, context precision, context utilization, context
relevance, response groundedness, and two `AspectCritic` checks written for
this domain — whether a proposition is attributed to a named case, and whether
the answer admits when the context does not settle the question.

It runs in its own virtualenv, because RAGAS pins `langchain-core` below 0.3
and nothing else here can live with that.

Two things were dropped after being tried. RAGAS's non-LLM context metrics
compare reference against retrieved by Levenshtein similarity, which between a
gold paragraph and a correctly retrieved chunk is near zero — they scored 0.05
and 0.25 on answers the LLM metrics rated 1.00 faithful. That is a units
artifact, and `Recall@k` already answers the question properly.

**A thin run is withheld, not published.** RAGAS assigns NaN to any job that
exhausts its quota and carries on, so a run where most jobs failed still prints
a confident mean. Metrics scored on under 60% of the sample are reported as
`--` with their coverage, and the results file records `complete: false`.
Groq's free tier meters 200,000 tokens per day per model and eight metrics cost
roughly 20k tokens a question, so the full 67-question RAGAS sweep is a
paid-tier run; what fits free is a stratified sample.

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

- **40 files, 37 unique cases.** Three are duplicate pairs — the same judgment
  as filed by two reporters, which a word-sequence comparison puts at 0.86 to
  0.97 similarity. They are kept because the GraphRAG index was built over all
  40, and removing them from `input/` without re-indexing would make the two
  pipelines incomparable again. Both copies are listed as relevant in the gold
  set, since retrieving either is correct.
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
  cache.py           on-disk cache for temperature-0 calls
  router.py          NAIVE vs GRAPH by prototype similarity, no LLM call
  generation.py      prompt, generate, act on the citation check
  retrieval/         bm25 · dense · fusion (RRF) · rerank · hyde · graph · pipeline
  eval/              retrieval_metrics · judge
  guardrails/        abstention · citations
scripts/
  build_index.py         index input/ into ChromaDB, one collection per strategy
  validate_goldset.py    verify every annotation against the corpus text
  eval_retrieval.py      the ablation table; no LLM, runs in about a minute
  eval_answers.py        judged answer quality with an independent judge
  eval_ragas.py          the same answers, scored by RAGAS as a cross-check
  calibrate_abstention.py  choose the refusal threshold from data
  derive_paragraph_labels.py  locate which paragraphs carry each answer
  validate_judge.py      known-answer probes and cross-judge agreement
  sweep_chunk_size.py    is 500 words right, or just inherited?
  verify_claims.py       check every number in this README against reports/
  drift_check.py         scheduled regression check
  list_judge_models.py   which judge models a given key can actually reach
data/goldset.json    77 questions: 25 standard, 30 hard, 12 multi-hop, 10 out-of-corpus
tests/               128 tests, no network or GPU required
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
python scripts/derive_paragraph_labels.py
python scripts/eval_retrieval.py    # the ablation table, no LLM
python scripts/calibrate_abstention.py
python scripts/eval_answers.py --generator ollama:qwen2.5:3b
python scripts/validate_judge.py --probes
python scripts/make_report.py       # renders reports/EVALUATION.md

streamlit run app.py
```

The graph arms need `output/*.parquet` from `graphrag index`, which is not
committed; `eval_retrieval.py` skips them rather than failing when the index is
absent. HyDE arms cost one generation per query and are excluded from the
default sweep:

```bash
python scripts/eval_retrieval.py --configs hybrid hyde-hybrid --baseline hybrid
python scripts/sweep_chunk_size.py
```

Every temperature-0 call is cached to `.llm_cache/`, which is what makes
re-running any of this practical — 8.83s becomes 0.01s on a repeat. Sampling
calls are never cached, since replaying one would turn a sampling experiment
into a constant. `LEXGRAPH_CACHE=off` disables it.

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

## Deploying it

`deploy/DEPLOY.md` has the full procedure. The short version is that nothing
here was deployable, and the three blockers were worth fixing rather than
working around:

| blocker | resolution |
|---|---|
| dense retrieval called Ollama for every query embedding | ONNX on CPU — 6ms per query, no server, no quota |
| the index was 84MB of gitignored ChromaDB | a 1.9MB committed matrix and brute-force cosine |
| the GraphRAG tab shells out to the `graphrag` CLI | disabled when hosted; the measured result is shown instead |

Gemini embeddings were the obvious answer and do not work. The free tier allows
**1,000 embed requests per day** and counts a batch of 32 as 32 requests, so a
1,384-chunk index does not fit inside one day's quota, and every live query
would afterwards compete with rebuilds for what was left. Found by hitting it.

Swapping the embedder is a change to the thing being measured, so it was
measured. Against the same gold set:

| | local (nomic-embed-text + Chroma) | deployed (bge-small + numpy) |
|---|---|---|
| nDCG@10 | 0.860 | 0.859 |
| Recall@5 | 0.886 | 0.879 |
| ParaR@5 | 0.805 | **0.828** |
| p50 | 4161ms | **2316ms** |

Equivalent quality, better paragraph precision, and half the latency. One thing
did change: under the weaker base retriever, **reranking becomes statistically
distinguishable** for the first time in this project (+0.058 nDCG,
[+0.008, +0.114], 11 wins to 4). Reranking's value depends on how much its base
leaves to fix, which the local numbers were too strong to show.

The abstention threshold does **not** transfer between embedders — 0.045
deployed against 0.027 locally — so the deployment carries its own calibration
file. Verified end to end on all 65 gold-set answers: 87% of answerable questions
answered, 50% of out-of-corpus refused, matching the calibration exactly.

The hosted build answers gold-set questions from precomputed results so a public
URL cannot drain a free-tier key. Retrieval still runs live beside them, and the
UI says which half is which.

---

## Limitations

- **67 answerable questions establish three differences in the ablation.** Under a paired bootstrap `hybrid-rerank` is distinguishably better than `dense`, `bm25` and `graph-community`; `hybrid` and `graph-units` remain indistinguishable from it. The tier split and the R@1 movement are real
  observations; they are not the same claim, and this README no longer states
  them as one.
- **The judge has not been validated against human labels.** What it has been
  checked for is stated precisely, because the gap matters. It ranks 5 of 5
  known-answer probes correctly and separates them widely — grounded text over
  contradiction, real citation over invented one — so the scores are not noise
  (`scripts/validate_judge.py --probes`). That is a floor on validity and says
  nothing about the 3-versus-4 distinctions that actually move the reported
  means. The `context precision` control bounds self-disagreement at roughly
  0.37 points, which is consistency, not correctness: a judge can be perfectly
  consistent and perfectly wrong. Hand-scored labels and a Spearman ρ against
  them remain the missing piece, and no amount of model-on-model agreement
  substitutes for them.
- Scoring every metric in one judge call lets the answer leak into metrics that
  should not see it, which is part of that 0.37. Separate calls would remove
  it at seven times the quota, which the free tier does not support.
- **The RAGAS cross-check is built but not yet run at scale.** All eight metrics
  produce sensible values on a smoke sample, and the plumbing is verified end to
  end — but Groq's free tier allows 200,000 tokens per day per model and eight
  metrics cost roughly 20k tokens a question, so a stratified sample exhausted
  both models' daily budgets before completing. The script withholds any metric
  scored on under 60% of the sample rather than publishing a mean derived from
  one question, which is what a thin run otherwise produces. The full sweep is
  one command once quota resets, or immediately on a paid tier.
- **The judged answer-quality numbers were produced against the 45-question gold
  set**, before the hard tier was expanded. The retrieval tables are current at
  77 questions; the judged tables are not, and are labelled where they appear.
  Re-running them costs an hour of local generation per generator plus a day of
  judge quota.
- Paragraph-level ground truth covers 49 of 67 answerable questions. Six
  judgments carry no usable numbering, and a term-bearing paragraph marks where
  an answer is stated rather than the whole of where it is reasoned.
- The graph arms are scored through this repository's own retrieval over
  GraphRAG's artefacts, not through GraphRAG's query engine. That is deliberate
  — it holds the embedder and the metrics fixed so the comparison isolates the
  retrieval unit — but it means the result is about *what the graph produced*,
  not about how `global` search would rank with its own prompt-based
  aggregation on top.
- `graph-community` recall is inflated, not depressed, by expansion: a report
  covering 28 judgments fills 28 document slots. The finding survives that,
  which makes it safe in one direction only. Do not read its *absolute* recall
  as comparable to a chunk retriever's.
- **GraphRAG's `local` search method still crashes** on this index. Only
  `entity_description` was ever embedded into LanceDB — the text-unit and
  community stores local search reads were never built. Fixing it needs the
  embedding pass, not the hours-long extraction, but this repository routes
  around it rather than repairing GraphRAG's own store.
- HyDE, the citation `retry`/`refuse` policies, and the chunk-size sweep change
  what the pipeline *can* do; only the chunk-size sweep and the retrieval arms
  have been measured end to end. The judged effect of the citation policies on
  answer quality has not been run.
