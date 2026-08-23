# LexGraph — evaluation results

Generated 2026-08-24 03:25. Gold set: 67 answerable, 10 out-of-corpus, over 40 judgments (38 unique cases).

Regenerate with `python scripts/make_report.py`.

## Retrieval

Document-level ground truth, no LLM in the loop.

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense` | 0.507 | 0.815 | [0.73, 0.89] | 0.777 | 0.823 | 2212ms |
| `bm25` | 0.522 | 0.788 | [0.70, 0.87] | 0.770 | 0.822 | 2ms |
| `hybrid` | 0.558 | 0.843 | [0.77, 0.91] | 0.827 | 0.878 | 2211ms |
| `hybrid-rerank` | 0.581 | 0.863 | [0.79, 0.93] | 0.848 | 0.880 | 4288ms |
| `graph-units` | 0.522 | 0.814 | [0.73, 0.89] | 0.806 | 0.830 | 2188ms |
| `graph-community` | 0.206 | 0.456 | [0.35, 0.56] | 0.433 | 0.432 | 2182ms |

### By difficulty tier

| configuration | hard R@5 | hard nDCG@10 | multihop R@5 | multihop nDCG@10 | standard R@5 | standard nDCG@10 |
|---|---|---|---|---|---|---|
| `dense` | 0.872 | 0.775 | 0.494 | 0.541 | 0.900 | 0.894 |
| `bm25` | 0.756 | 0.707 | 0.722 | 0.747 | 0.857 | 0.857 |
| `hybrid` | 0.872 | 0.803 | 0.681 | 0.739 | 0.885 | 0.899 |
| `hybrid-rerank` | 0.872 | 0.801 | 0.760 | 0.796 | 0.902 | 0.930 |
| `graph-units` | 0.889 | 0.796 | 0.467 | 0.584 | 0.890 | 0.925 |
| `graph-community` | 0.428 | 0.399 | 0.374 | 0.412 | 0.530 | 0.485 |

n per tier: hard = 30, multihop = 12, standard = 25.

Intervals are percentile bootstrap over per-question scores. Where they overlap, the difference between configurations is not established at this sample size.

## Abstention calibration

| retriever | answerable | out-of-corpus | separation | threshold | answers | refuses |
|---|---|---|---|---|---|---|
| `dense` | 0.701 | 0.696 | +0.005 | 0.631 | 91% | 20% |
| `bm25` | 21.040 | 14.467 | +6.574 | 13.202 | 87% | 60% |
| `hybrid-rerank` | 0.602 | 0.248 | +0.354 | 0.027 | 91% | 50% |

Thresholds maximise Youden's J against the gold set's out-of-corpus questions. Separation is the gap between the two populations' mean confidence; a retriever whose separation is near zero cannot support abstention at any threshold.

## Answer quality

### `ollama:llama3.1` · `hybrid-rerank`

Judge: `gemini:gemini-3-flash-preview -> gemini:gemini-3.5-flash -> gemini:gemini-flash-latest -> gemini:gemini-3.7-flash -> gemini:gemini-3.1-flash-lite -> gemini:gemini-3.1-flash-lite-preview` — a different model family from the generator.

| metric | mean | 95% CI | n |
|---|---|---|---|
| answer relevancy | 4.54 | [4.11, 4.89] | 35 |
| citation accuracy | 4.90 | [4.71, 5.00] | 35 |
| coherence | 4.91 | [4.77, 5.00] | 35 |
| completeness | 3.83 | [3.37, 4.26] | 35 |
| context precision | 4.20 | [3.83, 4.54] | 35 |
| faithfulness | 3.23 | [2.80, 3.66] | 35 |
| hallucination | 3.26 | [2.80, 3.71] | 35 |
| legal reasoning | 3.57 | [3.11, 4.03] | 35 |

Median answer length: 245 words. Median latency: 42.7s.

Abstention: refused 8/10 out-of-corpus questions; wrongly refused 6 answerable.

### `ollama:qwen2.5:3b` · `hybrid-rerank`

Judge: `gemini:gemini-3-flash-preview -> gemini:gemini-3.5-flash -> gemini:gemini-flash-latest -> gemini:gemini-3.7-flash -> gemini:gemini-3.1-flash-lite -> gemini:gemini-3.1-flash-lite-preview` — a different model family from the generator.

| metric | mean | 95% CI | n |
|---|---|---|---|
| answer relevancy | 3.86 | [3.37, 4.34] | 35 |
| citation accuracy | 4.92 | [4.78, 5.00] | 35 |
| coherence | 4.80 | [4.63, 4.94] | 35 |
| completeness | 3.11 | [2.60, 3.63] | 35 |
| context precision | 4.57 | [4.23, 4.86] | 35 |
| faithfulness | 3.03 | [2.51, 3.54] | 35 |
| hallucination | 3.86 | [3.43, 4.26] | 35 |
| legal reasoning | 2.94 | [2.46, 3.46] | 35 |

Median answer length: 221 words. Median latency: 11.8s.

Abstention: refused 8/10 out-of-corpus questions; wrongly refused 6 answerable.

