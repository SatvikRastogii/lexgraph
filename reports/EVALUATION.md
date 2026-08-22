# LexGraph — evaluation results

Generated 2026-08-22 17:51. Gold set: 35 answerable, 10 out-of-corpus, over 40 judgments (38 unique cases).

Regenerate with `python scripts/make_report.py`.

## Retrieval

Document-level ground truth, no LLM in the loop.

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense-fixed` | 0.681 | 0.889 | [0.79, 0.97] | 0.906 | 0.980 | 2190ms |
| `dense` | 0.628 | 0.902 | [0.82, 0.97] | 0.897 | 0.940 | 2193ms |
| `bm25` | 0.636 | 0.857 | [0.74, 0.96] | 0.857 | 0.933 | 1ms |
| `hybrid` | 0.676 | 0.885 | [0.78, 0.97] | 0.900 | 0.960 | 2195ms |
| `hybrid-rerank` | 0.716 | 0.905 | [0.82, 0.97] | 0.933 | 0.980 | 4168ms |

Intervals are percentile bootstrap over per-question scores. Where they overlap, the difference between configurations is not established at this sample size.

## Abstention calibration

| retriever | answerable | out-of-corpus | separation | threshold | answers | refuses |
|---|---|---|---|---|---|---|
| `dense` | 0.737 | 0.698 | +0.039 | 0.758 | 48% | 90% |
| `bm25` | 27.748 | 14.478 | +13.270 | 21.069 | 72% | 100% |
| `hybrid-rerank` | 0.893 | 0.243 | +0.649 | 0.423 | 100% | 80% |

Thresholds maximise Youden's J against the gold set's out-of-corpus questions. Separation is the gap between the two populations' mean confidence; a retriever whose separation is near zero cannot support abstention at any threshold.

## Answer quality

### `ollama:qwen2.5:3b` · `hybrid-rerank` _(partial run)_

Judge: `gemini:gemini-3-flash-preview -> gemini:gemini-flash-latest -> gemini:gemini-3.7-flash -> gemini:gemini-3.1-flash-lite` — a different model family from the generator.

| metric | mean | 95% CI | n |
|---|---|---|---|
| answer relevancy | 4.22 | [3.78, 4.61] | 23 |
| citation accuracy | 4.88 | [4.67, 5.00] | 23 |
| coherence | 4.43 | [4.09, 4.74] | 23 |
| completeness | 3.39 | [2.83, 3.96] | 23 |
| context precision | 4.61 | [4.30, 4.87] | 23 |
| faithfulness | 3.04 | [2.48, 3.61] | 23 |
| hallucination | 3.48 | [2.96, 4.00] | 23 |
| legal reasoning | 3.04 | [2.48, 3.61] | 23 |

Median answer length: 246 words. Median latency: 12.7s.

