# LexGraph — evaluation results

Generated 2026-08-22 21:44. Gold set: 35 answerable, 10 out-of-corpus, over 40 judgments (38 unique cases).

Regenerate with `python scripts/make_report.py`.

## Retrieval

Document-level ground truth, no LLM in the loop.

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense-fixed` | 0.610 | 0.873 | [0.78, 0.95] | 0.859 | 0.914 | 2152ms |
| `dense` | 0.586 | 0.890 | [0.80, 0.96] | 0.861 | 0.895 | 2184ms |
| `bm25` | 0.578 | 0.817 | [0.71, 0.92] | 0.803 | 0.858 | 2ms |
| `hybrid` | 0.621 | 0.880 | [0.79, 0.96] | 0.867 | 0.914 | 2188ms |
| `hybrid-rerank` | 0.664 | 0.892 | [0.81, 0.96] | 0.890 | 0.924 | 4353ms |

### By difficulty tier

| configuration | hard R@5 | hard nDCG@10 | standard R@5 | standard nDCG@10 |
|---|---|---|---|---|
| `dense-fixed` | 0.833 | 0.742 | 0.889 | 0.906 |
| `dense` | 0.867 | 0.783 | 0.900 | 0.892 |
| `bm25` | 0.717 | 0.671 | 0.857 | 0.857 |
| `hybrid` | 0.867 | 0.795 | 0.885 | 0.897 |
| `hybrid-rerank` | 0.867 | 0.791 | 0.902 | 0.930 |

n per tier: hard = 10, standard = 25.

Intervals are percentile bootstrap over per-question scores. Where they overlap, the difference between configurations is not established at this sample size.

## Abstention calibration

| retriever | answerable | out-of-corpus | separation | threshold | answers | refuses |
|---|---|---|---|---|---|---|
| `dense` | 0.718 | 0.698 | +0.020 | 0.758 | 34% | 90% |
| `bm25` | 24.802 | 14.478 | +10.324 | 21.069 | 57% | 100% |
| `hybrid-rerank` | 0.736 | 0.243 | +0.492 | 0.375 | 83% | 80% |

Thresholds maximise Youden's J against the gold set's out-of-corpus questions. Separation is the gap between the two populations' mean confidence; a retriever whose separation is near zero cannot support abstention at any threshold.

## Answer quality

### `ollama:llama3.1` · `hybrid-rerank` _(partial run)_

Judge: `gemini:gemini-3-flash-preview -> gemini:gemini-flash-latest -> gemini:gemini-3.7-flash -> gemini:gemini-3.1-flash-lite` — a different model family from the generator.

| metric | mean | 95% CI | n |
|---|---|---|---|
| answer relevancy | 4.95 | [4.86, 5.00] | 22 |
| citation accuracy | 4.83 | [4.54, 5.00] | 22 |
| coherence | 4.86 | [4.64, 5.00] | 22 |
| completeness | 4.05 | [3.59, 4.45] | 22 |
| context precision | 4.36 | [3.95, 4.73] | 22 |
| faithfulness | 3.41 | [2.91, 3.91] | 22 |
| hallucination | 3.00 | [2.45, 3.59] | 22 |
| legal reasoning | 3.73 | [3.23, 4.23] | 22 |

Median answer length: 262 words. Median latency: 52.5s.

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

