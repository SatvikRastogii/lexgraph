# LexGraph — evaluation results

Generated 2026-08-23 01:25. Gold set: 55 answerable, 10 out-of-corpus, over 40 judgments (38 unique cases).

Regenerate with `python scripts/make_report.py`.

## Retrieval

Document-level ground truth, no LLM in the loop.

| configuration | R@1 | R@5 | R@5 95% CI | nDCG@10 | MRR | p50 |
|---|---|---|---|---|---|---|
| `dense-fixed` | 0.570 | 0.865 | [0.78, 0.94] | 0.809 | 0.830 | 2195ms |
| `dense` | 0.582 | 0.886 | [0.81, 0.95] | 0.831 | 0.839 | 2223ms |
| `bm25` | 0.586 | 0.802 | [0.70, 0.89] | 0.775 | 0.810 | 4ms |
| `hybrid` | 0.632 | 0.879 | [0.80, 0.95] | 0.847 | 0.873 | 2203ms |
| `hybrid-rerank` | 0.659 | 0.886 | [0.81, 0.95] | 0.860 | 0.879 | 4456ms |

### By difficulty tier

| configuration | hard R@5 | hard nDCG@10 | standard R@5 | standard nDCG@10 |
|---|---|---|---|---|
| `dense-fixed` | 0.844 | 0.728 | 0.889 | 0.906 |
| `dense` | 0.872 | 0.775 | 0.902 | 0.897 |
| `bm25` | 0.756 | 0.707 | 0.857 | 0.857 |
| `hybrid` | 0.872 | 0.803 | 0.887 | 0.901 |
| `hybrid-rerank` | 0.872 | 0.801 | 0.902 | 0.929 |

n per tier: hard = 30, standard = 25.

Intervals are percentile bootstrap over per-question scores. Where they overlap, the difference between configurations is not established at this sample size.

## Abstention calibration

| retriever | answerable | out-of-corpus | separation | threshold | answers | refuses |
|---|---|---|---|---|---|---|
| `dense` | 0.699 | 0.696 | +0.004 | 0.631 | 91% | 20% |
| `bm25` | 22.134 | 14.467 | +7.668 | 14.923 | 82% | 70% |
| `hybrid-rerank` | 0.576 | 0.248 | +0.328 | 0.027 | 89% | 50% |

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

