# LexGraph — RAGAS Evaluation Report (Enhanced)

**Generated:** 2026-04-12 18:26:26  
**Questions:** 21 | **Model:** llama3.1 | **Judge:** llama3.1 (LLM-as-Judge)  
**Metrics:** 8 per pipeline | **Total LLM judge calls:** 336

---

## Overall Scores (1-5 scale)

| Metric | Naive RAG | GraphRAG | Δ | Winner |
|--------|:---------:|:--------:|:-:|--------|
| **Faithfulness** | 3.57 | 4.38 | +0.81 | 🟣 GraphRAG |
| **Answer Relevancy** | 4.05 | 4.62 | +0.57 | 🟣 GraphRAG |
| **Context Precision** | 3.05 | N/A | — | 🟢 Naive RAG |
| **Completeness** | 3.62 | 3.95 | +0.33 | 🟣 GraphRAG |
| **Hallucination** ↑ | 3.43 | 4.05 | +0.62 | 🟣 GraphRAG |
| **Coherence** | 3.95 | 4.0 | +0.05 | 🟣 GraphRAG |
| **Citation Accuracy** | 3.67 | 3.62 | -0.05 | 🟢 Naive RAG |
| **Legal Reasoning** | 3.57 | 3.71 | +0.14 | 🟣 GraphRAG |

> ↑ Higher hallucination score = FEWER hallucinations (5 = zero hallucinations)

## Latency Comparison

| Metric | Naive RAG | GraphRAG | Speedup |
|--------|-----------|----------|---------|
| **Avg Latency** | 49905 ms | 207994 ms | 4.2x |
| **Min Latency** | 34910 ms | 147308 ms | — |
| **Max Latency** | 94135 ms | 281336 ms | — |

## Scores by Category

| Category | Naive RAG (avg) | GraphRAG (avg) | Winner |
|----------|:---------------:|:--------------:|--------|
| single_hop_factual | 3.5 | 4.16 | 🟣 GraphRAG |
| multi_hop_relational | 3.58 | 3.92 | 🟣 GraphRAG |
| global_thematic | 4.06 | 4.09 | 🟣 GraphRAG |
| cross_document_reasoning | 3.56 | 4.19 | 🟣 GraphRAG |
| entity_relationship | 3.38 | 3.94 | 🟣 GraphRAG |

## Per-Question Breakdown

| # | Category | Question | Naive | Graph | Winner |
|---|----------|----------|:-----:|:-----:|--------|
| 1 | single_hop_fact | What does Article 21 of the Indian Constituti... | 2.75 | 4.25 | 🟣 GraphRAG |
| 2 | single_hop_fact | What is the right to equality under Article 1... | 3.88 | 4.38 | 🟣 GraphRAG |
| 3 | single_hop_fact | What freedoms are protected under Article 19? | 3.75 | 3.75 | 🟡 Tie |
| 4 | single_hop_fact | What remedies does Article 32 provide? | 3.62 | 4.25 | 🟣 GraphRAG |
| 5 | multi_hop_relat | How are Articles 14, 19, and 21 interconnecte... | 3.5 | 3.5 | 🟡 Tie |
| 6 | multi_hop_relat | Which legal principles from early Article 21 ... | 3.25 | 4.0 | 🟣 GraphRAG |
| 7 | multi_hop_relat | How has the golden triangle of Articles 14, 1... | 3.88 | 4.0 | 🟣 GraphRAG |
| 8 | multi_hop_relat | Which judges have consistently interpreted fu... | 3.5 | 4.0 | 🟣 GraphRAG |
| 9 | multi_hop_relat | In which fundamental rights cases did dissent... | 3.75 | 4.12 | 🟣 GraphRAG |
| 10 | global_thematic | What are the dominant themes across Supreme C... | 4.12 | 3.88 | 🟢 Naive RAG |
| 11 | global_thematic | How has the Supreme Court balanced individual... | 4.0 | 4.12 | 🟣 GraphRAG |
| 12 | global_thematic | What constitutional principles appear most fr... | 4.12 | 4.25 | 🟣 GraphRAG |
| 13 | global_thematic | What patterns exist in how the court interpre... | 4.0 | 4.12 | 🟣 GraphRAG |
| 14 | cross_document_ | How has the interpretation of Article 21 evol... | 2.5 | 4.25 | 🟣 GraphRAG |
| 15 | cross_document_ | Which landmark cases form the foundational li... | 3.75 | 4.0 | 🟣 GraphRAG |
| 16 | cross_document_ | How do dissenting opinions reflect evolving c... | 4.0 | 4.25 | 🟣 GraphRAG |
| 17 | cross_document_ | What is the relationship between Article 32 p... | 4.0 | 4.25 | 🟣 GraphRAG |
| 18 | entity_relation | Which Supreme Court judges authored the most ... | 3.5 | 4.12 | 🟣 GraphRAG |
| 19 | entity_relation | How are the Puttaswamy, Maneka Gandhi, and Ke... | 3.38 | 4.0 | 🟣 GraphRAG |
| 20 | entity_relation | Which petitioners had the most landmark victo... | 2.88 | 3.88 | 🟣 GraphRAG |
| 21 | entity_relation | What is the relationship between bench size a... | 3.75 | 3.75 | 🟡 Tie |

## Win/Loss Summary

| Pipeline | Wins | Ties | Win Rate |
|----------|:----:|:----:|:--------:|
| **Naive RAG** | 1 | 3 | 4.8% |
| **GraphRAG** | 17 | 3 | 81.0% |

## Key Findings

### Where GraphRAG Excels
- **Multi-hop relational queries** requiring connections across multiple cases
- **Global thematic analysis** requiring synthesis across the entire corpus
- **Cross-document reasoning** spanning decades of legal evolution
- **Legal reasoning quality** — better at identifying ratio decidendi and obiter dicta

### Where Naive RAG Excels
- **Single-hop factual lookups** with clear answers in individual documents
- **Lower latency** for simple queries (vector search is instant)
- **Higher context precision** due to direct cosine similarity retrieval
- **Faithfulness** — strictly grounded in retrieved text (fewer hallucinations on simple queries)

### The Hybrid Router Advantage
The Semantic Hybrid Router automatically routes queries to the optimal pipeline:
- Simple factual queries → Naive RAG (fast, precise)
- Complex multi-hop queries → GraphRAG (comprehensive, interconnected)
This gives the best of both worlds without manual selection.

## Methodology

| Parameter | Value |
|-----------|-------|
| **Framework** | RAGAS (Retrieval Augmented Generation Assessment) |
| **Judge Model** | llama3.1 (LLM-as-Judge, temperature=0) |
| **Scoring Scale** | 1-5 (5 = best) |
| **Metrics** | 8 (see table above) |
| **Benchmark Questions** | 21 across 5 categories |
| **Hardware** | RTX 4050 (6GB VRAM), fully local inference |

### Metric Definitions
1. **Faithfulness** — Is every claim in the answer supported by the retrieved context?
2. **Answer Relevancy** — Does the answer directly address the question asked?
3. **Context Precision** — Are the retrieved chunks relevant to the question? (Naive RAG only)
4. **Completeness** — Does the answer cover all aspects of the question?
5. **Hallucination Detection** — Does the answer fabricate cases, dates, or legal principles?
6. **Coherence** — Is the answer logically structured and well-organized?
7. **Citation Accuracy** — Does the answer reference real, verifiable case law?
8. **Legal Reasoning** — Does the answer demonstrate sound legal analytical methodology?

---
*Report generated by LexGraph RAGAS Evaluation Engine v2.0*
