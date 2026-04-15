<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_8B-FF6F00?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-GraphRAG-6366F1?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Inference-100%25_Local-10B981?style=for-the-badge" />
</p>

<h1 align="center">⚖️ LexGraph</h1>
<h3 align="center">GraphRAG-Powered Legal Knowledge Navigator</h3>

<p align="center">
  <em>A production-grade, fully local RAG system that transforms 40 Supreme Court judgments into an interconnected knowledge graph — then proves its superiority over traditional vector search with an 8-metric RAGAS benchmark.</em>
</p>

<p align="center">
  <strong>GraphRAG wins 81% of questions (17/21) across ALL query categories.</strong>
</p>

---

## 🎯 What is LexGraph?

LexGraph is an end-to-end legal AI research system that demonstrates the **measurable superiority of Graph-based Retrieval Augmented Generation (GraphRAG)** over traditional Naive RAG for complex legal reasoning tasks.

Built entirely for local inference on consumer hardware (RTX 4050, 6GB VRAM), the system:

1. **Scrapes** 40 landmark Indian Supreme Court judgments from Indian Kanoon
2. **Indexes** them into a knowledge graph with 3,750 entities and 1,505 relationships using Microsoft's GraphRAG
3. **Routes** user queries through a Hybrid Semantic Router to the optimal pipeline
4. **Evaluates** both pipelines across 8 RAGAS metrics using LLM-as-Judge
5. **Visualizes** everything through a premium Streamlit dashboard with 7 interactive tabs

> **Why does this matter?** Traditional RAG treats documents as isolated text chunks. GraphRAG understands that *Maneka Gandhi v. Union of India* (1978) expanded rights established in *A.K. Gopalan v. State of Madras* (1950), and that both were synthesized in *K.S. Puttaswamy v. Union of India* (2017). This structural awareness produces fundamentally better answers for complex legal queries.

---

## 📊 Benchmark Results — RAGAS Evaluation

> **21 questions · 8 metrics · 336 LLM judge calls · 100% local inference**

### Overall Scores (1–5 scale)

| Metric | Naive RAG | GraphRAG | Δ | Winner |
|--------|:---------:|:--------:|:-:|--------|
| **Faithfulness** | 3.57 | **4.38** | +0.81 | 🟣 GraphRAG |
| **Answer Relevancy** | 4.05 | **4.62** | +0.57 | 🟣 GraphRAG |
| **Completeness** | 3.62 | **3.95** | +0.33 | 🟣 GraphRAG |
| **Hallucination Detection** ↑ | 3.43 | **4.05** | +0.62 | 🟣 GraphRAG |
| **Coherence** | 3.95 | **4.00** | +0.05 | 🟣 GraphRAG |
| **Citation Accuracy** | **3.67** | 3.62 | -0.05 | 🟢 Naive RAG |
| **Legal Reasoning** | 3.57 | **3.71** | +0.14 | 🟣 GraphRAG |
| **Context Precision** | **3.05** | N/A | — | 🟢 Naive RAG |

> ↑ Higher = fewer hallucinations (5 = zero)

### Performance by Query Category

| Category | Naive RAG | GraphRAG | Winner |
|----------|:---------:|:--------:|--------|
| Single-Hop Factual | 3.50 | **4.16** | 🟣 GraphRAG |
| Multi-Hop Relational | 3.58 | **3.92** | 🟣 GraphRAG |
| Global Thematic | 4.06 | **4.09** | 🟣 GraphRAG |
| Cross-Document Reasoning | 3.56 | **4.19** | 🟣 GraphRAG |
| Entity Relationship | 3.38 | **3.94** | 🟣 GraphRAG |

### Win/Loss Summary

| Pipeline | Wins | Ties | Win Rate |
|----------|:----:|:----:|:--------:|
| **GraphRAG** | **17** | 3 | **81.0%** |
| Naive RAG | 1 | 3 | 4.8% |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   HYBRID SEMANTIC      │  Latency: ~50ms
              │      ROUTER            │  No LLM inference
              │  (nomic-embed-text)    │  Cosine similarity
              └─────┬──────────┬───────┘
                    │          │
          ┌─────────┘          └──────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────────┐
│   NAIVE RAG      │           │     GRAPHRAG          │
│                  │           │                       │
│  ChromaDB        │           │  3,750 Entities       │
│  Vector Search   │           │  1,505 Relationships  │
│  Top-K Retrieval │           │  184 Communities       │
│  ~50s latency    │           │  Global Synthesis      │
│                  │           │  ~208s latency         │
└───────┬──────────┘           └───────┬───────────────┘
        │                              │
        └──────────┬───────────────────┘
                   ▼
        ┌──────────────────────┐
        │   LLAMA 3.1 8B       │
        │   (Ollama, local)    │
        │   Answer Generation  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  STREAMLIT DASHBOARD │
        │  7 Interactive Tabs  │
        │  Dark Glassmorphism  │
        └──────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid Router** over static rules | Embedding-based classification adapts to unseen queries without LLM inference (~50ms) |
| **GraphRAG Global** over Local mode | Global synthesizes across community reports — the true differentiator over vector search |
| **LLM-as-Judge** over heuristic metrics | Automated evaluation at scale without human annotators; reproducible benchmarks |
| **Ollama** over cloud APIs | 100% local inference; zero API costs; works offline; data privacy |
| **ChromaDB** for Naive RAG | Lightweight persistent vector store; no external server needed |

---

## ✨ Features

### 🔍 Query Engine
- **Side-by-side comparison mode** — run both pipelines on the same query
- **Automatic semantic routing** — queries classified in ~50ms without LLM
- **Confidence scoring** — cosine similarity-based confidence for Naive RAG
- **Pipeline override** — force a specific pipeline or run both

### ⏳ Temporal Timeline
- **Interactive Plotly scatter chart** — every case plotted chronologically, color-coded by era
- **4 legal eras** — Foundation (1950–69), Expansion (1970–84), Golden Triangle (1985–04), Digital Rights (2005+)
- **Vertical timeline cards** — glassmorphism cards with article badges, judges, and Indian Kanoon links
- **Radar chart** — most cited constitutional articles across the corpus
- **AI narrative generator** — Llama 3.1 writes a scholarly evolution narrative for any legal topic

### 🕸️ Knowledge Graph
- **Interactive network visualization** — 3,750 entities with spring-layout positioning
- **Color-coded entity types** — Cases, Persons, Articles, Legal Principles, Courts, Doctrines
- **Adjustable density** — slider to control displayed nodes (30–300)
- **Entity type distribution** — bar chart of the top 15 entity types

### ⚔️ Contradiction Detector
- **Keyword-based conflict detection** — scans relationships for dispute indicators (overruled, dissented, reversed, etc.)
- **AI analysis** — Llama 3.1 explains the legal significance of each dispute
- **Weighted ranking** — disputes sorted by edge weight for importance

### 🏆 RAGAS Benchmark Dashboard
- **Radar chart** — overlaid Naive vs GraphRAG across 7 metrics
- **Grouped bar chart** — head-to-head metric comparison
- **Category heatmap** — performance breakdown by query type
- **Win/loss donut chart** — 17/21 GraphRAG victories
- **Per-question breakdown** — full 21-question results table
- **Latency line chart** — response time comparison per question
- **Methodology cards** — evaluation framework details

### 📚 Data Explorer
- **Raw data inspection** — entities, relationships, communities, and documents
- **Entity search** — fuzzy search across 3,750 knowledge graph entities

### 📊 Session Analytics
- **Route distribution** — Naive vs GraphRAG usage pie chart
- **Query history** — timestamped log of all session queries
- **Knowledge graph health** — edge weights, entity types, dispute rates

---

## 🗂️ Project Structure

```
graphrag-project/
├── app.py                    # Streamlit UI (7 tabs, 1750+ lines)
├── naive_rag.py              # Custom Naive RAG pipeline with ChromaDB
├── hybrid_router.py          # Semantic embedding router (nomic-embed-text)
├── analyze_contradictions.py # Judicial contradiction detector
├── ragas_evaluation.py       # 8-metric RAGAS benchmark engine
├── scraper.py                # Indian Kanoon judgment scraper
├── settings.yaml             # GraphRAG configuration (optimized for RTX 4050)
├── benchmark_questions.json  # 21 evaluation questions across 5 categories
├── corpus_metadata.json      # Case metadata (titles, years, judges, articles)
├── ragas_results.json        # Raw evaluation results (all 21 questions)
├── ragas_report.md           # Generated benchmark report
├── input/                    # 40 Supreme Court judgment texts
├── output/                   # GraphRAG index (entities, relationships, communities)
├── chroma_db/                # ChromaDB vector store for Naive RAG
├── prompts/                  # Custom GraphRAG prompts
└── .gitignore                # Excludes large binary files
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| Ollama | Latest | Local LLM inference |
| NVIDIA GPU | 6GB+ VRAM | LLM acceleration (CPU fallback available) |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SatvikRastogii/lexgraph.git
cd lexgraph

# 2. Install dependencies
pip install streamlit plotly networkx pandas numpy requests chromadb pyarrow

# 3. Install and start Ollama
# Download from https://ollama.ai
ollama pull llama3.1
ollama pull nomic-embed-text
ollama serve  # Keep this running in a separate terminal

# 4. Install GraphRAG
pip install graphrag
```

### Building the Knowledge Graph Index

```bash
# This takes ~45 hours on RTX 4050 (6GB VRAM)
# The index only needs to be built once
graphrag index --root .
```

### Building the Naive RAG Vector Store

```bash
# Builds the ChromaDB collection from the input documents
python naive_rag.py
```

### Running the Application

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Running the RAGAS Evaluation

```bash
# Takes ~2 hours on RTX 4050
python ragas_evaluation.py
# Generates: ragas_results.json + ragas_report.md
```

---

## 📐 Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| **Source Documents** | 40 Supreme Court judgments |
| **Entities Extracted** | 3,750 |
| **Relationships Mapped** | 1,505 |
| **Community Reports** | 184 |
| **Entity Types** | 99 (Person, Organization, Legal Principle, Article, etc.) |
| **Indexing Time** | ~45 hours (RTX 4050, 6GB VRAM) |
| **Vector Store Chunks** | ~2,000 (ChromaDB) |

---

## 🧪 Evaluation Methodology

### Framework
The evaluation uses a custom **RAGAS (Retrieval Augmented Generation Assessment)** implementation with **LLM-as-Judge** scoring. Each question is:

1. Sent to **both** Naive RAG and GraphRAG pipelines
2. Both answers are scored independently by Llama 3.1 on **8 metrics**
3. Scores are on a **1–5 scale** (5 = best)
4. Results are aggregated across **21 questions** in **5 categories**

### Metrics

| # | Metric | What It Measures |
|---|--------|------------------|
| 1 | **Faithfulness** | Is every claim supported by the retrieved context? |
| 2 | **Answer Relevancy** | Does the answer directly address the question? |
| 3 | **Context Precision** | Are the retrieved chunks relevant? (Naive RAG only) |
| 4 | **Completeness** | Does the answer cover all aspects of the question? |
| 5 | **Hallucination Detection** | Does the answer fabricate cases, dates, or legal principles? |
| 6 | **Coherence** | Is the answer logically structured and well-organized? |
| 7 | **Citation Accuracy** | Does the answer reference real, verifiable case law? |
| 8 | **Legal Reasoning** | Does the answer demonstrate sound legal analytical methodology? |

### Question Categories

| Category | Count | Example |
|----------|:-----:|---------|
| Single-Hop Factual | 4 | *"What does Article 21 guarantee?"* |
| Multi-Hop Relational | 5 | *"How are Articles 14, 19, and 21 interconnected?"* |
| Global Thematic | 4 | *"What are the dominant themes across Supreme Court judgments?"* |
| Cross-Document Reasoning | 4 | *"How has Article 21 interpretation evolved from the 1950s to 2020s?"* |
| Entity Relationship | 4 | *"How are Puttaswamy, Maneka Gandhi, and Kesavananda Bharati linked?"* |

---

## ⚙️ Technical Deep Dives

### Hybrid Semantic Router

The router classifies queries **without any LLM inference** using pure embedding similarity:

```python
# Pre-embedded prototype questions for each category
SIMPLE_PROTOTYPES = ["What does Article 21 guarantee?", ...]
COMPLEX_PROTOTYPES = ["How has Article 21 evolved across decades?", ...]

# At query time: embed → cosine similarity → route decision
query_embedding = ollama_embed(query)
simple_score = mean(top_3_cosine_similarities(query_embedding, simple_embeddings))
complex_score = mean(top_3_cosine_similarities(query_embedding, complex_embeddings))

route = "NAIVE" if simple_score > complex_score else "GRAPH"
```

**Latency:** ~50ms (embedding only, no LLM inference)

### GraphRAG Global Synthesis

Unlike Naive RAG's point-lookup retrieval, GraphRAG's global mode:

1. Reads **184 community reports** synthesizing entity clusters
2. Maps the query to relevant communities
3. Generates an answer that synthesizes information **across the entire knowledge graph**
4. This enables multi-hop reasoning: *Case A → Principle B → Case C → Modern Interpretation D*

### Contradiction Detection

Two detection methods:

1. **Keyword Scan** — Regex matching against 19 dispute indicators (`overruled`, `dissented`, `reversed`, etc.)
2. **Triangle Principle** — Detects when Entity A has conflicting relationships with Entity B and Entity C (A→B positive, A→C positive, B→C negative)

---

## 🎨 UI Design

The dashboard uses a **Dark Glassmorphism** theme with:

- `backdrop-filter: blur(20px)` glass cards
- Indigo-purple gradient color palette (`#6366f1` → `#a78bfa` → `#c084fc`)
- Inter + JetBrains Mono typography
- Smooth hover animations with `cubic-bezier` easing
- Custom CSS scrollbars and tab styling
- Plotly charts with transparent backgrounds matching the dark theme

---

## 🔧 Hardware & Performance

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4050 (6GB VRAM) |
| LLM | Llama 3.1 8B (Q4_K_M quantization via Ollama) |
| Embedding Model | nomic-embed-text (768 dimensions) |
| Indexing Time | ~45 hours (one-time) |
| Naive RAG Latency | ~50s avg per query |
| GraphRAG Latency | ~208s avg per query |
| Router Latency | ~50ms per classification |
| RAGAS Evaluation | ~2 hours (21 questions × 2 pipelines × 8 metrics) |

---

## 📚 Corpus

The system indexes **40 landmark Indian Supreme Court judgments** spanning 1950–2024, covering:

- **Constitutional Articles:** 14 (Equality), 19 (Freedoms), 21 (Life & Liberty), 32 (Remedies)
- **Landmark Cases:** Kesavananda Bharati, Maneka Gandhi, K.S. Puttaswamy, Vishakha, and 36 others
- **Legal Principles:** Basic Structure Doctrine, Golden Triangle, Due Process, Right to Privacy
- **Source:** [Indian Kanoon](https://indiankanoon.org/) — India's largest legal search engine

---

## 🏆 Key Takeaways for Interviews

1. **Systems Engineering** — Managed a 45-hour indexing pipeline on resource-constrained hardware (6GB VRAM), implementing retry logic and concurrency limits
2. **Architecture Design** — The Hybrid Semantic Router is a novel contribution: sub-100ms query classification without LLM inference, using embedding similarity against prototype question banks
3. **Rigorous Evaluation** — 8-metric RAGAS benchmark with 336 automated LLM judge calls, demonstrating GraphRAG's 81% win rate with statistical breakdowns by category
4. **Domain Expertise** — Deep understanding of Indian constitutional law jurisprudence, from the Basic Structure Doctrine to the Right to Privacy
5. **Full-Stack Delivery** — End-to-end implementation: web scraping → data processing → knowledge graph construction → dual RAG pipelines → evaluation framework → premium UI

---

## 📄 License

This project is for educational and portfolio demonstration purposes.

---

## 👤 Author

**Satvik Rastogi**

Built with ❤️ using GraphRAG, Ollama, Streamlit, and 40 Supreme Court judgments — running 100% locally on consumer hardware.

---

<p align="center">
  <em>⚖️ "The Constitution is not a mere lawyer's document, it is a vehicle of life." — Dr. B.R. Ambedkar</em>
</p>
