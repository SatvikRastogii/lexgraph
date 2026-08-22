"""
LexGraph — GraphRAG Legal Knowledge Navigator
Premium Streamlit UI — Dark Glassmorphism Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import os
import time
import re
import requests
from datetime import datetime

import telemetry
telemetry.init_db()

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LexGraph — Legal Knowledge Navigator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS (Premium Dark Glassmorphism) ──────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg:#13110d;
        --bg-soft:#181510;
        --surface:#1b1813;
        --surface-2:#221e17;
        --line:#332e24;
        --line-soft:#272219;
        --ink:#ece6d8;
        --ink-muted:#a89e8b;
        --ink-dim:#766d5d;
        --brass:#c2a45a;
        --brass-bright:#ddbe77;
        --brass-dim:#7e6a34;
        --sage:#6fae8e;
        --slate:#7188a8;
        --brick:#c2574d;
        --amber:#cc9a44;
    }

    /* ── Global ── */
    .stApp{
        background:
            radial-gradient(900px 520px at 88% -8%, rgba(194,164,90,0.07), transparent 62%),
            var(--bg);
        color:var(--ink);
        font-family:'Inter', system-ui, -apple-system, sans-serif;
    }
    .block-container{ padding-top:2.4rem; max-width:1280px; }
    p, span, li, label, .stMarkdown{ color:var(--ink); }
    a{ color:var(--brass-bright); text-decoration:none; }
    a:hover{ color:var(--brass); text-decoration:underline; }

    [data-testid="stSidebar"]{
        background:var(--bg-soft);
        border-right:1px solid var(--line);
    }
    [data-testid="stSidebar"] *{ color:var(--ink-muted); }

    /* ── Typography ── */
    h1, h2, h3, h4{
        font-family:'Fraunces', Georgia, 'Times New Roman', serif !important;
        font-weight:600 !important;
        color:var(--ink) !important;
        letter-spacing:-0.01em;
    }
    h3{ font-size:1.45rem !important; margin-bottom:0.1rem !important; }
    h4{ font-size:1.12rem !important; color:var(--ink) !important; }
    /* italic intro lines under section headers read as editorial standfirst */
    .stMarkdown em{ color:var(--ink-dim); font-style:italic; }

    /* ── Cards (class name kept: .glass-card) ── */
    .glass-card{
        background:var(--surface);
        border:1px solid var(--line);
        border-left:3px solid var(--brass);
        border-radius:4px;
        padding:22px 26px;
        margin:14px 0;
        line-height:1.7;
        color:var(--ink);
        box-shadow:0 1px 0 rgba(0,0,0,0.3);
    }

    /* ── Masthead ── */
    .hero-title{
        font-family:'Fraunces', Georgia, serif;
        font-size:3.1rem;
        font-weight:600;
        color:var(--ink);
        text-align:left;
        letter-spacing:-0.02em;
        margin:0 0 2px 0;
        line-height:1.05;
    }
    .hero-title .mark{ color:var(--brass); }
    .hero-subtitle{
        text-align:left;
        color:var(--ink-dim);
        font-family:'Inter', sans-serif;
        font-size:0.78rem;
        font-weight:500;
        text-transform:uppercase;
        letter-spacing:0.32em;
        margin:0;
    }
    .masthead-rule{
        height:1px;
        background:linear-gradient(90deg, var(--brass) 0%, var(--line) 38%, transparent 100%);
        margin:16px 0 4px 0;
    }

    /* ── Stat tiles ── */
    .stat-card{
        background:var(--surface);
        border:1px solid var(--line);
        border-top:2px solid var(--brass-dim);
        border-radius:4px;
        padding:18px 16px;
        text-align:center;
        transition:border-color 0.2s ease;
    }
    .stat-card:hover{ border-top-color:var(--brass); }
    .stat-number{
        font-family:'Fraunces', Georgia, serif;
        font-size:2.2rem;
        font-weight:600;
        color:var(--brass);
        line-height:1.1;
    }
    .stat-label{
        color:var(--ink-dim);
        font-size:0.7rem;
        font-weight:500;
        text-transform:uppercase;
        letter-spacing:0.18em;
        margin-top:6px;
    }

    /* ── Pipeline tags ── */
    .tag-naive, .tag-graph{
        padding:3px 11px;
        border-radius:3px;
        font-size:0.68rem;
        font-weight:600;
        letter-spacing:0.12em;
        text-transform:uppercase;
        display:inline-block;
        border:1px solid;
    }
    .tag-naive{ background:rgba(111,174,142,0.12); color:var(--sage); border-color:rgba(111,174,142,0.4); }
    .tag-graph{ background:rgba(194,164,90,0.12); color:var(--brass-bright); border-color:rgba(194,164,90,0.45); }

    /* ── Confidence badges ── */
    .conf-high, .conf-medium, .conf-low{
        padding:5px 14px; border-radius:3px; font-weight:600;
        font-size:0.85rem; display:inline-block; border:1px solid;
    }
    .conf-high{ background:rgba(111,174,142,0.12); border-color:var(--sage); color:var(--sage); }
    .conf-medium{ background:rgba(204,154,68,0.12); border-color:var(--amber); color:var(--amber); }
    .conf-low{ background:rgba(194,87,77,0.12); border-color:var(--brick); color:#e0978f; }

    /* ── Dispute card ── */
    .dispute-card{
        background:var(--surface);
        border:1px solid var(--line);
        border-left:3px solid var(--brick);
        border-radius:4px;
        padding:18px 22px;
        margin:12px 0;
        color:var(--ink);
    }

    /* ── Citation ── */
    .citation-box{
        background:var(--bg-soft);
        border:1px solid var(--line);
        border-radius:3px;
        padding:11px 15px;
        margin:6px 0;
        font-size:0.88rem;
        color:var(--ink-muted);
    }
    .citation-box a{ color:var(--brass-bright); }

    /* ── Inputs ── */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div{
        background:var(--surface) !important;
        border:1px solid var(--line) !important;
        border-radius:3px !important;
        color:var(--ink) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus{
        border-color:var(--brass) !important;
        box-shadow:none !important;
    }
    .stTextInput input::placeholder{ color:var(--ink-dim) !important; }

    /* ── Buttons ── */
    .stButton > button{
        border-radius:3px !important;
        font-weight:600 !important;
        letter-spacing:0.02em !important;
        border:1px solid var(--line) !important;
        background:var(--surface) !important;
        color:var(--ink) !important;
        transition:all 0.18s ease !important;
    }
    .stButton > button:hover{ border-color:var(--brass) !important; color:var(--brass-bright) !important; }
    .stButton > button[kind="primary"]{
        background:var(--brass) !important;
        border-color:var(--brass) !important;
        color:#1a160d !important;
    }
    .stButton > button[kind="primary"]:hover{
        background:var(--brass-bright) !important;
        border-color:var(--brass-bright) !important;
        color:#1a160d !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"]{
        font-family:'Fraunces', Georgia, serif; color:var(--ink); font-weight:600;
    }
    [data-testid="stMetricLabel"]{
        text-transform:uppercase; letter-spacing:0.12em; font-size:0.7rem; color:var(--ink-dim);
    }

    /* ── Tabs (flat, underline on active) ── */
    .stTabs [data-baseweb="tab-list"]{
        gap:4px;
        background:transparent;
        border-bottom:1px solid var(--line);
        padding:0;
    }
    .stTabs [data-baseweb="tab"]{
        border-radius:0;
        color:var(--ink-dim);
        font-family:'Inter', sans-serif;
        font-weight:600;
        font-size:0.86rem;
        letter-spacing:0.02em;
        padding:8px 14px;
        border-bottom:2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover{ color:var(--ink-muted); }
    .stTabs [aria-selected="true"]{
        background:transparent !important;
        color:var(--brass-bright) !important;
        border-bottom:2px solid var(--brass) !important;
    }

    hr, [data-testid="stMarkdownContainer"] hr{ border-color:var(--line); }

    /* ── Hide chrome ── */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:var(--bg); }
    ::-webkit-scrollbar-thumb { background:var(--line); border-radius:0; }
    ::-webkit-scrollbar-thumb:hover { background:var(--brass-dim); }

    /* ── Timeline ── */
    .timeline-container{ position:relative; padding:18px 0 18px 30px; }
    .timeline-container::before{
        content:''; position:absolute; left:13px; top:0; bottom:0;
        width:1px; background:var(--line);
    }
    .timeline-era{
        font-family:'Fraunces', Georgia, serif;
        font-size:1.15rem; font-weight:600; letter-spacing:0.06em;
        margin:26px 0 10px 6px; padding:5px 14px;
        border-radius:3px; display:inline-block; border:1px solid;
    }
    .era-foundation{ background:rgba(111,174,142,0.1); color:var(--sage); border-color:rgba(111,174,142,0.32); }
    .era-expansion{ background:rgba(194,164,90,0.1); color:var(--brass-bright); border-color:rgba(194,164,90,0.34); }
    .era-golden{ background:rgba(204,138,74,0.1); color:#d49a5c; border-color:rgba(204,138,74,0.34); }
    .era-modern{ background:rgba(113,136,168,0.1); color:#8fa4c2; border-color:rgba(113,136,168,0.34); }
    .timeline-card{
        position:relative; background:var(--surface);
        border:1px solid var(--line); border-radius:4px;
        padding:16px 20px; margin:9px 0 9px 6px;
        transition:border-color 0.2s ease;
    }
    .timeline-card:hover{ border-color:var(--brass-dim); }
    .timeline-card::before{
        content:''; position:absolute; left:-24px; top:22px;
        width:9px; height:9px; border-radius:50%;
        border:2px solid var(--brass); background:var(--bg); z-index:2;
    }
    .timeline-year{
        font-family:'JetBrains Mono', monospace;
        font-size:0.76rem; font-weight:500; color:var(--brass);
        letter-spacing:0.1em;
    }
    .timeline-title{
        font-family:'Fraunces', Georgia, serif;
        font-size:1.05rem; font-weight:600; color:var(--ink);
        margin:3px 0; line-height:1.35;
    }
    .timeline-article-badge{
        display:inline-block; background:var(--bg-soft);
        border:1px solid var(--line); color:var(--ink-muted);
        padding:1px 9px; border-radius:3px; font-size:0.7rem;
        font-weight:500; margin:2px 3px 2px 0; letter-spacing:0.03em;
    }
    .timeline-judges{ font-size:0.8rem; color:var(--ink-dim); margin-top:4px; font-style:italic; }
    .timeline-link{ color:var(--brass-bright); font-size:0.82rem; font-weight:600; }
    .timeline-link:hover{ color:var(--brass); text-decoration:underline; }
    .timeline-narrative{
        background:var(--surface);
        border:1px solid var(--line);
        border-left:3px solid var(--brass);
        border-radius:4px; padding:22px 26px; margin:16px 0;
        line-height:1.85; color:var(--ink); font-size:0.96rem;
        font-family:'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS & HELPERS ─────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
LLM_MODEL = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
OUTPUT_DIR = "output"
METADATA_FILE = "corpus_metadata.json"

DISPUTE_KEYWORDS = [
    "overrule", "overruled", "dissent", "dissented", "contradict",
    "conflict", "oppose", "departed", "disagree", "struck down",
    "invalidate", "reversed", "set aside", "distinguished", "narrowed",
    "modified", "diluted", "curtailed", "restricted",
]

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "naive_count" not in st.session_state:
    st.session_state.naive_count = 0
if "graph_count" not in st.session_state:
    st.session_state.graph_count = 0


# ─── DATA LOADING (Cached) ───────────────────────────────────────────────────

@st.cache_data
def load_entities():
    path = os.path.join(OUTPUT_DIR, "entities.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_data
def load_relationships():
    path = os.path.join(OUTPUT_DIR, "relationships.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_data
def load_communities():
    path = os.path.join(OUTPUT_DIR, "community_reports.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def detect_contradictions():
    """Pre-compute contradictions from relationships."""
    df = load_relationships()
    if df.empty or "description" not in df.columns:
        return pd.DataFrame()
    df["description"] = df["description"].fillna("").astype(str)
    pattern = "|".join([rf"\b{kw}\b" for kw in DISPUTE_KEYWORDS])
    conflicts = df[df["description"].str.contains(pattern, case=False, na=False)].copy()
    if "weight" in conflicts.columns:
        conflicts = conflicts.sort_values(by="weight", ascending=False)
    return conflicts


# ─── OLLAMA INTERFACE ─────────────────────────────────────────────────────────

def ollama_chat(prompt, model=LLM_MODEL, max_tokens=500):
    """Call Ollama via REST API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "⚠ Cannot connect to Ollama. Please ensure `ollama serve` is running."
    except Exception as e:
        return f"⚠ Error: {str(e)}"


def ollama_embed(text):
    """Get embedding from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"][0])
    except:
        return None


# ─── HYBRID SEMANTIC ROUTER ──────────────────────────────────────────────────

SIMPLE_PROTOTYPES = [
    "What does Article 21 of the Indian Constitution guarantee?",
    "What is the right to equality under Article 14?",
    "What year was the Maneka Gandhi case decided?",
    "Who was the petitioner in the Kesavananda Bharati case?",
    "What did the Supreme Court hold regarding passport cancellation?",
    "What freedoms are protected under Article 19?",
    "What remedies does Article 32 provide?",
    "What is the basic structure doctrine?",
    "Define reasonable restrictions under Article 19(2).",
    "What are the grounds for preventive detention under Article 22?",
]
COMPLEX_PROTOTYPES = [
    "How are Articles 14, 19, and 21 interconnected in Supreme Court judgments?",
    "How has the interpretation of Article 21 evolved from the 1950s to 2020s?",
    "Which legal principles from early Article 21 cases were expanded in privacy judgments?",
    "What is the relationship between the Maneka Gandhi, Puttaswamy, and Kesavananda Bharati cases?",
    "How has the Supreme Court balanced individual rights against state power across decades?",
    "Which landmark cases form the foundational lineage of privacy rights in India?",
    "How did the golden triangle of Articles 14, 19, and 21 evolve across constitutional bench decisions?",
    "Compare the approaches of different benches to the scope of personal liberty.",
    "What patterns exist in how the court interprets reasonable restrictions across multiple cases?",
    "In which fundamental rights cases did dissenting opinions later become majority views?",
]

@st.cache_resource
def init_router_embeddings():
    """Pre-embed router prototypes (runs once at server boot)."""
    simple_embs, complex_embs = [], []
    for q in SIMPLE_PROTOTYPES:
        emb = ollama_embed(q)
        if emb is not None:
            simple_embs.append(emb)
    for q in COMPLEX_PROTOTYPES:
        emb = ollama_embed(q)
        if emb is not None:
            complex_embs.append(emb)
    return np.array(simple_embs) if simple_embs else None, np.array(complex_embs) if complex_embs else None


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def route_query(query, simple_embs, complex_embs):
    """Classify query as NAIVE or GRAPH via semantic similarity."""
    if simple_embs is None or complex_embs is None:
        return "GRAPH", 0.5  # fallback

    q_emb = ollama_embed(query)
    if q_emb is None:
        return "GRAPH", 0.5

    simple_scores = [cosine_sim(q_emb, e) for e in simple_embs]
    complex_scores = [cosine_sim(q_emb, e) for e in complex_embs]

    # Single best match per bank, not top-3 average: averaging let queries that
    # share keywords with several mediocre complex-bank matches (e.g. "what is
    # article 21" vs multiple Article-21-flavored complex prototypes) out-vote
    # one excellent simple-bank match. Verified against benchmark_questions.json:
    # max-of-bank scores 21/22 vs top-3-average's 19/22, no regressions.
    avg_simple = max(simple_scores)
    avg_complex = max(complex_scores)

    route = "NAIVE" if avg_simple > avg_complex else "GRAPH"

    # Confidence = margin between the winning and losing bank's best match,
    # scaled against the typical margin range seen on real (non-prototype-
    # identical) queries in benchmark_questions.json (~0.02-0.3). The old
    # ratio formula (max/(max+min))*2-1 compressed even healthy margins
    # toward 0 because both banks' top cosine similarities are usually high
    # for any in-domain legal query -- it couldn't distinguish a correct-but-
    # close call from a genuinely wrong one.
    margin = abs(avg_simple - avg_complex)
    confidence = round(min(1.0, margin / 0.3), 3)

    return route, confidence


# ─── GRAPHRAG QUERY (via CLI) ────────────────────────────────────────────────

# GraphRAG global search fans out many sequential LLM calls over the community
# reports, so a complex query can run for several minutes on local hardware.
# A simple query already measured ~250s here, so 300s killed harder queries mid-
# run; 600s gives real headroom. (The durable fix is a model that isn't CPU-
# offloaded — see the latency notes.) Override via GRAPHRAG_QUERY_TIMEOUT.
GRAPHRAG_TIMEOUT = int(os.getenv("GRAPHRAG_QUERY_TIMEOUT", "600"))


def run_graphrag_query(query, method="local"):
    """Run graphrag query via subprocess."""
    import subprocess
    try:
        result = subprocess.run(
            ["graphrag", "query", "--root", ".", "--method", method, query],
            capture_output=True, text=True, timeout=GRAPHRAG_TIMEOUT,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = result.stdout
        # Extract just the answer (after SUCCESS line)
        if "SUCCESS" in output:
            parts = output.split("SUCCESS")
            return parts[-1].strip() if len(parts) > 1 else output.strip()
        return output.strip() if output.strip() else result.stderr.strip()
    except subprocess.TimeoutExpired:
        mins = GRAPHRAG_TIMEOUT // 60
        return (f"⚠ GraphRAG query exceeded the {mins}-minute limit. Global search "
                "fans out many LLM calls and is expensive on local hardware — try a "
                "narrower question, or use a faster model so each call completes sooner.")
    except Exception as e:
        return f"⚠ Error: {str(e)}"


# ─── NAIVE RAG QUERY (inline) ────────────────────────────────────────────────

def run_naive_rag_query(query):
    """Run a lightweight Naive RAG query using ChromaDB + Ollama."""
    latency = {}
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        t0 = time.perf_counter()
        client = chromadb.PersistentClient(path="chroma_db")
        embed_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name=EMBEDDING_MODEL, url=OLLAMA_HOST
        )
        collection = client.get_collection(name="legal_judgments", embedding_function=embed_fn)

        results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas", "distances"])
        latency["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        chunks = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        similarities = [round(1 - d, 4) for d in distances]
        avg_conf = sum(similarities) / len(similarities) if similarities else 0

        context = "\n\n---\n\n".join([
            f"[Source: {m.get('source', '?')} | Year: {m.get('year', '?')}]\n{c}"
            for c, m in zip(chunks, metadatas)
        ])

        prompt = f"""You are a legal research assistant specializing in Indian constitutional law.
Answer the question based ONLY on the provided court judgment excerpts.
Be specific, cite the sources, and acknowledge if the context is insufficient.

QUESTION: {query}

RELEVANT JUDGMENT EXCERPTS:
{context}

ANSWER:"""

        t0 = time.perf_counter()
        answer = ollama_chat(prompt, max_tokens=800)
        latency["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        sources = []
        for m, sim in zip(metadatas, similarities):
            sources.append({
                "source": m.get("source", "unknown"),
                "year": m.get("year", "?"),
                "similarity": sim,
            })

        return {
            "answer": answer,
            "confidence": avg_conf,
            "sources": sources,
            "latency": latency,
        }
    except Exception as e:
        return {"answer": f"⚠ Naive RAG error: {str(e)}", "confidence": 0.0, "sources": [], "latency": latency}


# ─── KNOWLEDGE GRAPH VISUALIZATION ──────────────────────────────────────────

def build_knowledge_graph_figure(entities_df, relationships_df, max_nodes=150):
    """Build an interactive Plotly network graph."""
    G = nx.Graph()

    # Add top entities by degree
    if "title" not in entities_df.columns:
        return None

    # Count entity appearances in relationships
    source_counts = relationships_df["source"].value_counts() if "source" in relationships_df.columns else pd.Series()
    target_counts = relationships_df["target"].value_counts() if "target" in relationships_df.columns else pd.Series()
    all_counts = source_counts.add(target_counts, fill_value=0).sort_values(ascending=False)
    top_entities = set(all_counts.head(max_nodes).index)

    # Filter entities and relationships
    ent_filtered = entities_df[entities_df["title"].isin(top_entities)]
    rel_filtered = relationships_df[
        (relationships_df["source"].isin(top_entities)) & 
        (relationships_df["target"].isin(top_entities))
    ]

    # Map types to colors
    type_colors = {
        "CASE": "#7188a8",
        "PERSON": "#cc9a44",
        "ORGANIZATION": "#6fae8e",
        "LEGAL PRINCIPLE": "#c2574d",
        "ARTICLE": "#c2a45a",
        "COURT": "#5b9aa0",
        "DOCTRINE": "#a87fb0",
        "EVENT": "#cf8a4a",
    }

    for _, row in ent_filtered.iterrows():
        title = row["title"]
        etype = row.get("type", "UNKNOWN").upper()
        G.add_node(title, type=etype, color=type_colors.get(etype, "#766d5d"))

    for _, row in rel_filtered.iterrows():
        if row["source"] in G.nodes and row["target"] in G.nodes:
            weight = row.get("weight", 1)
            G.add_edge(row["source"], row["target"], weight=weight)

    if len(G.nodes) == 0:
        return None

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="rgba(194, 164, 90, 0.2)"),
        hoverinfo="none", mode="lines"
    )

    # Node traces
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(f"{node}<br>Type: {G.nodes[node].get('type', '?')}<br>Connections: {degree}")
        node_color.append(G.nodes[node].get("color", "#766d5d"))
        node_size.append(max(8, min(35, degree * 3)))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hoverinfo="text", text=[n[:20] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=7, color="#a89e8b"),
        hovertext=node_text,
        marker=dict(
            size=node_size, color=node_color,
            line=dict(width=1, color="rgba(255,255,255,0.1)"),
            opacity=0.9,
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(19, 17, 13, 0)",
        paper_bgcolor="rgba(19, 17, 13, 0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650,
        hoverlabel=dict(bgcolor="#221e17", font_color="#ece6d8", bordercolor="#c2a45a"),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ██  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.7rem;">Lex<span class="mark">Graph</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle" style="font-size:0.62rem;letter-spacing:0.22em;">Legal Knowledge Navigator</p>', unsafe_allow_html=True)
    st.markdown("---")

    # System status
    st.markdown("### System Status")
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        st.success("Ollama: Online", icon="✅")
    except:
        st.error("Ollama: Offline", icon="❌")

    entities_df = load_entities()
    relationships_df = load_relationships()
    communities_df = load_communities()

    if not entities_df.empty:
        st.success(f"GraphRAG Index: {len(entities_df)} entities", icon="✅")
    else:
        st.warning("GraphRAG: No index found", icon="⚠️")

    st.markdown("---")

    # Live stats
    st.markdown("### Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Disputes", len(detect_contradictions()) if not relationships_df.empty else 0)

    col3, col4 = st.columns(2)
    col3.metric("→ Naive", st.session_state.naive_count)
    col4.metric("→ Graph", st.session_state.graph_count)

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Built with GraphRAG v2.0, ChromaDB, NetworkX, "
        "and Llama 3.1 8B running 100% locally on RTX 4050."
    )
    st.caption("© 2026 LexGraph — Satvik Rastogi")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown('<h1 class="hero-title">Lex<span class="mark">Graph</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Graph-Augmented Retrieval for Indian Constitutional Law</p>', unsafe_allow_html=True)
st.markdown('<div class="masthead-rule"></div>', unsafe_allow_html=True)
st.markdown("")

# Stats row
entities_df = load_entities()
relationships_df = load_relationships()
communities_df = load_communities()
contradictions_df = detect_contradictions()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(entities_df)}</div>
        <div class="stat-label">Entities</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(relationships_df)}</div>
        <div class="stat-label">Relationships</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(communities_df)}</div>
        <div class="stat-label">Communities</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{len(contradictions_df)}</div>
        <div class="stat-label">Disputes</div>
    </div>""", unsafe_allow_html=True)
with col5:
    entity_types = entities_df["type"].nunique() if "type" in entities_df.columns else 0
    st.markdown(f"""<div class="stat-card">
        <div class="stat-number">{entity_types}</div>
        <div class="stat-label">Entity Types</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_query, tab_timeline, tab_graph, tab_disputes, tab_explorer, tab_ragas, tab_analytics = st.tabs([
    "Query",
    "Timeline",
    "Knowledge Graph",
    "Contradictions",
    "Data Explorer",
    "Benchmark",
    "Analytics",
])

# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 1: QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.markdown("### Intelligent Legal Query Engine")
    st.markdown("*Queries are automatically routed via the Semantic Hybrid Router to the optimal pipeline.*")

    query = st.text_input(
        "Ask a legal question about Indian constitutional law:",
        placeholder="e.g., How has the right to privacy evolved across Supreme Court decisions?",
        key="main_query",
    )

    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        # "local" defaults last: it crashes (LanceDB vector-dim mismatch between
        # the index's stored embeddings and the live nomic-embed-text config) --
        # same issue ragas_evaluation.py already worked around by forcing "global".
        graphrag_method = st.selectbox("GraphRAG Method", ["global", "local"], index=0)
    with col_mode2:
        force_pipeline = st.selectbox("Pipeline Override", ["Auto (Hybrid Router)", "Force Naive RAG", "Force GraphRAG", "Run Both (Compare)"], index=0)

    if st.button("Execute Query", type="primary", use_container_width=True) and query:
        st.session_state.total_queries += 1

        # Route
        with st.spinner("🧭 Routing query via Semantic Embedding Router..."):
            simple_embs, complex_embs = init_router_embeddings()
            route_start = time.perf_counter()
            route, route_conf = route_query(query, simple_embs, complex_embs)
            route_ms = round((time.perf_counter() - route_start) * 1000, 1)

        # Override
        run_both = False
        if force_pipeline == "Force Naive RAG":
            route = "NAIVE"
        elif force_pipeline == "Force GraphRAG":
            route = "GRAPH"
        elif force_pipeline == "Run Both (Compare)":
            run_both = True

        # Display routing decision
        if route == "NAIVE":
            st.markdown(f'<span class="tag-naive">ROUTED → NAIVE RAG</span> &nbsp; Latency: {route_ms}ms', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="tag-graph">ROUTED → GRAPHRAG</span> &nbsp; Latency: {route_ms}ms', unsafe_allow_html=True)

        st.markdown("")

        if run_both:
            col_naive, col_graph = st.columns(2)

            with col_naive:
                st.markdown("#### Naive RAG")
                with st.spinner("Searching vectors..."):
                    t0 = time.perf_counter()
                    naive_result = run_naive_rag_query(query)
                    naive_ms = round((time.perf_counter() - t0) * 1000, 1)

                st.markdown(naive_result["answer"])
                st.caption(f"⏱ {naive_ms}ms")

                if naive_result["sources"]:
                    with st.expander("📚 Citations"):
                        for s in naive_result["sources"]:
                            st.markdown(f'<div class="citation-box">{s["source"]} ({s["year"]}) — sim: {s["similarity"]}</div>', unsafe_allow_html=True)

            with col_graph:
                st.markdown("#### GraphRAG")
                with st.spinner("Traversing knowledge graph..."):
                    t0 = time.perf_counter()
                    graph_answer = run_graphrag_query(query, graphrag_method)
                    graph_ms = round((time.perf_counter() - t0) * 1000, 1)

                st.markdown(f'<div class="conf-high">Knowledge Graph Query</div>', unsafe_allow_html=True)
                st.markdown(graph_answer)
                st.caption(f"⏱ {graph_ms}ms")

            # Latency comparison chart
            st.markdown("#### Latency Comparison")
            lat_fig = go.Figure(data=[
                go.Bar(name="Naive RAG", x=["Pipeline"], y=[naive_ms], marker_color="#6fae8e"),
                go.Bar(name="GraphRAG", x=["Pipeline"], y=[graph_ms], marker_color="#c2a45a"),
            ])
            lat_fig.update_layout(
                barmode="group", height=250,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a89e8b"),
                yaxis_title="Milliseconds",
            )
            st.plotly_chart(lat_fig, use_container_width=True)

            st.session_state.naive_count += 1
            st.session_state.graph_count += 1

            naive_ok = not str(naive_result["answer"]).strip().startswith("⚠")
            telemetry.log_query_event(
                query_text=query, route="NAIVE",
                route_confidence=route_conf, route_latency_ms=route_ms,
                pipeline_latency_ms=naive_ms, total_latency_ms=route_ms + naive_ms,
                success=naive_ok,
                error_message=None if naive_ok else naive_result["answer"],
                naive_confidence=naive_result["confidence"],
                latency_breakdown=naive_result.get("latency"),
            )
            graph_ok = not str(graph_answer).strip().startswith("⚠")
            telemetry.log_query_event(
                query_text=query, route="GRAPH",
                route_confidence=route_conf, route_latency_ms=route_ms,
                pipeline_latency_ms=graph_ms, total_latency_ms=route_ms + graph_ms,
                success=graph_ok,
                error_message=None if graph_ok else graph_answer,
            )

        elif route == "NAIVE":
            with st.spinner("🔍 Searching vector store..."):
                t0 = time.perf_counter()
                result = run_naive_rag_query(query)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)

            st.markdown(f'<div class="glass-card">{result["answer"]}</div>', unsafe_allow_html=True)

            if result["sources"]:
                with st.expander("📚 Citations & Sources"):
                    for s in result["sources"]:
                        st.markdown(f'<div class="citation-box">{s["source"]} ({s["year"]}) — Similarity: {s["similarity"]}</div>', unsafe_allow_html=True)

            st.caption(f"⏱ Total latency: {elapsed}ms")
            st.session_state.naive_count += 1

            naive_ok = not str(result["answer"]).strip().startswith("⚠")
            telemetry.log_query_event(
                query_text=query, route="NAIVE",
                route_confidence=route_conf, route_latency_ms=route_ms,
                pipeline_latency_ms=elapsed, total_latency_ms=route_ms + elapsed,
                success=naive_ok,
                error_message=None if naive_ok else result["answer"],
                naive_confidence=result["confidence"],
                latency_breakdown=result.get("latency"),
            )

        else:  # GRAPH
            with st.spinner("Querying Knowledge Graph (global search can take several minutes on local hardware)..."):
                t0 = time.perf_counter()
                answer = run_graphrag_query(query, graphrag_method)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)

            st.markdown(f'<div class="conf-high">Knowledge Graph Response</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f'<div class="glass-card">{answer}</div>', unsafe_allow_html=True)
            st.caption(f"⏱ Total latency: {elapsed}ms")
            st.session_state.graph_count += 1

            graph_ok = not str(answer).strip().startswith("⚠")
            telemetry.log_query_event(
                query_text=query, route="GRAPH",
                route_confidence=route_conf, route_latency_ms=route_ms,
                pipeline_latency_ms=elapsed, total_latency_ms=route_ms + elapsed,
                success=graph_ok,
                error_message=None if graph_ok else answer,
            )

        # Save to history
        st.session_state.query_history.append({
            "query": query, "route": route, "timestamp": datetime.now().isoformat(),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 2: TEMPORAL TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_timeline:
    st.markdown("### Temporal Evolution of Indian Constitutional Law")
    st.markdown("*Trace how fundamental rights jurisprudence evolved across decades — from independence to the digital age.*")

    # Load metadata
    metadata = load_metadata()
    case_docs = metadata.get("documents", [])

    # Filter cases with valid years and build timeline data
    timeline_data = []
    for doc in case_docs:
        year = doc.get("year", None)
        title = doc.get("title", "Unknown Case")
        if year and str(year).isdigit() and int(str(year)) >= 1950:
            timeline_data.append({
                "year": int(str(year)),
                "title": title[:80],
                "full_title": title,
                "article_focus": doc.get("article_focus", ""),
                "articles_cited": doc.get("articles_cited", []),
                "judges": doc.get("judges", ""),
                "url": doc.get("url", ""),
                "filename": doc.get("filename", ""),
            })

    # Sort by year
    timeline_data.sort(key=lambda x: x["year"])

    if not timeline_data:
        st.warning("No metadata found. Ensure `corpus_metadata.json` exists with case data.")
    else:
        # ── Interactive Plotly Scatter Timeline ──
        st.markdown("#### Interactive Timeline")

        # Define eras and assign colors
        def get_era(year):
            if year < 1970:
                return "Foundation Era"
            elif year < 1985:
                return "Expansion Era"
            elif year < 2005:
                return "Golden Triangle Era"
            else:
                return "Digital Rights Era"

        era_colors = {
            "Foundation Era": "#6fae8e",
            "Expansion Era": "#c2a45a",
            "Golden Triangle Era": "#cf8a4a",
            "Digital Rights Era": "#7188a8",
        }

        # Build scatter data
        years = [d["year"] for d in timeline_data]
        titles = [d["title"] for d in timeline_data]
        eras = [get_era(y) for y in years]
        colors = [era_colors[e] for e in eras]
        articles = [d["article_focus"] or "General" for d in timeline_data]
        urls = [d["url"] for d in timeline_data]

        # Stagger y-positions to prevent overlap
        y_positions = []
        year_count = {}
        for y in years:
            year_count[y] = year_count.get(y, 0) + 1
            y_positions.append(year_count[y])

        # Create figure with traces per era
        fig_timeline = go.Figure()

        for era_name, era_color in era_colors.items():
            era_indices = [i for i, e in enumerate(eras) if e == era_name]
            if not era_indices:
                continue

            era_years = [years[i] for i in era_indices]
            era_y = [y_positions[i] for i in era_indices]
            era_titles = [titles[i] for i in era_indices]
            era_articles = [articles[i] for i in era_indices]

            hover_text = [
                f"<b>{t}</b><br>Year: {y}<br>Focus: {a}"
                for t, y, a in zip(era_titles, era_years, era_articles)
            ]

            fig_timeline.add_trace(go.Scatter(
                x=era_years, y=era_y,
                mode="markers+text",
                name=era_name,
                text=[t[:25] + "..." if len(t) > 25 else t for t in era_titles],
                textposition="top center",
                textfont=dict(size=8, color="#a89e8b"),
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=14,
                    color=era_color,
                    opacity=0.85,
                    line=dict(width=2, color="rgba(255,255,255,0.15)"),
                    symbol="diamond",
                ),
            ))

        # Add decade marker lines
        for decade in range(1950, 2040, 10):
            fig_timeline.add_vline(
                x=decade, line_dash="dot",
                line_color="rgba(194, 164, 90, 0.15)", line_width=1,
            )
            fig_timeline.add_annotation(
                x=decade, y=0, text=f"{decade}s",
                showarrow=False,
                font=dict(size=10, color="rgba(194, 164, 90, 0.4)"),
                yshift=-20,
            )

        fig_timeline.update_layout(
            height=500,
            plot_bgcolor="rgba(19, 17, 13, 0)",
            paper_bgcolor="rgba(19, 17, 13, 0)",
            font=dict(color="#a89e8b", family="Inter"),
            xaxis=dict(
                title="Year", gridcolor="rgba(194,164,90,0.08)",
                range=[1948, 2028], dtick=5,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="", showgrid=False, showticklabels=False, zeroline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
                bgcolor="rgba(27,24,19,0.7)",
                bordercolor="rgba(194,164,90,0.2)",
                borderwidth=1,
                font=dict(size=11),
            ),
            hoverlabel=dict(
                bgcolor="#221e17", font_color="#ece6d8",
                bordercolor="#c2a45a", font_size=12,
            ),
            margin=dict(l=20, r=20, t=40, b=60),
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

        # ── Era Filter ──
        st.markdown("---")
        col_filter1, col_filter2 = st.columns([1, 2])
        with col_filter1:
            selected_era = st.selectbox("Filter by Era", [
                "All Eras",
                "Foundation Era (1950–1969)",
                "Expansion Era (1970–1984)",
                "Golden Triangle Era (1985–2004)",
                "Digital Rights Era (2005–present)",
            ])
        with col_filter2:
            article_filter = st.text_input(
                "Filter by Article",
                placeholder="e.g., 21, 14, 19",
                key="timeline_article_filter",
            )

        # Apply filters
        filtered = timeline_data
        if "Foundation" in selected_era:
            filtered = [d for d in filtered if d["year"] < 1970]
        elif "Expansion" in selected_era:
            filtered = [d for d in filtered if 1970 <= d["year"] < 1985]
        elif "Golden" in selected_era:
            filtered = [d for d in filtered if 1985 <= d["year"] < 2005]
        elif "Digital" in selected_era:
            filtered = [d for d in filtered if d["year"] >= 2005]

        if article_filter:
            art_search = article_filter.strip()
            filtered = [
                d for d in filtered
                if art_search in str(d.get("article_focus", ""))
                or art_search in str(d.get("articles_cited", []))
            ]

        # ── Vertical Timeline Cards ──
        st.markdown(f"#### Case Timeline ({len(filtered)} cases)")

        # Group by era
        current_era = None
        era_map = {
            "Foundation Era": ("era-foundation", "🏛️ FOUNDATION ERA", "1950–1969 · Birth of Constitutional Jurisprudence"),
            "Expansion Era": ("era-expansion", "📖 EXPANSION ERA", "1970–1984 · Broadening Fundamental Rights"),
            "Golden Triangle Era": ("era-golden", "⚖️ GOLDEN TRIANGLE ERA", "1985–2004 · Articles 14, 19, 21 Convergence"),
            "Digital Rights Era": ("era-modern", "🔮 DIGITAL RIGHTS ERA", "2005–Present · Privacy, Dignity, Technology"),
        }

        st.markdown('<div class="timeline-container">', unsafe_allow_html=True)

        for d in filtered:
            era = get_era(d["year"])
            if era != current_era:
                current_era = era
                era_class, era_label, era_desc = era_map.get(era, ("", era, ""))
                st.markdown(
                    f'<div class="timeline-era {era_class}">{era_label}<br>'
                    f'<span style="font-size:0.7rem;font-weight:400;letter-spacing:0.5px;">{era_desc}</span></div>',
                    unsafe_allow_html=True,
                )

            # Build article badges
            art_badges = ""
            focus = d.get("article_focus", "")
            if focus:
                for art in str(focus).split(","):
                    art = art.strip()
                    if art:
                        art_badges += f'<span class="timeline-article-badge">{art}</span>'

            # Build judges line
            judges_html = ""
            judges = d.get("judges", "")
            if judges and judges != "Unknown":
                judges_str = judges if isinstance(judges, str) else ", ".join(judges[:3])
                if len(judges_str) > 80:
                    judges_str = judges_str[:80] + "..."
                judges_html = f'<div class="timeline-judges">{judges_str}</div>'

            # Build URL link
            link_html = ""
            if d.get("url"):
                link_html = f'<a class="timeline-link" href="{d["url"]}" target="_blank">View on Indian Kanoon →</a>'

            st.markdown(f"""
            <div class="timeline-card">
                <div class="timeline-year">{d["year"]}</div>
                <div class="timeline-title">{d["full_title"][:100]}</div>
                {art_badges}
                {judges_html}
                {link_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Year Distribution Chart ──
        st.markdown("---")
        st.markdown("#### Judicial Activity by Decade")

        decade_counts = {}
        for d in timeline_data:
            decade = (d["year"] // 10) * 10
            decade_label = f"{decade}s"
            decade_counts[decade_label] = decade_counts.get(decade_label, 0) + 1

        sorted_decades = sorted(decade_counts.items(), key=lambda x: x[0])
        decade_labels = [x[0] for x in sorted_decades]
        decade_values = [x[1] for x in sorted_decades]

        fig_decades = go.Figure(data=[
            go.Bar(
                x=decade_labels, y=decade_values,
                marker=dict(
                    color=decade_values,
                    colorscale=[[0, "#3a3320"], [0.5, "#c2a45a"], [1, "#7188a8"]],
                    line=dict(width=0),
                ),
                text=decade_values,
                textposition="outside",
                textfont=dict(color="#d2b66a", size=12, family="Inter"),
            )
        ])
        fig_decades.update_layout(
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b", family="Inter"),
            xaxis=dict(title="Decade", gridcolor="rgba(194,164,90,0.05)"),
            yaxis=dict(title="Number of Cases", gridcolor="rgba(194,164,90,0.08)"),
            margin=dict(l=40, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_decades, use_container_width=True)

        # ── Article Frequency Radar ──
        st.markdown("#### Most Cited Articles")

        article_counts = {}
        for d in timeline_data:
            cited = d.get("articles_cited", [])
            if isinstance(cited, list):
                for art in cited:
                    art_clean = str(art).strip()
                    if art_clean and art_clean != "?":
                        label = f"Article {art_clean}" if not art_clean.startswith("Article") else art_clean
                        article_counts[label] = article_counts.get(label, 0) + 1

        if article_counts:
            top_articles = sorted(article_counts.items(), key=lambda x: -x[1])[:12]
            art_labels = [a[0] for a in top_articles]
            art_values = [a[1] for a in top_articles]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=art_values + [art_values[0]],  # close the shape
                theta=art_labels + [art_labels[0]],
                fill="toself",
                fillcolor="rgba(194, 164, 90, 0.15)",
                line=dict(color="#d2b66a", width=2),
                marker=dict(size=6, color="#ddbe77"),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(
                        tickfont=dict(size=10, color="#a89e8b"),
                        gridcolor="rgba(194,164,90,0.1)",
                        linecolor="rgba(194,164,90,0.15)",
                    ),
                    radialaxis=dict(
                        visible=True,
                        gridcolor="rgba(194,164,90,0.08)",
                        tickfont=dict(color="#766d5d", size=9),
                    ),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a89e8b"),
                height=450,
                margin=dict(l=60, r=60, t=40, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── AI Narrative Button ──
        st.markdown("---")
        st.markdown("#### AI-Generated Legal Evolution Narrative")
        st.caption("Ask Llama 3.1 to narrate the evolution of a specific constitutional theme across the timeline.")

        narrative_topic = st.text_input(
            "Topic to trace through history:",
            placeholder="e.g., Right to Privacy, Article 21, Freedom of Speech",
            key="narrative_input",
        )

        if st.button("Generate Narrative", type="primary", key="gen_narrative") and narrative_topic:
            # Gather case names for context
            case_list = "\n".join([
                f"- {d['year']}: {d['title'][:70]} (Focus: {d.get('article_focus', '?')})"
                for d in timeline_data
            ])

            prompt = f"""You are an elite Indian constitutional law historian.
Using the following chronological list of Supreme Court cases, write a compelling narrative
about how "{narrative_topic}" evolved across decades of Indian constitutional jurisprudence.

Cases in the corpus:
{case_list[:3000]}

Write a 300-word narrative that:
1. Identifies the key turning points and landmark cases
2. Explains how the legal doctrine expanded or contracted over time
3. Highlights the connections between cases across different eras
4. Ends with the current state of the law

Use a scholarly but accessible tone. Reference specific case names and years."""

            with st.spinner("🧠 Generating legal evolution narrative..."):
                narrative = ollama_chat(prompt, max_tokens=600)

            st.markdown(
                f'<div class="timeline-narrative">{narrative}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 3: KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown("### Interactive Knowledge Graph")
    st.markdown("*Visualizing the legal knowledge extracted by GraphRAG from 40 Supreme Court judgments.*")

    if entities_df.empty:
        st.warning("No GraphRAG index found. Run `graphrag index --root .` first.")
    else:
        max_nodes = st.slider("Max nodes to display", 30, 300, 100, step=10)

        with st.spinner("Building graph visualization..."):
            fig = build_knowledge_graph_figure(entities_df, relationships_df, max_nodes)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not build graph. Check entity data.")

        # Entity type distribution
        if "type" in entities_df.columns:
            st.markdown("#### Entity Type Distribution")
            type_counts = entities_df["type"].value_counts().head(15)
            fig_types = px.bar(
                x=type_counts.index, y=type_counts.values,
                labels={"x": "Entity Type", "y": "Count"},
                color=type_counts.values,
                color_continuous_scale=[[0, "#3a3320"], [0.5, "#9a8038"], [1, "#ddbe77"]],
            )
            fig_types.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a89e8b"), height=350,
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_types, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 4: CONTRADICTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_disputes:
    st.markdown("### Contradiction Detector")
    st.markdown("*Algorithmically detected judicial disputes, overruled precedents, and conflicting legal principles.*")

    if contradictions_df.empty:
        st.info("No contradictions detected. This may occur if the graph has no dispute-related edges.")
    else:
        st.markdown(f"**{len(contradictions_df)} potential judicial disputes detected** across the knowledge graph.")

        # Show disputes
        for idx, (_, row) in enumerate(contradictions_df.head(15).iterrows(), 1):
            source = row.get("source", "Unknown")
            target = row.get("target", "Unknown")
            desc = str(row.get("description", ""))[:300]
            weight = row.get("weight", 0)

            st.markdown(f"""
            <div class="dispute-card">
                <strong>Dispute #{idx}</strong> &nbsp; <span style="color:#766d5d;">Weight: {weight}</span><br>
                <span style="color:#ef4444;font-weight:700;">{source}</span> 
                &nbsp;↔&nbsp; 
                <span style="color:#f59e0b;font-weight:700;">{target}</span><br>
                <span style="color:#a89e8b;font-size:0.9rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # AI analysis button
        st.markdown("---")
        if st.button("Generate AI Analysis of Top 3 Disputes", type="primary"):
            for idx, (_, row) in enumerate(contradictions_df.head(3).iterrows(), 1):
                source = row.get("source", "Unknown")
                target = row.get("target", "Unknown")
                desc = str(row.get("description", ""))

                with st.spinner(f"Analyzing dispute #{idx}: {source} vs {target}..."):
                    prompt = f"""You are an elite Indian constitutional law analyst.
Entity A: {source}
Entity B: {target}
Relationship: {desc}
Concisely explain the legal contradiction or overruling. Keep under 150 words."""
                    analysis = ollama_chat(prompt, max_tokens=200)

                st.markdown(f"""
                <div class="glass-card">
                    <strong>🧠 AI Analysis — {source} vs {target}</strong><br><br>
                    {analysis}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 5: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    st.markdown("### Data Explorer")
    st.markdown("*Inspect the raw GraphRAG data tables.*")

    explorer_tab = st.selectbox("Select Dataset", [
        "Entities", "Relationships", "Communities", "Documents"
    ])

    if explorer_tab == "Entities" and not entities_df.empty:
        st.dataframe(entities_df, use_container_width=True, height=500)
    elif explorer_tab == "Relationships" and not relationships_df.empty:
        st.dataframe(relationships_df, use_container_width=True, height=500)
    elif explorer_tab == "Communities" and not communities_df.empty:
        st.dataframe(communities_df, use_container_width=True, height=500)
    elif explorer_tab == "Documents":
        docs_path = os.path.join(OUTPUT_DIR, "documents.parquet")
        if os.path.exists(docs_path):
            docs_df = pd.read_parquet(docs_path)
            st.dataframe(docs_df, use_container_width=True, height=500)
        else:
            st.info("No documents parquet found.")
    else:
        st.info("No data available for this dataset.")

    # Search entities
    st.markdown("---")
    st.markdown("#### Entity Search")
    search_term = st.text_input("Search for an entity:", placeholder="e.g., Maneka Gandhi")
    if search_term and not entities_df.empty and "title" in entities_df.columns:
        results = entities_df[entities_df["title"].str.contains(search_term, case=False, na=False)]
        if len(results) > 0:
            st.success(f"Found {len(results)} matching entities.")
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No matching entities found.")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 6: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.markdown("### Session Analytics")

    if not st.session_state.query_history:
        st.info("No queries yet. Start asking questions in the Query Engine tab!")
    else:
        # Route distribution pie chart
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Route Distribution")
            naive_n = st.session_state.naive_count
            graph_n = st.session_state.graph_count
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Naive RAG", "GraphRAG"],
                values=[naive_n, graph_n],
                marker=dict(colors=["#6fae8e", "#c2a45a"]),
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(color="#ece6d8"),
            )])
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a89e8b"), height=300, showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("#### Query History")
            for entry in reversed(st.session_state.query_history[-10:]):
                tag = "tag-naive" if entry["route"] == "NAIVE" else "tag-graph"
                st.markdown(
                    f'<span class="{tag}">{entry["route"]}</span> &nbsp; {entry["query"][:60]}...',
                    unsafe_allow_html=True,
                )

    # Graph health metrics
    st.markdown("---")
    st.markdown("#### Knowledge Graph Health")
    
    if not entities_df.empty and not relationships_df.empty:
        health_col1, health_col2, health_col3, health_col4 = st.columns(4)
        
        avg_weight = relationships_df["weight"].mean() if "weight" in relationships_df.columns else 0
        max_weight = relationships_df["weight"].max() if "weight" in relationships_df.columns else 0
        
        health_col1.metric("Avg Edge Weight", f"{avg_weight:.2f}")
        health_col2.metric("Max Edge Weight", f"{max_weight:.1f}")
        health_col3.metric("Entity Types", entities_df["type"].nunique() if "type" in entities_df.columns else 0)
        health_col4.metric("Dispute Rate", f"{len(contradictions_df)/max(len(relationships_df),1)*100:.1f}%")

    # ─── Persistent Query Telemetry (survives restarts — backed by SQLite) ──
    st.markdown("---")
    st.markdown("#### Query Telemetry (Persistent)")

    telemetry_events = telemetry.fetch_recent_events(limit=200)

    if not telemetry_events:
        st.info("No durable query history yet. Run some queries in the Query Engine tab!")
    else:
        tel_df = pd.DataFrame(telemetry_events)
        tel_df["timestamp"] = pd.to_datetime(tel_df["timestamp"])

        tel_col1, tel_col2, tel_col3 = st.columns(3)
        failure_rate = telemetry.fetch_failure_rate(window=50)
        tel_col1.metric("Logged Events", len(tel_df))
        tel_col2.metric("Failure Rate (last 50)", f"{failure_rate * 100:.1f}%")
        tel_col3.metric(
            "Median Latency",
            f"{tel_df['pipeline_latency_ms'].median():.0f} ms" if tel_df["pipeline_latency_ms"].notna().any() else "N/A",
        )

        lat_trend_fig = px.scatter(
            tel_df.sort_values("timestamp"),
            x="timestamp", y="pipeline_latency_ms", color="route",
            color_discrete_map={"NAIVE": "#6fae8e", "GRAPH": "#c2a45a"},
            labels={"pipeline_latency_ms": "Latency (ms)", "timestamp": "Time"},
        )
        lat_trend_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b"), height=300,
        )
        st.plotly_chart(lat_trend_fig, use_container_width=True)

        with st.expander("📋 Raw Query Log"):
            route_filter = st.selectbox("Filter by route", ["All", "NAIVE", "GRAPH"], key="telemetry_route_filter")
            show_df = tel_df if route_filter == "All" else tel_df[tel_df["route"] == route_filter]
            st.dataframe(
                show_df[["timestamp", "route", "query_text", "pipeline_latency_ms", "success", "error_message"]],
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ██  TAB 7: RAGAS BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ragas:
    st.markdown("### RAGAS Evaluation Benchmark")
    st.markdown("*Comprehensive 8-metric comparison of Naive RAG vs GraphRAG using LLM-as-Judge evaluation.*")

    RAGAS_FILE = "ragas_results.json"

    if not os.path.exists(RAGAS_FILE):
        st.warning("No RAGAS results found. Run `python ragas_evaluation.py` to generate the benchmark.")
    else:
        ragas_data = json.load(open(RAGAS_FILE, "r", encoding="utf-8"))
        METRICS = ["faithfulness", "relevancy", "completeness", "hallucination", "coherence", "citation_accuracy", "legal_reasoning"]
        METRIC_LABELS = ["Faithfulness", "Relevancy", "Completeness", "Hallucination↑", "Coherence", "Citation Acc.", "Legal Reasoning"]

        # ── Compute averages ──
        naive_avgs, graph_avgs = [], []
        for m in METRICS:
            n_scores = [d["naive"][m]["score"] for d in ragas_data if m in d["naive"] and d["naive"][m]["score"] > 0]
            g_scores = [d["graphrag"][m]["score"] for d in ragas_data if m in d["graphrag"] and d["graphrag"][m]["score"] > 0]
            naive_avgs.append(round(sum(n_scores)/len(n_scores), 2) if n_scores else 0)
            graph_avgs.append(round(sum(g_scores)/len(g_scores), 2) if g_scores else 0)

        # ── Hero stat cards ──
        wins_g = sum(1 for n, g in zip(naive_avgs, graph_avgs) if g > n)
        wins_n = sum(1 for n, g in zip(naive_avgs, graph_avgs) if n > g)
        ties = sum(1 for n, g in zip(naive_avgs, graph_avgs) if n == g)

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{len(ragas_data)}</div><div class="stat-label">Questions</div></div>', unsafe_allow_html=True)
        with hc2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{len(METRICS)}</div><div class="stat-label">Metrics</div></div>', unsafe_allow_html=True)
        with hc3:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="-webkit-text-fill-color:#ddbe77;">{wins_g}</div><div class="stat-label">GraphRAG Wins</div></div>', unsafe_allow_html=True)
        with hc4:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="-webkit-text-fill-color:#10b981;">{wins_n}</div><div class="stat-label">Naive RAG Wins</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Radar Chart ──
        st.markdown("#### Metric Radar — Naive RAG vs GraphRAG")

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=naive_avgs + [naive_avgs[0]], theta=METRIC_LABELS + [METRIC_LABELS[0]],
            fill="toself", fillcolor="rgba(111, 174, 142, 0.12)",
            line=dict(color="#6fae8e", width=2), name="Naive RAG",
            marker=dict(size=6),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=graph_avgs + [graph_avgs[0]], theta=METRIC_LABELS + [METRIC_LABELS[0]],
            fill="toself", fillcolor="rgba(194, 164, 90, 0.12)",
            line=dict(color="#c2a45a", width=2), name="GraphRAG",
            marker=dict(size=6),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 5], gridcolor="rgba(194,164,90,0.1)", tickfont=dict(color="#766d5d", size=9)),
                angularaxis=dict(tickfont=dict(size=11, color="#cfc6b5"), gridcolor="rgba(194,164,90,0.12)"),
            ),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b"), height=480, margin=dict(l=60, r=60, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Grouped Bar Chart ──
        st.markdown("#### Head-to-Head Comparison")

        fig_bars = go.Figure(data=[
            go.Bar(name="Naive RAG", x=METRIC_LABELS, y=naive_avgs, marker_color="#6fae8e",
                   text=[f"{v:.1f}" for v in naive_avgs], textposition="outside",
                   textfont=dict(color="#9bd0b8", size=11)),
            go.Bar(name="GraphRAG", x=METRIC_LABELS, y=graph_avgs, marker_color="#c2a45a",
                   text=[f"{v:.1f}" for v in graph_avgs], textposition="outside",
                   textfont=dict(color="#ddbe77", size=11)),
        ])
        fig_bars.update_layout(
            barmode="group", height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b"), yaxis=dict(range=[0, 5.5], gridcolor="rgba(194,164,90,0.08)", title="Score (1-5)"),
            xaxis=dict(tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # ── Category Heatmap ──
        st.markdown("---")
        st.markdown("#### Performance by Category")

        categories = list(dict.fromkeys(d["category"] for d in ragas_data))
        cat_naive, cat_graph = [], []
        for cat in categories:
            cat_items = [d for d in ragas_data if d["category"] == cat]
            n_vals = []
            g_vals = []
            for d in cat_items:
                for m in METRICS:
                    if m in d["naive"] and d["naive"][m]["score"] > 0:
                        n_vals.append(d["naive"][m]["score"])
                    if m in d["graphrag"] and d["graphrag"][m]["score"] > 0:
                        g_vals.append(d["graphrag"][m]["score"])
            cat_naive.append(round(sum(n_vals)/len(n_vals), 2) if n_vals else 0)
            cat_graph.append(round(sum(g_vals)/len(g_vals), 2) if g_vals else 0)

        cat_labels = [c.replace("_", " ").title() for c in categories]
        deltas = [round(g - n, 2) for n, g in zip(cat_naive, cat_graph)]

        fig_cat = go.Figure(data=[
            go.Bar(name="Naive RAG", y=cat_labels, x=cat_naive, orientation="h", marker_color="#6fae8e",
                   text=[f"{v:.2f}" for v in cat_naive], textposition="inside", textfont=dict(color="white", size=11)),
            go.Bar(name="GraphRAG", y=cat_labels, x=cat_graph, orientation="h", marker_color="#c2a45a",
                   text=[f"{v:.2f}" for v in cat_graph], textposition="inside", textfont=dict(color="white", size=11)),
        ])
        fig_cat.update_layout(
            barmode="group", height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b"), xaxis=dict(range=[0, 5], title="Avg Score", gridcolor="rgba(194,164,90,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=180, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # ── Delta advantage column ──
        st.markdown("**GraphRAG Advantage (Δ) by Category:**")
        delta_cols = st.columns(len(categories))
        for i, (cat, d_val) in enumerate(zip(cat_labels, deltas)):
            color = "#ddbe77" if d_val > 0 else "#ef4444" if d_val < 0 else "#766d5d"
            sign = "+" if d_val > 0 else ""
            delta_cols[i].markdown(
                f'<div style="text-align:center;"><span style="font-size:1.5rem;font-weight:800;color:{color};">{sign}{d_val}</span>'
                f'<br><span style="font-size:0.7rem;color:#766d5d;">{cat}</span></div>', unsafe_allow_html=True)

        # ── Win/Loss Donut ──
        st.markdown("---")
        col_donut, col_table = st.columns([1, 2])

        with col_donut:
            st.markdown("#### Win Rate (Per Question)")
            q_wins_g, q_wins_n, q_ties = 0, 0, 0
            for d in ragas_data:
                n_avg = sum(d["naive"][m]["score"] for m in METRICS if m in d["naive"] and d["naive"][m]["score"] > 0)
                g_avg = sum(d["graphrag"][m]["score"] for m in METRICS if m in d["graphrag"] and d["graphrag"][m]["score"] > 0)
                if g_avg > n_avg:
                    q_wins_g += 1
                elif n_avg > g_avg:
                    q_wins_n += 1
                else:
                    q_ties += 1

            fig_donut = go.Figure(data=[go.Pie(
                labels=["GraphRAG Wins", "Naive RAG Wins", "Ties"],
                values=[q_wins_g, q_wins_n, q_ties],
                marker=dict(colors=["#c2a45a", "#6fae8e", "#332e24"]),
                hole=0.55, textinfo="label+value",
                textfont=dict(color="#ece6d8", size=11),
            )])
            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a89e8b"), height=300, showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                annotations=[dict(text=f"<b>{q_wins_g}/{len(ragas_data)}</b>", x=0.5, y=0.5, font_size=22, font_color="#ddbe77", showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Per-Question Breakdown Table ──
        with col_table:
            st.markdown("#### Per-Question Scores")
            table_rows = []
            for i, d in enumerate(ragas_data, 1):
                n_vals = [d["naive"][m]["score"] for m in METRICS if m in d["naive"] and d["naive"][m]["score"] > 0]
                g_vals = [d["graphrag"][m]["score"] for m in METRICS if m in d["graphrag"] and d["graphrag"][m]["score"] > 0]
                n_avg = round(sum(n_vals)/len(n_vals), 2) if n_vals else 0
                g_avg = round(sum(g_vals)/len(g_vals), 2) if g_vals else 0
                winner = "🟣" if g_avg > n_avg else "🟢" if n_avg > g_avg else "🟡"
                table_rows.append({
                    "#": i, "Question": d["question"][:55] + "...",
                    "Category": d["category"].replace("_", " ").title()[:18],
                    "Naive": n_avg, "Graph": g_avg, "W": winner,
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=350, hide_index=True)

        # ── Latency Comparison ──
        st.markdown("---")
        st.markdown("#### Latency Comparison")

        naive_lats = [d["naive"].get("latency_ms", 0) for d in ragas_data]
        graph_lats = [d["graphrag"].get("latency_ms", 0) for d in ragas_data]

        lat_col1, lat_col2, lat_col3 = st.columns(3)
        avg_n_lat = sum(naive_lats)/len(naive_lats)/1000 if naive_lats else 0
        avg_g_lat = sum(graph_lats)/len(graph_lats)/1000 if graph_lats else 0
        lat_col1.metric("Naive RAG Avg", f"{avg_n_lat:.1f}s")
        lat_col2.metric("GraphRAG Avg", f"{avg_g_lat:.1f}s")
        lat_col3.metric("Speedup Factor", f"{avg_g_lat/avg_n_lat:.1f}x" if avg_n_lat > 0 else "N/A")

        q_labels = [f"Q{i+1}" for i in range(len(ragas_data))]
        fig_lat = go.Figure(data=[
            go.Scatter(x=q_labels, y=[l/1000 for l in naive_lats], mode="lines+markers",
                       name="Naive RAG", line=dict(color="#6fae8e", width=2), marker=dict(size=6)),
            go.Scatter(x=q_labels, y=[l/1000 for l in graph_lats], mode="lines+markers",
                       name="GraphRAG", line=dict(color="#c2a45a", width=2), marker=dict(size=6)),
        ])
        fig_lat.update_layout(
            height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a89e8b"), yaxis=dict(title="Seconds", gridcolor="rgba(194,164,90,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # ── Methodology ──
        st.markdown("---")
        st.markdown("#### Methodology")
        meth_col1, meth_col2 = st.columns(2)
        with meth_col1:
            st.markdown('''
            <div class="glass-card">
                <strong>Evaluation Framework</strong><br>
                <span style="color:#cfc6b5;">RAGAS (Retrieval Augmented Generation Assessment)</span><br><br>
                <strong>Judge Model</strong><br>
                <span style="color:#cfc6b5;">Llama 3.1 8B — LLM-as-Judge, temperature=0</span><br><br>
                <strong>Hardware</strong><br>
                <span style="color:#cfc6b5;">RTX 4050 (6GB VRAM), 100% local inference</span>
            </div>
            ''', unsafe_allow_html=True)
        with meth_col2:
            st.markdown('''
            <div class="glass-card">
                <strong>Metrics (8 total)</strong><br>
                <span style="color:#6ee7b7;">1.</span> Faithfulness · 
                <span style="color:#6ee7b7;">2.</span> Answer Relevancy<br>
                <span style="color:#6ee7b7;">3.</span> Completeness · 
                <span style="color:#6ee7b7;">4.</span> Hallucination Detection<br>
                <span style="color:#6ee7b7;">5.</span> Coherence · 
                <span style="color:#6ee7b7;">6.</span> Citation Accuracy<br>
                <span style="color:#6ee7b7;">7.</span> Legal Reasoning · 
                <span style="color:#6ee7b7;">8.</span> Context Precision<br><br>
                <strong>Scoring:</strong> <span style="color:#cfc6b5;">1–5 scale (5 = best)</span>
            </div>
            ''', unsafe_allow_html=True)
