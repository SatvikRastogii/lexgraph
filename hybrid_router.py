"""
Hybrid Semantic Search Router — LexGraph
Routes incoming legal queries to the optimal pipeline:
  - NAIVE RAG: For simple, single-hop factual questions (fast, cheap)
  - GRAPH RAG: For complex, multi-hop relational questions (accurate, expensive)

Uses embedding-based semantic similarity against prototype questions.
No LLM inference required — routing decision takes ~50ms.
"""

import numpy as np
import ollama
import json
import time

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "nomic-embed-text"

# ─── PROTOTYPE QUESTIONS ─────────────────────────────────────────────────────
# These are carefully curated examples that define each routing category.
# The router compares incoming queries against these prototypes using cosine similarity.

SIMPLE_PROTOTYPES = [
    # Single-hop factual questions — Naive RAG handles these perfectly
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
    # Multi-hop relational questions — GraphRAG excels here
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

# ─── EMBEDDING UTILITIES ─────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a single text using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return np.array(response["embeddings"][0])


def get_embeddings_batch(texts: list) -> np.ndarray:
    """Get embedding vectors for a batch of texts using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return np.array(response["embeddings"])


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ─── ROUTER CLASS ─────────────────────────────────────────────────────────────

class HybridSemanticRouter:
    """
    Semantic Embedding Router for LexGraph.
    
    At initialization, pre-embeds prototype questions for both categories.
    At query time, embeds the user's question (~50ms) and computes cosine
    similarity against all prototypes. Routes to the pipeline whose
    prototypes are semantically closest.
    
    This approach is:
    - Faster than LLM classification (~50ms vs ~2000ms)
    - More robust than keyword heuristics (handles rephrased queries)
    - Zero additional GPU compute (embeddings are tiny compared to generation)
    """

    def __init__(self):
        """Pre-embed all prototype questions at initialization."""
        print("Initializing Hybrid Semantic Router...")
        start = time.perf_counter()

        # Pre-embed prototypes
        print("  Embedding NAIVE prototypes...")
        self.simple_embeddings = get_embeddings_batch(SIMPLE_PROTOTYPES)

        print("  Embedding GRAPH prototypes...")
        self.complex_embeddings = get_embeddings_batch(COMPLEX_PROTOTYPES)

        elapsed = round((time.perf_counter() - start) * 1000)
        print(f"  Router initialized in {elapsed}ms")
        print(f"  Prototypes: {len(SIMPLE_PROTOTYPES)} simple, {len(COMPLEX_PROTOTYPES)} complex\n")

    def classify(self, query: str) -> dict:
        """
        Classify a query as NAIVE or GRAPH.
        
        Returns a dict with:
          - route: "NAIVE" or "GRAPH"
          - confidence: float (0.0 to 1.0) — how confident the router is
          - simple_score: average cosine similarity to simple prototypes
          - complex_score: average cosine similarity to complex prototypes
          - latency_ms: time taken for classification
        """
        start = time.perf_counter()

        # Embed the incoming query
        query_embedding = get_embedding(query)

        # Compute similarity against all simple prototypes
        simple_scores = [
            cosine_similarity(query_embedding, proto)
            for proto in self.simple_embeddings
        ]

        # Compute similarity against all complex prototypes
        complex_scores = [
            cosine_similarity(query_embedding, proto)
            for proto in self.complex_embeddings
        ]

        # Use top-3 average (more robust than single max or full average)
        top_k = 3
        avg_simple = float(np.mean(sorted(simple_scores, reverse=True)[:top_k]))
        avg_complex = float(np.mean(sorted(complex_scores, reverse=True)[:top_k]))

        # Route decision
        route = "NAIVE" if avg_simple > avg_complex else "GRAPH"

        # Confidence = how much the winning score dominates
        total = avg_simple + avg_complex
        if total > 0:
            winning_score = max(avg_simple, avg_complex)
            confidence = round((winning_score / total) * 2 - 1, 4)  # normalize to 0-1
        else:
            confidence = 0.0

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "route": route,
            "confidence": confidence,
            "simple_score": round(avg_simple, 4),
            "complex_score": round(avg_complex, 4),
            "latency_ms": elapsed_ms,
        }


# ─── STANDALONE TEST ──────────────────────────────────────────────────────────

def test_router():
    """
    Test the router against all benchmark questions.
    Expected: single_hop_factual → NAIVE, everything else → GRAPH.
    """
    router = HybridSemanticRouter()

    # Load benchmark questions
    with open("benchmark_questions.json", "r") as f:
        benchmarks = json.load(f)

    # Expected routes per category
    expected_routes = {
        "single_hop_factual": "NAIVE",
        "multi_hop_relational": "GRAPH",
        "global_thematic": "GRAPH",
        "cross_document_reasoning": "GRAPH",
        "entity_relationship": "GRAPH",
    }

    total = 0
    correct = 0


    print("=" * 70)
    print("HYBRID SEMANTIC ROUTER — BENCHMARK TEST")
    print("=" * 70)

    for category, questions in benchmarks.items():
        expected = expected_routes.get(category, "GRAPH")
        print(f"\n📂 Category: {category} (expected: {expected})")
        print("-" * 60)

        for q in questions:
            result = router.classify(q)
            total += 1
            is_correct = result["route"] == expected

            if is_correct:
                correct += 1
                icon = " ✅"
            else:
                icon = " ❌"

            print(f"{icon} [{result['route']}] "
                  f"(conf: {result['confidence']:.2f}, "
                  f"S:{result['simple_score']:.3f} vs C:{result['complex_score']:.3f}, "
                  f"{result['latency_ms']}ms)")
            print(f"    Q: {q[:70]}...")

    accuracy = round(correct / total * 100, 1)
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{total} correct ({accuracy}% accuracy)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_router()
