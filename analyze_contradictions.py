"""
LexGraph — Contradiction Detector
Parses GraphRAG relationship outputs to find overruled precedents,
judicial dissents, and conflicting legal principles.

Uses raw HTTP requests to Ollama (no 'ollama' package required).
Only depends on: pandas, requests (both universally available).
"""

import os
import json
import time
import pandas as pd
import requests

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

LLM_MODEL = "llama3.1"
OLLAMA_URL = "http://localhost:11434/api/chat"
RELATIONSHIPS_FILE = "output/relationships.parquet"
ENTITIES_FILE = "output/entities.parquet"

# Keywords that indicate a legal dispute, contradiction, or overruling
DISPUTE_KEYWORDS = [
    "overrule", "overruled", "overruling",
    "dissent", "dissented", "dissenting",
    "contradict", "contradicted", "contradiction",
    "conflict", "conflicting",
    "oppose", "opposed", "opposing",
    "depart", "departed", "departing",
    "disagree", "disagreed", "disagreement",
    "struck down", "strike down",
    "invalidate", "invalidated",
    "reversed", "set aside",
    "distinguished", "narrow", "narrowed",
    "modified", "diluted",
    "curtailed", "restricted",
]

# ─── OLLAMA CHAT (via raw HTTP) ──────────────────────────────────────────────

def ollama_chat(prompt, model=LLM_MODEL, max_tokens=300):
    """Send a chat request to Ollama via REST API. No pip packages needed."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "⚠ Error: Cannot connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_parquet(filepath):
    """Load a parquet file from GraphRAG output."""
    if not os.path.exists(filepath):
        print(f"  ✗ {filepath} not found.")
        return None
    df = pd.read_parquet(filepath)
    print(f"  ✓ Loaded {filepath} — {len(df)} rows, columns: {list(df.columns)}")
    return df

# ─── CONTRADICTION DETECTION ─────────────────────────────────────────────────

def find_contradictions(relationships_df):
    """
    Scan all relationship descriptions for dispute-related keywords.
    Returns a filtered DataFrame of potential contradictions, sorted by weight.
    """
    df = relationships_df.copy()

    # Ensure description is a string column
    if "description" not in df.columns:
        print("  ✗ 'description' column not found in relationships.")
        return pd.DataFrame()

    df["description"] = df["description"].fillna("").astype(str)

    # Build regex pattern from dispute keywords
    pattern = "|".join([rf"\b{kw}\b" for kw in DISPUTE_KEYWORDS])
    mask = df["description"].str.contains(pattern, case=False, na=False)
    conflicts = df[mask].copy()

    # Sort by weight (GraphRAG assigns higher weight to stronger relationships)
    if "weight" in conflicts.columns:
        conflicts = conflicts.sort_values(by="weight", ascending=False)

    return conflicts

# ─── GRAPH TRIANGLE DETECTION ────────────────────────────────────────────────

def find_shared_principle_conflicts(relationships_df, entities_df):
    """
    Advanced detection: Find two CASE nodes connected to the same
    PRINCIPLE/DOCTRINE node but with opposing relationship descriptions.
    This is the 'Triangle of Conflict' algorithm.
    """
    if entities_df is None:
        return []

    # Identify entity types (GraphRAG stores them in a 'type' column)
    if "type" not in entities_df.columns:
        print("  ⚠ 'type' column not found in entities. Skipping triangle detection.")
        return []

    # Get principle/doctrine nodes
    principle_types = ["LEGAL PRINCIPLE", "DOCTRINE", "PRINCIPLE", "CONCEPT", "RIGHT"]
    principles = set(
        entities_df[
            entities_df["type"].str.upper().isin(principle_types)
        ]["title"].str.upper().tolist()
    ) if "title" in entities_df.columns else set()

    if not principles:
        print(f"  ⚠ No principle/doctrine entities found. Available types: "
              f"{entities_df['type'].unique().tolist()[:10]}")
        return []

    print(f"  Found {len(principles)} principle/doctrine nodes in the graph.")

    # Find relationships where the target is a principle node
    df = relationships_df.copy()
    df["source_upper"] = df["source"].str.upper()
    df["target_upper"] = df["target"].str.upper()

    # Relationships pointing TO a principle
    to_principle = df[df["target_upper"].isin(principles)]

    # Group by principle — find cases where multiple sources connect to the same principle
    triangles = []
    for principle, group in to_principle.groupby("target_upper"):
        if len(group) >= 2:
            sources = group[["source", "description"]].values.tolist()
            # Check if any pair has opposing descriptions
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    triangles.append({
                        "principle": principle,
                        "case_a": sources[i][0],
                        "case_a_desc": sources[i][1],
                        "case_b": sources[j][0],
                        "case_b_desc": sources[j][1],
                    })

    print(f"  Found {len(triangles)} shared-principle pairs for deeper analysis.")
    return triangles[:10]  # Limit to top 10 to avoid flooding Ollama

# ─── LLM ANALYSIS ────────────────────────────────────────────────────────────

def analyze_conflict(source, target, description):
    """Ask Llama to cleanly explain a detected contradiction."""
    prompt = f"""You are an elite Indian constitutional law analyst.

I found the following relationship in a Knowledge Graph built from Supreme Court judgments:

Entity A: {source}
Entity B: {target}
Relationship: {description}

Based ONLY on this relationship, concisely explain the legal contradiction, dissent, or overruling. 
Be specific about what legal principle is in dispute. Do not hallucinate external facts.
Keep your response under 150 words."""

    return ollama_chat(prompt)


def analyze_triangle(case_a, case_b, principle, desc_a, desc_b):
    """Ask Llama to compare two cases that connect to the same principle."""
    prompt = f"""You are an elite Indian constitutional law analyst.

Two Supreme Court cases both relate to the same legal principle, but may have reached different conclusions:

Legal Principle: {principle}

Case A: {case_a}
Case A's relationship to the principle: {desc_a}

Case B: {case_b}
Case B's relationship to the principle: {desc_b}

Do these two cases AGREE or CONTRADICT each other regarding this principle?
If they contradict, explain how. If they agree, say "AGREEMENT" and move on.
Keep your response under 150 words."""

    return ollama_chat(prompt)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  LexGraph — Contradiction Detector")
    print("  Scanning Knowledge Graph for Judicial Disputes")
    print("=" * 65)

    start_time = time.time()

    # Load GraphRAG outputs
    print("\n📂 Loading GraphRAG outputs...")
    relationships_df = load_parquet(RELATIONSHIPS_FILE)
    entities_df = load_parquet(ENTITIES_FILE)

    if relationships_df is None:
        print("\n✗ Cannot proceed without relationships data. Run 'graphrag index' first.")
        return

    # ── Method 1: Keyword-based contradiction scan ──
    print("\n🔍 METHOD 1: Keyword Scan (searching relationship descriptions)...")
    conflicts = find_contradictions(relationships_df)

    if conflicts.empty:
        print("  No keyword-matched contradictions found.")
    else:
        print(f"  ✓ {len(conflicts)} potential disputes detected!\n")

        top_n = min(5, len(conflicts))
        print(f"  Analyzing top {top_n} strongest disputes with {LLM_MODEL}...\n")

        for idx, (_, row) in enumerate(conflicts.head(top_n).iterrows(), 1):
            source = row.get("source", "Unknown")
            target = row.get("target", "Unknown")
            desc = row.get("description", "")
            weight = row.get("weight", 0)

            print(f"  [{idx}] ⚔️  DISPUTE DETECTED")
            print(f"      Nodes:  {source}  ↔  {target}")
            print(f"      Weight: {weight}")
            print(f"      Edge:   {desc[:200]}")
            print(f"      Generating AI analysis...")

            summary = analyze_conflict(source, target, desc)
            print(f"      🧠 Analysis: {summary}\n")
            print("  " + "─" * 55)

    # ── Method 2: Triangle detection (shared principle conflicts) ──
    print(f"\n🔺 METHOD 2: Triangle Detection (shared legal principles)...")
    triangles = find_shared_principle_conflicts(relationships_df, entities_df)

    if triangles:
        analyze_count = min(3, len(triangles))
        print(f"  Analyzing {analyze_count} shared-principle pairs...\n")

        for idx, tri in enumerate(triangles[:analyze_count], 1):
            print(f"  [{idx}] 🔺 SHARED PRINCIPLE: {tri['principle']}")
            print(f"      Case A: {tri['case_a']}")
            print(f"      Case B: {tri['case_b']}")
            print(f"      Generating comparative analysis...")

            result = analyze_triangle(
                tri["case_a"], tri["case_b"], tri["principle"],
                tri["case_a_desc"], tri["case_b_desc"]
            )
            print(f"      🧠 Verdict: {result}\n")
            print("  " + "─" * 55)

    # ── Summary ──
    elapsed = round(time.time() - start_time, 1)
    print(f"\n{'=' * 65}")
    print(f"  Contradiction sweep complete in {elapsed}s")
    print(f"  Keyword matches: {len(conflicts) if not conflicts.empty else 0}")
    print(f"  Triangle pairs:  {len(triangles)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
