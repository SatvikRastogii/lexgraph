"""
Naive RAG Pipeline - Built from Scratch
No LangChain. Pure Python.
For benchmarking against GraphRAG on Indian Supreme Court judgments.
"""

import os
import json
import re
import time
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CORPUS_DIR = "legal_corpus"           # your 230 judgment files
CHROMA_DIR = "chroma_db"              # where vectors are stored
COLLECTION_NAME = "legal_judgments"
EMBEDDING_MODEL = "nomic-embed-text"  # same model as GraphRAG
LLM_MODEL = "llama3.1"
CHUNK_SIZE = 500                      # tokens approx, same as GraphRAG
CHUNK_OVERLAP = 50                  # overlap between chunks

# ─── STEP 1: CHUNKER ──────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    Uses word boundaries — never cuts mid-word.
    Same chunk size as GraphRAG for fair comparison.
    """
    # Split into words
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        # Move forward with overlap
        start += chunk_size - overlap

        # Stop if we've covered everything
        if end >= len(words):
            break

    return chunks

def load_documents(corpus_dir):
    """
    Load all judgment files from corpus directory.
    Returns list of (filename, text) tuples.
    """
    documents = []
    files = sorted([f for f in os.listdir(corpus_dir) if f.endswith('.txt')])

    print(f"Loading {len(files)} documents from {corpus_dir}...")

    # Keywords corresponding to the constitutional cases requested to filter the dataset
    target_keywords = [
        "maneka gandhi", "francis coralie mullin", "olga tellis", "unni krishnan",
        "pucl", "people's union for civil liberties", "peoples union for civil liberties",
        "vishaka", "paschim banga khet mazdoor", "mc mehta", "m.c. mehta", "m c mehta",
        "parmanand katara", "dk basu", "d.k. basu", "selvi", "puttaswamy", "royappa",
        "nargesh meerza", "nakara", "ajay hasia", "indra sawhney", "anuj garg",
        "navtej singh johar", "romesh thappar", "brij bhushan", "v.g. row", "vg row",
        "sakal papers", "bennett coleman", "indian express newspapers", "shreya singhal",
        "anuradha bhasin", "bandhua mukti morcha", "aruna shanbaug", "kesavananda bharati",
        "minerva mills", "s.p. gupta", "sp gupta", "s.r. bommai", "sr bommai", 
        "i.r. coelho", "ir coelho", "kihoto hollohan"
    ]

    for filename in files:
        filepath = os.path.join(corpus_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            if len(text) > 500:  # skip empty files
                # Only include documents that are in our target list
                intro_text = text[:1000].lower()
                is_target = any(keyword in intro_text for keyword in target_keywords)
                
                if is_target:
                    documents.append((filename, text))
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"Successfully loaded {len(documents)} matching documents out of {len(files)}")
    return documents

# ─── STEP 2: EMBEDDER + VECTOR STORE ──────────────────────────────────────────

def build_vector_store(corpus_dir, chroma_dir, collection_name):
    """
    Build ChromaDB vector store from corpus.
    Chunks documents, embeds them, stores vectors.
    This is the naive RAG index — equivalent to GraphRAG's indexing phase.
    """
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)

    # Check if collection already exists
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"Vector store already exists at {chroma_dir}")
        print("Loading existing store...")
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.OllamaEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                url="http://localhost:11434"
            )
        )
        print(f"Loaded {collection.count()} chunks")
        return collection

    print("Building vector store from scratch...")
    print(f"Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")

    # Create embedding function using same model as GraphRAG
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        url="http://localhost:11434"
    )

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    # Load and chunk documents
    documents = load_documents(corpus_dir)

    total_chunks = 0
    batch_ids = []
    batch_texts = []
    batch_metadata = []
    batch_size = 50  # process in batches to avoid memory issues

    for doc_idx, (filename, text) in enumerate(documents):
        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}__chunk_{chunk_idx:04d}"

            # Extract basic metadata from chunk
            articles = re.findall(r'Article\s+(\d+[A-Z]?)', chunk)
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', chunk)

            batch_ids.append(chunk_id)
            batch_texts.append(chunk)
            batch_metadata.append({
                "source": filename,
                "chunk_index": chunk_idx,
                "articles_mentioned": ", ".join(set(articles[:5])),
                "year": year_match.group(1) if year_match else "unknown",
                "chunk_length": len(chunk)
            })
            total_chunks += 1

            # Process batch
            if len(batch_ids) >= batch_size:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadata
                )
                print(f"  Indexed {total_chunks} chunks "
                      f"({doc_idx+1}/{len(documents)} documents)...")
                batch_ids, batch_texts, batch_metadata = [], [], []

    # Process remaining chunks
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata
        )

    print(f"\nVector store built successfully!")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Average chunks per document: {total_chunks // len(documents)}")

    return collection

# ─── STEP 3: RETRIEVER ────────────────────────────────────────────────────────

def retrieve(collection, query, top_k=5):
    """
    Retrieve most relevant chunks for a query.
    Uses cosine similarity between query embedding and chunk embeddings.
    This is the core of naive RAG — find similar text, return it.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        retrieved.append({
            "text": chunk,
            "source": meta.get("source", "unknown"),
            "similarity": round(1 - dist, 4),  # convert distance to similarity
            "articles": meta.get("articles_mentioned", ""),
            "year": meta.get("year", "unknown")
        })

    return retrieved

# ─── STEP 4: GENERATOR ────────────────────────────────────────────────────────

def generate_answer(query, retrieved_chunks):
    """
    Generate answer using retrieved chunks as context.
    Sends query + relevant chunks to local LLM.
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']} | "
            f"Similarity: {chunk['similarity']} | "
            f"Year: {chunk['year']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Prompt designed for legal reasoning
    prompt = f"""You are a legal research assistant specializing in Indian constitutional law.
Answer the question based ONLY on the provided court judgment excerpts.
Be specific, cite the sources, and acknowledge if the context is insufficient.

QUESTION: {query}

RELEVANT JUDGMENT EXCERPTS:
{context}

ANSWER:"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 1000}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating answer: {e}"

# ─── STEP 5: FULL RAG PIPELINE ────────────────────────────────────────────────

def naive_rag_query(collection, query, top_k=5):
    """
    Complete naive RAG pipeline:
    Query -> Retrieve relevant chunks -> Generate answer
    Returns answer + retrieved context for RAGAS evaluation
    """
    start_time = time.time()

    # Retrieve
    retrieved = retrieve(collection, query, top_k)

    # Generate
    answer = generate_answer(query, retrieved)

    elapsed = round(time.time() - start_time, 2)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "contexts": [r["text"] for r in retrieved],
        "sources": [r["source"] for r in retrieved],
        "latency_seconds": elapsed
    }

# ─── STEP 6: BATCH EVALUATION ─────────────────────────────────────────────────

def run_benchmark(collection, questions_file="benchmark_questions.json"):
    """
    Run all benchmark questions through naive RAG.
    Saves results for RAGAS evaluation and comparison with GraphRAG.
    """
    if not os.path.exists(questions_file):
        print(f"Questions file not found: {questions_file}")
        return

    with open(questions_file, "r") as f:
        questions = json.load(f)

    results = {}
    total = sum(len(q) for q in questions.values())
    count = 0

    print(f"\nRunning benchmark: {total} questions across {len(questions)} categories")
    print("This will take 20-30 minutes...\n")

    for category, category_questions in questions.items():
        print(f"\nCategory: {category}")
        results[category] = []

        for question in category_questions:
            count += 1
            print(f"  [{count}/{total}] {question[:60]}...")

            result = naive_rag_query(collection, question)
            result["category"] = category
            results[category].append(result)

            print(f"  Latency: {result['latency_seconds']}s")

    # Save results
    output_file = "naive_rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark complete. Results saved to {output_file}")
    return results

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Naive RAG Pipeline")
    print("Indian Supreme Court Judgments")
    print("=" * 60)

    # Build or load vector store
    collection = build_vector_store(CORPUS_DIR, CHROMA_DIR, COLLECTION_NAME)

    # Interactive query mode
    print("\nNaive RAG ready. Type your questions.")
    print("Commands: 'benchmark' to run full eval, 'quit' to exit\n")

    while True:
        query = input("Question: ").strip()

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "benchmark":
            run_benchmark(collection)
            continue

        result = naive_rag_query(collection, query)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for chunk in result["retrieved_chunks"]:
            print(f"  - {chunk['source']} "
                  f"(similarity: {chunk['similarity']}, "
                  f"year: {chunk['year']})")
        print(f"\nLatency: {result['latency_seconds']}s")
        print("-" * 60)

if __name__ == "__main__":
    main()