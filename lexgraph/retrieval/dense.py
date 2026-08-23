"""Dense vector retrieval over ChromaDB.

The collection is built from ``input/`` -- the same directory GraphRAG indexes.
Earlier versions read from ``legal_corpus/`` through a keyword filter, which
silently gave the two pipelines disjoint corpora and made every comparison
between them meaningless. Anything that builds an index here goes through
``lexgraph.corpus.load_corpus`` for that reason.

Vectors are computed by ``lexgraph.embeddings.OllamaEmbedder`` and handed to
Chroma directly rather than registering an embedding function, so the timeout
and retry policy stays under our control on both the write and the query path.

ChromaDB is imported lazily, not at module scope. The deployed build has no
ChromaDB at all -- it reads a committed matrix through
``lexgraph.retrieval.vectors`` instead -- and a module-level import made simply
importing the pipeline fail there, before any code had chosen a backend. Same
reason the reranker defers flashrank: importing a module should not require
every optional dependency any of its classes might use.
"""

from __future__ import annotations

from ..chunking import Chunk
from ..embeddings import DEFAULT_MODEL, OllamaEmbedder
from .base import Hit

DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_BATCH_SIZE = 32


def collection_name(strategy: str) -> str:
    """One collection per chunking strategy, so the ablation can compare them."""
    return f"judgments_{strategy}"


class DenseRetriever:
    """Cosine-similarity search over an existing Chroma collection."""

    def __init__(
        self,
        strategy: str = "paragraph",
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        embedding_model: str = DEFAULT_MODEL,
        embedder: OllamaEmbedder | None = None,
    ):
        import chromadb

        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(name=collection_name(strategy))
        self.embedder = embedder or OllamaEmbedder(model=embedding_model)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        results = self.collection.query(
            query_embeddings=[self.embedder.embed_one(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for chunk_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            similarity = 1.0 - distance
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", "unknown"),
                    text=text,
                    score=similarity,
                    title=meta.get("title", ""),
                    year=meta.get("year", "unknown"),
                    para_label=meta.get("para_label", ""),
                    components={"dense": similarity},
                )
            )
        return hits


def build_collection(
    chunks: list[Chunk],
    strategy: str,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    embedding_model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace: bool = True,
    progress=print,
):
    """Embed ``chunks`` into a Chroma collection named after ``strategy``.

    Chunks are indexed as ``Chunk.indexed_text()`` -- body text prefixed with
    case name, year and paragraph span -- so a mid-judgment passage remains
    findable by the case it belongs to.

    Embeddings are computed up front. If Ollama fails partway the collection is
    never created, rather than being left empty but present.
    """
    embedder = OllamaEmbedder(model=embedding_model, batch_size=batch_size)
    embedder.health_check()

    texts = [chunk.indexed_text() for chunk in chunks]
    vectors = embedder.embed(
        texts,
        progress=lambda done, total: progress(f"  embedded {done}/{total} chunks"),
    )

    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    name = collection_name(strategy)
    if replace and name in {c.name for c in client.list_collections()}:
        progress(f"  replacing existing collection {name!r}")
        client.delete_collection(name)

    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), 256):
        window = slice(start, start + 256)
        batch = chunks[window]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts[window],
            embeddings=vectors[window],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "year": chunk.year,
                    "article_focus": chunk.article_focus,
                    "para_label": chunk.para_label,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in batch
            ],
        )

    if collection.count() != len(chunks):
        raise RuntimeError(
            f"collection {name!r} holds {collection.count()} chunks, expected {len(chunks)}"
        )
    return collection
