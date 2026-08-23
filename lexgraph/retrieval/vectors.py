"""A vector store that is one file and no server.

ChromaDB is right for the local workflow: it persists incrementally, survives
rebuilds, and holds two collections at once. For deployment it is 84MB of
gitignored state built by an embedder that needs a GPU, which makes it the
single thing standing between this project and a public URL.

At 1,384 chunks the index is not doing any work worth 84MB. A dense matrix of
1,384 x 768 float16 is 2MB, brute-force cosine over it takes about a
millisecond, and the result is exact rather than approximate. It commits to
git, loads with no server, and cannot disagree with the vectors it was built
from -- which is precisely the failure that broke GraphRAG's local search on
this same corpus.

The store records which embedder produced it and refuses to load under a
different one. Querying nomic-embed-text vectors with a Gemini query embedding
does not error; it returns confident nonsense, ranked, with no indication
anything is wrong. That is exactly the class of silent failure this repository
exists to catch, so it is made loud here.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import Hit

DEFAULT_STORE = os.path.join("data", "vectors")


class VectorStoreMismatch(RuntimeError):
    """The store was built by a different embedder than the one querying it."""


def store_path(directory: str, strategy: str) -> str:
    return os.path.join(directory, f"{strategy}.npz")


def save_store(
    path: str,
    chunks,
    vectors,
    embedder_model: str,
    dimensions: int,
) -> None:
    """Write vectors plus the metadata needed to rebuild a Hit."""
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        # float16 halves the committed size. The vectors are already
        # normalised and cosine similarity is compared, not reported, so three
        # decimal places of precision is far more than the ranking needs.
        matrix=matrix.astype(np.float16),
        metadata=np.array(
            [
                json.dumps({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.indexed_text(),
                    "title": c.title,
                    "year": c.year,
                    "para_label": c.para_label,
                })
                for c in chunks
            ],
            dtype=object,
        ),
        manifest=np.array(
            [json.dumps({
                "embedder": embedder_model,
                "dimensions": dimensions,
                "chunks": len(chunks),
            })],
            dtype=object,
        ),
    )


class NumpyDenseRetriever:
    """Exact dense retrieval over a committed matrix."""

    def __init__(self, path: str, embedder):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run `python scripts/build_deploy_index.py`"
            )
        stored = np.load(path, allow_pickle=True)
        self.manifest = json.loads(stored["manifest"][0])

        if self.manifest["embedder"] != embedder.model:
            raise VectorStoreMismatch(
                f"{path} was built with {self.manifest['embedder']!r} but the "
                f"query embedder is {embedder.model!r}. Mixing them returns "
                f"plausible, wrongly ranked results rather than an error, so "
                f"the store refuses to load. Rebuild it or switch embedder."
            )

        self.embedder = embedder
        self.matrix = stored["matrix"].astype(np.float32)
        self.records = [json.loads(m) for m in stored["metadata"]]

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        vector = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        scores = self.matrix @ vector

        hits = []
        for rank in np.argsort(-scores)[:top_k]:
            record = self.records[rank]
            hits.append(
                Hit(
                    score=float(scores[rank]),
                    components={"dense": float(scores[rank])},
                    **record,
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self.records)
