"""Score GraphRAG's own retrieval on the gold set.

Until now GraphRAG was compared on the answer side only, so the interesting
question -- does a knowledge graph find the right judgment more often than a
vector store -- had never been asked in a form that could be answered. This
puts its retrieval units on the same gold set, the same metrics and the same
embedding model as every other configuration, so the only thing that varies is
what a retrieval unit *is*.

There are two of them, and they correspond to what GraphRAG's own search
methods read:

``graph-units``
    The 1,556 text units GraphRAG chunked the corpus into. This is the honest
    control rather than the interesting arm: it is a different chunking of the
    same text, with no graph structure involved. If the graph is doing work,
    the other arm should beat this one.

``graph-community``
    The 184 community reports -- LLM-written summaries of clustered entities
    and relationships, which is where the graph actually lives and what global
    search fans out across. A report is not tied to one judgment, so it
    resolves to the set of documents its member text units came from.

That last detail is a measurement hazard, not a footnote. A level-0 report can
span twenty judgments, so a single retrieved report fills twenty document
slots and recall@k stops meaning what it means for a chunk retriever.
``breadth`` reports the mean documents per hit so the inflation is visible
rather than quietly banked; read the community numbers with it in hand.

No re-indexing is involved. GraphRAG's expensive phase is entity and
relationship extraction, and its artefacts are already on disk -- 3,750
entities, 1,505 relationships, 184 reports. What was missing is only the
embeddings, which is why ``local`` search fails: the LanceDB store it reads
was never built. Embedding these units takes minutes, and the vectors are
cached so it happens once.
"""

from __future__ import annotations

import os

import numpy as np

from ..embeddings import build_embedder
from .base import Hit

GRAPH_ROOT = "output"

# Committed, not cached. output/ is gitignored apart from the parquets, so a
# store written there would be rebuilt on every deployment -- which needs an
# embedder the deployment has, and minutes it does not have at boot. These are
# small (184 community reports and 3,750 entities at 384 dimensions is under
# 3MB) and ship with the repository.
VECTOR_CACHE = os.path.join("data", "graph_vectors")

KINDS = {
    "units": ("text_units.parquet", "text"),
    "community": ("community_reports.parquet", "full_content"),
    "entity": ("entities.parquet", "description"),
}


class GraphArtefactsMissing(FileNotFoundError):
    """The GraphRAG index has not been built, or is not where it was expected."""


def _require(path: str):
    if not os.path.exists(path):
        raise GraphArtefactsMissing(
            f"{path} not found -- run `graphrag index --root .` or point "
            f"LEXGRAPH_GRAPH_ROOT at an existing index"
        )
    import pandas as pd

    return pd.read_parquet(path)


def load_units(kind: str, root: str = GRAPH_ROOT) -> list[tuple[str, str, list[str]]]:
    """Return ``(unit_id, text, [document filenames])`` for one retrieval unit.

    Filenames rather than GraphRAG's content-hash ids, because the gold set is
    annotated against ``input/`` and a comparison that needed a translation
    table on one side would be a place for a mismatch to hide.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown graph unit kind: {kind!r}")

    documents = _require(os.path.join(root, "documents.parquet"))
    filename_by_id = dict(zip(documents["id"], documents["title"], strict=True))

    text_units = _require(os.path.join(root, "text_units.parquet"))
    document_by_unit = {
        unit_id: filename_by_id.get(document_id)
        for unit_id, document_id in zip(
            text_units["id"], text_units["document_id"], strict=True
        )
    }

    if kind == "units":
        return [
            (unit_id, text, [document_by_unit[unit_id]])
            for unit_id, text in zip(text_units["id"], text_units["text"], strict=True)
            if document_by_unit.get(unit_id)
        ]

    if kind == "entity":
        # An entity stands for the judgments its mentions came from. This is
        # local search's substrate: the query matches an entity description,
        # and the entity leads back to the passages that produced it.
        entities = _require(os.path.join(root, "entities.parquet"))
        rows = []
        for title, description, unit_ids in zip(
            entities["title"], entities["description"], entities["text_unit_ids"],
            strict=True,
        ):
            filenames = sorted({
                document_by_unit[unit_id]
                for unit_id in unit_ids
                if document_by_unit.get(unit_id)
            })
            if filenames and isinstance(description, str) and description.strip():
                rows.append((f"entity_{title}", f"{title}: {description}", filenames))
        return rows

    communities = _require(os.path.join(root, "communities.parquet"))
    units_by_community = dict(
        zip(communities["community"], communities["text_unit_ids"], strict=True)
    )
    reports = _require(os.path.join(root, "community_reports.parquet"))

    rows = []
    for community, text in zip(reports["community"], reports["full_content"], strict=True):
        members = units_by_community.get(community, [])
        filenames = sorted({
            document_by_unit[unit_id]
            for unit_id in members
            if document_by_unit.get(unit_id)
        })
        if filenames:
            rows.append((f"community_{community}", text, filenames))
    return rows


class GraphRetriever:
    """Dense retrieval over GraphRAG's units, using the same embedder as `dense`.

    Sharing the embedding model is the point. Swapping it as well would leave
    any difference unattributable between the graph and the encoder.
    """

    def __init__(
        self,
        kind: str = "units",
        root: str = GRAPH_ROOT,
        embedder=None,
        cache_dir: str = VECTOR_CACHE,
    ):
        self.kind = kind
        # Follows LEXGRAPH_EMBEDDER like the rest of the stack. Hardcoding
        # OllamaEmbedder here meant the graph retriever ignored the deployment
        # configuration entirely: it would have needed Ollama on a host that
        # has none, and locally it silently re-embedded 3,747 entities with the
        # wrong model rather than reading the committed store.
        self.embedder = embedder or build_embedder()
        rows = load_units(kind, root)
        self.unit_ids = [row[0] for row in rows]
        self.texts = [row[1] for row in rows]
        self.documents = [row[2] for row in rows]
        self.matrix = self._vectors(cache_dir)

    def _vectors(self, cache_dir: str) -> np.ndarray:
        """Embed every unit once and keep the result next to the index."""
        safe_model = self.embedder.model.replace("/", "_")
        path = os.path.join(cache_dir, f"{self.kind}_{safe_model}.npz")
        if os.path.exists(path):
            stored = np.load(path, allow_pickle=True)
            # A stale cache is worse than no cache: it would score a retriever
            # against vectors for text that has since been re-indexed.
            if list(stored["unit_ids"]) == self.unit_ids:
                return stored["matrix"].astype(np.float32)

        vectors = np.asarray(
            self.embedder.embed(self.texts, progress=_progress(self.kind)),
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

        os.makedirs(cache_dir, exist_ok=True)
        # float16 and compressed, because these files are committed. The
        # vectors are normalised and only compared, never reported, so the
        # precision lost is far below what changes a ranking -- and it halves
        # 3,747 entity vectors from 5.6MB to under 3.
        np.savez_compressed(
            path,
            matrix=vectors.astype(np.float16),
            unit_ids=np.array(self.unit_ids, dtype=object),
        )
        return vectors

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        """Rank units, then emit one hit per document each unit stands for.

        A community report is evidence for every judgment it was built from,
        so returning only the first would understate what global search
        surfaces. Expanding also makes the cost visible in the right place: a
        report covering twenty-eight judgments consumes twenty-eight of the
        top-k slots, which is the real behaviour and should not be hidden by
        counting it as one.
        """
        vector = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        scores = self.matrix @ vector

        # Brute force. At 1,556 units this is well under a millisecond, and it
        # removes an index that could disagree with the vectors it was built
        # from -- the failure that broke GraphRAG's own local search.
        hits: list[Hit] = []
        seen: set[str] = set()
        for rank in np.argsort(-scores):
            for filename in self.documents[rank]:
                if filename in seen:
                    continue
                seen.add(filename)
                hits.append(
                    Hit(
                        chunk_id=f"{self.unit_ids[rank]}::{filename}",
                        doc_id=filename,
                        text=self.texts[rank],
                        score=float(scores[rank]),
                        title=filename,
                        components={"unit_documents": len(self.documents[rank])},
                    )
                )
                if len(hits) >= top_k:
                    return hits
        return hits

    @property
    def breadth(self) -> float:
        """Mean documents covered by one retrieval unit.

        1.0 for text units. Well above it for community reports, which is why
        their recall is not directly comparable to a chunk retriever's.
        """
        if not self.documents:
            return 0.0
        return sum(len(d) for d in self.documents) / len(self.documents)


# What each method retrieves over, and how the UI should describe it. The
# wording matters: this is not GraphRAG's own query engine, which drives dozens
# of sequential LLM calls per question and needs ~500MB of dependencies. It is
# retrieval over the artefacts that engine produced, read by one generation
# call -- so it runs on a free host in seconds instead of minutes.
GRAPH_METHODS = {
    "global": (
        "community",
        "Community reports — LLM-written summaries of clustered entities and "
        "relationships. This is what GraphRAG's global search reads.",
    ),
    "local": (
        "entity",
        "Entity descriptions, resolved back to the judgments their mentions "
        "came from. This is what GraphRAG's local search reads.",
    ),
}


def build_graph_retriever(method: str = "global", **kwargs):
    """A retriever for one of GraphRAG's two search substrates."""
    if method not in GRAPH_METHODS:
        raise ValueError(
            f"unknown graph method {method!r}; expected one of {list(GRAPH_METHODS)}"
        )
    return GraphRetriever(kind=GRAPH_METHODS[method][0], **kwargs)


def _progress(kind: str):
    def report(done: int, total: int) -> None:
        if done % 256 == 0 or done == total:
            print(f"  embedded {done}/{total} {kind}")

    return report
