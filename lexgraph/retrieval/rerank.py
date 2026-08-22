"""Cross-encoder reranking.

Bi-encoders (the dense retriever) embed query and passage independently, so
they can only ever measure how close two summaries of meaning are. A cross
-encoder reads the query and passage together and scores the pair directly,
which is markedly better at judging relevance -- and far too slow to run over
a whole corpus. The standard arrangement, used here, is to let cheap retrieval
propose a shortlist and let the cross-encoder reorder it.

flashrank is used rather than sentence-transformers: it is an ONNX runtime
model of about 20MB that scores on CPU in tens of milliseconds, against
roughly 2GB of torch for an equivalent result. The model is downloaded once
and cached locally.
"""

from __future__ import annotations

from .base import Hit

DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"


class CrossEncoderReranker:
    """Reorders a shortlist of hits by cross-encoder relevance score.

    The underlying model is loaded lazily so that importing this module -- and
    therefore running the test suite -- never touches the network.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker = None

    @property
    def ranker(self):
        if self._ranker is None:
            from flashrank import Ranker

            kwargs = {"model_name": self.model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            self._ranker = Ranker(**kwargs)
        return self._ranker

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        """Return ``hits`` reordered by cross-encoder score, best first."""
        if not hits:
            return []

        from flashrank import RerankRequest

        passages = [
            {"id": index, "text": hit.text, "meta": {}}
            for index, hit in enumerate(hits)
        ]
        ranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))

        reordered = []
        for entry in ranked:
            hit = hits[int(entry["id"])]
            score = float(entry["score"])
            reordered.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    score=score,
                    title=hit.title,
                    year=hit.year,
                    para_label=hit.para_label,
                    components={**hit.components, "rerank": score},
                )
            )
        return reordered[:top_k] if top_k else reordered
