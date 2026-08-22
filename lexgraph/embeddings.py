"""Ollama embedding client.

ChromaDB ships an ``OllamaEmbeddingFunction``, but it issues one HTTP request
per ``add`` batch with a short fixed timeout and no retry. On a 6GB card a
batch of 64 judgment chunks reliably exceeds it, and the failure surfaces as an
``httpx.ReadTimeout`` from deep inside ``collection.add`` *after* the
collection has already been created -- leaving an empty collection behind that
looks successfully built.

Embedding here instead keeps the timeout, batch size and retry policy explicit,
and lets the same vectors be reused by the BM25/rerank path without a second
round trip.
"""

from __future__ import annotations

import os
import time

import requests

DEFAULT_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an unusable response."""


class OllamaEmbedder:
    """Batched embedding with a generous timeout and bounded retries."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        batch_size: int = 16,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.host = (host or ollama_host()).rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, inputs: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.host}/api/embed",
                    json={"model": self.model, "input": inputs},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                embeddings = response.json().get("embeddings")
                if not embeddings or len(embeddings) != len(inputs):
                    raise EmbeddingError(
                        f"expected {len(inputs)} embeddings, got "
                        f"{len(embeddings) if embeddings else 0}"
                    )
                return embeddings
            except Exception as error:  # noqa: BLE001 - retried and re-raised below
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise EmbeddingError(
            f"embedding failed after {self.max_retries} attempts against "
            f"{self.host}: {last_error}"
        ) from last_error

    def embed(self, texts: list[str], progress=None) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving input order."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._post(batch))
            if progress:
                progress(min(start + self.batch_size, len(texts)), len(texts))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self._post([text])[0]

    def health_check(self) -> None:
        """Fail fast with a readable message if Ollama or the model is missing."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except Exception as error:  # noqa: BLE001
            raise EmbeddingError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from error

        available = {m["name"].split(":")[0] for m in response.json().get("models", [])}
        if self.model.split(":")[0] not in available:
            raise EmbeddingError(
                f"Model {self.model!r} not found on {self.host}. "
                f"Run: ollama pull {self.model}"
            )
