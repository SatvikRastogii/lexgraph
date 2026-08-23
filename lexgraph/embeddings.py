"""Embedding clients.

``OllamaEmbedder`` is what everything local uses. ``GeminiEmbedder`` exists for
deployment: no free host offers a GPU, so the model that produces query
embeddings has to live behind an API there. They present the same three
methods so the retrieval stack does not know or care which one it has.

Ollama notes follow, because they are the reason this module exists at all.

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
        self._query_cache: dict[str, list[float]] = {}

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
        """Embed a single text, memoised per process.

        Measured on this hardware, an Ollama embed request costs ~2.15s of
        fixed overhead regardless of payload: batches of 1, 2 and 8 all take
        the same wall time, while a batch of 32 amortises to ~94ms per item.
        So the per-request constant, not the model, dominates single-query
        retrieval.

        The evaluation sweeps re-embed the same gold-set questions once per
        configuration, which is pure waste against a constant that large. The
        cache is on the query path only; indexing already batches.
        """
        cached = self._query_cache.get(text)
        if cached is None:
            cached = self._post([text])[0]
            self._query_cache[text] = cached
        return cached

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


GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

# gemini-embedding-001 returns 3072 dimensions by default. Truncating to 768
# keeps the committed vector store at 2MB instead of 8MB and matches the
# dimensionality the local nomic-embed-text index already uses, so the two are
# at least shaped alike when their numbers are compared.
GEMINI_DIMENSIONS = 768

# The free tier meters requests per minute. Embedding is one request per batch
# during a build and one per query at runtime, so the build is what needs pacing.
GEMINI_BATCH = 32


class GeminiEmbedder:
    """Hosted embedding, for deployments with no GPU to run a local model on.

    Interchangeable with ``OllamaEmbedder``: same ``embed``/``embed_one``/
    ``health_check``, same normalisation left to the caller, same per-process
    query memoisation. Anything that takes an embedder takes either.

    Vectors from the two are **not** comparable. A store built with one and
    queried with the other returns confident nonsense rather than an error,
    which is why the vector store records the model that produced it and
    refuses to load under a different one.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = GEMINI_EMBED_MODEL,
        api_key: str | None = None,
        dimensions: int = GEMINI_DIMENSIONS,
        batch_size: int = GEMINI_BATCH,
        timeout: float = 120.0,
        max_retries: int = 4,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._query_cache: dict[str, list[float]] = {}

        self.api_key = api_key or _google_api_key()
        if not self.api_key:
            raise EmbeddingError(
                "No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in the "
                "environment, .env, or the host's secrets store."
            )

    def _request(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{GEMINI_ENDPOINT}/models/{self.model}:{path}",
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    # Per-minute limit: waiting is exactly the fix.
                    time.sleep(15 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as error:  # noqa: BLE001 - retried, then re-raised
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        raise EmbeddingError(
            f"Gemini embedding failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def embed(self, texts: list[str], progress=None) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": text}]},
                        "outputDimensionality": self.dimensions,
                    }
                    for text in batch
                ]
            }
            body = self._request("batchEmbedContents", payload)
            embeddings = [e["values"] for e in body.get("embeddings", [])]
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"expected {len(batch)} embeddings, got {len(embeddings)}"
                )
            vectors.extend(embeddings)
            if progress:
                progress(min(start + self.batch_size, len(texts)), len(texts))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is None:
            body = self._request(
                "embedContent",
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimensions,
                },
            )
            cached = body["embedding"]["values"]
            self._query_cache[text] = cached
        return cached

    def health_check(self) -> None:
        vector = self.embed_one("Article 21 protects life and personal liberty.")
        if len(vector) != self.dimensions:
            raise EmbeddingError(
                f"{self.model} returned {len(vector)} dimensions, expected "
                f"{self.dimensions}"
            )


def _google_api_key() -> str | None:
    """The key from the environment, .env, or Streamlit's secrets store.

    Three places because the deployment targets disagree: Hugging Face Spaces
    injects secrets as environment variables, Streamlit Community Cloud exposes
    them through st.secrets, and local runs use .env.
    """
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value

    from .llm import load_dotenv

    load_dotenv()
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value

    try:  # Streamlit is not a dependency of the library path.
        import streamlit

        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            if name in streamlit.secrets:
                return streamlit.secrets[name]
    except Exception:  # noqa: BLE001 - no streamlit, or no secrets file
        pass
    return None


FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_DIMENSIONS = 384


class FastEmbedEmbedder:
    """Local ONNX embedding on CPU, with no server and no quota.

    This is what deployment actually needed. The alternatives each failed on
    something specific:

    Ollama needs a GPU and a model server, and no free host has either. Gemini
    meters ``EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`` at
    1,000 requests per day and counts a batch of 32 as 32 requests, so building
    a 1,384-chunk index does not fit inside one day's quota at all -- and every
    live query would then compete with rebuilds for what remained.

    ONNX on CPU has neither problem. It also happens to be far faster on the
    query path that matters: measured on this machine, 6ms against Ollama's
    ~2,150ms of fixed per-request overhead. The whole index embeds in about
    four seconds.

    The model is smaller than nomic-embed-text and 384-dimensional rather than
    768, so retrieval quality is a question rather than an assumption. It is
    measured against the same gold set as everything else; see the README.

    Weights download once (~130MB) and are cached, which is a first-boot cost
    on a fresh host and nothing thereafter.
    """

    name = "fastembed"

    def __init__(self, model: str = FASTEMBED_MODEL, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.dimensions = FASTEMBED_DIMENSIONS
        self._model = None
        self._query_cache: dict[str, list[float]] = {}

    @property
    def backend(self):
        """Loaded on first use, not at import.

        Constructing the model downloads weights on a cold host. Doing that at
        import time would make merely importing the package reach the network,
        which is the behaviour the reranker was already written to avoid.
        """
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model)
        return self._model

    def embed(self, texts: list[str], progress=None) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(v.tolist() for v in self.backend.embed(batch))
            if progress:
                progress(min(start + self.batch_size, len(texts)), len(texts))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is None:
            cached = next(iter(self.backend.embed([text]))).tolist()
            self._query_cache[text] = cached
        return cached

    def health_check(self) -> None:
        vector = self.embed_one("Article 21 protects life and personal liberty.")
        if len(vector) != self.dimensions:
            raise EmbeddingError(
                f"{self.model} returned {len(vector)} dimensions, expected "
                f"{self.dimensions}"
            )


def build_embedder(spec: str | None = None):
    """``ollama``/``gemini``/``fastembed`` -> an embedder.

    ``LEXGRAPH_EMBEDDER`` selects it, so the deployment switches embedder with
    an environment variable rather than an edit.
    """
    choice = (spec or os.getenv("LEXGRAPH_EMBEDDER", "ollama")).lower()
    if choice.startswith("gemini"):
        return GeminiEmbedder()
    if choice.startswith("fastembed") or choice.startswith("bge"):
        return FastEmbedEmbedder()
    if choice.startswith("ollama"):
        return OllamaEmbedder()
    raise ValueError(
        f"unknown embedder {choice!r}; expected 'ollama', 'gemini' or 'fastembed'"
    )
