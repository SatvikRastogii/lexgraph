"""On-disk cache for deterministic LLM calls.

Re-running an evaluation costs an hour of local generation per generator and a
day of judge quota, and most of that is spent recomputing answers that have not
changed. Nothing about a retrieval experiment requires re-generating the answer
to question 14 for the fourth time.

Only temperature 0 is cached. Above it the model is being asked for a different
sample each call, and returning the same one would silently turn a sampling
experiment into a constant -- the sort of quiet substitution this repository is
otherwise about catching.

The key covers the model, the prompt, and the decoding parameters, so switching
generator or token budget is a miss rather than a stale hit. Entries are plain
JSON files, one per call, which makes the cache inspectable and lets a single
bad entry be deleted by hand.

One interaction is worth knowing about. The judge's self-disagreement is
estimated from `context_precision`, which scores only the question and the
context -- byte-identical across two generator runs, so any difference is the
judge disagreeing with itself. A cache that collapsed those two calls would
report a noise floor of exactly zero and make the judge look perfect. It does
not, because the batched judge prompt also carries the answer and the answers
differ. That is luck rather than design, so it is written down: if metrics are
ever scored in separate calls, the answer-independent ones must bypass the
cache or the control stops measuring anything.

    LEXGRAPH_CACHE=off        disable
    LEXGRAPH_CACHE_DIR=path   somewhere other than .llm_cache/
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

DEFAULT_CACHE_DIR = ".llm_cache"


def cache_enabled() -> bool:
    return os.getenv("LEXGRAPH_CACHE", "on").lower() not in {"off", "0", "false", "no"}


def cache_dir() -> str:
    return os.getenv("LEXGRAPH_CACHE_DIR", DEFAULT_CACHE_DIR)


class ResponseCache:
    """Content-addressed store of prompt -> completion."""

    def __init__(self, directory: str | None = None):
        self.directory = directory or cache_dir()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def key(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        payload = json.dumps(
            [model, prompt, max_tokens, temperature], sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path(self, key: str) -> str:
        # One directory per two-character prefix: a flat directory of tens of
        # thousands of files is slow to list on every filesystem this runs on.
        return os.path.join(self.directory, key[:2], f"{key}.json")

    def get(self, key: str) -> str | None:
        try:
            with open(self._path(key), encoding="utf-8") as handle:
                response = json.load(handle)["response"]
        except (OSError, ValueError, KeyError):
            # A corrupt or half-written entry is a miss, never an error. The
            # cache is an optimisation and must not be able to fail a run.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return response

    def put(self, key: str, model: str, prompt: str, response: str) -> None:
        path = self._path(key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Write then rename, so a run killed mid-write cannot leave a
            # truncated entry that later reads as a real response.
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {"model": model, "prompt": prompt, "response": response},
                    handle,
                )
            os.replace(temporary, path)
        except OSError:
            pass

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


CACHE = ResponseCache()
