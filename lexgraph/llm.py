"""Provider-agnostic chat client.

The generator runs locally on Ollama. The judge must not: scoring llama3.1's
answers with llama3.1 is self-evaluation, and it inflates every metric in a
direction nobody can measure from inside the loop. So the same interface has
to reach a second, independent model.

This is deliberately a dispatch function and not a gateway library. There is
one model per role and no routing, failover or spend management to do, so
LiteLLM or Portkey would add a dependency and a configuration surface to solve
a problem that does not exist here.

Clients are named ``provider:model``:

    ollama:llama3.1
    gemini:gemini-2.5-flash
    groq:llama-3.3-70b-versatile
    openai:gpt-4o-mini

Keys come from the environment (GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY,
CEREBRAS_API_KEY), loaded from .env if present.
"""

from __future__ import annotations

import os
import threading
import time

import requests


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader. Existing environment variables always win."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Enforces a minimum gap between calls.

    Free tiers are quoted in requests per minute, and tripping the limit costs
    far more time than pacing does.
    """

    def __init__(self, requests_per_minute: float | None):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


class BaseClient:
    name = "base"

    def __init__(self, model: str, requests_per_minute: float | None = None, max_retries: int = 4):
        self.model = model
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                return self._chat(prompt, max_tokens, temperature)
            except Exception as error:  # noqa: BLE001 - retried, then re-raised
                last_error = error
                if attempt < self.max_retries - 1:
                    # Back off hard on rate limiting, gently on anything else.
                    is_rate_limited = "429" in str(error) or "quota" in str(error).lower()
                    time.sleep((10 if is_rate_limited else 2) * (attempt + 1))
        raise LLMError(f"{self.label} failed after {self.max_retries} attempts: {last_error}")

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class OllamaClient(BaseClient):
    name = "ollama"

    def __init__(self, model: str = "llama3.1", host: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class GeminiClient(BaseClient):
    name = "gemini"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, **kwargs):
        kwargs.setdefault("requests_per_minute", 10)
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and put it in .env"
            )

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = requests.post(
            f"{self.endpoint}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise LLMError(f"{response.status_code}: {response.text[:300]}")

        candidates = response.json().get("candidates") or []
        if not candidates:
            raise LLMError(f"no candidates returned: {response.text[:300]}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            # A hit MAX_TOKENS or safety stop returns a candidate with no text.
            raise LLMError(f"empty response (finishReason={candidates[0].get('finishReason')})")
        return text


class OpenAICompatibleClient(BaseClient):
    """Any endpoint speaking the OpenAI chat-completions schema."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key_var: str, **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_var)
        if not self.api_key:
            raise LLMError(f"{api_key_var} is not set")

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise LLMError(f"{response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"].strip()


_OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 30),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 30),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 20),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
}

DEFAULT_GENERATOR = "ollama:llama3.1"
DEFAULT_JUDGE = "gemini:gemini-2.5-flash"


def get_client(spec: str) -> BaseClient:
    """Build a client from a ``provider:model`` string."""
    load_dotenv()
    provider, _, model = spec.partition(":")
    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(model or "llama3.1")
    if provider == "gemini":
        return GeminiClient(model or "gemini-2.5-flash")
    if provider in _OPENAI_COMPATIBLE:
        base_url, key_var, rpm = _OPENAI_COMPATIBLE[provider]
        if not model:
            raise LLMError(f"{provider} requires an explicit model, e.g. {provider}:<model>")
        return OpenAICompatibleClient(model, base_url, key_var, requests_per_minute=rpm)

    raise LLMError(
        f"unknown provider {provider!r}. Expected ollama, gemini, "
        f"or one of {sorted(_OPENAI_COMPATIBLE)}"
    )


def judge_is_independent(generator_spec: str, judge_spec: str) -> bool:
    """True when judge and generator are genuinely different models.

    Called before an evaluation so a self-judging run fails loudly instead of
    quietly producing inflated scores.
    """
    return generator_spec.strip().lower() != judge_spec.strip().lower()
