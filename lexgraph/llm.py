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
    gemini:gemini-3-flash-preview
    groq:llama-3.3-70b-versatile
    openai:gpt-4o-mini

Gemini model names are retired fairly aggressively and availability varies by
key, so scripts/list_judge_models.py reports what a given key can actually
reach rather than leaving a 404 to be decoded.

Keys come from the environment (GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY,
CEREBRAS_API_KEY), loaded from .env if present.
"""

from __future__ import annotations

import os
import threading
import time

import requests

from .cache import CACHE, cache_enabled


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader. Existing environment variables always win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LLMError(RuntimeError):
    pass


# Gemini names the exhausted quota in the error body. The per-day one is not
# worth waiting on; the per-minute one is.
DAILY_QUOTA_MARKERS = ("perday", "requestsperday", "quota_exceeded_per_day")


def is_daily_quota_error(error: Exception) -> bool:
    text = str(error).lower().replace(" ", "")
    return "429" in text and any(marker in text for marker in DAILY_QUOTA_MARKERS)


def _quota_id(response) -> str:
    """Pull the violated quota's id to the front of the error text.

    Gemini reports which quota was exhausted in error.details[].violations[],
    but its human-readable message runs to about 250 characters, so the id sat
    past the point where the error text was truncated. The daily-quota check
    could never see its own marker, and every model in the rotation spent a
    full backoff cycle before being retired -- minutes per call, to reach a
    failure that was certain from the first response.
    """
    try:
        for detail in response.json().get("error", {}).get("details", []):
            for violation in detail.get("violations", []):
                if violation.get("quotaId"):
                    return f"[{violation['quotaId']}] "
    except (ValueError, AttributeError):
        pass
    return ""


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
        # Only greedy decoding is cacheable. Above temperature 0 the caller is
        # asking for a fresh sample, and replaying one would turn a sampling
        # experiment into a constant without saying so.
        cacheable = temperature == 0.0 and cache_enabled()
        key = CACHE.key(self.label, prompt, max_tokens, temperature) if cacheable else None
        if key is not None:
            cached = CACHE.get(key)
            if cached is not None:
                return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                response = self._chat(prompt, max_tokens, temperature)
                if key is not None:
                    CACHE.put(key, self.label, prompt, response)
                return response
            except Exception as error:  # noqa: BLE001 - retried, then re-raised
                last_error = error
                # A per-day quota will not clear inside a retry window, so
                # retrying spends minutes to reach the same failure. Raise at
                # once and let the caller rotate to another model. A per-minute
                # limit is the opposite case: waiting is exactly the fix.
                if is_daily_quota_error(error):
                    raise LLMError(f"{self.label} daily quota exhausted") from error
                if attempt < self.max_retries - 1:
                    time.sleep((10 if "429" in str(error) else 2) * (attempt + 1))
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

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: str | None = None,
        thinking_budget: int | None = 0,
        **kwargs,
    ):
        kwargs.setdefault("requests_per_minute", 10)
        super().__init__(model, **kwargs)
        # Gemini 3.x models spend output budget on internal reasoning before
        # emitting anything. At maxOutputTokens=200 a scoring call was measured
        # using 192 tokens on thoughts and 4 on output, returning a truncated
        # '{"score":' that parses as nothing. Scoring against a fixed rubric
        # does not need chain-of-thought, so it is disabled by default; this
        # also leaves the whole budget for the answer and conserves free-tier
        # quota. Set thinking_budget=None to let the model think.
        self.thinking_budget = thinking_budget
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            # Constructing this class directly should work without the caller
            # having to know that get_client() is what loads .env.
            load_dotenv()
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and put it in .env"
            )

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        config = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        response = requests.post(
            f"{self.endpoint}/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
            timeout=180,
        )
        if response.status_code == 404 and "no longer available" in response.text:
            raise LLMError(
                f"Model {self.model!r} is retired or not enabled for this key. "
                f"Run scripts/list_judge_models.py to see what this key can reach."
            )
        if response.status_code != 200:
            raise LLMError(f"{response.status_code}: {_quota_id(response)}{response.text[:300]}")

        candidates = response.json().get("candidates") or []
        if not candidates:
            raise LLMError(f"no candidates returned: {response.text[:300]}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            finish = candidates[0].get("finishReason")
            hint = (
                " -- the token budget was consumed by internal thinking; raise "
                "max_tokens or set thinking_budget=0"
                if finish == "MAX_TOKENS"
                else ""
            )
            raise LLMError(f"empty response (finishReason={finish}){hint}")
        return text


class OpenAICompatibleClient(BaseClient):
    """Any endpoint speaking the OpenAI chat-completions schema."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key_var: str,
                 reasoning_effort: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        # Reasoning models on this schema spend the token budget on a private
        # chain before writing anything, and the answer is what is left. On a
        # long retrieved context there is often nothing left. Asking for less
        # reasoning is the same fix as Gemini's thinkingBudget: 0, which this
        # module already applies for the same reason. Measured on gpt-oss-120b:
        # 177 characters of reasoning unset, 34 at "low".
        self.reasoning_effort = reasoning_effort
        self.api_key = os.getenv(api_key_var)
        if not self.api_key:
            raise LLMError(f"{api_key_var} is not set")

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=180,
        )
        if response.status_code != 200:
            raise LLMError(f"{response.status_code}: {response.text[:300]}")

        message = response.json()["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if content:
            return content

        # Reasoning models on this schema return their chain in a separate
        # "reasoning" field and the answer in "content". A tight max_tokens is
        # spent on the former, leaving the latter empty -- the same failure
        # Gemini's thinkingBudget produced, arriving through a different door.
        # Measured: gpt-oss-20b returns nothing at max_tokens=100 and a correct
        # answer at 400.
        if message.get("reasoning"):
            raise LLMError(
                f"{self.label} spent its {max_tokens}-token budget on reasoning "
                f"and returned no content. Raise max_tokens."
            )
        raise LLMError(f"{self.label} returned an empty response")


_OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 30),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 30),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 20),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", None),
}

class FallbackClient(BaseClient):
    """Tries several clients in order, moving on when one exhausts its quota.

    Gemini's free tier meters requests per day *per model*, and the cap is low
    enough that a single evaluation sweep exhausts it. Because the quota is
    per-model, rotating across several models multiplies the available budget
    without changing anything about the evaluation -- each call is still one
    independent judge scoring against the same rubric.

    A model is only retired for the rest of the run on a quota error. Ordinary
    failures fall through to the next client but leave the first in rotation.
    """

    name = "fallback"

    def __init__(self, clients: list[BaseClient]):
        if not clients:
            raise LLMError("FallbackClient needs at least one client")
        self.clients = clients
        self.exhausted: set[str] = set()
        super().__init__(clients[0].model, max_retries=1)

    @property
    def label(self) -> str:
        return " -> ".join(c.label for c in self.clients)

    @property
    def active(self) -> BaseClient:
        for client in self.clients:
            if client.label not in self.exhausted:
                return client
        return self.clients[-1]

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        errors = []
        for client in self.clients:
            if client.label in self.exhausted:
                continue
            try:
                return client.chat(prompt, max_tokens, temperature)
            except Exception as error:  # noqa: BLE001
                message = str(error)
                errors.append(f"{client.label}: {message[:120]}")
                if "429" in message or "quota" in message.lower():
                    self.exhausted.add(client.label)
        raise LLMError("every fallback client failed: " + " | ".join(errors))


# Gemini free-tier quota is per model per day, so a judge that rotates across
# several has several times the budget. Ordered strongest-first.
GEMINI_JUDGE_ROTATION = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
]


def build_judge(spec: str | None = None) -> BaseClient:
    """Build the judge, rotating across Gemini models when none is named."""
    load_dotenv()
    if spec and spec != DEFAULT_JUDGE:
        return get_client(spec)
    return FallbackClient([GeminiClient(model) for model in GEMINI_JUDGE_ROTATION])


def build_generator(spec: str, rotate: bool = False) -> BaseClient:
    """A generator, optionally rotating across Gemini models on quota.

    The quota is per model per day, so rotating multiplies the budget by the
    number of models a key can reach. Off by default because an evaluation run
    must know exactly which model produced each answer -- silently continuing
    on a different one would put two models in the same column.

    Batch jobs that only need to finish are the exception: pre-computing the
    replay set stopped at question 29 of 65 on a single model, and there is
    nothing to attribute wrongly when every answer records the model that
    produced it.
    """
    load_dotenv()
    if not rotate or not spec.startswith("gemini:"):
        return get_client(spec)

    named = spec.partition(":")[2]
    ordered = [named] + [m for m in GEMINI_JUDGE_ROTATION if m != named]
    return FallbackClient([GeminiClient(model) for model in ordered])


DEFAULT_GENERATOR = "ollama:llama3.1"

# What a deployment generates with. Ollama is the local default and needs a GPU
# no free host has, so something hosted has to stand in.
#
# Not Gemini. Gemini is the *judge*, and putting generation on it costs twice:
# it spends the same 1,000-request daily quota the evaluation needs, and it
# collapses generator and judge onto one provider, which is the arrangement
# require_independent_judge() exists to prevent. Groq's free tier is far larger
# and its keys are separate, so the judge stays independent and stays funded.
#
# The model was chosen by probing the key rather than from memory: Groq's
# catalogue had moved on and the llama-3.3 name assumed here did not exist on
# it. Of what was actually reachable, this one answered a grounded RAG prompt
# correctly in 1.3s. qwen3.6 was rejected for leaking <think> blocks into the
# answer; groq/compound for taking three times as long to do the same job.
DEFAULT_DEPLOY_GENERATOR = "groq:openai/gpt-oss-120b"
# HyDE runs once per query and its output is thrown away after embedding, so
# the small model that fits the card entirely is the right one: it needs the
# legal register, not the reasoning.
DEFAULT_HYDE_GENERATOR = "ollama:qwen2.5:3b"
DEFAULT_JUDGE = "gemini:gemini-3-flash-preview"


def get_client(spec: str) -> BaseClient:
    """Build a client from a ``provider:model`` string."""
    load_dotenv()
    provider, _, model = spec.partition(":")
    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient(model or "llama3.1")
    if provider == "gemini":
        return GeminiClient(model or "gemini-3-flash-preview")
    if provider in _OPENAI_COMPATIBLE:
        base_url, key_var, rpm = _OPENAI_COMPATIBLE[provider]
        if not model:
            raise LLMError(f"{provider} requires an explicit model, e.g. {provider}:<model>")
        # Only where the parameter is known to exist. Sending it blindly would
        # turn an unsupported field into a 400 on providers that reject extras.
        effort = "low" if "gpt-oss" in model else None
        return OpenAICompatibleClient(
            model, base_url, key_var, requests_per_minute=rpm, reasoning_effort=effort
        )

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
