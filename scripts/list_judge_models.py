"""Report which judge models the configured API key can actually reach.

Gemini retires model names on a short cycle and availability differs between
keys -- a name listed by the API can still return "no longer available to new
users" when called. Rather than leaving that as a 404 to decode mid-run, this
lists the candidates and probes them.

    python scripts/list_judge_models.py
    python scripts/list_judge_models.py --probe
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from lexgraph.llm import GeminiClient, LLMError, load_dotenv

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="actually call each text model instead of only listing it")
    parser.add_argument("--filter", default="flash",
                        help="substring a model name must contain (default: flash)")
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY / GOOGLE_API_KEY is not set. See .env.example")

    response = requests.get(ENDPOINT, headers={"x-goog-api-key": key}, timeout=60)
    if response.status_code != 200:
        sys.exit(f"Could not list models: {response.status_code} {response.text[:200]}")

    names = [
        model["name"].replace("models/", "")
        for model in response.json().get("models", [])
        if "generateContent" in model.get("supportedGenerationMethods", [])
    ]
    candidates = [
        name for name in names
        if args.filter in name
        and not any(skip in name for skip in ("image", "tts", "robotics", "audio"))
    ]

    print(f"{len(names)} text models visible to this key; "
          f"{len(candidates)} match {args.filter!r}\n")

    if not args.probe:
        for name in candidates:
            print(f"  {name}")
        print("\nRe-run with --probe to check which of these actually answer.")
        return

    print(f"{'model':<34}{'status':<10}{'latency':>10}")
    print("-" * 54)
    for name in candidates:
        client = GeminiClient(name)
        started = time.perf_counter()
        try:
            client.chat('Reply with only: {"score": 5}', max_tokens=256)
            elapsed = (time.perf_counter() - started) * 1000
            print(f"  {name:<32}{'ok':<10}{elapsed:>8.0f}ms")
        except LLMError as error:
            print(f"  {name:<32}{'FAILED':<10}  {str(error)[:60]}")


if __name__ == "__main__":
    main()
