import json
import os

from lexgraph.cache import ResponseCache


def test_round_trips_a_response(tmp_path):
    cache = ResponseCache(str(tmp_path))
    key = cache.key("ollama:llama3.1", "What is Article 21?", 512, 0.0)
    assert cache.get(key) is None

    cache.put(key, "ollama:llama3.1", "What is Article 21?", "It protects life.")
    assert cache.get(key) == "It protects life."


def test_key_covers_every_parameter_that_changes_the_answer(tmp_path):
    cache = ResponseCache(str(tmp_path))
    base = cache.key("ollama:llama3.1", "q", 512, 0.0)

    assert cache.key("ollama:qwen2.5:3b", "q", 512, 0.0) != base, "model"
    assert cache.key("ollama:llama3.1", "other", 512, 0.0) != base, "prompt"
    assert cache.key("ollama:llama3.1", "q", 800, 0.0) != base, "token budget"
    assert cache.key("ollama:llama3.1", "q", 512, 0.7) != base, "temperature"


def test_corrupt_entry_is_a_miss_not_an_error(tmp_path):
    # The cache is an optimisation and must never be able to fail a run.
    cache = ResponseCache(str(tmp_path))
    key = cache.key("m", "p", 512, 0.0)
    path = os.path.join(str(tmp_path), key[:2], f"{key}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"response": "truncated')

    assert cache.get(key) is None


def test_entry_without_the_expected_shape_is_a_miss(tmp_path):
    cache = ResponseCache(str(tmp_path))
    key = cache.key("m", "p", 512, 0.0)
    path = os.path.join(str(tmp_path), key[:2], f"{key}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"unexpected": "shape"}, handle)

    assert cache.get(key) is None


def test_put_leaves_no_temporary_file_behind(tmp_path):
    cache = ResponseCache(str(tmp_path))
    key = cache.key("m", "p", 512, 0.0)
    cache.put(key, "m", "p", "answer")

    leftovers = [
        name
        for _, _, names in os.walk(str(tmp_path))
        for name in names
        if name.endswith(".tmp")
    ]
    assert not leftovers


def test_stats_track_hits_and_misses(tmp_path):
    cache = ResponseCache(str(tmp_path))
    key = cache.key("m", "p", 512, 0.0)
    cache.get(key)
    cache.put(key, "m", "p", "a")
    cache.get(key)
    cache.get(key)

    assert cache.stats == {"hits": 2, "misses": 1, "hit_rate": 2 / 3}


def test_unwritable_directory_does_not_raise(tmp_path):
    # A read-only cache directory should cost the speedup, not the run.
    cache = ResponseCache(os.path.join(str(tmp_path), "file-not-a-dir"))
    with open(os.path.join(str(tmp_path), "file-not-a-dir"), "w") as handle:
        handle.write("x")

    cache.put(cache.key("m", "p", 512, 0.0), "m", "p", "a")  # must not raise
