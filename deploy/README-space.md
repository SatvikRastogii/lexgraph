---
title: LexGraph
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
dockerfile_path: Dockerfile.spaces
pinned: false
license: mit
short_description: An evaluation harness for retrieval over Indian case law
---

# LexGraph

An evaluation harness for retrieval systems, built over 40 Indian court
judgments. It measures retrieval configurations against a machine-verified
gold set and reports what the measurement can and cannot support.

Source and full write-up: https://github.com/SatvikRastogii/lexgraph

## What runs here

Retrieval is fully live. Embeddings are ONNX on CPU (~6ms per query) and
reranking is a cross-encoder, so every source, score and citation check you
see is computed on the spot.

Answers to the 65 gold-set questions are precomputed. They come from this same
pipeline — same retriever, prompt, threshold and citation check — and are
served instantly so a public demo cannot drain a free-tier quota. Ask something
outside that set and it generates live, within a small per-session budget.

GraphRAG's own query engine is not available here: it drives many sequential
local LLM calls and needs a GPU. Its retrieval is measured instead, on the same
gold set as everything else — see the Benchmark tab.

## Configuration

`GOOGLE_API_KEY` must be set as a Space secret for live generation. Without it
the precomputed answers, the benchmark and every visualisation still work.
