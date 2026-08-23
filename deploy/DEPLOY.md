# Deploying LexGraph

Two targets, both free, neither needing a GPU, an Ollama server, or a re-index.

**Streamlit Community Cloud** is the path to take — the `streamlit-deploy`
branch is ready and the whole thing is four clicks and one secrets block.
**Hugging Face Spaces** is documented below it and is the better host if the
app ever outgrows 1GB of memory.

The configuration was verified before either was written: a clean virtualenv
containing only `requirements-deploy.txt`, the app booted in it, and a gold-set
question answered end to end in the browser. That found four bugs that no
amount of reading would have — see the commit history.

## What had to change first

The app could not be deployed as written. Three things blocked it, and each was
resolved rather than worked around:

| blocker | resolution |
|---|---|
| dense retrieval called Ollama for every query embedding | ONNX on CPU via `fastembed` — 6ms per query, no server, no quota |
| the index was 84MB of gitignored ChromaDB | a 1.9MB committed matrix and brute-force cosine; exact, not approximate |
| the GraphRAG tab shells out to the `graphrag` CLI | disabled under `LEXGRAPH_DEPLOY`; the measured result is shown instead |

Gemini embeddings were the obvious answer and do not work: the free tier allows
**1,000 embed requests per day** and counts a batch of 32 as 32 requests, so a
1,384-chunk index does not fit inside one day's quota, and every live query
would afterwards compete with rebuilds for the remainder. `GeminiEmbedder` is
kept for anyone on a paid tier, but it is not the default.

The substitution was measured, not assumed. Against the same 65-question gold
set, `hybrid-rerank` scores nDCG@10 0.859 deployed against 0.860 locally, with
*better* paragraph-level precision (0.828 against 0.805) and less than half the
latency. See `reports/deploy_ablation.json`.

## Prerequisites

```bash
python scripts/build_deploy_index.py          # data/vectors/paragraph.npz, ~1.9MB
python scripts/calibrate_abstention.py --output reports/abstention_deploy.json
python scripts/build_replay.py --generator gemini:gemini-3-flash-preview
```

Run the last two with the deployment's own environment, or they will calibrate
and answer against the wrong score scale:

```bash
LEXGRAPH_DENSE_BACKEND=numpy LEXGRAPH_EMBEDDER=fastembed ...
```

The threshold is **not** transferable between embedders. `hybrid-rerank`
calibrates to 0.045 under the deployment embedder and 0.027 locally; using one
for the other produces a plausible number that refuses the wrong questions.

All three outputs are committed.

## Streamlit Community Cloud

Streamlit Cloud installs from `requirements.txt` at the repository root and
offers no way to point it elsewhere. The full file pulls `graphrag`, `lancedb`
and `chromadb`, none of which the deployment uses and which together will not
fit in the free tier's memory.

The **`streamlit-deploy` branch** exists for exactly this. It is `main` plus two
files — `requirements.txt` (a copy of `requirements-deploy.txt`) and
`runtime.txt` pinning Python 3.11 — so `main` stays honest about what the
project actually depends on.

### Steps

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository `SatvikRastogii/lexgraph`, branch **`streamlit-deploy`**, main
   file **`app.py`**.
4. Open **Advanced settings → Secrets** and paste the block below.
5. Deploy. The first boot installs dependencies and downloads the embedding and
   reranking weights (~150MB), so it takes a few minutes; later boots are fast.

### Secrets

Everything goes in this one box. Streamlit Cloud has **no environment-variable
UI**, which is why `app.py` copies these into `os.environ` at startup — without
that the app would silently run the local configuration and fail on the first
query.

```toml
GOOGLE_API_KEY = "your-key"

LEXGRAPH_DEPLOY = "1"
LEXGRAPH_DENSE_BACKEND = "numpy"
LEXGRAPH_EMBEDDER = "fastembed"
LEXGRAPH_RETRIEVER = "hybrid-rerank"
LEXGRAPH_GENERATOR = "gemini:gemini-3-flash-preview"
LEXGRAPH_LIVE_BUDGET = "5"
```

Only `GOOGLE_API_KEY` is a secret in the real sense; the rest are configuration
that has nowhere else to live on this host. Without the key everything still
works except live generation of questions outside the gold set.

### Keeping it current

```bash
git checkout streamlit-deploy
git rebase main
git push --force-with-lease origin streamlit-deploy
```

Streamlit Cloud redeploys on push. Nothing but those two files differs, so the
rebase should never conflict.

### Memory

The free tier gives about 1GB. The slim set stays well inside it: no `torch`,
no `chromadb`, no `graphrag`. Both models are ONNX and the vector store is a
1.9MB array. If it is ever tight, `LEXGRAPH_RETRIEVER = "hybrid"` drops the
cross-encoder — but that configuration has its own calibrated threshold, and
using `hybrid-rerank`'s would be meaningless.

## Hugging Face Spaces

1. Create a Space: **SDK = Docker**, hardware = CPU basic (free).
2. Push this repository to the Space remote.
3. Copy `deploy/README-space.md` to the Space's `README.md` — Spaces reads its
   configuration from that file's YAML frontmatter, including
   `dockerfile_path: Dockerfile.spaces` and `app_port: 7860`.
4. Add `GOOGLE_API_KEY` under **Settings → Variables and secrets**. Spaces
   injects secrets as environment variables, which is where `_google_api_key()`
   looks first.

The image downloads the embedding and reranking weights at build time, so the
first visitor does not pay ~30s of model download while watching.

## Environment reference

| variable | default | purpose |
|---|---|---|
| `LEXGRAPH_DEPLOY` | unset | replay-first answers, live budget, GraphRAG CLI disabled |
| `LEXGRAPH_DENSE_BACKEND` | `chroma` | `numpy` reads the committed matrix |
| `LEXGRAPH_EMBEDDER` | `ollama` | `fastembed` or `gemini` |
| `LEXGRAPH_GENERATOR` | `ollama:llama3.1` | any `provider:model` |
| `LEXGRAPH_RETRIEVER` | `hybrid-rerank` | any configuration in the ablation |
| `LEXGRAPH_LIVE_BUDGET` | `5` | live generations per browser session |

## Checking it locally first

```bash
LEXGRAPH_DEPLOY=1 LEXGRAPH_DENSE_BACKEND=numpy LEXGRAPH_EMBEDDER=fastembed \
  streamlit run app.py
```

This is the exact configuration the hosted build runs, so a problem shows up
here rather than in front of whoever you sent the link to.

## Measured on the deployment configuration

| | value |
|---|---|
| retriever build (cold) | 0.7s |
| query embedding | ~6ms |
| `hybrid-rerank` retrieval | ~2.4s (the cross-encoder is the cost) |
| replayed answer, end to end | ~2.4s — all retrieval |
| live answer, end to end | ~9s |
| committed payload | 1.9MB vectors + 5.5MB parquets + 148KB replay |

Replay is not instant, and the README does not claim it is. Only generation is
cached; the retrieval shown next to the answer is really being run.

Guardrail behaviour across the whole gold set, at the deployment threshold:
48 of 55 answerable questions answered (87%), 5 of 10 out-of-corpus refused
(50%) — matching the calibration exactly, which is the check that it transfers.

## What the demo does not do

- **No GraphRAG query engine.** It needs a GPU and minutes per query. Its
  retrieval is scored in the Benchmark tab instead, which is the more useful
  artefact: community-report retrieval reaches 0.474 Recall@5 against
  `hybrid-rerank`'s 0.879.
- **The live budget is a courtesy, not a security boundary.** It is per browser
  session with no server-side counter, and reloading resets it. It exists to
  stop casual quota exhaustion, and is not pretending to be more.
- **Answer text for gold-set questions is precomputed.** Retrieval beside it is
  not. The distinction is shown in the UI rather than hidden.
