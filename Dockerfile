FROM python:3.11-slim

# graphrag and chromadb need build tools for native deps; curl is used by Compose healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and config only — output/, cache/, chroma_db/, logs/, legal_corpus/, input/
# are bind-mounted at runtime (see docker-compose.yml), not baked into the image.
COPY app.py analyze_contradictions.py telemetry.py ./
COPY lexgraph/ ./lexgraph/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY settings.yaml ./settings.yaml
COPY prompts/ ./prompts/
COPY corpus_metadata.json ./
COPY .streamlit/ ./.streamlit/
COPY monitoring/ ./monitoring/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["entrypoint.sh"]
