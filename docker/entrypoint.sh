#!/bin/sh
set -e

# settings.yaml has no native env-var interpolation. It ships with the real
# default (http://localhost:11434) so it stays valid for local, non-Docker
# use; here we rewrite that default to $OLLAMA_HOST in place, which is a
# no-op when OLLAMA_HOST is unset/still localhost.
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
sed -i "s#http://localhost:11434#${OLLAMA_HOST}#g" /app/settings.yaml

# Bind address and port are set here rather than in .streamlit/config.toml.
# A container has to listen on 0.0.0.0 to be reachable through a published
# port, but that file is also read by Streamlit Cloud, where pinning a port
# fights the host -- so the container-specific part lives with the container.
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${STREAMLIT_PORT:-8501}" \
  --server.headless=true
