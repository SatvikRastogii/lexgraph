#!/bin/sh
set -e

# settings.yaml has no native env-var interpolation. It ships with the real
# default (http://localhost:11434) so it stays valid for local, non-Docker
# use; here we rewrite that default to $OLLAMA_HOST in place, which is a
# no-op when OLLAMA_HOST is unset/still localhost.
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
sed -i "s#http://localhost:11434#${OLLAMA_HOST}#g" /app/settings.yaml

exec streamlit run app.py
