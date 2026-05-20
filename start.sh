#!/bin/bash
# ============================================================
# PeptoMatch Cell 시작 스크립트 (Railway)
# ============================================================
set -e

echo "=== PeptoMatch Cell Starting ==="
echo "PORT=${PORT:-8000}"
echo "PWD=$(pwd)"

if [ -d "/opt/venv" ]; then
    export PATH="/opt/venv/bin:$PATH"
    echo "venv active: $(which python)"
fi

# Ensure data dir exists (DB lives here)
mkdir -p data

# Sync build-time xlsx into /app/data if Volume mount overshadows them
BUILD_DATA="/app/_build_data/data"
if [ -d "$BUILD_DATA" ]; then
    echo "[volume-sync] copying build xlsx files into ./data ..."
    shopt -s nullglob
    for f in "$BUILD_DATA"/*.xlsx; do
        cp -v "$f" ./data/
    done
    shopt -u nullglob
fi

echo "Starting FastAPI on 0.0.0.0:${PORT:-8000}..."
exec uvicorn gateway:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --log-level info \
    --access-log
