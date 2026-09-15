#!/usr/bin/env bash
set -e

CONFIG_PATH=/data/options.json
PORT=5300
API_KEY=""

if [ -f "$CONFIG_PATH" ]; then
    PORT=$(jq -r '.port // 5300' "$CONFIG_PATH")
    API_KEY=$(jq -r '.api_key // ""' "$CONFIG_PATH")
fi

export PORT
export API_KEY
if [ -n "$API_KEY" ]; then
    echo "Starting OpenVINO upscaler API on port ${PORT} (API key required)"
else
    echo "Starting OpenVINO upscaler API on port ${PORT} (no API key set - open access)"
fi
exec python3 /app/app.py
