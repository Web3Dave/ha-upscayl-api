#!/usr/bin/env bash
set -e

CONFIG_PATH=/data/options.json
PORT=5300

if [ -f "$CONFIG_PATH" ]; then
    PORT=$(jq -r '.port // 5300' "$CONFIG_PATH")
fi

export PORT
echo "Starting OpenVINO upscaler API on port ${PORT}"
exec python3 /app/app.py
