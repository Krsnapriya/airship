#!/bin/bash
set -e

echo "🚀 Starting DebugOps-RX Airship (FastAPI + Chaos Mode) on HF Spaces..."

python -c "from server.env import AirshipEnv; print('✅ OpenEnv + Meta-Controller loaded')"

exec uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 7860 \
  --log-level info
