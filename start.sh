#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

echo "Starting Airship API on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

echo "Starting Airship Dashboard on port 7860..."
# Use --server.enableCORS=false and --server.enableXsrfProtection=false for full HF stability
streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
