#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

echo "Starting Airship API on port 8000..."
# Critical: Hugging Face exposes exactly ONE port (7860) to the public web by default.
# Since the LLM Evaluator must hit the FastAPI endpoints (/reset, /step), uvicorn MUST be on 7860.
echo "Starting Airship FastAPI (Round 1 Evaluator Target) on public port 7860..."
uvicorn server.app:app --host 0.0.0.0 --port 7860 &

echo "Starting internal Airship Dashboard on internal port 8501..."
# UI is preserved internally (port-forwarding required locally)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
