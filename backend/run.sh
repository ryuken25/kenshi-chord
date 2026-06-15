#!/usr/bin/env bash
# Start backend (FASE 1 dev)
set -e
cd "$(dirname "$0")"
[ -d venv ] || python -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
