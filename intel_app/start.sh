#!/bin/bash
set -e

echo ">>> Starting main FastAPI app on port 9000..."
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:9000 \
  --workers 1 \
  --timeout 300 \
  --graceful-timeout 120 \
  --log-level debug &
MAIN_PID=$!

echo ">>> Starting worker FastAPI app on port 9500..."
gunicorn worker_app.app_worker:app \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:9500 \
  --workers 1 \
  --timeout 300 \
  --graceful-timeout 180 \
  --max-requests 200 \
  --max-requests-jitter 40 \
  --keep-alive 5 \
  --log-level debug &
WORKER_PID=$!

echo ">>> Waiting for processes..."
wait $MAIN_PID $WORKER_PID
