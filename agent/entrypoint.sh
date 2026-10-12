#!/bin/bash
set -e

echo "Starting Virtual Spot & Tinybot Simulator Backend on 127.0.0.1:8888..."
python3 /app/mock_robot_backend.py &

# Wait briefly for mock backend to listen on 8888
sleep 1

echo "Starting Gemini Robotics ER 2.0 Live Agent Server on port ${PORT:-8080}..."
exec python3 /app/server.py \
  --port "${PORT:-8080}" \
  --robot_url "${ROBOT_URL:-http://127.0.0.1:8888}"
