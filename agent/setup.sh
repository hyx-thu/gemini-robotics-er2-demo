#!/bin/bash
# setup.sh - Create the virtual environment and install dependencies.
#
# On Cloudtop/CitC the workspace is a FUSE mount and importing thousands of
# site-packages files from it is painfully slow (fastapi alone ~12s). So the
# real venv lives on local disk under ~/.venvs/ and ./.venv is a symlink to it.
# Override with VENV_DIR=/some/path ./setup.sh

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PROJECT_NAME="gemini-robotics-er-demo"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/$PROJECT_NAME}"

echo "=== Gemini Robotics-ER Agent Setup ==="
echo "venv location: $VENV_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync

# Point ./.venv at the real environment (replace stale dir/symlink if present).
if [ -L .venv ] || [ -d .venv ]; then
    rm -rf .venv
fi
ln -s "$VENV_DIR" .venv

if [ ! -f .env ]; then
    cp .env.example .env
    echo
    echo "!! Created .env from .env.example — edit it and set GEMINI_API_KEY."
fi

echo "=== Setup Complete ==="
echo "Run the server with: ./run_agent.sh"
