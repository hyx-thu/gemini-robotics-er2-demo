#!/bin/bash
# setup.sh - Setup virtual environment and install dependencies for tinybot

set -e

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

VENV_DIR=".venv"

echo "=== Tinybot Setup ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source "$VENV_DIR"/bin/activate

echo "Upgrading pip..."
# We use PyPI index as fallback/default if the system one is restricted/empty
pip install --index-url https://pypi.org/simple --upgrade pip

echo "Installing setuptools..."
pip install --index-url https://pypi.org/simple "setuptools>=61.0.0"

echo "Installing tinybot in editable mode with dependencies..."
# We use --no-build-isolation to use the setuptools we just installed
pip install --index-url https://pypi.org/simple --no-build-isolation -e .

PT_HEAD_DIR="deps/pt_head"
if [ ! -d "$PT_HEAD_DIR" ]; then
    PT_HEAD_DIR="../pt_head"
fi

echo "Installing pt_head dependency from $PT_HEAD_DIR..."
pip install --index-url https://pypi.org/simple --no-build-isolation -e "$PT_HEAD_DIR"

echo "=== Setup Complete ==="
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate"
