#!/bin/bash

# This script should be sourced, not executed.
# Usage: source activate.sh

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_EXE="python3.12"
    else
        PYTHON_EXE="python3"
    fi
    echo "Virtual environment '$VENV_DIR' not found. Creating it with $PYTHON_EXE..."
    $PYTHON_EXE -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
            exit 1
        else
            return 1
        fi
    fi
    echo "Virtual environment created."
fi

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Error: This script must be sourced, not executed directly."
    echo "Use: source activate.sh"
    exit 1
else
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment '$VENV_DIR' activated."
fi
