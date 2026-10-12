#!/bin/bash
# run_agent.sh - Run the physical-agent server with correct environment

set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -d "$DIR/.venv" ]; then
    echo "Virtual environment not found. Running setup.sh..."
    "$DIR/setup.sh"
fi

if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi

exec "$DIR"/.venv/bin/python "$DIR"/server.py "$@"
