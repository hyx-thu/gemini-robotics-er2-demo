#!/bin/bash
# run_robot.sh - Run the robot API server with correct environment and workarounds

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run with LD_PRELOAD workaround and using the venv python
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$DIR"/.venv/bin/python "$DIR"/src/robot/run_robot.py "$@"
