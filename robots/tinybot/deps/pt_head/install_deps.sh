#!/bin/bash

# Check if the script is run with root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo: sudo ./install_deps.sh"
  exit 1
fi

echo "--- Updating package list ---"
# Update the package list to ensure you get the latest versions
sudo apt-get update -y

echo "--- Installing common Python dependencies ---"
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.12-dev \
    python3.12-venv \
    git \
    jupyter-core    \
    jupyter-notebook \
    libportaudio2


echo "--- Python dependencies installation complete ---"
