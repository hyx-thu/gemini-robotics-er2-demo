## PT Head 

A tiny codebase for a tiny robot.

### Installation

To set up the environment and install the package, follow these steps:

1.  **Install system-level dependencies:**
    This script installs necessary system packages (such as `libportaudio2` for audio playback and development tools) required by the Python packages.
    ```bash
    sudo ./install_deps.sh
    ```

2.  **Source the activation script:**
    This script will create a local Python virtual environment (`.venv`) if it doesn’t exist and activate it for your current terminal session.
    ```bash
    source activate.sh
    ```

3.  **Install the Python package in editable mode:**
    ```bash
    pip install -e .
    ```

### Troubleshooting

- **`OSError: PortAudio library not found`**
  If you see this error when running scripts (like `examples/emotes.py`), it means the `libportaudio2` package was not installed on your system. Make sure you have run Step 1 (`sudo ./install_deps.sh`) successfully.

- **Permission denied on `/dev/ttyUSB0`**
  If the script fails to open the serial port to communicate with the hardware, you likely need to add your user to the `dialout` group:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  *(Note: You will need to log out and log back in for this change to take effect.)*