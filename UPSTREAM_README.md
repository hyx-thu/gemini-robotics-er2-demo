# Gemini Robotics ER - Live API Examples

A repository of examples of connecting Gemini Robotics ER with physical robot embodiments using Live API for task orchestration, voice interactions, etc. The embodiments include Boston Dynamics Spot, Tinybot (a custom stationary robot hardware), and human operators.

---

## Repository Structure & Packages

| Package | Purpose & Features | Environment / Stack | Link |
| :--- | :--- | :--- | :--- |
| **`agent`** | **Physical Agent Server**: Core agent server interfacing with the Gemini Live API over WebSockets for real-time audio/video interaction, tool dispatch, and robot control. | Python 3.10+ (FastAPI, WebSockets, `google-genai`, `uv`) | [`./agent`](./agent/README.md) |
| **`spot`** | **Boston Dynamics Spot SDK Integration**: Suite of CLI tools, REST APIs, object detection/manipulation pipelines, and autonomous delivery applications for Spot. | Python (`bosdyn-client`, `google-genai`), Node.js/React (`apps/hydration`) | [`./spot`](./spot/README.md) |
| **`tinybot`** | **Compact Robot Controller**: Lightweight server providing camera video streaming and basic REST API controls for compact or custom robot hardware. | Python (FastAPI, OpenCV) | [`./tinybot`](./tinybot) |

---

## Detailed Package Overview

### 1. [Agent Server (`./agent`)](./agent/README.md)

The `agent` directory contains the main orchestration server connecting multimodal AI models to physical hardware.

* **Gemini Live API Integration**: Handles bi-directional audio/video streaming with Gemini Live API using WebSockets.
* **Embodiment Architecture**: Modular client abstractions (`agent/embodiment/`) to control different physical targets (`spot`, `tinybot`, `human`).
* **Web UI & Camera Poller**: Includes a built-in web interface and camera polling service (`camera_poller.py`) for real-time visual feeds.
* **Quickstart**:
  ```bash
  cd agent
  UV_CACHE_DIR=.uv-cache uv sync
  UV_CACHE_DIR=.uv-cache uv run python server.py --port 8000
  ```

---

### 2. [Spot Applications & SDK (`./spot`)](./spot/README.md)

The `spot` directory contains Boston Dynamics Spot integrations powered by `bosdyn-client` and Gemini vision tools.

* **Navigation App ([`apps/navigation`](./spot/apps/navigation))**: Manage GraphNav waypoints, register named locations, and command Spot to navigate autonomously via CLI.
* **Manipulation App ([`apps/manipulation`](./spot/apps/manipulation))**: Arm deployment, Gemini-based 2D/3D object detection, force-change detection, and picking.
* **FastAPI Server ([`apps/api`](./spot/apps/api))**: Exposes HTTP REST endpoints for Spot movement, leases, arm control, and waypoints (`http://localhost:8000/docs`).
* **Hydration Delivery Service ([`apps/hydration`](./spot/apps/hydration))**: Full-stack Node/React app and order worker that commands Spot to deliver drinks.
* **Quickstart**:
  ```bash
  cd spot
  UV_CACHE_DIR=.uv-cache uv sync
  UV_CACHE_DIR=.uv-cache uv run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
  ```

---

### 3. [Tinybot Hardware Controller (`./tinybot`)](./tinybot)

The `tinybot` directory provides lightweight camera streaming and basic hardware control endpoints for smaller physical robot hardware.

* **Camera Streamer ([`src/robot/camera_streamer.py`](./tinybot/src/robot/camera_streamer.py))**: Captures and streams live camera feeds for vision processing.
* **Robot REST API ([`src/robot/robot_api.py`](./tinybot/src/robot/robot_api.py))**: Exposes REST endpoints for low-level movement execution.
* **Quickstart**:
  ```bash
  cd tinybot
  ./setup.sh
  ./run_robot.sh
  ```

---

## Environment Setup

All Python subpackages use [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management. To keep virtual environments isolated and clean:

```bash
# Sync dependencies within any subfolder using local cache:
UV_CACHE_DIR=.uv-cache uv sync
```
