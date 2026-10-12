# Physical Agent Server

Physical Agent Server managed with [`uv`](https://github.com/astral-sh/uv). It exposes a FastAPI server, WebSocket endpoints for live streaming audio/video with Gemini Live API, and robot embodiment integrations (Spot, Human, Tinybot).

---

## Prerequisites

- **Python**: `>=3.10`
- **`uv`**: Installed locally (e.g. `uv 0.11+`)

---

## Quickstart

### 1. Synchronize Dependencies

Run `uv sync` to set up the virtual environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --default-index https://pypi.org/simple
```

---

### 2. Set API Keys (Optional)

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

### 3. Start the Agent Server

#### Basic Launch
```bash
UV_CACHE_DIR=.uv-cache uv run --default-index https://pypi.org/simple python server.py --port 8000
```

#### Launch with Custom Model & Spot Robot Endpoint
```bash
UV_CACHE_DIR=.uv-cache uv run --default-index https://pypi.org/simple python server.py \
    --model gemini-3.1-flash-live-preview \
    --robot_url http://<spot_pc_ip>:8000 \
    --port 8000
```

Once running, access the web UI at:
- **Local Web UI**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

---

## Running Tests

Run all unit tests via `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv run --default-index https://pypi.org/simple python -m unittest discover -p "*_test.py"
```

---

## Command Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Gemini model name | `gemini-3.1-flash-live-preview` |
| `--robot_url` | Robot OpenAPI / HTTP base URL | `None` |
| `--api_key` | Gemini API Key | `GEMINI_API_KEY` env var |
| `--use_tts` / `--no_tts` | Enable or disable Text-to-Speech | `Disabled` |
| `--port` | HTTP server port | `8000` |
| `--media_resolution` | Video input frame resolution (`low`, `medium`, `high`, `ultra_high`) | `low` |
| `--heartbeat_enabled` | Enable proactive heartbeat | `True` |
| `--agent_peers` | Comma-separated peer agents (`name=url,name=url`) | `None` |
