# Gemini Robotics-ER Demo

Voice-driven embodied agent built on **Gemini Robotics-ER 2 (streaming / Live API)**.
Talk to the agent, show it a camera feed, and it reasons about the scene and
drives a robot embodiment (Spot, Tinybot, or a simulated "human").

Forked from [google-gemini/robotics-samples/live-api](https://github.com/google-gemini/robotics-samples/tree/main/live-api)
and extended with streaming TTS + barge-in, grasp-point suggestion, tool-call
visualisation, session resumption, and a Spot simulator.

- **Live demo**: https://gemini-robotic-er2-242017771503.us-central1.run.app
- **GCP project**: `elated-terminal-503102-u0` / `us-central1` / Cloud Run service `gemini-robotic-er2`

## Layout

```
gemini-robotics-er-demo/
├── agent/                 # FastAPI server + web UI (this is what gets deployed)
│   ├── server.py          # entry point: WebSocket /ws, static UI, REST
│   ├── session_manager.py # Live API session, reconnect, VAD, TTS pipeline
│   ├── core/              # observation / decision / tool-call handling / watchdog
│   ├── embodiment/        # human | spot | tinybot drivers (+ spot simulator)
│   ├── model/             # Live API client, streaming TTS client
│   ├── prompt/            # system instructions per embodiment
│   ├── tool/              # tool declarations exposed to the model
│   ├── ui/                # browser client (agent-ui.js, media-handler.js)
│   ├── mock_robot_backend.py  # local Spot/Tinybot simulator on :8888
│   ├── .env.example       # local config template  ← copy to .env
│   └── Dockerfile / entrypoint.sh
├── robots/
│   ├── spot/              # Boston Dynamics Spot bridge (upstream)
│   └── tinybot/           # Tinybot firmware/hardware (upstream)
├── deploy.sh              # Cloud Run deploy with Secret Manager
├── scripts/pre-commit     # secret scanner git hook
└── UPSTREAM_README.md     # original upstream docs
```

## Secrets — read this first

| Where | How the key gets there | Committed? |
|---|---|---|
| Local dev | `agent/.env` (auto-loaded by `server.py`) | **No** – gitignored + dockerignored |
| Cloud Run | Secret Manager secret `gemini-api-key` → env var via `--set-secrets` | n/a |
| Logs | `server.py` prints a redacted config (`AIzaSy…XXXX`) | – |
| Git | `scripts/pre-commit` blocks `.env` files and `AIza…` strings | – |

Never pass the key on a command line (`--set-env-vars`, `--api_key`) — it ends up
in shell history, Cloud Build logs and the revision spec.

## Local development

```bash
cd agent
cp .env.example .env            # fill in GEMINI_API_KEY
./setup.sh                      # creates .venv and installs deps (once)

# terminal 1 – robot simulator
.venv/bin/python mock_robot_backend.py

# terminal 2 – agent server (reads .env automatically)
./run_agent.sh --robot_url http://127.0.0.1:8888
# open http://localhost:8080
```

Run tests:

```bash
cd agent && .venv/bin/python -m unittest discover -p '*_test.py'
```

## Deploy to Cloud Run

```bash
# one-time: push the key from agent/.env into Secret Manager + grant access
./deploy.sh --init-secret

# every time
./deploy.sh
```

Override defaults with env vars: `PROJECT_ID`, `REGION`, `SERVICE`, `SECRET_NAME`.

## Git setup (before the first commit)

```bash
git init
./scripts/install-hooks.sh      # installs the secret-scanning pre-commit hook
git add -A && git status        # verify agent/.env is NOT listed
```

## Models

| Purpose | Model |
|---|---|
| Reasoning / Live session | `models/gemini-robotics-er-2-streaming-preview` |
| Streaming TTS | `gemini-3.8-flash-tts` (voice *Puck*) |

See `GET /api/server_defaults` on a running server for the full selectable list.
