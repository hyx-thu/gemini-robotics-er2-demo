# Gemini Live Hydration Agent

This is the agent-driven counterpart to `apps/hydration`. Gemini Live chooses and calls guarded tools; the Node server executes those tools through the Spot FastAPI service.

## Run

From this directory:

```bash
npm install
npm run build
npm start
```

Open `http://127.0.0.1:3001`.

The app reads the repository-level `.config`. Supported settings:

```ini
GEMINI_API_KEY=...
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
FASTAPI_BASE_URL=http://127.0.0.1:8000
HYDRATION_AGENT_PORT=3001
HYDRATION_AGENT_AUTO_START=false
```

Spot credentials remain server-side. The browser never receives the API key or robot password.

## Behavior

- Structured orders enter a queue and are dispatched to the Live agent one at a time.
- Free-form chat can create orders, inspect state, or call robot tools for debugging.
- While Spot is connected, the server sends the gripper color camera to Gemini Live at 1 FPS for general visual questions.
- Each model function call maps to a validated FastAPI operation.
- Tool calls are serialized. A failed tool skips any remaining calls in that batch, marks the active order failed, and pauses service.
- Pause and stop call `/actions/stop` to interrupt tracked FastAPI actions.
- `GET /api/prompt` returns the active generated system prompt without credentials.
- Context compression and Live session resumption keep the audio/video session available across periodic WebSocket reconnects.
