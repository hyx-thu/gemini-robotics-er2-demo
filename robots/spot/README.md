# spot-agent

Python project managed with `uv` for Boston Dynamics Spot SDK and Gemini API work.

## Environment

Use the project-local cache so `uv` does not write outside this workspace:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Run Python commands through the locked environment:

```bash
UV_CACHE_DIR=.uv-cache uv run python main.py
```

Run the named-place navigation app:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.navigation.cli --help
```

## Dependencies

Primary SDK packages:

- `bosdyn-client`
- `bosdyn-mission`
- `bosdyn-choreography-client`
- `google-genai`

Add future packages with:

```bash
UV_CACHE_DIR=.uv-cache uv add <package>
```

## Navigation App

The navigation app lives in `apps/navigation`. It stores named GraphNav keypoints in
`apps/navigation/keypoints.json` and can command Spot to navigate to a saved place
by name.

Set credentials with environment variables to avoid putting passwords in shell
history:

```bash
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=password
```

List saved places:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.navigation.cli list
```

Register a place manually with a GraphNav waypoint ID:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.navigation.cli register kitchen waypoint-id-123
```

Sync named waypoints from the map currently loaded on Spot:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.navigation.cli sync --hostname 192.168.80.3
```

Navigate to a saved place:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.navigation.cli go kitchen --hostname 192.168.80.3 --power-on --stand
```

## Manipulation App

The manipulation APIs live in `apps/manipulation`. They cover arm deployment,
Gemini-based object detection, 2D-to-3D projection with Spot hand depth, native
Spot image-pixel picking, force-change detection, gripper opening, and arm stow.

```bash
UV_CACHE_DIR=.uv-cache uv run python -m apps.manipulation.cli --help
```

## FastAPI Server

The HTTP API server lives in `apps/api` and exposes navigation, waypoint,
visualization, arm, gripper, detection, pick, force, and lease endpoints.

```bash
UV_CACHE_DIR=.uv-cache uv run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Open docs:

```text
http://127.0.0.1:8000/docs
```

The server can hold Spot's lease across calls. Use `/lease/take` or call
`/connect` with `take_lease: true`. It does not command sit or power-off when
the server stops.

The server also reads local credentials from `.config` if present. Copy
`.config.example` to `.config` and fill in the Spot password and Gemini API key:

```text
BOSDYN_CLIENT_PASSWORD=...
GEMINI_API_KEY=...
```

## Hydration Service

The drink delivery web app lives in `apps/hydration`. It is a separate Node app
serving a React UI and an order worker. The worker only controls Spot by calling
the FastAPI server.

Run FastAPI first, then start the hydration app with Node 18 or newer:

```bash
cd apps/hydration
npm start
```

Open:

```text
http://127.0.0.1:3000
```

Orders go through this sequence: navigate to `snack1`, detect the selected
drink with Gemini, open and rotate the gripper, approach and grasp the drink,
carry it to the selected waypoint, wait for delivery, then either serve the next
order or return to `home`.
