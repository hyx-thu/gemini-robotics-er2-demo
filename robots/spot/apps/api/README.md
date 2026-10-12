# Spot FastAPI Server

Start the API server:

```bash
export SPOT_HOSTNAME=192.168.80.3
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=password
export GEMINI_API_KEY=...

UV_CACHE_DIR=.uv-cache uv run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

The server keeps one Spot connection in process. If you call `/lease/take` or start
with `SPOT_TAKE_LEASE=true`, it keeps the lease alive until `/lease/release` or
server shutdown. It does not command sit or power-off on shutdown.

Common calls:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/lease/take
curl http://127.0.0.1:8000/waypoints
curl -X POST http://127.0.0.1:8000/navigate \
  -H 'content-type: application/json' \
  -d '{"name":"peng-desk","take_lease":true}'
curl -X POST http://127.0.0.1:8000/arm/deploy \
  -H 'content-type: application/json' \
  -d '{"take_lease":true}'
curl -X POST http://127.0.0.1:8000/gripper/open \
  -H 'content-type: application/json' \
  -d '{"take_lease":true}'
curl -X POST http://127.0.0.1:8000/detect \
  -H 'content-type: application/json' \
  -d '{"instruction":"the red cup"}'
curl -X POST http://127.0.0.1:8000/pick \
  -H 'content-type: application/json' \
  -d '{"instruction":"the red cup","take_lease":true}'
curl -X POST http://127.0.0.1:8000/arm/stow \
  -H 'content-type: application/json' \
  -d '{"take_lease":true}'
```
