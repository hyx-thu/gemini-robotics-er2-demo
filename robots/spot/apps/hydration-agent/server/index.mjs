import express from "express";
import { existsSync } from "node:fs";
import path from "node:path";
import { appDirectory, loadConfig } from "./config.mjs";
import { DRINKS, findDrink, isDeliveryWaypoint } from "./catalog.mjs";
import { HydrationState } from "./state.mjs";
import { FastApiClient } from "./fastapi.mjs";
import { HydrationTools } from "./tools.mjs";
import { GeminiHydrationAgent } from "./agent.mjs";

const config = loadConfig();
const port = Number(config.HYDRATION_AGENT_PORT || 3001);
const host = config.HYDRATION_AGENT_HOST || "127.0.0.1";
const fastApiBaseUrl = config.FASTAPI_BASE_URL || "http://127.0.0.1:8000";
const model = config.GEMINI_LIVE_MODEL || "gemini-3.1-flash-live-preview";

const state = new HydrationState(model);
const fastApi = new FastApiClient(fastApiBaseUrl);
const tools = new HydrationTools({ fastApi, state, config });
const agent = new GeminiHydrationAgent({ state, tools, config });
const app = express();
const eventClients = new Set();

app.use(express.json({ limit: "1mb" }));

app.get("/api/bootstrap", async (_req, res) => {
  try {
    await tools.refreshWaypoints();
  } catch (error) {
    state.log("warning", `Waypoint refresh failed: ${errorMessage(error)}`);
  }
  res.json({
    drinks: DRINKS,
    waypoints: state.waypoints,
    model,
    fastApiBaseUrl,
    hasGeminiKey: Boolean(config.GEMINI_API_KEY),
    hasSpotPassword: Boolean(config.BOSDYN_CLIENT_PASSWORD),
  });
});

app.get("/api/state", (_req, res) => res.json(state.snapshot()));

app.get("/api/events", (req, res) => {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  res.write(`data: ${JSON.stringify(state.snapshot())}\n\n`);
  eventClients.add(res);
  const keepAlive = setInterval(() => res.write(": keep-alive\n\n"), 15_000);
  req.on("close", () => {
    clearInterval(keepAlive);
    eventClients.delete(res);
  });
});

state.on("changed", (snapshot) => {
  const event = `data: ${JSON.stringify(snapshot)}\n\n`;
  for (const client of eventClients) client.write(event);
});

app.post("/api/orders", asyncHandler(async (req, res) => {
  const drink = findDrink(String(req.body.drinkId || ""));
  if (!drink) return res.status(400).json({ error: "Unknown drink." });
  const destination = String(req.body.destination || "").trim();
  if (!isDeliveryWaypoint(destination, state.waypoints)) {
    return res.status(400).json({ error: "Choose a current delivery waypoint other than home or snack1." });
  }
  const order = state.createOrder(drink, destination);
  agent.queueOrder(order);
  res.status(201).json({ order, state: state.snapshot() });
}));

app.post("/api/chat", asyncHandler(async (req, res) => {
  const text = String(req.body.message || "").trim();
  if (!text) return res.status(400).json({ error: "Message is required." });
  state.addMessage("user", text);
  await agent.connect();
  agent.queueChat(text);
  res.status(202).json(state.snapshot());
}));

app.post("/api/service/start", asyncHandler(async (_req, res) => res.json(await agent.start())));
app.post("/api/service/resume", asyncHandler(async (_req, res) => res.json(await agent.resume())));
app.post("/api/service/pause", asyncHandler(async (_req, res) => res.json(await agent.pause())));
app.post("/api/service/stop", asyncHandler(async (_req, res) => res.json(await agent.stop())));
app.post("/api/service/reset-session", asyncHandler(async (_req, res) => res.json(await agent.resetSession())));

app.post("/api/service/connect-robot", asyncHandler(async (_req, res) => {
  const trace = state.addToolCall("connect_robot", {});
  try {
    const result = await tools.execute("connect_robot", {});
    state.finishToolCall(trace, result);
    res.json({ result, state: state.snapshot() });
  } catch (error) {
    state.finishToolCall(trace, null, errorMessage(error));
    throw error;
  }
}));

app.post("/api/service/stop-actions", asyncHandler(async (_req, res) => {
  const result = await fastApi.stopRobot();
  state.log("service", "Immediate robot stop requested by operator");
  res.json({ result, state: state.snapshot() });
}));

app.get("/api/prompt", (_req, res) => res.json({ prompt: agent.systemPrompt() }));

app.get("/api/camera/gripper", asyncHandler(async (_req, res) => {
  const response = await fetch(`${fastApiBaseUrl}/images/hand_color_image?quality_percent=75`);
  if (!response.ok) throw new Error(await response.text());
  res.set("content-type", response.headers.get("content-type") || "image/jpeg");
  res.set("cache-control", "no-store");
  res.send(Buffer.from(await response.arrayBuffer()));
}));

const distDirectory = path.join(appDirectory, "dist");
if (existsSync(distDirectory)) {
  app.use(express.static(distDirectory));
  app.get("/{*path}", (_req, res) => res.sendFile(path.join(distDirectory, "index.html")));
} else {
  app.get("/", (_req, res) => res.status(503).send("Frontend is not built. Run npm run build."));
}

app.use((error, _req, res, _next) => {
  const message = errorMessage(error);
  state.lastError = message;
  state.log("error", message);
  res.status(500).json({ error: message });
});

const server = app.listen(port, host, () => {
  state.log("server", `Gemini Live hydration agent listening on http://${host}:${port}`);
});

if (config.HYDRATION_AGENT_AUTO_START === "true") {
  agent.start().catch((error) => state.log("error", `Auto-start failed: ${errorMessage(error)}`));
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    try {
      agent.session?.close();
    } finally {
      server.close(() => process.exit(0));
    }
  });
}

function asyncHandler(handler) {
  return (req, res, next) => Promise.resolve(handler(req, res, next)).catch(next);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
