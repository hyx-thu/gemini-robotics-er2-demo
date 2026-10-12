import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, "public");

const config = loadConfig();
const FASTAPI_BASE_URL = config.FASTAPI_BASE_URL || "http://127.0.0.1:8000";
const PORT = Number(config.HYDRATION_PORT || 3000);

const DRINKS = [
  {
    id: "blue-hint-water",
    name: "Blue Hint Water",
    subtitle: "Blackberry flavor",
    detectInstruction: "middle of blue drink can",
    accent: "#2f72c7",
  },
  {
    id: "red-hint-water",
    name: "Red Hint Water",
    subtitle: "Lemon flavor",
    detectInstruction: "middle of red drink can",
    accent: "#c94438",
  },
  {
    id: "monster-energy",
    name: "Monster Energy Drink",
    subtitle: "Energy drink",
    detectInstruction: "black drink can",
    accent: "#2e8b42",
  },
];

const state = {
  serviceRunning: false,
  paused: false,
  workerActive: false,
  currentOrderId: null,
  orders: [],
  events: [],
  lastError: null,
  nextOrderId: 1,
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname.startsWith("/api/")) {
      await handleApi(req, res, url);
      return;
    }
    await serveStatic(req, res, url);
  } catch (error) {
    sendJson(res, 500, { error: errorMessage(error) });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  logEvent("server", `Hydration service listening on http://127.0.0.1:${PORT}`);
  if (config.HYDRATION_AUTO_CONNECT === "true") {
    connectFastApi().catch((error) => {
      state.lastError = errorMessage(error);
      logEvent("error", `Auto-connect failed: ${state.lastError}`);
    });
  }
});

async function handleApi(req, res, url) {
  if (req.method === "GET" && url.pathname === "/api/camera/gripper") {
    await proxyFastApiImage(res, "/images/hand_color_image?quality_percent=75");
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/bootstrap") {
    sendJson(res, 200, {
      drinks: DRINKS,
      waypoints: await deliveryWaypoints(),
      config: {
        fastApiBaseUrl: FASTAPI_BASE_URL,
        hasSpotPassword: Boolean(config.BOSDYN_CLIENT_PASSWORD),
        hasGeminiKey: Boolean(config.GEMINI_API_KEY),
      },
    });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/orders") {
    sendJson(res, 200, publicState());
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/orders") {
    const body = await readJson(req);
    const drink = DRINKS.find((item) => item.id === body.drinkId);
    if (!drink) {
      sendJson(res, 400, { error: "Unknown drink." });
      return;
    }
    const destination = String(body.destination || "").trim();
    if (!destination || destination === "home" || destination === "snack1") {
      sendJson(res, 400, { error: "Choose a delivery waypoint other than home or snack1." });
      return;
    }
    const order = {
      id: state.nextOrderId++,
      drinkId: drink.id,
      drinkName: drink.name,
      detectInstruction: drink.detectInstruction,
      destination,
      status: "pending",
      steps: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    state.orders.push(order);
    logEvent("order", `Order ${order.id} queued: ${drink.name} to ${destination}`);
    startWorker();
    sendJson(res, 201, publicState());
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/service/start") {
    state.serviceRunning = true;
    state.paused = false;
    logEvent("service", "Service started");
    startWorker();
    sendJson(res, 200, publicState());
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/service/pause") {
    state.paused = true;
    logEvent("service", "Service paused");
    sendJson(res, 200, publicState());
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/service/resume") {
    state.serviceRunning = true;
    state.paused = false;
    logEvent("service", "Service resumed");
    startWorker();
    sendJson(res, 200, publicState());
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/service/connect") {
    const result = await connectFastApi();
    sendJson(res, 200, { ...publicState(), connect: result });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/service/stop-actions") {
    state.serviceRunning = false;
    state.paused = true;
    const result = await fastApi("/actions/stop", {
      take_lease: true,
      freeze_arm: true,
    });
    logEvent("service", "Stop actions requested");
    sendJson(res, 200, { ...publicState(), stop: result });
    return;
  }

  sendJson(res, 404, { error: "Not found." });
}

async function proxyFastApiImage(res, path) {
  const response = await fetch(`${FASTAPI_BASE_URL}${path}`);
  if (!response.ok) {
    sendJson(res, response.status, { error: await response.text() });
    return;
  }
  const contentType = response.headers.get("content-type") || "image/jpeg";
  const body = Buffer.from(await response.arrayBuffer());
  res.writeHead(200, {
    "content-type": contentType,
    "cache-control": "no-store",
  });
  res.end(body);
}

async function serveStatic(req, res, url) {
  const filePath = url.pathname === "/"
    ? path.join(publicDir, "index.html")
    : path.join(publicDir, path.normalize(url.pathname));
  if (!filePath.startsWith(publicDir) || !existsSync(filePath)) {
    sendJson(res, 404, { error: "Not found." });
    return;
  }
  const content = await readFile(filePath);
  const contentType = filePath.endsWith(".html") ? "text/html; charset=utf-8" : "text/plain; charset=utf-8";
  res.writeHead(200, { "content-type": contentType });
  res.end(content);
}

function startWorker() {
  if (state.workerActive) return;
  state.workerActive = true;
  setTimeout(() => workerLoop().catch((error) => {
    state.workerActive = false;
    state.lastError = errorMessage(error);
    logEvent("error", state.lastError);
  }), 0);
}

async function workerLoop() {
  while (state.serviceRunning) {
    await waitIfPaused();
    const order = state.orders.find((item) => item.status === "pending");
    if (!order) {
      await idleReturnHome();
      break;
    }
    state.currentOrderId = order.id;
    await processOrder(order);
    state.currentOrderId = null;
  }
  state.workerActive = false;
}

async function processOrder(order) {
  order.status = "running";
  order.startedAt = new Date().toISOString();
  updateOrder(order);
  try {
    await runStep(order, "Navigate to drink station", () => fastApi("/navigate", {
      name: "snack1",
      take_lease: true,
      power_on: true,
      stand: true,
      timeout: 180,
    }));
    await runStep(order, "Back up 0.3 m from drink station", () => fastApi("/teleop/velocity", {
      v_x: -0.25,
      v_y: 0,
      v_rot: 0,
      duration: 1.2,
      take_lease: true,
      power_on: false,
      stand: false,
    }));
    await runStep(order, "Move arm to carry pose", () => fastApi("/arm/carry", { take_lease: true, timeout: 10 }));
    const scene = await detectDrink(order, "Detect");
    await runStep(order, "Open gripper to 60%", () => fastApi("/gripper/open", {
      open_fraction: 0.6,
      max_vel: 0.5,
      max_acc: 1.0,
      take_lease: true,
      timeout: 5,
    }));
    await runStep(order, "Rotate gripper 90 degrees clockwise", () => fastApi("/arm/camera-roll", {
      direction: "clockwise",
      angle_rad: Math.PI / 2,
      seconds: 0.7,
      take_lease: true,
      timeout: 3,
    }));
    await runStep(order, "Whole-body approach detected drink", () => fastApi("/arm/approach-whole-body", {
      pose: scene.pose,
      standoff_m: 0,
      max_step_m: 0.8,
      seconds: 2.0,
      take_lease: true,
      timeout: 8,
    }));
    await runStep(order, "Close gripper slowly", () => fastApi("/gripper/close", {
      max_vel: 0.25,
      max_acc: 0.5,
      take_lease: true,
      timeout: 10,
    }));
    await runStep(order, "Stow arm before navigation", () => fastApi("/arm/stow", {
      take_lease: true,
      timeout: 20,
    }));
    await runStep(order, `Navigate to ${order.destination}`, () => fastApi("/navigate", {
      name: order.destination,
      take_lease: true,
      power_on: true,
      stand: true,
      timeout: 180,
    }));
    await runStep(order, "Move arm to carry pose for delivery", () => fastApi("/arm/carry", {
      take_lease: true,
      timeout: 10,
    }));
    await runStep(order, "Wait for delivery handoff", () => fastApi("/delivery/wait", {
      monitor_sec: 30,
      upward_threshold_m: 0.02,
      sample_interval: 0.1,
      open_duration_sec: 3,
      take_lease: true,
      gripper_timeout: 5,
      stow_timeout: 10,
    }));
    order.status = "finished";
    order.finishedAt = new Date().toISOString();
    logEvent("order", `Order ${order.id} finished`);
  } catch (error) {
    order.status = "failed";
    order.error = errorMessage(error);
    state.paused = true;
    state.lastError = order.error;
    logEvent("error", `Order ${order.id} failed: ${order.error}`);
    logEvent("service", "Service paused after order failure");
  } finally {
    updateOrder(order);
  }
}

async function detectDrink(order, prefix) {
  return runStep(order, `${prefix} ${order.drinkName}`, () => fastApi("/detect", {
    instruction: order.detectInstruction,
    api_key: config.GEMINI_API_KEY || null,
    include_point_cloud: false,
  }));
}

async function idleReturnHome() {
  const hasPending = state.orders.some((item) => item.status === "pending");
  if (hasPending || state.paused) return;
  try {
    logEvent("service", "No pending orders; returning home");
    const homeResult = await fastApi("/navigate", {
      name: "home",
      take_lease: true,
      power_on: true,
      stand: true,
      timeout: 180,
    });
    validateStepResult("Return home", homeResult);
    logEvent("service", "Returned home; sitting to save power");
    const sitResult = await fastApi("/sit", {
      take_lease: true,
      timeout: 15,
    });
    validateStepResult("Sit after return home", sitResult);
  } catch (error) {
    state.lastError = errorMessage(error);
    logEvent("error", `Return home failed: ${state.lastError}`);
  }
}

async function runStep(order, label, action) {
  await waitIfPaused();
  const step = {
    label,
    status: "running",
    startedAt: new Date().toISOString(),
  };
  order.steps.push(step);
  updateOrder(order);
  logEvent("step", `Order ${order.id}: ${label}`);
  try {
    const result = await action();
    validateStepResult(label, result);
    step.status = "finished";
    step.finishedAt = new Date().toISOString();
    step.result = summarizeResult(result);
    updateOrder(order);
    return result;
  } catch (error) {
    step.status = "failed";
    step.finishedAt = new Date().toISOString();
    step.error = errorMessage(error);
    updateOrder(order);
    throw error;
  }
}

function validateStepResult(label, result) {
  if (!result || typeof result !== "object") return;

  if (result.reached_goal === false) {
    throw new Error(`${label} failed: ${result.status || "goal not reached"}`);
  }
  if (typeof result.status === "string" && /STUCK|CANCELLED|TIMED_OUT|LOST|FAILED|ERROR/.test(result.status)) {
    throw new Error(`${label} failed: ${result.status}`);
  }
  if (result.arrived === false) {
    throw new Error(`${label} failed: arm did not arrive`);
  }
  if (result.at_goal === false) {
    throw new Error(`${label} failed: gripper did not reach goal`);
  }
  if ((label.startsWith("Detect ") || label.startsWith("Re-detect ")) && !result.pose) {
    throw new Error(`${label} failed: no 3D pose returned (${JSON.stringify(result.errors || {})})`);
  }
  if (label === "Wait for delivery handoff" && result.triggered !== true) {
    throw new Error(`${label} failed: ${result.reason || "handoff not detected"}`);
  }
  if (Array.isArray(result.commands)) {
    const failedCommand = result.commands.find((command) => command.arrived === false || command.at_goal === false);
    if (failedCommand) {
      throw new Error(`${label} failed: command did not complete`);
    }
  }
}

async function waitIfPaused() {
  while (state.serviceRunning && state.paused) {
    await sleep(400);
  }
  if (!state.serviceRunning) {
    throw new Error("Service stopped.");
  }
}

async function deliveryWaypoints() {
  try {
    const items = await fastApiGet("/waypoints");
    return items
      .map((item) => item.name)
      .filter((name) => name && name !== "home" && name !== "snack1")
      .sort((a, b) => a.localeCompare(b));
  } catch {
    return [];
  }
}

async function connectFastApi() {
  if (!config.BOSDYN_CLIENT_PASSWORD) {
    throw new Error("Missing BOSDYN_CLIENT_PASSWORD in .config.");
  }
  const result = await fastApi("/connect", {
    hostname: config.SPOT_HOSTNAME || config.BOSDYN_CLIENT_HOSTNAME || "192.168.80.3",
    username: config.BOSDYN_CLIENT_USERNAME || "user",
    password: config.BOSDYN_CLIENT_PASSWORD,
    take_lease: true,
  });
  logEvent("service", "Connected to Spot and took lease");
  return result;
}

async function fastApi(pathname, body) {
  const response = await fetch(`${FASTAPI_BASE_URL}${pathname}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return responseJson(response);
}

async function fastApiGet(pathname) {
  const response = await fetch(`${FASTAPI_BASE_URL}${pathname}`);
  return responseJson(response);
}

async function responseJson(response) {
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    throw new Error(data?.detail || data?.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(data));
}

function publicState() {
  return {
    serviceRunning: state.serviceRunning,
    paused: state.paused,
    workerActive: state.workerActive,
    currentOrderId: state.currentOrderId,
    lastError: state.lastError,
    orders: state.orders,
    events: state.events.slice(-80),
  };
}

function updateOrder(order) {
  order.updatedAt = new Date().toISOString();
}

function logEvent(type, message) {
  state.events.push({ type, message, at: new Date().toISOString() });
}

function summarizeResult(result) {
  if (!result || typeof result !== "object") return result;
  const summary = {};
  for (const key of ["status", "reached_goal", "command_id", "arrived", "at_goal", "triggered", "reason", "steps"]) {
    if (key in result) summary[key] = result[key];
  }
  if (result.pose) summary.pose = result.pose;
  return Object.keys(summary).length ? summary : { ok: true };
}

function loadConfig() {
  const paths = [
    path.join(__dirname, ".config"),
    path.join(__dirname, "..", "..", ".config"),
  ];
  const values = {};
  for (const configPath of paths) {
    if (!existsSync(configPath)) continue;
    const content = readFileSync(configPath, "utf8");
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const index = line.indexOf("=");
      const key = line.slice(0, index).trim();
      const value = line.slice(index + 1).trim().replace(/^["']|["']$/g, "");
      if (key) values[key] = value;
    }
  }
  return { ...values, ...process.env };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
