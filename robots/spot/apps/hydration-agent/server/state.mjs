import { EventEmitter } from "node:events";

export class HydrationState extends EventEmitter {
  constructor(model) {
    super();
    this.model = model;
    this.serviceRunning = false;
    this.paused = false;
    this.agentConnected = false;
    this.agentBusy = false;
    this.robotConnected = false;
    this.cameraStreaming = false;
    this.cameraFrameAt = null;
    this.cameraFramesSent = 0;
    this.cameraError = null;
    this.activeOrderId = null;
    this.lastError = null;
    this.nextOrderId = 1;
    this.nextVisualWatchId = 1;
    this.nextEntryId = 1;
    this.orders = [];
    this.messages = [];
    this.toolCalls = [];
    this.events = [];
    this.waypoints = [];
    this.visualWatches = [];
  }

  snapshot() {
    return {
      model: this.model,
      serviceRunning: this.serviceRunning,
      paused: this.paused,
      agentConnected: this.agentConnected,
      agentBusy: this.agentBusy,
      robotConnected: this.robotConnected,
      cameraStreaming: this.cameraStreaming,
      cameraFrameAt: this.cameraFrameAt,
      cameraFramesSent: this.cameraFramesSent,
      cameraError: this.cameraError,
      activeOrderId: this.activeOrderId,
      lastError: this.lastError,
      orders: this.orders,
      messages: this.messages.slice(-120),
      toolCalls: this.toolCalls.slice(-80),
      events: this.events.slice(-80),
      waypoints: this.waypoints,
      visualWatches: this.visualWatches.slice(-30),
    };
  }

  changed() {
    this.emit("changed", this.snapshot());
  }

  setWaypoints(waypoints) {
    this.waypoints = [...new Set(waypoints)].sort((a, b) => a.localeCompare(b));
    this.changed();
  }

  createOrder(drink, destination, source = "ui") {
    const now = new Date().toISOString();
    const order = {
      id: this.nextOrderId++,
      drinkId: drink.id,
      drinkName: drink.name,
      detectInstruction: drink.detectInstruction,
      destination,
      source,
      status: "pending",
      note: null,
      createdAt: now,
      updatedAt: now,
    };
    this.orders.push(order);
    this.log("order", `Order #${order.id} queued: ${order.drinkName} to ${destination}`);
    return order;
  }

  updateOrder(id, status, note = null) {
    const order = this.orders.find((candidate) => candidate.id === Number(id));
    if (!order) throw new Error(`Unknown order #${id}.`);
    order.status = status;
    order.note = note || order.note;
    order.updatedAt = new Date().toISOString();
    if (status === "running" && !order.startedAt) order.startedAt = order.updatedAt;
    if (["finished", "failed", "cancelled"].includes(status)) order.finishedAt = order.updatedAt;
    this.log("order", `Order #${order.id} marked ${status}${note ? `: ${note}` : ""}`);
    return order;
  }

  createVisualWatch({ subject, condition, responseMessage, requiresPriorPresence }) {
    const now = new Date().toISOString();
    const watch = {
      id: this.nextVisualWatchId++,
      subject,
      condition,
      responseMessage,
      requiresPriorPresence,
      armed: !requiresPriorPresence,
      consecutiveMatches: 0,
      status: "active",
      observation: null,
      createdAt: now,
      updatedAt: now,
    };
    this.visualWatches.push(watch);
    this.log("watch", `Visual watch #${watch.id} active: ${watch.condition}`);
    return watch;
  }

  evaluateVisualWatch(id, { subjectVisible, conditionMet, observation }) {
    const watch = this.visualWatches.find((candidate) => candidate.id === Number(id));
    if (!watch) throw new Error(`Unknown visual watch #${id}.`);
    if (watch.status !== "active") throw new Error(`Visual watch #${id} is ${watch.status}.`);

    if (subjectVisible) watch.armed = true;
    const acceptedMatch = watch.armed && conditionMet;
    watch.consecutiveMatches = acceptedMatch ? watch.consecutiveMatches + 1 : 0;
    watch.observation = observation || null;
    watch.updatedAt = new Date().toISOString();

    if (watch.consecutiveMatches >= 2) {
      watch.status = "triggered";
      watch.triggeredAt = watch.updatedAt;
      this.addMessage("assistant", watch.responseMessage, { proactive: true, visualWatchId: watch.id });
      this.log("watch", `Visual watch #${watch.id} triggered: ${watch.observation || watch.condition}`);
    } else {
      this.changed();
    }
    return watch;
  }

  cancelVisualWatch(id) {
    const watch = this.visualWatches.find((candidate) => candidate.id === Number(id));
    if (!watch) throw new Error(`Unknown visual watch #${id}.`);
    if (watch.status === "active") {
      watch.status = "cancelled";
      watch.updatedAt = new Date().toISOString();
      this.log("watch", `Visual watch #${watch.id} cancelled`);
    }
    return watch;
  }

  addMessage(role, text, meta = {}) {
    const message = {
      id: this.nextEntryId++,
      role,
      text,
      at: new Date().toISOString(),
      ...meta,
    };
    this.messages.push(message);
    this.changed();
    return message;
  }

  appendMessage(id, text) {
    const message = this.messages.find((candidate) => candidate.id === id);
    if (!message) return;
    message.text += text;
    this.changed();
  }

  addToolCall(name, args) {
    const call = {
      id: this.nextEntryId++,
      name,
      args,
      status: "running",
      startedAt: new Date().toISOString(),
    };
    this.toolCalls.push(call);
    this.changed();
    return call;
  }

  finishToolCall(call, result, error = null) {
    call.status = error ? "failed" : "finished";
    call.finishedAt = new Date().toISOString();
    if (error) call.error = error;
    else call.result = summarize(result);
    this.changed();
  }

  log(type, message) {
    this.events.push({ id: this.nextEntryId++, type, message, at: new Date().toISOString() });
    this.changed();
  }
}

function summarize(result) {
  if (!result || typeof result !== "object") return result;
  const summary = {};
  for (const key of [
    "ok", "status", "reached_goal", "arrived", "at_goal", "triggered", "reason",
    "order_id", "detection_id", "pending_orders", "percentage", "connected", "holding_lease",
  ]) {
    if (key in result) summary[key] = result[key];
  }
  if (result.pose) summary.pose = result.pose;
  return Object.keys(summary).length ? summary : { ok: true };
}
