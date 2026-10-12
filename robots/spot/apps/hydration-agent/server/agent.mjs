import { GoogleGenAI, Modality } from "@google/genai";
import { DRINKS } from "./catalog.mjs";
import { buildSystemPrompt } from "./prompt.mjs";
import { TOOL_DECLARATIONS } from "./tools.mjs";

export class GeminiHydrationAgent {
  constructor({ state, tools, config }) {
    this.state = state;
    this.tools = tools;
    this.config = config;
    this.ai = null;
    this.session = null;
    this.connectPromise = null;
    this.pendingInputs = [];
    this.currentInputKind = null;
    this.currentAssistantMessageId = null;
    this.messageChain = Promise.resolve();
    this.continuationCounts = new Map();
    this.pumpScheduled = false;
    this.sessionGeneration = 0;
    this.sessionResumptionHandle = null;
    this.reconnectTimer = null;
    this.cameraInterval = null;
    this.cameraFrameInFlight = false;
    this.cameraIntervalMs = Math.max(1_000, Number(config.HYDRATION_AGENT_CAMERA_INTERVAL_MS || 1_000));
    this.lastCameraHealthCheck = 0;
    this.toolExecutionActive = 0;
    this.visualWatchTickQueued = false;
    this.lastVisualWatchTick = 0;
    this.visualWatchTurnTimer = null;
    this.visualWatchIntervalMs = Math.max(
      1_000,
      Number(config.HYDRATION_AGENT_VISUAL_WATCH_INTERVAL_MS || 2_000),
    );
    this.visualWatchTurnTimeoutMs = Math.max(
      5_000,
      Number(config.HYDRATION_AGENT_VISUAL_WATCH_TURN_TIMEOUT_MS || 10_000),
    );
  }

  systemPrompt() {
    return buildSystemPrompt({
      drinks: DRINKS,
      waypoints: this.state.waypoints,
      toolDescriptions: this.tools.descriptions(),
    });
  }

  async connect() {
    if (this.session) return this.session;
    if (this.connectPromise) return this.connectPromise;
    if (!this.config.GEMINI_API_KEY) throw new Error("GEMINI_API_KEY is missing from .config.");

    this.connectPromise = this.openSession();
    try {
      return await this.connectPromise;
    } finally {
      this.connectPromise = null;
    }
  }

  async openSession() {
    const generation = ++this.sessionGeneration;
    try {
      try {
        await this.tools.refreshWaypoints();
      } catch (error) {
        this.state.log("warning", `Could not refresh waypoints for prompt: ${errorMessage(error)}`);
      }
      try {
        await this.tools.getRobotStatus();
      } catch (error) {
        this.state.log("warning", `Could not read Spot status for Live vision: ${errorMessage(error)}`);
      }

      this.ai = new GoogleGenAI({ apiKey: this.config.GEMINI_API_KEY });
      const session = await this.ai.live.connect({
        model: this.state.model,
        config: {
          responseModalities: [Modality.AUDIO],
          outputAudioTranscription: {},
          systemInstruction: this.systemPrompt(),
          tools: [{ functionDeclarations: TOOL_DECLARATIONS }],
          contextWindowCompression: { slidingWindow: {} },
          sessionResumption: this.sessionResumptionHandle
            ? { handle: this.sessionResumptionHandle }
            : {},
          thinkingConfig: { thinkingLevel: "low" },
        },
        callbacks: {
          onopen: () => {
            if (generation !== this.sessionGeneration) return;
            this.state.agentConnected = true;
            this.state.lastError = null;
            this.state.log("agent", `Gemini Live connected: ${this.state.model}`);
            this.startCameraStream();
          },
          onmessage: (message) => {
            if (generation !== this.sessionGeneration) return;
            this.messageChain = this.messageChain
              .then(() => this.handleMessage(message, generation))
              .catch((error) => this.handleAgentError(error));
          },
          onerror: (error) => {
            if (generation === this.sessionGeneration) this.handleAgentError(error);
          },
          onclose: (event) => {
            if (generation !== this.sessionGeneration) return;
            this.stopCameraStream();
            this.clearVisualWatchTurnTimer();
            if (this.currentInputKind === "visual-watch") this.visualWatchTickQueued = false;
            this.currentInputKind = null;
            this.session = null;
            this.state.agentConnected = false;
            if (!this.toolExecutionActive) this.state.agentBusy = false;
            this.state.log("agent", `Gemini Live closed${event?.reason ? `: ${event.reason}` : ""}`);
            this.scheduleReconnect();
          },
        },
      });
      this.session = session;
      this.state.agentConnected = true;
      this.state.changed();
      return session;
    } catch (error) {
      this.ai = null;
      this.session = null;
      this.state.agentConnected = false;
      this.state.lastError = errorMessage(error);
      this.state.changed();
      throw error;
    }
  }

  async start() {
    this.state.serviceRunning = true;
    this.state.paused = false;
    await this.connect();
    this.state.log("service", "Agent service started");
    this.schedulePump();
    return this.state.snapshot();
  }

  async resume() {
    this.state.serviceRunning = true;
    this.state.paused = false;
    await this.connect();
    this.state.log("service", "Agent service resumed");
    this.schedulePump();
    return this.state.snapshot();
  }

  async pause() {
    this.state.paused = true;
    this.state.log("service", "Agent service paused; stopping active robot actions");
    try {
      await this.tools.fastApi.stopRobot();
    } catch (error) {
      this.state.log("warning", `Robot stop while pausing: ${errorMessage(error)}`);
    }
    return this.state.snapshot();
  }

  async stop() {
    this.state.serviceRunning = false;
    this.state.paused = true;
    this.state.log("service", "Agent service stopped; stopping active robot actions");
    try {
      await this.tools.fastApi.stopRobot();
    } catch (error) {
      this.state.log("warning", `Robot stop: ${errorMessage(error)}`);
    }
    return this.state.snapshot();
  }

  async resetSession() {
    this.stopCameraStream();
    this.clearVisualWatchTurnTimer();
    this.clearReconnectTimer();
    this.sessionGeneration += 1;
    this.sessionResumptionHandle = null;
    if (this.session) {
      try {
        this.session.close();
      } catch {
        // A closed Live socket needs no further cleanup.
      }
    }
    this.session = null;
    this.ai = null;
    this.state.agentConnected = false;
    this.state.agentBusy = false;
    this.currentAssistantMessageId = null;
    this.currentInputKind = null;
    this.visualWatchTickQueued = false;
    this.state.log("agent", "Gemini Live session reset");
    if (this.state.serviceRunning) await this.connect();
    this.schedulePump();
    return this.state.snapshot();
  }

  queueOrder(order) {
    this.state.log("agent", `Order #${order.id} is ready for agent dispatch`);
    this.schedulePump();
  }

  queueChat(text) {
    this.pendingInputs.unshift({ kind: "chat", text });
    this.schedulePump();
  }

  schedulePump() {
    if (this.pumpScheduled) return;
    this.pumpScheduled = true;
    setTimeout(() => {
      this.pumpScheduled = false;
      this.pump().catch((error) => this.handleAgentError(error));
    }, 0);
  }

  async pump() {
    if (this.state.agentBusy || this.toolExecutionActive) return;
    if (!this.session) await this.connect();

    const input = this.pendingInputs.shift();
    if (input) {
      this.sendText(input.text, input.kind);
      return;
    }

    if (!this.state.serviceRunning || this.state.paused) return;
    if (this.state.activeOrderId) {
      const activeOrder = this.state.orders.find((candidate) => candidate.id === this.state.activeOrderId);
      if (activeOrder && !["finished", "failed", "cancelled"].includes(activeOrder.status)) {
        this.sendText(this.activeOrderRecoveryPrompt(activeOrder), "order-recovery");
        return;
      }
    }
    const order = this.state.orders.find((candidate) => candidate.status === "pending");
    if (!order) return;

    this.state.activeOrderId = order.id;
    this.state.addMessage(
      "system",
      `Dispatching order #${order.id}: ${order.drinkName} to ${order.destination}.`,
      { orderId: order.id },
    );
    this.sendText(
      `Execute hydration order #${order.id}. Drink ID: ${order.drinkId}. Drink: ${order.drinkName}. ` +
      `Use detection instruction exactly: "${order.detectInstruction}". Delivery waypoint: ${order.destination}. ` +
      "Follow the normal order workflow completely, update the tracked order status, and stop immediately on any error.",
      "order",
    );
  }

  sendText(text, kind = "chat") {
    if (!this.session) throw new Error("Gemini Live session is not connected.");
    this.state.agentBusy = true;
    this.currentInputKind = kind;
    this.currentAssistantMessageId = null;
    this.state.changed();
    this.session.sendRealtimeInput({ text });
    if (kind === "visual-watch") {
      this.clearVisualWatchTurnTimer();
      this.visualWatchTurnTimer = setTimeout(
        () => this.recoverStuckVisualWatchTurn(),
        this.visualWatchTurnTimeoutMs,
      );
    }
  }

  async handleMessage(message, generation) {
    const resumption = message.sessionResumptionUpdate;
    if (resumption?.resumable && resumption.newHandle) {
      this.sessionResumptionHandle = resumption.newHandle;
    }
    if (message.goAway?.timeLeft) {
      this.state.log("agent", `Gemini Live connection renewal in ${message.goAway.timeLeft}`);
    }

    const transcription = message.serverContent?.outputTranscription?.text;
    const modelText = extractModelText(message);
    const text = transcription || modelText;
    if (text && this.currentInputKind !== "visual-watch") {
      if (!this.currentAssistantMessageId) {
        this.currentAssistantMessageId = this.state.addMessage("assistant", "").id;
      }
      this.state.appendMessage(this.currentAssistantMessageId, text);
    }

    if (message.toolCall?.functionCalls?.length) {
      await this.handleToolCalls(message.toolCall.functionCalls, generation, this.currentInputKind);
    }

    if (message.serverContent?.turnComplete) {
      this.clearVisualWatchTurnTimer();
      if (this.currentInputKind === "visual-watch") this.visualWatchTickQueued = false;
      this.currentInputKind = null;
      this.currentAssistantMessageId = null;
      this.state.agentBusy = false;
      this.finishActiveTurn();
      this.state.changed();
      this.schedulePump();
    }

    if (message.serverContent?.interrupted) {
      this.state.log("agent", "Gemini response interrupted");
    }
  }

  async handleToolCalls(functionCalls, generation, inputKind) {
    this.toolExecutionActive += 1;
    const responses = [];
    let failure = null;
    const evaluatedWatches = new Set();
    try {
      for (const functionCall of functionCalls) {
        const args = functionCall.args || {};
        const trace = this.state.addToolCall(functionCall.name, args);
        let result;
        const watchId = Number(args.watch_id);
        const visualTickViolation = inputKind === "visual-watch" && (
          functionCall.name !== "evaluate_visual_watch" || evaluatedWatches.has(watchId)
        );
        const externalEvaluation = inputKind !== "visual-watch" && functionCall.name === "evaluate_visual_watch";
        if (visualTickViolation || externalEvaluation) {
          const reason = visualTickViolation
            ? "Visual-watch ticks may only evaluate each listed watch once."
            : "evaluate_visual_watch is only available during a private visual-watch tick.";
          result = { ok: false, error: reason };
          this.state.finishToolCall(trace, null, reason);
        } else if (failure) {
          result = { ok: false, error: `Skipped because ${failure.tool} failed: ${failure.error}` };
          this.state.finishToolCall(trace, null, result.error);
        } else {
          try {
            if (functionCall.name === "evaluate_visual_watch") evaluatedWatches.add(watchId);
            result = await this.tools.execute(functionCall.name, args);
            this.state.finishToolCall(trace, result);
          } catch (error) {
            const message = errorMessage(error);
            failure = { tool: functionCall.name, error: message };
            result = { ok: false, error: message, instruction: "Stop this workflow; do not call another action." };
            this.state.finishToolCall(trace, null, message);
            if (inputKind === "visual-watch") {
              this.state.log("warning", `Visual watch evaluation failed: ${message}`);
            } else {
              this.failActiveOrder(message);
            }
          }
        }
        responses.push({
          id: functionCall.id,
          name: functionCall.name,
          response: { result },
        });
      }
    } finally {
      this.toolExecutionActive -= 1;
    }

    if (this.session && generation === this.sessionGeneration) {
      this.session.sendToolResponse({ functionResponses: responses });
      return;
    }

    this.sessionResumptionHandle = null;
    this.state.agentBusy = false;
    if (!failure) {
      this.pendingInputs.unshift({
        kind: "recovery",
        text: `The Live connection closed after these tools completed: ${compactJson(responses)}. ` +
          "Do not repeat successful tools. Report their results and continue from the tracked service state.",
      });
    }
    this.scheduleReconnect(0);
    this.schedulePump();
  }

  startCameraStream() {
    this.stopCameraStream();
    this.state.cameraStreaming = true;
    this.state.cameraError = null;
    this.state.changed();
    this.sendCameraFrame();
    this.cameraInterval = setInterval(() => this.sendCameraFrame(), this.cameraIntervalMs);
  }

  stopCameraStream() {
    if (this.cameraInterval) clearInterval(this.cameraInterval);
    this.cameraInterval = null;
    this.cameraFrameInFlight = false;
    if (this.state.cameraStreaming) {
      this.state.cameraStreaming = false;
      this.state.changed();
    }
  }

  async sendCameraFrame() {
    if (this.cameraFrameInFlight || !this.session) return;
    this.cameraFrameInFlight = true;
    const generation = this.sessionGeneration;
    const session = this.session;
    try {
      if (!this.state.robotConnected && Date.now() - this.lastCameraHealthCheck >= 5_000) {
        this.lastCameraHealthCheck = Date.now();
        const health = await this.tools.fastApi.get("/health", { timeoutMs: 3_000, tracked: false });
        this.state.robotConnected = health.connected === true;
      }
      if (!this.state.robotConnected) return;

      const frame = await this.tools.fastApi.getBinary(
        "/images/hand_color_image?quality_percent=65",
        { timeoutMs: 4_000 },
      );
      if (generation !== this.sessionGeneration || session !== this.session) return;
      session.sendRealtimeInput({
        video: {
          data: frame.data.toString("base64"),
          mimeType: frame.mimeType,
        },
      });
      this.state.cameraFrameAt = new Date().toISOString();
      this.state.cameraFramesSent += 1;
      this.state.cameraError = null;
      if (this.state.cameraFramesSent === 1 || this.state.cameraFramesSent % 5 === 0) this.state.changed();
      this.scheduleVisualWatchTick();
    } catch (error) {
      const message = errorMessage(error);
      if (this.state.cameraError !== message) {
        this.state.cameraError = message;
        this.state.log("warning", `Gemini camera stream: ${message}`);
      }
    } finally {
      this.cameraFrameInFlight = false;
    }
  }

  scheduleVisualWatchTick() {
    const activeWatches = this.state.visualWatches.filter((watch) => watch.status === "active");
    if (!activeWatches.length || this.visualWatchTickQueued || this.currentInputKind === "visual-watch") return;
    if (Date.now() - this.lastVisualWatchTick < this.visualWatchIntervalMs) return;

    this.lastVisualWatchTick = Date.now();
    this.visualWatchTickQueued = true;
    this.pendingInputs.push({
      kind: "visual-watch",
      text: "PRIVATE VISUAL WATCH TICK. Inspect only the latest gripper-camera frame. " +
        "Call evaluate_visual_watch exactly once for each active watch below. " +
        "Set subject_visible and condition_met conservatively from visible evidence. " +
        "Do not call any other tool. After all evaluations, respond with exactly DONE; this response is hidden. " +
        "Active watches: " +
        compactJson(activeWatches.map((watch) => ({
          watch_id: watch.id,
          subject: watch.subject,
          condition: watch.condition,
          requires_prior_presence: watch.requiresPriorPresence,
          armed: watch.armed,
        }))),
    });
    this.schedulePump();
  }

  scheduleReconnect(delayMs = 1_000) {
    if (!this.state.serviceRunning || this.session || this.connectPromise || this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.toolExecutionActive) {
        this.scheduleReconnect(500);
        return;
      }
      this.connect()
        .then(() => this.schedulePump())
        .catch((error) => {
          this.state.log("error", `Gemini Live reconnect failed: ${errorMessage(error)}`);
          this.scheduleReconnect(2_000);
        });
    }, delayMs);
  }

  clearReconnectTimer() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  clearVisualWatchTurnTimer() {
    if (this.visualWatchTurnTimer) clearTimeout(this.visualWatchTurnTimer);
    this.visualWatchTurnTimer = null;
  }

  recoverStuckVisualWatchTurn() {
    this.visualWatchTurnTimer = null;
    if (this.currentInputKind !== "visual-watch") return;
    this.state.log("warning", "Visual watch turn timed out; renewing the Gemini Live session");
    this.visualWatchTickQueued = false;
    this.resetSession().catch((error) => this.handleAgentError(error));
  }

  activeOrderRecoveryPrompt(order) {
    const recentTools = this.state.toolCalls.slice(-12).map((call) => ({
      name: call.name,
      status: call.status,
      result: call.result,
      error: call.error,
    }));
    return `Resume active hydration order #${order.id} (${order.drinkName} to ${order.destination}). ` +
      `Current order status: ${order.status}. Recent completed tool history: ${compactJson(recentTools)}. ` +
      "Do not repeat successful physical actions. Continue from the next required workflow step.";
  }

  failActiveOrder(message) {
    this.state.lastError = message;
    if (this.state.activeOrderId) {
      const order = this.state.orders.find((candidate) => candidate.id === this.state.activeOrderId);
      if (order && !["finished", "failed", "cancelled"].includes(order.status)) {
        this.state.updateOrder(order.id, "failed", message);
      }
      this.state.paused = true;
      this.state.log("service", "Service paused after tool failure");
    }
    this.state.changed();
  }

  finishActiveTurn() {
    if (!this.state.activeOrderId) return;
    const order = this.state.orders.find((candidate) => candidate.id === this.state.activeOrderId);
    if (!order || ["finished", "failed", "cancelled"].includes(order.status)) {
      this.continuationCounts.delete(this.state.activeOrderId);
      this.state.activeOrderId = null;
      return;
    }
    if (this.state.paused || !this.state.serviceRunning) return;

    const attempts = (this.continuationCounts.get(order.id) || 0) + 1;
    this.continuationCounts.set(order.id, attempts);
    if (attempts <= 2) {
      this.pendingInputs.unshift({
        kind: "continuation",
        text: `Order #${order.id} is still ${order.status}. Continue the workflow now. If it cannot continue, mark it failed and explain why.`,
      });
      return;
    }
    this.state.updateOrder(order.id, "failed", "Agent ended without completing the order workflow.");
    this.state.paused = true;
    this.state.lastError = `Order #${order.id} did not reach a terminal state.`;
    this.state.log("service", "Service paused because the agent stopped before completing its order");
  }

  handleAgentError(error) {
    const message = errorMessage(error);
    this.state.lastError = message;
    this.state.agentBusy = false;
    this.state.log("error", message);
    this.failActiveOrder(message);
  }
}

function extractModelText(message) {
  const parts = message.serverContent?.modelTurn?.parts || [];
  return parts.map((part) => part.text || "").join("");
}

function compactJson(value, maxLength = 4_000) {
  const text = JSON.stringify(value);
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
