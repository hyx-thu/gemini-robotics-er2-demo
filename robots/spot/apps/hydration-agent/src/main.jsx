import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  Bot,
  CirclePause,
  CirclePlay,
  Coffee,
  Link,
  MessageSquareText,
  Power,
  RefreshCw,
  RotateCcw,
  Send,
  Square,
  Wrench,
} from "lucide-react";
import "./styles.css";

const EMPTY_STATE = {
  orders: [],
  messages: [],
  toolCalls: [],
  events: [],
  waypoints: [],
  serviceRunning: false,
  paused: false,
  agentConnected: false,
  agentBusy: false,
  robotConnected: false,
  cameraStreaming: false,
  cameraFrameAt: null,
  cameraFramesSent: 0,
  cameraError: null,
  visualWatches: [],
};

function App() {
  const [bootstrap, setBootstrap] = useState({ drinks: [], waypoints: [] });
  const [state, setState] = useState(EMPTY_STATE);
  const [drinkId, setDrinkId] = useState("");
  const [destination, setDestination] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [cameraTick, setCameraTick] = useState(Date.now());
  const chatEnd = useRef(null);

  useEffect(() => {
    apiGet("/api/bootstrap")
      .then((data) => {
        setBootstrap(data);
        setDrinkId(data.drinks[0]?.id || "");
        setDestination(data.waypoints[0] || "");
      })
      .catch((nextError) => setError(nextError.message));

    const events = new EventSource("/api/events");
    events.onmessage = (event) => setState(JSON.parse(event.data));
    events.onerror = () => setError("Lost the hydration agent event stream.");
    return () => events.close();
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [state.messages]);

  useEffect(() => {
    if (!state.robotConnected) return undefined;
    const timer = setInterval(() => setCameraTick(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [state.robotConnected]);

  const counts = useMemo(() => ({
    pending: state.orders.filter((order) => order.status === "pending").length,
    running: state.orders.filter((order) => order.status === "running").length,
    finished: state.orders.filter((order) => order.status === "finished").length,
    failed: state.orders.filter((order) => order.status === "failed").length,
  }), [state.orders]);

  async function runAction(name, path, body = {}) {
    setBusy(name);
    setError("");
    try {
      const result = await apiPost(path, body);
      if (result?.state) setState(result.state);
      else if (result?.orders) setState(result);
      return result;
    } catch (nextError) {
      setError(nextError.message);
      return null;
    } finally {
      setBusy("");
    }
  }

  async function submitOrder() {
    await runAction("order", "/api/orders", { drinkId, destination });
  }

  async function submitChat(event) {
    event.preventDefault();
    const text = message.trim();
    if (!text || busy === "chat") return;
    setMessage("");
    await runAction("chat", "/api/chat", { message: text });
  }

  const serviceLabel = state.paused
    ? "Paused"
    : state.agentBusy
      ? "Agent working"
      : state.serviceRunning
        ? "Ready"
        : "Stopped";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Bot size={18} /></span>
          <div>
            <h1>Gemini Robotics Hydration Agent</h1>
            <span>{state.model || bootstrap.model}</span>
          </div>
        </div>
        <div className="topbar-status">
          <StatusDot active={state.agentConnected} label="Gemini Live" />
          <StatusDot active={state.robotConnected} label="Spot" />
          <StatusDot active={state.cameraStreaming && Boolean(state.cameraFrameAt)} label="Vision 1 FPS" />
          <span className={`service-state ${state.paused ? "paused" : ""}`}>{serviceLabel}</span>
        </div>
        <div className="topbar-actions">
          <IconButton
            icon={Link}
            label="Connect Spot"
            disabled={Boolean(busy)}
            onClick={() => runAction("connect", "/api/service/connect-robot")}
          />
          <IconButton
            icon={state.paused ? CirclePlay : Power}
            label={state.paused ? "Resume agent" : "Start agent"}
            disabled={Boolean(busy)}
            onClick={() => runAction("start", state.paused ? "/api/service/resume" : "/api/service/start")}
          />
          <IconButton
            icon={CirclePause}
            label="Pause agent"
            disabled={Boolean(busy)}
            onClick={() => runAction("pause", "/api/service/pause")}
          />
          <IconButton
            icon={Square}
            label="Stop robot actions"
            danger
            disabled={Boolean(busy)}
            onClick={() => runAction("stop", "/api/service/stop-actions")}
          />
          <IconButton
            icon={RotateCcw}
            label="Reset Gemini session"
            disabled={Boolean(busy)}
            onClick={() => runAction("reset", "/api/service/reset-session")}
          />
        </div>
      </header>

      <main className="workspace">
        <aside className="order-panel">
          <PanelHeading icon={Coffee} title="New Order" />
          <div className="drink-list">
            {bootstrap.drinks.map((drink) => (
              <button
                className={`drink-option ${drink.id === drinkId ? "selected" : ""}`}
                key={drink.id}
                onClick={() => setDrinkId(drink.id)}
                type="button"
              >
                <span className="drink-swatch" style={{ background: drink.color }} />
                <span>
                  <strong>{drink.name}</strong>
                  <small>{drink.flavor}</small>
                </span>
              </button>
            ))}
          </div>
          <label className="field-label" htmlFor="destination">Delivery position</label>
          <select
            id="destination"
            value={destination}
            onChange={(event) => setDestination(event.target.value)}
          >
            {bootstrap.waypoints.map((waypoint) => (
              <option key={waypoint} value={waypoint}>{waypoint}</option>
            ))}
          </select>
          <button
            className="primary-command"
            type="button"
            disabled={Boolean(busy) || !drinkId || !destination}
            onClick={submitOrder}
          >
            <Coffee size={16} />
            Queue order
          </button>

          <div className="metrics">
            <Metric label="Queued" value={counts.pending} />
            <Metric label="Active" value={counts.running} />
            <Metric label="Done" value={counts.finished} />
            <Metric label="Failed" value={counts.failed} tone="danger" />
          </div>

          <div className="section-rule" />
          <PanelHeading icon={RefreshCw} title="Orders" compact />
          <div className="order-list">
            {state.orders.length ? state.orders.slice().reverse().map((order) => (
              <OrderRow key={order.id} order={order} active={order.id === state.activeOrderId} />
            )) : <Empty label="No orders" />}
          </div>
        </aside>

        <section className="chat-panel">
          <div className="panel-bar">
            <PanelHeading icon={MessageSquareText} title="Operator Chat" compact />
            {state.agentBusy ? <span className="working-indicator"><span /> Working</span> : null}
          </div>
          <div className="chat-log">
            {state.messages.length ? state.messages.map((item) => (
              <ChatMessage key={item.id} message={item} />
            )) : (
              <div className="chat-empty">
                <Bot size={26} />
                <span>Agent ready</span>
              </div>
            )}
            <div ref={chatEnd} />
          </div>
          {error || state.lastError ? (
            <div className="error-strip"><AlertTriangle size={15} />{error || state.lastError}</div>
          ) : null}
          <form className="composer" onSubmit={submitChat}>
            <textarea
              value={message}
              rows={2}
              placeholder="Send an instruction"
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submitChat(event);
                }
              }}
            />
            <button className="send-button" type="submit" disabled={!message.trim() || busy === "chat"} title="Send">
              <Send size={18} />
            </button>
          </form>
        </section>

        <aside className="activity-panel">
          <div className="camera-section">
            <div className="panel-bar">
              <PanelHeading icon={Bot} title="Gripper Camera" compact />
              <span className={`live-label ${state.cameraStreaming ? "" : "inactive"}`}>
                {state.cameraStreaming ? "GEMINI 1 FPS" : "OFFLINE"}
              </span>
            </div>
            <div className="camera-frame">
              {state.robotConnected
                ? <img src={`/api/camera/gripper?t=${cameraTick}`} alt="Spot gripper camera" />
                : <span>Spot disconnected</span>}
            </div>
            {state.cameraError ? <div className="camera-error">{state.cameraError}</div> : null}
          </div>
          <div className="tool-section">
            <PanelHeading icon={Wrench} title="Tool Activity" compact />
            <div className="tool-list">
              {state.visualWatches.slice().reverse().map((watch) => (
                <VisualWatchRow key={`watch-${watch.id}`} watch={watch} />
              ))}
              {state.toolCalls.length ? state.toolCalls.slice().reverse().map((call) => (
                <ToolRow key={call.id} call={call} />
              )) : state.visualWatches.length ? null : <Empty label="No tool calls" />}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

function IconButton({ icon: Icon, label, onClick, disabled, danger = false }) {
  return (
    <button className={`icon-button ${danger ? "danger" : ""}`} title={label} aria-label={label} onClick={onClick} disabled={disabled}>
      <Icon size={17} />
    </button>
  );
}

function StatusDot({ active, label }) {
  return <span className="status-dot"><i className={active ? "active" : ""} />{label}</span>;
}

function PanelHeading({ icon: Icon, title, compact = false }) {
  return <div className={`panel-heading ${compact ? "compact" : ""}`}><Icon size={15} /><h2>{title}</h2></div>;
}

function Metric({ label, value, tone = "" }) {
  return <div className={`metric ${tone}`}><strong>{value}</strong><span>{label}</span></div>;
}

function OrderRow({ order, active }) {
  return (
    <article className={`order-row ${active ? "active" : ""}`}>
      <div><strong>#{order.id} {order.drinkName}</strong><span>{order.destination}</span></div>
      <span className={`badge ${order.status}`}>{order.status}</span>
      {order.note ? <p>{order.note}</p> : null}
    </article>
  );
}

function ChatMessage({ message }) {
  return (
    <div className={`chat-message ${message.role}`}>
      <span>{message.role === "assistant" ? "Gemini" : message.role === "system" ? "Service" : "You"}</span>
      <p>{message.text || "..."}</p>
      <time>{new Date(message.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
    </div>
  );
}

function ToolRow({ call }) {
  return (
    <article className={`tool-row ${call.status}`}>
      <div className="tool-row-head"><strong>{call.name}</strong><span>{call.status}</span></div>
      <code>{compactJson(call.args)}</code>
      {call.error ? <p>{call.error}</p> : null}
    </article>
  );
}

function VisualWatchRow({ watch }) {
  return (
    <article className={`watch-row ${watch.status}`}>
      <div className="tool-row-head">
        <strong>Visual watch #{watch.id}</strong>
        <span>{watch.status === "active" && !watch.armed ? "acquiring" : watch.status}</span>
      </div>
      <p>{watch.condition}</p>
      {watch.observation ? <small>{watch.observation}</small> : null}
    </article>
  );
}

function Empty({ label }) {
  return <div className="empty-state">{label}</div>;
}

function compactJson(value) {
  const text = JSON.stringify(value || {});
  return text.length > 150 ? `${text.slice(0, 147)}...` : text;
}

async function apiGet(path) {
  return responseJson(await fetch(path));
}

async function apiPost(path, body) {
  return responseJson(await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }));
}

async function responseJson(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.detail || response.statusText);
  return data;
}

createRoot(document.getElementById("root")).render(<App />);
