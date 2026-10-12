export class FastApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.activeControllers = new Set();
  }

  get(pathname, options = {}) {
    return this.request(pathname, { ...options, method: "GET" });
  }

  post(pathname, body = {}, options = {}) {
    return this.request(pathname, { ...options, method: "POST", body });
  }

  async getBinary(pathname, { timeoutMs = 5_000 } = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new Error(`Timed out calling ${pathname}.`)), timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${pathname}`, { signal: controller.signal });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `${response.status} ${response.statusText}`);
      }
      return {
        data: Buffer.from(await response.arrayBuffer()),
        mimeType: response.headers.get("content-type") || "image/jpeg",
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async request(pathname, { method, body, timeoutMs = 190_000, tracked = true }) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new Error(`Timed out calling ${pathname}.`)), timeoutMs);
    if (tracked) this.activeControllers.add(controller);
    try {
      const response = await fetch(`${this.baseUrl}${pathname}`, {
        method,
        headers: body === undefined ? undefined : { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
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
      validateRobotResult(pathname, data);
      return data;
    } finally {
      clearTimeout(timeout);
      this.activeControllers.delete(controller);
    }
  }

  cancelActive() {
    for (const controller of this.activeControllers) controller.abort(new Error("Action stopped by operator."));
    this.activeControllers.clear();
  }

  async stopRobot() {
    this.cancelActive();
    return this.post(
      "/actions/stop",
      { take_lease: true, freeze_arm: true },
      { timeoutMs: 15_000, tracked: false },
    );
  }
}

export function validateRobotResult(pathname, result) {
  if (!result || typeof result !== "object") return;
  if (result.reached_goal === false) throw new Error(`${pathname}: ${result.status || "goal not reached"}`);
  if (result.arrived === false) throw new Error(`${pathname}: arm did not arrive`);
  if (result.at_goal === false) throw new Error(`${pathname}: gripper did not reach goal`);
  if (typeof result.status === "string" && /STUCK|CANCELLED|TIMED_OUT|LOST|FAILED|ERROR/.test(result.status)) {
    throw new Error(`${pathname}: ${result.status}`);
  }
}
