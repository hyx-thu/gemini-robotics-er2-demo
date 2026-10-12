import { Type } from "@google/genai";
import { DRINKS, findDrink, isDeliveryWaypoint } from "./catalog.mjs";

const objectSchema = (properties = {}, required = []) => ({
  type: Type.OBJECT,
  properties,
  ...(required.length ? { required } : {}),
});

export const TOOL_DECLARATIONS = [
  {
    name: "get_robot_status",
    description: "Read FastAPI /health, /battery, and current service/order state before acting.",
  },
  {
    name: "connect_robot",
    description: "Connect FastAPI to Spot using server-side .config credentials and take the lease. Never asks for or returns credentials.",
  },
  {
    name: "take_lease",
    description: "Take control using FastAPI POST /lease/take.",
  },
  {
    name: "list_waypoints",
    description: "List current named waypoints using FastAPI GET /waypoints.",
  },
  {
    name: "create_order",
    description: "Queue a hydration order from natural-language chat. Do not serve it in this same conversational turn; the scheduler dispatches it next.",
    parameters: objectSchema({
      drink_id: { type: Type.STRING, enum: DRINKS.map((drink) => drink.id) },
      destination: { type: Type.STRING, description: "Exact named delivery waypoint." },
    }, ["drink_id", "destination"]),
  },
  {
    name: "create_visual_watch",
    description: "Register a persistent gripper-camera watch for a future visual event. Required when the operator says when, until, leaves, arrives, appears, or disappears; acknowledging verbally is not sufficient.",
    parameters: objectSchema({
      subject: { type: Type.STRING, description: "The visual subject to track, such as 'the man in the red shirt'." },
      condition: { type: Type.STRING, description: "The exact future visual condition that triggers the response." },
      response_message: { type: Type.STRING, description: "Exact short message to emit when the condition is confirmed." },
      requires_prior_presence: { type: Type.BOOLEAN, description: "True for leaves/disappears conditions so the subject must first be seen; false for appears/arrives conditions." },
    }, ["subject", "condition", "response_message", "requires_prior_presence"]),
  },
  {
    name: "cancel_visual_watch",
    description: "Cancel an active visual watch requested by the operator.",
    parameters: objectSchema({
      watch_id: { type: Type.INTEGER },
    }, ["watch_id"]),
  },
  {
    name: "evaluate_visual_watch",
    description: "Internal visual-tick tool. Evaluate one active watch against the latest frame. This tool is only valid during a server visual-watch tick.",
    parameters: objectSchema({
      watch_id: { type: Type.INTEGER },
      subject_visible: { type: Type.BOOLEAN },
      condition_met: { type: Type.BOOLEAN },
      observation: { type: Type.STRING, description: "Brief factual observation from the latest frame." },
    }, ["watch_id", "subject_visible", "condition_met", "observation"]),
  },
  {
    name: "set_order_status",
    description: "Update an order's tracked state. Use finished only after a confirmed handoff, and failed after an unrecoverable problem.",
    parameters: objectSchema({
      order_id: { type: Type.INTEGER },
      status: { type: Type.STRING, enum: ["running", "finished", "failed", "cancelled"] },
      note: { type: Type.STRING },
    }, ["order_id", "status"]),
  },
  {
    name: "navigate_to",
    description: "Navigate Spot to an exact named waypoint using FastAPI POST /navigate. Powers on, stands, and takes lease.",
    parameters: objectSchema({
      waypoint: { type: Type.STRING },
    }, ["waypoint"]),
  },
  {
    name: "move_base",
    description: "Move Spot's base up to 0.5 m using FastAPI POST /teleop/velocity. Intended for short station alignment, not waypoint navigation.",
    parameters: objectSchema({
      direction: { type: Type.STRING, enum: ["forward", "backward", "left", "right"] },
      distance_m: { type: Type.NUMBER, description: "Distance from 0.01 to 0.5 meters." },
    }, ["direction", "distance_m"]),
  },
  {
    name: "set_arm_pose",
    description: "Move the arm to carry, deploy/ready, stow, or freeze using the matching FastAPI /arm endpoint.",
    parameters: objectSchema({
      pose: { type: Type.STRING, enum: ["carry", "deploy", "stow", "freeze"] },
    }, ["pose"]),
  },
  {
    name: "aim_gripper_camera",
    description: "Aim the gripper camera toward a user-provided relative location through FastAPI POST /arm/jog. Use this before declaring a spatially described object absent.",
    parameters: objectSchema({
      horizontal: { type: Type.STRING, enum: ["left", "center", "right"] },
      vertical: { type: Type.STRING, enum: ["up", "level", "down"] },
      degrees: { type: Type.NUMBER, description: "Small camera rotation from 3 to 25 degrees; normally 12." },
    }, ["horizontal", "vertical"]),
  },
  {
    name: "detect_object",
    description: "Detect an object in the gripper RGB/depth images through FastAPI POST /detect. Returns a server detection_id and 3D pose.",
    parameters: objectSchema({
      instruction: { type: Type.STRING, description: "Exact language instruction describing the target point." },
    }, ["instruction"]),
  },
  {
    name: "pick_object",
    description: "Detect and pick an object currently in the gripper camera view through FastAPI POST /pick and Spot's Manipulation API. Aim the camera first when the operator gives a relative location.",
    parameters: objectSchema({
      instruction: { type: Type.STRING, description: "Visual description of the object to pick, including useful location or appearance details." },
    }, ["instruction"]),
  },
  {
    name: "open_gripper",
    description: "Open the gripper through FastAPI POST /gripper/open. Use fraction 0.6 before grasping a drink.",
    parameters: objectSchema({
      fraction: { type: Type.NUMBER, description: "Open fraction from 0.0 to 1.0." },
    }, ["fraction"]),
  },
  {
    name: "close_gripper",
    description: "Close the gripper through FastAPI POST /gripper/close. Set slow=true when grasping a drink.",
    parameters: objectSchema({
      slow: { type: Type.BOOLEAN },
    }, ["slow"]),
  },
  {
    name: "rotate_gripper_camera",
    description: "Roll the gripper camera through FastAPI POST /arm/camera-roll.",
    parameters: objectSchema({
      direction: { type: Type.STRING, enum: ["clockwise", "counterclockwise"] },
      degrees: { type: Type.NUMBER, description: "Rotation angle from 1 to 180 degrees." },
    }, ["direction", "degrees"]),
  },
  {
    name: "approach_detection",
    description: "Approach a stored 3D detection with coordinated arm and base motion through FastAPI POST /arm/approach-whole-body.",
    parameters: objectSchema({
      detection_id: { type: Type.STRING },
      standoff_m: { type: Type.NUMBER, description: "Offset from target in meters; normally 0 for drink pickup." },
    }, ["detection_id"]),
  },
  {
    name: "wait_for_delivery",
    description: "Monitor upward gripper motion, open for handoff, close, and stow using FastAPI POST /delivery/wait.",
    parameters: objectSchema({
      monitor_seconds: { type: Type.NUMBER, description: "Monitoring duration, normally 30 seconds." },
    }),
  },
  {
    name: "stand_robot",
    description: "Power on and stand using FastAPI POST /stand.",
  },
  {
    name: "sit_robot",
    description: "Sit using FastAPI POST /sit.",
  },
  {
    name: "localize_robot",
    description: "Initialize GraphNav localization at a named waypoint using FastAPI POST /localize.",
    parameters: objectSchema({
      waypoint: { type: Type.STRING },
    }, ["waypoint"]),
  },
  {
    name: "clear_behavior_faults",
    description: "Clear Spot behavior faults using FastAPI POST /faults/behavior/clear.",
  },
  {
    name: "stop_actions",
    description: "Immediately cancel motion and freeze the arm using FastAPI POST /actions/stop.",
  },
];

const READ_ONLY_TOOLS = new Set([
  "get_robot_status", "list_waypoints", "create_order", "set_order_status", "create_visual_watch",
  "cancel_visual_watch", "evaluate_visual_watch", "stop_actions",
]);

export class HydrationTools {
  constructor({ fastApi, state, config }) {
    this.fastApi = fastApi;
    this.state = state;
    this.config = config;
    this.detections = new Map();
    this.nextDetectionId = 1;
  }

  descriptions() {
    return TOOL_DECLARATIONS.map((tool) => `- ${tool.name}: ${tool.description}`).join("\n");
  }

  async refreshWaypoints() {
    const items = await this.fastApi.get("/waypoints", { timeoutMs: 10_000 });
    const names = items.map((item) => item.name).filter(Boolean);
    this.state.setWaypoints(names.filter((name) => name !== "home" && name !== "snack1"));
    return names;
  }

  async execute(name, args = {}) {
    const resolvedName = TOOL_ALIASES[name] || name;
    if (this.state.paused && !READ_ONLY_TOOLS.has(resolvedName)) {
      throw new Error(`Service is paused; ${resolvedName} was not executed.`);
    }

    switch (resolvedName) {
      case "get_robot_status":
        return this.getRobotStatus();
      case "connect_robot":
        return this.connectRobot();
      case "take_lease":
        return this.fastApi.post("/lease/take", {}, { timeoutMs: 20_000 });
      case "list_waypoints":
        return { waypoints: await this.refreshWaypoints(), delivery_waypoints: this.state.waypoints };
      case "create_order":
        return this.createOrder(args);
      case "create_visual_watch":
        return this.createVisualWatch(args);
      case "cancel_visual_watch":
        return this.cancelVisualWatch(args);
      case "evaluate_visual_watch":
        return this.evaluateVisualWatch(args);
      case "set_order_status":
        return this.setOrderStatus(args);
      case "navigate_to":
        return this.navigate(args);
      case "move_base":
        return this.moveBase(args);
      case "set_arm_pose":
        return this.setArmPose(args);
      case "aim_gripper_camera":
        return this.aimGripperCamera(args);
      case "detect_object":
        return this.detectObject(args);
      case "pick_object":
        return this.pickObject(args);
      case "open_gripper":
        return this.openGripper(args);
      case "close_gripper":
        return this.closeGripper(args);
      case "rotate_gripper_camera":
        return this.rotateCamera(args);
      case "approach_detection":
        return this.approachDetection(args);
      case "wait_for_delivery":
        return this.waitForDelivery(args);
      case "stand_robot":
        return this.fastApi.post("/stand", { power_on: true, take_lease: true, timeout: 15 }, { timeoutMs: 20_000 });
      case "sit_robot":
        return this.fastApi.post("/sit", { take_lease: true, timeout: 15 }, { timeoutMs: 20_000 });
      case "localize_robot":
        return this.fastApi.post("/localize", { waypoint_name: requiredString(args.waypoint, "waypoint") }, { timeoutMs: 40_000 });
      case "clear_behavior_faults":
        return this.fastApi.post("/faults/behavior/clear", {}, { timeoutMs: 20_000 });
      case "stop_actions":
        return this.fastApi.stopRobot();
      default:
        throw new Error(`Unknown tool: ${resolvedName}`);
    }
  }

  async getRobotStatus() {
    const health = await this.fastApi.get("/health", { timeoutMs: 10_000 });
    this.state.robotConnected = health.connected === true;
    let battery = null;
    if (health.connected) {
      try {
        battery = await this.fastApi.get("/battery", { timeoutMs: 10_000 });
      } catch (error) {
        battery = { error: errorMessage(error) };
      }
    }
    return { ...health, battery, service: this.serviceSummary() };
  }

  async connectRobot() {
    const password = this.config.BOSDYN_CLIENT_PASSWORD;
    if (!password) throw new Error("BOSDYN_CLIENT_PASSWORD is missing from .config.");
    const result = await this.fastApi.post("/connect", {
      hostname: this.config.SPOT_HOSTNAME || this.config.BOSDYN_CLIENT_HOSTNAME || "192.168.80.3",
      username: this.config.BOSDYN_CLIENT_USERNAME || "user",
      password,
      take_lease: true,
    }, { timeoutMs: 60_000 });
    this.state.robotConnected = true;
    this.state.changed();
    return result;
  }

  createOrder(args) {
    const drink = findDrink(requiredString(args.drink_id, "drink_id"));
    if (!drink) throw new Error(`Unknown drink ID: ${args.drink_id}`);
    const destination = requiredString(args.destination, "destination");
    if (!isDeliveryWaypoint(destination, this.state.waypoints)) {
      throw new Error(`Invalid delivery waypoint: ${destination}`);
    }
    const order = this.state.createOrder(drink, destination, "chat");
    return { order_id: order.id, status: order.status, queued: true };
  }

  createVisualWatch(args) {
    const watch = this.state.createVisualWatch({
      subject: requiredString(args.subject, "subject"),
      condition: requiredString(args.condition, "condition"),
      responseMessage: requiredString(args.response_message, "response_message"),
      requiresPriorPresence: args.requires_prior_presence === true,
    });
    return {
      watch_id: watch.id,
      status: watch.status,
      armed: watch.armed,
      condition: watch.condition,
    };
  }

  cancelVisualWatch(args) {
    const watch = this.state.cancelVisualWatch(Number(args.watch_id));
    return { watch_id: watch.id, status: watch.status };
  }

  evaluateVisualWatch(args) {
    const watch = this.state.evaluateVisualWatch(Number(args.watch_id), {
      subjectVisible: args.subject_visible === true,
      conditionMet: args.condition_met === true,
      observation: requiredString(args.observation, "observation"),
    });
    return {
      watch_id: watch.id,
      status: watch.status,
      armed: watch.armed,
      consecutive_matches: watch.consecutiveMatches,
      triggered: watch.status === "triggered",
    };
  }

  setOrderStatus(args) {
    const status = requiredString(args.status, "status");
    if (!["running", "finished", "failed", "cancelled"].includes(status)) {
      throw new Error(`Invalid order status: ${status}`);
    }
    const order = this.state.updateOrder(Number(args.order_id), status, optionalString(args.note));
    const pending = this.state.orders.filter((candidate) => candidate.status === "pending");
    return {
      order_id: order.id,
      status: order.status,
      pending_orders: pending.length,
      next_action: pending.length ? "Serve the next pending order." : "Navigate home and sit.",
    };
  }

  navigate(args) {
    const waypoint = requiredString(args.waypoint, "waypoint");
    const known = ["home", "snack1", ...this.state.waypoints];
    if (!known.includes(waypoint)) throw new Error(`Unknown waypoint: ${waypoint}`);
    return this.fastApi.post("/navigate", {
      name: waypoint,
      take_lease: true,
      power_on: true,
      stand: true,
      timeout: 180,
    }, { timeoutMs: 190_000 });
  }

  moveBase(args) {
    const direction = requiredString(args.direction, "direction");
    const distance = boundedNumber(args.distance_m, "distance_m", 0.01, 0.5);
    const speed = 0.25;
    const signs = {
      forward: { v_x: speed, v_y: 0 },
      backward: { v_x: -speed, v_y: 0 },
      left: { v_x: 0, v_y: speed },
      right: { v_x: 0, v_y: -speed },
    };
    if (!signs[direction]) throw new Error(`Invalid direction: ${direction}`);
    return this.fastApi.post("/teleop/velocity", {
      ...signs[direction],
      v_rot: 0,
      duration: distance / speed,
      take_lease: true,
      power_on: false,
      stand: false,
    }, { timeoutMs: 10_000 });
  }

  setArmPose(args) {
    const pose = requiredString(args.pose, "pose");
    const endpoints = {
      carry: "/arm/carry",
      deploy: "/arm/deploy",
      stow: "/arm/stow",
      freeze: "/arm/freeze",
    };
    const endpoint = endpoints[pose];
    if (!endpoint) throw new Error(`Invalid arm pose: ${pose}`);
    return this.fastApi.post(endpoint, {
      take_lease: true,
      timeout: pose === "stow" ? 20 : 10,
      ...(pose === "deploy" ? { power_on: true } : {}),
    }, { timeoutMs: 30_000 });
  }

  async aimGripperCamera(args) {
    const horizontal = requiredEnum(args.horizontal, "horizontal", ["left", "center", "right"]);
    const vertical = requiredEnum(args.vertical, "vertical", ["up", "level", "down"]);
    const degrees = args.degrees === undefined
      ? 12
      : boundedNumber(args.degrees, "degrees", 3, 25);
    const angle = degrees * Math.PI / 180;
    const dyaw = horizontal === "left" ? angle : horizontal === "right" ? -angle : 0;
    const dpitch = vertical === "up" ? -angle : vertical === "down" ? angle : 0;
    if (dyaw === 0 && dpitch === 0) {
      return { aimed: false, horizontal, vertical, degrees, reason: "Camera direction was unchanged." };
    }

    const result = await this.fastApi.post("/arm/jog", {
      dyaw,
      dpitch,
      seconds: 1.0,
      take_lease: true,
      timeout: 4,
    }, { timeoutMs: 10_000 });
    const settleMs = Math.max(0, Number(this.config.HYDRATION_AGENT_CAMERA_SETTLE_MS ?? 1_200));
    if (settleMs) await delay(settleMs);
    return {
      ...result,
      aimed: true,
      horizontal,
      vertical,
      degrees,
      camera_settled: true,
    };
  }

  async detectObject(args) {
    const instruction = requiredString(args.instruction, "instruction");
    const scene = await this.fastApi.post("/detect", {
      instruction,
      api_key: this.config.GEMINI_API_KEY || null,
      include_point_cloud: false,
    }, { timeoutMs: 60_000 });
    if (!scene.pose) throw new Error(`Detection returned no 3D pose: ${JSON.stringify(scene.errors || {})}`);
    const detectionId = `detection-${this.nextDetectionId++}`;
    this.detections.set(detectionId, scene.pose);
    return {
      detection_id: detectionId,
      pose: scene.pose,
      detection: scene.detection || scene.point_2d || null,
      image: scene.image ? { width: scene.image.width, height: scene.image.height } : null,
      model_output: scene.model_output || scene.raw_model_output || null,
    };
  }

  async pickObject(args) {
    const instruction = requiredString(args.instruction, "instruction");
    const result = await this.fastApi.post("/pick", {
      instruction,
      model: "gemini-robotics-er-1.6-preview",
      api_key: this.config.GEMINI_API_KEY || null,
      take_lease: true,
      timeout: 60,
    }, { timeoutMs: 75_000 });
    const state = String(result.state || "");
    if (!/GRASP_SUCCEEDED|MANIP_STATE_DONE/.test(state)) {
      throw new Error(`/pick did not confirm a grasp: ${state || "missing manipulation state"}`);
    }
    return result;
  }

  openGripper(args) {
    const fraction = boundedNumber(args.fraction, "fraction", 0, 1);
    return this.fastApi.post("/gripper/open", {
      open_fraction: fraction,
      max_vel: 0.5,
      max_acc: 1.0,
      take_lease: true,
      timeout: 5,
    }, { timeoutMs: 10_000 });
  }

  closeGripper(args) {
    const slow = args.slow !== false;
    return this.fastApi.post("/gripper/close", {
      max_vel: slow ? 0.25 : 0.5,
      max_acc: slow ? 0.5 : 1.0,
      take_lease: true,
      timeout: 10,
    }, { timeoutMs: 15_000 });
  }

  rotateCamera(args) {
    const direction = requiredString(args.direction, "direction");
    if (!["clockwise", "counterclockwise"].includes(direction)) throw new Error(`Invalid rotation: ${direction}`);
    const degrees = boundedNumber(args.degrees, "degrees", 1, 180);
    return this.fastApi.post("/arm/camera-roll", {
      direction,
      angle_rad: degrees * Math.PI / 180,
      seconds: Math.max(0.7, degrees / 90 * 0.7),
      take_lease: true,
      timeout: 5,
    }, { timeoutMs: 10_000 });
  }

  approachDetection(args) {
    const id = requiredString(args.detection_id, "detection_id");
    const pose = this.detections.get(id);
    if (!pose) throw new Error(`Unknown or expired detection ID: ${id}`);
    const standoff = args.standoff_m === undefined ? 0 : boundedNumber(args.standoff_m, "standoff_m", -0.1, 0.1);
    return this.fastApi.post("/arm/approach-whole-body", {
      pose,
      standoff_m: standoff,
      max_step_m: 0.8,
      seconds: 2,
      take_lease: true,
      timeout: 10,
    }, { timeoutMs: 20_000 });
  }

  async waitForDelivery(args) {
    const monitorSeconds = args.monitor_seconds === undefined
      ? 30
      : boundedNumber(args.monitor_seconds, "monitor_seconds", 5, 120);
    const result = await this.fastApi.post("/delivery/wait", {
      monitor_sec: monitorSeconds,
      upward_threshold_m: 0.02,
      sample_interval: 0.1,
      open_duration_sec: 3,
      take_lease: true,
      gripper_timeout: 5,
      stow_timeout: 10,
    }, { timeoutMs: (monitorSeconds + 25) * 1000 });
    if (result.triggered !== true) throw new Error(result.reason || "Delivery handoff was not detected.");
    return result;
  }

  serviceSummary() {
    return {
      running: this.state.serviceRunning,
      paused: this.state.paused,
      active_order_id: this.state.activeOrderId,
      pending_orders: this.state.orders.filter((order) => order.status === "pending").length,
    };
  }
}

function requiredString(value, name) {
  const result = String(value ?? "").trim();
  if (!result) throw new Error(`${name} is required.`);
  return result;
}

function optionalString(value) {
  const result = String(value ?? "").trim();
  return result || null;
}

function boundedNumber(value, name, minimum, maximum) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < minimum || number > maximum) {
    throw new Error(`${name} must be between ${minimum} and ${maximum}.`);
  }
  return number;
}

function requiredEnum(value, name, values) {
  const result = requiredString(value, name);
  if (!values.includes(result)) throw new Error(`${name} must be one of: ${values.join(", ")}.`);
  return result;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

const TOOL_ALIASES = Object.freeze({
  arm_set_pose: "set_arm_pose",
});
