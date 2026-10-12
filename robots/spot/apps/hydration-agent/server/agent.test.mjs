import assert from "node:assert/strict";
import test from "node:test";
import { DRINKS, isDeliveryWaypoint } from "./catalog.mjs";
import { validateRobotResult } from "./fastapi.mjs";
import { buildSystemPrompt } from "./prompt.mjs";
import { HydrationState } from "./state.mjs";
import { HydrationTools } from "./tools.mjs";

test("delivery waypoints exclude service-only positions", () => {
  const waypoints = ["peng-desk", "caden-desk"];
  assert.equal(isDeliveryWaypoint("peng-desk", waypoints), true);
  assert.equal(isDeliveryWaypoint("home", waypoints), false);
  assert.equal(isDeliveryWaypoint("snack1", waypoints), false);
  assert.equal(isDeliveryWaypoint("made-up", waypoints), false);
});

test("prompt contains dynamic catalog, waypoints, and fail-stop rules", () => {
  const prompt = buildSystemPrompt({
    drinks: DRINKS,
    waypoints: ["peng-desk"],
    toolDescriptions: "- navigate_to: calls /navigate",
  });
  assert.match(prompt, /blue-hint-water/);
  assert.match(prompt, /middle of red drink can/);
  assert.match(prompt, /peng-desk/);
  assert.match(prompt, /continuously receive the gripper color camera as video at 1 frame per second/);
  assert.match(prompt, /Never say that you cannot passively stream video/);
  assert.match(prompt, /you MUST call create_visual_watch/);
  assert.match(prompt, /call evaluate_visual_watch exactly once/);
  assert.match(prompt, /Map "right side of the floor" to horizontal=right and vertical=down/);
  assert.match(prompt, /use aim_gripper_camera before saying the object is not visible/);
  assert.match(prompt, /Stop the workflow immediately after any tool error/);
  assert.match(prompt, /navigate_to: calls \/navigate/);
});

test("camera aiming maps right and down to Spot arm rotations", async () => {
  const calls = [];
  const tools = new HydrationTools({
    fastApi: {
      post: async (path, body) => {
        calls.push({ path, body });
        return { arrived: true };
      },
    },
    state: { paused: false },
    config: { HYDRATION_AGENT_CAMERA_SETTLE_MS: 0 },
  });

  const result = await tools.execute("aim_gripper_camera", {
    horizontal: "right",
    vertical: "down",
    degrees: 12,
  });

  assert.equal(calls[0].path, "/arm/jog");
  assert.ok(calls[0].body.dyaw < 0);
  assert.ok(calls[0].body.dpitch > 0);
  assert.equal(result.camera_settled, true);
});

test("arm_set_pose compatibility alias dispatches to the declared arm pose tool", async () => {
  const calls = [];
  const tools = new HydrationTools({
    fastApi: {
      post: async (path, body) => {
        calls.push({ path, body });
        return { arrived: true };
      },
    },
    state: { paused: false },
    config: {},
  });

  await tools.execute("arm_set_pose", { pose: "stow" });

  assert.equal(calls[0].path, "/arm/stow");
  assert.equal(calls[0].body.take_lease, true);
});

test("robot result validation rejects incomplete actions", () => {
  assert.throws(
    () => validateRobotResult("/navigate", { reached_goal: false, status: "STATUS_STUCK" }),
    /STATUS_STUCK/,
  );
  assert.throws(
    () => validateRobotResult("/arm/stow", { arrived: false }),
    /did not arrive/,
  );
  assert.doesNotThrow(() => validateRobotResult("/navigate", { reached_goal: true, status: "STATUS_REACHED_GOAL" }));
});

test("leave watches arm on presence and trigger after two matching frames", () => {
  const state = new HydrationState("test-model");
  const watch = state.createVisualWatch({
    subject: "the man in the red shirt",
    condition: "the man in the red shirt leaves the camera view",
    responseMessage: "Hi!",
    requiresPriorPresence: true,
  });

  state.evaluateVisualWatch(watch.id, {
    subjectVisible: false,
    conditionMet: true,
    observation: "No red-shirted person is visible.",
  });
  assert.equal(watch.armed, false);
  assert.equal(watch.consecutiveMatches, 0);

  state.evaluateVisualWatch(watch.id, {
    subjectVisible: true,
    conditionMet: false,
    observation: "A man in a red shirt is visible.",
  });
  assert.equal(watch.armed, true);

  state.evaluateVisualWatch(watch.id, {
    subjectVisible: false,
    conditionMet: true,
    observation: "The previously visible man is absent.",
  });
  assert.equal(watch.status, "active");
  state.evaluateVisualWatch(watch.id, {
    subjectVisible: false,
    conditionMet: true,
    observation: "The man remains absent.",
  });

  assert.equal(watch.status, "triggered");
  assert.equal(state.messages.at(-1).text, "Hi!");
  assert.equal(state.messages.at(-1).proactive, true);
});
