// Robot Twin — a stylised 3D quadruped with a manipulator arm that mirrors
// the tool calls the model issues (stand / sit / stow / look / detect / pick /
// place / drive / navigate / stop). It is a *schematic* visualisation, not a
// physical simulation: each tool call is mapped to a short keyframed motion.
//
// Integration (all via window events, so agent-ui.js stays a classic script):
//   robot-twin:tool    {name, args, result}       — a non-ack tool call
//   robot-twin:target  {x, y, label}              — a fresh detect target (0-1000)
//   robot-twin:status  {state, ...}               — action watchdog status
//   robot-twin:reset   {}                         — session reset
//
// The design is deliberately generic (white/graphite body, blue accents) and
// does not reproduce any commercial robot.

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const canvas = document.getElementById('robot-twin-canvas');
const panel = document.getElementById('robot-twin');
if (canvas && panel) init();

function init() {
  // ---------------------------------------------------------------- scene --
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
  // Framed for a small square panel: robot + table, with headroom for rearing up.
  const HOME_CAM = new THREE.Vector3(2.0, 1.55, 2.75);
  const HOME_TARGET = new THREE.Vector3(0.38, 0.5, 0);
  camera.position.copy(HOME_CAM);

  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(HOME_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.2;
  controls.maxDistance = 6;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.enablePan = false;
  controls.update();

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9bb0, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2.5, 4, 2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 0.5, far: 10 });
  scene.add(sun);

  // World group: floor grid + table. Rotated/translated to fake locomotion
  // while the robot stays centred in view.
  const world = new THREE.Group();
  scene.add(world);
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(6, 64),
    new THREE.ShadowMaterial({ opacity: 0.18 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(12, 24, 0x8ab4f8, 0x9aa0a6);
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  world.add(grid);

  // Low table in front of the robot where detect / pick / place happen.
  const TABLE_TOP = 0.36;
  const TABLE_X = 1.05;          // table centre, forward of robot centre
  const TABLE_HALF = { x: 0.26, z: 0.42 };
  const table = new THREE.Group();
  const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.08, ...o });
  const tableTop = new THREE.Mesh(new THREE.BoxGeometry(TABLE_HALF.x * 2, 0.03, TABLE_HALF.z * 2), mat(0xc9b79c));
  tableTop.position.y = TABLE_TOP - 0.015;
  tableTop.castShadow = tableTop.receiveShadow = true;
  table.add(tableTop);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.03, TABLE_TOP - 0.03, 0.03), mat(0x8d7b63));
    leg.position.set(sx * (TABLE_HALF.x - 0.03), (TABLE_TOP - 0.03) / 2, sz * (TABLE_HALF.z - 0.03));
    leg.castShadow = true;
    table.add(leg);
  }
  table.position.x = TABLE_X;
  scene.add(table);  // table stays fixed relative to the robot (manipulation zone)

  // ---------------------------------------------------------------- robot --
  const C_BODY = 0xeef1f5, C_DARK = 0x3c4043, C_ACCENT = 0x4285f4, C_JOINT = 0x5f6368;
  const robot = new THREE.Group();
  scene.add(robot);
  const body = new THREE.Group();  // pitch / roll / height applied here
  robot.add(body);

  const BODY_L = 0.9, BODY_W = 0.34, BODY_H = 0.2;
  const torso = new THREE.Mesh(roundedBox(BODY_L, BODY_H, BODY_W, 0.06), mat(C_BODY));
  torso.castShadow = true;
  body.add(torso);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(BODY_L * 0.7, 0.025, BODY_W + 0.004), mat(C_ACCENT, { roughness: 0.4 }));
  stripe.position.y = 0.02;
  body.add(stripe);
  // Front sensor head with two "eyes".
  const head = new THREE.Mesh(roundedBox(0.12, 0.13, 0.28, 0.04), mat(C_DARK));
  head.position.set(BODY_L / 2 + 0.02, -0.01, 0);
  head.castShadow = true;
  body.add(head);
  for (const z of [-0.07, 0.07]) {
    const eye = new THREE.Mesh(new THREE.CircleGeometry(0.022, 20),
      new THREE.MeshStandardMaterial({ color: 0x8ab4f8, emissive: 0x4285f4, emissiveIntensity: 0.9 }));
    eye.position.set(BODY_L / 2 + 0.081, 0.0, z);
    eye.rotation.y = Math.PI / 2;
    body.add(eye);
  }

  // Legs: hip (pitch) -> thigh -> knee (pitch) -> shin -> foot.
  const L1 = 0.32, L2 = 0.34;
  const HIP_X = 0.33, HIP_Z = 0.2, HIP_Y = -0.06;
  const legs = [];
  for (const [sx, sz] of [[1, -1], [1, 1], [-1, -1], [-1, 1]]) {
    const hip = new THREE.Group();
    hip.position.set(sx * HIP_X, HIP_Y, sz * HIP_Z);
    body.add(hip);
    const hipCap = new THREE.Mesh(new THREE.SphereGeometry(0.055, 18, 12), mat(C_JOINT));
    hip.add(hipCap);
    const thigh = new THREE.Mesh(capsule(0.045, L1), mat(C_DARK));
    thigh.position.y = -L1 / 2;
    thigh.castShadow = true;
    hip.add(thigh);
    const knee = new THREE.Group();
    knee.position.y = -L1;
    hip.add(knee);
    const kneeCap = new THREE.Mesh(new THREE.SphereGeometry(0.042, 16, 10), mat(C_ACCENT));
    knee.add(kneeCap);
    const shin = new THREE.Mesh(capsule(0.032, L2), mat(C_DARK));
    shin.position.y = -L2 / 2;
    shin.castShadow = true;
    knee.add(shin);
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.04, 14, 10), mat(0x202124));
    foot.position.y = -L2;
    knee.add(foot);
    legs.push({ hip, knee, sx, sz, phase: (sx * sz > 0) ? 0 : Math.PI });
  }

  // Arm: base yaw -> shoulder pitch -> upper arm -> elbow -> forearm -> wrist -> gripper.
  const A1 = 0.5, A2 = 0.46, GRIP_LEN = 0.17;
  const ARM_BASE = new THREE.Vector3(0.28, BODY_H / 2 + 0.03, 0);
  const armYaw = new THREE.Group();
  armYaw.position.copy(ARM_BASE);
  body.add(armYaw);
  const turret = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.06, 24), mat(C_JOINT));
  armYaw.add(turret);
  const shoulder = new THREE.Group();
  shoulder.position.y = 0.05;
  armYaw.add(shoulder);
  shoulder.add(new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), mat(C_ACCENT)));
  const upper = new THREE.Mesh(capsule(0.038, A1, true), mat(C_BODY));
  upper.position.x = A1 / 2;
  upper.castShadow = true;
  shoulder.add(upper);
  const elbow = new THREE.Group();
  elbow.position.x = A1;
  shoulder.add(elbow);
  elbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.048, 16, 10), mat(C_ACCENT)));
  const fore = new THREE.Mesh(capsule(0.032, A2, true), mat(C_BODY));
  fore.position.x = A2 / 2;
  fore.castShadow = true;
  elbow.add(fore);
  const wrist = new THREE.Group();
  wrist.position.x = A2;
  elbow.add(wrist);
  wrist.add(new THREE.Mesh(new THREE.SphereGeometry(0.036, 14, 10), mat(C_JOINT)));
  const palm = new THREE.Mesh(roundedBox(0.08, 0.07, 0.07, 0.015), mat(C_DARK));
  palm.position.x = 0.05;
  palm.castShadow = true;
  wrist.add(palm);
  // Gripper camera lens + detect scan cone.
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.01, 16),
    new THREE.MeshStandardMaterial({ color: 0x8ab4f8, emissive: 0x4285f4, emissiveIntensity: 1 }));
  lens.rotation.z = Math.PI / 2;
  lens.position.set(0.07, 0.042, 0);
  wrist.add(lens);
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.32, 0.9, 32, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x4285f4, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  cone.rotation.z = Math.PI / 2;     // apex at the lens, opening along +x
  cone.position.set(0.07 + 0.45, 0.042, 0);
  wrist.add(cone);
  const fingers = [];
  for (const s of [-1, 1]) {
    const f = new THREE.Group();
    f.position.set(0.09, 0, s * 0.025);
    wrist.add(f);
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.03, 0.014), mat(0x202124));
    m.position.x = 0.045;
    m.castShadow = true;
    f.add(m);
    fingers.push({ g: f, s });
  }
  const gripPoint = new THREE.Object3D();   // between the finger tips
  gripPoint.position.x = GRIP_LEN - 0.03;
  wrist.add(gripPoint);

  // Manipulated object (appears on the table after detect).
  const item = new THREE.Mesh(roundedBox(0.07, 0.09, 0.07, 0.015), mat(0xea4335, { roughness: 0.35 }));
  item.castShadow = true;
  item.visible = false;
  scene.add(item);
  let itemHeld = false;
  // Placement marker ring.
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.075, 32),
    new THREE.MeshBasicMaterial({ color: 0x34a853, transparent: true, opacity: 0, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);

  // ------------------------------------------------------------- pose state --
  const STAND_H = 0.52, SIT_H = 0.3;
  const STOW = { yaw: 0, s: 3.06, e: -2.98, w: -0.06 };
  // rear: 0 = on four legs, 1 = reared up on the hind legs (front feet in the air).
  const REAR_PITCH = 0.95;
  const cur = { h: STAND_H, pitch: 0, roll: 0, yaw: 0, s: STOW.s, e: STOW.e, w: STOW.w, grip: 0, rear: 0 };
  const tgt = { ...cur };
  let armStowed = true;
  let sitting = false;
  let gait = null;          // {vx, vy, vrot, until}
  let heading = 0;
  const pos = new THREE.Vector2();
  let scanUntil = 0;
  let flash = null;         // {color, until}
  let queue = [];           // [{dur, enter(), until}]
  let step = null;

  function setArm(j) { Object.assign(tgt, { yaw: j.yaw, s: j.s, e: j.e, w: j.w }); }

  // IK for the arm. Target is in *robot* frame (x fwd, y up, z right),
  // assuming the body is level at STAND_H. toolPitch: 0 = gripper horizontal,
  // -PI/2 = pointing straight down.
  function armIK(x, y, z, toolPitch) {
    const bx = x - ARM_BASE.x, bz = z;
    const by = y - (STAND_H + ARM_BASE.y + 0.05);
    const yaw = Math.atan2(-bz, bx);
    // Wrist centre = grip target minus the gripper length along tool direction.
    const r0 = Math.hypot(bx, bz);
    const r = r0 - Math.cos(toolPitch) * GRIP_LEN;
    const h = by - Math.sin(toolPitch) * GRIP_LEN;
    let d = Math.hypot(r, h);
    d = Math.min(Math.max(d, Math.abs(A1 - A2) + 0.02), A1 + A2 - 0.005);
    const a = Math.atan2(h, r);
    const b = Math.acos(clamp((A1 * A1 + d * d - A2 * A2) / (2 * A1 * d), -1, 1));
    const k = Math.acos(clamp((A1 * A1 + A2 * A2 - d * d) / (2 * A1 * A2), -1, 1));
    const s = a + b;                 // elbow up
    const e = -(Math.PI - k);
    const w = toolPitch - (s + e);
    return { yaw, s, e, w };
  }

  const READY = () => armIK(0.62, STAND_H + 0.32, 0, -0.25);

  // Map 0-1000 image coordinates (gripper camera looking at the table) to a
  // point on the table top in robot frame.
  function imageToTable(x, y) {
    const tx = TABLE_X + TABLE_HALF.x * 0.8 * (1 - 2 * clamp(y / 1000, 0, 1)) * 0.9;
    const tz = TABLE_HALF.z * 0.85 * (2 * clamp(x / 1000, 0, 1) - 1);
    return new THREE.Vector3(tx, TABLE_TOP, tz);
  }

  // -------------------------------------------------------------- actions --
  function enqueue(steps) { queue.push(...steps); }
  function clearQueue() { queue = []; step = null; }
  const wait = (dur, enter) => ({ dur, enter: enter || (() => {}) });

  function ensureStanding() {
    if (!sitting) return [];
    return [wait(0.9, () => { sitting = false; tgt.h = STAND_H; tgt.pitch = 0; })];
  }
  function ensureDeployed() {
    if (!armStowed) return [];
    return [wait(0.9, () => { armStowed = false; setArm(READY()); tgt.grip = 0.2; })];
  }

  let pendingTarget = null;   // latest detect target (robot frame)

  const actions = {
    // Stand: get up on four legs, then rear up on the hind legs with the front
    // feet lifted so the motion reads clearly, hold, and come back down.
    stand: () => enqueue([
      wait(0.9, () => { sitting = false; tgt.h = STAND_H; tgt.pitch = 0; tgt.roll = 0; tgt.rear = 0; }),
      wait(1.1, () => { tgt.rear = 1; }),
      wait(0.9, () => {}),
      wait(1.0, () => { tgt.rear = 0; }),
    ]),
    sit: () => enqueue([
      ...(armStowed ? [] : [wait(0.8, () => { armStowed = true; setArm(STOW); tgt.grip = 0; })]),
      wait(1.0, () => { sitting = true; tgt.h = SIT_H; tgt.pitch = 0.36; }),
    ]),
    stow: () => enqueue([wait(0.9, () => { armStowed = true; setArm(STOW); if (!itemHeld) tgt.grip = 0; })]),
    stow_arm: () => actions.stow(),
    deploy_arm: () => enqueue([...ensureStanding(), wait(0.9, () => { armStowed = false; setArm(READY()); })]),
    open_gripper: () => enqueue([wait(0.4, () => { tgt.grip = 1; releaseItem(); })]),
    close_gripper: () => enqueue([wait(0.4, () => { tgt.grip = 0; })]),
    look: (a) => {
      const ang = Number(a.angle_rad) || 0.2;
      enqueue([...ensureStanding(), ...ensureDeployed(), wait(0.6, () => {
        const dir = String(a.direction || '');
        if (dir === 'left') tgt.yaw += ang;
        else if (dir === 'right') tgt.yaw -= ang;
        else if (dir === 'up') tgt.w += ang;
        else if (dir === 'down') tgt.w -= ang;
        scanUntil = now() + 0.8;
      })]);
    },
    turn_head_to_uv: () => enqueue([wait(0.6, () => { scanUntil = now() + 0.8; })]),
    detect: () => enqueue([...ensureStanding(), ...ensureDeployed(),
      wait(1.2, () => { setArm(armIK(0.66, STAND_H + 0.3, 0, -0.75)); scanUntil = now() + 1.6; })]),
    pick: (a, r) => {
      const t = targetFrom(r && r.target, a.grasp_x, a.grasp_y) || pendingTarget || imageToTable(500, 500);
      if (!itemHeld) { item.visible = true; item.position.set(t.x, TABLE_TOP + 0.045, t.z); }
      const above = t.y + 0.28;
      enqueue([...ensureStanding(), ...ensureDeployed(),
        wait(0.4, () => { tgt.grip = 1; }),
        wait(0.9, () => setArm(armIK(t.x, above, t.z, -Math.PI / 2))),
        wait(0.7, () => setArm(armIK(t.x, t.y + 0.05, t.z, -Math.PI / 2))),
        wait(0.45, () => { tgt.grip = 0.25; }),
        wait(0.05, () => grabItem()),
        wait(0.8, () => setArm(armIK(t.x - 0.08, above + 0.05, t.z, -Math.PI / 2))),
        wait(0.8, () => setArm(armIK(0.7, STAND_H + 0.36, 0, -0.35))),
      ]);
    },
    place: (a, r) => {
      const t = targetFrom(r && r.target, a.x, a.y) || pendingTarget || imageToTable(500, 300);
      showRing(t);
      const above = t.y + 0.3;
      enqueue([...ensureStanding(), ...ensureDeployed(),
        wait(1.0, () => setArm(armIK(t.x, above, t.z, -Math.PI / 2))),
        wait(0.7, () => setArm(armIK(t.x, t.y + 0.1, t.z, -Math.PI / 2))),
        wait(0.4, () => { tgt.grip = 1; }),
        wait(0.05, () => releaseItem(t)),
        wait(0.7, () => setArm(armIK(t.x - 0.05, above, t.z, -Math.PI / 2))),
        wait(0.8, () => { setArm(READY()); tgt.grip = 0.2; }),
      ]);
    },
    drive: (a) => {
      const dur = clamp(Number(a.duration) || 1, 0.6, 3);
      enqueue([...ensureStanding(), wait(dur, () => {
        gait = { vx: Number(a.v_x) || 0, vy: Number(a.v_y) || 0, vrot: Number(a.v_rot) || 0 };
      }), wait(0.3, () => { gait = null; })]);
    },
    navigate: () => enqueue([...ensureStanding(),
      ...(armStowed ? [] : [wait(0.8, () => { armStowed = true; setArm(STOW); })]),
      wait(3.0, () => { gait = { vx: 0.5, vy: 0, vrot: 0.15 }; }),
      wait(0.3, () => { gait = null; })]),
    make_gesture: () => enqueue([...ensureDeployed(),
      wait(0.5, () => setArm(armIK(0.6, STAND_H + 0.55, -0.1, 0.6))),
      wait(0.4, () => { tgt.w += 0.5; }), wait(0.4, () => { tgt.w -= 1.0; }),
      wait(0.4, () => { tgt.w += 1.0; }), wait(0.6, () => setArm(READY()))]),
    stop: () => { clearQueue(); gait = null; Object.assign(tgt, { ...cur }); flashColor(0xea4335, 0.8); },
  };

  function targetFrom(t, x, y) {
    if (Number.isFinite(Number(x)) && Number.isFinite(Number(y)) && x !== undefined && y !== undefined) {
      return imageToTable(Number(x), Number(y));
    }
    if (t && typeof t === 'object') {
      const tx = Number(t.x ?? t.normalized_x), ty = Number(t.y ?? t.normalized_y);
      if (Number.isFinite(tx) && Number.isFinite(ty)) return imageToTable(tx, ty);
    }
    return null;
  }

  function grabItem() {
    if (!item.visible) return;
    itemHeld = true;
    gripPoint.add(item);
    item.position.set(0, 0, 0);
    item.rotation.set(0, 0, Math.PI / 2);
  }
  function releaseItem(t) {
    if (!itemHeld) return;
    itemHeld = false;
    const wp = new THREE.Vector3();
    gripPoint.getWorldPosition(wp);
    scene.add(item);
    item.rotation.set(0, 0, 0);
    item.position.copy(wp);
    dropTo = t ? new THREE.Vector3(t.x, TABLE_TOP + 0.045, t.z)
               : new THREE.Vector3(wp.x, onTable(wp) ? TABLE_TOP + 0.045 : 0.045, wp.z);
  }
  let dropTo = null;
  function onTable(p) {
    return Math.abs(p.x - TABLE_X) < TABLE_HALF.x && Math.abs(p.z) < TABLE_HALF.z;
  }
  function showRing(t) { ring.position.set(t.x, TABLE_TOP + 0.002, t.z); ring.material.opacity = 0.9; ringUntil = now() + 6; }
  let ringUntil = 0;
  function flashColor(color, secs) { flash = { color, until: now() + secs }; }

  // ------------------------------------------------------------ HUD wiring --
  const actionEl = document.getElementById('twin-action');
  const statusEl = document.getElementById('twin-status');
  const ACTION_LABEL = {
    stand: '🧍 Stand', sit: '🪑 Sit', stow: '📥 Stow arm', stow_arm: '📥 Stow arm',
    deploy_arm: '🦾 Deploy arm', open_gripper: '🖐 Open gripper', close_gripper: '✊ Close gripper',
    look: '👀 Look', turn_head_to_uv: '👀 Look', detect: '🔍 Detect', pick: '🤏 Pick',
    place: '📦 Place', drive: '🐾 Drive', navigate: '🧭 Navigate', stop: '⏹ Stop', make_gesture: '👋 Gesture',
  };
  function setActionLabel(name, a) {
    if (!actionEl) return;
    let label = ACTION_LABEL[name] || ('🔧 ' + name);
    if (name === 'detect' && a.instruction) label += ' · ' + String(a.instruction).slice(0, 40);
    if (name === 'pick' && a.grasp_part) label += ' · ' + String(a.grasp_part).slice(0, 24);
    if (name === 'look' && a.direction) label += ' · ' + a.direction;
    if (name === 'navigate' && a.waypoint) label += ' → ' + String(a.waypoint).slice(0, 30);
    actionEl.textContent = label;
    actionEl.classList.add('active');
  }

  window.addEventListener('robot-twin:tool', (ev) => {
    const { name, args, result } = ev.detail || {};
    const fn = actions[name];
    if (!fn) return;
    const res = (result && typeof result === 'object') ? result : {};
    if (res.error || res.status === 'rejected' || res.status === 'error') {
      setActionLabel(name, args || {});
      flashColor(0xea4335, 0.8);
      return;
    }
    setActionLabel(name, args || {});
    fn(args || {}, res);
  });
  window.addEventListener('robot-twin:target', (ev) => {
    const p = ev.detail || {};
    const t = imageToTable(Number(p.x), Number(p.y));
    pendingTarget = t;
    if (itemHeld) {
      showRing(t);
    } else {
      item.visible = true;
      dropTo = null;
      item.position.set(t.x, TABLE_TOP + 0.045, t.z);
    }
  });
  window.addEventListener('robot-twin:status', (ev) => {
    if (!statusEl) return;
    const st = (ev.detail && ev.detail.state) || 'idle';
    const MAP = {
      pending: ['⏳ verifying', 'pending'], retry: ['🔁 retry', 'pending'],
      confirmed: ['✅ confirmed', 'ok'], timed_out: ['⌛ timed out', 'bad'],
      failed: ['❌ failed', 'bad'], aborted: ['❌ aborted', 'bad'],
    };
    const m = MAP[st];
    if (!m) { statusEl.classList.add('hidden'); return; }
    statusEl.textContent = m[0];
    statusEl.className = 'twin-status ' + m[1];
    if (st === 'confirmed') flashColor(0x34a853, 0.8);
    if (m[1] === 'bad') flashColor(0xea4335, 0.8);
  });
  window.addEventListener('robot-twin:reset', () => {
    clearQueue(); gait = null; itemHeld = false; scene.add(item); item.visible = false;
    pendingTarget = null; dropTo = null; ring.material.opacity = 0;
    if (actionEl) { actionEl.textContent = 'Idle'; actionEl.classList.remove('active'); }
    if (statusEl) statusEl.classList.add('hidden');
  });
  const resetBtn = document.getElementById('twin-reset');
  if (resetBtn) resetBtn.onclick = () => { camera.position.copy(HOME_CAM); controls.target.copy(HOME_TARGET); controls.update(); };

  // ----------------------------------------------------------- main loop --
  const clock = new THREE.Clock();
  let t = 0;
  function now() { return t; }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) ||
        canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }
  new ResizeObserver(resize).observe(canvas);

  const tmpV = new THREE.Vector3();
  const tmpM = new THREE.Matrix4();

  function frame() {
    requestAnimationFrame(frame);
    if (panel.offsetParent === null) return;   // hidden
    const dt = Math.min(clock.getDelta(), 0.05);
    t += dt;

    // Step queue.
    if (!step || t >= step.until) {
      step = queue.shift() || null;
      if (step) { step.until = t + step.dur; step.enter(); }
      else if (actionEl && !gait && actionEl.classList.contains('active') && t > idleAt) {
        actionEl.classList.remove('active');
      }
      if (step) idleAt = t + step.dur + 2.5;
    }

    // Smooth towards targets.
    const k = 1 - Math.exp(-dt * 7);
    for (const key of Object.keys(tgt)) cur[key] += (tgt[key] - cur[key]) * k;

    // Locomotion: integrate heading / position, move the grid.
    let walking = 0;
    if (gait) {
      heading += gait.vrot * dt;
      const fx = Math.cos(heading), fz = -Math.sin(heading);
      pos.x += (gait.vx * fx + gait.vy * fz) * dt;    // v_y = left
      pos.y += (gait.vx * fz - gait.vy * fx) * dt;
      walking = 1;
    }
    world.rotation.y = -heading;
    grid.position.set(-(pos.x % 0.5), 0, -(pos.y % 0.5));
    walkAmt += ((walking ? 1 : 0) - walkAmt) * k;
    gaitPhase += dt * 9 * walkAmt;

    // Body pose.
    const breathe = Math.sin(t * 2.2) * 0.006 * (1 - walkAmt);
    // Rearing pivots the body about the hind hips (kept in place).
    const q = cur.rear * REAR_PITCH;
    const rx = -HIP_X, ry = HIP_Y;      // hind hip in body frame
    const pivX = rx - (rx * Math.cos(q) - ry * Math.sin(q));
    const pivY = ry - (rx * Math.sin(q) + ry * Math.cos(q));
    body.position.x = pivX;
    body.position.y = cur.h + pivY + breathe + Math.abs(Math.sin(gaitPhase)) * 0.015 * walkAmt;
    body.rotation.set(cur.roll, 0, cur.pitch + q);
    robot.rotation.y = gait ? Math.sin(t * 3) * 0.02 : 0;

    // Leg IK: feet planted under the neutral hip positions (with gait offsets).
    body.updateMatrixWorld(true);
    tmpM.copy(body.matrixWorld).invert();
    for (const L of legs) {
      const ph = gaitPhase + L.phase;
      const stride = 0.09 * walkAmt * Math.sign((gait && gait.vx) || 1);
      const footW = tmpV.set(L.sx * HIP_X + Math.sin(ph) * stride, Math.max(0, Math.cos(ph)) * 0.07 * walkAmt, L.sz * HIP_Z);
      // Sitting: hind feet slide slightly forward; rearing: hind feet step in
      // under the raised body.
      if (L.sx < 0) footW.x += 0.12 * (STAND_H - cur.h) / (STAND_H - SIT_H) + 0.08 * cur.rear;
      footW.applyMatrix4(tmpM);   // into body frame
      if (L.sx > 0 && cur.rear > 0.001) {
        // Front feet lift off and tuck in front of the chest while rearing.
        const tuckX = L.hip.position.x + 0.16, tuckY = L.hip.position.y - 0.27;
        const wave = Math.sin(t * 6 + (L.sz > 0 ? 0 : Math.PI)) * 0.03 * cur.rear;
        footW.x += (tuckX - footW.x) * cur.rear;
        footW.y += (tuckY + wave - footW.y) * cur.rear;
      }
      const dx = footW.x - L.hip.position.x, dy = footW.y - L.hip.position.y;
      let d = Math.hypot(dx, dy);
      d = clamp(d, Math.abs(L1 - L2) + 0.01, L1 + L2 - 0.001);
      const a = Math.atan2(dx, -dy);
      const beta = Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
      const kInt = Math.acos(clamp((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1));
      L.hip.rotation.z = a - beta;          // knee points backwards
      L.knee.rotation.z = Math.PI - kInt;
    }

    // Arm joints.
    armYaw.rotation.y = cur.yaw;
    shoulder.rotation.z = cur.s;
    elbow.rotation.z = cur.e;
    wrist.rotation.z = cur.w;
    for (const f of fingers) f.g.rotation.y = -f.s * (0.05 + cur.grip * 0.45);

    // Detect scan cone.
    const scanning = t < scanUntil;
    cone.material.opacity += ((scanning ? 0.14 + Math.sin(t * 14) * 0.05 : 0) - cone.material.opacity) * k;

    // Dropped item falls onto the table / floor.
    if (dropTo && !itemHeld) {
      item.position.lerp(dropTo, 1 - Math.exp(-dt * 10));
      if (item.position.distanceTo(dropTo) < 0.002) dropTo = null;
    }
    if (t > ringUntil) ring.material.opacity *= 0.95;
    else ring.scale.setScalar(1 + Math.sin(t * 5) * 0.12);

    // Status flash on the eyes / accent stripe.
    const fl = flash && t < flash.until ? flash.color : null;
    stripe.material.emissive.setHex(fl || 0x000000);
    stripe.material.emissiveIntensity = fl ? 0.6 + Math.sin(t * 20) * 0.4 : 0;

    controls.update();
    renderer.render(scene, camera);
  }
  let walkAmt = 0, gaitPhase = 0, idleAt = 0;
  resize();
  frame();
}

// ---------------------------------------------------------------- helpers --
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

function capsule(r, len, alongX = false) {
  const g = new THREE.CapsuleGeometry(r, Math.max(0.001, len - 2 * r), 6, 14);
  if (alongX) g.rotateZ(Math.PI / 2);
  return g;
}

function roundedBox(w, h, d, r) {
  const shape = new THREE.Shape();
  const x = -w / 2, y = -d / 2;
  shape.moveTo(x + r, y);
  shape.lineTo(x + w - r, y);
  shape.quadraticCurveTo(x + w, y, x + w, y + r);
  shape.lineTo(x + w, y + d - r);
  shape.quadraticCurveTo(x + w, y + d, x + w - r, y + d);
  shape.lineTo(x + r, y + d);
  shape.quadraticCurveTo(x, y + d, x, y + d - r);
  shape.lineTo(x, y + r);
  shape.quadraticCurveTo(x, y, x + r, y);
  const bevel = Math.min(r, h / 2) * 0.6;
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: h - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel * 0.5, bevelSegments: 3, curveSegments: 6,
  });
  g.rotateX(-Math.PI / 2);   // extrude along +y
  g.center();
  return g;
}
