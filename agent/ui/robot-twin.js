// Robot Twin — stylised 3D robots that mirror the tool calls the model issues
// (stand / sit / stow / look / detect / pick / place / drive / navigate / stop).
// Two interchangeable rigs share one scene, one action queue and one HUD:
//   * dog      — quadruped with a top-mounted manipulator arm
//   * humanoid — biped with two arms (right arm manipulates), head camera
// It is a *schematic* visualisation, not a physical simulation: each tool
// call is mapped to a short keyframed motion.
//
// Integration (all via window events, so agent-ui.js stays a classic script):
//   robot-twin:tool    {name, args, result}       — a non-ack tool call
//   robot-twin:target  {x, y, label}              — a fresh detect target (0-1000)
//   robot-twin:status  {state, ...}               — action watchdog status
//   robot-twin:reset   {}                         — session reset
//
// The designs are deliberately generic and do not reproduce any commercial robot.

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
// [INTERNAL-TEST-ONLY: Real2Sim] Test-only CAD robot rigs (Unitree R1, Rainbow RB-Y1, LG CLOiD)
import { buildReal2SimRigs } from './real2sim-rigs.js?v=2';

const canvas = document.getElementById('robot-twin-canvas');
const panel = document.getElementById('robot-twin');

const C_BODY = 0xeef1f5, C_DARK = 0x3c4043, C_ACCENT = 0x4285f4, C_JOINT = 0x5f6368;
const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.08, ...o });
const glow = (c) => new THREE.MeshStandardMaterial({ color: 0x8ab4f8, emissive: c, emissiveIntensity: 0.9 });

// Shared 2-link planar IK (yaw about y, then shoulder/elbow/wrist pitch about z).
// base: arm base in robot frame (x fwd, y up, z right) for the given body height.
function ik2(base, A1, A2, GRIP, x, y, z, toolPitch) {
  const bx = x - base.x, bz = z - base.z, by = y - base.y;
  const yaw = Math.atan2(-bz, bx);
  const r0 = Math.hypot(bx, bz);
  const r = r0 - Math.cos(toolPitch) * GRIP;
  const h = by - Math.sin(toolPitch) * GRIP;
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

// Planar 2-link leg: thigh hangs along -y from the hip; returns hip/knee angles
// about z for a foot at (dx fwd, dy up) relative to the hip. kneeDir: +1 knee
// forward (human), -1 knee backward (dog hind-leg style).
function legIK(dx, dy, L1, L2, kneeDir) {
  let d = Math.hypot(dx, dy);
  d = clamp(d, Math.abs(L1 - L2) + 0.01, L1 + L2 - 0.001);
  const a = Math.atan2(dx, -dy);
  const beta = Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
  const kInt = Math.acos(clamp((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1));
  return { hip: a + kneeDir * beta, knee: -kneeDir * (Math.PI - kInt) };
}

function makeGripper(parent, GRIP_LEN) {
  const palm = new THREE.Mesh(roundedBox(0.08, 0.07, 0.07, 0.015), mat(C_DARK));
  palm.position.x = 0.05; palm.castShadow = true; parent.add(palm);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.01, 16), glow(0x4285f4));
  lens.rotation.z = Math.PI / 2; lens.position.set(0.07, 0.042, 0); parent.add(lens);
  const fingers = [];
  for (const s of [-1, 1]) {
    const f = new THREE.Group(); f.position.set(0.09, 0, s * 0.025); parent.add(f);
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.03, 0.014), mat(0x202124));
    m.position.x = 0.045; m.castShadow = true; f.add(m);
    fingers.push({ g: f, s });
  }
  const gripPoint = new THREE.Object3D(); gripPoint.position.x = GRIP_LEN - 0.03; parent.add(gripPoint);
  return { fingers, gripPoint, lens };
}

function makeCone(parent, pos, len = 0.9, radius = 0.32) {
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(radius, len, 32, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x4285f4, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  cone.rotation.z = Math.PI / 2;     // apex at pos, opening along +x
  cone.position.set(pos.x + len / 2, pos.y, pos.z);
  parent.add(cone);
  return cone;
}

// ============================================================ DOG RIG ======
function buildDog() {
  const group = new THREE.Group();
  const body = new THREE.Group();
  group.add(body);
  const BODY_L = 0.9, BODY_W = 0.34, BODY_H = 0.2;
  const torso = new THREE.Mesh(roundedBox(BODY_L, BODY_H, BODY_W, 0.06), mat(C_BODY));
  torso.castShadow = true; body.add(torso);
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(BODY_L * 0.7, 0.025, BODY_W + 0.004), mat(C_ACCENT, { roughness: 0.4 }));
  stripe.position.y = 0.02; body.add(stripe);
  const head = new THREE.Mesh(roundedBox(0.12, 0.13, 0.28, 0.04), mat(C_DARK));
  head.position.set(BODY_L / 2 + 0.02, -0.01, 0); head.castShadow = true; body.add(head);
  for (const z of [-0.07, 0.07]) {
    const eye = new THREE.Mesh(new THREE.CircleGeometry(0.022, 20), glow(0x4285f4));
    eye.position.set(BODY_L / 2 + 0.081, 0, z); eye.rotation.y = Math.PI / 2; body.add(eye);
  }
  const L1 = 0.32, L2 = 0.34, HIP_X = 0.33, HIP_Z = 0.2, HIP_Y = -0.06;
  const legs = [];
  for (const [sx, sz] of [[1, -1], [1, 1], [-1, -1], [-1, 1]]) {
    const hip = new THREE.Group(); hip.position.set(sx * HIP_X, HIP_Y, sz * HIP_Z); body.add(hip);
    hip.add(new THREE.Mesh(new THREE.SphereGeometry(0.055, 18, 12), mat(C_JOINT)));
    const thigh = new THREE.Mesh(capsule(0.045, L1), mat(C_DARK)); thigh.position.y = -L1 / 2; thigh.castShadow = true; hip.add(thigh);
    const knee = new THREE.Group(); knee.position.y = -L1; hip.add(knee);
    knee.add(new THREE.Mesh(new THREE.SphereGeometry(0.042, 16, 10), mat(C_ACCENT)));
    const shin = new THREE.Mesh(capsule(0.032, L2), mat(C_DARK)); shin.position.y = -L2 / 2; shin.castShadow = true; knee.add(shin);
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.04, 14, 10), mat(0x202124)); foot.position.y = -L2; knee.add(foot);
    legs.push({ hip, knee, sx, sz, phase: (sx * sz > 0) ? 0 : Math.PI });
  }
  const A1 = 0.5, A2 = 0.46, GRIP_LEN = 0.17;
  const ARM_BASE = new THREE.Vector3(0.28, BODY_H / 2 + 0.03, 0);
  const armYaw = new THREE.Group(); armYaw.position.copy(ARM_BASE); body.add(armYaw);
  armYaw.add(new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.06, 24), mat(C_JOINT)));
  const shoulder = new THREE.Group(); shoulder.position.y = 0.05; armYaw.add(shoulder);
  shoulder.add(new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), mat(C_ACCENT)));
  const upper = new THREE.Mesh(capsule(0.038, A1, true), mat(C_BODY)); upper.position.x = A1 / 2; upper.castShadow = true; shoulder.add(upper);
  const elbow = new THREE.Group(); elbow.position.x = A1; shoulder.add(elbow);
  elbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.048, 16, 10), mat(C_ACCENT)));
  const fore = new THREE.Mesh(capsule(0.032, A2, true), mat(C_BODY)); fore.position.x = A2 / 2; fore.castShadow = true; elbow.add(fore);
  const wrist = new THREE.Group(); wrist.position.x = A2; elbow.add(wrist);
  wrist.add(new THREE.Mesh(new THREE.SphereGeometry(0.036, 14, 10), mat(C_JOINT)));
  const { fingers, gripPoint } = makeGripper(wrist, GRIP_LEN);
  const cone = makeCone(wrist, { x: 0.07, y: 0.042, z: 0 });

  const STAND_H = 0.52, SIT_H = 0.3, REAR_PITCH = 0.95;
  const STOW = { yaw: 0, s: 3.06, e: -2.98, w: -0.06 };
  const base = () => new THREE.Vector3(ARM_BASE.x, STAND_H + ARM_BASE.y + 0.05, 0);
  const ik = (x, y, z, p) => ik2(base(), A1, A2, GRIP_LEN, x, y, z, p);
  const tmpV = new THREE.Vector3(), tmpM = new THREE.Matrix4();
  let pawMode = 'shake';

  return {
    name: 'dog', icon: '🐕', label: 'Spot', spec: 'Quadruped + 6-DOF Top Arm',
    group, gripPoint, cone, flashMat: stripe.material,
    STAND_H, SIT_H, STOW, ik,
    ready: () => ik(0.62, STAND_H + 0.32, 0, -0.25),
    carry: () => ik(0.7, STAND_H + 0.36, 0, -0.35),
    table: { x: 1.05, top: 0.36, half: { x: 0.26, z: 0.42 } },
    homeCam: new THREE.Vector3(2.75, 1.55, -1.95), homeTarget: new THREE.Vector3(0.4, 0.45, 0),
    defaults: { h: STAND_H, pitch: 0, roll: 0, yaw: STOW.yaw, s: STOW.s, e: STOW.e, w: STOW.w, grip: 0, rear: 0, hy: 0, hp: 0, ls: 0, le: 0, paw: 0, pawAmp: 0 },
    // Stand = rear up on the hind legs and stay there. Sit = hind legs down.
    standEnter: (tgt) => { tgt.rear = 1; },
    isUp: (tgt) => tgt.rear > 0.5,
    downEnter: (tgt) => { tgt.rear = 0; tgt.h = STAND_H; tgt.pitch = 0; },
    sitEnter: (tgt) => { tgt.h = SIT_H; tgt.pitch = 0.36; },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.yaw += ang; else if (dir === 'right') tgt.yaw -= ang;
      else if (dir === 'up') tgt.w += ang; else if (dir === 'down') tgt.w -= ang;
    },
    detectEnter: (tgt, setArm) => setArm(ik(0.66, STAND_H + 0.3, 0, -0.75)),
    // Social gestures use the FRONT PAW (the one nearer the camera), like a
    // real dog: shift weight back into a half-sit, then lift the paw.
    gestures: {
      high_wave: (tgt, setArm, ctx) => [
        ...ctx.stowArm(),
        [0.7, () => { pawMode = 'wave'; tgt.h = STAND_H - 0.05; tgt.pitch = 0.24; tgt.roll = 0.06; tgt.paw = 1; tgt.pawAmp = 0; }],
        [2.4, () => { tgt.pawAmp = 1; }],
        [0.8, () => { tgt.pawAmp = 0; tgt.paw = 0; tgt.roll = 0; ctx.restorePosture(); }],
      ],
      shake_hand: (tgt, setArm, ctx) => [
        ...ctx.stowArm(),
        [0.8, () => { pawMode = 'shake'; tgt.h = Math.min(tgt.h, 0.42); tgt.pitch = 0.2; tgt.roll = 0.05; tgt.paw = 1; tgt.pawAmp = 0; }],
        [0.4, () => {}],
        [1.8, () => { tgt.pawAmp = 1; }],
        [0.8, () => { tgt.pawAmp = 0; tgt.paw = 0; tgt.roll = 0; ctx.restorePosture(); }],
      ],
      heart: (tgt, setArm, ctx) => [
        ...ctx.stowArm(),
        [0.7, () => { tgt.rear = 1; tgt.grip = 0.6; }],
        [1.8, () => { setArm({ yaw: 0, s: 1.65, e: -1.1, w: -0.4 }); }],
        [0.8, () => { tgt.rear = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
    },
    update(c) {
      const { cur, t, walkAmt, gaitPhase, gait } = c;
      const breathe = Math.sin(t * 2.2) * 0.006 * (1 - walkAmt);
      const q = cur.rear * REAR_PITCH;
      const rx = -HIP_X, ry = HIP_Y;
      const pivX = rx - (rx * Math.cos(q) - ry * Math.sin(q));
      const pivY = ry - (rx * Math.sin(q) + ry * Math.cos(q));
      body.position.x = pivX;
      body.position.y = cur.h + pivY + breathe + Math.abs(Math.sin(gaitPhase)) * 0.015 * walkAmt;
      body.rotation.set(cur.roll, 0, cur.pitch + q);
      group.rotation.y = gait ? Math.sin(t * 3) * 0.02 : 0;
      body.updateMatrixWorld(true);
      tmpM.copy(body.matrixWorld).invert();
      for (const L of legs) {
        const ph = gaitPhase + L.phase;
        const stride = 0.09 * walkAmt * Math.sign((gait && gait.vx) || 1);
        const footW = tmpV.set(L.sx * HIP_X + Math.sin(ph) * stride, Math.max(0, Math.cos(ph)) * 0.07 * walkAmt, L.sz * HIP_Z);
        if (L.sx < 0) footW.x += 0.12 * (STAND_H - cur.h) / (STAND_H - SIT_H) + 0.08 * cur.rear;
        footW.applyMatrix4(tmpM);
        if (L.sx > 0 && cur.rear > 0.001) {
          const tuckX = L.hip.position.x + 0.16, tuckY = L.hip.position.y - 0.27;
          const wave = Math.sin(t * 6 + (L.sz > 0 ? 0 : Math.PI)) * 0.03 * cur.rear;
          footW.x += (tuckX - footW.x) * cur.rear;
          footW.y += (tuckY + wave - footW.y) * cur.rear;
        }
        if (L.sx > 0 && L.sz < 0 && cur.paw > 0.001) {
          // Gesture paw (front leg nearer the camera), in body frame.
          const hx = L.hip.position.x, hy = L.hip.position.y, A = cur.pawAmp;
          let px, py;
          if (pawMode === 'wave') {      // raised high, sweeping hello
            px = hx + 0.36 + Math.sin(t * 7.5) * 0.08 * A;
            py = hy + 0.32 + Math.cos(t * 7.5) * 0.05 * A;
          } else {                       // offered forward at chest height, pumping
            px = hx + 0.5;
            py = hy - 0.2 + Math.sin(t * 9) * 0.045 * A;
          }
          footW.x += (px - footW.x) * cur.paw;
          footW.y += (py - footW.y) * cur.paw;
        }
        const j = legIK(footW.x - L.hip.position.x, footW.y - L.hip.position.y, L1, L2, -1);
        L.hip.rotation.z = j.hip; L.knee.rotation.z = j.knee;
      }
      armYaw.rotation.y = cur.yaw; shoulder.rotation.z = cur.s; elbow.rotation.z = cur.e; wrist.rotation.z = cur.w;
      for (const f of fingers) f.g.rotation.y = -f.s * (0.05 + cur.grip * 0.45);
    },
  };
}

// ======================================================= HUMANOID RIG ======
function buildHumanoid() {
  const group = new THREE.Group();
  const body = new THREE.Group();          // pelvis root (height = cur.h)
  group.add(body);
  const PEL_W = 0.3;
  const pelvis = new THREE.Mesh(roundedBox(0.2, 0.14, PEL_W, 0.04), mat(C_DARK));
  pelvis.castShadow = true; body.add(pelvis);
  const torsoG = new THREE.Group(); torsoG.position.y = 0.07; body.add(torsoG);
  const TORSO_H = 0.5;
  const torso = new THREE.Mesh(roundedBox(0.26, TORSO_H, 0.4, 0.06), mat(C_BODY));
  torso.position.y = TORSO_H / 2; torso.castShadow = true; torsoG.add(torso);
  const chest = new THREE.Mesh(new THREE.CircleGeometry(0.035, 24), mat(C_ACCENT, { roughness: 0.4 }));
  chest.position.set(0.131, TORSO_H * 0.65, 0); chest.rotation.y = Math.PI / 2; torsoG.add(chest);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 0.06, 16), mat(C_JOINT));
  neck.position.y = TORSO_H + 0.03; torsoG.add(neck);
  const headG = new THREE.Group(); headG.position.y = TORSO_H + 0.06; torsoG.add(headG);
  const head = new THREE.Mesh(roundedBox(0.2, 0.22, 0.22, 0.06), mat(C_BODY));
  head.position.y = 0.11; head.castShadow = true; headG.add(head);
  const visor = new THREE.Mesh(roundedBox(0.03, 0.07, 0.17, 0.02), mat(0x202124));
  visor.position.set(0.1, 0.12, 0); headG.add(visor);
  for (const z of [-0.045, 0.045]) {
    const eye = new THREE.Mesh(new THREE.CircleGeometry(0.016, 18), glow(0x4285f4));
    eye.position.set(0.116, 0.12, z); eye.rotation.y = Math.PI / 2; headG.add(eye);
  }
  const cone = makeCone(headG, { x: 0.12, y: 0.12, z: 0 }, 1.1, 0.36);

  // Arms. Right arm (z>0) is the manipulator; left arm mirrors a relaxed pose.
  const A1 = 0.4, A2 = 0.36, GRIP_LEN = 0.17, SH_Y = TORSO_H - 0.03, SH_Z = 0.25;
  function makeArm(side) {
    const yawG = new THREE.Group(); yawG.position.set(0, SH_Y, side * SH_Z); torsoG.add(yawG);
    yawG.add(new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), mat(C_ACCENT)));
    const sh = new THREE.Group(); yawG.add(sh);
    const up = new THREE.Mesh(capsule(0.045, A1, true), mat(C_BODY)); up.position.x = A1 / 2; up.castShadow = true; sh.add(up);
    const el = new THREE.Group(); el.position.x = A1; sh.add(el);
    el.add(new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 10), mat(C_JOINT)));
    const fo = new THREE.Mesh(capsule(0.036, A2, true), mat(C_DARK)); fo.position.x = A2 / 2; fo.castShadow = true; el.add(fo);
    const wr = new THREE.Group(); wr.position.x = A2; el.add(wr);
    return { yawG, sh, el, wr };
  }
  const R = makeArm(1), Lm = makeArm(-1);
  const { fingers, gripPoint } = makeGripper(R.wr, GRIP_LEN);
  const lhand = new THREE.Mesh(roundedBox(0.1, 0.06, 0.07, 0.015), mat(C_DARK));
  lhand.position.x = 0.05; Lm.wr.add(lhand);

  // Legs: hip -> thigh -> knee -> shin -> foot (knee forward).
  const L1 = 0.42, L2 = 0.42, HIP_Z = 0.1;
  const legs = [];
  for (const sz of [-1, 1]) {
    const hip = new THREE.Group(); hip.position.set(0, -0.06, sz * HIP_Z); body.add(hip);
    hip.add(new THREE.Mesh(new THREE.SphereGeometry(0.06, 18, 12), mat(C_JOINT)));
    const thigh = new THREE.Mesh(capsule(0.055, L1), mat(C_BODY)); thigh.position.y = -L1 / 2; thigh.castShadow = true; hip.add(thigh);
    const knee = new THREE.Group(); knee.position.y = -L1; hip.add(knee);
    knee.add(new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 10), mat(C_ACCENT)));
    const shin = new THREE.Mesh(capsule(0.045, L2), mat(C_DARK)); shin.position.y = -L2 / 2; shin.castShadow = true; knee.add(shin);
    const ankle = new THREE.Group(); ankle.position.y = -L2; knee.add(ankle);
    const foot = new THREE.Mesh(roundedBox(0.22, 0.05, 0.1, 0.02), mat(0x202124));
    foot.position.set(0.05, -0.025, 0); foot.castShadow = true; ankle.add(foot);
    legs.push({ hip, knee, ankle, sz, phase: sz > 0 ? 0 : Math.PI });
  }

  const STAND_H = 0.9, SIT_H = 0.58;
  const STOW = { yaw: 0, s: -1.5, e: -0.15, w: 0.1 };          // arm hanging down
  const base = () => new THREE.Vector3(0, STAND_H + 0.07 + SH_Y, SH_Z);
  const ik = (x, y, z, p) => ik2(base(), A1, A2, GRIP_LEN, x, y, z, p);
  const tmpV = new THREE.Vector3(), tmpM = new THREE.Matrix4();

  return {
    name: 'humanoid', icon: '🧍', label: 'Humanoid', spec: 'Stylised Biped + 2-Arm Manipulator',
    group, gripPoint, cone, flashMat: chest.material,
    STAND_H, SIT_H, STOW, ik,
    ready: () => ik(0.45, STAND_H + 0.55, 0.25, -0.3),
    carry: () => ik(0.42, STAND_H + 0.62, 0.22, -0.2),
    table: { x: 0.72, top: 0.78, half: { x: 0.26, z: 0.42 } },
    homeCam: new THREE.Vector3(3.15, 1.9, -2.3), homeTarget: new THREE.Vector3(0.3, 0.85, 0),
    defaults: { h: STAND_H, pitch: 0, roll: 0, yaw: STOW.yaw, s: STOW.s, e: STOW.e, w: STOW.w, grip: 0, rear: 0, hy: 0, hp: 0, ls: -1.5, le: -0.15, paw: 0, pawAmp: 0 },
    // Stand = upright, chest out, arms at the sides. Sit = squat.
    standEnter: (tgt) => { tgt.h = STAND_H; tgt.pitch = 0; tgt.rear = 1; tgt.ls = -1.5; tgt.le = -0.15; },
    isUp: () => false,                       // nothing to undo before other actions
    downEnter: (tgt) => { tgt.h = STAND_H; tgt.pitch = 0; tgt.rear = 0; },
    sitEnter: (tgt) => { tgt.h = SIT_H; tgt.pitch = 0.25; tgt.rear = 0; tgt.ls = -0.9; tgt.le = -1.2; },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.hy += ang; else if (dir === 'right') tgt.hy -= ang;
      else if (dir === 'up') tgt.hp += ang; else if (dir === 'down') tgt.hp -= ang;
    },
    detectEnter: (tgt, setArm) => { tgt.hp = -0.45; setArm(ik(0.62, STAND_H + 0.5, 0.2, -0.6)); },
    // Natural human poses, set as joint angles (yaw<0 swings the right arm
    // out to the side; s = shoulder raise, e = elbow, w = wrist).
    gestures: {
      // Hello: upper arm out and up (~30°), forearm vertical, palm open;
      // the forearm swings side to side from the elbow.
      high_wave: (tgt, setArm, ctx) => [
        [0.6, () => { setArm({ yaw: -1.45, s: 0.55, e: 1.0, w: -0.05 }); tgt.grip = 1; tgt.hp = 0.06; tgt.hy = -0.18; }],
        ...[0.62, 1.38, 0.62, 1.38, 0.62, 1.38, 1.0].map(e => [0.27, () => { tgt.e = e; }]),
        [0.7, () => { ctx.restoreArm(); tgt.grip = 0; tgt.hp = 0; tgt.hy = 0; }],
      ],
      // Handshake: elbow near the side, forearm horizontal forward, slight
      // forward lean and nod; three small pumps from the elbow.
      shake_hand: (tgt, setArm, ctx) => [
        [0.7, () => { setArm({ yaw: 0.12, s: -1.15, e: 1.05, w: -0.05 }); tgt.grip = 1; tgt.pitch = -0.07; tgt.hp = -0.12; }],
        [0.4, () => { tgt.grip = 0.35; }],
        ...[1.17, 0.93, 1.17, 0.93, 1.17, 1.05].map(e => [0.2, () => { tgt.e = e; }]),
        [0.4, () => { tgt.grip = 1; }],
        [0.7, () => { ctx.restoreArm(); tgt.grip = 0; tgt.pitch = 0; tgt.hp = 0; }],
      ],
      heart: (tgt, setArm, ctx) => [
        [0.7, () => { setArm({ yaw: -0.5, s: 0.85, e: 1.35, w: -0.4 }); tgt.ls = 0.85; tgt.le = 1.35; tgt.hp = 0.08; }],
        [1.8, () => { tgt.hp = -0.06; }],
        [0.7, () => { ctx.restoreArm(); tgt.ls = -1.5; tgt.le = -0.15; tgt.hp = 0; }],
      ],
    },
    update(c) {
      const { cur, t, walkAmt, gaitPhase, gait } = c;
      const breathe = Math.sin(t * 1.8) * 0.005 * (1 - walkAmt);
      body.position.y = cur.h + breathe + Math.abs(Math.sin(gaitPhase)) * 0.02 * walkAmt;
      body.rotation.set(cur.roll, 0, 0);
      torsoG.rotation.z = cur.pitch + (gait ? 0.08 : 0) - 0.05 * cur.rear;
      torsoG.rotation.x = Math.sin(gaitPhase) * 0.03 * walkAmt;
      headG.rotation.y = cur.hy; headG.rotation.z = cur.hp;
      // Legs: feet planted below the hips, stride + lift when walking.
      body.updateMatrixWorld(true);
      tmpM.copy(body.matrixWorld).invert();
      for (const L of legs) {
        const ph = gaitPhase + L.phase;
        const stride = 0.16 * walkAmt * Math.sign((gait && gait.vx) || 1);
        const squat = (STAND_H - cur.h) / (STAND_H - SIT_H);
        const footW = tmpV.set(Math.sin(ph) * stride + 0.05 * squat, Math.max(0, Math.cos(ph)) * 0.08 * walkAmt, L.sz * HIP_Z);
        footW.applyMatrix4(tmpM);
        const j = legIK(footW.x - L.hip.position.x, footW.y - L.hip.position.y, L1, L2, +1);
        L.hip.rotation.z = j.hip; L.knee.rotation.z = j.knee;
        L.ankle.rotation.z = -(j.hip + j.knee);     // keep the foot level
      }
      // Right (manipulator) arm from the shared pose; left arm relaxed / swinging.
      R.yawG.rotation.y = cur.yaw; R.sh.rotation.z = cur.s; R.el.rotation.z = cur.e; R.wr.rotation.z = cur.w;
      for (const f of fingers) f.g.rotation.y = -f.s * (0.05 + cur.grip * 0.45);
      const swing = Math.sin(gaitPhase) * 0.5 * walkAmt;
      Lm.yawG.rotation.y = 0; Lm.sh.rotation.z = cur.ls + swing; Lm.el.rotation.z = cur.le; Lm.wr.rotation.z = 0;
      Lm.sh.rotation.x = 0.08;   // elbows slightly out
      R.sh.rotation.x = -0.08;
    },
  };
}

// ================================================================ MAIN ======
function init() {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minDistance = 1.2; controls.maxDistance = 7;
  controls.maxPolarAngle = Math.PI * 0.49; controls.enablePan = false;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9bb0, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2.5, 4, -2);   // key light from the camera side sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 0.5, far: 10 });
  scene.add(sun);
  const world = new THREE.Group(); scene.add(world);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(6, 64), new THREE.ShadowMaterial({ opacity: 0.18 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(12, 24, 0x8ab4f8, 0x9aa0a6);
  grid.material.transparent = true; grid.material.opacity = 0.35; world.add(grid);

  // Table (manipulation zone). Hidden until detect / pick / place happen.
  const table = new THREE.Group(); scene.add(table);
  const tableTop = new THREE.Mesh(new THREE.BoxGeometry(1, 0.03, 1), mat(0xc9b79c));
  tableTop.castShadow = tableTop.receiveShadow = true; table.add(tableTop);
  const tableLegs = [];
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.03, 1, 0.03), mat(0x8d7b63));
    leg.castShadow = true; table.add(leg); tableLegs.push({ leg, sx, sz });
  }
  let T = { x: 1, top: 0.36, half: { x: 0.26, z: 0.42 } };
  function layoutTable(cfg) {
    T = cfg;
    table.position.x = T.x;
    tableTop.scale.set(T.half.x * 2, 1, T.half.z * 2);
    tableTop.position.y = T.top - 0.015;
    for (const { leg, sx, sz } of tableLegs) {
      leg.scale.y = T.top - 0.03;
      leg.position.set(sx * (T.half.x - 0.03), (T.top - 0.03) / 2, sz * (T.half.z - 0.03));
    }
  }
  let tableShow = 0, tableAmt = 0, tableHideAt = Infinity;
  function showTable(secs = 14) { tableShow = 1; tableHideAt = now() + secs; }
  function hideTable() { tableShow = 0; }

  const item = new THREE.Mesh(roundedBox(0.07, 0.09, 0.07, 0.015), mat(0xea4335, { roughness: 0.35 }));
  item.castShadow = true; item.visible = false; scene.add(item);
  let itemHeld = false, dropTo = null;
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.075, 32),
    new THREE.MeshBasicMaterial({ color: 0x34a853, transparent: true, opacity: 0, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; scene.add(ring);
  let ringUntil = 0;

  // ----------------------------------------------------------- rigs --------
  // [INTERNAL-TEST-ONLY: Real2Sim] Include Unitree R1, Rainbow RB-Y1, and LG CLOiD rigs
  const rigs = { dog: buildDog(), ...buildReal2SimRigs(THREE), humanoid: buildHumanoid() };
  const RIG_ORDER = ['dog', 'unitree_r1', 'rby1', 'lg_cloid', 'humanoid'];
  for (const r of Object.values(rigs)) { scene.add(r.group); r.group.visible = false; }
  let R = null;
  const cur = {}, tgt = {};
  let armStowed = true, sitting = false, gait = null, heading = 0, scanUntil = 0, flash = null;
  const pos = new THREE.Vector2();
  let queue = [], step = null, pendingTarget = null;

  function setArm(j) { if (j) Object.assign(tgt, j); }

  function syncReal2SimBar() {
    if (!R) return;
    const specEl = document.getElementById('twin-real2sim-spec');
    if (specEl) specEl.textContent = R.spec || R.label || R.name;
    document.querySelectorAll('[data-real2sim-rig]').forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-real2sim-rig') === R.name);
    });
  }

  function activateRig(name, { keepView = false } = {}) {
    const next = rigs[name] || rigs.dog;
    if (R === next) { syncReal2SimBar(); return; }
    const wasHeld = itemHeld;
    if (wasHeld) { scene.add(item); itemHeld = false; }
    if (R) R.group.visible = false;
    R = next; R.group.visible = true;
    if (R.ensureLoaded) R.ensureLoaded();
    clearQueue(); gait = null; armStowed = true; sitting = false;
    for (const k of Object.keys(tgt)) delete tgt[k];
    for (const k of Object.keys(cur)) delete cur[k];
    Object.assign(tgt, R.defaults); Object.assign(cur, R.defaults);
    layoutTable(R.table);
    if (!wasHeld) {
      hideTable();
      item.visible = false;
      pendingTarget = null;
      dropTo = null;
      ring.material.opacity = 0;
    }
    if (wasHeld) { armStowed = false; setArm(R.carry()); tgt.grip = 0.25; Object.assign(cur, tgt); grabItem(); }
    if (!keepView) { camera.position.copy(R.homeCam); controls.target.copy(R.homeTarget); controls.update(); }
    localStorage.setItem('twin_rig', R.name);
    const sw = document.getElementById('twin-switch');
    if (sw) { sw.textContent = R.icon; sw.title = `Robot: ${R.label || R.name} — click to switch`; }
    const actEl = document.getElementById('twin-action');
    if (actEl && !wasHeld) { actEl.textContent = 'Idle'; actEl.classList.remove('active'); }
    syncReal2SimBar();
  }

  // Map 0-1000 image coordinates (camera looking at the table) to the table top.
  function imageToTable(x, y) {
    const tx = T.x + T.half.x * 0.72 * (1 - 2 * clamp(y / 1000, 0, 1));
    const tz = T.half.z * 0.85 * (2 * clamp(x / 1000, 0, 1) - 1);
    const v = new THREE.Vector3(tx, T.top, tz); v.ix = x; v.iy = y; return v;
  }

  // -------------------------------------------------------------- actions --
  function enqueue(steps) { queue.push(...steps); }
  function clearQueue() { queue = []; step = null; }
  const wait = (dur, enter) => ({ dur, enter: enter || (() => {}) });

  function ensureStanding() {
    if (!sitting && !R.isUp(tgt)) return [];
    return [wait(0.9, () => { sitting = false; R.downEnter(tgt); })];
  }
  function ensureDeployed() {
    if (!armStowed) return [];
    return [wait(0.9, () => { armStowed = false; setArm(R.ready()); tgt.grip = 0.2; })];
  }
  function stowSteps() {
    return armStowed ? [] : [wait(0.8, () => { armStowed = true; setArm(R.STOW); if (!itemHeld) tgt.grip = 0; })];
  }

  const actions = {
    stand: () => enqueue([
      ...(sitting ? [wait(0.9, () => { sitting = false; R.downEnter(tgt); })] : []),
      wait(1.1, () => R.standEnter(tgt)),
    ]),
    sit: () => enqueue([...stowSteps(), ...(R.isUp(tgt) ? [wait(0.9, () => R.downEnter(tgt))] : []),
      wait(1.0, () => { sitting = true; R.sitEnter(tgt); hideTable(); })]),
    stow: () => enqueue([...stowSteps(), wait(0.1, () => { if (!itemHeld) hideTable(); })]),
    stow_arm: () => actions.stow(),
    deploy_arm: () => enqueue([...ensureStanding(), wait(0.9, () => { armStowed = false; setArm(R.ready()); })]),
    open_gripper: () => enqueue([wait(0.4, () => { tgt.grip = 1; releaseItem(); })]),
    close_gripper: () => enqueue([wait(0.4, () => { tgt.grip = 0; })]),
    look: (a) => {
      const ang = Number(a.angle_rad) || 0.2;
      enqueue([...ensureStanding(), ...(R.name === 'dog' ? ensureDeployed() : []),
        wait(0.6, () => { R.look(tgt, String(a.direction || ''), ang); scanUntil = now() + 0.8; })]);
    },
    turn_head_to_uv: () => enqueue([wait(0.6, () => { scanUntil = now() + 0.8; })]),
    detect: () => enqueue([...ensureStanding(), ...ensureDeployed(),
      wait(1.2, () => { showTable(); R.detectEnter(tgt, setArm); scanUntil = now() + 1.6; })]),
    pick: (a, r) => {
      const t = targetFrom(r && r.target, a.grasp_x, a.grasp_y) || pendingTarget || imageToTable(500, 500);
      showTable(18);
      if (!itemHeld) { item.visible = true; dropTo = null; item.position.set(t.x, T.top + 0.045, t.z); }
      const above = t.y + (R.approachH || 0.28);
      const graspY = t.y + (R.graspH || 0.05);
      enqueue([...ensureStanding(), ...ensureDeployed(),
        wait(0.35, () => {
          setActionLabel('pick', a);
          showTable(18);
          if (!itemHeld) { item.visible = true; dropTo = null; item.position.set(t.x, T.top + 0.045, t.z); }
          if (R.preparePick) R.preparePick(t.x, t.y, t.z);
          tgt.grip = 1;
          tgt.hp = -0.35;
          if (R.look) R.look(tgt, 'down', 0.24);
        }),
        wait(0.85, () => setArm(R.ik(t.x, above, t.z, -Math.PI / 2))),
        wait(0.7, () => setArm(R.ik(t.x, graspY, t.z, -Math.PI / 2))),
        wait(0.45, () => { tgt.grip = 0.25; }),
        wait(0.05, () => grabItem()),
        wait(0.75, () => setArm(R.ik(t.x - 0.05, above + 0.04, t.z, -Math.PI / 2))),
        wait(0.75, () => { setArm(R.carry()); tgt.hp = 0; if ('head_pitch_joint' in tgt) tgt.head_pitch_joint = 0.08; if ('head_1' in tgt) tgt.head_1 = 0.18; if ('neck_pitch_joint' in tgt) tgt.neck_pitch_joint = 0.08; }),
      ]);
    },
    place: (a, r) => {
      const t = targetFrom(r && r.target, a.x, a.y) || pendingTarget || imageToTable(500, 300);
      const above = t.y + (R.approachH || 0.28);
      const placeY = t.y + (R.graspH || 0.05) + 0.03;
      enqueue([...ensureStanding(), ...ensureDeployed(),
        wait(0.9, () => {
          setActionLabel('place', a);
          showTable(14);
          showRing(t);
          if (!itemHeld && R.preparePick) R.preparePick(t.x, t.y, t.z);
          tgt.hp = -0.35;
          if (R.look) R.look(tgt, 'down', 0.22);
          setArm(R.ik(t.x, above, t.z, -Math.PI / 2));
        }),
        wait(0.7, () => setArm(R.ik(t.x, placeY, t.z, -Math.PI / 2))),
        wait(0.4, () => { tgt.grip = 1; }),
        wait(0.05, () => releaseItem(t)),
        wait(0.7, () => setArm(R.ik(t.x - 0.05, above, t.z, -Math.PI / 2))),
        wait(0.8, () => { setArm(R.ready()); tgt.grip = 0.2; tgt.hp = 0; if ('head_pitch_joint' in tgt) tgt.head_pitch_joint = 0.08; if ('head_1' in tgt) tgt.head_1 = 0.18; if ('neck_pitch_joint' in tgt) tgt.neck_pitch_joint = 0.08; }),
      ]);
    },
    drive: (a) => {
      const dur = clamp(Number(a.duration) || 1, 0.6, 3);
      enqueue([...ensureStanding(), wait(dur, () => {
        hideTable();
        gait = { vx: Number(a.v_x) || 0, vy: Number(a.v_y) || 0, vrot: Number(a.v_rot) || 0 };
      }), wait(0.3, () => { gait = null; })]);
    },
    navigate: () => enqueue([...ensureStanding(), ...stowSteps(),
      wait(3.0, () => { hideTable(); gait = { vx: 0.5, vy: 0, vrot: 0.15 }; }),
      wait(0.3, () => { gait = null; })]),
    make_gesture: (a) => {
      const g = R.gestures[a.gesture] || R.gestures.high_wave;
      const wasSitting = sitting;
      const ctx = {
        // Dog: arm out of the way first (the paw does the gesture).
        stowArm: () => (armStowed ? [] : [[0.7, () => { armStowed = true; setArm(R.STOW); tgt.grip = 0; }]]),
        restorePosture: () => { if (wasSitting) R.sitEnter(tgt); else { tgt.h = R.STAND_H; tgt.pitch = 0; } },
        restoreArm: () => setArm(armStowed ? R.STOW : R.ready()),
      };
      enqueue([...(R.isUp(tgt) ? [wait(0.9, () => R.downEnter(tgt))] : []),
        ...g(tgt, setArm, ctx).map(([d, f]) => wait(d, f))]);
    },
    stop: () => { clearQueue(); gait = null; Object.assign(tgt, { ...cur }); flashColor(0xea4335, 0.8); },
  };

  function targetFrom(t, x, y) {
    if (x !== undefined && y !== undefined && Number.isFinite(Number(x)) && Number.isFinite(Number(y))) {
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
    itemHeld = true; R.gripPoint.add(item);
    item.position.set(0, 0, 0); item.rotation.set(0, 0, Math.PI / 2);
  }
  function releaseItem(t) {
    if (!itemHeld) return;
    itemHeld = false;
    const wp = new THREE.Vector3(); R.gripPoint.getWorldPosition(wp);
    scene.add(item); item.rotation.set(0, 0, 0); item.position.copy(wp);
    const onTable = Math.abs(wp.x - T.x) < T.half.x && Math.abs(wp.z) < T.half.z;
    dropTo = t ? new THREE.Vector3(t.x, T.top + 0.045, t.z)
               : new THREE.Vector3(wp.x, onTable ? T.top + 0.045 : 0.045, wp.z);
  }
  function showRing(t) { ring.position.set(t.x, T.top + 0.002, t.z); ring.material.opacity = 0.9; ringUntil = now() + 6; }
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
    if (name === 'make_gesture') {
      label = a.gesture === 'shake_hand' ? '🤝 Shake hand'
            : a.gesture === 'heart'      ? '🫶 Heart'
            :                              '👋 High wave';
    }
    actionEl.textContent = label; actionEl.classList.add('active');
  }
  window.addEventListener('robot-twin:tool', (ev) => {
    const { name, args, result } = ev.detail || {};
    const fn = actions[name];
    if (!fn) return;
    const res = (result && typeof result === 'object') ? result : {};
    if (!step && queue.length === 0) setActionLabel(name, args || {});
    if (res.error || res.status === 'rejected' || res.status === 'error') { flashColor(0xea4335, 0.8); return; }
    fn(args || {}, res);
  });
  window.addEventListener('robot-twin:target', (ev) => {
    const p = ev.detail || {};
    const t = imageToTable(Number(p.x), Number(p.y));
    pendingTarget = t; showTable();
    if (itemHeld) showRing(t);
    else { item.visible = true; dropTo = null; item.position.set(t.x, T.top + 0.045, t.z); }
  });
  window.addEventListener('robot-twin:status', (ev) => {
    if (!statusEl) return;
    const st = (ev.detail && ev.detail.state) || 'idle';
    const MAP = { pending: ['⏳ verifying', 'pending'], retry: ['🔁 retry', 'pending'], confirmed: ['✅ confirmed', 'ok'],
      timed_out: ['⌛ timed out', 'bad'], failed: ['❌ failed', 'bad'], aborted: ['❌ aborted', 'bad'] };
    const m = MAP[st];
    if (!m) { statusEl.classList.add('hidden'); return; }
    statusEl.textContent = m[0]; statusEl.className = 'twin-status ' + m[1];
    if (st === 'confirmed') flashColor(0x34a853, 0.8);
    if (m[1] === 'bad') flashColor(0xea4335, 0.8);
  });
  window.addEventListener('robot-twin:reset', () => {
    clearQueue(); gait = null; itemHeld = false; scene.add(item); item.visible = false;
    pendingTarget = null; dropTo = null; ring.material.opacity = 0; hideTable();
    sitting = false; armStowed = true; Object.assign(tgt, R.defaults);
    if (actionEl) { actionEl.textContent = 'Idle'; actionEl.classList.remove('active'); }
    if (statusEl) statusEl.classList.add('hidden');
  });
  window.addEventListener('robot-twin:rig-loaded', (ev) => {
    const d = ev.detail || {};
    if (R && d.name === R.name && itemHeld) R.gripPoint.add(item);
  });
  const resetBtn = document.getElementById('twin-reset');
  if (resetBtn) resetBtn.onclick = () => { camera.position.copy(R.homeCam); controls.target.copy(R.homeTarget); controls.update(); };
  const switchBtn = document.getElementById('twin-switch');
  if (switchBtn) {
    switchBtn.onclick = () => {
      const idx = RIG_ORDER.indexOf(R.name);
      activateRig(RIG_ORDER[(idx + 1) % RIG_ORDER.length]);
    };
  }
  const expandBtn = document.getElementById('twin-expand');
  function setExpanded(exp) {
    panel.classList.toggle('expanded', exp);
    if (expandBtn) {
      expandBtn.setAttribute('aria-expanded', String(exp));
      expandBtn.title = exp ? 'Restore 3D view (1×)' : 'Expand 3D view (4×)';
      expandBtn.setAttribute('aria-label', exp ? 'Restore 3D view' : 'Expand 3D view');
    }
    if (exp) {
      // [INTERNAL-TEST-ONLY: Real2Sim] Preload all CAD bundles when expanded so switching is instant
      for (const r of Object.values(rigs)) {
        if (r.ensureLoaded) r.ensureLoaded();
      }
      syncReal2SimBar();
    }
    resize();
  }
  if (expandBtn) {
    expandBtn.onclick = (e) => {
      e.stopPropagation();
      setExpanded(!panel.classList.contains('expanded'));
    };
  }
  // [INTERNAL-TEST-ONLY: Real2Sim] Wire expanded-mode robot model selector chips & quick action tests
  document.querySelectorAll('[data-real2sim-rig]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const rigName = btn.getAttribute('data-real2sim-rig');
      if (rigName) activateRig(rigName);
    });
  });
  document.querySelectorAll('[data-real2sim-demo]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const demo = btn.getAttribute('data-real2sim-demo');
      if (demo === 'stand') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'stand', args: {}, result: {} } }));
      } else if (demo === 'sit') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'sit', args: {}, result: {} } }));
      } else if (demo === 'wave') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'make_gesture', args: { gesture: 'high_wave' }, result: {} } }));
      } else if (demo === 'shake') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'make_gesture', args: { gesture: 'shake_hand' }, result: {} } }));
      } else if (demo === 'heart') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'make_gesture', args: { gesture: 'heart' }, result: {} } }));
      } else if (demo === 'detect') {
        window.dispatchEvent(new CustomEvent('robot-twin:target', { detail: { x: 640, y: 520, label: 'red_cube' } }));
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'detect', args: { instruction: 'red cube' }, result: {} } }));
      } else if (demo === 'pick_place') {
        clearQueue();
        window.dispatchEvent(new CustomEvent('robot-twin:target', { detail: { x: 660, y: 540, label: 'target' } }));
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'pick', args: { grasp_x: 660, grasp_y: 540, grasp_part: 'body' }, result: {} } }));
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'place', args: { x: 420, y: 460 }, result: {} } }));
      } else if (demo === 'drive') {
        window.dispatchEvent(new CustomEvent('robot-twin:tool', { detail: { name: 'drive', args: { v_x: 0.45, v_y: 0, v_rot: 0.12, duration: 2.4 }, result: {} } }));
      }
    });
  });
  activateRig(localStorage.getItem('twin_rig') || 'dog');

  // ----------------------------------------------------------- main loop --
  const clock = new THREE.Clock();
  let t = 0, walkAmt = 0, gaitPhase = 0, idleAt = 0;
  function now() { return t; }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }
  new ResizeObserver(resize).observe(canvas);

  function frame() {
    requestAnimationFrame(frame);
    if (panel.offsetParent === null) return;
    resize();
    const dt = Math.min(clock.getDelta(), 0.15);
    t += dt;
    if (!step || t >= step.until) {
      step = queue.shift() || null;
      if (step) { step.until = t + step.dur; step.enter(); idleAt = t + step.dur + 2.5; }
      else if (actionEl && !gait && actionEl.classList.contains('active') && t > idleAt) actionEl.classList.remove('active');
    }
    const k = 1 - Math.exp(-dt * 7);
    for (const key of Object.keys(tgt)) {
      if (!Number.isFinite(cur[key])) cur[key] = tgt[key];
      else cur[key] += (tgt[key] - cur[key]) * k;
    }

    let walking = 0;
    if (gait) {
      heading += gait.vrot * dt;
      const fx = Math.cos(heading), fz = -Math.sin(heading);
      pos.x += (gait.vx * fx + gait.vy * fz) * dt;
      pos.y += (gait.vx * fz - gait.vy * fx) * dt;
      walking = 1;
    }
    world.rotation.y = -heading;
    grid.position.set(-(pos.x % 0.5), 0, -(pos.y % 0.5));
    walkAmt += ((walking ? 1 : 0) - walkAmt) * k;
    gaitPhase += dt * (R.name === 'dog' ? 9 : 6.5) * walkAmt;

    R.update({ cur, t, dt, k, walkAmt, gaitPhase, gait });

    const scanning = t < scanUntil;
    R.cone.material.opacity += ((scanning ? 0.14 + Math.sin(t * 14) * 0.05 : 0) - R.cone.material.opacity) * k;

    // Table: grow in / shrink away; auto-hide after the manipulation is over.
    if (tableShow && !itemHeld && t > tableHideAt) tableShow = 0;
    tableAmt += (tableShow - tableAmt) * (1 - Math.exp(-dt * 6));
    table.visible = tableAmt > 0.01;
    table.scale.setScalar(Math.max(0.001, tableAmt));
    table.position.y = 0;
    if (!itemHeld && !tableShow && tableAmt < 0.05) item.visible = false;
    item.scale.setScalar(itemHeld ? 1 : Math.max(0.001, tableAmt));
    if (dropTo && !itemHeld) {
      item.position.lerp(dropTo, 1 - Math.exp(-dt * 10));
      if (item.position.distanceTo(dropTo) < 0.002) dropTo = null;
    }
    if (t > ringUntil || !tableShow) ring.material.opacity *= 0.95; else ring.scale.setScalar(1 + Math.sin(t * 5) * 0.12);

    const fl = flash && t < flash.until ? flash.color : null;
    R.flashMat.emissive.setHex(fl || 0x000000);
    R.flashMat.emissiveIntensity = fl ? 0.6 + Math.sin(t * 20) * 0.4 : 0;

    controls.update();
    renderer.render(scene, camera);
  }
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
  g.rotateX(-Math.PI / 2);
  g.center();
  return g;
}

// Entry point last: module-level consts above must be initialised first.
if (canvas && panel) init();
