/**
 * [INTERNAL-TEST-ONLY: Real2Sim]
 * Multi-Embodiment CAD Digital Twin Rigs for Gemini Real2Sim Showcase.
 *
 * NOTE: This module is an internal test-only feature (`dev` branch /
 * `gemini-robotic-er2-test`) and must be skipped when merging to `main` / stable.
 *
 * Provides three authentic CAD robot rigs compatible with `robot-twin.js`:
 *   1. `unitree_r1` — Unitree R1-EDU (宇树 R1, 26-DOF Bipedal Humanoid + Dual 15-DOF 5-Finger Dex5 Hands)
 *   2. `rby1`       — Rainbow Robotics RB-Y1 (28-DOF Bimanual Wheeled Mobile Manipulator, MuJoCo Menagerie)
 *   3. `lg_cloid`   — LG CLOiD (36-DOF CES 2026 Home Humanoid + 60 FPS OLED Visor + Dual 5-Finger Hands)
 */

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function solve3x3(M, b) {
  const a = M.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < 3; i++) {
    let maxRow = i;
    for (let k = i + 1; k < 3; k++) {
      if (Math.abs(a[k][i]) > Math.abs(a[maxRow][i])) maxRow = k;
    }
    [a[i], a[maxRow]] = [a[maxRow], a[i]];
    const pivot = a[i][i] || 1e-8;
    for (let k = i + 1; k < 3; k++) {
      const factor = a[k][i] / pivot;
      for (let j = i; j <= 3; j++) a[k][j] -= factor * a[i][j];
    }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    let sum = a[i][3];
    for (let j = i + 1; j < 3; j++) sum -= a[i][j] * x[j];
    x[i] = sum / (a[i][i] || 1e-8);
  }
  return x;
}

function makeVisionCone(THREE) {
  const coneGeo = new THREE.ConeGeometry(0.36, 0.72, 24, 1, true);
  coneGeo.translate(0, -0.36, 0);
  coneGeo.rotateZ(-Math.PI / 2); // +X forward in MuJoCo link frame
  const coneMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  return new THREE.Mesh(coneGeo, coneMat);
}

/**
 * Builds the Unitree R1/R2 15-Joint 5-Finger Anthropomorphic Dexterous Hand (`Dex5`)
 * mounted at `x = 0.026 m` on `left_wrist_roll_link` / `right_wrist_roll_link`.
 */
function makeUnitreeFiveFingerHand(THREE, side) {
  const innerSign = side === 'left' ? -1 : 1; // In MuJoCo Z-up frame: +Y is left, -Y is right
  // Wait: in Unitree R1's STL local coordinates for wrist_roll_link:
  // In build_unitree_r1_twin_bundle.py, vertices are in raw MuJoCo frame (x, y, z) where +X is distal along forearm,
  // +Y is left, +Z is dorsal/up. Let's orient the hand in MuJoCo wrist_roll_link frame:
  // +X = distal (finger extension), +Z = dorsal (back of hand), -Z = palmar (palm underside),
  // +Y (for right hand) or -Y (for left hand) = radial (thumb side facing inward toward table center).
  const thumbSideY = side === 'left' ? -1 : 1;
  const root = new THREE.Group();
  root.position.set(0.026, 0.0, 0.0);

  const titaniumMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.30,
    metalness: 0.52,
  });
  const jointPinMat = new THREE.MeshStandardMaterial({
    color: 0x475569,
    roughness: 0.25,
    metalness: 0.72,
  });
  const whiteArmorMat = new THREE.MeshStandardMaterial({
    color: 0xe8eef6,
    roughness: 0.20,
    metalness: 0.16,
  });
  const palmarElastomerMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.55,
    metalness: 0.12,
  });
  const sensorMat = new THREE.MeshStandardMaterial({
    color: side === 'left' ? 0x38bdf8 : 0x10b981,
    roughness: 0.35,
    metalness: 0.12,
    emissive: new THREE.Color(side === 'left' ? 0x0284c7 : 0x059669),
    emissiveIntensity: 0.40,
  });

  // Build in Three.js hand-local frame (+X distal, +Y dorsal, +Z radial) and rotate +90deg around X
  // so +Y_local -> +Z_mj (dorsal) and +Z_local -> -Y_mj!
  const handLocal = new THREE.Group();
  handLocal.rotation.x = Math.PI / 2;
  root.add(handLocal);
  const sgn = -thumbSideY;

  const collarMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.021, 0.023, 0.012, 18), titaniumMat);
  collarMesh.rotation.z = Math.PI / 2;
  collarMesh.position.set(0.005, 0.0, 0.0);
  handLocal.add(collarMesh);

  const wristRing = new THREE.Mesh(new THREE.CylinderGeometry(0.0222, 0.0222, 0.0025, 18), sensorMat);
  wristRing.rotation.z = Math.PI / 2;
  wristRing.position.set(0.009, 0.0, 0.0);
  handLocal.add(wristRing);

  const palmChassis = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.014, 0.058), titaniumMat);
  palmChassis.position.set(0.032, 0.0, 0.0);
  palmChassis.castShadow = true;
  handLocal.add(palmChassis);

  const dorsalShell = new THREE.Mesh(new THREE.BoxGeometry(0.040, 0.0055, 0.055), whiteArmorMat);
  dorsalShell.position.set(0.032, 0.0085, 0.0);
  dorsalShell.castShadow = true;
  handLocal.add(dorsalShell);

  const makePhalanx = (L, r, isDistal) => {
    const seg = new THREE.Group();
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.02, r * 1.02, r * 2.05, 12), jointPinMat);
    pin.rotation.x = Math.PI / 2;
    seg.add(pin);

    const rTip = isDistal ? r * 0.78 : r * 0.90;
    const bodyCyl = new THREE.Mesh(new THREE.CylinderGeometry(rTip, r * 0.96, L, 12), titaniumMat);
    bodyCyl.rotation.z = -Math.PI / 2;
    bodyCyl.position.set(L * 0.5, 0, 0);
    bodyCyl.castShadow = true;
    seg.add(bodyCyl);

    const shield = new THREE.Mesh(new THREE.BoxGeometry(L * 0.80, r * 0.52, r * 1.68), whiteArmorMat);
    shield.position.set(L * 0.5, r * 0.68, 0);
    seg.add(shield);

    const gripStrip = new THREE.Mesh(
      new THREE.BoxGeometry(L * 0.74, r * 0.38, r * 1.45),
      isDistal ? sensorMat : palmarElastomerMat
    );
    gripStrip.position.set(L * (isDistal ? 0.56 : 0.50), -r * 0.68, 0);
    seg.add(gripStrip);

    if (isDistal) {
      const tipDome = new THREE.Mesh(new THREE.SphereGeometry(rTip * 1.02, 12, 10), titaniumMat);
      tipDome.position.set(L, 0, 0);
      seg.add(tipDome);
    }
    return seg;
  };

  // Thumb (3 joints)
  const thumbMount = new THREE.Group();
  thumbMount.position.set(0.022, -0.006, sgn * 0.029);
  handLocal.add(thumbMount);

  const thumbCmc = new THREE.Group();
  thumbMount.add(thumbCmc);
  const L_tm = 0.024, L_tp = 0.021, L_td = 0.019;
  thumbCmc.add(makePhalanx(L_tm, 0.0072, false));

  const thumbMcp = new THREE.Group();
  thumbMcp.position.set(L_tm, 0, 0);
  thumbCmc.add(thumbMcp);
  thumbMcp.add(makePhalanx(L_tp, 0.0066, false));

  const thumbIp = new THREE.Group();
  thumbIp.position.set(L_tp, 0, 0);
  thumbMcp.add(thumbIp);
  thumbIp.add(makePhalanx(L_td, 0.0060, true));

  const fingerSpecs = [
    { name: 'index',  x: 0.054, y: 0.001, z: sgn * 0.022,  splay: sgn * 0.045,  L1: 0.024, L2: 0.017, L3: 0.014, r: 0.0054 },
    { name: 'middle', x: 0.057, y: 0.002, z: sgn * 0.007,  splay: 0.000,        L1: 0.027, L2: 0.019, L3: 0.015, r: 0.0056 },
    { name: 'ring',   x: 0.054, y: 0.001, z: -sgn * 0.008, splay: -sgn * 0.040, L1: 0.024, L2: 0.017, L3: 0.014, r: 0.0053 },
    { name: 'pinky',  x: 0.049, y: 0.000, z: -sgn * 0.022, splay: -sgn * 0.085, L1: 0.019, L2: 0.014, L3: 0.012, r: 0.0048 },
  ];
  const fingers = {};
  for (const spec of fingerSpecs) {
    const mcpGroup = new THREE.Group();
    mcpGroup.position.set(spec.x, spec.y, spec.z);
    mcpGroup.rotation.y = spec.splay;
    handLocal.add(mcpGroup);
    mcpGroup.add(makePhalanx(spec.L1, spec.r, false));

    const pipGroup = new THREE.Group();
    pipGroup.position.set(spec.L1, 0, 0);
    mcpGroup.add(pipGroup);
    pipGroup.add(makePhalanx(spec.L2, spec.r * 0.92, false));

    const dipGroup = new THREE.Group();
    dipGroup.position.set(spec.L2, 0, 0);
    pipGroup.add(dipGroup);
    dipGroup.add(makePhalanx(spec.L3, spec.r * 0.85, true));

    fingers[spec.name] = { mcpGroup, pipGroup, dipGroup, baseSplay: spec.splay };
  }

  const gripAnchor = new THREE.Group();
  gripAnchor.position.set(0.068, -0.026, sgn * 0.008);
  handLocal.add(gripAnchor);

  function setCloseAmount(closeAmt, mode = 'normal') {
    const c = clamp(closeAmt, 0, 1);
    let tAbd = lerp(0.22, 1.05, c);
    let tMcp = lerp(0.12, 0.72, c);
    let tIp  = lerp(0.10, 0.82, c);
    const fFlex = {
      index:  lerp(0.14, 0.92, c),
      middle: lerp(0.18, 0.98, c),
      ring:   lerp(0.22, 1.08, c),
      pinky:  lerp(0.26, 1.15, c),
    };
    if (mode === 'heart') {
      tAbd = 0.95; tMcp = 0.35; tIp = 0.45;
      fFlex.index = 0.68; fFlex.middle = 1.25; fFlex.ring = 1.32; fFlex.pinky = 1.38;
    } else if (mode === 'open') {
      tAbd = 0.20; tMcp = 0.06; tIp = 0.05;
      fFlex.index = 0.06; fFlex.middle = 0.06; fFlex.ring = 0.08; fFlex.pinky = 0.10;
    }
    thumbCmc.rotation.set(
      sgn * (0.65 + 0.45 * tAbd),
      sgn * (0.58 - 0.18 * tAbd),
      -0.32 - 0.38 * tAbd
    );
    thumbMcp.rotation.z = -tMcp * 0.85;
    thumbIp.rotation.z = -tIp * 0.92;

    for (const fname of ['index', 'middle', 'ring', 'pinky']) {
      const fg = fingers[fname];
      const fl = fFlex[fname];
      fg.mcpGroup.rotation.y = fg.baseSplay;
      fg.mcpGroup.rotation.z = -fl * 0.85;
      fg.pipGroup.rotation.z = -fl * 1.02;
      fg.dipGroup.rotation.z = -fl * 0.78;
    }
  }

  setCloseAmount(0.25);
  return { root, gripAnchor, setCloseAmount };
}

/**
 * Universal MJCF Binary Asset Bundle Loader (`UR1A`, `RBY1`, `CLOD`).
 */
async function fetchMjcfBundle(THREE, url, expectedMagic, customMaterialsFn) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const buffer = await resp.arrayBuffer();
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== expectedMagic) {
    throw new Error(`Invalid bundle magic for ${url}: expected ${expectedMagic}, got ${magic}`);
  }
  const view = new DataView(buffer);
  const jsonLen = view.getUint32(4, true);
  const jsonText = new TextDecoder().decode(new Uint8Array(buffer, 8, jsonLen));
  const meta = JSON.parse(jsonText);
  const binStart = 8 + jsonLen;

  const { materials, defaultMat, flashMat } = customMaterialsFn(THREE, meta.materials || {});

  const geometries = {};
  for (const [meshName, info] of Object.entries(meta.meshes)) {
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(buffer, binStart + info.posByteOffset, info.vertexCount * 3);
    const normArr = new Float32Array(buffer, binStart + info.normByteOffset, info.vertexCount * 3);
    const idxArr = new Uint32Array(buffer, binStart + info.idxByteOffset, info.indexCount);
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));
    geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    geometries[meshName] = geo;
  }

  const bodyNodes = {};
  const jointNodes = {};

  const buildBodyNode = (nodeData) => {
    const group = new THREE.Group();
    group.name = nodeData.name;
    const restPos = new THREE.Vector3(...nodeData.pos);
    const restQuat = new THREE.Quaternion(
      nodeData.quat[1],
      nodeData.quat[2],
      nodeData.quat[3],
      nodeData.quat[0]
    );
    group.position.copy(restPos);
    group.quaternion.copy(restQuat);
    bodyNodes[nodeData.name] = group;

    const validJoints = (nodeData.joints || []).filter((j) => j.type !== 'free');
    if (validJoints.length > 0) {
      const j = validJoints[0];
      jointNodes[j.name] = {
        name: j.name,
        type: j.type,
        axis: new THREE.Vector3(...j.axis).normalize(),
        range: j.range || [-3.14159, 3.14159],
        group,
        restPos,
        restQuat,
      };
    }

    for (const g of nodeData.geoms || []) {
      const geo = geometries[g.mesh];
      if (!geo) continue;
      const mat = materials[g.material] || defaultMat;
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...g.pos);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      group.add(mesh);
    }

    for (const child of nodeData.children || []) {
      group.add(buildBodyNode(child));
    }
    return group;
  };

  const robotRoot = buildBodyNode(meta.tree);
  return { meta, robotRoot, bodyNodes, jointNodes, flashMat };
}

function applyJointMapToHierarchy(THREE, jointNodes, jointValues) {
  const tmpQuat = new THREE.Quaternion();
  for (const [jName, jInfo] of Object.entries(jointNodes)) {
    const val = jointValues[jName] ?? 0;
    if (jInfo.type === 'slide') {
      const disp = clamp(val, jInfo.range[0], jInfo.range[1]);
      jInfo.group.position.copy(jInfo.restPos).addScaledVector(jInfo.axis, disp);
      jInfo.group.quaternion.copy(jInfo.restQuat);
    } else {
      const angle = clamp(val, jInfo.range[0], jInfo.range[1]);
      tmpQuat.setFromAxisAngle(jInfo.axis, angle);
      jInfo.group.position.copy(jInfo.restPos);
      jInfo.group.quaternion.copy(jInfo.restQuat).multiply(tmpQuat);
    }
  }
}

// ============================================================================
// 1. UNITREE R1-EDU (宇树 R1 · 26-DOF Humanoid + Dual 15-DOF 5-Finger Hands)
// ============================================================================
function buildUnitreeR1Rig(THREE) {
  const group = new THREE.Group();
  const mujocoRoot = new THREE.Group();
  mujocoRoot.rotation.x = -Math.PI / 2; // MuJoCo Z-up -> Three.js Y-up
  group.add(mujocoRoot);

  const poseGroup = new THREE.Group();
  mujocoRoot.add(poseGroup);

  const gripPoint = new THREE.Group();
  group.add(gripPoint);

  const cone = makeVisionCone(THREE);
  group.add(cone);

  const flashMat = new THREE.MeshStandardMaterial({
    color: 0xe8eef6,
    roughness: 0.24,
    metalness: 0.16,
  });

  const STAND_H = 0.732;
  const SIT_H = 0.555;

  const READY_JOINTS = {
    left_hip_pitch_joint: -0.12,
    left_hip_roll_joint: 0.0,
    left_hip_yaw_joint: 0.0,
    left_knee_joint: 0.32,
    left_ankle_pitch_joint: -0.20,
    left_ankle_roll_joint: 0.0,
    right_hip_pitch_joint: -0.12,
    right_hip_roll_joint: 0.0,
    right_hip_yaw_joint: 0.0,
    right_knee_joint: 0.32,
    right_ankle_pitch_joint: -0.20,
    right_ankle_roll_joint: 0.0,
    waist_roll_joint: 0.0,
    waist_yaw_joint: 0.0,
    left_shoulder_pitch_joint: 0.10,
    left_shoulder_roll_joint: 0.22,
    left_shoulder_yaw_joint: -0.12,
    left_elbow_joint: 0.82,
    left_wrist_roll_joint: 0.0,
    right_shoulder_pitch_joint: 0.10,
    right_shoulder_roll_joint: -0.22,
    right_shoulder_yaw_joint: 0.12,
    right_elbow_joint: 0.82,
    right_wrist_roll_joint: 0.0,
    head_pitch_joint: 0.06,
    head_yaw_joint: 0.0,
  };

  const SIT_JOINTS = {
    ...READY_JOINTS,
    left_hip_pitch_joint: -0.92,
    left_knee_joint: 1.72,
    left_ankle_pitch_joint: -0.80,
    right_hip_pitch_joint: -0.92,
    right_knee_joint: 1.72,
    right_ankle_pitch_joint: -0.80,
    left_shoulder_pitch_joint: -0.18,
    left_elbow_joint: 0.15,
    right_shoulder_pitch_joint: -0.18,
    right_elbow_joint: 0.15,
  };

  const STOW = {
    waist_yaw_joint: 0.0,
    left_shoulder_pitch_joint: 0.10,
    left_shoulder_roll_joint: 0.22,
    left_shoulder_yaw_joint: -0.12,
    left_elbow_joint: 0.82,
    left_wrist_roll_joint: 0.0,
    right_shoulder_pitch_joint: 0.10,
    right_shoulder_roll_joint: -0.22,
    right_shoulder_yaw_joint: 0.12,
    right_elbow_joint: 0.82,
    right_wrist_roll_joint: 0.0,
  };

  const defaults = {
    h: STAND_H,
    pitch: 0,
    roll: 0,
    hy: 0,
    hp: 0.06,
    grip: 0.25,
    handMode: 0, // 0=normal, 1=open, 2=heart
    ...READY_JOINTS,
  };

  let loaded = false;
  let loadingPromise = null;
  let bodyNodes = {};
  let jointNodes = {};
  let robotRoot = null;
  let hands = { left: null, right: null };
  let activeArm = 'right';
  let latestCur = { ...defaults };

  function ensureLoaded() {
    if (loadingPromise) return loadingPromise;
    loadingPromise = fetchMjcfBundle(
      THREE,
      '/static/unitree_r1_assets.bin?v=1',
      'UR1A',
      (T, rawMats) => {
        const materials = {};
        for (const [k, rgba] of Object.entries(rawMats)) {
          const avg = (rgba[0] + rgba[1] + rgba[2]) / 3;
          if (avg > 0.75) {
            materials[k] = flashMat;
          } else {
            materials[k] = new T.MeshStandardMaterial({
              color: new T.Color(rgba[0], rgba[1], rgba[2]),
              roughness: avg < 0.35 ? 0.36 : 0.28,
              metalness: avg < 0.35 ? 0.45 : 0.22,
            });
          }
        }
        return { materials, defaultMat: flashMat, flashMat };
      }
    ).then((res) => {
      robotRoot = res.robotRoot;
      bodyNodes = res.bodyNodes;
      jointNodes = res.jointNodes;
      poseGroup.add(robotRoot);

      // Attach Dual 15-Joint 5-Finger Dexterous Hands (`Dex5`)
      if (bodyNodes.left_wrist_roll_link) {
        hands.left = makeUnitreeFiveFingerHand(THREE, 'left');
        bodyNodes.left_wrist_roll_link.add(hands.left.root);
      }
      if (bodyNodes.right_wrist_roll_link) {
        hands.right = makeUnitreeFiveFingerHand(THREE, 'right');
        bodyNodes.right_wrist_roll_link.add(hands.right.root);
        hands.right.gripAnchor.add(gripPoint);
        gripPoint.position.set(0, 0, 0);
      }

      // Attach glowing Unitree cyan visor strip + vision cone on head_yaw_link
      if (bodyNodes.head_yaw_link) {
        const visor = new THREE.Mesh(
          new THREE.BoxGeometry(0.014, 0.088, 0.018),
          new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
        );
        visor.position.set(0.068, 0, 0.012);
        bodyNodes.head_yaw_link.add(visor);
        cone.position.set(0.072, 0, 0.012);
        bodyNodes.head_yaw_link.add(cone);
      }

      loaded = true;
      update({ cur: latestCur, t: 0, dt: 0.016, k: 1, walkAmt: 0, gaitPhase: 0, gait: null });
      window.dispatchEvent(new CustomEvent('robot-twin:rig-loaded', { detail: { name: 'unitree_r1' } }));
    }).catch((err) => {
      console.error('[Real2Sim] Failed to load Unitree R1 bundle:', err);
    });
    return loadingPromise;
  }

  function getPalmPosMj(arm, jointState, pelvisZ = STAND_H, pelvisPitch = 0.08) {
    if (!loaded) {
      return new THREE.Vector3(0.32, arm === 'left' ? 0.18 : -0.18, 0.78);
    }
    applyJointMapToHierarchy(THREE, jointNodes, jointState);
    robotRoot.position.set(0, 0, pelvisZ);
    robotRoot.rotation.set(0, pelvisPitch, 0);
    mujocoRoot.updateMatrixWorld(true);
    const wristNode = bodyNodes[arm === 'left' ? 'left_wrist_roll_link' : 'right_wrist_roll_link'];
    if (!wristNode) return new THREE.Vector3(0.32, arm === 'left' ? 0.18 : -0.18, 0.78);
    const tipLocal = new THREE.Vector3(0.082, 0, -0.015);
    tipLocal.applyMatrix4(wristNode.matrixWorld);
    mujocoRoot.worldToLocal(tipLocal);
    return tipLocal;
  }

  function preparePick(tx, ty, tz) {
    activeArm = tz < -0.04 ? 'left' : 'right';
    const hand = hands[activeArm];
    if (hand && hand.gripAnchor) {
      hand.gripAnchor.add(gripPoint);
      gripPoint.position.set(0, 0, 0);
    }
  }

  function solveIkForTarget(tx, ty, tz) {
    // Convert Three.js world (tx, ty, tz) -> MuJoCo Z-up (x_mj = tx, y_mj = -tz, z_mj = ty)
    const targetMj = new THREE.Vector3(tx, -tz, ty);
    const arm = activeArm || (tz < -0.04 ? 'left' : 'right');
    const pfx = arm === 'left' ? 'left' : 'right';

    if (!loaded) {
      return {
        pitch: 0.08,
        [`${pfx}_shoulder_pitch_joint`]: -0.65,
        [`${pfx}_shoulder_roll_joint`]: arm === 'left' ? 0.28 : -0.28,
        [`${pfx}_shoulder_yaw_joint`]: 0.0,
        [`${pfx}_elbow_joint`]: -0.25,
      };
    }

    const workJoints = { ...READY_JOINTS, ...latestCur };
    const jNames = [
      'waist_yaw_joint',
      `${pfx}_shoulder_pitch_joint`,
      `${pfx}_shoulder_roll_joint`,
      `${pfx}_shoulder_yaw_joint`,
      `${pfx}_elbow_joint`,
    ];
    const eps = 1e-3;
    const lam = 0.06;
    const pelvisPitch = 0.09;

    for (let iter = 0; iter < 16; iter++) {
      const curPos = getPalmPosMj(arm, workJoints, STAND_H, pelvisPitch);
      const err = new THREE.Vector3().subVectors(targetMj, curPos);
      if (err.length() < 0.004) break;
      if (err.length() > 0.08) err.setLength(0.08);

      const J = [];
      for (const jn of jNames) {
        const orig = workJoints[jn] ?? 0;
        workJoints[jn] = orig + eps;
        const pPlus = getPalmPosMj(arm, workJoints, STAND_H, pelvisPitch);
        workJoints[jn] = orig;
        J.push([
          (pPlus.x - curPos.x) / eps,
          (pPlus.y - curPos.y) / eps,
          (pPlus.z - curPos.z) / eps,
        ]);
      }

      const A = [
        [lam * lam, 0, 0],
        [0, lam * lam, 0],
        [0, 0, lam * lam],
      ];
      for (let j = 0; j < jNames.length; j++) {
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) A[r][c] += J[j][r] * J[j][c];
        }
      }
      const y = solve3x3(A, [err.x, err.y, err.z]);
      for (let j = 0; j < jNames.length; j++) {
        const jn = jNames[j];
        const jInfo = jointNodes[jn];
        const dq = J[j][0] * y[0] + J[j][1] * y[1] + J[j][2] * y[2];
        let lo = jInfo ? jInfo.range[0] : -2.0;
        let hi = jInfo ? jInfo.range[1] : 2.0;
        if (jn === 'waist_yaw_joint') { lo = -0.55; hi = 0.55; }
        if (jn.endsWith('_elbow_joint')) { hi = Math.min(hi, 1.15); }
        workJoints[jn] = clamp((workJoints[jn] ?? 0) + dq, lo, hi);
      }
    }

    const out = { pitch: pelvisPitch, handMode: 0 };
    for (const jn of jNames) out[jn] = workJoints[jn];
    return out;
  }

  function readyPose() {
    return {
      pitch: 0,
      handMode: 0,
      waist_yaw_joint: 0.0,
      left_shoulder_pitch_joint: -0.22,
      left_shoulder_roll_joint: 0.28,
      left_shoulder_yaw_joint: -0.16,
      left_elbow_joint: 0.25,
      left_wrist_roll_joint: 0.0,
      right_shoulder_pitch_joint: -0.22,
      right_shoulder_roll_joint: -0.28,
      right_shoulder_yaw_joint: 0.16,
      right_elbow_joint: 0.25,
      right_wrist_roll_joint: 0.0,
    };
  }

  function carryPose() {
    const pfx = activeArm === 'left' ? 'left' : 'right';
    const sgn = activeArm === 'left' ? 1 : -1;
    return {
      ...readyPose(),
      [`${pfx}_shoulder_pitch_joint`]: -0.45,
      [`${pfx}_shoulder_roll_joint`]: sgn * 0.24,
      [`${pfx}_shoulder_yaw_joint`]: -sgn * 0.18,
      [`${pfx}_elbow_joint`]: -0.15,
    };
  }

  function update({ cur, t, walkAmt, gaitPhase, gait, talk }) {
    latestCur = cur;
    if (!loaded || !robotRoot) {
      ensureLoaded();
      return;
    }

    const stepL = Math.sin(gaitPhase) * walkAmt;
    const stepR = Math.sin(gaitPhase + Math.PI) * walkAmt;
    const liftL = Math.max(0, stepL) * walkAmt;
    const liftR = Math.max(0, stepR) * walkAmt;
    const breathe = Math.sin(t * 1.9) * 0.012 * (1 - walkAmt);

    const frameJoints = { ...cur };
    frameJoints.left_hip_pitch_joint = (cur.left_hip_pitch_joint ?? -0.12) - stepL * 0.36 - liftL * 0.22;
    frameJoints.left_knee_joint = (cur.left_knee_joint ?? 0.32) + liftL * 0.62;
    frameJoints.left_ankle_pitch_joint = (cur.left_ankle_pitch_joint ?? -0.20) - liftL * 0.32 + stepL * 0.12;

    frameJoints.right_hip_pitch_joint = (cur.right_hip_pitch_joint ?? -0.12) - stepR * 0.36 - liftR * 0.22;
    frameJoints.right_knee_joint = (cur.right_knee_joint ?? 0.32) + liftR * 0.62;
    frameJoints.right_ankle_pitch_joint = (cur.right_ankle_pitch_joint ?? -0.20) - liftR * 0.32 + stepR * 0.12;

    // Natural counter-swing of arms while walking
    frameJoints.left_shoulder_pitch_joint = (cur.left_shoulder_pitch_joint ?? 0.05) + stepR * 0.28 + breathe;
    frameJoints.right_shoulder_pitch_joint = (cur.right_shoulder_pitch_joint ?? 0.05) + stepL * 0.28 + breathe;

    frameJoints.head_yaw_joint = cur.hy ?? 0;
    frameJoints.head_pitch_joint = -(cur.hp ?? 0) * 0.65;

    // Co-speech head & phrase-level 3D gesture overlay (relaxed low baseline + asymmetric outward arcs)
    const tb = (talk && talk.blend > 0.001 && (walkAmt || 0) < 0.05) ? talk.blend : 0;
    const te = tb ? talk.env : 0;
    const ta = tb * ((talk && talk.armScale) ?? 1);
    let rOpen = 0.12, lOpen = 0.12;
    if (tb > 0 && talk.R && talk.L) {
      const gR = talk.R, gL = talk.L;
      rOpen = gR.open;
      lOpen = gL.open;
      // Unitree R1 MJCF convention:
      // +head_pitch_joint nods DOWN (-talk.headPitch); waist_yaw_joint Z-axis, waist_roll_joint X-axis;
      // right_shoulder_yaw_joint > 0 and left_shoulder_yaw_joint < 0 bring forearms INWARD to Center-Center!
      frameJoints.head_yaw_joint += talk.headYaw;
      frameJoints.head_pitch_joint += -talk.headPitch;
      frameJoints.waist_yaw_joint = (frameJoints.waist_yaw_joint ?? 0) + talk.torsoYaw;
      frameJoints.waist_roll_joint = (frameJoints.waist_roll_joint ?? 0) + talk.torsoRoll;
      if (ta > 0) {
        const lSp = 0.06 - 0.92 * gL.lift;
        const lSr = 0.20 + 0.55 * gL.abd;
        const lSy = -0.08 - 0.56 * gL.inward;
        const lEl = 0.78 - 1.38 * gL.flex;
        const lWr = -0.48 * gL.supinate;

        const rSp = 0.06 - 0.92 * gR.lift;
        const rSr = -0.20 - 0.55 * gR.abd;
        const rSy = 0.08 + 0.56 * gR.inward;
        const rEl = 0.78 - 1.38 * gR.flex;
        const rWr = 0.48 * gR.supinate;

        frameJoints.left_shoulder_pitch_joint = lerp(frameJoints.left_shoulder_pitch_joint, lSp, ta);
        frameJoints.left_shoulder_roll_joint = lerp(frameJoints.left_shoulder_roll_joint ?? 0.22, lSr, ta);
        frameJoints.left_shoulder_yaw_joint = lerp(frameJoints.left_shoulder_yaw_joint ?? -0.10, lSy, ta);
        frameJoints.left_elbow_joint = lerp(frameJoints.left_elbow_joint ?? 0.82, lEl, ta);
        frameJoints.left_wrist_roll_joint = lerp(frameJoints.left_wrist_roll_joint ?? 0, lWr, ta);
        frameJoints.right_shoulder_pitch_joint = lerp(frameJoints.right_shoulder_pitch_joint, rSp, ta);
        frameJoints.right_shoulder_roll_joint = lerp(frameJoints.right_shoulder_roll_joint ?? -0.22, rSr, ta);
        frameJoints.right_shoulder_yaw_joint = lerp(frameJoints.right_shoulder_yaw_joint ?? 0.10, rSy, ta);
        frameJoints.right_elbow_joint = lerp(frameJoints.right_elbow_joint ?? 0.82, rEl, ta);
        frameJoints.right_wrist_roll_joint = lerp(frameJoints.right_wrist_roll_joint ?? 0, rWr, ta);
      }
    }

    applyJointMapToHierarchy(THREE, jointNodes, frameJoints);

    const bounce = Math.abs(Math.sin(gaitPhase)) * 0.022 * walkAmt;
    const sway = Math.cos(gaitPhase) * 0.018 * walkAmt;
    robotRoot.position.set(0, sway, (cur.h ?? STAND_H) + bounce);
    robotRoot.rotation.set(
      (cur.roll ?? 0) + Math.cos(gaitPhase) * 0.03 * walkAmt,
      (cur.pitch ?? 0) + (tb > 0 ? talk.torsoLean * 0.65 : 0) + (gait ? clamp(gait.vx * 0.08, -0.08, 0.10) : 0),
      0
    );

    // Update Dual 5-Finger Dexterous Hands (`cur.grip`: 1 = open, 0.25 = closed grasp)
    const rawClose = clamp(1 - (cur.grip ?? 0.25), 0, 1);
    const closeL = ta > 0 ? lerp(rawClose, clamp(0.62 - 0.50 * lOpen, 0.10, 0.65), ta) : rawClose;
    const closeR = ta > 0 ? lerp(rawClose, clamp(0.62 - 0.50 * rOpen, 0.10, 0.65), ta) : rawClose;
    const modeL = cur.handMode === 2 ? 'heart' : (cur.handMode === 1 || (ta > 0.25 && lOpen > 0.42)) ? 'open' : 'normal';
    const modeR = cur.handMode === 2 ? 'heart' : (cur.handMode === 1 || (ta > 0.25 && rOpen > 0.42)) ? 'open' : 'normal';
    if (hands.left) hands.left.setCloseAmount(closeL, modeL);
    if (hands.right) hands.right.setCloseAmount(closeR, modeR);
  }

  return {
    name: 'unitree_r1',
    icon: '🦾',
    label: 'Unitree R1-EDU (宇树 R1)',
    spec: '26-DOF Bipedal + Dual 15-DOF 5-Finger Dex5 Hands',
    group,
    gripPoint,
    cone,
    flashMat,
    STAND_H,
    SIT_H,
    STOW,
    approachH: 0.15,
    graspH: 0.045,
    ensureLoaded,
    preparePick,
    ik: solveIkForTarget,
    ready: readyPose,
    carry: carryPose,
    table: { x: 0.39, top: 0.64, half: { x: 0.18, z: 0.32 } },
    homeCam: new THREE.Vector3(1.62, 1.22, 1.56),
    homeTarget: new THREE.Vector3(0.14, 0.56, 0),
    defaults,
    standEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, pitch: 0, roll: 0, handMode: 0 });
    },
    isUp: (tgt) => (tgt.h ?? STAND_H) > STAND_H + 0.02,
    downEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, pitch: 0, roll: 0, handMode: 0 });
    },
    sitEnter: (tgt) => {
      Object.assign(tgt, SIT_JOINTS, { h: SIT_H, pitch: 0.12, roll: 0, handMode: 0 });
    },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.hy = clamp((tgt.hy ?? 0) + ang, -0.85, 0.85);
      else if (dir === 'right') tgt.hy = clamp((tgt.hy ?? 0) - ang, -0.85, 0.85);
      else if (dir === 'up') tgt.hp = clamp((tgt.hp ?? 0) + ang, -0.5, 0.5);
      else if (dir === 'down') tgt.hp = clamp((tgt.hp ?? 0) - ang, -0.5, 0.5);
      else { tgt.hy = 0; tgt.hp = 0.06; }
    },
    detectEnter: (tgt, setArm) => {
      tgt.hp = -0.36;
      tgt.hy = 0;
      setArm(readyPose());
    },
    gestures: {
      high_wave: (tgt, setArm, ctx) => [
        [0.45, () => {
          tgt.handMode = 1;
          setArm({
            right_shoulder_pitch_joint: -1.35,
            right_shoulder_roll_joint: -0.78,
            right_shoulder_yaw_joint: 0.35,
            right_elbow_joint: -0.55,
            waist_yaw_joint: -0.12,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          tgt.right_shoulder_roll_joint = -0.78 + (i % 2 ? 0.26 : -0.26);
          tgt.right_wrist_roll_joint = i % 2 ? 0.45 : -0.45;
          tgt.hy = (i % 2 ? 1 : -1) * 0.12;
        }]),
        [0.65, () => { tgt.hy = 0; tgt.handMode = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      shake_hand: (tgt, setArm, ctx) => [
        [0.5, () => {
          tgt.handMode = 1;
          tgt.pitch = 0.08;
          setArm({
            right_shoulder_pitch_joint: -0.78,
            right_shoulder_roll_joint: -0.15,
            right_shoulder_yaw_joint: 0.12,
            right_elbow_joint: -0.32,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          tgt.right_shoulder_pitch_joint = -0.78 + (i % 2 ? -0.14 : 0.14);
          tgt.right_elbow_joint = -0.32 + (i % 2 ? -0.12 : 0.12);
        }]),
        [0.65, () => { tgt.pitch = 0; tgt.handMode = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      heart: (tgt, setArm, ctx) => [
        [0.55, () => {
          tgt.handMode = 2;
          setArm({
            left_shoulder_pitch_joint: -1.15,
            left_shoulder_roll_joint: 0.45,
            left_shoulder_yaw_joint: -0.62,
            left_elbow_joint: -0.68,
            right_shoulder_pitch_joint: -1.15,
            right_shoulder_roll_joint: -0.45,
            right_shoulder_yaw_joint: 0.62,
            right_elbow_joint: -0.68,
          });
        }],
        [1.2, () => { tgt.roll = 0.05; }],
        [0.65, () => { tgt.roll = 0; tgt.handMode = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      clap: (tgt, setArm, ctx) => [
        [0.4, () => {
          tgt.handMode = 1;
          setArm({
            left_shoulder_pitch_joint: -0.78,
            left_shoulder_roll_joint: 0.25,
            left_shoulder_yaw_joint: -0.55,
            left_elbow_joint: -0.42,
            right_shoulder_pitch_joint: -0.78,
            right_shoulder_roll_joint: -0.25,
            right_shoulder_yaw_joint: 0.55,
            right_elbow_joint: -0.42,
          });
        }],
        ...[0, 1, 2, 3, 4, 5].map((i) => [0.22, () => {
          const open = i % 2 === 0 ? 0.36 : 0.08;
          tgt.left_shoulder_roll_joint = open;
          tgt.right_shoulder_roll_joint = -open;
        }]),
        [0.6, () => { tgt.handMode = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
    },
    update,
  };
}

// ============================================================================
// 2. RAINBOW ROBOTICS RB-Y1 (28-DOF Bimanual Mobile Manipulator)
// ============================================================================
function buildRby1Rig(THREE) {
  const group = new THREE.Group();
  const mujocoRoot = new THREE.Group();
  mujocoRoot.rotation.x = -Math.PI / 2;
  group.add(mujocoRoot);

  const poseGroup = new THREE.Group();
  mujocoRoot.add(poseGroup);

  const gripPoint = new THREE.Group();
  group.add(gripPoint);

  const cone = makeVisionCone(THREE);
  group.add(cone);

  const flashMat = new THREE.MeshStandardMaterial({
    color: 0xf1f5f9,
    roughness: 0.28,
    metalness: 0.18,
  });

  const STAND_H = 0.0;
  const SIT_H = -0.28;

  const READY_JOINTS = {
    right_wheel: 0,
    left_wheel: 0,
    torso_0: 0,
    torso_1: 0,
    torso_2: 0,
    torso_3: 0,
    torso_4: 0,
    torso_5: 0,
    head_0: 0,
    head_1: 0.12,
    right_arm_0: -0.24,
    right_arm_1: -0.15,
    right_arm_2: 0.0,
    right_arm_3: -0.58,
    right_arm_4: 0.0,
    right_arm_5: -0.22,
    right_arm_6: 0.0,
    left_arm_0: -0.24,
    left_arm_1: 0.15,
    left_arm_2: 0.0,
    left_arm_3: -0.58,
    left_arm_4: 0.0,
    left_arm_5: -0.22,
    left_arm_6: 0.0,
    gripper_finger_r1: -0.025,
    gripper_finger_r2: -0.025,
    gripper_finger_l1: -0.025,
    gripper_finger_l2: -0.025,
  };

  const SIT_JOINTS = {
    ...READY_JOINTS,
    torso_1: 1.05,
    torso_2: -2.10,
    torso_3: 1.05,
    head_1: 0.08,
    right_arm_0: -0.45,
    right_arm_1: -0.18,
    right_arm_2: 0.15,
    right_arm_3: -0.95,
    left_arm_0: -0.45,
    left_arm_1: 0.18,
    left_arm_2: -0.15,
    left_arm_3: -0.95,
  };

  const STOW = {
    torso_3: 0,
    right_arm_0: -0.24,
    right_arm_1: -0.15,
    right_arm_2: 0.0,
    right_arm_3: -0.58,
    right_arm_4: 0.0,
    right_arm_5: -0.22,
    right_arm_6: 0.0,
    left_arm_0: -0.24,
    left_arm_1: 0.15,
    left_arm_2: 0.0,
    left_arm_3: -0.58,
    left_arm_4: 0.0,
    left_arm_5: -0.22,
    left_arm_6: 0.0,
  };

  const defaults = {
    h: STAND_H,
    pitch: 0,
    roll: 0,
    hy: 0,
    hp: -0.12,
    grip: 0.25,
    ...READY_JOINTS,
  };

  let loaded = false;
  let loadingPromise = null;
  let bodyNodes = {};
  let jointNodes = {};
  let robotRoot = null;
  let activeArm = 'right';
  let wheelAngleL = 0;
  let wheelAngleR = 0;
  let latestCur = { ...defaults };

  function ensureLoaded() {
    if (loadingPromise) return loadingPromise;
    loadingPromise = fetchMjcfBundle(
      THREE,
      '/static/rby1_assets.bin?v=3',
      'RBY1',
      (T, rawMats) => {
        const materials = {};
        for (const [name, rgba] of Object.entries(rawMats)) {
          const isDark = rgba[0] < 0.2 && rgba[1] < 0.2 && rgba[2] < 0.2;
          const isWhite = rgba[0] > 0.85 && rgba[1] > 0.85 && rgba[2] > 0.85;
          materials[name] = isWhite
            ? flashMat
            : new T.MeshStandardMaterial({
                color: isDark
                  ? new T.Color(0x1e293b)
                  : new T.Color(rgba[0], rgba[1], rgba[2]),
                roughness: isDark ? 0.48 : 0.35,
                metalness: isDark ? 0.55 : 0.40,
              });
        }
        return { materials, defaultMat: flashMat, flashMat };
      }
    ).then((res) => {
      robotRoot = res.robotRoot;
      bodyNodes = res.bodyNodes;
      jointNodes = res.jointNodes;
      poseGroup.add(robotRoot);

      if (bodyNodes.link_head_2) {
        const visorGeo = new THREE.BoxGeometry(0.012, 0.088, 0.016);
        const visorMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        const visorMesh = new THREE.Mesh(visorGeo, visorMat);
        visorMesh.position.set(0.095, 0, 0.055);
        bodyNodes.link_head_2.add(visorMesh);
        cone.position.set(0.10, 0, 0.055);
        bodyNodes.link_head_2.add(cone);
      }

      if (bodyNodes.EE_BODY_R) {
        bodyNodes.EE_BODY_R.add(gripPoint);
        gripPoint.position.set(0, 0, -0.074);
      }

      loaded = true;
      update({ cur: latestCur, t: 0, dt: 0.016, k: 1, walkAmt: 0, gaitPhase: 0, gait: null });
      window.dispatchEvent(new CustomEvent('robot-twin:rig-loaded', { detail: { name: 'rby1' } }));
    }).catch((err) => {
      console.error('[Real2Sim] Failed to load RB-Y1 bundle:', err);
    });
    return loadingPromise;
  }

  function getEePosMj(arm, jointState) {
    if (!loaded) return new THREE.Vector3(0.45, arm === 'left' ? 0.22 : -0.22, 0.85);
    applyJointMapToHierarchy(THREE, jointNodes, jointState);
    mujocoRoot.updateMatrixWorld(true);
    const eeNode = bodyNodes[arm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (!eeNode) return new THREE.Vector3(0.45, arm === 'left' ? 0.22 : -0.22, 0.85);
    const tipLocal = new THREE.Vector3(0, 0, -0.074);
    tipLocal.applyMatrix4(eeNode.matrixWorld);
    mujocoRoot.worldToLocal(tipLocal);
    return tipLocal;
  }

  function preparePick(tx, ty, tz) {
    activeArm = tz < -0.04 ? 'left' : 'right';
    const eeNode = bodyNodes[activeArm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (eeNode) {
      eeNode.add(gripPoint);
      gripPoint.position.set(0, 0, -0.074);
    }
  }

  function solveIkForTarget(tx, ty, tz) {
    const targetMj = new THREE.Vector3(tx, -tz, ty);
    const arm = activeArm || (tz < -0.04 ? 'left' : 'right');
    const prefix = arm === 'left' ? 'left_arm_' : 'right_arm_';
    const jNames = ['torso_4', ...[0, 1, 2, 3, 4, 5].map((i) => `${prefix}${i}`)];

    if (!loaded) {
      return {
        torso_3: 0.16,
        torso_4: 0.0,
        [`${prefix}0`]: -0.68,
        [`${prefix}1`]: arm === 'left' ? 0.18 : -0.18,
        [`${prefix}3`]: -0.82,
      };
    }

    const workJoints = { ...READY_JOINTS, ...latestCur, torso_3: 0.24 };
    const eps = 1e-3;
    const lam = 0.07;

    for (let iter = 0; iter < 16; iter++) {
      const curPos = getEePosMj(arm, workJoints);
      const err = new THREE.Vector3().subVectors(targetMj, curPos);
      if (err.length() < 0.004) break;
      if (err.length() > 0.08) err.setLength(0.08);

      const J = [];
      for (const jn of jNames) {
        const orig = workJoints[jn] ?? 0;
        workJoints[jn] = orig + eps;
        const pPlus = getEePosMj(arm, workJoints);
        workJoints[jn] = orig;
        J.push([
          (pPlus.x - curPos.x) / eps,
          (pPlus.y - curPos.y) / eps,
          (pPlus.z - curPos.z) / eps,
        ]);
      }

      const A = [
        [lam * lam, 0, 0],
        [0, lam * lam, 0],
        [0, 0, lam * lam],
      ];
      for (let j = 0; j < jNames.length; j++) {
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) A[r][c] += J[j][r] * J[j][c];
        }
      }
      const y = solve3x3(A, [err.x, err.y, err.z]);
      for (let j = 0; j < jNames.length; j++) {
        const jn = jNames[j];
        const jInfo = jointNodes[jn];
        const dq = J[j][0] * y[0] + J[j][1] * y[1] + J[j][2] * y[2];
        let lo = jInfo ? jInfo.range[0] : -2.0;
        let hi = jInfo ? jInfo.range[1] : 2.0;
        if (jn === 'torso_4') { lo = -0.55; hi = 0.55; }
        workJoints[jn] = clamp((workJoints[jn] ?? 0) + dq, lo, hi);
      }
    }

    const out = { torso_3: 0.24 };
    for (const jn of jNames) out[jn] = workJoints[jn];
    return out;
  }

  function readyPose() {
    return {
      torso_3: 0.0,
      torso_4: 0.0,
      right_arm_0: -0.36,
      right_arm_1: -0.18,
      right_arm_2: 0.0,
      right_arm_3: -0.78,
      right_arm_4: 0.0,
      right_arm_5: -0.25,
      right_arm_6: 0.0,
      left_arm_0: -0.36,
      left_arm_1: 0.18,
      left_arm_2: 0.0,
      left_arm_3: -0.78,
      left_arm_4: 0.0,
      left_arm_5: -0.25,
      left_arm_6: 0.0,
    };
  }

  function carryPose() {
    const prefix = activeArm === 'left' ? 'left_arm_' : 'right_arm_';
    const sgn = activeArm === 'left' ? 1 : -1;
    return {
      ...readyPose(),
      [`${prefix}0`]: -0.62,
      [`${prefix}1`]: sgn * 0.22,
      [`${prefix}3`]: -1.05,
      [`${prefix}5`]: -0.18,
    };
  }

  function update({ cur, t, dt, gait, talk }) {
    latestCur = cur;
    if (!loaded || !robotRoot) {
      ensureLoaded();
      return;
    }

    if (gait) {
      const rWheel = 0.1;
      wheelAngleR += ((gait.vx + 0.265 * gait.vrot) / rWheel) * dt;
      wheelAngleL += ((gait.vx - 0.265 * gait.vrot) / rWheel) * dt;
    }

    const breathe = Math.sin(t * 1.8) * 0.012;
    const tb = (talk && talk.blend > 0.001 && !gait) ? talk.blend : 0;
    const ta = tb * ((talk && talk.armScale) ?? 1);
    const gripR = (ta > 0 && talk.R) ? lerp(cur.grip ?? 0.25, 0.20 + 0.68 * talk.R.open, ta) : (cur.grip ?? 0.25);
    const gripL = (ta > 0 && talk.L) ? lerp(cur.grip ?? 0.25, 0.20 + 0.68 * talk.L.open, ta) : (cur.grip ?? 0.25);
    const fingerValR = lerp(-0.004, -0.045, clamp(gripR, 0, 1));
    const fingerValL = lerp(-0.004, -0.045, clamp(gripL, 0, 1));
    const frameJoints = {
      ...cur,
      right_wheel: wheelAngleR,
      left_wheel: wheelAngleL,
      torso_3: (cur.torso_3 ?? 0) + breathe + (gait ? clamp(gait.vx * 0.12, -0.1, 0.14) : 0),
      head_0: cur.hy ?? 0,
      head_1: -(cur.hp ?? -0.12),
      gripper_finger_r1: fingerValR,
      gripper_finger_r2: fingerValR,
      gripper_finger_l1: fingerValL,
      gripper_finger_l2: fingerValL,
    };

    if (tb > 0 && talk.R && talk.L) {
      const gR = talk.R, gL = talk.L;
      // RB-Y1 MJCF convention:
      // torso_3 = Y-lean, torso_4 = X-roll, torso_5 = Z-yaw;
      // right_arm_2 > 0 and left_arm_2 < 0 bring bent forearms INWARD into Center-Center!
      frameJoints.head_0 += talk.headYaw;
      frameJoints.head_1 += -talk.headPitch;
      frameJoints.torso_3 += talk.torsoLean;
      frameJoints.torso_4 = (frameJoints.torso_4 ?? 0) + talk.torsoRoll;
      frameJoints.torso_5 = (frameJoints.torso_5 ?? 0) + talk.torsoYaw;
      if (ta > 0) {
        const r0 = -0.16 - 0.74 * gR.lift;
        const r1 = -0.14 - 0.52 * gR.abd;
        const r2 =  0.62 * gR.inward;
        const r3 = -0.44 - 1.12 * gR.flex;
        const r4 =  0.56 * gR.supinate;
        const r5 = -0.16 + 0.28 * gR.wristPitch;

        const l0 = -0.16 - 0.74 * gL.lift;
        const l1 =  0.14 + 0.52 * gL.abd;
        const l2 = -0.62 * gL.inward;
        const l3 = -0.44 - 1.12 * gL.flex;
        const l4 = -0.56 * gL.supinate;
        const l5 = -0.16 + 0.28 * gL.wristPitch;

        frameJoints.right_arm_0 = lerp(frameJoints.right_arm_0 ?? -0.24, r0, ta);
        frameJoints.right_arm_1 = lerp(frameJoints.right_arm_1 ?? -0.15, r1, ta);
        frameJoints.right_arm_2 = lerp(frameJoints.right_arm_2 ?? 0, r2, ta);
        frameJoints.right_arm_3 = lerp(frameJoints.right_arm_3 ?? -0.58, r3, ta);
        frameJoints.right_arm_4 = lerp(frameJoints.right_arm_4 ?? 0, r4, ta);
        frameJoints.right_arm_5 = lerp(frameJoints.right_arm_5 ?? -0.22, r5, ta);
        frameJoints.left_arm_0 = lerp(frameJoints.left_arm_0 ?? -0.24, l0, ta);
        frameJoints.left_arm_1 = lerp(frameJoints.left_arm_1 ?? 0.15, l1, ta);
        frameJoints.left_arm_2 = lerp(frameJoints.left_arm_2 ?? 0, l2, ta);
        frameJoints.left_arm_3 = lerp(frameJoints.left_arm_3 ?? -0.58, l3, ta);
        frameJoints.left_arm_4 = lerp(frameJoints.left_arm_4 ?? 0, l4, ta);
        frameJoints.left_arm_5 = lerp(frameJoints.left_arm_5 ?? -0.22, l5, ta);
      }
    }

    applyJointMapToHierarchy(THREE, jointNodes, frameJoints);
  }

  return {
    name: 'rby1',
    icon: '🤖',
    label: 'Rainbow Robotics RB-Y1',
    spec: '28-DOF Bimanual Mobile Manipulator (MuJoCo Menagerie)',
    group,
    gripPoint,
    cone,
    flashMat,
    STAND_H,
    SIT_H,
    STOW,
    ensureLoaded,
    preparePick,
    ik: solveIkForTarget,
    ready: readyPose,
    carry: carryPose,
    approachH: 0.16,
    graspH: 0.045,
    table: { x: 0.56, top: 0.68, half: { x: 0.22, z: 0.38 } },
    homeCam: new THREE.Vector3(1.94, 1.36, 1.76),
    homeTarget: new THREE.Vector3(0.16, 0.66, 0),
    defaults,
    standEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, hy: 0, hp: -0.12 });
    },
    isUp: (tgt) => (tgt.h ?? STAND_H) > STAND_H + 0.02,
    downEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, hy: 0, hp: -0.12 });
    },
    sitEnter: (tgt) => {
      Object.assign(tgt, SIT_JOINTS, { h: SIT_H, hy: 0, hp: -0.08 });
    },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.hy = clamp((tgt.hy ?? 0) + ang, -1.1, 1.1);
      else if (dir === 'right') tgt.hy = clamp((tgt.hy ?? 0) - ang, -1.1, 1.1);
      else if (dir === 'up') tgt.hp = clamp((tgt.hp ?? -0.12) + ang, -0.7, 0.35);
      else if (dir === 'down') tgt.hp = clamp((tgt.hp ?? -0.12) - ang, -0.7, 0.35);
      else { tgt.hy = 0; tgt.hp = -0.12; }
    },
    detectEnter: (tgt, setArm) => {
      tgt.hp = -0.36;
      tgt.hy = 0;
      setArm(readyPose());
    },
    gestures: {
      high_wave: (tgt, setArm, ctx) => [
        [0.45, () => setArm({
          right_arm_0: -0.55,
          right_arm_1: -0.92,
          right_arm_2: -0.95,
          right_arm_3: -1.75,
          right_arm_5: 0.15,
        })],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          tgt.right_arm_2 = -0.95 + (i % 2 ? 0.36 : -0.36);
          tgt.right_arm_5 = i % 2 ? 0.25 : -0.25;
          tgt.hy = (i % 2 ? 1 : -1) * 0.08;
        }]),
        [0.65, () => { tgt.hy = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      shake_hand: (tgt, setArm, ctx) => [
        [0.5, () => setArm({
          torso_3: 0.14,
          right_arm_0: -0.75,
          right_arm_1: -0.08,
          right_arm_3: -1.05,
        })],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          const pump = i % 2 ? 0.16 : -0.16;
          tgt.right_arm_0 = -0.75 + pump;
          tgt.right_arm_3 = -1.05 - pump;
        }]),
        [0.65, () => { tgt.torso_3 = 0; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      heart: (tgt, setArm, ctx) => [
        [0.55, () => setArm({
          right_arm_0: -2.15,
          right_arm_1: -0.68,
          right_arm_2: 0.65,
          right_arm_3: -1.65,
          left_arm_0: -2.15,
          left_arm_1: 0.68,
          left_arm_2: -0.65,
          left_arm_3: -1.65,
        })],
        [1.2, () => {}],
        [0.65, () => { ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      clap: (tgt, setArm, ctx) => [
        [0.4, () => setArm({
          right_arm_0: -0.88,
          right_arm_1: -0.18,
          right_arm_2: 0.65,
          right_arm_3: -1.35,
          left_arm_0: -0.88,
          left_arm_1: 0.18,
          left_arm_2: -0.65,
          left_arm_3: -1.35,
        })],
        ...[0, 1, 2, 3, 4, 5].map((i) => [0.22, () => {
          const ab = i % 2 === 0 ? 0.28 : 0.10;
          tgt.right_arm_1 = -ab;
          tgt.left_arm_1 = ab;
        }]),
        [0.6, () => { ctx.restoreArm(); ctx.restorePosture(); }],
      ],
    },
    update,
  };
}

// ============================================================================
// 3. LG CLOiD (36-DOF CES 2026 Home Humanoid + OLED Visor + Dual 5F Hands)
// ============================================================================
function buildLgCloidRig(THREE) {
  const group = new THREE.Group();
  const mujocoRoot = new THREE.Group();
  mujocoRoot.rotation.x = -Math.PI / 2;
  group.add(mujocoRoot);

  const poseGroup = new THREE.Group();
  mujocoRoot.add(poseGroup);

  const gripPoint = new THREE.Group();
  group.add(gripPoint);

  const cone = makeVisionCone(THREE);
  group.add(cone);

  const flashMat = new THREE.MeshPhysicalMaterial({
    color: 0xf4f6fa,
    roughness: 0.24,
    metalness: 0.08,
    clearcoat: 0.45,
    clearcoatRoughness: 0.22,
  });

  // 60 FPS OLED Facial Expression CanvasTexture (`0 0` hollow pill eyes + bottom-right green indicator)
  const faceCanvas = document.createElement('canvas');
  faceCanvas.width = 512;
  faceCanvas.height = 256;
  const faceCtx = faceCanvas.getContext('2d');
  const faceTexture = new THREE.CanvasTexture(faceCanvas);
  faceTexture.colorSpace = THREE.SRGBColorSpace;

  function renderOledFace(nowSec, exprCode, headYaw, headPitch) {
    if (!faceCtx) return;
    const W = 512, H = 256;
    faceCtx.fillStyle = '#050811';
    faceCtx.fillRect(0, 0, W, H);

    // exprCode: 0=normal, 1=happy, 2=heart, 3=focused
    const blinkPhase = nowSec % 3.8;
    const isBlinking = (exprCode === 0 || exprCode === 3) && blinkPhase > 3.62 && blinkPhase < 3.76;
    const gazeX = clamp(-headYaw * 28, -28, 28);
    const gazeY = clamp(headPitch * 18, -16, 16);
    const leftEyeX = 176 + gazeX;
    const rightEyeX = 336 + gazeX;
    const eyeY = 132 + gazeY;

    faceCtx.save();
    faceCtx.shadowColor = '#67e8f9';
    faceCtx.shadowBlur = 18;
    faceCtx.strokeStyle = '#ffffff';
    faceCtx.fillStyle = '#ffffff';
    faceCtx.lineCap = 'round';
    faceCtx.lineJoin = 'round';

    const drawPill = (cx, cy, rx, ry, lw = 15) => {
      faceCtx.lineWidth = lw;
      faceCtx.beginPath();
      faceCtx.roundRect(cx - rx, cy - ry, rx * 2, ry * 2, rx);
      faceCtx.stroke();
    };
    const drawHappy = (cx, cy, rx) => {
      faceCtx.lineWidth = 16;
      faceCtx.beginPath();
      faceCtx.arc(cx, cy + 16, rx * 1.15, Math.PI * 1.12, Math.PI * 1.88, false);
      faceCtx.stroke();
    };
    const drawHeart = (cx, cy, scale = 1.12) => {
      faceCtx.save();
      faceCtx.translate(cx, cy);
      faceCtx.scale(scale, scale);
      faceCtx.shadowColor = '#fb7185';
      faceCtx.fillStyle = '#fda4af';
      faceCtx.beginPath();
      faceCtx.moveTo(0, 18);
      faceCtx.bezierCurveTo(-36, -6, -38, -42, -16, -42);
      faceCtx.bezierCurveTo(-4, -42, 0, -28, 0, -22);
      faceCtx.bezierCurveTo(0, -28, 4, -42, 16, -42);
      faceCtx.bezierCurveTo(38, -42, 36, -6, 0, 18);
      faceCtx.closePath();
      faceCtx.fill();
      faceCtx.restore();
    };

    if (isBlinking) {
      faceCtx.lineWidth = 14;
      faceCtx.beginPath();
      faceCtx.moveTo(leftEyeX - 34, eyeY);
      faceCtx.lineTo(leftEyeX + 34, eyeY);
      faceCtx.moveTo(rightEyeX - 34, eyeY);
      faceCtx.lineTo(rightEyeX + 34, eyeY);
      faceCtx.stroke();
    } else if (exprCode === 1) {
      drawHappy(leftEyeX, eyeY, 34);
      drawHappy(rightEyeX, eyeY, 34);
    } else if (exprCode === 2) {
      drawHeart(leftEyeX, eyeY + 8, 1.12);
      drawHeart(rightEyeX, eyeY + 8, 1.12);
    } else if (exprCode === 3) {
      drawPill(leftEyeX, eyeY, 32, 42, 16);
      drawPill(rightEyeX, eyeY, 32, 42, 16);
    } else {
      drawPill(leftEyeX, eyeY, 30, 58, 15);
      drawPill(rightEyeX, eyeY, 30, 58, 15);
    }

    // Signature LG CLOiD CES 2026 bottom-right green status dot
    faceCtx.shadowColor = '#10b981';
    faceCtx.shadowBlur = 12;
    faceCtx.fillStyle = '#34d399';
    faceCtx.beginPath();
    faceCtx.arc(rightEyeX + 32, eyeY + 48, 8.5, 0, Math.PI * 2);
    faceCtx.fill();
    faceCtx.restore();
    faceTexture.needsUpdate = true;
  }

  renderOledFace(0, 0, 0, 0.08);

  const STAND_H = 0.0;
  const SIT_H = -0.32;

  const READY_JOINTS = {
    right_wheel: 0,
    left_wheel: 0,
    torso_0: 0,
    torso_1: 0.14,
    torso_2: -0.28,
    torso_3: 0.14,
    torso_4: 0,
    torso_5: 0,
    head_0: 0,
    head_1: 0.08,
    right_arm_0: -0.20,
    right_arm_1: -0.14,
    right_arm_2: 0.0,
    right_arm_3: -0.52,
    right_arm_4: 0.0,
    right_arm_5: -0.18,
    right_arm_6: 0.0,
    left_arm_0: -0.20,
    left_arm_1: 0.14,
    left_arm_2: 0.0,
    left_arm_3: -0.52,
    left_arm_4: 0.0,
    left_arm_5: -0.18,
    left_arm_6: 0.0,
    r_thumb: 0.12,
    r_index: 0.15,
    r_middle: 0.18,
    r_ring: 0.20,
    r_pinky: 0.22,
    l_thumb: 0.12,
    l_index: 0.15,
    l_middle: 0.18,
    l_ring: 0.20,
    l_pinky: 0.22,
  };

  const SIT_JOINTS = {
    ...READY_JOINTS,
    torso_1: 1.02,
    torso_2: -2.04,
    torso_3: 1.02,
    head_1: 0.05,
    right_arm_0: -0.42,
    right_arm_1: -0.16,
    right_arm_2: 0.15,
    right_arm_3: -0.92,
    left_arm_0: -0.42,
    left_arm_1: 0.16,
    left_arm_2: -0.15,
    left_arm_3: -0.92,
  };

  const STOW = {
    torso_3: 0.14,
    right_arm_0: -0.20,
    right_arm_1: -0.14,
    right_arm_2: 0.0,
    right_arm_3: -0.52,
    right_arm_4: 0.0,
    right_arm_5: -0.18,
    right_arm_6: 0.0,
    left_arm_0: -0.20,
    left_arm_1: 0.14,
    left_arm_2: 0.0,
    left_arm_3: -0.52,
    left_arm_4: 0.0,
    left_arm_5: -0.18,
    left_arm_6: 0.0,
  };

  const defaults = {
    h: STAND_H,
    pitch: 0,
    roll: 0,
    hy: 0,
    hp: -0.08,
    grip: 0.25,
    exprCode: 0,
    ...READY_JOINTS,
  };

  let loaded = false;
  let loadingPromise = null;
  let bodyNodes = {};
  let jointNodes = {};
  let robotRoot = null;
  let activeArm = 'right';
  let wheelAngleL = 0;
  let wheelAngleR = 0;
  let latestCur = { ...defaults };

  function ensureLoaded() {
    if (loadingPromise) return loadingPromise;
    loadingPromise = fetchMjcfBundle(
      THREE,
      '/static/cloid_assets.bin?v=1',
      'CLOD',
      (T) => {
        const materials = {
          lg_pearl_white: flashMat,
          lg_visor_black: new T.MeshPhysicalMaterial({
            color: 0x070a12,
            roughness: 0.12,
            metalness: 0.25,
            clearcoat: 0.9,
            clearcoatRoughness: 0.08,
          }),
          lg_graphite_metal: new T.MeshStandardMaterial({
            color: 0x2d333f,
            roughness: 0.42,
            metalness: 0.62,
          }),
          lg_hand_matte_black: new T.MeshStandardMaterial({
            color: 0x181b22,
            roughness: 0.55,
            metalness: 0.22,
          }),
          lg_speaker_grille: new T.MeshStandardMaterial({
            color: 0xcbd5e1,
            roughness: 0.48,
            metalness: 0.32,
          }),
          lg_cyan_led: new T.MeshStandardMaterial({
            color: 0x38bdf8,
            emissive: 0x38bdf8,
            emissiveIntensity: 1.6,
            roughness: 0.2,
          }),
        };
        return { materials, defaultMat: flashMat, flashMat };
      }
    ).then((res) => {
      robotRoot = res.robotRoot;
      bodyNodes = res.bodyNodes;
      jointNodes = res.jointNodes;
      poseGroup.add(robotRoot);

      // Attach curved superellipsoid OLED Facial Expression Screen onto link_head_2
      if (bodyNodes.link_head_2) {
        const screenGeo = new THREE.PlaneGeometry(0.175, 0.088, 24, 12);
        const posAttr = screenGeo.attributes.position;
        for (let i = 0; i < posAttr.count; i++) {
          const u = posAttr.getX(i);
          const v = posAttr.getY(i);
          const curveZ =
            -0.009 * Math.pow(Math.abs(u) / 0.0875, 2.8) -
            0.0035 * Math.pow(Math.abs(v) / 0.044, 2.6);
          posAttr.setZ(i, curveZ);
        }
        screenGeo.computeVertexNormals();
        const screenMat = new THREE.MeshBasicMaterial({
          map: faceTexture,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });
        const faceScreenMesh = new THREE.Mesh(screenGeo, screenMat);
        faceScreenMesh.position.set(0.1125, 0.0, 0.058);
        faceScreenMesh.rotation.order = 'ZYX';
        faceScreenMesh.rotation.z = Math.PI / 2;
        faceScreenMesh.rotation.x = Math.PI / 2;
        bodyNodes.link_head_2.add(faceScreenMesh);

        cone.position.set(0.115, 0, 0.058);
        bodyNodes.link_head_2.add(cone);
      }

      if (bodyNodes.EE_BODY_R) {
        bodyNodes.EE_BODY_R.add(gripPoint);
        gripPoint.position.set(0, 0, -0.075);
      }

      loaded = true;
      update({ cur: latestCur, t: 0, dt: 0.016, k: 1, walkAmt: 0, gaitPhase: 0, gait: null });
      window.dispatchEvent(new CustomEvent('robot-twin:rig-loaded', { detail: { name: 'lg_cloid' } }));
    }).catch((err) => {
      console.error('[Real2Sim] Failed to load LG CLOiD bundle:', err);
    });
    return loadingPromise;
  }

  function getEePosMj(arm, jointState) {
    if (!loaded) return new THREE.Vector3(0.42, arm === 'left' ? 0.22 : -0.22, 0.78);
    applyJointMapToHierarchy(THREE, jointNodes, jointState);
    mujocoRoot.updateMatrixWorld(true);
    const eeNode = bodyNodes[arm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (!eeNode) return new THREE.Vector3(0.42, arm === 'left' ? 0.22 : -0.22, 0.78);
    const tipLocal = new THREE.Vector3(0, 0, -0.075);
    tipLocal.applyMatrix4(eeNode.matrixWorld);
    mujocoRoot.worldToLocal(tipLocal);
    return tipLocal;
  }

  function preparePick(tx, ty, tz) {
    activeArm = tz < -0.04 ? 'left' : 'right';
    const eeNode = bodyNodes[activeArm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (eeNode) {
      eeNode.add(gripPoint);
      gripPoint.position.set(0, 0, -0.075);
    }
  }

  function solveIkForTarget(tx, ty, tz) {
    const targetMj = new THREE.Vector3(tx, -tz, ty);
    const arm = activeArm || (tz < -0.04 ? 'left' : 'right');
    const prefix = arm === 'left' ? 'left_arm_' : 'right_arm_';
    const jNames = ['torso_4', ...[0, 1, 2, 3, 4, 5].map((i) => `${prefix}${i}`)];

    if (!loaded) {
      return {
        exprCode: 3,
        torso_3: 0.28,
        [`${prefix}0`]: -0.68,
        [`${prefix}1`]: arm === 'left' ? 0.16 : -0.16,
        [`${prefix}3`]: -0.78,
      };
    }

    const workJoints = { ...READY_JOINTS, ...latestCur, torso_3: 0.28 };
    const eps = 1e-3;
    const lam = 0.07;

    for (let iter = 0; iter < 15; iter++) {
      const curPos = getEePosMj(arm, workJoints);
      const err = new THREE.Vector3().subVectors(targetMj, curPos);
      if (err.length() < 0.004) break;
      if (err.length() > 0.08) err.setLength(0.08);

      const J = [];
      for (const jn of jNames) {
        const orig = workJoints[jn] ?? 0;
        workJoints[jn] = orig + eps;
        const pPlus = getEePosMj(arm, workJoints);
        workJoints[jn] = orig;
        J.push([
          (pPlus.x - curPos.x) / eps,
          (pPlus.y - curPos.y) / eps,
          (pPlus.z - curPos.z) / eps,
        ]);
      }

      const A = [
        [lam * lam, 0, 0],
        [0, lam * lam, 0],
        [0, 0, lam * lam],
      ];
      for (let j = 0; j < jNames.length; j++) {
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) A[r][c] += J[j][r] * J[j][c];
        }
      }
      const y = solve3x3(A, [err.x, err.y, err.z]);
      for (let j = 0; j < jNames.length; j++) {
        const jn = jNames[j];
        const jInfo = jointNodes[jn];
        const dq = J[j][0] * y[0] + J[j][1] * y[1] + J[j][2] * y[2];
        const lo = jn === 'torso_4' ? -0.55 : (jInfo ? jInfo.range[0] : -Math.PI);
        const hi = jn === 'torso_4' ? 0.55 : (jInfo ? jInfo.range[1] : Math.PI);
        workJoints[jn] = clamp((workJoints[jn] ?? 0) + dq, lo, hi);
      }
    }

    const out = { torso_3: 0.28, exprCode: 3 };
    for (const jn of jNames) out[jn] = workJoints[jn];
    return out;
  }

  function readyPose() {
    return {
      exprCode: 0,
      torso_3: 0.14,
      torso_4: 0.0,
      right_arm_0: -0.34,
      right_arm_1: -0.16,
      right_arm_2: 0.0,
      right_arm_3: -0.72,
      right_arm_4: 0.0,
      right_arm_5: -0.22,
      right_arm_6: 0.0,
      left_arm_0: -0.34,
      left_arm_1: 0.16,
      left_arm_2: 0.0,
      left_arm_3: -0.72,
      left_arm_4: 0.0,
      left_arm_5: -0.22,
      left_arm_6: 0.0,
    };
  }

  function carryPose() {
    const prefix = activeArm === 'left' ? 'left_arm_' : 'right_arm_';
    const sgn = activeArm === 'left' ? 1 : -1;
    return {
      ...readyPose(),
      exprCode: 1,
      [`${prefix}0`]: -0.58,
      [`${prefix}1`]: sgn * 0.20,
      [`${prefix}3`]: -0.98,
      [`${prefix}5`]: -0.16,
    };
  }

  function update({ cur, t, dt, gait, talk }) {
    latestCur = cur;
    if (!loaded || !robotRoot) {
      ensureLoaded();
      return;
    }

    if (gait) {
      const rWheel = 0.072;
      wheelAngleR += ((gait.vx + 0.185 * gait.vrot) / rWheel) * dt;
      wheelAngleL += ((gait.vx - 0.185 * gait.vrot) / rWheel) * dt;
    }

    const breathe = Math.sin(t * 1.8) * 0.012;
    const tb = (talk && talk.blend > 0.001 && !gait) ? talk.blend : 0;
    const te = tb ? talk.env : 0;
    const ta = tb * ((talk && talk.armScale) ?? 1);
    const tst = (talk && talk.style) || 'explain';
    const rawClose = clamp(1 - (cur.grip ?? 0.25), 0, 1);
    const closeAmt = ta > 0 ? lerp(rawClose, 0.15, ta) : rawClose;
    const thumbVal = lerp(-0.12, 0.72, closeAmt);
    const fingerVal = lerp(-0.10, 0.82, closeAmt);

    let headYaw = cur.hy ?? 0;
    let headPitch = -(cur.hp ?? -0.08);
    let expr = Math.round(cur.exprCode ?? 0);

    const frameJoints = {
      ...cur,
      right_wheel: wheelAngleR,
      left_wheel: wheelAngleL,
      torso_3: (cur.torso_3 ?? 0.14) + breathe + (gait ? clamp(gait.vx * 0.12, -0.1, 0.14) : 0),
      head_0: headYaw,
      head_1: headPitch,
      r_thumb: thumbVal,
      r_index: fingerVal,
      r_middle: fingerVal,
      r_ring: fingerVal,
      r_pinky: fingerVal,
      l_thumb: thumbVal,
      l_index: fingerVal,
      l_middle: fingerVal,
      l_ring: fingerVal,
      l_pinky: fingerVal,
    };

    if (tb > 0 && talk.R && talk.L) {
      const gR = talk.R, gL = talk.L;
      const tst = talk.style || 'explain';
      if (expr === 0 && tb > 0.25) {
        if (tst === 'affirm' || tst === 'emphasize') expr = 1;
        else if (tst === 'think') expr = 3;
      }
      // LG CLOiD MJCF convention:
      // torso_3 = Y-lean, torso_4 = X-roll, torso_5 = Z-yaw;
      // right_arm_2 > 0 and left_arm_2 < 0 bring bent forearms INWARD into Center-Center!
      headYaw += talk.headYaw;
      headPitch += -talk.headPitch;
      frameJoints.head_0 = headYaw;
      frameJoints.head_1 = headPitch;
      frameJoints.torso_3 += talk.torsoLean;
      frameJoints.torso_4 = (frameJoints.torso_4 ?? 0) + talk.torsoRoll;
      frameJoints.torso_5 = (frameJoints.torso_5 ?? 0) + talk.torsoYaw;
      if (ta > 0) {
        const r0 = -0.15 - 0.72 * gR.lift;
        const r1 = -0.13 - 0.50 * gR.abd;
        const r2 =  0.60 * gR.inward;
        const r3 = -0.44 - 1.10 * gR.flex;
        const r4 =  0.56 * gR.supinate;
        const r5 = -0.16 + 0.28 * gR.wristPitch;

        const l0 = -0.15 - 0.72 * gL.lift;
        const l1 =  0.13 + 0.50 * gL.abd;
        const l2 = -0.60 * gL.inward;
        const l3 = -0.44 - 1.10 * gL.flex;
        const l4 = -0.56 * gL.supinate;
        const l5 = -0.16 + 0.28 * gL.wristPitch;

        frameJoints.right_arm_0 = lerp(frameJoints.right_arm_0 ?? -0.20, r0, ta);
        frameJoints.right_arm_1 = lerp(frameJoints.right_arm_1 ?? -0.14, r1, ta);
        frameJoints.right_arm_2 = lerp(frameJoints.right_arm_2 ?? 0, r2, ta);
        frameJoints.right_arm_3 = lerp(frameJoints.right_arm_3 ?? -0.52, r3, ta);
        frameJoints.right_arm_4 = lerp(frameJoints.right_arm_4 ?? 0, r4, ta);
        frameJoints.right_arm_5 = lerp(frameJoints.right_arm_5 ?? -0.18, r5, ta);
        frameJoints.left_arm_0 = lerp(frameJoints.left_arm_0 ?? -0.20, l0, ta);
        frameJoints.left_arm_1 = lerp(frameJoints.left_arm_1 ?? 0.14, l1, ta);
        frameJoints.left_arm_2 = lerp(frameJoints.left_arm_2 ?? 0, l2, ta);
        frameJoints.left_arm_3 = lerp(frameJoints.left_arm_3 ?? -0.52, l3, ta);
        frameJoints.left_arm_4 = lerp(frameJoints.left_arm_4 ?? 0, l4, ta);
        frameJoints.left_arm_5 = lerp(frameJoints.left_arm_5 ?? -0.18, l5, ta);
        // Expressively open the active gesturing 5-finger hand while keeping the resting hand softly relaxed
        const rCurl = lerp(0.36, 0.06, clamp(gR.open, 0, 1));
        const lCurl = lerp(0.36, 0.06, clamp(gL.open, 0, 1));
        frameJoints.r_thumb = lerp(frameJoints.r_thumb, rCurl * 0.7, ta);
        frameJoints.r_index = lerp(frameJoints.r_index, rCurl * 0.85, ta);
        frameJoints.r_middle = lerp(frameJoints.r_middle, rCurl, ta);
        frameJoints.r_ring = lerp(frameJoints.r_ring, rCurl * 1.1, ta);
        frameJoints.r_pinky = lerp(frameJoints.r_pinky, rCurl * 1.2, ta);
        frameJoints.l_thumb = lerp(frameJoints.l_thumb, lCurl * 0.7, ta);
        frameJoints.l_index = lerp(frameJoints.l_index, lCurl * 0.85, ta);
        frameJoints.l_middle = lerp(frameJoints.l_middle, lCurl, ta);
        frameJoints.l_ring = lerp(frameJoints.l_ring, lCurl * 1.1, ta);
        frameJoints.l_pinky = lerp(frameJoints.l_pinky, lCurl * 1.2, ta);
      }
    }

    renderOledFace(t, expr, headYaw, headPitch);
    applyJointMapToHierarchy(THREE, jointNodes, frameJoints);
  }

  return {
    name: 'lg_cloid',
    icon: '✨',
    label: 'LG CLOiD (CES 2026)',
    spec: '36-DOF Home Humanoid + OLED Visor + Dual 5F Hands',
    group,
    gripPoint,
    cone,
    flashMat,
    STAND_H,
    SIT_H,
    STOW,
    ensureLoaded,
    preparePick,
    ik: solveIkForTarget,
    ready: readyPose,
    carry: carryPose,
    approachH: 0.16,
    graspH: 0.045,
    table: { x: 0.54, top: 0.66, half: { x: 0.22, z: 0.38 } },
    homeCam: new THREE.Vector3(1.80, 1.28, 1.64),
    homeTarget: new THREE.Vector3(0.14, 0.60, 0),
    defaults,
    standEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, hy: 0, hp: -0.08, exprCode: 0 });
    },
    isUp: (tgt) => (tgt.h ?? STAND_H) > STAND_H + 0.02,
    downEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, hy: 0, hp: -0.08, exprCode: 0 });
    },
    sitEnter: (tgt) => {
      Object.assign(tgt, SIT_JOINTS, { h: SIT_H, hy: 0, hp: -0.05, exprCode: 0 });
    },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.hy = clamp((tgt.hy ?? 0) + ang, -1.1, 1.1);
      else if (dir === 'right') tgt.hy = clamp((tgt.hy ?? 0) - ang, -1.1, 1.1);
      else if (dir === 'up') tgt.hp = clamp((tgt.hp ?? -0.08) + ang, -0.7, 0.35);
      else if (dir === 'down') tgt.hp = clamp((tgt.hp ?? -0.08) - ang, -0.7, 0.35);
      else { tgt.hy = 0; tgt.hp = -0.08; }
    },
    detectEnter: (tgt, setArm) => {
      tgt.hp = -0.34;
      tgt.hy = 0;
      tgt.exprCode = 3;
      setArm(readyPose());
    },
    gestures: {
      high_wave: (tgt, setArm, ctx) => [
        [0.45, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.9;
          setArm({
            right_arm_0: -0.55,
            right_arm_1: -0.90,
            right_arm_2: -0.95,
            right_arm_3: -1.72,
            right_arm_5: 0.15,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          tgt.right_arm_2 = -0.95 + (i % 2 ? 0.36 : -0.36);
          tgt.right_arm_5 = i % 2 ? 0.24 : -0.24;
          tgt.hy = (i % 2 ? 1 : -1) * 0.08;
        }]),
        [0.65, () => { tgt.hy = 0; tgt.exprCode = 0; tgt.grip = 0.25; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      shake_hand: (tgt, setArm, ctx) => [
        [0.5, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.85;
          setArm({
            torso_3: 0.26,
            right_arm_0: -0.75,
            right_arm_1: -0.08,
            right_arm_3: -1.05,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.26, () => {
          const pump = i % 2 ? 0.16 : -0.16;
          tgt.right_arm_0 = -0.75 + pump;
          tgt.right_arm_3 = -1.05 - pump;
        }]),
        [0.65, () => { tgt.torso_3 = 0.14; tgt.exprCode = 0; tgt.grip = 0.25; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      heart: (tgt, setArm, ctx) => [
        [0.55, () => {
          tgt.exprCode = 2;
          tgt.grip = 0.85;
          setArm({
            right_arm_0: -2.15,
            right_arm_1: -0.68,
            right_arm_2: 0.65,
            right_arm_3: -1.65,
            left_arm_0: -2.15,
            left_arm_1: 0.68,
            left_arm_2: -0.65,
            left_arm_3: -1.65,
          });
        }],
        [1.2, () => {}],
        [0.65, () => { tgt.exprCode = 0; tgt.grip = 0.25; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
      clap: (tgt, setArm, ctx) => [
        [0.4, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.9;
          setArm({
            right_arm_0: -0.88,
            right_arm_1: -0.18,
            right_arm_2: 0.65,
            right_arm_3: -1.35,
            left_arm_0: -0.88,
            left_arm_1: 0.18,
            left_arm_2: -0.65,
            left_arm_3: -1.35,
          });
        }],
        ...[0, 1, 2, 3, 4, 5].map((i) => [0.22, () => {
          const ab = i % 2 === 0 ? 0.28 : 0.10;
          tgt.right_arm_1 = -ab;
          tgt.left_arm_1 = ab;
        }]),
        [0.6, () => { tgt.exprCode = 0; tgt.grip = 0.25; ctx.restoreArm(); ctx.restorePosture(); }],
      ],
    },
    update,
  };
}

// ============================================================================
// 4. ENCHANTED TOOLS MIROKAÏ (28-DOF Ballbot + 60 FPS Curved Anime Face + Fox Ears)
//    [INTERNAL-TEST-ONLY: Real2Sim] Company-specific robot — stays on dev/test only
// ============================================================================
function buildMirokaiRig(THREE) {
  const group = new THREE.Group();
  const mujocoRoot = new THREE.Group();
  mujocoRoot.rotation.x = -Math.PI / 2; // MuJoCo Z-up -> Three.js Y-up
  group.add(mujocoRoot);

  const poseGroup = new THREE.Group();
  mujocoRoot.add(poseGroup);

  const gripPoint = new THREE.Group();
  group.add(gripPoint);

  const cone = makeVisionCone(THREE);
  group.add(cone);

  const flashMat = new THREE.MeshPhysicalMaterial({
    color: 0xf5f7fc,
    roughness: 0.22,
    metalness: 0.06,
    clearcoat: 0.48,
    clearcoatRoughness: 0.20,
  });

  const runeGlowMat = new THREE.MeshStandardMaterial({
    color: 0xff9e44,
    emissive: 0xff7a1a,
    emissiveIntensity: 1.45,
    roughness: 0.22,
    metalness: 0.10,
  });

  // 60 FPS 3D-Projected Curved Mask Character Face (Gobelins-inspired animated eyes, brows, blush & lip-sync)
  const faceCanvas = document.createElement('canvas');
  faceCanvas.width = 512;
  faceCanvas.height = 384;
  const faceCtx = faceCanvas.getContext('2d');
  const faceTexture = new THREE.CanvasTexture(faceCanvas);
  faceTexture.colorSpace = THREE.SRGBColorSpace;

  function renderMirokaiFace(nowSec, exprCode, headYaw, headPitch, mouthOpen = 0) {
    if (!faceCtx) return;
    const W = 512, H = 384;
    faceCtx.clearRect(0, 0, W, H);

    // Subtle enchanted inner projection aura (fades to 0 alpha at edges so curved dome has zero seam)
    const aura = faceCtx.createRadialGradient(W * 0.5, H * 0.50, 18, W * 0.5, H * 0.50, 215);
    aura.addColorStop(0.0, 'rgba(30, 64, 175, 0.24)');
    aura.addColorStop(0.58, 'rgba(15, 23, 42, 0.14)');
    aura.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
    faceCtx.fillStyle = aura;
    faceCtx.fillRect(0, 0, W, H);

    // exprCode: 0=normal, 1=happy, 2=heart, 3=focused, 4=curious/think, 5=empathy
    const blinkPhase = nowSec % 3.9;
    const isBlinking = (exprCode === 0 || exprCode === 3 || exprCode === 4 || exprCode === 5)
      && blinkPhase > 3.72 && blinkPhase < 3.86;
    const gazeX = clamp(-headYaw * 28 + (exprCode === 4 ? -12 : 0), -28, 28);
    const gazeY = clamp(headPitch * 20 + (exprCode === 4 ? -9 : 0), -18, 18);
    const leftEyeX = 154 + gazeX;
    const rightEyeX = 358 + gazeX;
    const eyeY = 172 + gazeY;

    // Warm coral cheek blush
    const blushAlpha = (exprCode === 1 || exprCode === 2 || exprCode === 5) ? 0.45 : 0.27;
    for (const bx of [96 + gazeX * 0.5, 416 + gazeX * 0.5]) {
      const bg = faceCtx.createRadialGradient(bx, eyeY + 74, 2, bx, eyeY + 74, 42);
      bg.addColorStop(0.0, `rgba(251, 113, 133, ${blushAlpha})`);
      bg.addColorStop(1.0, 'rgba(251, 113, 133, 0.0)');
      faceCtx.fillStyle = bg;
      faceCtx.beginPath();
      faceCtx.ellipse(bx, eyeY + 74, 42, 24, 0, 0, Math.PI * 2);
      faceCtx.fill();
    }

    const drawAnimeEye = (cx, cy, isLeft, ryScale = 1.0) => {
      const rx = 52, ry = 63 * ryScale;
      faceCtx.save();
      // Outer warm cyan-gold luminescence
      faceCtx.shadowColor = '#38bdf8';
      faceCtx.shadowBlur = 18;
      faceCtx.fillStyle = '#081024';
      faceCtx.beginPath();
      faceCtx.ellipse(cx, cy, rx + 5, ry + 5, 0, 0, Math.PI * 2);
      faceCtx.fill();

      // Luminous sapphire -> amber-gold iris gradient
      const ig = faceCtx.createLinearGradient(cx, cy - ry, cx, cy + ry);
      ig.addColorStop(0.0, '#0284c7');
      ig.addColorStop(0.45, '#0ea5e9');
      ig.addColorStop(0.78, '#f59e0b');
      ig.addColorStop(1.0, '#fef08a');
      faceCtx.fillStyle = ig;
      faceCtx.beginPath();
      faceCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      faceCtx.fill();

      // Deep velvety pupil
      faceCtx.shadowBlur = 0;
      faceCtx.fillStyle = '#070b19';
      faceCtx.beginPath();
      faceCtx.ellipse(cx + gazeX * 0.14, cy + gazeY * 0.14, rx * 0.52, ry * 0.56, 0, 0, Math.PI * 2);
      faceCtx.fill();

      // Lower golden crescent inner rim glow
      faceCtx.strokeStyle = 'rgba(254, 240, 138, 0.88)';
      faceCtx.lineWidth = 5.0;
      faceCtx.beginPath();
      faceCtx.arc(cx, cy + 5, rx * 0.72, 0.18 * Math.PI, 0.82 * Math.PI);
      faceCtx.stroke();

      // Dual crisp starlight specular catchlights
      faceCtx.fillStyle = '#ffffff';
      faceCtx.beginPath();
      faceCtx.ellipse(cx - 16, cy - 21 * ryScale, 13.5, 15.5 * ryScale, -0.25, 0, Math.PI * 2);
      faceCtx.fill();
      faceCtx.beginPath();
      faceCtx.arc(cx + 18, cy + 16 * ryScale, 6.5, 0, Math.PI * 2);
      faceCtx.fill();

      // Upper warm eyelash arch
      faceCtx.strokeStyle = '#fff7ed';
      faceCtx.lineWidth = 7.0;
      faceCtx.lineCap = 'round';
      faceCtx.beginPath();
      const startA = isLeft ? Math.PI * 1.08 : Math.PI * 1.14;
      const endA = isLeft ? Math.PI * 1.86 : Math.PI * 1.92;
      faceCtx.ellipse(cx, cy - 2, rx + 2, ry + 2, 0, startA, endA);
      faceCtx.stroke();
      faceCtx.restore();
    };

    const drawHappyArc = (cx, cy) => {
      faceCtx.save();
      faceCtx.shadowColor = '#fbbf24';
      faceCtx.shadowBlur = 18;
      faceCtx.strokeStyle = '#fffbeb';
      faceCtx.lineWidth = 13;
      faceCtx.lineCap = 'round';
      faceCtx.beginPath();
      faceCtx.arc(cx, cy + 16, 40, Math.PI * 1.12, Math.PI * 1.88, false);
      faceCtx.stroke();
      faceCtx.restore();
    };

    const drawHeartEye = (cx, cy) => {
      faceCtx.save();
      faceCtx.translate(cx, cy + 6);
      faceCtx.scale(1.16, 1.16);
      faceCtx.shadowColor = '#fb7185';
      faceCtx.shadowBlur = 18;
      faceCtx.fillStyle = '#fda4af';
      faceCtx.beginPath();
      faceCtx.moveTo(0, 20);
      faceCtx.bezierCurveTo(-38, -6, -40, -42, -17, -42);
      faceCtx.bezierCurveTo(-4, -42, 0, -28, 0, -21);
      faceCtx.bezierCurveTo(0, -28, 4, -42, 17, -42);
      faceCtx.bezierCurveTo(40, -42, 38, -6, 0, 20);
      faceCtx.closePath();
      faceCtx.fill();
      faceCtx.restore();
    };

    if (isBlinking) {
      faceCtx.save();
      faceCtx.strokeStyle = '#fff7ed';
      faceCtx.shadowColor = '#38bdf8';
      faceCtx.shadowBlur = 12;
      faceCtx.lineWidth = 10;
      faceCtx.lineCap = 'round';
      for (const cx of [leftEyeX, rightEyeX]) {
        faceCtx.beginPath();
        faceCtx.arc(cx, eyeY - 6, 38, 0.15 * Math.PI, 0.85 * Math.PI);
        faceCtx.stroke();
      }
      faceCtx.restore();
    } else if (exprCode === 1) {
      drawHappyArc(leftEyeX, eyeY);
      drawHappyArc(rightEyeX, eyeY);
    } else if (exprCode === 2) {
      drawHeartEye(leftEyeX, eyeY);
      drawHeartEye(rightEyeX, eyeY);
    } else if (exprCode === 3) {
      drawAnimeEye(leftEyeX, eyeY, true, 0.84);
      drawAnimeEye(rightEyeX, eyeY, false, 0.84);
    } else if (exprCode === 4) {
      drawAnimeEye(leftEyeX, eyeY + 2, true, 0.90);
      drawAnimeEye(rightEyeX, eyeY - 3, false, 1.04);
    } else {
      drawAnimeEye(leftEyeX, eyeY, true, 0.98);
      drawAnimeEye(rightEyeX, eyeY, false, 0.98);
    }

    // Expressive eyebrows
    faceCtx.save();
    faceCtx.strokeStyle = '#fef3c7';
    faceCtx.shadowColor = '#f59e0b';
    faceCtx.shadowBlur = 10;
    faceCtx.lineWidth = 7.5;
    faceCtx.lineCap = 'round';
    const drawBrow = (cx, yOff, tilt) => {
      faceCtx.beginPath();
      faceCtx.moveTo(cx - 30, eyeY - 68 + yOff + tilt * 12);
      faceCtx.quadraticCurveTo(cx, eyeY - 82 + yOff, cx + 30, eyeY - 68 + yOff - tilt * 12);
      faceCtx.stroke();
    };
    if (exprCode === 1 || exprCode === 2) {
      drawBrow(leftEyeX, -8, 0.15);
      drawBrow(rightEyeX, -8, -0.15);
    } else if (exprCode === 3) {
      drawBrow(leftEyeX, 8, -0.32);
      drawBrow(rightEyeX, 8, 0.32);
    } else if (exprCode === 4) {
      drawBrow(leftEyeX, 4, -0.18);
      drawBrow(rightEyeX, -14, -0.25);
    } else if (exprCode === 5) {
      drawBrow(leftEyeX, -2, 0.38);
      drawBrow(rightEyeX, -2, -0.38);
    } else {
      drawBrow(leftEyeX, 0, 0.08);
      drawBrow(rightEyeX, 0, -0.08);
    }
    faceCtx.restore();

    // Speech-reactive animated character mouth
    const mx = 256 + gazeX * 0.45;
    const my = 282 + gazeY * 0.42;
    faceCtx.save();
    faceCtx.shadowColor = '#fb923c';
    faceCtx.shadowBlur = 10;
    faceCtx.strokeStyle = '#fff7ed';
    faceCtx.lineWidth = 6.5;
    faceCtx.lineCap = 'round';
    faceCtx.lineJoin = 'round';

    if (mouthOpen > 0.06) {
      const mw = 22 + 12 * (1 - 0.25 * mouthOpen);
      const mh = 8 + 26 * clamp(mouthOpen, 0, 1);
      faceCtx.fillStyle = '#4c1d24';
      faceCtx.beginPath();
      faceCtx.moveTo(mx - mw, my - mh * 0.22);
      faceCtx.quadraticCurveTo(mx, my - mh * 0.42, mx + mw, my - mh * 0.22);
      faceCtx.bezierCurveTo(mx + mw * 0.95, my + mh, mx - mw * 0.95, my + mh, mx - mw, my - mh * 0.22);
      faceCtx.closePath();
      faceCtx.fill();
      // Warm coral tongue highlight
      faceCtx.save();
      faceCtx.clip();
      faceCtx.fillStyle = '#fb7185';
      faceCtx.beginPath();
      faceCtx.ellipse(mx, my + mh * 0.68, mw * 0.68, mh * 0.52, 0, 0, Math.PI * 2);
      faceCtx.fill();
      faceCtx.restore();
      faceCtx.stroke();
    } else if (exprCode === 4) {
      // Curious pondering little 'o' mouth
      faceCtx.fillStyle = '#4c1d24';
      faceCtx.beginPath();
      faceCtx.ellipse(mx + 4, my + 2, 10, 12, 0, 0, Math.PI * 2);
      faceCtx.fill();
      faceCtx.stroke();
    } else if (exprCode === 1 || exprCode === 2) {
      // Joyful open smile
      faceCtx.fillStyle = '#4c1d24';
      faceCtx.beginPath();
      faceCtx.arc(mx, my - 4, 28, 0.08 * Math.PI, 0.92 * Math.PI, false);
      faceCtx.quadraticCurveTo(mx, my + 2, mx - 27, my + 3);
      faceCtx.closePath();
      faceCtx.fill();
      faceCtx.stroke();
    } else {
      // Signature warm Mirokaï gentle cat-like smile
      faceCtx.beginPath();
      faceCtx.moveTo(mx - 24, my - 2);
      faceCtx.quadraticCurveTo(mx - 11, my + 11, mx, my + 3);
      faceCtx.quadraticCurveTo(mx + 11, my + 11, mx + 24, my - 2);
      faceCtx.stroke();
    }
    faceCtx.restore();

    faceTexture.needsUpdate = true;
  }

  renderMirokaiFace(0, 0, 0, 0.05, 0);

  const STAND_H = 0.0;
  const SIT_H = -0.048;

  const READY_JOINTS = {
    ball_roll: 0,
    ball_pitch: 0,
    balance_pitch: 0,
    balance_roll: 0,
    skirt_z: 0,
    torso_pitch: 0.04,
    torso_roll: 0,
    torso_yaw: 0,
    head_0: 0,
    head_1: 0.05,
    head_2: 0,
    ear_l_pitch: 0.06,
    ear_l_flare: -0.06,
    ear_r_pitch: 0.06,
    ear_r_flare: 0.06,
    right_arm_0: -0.22,
    right_arm_1: -0.15,
    right_arm_2: 0.06,
    right_arm_3: -0.56,
    right_arm_4: 0.0,
    right_arm_5: -0.16,
    right_arm_6: 0.0,
    r_fingers: 0.22,
    r_thumb: 0.18,
    left_arm_0: -0.22,
    left_arm_1: 0.15,
    left_arm_2: -0.06,
    left_arm_3: -0.56,
    left_arm_4: 0.0,
    left_arm_5: -0.16,
    left_arm_6: 0.0,
    l_fingers: 0.22,
    l_thumb: 0.18,
  };

  const SIT_JOINTS = {
    ...READY_JOINTS,
    skirt_z: -0.048,
    torso_pitch: 0.18,
    head_1: 0.08,
    ear_l_pitch: -0.12,
    ear_l_flare: -0.26,
    ear_r_pitch: -0.12,
    ear_r_flare: 0.26,
    right_arm_0: -0.38,
    right_arm_1: -0.14,
    right_arm_2: 0.24,
    right_arm_3: -0.92,
    left_arm_0: -0.38,
    left_arm_1: 0.14,
    left_arm_2: -0.24,
    left_arm_3: -0.92,
  };

  const STOW = {
    torso_pitch: 0.04,
    torso_yaw: 0.0,
    right_arm_0: -0.22,
    right_arm_1: -0.15,
    right_arm_2: 0.06,
    right_arm_3: -0.56,
    right_arm_4: 0.0,
    right_arm_5: -0.16,
    right_arm_6: 0.0,
    left_arm_0: -0.22,
    left_arm_1: 0.15,
    left_arm_2: -0.06,
    left_arm_3: -0.56,
    left_arm_4: 0.0,
    left_arm_5: -0.16,
    left_arm_6: 0.0,
  };

  const defaults = {
    h: STAND_H,
    pitch: 0,
    roll: 0,
    hy: 0,
    hp: -0.05,
    grip: 0.25,
    exprCode: 0,
    ...READY_JOINTS,
  };

  let loaded = false;
  let loadingPromise = null;
  let bodyNodes = {};
  let jointNodes = {};
  let robotRoot = null;
  let activeArm = 'right';
  let ballPitchAngle = 0;
  let ballRollAngle = 0;
  let dynLeanPitch = 0;
  let dynLeanRoll = 0;
  let latestCur = { ...defaults };

  function ensureLoaded() {
    if (loadingPromise) return loadingPromise;
    loadingPromise = fetchMjcfBundle(
      THREE,
      '/static/mirokai_assets.bin?v=1',
      'MRKI',
      (T) => {
        const materials = {
          miro_pearl_white: flashMat,
          miro_space_indigo: new T.MeshStandardMaterial({
            color: 0x1e2540,
            roughness: 0.42,
            metalness: 0.28,
          }),
          miro_copper_orange: new T.MeshPhysicalMaterial({
            color: 0xea580c,
            roughness: 0.34,
            metalness: 0.16,
            clearcoat: 0.35,
            clearcoatRoughness: 0.28,
          }),
          miro_gold_trim: new T.MeshStandardMaterial({
            color: 0xf59e0b,
            roughness: 0.26,
            metalness: 0.72,
          }),
          miro_visor_dark: new T.MeshPhysicalMaterial({
            color: 0x060814,
            roughness: 0.10,
            metalness: 0.22,
            clearcoat: 0.94,
            clearcoatRoughness: 0.06,
          }),
          miro_ball_slate: new T.MeshStandardMaterial({
            color: 0x262b38,
            roughness: 0.62,
            metalness: 0.18,
          }),
          miro_ball_tread: new T.MeshStandardMaterial({
            color: 0x475569,
            roughness: 0.50,
            metalness: 0.32,
          }),
          miro_amber_led: new T.MeshStandardMaterial({
            color: 0xfbbf24,
            emissive: 0xf59e0b,
            emissiveIntensity: 1.35,
            roughness: 0.22,
          }),
          miro_rune_glow: runeGlowMat,
        };
        return { materials, defaultMat: flashMat, flashMat };
      }
    ).then((res) => {
      robotRoot = res.robotRoot;
      bodyNodes = res.bodyNodes;
      jointNodes = res.jointNodes;
      poseGroup.add(robotRoot);

      // Conformally contour the 60 FPS 3D-Projected Face Screen onto head_visor_dark
      // Exact superellipsoid front surface: center=[0.082, 0, 0.102], rx=0.050, ry=0.108, rz=0.082, n=0.82 (p=2/0.82)
      if (bodyNodes.link_head) {
        const screenGeo = new THREE.PlaneGeometry(0.168, 0.122, 28, 20);
        const posAttr = screenGeo.attributes.position;
        const pExp = 2.0 / 0.82;
        for (let i = 0; i < posAttr.count; i++) {
          const u = posAttr.getX(i);
          const v = posAttr.getY(i);
          const sRem = Math.max(
            0,
            1.0 - Math.pow(Math.abs(u) / 0.108, pExp) - Math.pow(Math.abs(v) / 0.082, pExp)
          );
          const curveZ = 0.050 * (Math.pow(sRem, 0.41) - 1.0);
          posAttr.setZ(i, curveZ);
        }
        screenGeo.computeVertexNormals();
        const screenMat = new THREE.MeshBasicMaterial({
          map: faceTexture,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });
        const faceScreenMesh = new THREE.Mesh(screenGeo, screenMat);
        faceScreenMesh.position.set(0.1348, 0.0, 0.102);
        faceScreenMesh.rotation.order = 'ZYX';
        faceScreenMesh.rotation.z = Math.PI / 2;
        faceScreenMesh.rotation.x = Math.PI / 2;
        bodyNodes.link_head.add(faceScreenMesh);

        cone.position.set(0.136, 0, 0.102);
        bodyNodes.link_head.add(cone);
      }

      if (bodyNodes.EE_BODY_R) {
        bodyNodes.EE_BODY_R.add(gripPoint);
        gripPoint.position.set(0, 0, -0.068);
      }

      loaded = true;
      update({ cur: latestCur, t: 0, dt: 0.016, k: 1, walkAmt: 0, gaitPhase: 0, gait: null });
      window.dispatchEvent(new CustomEvent('robot-twin:rig-loaded', { detail: { name: 'mirokai' } }));
    }).catch((err) => {
      console.error('[Real2Sim] Failed to load Mirokaï bundle:', err);
    });
    return loadingPromise;
  }

  function getEePosMj(arm, jointState) {
    if (!loaded) return new THREE.Vector3(0.40, arm === 'left' ? 0.18 : -0.18, 0.64);
    applyJointMapToHierarchy(THREE, jointNodes, jointState);
    mujocoRoot.updateMatrixWorld(true);
    const eeNode = bodyNodes[arm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (!eeNode) return new THREE.Vector3(0.40, arm === 'left' ? 0.18 : -0.18, 0.64);
    const tipLocal = new THREE.Vector3(0, 0, -0.068);
    tipLocal.applyMatrix4(eeNode.matrixWorld);
    mujocoRoot.worldToLocal(tipLocal);
    return tipLocal;
  }

  function preparePick(tx, ty, tz) {
    activeArm = tz < -0.04 ? 'left' : 'right';
    const eeNode = bodyNodes[activeArm === 'left' ? 'EE_BODY_L' : 'EE_BODY_R'];
    if (eeNode) {
      eeNode.add(gripPoint);
      gripPoint.position.set(0, 0, -0.068);
    }
  }

  function solveIkForTarget(tx, ty, tz) {
    const targetMj = new THREE.Vector3(tx, -tz, ty);
    const arm = activeArm || (tz < -0.04 ? 'left' : 'right');
    const prefix = arm === 'left' ? 'left_arm_' : 'right_arm_';
    const jNames = ['torso_yaw', ...[0, 1, 2, 3, 4, 5].map((i) => `${prefix}${i}`)];

    if (!loaded) {
      return {
        exprCode: 3,
        torso_pitch: 0.20,
        [`${prefix}0`]: -0.72,
        [`${prefix}1`]: arm === 'left' ? 0.16 : -0.16,
        [`${prefix}3`]: -0.82,
      };
    }

    const workJoints = {
      ...READY_JOINTS,
      ...latestCur,
      balance_pitch: 0,
      balance_roll: 0,
      torso_pitch: 0.20,
      torso_roll: 0,
    };
    const eps = 1e-3;
    const lam = 0.065;

    for (let iter = 0; iter < 16; iter++) {
      const curPos = getEePosMj(arm, workJoints);
      const err = new THREE.Vector3().subVectors(targetMj, curPos);
      if (err.length() < 0.004) break;
      if (err.length() > 0.08) err.setLength(0.08);

      const J = [];
      for (const jn of jNames) {
        const orig = workJoints[jn] ?? 0;
        workJoints[jn] = orig + eps;
        const pPlus = getEePosMj(arm, workJoints);
        workJoints[jn] = orig;
        J.push([
          (pPlus.x - curPos.x) / eps,
          (pPlus.y - curPos.y) / eps,
          (pPlus.z - curPos.z) / eps,
        ]);
      }

      const A = [
        [lam * lam, 0, 0],
        [0, lam * lam, 0],
        [0, 0, lam * lam],
      ];
      for (let j = 0; j < jNames.length; j++) {
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) A[r][c] += J[j][r] * J[j][c];
        }
      }
      const y = solve3x3(A, [err.x, err.y, err.z]);
      for (let j = 0; j < jNames.length; j++) {
        const jn = jNames[j];
        const jInfo = jointNodes[jn];
        const dq = J[j][0] * y[0] + J[j][1] * y[1] + J[j][2] * y[2];
        const lo = jn === 'torso_yaw' ? -0.60 : (jInfo ? jInfo.range[0] : -Math.PI);
        const hi = jn === 'torso_yaw' ? 0.60 : (jInfo ? jInfo.range[1] : Math.PI);
        workJoints[jn] = clamp((workJoints[jn] ?? 0) + dq, lo, hi);
      }
    }

    const out = {
      torso_pitch: 0.20,
      exprCode: 3,
      ear_l_pitch: 0.24,
      ear_r_pitch: 0.24,
    };
    for (const jn of jNames) out[jn] = workJoints[jn];
    return out;
  }

  function readyPose() {
    return {
      exprCode: 0,
      torso_pitch: 0.06,
      torso_yaw: 0.0,
      ear_l_pitch: 0.08,
      ear_l_flare: -0.08,
      ear_r_pitch: 0.08,
      ear_r_flare: 0.08,
      right_arm_0: -0.34,
      right_arm_1: -0.16,
      right_arm_2: 0.08,
      right_arm_3: -0.74,
      right_arm_4: 0.0,
      right_arm_5: -0.20,
      right_arm_6: 0.0,
      left_arm_0: -0.34,
      left_arm_1: 0.16,
      left_arm_2: -0.08,
      left_arm_3: -0.74,
      left_arm_4: 0.0,
      left_arm_5: -0.20,
      left_arm_6: 0.0,
    };
  }

  function carryPose() {
    const prefix = activeArm === 'left' ? 'left_arm_' : 'right_arm_';
    const sgn = activeArm === 'left' ? 1 : -1;
    return {
      ...readyPose(),
      torso_pitch: 0.04,
      exprCode: 1,
      ear_l_pitch: 0.18,
      ear_r_pitch: 0.18,
      [`${prefix}0`]: -0.96,
      [`${prefix}1`]: sgn * 0.18,
      [`${prefix}2`]: -sgn * 0.26,
      [`${prefix}3`]: -1.26,
      [`${prefix}5`]: -0.08,
    };
  }

  function update({ cur, t, dt, gait, talk }) {
    latestCur = cur;
    if (!loaded || !robotRoot) {
      ensureLoaded();
      return;
    }

    const rBall = 0.135;
    let targetLeanPitch = 0;
    let targetLeanRoll = 0;
    if (gait) {
      ballPitchAngle += (gait.vx / rBall) * dt;
      ballRollAngle -= ((gait.vy + 0.14 * gait.vrot) / rBall) * dt;
      targetLeanPitch = clamp(gait.vx * 0.18, -0.14, 0.16);
      targetLeanRoll = clamp(-gait.vy * 0.14 - gait.vrot * 0.10, -0.14, 0.14);
    } else {
      // Subtle inverted-pendulum ballbot active self-balancing micro-sway
      targetLeanPitch = Math.sin(t * 1.45) * 0.008 + Math.sin(t * 2.73 + 0.6) * 0.004;
      targetLeanRoll = Math.sin(t * 1.18 + 1.2) * 0.007;
    }
    const kLean = 1 - Math.exp(-Math.max(0.001, dt) * 6.5);
    dynLeanPitch += (targetLeanPitch - dynLeanPitch) * kLean;
    dynLeanRoll += (targetLeanRoll - dynLeanRoll) * kLean;

    const breathe = Math.sin(t * 2.05) * 0.012;
    const tb = (talk && talk.blend > 0.001 && !gait) ? talk.blend : 0;
    const te = tb ? talk.env : 0;
    const ta = tb * ((talk && talk.armScale) ?? 1);
    const tst = (talk && talk.style) || 'explain';

    // Chest Rune-Heart core pulses softly in idle and glows vibrantly with speech prosody
    runeGlowMat.emissiveIntensity = tb > 0.02
      ? 1.45 + 1.55 * te + 0.55 * ((talk && talk.beat) || 0)
      : 1.25 + 0.25 * Math.sin(t * 2.2);

    const rawClose = clamp(1 - (cur.grip ?? 0.25), 0, 1);
    const closeAmt = ta > 0 ? lerp(rawClose, 0.16, ta) : rawClose;
    const fingerVal = lerp(-0.14, 0.96, closeAmt);
    const thumbVal = lerp(-0.18, 0.84, closeAmt);

    let headYaw = cur.hy ?? 0;
    let headPitch = -(cur.hp ?? -0.05) - dynLeanPitch * 0.65;
    let headRoll = (cur.head_2 ?? 0) - dynLeanRoll * 0.5;
    let expr = Math.round(cur.exprCode ?? 0);

    // Organic idle ear twitches
    const earTwitchL = Math.sin(t * 1.62) * 0.025 + Math.pow(Math.max(0, Math.sin(t * 0.83)), 14) * 0.14;
    const earTwitchR = Math.sin(t * 1.48 + 1.4) * 0.025 + Math.pow(Math.max(0, Math.sin(t * 0.71 + 2.1)), 14) * 0.14;
    let earLP = (cur.ear_l_pitch ?? 0.06) + earTwitchL + (gait ? -clamp(gait.vx * 0.35, -0.1, 0.24) : 0);
    let earLF = (cur.ear_l_flare ?? -0.06);
    let earRP = (cur.ear_r_pitch ?? 0.06) + earTwitchR + (gait ? -clamp(gait.vx * 0.35, -0.1, 0.24) : 0);
    let earRF = (cur.ear_r_flare ?? 0.06);

    const frameJoints = {
      ...cur,
      ball_roll: ballRollAngle,
      ball_pitch: ballPitchAngle,
      balance_pitch: dynLeanPitch,
      balance_roll: dynLeanRoll,
      torso_pitch: (cur.torso_pitch ?? 0.04) + breathe,
      head_0: headYaw,
      head_1: headPitch,
      head_2: headRoll,
      ear_l_pitch: earLP,
      ear_l_flare: earLF,
      ear_r_pitch: earRP,
      ear_r_flare: earRF,
      r_fingers: fingerVal,
      r_thumb: thumbVal,
      l_fingers: fingerVal,
      l_thumb: thumbVal,
    };

    let mouthOpen = 0;
    if (tb > 0 && talk.R && talk.L) {
      const gR = talk.R, gL = talk.L;
      const beat = talk.beat || 0;
      mouthOpen = clamp((te - 0.10) * 1.25, 0, 1) * tb;

      if (expr === 0 && tb > 0.25) {
        if (tst === 'affirm') expr = 1;
        else if (tst === 'think') expr = 4;
        else if (tst === 'empathy') expr = 5;
      }

      // Expressive fox-ear prosody & emotion choreography
      if (tst === 'think') {
        earRP = lerp(earRP, 0.28 + 0.08 * beat, tb);
        earRF = lerp(earRF, 0.26, tb);
        earLP = lerp(earLP, -0.16, tb);
        earLF = lerp(earLF, -0.05, tb);
      } else if (tst === 'empathy') {
        earLP = lerp(earLP, -0.08 + 0.06 * beat, tb);
        earLF = lerp(earLF, -0.34, tb);
        earRP = lerp(earRP, -0.08 + 0.06 * beat, tb);
        earRF = lerp(earRF, 0.34, tb);
      } else if (tst === 'affirm' || tst === 'emphasize') {
        earLP = lerp(earLP, 0.22 + 0.14 * beat, tb);
        earLF = lerp(earLF, -0.14, tb);
        earRP = lerp(earRP, 0.22 + 0.14 * beat, tb);
        earRF = lerp(earRF, 0.14, tb);
      } else {
        earLP = lerp(earLP, 0.14 + 0.11 * beat, tb);
        earLF = lerp(earLF, -0.10 - 0.04 * beat, tb);
        earRP = lerp(earRP, 0.14 + 0.11 * beat, tb);
        earRF = lerp(earRF, 0.10 + 0.04 * beat, tb);
      }
      frameJoints.ear_l_pitch = earLP;
      frameJoints.ear_l_flare = earLF;
      frameJoints.ear_r_pitch = earRP;
      frameJoints.ear_r_flare = earRF;

      headYaw += talk.headYaw;
      headPitch += -talk.headPitch;
      headRoll += talk.headRoll * 1.15;
      frameJoints.head_0 = headYaw;
      frameJoints.head_1 = headPitch;
      frameJoints.head_2 = headRoll;
      frameJoints.torso_pitch += talk.torsoLean * 0.9;
      frameJoints.torso_roll = (frameJoints.torso_roll ?? 0) + talk.torsoRoll;
      frameJoints.torso_yaw = (frameJoints.torso_yaw ?? 0) + talk.torsoYaw;

      if (ta > 0) {
        // Center-Center gesture space: right_arm_2 > 0 & left_arm_2 < 0 bring bent forearms cleanly inward
        const r0 = -0.16 - 0.68 * gR.lift;
        const r1 = -0.14 - 0.46 * gR.abd;
        const r2 =  0.58 * gR.inward;
        const r3 = -0.46 - 1.06 * gR.flex;
        const r4 =  0.54 * gR.supinate;
        const r5 = -0.14 + 0.26 * gR.wristPitch;

        const l0 = -0.16 - 0.68 * gL.lift;
        const l1 =  0.14 + 0.46 * gL.abd;
        const l2 = -0.58 * gL.inward;
        const l3 = -0.46 - 1.06 * gL.flex;
        const l4 = -0.54 * gL.supinate;
        const l5 = -0.14 + 0.26 * gL.wristPitch;

        frameJoints.right_arm_0 = lerp(frameJoints.right_arm_0 ?? -0.22, r0, ta);
        frameJoints.right_arm_1 = lerp(frameJoints.right_arm_1 ?? -0.15, r1, ta);
        frameJoints.right_arm_2 = lerp(frameJoints.right_arm_2 ?? 0.06, r2, ta);
        frameJoints.right_arm_3 = lerp(frameJoints.right_arm_3 ?? -0.56, r3, ta);
        frameJoints.right_arm_4 = lerp(frameJoints.right_arm_4 ?? 0, r4, ta);
        frameJoints.right_arm_5 = lerp(frameJoints.right_arm_5 ?? -0.16, r5, ta);
        frameJoints.left_arm_0 = lerp(frameJoints.left_arm_0 ?? -0.22, l0, ta);
        frameJoints.left_arm_1 = lerp(frameJoints.left_arm_1 ?? 0.15, l1, ta);
        frameJoints.left_arm_2 = lerp(frameJoints.left_arm_2 ?? -0.06, l2, ta);
        frameJoints.left_arm_3 = lerp(frameJoints.left_arm_3 ?? -0.56, l3, ta);
        frameJoints.left_arm_4 = lerp(frameJoints.left_arm_4 ?? 0, l4, ta);
        frameJoints.left_arm_5 = lerp(frameJoints.left_arm_5 ?? -0.16, l5, ta);

        const rOpenCurl = lerp(0.42, 0.04, clamp(gR.open, 0, 1));
        const lOpenCurl = lerp(0.42, 0.04, clamp(gL.open, 0, 1));
        frameJoints.r_fingers = lerp(frameJoints.r_fingers, rOpenCurl, ta);
        frameJoints.r_thumb = lerp(frameJoints.r_thumb, rOpenCurl * 0.75, ta);
        frameJoints.l_fingers = lerp(frameJoints.l_fingers, lOpenCurl, ta);
        frameJoints.l_thumb = lerp(frameJoints.l_thumb, lOpenCurl * 0.75, ta);
      }
    }

    renderMirokaiFace(t, expr, headYaw, headPitch, mouthOpen);
    applyJointMapToHierarchy(THREE, jointNodes, frameJoints);
  }

  return {
    name: 'mirokai',
    icon: '🦊',
    label: 'Mirokaï (Enchanted Tools)',
    spec: '28-DOF Ballbot + 60 FPS Anime Face + Fox Ears + Rune-Grip',
    group,
    gripPoint,
    cone,
    flashMat,
    STAND_H,
    SIT_H,
    STOW,
    ensureLoaded,
    preparePick,
    ik: solveIkForTarget,
    ready: readyPose,
    carry: carryPose,
    approachH: 0.18,
    graspH: 0.048,
    table: { x: 0.52, top: 0.44, half: { x: 0.20, z: 0.36 } },
    homeCam: new THREE.Vector3(1.66, 1.16, 1.48),
    homeTarget: new THREE.Vector3(0.10, 0.60, 0),
    defaults,
    standEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, {
        h: STAND_H,
        skirt_z: 0.025,
        hy: 0,
        hp: -0.05,
        exprCode: 1,
        ear_l_pitch: 0.22,
        ear_r_pitch: 0.22,
      });
    },
    isUp: (tgt) => (tgt.skirt_z ?? 0) > 0.012,
    downEnter: (tgt) => {
      Object.assign(tgt, READY_JOINTS, { h: STAND_H, skirt_z: 0, hy: 0, hp: -0.05, exprCode: 0 });
    },
    sitEnter: (tgt) => {
      Object.assign(tgt, SIT_JOINTS, { h: SIT_H, hy: 0, hp: -0.04, exprCode: 0 });
    },
    look: (tgt, dir, ang) => {
      if (dir === 'left') tgt.hy = clamp((tgt.hy ?? 0) + ang, -1.1, 1.1);
      else if (dir === 'right') tgt.hy = clamp((tgt.hy ?? 0) - ang, -1.1, 1.1);
      else if (dir === 'up') tgt.hp = clamp((tgt.hp ?? -0.05) + ang, -0.55, 0.38);
      else if (dir === 'down') tgt.hp = clamp((tgt.hp ?? -0.05) - ang, -0.55, 0.38);
      else { tgt.hy = 0; tgt.hp = -0.05; }
    },
    detectEnter: (tgt, setArm) => {
      tgt.hp = -0.32;
      tgt.hy = 0;
      tgt.exprCode = 3;
      tgt.ear_l_pitch = 0.26;
      tgt.ear_r_pitch = 0.26;
      setArm(readyPose());
    },
    gestures: {
      high_wave: (tgt, setArm, ctx) => [
        [0.45, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.92;
          tgt.head_2 = 0.12;
          tgt.ear_l_pitch = 0.26;
          tgt.ear_r_pitch = 0.28;
          tgt.ear_l_flare = -0.18;
          tgt.ear_r_flare = 0.18;
          setArm({
            right_arm_0: -0.58,
            right_arm_1: -0.88,
            right_arm_2: -0.92,
            right_arm_3: -1.68,
            right_arm_5: 0.14,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.25, () => {
          const s = i % 2 ? 1 : -1;
          tgt.right_arm_2 = -0.92 + s * 0.34;
          tgt.right_arm_5 = s * 0.22;
          tgt.hy = s * 0.08;
          tgt.ear_l_flare = -0.18 - s * 0.08;
          tgt.ear_r_flare = 0.18 - s * 0.08;
        }]),
        [0.65, () => {
          tgt.hy = 0;
          tgt.head_2 = 0;
          tgt.exprCode = 0;
          tgt.ear_l_pitch = 0.06;
          tgt.ear_r_pitch = 0.06;
          tgt.ear_l_flare = -0.06;
          tgt.ear_r_flare = 0.06;
          tgt.grip = 0.25;
          ctx.restoreArm();
          ctx.restorePosture();
        }],
      ],
      shake_hand: (tgt, setArm, ctx) => [
        [0.5, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.85;
          tgt.ear_l_pitch = 0.20;
          tgt.ear_r_pitch = 0.20;
          setArm({
            torso_pitch: 0.22,
            right_arm_0: -0.78,
            right_arm_1: -0.08,
            right_arm_2: 0.10,
            right_arm_3: -1.02,
          });
        }],
        ...[0, 1, 2, 3, 4].map((i) => [0.25, () => {
          const pump = i % 2 ? 0.16 : -0.16;
          tgt.right_arm_0 = -0.78 + pump;
          tgt.right_arm_3 = -1.02 - pump;
        }]),
        [0.65, () => {
          tgt.torso_pitch = 0.04;
          tgt.exprCode = 0;
          tgt.ear_l_pitch = 0.06;
          tgt.ear_r_pitch = 0.06;
          tgt.grip = 0.25;
          ctx.restoreArm();
          ctx.restorePosture();
        }],
      ],
      heart: (tgt, setArm, ctx) => [
        [0.55, () => {
          tgt.exprCode = 2;
          tgt.grip = 0.72;
          tgt.head_2 = 0.09;
          tgt.ear_l_pitch = 0.24;
          tgt.ear_r_pitch = 0.24;
          tgt.ear_l_flare = -0.28;
          tgt.ear_r_flare = 0.28;
          setArm({
            right_arm_0: -1.18,
            right_arm_1: -0.26,
            right_arm_2: 0.88,
            right_arm_3: -1.78,
            right_arm_5: 0.42,
            left_arm_0: -1.18,
            left_arm_1: 0.26,
            left_arm_2: -0.88,
            left_arm_3: -1.78,
            left_arm_5: 0.42,
          });
        }],
        [1.25, () => { tgt.head_2 = -0.10; }],
        [0.65, () => {
          tgt.head_2 = 0;
          tgt.exprCode = 0;
          tgt.ear_l_pitch = 0.06;
          tgt.ear_r_pitch = 0.06;
          tgt.ear_l_flare = -0.06;
          tgt.ear_r_flare = 0.06;
          tgt.grip = 0.25;
          ctx.restoreArm();
          ctx.restorePosture();
        }],
      ],
      clap: (tgt, setArm, ctx) => [
        [0.4, () => {
          tgt.exprCode = 1;
          tgt.grip = 0.9;
          tgt.ear_l_pitch = 0.24;
          tgt.ear_r_pitch = 0.24;
          setArm({
            right_arm_0: -0.88,
            right_arm_1: -0.18,
            right_arm_2: 0.64,
            right_arm_3: -1.34,
            left_arm_0: -0.88,
            left_arm_1: 0.18,
            left_arm_2: -0.64,
            left_arm_3: -1.34,
          });
        }],
        ...[0, 1, 2, 3, 4, 5].map((i) => [0.22, () => {
          const ab = i % 2 === 0 ? 0.28 : 0.09;
          tgt.right_arm_1 = -ab;
          tgt.left_arm_1 = ab;
        }]),
        [0.6, () => {
          tgt.exprCode = 0;
          tgt.ear_l_pitch = 0.06;
          tgt.ear_r_pitch = 0.06;
          tgt.grip = 0.25;
          ctx.restoreArm();
          ctx.restorePosture();
        }],
      ],
    },
    update,
  };
}

export function buildReal2SimRigs(THREE) {
  return {
    unitree_r1: buildUnitreeR1Rig(THREE),
    rby1: buildRby1Rig(THREE),
    lg_cloid: buildLgCloidRig(THREE),
    mirokai: buildMirokaiRig(THREE),
  };
}
