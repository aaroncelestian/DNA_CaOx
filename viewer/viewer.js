import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODELS = window.DNA_CAOX_MODELS || { templating_gel: window.DNA_CAOX_MODEL };
const DEFAULT_GEOM = "templating_gel_thick";
let MODEL =
  MODELS[DEFAULT_GEOM] ||
  MODELS.templating_gel ||
  window.DNA_CAOX_MODEL;
if (!MODEL) {
  throw new Error("model-data.js did not load");
}

const FILTER_SLAB_FULL = 74;
const PHASES = ["amorphous", "intermediate", "crystalline"];
// Warm gold yellow — triad with intermediate purple (#7570b3) and amorphous teal (#1b9e77)
const ACCENT_YELLOW = {
  core: 0xfff066,
  bright: 0xfff9a8,
  deep: 0xffd24d,
  emissive: 0xffe033,
  glow: 0xfff2aa,
};
const PHASE_COLOR = {
  amorphous: 0x1b9e77,
  intermediate: 0x7570b3,
  crystalline: 0xffffff,
  nucleation: ACCENT_YELLOW.bright,
  shell: 0xd95f02,
};
// DNA backbone + phosphate palette (pink family for legibility vs gold hotspots)
const DNA_PINK = {
  backbone: 0xf5b0d0,
  phosphate: 0xff9ec8,
  trace: 0xa8386e,
};
const STRAND_COLOR = {
  A: DNA_PINK.backbone,
  B: DNA_PINK.backbone,
  C: DNA_PINK.backbone,
  D: DNA_PINK.backbone,
};

let ca = MODEL.ca;
let nCa = ca.x.length;
let trajData = null;
let trajFrame = 0;
let trajPlaying = false;
let trajTimer = null;
let trajFrame0Ca = null;
let trajFrame0Oxalate = null;
let trajAmp = 1;
let trajMaxDisp = 0;
let pendingTrajFrame = null;
let sessionRestoring = false;
let sessionSaveTimer = null;
const caRest = { x: [...ca.x], y: [...ca.y], z: [...ca.z] };
let oxalateLines = null;
let oxalateRest = null;
let oxalateUnitOffsets = [];
let waterMesh = null;

const canvas = document.getElementById("c");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: true,
  });
} catch (err) {
  document.getElementById("source-label").textContent =
    "WebGL did not start. Try Chrome or Safari, not a file:// URL.";
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080a);

const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 800);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = false;
controls.autoRotateSpeed = 0.45; // ~2.2 min per orbit at 60 fps
controls.zoomToCursor = false;
controls.enablePan = false; // all pan via viewPan below
controls.touches = { ONE: THREE.TOUCH.ROTATE };
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};
controls.target.set(0, 0, 0);
controls.cursor.copy(controls.target);

const helixCenter = new THREE.Vector3();
const panRight = new THREE.Vector3();
const panUp = new THREE.Vector3();
const viewPan = new THREE.Vector3();
const panState = { active: false, pointerId: null, lastX: 0, lastY: 0 };
let spacePanArm = false;
let canvasPointers = 0;

function getHelixCenter() {
  const hx = MODEL.helix || {};
  const yMid = 0.5 * ((hx.dnaZmin ?? 0) + (hx.dnaZmax ?? 0));
  helixCenter.set(0, yMid, 0);
  return helixCenter;
}

function syncOrbitTarget() {
  getHelixCenter();
  controls.target.copy(helixCenter);
  controls.cursor.copy(helixCenter);
}

function panPixelScale() {
  const el = canvas;
  if (!el.clientHeight) return 0;
  let dist = camera.position.distanceTo(controls.target);
  dist *= Math.tan((camera.fov * Math.PI) / 360);
  return (2 * dist) / el.clientHeight;
}

function wheelPanDeltas(e) {
  let dx = e.deltaX;
  let dy = e.deltaY;
  if (e.deltaMode === 1) {
    dx *= 16;
    dy *= 16;
  } else if (e.deltaMode === 2) {
    dx *= 100;
    dy *= 100;
  }
  return { dx, dy };
}

function panViewByPixels(deltaX, deltaY) {
  const scale = panPixelScale();
  if (!scale) return;
  camera.updateMatrixWorld();
  panRight.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(-deltaX * scale);
  viewPan.add(panRight);
  panUp.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(deltaY * scale);
  viewPan.add(panUp);
}

function wantsPanPointer(e) {
  if (spacePanArm && e.button === 0) return true;
  if (canvasPointers > 1) return true;
  return (
    e.button === 2 ||
    e.button === 1 ||
    (e.button === 0 && (e.shiftKey || e.altKey))
  );
}

function onDocumentPanMove(e) {
  if (!panState.active) return;
  if (panState.pointerId != null && e.pointerId !== panState.pointerId) return;
  const dx = e.clientX - panState.lastX;
  const dy = e.clientY - panState.lastY;
  panState.lastX = e.clientX;
  panState.lastY = e.clientY;
  if (dx || dy) panViewByPixels(dx, dy);
  e.preventDefault();
}

function onDocumentPanEnd(e) {
  if (panState.pointerId != null && e.pointerId !== panState.pointerId) return;
  panState.active = false;
  panState.pointerId = null;
  document.removeEventListener("pointermove", onDocumentPanMove);
  document.removeEventListener("pointerup", onDocumentPanEnd);
  document.removeEventListener("pointercancel", onDocumentPanEnd);
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (_) {
    /* ignore */
  }
}

function beginPanDrag(e) {
  controls.state = -1;
  panState.active = true;
  panState.pointerId = e.pointerId;
  panState.lastX = e.clientX;
  panState.lastY = e.clientY;
  if (e.pointerId === -1) {
    document.addEventListener("mousemove", onMousePanMove);
    document.addEventListener("mouseup", onMousePanEnd);
  } else {
    document.addEventListener("pointermove", onDocumentPanMove);
    document.addEventListener("pointerup", onDocumentPanEnd);
    document.addEventListener("pointercancel", onDocumentPanEnd);
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch (_) {
      /* ignore */
    }
  }
  e.preventDefault();
  e.stopImmediatePropagation();
}

function onMousePanMove(e) {
  if (!panState.active || panState.pointerId !== -1) return;
  const dx = e.clientX - panState.lastX;
  const dy = e.clientY - panState.lastY;
  panState.lastX = e.clientX;
  panState.lastY = e.clientY;
  if (dx || dy) panViewByPixels(dx, dy);
  e.preventDefault();
}

function onMousePanEnd(e) {
  if (!panState.active || panState.pointerId !== -1) return;
  panState.active = false;
  panState.pointerId = null;
  document.removeEventListener("mousemove", onMousePanMove);
  document.removeEventListener("mouseup", onMousePanEnd);
}

function setupCanvasPan() {
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat) return;
    if (e.target.closest("#panel input, #panel textarea, #panel select, #panel button")) return;
    spacePanArm = true;
    e.preventDefault();
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space") spacePanArm = false;
  });

  canvas.addEventListener(
    "pointerdown",
    (e) => {
      canvasPointers++;
      if (!wantsPanPointer(e)) return;
      beginPanDrag(e);
    },
    { capture: true }
  );

  canvas.addEventListener(
    "pointerup",
    (e) => {
      canvasPointers = Math.max(0, canvasPointers - 1);
    },
    { capture: true }
  );
  canvas.addEventListener(
    "pointercancel",
    (e) => {
      canvasPointers = Math.max(0, canvasPointers - 1);
    },
    { capture: true }
  );

  canvas.addEventListener(
    "mousedown",
    (e) => {
      if (!wantsPanPointer(e) || panState.active) return;
      beginPanDrag({
        pointerId: -1,
        clientX: e.clientX,
        clientY: e.clientY,
        preventDefault: () => e.preventDefault(),
        stopImmediatePropagation: () => e.stopImmediatePropagation(),
      });
    },
    { capture: true }
  );

  // Two-finger trackpad scroll/drag → pan; pinch (ctrlKey) → OrbitControls zoom
  const onWheelPan = (e) => {
    if (e.ctrlKey) return;
    if (e.target.closest("#panel")) return;
    const app = document.getElementById("app");
    if (!app || !app.contains(e.target)) return;
    const { dx, dy } = wheelPanDeltas(e);
    if (!dx && !dy) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    e.stopPropagation();
    panViewByPixels(-dx, -dy);
  };
  document.addEventListener("wheel", onWheelPan, { passive: false, capture: true });
}

setupCanvasPan();
canvas.style.touchAction = "none";

scene.add(new THREE.AmbientLight(0x9aa4b2, 0.55));
const key = new THREE.DirectionalLight(0xfff4e5, 1.15);
key.position.set(40, 30, 55);
scene.add(key);
const fill = new THREE.DirectionalLight(0x8fb7ff, 0.35);
fill.position.set(-50, -10, -30);
scene.add(fill);
scene.add(new THREE.HemisphereLight(0xced6e0, 0x1a1c20, 0.35));

const root = new THREE.Group();
scene.add(root);

const dnaGroup = new THREE.Group();
const mineralGroup = new THREE.Group();
const envGroup = new THREE.Group();
const shellGroup = new THREE.Group();
const hotspotGroup = new THREE.Group();
const guideGroup = new THREE.Group();
root.add(dnaGroup, mineralGroup, envGroup, shellGroup, hotspotGroup, guideGroup);

function catmull(points, samples) {
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
  return curve.getSpacedPoints(samples);
}

function ribbonGeometry(center, toward, width, thickness) {
  const n = center.length;
  const pos = [];
  const nrm = [];
  const idx = [];
  const hw = width / 2;
  const ht = thickness / 2;

  const frames = [];
  for (let i = 0; i < n; i++) {
    const p = center[i];
    const t = new THREE.Vector3();
    if (i < n - 1) t.subVectors(center[i + 1], p);
    else t.subVectors(p, center[i - 1]);
    t.normalize();
    let u = toward[i].clone();
    u.addScaledVector(t, -u.dot(t));
    if (u.lengthSq() < 1e-8) {
      u = Math.abs(t.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      u.cross(t);
    }
    u.normalize();
    const v = new THREE.Vector3().crossVectors(t, u).normalize();
    frames.push({ p, t, u, v });
  }

  for (let i = 0; i < n; i++) {
    const { p, u, v } = frames[i];
    const corners = [
      p.clone().addScaledVector(u, hw).addScaledVector(v, ht),
      p.clone().addScaledVector(u, -hw).addScaledVector(v, ht),
      p.clone().addScaledVector(u, -hw).addScaledVector(v, -ht),
      p.clone().addScaledVector(u, hw).addScaledVector(v, -ht),
    ];
    for (const c of corners) pos.push(c.x, c.y, c.z);
    for (let k = 0; k < 4; k++) nrm.push(u.x, u.y, u.z);
    if (i < n - 1) {
      const a = i * 4;
      const b = (i + 1) * 4;
      const quads = [
        [0, 1, 1, 0],
        [1, 2, 2, 1],
        [2, 3, 3, 2],
        [3, 0, 0, 3],
      ];
      for (const [a0, a1, b1, b0] of quads) {
        idx.push(a + a0, a + a1, b + b1);
        idx.push(a + a0, b + b1, b + b0);
      }
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function buildDNA() {
  MODEL.strands.forEach((strand) => {
    const c1 = strand.residues.map((r) => new THREE.Vector3(...r.C1));
    const ng = strand.residues.map((r) => new THREE.Vector3(...r.N));
    const p = strand.residues.map((r) => new THREE.Vector3(...r.P));
    const samples = 48;
    const cSmooth = catmull(c1, samples);
    const nSmooth = catmull(ng, samples);
    const toward = cSmooth.map((pt, i) => nSmooth[i].clone().sub(pt));
    const geom = ribbonGeometry(cSmooth, toward, 2.15, 0.42);
    const mat = new THREE.MeshPhysicalMaterial({
      color: STRAND_COLOR[strand.chain] || 0xdddddd,
      roughness: 0.35,
      metalness: 0.05,
      clearcoat: 0.25,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.name = "ribbon";
    dnaGroup.add(mesh);

    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3(p, false, "centripetal"), 40, 0.28, 8, false),
      new THREE.MeshPhysicalMaterial({ color: DNA_PINK.trace, roughness: 0.4 })
    );
    tube.name = "phosphate-trace";
    dnaGroup.add(tube);

    const pGeom = new THREE.SphereGeometry(0.85, 16, 12);
    const pMat = new THREE.MeshPhysicalMaterial({ color: DNA_PINK.phosphate, roughness: 0.3 });
    p.forEach((pt) => {
      const s = new THREE.Mesh(pGeom, pMat);
      s.position.copy(pt);
      s.name = "phosphate";
      dnaGroup.add(s);
    });
  });

  const rungMat = new THREE.MeshPhysicalMaterial({
    color: 0x8fa8c4,
    roughness: 0.45,
    transparent: true,
    opacity: 0.9,
  });
  MODEL.pairs.forEach((pair) => {
    const a = new THREE.Vector3(...pair.a);
    const b = new THREE.Vector3(...pair.b);
    const axis = new THREE.Vector3().subVectors(b, a);
    const len = axis.length();
    axis.normalize();
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const box = new THREE.Mesh(new THREE.BoxGeometry(len, 0.32, 3.3), rungMat);
    box.position.copy(mid);
    let side = new THREE.Vector3().crossVectors(axis, new THREE.Vector3(0, 1, 0));
    if (side.lengthSq() < 1e-8) side = new THREE.Vector3(0, 0, 1);
    side.normalize();
    const up = new THREE.Vector3().crossVectors(side, axis).normalize();
    box.setRotationFromMatrix(new THREE.Matrix4().makeBasis(axis, up, side));
    box.name = "pair";
    dnaGroup.add(box);
  });

  const seedMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.2,
    emissive: 0x335577,
    emissiveIntensity: 0.25,
  });
  const seedGeom = new THREE.OctahedronGeometry(1.35, 0);
  MODEL.seeds.forEach((xyz) => {
    const s = new THREE.Mesh(seedGeom, seedMat);
    s.position.set(...xyz);
    s.name = "seed";
    dnaGroup.add(s);
  });

  const oxalate = MODEL.oxalate || [];
  if (oxalate.length) {
    const pos = new Float32Array(oxalate.length * 6);
    oxalate.forEach((seg, i) => {
      pos[i * 6] = seg[0][0];
      pos[i * 6 + 1] = seg[0][1];
      pos[i * 6 + 2] = seg[0][2];
      pos[i * 6 + 3] = seg[1][0];
      pos[i * 6 + 4] = seg[1][1];
      pos[i * 6 + 5] = seg[1][2];
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const lines = new THREE.LineSegments(
      geom,
      new THREE.LineBasicMaterial({
        color: 0xc4b49a,
        transparent: true,
        opacity: 0.7,
      })
    );
    lines.name = "oxalate";
    dnaGroup.add(lines);
    oxalateLines = lines;
    oxalateRest = new Float32Array(pos);
  }
}

function buildEnvelopeGeometry(env) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(env.vertices.flat(), 3)
  );
  geom.setIndex(env.indices);
  geom.computeVertexNormals();
  return geom;
}

function buildEnvelopes() {
  const names = [...PHASES];
  if (MODEL.envelopes?.shell?.vertices?.length) names.push("shell");
  if (MODEL.envelopes?.nucleation?.vertices?.length) names.push("nucleation");
  names.forEach((name) => {
    const env = MODEL.envelopes[name];
    if (!env || !env.vertices.length) return;
    const geom = buildEnvelopeGeometry(env);
    if (name === "nucleation") {
      const mat = new THREE.MeshPhysicalMaterial({
        color: ACCENT_YELLOW.bright,
        transparent: true,
        opacity: 0.48,
        side: THREE.DoubleSide,
        depthWrite: false,
        roughness: 0.18,
        metalness: 0.0,
        emissive: ACCENT_YELLOW.emissive,
        emissiveIntensity: 1.35,
        toneMapped: false,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.userData.phase = name;
      mesh.renderOrder = 1;
      envGroup.add(mesh);
      return;
    }
    const mat = new THREE.MeshLambertMaterial({
      color: PHASE_COLOR[name] || PHASE_COLOR.intermediate,
      transparent: true,
      opacity: name === "shell" ? 0.16 : name === "amorphous" ? 0.32 : 0.2,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    if (name === "amorphous") {
      mat.emissive = new THREE.Color(PHASE_COLOR.amorphous);
      mat.emissiveIntensity = 0.55;
    }
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.phase = name;
    envGroup.add(mesh);
  });
}

let hotspotMesh = null;
let hotspotGlowMesh = null;
let hotspotIndices = [];
let hotspotClusterMeshes = [];

function dpColor(dp) {
  const t = Math.min(1, (dp || 0) / 32);
  return new THREE.Color().setHSL(0.08 + 0.55 * (1 - t), 0.75, 0.5);
}

function threeColorToCss(color) {
  return `#${color.getHexString()}`;
}

function gradientCssFromFn(fn, steps = 10) {
  const stops = [];
  for (let i = 0; i <= steps; i++) {
    const pct = (i / steps) * 100;
    stops.push(`${threeColorToCss(fn(i / steps))} ${pct}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

const COLOR_LEGEND = {
  distance: {
    title: "Distance from P",
    type: "gradient",
    gradient: () => gradientCssFromFn((t) => dpColor(t * 32)),
    low: "near P (0 Å)",
    high: "far (32 Å)",
  },
  phase: {
    title: "Phase",
    type: "items",
    items: [
      { color: PHASE_COLOR.amorphous, label: "Amorphous" },
      { color: PHASE_COLOR.intermediate, label: "Nucleation (phase)" },
      { color: PHASE_COLOR.crystalline, label: "Strong order" },
    ],
  },
  score: {
    title: "COM-net score",
    type: "gradient",
    gradient: () =>
      gradientCssFromFn((t) => {
        const c = new THREE.Color().setHSL(0.55 - 0.45 * t, 0.65, 0.45);
        return c;
      }),
    low: "low (0)",
    high: "high (0.45)",
  },
  comRegistry: {
    title: "COM pair correlation",
    type: "gradient",
    gradient: () =>
      gradientCssFromFn((t) => {
        const c = new THREE.Color().setHSL(0.13 - 0.1 * t, 0.9, 0.42 + 0.18 * t);
        return c;
      }),
    low: "weak (0)",
    high: "strong (0.28)",
  },
};

function updateColorLegend(colorMode) {
  const root = $("color-legend");
  if (!root) return;
  const spec = COLOR_LEGEND[colorMode];
  if (!spec) {
    root.hidden = true;
    return;
  }
  const titleEl = root.querySelector(".color-legend-title");
  const bodyEl = root.querySelector(".color-legend-body");
  if (!titleEl || !bodyEl) return;
  titleEl.textContent = spec.title;
  if (spec.type === "gradient") {
    bodyEl.innerHTML = `
      <div class="color-legend-gradient">
        <div class="gradient-bar" style="background: ${spec.gradient()}"></div>
        <div class="gradient-labels">
          <span>${spec.low}</span>
          <span>${spec.high}</span>
        </div>
      </div>`;
  } else {
    bodyEl.innerHTML = `
      <div class="color-legend-items">
        ${spec.items
          .map(
            ({ color, label }) =>
              `<span><i style="background:#${color.toString(16).padStart(6, "0")}"></i>${label}</span>`
          )
          .join("")}</div>`;
  }
  root.hidden = false;
}

function caKeep(i, dmin, dmax, rmax, slab) {
  return (
    ca.dP[i] >= dmin &&
    ca.dP[i] <= dmax &&
    ca.radial[i] <= rmax &&
    Math.abs(ca.y[i]) <= slab
  );
}

function hotspotColorFromDp(dp) {
  const t = Math.min(1, (dp || 0) / 32);
  // Bright lemon near P → warm gold far; stay luminous against pink DNA.
  return new THREE.Color().setHSL(0.13 - 0.07 * t, 1.0, 0.72 - 0.1 * t);
}

function hotspotGlowColorFromDp(dp) {
  const c = hotspotColorFromDp(dp);
  c.lerp(new THREE.Color(0xffffff), 0.22);
  return c;
}

function hotspotColor(i) {
  return hotspotColorFromDp(ca.dP[i]);
}

function setHotspotInstance(mesh, k, i, scale, on, colorFn = hotspotColor) {
  const dummy = new THREE.Object3D();
  dummy.position.set(ca.x[i], ca.y[i], ca.z[i]);
  dummy.scale.setScalar(on ? scale : 0.001);
  dummy.updateMatrix();
  mesh.setMatrixAt(k, dummy.matrix);
  mesh.setColorAt(k, colorFn(i));
}

function updateHotspotMarkers(dmin, dmax, rmax, slab) {
  if (!hotspotMesh || !hotspotIndices.length) return;
  for (let k = 0; k < hotspotIndices.length; k++) {
    const i = hotspotIndices[k];
    const on = caKeep(i, dmin, dmax, rmax, slab);
    setHotspotInstance(hotspotMesh, k, i, 1.45, on);
    if (hotspotGlowMesh) {
      setHotspotInstance(hotspotGlowMesh, k, i, 2.35, on, (idx) =>
        hotspotGlowColorFromDp(ca.dP[idx])
      );
    }
  }
  hotspotMesh.instanceMatrix.needsUpdate = true;
  if (hotspotMesh.instanceColor) hotspotMesh.instanceColor.needsUpdate = true;
  if (hotspotGlowMesh) {
    hotspotGlowMesh.instanceMatrix.needsUpdate = true;
    if (hotspotGlowMesh.instanceColor) hotspotGlowMesh.instanceColor.needsUpdate = true;
  }
}

function buildHotspotClusterRings() {
  hotspotClusterMeshes.forEach((mesh) => {
    hotspotGroup.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  });
  hotspotClusterMeshes = [];
}

function buildHotspotMarkers() {
  hotspotIndices = [];
  if (!ca.hotspot) return;
  const pts = [];
  for (let i = 0; i < nCa; i++) {
    if (ca.hotspot[i]) pts.push(i);
  }
  if (!pts.length) return;
  hotspotIndices = pts;
  const geom = new THREE.SphereGeometry(1, 14, 12);
  const mat = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.18,
    metalness: 0.0,
    emissive: ACCENT_YELLOW.emissive,
    emissiveIntensity: 0.95,
    transparent: true,
    opacity: 0.98,
    toneMapped: false,
  });
  hotspotMesh = new THREE.InstancedMesh(geom, mat, pts.length);
  hotspotMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(pts.length * 3),
    3
  );
  const glowMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  });
  hotspotGlowMesh = new THREE.InstancedMesh(geom, glowMat, pts.length);
  hotspotGlowMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(pts.length * 3),
    3
  );
  hotspotGlowMesh.renderOrder = 1;
  for (let k = 0; k < pts.length; k++) {
    const i = pts[k];
    setHotspotInstance(hotspotMesh, k, i, 1.45, true);
    const dummy = new THREE.Object3D();
    dummy.position.set(ca.x[i], ca.y[i], ca.z[i]);
    dummy.scale.setScalar(2.35);
    dummy.updateMatrix();
    hotspotGlowMesh.setMatrixAt(k, dummy.matrix);
    hotspotGlowMesh.setColorAt(k, hotspotGlowColorFromDp(ca.dP[i]));
  }
  hotspotMesh.instanceMatrix.needsUpdate = true;
  hotspotMesh.instanceColor.needsUpdate = true;
  hotspotGlowMesh.instanceMatrix.needsUpdate = true;
  hotspotGlowMesh.instanceColor.needsUpdate = true;
  hotspotGroup.add(hotspotGlowMesh);
  hotspotGroup.add(hotspotMesh);
  buildHotspotClusterRings();
}

function buildShells() {
  const mat = new THREE.MeshBasicMaterial({
    color: 0x8899aa,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
    depthWrite: false,
    wireframe: true,
  });
  if ((MODEL.cutKind || "spheres") === "spheres") {
    const r = MODEL.seedRadius || 30;
    const segs = r <= 10 ? 20 : 28;
    const g = new THREE.SphereGeometry(r, segs, Math.max(10, segs - 8));
    MODEL.seeds.forEach((xyz) => {
      const mesh = new THREE.Mesh(g, mat);
      mesh.position.set(...xyz);
      shellGroup.add(mesh);
    });
  } else {
    const { rDna, rCoat, dnaZmin, dnaZmax } = MODEL.helix;
    const h = dnaZmax - dnaZmin;
    const y = 0.5 * (dnaZmin + dnaZmax);
    [rDna, rCoat].forEach((r, i) => {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, h, 48, 1, true),
        mat
      );
      mesh.position.y = y;
      mesh.material = mat.clone();
      mesh.material.opacity = i === 0 ? 0.22 : 0.12;
      shellGroup.add(mesh);
    });
  }
  shellGroup.visible = false;
}

function buildRadialGuides() {
  clearGroup(guideGroup);
  const hx = MODEL.helix || {};
  const rP = hx.rPhosphate || 9.0;
  const comA = hx.comA || 6.29;
  const y0 = hx.dnaZmin ?? -30;
  const y1 = hx.dnaZmax ?? 30;
  const h = Math.max(12, y1 - y0);
  const y = 0.5 * (y0 + y1);
  const rings = [
    { r: rP, color: DNA_PINK.trace, opacity: 0.5 },
    { r: rP + comA, color: 0x1b9e77, opacity: 0.38 },
    { r: rP + 2 * comA, color: 0x88c9b0, opacity: 0.22 },
  ];
  rings.forEach(({ r, color, opacity }) => {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(r, r, h, 64, 1, true),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
        wireframe: true,
      })
    );
    mesh.position.y = y;
    guideGroup.add(mesh);
  });
}

function syncUnrollCaption(shown, total, dmin, dmax, rmax, slab) {
  const el = $("unroll-caption");
  if (!el) return;
  const w = MODEL.nucleationWhere;
  const geom = MODEL.geometry || MODEL.source || "model";
  const filter =
    `filters: d(P) ${dmin}–${dmax} Å, peel ≤${rmax} Å` +
    (slab >= FILTER_SLAB_FULL ? ", full length" : `, |axis| ≤ ${slab} Å`);
  const exported = window.DNA_CAOX_EXPORTED_AT;
  const parts = [
    `${geom}: ${shown} of ${total} hotspot Ca on unroll (${filter}).` +
      (exported ? ` Data ${exported}.` : ""),
    "Pink dots = phosphates; bright gold dots = symmetry hotspots (lemon = near P, gold = far).",
  ];
  if (w && w.nPWithHot8 != null && w.nP != null) {
    parts.push(
      `P with hotspot ≤8 Å: ${w.nPWithHot8}/${w.nP}.` +
        (w.medianDpHot != null
          ? ` Median hotspot d(P) ${w.medianDpHot} Å vs ${w.medianDpAll} Å for all Ca.`
          : "")
    );
  } else if (!total) {
    parts.push("No COM pair-order hotspots in this model.");
  }
  el.textContent = parts.join(" ");
}

function drawUnroll(dmin, dmax, rmax, slab) {
  const el = $("unroll");
  if (!el) return;
  const ctx = el.getContext("2d");
  const w = el.width;
  const h = el.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c0e12";
  ctx.fillRect(0, 0, w, h);
  const yMin = MODEL.helix.dnaZmin;
  const yMax = MODEL.helix.dnaZmax;
  const span = yMax - yMin || 1;
  const padL = 22;
  const padR = 8;
  const padT = 18;
  const padB = 16;
  const xOf = (phi) => padL + ((phi + 180) / 360) * (w - padL - padR);
  const yOf = (y) => padT + (1 - (y - yMin) / span) * (h - padT - padB);
  ctx.strokeStyle = "#2a2e36";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, h - padB);
  ctx.lineTo(w - padR, h - padB);
  ctx.stroke();
  (MODEL.strands || []).forEach((s) => {
    (s.residues || []).forEach((r) => {
      const p = r.P;
      const phi = (Math.atan2(p[2], p[0]) * 180) / Math.PI;
      ctx.fillStyle = "#ff9ec8";
      ctx.beginPath();
      ctx.arc(xOf(phi), yOf(p[1]), 2.3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  let shown = 0;
  let total = 0;
  if (ca.hotspot) {
    for (let i = 0; i < nCa; i++) {
      if (!ca.hotspot[i]) continue;
      total += 1;
      if (!caKeep(i, dmin, dmax, rmax, slab)) continue;
      shown += 1;
      const phi =
        ca.phi != null ? ca.phi[i] : (Math.atan2(ca.z[i], ca.x[i]) * 180) / Math.PI;
      const col = hotspotColor(i);
      ctx.fillStyle = `rgb(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)})`;
      ctx.beginPath();
      ctx.arc(xOf(phi), yOf(ca.y[i]), 2.7, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#c8c4bc";
  ctx.font = "bold 9px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(`${shown}/${total} hotspots`, padL, padT - 1);

  ctx.fillStyle = "#8d8b82";
  ctx.font = "9px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("−180°", padL, h - 4);
  ctx.fillText("+180°", w - padR - 28, h - 4);
  ctx.save();
  ctx.translate(10, h * 0.55);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("axis", 0, 0);
  ctx.restore();

  const legW = 68;
  const legH = 7;
  const legX = w - padR - legW;
  const legY = padT;
  const grad = ctx.createLinearGradient(legX, legY, legX + legW, legY);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const col = hotspotColorFromDp(t * 32);
    grad.addColorStop(
      t,
      `rgb(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)})`
    );
  }
  ctx.fillStyle = grad;
  ctx.fillRect(legX, legY, legW, legH);
  ctx.strokeStyle = "#2a2e36";
  ctx.lineWidth = 1;
  ctx.strokeRect(legX, legY, legW, legH);
  ctx.fillStyle = "#8d8b82";
  ctx.font = "8px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("near P", legX, legY + legH + 9);
  ctx.fillText("far", legX + legW - 16, legY + legH + 9);
  syncUnrollCaption(shown, total, dmin, dmax, rmax, slab);
  return { shown, total };
}

const sprite = (() => {
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 64;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.55)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
})();

function colorFor(i, mode) {
  if (mode === "distance") {
    return dpColor(ca.dP[i]);
  }
  if (mode === "score") {
    const t = Math.min(1, ca.score[i] / 0.45);
    return new THREE.Color().setHSL(0.55 - 0.45 * t, 0.65, 0.45);
  }
  if (mode === "comRegistry") {
    const t = Math.min(1, (ca.comRegistry?.[i] || 0) / 0.28);
    return new THREE.Color().setHSL(0.13 - 0.1 * t, 0.9, 0.42 + 0.18 * t);
  }
  return new THREE.Color(PHASE_COLOR[PHASES[ca.phase[i]]]);
}

function makePackedPoints(size, opacity, additive) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nCa * 3), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nCa * 3), 3));
  geom.setDrawRange(0, 0);
  const mat = new THREE.PointsMaterial({
    size,
    map: additive ? sprite : null,
    vertexColors: true,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    sizeAttenuation: true,
  });
  const pts = new THREE.Points(geom, mat);
  pts.frustumCulled = false;
  mineralGroup.add(pts);
  return pts;
}

let cloudByPhase = [];
let pointsByPhase = [];
let caSpheres = null;
const dummy = new THREE.Object3D();

function rebuildMineral() {
  while (mineralGroup.children.length) {
    const obj = mineralGroup.children[0];
    mineralGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  cloudByPhase = [
    makePackedPoints(12.0, 0.38, true),
    makePackedPoints(7.2, 0.48, true),
    makePackedPoints(4.6, 0.72, true),
  ];
  pointsByPhase = [
    makePackedPoints(2.4, 0.95, false),
    makePackedPoints(2.0, 0.95, false),
    makePackedPoints(2.8, 1.0, false),
  ];
  const sphereGeom = new THREE.SphereGeometry(1, 12, 8);
  const sphereMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.45,
    metalness: 0.05,
  });
  caSpheres = new THREE.InstancedMesh(sphereGeom, sphereMat, nCa);
  caSpheres.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  caSpheres.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(nCa * 3), 3);
  mineralGroup.add(caSpheres);
  buildWater();
}

function buildWater() {
  waterMesh = null;
  const w = MODEL.water;
  if (!w || !w.x || !w.x.length) return;
  const n = w.x.length;
  const geom = new THREE.SphereGeometry(0.68, 8, 6);
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x6a9cc9,
    roughness: 0.4,
    transparent: true,
    opacity: 0.5,
  });
  waterMesh = new THREE.InstancedMesh(geom, mat, n);
  waterMesh.name = "water";
  const d = new THREE.Object3D();
  for (let i = 0; i < n; i++) {
    d.position.set(w.x[i], w.y[i], w.z[i]);
    d.updateMatrix();
    waterMesh.setMatrixAt(i, d.matrix);
  }
  waterMesh.instanceMatrix.needsUpdate = true;
  mineralGroup.add(waterMesh);
}

function applyWater(rmax, slab) {
  if (!waterMesh || !MODEL.water?.x?.length) return 0;
  const w = MODEL.water;
  const n = w.x.length;
  let shown = 0;
  for (let i = 0; i < n; i++) {
    const r = Math.hypot(w.x[i], w.z[i]);
    const keep = r <= rmax && Math.abs(w.y[i]) <= slab;
    if (!keep) {
      dummy.position.set(0, 1e6, 0);
      dummy.scale.setScalar(0.001);
    } else {
      dummy.position.set(w.x[i], w.y[i], w.z[i]);
      dummy.scale.setScalar(1);
      shown += 1;
    }
    dummy.updateMatrix();
    waterMesh.setMatrixAt(i, dummy.matrix);
  }
  waterMesh.instanceMatrix.needsUpdate = true;
  return shown;
}

function packPoints(obj, packed) {
  const pos = obj.geometry.getAttribute("position");
  const col = obj.geometry.getAttribute("color");
  for (let k = 0; k < packed.length; k++) {
    const p = packed[k];
    pos.setXYZ(k, p.x, p.y, p.z);
    col.setXYZ(k, p.r, p.g, p.b);
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
  obj.geometry.setDrawRange(0, packed.length);
}

function applyCa(style, colorMode, phaseOn, dmin, dmax, rmax, slab) {
  let shown = 0;
  const shownPh = [0, 0, 0];
  const packed = [[], [], []];

  for (let i = 0; i < nCa; i++) {
    const ph = ca.phase[i];
    const keep =
      phaseOn[ph] &&
      ca.dP[i] >= dmin &&
      ca.dP[i] <= dmax &&
      ca.radial[i] <= rmax &&
      Math.abs(ca.y[i]) <= slab;
    const col = colorFor(i, colorMode);
    if (!keep) {
      dummy.position.set(0, 1e6, 0);
      dummy.scale.setScalar(0.001);
      dummy.updateMatrix();
      caSpheres.setMatrixAt(i, dummy.matrix);
      caSpheres.setColorAt(i, col);
      continue;
    }
    dummy.position.set(ca.x[i], ca.y[i], ca.z[i]);
    const hot = ca.hotspot && ca.hotspot[i];
    dummy.scale.setScalar(hot ? 1.55 : ph === 2 ? 1.35 : ph === 0 ? 0.85 : 1.0);
    dummy.updateMatrix();
    caSpheres.setMatrixAt(i, dummy.matrix);
    caSpheres.setColorAt(i, col);
    packed[ph].push({
      x: ca.x[i],
      y: ca.y[i],
      z: ca.z[i],
      r: col.r,
      g: col.g,
      b: col.b,
    });
    shown += 1;
    shownPh[ph] += 1;
  }
  caSpheres.instanceMatrix.needsUpdate = true;
  caSpheres.instanceColor.needsUpdate = true;

  for (let ph = 0; ph < 3; ph++) {
    packPoints(cloudByPhase[ph], packed[ph]);
    packPoints(pointsByPhase[ph], packed[ph]);
    cloudByPhase[ph].visible = style === "cloud";
    pointsByPhase[ph].visible = style === "points";
  }
  caSpheres.visible = style === "spheres";
  return { shown, shownPh };
}

function setView(kind) {
  const span = Math.max(
    MODEL.helix.rCoat || 42,
    (MODEL.helix.zmax - MODEL.helix.zmin) * 0.55,
    MODEL.seedRadius || 30
  );
  const r = span + 20;
  if (kind === "end") camera.position.set(0, r * 1.6, 0.01);
  else if (kind === "seed") camera.position.set(r * 1.35, 8, 18);
  else camera.position.set(18, 8, r * 1.15);
  viewPan.set(0, 0, 0);
  syncOrbitTarget();
  controls.update();
}

function clearGroup(group) {
  while (group.children.length) {
    const obj = group.children[0];
    group.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
}

function updateOxalateFromTraj(segments) {
  if (!oxalateLines || !oxalateRest) return;
  const pos = oxalateLines.geometry.attributes.position.array;
  const nSeg = pos.length / 6;

  if (segments && segments.length) {
    const n = Math.min(segments.length, nSeg);
    for (let i = 0; i < n; i++) {
      const seg = segments[i];
      pos[i * 6] = seg[0][0];
      pos[i * 6 + 1] = seg[0][1];
      pos[i * 6 + 2] = seg[0][2];
      pos[i * 6 + 3] = seg[1][0];
      pos[i * 6 + 4] = seg[1][1];
      pos[i * 6 + 5] = seg[1][2];
    }
    for (let i = n; i < nSeg; i++) {
      for (let c = 0; c < 6; c++) pos[i * 6 + c] = oxalateRest[i * 6 + c];
    }
  } else if (oxalateUnitOffsets.length) {
    pos.set(oxalateRest);
    for (let u = 0; u < oxalateUnitOffsets.length && u < nCa; u++) {
      const dx = ca.x[u] - caRest.x[u];
      const dy = ca.y[u] - caRest.y[u];
      const dz = ca.z[u] - caRest.z[u];
      const i0 = oxalateUnitOffsets[u];
      const i1 =
        u + 1 < oxalateUnitOffsets.length ? oxalateUnitOffsets[u + 1] : nSeg;
      for (let si = i0; si < i1; si++) {
        for (let c = 0; c < 6; c += 3) {
          const ri = si * 6 + c;
          pos[ri] = oxalateRest[ri] + dx;
          pos[ri + 1] = oxalateRest[ri + 1] + dy;
          pos[ri + 2] = oxalateRest[ri + 2] + dz;
        }
      }
    }
  }
  oxalateLines.geometry.attributes.position.needsUpdate = true;
}

function trajDisplacementStats(frames) {
  if (!frames?.length) return 0;
  const f0 = frames[0].ca;
  let maxDisp = 0;
  for (const f of frames) {
    const n = Math.min(f0.x.length, f.ca.x.length);
    for (let i = 0; i < n; i++) {
      const dx = f.ca.x[i] - f0.x[i];
      const dy = f.ca.y[i] - f0.y[i];
      const dz = f.ca.z[i] - f0.z[i];
      maxDisp = Math.max(maxDisp, Math.hypot(dx, dy, dz));
    }
  }
  return maxDisp;
}

function captureTrajOxalate(segments) {
  if (!segments?.length) return null;
  const pos = new Float32Array(segments.length * 6);
  segments.forEach((seg, i) => {
    pos[i * 6] = seg[0][0];
    pos[i * 6 + 1] = seg[0][1];
    pos[i * 6 + 2] = seg[0][2];
    pos[i * 6 + 3] = seg[1][0];
    pos[i * 6 + 4] = seg[1][1];
    pos[i * 6 + 5] = seg[1][2];
  });
  return pos;
}

function applyTrajOxalate(segments) {
  if (!oxalateLines || !oxalateRest) return;
  const pos = oxalateLines.geometry.attributes.position.array;
  const nSeg = pos.length / 6;
  if (segments?.length && trajFrame0Oxalate && trajAmp !== 1) {
    const n = Math.min(segments.length, nSeg, trajFrame0Oxalate.length / 6);
    for (let i = 0; i < n; i++) {
      const seg = segments[i];
      for (let c = 0; c < 2; c++) {
        for (let k = 0; k < 3; k++) {
          const ri = i * 6 + c * 3 + k;
          const base = trajFrame0Oxalate[ri];
          const target = seg[c][k];
          pos[ri] = base + trajAmp * (target - base);
        }
      }
    }
  } else {
    updateOxalateFromTraj(segments);
    return;
  }
  oxalateLines.geometry.attributes.position.needsUpdate = true;
}

function trajStatusText(url) {
  if (!trajData) return "No trajectory loaded.";
  const name = url.split("/").pop();
  const n = trajData.frames.length;
  let line = `${n} frames · ${name}`;
  if (trajMaxDisp < 0.5) {
    line += ` · FIRE motion ${trajMaxDisp.toFixed(2)} Å (×${trajAmp.toFixed(0)} for playback)`;
  } else {
    line += ` · max Ca move ${trajMaxDisp.toFixed(1)} Å`;
  }
  return line;
}

const SESSION_KEY = "dnaCaoxViewer.session.v1";
const SESSION_CHECKBOX_IDS = [
  "dna-ribbons",
  "dna-pairs",
  "dna-p",
  "dna-seeds",
  "oxalate",
  "water",
  "envelopes",
  "ph-shell",
  "ph-amorphous",
  "ph-intermediate",
  "ph-crystalline",
  "ph-nucleation",
  "nucleation-hotspots",
  "radial-guides",
  "shells",
  "auto-rotate",
  "g16-filter-visible",
  "g16-hotspots-only",
  "g16-include-dna",
  "g16-include-oxalate",
  "g16-include-water",
];
const SESSION_VALUE_IDS = [
  "dmin",
  "dmax",
  "rmax",
  "slab",
  "env-opacity",
  "g16-preset",
  "g16-route",
  "g16-charge",
  "g16-mult",
  "g16-mem",
  "g16-nproc",
  "g16-chk",
  "g16-radius",
  "g16-max-atoms",
];

function sessionStatus(msg) {
  const el = $("session-status");
  if (el) el.textContent = msg;
}

function readSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    return data && typeof data === "object" ? data : null;
  } catch (_) {
    return null;
  }
}

function collectSession() {
  const checks = {};
  SESSION_CHECKBOX_IDS.forEach((id) => {
    const el = $(id);
    if (el) checks[id] = !!el.checked;
  });
  const values = {};
  SESSION_VALUE_IDS.forEach((id) => {
    const el = $(id);
    if (el) values[id] = el.value;
  });
  const style = document.querySelector('input[name="style"]:checked');
  const color = document.querySelector('input[name="color"]:checked');
  return {
    v: 1,
    saved: new Date().toISOString(),
    geometry: MODEL.geometry || DEFAULT_GEOM,
    checks,
    values,
    style: style ? style.value : "cloud",
    color: color ? color.value : "distance",
    g16ChkEdited: $("g16-chk")?.dataset.userEdited === "1",
    trajFrame: trajFrame || 0,
    camera: {
      pos: [camera.position.x, camera.position.y, camera.position.z],
      target: [controls.target.x, controls.target.y, controls.target.z],
      pan: [viewPan.x, viewPan.y, viewPan.z],
    },
  };
}

function writeSession() {
  if (sessionRestoring) return;
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(collectSession()));
  } catch (_) {
    /* quota / private mode */
  }
}

function scheduleSessionSave() {
  if (sessionRestoring) return;
  clearTimeout(sessionSaveTimer);
  sessionSaveTimer = setTimeout(writeSession, 500);
}

function applySessionFields(data) {
  if (!data) return;
  if (data.checks) {
    SESSION_CHECKBOX_IDS.forEach((id) => {
      const el = $(id);
      if (el && typeof data.checks[id] === "boolean") el.checked = data.checks[id];
    });
  }
  const preset = data.values?.["g16-preset"];
  if (preset && $("g16-preset")) {
    $("g16-preset").value = preset;
    applyG16Preset(preset);
  }
  if (data.values) {
    SESSION_VALUE_IDS.forEach((id) => {
      const el = $(id);
      if (el && data.values[id] != null && data.values[id] !== "") el.value = data.values[id];
    });
  }
  if (data.style) {
    const el = document.querySelector(`input[name="style"][value="${data.style}"]`);
    if (el) el.checked = true;
  }
  if (data.color) {
    const el = document.querySelector(`input[name="color"][value="${data.color}"]`);
    if (el) el.checked = true;
  }
  if (data.g16ChkEdited && $("g16-chk")) $("g16-chk").dataset.userEdited = "1";
}

function restoreCamera(cam) {
  if (!cam?.pos) return;
  camera.position.set(cam.pos[0], cam.pos[1], cam.pos[2]);
  if (cam.target) controls.target.set(cam.target[0], cam.target[1], cam.target[2]);
  if (cam.pan) viewPan.set(cam.pan[0], cam.pan[1], cam.pan[2]);
  syncOrbitTarget();
  controls.update();
}

function applySession(data, { downloaded } = {}) {
  if (!data || data.v !== 1) {
    sessionStatus("Could not load session (unknown format).");
    return;
  }
  sessionRestoring = true;
  pendingTrajFrame = data.trajFrame ?? 0;
  const geom = data.geometry && MODELS[data.geometry] ? data.geometry : DEFAULT_GEOM;
  loadGeometry(geom, { preserveUi: true });
  applySessionFields(data);
  restoreCamera(data.camera);
  applyUi();
  sessionRestoring = false;
  writeSession();
  const when = data.saved ? new Date(data.saved).toLocaleString() : "now";
  sessionStatus(
    downloaded ? `Loaded session file from ${when}.` : `Restored session from ${when}.`
  );
}

function saveSessionNow({ download } = {}) {
  const data = collectSession();
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(data));
  } catch (_) {
    sessionStatus("Could not write browser session (storage blocked).");
    if (!download) return;
  }
  if (download) {
    downloadBlob(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
      `dna-caox-session-${data.geometry || "viewer"}.json`
    );
    sessionStatus(`Saved session file (${data.geometry}).`);
    return;
  }
  sessionStatus(`Saved in this browser (${new Date(data.saved).toLocaleTimeString()}).`);
}

function resetSession() {
  localStorage.removeItem(SESSION_KEY);
  sessionStatus("Session cleared. Reloading defaults…");
  window.location.reload();
}

function wireSession() {
  $("session-save")?.addEventListener("click", () => saveSessionNow());
  $("session-download")?.addEventListener("click", () => saveSessionNow({ download: true }));
  $("session-reset")?.addEventListener("click", () => resetSession());
  const fileEl = $("session-file");
  $("session-load")?.addEventListener("click", () => fileEl?.click());
  fileEl?.addEventListener("change", async () => {
    const file = fileEl.files?.[0];
    fileEl.value = "";
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      applySession(data, { downloaded: true });
    } catch (err) {
      sessionStatus(`Load failed: ${err.message || err}`);
    }
  });
  controls.addEventListener("end", scheduleSessionSave);
}

function loadGeometry(name, opts = {}) {
  if (!MODELS[name]) return;
  stopTraj();
  MODEL = MODELS[name];
  ca = MODEL.ca;
  nCa = ca.x.length;
  caRest.x = [...ca.x];
  caRest.y = [...ca.y];
  caRest.z = [...ca.z];
  oxalateLines = null;
  oxalateRest = null;
  oxalateUnitOffsets = MODEL.oxalateUnitOffsets || [];
  clearGroup(dnaGroup);
  clearGroup(envGroup);
  clearGroup(shellGroup);
  clearGroup(hotspotGroup);
  clearGroup(guideGroup);
  viewPan.set(0, 0, 0);
  hotspotMesh = null;
  hotspotGlowMesh = null;
  hotspotIndices = [];
  hotspotClusterMeshes = [];
  buildDNA();
  buildEnvelopes();
  buildShells();
  buildRadialGuides();
  buildHotspotMarkers();
  rebuildMineral();
  syncGeomSelect(name);
  syncLabels();
  if (opts.preserveUi) {
    applyUi();
  } else if (MODELS[name]) {
    setCoatView();
  } else {
    applyUi();
    setView("side");
  }
  loadTrajectory(MODEL.traj);
}

function modelOptionLabel(key, model) {
  return model?.title || key;
}

const MODEL_ORDER = [
  "sphere",
  "slab",
  "allp",
  "local10",
  "local",
  "altp",
  "gel",
  "shell15",
  "gel_altp_geom",
  "shell_lattice",
  "shell_lattice_seeded",
  "templating_gel",
  "templating_gel_thick",
  "templating_gel_10shell",
  "templating_gel_15shell",
  "templating_nodna",
];

function populateGeomSelect() {
  const sel = $("geom-select");
  if (!sel) return;
  const keys = [
    ...MODEL_ORDER.filter((k) => MODELS[k]),
    ...Object.keys(MODELS)
      .filter((k) => !MODEL_ORDER.includes(k))
      .sort((a, b) =>
        String(MODELS[a]?.title || a).localeCompare(String(MODELS[b]?.title || b))
      ),
  ];
  sel.innerHTML = keys
    .map((key) => {
      const label = modelOptionLabel(key, MODELS[key]);
      return `<option value="${key}">${label}</option>`;
    })
    .join("");
  syncGeomSelect(MODEL.geometry || DEFAULT_GEOM);
}

function syncGeomSelect(name) {
  const sel = $("geom-select");
  if (!sel) return;
  if (name && MODELS[name]) sel.value = name;
}

async function loadTrajectory(url) {
  trajData = null;
  trajFrame = 0;
  trajFrame0Ca = null;
  trajFrame0Oxalate = null;
  trajAmp = 1;
  trajMaxDisp = 0;
  const section = $("traj-section");
  const status = $("traj-status");
  if (!url) {
    if (section) section.hidden = true;
    if (status) status.textContent = "No trajectory for this model.";
    return;
  }
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status}`);
    trajData = await res.json();
    const slider = $("traj-step");
    const maxF = Math.max(0, trajData.frames.length - 1);
    if (slider) {
      slider.min = 0;
      slider.max = String(maxF);
      slider.value = "0";
    }
    if (section) section.hidden = false;
    const f0 = trajData.frames[0]?.ca;
    if (f0?.x?.length) {
      trajFrame0Ca = { x: [...f0.x], y: [...f0.y], z: [...f0.z] };
      trajFrame0Oxalate = captureTrajOxalate(trajData.frames[0].oxalate);
      trajMaxDisp = trajDisplacementStats(trajData.frames);
      if (trajMaxDisp < 0.5) {
        trajAmp = Math.min(40, 1.2 / Math.max(trajMaxDisp, 1e-4));
      }
    }
    if (status) status.textContent = trajStatusText(url);
    const start =
      pendingTrajFrame != null ? Math.max(0, Math.min(maxF, Number(pendingTrajFrame))) : 0;
    pendingTrajFrame = null;
    applyTrajFrame(start);
  } catch (err) {
    if (section) section.hidden = false;
    if (status) status.textContent = `Trajectory unavailable: ${err.message}`;
  }
}

function applyTrajFrame(idx) {
  if (!trajData || !trajData.frames[idx]) return;
  const f = trajData.frames[idx];
  const n = Math.min(nCa, f.ca.x.length);
  for (let i = 0; i < n; i++) {
    if (trajFrame0Ca) {
      ca.x[i] = trajFrame0Ca.x[i] + trajAmp * (f.ca.x[i] - trajFrame0Ca.x[i]);
      ca.y[i] = trajFrame0Ca.y[i] + trajAmp * (f.ca.y[i] - trajFrame0Ca.y[i]);
      ca.z[i] = trajFrame0Ca.z[i] + trajAmp * (f.ca.z[i] - trajFrame0Ca.z[i]);
    } else {
      ca.x[i] = f.ca.x[i];
      ca.y[i] = f.ca.y[i];
      ca.z[i] = f.ca.z[i];
    }
  }
  if (f.oxalate) {
    applyTrajOxalate(f.oxalate);
  } else {
    updateOxalateFromTraj(null);
  }
  trajFrame = idx;
  const slider = $("traj-step");
  if (slider) slider.value = String(idx);
  const label = $("traj-step-v");
  if (label) {
    const phase = f.phase ? ` [${f.phase}]` : "";
    const e = f.energy != null ? `  E=${f.energy.toExponential(2)}` : "";
    label.textContent = `${f.step}${phase}${e}`;
  }
  applyUi();
}

function stopTraj() {
  trajPlaying = false;
  if (trajTimer) {
    clearInterval(trajTimer);
    trajTimer = null;
  }
  const btn = $("traj-play");
  if (btn) btn.textContent = "Play";
}

function toggleTrajPlay() {
  if (!trajData) return;
  if (trajPlaying) {
    stopTraj();
    return;
  }
  trajPlaying = true;
  const btn = $("traj-play");
  if (btn) btn.textContent = "Pause";
  trajTimer = setInterval(() => {
    const next = (trajFrame + 1) % trajData.frames.length;
    applyTrajFrame(next);
  }, 180);
}

function resetTraj() {
  stopTraj();
  for (let i = 0; i < nCa; i++) {
    ca.x[i] = caRest.x[i];
    ca.y[i] = caRest.y[i];
    ca.z[i] = caRest.z[i];
  }
  if (oxalateLines && oxalateRest) {
    oxalateLines.geometry.attributes.position.array.set(oxalateRest);
    oxalateLines.geometry.attributes.position.needsUpdate = true;
  }
  if (trajData) applyTrajFrame(0);
  else rebuildMineral();
}

function $(id) {
  return document.getElementById(id);
}

function syncLabels() {
  $("source-label").textContent = MODEL.title || MODEL.source.replace(".pdb", "");
  $("cut-label").textContent = MODEL.cut || "";
  PHASES.forEach((name) => {
    const el = $(`n-${name}`);
    if (el) el.textContent = `(${MODEL.counts[name]})`;
  });
  const nh = $("n-nucleation");
  if (nh && MODEL.counts.nucleationHotspots != null) {
    nh.textContent = `(${MODEL.counts.nucleationHotspots})`;
  }
  syncHotspotInfo();
}

function syncHotspotInfo() {
  const el = $("hotspot-info");
  const whereEl = $("where-answers");
  const w = MODEL.nucleationWhere;
  if (whereEl) {
    const geom = MODEL.geometry || MODEL.source || "";
    const header = geom ? `[${geom}] ` : "";
    if (w && w.answers && w.answers.length) {
      whereEl.textContent = header + w.answers.map((a, i) => `${i + 1}) ${a}`).join("\n\n");
    } else {
      whereEl.textContent =
        header +
        "Nucleation hotspots vs local COM symmetry — not occupancy blobs.";
    }
  }
  if (!el) return;
  const n = MODEL.counts?.nucleationHotspots || 0;
  const nIn = MODEL.counts?.insidePhosphate;
  if (!n) {
    el.textContent =
      "No COM pair-order hotspots in this cut (need neighbors at both ~3.84 Å and ~6.29 Å).";
    return;
  }
  const lines = [
    `${n} P-tethered hotspot Ca in ${MODEL.counts?.nucleationClusters ?? "?"} clusters (d(P)<12 Å; bright gold by distance from P).`,
    w
      ? w.medianPairCorrHot != null
        ? `median pair-corr ${w.medianPairCorrHot} (hotspots) vs ${w.medianPairCorrAll} (all); d(P) ${w.medianDpHot} vs ${w.medianDpAll} Å.`
        : `median d(P) ${w.medianDpHot} Å vs ${w.medianDpAll} Å for all Ca.`
      : "",
    nIn != null ? `${nIn} Ca packed inside the phosphate cylinder (grooves) — occupancy cloud, not nucleation.` : "",
  ].filter(Boolean);
  const nShell = MODEL.counts?.nucleationHotspotsShell || 0;
  if (nShell) {
    lines.push(
      `${nShell} outer-coat Ca with COM-like neighbors (d(P)≥12 Å) are not shown — thick gel packing, not helix-tethered nucleation.`
    );
  }
  el.textContent = lines.join("\n");
}

function applyUi() {
  const phaseOn = PHASES.map((name) => $(`ph-${name}`).checked);
  const style = document.querySelector("input[name=style]:checked").value;
  const colorMode = document.querySelector("input[name=color]:checked").value;
  const dmin = Number($("dmin").value);
  const dmax = Number($("dmax").value);
  const rmax = Number($("rmax").value);
  const slab = Number($("slab").value);
  $("dmin-v").textContent = `${dmin} Å`;
  $("dmax-v").textContent = `${dmax} Å`;
  $("rmax-v").textContent = `${rmax} Å`;
  $("slab-v").textContent = slab >= FILTER_SLAB_FULL ? "full" : `±${slab} Å`;

  dnaGroup.traverse((obj) => {
    if (obj.name === "ribbon") obj.visible = $("dna-ribbons").checked;
    if (obj.name === "pair") obj.visible = $("dna-pairs").checked;
    if (obj.name === "phosphate" || obj.name === "phosphate-trace") {
      obj.visible = $("dna-p").checked;
    }
    if (obj.name === "seed") obj.visible = $("dna-seeds").checked;
    if (obj.name === "oxalate") obj.visible = $("oxalate").checked;
  });
  let nWaterShown = 0;
  if (waterMesh) {
    const wcb = $("water");
    waterMesh.visible = !wcb || wcb.checked;
    nWaterShown = applyWater(rmax, slab);
  }

  envGroup.visible = $("envelopes").checked;
  const op = Number($("env-opacity").value) / 100;
  envGroup.children.forEach((mesh) => {
    const ph = mesh.userData.phase;
    let cb;
    if (ph === "nucleation") cb = $("ph-nucleation");
    else if (ph === "shell") cb = $("ph-shell");
    else cb = $(`ph-${ph}`);
    const on = cb ? cb.checked : true;
    mesh.visible = on;
    if (ph === "nucleation") mesh.material.opacity = op * 0.78; else if (ph === "shell") mesh.material.opacity = op * 0.65;
    else if (ph === "amorphous") mesh.material.opacity = Math.min(0.92, op * 1.9);
    else mesh.material.opacity = op;
  });
  const hotEl = $("nucleation-hotspots");
  hotspotGroup.visible = !!(hotEl && hotEl.checked);
  hotspotClusterMeshes.forEach((mesh) => {
    mesh.visible = hotspotGroup.visible;
  });
  updateHotspotMarkers(dmin, dmax, rmax, slab);
  shellGroup.visible = $("shells").checked;
  const guides = $("radial-guides");
  guideGroup.visible = !!(guides && guides.checked);
  const autoRotate = $("auto-rotate");
  controls.autoRotate = !!(autoRotate && autoRotate.checked);
  drawUnroll(dmin, dmax, rmax, slab);
  updateColorLegend(colorMode);

  const { shown, shownPh } = applyCa(style, colorMode, phaseOn, dmin, dmax, rmax, slab);
  const nWater = MODEL.water?.x?.length || 0;
  $("visible-count").textContent = `Showing ${shown} of ${nCa} Ca  ·  ${shownPh[0]} / ${shownPh[1]} / ${shownPh[2]}${
    nWater ? `  ·  ${nWaterShown} / ${nWater} water O` : ""
  }`;
  scheduleSessionSave();
}

function exportSlug() {
  const g = MODEL.geometry || "dna-caox";
  const title = (MODEL.title || MODEL.source || g).toLowerCase();
  const safe = title.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return safe || g;
}

const G16_PRESETS = {
  association_sp: {
    route: "#p B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ SCRF=(SMD,Solvent=Water)",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
  opt_freq: {
    route: "#p Opt Freq B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Int=UltraFine SCRF=(SMD,Solvent=Water)",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
  raman: {
    route: "#p Opt Freq=Raman B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Int=UltraFine SCRF=(SMD,Solvent=Water)",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
  screen_pm6: {
    route: "#p PM6 SCRF=(SMD,Solvent=Water)",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
  lanl2dz_ca: {
    route: "#p B3LYP/LANL2DZ EmpiricalDispersion=GD3BJ SCRF=(SMD,Solvent=Water)",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
  custom: {
    route: "",
    mem: "128GB",
    nproc: 28,
    charge: -1,
    mult: 1,
  },
};

function applyG16Preset(presetId) {
  const preset = G16_PRESETS[presetId] || G16_PRESETS.association_sp;
  const routeEl = $("g16-route");
  const memEl = $("g16-mem");
  const nprocEl = $("g16-nproc");
  const chargeEl = $("g16-charge");
  const multEl = $("g16-mult");
  if (routeEl && presetId !== "custom") routeEl.value = preset.route;
  if (memEl) memEl.value = preset.mem;
  if (nprocEl) nprocEl.value = preset.nproc;
  if (chargeEl && presetId !== "custom") chargeEl.value = preset.charge;
  if (multEl && presetId !== "custom") multEl.value = preset.mult;
  const chkEl = $("g16-chk");
  if (chkEl && !chkEl.dataset.userEdited) chkEl.value = `${exportSlug()}_g16.chk`;
}

function dist3(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function oxalateAtomsFromSegments(segments) {
  if (!segments?.length) return [];
  const tol = 0.28;
  const pts = [];
  const edges = [];

  function addPoint(p) {
    for (let i = 0; i < pts.length; i++) {
      if (dist3(p, pts[i].xyz) < tol) return i;
    }
    pts.push({ xyz: [p[0], p[1], p[2]], element: "O", kind: "oxalate" });
    return pts.length - 1;
  }

  segments.forEach((seg) => {
    const a = addPoint(seg[0]);
    const b = addPoint(seg[1]);
    edges.push({ a, b, d: dist3(seg[0], seg[1]) });
  });

  const elements = pts.map(() => "O");
  edges.forEach(({ a, b, d }) => {
    if (d >= 1.15 && d <= 1.68) {
      elements[a] = "C";
      elements[b] = "C";
    }
  });
  edges.forEach(({ a, b, d }) => {
    if (d >= 1.0 && d <= 1.45) {
      if (elements[a] === "C" && elements[b] === "O") elements[b] = "O";
      else if (elements[b] === "C" && elements[a] === "O") elements[a] = "O";
      else if (elements[a] === "C") elements[b] = "O";
      else if (elements[b] === "C") elements[a] = "O";
    }
  });

  return pts.map((p, i) => ({
    element: elements[i] === "C" ? "C" : "O",
    x: p.xyz[0],
    y: p.xyz[1],
    z: p.xyz[2],
    kind: "oxalate",
  }));
}

function currentOxalateSegments() {
  if (oxalateLines?.geometry?.attributes?.position) {
    const pos = oxalateLines.geometry.attributes.position.array;
    const nSeg = pos.length / 6;
    const segs = [];
    for (let i = 0; i < nSeg; i++) {
      segs.push([
        [pos[i * 6], pos[i * 6 + 1], pos[i * 6 + 2]],
        [pos[i * 6 + 3], pos[i * 6 + 4], pos[i * 6 + 5]],
      ]);
    }
    return segs;
  }
  return MODEL.oxalate || [];
}

function getCaExportFilters() {
  const phaseOn = PHASES.map((p) => $(`ph-${p}`).checked);
  return {
    dmin: Number($("dmin").value),
    dmax: Number($("dmax").value),
    rmax: Number($("rmax").value),
    slab: Number($("slab").value),
    phaseOn,
    filterVisible: $("g16-filter-visible")?.checked ?? true,
    hotspotsOnly: $("g16-hotspots-only")?.checked ?? false,
  };
}

function gatherGaussianAtoms(options) {
  const {
    includeDna,
    includeOxalate,
    includeWater,
    clusterRadius,
    maxAtoms,
    center,
  } = options;
  const atoms = [];
  const caFilters = getCaExportFilters();

  if (includeDna && MODEL.strands?.length) {
    MODEL.strands.forEach((strand) => {
      strand.residues.forEach((res, ri) => {
        const push = (element, xyz, kind) => {
          atoms.push({
            element,
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            kind,
            resseq: res.resseq ?? ri + 1,
            chain: strand.chain,
          });
        };
        if (res.P) push("P", res.P, "dna");
        if (res.C1) push("C", res.C1, "dna");
        if (res.N) push("N", res.N, "dna");
      });
    });
  }

  for (let i = 0; i < nCa; i++) {
    if (caFilters.hotspotsOnly && !ca.hotspot?.[i]) continue;
    if (caFilters.filterVisible) {
      const ph = ca.phase[i];
      if (
        !caFilters.phaseOn[ph] ||
        ca.dP[i] < caFilters.dmin ||
        ca.dP[i] > caFilters.dmax ||
        ca.radial[i] > caFilters.rmax ||
        Math.abs(ca.y[i]) > caFilters.slab
      ) {
        continue;
      }
    }
    atoms.push({
      element: "Ca",
      x: ca.x[i],
      y: ca.y[i],
      z: ca.z[i],
      kind: "ca",
    });
  }

  if (includeOxalate) {
    atoms.push(...oxalateAtomsFromSegments(currentOxalateSegments()));
  }

  if (includeWater && MODEL.water?.x?.length) {
    const wx = MODEL.water.x;
    const wy = MODEL.water.y;
    const wz = MODEL.water.z;
    for (let i = 0; i < wx.length; i++) {
      if (caFilters.filterVisible) {
        const r = Math.hypot(wx[i], wz[i]);
        if (r > caFilters.rmax || Math.abs(wy[i]) > caFilters.slab) continue;
      }
      atoms.push({ element: "O", x: wx[i], y: wy[i], z: wz[i], kind: "water" });
    }
  }

  let filtered = atoms;
  if (clusterRadius > 0) {
    filtered = atoms.filter((a) =>
      Math.hypot(a.x - center.x, a.y - center.y, a.z - center.z) <= clusterRadius
    );
  }

  if (maxAtoms > 0 && filtered.length > maxAtoms) {
    filtered = filtered
      .map((a) => ({
        a,
        d: Math.hypot(a.x - center.x, a.y - center.y, a.z - center.z),
      }))
      .sort((u, v) => u.d - v.d)
      .slice(0, maxAtoms)
      .map((row) => row.a);
  }

  return filtered;
}

function estimateGaussianCharge(atoms) {
  let nP = 0;
  let nCa = 0;
  let nOxC = 0;
  atoms.forEach((a) => {
    if (a.element === "P") nP += 1;
    else if (a.element === "Ca") nCa += 1;
    else if (a.kind === "oxalate" && a.element === "C") nOxC += 1;
  });
  const nOxUnits = Math.floor(nOxC / 2);
  return nP * -1 + nCa * 2 + nOxUnits * -2;
}

const GAUSSIAN_Z = {
  H: 1,
  C: 6,
  N: 7,
  O: 8,
  F: 9,
  Na: 11,
  Mg: 12,
  P: 15,
  S: 16,
  Cl: 17,
  K: 19,
  Ca: 20,
};

function totalAtomicNumber(atoms) {
  return atoms.reduce((sum, a) => sum + (GAUSSIAN_Z[a.element] || 0), 0);
}

function fitChargeToMultiplicity(charge, ztot, mult) {
  const wantOdd = (mult - 1) % 2 === 1;
  const nelecOdd = (ztot - charge) % 2 !== 0;
  if (nelecOdd === wantOdd) return charge;
  return charge <= 0 ? charge - 1 : charge + 1;
}

function resolveGaussianCharge(atoms, requested, mult) {
  const ztot = totalAtomicNumber(atoms);
  const chem = estimateGaussianCharge(atoms);
  const start = Number.isFinite(requested) ? requested : chem;
  return fitChargeToMultiplicity(start, ztot, mult);
}

function gaussianElementSymbol(el) {
  const u = String(el || "").trim();
  if (u.length === 1) return `${u} `;
  return u.slice(0, 2);
}

function buildGaussianComText(opts) {
  const {
    chk,
    mem,
    nproc,
    route,
    charge,
    mult,
    title,
    atoms,
  } = opts;
  const lines = [
    `%chk=${chk}`,
    `%mem=${mem}`,
    `%nprocshared=${nproc}`,
    route.trim(),
    "",
    title || "DNA-CaOx cluster export for Gaussian 16",
    "",
    `${charge} ${mult}`,
  ];
  atoms.forEach((a) => {
    lines.push(
      `${gaussianElementSymbol(a.element)} ${a.x.toFixed(6).padStart(12)} ${a.y
        .toFixed(6)
        .padStart(12)} ${a.z.toFixed(6).padStart(12)}`
    );
  });
  lines.push("");
  lines.push("");
  return lines.join("\n");
}

function prepareGaussianJob(options = {}) {
  const forGaussView = options.forGaussView ?? false;
  const presetId = $("g16-preset")?.value || "association_sp";
  const route =
    $("g16-route")?.value?.trim() ||
    G16_PRESETS[presetId]?.route ||
    G16_PRESETS.association_sp.route;
  if (!route.startsWith("#")) {
    throw new Error("Route line must start with # (e.g. #p B3LYP/6-31G(d) …).");
  }

  getHelixCenter();
  const center = helixCenter.clone();
  const atoms = gatherGaussianAtoms({
    includeDna: $("g16-include-dna")?.checked ?? true,
    includeOxalate: $("g16-include-oxalate")?.checked ?? true,
    includeWater: $("g16-include-water")?.checked ?? true,
    clusterRadius: Number($("g16-radius")?.value || 0),
    maxAtoms: Number($("g16-max-atoms")?.value || 0),
    center,
  });

  if (atoms.length < 3) {
    throw new Error("Fewer than 3 atoms selected — widen cluster radius or filters.");
  }
  if (atoms.length > 250) {
    throw new Error(
      `${atoms.length} atoms is too large for a typical G16 cluster job — lower max atoms or radius.`
    );
  }

  const requested = Number($("g16-charge")?.value ?? NaN);
  const mult = Math.max(1, Number($("g16-mult")?.value ?? 1));
  const charge = resolveGaussianCharge(atoms, requested, mult);
  const chargeEl = $("g16-charge");
  if (chargeEl && Number(chargeEl.value) !== charge) {
    chargeEl.value = charge;
  }
  const slug = exportSlug();
  const chk = forGaussView
    ? `${slug}_g16.chk`
    : ($("g16-chk")?.value || `${slug}_g16.chk`).trim();
  const comText = buildGaussianComText({
    chk,
    mem: ($("g16-mem")?.value || "16GB").trim(),
    nproc: Math.max(1, Number($("g16-nproc")?.value || 1)),
    route,
    charge,
    mult,
    title: `${MODEL.title || slug} - ${atoms.length} atoms`,
    atoms,
  });

  return { comText, atoms, charge, mult, route, slug, chk };
}

let g16PollTimer = null;
let g16ServerAvailable = false;
let g16BinaryAvailable = false;
let g16CheckTimer = null;

function g16ApiUrl(path) {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  const origin = window.location.origin;
  if (!origin || origin === "null" || window.location.protocol === "file:") {
    return `http://127.0.0.1:8765/api/g16${suffix}`;
  }
  const parts = window.location.pathname.split("/");
  const viewerIdx = parts.indexOf("viewer");
  const base = viewerIdx >= 0 ? parts.slice(0, viewerIdx).join("/") : "";
  return `${origin}${base}/api/g16${suffix}`;
}

async function fetchG16Json(path, options) {
  const res = await fetch(g16ApiUrl(path), options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Server error ${res.status}`);
  }
  return data;
}

function updateG16SubmitButton() {
  const btn = $("submit-g16");
  if (!btn) return;
  if (btn.dataset.busy === "1") return;
  btn.disabled = !g16ServerAvailable;
  btn.classList.toggle("is-busy", false);
  if (!g16ServerAvailable) {
    btn.title =
      "Start python3 scripts/viewer_server.py, then open http://localhost:8765/viewer/";
  } else if (!g16BinaryAvailable) {
    btn.title =
      "Viewer server OK — restart server after installing g16, or set G16_COMMAND=/Applications/g16/g16";
  } else {
    btn.title = "Submit cluster job to local Gaussian (g16)";
  }
}

function formatG16JobLabel(job) {
  const state = job.state || "?";
  const label = job.label || job.id;
  const energy =
    job.lastEnergy != null ? ` E=${Number(job.lastEnergy).toFixed(4)}` : "";
  return `${label} [${state}]${energy}`;
}

function updateG16JobSelect(jobs, selectedId) {
  const sel = $("g16-job-select");
  if (!sel) return;
  if (!jobs.length) {
    sel.innerHTML = "<option value=\"\">No jobs yet</option>";
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  const prev = selectedId || sel.value;
  sel.innerHTML = jobs
    .map((j) => {
      const id = j.id;
      const text = formatG16JobLabel(j);
      return `<option value="${id}">${text}</option>`;
    })
    .join("");
  if (prev && jobs.some((j) => j.id === prev)) sel.value = prev;
  else sel.value = jobs[0].id;
}

function updateG16JobLog(job) {
  const logEl = $("g16-job-log");
  const cancelBtn = $("g16-cancel-job");
  const liveBtn = $("g16-live-log");
  if (!logEl) return;
  if (!job) {
    logEl.textContent = "";
    if (cancelBtn) cancelBtn.disabled = true;
    if (liveBtn) liveBtn.disabled = true;
    return;
  }
  const header = [
    `Job ${job.id} — ${job.state || "?"}`,
    job.route || "",
    job.lastEnergy != null ? `Last SCF energy: ${job.lastEnergy}` : "",
    job.error ? `Error: ${job.error}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  const tail = (job.logTail || "").trim();
  logEl.textContent = tail
    ? `${header}\n\n${tail}`
    : `${header}\n\n(log empty — job starting…)`;
  if (cancelBtn) {
    cancelBtn.disabled = !["queued", "running"].includes(job.state);
  }
  if (liveBtn) liveBtn.disabled = false;
}

async function refreshG16Jobs(selectId) {
  try {
    const data = await fetchG16Json("/jobs");
    const jobs = data.jobs || [];
    updateG16JobSelect(jobs, selectId);
    const sel = $("g16-job-select");
    const id = sel?.value;
    const job = id ? jobs.find((j) => j.id === id) : jobs[0];
    if (job) updateG16JobLog(job);
    const running = jobs.some((j) => j.state === "queued" || j.state === "running");
    if (running && !g16PollTimer) {
      g16PollTimer = setInterval(() => refreshG16Jobs(), 3000);
    } else if (!running && g16PollTimer) {
      clearInterval(g16PollTimer);
      g16PollTimer = null;
    }
    return jobs;
  } catch (_) {
    return null;
  }
}

async function checkG16Server() {
  const el = $("g16-server-status");
  try {
    const env = await fetchG16Json("/env");
    g16ServerAvailable = true;
    g16BinaryAvailable = Boolean(env.available);
    if (g16CheckTimer) {
      clearInterval(g16CheckTimer);
      g16CheckTimer = null;
    }
    if (el) {
      el.textContent = env.available
        ? `Gaussian ready (${env.command}).`
        : "Viewer server OK — g16 not found. Set G16_COMMAND=/Applications/g16/g16 and restart server.";
    }
    updateG16SubmitButton();
    await refreshG16Jobs();
  } catch (_) {
    g16ServerAvailable = false;
    g16BinaryAvailable = false;
    if (el) {
      el.textContent =
        "Gaussian API offline — run python3 scripts/viewer_server.py and open http://localhost:8765/viewer/ (not file:// or plain http.server).";
    }
    updateG16SubmitButton();
    if (!g16CheckTimer) {
      g16CheckTimer = setInterval(() => checkG16Server(), 5000);
    }
  }
}

async function exportGaussian16() {
  const status = $("export-status");
  const btn = $("export-g16-com");
  if (btn) btn.disabled = true;
  status.textContent = "Building Gaussian 16 input…";
  try {
    const { comText, atoms, charge, mult } = prepareGaussianJob();
    downloadBlob(new Blob([comText], { type: "text/plain" }), `${exportSlug()}_g16.com`);
    status.textContent = `Saved Gaussian .com (${atoms.length} atoms, charge ${charge}, mult ${mult}).`;
  } catch (err) {
    status.textContent = `G16 export failed: ${err.message || err}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function exportGaussianGaussView() {
  const status = $("export-status");
  const btn = $("export-g16-gjf");
  if (btn) btn.disabled = true;
  status.textContent = "Building GaussView input from the full model PDB…";
  try {
    const geom = MODEL.geometry || DEFAULT_GEOM;
    const slug = exportSlug();
    const stem = `${slug}_g16`;
    if (!g16ServerAvailable) {
      throw new Error(
        "Viewer server required for full-model GaussView export. Run python3 scripts/viewer_server.py"
      );
    }
    const data = await fetchG16Json("/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "pdb",
        geometry: geom,
        label: slug,
        format: "both",
        route: $("g16-route")?.value?.trim() || "",
        mem: ($("g16-mem")?.value || "128GB").trim(),
        nproc: Math.max(1, Number($("g16-nproc")?.value || 28)),
        mult: Math.max(1, Number($("g16-mult")?.value || 1)),
        includeDna: $("g16-include-dna")?.checked ?? true,
        includeOxalate: $("g16-include-oxalate")?.checked ?? true,
        includeWater: $("g16-include-water")?.checked ?? true,
      }),
    });
    const gjf = data.paths?.gjf || `${data.exportsDir}/${stem}.gjf`;
    const pdb = data.paths?.pdb;
    status.textContent =
      `GaussView full model (${data.nAtoms} atoms, ${geom}, charge ${data.charge}): ${gjf}` +
      (pdb ? ` and ${pdb}` : "") +
      " — open the .pdb in GaussView for bonds, or the .gjf to edit/submit.";
  } catch (err) {
    status.textContent = `GaussView export failed: ${err.message || err}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function submitGaussian16() {
  const status = $("export-status");
  const btn = $("submit-g16");
  if (btn) {
    btn.dataset.busy = "1";
    btn.disabled = true;
    btn.classList.add("is-busy");
  }
  status.textContent = "Submitting Gaussian job…";
  try {
    const { comText, atoms, charge, mult } = prepareGaussianJob();
    const data = await fetchG16Json("/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ com: comText, label: exportSlug() }),
    });
    status.textContent = `Submitted job ${data.jobId} (${atoms.length} atoms, charge ${data.charge ?? charge}, mult ${mult}).`;
    if (data.chargeAdjustedFrom != null) {
      status.textContent += ` Charge ${data.chargeAdjustedFrom} → ${data.charge} so the electron count matches multiplicity ${mult}.`;
    }
    await refreshG16Jobs(data.jobId);
    if (!g16PollTimer) {
      g16PollTimer = setInterval(() => refreshG16Jobs(), 3000);
    }
    openG16LiveLog(data.jobId);
  } catch (err) {
    status.textContent = `G16 submit failed: ${err.message || err}`;
  } finally {
    if (btn) {
      delete btn.dataset.busy;
      btn.classList.remove("is-busy");
    }
    updateG16SubmitButton();
  }
}

async function cancelG16Job() {
  const sel = $("g16-job-select");
  const id = sel?.value;
  if (!id) return;
  const status = $("export-status");
  try {
    await fetchG16Json(`/jobs/${id}/cancel`, { method: "POST" });
    status.textContent = `Cancelled job ${id}.`;
    await refreshG16Jobs(id);
  } catch (err) {
    status.textContent = `Cancel failed: ${err.message || err}`;
  }
}

let g16LiveLogTimer = null;
let g16LiveLogJobId = null;

function selectedG16JobId() {
  return $("g16-job-select")?.value || g16LiveLogJobId || "";
}

function closeG16LiveLog() {
  if (g16LiveLogTimer) {
    clearInterval(g16LiveLogTimer);
    g16LiveLogTimer = null;
  }
  g16LiveLogJobId = null;
  const overlay = $("g16-log-overlay");
  if (overlay) overlay.hidden = true;
}

function renderG16LiveLog(job) {
  const full = $("g16-log-full");
  const meta = $("g16-log-meta");
  const poll = $("g16-log-poll-status");
  const follow = $("g16-log-follow");
  if (!full) return;
  const running = ["queued", "running"].includes(job.state);
  const bytes = job.logBytes != null ? ` · ${job.logBytes} bytes` : "";
  if (meta) {
    meta.textContent = `${job.label || job.id} [${job.state || "?"}]${
      job.lastEnergy != null ? `  E=${Number(job.lastEnergy).toFixed(6)}` : ""
    }${bytes}`;
  }
  const header = [
    job.route || "",
    job.error ? `Error: ${job.error}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  const tail = (job.logTail || "").trimEnd();
  const body = tail
    ? `${header ? `${header}\n\n` : ""}${tail}`
    : `${header ? `${header}\n\n` : ""}(log empty — waiting for Gaussian to write output…)`;
  const atBottom =
    full.scrollHeight - full.scrollTop - full.clientHeight < 40;
  full.textContent = body;
  if (follow?.checked || atBottom) {
    full.scrollTop = full.scrollHeight;
  }
  if (poll) {
    const stamp = new Date().toLocaleTimeString();
    poll.textContent = running
      ? `Live — refreshing every 1 s (${stamp})`
      : `Job ${job.state || "ended"} — last refresh ${stamp}`;
  }
}

async function refreshG16LiveLog() {
  const id = g16LiveLogJobId || selectedG16JobId();
  if (!id) return;
  try {
    const job = await fetchG16Json(`/jobs/${id}?tail=120000`);
    g16LiveLogJobId = job.id || id;
    renderG16LiveLog(job);
    updateG16JobLog(job);
    if (!["queued", "running"].includes(job.state) && g16LiveLogTimer) {
      clearInterval(g16LiveLogTimer);
      g16LiveLogTimer = setInterval(() => refreshG16LiveLog(), 4000);
    }
  } catch (err) {
    const poll = $("g16-log-poll-status");
    if (poll) poll.textContent = `Could not load log: ${err.message || err}`;
  }
}

function openG16LiveLog(jobId) {
  const id = jobId || selectedG16JobId();
  if (!id) return;
  g16LiveLogJobId = id;
  const overlay = $("g16-log-overlay");
  const full = $("g16-log-full");
  const poll = $("g16-log-poll-status");
  if (full) full.textContent = "Loading log…";
  if (poll) poll.textContent = "Connecting…";
  if (overlay) overlay.hidden = false;
  if (g16LiveLogTimer) clearInterval(g16LiveLogTimer);
  g16LiveLogTimer = setInterval(() => refreshG16LiveLog(), 1000);
  refreshG16LiveLog();
}

function captureFrame(scale, mime = "image/png", quality, { transparent = false } = {}) {
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const outW = Math.max(1, Math.round(cssW * scale));
  const outH = Math.max(1, Math.round(cssH * scale));
  const prevSize = new THREE.Vector2();
  renderer.getSize(prevSize);
  const prevPR = renderer.getPixelRatio();
  const prevBackground = scene.background;
  const prevClearColor = renderer.getClearColor(new THREE.Color());
  const prevClearAlpha = renderer.getClearAlpha();

  let dataUrl;
  try {
    camera.position.sub(viewPan);
    syncOrbitTarget();
    controls.update();
    camera.position.add(viewPan);
    renderer.setPixelRatio(1);
    renderer.setSize(outW, outH, false);
    camera.aspect = outW / outH;
    camera.updateProjectionMatrix();
    if (transparent) {
      scene.background = null;
      renderer.setClearColor(0x000000, 0);
    }
    renderer.render(scene, camera);
    dataUrl = renderer.domElement.toDataURL(mime, quality);
  } finally {
    scene.background = prevBackground;
    renderer.setClearColor(prevClearColor, prevClearAlpha);
    renderer.setPixelRatio(prevPR);
    renderer.setSize(prevSize.x, prevSize.y, false);
    camera.aspect = cssW / Math.max(cssH, 1);
    camera.updateProjectionMatrix();
    camera.position.sub(viewPan);
    syncOrbitTarget();
    controls.update();
    camera.position.add(viewPan);
    renderer.render(scene, camera);
  }

  return { dataUrl, width: outW, height: outH };
}

function dataUrlToBytes(dataUrl) {
  const bin = atob(dataUrl.split(",")[1]);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildPdfFromJpeg(jpegBytes, imgW, imgH, dpi = 300) {
  const ptW = (imgW * 72) / dpi;
  const ptH = (imgH * 72) / dpi;
  const enc = new TextEncoder();
  const chunks = [];
  let pos = 0;
  const xref = [];

  const addStr = (s) => {
    chunks.push(s);
    pos += enc.encode(s).length;
  };
  const addBin = (b) => {
    chunks.push(b);
    pos += b.length;
  };
  const mark = () => xref.push(pos);

  addStr("%PDF-1.4\n");
  mark();
  addStr("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  mark();
  addStr("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
  mark();
  addStr(
    `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${ptW.toFixed(3)} ${ptH.toFixed(3)}] ` +
      `/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n`
  );
  mark();
  addStr(
    `4 0 obj\n<< /Type /XObject /Subtype /Image /Width ${imgW} /Height ${imgH} ` +
      `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode ` +
      `/Length ${jpegBytes.length} >>\nstream\n`
  );
  addBin(jpegBytes);
  addStr("\nendstream\nendobj\n");
  mark();
  addStr(
    `5 0 obj\n<< /Length 44 >>\nstream\nq ${ptW.toFixed(3)} 0 0 ${ptH.toFixed(3)} 0 0 cm /Im0 Do Q\nendstream\nendobj\n`
  );

  const xrefStart = pos;
  let xrefStr = "xref\n0 6\n0000000000 65535 f \n";
  for (let i = 1; i < xref.length; i++) {
    xrefStr += `${String(xref[i]).padStart(10, "0")} 00000 n \n`;
  }
  addStr(xrefStr);
  addStr(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`);
  return new Blob(chunks, { type: "application/pdf" });
}

async function exportPNG(scale) {
  const status = $("export-status");
  const buttons = ["export-png-2x", "export-png-4x", "export-pdf"].map((id) => $(id));
  buttons.forEach((b) => {
    if (b) b.disabled = true;
  });
  status.textContent = `Rendering ${scale}× PNG…`;
  try {
    const { dataUrl, width, height } = captureFrame(scale, "image/png", undefined, {
      transparent: true,
    });
    const bytes = dataUrlToBytes(dataUrl);
    downloadBlob(new Blob([bytes], { type: "image/png" }), `${exportSlug()}_${scale}x.png`);
    status.textContent = `Saved ${width}×${height} PNG.`;
  } catch (err) {
    status.textContent = `Export failed: ${err.message || err}`;
  } finally {
    buttons.forEach((b) => {
      if (b) b.disabled = false;
    });
  }
}

async function exportPDF(scale = 4) {
  const status = $("export-status");
  const buttons = ["export-png-2x", "export-png-4x", "export-pdf"].map((id) => $(id));
  buttons.forEach((b) => {
    if (b) b.disabled = true;
  });
  status.textContent = `Rendering ${scale}× PDF…`;
  try {
    const { dataUrl, width, height } = captureFrame(scale, "image/jpeg", 0.94);
    const jpegBytes = dataUrlToBytes(dataUrl);
    const pdf = buildPdfFromJpeg(jpegBytes, width, height, 300);
    downloadBlob(pdf, `${exportSlug()}_${scale}x.pdf`);
    status.textContent = `Saved ${width}×${height} PDF @ 300 dpi.`;
  } catch (err) {
    status.textContent = `Export failed: ${err.message || err}`;
  } finally {
    buttons.forEach((b) => {
      if (b) b.disabled = false;
    });
  }
}

function setCoatView() {
  $("dmin").value = "0";
  $("dmax").value = "75";
  $("rmax").value = "75";
  $("slab").value = "75";
  $("ph-amorphous").checked = true;
  $("ph-intermediate").checked = true;
  $("ph-crystalline").checked = true;
  if ($("ph-shell")) $("ph-shell").checked = false;
  if ($("ph-nucleation")) $("ph-nucleation").checked = false;
  $("envelopes").checked = false;
  if ($("nucleation-hotspots")) $("nucleation-hotspots").checked = true;
  if ($("radial-guides")) $("radial-guides").checked = false;
  if ($("oxalate")) $("oxalate").checked = true;
  if ($("water")) $("water").checked = true;
  const cloud = document.querySelector('input[name="style"][value="cloud"]');
  if (cloud) cloud.checked = true;
  const dist = document.querySelector('input[name="color"][value="distance"]');
  if (dist) dist.checked = true;
  applyUi();
  setView("side");
}

function setNucleationView() {
  $("dmin").value = "0";
  $("dmax").value = "75";
  $("rmax").value = "75";
  $("slab").value = "75";
  $("ph-amorphous").checked = true;
  $("ph-intermediate").checked = true;
  $("ph-crystalline").checked = true;
  if ($("ph-shell")) $("ph-shell").checked = false;
  if ($("ph-nucleation")) $("ph-nucleation").checked = false;
  $("envelopes").checked = false;
  if ($("nucleation-hotspots")) $("nucleation-hotspots").checked = true;
  if ($("radial-guides")) $("radial-guides").checked = false;
  if ($("oxalate")) $("oxalate").checked = false;
  if ($("water")) $("water").checked = false;
  $("env-opacity").value = "18";
  const dist = document.querySelector('input[name="color"][value="distance"]');
  if (dist) dist.checked = true;
  const pts = document.querySelector('input[name="style"][value="points"]');
  if (pts) pts.checked = true;
  applyUi();
  setView("end");
}

function ui() {
  syncLabels();
  document.querySelectorAll("input").forEach((el) => {
    el.addEventListener("input", applyUi);
    el.addEventListener("change", applyUi);
  });
  $("view-end").addEventListener("click", () => setView("end"));
  $("view-side").addEventListener("click", () => setView("side"));
  $("view-seed").addEventListener("click", () => setView("seed"));
  const vn = $("view-nucleation");
  if (vn) vn.addEventListener("click", setNucleationView);
  populateGeomSelect();
  const geomSelect = $("geom-select");
  if (geomSelect) {
    geomSelect.addEventListener("change", () => {
      if (geomSelect.value) loadGeometry(geomSelect.value);
    });
  }
  $("traj-step").addEventListener("input", () => {
    stopTraj();
    applyTrajFrame(Number($("traj-step").value));
  });
  $("traj-play").addEventListener("click", toggleTrajPlay);
  $("traj-reset").addEventListener("click", resetTraj);
  $("export-png-2x").addEventListener("click", () => exportPNG(2));
  $("export-png-4x").addEventListener("click", () => exportPNG(4));
  $("export-pdf").addEventListener("click", () => exportPDF(4));
  const g16Preset = $("g16-preset");
  const g16Chk = $("g16-chk");
  if (g16Preset) {
    g16Preset.addEventListener("change", () => applyG16Preset(g16Preset.value));
    applyG16Preset(g16Preset.value);
  }
  if (g16Chk) {
    g16Chk.addEventListener("input", () => {
      g16Chk.dataset.userEdited = "1";
    });
  }
  const g16Est = $("g16-estimate-charge");
  if (g16Est) {
    g16Est.addEventListener("click", () => {
      getHelixCenter();
      const atoms = gatherGaussianAtoms({
        includeDna: $("g16-include-dna")?.checked ?? true,
        includeOxalate: $("g16-include-oxalate")?.checked ?? true,
        includeWater: $("g16-include-water")?.checked ?? true,
        clusterRadius: Number($("g16-radius")?.value || 0),
        maxAtoms: Number($("g16-max-atoms")?.value || 0),
        center: helixCenter.clone(),
      });
      const q = resolveGaussianCharge(atoms, estimateGaussianCharge(atoms), 1);
      const chargeEl = $("g16-charge");
      if (chargeEl) chargeEl.value = q;
      const hint = $("g16-hint");
      if (hint) {
        const z = totalAtomicNumber(atoms);
        hint.textContent = `Estimated charge ${q} from ${atoms.length} atoms (P −1, Ca +2, oxalate −2 per C₂O₄; adjusted for ${z - q} electrons / singlet). Verify before running.`;
      }
    });
  }
  const g16Export = $("export-g16-com");
  if (g16Export) g16Export.addEventListener("click", () => exportGaussian16());
  const g16Gjf = $("export-g16-gjf");
  if (g16Gjf) g16Gjf.addEventListener("click", () => exportGaussianGaussView());
  const g16Submit = $("submit-g16");
  if (g16Submit) {
    g16Submit.disabled = true;
    g16Submit.addEventListener("click", () => submitGaussian16());
  }
  const g16JobSel = $("g16-job-select");
  if (g16JobSel) {
    g16JobSel.addEventListener("change", async () => {
      const id = g16JobSel.value;
      if (!id) return;
      try {
        const job = await fetchG16Json(`/jobs/${id}`);
        updateG16JobLog(job);
      } catch (_) {
        /* ignore */
      }
    });
  }
  const g16Cancel = $("g16-cancel-job");
  if (g16Cancel) g16Cancel.addEventListener("click", () => cancelG16Job());
  const g16Live = $("g16-live-log");
  if (g16Live) g16Live.addEventListener("click", () => openG16LiveLog());
  const g16LogClose = $("g16-log-close");
  if (g16LogClose) g16LogClose.addEventListener("click", () => closeG16LiveLog());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("g16-log-overlay")?.hidden) {
      closeG16LiveLog();
    }
  });
  checkG16Server();
  wireSession();
  applyUi();
}

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

function updateScaleBar() {
  const dist = camera.position.distanceTo(controls.target);
  const worldH = 2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
  const px = (20 / worldH) * canvas.clientHeight;
  document.querySelector("#scale i").style.width = `${Math.max(24, Math.min(220, px))}px`;
}

if (MODELS[DEFAULT_GEOM]) {
  loadGeometry(DEFAULT_GEOM);
} else {
  buildDNA();
  buildEnvelopes();
  buildShells();
  buildRadialGuides();
  buildHotspotMarkers();
  rebuildMineral();
  setView("side");
}
ui();
const savedSession = readSession();
if (savedSession?.geometry && MODELS[savedSession.geometry]) {
  applySession(savedSession);
}
resize();
window.addEventListener("resize", resize);

function tick() {
  camera.position.sub(viewPan);
  syncOrbitTarget();
  controls.update();
  camera.position.add(viewPan);
  updateScaleBar();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
