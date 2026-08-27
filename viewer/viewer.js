import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODELS = window.DNA_CAOX_MODELS || { shell_lattice: window.DNA_CAOX_MODEL };
const DEFAULT_GEOM = "shell_lattice";
let MODEL =
  MODELS[DEFAULT_GEOM] ||
  MODELS.templating_gel ||
  MODELS.slab ||
  window.DNA_CAOX_MODEL;
if (!MODEL) {
  throw new Error("model-data.js did not load");
}

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
    alpha: false,
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
controls.enablePan = false; // custom pan moves camera; pivot stays on helix axis
controls.target.set(0, 0, 0);
controls.cursor.copy(controls.target);

const orbitPivot = new THREE.Vector3();
const panRight = new THREE.Vector3();
const panUp = new THREE.Vector3();
const viewPan = new THREE.Vector3();
const panState = { active: false, pointerId: null, lastX: 0, lastY: 0 };
let spacePanArm = false;

function getHelixCenter() {
  const hx = MODEL.helix || {};
  const yMid = 0.5 * ((hx.dnaZmin ?? 0) + (hx.dnaZmax ?? 0));
  orbitPivot.set(0, yMid, 0);
  return orbitPivot;
}

function lockOrbitToHelixCenter() {
  getHelixCenter();
  controls.target.copy(orbitPivot).add(viewPan);
  controls.cursor.copy(controls.target);
}

function panPixelScale() {
  const el = canvas;
  if (!el.clientHeight) return 0;
  getHelixCenter();
  panUp.copy(orbitPivot).add(viewPan); // temp: orbit target
  panRight.copy(camera.position).sub(panUp);
  let dist = panRight.length();
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

function panCameraByPixels(deltaX, deltaY) {
  const scale = panPixelScale();
  if (!scale) return;
  camera.updateMatrixWorld();
  panRight.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(-deltaX * scale);
  viewPan.add(panRight);
  camera.position.add(panRight);
  panUp.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(deltaY * scale);
  viewPan.add(panUp);
  camera.position.add(panUp);
  lockOrbitToHelixCenter();
}

function wantsPanPointer(e) {
  if (spacePanArm) return true;
  return (
    e.button === 2 ||
    e.button === 1 ||
    (e.button === 0 && (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey))
  );
}

function onDocumentPanMove(e) {
  if (!panState.active) return;
  if (panState.pointerId != null && e.pointerId !== panState.pointerId) return;
  const dx = e.clientX - panState.lastX;
  const dy = e.clientY - panState.lastY;
  panState.lastX = e.clientX;
  panState.lastY = e.clientY;
  if (dx || dy) panCameraByPixels(dx, dy);
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
      if (!wantsPanPointer(e)) return;
      panState.active = true;
      panState.pointerId = e.pointerId;
      panState.lastX = e.clientX;
      panState.lastY = e.clientY;
      document.addEventListener("pointermove", onDocumentPanMove);
      document.addEventListener("pointerup", onDocumentPanEnd);
      document.addEventListener("pointercancel", onDocumentPanEnd);
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch (_) {
        /* ignore */
      }
      e.preventDefault();
      e.stopImmediatePropagation();
    },
    { capture: true }
  );

  canvas.addEventListener(
    "wheel",
    (e) => {
      if (e.ctrlKey) return; // trackpad pinch → OrbitControls zoom
      const { dx, dy } = wheelPanDeltas(e);
      e.preventDefault();
      e.stopImmediatePropagation();
      panCameraByPixels(-dx, -dy);
    },
    { passive: false, capture: true }
  );

  // Safari / trackpad right-click drag sometimes uses mouse events only
  canvas.addEventListener(
    "mousedown",
    (e) => {
      if (!wantsPanPointer(e) || panState.active) return;
      panState.active = true;
      panState.pointerId = -1;
      panState.lastX = e.clientX;
      panState.lastY = e.clientY;
      document.addEventListener("mousemove", onMousePanMove);
      document.addEventListener("mouseup", onMousePanEnd);
      e.preventDefault();
      e.stopImmediatePropagation();
    },
    { capture: true }
  );
}

function onMousePanMove(e) {
  if (!panState.active || panState.pointerId !== -1) return;
  const dx = e.clientX - panState.lastX;
  const dy = e.clientY - panState.lastY;
  panState.lastX = e.clientX;
  panState.lastY = e.clientY;
  if (dx || dy) panCameraByPixels(dx, dy);
  e.preventDefault();
}

function onMousePanEnd(e) {
  if (!panState.active || panState.pointerId !== -1) return;
  panState.active = false;
  panState.pointerId = null;
  document.removeEventListener("mousemove", onMousePanMove);
  document.removeEventListener("mouseup", onMousePanEnd);
}

setupCanvasPan();

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
      const glowMat = new THREE.MeshBasicMaterial({
        color: ACCENT_YELLOW.glow,
        transparent: true,
        opacity: 0.38,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
      });
      const glowMesh = new THREE.Mesh(geom.clone(), glowMat);
      glowMesh.scale.setScalar(1.05);
      glowMesh.userData.phase = name;
      glowMesh.userData.glow = true;
      glowMesh.renderOrder = 0;
      envGroup.add(glowMesh);

      const mat = new THREE.MeshPhysicalMaterial({
        color: ACCENT_YELLOW.bright,
        transparent: true,
        opacity: 0.34,
        side: THREE.DoubleSide,
        depthWrite: false,
        roughness: 0.18,
        metalness: 0.0,
        emissive: ACCENT_YELLOW.emissive,
        emissiveIntensity: 0.9,
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
      opacity: name === "shell" ? 0.16 : 0.2,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
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
    (slab >= 49 ? ", full length" : `, |axis| ≤ ${slab} Å`);
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
  const center = getHelixCenter();
  controls.target.copy(center);
  controls.cursor.copy(center);
  viewPan.set(0, 0, 0);
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

function loadGeometry(name) {
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
  const geomIds = {
    templating_gel: "geom-templating",
    templating_gel_thick: "geom-templating-thick",
    templating_nodna: "geom-nodna",
    shell_lattice: "geom-shell-lattice",
    slab: "geom-slab",
  };
  Object.entries(geomIds).forEach(([g, id]) => {
    const el = $(id);
    if (el) el.classList.toggle("active", name === g);
  });
  syncLabels();
  if (
    name === "templating_gel" ||
    name === "templating_gel_thick" ||
    name === "templating_nodna"
  ) {
    setCoatView();
  } else if (name === "shell_lattice") {
    setNucleationView();
  } else {
    applyUi();
    setView("side");
  }
  loadTrajectory(MODEL.traj);
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
      slider.value = 0;
    }
    const f0 = trajData.frames[0]?.ca;
    if (f0?.x?.length) {
      trajFrame0Ca = { x: [...f0.x], y: [...f0.y], z: [...f0.z] };
      trajFrame0Oxalate = captureTrajOxalate(trajData.frames[0].oxalate);
      trajMaxDisp = trajDisplacementStats(trajData.frames);
      if (trajMaxDisp < 0.5) {
        trajAmp = Math.min(40, 1.2 / Math.max(trajMaxDisp, 1e-4));
      }
    }
    if (section) section.hidden = false;
    if (status) status.textContent = trajStatusText(url);
    applyTrajFrame(0);
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
  $("slab-v").textContent = slab >= 49 ? "full" : `±${slab} Å`;

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
    if (ph === "nucleation") {
      mesh.material.opacity = mesh.userData.glow ? op * 0.58 : op * 0.42;
    } else if (ph === "shell") mesh.material.opacity = op * 0.65;
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
}

function exportSlug() {
  const g = MODEL.geometry || "dna-caox";
  const title = (MODEL.title || MODEL.source || g).toLowerCase();
  const safe = title.replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return safe || g;
}

function captureFrame(scale, mime = "image/png", quality) {
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const outW = Math.max(1, Math.round(cssW * scale));
  const outH = Math.max(1, Math.round(cssH * scale));
  const prevSize = new THREE.Vector2();
  renderer.getSize(prevSize);
  const prevPR = renderer.getPixelRatio();

  controls.update();
  lockOrbitToHelixCenter();
  renderer.setPixelRatio(1);
  renderer.setSize(outW, outH, false);
  camera.aspect = outW / outH;
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);

  const dataUrl = renderer.domElement.toDataURL(mime, quality);

  renderer.setPixelRatio(prevPR);
  renderer.setSize(prevSize.x, prevSize.y, false);
  camera.aspect = cssW / Math.max(cssH, 1);
  camera.updateProjectionMatrix();
  controls.update();
  lockOrbitToHelixCenter();
  renderer.render(scene, camera);

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
    const { dataUrl, width, height } = captureFrame(scale, "image/png");
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
  $("dmax").value = "40";
  $("rmax").value = "50";
  $("slab").value = "50";
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
  const sph = document.querySelector('input[name="style"][value="spheres"]');
  if (sph) sph.checked = true;
  const dist = document.querySelector('input[name="color"][value="distance"]');
  if (dist) dist.checked = true;
  applyUi();
  setView("side");
}

function setNucleationView() {
  $("dmin").value = "0";
  $("dmax").value = "28";
  $("rmax").value = "50";
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
  $("geom-templating").addEventListener("click", () => loadGeometry("templating_gel"));
  $("geom-templating-thick").addEventListener("click", () =>
    loadGeometry("templating_gel_thick")
  );
  $("geom-nodna").addEventListener("click", () => loadGeometry("templating_nodna"));
  $("geom-shell-lattice").addEventListener("click", () => loadGeometry("shell_lattice"));
  $("geom-slab").addEventListener("click", () => loadGeometry("slab"));
  $("traj-step").addEventListener("input", () => {
    stopTraj();
    applyTrajFrame(Number($("traj-step").value));
  });
  $("traj-play").addEventListener("click", toggleTrajPlay);
  $("traj-reset").addEventListener("click", resetTraj);
  $("export-png-2x").addEventListener("click", () => exportPNG(2));
  $("export-png-4x").addEventListener("click", () => exportPNG(4));
  $("export-pdf").addEventListener("click", () => exportPDF(4));
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
resize();
window.addEventListener("resize", resize);

function tick() {
  lockOrbitToHelixCenter();
  controls.update();
  lockOrbitToHelixCenter();
  updateScaleBar();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
