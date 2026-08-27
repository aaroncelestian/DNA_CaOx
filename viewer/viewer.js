import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODELS = window.DNA_CAOX_MODELS || { sphere: window.DNA_CAOX_MODEL };
let MODEL = MODELS.altp || MODELS.local || MODELS.sphere || window.DNA_CAOX_MODEL;
if (!MODEL) {
  throw new Error("model-data.js did not load");
}

const PHASES = ["amorphous", "intermediate", "crystalline"];
const PHASE_COLOR = {
  amorphous: 0xd95f02,
  intermediate: 0x7570b3,
  crystalline: 0x1b9e77,
  nucleation: 0xffdd57,
};
const STRAND_COLOR = { A: 0xf2e6c9, B: 0xb7c9d9, C: 0xf2e6c9, D: 0xb7c9d9 };

let ca = MODEL.ca;
let nCa = ca.x.length;
let trajData = null;
let trajFrame = 0;
let trajPlaying = false;
let trajTimer = null;
const caRest = { x: [...ca.x], y: [...ca.y], z: [...ca.z] };
let oxalateLines = null;
let oxalateRest = null;
let oxalateUnitOffsets = [];

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
controls.target.set(0, 0, 0);

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
root.add(dnaGroup, mineralGroup, envGroup, shellGroup, hotspotGroup);

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
      new THREE.MeshPhysicalMaterial({ color: 0xc45c26, roughness: 0.4 })
    );
    tube.name = "phosphate-trace";
    dnaGroup.add(tube);

    const pGeom = new THREE.SphereGeometry(0.85, 16, 12);
    const pMat = new THREE.MeshPhysicalMaterial({ color: 0xe8a14b, roughness: 0.3 });
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

function buildEnvelopes() {
  const names = [...PHASES];
  if (MODEL.envelopes?.nucleation?.vertices?.length) names.push("nucleation");
  names.forEach((name) => {
    const env = MODEL.envelopes[name];
    if (!env || !env.vertices.length) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(env.vertices.flat(), 3)
    );
    geom.setIndex(env.indices);
    geom.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({
      color: PHASE_COLOR[name] || PHASE_COLOR.intermediate,
      transparent: true,
      opacity: name === "nucleation" ? 0.14 : 0.18,
      side: THREE.DoubleSide,
      depthWrite: false,
      emissive: name === "nucleation" ? 0x221a00 : 0x000000,
      emissiveIntensity: name === "nucleation" ? 0.12 : 0,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.phase = name;
    envGroup.add(mesh);
  });
}

let hotspotMesh = null;
let hotspotIndices = [];

function updateHotspotMarkers() {
  if (!hotspotMesh || !hotspotIndices.length) return;
  const dummy = new THREE.Object3D();
  for (let k = 0; k < hotspotIndices.length; k++) {
    const i = hotspotIndices[k];
    dummy.position.set(ca.x[i], ca.y[i], ca.z[i]);
    const s = 0.9 + 0.5 * Math.min(1, (ca.comRegistry?.[i] || 0) / 0.28);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    hotspotMesh.setMatrixAt(k, dummy.matrix);
  }
  hotspotMesh.instanceMatrix.needsUpdate = true;
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
  const geom = new THREE.SphereGeometry(1, 10, 8);
  const mat = new THREE.MeshBasicMaterial({
    color: 0xc9b86a,
    transparent: true,
    opacity: 0.38,
    depthTest: true,
  });
  hotspotMesh = new THREE.InstancedMesh(geom, mat, pts.length);
  const dummy = new THREE.Object3D();
  for (let k = 0; k < pts.length; k++) {
    const i = pts[k];
    dummy.position.set(ca.x[i], ca.y[i], ca.z[i]);
    const s = 0.9 + 0.5 * Math.min(1, (ca.comRegistry?.[i] || 0) / 0.28);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    hotspotMesh.setMatrixAt(k, dummy.matrix);
  }
  hotspotMesh.instanceMatrix.needsUpdate = true;
  hotspotGroup.add(hotspotMesh);
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
    const t = Math.min(1, ca.dP[i] / 32);
    return new THREE.Color().setHSL(0.08 + 0.55 * (1 - t), 0.7, 0.48);
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
  controls.target.set(0, 0, 0);
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
  hotspotMesh = null;
  hotspotIndices = [];
  buildDNA();
  buildEnvelopes();
  buildShells();
  buildHotspotMarkers();
  rebuildMineral();
  const geomIds = {
    sphere: "geom-sphere",
    slab: "geom-slab",
    allp: "geom-allp",
    local: "geom-local",
    altp: "geom-altp",
    gel: "geom-gel",
    shell15: "geom-shell15",
    gel_altp_geom: "geom-gel-altp",
    shell_lattice: "geom-shell-lattice",
  };
  Object.entries(geomIds).forEach(([g, id]) => {
    const el = $(id);
    if (el) el.classList.toggle("active", name === g);
  });
  syncLabels();
  applyUi();
  setView("side");
  if (name === "shell_lattice" || name === "shell_lattice_seeded") setNucleationView();
  else loadTrajectory(MODEL.traj);
}

async function loadTrajectory(url) {
  trajData = null;
  trajFrame = 0;
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
    if (section) section.hidden = false;
    if (status) {
      status.textContent = `${trajData.frames.length} frames · ${url.split("/").pop()} · hotspots track Ca (OMM labels)`;
    }
    applyTrajFrame(0);
  } catch (err) {
    if (section) section.hidden = true;
    if (status) status.textContent = `Trajectory unavailable: ${err.message}`;
  }
}

function applyTrajFrame(idx) {
  if (!trajData || !trajData.frames[idx]) return;
  const f = trajData.frames[idx];
  const n = Math.min(nCa, f.ca.x.length);
  for (let i = 0; i < n; i++) {
    ca.x[i] = f.ca.x[i];
    ca.y[i] = f.ca.y[i];
    ca.z[i] = f.ca.z[i];
  }
  if (f.oxalate) {
    updateOxalateFromTraj(f.oxalate);
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
  rebuildMineral();
  updateHotspotMarkers();
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

  envGroup.visible = $("envelopes").checked;
  const op = Number($("env-opacity").value) / 100;
  envGroup.children.forEach((mesh) => {
    const ph = mesh.userData.phase;
    const cb =
      ph === "nucleation"
        ? $("ph-nucleation")
        : $(`ph-${ph}`);
    let on = cb ? cb.checked : true;
    // Nucleation envelope is from OMM endpoint; it does not follow FIRE trajectory.
    if (ph === "nucleation" && trajData) on = false;
    mesh.visible = on;
    mesh.material.opacity = ph === "nucleation" ? op * 0.45 : op;
  });
  const hotEl = $("nucleation-hotspots");
  hotspotGroup.visible = !!(hotEl && hotEl.checked);
  shellGroup.visible = $("shells").checked;

  const { shown, shownPh } = applyCa(style, colorMode, phaseOn, dmin, dmax, rmax, slab);
  $("visible-count").textContent = `Showing ${shown} of ${nCa} Ca  ·  ${shownPh[0]} / ${shownPh[1]} / ${shownPh[2]}`;
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

function setNucleationView() {
  $("dmin").value = "0";
  $("dmax").value = "18";
  $("rmax").value = "50";
  $("ph-amorphous").checked = false;
  $("ph-intermediate").checked = true;
  $("ph-crystalline").checked = true;
  if ($("ph-nucleation")) $("ph-nucleation").checked = true;
  $("envelopes").checked = true;
  if ($("nucleation-hotspots")) $("nucleation-hotspots").checked = true;
  $("env-opacity").value = "22";
  const phase = document.querySelector('input[name="color"][value="phase"]');
  if (phase) phase.checked = true;
  applyUi();
  setView("side");
  if (MODEL.traj) loadTrajectory(MODEL.traj);
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
  $("geom-sphere").addEventListener("click", () => loadGeometry("sphere"));
  $("geom-slab").addEventListener("click", () => loadGeometry("slab"));
  $("geom-allp").addEventListener("click", () => loadGeometry("allp"));
  $("geom-local").addEventListener("click", () => loadGeometry("local"));
  $("geom-altp").addEventListener("click", () => loadGeometry("altp"));
  $("geom-gel").addEventListener("click", () => loadGeometry("gel"));
  $("geom-shell15").addEventListener("click", () => loadGeometry("shell15"));
  $("geom-gel-altp").addEventListener("click", () => loadGeometry("gel_altp_geom"));
  $("geom-shell-lattice").addEventListener("click", () => loadGeometry("shell_lattice"));
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

buildDNA();
buildEnvelopes();
buildShells();
buildHotspotMarkers();
rebuildMineral();
setView("side");
ui();
resize();
window.addEventListener("resize", resize);

function tick() {
  controls.update();
  updateScaleBar();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
