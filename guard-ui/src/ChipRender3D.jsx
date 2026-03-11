import React, { useRef, useEffect, useState, useCallback } from "react";
import * as THREE from "three";

const DRUG_HEX = { RIF: 0x3288bd, INH: 0x66c2a5, EMB: 0xabdda4, PZA: 0xfee08b, FQ: 0xf46d43, AG: 0xd53e4f, CTRL: 0x888888 };
const DRUG_CSS = { RIF: "#3288bd", INH: "#66c2a5", EMB: "#abdda4", PZA: "#fee08b", FQ: "#f46d43", AG: "#d53e4f", CTRL: "#888888" };
const addAt = (parent, mesh, x, y, z) => { mesh.position.set(x, y, z); parent.add(mesh); return mesh; };

// ── Electrochemistry curve generators ──
const miniSWV = (Gamma, G0) => {
  const nFRT = (2 * 96485) / (8.314 * 310.15);
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const E = -0.05 - i * 0.004375;
    const xp = nFRT * (E + 0.025 - (-0.22));
    const xm = nFRT * (E - 0.025 - (-0.22));
    pts.push({ E, I: (Gamma / G0) * (1 / (1 + Math.exp(xp)) - 1 / (1 + Math.exp(xm))) * 3.0 });
  }
  return pts;
};

const miniDPV = (Gamma, G0) => {
  const nFRT = (2 * 96485) / (8.314 * 310.15);
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const E = -0.05 - i * 0.004375;
    const xp = nFRT * (E + 0.05 - (-0.22));
    const xm = nFRT * (E - (-0.22));
    pts.push({ E, I: (Gamma / G0) * (1 / (1 + Math.exp(xp)) - 1 / (1 + Math.exp(xm))) * 2.5 });
  }
  return pts;
};

const miniEIS = (Gamma, G0) => {
  const Rs = 50, Rct0 = 2000, Cdl = 20e-6;
  const Rct = Rct0 * (Gamma / G0) + 100;
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const omega = 2 * Math.PI * Math.pow(10, i * 0.1);
    const d = 1 + (omega * Rct * Cdl) ** 2;
    pts.push({ Zr: Rs + Rct / d, Zi: omega * Rct * Rct * Cdl / d });
  }
  return pts;
};

// Sprite text label helper
const makeSprite = (text, color, size) => {
  const canvas = document.createElement("canvas");
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.font = `bold ${Math.round(size || 18)}px monospace`;
  ctx.fillStyle = color || "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size ? size * 0.25 : 4.5, size ? size * 0.0625 : 1.1, 1);
  return sprite;
};

export default function ChipRender3D({ electrodeLayout, targetDrug, targetStrategy, getEfficiency, results, computeGamma, echemTime, echemKtrans, echemGamma0_mol, HEADING, MONO }) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);
  const [mode, setMode] = useState(1);
  const [selectedPad, setSelectedPad] = useState(null);
  const [cas12aActive, setCas12aActive] = useState(false);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [incubationMin, setIncubationMin] = useState(echemTime);
  const [curveMode, setCurveMode] = useState("SWV");
  const [showMicrofluidics, setShowMicrofluidics] = useState(true);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    const W = container.clientWidth;
    const H = Math.round(W * 9 / 16);
    container.style.height = H + "px";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.setClearColor(0xF8F9FA);
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 500);
    const scene = new THREE.Scene();

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xFFF5E6, 0.75);
    key.position.set(-30, 50, 40);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1; key.shadow.camera.far = 200;
    key.shadow.camera.left = -50; key.shadow.camera.right = 50;
    key.shadow.camera.top = 50; key.shadow.camera.bottom = -50;
    key.shadow.bias = -0.001;
    scene.add(key);
    scene.add(new THREE.DirectionalLight(0xE6F0FF, 0.3).translateX(40).translateY(30).translateZ(-20));
    scene.add(new THREE.DirectionalLight(0xFFFFFF, 0.2).translateY(20).translateZ(-50));
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ opacity: 0.15 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -1; ground.receiveShadow = true;
    scene.add(ground);

    // ══════════════════════════════════════════════════════════
    // MODE 1: CHIP OVERVIEW — Layered substrate + microfluidics
    // ══════════════════════════════════════════════════════════
    const chipGroup = new THREE.Group();
    scene.add(chipGroup);

    // Materials
    const kaptonMat = new THREE.MeshPhysicalMaterial({ color: 0xC8943E, roughness: 0.25, clearcoat: 0.3, clearcoatRoughness: 0.4, transparent: true, opacity: 0.94 });
    const wellMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.95 });
    const channelMat = new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.8 });
    const goldMat = new THREE.MeshStandardMaterial({ color: 0xDAA520, metalness: 0.85, roughness: 0.15 });
    const pdmsMat = new THREE.MeshPhysicalMaterial({ color: 0xCCDDEE, transparent: true, opacity: 0.22, roughness: 0.1, clearcoat: 0.6, side: THREE.DoubleSide });
    const fluidicChannelMat = new THREE.MeshStandardMaterial({ color: 0x4488AA, transparent: true, opacity: 0.35, roughness: 0.4 });
    const chamberWallMat = new THREE.MeshStandardMaterial({ color: 0x336677, transparent: true, opacity: 0.5, roughness: 0.4, side: THREE.DoubleSide });
    const agMat = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, metalness: 0.7, roughness: 0.2 });

    // ── Kapton substrate (beige base — visible from side) ──
    const body = new THREE.Mesh(new THREE.BoxGeometry(65, 1.2, 35), kaptonMat);
    body.castShadow = true; body.receiveShadow = true;
    chipGroup.add(body);

    // ── LIG patterned regions on top surface ──
    const ligTopMat = new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.85, metalness: 0.1 });
    // LIG covers the detection grid area + CE strip + traces
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(36, 0.08, 22), ligTopMat), 17, 0.64, 0);

    // Detection pads — 5×3 grid
    const flatPads = electrodeLayout.flat();
    const padMeshes = [];
    const padPositions = [];
    const gX = 5, gZ = -7, sX = 6, sZ = 7;

    // ── Passivation mask overlay ──
    // Semi-transparent dielectric with circular openings at each WE
    const maskShape = new THREE.Shape();
    maskShape.moveTo(-33, -18); maskShape.lineTo(33, -18);
    maskShape.lineTo(33, 18); maskShape.lineTo(-33, 18); maskShape.closePath();
    flatPads.forEach((_, idx) => {
      const row = Math.floor(idx / 5), col = idx % 5;
      const px = gX + col * sX, pz = gZ + row * sZ;
      const hole = new THREE.Path();
      hole.absarc(px, -pz, 1.5, 0, Math.PI * 2, false);
      maskShape.holes.push(hole);
    });
    const maskGeo = new THREE.ShapeGeometry(maskShape);
    const passivationMesh = new THREE.Mesh(maskGeo, new THREE.MeshStandardMaterial({
      color: 0x2E7D32, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false
    }));
    passivationMesh.rotation.x = -Math.PI / 2;
    passivationMesh.position.y = 0.70;
    chipGroup.add(passivationMesh);

    // ── Microfluidic overlay group (toggleable) ──
    const fluidicsGroup = new THREE.Group();
    chipGroup.add(fluidicsGroup);

    // PDMS slab on top
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(65, 0.25, 35), pdmsMat), 0, 0.87, 0);

    // Sample inlet port (top-left)
    const inletGeo = new THREE.CylinderGeometry(1.2, 1.2, 0.35, 16);
    addAt(fluidicsGroup, new THREE.Mesh(inletGeo, new THREE.MeshStandardMaterial({ color: 0x3388AA, transparent: true, opacity: 0.55, roughness: 0.3 })), -25, 0.95, -8);
    addAt(fluidicsGroup, makeSprite("INLET", "#1a6680", 14), -25, 2.0, -8);

    // Lysis chamber (existing sample well area, now labeled)
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(12, 0.1, 10), wellMat), -22, 0.65, 0);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(11.5, 0.06, 9.5), new THREE.MeshStandardMaterial({ color: 0x93C5FD, transparent: true, opacity: 0.3 })), -22, 0.72, 0);
    const beadMat = new THREE.MeshStandardMaterial({ color: 0x4B5563, metalness: 0.6, roughness: 0.3 });
    for (let i = 0; i < 12; i++) {
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.25, 6, 6), beadMat);
      b.position.set(-22 + (Math.random() - 0.5) * 9, 0.85, (Math.random() - 0.5) * 7);
      chipGroup.add(b);
    }
    addAt(fluidicsGroup, makeSprite("LYSIS", "#1a6680", 12), -22, 2.0, 0);

    // Channel: inlet → lysis
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.12, 8), fluidicChannelMat), -25, 0.78, -4);

    // Purification zone
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 0.15, 16), new THREE.MeshStandardMaterial({ color: 0x55AA88, transparent: true, opacity: 0.45 })), -14, 0.78, 0);
    addAt(fluidicsGroup, makeSprite("PURIFY", "#2d7a55", 11), -14, 2.0, 0);

    // Channel: lysis → purification
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4, 0.08, 0.4), fluidicChannelMat), -18, 0.78, 0);

    // RPA amplification chambers (3 grouped)
    const rpaX = -7, rpaLabels = ["RPA-A", "RPA-B", "RPA-C"];
    for (let i = 0; i < 3; i++) {
      const rz = -5 + i * 5;
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.15, 12), new THREE.MeshStandardMaterial({ color: 0xCC8844, transparent: true, opacity: 0.45 })), rpaX, 0.78, rz);
      addAt(fluidicsGroup, makeSprite(rpaLabels[i], "#996633", 10), rpaX, 1.9, rz);
    }
    // Channel: purification → RPA
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(5, 0.08, 0.4), fluidicChannelMat), -10.5, 0.78, 0);
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.08, 10), fluidicChannelMat), rpaX, 0.78, 0);

    // Distribution manifold: RPA → detection grid
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(12, 0.08, 0.4), fluidicChannelMat), (rpaX + gX) / 2 + 2, 0.78, 0);

    // Per-row distribution trunks
    for (let r = 0; r < 3; r++) {
      const rz = gZ + r * sZ;
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(sX * 4 + 4, 0.06, 0.35), fluidicChannelMat), gX + sX * 2, 0.78, rz);
      if (r !== 1) {
        addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.06, Math.abs(rz)), fluidicChannelMat), gX - 1, 0.78, rz / 2);
      }
    }

    // Detection chambers (thin walls around each WE)
    flatPads.forEach((_, idx) => {
      const row = Math.floor(idx / 5), col = idx % 5;
      const px = gX + col * sX, pz = gZ + row * sZ;
      const cw = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 0.4, 24, 1, true), chamberWallMat);
      cw.position.set(px, 0.82, pz); fluidicsGroup.add(cw);
    });

    // Waste reservoir (right side)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4, 0.12, 8), new THREE.MeshStandardMaterial({ color: 0x996666, transparent: true, opacity: 0.35 })), 31, 0.78, 0);
    addAt(fluidicsGroup, makeSprite("WASTE", "#884444", 12), 31, 2.0, 0);
    // Channel: grid → waste
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(2, 0.06, 0.35), fluidicChannelMat), 29, 0.78, 0);

    // ── Detection pads with target labels ──
    flatPads.forEach((target, idx) => {
      const row = Math.floor(idx / 5), col = idx % 5;
      const px = gX + col * sX, pz = gZ + row * sZ;
      padPositions.push({ x: px, z: pz, target, row, col });
      const drug = targetDrug(target);
      const padColor = DRUG_HEX[drug] || 0x888888;

      // Recessed well wall
      const ww = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.5, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9, side: THREE.DoubleSide }));
      ww.position.set(px, 0.35, pz); chipGroup.add(ww);

      // LIG floor with porous texture hints
      const ligFloor = addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.06, 24), wellMat), px, 0.11, pz);
      // Tiny pore dots on LIG floor
      for (let p = 0; p < 6; p++) {
        const a2 = Math.random() * Math.PI * 2, rd2 = Math.random() * 1.1;
        addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.07, 4), new THREE.MeshStandardMaterial({ color: 0x111111 })), px + Math.cos(a2) * rd2, 0.15, pz + Math.sin(a2) * rd2);
      }

      // Drug glow ring
      const gr = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.5, 24), new THREE.MeshStandardMaterial({ color: padColor, emissive: padColor, emissiveIntensity: 0.35, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
      gr.rotation.x = -Math.PI / 2; gr.position.set(px, 0.15, pz); chipGroup.add(gr);

      // AuNP specks (smaller than before — 200-500 µm scale)
      for (let i = 0; i < 10; i++) {
        const a = Math.random() * Math.PI * 2, rd = Math.random() * 1.1;
        const au = new THREE.Mesh(new THREE.SphereGeometry(0.04, 5, 5, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9, roughness: 0.1 }));
        au.position.set(px + Math.cos(a) * rd, 0.14, pz + Math.sin(a) * rd); chipGroup.add(au);
      }

      // Target label sprite (gene + mutation)
      const shortLabel = target.replace("_", "\n");
      const lbl = makeSprite(target.split("_").join(" "), "#ffffff", 11);
      lbl.position.set(px, 1.4, pz);
      chipGroup.add(lbl);

      // Raycast mesh
      const pm = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 1.0, 16), new THREE.MeshBasicMaterial({ visible: false }));
      pm.position.set(px, 0.5, pz); pm.userData = { target, drug, idx, padColor };
      chipGroup.add(pm); padMeshes.push(pm);
    });

    // ── Counter electrode — larger, LIG textured, labeled ──
    const ceX = gX + sX * 4 + 5;
    const ceMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.85, metalness: 0.15 });
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.12, 24), ceMat), ceX, 0.66, 0);
    // LIG porous texture on CE
    for (let i = 0; i < 20; i++) {
      const tz = (Math.random() - 0.5) * 22, tx = (Math.random() - 0.5) * 2.8;
      addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.13, 4), new THREE.MeshStandardMaterial({ color: 0x111111 })), ceX + tx, 0.73, tz);
    }
    addAt(chipGroup, makeSprite("CE (LIG)", "#aaaaaa", 11), ceX, 1.6, 0);
    for (let r = 0; r < 3; r++) {
      addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.05, 0.15), goldMat), ceX, 0.73, gZ + r * sZ);
    }

    // ── Ag/AgCl reference electrode strip ──
    const reX = gX + sX * 4 + 9;
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.1, 20), agMat), reX, 0.66, 0);
    // Slight AgCl tint overlay
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.02, 20), new THREE.MeshStandardMaterial({ color: 0xE8E8F0, transparent: true, opacity: 0.5, metalness: 0.3 })), reX, 0.72, 0);
    addAt(chipGroup, makeSprite("RE (Ag/AgCl)", "#8888aa", 10), reX, 1.6, 0);

    // ── Contact pads + thin traces ──
    const edgeZ = 17;
    for (let c = 0; c < 5; c++) {
      for (let r = 0; r < 3; r++) {
        const pi = r * 5 + c;
        const pad = padPositions[pi];
        if (!pad) continue;
        const cx = pad.x + (r - 1) * 0.6;
        addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.12, 2.2), goldMat), cx, 0.66, edgeZ);
        // Thin trace (~200-500 µm visual width = 0.12 units)
        const traceLen = edgeZ - 1.5 - pad.z;
        if (traceLen > 0.5) {
          addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, traceLen), channelMat), cx, 0.64, pad.z + traceLen / 2);
        }
        const dx = cx - pad.x;
        if (Math.abs(dx) > 0.1) {
          addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(Math.abs(dx) + 0.1, 0.03, 0.12), channelMat), (cx + pad.x) / 2, 0.64, pad.z);
        }
      }
    }
    // CE + RE contacts
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.12, 2.2), goldMat), ceX, 0.66, edgeZ);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, edgeZ - 12), channelMat), ceX, 0.64, (edgeZ + 12) / 2 - 3);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.12, 2.2), goldMat), reX, 0.66, edgeZ);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, edgeZ - 10), channelMat), reX, 0.64, (edgeZ + 10) / 2 - 2);

    // Insertion guide
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(58, 0.25, 0.35), new THREE.MeshStandardMaterial({ color: 0xB8860B, roughness: 0.4, metalness: 0.3 })), 10, 0.66, 17.8);

    // ══════════════════════════════════════════════════════════
    // MODE 2: CROSS-SECTION — Accurate proportions + porosity
    // ══════════════════════════════════════════════════════════
    const crossGroup = new THREE.Group();
    crossGroup.visible = false;
    scene.add(crossGroup);

    const secR = 3.5;

    // Kapton substrate — THICK (125 µm → 4.0 units, dominant layer)
    const kaptonXS = new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 4.0, 32), new THREE.MeshStandardMaterial({ color: 0xD4A76A, roughness: 0.35 }));
    kaptonXS.position.y = 2.0; kaptonXS.castShadow = true; crossGroup.add(kaptonXS);

    // LIG layer — THIN (20-50 µm → 0.8 units) with dramatic porosity
    const ligXS = new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 0.8, 32), new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.9 }));
    ligXS.position.y = 4.4; ligXS.castShadow = true; crossGroup.add(ligXS);

    // Heavy pore texture (foam/sponge-like) — many varied pores
    const poreMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 1 });
    for (let i = 0; i < 80; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.2);
      const pSize = 0.03 + Math.random() * 0.08;
      const pDepth = 0.05 + Math.random() * 0.3;
      const p = new THREE.Mesh(new THREE.CylinderGeometry(pSize, pSize * 0.7, pDepth, 5), poreMat);
      p.position.set(Math.cos(a) * rr, 4.8 + Math.random() * 0.05, Math.sin(a) * rr);
      p.rotation.x = (Math.random() - 0.5) * 0.4;
      p.rotation.z = (Math.random() - 0.5) * 0.4;
      crossGroup.add(p);
    }
    // Lateral pore channels visible on cylinder side
    for (let i = 0; i < 30; i++) {
      const a = Math.random() * Math.PI * 2;
      const py = 4.1 + Math.random() * 0.6;
      const p = new THREE.Mesh(new THREE.SphereGeometry(0.04 + Math.random() * 0.06, 4, 4), poreMat);
      p.position.set(Math.cos(a) * (secR - 0.02), py, Math.sin(a) * (secR - 0.02));
      crossGroup.add(p);
    }

    // AuNP base layer — very thin
    addAt(crossGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 0.06, 32), new THREE.MeshStandardMaterial({ color: 0xFFD700, roughness: 0.2, metalness: 0.8 })), 0, 4.83, 0);

    // Discrete AuNP hemispheres — SMALL (10-50 nm → 0.06 radius)
    const auSurf = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9, roughness: 0.1 });
    for (let i = 0; i < 50; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.15);
      const au = new THREE.Mesh(new THREE.SphereGeometry(0.04 + Math.random() * 0.04, 6, 6, 0, Math.PI * 2, 0, Math.PI / 2), auSurf);
      au.position.set(Math.cos(a) * rr, 4.86, Math.sin(a) * rr); crossGroup.add(au);
    }

    // MCH backfill — dense carpet of short individual stubs
    const mchMat = new THREE.MeshStandardMaterial({ color: 0x999999, roughness: 0.6 });
    for (let i = 0; i < 80; i++) {
      const mx = (Math.random() - 0.5) * (secR * 2 - 0.6), mz = (Math.random() - 0.5) * (secR * 2 - 0.6);
      if (mx * mx + mz * mz > (secR - 0.3) ** 2) continue;
      const mh = 0.15 + Math.random() * 0.1; // MCH ~0.8 nm → short
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, mh, 3), mchMat);
      m.position.set(mx, 4.86 + mh / 2, mz);
      m.rotation.x = (Math.random() - 0.5) * 0.15;
      m.rotation.z = (Math.random() - 0.5) * 0.15;
      crossGroup.add(m);
    }

    const baseY = 4.89;

    // ── ssDNA reporters: floppy, wavy, variable conformations ──
    const strandMat = new THREE.MeshStandardMaterial({ color: 0x66c2a5 });
    const mbMat = new THREE.MeshStandardMaterial({ color: 0x3288bd, emissive: 0x1a4470, emissiveIntensity: 0.3 });
    const thiolMat = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.8, roughness: 0.15 });
    const cutMat = new THREE.MeshStandardMaterial({ color: 0x3d7a63 });

    const reporters = [];
    for (let i = 0; i < 24; i++) {
      const x = (Math.random() - 0.5) * (secR * 2 - 1.2);
      const z = (Math.random() - 0.5) * (secR * 2 - 1.2);
      if (x * x + z * z > (secR - 0.6) ** 2) continue;

      const h = 1.8 + Math.random() * 0.7;
      // Random "bent-ness" — some reporters bend toward surface (MB near Au = electron transfer)
      const bentToward = Math.random() < 0.25;
      const bendX = bentToward ? (Math.random() - 0.5) * 0.8 : (Math.random() - 0.5) * 0.3;
      const bendZ = bentToward ? (Math.random() - 0.5) * 0.8 : (Math.random() - 0.5) * 0.3;

      // Thiol-Au bond marker (yellow-orange S atom, larger for visibility)
      const th = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 8), thiolMat);
      th.position.set(x, baseY, z); crossGroup.add(th);
      // Tiny sulfur indicator (darker gold)
      const sAtom = new THREE.Mesh(new THREE.SphereGeometry(0.04, 6, 6), new THREE.MeshStandardMaterial({ color: 0xCCA000 }));
      sAtom.position.set(x, baseY + 0.06, z); crossGroup.add(sAtom);

      // Floppy ssDNA: 6 segments with random coil trajectory
      const segs = [];
      const nSeg = 6, segH = h / nSeg;
      let cx = x, cz = z, cy = baseY;
      for (let s = 0; s < nSeg; s++) {
        // Random walk for floppy single-stranded DNA
        const waveMag = bentToward ? 0.12 + s * 0.05 : 0.06 + Math.random() * 0.08;
        cx += Math.sin(i * 2.3 + s * 1.7) * waveMag + bendX * (s / nSeg);
        cz += Math.cos(i * 1.9 + s * 1.3) * waveMag + bendZ * (s / nSeg);
        const segMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, segH, 4), strandMat.clone());
        segMesh.position.set(cx, cy + segH / 2, cz);
        // Tilt segments to create wavy path
        segMesh.rotation.x = bendX * 0.3 + Math.sin(s * 1.4 + i) * 0.15;
        segMesh.rotation.z = bendZ * 0.3 + Math.cos(s * 1.1 + i * 0.7) * 0.15;
        segMesh.userData._rx0 = segMesh.rotation.x;
        segMesh.userData._rz0 = segMesh.rotation.z;
        crossGroup.add(segMesh); segs.push(segMesh);
        cy += segH;
      }

      // MB sphere at tip
      const mbY = bentToward ? baseY + h * 0.4 : baseY + h;
      const mb = new THREE.Mesh(new THREE.SphereGeometry(0.14, 10, 10), mbMat.clone());
      mb.position.set(cx, mbY, cz);
      crossGroup.add(mb);

      // Cut stub (for Cas12a) — random cleavage time for progressive animation
      const cleavageTime = 2 + Math.random() * 25; // minutes when this reporter gets cleaved
      const cutH = h * (0.2 + Math.random() * 0.25);
      const st = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, cutH, 4), cutMat);
      st.position.set(x, baseY + cutH / 2, z);
      st.rotation.x = bendX * 0.1; st.rotation.z = bendZ * 0.1;
      st.visible = false; crossGroup.add(st);

      // Detached MB fragment (floats away when cleaved)
      const detachedMB = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 8), new THREE.MeshStandardMaterial({ color: 0x3288bd, transparent: true, opacity: 0.7 }));
      detachedMB.position.set(cx + (Math.random() - 0.5) * 0.5, baseY + h + 1.5 + Math.random() * 2, cz + (Math.random() - 0.5) * 0.5);
      detachedMB.visible = false; crossGroup.add(detachedMB);

      reporters.push({ segs, mb, stub: st, thiol: th, sAtom, detachedMB, cleavageTime, bentToward, baseX: x, baseZ: z });
    }

    // ── Solution-phase elements ──
    // Cas12a:crRNA RNP complexes (green, ~7 nm → shown as small polyhedra)
    const rnpMat = new THREE.MeshStandardMaterial({ color: 0x33AA55, emissive: 0x115522, emissiveIntensity: 0.2, transparent: true, opacity: 0.75 });
    const rnps = [];
    for (let i = 0; i < 8; i++) {
      const rx = (Math.random() - 0.5) * (secR * 1.4);
      const rz = (Math.random() - 0.5) * (secR * 1.4);
      const ry = baseY + 3.0 + Math.random() * 3.0;
      const rnp = new THREE.Mesh(new THREE.IcosahedronGeometry(0.18, 0), rnpMat);
      rnp.position.set(rx, ry, rz);
      rnp.visible = false; crossGroup.add(rnp);
      rnps.push(rnp);
    }

    // RPA amplicon dsDNA (double helix representation — shown as cylinder pair)
    const ampliconGroup = new THREE.Group();
    ampliconGroup.visible = false;
    const helixMat1 = new THREE.MeshStandardMaterial({ color: 0x5577CC, transparent: true, opacity: 0.6 });
    const helixMat2 = new THREE.MeshStandardMaterial({ color: 0xCC5577, transparent: true, opacity: 0.6 });
    for (let t = 0; t < 30; t++) {
      const angle = t * 0.4;
      const y = baseY + 5.5 + t * 0.08;
      const b1 = new THREE.Mesh(new THREE.SphereGeometry(0.06, 4, 4), helixMat1);
      b1.position.set(Math.cos(angle) * 0.2, y, Math.sin(angle) * 0.2);
      ampliconGroup.add(b1);
      const b2 = new THREE.Mesh(new THREE.SphereGeometry(0.06, 4, 4), helixMat2);
      b2.position.set(Math.cos(angle + Math.PI) * 0.2, y, Math.sin(angle + Math.PI) * 0.2);
      ampliconGroup.add(b2);
    }
    crossGroup.add(ampliconGroup);

    // ── Scale bars (left side of cross-section) ──
    const scaleMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    // Kapton scale bar
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 4.0, 0.03), scaleMat), -secR - 0.8, 2.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), scaleMat), -secR - 0.8, 0.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), scaleMat), -secR - 0.8, 4.0, 0);
    addAt(crossGroup, makeSprite("125 µm", "#777777", 9), -secR - 1.6, 2.0, 0);
    // LIG scale bar
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.8, 0.03), scaleMat), -secR - 0.5, 4.4, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), scaleMat), -secR - 0.5, 4.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), scaleMat), -secR - 0.5, 4.8, 0);
    addAt(crossGroup, makeSprite("20-50 µm", "#777777", 8), -secR - 1.5, 4.4, 0);

    // ── Camera orbit ──
    let orbit = { theta: 0.3, phi: -0.45, dist: 90, target: new THREE.Vector3(10, 0, 2) };
    let tgtOrbit = { theta: 0.3, phi: -0.45, dist: 90, target: new THREE.Vector3(10, 0, 2) };
    let isDragging = false, prevMouse = { x: 0, y: 0 };
    const canvas = renderer.domElement;
    const raycaster = new THREE.Raycaster();

    const onDown = (e) => { isDragging = true; const p = e.touches ? e.touches[0] : e; prevMouse = { x: p.clientX, y: p.clientY }; };
    const onUp = () => { isDragging = false; };
    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      if (!isDragging) {
        const mx = ((p.clientX - rect.left) / rect.width) * 2 - 1;
        const my = -((p.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
        const hits = raycaster.intersectObjects(padMeshes);
        if (hits.length > 0) {
          const ud = hits[0].object.userData;
          canvas.style.cursor = "pointer";
          stateRef.current._hovIdx = ud.idx;
          setTooltipInfo({ target: ud.target, drug: ud.drug });
          setTooltipPos({ x: p.clientX - rect.left, y: p.clientY - rect.top });
        } else {
          canvas.style.cursor = "grab";
          stateRef.current._hovIdx = -1;
          setTooltipInfo(null);
        }
      }
      if (!isDragging) return;
      const dx = p.clientX - prevMouse.x, dy = p.clientY - prevMouse.y;
      tgtOrbit.theta += dx * 0.005;
      tgtOrbit.phi = Math.max(-1.3, Math.min(-0.08, tgtOrbit.phi + dy * 0.005));
      prevMouse = { x: p.clientX, y: p.clientY };
    };
    const onWheel = (e) => { e.preventDefault(); tgtOrbit.dist = Math.max(15, Math.min(150, tgtOrbit.dist + e.deltaY * 0.08)); };
    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
      const hits = raycaster.intersectObjects(padMeshes);
      if (hits.length > 0) stateRef.current._selectPad(hits[0].object.userData.idx, hits[0].object.userData.target);
    };

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("touchstart", onDown, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("touchmove", onMove, { passive: true });
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("click", onClick);

    // ── Animation loop ──
    let frameId, time = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      time += 0.016;
      orbit.theta += (tgtOrbit.theta - orbit.theta) * 0.07;
      orbit.phi += (tgtOrbit.phi - orbit.phi) * 0.07;
      orbit.dist += (tgtOrbit.dist - orbit.dist) * 0.07;
      orbit.target.lerp(tgtOrbit.target instanceof THREE.Vector3 ? tgtOrbit.target : new THREE.Vector3(tgtOrbit.target.x, tgtOrbit.target.y, tgtOrbit.target.z), 0.07);
      camera.position.x = orbit.target.x + orbit.dist * Math.sin(orbit.theta) * Math.cos(orbit.phi);
      camera.position.y = orbit.target.y + orbit.dist * Math.sin(-orbit.phi);
      camera.position.z = orbit.target.z + orbit.dist * Math.cos(orbit.theta) * Math.cos(orbit.phi);
      camera.lookAt(orbit.target);

      // Animate reporters (floppy thermal motion)
      reporters.forEach((r, i) => {
        r.segs.forEach((seg, si) => {
          if (seg.visible) {
            seg.rotation.x = seg.userData._rx0 + Math.sin(time * 1.9 + i * 0.7 + si * 0.5) * 0.06;
            seg.rotation.z = seg.userData._rz0 + Math.cos(time * 1.5 + i * 1.1 + si * 0.3) * 0.06;
          }
        });
        if (r.mb.visible) r.mb.material.emissiveIntensity = 0.2 + 0.15 * Math.sin(time * 3.14 + i);
        // Floating detached MB drift upward slowly
        if (r.detachedMB.visible) {
          r.detachedMB.position.y += Math.sin(time * 0.5 + i) * 0.002;
          r.detachedMB.position.x += Math.sin(time * 0.3 + i * 1.3) * 0.001;
        }
      });

      // Animate RNP complexes (gentle drift)
      rnps.forEach((rnp, i) => {
        if (rnp.visible) {
          rnp.rotation.y += 0.01;
          rnp.position.y += Math.sin(time * 0.7 + i * 2) * 0.003;
          rnp.position.x += Math.cos(time * 0.4 + i * 1.5) * 0.002;
        }
      });

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = container.clientWidth, h2 = Math.round(w * 9 / 16);
      container.style.height = h2 + "px";
      renderer.setSize(w, h2);
      camera.aspect = w / h2; camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    stateRef.current = {
      chipGroup, crossGroup, fluidicsGroup, reporters, padMeshes, rnps, ampliconGroup, orbit, tgtOrbit,
      _hovIdx: -1,
      _selectPad: (idx, target) => { setSelectedPad({ idx, target }); setMode(2); },
      _toMode1: () => { setMode(1); setSelectedPad(null); setCas12aActive(false); },
    };

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
      window.removeEventListener("resize", onResize);
      scene.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) { if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose()); else obj.material.dispose(); }
      });
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
    };
  }, []);

  // Mode transitions
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    if (mode === 1) {
      s.chipGroup.visible = true; s.crossGroup.visible = false;
      s.tgtOrbit.dist = 90; s.tgtOrbit.theta = 0.3; s.tgtOrbit.phi = -0.45;
      s.tgtOrbit.target = new THREE.Vector3(10, 0, 2);
    } else {
      s.chipGroup.visible = false; s.crossGroup.visible = true;
      s.tgtOrbit.dist = 18; s.tgtOrbit.theta = 0.4; s.tgtOrbit.phi = -0.3;
      s.tgtOrbit.target = new THREE.Vector3(0, 4.5, 0);
    }
  }, [mode, selectedPad]);

  // Microfluidics visibility toggle
  useEffect(() => {
    const s = stateRef.current;
    if (!s?.fluidicsGroup) return;
    s.fluidicsGroup.visible = showMicrofluidics;
  }, [showMicrofluidics]);

  // Progressive Cas12a cleavage — tied to incubation time
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.reporters.forEach(r => {
      if (cas12aActive) {
        const cleaved = incubationMin >= r.cleavageTime;
        r.segs.forEach(seg => { seg.visible = !cleaved; });
        r.mb.visible = !cleaved;
        r.stub.visible = cleaved;
        r.detachedMB.visible = cleaved;
      } else {
        r.segs.forEach(seg => { seg.visible = true; });
        r.mb.visible = true;
        r.stub.visible = false;
        r.detachedMB.visible = false;
      }
    });
    // Show/hide RNP complexes and amplicon
    s.rnps.forEach(rnp => { rnp.visible = cas12aActive; });
    s.ampliconGroup.visible = cas12aActive;
  }, [cas12aActive, incubationMin]);

  // ── Computed data for selected pad ──
  const selDrug = selectedPad ? targetDrug(selectedPad.target) : null;
  const selEff = selectedPad ? getEfficiency(selectedPad.target) : null;
  const selStrat = selectedPad ? targetStrategy(selectedPad.target) : null;
  const selR = selectedPad ? results.find(x => x.label === selectedPad.target) : null;
  const selDisc = selR?.disc && selR.disc < 900 ? selR.disc : null;
  const selScore = selR?.ensembleScore || selR?.score || null;
  const selCoAmp = (() => {
    if (!selectedPad) return null;
    const gs = [["rpoB_H445Y","rpoB_H445D"],["rpoB_S450L","rpoB_S450W"],["katG_S315T","katG_S315N"],["embB_M306V","embB_M306I"]];
    const g = gs.find(g => g.includes(selectedPad.target));
    return g ? g.find(x => x !== selectedPad.target) : null;
  })();

  const deltaI = selectedPad && computeGamma && echemGamma0_mol ? (() => {
    const G = computeGamma(incubationMin * 60, getEfficiency(selectedPad.target), echemKtrans);
    return ((1 - G / echemGamma0_mol) * 100).toFixed(1);
  })() : null;

  // ── Curve data generation ──
  const curveData = selectedPad ? (() => {
    const G_after = computeGamma(incubationMin * 60, getEfficiency(selectedPad.target), echemKtrans);
    if (curveMode === "SWV") return { before: miniSWV(echemGamma0_mol, echemGamma0_mol), after: miniSWV(G_after, echemGamma0_mol) };
    if (curveMode === "DPV") return { before: miniDPV(echemGamma0_mol, echemGamma0_mol), after: miniDPV(G_after, echemGamma0_mol) };
    if (curveMode === "EIS") return { before: miniEIS(echemGamma0_mol, echemGamma0_mol), after: miniEIS(G_after, echemGamma0_mol) };
    return null;
  })() : null;

  const svgPathVolt = (pts, w, h) => {
    if (!pts?.length) return "";
    const maxI = Math.max(...pts.map(p => Math.abs(p.I)), 0.001);
    return pts.map((p, i) => {
      const x = (i / (pts.length - 1)) * w;
      const y = h - (Math.abs(p.I) / maxI) * h * 0.85 - h * 0.05;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  const svgPathEIS = (pts, w, h) => {
    if (!pts?.length) return "";
    const maxZr = Math.max(...pts.map(p => p.Zr), 1);
    const maxZi = Math.max(...pts.map(p => p.Zi), 1);
    return pts.map((p, i) => {
      const x = ((p.Zr - 40) / (maxZr - 40 + 1)) * w * 0.9 + w * 0.05;
      const y = h - (p.Zi / (maxZi + 1)) * h * 0.85 - h * 0.05;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  // Cleavage progress for display
  const cleavedCount = cas12aActive ? reporters.length > 0 ? Math.round(reporters.length * Math.min(1, incubationMin / 28)) : 0 : 0;

  return (
    <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#F8F9FA" }}>
      <div ref={mountRef} style={{ width: "100%", minHeight: 280 }} />

      {/* ═══ MODE 1: CHIP OVERVIEW OVERLAY ═══ */}
      {mode === 1 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            GUARD MDR-TB Diagnostic Chip
          </div>
          <div style={{ position: "absolute", top: 14, right: 16, display: "flex", gap: 8, alignItems: "center" }}>
            <button
              onClick={() => setShowMicrofluidics(!showMicrofluidics)}
              style={{
                fontSize: 9, fontWeight: 600, padding: "3px 10px", borderRadius: 4,
                border: "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
                background: showMicrofluidics ? "rgba(51,136,170,0.15)" : "rgba(255,255,255,0.88)",
                color: showMicrofluidics ? "#1a6680" : "#6B7280",
              }}
            >
              {showMicrofluidics ? "Hide" : "Show"} Microfluidics
            </button>
            <span style={{ fontSize: 10, color: "#6B7280", textShadow: "0 1px 2px rgba(255,255,255,0.8)" }}>
              Click pad to inspect
            </span>
          </div>

          {/* Drug class legend */}
          <div style={{ position: "absolute", bottom: 12, left: 16, display: "flex", gap: 6, flexWrap: "wrap" }}>
            {Object.entries(DRUG_CSS).map(([d, c]) => (
              <span key={d} style={{ fontSize: 9, fontWeight: 700, fontFamily: MONO, display: "flex", alignItems: "center", gap: 3, background: "rgba(255,255,255,0.88)", padding: "2px 7px", borderRadius: 4 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: c, display: "inline-block" }} />{d}
              </span>
            ))}
          </div>

          {/* Chip dimensions + layer info */}
          <div style={{ position: "absolute", bottom: 12, right: 16, display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 9, color: "#6B7280", background: "rgba(255,255,255,0.88)", padding: "2px 8px", borderRadius: 4, fontFamily: MONO }}>
              65 × 35 mm · Kapton + LIG
            </span>
            <span style={{ fontSize: 9, color: "#6B7280", background: "rgba(255,255,255,0.88)", padding: "2px 8px", borderRadius: 4, fontFamily: MONO, display: "flex", alignItems: "center", gap: 3 }}>
              <span style={{ width: 6, height: 6, background: "rgba(46,125,50,0.3)", display: "inline-block" }} />passivation
            </span>
          </div>
        </>
      )}

      {/* ═══ MODE 2: CROSS-SECTION OVERLAY ═══ */}
      {mode === 2 && selectedPad && (
        <>
          {/* Enhanced info panel with crRNA, PAM, amplicon */}
          <div style={{ position: "absolute", top: 12, left: 16, background: "rgba(255,255,255,0.95)", padding: "12px 16px", borderRadius: 8, border: "1px solid #E3E8EF", maxWidth: 280, backdropFilter: "blur(8px)" }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING }}>
              {selectedPad.target} · <span style={{ color: DRUG_CSS[selDrug] || "#888" }}>{selDrug}</span>
            </div>
            <div style={{ fontSize: 10, color: "#6B7280", fontFamily: MONO, marginTop: 4, lineHeight: 1.7 }}>
              S_eff = {selEff?.toFixed(3)}{selScore != null && ` · Score = ${selScore.toFixed(2)}`}{selDisc ? ` · D = ${selDisc.toFixed(1)}×` : ""}<br />
              {selStrat} detection{selCoAmp && ` · co-amplicon: ${selCoAmp}`}
              {selR?.hasPrimers && <span style={{ color: "#16A34A" }}> · RPA primers</span>}
            </div>
            {/* crRNA / PAM / amplicon details */}
            {selR && (
              <div style={{ fontSize: 9, color: "#9CA3AF", fontFamily: MONO, marginTop: 4, lineHeight: 1.6, borderTop: "1px solid #E3E8EF", paddingTop: 4 }}>
                {selR.spacer && <div>crRNA: <span style={{ color: "#374151", letterSpacing: "0.5px" }}>{selR.spacer.slice(0, 20)}{selR.spacer.length > 20 ? "…" : ""}</span></div>}
                {selR.pam && <div>PAM: <span style={{ color: "#374151" }}>{selR.pam}</span>{selR.pamVariant && <span> ({selR.pamVariant})</span>}{selR.isCanonicalPam === true && <span style={{ color: "#16A34A" }}> canonical</span>}</div>}
                {selR.amplicon && <div>Amplicon: <span style={{ color: "#374151" }}>{selR.amplicon} bp</span></div>}
                {deltaI != null && <div>Expected ΔI%: <span style={{ color: "#2563EB", fontWeight: 700 }}>{deltaI}%</span> @ {incubationMin} min</div>}
              </div>
            )}
          </div>

          <button onClick={() => stateRef.current?._toMode1()} style={{ position: "absolute", top: 12, right: 16, fontSize: 11, fontWeight: 600, padding: "7px 16px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}>
            ← Back to chip
          </button>

          {/* Layer labels with scale indicators */}
          <div style={{ position: "absolute", left: 16, top: "38%", transform: "translateY(-50%)", display: "flex", flexDirection: "column", gap: 4 }}>
            {[
              { label: "MB (E° = −0.22 V, n = 2)", color: "#3288bd", dim: "~1 nm" },
              { label: "ssDNA reporter (12–20 nt)", color: "#66c2a5", dim: "4–7 nm" },
              { label: "Thiol–Au bond (S–Au)", color: "#CCA000", dim: "" },
              { label: "MCH backfill (C6-OH)", color: "#999999", dim: "~0.8 nm" },
              { label: "AuNP hemispheres", color: "#FFD700", dim: "10–50 nm" },
              { label: "LIG (porous, ~340 m²/g)", color: "#2D2D2D", dim: "20–50 µm" },
              { label: "Kapton HN substrate", color: "#D4A76A", dim: "125 µm" },
            ].map(l => (
              <div key={l.label} style={{ fontSize: 9, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.92)", padding: "3px 8px", borderRadius: 4, borderLeft: `3px solid ${l.color}`, lineHeight: 1.3 }}>
                ← {l.label}{l.dim && <span style={{ color: "#9CA3AF" }}> ({l.dim})</span>}
              </div>
            ))}
            <div style={{ fontSize: 8, color: "#B0B0B0", fontFamily: MONO, fontStyle: "italic", marginTop: 2, paddingLeft: 4 }}>
              ⚠ vertical scale exaggerated
            </div>
          </div>

          {/* ── Electrochemistry curves panel (DPV / SWV / EIS) ── */}
          {curveData && (
            <div style={{ position: "absolute", bottom: 60, right: 16, background: "rgba(255,255,255,0.95)", padding: "8px 12px", borderRadius: 8, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)", minWidth: 180 }}>
              {/* Curve mode tabs */}
              <div style={{ display: "flex", gap: 2, marginBottom: 4 }}>
                {["DPV", "SWV", "EIS"].map(m => (
                  <button key={m} onClick={() => setCurveMode(m)} style={{
                    fontSize: 8, fontWeight: curveMode === m ? 700 : 500, fontFamily: MONO,
                    padding: "2px 8px", borderRadius: 3, border: "none", cursor: "pointer",
                    background: curveMode === m ? "#2563EB" : "#F3F4F6",
                    color: curveMode === m ? "#fff" : "#6B7280",
                  }}>{m}</button>
                ))}
              </div>

              <svg width={160} height={80} viewBox="0 0 160 80">
                {curveMode !== "EIS" ? (
                  <>
                    <path d={svgPathVolt(curveData.before, 160, 80)} fill="none" stroke="#93C5FD" strokeWidth="1.5" />
                    <path d={svgPathVolt(curveData.after, 160, 80)} fill="none" stroke="#2563EB" strokeWidth="1.5" />
                    <line x1="0" y1="77" x2="160" y2="77" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="76" fontSize="6" fill="#9CA3AF">−0.05</text>
                    <text x="128" y="76" fontSize="6" fill="#9CA3AF">−0.40 V</text>
                    {/* MB peak annotation */}
                    <line x1="80" y1="0" x2="80" y2="80" stroke="#E5E7EB" strokeWidth="0.5" strokeDasharray="2,2" />
                    <text x="82" y="8" fontSize="5" fill="#9CA3AF">−0.22 V</text>
                  </>
                ) : (
                  <>
                    <path d={svgPathEIS(curveData.before, 160, 80)} fill="none" stroke="#93C5FD" strokeWidth="1.5" />
                    <path d={svgPathEIS(curveData.after, 160, 80)} fill="none" stroke="#2563EB" strokeWidth="1.5" />
                    <line x1="0" y1="77" x2="160" y2="77" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="76" fontSize="6" fill="#9CA3AF">Z' (Ω)</text>
                    <line x1="2" y1="0" x2="2" y2="77" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="4" y="8" fontSize="6" fill="#9CA3AF">−Z'' (Ω)</text>
                  </>
                )}
              </svg>

              {/* Legend + parameters */}
              <div style={{ fontSize: 8, color: "#9CA3AF", fontFamily: MONO, display: "flex", gap: 10, marginTop: 2 }}>
                <span><span style={{ color: "#93C5FD" }}>—</span> baseline</span>
                <span><span style={{ color: "#2563EB" }}>—</span> t={incubationMin}m</span>
              </div>
              <div style={{ fontSize: 7, color: "#C0C0C0", fontFamily: MONO, marginTop: 2 }}>
                {curveMode === "SWV" && "amp=25 mV · f=25 Hz · step=4.375 mV"}
                {curveMode === "DPV" && "amp=50 mV · width=50 ms · step=4.375 mV"}
                {curveMode === "EIS" && "10⁰–10⁶ Hz · 5 mV AC · at E°(MB)"}
              </div>
            </div>
          )}

          {/* ── Cas12a controls + incubation slider ── */}
          <div style={{ position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 12, alignItems: "center", background: "rgba(255,255,255,0.95)", padding: "8px 18px", borderRadius: 8, border: "1px solid #E3E8EF", flexWrap: "wrap", justifyContent: "center", backdropFilter: "blur(8px)" }}>
            <button onClick={() => setCas12aActive(!cas12aActive)} style={{
              fontSize: 11, fontWeight: 700, padding: "7px 16px", borderRadius: 6, cursor: "pointer", fontFamily: MONO,
              background: cas12aActive ? "#DC2626" : "#16A34A", color: "#fff", border: "none", transition: "background 0.2s",
            }}>
              {cas12aActive ? "Reset reporters" : "Activate Cas12a"}
            </button>
            {cas12aActive && (
              <div style={{ fontSize: 10, fontFamily: MONO, color: "#6B7280", display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontWeight: 700, color: "#2563EB" }}>ΔI% = {deltaI}%</span>
                <span style={{ color: "#9CA3AF" }}>·</span>
                <span style={{ color: "#33AA55" }}>{cleavedCount}/{reporters.length} cleaved</span>
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 9, color: "#6B7280", fontFamily: MONO }}>t =</span>
              <input type="range" min="5" max="60" step="1" value={incubationMin} onChange={e => setIncubationMin(+e.target.value)} style={{ width: 90, accentColor: "#2563EB" }} />
              <span style={{ fontSize: 10, fontWeight: 700, fontFamily: MONO, color: "#374151", minWidth: 36 }}>{incubationMin} min</span>
            </div>
          </div>

          {/* Solution-phase legend (visible when Cas12a active) */}
          {cas12aActive && (
            <div style={{ position: "absolute", right: 16, top: "38%", display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                { label: "Cas12a:crRNA RNP", color: "#33AA55" },
                { label: "Cleaved MB (detached)", color: "#3288bd" },
                { label: "RPA amplicon (dsDNA)", color: "#5577CC" },
              ].map(l => (
                <div key={l.label} style={{ fontSize: 8, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3, borderLeft: `3px solid ${l.color}` }}>
                  {l.label}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Tooltip */}
      {tooltipInfo && mode === 1 && (
        <div style={{ position: "absolute", left: tooltipPos.x + 14, top: tooltipPos.y - 12, pointerEvents: "none", background: "rgba(255,255,255,0.96)", border: "1px solid #E3E8EF", borderRadius: 6, padding: "8px 12px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)", zIndex: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, fontFamily: MONO, color: DRUG_CSS[tooltipInfo.drug] || "#333" }}>{tooltipInfo.target}</div>
          <div style={{ fontSize: 10, color: "#6B7280", marginTop: 2 }}>
            {tooltipInfo.drug} · S_eff = {getEfficiency(tooltipInfo.target).toFixed(3)}
            {(() => { const r = results.find(x => x.label === tooltipInfo.target); return r?.disc && r.disc < 900 ? ` · D = ${r.disc.toFixed(1)}×` : ""; })()}
          </div>
          <div style={{ fontSize: 9, color: "#9CA3AF", marginTop: 1 }}>{targetStrategy(tooltipInfo.target)} detection</div>
        </div>
      )}
    </div>
  );
}
