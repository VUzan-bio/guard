import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS & HELPERS
   ═══════════════════════════════════════════════════════════════ */
const DRUG_HEX = { RIF: 0xef4444, INH: 0xf97316, EMB: 0xeab308, PZA: 0xeab308, FQ: 0xa855f7, AG: 0xec4899, CTRL: 0x22d3ee, OTHER: 0x888888 };
const DRUG_CSS = { RIF: "#ef4444", INH: "#f97316", EMB: "#eab308", PZA: "#eab308", FQ: "#a855f7", AG: "#ec4899", CTRL: "#22d3ee", OTHER: "#888888" };
const addAt = (parent, mesh, x, y, z) => { mesh.position.set(x, y, z); parent.add(mesh); return mesh; };

// ── Electrochemistry curve generators (Nernst-based) ──
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

// Sprite text label helper — sharp, crisp text
const makeSprite = (text, color, size, bold) => {
  const canvas = document.createElement("canvas");
  const dpr = 2;
  canvas.width = 512 * dpr; canvas.height = 64 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = `${bold ? "bold " : ""}${Math.round(size || 18)}px monospace`;
  ctx.fillStyle = color || "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 256, 32);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(size ? size * 0.28 : 5.0, size ? size * 0.07 : 1.2, 1);
  return sprite;
};

// 14-target electrode layout: 7 columns × 2 rows
const ELECTRODE_TARGETS = [
  // Row 0 (top): cols 0-6
  { row: 0, col: 0, target: "IS6110", drug: "CTRL", color: 0x22d3ee, label: "IS6110\nTB ID" },
  { row: 0, col: 1, target: "IS1081", drug: "CTRL", color: 0x22d3ee, label: "IS1081\nTB ID" },
  { row: 0, col: 2, target: "rpoB_S531L", drug: "RIF", color: 0xef4444, label: "rpoB S531L\nRIF" },
  { row: 0, col: 3, target: "rpoB_H526Y", drug: "RIF", color: 0xef4444, label: "rpoB H526Y\nRIF" },
  { row: 0, col: 4, target: "katG_S315T", drug: "INH", color: 0xf97316, label: "katG S315T\nINH" },
  { row: 0, col: 5, target: "inhA_C-15T", drug: "INH", color: 0xf97316, label: "inhA C-15T\nINH" },
  { row: 0, col: 6, target: "embB_M306V", drug: "EMB", color: 0xeab308, label: "embB M306V\nEMB" },
  // Row 1 (bottom): cols 0-6
  { row: 1, col: 0, target: "pncA", drug: "PZA", color: 0xeab308, label: "pncA\nPZA" },
  { row: 1, col: 1, target: "gyrA_D94G", drug: "FQ", color: 0xa855f7, label: "gyrA D94G\nFQ" },
  { row: 1, col: 2, target: "gyrA_A90V", drug: "FQ", color: 0xa855f7, label: "gyrA A90V\nFQ" },
  { row: 1, col: 3, target: "gyrB", drug: "FQ", color: 0xa855f7, label: "gyrB\nFQ" },
  { row: 1, col: 4, target: "rrs_A1401G", drug: "AG", color: 0xec4899, label: "rrs A1401G\nKAN/AMK" },
  { row: 1, col: 5, target: "eis_C-14T", drug: "AG", color: 0xec4899, label: "eis C-14T\nKAN" },
  { row: 1, col: 6, target: "RNaseP", drug: "CTRL", color: 0x22d3ee, label: "RNaseP\nHuman Ctrl" },
];

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function ChipRender3D({ electrodeLayout, targetDrug, targetStrategy, getEfficiency, results, computeGamma, echemTime, echemKtrans, echemGamma0_mol, HEADING, MONO }) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);
  const [mode, setMode] = useState(1); // 1=chip, 2=cross-section, 3=side-profile
  const [selectedPad, setSelectedPad] = useState(null);
  const [cas12aActive, setCas12aActive] = useState(false);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [incubationMin, setIncubationMin] = useState(echemTime);
  const [curveMode, setCurveMode] = useState("SWV");
  const [showMicrofluidics, setShowMicrofluidics] = useState(true);
  const [showPassivation, setShowPassivation] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

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

    // ── Lighting ──
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xFFF5E6, 0.75);
    key.position.set(-30, 50, 40); key.castShadow = true;
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
    // MODE 1: CHIP OVERVIEW — 60×30mm, 7×2 WE array
    // ══════════════════════════════════════════════════════════
    const chipGroup = new THREE.Group();
    scene.add(chipGroup);

    // Materials
    const kaptonMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a843, roughness: 0.25, clearcoat: 0.3, clearcoatRoughness: 0.4 });
    const ligMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.85, metalness: 0.1 });
    const goldMat = new THREE.MeshStandardMaterial({ color: 0xc5a03f, metalness: 0.85, roughness: 0.15 });
    const agMat = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, metalness: 0.7, roughness: 0.2 });
    const cocMat = new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, transparent: true, opacity: 0.22, roughness: 0.1, clearcoat: 0.6, side: THREE.DoubleSide });
    const channelMat = new THREE.MeshStandardMaterial({ color: 0x4488AA, transparent: true, opacity: 0.35, roughness: 0.4 });
    const chamberMat = new THREE.MeshStandardMaterial({ color: 0x336677, transparent: true, opacity: 0.4, roughness: 0.4, side: THREE.DoubleSide });
    const passivMat = new THREE.MeshStandardMaterial({ color: 0x1a2744, transparent: true, opacity: 0.15, side: THREE.DoubleSide, depthWrite: false });

    // ── Kapton substrate: 60 × 30 mm (1 unit = 1 mm) ──
    const chipW = 60, chipD = 30, chipH = 1.25; // 125 µm = 1.25 scaled
    const body = new THREE.Mesh(new THREE.BoxGeometry(chipW, chipH, chipD), kaptonMat);
    body.castShadow = true; body.receiveShadow = true;
    chipGroup.add(body);

    // ── Electrode grid: 7 cols × 2 rows ──
    // Center the grid horizontally with space for CE/RE on right
    const gridOriginX = -16; // left-offset to leave room for CE/RE on right
    const gridOriginZ = -4; // vertical center offset
    const colSpacing = 5; // 5 mm center-to-center
    const rowSpacing = 8; // 8 mm center-to-center
    const weRadius = 1.5; // 3 mm diameter / 2

    const padPositions = [];
    const padMeshes = [];

    // ── LIG patterned area under the grid ──
    const ligPadX = gridOriginX + 3 * colSpacing;
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(38, 0.06, 20), ligMat), ligPadX, chipH / 2 + 0.03, gridOriginZ + rowSpacing / 2);

    // ── Passivation mask with circular openings ──
    const maskShape = new THREE.Shape();
    const mw = 42, mh = 22;
    maskShape.moveTo(-mw / 2, -mh / 2); maskShape.lineTo(mw / 2, -mh / 2);
    maskShape.lineTo(mw / 2, mh / 2); maskShape.lineTo(-mw / 2, mh / 2); maskShape.closePath();

    ELECTRODE_TARGETS.forEach(e => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      const hole = new THREE.Path();
      hole.absarc(px - ligPadX, -(pz - gridOriginZ - rowSpacing / 2), weRadius, 0, Math.PI * 2, false);
      maskShape.holes.push(hole);
    });
    const passivGeo = new THREE.ShapeGeometry(maskShape);
    const passivMesh = new THREE.Mesh(passivGeo, passivMat);
    passivMesh.rotation.x = -Math.PI / 2;
    passivMesh.position.set(ligPadX, chipH / 2 + 0.08, gridOriginZ + rowSpacing / 2);
    chipGroup.add(passivMesh);

    // ── Working Electrodes (14 pads) ──
    const labelSprites = [];
    ELECTRODE_TARGETS.forEach((e, idx) => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      padPositions.push({ x: px, z: pz, ...e, idx });

      // LIG floor (dark, porous)
      addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(weRadius, weRadius, 0.06, 24), ligMat), px, chipH / 2 + 0.06, pz);

      // Pore texture dots on WE
      for (let p = 0; p < 8; p++) {
        const a2 = Math.random() * Math.PI * 2, rd2 = Math.random() * (weRadius - 0.2);
        addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.07, 4), new THREE.MeshStandardMaterial({ color: 0x111111 })),
          px + Math.cos(a2) * rd2, chipH / 2 + 0.1, pz + Math.sin(a2) * rd2);
      }

      // Drug class color ring
      const ringMat = new THREE.MeshStandardMaterial({ color: e.color, emissive: e.color, emissiveIntensity: 0.35, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
      const ring = new THREE.Mesh(new THREE.RingGeometry(weRadius - 0.3, weRadius, 24), ringMat);
      ring.rotation.x = -Math.PI / 2; addAt(chipGroup, ring, px, chipH / 2 + 0.09, pz);

      // AuNP specks (tiny gold hemispheres)
      for (let i = 0; i < 8; i++) {
        const a = Math.random() * Math.PI * 2, rd = Math.random() * (weRadius - 0.3);
        addAt(chipGroup, new THREE.Mesh(new THREE.SphereGeometry(0.03, 5, 5, 0, Math.PI * 2, 0, Math.PI / 2),
          new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9, roughness: 0.1 })),
          px + Math.cos(a) * rd, chipH / 2 + 0.09, pz + Math.sin(a) * rd);
      }

      // Target label sprite
      const lbl = makeSprite(e.target, "#ffffff", 9, true);
      lbl.position.set(px, chipH / 2 + 1.6, pz);
      chipGroup.add(lbl);
      labelSprites.push(lbl);

      // Drug class sub-label
      const drugLbl = makeSprite(e.drug, DRUG_CSS[e.drug] || "#888", 7);
      drugLbl.position.set(px, chipH / 2 + 1.15, pz);
      chipGroup.add(drugLbl);
      labelSprites.push(drugLbl);

      // Raycast hit mesh
      const pm = new THREE.Mesh(new THREE.CylinderGeometry(weRadius, weRadius, 1.0, 16), new THREE.MeshBasicMaterial({ visible: false }));
      pm.position.set(px, chipH / 2 + 0.5, pz);
      pm.userData = { target: e.target, drug: e.drug, idx, padColor: e.color };
      chipGroup.add(pm); padMeshes.push(pm);
    });

    // ── Counter Electrode — 8×8 mm, left of WE grid ──
    const ceX = gridOriginX - 7;
    const ceZ = gridOriginZ + rowSpacing / 2;
    const ceW = 8, ceD = 8;
    const ceMesh = new THREE.Mesh(new THREE.BoxGeometry(ceW, 0.12, ceD), new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.85, metalness: 0.15 }));
    addAt(chipGroup, ceMesh, ceX, chipH / 2 + 0.06, ceZ);
    // CE porous texture
    for (let i = 0; i < 30; i++) {
      const tx = (Math.random() - 0.5) * (ceW - 0.4), tz = (Math.random() - 0.5) * (ceD - 0.4);
      addAt(chipGroup, new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.13, 4), new THREE.MeshStandardMaterial({ color: 0x111111 })),
        ceX + tx, chipH / 2 + 0.13, ceZ + tz);
    }
    addAt(chipGroup, makeSprite("CE", "#aaaaaa", 12, true), ceX, chipH / 2 + 1.8, ceZ);

    // ── Ag/AgCl Reference Electrode — 2×12 mm strip, right of grid ──
    const reX = gridOriginX + 6 * colSpacing + 5;
    const reZ = gridOriginZ + rowSpacing / 2;
    // LIG base
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(2, 0.1, 12), ligMat), reX, chipH / 2 + 0.05, reZ);
    // Silver coating
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(2, 0.02, 12), new THREE.MeshStandardMaterial({ color: 0xE8E8F0, transparent: true, opacity: 0.6, metalness: 0.5 })), reX, chipH / 2 + 0.11, reZ);
    addAt(chipGroup, makeSprite("Ag/AgCl RE", "#8888aa", 9, true), reX, chipH / 2 + 1.8, reZ);

    // ── Contact Pads — 16 pads along bottom edge ──
    const padEdgeZ = chipD / 2 - 1.5;
    const padSize = 1.5;
    const numPads = 16;
    const padStartX = -chipW / 2 + 5;
    const padSpacing = (chipW - 10) / (numPads - 1);
    for (let i = 0; i < numPads; i++) {
      const px = padStartX + i * padSpacing;
      addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(padSize, 0.1, padSize), goldMat), px, chipH / 2 + 0.05, padEdgeZ);
    }
    // Thin LIG traces from each WE to nearest contact pad
    ELECTRODE_TARGETS.forEach((e, idx) => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      const padX = padStartX + idx * padSpacing;
      const traceW = 0.12; // ~300 µm trace width
      // Vertical segment to edge
      const vLen = padEdgeZ - pz - weRadius;
      if (vLen > 0.5) {
        addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(traceW, 0.03, vLen), ligMat),
          px, chipH / 2 + 0.02, pz + weRadius + vLen / 2);
      }
      // Horizontal jog if needed
      const dx = padX - px;
      if (Math.abs(dx) > 0.2) {
        addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(Math.abs(dx), 0.03, traceW), ligMat),
          (px + padX) / 2, chipH / 2 + 0.02, padEdgeZ - 1);
      }
      // Final segment to pad
      addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(traceW, 0.03, 1.5), ligMat),
        padX, chipH / 2 + 0.02, padEdgeZ - 0.5);
    });
    // CE and RE traces to last 2 pads
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, padEdgeZ - ceZ - ceD / 2), ligMat),
      ceX, chipH / 2 + 0.02, ceZ + ceD / 2 + (padEdgeZ - ceZ - ceD / 2) / 2);
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.03, padEdgeZ - reZ - 6), ligMat),
      reX, chipH / 2 + 0.02, reZ + 6 + (padEdgeZ - reZ - 6) / 2);

    // Insertion guide bar
    addAt(chipGroup, new THREE.Mesh(new THREE.BoxGeometry(chipW - 4, 0.2, 0.3),
      new THREE.MeshStandardMaterial({ color: 0xB8860B, roughness: 0.4, metalness: 0.3 })), 0, chipH / 2 + 0.1, chipD / 2 - 0.5);

    // ── Microfluidic overlay group (toggleable) ──
    const fluidicsGroup = new THREE.Group();
    chipGroup.add(fluidicsGroup);

    // COC slab (semi-transparent top layer)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(chipW, 0.25, chipD), cocMat), 0, chipH / 2 + 0.5, 0);

    // Sample inlet port (top center)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.35, 16),
      new THREE.MeshStandardMaterial({ color: 0x3388AA, transparent: true, opacity: 0.55, roughness: 0.3 })), 0, chipH / 2 + 0.75, -chipD / 2 + 3);
    addAt(fluidicsGroup, makeSprite("Sample In", "#1a6680", 10, true), 0, chipH / 2 + 2.2, -chipD / 2 + 3);

    // Lysis chamber (5×5 mm)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(5, 0.15, 5),
      new THREE.MeshStandardMaterial({ color: 0x93C5FD, transparent: true, opacity: 0.3 })), -10, chipH / 2 + 0.5, -chipD / 2 + 5);
    // Zirconia beads
    const beadMat = new THREE.MeshStandardMaterial({ color: 0x4B5563, metalness: 0.6, roughness: 0.3 });
    for (let i = 0; i < 10; i++) {
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.SphereGeometry(0.2, 6, 6), beadMat),
        -10 + (Math.random() - 0.5) * 4, chipH / 2 + 0.7, -chipD / 2 + 5 + (Math.random() - 0.5) * 4);
    }
    addAt(fluidicsGroup, makeSprite("Lysis", "#1a6680", 9), -10, chipH / 2 + 2.0, -chipD / 2 + 5);

    // Channel: inlet → lysis
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(10, 0.08, 0.4), channelMat), -5, chipH / 2 + 0.5, -chipD / 2 + 3);

    // Purification zone (2×3 mm)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(3, 0.12, 2),
      new THREE.MeshStandardMaterial({ color: 0x55AA88, transparent: true, opacity: 0.45 })), -3, chipH / 2 + 0.5, -chipD / 2 + 5);
    addAt(fluidicsGroup, makeSprite("Purify", "#2d7a55", 8), -3, chipH / 2 + 2.0, -chipD / 2 + 5);

    // Channel: lysis → purification
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4, 0.08, 0.4), channelMat), -6.5, chipH / 2 + 0.5, -chipD / 2 + 5);

    // RPA amplification chambers (4 chambers, 4×3 mm each)
    const rpaStartX = 4;
    const rpaLabels = [
      "RPA-A: IS6110,\nIS1081, rpoB",
      "RPA-B: katG,\ninhA, embB",
      "RPA-C: pncA,\ngyrA, gyrB",
      "RPA-D: rrs,\neis, RNaseP"
    ];
    const rpaShortLabels = ["RPA-A", "RPA-B", "RPA-C", "RPA-D"];
    for (let i = 0; i < 4; i++) {
      const rx = rpaStartX + i * 6;
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(4, 0.15, 3),
        new THREE.MeshStandardMaterial({ color: 0xCC8844, transparent: true, opacity: 0.45 })), rx, chipH / 2 + 0.5, -chipD / 2 + 5);
      addAt(fluidicsGroup, makeSprite(rpaShortLabels[i], "#996633", 8), rx, chipH / 2 + 2.0, -chipD / 2 + 5);
    }

    // Channel: purification → RPA manifold
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(rpaStartX + 18 + 3, 0.08, 0.4), channelMat),
      ((-3) + rpaStartX + 18) / 2, chipH / 2 + 0.5, -chipD / 2 + 3.5);

    // Distribution manifold: RPA → detection grid (tree/fishbone)
    // Main trunk running horizontally across the grid
    const trunkZ = gridOriginZ - 2;
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(6 * colSpacing + 4, 0.06, 0.35), channelMat),
      gridOriginX + 3 * colSpacing, chipH / 2 + 0.5, trunkZ);
    // Vertical channels from trunk down to each column
    for (let c = 0; c < 7; c++) {
      const cx = gridOriginX + c * colSpacing;
      // Branch to row 0
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, Math.abs(gridOriginZ - trunkZ)), channelMat),
        cx, chipH / 2 + 0.5, (gridOriginZ + trunkZ) / 2);
      // Branch to row 1
      addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.06, rowSpacing), channelMat),
        cx, chipH / 2 + 0.5, gridOriginZ + rowSpacing / 2);
    }
    // Vertical from RPA area down to trunk
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.06, Math.abs(trunkZ - (-chipD / 2 + 5))), channelMat),
      gridOriginX + 3 * colSpacing, chipH / 2 + 0.5, (trunkZ + (-chipD / 2 + 5)) / 2);

    // 14 detection chambers (3.5×3.5 mm wells around each WE)
    ELECTRODE_TARGETS.forEach(e => {
      const px = gridOriginX + e.col * colSpacing;
      const pz = gridOriginZ + e.row * rowSpacing;
      // Chamber wall (open cylinder)
      const cw = new THREE.Mesh(new THREE.CylinderGeometry(2.0, 2.0, 0.4, 24, 1, true),
        new THREE.MeshStandardMaterial({ color: e.color, transparent: true, opacity: 0.2, side: THREE.DoubleSide }));
      addAt(fluidicsGroup, cw, px, chipH / 2 + 0.4, pz);
    });

    // Waste reservoir (5×8 mm)
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(5, 0.12, 8),
      new THREE.MeshStandardMaterial({ color: 0x996666, transparent: true, opacity: 0.35 })),
      chipW / 2 - 5, chipH / 2 + 0.5, 0);
    addAt(fluidicsGroup, makeSprite("Waste", "#884444", 10), chipW / 2 - 5, chipH / 2 + 2.0, 0);
    // Channels from grid to waste
    addAt(fluidicsGroup, new THREE.Mesh(new THREE.BoxGeometry(5, 0.06, 0.35), channelMat),
      gridOriginX + 6 * colSpacing + 3, chipH / 2 + 0.5, gridOriginZ + rowSpacing / 2);

    // ══════════════════════════════════════════════════════════
    // MODE 2: CROSS-SECTION — Accurate single electrode
    // ══════════════════════════════════════════════════════════
    const crossGroup = new THREE.Group();
    crossGroup.visible = false;
    scene.add(crossGroup);

    const secR = 3.5;

    // Kapton substrate — 125 µm → 4.0 units (thickest)
    addAt(crossGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 4.0, 32),
      new THREE.MeshStandardMaterial({ color: 0xD4A843, roughness: 0.35 })), 0, 2.0, 0).castShadow = true;

    // LIG layer — 20-50 µm → 0.8 units with dramatic porosity
    addAt(crossGroup, new THREE.Mesh(new THREE.CylinderGeometry(secR, secR, 0.8, 32),
      new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.9 })), 0, 4.4, 0).castShadow = true;

    // ── Heavy pore texture (foam/sponge-like LIG porosity) ──
    const poreMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 1 });
    // Top surface pores — many sizes, irregular
    for (let i = 0; i < 100; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.15);
      const pSize = 0.02 + Math.random() * 0.1;
      const pDepth = 0.04 + Math.random() * 0.35;
      const p = new THREE.Mesh(new THREE.CylinderGeometry(pSize, pSize * 0.6, pDepth, 5), poreMat);
      p.position.set(Math.cos(a) * rr, 4.8 + Math.random() * 0.05, Math.sin(a) * rr);
      p.rotation.x = (Math.random() - 0.5) * 0.5;
      p.rotation.z = (Math.random() - 0.5) * 0.5;
      crossGroup.add(p);
    }
    // Side pores (visible on cylinder wall — coral-like)
    for (let i = 0; i < 40; i++) {
      const a = Math.random() * Math.PI * 2;
      const py = 4.05 + Math.random() * 0.7;
      const p = new THREE.Mesh(new THREE.SphereGeometry(0.03 + Math.random() * 0.07, 4, 4), poreMat);
      p.position.set(Math.cos(a) * (secR - 0.01), py, Math.sin(a) * (secR - 0.01));
      crossGroup.add(p);
    }
    // Interconnected pore ridges (foam struts)
    for (let i = 0; i < 25; i++) {
      const a1 = Math.random() * Math.PI * 2, r1 = Math.random() * (secR - 0.5);
      const a2 = a1 + (Math.random() - 0.5) * 0.8, r2 = r1 + (Math.random() - 0.5) * 1.5;
      const x1 = Math.cos(a1) * r1, z1 = Math.sin(a1) * r1;
      const x2 = Math.cos(a2) * Math.min(r2, secR - 0.2), z2 = Math.sin(a2) * Math.min(r2, secR - 0.2);
      const len = Math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2);
      if (len < 0.1) continue;
      const ridge = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, len, 3), new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.95 }));
      ridge.position.set((x1 + x2) / 2, 4.82, (z1 + z2) / 2);
      ridge.rotation.z = Math.PI / 2;
      ridge.rotation.y = Math.atan2(z2 - z1, x2 - x1);
      crossGroup.add(ridge);
    }

    // ── AuNPs — discrete hemispheres nestled in pores ──
    const auSurf = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9, roughness: 0.1 });
    const auPositions = [];
    for (let i = 0; i < 35; i++) {
      const a = Math.random() * Math.PI * 2, rr = Math.random() * (secR - 0.3);
      const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
      const size = 0.03 + Math.random() * 0.04;
      addAt(crossGroup, new THREE.Mesh(new THREE.SphereGeometry(size, 8, 8, 0, Math.PI * 2, 0, Math.PI / 2), auSurf), x, 4.84, z);
      auPositions.push({ x, z, size });
    }

    // ── MCH backfill — dense carpet of SHORT stubs ──
    const mchMat = new THREE.MeshStandardMaterial({ color: 0x9ca3af, roughness: 0.6 });
    for (let i = 0; i < 120; i++) {
      const mx = (Math.random() - 0.5) * (secR * 2 - 0.4);
      const mz = (Math.random() - 0.5) * (secR * 2 - 0.4);
      if (mx * mx + mz * mz > (secR - 0.2) ** 2) continue;
      const mh = 0.12 + Math.random() * 0.08; // ~0.8 nm → clearly shorter than reporters
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, mh, 3), mchMat);
      m.position.set(mx, 4.86 + mh / 2, mz);
      m.rotation.x = (Math.random() - 0.5) * 0.2;
      m.rotation.z = (Math.random() - 0.5) * 0.2;
      crossGroup.add(m);
      // OH terminus (tiny sphere at top)
      if (Math.random() < 0.3) {
        addAt(crossGroup, new THREE.Mesh(new THREE.SphereGeometry(0.015, 4, 4),
          new THREE.MeshStandardMaterial({ color: 0xd1d5db })), mx, 4.86 + mh, mz);
      }
    }

    const baseY = 4.88;

    // ── ssDNA reporters: FLOPPY, wavy, variable conformations ──
    const strandMat = new THREE.MeshStandardMaterial({ color: 0x67e8f9 });
    const mbMat = new THREE.MeshStandardMaterial({ color: 0x2563eb, emissive: 0x1a3a6b, emissiveIntensity: 0.3 });
    const thiolMat = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.8, roughness: 0.15 });
    const cutMat = new THREE.MeshStandardMaterial({ color: 0x3d7a63 });

    const reporters = [];
    // Place reporters ON AuNP positions (3-5 per AuNP)
    auPositions.forEach((au, ai) => {
      const nRep = 3 + Math.floor(Math.random() * 3); // 3-5 reporters per AuNP
      for (let ri = 0; ri < nRep; ri++) {
        const jitter = 0.06;
        const x = au.x + (Math.random() - 0.5) * jitter;
        const z = au.z + (Math.random() - 0.5) * jitter;
        if (x * x + z * z > (secR - 0.4) ** 2) continue;

        const h = 1.5 + Math.random() * 0.8;
        // Conformation distribution: 30% bent down, 40% intermediate, 30% extended
        const rnd = Math.random();
        const bentToward = rnd < 0.3;
        const extended = rnd > 0.7;
        const bendMag = bentToward ? 0.15 : extended ? 0.03 : 0.08;
        const bendX = (Math.random() - 0.5) * (bentToward ? 1.2 : 0.4);
        const bendZ = (Math.random() - 0.5) * (bentToward ? 1.2 : 0.4);

        // Thiol-Au bond marker (S atom — yellow)
        const th = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), thiolMat);
        addAt(crossGroup, th, x, baseY, z);

        // C6 spacer (very short gray connector)
        const spacer = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.08, 3),
          new THREE.MeshStandardMaterial({ color: 0x888888 }));
        addAt(crossGroup, spacer, x, baseY + 0.04, z);

        // Floppy ssDNA: 6 segments with random-coil trajectory
        const segs = [];
        const nSeg = 6, segH = h / nSeg;
        let cx = x, cz = z, cy = baseY + 0.08;
        for (let s = 0; s < nSeg; s++) {
          const t_param = s / nSeg;
          const waveMag = bendMag + Math.random() * 0.06;
          cx += Math.sin(ai * 2.3 + ri * 1.7 + s * 1.7) * waveMag + bendX * (t_param * t_param);
          cz += Math.cos(ai * 1.9 + ri * 1.3 + s * 1.3) * waveMag + bendZ * (t_param * t_param);
          const segMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, segH, 4), strandMat.clone());
          segMesh.position.set(cx, cy + segH / 2, cz);
          segMesh.rotation.x = bendX * 0.3 + Math.sin(s * 1.4 + ai + ri) * 0.2;
          segMesh.rotation.z = bendZ * 0.3 + Math.cos(s * 1.1 + ai * 0.7 + ri) * 0.2;
          segMesh.userData._rx0 = segMesh.rotation.x;
          segMesh.userData._rz0 = segMesh.rotation.z;
          crossGroup.add(segMesh); segs.push(segMesh);
          cy += segH;
        }

        // MB sphere at tip — position depends on conformation
        const mbY = bentToward ? baseY + h * 0.35 : baseY + h + 0.08;
        const mb = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 10), mbMat.clone());
        mb.position.set(cx, mbY, cz);
        crossGroup.add(mb);

        // Cut stub (for Cas12a cleavage animation)
        const cleavageTime = 2 + Math.random() * 26;
        const cutH = h * (0.15 + Math.random() * 0.2);
        const st = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, cutH, 4), cutMat);
        st.position.set(x, baseY + cutH / 2, z);
        st.visible = false; crossGroup.add(st);

        // Detached MB fragment
        const detachedMB = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 8),
          new THREE.MeshStandardMaterial({ color: 0x2563eb, transparent: true, opacity: 0.65 }));
        detachedMB.position.set(cx + (Math.random() - 0.5) * 0.6, baseY + h + 1.5 + Math.random() * 3, cz + (Math.random() - 0.5) * 0.6);
        detachedMB.visible = false; crossGroup.add(detachedMB);

        reporters.push({ segs, mb, stub: st, thiol: th, detachedMB, cleavageTime, bentToward, baseX: x, baseZ: z });
      }
    });

    // ── Solution-phase elements ──
    // Cas12a:crRNA RNP complexes (green bilobed, ~7 nm)
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

    // RPA amplicon dsDNA (double helix)
    const ampliconGroup = new THREE.Group();
    ampliconGroup.visible = false;
    const helixMat1 = new THREE.MeshStandardMaterial({ color: 0x5577CC, transparent: true, opacity: 0.6 });
    const helixMat2 = new THREE.MeshStandardMaterial({ color: 0xCC5577, transparent: true, opacity: 0.6 });
    for (let t = 0; t < 30; t++) {
      const angle = t * 0.4;
      const y = baseY + 5.5 + t * 0.08;
      addAt(ampliconGroup, new THREE.Mesh(new THREE.SphereGeometry(0.05, 4, 4), helixMat1),
        Math.cos(angle) * 0.15, y, Math.sin(angle) * 0.15);
      addAt(ampliconGroup, new THREE.Mesh(new THREE.SphereGeometry(0.05, 4, 4), helixMat2),
        Math.cos(angle + Math.PI) * 0.15, y, Math.sin(angle + Math.PI) * 0.15);
    }
    crossGroup.add(ampliconGroup);

    // ── Scale bars (left side) ──
    const sbMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    // Kapton: 0 → 4.0
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 4.0, 0.03), sbMat), -secR - 0.8, 2.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), sbMat), -secR - 0.8, 0.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), sbMat), -secR - 0.8, 4.0, 0);
    addAt(crossGroup, makeSprite("125 µm", "#777", 9), -secR - 1.7, 2.0, 0);
    // LIG: 4.0 → 4.8
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.8, 0.03), sbMat), -secR - 0.5, 4.4, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), sbMat), -secR - 0.5, 4.0, 0);
    addAt(crossGroup, new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.03, 0.03), sbMat), -secR - 0.5, 4.8, 0);
    addAt(crossGroup, makeSprite("20–50 µm", "#777", 8), -secR - 1.6, 4.4, 0);

    // ══════════════════════════════════════════════════════════
    // MODE 3: SIDE PROFILE VIEW
    // ══════════════════════════════════════════════════════════
    const sideGroup = new THREE.Group();
    sideGroup.visible = false;
    scene.add(sideGroup);

    // Side profile: show layered cross-section of full chip assembly
    const sW = 60, sKapH = 2.5, sLigH = 0.4, sPassH = 0.15, sCocH = 1.2;
    // Kapton base
    addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(sW, sKapH, 8),
      new THREE.MeshStandardMaterial({ color: 0xD4A843, roughness: 0.3 })), 0, sKapH / 2, 0);
    // LIG patterns on top (discontinuous — only where electrodes are)
    for (let i = 0; i < 7; i++) {
      const lx = -18 + i * 5;
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(3, sLigH, 8),
        new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.85 })), lx, sKapH + sLigH / 2, 0);
      // Pore dots on LIG
      for (let p = 0; p < 5; p++) {
        addAt(sideGroup, new THREE.Mesh(new THREE.SphereGeometry(0.04, 4, 4), poreMat),
          lx + (Math.random() - 0.5) * 2.5, sKapH + sLigH + 0.02, (Math.random() - 0.5) * 6);
      }
    }
    // Passivation layer
    addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(sW, sPassH, 8),
      new THREE.MeshStandardMaterial({ color: 0x1a2744, transparent: true, opacity: 0.35 })), 0, sKapH + sLigH + sPassH / 2, 0);
    // Openings in passivation (show as gaps)
    for (let i = 0; i < 3; i++) {
      const ox = -8 + i * 5;
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(3.2, sPassH + 0.01, 8.1),
        new THREE.MeshStandardMaterial({ color: 0xD4A843, transparent: true, opacity: 0 })), ox, sKapH + sLigH + sPassH / 2, 0);
    }
    // COC microfluidic lid with chambers
    const cocBaseY = sKapH + sLigH + sPassH;
    addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(sW, sCocH, 8),
      new THREE.MeshPhysicalMaterial({ color: 0xe8e8e8, transparent: true, opacity: 0.3, roughness: 0.1, clearcoat: 0.5 })), 0, cocBaseY + sCocH / 2, 0);
    // Chamber cavities (visible inside COC)
    for (let i = 0; i < 3; i++) {
      const cx = -8 + i * 5;
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.6, 6),
        new THREE.MeshStandardMaterial({ color: 0x4488AA, transparent: true, opacity: 0.3 })), cx, cocBaseY + 0.3, 0);
      // Solution inside chamber
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(3, 0.4, 5.5),
        new THREE.MeshStandardMaterial({ color: 0x93C5FD, transparent: true, opacity: 0.2 })), cx, cocBaseY + 0.2, 0);
    }
    // Contact pads at edge
    for (let i = 0; i < 6; i++) {
      addAt(sideGroup, new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.15, 1.5), goldMat), -15 + i * 6, sKapH + 0.08, 5);
    }
    // Scale bars for side view
    addAt(sideGroup, makeSprite("Kapton 125 µm", "#B8860B", 9), -sW / 2 - 4, sKapH / 2, 0);
    addAt(sideGroup, makeSprite("LIG 20–50 µm", "#555", 8), -sW / 2 - 4, sKapH + sLigH / 2, 0);
    addAt(sideGroup, makeSprite("COC lid ~1 mm", "#888", 8), -sW / 2 - 4, cocBaseY + sCocH / 2, 0);
    addAt(sideGroup, makeSprite("Contact pads →", "#B8860B", 8), 18, sKapH + 0.5, 5);

    // ══════════════════════════════════════════════════════════
    // CAMERA ORBIT
    // ══════════════════════════════════════════════════════════
    let orbit = { theta: 0.3, phi: -0.45, dist: 80, target: new THREE.Vector3(0, 0, 0) };
    let tgtOrbit = { theta: 0.3, phi: -0.45, dist: 80, target: new THREE.Vector3(0, 0, 0) };
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
    const onWheel = (e) => { e.preventDefault(); tgtOrbit.dist = Math.max(12, Math.min(150, tgtOrbit.dist + e.deltaY * 0.08)); };
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
        if (r.detachedMB.visible) {
          r.detachedMB.position.y += Math.sin(time * 0.5 + i) * 0.002;
          r.detachedMB.position.x += Math.sin(time * 0.3 + i * 1.3) * 0.001;
        }
      });

      // Animate RNP complexes
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
      chipGroup, crossGroup, sideGroup, fluidicsGroup, passivMesh, labelSprites,
      reporters, padMeshes, rnps, ampliconGroup, orbit, tgtOrbit,
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

  // ── Mode transitions ──
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.chipGroup.visible = mode === 1;
    s.crossGroup.visible = mode === 2;
    s.sideGroup.visible = mode === 3;
    if (mode === 1) {
      s.tgtOrbit.dist = 80; s.tgtOrbit.theta = 0.3; s.tgtOrbit.phi = -0.45;
      s.tgtOrbit.target = new THREE.Vector3(0, 0, 0);
    } else if (mode === 2) {
      s.tgtOrbit.dist = 18; s.tgtOrbit.theta = 0.4; s.tgtOrbit.phi = -0.3;
      s.tgtOrbit.target = new THREE.Vector3(0, 4.5, 0);
    } else if (mode === 3) {
      s.tgtOrbit.dist = 30; s.tgtOrbit.theta = 0.0; s.tgtOrbit.phi = -0.1;
      s.tgtOrbit.target = new THREE.Vector3(0, 2, 0);
    }
  }, [mode, selectedPad]);

  // ── Toggle visibility ──
  useEffect(() => {
    const s = stateRef.current;
    if (s?.fluidicsGroup) s.fluidicsGroup.visible = showMicrofluidics;
  }, [showMicrofluidics]);

  useEffect(() => {
    const s = stateRef.current;
    if (s?.passivMesh) s.passivMesh.visible = showPassivation;
  }, [showPassivation]);

  useEffect(() => {
    const s = stateRef.current;
    if (s?.labelSprites) s.labelSprites.forEach(l => { l.visible = showLabels; });
  }, [showLabels]);

  // ── Progressive Cas12a cleavage ──
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
    s.rnps.forEach(rnp => { rnp.visible = cas12aActive; });
    s.ampliconGroup.visible = cas12aActive;
  }, [cas12aActive, incubationMin]);

  // ── Computed data for selected pad ──
  const selTarget = selectedPad ? ELECTRODE_TARGETS[selectedPad.idx] : null;
  const selDrug = selTarget?.drug || null;
  const selR = selectedPad ? results.find(x => x.label === selectedPad.target) : null;
  const selEff = selectedPad ? getEfficiency(selectedPad.target) : null;
  const selStrat = selectedPad ? targetStrategy(selectedPad.target) : null;
  const selDisc = selR?.disc && selR.disc < 900 ? selR.disc : null;
  const selScore = selR?.ensembleScore || selR?.score || null;

  const deltaI = selectedPad && computeGamma && echemGamma0_mol ? (() => {
    const G = computeGamma(incubationMin * 60, getEfficiency(selectedPad.target), echemKtrans);
    return ((1 - G / echemGamma0_mol) * 100).toFixed(1);
  })() : null;

  // ── Curve data ──
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
      const y = h - (Math.abs(p.I) / maxI) * h * 0.82 - h * 0.06;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  const svgPathEIS = (pts, w, h) => {
    if (!pts?.length) return "";
    const maxZr = Math.max(...pts.map(p => p.Zr), 1);
    const maxZi = Math.max(...pts.map(p => p.Zi), 1);
    return pts.map((p, i) => {
      const x = ((p.Zr - 40) / (maxZr - 40 + 1)) * w * 0.88 + w * 0.06;
      const y = h - (p.Zi / (maxZi + 1)) * h * 0.82 - h * 0.06;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  };

  const cleavedCount = cas12aActive && reporters.length > 0 ? Math.round(reporters.length * Math.min(1, incubationMin / 28)) : 0;

  // ══════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════
  return (
    <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#F8F9FA" }}>
      <div ref={mountRef} style={{ width: "100%", minHeight: 280 }} />

      {/* ═══ MODE 1: CHIP OVERVIEW ═══ */}
      {mode === 1 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            GUARD 14-Plex MDR-TB Chip
          </div>
          <div style={{ position: "absolute", top: 12, right: 16, display: "flex", gap: 6, alignItems: "center" }}>
            {[
              { label: "Fluidics", active: showMicrofluidics, toggle: () => setShowMicrofluidics(!showMicrofluidics) },
              { label: "Passivation", active: showPassivation, toggle: () => setShowPassivation(!showPassivation) },
              { label: "Labels", active: showLabels, toggle: () => setShowLabels(!showLabels) },
            ].map(b => (
              <button key={b.label} onClick={b.toggle} style={{
                fontSize: 9, fontWeight: 600, padding: "3px 10px", borderRadius: 4,
                border: "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
                background: b.active ? "rgba(51,136,170,0.15)" : "rgba(255,255,255,0.88)",
                color: b.active ? "#1a6680" : "#6B7280",
              }}>
                {b.active ? "Hide" : "Show"} {b.label}
              </button>
            ))}
            <button onClick={() => setMode(3)} style={{
              fontSize: 9, fontWeight: 600, padding: "3px 10px", borderRadius: 4,
              border: "1px solid #D1D5DB", cursor: "pointer", fontFamily: MONO,
              background: "rgba(255,255,255,0.88)", color: "#6B7280",
            }}>
              Side Profile
            </button>
          </div>

          {/* Drug class legend */}
          <div style={{ position: "absolute", bottom: 12, left: 16, display: "flex", gap: 5, flexWrap: "wrap" }}>
            {[
              { d: "TB ID", c: "#22d3ee" }, { d: "RIF", c: "#ef4444" }, { d: "INH", c: "#f97316" },
              { d: "EMB", c: "#eab308" }, { d: "PZA", c: "#eab308" }, { d: "FQ", c: "#a855f7" },
              { d: "KAN/AMK", c: "#ec4899" }, { d: "Human Ctrl", c: "#22d3ee" },
            ].map(l => (
              <span key={l.d} style={{ fontSize: 8, fontWeight: 700, fontFamily: MONO, display: "flex", alignItems: "center", gap: 3, background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: l.c, display: "inline-block" }} />{l.d}
              </span>
            ))}
          </div>

          {/* Chip specs */}
          <div style={{ position: "absolute", bottom: 12, right: 16, display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ fontSize: 8, color: "#6B7280", background: "rgba(255,255,255,0.9)", padding: "2px 7px", borderRadius: 3, fontFamily: MONO }}>
              60 × 30 mm · Kapton HN + LIG · 14 WE + CE + RE
            </span>
          </div>
        </>
      )}

      {/* ═══ MODE 2: CROSS-SECTION ═══ */}
      {mode === 2 && selectedPad && (
        <>
          {/* Info panel */}
          <div style={{ position: "absolute", top: 12, left: 16, background: "rgba(255,255,255,0.95)", padding: "12px 16px", borderRadius: 8, border: "1px solid #E3E8EF", maxWidth: 300, backdropFilter: "blur(8px)" }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING }}>
              {selectedPad.target} · <span style={{ color: DRUG_CSS[selDrug] || "#888" }}>{selDrug}</span>
            </div>
            <div style={{ fontSize: 10, color: "#6B7280", fontFamily: MONO, marginTop: 4, lineHeight: 1.7 }}>
              S_eff = {selEff?.toFixed(3)}{selScore != null && ` · Score = ${selScore.toFixed(2)}`}{selDisc ? ` · D = ${selDisc.toFixed(1)}×` : ""}<br />
              {selStrat} detection
              {selR?.hasPrimers && <span style={{ color: "#16A34A" }}> · RPA primers</span>}
            </div>
            {selR && (
              <div style={{ fontSize: 9, color: "#9CA3AF", fontFamily: MONO, marginTop: 4, lineHeight: 1.6, borderTop: "1px solid #E3E8EF", paddingTop: 4 }}>
                {selR.spacer && <div>crRNA: <span style={{ color: "#374151", letterSpacing: "0.5px" }}>{selR.spacer.slice(0, 20)}{selR.spacer.length > 20 ? "…" : ""}</span></div>}
                {selR.pam && <div>PAM: <span style={{ color: "#374151" }}>{selR.pam}</span>{selR.pamVariant && <span> ({selR.pamVariant})</span>}</div>}
                {deltaI != null && <div>Expected ΔI%: <span style={{ color: "#16A34A", fontWeight: 700 }}>{deltaI}%</span> @ {incubationMin} min</div>}
              </div>
            )}
          </div>

          <button onClick={() => stateRef.current?._toMode1()} style={{ position: "absolute", top: 12, right: 16, fontSize: 11, fontWeight: 600, padding: "7px 16px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}>
            ← Back to chip
          </button>

          {/* Layer labels */}
          <div style={{ position: "absolute", left: 16, top: "36%", transform: "translateY(-50%)", display: "flex", flexDirection: "column", gap: 3 }}>
            {[
              { label: "MB (E° = −0.22 V, n = 2e⁻)", color: "#2563eb", dim: "~1 nm" },
              { label: "ssDNA reporter (12–20 nt, poly-T)", color: "#67e8f9", dim: "4–7 nm" },
              { label: "S–Au thiol bond (~170 kJ/mol)", color: "#FFD700", dim: "" },
              { label: "C6 spacer (~0.7 nm)", color: "#888", dim: "" },
              { label: "MCH backfill (C6-OH, hydrophilic)", color: "#9ca3af", dim: "~0.8 nm" },
              { label: "AuNPs (electrodeposited)", color: "#FFD700", dim: "10–50 nm" },
              { label: "LIG (~30 Ω/□, ~340 m²/g)", color: "#2D2D2D", dim: "20–50 µm" },
              { label: "Kapton HN polyimide", color: "#D4A843", dim: "125 µm" },
            ].map(l => (
              <div key={l.label} style={{ fontSize: 8, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.92)", padding: "2px 7px", borderRadius: 3, borderLeft: `3px solid ${l.color}`, lineHeight: 1.3 }}>
                ← {l.label}{l.dim && <span style={{ color: "#9CA3AF" }}> ({l.dim})</span>}
              </div>
            ))}
            <div style={{ fontSize: 7, color: "#B0B0B0", fontFamily: MONO, fontStyle: "italic", marginTop: 2, paddingLeft: 4 }}>
              ⚠ vertical scale exaggerated
            </div>
          </div>

          {/* ── Electrochemistry curves panel ── */}
          {curveData && (
            <div style={{ position: "absolute", bottom: 60, right: 16, background: "rgba(255,255,255,0.96)", padding: "10px 14px", borderRadius: 8, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)", minWidth: 200 }}>
              {/* Curve mode tabs */}
              <div style={{ display: "flex", gap: 2, marginBottom: 6 }}>
                {["DPV", "SWV", "EIS"].map(m => (
                  <button key={m} onClick={() => setCurveMode(m)} style={{
                    fontSize: 8, fontWeight: curveMode === m ? 700 : 500, fontFamily: MONO,
                    padding: "2px 10px", borderRadius: 3, border: "none", cursor: "pointer",
                    background: curveMode === m ? "#374151" : "#F3F4F6",
                    color: curveMode === m ? "#fff" : "#6B7280",
                  }}>{m}</button>
                ))}
              </div>

              <svg width={200} height={100} viewBox="0 0 200 100">
                {curveMode !== "EIS" ? (
                  <>
                    <path d={svgPathVolt(curveData.before, 200, 100)} fill="none" stroke="#93C5FD" strokeWidth="1.5" strokeDasharray="3,2" />
                    <path d={svgPathVolt(curveData.after, 200, 100)} fill="none" stroke="#ef4444" strokeWidth="1.5" />
                    <line x1="0" y1="96" x2="200" y2="96" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="95" fontSize="6" fill="#9CA3AF">−0.05</text>
                    <text x="160" y="95" fontSize="6" fill="#9CA3AF">−0.40 V</text>
                    {/* MB peak annotation at -0.22 V */}
                    <line x1="100" y1="0" x2="100" y2="100" stroke="#E5E7EB" strokeWidth="0.5" strokeDasharray="2,2" />
                    <text x="102" y="8" fontSize="5.5" fill="#9CA3AF">−0.22 V (MB)</text>
                    {/* ΔI% annotation */}
                    {deltaI && <text x="140" y="20" fontSize="7" fill="#ef4444" fontWeight="bold">ΔI = {deltaI}%</text>}
                    {/* Threshold lines */}
                    <line x1="0" y1={100 - 5} x2="200" y2={100 - 5} stroke="#d4d4d4" strokeWidth="0.3" strokeDasharray="1,2" />
                    <text x="170" y={100 - 6} fontSize="4" fill="#d4d4d4">5%</text>
                    <line x1="0" y1={100 - 30} x2="200" y2={100 - 30} stroke="#d4d4d4" strokeWidth="0.3" strokeDasharray="1,2" />
                    <text x="170" y={100 - 31} fontSize="4" fill="#d4d4d4">30%</text>
                  </>
                ) : (
                  <>
                    <path d={svgPathEIS(curveData.before, 200, 100)} fill="none" stroke="#93C5FD" strokeWidth="1.5" strokeDasharray="3,2" />
                    <path d={svgPathEIS(curveData.after, 200, 100)} fill="none" stroke="#ef4444" strokeWidth="1.5" />
                    <line x1="0" y1="96" x2="200" y2="96" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="2" y="95" fontSize="6" fill="#9CA3AF">Z' (Ω)</text>
                    <line x1="2" y1="0" x2="2" y2="96" stroke="#D1D5DB" strokeWidth="0.5" />
                    <text x="4" y="8" fontSize="6" fill="#9CA3AF">−Z'' (Ω)</text>
                    {/* Randles circuit label */}
                    <text x="100" y="94" fontSize="5" fill="#B0B0B0" textAnchor="middle">Rs — [Cdl ‖ (Rct — Zw)]</text>
                  </>
                )}
              </svg>

              {/* Legend */}
              <div style={{ fontSize: 7, color: "#9CA3AF", fontFamily: MONO, display: "flex", gap: 10, marginTop: 3 }}>
                <span><span style={{ color: "#93C5FD" }}>---</span> baseline</span>
                <span><span style={{ color: "#ef4444" }}>—</span> t={incubationMin}m</span>
              </div>
              {/* Parameters */}
              <div style={{ fontSize: 6.5, color: "#C0C0C0", fontFamily: MONO, marginTop: 2, lineHeight: 1.4 }}>
                {curveMode === "DPV" && "Pulse: 50 mV | Width: 50 ms | Step: 4 mV | Window: −0.5 to 0 V"}
                {curveMode === "SWV" && "Freq: 50 Hz | Amp: 25 mV | Step: 4 mV"}
                {curveMode === "EIS" && "0.1 Hz – 100 kHz | 10 mV AC | Bias: OCP"}
              </div>
            </div>
          )}

          {/* ── Molecular inset: Thiol-Au bond ── */}
          <div style={{ position: "absolute", bottom: 60, left: 16, background: "rgba(255,255,255,0.94)", padding: "6px 10px", borderRadius: 6, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)" }}>
            <div style={{ fontSize: 7, fontWeight: 700, color: "#374151", fontFamily: MONO, marginBottom: 3 }}>Thiol–Au Anchor</div>
            <svg width={90} height={50} viewBox="0 0 90 50">
              {/* Au lattice (FCC) */}
              {[0, 12, 24, 36].map((x, i) => [0, 12].map((y, j) => (
                <circle key={`au-${i}-${j}`} cx={8 + x} cy={40 - y} r={5} fill="#FFD700" opacity={0.7} />
              ))).flat()}
              {/* S atom */}
              <circle cx={20} cy={22} r={4} fill="#CCAA00" />
              <text x={20} y={24} fontSize="5" fill="#fff" textAnchor="middle" fontWeight="bold">S</text>
              {/* C6 spacer */}
              <line x1={20} y1={18} x2={40} y2={10} stroke="#888" strokeWidth="1.5" />
              <text x={30} y={8} fontSize="4" fill="#9CA3AF">C6</text>
              {/* First nucleotide */}
              <circle cx={45} cy={8} r={3} fill="#67e8f9" />
              <text x={45} y={9.5} fontSize="3.5" fill="#fff" textAnchor="middle">T</text>
              <line x1={48} y1={8} x2={60} y2={6} stroke="#67e8f9" strokeWidth="1" />
              <text x={65} y={7} fontSize="4" fill="#67e8f9">ssDNA →</text>
              {/* Dimensions */}
              <text x={5} y={48} fontSize="3.5" fill="#B0B0B0">Au(111)</text>
              <text x={15} y={16} fontSize="3.5" fill="#CCAA00">170 kJ/mol</text>
              <text x={28} y={15} fontSize="3.5" fill="#888">0.7 nm</text>
            </svg>
          </div>

          {/* ── Electron transfer inset ── */}
          <div style={{ position: "absolute", bottom: 115, left: 16, background: "rgba(255,255,255,0.94)", padding: "6px 10px", borderRadius: 6, border: "1px solid #E3E8EF", backdropFilter: "blur(8px)" }}>
            <div style={{ fontSize: 7, fontWeight: 700, color: "#374151", fontFamily: MONO, marginBottom: 3 }}>e⁻ Transfer Mechanism</div>
            <svg width={90} height={35} viewBox="0 0 90 35">
              {/* Gold surface */}
              <rect x={0} y={28} width={90} height={7} fill="#FFD700" rx={1} opacity={0.5} />
              {/* Bent reporter (close — transfers) */}
              <path d="M20,28 Q18,18 22,12" fill="none" stroke="#67e8f9" strokeWidth="1" />
              <circle cx={22} cy={12} r={3} fill="#2563eb" />
              <text x={22} y={13.5} fontSize="3" fill="#fff" textAnchor="middle">MB</text>
              <text x={10} y={22} fontSize="3.5" fill="#16A34A">⚡ e⁻</text>
              <text x={5} y={8} fontSize="3.5" fill="#16A34A">&lt;2 nm</text>
              {/* Extended reporter (far — no transfer) */}
              <path d="M65,28 Q63,20 68,4" fill="none" stroke="#67e8f9" strokeWidth="1" />
              <circle cx={68} cy={4} r={3} fill="#2563eb" />
              <text x={68} y={5.5} fontSize="3" fill="#fff" textAnchor="middle">MB</text>
              <text x={74} y={10} fontSize="3.5" fill="#ef4444">✗ too far</text>
              <text x={74} y={15} fontSize="3.5" fill="#ef4444">&gt;3 nm</text>
              {/* Label */}
              <text x={45} y={33} fontSize="3.5" fill="#B0B0B0" textAnchor="middle">Au surface</text>
            </svg>
          </div>

          {/* ── Cas12a controls + slider ── */}
          <div style={{ position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 12, alignItems: "center", background: "rgba(255,255,255,0.95)", padding: "8px 18px", borderRadius: 8, border: "1px solid #E3E8EF", flexWrap: "wrap", justifyContent: "center", backdropFilter: "blur(8px)" }}>
            <button onClick={() => setCas12aActive(!cas12aActive)} style={{
              fontSize: 11, fontWeight: 700, padding: "7px 16px", borderRadius: 6, cursor: "pointer", fontFamily: MONO,
              background: cas12aActive ? "#DC2626" : "#16A34A", color: "#fff", border: "none", transition: "background 0.2s",
            }}>
              {cas12aActive ? "Reset reporters" : "Activate Cas12a"}
            </button>
            {cas12aActive && (
              <div style={{ fontSize: 10, fontFamily: MONO, color: "#6B7280", display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontWeight: 700, color: "#16A34A" }}>ΔI% = {deltaI}%</span>
                <span style={{ color: "#9CA3AF" }}>·</span>
                <span style={{ color: "#33AA55" }}>{cleavedCount}/{reporters.length} cleaved</span>
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 9, color: "#6B7280", fontFamily: MONO }}>t =</span>
              <input type="range" min="0" max="30" step="1" value={incubationMin} onChange={e => setIncubationMin(+e.target.value)} style={{ width: 90, accentColor: "#374151" }} />
              <span style={{ fontSize: 10, fontWeight: 700, fontFamily: MONO, color: "#374151", minWidth: 36 }}>{incubationMin} min</span>
            </div>
          </div>

          {/* Solution-phase legend */}
          {cas12aActive && (
            <div style={{ position: "absolute", right: 16, top: "38%", display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                { label: "Cas12a:crRNA RNP (~1250 turnovers/hr)", color: "#33AA55" },
                { label: "Cleaved MB (diffusing away)", color: "#2563eb" },
                { label: "RPA amplicon dsDNA (~150 bp)", color: "#5577CC" },
              ].map(l => (
                <div key={l.label} style={{ fontSize: 7, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.9)", padding: "2px 6px", borderRadius: 3, borderLeft: `3px solid ${l.color}` }}>
                  {l.label}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ═══ MODE 3: SIDE PROFILE ═══ */}
      {mode === 3 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            Side Profile — Layer Assembly
          </div>
          <button onClick={() => { setMode(1); }} style={{ position: "absolute", top: 12, right: 16, fontSize: 11, fontWeight: 600, padding: "7px 16px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}>
            ← Back to chip
          </button>
          <div style={{ position: "absolute", bottom: 12, left: 16, background: "rgba(255,255,255,0.94)", padding: "8px 14px", borderRadius: 8, border: "1px solid #E3E8EF" }}>
            <div style={{ fontSize: 8, fontFamily: MONO, color: "#374151", lineHeight: 1.8 }}>
              <div style={{ borderLeft: "3px solid #D4A843", paddingLeft: 6, marginBottom: 3 }}>Kapton HN polyimide — 125 µm (DuPont commercial)</div>
              <div style={{ borderLeft: "3px solid #2D2D2D", paddingLeft: 6, marginBottom: 3 }}>LIG — CO₂ laser 10.6 µm, 20–50 µm, sp³→sp²</div>
              <div style={{ borderLeft: "3px solid #1a2744", paddingLeft: 6, marginBottom: 3 }}>SU-8 passivation — 10–50 µm, circular openings at WE</div>
              <div style={{ borderLeft: "3px solid #e8e8e8", paddingLeft: 6, marginBottom: 3 }}>COC microfluidic lid — injection-molded, ~1 mm</div>
              <div style={{ borderLeft: "3px solid #c5a03f", paddingLeft: 6 }}>Contact pads — reader pogo pins connect here</div>
            </div>
          </div>
        </>
      )}

      {/* ═══ TOOLTIP ═══ */}
      {tooltipInfo && mode === 1 && (
        <div style={{ position: "absolute", left: tooltipPos.x + 14, top: tooltipPos.y - 12, pointerEvents: "none", background: "rgba(255,255,255,0.96)", border: "1px solid #E3E8EF", borderRadius: 6, padding: "8px 12px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)", zIndex: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, fontFamily: MONO, color: DRUG_CSS[tooltipInfo.drug] || "#333" }}>{tooltipInfo.target}</div>
          <div style={{ fontSize: 10, color: "#6B7280", marginTop: 2 }}>
            {tooltipInfo.drug} · S_eff = {getEfficiency(tooltipInfo.target).toFixed(3)}
            {(() => { const r = results.find(x => x.label === tooltipInfo.target); return r?.disc && r.disc < 900 ? ` · D = ${r.disc.toFixed(1)}×` : ""; })()}
          </div>
          <div style={{ fontSize: 9, color: "#9CA3AF", marginTop: 1 }}>{targetStrategy(tooltipInfo.target)} detection · Click to inspect</div>
        </div>
      )}
    </div>
  );
}
