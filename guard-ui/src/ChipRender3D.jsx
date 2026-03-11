import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

const DRUG_HEX = { RIF: 0x3288bd, INH: 0x66c2a5, EMB: 0xabdda4, PZA: 0xfee08b, FQ: 0xf46d43, AG: 0xd53e4f, CTRL: 0x888888 };
const DRUG_CSS = { RIF: "#3288bd", INH: "#66c2a5", EMB: "#abdda4", PZA: "#fee08b", FQ: "#f46d43", AG: "#d53e4f", CTRL: "#888888" };

export default function ChipRender3D({ electrodeLayout, targetDrug, targetStrategy, getEfficiency, results, computeGamma, echemTime, echemKtrans, echemGamma0_mol, T, HEADING, MONO, mobile }) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);
  const [mode, setMode] = useState(1);
  const [selectedPad, setSelectedPad] = useState(null);
  const [cas12aActive, setCas12aActive] = useState(false);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    const W = container.clientWidth;
    const H = Math.round(W * 9 / 16);
    container.style.height = H + "px";

    // ── Renderer ──
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

    // ── Lighting — 3-point + hemisphere ──
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
    const fill = new THREE.DirectionalLight(0xE6F0FF, 0.3);
    fill.position.set(40, 30, -20);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xFFFFFF, 0.2);
    rim.position.set(0, 20, -50);
    scene.add(rim);

    // Ground
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.ShadowMaterial({ opacity: 0.15 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1;
    ground.receiveShadow = true;
    scene.add(ground);

    // ══════════ MODE 1: CHIP OVERVIEW ══════════
    const chipGroup = new THREE.Group();
    scene.add(chipGroup);

    // A. Chip body — 65×35×1.5 Kapton
    const bodyMat = new THREE.MeshPhysicalMaterial({
      color: 0xC8943E, roughness: 0.25, metalness: 0.0,
      clearcoat: 0.3, clearcoatRoughness: 0.4,
      transparent: true, opacity: 0.92,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(65, 1.5, 35), bodyMat);
    body.castShadow = true;
    body.receiveShadow = true;
    chipGroup.add(body);

    // B. Sample prep well — recessed chamber
    const wellMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.95 });
    const prepWell = new THREE.Mesh(new THREE.BoxGeometry(15, 0.1, 12), wellMat);
    prepWell.position.set(-22, 0.76, 0);
    chipGroup.add(prepWell);
    // Fluid tint
    const fluidMat = new THREE.MeshStandardMaterial({ color: 0x93C5FD, transparent: true, opacity: 0.35 });
    const fluid = new THREE.Mesh(new THREE.BoxGeometry(14.5, 0.05, 11.5), fluidMat);
    fluid.position.set(-22, 0.82, 0);
    chipGroup.add(fluid);
    // Magnetic beads
    const beadMat = new THREE.MeshStandardMaterial({ color: 0x4B5563, metalness: 0.6, roughness: 0.3 });
    for (let i = 0; i < 15; i++) {
      const bead = new THREE.Mesh(new THREE.SphereGeometry(0.3, 6, 6), beadMat);
      bead.position.set(-22 + (Math.random() - 0.5) * 12, 0.95, (Math.random() - 0.5) * 9);
      chipGroup.add(bead);
    }

    // B2. Channel from prep to center pad
    const channelMat = new THREE.MeshStandardMaterial({ color: 0x2D2D2D, roughness: 0.8 });
    const prepChannel = new THREE.Mesh(new THREE.BoxGeometry(8, 0.06, 0.5), channelMat);
    prepChannel.position.set(-11, 0.78, 0);
    chipGroup.add(prepChannel);

    // C. Central application pad — recessed well with wax rim
    const centerFloor = new THREE.Mesh(new THREE.CylinderGeometry(2, 2, 0.08, 32), wellMat);
    centerFloor.position.set(-6, 0.76, 0);
    chipGroup.add(centerFloor);
    const waxRimMat = new THREE.MeshPhysicalMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.5, roughness: 0.2 });
    const centerRim = new THREE.Mesh(new THREE.TorusGeometry(2, 0.12, 8, 32), waxRimMat);
    centerRim.rotation.x = Math.PI / 2;
    centerRim.position.set(-6, 0.9, 0);
    chipGroup.add(centerRim);
    // Funnel
    const funnel = new THREE.Mesh(new THREE.CylinderGeometry(2, 1, 0.3, 24, 1, true), new THREE.MeshStandardMaterial({ color: 0xC8943E, transparent: true, opacity: 0.4, side: THREE.DoubleSide }));
    funnel.position.set(-6, 1.1, 0);
    chipGroup.add(funnel);

    // D. Detection pads — 5×3 grid
    const flatPads = electrodeLayout.flat();
    const padMeshes = [];
    const padPositions = [];
    const gridStartX = 5;
    const gridStartZ = -7;
    const spacingX = 6;
    const spacingZ = 7;
    const goldMat = new THREE.MeshStandardMaterial({ color: 0xDAA520, metalness: 0.85, roughness: 0.15 });

    // Distribution trunk channels: 3 horizontal trunks (per row)
    for (let r = 0; r < 3; r++) {
      const rz = gridStartZ + r * spacingZ;
      const trunk = new THREE.Mesh(new THREE.BoxGeometry(spacingX * 4 + 4, 0.06, 0.4), channelMat);
      trunk.position.set(gridStartX + spacingX * 2, 0.78, rz);
      chipGroup.add(trunk);
      // Trunk connection from center
      const connLen = Math.sqrt(Math.pow(gridStartX - (-6), 2) + Math.pow(rz, 2));
      const conn = new THREE.Mesh(new THREE.BoxGeometry(connLen, 0.06, 0.35), channelMat);
      conn.position.set((-6 + gridStartX) / 2, 0.78, rz / 2);
      conn.rotation.y = -Math.atan2(rz, gridStartX - (-6));
      chipGroup.add(conn);
    }

    flatPads.forEach((target, idx) => {
      const row = Math.floor(idx / 5);
      const col = idx % 5;
      const px = gridStartX + col * spacingX;
      const pz = gridStartZ + row * spacingZ;
      padPositions.push({ x: px, z: pz, target, row, col });

      const drug = targetDrug(target);
      const padColor = DRUG_HEX[drug] || 0x888888;

      // Well cavity (dark floor)
      const padFloor = new THREE.Mesh(new THREE.CylinderGeometry(1.75, 1.75, 0.08, 24), wellMat);
      padFloor.position.set(px, 0.2, pz);
      chipGroup.add(padFloor);

      // Drug-colored ring on electrode
      const drugRing = new THREE.Mesh(
        new THREE.RingGeometry(1.5, 1.75, 24),
        new THREE.MeshBasicMaterial({ color: padColor, side: THREE.DoubleSide })
      );
      drugRing.rotation.x = -Math.PI / 2;
      drugRing.position.set(px, 0.26, pz);
      chipGroup.add(drugRing);

      // AuNP speckles on pad floor
      for (let i = 0; i < 12; i++) {
        const a = Math.random() * Math.PI * 2;
        const r2 = Math.random() * 1.3;
        const au = new THREE.Mesh(new THREE.SphereGeometry(0.05, 4, 4), new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9 }));
        au.position.set(px + Math.cos(a) * r2, 0.28, pz + Math.sin(a) * r2);
        chipGroup.add(au);
      }

      // Wax barrier rim
      const padRim = new THREE.Mesh(new THREE.TorusGeometry(1.75, 0.1, 6, 24), waxRimMat);
      padRim.rotation.x = Math.PI / 2;
      padRim.position.set(px, 0.85, pz);
      chipGroup.add(padRim);

      // Lyophilized reagent pellet
      const pellet = new THREE.Mesh(
        new THREE.CylinderGeometry(0.7, 0.7, 0.1, 12),
        new THREE.MeshStandardMaterial({ color: 0xF5F0E0, roughness: 0.8, transparent: true, opacity: 0.6 })
      );
      pellet.position.set(px, 0.5, pz);
      chipGroup.add(pellet);

      // Clickable pad mesh (invisible, for raycasting)
      const padGeo = new THREE.CylinderGeometry(1.75, 1.75, 1.0, 16);
      const padMat = new THREE.MeshBasicMaterial({ visible: false });
      const padMesh = new THREE.Mesh(padGeo, padMat);
      padMesh.position.set(px, 0.5, pz);
      padMesh.userData = { target, drug, idx, padColor };
      chipGroup.add(padMesh);
      padMeshes.push(padMesh);
    });

    // Shared counter electrode (right side strip)
    const ceStrip = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.08, 20), channelMat);
    ceStrip.position.set(gridStartX + spacingX * 4 + 4, 0.78, 0);
    chipGroup.add(ceStrip);

    // E. SWV contact pads — bottom edge
    for (let i = 0; i < 16; i++) {
      const cx = -3 + i * 3.6;
      const cz = 18;
      const contact = new THREE.Mesh(new THREE.BoxGeometry(i === 15 ? 2 : 1.2, 0.15, 2.5), goldMat);
      contact.position.set(cx, 0.78, cz);
      chipGroup.add(contact);
      // Trace from contact to pad (or CE)
      if (i < 15 && padPositions[i]) {
        const tgt = padPositions[i];
        const dist = Math.sqrt(Math.pow(cx - tgt.x, 2) + Math.pow(cz - tgt.z, 2));
        const trace = new THREE.Mesh(new THREE.BoxGeometry(dist, 0.04, 0.25), channelMat);
        trace.position.set((cx + tgt.x) / 2, 0.76, (cz + tgt.z) / 2);
        trace.rotation.y = -Math.atan2(cz - tgt.z, cx - tgt.x);
        chipGroup.add(trace);
      }
    }
    // Insertion guide ridge
    const ridge = new THREE.Mesh(new THREE.BoxGeometry(58, 0.3, 0.4), new THREE.MeshStandardMaterial({ color: 0xB8860B, roughness: 0.4, metalness: 0.3 }));
    ridge.position.set(24, 0.78, 19.5);
    chipGroup.add(ridge);

    // ══════════ MODE 2: CROSS-SECTION ══════════
    const crossGroup = new THREE.Group();
    crossGroup.visible = false;
    scene.add(crossGroup);

    // Layer stack
    const layers = [
      { h: 2.0, color: 0xD4A76A, rough: 0.35, metal: 0 },   // Kapton
      { h: 1.5, color: 0x2D2D2D, rough: 0.9, metal: 0 },    // LIG
      { h: 0.3, color: 0xFFD700, rough: 0.2, metal: 0.8 },   // AuNP base
    ];
    let yOff = 0;
    layers.forEach(l => {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(6, l.h, 6),
        new THREE.MeshStandardMaterial({ color: l.color, roughness: l.rough, metalness: l.metal })
      );
      m.position.y = yOff + l.h / 2;
      m.castShadow = true;
      crossGroup.add(m);
      yOff += l.h;
    });

    // AuNP spheres
    const auSurfMat = new THREE.MeshStandardMaterial({ color: 0xFFD700, metalness: 0.9, roughness: 0.1 });
    for (let i = 0; i < 40; i++) {
      const au = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 8), auSurfMat);
      au.position.set((Math.random() - 0.5) * 5, yOff + Math.random() * 0.1, (Math.random() - 0.5) * 5);
      crossGroup.add(au);
    }
    const baseY = yOff + 0.15;

    // ssDNA + MB + MCH
    const strandMat = new THREE.MeshStandardMaterial({ color: 0x66c2a5 });
    const strandCutMat = new THREE.MeshStandardMaterial({ color: 0x3d7a63 });
    const mbMat = new THREE.MeshStandardMaterial({ color: 0x3288bd, emissive: 0x1a4470, emissiveIntensity: 0.3 });
    const mchMat = new THREE.MeshStandardMaterial({ color: 0xAAAAAA });

    const reporters = [];
    for (let i = 0; i < 20; i++) {
      const x = (Math.random() - 0.5) * 4.5;
      const z = (Math.random() - 0.5) * 4.5;
      const h = 2.5 + Math.random() * 0.5;
      const rx = (Math.random() - 0.5) * 0.25;
      const rz = (Math.random() - 0.5) * 0.25;

      const strand = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, h, 4), strandMat.clone());
      strand.position.set(x, baseY + h / 2, z);
      strand.rotation.x = rx; strand.rotation.z = rz;
      strand.userData._rx0 = rx; strand.userData._rz0 = rz;
      crossGroup.add(strand);

      const tipY = baseY + h;
      const mb = new THREE.Mesh(new THREE.SphereGeometry(0.15, 8, 8), mbMat.clone());
      mb.position.set(x + rz * h * 0.3, tipY, z - rx * h * 0.3);
      mb.userData._basePos = mb.position.clone();
      crossGroup.add(mb);

      const cutH = h * (0.3 + Math.random() * 0.2);
      const stub = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, cutH, 4), strandCutMat);
      stub.position.set(x, baseY + cutH / 2, z);
      stub.rotation.x = rx; stub.rotation.z = rz;
      stub.visible = false;
      crossGroup.add(stub);

      reporters.push({ strand, mb, stub, fullH: h, cutH, x, z, rx, rz });
    }
    for (let i = 0; i < 30; i++) {
      const mch = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.3, 4), mchMat);
      mch.position.set((Math.random() - 0.5) * 4.5, baseY + 0.15, (Math.random() - 0.5) * 4.5);
      crossGroup.add(mch);
    }

    // ── Camera orbit ──
    let orbit = { theta: 0.3, phi: -0.45, dist: 90, target: new THREE.Vector3(10, 0, 2) };
    let tgtOrbit = { theta: 0.3, phi: -0.45, dist: 90, target: new THREE.Vector3(10, 0, 2) };
    let isDragging = false, prevMouse = { x: 0, y: 0 };
    const canvas = renderer.domElement;

    const onDown = (e) => { isDragging = true; const p = e.touches ? e.touches[0] : e; prevMouse = { x: p.clientX, y: p.clientY }; };
    const onUp = () => { isDragging = false; };
    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      // Hover detection
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
          canvas.style.cursor = isDragging ? "grabbing" : "grab";
          stateRef.current._hovIdx = -1;
          setTooltipInfo(null);
        }
      }
      if (!isDragging) return;
      const dx = p.clientX - prevMouse.x;
      const dy = p.clientY - prevMouse.y;
      tgtOrbit.theta += dx * 0.005;
      tgtOrbit.phi = Math.max(-1.3, Math.min(-0.08, tgtOrbit.phi + dy * 0.005));
      prevMouse = { x: p.clientX, y: p.clientY };
    };
    const onWheel = (e) => { e.preventDefault(); tgtOrbit.dist = Math.max(15, Math.min(150, tgtOrbit.dist + e.deltaY * 0.08)); };
    const raycaster = new THREE.Raycaster();

    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(mx, my), camera);
      const hits = raycaster.intersectObjects(padMeshes);
      if (hits.length > 0) {
        const ud = hits[0].object.userData;
        stateRef.current._selectPad(ud.idx, ud.target);
      }
    };

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("touchstart", onDown, { passive: true });
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("touchmove", onMove, { passive: true });
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("click", onClick);

    // ── Animation ──
    let frameId, time = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      time += 0.016;

      // Smooth orbit
      orbit.theta += (tgtOrbit.theta - orbit.theta) * 0.07;
      orbit.phi += (tgtOrbit.phi - orbit.phi) * 0.07;
      orbit.dist += (tgtOrbit.dist - orbit.dist) * 0.07;
      orbit.target.lerp(tgtOrbit.target instanceof THREE.Vector3 ? tgtOrbit.target : new THREE.Vector3(tgtOrbit.target.x, tgtOrbit.target.y, tgtOrbit.target.z), 0.07);

      camera.position.x = orbit.target.x + orbit.dist * Math.sin(orbit.theta) * Math.cos(orbit.phi);
      camera.position.y = orbit.target.y + orbit.dist * Math.sin(-orbit.phi);
      camera.position.z = orbit.target.z + orbit.dist * Math.cos(orbit.theta) * Math.cos(orbit.phi);
      camera.lookAt(orbit.target);

      // ssDNA sway + MB glow
      reporters.forEach((r, i) => {
        if (r.strand.visible) {
          r.strand.rotation.x = r.rx + Math.sin(time * 1.9 + i * 0.7) * 0.04;
          r.strand.rotation.z = r.rz + Math.cos(time * 1.5 + i * 1.1) * 0.04;
        }
        if (r.mb.visible) {
          r.mb.material.emissiveIntensity = 0.2 + 0.15 * Math.sin(time * 3.14 + i);
        }
      });


      renderer.render(scene, camera);
    };
    animate();

    // Resize
    const onResize = () => {
      const w = container.clientWidth;
      const h2 = Math.round(w * 9 / 16);
      container.style.height = h2 + "px";
      renderer.setSize(w, h2);
      camera.aspect = w / h2;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    stateRef.current = {
      chipGroup, crossGroup, reporters, padMeshes, orbit, tgtOrbit,
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
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
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
      s.chipGroup.visible = true;
      s.crossGroup.visible = false;
      s.tgtOrbit.dist = 90;
      s.tgtOrbit.theta = 0.3;
      s.tgtOrbit.phi = -0.45;
      s.tgtOrbit.target = new THREE.Vector3(10, 0, 2);
    } else if (mode === 2) {
      s.chipGroup.visible = false;
      s.crossGroup.visible = true;
      s.tgtOrbit.dist = 16;
      s.tgtOrbit.theta = 0.4;
      s.tgtOrbit.phi = -0.3;
      s.tgtOrbit.target = new THREE.Vector3(0, 2.5, 0);
    }
  }, [mode, selectedPad]);

  // Cas12a toggle
  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    s.reporters.forEach(r => {
      r.strand.visible = !cas12aActive;
      r.mb.visible = !cas12aActive;
      r.stub.visible = cas12aActive;
    });
  }, [cas12aActive]);

  const deltaI = selectedPad && computeGamma && echemGamma0_mol
    ? (() => {
        const eff = getEfficiency(selectedPad.target);
        const G = computeGamma(echemTime * 60, eff, echemKtrans);
        return ((1 - G / echemGamma0_mol) * 100).toFixed(1);
      })()
    : null;
  const selDrug = selectedPad ? targetDrug(selectedPad.target) : null;
  const selEff = selectedPad ? getEfficiency(selectedPad.target) : null;
  const selStrat = selectedPad ? targetStrategy(selectedPad.target) : null;
  const selR = selectedPad ? results.find(x => x.label === selectedPad.target) : null;
  const selDisc = selR?.disc && selR.disc < 900 ? selR.disc : null;

  return (
    <div style={{ position: "relative", borderRadius: "10px", overflow: "hidden", background: "#F8F9FA" }}>
      <div ref={mountRef} style={{ width: "100%", minHeight: 280 }} />

      {/* ── Mode 1 overlays ── */}
      {mode === 1 && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: HEADING, textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}>
            GUARD MDR-TB Diagnostic Chip
          </div>
          <div style={{ position: "absolute", top: 14, right: 16, fontSize: 10, color: "#6B7280", textShadow: "0 1px 2px rgba(255,255,255,0.8)" }}>
            Click pad to inspect · Drag to rotate · Scroll to zoom
          </div>
          <div style={{ position: "absolute", bottom: 12, left: 16, display: "flex", gap: 6, flexWrap: "wrap" }}>
            {Object.entries(DRUG_CSS).map(([d, c]) => (
              <span key={d} style={{ fontSize: 9, fontWeight: 700, fontFamily: MONO, display: "flex", alignItems: "center", gap: 3, background: "rgba(255,255,255,0.88)", padding: "2px 7px", borderRadius: 4 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: c, display: "inline-block" }} />{d}
              </span>
            ))}
          </div>
          <div style={{ position: "absolute", bottom: 12, right: 16, fontSize: 9, color: "#9CA3AF", background: "rgba(255,255,255,0.88)", padding: "2px 8px", borderRadius: 4, fontFamily: MONO }}>
            65 × 35 mm
          </div>
        </>
      )}

      {/* ── Mode 2 overlays ── */}
      {mode === 2 && selectedPad && (
        <>
          <div style={{ position: "absolute", top: 12, left: 16, background: "rgba(255,255,255,0.9)", padding: "8px 14px", borderRadius: 8, border: "1px solid #E3E8EF" }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#111827", fontFamily: HEADING }}>
              {selectedPad.target} · <span style={{ color: DRUG_CSS[selDrug] || "#888" }}>{selDrug}</span>
            </div>
            <div style={{ fontSize: 10, color: "#6B7280", fontFamily: MONO, marginTop: 2 }}>
              S_eff = {selEff?.toFixed(3)} {selDisc ? `· D = ${selDisc.toFixed(1)}×` : ""} · {selStrat}
            </div>
          </div>
          <button
            onClick={() => stateRef.current?._toMode1()}
            style={{ position: "absolute", top: 12, right: 16, fontSize: 11, fontWeight: 600, padding: "7px 16px", borderRadius: 6, border: "1px solid #E3E8EF", background: "#fff", cursor: "pointer", fontFamily: HEADING, color: "#374151" }}
          >
            ← Back to chip
          </button>
          {/* Layer labels */}
          <div style={{ position: "absolute", left: 16, top: "38%", transform: "translateY(-50%)", display: "flex", flexDirection: "column", gap: 5 }}>
            {[
              { label: "MB (E° = −0.22 V, n = 2)", color: "#3288bd" },
              { label: "ssDNA reporter (12–20 nt)", color: "#66c2a5" },
              { label: "MCH backfill (C6-thiol)", color: "#AAAAAA" },
              { label: "AuNP (thiol-Au, 170 kJ/mol)", color: "#FFD700" },
              { label: "LIG (23 Ω/sq, k₀ ≈ 0.01 cm/s)", color: "#2D2D2D" },
              { label: "Kapton (125 μm, 10¹⁷ Ω·cm)", color: "#D4A76A" },
            ].map(l => (
              <div key={l.label} style={{ fontSize: 9, fontFamily: MONO, color: "#374151", background: "rgba(255,255,255,0.9)", padding: "3px 8px", borderRadius: 4, borderLeft: `3px solid ${l.color}`, lineHeight: 1.3 }}>
                ← {l.label}
              </div>
            ))}
          </div>
          {/* Cas12a toggle */}
          <div style={{ position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 12, alignItems: "center", background: "rgba(255,255,255,0.94)", padding: "8px 18px", borderRadius: 8, border: "1px solid #E3E8EF" }}>
            <button
              onClick={() => setCas12aActive(!cas12aActive)}
              style={{
                fontSize: 11, fontWeight: 700, padding: "7px 16px", borderRadius: 6, cursor: "pointer", fontFamily: MONO,
                background: cas12aActive ? "#DC2626" : "#16A34A", color: "#fff", border: "none",
                transition: "background 0.2s",
              }}
            >
              {cas12aActive ? "Reset reporters" : "Activate Cas12a"}
            </button>
            {cas12aActive && deltaI != null && (
              <span style={{ fontSize: 13, fontWeight: 700, fontFamily: MONO, color: "#2563EB" }}>
                ΔI% = {deltaI}%
              </span>
            )}
          </div>
        </>
      )}

      {/* Tooltip */}
      {tooltipInfo && mode === 1 && (
        <div style={{
          position: "absolute", left: tooltipPos.x + 14, top: tooltipPos.y - 12, pointerEvents: "none",
          background: "rgba(255,255,255,0.96)", border: "1px solid #E3E8EF", borderRadius: 6,
          padding: "8px 12px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)", zIndex: 10,
        }}>
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
