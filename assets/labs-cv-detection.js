/* Computer Vision · Module 6 (object detection) labs.
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes).
 *
 *   <div data-lab="detiou"></div>          IoU / GIoU with draggable, resizable boxes and a sliding sweep
 *   <div data-lab="detslide"></div>        sliding window vs R-CNN vs Fast R-CNN vs Faster R-CNN cost
 *   <div data-lab="detanchors"></div>      anchors on a feature-map grid, IoU matching, offset regression
 *   <div data-lab="detyolo"></div>         YOLO grid, responsible cell, output tensor, heatmap
 *   <div data-lab="detnms"></div>          greedy NMS / soft-NMS step by step
 *   <div data-lab="detdetr"></div>         object queries + Hungarian matching on an editable cost matrix
 *   <div data-lab="detpr"></div>           ranked detections -> precision/recall curve -> AP, mAP
 *   <div data-lab="detblueprint" data-models="frcnn,yolov8,yolov1,detr"></div>   animated architecture blueprints
 */
(function () {
  const CV = AI.cv, clamp = AI.clamp, lerp = AI.lerp, smooth = AI.smooth, fmt = AI.fmt;
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const rgba = (hex, a) => { const c = AI.rgb(hex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
  const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : '–');
  const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–');

  /* ---------- box maths (boxes are [x1,y1,x2,y2]) ---------- */
  const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const interBox = (a, b) => [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
  const interArea = (a, b) => area(interBox(a, b));
  const iouOf = (a, b) => { const I = interArea(a, b), U = area(a) + area(b) - I; return U > 0 ? I / U : 0; };
  const encBox = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
  const giouOf = (a, b) => { const I = interArea(a, b), U = area(a) + area(b) - I, Cc = area(encBox(a, b)); return Cc > 0 ? (U > 0 ? I / U : 0) - (Cc - U) / Cc : 0; };
  const cxcywh = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]];
  const fromCxcywh = (c) => [c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2];
  const encode = (an, gt) => { const a = cxcywh(an), g = cxcywh(gt); return [(g[0] - a[0]) / a[2], (g[1] - a[1]) / a[3], Math.log(g[2] / a[2]), Math.log(g[3] / a[3])]; };
  const decode = (an, t) => { const a = cxcywh(an); return fromCxcywh([a[0] + t[0] * a[2], a[1] + t[1] * a[3], a[2] * Math.exp(t[2]), a[3] * Math.exp(t[3])]); };

  /* ---------- canvas helpers (draw in a fixed "world", scaled to fit) ---------- */
  const fitW = (c, W, H) => { const s = Math.min(c.w / W, c.h / H); return { s, ox: (c.w - W * s) / 2, oy: (c.h - H * s) / 2, W, H }; };
  const enter = (c, m) => { const g = c.ctx; g.clearRect(0, 0, c.w, c.h); g.save(); g.translate(m.ox, m.oy); g.scale(m.s, m.s); return g; };
  const drawBox = (g, m, b, col, o = {}) => {
    g.beginPath(); g.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    if (o.fill) { g.fillStyle = rgba(col, o.fill); g.fill(); }
    g.lineWidth = (o.lw || 2) / m.s; g.strokeStyle = col; g.setLineDash(o.dash ? o.dash.map((d) => d / m.s) : []);
    g.stroke(); g.setLineDash([]);
  };
  const label = (g, m, s, x, y, col, o = {}) => {
    const px = (o.size || 12) / m.s;
    g.font = `${o.bold ? 'bold ' : ''}${px}px ${MONO}`; g.textAlign = o.align || 'left'; g.textBaseline = 'middle';
    if (o.bg) { const w = g.measureText(s).width, bx = o.align === 'center' ? x - w / 2 : o.align === 'right' ? x - w : x; g.fillStyle = o.bg; g.fillRect(bx - 3 / m.s, y - px * 0.68, w + 6 / m.s, px * 1.36); }
    g.fillStyle = col; g.fillText(s, x, y);
  };
  const line = (g, m, x0, y0, x1, y1, col, lw = 1.5, dash) => {
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.lineWidth = lw / m.s; g.strokeStyle = col;
    g.setLineDash(dash ? dash.map((d) => d / m.s) : []); g.stroke(); g.setLineDash([]);
  };
  let sceneCv = null;
  const sceneImg = () => {
    if (sceneCv) return sceneCv;
    const d = CV.scene('shapes'), c = document.createElement('canvas'); c.width = c.height = 120;
    const g = c.getContext('2d'), id = g.createImageData(120, 120); id.data.set(d); g.putImageData(id, 0, 0);
    return (sceneCv = c);
  };
  /* the 'shapes' scene upscaled x4 to 480x480; ground-truth boxes of its four objects */
  const OBJ = [
    { n: 'circle', b: [64, 72, 240, 248], k: 'red' },
    { n: 'square', b: [264, 248, 416, 400], k: 'blue' },
    { n: 'triangle', b: [104, 296, 216, 432], k: 'green' },
    { n: 'sun', b: [332, 52, 420, 140], k: 'yellow' },
  ];

  /* ---------- UI helpers ---------- */
  const wide = (root) => { const b = root.querySelector('.lab-body'), v = root.querySelector('.lab-view'), c = root.querySelector('.lab-ctl'); b.style.gridTemplateColumns = '1fr'; v.style.borderRight = '0'; v.style.borderBottom = '1px solid var(--border)'; c.style.display = 'grid'; c.style.gridTemplateColumns = 'repeat(auto-fit,minmax(230px,1fr))'; c.style.alignItems = 'start'; };
  const chip = (parent, text, fn) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = text; b.onclick = fn; parent.appendChild(b); return b; };
  const checkRow = (id, text, on) => `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="${id}"${on ? ' checked' : ''}> ${text}</label>`;
  const transport = (def) => `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="${def || 1}"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="back">◀ Step</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">↺ Restart</button></div>`;

  /** Step player: pos runs 0..n in real time (dur(i) seconds per step, scaled by the speed slider). */
  function player(root, cfg) {
    const q = (s) => AI.$(s, root);
    const P = { pos: 0, playing: true, n: cfg.n, dur: cfg.dur || (() => 1.6), hold: 1.2, onJump: cfg.onJump || (() => {}) };
    P.idx = () => clamp(Math.floor(P.pos), 0, P.n - 1);
    P.frac = () => (P.pos >= P.n ? 1 : P.pos - Math.floor(P.pos));
    P.set = (i) => { P.pos = clamp(i, 0, P.n - 1) + 0.999; P.onJump(); };
    P.restart = () => { P.pos = 0; P.playing = true; q('#play').textContent = '⏸ Pause'; P.onJump(); };
    P.update = (dt) => {
      if (root.dataset.start !== undefined && !P._started) { P._started = 1; P.pos = +root.dataset.start + 0.7; P.playing = false; q('#play').textContent = '▶ Play'; }
      q('#spv').textContent = (+q('#sp').value).toFixed(2) + '×';
      if (!P.playing) return;
      P.pos += (dt * +q('#sp').value) / P.dur(P.idx());
      if (P.pos >= P.n + P.hold) P.pos = 0;
    };
    q('#play').onclick = () => { P.playing = !P.playing; q('#play').textContent = P.playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { P.playing = false; q('#play').textContent = '▶ Play'; P.set(P.pos >= P.n - 0.5 ? 0 : P.idx() + 1); };
    q('#back').onclick = () => { P.playing = false; q('#play').textContent = '▶ Play'; P.set(P.idx() - 1); };
    q('#reset').onclick = () => P.restart();
    return P;
  }

  /** Pointer editor for boxes on a canvas. items: [{b, resize, name}] (arrays edited in place). */
  function editor(cv, getM, items, opts) {
    let drag = null;
    const pt = (e) => { const r = cv.getBoundingClientRect(), m = getM(); return [(e.clientX - r.left - m.ox) / m.s, (e.clientY - r.top - m.oy) / m.s]; };
    const hit = (p) => {
      const m = getM(), tol = 11 / m.s;
      for (let k = items.length - 1; k >= 0; k--) {
        const it = items[k], b = it.b;
        if (it.resize) for (let c = 0; c < 4; c++) if (Math.abs(p[0] - (c & 1 ? b[2] : b[0])) < tol && Math.abs(p[1] - (c & 2 ? b[3] : b[1])) < tol) return { k, c };
      }
      for (let k = items.length - 1; k >= 0; k--) { const b = items[k].b; if (items[k].fixed) continue; if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]) return { k, c: -1 }; }
      return null;
    };
    cv.addEventListener('pointerdown', (e) => {
      const p = pt(e), h = hit(p); if (!h) return;
      drag = { k: h.k, c: h.c, p0: p, b0: items[h.k].b.slice() }; cv.setPointerCapture(e.pointerId);
      if (opts.onStart) opts.onStart(h.k); e.preventDefault();
    });
    cv.addEventListener('pointermove', (e) => {
      const p = pt(e);
      if (!drag) { const h = hit(p); cv.style.cursor = !h ? 'default' : h.c < 0 ? 'move' : 'nwse-resize'; return; }
      const b = items[drag.k].b, b0 = drag.b0, dx = p[0] - drag.p0[0], dy = p[1] - drag.p0[1], W = opts.W, H = opts.H;
      if (drag.c < 0) {
        const w = b0[2] - b0[0], h = b0[3] - b0[1], x = clamp(b0[0] + dx, 0, W - w), y = clamp(b0[1] + dy, 0, H - h);
        b[0] = x; b[1] = y; b[2] = x + w; b[3] = y + h;
      } else {
        const xi = drag.c & 1 ? 2 : 0, yi = drag.c & 2 ? 3 : 1;
        b[xi] = clamp(b0[xi] + dx, 0, W); b[yi] = clamp(b0[yi] + dy, 0, H);
        if (b[0] > b[2]) { const t = b[0]; b[0] = b[2]; b[2] = t; drag.c ^= 1; const t2 = b0[0]; b0[0] = b0[2]; b0[2] = t2; }
        if (b[1] > b[3]) { const t = b[1]; b[1] = b[3]; b[3] = t; drag.c ^= 2; const t2 = b0[1]; b0[1] = b0[3]; b0[3] = t2; }
      }
      opts.onChange(drag.k);
    });
    const up = () => { if (drag && opts.onEnd) opts.onEnd(drag.k); drag = null; };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  }

  /* ======================================================================================
   *  1. IoU lab
   * ====================================================================================== */
  AI.labs.detiou = function (root) {
    const W = 640, H = 400;
    const { view, ctl } = AI.labShell(root, 'Intersection over Union & GIoU', 'drag a box by its body, resize it by its corners — or let box B slide across A',
      `<canvas id="c" style="aspect-ratio:640/400;touch-action:none"></canvas><canvas id="p" style="height:160px"></canvas><div class="caption" id="cap"></div>`,
      `<div>Presets<div class="chips" id="pre" style="margin-top:4px"></div></div>
       <div>Box A (blue): x1 y1 x2 y2<div id="ia" style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:4px"></div></div>
       <div>Box B (orange): x1 y1 x2 y2<div id="ib" style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:4px"></div></div>
       ${checkRow('enc', 'show the smallest enclosing box C (for GIoU)', true)}
       <div class="row"><button class="btn primary" id="sw">▶ Slide B across A</button></div>
       <label>Slide speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="2" step="0.05" value="1"></label>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c')), cp = AI.canvas(q('#p'));
    const A = [100, 100, 300, 260], B = [180, 150, 380, 330];
    let sweep = false, ph = -Math.PI / 2, dx = 0, last = '', curve = null, curveKey = '';
    const mk = (box, host) => [0, 1, 2, 3].map((i) => {
      const e = document.createElement('input'); e.type = 'number'; e.step = 5; e.style.textAlign = 'center';
      e.oninput = () => { const v = +e.value; if (Number.isFinite(v)) { box[i] = clamp(v, 0, i & 1 ? H : W); last = ''; } };
      q(host).appendChild(e); return e;
    });
    const inA = mk(A, '#ia'), inB = mk(B, '#ib');
    const sync = () => { [[A, inA], [B, inB]].forEach(([b, ins]) => ins.forEach((e, i) => { if (document.activeElement !== e) e.value = Math.round(b[i]); })); };
    const PRE = {
      'worked example': [[100, 100, 300, 260], [180, 150, 380, 330]], identical: [[150, 100, 350, 300], [150, 100, 350, 300]], 'B inside A': [[100, 60, 400, 340], [180, 120, 300, 260]],
      'just touching': [[100, 100, 260, 260], [260, 100, 420, 260]], 'near miss': [[100, 100, 260, 260], [300, 120, 460, 280]], 'far apart': [[60, 120, 200, 260], [440, 120, 580, 260]],
    };
    Object.entries(PRE).forEach(([n, [a, b]]) => chip(q('#pre'), n, () => { A.splice(0, 4, ...a); B.splice(0, 4, ...b); sweep = false; dx = 0; q('#sw').textContent = '▶ Slide B across A'; sync(); last = ''; }));
    q('#sw').onclick = () => { sweep = !sweep; if (!sweep) dx = 0; else ph = -Math.PI / 2; q('#sw').textContent = sweep ? '⏹ Stop sliding' : '▶ Slide B across A'; last = ''; };
    const getM = () => fitW(c, W, H);
    editor(c.cv, getM, [{ b: A, resize: true }, { b: B, resize: true }], { W, H, onStart: () => { if (sweep) { sweep = false; dx = 0; q('#sw').textContent = '▶ Slide B across A'; } }, onChange: () => { sync(); last = ''; } });
    const shifted = (b, d) => [b[0] + d, b[1], b[2] + d, b[3]];
    q('#enc').addEventListener('input', () => (last = ''));
    sync();

    AI.loop(root, (dt) => {
      q('#spv').textContent = (+q('#sp').value).toFixed(2) + '×';
      if (sweep) { ph += dt * 0.35 * +q('#sp').value; dx = 430 * Math.sin(ph); }
      const key = [A, B, dx.toFixed(1), q('#enc').checked, c.w, c.h, cp.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = getM(), Bd = shifted(B, dx), g = enter(c, m);
      // canvas background + grid
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
      for (let x = 0; x <= W; x += 80) line(g, m, x, 0, x, H, C.border, 1); for (let y = 0; y <= H; y += 80) line(g, m, 0, y, W, y, C.border, 1);
      g.save(); g.beginPath(); g.rect(0, 0, W, H); g.clip();
      const I = interBox(A, Bd), hasI = I[2] > I[0] && I[3] > I[1], enc = encBox(A, Bd);
      if (q('#enc').checked) drawBox(g, m, enc, C.purple, { dash: [6, 5], lw: 1.5, fill: 0.05 });
      drawBox(g, m, A, C.blue, { fill: 0.2, lw: 2.5 }); drawBox(g, m, Bd, C.orange, { fill: 0.2, lw: 2.5 });
      if (hasI) { drawBox(g, m, I, C.green, { fill: 0.55, lw: 2 }); label(g, m, 'A∩B', (I[0] + I[2]) / 2, (I[1] + I[3]) / 2, '#fff', { align: 'center', bold: true, size: 12 }); }
      [[A, C.blue], [Bd, C.orange]].forEach(([b, col], k) => { for (let i = 0; i < 4; i++) { const x = i & 1 ? b[2] : b[0], y = i & 2 ? b[3] : b[1], r = 5 / m.s; g.fillStyle = C.bg; g.strokeStyle = col; g.lineWidth = 2 / m.s; g.beginPath(); g.rect(x - r, y - r, 2 * r, 2 * r); g.fill(); g.stroke(); } label(g, m, k ? 'B' : 'A', b[0] + 8, b[1] + 14, col, { bold: true, size: 14 }); });
      g.restore();
      label(g, m, 'image 640 × 400 px', 6, H - 10, C.muted, { size: 11 });
      g.restore();
      // numbers
      const iA = area(A), iB = area(Bd), iI = interArea(A, Bd), iU = iA + iB - iI, iC = area(enc), iou = iU > 0 ? iI / iU : 0, giou = iC > 0 ? iou - (iC - iU) / iC : 0;
      const ca = cxcywh(A), cb = cxcywh(Bd), d2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2, c2 = (enc[2] - enc[0]) ** 2 + (enc[3] - enc[1]) ** 2, diou = iou - (c2 > 0 ? d2 / c2 : 0);
      const wh = (b) => `${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}`;
      q('#out').innerHTML = `A  ${wh(A)} = <b>${Math.round(iA)}</b>\nB  ${wh(Bd)} = <b>${Math.round(iB)}</b>\nA∩B ${hasI ? wh(I) : '0×0'} = <b class="g">${Math.round(iI)}</b>\nA∪B ${Math.round(iA)}+${Math.round(iB)}−${Math.round(iI)} = <b class="b">${Math.round(iU)}</b>\n<b class="o">IoU = ${f3(iou)}</b>\nC   area ${Math.round(iC)}\n<b class="o">GIoU = ${f3(giou)}</b>\nDIoU = ${f3(diou)}`;
      q('#cap').textContent = !hasI ? 'The boxes do not overlap: the intersection is empty, so IoU = 0 — and it stays 0 however far apart they are. GIoU is still informative: it falls towards −1 as the boxes drift apart, so a loss built on it still gives a direction to move.'
        : iou > 0.95 ? 'Almost identical boxes: IoU ≈ 1 (the best possible).' : iou >= 0.5 ? `IoU ${f2(iou)} ≥ 0.5: this would count as a correct detection under the classic PASCAL VOC rule.` : `IoU ${f2(iou)} < 0.5: overlapping, but too loosely to count as a match at the usual 0.5 threshold.`;
      // curve: IoU & GIoU as B slides horizontally
      const ck = [A, B].join();
      if (ck !== curveKey) { curveKey = ck; curve = []; for (let k = 0; k <= 120; k++) { const d = -450 + (k * 900) / 120, bb = shifted(B, d); curve.push([d, iouOf(A, bb), giouOf(A, bb)]); } }
      const v = AI.view(cp, -450, 450, -1, 1, { l: 38, r: 10, t: 10, b: 22 }), g2 = cp.ctx; g2.clearRect(0, 0, cp.w, cp.h);
      AI.axes(g2, v);
      [[1, C.blue, 'IoU'], [2, C.orange, 'GIoU']].forEach(([i, col, nm]) => { g2.beginPath(); curve.forEach((p, k) => (k ? g2.lineTo(v.X(p[0]), v.Y(p[i])) : g2.moveTo(v.X(p[0]), v.Y(p[i])))); g2.strokeStyle = col; g2.lineWidth = 2.2; g2.stroke(); });
      g2.font = `11px ${MONO}`; g2.textAlign = 'left'; g2.fillStyle = C.blue; g2.fillText('IoU', v.X(-440), v.Y(0.92)); g2.fillStyle = C.orange; g2.fillText('GIoU', v.X(-440), v.Y(-0.85));
      g2.fillStyle = C.muted; g2.textAlign = 'right'; g2.fillText('B shifted by Δx →', v.X(445), v.Y(-0.1));
      g2.strokeStyle = C.fg; g2.lineWidth = 1.5; g2.beginPath(); g2.moveTo(v.X(dx), v.pad.t); g2.lineTo(v.X(dx), v.pad.t + v.ph); g2.stroke();
      g2.fillStyle = C.blue; g2.beginPath(); g2.arc(v.X(dx), v.Y(iou), 4.5, 0, 7); g2.fill(); g2.fillStyle = C.orange; g2.beginPath(); g2.arc(v.X(dx), v.Y(giou), 4.5, 0, 7); g2.fill();
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  2. sliding window vs proposals
   * ====================================================================================== */
  AI.labs.detslide = function (root) {
    const N = 480, PW = 250, W = N + PW + 24, H = N;
    const { view, ctl } = AI.labShell(root, 'How many times must the network run?', 'a window scanning the image, versus proposals, versus computing features once',
      '<canvas id="c" style="aspect-ratio:744/480"></canvas><div class="caption" id="cap"></div>',
      `<label>Method <select id="mode"><option value="slide">Sliding window (3 sizes)</option><option value="rcnn">R-CNN: proposals, CNN on every crop</option><option value="fast">Fast R-CNN: CNN once, RoI pooling</option><option value="faster">Faster R-CNN: CNN once + learned RPN</option></select></label>
       <label>Window stride (px) <span class="v" id="stv"></span><input type="range" id="stride" min="8" max="96" step="8" value="48"></label>
       ${checkRow('ar', 'three aspect ratios (1:2, 1:1, 2:1)', false)}
       <label>Proposals N <span class="v" id="nv"></span><input type="range" id="np" min="50" max="2000" step="50" value="300"></label>
       <label>One CNN pass per crop (ms) <input type="number" id="ms" value="15" min="1" max="500"></label>
       <label>Animation speed <span class="v" id="rv"></span><input type="range" id="sp" min="1" max="40" value="8"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">↺ Restart</button></div>
       <div class="readout" id="out" style="font-size:11.5px"></div>`);
    wide(root);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c')), img = sceneImg();
    const rnd = AI.rng(11), PROPS = [];
    for (let i = 0; i < 2000; i++) {
      let b;
      if (i % 3 !== 2) { const o = OBJ[Math.floor(rnd() * 4)].b, w = o[2] - o[0], h = o[3] - o[1], cx = (o[0] + o[2]) / 2 + (rnd() - 0.5) * 0.35 * w, cy = (o[1] + o[3]) / 2 + (rnd() - 0.5) * 0.35 * h, sw = w * (0.75 + rnd() * 0.5), sh = h * (0.75 + rnd() * 0.5); b = [cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2]; }
      else { const w = 50 + rnd() * 200, h = 50 + rnd() * 200, x = rnd() * (N - w), y = rnd() * (N - h); b = [x, y, x + w, y + h]; }
      PROPS.push([clamp(b[0], 0, N), clamp(b[1], 0, N), clamp(b[2], 0, N), clamp(b[3], 0, N)]);
    }
    const FEAT = CV.resample('shapes', 30), FG = CV.gray(FEAT);
    let k = 0, done = 0, pre = 0, playing = true, hits = [], last = '', cfgKey = '', cfgs = [], cum = [], total = 0, mode = 'slide', stepReq = 0;
    const setup = () => {
      mode = q('#mode').value; const stride = +q('#stride').value, ratios = q('#ar').checked ? [0.5, 1, 2] : [1];
      cfgs = []; cum = []; total = 0;
      [64, 128, 256].forEach((sc) => ratios.forEach((r) => { const w = Math.round(sc * Math.sqrt(r)), h = Math.round(sc / Math.sqrt(r)), nx = w <= N ? Math.floor((N - w) / stride) + 1 : 0, ny = h <= N ? Math.floor((N - h) / stride) + 1 : 0; cfgs.push({ w, h, nx, ny, n: nx * ny, start: total }); total += nx * ny; }));
    };
    const winAt = (i) => { for (const f of cfgs) if (i < f.start + f.n) { const j = i - f.start, ix = j % f.nx, iy = Math.floor(j / f.nx), s = +q('#stride').value; return [ix * s, iy * s, ix * s + f.w, iy * s + f.h]; } return null; };
    const boxAt = (i) => (mode === 'slide' ? winAt(i) : PROPS[i]);
    const count = () => (mode === 'slide' ? total : +q('#np').value);
    const reset = () => { setup(); k = 0; done = 0; pre = 0; hits = []; last = ''; };
    ['#mode', '#stride', '#ar', '#np'].forEach((s) => q(s).addEventListener('input', reset));
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; if (mode !== 'slide' && pre < 1) pre = 1; else stepReq++; };
    q('#reset').onclick = () => { reset(); playing = true; q('#play').textContent = '⏸ Pause'; };
    setup();
    const bestObj = (b) => { let bi = -1, bv = 0; OBJ.forEach((o, i) => { const v = iouOf(b, o.b); if (v > bv) { bv = v; bi = i; } }); return [bi, bv]; };

    AI.loop(root, (dt) => {
      const rate = +q('#sp').value, ms = Math.max(1, +q('#ms').value || 15), n = count();
      q('#stv').textContent = q('#stride').value + ' px'; q('#nv').textContent = q('#np').value; q('#rv').textContent = rate + ' boxes/s';
      q('#stride').disabled = mode !== 'slide'; q('#ar').disabled = mode !== 'slide'; q('#np').disabled = mode === 'slide';
      if (playing) { if (mode !== 'slide' && pre < 1) pre = Math.min(1, pre + dt / 3.5); else k += dt * rate; }
      if (stepReq) { k += stepReq; stepReq = 0; }
      if (k >= n + 3) { k = 0; done = 0; hits = []; }
      const cur = Math.min(n - 1, Math.floor(k)), upto = Math.min(n, Math.floor(k) + 1);
      for (; done < upto; done++) { const b = boxAt(done); if (b) { const [oi, v] = bestObj(b); if (v >= 0.5 && hits.length < 400) hits.push([b, oi]); } }
      const key = [k.toFixed(2), pre.toFixed(2), mode, n, hits.length, c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = fitW(c, W, H), g = enter(c, m);
      g.drawImage(img, 0, 0, N, N);
      g.fillStyle = 'rgba(0,0,0,0.12)'; g.fillRect(0, 0, N, N);
      const cb = boxAt(cur), px = N + 24;
      const ready = mode === 'slide' || pre >= 1;
      // feature-map sweep for shared-backbone modes
      if (mode !== 'slide' && pre < 1) { const x = pre * N; g.fillStyle = rgba(C.blue, 0.25); g.fillRect(0, 0, x, N); line(g, m, x, 0, x, N, C.blue, 3); label(g, m, 'backbone runs ONCE over the whole image…', 12, 22, C.fg, { size: 13, bg: rgba(C.bg, 0.75), bold: true }); }
      if (ready) {
        hits.forEach(([b, oi]) => drawBox(g, m, b, C[OBJ[oi].k] || C.green, { lw: 1, fill: 0.04 }));
        for (let t = Math.max(0, cur - 5); t < cur; t++) { const b = boxAt(t); if (b) drawBox(g, m, b, C.muted, { lw: 1, dash: [3, 3] }); }
        if (cb) drawBox(g, m, cb, C.blue, { lw: 3.5, fill: 0.14 });
      }
      // right panel
      g.fillStyle = C.bg2; g.fillRect(px - 10, 0, PW + 10, H);
      label(g, m, mode === 'slide' || mode === 'rcnn' ? 'crop → resize → CNN' : 'shared feature map (stride 16)', px, 16, C.fg, { size: 12, bold: true });
      const tx = px, ty = 28, ts = 200;
      if (cb && ready && (mode === 'slide' || mode === 'rcnn')) {
        const sx = clamp(cb[0], 0, N), sy = clamp(cb[1], 0, N), sw = Math.max(2, Math.min(N, cb[2]) - sx), sh = Math.max(2, Math.min(N, cb[3]) - sy);
        g.drawImage(img, sx / 4, sy / 4, sw / 4, sh / 4, tx, ty, ts, ts); g.strokeStyle = C.blue; g.lineWidth = 2 / m.s; g.strokeRect(tx, ty, ts, ts);
      } else if (mode === 'fast' || mode === 'faster') {
        const cs = ts / 30, show = mode !== 'slide' ? pre : 1;
        for (let y = 0; y < 30; y++) for (let x = 0; x < 30; x++) { if (x / 30 > show) continue; const v = FG[y * 30 + x]; g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(tx + x * cs, ty + y * cs, cs + 0.5, cs + 0.5); }
        g.strokeStyle = C.border; g.lineWidth = 1 / m.s; g.strokeRect(tx, ty, ts, ts);
        if (cb && ready) { g.strokeStyle = C.blue; g.lineWidth = 2.5 / m.s; g.strokeRect(tx + (cb[0] / N) * ts, ty + (cb[1] / N) * ts, ((cb[2] - cb[0]) / N) * ts, ((cb[3] - cb[1]) / N) * ts); }
      }
      if (cb && ready) {
        const [oi, v] = bestObj(cb), cls = v >= 0.5 ? OBJ[oi].n : 'background';
        if (mode === 'fast' || mode === 'faster') { label(g, m, 'RoI Align → 7×7×256', tx, ty + ts + 16, C.fg, { size: 12, bold: true }); const cs2 = 11; for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) { g.fillStyle = rgba(C.blue, 0.15 + 0.4 * ((x * 7 + y * 3) % 5) / 5); g.fillRect(tx + x * cs2, ty + ts + 26 + y * cs2, cs2 - 1, cs2 - 1); } }
        const yy = ty + ts + (mode === 'fast' || mode === 'faster' ? 115 : 22);
        label(g, m, `→ ${cls}`, tx, yy, v >= 0.5 ? C[OBJ[oi].k] || C.green : C.muted, { size: 14, bold: true });
        label(g, m, `best IoU with an object: ${f2(v)}`, tx, yy + 18, C.muted, { size: 11 });
      }
      const t1 = (cur + 1) * ms;
      label(g, m, `boxes processed: ${Math.min(n, cur + 1)} / ${n}`, px, H - 44, C.fg, { size: 12, bold: true });
      label(g, m, mode === 'slide' || mode === 'rcnn' ? `CNN time so far ≈ ${fmt(t1 / 1000, 2)} s` : mode === 'fast' || mode === 'faster' ? 'backbone cost already paid' : '', px, H - 24, C.orange, { size: 12 });
      g.restore();
      // readout: all four methods side by side
      const cmpTotal = (() => { const s = total, np = +q('#np').value; return { slide: s * ms, rcnn: 2000 + np * ms, fast: 2000 + 150 + np * 0.085, faster: 150 + 10 + np * 0.085 }; })();
      const row = (id, name, n2, t) => `${mode === id ? '<b class="o">▶ ' : '  <span class="m">'}${name.padEnd(13)} ${String(n2).padStart(6)} CNN${id === 'fast' || id === 'faster' ? ' (×1)' : ' runs '} ${fmt(t / 1000, 2).padStart(7)} s${mode === id ? '</b>' : '</span>'}`;
      q('#out').innerHTML = `time per image (illustrative GPU numbers)\n${row('slide', 'sliding win.', total, cmpTotal.slide)}\n${row('rcnn', 'R-CNN', +q('#np').value, cmpTotal.rcnn)}\n${row('fast', 'Fast R-CNN', 1, cmpTotal.fast)}\n${row('faster', 'Faster R-CNN', 1, cmpTotal.faster)}\n<span class="m">R-CNN/Fast add ~2 s of selective search;\nFaster replaces it by a 10 ms RPN.</span>`;
      q('#cap').textContent = mode === 'slide' ? `Every window position, at every size, needs a full CNN pass. With stride ${q('#stride').value}px this picture already needs ${total} passes; a real photo with a stride of 8 px needs tens of thousands.`
        : mode === 'rcnn' ? 'R-CNN: a cheap algorithm proposes ~2000 regions; each is cut out, warped to 224×224 and sent through the CNN separately — 2000 full network passes (≈47 s per image on a 2014 GPU).'
          : mode === 'fast' ? 'Fast R-CNN: run the backbone once over the whole image. Every proposal is just a rectangle on the shared feature map, pooled to a fixed 7×7 grid and sent to small FC layers.'
            : 'Faster R-CNN: the proposals come from a small learned network (the RPN) that reads the same shared feature map, so almost nothing is computed twice.';
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  3. anchors
   * ====================================================================================== */
  AI.labs.detanchors = function (root) {
    const S = 512;
    const { view, ctl } = AI.labShell(root, 'Anchors: priors laid over the feature map', 'match to a ground-truth box by IoU, then learn the offsets (tx, ty, tw, th)',
      '<canvas id="c" style="aspect-ratio:1/1;max-height:520px;margin:0 auto;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Feature-map stride <select id="stride"><option value="128">128 px → 4×4 cells</option><option value="64" selected>64 px → 8×8 cells</option><option value="32">32 px → 16×16 cells</option></select></label>
       <div>Anchor scales (px)<div class="row" id="scs" style="margin-top:4px">${[64, 128, 256].map((s, i) => checkRow('s' + s, s, true)).join('')}</div></div>
       <div>Aspect ratios (height / width)<div class="row" id="rts" style="margin-top:4px">${[0.5, 1, 2].map((r) => checkRow('r' + r * 10, r, true)).join('')}</div></div>
       <label>Matching thresholds (positive / negative)<select id="thr"><option value="0.7,0.3">RPN: ≥ 0.7 / &lt; 0.3</option><option value="0.5,0.4" selected>RetinaNet: ≥ 0.5 / &lt; 0.4</option><option value="0.5,0.5">SSD: ≥ 0.5</option></select></label>
       ${transport(1)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c'));
    const GT = [176, 120, 368, 336]; let cell = null, last = '';
    const STEPS = ['① the feature map: one cell = one location, each owns K anchors', '② all anchors on all cells', '③ IoU of every anchor with the ground truth → positive / ignore / negative', '④ the best anchor learns to move: offsets (tx, ty, tw, th)', '⑤ decoding the offsets recovers the ground-truth box'];
    const P = player(root, { n: 5, dur: (i) => (i === 3 ? 5 : 3.6), onJump: () => (last = '') });
    const anchorsOf = () => {
      const st = +q('#stride').value, g = S / st, sc = [64, 128, 256].filter((s) => q('#s' + s).checked), rt = [0.5, 1, 2].filter((r) => q('#r' + r * 10).checked), out = [];
      for (let j = 0; j < g; j++) for (let i = 0; i < g; i++) for (const s of sc) for (const r of rt) { const w = s / Math.sqrt(r), h = s * Math.sqrt(r), cx = (i + 0.5) * st, cy = (j + 0.5) * st; out.push({ i, j, s, r, b: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] }); }
      return { st, g, K: sc.length * rt.length, list: out };
    };
    const getM = () => fitW(c, S, S);
    editor(c.cv, getM, [{ b: GT, resize: true }], { W: S, H: S, onChange: () => (last = '') });
    c.cv.addEventListener('click', (e) => { const r = c.cv.getBoundingClientRect(), m = getM(), st = +q('#stride').value, x = (e.clientX - r.left - m.ox) / m.s, y = (e.clientY - r.top - m.oy) / m.s; if (x >= 0 && y >= 0 && x < S && y < S && P.idx() === 0) { cell = [Math.floor(x / st), Math.floor(y / st)]; last = ''; } });
    ['#stride', '#scs', '#rts', '#thr'].forEach((s) => q(s).addEventListener('input', () => { cell = null; last = ''; }));

    AI.loop(root, (dt) => {
      P.update(dt);
      const idx = P.idx(), fr = P.frac(), key = [P.pos.toFixed(3), c.w, GT.join(), cell, q('#stride').value, q('#scs').textContent, [...root.querySelectorAll('input[type=checkbox]')].map((x) => +x.checked).join(''), q('#thr').value, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = getM(), g = enter(c, m), A = anchorsOf(), [hi, lo] = q('#thr').value.split(',').map(Number), st = A.st;
      const gradient = g.createLinearGradient(0, 0, S, S); gradient.addColorStop(0, rgba(C.blue, 0.1)); gradient.addColorStop(1, rgba(C.purple, 0.1)); g.fillStyle = gradient; g.fillRect(0, 0, S, S);
      for (let x = 0; x <= S; x += st) { line(g, m, x, 0, x, S, C.border, 1); line(g, m, 0, x, S, x, C.border, 1); }
      if (!cell) cell = [clamp(Math.floor(((GT[0] + GT[2]) / 2) / st), 0, A.g - 1), clamp(Math.floor(((GT[1] + GT[3]) / 2) / st), 0, A.g - 1)];
      cell = [Math.min(cell[0], A.g - 1), Math.min(cell[1], A.g - 1)];
      // iou of all anchors
      let best = null, pos = 0, neg = 0, ign = 0;
      A.list.forEach((a) => { a.iou = iouOf(a.b, GT); if (!best || a.iou > best.iou) best = a; });
      A.list.forEach((a) => { a.cls = a === best || a.iou >= hi ? 'pos' : a.iou < lo ? 'neg' : 'ign'; if (a.cls === 'pos') pos++; else if (a.cls === 'neg') neg++; else ign++; });
      drawBox(g, m, GT, C.green, { lw: 3, dash: [7, 5], fill: 0.07 }); label(g, m, 'ground truth', GT[0] + 4, GT[1] - 9, C.green, { size: 12, bold: true });
      const colOf = (a) => (a.cls === 'pos' ? C.green : a.cls === 'ign' ? C.yellow : C.muted);
      g.save(); g.beginPath(); g.rect(0, 0, S, S); g.clip();
      if (idx === 0) {
        g.fillStyle = rgba(C.blue, 0.18); g.fillRect(cell[0] * st, cell[1] * st, st, st);
        const mine = A.list.filter((a) => a.i === cell[0] && a.j === cell[1]), show = Math.ceil(smooth(Math.min(1, fr * 1.3)) * mine.length);
        mine.slice(0, show).forEach((a, t) => drawBox(g, m, a.b, [C.blue, C.orange, C.purple, C.pink, C.teal, C.red][t % 6], { lw: 2 }));
        g.fillStyle = C.fg; g.beginPath(); g.arc((cell[0] + 0.5) * st, (cell[1] + 0.5) * st, 4 / m.s, 0, 7); g.fill();
      } else if (idx === 1) {
        const upto = Math.floor(smooth(Math.min(1, fr * 1.15)) * A.list.length);
        A.list.slice(0, upto).forEach((a) => drawBox(g, m, a.b, rgba(C.blue, 0.5), { lw: 0.8 }));
        const gg = A.g; g.fillStyle = C.fg; for (let j = 0; j < gg; j++) for (let i = 0; i < gg; i++) { g.beginPath(); g.arc((i + 0.5) * st, (j + 0.5) * st, 2 / m.s, 0, 7); g.fill(); }
      } else if (idx === 2) {
        const reveal = smooth(Math.min(1, fr * 1.3));
        A.list.forEach((a) => { if (a.cls === 'neg') drawBox(g, m, a.b, rgba(C.muted, 0.25 * reveal + 0.1 * (1 - reveal)), { lw: 0.6 }); });
        A.list.forEach((a) => { if (a.cls === 'ign') drawBox(g, m, a.b, rgba(C.yellow, 0.7 * reveal), { lw: 1.2 }); });
        A.list.forEach((a) => { if (a.cls === 'pos') drawBox(g, m, a.b, rgba(C.green, 0.85 * reveal), { lw: a === best ? 4 : 1.8, fill: 0.08 * reveal }); });
      } else {
        const a = best, t = idx === 3 ? smooth(Math.min(1, fr * 1.25)) : 1, enc = encode(a.b, GT), cur = decode(a.b, enc.map((v) => v * t));
        drawBox(g, m, a.b, C.orange, { lw: 2.5, dash: [6, 4] }); label(g, m, 'anchor', a.b[0] + 4, a.b[3] + 11, C.orange, { size: 12, bold: true });
        drawBox(g, m, cur, C.purple, { lw: 3.5, fill: 0.1 });
        const ca = cxcywh(a.b), cc = cxcywh(cur); if (t > 0.01) { AI.arrow(g, ca[0], ca[1], cc[0], cc[1], C.fg, 2 / m.s); }
        g.fillStyle = C.orange; g.beginPath(); g.arc(ca[0], ca[1], 4 / m.s, 0, 7); g.fill();
        if (idx === 4) label(g, m, 'decode(anchor, t) = ground truth', cur[0] + 4, cur[1] - 9, C.purple, { size: 12, bold: true });
      }
      g.restore();
      g.restore();
      const e = encode(best.b, GT), a = cxcywh(best.b), gt = cxcywh(GT);
      q('#out').innerHTML = `K = ${A.K} anchors/cell\ngrid ${A.g}×${A.g} → <b>${A.list.length}</b> anchors\n<b class="g">positive ${pos}</b>  <span class="o">ignore ${ign}</span>  <span class="m">negative ${neg}</span>\nbest anchor: scale ${best.s}, ratio ${best.r}\nIoU = <b class="o">${f3(best.iou)}</b>\nanchor (cx,cy,w,h) = ${a.map(Math.round).join(', ')}\nGT     (cx,cy,w,h) = ${gt.map(Math.round).join(', ')}\ntx = (${Math.round(gt[0])}−${Math.round(a[0])})/${Math.round(a[2])} = <b>${f3(e[0])}</b>\nty = (${Math.round(gt[1])}−${Math.round(a[1])})/${Math.round(a[3])} = <b>${f3(e[1])}</b>\ntw = ln(${Math.round(gt[2])}/${Math.round(a[2])}) = <b>${f3(e[2])}</b>\nth = ln(${Math.round(gt[3])}/${Math.round(a[3])}) = <b>${f3(e[3])}</b>`;
      q('#cap').textContent = STEPS[idx] + (idx === 0 ? '  (click a cell to inspect it; drag the green box)' : idx === 2 ? '  (the best anchor is always positive, even if its IoU is below the threshold)' : '');
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  4. YOLO grid
   * ====================================================================================== */
  AI.labs.detyolo = function (root) {
    const N = 480, PW = 290, W = N + PW + 20, H = N;
    const { view, ctl } = AI.labShell(root, 'YOLO: every cell predicts, the centre cell is responsible', 'the image is cut into S × S cells; each cell outputs B boxes and C class probabilities',
      '<canvas id="c" style="aspect-ratio:790/480;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Grid size S <span class="v" id="sv"></span><input type="range" id="S" min="3" max="80" value="7"></label>
       <div>Presets<div class="chips" id="pre" style="margin-top:4px"></div></div>
       <label>Boxes per cell B <span class="v" id="bv"></span><input type="range" id="B" min="1" max="5" value="2"></label>
       <label>Classes C <span class="v" id="cv"></span><input type="range" id="C" min="3" max="80" value="3"></label>
       ${checkRow('hm', 'confidence × class heatmap (illustrative scores of a trained net)', false)}
       <div>Class shown<div class="chips" id="cls" style="margin-top:4px"></div></div>
       <label>Input size (modern anchor-free head)<select id="inp"><option>320</option><option>416</option><option selected>640</option><option>1280</option></select></label>
       ${transport(1)}
       <div class="readout" id="out" style="font-size:11.5px"></div>`);
    wide(root);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c')), img = sceneImg();
    const objs = OBJ.slice(0, 3).map((o) => ({ n: o.n, k: o.k, b: o.b.slice() }));
    let last = '', showCls = -1;
    const P = player(root, { n: 1 + objs.length * 4, dur: (i) => (i === 0 ? 3.2 : 3), onJump: () => (last = '') });
    [['S=7 (YOLOv1)', 7], ['13 (YOLOv2)', 13], ['stride 32: 20', 20], ['stride 16: 40', 40], ['stride 8: 80', 80]].forEach(([t, v]) => chip(q('#pre'), t, () => { q('#S').value = v; last = ''; }));
    [['all', -1], ...objs.map((o, i) => [o.n, i])].forEach(([t, v]) => chip(q('#cls'), t, () => { showCls = v; last = ''; }));
    const getM = () => fitW(c, W, H);
    editor(c.cv, getM, objs.map((o) => ({ b: o.b })), { W: N, H: N, onChange: () => (last = '') });
    const names = (C) => objs.map((o) => o.n).concat(Array.from({ length: Math.max(0, C - 3) }, (_, i) => 'c' + (i + 4)));

    AI.loop(root, (dt) => {
      P.update(dt);
      const S = +q('#S').value, B = +q('#B').value, Cn = +q('#C').value, inp = +q('#inp').value;
      const key = [P.pos.toFixed(3), S, B, Cn, inp, q('#hm').checked, showCls, objs.map((o) => o.b.join()).join(), c.w, document.documentElement.dataset.theme].join();
      q('#sv').textContent = `${S}×${S}`; q('#bv').textContent = B; q('#cv').textContent = Cn;
      if (key === last) return; last = key;
      const C = AI.col(), m = getM(), g = enter(c, m), cs = N / S, idx = P.idx(), fr = P.frac();
      g.drawImage(img, 0, 0, N, N); g.fillStyle = 'rgba(0,0,0,0.18)'; g.fillRect(0, 0, N, N);
      // heatmap
      if (q('#hm').checked) {
        for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
          const cx = (i + 0.5) * cs, cy = (j + 0.5) * cs; let v = 0;
          objs.forEach((o, oi) => { if (showCls >= 0 && showCls !== oi) return; const ow = o.b[2] - o.b[0], oh = o.b[3] - o.b[1], ex = (cx - (o.b[0] + o.b[2]) / 2) / (ow / 2), ey = (cy - (o.b[1] + o.b[3]) / 2) / (oh / 2); v = Math.max(v, Math.exp(-(ex * ex + ey * ey) * 1.3) * 0.95); });
          if (v > 0.03) { g.fillStyle = rgba(C.red, 0.15 + 0.7 * v); g.fillRect(i * cs, j * cs, cs, cs); }
        }
      }
      if (cs >= 4) { for (let i = 0; i <= S; i++) { line(g, m, i * cs, 0, i * cs, N, 'rgba(255,255,255,0.45)', cs < 8 ? 0.5 : 1); line(g, m, 0, i * cs, N, i * cs, 'rgba(255,255,255,0.45)', cs < 8 ? 0.5 : 1); } }
      else { label(g, m, `grid ${S}×${S}: cells too small to draw`, 8, 14, '#fff', { bg: 'rgba(0,0,0,.5)' }); }
      objs.forEach((o, oi) => {
        const cxy = cxcywh(o.b), col = C[o.k] || C.green, ci = Math.floor(cxy[0] / cs), cj = Math.floor(cxy[1] / cs);
        const phase = idx === 0 ? -1 : Math.floor((idx - 1) / 4) === oi ? (idx - 1) % 4 : Math.floor((idx - 1) / 4) > oi ? 9 : -1;
        const dim = idx > 0 && phase < 0 ? 0.35 : 1;
        drawBox(g, m, o.b, rgba(col, dim), { lw: 3 });
        label(g, m, o.n, o.b[0] + 4, o.b[1] - 9, col, { size: 12, bold: true, bg: 'rgba(0,0,0,.45)' });
        if (idx === 0 || phase >= 0) { g.fillStyle = col; g.strokeStyle = '#fff'; g.lineWidth = 2 / m.s; g.beginPath(); g.arc(cxy[0], cxy[1], 6 / m.s, 0, 7); g.fill(); g.stroke(); }
        if (phase >= 1) { g.fillStyle = rgba(col, phase === 1 ? 0.25 + 0.3 * Math.abs(Math.sin(fr * 6)) : 0.45); g.fillRect(ci * cs, cj * cs, cs, cs); g.strokeStyle = col; g.lineWidth = 3 / m.s; g.strokeRect(ci * cs, cj * cs, cs, cs); }
        if (phase === 3) { // neighbours: confidence target 0
          for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { if (i === ci && j === cj) continue; if ((i + 0.5) * cs > o.b[0] && (i + 0.5) * cs < o.b[2] && (j + 0.5) * cs > o.b[1] && (j + 0.5) * cs < o.b[3]) { const x = i * cs, y = j * cs, p = cs * 0.25; line(g, m, x + p, y + p, x + cs - p, y + cs - p, C.red, 1.6); line(g, m, x + cs - p, y + p, x + p, y + cs - p, C.red, 1.6); } }
        }
      });
      // right panel: the output vector of the responsible cell
      const px = N + 20; g.fillStyle = C.bg2; g.fillRect(px - 10, 0, PW + 10, H);
      label(g, m, `output tensor ${S} × ${S} × ${B * 5 + Cn}`, px, 16, C.fg, { size: 13, bold: true });
      label(g, m, `= ${S * S * (B * 5 + Cn)} numbers  (B·5 + C = ${B}·5 + ${Cn})`, px, 34, C.muted, { size: 11 });
      const oi = idx === 0 ? 0 : Math.floor((idx - 1) / 4), ph = idx === 0 ? 0 : (idx - 1) % 4, o = objs[oi], cxy = cxcywh(o.b), ci = Math.floor(cxy[0] / cs), cj = Math.floor(cxy[1] / cs), col = C[o.k] || C.green;
      label(g, m, `cell (row ${cj}, col ${ci}) vector — ${B * 5 + Cn} values`, px, 62, col, { size: 12, bold: true });
      const vec = [], cl = names(Cn); const ox = cxy[0] / cs - ci, oy = cxy[1] / cs - cj;
      for (let b = 0; b < B; b++) { vec.push(['x', b === 0 ? ox : 0], ['y', b === 0 ? oy : 0], ['w', b === 0 ? (o.b[2] - o.b[0]) / N : 0], ['h', b === 0 ? (o.b[3] - o.b[1]) / N : 0], ['conf', b === 0 ? 1 : 0]); }
      const show = Math.min(vec.length, 10); let y = 80; const rows = Math.min(14, B * 5 + Cn);
      const cellW = 40, cellH = 17, cols = 5;
      const items = vec.map((v, i) => [`${v[0]}${Math.floor(i / 5) + 1}`, v[1], Math.floor(i / 5) === 0 ? col : C.muted]).concat(cl.slice(0, Cn).map((n, i) => [`p(${n.slice(0, 4)})`, i === oi ? 1 : 0, C.teal]));
      const reveal = idx === 0 ? 0 : phase3(idx, oi, ph, fr, items.length);
      items.slice(0, 40).forEach((it, t) => {
        const xx = px + (t % cols) * (cellW + 8), yy = y + Math.floor(t / cols) * (cellH + 6); if (yy > H - 60) return;
        const on = t < reveal; g.fillStyle = on ? rgba(it[2], 0.28) : C.bg; g.fillRect(xx, yy, cellW + 4, cellH); g.strokeStyle = on ? it[2] : C.border; g.lineWidth = 1 / m.s; g.strokeRect(xx, yy, cellW + 4, cellH);
        label(g, m, on ? (Math.abs(it[1]) < 1e-9 ? '0' : f2(it[1])) : it[0], xx + (cellW + 4) / 2, yy + cellH / 2, on ? C.fg : C.muted, { size: 10.5, align: 'center' });
      });
      if (items.length > 40) label(g, m, `… ${items.length - 40} more class slots`, px, H - 66, C.muted, { size: 11 });
      label(g, m, 'box 1 (responsible): x,y = offset in cell;', px, H - 46, C.muted, { size: 10.5 });
      label(g, m, 'w,h = size/image; conf target ≈ 1.', px, H - 32, C.muted, { size: 10.5 });
      label(g, m, 'other boxes and cells: conf target 0.', px, H - 18, C.muted, { size: 10.5 });
      g.restore();
      // readout: modern head sizes
      const lv = [8, 16, 32].map((s) => ({ s, g: inp / s })), tot = lv.reduce((a, l) => a + l.g * l.g, 0);
      q('#out').innerHTML = `<b>classic</b> YOLOv1 S=7,B=2,C=20 → 7·7·30 = 1470\nthis lab: ${S}·${S}·${B * 5 + Cn} = <b class="o">${S * S * (B * 5 + Cn)}</b>\n<b>modern anchor-free</b>, input ${inp}×${inp}\n${lv.map((l) => `stride ${String(l.s).padStart(2)}: ${l.g}×${l.g} = ${String(l.g * l.g).padStart(5)} cells`).join('\n')}\n<b class="o">${tot} predictions</b>, each 4 box + ${Cn} cls\nper level: (4·16 + ${Cn}) = ${64 + Cn} channels`;
      const capIdx = idx === 0 ? 'Drag the three objects. Each has a centre (dot); the grid is laid over the image.' : [`the object and its centre`, `the cell that contains the centre is RESPONSIBLE for this object`, `that cell's output vector is filled: box offset, size, confidence 1, class one-hot`, `neighbouring cells also lie inside the box, but their confidence target is 0 (red ×)`][ph];
      q('#cap').textContent = (idx === 0 ? '' : `Object ${oi + 1}/${objs.length} — ${o.n}: `) + capIdx;
    }, { fps: 30 });
    function phase3(idx, oi, ph, fr, total) { return ph < 2 ? 0 : ph === 2 ? Math.ceil(smooth(Math.min(1, fr * 1.25)) * total) : total; }
  };

  /* ======================================================================================
   *  5. NMS
   * ====================================================================================== */
  AI.labs.detnms = function (root) {
    const W = 640, H = 400;
    const { view, ctl } = AI.labShell(root, 'Non-maximum suppression, step by step', 'keep the best box, suppress what overlaps it too much, repeat',
      '<canvas id="c" style="aspect-ratio:640/400;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<div>Scenario<div class="chips" id="pre" style="margin-top:4px"></div></div>
       <label>IoU threshold <span class="v" id="tv"></span><input type="range" id="thr" min="0.1" max="0.9" step="0.05" value="0.5"></label>
       ${checkRow('soft', 'soft-NMS (Gaussian): decay scores instead of deleting', false)}
       <label>soft-NMS σ <span class="v" id="sgv"></span><input type="range" id="sg" min="0.1" max="1" step="0.05" value="0.5"></label>
       ${checkRow('cw', 'class-wise (only compare boxes of the same class)', true)}
       ${transport(1)}
       <div>Boxes (edit scores; drag boxes on the canvas)<div id="tbl" style="margin-top:4px;font-size:12px"></div></div>
       <div class="readout" id="out" style="font-size:11.5px"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c'));
    const SC = {
      'worked example (5 boxes)': [['A', [100, 100, 300, 300], 0.95, 0], ['B', [120, 110, 310, 310], 0.9, 0], ['C', [90, 120, 290, 290], 0.7, 0], ['D', [360, 150, 500, 320], 0.8, 0], ['E', [420, 170, 560, 330], 0.6, 0]],
      'crowd, two classes (9 boxes)': [['A', [60, 60, 230, 240], 0.92, 0], ['B', [75, 70, 250, 250], 0.88, 0], ['C', [50, 80, 220, 230], 0.74, 0], ['D', [280, 150, 440, 330], 0.85, 0], ['E', [295, 160, 455, 345], 0.8, 0], ['F', [330, 170, 500, 340], 0.55, 0], ['G', [110, 150, 300, 300], 0.65, 1], ['H', [120, 160, 310, 310], 0.6, 1], ['I', [480, 40, 600, 120], 0.35, 1]],
    };
    let boxes = [], plan = null, last = '';
    const P = player(root, { n: 1, dur: () => 1.5, onJump: () => (last = '') });
    const COLS = ['blue', 'orange'];
    const load = (name) => { boxes = SC[name].map(([id, b, s, cl]) => ({ id, b: b.slice(), s, cl })); build(); P.restart(); buildTable(); };
    Object.keys(SC).forEach((n) => chip(q('#pre'), n, () => load(n)));
    function build() {
      const thr = +q('#thr').value, soft = q('#soft').checked, sg = +q('#sg').value, cw = q('#cw').checked, n = boxes.length;
      const sc = boxes.map((b) => b.s), st = boxes.map(() => 'pending'), ev = [], snaps = [];
      const snap = (e) => { ev.push(e); snaps.push({ st: st.slice(), sc: sc.slice() }); };
      snaps.push({ st: st.slice(), sc: sc.slice() }); ev.push({ t: 'start' });
      const alive = () => boxes.map((_, i) => i).filter((i) => st[i] === 'pending' && sc[i] >= (soft ? 0.05 : 0));
      for (;;) {
        const al = alive(); if (!al.length) break;
        al.sort((a, b) => sc[b] - sc[a]); const i = al[0]; st[i] = 'kept'; snap({ t: 'pick', i, s: sc[i] });
        al.slice(1).forEach((j) => {
          if (cw && boxes[i].cl !== boxes[j].cl) return;
          const v = iouOf(boxes[i].b, boxes[j].b);
          if (soft) { const old = sc[j]; sc[j] = old * Math.exp(-(v * v) / sg); snap({ t: 'cmp', i, j, iou: v, old, nw: sc[j], act: 'decay' }); if (sc[j] < 0.05) st[j] = 'suppressed'; }
          else if (v > thr) { st[j] = 'suppressed'; snap({ t: 'cmp', i, j, iou: v, act: 'suppress' }); } else snap({ t: 'cmp', i, j, iou: v, act: 'keep' });
        });
      }
      snap({ t: 'done' });
      plan = { ev, snaps }; P.n = ev.length;
    }
    function buildTable() {
      const t = q('#tbl'); t.innerHTML = '';
      boxes.forEach((b, i) => {
        const r = document.createElement('div'); r.style.cssText = 'display:flex;gap:6px;align-items:center;margin-bottom:3px';
        r.innerHTML = `<b style="width:16px;color:var(--${COLS[b.cl]})">${b.id}</b><input type="number" step="0.05" min="0" max="1" value="${b.s}" style="width:70px;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:3px 6px;color:var(--fg)"><span class="st" style="color:var(--muted)"></span>`;
        r.querySelector('input').oninput = (e) => { const v = +e.target.value; if (Number.isFinite(v)) { b.s = clamp(v, 0, 1); build(); P.restart(); } };
        t.appendChild(r);
      });
    }
    ['#thr', '#soft', '#sg', '#cw'].forEach((s) => q(s).addEventListener('input', () => { build(); P.restart(); }));
    const getM = () => fitW(c, W, H);
    load('worked example (5 boxes)');
    const items = new Proxy([], { get: (t, p) => (p === 'length' ? boxes.length : p === Symbol.iterator ? undefined : typeof p === 'string' && /^\d+$/.test(p) ? { b: boxes[+p].b } : t[p]) });
    editor(c.cv, getM, items, { W, H, onChange: () => (last = ''), onEnd: () => { build(); P.restart(); } });

    AI.loop(root, (dt) => {
      P.update(dt);
      q('#tv').textContent = (+q('#thr').value).toFixed(2); q('#sgv').textContent = (+q('#sg').value).toFixed(2);
      q('#thr').disabled = q('#soft').checked; q('#sg').disabled = !q('#soft').checked;
      const idx = P.idx(), fr = P.frac(), key = [P.pos.toFixed(3), boxes.map((b) => b.b.join()).join('|'), c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = getM(), g = enter(c, m), e = plan.ev[idx], sn = plan.snaps[Math.min(idx, plan.snaps.length - 1)], prev = plan.snaps[Math.max(0, idx - 1)];
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H); for (let x = 0; x <= W; x += 80) line(g, m, x, 0, x, H, C.border, 1); for (let y = 0; y <= H; y += 80) line(g, m, 0, y, W, y, C.border, 1);
      const state = (i) => sn.st[i], scr = (i) => sn.sc[i];
      boxes.forEach((b, i) => {
        const s = state(i), base = C[COLS[b.cl]];
        if (s === 'suppressed') { drawBox(g, m, b.b, rgba(C.red, 0.45), { lw: 1.5, dash: [4, 4] }); }
        else if (s === 'kept') drawBox(g, m, b.b, C.green, { lw: 3.5, fill: 0.1 });
        else drawBox(g, m, b.b, base, { lw: 2, fill: 0.04 });
      });
      boxes.forEach((b, i) => { const s = state(i); label(g, m, `${b.id} ${f2(scr(i))}`, b.b[0] + 4, b.b[1] + 12, s === 'suppressed' ? C.red : s === 'kept' ? C.green : C.fg, { size: 12, bold: true, bg: rgba(C.bg, 0.75) }); });
      if (e && e.t === 'cmp') {
        const a = boxes[e.i].b, b = boxes[e.j].b, ca = cxcywh(a), cb = cxcywh(b), t = smooth(Math.min(1, fr * 1.4));
        drawBox(g, m, boxes[e.i].b, C.green, { lw: 4 }); drawBox(g, m, boxes[e.j].b, C.orange, { lw: 4 });
        const I = interBox(a, b); if (I[2] > I[0] && I[3] > I[1]) drawBox(g, m, I, C.purple, { fill: 0.45 * t, lw: 1.5 });
        line(g, m, ca[0], ca[1], lerp(ca[0], cb[0], t), lerp(ca[1], cb[1], t), C.fg, 2);
        if (t > 0.7) label(g, m, `IoU(${boxes[e.i].id},${boxes[e.j].id}) = ${f2(e.iou)}`, (ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2 - 10, C.fg, { size: 13, bold: true, align: 'center', bg: rgba(C.bg, 0.85) });
      }
      g.restore();
      // table status + readout
      AI.$$('#tbl > div', q('#tbl').parentNode).forEach((r, i) => { const s = state(i); r.querySelector('.st').innerHTML = s === 'kept' ? '<b style="color:var(--green)">kept</b>' : s === 'suppressed' ? '<b style="color:var(--red)">suppressed</b>' : '<span>pending</span>'; if (document.activeElement !== r.querySelector('input')) r.querySelector('input').value = +f3(scr(i)).replace(/0+$/, '').replace(/\.$/, '') || 0; });
      const thr = +q('#thr').value, soft = q('#soft').checked;
      const kept = boxes.filter((_, i) => state(i) === 'kept').map((b) => b.id).join(', ') || '–';
      q('#out').innerHTML = `step ${idx + 1} / ${plan.ev.length}\nthreshold ${soft ? '(soft-NMS: σ=' + (+q('#sg').value).toFixed(2) + ')' : thr.toFixed(2)}\nkept so far: <b class="g">${kept}</b>`;
      q('#cap').textContent = !e || e.t === 'start' ? 'Start: candidate boxes with their scores. Greedy NMS repeatedly takes the highest-scoring box that is still pending.'
        : e.t === 'pick' ? `Pick the highest remaining score: box ${boxes[e.i].id} (${f2(e.s)}). It is kept. Now compare it with every other pending box.`
          : e.t === 'cmp' ? (e.act === 'suppress' ? `IoU ${f2(e.iou)} > ${thr.toFixed(2)}: box ${boxes[e.j].id} is a duplicate of ${boxes[e.i].id} → suppressed (deleted).` : e.act === 'keep' ? `IoU ${f2(e.iou)} ≤ ${thr.toFixed(2)}: box ${boxes[e.j].id} is different enough → stays in the pool.` : `Soft-NMS: score of ${boxes[e.j].id} decays ${f2(e.old)} × exp(−${f2(e.iou)}²/σ) = ${f2(e.nw)}. It stays, but ranks lower.`)
            : 'Done. Everything left is either kept or suppressed. Try raising the threshold (more survivors, more duplicates) or lowering it (fewer duplicates, but real neighbours get killed).';
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  6. DETR: queries + Hungarian matching
   * ====================================================================================== */
  AI.labs.detdetr = function (root) {
    const IW = 300, H = 300, W = 780;
    const { view, ctl } = AI.labShell(root, 'DETR: object queries and Hungarian matching', 'each query predicts one box; training needs a one-to-one assignment to the ground-truth objects',
      `<canvas id="c" style="aspect-ratio:780/300"></canvas><div class="caption" id="cap"></div>
       <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;font-size:12px"><div><b>Cost matrix (edit the numbers)</b><div id="mx" style="margin-top:4px"></div></div><div><b>Working matrix</b><div id="wk" style="margin-top:4px"></div></div></div>`,
      `<label>Queries N (rows) <select id="n"><option value="3">3 queries</option><option value="4">4 queries</option></select></label>
       <div class="row"><button class="btn" id="def">Reset costs</button></div>
       ${transport(1)}
       <div class="readout" id="out" style="font-size:11.5px"></div>`);
    wide(root);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c')), mcols = 3;
    const DEF = [[4, 1, 3], [2, 0, 5], [3, 2, 2], [5, 4, 6]];
    let M = DEF.map((r) => r.slice()), last = '', n = 3, plan = null;
    const GT = [[20, 30, 120, 130], [150, 60, 280, 170], [60, 190, 200, 280]];
    const QB = [[158, 72, 276, 164], [90, 50, 210, 150], [66, 196, 196, 278], [200, 200, 280, 280]];
    const QC = ['blue', 'orange', 'purple', 'pink'];
    const P = player(root, { n: 5, dur: (i) => (i >= 4 && i < plan.P + 4 ? 0.75 : 3.4), onJump: () => (last = '') });
    function build() {
      n = +q('#n').value; const rows = M.slice(0, n);
      const rm = rows.map((r) => Math.min(...r)), R1 = rows.map((r, i) => r.map((v) => v - rm[i])), cmn = [0, 1, 2].map((j) => Math.min(...R1.map((r) => r[j]))), R2 = R1.map((r) => r.map((v, j) => v - cmn[j]));
      const cands = []; const rec = (j, used, asg) => { if (j === mcols) { cands.push(asg.slice()); return; } for (let i = 0; i < n; i++) if (!used.has(i)) { used.add(i); asg.push(i); rec(j + 1, used, asg); asg.pop(); used.delete(i); } };
      rec(0, new Set(), []);
      const cost = (a) => a.reduce((s, i, j) => s + rows[i][j], 0); let bi = 0; cands.forEach((a, i) => { if (cost(a) < cost(cands[bi])) bi = i; });
      plan = { rows, rm, R1, cmn, R2, cands, cost, best: bi, P: cands.length }; P.n = 5 + cands.length - 1 + 1;
    }
    function buildMatrix() {
      const mx = q('#mx'); mx.innerHTML = ''; const tb = document.createElement('table'); tb.className = 'tbl'; tb.style.cssText = 'width:auto;font-size:12px';
      let h = '<tr><th></th>' + [1, 2, 3].map((j) => `<th>GT${j}</th>`).join('') + '</tr>';
      for (let i = 0; i < n; i++) h += `<tr><th>Q${i + 1}</th>${[0, 1, 2].map((j) => `<td><input type="number" step="0.5" data-i="${i}" data-j="${j}" value="${M[i][j]}" style="width:52px;text-align:center;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:2px;color:var(--fg)"></td>`).join('')}</tr>`;
      tb.innerHTML = h; mx.appendChild(tb);
      tb.querySelectorAll('input').forEach((e) => (e.oninput = () => { const v = +e.value; if (Number.isFinite(v)) { M[+e.dataset.i][+e.dataset.j] = v; build(); P.restart(); } }));
    }
    q('#n').addEventListener('input', () => { build(); buildMatrix(); P.restart(); });
    q('#def').onclick = () => { M = DEF.map((r) => r.slice()); build(); buildMatrix(); P.restart(); };
    build(); buildMatrix();

    AI.loop(root, (dt) => {
      P.update(dt);
      const idx = P.idx(), fr = P.frac(), key = [P.pos.toFixed(3), c.w, n, M.join(), document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = fitW(c, W, H), g = enter(c, m), pl = plan, phase = idx < 4 ? idx : idx < 4 + pl.P ? 4 : 5, ci = idx - 4;
      // image
      g.fillStyle = C.bg2; g.fillRect(0, 0, IW, H); g.strokeStyle = C.border; g.lineWidth = 1 / m.s; g.strokeRect(0, 0, IW, H);
      GT.forEach((b, j) => { drawBox(g, m, b, C.green, { lw: 2, dash: [6, 4] }); label(g, m, 'GT' + (j + 1), b[0] + 3, b[1] + 11, C.green, { size: 11, bold: true }); });
      const cur = phase === 4 ? pl.cands[ci] : phase === 5 ? pl.cands[pl.best] : null;
      for (let i = 0; i < n; i++) { const matched = cur && cur.indexOf(i) >= 0; drawBox(g, m, QB[i], C[QC[i]], { lw: matched ? 3.5 : 1.8, fill: matched ? 0.15 : 0.04 }); label(g, m, 'Q' + (i + 1), QB[i][2] - 4, QB[i][3] - 9, C[QC[i]], { size: 11, bold: true, align: 'right' }); }
      // bipartite graph
      const qx = 440, gx = 700, yq = (i) => 50 + i * (n === 4 ? 66 : 88), yg = (j) => 50 + j * 100 + (n === 4 ? 33 : 38);
      label(g, m, 'queries (N)', qx, 18, C.fg, { align: 'center', size: 12, bold: true }); label(g, m, 'ground truth (padded with ∅)', gx - 20, 18, C.fg, { align: 'center', size: 12, bold: true });
      for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) { const on = cur && cur[j] === i; line(g, m, qx + 16, yq(i), gx - 16, yg(j), on ? C.green : C.border, on ? 4 : 1); if (phase >= 1) label(g, m, String(pl.rows[i][j]), lerp(qx + 16, gx - 16, 0.2 + 0.13 * ((i + j) % 4)), lerp(yq(i), yg(j), 0.2 + 0.13 * ((i + j) % 4)), on ? C.green : C.muted, { size: 10.5, align: 'center', bg: rgba(C.bg2, 0.9) }); }
      for (let i = 0; i < n; i++) { const matched = cur && cur.indexOf(i) >= 0; g.fillStyle = rgba(C[QC[i]], matched || !cur ? 0.9 : 0.35); g.beginPath(); g.arc(qx, yq(i), 15, 0, 7); g.fill(); label(g, m, 'Q' + (i + 1), qx, yq(i), '#fff', { align: 'center', size: 11, bold: true }); if (phase === 5 && !matched) label(g, m, '→ ∅ (no object)', qx - 24, yq(i), C.muted, { align: 'right', size: 11 }); }
      for (let j = 0; j < 3; j++) { g.fillStyle = C.green; g.beginPath(); g.arc(gx, yg(j), 15, 0, 7); g.fill(); label(g, m, 'GT' + (j + 1), gx, yg(j), '#fff', { align: 'center', size: 11, bold: true }); }
      g.restore();
      // working matrix
      const useR = phase >= 3 ? pl.R2 : phase === 2 ? pl.R1 : pl.rows, hl = new Set(); if (cur) cur.forEach((i, j) => hl.add(i + ',' + j));
      let h = '<table class="tbl" style="width:auto;font-size:12px"><tr><th></th>' + [1, 2, 3].map((j) => `<th>GT${j}</th>`).join('') + '</tr>';
      for (let i = 0; i < n; i++) { h += `<tr><th>Q${i + 1}</th>`; for (let j = 0; j < 3; j++) { const v = useR[i][j], on = hl.has(i + ',' + j), z = phase >= 2 && v === 0; h += `<td style="text-align:center;width:52px;${on ? 'background:var(--green);color:#fff;font-weight:700;' : z ? 'color:var(--green);font-weight:700;' : ''}">${v}</td>`; } h += '</tr>'; }
      q('#wk').innerHTML = h + '</table>';
      // readout + caption
      const bestCost = pl.cost(pl.cands[pl.best]);
      let bsf = Infinity; if (phase >= 4) for (let k = 0; k <= Math.min(ci, pl.P - 1); k++) bsf = Math.min(bsf, pl.cost(pl.cands[k]));
      if (phase === 5) bsf = bestCost;
      q('#out').innerHTML = phase < 4 ? `row minima ${pl.rm.join(', ')}\ncolumn minima (after) ${pl.cmn.join(', ')}` : `candidate ${phase === 5 ? '–' : ci + 1} / ${pl.P}\nthis assignment: ${phase === 5 ? '–' : '<b>' + pl.cost(cur) + '</b>'}\nbest so far: <b class="g">${bsf}</b>\n${phase === 5 ? '<b class="o">optimal total cost = ' + bestCost + '</b>\n' + cur.map((i, j) => `GT${j + 1} ← Q${i + 1}`).join('\n') : ''}`;
      q('#cap').textContent = ['① N = ' + n + ' queries each predict a box; the image has 3 real objects (green). We need ONE query per object — no two queries may claim the same object.',
        '② Cost matrix: cost(Qi, GTj) = −class probability + box L1 distance + (1 − GIoU). Low cost = good match. (Numbers on the lines; edit the matrix below.)',
        '③ Hungarian step 1: subtract each row\'s minimum — every row now has a zero.',
        '④ Hungarian step 2: subtract each column\'s minimum — zeros mark the cheap candidate pairs. (If the zeros cannot form a full matching, the algorithm adjusts the matrix further; here we just verify by checking every assignment.)',
        `⑤ Checking assignment ${ci + 1} of ${pl.P}: total = sum of the 3 chosen costs. n!/(n−3)! assignments grow explosively — the real algorithm needs only O(N³).`,
        '⑥ The optimal assignment. Matched queries receive class + box loss; every unmatched query is trained to say ∅ (no object).'][phase];
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  7. PR curve / AP
   * ====================================================================================== */
  AI.labs.detpr = function (root) {
    const { view, ctl } = AI.labShell(root, 'Precision–recall curve and Average Precision', 'detections are added one at a time, from the most confident to the least',
      `<canvas id="c" style="height:300px"></canvas><div class="caption" id="cap"></div><div id="tbl" style="margin-top:8px;overflow-x:auto"></div>`,
      `<label>IoU threshold for a true positive <span class="v" id="tv"></span><input type="range" id="thr" min="0.5" max="0.95" step="0.05" value="0.5"></label>
       <label>Ground-truth objects in the dataset <input type="number" id="ngt" min="1" max="20" value="5"></label>
       <label>AP interpolation <select id="am"><option value="all">all-point (area under the envelope)</option><option value="11">11-point (PASCAL VOC 2007)</option><option value="101">101-point (COCO)</option></select></label>
       ${transport(1)}
       <div class="readout" id="out" style="font-size:11.5px"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c'));
    let D = [[0.95, 1, 0.82], [0.9, 2, 0.74], [0.85, 1, 0.65], [0.8, 3, 0.58], [0.7, 0, 0.2], [0.65, 4, 0.44], [0.55, 4, 0.62], [0.4, 0, 0.1]].map(([s, g, i]) => ({ s, g, i }));
    let last = '', res = null;
    const P = player(root, { n: D.length + 1, dur: () => 1.5, onJump: () => (last = '') });
    function evalAt(thr, dets, ngt, method) {
      const used = new Set(), flags = [], pr = []; let tp = 0, fp = 0;
      dets.forEach((d) => { const ok = d.g > 0 && d.i >= thr - 1e-9 && !used.has(d.g); if (ok) { used.add(d.g); tp++; } else fp++; flags.push(ok); pr.push([tp / ngt, tp / (tp + fp), tp, fp]); });
      const env = pr.map((_, k) => Math.max(0, ...pr.slice(k).map((p) => p[1])));
      let ap = 0;
      if (method === 'all') { let r0 = 0; pr.forEach((p, k) => { ap += (p[0] - r0) * env[k]; r0 = p[0]; }); }
      else { const K = method === '11' ? 11 : 101; for (let t = 0; t < K; t++) { const r = t / (K - 1); let mx = 0; pr.forEach((p) => { if (p[0] >= r - 1e-9) mx = Math.max(mx, p[1]); }); ap += mx; } ap /= K; }
      return { flags, pr, env, ap };
    }
    function table() {
      const t = q('#tbl'); t.innerHTML = '<table class="tbl" style="width:100%;font-size:12px"><thead><tr><th>rank</th><th>score</th><th>matches GT #</th><th>IoU with it</th><th>verdict</th><th>TP</th><th>FP</th><th>precision</th><th>recall</th></tr></thead><tbody></tbody></table>';
      const tb = t.querySelector('tbody');
      D.forEach((d, k) => {
        const r = document.createElement('tr'); r.innerHTML = `<td>${k + 1}</td><td><input type="number" step="0.05" min="0" max="1" value="${d.s}" data-f="s"></td><td><input type="number" step="1" min="0" max="20" value="${d.g}" data-f="g"></td><td><input type="number" step="0.02" min="0" max="1" value="${d.i}" data-f="i"></td><td class="v"></td><td class="tp"></td><td class="fp"></td><td class="p"></td><td class="r"></td>`;
        r.querySelectorAll('input').forEach((e) => { e.style.cssText = 'width:58px;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:2px 4px;color:var(--fg)'; e.oninput = () => { const v = +e.value; if (Number.isFinite(v)) { d[e.dataset.f] = v; last = ''; } }; e.onchange = () => { D.sort((a, b) => b.s - a.s); table(); last = ''; }; });
        tb.appendChild(r);
      });
    }
    ['#thr', '#ngt', '#am'].forEach((s) => q(s).addEventListener('input', () => (last = '')));
    table();

    AI.loop(root, (dt) => {
      P.update(dt);
      const thr = +q('#thr').value, ngt = Math.max(1, +q('#ngt').value || 1), method = q('#am').value;
      q('#tv').textContent = thr.toFixed(2);
      const idx = P.idx(), fr = P.frac(), shown = Math.min(D.length, idx + (fr > 0.5 || idx >= D.length ? 1 : 0)), done = P.pos >= D.length - 0.001 || idx >= D.length;
      const key = [P.pos.toFixed(3), thr, ngt, method, D.map((d) => d.s + ',' + d.g + ',' + d.i).join(';'), c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(); res = evalAt(thr, D, ngt, method);
      const upto = P.pos >= D.length ? D.length : Math.min(D.length, Math.floor(P.pos) + 1);
      const g = c.ctx, v = AI.view(c, 0, 1, 0, 1.05, { l: 44, r: 14, t: 12, b: 32 }); g.clearRect(0, 0, c.w, c.h);
      AI.axes(g, v);
      g.fillStyle = C.muted; g.font = `11px ${MONO}`; g.textAlign = 'center'; g.fillText('recall', v.X(0.5), c.h - 4); g.save(); g.translate(11, v.Y(0.5)); g.rotate(-Math.PI / 2); g.fillText('precision', 0, 0); g.restore();
      const pts = res.pr.slice(0, upto);
      // envelope (step) and AP area
      if (pts.length) {
        g.beginPath(); g.moveTo(v.X(0), v.Y(0)); let r0 = 0;
        pts.forEach((p, k) => { g.lineTo(v.X(r0), v.Y(res.env[k])); g.lineTo(v.X(p[0]), v.Y(res.env[k])); r0 = p[0]; });
        g.lineTo(v.X(r0), v.Y(0)); g.closePath(); g.fillStyle = rgba(C.green, done ? 0.3 : 0.15); g.fill();
        g.beginPath(); r0 = 0; pts.forEach((p, k) => { if (k === 0) g.moveTo(v.X(0), v.Y(res.env[0])); g.lineTo(v.X(r0), v.Y(res.env[k])); g.lineTo(v.X(p[0]), v.Y(res.env[k])); r0 = p[0]; }); g.strokeStyle = C.green; g.lineWidth = 2.5; g.stroke();
        g.beginPath(); pts.forEach((p, k) => (k ? g.lineTo(v.X(p[0]), v.Y(p[1])) : g.moveTo(v.X(p[0]), v.Y(p[1])))); g.strokeStyle = rgba(C.blue, 0.8); g.lineWidth = 1.5; g.stroke();
        pts.forEach((p, k) => { const tpFlag = res.flags[k], newest = k === upto - 1 && !done; g.fillStyle = tpFlag ? C.green : C.red; g.beginPath(); g.arc(v.X(p[0]), v.Y(p[1]), newest ? 7 : 4.5, 0, 7); g.fill(); g.strokeStyle = C.bg; g.lineWidth = 1.5; g.stroke(); g.fillStyle = C.fg; g.font = `10px ${MONO}`; g.textAlign = 'left'; g.fillText(String(k + 1), v.X(p[0]) + 7, v.Y(p[1]) - 7); });
      }
      g.font = `11px ${MONO}`; g.textAlign = 'right'; g.fillStyle = C.blue; g.fillText('● raw precision at each rank', v.X(1) - 4, v.Y(1.0) + 12); g.fillStyle = C.green; g.fillText('envelope = max precision at recall ≥ r (area = AP)', v.X(1) - 4, v.Y(1.0) + 26);
      // table
      AI.$$('tbody tr', q('#tbl')).forEach((r, k) => {
        const p = res.pr[k], on = k < upto, cur = k === upto - 1 && !done;
        r.style.background = cur ? rgba(C.blue, 0.2) : ''; r.style.opacity = on ? 1 : 0.4;
        r.querySelector('.v').innerHTML = on ? (res.flags[k] ? '<b style="color:var(--green)">TP</b>' : `<b style="color:var(--red)">FP</b>${D[k].g > 0 && D[k].i < thr ? ' <span style="color:var(--muted)">(IoU&lt;thr)</span>' : D[k].g > 0 ? ' <span style="color:var(--muted)">(duplicate)</span>' : ''}`) : '';
        r.querySelector('.tp').textContent = on ? p[2] : ''; r.querySelector('.fp').textContent = on ? p[3] : '';
        r.querySelector('.p').textContent = on ? `${p[2]}/${p[2] + p[3]} = ${f2(p[1])}` : ''; r.querySelector('.r').textContent = on ? `${p[2]}/${ngt} = ${f2(p[0])}` : '';
      });
      // mAP@[.5:.95]
      let mAP = 0; for (let t = 0; t < 10; t++) mAP += evalAt(0.5 + 0.05 * t, D, ngt, method).ap; mAP /= 10;
      const ap50 = evalAt(0.5, D, ngt, method).ap, ap75 = evalAt(0.75, D, ngt, method).ap;
      q('#out').innerHTML = `detections shown: ${upto} / ${D.length}\nAP @ IoU ${thr.toFixed(2)} = <b class="o">${done ? f3(res.ap) : '… (finish the sweep)'}</b>\nAP50 = ${f3(ap50)}   AP75 = ${f3(ap75)}\n<b class="g">mAP@[.5:.95] = ${f3(mAP)}</b>\n<span class="m">(mean of AP at IoU 0.50, 0.55 … 0.95;\none class here — mAP would also average classes)</span>`;
      const k = upto - 1, p = res.pr[k];
      q('#cap').textContent = done ? `All ${D.length} detections processed. AP = area under the green envelope = ${f3(res.ap)}. Raise the IoU threshold and watch true positives turn into false positives as boxes stop being accurate enough.`
        : `Detection ${k + 1} (score ${f2(D[k].s)}): ${res.flags[k] ? `IoU ${f2(D[k].i)} ≥ ${thr.toFixed(2)} and GT #${D[k].g} not yet matched → true positive` : D[k].g === 0 ? 'overlaps no object → false positive' : D[k].i < thr ? `IoU ${f2(D[k].i)} < ${thr.toFixed(2)} → false positive (box too loose)` : `GT #${D[k].g} was already claimed by a better-scoring detection → false positive (duplicate)`}. Now precision = ${p[2]}/${p[2] + p[3]} = ${f2(p[1])}, recall = ${p[2]}/${ngt} = ${f2(p[0])}.`;
    }, { fps: 30 });
  };

  /* ======================================================================================
   *  8. architecture blueprints (animated)
   * ====================================================================================== */
  const BP = {
    frcnn: {
      title: 'Faster R-CNN (ResNet-50-FPN), 640×640 input', W: 1090, H: 400,
      blocks: {
        img: { c: 0, r: 1, t: 'Image', s: '3×640×640', col: 'muted', note: 'Input image (torchvision resizes the short side to 800; 640 keeps the numbers round). Normalised RGB, one tensor.' },
        bb: { c: 1, r: 1, t: 'Backbone', s: 'ResNet-50 C2…C5', col: 'blue', note: 'Four stages give C2 256×160×160 (stride 4), C3 512×80×80, C4 1024×40×40, C5 2048×20×20. Backbone without its classifier: 23.5 M parameters.' },
        fpn: { c: 2, r: 1, t: 'FPN', s: 'P2…P6, 256 ch', col: 'teal', note: '1×1 lateral convs to 256 channels, top-down addition, 3×3 smoothing: P2 160², P3 80², P4 40², P5 20², P6 10² (all 256 channels). ≈3.3 M parameters.' },
        rpn: { c: 3, r: 0, t: 'RPN head', s: '3×3 conv + 1×1s', col: 'orange', note: 'One small head shared by all levels: 3×3 conv 256→256 (590,080), objectness 256→3 (771), deltas 256→12 (3,084) = 593,935 parameters. 3 anchors per cell → (160²+80²+40²+20²+10²)·3 = 102,300 anchors.' },
        prop: { c: 4, r: 0, t: 'Proposals', s: '1000 × 4', col: 'purple', note: 'Decode the deltas, clip to the image, keep the best by objectness per level, NMS at IoU 0.7 → about 1000 proposal boxes.' },
        roi: { c: 4, r: 2, t: 'RoIAlign', s: '1000×256×7×7', col: 'pink', note: 'Each proposal is assigned a pyramid level from its size and its feature window is bilinearly sampled onto a fixed 7×7 grid — no rounding, so no misalignment.' },
        head: { c: 5, r: 2, t: 'Box head', s: 'FC 12544→1024→1024', col: 'green', note: 'Flatten 256·7·7 = 12,544, FC to 1024 (12,846,080 params), FC to 1024 (1,049,600). The two FC layers hold most of the head\'s parameters.' },
        out: { c: 6, r: 2, t: 'Class + box', s: '81 cls · 324 deltas', col: 'red', note: 'Classifier FC 1024→81 (83,025) and box regressor FC 1024→324 (332,100: one (tx,ty,tw,th) per class). Then score filter, per-class NMS 0.5, top 100. Whole model ≈ 41.8 M parameters.' },
      },
      edges: [['img', 'bb', 'rr'], ['bb', 'fpn', 'rr'], ['fpn', 'rpn', 'rr'], ['rpn', 'prop', 'rr'], ['prop', 'roi', 'bt'], ['fpn', 'roi', 'bl'], ['roi', 'head', 'rr'], ['head', 'out', 'rr']],
      path: ['img', 'bb', 'fpn', 'rpn', 'prop', 'roi', 'head', 'out'],
    },
    yolov8: {
      title: 'YOLOv8-s style anchor-free detector, 640×640 input', W: 1090, H: 400,
      blocks: {
        img: { c: 0, r: 1, t: 'Image', s: '3×640×640', col: 'muted', note: 'Letterboxed 640×640 input.' },
        bb: { c: 1, r: 1, t: 'Backbone', s: 'CSPDarknet', col: 'blue', note: 'Strided convs and C2f blocks. Taps three feature maps: 128×80×80 (stride 8), 256×40×40 (16), 512×20×20 (32).' },
        neck: { c: 2, r: 1, t: 'Neck', s: 'PAN-FPN', col: 'teal', note: 'Top-down path (upsample + concat) then bottom-up path (strided conv + concat): every level receives both fine detail and strong semantics.' },
        h3: { c: 3, r: 0, t: 'Head P3', s: '144×80×80', col: 'orange', note: 'Decoupled head, stride 8: box branch (conv→conv→1×1, 4×16 = 64 channels, distribution over 16 bins per side) and class branch (…→80). 6,400 cells.' },
        h4: { c: 3, r: 1, t: 'Head P4', s: '144×40×40', col: 'orange', note: 'Same head at stride 16: 1,600 cells (medium objects).' },
        h5: { c: 3, r: 2, t: 'Head P5', s: '144×20×20', col: 'orange', note: 'Same head at stride 32: 400 cells (large objects).' },
        cat: { c: 4, r: 1, t: 'Concat + decode', s: '8400 × 84', col: 'purple', note: '6,400 + 1,600 + 400 = 8,400 predictions. Each: 4 box distances (decoded to x,y,w,h) + 80 class scores. No objectness, no anchors.' },
        nms: { c: 5, r: 1, t: 'NMS', s: 'IoU 0.7', col: 'red', note: 'Score threshold, then (class-wise) NMS removes duplicate boxes from neighbouring cells and neighbouring levels.' },
        out: { c: 6, r: 1, t: 'Detections', s: '≤ 300 × 6', col: 'green', note: 'x1, y1, x2, y2, score, class for each survivor. Whole model ≈ 11.2 M parameters, ≈ 28.6 GFLOPs at 640².' },
      },
      edges: [['img', 'bb', 'rr'], ['bb', 'neck', 'rr'], ['neck', 'h3', 'rr'], ['neck', 'h4', 'rr'], ['neck', 'h5', 'rr'], ['h3', 'cat', 'rr'], ['h4', 'cat', 'rr'], ['h5', 'cat', 'rr'], ['cat', 'nms', 'rr'], ['nms', 'out', 'rr']],
      path: ['img', 'bb', 'neck', 'h3', 'h4', 'h5', 'cat', 'nms', 'out'],
    },
    yolov1: {
      title: 'YOLOv1 (2016), 448×448 input', W: 1090, H: 400,
      blocks: {
        img: { c: 0, r: 1, t: 'Image', s: '3×448×448', col: 'muted', note: 'Resized to 448×448.' },
        conv: { c: 1, r: 1, t: '24 conv layers', s: '→ 1024×7×7', col: 'blue', note: 'A GoogLeNet-inspired stack of 1×1 and 3×3 convs and 5 max-pools, total stride 64: 448/64 = 7 — the 7×7 feature map IS the grid.' },
        fc1: { c: 2, r: 1, t: 'FC', s: '50176 → 4096', col: 'orange', note: 'Flatten 7·7·1024 = 50,176 and a dense layer to 4096: 205.5 M weights in this single layer, most of the network.' },
        fc2: { c: 3, r: 1, t: 'FC', s: '4096 → 1470', col: 'orange', note: '4096·1470 + 1470 = 6,022,590 parameters; 1470 = 7·7·30.' },
        ten: { c: 4, r: 1, t: 'Reshape', s: '7×7×30', col: 'purple', note: 'Per cell: B=2 boxes × (x, y, w, h, conf) = 10 numbers + C=20 class probabilities = 30.' },
        thr: { c: 5, r: 1, t: 'Score + NMS', s: 'conf × p(class)', col: 'red', note: 'Multiply box confidence by the class probability, drop low scores, run NMS per class.' },
        out: { c: 6, r: 1, t: 'Detections', s: 'boxes + classes', col: 'green', note: '45 frames/s in 2016, but weaker on small and crowded objects: one cell can only predict 2 boxes and 1 class.' },
      },
      edges: [['img', 'conv', 'rr'], ['conv', 'fc1', 'rr'], ['fc1', 'fc2', 'rr'], ['fc2', 'ten', 'rr'], ['ten', 'thr', 'rr'], ['thr', 'out', 'rr']],
      path: ['img', 'conv', 'fc1', 'fc2', 'ten', 'thr', 'out'],
    },
    detr: {
      title: 'DETR (ResNet-50), 640×640 input', W: 1090, H: 400,
      blocks: {
        img: { c: 0, r: 1, t: 'Image', s: '3×640×640', col: 'muted', note: 'Input image (DETR typically resizes to ~800 px; 640 for round numbers).' },
        bb: { c: 1, r: 1, t: 'Backbone', s: 'ResNet-50 → 2048×20×20', col: 'blue', note: 'Only the last stage output is used (stride 32): 20×20 = 400 positions of 2048 channels. 23.5 M parameters.' },
        proj: { c: 2, r: 1, t: 'Project + position', s: '400 × 256', col: 'teal', note: '1×1 conv 2048→256 (524,544 params), flatten the 20×20 map to 400 tokens, add fixed sine position encodings.' },
        enc: { c: 3, r: 1, t: 'Encoder ×6', s: '400 × 256', col: 'orange', note: '6 layers of self-attention (8 heads) + FFN 256→2048→256. 1,315,072 params per layer, 7.9 M total. Every position can look at every other.' },
        qry: { c: 4, r: 0, t: 'Object queries', s: '100 × 256', col: 'pink', note: 'N = 100 learned vectors (25,600 parameters). They are the "slots" in which objects will appear.' },
        dec: { c: 4, r: 1, t: 'Decoder ×6', s: '100 × 256', col: 'purple', note: 'Each layer: self-attention among the queries, cross-attention to the 400 encoder tokens, FFN. 1,578,752 params per layer, 9.5 M total.' },
        cls: { c: 5, r: 0, t: 'Class head', s: '100 × 92', col: 'green', note: 'Linear 256→92 (23,644): 91 COCO ids + ∅ "no object".' },
        box: { c: 5, r: 2, t: 'Box head', s: '100 × 4', col: 'green', note: '3-layer MLP 256→256→256→4 (132,612 params) then sigmoid: (cx, cy, w, h) in 0..1.' },
        out: { c: 6, r: 1, t: 'Set of 100', s: 'no NMS', col: 'red', note: 'Training: Hungarian matching assigns queries to ground truth. Inference: keep predictions whose best non-∅ class score is high. Total ≈ 41 M parameters, ≈ 86 GFLOPs.' },
      },
      edges: [['img', 'bb', 'rr'], ['bb', 'proj', 'rr'], ['proj', 'enc', 'rr'], ['enc', 'dec', 'rr'], ['qry', 'dec', 'bt'], ['dec', 'cls', 'rr'], ['dec', 'box', 'rr'], ['cls', 'out', 'rr'], ['box', 'out', 'rr']],
      path: ['img', 'bb', 'proj', 'enc', 'qry', 'dec', 'cls', 'box', 'out'],
    },
  };
  AI.labs.detblueprint = function (root) {
    const names = (root.dataset.models || 'frcnn,yolov8,yolov1,detr').split(',').map((s) => s.trim()).filter((s) => BP[s]);
    const { view, ctl } = AI.labShell(root, root.dataset.title || 'Architecture blueprint', 'a pulse flows through the blocks; click any block for its tensor shape and parameters',
      '<div style="overflow-x:auto"><canvas id="c" style="min-width:920px;aspect-ratio:1090/400;cursor:pointer"></canvas></div><div class="caption" id="cap" style="min-height:3em"></div>',
      `<div>Model<div class="chips" id="mods" style="margin-top:4px"></div></div>
       ${transport(1)}
       <div class="readout" id="out" style="white-space:pre-wrap;font-size:12px;line-height:1.55"></div>`);
    wide(root);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#c'));
    let cur = names[0], last = '', sel = null;
    const P = player(root, { n: BP[cur].path.length, dur: () => 2.1, onJump: () => (last = '') });
    const chips = names.map((nm) => chip(q('#mods'), ({ frcnn: 'Faster R-CNN', yolov8: 'YOLOv8-style', yolov1: 'YOLOv1', detr: 'DETR' })[nm], () => { cur = nm; sel = null; P.n = BP[nm].path.length; P.restart(); chips.forEach((b, i) => b.classList.toggle('on', names[i] === cur)); }));
    chips.forEach((b, i) => b.classList.toggle('on', names[i] === cur));
    const BW = 118, BH = 92, X = (b) => 12 + b.c * 150, Y = (b) => 28 + b.r * 125;
    const rectOf = (b) => ({ x: X(b), y: Y(b), w: BW, h: BH });
    c.cv.addEventListener('click', (e) => {
      const r = c.cv.getBoundingClientRect(), mod = BP[cur], m = fitW(c, mod.W, mod.H), x = (e.clientX - r.left - m.ox) / m.s, y = (e.clientY - r.top - m.oy) / m.s;
      for (const id in mod.blocks) { const k = rectOf(mod.blocks[id]); if (x >= k.x && x <= k.x + k.w && y >= k.y && y <= k.y + k.h) { sel = id; P.playing = false; q('#play').textContent = '▶ Play'; const pi = mod.path.indexOf(id); if (pi >= 0) P.pos = pi + 0.999; last = ''; return; } }
    });
    const edgePts = (a, b, t) => {
      const A = rectOf(a), B = rectOf(b);
      if (t === 'rr') { const x0 = A.x + A.w, y0 = A.y + A.h / 2, x1 = B.x, y1 = B.y + B.h / 2, xm = (x0 + x1) / 2; return [[x0, y0], [xm, y0], [xm, y1], [x1, y1]]; }
      if (t === 'bt') return A.y < B.y ? [[A.x + A.w / 2, A.y + A.h], [B.x + B.w / 2, B.y]] : [[A.x + A.w / 2, A.y], [B.x + B.w / 2, B.y + B.h]];
      return [[A.x + A.w / 2, A.y + A.h], [A.x + A.w / 2, B.y + B.h / 2], [B.x, B.y + B.h / 2]]; // 'bl'
    };
    const along = (pts, t) => { let tot = 0; const L = []; for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(d); tot += d; } let s = t * tot; for (let i = 0; i < L.length; i++) { if (s <= L[i] || i === L.length - 1) { const u = L[i] ? Math.min(1, s / L[i]) : 0; return [lerp(pts[i][0], pts[i + 1][0], u), lerp(pts[i][1], pts[i + 1][1], u)]; } s -= L[i]; } return pts[0]; };

    AI.loop(root, (dt) => {
      P.update(dt);
      const mod = BP[cur], idx = P.idx(), fr = P.frac(), key = [P.pos.toFixed(3), cur, sel, c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), m = fitW(c, mod.W, mod.H), g = enter(c, m), path = mod.path, active = path[idx], prevId = idx > 0 ? path[idx - 1] : null;
      label(g, m, mod.title, 12, 12, C.fg, { size: 13, bold: true });
      // edges
      mod.edges.forEach(([a, b, t]) => { const pts = edgePts(mod.blocks[a], mod.blocks[b], t), on = (prevId === a && active === b); g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.strokeStyle = on ? C.fg : C.muted; g.lineWidth = (on ? 2.5 : 1.5) / m.s; g.stroke(); const e = pts[pts.length - 1], p0 = pts[pts.length - 2]; AI.arrow(g, p0[0] + (e[0] - p0[0]) * 0.9, p0[1] + (e[1] - p0[1]) * 0.9, e[0], e[1], on ? C.fg : C.muted, 1.5 / m.s); });
      // blocks
      for (const id in mod.blocks) {
        const b = mod.blocks[id], r = rectOf(b), col = C[b.col], on = id === active, vis = path.indexOf(id) <= idx || path.indexOf(id) < 0;
        g.fillStyle = on ? rgba(col, 0.38) : rgba(col, vis ? 0.16 : 0.06); g.fillRect(r.x, r.y, r.w, r.h);
        g.strokeStyle = id === sel ? C.fg : col; g.lineWidth = (on || id === sel ? 3.5 : 1.6) / m.s; g.strokeRect(r.x, r.y, r.w, r.h);
        label(g, m, b.t, r.x + r.w / 2, r.y + 20, C.fg, { align: 'center', size: 12.5, bold: true });
        const parts = b.s.split(' '); let lines = [], cur2 = ''; parts.forEach((w) => { if ((cur2 + ' ' + w).trim().length > 15) { lines.push(cur2); cur2 = w; } else cur2 = (cur2 + ' ' + w).trim(); }); lines.push(cur2);
        lines.slice(0, 3).forEach((s, i) => label(g, m, s, r.x + r.w / 2, r.y + 48 + i * 15, C.muted, { align: 'center', size: 11 }));
      }
      // pulse
      if (prevId && active) { const e = mod.edges.find(([a, b]) => a === prevId && b === active); if (e && P.pos < path.length) { const pts = edgePts(mod.blocks[prevId], mod.blocks[active], e[2]), p = along(pts, smooth(Math.min(1, fr * 1.6))); g.fillStyle = C.fg; g.strokeStyle = C.blue; g.lineWidth = 3 / m.s; g.beginPath(); g.arc(p[0], p[1], 8 / m.s, 0, 7); g.fill(); g.stroke(); } }
      g.restore();
      const show = sel || active, b = mod.blocks[show];
      q('#out').innerHTML = `<b>${b.t}</b>\nshape: <b class="o">${b.s}</b>\n${b.note}`;
      q('#cap').textContent = `${idx + 1}/${path.length} — ${mod.blocks[active].t}: ${mod.blocks[active].note}`;
    }, { fps: 30 });
  };
})();
