/* GeoAI · Module 3 labs (geo-preparation.html)
 * rasterLab    polygon rasteriser: the pixel-centre rule, ray casting cell by cell
 * tileLab      tiling & stitching: stride, overlap, edge rules, overwrite vs feathered blending
 * normLab      percentile normalisation: training statistics applied to a test scene
 * splitLab     random vs spatial block splits: leakage counted neighbour by neighbour
 * augLab       geometric and band-aware augmentation: what keeps geography and indices
 * manifestLab  dataset manifest builder with validation
 */
(function () {
  const TAU = Math.PI * 2;
  const clamp = AI.clamp, lerp = AI.lerp;
  const F = (v, d) => AI.fmt(v, d);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const rgba = (hex, a) => { const c = AI.rgb(hex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
  const rgbaA = (c, a) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
  const themeKey = () => document.documentElement.dataset.theme || '';
  const FONT = '10px ui-monospace,Menlo,Consolas,monospace';

  /* ---------- shared lab chrome ---------- */
  const canvasIn = (view, h) => { const cv = el('canvas'); cv.style.height = h + 'px'; view.appendChild(cv); return AI.canvas(cv); };
  const capIn = (view) => { const c = el('div', 'geo-cap'); c.style.cssText = 'font-size:13px;line-height:1.55;margin-top:8px;min-height:3.1em'; view.appendChild(c); return c; };
  const setHTML = (node, html) => { if (node.__h !== html) { node.__h = html; node.innerHTML = html; } };
  const setText = (node, s) => { if (node.__t !== s) { node.__t = s; node.textContent = s; } };

  // play / pause / step / restart and speed. Returns a state object the lab reads each frame.
  function transport(ctl, speed0) {
    const T = { playing: true, speed: speed0 || 1, stepReq: 0, restartReq: false };
    const row = el('div', 'row');
    row.innerHTML = '<button class="btn small" data-a="pp">⏸ Pause</button><button class="btn small" data-a="step">Step</button>' +
      '<button class="btn small" data-a="rs">Restart</button>' +
      '<label style="flex-direction:row;align-items:center;gap:6px;color:var(--muted)">Speed <input type="range" min="0.25" max="3" step="0.05" value="' + T.speed + '" style="width:120px"><span class="v" style="color:var(--fg)">' + T.speed.toFixed(2) + '×</span></label>';
    const pp = row.querySelector('[data-a=pp]');
    pp.onclick = () => { T.playing = !T.playing; pp.textContent = T.playing ? '⏸ Pause' : '▶ Play'; };
    row.querySelector('[data-a=step]').onclick = () => { T.playing = false; pp.textContent = '▶ Play'; T.stepReq++; };
    row.querySelector('[data-a=rs]').onclick = () => { T.restartReq = true; };
    const sl = row.querySelector('input'), sv = row.querySelector('.v');
    sl.oninput = () => { T.speed = +sl.value; sv.textContent = T.speed.toFixed(2) + '×'; };
    ctl.appendChild(row);
    return T;
  }

  function slider(ctl, label, min, max, step, val, fmt, onInput) {
    const l = el('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const show = () => { v.textContent = fmt ? fmt(+inp.value) : String(+inp.value); };
    inp.oninput = () => { show(); if (onInput) onInput(+inp.value); };
    show();
    ctl.appendChild(l);
    return { get: () => +inp.value };
  }

  function seg(ctl, label, opts, cur, onPick) {
    const wrap = el('div', 'row');
    wrap.innerHTML = `<span style="color:var(--muted);font-size:12px;width:100%">${label}</span>`;
    let val = cur;
    const btns = opts.map(([k, t]) => {
      const b = el('button', 'btn small' + (k === cur ? ' on' : ''), t);
      b.onclick = () => {
        val = k;
        btns.forEach((x, i) => x.classList.toggle('on', opts[i][0] === k));
        if (onPick) onPick(k);
      };
      wrap.appendChild(b);
      return b;
    });
    ctl.appendChild(wrap);
    return { get: () => val };
  }

  function check(ctl, label, val, onChange) {
    const l = el('label');
    l.style.cssText = 'flex-direction:row;align-items:center;gap:8px;color:var(--fg)';
    l.innerHTML = `<input type="checkbox" ${val ? 'checked' : ''}><span>${label}</span>`;
    const i = l.querySelector('input');
    i.onchange = () => { if (onChange) onChange(i.checked); };
    ctl.appendChild(l);
    return { get: () => i.checked };
  }

  function text(ctl, label, val, onInput) {
    const l = el('label');
    l.innerHTML = `<span>${label}</span><input type="text" spellcheck="false" value="${String(val).replace(/"/g, '&quot;')}">`;
    const i = l.querySelector('input');
    i.oninput = () => { if (onInput) onInput(i.value); };
    ctl.appendChild(l);
    return { get: () => i.value, el: i };
  }

  function select(ctl, label, opts, val, onChange) {
    const l = el('label');
    l.innerHTML = `<span>${label}</span><select>${opts.map((o) => `<option value="${o}" ${o === val ? 'selected' : ''}>${o}</option>`).join('')}</select>`;
    const s = l.querySelector('select');
    s.onchange = () => { if (onChange) onChange(s.value); };
    ctl.appendChild(l);
    return { get: () => s.value };
  }

  /* =====================================================================
   * 1. Polygon rasteriser (labels-rasterization)
   * ===================================================================== */
  const DEF_POLY = '1.3,1.2; 4.8,1.6; 4.2,4.7; 1.6,4.1';

  function parseVerts(s) {
    const parts = s.split(';').map((p) => p.trim()).filter(Boolean);
    if (parts.length < 3) throw new Error('Need at least 3 vertices, separated by semicolons');
    return parts.map((p, i) => {
      const raw = p.split(',');
      if (raw.length !== 2 || raw.some((t) => t.trim() === '')) throw new Error(`Vertex ${i + 1} must look like x,y`);
      const xy = raw.map(Number);
      if (xy.some((v) => !Number.isFinite(v))) throw new Error(`Vertex ${i + 1} has a value that is not a number`);
      if (xy.some((v) => v < 0 || v > 6)) throw new Error('Keep every coordinate between 0 and 6 (the 6 × 6 grid)');
      return xy;
    });
  }

  // Edge crossings to the right of (px, py): the ray-casting test from the lesson.
  function crossings(px, py, P) {
    const xs = [];
    for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
      const [xi, yi] = P[i], [xj, yj] = P[j];
      if ((yi > py) !== (yj > py)) {
        const x = xj + ((py - yj) * (xi - xj)) / (yi - yj);
        if (px < x) xs.push(x);
      }
    }
    return xs;
  }

  function coverage(P) {
    const out = [], S = 50;
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
      let n = 0;
      for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
        if (crossings(c + (a + 0.5) / S, r + (b + 0.5) / S, P).length % 2) n++;
      }
      out.push(n / (S * S));
    }
    return out;
  }

  function shoelace(P) {
    let s = 0;
    for (let i = 0; i < P.length; i++) { const [x1, y1] = P[i], [x2, y2] = P[(i + 1) % P.length]; s += x1 * y2 - x2 * y1; }
    return Math.abs(s) / 2;
  }

  AI.labs.rasterLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Polygon rasteriser', 'the pixel-centre rule, one cell at a time', '', '');
    const mapC = canvasIn(view, 300), clsC = canvasIn(view, 150), cap = capIn(view);
    const err = el('div', 'err');
    let P = parseVerts(DEF_POLY), info = [], cov = [], showCov = false;
    let k = 0, hold = 0, last = '';
    const tick = AI.ticker();

    function build() {
      info = [];
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
        const px = c + 0.5, py = r + 0.5, xs = crossings(px, py, P);
        info.push({ r, c, px, py, xs, lab: xs.length % 2 });
      }
      cov = coverage(P);
    }
    build();

    const vin = text(ctl, 'polygon vertices, x,y; x,y; … (grid 0 to 6)', DEF_POLY, (s) => {
      try {
        P = parseVerts(s);
        vin.el.classList.remove('bad');
        err.textContent = '';
        build();
        k = 0; hold = 0;
      } catch (e) {
        vin.el.classList.add('bad');
        err.textContent = e.message;
      }
    });
    ctl.appendChild(err);
    check(ctl, 'show coverage (share of each pixel inside the polygon)', false, (v) => { showCov = v; });
    const ro = el('div', 'readout');
    ctl.appendChild(ro);
    const T = transport(ctl, 1);

    function draw() {
      const C = AI.col(), c = mapC, g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const side = Math.min(c.w, c.h) - 16, sc = side / 6, ox = (c.w - side) / 2, oy = 8;
      const X = (x) => ox + x * sc, Y = (y) => oy + y * sc;
      let labelled = 0;
      for (let i = 0; i < 36; i++) {
        const q = info[i], x0 = X(q.c), y0 = Y(q.r);
        if (i < k && q.lab) { g.fillStyle = rgba(C.blue, 0.22); g.fillRect(x0, y0, sc, sc); }
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x0, y0, sc, sc);
        if (showCov) { g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left'; g.fillText(cov[i].toFixed(2), x0 + 3, y0 + 12); }
        if (i < k && q.lab) labelled++;
      }
      g.beginPath();
      P.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
      g.closePath();
      g.fillStyle = rgba(C.blue, 0.08); g.fill();
      g.strokeStyle = C.blue; g.lineWidth = 2; g.stroke();
      for (let i = 0; i < k; i++) {
        const q = info[i];
        g.beginPath(); g.arc(X(q.px), Y(q.py), 3, 0, TAU);
        g.fillStyle = q.lab ? C.blue : C.muted; g.fill();
      }
      if (k < 36) {
        const q = info[k];
        g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(X(q.c), Y(q.r), sc, sc);
        g.setLineDash([5, 4]); g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(X(q.px), Y(q.py)); g.lineTo(X(6), Y(q.py)); g.stroke();
        g.setLineDash([]);
        q.xs.forEach((x) => { g.beginPath(); g.arc(X(x), Y(q.py), 4.5, 0, TAU); g.fillStyle = C.red; g.fill(); });
        g.beginPath(); g.arc(X(q.px), Y(q.py), 5, 0, TAU); g.fillStyle = C.orange; g.fill();
      }

      const d = clsC, h = d.ctx;
      h.clearRect(0, 0, d.w, d.h);
      const s2 = Math.min(d.w, d.h) - 16, sc2 = s2 / 6, ox2 = (d.w - s2) / 2, oy2 = 8;
      for (let i = 0; i < 36; i++) {
        const q = info[i], x0 = ox2 + q.c * sc2, y0 = oy2 + q.r * sc2;
        h.fillStyle = i < k ? (q.lab ? C.blue : C.bg3) : C.bg2;
        h.fillRect(x0, y0, sc2, sc2);
        h.strokeStyle = i === k && k < 36 ? C.orange : C.border;
        h.lineWidth = i === k && k < 36 ? 2.5 : 1;
        h.strokeRect(x0, y0, sc2, sc2);
      }

      const A = shoelace(P), total = info.reduce((s, q) => s + q.lab, 0);
      if (k >= 36) {
        setHTML(cap, `<b>Done.</b> ${total} pixels are labelled, so ${total} m² is building. The polygon's true area by the shoelace formula is ${F(A, 3)} m². The two differ by ${F(Math.abs(total - A), 2)} m²: that is the boundary effect of a point test.`);
      } else if (k === 0) {
        setHTML(cap, 'Press play. Each cell is tested at its centre (the dot). The dashed ray runs to the right; each red tick is an edge crossing. An odd number of ticks means inside.');
      } else {
        const q = info[k], n = q.xs.length;
        setHTML(cap, `Cell (row ${q.r}, column ${q.c}), centre (${q.px}, ${q.py}). The ray crosses the boundary ${n} time${n === 1 ? '' : 's'}: ${n % 2 ? 'odd, so inside, label 1' : 'even, so outside, label 0'}.`);
      }
      setHTML(ro, `labelled ${labelled} of ${k} cells tested\nfinal labelled area ${total} m² · polygon area (shoelace) ${F(A, 3)} m²\npixel 1 m, grid 6 × 6, y points down`);
    }

    AI.loop(root, (dt) => {
      if (T.restartReq) { T.restartReq = false; k = 0; hold = 0; }
      if (T.stepReq) { T.stepReq = 0; k = k >= 36 ? 0 : k + 1; }
      if (T.playing) {
        if (k < 36) { const n = tick(dt, 0.8 * T.speed); if (n) { k = Math.min(36, k + n); } }
        else { hold += dt; if (hold > 3) { k = 0; hold = 0; } }
      }
      const key = [k, JSON.stringify(P), showCov, themeKey(), mapC.w, mapC.h, clsC.w, clsC.h].join('|');
      if (key === last) return;
      last = key;
      draw();
    });
  };

  /* =====================================================================
   * 2. Tiling & stitching (tiling-chipping)
   * ===================================================================== */
  // Illustrative scene: smooth field values in [0, 1].
  const sceneV = (x, y) => 0.5 + 0.25 * Math.sin(x / 70) * Math.cos(y / 90) + 0.15 * Math.sin((x + y) / 40);

  function putCol(d, p, v) {
    v = clamp(v, 0, 1);
    d[p] = 255 * v; d[p + 1] = 255 * (1 - Math.abs(2 * v - 1)) * 0.85; d[p + 2] = 255 * (1 - v); d[p + 3] = 255;
  }

  AI.labs.tileLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Tiling & stitching', 'cut a scene into windows, then stitch the predictions back', '', '');
    const cv = canvasIn(view, 330), cap = capIn(view);
    const ro = el('div', 'readout');
    const st = { S: 1000, T: 256, O: 32, edge: 'shift', blend: 'feather' };
    const tick = AI.ticker();
    let G = null, tiles = [], W = 0, acc = null, rawC = null, accC = null;
    let phase = 'cut', q = 0, hold = 0, dirty = true, lastSize = '';
    cv.onresize = () => { dirty = true; };

    function geom() {
      const S = st.S, T = st.T, O = Math.min(st.O, T - 32), s = T - O;
      const starts = [];
      let n;
      if (st.edge === 'drop') {
        n = S >= T ? Math.floor((S - T) / s) + 1 : 0;
        for (let k = 0; k < n; k++) starts.push(k * s);
      } else {
        n = S > T ? Math.ceil((S - T) / s) + 1 : 1;
        for (let k = 0; k < n; k++) starts.push(st.edge === 'shift' ? Math.min(k * s, Math.max(0, S - T)) : k * s);
      }
      const covered = n ? starts[n - 1] + T : 0;
      return { S, T, O, s, n, starts, uncovered: Math.max(0, S - covered), pad: Math.max(0, covered - S) };
    }

    function paintRaw() {
      const ctx = rawC.getContext('2d'), img = ctx.createImageData(W, W), d = img.data;
      for (let i = 0; i < W; i++) for (let j = 0; j < W; j++) putCol(d, (i * W + j) * 4, sceneV((j + 0.5) * 2, (i + 0.5) * 2));
      ctx.putImageData(img, 0, 0);
    }

    function paintAcc() {
      const ctx = accC.getContext('2d'), img = ctx.createImageData(W, W), d = img.data;
      for (let idx = 0; idx < W * W; idx++) {
        const p = idx * 4, w = acc.w[idx];
        if (w > 0) putCol(d, p, acc.sum[idx] / w);
        else { d[p] = 30; d[p + 1] = 34; d[p + 2] = 42; d[p + 3] = 255; }
      }
      ctx.putImageData(img, 0, 0);
    }

    // Stitch one tile: its prediction = scene value + an edge artefact that alternates in sign between neighbours.
    function stitch(t) {
      const K = 2, S = G.S, T = t.T, x0 = t.x0, y0 = t.y0;
      const j0 = Math.max(0, Math.ceil(x0 / K - 0.5)), j1 = Math.min(W - 1, Math.floor((x0 + T) / K - 0.5));
      const i0 = Math.max(0, Math.ceil(y0 / K - 0.5)), i1 = Math.min(W - 1, Math.floor((y0 + T) / K - 0.5));
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const x = (j + 0.5) * K, y = (i + 0.5) * K;
        if (x >= S || y >= S) continue;
        const dEdge = Math.max(0, Math.min(x - x0, x0 + T - x, y - y0, y0 + T - y));
        const p = sceneV(x, y) + t.sign * 0.3 * Math.exp(-dEdge / 10);
        const idx = i * W + j;
        if (st.blend === 'feather') { const w = dEdge + 1; acc.sum[idx] += w * p; acc.w[idx] += w; }
        else { acc.sum[idx] = p; acc.w[idx] = 1; }
      }
    }

    function rebuild() {
      G = geom();
      tiles = [];
      for (let i = 0; i < G.n; i++) for (let j = 0; j < G.n; j++) {
        tiles.push({ x0: G.starts[j], y0: G.starts[i], T: G.T, sign: (i + j) % 2 ? -1 : 1 });
      }
      W = Math.round(G.S / 2);
      acc = { sum: new Float32Array(W * W), w: new Float32Array(W * W) };
      rawC = document.createElement('canvas'); rawC.width = W; rawC.height = W;
      accC = document.createElement('canvas'); accC.width = W; accC.height = W;
      paintRaw();
      phase = 'cut'; q = 0; hold = 0; dirty = true;
    }

    slider(ctl, 'scene size', 500, 2000, 100, 1000, (v) => v + ' px', (v) => { st.S = v; rebuild(); });
    slider(ctl, 'tile size', 64, 512, 32, 256, (v) => v + ' px', (v) => { st.T = v; rebuild(); });
    slider(ctl, 'overlap (kept at most tile − 32)', 0, 480, 16, 32, (v) => v + ' px', (v) => { st.O = v; rebuild(); });
    seg(ctl, 'edge rule', [['shift', 'shift last tile'], ['pad', 'pad scene'], ['drop', 'drop partial (naive)']], 'shift', (v) => { st.edge = v; rebuild(); });
    seg(ctl, 'stitching', [['overwrite', 'overwrite (no blend)'], ['feather', 'feather blend']], 'feather', (v) => { st.blend = v; rebuild(); });
    ctl.appendChild(ro);
    const T = transport(ctl, 1);
    rebuild();

    function stepOnce() {
      if (phase === 'cut') {
        if (q < tiles.length) { q++; dirty = true; }
        else { phase = 'stitch'; q = 0; acc.sum.fill(0); acc.w.fill(0); dirty = true; }
      } else if (phase === 'stitch') {
        if (q < tiles.length) { stitch(tiles[q]); paintAcc(); q++; dirty = true; }
        else { phase = 'hold'; hold = 0; dirty = true; }
      }
    }

    function draw() {
      const C = AI.col(), c = cv, g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const side = Math.min(c.w, c.h - 4), ox = (c.w - side) / 2, oy = 2, f = side / G.S;
      g.save();
      g.beginPath(); g.rect(ox, oy, side, side); g.clip();
      g.drawImage(phase === 'cut' ? rawC : accC, ox, oy, side, side);
      tiles.forEach((t, e) => {
        const cur = phase !== 'hold' && e === q - 1;
        const shown = phase === 'cut' ? e < q : true;
        if (!shown) return;
        g.strokeStyle = cur ? C.orange : rgba(C.fg, 0.35);
        g.lineWidth = cur ? 2.5 : 1;
        g.strokeRect(ox + t.x0 * f, oy + t.y0 * f, G.T * f, G.T * f);
      });
      g.restore();
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(ox, oy, side, side);

      const N = tiles.length;
      if (phase === 'cut') {
        const t = tiles[Math.max(0, q - 1)];
        setHTML(cap, q === 0
          ? `Cutting begins. The raw scene is shown; each window will be cut out in turn. Neighbouring windows start ${G.s} px apart${G.O > 0 ? ', so they share ' + G.O + ' px' : ', so they touch with no shared pixels'}.`
          : `Cutting tile ${q} of ${N} at x = ${t.x0}, y = ${t.y0} px.`);
      } else if (phase === 'stitch') {
        const t = tiles[Math.max(0, q - 1)];
        let ov = 0;
        for (let e = 0; e < q - 1; e++) {
          const t2 = tiles[e];
          if (t2.x0 < t.x0 + G.T && t.x0 < t2.x0 + G.T && t2.y0 < t.y0 + G.T && t.y0 < t2.y0 + G.T) ov++;
        }
        setHTML(cap, `Stitching tile ${q} of ${N}. It overlaps ${ov} earlier tile${ov === 1 ? '' : 's'}. ${st.blend === 'feather'
          ? "Each pixel is averaged with weight = distance to that tile's own edge + 1, so the tile whose interior is nearer wins."
          : 'Overwrite: the tile written last owns every shared pixel, so each tile border shows up as a seam.'}`);
      } else {
        setHTML(cap, `Stitched all ${N} tiles. ${st.blend === 'feather' ? 'Seams are softened wherever neighbours overlap.' : 'Seams stay visible where tiles meet. Switch the overlap to 0 and back to compare.'}`);
      }
      const Nt = G.n * G.n;
      setHTML(ro, `tiles ${G.n} × ${G.n} = ${Nt} · stride ${G.s} px · overlap ${G.O} px\n` +
        `formula: ${st.edge === 'drop' ? '⌊(S−T)/s⌋+1' : '⌈(S−T)/s⌉+1'} = ${st.edge === 'drop' ? '⌊' : '⌈'}(${G.S}−${G.T})/${G.s}${st.edge === 'drop' ? '⌋' : '⌉'}+1 = ${G.n}\n` +
        `each pixel is read ${(Nt * G.T * G.T / (G.S * G.S)).toFixed(2)}× on average · ` +
        (G.uncovered ? `<span class="r">uncovered strip ${G.uncovered} px</span>` : G.pad ? `padding ${G.pad} px (no-data)` : 'every pixel covered'));
    }

    AI.loop(root, (dt) => {
      if (T.restartReq) { T.restartReq = false; phase = 'cut'; q = 0; hold = 0; acc.sum.fill(0); acc.w.fill(0); dirty = true; }
      if (T.stepReq) { T.stepReq = 0; if (phase === 'hold') { phase = 'cut'; q = 0; } stepOnce(); }
      if (T.playing) {
        if (phase === 'hold') { hold += dt; if (hold > 3) { phase = 'cut'; q = 0; dirty = true; } }
        else { const n = tick(dt, (phase === 'cut' ? 4 : 1.2) * T.speed); for (let i = 0; i < n; i++) stepOnce(); }
      }
      const size = cv.w + ',' + cv.h;
      if (!dirty && size === lastSize) return;
      lastSize = size; dirty = false;
      draw();
    });
  };

  /* =====================================================================
   * 3. Percentile normalisation (band-normalization)
   * ===================================================================== */
  AI.labs.normLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Percentile normalisation', 'training pixels set the scale; the same two numbers are applied to every scene', '', '');
    const cv = canvasIn(view, 340), cap = capIn(view);
    const rT = AI.rng(11), rS = AI.rng(12);
    // Illustrative band: land around 1800, water near 700, a small share of bright clouds near 9000-11000.
    const gen = (r, n, shift) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const u = r(); let v;
        if (u < 0.02) v = 9000 + 2000 * r();
        else if (u < 0.32) v = 700 + 120 * AI.randn(r);
        else v = 1800 + 300 * AI.randn(r);
        out.push(Math.round(clamp(v + shift, 1, 12000)));
      }
      return out.sort((a, b) => a - b);
    };
    const train = gen(rT, 420, 0), test = gen(rS, 200, 350); // the test scene is hazier (+350 DN)
    const pct = (s, p) => {
      const h = (p / 100) * (s.length - 1), lo = Math.floor(h), f = h - lo;
      return lo + 1 < s.length ? s[lo] + f * (s[lo + 1] - s[lo]) : s[lo];
    };
    const nrm = (v, a, b) => (b > a ? clamp((v - a) / (b - a), 0, 1) : 0);
    const st = { pLo: 2, pHi: 98, test: 'train', target: 1, m: 0, ex: 1650, exT: 2600 };
    const stats = () => ({
      lo: pct(train, st.pLo), hi: pct(train, st.pHi),
      tlo: pct(test, st.pLo), thi: pct(test, st.pHi),
    });

    slider(ctl, 'lower percentile', 0, 20, 0.5, 2, (v) => v.toFixed(1) + ' %', (v) => { st.pLo = v; });
    slider(ctl, 'upper percentile', 80, 100, 0.5, 98, (v) => v.toFixed(1) + ' %', (v) => { st.pHi = v; });
    slider(ctl, 'example training pixel', 0, 12000, 10, 1650, (v) => v + ' DN', (v) => { st.ex = v; });
    slider(ctl, 'example test pixel', 0, 12000, 10, 2600, (v) => v + ' DN', (v) => { st.exT = v; });
    seg(ctl, 'test scene is scaled with', [['train', 'training statistics (correct)'], ['own', 'its own statistics (leaks scale)']], 'train', (v) => { st.test = v; });
    seg(ctl, 'show', [['raw', 'raw DN'], ['norm', 'normalised']], 'norm', (v) => { st.target = v === 'norm' ? 1 : 0; });
    const ro = el('div', 'readout');
    ctl.appendChild(ro);
    const T = transport(ctl, 1);

    let last = '', dirty = true;
    cv.onresize = () => { dirty = true; };

    function draw() {
      const C = AI.col(), c = cv, g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const padL = 14, pw = W - 28;
      const S = stats(), ok = st.pLo < st.pHi;
      const lo = S.lo, hi = S.hi;
      const hT = Math.round(H * 0.40), bins = 48, cnt = new Array(bins).fill(0);
      train.forEach((v) => { cnt[Math.min(bins - 1, Math.floor((v / 12000) * bins))]++; });
      const mx = Math.max(...cnt), bw = pw / bins;
      g.fillStyle = rgba(C.blue, 0.55);
      cnt.forEach((n, i) => { const hh = (n / mx) * (hT - 24); g.fillRect(padL + i * bw + 0.5, hT - hh, bw - 1, hh); });
      g.strokeStyle = C.border; g.lineWidth = 1;
      g.beginPath(); g.moveTo(padL, hT + 0.5); g.lineTo(padL + pw, hT + 0.5); g.stroke();
      g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
      g.fillText('training histogram, raw DN 0 to 12000', padL, 12);
      if (ok) {
        const xLo = padL + clamp(lo / 12000, 0, 1) * pw, xHi = padL + clamp(hi / 12000, 0, 1) * pw;
        g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(xLo, 16); g.lineTo(xLo, hT); g.stroke();
        g.strokeStyle = C.red; g.beginPath(); g.moveTo(xHi, 16); g.lineTo(xHi, hT); g.stroke();
        g.fillStyle = C.orange; g.textAlign = 'left'; g.fillText('Q' + st.pLo + ' ' + F(lo, 0), xLo + 3, 26);
        g.fillStyle = C.red; g.textAlign = 'right'; g.fillText('Q' + st.pHi + ' ' + F(hi, 0), xHi - 3, 26);
      }

      const yTrain = Math.round(H * 0.58), yTest = Math.round(H * 0.84);
      const a = st.test === 'train' ? S.lo : S.tlo, b = st.test === 'train' ? S.hi : S.thi;
      const posTrain = (v) => lerp(padL + clamp(v / 12000, 0, 1) * pw, padL + nrm(v, lo, hi) * pw, st.m);
      const posTest = (v) => lerp(padL + clamp(v / 12000, 0, 1) * pw, padL + nrm(v, a, b) * pw, st.m);
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('training pixels', padL, yTrain - 12);
      g.fillText('test scene', padL, yTest - 12);
      g.fillStyle = rgba(C.blue, 0.6);
      train.forEach((v) => { g.beginPath(); g.arc(posTrain(v), yTrain, 1.9, 0, TAU); g.fill(); });
      g.fillStyle = rgba(C.orange, 0.7);
      test.forEach((v) => { g.beginPath(); g.arc(posTest(v), yTest, 1.9, 0, TAU); g.fill(); });
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText(st.m > 0.5 ? 'normalised 0 … 1' : 'raw DN 0 … 12000', padL, H - 6);

      const exN = nrm(st.ex, lo, hi), exTrain = nrm(st.exT, lo, hi), exOwn = nrm(st.exT, S.tlo, S.thi);
      setHTML(cap, st.m > 0.5
        ? `Each dot is one pixel, mapped to (x − Q<sub>lo</sub>) / (Q<sub>hi</sub> − Q<sub>lo</sub>) and clipped to 0–1. Dark water below Q<sub>lo</sub> piles up at 0; bright clouds above Q<sub>hi</sub> pile up at 1. The test scene is hazier, so its dots sit further right unless you scale it with its own statistics.`
        : `Raw digital numbers. The training pixels set Q<sub>lo</sub> = ${F(lo, 1)} (orange line) and Q<sub>hi</sub> = ${F(hi, 1)} (red line). Choose "normalised" to watch every dot move.`);
      setHTML(ro, `training: Q<sub>lo</sub> = ${F(lo, 1)}, Q<sub>hi</sub> = ${F(hi, 1)}, denominator ${F(hi - lo, 1)}\n` +
        `example training pixel ${st.ex}: (${st.ex} − ${F(lo, 1)}) / ${F(hi - lo, 1)} = ${F(exN, 4)}\n` +
        `example test pixel ${st.exT}: training statistics → ${F(exTrain, 4)} · own statistics (Q<sub>lo</sub> ${F(S.tlo, 0)}, Q<sub>hi</sub> ${F(S.thi, 0)}) → ${F(exOwn, 4)}`);
    }

    AI.loop(root, (dt) => {
      if (T.restartReq) { T.restartReq = false; st.m = 0; }
      if (T.stepReq) { T.stepReq = 0; st.m = st.target; }
      if (T.playing) st.m += (st.target - st.m) * (1 - Math.exp(-dt * 2.2 * T.speed));
      if (Math.abs(st.target - st.m) < 0.002) st.m = st.target;
      const key = [st.pLo, st.pHi, st.test, st.ex, st.exT, st.m.toFixed(3), cv.w, cv.h, themeKey()].join('|');
      if (key === last && !dirty) return;
      last = key; dirty = false;
      draw();
    });
  };

  /* =====================================================================
   * 4. Spatial splits (spatial-splits)
   * ===================================================================== */
  const NB = 6;
  const PAIRS = [];
  for (let r = 0; r < NB; r++) for (let c = 0; c < NB; c++) {
    if (c < NB - 1) PAIRS.push([r * NB + c, r * NB + c + 1, 'h', r, c]);
    if (r < NB - 1) PAIRS.push([r * NB + c, (r + 1) * NB + c, 'v', r, c]);
  }

  // Random block assignment: one uniform draw per block in row-major order (train < 0.70, val < 0.85, else test).
  function assign(mode, seed) {
    const g = [];
    if (mode === 'random') {
      const R = AI.rng(seed);
      for (let i = 0; i < NB * NB; i++) { const u = R(); g.push(u < 0.7 ? 'train' : u < 0.85 ? 'val' : 'test'); }
    } else {
      for (let r = 0; r < NB; r++) for (let c = 0; c < NB; c++) g.push(c < 4 ? 'train' : c === 4 ? 'val' : 'test');
    }
    return g;
  }

  // Steps (across shared edges) from each block to the nearest held-out block.
  function distToHeld(split) {
    const d = new Array(NB * NB).fill(Infinity), q = [];
    split.forEach((s, i) => { if (s !== 'train') { d[i] = 0; q.push(i); } });
    for (let h = 0; h < q.length; h++) {
      const i = q[h], r = Math.floor(i / NB), c = i % NB;
      [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([dr, dc]) => {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || cc < 0 || rr >= NB || cc >= NB) return;
        const j = rr * NB + cc;
        if (d[j] === Infinity) { d[j] = d[i] + 1; q.push(j); }
      });
    }
    return d;
  }

  AI.labs.splitLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Random versus spatial block splits', 'a leak is a neighbouring pair that links a training block to a held-out block', '', '');
    const cv = canvasIn(view, 340), cap = capIn(view);
    const st = { mode: 'random', buffer: 0, seed: 7 };
    const tick = AI.ticker();
    let status = [], pairs = [], cur = null, p = 0, hold = 0, sum = {}, dirty = true, last = '';
    cv.onresize = () => { dirty = true; };

    function rebuild() {
      const split = assign(st.mode, st.seed), d = distToHeld(split);
      status = split.map((s, i) => (s === 'train' && d[i] <= st.buffer ? 'buffer' : s));
      sum = { train: 0, val: 0, test: 0, buffer: 0 };
      status.forEach((s) => { sum[s]++; });
      pairs = PAIRS.map(([a, b, o, r, c]) => {
        const sa = status[a], sb = status[b];
        let type;
        if (sa === 'buffer' || sb === 'buffer') type = 'buffer';
        else if (sa === sb) type = 'same';
        else if (sa === 'train' || sb === 'train') type = 'leak';
        else type = 'heldout';
        return { a, b, o, r, c, type, sa, sb };
      });
      sum.leak = pairs.filter((x) => x.type === 'leak').length;
      sum.heldPairs = pairs.filter((x) => x.type === 'heldout').length;
      p = 0; hold = 0; dirty = true;
    }

    seg(ctl, 'split by', [['random', 'random blocks'], ['spatial', 'spatial blocks']], 'random', (v) => { st.mode = v; rebuild(); });
    slider(ctl, 'buffer width (train blocks removed next to held-out)', 0, 2, 1, 0, (v) => v + (v === 1 ? ' block' : ' blocks'), (v) => { st.buffer = v; rebuild(); });
    const bRow = el('div', 'row');
    const reseed = el('button', 'btn small', 'new random seed (now 7)');
    reseed.onclick = () => { st.seed++; reseed.textContent = 'new random seed (now ' + st.seed + ')'; rebuild(); };
    bRow.appendChild(reseed);
    ctl.appendChild(bRow);
    const ro = el('div', 'readout');
    ctl.appendChild(ro);
    const T = transport(ctl, 1);
    rebuild();

    const tgt = (s, C) => AI.rgb(s === 'train' ? C.blue : s === 'val' ? C.orange : s === 'test' ? C.green : C.bg3);

    function draw() {
      const C = AI.col(), c = cv, g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const side = Math.min(c.w, c.h - 6), cs = side / NB, ox = (c.w - side) / 2, oy = 4, gap = 6;
      for (let i = 0; i < NB * NB; i++) {
        const r = Math.floor(i / NB), cc = i % NB;
        const x = ox + cc * cs + gap / 2, y = oy + r * cs + gap / 2, w = cs - gap;
        if (status[i] === 'buffer') {
          g.fillStyle = C.bg3; g.fillRect(x, y, w, w);
          g.strokeStyle = C.muted; g.lineWidth = 1;
          g.beginPath(); for (let t = -w; t < w; t += 8) { g.moveTo(x + t, y + w); g.lineTo(x + t + w, y); } g.stroke();
        } else {
          g.fillStyle = rgbaA(cur[i], 0.85); g.fillRect(x, y, w, w);
        }
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x, y, w, w);
      }
      const edge = (pr) => {
        const { o, r, c: cc } = pr;
        if (o === 'h') return [ox + (cc + 1) * cs, oy + r * cs + gap, ox + (cc + 1) * cs, oy + (r + 1) * cs - gap];
        return [ox + cc * cs + gap, oy + (r + 1) * cs, ox + (cc + 1) * cs - gap, oy + (r + 1) * cs];
      };
      let found = 0;
      for (let i = 0; i < Math.min(p, pairs.length); i++) {
        const pr = pairs[i];
        if (pr.type === 'leak') found++;
        if (pr.type === 'buffer') continue;
        const [x1, y1, x2, y2] = edge(pr);
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2);
        g.strokeStyle = pr.type === 'leak' ? C.red : pr.type === 'heldout' ? C.orange : C.muted;
        g.lineWidth = pr.type === 'leak' ? 5 : pr.type === 'heldout' ? 3 : 1;
        g.stroke();
      }
      if (p < pairs.length) {
        const pr = pairs[p], [x1, y1, x2, y2] = edge(pr);
        g.setLineDash([4, 3]); g.strokeStyle = C.fg; g.lineWidth = 2;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.setLineDash([]);
      }
      const names = { train: 'train', val: 'validation', test: 'test', buffer: 'buffer (removed)' };
      if (p < pairs.length) {
        const pr = pairs[p];
        const r1 = Math.floor(pr.a / NB), c1 = pr.a % NB, r2 = Math.floor(pr.b / NB), c2 = pr.b % NB;
        const head = `Pair ${p + 1} of 60: block (${r1}, ${c1}) is ${names[pr.sa]}, its neighbour (${r2}, ${c2}) is ${names[pr.sb]}. `;
        const body = pr.type === 'leak' ? 'They touch, so their pixels are alike: a leak.'
          : pr.type === 'same' ? 'Same split, so no leak.'
          : pr.type === 'heldout' ? 'Two held-out blocks touch. No training data crosses this border, but tuning on one and testing on the other is still optimistic.'
          : 'One of them is a buffer block, removed from training, so it cannot leak.';
        setHTML(cap, head + body + ` Leaks found so far: ${found}.`);
      } else {
        setHTML(cap, `All 60 neighbour pairs checked: ${sum.leak} leak (training next to held-out). Training blocks left: ${sum.train} of 36${sum.buffer ? ', with ' + sum.buffer + ' buffered' : ''}.`);
      }
      setHTML(ro, `split: ${st.mode}${st.mode === 'random' ? ' (seed ' + st.seed + ')' : ''} · train ${sum.train} · val ${sum.val} · test ${sum.test} · buffered ${sum.buffer}\n` +
        `leaking pairs (train ↔ held-out): ${sum.leak} of 60 · held-out ↔ held-out: ${sum.heldPairs}`);
    }

    AI.loop(root, (dt) => {
      const C = AI.col();
      if (!cur) cur = status.map((s) => tgt(s, C).slice());
      if (T.restartReq) { T.restartReq = false; p = 0; hold = 0; dirty = true; }
      if (T.stepReq) { T.stepReq = 0; p = p >= pairs.length ? 0 : p + 1; dirty = true; }
      if (T.playing) {
        if (p < pairs.length) { const n = tick(dt, 3 * T.speed); if (n) { p = Math.min(pairs.length, p + n); dirty = true; } }
        else { hold += dt; if (hold > 2.5) { p = 0; hold = 0; dirty = true; } }
      }
      const f = 1 - Math.exp(-dt * 4 * T.speed);
      let moved = false;
      status.forEach((s, i) => {
        const t = tgt(s, C);
        for (let k = 0; k < 3; k++) { const dl = (t[k] - cur[i][k]) * f; if (Math.abs(dl) > 0.05) moved = true; cur[i][k] += dl; }
      });
      const key = [p, status.join(''), themeKey(), cv.w, cv.h].join('|');
      if (!dirty && !moved && key === last) return;
      last = key; dirty = false;
      draw();
    });
  };

  /* =====================================================================
   * 5. Geometric & band-aware augmentation (augmentation-geo)
   * ===================================================================== */
  AI.labs.augLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Geometric and band-aware augmentation', 'a 4 × 4 tile with north, a road and a house', '', '');
    const cv = canvasIn(view, 250), cap = capIn(view), sp = canvasIn(view, 130);
    const ro = el('div', 'readout');
    view.appendChild(ro);
    // grey-level image (4 × 4) and labels: 2 = road, 1 = house, 0 = background
    const IMG = [[0.9, 0.9, 0.9, 0.9], [0.3, 0.3, 0.3, 0.3], [0.3, 0.6, 0.6, 0.12], [0.3, 0.6, 0.6, 0.12]];
    const LAB = [[2, 2, 2, 2], [0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]];
    const st = { ops: [], labelsMove: true, georef: false, red: 0.10, nir: 0.40, k: 1.2, jr: 1.2, jn: 0.8 };
    const rot = (m) => m.map((_, r) => m.map((_, c) => m[m.length - 1 - c][r]));
    const fh = (m) => m.map((row) => row.slice().reverse());
    const fv = (m) => m.slice().reverse();
    const OP = { rot, fh, fv };
    const VEC = { rot: (v) => [-v[1], v[0]], fh: (v) => [-v[0], v[1]], fv: (v) => [v[0], -v[1]] };
    const LABEL = { rot: 'rotate 90° CW', fh: 'flip left–right', fv: 'flip top–bottom' };
    let bars = [0, 0, 0], last = '';

    const bRow = el('div', 'row');
    const mk = (t, fn) => { const b = el('button', 'btn small', t); b.onclick = fn; bRow.appendChild(b); };
    mk('Rotate 90° CW', () => st.ops.push('rot'));
    mk('Flip left–right', () => st.ops.push('fh'));
    mk('Flip top–bottom', () => st.ops.push('fv'));
    mk('Undo', () => st.ops.pop());
    mk('Reset', () => { st.ops = []; });
    ctl.appendChild(bRow);
    check(ctl, 'transform labels together with the image', true, (v) => { st.labelsMove = v; });
    check(ctl, 'update the georeference with each operation', false, (v) => { st.georef = v; });
    slider(ctl, 'red reflectance', 0.02, 0.5, 0.01, 0.10, (v) => v.toFixed(2), (v) => { st.red = v; });
    slider(ctl, 'near-infrared reflectance', 0.05, 0.8, 0.01, 0.40, (v) => v.toFixed(2), (v) => { st.nir = v; });
    slider(ctl, 'common brightness factor (all bands)', 0.5, 1.5, 0.05, 1.2, (v) => '× ' + v.toFixed(2), (v) => { st.k = v; });
    slider(ctl, 'independent jitter, red', 0.5, 1.5, 0.05, 1.2, (v) => '× ' + v.toFixed(2), (v) => { st.jr = v; });
    slider(ctl, 'independent jitter, near-infrared', 0.5, 1.5, 0.05, 0.8, (v) => '× ' + v.toFixed(2), (v) => { st.jn = v; });

    const nd = (n, r) => (n + r > 0 ? (n - r) / (n + r) : 0);
    const dirName = (v) => (Math.abs(v[0]) > Math.abs(v[1]) ? (v[0] > 0 ? 'to the right' : 'to the left') : (v[1] < 0 ? 'up' : 'down'));

    function draw() {
      const C = AI.col(), c = cv, g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const ops = st.ops;
      const data = ops.reduce((m, op) => OP[op](m), IMG);
      const labs = st.labelsMove ? ops.reduce((m, op) => OP[op](m), LAB) : LAB;
      const nv = ops.reduce((v, op) => VEC[op](v), [0, -1]);
      const mirrored = ops.filter((o) => o !== 'rot').length % 2 === 1;
      const side = Math.min(c.h - 20, c.w * 0.5), cell = side / 4;
      const ox = (c.w - side) / 2 + 36, oy = (c.h - side) / 2;
      for (let r = 0; r < 4; r++) for (let cc = 0; cc < 4; cc++) {
        const v = data[r][cc], x = ox + cc * cell, y = oy + r * cell;
        const grey = Math.round(255 * v);
        g.fillStyle = `rgb(${grey},${grey},${grey})`; g.fillRect(x, y, cell, cell);
        const lb = labs[r][cc];
        if (lb === 1) { g.fillStyle = rgba(C.orange, 0.45); g.fillRect(x, y, cell, cell); }
        if (lb === 2) { g.fillStyle = rgba(C.yellow, 0.5); g.fillRect(x, y, cell, cell); }
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x, y, cell, cell);
      }
      // two north arrows: where north really is (green, drawn first) and what the georeference says (blue, on top)
      const ax = ox - 22, ay = oy + side / 2, len = 20, arrowTo = (v) => [ax + v[0] * len, ay + v[1] * len];
      const truth = arrowTo(nv), geo = arrowTo(st.georef ? nv : [0, -1]);
      AI.arrow(g, ax, ay, truth[0], truth[1], C.green, 2);
      AI.arrow(g, ax, ay, geo[0], geo[1], C.blue, 2);
      g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'left';
      g.fillText('blue = georef N', 6, c.h - 6);
      g.fillStyle = C.green; g.fillText('green = real N', 6, c.h - 18);

      // spectral bars: NDVI for base, common brightness and independent jitter
      const base = nd(st.nir, st.red), scaled = nd(st.nir * st.k, st.red * st.k), jit = nd(st.nir * st.jn, st.red * st.jr);
      const targets = [base, scaled, jit];
      const names = ['base NDVI', 'brightness × k on both bands', 'independent jitter'];
      const s = sp, h = s.ctx;
      h.clearRect(0, 0, s.w, s.h);
      const zx = s.w * 0.58, half = s.w * 0.36, rowH = s.h / 3;
      h.strokeStyle = C.border; h.beginPath(); h.moveTo(zx, 4); h.lineTo(zx, s.h - 4); h.stroke();
      targets.forEach((t, i) => {
        const y = i * rowH + rowH * 0.35;
        h.fillStyle = C.muted; h.font = FONT; h.textAlign = 'left'; h.fillText(names[i], 6, y - 6);
        const bw = bars[i] * half;
        h.fillStyle = i === 2 ? rgba(C.orange, 0.8) : rgba(C.blue, 0.75);
        h.fillRect(bw >= 0 ? zx : zx + bw, y, Math.abs(bw), rowH * 0.36);
        h.fillStyle = C.fg; h.textAlign = bw >= 0 ? 'left' : 'right';
        h.fillText(F(bars[i], 3), bw >= 0 ? zx + bw + 4 : zx + bw - 4, y + rowH * 0.36 - 2);
      });

      // status line
      const issues = [];
      if (!st.labelsMove && ops.length) issues.push('the labels stayed in place, so they no longer sit on the pixels they describe');
      if (!st.georef && (nv[0] !== 0 || nv[1] !== -1)) issues.push('the georeference still says north is up, but north now points ' + dirName(nv));
      if (!st.georef && mirrored) issues.push('a flip mirrors east and west, which an unchanged georeference cannot describe');
      let msg;
      if (!ops.length) msg = '<b>Consistent.</b> The tile is in its original north-up frame: the road lies north of the house, and the shadow falls east, so the sun is in the west.';
      else if (issues.length) msg = '<b>Mismatch.</b> ' + issues.join('; ') + '.';
      else msg = '<b>Consistent.</b> ' + (st.georef ? 'The georeference was updated, so the tile is no longer north-up; store the new transform.' : 'Nothing has moved out of step.') + (mirrored && st.georef ? ' The mirror needs a negative pixel width in the transform.' : '');
      if (ops.some((o) => o !== 'rot')) msg += ' A flip also moves the shadow to the other side, which implies a different sun position.';
      setHTML(cap, msg + (ops.length ? ' <span class="m">Operations: ' + ops.map((o) => LABEL[o]).join(', then ') + '.</span>' : ''));
      setHTML(ro, `NDVI base ${F(base, 4)} · brightness × ${F(st.k, 2)} on both bands ${F(scaled, 4)} · independent jitter ${F(jit, 4)}\n` +
        `invariance: a common factor cancels in the ratio; independent band changes do not`);
    }

    AI.loop(root, (dt) => {
      const targets = [nd(st.nir, st.red), nd(st.nir * st.k, st.red * st.k), nd(st.nir * st.jn, st.red * st.jr)];
      const f = 1 - Math.exp(-dt * 4);
      let moved = false;
      for (let i = 0; i < 3; i++) { const d = (targets[i] - bars[i]) * f; if (Math.abs(d) > 1e-4) moved = true; bars[i] += d; }
      const key = JSON.stringify(st) + themeKey() + cv.w + ',' + cv.h + ',' + sp.w + ',' + sp.h;
      if (!moved && key === last) return;
      last = key;
      draw();
    });
  };

  /* =====================================================================
   * 6. Dataset manifest builder (dataset-manifests)
   * ===================================================================== */
  const bandsOf = (F) => F.bands.split(',').map((s) => s.trim()).filter(Boolean);
  const RULES = [
    ['name is set', (F) => F.name.trim() !== '', 'give the dataset a name'],
    ['version is MAJOR.MINOR.PATCH', (F) => /^\d+\.\d+\.\d+$/.test(F.version.trim()), 'for example 1.2.0'],
    ['created is YYYY-MM-DD', (F) => /^\d{4}-\d{2}-\d{2}$/.test(F.created.trim()), 'for example 2026-10-31'],
    ['crs is EPSG:<number>', (F) => /^EPSG:\d+$/.test(F.crs.trim()), 'for example EPSG:32636 (UTM zone 36N)'],
    ['resolution is a positive number of metres', (F) => Number(F.res) > 0, 'for example 10'],
    ['band_order has unique, non-empty names', (F) => { const b = bandsOf(F); return b.length > 0 && new Set(b).size === b.length; }, 'separate names with commas; no repeats'],
    ['source is an http(s) link', (F) => /^https?:\/\/\S+$/.test(F.source.trim()), 'start with https://'],
    ['licence is stated', (F) => F.licence.trim() !== '', 'state the licence of the images and of the labels'],
    ['normalisation was fitted on train only', (F) => F.fitted === 'train', 'statistics from other splits leak the test set'],
  ];

  AI.labs.manifestLab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Dataset manifest builder', 'fill the fields; the JSON updates and the checks say what is missing', '', '');
    const list = el('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:6px;font-size:13px;margin-top:10px';
    const status = el('div', 'readout');
    status.style.marginTop = '10px';
    const pv = el('div', 'readout');
    pv.style.cssText = 'margin-top:10px;max-height:340px;overflow:auto';
    view.appendChild(list); view.appendChild(status); view.appendChild(pv);
    const F = { name: 'coastal-wetlands-demo', version: '1.2.0', created: '2026-10-31', crs: 'EPSG:32636', res: '10', bands: 'B02, B03, B04, B08', dtype: 'uint16', nodata: '0', fitted: 'train', source: 'https://example.com/stac/items/demo-scene-001', licence: 'CC-BY-4.0' };
    text(ctl, 'name', F.name, (v) => { F.name = v; });
    text(ctl, 'version', F.version, (v) => { F.version = v; });
    text(ctl, 'created (YYYY-MM-DD)', F.created, (v) => { F.created = v; });
    text(ctl, 'crs', F.crs, (v) => { F.crs = v; });
    text(ctl, 'resolution (metres)', F.res, (v) => { F.res = v; });
    text(ctl, 'band_order (comma separated)', F.bands, (v) => { F.bands = v; });
    select(ctl, 'dtype', ['uint8', 'uint16', 'int16', 'float32'], F.dtype, (v) => { F.dtype = v; });
    text(ctl, 'nodata (blank for none)', F.nodata, (v) => { F.nodata = v; });
    select(ctl, 'normalisation fitted on', ['train', 'train+val', 'all data'], F.fitted, (v) => { F.fitted = v === 'all data' ? 'all' : v; });
    text(ctl, 'source (STAC item URL)', F.source, (v) => { F.source = v; });
    text(ctl, 'licence', F.licence, (v) => { F.licence = v; });

    const toObj = () => ({
      name: F.name.trim(), version: F.version.trim(), created: F.created.trim(), crs: F.crs.trim(),
      resolution_m: Number(F.res) > 0 ? Number(F.res) : null,
      band_order: bandsOf(F), dtype: F.dtype,
      nodata: F.nodata.trim() === '' ? null : Number(F.nodata),
      normalisation: { method: 'percentile', percentiles: [2, 98], fitted_on: F.fitted },
      sources: [{ url: F.source.trim() }],
      licence: F.licence.trim(),
    });

    const rows = RULES.map(([label]) => {
      const row = el('div');
      row.style.cssText = 'display:flex;gap:8px;align-items:flex-start';
      row.innerHTML = '<span style="width:12px;height:12px;border-radius:50%;flex:none;margin-top:3px"></span><span><b class="t"></b><br><span class="h" style="color:var(--muted);font-size:12px"></span></span>';
      list.appendChild(row);
      return { row, dot: row.firstChild, t: row.querySelector('.t'), h: row.querySelector('.h'), v: 0, label };
    });
    rows.forEach((r) => { r.t.textContent = r.label; });

    let last = '';
    AI.loop(root, (dt) => {
      const C = AI.col();
      const results = RULES.map(([, test]) => test(F));
      let moved = false;
      rows.forEach((r, i) => {
        const target = results[i] ? 1 : 0, d = (target - r.v) * (1 - Math.exp(-dt * 5));
        if (Math.abs(d) > 1e-3) moved = true;
        r.v += d;
        if (Math.abs(target - r.v) < 1e-3) r.v = target;
      });
      const key = JSON.stringify(F) + themeKey() + results.join('') + rows.map((r) => r.v.toFixed(3)).join(',');
      if (!moved && key === last) return;
      last = key;
      rows.forEach((r, i) => {
        const c = AI.rgb(C.red), g = AI.rgb(C.green);
        const col = `rgb(${c.map((x, k) => Math.round(x + (g[k] - x) * r.v)).join(',')})`;
        r.dot.style.background = col;
        setHTML(r.h, results[i] ? '' : RULES[i][2]);
      });
      const failing = results.filter((x) => !x).length;
      setHTML(status, failing
        ? `${failing} check${failing === 1 ? '' : 's'} failing. Fix the red rows; the JSON below is what the loader would read.`
        : `Valid manifest: all ${RULES.length} checks pass.`);
      setText(pv, JSON.stringify(toObj(), null, 2));
    });
  };
})();
