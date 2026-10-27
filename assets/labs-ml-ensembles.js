/* Labs for ML Module 4 (ensembles & boosting): mlBagging, mlForest, mlBoost, mlXgbGain, mlStacking.
 * All simulations are small, real implementations run in the browser (trees, bootstrap, boosting) - labelled "illustrative" where they stand in for production libraries. */
(function () {
  const R = 1.3, RES = 40;
  const $ = (s, r) => AI.$(s, r);

  /* ---------- shared helpers ---------- */
  const cellXY = (i, j) => [((i + 0.5) / RES) * 2 * R - R, R - ((j + 0.5) / RES) * 2 * R];
  const makeGrid = (fn) => { const g = new Float32Array(RES * RES); for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) { const [x, y] = cellXY(i, j); g[j * RES + i] = fn(x, y); } return g; };
  const paintGrid = (ctx, w, h, g) => ML.boundary(ctx, w, h, (x, y) => {
    const i = Math.min(RES - 1, Math.max(0, Math.floor(((x + R) / (2 * R)) * RES))), j = Math.min(RES - 1, Math.max(0, Math.floor(((R - y) / (2 * R)) * RES)));
    const p = g[j * RES + i]; return [1 - p, p];
  }, { range: R, res: RES });
  const sig = (z) => 1 / (1 + Math.exp(-z));

  /* classification CART (gini) with optional random feature subset per split */
  const gini = (c0, c1) => { const n = c0 + c1; return n ? 1 - (c0 * c0 + c1 * c1) / (n * n) : 0; };
  function growClf(X, y, idx, o, d = 0) {
    let c1 = 0; for (const i of idx) c1 += y[i];
    const n = idx.length, c0 = n - c1, imp = gini(c0, c1), node = { p: n ? c1 / n : 0.5 };
    if (d >= o.maxDepth || imp === 0 || n < 2 * (o.minLeaf || 1)) return node;
    const D = X[0].length; let feats;
    if (o.m && o.m < D) { feats = []; const pool = [...Array(D).keys()]; for (let k = 0; k < o.m; k++) feats.push(pool.splice(Math.floor(o.rng() * pool.length), 1)[0]); } else feats = [...Array(D).keys()];
    let best = null;
    for (const f of feats) {
      const s = [...idx].sort((a, b) => X[a][f] - X[b][f]); let l0 = 0, l1 = 0;
      for (let j = 0; j < n - 1; j++) {
        if (y[s[j]]) l1++; else l0++;
        if (X[s[j]][f] === X[s[j + 1]][f]) continue;
        const nl = j + 1, nr = n - nl, gain = imp - (nl * gini(l0, l1) + nr * gini(c0 - l0, c1 - l1)) / n;
        if (!best || gain > best.gain + 1e-12) best = { f, gain, t: (X[s[j]][f] + X[s[j + 1]][f]) / 2, at: nl, s };
      }
    }
    if (!best || best.gain <= 1e-12) return node;
    node.f = best.f; node.t = best.t; if (d === 0) node.feats = feats;
    if (o.imp) o.imp[best.f] += best.gain * n;
    node.l = growClf(X, y, best.s.slice(0, best.at), o, d + 1); node.r = growClf(X, y, best.s.slice(best.at), o, d + 1);
    return node;
  }
  const predClf = (node, p) => { while (node.f !== undefined) node = p[node.f] <= node.t ? node.l : node.r; return node.p; };

  /* regression CART (SSE) used by boosting; o.leaf(idx) gives a leaf value */
  function growReg(X, r, idx, o, d = 0) {
    const n = idx.length; let S = 0; for (const i of idx) S += r[i];
    const node = { v: o.leaf(idx), n };
    if (d >= o.maxDepth || n < 2) return node;
    let best = null; const D = X[0].length;
    for (let f = 0; f < D; f++) {
      const s = [...idx].sort((a, b) => X[a][f] - X[b][f]); let sl = 0;
      for (let j = 0; j < n - 1; j++) {
        sl += r[s[j]]; if (X[s[j]][f] === X[s[j + 1]][f]) continue;
        const nl = j + 1, nr = n - nl, gain = (sl * sl) / nl + ((S - sl) * (S - sl)) / nr - (S * S) / n;
        if (!best || gain > best.gain + 1e-12) best = { f, gain, t: (X[s[j]][f] + X[s[j + 1]][f]) / 2, at: nl, s };
      }
    }
    if (!best || best.gain <= 1e-12) return node;
    node.f = best.f; node.t = best.t;
    node.l = growReg(X, r, best.s.slice(0, best.at), o, d + 1); node.r = growReg(X, r, best.s.slice(best.at), o, d + 1);
    return node;
  }
  const predReg = (node, p) => { while (node.f !== undefined) node = p[node.f] <= node.t ? node.l : node.r; return node.v; };
  function splitsOf(node, box, out) { // box=[x0,x1,y0,y1] -> list of segments for drawing partitions
    if (node.f === undefined) return out;
    const [x0, x1, y0, y1] = box;
    if (node.f === 0) { out.push([node.t, y0, node.t, y1]); splitsOf(node.l, [x0, node.t, y0, y1], out); splitsOf(node.r, [node.t, x1, y0, y1], out); }
    else { out.push([x0, node.t, x1, node.t]); splitsOf(node.l, [x0, x1, y0, node.t], out); splitsOf(node.r, [x0, x1, node.t, y1], out); }
    return out;
  }

  /* small line plot on an AI.canvas */
  function plot(c, series, o) {
    const g = c.ctx, w = c.w, h = c.h, C = AI.col(), pl = 38, pr = 8, pt = 18, pb = 18;
    g.clearRect(0, 0, w, h);
    let ymin = o.ymin, ymax = o.ymax; const auto0 = ymin === undefined, auto1 = ymax === undefined;
    if (auto0) ymin = Infinity; if (auto1) ymax = -Infinity;
    series.forEach((s) => s.ys.forEach((v) => { if (v != null && isFinite(v)) { if (auto0) ymin = Math.min(ymin, v); if (auto1) ymax = Math.max(ymax, v); } }));
    if (!isFinite(ymin) || !isFinite(ymax)) { ymin = 0; ymax = 1; }
    if (ymax - ymin < 1e-9) ymax = ymin + 1;
    const pad = (ymax - ymin) * 0.08; if (auto0) ymin -= pad; if (auto1) ymax += pad;
    const nx = Math.max(2, o.xmax), X = (i) => pl + (i / (nx - 1)) * (w - pl - pr), Y = (v) => pt + (1 - (v - ymin) / (ymax - ymin)) * (h - pt - pb);
    g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(pl, pt); g.lineTo(pl, h - pb); g.lineTo(w - pr, h - pb); g.stroke();
    g.fillStyle = C.muted; g.font = '10px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'right';
    for (let k = 0; k <= 2; k++) { const v = ymin + ((ymax - ymin) * k) / 2; g.fillText(AI.fmt(v, 3), pl - 4, Y(v) + 3); g.globalAlpha = 0.25; g.beginPath(); g.moveTo(pl, Y(v)); g.lineTo(w - pr, Y(v)); g.stroke(); g.globalAlpha = 1; }
    g.textAlign = 'center'; g.fillText(o.xlabel || '', (pl + w - pr) / 2, h - 4);
    g.textAlign = 'left'; let lx = pl;
    series.forEach((s) => {
      g.strokeStyle = s.color; g.lineWidth = s.w || 2; g.setLineDash(s.dash || []); g.beginPath(); let st = false;
      s.ys.forEach((v, i) => { if (v == null || !isFinite(v)) return; st ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); st = true; }); g.stroke(); g.setLineDash([]);
      g.fillStyle = s.color; g.font = '11px sans-serif'; g.fillText(s.label || '', lx, 12); lx += g.measureText(s.label || '').width + 12;
    });
    if (o.marks) o.marks.forEach((m) => { const x = X(m.i); g.strokeStyle = m.color; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x, pt); g.lineTo(x, h - pb); g.stroke(); g.setLineDash([]); });
  }

  const SPEED = (def = 1) => `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.25" value="${def}"></label>`;
  const BTNS = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="newd">New data</button></div>`;

  /* ======================================================================== */
  /* Bagging lab                                                              */
  /* ======================================================================== */
  AI.labs.mlBagging = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bagging lab: a committee of trees', 'bootstrap sample → deep tree → average (illustrative, runs a real CART in your browser)',
      `<div style="display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:10px"><div><canvas id="main" style="height:320px"></canvas></div>
         <div><canvas id="one" style="height:150px"></canvas><canvas id="vr" style="height:160px"></canvas></div></div>
       <div class="caption" id="cap" style="min-height:3em"></div>`,
      `<label>Number of trees (max) <span class="v" id="nv"></span><input type="range" id="n" min="1" max="60" step="1" value="25"></label>
       <label>Data noise <span class="v" id="nzv"></span><input type="range" id="nz" min="0.05" max="0.45" step="0.01" value="0.22"></label>
       <label>Tree depth <span class="v" id="dv"></span><input type="range" id="dp" min="2" max="12" step="1" value="10"></label>
       ${SPEED(1)}${BTNS}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cm = AI.canvas(q('#main')), c1 = AI.canvas(q('#one')), cv = AI.canvas(q('#vr'));
    const N = 80, TEST = 300, cols = () => ML.COLORS(2);
    let seed = 3, data, test, trees = [], phase = 'idle', pt = 0, playing = true, dirty = true, cur = null, seq = [], counts = [];
    let shown = new Float32Array(RES * RES).fill(0.5), target = new Float32Array(RES * RES).fill(0.5), fade = 0;
    let sumTest = null, ens = [], single = [];
    const build = () => { data = ML.data('moons', N, seed, { noise: +q('#nz').value }); test = ML.data('moons', TEST, seed + 500, { noise: +q('#nz').value }); reset(); };
    const reset = () => { trees = []; phase = 'idle'; pt = 0; cur = null; counts = []; sumTest = new Float32Array(TEST); ens = []; single = []; shown.fill(0.5); target.fill(0.5); dirty = true; };
    const brier = (p) => { let s = 0; for (let i = 0; i < TEST; i++) s += (p[i] - test.y[i]) ** 2; return s / TEST; };
    const retarget = () => {
      const K = Math.min(trees.length, +q('#n').value); target.fill(0); if (!K) { target.fill(0.5); return; }
      for (let t = 0; t < K; t++) for (let k = 0; k < target.length; k++) target[k] += trees[t].grid[k] / K;
    };
    const startTree = () => {
      const r = ML.rng(seed * 1000 + trees.length * 7 + 1); seq = Array.from({ length: N }, () => Math.floor(r() * N));
      const idx = seq.slice(), node = growClf(data.X, data.y, idx, { maxDepth: +q('#dp').value, minLeaf: 1, rng: r });
      const grid = makeGrid((x, y) => predClf(node, [x, y])), tp = new Float32Array(TEST);
      for (let i = 0; i < TEST; i++) { tp[i] = predClf(node, test.X[i]); sumTest[i] += tp[i]; }
      const k = trees.length + 1, e = new Float32Array(TEST); for (let i = 0; i < TEST; i++) e[i] = sumTest[i] / k;
      cur = { node, grid, n: k, bs: brier(tp), eb: brier(e) }; phase = 'boot'; pt = 0; fade = 0; counts = new Array(N).fill(0);
    };
    const finishTree = () => { trees.push(cur); single.push(cur.bs); ens.push(cur.eb); retarget(); phase = 'idle'; pt = 0; };
    const advance = (dt) => {
      const sp = +q('#sp').value;
      if (phase === 'idle') { if (trees.length < +q('#n').value) startTree(); return; }
      if (phase === 'boot') { pt += (dt * sp) / 2.6; const d = Math.min(N, Math.floor(pt * N)); counts.fill(0); for (let i = 0; i < d; i++) counts[data.X.length ? seq[i] : 0]++; if (pt >= 1) { phase = 'fit'; pt = 0; } }
      else if (phase === 'fit') { pt += (dt * sp) / 1.2; if (pt >= 1) { phase = 'merge'; pt = 0; } }
      else if (phase === 'merge') { pt += (dt * sp) / 1.6; fade = Math.min(1, pt * 1.5); if (pt >= 1) finishTree(); }
      dirty = true;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; if (phase === 'idle') { if (trees.length < +q('#n').value) startTree(); } else { const o = { boot: 'fit', fit: 'merge', merge: null }[phase]; if (o) { phase = o; pt = 0; if (o === 'merge') fade = 0; } else finishTree(); } dirty = true; };
    q('#reset').onclick = reset; q('#newd').onclick = () => { seed += 17; build(); };
    ['#nz', '#dp'].forEach((s) => (q(s).onchange = build)); q('#n').oninput = () => { retarget(); dirty = true; };
    cm.onresize = c1.onresize = cv.onresize = () => (dirty = true);
    build();

    const drawMain = () => {
      const g = cm.ctx, w = cm.w, h = cm.h, C = AI.col(), m = ML.toPx(w, h, R), cs = cols();
      paintGrid(g, w, h, shown);
      const inb = phase === 'boot' || phase === 'fit' || phase === 'merge';
      data.X.forEach((p, i) => {
        const c = inb ? counts[i] : 1; g.globalAlpha = inb && c === 0 && (phase !== 'boot' || pt > 0.02) ? 0.25 : 1;
        g.fillStyle = cs[data.y[i]]; g.strokeStyle = C.fg; g.lineWidth = c > 1 ? 2 : 1; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 4 + (inb && c > 1 ? 1.6 * (c - 1) : 0), 0, 7); g.fill(); g.stroke();
      });
      g.globalAlpha = 1; g.fillStyle = C.fg; g.font = '12px sans-serif'; g.textAlign = 'left';
      g.fillText(`committee of ${Math.min(trees.length, +q('#n').value)} tree(s): averaged probability`, 8, 16);
    };
    const drawOne = () => {
      const g = c1.ctx, w = c1.w, h = c1.h, C = AI.col(), cs = cols(), m = ML.toPx(w, h, R);
      g.clearRect(0, 0, w, h); const t = cur && phase !== 'idle' ? cur : trees[trees.length - 1];
      if (!t) { g.fillStyle = C.muted; g.font = '12px sans-serif'; g.fillText('the newest tree appears here', 10, 20); return; }
      const showing = !(phase === 'boot') && !(phase === 'fit' && pt < 0.1);
      if (showing) { paintGrid(g, w, h, t.grid); }
      else { g.fillStyle = C.bg2; g.fillRect(0, 0, w, h); }
      data.X.forEach((p, i) => { g.fillStyle = cs[data.y[i]]; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 2.4, 0, 7); g.fill(); });
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.fillText(`tree ${t.n} alone ${showing ? '(jagged)' : '(not grown yet)'}`, 6, 13);
    };
    const drawVR = () => {
      const K = Math.min(trees.length, +q('#n').value), avgS = single.length ? single.reduce((a, b) => a + b, 0) / single.length : null, C = AI.col();
      plot(cv, [{ ys: ens.slice(0, K), color: C.green, label: 'committee' }, { ys: single.map(() => avgS), color: C.orange, dash: [4, 4], label: 'avg single tree', w: 1.5 }], { xmax: Math.max(+q('#n').value, 2), xlabel: 'trees → (Brier, fresh data)', ymin: 0 });
    };
    AI.loop(root, (dt) => {
      q('#nv').textContent = q('#n').value; q('#nzv').textContent = q('#nz').value; q('#dv').textContent = q('#dp').value; q('#spv').textContent = '×' + q('#sp').value;
      if (playing) advance(dt);
      let diff = 0; for (let k = 0; k < shown.length; k++) { const d = target[k] - shown[k]; if (Math.abs(d) > 0.002) { shown[k] += d * (1 - Math.exp(-dt * 3)); diff = 1; } else shown[k] = target[k]; }
      if (diff) dirty = true;
      if (!dirty) return; dirty = false;
      drawMain(); drawOne(); drawVR();
      const K = Math.min(trees.length, +q('#n').value), n = trees.length;
      const cap = { idle: n >= +q('#n').value ? `Done: ${n} trees. Lower or raise the slider, change noise/depth, or press Reset.` : 'Ready.', boot: `① Tree ${n + 1}: draw ${N} rows with replacement (${Math.floor(pt * N)}/${N}). Big dots were picked several times, faded dots not at all (out-of-bag: about 37%).`, fit: `② Grow a deep tree on this bootstrap sample. Alone it is jagged: it memorises quirks of its own sample.`, merge: `③ Average its probabilities into the committee (1/${n + 1} weight). Quirks that differ between trees cancel; shared structure stays.` };
      q('#cap').textContent = cap[phase];
      const bs = single.length ? single.reduce((a, b) => a + b, 0) / single.length : NaN, eb = K ? ens[K - 1] : NaN;
      q('#out').innerHTML = n ? `trees built        <span class="b">${n}</span>\nsingle-tree Brier  <span class="o">${AI.fmt(bs, 3)}</span>\ncommittee Brier    <span class="g">${AI.fmt(eb, 3)}</span>  (${K} trees)\nvariance removed   <span class="g">${AI.fmt(((bs - eb) / bs) * 100, 1)}%</span> of error` : 'press play';
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* Random forest lab                                                        */
  /* ======================================================================== */
  AI.labs.mlForest = function (root) {
    const NAMES = ['x1', 'x2', "x1'", 'n1', 'n2', 'n3'], D = 6, N = 200, TEST = 200;
    const { view, ctl } = AI.labShell(root, 'Random-forest lab: decorrelating trees', 'forest (m features per split) versus plain bagging (all 6) - illustrative real CART in the browser',
      `<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px"><div><canvas id="oob" style="height:230px"></canvas></div><div><canvas id="imp" style="height:230px"></canvas></div></div>
       <div class="caption" id="cap" style="min-height:3em"></div>
       <div class="caption">Features: x1, x2 = the two informative coordinates of a moons data set; x1' = a noisy copy of x1 (redundant, strongly correlated); n1-n3 = pure noise. 200 rows.</div>`,
      `<label>Features per split m <span class="v" id="mv"></span><input type="range" id="m" min="1" max="6" step="1" value="2"></label>
       <label>Number of trees <span class="v" id="nv"></span><input type="range" id="n" min="5" max="100" step="1" value="50"></label>
       ${SPEED(1)}${BTNS}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), co = AI.canvas(q('#oob')), ci = AI.canvas(q('#imp'));
    let seed = 5, X, y, TX, A, B, shown = 0, phase = 'idle', pt = 0, playing = true, dirty = true;
    const impShown = new Float64Array(D);
    const mkData = (n, s) => { const d = ML.data('moons', n, s, { noise: 0.25 }), r = ML.rng(s + 99); return { X: d.X.map((p) => [p[0], p[1], 0.9 * p[0] + 0.12 * ML.randn(r), 0.6 * ML.randn(r), 0.6 * ML.randn(r), 0.6 * ML.randn(r)]), y: d.y }; };
    const mk = (m) => ({ m, k: 0, oobS: new Float32Array(N), oobC: new Int16Array(N), imp: new Float64Array(D), curve: [], impHist: [], corrHist: [], tps: [], cs: 0, cn: 0, roots: [] });
    const pearson = (a, b) => { let ma = 0, mb = 0; for (let i = 0; i < TEST; i++) { ma += a[i]; mb += b[i]; } ma /= TEST; mb /= TEST; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < TEST; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return saa > 1e-9 && sbb > 1e-9 ? sab / Math.sqrt(saa * sbb) : null; };
    const grow = (F) => {
      const r = ML.rng(seed * 977 + F.k * 13 + F.m * 101 + 3), idx = Array.from({ length: N }, () => Math.floor(r() * N)), inb = new Uint8Array(N); idx.forEach((i) => (inb[i] = 1));
      const imp = new Float64Array(D), node = growClf(X, y, idx, { maxDepth: 12, minLeaf: 1, m: F.m, rng: r, imp });
      for (let f = 0; f < D; f++) F.imp[f] += imp[f];
      for (let i = 0; i < N; i++) if (!inb[i]) { F.oobS[i] += predClf(node, X[i]); F.oobC[i]++; }
      let wrong = 0, cnt = 0; for (let i = 0; i < N; i++) if (F.oobC[i]) { cnt++; if ((F.oobS[i] / F.oobC[i] > 0.5 ? 1 : 0) !== y[i]) wrong++; }
      const tp = new Float32Array(TEST); for (let i = 0; i < TEST; i++) tp[i] = predClf(node, TX[i]);
      for (const o of F.tps.slice(-8)) { const c = pearson(tp, o); if (c !== null) { F.cs += c; F.cn++; } }
      F.tps.push(tp); if (F.tps.length > 9) F.tps.shift();
      const tot = F.imp.reduce((a, b) => a + b, 0) || 1; F.impHist.push(Array.from(F.imp, (v) => v / tot));
      F.curve.push(cnt ? wrong / cnt : null); F.corrHist.push(F.cn ? F.cs / F.cn : null); F.roots.push({ feats: node.feats || [], f: node.f }); F.k++;
    };
    const reset = () => { const d = mkData(N, seed), t = mkData(TEST, seed + 700); X = d.X; y = d.y; TX = t.X; A = mk(+q('#m').value); B = mk(D); shown = 0; phase = 'idle'; pt = 0; impShown.fill(0); dirty = true; };
    const total = () => +q('#n').value;
    const next = () => { if (A.k < total()) { grow(A); grow(B); } phase = 'draw'; pt = 0; };
    const advance = (dt) => {
      const sp = +q('#sp').value;
      if (phase === 'idle') { if (shown < total()) next(); return; }
      const dur = { draw: 0.9, subset: 1.3, add: 0.9 }[phase]; pt += (dt * sp) / dur;
      if (pt >= 1) { pt = 0; if (phase === 'draw') phase = 'subset'; else if (phase === 'subset') phase = 'add'; else { shown = A.k; phase = 'idle'; } }
      dirty = true;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; if (phase === 'idle') { if (shown < total()) next(); } else { pt = 0; if (phase === 'draw') phase = 'subset'; else if (phase === 'subset') phase = 'add'; else { shown = A.k; phase = 'idle'; } } dirty = true; };
    q('#reset').onclick = reset; q('#newd').onclick = () => { seed += 13; reset(); }; q('#m').onchange = reset; q('#n').oninput = () => (dirty = true);
    co.onresize = ci.onresize = () => (dirty = true);
    reset();
    AI.loop(root, (dt) => {
      q('#mv').textContent = q('#m').value + (+q('#m').value === D ? ' (= bagging)' : ''); q('#nv').textContent = q('#n').value; q('#spv').textContent = '×' + q('#sp').value;
      if (playing) advance(dt);
      const tgt = shown ? A.impHist[shown - 1] : null; let moving = false;
      for (let f = 0; f < D; f++) { const t = tgt ? tgt[f] : 0, d = t - impShown[f]; if (Math.abs(d) > 0.0005) { impShown[f] += d * (1 - Math.exp(-dt * 4)); moving = true; } else impShown[f] = t; }
      if (moving) dirty = true;
      if (!dirty) return; dirty = false;
      const C = AI.col(), n = shown;
      plot(co, [{ ys: B.curve.slice(0, n), color: C.muted, label: 'bagging (m=6)', dash: [4, 3], w: 1.6 }, { ys: A.curve.slice(0, n), color: C.green, label: 'forest (m=' + A.m + ')' }], { xmax: total(), ymin: 0, xlabel: 'trees →  (out-of-bag error rate)' });
      const g = ci.ctx, w = ci.w, h = ci.h; g.clearRect(0, 0, w, h); g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText('importance, m=' + A.m + ' (dashed: bagging)', 8, 14);
      const rowH = (h - 26) / D, bx = 40, bw = w - bx - 50, scale = Math.max(0.4, ...impShown), rt = A.roots[shown], hot = phase === 'subset' && rt ? rt.feats : [], bimp = shown ? B.impHist[shown - 1] : null;
      for (let f = 0; f < D; f++) {
        const y0 = 24 + f * rowH, glow = hot.includes(f), chosen = phase === 'subset' && rt && rt.f === f && pt > 0.55;
        if (glow) { g.fillStyle = C.orange; g.globalAlpha = 0.2 + 0.1 * Math.sin(pt * 12); g.fillRect(2, y0, w - 4, rowH - 3); g.globalAlpha = 1; }
        g.fillStyle = f < 3 ? C.green : C.muted; g.fillText(NAMES[f], 8, y0 + rowH * 0.62);
        g.globalAlpha = 0.85; g.fillRect(bx, y0 + 3, bw * (impShown[f] / scale), rowH - 10); g.globalAlpha = 1;
        if (bimp) { g.strokeStyle = C.fg; g.globalAlpha = 0.5; g.setLineDash([3, 2]); g.strokeRect(bx, y0 + 3, bw * (bimp[f] / scale), rowH - 10); g.setLineDash([]); g.globalAlpha = 1; }
        g.fillStyle = C.fg; g.fillText(AI.fmt(impShown[f], 2), bx + bw * (impShown[f] / scale) + 4, y0 + rowH * 0.62);
        if (chosen) { g.fillStyle = C.orange; g.fillText('◀ root', w - 48, y0 + rowH * 0.62); }
      }
      const k = A.k, cap = { idle: shown >= total() ? `Done: ${shown} trees. Move m and see how decorrelation changes the out-of-bag error.` : 'Ready.', draw: `① Tree ${k}: bootstrap sample drawn (about 63% distinct rows in-bag; the other 37% are this tree's out-of-bag test set).`, subset: `② At the root only these ${A.m} of ${D} features are offered (highlighted); the tree must choose among them. Plain bagging would see all ${D} and almost always pick x1 or x1'.`, add: `③ The tree votes on its out-of-bag rows; the out-of-bag error and importance bars update.` };
      q('#cap').textContent = cap[phase];
      const f = (v) => (v == null ? '  -  ' : AI.fmt(v, 3)), n1 = Math.max(0, shown - 1);
      q('#out').innerHTML = shown ? `trees                       <span class="b">${shown}</span>\nOOB error, forest           <span class="g">${f(A.curve[n1])}</span>\nOOB error, bagging          <span class="m">${f(B.curve[n1])}</span>\ntree-tree corr rho, forest  <span class="g">${f(A.corrHist[n1])}</span>\ntree-tree corr rho, bagging <span class="m">${f(B.corrHist[n1])}</span>` : 'press play';
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* Gradient boosting lab: trees added one at a time                         */
  /* ======================================================================== */
  AI.labs.mlBoost = function (root) {
    const { view, ctl } = AI.labShell(root, 'Boosting lab: one tree at a time', 'each round fits a small tree to the residuals, then adds a fraction of it (illustrative real implementation)',
      `<canvas id="main" style="height:290px"></canvas><canvas id="aux" style="height:120px"></canvas><canvas id="loss" style="height:130px"></canvas>
       <div class="caption" id="cap" style="min-height:3em"></div>`,
      `<label>Problem <select id="mode"><option value="reg">regression (squared loss, residual y - F)</option><option value="clf">classification (log-loss, residual y - p)</option></select></label>
       <label>Learning rate η <span class="v" id="lrv"></span><input type="range" id="lr" min="0.05" max="1" step="0.05" value="0.3"></label>
       <label>Tree depth <span class="v" id="dv"></span><input type="range" id="dp" min="1" max="4" step="1" value="2"></label>
       <label>Data noise <span class="v" id="nzv"></span><input type="range" id="nz" min="0.1" max="0.8" step="0.05" value="0.4"></label>
       <label>Rounds (max) <span class="v" id="rv"></span><input type="range" id="rmax" min="1" max="150" step="1" value="60"></label>
       ${SPEED(1)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="slow" checked> Slow motion: show the 3 phases of every round (uncheck to fast-forward)</label>
       ${BTNS}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cm = AI.canvas(q('#main')), ca = AI.canvas(q('#aux')), cl = AI.canvas(q('#loss'));
    const GN = 161, gx = Array.from({ length: GN }, (_, i) => -3.2 + (6.4 * i) / (GN - 1)), tick = AI.ticker();
    let seed = 4, mode = 'reg', Xtr, ytr, Xva, yva, Ftr, Fva, Fg, tl, vl, rounds, phase = 'idle', pt = 0, cur = null, playing = true, dirty = true;
    const lr = () => +q('#lr').value;
    const loss = (F, y) => { let s = 0; for (let i = 0; i < y.length; i++) { if (mode === 'reg') s += (y[i] - F[i]) ** 2; else { const p = Math.min(1 - 1e-9, Math.max(1e-9, sig(F[i]))); s -= y[i] ? Math.log(p) : Math.log(1 - p); } } return s / y.length; };
    const build = () => {
      mode = q('#mode').value; const nz = +q('#nz').value;
      if (mode === 'reg') {
        const mk = (n, s) => { const r = ML.rng(s), xs = Array.from({ length: n }, (_, i) => -3 + (6 * (i + r() * 0.8)) / n), ys = xs.map((x) => 1.5 * Math.sin(1.2 * x) + nz * ML.randn(r)); return { X: xs.map((x) => [x]), y: ys }; };
        const a = mk(30, seed), b = mk(60, seed + 300); Xtr = a.X; ytr = a.y; Xva = b.X; yva = b.y;
      } else {
        const a = ML.data('moons', 100, seed, { noise: 0.1 + nz * 0.4 }), b = ML.data('moons', 200, seed + 300, { noise: 0.1 + nz * 0.4 }); Xtr = a.X; ytr = a.y; Xva = b.X; yva = b.y;
      }
      q('#aux').style.display = mode === 'reg' ? 'block' : 'none'; reset();
    };
    const reset = () => {
      let f0; if (mode === 'reg') f0 = ytr.reduce((a, b) => a + b, 0) / ytr.length; else { const p = ytr.reduce((a, b) => a + b, 0) / ytr.length; f0 = Math.log(p / (1 - p)); }
      Ftr = new Float64Array(ytr.length).fill(f0); Fva = new Float64Array(yva.length).fill(f0); Fg = new Float64Array(mode === 'reg' ? GN : RES * RES).fill(f0);
      tl = [loss(Ftr, ytr)]; vl = [loss(Fva, yva)]; rounds = 0; phase = 'idle'; pt = 0; cur = null; dirty = true;
    };
    const startRound = () => {
      const n = ytr.length, r = new Array(n), w = new Array(n);
      for (let i = 0; i < n; i++) { if (mode === 'reg') { r[i] = ytr[i] - Ftr[i]; w[i] = 1; } else { const p = sig(Ftr[i]); r[i] = ytr[i] - p; w[i] = p * (1 - p); } }
      const leaf = (idx) => { let a = 0, b = 0; for (const i of idx) { a += r[i]; b += w[i]; } return mode === 'reg' ? a / idx.length : a / (b + 1); };
      const node = growReg(Xtr, r, Xtr.map((_, i) => i), { maxDepth: +q('#dp').value, leaf }, 0);
      const hG = mode === 'reg' ? gx.map((x) => predReg(node, [x])) : Array.from(makeGrid((x, y) => predReg(node, [x, y])));
      cur = { node, r, hG, hTr: Xtr.map((p) => predReg(node, p)), hVa: Xva.map((p) => predReg(node, p)), seg: mode === 'clf' ? splitsOf(node, [-R, R, -R, R], []) : null };
      phase = 'res'; pt = 0;
    };
    const commit = () => {
      const e = lr(); for (let i = 0; i < Ftr.length; i++) Ftr[i] += e * cur.hTr[i]; for (let i = 0; i < Fva.length; i++) Fva[i] += e * cur.hVa[i]; for (let i = 0; i < Fg.length; i++) Fg[i] += e * cur.hG[i];
      rounds++; tl.push(loss(Ftr, ytr)); vl.push(loss(Fva, yva)); phase = 'idle'; pt = 0; cur = null;
    };
    const advance = (dt) => {
      const sp = +q('#sp').value, max = +q('#rmax').value;
      if (!q('#slow').checked) { if (phase !== 'idle') { commit(); } const n = tick(dt, 4 * sp); for (let i = 0; i < n && rounds < max; i++) { startRound(); commit(); } dirty = true; return; }
      if (phase === 'idle') { if (rounds < max) startRound(); return; }
      pt += (dt * sp) / 1.5; if (pt >= 1) { pt = 0; if (phase === 'res') phase = 'fit'; else if (phase === 'fit') phase = 'upd'; else commit(); }
      dirty = true;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; if (phase === 'idle') { if (rounds < +q('#rmax').value) startRound(); } else { pt = 0; if (phase === 'res') phase = 'fit'; else if (phase === 'fit') phase = 'upd'; else commit(); } dirty = true; };
    q('#reset').onclick = reset; q('#newd').onclick = () => { seed += 9; build(); };
    ['#mode', '#nz'].forEach((s) => (q(s).onchange = build)); q('#dp').onchange = reset; q('#lr').onchange = reset; q('#rmax').oninput = () => (dirty = true);
    cm.onresize = ca.onresize = cl.onresize = () => (dirty = true);
    build();

    const draw = () => {
      const C = AI.col(), g = cm.ctx, e = lr(), t = phase === 'upd' ? AI.smooth(pt) : 0;
      g.clearRect(0, 0, cm.w, cm.h);
      if (mode === 'reg') {
        const v = AI.view(cm, -3.2, 3.2, -3, 3); AI.axes(g, v);
        const Fcur = (i) => Fg[i] + (cur ? e * cur.hG[i] * t : 0);
        if (cur && (phase === 'res' || phase === 'fit')) {
          g.strokeStyle = C.red; g.lineWidth = 1.6; g.globalAlpha = phase === 'res' ? AI.smooth(Math.min(1, pt * 2)) : 0.6;
          Xtr.forEach((p, i) => { g.beginPath(); g.moveTo(v.X(p[0]), v.Y(ytr[i])); g.lineTo(v.X(p[0]), v.Y(Ftr[i])); g.stroke(); }); g.globalAlpha = 1;
        }
        g.strokeStyle = C.orange; g.lineWidth = 2.6; g.beginPath(); gx.forEach((x, i) => { const yy = AI.clamp(Fcur(i), -4, 4); i ? g.lineTo(v.X(x), v.Y(yy)) : g.moveTo(v.X(x), v.Y(yy)); }); g.stroke();
        g.strokeStyle = C.muted; g.lineWidth = 1; g.setLineDash([2, 3]); g.beginPath(); gx.forEach((x, i) => { const yy = 1.5 * Math.sin(1.2 * x); i ? g.lineTo(v.X(x), v.Y(yy)) : g.moveTo(v.X(x), v.Y(yy)); }); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.blue; Xtr.forEach((p, i) => { g.beginPath(); g.arc(v.X(p[0]), v.Y(ytr[i]), 4, 0, 7); g.fill(); });
        g.strokeStyle = C.muted; g.globalAlpha = 0.5; Xva.forEach((p, i) => { g.beginPath(); g.arc(v.X(p[0]), v.Y(yva[i]), 2.2, 0, 7); g.stroke(); }); g.globalAlpha = 1;
        g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.blue; g.fillText('● train', v.pad.l + 6, v.pad.t + 13); g.fillStyle = C.muted; g.fillText('○ validation   ┄ true curve', v.pad.l + 56, v.pad.t + 13); g.fillStyle = C.orange; g.fillText('— model F(x)', v.pad.l + 220, v.pad.t + 13); g.fillStyle = C.red; g.fillText('| residual', v.pad.l + 305, v.pad.t + 13);
        // residual panel
        const g2 = ca.ctx; g2.clearRect(0, 0, ca.w, ca.h); const v2 = AI.view(ca, -3.2, 3.2, -2, 2, { l: 40, r: 12, t: 14, b: 20 }); AI.axes(g2, v2);
        const res = cur ? cur.r : ytr.map((y, i) => y - Ftr[i]); g2.fillStyle = C.red; res.forEach((r, i) => { g2.beginPath(); g2.arc(v2.X(Xtr[i][0]), v2.Y(AI.clamp(r, -2, 2)), 3, 0, 7); g2.fill(); });
        if (cur && (phase === 'fit' || phase === 'upd')) { g2.strokeStyle = C.green; g2.lineWidth = 2.6; g2.globalAlpha = phase === 'fit' ? AI.smooth(Math.min(1, pt * 1.5)) : 1; g2.beginPath(); gx.forEach((x, i) => { const yy = AI.clamp(cur.hG[i], -2, 2); i ? g2.lineTo(v2.X(x), v2.Y(yy)) : g2.moveTo(v2.X(x), v2.Y(yy)); }); g2.stroke(); g2.globalAlpha = 1; }
        g2.font = '11px sans-serif'; g2.textAlign = 'left'; g2.fillStyle = C.red; g2.fillText('● residuals y - F (what is still wrong)', v2.pad.l + 6, 11); g2.fillStyle = C.green; g2.fillText('— new tree h(x), fitted to them', v2.pad.l + 230, 11);
      } else {
        const pg = new Float32Array(Fg.length); for (let k = 0; k < pg.length; k++) pg[k] = sig(Fg[k] + (cur ? e * cur.hG[k] * t : 0));
        paintGrid(g, cm.w, cm.h, pg);
        const m = ML.toPx(cm.w, cm.h, R), cs = ML.COLORS(2), showRes = cur && (phase === 'res' || phase === 'fit');
        Xtr.forEach((p, i) => { const pr = sig(Ftr[i]), rr = Math.abs(ytr[i] - pr); g.fillStyle = cs[ytr[i]]; g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), showRes ? 2.5 + 7 * rr : 4, 0, 7); g.fill(); g.stroke(); });
        if (cur && phase !== 'res') { g.strokeStyle = C.fg; g.lineWidth = 1.8; g.setLineDash([6, 4]); g.globalAlpha = phase === 'fit' ? AI.smooth(Math.min(1, pt * 1.5)) : 0.5; cur.seg.forEach((s) => { g.beginPath(); g.moveTo(m.X(s[0]), m.Y(s[1])); g.lineTo(m.X(s[2]), m.Y(s[3])); g.stroke(); }); g.setLineDash([]); g.globalAlpha = 1; }
        g.fillStyle = C.fg; g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillText(showRes ? 'dot size = residual |y - p| (how wrong the current model is)' : 'colour = current p(class 1)', 8, 16);
      }
      const iBest = vl.indexOf(Math.min(...vl));
      plot(cl, [{ ys: tl, color: C.blue, label: 'train loss' }, { ys: vl, color: C.orange, label: 'validation loss' }], { xmax: Math.max(10, +q('#rmax').value + 1), xlabel: 'rounds →  (' + (mode === 'reg' ? 'MSE' : 'log-loss') + ')', ymin: 0, marks: iBest > 0 ? [{ i: iBest, color: C.green }] : [] });
      const k = rounds + 1, cap = {
        idle: rounds >= +q('#rmax').value ? `Finished ${rounds} rounds. Compare the two loss curves: when validation turns up while training keeps falling, extra rounds are fitting noise (overfitting).` : 'Ready.',
        res: mode === 'reg' ? `① Round ${k}: the orange curve is the current prediction F. Red vertical segments are the residuals y - F: exactly what the next tree must explain.` : `① Round ${k}: for log-loss the residual is y - p. Big dots are points the current model gets most wrong.`,
        fit: mode === 'reg' ? `② Fit a depth-${q('#dp').value} tree to the residuals (green step function, lower panel). It is crude on purpose: a weak learner.` : `② Fit a depth-${q('#dp').value} tree to y - p (dashed lines are its splits); leaf values use a Newton step (sum residual / (sum p(1-p) + 1)).`,
        upd: `③ Update F ← F + η·h with η = ${AI.fmt(e, 2)}: a small, safe step. Loss after this round is plotted, then the next round starts.`,
      };
      q('#cap').textContent = cap[phase];
      const f = (a) => AI.fmt(a[a.length - 1], 3);
      q('#out').innerHTML = `rounds (trees)   <span class="b">${rounds}</span>\ntrain loss      <span class="b">${f(tl)}</span>\nvalidation loss <span class="o">${f(vl)}</span>\nbest validation <span class="g">${AI.fmt(Math.min(...vl), 3)}</span> at round <span class="g">${iBest}</span>${rounds > iBest + 8 ? '\n<span class="r">validation is rising: overfitting</span>' : ''}`;
    };
    AI.loop(root, (dt) => {
      q('#lrv').textContent = AI.fmt(lr(), 2); q('#dv').textContent = q('#dp').value; q('#nzv').textContent = q('#nz').value; q('#rv').textContent = q('#rmax').value; q('#spv').textContent = q('#slow').checked ? '×' + q('#sp').value : 4 * q('#sp').value + ' rounds / s';
      if (playing) advance(dt);
      if (!dirty) return; dirty = false; draw();
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* XGBoost split-gain lab                                                   */
  /* ======================================================================== */
  AI.labs.mlXgbGain = function (root) {
    const PRESETS = {
      'Squared loss at F0 = 7.5 (the lesson)': { g: [4.5, 2.5, 3.5, -2.5, -4.5, -3.5], h: [1, 1, 1, 1, 1, 1] },
      'Log-loss at p = 0.5, y = 0,1,0,0,1,1': { g: [0.5, -0.5, 0.5, 0.5, -0.5, -0.5], h: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25] },
      'One outlier row': { g: [0.2, -0.1, 0.3, -0.2, 0.1, 8], h: [1, 1, 1, 1, 1, 1] },
      'Unequal hessians (8 rows)': { g: [-2, -1, -1.5, 1, 2, 1.5, 0.5, -0.5], h: [1, 0.2, 0.9, 0.4, 1.2, 0.3, 0.8, 0.2] },
    };
    const { view, ctl } = AI.labShell(root, 'XGBoost split-gain lab: price every split', 'edit gradients g and hessians h, then change λ and γ',
      `<canvas id="cv" style="height:340px"></canvas><div class="sub" style="overflow-x:auto"><table class="tbl" id="tbl"></table></div>
       <div class="readout" id="fm" style="margin-top:8px"></div><div class="caption" id="cap" style="min-height:3em"></div>`,
      `<label>Example <select id="pre">${Object.keys(PRESETS).map((k) => `<option>${k}</option>`).join('')}</select></label>
       <label>λ (L2 on leaf weights) <span class="v" id="lv"></span><input type="range" id="lam" min="0" max="10" step="0.1" value="1"></label>
       <label>γ (cost of a new leaf) <span class="v" id="gv"></span><input type="range" id="gam" min="0" max="40" step="0.5" value="0"></label>
       ${SPEED(1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="replay">Replay sweep</button></div>
       <div class="caption">Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ. Leaf weight w = −G/(H+λ). Edit any number in the table.</div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    let g = [], h = [], cand = [], best = -1, cp = 0, playing = true, dirty = true;
    const load = () => { const p = PRESETS[q('#pre').value]; g = p.g.slice(); h = p.h.slice(); buildTable(); recompute(true); };
    const buildTable = () => {
      const t = q('#tbl'); t.innerHTML = `<tr><th>row</th>${g.map((_, i) => `<th>${i + 1}</th>`).join('')}</tr><tr><td>g</td>${g.map((v, i) => `<td><input type="number" step="0.1" data-k="g" data-i="${i}" value="${v}" style="width:64px;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:3px 5px"></td>`).join('')}</tr><tr><td>h</td>${h.map((v, i) => `<td><input type="number" step="0.05" min="0" data-k="h" data-i="${i}" value="${v}" style="width:64px;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:3px 5px"></td>`).join('')}</tr>`;
      t.querySelectorAll('input').forEach((inp) => (inp.oninput = () => { const v = parseFloat(inp.value); if (!isFinite(v)) return; (inp.dataset.k === 'g' ? g : h)[+inp.dataset.i] = Math.max(inp.dataset.k === 'h' ? 0 : -1e9, v); recompute(true); }));
    };
    const recompute = (finish) => {
      const lam = +q('#lam').value, gam = +q('#gam').value, n = g.length; let G = 0, H = 0; g.forEach((v) => (G += v)); h.forEach((v) => (H += v));
      const parent = (G * G) / (H + lam); cand = []; let GL = 0, HL = 0;
      for (let k = 0; k < n - 1; k++) { GL += g[k]; HL += h[k]; const GR = G - GL, HR = H - HL, a = (GL * GL) / (HL + lam), b = (GR * GR) / (HR + lam); cand.push({ k, GL, HL, GR, HR, a, b, parent, gain: 0.5 * (a + b - parent) - gam, wL: -GL / (HL + lam), wR: -GR / (HR + lam) }); }
      best = -1; cand.forEach((c, i) => { if (best < 0 || c.gain > cand[best].gain + 1e-12) best = i; });
      if (finish) cp = cand.length; dirty = true;
    };
    q('#pre').onchange = load; ['#lam', '#gam'].forEach((s) => (q(s).oninput = () => recompute(true)));
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#replay').onclick = () => { cp = 0; playing = true; q('#play').textContent = '⏸ Pause'; dirty = true; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; cp = Math.min(cand.length, Math.floor(cp) + 1); dirty = true; };
    cv.onresize = () => (dirty = true);
    load(); cp = 0;
    AI.loop(root, (dt) => {
      q('#lv').textContent = q('#lam').value; q('#gv').textContent = q('#gam').value; q('#spv').textContent = '×' + q('#sp').value;
      if (playing && cp < cand.length) { cp = Math.min(cand.length, cp + (dt * +q('#sp').value) / 1.6); dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), ctx = cv.ctx, w = cv.w, hh = cv.h, n = g.length; ctx.clearRect(0, 0, w, hh);
      const left = 36, cw = (w - left - 10) / n, topH = 150, gmax = Math.max(1e-6, ...g.map(Math.abs)), cur = cand[Math.min(cand.length - 1, Math.floor(cp))], done = cp >= cand.length;
      const shownK = done ? (best >= 0 ? cand[best].k : -1) : cur.k;
      // upper: rows with gradients
      ctx.font = '11px ui-monospace,Menlo,Consolas,monospace'; ctx.textAlign = 'center';
      for (let i = 0; i < n; i++) {
        const x0 = left + i * cw; if (shownK >= 0) { ctx.fillStyle = i <= shownK ? C.blue : C.orange; ctx.globalAlpha = 0.12; ctx.fillRect(x0 + 1, 8, cw - 2, topH); ctx.globalAlpha = 1; }
        const base = 8 + topH / 2, bh = (g[i] / gmax) * (topH / 2 - 26); ctx.fillStyle = g[i] >= 0 ? C.red : C.green; ctx.fillRect(x0 + cw * 0.25, Math.min(base, base - bh), cw * 0.5, Math.abs(bh));
        ctx.fillStyle = C.fg; ctx.fillText(AI.fmt(g[i], 2), x0 + cw / 2, g[i] >= 0 ? base - bh - 3 : base - bh + 12); ctx.fillStyle = C.muted; ctx.fillText('h=' + AI.fmt(h[i], 2), x0 + cw / 2, topH + 4);
      }
      ctx.strokeStyle = C.muted; ctx.beginPath(); ctx.moveTo(left, 8 + topH / 2); ctx.lineTo(w - 10, 8 + topH / 2); ctx.stroke();
      ctx.textAlign = 'left'; ctx.fillStyle = C.muted; ctx.fillText('g', 6, 8 + topH / 2 + 4);
      if (shownK >= 0) { const sx = left + (shownK + 1) * cw; ctx.strokeStyle = done ? C.green : C.fg; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(sx, 4); ctx.lineTo(sx, topH + 8); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; }
      // lower: gain bars
      const y0 = topH + 40, bh0 = hh - y0 - 26, gm = Math.max(1e-6, ...cand.map((c) => Math.abs(c.gain))), zero = cand.some((c) => c.gain < 0) ? y0 + bh0 * 0.65 : y0 + bh0, up = zero - y0 - 14, dn = y0 + bh0 - zero;
      ctx.fillStyle = C.fg; ctx.textAlign = 'left'; ctx.font = '12px sans-serif'; ctx.fillText('gain of each candidate threshold (between rows)', left, y0 - 8);
      ctx.strokeStyle = C.border; ctx.beginPath(); ctx.moveTo(left, zero); ctx.lineTo(w - 10, zero); ctx.stroke(); ctx.font = '11px ui-monospace,Menlo,Consolas,monospace'; ctx.textAlign = 'center';
      cand.forEach((c, j) => {
        const f = j < Math.floor(cp) ? 1 : j === Math.floor(cp) && !done ? AI.smooth(cp - j) : 0; if (f === 0) return;
        const v = c.gain * f, x = left + (c.k + 1) * cw, bhh = (v / gm) * (v >= 0 ? up : dn);
        ctx.fillStyle = done && j === best ? (c.gain > 0 ? C.green : C.red) : c.gain < 0 ? C.red : C.blue; ctx.globalAlpha = done && j !== best ? 0.55 : 1; ctx.fillRect(x - cw * 0.3, v >= 0 ? zero - bhh : zero, cw * 0.6, Math.abs(bhh)); ctx.globalAlpha = 1;
        ctx.fillStyle = C.fg; ctx.fillText(AI.fmt(c.gain * f, 2), x, v >= 0 ? zero - bhh - 4 : zero - bhh + 13);
        ctx.fillStyle = C.muted; ctx.fillText('x<' + (c.k + 1.5), x, y0 + bh0 + 16);
      });
      // readout
      const c = done ? cand[best] : cur, lam = +q('#lam').value, gam = +q('#gam').value, f = (v) => AI.fmt(v, 4);
      if (c) q('#fm').innerHTML = `${done ? '<span class="g">best split</span>' : '<span class="b">evaluating</span>'}  x &lt; ${c.k + 1.5}\nG_L=${f(c.GL)}  H_L=${f(c.HL)}   G_R=${f(c.GR)}  H_R=${f(c.HR)}   G=${f(c.GL + c.GR)}  H=${f(c.HL + c.HR)}\nG_L²/(H_L+λ) = ${f(c.a)}    G_R²/(H_R+λ) = ${f(c.b)}    G²/(H+λ) = ${f(c.parent)}\nGain = ½[${f(c.a)} + ${f(c.b)} − ${f(c.parent)}] − ${gam} = <span class="${c.gain > 0 ? 'g' : 'r'}">${f(c.gain)}</span>\nleaf weights: w_L = −G_L/(H_L+λ) = <span class="o">${f(c.wL)}</span>,  w_R = <span class="o">${f(c.wR)}</span>`;
      q('#cap').textContent = !c ? '' : done ? (c.gain > 0 ? `Best split x < ${c.k + 1.5} has gain ${f(c.gain)} > 0: the tree splits, with leaf weights ${f(c.wL)} (left) and ${f(c.wR)} (right).` : `Best gain is ${f(c.gain)} ≤ 0: after paying γ = ${gam} the split does not earn its keep, so XGBoost keeps a single leaf with weight ${f(-(c.GL + c.GR) / (c.HL + c.HR + lam))}.`) : `Sweep: slide the threshold across the rows. Left rows (blue) and right rows (orange) each contribute G²/(H+λ); the bar shows the gain.`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* Stacking lab                                                             */
  /* ======================================================================== */
  AI.labs.mlStacking = function (root) {
    const n = 20, K = 5, MN = ['A linear', 'B 1-NN', 'C tree d=2'];
    const { view, ctl } = AI.labShell(root, 'Stacking lab: out-of-fold predictions feed a meta-model', 'folds rotate; every row is predicted by models that never saw it (illustrative)',
      `<canvas id="flow" style="height:240px"></canvas><div class="sub" style="overflow-x:auto"><table class="tbl" id="tbl"></table></div>
       <canvas id="fit" style="height:190px"></canvas><div class="caption" id="cap" style="min-height:3em"></div>`,
      `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="leak"> Leaky mode: base models predict rows they were trained on</label>
       ${SPEED(1)}${BTNS}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cf = AI.canvas(q('#flow')), cx = AI.canvas(q('#fit'));
    let seed = 2, xs, ys, X, fold, oof, lk, wH, wL, fullF, testX, testY, fi, phase, pt, playing = true, dirty = true, tkey = '';
    const solve = (A, b) => { const m = b.length, M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < m; i++) { let p = i; for (let r = i + 1; r < m; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; for (let r = 0; r < m; r++) if (r !== i) { const f = M[r][i] / M[i][i]; for (let c = i; c <= m; c++) M[r][c] -= f * M[i][c]; } } return M.map((r, i) => r[m] / r[i]); };
    const fitW = (Z) => { // non-negative ridge via active set, Z = [A[],B[],C[]]
      let act = [0, 1, 2], w = [0, 0, 0];
      for (let it = 0; it < 3; it++) {
        const A = act.map((a) => act.map((b) => Z[a].reduce((s, v, i) => s + v * Z[b][i], 0) + (a === b ? 0.05 : 0))), b = act.map((a) => Z[a].reduce((s, v, i) => s + v * ys[i], 0)), s = solve(A, b);
        const neg = s.findIndex((v) => v < 0); if (neg >= 0 && act.length > 1) { act.splice(neg, 1); continue; } w = [0, 0, 0]; act.forEach((a, i) => (w[a] = Math.max(0, s[i]))); break;
      }
      return w;
    };
    const fitters = [
      (tr) => { const mx = tr.reduce((s, i) => s + xs[i], 0) / tr.length, my = tr.reduce((s, i) => s + ys[i], 0) / tr.length; let sxy = 0, sxx = 0; tr.forEach((i) => { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }); const w = sxy / sxx; return (x) => my + w * (x - mx); },
      (tr) => (x) => { let b = tr[0]; tr.forEach((i) => { if (Math.abs(xs[i] - x) < Math.abs(xs[b] - x)) b = i; }); return ys[b]; },
      (tr) => { const node = growReg(X, ys, tr, { maxDepth: 2, leaf: (idx) => idx.reduce((s, i) => s + ys[i], 0) / idx.length }, 0); return (x) => predReg(node, [x]); },
    ];
    const build = () => {
      const r = ML.rng(seed); xs = Array.from({ length: n }, (_, i) => i + 1); const f = (x) => 0.6 * x + Math.sin(0.8 * x); ys = xs.map((x) => f(x) + 1.0 * ML.randn(r)); X = xs.map((x) => [x]);
      const perm = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      fold = new Array(n); perm.forEach((p, j) => (fold[p] = j % K));
      oof = [0, 1, 2].map(() => new Array(n)); lk = [0, 1, 2].map(() => new Array(n));
      fullF = fitters.map((ft) => ft([...Array(n).keys()]));
      for (let k = 0; k < K; k++) { const tr = [...Array(n).keys()].filter((i) => fold[i] !== k); fitters.forEach((ft, m) => { const fm = ft(tr); for (let i = 0; i < n; i++) if (fold[i] === k) oof[m][i] = fm(xs[i]); }); }
      for (let m = 0; m < 3; m++) for (let i = 0; i < n; i++) lk[m][i] = fullF[m](xs[i]);
      wH = fitW(oof); wL = fitW(lk);
      testX = Array.from({ length: 200 }, (_, i) => 1 + ((n - 1) * i) / 199); testY = testX.map((x) => f(x) + 1.0 * ML.randn(r));
      reset();
    };
    const reset = () => { fi = 0; phase = 'pick'; pt = 0; tkey = ''; dirty = true; };
    const leaky = () => q('#leak').checked, Z = () => (leaky() ? lk : oof), W = () => (leaky() ? wL : wH);
    const advance = (dt) => {
      const sp = +q('#sp').value; if (phase === 'done') return;
      const dur = { pick: 1.3, train: 1.3, predict: 1.8, meta: 1.8 }[phase]; pt += (dt * sp) / dur;
      if (pt >= 1) { pt = 0; if (phase === 'pick') phase = 'train'; else if (phase === 'train') phase = 'predict'; else if (phase === 'predict') { fi++; phase = fi >= K ? 'meta' : 'pick'; } else phase = 'done'; }
      dirty = true;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; pt = 0.999; advance(0.01); dirty = true; };
    q('#reset').onclick = reset; q('#newd').onclick = () => { seed += 5; build(); }; q('#leak').onchange = reset; cf.onresize = cx.onresize = () => (dirty = true);
    build();
    const testMse = (w) => { let s = 0; testX.forEach((x, i) => { const p = w[0] * fullF[0](x) + w[1] * fullF[1](x) + w[2] * fullF[2](x); s += (testY[i] - p) ** 2; }); return s / testX.length; };
    const baseMse = (m) => { let s = 0; testX.forEach((x, i) => (s += (testY[i] - fullF[m](x)) ** 2)); return s / testX.length; };
    AI.loop(root, (dt, time) => {
      q('#spv').textContent = '×' + q('#sp').value;
      if (playing) advance(dt);
      const lk0 = leaky();
      const filled = (i) => (fold[i] < fi || phase === 'meta' || phase === 'done' ? 1 : fold[i] === fi && phase === 'predict' ? 0.5 : 0), anim = phase === 'predict' || phase === 'meta';
      const key = [fi, phase, phase === 'predict' ? Math.floor(pt * 4) : 0, lk0].join('|');
      if (key !== tkey) {
        tkey = key; const Zv = Z(), rows = [];
        for (let i = 0; i < n; i++) { const cur = fold[i] === fi && phase !== 'meta' && phase !== 'done', ok = fold[i] < fi || phase === 'meta' || phase === 'done' || (fold[i] === fi && phase === 'predict' && pt > 0.3);
          const bg = cur ? 'background:rgba(240,136,62,.18)' : ''; rows.push(`<tr style="${bg}"><td>${xs[i]}</td><td>${fold[i] + 1}</td><td>${AI.fmt(ys[i], 2)}</td>${[0, 1, 2].map((m) => `<td>${ok ? AI.fmt(Zv[m][i], 2) : '·'}</td>`).join('')}</tr>`); }
        q('#tbl').innerHTML = `<tr><th>x</th><th>fold</th><th>y</th>${MN.map((m) => `<th>${m} ${lk0 ? '(in-sample)' : '(OOF)'}</th>`).join('')}</tr>${rows.join('')}`;
      }
      if (!dirty && !anim) return; dirty = false;
      const C = AI.col(), g = cf.ctx, w = cf.w, h = cf.h; g.clearRect(0, 0, w, h);
      const sq = Math.min(30, (w - 28) / n - 3), sx0 = 12, sy0 = 26, role = (i) => (fold[i] < fi && phase !== 'pick' ? 'done' : phase === 'meta' || phase === 'done' ? 'done' : fold[i] === fi ? (lk0 ? 'train' : 'held') : 'train');
      g.font = '11px sans-serif'; g.textAlign = 'center';
      for (let i = 0; i < n; i++) { const r = lk0 && phase !== 'meta' && phase !== 'done' ? 'train' : role(i), x0 = sx0 + i * (sq + 3); g.fillStyle = r === 'held' ? C.orange : r === 'train' ? C.blue : C.muted; g.globalAlpha = r === 'done' ? 0.35 : phase === 'pick' && r === 'held' ? 0.55 + 0.45 * Math.sin(pt * 9) ** 2 : 0.9; g.fillRect(x0, sy0, sq, sq); g.globalAlpha = 1; g.fillStyle = C.bg; g.fillText(xs[i], x0 + sq / 2, sy0 + sq / 2 + 4); g.fillStyle = C.muted; g.fillText(fold[i] + 1, x0 + sq / 2, sy0 + sq + 13); }
      g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText(phase === 'meta' || phase === 'done' ? 'all folds done' : `fold ${Math.min(K, fi + 1)} of ${K}: ${lk0 ? 'train on ALL rows' : 'held-out rows are orange'}`, sx0, 14);
      const mx = w * 0.46, mw = w * 0.17, my = [84, 128, 172], mh = 34, bx = w * 0.78, by = 104, bw = w * 0.2, bh = 64;
      const box = (x, y, ww, hh, txt, col, a) => { g.globalAlpha = a; g.fillStyle = C.bg2; g.strokeStyle = col; g.lineWidth = 2; g.fillRect(x, y, ww, hh); g.strokeRect(x, y, ww, hh); g.globalAlpha = 1; g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(txt, x + ww / 2, y + hh / 2 + 4); };
      MN.forEach((m, k) => { const active = phase === 'train' || phase === 'predict'; box(mx, my[k], mw, mh, m, active ? C.blue : C.border, active ? 1 : 0.7);
        const a = phase === 'train' ? AI.smooth(Math.min(1, pt * 2)) : phase === 'predict' ? 0.4 : 0; if (a > 0) { g.strokeStyle = C.blue; g.globalAlpha = a; g.lineWidth = 1.5; g.beginPath(); g.moveTo(sx0 + (n * (sq + 3)) / 2, sy0 + sq + 18); g.lineTo(mx, my[k] + mh / 2); g.stroke(); g.globalAlpha = 1; }
        const b = phase === 'predict' ? 1 : phase === 'meta' || phase === 'done' ? 0.7 : 0.2; g.strokeStyle = C.orange; g.globalAlpha = b; g.lineWidth = 1.5; g.beginPath(); g.moveTo(mx + mw, my[k] + mh / 2); g.lineTo(bx, by + bh / 2); g.stroke(); g.globalAlpha = 1;
        if (anim) for (let p = 0; p < 5; p++) { const s = (time * 0.45 + p / 5 + k * 0.13) % 1; g.fillStyle = C.orange; g.globalAlpha = Math.sin(s * Math.PI); g.beginPath(); g.arc(AI.lerp(mx + mw, bx, s), AI.lerp(my[k] + mh / 2, by + bh / 2, s), 3.2, 0, 7); g.fill(); g.globalAlpha = 1; } });
      box(bx, by, bw, bh, '', C.purple, 1); g.fillStyle = C.purple; g.textAlign = 'center'; g.fillText('meta-model', bx + bw / 2, by - 6);
      const wt = W(), show = phase === 'meta' ? AI.smooth(pt) : phase === 'done' ? 1 : 0;
      MN.forEach((m, k) => { g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText('wABC'[k + 1] + ' ' + (show ? AI.fmt(wt[k], 2) : '?'), bx + 6, by + 16 + k * 16); g.fillStyle = C.purple; g.globalAlpha = 0.8; g.fillRect(bx + 62, by + 8 + k * 16, Math.min(1, wt[k]) * (bw - 70) * show, 9); g.globalAlpha = 1; });
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('base models', mx, 78); g.fillStyle = C.orange; g.fillText(lk0 ? 'in-sample predictions →' : 'out-of-fold predictions →', mx + mw + 4, 222);
      // fit plot
      const g2 = cx.ctx, v = AI.view(cx, 0, n + 1, -3, 9); g2.clearRect(0, 0, cx.w, cx.h); AI.axes(g2, v);
      const curve = (fn, col, wd, dash) => { g2.strokeStyle = col; g2.lineWidth = wd; g2.setLineDash(dash || []); g2.beginPath(); for (let i = 0; i <= 110; i++) { const x = 1 + ((n - 1) * i) / 110, y = AI.clamp(fn(x), -3, 9); i ? g2.lineTo(v.X(x), v.Y(y)) : g2.moveTo(v.X(x), v.Y(y)); } g2.stroke(); g2.setLineDash([]); };
      const cs = [C.blue, C.green, C.teal]; fullF.forEach((fm, k) => curve(fm, cs[k], 1.3, [4, 3]));
      if (phase === 'done' || phase === 'meta') { g2.globalAlpha = phase === 'meta' ? AI.smooth(pt) : 1; curve((x) => wt[0] * fullF[0](x) + wt[1] * fullF[1](x) + wt[2] * fullF[2](x), C.purple, 3); g2.globalAlpha = 1; }
      g2.fillStyle = C.fg; xs.forEach((x, i) => { g2.beginPath(); g2.arc(v.X(x), v.Y(ys[i]), 3.5, 0, 7); g2.fill(); });
      g2.font = '11px sans-serif'; g2.textAlign = 'left'; let lx = v.pad.l + 6; [['● data', C.fg], ['┄ A linear', cs[0]], ['┄ B 1-NN', cs[1]], ['┄ C tree', cs[2]], ['— stacked (Σ w·model)', C.purple]].forEach(([t, c]) => { g2.fillStyle = c; g2.fillText(t, lx, 12); lx += g2.measureText(t).width + 12; });
      const cap = { pick: lk0 ? `① Leaky mode: every base model will train on ALL ${n} rows.` : `① Fold ${fi + 1}: hold out the orange rows. The base models will not see them.`, train: lk0 ? '② Train all three base models on every row (including the ones they will be asked to predict).' : `② Train every base model on the other ${n - n / K} rows (blue).`, predict: lk0 ? '③ Predict the rows the models memorised: 1-NN is perfect, so these predictions are too good.' : '③ Each model predicts the held-out rows. These out-of-fold predictions become the meta-model inputs.', meta: `④ Fit the meta-model on the ${n} × 3 table (non-negative ridge regression): it learns how much to trust each base model.`, done: 'Done. The purple curve is the weighted sum of the base models (each refitted on all rows). Toggle leaky mode and compare the weights and test errors.' };
      q('#cap').textContent = cap[phase];
      const f = (v) => AI.fmt(v, 3);
      q('#out').innerHTML = phase === 'done' ? `weights (OOF)    ${wH.map((v) => f(v)).join('  ')}\nweights (leaky)  ${wL.map((v) => f(v)).join('  ')}\ntest MSE, fresh data (each model fit on all rows)\n  A ${f(baseMse(0))}   B ${f(baseMse(1))}   C ${f(baseMse(2))}\n  stack with OOF weights   <span class="g">${f(testMse(wH))}</span>\n  stack with leaky weights <span class="r">${f(testMse(wL))}</span>` : 'run to the end to see weights and fresh-data errors';
    }, { fps: 30 });
  };

})();
