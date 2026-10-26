/* Machine Learning labs: preprocessing, regression + gradient descent, neural-net playground, threshold/ROC. */

/* ================= Preprocessing ================= */
AI.labs.preprocess = function (root) {
  const { view, ctl } = AI.labShell(root, 'Preprocessing lab', 'edit the transform, watch the data move',
    `<canvas style="height:340px"></canvas><div class="caption">Top: raw feature. Bottom: after your transform. Particles show individual values travelling between them.</div>`,
    `<label>Feature
       <select id="feat"><option value="income">income (k$, log-normal)</option><option value="age">age (years, normal)</option><option value="latency">latency (ms, exponential + outliers)</option></select></label>
     <label>Transform &nbsp;x′ =<input type="text" id="expr" value="(x - mean) / std" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <div class="caption">Variables: <code>x mean std min max median p05 p95</code>. Functions: <code>log sqrt abs min max tanh sigmoid …</code></div>
     <table class="stats"><thead><tr><th></th><th>before</th><th>after</th></tr></thead><tbody id="tbl"></tbody></table>`);
  const cv = view.querySelector('canvas'), c = AI.canvas(cv);
  const presets = {
    'Standardize': '(x - mean) / std', 'Min-max': '(x - min) / (max - min)', 'Log': 'log(x)',
    'Clip': 'min(max(x, p05), p95)', 'Robust': '(x - median) / (p95 - p05)', 'Square': 'x^2',
  };
  const BINS = 32, N = 500;
  const gen = {
    income: (r) => Math.exp(Math.log(40) + 0.7 * AI.randn(r)),
    age: (r) => AI.clamp(38 + 12 * AI.randn(r), 18, 90),
    latency: (r) => (r() < 0.03 ? 600 + 400 * r() : -120 * Math.log(1 - r())) + 5,
  };
  let raw = [], out = [], fn = null, particles = [];
  const rows = { top: { lo: 0, hi: 1, h: new Array(BINS).fill(0) }, bot: { lo: 0, hi: 1, h: new Array(BINS).fill(0) } };
  const targets = { top: null, bot: null };
  const ids = (s) => AI.$(s, ctl);

  const hist = (vals) => {
    const f = vals.filter(Number.isFinite);
    if (!f.length) return { lo: 0, hi: 1, h: new Array(BINS).fill(0), n: 0 };
    let lo = Math.min(...f), hi = Math.max(...f);
    if (lo === hi) { lo -= 1; hi += 1; }
    const h = new Array(BINS).fill(0);
    f.forEach((v) => h[Math.min(BINS - 1, Math.floor(((v - lo) / (hi - lo)) * BINS))]++);
    const m = Math.max(...h);
    return { lo, hi, h: h.map((v) => v / m), n: f.length };
  };
  const load = () => {
    const r = AI.rng(7), g = gen[ids('#feat').value];
    raw = Array.from({ length: N }, () => g(r));
    targets.top = hist(raw);
    particles = Array.from({ length: 70 }, (_, i) => ({ i: Math.floor(r() * N), ph: r() }));
    apply();
  };
  const apply = () => {
    if (!fn) return;
    const s = [...raw].sort((a, b) => a - b), st = AI.stats(raw);
    const sc = { mean: st.mean, std: st.std, min: st.min, max: st.max, median: AI.quantile(s, 0.5), p05: AI.quantile(s, 0.05), p95: AI.quantile(s, 0.95) };
    out = raw.map((x) => fn({ ...sc, x }));
    targets.bot = hist(out);
    const a = AI.stats(raw), b = AI.stats(out.filter(Number.isFinite));
    const row = (n, k, d = 2) => `<tr><td>${n}</td><td>${AI.fmt(a[k], d)}</td><td>${AI.fmt(b[k], d)}</td></tr>`;
    ids('#tbl').innerHTML = row('mean', 'mean') + row('std', 'std') + row('min', 'min') + row('max', 'max') + row('skew', 'skew') + `<tr><td>invalid</td><td>0</td><td>${N - targets.bot.n}</td></tr>`;
  };
  const ei = AI.exprInput(ids('#expr'), ids('#err'), {
    allowed: ['x', 'mean', 'std', 'min', 'max', 'median', 'p05', 'p95'], onChange: (f) => { fn = f; apply(); },
  });
  const pc = ids('#presets');
  Object.entries(presets).forEach(([k, v]) => {
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => ei.set(v); pc.appendChild(b);
  });
  ids('#feat').onchange = load;
  ei.run(); load();

  const drawHist = (g, st, x0, w, y0, h, color, label) => {
    const C = AI.col(), bw = w / BINS;
    g.fillStyle = color; g.globalAlpha = 0.85;
    st.h.forEach((v, i) => g.fillRect(x0 + i * bw + 1, y0 + h - v * h, bw - 2, v * h));
    g.globalAlpha = 1;
    g.strokeStyle = C.border; g.beginPath(); g.moveTo(x0, y0 + h + 0.5); g.lineTo(x0 + w, y0 + h + 0.5); g.stroke();
    g.fillStyle = C.muted; g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
    const step = AI.niceStep(st.hi - st.lo, 5);
    for (let t = Math.ceil(st.lo / step) * step; t <= st.hi; t += step) g.fillText(AI.fmt(+t.toFixed(6), 2), x0 + ((t - st.lo) / (st.hi - st.lo)) * w, y0 + h + 14);
    g.textAlign = 'left'; g.fillStyle = color; g.fillText(label, x0, y0 - 4);
  };
  AI.loop(root, (dt, time) => {
    const k = 1 - Math.exp(-dt * 7);
    for (const key of ['top', 'bot']) {
      const t = targets[key], r = rows[key]; if (!t) continue;
      r.lo += (t.lo - r.lo) * k; r.hi += (t.hi - r.hi) * k;
      for (let i = 0; i < BINS; i++) r.h[i] += (t.h[i] - r.h[i]) * k;
    }
    const C = AI.col(), g = c.ctx, x0 = 24, w = c.w - 48, hh = 90;
    g.clearRect(0, 0, c.w, c.h);
    const yT = 22, yB = c.h - 24 - hh;
    drawHist(g, rows.top, x0, w, yT, hh, C.blue, 'raw x');
    drawHist(g, rows.bot, x0, w, yB, hh, C.green, "transformed x′");
    // travelling particles
    for (const p of particles) {
      const a = raw[p.i], b = out[p.i];
      if (!Number.isFinite(b)) continue;
      const s = AI.smooth(((time * 0.35 + p.ph) % 1) * 1.3 - 0.15);
      const xa = x0 + ((a - rows.top.lo) / (rows.top.hi - rows.top.lo)) * w, xb = x0 + ((b - rows.bot.lo) / (rows.bot.hi - rows.bot.lo)) * w;
      const ya = yT + hh + 22, yb = yB - 8;
      g.globalAlpha = Math.sin(s * Math.PI) * 0.9;
      g.fillStyle = AI.mix(C.blue, C.green, s);
      g.beginPath(); g.arc(AI.lerp(xa, xb, s), AI.lerp(ya, yb, s), 3, 0, 7); g.fill();
    }
    g.globalAlpha = 1;
  });
};

/* ================= Regression + gradient descent ================= */
AI.labs.regression = function (root) {
  const { view, ctl } = AI.labShell(root, 'Training lab: fit any model with gradient descent', 'edit the equation, press play',
    `<canvas id="fit" style="height:300px"></canvas><canvas id="loss" style="height:130px"></canvas>`,
    `<label>Dataset <select id="data"><option value="linear">line + noise</option><option value="quad">parabola + noise</option><option value="sine">sine wave</option><option value="outlier">line with outliers</option></select></label>
     <label>Model &nbsp;ŷ = f(x; w, b, c) <input type="text" id="expr" value="w*x + b" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <label>Learning rate η <span class="v" id="lrv"></span><input type="range" id="lr" min="-3.5" max="-0.3" step="0.05" value="-1.3"></label>
     <label>Steps per frame <span class="v" id="spv"></span><input type="range" id="sp" min="1" max="20" step="1" value="2"></label>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="noise">New data</button></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c1 = AI.canvas(q('#fit')), c2 = AI.canvas(q('#loss'));
  const models = { 'line': 'w*x + b', 'parabola': 'w*x^2 + b*x + c', 'sine': 'w*sin(b*x) + c', 'tanh': 'w*tanh(b*x) + c' };
  const sugg = { linear: 'line', quad: 'parabola', sine: 'sine', outlier: 'line' };
  const P = ['w', 'b', 'c'], init = { w: -1, b: 1.5, c: 0.5 };
  let xs = [], ys = [], seed = 3, fn = null, used = [], p = { ...init }, hist = [], playing = true, diverged = false, lastG = {}, steps = 0;

  const gen = () => {
    const r = AI.rng(seed), d = q('#data').value;
    xs = Array.from({ length: 40 }, (_, i) => -3 + (6 * i) / 39 + (r() - 0.5) * 0.1);
    ys = xs.map((x) => {
      const n = 0.35 * AI.randn(r);
      return d === 'linear' ? 1.5 * x + 0.5 + n : d === 'quad' ? 0.6 * x * x - 1 + n : d === 'sine' ? 2 * Math.sin(1.3 * x) + n : 1.5 * x + 0.5 + n + (r() < 0.1 ? 5 * (r() - 0.5) * 2 : 0);
    });
  };
  const loss = (pp) => { let s = 0; for (let i = 0; i < xs.length; i++) { const e = fn({ x: xs[i], ...pp }) - ys[i]; s += e * e; } return s / xs.length; };
  const reset = () => { p = { ...init }; hist = []; steps = 0; diverged = false; lastG = {}; };
  const stepOnce = () => {
    if (!fn || diverged) return;
    const lr = Math.pow(10, +q('#lr').value), g = {};
    for (const k of used) {
      const h = 1e-5; g[k] = (loss({ ...p, [k]: p[k] + h }) - loss({ ...p, [k]: p[k] - h })) / (2 * h);
    }
    for (const k of used) p[k] -= lr * g[k];
    lastG = g; steps++;
    const L = loss(p);
    hist.push(L);
    if (!Number.isFinite(L) || L > 1e8) diverged = true;
  };
  const ei = AI.exprInput(q('#expr'), q('#err'), {
    allowed: ['x', ...P], check: (f) => { if (!Number.isFinite(f({ x: 1, w: 1, b: 1, c: 1 }))) throw new Error('Model gives a non-finite value'); },
    onChange: (f, ast) => { fn = f; used = P.filter((k) => AI.vars(ast).has(k)); reset(); },
  });
  const pc = q('#presets');
  Object.entries(models).forEach(([k, v]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => ei.set(v); pc.appendChild(b); });
  q('#data').onchange = () => { ei.set(models[sugg[q('#data').value]]); gen(); };
  q('#noise').onclick = () => { seed += 11; gen(); reset(); };
  q('#reset').onclick = reset;
  q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; stepOnce(); };
  q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
  gen(); ei.run();

  AI.loop(root, () => {
    q('#lrv').textContent = AI.fmt(Math.pow(10, +q('#lr').value), 4);
    q('#spv').textContent = q('#sp').value;
    if (playing) for (let i = 0; i < +q('#sp').value; i++) stepOnce();
    const C = AI.col();
    // fit
    let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
    const v = AI.view(c1, -3.5, 3.5, -6, 6);
    AI.axes(g, v);
    const curve = (x) => fn({ x, ...p });
    if (!diverged) {
      g.strokeStyle = C.red; g.globalAlpha = 0.35; g.lineWidth = 1;
      xs.forEach((x, i) => { const yh = curve(x); if (!Number.isFinite(yh)) return; g.beginPath(); g.moveTo(v.X(x), v.Y(ys[i])); g.lineTo(v.X(x), v.Y(AI.clamp(yh, -20, 20))); g.stroke(); });
      g.globalAlpha = 1;
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath();
      for (let i = 0; i <= 160; i++) { const x = -3.5 + (7 * i) / 160, y = AI.clamp(curve(x), -30, 30); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
      g.stroke();
    }
    g.fillStyle = C.blue;
    xs.forEach((x, i) => { g.beginPath(); g.arc(v.X(x), v.Y(ys[i]), 3.5, 0, 7); g.fill(); });
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
    g.fillText('● data   — model   | residuals', v.pad.l + 6, v.pad.t + 12);
    // loss
    g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
    const pad = { l: 40, r: 12, t: 16, b: 22 };
    const vals = hist.map((l) => Math.log10(Math.max(l, 1e-6)));
    if (vals.length > 1) {
      const lo = Math.min(...vals), hi = Math.max(...vals) + 1e-9, n = Math.max(vals.length - 1, 1);
      const vv = AI.view(c2, 0, Math.max(n, 20), lo - 0.2, hi + 0.2, pad);
      AI.axes(g, vv);
      g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath();
      vals.forEach((y, i) => (i ? g.lineTo(vv.X(i), vv.Y(y)) : g.moveTo(vv.X(i), vv.Y(y)))); g.stroke();
    }
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
    g.fillText('log₁₀ loss vs step', pad.l + 6, 12);
    // readout
    const L = hist.length ? hist[hist.length - 1] : loss(p), lr = Math.pow(10, +q('#lr').value);
    let t = `<span class="m">step</span> ${steps}   <span class="m">L = MSE</span> <span class="o">${AI.fmt(L, 4)}</span>\n`;
    if (diverged) t += `<span class="r">Diverged — lower the learning rate η and Reset.</span>\n`;
    used.forEach((k) => {
      const gk = lastG[k];
      t += `<span class="b">${k}</span> ← ${AI.fmt(p[k] + (gk !== undefined ? lr * gk : 0), 3)} − η·(<span class="g">${gk === undefined ? '…' : AI.fmt(gk, 3)}</span>) = <b>${AI.fmt(p[k], 3)}</b>\n`;
    });
    q('#out').innerHTML = t;
  });
};

/* ================= Neural network playground ================= */
AI.labs.nn = function (root) {
  const { view, ctl } = AI.labShell(root, 'Neural network playground', 'change the architecture, then train; click the map to trace a point',
    `<canvas id="map" style="height:340px;cursor:crosshair"></canvas><canvas id="net" style="height:230px"></canvas>`,
    `<label>Dataset <select id="data"><option value="circle">circle</option><option value="xor">xor</option><option value="spiral">spiral</option><option value="gauss">two clusters</option></select></label>
     <div>Input features<div class="row" id="feats"></div></div>
     <div>Hidden layers <div class="layers" id="layers"></div></div>
     <label>Activation <select id="act"><option>tanh</option><option>relu</option><option>sigmoid</option></select></label>
     <label>Learning rate <span class="v" id="lrv"></span><input type="range" id="lr" min="-2.5" max="0.3" step="0.05" value="-1"></label>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Reset weights</button></div>
     <div class="readout" id="out"></div>
     <div class="caption" id="formula"></div>`);
  const q = (s) => AI.$(s, root);
  const cm = AI.canvas(q('#map')), cn = AI.canvas(q('#net'));
  const FEATS = [
    { n: 'x', f: (x) => x }, { n: 'y', f: (x, y) => y }, { n: 'x²', f: (x) => x * x }, { n: 'y²', f: (x, y) => y * y },
    { n: 'xy', f: (x, y) => x * y }, { n: 'sin x', f: (x) => Math.sin(3 * x) }, { n: 'sin y', f: (x, y) => Math.sin(3 * y) },
  ];
  const ACT = {
    tanh: [Math.tanh, (a) => 1 - a * a], relu: [(z) => Math.max(0, z), (a) => (a > 0 ? 1 : 0)],
    sigmoid: [(z) => 1 / (1 + Math.exp(-z)), (a) => a * (1 - a)],
  };
  let on = [0, 1], hidden = [6, 4], pts = [], net, hist = [], playing = true, steps = 0, probe = null, seed = 1;
  const off = document.createElement('canvas'); off.width = off.height = 48;
  let frame = 0;

  const data = () => {
    const r = AI.rng(5), d = q('#data').value, P = [];
    for (let i = 0; i < 200; i++) {
      let x, y, l;
      if (d === 'circle') { l = i % 2; const a = r() * 6.283, rad = l ? 0.1 + 0.3 * r() : 0.6 + 0.3 * r(); x = rad * Math.cos(a); y = rad * Math.sin(a); }
      else if (d === 'xor') { x = r() * 2 - 1; y = r() * 2 - 1; x += 0.08 * Math.sign(x); y += 0.08 * Math.sign(y); l = x * y > 0 ? 1 : 0; x = AI.clamp(x, -1, 1); y = AI.clamp(y, -1, 1); }
      else if (d === 'spiral') { l = i % 2; const t = (i / 200) * 2 * 3.1 + 0.2, rad = t / 7; const a = t * 1.6 + l * Math.PI; x = rad * Math.cos(a) * 1.35 + 0.03 * AI.randn(r); y = rad * Math.sin(a) * 1.35 + 0.03 * AI.randn(r); }
      else { l = i % 2; const s = l ? 0.5 : -0.5; x = s + 0.22 * AI.randn(r); y = s + 0.22 * AI.randn(r); }
      P.push({ x, y, l });
    }
    pts = P;
  };
  const build = () => {
    const sizes = [on.length, ...hidden, 1], r = AI.rng(seed++);
    net = {
      sizes, W: [], B: [],
    };
    for (let l = 1; l < sizes.length; l++) {
      const s = Math.sqrt(2 / sizes[l - 1]);
      net.W.push(Array.from({ length: sizes[l] }, () => Array.from({ length: sizes[l - 1] }, () => AI.randn(r) * s)));
      net.B.push(new Array(sizes[l]).fill(0));
    }
    hist = []; steps = 0;
    q('#formula').textContent = 'ŷ = σ(' + Array.from({ length: hidden.length }, (_, i) => `W${hidden.length + 1 - i}·${q('#act').value}(`).join('') + 'W1·φ(x,y)+b1' + Array.from({ length: hidden.length }, (_, i) => `)+b${i + 2}`).join('') + ')';
  };
  const feat = (x, y) => on.map((i) => FEATS[i].f(x, y));
  const forward = (inp) => {
    const act = ACT[q('#act').value][0], A = [inp];
    for (let l = 0; l < net.W.length; l++) {
      const last = l === net.W.length - 1, prev = A[l];
      A.push(net.W[l].map((w, j) => { let z = net.B[l][j]; for (let i = 0; i < w.length; i++) z += w[i] * prev[i]; return last ? 1 / (1 + Math.exp(-z)) : act(z); }));
    }
    return A;
  };
  const trainBatch = (lr) => {
    const d = ACT[q('#act').value][1], gW = net.W.map((m) => m.map((r) => r.map(() => 0))), gB = net.B.map((b) => b.map(() => 0)), bs = 16;
    for (let n = 0; n < bs; n++) {
      const pt = pts[Math.floor(Math.random() * pts.length)], A = forward(feat(pt.x, pt.y));
      let delta = [A[A.length - 1][0] - pt.l];
      for (let l = net.W.length - 1; l >= 0; l--) {
        const prev = A[l], nd = new Array(prev.length).fill(0);
        for (let j = 0; j < delta.length; j++) {
          gB[l][j] += delta[j];
          for (let i = 0; i < prev.length; i++) { gW[l][j][i] += delta[j] * prev[i]; nd[i] += delta[j] * net.W[l][j][i]; }
        }
        if (l > 0) delta = nd.map((v, i) => v * d(prev[i]));
      }
    }
    for (let l = 0; l < net.W.length; l++) for (let j = 0; j < net.W[l].length; j++) {
      net.B[l][j] -= (lr * gB[l][j]) / bs;
      for (let i = 0; i < net.W[l][j].length; i++) net.W[l][j][i] -= (lr * gW[l][j][i]) / bs;
    }
    steps++;
  };
  const fullLoss = () => {
    let s = 0;
    for (const p of pts) { const o = forward(feat(p.x, p.y)).pop()[0]; s -= p.l ? Math.log(o + 1e-9) : Math.log(1 - o + 1e-9); }
    return s / pts.length;
  };

  // controls
  const drawFeats = () => {
    q('#feats').innerHTML = '';
    FEATS.forEach((f, i) => {
      const b = document.createElement('button'); b.className = 'chip'; b.textContent = f.n;
      if (on.includes(i)) b.style.borderColor = 'var(--blue)', b.style.color = 'var(--blue)';
      b.onclick = () => { on = on.includes(i) ? on.filter((v) => v !== i) : [...on, i].sort(); if (!on.length) on = [i]; drawFeats(); build(); };
      q('#feats').appendChild(b);
    });
  };
  const drawLayers = () => {
    const el = q('#layers'); el.innerHTML = '';
    hidden.forEach((n, i) => {
      const inp = document.createElement('input'); inp.type = 'number'; inp.min = 1; inp.max = 8; inp.value = n;
      inp.onchange = () => { hidden[i] = AI.clamp(+inp.value || 1, 1, 8); build(); };
      el.appendChild(inp);
    });
    const add = document.createElement('button'); add.className = 'btn small'; add.textContent = '+ layer'; add.disabled = hidden.length >= 3;
    add.onclick = () => { hidden.push(4); drawLayers(); build(); };
    const rm = document.createElement('button'); rm.className = 'btn small'; rm.textContent = '− layer'; rm.disabled = hidden.length <= 1;
    rm.onclick = () => { hidden.pop(); drawLayers(); build(); };
    el.append(add, rm);
  };
  q('#data').onchange = () => { data(); build(); };
  q('#act').onchange = build;
  q('#reset').onclick = build;
  q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
  q('#map').addEventListener('click', (e) => {
    const r = q('#map').getBoundingClientRect();
    probe = { x: ((e.clientX - r.left) / r.width) * 2.4 - 1.2, y: -(((e.clientY - r.top) / r.height) * 2.4 - 1.2) };
  });
  data(); drawFeats(); drawLayers(); build();

  AI.loop(root, (dt, time) => {
    q('#lrv').textContent = AI.fmt(Math.pow(10, +q('#lr').value), 3);
    if (playing) { const lr = Math.pow(10, +q('#lr').value); for (let i = 0; i < 6; i++) trainBatch(lr); if (frame % 4 === 0) hist.push(fullLoss()); if (hist.length > 300) hist.shift(); }
    const C = AI.col(), c0 = AI.rgb(C.orange), c1 = AI.rgb(C.blue), bg = AI.rgb(C.bg);
    // decision map
    if (frame++ % 3 === 0 || !playing) {
      const g = off.getContext('2d'), im = g.createImageData(48, 48);
      for (let j = 0; j < 48; j++) for (let i = 0; i < 48; i++) {
        const x = ((i + 0.5) / 48) * 2.4 - 1.2, y = -(((j + 0.5) / 48) * 2.4 - 1.2), p = forward(feat(x, y)).pop()[0], s = 0.2 + 0.6 * Math.abs(2 * p - 1);
        for (let k = 0; k < 3; k++) im.data[(j * 48 + i) * 4 + k] = bg[k] + (c0[k] + (c1[k] - c0[k]) * p - bg[k]) * s;
        im.data[(j * 48 + i) * 4 + 3] = 255;
      }
      g.putImageData(im, 0, 0);
    }
    let g = cm.ctx; g.clearRect(0, 0, cm.w, cm.h);
    g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, cm.w, cm.h);
    const M = (x, y) => [((x + 1.2) / 2.4) * cm.w, ((1.2 - y) / 2.4) * cm.h];
    pts.forEach((p) => { const [px, py] = M(p.x, p.y); g.fillStyle = p.l ? C.blue : C.orange; g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.arc(px, py, 3.5, 0, 7); g.fill(); g.globalAlpha = 0.6; g.stroke(); g.globalAlpha = 1; });
    if (probe) { const [px, py] = M(probe.x, probe.y); g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.arc(px, py, 8, 0, 7); g.moveTo(px - 12, py); g.lineTo(px + 12, py); g.moveTo(px, py - 12); g.lineTo(px, py + 12); g.stroke(); }
    // loss sparkline on the map
    if (hist.length > 1) {
      const hi = Math.max(...hist, 0.8); g.strokeStyle = C.fg; g.globalAlpha = 0.8; g.lineWidth = 1.5; g.beginPath();
      hist.forEach((l, i) => { const x = 10 + (i / 300) * 110, y = 36 - (l / hi) * 26; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.globalAlpha = 1;
      g.fillStyle = C.fg; g.font = '11px ui-monospace,Menlo,monospace'; g.textAlign = 'left'; g.fillText('loss ' + AI.fmt(hist[hist.length - 1], 3), 10, 50);
    }
    // network diagram
    g = cn.ctx; g.clearRect(0, 0, cn.w, cn.h);
    const sizes = net.sizes, L = sizes.length, act = probe ? forward(feat(probe.x, probe.y)) : null;
    const pos = sizes.map((n, l) => Array.from({ length: n }, (_, i) => [40 + (l * (cn.w - 100)) / (L - 1), ((i + 1) * cn.h) / (n + 1)]));
    for (let l = 0; l < L - 1; l++) for (let j = 0; j < sizes[l + 1]; j++) for (let i = 0; i < sizes[l]; i++) {
      const w = net.W[l][j][i], [x0, y0] = pos[l][i], [x1, y1] = pos[l + 1][j], a = Math.abs(w);
      g.strokeStyle = w > 0 ? C.blue : C.orange; g.globalAlpha = AI.clamp(a * 0.5, 0.05, 0.9); g.lineWidth = AI.clamp(a * 1.4, 0.4, 4);
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      if (a > 0.3) { const ph = (time * 0.5 + (i * 7 + j * 13 + l * 3) * 0.137) % 1; g.globalAlpha = Math.min(1, a); g.fillStyle = C.fg; g.beginPath(); g.arc(AI.lerp(x0, x1, ph), AI.lerp(y0, y1, ph), 1.8, 0, 7); g.fill(); }
    }
    g.globalAlpha = 1;
    pos.forEach((col, l) => col.forEach(([x, y], i) => {
      const a = act ? act[l][i] : 0, isOut = l === L - 1;
      const t = AI.clamp(isOut || q('#act').value === 'sigmoid' || q('#act').value === 'relu' ? a : a * 0.5 + 0.5, 0, 1);
      g.fillStyle = act ? (isOut ? AI.mix(C.orange, C.blue, a) : AI.mix(C.bg3, C.purple, Math.min(1, Math.abs(a)))) : C.bg2;
      g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 9, 0, 7); g.fill(); g.stroke();
      if (l === 0) { g.fillStyle = C.muted; g.font = '10px sans-serif'; g.textAlign = 'right'; g.fillText(FEATS[on[i]].n, x - 13, y + 3); }
    }));
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
    g.fillText(probe ? `trace: output p(blue) = ${AI.fmt(act[L - 1][0], 3)}` : 'click the map to trace a point through the network', 10, 14);
    const cnt = net.W.reduce((s, m) => s + m.length * m[0].length + m.length, 0);
    q('#out').innerHTML = `<span class="m">step</span> ${steps}  <span class="m">params</span> ${cnt}\n<span class="m">loss</span> <span class="o">${hist.length ? AI.fmt(hist[hist.length - 1], 4) : '…'}</span>`;
  });
};

/* ================= Threshold / ROC ================= */
AI.labs.threshold = function (root) {
  const { view, ctl } = AI.labShell(root, 'Post-processing lab: choose a threshold', 'scores stream in; your cut-off turns them into decisions',
    `<canvas id="dist" style="height:250px"></canvas><canvas id="roc" style="height:220px"></canvas>`,
    `<label>Model separation d′ <span class="v" id="dv"></span><input type="range" id="d" min="0" max="4" step="0.1" value="1.8"></label>
     <label>Score noise σ <span class="v" id="sv"></span><input type="range" id="s" min="0.5" max="1.6" step="0.05" value="1"></label>
     <label>Threshold t <span class="v" id="tv"></span><input type="range" id="t" min="-2" max="6" step="0.05" value="0.9"></label>
     <div class="legend"><span><i style="background:var(--green)"></i>TP</span><span><i style="background:var(--red)"></i>FP</span><span><i style="background:var(--muted)"></i>TN</span><span><i style="background:var(--orange)"></i>FN</span></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c1 = AI.canvas(q('#dist')), c2 = AI.canvas(q('#roc'));
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  let parts = [], acc = 0;

  AI.loop(root, (dt) => {
    const d = +q('#d').value, s = +q('#s').value, t = +q('#t').value, C = AI.col();
    q('#dv').textContent = d; q('#sv').textContent = s; q('#tv').textContent = AI.fmt(t, 2);
    const v = AI.view(c1, -4, 8, 0, 0.5, { l: 12, r: 12, t: 12, b: 24 });
    // stream
    acc += dt * 22;
    while (acc > 1) { acc--; const pos = Math.random() < 0.5; parts.push({ pos, x: (pos ? d : 0) + s * AI.randn(), y: 0, vy: 40 + Math.random() * 30 }); }
    parts.forEach((p) => { p.y += p.vy * dt; });
    let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
    AI.axes(g, v, { noY: true, noGrid: true });
    const fill = (m, cLeft, cRight) => {
      for (const [from, to, col] of [[-4, t, cLeft], [t, 8, cRight]]) {
        g.fillStyle = col; g.globalAlpha = 0.45; g.beginPath(); g.moveTo(v.X(from), v.Y(0));
        for (let x = from; x <= to + 1e-9; x += 0.05) g.lineTo(v.X(x), v.Y(pdf(x, m, s) * 0.5));
        g.lineTo(v.X(to), v.Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1;
      }
    };
    fill(0, C.muted, C.red);
    fill(d, C.orange, C.green);
    g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v.X(t), 8); g.lineTo(v.X(t), v.Y(0)); g.stroke(); g.setLineDash([]);
    g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('t', v.X(t) + 4, 18);
    g.fillStyle = C.muted; g.fillText('negatives', v.X(0) - 24, v.Y(pdf(0, 0, s) * 0.5) - 6); g.fillText('positives', v.X(d) - 22, v.Y(pdf(d, d, s) * 0.5) - 6);
    for (const p of parts) {
      const x = v.X(p.x), yTop = 18 + p.y, yEnd = Math.min(yTop, v.Y(0) - 4);
      const col = p.pos ? (p.x > t ? C.green : C.orange) : p.x > t ? C.red : C.muted;
      g.fillStyle = col; g.globalAlpha = 0.9; g.beginPath(); g.arc(x, yEnd, 3, 0, 7); g.fill();
    }
    g.globalAlpha = 1;
    parts = parts.filter((p) => 18 + p.y < v.Y(0) + 40);
    // ROC
    g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
    const pad = { l: 40, r: 12, t: 14, b: 26 }, vr = AI.view(c2, 0, 1, 0, 1, pad);
    AI.axes(g, vr);
    g.strokeStyle = C.border; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(vr.X(0), vr.Y(0)); g.lineTo(vr.X(1), vr.Y(1)); g.stroke(); g.setLineDash([]);
    g.strokeStyle = C.purple; g.lineWidth = 2.5; g.beginPath();
    for (let x = 9; x >= -5; x -= 0.1) { const fpr = 1 - Phi(x / s), tpr = 1 - Phi((x - d) / s); x === 9 ? g.moveTo(vr.X(fpr), vr.Y(tpr)) : g.lineTo(vr.X(fpr), vr.Y(tpr)); }
    g.stroke();
    const FPR = 1 - Phi(t / s), TPR = 1 - Phi((t - d) / s);
    g.fillStyle = C.fg; g.beginPath(); g.arc(vr.X(FPR), vr.Y(TPR), 6, 0, 7); g.fill();
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText('false positive rate', vr.X(0.5), c2.h - 4);
    g.save(); g.translate(11, vr.Y(0.5)); g.rotate(-Math.PI / 2); g.fillText('true positive rate', 0, 0); g.restore();
    // metrics on 1000 samples/class
    const TP = 1000 * TPR, FN = 1000 - TP, FP = 1000 * FPR, TN = 1000 - FP;
    const prec = TP / (TP + FP || 1), rec = TPR, f1 = (2 * prec * rec) / (prec + rec || 1), accu = (TP + TN) / 2000;
    q('#out').innerHTML = `<span class="m">per 1000+1000 cases</span>\n<span class="g">TP ${Math.round(TP)}</span>  <span class="r">FP ${Math.round(FP)}</span>\n<span class="o">FN ${Math.round(FN)}</span>  <span class="m">TN ${Math.round(TN)}</span>\n\nprecision <b>${AI.fmt(prec, 3)}</b>\nrecall    <b>${AI.fmt(rec, 3)}</b>\nF1        <b>${AI.fmt(f1, 3)}</b>\naccuracy  <b>${AI.fmt(accu, 3)}</b>`;
  });
};
