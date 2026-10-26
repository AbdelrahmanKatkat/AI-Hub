/* Mathematics labs: matrix transforms, derivatives, optimizers on a surface, backprop graph, sampling / CLT. */

/* ================= Linear algebra ================= */
AI.labs.matrix = function (root) {
  const { view, ctl } = AI.labShell(root, 'Matrix transform lab', 'a matrix is a function that moves space',
    `<canvas style="height:420px"></canvas>`,
    `<div>Matrix M <div class="row" id="m" style="margin-top:6px;display:grid;grid-template-columns:1fr 1fr;gap:6px"></div></div>
     <div class="chips" id="presets"></div>
     <label>Vector v = (x, y)
       <div class="row"><input type="range" id="vx" min="-3" max="3" step="0.1" value="1" style="flex:1"><input type="range" id="vy" min="-3" max="3" step="0.1" value="1" style="flex:1"></div></label>
     <div class="row"><button class="btn primary" id="anim">▶ Animate from identity</button></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c = AI.canvas(AI.$('canvas', view));
  const names = ['a', 'b', 'c', 'd'], M = [1, 0, 0, 1], D = [1, 0, 0, 1];
  const inputs = names.map((n, i) => {
    const el = document.createElement('input'); el.type = 'number'; el.step = 0.1; el.value = M[i]; el.title = n;
    el.oninput = () => (M[i] = +el.value || 0); q('#m').appendChild(el); return el;
  });
  const setM = (m) => m.forEach((v, i) => { M[i] = +v.toFixed(4); inputs[i].value = M[i]; });
  const rot = (deg) => { const a = (deg * Math.PI) / 180; return [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)]; };
  const presets = { Identity: [1, 0, 0, 1], 'Rotate 45°': rot(45), 'Scale 1.5×': [1.5, 0, 0, 1.5], Stretch: [2, 0, 0, 0.5], Shear: [1, 1, 0, 1], Reflect: [-1, 0, 0, 1], Squash: [1, 1, 1, 1], 'Spin + scale': rot(30).map((v) => v * 1.4) };
  Object.entries(presets).forEach(([k, m]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => setM(m); q('#presets').appendChild(b); });
  q('#anim').onclick = () => { D[0] = 1; D[1] = 0; D[2] = 0; D[3] = 1; };
  setM(presets['Spin + scale']);

  AI.loop(root, (dt) => {
    const k = 1 - Math.exp(-dt * 1.6);
    for (let i = 0; i < 4; i++) D[i] += (M[i] - D[i]) * k;
    const C = AI.col(), g = c.ctx, R = 4.5, s = Math.min(c.w, c.h) / (2 * R), ox = c.w / 2, oy = c.h / 2;
    const P = (x, y) => [ox + x * s, oy - y * s], T = (x, y) => [D[0] * x + D[1] * y, D[2] * x + D[3] * y];
    g.clearRect(0, 0, c.w, c.h);
    // original grid (faint)
    g.lineWidth = 1; g.strokeStyle = C.border; g.globalAlpha = 0.6;
    for (let i = -R; i <= R; i++) { g.beginPath(); g.moveTo(...P(i, -R)); g.lineTo(...P(i, R)); g.moveTo(...P(-R, i)); g.lineTo(...P(R, i)); g.stroke(); }
    g.globalAlpha = 1;
    // transformed grid
    const line = (a, b, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(...P(...T(...a))); g.lineTo(...P(...T(...b))); g.stroke(); };
    for (let i = -R; i <= R; i++) { const ax = i === 0; line([i, -R * 2], [i, R * 2], ax ? C.fg : C.blue, ax ? 1.5 : 1); line([-R * 2, i], [R * 2, i], ax ? C.fg : C.blue, ax ? 1.5 : 1); }
    // unit square
    g.fillStyle = C.purple; g.globalAlpha = 0.3; g.beginPath();
    [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(([x, y], i) => { const p = P(...T(x, y)); i ? g.lineTo(...p) : g.moveTo(...p); }); g.fill(); g.globalAlpha = 1;
    // eigenvectors
    const tr = D[0] + D[3], det = D[0] * D[3] - D[1] * D[2], disc = tr * tr / 4 - det;
    if (disc >= 0) {
      for (const sg of [1, -1]) {
        const lam = tr / 2 + sg * Math.sqrt(disc);
        let ex = D[1], ey = lam - D[0]; if (Math.abs(ex) + Math.abs(ey) < 1e-6) { ex = lam - D[3]; ey = D[2]; }
        if (Math.abs(ex) + Math.abs(ey) < 1e-6) { ex = sg > 0 ? 1 : 0; ey = sg > 0 ? 0 : 1; }
        const n = Math.hypot(ex, ey); ex /= n; ey /= n;
        g.strokeStyle = C.yellow; g.setLineDash([6, 5]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(...P(-ex * R * 2, -ey * R * 2)); g.lineTo(...P(ex * R * 2, ey * R * 2)); g.stroke(); g.setLineDash([]);
      }
    }
    AI.arrow(g, ...P(0, 0), ...P(D[0], D[2]), C.green, 3);
    AI.arrow(g, ...P(0, 0), ...P(D[1], D[3]), C.red, 3);
    const vx = +q('#vx').value, vy = +q('#vy').value, [wx, wy] = T(vx, vy);
    AI.arrow(g, ...P(0, 0), ...P(vx, vy), C.muted, 2);
    AI.arrow(g, ...P(0, 0), ...P(wx, wy), C.orange, 3);
    g.font = '12px sans-serif'; g.fillStyle = C.green; g.fillText('î', ...P(D[0], D[2]).map((v, i) => v + (i ? -6 : 6)));
    g.fillStyle = C.red; g.fillText('ĵ', ...P(D[1], D[3]).map((v, i) => v + (i ? -6 : 6)));
    g.fillStyle = C.orange; g.fillText('Mv', ...P(wx, wy).map((v, i) => v + (i ? -8 : 8)));
    g.fillStyle = C.muted; g.fillText('v', ...P(vx, vy).map((v, i) => v + (i ? -8 : 8)));
    // info (uses target matrix)
    const [a, b, cc, d] = M, t2 = a + d, dt2 = a * d - b * cc, ds = (t2 * t2) / 4 - dt2;
    const eig = ds >= 0 ? `${AI.fmt(t2 / 2 + Math.sqrt(ds), 3)}, ${AI.fmt(t2 / 2 - Math.sqrt(ds), 3)}` : `${AI.fmt(t2 / 2, 3)} ± ${AI.fmt(Math.sqrt(-ds), 3)}i  <span class="m">(rotation)</span>`;
    const [mx, my] = [a * vx + b * vy, cc * vx + d * vy];
    q('#out').innerHTML = `<span class="g">î → (${AI.fmt(a, 2)}, ${AI.fmt(cc, 2)})</span>  <span class="r">ĵ → (${AI.fmt(b, 2)}, ${AI.fmt(d, 2)})</span>\n<span class="o">Mv = (${AI.fmt(mx, 2)}, ${AI.fmt(my, 2)})</span>\n\ndet = <b>${AI.fmt(dt2, 3)}</b> <span class="m">area scale${dt2 < 0 ? ', flipped' : ''}${Math.abs(dt2) < 1e-6 ? ', collapses a dimension' : ''}</span>\ntrace = ${AI.fmt(t2, 3)}\nλ = ${eig}\n<span class="m">yellow dashed lines = eigenvectors: directions M only stretches</span>`;
  });
};

/* ================= Derivatives ================= */
AI.labs.derivative = function (root) {
  const { view, ctl } = AI.labShell(root, 'Derivative explorer', 'the tangent is the limit of secants',
    `<canvas style="height:380px"></canvas>`,
    `<label>f(x) = <input type="text" id="expr" value="sin(x) + x^2/4" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <label>Point x₀ <span class="v" id="xv"></span><input type="range" id="x0" min="-5" max="5" step="0.05" value="1"></label>
     <label>Step h <span class="v" id="hv"></span><input type="range" id="h" min="-3" max="0.4" step="0.01" value="0.2"></label>
     <div class="row"><button class="btn primary" id="shrink">▶ Shrink h → 0</button><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="showd" checked> show f′(x)</label></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c = AI.canvas(AI.$('canvas', view));
  let fn = null, ast = null, shrinking = false, yr = [-3, 3];
  const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f, a) => { fn = f; ast = a; } });
  ['sin(x) + x^2/4', 'x^3 - 3*x', '3*exp(-x^2)', 'abs(x)', '2*tanh(2*x)', 'x*sin(3*x)'].forEach((p) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = p; b.onclick = () => ei.set(p); q('#presets').appendChild(b); });
  ei.run();
  q('#shrink').onclick = () => { q('#h').value = 0.4; shrinking = true; };
  q('#h').oninput = () => (shrinking = false);

  AI.loop(root, (dt) => {
    if (shrinking) { const v = +q('#h').value - dt * 0.22; if (v <= -3) shrinking = false; q('#h').value = Math.max(-3, v); }
    const h = Math.pow(10, +q('#h').value), x0 = +q('#x0').value, C = AI.col(), g = c.ctx;
    q('#xv').textContent = AI.fmt(x0, 2); q('#hv').textContent = AI.fmt(h, 4);
    // y-range from samples
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 120; i++) { const y = fn({ x: -6 + (12 * i) / 120 }); if (Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    lo = Math.max(lo, -40); hi = Math.min(hi, 40); const pd = (hi - lo) * 0.15 + 0.5;
    yr[0] += (lo - pd - yr[0]) * 0.15; yr[1] += (hi + pd - yr[1]) * 0.15;
    const v = AI.view(c, -6, 6, yr[0], yr[1]);
    g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
    const plot = (f, col, w, dash) => {
      g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath(); let pen = false;
      for (let i = 0; i <= 300; i++) { const x = -6 + (12 * i) / 300, y = f(x); if (!Number.isFinite(y)) { pen = false; continue; } pen ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); pen = true; }
      g.stroke(); g.setLineDash([]);
    };
    const d = (x) => AI.autodiff(ast, { x }).varGrad.x ?? 0;
    plot((x) => fn({ x }), C.blue, 2.5);
    if (q('#showd').checked) plot(d, C.green, 1.5, [5, 4]);
    const f0 = fn({ x: x0 }), f1 = fn({ x: x0 + h }), sec = (f1 - f0) / h, tan = d(x0);
    const ext = (m, col, w, dash) => {
      g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
      g.moveTo(v.X(-6), v.Y(f0 + m * (-6 - x0))); g.lineTo(v.X(6), v.Y(f0 + m * (6 - x0))); g.stroke(); g.setLineDash([]);
    };
    if (Number.isFinite(sec)) ext(sec, C.orange, 2);
    if (Number.isFinite(tan)) ext(tan, C.red, 1.5, [6, 4]);
    g.fillStyle = C.fg; for (const [x, y] of [[x0, f0], [x0 + h, f1]]) { g.beginPath(); g.arc(v.X(x), v.Y(y), 5, 0, 7); g.fill(); }
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
    g.fillText('— f(x)   — secant   - - tangent (autodiff)   - - f′(x)', v.pad.l + 8, v.pad.t + 12);
    q('#out').innerHTML = `f(x₀)         = ${AI.fmt(f0, 4)}\nsecant slope  = <span class="o">${AI.fmt(sec, 5)}</span>\n  [f(x₀+h) − f(x₀)] / h\ntrue f′(x₀)   = <span class="r">${AI.fmt(tan, 5)}</span>\ngap           = ${AI.fmt(Math.abs(sec - tan), 5)}`;
  });
};

/* ================= Optimization ================= */
AI.labs.descent = function (root) {
  const { view, ctl } = AI.labShell(root, 'Optimizer race on a loss surface', 'click the map to choose a start; edit f(x, y) freely',
    `<canvas style="height:420px;cursor:crosshair"></canvas>`,
    `<label>Loss f(x, y) = <input type="text" id="expr" value="x^2 + 3*y^2" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <label>Learning rate η <span class="v" id="lrv"></span><input type="range" id="lr" min="-3.5" max="-0.3" step="0.05" value="-1.5"></label>
     <label>Speed <span class="v">steps / second</span><input type="range" id="sp" min="1" max="60" step="1" value="6"></label>
     <div class="legend"><span><i style="background:var(--blue)"></i>SGD</span><span><i style="background:var(--green)"></i>Momentum</span><span><i style="background:var(--orange)"></i>Adam</span></div>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Restart</button></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c = AI.canvas(AI.$('canvas', view));
  const PRE = {
    Bowl: { e: 'x^2 + 3*y^2', v: [-3, 3, -3, 3], s: [2.5, 2], lr: -1.3 },
    Valley: { e: '(1-x)^2 + 10*(y-x^2)^2', v: [-2, 2, -1, 3], s: [-1.5, 2.2], lr: -2.8 },
    Himmelblau: { e: '(x^2+y-11)^2 + (x+y^2-7)^2', v: [-5, 5, -5, 5], s: [0, 0], lr: -2.6 },
    Ripples: { e: '0.3*(x^2+y^2) + 2*sin(2*x)*sin(2*y)', v: [-3, 3, -3, 3], s: [2.2, 1.8], lr: -1.5 },
    Saddle: { e: 'x^2 - y^2 + 0.05*(x^4 + y^4)', v: [-3, 3, -3, 3], s: [2, 0.05], lr: -1.5 },
  };
  const OPT = [['SGD', 'blue'], ['Momentum', 'green'], ['Adam', 'orange']];
  const tickD = AI.ticker();
  let fn = null, view2 = PRE.Bowl.v, start = [...PRE.Bowl.s], runs = [], playing = true, img = null, imgDirty = true, range = [0, 1], t = 0;
  const off = document.createElement('canvas');

  const reset = () => { runs = OPT.map(([n, col]) => ({ n, col, p: [...start], v: [0, 0], m: [0, 0], s: [0, 0], path: [[...start]], dead: false, k: 0 })); t = 0; };
  const grad = (p) => { const h = 1e-4; return [(fn({ x: p[0] + h, y: p[1] }) - fn({ x: p[0] - h, y: p[1] })) / (2 * h), (fn({ x: p[0], y: p[1] + h }) - fn({ x: p[0], y: p[1] - h })) / (2 * h)]; };
  const stepRun = (r, lr) => {
    if (r.dead) return;
    const g = grad(r.p); r.k++;
    for (let i = 0; i < 2; i++) {
      if (r.n === 'SGD') r.p[i] -= lr * g[i];
      else if (r.n === 'Momentum') { r.v[i] = 0.9 * r.v[i] + g[i]; r.p[i] -= lr * r.v[i]; }
      else { r.m[i] = 0.9 * r.m[i] + 0.1 * g[i]; r.s[i] = 0.999 * r.s[i] + 0.001 * g[i] * g[i]; r.p[i] -= (lr * 5 * (r.m[i] / (1 - 0.9 ** r.k))) / (Math.sqrt(r.s[i] / (1 - 0.999 ** r.k)) + 1e-8); }
    }
    if (!r.p.every(Number.isFinite) || Math.abs(r.p[0]) > 1e3 || Math.abs(r.p[1]) > 1e3) { r.dead = true; return; }
    r.path.push([...r.p]); if (r.path.length > 400) r.path.shift();
  };
  const ei = AI.exprInput(q('#expr'), q('#err'), {
    allowed: ['x', 'y'], check: (f) => { if (!Number.isFinite(f({ x: 0.5, y: 0.5 }))) throw new Error('f(x, y) is not finite'); },
    onChange: (f) => { fn = f; imgDirty = true; reset(); },
  });
  Object.entries(PRE).forEach(([k, p]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => { view2 = p.v; start = [...p.s]; q('#lr').value = p.lr; ei.set(p.e); }; q('#presets').appendChild(b); });
  q('#reset').onclick = reset;
  q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
  AI.$('canvas', view).addEventListener('click', (e) => {
    const r = e.target.getBoundingClientRect(), [x0, x1, y0, y1] = view2;
    start = [x0 + ((e.clientX - r.left) / r.width) * (x1 - x0), y1 - ((e.clientY - r.top) / r.height) * (y1 - y0)]; reset();
  });
  ei.run();

  const paint = () => {
    const W = 96, H = Math.max(40, Math.round((96 * c.h) / c.w)); off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H), vals = new Float64Array(W * H), [x0, x1, y0, y1] = view2;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) vals[j * W + i] = fn({ x: x0 + ((i + 0.5) / W) * (x1 - x0), y: y1 - ((j + 0.5) / H) * (y1 - y0) });
    const sorted = [...vals].filter(Number.isFinite).sort((a, b) => a - b);
    const lo = sorted[0] ?? 0, hi = AI.quantile(sorted, 0.95) || lo + 1, C = AI.col(), A = AI.rgb(C.bg), B = AI.rgb(C.purple), Cc = AI.rgb(C.orange);
    for (let k = 0; k < vals.length; k++) {
      const tt = AI.clamp(Math.pow(AI.clamp((vals[k] - lo) / (hi - lo || 1), 0, 1), 0.45), 0, 1), band = (tt * 14) % 1, edge = band < 0.12 ? 0.18 : 0;
      for (let ch = 0; ch < 3; ch++) im.data[k * 4 + ch] = Math.min(255, (tt < 0.5 ? A[ch] + (B[ch] - A[ch]) * tt * 2 : B[ch] + (Cc[ch] - B[ch]) * (tt - 0.5) * 2) * (0.85 + edge) + edge * 60);
      im.data[k * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0); imgDirty = false;
  };
  window.addEventListener('themechange', () => (imgDirty = true));

  AI.loop(root, (dt) => {
    const lr = Math.pow(10, +q('#lr').value); q('#lrv').textContent = AI.fmt(lr, 4);
    if (imgDirty) paint();
    if (playing) { const n = tickD(dt, +q('#sp').value); for (let i = 0; i < n; i++) runs.forEach((r) => stepRun(r, lr)); }
    const C = AI.col(), g = c.ctx, [x0, x1, y0, y1] = view2;
    g.clearRect(0, 0, c.w, c.h); g.drawImage(off, 0, 0, c.w, c.h);
    const X = (x) => ((x - x0) / (x1 - x0)) * c.w, Y = (y) => ((y1 - y) / (y1 - y0)) * c.h;
    g.fillStyle = C.fg; g.beginPath(); g.arc(X(start[0]), Y(start[1]), 5, 0, 7); g.fill();
    let txt = '';
    for (const r of runs) {
      g.strokeStyle = C[r.col]; g.lineWidth = 2.5; g.beginPath();
      r.path.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.stroke();
      if (r.path.length < 160) { g.fillStyle = C[r.col]; r.path.forEach(([x, y]) => { g.beginPath(); g.arc(X(x), Y(y), 2.2, 0, 7); g.fill(); }); } // one dot per step
      const [px, py] = r.p;
      g.fillStyle = C[r.col]; g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(X(px), Y(py), 6, 0, 7); g.fill(); g.stroke();
      if (!r.dead) { const gr = grad(r.p), n = Math.hypot(...gr) || 1; AI.arrow(g, X(px), Y(py), X(px) - (gr[0] / n) * 26, Y(py) + (gr[1] / n) * 26, C[r.col], 2); }
      const fv = fn({ x: px, y: py });
      txt += `<span class="${r.col === 'blue' ? 'b' : r.col === 'green' ? 'g' : 'o'}">${r.n.padEnd(9)}</span> f=${r.dead ? '<span class="r">diverged</span>' : AI.fmt(fv, 4)}  (${AI.fmt(px, 2)}, ${AI.fmt(py, 2)})\n`;
    }
    g.fillStyle = C.fg; g.font = '11px sans-serif'; g.fillText('each dot = one step; arrow = −∇f (downhill)', 10, 16);
    q('#out').innerHTML = txt + '<span class="m">Adam uses 5×η here so it is visible at the same scale.</span>';
  });
};

/* ================= Backprop graph ================= */
AI.labs.backprop = function (root) {
  const { view, ctl } = AI.labShell(root, 'Computational graph: forward then backward', 'edit the expression, set the inputs, watch values and gradients flow',
    `<svg id="svg" style="width:100%;min-height:300px;background:var(--bg);border-radius:8px"></svg><div class="caption" id="cap">Press “Forward + backward”.</div>`,
    `<label>f = <input type="text" id="expr" value="sin(x*y) + tanh(z)*x" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <div id="vars" class="row"></div>
     <div class="row"><button class="btn primary" id="go">▶ Forward + backward</button><button class="btn" id="fw">Forward</button><button class="btn" id="bw">Backward</button></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const svg = q('#svg'), NS = 'http://www.w3.org/2000/svg';
  let ast = null, vals = { x: 0.5, y: -1, z: 2 }, res = null, nodes = [], token = 0, showV = new Set(), showG = new Set();

  const label = (n) => n.t === 'num' ? AI.fmt(n.v, 3) : n.t === 'var' ? n.n : n.t === 'neg' ? '−' : n.t === 'fn' || n.t === 'fn2' ? n.n : n.op === '*' ? '×' : n.op === '-' ? '−' : n.op;
  const layout = () => {
    nodes = []; let leaf = 0, maxD = 0;
    const walk = (n, d, par) => {
      const o = { n, d, par, kids: [] }; maxD = Math.max(maxD, d);
      const kids = [n.a, n.b].filter(Boolean);
      kids.forEach((k) => o.kids.push(walk(k, d + 1, o)));
      o.x = kids.length ? o.kids.reduce((s, k) => s + k.x, 0) / kids.length : leaf++;
      o.h = kids.length ? 1 + Math.max(...o.kids.map((k) => k.h)) : 0;
      nodes.push(o); return o;
    };
    const root0 = walk(ast, 0, null);
    return { root0, leaves: leaf, maxD };
  };
  let L = null;
  const draw = () => {
    L = layout();
    const W = Math.max(320, L.leaves * 96 + 40), H = (L.maxD + 1) * 92 + 50;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.style.height = Math.min(520, Math.max(300, (H * svg.clientWidth) / W)) + 'px'; svg.innerHTML = '';
    const P = (o) => [30 + 28 + o.x * 96 + (W - 60 - L.leaves * 96) / 2 - 0, 40 + o.d * 92];
    const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
    nodes.forEach((o) => { o.pos = P(o); });
    nodes.forEach((o) => o.kids.forEach((k) => { o.edge = o.edge || []; const e = mk('line', { x1: o.pos[0], y1: o.pos[1], x2: k.pos[0], y2: k.pos[1], stroke: 'var(--border)', 'stroke-width': 2 }); k.edgeEl = e; }));
    nodes.forEach((o) => {
      const [x, y] = o.pos, leafN = !o.kids.length;
      o.circ = mk('circle', { cx: x, cy: y, r: 20, fill: 'var(--bg2)', stroke: leafN ? 'var(--blue)' : 'var(--purple)', 'stroke-width': 2 });
      const t = mk('text', { x, y: y + 5, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': 14, 'font-family': 'ui-monospace,Menlo,monospace' }); t.textContent = label(o.n);
      o.vt = mk('text', { x, y: y - 27, 'text-anchor': 'middle', fill: 'var(--blue)', 'font-size': 11, 'font-family': 'ui-monospace,Menlo,monospace', opacity: 0 });
      o.gt = mk('text', { x, y: y + 38, 'text-anchor': 'middle', fill: 'var(--orange)', 'font-size': 11, 'font-family': 'ui-monospace,Menlo,monospace', opacity: 0 });
    });
    refresh();
  };
  const refresh = () => {
    try { res = AI.autodiff(ast, vals); } catch (e) { return; }
    nodes.forEach((o) => {
      o.vt.textContent = AI.fmt(res.val.get(o.n.id), 3); o.gt.textContent = '∂ ' + AI.fmt(res.grad.get(o.n.id), 3);
      o.vt.setAttribute('opacity', showV.has(o) ? 1 : 0); o.gt.setAttribute('opacity', showG.has(o) ? 1 : 0);
      o.circ.setAttribute('fill', showV.has(o) ? 'var(--bg3)' : 'var(--bg2)');
    });
    const rows = Object.keys(res.varGrad).map((k) => {
      const h = 1e-5, f = (d) => AI.compile(ast)({ ...vals, [k]: vals[k] + d }), num = (f(h) - f(-h)) / (2 * h);
      return `<span class="o">∂f/∂${k}</span> = ${AI.fmt(res.varGrad[k], 5)}   <span class="m">numeric ${AI.fmt(num, 5)} ${Math.abs(num - res.varGrad[k]) < 1e-4 * (1 + Math.abs(num)) ? '✓' : '✗'}</span>`;
    });
    q('#out').innerHTML = `<span class="b">f</span> = ${AI.fmt(res.value, 5)}\n` + rows.join('\n');
  };
  const rule = (o, back) => {
    const n = o.n, v = (k) => AI.fmt(res.val.get(k.id), 3), g = AI.fmt(res.grad.get(n.id), 3);
    if (!back) return o.kids.length ? `Forward: ${label(n)} of (${o.kids.map((k) => v(k.n)).join(', ')}) = ${v(n)}` : `Input ${label(n)} = ${v(n)}`;
    if (!o.kids.length) return `Leaf ${label(n)} receives ∂f/∂${label(n)} contribution ${g}`;
    const what = n.t === 'bin' ? ({ '+': 'local derivative 1 for both inputs', '-': 'local derivatives +1 and −1', '*': 'each input gets grad × the other input', '/': 'quotient rule', '^': 'power rule' })[n.op] : 'chain rule: grad × f′(input)';
    return `Backward at ${label(n)}: incoming ∂ = ${g}; ${what}`;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const forward = async (tk) => {
    showV = new Set(); showG = new Set(); refresh();
    for (const o of [...nodes].sort((a, b) => a.h - b.h)) {
      if (tk !== token) return false;
      showV.add(o); if (o.edgeEl) o.edgeEl.setAttribute('stroke', 'var(--blue)');
      refresh(); q('#cap').textContent = rule(o, false); await sleep(1100);
    }
    return true;
  };
  const backward = async (tk) => {
    showG = new Set(); nodes.forEach((o) => o.edgeEl && o.edgeEl.setAttribute('stroke', 'var(--border)')); refresh();
    for (const o of [...nodes].sort((a, b) => a.d - b.d)) {
      if (tk !== token) return false;
      showG.add(o); if (o.edgeEl) o.edgeEl.setAttribute('stroke', 'var(--orange)');
      refresh(); q('#cap').textContent = rule(o, true); await sleep(1100);
    }
    return true;
  };
  q('#go').onclick = async () => { const tk = ++token; if (await forward(tk)) await backward(tk); };
  q('#fw').onclick = () => { forward(++token); };
  q('#bw').onclick = () => { const tk = ++token; showV = new Set(nodes); refresh(); backward(tk); };

  const ei = AI.exprInput(q('#expr'), q('#err'), {
    check: (f, a) => { if (AI.vars(a).size > 5) throw new Error('Use at most 5 variables'); },
    onChange: (f, a) => {
      token++; ast = a; showV = new Set(); showG = new Set();
      const names = [...AI.vars(a)], box = q('#vars'); box.innerHTML = '';
      names.forEach((k) => {
        if (!(k in vals)) vals[k] = 1;
        const lab = document.createElement('label'); lab.style.flex = '1'; lab.style.minWidth = '70px'; lab.textContent = k;
        const inp = document.createElement('input'); inp.type = 'number'; inp.step = 0.1; inp.value = vals[k];
        inp.oninput = () => { vals[k] = +inp.value || 0; refresh(); };
        lab.appendChild(inp); box.appendChild(lab);
      });
      draw(); q('#cap').textContent = 'Press “Forward + backward”.';
    },
  });
  ['(x + y) * z', 'sin(x*y) + tanh(z)*x', 'sigmoid(w*x + b)', 'x^2 * y + exp(x)', '(w*x - t)^2'].forEach((p) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = p; b.onclick = () => ei.set(p); q('#presets').appendChild(b); });
  ei.run();
  new ResizeObserver(() => ast && draw()).observe(svg.parentElement);
};

/* ================= Sampling & CLT ================= */
AI.labs.sampling = function (root) {
  const { view, ctl } = AI.labShell(root, 'Sampling lab: build a distribution from uniform noise', 'edit the formula, average n draws, watch the bell curve appear',
    `<canvas style="height:330px"></canvas>`,
    `<label>Sample = g(u, v)  <span class="m" style="color:var(--muted)">u, v ~ Uniform(0,1)</span><input type="text" id="expr" value="-log(1-u)" spellcheck="false"></label>
     <div class="err" id="err"></div>
     <div class="chips" id="presets"></div>
     <label>Average n draws <span class="v" id="nv"></span><input type="range" id="n" min="1" max="60" step="1" value="1"></label>
     <label>Samples per second <input type="range" id="rate" min="5" max="300" step="5" value="30"></label>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Clear</button></div>
     <div class="readout" id="out"></div>`);
  const q = (s) => AI.$(s, root);
  const c = AI.canvas(AI.$('canvas', view));
  const BINS = 40, U = () => Math.random() * (1 - 2e-12) + 1e-12;
  let fn = null, samples = [], lo = null, hi = null, counts = new Array(BINS).fill(0), drops = [], playing = true, acc = 0, lastN = 1;
  const presets = { Uniform: 'u', Exponential: '-log(1-u)', 'Normal (Box–Muller)': 'sqrt(-2*log(u))*cos(2*pi*v)', Skewed: 'u^4', 'Cauchy (heavy tail)': 'tan(pi*(u-0.5))' };
  const clear = () => { samples = []; counts.fill(0); lo = hi = null; drops = []; };
  const draw1 = () => { const n = +q('#n').value; let s = 0; for (let i = 0; i < n; i++) s += fn({ u: U(), v: U() }); return s / n; };
  const bin = (x) => Math.floor(((x - lo) / (hi - lo)) * BINS);
  const ei = AI.exprInput(q('#expr'), q('#err'), {
    allowed: ['u', 'v'], check: (f) => { if (!Number.isFinite(f({ u: 0.3, v: 0.6 }))) throw new Error('Not finite for u=0.3, v=0.6'); },
    onChange: (f) => { fn = f; clear(); },
  });
  Object.entries(presets).forEach(([k, p]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => ei.set(p); q('#presets').appendChild(b); });
  q('#reset').onclick = clear;
  q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
  q('#n').oninput = clear;
  ei.run();

  AI.loop(root, (dt) => {
    q('#nv').textContent = q('#n').value;
    if (playing) {
      acc += dt * +q('#rate').value;
      while (acc >= 1) {
        acc--; const x = draw1(); if (!Number.isFinite(x)) continue; samples.push(x);
        if (lo === null && samples.length >= 150) {
          const s = [...samples].sort((a, b) => a - b); lo = AI.quantile(s, 0.01); hi = AI.quantile(s, 0.99); if (lo === hi) { lo -= 1; hi += 1; }
          const pd = (hi - lo) * 0.1; lo -= pd; hi += pd; counts.fill(0); samples.forEach((v) => { const b = bin(v); if (b >= 0 && b < BINS) counts[b]++; });
        } else if (lo !== null) { const b = bin(x); if (b >= 0 && b < BINS) { counts[b]++; drops.push({ x, y: 0, b }); } }
      }
    }
    const C = AI.col(), g = c.ctx, pad = { l: 12, r: 12, t: 16, b: 26 };
    g.clearRect(0, 0, c.w, c.h);
    if (lo === null) { g.fillStyle = C.muted; g.font = '13px sans-serif'; g.textAlign = 'center'; g.fillText(`collecting samples… ${samples.length}/150`, c.w / 2, c.h / 2); return; }
    const mx = Math.max(...counts, 1), v = AI.view(c, lo, hi, 0, mx * 1.15, pad), bw = v.pw / BINS;
    AI.axes(g, v, { noY: true, noGrid: true });
    g.fillStyle = C.blue; g.globalAlpha = 0.85;
    counts.forEach((n, i) => { const h = (n / (mx * 1.15)) * v.ph; g.fillRect(v.pad.l + i * bw + 1, v.pad.t + v.ph - h, bw - 2, h); });
    g.globalAlpha = 1;
    const st = AI.stats(samples);
    // gaussian overlay with the same mean / std
    g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath();
    const tot = counts.reduce((a, b) => a + b, 0), binW = (hi - lo) / BINS;
    for (let i = 0; i <= 200; i++) { const x = lo + ((hi - lo) * i) / 200, y = tot * binW * Math.exp(-0.5 * ((x - st.mean) / st.std) ** 2) / (st.std * Math.sqrt(2 * Math.PI)); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
    g.stroke();
    // falling drops
    drops.forEach((d) => (d.y += dt * 300));
    for (const d of drops) { const ty = v.Y(counts[d.b] ?? 0); const y = Math.min(d.y, ty); g.fillStyle = C.fg; g.globalAlpha = Math.max(0, 1 - (d.y - ty + 40) / 60); g.beginPath(); g.arc(v.X(d.x), y + 6, 2.5, 0, 7); g.fill(); }
    g.globalAlpha = 1; drops = drops.filter((d) => d.y < v.Y(counts[d.b] ?? 0) + 30);
    g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('■ histogram of samples   — Gaussian with the same mean & std', v.pad.l + 4, 12);
    q('#out').innerHTML = `samples  ${samples.length}\nmean     <b>${AI.fmt(st.mean, 3)}</b>\nstd      <b>${AI.fmt(st.std, 3)}</b>\nskew     <span class="o">${AI.fmt(st.skew, 3)}</span> <span class="m">(0 = symmetric)</span>\nex.kurt  <span class="o">${AI.fmt(st.kurt, 3)}</span> <span class="m">(0 = bell-like)</span>\n<span class="m">Raise n: skew and kurtosis shrink toward 0 — except for Cauchy.</span>`;
  });
};
