/* Module 7 labs — the mathematics of neural networks.
   neuron, universal, softmaxstab, losses, mlpback, shapes, gradflow, attention, attnscale, posenc.
   Conventions: everything animates in real time (dt); every animated lab has a speed slider and pause/step. */
(function () {
  const $ = (s, r) => AI.$(s, r);
  const sl = (id, label, min, max, step, val) => `<label>${label}<span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const wire = (root, ids, cb) => ids.forEach((id) => {
    const el = $('#' + id, root), upd = () => { const v = $('#' + id + 'v', root); if (v) v.textContent = AI.fmt(+el.value, 2); };
    el.addEventListener('input', () => { upd(); cb(id); }); upd();
  });
  const txt = (g, s, x, y, col, al = 'left', font = '12px ui-monospace,Menlo,Consolas,monospace') => { g.fillStyle = col; g.textAlign = al; g.font = font; g.fillText(s, x, y); };
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const lerpc = (A, B, t) => `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
  const ease = (dt, k) => 1 - Math.exp(-dt * k);
  const ACT = {
    step: { f: (z) => (z > 0 ? 1 : 0), lo: 0, hi: 1 },
    sigmoid: { f: sig, lo: 0, hi: 1 },
    tanh: { f: Math.tanh, lo: -1, hi: 1 },
    relu: { f: (z) => Math.max(0, z), lo: 0, hi: 4 },
  };
  const playRow = (extra = '') => `<div class="row"><button class="btn primary" data-r="play">⏸ Pause</button><button class="btn" data-r="back">◀ Step</button><button class="btn" data-r="step">Step ▶</button><button class="btn" data-r="reset">↺ Restart</button>${extra}</div>`;

  /* ================= 7.1  a single neuron ================= */
  AI.labs.neuron = function (root) {
    const { view, ctl } = AI.labShell(root, 'One neuron: a = φ(w₁x₁ + w₂x₂ + b)', 'weights tilt the boundary, the bias slides it; click the map to place the probe',
      `<canvas style="height:380px;cursor:crosshair"></canvas>`,
      `${sl('w1', 'weight w₁', -3, 3, 0.1, 1.5)}${sl('w2', 'weight w₂', -3, 3, 0.1, 1)}${sl('b', 'bias b', -3, 3, 0.1, 0)}
       <label>Activation φ<select id="act"><option value="step">step (perceptron)</option><option value="sigmoid" selected>sigmoid</option><option value="tanh">tanh</option><option value="relu">ReLU</option></select></label>
       <div class="row"><button class="btn" id="auto">▶ Move probe</button></div>${sl('sp', 'probe speed (× slow)', 0.2, 3, 0.1, 1)}
       <div class="readout" id="out"></div>`);
    const c = AI.canvas($('canvas', view)), off = document.createElement('canvas');
    off.width = 150; off.height = 100;
    let S = {}, probe = [1.2, 0.8], auto = false, t = 0, dirty = true;
    const read = () => { S = { w1: +$('#w1', root).value, w2: +$('#w2', root).value, b: +$('#b', root).value, act: $('#act', root).value }; dirty = true; };
    wire(root, ['w1', 'w2', 'b', 'sp'], read); $('#act', root).onchange = read; read();
    $('#auto', root).onclick = () => { auto = !auto; $('#auto', root).textContent = auto ? '⏸ Stop probe' : '▶ Move probe'; };
    c.onresize = () => (dirty = true);
    const ext = 3;
    $('canvas', view).addEventListener('click', (e) => {
      const r = e.target.getBoundingClientRect(), ex = ext * c.w / c.h;
      probe = [-ex + ((e.clientX - r.left) / r.width) * 2 * ex, ext - ((e.clientY - r.top) / r.height) * 2 * ext]; auto = false; $('#auto', root).textContent = '▶ Move probe'; dirty = true;
    });
    window.addEventListener('themechange', () => (dirty = true));
    AI.loop(root, (dt) => {
      if (auto) { t += dt * +$('#sp', root).value * 0.35; probe = [2.4 * Math.cos(t), 1.9 * Math.sin(0.73 * t)]; dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, ex = ext * c.w / c.h, A = ACT[S.act], cA = AI.rgb(C.bg3), cB = AI.rgb(C.orange), cL = AI.rgb(C.blue);
      const ig = off.getContext('2d'), im = ig.createImageData(150, 100);
      for (let j = 0; j < 100; j++) for (let i = 0; i < 150; i++) {
        const x = -ex + ((i + 0.5) / 150) * 2 * ex, y = ext - ((j + 0.5) / 100) * 2 * ext;
        const v = A.f(S.w1 * x + S.w2 * y + S.b), tt = AI.clamp((v - A.lo) / (A.hi - A.lo), 0, 1), k = (j * 150 + i) * 4;
        const col = tt < 0.5 ? [cL, cA, tt * 2] : [cA, cB, (tt - 0.5) * 2];
        for (let ch = 0; ch < 3; ch++) im.data[k + ch] = col[0][ch] + (col[1][ch] - col[0][ch]) * col[2];
        im.data[k + 3] = 255;
      }
      ig.putImageData(im, 0, 0);
      g.clearRect(0, 0, c.w, c.h); g.imageSmoothingEnabled = S.act !== 'step'; g.drawImage(off, 0, 0, c.w, c.h);
      const X = (x) => ((x + ex) / (2 * ex)) * c.w, Y = (y) => ((ext - y) / (2 * ext)) * c.h;
      g.strokeStyle = 'rgba(128,128,128,.45)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(X(-ex), Y(0)); g.lineTo(X(ex), Y(0)); g.moveTo(X(0), Y(-ext)); g.lineTo(X(0), Y(ext)); g.stroke();
      txt(g, 'x₁', c.w - 8, Y(0) - 6, C.fg, 'right'); txt(g, 'x₂', X(0) + 6, 14, C.fg);
      const n2 = S.w1 * S.w1 + S.w2 * S.w2;
      if (n2 > 1e-9) {
        const L = Math.sqrt(n2), ux = S.w1 / L, uy = S.w2 / L, px = (-S.b * S.w1) / n2, py = (-S.b * S.w2) / n2;
        g.strokeStyle = C.fg; g.lineWidth = 2.5; g.setLineDash([7, 5]); g.beginPath(); g.moveTo(X(px - uy * 20), Y(py + ux * 20)); g.lineTo(X(px + uy * 20), Y(py - ux * 20)); g.stroke(); g.setLineDash([]);
        AI.arrow(g, X(px), Y(py), X(px + ux * 1.2), Y(py + uy * 1.2), C.green, 3);
        txt(g, 'w (points to higher z)', X(px + ux * 1.25) + 4, Y(py + uy * 1.25), C.green);
      }
      const z = S.w1 * probe[0] + S.w2 * probe[1] + S.b, a = A.f(z);
      g.fillStyle = '#fff'; g.strokeStyle = C.bg; g.lineWidth = 2; g.beginPath(); g.arc(X(probe[0]), Y(probe[1]), 7, 0, 7); g.fill(); g.stroke();
      txt(g, 'dashed line: z = 0 (decision boundary). colour = neuron output a', 10, c.h - 10, C.fg);
      $('#out', root).innerHTML = `probe x = (${AI.fmt(probe[0], 2)}, ${AI.fmt(probe[1], 2)})\n<span class="b">z = w·x + b</span> = (${AI.fmt(S.w1, 2)})(${AI.fmt(probe[0], 2)}) + (${AI.fmt(S.w2, 2)})(${AI.fmt(probe[1], 2)}) + ${AI.fmt(S.b, 2)}\n    = <span class="o">${AI.fmt(z, 3)}</span>\n<span class="g">a = φ(z)</span> = <span class="o">${AI.fmt(a, 3)}</span>\nsigned distance to boundary = z/‖w‖ = ${n2 > 1e-9 ? AI.fmt(z / Math.sqrt(n2), 3) : '–'}`;
    });
  };

  /* ================= 7.1  universal approximation ================= */
  AI.labs.universal = function (root) {
    const { view, ctl } = AI.labShell(root, 'Stacking neurons to approximate any curve', 'hidden neurons = sigmoid steps or ReLU ramps; the output layer just adds them up',
      `<canvas style="height:340px"></canvas><div class="caption" id="cap" style="font-size:12px;color:var(--muted);margin-top:6px"></div>`,
      `<label>Target f(x) = <input type="text" id="expr" value="sin(2*x) + 0.4*x" spellcheck="false"></label><div class="err" id="err"></div>
       <div class="chips" id="pre"></div>
       <label>Hidden neuron type<select id="kind"><option value="sig">sigmoid steps</option><option value="relu">ReLU ramps</option></select></label>
       ${sl('N', 'number of hidden neurons N', 1, 40, 1, 8)}${sl('k', 'step steepness k (sigmoid only)', 2, 60, 1, 14)}
       <div class="row"><button class="btn primary" id="build">▶ Build neuron by neuron</button></div>${sl('sp', 'build speed (neurons / s)', 0.5, 6, 0.5, 1.5)}
       <div class="readout" id="out"></div>`);
    const c = AI.canvas($('canvas', view)), tick = AI.ticker();
    c.onresize = () => (dirty = true);
    let fn = null, dirty = true, shown = 8, building = false, acc = 0, kind = 'sig';
    const q = (s) => $(s, root);
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], check: (f) => { if (!Number.isFinite(f({ x: 0.3 }))) throw new Error('f must be finite on [-3, 3]'); }, onChange: (f) => { fn = f; dirty = true; } });
    [['sin wave', 'sin(2*x) + 0.4*x'], ['bump', 'exp(-x^2)*3'], ['sawtooth-ish', 'abs(x) - 1.5'], ['wiggly', 'sin(3*x)*cos(x)*2'], ['square-ish', 'tanh(6*sin(x))*2']].forEach(([n, e]) => {
      const b = document.createElement('button'); b.className = 'chip'; b.textContent = n; b.onclick = () => ei.set(e); q('#pre').appendChild(b);
    });
    ei.run();
    wire(root, ['N', 'k', 'sp'], (id) => { if (id === 'N') { shown = +q('#N').value; building = false; q('#build').textContent = '▶ Build neuron by neuron'; } dirty = true; });
    q('#kind').onchange = () => { kind = q('#kind').value; dirty = true; };
    q('#build').onclick = () => { if (building) { building = false; q('#build').textContent = '▶ Build neuron by neuron'; return; } shown = 1; acc = 0; building = true; q('#build').textContent = '⏸ Stop'; dirty = true; };
    window.addEventListener('themechange', () => (dirty = true));
    const approx = (n, k, kind0) => { // returns {p, coef, evalAt(x), parts(x)}
      const p = []; for (let i = 0; i <= n; i++) p.push(-3 + (6 * i) / n);
      const fv = p.map((x) => fn({ x }));
      const h = 6 / n, parts = [];
      if (kind0 === 'sig') for (let i = 1; i <= n; i++) parts.push({ c: fv[i] - fv[i - 1], t: (p[i - 1] + p[i]) / 2, ph: (x) => sig(k * (x - (p[i - 1] + p[i]) / 2)) });
      else { let prev = 0; for (let i = 1; i <= n; i++) { const s = (fv[i] - fv[i - 1]) / h; parts.push({ c: s - prev, t: p[i - 1], ph: (x) => Math.max(0, x - p[i - 1]) }); prev = s; } }
      return { base: fv[0], parts };
    };
    AI.loop(root, (dt) => {
      if (building) { acc += dt * +q('#sp').value; while (acc >= 1) { acc -= 1; if (shown < +q('#N').value) { shown++; dirty = true; } else { building = false; q('#build').textContent = '▶ Build neuron by neuron'; } } }
      if (!dirty || !fn) return; dirty = false;
      const C = AI.col(), g = c.ctx, k = +q('#k').value, A = approx(shown, k, kind);
      const xs = []; for (let i = 0; i <= 240; i++) xs.push(-3 + (6 * i) / 240);
      const tv = xs.map((x) => fn({ x })), yv = xs.map((x) => A.base + A.parts.reduce((s, p) => s + p.c * p.ph(x), 0));
      let lo = Math.min(...tv), hi = Math.max(...tv); const pad = (hi - lo) * 0.25 + 0.2; lo -= pad; hi += pad;
      const v = AI.view(c, -3, 3, lo, hi);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.lineWidth = 1.2;
      A.parts.forEach((p, i) => { g.strokeStyle = AI.mix(C.purple, C.bg, 0.45); g.beginPath(); xs.forEach((x, j) => { const y = A.base + p.c * p.ph(x); j ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }); g.stroke(); });
      g.lineWidth = 3; g.setLineDash([6, 5]); g.strokeStyle = C.fg; g.beginPath(); xs.forEach((x, j) => (j ? g.lineTo(v.X(x), v.Y(tv[j])) : g.moveTo(v.X(x), v.Y(tv[j])))); g.stroke(); g.setLineDash([]);
      g.lineWidth = 3; g.strokeStyle = C.blue; g.beginPath(); xs.forEach((x, j) => (j ? g.lineTo(v.X(x), v.Y(yv[j])) : g.moveTo(v.X(x), v.Y(yv[j])))); g.stroke();
      A.parts.forEach((p) => { g.fillStyle = C.purple; g.beginPath(); g.arc(v.X(p.t), v.Y(lo + (hi - lo) * 0.03), 3, 0, 7); g.fill(); });
      txt(g, 'dashed: target   blue: network   faint: each neuron × its output weight   dots: switch-on points', v.pad.l + 4, 14, C.fg, 'left', '11px sans-serif');
      let se = 0, mx = 0; xs.forEach((x, j) => { const e = Math.abs(yv[j] - tv[j]); se += e * e; mx = Math.max(mx, e); });
      q('#out').innerHTML = `hidden neurons N = <span class="o">${shown}</span>   (type: ${kind === 'sig' ? 'sigmoid' : 'ReLU'})\nparameters = 3N + 1 = ${3 * shown + 1}\nRMS error = <span class="o">${AI.fmt(Math.sqrt(se / xs.length), 4)}</span>\nmax error = ${AI.fmt(mx, 4)}`;
      q('#cap').textContent = kind === 'sig' ? 'Each sigmoid neuron is a smoothed step: it adds a jump of height f(pᵢ) − f(pᵢ₋₁) where the curve moves. Many small steps trace the curve.' : 'Each ReLU neuron is a ramp that bends the line at one knot. Bending the line at N knots gives a piecewise-linear curve.';
    });
  };

  /* ================= 7.2  softmax & numerical stability ================= */
  AI.labs.softmaxstab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Softmax: logits → probabilities (and why we subtract the max)', 'temperature reshapes the distribution; the shift c breaks the naive formula',
      `<canvas style="height:250px"></canvas><div class="readout" id="out" style="margin-top:10px;white-space:pre-wrap"></div>`,
      `${sl('z1', 'logit z₁', -10, 10, 0.1, 2)}${sl('z2', 'logit z₂', -10, 10, 0.1, 1)}${sl('z3', 'logit z₃', -10, 10, 0.1, 0.1)}${sl('z4', 'logit z₄', -10, 10, 0.1, -1)}
       ${sl('T', 'temperature T', 0.1, 5, 0.1, 1)}${sl('c', 'shift c added to every logit', 0, 1000, 1, 0)}
       <label>Arithmetic<select id="prec"><option value="32" selected>float32 (GPU default)</option><option value="64">float64 (Python / JS)</option></select></label>
       <div class="row"><button class="btn" id="big">Set c = 100</button><button class="btn" id="zero">Set c = 0</button></div>`);
    const c = AI.canvas($('canvas', view)); const q = (s) => $(s, root);
    c.onresize = () => (dirty = true);
    let dirty = true, disp = [0.25, 0.25, 0.25, 0.25], tgt = [0.25, 0.25, 0.25, 0.25], info = {};
    const calc = () => {
      const z = [1, 2, 3, 4].map((i) => +q('#z' + i).value), T = +q('#T').value, sh = +q('#c').value, f32 = q('#prec').value === '32';
      const r = f32 ? Math.fround : (x) => x, a = z.map((v) => r((v + sh) / T));
      const ne = a.map((v) => r(Math.exp(v))), ns = ne.reduce((s, v) => r(s + v), 0), np = ne.map((v) => r(v / ns));
      const m = Math.max(...a), se = a.map((v) => r(Math.exp(r(v - m)))), ss = se.reduce((s, v) => r(s + v), 0), sp = se.map((v) => r(v / ss));
      const lse = m + Math.log(ss);
      info = { a, ne, ns, np, m, se, ss, sp, lse, f32, T, sh };
      tgt = sp; dirty = true;
    };
    wire(root, ['z1', 'z2', 'z3', 'z4', 'T', 'c'], calc); q('#prec').onchange = calc;
    q('#big').onclick = () => { q('#c').value = 100; $('#cv', root).textContent = '100'; calc(); };
    q('#zero').onclick = () => { q('#c').value = 0; $('#cv', root).textContent = '0'; calc(); };
    calc(); window.addEventListener('themechange', () => (dirty = true));
    const f = (v) => (Number.isFinite(v) ? AI.fmt(v, 4) : Number.isNaN(v) ? '<span class="r">NaN</span>' : '<span class="r">inf</span>');
    const draw = () => {
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const bw = Math.min(70, c.w / 5), gap = (c.w - 4 * bw) / 5, base = c.h - 30, top = 28;
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(0, base); g.lineTo(c.w, base); g.stroke();
      for (let i = 0; i < 4; i++) {
        const x = gap + i * (bw + gap), h = (base - top) * disp[i];
        g.fillStyle = [C.blue, C.green, C.orange, C.purple][i]; g.fillRect(x, base - h, bw, h);
        txt(g, AI.fmt(disp[i], 3), x + bw / 2, base - h - 6, C.fg, 'center');
        txt(g, 'class ' + (i + 1), x + bw / 2, base + 16, C.muted, 'center');
        const np = info.np[i];
        if (Number.isFinite(np)) { g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([4, 3]); g.strokeRect(x - 3, base - (base - top) * np, bw + 6, 0.1); g.setLineDash([]); }
        else txt(g, 'naive: NaN ✗', x + bw / 2, top - 8, C.red, 'center');
      }
      txt(g, 'bars: stable softmax(z/T)   dashed tick: naive result (when it works)', 8, 14, C.muted, 'left', '11px sans-serif');
      const I = info, rowN = (arr) => arr.map(f).join('  '), p = I.f32 ? 'float32' : 'float64';
      q('#out').innerHTML = `<span class="m">arithmetic: ${p}; scaled logits a = (z + c)/T = [${I.a.map((v) => AI.fmt(v, 2)).join(', ')}]</span>
<b>NAIVE</b>   e = exp(a)   = [${rowN(I.ne)}]\n         sum         = ${f(I.ns)}\n         p = e / sum = [${rowN(I.np)}]
<b>STABLE</b>  m = max(a) = ${AI.fmt(I.m, 2)};  exp(a − m) = [${rowN(I.se)}]\n         sum = ${f(I.ss)};  p = [${rowN(I.sp)}]\n         log-sum-exp = m + ln(sum) = <span class="o">${AI.fmt(I.lse, 4)}</span>;  log p₁ = a₁ − LSE = <span class="g">${AI.fmt(I.a[0] - I.lse, 4)}</span>`;
    };
    AI.loop(root, (dt) => {
      let moving = false; const k = ease(dt, 5);
      for (let i = 0; i < 4; i++) { const d = tgt[i] - disp[i]; if (Math.abs(d) > 1e-4) { disp[i] += d * k; moving = true; } else disp[i] = tgt[i]; }
      if (dirty || moving) { dirty = false; draw(); }
    });
  };

  /* ================= 7.3  loss functions ================= */
  AI.labs.losses = function (root) {
    const { view, ctl } = AI.labShell(root, 'Loss functions: shape, gradient and what they assume', 'top: loss as a function of the error; bottom: where each loss puts a "best constant" when data has an outlier',
      `<canvas style="height:290px"></canvas><canvas style="height:150px"></canvas>`,
      `<label>Problem<select id="mode"><option value="reg">Regression: loss vs error e = ŷ − y</option><option value="cls">Classification: loss vs margin m = y·s (y = ±1)</option></select></label>
       <label><span>show</span><select id="what"><option value="loss">the loss L</option><option value="grad">its gradient dL/de (the push on the model)</option></select></label>
       ${sl('d', 'Huber threshold δ', 0.3, 3, 0.1, 1)}${sl('o', 'outlier value (other points sit near 1)', 1, 20, 0.5, 1)}
       <div class="readout" id="out" style="white-space:pre-wrap"></div>`);
    const [c1, c2] = AI.$$('canvas', view).map((cv) => AI.canvas(cv)); const q = (s) => $(s, root);
    const DATA = [0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3];
    let dirty = true, est = [1, 1, 1], tg = [1, 1, 1];
    c1.onresize = c2.onresize = () => (dirty = true);
    const hub = (e, d) => (Math.abs(e) <= d ? 0.5 * e * e : d * (Math.abs(e) - 0.5 * d));
    const solve = () => {
      const o = +q('#o').value, d = +q('#d').value, xs = DATA.concat([o]), sorted = [...xs].sort((a, b) => a - b), n = xs.length;
      const mean = xs.reduce((a, b) => a + b) / n, med = (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
      let lo = 0, hi = 21; for (let i = 0; i < 80; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; const f1 = xs.reduce((s, x) => s + hub(m1 - x, d), 0), f2 = xs.reduce((s, x) => s + hub(m2 - x, d), 0); f1 < f2 ? (hi = m2) : (lo = m1); }
      tg = [mean, med, (lo + hi) / 2]; dirty = true;
    };
    wire(root, ['d', 'o'], solve); q('#mode').onchange = () => (dirty = true); q('#what').onchange = () => (dirty = true); solve(); est = [...tg];
    window.addEventListener('themechange', () => (dirty = true));
    const draw = () => {
      const C = AI.col(), mode = q('#mode').value, grad = q('#what').value === 'grad', d = +q('#d').value;
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      let defs, xr, yr, xl;
      if (mode === 'reg') {
        xr = [-4, 4]; yr = grad ? [-3.5, 3.5] : [0, 8]; xl = 'error e = ŷ − y';
        defs = [['MSE ½e²', C.blue, (e) => (grad ? e : 0.5 * e * e)], ['MAE |e|', C.green, (e) => (grad ? Math.sign(e) : Math.abs(e))], ['Huber δ', C.orange, (e) => (grad ? AI.clamp(e, -d, d) : hub(e, d))]];
      } else {
        xr = [-3, 3]; yr = grad ? [-3.2, 0.5] : [0, 4]; xl = 'margin m = y·s  (m > 0: correct side)';
        defs = [['0-1 loss', C.muted, (m) => (grad ? 0 : m < 0 ? 1 : 0)], ['hinge max(0,1−m)', C.green, (m) => (grad ? (m < 1 ? -1 : 0) : Math.max(0, 1 - m))], ['logistic ln(1+e^−m)', C.orange, (m) => (grad ? -sig(-m) : Math.log(1 + Math.exp(-m)))], ['squared (1−m)²', C.blue, (m) => (grad ? -2 * (1 - m) : (1 - m) ** 2)]];
      }
      const v = AI.view(c1, xr[0], xr[1], yr[0], yr[1]); AI.axes(g, v);
      defs.forEach(([n, col, fn], i) => {
        g.strokeStyle = col; g.lineWidth = 3; g.beginPath();
        for (let j = 0; j <= 300; j++) { const x = xr[0] + ((xr[1] - xr[0]) * j) / 300, y = AI.clamp(fn(x), yr[0] - 5, yr[1] + 5); j ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
        g.stroke(); txt(g, n, v.pad.l + 10 + i * 135, 16, col, 'left', '12px sans-serif');
      });
      txt(g, xl + (grad ? '   (vertical axis: gradient)' : '   (vertical axis: loss)'), c1.w - 8, c1.h - 4, C.muted, 'right', '11px sans-serif');
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const o = +q('#o').value, v2 = AI.view(c2, 0, 21, 0, 1, { l: 14, r: 14, t: 6, b: 24 }), xs = DATA.concat([o]);
      AI.axes(g, v2, { noY: true }); const yb = v2.Y(0.62);
      xs.forEach((x, i) => { g.fillStyle = i === xs.length - 1 ? C.red : C.fg; g.beginPath(); g.arc(v2.X(x), yb, i === xs.length - 1 ? 7 : 5, 0, 7); g.fill(); });
      [['mean (MSE)', C.blue, 0.38], ['median (MAE)', C.green, 0.2], ['Huber', C.orange, 0.08]].forEach(([n, col, yy], i) => {
        g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(v2.X(est[i]), v2.Y(yy) + 14); g.lineTo(v2.X(est[i]), yb + 8); g.stroke();
        txt(g, `${n} = ${AI.fmt(est[i], 2)}`, v2.X(est[i]) + 5, v2.Y(yy) + 12, col, 'left', '11px sans-serif');
      });
      txt(g, 'best single constant for 8 numbers (red = outlier)', 14, 14, C.muted, 'left', '11px sans-serif');
      q('#out').innerHTML = mode === 'reg'
        ? `<span class="b">MSE</span> = −log-likelihood of <b>Gaussian</b> noise. Gradient grows with e: outliers shout.\n<span class="g">MAE</span> = <b>Laplace</b> noise. Constant push ±1: ignores size, estimates the median.\n<span class="o">Huber</span> = Gaussian in the middle, Laplace in the tails (robust, smooth).\nHuber(δ=${AI.fmt(d, 1)}) at e=3: ${AI.fmt(hub(3, d), 3)} vs MSE ${AI.fmt(4.5, 3)}`
        : `<span class="o">logistic (= BCE with p = σ(s))</span> = <b>Bernoulli</b> likelihood; never exactly zero, always pushes margins up.\n<span class="g">hinge</span> = max-margin (SVM): zero gradient once m ≥ 1.\n<span class="b">squared</span> punishes being "too correct" (m > 1) - a flaw.\n0-1 loss has zero gradient everywhere: cannot be trained by gradient descent.`;
    };
    AI.loop(root, (dt) => {
      let mv = false; const k = ease(dt, 4);
      for (let i = 0; i < 3; i++) { const e = tg[i] - est[i]; if (Math.abs(e) > 1e-3) { est[i] += e * k; mv = true; } else est[i] = tg[i]; }
      if (dirty || mv) { dirty = false; draw(); }
    });
  };

  /* ================= 7.4  backprop through a 2-2-1 network ================= */
  AI.labs.mlpback = function (root) {
    const KEYS = ['x1', 'x2', 't', 'w11', 'w12', 'w21', 'w22', 'b1', 'b2', 'v1', 'v2', 'c'];
    const DEF = { x1: 1, x2: 2, t: 1, w11: 0.1, w12: 0.3, w21: 0.2, w22: -0.4, b1: 0.1, b2: -0.1, v1: 0.5, v2: -0.6, c: 0.2 };
    const LBL = { x1: 'x₁', x2: 'x₂', t: 'target y', w11: 'w₁₁', w12: 'w₁₂', w21: 'w₂₁', w22: 'w₂₂', b1: 'b₁', b2: 'b₂', v1: 'v₁', v2: 'v₂', c: 'c' };
    const { view, ctl } = AI.labShell(root, 'A 2-2-1 network, forward then backward — every number shown', 'hidden: h = φ(W₁x + b); output: ŷ = v·h + c; loss L = ½(ŷ − y)²',
      `<svg id="svg" viewBox="0 0 640 330" style="width:100%;background:var(--bg);border-radius:8px"></svg><div class="caption" id="cap" style="min-height:64px;margin-top:8px;font-size:13px"></div>`,
      `<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 14px">${KEYS.map((k) => sl(k, LBL[k], k.startsWith('x') || k === 't' ? -3 : -2, k.startsWith('x') || k === 't' ? 3 : 2, 0.05, DEF[k])).join('')}</div>
       <label>Hidden activation φ<select id="act"><option value="sigmoid">sigmoid</option><option value="tanh">tanh</option><option value="relu">ReLU</option></select></label>
       ${sl('dur', 'seconds per step', 1, 8, 0.5, 3.5)}
       ${playRow()}
       <div class="row"><button class="btn" id="sgd">Take one gradient step (η)</button></div>${sl('lr', 'learning rate η', 0.05, 2, 0.05, 0.5)}
       <div class="readout" id="out" style="white-space:pre-wrap;font-size:11.5px"></div>`);
    const q = (s) => $(s, root), svg = q('#svg'), NS = 'http://www.w3.org/2000/svg';
    const P = {}; let A = { f: sig, d: (z, h) => h * (1 - h) };
    const ACTS = { sigmoid: { f: sig, d: (z, h) => h * (1 - h) }, tanh: { f: Math.tanh, d: (z, h) => 1 - h * h }, relu: { f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0) } };
    const calc = (P) => {
      const z1 = P.w11 * P.x1 + P.w12 * P.x2 + P.b1, z2 = P.w21 * P.x1 + P.w22 * P.x2 + P.b2, h1 = A.f(z1), h2 = A.f(z2);
      const yh = P.v1 * h1 + P.v2 * h2 + P.c, dy = yh - P.t, dh1 = dy * P.v1, dh2 = dy * P.v2, dz1 = dh1 * A.d(z1, h1), dz2 = dh2 * A.d(z2, h2);
      return { z1, z2, h1, h2, yh, L: 0.5 * dy * dy, dy, dv1: dy * h1, dv2: dy * h2, dc: dy, dh1, dh2, dz1, dz2, dw11: dz1 * P.x1, dw12: dz1 * P.x2, dw21: dz2 * P.x1, dw22: dz2 * P.x2, db1: dz1, db2: dz2 };
    };
    const POS = { x1: [60, 105], x2: [60, 235], h1: [270, 85], h2: [270, 255], o: [470, 170], L: [585, 170] };
    const EDG = [['x1', 'h1', 'w11'], ['x2', 'h1', 'w12'], ['x1', 'h2', 'w21'], ['x2', 'h2', 'w22'], ['h1', 'o', 'v1'], ['h2', 'o', 'v2'], ['o', 'L', '']];
    const mk = (tag, at, par = svg) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); par.appendChild(e); return e; };
    const FF = 'ui-monospace,Menlo,Consolas,monospace', edgeEl = [], eV = [], eG = [], dots = [], nT = {}, texts = [];
    EDG.forEach(([a, b], i) => { edgeEl[i] = mk('line', { x1: POS[a][0], y1: POS[a][1], x2: POS[b][0], y2: POS[b][1], stroke: 'var(--border)', 'stroke-width': 2.5 }); });
    Object.entries(POS).forEach(([k, [x, y]]) => {
      mk('circle', { cx: x, cy: y, r: 24, fill: 'var(--bg2)', stroke: k[0] === 'x' ? 'var(--blue)' : k === 'L' ? 'var(--red)' : 'var(--purple)', 'stroke-width': 2 });
      const lab = { x1: 'x₁', x2: 'x₂', h1: 'h₁', h2: 'h₂', o: 'ŷ', L: 'L' }[k]; mk('text', { x, y: y + 5, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': 15 }).textContent = lab;
    });
    const T = (x, y, col, anchor = 'middle') => mk('text', { x, y, fill: col, 'font-size': 11, 'font-family': FF, 'text-anchor': anchor, opacity: 0 });
    // text items: {el, at (stage index when it appears), get(R) -> string}
    const item = (x, y, col, at, get, anchor) => { const el = T(x, y, col, anchor); texts.push({ el, at, get }); return el; };
    const F = (v) => AI.fmt(v, 3), G = (v) => AI.fmt(v, 4);
    item(60, 66, 'var(--blue)', 0, () => 'x₁=' + F(P.x1)); item(60, 196, 'var(--blue)', 0, () => 'x₂=' + F(P.x2));
    item(60, 143, 'var(--blue)', 0, () => '', 'middle');
    [['h1', 'z1', 'h1', 'dh1', 'dz1', 0, 6, 7], ['h2', 'z2', 'h2', 'dh2', 'dz2', 1, 6, 7]].forEach(([n, z, h, dh, dz, at, a1, a2]) => {
      const [x, y] = POS[n];
      item(x, y - 46, 'var(--blue)', at, (R) => `z=${F(R[z])}`); item(x, y - 33, 'var(--blue)', at, (R) => `h=${F(R[h])}`);
      item(x, y + 40, 'var(--orange)', a1, (R) => `∂L/∂h=${G(R[dh])}`); item(x, y + 53, 'var(--orange)', a2, (R) => `∂L/∂z=${G(R[dz])}`);
    });
    item(470, 124, 'var(--blue)', 2, (R) => `ŷ=${F(R.yh)}`); item(470, 212, 'var(--orange)', 4, (R) => `∂L/∂ŷ=${G(R.dy)}`); item(470, 225, 'var(--orange)', 5, (R) => `∂L/∂c=${G(R.dc)}  (c=${F(P.c)})`);
    item(585, 124, 'var(--blue)', 3, (R) => `L=${G(R.L)}`);
    item(160, 30, 'var(--muted)', -1, () => ''); // spacer
    item(270, 150, 'var(--orange)', 8, (R) => `∂b₁=${G(R.db1)}`); item(270, 322, 'var(--orange)', 8, (R) => `∂b₂=${G(R.db2)}`);
    const edgeLab = [[0.22, -7], [0.22, -7], [0.22, 15], [0.22, 15], [0.5, -9], [0.5, 15], [0.5, -9]];
    EDG.forEach(([a, b, w], i) => {
      if (!w) return; const [t, off] = edgeLab[i], x = POS[a][0] + (POS[b][0] - POS[a][0]) * t, y = POS[a][1] + (POS[b][1] - POS[a][1]) * t;
      eV[i] = mk('text', { x, y: y + off - 6, fill: 'var(--fg)', 'font-size': 11, 'font-family': FF, 'text-anchor': 'middle' });
      eG[i] = mk('text', { x, y: y + off + 7, fill: 'var(--orange)', 'font-size': 11, 'font-family': FF, 'text-anchor': 'middle', opacity: 0 });
    });
    for (let i = 0; i < 7; i++) dots[i] = mk('circle', { r: 6, fill: 'var(--blue)', opacity: 0 });
    // stages
    const STG = [
      { dir: 1, e: [0, 1], cap: (R) => `<b>① Forward, hidden neuron 1.</b> z₁ = w₁₁x₁ + w₁₂x₂ + b₁ = (${F(P.w11)})(${F(P.x1)}) + (${F(P.w12)})(${F(P.x2)}) + ${F(P.b1)} = <b>${F(R.z1)}</b>, so h₁ = φ(z₁) = <b>${F(R.h1)}</b>.` },
      { dir: 1, e: [2, 3], cap: (R) => `<b>② Forward, hidden neuron 2.</b> z₂ = (${F(P.w21)})(${F(P.x1)}) + (${F(P.w22)})(${F(P.x2)}) + ${F(P.b2)} = <b>${F(R.z2)}</b>, so h₂ = φ(z₂) = <b>${F(R.h2)}</b>.` },
      { dir: 1, e: [4, 5], cap: (R) => `<b>③ Forward, output.</b> ŷ = v₁h₁ + v₂h₂ + c = (${F(P.v1)})(${F(R.h1)}) + (${F(P.v2)})(${F(R.h2)}) + ${F(P.c)} = <b>${F(R.yh)}</b>.` },
      { dir: 1, e: [6], cap: (R) => `<b>④ Forward, loss.</b> L = ½(ŷ − y)² = ½(${F(R.yh)} − ${F(P.t)})² = ½(${F(R.dy)})² = <b>${G(R.L)}</b>.   The forward pass is finished; every number is stored.` },
      { dir: -1, e: [6], cap: (R) => `<b>⑤ Backward starts at the loss.</b> ∂L/∂ŷ = ŷ − y = ${F(R.yh)} − ${F(P.t)} = <b>${G(R.dy)}</b>. (If ŷ moves up by 1, L changes by about this much.)` },
      { dir: -1, e: [4, 5], cap: (R) => `<b>⑥ Output weights.</b> ∂L/∂vⱼ = (∂L/∂ŷ)·hⱼ: ∂L/∂v₁ = ${G(R.dy)}×${F(R.h1)} = <b>${G(R.dv1)}</b>, ∂L/∂v₂ = ${G(R.dy)}×${F(R.h2)} = <b>${G(R.dv2)}</b>; ∂L/∂c = <b>${G(R.dc)}</b>.` },
      { dir: -1, e: [4, 5], cap: (R) => `<b>⑦ Into the hidden activations.</b> ∂L/∂hⱼ = (∂L/∂ŷ)·vⱼ: ∂L/∂h₁ = ${G(R.dy)}×${F(P.v1)} = <b>${G(R.dh1)}</b>, ∂L/∂h₂ = ${G(R.dy)}×${F(P.v2)} = <b>${G(R.dh2)}</b>.` },
      { dir: -1, e: [0, 1, 2, 3], cap: (R) => `<b>⑧ Through φ.</b> ∂L/∂zⱼ = (∂L/∂hⱼ)·φ′(zⱼ): ∂L/∂z₁ = ${G(R.dh1)}×${G(A.d(R.z1, R.h1))} = <b>${G(R.dz1)}</b>, ∂L/∂z₂ = ${G(R.dh2)}×${G(A.d(R.z2, R.h2))} = <b>${G(R.dz2)}</b>.` },
      { dir: -1, e: [0, 1, 2, 3], cap: (R) => `<b>⑨ First-layer weights.</b> ∂L/∂wᵢⱼ = (∂L/∂zᵢ)·xⱼ: ∂L/∂w₁₁ = ${G(R.dz1)}×${F(P.x1)} = <b>${G(R.dw11)}</b>, ∂L/∂w₁₂ = <b>${G(R.dw12)}</b>, ∂L/∂w₂₁ = <b>${G(R.dw21)}</b>, ∂L/∂w₂₂ = <b>${G(R.dw22)}</b>; biases ∂L/∂bᵢ = ∂L/∂zᵢ. <b>Done.</b>` },
    ];
    const gKey = ['dw11', 'dw12', 'dw21', 'dw22', 'dv1', 'dv2'], gAt = [8, 8, 8, 8, 5, 5], wKey = ['w11', 'w12', 'w21', 'w22', 'v1', 'v2'];
    let cur = -1, st = 0, playing = true, dirty = true, R = null, lastL = null, fdDirty = true;
    const recalc = () => { KEYS.forEach((k) => (P[k] = +q('#' + k).value)); A = ACTS[q('#act').value]; R = calc(P); dirty = true; fdDirty = true; };
    wire(root, KEYS, recalc); wire(root, ['dur', 'lr'], () => {}); q('#act').onchange = recalc; recalc();
    const setPlay = (p) => { playing = p; AI.$('[data-r=play]', root).textContent = p ? '⏸ Pause' : '▶ Play'; };
    AI.$('[data-r=play]', root).onclick = () => { if (cur >= STG.length - 1 && !playing) { cur = -1; st = 0; } setPlay(!playing); dirty = true; };
    AI.$('[data-r=step]', root).onclick = () => { setPlay(false); if (cur < STG.length - 1) { cur++; st = 0; dirty = true; } };
    AI.$('[data-r=back]', root).onclick = () => { setPlay(false); if (cur > -1) { cur--; st = 0; dirty = true; } };
    AI.$('[data-r=reset]', root).onclick = () => { cur = -1; st = 0; setPlay(true); dirty = true; };
    q('#sgd').onclick = () => {
      lastL = R.L; const lr = +q('#lr').value;
      ['w11', 'w12', 'w21', 'w22', 'b1', 'b2', 'v1', 'v2', 'c'].forEach((k) => { const g = R['d' + k]; const el = q('#' + k); el.value = AI.clamp(P[k] - lr * g, -2, 2); $('#' + k + 'v', root).textContent = AI.fmt(+el.value, 2); });
      recalc(); cur = 8; st = 0; dirty = true;
    };
    const fd = () => {
      const ks = ['w11', 'w12', 'w21', 'w22', 'b1', 'b2', 'v1', 'v2', 'c'], e = 1e-5; let s = '<span class="m">finite-difference check  (L(p+ε) − L(p−ε)) / 2ε,  ε = 1e-5</span>\n<span class="m">param    backprop    numeric   </span>\n', mx = 0;
      ks.forEach((k) => { const a = { ...P }, b = { ...P }; a[k] += e; b[k] -= e; const num = (calc(a).L - calc(b).L) / (2 * e), an = R['d' + k]; mx = Math.max(mx, Math.abs(num - an)); s += `${k.padEnd(6)}  ${G(an).padStart(9)}  ${G(num).padStart(9)}\n`; });
      return s + `max |difference| = <span class="g">${mx.toExponential(1)}</span> → backprop is correct${lastL !== null ? `\n\nlast gradient step: L ${G(lastL)} → ${G(R.L)}` : ''}`;
    };
    AI.loop(root, (dt) => {
      let anim = false;
      if (playing && cur < STG.length - 1) { st += dt; if (cur >= 0 && st > +q('#dur').value) { cur++; st = 0; } else if (cur < 0 && st > 0.8) { cur = 0; st = 0; } anim = true; }
      else if (playing && cur === STG.length - 1 && st < +q('#dur').value) { st += dt; anim = true; }
      const cap = q('#cap');
      if (!dirty && !anim) return;
      const S = STG[Math.max(0, cur)], p = AI.smooth(st / (+q('#dur').value * 0.7));
      texts.forEach((t) => { const on = cur >= t.at && t.at >= 0; t.el.setAttribute('opacity', on ? 1 : 0); if (on) t.el.textContent = t.get(R); });
      EDG.forEach(([a, b, w], i) => {
        const act = cur >= 0 && S.e.includes(i), done = (STG.slice(0, cur + 1).some((s) => s.e.includes(i)));
        edgeEl[i].setAttribute('stroke', act ? (S.dir > 0 ? 'var(--blue)' : 'var(--orange)') : done ? 'var(--muted)' : 'var(--border)');
        edgeEl[i].setAttribute('stroke-width', act ? 4 : 2.5);
        if (w) { eV[i].textContent = `${w}=${F(P[w])}`; const gi = gKey.findIndex((k, j) => wKey[j] === w); eG[i].textContent = '∂=' + G(R[gKey[gi]]); eG[i].setAttribute('opacity', cur >= gAt[gi] ? 1 : 0); }
        const d = dots[i]; if (act) { const f = S.dir > 0 ? p : 1 - p; d.setAttribute('cx', POS[a][0] + (POS[b][0] - POS[a][0]) * f); d.setAttribute('cy', POS[a][1] + (POS[b][1] - POS[a][1]) * f); d.setAttribute('fill', S.dir > 0 ? 'var(--blue)' : 'var(--orange)'); d.setAttribute('opacity', 1); } else d.setAttribute('opacity', 0);
      });
      cap.innerHTML = cur < 0 ? 'Press <b>▶ Play</b> (it is already running) or <b>Step ▶</b>. Forward pass first (blue), then backward (orange). Edit any weight with the sliders.' : S.cap(R);
      if (fdDirty) { q('#out').innerHTML = fd(); fdDirty = false; } dirty = false;
    }, { fps: 30 });
  };

  /* ================= 7.5  shapes of matrix calculus ================= */
  AI.labs.shapes = function (root) {
    const { view, ctl } = AI.labShell(root, 'Shape checker for a linear layer  Y = X Wᵀ + b', 'B examples (rows), n_in features in, n_out features out. Every gradient has the shape of the thing it is a gradient of',
      `<canvas style="height:430px"></canvas><div class="caption" id="cap" style="font-size:13px;min-height:44px;margin-top:6px"></div>`,
      `${sl('nin', 'n_in (inputs per example)', 1, 7, 1, 3)}${sl('nout', 'n_out (neurons in the layer)', 1, 7, 1, 2)}${sl('B', 'batch size B', 1, 6, 1, 4)}
       ${sl('sp', 'seconds per step', 1, 6, 0.5, 3)}${playRow()}
       <div class="row"><button class="btn" id="chk">Verify with random numbers</button></div><div class="readout" id="out" style="white-space:pre-wrap"></div>`);
    const c = AI.canvas($('canvas', view)), q = (s) => $(s, root);
    c.onresize = () => (dirty = true);
    let nin = 3, nout = 2, B = 4, cur = 0, st = 0, playing = true, dirty = true;
    const read = () => { nin = +q('#nin').value; nout = +q('#nout').value; B = +q('#B').value; dirty = true; };
    wire(root, ['nin', 'nout', 'B', 'sp'], read); read();
    const setPlay = (p) => { playing = p; AI.$('[data-r=play]', root).textContent = p ? '⏸ Pause' : '▶ Play'; };
    AI.$('[data-r=play]', root).onclick = () => setPlay(!playing);
    AI.$('[data-r=step]', root).onclick = () => { setPlay(false); cur = (cur + 1) % 5; st = 0; dirty = true; };
    AI.$('[data-r=back]', root).onclick = () => { setPlay(false); cur = (cur + 4) % 5; st = 0; dirty = true; };
    AI.$('[data-r=reset]', root).onclick = () => { cur = 0; st = 0; setPlay(true); dirty = true; };
    window.addEventListener('themechange', () => (dirty = true));
    const mm = (a, b) => { const o = []; for (let i = 0; i < a.length; i++) { o.push([]); for (let j = 0; j < b[0].length; j++) { let s = 0; for (let k = 0; k < b.length; k++) s += a[i][k] * b[k][j]; o[i].push(s); } } return o; };
    const tr = (a) => a[0].map((_, j) => a.map((r) => r[j]));
    const rnd = () => Math.random() * 2 - 1, M = (r, cc) => Array.from({ length: r }, () => Array.from({ length: cc }, rnd));
    q('#chk').onclick = () => {
      const X = M(B, nin), W = M(nout, nin), b = M(1, nout)[0], Rr = M(B, nout);
      const L = (X, W, b) => { const Y = mm(X, tr(W)).map((r) => r.map((v, j) => v + b[j])); let s = 0; Y.forEach((r, i) => r.forEach((v, j) => (s += v * Rr[i][j]))); return s; };
      const dW = mm(tr(Rr), X), dX = mm(Rr, W), db = Rr[0].map((_, j) => Rr.reduce((s, r) => s + r[j], 0)), e = 1e-6; let err = 0;
      for (let i = 0; i < nout; i++) for (let j = 0; j < nin; j++) { const W2 = W.map((r) => [...r]); W2[i][j] += e; err = Math.max(err, Math.abs((L(X, W2, b) - L(X, W, b)) / e - dW[i][j])); }
      for (let i = 0; i < B; i++) for (let j = 0; j < nin; j++) { const X2 = X.map((r) => [...r]); X2[i][j] += e; err = Math.max(err, Math.abs((L(X2, W, b) - L(X, W, b)) / e - dX[i][j])); }
      for (let j = 0; j < nout; j++) { const b2 = [...b]; b2[j] += e; err = Math.max(err, Math.abs((L(X, W, b2) - L(X, W, b)) / e - db[j])); }
      q('#out').innerHTML = `random X(${B}×${nin}), W(${nout}×${nin}), b, and upstream gradient Δ(${B}×${nout}); L = Σ Δ∘Y.\nΔᵀX, ΔW and column-sums of Δ vs finite differences:\nmax |error| = <span class="g">${err.toExponential(1)}</span> ✓`;
    };
    const STEPS = [
      ['Forward: Y = X Wᵀ + b', 'X is B×n_in, Wᵀ is n_in×n_out, so XWᵀ is B×n_out; the bias row b (1×n_out) is added to every row (broadcast).'],
      ['Backward starts with Δ = ∂L/∂Y', 'Δ has the shape of Y: B×n_out. Entry (i, j): how much the loss changes if output j of example i changes.'],
      ['∂L/∂W = Δᵀ X', 'Must be n_out×n_in (the shape of W). Δᵀ is n_out×B, X is B×n_in: the inner B is summed away — gradients of all B examples add up.'],
      ['∂L/∂X = Δ W', 'Must be B×n_in. Δ is B×n_out, W is n_out×n_in: inner n_out matches. This is what we send to the previous layer.'],
      ['∂L/∂b = Σ over examples of Δ', 'Must be 1×n_out (the shape of b). Summing the B rows of Δ undoes the broadcast of b in the forward pass.'],
    ];
    const cap = (n, s) => (s || 'rgba(128,128,128,0)');
    AI.loop(root, (dt) => {
      if (playing) { st += dt; if (st > +q('#sp').value) { st = 0; cur = (cur + 1) % 5; dirty = true; } }
      if (!dirty && !playing) return; dirty = false;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const mxDim = Math.max(nin, nout, B), cell = Math.max(9, Math.min(20, 84 / Math.max(nin, nout), 150 / mxDim));
      const BW = 96;
      const mat = (x, y, r, cc, col, label, dims, hot) => {
        const gx = x + (BW - cc * cell) / 2;
        g.save(); g.globalAlpha = hot ? 1 : 0.35;
        for (let i = 0; i < r; i++) for (let j = 0; j < cc; j++) { g.fillStyle = AI.mix(C.bg3, col, 0.5 + 0.3 * Math.sin(i * 2.1 + j * 1.3)); g.fillRect(gx + j * cell, y + i * cell, cell - 2, cell - 2); }
        g.restore(); g.strokeStyle = hot ? col : C.border; g.lineWidth = hot ? 2.5 : 1; g.strokeRect(gx - 3, y - 3, cc * cell + 4, r * cell + 4);
        txt(g, label, x + BW / 2, y - 10, hot ? col : C.muted, 'center', 'bold 11px sans-serif'); txt(g, dims, x + BW / 2, y + r * cell + 15, hot ? C.fg : C.muted, 'center', '10px sans-serif');
        return BW + 16;
      };
      const row1 = 38, row2 = 38 + Math.max(B, nout, nin) * cell + 70; let x = 16;
      const hotF = cur === 0, hot = (i) => cur === i;
      x += mat(x, row1, B, nin, C.blue, 'X', `B×n_in = ${B}×${nin}`, hotF || hot(3) || hot(2));
      txt(g, '·', x - 14, row1 + (B * cell) / 2 + 4, C.fg, 'center', '18px sans-serif');
      x += mat(x, row1, nin, nout, C.green, 'Wᵀ', `n_in×n_out = ${nin}×${nout}`, hotF);
      txt(g, '+', x - 14, row1 + (B * cell) / 2 + 4, C.fg, 'center', '18px sans-serif');
      x += mat(x, row1, 1, nout, C.purple, 'b', `1×n_out = 1×${nout}`, hotF || hot(4));
      txt(g, '=', x - 14, row1 + (B * cell) / 2 + 4, C.fg, 'center', '18px sans-serif');
      mat(x, row1, B, nout, C.orange, 'Y', `B×n_out = ${B}×${nout}`, hotF || hot(1));
      txt(g, 'forward', 16, 16, C.muted, 'left', '11px sans-serif');
      txt(g, 'backward  (everything here is a gradient ∂L/∂·)', 16, row2 - 30, C.muted, 'left', '11px sans-serif');
      x = 16;
      x += mat(x, row2, B, nout, C.orange, 'Δ = ∂L/∂Y', `${B}×${nout}`, hot(1) || hot(2) || hot(3) || hot(4));
      txt(g, '→', x - 14, row2 + (nout * cell) / 2 + 4, C.fg, 'center', '16px sans-serif');
      x += mat(x, row2, nout, nin, C.green, '∂L/∂W = ΔᵀX', `n_out×n_in = ${nout}×${nin}`, hot(2));
      x += mat(x, row2, B, nin, C.blue, '∂L/∂X = ΔW', `B×n_in = ${B}×${nin}`, hot(3));
      mat(x, row2, 1, nout, C.purple, '∂L/∂b = 1ᵀΔ', `1×n_out = 1×${nout}`, hot(4));
      const bar = c.w - 32, pr = playing ? AI.clamp(st / +q('#sp').value, 0, 1) : 1;
      g.fillStyle = C.border; g.fillRect(16, c.h - 10, bar, 3); g.fillStyle = C.blue; g.fillRect(16, c.h - 10, (bar * (cur + pr)) / 5, 3);
      q('#cap').innerHTML = `<b>${STEPS[cur][0]}</b> — ${STEPS[cur][1]}`;
    });
  };

  /* ================= 7.6  gradient flow through depth ================= */
  AI.labs.gradflow = function (root) {
    const { view, ctl } = AI.labShell(root, 'A deep stack: how big are activations (blue) and gradients (orange) at each layer?', 'width 100, random weights. Bars are log₁₀ of the root-mean-square value per layer; a signal sweeps forward, then the gradient sweeps backward',
      `<canvas style="height:340px"></canvas><div class="readout" id="out" style="margin-top:8px;white-space:pre-wrap"></div>`,
      `<label>Activation<select id="act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">sigmoid</option></select></label>
       <label>Weight initialisation<select id="init"><option value="small">N(0, 0.02²) — too small</option><option value="unit">N(0, 1) — too large</option><option value="xavier">Xavier: Var = 2/(n_in+n_out)</option><option value="he" selected>He: Var = 2/n_in</option></select></label>
       ${sl('L', 'depth (layers)', 2, 30, 1, 20)}
       <label><span>Residual connections</span><span class="row" style="gap:8px"><input type="checkbox" id="res"> h ← h + φ(Wh)</span></label>
       ${sl('sp', 'sweep speed (layers / s)', 1, 12, 0.5, 4)}${playRow('<button class="btn" id="rs">New random weights</button>')}
       `);
    const c = AI.canvas($('canvas', view)), q = (s) => $(s, root), n = 100;
    c.onresize = () => (dirty = true);
    let sim = null, seed = 7, dirty = true, pos = 0, phase = 0, playing = true, hold = 0;
    const FA = { tanh: [Math.tanh, (z, h) => 1 - h * h], relu: [(z) => Math.max(0, z), (z) => (z > 0 ? 1 : 0)], sigmoid: [sig, (z, h) => h * (1 - h)] };
    const run = () => {
      const act = q('#act').value, init = q('#init').value, L = +q('#L').value, res = q('#res').checked, [f, df] = FA[act], rnd = AI.rng(seed), N = () => AI.randn(rnd);
      const sd = { small: 0.02, unit: 1, xavier: Math.sqrt(1 / n), he: Math.sqrt(2 / n) }[init];
      let h = Array.from({ length: n }, N); const hs = [Math.sqrt(h.reduce((s, v) => s + v * v, 0) / n)], Z = [], Hh = [], Ws = [];
      for (let l = 0; l < L; l++) {
        const W = Array.from({ length: n }, () => Float64Array.from({ length: n }, () => N() * sd)), z = W.map((r) => { let s = 0; for (let j = 0; j < n; j++) s += r[j] * h[j]; return s; });
        const a = z.map(f); Hh.push(a); Z.push(z); Ws.push(W); h = res ? h.map((v, i) => v + a[i]) : a; hs.push(Math.sqrt(h.reduce((s, v) => s + v * v, 0) / n));
      }
      let g = Array.from({ length: n }, N); const gs = new Array(L + 1); gs[L] = Math.sqrt(g.reduce((s, v) => s + v * v, 0) / n);
      for (let l = L - 1; l >= 0; l--) {
        const u = g.map((v, i) => v * df(Z[l][i], Hh[l][i])), gn = new Array(n).fill(0);
        for (let i = 0; i < n; i++) { const r = Ws[l][i], ui = u[i]; for (let j = 0; j < n; j++) gn[j] += r[j] * ui; }
        g = res ? gn.map((v, j) => v + g[j]) : gn; gs[l] = Math.sqrt(g.reduce((s, v) => s + v * v, 0) / n);
      }
      sim = { hs, gs, L }; dirty = true;
    };
    const redo = () => { run(); pos = 0; phase = 0; hold = 0; };
    wire(root, ['L', 'sp'], (id) => { if (id === 'L') redo(); }); q('#act').onchange = redo; q('#init').onchange = redo; q('#res').onchange = redo; redo();
    q('#rs').onclick = () => { seed = (seed * 7 + 3) | 0; redo(); };
    const setPlay = (p) => { playing = p; AI.$('[data-r=play]', root).textContent = p ? '⏸ Pause' : '▶ Play'; };
    AI.$('[data-r=play]', root).onclick = () => setPlay(!playing);
    AI.$('[data-r=step]', root).onclick = () => { setPlay(false); adv(1); };
    AI.$('[data-r=back]', root).onclick = () => { setPlay(false); adv(-1); };
    AI.$('[data-r=reset]', root).onclick = () => { pos = 0; phase = 0; hold = 0; setPlay(true); dirty = true; };
    const adv = (d) => { // step one layer along the sweep
      if (phase === 0) { pos += d; if (pos > sim.L) { phase = 1; pos = sim.L; } if (pos < 0) pos = 0; } else if (phase === 1) { pos -= d; if (pos < 0) { phase = 2; pos = 0; } if (pos > sim.L) { phase = 0; pos = sim.L; } } dirty = true;
    };
    window.addEventListener('themechange', () => (dirty = true));
    const ly = (v) => AI.clamp(Math.log10(Math.max(v, 1e-30)), -10, 6);
    AI.loop(root, (dt) => {
      if (playing) {
        if (phase === 2) { hold += dt; if (hold > 3) { pos = 0; phase = 0; hold = 0; } dirty = true; }
        else { const sp = +q('#sp').value * dt; if (phase === 0) { pos += sp; if (pos >= sim.L) { pos = sim.L; phase = 1; } } else { pos -= sp; if (pos <= 0) { pos = 0; phase = 2; hold = 0; } } dirty = true; }
      }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, L = sim.L; g.clearRect(0, 0, c.w, c.h);
      const pad = { l: 46, r: 10, t: 22, b: 26 }, pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b, Y = (lv) => pad.t + ph * (1 - (lv + 10) / 16), bw = pw / (L + 1);
      g.font = '10px ui-monospace,Menlo,monospace';
      for (let e = -10; e <= 6; e += 2) { g.strokeStyle = e === 0 ? C.muted : C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(pad.l, Y(e)); g.lineTo(pad.l + pw, Y(e)); g.stroke(); txt(g, '1e' + e, pad.l - 5, Y(e) + 3, C.muted, 'right', '10px monospace'); }
      const fwdPos = phase === 0 ? pos : L, bwdPos = phase === 0 ? L + 1 : pos;
      for (let l = 0; l <= L; l++) {
        const x = pad.l + l * bw;
        if (l <= fwdPos + 1e-9) { g.fillStyle = C.blue; const y = Y(ly(sim.hs[l])); g.fillRect(x + bw * 0.08, y, bw * 0.4, Y(-10) - y); }
        if (l >= bwdPos - 1e-9) { g.fillStyle = C.orange; const y = Y(ly(sim.gs[l])); g.fillRect(x + bw * 0.5, y, bw * 0.4, Y(-10) - y); }
        if (l % (L > 15 ? 5 : 1) === 0) txt(g, l, x + bw / 2, c.h - 8, C.muted, 'center', '10px monospace');
      }
      const gx = pad.l + (pos + 0.5) * bw; g.strokeStyle = phase === 0 ? C.blue : C.orange; g.lineWidth = 2; g.globalAlpha = 0.7; g.beginPath(); g.moveTo(gx, pad.t); g.lineTo(gx, pad.t + ph); g.stroke(); g.globalAlpha = 1;
      txt(g, phase === 0 ? `forward signal at layer ${Math.round(pos)}` : phase === 1 ? `backward gradient at layer ${Math.round(pos)}` : 'sweep finished', pad.l + 6, 14, phase === 0 ? C.blue : C.orange, 'left', '12px sans-serif');
      txt(g, 'x axis: layer (0 = input)', c.w - 8, 14, C.muted, 'right', '10px sans-serif');
      const ratio = sim.gs[0] / sim.gs[L], verdict = ratio < 1e-3 ? '<span class="r">VANISHING: early layers receive almost no learning signal</span>' : ratio > 1e3 ? '<span class="r">EXPLODING: updates for early layers are enormous</span>' : '<span class="g">healthy: gradients keep a usable size</span>';
      q('#out').innerHTML = `activation rms:  input ${AI.fmt(sim.hs[0], 3)} → last layer ${AI.fmt(sim.hs[L], 3)}\ngradient rms:    last layer ${AI.fmt(sim.gs[L], 3)} → first layer ${AI.fmt(sim.gs[0], 3)}   (ratio first/last = ${AI.fmt(ratio, 3)})\n${verdict}`;
    }, { fps: 30 });
  };

  /* ================= 7.7  attention ================= */
  AI.labs.attention = function (root) {
    const DQ = [[1, 0], [0, 1], [1, 1], [-1, 0.5]], DV = [[1, 0], [0, 2], [3, 1], [-1, -1]], NM = ['the', 'cat', 'sat', 'down'];
    const cells = (kind, ij) => DQ.map((_, i) => `<input type="number" step="0.1" data-k="${kind}" data-i="${i}" data-j="${ij}" style="width:100%">`);
    const rowsHtml = DQ.map((_, i) => `<tr><td style="color:var(--muted)">${i + 1} ${NM[i]}</td>${['q', 'k', 'v'].map((k) => [0, 1].map((j) => `<td><input type="number" step="0.1" data-k="${k}" data-i="${i}" data-j="${j}" style="width:40px;background:var(--bg);border:1px solid var(--border);border-radius:5px;color:var(--fg);padding:2px 3px"></td>`).join('')).join('')}</tr>`).join('');
    const { view, ctl } = AI.labShell(root, 'Scaled dot-product attention with 4 tokens (d_k = 2)', 'scores = QKᵀ/√d_k → softmax per row → output = weights · V. Edit any number.',
      `<canvas style="height:470px"></canvas><div class="caption" id="cap" style="font-size:13px;min-height:46px;margin-top:6px"></div>`,
      `<table style="border-collapse:separate;border-spacing:3px 2px;font-size:12px"><tr style="color:var(--muted)"><td></td><td colspan="2">query q</td><td colspan="2">key k</td><td colspan="2">value v</td></tr>${rowsHtml}</table>
       <div class="chips" id="qsel"></div>
       <div class="row"><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="sc" checked> divide by √d_k</label><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="mk"> causal mask</label></div>
       ${sl('sp', 'seconds per added term', 0.5, 4, 0.25, 1.75)}${playRow()}`);
    const c = AI.canvas($('canvas', view)), q = (s) => $(s, root);
    c.onresize = () => (dirty = true);
    const Q = DQ.map((r) => [...r]), K = DQ.map((r) => [...r]), V = DV.map((r) => [...r]), M = { q: Q, k: K, v: V };
    AI.$$('input[data-k]', root).forEach((el) => { el.value = M[el.dataset.k][+el.dataset.i][+el.dataset.j]; el.addEventListener('input', () => { const v = parseFloat(el.value); if (Number.isFinite(v)) { M[el.dataset.k][+el.dataset.i][+el.dataset.j] = v; compute(); } }); });
    let qi = 2, S = [], Aw = [], O = [], cur = 0, st = 0, playing = true, dirty = true;
    AI.$$('#qsel', root).forEach(() => {});
    NM.forEach((nm, i) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = `query: token ${i + 1} “${nm}”`; b.onclick = () => { qi = i; cur = 0; st = 0; dirty = true; }; q('#qsel').appendChild(b); });
    const compute = () => {
      const sc = q('#sc').checked, mk = q('#mk').checked, d = 2; S = []; Aw = []; O = [];
      for (let i = 0; i < 4; i++) {
        const s = K.map((k, j) => (mk && j > i ? -Infinity : (Q[i][0] * k[0] + Q[i][1] * k[1]) / (sc ? Math.sqrt(d) : 1)));
        const m = Math.max(...s), e = s.map((v) => (v === -Infinity ? 0 : Math.exp(v - m))), z = e.reduce((a, b) => a + b), a = e.map((v) => v / z);
        S.push(s); Aw.push(a); O.push([0, 1].map((j) => a.reduce((t, w, r) => t + w * V[r][j], 0)));
      }
      cur = 0; st = 0; dirty = true;
    };
    q('#sc').onchange = compute; q('#mk').onchange = compute; compute();
    wire(root, ['sp'], () => {});
    const setPlay = (p) => { playing = p; AI.$('[data-r=play]', root).textContent = p ? '⏸ Pause' : '▶ Play'; };
    AI.$('[data-r=play]', root).onclick = () => setPlay(!playing);
    AI.$('[data-r=step]', root).onclick = () => { setPlay(false); cur = Math.min(4, cur + 1); st = 0; dirty = true; };
    AI.$('[data-r=back]', root).onclick = () => { setPlay(false); cur = Math.max(0, cur - 1); st = 0; dirty = true; };
    AI.$('[data-r=reset]', root).onclick = () => { cur = 0; st = 0; setPlay(true); dirty = true; };
    window.addEventListener('themechange', () => (dirty = true));
    AI.loop(root, (dt) => {
      let anim = false;
      if (playing) { st += dt; const D = +q('#sp').value; if (st >= D) { st = 0; if (cur < 4) cur++; else if (cur === 4) { cur = 5; } else { cur = 0; } } anim = true; }
      if (!dirty && !anim) return; dirty = false;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const cs = Math.min(40, (c.w * 0.5 - 70) / 4), x0 = 34, y0 = 40, y1 = y0 + 4 * cs + 46;
      let mx = 1e-9; S.forEach((r) => r.forEach((v) => { if (Number.isFinite(v)) mx = Math.max(mx, Math.abs(v)); }));
      const heat = (x, y, fn, label) => {
        txt(g, label, x + 2 * cs, y - 22, C.fg, 'center', 'bold 12px sans-serif');
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          const [col, val] = fn(i, j); g.fillStyle = col; g.fillRect(x + j * cs, y + i * cs, cs - 2, cs - 2);
          txt(g, val, x + j * cs + cs / 2 - 1, y + i * cs + cs / 2 + 4, C.fg, 'center', '11px monospace');
        }
        for (let j = 0; j < 4; j++) txt(g, 'k' + (j + 1), x + j * cs + cs / 2, y - 6, C.muted, 'center', '10px sans-serif');
        for (let i = 0; i < 4; i++) txt(g, 'q' + (i + 1), x - 5, y + i * cs + cs / 2 + 4, i === qi ? C.orange : C.muted, 'right', i === qi ? 'bold 11px sans-serif' : '10px sans-serif');
        g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(x - 2, y + qi * cs - 2, 4 * cs + 2, cs + 2);
      };
      const cB = AI.rgb(C.blue), cO = AI.rgb(C.orange), cG = AI.rgb(C.bg3);
      heat(x0, y0, (i, j) => { const v = S[i][j]; if (!Number.isFinite(v)) return [AI.mix(C.bg, C.bg3, 0.6), '−∞']; const t = v / mx; return [t >= 0 ? lerpc(cG, cO, t) : lerpc(cG, cB, -t), AI.fmt(v, 2)]; }, q('#sc').checked ? 'scores  QKᵀ/√d_k' : 'scores  QKᵀ');
      heat(x0, y1, (i, j) => [lerpc(cG, cO, Aw[i][j]), AI.fmt(Aw[i][j], 2)], 'weights  softmax(row)');
      // right: vector plane
      const px0 = x0 + 4 * cs + 40, pw = c.w - px0 - 14, ph = c.h - 50, cx = px0 + pw / 2, cy = 30 + ph / 2;
      const acc = [[0, 0]]; for (let j = 0; j < 4; j++) acc.push([acc[j][0] + Aw[qi][j] * V[j][0], acc[j][1] + Aw[qi][j] * V[j][1]]);
      let ext = 0.5; V.forEach((v) => (ext = Math.max(ext, Math.abs(v[0]), Math.abs(v[1])))); acc.forEach((p) => (ext = Math.max(ext, Math.abs(p[0]), Math.abs(p[1])))); ext *= 1.15;
      const sc = Math.min(pw, ph) / 2 / ext, X = (x) => cx + x * sc, Y = (y) => cy - y * sc;
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(px0, 30, pw, ph); g.beginPath(); g.moveTo(px0, cy); g.lineTo(px0 + pw, cy); g.moveTo(cx, 30); g.lineTo(cx, 30 + ph); g.stroke();
      txt(g, `value space: build o${qi + 1} = Σⱼ a${qi + 1}ⱼ vⱼ`, px0 + pw / 2, 20, C.fg, 'center', 'bold 12px sans-serif');
      V.forEach((v, j) => { AI.arrow(g, X(0), Y(0), X(v[0]), Y(v[1]), AI.mix(C.purple, C.bg, 0.55), 1.5); txt(g, 'v' + (j + 1), X(v[0]) + 4, Y(v[1]) - 4, C.purple, 'left', '11px sans-serif'); });
      let capTxt = 'Row <b>q' + (qi + 1) + '</b> of the weights says how much of each value to mix. Press Play or Step to watch the weighted sum build tip-to-tail.';
      const done = Math.min(cur, 4);
      for (let j = 0; j < 4; j++) {
        const f = j < done ? 1 : j === done && cur < 4 ? AI.smooth(st / (+q('#sp').value * 0.8)) : 0;
        if (f <= 0) continue; const a = acc[j], b = acc[j + 1], e = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
        if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6) AI.arrow(g, X(a[0]), Y(a[1]), X(e[0]), Y(e[1]), [C.blue, C.green, C.orange, C.pink][j], 3);
        if (j === done && cur < 4) capTxt = `<b>Term ${j + 1}:</b> a<sub>${qi + 1}${j + 1}</sub>·v${j + 1} = ${AI.fmt(Aw[qi][j], 3)} × (${AI.fmt(V[j][0], 2)}, ${AI.fmt(V[j][1], 2)}) = (${AI.fmt(Aw[qi][j] * V[j][0], 3)}, ${AI.fmt(Aw[qi][j] * V[j][1], 3)}); running sum = (${AI.fmt(b[0], 3)}, ${AI.fmt(b[1], 3)}).`;
      }
      if (cur >= 4) { g.fillStyle = '#fff'; g.strokeStyle = C.red; g.lineWidth = 3; g.beginPath(); g.arc(X(acc[4][0]), Y(acc[4][1]), 7, 0, 7); g.fill(); g.stroke(); txt(g, `o${qi + 1}`, X(acc[4][0]) + 10, Y(acc[4][1]) - 8, C.red, 'left', 'bold 12px sans-serif'); capTxt = `<b>Output o${qi + 1} = (${AI.fmt(O[qi][0], 3)}, ${AI.fmt(O[qi][1], 3)})</b> — a blend of the value vectors, weighted by how well query ${qi + 1} matches each key. Weights in row ${qi + 1}: ${Aw[qi].map((v) => AI.fmt(v, 2)).join(', ')} (sum = 1).`; }
      q('#cap').innerHTML = capTxt;
    }, { fps: 30 });
  };

  /* ================= 7.7  why divide by sqrt(d_k) ================= */
  AI.labs.attnscale = function (root) {
    const { view, ctl } = AI.labShell(root, 'Why divide by √d_k: softmax saturation', 'one query against 8 random keys; entries have mean 0 and variance 1',
      `<canvas style="height:300px"></canvas>`,
      `${sl('e', 'key dimension d_k = 2^k, k =', 0, 9, 1, 1)}<div class="row"><button class="btn" id="rs">New random query/keys</button></div>
       <div class="readout" id="out" style="white-space:pre-wrap"></div>`);
    const c = AI.canvas($('canvas', view)), q = (s) => $(s, root);
    c.onresize = () => (dirty = true);
    let seed = 3, dirty = true, d = 2, raw = [], wRaw = new Array(8).fill(0.125), wSc = new Array(8).fill(0.125), tR = wRaw, tS = wSc, info = {};
    const softmax = (s) => { const m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); };
    const calc = () => {
      d = 2 ** +q('#e').value; $('#ev', root).textContent = 'd_k = ' + d; const r = AI.rng(seed), N = () => AI.randn(r), qv = Array.from({ length: d }, N);
      raw = Array.from({ length: 8 }, () => { let s = 0; for (let i = 0; i < d; i++) s += qv[i] * N(); return s; });
      let m2 = 0, cnt = 600; for (let t = 0; t < cnt; t++) { let s = 0; for (let i = 0; i < d; i++) s += N() * N(); m2 += s * s; }
      tR = softmax(raw); tS = softmax(raw.map((v) => v / Math.sqrt(d)));
      info = { sd: Math.sqrt(m2 / cnt), rawMax: Math.max(...tR), scMax: Math.max(...tS), gR: 1 - tR.reduce((s, v) => s + v * v, 0), gS: 1 - tS.reduce((s, v) => s + v * v, 0) }; dirty = true;
    };
    wire(root, ['e'], calc); $('#ev', root).textContent = 'd_k = 2'; q('#rs').onclick = () => { seed = (seed * 13 + 5) | 0; calc(); }; calc();
    window.addEventListener('themechange', () => (dirty = true));
    const bars = (g, C, x, y, w, h, arr, col, title) => {
      txt(g, title, x, y - 6, C.fg, 'left', 'bold 12px sans-serif'); const bw = w / 8;
      g.strokeStyle = C.border; g.strokeRect(x, y, w, h);
      arr.forEach((v, i) => { g.fillStyle = col; g.fillRect(x + i * bw + 4, y + h - v * h, bw - 8, v * h); txt(g, AI.fmt(v, 2), x + i * bw + bw / 2, y + h + 13, C.muted, 'center', '10px monospace'); });
    };
    AI.loop(root, (dt) => {
      let mv = false; const k = ease(dt, 5);
      for (let i = 0; i < 8; i++) { const a = tR[i] - wRaw[i], b = tS[i] - wSc[i]; if (Math.abs(a) > 1e-4 || Math.abs(b) > 1e-4) { wRaw[i] += a * k; wSc[i] += b * k; mv = true; } else { wRaw[i] = tR[i]; wSc[i] = tS[i]; } }
      if (!dirty && !mv) return; dirty = false;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); const w = c.w - 60, h = (c.h - 100) / 2;
      bars(g, C, 30, 28, w, h, wRaw, C.red, 'softmax(q·k)  — no scaling');
      bars(g, C, 30, 28 + h + 46, w, h, wSc, C.green, 'softmax(q·k / √d_k) — scaled');
      q('#out').innerHTML = `d_k = ${d}\nstd of q·k (measured over 600 random pairs) = <span class="o">${AI.fmt(info.sd, 2)}</span>   theory √d_k = ${AI.fmt(Math.sqrt(d), 2)}\nlargest weight: unscaled <span class="r">${AI.fmt(info.rawMax, 3)}</span>   scaled <span class="g">${AI.fmt(info.scMax, 3)}</span>\nsoftmax gradient size Σ aⱼ(1−aⱼ): unscaled <span class="r">${AI.fmt(info.gR, 3)}</span>   scaled <span class="g">${AI.fmt(info.gS, 3)}</span>\n<span class="m">uniform weights would be 0.125 each; a one-hot weight vector has zero gradient (saturated).</span>`;
    });
  };

  /* ================= 7.7  positional encoding ================= */
  AI.labs.posenc = function (root) {
    const { view, ctl } = AI.labShell(root, 'Sinusoidal positional encoding', 'row = position p, column = dimension; pair (2i, 2i+1) = (sin, cos)(p / 10000^(2i/d))',
      `<canvas style="height:260px"></canvas><canvas style="height:130px"></canvas>`,
      `${sl('d', 'model dimension d', 8, 128, 8, 64)}${sl('base', 'wavelength base (10000 in the paper)', 100, 20000, 100, 10000)}${sl('p', 'reference position p', 0, 63, 1, 20)}
       <div class="row"><button class="btn" id="sw">▶ Sweep p</button></div>${sl('sp', 'sweep speed (positions / s)', 1, 12, 1, 4)}
       <div class="readout" id="out" style="white-space:pre-wrap"></div>`);
    const [c1, c2] = AI.$$('canvas', view).map((cv) => AI.canvas(cv)), q = (s) => $(s, root), off = document.createElement('canvas');
    c1.onresize = c2.onresize = () => (dirty = true);
    let d = 64, base = 1e4, p = 20, dirty = true, sweep = false, pf = 20, hdirty = true;
    const PE = (pos, dd, bs) => Array.from({ length: dd }, (_, j) => { const i = j >> 1, a = pos / Math.pow(bs, (2 * i) / dd); return j % 2 ? Math.cos(a) : Math.sin(a); });
    const read = () => { d = +q('#d').value; base = +q('#base').value; if (!sweep) { p = +q('#p').value; pf = p; } dirty = true; hdirty = true; };
    wire(root, ['d', 'base', 'p', 'sp'], read);
    q('#sw').onclick = () => { sweep = !sweep; q('#sw').textContent = sweep ? '⏸ Stop sweep' : '▶ Sweep p'; if (!sweep) pf = p = Math.round(pf); };
    window.addEventListener('themechange', () => { dirty = true; hdirty = true; });
    AI.loop(root, (dt) => {
      if (sweep) { pf = (pf + dt * +q('#sp').value) % 64; if (Math.floor(pf) !== p) { p = Math.floor(pf); } dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c1.ctx, cA = AI.rgb(C.blue), cB = AI.rgb(C.orange), cM = AI.rgb(C.bg3);
      if (hdirty) {
        off.width = d; off.height = 64; const ig = off.getContext('2d'), im = ig.createImageData(d, 64);
        for (let r = 0; r < 64; r++) { const v = PE(r, d, base); for (let j = 0; j < d; j++) { const t = v[j], col = t >= 0 ? [cM, cB, t] : [cM, cA, -t], k = (r * d + j) * 4; for (let ch = 0; ch < 3; ch++) im.data[k + ch] = col[0][ch] + (col[1][ch] - col[0][ch]) * col[2]; im.data[k + 3] = 255; } }
        ig.putImageData(im, 0, 0); hdirty = false;
      }
      const pl = 40, pt = 22, pw = c1.w - pl - 10, ph = c1.h - pt - 8;
      g.clearRect(0, 0, c1.w, c1.h); g.imageSmoothingEnabled = false; g.drawImage(off, pl, pt, pw, ph);
      g.strokeStyle = C.fg; g.lineWidth = 2; const yy = pt + (p + 0.5) * (ph / 64); g.beginPath(); g.moveTo(pl - 6, yy); g.lineTo(pl + pw, yy); g.stroke();
      txt(g, `p=${p}`, pl - 8, yy + 4, C.fg, 'right', '11px monospace'); txt(g, 'dimension 0 … d−1 → (fast waves on the left, slow waves on the right)', pl, 14, C.fg, 'left', '11px sans-serif'); txt(g, 'position ↓', 2, pt + 10, C.muted, 'left', '10px sans-serif');
      const g2 = c2.ctx; g2.clearRect(0, 0, c2.w, c2.h); const ref = PE(p, d, base), dots = [];
      for (let o = 0; o < 64; o++) { const v = PE(o, d, base); dots.push(v.reduce((s, x, i) => s + x * ref[i], 0) / (d / 2)); }
      const v = AI.view(c2, 0, 63, -0.6, 1.05, { l: 40, r: 10, t: 16, b: 22 }); AI.axes(g2, v);
      g2.strokeStyle = C.green; g2.lineWidth = 2.5; g2.beginPath(); dots.forEach((y, o) => (o ? g2.lineTo(v.X(o), v.Y(y)) : g2.moveTo(v.X(o), v.Y(y)))); g2.stroke();
      g2.fillStyle = C.fg; g2.beginPath(); g2.arc(v.X(p), v.Y(dots[p]), 5, 0, 7); g2.fill();
      txt(g2, 'PE(p)·PE(o) / (d/2) for every position o: 1 at o = p, fading with distance', 46, 12, C.fg, 'left', '11px sans-serif');
      q('#out').innerHTML = `PE(p=${p}) first four numbers: [${ref.slice(0, 4).map((x) => AI.fmt(x, 3)).join(', ')}]\nwavelengths run from 2π ≈ 6.3 positions (dim 0) up to 2π·${AI.fmt(base, 0)}^((d−2)/d) ≈ ${AI.fmt(2 * Math.PI * Math.pow(base, (d - 2) / d), 0)} (last pair)\nPE(p)·PE(p) / (d/2) = ${AI.fmt(dots[p], 3)} (each sin²+cos² pair adds exactly 1)`;
    }, { fps: 30 });
  };
})();
