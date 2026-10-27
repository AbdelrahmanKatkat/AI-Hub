/* Module 0 labs for "What machine learning is": rules vs learning, learning from examples, task gallery,
 * table explorer, split/overfitting, learning curves, end-to-end pipeline flow.
 * All animation is dt-based with speed slider, pause/step/reset and captions. Cost numbers are illustrative estimates. */
(function () {
  const fx = (AI.fx = {});
  /* ---------- small numeric helpers ---------- */
  fx.solve = (A, b) => {
    const n = b.length; A = A.map((r, i) => [...r, b[i]]);
    for (let i = 0; i < n; i++) {
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; if (Math.abs(A[i][i]) < 1e-14) continue;
      for (let r = 0; r < n; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let c = i; c <= n; c++) A[r][c] -= f * A[i][c]; }
    }
    return A.map((r, i) => (Math.abs(r[i]) < 1e-14 ? 0 : r[n] / r[i]));
  };
  /* polynomial least squares on t = x/3 (so x in [-3,3] -> t in [-1,1]); tiny ridge for stability */
  fx.polyfit = (xs, ys, deg, lam = 1e-8) => {
    const A = [], b = [], n = xs.length, pw = xs.map((x) => { const t = x / 3, a = [1]; for (let k = 1; k <= 2 * deg; k++) a.push(a[k - 1] * t); return a; });
    for (let i = 0; i <= deg; i++) { A.push([]); let s = 0; for (let k = 0; k < n; k++) s += ys[k] * pw[k][i]; b.push(s); for (let j = 0; j <= deg; j++) { let t = 0; for (let k = 0; k < n; k++) t += pw[k][i + j]; A[i].push(t + (i === j ? lam * Math.max(1, n) : 0)); } }
    return fx.solve(A, b);
  };
  fx.polyval = (c, x) => { const t = x / 3; let s = 0; for (let i = c.length - 1; i >= 0; i--) s = s * t + c[i]; return s; };
  fx.mse = (c, xs, ys) => { if (!xs.length) return NaN; let s = 0; for (let i = 0; i < xs.length; i++) s += (fx.polyval(c, xs[i]) - ys[i]) ** 2; return s / xs.length; };
  fx.cap = (g, x, y, w, text, h = 22) => { const C = AI.col(); g.save(); g.fillStyle = C.bg2; g.globalAlpha = 0.92; g.fillRect(x, y, w, h); g.globalAlpha = 1; g.fillStyle = C.fg; g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillText(text, x + 6, y + h / 2 + 4); g.restore(); };
  fx.player = (root, h) => {
    const q = (s) => root.querySelector(s), st = { playing: true, speed: 1 };
    const lab = () => (q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play');
    q('#play').onclick = () => { st.playing = !st.playing; lab(); };
    q('#step').onclick = () => { st.playing = false; lab(); h.step && h.step(); };
    q('#reset').onclick = () => { h.reset && h.reset(); };
    q('#sp').oninput = () => { st.speed = +q('#sp').value; q('#spv').textContent = '×' + AI.fmt(st.speed, 2); };
    q('#spv').textContent = '×1';
    return st;
  };
  fx.PLAYCTL = `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  fx.truth = (kind) => (kind === 'curve' ? (x) => 1.8 * Math.sin(1.1 * x) : (x) => 1.2 * x + 0.5);

  /* =============== Lab: rules versus learning =============== */
  AI.labs.rulesvslearn = function (root) {
    const { view, ctl } = AI.labShell(root, 'Rules versus learning', 'a hand-written rule (dashed) against a model learned from examples (solid)',
      `<canvas id="cv" style="height:340px"></canvas><canvas id="ac" style="height:130px"></canvas>`,
      `<label>Examples shown, n <span class="v" id="nv"></span><input type="range" id="n" min="0" max="200" step="1" value="6"></label>
       <label>Your rule: direction <span class="v" id="av"></span><input type="range" id="ang" min="-180" max="180" step="1" value="0"></label>
       <label>Your rule: offset <span class="v" id="ov"></span><input type="range" id="off" min="-0.6" max="0.6" step="0.01" value="0"></label>
       <div class="row"><button class="btn" id="shift">Shift the world</button></div>` + fx.PLAYCTL + `<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#cv')), c2 = AI.canvas(q('#ac')), R = 1.3, m = (c) => ML.toPx(c.w, c.h, R);
    let world = 0, seed = 5, pool = [], test = [], shown = 0, w = [0, 0], b = 0, steps = 0, hist = [], hT = 0, lastN = -1, pulse = 0;
    const gen = () => {
      const r = AI.rng(seed * 7 + world * 13), phi = world ? 2.35 : 0.6, cx = 0.45 * Math.cos(phi), cy = 0.45 * Math.sin(phi), mk = (k) => { const out = []; for (let i = 0; i < k; i++) { const c = i % 2, s = c ? 1 : -1; out.push({ x: s * cx + 0.3 * AI.randn(r), y: s * cy + 0.3 * AI.randn(r), c }); } return out; };
      pool = mk(200); test = mk(240);
    };
    const accOf = (pred) => test.reduce((s, p) => s + (pred(p) === p.c ? 1 : 0), 0) / test.length;
    const rulePred = (p) => { const a = (+q('#ang').value * Math.PI) / 180; return Math.cos(a) * p.x + Math.sin(a) * p.y - +q('#off').value > 0 ? 1 : 0; };
    const modPred = (p) => (w[0] * p.x + w[1] * p.y + b > 0 ? 1 : 0);
    const stepOnce = () => {
      const k = Math.max(2, Math.floor(shown)); let g0 = 0, g1 = 0, gb = 0;
      for (let i = 0; i < k; i++) { const p = pool[i], e = 1 / (1 + Math.exp(-(w[0] * p.x + w[1] * p.y + b))) - p.c; g0 += e * p.x; g1 += e * p.y; gb += e; }
      const lr = 2.2; w[0] -= (lr * g0) / k; w[1] -= (lr * g1) / k; b -= (lr * gb) / k; steps++;
    };
    const reset = () => { w = [0, 0]; b = 0; steps = 0; hist = []; shown = 0; };
    const st = fx.player(root, { step: () => { stepOnce(); }, reset });
    q('#shift').onclick = () => { world = 1 - world; gen(); hist = []; };
    gen();
    const line = (g, mp, a, bb, cc, col, dash) => { // a*x + bb*y + cc = 0
      g.strokeStyle = col; g.lineWidth = 2.5; g.setLineDash(dash || []); g.beginPath();
      if (Math.abs(bb) > Math.abs(a)) { g.moveTo(mp.X(-R), mp.Y(-(a * -R + cc) / bb)); g.lineTo(mp.X(R), mp.Y(-(a * R + cc) / bb)); }
      else if (Math.abs(a) > 1e-9) { g.moveTo(mp.X(-(bb * -R + cc) / a), mp.Y(-R)); g.lineTo(mp.X(-(bb * R + cc) / a), mp.Y(R)); }
      g.stroke(); g.setLineDash([]);
    };
    const tick = AI.ticker();
    AI.loop(root, (dt) => {
      const target = +q('#n').value, sp = st.speed;
      q('#nv').textContent = target; q('#av').textContent = q('#ang').value + '°'; q('#ov').textContent = q('#off').value;
      if (st.playing) {
        shown += Math.sign(target - shown) * Math.min(Math.abs(target - shown), dt * 5 * sp); // examples arrive ~5 per second
        const n = tick(dt, 12 * sp); for (let i = 0; i < n; i++) stepOnce();
      }
      const C = AI.col(), cols = ML.COLORS(2), mp = m(c1), k = Math.floor(shown);
      if (k !== lastN) { lastN = k; pulse = 1; } pulse = Math.max(0, pulse - dt * 2);
      hT += dt; if (st.playing && hT > 0.4) { hT = 0; hist.push([shown, accOf(modPred)]); if (hist.length > 400) hist.shift(); }
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      g.fillStyle = C.bg2; g.fillRect(0, 0, c1.w, c1.h);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(mp.X(0), 0); g.lineTo(mp.X(0), c1.h); g.moveTo(0, mp.Y(0)); g.lineTo(c1.w, mp.Y(0)); g.stroke();
      line(g, mp, Math.cos((+q('#ang').value * Math.PI) / 180), Math.sin((+q('#ang').value * Math.PI) / 180), -+q('#off').value, C.yellow, [7, 5]);
      if (steps > 0) line(g, mp, w[0], w[1], b, C.green);
      for (let i = 0; i < k; i++) { const p = pool[i], r = 4.5 + (i === k - 1 ? 4 * pulse : 0); g.fillStyle = cols[p.c]; g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.arc(mp.X(p.x), mp.Y(p.y), r, 0, 7); g.fill(); g.globalAlpha = 0.6; g.stroke(); g.globalAlpha = 1; }
      g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText('dashed = your rule   solid = learned model   colour = true class', 8, c1.h - 8);
      fx.cap(g, 6, 6, Math.min(c1.w - 12, 560), k < 3 ? '① few examples: the learner has almost nothing to go on.' : '② each example nudges the model: it re-fits the line as more data arrives.');
      // accuracy chart
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v = AI.view(c2, 0, 200, 0.3, 1.0, { l: 40, r: 10, t: 18, b: 22 }); AI.axes(g, v);
      const ra = accOf(rulePred); g.strokeStyle = C.yellow; g.setLineDash([6, 4]); g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(0), v.Y(ra)); g.lineTo(v.X(200), v.Y(ra)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => (i ? g.lineTo(v.X(h[0]), v.Y(h[1])) : g.moveTo(v.X(h[0]), v.Y(h[1])))); g.stroke();
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('accuracy on fresh data vs examples n  (green = learned, dashed = your rule)', 46, 12);
      q('#out').innerHTML = `<span class="m">world</span> ${world ? 'after the shift' : 'original'}   <span class="m">n</span> ${k}   <span class="m">steps</span> ${steps}\n<span class="o">rule accuracy</span> ${AI.fmt(ra * 100, 1)}%   <span class="g">learned accuracy</span> ${AI.fmt(accOf(modPred) * 100, 1)}%`;
    }, { fps: 40 });
  };

  /* =============== Lab: learning from examples (drag the points) =============== */
  AI.labs.fitexamples = function (root) {
    const { view, ctl } = AI.labShell(root, 'Learning from examples', 'drag the points · raise n · watch the model and the error on new data',
      `<canvas id="cv" style="height:320px;touch-action:none;cursor:grab"></canvas><canvas id="ec" style="height:150px"></canvas>`,
      `<label>Examples n <span class="v" id="nv"></span><input type="range" id="n" min="2" max="60" step="1" value="8"></label>
       <label>True pattern <select id="kind"><option value="line">a straight line</option><option value="curve">a curve (the line cannot fit it)</option></select></label>
       <label>Noise <span class="v" id="sv"></span><input type="range" id="sig" min="0.1" max="2" step="0.05" value="0.8"></label>
       <div class="row"><button class="btn" id="new">New examples</button></div>` + fx.PLAYCTL + `<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#cv')), c2 = AI.canvas(q('#ec'));
    const CAPS = ['① Predict: the model (a line ŷ = w·x + b) draws its current guess.', '② Measure: red segments are the errors; their squares average to the loss.', '③ Update: w and b nudge a little downhill.'];
    let seed = 2, xs = [], ys = [], birth = [], w = 0, b = 0, ws = 0, bs = 0, phase = 0, phT = 0, steps = 0, dirty = true, curve = null, dragI = -1, dT = 0;
    const gen = () => { const r = AI.rng(seed * 31 + 7), f = fx.truth(q('#kind').value); xs = []; ys = []; birth = []; for (let i = 0; i < 60; i++) { const x = -2.8 + 5.6 * r(); xs.push(x); ys.push(f(x) + +q('#sig').value * AI.randn(r)); birth.push(-10); } dirty = true; };
    const resetModel = () => { w = 0; b = 0; ws = 0; bs = 0; steps = 0; phase = 0; phT = 0; };
    const lsq = (m) => { let sx = 0, sy = 0; for (let i = 0; i < m; i++) { sx += xs[i]; sy += ys[i]; } const mx = sx / m, my = sy / m; let sxy = 0, sxx = 0; for (let i = 0; i < m; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; } const ww = sxx > 1e-9 ? sxy / sxx : 0; return [ww, my - ww * mx]; };
    const testErr = (ww, bb, f, sig) => { let s = 0; for (let i = 0; i <= 60; i++) { const x = -2.8 + (5.6 * i) / 60; s += (ww * x + bb - f(x)) ** 2; } return s / 61 + sig * sig; };
    const gd = () => { const n = +q('#n').value; let gw = 0, gb = 0; for (let i = 0; i < n; i++) { const e = w * xs[i] + b - ys[i]; gw += e * xs[i]; gb += e; } w -= (0.1 * 2 * gw) / n; b -= (0.1 * 2 * gb) / n; steps++; };
    const st = fx.player(root, { step: () => { gd(); }, reset: resetModel });
    q('#new').onclick = () => { seed += 1; gen(); resetModel(); };
    q('#kind').onchange = () => { gen(); resetModel(); }; q('#sig').oninput = () => { gen(); };
    let lastN = 8; q('#n').oninput = () => { const n = +q('#n').value; for (let i = lastN; i < n; i++) birth[i] = performance.now() / 1000; lastN = n; dirty = true; };
    for (let i = 0; i < 8; i++) birth[i] = -10;
    gen();
    const pos = (e) => { const r = q('#cv').getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    q('#cv').onpointerdown = (e) => { const v = AI.view(c1, -3.2, 3.2, -6, 6), [px, py] = pos(e), n = +q('#n').value; let bi = -1, bd = 18; for (let i = 0; i < n; i++) { const d = Math.hypot(v.X(xs[i]) - px, v.Y(ys[i]) - py); if (d < bd) { bd = d; bi = i; } } if (bi >= 0) { dragI = bi; q('#cv').setPointerCapture(e.pointerId); q('#cv').style.cursor = 'grabbing'; } };
    q('#cv').onpointermove = (e) => { if (dragI < 0) return; const v = AI.view(c1, -3.2, 3.2, -6, 6), [px, py] = pos(e); xs[dragI] = AI.clamp(v.ix(px), -3, 3); ys[dragI] = AI.clamp(v.iy(py), -6, 6); dirty = true; };
    q('#cv').onpointerup = () => { dragI = -1; q('#cv').style.cursor = 'grab'; };
    AI.loop(root, (dt, now) => {
      const n = +q('#n').value, sp = st.speed, f = fx.truth(q('#kind').value), sig = +q('#sig').value;
      q('#nv').textContent = n; q('#sv').textContent = AI.fmt(sig, 2);
      if (st.playing) { phT += dt * sp; if (phT >= 1.2) { phT = 0; phase = (phase + 1) % 3; if (phase === 0) for (let i = 0; i < 3; i++) gd(); } }
      const kk = 1 - Math.exp(-dt * 4); ws += (w - ws) * kk; bs += (b - bs) * kk;
      if (dirty && (dT += dt) > 0.12) { dT = 0; dirty = false; curve = []; for (let m = 2; m <= 60; m++) { const [ww, bb] = lsq(m); let tr = 0; for (let i = 0; i < m; i++) tr += (ww * xs[i] + bb - ys[i]) ** 2; curve.push([m, tr / m, testErr(ww, bb, f, sig)]); } }
      const C = AI.col(); let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = AI.view(c1, -3.2, 3.2, -6, 6); AI.axes(g, v);
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.lineWidth = 1.5; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = -3.2 + (6.4 * i) / 100; i ? g.lineTo(v.X(x), v.Y(f(x))) : g.moveTo(v.X(x), v.Y(f(x))); } g.stroke(); g.setLineDash([]);
      const show = phase === 1;
      let tr = 0; for (let i = 0; i < n; i++) { const yh = ws * xs[i] + bs, e = yh - ys[i]; tr += e * e; g.strokeStyle = C.red; g.globalAlpha = show ? 0.95 : 0.3; g.lineWidth = show ? 1.6 : 1; g.beginPath(); g.moveTo(v.X(xs[i]), v.Y(ys[i])); g.lineTo(v.X(xs[i]), v.Y(yh)); g.stroke(); g.globalAlpha = 1; }
      g.strokeStyle = C.orange; g.lineWidth = 2.8; g.beginPath(); g.moveTo(v.X(-3.2), v.Y(-3.2 * ws + bs)); g.lineTo(v.X(3.2), v.Y(3.2 * ws + bs)); g.stroke();
      for (let i = 0; i < n; i++) { const age = now - birth[i], r = 4 + (age < 1 ? 5 * (1 - age) : 0); g.fillStyle = C.blue; g.beginPath(); g.arc(v.X(xs[i]), v.Y(ys[i]), r, 0, 7); g.fill(); g.strokeStyle = C.fg; g.globalAlpha = 0.5; g.stroke(); g.globalAlpha = 1; }
      fx.cap(g, v.pad.l + 4, v.pad.t + 4, Math.min(v.pw - 8, 520), CAPS[phase]);
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('dashed grey = the true pattern (hidden from the learner)', v.pad.l + 6, v.pad.t + 40);
      // data vs error chart
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v2 = AI.view(c2, 0, 60, 0, 6, { l: 40, r: 10, t: 18, b: 22 }); AI.axes(g, v2);
      if (curve) {
        g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); curve.forEach((p, i) => { const y = Math.min(6, p[1]); i ? g.lineTo(v2.X(p[0]), v2.Y(y)) : g.moveTo(v2.X(p[0]), v2.Y(y)); }); g.stroke();
        g.strokeStyle = C.orange; g.beginPath(); curve.forEach((p, i) => { const y = Math.min(6, p[2]); i ? g.lineTo(v2.X(p[0]), v2.Y(y)) : g.moveTo(v2.X(p[0]), v2.Y(y)); }); g.stroke();
        g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v2.X(0), v2.Y(sig * sig)); g.lineTo(v2.X(60), v2.Y(sig * sig)); g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.green; g.beginPath(); g.moveTo(v2.X(n), v2.pad.t); g.lineTo(v2.X(n), v2.pad.t + v2.ph); g.stroke();
      }
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('error vs examples: blue = on examples, orange = on new data, dashed = noise σ²', 46, 12);
      const te = testErr(ws, bs, f, sig);
      q('#out').innerHTML = `<span class="b">w</span> ${AI.fmt(ws, 3)}  <span class="b">b</span> ${AI.fmt(bs, 3)}  <span class="m">step</span> ${steps}\n<span class="m">loss on examples</span> <span class="o">${AI.fmt(tr / n, 3)}</span>\n<span class="m">expected error on new data</span> <span class="g">${AI.fmt(te, 3)}</span>  (floor ${AI.fmt(sig * sig, 3)})`;
    }, { fps: 40 });
  };

  /* =============== Lab: task gallery (same data, four framings) =============== */
  AI.labs.taskgallery = function (root) {
    const { view, ctl } = AI.labShell(root, 'One table, four questions', 'students: hours studied (x), hours slept (y) · click the plot to place a new student',
      `<canvas id="cv" style="height:380px;cursor:crosshair"></canvas>`,
      `<div class="chips" id="modes"></div>
       <label>Clusters k <span class="v" id="kv"></span><input type="range" id="k" min="2" max="5" step="1" value="2"></label>` + fx.PLAYCTL + `<div class="readout" id="out"></div>
       <div class="caption" id="why"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv')), R = 1.3;
    const MODES = [['cls', 'Classification'], ['reg', 'Regression'], ['clu', 'Clustering'], ['rank', 'Ranking']];
    const WHY = {
      cls: 'Label given: passed / failed. The model learns a boundary and answers with a category and a probability.',
      reg: 'Target given: the exam score (a number). The model learns a plane and answers with a number.',
      clu: 'No labels at all. The learner groups similar students; the groups have no names until a human gives them one.',
      rank: 'We only need an order: who is most likely to score highly? Rank by the regression score and keep the top 10.',
    };
    const r0 = AI.rng(11), H = [], S = [], SC = [], PASS = [], X = [];
    for (let i = 0; i < 30; i++) { const h = 10 * r0(), s = 4 + 6 * r0(), sc = AI.clamp(20 + 5.5 * h + 2.2 * s + 4 * AI.randn(r0), 5, 100); H.push(h); S.push(s); SC.push(sc); PASS.push(sc >= 60 ? 1 : 0); X.push([(h - 5) / 5, (s - 7) / 3]); }
    const mean = SC.reduce((a, b) => a + b, 0) / SC.length;
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bb = [0, 0, 0];
    X.forEach((p, i) => { const f = [1, p[0], p[1]]; for (let a = 0; a < 3; a++) { bb[a] += f[a] * SC[i]; for (let d = 0; d < 3; d++) A[a][d] += f[a] * f[d]; } });
    const beta = fx.solve(A, bb), fit = (u, v) => beta[0] + beta[1] * u + beta[2] * v;
    let mode = 'cls', lg, km, steps = 0, tReg = 0, phT = 0, cs = [], ch = -1, still = 0, kmPhase = '', rankN = 0, query = [0, 0], heatKey = '', heatT = 0, accT = 0;
    const hc = document.createElement('canvas'), tiny = document.createElement('canvas'); tiny.width = tiny.height = 36;
    const init = () => {
      steps = 0; tReg = 0; phT = 0; rankN = 0; ch = -1; still = 0; heatKey = '';
      lg = ML.logistic(X, PASS, {});
      km = ML.kmeans(X, +q('#k').value, 3); cs = km.C.map((p) => [...p]); kmPhase = 'init';
    };
    const st = fx.player(root, { step: () => advance(1), reset: init });
    const advance = (n) => {
      if (mode === 'cls') for (let i = 0; i < n; i++) { lg.step(1.2); steps++; }
      else if (mode === 'clu') { ch = km.step(); kmPhase = km.phase === 'update' ? 'assigned' : 'updated'; if (ch === 0) still++; else if (ch > 0) still = 0; }
      else if (mode === 'reg') tReg = Math.min(1, tReg + 0.25);
      else if (mode === 'rank') rankN = Math.min(10, rankN + 1);
    };
    const mp = () => ML.toPx(c.w, c.h, R);
    const regCol = (val) => { const C = AI.col(), a = AI.rgb(C.blue), b = AI.rgb(C.orange), t = AI.clamp((val - 20) / 80, 0, 1); return a.map((x, i) => x + (b[i] - x) * t); };
    const paintHeat = () => {
      const C = AI.col(); hc.width = c.w; hc.height = c.h; const g = hc.getContext('2d');
      if (mode === 'cls') ML.boundary(g, c.w, c.h, (x, y) => lg.proba([x, y]), { range: R, res: 44, K: 2 });
      else if (mode === 'reg' || mode === 'rank') {
        const tg = tiny.getContext('2d'), im = tg.createImageData(36, 36), bg = AI.rgb(C.bg), t = mode === 'rank' ? 1 : tReg;
        for (let j = 0; j < 36; j++) for (let i = 0; i < 36; i++) {
          const x = ((i + 0.5) / 36) * 2 * R - R, y = R - ((j + 0.5) / 36) * 2 * R, col = regCol(mean + t * (fit(x, y) - mean));
          for (let k = 0; k < 3; k++) im.data[(j * 36 + i) * 4 + k] = bg[k] + (col[k] - bg[k]) * (mode === 'rank' ? 0.28 : 0.6); im.data[(j * 36 + i) * 4 + 3] = 255;
        }
        tg.putImageData(im, 0, 0); g.imageSmoothingEnabled = true; g.drawImage(tiny, 0, 0, c.w, c.h);
      } else { g.fillStyle = C.bg2; g.fillRect(0, 0, c.w, c.h); }
    };
    const setMode = (m) => { mode = m; AI.$$('#modes button', root).forEach((b) => b.classList.toggle('primary', b.dataset.m === m)); q('#why').textContent = WHY[m]; init(); };
    const mb = q('#modes'); MODES.forEach(([k, name]) => { const b = document.createElement('button'); b.className = 'btn small'; b.dataset.m = k; b.textContent = name; b.onclick = () => setMode(k); mb.appendChild(b); });
    q('#k').oninput = () => { if (mode === 'clu') init(); };
    q('#cv').onpointerdown = (e) => { const r = q('#cv').getBoundingClientRect(), m = mp(); query = [AI.clamp(m.ix(e.clientX - r.left), -1, 1), AI.clamp(m.iy(e.clientY - r.top), -1, 1)]; };
    setMode('cls');
    const tick = AI.ticker(); const PH = 1.6;
    AI.loop(root, (dt) => {
      const sp = st.speed, C = AI.col(), m = mp(); q('#kv').textContent = q('#k').value;
      if (st.playing) {
        if (mode === 'cls') { const n = tick(dt, 10 * sp); advance(n); }
        else if (mode === 'reg') tReg = Math.min(1, tReg + dt * 0.2 * sp);
        else if (mode === 'clu') { phT += dt * sp; if (phT >= PH && still < 2) { phT = 0; advance(1); } }
        else { phT += dt * sp; if (phT >= 0.7) { phT = 0; advance(1); } }
      }
      const kk = 1 - Math.exp(-dt * 4); if (mode === 'clu') km.C.forEach((p, j) => { cs[j][0] += (p[0] - cs[j][0]) * kk; cs[j][1] += (p[1] - cs[j][1]) * kk; });
      const key = mode + '|' + steps + '|' + AI.fmt(tReg, 2) + '|' + c.w + '|' + c.h;
      if (key !== heatKey && (heatT += dt) > 0.1) { heatKey = key; heatT = 0; paintHeat(); }
      const g = c.ctx; g.clearRect(0, 0, c.w, c.h); g.fillStyle = C.bg2; g.fillRect(0, 0, c.w, c.h);
      if (hc.width) g.drawImage(hc, 0, 0, c.w, c.h);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(m.X(0), 0); g.lineTo(m.X(0), c.h); g.moveTo(0, m.Y(0)); g.lineTo(c.w, m.Y(0)); g.stroke();
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'right'; g.fillText('hours studied (0 … 10) →', c.w - 6, c.h - 6); g.textAlign = 'left'; g.fillText('↑ hours slept (4 … 10)', 6, c.h - 6);
      const cols = ML.COLORS(5), order = X.map((_, i) => i).sort((a, b) => fit(X[b][0], X[b][1]) - fit(X[a][0], X[a][1]));
      if (mode === 'clu' && kmPhase === 'assigned') { g.globalAlpha = 0.35; g.lineWidth = 1; X.forEach((p, i) => { const j = km.labels[i]; if (j < 0) return; g.strokeStyle = cols[j]; g.beginPath(); g.moveTo(m.X(p[0]), m.Y(p[1])); g.lineTo(m.X(cs[j][0]), m.Y(cs[j][1])); g.stroke(); }); g.globalAlpha = 1; }
      X.forEach((p, i) => {
        let col = C.muted;
        if (mode === 'cls') col = PASS[i] ? C.orange : C.blue; else if (mode === 'reg') { const rgb = regCol(SC[i]); col = `rgb(${rgb.map(Math.round)})`; } else if (mode === 'clu') col = km.labels[i] >= 0 ? cols[km.labels[i]] : C.muted; else col = C.fg;
        g.fillStyle = col; g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 5.5, 0, 7); g.fill(); g.globalAlpha = 0.7; g.stroke(); g.globalAlpha = 1;
        if (mode === 'cls' && steps > 0 && (lg.proba(p)[1] > 0.5 ? 1 : 0) !== PASS[i]) { g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 9, 0, 7); g.stroke(); }
      });
      if (mode === 'rank') order.slice(0, rankN).forEach((i, r) => { const p = X[i]; g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 10, 0, 7); g.stroke(); g.fillStyle = C.fg; g.font = 'bold 11px sans-serif'; g.textAlign = 'left'; g.fillText('#' + (r + 1), m.X(p[0]) + 11, m.Y(p[1]) - 8); });
      if (mode === 'clu') cs.forEach((p, j) => { g.fillStyle = cols[j]; g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(m.X(p[0]), m.Y(p[1]) - 11); g.lineTo(m.X(p[0]) + 10, m.Y(p[1])); g.lineTo(m.X(p[0]), m.Y(p[1]) + 11); g.lineTo(m.X(p[0]) - 10, m.Y(p[1])); g.closePath(); g.fill(); g.stroke(); });
      g.fillStyle = C.fg; g.strokeStyle = C.bg; g.lineWidth = 2; g.beginPath(); g.arc(m.X(query[0]), m.Y(query[1]), 7, 0, 7); g.fill(); g.stroke(); g.font = 'bold 11px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText('new student', m.X(query[0]) + 10, m.Y(query[1]) + 4);
      const hq = (query[0] * 5 + 5), sq = query[1] * 3 + 7;
      const caps = { cls: steps < 3 ? '① start: the boundary knows nothing (50/50 everywhere).' : '② the boundary swings to separate passes (orange) from fails (blue); red = wrong.',
        reg: tReg < 0.99 ? '① the plane tilts from "always the average" toward the best fit; colour = predicted score.' : '② fitted: colour of the background is the predicted score; points show true scores.',
        clu: kmPhase === 'init' ? '① choose k random starting centres (diamonds).' : kmPhase === 'assigned' ? '② assign: every student joins the nearest centre.' : '③ update: each centre moves to the mean of its group.',
        rank: '① score everyone with the regression model, ② sort, ③ keep the best 10 (yellow).' };
      fx.cap(g, 6, 6, Math.min(c.w - 12, 640), caps[mode]);
      let t = `<span class="m">new student</span> hours ${AI.fmt(hq, 1)}, sleep ${AI.fmt(sq, 1)}\n`;
      if (mode === 'cls') { const p = lg.proba(query)[1], acc = X.reduce((s, x, i) => s + ((lg.proba(x)[1] > 0.5 ? 1 : 0) === PASS[i] ? 1 : 0), 0) / X.length; t += `<span class="o">P(pass)</span> ${AI.fmt(p, 2)} → <b>${p > 0.5 ? 'PASS' : 'FAIL'}</b>\n<span class="m">training accuracy</span> ${AI.fmt(acc * 100, 0)}%  step ${steps}`; }
      else if (mode === 'reg') { const p = mean + tReg * (fit(query[0], query[1]) - mean); let mse = 0; X.forEach((x, i) => (mse += (mean + tReg * (fit(x[0], x[1]) - mean) - SC[i]) ** 2)); t += `<span class="o">predicted score</span> <b>${AI.fmt(p, 1)}</b>\n<span class="m">MSE</span> ${AI.fmt(mse / X.length, 1)}`; }
      else if (mode === 'clu') { let best = 0, bd = 1e9; km.C.forEach((p, j) => { const d = (p[0] - query[0]) ** 2 + (p[1] - query[1]) ** 2; if (d < bd) { bd = d; best = j; } }); t += `<span class="o">cluster</span> <b>${best + 1}</b> of ${km.C.length}\n<span class="m">within-cluster spread</span> ${AI.fmt(km.inertia(), 2)}  ${still >= 2 ? '— converged' : ''}`; }
      else { const f = fit(query[0], query[1]); let rk = 1; X.forEach((x) => { if (fit(x[0], x[1]) > f) rk++; }); t += `<span class="o">would rank</span> <b>#${rk}</b> of ${X.length + 1}`; }
      q('#out').innerHTML = t;
    }, { fps: 30 });
  };

  /* =============== Lab: feature / label table explorer =============== */
  AI.labs.tableexplorer = function (root) {
    const { view, ctl } = AI.labShell(root, 'From spreadsheet to X and y', 'edit the table · choose encodings · convert it into the matrix a model eats',
      `<div class="caption">① The table you were given (editable)</div><div id="tbl" style="overflow-x:auto"></div>
       <div class="caption" id="cap2" style="margin-top:10px"></div><div id="mat" style="overflow-x:auto"></div>`,
      `<label>Encode "area" <select id="enc"><option value="onehot">one-hot (A,B,C → 3 columns)</option><option value="int">integer code (A=1, B=2, C=3)</option></select></label>
       <label>Scale numbers <select id="scl"><option value="none">none</option><option value="std">standardise (mean 0, std 1)</option></select></label>
       <div class="caption">Use as feature:</div><div id="fchk" style="display:flex;gap:10px;flex-wrap:wrap;font-size:13px"></div>
       <div class="row"><button class="btn" id="add">+ row</button><button class="btn" id="del">− row</button></div>
       <div class="row"><button class="btn primary" id="conv">▶ Convert</button></div>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>
       <div class="row"><button class="btn" id="step">Step</button></div>
       <div class="readout" id="out"></div><div class="caption" id="warn"></div>`);
    const q = (s) => AI.$(s, root);
    const FEATS = [['size', 'size m²'], ['beds', 'bedrooms'], ['area', 'area'], ['garden', 'garden']];
    let rows = [[50, 1, 'A', 0, 150], [70, 2, 'B', 0, 190], [90, 3, 'A', 1, 250], [110, 3, 'C', 1, 290], [80, 2, 'B', 1, 220]];
    const use = { size: true, beds: true, area: true, garden: true };
    let reveal = 5, playing = false, speed = 1, key = '', prevRev = -1;
    FEATS.forEach(([k, name]) => { const l = document.createElement('label'); l.style.cssText = 'display:flex;gap:4px;align-items:center'; l.innerHTML = `<input type="checkbox" checked> ${name}`; l.firstChild.onchange = (e) => { use[k] = e.target.checked; reveal = rows.length; build(); }; q('#fchk').appendChild(l); });
    const build = () => {
      let h = '<table class="tbl"><tr><th>#</th><th>size m²</th><th>bedrooms</th><th>area</th><th>garden</th><th>price k$ (label)</th></tr>';
      rows.forEach((r, i) => { h += `<tr data-r="${i}"><td>${i + 1}</td><td><input type="number" data-c="0" value="${r[0]}" style="width:64px;color:#111"></td><td><input type="number" data-c="1" value="${r[1]}" style="width:50px;color:#111"></td>
        <td><select data-c="2" style="color:#111">${['A', 'B', 'C'].map((a) => `<option${a === r[2] ? ' selected' : ''}>${a}</option>`).join('')}</select></td>
        <td><select data-c="3" style="color:#111"><option value="0"${r[3] ? '' : ' selected'}>no</option><option value="1"${r[3] ? ' selected' : ''}>yes</option></select></td>
        <td><input type="number" data-c="4" value="${r[4]}" style="width:64px;color:#111"></td></tr>`; });
      q('#tbl').innerHTML = h + '</table>';
      AI.$$('#tbl [data-c]', root).forEach((el) => (el.onchange = () => { const i = +el.closest('tr').dataset.r, c = +el.dataset.c; rows[i][c] = c === 2 ? el.value : +el.value || 0; reveal = rows.length; key = ''; }));
      key = '';
    };
    q('#add').onclick = () => { if (rows.length < 8) { rows.push([60, 2, 'A', 0, 180]); reveal = rows.length; build(); } };
    q('#del').onclick = () => { if (rows.length > 2) { rows.pop(); reveal = rows.length; build(); } };
    q('#conv').onclick = () => { reveal = 0; playing = true; };
    q('#step').onclick = () => { playing = false; reveal = Math.min(rows.length, Math.floor(reveal) + 1); };
    q('#sp').oninput = () => { speed = +q('#sp').value; q('#spv').textContent = '×' + AI.fmt(speed, 2); }; q('#spv').textContent = '×1';
    q('#enc').onchange = q('#scl').onchange = () => { reveal = rows.length; key = ''; };
    build();
    const matrix = () => {
      const enc = q('#enc').value, scl = q('#scl').value, names = [], cols = [];
      const num = (j, nm) => { let v = rows.map((r) => r[j]); if (scl === 'std') { const m = v.reduce((a, b) => a + b, 0) / v.length, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) || 1; v = v.map((x) => (x - m) / sd); } names.push(nm); cols.push(v); };
      if (use.size) num(0, 'size'); if (use.beds) num(1, 'bedrooms');
      if (use.area) { if (enc === 'int') { names.push('area'); cols.push(rows.map((r) => 'ABC'.indexOf(r[2]) + 1)); } else ['A', 'B', 'C'].forEach((a) => { names.push('area=' + a); cols.push(rows.map((r) => (r[2] === a ? 1 : 0))); }); }
      if (use.garden) { names.push('garden'); cols.push(rows.map((r) => r[3])); }
      return { names, cols };
    };
    AI.loop(root, (dt) => {
      if (playing) { reveal += dt * 0.8 * speed; if (reveal >= rows.length) { reveal = rows.length; playing = false; } }
      const N = rows.length, k = `${N}|${AI.fmt(reveal, 2)}|${JSON.stringify(rows)}|${q('#enc').value}|${q('#scl').value}|${JSON.stringify(use)}`;
      if (k === key) return; key = k;
      const { names, cols } = matrix(), d = cols.length, fmt = (v) => (Number.isInteger(v) ? String(v) : AI.fmt(v, 2));
      q('#cap2').textContent = `② The model's view: X is ${N} × ${d} (rows = examples, columns = features), y has ${N} entries`;
      let h = '<table class="tbl"><tr><th></th>' + names.map((n) => `<th>${n}</th>`).join('') + '<th style="border-left:2px solid var(--border)">y = price</th></tr>';
      for (let i = 0; i < N; i++) {
        const on = i < Math.floor(reveal), cur = i === Math.floor(reveal) && reveal < N, st = on ? '' : 'opacity:0.12', hl = on && reveal - i < 1 ? 'background:var(--bg3)' : '';
        h += `<tr style="${st};${hl}"><td>x<sub>${i + 1}</sub></td>${cols.map((c) => `<td>${fmt(c[i])}</td>`).join('')}<td style="border-left:2px solid var(--border)">${rows[i][4]}</td></tr>`;
        void cur;
      }
      q('#mat').innerHTML = h + '</table>';
      AI.$$('#tbl tr[data-r]', root).forEach((tr, i) => (tr.style.background = reveal < N && i === Math.floor(reveal) ? 'var(--bg3)' : ''));
      q('#out').innerHTML = `<span class="m">X shape</span> <b>${N} × ${d}</b>   <span class="m">y shape</span> <b>${N}</b>\n<span class="m">memory (float32)</span> ${N * d * 4} bytes\n<span class="m">types</span> numeric · categorical · binary`;
      q('#warn').innerHTML = q('#enc').value === 'int' && use.area ? '<span style="color:var(--red)">Trap: with codes A=1, B=2, C=3 the model is told that B is the average of A and C and that C is "three times" A.</span>' : (q('#scl').value === 'std' ? 'Standardised columns have mean 0 and standard deviation 1, so size (50…110) no longer outweighs the 0/1 columns.' : '');
    }, { fps: 30 });
  };

  /* =============== Lab: train / validation / test split and polynomial degree =============== */
  AI.labs.overfit = function (root) {
    const { view, ctl } = AI.labShell(root, 'Split the data, bend the model', 'pick the degree with the validation set · peek at the test set only once',
      `<canvas id="cv" style="height:320px"></canvas><canvas id="ec" style="height:170px"></canvas>`,
      `<label>Training share <span class="v" id="tv"></span><input type="range" id="tr" min="20" max="80" step="5" value="50"></label>
       <label>Validation share <span class="v" id="vv"></span><input type="range" id="va" min="10" max="40" step="5" value="25"></label>
       <label>Model flexibility: polynomial degree <span class="v" id="dv"></span><input type="range" id="deg" min="1" max="12" step="1" value="1"></label>
       <div class="row"><button class="btn" id="sweep">Sweep degree 1 → 12</button><button class="btn" id="new">New data</button></div>
       <div class="row"><button class="btn" id="peek">Reveal test error</button></div>` + fx.PLAYCTL + `<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#cv')), c2 = AI.canvas(q('#ec'));
    const N = 60, SIG = 0.5, f = fx.truth('curve');
    let seed = 4, X = [], Y = [], sets = null, coefs = [], tr = [], va = [], te = [], curves = [], cur = null, key = '', sweeping = false, degF = 1, peeks = 0, shownTest = null, lastDeg = 1;
    const gen = () => { const r = AI.rng(seed * 17 + 3); X = []; Y = []; for (let i = 0; i < N; i++) { const x = -2.9 + 5.8 * r(); X.push(x); Y.push(f(x) + SIG * AI.randn(r)); } key = ''; shownTest = null; peeks = 0; };
    const reset = () => { degF = 1; q('#deg').value = 1; sweeping = false; peeks = 0; shownTest = null; };
    const recompute = () => {
      const ntr = Math.round((N * +q('#tr').value) / 100); let nva = Math.round((N * +q('#va').value) / 100); nva = Math.min(nva, N - ntr - 6);
      sets = { a: [0, ntr], b: [ntr, ntr + nva], c: [ntr + nva, N] };
      const sl = (s) => [X.slice(...s), Y.slice(...s)], [tx, ty] = sl(sets.a), [vx, vy] = sl(sets.b), [ex, ey] = sl(sets.c);
      coefs = []; tr = []; va = []; te = []; curves = [];
      for (let d = 1; d <= 12; d++) { const cf = fx.polyfit(tx, ty, d); coefs[d] = cf; tr[d] = fx.mse(cf, tx, ty); va[d] = fx.mse(cf, vx, vy); te[d] = fx.mse(cf, ex, ey); const cv = []; for (let i = 0; i <= 120; i++) cv.push(fx.polyval(cf, -3.2 + (6.4 * i) / 120)); curves[d] = cv; }
      if (!cur) cur = curves[1].slice();
    };
    const st = fx.player(root, { step: () => { degF = Math.min(12, Math.round(degF) + 1); q('#deg').value = degF; }, reset });
    q('#sweep').onclick = () => { degF = 1; q('#deg').value = 1; sweeping = true; st.playing = true; q('#play').textContent = '⏸ Pause'; };
    q('#new').onclick = () => { seed++; gen(); };
    q('#deg').oninput = () => { degF = +q('#deg').value; sweeping = false; };
    q('#peek').onclick = () => { const d = Math.round(degF); peeks++; shownTest = { d, v: te[d] }; };
    gen();
    AI.loop(root, (dt) => {
      if (st.playing && sweeping) { degF += dt * 0.4 * st.speed; if (degF >= 12) { degF = 12; sweeping = false; } q('#deg').value = Math.round(degF); }
      const D = Math.round(degF), k = `${q('#tr').value}|${q('#va').value}|${seed}`;
      if (k !== key) { key = k; recompute(); }
      q('#tv').textContent = q('#tr').value + '%'; q('#vv').textContent = q('#va').value + '%'; q('#dv').textContent = D;
      const C = AI.col(), kk = 1 - Math.exp(-dt * 3), tg = curves[D]; for (let i = 0; i < cur.length; i++) cur[i] += (tg[i] - cur[i]) * kk;
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = AI.view(c1, -3.2, 3.2, -6, 6); AI.axes(g, v);
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.lineWidth = 1.5; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = -3.2 + (6.4 * i) / 100; i ? g.lineTo(v.X(x), v.Y(f(x))) : g.moveTo(v.X(x), v.Y(f(x))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.purple; g.lineWidth = 2.8; g.beginPath(); cur.forEach((y, i) => { const px = v.X(-3.2 + (6.4 * i) / 120), py = v.Y(AI.clamp(y, -9, 9)); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
      const dot = (i, kind) => { const x = v.X(X[i]), y = v.Y(Y[i]); g.beginPath(); g.arc(x, y, 4.5, 0, 7); if (kind === 'a') { g.fillStyle = C.blue; g.fill(); } else if (kind === 'b') { g.fillStyle = C.orange; g.fill(); } else { g.strokeStyle = C.muted; g.lineWidth = 1.5; g.globalAlpha = shownTest ? 1 : 0.5; g.stroke(); g.globalAlpha = 1; } };
      for (let i = 0; i < N; i++) dot(i, i < sets.a[1] ? 'a' : i < sets.b[1] ? 'b' : 'c');
      const trE = tr[D], vaE = va[D], gap = vaE - trE;
      const diag = trE > 0.6 ? 'underfitting: even the training data is fitted badly' : gap > 0.35 + trE ? 'overfitting: great on training points, worse on unseen ones' : 'a healthy fit: training and validation errors are close';
      fx.cap(g, v.pad.l + 4, v.pad.t + 4, Math.min(v.pw - 8, 560), 'degree ' + D + ' → ' + diag);
      g.font = '11px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('● train   ● validation   ○ test (hidden until revealed)   dashed = truth', v.pad.l + 6, v.pad.t + 40);
      // error chart
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v2 = AI.view(c2, 1, 12, -1.5, 3, { l: 40, r: 10, t: 18, b: 22 }), L = (e) => Math.log10(Math.max(e, 0.01)); AI.axes(g, v2);
      const draw = (arr, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let d = 1; d <= 12; d++) { const x = v2.X(d), y = v2.Y(AI.clamp(L(arr[d]), -1.5, 3)); d > 1 ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); };
      draw(tr, C.blue); draw(va, C.orange);
      [[tr, C.blue], [va, C.orange]].forEach(([a, col]) => { g.fillStyle = col; g.beginPath(); g.arc(v2.X(D), v2.Y(AI.clamp(L(a[D]), -1.5, 3)), 5, 0, 7); g.fill(); });
      if (shownTest) { g.fillStyle = C.green; g.beginPath(); g.arc(v2.X(shownTest.d), v2.Y(AI.clamp(L(shownTest.v), -1.5, 3)), 6, 0, 7); g.fill(); }
      g.strokeStyle = C.muted; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(v2.X(D), v2.pad.t); g.lineTo(v2.X(D), v2.pad.t + v2.ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('log₁₀ error vs degree: blue = train, orange = validation, green = test', 46, 12);
      let best = 1; for (let d = 2; d <= 12; d++) if (va[d] < va[best]) best = d;
      q('#out').innerHTML = `<span class="m">sizes</span> train ${sets.a[1]}  validation ${sets.b[1] - sets.b[0]}  test ${N - sets.b[1]}\n<span class="b">train MSE</span> ${AI.fmt(trE, 3)}   <span class="o">validation MSE</span> ${AI.fmt(vaE, 3)}\n<span class="m">best degree by validation</span> ${best}\n<span class="g">test MSE</span> ${shownTest ? AI.fmt(shownTest.v, 3) + ' (degree ' + shownTest.d + ')' : 'hidden'}   <span class="m">peeks</span> ${peeks}${peeks > 1 ? '  <span class="r">— the test set is now contaminated</span>' : ''}`;
    }, { fps: 30 });
  };

  /* =============== Lab: learning curves =============== */
  AI.labs.learncurve = function (root) {
    const { view, ctl } = AI.labShell(root, 'Learning curves', 'grow the dataset: how do the errors on seen and unseen data change?',
      `<canvas id="cv" style="height:300px"></canvas><canvas id="ec" style="height:190px"></canvas>`,
      `<label>Model flexibility: degree <span class="v" id="dv"></span><input type="range" id="deg" min="1" max="12" step="1" value="9"></label>
       <label>Examples n <span class="v" id="nv"></span><input type="range" id="n" min="4" max="120" step="1" value="10"></label>
       <div class="row"><button class="btn" id="new">New data</button></div>` + fx.PLAYCTL + `<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#cv')), c2 = AI.canvas(q('#ec'));
    const M = 120, SIG = 0.5, f = fx.truth('curve'), ORD = 4;
    let seed = 8, X = [], Y = [], perms = [], curve = [], key = '', fitKey = '', cur = new Array(121).fill(0), tgt = null, nF = 10, T = 0;
    const gen = () => { const r = AI.rng(seed * 23 + 1); X = []; Y = []; for (let i = 0; i < M; i++) { const x = -2.9 + 5.8 * r(); X.push(x); Y.push(f(x) + SIG * AI.randn(r)); } perms = [X.map((_, i) => i)]; for (let o = 1; o < ORD; o++) { const p = X.map((_, i) => i), rr = AI.rng(seed * 5 + o); for (let i = M - 1; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } perms.push(p); } key = ''; fitKey = ''; };
    const grid = []; for (let i = 0; i <= 60; i++) grid.push(-2.9 + (5.8 * i) / 60);
    const tErr = (cf) => grid.reduce((s, x) => s + (fx.polyval(cf, x) - f(x)) ** 2, 0) / grid.length + SIG * SIG;
    const recompute = () => {
      const D = +q('#deg').value; curve = [];
      for (let m = 6; m <= M; m += 3) { let a = 0, b = 0; for (let o = 0; o < ORD; o++) { const idx = perms[o].slice(0, m), xs = idx.map((i) => X[i]), ys = idx.map((i) => Y[i]), cf = fx.polyfit(xs, ys, D); a += fx.mse(cf, xs, ys); b += tErr(cf); } curve.push([m, a / ORD, b / ORD]); }
    };
    const reset = () => { nF = 4; q('#n').value = 4; };
    const st = fx.player(root, { step: () => { nF = Math.min(M, Math.floor(nF) + 5); q('#n').value = Math.floor(nF); }, reset });
    q('#new').onclick = () => { seed++; gen(); };
    q('#n').oninput = () => { nF = +q('#n').value; };
    let dT = 0; gen();
    AI.loop(root, (dt) => {
      const D = +q('#deg').value; q('#dv').textContent = D;
      if (st.playing && nF < M) { nF = Math.min(M, nF + dt * 6 * st.speed); q('#n').value = Math.floor(nF); }
      const n = Math.max(4, Math.floor(nF)); q('#nv').textContent = n;
      const k = D + '|' + seed; if (k !== key && (dT += dt) > 0.1) { dT = 0; key = k; recompute(); }
      const fk = D + '|' + seed + '|' + n; let cf = null; if (fk !== fitKey) { fitKey = fk; const idx = perms[0].slice(0, n); tgt = []; cf = fx.polyfit(idx.map((i) => X[i]), idx.map((i) => Y[i]), D); for (let i = 0; i <= 120; i++) tgt.push(fx.polyval(cf, -3.2 + (6.4 * i) / 120)); T = { tr: fx.mse(cf, idx.map((i) => X[i]), idx.map((i) => Y[i])), te: tErr(cf) }; }
      const C = AI.col(), kk = 1 - Math.exp(-dt * 3); if (tgt) for (let i = 0; i < 121; i++) cur[i] += (tgt[i] - cur[i]) * kk;
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = AI.view(c1, -3.2, 3.2, -6, 6); AI.axes(g, v);
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.lineWidth = 1.5; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = -3.2 + (6.4 * i) / 100; i ? g.lineTo(v.X(x), v.Y(f(x))) : g.moveTo(v.X(x), v.Y(f(x))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.purple; g.lineWidth = 2.8; g.beginPath(); cur.forEach((y, i) => { const px = v.X(-3.2 + (6.4 * i) / 120), py = v.Y(AI.clamp(y, -9, 9)); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
      const idx = perms[0]; g.fillStyle = C.blue; for (let i = 0; i < n; i++) { g.beginPath(); g.arc(v.X(X[idx[i]]), v.Y(Y[idx[i]]), 4, 0, 7); g.fill(); }
      const gap = T.te - T.tr, diag = T.te > 0.6 && gap < 0.3 ? 'both errors high and close: underfitting — more data will not help' : gap > 0.4 ? 'big gap: overfitting — more data would help' : 'small gap and low error: a good fit';
      fx.cap(g, v.pad.l + 4, v.pad.t + 4, Math.min(v.pw - 8, 560), 'n = ' + n + ' → ' + diag);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v2 = AI.view(c2, 0, M, 0, 2.5, { l: 40, r: 10, t: 18, b: 22 }); AI.axes(g, v2);
      g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(v2.X(0), v2.Y(SIG * SIG)); g.lineTo(v2.X(M), v2.Y(SIG * SIG)); g.stroke(); g.setLineDash([]);
      [[1, C.blue], [2, C.orange]].forEach(([j, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); let s = false; curve.forEach((p) => { if (p[0] > n) return; const x = v2.X(p[0]), y = v2.Y(Math.min(2.5, p[j])); s ? g.lineTo(x, y) : g.moveTo(x, y); s = true; }); g.stroke(); });
      g.strokeStyle = C.green; g.beginPath(); g.moveTo(v2.X(n), v2.pad.t); g.lineTo(v2.X(n), v2.pad.t + v2.ph); g.stroke();
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('error vs n (mean of 4 shuffles): blue = training, orange = unseen, dashed = noise', 46, 12);
      q('#out').innerHTML = `<span class="m">degree</span> ${D}   <span class="m">n</span> ${n}\n<span class="b">training error</span> ${AI.fmt(T.tr, 3)}   <span class="o">error on new data</span> ${AI.fmt(T.te, 3)}\n<span class="m">gap</span> ${AI.fmt(T.te - T.tr, 3)}   <span class="m">noise floor</span> ${AI.fmt(SIG * SIG, 2)}`;
    }, { fps: 30 });
  };

  /* =============== Lab: end-to-end pipeline flow with shapes, sizes and costs (illustrative estimates) =============== */
  AI.labs.mlflow = function (root) {
    const { view, ctl } = AI.labShell(root, 'A dataset flows through the pipeline', 'edit the settings · shapes, sizes and costs update at every stage · estimates are illustrative',
      `<canvas id="cv" style="height:330px;cursor:pointer"></canvas><div class="readout" id="det" style="margin-top:8px;font-size:12.5px;line-height:1.55;white-space:normal"></div>`,
      `<label>Rows N (log) <span class="v" id="nv"></span><input type="range" id="N" min="3" max="8" step="0.1" value="4"></label>
       <label>Numeric columns <span class="v" id="cv2"></span><input type="range" id="nn" min="1" max="200" step="1" value="8"></label>
       <label>Category sizes (comma list) <input type="text" id="cards" value="4,6" spellcheck="false"></label>
       <label>Model <select id="mod"><option value="lin">linear model</option><option value="forest">forest (100 trees, depth 8)</option><option value="mlps" selected>MLP d→32→32→1</option><option value="mlpb">MLP d→512→512→512→1</option></select></label>
       <label>Training epochs <span class="v" id="ev"></span><input type="range" id="ep" min="1" max="100" step="1" value="10"></label>
       <label>Hardware <select id="hw"><option value="cpu">CPU (≈ 50 GFLOP/s)</option><option value="gpu">GPU (≈ 3 TFLOP/s usable)</option></select></label>
       <label>Requests / day (log) <span class="v" id="rv"></span><input type="range" id="rq" min="3" max="9" step="0.1" value="6"></label>
       <label>Bytes per logged request <span class="v" id="bv"></span><input type="range" id="by" min="50" max="2000" step="50" value="200"></label>
       <label>Log retention (days) <span class="v" id="dv"></span><input type="range" id="rt" min="1" max="730" step="1" value="365"></label>
       <label>Decision threshold <span class="v" id="tv"></span><input type="range" id="th" min="0.05" max="0.95" step="0.01" value="0.5"></label>` + fx.PLAYCTL);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv'));
    const NAMES = ['Data', 'Preprocessing', 'Training', 'Inference', 'Post-processing', 'Storage & monitoring'], COLS = ['blue', 'teal', 'purple', 'orange', 'pink', 'green'];
    const MODE = ['offline · batch', 'offline · batch', 'offline · heavy', 'online · per request', 'online · per request', 'continuous'];
    const fmtN = (n) => (n >= 1e9 ? AI.fmt(n / 1e9, 3) + 'B' : n >= 1e6 ? AI.fmt(n / 1e6, 3) + 'M' : n >= 1e3 ? AI.fmt(n / 1e3, 3) + 'k' : AI.fmt(n, 3));
    const fmtB = (b) => { const u = ['B', 'kB', 'MB', 'GB', 'TB', 'PB']; let i = 0; while (b >= 1000 && i < 5) { b /= 1000; i++; } return AI.fmt(b, 3) + ' ' + u[i]; };
    const fmtT = (s) => (s < 1e-3 ? AI.fmt(s * 1e6, 3) + ' µs' : s < 1 ? AI.fmt(s * 1e3, 3) + ' ms' : s < 120 ? AI.fmt(s, 3) + ' s' : s < 7200 ? AI.fmt(s / 60, 3) + ' min' : s < 172800 ? AI.fmt(s / 3600, 3) + ' h' : AI.fmt(s / 86400, 3) + ' days');
    const survive = (t) => { let s = 0; const K = 400; for (let i = 0; i < K; i++) { const x = t + ((1 - t) * (i + 0.5)) / K; s += 30 * x * Math.pow(1 - x, 4) * ((1 - t) / K); } return s; }; // assumes scores ~ Beta(2,5)
    let M = {}, stage = 0, phase = 'work', t = 0, pinned = -1, pw = 16, key = '';
    const compute = () => {
      const N = Math.round(Math.pow(10, +q('#N').value)), nn = +q('#nn').value, cards = q('#cards').value.split(',').map((s) => parseInt(s, 10)).filter((v) => v > 0 && v < 1e6), d = nn + cards.reduce((a, b) => a + b, 0);
      const mod = q('#mod').value, ep = mod === 'forest' ? 1 : +q('#ep').value, dev = q('#hw').value === 'gpu' && mod !== 'forest' ? 3e12 : 5e10, req = Math.pow(10, +q('#rq').value), by = +q('#by').value, days = +q('#rt').value, th = +q('#th').value;
      let P, fl, flReq, label;
      if (mod === 'lin') { P = d + 1; fl = 6 * P * N; flReq = 2 * P; label = 'linear'; }
      else if (mod === 'forest') { const nodes = Math.min(511, 2 * N); P = 100 * nodes * 3; fl = 100 * N * d * 8; flReq = 100 * 8 * 2; label = 'forest'; }
      else if (mod === 'mlps') { P = d * 32 + 32 + 32 * 32 + 32 + 32 + 1; fl = 6 * P * N; flReq = 2 * P; label = 'MLP-small'; }
      else { P = d * 512 + 512 + 2 * (512 * 512 + 512) + 512 + 1; fl = 6 * P * N; flReq = 2 * P; label = 'MLP-big'; }
      const rawB = N * (nn * 8 + cards.length * 10), XB = N * d * 4, trainFl = fl * ep, trainT = trainFl / dev, lat = flReq / 5e10 + 0.002, qps = req / 86400, cores = Math.max(1, Math.ceil(qps * 3 * (flReq / 5e10))), flag = survive(th), logDay = req * by;
      M = { N, nn, nc: cards.length, d, mod, label, ep, P, fl, trainFl, trainT, flReq, lat, qps, cores, rawB, XB, req, by, days, th, flag, logDay, logTot: logDay * days, ckpt: P * 4, mem: XB + P * 16, dev };
    };
    const sizes = () => [M.rawB, M.XB, M.ckpt, M.d * 4, 8, M.logTot];
    const lines = () => [
      [`${fmtN(M.N)} × ${M.nn + M.nc}`, fmtB(M.rawB), 'collect + label'],
      [`${fmtN(M.N)} × ${M.d}`, fmtB(M.XB), fmtT(M.N * M.d * 1e-8) + ' CPU'],
      [`${fmtN(M.P)} params`, fmtB(M.ckpt) + ' file', fmtT(M.trainT)],
      [`1 × ${M.d} → 1`, fmtN(M.flReq) + ' FLOP/req', fmtT(M.lat) + ' latency'],
      [`score → decision`, `flag ${AI.fmt(M.flag * 100, 3)}%`, '≈ free'],
      [`${fmtN(M.req)} req/day`, fmtB(M.logTot), fmtB(M.logDay) + '/day'],
    ];
    const details = () => [
      `<b>Data collection.</b> Raw table: N = ${M.N.toLocaleString()} rows × ${M.nn + M.nc} columns (${M.nn} numeric at 8 bytes, ${M.nc} categorical stored as ≈10-byte strings) ≈ <b>${fmtB(M.rawB)}</b>. Cost is dominated by I/O, labelling and cleaning, not arithmetic.`,
      `<b>Preprocessing.</b> One-hot encoding turns ${M.nc} categorical columns into ${M.d - M.nn} columns, so d = ${M.nn} + ${M.d - M.nn} = <b>${M.d}</b>. X (float32) = N·d·4 = ${M.N.toLocaleString()}·${M.d}·4 ≈ <b>${fmtB(M.XB)}</b>. Time ≈ N·d·10 ns = ${fmtT(M.N * M.d * 1e-8)} on one CPU. Fit scalers on the training split only.`,
      `<b>Training</b> (${M.label}). Parameters P = <b>${M.P.toLocaleString()}</b>, checkpoint ≈ 4P = ${fmtB(M.ckpt)}. Work ≈ ${M.mod === 'forest' ? '100·N·d·depth' : '6·P·N per epoch'} = ${fmtN(M.fl)} FLOP${M.mod === 'forest' ? '' : ' × ' + M.ep + ' epochs = ' + fmtN(M.trainFl) + ' FLOP'} at ${M.dev > 1e12 ? '≈3 TFLOP/s' : '≈50 GFLOP/s'} → <b>${fmtT(M.trainT)}</b>. Training memory ≈ X + weights, gradients and optimiser state ≈ ${fmtB(M.mem)}.`,
      `<b>Inference.</b> One request: ≈ 2P = ${fmtN(M.flReq)} FLOP (forest: comparisons) → compute ${fmtT(M.flReq / 5e10)}, plus ≈ 2 ms network/overhead = <b>${fmtT(M.lat)}</b>. At ${fmtN(M.req)} requests/day ≈ ${AI.fmt(M.qps, 3)} req/s on average (peak ×3) → ≈ <b>${M.cores}</b> CPU core(s). The model must sit in memory: ${fmtB(M.ckpt)}.`,
      `<b>Post-processing.</b> Scores become decisions. Assuming illustrative scores ~ Beta(2, 5), a threshold of ${M.th} flags <b>${AI.fmt(M.flag * 100, 3)}%</b> of requests ≈ ${fmtN(M.req * M.flag)} per day. Moving the threshold trades false alarms against misses (see the threshold lab in Module 5). Compute is negligible.`,
      `<b>Storage &amp; monitoring.</b> Logging ${M.by} bytes per request: ${fmtN(M.req)} × ${M.by} B = ${fmtB(M.logDay)}/day; kept ${M.days} days = <b>${fmtB(M.logTot)}</b> — often far larger than the model. Drift checks and new labels flow back to stage 1: the loop.`,
    ];
    const reset = () => { stage = 0; phase = 'work'; t = 0; };
    const st = fx.player(root, { step: () => { stage = (stage + 1) % 6; phase = 'work'; t = 0; }, reset });
    const boxes = () => { const bw = (c.w - 16) / 6; return NAMES.map((_, i) => ({ x: 8 + i * bw + 3, w: bw - 6, y: 92, h: 150 })); };
    q('#cv').onclick = (e) => { const r = q('#cv').getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top; const b = boxes().findIndex((b) => px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h); pinned = b === pinned ? -1 : b; };
    const WORK = 2.0, MOVE = 1.0, CAPS = ['① raw examples arrive from the sources.', '② clean, encode, scale: raw table → numeric matrix X.', '③ search for parameters θ: all that data is squeezed into a small model file.', '④ the frozen model answers one request at a time.', '⑤ scores become decisions (threshold, rank, explain).', '⑥ log everything, monitor drift; fresh labelled data loops back to ①.'];
    let wS = 20;
    AI.loop(root, (dt) => {
      [['#nv', 'N', (v) => fmtN(Math.pow(10, v))], ['#rv', 'rq', (v) => fmtN(Math.pow(10, v))]].forEach(([id, k, f]) => (q(id).textContent = f(+q('#' + k).value)));
      q('#cv2').textContent = q('#nn').value; q('#ev').textContent = q('#ep').value; q('#bv').textContent = q('#by').value; q('#dv').textContent = q('#rt').value; q('#tv').textContent = q('#th').value;
      const k = ['N', 'nn', 'cards', 'mod', 'ep', 'hw', 'rq', 'by', 'rt', 'th'].map((i) => q('#' + i).value).join('|');
      if (k !== key) { key = k; compute(); }
      if (st.playing) { t += dt * st.speed; if (phase === 'work' && t >= WORK) { phase = 'move'; t = 0; } else if (phase === 'move' && t >= MOVE) { stage = (stage + 1) % 6; phase = 'work'; t = 0; } }
      const C = AI.col(), g = c.ctx, B = boxes(), S = sizes(), L = lines();
      g.clearRect(0, 0, c.w, c.h);
      const fs = AI.clamp(B[0].w / 9.5, 8, 11.5);
      B.forEach((b, i) => {
        const col = C[COLS[i]], act = i === stage && phase === 'work', sel = i === pinned;
        g.fillStyle = C.bg2; g.strokeStyle = act || sel ? col : C.border; g.lineWidth = act || sel ? 3 : 1.2; g.beginPath(); g.roundRect(b.x, b.y, b.w, b.h, 8); g.fill(); g.stroke();
        g.fillStyle = col; g.globalAlpha = act ? 0.22 : 0.1; g.beginPath(); g.roundRect(b.x, b.y, b.w, 26, [8, 8, 0, 0]); g.fill(); g.globalAlpha = 1;
        g.fillStyle = C.fg; g.textAlign = 'center'; g.font = `bold ${fs}px sans-serif`; g.fillText((i + 1) + '. ' + NAMES[i].split(' ')[0], b.x + b.w / 2, b.y + 17);
        g.font = `${fs}px sans-serif`; const cap = ['shape', 'size', 'cost'];
        L[i].forEach((s, j) => { g.fillStyle = C.muted; g.fillText(cap[j], b.x + b.w / 2, b.y + 44 + j * 34); g.fillStyle = C.fg; g.fillText(s, b.x + b.w / 2, b.y + 58 + j * 34, b.w - 6); });
        if (act) { g.fillStyle = C.border; g.fillRect(b.x + 6, b.y + b.h - 10, b.w - 12, 4); g.fillStyle = col; g.fillRect(b.x + 6, b.y + b.h - 10, (b.w - 12) * AI.smooth(t / WORK), 4); }
        g.fillStyle = C.muted; g.font = `${Math.max(8, fs - 1)}px sans-serif`; g.fillText(MODE[i], b.x + b.w / 2, b.y + b.h + 18, b.w);
      });
      // feedback loop
      g.strokeStyle = C.green; g.globalAlpha = phase === 'move' && stage === 5 ? 1 : 0.4; g.lineWidth = 2; g.setLineDash([6, 5]); g.beginPath();
      const a = B[5], z = B[0], yy = 92 + 150 + 38; g.moveTo(a.x + a.w / 2, a.y + a.h + 24); g.lineTo(a.x + a.w / 2, yy + 14); g.lineTo(z.x + z.w / 2, yy + 14); g.lineTo(z.x + z.w / 2, z.y + z.h + 24); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText('feedback: new labelled data returns to stage 1', c.w / 2, yy + 28);
      // packet
      const cx = (i) => B[i].x + B[i].w / 2, target = AI.clamp(10 + 5 * (Math.log10(Math.max(S[stage], 10)) - 1), 10, 74);
      let px = cx(stage), label = stage === 2 ? 'model θ' : 'data'; const kk = 1 - Math.exp(-dt * 4); wS += (target - wS) * kk;
      if (phase === 'move') { const nx = (stage + 1) % 6, e = AI.smooth(t / MOVE); if (stage === 5) { px = cx(5) + (cx(0) - cx(5)) * e; } else px = cx(stage) + (cx(nx) - cx(stage)) * e; }
      const pcol = stage === 2 && phase === 'work' ? C.purple : C[COLS[stage]];
      g.fillStyle = pcol; g.globalAlpha = 0.9; g.beginPath(); g.roundRect(px - wS / 2, 44, wS, 26, 6); g.fill(); g.globalAlpha = 1; g.strokeStyle = C.fg; g.lineWidth = 1; g.stroke();
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText(label + ' · ' + fmtB(S[stage]), Math.min(Math.max(px, 70), c.w - 70), 38);
      g.textAlign = 'left'; g.fillStyle = C.fg; g.font = '12px sans-serif'; g.fillText(CAPS[stage], 10, 18);
      const sh = pinned >= 0 ? pinned : stage; const html = details()[sh] + (pinned >= 0 ? '' : ' <span style="color:var(--muted)">(click a stage to pin its details)</span>');
      if (q('#det').dataset.k !== sh + '|' + key) { q('#det').dataset.k = sh + '|' + key; q('#det').innerHTML = html; }
    }, { fps: 30 });
  };
})();
