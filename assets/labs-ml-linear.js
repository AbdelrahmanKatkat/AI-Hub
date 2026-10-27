/* Labs for ML Module 2 · Linear models.
 *   linGeom      draggable points, gradient descent vs closed form, residual squares, R², loss surface
 *   regPoly      polynomial degree + λ, ridge / lasso / elastic net, coefficient bars, error curve, coefficient path
 *   logOdds      probability <-> odds <-> log-odds converter, sigmoid, odds-ratio meaning of a coefficient
 *   logisticLab  2-D classes, boundary rotating while logistic regression trains, loss curve, learning rate, L2
 *   softmaxLab   3 classes, regions forming, p − y for a chosen example
 *   optLab       loss contours, descent path with/without feature scaling, learning-rate regimes
 * Needs lib.js, mllib.js. All animation is dt-based; every lab has speed, pause, step, reset. Simulated numbers are computed live.
 */
(function () {
  const PH = 1.2; // seconds per phase in slow-motion mode (at speed x1)
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const symlog = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v) / 0.05);
  const R = (root) => (s) => AI.$(s, root);

  /** Gaussian elimination with partial pivoting: solves A x = b (A is n×n array of rows). */
  function gauss(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let i = 0; i < n; i++) {
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
      [M[i], M[p]] = [M[p], M[i]];
      const d = M[i][i] || 1e-300;
      for (let r = i + 1; r < n; r++) { const f = M[r][i] / d; if (f) for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c]; x[i] = s / (M[i][i] || 1e-300); }
    return x;
  }

  /** Caption bar drawn on a canvas, with an optional progress strip (frac 0..1). */
  function caption(g, C, x, y, wMax, text, frac) {
    g.font = '12px sans-serif'; g.textAlign = 'left';
    const bw = Math.min(wMax, g.measureText(text).width + 16);
    g.fillStyle = C.bg2; g.globalAlpha = 0.92; g.fillRect(x, y, bw, 24); g.globalAlpha = 1;
    g.fillStyle = C.fg; g.fillText(text, x + 8, y + 16);
    if (frac !== undefined) { g.fillStyle = C.border; g.fillRect(x, y + 25, bw, 3); g.fillStyle = C.blue; g.fillRect(x, y + 25, bw * frac, 3); }
  }

  /** Training clock shared by the step-wise labs: slow-motion 4-phase mode or continuous steps/second. */
  function clock(q, stepOnce, o = {}) {
    const tick = AI.ticker(), c = { playing: o.playing !== false, phase: 0, phT: 0 };
    const setBtn = () => (q('#play').textContent = c.playing ? '⏸ Pause' : '▶ Play');
    c.update = (dt) => {
      const sp = +q('#sp').value, slow = q('#slow') && q('#slow').checked;
      q('#spv').textContent = slow ? 'speed ×' + AI.fmt(sp / 4, 2) : sp + ' steps / s';
      if (!c.playing) return;
      if (slow) { c.phT += (dt * sp) / 4; if (c.phT >= PH) { c.phT = 0; c.phase = (c.phase + 1) % 4; if (c.phase === 3) stepOnce(); } }
      else { const n = tick(dt, sp); for (let i = 0; i < n; i++) stepOnce(); }
    };
    c.slow = () => q('#slow') && q('#slow').checked;
    c.reset = () => { c.phase = 0; c.phT = 0; };
    q('#play').onclick = () => { c.playing = !c.playing; setBtn(); };
    q('#step').onclick = () => { c.playing = false; setBtn(); stepOnce(); };
    setBtn();
    return c;
  }

  const slider = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const transport = (extra = '') => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button>${extra}</div>`;
  const speedSlider = (val, slowChecked) => `${slider('sp', 'Speed', 1, 60, 1, val)}<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="slow" ${slowChecked ? 'checked' : ''}> Slow motion: show each phase (predict → measure → gradient → update)</label>`;
  const SLOWCAPS = ['① Predict: run the current model on every example.', '② Measure: compare predictions with the truth (the loss).', '③ Gradient: which way does each knob lower the loss?', '④ Update: move every knob a small step downhill.'];

  /* ===================================================================================================================
   * 1. Regression geometry
   * ================================================================================================================ */
  AI.labs.linGeom = function (root) {
    const { ctl } = AI.labShell(root, 'Regression geometry: gradient descent walks to the closed form', 'drag points · click empty space to add · double-click a point to remove · click the loss map to restart from there',
      `<canvas id="fit" style="height:300px"></canvas><canvas id="surf" style="height:230px"></canvas>`,
      `${slider('lr', 'Learning rate η', -3, -0.8, 0.02, -1.5)}${speedSlider(6, true)}
       ${transport()}
       <div class="chips" id="sets"></div>
       <div class="readout" id="out"></div>`);
    const q = R(root), c1 = AI.canvas(q('#fit')), c2 = AI.canvas(q('#surf'));
    const WORKED = [[1, 2], [2, 3], [3, 5], [4, 4], [5, 6]];
    let pts = WORKED.map((p) => [...p]), w = -0.3, b = 3.5, wS = w, bS = b, hist = [], path = [[w, b]], steps = 0, lastG = [0, 0], drag = -1, dirty = true, surfDirty = true, seed = 5, diverged = false, key = '';
    const off = document.createElement('canvas'); off.width = off.height = 56;
    const closed = () => {
      const n = pts.length, mx = pts.reduce((a, p) => a + p[0], 0) / n, my = pts.reduce((a, p) => a + p[1], 0) / n;
      let sxy = 0, sxx = 0; pts.forEach((p) => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
      const ww = sxx > 1e-9 ? sxy / sxx : 0; return { w: ww, b: my - ww * mx };
    };
    let cf = closed();
    const mse = (ww, bb) => pts.reduce((a, p) => a + (ww * p[0] + bb - p[1]) ** 2, 0) / pts.length;
    const sst = () => { const my = pts.reduce((a, p) => a + p[1], 0) / pts.length; return pts.reduce((a, p) => a + (p[1] - my) ** 2, 0); };
    const r2 = (ww, bb) => { const s = sst(); return s > 1e-12 ? 1 - (mse(ww, bb) * pts.length) / s : NaN; };
    const lr = () => Math.pow(10, +q('#lr').value);
    const stepOnce = () => {
      if (diverged) return;
      let gw = 0, gb = 0; const n = pts.length;
      pts.forEach((p) => { const e = w * p[0] + b - p[1]; gw += (2 / n) * e * p[0]; gb += (2 / n) * e; });
      w -= lr() * gw; b -= lr() * gb; lastG = [gw, gb]; steps++;
      const L = mse(w, b); hist.push(L); path.push([w, b]); if (path.length > 3000) path.shift();
      if (!Number.isFinite(L) || L > 1e9) diverged = true;
      dirty = true;
    };
    const restart = (ww, bb) => { w = ww; b = bb; hist = []; path = [[w, b]]; steps = 0; diverged = false; ck.reset(); dirty = true; };
    const ck = clock(q, stepOnce);
    q('#reset').onclick = () => restart(-0.3, cf.b > 2 ? cf.b - 2.5 : 3.5 - 0 * cf.b);
    const setData = (arr) => { pts = arr; cf = closed(); surfDirty = true; dirty = true; hist = []; path = [[w, b]]; steps = 0; diverged = false; };
    const sets = q('#sets');
    [['worked example', () => setData(WORKED.map((p) => [...p]))],
     ['random', () => { const r = AI.rng((seed += 7)); const s = (r() - 0.5) * 2, i = 1 + r() * 3; setData(Array.from({ length: 8 }, (_, k) => { const x = 0.5 + (k * 5.5) / 7 + (r() - 0.5) * 0.4; return [x, i + s * x * 0.8 + 0.6 * AI.randn(r)]; })); }],
     ['add an outlier', () => setData([...pts, [5.5, 0.5]])]].forEach(([t, f]) => { const bt = document.createElement('button'); bt.className = 'chip'; bt.textContent = t; bt.onclick = f; sets.appendChild(bt); });

    // --- pointer interaction on the fit canvas
    const fv = () => AI.view(c1, -0.5, 7, -1, 8.5);
    const pos = (e, cv) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const near = (px, py) => { const v = fv(); let best = -1, bd = 14; pts.forEach((p, i) => { const d = Math.hypot(v.X(p[0]) - px, v.Y(p[1]) - py); if (d < bd) { bd = d; best = i; } }); return best; };
    q('#fit').addEventListener('pointerdown', (e) => {
      const [px, py] = pos(e, q('#fit')), v = fv(), i = near(px, py);
      if (i >= 0) { drag = i; q('#fit').setPointerCapture(e.pointerId); }
      else if (px > v.pad.l && px < v.pad.l + v.pw && py > v.pad.t && py < v.pad.t + v.ph) { pts.push([v.ix(px), v.iy(py)]); drag = pts.length - 1; q('#fit').setPointerCapture(e.pointerId); setData(pts); }
    });
    q('#fit').addEventListener('pointermove', (e) => { if (drag < 0) return; const [px, py] = pos(e, q('#fit')), v = fv(); pts[drag] = [AI.clamp(v.ix(px), v.xmin, v.xmax), AI.clamp(v.iy(py), v.ymin, v.ymax)]; cf = closed(); surfDirty = true; dirty = true; });
    q('#fit').addEventListener('pointerup', () => (drag = -1));
    q('#fit').addEventListener('dblclick', (e) => { const [px, py] = pos(e, q('#fit')), i = near(px, py); if (i >= 0 && pts.length > 2) { pts.splice(i, 1); setData(pts); } });
    const sv = () => AI.view(c2, cf.w - 2, cf.w + 2, cf.b - 4.5, cf.b + 4.5, { l: 40, r: 10, t: 12, b: 28 });
    q('#surf').addEventListener('pointerdown', (e) => { const [px, py] = pos(e, q('#surf')), v = sv(); restart(v.ix(px), v.iy(py)); });
    c1.onresize = c2.onresize = () => { dirty = true; surfDirty = true; };

    AI.loop(root, (dt) => {
      ck.update(dt);
      q('#lrv').textContent = AI.fmt(lr(), 4);
      const k = 1 - Math.exp(-dt * (ck.slow() ? 3 : 14)), pw = wS, pb = bS;
      wS += (w - wS) * k; bS += (b - bS) * k;
      const moving = Math.abs(wS - pw) + Math.abs(bS - pb) > 1e-5;
      const nk = [wS.toFixed(4), bS.toFixed(4), pts.length, ck.phase, ck.playing, ck.slow(), steps, c1.w, drag].join('|');
      if (!dirty && !moving && nk === key) return; key = nk; dirty = false;
      const C = AI.col();
      // ---- fit canvas
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = fv(); AI.axes(g, v);
      const showSq = ck.slow() ? ck.phase === 1 || ck.phase === 2 : true;
      pts.forEach((p) => {
        const yh = AI.clamp(wS * p[0] + bS, -30, 30), y0 = v.Y(p[1]), y1 = v.Y(yh), side = Math.abs(y1 - y0), top = Math.min(y0, y1);
        g.strokeStyle = C.red; g.globalAlpha = showSq ? 0.9 : 0.3; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(p[0]), y0); g.lineTo(v.X(p[0]), y1); g.stroke();
        if (showSq) { g.fillStyle = C.red; g.globalAlpha = 0.16; g.fillRect(v.X(p[0]), top, side, side); g.globalAlpha = 0.7; g.strokeRect(v.X(p[0]), top, side, side); }
        g.globalAlpha = 1;
      });
      g.setLineDash([6, 5]); g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(v.xmin), v.Y(cf.w * v.xmin + cf.b)); g.lineTo(v.X(v.xmax), v.Y(cf.w * v.xmax + cf.b)); g.stroke(); g.setLineDash([]);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.strokeStyle = C.orange; g.lineWidth = 2.6; g.beginPath(); g.moveTo(v.X(v.xmin), v.Y(wS * v.xmin + bS)); g.lineTo(v.X(v.xmax), v.Y(wS * v.xmax + bS)); g.stroke(); g.restore();
      g.fillStyle = C.blue; pts.forEach((p) => { g.beginPath(); g.arc(v.X(p[0]), v.Y(p[1]), 5, 0, 7); g.fill(); });
      g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText('● data   — gradient-descent line   - - closed-form line   ▢ squared errors', v.pad.l + 6, v.pad.t + v.ph - 6);
      if (ck.slow() && ck.playing && !diverged) caption(g, C, v.pad.l + 4, v.pad.t + 4, v.pw - 8, SLOWCAPS[ck.phase], (ck.phase + ck.phT / PH) / 4);
      else if (!diverged) caption(g, C, v.pad.l + 4, v.pad.t + 4, v.pw - 8, ck.playing ? 'Gradient descent: every step moves (w, b) downhill on the loss map below.' : 'Paused. Step, drag points, or press Play.');
      else caption(g, C, v.pad.l + 4, v.pad.t + 4, v.pw - 8, 'Diverged: η too large. Lower it and Reset.');
      // ---- loss surface
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const sv2 = sv(); const px0 = sv2.pad.l, py0 = sv2.pad.t;
      if (surfDirty) {
        const og = off.getContext('2d'), im = og.createImageData(56, 56), bg = AI.rgb(C.bg), bl = AI.rgb(C.blue), vals = [], Lmin = mse(cf.w, cf.b);
        for (let j = 0; j < 56; j++) for (let i = 0; i < 56; i++) { const ww = sv2.xmin + ((i + 0.5) / 56) * (sv2.xmax - sv2.xmin), bb = sv2.ymax - ((j + 0.5) / 56) * (sv2.ymax - sv2.ymin); vals.push(Math.log10(mse(ww, bb) - Lmin + 0.02)); }
        const lo = Math.min(...vals), hi = Math.max(...vals);
        vals.forEach((u, idx) => { const t = Math.floor(((u - lo) / (hi - lo + 1e-9)) * 12) / 12, s = 0.75 - 0.65 * t; for (let c = 0; c < 3; c++) im.data[idx * 4 + c] = bg[c] + (bl[c] - bg[c]) * s; im.data[idx * 4 + 3] = 255; });
        og.putImageData(im, 0, 0); surfDirty = false;
      }
      g.imageSmoothingEnabled = false; g.drawImage(off, px0, py0, sv2.pw, sv2.ph); g.imageSmoothingEnabled = true;
      g.strokeStyle = C.border; g.strokeRect(px0 + 0.5, py0 + 0.5, sv2.pw, sv2.ph);
      g.save(); g.beginPath(); g.rect(px0, py0, sv2.pw, sv2.ph); g.clip();
      g.strokeStyle = C.orange; g.lineWidth = 1.8; g.beginPath(); path.forEach((p, i) => (i ? g.lineTo(sv2.X(p[0]), sv2.Y(p[1])) : g.moveTo(sv2.X(p[0]), sv2.Y(p[1])))); g.stroke();
      g.fillStyle = C.green; g.beginPath(); g.arc(sv2.X(cf.w), sv2.Y(cf.b), 6, 0, 7); g.fill();
      g.fillStyle = C.orange; g.strokeStyle = C.fg; g.beginPath(); g.arc(sv2.X(wS), sv2.Y(bS), 5, 0, 7); g.fill(); g.stroke(); g.restore();
      g.fillStyle = C.muted; g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
      g.fillText('w (slope)', px0 + sv2.pw / 2, c2.h - 4); g.textAlign = 'right';
      [sv2.ymin + 0.5, cf.b, sv2.ymax - 0.5].forEach((yy) => g.fillText(AI.fmt(yy, 1), px0 - 4, sv2.Y(yy) + 4)); g.textAlign = 'center';
      [sv2.xmin + 0.3, cf.w, sv2.xmax - 0.3].forEach((xx) => g.fillText(AI.fmt(xx, 1), sv2.X(xx), py0 + sv2.ph + 13));
      g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText('loss map L(w, b): darker band = higher loss · ● green = closed form', px0 + 6, py0 + 12);
      // ---- readout
      const L = mse(w, b), Lc = mse(cf.w, cf.b);
      q('#out').innerHTML = `<span class="m">step</span> ${steps}   <span class="m">η</span> ${AI.fmt(lr(), 4)}\n<span class="o">GD     </span> w=${AI.fmt(w, 4)}  b=${AI.fmt(b, 4)}  MSE=${AI.fmt(L, 4)}  R²=${AI.fmt(r2(w, b), 4)}\n<span class="g">closed </span> w=${AI.fmt(cf.w, 4)}  b=${AI.fmt(cf.b, 4)}  MSE=${AI.fmt(Lc, 4)}  R²=${AI.fmt(r2(cf.w, cf.b), 4)}\n<span class="m">∂L/∂w, ∂L/∂b</span> = ${AI.fmt(lastG[0], 3)}, ${AI.fmt(lastG[1], 3)}${diverged ? '\n<span class="r">Diverged. Lower η, then Reset.</span>' : ''}`;
    }, { fps: 30 });
  };

  /* ===================================================================================================================
   * 2. Polynomial degree + regularisation (ridge / lasso / elastic net)
   * ================================================================================================================ */
  AI.labs.regPoly = function (root) {
    AI.labShell(root, 'Overfitting and regularisation: degree, λ, ridge vs lasso', 'pick a degree, then raise λ (or press “Sweep λ down”) and watch the curve calm down and the coefficients shrink',
      `<canvas id="fit" style="height:250px"></canvas><canvas id="bars" style="height:110px"></canvas><canvas id="err" style="height:140px"></canvas><canvas id="path" style="height:140px"></canvas>`,
      `${slider('deg', 'Polynomial degree d', 1, 12, 1, 9)}
       ${slider('lam', 'log₁₀ λ (left end = no penalty)', -6, 1, 0.05, -2)}
       <label>Penalty <select id="pen"><option value="ridge">Ridge (L2)</option><option value="lasso">Lasso (L1)</option><option value="enet">Elastic net (½ L1 + ½ L2)</option></select></label>
       ${slider('n', 'Training points n', 8, 40, 1, 14)}${slider('nz', 'Noise σ', 0.05, 0.5, 0.01, 0.2)}
       ${slider('sp', 'Sweep speed', 0.5, 3, 0.5, 1)}
       <div class="row"><button class="btn primary" id="sweep">▶ Sweep λ down</button><button class="btn" id="stop">⏸ Pause</button><button class="btn" id="reset">Reset</button><button class="btn" id="new">New sample</button></div>
       <div class="readout" id="out"></div>`);
    const q = R(root), cF = AI.canvas(q('#fit')), cB = AI.canvas(q('#bars')), cE = AI.canvas(q('#err')), cP = AI.canvas(q('#path'));
    const truth = (x) => 0.8 * Math.sin(2.6 * x) + 0.3 * x;
    const NG = 36, S0 = -6, S1 = 1;
    let seed = 4, trX = [], trY = [], vaX = [], vaY = [], d = 9, mu = [], sd = [], my = 0, Cm = null, cv = null, grid = [], wT = [], wS = [], sweeping = false, key = '', dirty = true, s = -2;
    const feats = (x) => { const o = []; let p = 1; for (let j = 1; j <= d; j++) { p *= x; o.push(p); } return o; };
    const gen = () => {
      const r = AI.rng(seed), n = +q('#n').value, nz = +q('#nz').value;
      trX = Array.from({ length: n }, () => -1 + 2 * r()); trY = trX.map((x) => truth(x) + nz * AI.randn(r));
      const r2 = AI.rng(1000 + seed); vaX = Array.from({ length: 80 }, (_, i) => -1 + (2 * i) / 79); vaY = vaX.map((x) => truth(x) + nz * AI.randn(r2));
    };
    const prep = () => {
      d = +q('#deg').value; const n = trX.length, F = trX.map(feats);
      mu = []; sd = []; for (let j = 0; j < d; j++) { const col = F.map((f) => f[j]), m = col.reduce((a, c) => a + c, 0) / n, v = Math.sqrt(col.reduce((a, c) => a + (c - m) ** 2, 0) / n) || 1; mu.push(m); sd.push(v); }
      my = trY.reduce((a, c) => a + c, 0) / n;
      const Z = F.map((f) => f.map((v, j) => (v - mu[j]) / sd[j]));
      Cm = Array.from({ length: d }, (_, j) => Array.from({ length: d }, (_, k) => Z.reduce((a, z) => a + z[j] * z[k], 0) / n));
      cv = Array.from({ length: d }, (_, j) => Z.reduce((a, z, i) => a + z[j] * (trY[i] - my), 0) / n);
    };
    const solve = (lam, w0) => {
      const pen = q('#pen').value, al = pen === 'ridge' ? 0 : pen === 'lasso' ? 1 : 0.5;
      if (al === 0) return gauss(Cm.map((r, j) => r.map((v, k) => v + (j === k ? Math.max(lam, 1e-10) : 0))), cv);
      const w = w0 ? [...w0] : new Array(d).fill(0);
      for (let it = 0; it < 400; it++) {
        let ch = 0;
        for (let j = 0; j < d; j++) {
          let rho = cv[j]; for (let k = 0; k < d; k++) rho -= Cm[j][k] * w[k]; rho += w[j];
          const th = lam * al, nw = (Math.sign(rho) * Math.max(Math.abs(rho) - th, 0)) / (1 + lam * (1 - al)); ch = Math.max(ch, Math.abs(nw - w[j])); w[j] = nw;
        }
        if (ch < 1e-10) break;
      }
      return w;
    };
    const lamOf = (sv) => (sv <= S0 + 1e-9 ? 0 : Math.pow(10, sv));
    const pred = (w, x) => { const f = feats(x); let y = my; for (let j = 0; j < d; j++) y += (w[j] * (f[j] - mu[j])) / sd[j]; return y; };
    const err = (w, X, Y) => X.reduce((a, x, i) => a + (pred(w, x) - Y[i]) ** 2, 0) / X.length;
    const build = () => {
      prep(); grid = []; let w0 = null;
      for (let i = NG - 1; i >= 0; i--) { const sv = S0 + (i * (S1 - S0)) / (NG - 1), w = solve(lamOf(sv), w0); w0 = w; grid[i] = { s: sv, w, tr: err(w, trX, trY), va: err(w, vaX, vaY) }; }
      wT = solve(lamOf(s), null); if (!wS.length || wS.length !== d) wS = [...wT]; dirty = true;
    };
    const setS = (sv) => { s = AI.clamp(sv, S0, S1); q('#lam').value = s; wT = solve(lamOf(s), wT.length === d ? wT : null); dirty = true; };
    q('#lam').oninput = () => { sweeping = false; setS(+q('#lam').value); };
    q('#deg').oninput = () => { build(); }; q('#pen').onchange = () => build();
    q('#n').oninput = q('#nz').oninput = () => { gen(); build(); };
    q('#new').onclick = () => { seed += 13; gen(); build(); };
    q('#sweep').onclick = () => { s = S1; sweeping = true; setS(S1); q('#sweep').textContent = '↻ Restart sweep'; };
    q('#stop').onclick = () => { sweeping = !sweeping; q('#stop').textContent = sweeping ? '⏸ Pause' : '▶ Resume'; };
    q('#reset').onclick = () => { sweeping = false; q('#deg').value = 9; q('#pen').value = 'ridge'; q('#sweep').textContent = '▶ Sweep λ down'; gen(); build(); setS(-2); };
    gen(); build(); setS(-2);
    cF.onresize = cB.onresize = cE.onresize = cP.onresize = () => (dirty = true);
    const colsOf = (n) => { const C = AI.col(), base = [C.blue, C.orange, C.green, C.purple, C.pink, C.teal, C.yellow, C.red]; return Array.from({ length: n }, (_, i) => base[i % base.length]); };

    AI.loop(root, (dt) => {
      if (sweeping) { setS(s - dt * 0.3 * +q('#sp').value); if (s <= S0 + 1e-6) { sweeping = false; q('#stop').textContent = '▶ Resume'; } }
      let mv = 0; for (let j = 0; j < d; j++) { const t = wT[j] - wS[j]; wS[j] += t * (1 - Math.exp(-dt * 6)); mv += Math.abs(t); }
      const lam = lamOf(s); q('#degv').textContent = d; q('#lamv').textContent = lam === 0 ? 'λ = 0' : 'λ = ' + AI.fmt(lam, 4); q('#nv').textContent = trX.length; q('#nzv').textContent = AI.fmt(+q('#nz').value, 2); q('#spv').textContent = '×' + q('#sp').value;
      const k = [mv > 1e-4, s.toFixed(3), d, dirty, sweeping].join('|'); if (!dirty && mv < 1e-4 && k === key) return; key = k; dirty = false;
      const C = AI.col(), cols = colsOf(d);
      // fit
      let g = cF.ctx; g.clearRect(0, 0, cF.w, cF.h); let v = AI.view(cF, -1.1, 1.1, -2.2, 2.2); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.setLineDash([5, 5]); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = -1.1 + 2.2 * i / 100; i ? g.lineTo(v.X(x), v.Y(truth(x))) : g.moveTo(v.X(x), v.Y(truth(x))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.orange; g.lineWidth = 2.6; g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -1.1 + 2.2 * i / 160, y = AI.clamp(pred(wS, x), -9, 9); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke();
      g.fillStyle = C.green; g.globalAlpha = 0.35; vaX.forEach((x, i) => { if (i % 2) return; g.beginPath(); g.arc(v.X(x), v.Y(vaY[i]), 2, 0, 7); g.fill(); }); g.globalAlpha = 1;
      g.fillStyle = C.blue; trX.forEach((x, i) => { g.beginPath(); g.arc(v.X(x), v.Y(trY[i]), 4.5, 0, 7); g.fill(); }); g.restore();
      const gi = Math.round(((s - S0) / (S1 - S0)) * (NG - 1)), cur = grid[AI.clamp(gi, 0, NG - 1)], best = grid.reduce((a, c) => (c.va < a.va ? c : a), grid[0]);
      const msg = !cur ? '' : cur.va > 2.5 * best.va && lam < best.w.length * 0 + Math.pow(10, best.s) ? 'High variance: the curve chases the noise (train error tiny, validation error large).' : cur.va > 2.5 * best.va ? 'High bias: λ is so large the model cannot bend (underfits).' : 'Close to the best validation error: a good balance of bias and variance.';
      caption(g, C, v.pad.l + 4, v.pad.t + 4, v.pw - 8, msg);
      g.fillStyle = C.muted; g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillText('● train   ● validation   - - true curve   — fitted model', v.pad.l + 6, v.pad.t + v.ph - 6);
      // bars
      g = cB.ctx; g.clearRect(0, 0, cB.w, cB.h); const mxv = Math.max(1.2, ...wS.map((x) => Math.abs(symlog(x)))), vb = AI.view(cB, 0, d, -mxv, mxv, { l: 40, r: 12, t: 16, b: 18 });
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(vb.pad.l, vb.Y(0)); g.lineTo(vb.pad.l + vb.pw, vb.Y(0)); g.stroke();
      const bw = vb.pw / d; let nz = 0;
      wS.forEach((x, j) => { const yv = symlog(x), top = vb.Y(Math.max(yv, 0)), hh = Math.abs(vb.Y(yv) - vb.Y(0)); g.fillStyle = cols[j]; g.fillRect(vb.pad.l + j * bw + bw * 0.15, top, bw * 0.7, Math.max(hh, 0.5)); if (Math.abs(wT[j]) < 1e-9) { g.fillStyle = C.red; g.font = '10px sans-serif'; g.textAlign = 'center'; g.fillText('0', vb.pad.l + j * bw + bw / 2, vb.Y(0) - 3); } else nz++; g.fillStyle = C.muted; g.font = '10px sans-serif'; g.textAlign = 'center'; g.fillText('x' + (j + 1 > 1 ? (j + 1) : ''), vb.pad.l + j * bw + bw / 2, cB.h - 5); });
      g.textAlign = 'left'; g.fillStyle = C.fg; g.font = '11px sans-serif'; g.fillText('coefficients (standardised features, signed log scale) · red 0 = exactly zero', vb.pad.l + 4, 11);
      // error curve
      g = cE.ctx; g.clearRect(0, 0, cE.w, cE.h); const ve = AI.view(cE, S0, S1, -4, 1.5, { l: 40, r: 12, t: 16, b: 22 }), le = (e) => Math.log10(Math.max(e, 1e-4));
      AI.axes(g, ve); g.lineWidth = 2;
      [['tr', C.blue], ['va', C.orange]].forEach(([kk, cc]) => { g.strokeStyle = cc; g.beginPath(); grid.forEach((p, i) => { const y = AI.clamp(le(p[kk]), -4, 1.5); i ? g.lineTo(ve.X(p.s), ve.Y(y)) : g.moveTo(ve.X(p.s), ve.Y(y)); }); g.stroke(); });
      g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.moveTo(ve.X(s), ve.pad.t); g.lineTo(ve.X(s), ve.pad.t + ve.ph); g.stroke();
      g.fillStyle = C.green; g.beginPath(); g.arc(ve.X(best.s), ve.Y(AI.clamp(le(best.va), -4, 1.5)), 5, 0, 7); g.fill();
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('log₁₀ MSE vs log₁₀ λ:  train (blue) · validation (orange) · ● best λ', ve.pad.l + 4, 11);
      // path
      g = cP.ctx; g.clearRect(0, 0, cP.w, cP.h); const mp = Math.max(1.2, ...grid.map((p) => Math.max(...p.w.map((x) => Math.abs(symlog(x)))))), vp = AI.view(cP, S0, S1, -mp, mp, { l: 40, r: 12, t: 16, b: 22 });
      AI.axes(g, vp, { noY: true }); g.lineWidth = 1.6;
      for (let j = 0; j < d; j++) { g.strokeStyle = cols[j]; g.beginPath(); grid.forEach((p, i) => (i ? g.lineTo(vp.X(p.s), vp.Y(symlog(p.w[j]))) : g.moveTo(vp.X(p.s), vp.Y(symlog(p.w[j]))))); g.stroke(); }
      g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.moveTo(vp.X(s), vp.pad.t); g.lineTo(vp.X(s), vp.pad.t + vp.ph); g.stroke();
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('coefficient path: each line = one coefficient as λ grows → shrinks (ridge) or hits 0 (lasso)', vp.pad.l + 4, 11);
      q('#out').innerHTML = `<span class="m">degree</span> ${d}  <span class="m">non-zero coefficients</span> <span class="o">${nz} / ${d}</span>\n<span class="m">train MSE</span> ${AI.fmt(err(wT, trX, trY), 4)}   <span class="m">validation MSE</span> <span class="o">${AI.fmt(err(wT, vaX, vaY), 4)}</span>\n<span class="m">best λ on grid</span> ${AI.fmt(Math.pow(10, best.s), 3)} (val MSE ${AI.fmt(best.va, 4)})\n<span class="m">‖w‖₂</span> ${AI.fmt(Math.hypot(...wT), 4)}   <span class="m">noise floor σ²</span> ${AI.fmt((+q('#nz').value) ** 2, 3)}`;
    }, { fps: 30 });
  };

  /* ===================================================================================================================
   * 3. Log-odds converter
   * ================================================================================================================ */
  AI.labs.logOdds = function (root) {
    AI.labShell(root, 'Probability ↔ odds ↔ log-odds', 'move any slider: the other two follow. The sigmoid is the bridge from log-odds back to probability',
      `<canvas id="sg" style="height:230px"></canvas><canvas id="bar" style="height:110px"></canvas>`,
      `${slider('z', 'Log-odds z = ln(p/(1−p))', -6, 6, 0.01, 0)}${slider('p', 'Probability p', 0.002, 0.998, 0.002, 0.5)}
       ${slider('w', 'Coefficient w on a feature x', -2, 2, 0.05, 0.7)}${slider('dx', 'Raise x by', -3, 3, 0.1, 1)}
       ${slider('sp', 'Auto-sweep speed', 0.1, 2, 0.1, 0.5)}
       <div class="row"><button class="btn primary" id="play">▶ Sweep z</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div>`);
    const q = R(root), c1 = AI.canvas(q('#sg')), c2 = AI.canvas(q('#bar'));
    let z = 0, playing = false, dir = 1, key = '', dirty = true;
    const setZ = (v) => { z = AI.clamp(v, -6, 6); q('#z').value = z; q('#p').value = sig(z); dirty = true; };
    q('#z').oninput = () => { playing = false; q('#play').textContent = '▶ Sweep z'; setZ(+q('#z').value); };
    q('#p').oninput = () => { playing = false; q('#play').textContent = '▶ Sweep z'; const p = +q('#p').value; setZ(Math.log(p / (1 - p))); };
    q('#w').oninput = q('#dx').oninput = () => (dirty = true);
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Sweep z'; };
    q('#reset').onclick = () => { playing = false; q('#play').textContent = '▶ Sweep z'; setZ(0); q('#w').value = 0.7; q('#dx').value = 1; };
    c1.onresize = c2.onresize = () => (dirty = true);
    AI.loop(root, (dt) => {
      if (playing) { setZ(z + dir * dt * 1.5 * +q('#sp').value * 2); if (z >= 6) dir = -1; if (z <= -6) dir = 1; }
      const k = z.toFixed(3) + q('#w').value + q('#dx').value + c1.w; if (!dirty && k === key) return; key = k; dirty = false;
      const C = AI.col(), p = sig(z), odds = Math.exp(z), w = +q('#w').value, dx = +q('#dx').value;
      q('#zv').textContent = AI.fmt(z, 2); q('#pv').textContent = AI.fmt(p, 3); q('#wv').textContent = AI.fmt(w, 2); q('#dxv').textContent = AI.fmt(dx, 1); q('#spv').textContent = '×' + q('#sp').value;
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h); const v = AI.view(c1, -6, 6, -0.05, 1.05); AI.axes(g, v);
      g.strokeStyle = C.border; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(-6), v.Y(0.5)); g.lineTo(v.X(6), v.Y(0.5)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.purple; g.lineWidth = 2.6; g.beginPath(); for (let i = 0; i <= 200; i++) { const x = -6 + 12 * i / 200; i ? g.lineTo(v.X(x), v.Y(sig(x))) : g.moveTo(v.X(x), v.Y(sig(x))); } g.stroke();
      const z2 = AI.clamp(z + w * dx, -6, 6);
      g.setLineDash([3, 4]); g.strokeStyle = C.orange; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(z), v.Y(0)); g.lineTo(v.X(z), v.Y(p)); g.lineTo(v.X(-6), v.Y(p)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(z), v.Y(p), 6, 0, 7); g.fill();
      g.fillStyle = C.green; g.beginPath(); g.arc(v.X(z2), v.Y(sig(z2)), 5, 0, 7); g.fill();
      g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText('p = σ(z) = 1/(1+e^(−z))   ● you are here   ● after raising x by ' + AI.fmt(dx, 1), v.pad.l + 6, v.pad.t + 12);
      g.fillStyle = C.muted; g.fillText('z (log-odds)', v.pad.l + v.pw - 78, c1.h - 4);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const bw = c2.w - 60, x0 = 30;
      g.fillStyle = C.orange; g.fillRect(x0, 26, bw * p, 26); g.fillStyle = C.blue; g.fillRect(x0 + bw * p, 26, bw * (1 - p), 26);
      g.fillStyle = C.fg; g.font = '12px sans-serif'; g.textAlign = 'left'; g.fillText('P(yes) = ' + AI.fmt(p, 3), x0, 18); g.textAlign = 'right'; g.fillText('P(no) = ' + AI.fmt(1 - p, 3), x0 + bw, 18);
      g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText('odds = P(yes) : P(no) = ' + AI.fmt(odds, 3) + ' : 1  → "' + (odds >= 1 ? AI.fmt(odds, 2) + ' to 1 in favour' : '1 to ' + AI.fmt(1 / odds, 2) + ' against') + '"', x0, 76);
      const o2 = Math.exp(z2); q('#out').innerHTML = `<span class="m">p</span>        ${AI.fmt(p, 4)}\n<span class="m">odds</span>     ${AI.fmt(odds, 4)}\n<span class="m">log-odds</span> <span class="o">${AI.fmt(z, 4)}</span>\n<span class="m">raise x by ${AI.fmt(dx, 1)}:</span> log-odds ${AI.fmt(z, 2)} → <span class="g">${AI.fmt(z + w * dx, 2)}</span>\n  odds ratio e^(wΔx) =<span class="g">${AI.fmt(Math.exp(w * dx), 3)}</span>   (odds ratio)\n  p: ${AI.fmt(p, 3)} → ${AI.fmt(sig(z + w * dx), 3)}`; void o2;
    }, { fps: 30 });
  };

  /* ===================================================================================================================
   * 4. Logistic regression lab
   * ================================================================================================================ */
  AI.labs.logisticLab = function (root) {
    AI.labShell(root, 'Logistic regression: watch the boundary rotate and settle', 'colour = predicted probability; the white band is p = 0.5. Click the plot to add points; edit η and λ',
      `<canvas id="map" style="height:330px"></canvas><canvas id="loss" style="height:120px"></canvas>`,
      `<label>Dataset <select id="data"><option value="linear">two overlapping blobs</option><option value="blobs">two separated blobs</option><option value="moons">moons</option><option value="xor">XOR</option></select></label>
       <label>Features <select id="deg"><option value="1">x, y (straight boundary)</option><option value="2">+ x², xy, y² (curved)</option><option value="3">+ cubic terms (very flexible)</option></select></label>
       ${slider('lr', 'Learning rate η', -2, 0.8, 0.02, 0)}${slider('l2', 'L2 strength λ (log₁₀)', -4, 0, 0.1, -4)}${speedSlider(8, false)}
       <label>Click adds <select id="cls"><option value="0">class 0 (blue)</option><option value="1">class 1 (orange)</option></select></label>
       ${transport('<button class="btn" id="new">New data</button>')}
       <div class="readout" id="out"></div>`);
    const q = R(root), c1 = AI.canvas(q('#map')), c2 = AI.canvas(q('#loss'));
    let data, F, w, b, wS, bS, hist = [], steps = 0, seed = 2, dirty = true, key = '', lastG = 0, diverged = false;
    const D = () => +q('#deg').value;
    const phi = (x, y) => { const o = [x, y]; if (D() >= 2) o.push(x * x, x * y, y * y); if (D() >= 3) o.push(x * x * x, x * x * y, x * y * y, y * y * y); return o; };
    const lam = () => (+q('#l2').value <= -4 ? 0 : Math.pow(10, +q('#l2').value)), lr = () => Math.pow(10, +q('#lr').value);
    const feat = () => { F = data.X.map((p) => phi(p[0], p[1])); };
    const reset = () => { const d = phi(0, 0).length; w = new Array(d).fill(0); b = 0; wS = [...w]; bS = 0; hist = []; steps = 0; diverged = false; lastG = 0; ck && ck.reset(); dirty = true; };
    const gen = () => { data = ML.data(q('#data').value, 120, seed, { K: 2, noise: 0.1 }); feat(); };
    const z = (ww, bb, f) => { let s = bb; for (let j = 0; j < f.length; j++) s += ww[j] * f[j]; return s; };
    const lossOf = (ww, bb) => { let s = 0; F.forEach((f, i) => { const p = sig(z(ww, bb, f)); s -= data.y[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12); }); return s / F.length + 0.5 * lam() * ww.reduce((a, c) => a + c * c, 0); };
    const stepOnce = () => {
      if (diverged) return; const n = F.length, g = new Array(w.length).fill(0); let gb = 0;
      F.forEach((f, i) => { const e = sig(z(w, b, f)) - data.y[i]; for (let j = 0; j < f.length; j++) g[j] += (e * f[j]) / n; gb += e / n; });
      for (let j = 0; j < w.length; j++) w[j] -= lr() * (g[j] + lam() * w[j]); b -= lr() * gb; lastG = Math.hypot(gb, ...g); steps++;
      const L = lossOf(w, b); hist.push(L); if (!Number.isFinite(L)) diverged = true; dirty = true;
    };
    const ck = clock(q, stepOnce);
    gen(); reset();
    q('#data').onchange = () => { seed += 1; gen(); reset(); }; q('#deg').onchange = () => { feat(); reset(); }; q('#new').onclick = () => { seed += 9; gen(); reset(); };
    q('#reset').onclick = reset; q('#lr').oninput = q('#l2').oninput = () => { hist = hist.slice(0, 0); dirty = true; };
    q('#map').addEventListener('pointerdown', (e) => { const r = q('#map').getBoundingClientRect(), m = ML.toPx(c1.w, c1.h, 1.3); data.X.push([m.ix(e.clientX - r.left), m.iy(e.clientY - r.top)]); data.y.push(+q('#cls').value); feat(); hist = []; dirty = true; });
    c1.onresize = c2.onresize = () => (dirty = true);
    const acc = () => F.reduce((a, f, i) => a + ((z(w, b, f) > 0 ? 1 : 0) === data.y[i] ? 1 : 0), 0) / F.length;

    AI.loop(root, (dt) => {
      ck.update(dt); q('#lrv').textContent = AI.fmt(lr(), 3); q('#l2v').textContent = lam() === 0 ? '0' : AI.fmt(lam(), 4);
      const kk = 1 - Math.exp(-dt * (ck.slow() ? 3 : 10)); let mv = 0; for (let j = 0; j < w.length; j++) { mv += Math.abs(w[j] - wS[j]); wS[j] += (w[j] - wS[j]) * kk; } mv += Math.abs(b - bS); bS += (b - bS) * kk;
      const nk = ck.phase + '|' + ck.playing + '|' + c1.w; if (!dirty && mv < 2e-4 && nk === key) return; key = nk; dirty = false;
      const C = AI.col();
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      ML.boundary(g, c1.w, c1.h, (x, y) => { const p = sig(z(wS, bS, phi(x, y))); return [1 - p, p]; }, { range: 1.3, res: 44, K: 2 });
      ML.points(g, c1.w, c1.h, data, { range: 1.3, r: 4 });
      const m = ML.toPx(c1.w, c1.h, 1.3);
      if (D() === 1 && (Math.abs(wS[0]) + Math.abs(wS[1])) > 1e-6) {
        const pts = [], [a, c] = wS; for (const [x, y] of [[-1.3, null], [1.3, null], [null, -1.3], [null, 1.3]]) { let px, py; if (x !== null) { px = x; py = Math.abs(c) > 1e-9 ? -(a * x + bS) / c : null; } else { py = y; px = Math.abs(a) > 1e-9 ? -(c * y + bS) / a : null; } if (px !== null && py !== null && Math.abs(px) <= 1.3 + 1e-6 && Math.abs(py) <= 1.3 + 1e-6) pts.push([px, py]); }
        if (pts.length >= 2) { g.strokeStyle = C.fg; g.lineWidth = 2.5; g.beginPath(); g.moveTo(m.X(pts[0][0]), m.Y(pts[0][1])); g.lineTo(m.X(pts[1][0]), m.Y(pts[1][1])); g.stroke(); }
        const nrm = Math.hypot(a, c), cx = -(bS * a) / (nrm * nrm), cy = -(bS * c) / (nrm * nrm), len = Math.min(0.5, 0.15 + nrm * 0.1);
        if (Math.abs(cx) < 1.3 && Math.abs(cy) < 1.3) AI.arrow(g, m.X(cx), m.Y(cy), m.X(cx + (a / nrm) * len), m.Y(cy + (c / nrm) * len), C.fg, 2);
      }
      const cap = ck.slow() && ck.playing ? SLOWCAPS[ck.phase] : diverged ? 'Diverged: η too large. Lower it and Reset.' : ck.playing ? `step ${steps}: p = σ(w·φ(x)+b) → loss → gradient mean((p−y)φ) → update` : 'Paused. Step, add points, or press Play.';
      caption(g, C, 8, 8, c1.w - 16, cap, ck.slow() && ck.playing ? (ck.phase + ck.phT / PH) / 4 : undefined);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const pad = { l: 44, r: 12, t: 18, b: 20 };
      const L0 = Math.log(2), hi = Math.max(L0 + 0.05, ...hist.slice(0, 1)), lo = Math.max(0, Math.min(...(hist.length ? hist : [0.3])) - 0.02), v = AI.view(c2, 0, Math.max(40, hist.length), lo, hi, pad);
      AI.axes(g, v); if (hist.length > 1) { g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); hist.forEach((y, i) => (i ? g.lineTo(v.X(i + 1), v.Y(y)) : g.moveTo(v.X(i + 1), v.Y(y)))); g.stroke(); }
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('cross-entropy loss vs step (starts at ln 2 = 0.693, should fall)', pad.l + 4, 12);
      const L = lossOf(w, b); q('#out').innerHTML = `<span class="m">step</span> ${steps}   <span class="m">loss</span> <span class="o">${AI.fmt(L, 4)}</span>   <span class="m">accuracy</span> <span class="g">${AI.fmt(acc() * 100, 1)}%</span>\n<span class="m">w</span> = [${w.slice(0, 2).map((x) => AI.fmt(x, 3)).join(', ')}${w.length > 2 ? ', …' : ''}]  <span class="m">b</span> = ${AI.fmt(b, 3)}\n<span class="m">‖gradient‖</span> ${AI.fmt(lastG, 4)} (→ 0 at the optimum)${D() === 1 ? `\n<span class="m">odds ratios</span>  e^w₁ = ${AI.fmt(Math.exp(w[0]), 3)}, e^w₂ = ${AI.fmt(Math.exp(w[1]), 3)}` : ''}`;
    }, { fps: 30 });
  };

  /* ===================================================================================================================
   * 5. Softmax regression lab
   * ================================================================================================================ */
  AI.labs.softmaxLab = function (root) {
    AI.labShell(root, 'Softmax regression: three classes carve the plane', 'regions are the arg-max of three linear scores; click a point to see its softmax and gradient p − y',
      `<canvas id="map" style="height:330px"></canvas><canvas id="loss" style="height:110px"></canvas>`,
      `<label>Dataset <select id="data"><option value="blobs">three separated blobs</option><option value="overlap">three overlapping blobs</option></select></label>
       ${slider('lr', 'Learning rate η', -2, 0.8, 0.02, -0.2)}${slider('l2', 'L2 strength λ (log₁₀)', -4, 0, 0.1, -4)}${speedSlider(8, false)}
       ${transport('<button class="btn" id="new">New data</button>')}
       <div class="readout" id="out"></div>`);
    const q = R(root), c1 = AI.canvas(q('#map')), c2 = AI.canvas(q('#loss')), K = 3;
    let data, W, bb, WS, bS, hist = [], steps = 0, seed = 3, sel = 0, dirty = true, key = '', diverged = false;
    const lam = () => (+q('#l2').value <= -4 ? 0 : Math.pow(10, +q('#l2').value)), lr = () => Math.pow(10, +q('#lr').value);
    const gen = () => {
      if (q('#data').value === 'blobs') data = ML.data('blobs', 120, seed, { K: 3 });
      else { const r = AI.rng(seed), X = [], y = []; for (let i = 0; i < 120; i++) { const c = i % 3, a = (c / 3) * 6.283 + 0.5; X.push([0.45 * Math.cos(a) + 0.38 * AI.randn(r), 0.45 * Math.sin(a) + 0.38 * AI.randn(r)]); y.push(c); } data = { X, y, K: 3 }; }
      sel = 0;
    };
    const scores = (Wm, b, p) => Wm.map((r, k) => r[0] * p[0] + r[1] * p[1] + b[k]);
    const soft = (zv) => { const m = Math.max(...zv), e = zv.map((v) => Math.exp(v - m)), s = e.reduce((a, c) => a + c, 0); return e.map((v) => v / s); };
    const lossOf = () => { let s = 0; data.X.forEach((p, i) => (s -= Math.log(soft(scores(W, bb, p))[data.y[i]] + 1e-12))); return s / data.X.length + 0.5 * lam() * W.reduce((a, r) => a + r[0] * r[0] + r[1] * r[1], 0); };
    const reset = () => { W = [[0, 0], [0, 0], [0, 0]]; bb = [0, 0, 0]; WS = W.map((r) => [...r]); bS = [0, 0, 0]; hist = []; steps = 0; diverged = false; ck && ck.reset(); dirty = true; };
    const stepOnce = () => {
      if (diverged) return; const n = data.X.length, gW = [[0, 0], [0, 0], [0, 0]], gb = [0, 0, 0];
      data.X.forEach((p, i) => { const pr = soft(scores(W, bb, p)); for (let k = 0; k < K; k++) { const e = pr[k] - (data.y[i] === k ? 1 : 0); gW[k][0] += (e * p[0]) / n; gW[k][1] += (e * p[1]) / n; gb[k] += e / n; } });
      for (let k = 0; k < K; k++) { W[k][0] -= lr() * (gW[k][0] + lam() * W[k][0]); W[k][1] -= lr() * (gW[k][1] + lam() * W[k][1]); bb[k] -= lr() * gb[k]; }
      steps++; const L = lossOf(); hist.push(L); if (!Number.isFinite(L)) diverged = true; dirty = true;
    };
    const ck = clock(q, stepOnce); gen(); reset();
    q('#data').onchange = () => { seed++; gen(); reset(); }; q('#new').onclick = () => { seed += 9; gen(); reset(); }; q('#reset').onclick = reset;
    q('#map').addEventListener('pointerdown', (e) => { const r = q('#map').getBoundingClientRect(), m = ML.toPx(c1.w, c1.h, 1.3); let best = 0, bd = 1e9; data.X.forEach((p, i) => { const d = (m.X(p[0]) - (e.clientX - r.left)) ** 2 + (m.Y(p[1]) - (e.clientY - r.top)) ** 2; if (d < bd) { bd = d; best = i; } }); sel = best; dirty = true; });
    c1.onresize = c2.onresize = () => (dirty = true);
    AI.loop(root, (dt) => {
      ck.update(dt); q('#lrv').textContent = AI.fmt(lr(), 3); q('#l2v').textContent = lam() === 0 ? '0' : AI.fmt(lam(), 4);
      const kk = 1 - Math.exp(-dt * (ck.slow() ? 3 : 10)); let mv = 0; for (let k = 0; k < K; k++) { for (let j = 0; j < 2; j++) { mv += Math.abs(W[k][j] - WS[k][j]); WS[k][j] += (W[k][j] - WS[k][j]) * kk; } mv += Math.abs(bb[k] - bS[k]); bS[k] += (bb[k] - bS[k]) * kk; }
      const nk = ck.phase + '|' + ck.playing + '|' + c1.w + '|' + sel; if (!dirty && mv < 2e-4 && nk === key) return; key = nk; dirty = false;
      const C = AI.col(), cols = ML.COLORS(3);
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      ML.boundary(g, c1.w, c1.h, (x, y) => soft(scores(WS, bS, [x, y])), { range: 1.3, res: 44, K: 3 }); ML.points(g, c1.w, c1.h, data, { range: 1.3, r: 4 });
      const m = ML.toPx(c1.w, c1.h, 1.3), sp = data.X[sel]; g.strokeStyle = C.fg; g.lineWidth = 2.5; g.beginPath(); g.arc(m.X(sp[0]), m.Y(sp[1]), 9, 0, 7); g.stroke();
      caption(g, C, 8, 8, c1.w - 16, ck.slow() && ck.playing ? SLOWCAPS[ck.phase] : diverged ? 'Diverged: lower η and Reset.' : ck.playing ? `step ${steps}: scores → softmax → gradient (p − y)x → update each class's weights` : 'Paused. Step, click a point, or press Play.', ck.slow() && ck.playing ? (ck.phase + ck.phT / PH) / 4 : undefined);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const pad = { l: 44, r: 12, t: 18, b: 20 }, hi = Math.log(3) + 0.05, lo = Math.max(0, Math.min(...(hist.length ? hist : [0.3])) - 0.02), v = AI.view(c2, 0, Math.max(40, hist.length), lo, hi, pad);
      AI.axes(g, v); if (hist.length > 1) { g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); hist.forEach((y, i) => (i ? g.lineTo(v.X(i + 1), v.Y(y)) : g.moveTo(v.X(i + 1), v.Y(y)))); g.stroke(); }
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('cross-entropy vs step (starts at ln 3 = 1.099)', pad.l + 4, 12);
      const zv = scores(W, bb, sp), pr = soft(zv), y = data.y[sel], names = ['blue', 'orange', 'green'];
      let acc = 0; data.X.forEach((p, i) => { const s = scores(W, bb, p); if (s.indexOf(Math.max(...s)) === data.y[i]) acc++; });
      q('#out').innerHTML = `<span class="m">step</span> ${steps}  <span class="m">loss</span> <span class="o">${AI.fmt(lossOf(), 4)}</span>  <span class="m">accuracy</span> <span class="g">${AI.fmt((acc / data.X.length) * 100, 1)}%</span>\n<span class="m">selected point</span> true class = ${names[y]}\n<span class="m">scores z </span> ${zv.map((x) => AI.fmt(x, 2)).join('  ')}\n<span class="m">softmax p </span> ${pr.map((x) => AI.fmt(x, 3)).join('  ')}\n<span class="m">p − y     </span> <span class="o">${pr.map((x, k) => AI.fmt(x - (k === y ? 1 : 0), 3)).join('  ')}</span>\n<span class="m">loss −ln p_y</span> ${AI.fmt(-Math.log(pr[y] + 1e-12), 3)}`; void cols;
    }, { fps: 30 });
  };

  /* ===================================================================================================================
   * 6. Optimisation lab: contours, learning rate, feature scaling (condition number)
   * ================================================================================================================ */
  AI.labs.optLab = function (root) {
    AI.labShell(root, 'Why feature scaling and the learning rate matter', 'a quadratic loss L = ½ eᵀ H e, where e = w − w*. Stretch one feature and the bowl becomes a narrow valley',
      `<canvas id="map" style="height:300px"></canvas><canvas id="loss" style="height:130px"></canvas>`,
      `${slider('sc', 'Scale of feature 2 (feature 1 has scale 1)', 0, 1.3, 0.02, 0)}${slider('rho', 'Correlation ρ between the features', -0.9, 0.9, 0.05, 0)}
       ${slider('lr', 'Learning rate η (log₁₀)', -4, 0.5, 0.02, -0.7)}${slider('mom', 'Mini-batch noise (illustrative)', 0, 1, 0.05, 0)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="std"> Standardise the features (mean 0, std 1)</label>
       ${slider('sp', 'Speed', 1, 60, 1, 6)}
       <div class="chips" id="reg"></div>
       ${transport()}
       <div class="readout" id="out"></div>`);
    const q = R(root), c1 = AI.canvas(q('#map')), c2 = AI.canvas(q('#loss'));
    const E0 = [-1.3, 0.35], VR = 1.6;
    let H, eig, e = [...E0], eS = [...E0], path = [[...E0]], hist = [], ghost = [], steps = 0, dirty = true, key = '', status = '', diverged = false, conv = -1, rnd = AI.rng(7);
    const tick = AI.ticker(); let playing = true;
    const buildH = () => {
      const s = q('#std').checked ? 1 : Math.pow(10, +q('#sc').value), r = +q('#rho').value, a = 1, c = s * s, bb = r * s;
      H = [[a, bb], [bb, c]]; const tr = a + c, det = a * c - bb * bb, d = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + d, l2 = tr / 2 - d;
      const vec = (l) => { let v = Math.abs(bb) > 1e-12 ? [l - c, bb] : l === l1 ? (a >= c ? [1, 0] : [0, 1]) : (a >= c ? [0, 1] : [1, 0]); const n = Math.hypot(v[0], v[1]); return [v[0] / n, v[1] / n]; };
      eig = { l1, l2, v1: vec(l1), v2: vec(l2), kappa: l1 / Math.max(l2, 1e-12), scale: s };
    };
    const L = (ee) => 0.5 * (H[0][0] * ee[0] * ee[0] + 2 * H[0][1] * ee[0] * ee[1] + H[1][1] * ee[1] * ee[1]);
    const lr = () => Math.pow(10, +q('#lr').value);
    const restart = () => { if (hist.length > 3) ghost = hist.slice(0, 400); e = [...E0]; eS = [...E0]; path = [[...E0]]; hist = [L(E0)]; steps = 0; diverged = false; conv = -1; status = ''; dirty = true; };
    const stepOnce = () => {
      if (diverged) return; const noise = +q('#mom').value, gn = Math.sqrt(eig.l1) * noise * Math.sqrt(2 * L(e)) * 0.5;
      const g0 = H[0][0] * e[0] + H[0][1] * e[1] + gn * AI.randn(rnd), g1 = H[0][1] * e[0] + H[1][1] * e[1] + gn * AI.randn(rnd);
      e[0] -= lr() * g0; e[1] -= lr() * g1; steps++; const l = L(e); hist.push(l); path.push([e[0], e[1]]); if (path.length > 4000) path.shift();
      if (!Number.isFinite(l) || l > 1e8) diverged = true; if (conv < 0 && l / hist[0] < 1e-4) conv = steps; dirty = true;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; stepOnce(); };
    q('#reset').onclick = restart;
    ['#sc', '#rho', '#std'].forEach((s) => (q(s).oninput = q(s).onchange = () => { buildH(); restart(); }));
    q('#lr').oninput = () => { dirty = true; };
    const reg = q('#reg'); [['too slow (0.05× max)', 0.05], ['good (0.5× max)', 0.5], ['zig-zag (0.95× max)', 0.95], ['diverges (1.05× max)', 1.05]].forEach(([t, f]) => { const bt = document.createElement('button'); bt.className = 'chip'; bt.textContent = t; bt.onclick = () => { q('#lr').value = Math.log10(f * (2 / eig.l1)); restart(); }; reg.appendChild(bt); });
    buildH(); restart();
    c1.onresize = c2.onresize = () => (dirty = true);
    AI.loop(root, (dt) => {
      if (playing) { const n = tick(dt, +q('#sp').value); for (let i = 0; i < n; i++) stepOnce(); }
      q('#scv').textContent = q('#std').checked ? '1 (standardised)' : AI.fmt(Math.pow(10, +q('#sc').value), 3); q('#rhov').textContent = AI.fmt(+q('#rho').value, 2); q('#lrv').textContent = AI.fmt(lr(), 4); q('#momv').textContent = AI.fmt(+q('#mom').value, 2); q('#spv').textContent = q('#sp').value + ' steps / s';
      const kk = 1 - Math.exp(-dt * 12); const mv = Math.abs(e[0] - eS[0]) + Math.abs(e[1] - eS[1]); eS[0] += (e[0] - eS[0]) * kk; eS[1] += (e[1] - eS[1]) * kk;
      const nk = steps + '|' + c1.w + '|' + dirty + '|' + lr(); if (!dirty && mv < 1e-4 && nk === key) return; key = nk; dirty = false;
      const C = AI.col(); let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h); const v = AI.view(c1, -VR, VR, -VR, VR, { l: 36, r: 10, t: 10, b: 24 }); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const L0 = L(E0); g.strokeStyle = C.blue; g.lineWidth = 1.2;
      [1, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003].forEach((f, i) => { g.globalAlpha = 0.9 - i * 0.08; g.beginPath(); const a1 = Math.sqrt((2 * L0 * f) / eig.l1), a2 = Math.sqrt((2 * L0 * f) / Math.max(eig.l2, 1e-9)); for (let t = 0; t <= 96; t++) { const th = (t / 96) * 6.2832, u = a1 * Math.cos(th), w2 = a2 * Math.sin(th), x = u * eig.v1[0] + w2 * eig.v2[0], y = u * eig.v1[1] + w2 * eig.v2[1]; t ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke(); });
      g.globalAlpha = 1; g.strokeStyle = C.orange; g.lineWidth = 1.6; g.beginPath(); path.forEach((p, i) => (i ? g.lineTo(v.X(p[0]), v.Y(p[1])) : g.moveTo(v.X(p[0]), v.Y(p[1])))); g.stroke();
      g.fillStyle = C.orange; path.length < 400 && path.forEach((p) => { g.beginPath(); g.arc(v.X(p[0]), v.Y(p[1]), 2.2, 0, 7); g.fill(); });
      g.fillStyle = C.green; g.beginPath(); g.arc(v.X(0), v.Y(0), 6, 0, 7); g.fill(); g.fillStyle = C.orange; g.strokeStyle = C.fg; g.lineWidth = 1.5; g.beginPath(); g.arc(v.X(eS[0]), v.Y(eS[1]), 5.5, 0, 7); g.fill(); g.stroke(); g.restore();
      const etaMax = 2 / eig.l1, regime = diverged ? 'DIVERGED: η > 2/λmax' : lr() > etaMax ? 'unstable: η > 2/λmax' : lr() > 0.7 * etaMax ? 'zig-zag: η close to the limit' : conv > 0 ? 'converged' : 'stable';
      caption(g, C, v.pad.l + 4, v.pad.t + 4, v.pw - 8, `${playing ? 'step ' + steps : 'paused'} · ${regime} · ● optimum w*`);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const pad = { l: 44, r: 12, t: 18, b: 20 }, lg = (a, h0) => Math.log10(Math.max(a / h0, 1e-8));
      const n = Math.max(60, hist.length), vv = AI.view(c2, 0, n, -6, 0.5, pad); AI.axes(g, vv);
      if (ghost.length > 1) { g.strokeStyle = C.muted; g.globalAlpha = 0.6; g.lineWidth = 1.5; g.setLineDash([4, 4]); g.beginPath(); ghost.forEach((y, i) => { const yy = AI.clamp(lg(y, ghost[0]), -6, 0.5); i ? g.lineTo(vv.X(i), vv.Y(yy)) : g.moveTo(vv.X(i), vv.Y(yy)); }); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
      if (hist.length > 1) { g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); hist.forEach((y, i) => { const yy = AI.clamp(lg(y, hist[0]), -6, 0.5); i ? g.lineTo(vv.X(i), vv.Y(yy)) : g.moveTo(vv.X(i), vv.Y(yy)); }); g.stroke(); }
      g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('log₁₀ (L / L₀) vs step: orange = this run, dashed grey = previous run', pad.l + 4, 12);
      const slow = eig.kappa > 1 ? Math.ceil(Math.log(1e4) / (-Math.log(Math.max(1e-9, 1 - 2 * Math.min(lr(), etaMax * 0.999) * eig.l2 / 2 * 1)))) : 0; void slow;
      q('#out').innerHTML = `<span class="m">eigenvalues of H</span> ${AI.fmt(eig.l1, 4)}, ${AI.fmt(eig.l2, 4)}\n<span class="m">condition number κ = λmax/λmin</span> <span class="o">${AI.fmt(eig.kappa, 4)}</span>\n<span class="m">largest stable η = 2/λmax</span> ${AI.fmt(etaMax, 4)}   <span class="m">your η</span> ${AI.fmt(lr(), 4)}\n<span class="m">best fixed η = 2/(λmax+λmin)</span> ${AI.fmt(2 / (eig.l1 + eig.l2), 4)}\n<span class="m">step ${steps}</span>  L/L₀ = ${AI.fmt(hist[hist.length - 1] / hist[0], 3)}${conv > 0 ? `  <span class="g">reached 10⁻⁴ at step ${conv}</span>` : ''}${diverged ? '\n<span class="r">Diverged. Lower η, then Reset.</span>' : ''}`; void status;
    }, { fps: 30 });
  };
})();
