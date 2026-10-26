/* AI Learning Hub — Module 4 (Statistics) labs.
   mleCoin, mleGauss, mapCoin, regShrink, biasVar, ciCoverage, pTest, kfold, bootstrap, leakage.
   All motion is dt-based, slow by default, with speed sliders and pause/step. */
(function () {
  const q = (r, s) => AI.$(s, r);
  const nv = (r, id) => +AI.$('#' + id, r).value;
  const sl = (id, label, min, max, step, val, pre = '', suf = '') =>
    `<label>${label}<span class="v" id="${id}_v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}" data-pre="${pre}" data-suf="${suf}"></label>`;
  const sync = (r) => AI.$$('input[type=range]', r).forEach((i) => { const s = AI.$('#' + i.id + '_v', r); if (s) s.textContent = i.dataset.pre + AI.fmt(+i.value, 3) + i.dataset.suf; });
  const btns = (extra = '') => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▸</button><button class="btn" id="reset">Reset</button>${extra}</div>`;
  const capEl = (h = 3.2) => `<div class="sub" id="cap" style="font-size:13px;line-height:1.45;min-height:${h}em"></div>`;
  const cvs = (h) => `<canvas style="height:${h}px"></canvas>`;
  const text = (g, s, x, y, col, al = 'left', f = '12px sans-serif') => { g.fillStyle = col; g.textAlign = al; g.font = f; g.fillText(s, x, y); };
  const ease = (cur, tgt, dt, rate = 6) => cur + (tgt - cur) * (1 - Math.exp(-dt * rate));
  const erf = (x) => { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const playCtl = (root, st, extraReset) => {
    st.playing = true;
    st.setPlay = (on) => { st.playing = on; q(root, '#play').textContent = on ? '⏸ Pause' : '▶ Play'; };
    q(root, '#play').onclick = () => st.setPlay(!st.playing);
  };

  /* ---------- polynomial least squares in a Chebyshev basis (well conditioned on [-1,1]) ---------- */
  function cheb(x, d, out) { out[0] = 1; if (d >= 1) out[1] = x; for (let k = 2; k <= d; k++) out[k] = 2 * x * out[k - 1] - out[k - 2]; }
  function polyfit(xs, ys, d) {
    const m = d + 1, A = new Float64Array(m * m), b = new Float64Array(m), phi = new Float64Array(m);
    for (let i = 0; i < xs.length; i++) {
      cheb(xs[i], d, phi);
      for (let r = 0; r < m; r++) { b[r] += phi[r] * ys[i]; for (let c = 0; c < m; c++) A[r * m + c] += phi[r] * phi[c]; }
    }
    for (let r = 0; r < m; r++) A[r * m + r] += 1e-8;
    for (let c = 0; c < m; c++) {
      let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r * m + c]) > Math.abs(A[p * m + c])) p = r;
      if (p !== c) { for (let k = 0; k < m; k++) { const t = A[c * m + k]; A[c * m + k] = A[p * m + k]; A[p * m + k] = t; } const t = b[c]; b[c] = b[p]; b[p] = t; }
      for (let r = c + 1; r < m; r++) { const f = A[r * m + c] / A[c * m + c]; for (let k = c; k < m; k++) A[r * m + k] -= f * A[c * m + k]; b[r] -= f * b[c]; }
    }
    const w = new Float64Array(m);
    for (let r = m - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < m; k++) s -= A[r * m + k] * w[k]; w[r] = s / A[r * m + r]; }
    return w;
  }
  function polyval(w, x) { const d = w.length - 1, phi = polyval.buf || (polyval.buf = new Float64Array(16)); cheb(x, d, phi); let s = 0; for (let k = 0; k <= d; k++) s += w[k] * phi[k]; return s; }
  const trueF = (x) => Math.sin(3 * x);

  /* =====================================================================================
     1. mleCoin — likelihood of a coin's bias, flips arriving one by one
     ===================================================================================== */
  AI.labs.mleCoin = function (root) {
    const { view, ctl } = AI.labShell(root, 'Likelihood lab: estimating a coin’s bias', 'flips arrive one by one — watch the likelihood curve reshape and its peak move',
      cvs(300) + capEl(),
      sl('pt', 'True bias p (hidden from the estimator)', 0.05, 0.95, 0.05, 0.7) + sl('sp', 'Speed (flips per second)', 0.25, 4, 0.25, 0.75) +
      `<label>View<select id="mode"><option value="L">Likelihood L(p), scaled so the peak is 1</option><option value="LL">Log-likelihood ℓ(p), peak at 0</option></select></label>` +
      btns() + `<div class="row"><button class="btn" id="H">+ Heads</button><button class="btn" id="T">+ Tails</button></div><div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), N = 201, st = {};
    playCtl(root, st);
    let h = 0, t = 0, seq = '', flash = 0, last = '', mode = 'L';
    let disp = new Float64Array(N).fill(1), tgt = new Float64Array(N).fill(1);
    const tick = AI.ticker();
    const recompute = () => {
      const ll = new Float64Array(N); let mx = -Infinity;
      for (let i = 0; i < N; i++) { const p = (i + 0.5) / N; ll[i] = h * Math.log(p) + t * Math.log(1 - p); if (ll[i] > mx) mx = ll[i]; }
      for (let i = 0; i < N; i++) tgt[i] = mode === 'L' ? Math.exp(ll[i] - mx) : ll[i] - mx;
    };
    const add = (f) => { if (h + t >= 400) { st.setPlay(false); return; } if (f) h++; else t++; seq += f ? 'H' : 'T'; last = f ? 'H' : 'T'; flash = 1.6; recompute(); };
    const reset = () => { h = t = 0; seq = ''; last = ''; flash = 0; recompute(); disp.set(tgt); };
    q(root, '#reset').onclick = reset;
    q(root, '#step').onclick = () => { st.setPlay(false); add(Math.random() < nv(root, 'pt')); };
    q(root, '#H').onclick = () => add(true); q(root, '#T').onclick = () => add(false);
    q(root, '#mode').onchange = (e) => { mode = e.target.value; recompute(); disp.set(tgt); };
    AI.loop(root, (dt) => {
      sync(root);
      if (st.playing) { const k = tick(dt, nv(root, 'sp')); for (let i = 0; i < k; i++) add(Math.random() < nv(root, 'pt')); }
      for (let i = 0; i < N; i++) disp[i] = ease(disp[i], tgt[i], dt, 5);
      flash = Math.max(0, flash - dt);
      const C = AI.col(), g = c.ctx, n = h + t, pad = { l: 44, r: 14, t: 14, b: 30 };
      const ymin = mode === 'L' ? -0.05 : -12, ymax = mode === 'L' ? 1.12 : 0.8;
      const v = AI.view(c, 0, 1, ymin, ymax, pad);
      g.clearRect(0, 0, c.w, c.h);
      AI.axes(g, v);
      text(g, 'candidate value of p', v.pad.l + v.pw / 2, c.h - 4, C.muted, 'center');
      text(g, mode === 'L' ? 'L(p) / L(p̂)' : 'ℓ(p) − ℓ(p̂)', 4, 12, C.muted);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.beginPath();
      for (let i = 0; i < N; i++) { const x = v.X((i + 0.5) / N), y = v.Y(disp[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.stroke();
      if (mode === 'L') { g.lineTo(v.X(1), v.Y(0)); g.lineTo(v.X(0), v.Y(0)); g.closePath(); g.globalAlpha = 0.15; g.fillStyle = C.blue; g.fill(); g.globalAlpha = 1; }
      g.restore();
      const tp = nv(root, 'pt');
      g.setLineDash([5, 4]); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(tp), v.pad.t); g.lineTo(v.X(tp), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      text(g, 'true p', v.X(tp) + 4, v.pad.t + 12, C.muted);
      if (n > 0) {
        const ph = h / n; g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(ph), v.pad.t); g.lineTo(v.X(ph), v.pad.t + v.ph); g.stroke();
        text(g, `p̂ = ${AI.fmt(ph, 3)}`, v.X(ph) + (ph > 0.8 ? -6 : 6), v.pad.t + 46, C.green, ph > 0.8 ? 'right' : 'left', 'bold 12px sans-serif');
      }
      if (flash > 0) { g.globalAlpha = Math.min(1, flash); g.fillStyle = last === 'H' ? C.orange : C.purple; g.beginPath(); g.arc(v.pad.l + v.pw - 24, v.pad.t + 24, 14, 0, 7); g.fill(); text(g, last, v.pad.l + v.pw - 24, v.pad.t + 29, '#fff', 'center', 'bold 14px sans-serif'); g.globalAlpha = 1; }
      q(root, '#cap').innerHTML = n === 0 ? 'No data yet: the likelihood is the empty product, 1, for every p — every bias is equally compatible. Flips will start arriving (or add your own).'
        : `flip ${n}: <b>${last === 'H' ? 'heads → multiply L(p) by p' : 'tails → multiply L(p) by (1−p)'}</b>. After ${h} heads and ${t} tails, L(p) = p<sup>${h}</sup>(1−p)<sup>${t}</sup>; its peak sits at h/n = ${AI.fmt(h / n, 3)}.`;
      const ph = n ? h / n : NaN, lmax = n ? h * Math.log(Math.max(ph, 1e-300)) + t * Math.log(Math.max(1 - ph, 1e-300)) : 0;
      q(root, '#out').innerHTML = `flips n     <b>${n}</b>   heads h <b>${h}</b>   tails <b>${t}</b>\nMLE  p̂=h/n  <b class="g">${n ? AI.fmt(ph, 4) : '–'}</b>\nL(p̂)        <span class="o">${n ? (Math.exp(lmax)).toExponential(2) : '1'}</span>   <span class="m">(tiny!)</span>\nℓ(p̂)        <span class="o">${AI.fmt(lmax, 3)}</span>   <span class="m">(a sum: easy)</span>\n<span class="m">${seq.slice(-36) || 'sequence: (empty)'}</span>`;
    }, { fps: 40 });
    reset();
  };

  /* =====================================================================================
     2. mleGauss — MLE of a Gaussian mean = least squares
     ===================================================================================== */
  AI.labs.mleGauss = function (root) {
    const { view, ctl } = AI.labShell(root, 'Likelihood lab: why least squares is maximum likelihood', 'slide the Gaussian’s centre μ — likelihood rises exactly as squared error falls',
      cvs(210) + cvs(190) + capEl(2.6),
      `<label>Data (editable, 2–25 numbers)<input type="text" id="data" value="2, 4, 9" spellcheck="false"></label><div class="err" id="err"></div>` +
      sl('mu', 'Centre μ', -5, 15, 0.05, 0.5) + sl('sg', 'Spread σ (held fixed)', 0.5, 6, 0.1, 2.5) +
      `<div class="row"><button class="btn primary" id="sweep">▶ Sweep μ</button><button class="btn" id="snap">μ = mean</button></div>` + sl('sp', 'Sweep speed', 0.25, 3, 0.25, 1) + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e));
    let data = [2, 4, 9], sweeping = false, dir = 1, lo = -2, hi = 13, muF = 0;
    const mu = q(root, '#mu');
    const setData = () => {
      const a = q(root, '#data').value.split(/[,\s]+/).filter(Boolean).map(Number);
      if (a.length < 2 || a.length > 25 || a.some((x) => !Number.isFinite(x))) { q(root, '#err').textContent = 'Enter 2 to 25 numbers separated by commas.'; return; }
      q(root, '#err').textContent = ''; data = a;
      const mn = Math.min(...a), mx = Math.max(...a), pad = Math.max(3, (mx - mn) * 0.4);
      lo = mn - pad; hi = mx + pad; mu.min = lo.toFixed(2); mu.max = hi.toFixed(2); mu.value = Math.min(hi, Math.max(lo, +mu.value)); muF = +mu.value;
    };
    q(root, '#data').oninput = setData; setData(); mu.value = lo + 1; muF = +mu.value; mu.oninput = () => { muF = +mu.value; };
    q(root, '#sweep').onclick = () => { sweeping = !sweeping; q(root, '#sweep').textContent = sweeping ? '⏸ Stop sweep' : '▶ Sweep μ'; };
    q(root, '#snap').onclick = () => { sweeping = false; q(root, '#sweep').textContent = '▶ Sweep μ'; muF = data.reduce((a, b) => a + b, 0) / data.length; mu.value = muF; };
    AI.loop(root, (dt) => {
      sync(root);
      if (sweeping) { let m = muF + dir * dt * ((hi - lo) / 10) * nv(root, 'sp'); if (m > hi) { m = hi; dir = -1; } if (m < lo) { m = lo; dir = 1; } muF = m; mu.value = m; }
      const C = AI.col(), n = data.length, m = muF, s = nv(root, 'sg'), mean = data.reduce((a, b) => a + b, 0) / n;
      const sse = data.reduce((a, x) => a + (x - m) ** 2, 0), ll = (mu_) => -0.5 * n * Math.log(2 * Math.PI * s * s) - data.reduce((a, x) => a + (x - mu_) ** 2, 0) / (2 * s * s);
      // top: data and bell
      let c = cs[0], g = c.ctx, v = AI.view(c, lo, hi, -0.06 * 0.5, npdf(0, 0, s) * 1.25, { l: 12, r: 12, t: 12, b: 26 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v, { noY: true });
      g.beginPath(); for (let i = 0; i <= 160; i++) { const x = lo + ((hi - lo) * i) / 160, y = v.Y(npdf(x, m, s)); i ? g.lineTo(v.X(x), y) : g.moveTo(v.X(x), y); }
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.stroke(); g.lineTo(v.X(hi), v.Y(0)); g.lineTo(v.X(lo), v.Y(0)); g.globalAlpha = 0.12; g.fillStyle = C.blue; g.fill(); g.globalAlpha = 1;
      for (const x of data) {
        const d = npdf(x, m, s); g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(x), v.Y(0)); g.lineTo(v.X(x), v.Y(d)); g.stroke();
        g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(x), v.Y(d), 3.5, 0, 7); g.fill();
        g.fillStyle = C.fg; g.beginPath(); g.arc(v.X(x), v.Y(0), 5, 0, 7); g.fill();
        text(g, AI.fmt(d, 3), v.X(x), v.Y(d) - 8, C.orange, 'center', '11px sans-serif');
      }
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(m), v.Y(0)); g.lineTo(v.X(m), v.pad.t); g.stroke(); g.setLineDash([]);
      text(g, 'orange heights = density each data point receives; the likelihood is their product', v.pad.l + 2, 11, C.muted, 'left', '11px sans-serif');
      // bottom: LL(mu)
      c = cs[1]; g = c.ctx;
      const G = 120, ys = []; let ymx = -Infinity, ymn = Infinity;
      for (let i = 0; i <= G; i++) { const y = ll(lo + ((hi - lo) * i) / G); ys.push(y); ymx = Math.max(ymx, y); ymn = Math.min(ymn, y); }
      v = AI.view(c, lo, hi, ymn - 0.05 * (ymx - ymn), ymx + 0.12 * (ymx - ymn), { l: 52, r: 12, t: 12, b: 26 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.beginPath(); ys.forEach((y, i) => { const x = v.X(lo + ((hi - lo) * i) / G); i ? g.lineTo(x, v.Y(y)) : g.moveTo(x, v.Y(y)); }); g.strokeStyle = C.green; g.lineWidth = 2.5; g.stroke();
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(mean), v.pad.t); g.lineTo(v.X(mean), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      text(g, `peak at the mean ${AI.fmt(mean, 3)}`, v.X(mean) + 5, v.pad.t + 12, C.muted);
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(m), v.Y(ll(m)), 6, 0, 7); g.fill();
      text(g, 'log-likelihood ℓ(μ)', 4, 11, C.muted); text(g, 'μ', v.pad.l + v.pw / 2, c.h - 3, C.muted, 'center');
      q(root, '#cap').innerHTML = `Squared distances: ${data.map((x) => AI.fmt((x - m) ** 2, 2)).join(' + ')} = <b>${AI.fmt(sse, 3)}</b>. Move μ and compare: the smaller this sum, the higher ℓ(μ).`;
      q(root, '#out').innerHTML = `n = ${n}   mean x̄ = <b class="g">${AI.fmt(mean, 4)}</b>\nSSE(μ) = Σ(xᵢ−μ)²  <b class="o">${AI.fmt(sse, 3)}</b>\nℓ(μ) = −n/2·ln(2πσ²) − SSE/(2σ²)\n     = <b class="o">${AI.fmt(ll(m), 3)}</b>\nℓ at the mean       <b class="g">${AI.fmt(ll(mean), 3)}</b>\nσ̂²_MLE (÷n) = <b>${AI.fmt(data.reduce((a, x) => a + (x - mean) ** 2, 0) / n, 4)}</b>   (÷(n−1): ${AI.fmt(data.reduce((a, x) => a + (x - mean) ** 2, 0) / (n - 1), 4)})`;
    }, { fps: 40 });
  };

  /* =====================================================================================
     3. mapCoin — prior × likelihood = posterior
     ===================================================================================== */
  AI.labs.mapCoin = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bayes lab: prior × likelihood = posterior', 'a prior is “imaginary data”; real flips gradually outvote it',
      cvs(310) + capEl(),
      sl('s', 'Prior strength = imaginary flips', 0, 40, 1, 10) + sl('m', 'Prior belief: P(heads) =', 0.1, 0.9, 0.05, 0.5) +
      sl('pt', 'True bias p of the coin', 0.05, 0.95, 0.05, 0.8) + sl('sp', 'Speed (flips per second)', 0.25, 4, 0.25, 0.75) + btns() + `<div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), N = 201, st = {}; playCtl(root, st);
    let h = 0, t = 0, last = '';
    const A = { pri: new Float64Array(N), lik: new Float64Array(N), pos: new Float64Array(N) }, D = { pri: new Float64Array(N), lik: new Float64Array(N), pos: new Float64Array(N) };
    let ymaxD = 3;
    const tick = AI.ticker();
    const dens = (out, f) => { let s = 0; const raw = new Float64Array(N), l = new Float64Array(N); let mx = -Infinity; for (let i = 0; i < N; i++) { l[i] = f((i + 0.5) / N); if (l[i] > mx) mx = l[i]; } for (let i = 0; i < N; i++) { raw[i] = Math.exp(l[i] - mx); s += raw[i]; } for (let i = 0; i < N; i++) out[i] = raw[i] / (s / N); };
    const params = () => { const s = nv(root, 's'), m = nv(root, 'm'); return { a: 1 + m * s, b: 1 + (1 - m) * s, s, m }; };
    const recompute = () => { const { a, b } = params(); dens(A.pri, (p) => (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p)); dens(A.lik, (p) => h * Math.log(p) + t * Math.log(1 - p)); dens(A.pos, (p) => (h + a - 1) * Math.log(p) + (t + b - 1) * Math.log(1 - p)); };
    const add = () => { if (h + t >= 300) { st.setPlay(false); return; } if (Math.random() < nv(root, 'pt')) h++; else t++; last = ''; recompute(); };
    const reset = () => { h = t = 0; recompute(); for (const k in A) D[k].set(A[k]); };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { st.setPlay(false); add(); };
    ['s', 'm'].forEach((id) => (q(root, '#' + id).oninput = recompute));
    let ptPrev = nv(root, 'pt'); q(root, '#pt').oninput = () => { if (nv(root, 'pt') !== ptPrev) { ptPrev = nv(root, 'pt'); reset(); } };
    AI.loop(root, (dt) => {
      sync(root);
      if (st.playing) { const k = tick(dt, nv(root, 'sp')); for (let i = 0; i < k; i++) add(); }
      let mxT = 0; for (let i = 0; i < N; i++) for (const k in A) { D[k][i] = ease(D[k][i], A[k][i], dt, 5); if (A[k][i] > mxT) mxT = A[k][i]; }
      ymaxD = ease(ymaxD, Math.max(2.2, mxT * 1.12), dt, 3);
      const C = AI.col(), g = c.ctx, n = h + t, { a, b, s, m } = params(), v = AI.view(c, 0, 1, 0, ymaxD, { l: 38, r: 14, t: 14, b: 30 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v, { noGrid: false });
      text(g, 'candidate p', v.pad.l + v.pw / 2, c.h - 4, C.muted, 'center'); text(g, 'density', 4, 12, C.muted);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const curve = (arr, col, w, fill) => { g.beginPath(); for (let i = 0; i < N; i++) { const x = v.X((i + 0.5) / N), y = v.Y(arr[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.strokeStyle = col; g.lineWidth = w; g.stroke(); if (fill) { g.lineTo(v.X(1), v.Y(0)); g.lineTo(v.X(0), v.Y(0)); g.closePath(); g.globalAlpha = 0.16; g.fillStyle = col; g.fill(); g.globalAlpha = 1; } };
      curve(D.pri, C.purple, 2); curve(D.lik, C.orange, 2); curve(D.pos, C.green, 3.2, true);
      g.restore();
      const tick1 = (x, col, label, row) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(x), v.pad.t + v.ph); g.lineTo(v.X(x), v.pad.t + v.ph - 16); g.stroke(); text(g, label, v.X(x), v.pad.t + v.ph - 20 - row * 13, col, 'center', 'bold 11px sans-serif'); };
      const mle = n ? h / n : null, map = (h + a - 1) / (n + a + b - 2 || 1);
      tick1(m, C.purple, 'prior mode', 0); if (mle !== null) tick1(mle, C.orange, 'MLE', 1); tick1(map, C.green, 'MAP', 2);
      text(g, '— prior', v.pad.l + 6, v.pad.t + 12, C.purple, 'left', 'bold 12px sans-serif'); text(g, '— likelihood (rescaled to area 1)', v.pad.l + 60, v.pad.t + 12, C.orange, 'left', 'bold 12px sans-serif'); text(g, '— posterior ∝ prior × likelihood', v.pad.l + 6, v.pad.t + 27, C.green, 'left', 'bold 12px sans-serif');
      const w = (s) / (n + s || 1);
      q(root, '#cap').innerHTML = n === 0 ? `Before any data the posterior <i>is</i> the prior: Beta(${AI.fmt(a, 2)}, ${AI.fmt(b, 2)}), as if you had already seen ${AI.fmt(s, 0)} imaginary flips (${AI.fmt(m * s, 1)} heads).`
        : `${h} heads and ${t} tails so far. The posterior peak (MAP) is a weighted compromise: <b>${AI.fmt(100 * w, 0)}%</b> prior (${AI.fmt(s, 0)} imaginary flips) and <b>${AI.fmt(100 * (1 - w), 0)}%</b> data (${n} real flips).`;
      q(root, '#out').innerHTML = `data   h=${h}  t=${t}  n=${n}\nMLE  = h/n            <b class="o">${mle === null ? '–' : AI.fmt(mle, 4)}</b>\nMAP  = (h+m·s)/(n+s)  <b class="g">${AI.fmt(map, 4)}</b>\nprior ≡ Beta(${AI.fmt(a, 2)}, ${AI.fmt(b, 2)})\nposterior = Beta(${AI.fmt(a + h, 2)}, ${AI.fmt(b + t, 2)})\nposterior mean  <span class="o">${AI.fmt((a + h) / (a + b + n), 4)}</span>`;
    }, { fps: 40 });
    reset();
  };

  /* =====================================================================================
     4. regShrink — L2 / L1 penalties = Gaussian / Laplace priors
     ===================================================================================== */
  AI.labs.regShrink = function (root) {
    const { view, ctl } = AI.labShell(root, 'Regularization lab: one weight, pulled toward zero', 'penalty = prior; see the loss view and the probability view side by side',
      cvs(230) + cvs(170) + capEl(2.8),
      `<label>Penalty (= prior)<select id="mode"><option value="2">L2 = Gaussian prior</option><option value="1">L1 = Laplace prior</option></select></label>` +
      sl('w0', 'Unregularized best weight w₀ (OLS)', -3, 3, 0.05, 2) + sl('n', 'Amount of data n (Sxx = n)', 1, 100, 1, 10) + sl('lam', 'Penalty strength λ', 0, 60, 0.5, 10) +
      `<div class="row"><button class="btn primary" id="sweep">▶ Sweep λ</button></div>` + sl('sp', 'Sweep speed (λ per second)', 1, 12, 1, 4) + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e));
    let sweeping = false, dir = 1, wd = 2, ymaxD = 10, lamF = 10;
    q(root, '#lam').oninput = () => { lamF = +q(root, '#lam').value; };
    q(root, '#sweep').onclick = () => { sweeping = !sweeping; q(root, '#sweep').textContent = sweeping ? '⏸ Stop sweep' : '▶ Sweep λ'; };
    AI.loop(root, (dt) => {
      sync(root);
      const lamEl = q(root, '#lam');
      if (sweeping) { let l = lamF + dir * dt * nv(root, 'sp'); if (l > 60) { l = 60; dir = -1; } if (l < 0) { l = 0; dir = 1; } lamF = l; lamEl.value = l; }
      const C = AI.col(), L1 = q(root, '#mode').value === '1', w0 = nv(root, 'w0'), n = nv(root, 'n'), lam = lamF;
      const data = (w) => 0.5 * n * (w - w0) ** 2, pen = (w) => (L1 ? lam * Math.abs(w) : 0.5 * lam * w * w), tot = (w) => data(w) + pen(w);
      const ws = L1 ? Math.sign(w0) * Math.max(Math.abs(w0) - lam / n, 0) : (n * w0) / (n + lam);
      wd = ease(wd, ws, dt, 6);
      const ymT = Math.max(3, 1.5 * Math.max(data(0), tot(ws) + 1) , 0.8 * pen(w0) + 1); ymaxD = ease(ymaxD, ymT, dt, 4);
      let c = cs[0], g = c.ctx, v = AI.view(c, -4, 4, 0, ymaxD, { l: 44, r: 12, t: 12, b: 26 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const cur = (f, col, lw, dash) => { g.beginPath(); for (let i = 0; i <= 240; i++) { const w = -4 + (8 * i) / 240; i ? g.lineTo(v.X(w), v.Y(f(w))) : g.moveTo(v.X(w), v.Y(f(w))); } g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); };
      cur(data, C.blue, 2); cur(pen, C.orange, 2); cur(tot, C.green, 3);
      g.restore();
      g.strokeStyle = C.blue; g.globalAlpha = 0.6; g.beginPath(); g.moveTo(v.X(w0), v.pad.t + v.ph); g.lineTo(v.X(w0), v.pad.t + v.ph - 14); g.stroke(); g.globalAlpha = 1;
      text(g, 'w₀', v.X(w0), v.pad.t + v.ph - 17, C.blue, 'center', 'bold 11px sans-serif');
      g.fillStyle = C.green; g.beginPath(); g.arc(v.X(wd), v.Y(tot(wd)), 6, 0, 7); g.fill(); text(g, `w* = ${AI.fmt(wd, 3)}`, v.X(wd), v.Y(tot(wd)) - 11, C.green, 'center', 'bold 12px sans-serif');
      text(g, '— data loss ½n(w−w₀)²', v.pad.l + 6, 12, C.blue, 'left', 'bold 11px sans-serif'); text(g, `— penalty ${L1 ? 'λ|w|' : '½λw²'}`, v.pad.l + 170, 12, C.orange, 'left', 'bold 11px sans-serif'); text(g, '— total (minimised)', v.pad.l + 280, 12, C.green, 'left', 'bold 11px sans-serif');
      text(g, 'weight w', v.pad.l + v.pw / 2, c.h - 3, C.muted, 'center');
      // probability view (each curve rescaled to peak 1)
      c = cs[1]; g = c.ctx; v = AI.view(c, -4, 4, 0, 1.2, { l: 44, r: 12, t: 12, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      const mn = tot(wd), cur2 = (f, col, lw, fill) => { g.beginPath(); for (let i = 0; i <= 240; i++) { const w = -4 + (8 * i) / 240; i ? g.lineTo(v.X(w), v.Y(f(w))) : g.moveTo(v.X(w), v.Y(f(w))); } g.strokeStyle = col; g.lineWidth = lw; g.stroke(); if (fill) { g.lineTo(v.X(4), v.Y(0)); g.lineTo(v.X(-4), v.Y(0)); g.closePath(); g.globalAlpha = 0.14; g.fillStyle = col; g.fill(); g.globalAlpha = 1; } };
      cur2((w) => Math.exp(-data(w)), C.blue, 2); cur2((w) => Math.exp(-pen(w)), C.orange, 2); cur2((w) => Math.exp(-(tot(w) - mn)), C.green, 3, true);
      text(g, 'likelihood', v.pad.l + 6, 12, C.blue, 'left', 'bold 11px sans-serif'); text(g, 'prior', v.pad.l + 76, 12, C.orange, 'left', 'bold 11px sans-serif'); text(g, 'posterior (peak = MAP = w*)', v.pad.l + 120, 12, C.green, 'left', 'bold 11px sans-serif');
            const shrunk = w0 ? 100 * (1 - ws / w0) : 0;
      q(root, '#cap').innerHTML = '<span style="color:var(--muted)">(Bottom panel: every curve rescaled to peak 1.)</span><br>' + (L1 ? `L1: w* = sign(w₀)·max(|w₀| − λ/n, 0). The threshold is λ/n = ${AI.fmt(lam / n, 3)}${Math.abs(w0) <= lam / n && lam > 0 ? ' — larger than |w₀|, so the weight is snapped to <b>exactly 0</b> (sparsity)' : ''}.`
        : `L2: w* = n·w₀/(n+λ) = w₀ × ${AI.fmt(n / (n + lam), 3)}. Shrinks by a constant factor — never exactly 0. More data (bigger n) → weaker pull.`);
      q(root, '#out').innerHTML = `w₀ (no penalty)  <b class="b">${AI.fmt(w0, 3)}</b>\nw* (penalised)   <b class="g">${AI.fmt(ws, 3)}</b>\nshrinkage        <span class="o">${AI.fmt(shrunk, 1)}%</span>\nprior: ${L1 ? 'Laplace, scale b = σ²/λ' : 'Gaussian, variance τ² = σ²/λ'}\n<span class="m">λ = ${L1 ? 'σ²/b' : 'σ²/τ²'}: bigger λ = stronger prior belief in 0</span>`;
    }, { fps: 40 });
  };

  /* =====================================================================================
     5. biasVar — ghost curves, bias², variance, and train/test error vs degree
     ===================================================================================== */
  AI.labs.biasVar = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bias–variance lab: the same model, many datasets', 'each ghost curve is a fit to one fresh noisy dataset; their spread is the variance',
      cvs(300) + cvs(210) + capEl(3),
      sl('deg', 'Polynomial degree', 0, 9, 1, 3) + sl('sg', 'Noise σ', 0.05, 0.8, 0.05, 0.3) + sl('n', 'Points per dataset', 8, 40, 1, 14) + sl('sp', 'Speed (datasets per second)', 0.25, 4, 0.25, 0.75) + btns() + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e)), st = {}; playCtl(root, st);
    const G = 81, gx = new Float64Array(G); for (let i = 0; i < G; i++) gx[i] = -1 + (2 * i) / (G - 1);
    const ftrue = gx.map(trueF);
    let ghosts = [], key = '', table = null, fade = 1, curPts = null, avg = new Float64Array(G), bias2 = 0, varr = 0, adv = 0, trainEmp = 0;
    const tick = AI.ticker(), MAXG = 50, rngLive = Math.random;
    const genData = (n, s, r) => { const xs = new Float64Array(n), ys = new Float64Array(n); for (let i = 0; i < n; i++) { xs[i] = -1 + 2 * r(); ys[i] = trueF(xs[i]) + s * AI.randn(r); } return { xs, ys }; };
    const buildTable = () => {
      const n = nv(root, 'n'), s = nv(root, 'sg'), r = AI.rng(4242), R = 60; table = { b2: [], vr: [], tr: [], te: [] };
      for (let d = 0; d <= 9; d++) {
        const preds = [], sets = []; let trn = 0;
        for (let k = 0; k < R; k++) {
          const ds = genData(n, s, r), w = polyfit(ds.xs, ds.ys, d), p = new Float64Array(G); for (let i = 0; i < G; i++) p[i] = polyval(w, gx[i]); preds.push(p);
          let e = 0; for (let i = 0; i < n; i++) e += (polyval(w, ds.xs[i]) - ds.ys[i]) ** 2; trn += e / n;
        }
        let b2 = 0, vr = 0; for (let i = 0; i < G; i++) { let m = 0; for (const p of preds) m += p[i]; m /= R; let vv = 0; for (const p of preds) vv += (p[i] - m) ** 2; vr += vv / R; b2 += (m - ftrue[i]) ** 2; }
        b2 /= G; vr /= G; table.b2.push(b2); table.vr.push(vr); table.tr.push(trn / R); table.te.push(s * s + b2 + vr);
      }
    };
    const recomputeStats = () => {
      const m = ghosts.length; avg.fill(0); bias2 = 0; varr = 0; if (!m) return;
      for (const gh of ghosts) for (let i = 0; i < G; i++) avg[i] += gh.p[i] / m;
      for (let i = 0; i < G; i++) { bias2 += (avg[i] - ftrue[i]) ** 2; let vv = 0; for (const gh of ghosts) vv += (gh.p[i] - avg[i]) ** 2; varr += vv / m; }
      bias2 /= G; varr /= G;
    };
    const addDataset = () => {
      const n = nv(root, 'n'), s = nv(root, 'sg'), d = nv(root, 'deg'), ds = genData(n, s, rngLive), w = polyfit(ds.xs, ds.ys, d), p = new Float64Array(G);
      for (let i = 0; i < G; i++) p[i] = polyval(w, gx[i]);
      ghosts.push({ p }); if (ghosts.length > MAXG) ghosts.shift(); curPts = ds; fade = 0; recomputeStats();
    };
    const reset = () => { ghosts = []; curPts = null; recomputeStats(); };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { st.setPlay(false); addDataset(); };
    AI.loop(root, (dt) => {
      sync(root);
      const k2 = [nv(root, 'deg'), nv(root, 'sg'), nv(root, 'n')].join('|');
      if (k2 !== key) { const old = key.split('|'); key = k2; if (!table || old[1] !== String(nv(root, 'sg')) || old[2] !== String(nv(root, 'n'))) buildTable(); reset(); }
      if (st.playing) { const k = tick(dt, nv(root, 'sp')); for (let i = 0; i < k; i++) addDataset(); }
      fade = Math.min(1, fade + dt * 1.6);
      const C = AI.col(), deg = nv(root, 'deg'), sg = nv(root, 'sg');
      let c = cs[0], g = c.ctx, v = AI.view(c, -1, 1, -2, 2, { l: 36, r: 12, t: 12, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const path = (arr) => { g.beginPath(); for (let i = 0; i < G; i++) { const x = v.X(gx[i]), y = v.Y(arr[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); } };
      const m = ghosts.length;
      ghosts.forEach((gh, j) => { path(gh.p); g.strokeStyle = C.blue; g.globalAlpha = j === m - 1 ? 0.25 + 0.6 * fade * 0 + 0.35 * (1 - fade) + 0.2 : 0.14; g.lineWidth = j === m - 1 ? 2 : 1.2; g.stroke(); });
      g.globalAlpha = 1;
      if (curPts) { g.fillStyle = C.orange; g.globalAlpha = Math.min(1, 0.3 + 0.7 * fade); for (let i = 0; i < curPts.xs.length; i++) { g.beginPath(); g.arc(v.X(curPts.xs[i]), v.Y(curPts.ys[i]), 3.2, 0, 7); g.fill(); } g.globalAlpha = 1; }
      path(ftrue); g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
      if (m >= 2) { path(avg); g.strokeStyle = C.green; g.lineWidth = 3.2; g.stroke(); }
      g.restore();
      text(g, '- - true f(x)=sin 3x', v.pad.l + 6, 12, C.fg, 'left', 'bold 11px sans-serif'); text(g, '— average fit', v.pad.l + 140, 12, C.green, 'left', 'bold 11px sans-serif'); text(g, '— each ghost = one dataset’s fit', v.pad.l + 230, 12, C.blue, 'left', 'bold 11px sans-serif'); text(g, '● this dataset', v.pad.l + 430, 12, C.orange, 'left', 'bold 11px sans-serif');
      // bottom: error vs degree (log scale)
      c = cs[1]; g = c.ctx; const lo = -3, hi = 2, pad = { l: 44, r: 12, t: 12, b: 36 }, pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b;
      const X = (d) => pad.l + (d / 9) * pw, Y = (val) => pad.t + (1 - (Math.log10(Math.max(val, 1e-9)) - lo) / (hi - lo)) * ph;
      g.clearRect(0, 0, c.w, c.h); g.font = '11px ui-monospace,monospace';
      for (let e = lo; e <= hi; e++) { g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(pad.l, Y(10 ** e)); g.lineTo(pad.l + pw, Y(10 ** e)); g.stroke(); text(g, '10^' + e, pad.l - 5, Y(10 ** e) + 4, C.muted, 'right', '11px ui-monospace,monospace'); }
      for (let d = 0; d <= 9; d++) text(g, d, X(d), c.h - 20, C.muted, 'center', '11px ui-monospace,monospace');
      text(g, 'degree (model flexibility) →  ·  log scale, 60 datasets per degree', pad.l + pw / 2, c.h - 5, C.muted, 'center');
      const line = (arr, col, dash, lw) => { g.beginPath(); arr.forEach((val, d) => (d ? g.lineTo(X(d), Y(val)) : g.moveTo(X(d), Y(val)))); g.strokeStyle = col; g.lineWidth = lw || 2.2; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); };
      g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();
      line(table.b2, C.red, [5, 4], 1.8); line(table.vr, C.purple, [5, 4], 1.8); line(table.tr, C.blue, null, 2.6); line(table.te, C.orange, null, 2.6);
      g.strokeStyle = C.muted; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(pad.l, Y(sg * sg)); g.lineTo(pad.l + pw, Y(sg * sg)); g.stroke(); g.setLineDash([]);
      g.restore();
      text(g, 'σ² (irreducible)', pad.l + pw - 4, Y(sg * sg) - 4, C.muted, 'right', '11px sans-serif');
      g.strokeStyle = C.fg; g.globalAlpha = 0.5; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(deg), pad.t); g.lineTo(X(deg), pad.t + ph); g.stroke(); g.globalAlpha = 1;
      text(g, '— train error', pad.l + 6, 11, C.blue, 'left', 'bold 11px sans-serif'); text(g, '— test error', pad.l + 86, 11, C.orange, 'left', 'bold 11px sans-serif'); text(g, '-- bias²', pad.l + 160, 11, C.red, 'left', 'bold 11px sans-serif'); text(g, '-- variance', pad.l + 215, 11, C.purple, 'left', 'bold 11px sans-serif');
      const bar = (val, mx) => '█'.repeat(Math.max(0, Math.min(24, Math.round((24 * val) / mx)))), mxb = Math.max(0.05, bias2 + varr + sg * sg) ;
      q(root, '#cap').innerHTML = m < 2 ? 'Datasets arrive one at a time. Each one is a fresh noisy sample of the same curve; we fit the same model to it and draw the fit as a faint blue curve.'
        : `${m} datasets so far. The blue curves disagree with each other by <b>variance</b> = ${AI.fmt(varr, 3)}; their average (green) misses the truth by <b>bias²</b> = ${AI.fmt(bias2, 3)}. Low degree: bias dominates. High degree: variance dominates.`;
      q(root, '#out').innerHTML = `degree ${deg}   (shown over ${m} live datasets)\nbias²     <b class="o">${AI.fmt(bias2, 3)}</b> <span class="r">${bar(bias2, mxb)}</span>\nvariance  <b class="o">${AI.fmt(varr, 3)}</b> <span class="b">${bar(varr, mxb)}</span>\nnoise σ²  <b class="o">${AI.fmt(sg * sg, 3)}</b> <span class="m">${bar(sg * sg, mxb)}</span>\nsum = expected test error ≈ <b>${AI.fmt(bias2 + varr + sg * sg, 3)}</b>\nfrom the 60-dataset table: train <b class="b">${AI.fmt(table.tr[deg], 3)}</b>  test <b class="o">${AI.fmt(table.te[deg], 3)}</b>`;
    }, { fps: 30 });
  };

  /* =====================================================================================
     6. ciCoverage — confidence intervals one by one
     ===================================================================================== */
  const TDF = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 60, 100, 1000];
  const TT = { 0.68: [1.1344, 1.1037, 1.084, 1.0703, 1.0602, 1.0525, 1.0464, 1.0374, 1.0285, 1.0198, 1.0146, 1.0112, 1.007, 1.0028, 0.9994, 0.995], 0.8: [1.5332, 1.4759, 1.4398, 1.4149, 1.3968, 1.383, 1.3722, 1.3562, 1.3406, 1.3253, 1.3163, 1.3104, 1.3031, 1.2958, 1.2901, 1.2824], 0.9: [2.1318, 2.015, 1.9432, 1.8946, 1.8595, 1.8331, 1.8125, 1.7823, 1.7531, 1.7247, 1.7081, 1.6973, 1.6839, 1.6706, 1.6602, 1.6464], 0.95: [2.7764, 2.5706, 2.4469, 2.3646, 2.306, 2.2622, 2.2281, 2.1788, 2.1314, 2.086, 2.0595, 2.0423, 2.0211, 2.0003, 1.984, 1.9623], 0.99: [4.6041, 4.0321, 3.7074, 3.4995, 3.3554, 3.2498, 3.1693, 3.0545, 2.9467, 2.8453, 2.7874, 2.75, 2.7045, 2.6603, 2.6259, 2.5808] };
  const ZC = { 0.68: 0.9945, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96, 0.99: 2.5758 };
  function tcrit(level, df) { const a = TT[level]; if (df >= 1000) return a[a.length - 1]; for (let i = 0; i < TDF.length - 1; i++) if (df <= TDF[i + 1]) { const f = (1 / df - 1 / TDF[i]) / (1 / TDF[i + 1] - 1 / TDF[i]); return a[i] + (a[i + 1] - a[i]) * f; } return a[a.length - 1]; }

  AI.labs.ciCoverage = function (root) {
    const { view, ctl } = AI.labShell(root, 'Confidence-interval lab: what “95%” really means', 'each row is a fresh experiment; the dashed line is the true mean, which you normally never see',
      cvs(360) + capEl(3.2),
      sl('n', 'Sample size n', 5, 100, 1, 10) + `<label>Confidence level<select id="lv"><option value="0.68">68%</option><option value="0.8">80%</option><option value="0.9">90%</option><option value="0.95" selected>95%</option><option value="0.99">99%</option></select></label>` +
      `<label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="useT" checked> use Student-t critical value (else z)</label>` + sl('sp', 'Speed (intervals per second)', 0.25, 8, 0.25, 1.5) + btns(`<button class="btn" id="ff">+100 at once</button>`) + `<div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), st = {}; playCtl(root, st);
    const MU = 50, SD = 10, ROWS = 36; let list = [], total = 0, cov = 0, shift = 0, key = '';
    const tick = AI.ticker();
    const draw1 = () => {
      const n = nv(root, 'n'), lv = +q(root, '#lv').value; let s = 0, s2 = 0; const xs = [];
      for (let i = 0; i < n; i++) { const x = MU + SD * AI.randn(); xs.push(x); s += x; }
      const m = s / n; for (const x of xs) s2 += (x - m) ** 2; const sd = Math.sqrt(s2 / (n - 1)), crit = q(root, '#useT').checked ? tcrit(lv, n - 1) : ZC[lv], hw = (crit * sd) / Math.sqrt(n);
      const ok = m - hw <= MU && MU <= m + hw; total++; if (ok) cov++; list.unshift({ m, lo: m - hw, hi: m + hw, ok, sd, k: total }); if (list.length > ROWS) list.pop(); shift = 1;
    };
    const reset = () => { list = []; total = cov = 0; shift = 0; };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { st.setPlay(false); draw1(); };
    q(root, '#ff').onclick = () => { for (let i = 0; i < 100; i++) draw1(); shift = 0; };
    AI.loop(root, (dt) => {
      sync(root);
      const k2 = [nv(root, 'n'), q(root, '#lv').value, q(root, '#useT').checked].join('|'); if (k2 !== key) { key = k2; reset(); }
      if (st.playing) { const k = tick(dt, nv(root, 'sp')); for (let i = 0; i < k; i++) draw1(); }
      shift = Math.max(0, shift - dt * 3.5);
      const C = AI.col(), g = c.ctx, n = nv(root, 'n'), lv = +q(root, '#lv').value, half = 6.5 * SD / Math.sqrt(Math.max(n, 10));
      const v = AI.view(c, MU - half * 1.1, MU + half * 1.1, 0, 1, { l: 12, r: 12, t: 26, b: 24 }), rowH = (v.ph - 6) / ROWS;
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, { ...v, ymin: 0, ymax: 1 }, { noY: true, noGrid: false });
      g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v.X(MU), v.pad.t - 6); g.lineTo(v.X(MU), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      text(g, `true mean μ = ${MU}`, v.X(MU), 12, C.fg, 'center', 'bold 12px sans-serif');
      g.save(); g.beginPath(); g.rect(0, v.pad.t, c.w, v.ph); g.clip();
      list.forEach((it, i) => {
        const y = v.pad.t + (i + shift) * rowH + rowH / 2 + 2, col = it.ok ? C.green : C.red; g.globalAlpha = i === 0 ? 1 - shift : 1;
        g.strokeStyle = col; g.lineWidth = Math.max(2, rowH * 0.55); g.lineCap = 'round'; g.beginPath(); g.moveTo(v.X(it.lo), y); g.lineTo(v.X(it.hi), y); g.stroke();
        g.fillStyle = C.fg; g.beginPath(); g.arc(v.X(it.m), y, Math.max(1.6, rowH * 0.22), 0, 7); g.fill();
      });
      g.restore(); g.globalAlpha = 1;
      const f = total ? cov / total : NaN, a = list[0];
      q(root, '#cap').innerHTML = !a ? 'Press play: each experiment draws n numbers from a population with true mean 50, then builds an interval x̄ ± critical·s/√n around its own sample mean.'
        : `Experiment ${a.k}: x̄ = ${AI.fmt(a.m, 2)}, s = ${AI.fmt(a.sd, 2)} → interval [${AI.fmt(a.lo, 2)}, ${AI.fmt(a.hi, 2)}] → it ${a.ok ? '<b style="color:var(--green)">covers</b>' : '<b style="color:var(--red)">misses</b>'} the true mean. (Dot = x̄; green = covers, red = misses.)`;
      q(root, '#out').innerHTML = `intervals so far  <b>${total}</b>\ncovered the truth <b class="g">${cov}</b>   missed <b class="r">${total - cov}</b>\ncoverage          <b class="o">${total ? AI.fmt(100 * f, 1) + '%' : '–'}</b>   (nom. ${AI.fmt(100 * lv, 0)}%)\ncritical value    ${AI.fmt(q(root, '#useT').checked ? tcrit(lv, n - 1) : ZC[lv], 3)}\nstandard error    σ/√n = ${AI.fmt(SD / Math.sqrt(n), 3)}\n<span class="m">The 95% is about the <i>procedure</i>, not this one interval.</span>`;
    }, { fps: 40 });
  };

  /* =====================================================================================
     7. pTest — null distribution, observed statistic, p-value, and p-value histogram
     ===================================================================================== */
  AI.labs.pTest = function (root) {
    const { view, ctl } = AI.labShell(root, 'Hypothesis-test lab: p-values, power and false alarms', 'comparing two models on n test examples (paired differences, known spread σ = 10 points)',
      cvs(250) + cvs(150) + capEl(3),
      sl('d', 'TRUE difference in accuracy (points)', 0, 5, 0.25, 0, '', ' pts') + sl('n', 'Test examples n', 10, 400, 5, 100) + `<label>Significance level α<select id="al"><option value="0.1">0.10</option><option value="0.05" selected>0.05</option><option value="0.01">0.01</option></select></label>` +
      sl('sp', 'Speed (experiments per second)', 0.25, 6, 0.25, 0.75) + btns() + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e)), st = {}; playCtl(root, st);
    const SD = 10, ZC = { 0.1: 1.6449, 0.05: 1.96, 0.01: 2.5758 }, NB = 20;
    let obs = null, hist = new Array(NB).fill(0), total = 0, rej = 0, drop = 1, key = '', info = '';
    const tick = AI.ticker();
    const run = () => {
      const d = nv(root, 'd'), n = nv(root, 'n'), se = SD / Math.sqrt(n), al = +q(root, '#al').value, x = d + se * AI.randn(), z = x / se, p = 2 * (1 - Phi(Math.abs(z)));
      obs = { x, z, p }; drop = 0; total++; hist[Math.min(NB - 1, Math.floor(p * NB))]++; if (p < al) rej++;
      info = `Experiment ${total}: observed difference ${AI.fmt(x, 2)} pts → z = ${AI.fmt(z, 2)} → p = ${AI.fmt(p, 3)} → ${p < al ? '<b style="color:var(--red)">“significant”</b>' : 'not significant'} at α = ${al}.`;
    };
    const reset = () => { obs = null; hist.fill(0); total = rej = 0; info = ''; drop = 1; };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { st.setPlay(false); run(); };
    AI.loop(root, (dt) => {
      sync(root);
      const k2 = [nv(root, 'd'), nv(root, 'n'), q(root, '#al').value].join('|'); if (k2 !== key) { key = k2; reset(); }
      if (st.playing) { const k = tick(dt, nv(root, 'sp')); for (let i = 0; i < k; i++) run(); }
      drop = Math.min(1, drop + dt * 1.8);
      const C = AI.col(), d = nv(root, 'd'), n = nv(root, 'n'), se = SD / Math.sqrt(n), al = +q(root, '#al').value, zc = ZC[al];
      let c = cs[0], g = c.ctx; const R = Math.max(4.2 * se, d + 3.6 * se), v = AI.view(c, -R, R, 0, npdf(0, 0, se) * 1.25, { l: 14, r: 14, t: 16, b: 26 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v, { noY: true });
      const area = (f, a, b, col, al_) => { g.beginPath(); g.moveTo(v.X(a), v.Y(0)); for (let i = 0; i <= 60; i++) { const x = a + ((b - a) * i) / 60; g.lineTo(v.X(x), v.Y(f(x))); } g.lineTo(v.X(b), v.Y(0)); g.closePath(); g.globalAlpha = al_; g.fillStyle = col; g.fill(); g.globalAlpha = 1; };
      const f0 = (x) => npdf(x, 0, se), f1 = (x) => npdf(x, d, se);
      area(f0, -R, -zc * se, C.red, 0.35); area(f0, zc * se, R, C.red, 0.35);
      if (obs && drop > 0.6) { const ax = Math.abs(obs.x); area(f0, -R, -Math.min(ax, R), C.orange, 0.55); area(f0, Math.min(ax, R), R, C.orange, 0.55); }
      const curve = (f, col, dash, lw) => { g.beginPath(); for (let i = 0; i <= 200; i++) { const x = -R + (2 * R * i) / 200; i ? g.lineTo(v.X(x), v.Y(f(x))) : g.moveTo(v.X(x), v.Y(f(x))); } g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); };
      if (d > 0) curve(f1, C.green, [6, 4], 2); curve(f0, C.blue, null, 2.6);
      if (obs) { const y0 = v.pad.t, y1 = v.Y(f0(obs.x)), y = AI.lerp(y0, y1, AI.smooth(drop)); g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(obs.x), v.Y(0)); g.lineTo(v.X(obs.x), y); g.stroke(); g.fillStyle = C.fg; g.beginPath(); g.arc(v.X(obs.x), y, 6, 0, 7); g.fill(); }
      text(g, '— null world: no real difference', v.pad.l + 2, 11, C.blue, 'left', 'bold 11px sans-serif'); if (d > 0) text(g, '-- the real world', v.pad.l + 200, 11, C.green, 'left', 'bold 11px sans-serif');
      text(g, 'red = rejection region · orange = p-value tails · ● = observed', c.w / 2, c.h - 3, C.muted, 'center', '11px sans-serif');
      // p-value histogram
      c = cs[1]; g = c.ctx; const v2 = AI.view(c, 0, 1, 0, Math.max(5, ...hist) * 1.2, { l: 34, r: 14, t: 16, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v2, { noGrid: false }); const bw = v2.pw / NB;
      hist.forEach((cnt, i) => { const hgt = (cnt / (v2.ymax)) * v2.ph; g.fillStyle = (i + 1) / NB <= al + 1e-9 ? C.red : C.blue; g.globalAlpha = 0.85; g.fillRect(v2.pad.l + i * bw + 1, v2.pad.t + v2.ph - hgt, bw - 2, hgt); g.globalAlpha = 1; });
      g.strokeStyle = C.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v2.X(al), v2.pad.t); g.lineTo(v2.X(al), v2.pad.t + v2.ph); g.stroke(); g.setLineDash([]);
      text(g, `α = ${al}`, v2.X(al) + 4, v2.pad.t + 11, C.fg, 'left', '11px sans-serif'); text(g, 'histogram of the p-values from every experiment so far', v2.pad.l + v2.pw, 11, C.muted, 'right', '11px sans-serif'); text(g, 'p-value', v2.pad.l + v2.pw / 2, c.h - 3, C.muted, 'center');
      const zcc = zc, power = 1 - Phi(zcc - d / se) + Phi(-zcc - d / se);
      q(root, '#cap').innerHTML = info || 'Each experiment: both models are tested on n new examples; we record the observed accuracy difference. Press play, then try setting the true difference to 0.';
      q(root, '#out').innerHTML = `standard error σ/√n   <b>${AI.fmt(se, 3)}</b> pts\ncritical |difference| <b class="o">${AI.fmt(zc * se, 3)}</b> pts\nexperiments run       <b>${total}</b>\n“significant” so far  <b class="r">${rej}</b> (${total ? AI.fmt(100 * rej / total, 1) : '–'}%)\nTheory: P(significant)= <b class="g">${AI.fmt(100 * power, 1)}%</b> ${d === 0 ? '<span class="m">(= α: false alarms)</span>' : '<span class="m">(= power)</span>'}`;
    }, { fps: 40 });
  };

  /* =====================================================================================
     8. kfold — k-fold cross-validation diagram, with a real model
     ===================================================================================== */
  AI.labs.kfold = function (root) {
    const { view, ctl } = AI.labShell(root, 'Cross-validation lab: every example takes a turn as the test', 'a polynomial is really fitted on each fold’s training part and scored on its held-out part',
      cvs(250) + cvs(210) + capEl(3),
      sl('k', 'Number of folds k (20 = leave-one-out)', 2, 20, 1, 5) + sl('deg', 'Model: polynomial degree', 1, 9, 1, 3) + sl('sp', 'Speed', 0.25, 3, 0.25, 1) + btns(`<button class="btn" id="newd">New data</button>`) + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e)), st = {}; playCtl(root, st);
    const N = 20; let seed = 77, xs, ys, perm, fold = 0, tm = 0, key = '', fits = [], scores = [], summaryT = 0, cur = null;
    const gen = () => { const r = AI.rng(seed); xs = new Float64Array(N); ys = new Float64Array(N); for (let i = 0; i < N; i++) { xs[i] = -1 + 2 * r(); ys[i] = trueF(xs[i]) + 0.3 * AI.randn(r); } perm = [...Array(N).keys()]; for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; } };
    const foldOf = (i, k) => perm[i] % k;
    const build = () => {
      const k = nv(root, 'k'), d = nv(root, 'deg'); fits = []; scores = [];
      for (let f = 0; f < k; f++) {
        const tx = [], ty = [], vx = [], vy = []; for (let i = 0; i < N; i++) (foldOf(i, k) === f ? (vx.push(xs[i]), vy.push(ys[i])) : (tx.push(xs[i]), ty.push(ys[i])));
        const w = polyfit(tx, ty, d); fits.push(w); let e = 0; for (let i = 0; i < vx.length; i++) e += (polyval(w, vx[i]) - vy[i]) ** 2; scores.push(e / vx.length);
      }
      const all = polyfit(Array.from(xs), Array.from(ys), d); let te = 0; for (let i = 0; i < N; i++) te += (polyval(all, xs[i]) - ys[i]) ** 2; build.train = te / N;
      fold = 0; tm = 0; summaryT = 0; cur = Float64Array.from(fits[0]); build.prev = Float64Array.from(fits[0]);
    };
    gen(); q(root, '#newd').onclick = () => { seed = (seed * 7 + 13) % 100003; gen(); build(); };
    q(root, '#reset').onclick = build; q(root, '#step').onclick = () => { st.setPlay(false); const k = nv(root, 'k'); build.prev = Float64Array.from(cur); fold = (fold + 1) % k; tm = 0; summaryT = 0; };
    const DUR = 3.2;
    AI.loop(root, (dt) => {
      sync(root);
      const k2 = [nv(root, 'k'), nv(root, 'deg')].join('|'); if (k2 !== key) { key = k2; build(); }
      const k = nv(root, 'k'), sp = nv(root, 'sp'), d = nv(root, 'deg');
      if (st.playing) {
        if (summaryT > 0) { summaryT -= dt * sp; if (summaryT <= 0) { fold = 0; tm = 0; build.prev = Float64Array.from(cur); } }
        else { tm += dt * sp; if (tm > DUR) { if (fold === k - 1) { summaryT = 3; } else { build.prev = Float64Array.from(cur); fold++; tm = 0; } } }
      }
      const summary = summaryT > 0, shown = summary ? k : fold + 1;
      // morph coefficients toward the active fold's fit
      const target = fits[Math.min(fold, k - 1)], mt = AI.smooth((tm - 0.6) / 0.8);
      for (let i = 0; i < cur.length; i++) cur[i] = AI.lerp(build.prev[i] || 0, target[i], mt);
      const C = AI.col();
      let c = cs[0], g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const L = 56, R = c.w * 0.34, cw = (c.w - L - R) / N, rowH = Math.min(22, (c.h - 44) / k), top = 28;
      text(g, 'example # (shuffled into folds)', L, 14, C.muted, 'left', '11px sans-serif'); text(g, 'held-out error (MSE)', c.w - R + 8, 14, C.muted, 'left', '11px sans-serif');
      const mxS = Math.max(...scores) * 1.1 || 1;
      for (let f = 0; f < k; f++) {
        const y = top + f * rowH, active = !summary && f === fold, done = f < shown - (active ? 1 : 0) || summary, reached = f <= fold || summary;
        g.globalAlpha = reached ? 1 : 0.3;
        text(g, 'fold ' + (f + 1), L - 8, y + rowH * 0.68, active ? C.fg : C.muted, 'right', active ? 'bold 11px sans-serif' : '11px sans-serif');
        for (let i = 0; i < N; i++) {
          const isV = foldOf(i, k) === f; g.fillStyle = isV ? C.orange : C.blue; g.globalAlpha = (reached ? 1 : 0.3) * (isV ? 1 : 0.55);
          g.fillRect(L + i * cw + 1, y + 2, cw - 2, rowH - 4);
        }
        g.globalAlpha = 1;
        if (active) { g.strokeStyle = C.fg; g.lineWidth = 2; g.strokeRect(L - 1, y + 1, cw * N + 2, rowH - 2); }
        const grow = active ? AI.smooth((tm - 1.4) / 0.9) : done ? 1 : 0;
        if (grow > 0) { g.fillStyle = C.green; g.fillRect(c.w - R + 8, y + 3, ((R - 70) * scores[f] / mxS) * grow, rowH - 6); text(g, AI.fmt(scores[f], 3), c.w - R + 12 + ((R - 70) * scores[f] / mxS) * grow, y + rowH * 0.7, C.fg, 'left', '11px sans-serif'); }
      }
      // plot
      c = cs[1]; g = c.ctx; const v = AI.view(c, -1, 1, -2, 2, { l: 36, r: 12, t: 12, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.beginPath(); for (let i = 0; i <= 120; i++) { const x = -1 + (2 * i) / 120; i ? g.lineTo(v.X(x), v.Y(trueF(x))) : g.moveTo(v.X(x), v.Y(trueF(x))); } g.strokeStyle = C.muted; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
      if (!summary) { g.beginPath(); for (let i = 0; i <= 120; i++) { const x = -1 + (2 * i) / 120, y = polyval(cur, x); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.strokeStyle = C.green; g.lineWidth = 3; g.stroke(); }
      g.restore();
      for (let i = 0; i < N; i++) {
        const isV = !summary && foldOf(i, k) === fold; g.fillStyle = summary ? C.fg : isV ? C.orange : C.blue; g.beginPath(); g.arc(v.X(xs[i]), v.Y(ys[i]), isV ? 5 : 3.5, 0, 7); g.fill();
        if (isV && tm > 1.4) { const pe = polyval(cur, xs[i]); g.strokeStyle = C.orange; g.lineWidth = 2; g.globalAlpha = AI.smooth((tm - 1.4) / 0.6); g.beginPath(); g.moveTo(v.X(xs[i]), v.Y(ys[i])); g.lineTo(v.X(xs[i]), v.Y(pe)); g.stroke(); g.globalAlpha = 1; }
      }
      text(g, summary ? 'all folds done: average the k held-out errors' : '● training points   ● held-out points   — model trained only on the blue ones', v.pad.l + 6, 11, C.muted, 'left', '11px sans-serif');
      const mean = scores.reduce((a, b) => a + b, 0) / k, sdv = k > 1 ? Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (k - 1)) : 0;
      q(root, '#cap').innerHTML = summary ? `Summary: the k = ${k} held-out errors average to <b>${AI.fmt(mean, 3)}</b> (spread ${AI.fmt(sdv, 3)}). Every example was held out exactly once and used for training k−1 times.`
        : tm < 0.6 ? `① Hold out fold ${fold + 1} (orange) — the model is not allowed to see it.` : tm < 1.4 ? `② Train on the other ${k - 1} folds (blue): the fitted curve moves.` : tm < 2.3 ? '③ Predict the held-out points and measure the error (orange segments).' : `④ Record fold ${fold + 1}’s score = ${AI.fmt(scores[fold], 3)}, then move to the next fold.`;
      q(root, '#out').innerHTML = `degree ${d}, k = ${k}   (${N} examples)\nfold scores: ${scores.map((s) => AI.fmt(s, 2)).join('  ')}\nCV estimate  mean <b class="g">${AI.fmt(mean, 3)}</b>   sd ${AI.fmt(sdv, 3)}\ntrain MSE on all data  <b class="b">${AI.fmt(build.train, 3)}</b>\n<span class="m">train error is optimistic; CV is the honest estimate</span>`;
    }, { fps: 40 });
  };

  /* =====================================================================================
     9. bootstrap — resample with replacement, histogram of a statistic
     ===================================================================================== */
  AI.labs.bootstrap = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bootstrap lab: pretend your sample is the population', 'draw n values with replacement, compute the statistic, repeat — the histogram is its uncertainty',
      cvs(190) + cvs(190) + capEl(3),
      `<label>Statistic<select id="stat"><option value="mean">mean</option><option value="median">median</option><option value="sd">standard deviation</option></select></label>` + sl('sp', 'Speed (draws per second)', 1, 30, 1, 6) + btns(`<button class="btn" id="ff">+200 resamples</button>`) + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e)), st = {}; playCtl(root, st);
    const n = 15, r0 = AI.rng(99), data = Array.from({ length: n }, () => Math.round(10 * Math.exp(0.6 * AI.randn(r0)) * 10) / 10), NB = 36;
    const stat = (a, kind) => { const m = a.reduce((x, y) => x + y, 0) / a.length; if (kind === 'mean') return m; if (kind === 'sd') return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); const s = [...a].sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };
    let kind = 'mean', stats = [], counts = new Array(NB).fill(0), blo = 0, bhi = 1, cnt = new Array(n).fill(0), picks = 0, lastPick = -1, flash = 0, phase = 'pick', ph = 0, resStat = 0, key = '';
    const tick = AI.ticker(), dmin = Math.min(...data), dmax = Math.max(...data), obsStat = () => stat(data, kind);
    const oneRes = () => { const a = []; for (let i = 0; i < n; i++) a.push(data[Math.floor(Math.random() * n)]); return stat(a, kind); };
    const setup = () => {
      kind = q(root, '#stat').value; const pre = []; for (let i = 0; i < 1500; i++) pre.push(oneRes()); pre.sort((a, b) => a - b);
      blo = AI.quantile(pre, 0.003); bhi = AI.quantile(pre, 0.997); const pd = (bhi - blo) * 0.05; blo -= pd; bhi += pd;
      stats = []; counts.fill(0); cnt.fill(0); picks = 0; lastPick = -1; phase = 'pick';
    };
    const addStat = (s) => { stats.push(s); const b = Math.floor(((s - blo) / (bhi - blo)) * NB); if (b >= 0 && b < NB) counts[b]++; };
    q(root, '#reset').onclick = setup; q(root, '#ff').onclick = () => { for (let i = 0; i < 200; i++) addStat(oneRes()); };
    q(root, '#step').onclick = () => { st.setPlay(false); while (picks < n) { const i = Math.floor(Math.random() * n); cnt[i]++; picks++; lastPick = i; flash = 1; } resStat = stat(cnt.flatMap((c2, i) => Array(c2).fill(data[i])), kind); phase = 'drop'; ph = 0; };
    q(root, '#stat').onchange = setup;
    AI.loop(root, (dt) => {
      sync(root);
      if (st.playing) {
        if (phase === 'pick') { const k = tick(dt, nv(root, 'sp')); for (let j = 0; j < k && picks < n; j++) { const i = Math.floor(Math.random() * n); cnt[i]++; picks++; lastPick = i; flash = 1; } if (picks >= n) { resStat = stat(cnt.flatMap((c2, i) => Array(c2).fill(data[i])), kind); phase = 'drop'; ph = 0; } }
      }
      if (phase === 'drop' && (st.playing || ph > 0 || true)) { ph += dt * (st.playing ? 0.8 * Math.min(2, nv(root, 'sp') / 6) : 0.8); if (ph >= 1) { addStat(resStat); cnt.fill(0); picks = 0; lastPick = -1; phase = 'pick'; } }
      flash = Math.max(0, flash - dt * 2.5);
      const C = AI.col(), os = obsStat();
      let c = cs[0], g = c.ctx; const lo = dmin - 1, hi = dmax + 1, v = AI.view(c, lo, hi, 0, 1, { l: 12, r: 12, t: 12, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v, { noY: true, noGrid: true });
      const yO = 34, yB = c.h - 36;
      text(g, 'your original sample (n = 15)', v.pad.l, 14, C.blue, 'left', 'bold 11px sans-serif'); text(g, 'current resample: stacks show how many times each value was drawn', v.pad.l, 72, C.orange, 'left', 'bold 11px sans-serif');
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(v.pad.l, yO + 10); g.lineTo(v.pad.l + v.pw, yO + 10); g.stroke();
      data.forEach((x, i) => { g.fillStyle = C.blue; g.globalAlpha = i === lastPick ? 1 : 0.8; g.beginPath(); g.arc(v.X(x), yO, i === lastPick ? 5 + 3 * flash : 5, 0, 7); g.fill(); if (i === lastPick && flash > 0) { g.strokeStyle = C.orange; g.lineWidth = 2; g.globalAlpha = flash; g.beginPath(); g.moveTo(v.X(x), yO + 8); g.lineTo(v.X(x), yB - cnt[i] * 9); g.stroke(); } g.globalAlpha = 1; });
      data.forEach((x, i) => { for (let j = 0; j < cnt[i]; j++) { g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(x), yB - 4 - j * 9, 4.2, 0, 7); g.fill(); } });
      text(g, `drawn ${picks}/${n}`, v.pad.l + v.pw, 14, C.muted, 'right', '11px sans-serif');
      if (phase === 'drop') { g.strokeStyle = C.green; g.lineWidth = 3; g.beginPath(); g.moveTo(v.X(resStat), yB + 6); g.lineTo(v.X(resStat), yB - 14); g.stroke(); text(g, `${kind} = ${AI.fmt(resStat, 2)}`, v.X(resStat), yB - 18, C.green, 'center', 'bold 11px sans-serif'); }
      // histogram
      c = cs[1]; g = c.ctx; const mx = Math.max(4, ...counts) * 1.2, v2 = AI.view(c, blo, bhi, 0, mx, { l: 34, r: 12, t: 14, b: 24 }), bw = v2.pw / NB;
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v2);
      counts.forEach((k2, i) => { const hgt = (k2 / mx) * v2.ph; g.fillStyle = C.orange; g.globalAlpha = 0.85; g.fillRect(v2.pad.l + i * bw + 1, v2.pad.t + v2.ph - hgt, bw - 2, hgt); g.globalAlpha = 1; });
      if (phase === 'drop') { const sx = v2.X(resStat), y = AI.lerp(v2.pad.t, v2.Y(Math.max(0, (counts[Math.floor(((resStat - blo) / (bhi - blo)) * NB)] || 0) + 0.5)), AI.smooth(ph)); g.fillStyle = C.green; g.beginPath(); g.arc(sx, y, 5, 0, 7); g.fill(); }
      g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(v2.X(os), v2.pad.t); g.lineTo(v2.X(os), v2.pad.t + v2.ph); g.stroke(); text(g, `original ${kind} = ${AI.fmt(os, 2)}`, v2.X(os) + 5, v2.pad.t + 11, C.blue, 'left', 'bold 11px sans-serif');
      let lo95 = NaN, hi95 = NaN; const B = stats.length;
      if (B >= 20) { const s = [...stats].sort((a, b) => a - b); lo95 = AI.quantile(s, 0.025); hi95 = AI.quantile(s, 0.975); g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([5, 4]); [lo95, hi95].forEach((x) => { g.beginPath(); g.moveTo(v2.X(x), v2.pad.t + 16); g.lineTo(v2.X(x), v2.pad.t + v2.ph); g.stroke(); }); g.setLineDash([]); text(g, '95% interval', v2.X((lo95 + hi95) / 2), v2.pad.t + 26, C.green, 'center', 'bold 11px sans-serif'); }
      const m = B ? stats.reduce((a, b) => a + b, 0) / B : NaN, sdv = B > 1 ? Math.sqrt(stats.reduce((a, b) => a + (b - m) ** 2, 0) / (B - 1)) : NaN;
      q(root, '#cap').innerHTML = phase === 'pick' ? `Resample ${B + 1}: ${picks < n ? `pick a random value, <b>put it back</b>, pick again — draw ${picks + 1} of ${n}. Some values repeat, some never appear.` : ''}` : `Resample done: its ${kind} is ${AI.fmt(resStat, 2)} — drop it into the histogram.`;
      const sem = (() => { const m0 = data.reduce((a, b) => a + b, 0) / n; return Math.sqrt(data.reduce((a, b) => a + (b - m0) ** 2, 0) / (n - 1)) / Math.sqrt(n); })();
      q(root, '#out').innerHTML = `resamples B          <b>${B}</b>\noriginal ${kind}      <b class="b">${AI.fmt(os, 3)}</b>\nbootstrap SE (sd)     <b class="o">${AI.fmt(sdv, 3)}</b>\n95% percentile interval  <b class="g">${B >= 20 ? `[${AI.fmt(lo95, 2)}, ${AI.fmt(hi95, 2)}]` : '–'}</b>\n${kind === 'mean' ? `formula s/√n          ${AI.fmt(sem, 3)}  <span class="m">(should be close)</span>` : '<span class="m">no simple formula exists for this statistic — the bootstrap still works</span>'}`;
    }, { fps: 40 });
    setup();
  };

  /* =====================================================================================
     10. leakage — selecting features before splitting inflates the score
     ===================================================================================== */
  AI.labs.leakage = function (root) {
    const { view, ctl } = AI.labShell(root, 'Leakage lab: a perfect score on pure noise', 'labels are coin flips, features are noise — the true accuracy is 50%. Which pipeline “discovers” signal?',
      cvs(150) + cvs(200) + capEl(3),
      sl('n', 'Examples n', 30, 150, 3, 60) + sl('d', 'Noise features d', 50, 2000, 50, 1000) + sl('k', 'Features kept k', 1, 50, 1, 20) + sl('sp', 'Speed (trials per second)', 0.1, 2, 0.1, 0.5) + btns(`<button class="btn" id="ff">+30 trials</button>`) + `<div class="readout" id="out"></div>`);
    const cs = AI.$$('canvas', view).map((e) => AI.canvas(e)), st = {}; playCtl(root, st);
    const NBn = 21; let hA = new Array(NBn).fill(0), hB = new Array(NBn).fill(0), sA = 0, sB = 0, T = 0, stage = 4.99, last = null, key = '';
    const STG = 0.7;
    const trial = () => {
      const n = nv(root, 'n'), d = nv(root, 'd'), k = nv(root, 'k'), nt = Math.round(n / 3), X = [], y = [];
      for (let i = 0; i < n; i++) { y.push(Math.random() < 0.5 ? -1 : 1); const r = new Float64Array(d); for (let j = 0; j < d; j++) r[j] = AI.randn(); X.push(r); }
      const tr = [], te = []; for (let i = 0; i < n; i++) (i < n - nt ? tr : te).push(i);
      const sel = (idx) => {
        const m = idx.length; let my = 0; for (const i of idx) my += y[i]; my /= m; let vy = 0; for (const i of idx) vy += (y[i] - my) ** 2; const sc = new Float64Array(d);
        for (let j = 0; j < d; j++) { let mx = 0; for (const i of idx) mx += X[i][j]; mx /= m; let cxy = 0, vx = 0; for (const i of idx) { cxy += (X[i][j] - mx) * (y[i] - my); vx += (X[i][j] - mx) ** 2; } sc[j] = Math.abs(cxy) / Math.sqrt(vx * vy + 1e-12); }
        return [...Array(d).keys()].sort((a, b) => sc[b] - sc[a]).slice(0, k);
      };
      const acc = (f) => {
        const c1 = new Float64Array(f.length), c2 = new Float64Array(f.length); let n1 = 0, n2 = 0;
        for (const i of tr) { if (y[i] > 0) { n1++; f.forEach((j, t) => (c1[t] += X[i][j])); } else { n2++; f.forEach((j, t) => (c2[t] += X[i][j])); } }
        f.forEach((_, t) => { c1[t] /= n1 || 1; c2[t] /= n2 || 1; }); let ok = 0;
        for (const i of te) { let a = 0, b = 0; f.forEach((j, t) => { a += (X[i][j] - c1[t]) ** 2; b += (X[i][j] - c2[t]) ** 2; }); if ((a < b ? 1 : -1) === y[i]) ok++; }
        return ok / te.length;
      };
      return { a: acc(sel([...Array(n).keys()])), b: acc(sel(tr)) };
    };
    const rec = (r) => { const bi = (a) => Math.min(NBn - 1, Math.round(a * 20)); hA[bi(r.a)]++; hB[bi(r.b)]++; sA += r.a; sB += r.b; T++; };
    const reset = () => { hA.fill(0); hB.fill(0); sA = sB = T = 0; last = null; stage = 4.99; };
    q(root, '#reset').onclick = reset;
    q(root, '#ff').onclick = () => { for (let i = 0; i < 30; i++) rec(trial()); };
    q(root, '#step').onclick = () => { st.setPlay(false); if (stage >= 4) { last = trial(); stage = 0; } else stage = Math.floor(stage) + 1; if (stage >= 4) rec(last); };
    AI.loop(root, (dt) => {
      sync(root);
      const k2 = [nv(root, 'n'), nv(root, 'd'), nv(root, 'k')].join('|'); if (k2 !== key) { key = k2; reset(); }
      if (st.playing) {
        if (stage >= 4) { last = trial(); stage = 0; }
        else { stage += (dt * nv(root, 'sp')) / STG * 1.0; if (stage >= 4) { stage = 4; rec(last); } }
      }
      const C = AI.col();
      let c = cs[0], g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const rows = [{ name: 'A leaky', col: C.red, b: ['Select features on ALL data', 'Split train / test', 'Train', 'Test'], acc: last && last.a }, { name: 'B right', col: C.green, b: ['Split train / test', 'Select on TRAIN only', 'Train', 'Test'], acc: last && last.b }];
      const bx0 = 70, bw = (c.w - bx0 - 46) / 4 - 12;
      rows.forEach((row, r) => {
        const y = 14 + r * 66;
        text(g, row.name, 6, y + 26, row.col, 'left', 'bold 11px sans-serif');
        row.b.forEach((lab, i) => {
          const x = bx0 + i * (bw + 12), act = stage >= i && stage < i + 1 && stage < 4, done = stage >= i + 1 || (stage >= 4 && last);
          g.fillStyle = act ? row.col : done ? C.bg3 : C.bg2; g.globalAlpha = act ? 0.9 : 1; g.fillRect(x, y, bw, 44); g.globalAlpha = 1;
          g.strokeStyle = (act || done) && i === (r === 0 ? 0 : 1) ? C.orange : C.border; g.lineWidth = (act || done) && i === (r === 0 ? 0 : 1) ? 2.5 : 1; g.strokeRect(x, y, bw, 44);
          { const ws = lab.split(' '), ln = ['']; ws.forEach((w) => { if ((ln[ln.length - 1] + ' ' + w).trim().length > 11) ln.push(w); else ln[ln.length - 1] = (ln[ln.length - 1] + ' ' + w).trim(); }); ln.forEach((l2, j) => text(g, l2, x + bw / 2, y + 22 - (ln.length - 1) * 6 + j * 12, act ? '#fff' : C.fg, 'center', '10px sans-serif')); }
          if (i < 3) text(g, '→', x + bw + 6, y + 26, C.muted, 'center', '14px sans-serif');
        });
        if (stage >= 4 && last) text(g, AI.fmt(100 * row.acc, 0) + '%', c.w - 22, y + 28, row.col, 'center', 'bold 13px sans-serif');
      });
      // histogram
      c = cs[1]; g = c.ctx; const mx = Math.max(4, ...hA, ...hB) * 1.2, v = AI.view(c, 0, 1.05, 0, mx, { l: 34, r: 12, t: 14, b: 24 }), bw2 = v.pw / (NBn * 1.05) ;
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      const col = (h, color, off) => h.forEach((k3, i) => { const hgt = (k3 / mx) * v.ph; g.fillStyle = color; g.globalAlpha = 0.85; g.fillRect(v.X(i / 20) - bw2 / 2 + off, v.pad.t + v.ph - hgt, bw2 / 2 - 1, hgt); g.globalAlpha = 1; });
      col(hA, C.red, 0); col(hB, C.green, bw2 / 2);
      g.strokeStyle = C.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(0.5), v.pad.t); g.lineTo(v.X(0.5), v.pad.t + v.ph); g.stroke(); g.setLineDash([]); text(g, 'true accuracy 50%', v.X(0.5) + 4, v.pad.t + 11, C.fg, 'left', '11px sans-serif');
      text(g, 'held-out accuracy of each trial →', v.pad.l + v.pw, c.h - 3, C.muted, 'right', '11px sans-serif');
      const names = ['① a new random dataset: labels are coin flips', '② ', '③ ', '④ '];
      q(root, '#cap').innerHTML = stage >= 4 && last ? `Trial ${T}: A scored <b style="color:var(--red)">${AI.fmt(100 * last.a, 0)}%</b>, B scored <b style="color:var(--green)">${AI.fmt(100 * last.b, 0)}%</b> — on data with <b>no signal at all</b>.`
        : stage < 1 ? 'New random dataset generated (labels are coin flips). A picks its best-looking features using <b>every</b> label, test labels included; B splits first.' : stage < 2 ? 'A splits into train/test <i>after</i> selection; B now selects features using the training rows only.' : stage < 3 ? 'Both train a nearest-centroid classifier on the training rows using their k chosen features.' : 'Both are scored on the held-out rows. A’s features were chosen with a peek at those labels.';
      q(root, '#out').innerHTML = `trials              <b>${T}</b>\nA leaky   mean acc  <b class="r">${T ? AI.fmt(100 * sA / T, 1) + '%' : '–'}</b>\nB correct mean acc  <b class="g">${T ? AI.fmt(100 * sB / T, 1) + '%' : '–'}</b>\ntruth               50%\n<span class="m">A’s “skill” comes entirely from picking, among d noise features, the ones that happen to match the labels — including the test labels.</span>`;
    }, { fps: 40 });
  };
})();
