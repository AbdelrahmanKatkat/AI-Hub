/* Generative AI Module 8 labs: diffusion and flow matching.
   gen-df-forward, gen-df-loss, gen-df-ddpm, gen-df-fast, gen-df-langevin, gen-df-sde, gen-df-guidance,
   gen-df-latent, gen-df-xattn, gen-df-flow.
   Every lab is an illustrative toy on a 1-D Gaussian-mixture data distribution, where the optimal denoiser, the score and the
   flow-matching velocity are known in closed form, so arrows and trajectories are exact, not drawn by hand.
   Seeded random numbers (AI.rng), dt-based animation, slow defaults, speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };

  /* =====================================================================
     Shared numerics (exported as AI.genDF so they can be checked in node)
     Conventions. Diffusion: x_0 is DATA, x_T is noise, x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.
     Flow matching: x_0 is NOISE, x_1 is DATA, x_t = (1-t) x_0 + t x_1.
     ===================================================================== */
  const DF = {};
  DF.SQ2PI = Math.sqrt(2 * Math.PI);
  DF.npdf = (x, m, v) => Math.exp(-0.5 * (x - m) * (x - m) / v) / Math.sqrt(2 * Math.PI * v);
  /* beta_t for t = 1..T stored at index t-1 */
  DF.linearBetas = (T, b0, b1) => { const b = new Float64Array(T); for (let i = 0; i < T; i++) b[i] = T === 1 ? b0 : b0 + (b1 - b0) * i / (T - 1); return b; };
  DF.cosineBetas = (T) => {
    const s = 0.008, f = (u) => Math.cos(((u / T + s) / (1 + s)) * Math.PI / 2) ** 2, b = new Float64Array(T);
    for (let t = 1; t <= T; t++) b[t - 1] = Math.min(0.999, 1 - f(t) / f(t - 1));
    return b;
  };
  /* abar[0] = 1, abar[t] = prod_{i<=t} (1 - beta_i) */
  DF.abarOf = (betas) => { const a = new Float64Array(betas.length + 1); a[0] = 1; for (let i = 0; i < betas.length; i++) a[i + 1] = a[i] * (1 - betas[i]); return a; };
  /* data distribution: Gaussian mixture {w, m, s} (s = standard deviations) */
  DF.mixture = (sep, s) => ({ w: [0.5, 0.5], m: [-sep, sep], s: [s, s] });
  /* marginal of x_t when the signal scale is a = sqrt(abar) and the noise scale is b = sqrt(1-abar) */
  DF.comps = (M, a, b) => M.w.map((w, k) => ({ w, m: a * M.m[k], v: a * a * M.s[k] * M.s[k] + b * b }));
  DF.density = (x, M, a, b) => { let p = 0; for (let k = 0; k < M.w.length; k++) p += M.w[k] * DF.npdf(x, a * M.m[k], a * a * M.s[k] * M.s[k] + b * b); return p; };
  /* score of the marginal: d/dx log p_t(x) */
  DF.score = (x, M, a, b) => {
    const K = M.w.length; let mx = -Infinity; const lw = new Array(K), vv = new Array(K), mm = new Array(K);
    for (let k = 0; k < K; k++) { vv[k] = a * a * M.s[k] * M.s[k] + b * b; mm[k] = a * M.m[k]; lw[k] = Math.log(M.w[k]) - 0.5 * Math.log(vv[k]) - 0.5 * (x - mm[k]) * (x - mm[k]) / vv[k]; if (lw[k] > mx) mx = lw[k]; }
    let z = 0, sc = 0; for (let k = 0; k < K; k++) { lw[k] = Math.exp(lw[k] - mx); z += lw[k]; }
    for (let k = 0; k < K; k++) sc += (lw[k] / z) * (mm[k] - x) / vv[k];
    return sc;
  };
  /* optimal epsilon predictor: E[eps | x_t = x] = -sqrt(1-abar) * score */
  DF.epsStar = (x, M, a, b) => -b * DF.score(x, M, a, b);
  /* one reverse step from abar_t = at to abar_s = as (as >= at) given an epsilon estimate; eta = 1 is DDPM, 0 is DDIM */
  DF.sigma = (at, as, eta) => (as >= 1 ? 0 : eta * Math.sqrt((1 - as) / (1 - at)) * Math.sqrt(Math.max(0, 1 - at / as)));
  DF.x0hat = (x, at, e) => (x - Math.sqrt(1 - at) * e) / Math.sqrt(at);
  DF.stepTo = (x, at, as, e, eta, z) => {
    const sg = DF.sigma(at, as, eta) , x0 = DF.x0hat(x, at, e);
    return Math.sqrt(as) * x0 + Math.sqrt(Math.max(0, 1 - as - sg * sg)) * e + sg * z;
  };
  /* S respaced timesteps (descending) out of T: T*(S-i)/S rounded */
  DF.timesteps = (T, S) => { const r = []; for (let i = 0; i < S; i++) r.push(Math.max(1, Math.round(T * (S - i) / S))); return r; };
  /* abar sequence for S steps: returns array of length S+1, last entry = 1 (clean) */
  DF.abarSeq = (ab, T, S) => { const ts = DF.timesteps(T, S), r = ts.map((t) => ab[t]); r.push(1); return r; };
  /* Gaussian N(0,1) draws from a seed */
  DF.noise = (n, seed) => { const r = AI.rng(seed), o = new Float64Array(n); for (let i = 0; i < n; i++) o[i] = AI.randn(r); return o; };
  /* full run with the exact denoiser; returns the final samples. opts.zs(step)->Float64Array of fresh noises; opts.guide -> function (x, a, b) -> epsilon */
  DF.run = (x0s, seq, M, eta, rng, epsFn) => {
    const x = Float64Array.from(x0s), n = x.length;
    for (let i = 0; i + 1 < seq.length; i++) {
      const at = seq[i], as = seq[i + 1], a = Math.sqrt(at), b = Math.sqrt(1 - at);
      for (let j = 0; j < n; j++) { const e = epsFn ? epsFn(x[j], a, b) : DF.epsStar(x[j], M, a, b); x[j] = DF.stepTo(x[j], at, as, e, eta, eta > 0 ? AI.randn(rng) : 0); }
    }
    return x;
  };
  /* distance between a sample and the target mixture: mean absolute gap between sorted samples and target quantiles (1-D Wasserstein-1) */
  DF.mixCdf = (x, M) => { let c = 0; for (let k = 0; k < M.w.length; k++) c += M.w[k] * 0.5 * (1 + DF.erf((x - M.m[k]) / (M.s[k] * Math.SQRT2))); return c; };
  DF.erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  DF.mixQuantile = (p, M) => { let lo = -12, hi = 12; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (DF.mixCdf(m, M) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  DF.w1 = (xs, M) => {
    const s = Float64Array.from(xs).sort(), n = s.length; let d = 0;
    for (let i = 0; i < n; i++) d += Math.abs(s[i] - DF.mixQuantile((i + 0.5) / n, M));
    return d / n;
  };
  /* ---- flow matching: x_t = (1-t) x0 + t x1, x0 ~ N(0,1) noise, x1 ~ data mixture ---- */
  DF.flowV = (x, t, M) => {
    const K = M.w.length; let mx = -Infinity; const lw = new Array(K), V = new Array(K), vk = new Array(K);
    for (let k = 0; k < K; k++) {
      V[k] = (1 - t) * (1 - t) + t * t * M.s[k] * M.s[k];
      lw[k] = Math.log(M.w[k]) - 0.5 * Math.log(V[k]) - 0.5 * (x - t * M.m[k]) * (x - t * M.m[k]) / V[k];
      vk[k] = M.m[k] + (t * M.s[k] * M.s[k] - (1 - t)) / V[k] * (x - t * M.m[k]);
      if (lw[k] > mx) mx = lw[k];
    }
    let z = 0, v = 0; for (let k = 0; k < K; k++) { lw[k] = Math.exp(lw[k] - mx); z += lw[k]; }
    for (let k = 0; k < K; k++) v += (lw[k] / z) * vk[k];
    return v;
  };
  DF.flowDensity = (x, t, M) => { let p = 0; for (let k = 0; k < M.w.length; k++) p += M.w[k] * DF.npdf(x, t * M.m[k], (1 - t) * (1 - t) + t * t * M.s[k] * M.s[k]); return p; };
  DF.flowRun = (x0s, S, M) => { const x = Float64Array.from(x0s); for (let i = 0; i < S; i++) { const t = i / S; for (let j = 0; j < x.length; j++) x[j] += DF.flowV(x[j], t, M) / S; } return x; };
  /* continuous-time VP schedule: abar(t) = exp(-(bmin t + (bmax-bmin) t^2 / 2)), beta(t) = bmin + (bmax-bmin) t */
  DF.vp = { bmin: 0.1, bmax: 20 };
  DF.vpAbar = (t) => Math.exp(-(DF.vp.bmin * t + 0.5 * (DF.vp.bmax - DF.vp.bmin) * t * t));
  DF.vpBeta = (t) => DF.vp.bmin + (DF.vp.bmax - DF.vp.bmin) * t;
  AI.genDF = DF;

  /* ---------- shared drawing and stepping helpers ---------- */
  const T = 1000, AB = DF.abarOf(DF.linearBetas(T, 1e-4, 0.02));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const mk = (L, Tp, R, B, a0, a1, b0, b1) => ({ L, T: Tp, R, B, X: (a) => L + (a - a0) / (a1 - a0) * (R - L), Y: (b) => B - (b - b0) / (b1 - b0) * (B - Tp), a0, a1, b0, b1 });
  const tint = (col, a) => { const [r, g, b] = AI.rgb(col); return `rgba(${r},${g},${b},${a})`; };
  const hist = (xs, lo, hi, nb) => { const h = new Float64Array(nb); for (const x of xs) { const k = Math.floor((x - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) h[k]++; } const bw = (hi - lo) / nb; for (let k = 0; k < nb; k++) h[k] /= xs.length * bw; return h; };
  const stats = (xs) => { let m = 0, v = 0, p = 0; for (const x of xs) m += x; m /= xs.length; for (const x of xs) { v += (x - m) * (x - m); if (x > 0) p++; } return { m, sd: Math.sqrt(v / xs.length), pos: p / xs.length }; };
  const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
  const f3 = (v) => v.toFixed(3);
  const frame = (g, c, m, yticks, xticks, xlab, ylab) => {
    g.strokeStyle = c.border; g.lineWidth = 1; g.strokeRect(m.L, m.T, m.R - m.L, m.B - m.T);
    for (const v of yticks) { g.strokeStyle = v === 0 ? c.border : tint(c.border, 0.4); g.beginPath(); g.moveTo(m.L, m.Y(v)); g.lineTo(m.R, m.Y(v)); g.stroke(); txt(g, String(v), m.L - 6, m.Y(v) + 4, c.muted, '11px sans-serif', 'right'); }
    for (const [v, s] of xticks) txt(g, s, m.X(v), m.B + 14, c.muted, '11px sans-serif', 'center');
    if (xlab) txt(g, xlab, (m.L + m.R) / 2, m.B + 28, c.muted, '11px sans-serif', 'center');
    if (ylab) txt(g, ylab, m.L, m.T - 7, c.muted, '11px sans-serif');
  };
  /* horizontal density panel sharing the y axis of a plane: bars grow rightwards from x = L; curves are functions of x. scale = density that fills the panel */
  const histPanel = (g, c, m, L, R, xs, curves, scale, nb) => {
    const lo = m.b0, hi = m.b1, h = hist(xs, lo, hi, nb), bw = (hi - lo) / nb;
    g.strokeStyle = c.border; g.strokeRect(L, m.T, R - L, m.B - m.T);
    g.fillStyle = tint(c.orange, 0.55);
    for (let k = 0; k < nb; k++) { const y0 = m.Y(lo + k * bw), y1 = m.Y(lo + (k + 1) * bw); g.fillRect(L, y1 + 0.5, Math.min(R - L, h[k] / scale * (R - L)), y0 - y1 - 1); }
    for (const cv of curves) { g.strokeStyle = cv.col; g.lineWidth = 2; g.setLineDash(cv.dash || []); g.beginPath(); for (let i = 0; i <= 120; i++) { const x = lo + (hi - lo) * i / 120, px = L + Math.min(1.05, cv.f(x) / scale) * (R - L), py = m.Y(x); i ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke(); g.setLineDash([]); g.lineWidth = 1; }
  };
  /* drawing one polyline per tracked particle; pts = [[t, x], ...] in data units */
  const track = (g, m, pts, col, w) => { if (pts.length < 2) return; g.strokeStyle = col; g.lineWidth = w || 1; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])))); g.stroke(); g.lineWidth = 1; };
  const arrow = (g, x0, y0, x1, y1, col, hd) => { g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); const a = Math.atan2(y1 - y0, x1 - x0), h = hd || 5; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - h * Math.cos(a - 0.5), y1 - h * Math.sin(a - 0.5)); g.lineTo(x1 - h * Math.cos(a + 0.5), y1 - h * Math.sin(a + 0.5)); g.closePath(); g.fill(); g.lineWidth = 1; };

  /* A sampler that walks N particles through S reverse steps with the exact denoiser (or a given epsilon function).
     Positions are interpolated between steps so motion is smooth; dt-based through advance(dt, stepsPerSecond). */
  const Sampler = (o) => {
    const s = { i: 0, p: 0, done: false };
    s.init = () => {
      s.S = o.S(); s.eta = o.eta(); s.M = o.M();
      s.ts = DF.timesteps(T, s.S); s.seq = DF.abarSeq(AB, T, s.S); s.tt = s.ts.concat([0]);
      s.z0 = DF.noise(o.N, o.seed || 7); s.xa = Float64Array.from(s.z0); s.xb = new Float64Array(o.N); s.eps = new Float64Array(o.N); s.x0h = new Float64Array(o.N);
      s.rng = AI.rng((o.seed || 7) + 100); s.trail = []; for (let j = 0; j < o.NT; j++) s.trail.push([[s.tt[0], s.xa[j]]]);
      s.i = 0; s.p = 0; s.done = false; s.prep();
    };
    s.epsFn = (x, a, b) => (o.epsFn ? o.epsFn(x, a, b) : DF.epsStar(x, s.M, a, b));
    s.prep = () => {
      if (s.i >= s.S) { s.done = true; return; }
      const at = s.seq[s.i], as = s.seq[s.i + 1], a = Math.sqrt(at), b = Math.sqrt(1 - at), drop = o.dropNoise && o.dropNoise();
      for (let j = 0; j < o.N; j++) {
        const e = s.epsFn(s.xa[j], a, b); s.eps[j] = e; s.x0h[j] = DF.x0hat(s.xa[j], at, e);
        const z = s.eta > 0 ? AI.randn(s.rng) : 0;
        s.xb[j] = DF.stepTo(s.xa[j], at, as, e, drop ? 1 : s.eta, drop ? 0 : z);
        if (j === 0) { const sg = DF.sigma(at, as, drop ? 1 : s.eta); s.dbg = { at, as, x: s.xa[0], e, x0h: s.x0h[0], sg: drop ? 0 : sg, z, mean: DF.stepTo(s.xa[0], at, as, e, drop ? 1 : s.eta, 0), xn: s.xb[0] }; }
      }
    };
    s.commit = () => { for (let j = 0; j < o.N; j++) s.xa[j] = s.xb[j]; s.i++; for (let j = 0; j < o.NT; j++) s.trail[j].push([s.tt[s.i], s.xa[j]]); s.prep(); };
    s.advance = (dt, rate) => { if (s.done) return false; s.p += dt * rate; while (s.p >= 1 && !s.done) { s.p -= 1; s.commit(); } return true; };
    s.stepOnce = () => { if (!s.done) { s.p = 0; s.commit(); } };
    s.pos = (j) => (s.done ? s.xa[j] : s.xa[j] + (s.xb[j] - s.xa[j]) * s.p);
    s.tNow = () => (s.done ? 0 : s.tt[s.i] + (s.tt[s.i + 1] - s.tt[s.i]) * s.p);
    s.abNow = () => (s.done ? 1 : s.seq[s.i] + (s.seq[s.i + 1] - s.seq[s.i]) * s.p);
    s.cur = () => { const a = new Float64Array(o.N); for (let j = 0; j < o.N; j++) a[j] = s.pos(j); return a; };
    return s;
  };

  /* =====================================================================
     1. Forward noising: step by step versus the closed form
     ===================================================================== */
  AI.labs['gen-df-forward'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Forward noising: step by step versus the closed form', 'illustrative: 1 000 points from a two-bump data distribution, T = 1 000 steps, noise added one step at a time',
      `<canvas style="height:430px"></canvas><p class="caption" id="cap"></p>`,
      `<label>Noise schedule <select id="sch"><option value="lin">linear β (from 0.0001 to β max)</option><option value="cos">cosine ᾱ</option><option value="fx">my formula for ᾱ(u)</option></select></label>
       ${sl('bmax', 'β max (linear schedule)', 0.005, 0.05, 0.001, 0.02)}
       <label>ᾱ(u), u = t / T (used by "my formula"; must start at 1)<input type="text" id="fx" value="cos(u*pi/2)^2" spellcheck="false"></label><div class="err" id="err"></div>
       ${sl('sd', 'Width s of each data bump', 0, 0.6, 0.05, 0.3)}
       ${sl('rate', 'Steps per second', 10, 400, 10, 60)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:6.6em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 1000, NT = 60, SEP = 1;
    const st = { playing: true };
    let betas, ab, M, xs, hist_, t, dirty = true, tick = AI.ticker(), rs, fxFn = null;
    const ei = AI.exprInput(q(ctl, '#fx'), q(ctl, '#err'), { allowed: ['u'], check: (f) => { if (Math.abs(f({ u: 0 }) - 1) > 1e-6) throw new Error('ᾱ(0) must equal 1'); if (!isFinite(f({ u: 1 }))) throw new Error('ᾱ(1) is not a number'); }, onChange: (f) => { fxFn = f; if (q(ctl, '#sch').value === 'fx') reset(); } });
    const build = () => {
      const k = q(ctl, '#sch').value;
      if (k === 'lin') betas = DF.linearBetas(T, 1e-4, +q(ctl, '#bmax').value);
      else if (k === 'cos') betas = DF.cosineBetas(T);
      else { betas = new Float64Array(T); const f = fxFn || (() => 1); for (let i = 1; i <= T; i++) { const r = Math.max(1e-9, f({ u: i / T })) / Math.max(1e-9, f({ u: (i - 1) / T })); betas[i - 1] = Math.min(0.999, Math.max(0, 1 - r)); } }
      ab = DF.abarOf(betas);
    };
    const reset = () => {
      build(); M = DF.mixture(SEP, +q(ctl, '#sd').value); rs = AI.rng(13);
      xs = new Float64Array(N); for (let j = 0; j < N; j++) { const k = rs() < 0.5 ? 0 : 1; xs[j] = M.m[k] + M.s[k] * AI.randn(rs); }
      hist_ = new Float32Array(NT * (T + 1)); for (let j = 0; j < NT; j++) hist_[j * (T + 1)] = xs[j];
      t = 0; dirty = true;
    };
    const adv = (n) => { for (let i = 0; i < n && t < T; i++) { const b = betas[t], sa = Math.sqrt(1 - b), sb = Math.sqrt(b); for (let j = 0; j < N; j++) { xs[j] = sa * xs[j] + sb * AI.randn(rs); if (j < NT) hist_[j * (T + 1) + t + 1] = xs[j]; } t++; } dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 40, R = Math.round(W * 0.64), pb = Math.round(H * 0.68), m = mk(L, 22, R, pb, 0, T, -3.2, 3.2);
      frame(g, c, m, [-3, -2, -1, 0, 1, 2, 3], [[0, '0'], [250, '250'], [500, '500'], [750, '750'], [1000, '1000']], 'noising step t', 'x');
      const a = Math.sqrt(ab[t]), b = Math.sqrt(1 - ab[t]);
      // closed-form bands: where each bump should be at every t (centre sqrt(abar) m, spread sqrt(abar s^2 + 1 - abar))
      for (let k = 0; k < 2; k++) {
        g.fillStyle = tint(k ? c.orange : c.blue, 0.13); g.beginPath();
        const up = [], dn = [];
        for (let i = 0; i <= T; i += 10) { const ca = Math.sqrt(ab[i]), sdv = Math.sqrt(ab[i] * M.s[k] * M.s[k] + 1 - ab[i]); up.push([i, ca * M.m[k] + sdv]); dn.push([i, ca * M.m[k] - sdv]); }
        up.forEach((p, i) => (i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])))); for (let i = dn.length - 1; i >= 0; i--) g.lineTo(m.X(dn[i][0]), m.Y(dn[i][1])); g.closePath(); g.fill();
        g.strokeStyle = tint(k ? c.orange : c.blue, 0.8); g.setLineDash([5, 4]); g.beginPath(); for (let i = 0; i <= T; i += 10) { const y = Math.sqrt(ab[i]) * M.m[k]; i ? g.lineTo(m.X(i), m.Y(y)) : g.moveTo(m.X(i), m.Y(y)); } g.stroke(); g.setLineDash([]);
      }
      for (let j = 0; j < NT; j++) { const pts = []; const stp = Math.max(1, Math.floor(t / 250)); for (let i = 0; i <= t; i += stp) pts.push([i, hist_[j * (T + 1) + i]]); pts.push([t, hist_[j * (T + 1) + t]]); track(g, m, pts, tint(xs[0] === xs[0] && hist_[j * (T + 1)] < 0 ? c.blue : c.orange, 0.55), 1); }
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(t), m.T); g.lineTo(m.X(t), m.B); g.stroke(); g.setLineDash([]);
      // schedule strip
      const sb2 = mk(L, pb + 40, R, H - 20, 0, T, 0, 1);
      g.strokeStyle = c.border; g.strokeRect(sb2.L, sb2.T, sb2.R - sb2.L, sb2.B - sb2.T); txt(g, 'ᾱ_t (signal kept)', L, sb2.T - 6, c.muted, '11px sans-serif');
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= T; i += 5) (i ? g.lineTo(sb2.X(i), sb2.Y(ab[i])) : g.moveTo(sb2.X(i), sb2.Y(ab[i]))); g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.fg; g.beginPath(); g.arc(sb2.X(t), sb2.Y(ab[t]), 4, 0, 7); g.fill(); txt(g, '1', L - 6, sb2.T + 8, c.muted, '11px sans-serif', 'right'); txt(g, '0', L - 6, sb2.B, c.muted, '11px sans-serif', 'right');
      // histogram with closed form curve
      const HL = R + 16, HR = W - 10;
      txt(g, 'bars: the 1 000 points', HL, m.T - 19, c.muted, '10px sans-serif'); txt(g, 'line: closed form', HL, m.T - 7, c.muted, '10px sans-serif');
      histPanel(g, c, m, HL, HR, xs, [{ col: c.green, f: (x) => DF.density(x, M, a, b) }], 1.6, 32);
      // readout
      const s = stats(xs), v2 = M.w.reduce((u, w, k) => u + w * (ab[t] * (M.m[k] * M.m[k] + M.s[k] * M.s[k]) + 1 - ab[t]), 0);
      q(ctl, '#out').innerHTML = `step t = ${t} of ${T}\nᾱ_t = ${ab[t].toFixed(4)}   √ᾱ_t = ${a.toFixed(3)}   √(1-ᾱ_t) = ${b.toFixed(3)}\nSNR = ᾱ/(1-ᾱ) = ${t ? (ab[t] / (1 - ab[t])).toFixed(3) : '∞'}\n<span class="o">step by step: std ${f3(s.sd)}</span>   <span class="g">closed form: std ${f3(Math.sqrt(v2))}</span>`;
      q(view, '#cap').textContent = t === 0 ? 'Press play. Each point is nudged by a little fresh noise at every step, and shrunk slightly toward 0. The shaded bands are the closed-form prediction for where each bump should be at step t.' : ab[t] > 0.9 ? 'Early steps: the two bumps are still clearly two bumps, only blurred. Almost all of the signal is left (ᾱ close to 1).' : ab[t] > 0.3 ? 'Halfway: the bumps overlap and merge. A point in the middle could have come from either bump: that ambiguity is what the denoiser must resolve.' : ab[t] > 0.02 ? 'Most of the signal is gone. The shape is nearly a single bell curve; what is left is a faint pull toward the old bump centres.' : 'End of the process: the points are almost exactly standard normal noise, whatever the data looked like. This is where sampling will start.';
      showVals(ctl, { bmax: (v) => v.toFixed(3), sd: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, () => adv(10), reset, draw);
    ['sch', 'bmax', 'sd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { reset(); draw(); }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    ei.run(); o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && t < T) adv(tick(dt, +q(ctl, '#rate').value));
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Learning to predict the noise (one noise level)
     ===================================================================== */
  AI.labs['gen-df-loss'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Learning to predict the noise at one noise level', 'illustrative: a piecewise-linear model trained by stochastic gradient descent on freshly noised samples; the exact optimum is drawn for comparison',
      `<canvas style="height:400px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('tt', 'Noise level: step t (of 1 000)', 1, 1000, 1, 250)}
       ${sl('sd', 'Width s of each data bump', 0, 0.6, 0.05, 0.3)}
       ${sl('lr', 'Learning rate', 0.01, 0.4, 0.01, 0.1)}
       ${sl('rate', 'Training samples per second', 5, 400, 5, 40)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8.4em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), K = 41, XL = -4, HK = 0.2, NP = 500;
    const st = { playing: true };
    let w, pts, ptr, nseen, ema, floorLoss, last, hl, hist_, rs, M, a, b, dirty = true, tick = AI.ticker();
    const pred = (x) => { const u = Math.min(K - 1.0001, Math.max(0, (x - XL) / HK)), i = Math.floor(u), fr = u - i; return w[i] * (1 - fr) + w[i + 1] * fr; };
    const reset = () => {
      const tv = +q(ctl, '#tt').value; a = Math.sqrt(AB[tv]); b = Math.sqrt(1 - AB[tv]); M = DF.mixture(1, +q(ctl, '#sd').value);
      w = new Float64Array(K); pts = []; ptr = 0; nseen = 0; ema = 1; hist_ = []; last = null; rs = AI.rng(17);
      const r2 = AI.rng(99); let s = 0; for (let i = 0; i < 6000; i++) { const k = r2() < 0.5 ? 0 : 1, x0 = M.m[k] + M.s[k] * AI.randn(r2), e = AI.randn(r2), x = a * x0 + b * e; s += (e - DF.epsStar(x, M, a, b)) ** 2; } floorLoss = s / 6000;
      dirty = true;
    };
    const sample = () => {
      const lr = +q(ctl, '#lr').value, k = rs() < 0.5 ? 0 : 1, x0 = M.m[k] + M.s[k] * AI.randn(rs), e = AI.randn(rs), x = a * x0 + b * e;
      const u = Math.min(K - 1.0001, Math.max(0, (x - XL) / HK)), i = Math.floor(u), fr = u - i, p = w[i] * (1 - fr) + w[i + 1] * fr, err = p - e;
      w[i] -= lr * 2 * err * (1 - fr) * 0.5; w[i + 1] -= lr * 2 * err * fr * 0.5;
      ema += 0.01 * (err * err - ema); nseen++;
      last = { x0, e, x, p, err2: err * err, es: DF.epsStar(x, M, a, b) };
      if (pts.length < NP) pts.push([x, e]); else { pts[ptr] = [x, e]; ptr = (ptr + 1) % NP; }
      if (nseen % 10 === 0) hist_.push(ema);
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R = Math.round(W * 0.58), m = mk(40, 24, R, H - 34, -4, 4, -3.5, 3.5);
      frame(g, c, m, [-3, -2, -1, 0, 1, 2, 3], [[-4, '-4'], [-2, '-2'], [0, '0'], [2, '2'], [4, '4']], 'x_t (the noisy input the network sees)', 'ε (the noise that was added)');
      g.fillStyle = tint(c.muted, 0.35); for (const p of pts) { g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 2, 0, 7); g.fill(); }
      const curve = (fn, col, wd, dash) => { g.strokeStyle = col; g.lineWidth = wd; g.setLineDash(dash || []); g.beginPath(); for (let i = 0; i <= 200; i++) { const x = -4 + 8 * i / 200, y = Math.max(-3.5, Math.min(3.5, fn(x))); i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)); } g.stroke(); g.setLineDash([]); g.lineWidth = 1; };
      curve((x) => DF.epsStar(x, M, a, b), c.green, 2.5);
      curve(pred, c.orange, 2.5);
      if (last) { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(m.X(last.x), m.Y(last.e)); g.lineTo(m.X(last.x), m.Y(last.p)); g.stroke(); g.fillStyle = c.red; g.beginPath(); g.arc(m.X(last.x), m.Y(last.e), 5, 0, 7); g.fill(); g.lineWidth = 1; }
      txt(g, 'green: exact best E[ε | x_t]', m.L + 6, m.T + 14, c.green, '11px sans-serif'); txt(g, 'orange: the model', m.L + 6, m.T + 28, c.orange, '11px sans-serif'); txt(g, 'red: latest error', m.L + 6, m.T + 42, c.red, '11px sans-serif');
      // loss panel
      const l = mk(R + 38, 24, W - 10, H - 34, 0, Math.max(60, hist_.length), 0, 1.1);
      g.strokeStyle = c.border; g.strokeRect(l.L, l.T, l.R - l.L, l.B - l.T); txt(g, 'running loss', l.L, l.T - 8, c.muted, '11px sans-serif');
      for (const v of [0, 0.5, 1]) txt(g, String(v), l.L - 5, l.Y(v) + 4, c.muted, '11px sans-serif', 'right');
      const hline = (v, col, label) => { g.strokeStyle = col; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(l.L, l.Y(v)); g.lineTo(l.R, l.Y(v)); g.stroke(); g.setLineDash([]); txt(g, label, l.R - 4, l.Y(v) - 4, col, '10px sans-serif', 'right'); };
      hline(1, c.muted, 'always 0: 1.00'); hline(floorLoss, c.green, 'best: ' + floorLoss.toFixed(3));
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); hist_.forEach((v, i) => (i ? g.lineTo(l.X(i), l.Y(Math.min(1.1, v))) : g.moveTo(l.X(i), l.Y(Math.min(1.1, v))))); g.stroke(); g.lineWidth = 1;
      txt(g, 'samples seen: ' + nseen, (l.L + l.R) / 2, l.B + 16, c.fg, '11px sans-serif', 'center');
      q(ctl, '#out').innerHTML = last ? `ᾱ_t = ${AB[+q(ctl, '#tt').value].toFixed(4)}  (√ᾱ = ${f3(a)}, √(1-ᾱ) = ${f3(b)})\nlatest sample: x₀ = ${f2(last.x0)}, ε = ${f2(last.e)}\n  x_t = ${f3(a)}·${f2(last.x0)} + ${f3(b)}·${f2(last.e)} = ${f2(last.x)}\n  model says ${f2(last.p)}, truth ${f2(last.e)}, loss = ${f3(last.err2)}\n  best possible answer here: ${f2(last.es)}\nrunning loss ${f3(ema)}  vs  floor ${f3(floorLoss)}` : 'Press play: each step draws a data point x₀ and noise ε, forms x_t, and trains the model to output ε.';
      q(view, '#cap').textContent = nseen < 20 ? 'Each grey dot is one training example: the noisy value x_t (horizontal) and the noise ε that was added (vertical). The model must draw a curve through the cloud.' : ema > floorLoss * 1.3 ? 'The orange curve is still moving toward the green one. The loss falls but cannot reach zero: for a given x_t many different ε are possible.' : 'The model has found the green curve, the average noise for each x_t. The loss now sits at the floor: the part of ε that no function of x_t can predict.';
      showVals(ctl, { tt: (v) => v, sd: (v) => v.toFixed(2), lr: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, () => { for (let i = 0; i < 25; i++) sample(); }, reset, draw);
    q(ctl, '#tt').oninput = () => { reset(); draw(); }; q(ctl, '#sd').oninput = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < n; i++) sample(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. DDPM sampling with the exact denoiser
     ===================================================================== */
  AI.labs['gen-df-ddpm'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'DDPM sampling: pulling noise back into data', 'illustrative: the exact denoiser of a two-bump distribution stands in for a trained network; 600 particles, 30 drawn as paths',
      `<canvas style="height:430px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('S', 'Number of sampling steps S', 2, 200, 1, 40)}
       ${sl('sd', 'Width s of each data bump', 0.05, 0.6, 0.05, 0.3)}
       ${sl('rate', 'Steps per second', 1, 60, 1, 6)}
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="drop"> leave out the added noise z (keep the mean): a wrong sampler</label>
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="arr" checked> show the denoiser's guess x̂₀ for four particles</label>
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:10em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 600, NT = 30;
    const st = { playing: true };
    const sm = Sampler({ N, NT, seed: 7, S: () => +q(ctl, '#S').value, eta: () => 1, M: () => DF.mixture(1, +q(ctl, '#sd').value), dropNoise: () => q(ctl, '#drop').checked });
    let dirty = true;
    const reset = () => { sm.init(); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R = Math.round(W * 0.7), m = mk(40, 24, R, H - 40, 0, T, -3.4, 3.4);
      frame(g, c, m, [-3, -2, -1, 0, 1, 2, 3], [[0, '0 (data)'], [250, '250'], [500, '500'], [750, '750'], [1000, '1000 (noise)']], 'noise step t: sampling runs from right to left', 'x');
      const M = sm.M, tn = sm.tNow(), ab = sm.abNow(), a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
      // exact marginal band for reference: the particles should stay inside these shaded bumps at every t
      for (let k = 0; k < 2; k++) { g.fillStyle = tint(k ? c.orange : c.blue, 0.1); g.beginPath(); const up = [], dn = []; for (let i = 0; i <= T; i += 10) { const ca = Math.sqrt(AB[i]), sdv = Math.sqrt(AB[i] * M.s[k] * M.s[k] + 1 - AB[i]); up.push([i, ca * M.m[k] + sdv]); dn.push([i, ca * M.m[k] - sdv]); } up.forEach((p, i) => (i ? g.lineTo(m.X(p[0]), m.Y(p[1])) : g.moveTo(m.X(p[0]), m.Y(p[1])))); for (let i = dn.length - 1; i >= 0; i--) g.lineTo(m.X(dn[i][0]), m.Y(dn[i][1])); g.closePath(); g.fill(); }
      for (let j = 0; j < NT; j++) { const col = tint(sm.z0[j] < 0 ? c.blue : c.orange, 0.6), pts = sm.trail[j].slice(); if (!sm.done) pts.push([tn, sm.pos(j)]); track(g, m, pts, col, 1.2); }
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(tn), m.T); g.lineTo(m.X(tn), m.B); g.stroke(); g.setLineDash([]);
      if (q(ctl, '#arr').checked && !sm.done) for (let j = 0; j < 4; j++) { const x = sm.pos(j), x0 = sm.x0h[j]; g.strokeStyle = tint(c.green, 0.8); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(m.X(tn), m.Y(x)); g.lineTo(m.X(0), m.Y(x0)); g.stroke(); g.setLineDash([]); g.fillStyle = c.green; g.beginPath(); g.arc(m.X(0), m.Y(x0), 4, 0, 7); g.fill(); g.beginPath(); g.arc(m.X(tn), m.Y(x), 3.5, 0, 7); g.fill(); }
      const xs = sm.cur(), HL = R + 16;
      txt(g, 'bars: particles', HL, m.T - 19, c.muted, '10px sans-serif'); txt(g, 'line: exact density', HL, m.T - 7, c.muted, '10px sans-serif');
      histPanel(g, c, m, HL, W - 10, xs, [{ col: c.green, f: (x) => DF.density(x, M, a, b) }], 1.6, 32);
      const d = sm.dbg, s = stats(xs);
      let msg = `step ${Math.min(sm.i + (sm.done ? 0 : 1), sm.S)} of ${sm.S}`;
      if (!sm.done && d) msg += `   (ᾱ ${d.at.toFixed(4)} → ${d.as.toFixed(4)})\nparticle 1:  x_t = ${f2(d.x)}\n  ε̂ = ${f3(d.e)}   x̂₀ = ${f2(d.x0h)}\n  mean = ${f2(d.mean)}   σ = ${f3(d.sg)}   z = ${f2(d.z)}\n  x_next = ${f2(d.mean)} + ${f3(d.sg)}·${f2(d.z)} = ${f2(d.xn)}`;
      else msg += `   finished\nsamples: std ${f3(s.sd)}, ${(s.pos * 100).toFixed(1)}% in the right bump\ndata: std ${f3(Math.sqrt(1 + M.s[0] * M.s[0]))}, 50% in the right bump\ndistance to data (W1): ${f3(DF.w1(xs, M))}`;
      q(ctl, '#out').textContent = msg;
      q(view, '#cap').textContent = sm.done ? (q(ctl, '#drop').checked ? 'Without the added noise the particles collapse onto too narrow a set: the spread is far below the data spread. The random z is not decoration, it keeps the spread right.' : 'Finished. The particles reproduce the two bumps with the right width and the right 50/50 split. Change S or press Reset to run it again.') : sm.i < sm.S * 0.3 ? 'Early steps (high noise): the denoiser cannot tell which bump a particle belongs to, so its guess x̂₀ (green) sits near the middle. Steps are small, so the layout forms slowly.' : sm.i < sm.S * 0.7 ? 'Middle steps: the guess x̂₀ now leans clearly to one bump. Each step moves the particle part of the way there and adds a little fresh noise z.' : 'Late steps (low noise): the guess is almost the particle itself. The remaining steps only fine-tune the width of each bump.';
      showVals(ctl, { S: (v) => v, sd: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, () => sm.stepOnce(), reset, draw);
    ['S', 'sd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { reset(); draw(); }; });
    q(ctl, '#drop').onchange = () => { reset(); draw(); }; q(ctl, '#arr').onchange = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && sm.advance(dt, +q(ctl, '#rate').value)) dirty = true; if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     4. Fast samplers: DDPM and DDIM from the same noise
     ===================================================================== */
  AI.labs['gen-df-fast'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'DDPM and DDIM from the same starting noise', 'illustrative: exact denoiser of a two-bump distribution; both samplers use the same S steps, the same schedule and the same 800 starting points',
      `<canvas style="height:520px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('S', 'Number of sampling steps S', 2, 100, 1, 10)}
       ${sl('etaB', 'Sampler B: η (0 = DDIM, 1 = DDPM)', 0, 1, 0.1, 0)}
       ${sl('sd', 'Width s of each data bump', 0.05, 0.6, 0.05, 0.3)}
       ${sl('rate', 'Steps per second', 1, 40, 1, 4)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8.4em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 800, NT = 24, SL = [2, 3, 5, 10, 20, 50, 100];
    const st = { playing: true };
    const mk2 = (etaFn) => Sampler({ N, NT, seed: 7, S: () => +q(ctl, '#S').value, eta: etaFn, M: () => DF.mixture(1, +q(ctl, '#sd').value) });
    const A = mk2(() => 1), B = mk2(() => +q(ctl, '#etaB').value);
    let errs = null, dirty = true;
    const calcErrs = () => {
      const M = DF.mixture(1, +q(ctl, '#sd').value), z = DF.noise(2000, 7), eb = +q(ctl, '#etaB').value;
      errs = SL.map((S) => { const seq = DF.abarSeq(AB, T, S); return [DF.w1(DF.run(z, seq, M, 1, AI.rng(107)), M), DF.w1(DF.run(z, seq, M, eb, AI.rng(108)), M)]; });
    };
    const reset = () => { A.init(); B.init(); dirty = true; };
    const panel = (g, c, m, sm, title, col) => {
      frame(g, c, m, [-2, 0, 2], [[0, '0'], [500, '500'], [1000, '1000']], '', '');
      txt(g, title, m.L, m.T - 6, col, 'bold 11px sans-serif');
      const tn = sm.tNow();
      for (let j = 0; j < NT; j++) { const pts = sm.trail[j].slice(); if (!sm.done) pts.push([tn, sm.pos(j)]); track(g, m, pts, AI.mix(c.blue, c.orange, 1 / (1 + Math.exp(-2.2 * sm.z0[j]))), 1.3); }
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(tn), m.T); g.lineTo(m.X(tn), m.B); g.stroke(); g.setLineDash([]);
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const half = (W - 60) / 2, top = 24, mb = Math.round(H * 0.5);
      const m1 = mk(34, top, 34 + half, mb, 0, T, -3.2, 3.2), m2 = mk(34 + half + 22, top, W - 10, mb, 0, T, -3.2, 3.2);
      const eb = +q(ctl, '#etaB').value;
      panel(g, c, m1, A, 'A: DDPM (η = 1)', c.orange);
      panel(g, c, m2, B, `B: η = ${eb.toFixed(1)}${eb === 0 ? ' (DDIM)' : ''}`, c.blue);
      txt(g, 'noise step t; sampling runs right to left; same colour = same start', m1.L, mb + 30, c.muted, '10px sans-serif');
      // bottom left: histograms
      const M = A.M, xa = A.cur(), xb = B.cur(), hb = mk(34, mb + 52, 34 + half, H - 22, -3.2, 3.2, 0, 1.4);
      g.strokeStyle = c.border; g.strokeRect(hb.L, hb.T, hb.R - hb.L, hb.B - hb.T); txt(g, 'A orange, B blue, data dashed', hb.L, hb.T - 6, c.muted, '10px sans-serif');
      const ha = hist(xa, -3.2, 3.2, 40), hbb = hist(xb, -3.2, 3.2, 40), bw = 6.4 / 40;
      const step = (h, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let k = 0; k < 40; k++) { const x0 = hb.X(-3.2 + k * bw), x1 = hb.X(-3.2 + (k + 1) * bw), y = hb.Y(Math.min(1.4, h[k])); k ? g.lineTo(x0, y) : g.moveTo(x0, y); g.lineTo(x1, y); } g.stroke(); g.lineWidth = 1; };
      step(ha, c.orange); step(hbb, c.blue);
      g.strokeStyle = c.fg; g.setLineDash([4, 3]); g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -3.2 + 6.4 * i / 160, y = hb.Y(Math.min(1.4, DF.density(x, M, 1, 0))); i ? g.lineTo(hb.X(x), y) : g.moveTo(hb.X(x), y); } g.stroke(); g.setLineDash([]);
      // bottom right: error versus steps (log axis)
      if (!errs) calcErrs();
      const em = mk(m2.L + 8, mb + 52, W - 10, H - 22, Math.log(2), Math.log(100), 0, 0.8);
      g.strokeStyle = c.border; g.strokeRect(em.L, em.T, em.R - em.L, em.B - em.T); txt(g, 'final error (W1) versus S', em.L, em.T - 6, c.muted, '10px sans-serif');
      for (const S of [2, 5, 10, 20, 50, 100]) txt(g, String(S), em.X(Math.log(S)), em.B + 13, c.muted, '10px sans-serif', 'center');
      for (const v of [0, 0.4, 0.8]) txt(g, String(v), em.L - 4, em.Y(v) + 4, c.muted, '10px sans-serif', 'right');
      const ln = (idx, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); SL.forEach((S, i) => { const x = em.X(Math.log(S)), y = em.Y(Math.min(0.8, errs[i][idx])); i ? g.lineTo(x, y) : g.moveTo(x, y); g.fillRect(x - 2, y - 2, 4, 4); }); g.stroke(); g.lineWidth = 1; };
      g.fillStyle = c.orange; ln(0, c.orange); g.fillStyle = c.blue; ln(1, c.blue);
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(em.X(Math.log(A.S)), em.T); g.lineTo(em.X(Math.log(A.S)), em.B); g.stroke(); g.setLineDash([]);
      const sa = stats(xa), sb = stats(xb), done = A.done && B.done;
      q(ctl, '#out').innerHTML = `S = ${A.S} steps, same ${N} starting points\n<span class="o">A (DDPM): std ${f3(sa.sd)}, W1 ${f3(DF.w1(xa, M))}</span>\n<span class="b">B (η=${eb.toFixed(1)}): std ${f3(sb.sd)}, W1 ${f3(DF.w1(xb, M))}</span>\ndata std ${f3(Math.sqrt(1 + M.s[0] * M.s[0]))}  (values below about 0.05 are finite-sample noise)`;
      q(view, '#cap').textContent = done ? (A.S <= 5 ? 'With so few steps both samplers fail: the particles land in the wrong places and the spread is too small. Every step is a long jump over a curved path.' : eb === 0 ? 'Same starting noise, different routes. The DDIM paths (B) are smooth and never wiggle, the DDPM paths (A) jitter. Both end on the two bumps, and with few steps the smooth route loses less.' : 'Finished.') : 'Both samplers start from exactly the same points. DDPM re-injects fresh noise at every step; DDIM with η = 0 injects none, so each starting point has one fixed path to its answer.';
      showVals(ctl, { S: (v) => v, etaB: (v) => v.toFixed(1), sd: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, () => { A.stepOnce(); B.stepOnce(); }, reset, draw);
    q(ctl, '#S').oninput = () => { reset(); draw(); };
    q(ctl, '#etaB').oninput = () => { errs = null; reset(); draw(); }; q(ctl, '#sd').oninput = () => { errs = null; reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing) { const r = +q(ctl, '#rate').value, a1 = A.advance(dt, r), b1 = B.advance(dt, r); if (a1 || b1) dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     5a. Score and Langevin dynamics with an editable log-density
     ===================================================================== */
  AI.labs['gen-df-langevin'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The score and Langevin dynamics', 'illustrative: you type log p(x); the score is its derivative (found by automatic differentiation); 400 particles follow x ← x + (ε/2)·score + √ε·z',
      `<canvas style="height:430px"></canvas><p class="caption" id="cap"></p>`,
      `<label>log p(x), up to an added constant<input type="text" id="lp" value="log(exp(-(x-1)^2/0.18)+exp(-(x+1)^2/0.18))" spellcheck="false"></label><div class="err" id="err"></div>
       <label>Presets <select id="pre"><option value="log(exp(-(x-1)^2/0.18)+exp(-(x+1)^2/0.18))">two bumps (this module's toy)</option><option value="-(x-1)^2/0.5">one Gaussian bump, centre 1, variance 0.25</option><option value="log(exp(-(x+1.5)^2/0.1)+2*exp(-(x-1.5)^2/0.1)+exp(-x^2/0.1))">three bumps of different weight</option></select></label>
       ${sl('eps', 'Step size ε', 0.002, 0.5, 0.002, 0.02)}
       ${sl('nz', 'Noise strength (1 = Langevin, 0 = plain uphill walk)', 0, 1, 0.1, 1)}
       ${sl('rate', 'Steps per second', 1, 60, 1, 8)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:6.6em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 400, GL = -4, GH = 4, NG = 241;
    const st = { playing: true };
    let fn = null, ast = null, dens = new Float64Array(NG), tmean = 0, tsd = 1, xa, xb, p = 0, steps = 0, rs, dirty = true;
    const score = (x) => { try { const r = AI.autodiff(ast, { x }).varGrad.x; return isFinite(r) ? r : 0; } catch (e) { return 0; } };
    const prep = () => { const eps = +q(ctl, '#eps').value, nz = +q(ctl, '#nz').value; for (let j = 0; j < N; j++) { const s = score(xa[j]); xb[j] = Math.max(-6, Math.min(6, xa[j] + 0.5 * eps * s + nz * Math.sqrt(eps) * AI.randn(rs))); } };
    const setDensity = () => {
      let mx = -Infinity; const lv = new Float64Array(NG); for (let i = 0; i < NG; i++) { const v = fn({ x: GL + (GH - GL) * i / (NG - 1) }); lv[i] = isFinite(v) ? v : -Infinity; if (lv[i] > mx) mx = lv[i]; }
      let z = 0; for (let i = 0; i < NG; i++) { dens[i] = Math.exp(lv[i] - mx); z += dens[i]; } const dx = (GH - GL) / (NG - 1); for (let i = 0; i < NG; i++) dens[i] /= z * dx;
      tmean = 0; for (let i = 0; i < NG; i++) tmean += (GL + i * dx) * dens[i] * dx; let v = 0; for (let i = 0; i < NG; i++) v += (GL + i * dx - tmean) ** 2 * dens[i] * dx; tsd = Math.sqrt(v);
    };
    const dAt = (x) => { const u = (x - GL) / (GH - GL) * (NG - 1), i = Math.max(0, Math.min(NG - 2, Math.floor(u))), f = Math.max(0, Math.min(1, u - i)); return dens[i] * (1 - f) + dens[i + 1] * f; };
    const ei = AI.exprInput(q(ctl, '#lp'), q(ctl, '#err'), { allowed: ['x'], check: (f) => { if (!isFinite(f({ x: 0 }))) throw new Error('log p(0) is not a number'); }, onChange: (f, a) => { fn = f; ast = a; setDensity(); reset(); } });
    const reset = () => { if (!fn) return; rs = AI.rng(23); xa = new Float64Array(N); xb = new Float64Array(N); for (let j = 0; j < N; j++) xa[j] = -3.5 + 7 * rs(); steps = 0; p = 0; prep(); dirty = true; };
    const commit = () => { xa.set(xb); steps++; prep(); };
    const draw = () => {
      if (!fn) return;
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const top = mk(40, 22, W - 12, Math.round(H * 0.45), GL, GH, 0, 1.4), bot = mk(40, Math.round(H * 0.45) + 42, W - 12, H - 22, GL, GH, 0, 1.4);
      const pmax = Math.max(...dens), sc = pmax * 1.15;
      g.strokeStyle = c.border; g.strokeRect(top.L, top.T, top.R - top.L, top.B - top.T); txt(g, 'target p(x) and its score (arrows point uphill on log p)', top.L, top.T - 6, c.muted, '11px sans-serif');
      g.fillStyle = tint(c.green, 0.2); g.beginPath(); g.moveTo(top.X(GL), top.B); for (let i = 0; i < NG; i++) g.lineTo(top.X(GL + (GH - GL) * i / (NG - 1)), top.B - Math.min(1, dens[i] / sc) * (top.B - top.T - 30)); g.lineTo(top.X(GH), top.B); g.closePath(); g.fill();
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); for (let i = 0; i < NG; i++) { const x = top.X(GL + (GH - GL) * i / (NG - 1)), y = top.B - Math.min(1, dens[i] / sc) * (top.B - top.T - 30); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); g.lineWidth = 1;
      for (let k = -7; k <= 7; k++) { const x = k * 0.5, s = score(x), len = 34 * Math.tanh(s / 5), px = top.X(x), py = top.T + 14; if (Math.abs(len) > 1) arrow(g, px, py, px + len, py, c.purple, 5); else { g.fillStyle = c.purple; g.beginPath(); g.arc(px, py, 2, 0, 7); g.fill(); } }
      const xs = new Float64Array(N); for (let j = 0; j < N; j++) xs[j] = xa[j] + (xb[j] - xa[j]) * p;
      for (let k = -4; k <= 4; k += 2) txt(g, String(k), bot.X(k), bot.B + 14, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.border; g.strokeRect(bot.L, bot.T, bot.R - bot.L, bot.B - bot.T); txt(g, 'particles now (bars) and the target (line)', bot.L, bot.T - 6, c.muted, '11px sans-serif');
      const h = hist(xs, GL, GH, 40), bw = (GH - GL) / 40; g.fillStyle = tint(c.orange, 0.6);
      for (let k = 0; k < 40; k++) { const hh = Math.min(1, h[k] / sc) * (bot.B - bot.T - 6); g.fillRect(bot.X(GL + k * bw) + 1, bot.B - hh, bot.X(GL + bw) - bot.X(GL) - 2, hh); }
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); for (let i = 0; i < NG; i++) { const x = bot.X(GL + (GH - GL) * i / (NG - 1)), y = bot.B - Math.min(1, dens[i] / sc) * (bot.B - bot.T - 6); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); g.lineWidth = 1;
      const s = stats(xs), eps = +q(ctl, '#eps').value, nz = +q(ctl, '#nz').value;
      q(ctl, '#out').innerHTML = `step ${steps}   particles: mean ${f2(s.m)}, std ${f2(s.sd)}\ntarget:                mean ${f2(tmean)}, std ${f2(tsd)}\nper step: drift (ε/2)·score, noise ${f3(nz * Math.sqrt(eps))}·z`;
      q(view, '#cap').textContent = nz === 0 ? 'Noise off: every particle just climbs to the nearest peak and stops there. The bars become spikes. The noise is what makes them spread out to match the whole distribution.' : eps > 0.2 ? 'A large step size makes the particles overshoot and the bars drift away from the target: Langevin is only exact in the limit of small steps.' : steps < 40 ? 'Particles start spread evenly. The purple arrows show the score: it points toward higher probability. Each step moves a particle a little along it and adds a small random kick.' : 'The bars are settling onto the green curve: the score pulls particles toward the peaks, the noise keeps them spread inside each peak and lets them cross between peaks.';
      showVals(ctl, { eps: (v) => v.toFixed(3), nz: (v) => v.toFixed(1), rate: (v) => v });
    };
    wire(ctl, st, () => { p = 0; commit(); }, reset, draw);
    q(ctl, '#pre').onchange = () => ei.set(q(ctl, '#pre').value);
    q(ctl, '#eps').oninput = () => { if (fn) { prep(); dirty = true; } }; q(ctl, '#nz').oninput = () => { if (fn) { prep(); dirty = true; } }; q(ctl, '#rate').oninput = () => { dirty = true; };
    ei.run(); o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (!fn) return;
      if (st.playing) { p += dt * +q(ctl, '#rate').value; while (p >= 1) { p -= 1; commit(); } dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5b. Reverse SDE versus probability-flow ODE (continuous-time VP diffusion)
     ===================================================================== */
  AI.labs['gen-df-sde'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Reverse-time SDE and probability-flow ODE', 'illustrative: exact score of a two-bump distribution, variance-preserving diffusion with β(τ) = 0.1 + 19.9 τ; same starting noise on both sides',
      `<canvas style="height:480px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('lam', 'Left: noise strength λ in the reverse process (1 = the SDE)', 0, 2, 0.25, 1)}
       ${sl('sd', 'Width s of each data bump', 0.05, 0.6, 0.05, 0.3)}
       ${sl('rate', 'Diffusion time covered per second', 0.02, 0.4, 0.01, 0.08)}
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="fld" checked> draw the exact ODE arrows on the right panel</label>
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:7.5em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 500, NT = 24, H = 0.002, T0 = 0.002;
    const st = { playing: true };
    let M, z0, xL, xR, tau, acc, rL, rR, trL, trR, dirty = true;
    const reset = () => {
      M = DF.mixture(1, +q(ctl, '#sd').value); z0 = DF.noise(N, 7); xL = Float64Array.from(z0); xR = Float64Array.from(z0); tau = 1; acc = 0; rL = AI.rng(31); rR = AI.rng(32);
      trL = []; trR = []; for (let j = 0; j < NT; j++) { trL.push([[1, z0[j]]]); trR.push([[1, z0[j]]]); } dirty = true;
    };
    const sub = () => {
      const lam = +q(ctl, '#lam').value, ab = DF.vpAbar(tau), a = Math.sqrt(ab), b = Math.sqrt(1 - ab), be = DF.vpBeta(tau);
      for (let j = 0; j < N; j++) {
        let sc = DF.score(xL[j], M, a, b); xL[j] += (0.5 * be * xL[j] + 0.5 * (1 + lam * lam) * be * sc) * H + lam * Math.sqrt(be * H) * AI.randn(rL);
        sc = DF.score(xR[j], M, a, b); xR[j] += (0.5 * be * xR[j] + 0.5 * be * sc) * H;
      }
      tau -= H; if (Math.round((1 - tau) / H) % 5 === 0) for (let j = 0; j < NT; j++) { trL[j].push([tau, xL[j]]); trR[j].push([tau, xR[j]]); }
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const half = (W - 56) / 2, top = 24, mb = Math.round(Hh * 0.56);
      const m1 = mk(34, top, 34 + half, mb, 0, 1, -3.2, 3.2), m2 = mk(34 + half + 22, top, W - 10, mb, 0, 1, -3.2, 3.2), lam = +q(ctl, '#lam').value;
      const pan = (m, tr, xs, title, col) => {
        frame(g, c, m, [-2, 0, 2], [[0, '0 (data)'], [0.5, '0.5'], [1, '1 (noise)']], '', '');
        txt(g, title, m.L, m.T - 6, col, 'bold 11px sans-serif');
        for (let j = 0; j < NT; j++) { const pts = tr[j].slice(); pts.push([tau, xs[j]]); track(g, m, pts, tint(z0[j] < 0 ? c.blue : c.orange, 0.65), 1.2); }
        g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(tau), m.T); g.lineTo(m.X(tau), m.B); g.stroke(); g.setLineDash([]);
      };
      if (q(ctl, '#fld').checked) { for (let ti = 0; ti < 10; ti++) for (let xi = -4; xi <= 4; xi++) { const t = 0.95 - 0.1 * ti, x = xi * 0.75, ab = DF.vpAbar(t), be = DF.vpBeta(t), d = 0.5 * be * x + 0.5 * be * DF.score(x, M, Math.sqrt(ab), Math.sqrt(1 - ab)), dt = 0.06; arrow(g, m2.X(t), m2.Y(x), m2.X(t - dt), m2.Y(x + d * dt), tint(c.purple, 0.5), 4); } }
      pan(m1, trL, xL, `SDE (λ = ${lam})`, c.orange); pan(m2, trR, xR, 'ODE (λ = 0)', c.blue);
      txt(g, 'diffusion time τ; sampling runs right to left; purple = exact ODE velocity', m1.L, mb + 30, c.muted, '10px sans-serif');
      const hb = mk(34, mb + 50, W - 10, Hh - 20, -3.2, 3.2, 0, 1.4), ab = DF.vpAbar(Math.max(tau, 0)), a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
      g.strokeStyle = c.border; g.strokeRect(hb.L, hb.T, hb.R - hb.L, hb.B - hb.T); txt(g, 'SDE orange, ODE blue, exact density dashed', hb.L, hb.T - 6, c.muted, '10px sans-serif');
      const hl = hist(xL, -3.2, 3.2, 48), hr = hist(xR, -3.2, 3.2, 48), bw = 6.4 / 48;
      const step = (h, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let k = 0; k < 48; k++) { const x0 = hb.X(-3.2 + k * bw), x1 = hb.X(-3.2 + (k + 1) * bw), y = hb.Y(Math.min(1.4, h[k])); k ? g.lineTo(x0, y) : g.moveTo(x0, y); g.lineTo(x1, y); } g.stroke(); g.lineWidth = 1; };
      step(hl, c.orange); step(hr, c.blue);
      g.strokeStyle = c.fg; g.setLineDash([4, 3]); g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -3.2 + 6.4 * i / 160, y = hb.Y(Math.min(1.4, DF.density(x, M, a, b))); i ? g.lineTo(hb.X(x), y) : g.moveTo(hb.X(x), y); } g.stroke(); g.setLineDash([]);
      const sl_ = stats(xL), sr = stats(xR);
      q(ctl, '#out').innerHTML = `τ = ${tau.toFixed(3)}   ᾱ(τ) = ${ab.toFixed(4)}   β(τ) = ${DF.vpBeta(Math.max(tau, 0)).toFixed(2)}\n<span class="o">SDE: std ${f3(sl_.sd)}, W1 ${f3(DF.w1(xL, M))}</span>\n<span class="b">ODE: std ${f3(sr.sd)}, W1 ${f3(DF.w1(xR, M))}</span>\nfinished: ${tau <= T0 + 1e-9 ? 'yes' : 'no'}`;
      q(view, '#cap').textContent = tau > 0.98 ? 'Both sides start from the same noise. On the left every particle gets a random kick at every instant; on the right the motion is a fixed flow along the purple arrows.' : tau > T0 + 1e-9 ? 'Different paths, the same bell-then-bumps histogram at every moment: the dashed density is matched by the SDE (jittery paths) and by the ODE (smooth paths). They are two ways of moving the same distribution.' : 'Finished. Both sides reproduce the two bumps. Try λ = 0 on the left: it becomes the ODE and the two panels coincide.';
      showVals(ctl, { lam: (v) => v, sd: (v) => v.toFixed(2), rate: (v) => v.toFixed(2) });
    };
    wire(ctl, st, () => { for (let i = 0; i < 10 && tau > T0 + 1e-9; i++) sub(); }, reset, draw);
    q(ctl, '#lam').oninput = () => { reset(); draw(); }; q(ctl, '#sd').oninput = () => { reset(); draw(); }; q(ctl, '#fld').onchange = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && tau > T0 + 1e-9) { acc += dt * +q(ctl, '#rate').value; let n = 0; while (acc >= H && tau > T0 + 1e-9 && n < 60) { sub(); acc -= H; n++; } dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. Classifier-free guidance on a toy conditional distribution
     ===================================================================== */
  AI.labs['gen-df-guidance'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Guidance: ε_guided = ε_uncond + w·(ε_cond − ε_uncond)', 'illustrative: two overlapping classes (left and right bump); the unconditional model knows both, the conditional model knows only the right one; both are exact; deterministic DDIM sampling, 60 steps',
      `<canvas style="height:540px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('w', 'Guidance scale w (w = 1: plain conditional, w = 0: unconditional)', 0, 10, 0.5, 3)}
       ${sl('sep', 'Class centres at ±', 0.3, 1.2, 0.1, 0.8)}
       ${sl('sd', 'Width s of each class', 0.2, 0.6, 0.05, 0.4)}
       ${sl('px', 'Probe point x_t for the formula', -2, 2, 0.1, 0.2)}
       ${sl('rate', 'Steps per second', 1, 40, 1, 8)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:10em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 800, NT = 24;
    const st = { playing: true };
    let Mu, Mc, w, tilt = null, tm = 0, tsd = 0, dirty = true;
    const sm = Sampler({ N, NT, seed: 7, S: () => 60, eta: () => 0, M: () => Mu, epsFn: (x, a, b) => { const eu = DF.epsStar(x, Mu, a, b), ec = DF.epsStar(x, Mc, a, b); return eu + w * (ec - eu); } });
    const params = () => {
      const sep = +q(ctl, '#sep').value, s = +q(ctl, '#sd').value; w = +q(ctl, '#w').value; Mu = DF.mixture(sep, s); Mc = { w: [1], m: [sep], s: [s] };
      const G = 321, lo = -3.2, hi = 3.2; tilt = new Float64Array(G); let mx = -Infinity;
      for (let i = 0; i < G; i++) { const x = lo + (hi - lo) * i / (G - 1); tilt[i] = w * Math.log(DF.npdf(x, sep, s * s) + 1e-300) + (1 - w) * Math.log(DF.density(x, Mu, 1, 0) + 1e-300); if (tilt[i] > mx) mx = tilt[i]; }
      let z = 0; for (let i = 0; i < G; i++) { tilt[i] = Math.exp(tilt[i] - mx); z += tilt[i]; } const dx = (hi - lo) / (G - 1); for (let i = 0; i < G; i++) tilt[i] /= z * dx;
      tm = 0; for (let i = 0; i < G; i++) tm += (lo + i * dx) * tilt[i] * dx; let v = 0; for (let i = 0; i < G; i++) v += (lo + i * dx - tm) ** 2 * tilt[i] * dx; tsd = Math.sqrt(v);
    };
    const tiltAt = (x) => { const G = tilt.length, u = (x + 3.2) / 6.4 * (G - 1), i = Math.max(0, Math.min(G - 2, Math.floor(u))), f = Math.max(0, Math.min(1, u - i)); return tilt[i] * (1 - f) + tilt[i + 1] * f; };
    const reset = () => { params(); sm.init(); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R = Math.round(W * 0.66), mb = Math.round(H * 0.54), m = mk(40, 24, R, mb, 0, T, -3.2, 3.2);
      frame(g, c, m, [-3, -2, -1, 0, 1, 2, 3], [[0, '0 (data)'], [500, '500'], [1000, '1000 (noise)']], '', 'x');
      const tn = sm.tNow(), ab = sm.abNow(), a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
      for (let j = 0; j < NT; j++) { const pts = sm.trail[j].slice(); if (!sm.done) pts.push([tn, sm.pos(j)]); track(g, m, pts, tint(sm.z0[j] < 0 ? c.blue : c.orange, 0.6), 1.2); }
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(tn), m.T); g.lineTo(m.X(tn), m.B); g.stroke(); g.setLineDash([]);
      const xs = sm.cur(), HL = R + 14;
      txt(g, 'bars: particles now', HL, m.T - 7, c.muted, '10px sans-serif');
      histPanel(g, c, m, HL, W - 10, xs, [{ col: c.muted, f: (x) => DF.density(x, Mu, 1, 0), dash: [2, 3] }, { col: c.blue, f: (x) => DF.npdf(x, Mc.m[0], Mc.s[0] ** 2), dash: [5, 3] }, { col: c.purple, f: (x) => tiltAt(x) }], 1.7, 32);
      txt(g, 'curves at right: grey = all data, blue = the class,', m.L, mb + 30, c.muted, '10px sans-serif'); txt(g, 'purple = naive guess p_c^w · p_u^(1-w)', m.L, mb + 43, c.purple, '10px sans-serif');
      // epsilon curves at the current noise level
      const e = mk(40, mb + 70, W - 12, H - 22, -3, 3, -4, 4), px = +q(ctl, '#px').value;
      g.strokeStyle = c.border; g.strokeRect(e.L, e.T, e.R - e.L, e.B - e.T); txt(g, `noise predictions now (ᾱ = ${ab.toFixed(3)}), against x_t`, e.L, e.T - 6, c.muted, '10px sans-serif');
      for (const v of [-3, -2, -1, 0, 1, 2, 3]) txt(g, String(v), e.X(v), e.B + 13, c.muted, '10px sans-serif', 'center');
      g.strokeStyle = tint(c.border, 0.6); g.beginPath(); g.moveTo(e.L, e.Y(0)); g.lineTo(e.R, e.Y(0)); g.stroke();
      const eu = (x) => DF.epsStar(x, Mu, a, b), ec = (x) => DF.epsStar(x, Mc, a, b), eg = (x) => eu(x) + w * (ec(x) - eu(x));
      g.save(); g.beginPath(); g.rect(e.L, e.T, e.R - e.L, e.B - e.T); g.clip();
      const cur = (fn, col, wd, dash) => { g.strokeStyle = col; g.lineWidth = wd; g.setLineDash(dash || []); g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -3 + 6 * i / 160, y = fn(x); i ? g.lineTo(e.X(x), e.Y(y)) : g.moveTo(e.X(x), e.Y(y)); } g.stroke(); g.setLineDash([]); g.lineWidth = 1; };
      cur(eu, c.muted, 2); cur(ec, c.blue, 2); cur(eg, c.orange, 2.5);
      g.restore();
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(e.X(px), e.T); g.lineTo(e.X(px), e.B); g.stroke(); g.setLineDash([]);
      const dots = [[eu(px), c.muted], [ec(px), c.blue], [eg(px), c.orange]]; for (const [v, col] of dots) { g.fillStyle = col; g.beginPath(); g.arc(e.X(px), e.Y(Math.max(-4, Math.min(4, v))), 4, 0, 7); g.fill(); }
      txt(g, 'ε_uncond', e.L + 6, e.T + 12, c.muted, 'bold 11px sans-serif'); txt(g, 'ε_cond', e.L + 70, e.T + 12, c.blue, 'bold 11px sans-serif'); txt(g, 'ε_guided', e.L + 128, e.T + 12, c.orange, 'bold 11px sans-serif');
      const s = stats(xs), vu = eu(px), vc = ec(px);
      q(ctl, '#out').innerHTML = `at x_t = ${px.toFixed(1)}, current noise level:\n  ε_uncond = ${f3(vu)}\n  ε_cond   = ${f3(vc)}\n  ε_guided = ${f3(vu)} + ${w}·(${f3(vc)} - ${f3(vu)}) = <span class="o">${f3(vu + w * (vc - vu))}</span>\nsamples: mean ${f2(s.m)}, std ${f2(s.sd)}, ${(s.pos * 100).toFixed(0)}% right of 0\nclass: mean ${f2(Mc.m[0])}, std ${f2(Mc.s[0])}   naive guess: mean ${f2(tm)}, std ${f2(tsd)}`;
      q(view, '#cap').textContent = w === 0 ? 'w = 0 ignores the class: the samples cover both bumps, in the data proportions.' : w === 1 ? 'w = 1 is the plain conditional model: only the right bump, with its own width and its centre at the class centre.' : sm.done ? 'With w above 1 the samples are narrower than the class and pushed beyond its centre, away from the other class: higher fidelity to "right", less diversity. In this toy the result overshoots the naive purple guess, so guidance is not simply sampling from p_c^w·p_u^(1-w).' : 'Watch the orange guided curve: it is the unconditional prediction plus w times the push that the class adds. Larger w gives a larger push at every step.';
      showVals(ctl, { w: (v) => v.toFixed(1), sep: (v) => v.toFixed(1), sd: (v) => v.toFixed(2), px: (v) => v.toFixed(1), rate: (v) => v });
    };
    wire(ctl, st, () => sm.stepOnce(), reset, draw);
    ['w', 'sep', 'sd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { reset(); draw(); }; });
    q(ctl, '#px').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && sm.advance(dt, +q(ctl, '#rate').value)) dirty = true; if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     7a. Latent diffusion: what the autoencoder saves
     ===================================================================== */
  AI.labs['gen-df-latent'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'What diffusing in a latent space saves', 'illustrative calculator: counts of numbers and attention scores for one configuration you choose; it does not describe any particular system',
      `<canvas style="height:420px"></canvas><p class="caption" id="cap"></p>`,
      `<label>Image side (pixels) <select id="imgside"><option>256</option><option selected>512</option><option>1024</option></select></label>
       <label>Autoencoder downsampling factor f <select id="f"><option>1</option><option>2</option><option>4</option><option selected>8</option><option>16</option></select></label>
       ${sl('ch', 'Latent channels c', 1, 16, 1, 4)}
       <label>Transformer patch size p (patches of the latent) <select id="p"><option selected>1</option><option>2</option><option>4</option></select></label>
       ${sl('n', 'Denoising steps n at sampling time', 5, 100, 5, 30)}
       <div class="row"><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:11em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    let shown = null, dirty = true;
    const targets = () => {
      const S = +q(ctl, '#imgside').value, f = +q(ctl, '#f').value, c = +q(ctl, '#ch').value, p = +q(ctl, '#p').value, h = S / f, tok = (h / p) ** 2;
      return { S, f, c, p, h, pix: S * S * 3, lat: h * h * c, atP: (S * S) ** 2, atL: tok * tok, tok, pixTok: S * S };
    };
    const reset = () => { shown = null; dirty = true; };
    const fmt = (v) => Math.round(v).toLocaleString('en-US');
    const draw = (dt) => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, t = targets(); g.clearRect(0, 0, W, H);
      const keys = ['pix', 'lat', 'atP', 'atL'];
      if (!shown) shown = { pix: 1, lat: 1, atP: 1, atL: 1 };
      let moving = false; for (const k of keys) { const tl = Math.log10(t[k]), sl_ = Math.log10(shown[k]), d = tl - sl_; if (Math.abs(d) > 0.002) { shown[k] = Math.pow(10, sl_ + d * (1 - Math.exp(-(dt || 0.03) * 5))); moving = true; } else shown[k] = t[k]; }
      // pipeline sketch (box sizes follow log2 of the side length)
      const bs = (side) => 14 + 9 * Math.log2(side / 4), y0 = 26, BH = 90, slot = (W - 24) / 3;
      const stage = (i, side, col, label, sub) => { const s = bs(side), cx = 12 + slot * (i + 0.5); g.fillStyle = tint(col, 0.2); g.strokeStyle = col; g.lineWidth = 2; g.fillRect(cx - s / 2, y0 + (BH - s) / 2, s, s); g.strokeRect(cx - s / 2, y0 + (BH - s) / 2, s, s); g.lineWidth = 1; txt(g, label, cx, y0 + BH + 14, c.fg, 'bold 11px sans-serif', 'center'); txt(g, sub, cx, y0 + BH + 27, c.muted, '10px sans-serif', 'center'); return { cx, s }; };
      const a = stage(0, t.S, c.blue, 'image', `${t.S}×${t.S}×3`), b = stage(1, t.h, c.purple, 'latent', `${t.h}×${t.h}×${t.c}`), d = stage(2, t.S, c.blue, 'image', `${t.S}×${t.S}×3`);
      txt(g, `diffusion runs here: n = ${+q(ctl, '#n').value} steps`, b.cx, 14, c.purple, '10px sans-serif', 'center');
      arrow(g, a.cx + a.s / 2 + 6, y0 + BH / 2, b.cx - b.s / 2 - 6, y0 + BH / 2, c.muted, 6); txt(g, `encoder f=${t.f}`, (a.cx + b.cx) / 2, y0 + BH / 2 - 8, c.muted, '10px sans-serif', 'center');
      arrow(g, b.cx + b.s / 2 + 6, y0 + BH / 2, d.cx - d.s / 2 - 6, y0 + BH / 2, c.muted, 6); txt(g, 'decoder', (b.cx + d.cx) / 2, y0 + BH / 2 - 8, c.muted, '10px sans-serif', 'center');
      // bars
      const top = y0 + BH + 62, pitch = 44, bh = 16, Lb = 16, Rb = W - 100, mx = 13;
      const rows = [['numbers in the image', 'pix', c.blue], ['numbers in the latent', 'lat', c.purple], ['attention scores if run per pixel', 'atP', c.red], [`attention scores per ${t.p > 1 ? 'patch' : 'latent cell'}`, 'atL', c.green]];
      txt(g, 'bar lengths are on a log scale (each grid line is ×100)', Lb, top - 12, c.muted, '10px sans-serif');
      for (let k = 0; k <= mx; k += 2) { g.strokeStyle = tint(c.border, 0.5); g.beginPath(); g.moveTo(Lb + k / mx * (Rb - Lb), top - 4); g.lineTo(Lb + k / mx * (Rb - Lb), top + 4 * pitch - 8); g.stroke(); }
      rows.forEach((r, i) => { const y = top + i * pitch, len = Math.log10(shown[r[1]]) / mx * (Rb - Lb); txt(g, r[0], Lb, y + 8, c.muted, '11px sans-serif'); g.fillStyle = tint(r[2], 0.75); g.fillRect(Lb, y + 14, Math.max(2, len), bh); txt(g, fmt(shown[r[1]]), Lb + Math.max(2, len) + 6, y + 27, c.fg, '11px sans-serif'); });
      q(ctl, '#out').textContent = `image: ${t.S}·${t.S}·3 = ${fmt(t.pix)} numbers\nlatent: ${t.h}·${t.h}·${t.c} = ${fmt(t.lat)} numbers\nsaving: ${fmt(t.pix)} / ${fmt(t.lat)} = ${(t.pix / t.lat).toFixed(2)}×\nself-attention over ${fmt(t.tok)} tokens: ${fmt(t.atL)} scores\n  versus ${fmt(t.pixTok)} pixels: ${fmt(t.atP)} scores\n  ratio ${fmt(t.atP / t.atL)}  (= (f·p)^4 = ${(t.f * t.p) ** 4})\ndenoiser evaluations per image: n = ${+q(ctl, '#n').value}`;
      q(view, '#cap').textContent = t.f === 1 && t.p === 1 ? 'With f = 1 and p = 1 the "latent" is as large as the image: no saving. Raise f.' : 'Attention cost grows with the square of the number of tokens, and tokens fall with the square of f·p, so the attention saving is (f·p)^4. The price is paid elsewhere: the autoencoder must keep enough detail to rebuild the image, and the final quality cannot exceed what the decoder can reconstruct.';
      showVals(ctl, { ch: (v) => v, n: (v) => v });
      return moving;
    };
    q(ctl, '#reset').onclick = () => { reset(); };
    ['imgside', 'f', 'p'].forEach((id) => { q(ctl, '#' + id).onchange = () => { dirty = true; }; }); q(ctl, '#ch').oninput = () => { dirty = true; }; q(ctl, '#n').oninput = () => { dirty = true; };
    o.onresize = () => { dirty = true; }; draw(0.03);
    AI.loop(root, (dt) => { if (dirty) { if (!draw(dt)) dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     7b. Cross-attention to text tokens
     ===================================================================== */
  AI.labs['gen-df-xattn'] = function (root) {
    const names = ['red', 'cube', 'on', 'table'], defKey = [20, 110, 200, 290];
    const { view, ctl } = AI.labShell(root, 'Cross-attention: image positions asking the prompt', 'illustrative: 2-D queries and keys so the dot products can be seen as angles; 5 image positions, 4 prompt tokens; weights = softmax(q·k / √2)',
      `<canvas style="height:470px"></canvas><p class="caption" id="cap"></p>`,
      `${names.map((n, i) => sl('k' + i, `Key angle of "${n}" (degrees)`, 0, 359, 1, defKey[i])).join('')}
       ${sl('len', 'Query length (how decisive the positions are)', 0.5, 6, 0.25, 2)}
       ${sl('spread', 'Angle between neighbouring positions (degrees)', 0, 90, 5, 40)}
       ${sl('rate', 'Sweep speed (degrees per second)', 0, 90, 5, 20)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:7.5em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), P = 5;
    const st = { playing: true };
    let phi = 30, dirty = true;
    const reset = () => { phi = 30; dirty = true; };
    const weights = (i) => { const r = +q(ctl, '#len').value, th = (phi + i * +q(ctl, '#spread').value) * Math.PI / 180, qx = r * Math.cos(th), qy = r * Math.sin(th); const sc = names.map((_, k) => { const a = +q(ctl, '#k' + k).value * Math.PI / 180; return (qx * Math.cos(a) + qy * Math.sin(a)) / Math.SQRT2; }); const mx = Math.max(...sc), e = sc.map((v) => Math.exp(v - mx)), z = e.reduce((u, v) => u + v, 0); return { sc, w: e.map((v) => v / z), qx, qy }; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const tcol = [c.red, c.blue, c.green, c.orange], cx = W / 2, cy = 112, rad = 70;
      txt(g, 'keys (prompt tokens) in colour, queries (image positions) in white', cx, 14, c.muted, '10px sans-serif', 'center');
      g.strokeStyle = c.border; g.beginPath(); g.arc(cx, cy, rad, 0, 7); g.stroke();
      names.forEach((n, k) => { const a = +q(ctl, '#k' + k).value * Math.PI / 180; arrow(g, cx, cy, cx + rad * Math.cos(a), cy - rad * Math.sin(a), tcol[k], 7); txt(g, n, cx + (rad + 20) * Math.cos(a), cy - (rad + 16) * Math.sin(a) + 4, tcol[k], 'bold 12px sans-serif', 'center'); });
      const rs = [];
      for (let i = 0; i < P; i++) { const r = weights(i); rs.push(r); const ln = Math.hypot(r.qx, r.qy) / 6 * rad, ang = Math.atan2(r.qy, r.qx); arrow(g, cx, cy, cx + ln * Math.cos(ang), cy - ln * Math.sin(ang), tint(c.fg, i === 0 ? 0.95 : 0.4), 5); if (i === 0) txt(g, 'q₁', cx + (ln + 11) * Math.cos(ang), cy - (ln + 11) * Math.sin(ang) + 4, c.fg, '11px sans-serif', 'center'); }
      // attention matrix underneath
      const x0 = 70, cw = (W - x0 - 66) / 4, ch = 32, y0 = 252;
      txt(g, 'attention weights (each row sums to 1)', x0, y0 - 30, c.muted, '10px sans-serif');
      names.forEach((n, k) => txt(g, n, x0 + k * cw + cw / 2, y0 - 8, tcol[k], 'bold 11px sans-serif', 'center')); txt(g, 'mixed', x0 + 4 * cw + 30, y0 - 8, c.muted, '10px sans-serif', 'center');
      for (let i = 0; i < P; i++) {
        txt(g, 'position ' + (i + 1), x0 - 6, y0 + i * ch + ch / 2 + 4, c.muted, '10px sans-serif', 'right');
        let rr = 0, gg = 0, bb = 0;
        for (let k = 0; k < 4; k++) { const wv = rs[i].w[k]; g.fillStyle = tint(tcol[k], 0.15 + 0.8 * wv); g.fillRect(x0 + k * cw + 1, y0 + i * ch + 1, cw - 2, ch - 2); txt(g, wv.toFixed(2), x0 + k * cw + cw / 2, y0 + i * ch + ch / 2 + 4, c.fg, '12px sans-serif', 'center'); const rgb = AI.rgb(tcol[k]); rr += wv * rgb[0]; gg += wv * rgb[1]; bb += wv * rgb[2]; }
        g.fillStyle = `rgb(${Math.round(rr)},${Math.round(gg)},${Math.round(bb)})`; g.fillRect(x0 + 4 * cw + 10, y0 + i * ch + 4, 40, ch - 8);
        if (i === 0) { g.strokeStyle = c.fg; g.lineWidth = 2; g.strokeRect(x0, y0, 4 * cw, ch); g.lineWidth = 1; }
      }
      const r0 = rs[0];
      q(ctl, '#out').textContent = `position 1, query angle ${(((phi % 360) + 360) % 360) | 0} degrees\nscores q·k/√2: ${r0.sc.map((v) => v.toFixed(2)).join('  ')}\nweights: ${r0.w.map((v) => v.toFixed(3)).join('  ')}\nsum of weights: ${r0.w.reduce((u, v) => u + v, 0).toFixed(3)}`;
      q(view, '#cap').textContent = 'Each position (white arrow) gives most weight to the tokens whose key arrow points the same way: a small angle between query and key means a large dot product. The "mixed" swatch is the blend of the four token colours (the values) that the position receives. The image supplies the queries; the prompt supplies keys and values.';
      showVals(ctl, { k0: (v) => v, k1: (v) => v, k2: (v) => v, k3: (v) => v, len: (v) => v.toFixed(2), spread: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => { phi += 15; }, reset, draw);
    ['k0', 'k1', 'k2', 'k3', 'len', 'spread', 'rate'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && +q(ctl, '#rate').value > 0) { phi += dt * +q(ctl, '#rate').value; dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     8. Flow matching: straight pairs, exact marginal velocity, Euler transport
     ===================================================================== */
  AI.labs['gen-df-flow'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Flow matching: integrating the velocity field from noise to data', 'illustrative: x_t = (1-t)·x₀ + t·x₁, x₀ is noise, x₁ is data; the exact marginal velocity v(x,t) = E[x₁ - x₀ | x_t = x] of a two-bump distribution is integrated with S Euler steps',
      `<canvas style="height:480px"></canvas><p class="caption" id="cap"></p>`,
      `${sl('S', 'Number of Euler steps S', 1, 40, 1, 8)}
       ${sl('sd', 'Width s of each data bump', 0.1, 0.5, 0.05, 0.3)}
       ${sl('rate', 'Steps per second', 0.5, 20, 0.5, 2)}
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="pairs" checked> show 25 straight training pairs (x₀, x₁)</label>
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="fld" checked> show the exact velocity arrows</label>
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8.4em"></div>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), N = 600, NT = 30;
    const st = { playing: true };
    let M, S, xa, xb, trail, p, i, done, z0, pairs, dirty = true;
    const prep = () => { if (i >= S) { done = true; return; } const t = i / S; for (let j = 0; j < N; j++) xb[j] = xa[j] + DF.flowV(xa[j], t, M) / S; };
    const reset = () => {
      M = DF.mixture(1, +q(ctl, '#sd').value); S = +q(ctl, '#S').value; z0 = DF.noise(N, 7); xa = Float64Array.from(z0); xb = new Float64Array(N); p = 0; i = 0; done = false;
      trail = []; for (let j = 0; j < NT; j++) trail.push([[0, z0[j]]]);
      const r = AI.rng(41); pairs = []; for (let k = 0; k < 25; k++) { const kk = r() < 0.5 ? 0 : 1; pairs.push([AI.randn(r), M.m[kk] + M.s[kk] * AI.randn(r)]); }
      prep(); dirty = true;
    };
    const commit = () => { xa.set(xb); i++; for (let j = 0; j < NT; j++) trail[j].push([i / S, xa[j]]); prep(); };
    const pos = (j) => (done ? xa[j] : xa[j] + (xb[j] - xa[j]) * p), tNow = () => (done ? 1 : (i + p) / S);
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R = Math.round(W * 0.7), m = mk(40, 24, R, H - 40, 0, 1, -3.4, 3.4), tn = tNow();
      frame(g, c, m, [-3, -2, -1, 0, 1, 2, 3], [[0, '0 (noise)'], [0.25, '0.25'], [0.5, '0.5'], [0.75, '0.75'], [1, '1 (data)']], 'time t: here the flow runs from left (noise) to right (data)', 'x');
      if (q(ctl, '#pairs').checked) for (const pr of pairs) { g.strokeStyle = tint(c.muted, 0.35); g.beginPath(); g.moveTo(m.X(0), m.Y(pr[0])); g.lineTo(m.X(1), m.Y(pr[1])); g.stroke(); }
      if (q(ctl, '#fld').checked) for (let ti = 0; ti < 10; ti++) for (let xi = -4; xi <= 4; xi++) { const t = 0.05 + 0.1 * ti, x = xi * 0.75, d = 0.06; arrow(g, m.X(t), m.Y(x), m.X(t + d), m.Y(x + DF.flowV(x, t, M) * d), tint(c.purple, 0.55), 4); }
      for (let j = 0; j < NT; j++) { const pts = trail[j].slice(); if (!done) pts.push([tn, pos(j)]); track(g, m, pts, tint(z0[j] < 0 ? c.blue : c.orange, 0.85), 1.6); g.fillStyle = z0[j] < 0 ? c.blue : c.orange; g.beginPath(); g.arc(m.X(tn), m.Y(pos(j)), 2.5, 0, 7); g.fill(); }
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(m.X(tn), m.T); g.lineTo(m.X(tn), m.B); g.stroke(); g.setLineDash([]);
      const xs = new Float64Array(N); for (let j = 0; j < N; j++) xs[j] = pos(j);
      const HL = R + 16; txt(g, 'bars: particles', HL, m.T - 19, c.muted, '10px sans-serif'); txt(g, 'line: exact p_t(x)', HL, m.T - 7, c.muted, '10px sans-serif');
      histPanel(g, c, m, HL, W - 10, xs, [{ col: c.green, f: (x) => DF.flowDensity(x, tn, M) }], 1.6, 32);
      const s = stats(xs);
      const exactSd = Math.sqrt(tn * tn * 1 + (1 - tn) * (1 - tn) + tn * tn * M.s[0] * M.s[0]);
      let msg = `step ${Math.min(i + (done ? 0 : 1), S)} of ${S}   t = ${tn.toFixed(3)}\nparticles: std ${f3(s.sd)}\nexact marginal at this t: std ${f3(exactSd)}`;
      if (done) msg += `\nW1 to the data: ${f3(DF.w1(xs, M))}`;
      if (!done) msg += `\nvelocity of particle 1 at (x=${f2(xa[0])}, t=${(i / S).toFixed(2)}): ${f2((xb[0] - xa[0]) * S)}`;
      q(ctl, '#out').textContent = msg;
      q(view, '#cap').textContent = done ? (S <= 2 ? 'With one or two big Euler steps the particles land near the mean of the data instead of on the bumps: a single straight step follows the average velocity, not the individual one.' : 'Finished: the particles match the data bumps. At every earlier t the bars matched the green curve too, which is the property the velocity field is built to have.') : 'Every particle moves with the velocity field (purple arrows) read at its current position. Notice that near t = 0 the arrows point toward the middle: the field is an average over all data points that a noisy x could belong to, so its paths bend even though every training pair (grey) is a straight line.';
      showVals(ctl, { S: (v) => v, sd: (v) => v.toFixed(2), rate: (v) => v.toFixed(1) });
    };
    wire(ctl, st, () => { if (!done) { p = 0; commit(); } }, reset, draw);
    ['S', 'sd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { reset(); draw(); }; });
    q(ctl, '#pairs').onchange = () => { dirty = true; }; q(ctl, '#fld').onchange = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && !done) { p += dt * +q(ctl, '#rate').value; while (p >= 1 && !done) { p -= 1; commit(); } dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };
})();
