/* Generative AI Module 7 labs: latent-variable and adversarial generators.
   gen-lt-elbo-discrete, gen-lt-elbo-gauss, gen-lt-reparam, gen-lt-beta, gen-lt-vq,
   gen-lt-gan-discrete, gen-lt-nonsat, gen-lt-game, gen-lt-jsw, gen-lt-modehop, gen-lt-flow1d, gen-lt-coupling.
   All simulations are illustrative toy models, use seeded random numbers (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const ln = Math.log, sg = (a) => 1 / (1 + Math.exp(-a));
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const f3 = (v) => (Number.isFinite(v) ? (Math.abs(v) < 5e-5 ? '0.0000' : v.toFixed(4)) : 'NaN');
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    st.setPlaying = (p) => { st.playing = p; pb.textContent = p ? '⏸ Pause' : '▶ Play'; };
    pb.onclick = () => { st.setPlaying(!st.playing); };
    q(ctl, '#step').onclick = () => { st.setPlaying(false); onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const axisLine = (g, c, x0, y0, x1, y1) => { g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
  const KL = (a, b) => a.reduce((s, p, i) => s + (p > 0 ? p * ln(p / b[i]) : 0), 0);
  const JS = (a, b) => { const m = a.map((p, i) => (p + b[i]) / 2); return 0.5 * KL(a, m) + 0.5 * KL(b, m); };

  /* =====================================================================
     1. ELBO on a two-valued latent: log p(x) = ELBO + KL(q || posterior)
     ===================================================================== */
  AI.labs['gen-lt-elbo-discrete'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The ELBO on a two-valued latent', 'illustrative: the exact posterior is known, so you can watch the ELBO climb to log p(x)',
      `<canvas style="height:340px"></canvas>`,
      `${sl('pz', 'Prior p(z=1)', 0.05, 0.95, 0.05, 0.4)}
       ${sl('l0', 'Decoder p(x=c | z=0)', 0.02, 0.9, 0.01, 0.1)}
       ${sl('l1', 'Decoder p(x=c | z=1)', 0.02, 0.9, 0.01, 0.6)}
       ${sl('qq', 'Encoder q(z=1) (drag to set it by hand)', 0.01, 0.99, 0.01, 0.5)}
       ${sl('speed', 'Ascent speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: false };
    let a = 0, dirty = true;
    const calc = () => {
      const p1 = +q(ctl, '#pz').value, p0 = 1 - p1, l0 = +q(ctl, '#l0').value, l1 = +q(ctl, '#l1').value;
      const A0 = ln(l0) + ln(p0), A1 = ln(l1) + ln(p1), logpx = ln(Math.exp(A0) + Math.exp(A1)), post1 = Math.exp(A1 - logpx);
      return { p0, p1, l0, l1, A0, A1, logpx, post1, post0: 1 - post1 };
    };
    const elboAt = (m, q1) => {
      const q0 = 1 - q1, rec = q0 * ln(m.l0) + q1 * ln(m.l1), kl = q0 * ln(q0 / m.p0) + q1 * ln(q1 / m.p1), gap = q0 * ln(q0 / m.post0) + q1 * ln(q1 / m.post1);
      return { rec, kl, elbo: rec - kl, gap };
    };
    const syncSlider = () => { q(ctl, '#qq').value = sg(a).toFixed(2); };
    const reset = () => { a = 0; syncSlider(); st.setPlaying(true); dirty = true; };
    const step = () => { const m = calc(); a += 0.5 * (m.A1 - m.A0 - a); syncSlider(); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const m = calc(), q1 = sg(a), e = elboAt(m, q1);
      const L = 46, cr = Math.round(W * 0.58), top = 30, bot = H - 36;
      // left: ELBO curve over q(z=1)
      const ys = []; let lo = m.logpx, hi = m.logpx;
      for (let i = 0; i <= 100; i++) { const qq = 0.005 + 0.99 * i / 100, v = elboAt(m, qq).elbo; ys.push(v); lo = Math.min(lo, v); }
      lo = Math.max(lo, m.logpx - 4); const X = (v) => L + v * (cr - L - 10), Y = (v) => bot - (clamp(v, lo, hi + 0.2) - lo) / (hi + 0.2 - lo) * (bot - top);
      txt(g, 'ELBO against q(z=1)', L, 16, c.muted);
      axisLine(g, c, L, bot, cr - 10, bot); axisLine(g, c, L, top, L, bot);
      for (const v of [0, 0.5, 1]) txt(g, v, X(v), bot + 14, c.muted, '11px sans-serif', 'center');
      txt(g, 'q(z=1)', cr - 12, bot + 28, c.muted, '11px sans-serif', 'right');
      g.setLineDash([5, 4]); g.strokeStyle = c.green; g.beginPath(); g.moveTo(L, Y(m.logpx)); g.lineTo(cr - 10, Y(m.logpx)); g.stroke(); g.setLineDash([]);
      txt(g, 'log p(x)', L + 4, Y(m.logpx) - 5, c.green, 'bold 11px sans-serif');
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath();
      ys.forEach((v, i) => { const x = X(0.005 + 0.99 * i / 100), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.green; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(X(m.post1), top); g.lineTo(X(m.post1), bot); g.stroke(); g.setLineDash([]);
      txt(g, 'true posterior', X(m.post1) + 4, top + 10, c.green, '11px sans-serif');
      const px = X(q1), py = Y(e.elbo);
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(px, py); g.lineTo(px, Y(m.logpx)); g.stroke();
      txt(g, 'gap', px + 6, (py + Y(m.logpx)) / 2 + 4, c.red, '11px sans-serif');
      g.fillStyle = c.orange; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill();
      // right: bars for z = 0, 1
      const bx = cr + 24, bw = W - bx - 14, bh = bot - top - 6, gw = bw / 2;
      txt(g, 'over z', bx, 16, c.muted);
      const cols = [[c.blue, 'prior', [m.p0, m.p1]], [c.green, 'posterior', [m.post0, m.post1]], [c.orange, 'q', [1 - q1, q1]]];
      for (let z = 0; z < 2; z++) {
        cols.forEach(([cc, , v], k) => { const w = gw / 3.6, x = bx + z * gw + k * (w + 2), h = v[z] * bh; g.fillStyle = cc; g.fillRect(x, bot - h, w, h); });
        txt(g, 'z=' + z, bx + z * gw + gw / 2 - 10, bot + 14, c.muted, '11px sans-serif', 'center');
      }
      cols.forEach(([cc, nm], k) => { g.fillStyle = cc; g.fillRect(bx + k * (bw / 3), H - 12, 8, 8); txt(g, nm, bx + k * (bw / 3) + 11, H - 4, c.muted, '10px sans-serif'); });
      q(ctl, '#out').innerHTML = `log p(x) = ln(Σ p(z) p(x|z)) = <b>${f3(m.logpx)}</b>   posterior p(z=1|x) = ${f3(m.post1)}<br>` +
        `reconstruction E_q[ln p(x|z)] = ${f3(e.rec)}<br>KL(q‖prior) = ${f3(e.kl)}   so ELBO = rec − KL = <b>${f3(e.elbo)}</b><br>` +
        `gap KL(q‖posterior) = <b>${f3(e.gap)}</b>   check: ELBO + gap = ${f3(e.elbo + e.gap)} = log p(x)`;
      q(ctl, '#cap').textContent = e.gap < 0.003 ? 'The encoder equals the true posterior: the gap is zero and the ELBO touches log p(x). That is the best any q can do.' :
        (st.playing ? 'Ascent: q is nudged toward the value that raises the ELBO, which is exactly the direction that shrinks the gap. The green ceiling never moves, only the orange curve is climbed.' : 'Paused. Every q gives an ELBO below log p(x) by exactly KL(q‖posterior).');
      showVals(ctl, { pz: (v) => v.toFixed(2), l0: (v) => v.toFixed(2), l1: (v) => v.toFixed(2), qq: (v) => v.toFixed(2), speed: (v) => v + '×' });
    };
    wire(ctl, st, step, reset, draw);
    ['#pz', '#l0', '#l1'].forEach((id) => (q(ctl, id).oninput = () => { dirty = true; }));
    q(ctl, '#qq').oninput = () => { st.setPlaying(false); const v = +q(ctl, '#qq').value; a = ln(v / (1 - v)); dirty = true; };
    q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const m = calc(), tgt = m.A1 - m.A0;
        if (Math.abs(tgt - a) > 1e-3) { a += dt * 0.45 * +q(ctl, '#speed').value * (tgt - a); syncSlider(); dirty = true; }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. ELBO for a 1-D Gaussian VAE (linear decoder), closed forms vs Monte Carlo
     ===================================================================== */
  AI.labs['gen-lt-elbo-gauss'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A one-dimensional VAE: prior, encoder and exact posterior', 'illustrative: decoder p(x|z) = N(w z, s²), prior N(0,1), encoder q = N(μ, σ²); the exact posterior is Gaussian, so the gap is known',
      `<canvas style="height:400px"></canvas>`,
      `${sl('x', 'Observed x', -3, 3, 0.1, 2)}
       ${sl('w', 'Decoder weight w', 0.3, 2, 0.1, 1)}
       ${sl('s', 'Decoder noise s', 0.2, 1.5, 0.05, 0.5)}
       ${sl('mu', 'Encoder mean μ', -3, 3, 0.05, 0)}
       ${sl('sig', 'Encoder std σ', 0.05, 2, 0.01, 1)}
       ${sl('speed', 'Ascent speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let mu = 0, lsig = 0, dirty = true;
    const E = (() => { const r = AI.rng(21), a = []; for (let i = 0; i < 2000; i++) a.push(AI.randn(r)); return a; })();
    const P = () => ({ x: +q(ctl, '#x').value, w: +q(ctl, '#w').value, s: +q(ctl, '#s').value });
    const calc = () => {
      const { x, w, s } = P(), sig = Math.exp(lsig);
      const rec = -0.5 * ln(2 * Math.PI * s * s) - ((x - w * mu) ** 2 + w * w * sig * sig) / (2 * s * s);
      const kl = 0.5 * (mu * mu + sig * sig - 1 - 2 * ln(sig));
      const logp = -0.5 * ln(2 * Math.PI * (w * w + s * s)) - x * x / (2 * (w * w + s * s));
      const prec = 1 + w * w / (s * s), pm = (w * x / (s * s)) / prec, ps = 1 / Math.sqrt(prec);
      let mc = 0;
      for (const e of E) { const z = mu + sig * e; mc += ln(npdf(x, w * z, s)) + ln(npdf(z, 0, 1)) - ln(npdf(z, mu, sig)); }
      return { x, w, s, sig, rec, kl, elbo: rec - kl, logp, pm, ps, prec, mc: mc / E.length };
    };
    const sync = () => { q(ctl, '#mu').value = clamp(mu, -3, 3).toFixed(2); q(ctl, '#sig').value = clamp(Math.exp(lsig), 0.05, 2).toFixed(2); };
    const reset = () => { mu = 0; lsig = 0; sync(); st.setPlaying(true); dirty = true; };
    const stepFn = (dt) => {
      const { w, s, x } = P(), sig = Math.exp(lsig), prec = 1 + w * w / (s * s);
      const dmu = w * (x - w * mu) / (s * s) - mu, gl = 1 - prec * sig * sig;
      mu += dt * dmu / prec; lsig = clamp(lsig + dt * clamp(gl / 2, -1, 1), Math.log(0.05), Math.log(2)); sync();
    };
    const step = () => stepFn(0.5);
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const m = calc(), L = 16, R = W - 16, zl = -3.5, zh = 3.5, X = (z) => L + (z - zl) / (zh - zl) * (R - L);
      const t0 = 28, t1 = Math.round(H * 0.56), peak = Math.max(1.6, npdf(0, 0, 1) * 1.2, 1 / (m.sig * 2.5066) * 1.05, 1 / (m.ps * 2.5066) * 1.05);
      const Y = (d) => t1 - d / peak * (t1 - t0);
      txt(g, 'latent z: prior, exact posterior, encoder q, 5 samples', L, 14, c.muted);
      axisLine(g, c, L, t1, R, t1);
      for (let z = -3; z <= 3; z++) txt(g, z, X(z), t1 + 13, c.muted, '11px sans-serif', 'center');
      const curve = (fn, col, dash, fill) => { g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.beginPath(); for (let i = 0; i <= 200; i++) { const z = zl + (zh - zl) * i / 200, y = Y(fn(z)); i ? g.lineTo(X(z), y) : g.moveTo(X(z), y); } g.stroke(); g.setLineDash([]); g.lineWidth = 1;
        if (fill) { g.globalAlpha = 0.2; g.fillStyle = col; g.lineTo(R, t1); g.lineTo(L, t1); g.fill(); g.globalAlpha = 1; } };
      curve((z) => npdf(z, 0, 1), c.blue, [3, 4]);
      curve((z) => npdf(z, m.pm, m.ps), c.green, [6, 4]);
      curve((z) => npdf(z, mu, m.sig), c.orange, null, true);
      for (let i = 0; i < 5; i++) { const z = mu + m.sig * E[i]; g.fillStyle = c.red; g.beginPath(); g.arc(X(z), t1 - 8, 4, 0, 7); g.fill(); g.strokeStyle = c.red; g.beginPath(); g.moveTo(X(mu), t1 - 2); g.lineTo(X(z), t1 - 8); g.stroke(); }
      g.strokeStyle = c.fg; g.beginPath(); g.moveTo(X(m.x / m.w), t1 - 4); g.lineTo(X(m.x / m.w), t1 + 4); g.stroke();
      txt(g, 'x/w', X(m.x / m.w), t1 + 26, c.fg, '11px sans-serif', 'center');
      [['prior N(0,1)', c.blue], ['exact posterior', c.green], ['encoder q', c.orange]].forEach(([n, cc], k) => { g.fillStyle = cc; g.fillRect(R - 120, 24 + k * 14, 8, 8); txt(g, n, R - 108, 32 + k * 14, c.muted, '11px sans-serif'); });
      // bars
      const b0 = t1 + 40, rows = [['log p(x)', m.logp, c.green], ['ELBO', m.elbo, c.orange], ['reconstruction', m.rec, c.blue], ['−KL(q‖prior)', -m.kl, c.purple]];
      const mx = Math.max(3, ...rows.map((r) => -r[1])), bw = (R - L - 190), rh = (H - b0 - 8) / 4;
      rows.forEach(([n, v, cc], i) => { const y = b0 + i * rh; txt(g, n, L, y + rh * 0.6, c.muted, '11px sans-serif'); g.fillStyle = cc; g.globalAlpha = 0.8; g.fillRect(L + 110, y + 2, Math.max(1, -v / mx * bw), rh - 6); g.globalAlpha = 1; txt(g, f3(v), L + 114 + Math.max(1, -v / mx * bw), y + rh * 0.6, c.fg, '11px sans-serif'); });
      txt(g, 'longer bar = more negative', R, b0 - 4, c.muted, '10px sans-serif', 'right');
      const gap = m.logp - m.elbo;
      q(ctl, '#out').innerHTML = `exact posterior: N(${f3(m.pm)}, ${f3(m.ps)}²)   log p(x) = <b>${f3(m.logp)}</b><br>` +
        `reconstruction = ${f3(m.rec)}; KL(q‖N(0,1)) = (μ² + σ² − 1 − ln σ²)/2 = ${f3(m.kl)}<br>ELBO (closed form) = <b>${f3(m.elbo)}</b>; Monte Carlo over 2000 samples = ${f3(m.mc)}<br>gap log p(x) − ELBO = <b>${f3(gap)}</b> = KL(q‖posterior)`;
      q(ctl, '#cap').textContent = gap < 0.004 ? 'q has become the exact posterior: ELBO equals log p(x). Note the KL term is not zero: the encoder pays it to explain x.' :
        (st.playing ? 'Ascent in (μ, ln σ): q slides from the prior toward the posterior; the reconstruction rises, the KL grows, and the ELBO climbs toward the green bar.' : 'Paused. Drag μ and σ: the gap to log p(x) is never negative.');
      showVals(ctl, { x: (v) => v.toFixed(1), w: (v) => v.toFixed(1), s: (v) => v.toFixed(2), mu: (v) => v.toFixed(2), sig: (v) => v.toFixed(2), speed: (v) => v + '×' });
    };
    wire(ctl, st, step, reset, draw);
    ['#x', '#w', '#s'].forEach((id) => (q(ctl, id).oninput = () => { dirty = true; }));
    q(ctl, '#mu').oninput = () => { st.setPlaying(false); mu = +q(ctl, '#mu').value; dirty = true; };
    q(ctl, '#sig').oninput = () => { st.setPlaying(false); lsig = ln(+q(ctl, '#sig').value); dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const m = calc();
        if (m.logp - m.elbo > 1e-4) { stepFn(dt * 0.6 * +q(ctl, '#speed').value); dirty = true; }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. The reparameterisation trick with a numeric gradient check
     ===================================================================== */
  AI.labs['gen-lt-reparam'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Reparameterisation: z = μ + σ ε, gradients through the sample', 'illustrative: estimate d/dμ and d/dσ of E[f(z)] from a growing set of noise draws and check against finite differences',
      `<canvas style="height:400px"></canvas>`,
      `<label>f(z) (edit, variable z)<input type="text" id="expr" value="(z-3)^2" style="width:100%"></label><div class="err" id="err" style="color:var(--red);font-size:12px"></div>
       ${sl('mu', 'Mean μ', -2, 4, 0.1, 1)}
       ${sl('sig', 'Std σ', 0.1, 2, 0.05, 0.5)}
       ${sl('rate', 'Noise draws per second', 0.5, 20, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, tick = AI.ticker(), NMAX = 300;
    const EPS = (() => { const r = AI.rng(7), a = []; for (let i = 0; i < NMAX; i++) a.push(AI.randn(r)); return a; })();
    const REF = (() => { const r = AI.rng(99), a = []; for (let i = 0; i < 20000; i++) a.push(AI.randn(r)); return a; })();
    let fn = (z) => (z - 3) ** 2, n = 0, hm = [], hs = [], sumM = 0, sumS = 0, ref = { m: 0, s: 0 }, dirty = true;
    const d = (z) => { const h = 1e-4 * (1 + Math.abs(z)); return (fn({ z: z + h }) - fn({ z: z - h })) / (2 * h); };
    const F = (z) => fn({ z });
    const recRef = () => { const mu = +q(ctl, '#mu').value, sg0 = +q(ctl, '#sig').value; let a = 0, b = 0; for (const e of REF) { const g1 = d(mu + sg0 * e); a += g1; b += g1 * e; } ref = { m: a / REF.length, s: b / REF.length }; };
    const reset = () => { n = 0; hm = []; hs = []; sumM = 0; sumS = 0; recRef(); dirty = true; };
    const add = () => {
      if (n >= NMAX) return;
      const mu = +q(ctl, '#mu').value, s0 = +q(ctl, '#sig').value, e = EPS[n], g1 = d(mu + s0 * e);
      sumM += g1; sumS += g1 * e; n++; hm.push(sumM / n); hs.push(sumS / n); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const mu = +q(ctl, '#mu').value, s0 = +q(ctl, '#sig').value, L = 44, R = W - 14, t0 = 24, t1 = Math.round(H * 0.5);
      const zl = Math.min(-1, mu - 3.2 * s0 - 0.5), zh = Math.max(5, mu + 3.2 * s0 + 0.5), X = (z) => L + (z - zl) / (zh - zl) * (R - L);
      const vals = []; for (let i = 0; i <= 120; i++) { const v = F(zl + (zh - zl) * i / 120); if (Number.isFinite(v)) vals.push(v); }
      let ylo = Math.min(...vals, 0), yhi = Math.max(...vals); if (yhi - ylo > 40) yhi = ylo + 40; if (yhi === ylo) yhi = ylo + 1;
      const Y = (v) => t1 - (clamp(v, ylo, yhi) - ylo) / (yhi - ylo) * (t1 - t0);
      txt(g, 'f(z); dots are draws z = μ + σε, red = slope used', L, 14, c.muted);
      axisLine(g, c, L, t1, R, t1); axisLine(g, c, L, t0, L, t1);
      for (let z = Math.ceil(zl); z <= zh; z++) txt(g, z, X(z), t1 + 13, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 160; i++) { const z = zl + (zh - zl) * i / 160, v = F(z); if (!Number.isFinite(v)) continue; i ? g.lineTo(X(z), Y(v)) : g.moveTo(X(z), Y(v)); } g.stroke(); g.lineWidth = 1;
      g.globalAlpha = 0.18; g.fillStyle = c.orange; g.beginPath(); for (let i = 0; i <= 120; i++) { const z = zl + (zh - zl) * i / 120, y = t1 - npdf(z, mu, s0) / (npdf(0, 0, s0) * 1.0) * (t1 - t0) * 0.35; i ? g.lineTo(X(z), y) : g.moveTo(X(z), y); } g.lineTo(R, t1); g.lineTo(L, t1); g.fill(); g.globalAlpha = 1;
      const from = Math.max(0, n - 14);
      for (let i = from; i < n; i++) {
        const z = mu + s0 * EPS[i], v = F(z), age = (i - from) / Math.max(1, n - from - 1);
        g.globalAlpha = 0.35 + 0.65 * age; g.fillStyle = c.orange; g.beginPath(); g.arc(X(z), Y(v), 4, 0, 7); g.fill();
        const sl2 = d(z), dz = (zh - zl) * 0.05; g.strokeStyle = c.red; g.beginPath(); g.moveTo(X(z - dz), Y(v - sl2 * dz)); g.lineTo(X(z + dz), Y(v + sl2 * dz)); g.stroke(); g.globalAlpha = 1;
      }
      g.fillStyle = c.fg; g.fillRect(X(mu) - 1, t1 - 6, 2, 12); txt(g, 'μ', X(mu), t1 + 26, c.fg, '11px sans-serif', 'center');
      // bottom: running estimates
      const b0 = t1 + 46, b1 = H - 26, nm = Math.max(20, Math.ceil(n / 20) * 20);
      const all = [...hm, ...hs, ref.m, ref.s]; let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-6) { lo -= 1; hi += 1; } const pad = (hi - lo) * 0.15; lo -= pad; hi += pad;
      const XX = (k) => L + k / nm * (R - L), YY = (v) => b1 - (v - lo) / (hi - lo) * (b1 - b0);
      txt(g, 'running estimates (dashed: 20 000-draw reference)', L, b0 - 8, c.muted);
      axisLine(g, c, L, b1, R, b1); axisLine(g, c, L, b0, L, b1); txt(g, 'draws n = ' + n, R, H - 6, c.muted, '11px sans-serif', 'right');
      txt(g, f3(hi).slice(0, 6), L - 4, b0 + 8, c.muted, '10px sans-serif', 'right'); txt(g, f3(lo).slice(0, 6), L - 4, b1, c.muted, '10px sans-serif', 'right');
      [[ref.m, c.green, hm, 'd/dμ'], [ref.s, c.purple, hs, 'd/dσ']].forEach(([r, cc, h, nme]) => {
        g.strokeStyle = cc; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(L, YY(r)); g.lineTo(R, YY(r)); g.stroke(); g.setLineDash([]);
        g.lineWidth = 2; g.beginPath(); h.forEach((v, i) => (i ? g.lineTo(XX(i + 1), YY(v)) : g.moveTo(XX(1), YY(v)))); g.stroke(); g.lineWidth = 1; txt(g, nme, R - 4, YY(r) - 4, cc, 'bold 11px sans-serif', 'right');
      });
      // finite difference of the sample mean with the same noise
      let fd = [NaN, NaN];
      if (n) {
        const h = 1e-4, mean = (m2, s2) => { let a = 0; for (let i = 0; i < n; i++) a += F(m2 + s2 * EPS[i]); return a / n; };
        fd = [(mean(mu + h, s0) - mean(mu - h, s0)) / (2 * h), (mean(mu, s0 + h) - mean(mu, s0 - h)) / (2 * h)];
      }
      q(ctl, '#out').innerHTML = `n = ${n} draws<br>pathwise ∂/∂μ = mean f′(z) = <b>${n ? f3(sumM / n) : '–'}</b>   finite difference on the same draws = ${f3(fd[0])}<br>` +
        `pathwise ∂/∂σ = mean f′(z)·ε = <b>${n ? f3(sumS / n) : '–'}</b>   finite difference = ${f3(fd[1])}<br>reference (20 000 draws): ∂/∂μ ≈ ${f3(ref.m)}, ∂/∂σ ≈ ${f3(ref.s)}`;
      q(ctl, '#cap').textContent = n < 3 ? 'Press play: each new draw ε becomes z = μ + σε, and the slope of f at that z is averaged. The estimate wobbles at first.' :
        'The pathwise estimate equals the finite-difference estimate on the same draws (the chain rule is exact for them) and both settle onto the reference as n grows. The gap between a few draws and the reference is Monte Carlo noise, not a bug.';
      showVals(ctl, { mu: (v) => v.toFixed(1), sig: (v) => v.toFixed(2), rate: (v) => v });
    };
    const ei = AI.exprInput(q(ctl, '#expr'), q(ctl, '#err'), { allowed: ['z'], check: (f2) => { f2({ z: 1 }); }, onChange: (f2) => { fn = f2; reset(); } });
    wire(ctl, st, add, reset, draw);
    q(ctl, '#mu').oninput = reset; q(ctl, '#sig').oninput = reset; q(ctl, '#rate').oninput = () => { dirty = true; };
    ei.run(); o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < NMAX) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) add(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. beta-VAE in the linear-Gaussian case: rate, distortion and collapse
     ===================================================================== */
  AI.labs['gen-lt-beta'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The β knob: rate against distortion, and collapse', 'illustrative: linear-Gaussian VAE with the best encoder for each β (closed form); six data points near −2 and +2',
      `<canvas style="height:520px"></canvas>`,
      `${sl('lb', 'log₁₀ β (weight on the KL term)', -1, 1.5, 0.05, 0)}
       ${sl('w', 'Decoder weight w', 0.5, 2, 0.1, 1)}
       ${sl('s', 'Decoder noise s', 0.3, 1, 0.05, 0.5)}
       ${sl('speed', 'Sweep speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const X0 = [-2.2, -2.0, -1.8, 1.8, 2.0, 2.2], st = { playing: true };
    let dirty = true, ph = 0;
    const enc = (b, w, s, x) => { const s2 = 1 / (1 + w * w / (b * s * s)), mu = (w * x / (s * s)) / (b + w * w / (s * s)); return { s2, mu, kl: 0.5 * (mu * mu + s2 - 1 - ln(s2)), mse: (x - w * mu) ** 2 }; };
    const avg = (b, w, s) => { let k = 0, m = 0; for (const x of X0) { const e = enc(b, w, s, x); k += e.kl; m += e.mse; } return { kl: k / X0.length, mse: m / X0.length }; };
    const reset = () => { ph = 0.2; q(ctl, '#lb').value = 0; st.setPlaying(true); dirty = true; };
    const stepFn = () => { const el = q(ctl, '#lb'); el.value = clamp(+el.value + 0.25, -1, 1.5); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const b = Math.pow(10, +q(ctl, '#lb').value), w = +q(ctl, '#w').value, s = +q(ctl, '#s').value;
      const ac = avg(b, w, s), ex = enc(b, w, s, 2);
      const L1 = 16, R1 = W - 16, zl = -3, zh = 3, Xz = (z) => L1 + (z - zl) / (zh - zl) * (R1 - L1);
      txt(g, 'latent space: one posterior per data point', L1, 14, c.muted);
      const t1 = 138, ymax = Math.max(1, ...X0.map((x) => 1 / (Math.sqrt(enc(b, w, s, x).s2) * 2.5066)));
      axisLine(g, c, L1, t1, R1, t1); for (let z = -2; z <= 2; z += 1) txt(g, z, Xz(z), t1 + 13, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.muted; g.setLineDash([3, 4]); g.beginPath(); for (let i = 0; i <= 120; i++) { const z = zl + 6 * i / 120, y = t1 - npdf(z, 0, 1) / ymax * (t1 - 28); i ? g.lineTo(Xz(z), y) : g.moveTo(Xz(z), y); } g.stroke(); g.setLineDash([]);
      X0.forEach((x) => { const e = enc(b, w, s, x), col = x < 0 ? c.blue : c.orange; g.strokeStyle = col; g.globalAlpha = 0.8; g.beginPath(); for (let i = 0; i <= 100; i++) { const z = zl + 6 * i / 100, y = t1 - npdf(z, e.mu, Math.sqrt(e.s2)) / ymax * (t1 - 28); i ? g.lineTo(Xz(z), y) : g.moveTo(Xz(z), y); } g.stroke(); g.globalAlpha = 1; });
      txt(g, 'dashed = prior', R1, 14, c.muted, '11px sans-serif', 'right');
      const t2 = 262, Xx = (v) => L1 + (v - zl) / (zh - zl) * (R1 - L1), yA = 188, yB = 238;
      txt(g, 'data (circles) and decoded mean codes (diamonds)', L1, 168, c.muted); axisLine(g, c, L1, t2, R1, t2);
      for (let v = -2; v <= 2; v++) txt(g, v, Xx(v), t2 + 13, c.muted, '11px sans-serif', 'center');
      X0.forEach((x, i) => { const e = enc(b, w, s, x), dx = w * e.mu, col = x < 0 ? c.blue : c.orange, yy = yA + (i % 3) * 8, y2 = yB + (i % 3) * 5;
        g.strokeStyle = col; g.globalAlpha = 0.6; g.beginPath(); g.moveTo(Xx(x), yy); g.lineTo(Xx(dx), y2); g.stroke(); g.globalAlpha = 1;
        g.fillStyle = col; g.beginPath(); g.arc(Xx(x), yy, 4, 0, 7); g.fill();
        g.beginPath(); const px = Xx(dx); g.moveTo(px, y2 - 5); g.lineTo(px + 5, y2); g.lineTo(px, y2 + 5); g.lineTo(px - 5, y2); g.closePath(); g.fill(); });
      g.strokeStyle = c.fg; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(Xx(0), 178); g.lineTo(Xx(0), t2); g.stroke(); g.setLineDash([]); txt(g, 'average', Xx(0) + 4, 182, c.muted, '10px sans-serif');
      const b0 = 322, b1 = H - 26, L3 = 52, R3 = W - 20;
      const pts = []; for (let i = 0; i <= 60; i++) { const bb = Math.pow(10, -1 + 2.5 * i / 60), a2 = avg(bb, w, s); pts.push([a2.kl, a2.mse]); }
      const kmax = Math.max(...pts.map((p) => p[0])), dmax = Math.max(...pts.map((p) => p[1])), XK = (v) => L3 + v / kmax * (R3 - L3), YD = (v) => b1 - v / dmax * (b1 - b0);
      txt(g, 'rate (average KL) against distortion (average error)', L3, b0 - 10, c.muted);
      axisLine(g, c, L3, b1, R3, b1); axisLine(g, c, L3, b0, L3, b1);
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(XK(p[0]), YD(p[1])) : g.moveTo(XK(p[0]), YD(p[1])))); g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.red; g.beginPath(); g.arc(XK(ac.kl), YD(ac.mse), 6, 0, 7); g.fill();
      txt(g, 'rate (nats)', R3, H - 6, c.muted, '11px sans-serif', 'right'); txt(g, 'error', L3 - 6, b0 + 8, c.muted, '11px sans-serif', 'right');
      q(ctl, '#out').innerHTML = `β = ${b.toFixed(2)}: best posterior variance σ² = 1/(1 + w²/(β s²)) = <b>${f3(ex.s2)}</b> (prior variance is 1)<br>` +
        `for x = 2: mean code μ = ${f3(ex.mu)}, decoded w·μ = <b>${f3(w * ex.mu)}</b>, KL = ${f3(ex.kl)}, squared error ${f3(ex.mse)}<br>average over the six points: rate ${f3(ac.kl)} nats, distortion ${f3(ac.mse)}`;
      q(ctl, '#cap').textContent = b < 0.6 ? 'Small β: the code is cheap to send, so each point keeps its own position in latent space; the decoded diamonds sit on the data. High rate, low error.' :
        (ac.kl < 0.2 ? 'Large β: the KL term dominates. The posteriors all sit on top of the prior, the code carries almost nothing, and every point decodes to nearly the same blurry value near the average of the clumps.' : 'Middle: the posteriors are widening toward the prior and the decoded points are pulled inward toward the average. Less information in z, more blur.');
      showVals(ctl, { lb: (v) => '(β = ' + Math.pow(10, v).toFixed(2) + ')', w: (v) => v.toFixed(1), s: (v) => v.toFixed(2), speed: (v) => v + '×' });
    };
    wire(ctl, st, stepFn, reset, draw);
    ['#lb', '#w', '#s'].forEach((id) => (q(ctl, id).oninput = () => { if (id === '#lb') st.setPlaying(false); dirty = true; }));
    q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { ph += dt * 0.12 * +q(ctl, '#speed').value; const t = ph % 2, tri = t < 1 ? t : 2 - t; q(ctl, '#lb').value = (-1 + 2.5 * ease(tri)).toFixed(3); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5. VQ-VAE codebook: assign, then move; dead codes
     ===================================================================== */
  AI.labs['gen-lt-vq'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A VQ codebook: snap each vector to its nearest code', 'illustrative: 2-D stand-ins for encoder outputs; the codebook update here is the k-means-style mean (many VQ-VAEs use a moving average)',
      `<canvas style="height:380px"></canvas>`,
      `${sl('K', 'Codebook size K', 1, 8, 1, 3)}
       ${sl('speed', 'Speed', 0.25, 2, 0.25, 0.5)}
       <div class="row"><button class="btn" id="seed">New starting codes</button></div>
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    const pts = (() => { const r = AI.rng(5), cs = [[-1.5, -0.8], [1.2, 1.0], [0.7, -1.4]], a = []; for (let i = 0; i < 60; i++) { const k = i % 3; a.push([cs[k][0] + 0.38 * AI.randn(r), cs[k][1] + 0.38 * AI.randn(r)]); } return a; })();
    let seed = 1, codes, from, to, asg, phase, pt, iter, done, dirty = true;
    const initCodes = () => { const r = AI.rng(100 + seed), K = +q(ctl, '#K').value; codes = []; for (let k = 0; k < K; k++) codes.push([-2 + 4 * r(), -2 + 4 * r()]); };
    const assign = () => { asg = pts.map((p) => { let b = 0, bd = 1e9; codes.forEach((cc, k) => { const d = (p[0] - cc[0]) ** 2 + (p[1] - cc[1]) ** 2; if (d < bd) { bd = d; b = k; } }); return b; }); };
    const targets = () => codes.map((cc, k) => { const m = pts.filter((_, i) => asg[i] === k); return m.length ? [m.reduce((s, p) => s + p[0], 0) / m.length, m.reduce((s, p) => s + p[1], 0) / m.length] : cc.slice(); });
    const reset = () => { initCodes(); assign(); phase = 0; pt = 0; iter = 0; done = false; from = null; to = null; st.setPlaying(true); dirty = true; };
    const finishPhase = () => {
      if (phase === 0) { from = codes.map((x) => x.slice()); to = targets(); phase = 1; pt = 0; }
      else {
        const moved = Math.max(...codes.map((cc, k) => Math.hypot(to[k][0] - from[k][0], to[k][1] - from[k][1])));
        codes = to.map((x) => x.slice()); const old = asg.join(); assign(); iter++; phase = 0; pt = 0;
        if (moved < 1e-6 && old === asg.join()) done = true;
      }
      dirty = true;
    };
    const advance = (dt) => {
      if (done) return;
      pt += dt * +q(ctl, '#speed').value / 1.6;
      if (phase === 1) { const e = ease(Math.min(1, pt)); codes = codes.map((_, k) => [from[k][0] + (to[k][0] - from[k][0]) * e, from[k][1] + (to[k][1] - from[k][1]) * e]); }
      if (pt >= 1) finishPhase();
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pal = [c.blue, c.orange, c.green, c.purple, c.red, c.teal, c.yellow, c.pink], S = Math.min(W - 20, H - 24) / 4.6, cx = W / 2, cy = H / 2 + 6;
      const X = (v) => cx + v * S, Y = (v) => cy - v * S;
      g.strokeStyle = c.border; g.strokeRect(X(-2.3), Y(2.3), 4.6 * S, 4.6 * S);
      txt(g, `iteration ${iter}, phase: ${done ? 'converged' : phase === 0 ? '① assign each vector to its nearest code' : '② move each code to the mean of its vectors'}`, 12, 16, c.muted);
      const a = phase === 0 ? Math.min(1, pt * 1.5) : 1;
      pts.forEach((p, i) => { const k = asg[i]; g.strokeStyle = pal[k % 8]; g.globalAlpha = 0.35 * a; g.beginPath(); g.moveTo(X(p[0]), Y(p[1])); g.lineTo(X(codes[k][0]), Y(codes[k][1])); g.stroke(); g.globalAlpha = 1; g.fillStyle = pal[k % 8]; g.beginPath(); g.arc(X(p[0]), Y(p[1]), 3.5, 0, 7); g.fill(); });
      const cnt = new Array(codes.length).fill(0); asg.forEach((k) => cnt[k]++);
      codes.forEach((cc, k) => { const x = X(cc[0]), y = Y(cc[1]); g.fillStyle = c.bg; g.strokeStyle = cnt[k] ? pal[k % 8] : c.red; g.lineWidth = 3; g.fillRect(x - 8, y - 8, 16, 16); g.strokeRect(x - 8, y - 8, 16, 16); g.lineWidth = 1; txt(g, 'e' + (k + 1) + (cnt[k] ? '' : ' unused'), x + 11, y - 9, cnt[k] ? c.fg : c.red, 'bold 11px sans-serif'); });
      const mse = pts.reduce((s, p, i) => s + (p[0] - codes[asg[i]][0]) ** 2 + (p[1] - codes[asg[i]][1]) ** 2, 0) / pts.length;
      const used = cnt.filter((v) => v > 0).length, ent = -cnt.filter((v) => v > 0).reduce((s, v) => s + v / pts.length * ln(v / pts.length), 0);
      q(ctl, '#out').innerHTML = `quantisation error (mean squared distance to the chosen code) = <b>${f3(mse)}</b><br>codes used: ${used} of ${codes.length}; code usage counts: ${cnt.join(', ')}<br>effective number of codes exp(entropy of usage) = ${f3(Math.exp(ent))}`;
      q(ctl, '#cap').textContent = done ? 'Converged: assignments no longer change. Any code marked "unused" is a dead code; only the used ones carry information. Try a new start or a smaller K.' :
        (phase === 0 ? 'Every vector is replaced by the code nearest to it: that choice is the discrete token the decoder would receive.' : 'Each code moves to the average of the vectors that chose it, which lowers the quantisation error (it can only go down or stay).');
      showVals(ctl, { K: (v) => v, speed: (v) => v + '×' });
    };
    wire(ctl, st, () => { if (!done) finishPhase(); if (phase === 1 && !done) { /* leave tween to run */ } }, reset, draw);
    q(ctl, '#K').oninput = reset; q(ctl, '#seed').onclick = () => { seed++; reset(); };
    q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing) advance(dt); if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     6. GAN on four bins: optimal discriminator and JS
     ===================================================================== */
  AI.labs['gen-lt-gan-discrete'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The optimal discriminator on four values', 'illustrative: D is a table of four probabilities; watch it climb to p_data / (p_data + p_g) and the value V to −ln 4 + 2·JS',
      `<canvas style="height:380px"></canvas>`,
      `<label>Data distribution p_data <select id="dp"><option value="1,4,4,1">middle-heavy (0.1, 0.4, 0.4, 0.1)</option><option value="1,1,1,1">flat</option><option value="1,2,3,4">rising</option><option value="6,2,1,1">left-heavy</option></select></label>
       <label>Generator distribution p_g <select id="gp"><option value="4,1,1,4">edge-heavy (0.4, 0.1, 0.1, 0.4)</option><option value="1,1,1,1">flat</option><option value="4,3,2,1">falling</option><option value="1,1,1,6">right-heavy</option></select></label>
       ${sl('lam', 'Generator quality λ (0 = as set, 1 = equals the data)', 0, 1, 0.05, 0)}
       ${sl('speed', 'Discriminator speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let al = [0, 0, 0, 0], hist = [], acc = 0, dirty = true;
    const dist = () => {
      const nz = (p) => { const t = p.reduce((s, v) => s + v, 0); return p.map((v) => v / t); };
      const pd = nz(q(ctl, '#dp').value.split(',').map(Number)), g0 = nz(q(ctl, '#gp').value.split(',').map(Number)), l = +q(ctl, '#lam').value;
      return { pd, pg: g0.map((v, i) => (1 - l) * v + l * pd[i]) };
    };
    const V = (pd, pg, D) => pd.reduce((s, p, i) => s + p * ln(D[i]) + pg[i] * ln(1 - D[i]), 0);
    const reset = () => { al = [0, 0, 0, 0]; hist = []; acc = 0; st.setPlaying(true); dirty = true; };
    const stepFn = (dt) => { const { pd, pg } = dist(); al = al.map((a, i) => a + dt * (ln(pd[i] / pg[i]) - a)); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const { pd, pg } = dist(), D = al.map(sg), Ds = pd.map((p, i) => p / (p + pg[i])), v = V(pd, pg, D), js = JS(pd, pg), vs = -ln(4) + 2 * js;
      const L = 40, R = W - 14, top = 26, mid = Math.round(H * 0.58), gw = (R - L) / 4, mx = Math.max(...pd, ...pg, 0.5) * 1.1;
      txt(g, 'blue p_data, orange p_g; diamond D, ring D*', L, 14, c.muted);
      axisLine(g, c, L, mid, R, mid);
      for (let i = 0; i < 4; i++) {
        const x0 = L + i * gw + gw * 0.12, bw = gw * 0.26;
        g.fillStyle = c.blue; g.globalAlpha = 0.8; g.fillRect(x0, mid - pd[i] / mx * (mid - top), bw, pd[i] / mx * (mid - top));
        g.fillStyle = c.orange; g.fillRect(x0 + bw + 2, mid - pg[i] / mx * (mid - top), bw, pg[i] / mx * (mid - top)); g.globalAlpha = 1;
        const dx = x0 + bw * 2 + 14, yD = (d) => mid - d * (mid - top);
        g.strokeStyle = c.green; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(dx, yD(Ds[i])); g.lineTo(dx, yD(D[i])); g.stroke(); g.setLineDash([]);
        g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); g.arc(dx, yD(Ds[i]), 6, 0, 7); g.stroke(); g.lineWidth = 1;
        g.fillStyle = c.green; g.beginPath(); const dy = yD(D[i]); g.moveTo(dx, dy - 7); g.lineTo(dx + 7, dy); g.lineTo(dx, dy + 7); g.lineTo(dx - 7, dy); g.fill();
        txt(g, 'x=' + (i + 1), x0 + gw * 0.3, mid + 14, c.muted, '11px sans-serif', 'center');
        txt(g, D[i].toFixed(2) + ' → ' + Ds[i].toFixed(2), x0 + gw * 0.3, mid + 28, c.fg, '10px sans-serif', 'center');
      }
      // bottom: V over time
      const b0 = mid + 52, b1 = H - 18; if (acc > 0.1 || !hist.length) { hist.push(v); acc = 0; if (hist.length > 300) hist.shift(); }
      const lo = Math.min(-ln(4) - 0.15, ...hist), hi = Math.max(vs + 0.15, ...hist), Yv = (t) => b1 - (t - lo) / (hi - lo) * (b1 - b0), Xt = (k) => L + k / 300 * (R - L);
      txt(g, 'value V(D, G) over time (D is climbing)', L, b0 - 6, c.muted);
      axisLine(g, c, L, b1, R, b1);
      g.setLineDash([5, 4]); g.strokeStyle = c.green; g.beginPath(); g.moveTo(L, Yv(vs)); g.lineTo(R, Yv(vs)); g.stroke(); txt(g, 'V* = −ln 4 + 2 JS = ' + f3(vs), R, Yv(vs) - 4, c.green, '11px sans-serif', 'right');
      g.strokeStyle = c.muted; g.beginPath(); g.moveTo(L, Yv(-ln(4))); g.lineTo(R, Yv(-ln(4))); g.stroke(); g.setLineDash([]); txt(g, '−ln 4 (D = ½ everywhere)', R, Yv(-ln(4)) + 12, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); hist.forEach((t, i) => (i ? g.lineTo(Xt(i), Yv(t)) : g.moveTo(Xt(0), Yv(t)))); g.stroke(); g.lineWidth = 1;
      q(ctl, '#out').innerHTML = `V(D,G) = Σ p_data ln D + p_g ln(1−D) = <b>${f3(v)}</b><br>optimal D* = p_data/(p_data+p_g); V(D*,G) = <b>${f3(V(pd, pg, Ds))}</b><br>JS(p_data‖p_g) = ${f3(js)}, so −ln 4 + 2·JS = <b>${f3(vs)}</b>`;
      const done = Math.max(...D.map((d2, i) => Math.abs(d2 - Ds[i]))) < 0.01;
      q(ctl, '#cap').textContent = done ? (js < 0.002 ? 'The generator matches the data: D* is ½ everywhere and the best the discriminator can reach is −ln 4. Nothing is left to learn.' : 'D has reached the optimum D*. The height of the plateau, −ln 4 + 2·JS, measures how different the two distributions are.') : 'D climbs V by moving each of its four probabilities toward p_data/(p_data+p_g). Drag the generator sliders to see the target move.';
      showVals(ctl, { lam: (t) => t.toFixed(2), speed: (t) => t + '×' });
    };
    wire(ctl, st, () => stepFn(0.4), reset, draw);
    [...root.querySelectorAll('input[type=range]')].forEach((e) => { e.oninput = () => { if (e.id === 'speed') { dirty = true; return; } hist = []; dirty = true; }; });
    ['#dp', '#gp'].forEach((id) => (q(ctl, id).onchange = () => { hist = []; dirty = true; }));
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const { pd, pg } = dist(); if (al.some((a, i) => Math.abs(a - ln(pd[i] / pg[i])) > 1e-3)) { stepFn(dt * 0.35 * +q(ctl, '#speed').value); acc += dt; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. Saturating versus non-saturating generator loss
     ===================================================================== */
  AI.labs['gen-lt-nonsat'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Minimax versus non-saturating generator loss', 'illustrative: a = discriminator logit on a fake sample, D = σ(a); two generators start at the same a and follow different gradients',
      `<canvas style="height:400px"></canvas>`,
      `${sl('a0', 'Starting logit a (very negative = D is sure it is fake)', -8, 2, 0.25, -5)}
       ${sl('speed', 'Learning rate', 0.5, 5, 0.5, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let am, an, t = 0, dirty = true;
    const reset = () => { am = an = +q(ctl, '#a0').value; t = 0; st.setPlaying(true); dirty = true; };
    const stepFn = (dt) => { const k = +q(ctl, '#speed').value; am += dt * k * sg(am); an += dt * k * (1 - sg(an)); t += dt; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 44, R = W - 14, al = -8, ah = 8, X = (a) => L + (a - al) / (ah - al) * (R - L);
      const panel = (y0, y1, title, f1, f2, lo, hi, n1, n2) => {
        const Y = (v) => y1 - (clamp(v, lo, hi) - lo) / (hi - lo) * (y1 - y0);
        txt(g, title, L, y0 - 8, c.muted); axisLine(g, c, L, y1, R, y1); axisLine(g, c, L, y0, L, y1);
        if (lo < 0 && hi > 0) { g.setLineDash([2, 3]); axisLine(g, c, L, Y(0), R, Y(0)); g.setLineDash([]); }
        [[f1, c.red, n1], [f2, c.green, n2]].forEach(([f, cc, nm]) => { g.strokeStyle = cc; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 160; i++) { const a = al + (ah - al) * i / 160; i ? g.lineTo(X(a), Y(f(a))) : g.moveTo(X(a), Y(f(a))); } g.stroke(); g.lineWidth = 1; });
        [[am, f1, c.red], [an, f2, c.green]].forEach(([a, f, cc]) => { g.fillStyle = cc; g.beginPath(); g.arc(X(clamp(a, al, ah)), Y(f(clamp(a, al, ah))), 6, 0, 7); g.fill(); });
        txt(g, hi, L - 6, y0 + 8, c.muted, '10px sans-serif', 'right'); txt(g, lo, L - 6, y1, c.muted, '10px sans-serif', 'right');
        g.fillStyle = c.red; g.fillRect(R - 150, y0 + 8, 8, 8); txt(g, 'minimax ' + n1, R - 138, y0 + 16, c.muted, '11px sans-serif'); g.fillStyle = c.green; g.fillRect(R - 150, y0 + 22, 8, 8); txt(g, 'non-saturating ' + n2, R - 138, y0 + 30, c.muted, '11px sans-serif');
      };
      const hh = Math.round(H * 0.4);
      panel(26, hh, 'generator loss against the logit a', (a) => ln(1 - sg(a)), (a) => -ln(sg(a)), -8, 8, 'ln(1−D)', '−ln D');
      panel(hh + 44, H - 22, 'slope of the loss with respect to a', (a) => -sg(a), (a) => -(1 - sg(a)), -1.1, 0.1, '−D', '−(1−D)');
      for (let a = -8; a <= 8; a += 4) txt(g, a, X(a), H - 6, c.muted, '11px sans-serif', 'center');
      
      q(ctl, '#out').innerHTML = `time ${t.toFixed(1)} s<br><span style="color:var(--red)">minimax</span>: a = ${f3(am)}, D = ${f3(sg(am))}, slope ${f3(-sg(am))}<br><span style="color:var(--green)">non-saturating</span>: a = ${f3(an)}, D = ${f3(sg(an))}, slope ${f3(-(1 - sg(an)))}`;
      q(ctl, '#cap').textContent = +q(ctl, '#a0').value < -2 && t < 4 ? 'Both start where D is confident the sample is fake. The red slope is almost zero (nothing to learn from), the green slope is near −1: the same goal, very different learning signal.' :
        'Once D is no longer sure the sample is fake, both losses give a useful slope. The difference matters early, when the generator is poor.';
      showVals(ctl, { a0: (v) => v, speed: (v) => v });
    };
    wire(ctl, st, () => stepFn(0.5), reset, draw);
    q(ctl, '#a0').oninput = reset; q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && (am < 8 || an < 8)) { stepFn(dt); dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     8. Game dynamics on min over x, max over y of x*y
     ===================================================================== */
  AI.labs['gen-lt-game'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Why GAN training can spiral: the game min over x, max over y of x·y', 'illustrative: the simplest two-player game, played by gradient steps; the only equilibrium is (0, 0)',
      `<canvas style="height:400px"></canvas>`,
      `<label>Update rule <select id="mode"><option value="sim">simultaneous gradient steps</option><option value="alt">alternating steps</option><option value="opt">optimistic (uses the previous gradient)</option></select></label>
       ${sl('eta', 'Step size η', 0.02, 0.3, 0.01, 0.1)}
       ${sl('rate', 'Steps per second', 1, 30, 1, 8)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, tick = AI.ticker();
    let x, y, px, py, path, t, dirty = true;
    const reset = () => { x = 1; y = 0; px = 1; py = 0; path = [[1, 0]]; t = 0; st.setPlaying(true); dirty = true; };
    const one = () => {
      const e = +q(ctl, '#eta').value, m = q(ctl, '#mode').value;
      if (m === 'sim') { const nx = x - e * y, ny = y + e * x; x = nx; y = ny; }
      else if (m === 'alt') { x = x - e * y; y = y + e * x; }
      else { const nx = x - 2 * e * y + e * py, ny = y + 2 * e * x - e * px; px = x; py = y; x = nx; y = ny; }
      t++; path.push([x, y]); if (path.length > 700) path.shift(); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R0 = Math.max(1.5, ...path.map((p) => Math.hypot(p[0], p[1]))) * 1.1, S = Math.min(W, H) / 2 / R0 - 8, cx = W / 2, cy = H / 2;
      const X = (v) => cx + v * S, Y = (v) => cy - v * S;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(8, cy); g.lineTo(W - 8, cy); g.moveTo(cx, 8); g.lineTo(cx, H - 8); g.stroke();
      g.setLineDash([4, 4]); g.beginPath(); g.arc(cx, cy, S, 0, 7); g.stroke(); g.setLineDash([]);
      txt(g, 'x (the minimiser) →', W - 10, cy - 6, c.muted, '11px sans-serif', 'right'); txt(g, 'y (the maximiser) ↑', cx + 6, 16, c.muted, '11px sans-serif');
      txt(g, 'dashed circle: the starting radius 1', 12, H - 10, c.muted, '11px sans-serif');
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); path.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.red; g.beginPath(); g.arc(X(x), Y(y), 6, 0, 7); g.fill(); g.fillStyle = c.fg; g.beginPath(); g.arc(cx, cy, 4, 0, 7); g.fill();
      const e = +q(ctl, '#eta').value, m = q(ctl, '#mode').value, r = Math.hypot(x, y);
      q(ctl, '#out').innerHTML = `step ${t}: (x, y) = (${f3(x)}, ${f3(y)}), distance from the equilibrium r = <b>${f3(r)}</b><br>` + (m === 'sim' ? `theory: r after t steps = (1 + η²)^(t/2) = ${f3(Math.pow(1 + e * e, t / 2))}` : m === 'alt' ? 'theory: the alternating map keeps the area, so r stays bounded and the orbit is a closed ellipse' : 'theory: the optimistic step contracts the orbit toward (0, 0) for small η');
      q(ctl, '#cap').textContent = m === 'sim' ? 'Each player steps at the same moment against the other. The point circles the equilibrium and every lap is wider than the last: gradient steps are not enough here.' :
        (m === 'alt' ? 'Alternating (y sees the new x) stops the growth but the point still circles forever: it neither converges nor diverges.' : 'Optimism corrects for the fact that the other player is about to move; the spiral now winds inward.');
      showVals(ctl, { eta: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, one, reset, draw);
    q(ctl, '#mode').onchange = reset; q(ctl, '#eta').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && Math.hypot(x, y) < 1e6) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) one(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     9. JS versus Wasserstein for two sliding blocks
     ===================================================================== */
  AI.labs['gen-lt-jsw'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Two sliding blocks: JS goes flat, the Wasserstein distance does not', 'illustrative: data uniform on [0, 1), generator uniform on [θ, θ+1); both distances computed on a fine grid and by formula',
      `<canvas style="height:400px"></canvas>`,
      `${sl('th', 'Offset θ between generator and data', 0, 3, 0.05, 3)}
       ${sl('speed', 'Sweep speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let ph = 0, dirty = true;
    const NB = 500, lo = -0.5, hi = 4.5, dx = (hi - lo) / NB;
    const num = (th) => {
      const p = [], r = []; for (let i = 0; i < NB; i++) { const x = lo + (i + 0.5) * dx; p.push(x >= 0 && x < 1 ? dx : 0); r.push(x >= th && x < th + 1 ? dx : 0); }
      const sp = p.reduce((s, v) => s + v, 0), sr = r.reduce((s, v) => s + v, 0); const P = p.map((v) => v / sp), Q = r.map((v) => v / sr);
      let cp = 0, cq = 0, w1 = 0; for (let i = 0; i < NB; i++) { cp += P[i]; cq += Q[i]; w1 += Math.abs(cp - cq) * dx; }
      return { js: JS(P, Q), w1 };
    };
    const reset = () => { ph = 0; q(ctl, '#th').value = 3; st.setPlaying(true); dirty = true; };
    const stepFn = () => { const e = q(ctl, '#th'); e.value = clamp(+e.value - 0.25, 0, 3); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const th = +q(ctl, '#th').value, L = 44, R = W - 16, xl = -0.5, xh = 4.5, X = (v) => L + (v - xl) / (xh - xl) * (R - L), t1 = Math.round(H * 0.3);
      txt(g, 'data density (blue) and generator density (orange)', L, 14, c.muted); axisLine(g, c, L, t1, R, t1);
      for (let v = 0; v <= 4; v++) txt(g, v, X(v), t1 + 13, c.muted, '11px sans-serif', 'center');
      g.globalAlpha = 0.55; g.fillStyle = c.blue; g.fillRect(X(0), t1 - 50, X(1) - X(0), 50); g.fillStyle = c.orange; g.fillRect(X(th), t1 - 50, X(1) - X(0), 50); g.globalAlpha = 1;
      const b0 = t1 + 46, b1 = H - 26, XT = (v) => L + v / 3 * (R - L), YD = (v) => b1 - v / 3 * (b1 - b0);
      txt(g, 'distance as a function of the offset θ', L, b0 - 10, c.muted); axisLine(g, c, L, b1, R, b1); axisLine(g, c, L, b0, L, b1);
      for (let v = 0; v <= 3; v++) { txt(g, v, XT(v), b1 + 13, c.muted, '11px sans-serif', 'center'); txt(g, v, L - 6, YD(v) + 4, c.muted, '10px sans-serif', 'right'); }
      g.lineWidth = 2; g.strokeStyle = c.red; g.beginPath(); g.moveTo(XT(0), YD(0)); g.lineTo(XT(1), YD(ln(2))); g.lineTo(XT(3), YD(ln(2))); g.stroke();
      g.strokeStyle = c.green; g.beginPath(); g.moveTo(XT(0), YD(0)); g.lineTo(XT(3), YD(3)); g.stroke(); g.lineWidth = 1;
      txt(g, 'JS = ln 2 · min(θ, 1)', XT(1.2), YD(ln(2)) - 8, c.red, 'bold 11px sans-serif'); txt(g, 'Wasserstein-1 = θ', XT(1.6), YD(1.6) - 4, c.green, 'bold 11px sans-serif');
      const nm = num(th); g.fillStyle = c.red; g.beginPath(); g.arc(XT(th), YD(nm.js), 6, 0, 7); g.fill(); g.fillStyle = c.green; g.beginPath(); g.arc(XT(th), YD(nm.w1), 6, 0, 7); g.fill();
      q(ctl, '#out').innerHTML = `θ = ${th.toFixed(2)}<br>JS: grid ${f3(nm.js)}, formula ln 2·min(θ,1) = <b>${f3(ln(2) * Math.min(th, 1))}</b> (ln 2 = 0.6931)<br>Wasserstein-1: grid ${f3(nm.w1)}, formula θ = <b>${f3(th)}</b>`;
      q(ctl, '#cap').textContent = th >= 1 ? 'No overlap at all: JS is stuck at ln 2 no matter how close the blocks are, so its slope with respect to θ is zero and says nothing about which way to move. The Wasserstein distance still reads the gap.' :
        'The blocks now overlap, JS finally falls (slope ln 2 per unit offset) and the two distances move together.';
      showVals(ctl, { th: (v) => v.toFixed(2), speed: (v) => v + '×' });
    };
    wire(ctl, st, stepFn, reset, draw);
    q(ctl, '#th').oninput = () => { st.setPlaying(false); dirty = true; }; q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { ph += dt * 0.08 * +q(ctl, '#speed').value; const t = ph % 2, tri = t < 1 ? t : 2 - t; q(ctl, '#th').value = (3 * (1 - ease(tri))).toFixed(3); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     10. Mode hopping: a greedy generator against a perfect discriminator
     ===================================================================== */
  AI.labs['gen-lt-modehop'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Mode hopping: a generator that chases the current best discriminator', 'illustrative abstraction: three separated modes; D is optimal for the current generator, G puts a fraction α of its mass on the mode where D* is highest',
      `<canvas style="height:390px"></canvas>`,
      `${sl('d1', 'Data weight, mode 1', 0.5, 8, 0.5, 5)}${sl('d2', 'Data weight, mode 2', 0.5, 8, 0.5, 3)}${sl('d3', 'Data weight, mode 3', 0.5, 8, 0.5, 2)}
       ${sl('alpha', 'Generator step α (share of mass moved each step)', 0.02, 0.95, 0.01, 0.6)}
       ${sl('rate', 'Steps per second', 0.25, 3, 0.25, 0.75)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, tick = AI.ticker();
    let pg, show, hist, pick, nstep, dirty = true, tw = 1;
    const pdist = () => { const w = [1, 2, 3].map((i) => +q(ctl, '#d' + i).value), t = w[0] + w[1] + w[2]; return w.map((v) => v / t); };
    const reset = () => { pg = [1 / 3, 1 / 3, 1 / 3]; show = pg.slice(); hist = [{ p: pg.slice(), k: -1 }]; pick = -1; nstep = 0; tw = 1; st.setPlaying(true); dirty = true; };
    const one = () => {
      const pd = pdist(), a = +q(ctl, '#alpha').value, D = pd.map((p, i) => p / (p + pg[i])); let k = 0; for (let i = 1; i < 3; i++) if (D[i] > D[k]) k = i;
      pg = pg.map((p, i) => (1 - a) * p + (i === k ? a : 0)); pick = k; nstep++; hist.push({ p: pg.slice(), k }); if (hist.length > 40) hist.shift(); tw = 0; dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pd = pdist(), pal = [c.blue, c.orange, c.green], L = 40, R = W - 14, top = 30, bot = 170;
      const Dn = pd.map((p, i) => p / (p + show[i]));
      txt(g, 'filled = generator, dashed = data, diamond = D*', 12, 14, c.muted); axisLine(g, c, L, bot, R, bot);
      const gw = (R - L) / 3;
      for (let i = 0; i < 3; i++) {
        const x0 = L + i * gw + gw * 0.25, bw = gw * 0.5;
        g.fillStyle = pal[i]; g.globalAlpha = 0.75; g.fillRect(x0, bot - show[i] * (bot - top), bw, show[i] * (bot - top)); g.globalAlpha = 1;
        g.strokeStyle = c.fg; g.setLineDash([4, 3]); g.strokeRect(x0, bot - pd[i] * (bot - top), bw, pd[i] * (bot - top)); g.setLineDash([]);
        const yd = bot - Dn[i] * (bot - top); g.fillStyle = c.fg; g.beginPath(); g.moveTo(x0 + bw / 2, yd - 7); g.lineTo(x0 + bw / 2 + 7, yd); g.lineTo(x0 + bw / 2, yd + 7); g.lineTo(x0 + bw / 2 - 7, yd); g.fill();
        txt(g, 'mode ' + (i + 1) + '   D* = ' + Dn[i].toFixed(2), x0 + bw / 2, bot + 14, c.muted, '11px sans-serif', 'center');
      }
      const hb = H - 26, ht = bot + 52, hl = L, hr = R, bw2 = (hr - hl) / 40;
      txt(g, 'generator mass per mode, last 40 steps (dot = mode chosen)', L, ht - 8, c.muted); axisLine(g, c, hl, hb, hr, hb);
      hist.forEach((h, k) => { let y = hb; const x = hl + k * bw2; h.p.forEach((v, i) => { const hh = v * (hb - ht); g.fillStyle = pal[i]; g.globalAlpha = 0.8; g.fillRect(x, y - hh, bw2 - 1, hh); y -= hh; }); g.globalAlpha = 1; if (h.k >= 0) { g.fillStyle = pal[h.k]; g.beginPath(); g.arc(x + bw2 / 2, hb + 8, 3, 0, 7); g.fill(); } });
      const js = JS(pd, pg), mn = Math.min(...pg.map((p, i) => p / pd[i]));
      q(ctl, '#out').innerHTML = `step ${nstep}: generator mass ${pg.map((v) => v.toFixed(2)).join(' / ')} vs data ${pd.map((v) => v.toFixed(2)).join(' / ')}<br>JS(data‖generator) = <b>${f3(js)}</b>; worst-covered mode has ${(mn * 100).toFixed(0)}% of its data share<br>mode chosen last: ${pick < 0 ? '–' : pick + 1}`;
      const a = +q(ctl, '#alpha').value;
      q(ctl, '#cap').textContent = nstep < 2 ? 'Each step: D* is computed for the current generator, then G moves mass to the mode D* likes best.' : (a > 0.4 ? 'Big steps: nearly all the mass lands on one mode, which makes D* turn against that mode and favour another. The mass hops around and never settles on the data mixture.' : 'Small steps: each correction is gentle, the mass drifts toward the data mixture and the hopping shrinks (the JS value falls and wobbles).');
      showVals(ctl, { d1: (v) => v, d2: (v) => v, d3: (v) => v, alpha: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, one, reset, draw);
    ['#d1', '#d2', '#d3'].forEach((id) => (q(ctl, id).oninput = () => { reset(); }));
    q(ctl, '#alpha').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) one(); }
      if (tw < 1 || show.some((v, i) => Math.abs(v - pg[i]) > 1e-3)) { tw = Math.min(1, tw + dt * 3); show = show.map((v, i) => v + (pg[i] - v) * Math.min(1, dt * 8)); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     11. Normalising flow in 1-D: change of variables
     ===================================================================== */
  AI.labs['gen-lt-flow1d'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A 1-D flow: warp a Gaussian and keep the total at 1', 'illustrative: z ~ N(0,1), x = f_t(z) with f_t = (1 − t)·z + t·f(z); density from p_x(x) = p_z(z) / |f′(z)|',
      `<canvas style="height:400px"></canvas>`,
      `<label>f(z) (edit; must be increasing)<input type="text" id="expr" value="z + 1.5*tanh(z)" style="width:100%"></label><div class="err" id="err" style="color:var(--red);font-size:12px"></div>
       <div class="row" id="presets"></div>
       ${sl('t', 'Morph t (0 = identity, 1 = f)', 0, 1, 0.01, 0)}
       ${sl('z0', 'Probe point z₀', -2.5, 2.5, 0.1, 1)}
       ${sl('speed', 'Morph speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, NG = 1201, ZL = -6, ZH = 6, NS = 4000;
    const Z = new Float64Array(NG).map((_, i) => ZL + (ZH - ZL) * i / (NG - 1));
    const SZ = (() => { const r = AI.rng(31), a = []; for (let i = 0; i < NS; i++) a.push(AI.randn(r)); return a; })();
    let fn = null, FG = null, DG = null, FS = null, bad = '', ph = 0, dirty = true;
    const f1 = (z) => fn({ z });
    const recompute = () => {
      FG = new Float64Array(NG); DG = new Float64Array(NG); bad = '';
      for (let i = 0; i < NG; i++) { FG[i] = f1(Z[i]); const h = 1e-4; DG[i] = (f1(Z[i] + h) - f1(Z[i] - h)) / (2 * h); if (!Number.isFinite(FG[i]) || !Number.isFinite(DG[i])) { bad = 'f is not defined everywhere on [−6, 6]'; break; } }
      if (!bad) { const lo = Math.min(...DG); if (lo <= 0) bad = 'f′(z) ≤ 0 somewhere (minimum ' + lo.toFixed(3) + '): f is not increasing, so it is not invertible'; }
      FS = bad ? null : SZ.map((z) => f1(z));
      dirty = true;
    };
    const reset = () => { ph = 0; q(ctl, '#t').value = 0; st.setPlaying(true); dirty = true; };
    const stepFn = () => { const e = q(ctl, '#t'); e.value = clamp(+e.value + 0.1, 0, 1); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      if (!FG || bad) { txt(g, bad || 'waiting for a valid f', 16, 30, c.red, '13px sans-serif'); q(ctl, '#out').textContent = ''; q(ctl, '#cap').textContent = 'The change-of-variables formula needs an invertible map: edit f so that it is strictly increasing.'; return; }
      const t = +q(ctl, '#t').value, z0 = +q(ctl, '#z0').value, L = 30, R = W - 16, t1 = Math.round(H * 0.62);
      const xs = new Float64Array(NG), ps = new Float64Array(NG); let mass = 0, xmin = 1e9, xmax = -1e9;
      for (let i = 0; i < NG; i++) { xs[i] = (1 - t) * Z[i] + t * FG[i]; ps[i] = phi(Z[i]) / ((1 - t) + t * DG[i]); }
      for (let i = 0; i < NG - 1; i++) mass += 0.5 * (ps[i] + ps[i + 1]) * (xs[i + 1] - xs[i]);
      const xl = Math.min(-4.5, xs[Math.round(NG * 0.04)] - 0.3), xh = Math.max(4.5, xs[Math.round(NG * 0.96)] + 0.3), X = (v) => L + (v - xl) / (xh - xl) * (R - L);
      let pk = 0.45; for (let i = 0; i < NG; i++) if (Math.abs(xs[i]) < 40) pk = Math.max(pk, ps[i] * 1.1);
      const Y = (d) => t1 - Math.min(d, pk) / pk * (t1 - 40);
      txt(g, 'density of x: samples (bars), formula (curve), N(0,1) dashed', L, 14, c.muted); axisLine(g, c, L, t1, R, t1);
      for (let v = Math.ceil(xl); v <= xh; v++) if (v % 2 === 0) txt(g, v, X(v), t1 + 13, c.muted, '11px sans-serif', 'center');
      const NBn = 60, cnt = new Array(NBn).fill(0), bw = (xh - xl) / NBn; for (let i = 0; i < NS; i++) { const v = (1 - t) * SZ[i] + t * FS[i], k = Math.floor((v - xl) / bw); if (k >= 0 && k < NBn) cnt[k]++; }
      g.fillStyle = c.blue; g.globalAlpha = 0.4; cnt.forEach((n, k) => { const d = n / NS / bw; g.fillRect(X(xl + k * bw) + 1, Y(d), X(xl + bw) - X(xl) - 2, t1 - Y(d)); }); g.globalAlpha = 1;
      g.strokeStyle = c.muted; g.setLineDash([3, 4]); g.beginPath(); for (let i = 0; i <= 160; i++) { const v = xl + (xh - xl) * i / 160; i ? g.lineTo(X(v), Y(phi(v))) : g.moveTo(X(v), Y(phi(v))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); let first = true; for (let i = 0; i < NG; i++) { if (xs[i] < xl || xs[i] > xh) continue; first ? g.moveTo(X(xs[i]), Y(ps[i])) : g.lineTo(X(xs[i]), Y(ps[i])); first = false; } g.stroke(); g.lineWidth = 1;
      // mapping ticks z -> x
      const tk = [-2, -1, 0, 1, 2], yT = t1 + 30, yB = t1 + 60;
      txt(g, 'z', 8, yT + 4, c.muted, '11px sans-serif'); txt(g, 'x', 8, yB + 4, c.muted, '11px sans-serif'); axisLine(g, c, L, yT, R, yT); axisLine(g, c, L, yB, R, yB);
      const fx = (z) => { const k = Math.round((z - ZL) / (ZH - ZL) * (NG - 1)); return xs[clamp(k, 0, NG - 1)]; };
      tk.forEach((z) => { g.strokeStyle = c.purple; g.beginPath(); g.moveTo(X(z), yT); g.lineTo(X(fx(z)), yB); g.stroke(); txt(g, z, X(z), yT - 3, c.muted, '10px sans-serif', 'center'); });
      const k0 = Math.round((z0 - ZL) / (ZH - ZL) * (NG - 1)), x0 = xs[k0], d0 = (1 - t) + t * DG[k0];
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(X(x0), t1); g.lineTo(X(x0), Y(ps[k0])); g.stroke(); g.fillStyle = c.red; g.beginPath(); g.arc(X(x0), Y(ps[k0]), 5, 0, 7); g.fill();
      q(ctl, '#out').innerHTML = `integral of p_x over x (trapezoid) = <b>${mass.toFixed(6)}</b><br>probe z₀ = ${z0.toFixed(1)} → x = ${f3(x0)}, f_t′(z₀) = ${f3(d0)}, p_z(z₀) = ${f3(phi(z0))}<br>p_x(x) = p_z(z₀)/f_t′(z₀) = <b>${f3(phi(z0) / d0)}</b>`;
      q(ctl, '#cap').textContent = t < 0.02 ? 'At t = 0 the map is the identity and p_x is the standard normal.' : 'Where f stretches the axis (f′ > 1, ticks spread apart) the density is thinned; where it squeezes (f′ < 1) the density is piled up. The total area stays 1 at every t.';
      showVals(ctl, { t: (v) => v.toFixed(2), z0: (v) => v.toFixed(1), speed: (v) => v + '×' });
    };
    const ei = AI.exprInput(q(ctl, '#expr'), q(ctl, '#err'), { allowed: ['z'], check: (f2) => { f2({ z: 1 }); }, onChange: (f2) => { fn = f2; recompute(); } });
    [['tanh bump', 'z + 1.5*tanh(z)'], ['exp (lognormal)', 'exp(z)'], ['affine', '2*z+1'], ['cubic', 'z + 0.15*z^3'], ['not invertible', 'z^2']].forEach(([k, v]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => ei.set(v); q(ctl, '#presets').appendChild(b); });
    wire(ctl, st, stepFn, reset, draw);
    q(ctl, '#t').oninput = () => { st.setPlaying(false); dirty = true; }; q(ctl, '#z0').oninput = () => { dirty = true; }; q(ctl, '#speed').oninput = () => { dirty = true; };
    ei.run(); o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { ph += dt * 0.1 * +q(ctl, '#speed').value; const t = ph % 2, tri = t < 1 ? t : 2 - t; q(ctl, '#t').value = ease(tri).toFixed(3); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     12. Affine coupling layer in 2-D
     ===================================================================== */
  AI.labs['gen-lt-coupling'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'An affine coupling layer: y₁ = x₁, y₂ = x₂·exp(s(x₁)) + t(x₁)', 'illustrative: 2-D Gaussian samples pushed through one coupling layer; s and t may be any functions of x₁ (edit them)',
      `<canvas style="height:400px"></canvas>`,
      `<label>s(u), the log-scale (variable u)<input type="text" id="se" value="0.5*u" style="width:100%"></label><div class="err" id="eerr" style="color:var(--red);font-size:12px"></div>
       <label>t(u), the shift (variable u)<input type="text" id="te" value="0.3*u^2" style="width:100%"></label><div class="err" id="terr" style="color:var(--red);font-size:12px"></div>
       ${sl('tau', 'Morph τ (0 = no layer, 1 = full layer)', 0, 1, 0.01, 0)}
       ${sl('px', 'Probe x₁', -2, 2, 0.1, 1)}${sl('py', 'Probe x₂', -2, 2, 0.1, 0.5)}
       ${sl('speed', 'Morph speed', 0.25, 3, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    const S = (() => { const r = AI.rng(41), a = []; for (let i = 0; i < 700; i++) a.push([AI.randn(r), AI.randn(r)]); return a; })();
    let sf = (u) => 0.5 * u.u, tf = (u) => 0.3 * u.u * u.u, ph = 0, dirty = true;
    const sv = (u) => { const v = sf({ u }); return Number.isFinite(v) ? clamp(v, -3, 3) : 0; }, tv = (u) => { const v = tf({ u }); return Number.isFinite(v) ? clamp(v, -6, 6) : 0; };
    const map = (x1, x2, tau) => [x1, x2 * Math.exp(tau * sv(x1)) + tau * tv(x1)];
    const reset = () => { ph = 0; q(ctl, '#tau').value = 0; st.setPlaying(true); dirty = true; };
    const stepFn = () => { const e = q(ctl, '#tau'); e.value = clamp(+e.value + 0.1, 0, 1); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const tau = +q(ctl, '#tau').value, px = +q(ctl, '#px').value, py = +q(ctl, '#py').value, Sc = Math.min(W, H) / 9, cx = W / 2, cy = H / 2 + 4;
      const X = (v) => cx + v * Sc, Y = (v) => cy - v * Sc;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(8, cy); g.lineTo(W - 8, cy); g.moveTo(cx, 8); g.lineTo(cx, H - 8); g.stroke();
      txt(g, 'x₁ stays; x₂ is scaled by exp(s) and shifted by t', 10, 14, c.muted);
      g.lineWidth = 1; g.globalAlpha = 0.5;
      for (let k = -3; k <= 3; k++) {
        g.strokeStyle = c.muted; g.beginPath(); for (let i = 0; i <= 60; i++) { const a = -4 + 8 * i / 60, p = map(a, k, tau); i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])); } g.stroke();
        g.beginPath(); for (let i = 0; i <= 60; i++) { const a = -4 + 8 * i / 60, p = map(k, a, tau); i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])); } g.stroke();
      }
      g.globalAlpha = 0.8; S.forEach((p) => { const m = map(p[0], p[1], tau); g.fillStyle = AI.mix(c.blue, c.orange, clamp((p[0] + 2) / 4, 0, 1)); g.beginPath(); g.arc(X(m[0]), Y(m[1]), 2.3, 0, 7); g.fill(); }); g.globalAlpha = 1;
      const m = map(px, py, tau); g.fillStyle = c.red; g.beginPath(); g.arc(X(m[0]), Y(m[1]), 6, 0, 7); g.fill(); g.strokeStyle = c.red; g.beginPath(); g.arc(X(px), Y(py), 6, 0, 7); g.stroke();
      // numeric jacobian determinant at the probe
      const h = 1e-5, a1 = map(px + h, py, tau), a2 = map(px - h, py, tau), b1 = map(px, py + h, tau), b2 = map(px, py - h, tau);
      const J = [[(a1[0] - a2[0]) / (2 * h), (b1[0] - b2[0]) / (2 * h)], [(a1[1] - a2[1]) / (2 * h), (b1[1] - b2[1]) / (2 * h)]], detN = J[0][0] * J[1][1] - J[0][1] * J[1][0], ls = tau * sv(px);
      const pdens = phi(px) * phi(py), pinv = pdens / Math.abs(detN), pform = phi(m[0]) * phi((m[1] - tau * tv(px)) / Math.exp(ls)) / Math.exp(ls);
      q(ctl, '#out').innerHTML = `probe x = (${px.toFixed(1)}, ${py.toFixed(1)}) → y = (${f3(m[0])}, ${f3(m[1])})<br>log-determinant: τ·s(x₁) = <b>${f3(ls)}</b>, numeric Jacobian determinant ${f3(detN)} vs exp(${f3(ls)}) = ${f3(Math.exp(ls))}<br>` +
        `density at y by p_x(x)/|det J| = <b>${f3(pinv)}</b>; by the inverse-map formula = ${f3(pform)}<br>(p_x(x) = ${f3(pdens)})`;
      q(ctl, '#cap').textContent = tau < 0.02 ? 'At τ = 0 the grid is square and the samples are the Gaussian. Press play to apply the layer.' : 'The Jacobian is triangular, so its determinant is just exp(s(x₁)): cheap, even though t and s can be any function of x₁. The red ring is the probe in x, the red dot where it lands.';
      showVals(ctl, { tau: (v) => v.toFixed(2), px: (v) => v.toFixed(1), py: (v) => v.toFixed(1), speed: (v) => v + '×' });
    };
    const mk = (id, eid, set) => AI.exprInput(q(ctl, id), q(ctl, eid), { allowed: ['u'], check: (f2) => { f2({ u: 1 }); }, onChange: (f2) => { set(f2); dirty = true; } });
    const e1 = mk('#se', '#eerr', (f2) => (sf = f2)), e2 = mk('#te', '#terr', (f2) => (tf = f2));
    wire(ctl, st, stepFn, reset, draw);
    ['#px', '#py', '#speed'].forEach((id) => (q(ctl, id).oninput = () => { dirty = true; })); q(ctl, '#tau').oninput = () => { st.setPlaying(false); dirty = true; };
    e1.run(); e2.run(); o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { ph += dt * 0.1 * +q(ctl, '#speed').value; const t = ph % 2, tri = t < 1 ? t : 2 - t; q(ctl, '#tau').value = ease(tri).toFixed(3); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
