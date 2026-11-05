/* Data Science Module 3 labs: statistics for data science.
   ds-st-sampling, ds-st-sampling-dist, ds-st-ci, ds-st-pvalue, ds-st-power, ds-st-bootstrap.
   All simulations are illustrative, use seeded random numbers (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const sdev = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, a.length - 1)); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

  /* ---- distribution helpers (Student t, normal) ---- */
  const lgamma = (x) => {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015;
    for (const k of c) s += k / ++y;
    return -t + Math.log(2.5066282746310005 * s / x);
  };
  const betacf = (a, b, x) => {
    const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d; let h = d;
    for (let m = 1; m < 300; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-14) break;
    }
    return h;
  };
  const ibeta = (x, a, b) => {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  };
  const tcdf = (t, v) => { const p = 0.5 * ibeta(v / (v + t * t), v / 2, 0.5); return t > 0 ? 1 - p : p; };
  const tinv = (p, v) => { let lo = 0, hi = 400; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (tcdf(m, v) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const tpdf = (t, v) => Math.exp(lgamma((v + 1) / 2) - lgamma(v / 2)) / Math.sqrt(v * Math.PI) * Math.pow(1 + t * t / v, -(v + 1) / 2);
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const Phiinv = (p) => { let lo = -8, hi = 8; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (Phi(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

  /* common wiring: play/pause, step, reset */
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };

  /* =====================================================================
     1. Sampling bias: random versus convenience samples of customers
     ===================================================================== */
  AI.labs['ds-st-sampling'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Random sample versus convenience sample', 'illustrative: 2 000 customers, 20% use the shop app; one customer is added to each sample at a time',
      `<canvas style="height:380px"></canvas>`,
      `${sl('reach', 'Chance a non-app customer can be reached by the convenience survey', 0, 1, 0.05, 0)}
       ${sl('rate', 'Customers added per second', 0.5, 40, 0.5, 4)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    // population: 80% non-app (typical spend about 28), 20% app users (typical spend about 65)
    const rp = AI.rng(11), N = 2000, pop = [];
    for (let i = 0; i < N; i++) { const app = rp() < 0.2; pop.push({ app, v: Math.exp((app ? 4.1 : 3.2) + (app ? 0.4 : 0.5) * AI.randn(rp)) }); }
    const trueMean = mean(pop.map((p) => p.v)), apps = pop.filter((p) => p.app), non = pop.filter((p) => !p.app);
    const appMean = mean(apps.map((p) => p.v));
    const XMAX = 200, NB = 40;
    const bins = [new Array(NB).fill(0), new Array(NB).fill(0)];
    pop.forEach((p) => { bins[p.app ? 1 : 0][Math.min(NB - 1, Math.floor(p.v / XMAX * NB))]++; });
    const st = { playing: true };
    let rs, n, sumR, sumC, hR, hC, dirty = true, tick = AI.ticker();
    const reset = () => { rs = AI.rng(5); n = 0; sumR = 0; sumC = 0; hR = []; hC = []; dirty = true; };
    const add = () => {
      const reach = +q(ctl, '#reach').value;
      sumR += pop[Math.floor(rs() * N)].v;
      let p; do { p = pop[Math.floor(rs() * N)]; } while (!p.app && rs() > reach);
      sumC += p.v; n++; hR.push(sumR / n); hC.push(sumC / n); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 46, R = W - 14, split = Math.round(H * 0.46);
      // top: population histogram
      const bt = 22, bb = split - 24, bh = bb - bt, mx = Math.max(...bins[0].map((v, i) => v + bins[1][i]));
      const xv = (v) => L + (Math.min(v, XMAX) / XMAX) * (R - L);
      txt(g, 'population: spend per customer (£)', L, 14, c.muted);
      const bw = (R - L) / NB;
      for (let i = 0; i < NB; i++) {
        const h0 = bins[0][i] / mx * bh, h1 = bins[1][i] / mx * bh;
        g.fillStyle = c.blue; g.globalAlpha = 0.7; g.fillRect(L + i * bw + 1, bb - h0, bw - 2, h0);
        g.fillStyle = c.orange; g.fillRect(L + i * bw + 1, bb - h0 - h1, bw - 2, h1); g.globalAlpha = 1;
      }
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, bb); g.lineTo(R, bb); g.stroke();
      for (const v of [0, 50, 100, 150, 200]) txt(g, v === 200 ? '200+' : v, xv(v), bb + 13, c.muted, '11px sans-serif', 'center');
      txt(g, 'blue = no app, orange = app user', R, 14, c.muted, '11px sans-serif', 'right');
      const vline = (v, col, label, dy, dash) => { g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.beginPath(); g.moveTo(xv(v), bt); g.lineTo(xv(v), bb); g.stroke(); g.setLineDash([]); g.lineWidth = 1; txt(g, label, xv(v) + 4, bt + dy, col, 'bold 11px sans-serif'); };
      vline(trueMean, c.fg, `true mean ${trueMean.toFixed(1)}`, 10, [4, 4]);
      if (n) { vline(sumR / n, c.green, `random ${(sumR / n).toFixed(1)}`, 24); vline(sumC / n, c.red, `convenience ${(sumC / n).toFixed(1)}`, 38); }
      // bottom: running means
      const t0 = split + 18, t1 = H - 26, xm = Math.max(40, Math.ceil(n / 40) * 40);
      const ymax = Math.max(100, ...hC, ...hR) * 1.05, X = (k) => L + (k / xm) * (R - L), Y = (v) => t1 - (v / ymax) * (t1 - t0);
      txt(g, 'running average as the sample grows (£)', L, split + 10, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      txt(g, '0', L - 6, t1 + 4, c.muted, '11px sans-serif', 'right'); txt(g, Math.round(ymax), L - 6, t0 + 8, c.muted, '11px sans-serif', 'right');
      txt(g, `sample size n = ${n}`, R, H - 8, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(L, Y(trueMean)); g.lineTo(R, Y(trueMean)); g.stroke(); g.setLineDash([]);
      const line = (h, col) => { if (!h.length) return; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); h.forEach((v, i) => (i ? g.lineTo(X(i + 1), Y(v)) : g.moveTo(X(1), Y(v)))); g.stroke(); g.lineWidth = 1; };
      line(hR, c.green); line(hC, c.red);
      q(ctl, '#out').textContent = n ? `n = ${n}: random sample mean £${(sumR / n).toFixed(1)}, convenience sample mean £${(sumC / n).toFixed(1)}, true mean of all ${N} customers £${trueMean.toFixed(1)}. The convenience sample is off by £${(sumC / n - trueMean).toFixed(1)}.` : 'Press play: customers are added one at a time to both samples.';
      q(ctl, '#cap').textContent = n < 5 ? 'Both samples start noisy.' : (+q(ctl, '#reach').value < 0.15 ? 'The random average (green) is settling onto the true mean. The convenience average (red) is settling too, but onto the wrong number: only app users are reachable, and they spend more. More data does not fix that.' : 'With non-app customers reachable some of the time, the convenience sample is pulled toward the truth but is still biased upward. Set the slider to 1 and it becomes a random sample.');
      showVals(ctl, { reach: (v) => Math.round(v * 100) + '%', rate: (v) => v });
    };
    const step = () => { if (n < 2000) add(); };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#reach').oninput = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < 2000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) add(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Sampling distribution of the mean and the CLT
     ===================================================================== */
  AI.labs['ds-st-sampling-dist'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The sampling distribution of the mean', 'illustrative: draw many samples of size n, keep each sample mean, and watch the histogram of means form',
      `<canvas style="height:400px"></canvas>`,
      `<label>Population shape <select id="shape"><option value="exp">skewed (exponential)</option><option value="logn">very skewed (lognormal)</option><option value="bi">two humps</option><option value="uni">flat (uniform)</option></select></label>
       ${sl('n', 'Sample size n', 1, 100, 1, 5)}
       ${sl('rate', 'Samples per second', 0.25, 40, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const make = (shape) => {
      const r = AI.rng(3), a = [];
      for (let i = 0; i < 20000; i++) {
        let v;
        if (shape === 'exp') v = -10 * Math.log(1 - r());
        else if (shape === 'logn') v = 6 * Math.exp(AI.randn(r));
        else if (shape === 'bi') v = r() < 0.5 ? 3 + 0.7 * AI.randn(r) : 9 + 0.7 * AI.randn(r);
        else v = 10 * r();
        a.push(v);
      }
      return a;
    };
    let pop, mu, sigma, lo, hi, NB = 40;
    const setPop = () => {
      pop = make(q(ctl, '#shape').value); mu = mean(pop); sigma = Math.sqrt(mean(pop.map((x) => (x - mu) ** 2)));
      const s = [...pop].sort((a, b) => a - b); lo = Math.min(0, s[0]); hi = s[Math.floor(0.995 * s.length)];
      lo = q(ctl, '#shape').value === 'bi' ? -1 : 0;
    };
    const st = { playing: true };
    let rs, means, binsM, cur, popBins, dirty = true, tick = AI.ticker();
    const binOf = (v) => Math.max(0, Math.min(NB - 1, Math.floor((v - lo) / (hi - lo) * NB)));
    const reset = () => {
      setPop(); rs = AI.rng(17); means = []; binsM = new Array(NB).fill(0); cur = null; dirty = true;
      popBins = new Array(NB).fill(0); pop.forEach((v) => popBins[binOf(v)]++);
    };
    const sample = () => {
      const n = +q(ctl, '#n').value, pts = [];
      for (let i = 0; i < n; i++) pts.push(pop[Math.floor(rs() * pop.length)]);
      return { pts, m: mean(pts), t: 0 };
    };
    const commit = (s) => { means.push(s.m); binsM[binOf(s.m)]++; };
    const step = () => { if (cur && cur.t < 1) commit(cur); cur = sample(); cur.t = 1; commit(cur); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 40, R = W - 14, split = Math.round(H * 0.42), n = +q(ctl, '#n').value, se = sigma / Math.sqrt(n);
      const xv = (v) => L + (v - lo) / (hi - lo) * (R - L), bw = (R - L) / NB;
      // top: population
      const bt = 22, bb = split - 22, bh = bb - bt, pm = Math.max(...popBins);
      txt(g, 'population (20 000 values) and the current sample', L, 14, c.muted);
      g.fillStyle = c.blue; g.globalAlpha = 0.45;
      popBins.forEach((v, i) => g.fillRect(L + i * bw + 1, bb - v / pm * bh, bw - 2, v / pm * bh)); g.globalAlpha = 1;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, bb); g.lineTo(R, bb); g.stroke();
      if (cur) {
        g.fillStyle = c.orange; cur.pts.forEach((v, i) => { const y = bb - 6 - ((i * 37) % 7) * 5; g.beginPath(); g.arc(xv(Math.min(Math.max(v, lo), hi)), y, 3, 0, 7); g.fill(); });
      }
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(xv(mu), bt); g.lineTo(xv(mu), bb); g.stroke(); g.setLineDash([]);
      txt(g, `μ = ${mu.toFixed(2)}, σ = ${sigma.toFixed(2)}`, R, 14, c.muted, '11px sans-serif', 'right');
      // bottom: sampling distribution
      const t0 = split + 22, t1 = H - 26, th = t1 - t0, k = means.length;
      txt(g, `sample means so far (${k}); curve = normal with mean μ and SD σ/√n = ${se.toFixed(3)}`, L, split + 12, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t1); g.lineTo(R, t1); g.stroke();
      const peakN = npdf(mu, mu, se) * (hi - lo) / NB;
      const topCount = Math.max(...binsM, peakN * Math.max(k, 1), 5);
      g.fillStyle = c.green; g.globalAlpha = 0.75;
      binsM.forEach((v, i) => g.fillRect(L + i * bw + 1, t1 - v / topCount * th, bw - 2, v / topCount * th)); g.globalAlpha = 1;
      if (k) { g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 160; i++) { const x = lo + (hi - lo) * i / 160, y = t1 - npdf(x, mu, se) * (hi - lo) / NB * k / topCount * th; i ? g.lineTo(xv(x), y) : g.moveTo(xv(x), y); } g.stroke(); g.lineWidth = 1; }
      for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4; txt(g, v.toFixed(1), xv(v), t1 + 14, c.muted, '11px sans-serif', 'center'); }
      if (cur && cur.t < 1) {
        const e = ease(cur.t), x = xv(Math.min(Math.max(cur.m, lo), hi)), y = bb + (t1 - 10 - bb) * e;
        g.fillStyle = c.orange; g.beginPath(); g.moveTo(x, y + 8); g.lineTo(x - 6, y - 4); g.lineTo(x + 6, y - 4); g.fill();
        txt(g, 'mean = ' + cur.m.toFixed(2), x + 9, y + 4, c.orange, 'bold 11px sans-serif');
      }
      const sd = k > 1 ? sdev(means) : NaN;
      q(ctl, '#out').textContent = `n = ${n}. Samples collected: ${k}. Mean of the sample means ${k ? mean(means).toFixed(3) : '-'} (μ = ${mu.toFixed(3)}). SD of the sample means ${k > 1 ? sd.toFixed(3) : '-'}; the formula σ/√n = ${sigma.toFixed(3)}/√${n} = ${se.toFixed(3)}.`;
      q(ctl, '#cap').textContent = n < 3 ? 'With n = 1 or 2 the sample means look like the population itself, skew and all.' : n < 15 ? 'Each orange dot is one value drawn; the triangle is that sample\'s mean, falling into the green histogram. The histogram is already narrower than the population.' : 'Whatever the population looked like, the means now pile up in a bell shape, centred on μ with spread σ/√n.';
      showVals(ctl, { n: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#shape').onchange = () => { reset(); draw(); };
    q(ctl, '#n').oninput = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const rate = +q(ctl, '#rate').value;
        if (rate <= 2) {
          if (!cur || cur.t >= 1) { if (cur) dirty = true; cur = sample(); }
          cur.t += dt * rate * 1.0; dirty = true;
          if (cur.t >= 1) { cur.t = 1; commit(cur); }
        } else { const k = tick(dt, rate); for (let i = 0; i < k; i++) { cur = sample(); cur.t = 1; commit(cur); } if (k) dirty = true; }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Confidence intervals and coverage
     ===================================================================== */
  AI.labs['ds-st-ci'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Confidence intervals opening and covering the true value', 'illustrative: population is normal with true mean 50 and SD 10; each row is a new study of n values',
      `<canvas style="height:420px"></canvas>`,
      `${sl('n', 'Sample size n per study', 3, 100, 1, 10)}
       <label>Confidence level <select id="lvl"><option value="0.8">80%</option><option value="0.9">90%</option><option value="0.95" selected>95%</option><option value="0.99">99%</option></select></label>
       ${sl('rate', 'Studies per second', 0.25, 30, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), MU = 50, SIG = 10, ROWS = 40;
    const st = { playing: true };
    let rs, rows, total, hit, dirty = true, tick = AI.ticker();
    const reset = () => { rs = AI.rng(23); rows = []; total = 0; hit = 0; dirty = true; };
    const study = () => {
      if (rows[0]) rows[0].t = 1;
      const n = +q(ctl, '#n').value, lvl = +q(ctl, '#lvl').value, x = [];
      for (let i = 0; i < n; i++) x.push(MU + SIG * AI.randn(rs));
      const m = mean(x), s = sdev(x), tc = tinv(1 - (1 - lvl) / 2, n - 1), h = tc * s / Math.sqrt(n);
      const cover = m - h <= MU && MU <= m + h;
      total++; if (cover) hit++;
      rows.unshift({ m, h, cover, t: 0, n, s, tc }); if (rows.length > ROWS) rows.pop();
      dirty = true;
    };
    const step = () => { rows.forEach((r) => (r.t = 1)); study(); rows[0].t = 1; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 20, R = W - 20, top = 26, bot = H - 38, lo = 25, hi = 75, X = (v) => L + (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (R - L);
      const rh = (bot - top) / ROWS;
      txt(g, 'newest study on top; green = interval contains the true mean 50, red = misses it', L, 14, c.muted);
      rows.forEach((r, i) => {
        const y = top + (i + 0.5) * rh, e = ease(r.t), a = r.h * e, col = r.cover ? c.green : c.red;
        g.globalAlpha = i === 0 ? 1 : Math.max(0.35, 1 - i / ROWS * 0.6);
        g.strokeStyle = col; g.lineWidth = Math.max(2, rh * 0.45);
        g.beginPath(); g.moveTo(X(r.m - a), y); g.lineTo(X(r.m + a), y); g.stroke();
        g.fillStyle = c.bg; g.beginPath(); g.arc(X(r.m), y, Math.max(1.5, rh * 0.22), 0, 7); g.fill(); g.lineWidth = 1; g.globalAlpha = 1;
      });
      g.strokeStyle = c.fg; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(X(MU), top - 4); g.lineTo(X(MU), bot + 2); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      txt(g, 'true mean μ = 50 (fixed, not random)', X(MU), bot + 30, c.fg, 'bold 11px sans-serif', 'center');
      for (const v of [30, 40, 60, 70]) txt(g, v, X(v), bot + 14, c.muted, '11px sans-serif', 'center');
      const lvl = +q(ctl, '#lvl').value, r0 = rows[0];
      q(ctl, '#out').textContent = total ? `${total} studies so far: ${hit} intervals contained μ = ${(hit / total * 100).toFixed(1)}% (nominal ${lvl * 100}%). Latest: x̄ = ${r0.m.toFixed(2)}, s = ${r0.s.toFixed(2)}, t* = ${r0.tc.toFixed(3)}, interval [${(r0.m - r0.h).toFixed(2)}, ${(r0.m + r0.h).toFixed(2)}].` : 'Press play: studies arrive one at a time.';
      q(ctl, '#cap').textContent = total < 20 ? 'Each row is a different random sample, so each interval lands somewhere different. The true mean never moves.' : 'The intervals move, the true value does not. In the long run about the nominal share of intervals cover it, which is what "95% confidence" means: a property of the procedure, not of one interval.';
      showVals(ctl, { n: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#n').oninput = () => { reset(); draw(); };
    q(ctl, '#lvl').onchange = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const rate = +q(ctl, '#rate').value;
        if (rate <= 2) {
          if (!rows.length || rows[0].t >= 1) study();
          rows[0].t = Math.min(1, rows[0].t + dt * rate * 1.2); dirty = true;
        } else { const k = tick(dt, rate); for (let i = 0; i < k; i++) { study(); rows[0].t = 1; } }
      } else if (rows.length && rows[0].t < 1) { /* stay paused mid-animation */ }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. Hypothesis test: p-value accumulating as values arrive
     ===================================================================== */
  AI.labs['ds-st-pvalue'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One-sample t-test: the p-value as data arrive', 'illustrative: edit the list of values or the null mean; values are revealed one at a time',
      `<canvas style="height:360px"></canvas>`,
      `<label>Data (comma separated) <input type="text" id="data" value="7.2, 9.1, 6.4, 8.8, 10.3, 7.9, 8.5, 9.6" style="width:100%"></label>
       <label>Null value μ₀ (the claim) <input type="number" id="mu0" value="7" step="0.1" style="width:80px"></label>
       ${sl('tm', 'Or simulate 30 values with true mean', 5, 9, 0.1, 7)}
       <div class="row"><button class="btn" id="sim">Simulate 30 values</button></div>
       ${sl('rate', 'Values revealed per second', 0.2, 6, 0.2, 0.6)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let vals = [], k = 2, acc = 0, dirty = true, seed = 40;
    const parse = () => { vals = q(ctl, '#data').value.split(/[,\s;]+/).map(Number).filter((x) => Number.isFinite(x)); };
    const reset = () => { parse(); k = Math.min(2, vals.length); acc = 0; dirty = true; };
    const step = () => { if (k < vals.length) k++; };
    const stats = () => {
      const a = vals.slice(0, k), mu0 = +q(ctl, '#mu0').value;
      if (a.length < 2) return null;
      const m = mean(a), s = sdev(a), se = s / Math.sqrt(a.length), t = (m - mu0) / se, df = a.length - 1;
      return { m, s, se, t, df, p: 2 * (1 - tcdf(Math.abs(t), df)), n: a.length, mu0 };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const S = stats(), half = W * 0.44, L1 = 16, R1 = half - 8, L2 = half + 24, R2 = W - 16;
      txt(g, 'the data so far', L1, 14, c.muted);
      const mu0 = +q(ctl, '#mu0').value, all = vals.concat([mu0]), dmin = Math.min(...all), dmax = Math.max(...all), pad = (dmax - dmin) * 0.12 + 0.5;
      const lo = dmin - pad, hi = dmax + pad, X = (v) => L1 + (v - lo) / (hi - lo) * (R1 - L1), by = H * 0.62;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L1, by); g.lineTo(R1, by); g.stroke();
      const ticks = AI.niceStep(hi - lo, 5); for (let v = Math.ceil(lo / ticks) * ticks; v <= hi; v += ticks) txt(g, +v.toFixed(2), X(v), by + 14, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.orange; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(X(mu0), 40); g.lineTo(X(mu0), by); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      txt(g, 'μ₀ = ' + mu0, X(mu0) + 4, 38, c.orange, 'bold 11px sans-serif');
      const stack = {};
      vals.slice(0, k).forEach((v, i) => { const key = Math.round(X(v) / 9); stack[key] = (stack[key] || 0) + 1; const y = by - 8 - (stack[key] - 1) * 11; g.fillStyle = i === k - 1 ? c.green : c.blue; g.beginPath(); g.arc(X(v), y, 4.5, 0, 7); g.fill(); });
      if (S) { g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(X(S.m), by - 6); g.lineTo(X(S.m), by + 22); g.stroke(); g.lineWidth = 1; txt(g, 'x̄ = ' + S.m.toFixed(2), X(S.m), by + 36, c.blue, 'bold 11px sans-serif', 'center'); }
      // right: t distribution
      txt(g, S ? `t curve if H0 true (df = ${S.df})` : 'need at least 2 values', L2, 14, c.muted);
      if (S) {
        const tb = by, tt = 44, ymax = tpdf(0, S.df) * 1.08, XT = (t) => L2 + (t + 6) / 12 * (R2 - L2), YT = (p) => tb - p / ymax * (tb - tt), at = Math.min(6, Math.abs(S.t));
        g.fillStyle = c.red; g.globalAlpha = 0.5;
        for (const sgn of [-1, 1]) { g.beginPath(); g.moveTo(XT(sgn * at), tb); for (let i = 0; i <= 60; i++) { const t = at + (6 - at) * i / 60; g.lineTo(XT(sgn * t), YT(tpdf(t, S.df))); } g.lineTo(XT(sgn * 6), tb); g.closePath(); g.fill(); }
        g.globalAlpha = 1; g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath();
        for (let i = 0; i <= 200; i++) { const t = -6 + 12 * i / 200; i ? g.lineTo(XT(t), YT(tpdf(t, S.df))) : g.moveTo(XT(t), YT(tpdf(t, S.df))); } g.stroke(); g.lineWidth = 1;
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(L2, tb); g.lineTo(R2, tb); g.stroke();
        for (const t of [-4, -2, 0, 2, 4]) txt(g, t, XT(t), tb + 14, c.muted, '11px sans-serif', 'center');
        g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); g.moveTo(XT(Math.max(-6, Math.min(6, S.t))), tb); g.lineTo(XT(Math.max(-6, Math.min(6, S.t))), tt - 6); g.stroke(); g.lineWidth = 1;
        txt(g, 'observed t = ' + S.t.toFixed(3), XT(Math.max(-6, Math.min(6, S.t))) + (S.t > 2 ? -6 : 6), tt - 10, c.green, 'bold 11px sans-serif', S.t > 2 ? 'right' : 'left');
        txt(g, 'red area = p-value', R2, tb + 36, c.red, 'bold 11px sans-serif', 'right');
      }
      q(ctl, '#out').textContent = S ? `n = ${S.n}: x̄ = ${S.m.toFixed(3)}, s = ${S.s.toFixed(3)}, SE = s/√n = ${S.se.toFixed(3)}, t = (x̄ − μ₀)/SE = ${S.t.toFixed(3)}, df = ${S.df}, two-sided p = ${S.p.toFixed(4)}.` : 'Waiting for data.';
      q(ctl, '#cap').textContent = !S ? '' : S.p < 0.05 ? `p ${S.p < 0.001 ? '< 0.001' : '= ' + S.p.toFixed(3)} is below 0.05: data this far from μ₀ would be rare if H0 were true. That is evidence against H0, not proof, and says nothing about the size or importance of the difference.` : `p = ${S.p.toFixed(3)} is above 0.05: data like these are not unusual if H0 were true. That is not evidence that H0 is true; the sample may simply be too small. Do not stop the moment p dips below 0.05: peeking inflates false positives.`;
      showVals(ctl, { tm: (v) => v.toFixed(1), rate: (v) => v });
    };
    const reload = () => { reset(); draw(); };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#data').onchange = reload; q(ctl, '#mu0').oninput = () => { dirty = true; };
    q(ctl, '#sim').onclick = () => {
      const r = AI.rng(seed++), tm = +q(ctl, '#tm').value, a = [];
      for (let i = 0; i < 30; i++) a.push((tm + 1.3 * AI.randn(r)).toFixed(1));
      q(ctl, '#data').value = a.join(', '); reload();
    };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && k < vals.length) { acc += dt * +q(ctl, '#rate').value; while (acc >= 1 && k < vals.length) { acc -= 1; k++; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5. Errors, power and sample size
     ===================================================================== */
  AI.labs['ds-st-power'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Type I error, type II error and power', 'illustrative: a two-sided z-test of H0: μ = 0 when the true mean is really δ; each dot is one repeated experiment',
      `<canvas style="height:440px"></canvas>`,
      `${sl('d', 'True effect δ', 0, 10, 0.5, 5)}
       ${sl('s', 'Spread σ of one value', 4, 20, 1, 10)}
       ${sl('n', 'Sample size n', 4, 200, 1, 25)}
       <label>Significance level α <select id="a"><option value="0.1">0.10</option><option value="0.05" selected>0.05</option><option value="0.01">0.01</option></select></label>
       ${sl('rate', 'Experiments per second', 0.5, 60, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), LO = -20, HI = 25, NB = 90;
    const st = { playing: true };
    let rs, bins, rej, tot, dirty = true, tick = AI.ticker();
    const par = () => { const d = +q(ctl, '#d').value, s = +q(ctl, '#s').value, n = +q(ctl, '#n').value, a = +q(ctl, '#a').value, se = s / Math.sqrt(n), zc = Phiinv(1 - a / 2); return { d, s, n, a, se, zc, crit: zc * se }; };
    const powerAt = (d, se, zc) => 1 - Phi(zc - d / se) + Phi(-zc - d / se);
    const reset = () => { rs = AI.rng(31); bins = new Array(NB).fill(0).map(() => [0, 0]); rej = 0; tot = 0; dirty = true; };
    const run = () => {
      const p = par(), x = p.d + p.se * AI.randn(rs), b = Math.max(0, Math.min(NB - 1, Math.floor((x - LO) / (HI - LO) * NB))), r = Math.abs(x) > p.crit ? 1 : 0;
      bins[b][r]++; rej += r; tot++; dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const p = par(), L = 16, R = W - 16, base = Math.round(H * 0.4), split = base + 34 + 50 + 14, top = 44;
      const X = (v) => L + (v - LO) / (HI - LO) * (R - L), pk = npdf(0, 0, p.se), Y = (v) => base - v / (pk * 1.05) * (base - top);
      txt(g, 'sample mean x̄: red = if H0 true (mean 0), blue = if true mean is δ', L, 14, c.muted);
      const area = (m, from, to, col) => { g.fillStyle = col; g.beginPath(); g.moveTo(X(from), base); for (let i = 0; i <= 80; i++) { const x = from + (to - from) * i / 80; g.lineTo(X(x), Y(npdf(x, m, p.se))); } g.lineTo(X(to), base); g.closePath(); g.fill(); };
      g.globalAlpha = 0.55;
      area(0, LO, -p.crit, c.red); area(0, p.crit, HI, c.red);              // alpha
      area(p.d, -p.crit, p.crit, c.orange);                                // beta
      area(p.d, p.crit, HI, c.green); area(p.d, LO, -p.crit, c.green);     // power
      g.globalAlpha = 1;
      const curve = (m, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 300; i++) { const x = LO + (HI - LO) * i / 300; i ? g.lineTo(X(x), Y(npdf(x, m, p.se))) : g.moveTo(X(x), Y(npdf(x, m, p.se))); } g.stroke(); g.lineWidth = 1; };
      curve(0, c.red); curve(p.d, c.blue);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, base); g.lineTo(R, base); g.stroke();
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); [-p.crit, p.crit].forEach((v) => { g.beginPath(); g.moveTo(X(v), base); g.lineTo(X(v), top - 6); g.stroke(); }); g.setLineDash([]);
      txt(g, `reject H0 beyond ±${p.crit.toFixed(2)}`, X(p.crit) + 4, top - 8, c.fg, '11px sans-serif');
      for (const v of [-20, -10, 0, 10, 20]) txt(g, v, X(v), base + 13, c.muted, '11px sans-serif', 'center');
      // simulated experiments: histogram of their x-bar values under the axis
      const dTop = base + 34, dBot = dTop + 50, mxb = Math.max(1, ...bins.map((b) => b[0] + b[1])), bw = (R - L) / NB, sc = Math.min(5, 50 / mxb);
      txt(g, 'repeated experiments: green = rejected H0, orange = missed', L, base + 28, c.muted, '11px sans-serif');
      bins.forEach((b, i) => {
        if (b[0]) { g.fillStyle = c.orange; g.fillRect(L + i * bw + 1, dBot - b[0] * sc, bw - 2, b[0] * sc); }
        if (b[1]) { g.fillStyle = c.green; g.fillRect(L + i * bw + 1, dBot - (b[0] + b[1]) * sc, bw - 2, b[1] * sc); }
      });
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, dBot); g.lineTo(R, dBot); g.stroke();
      // bottom: power curve versus n
      const t0 = split + 26, t1 = H - 24, P0 = L + 30, P1 = R, nmax = 200;
      const Xn = (n) => P0 + (n - 4) / (nmax - 4) * (P1 - P0), Yp = (v) => t1 - v * (t1 - t0);
      txt(g, 'power versus sample size at the current δ, σ and α', L, split + 14, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(P0, t0); g.lineTo(P0, t1); g.lineTo(P1, t1); g.stroke();
      g.setLineDash([3, 4]); g.beginPath(); g.moveTo(P0, Yp(0.8)); g.lineTo(P1, Yp(0.8)); g.stroke(); g.setLineDash([]);
      txt(g, '80%', P0 - 4, Yp(0.8) + 4, c.muted, '11px sans-serif', 'right'); txt(g, '100%', P0 - 4, t0 + 8, c.muted, '11px sans-serif', 'right'); txt(g, '0', P0 - 4, t1, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath();
      for (let n = 4; n <= nmax; n += 2) { const se = p.s / Math.sqrt(n), v = powerAt(p.d, se, p.zc); n === 4 ? g.moveTo(Xn(n), Yp(v)) : g.lineTo(Xn(n), Yp(v)); } g.stroke(); g.lineWidth = 1;
      const pw = powerAt(p.d, p.se, p.zc);
      g.fillStyle = c.purple; g.beginPath(); g.arc(Xn(p.n), Yp(pw), 5, 0, 7); g.fill();
      for (const n of [50, 100, 150, 200]) txt(g, n, Xn(n), t1 + 13, c.muted, '11px sans-serif', 'center');
      txt(g, 'n', P1, t1 + 13, c.muted, '11px sans-serif', 'right');
      const alpha = 2 * (1 - Phi(p.zc)), beta = 1 - pw;
      q(ctl, '#out').textContent = `SE = σ/√n = ${p.s}/√${p.n} = ${p.se.toFixed(3)}. Type I error α = ${(alpha * 100).toFixed(1)}% (red). Power = ${(pw * 100).toFixed(1)}% (green), type II error β = ${(beta * 100).toFixed(1)}% (orange). ${tot ? `Simulated: ${rej} of ${tot} experiments rejected = ${(rej / tot * 100).toFixed(1)}%.` : ''}`;
      q(ctl, '#cap').textContent = p.d === 0 ? 'With δ = 0 there is no effect to find: the "power" equals α, and every rejection is a false alarm.' : pw < 0.8 ? 'Below 80% power: a real effect of this size would often go undetected. Raise n (narrower curves), lower σ, or accept that only larger effects are detectable.' : 'At or above 80% power: most repeated experiments detect an effect of this size. The long-run share of green dots settles near the power.';
      showVals(ctl, { d: (v) => v, s: (v) => v, n: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, run, reset, draw);
    ['d', 's', 'n'].forEach((id) => (q(ctl, '#' + id).oninput = () => { reset(); draw(); }));
    q(ctl, '#a').onchange = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && tot < 4000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) run(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. The bootstrap
     ===================================================================== */
  AI.labs['ds-st-bootstrap'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The bootstrap: resample your own data', 'illustrative: draw n values from your sample with replacement, record the statistic, repeat',
      `<canvas style="height:420px"></canvas>`,
      `<label>Your data (comma separated) <input type="text" id="data" value="7.2, 9.1, 6.4, 8.8, 10.3, 7.9, 8.5, 9.6" style="width:100%"></label>
       <label>Statistic <select id="stat"><option value="mean">mean</option><option value="median">median</option><option value="max">maximum</option></select></label>
       ${sl('rate', 'Resamples per second', 0.25, 100, 0.25, 0.5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), NB = 40;
    const st = { playing: true };
    let data = [], rs, counts, picks, last, stats, hold, bins, lo, hi, dirty = true, tick = AI.ticker();
    const fn = (a) => { const t = q(ctl, '#stat').value; if (t === 'mean') return mean(a); if (t === 'max') return Math.max(...a); const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
    const reset = () => {
      data = q(ctl, '#data').value.split(/[,\s;]+/).map(Number).filter((x) => Number.isFinite(x)).slice(0, 40);
      if (data.length < 2) data = [1, 2];
      rs = AI.rng(1); counts = new Array(data.length).fill(0); picks = 0; last = -1; stats = []; hold = 0; bins = new Array(NB).fill(0);
      const mn = Math.min(...data), mxv = Math.max(...data), pd = (mxv - mn) * 0.1 || 1; lo = mn - pd; hi = mxv + pd; dirty = true;
    };
    const finish = () => {
      const a = []; counts.forEach((c, i) => { for (let j = 0; j < c; j++) a.push(data[i]); });
      const v = fn(a); stats.push(v); bins[Math.max(0, Math.min(NB - 1, Math.floor((v - lo) / (hi - lo) * NB)))]++;
    };
    const pick = () => {
      const i = Math.floor(rs() * data.length); counts[i]++; picks++; last = i;
      if (picks === data.length) { finish(); hold = +q(ctl, '#rate').value <= 3 ? 0.9 : 0; picks = hold ? picks : 0; if (!hold) counts.fill(0); }
      dirty = true;
    };
    const step = () => { if (hold) { hold = 0; picks = 0; counts.fill(0); } pick(); };
    const pct = (p) => { const s = [...stats].sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 24, R = W - 24, X = (v) => L + (v - lo) / (hi - lo) * (R - L), n = data.length, split = Math.round(H * 0.44), base = split - 24;
      txt(g, `your ${n} values; circles = times drawn into this resample (${picks === n ? 'complete' : picks + ' of ' + n})`, L, 14, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, base); g.lineTo(R, base); g.stroke();
      const ticks = AI.niceStep(hi - lo, 6); for (let v = Math.ceil(lo / ticks) * ticks; v <= hi; v += ticks) txt(g, +v.toFixed(2), X(v), base + 14, c.muted, '11px sans-serif', 'center');
      data.forEach((v, i) => {
        g.fillStyle = c.muted; g.beginPath(); g.arc(X(v), base, 4, 0, 7); g.fill();
        for (let j = 0; j < counts[i]; j++) { g.fillStyle = i === last && j === counts[i] - 1 ? c.orange : c.blue; g.beginPath(); g.arc(X(v), base - 13 - j * 11, 4.6, 0, 7); g.fill(); }
      });
      const obs = fn(data);
      // histogram of bootstrap statistics
      const t0 = split + 22, t1 = H - 26, th = t1 - t0, bw = (R - L) / NB, mb = Math.max(1, ...bins);
      txt(g, `the ${stats.length} resample statistics so far (${q(ctl, '#stat').value})`, L, split + 10, c.muted);
      g.fillStyle = c.green; g.globalAlpha = 0.75; bins.forEach((v, i) => g.fillRect(L + i * bw + 1, t1 - v / mb * th, bw - 2, v / mb * th)); g.globalAlpha = 1;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t1); g.lineTo(R, t1); g.stroke();
      for (let v = Math.ceil(lo / ticks) * ticks; v <= hi; v += ticks) txt(g, +v.toFixed(2), X(v), t1 + 14, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.fg; g.lineWidth = 2; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(obs), t0); g.lineTo(X(obs), t1); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      txt(g, 'observed ' + obs.toFixed(3), X(obs) + 4, t0 + 8, c.fg, 'bold 11px sans-serif');
      let msg = `Observed ${q(ctl, '#stat').value} = ${obs.toFixed(3)}. Resamples: ${stats.length}.`;
      if (stats.length >= 20) {
        const a = pct(0.025), b = pct(0.975), m = mean(stats), se = sdev(stats);
        g.strokeStyle = c.purple; g.lineWidth = 3;
        [a, b].forEach((v) => { g.beginPath(); g.moveTo(X(v), t0 + 16); g.lineTo(X(v), t1); g.stroke(); }); g.lineWidth = 1;
        txt(g, `95% percentile interval [${a.toFixed(2)}, ${b.toFixed(2)}]`, (X(a) + X(b)) / 2, t0 + 28, c.purple, 'bold 11px sans-serif', 'center');
        msg += ` Bootstrap SE ${se.toFixed(3)}, mean of resamples ${m.toFixed(3)}, 2.5th and 97.5th percentiles [${a.toFixed(3)}, ${b.toFixed(3)}].`;
        if (q(ctl, '#stat').value === 'mean') { const sdd = sdev(data), tc = tinv(0.975, n - 1), h = tc * sdd / Math.sqrt(n); msg += ` Formula t-interval: [${(mean(data) - h).toFixed(3)}, ${(mean(data) + h).toFixed(3)}].`; }
      }
      q(ctl, '#out').textContent = msg;
      q(ctl, '#cap').textContent = q(ctl, '#stat').value === 'max' ? 'The bootstrap fails for the maximum: a resample can never exceed the biggest value you already have, so the histogram piles up at the observed maximum and badly understates the true uncertainty.' : stats.length < 10 ? 'Each resample draws n values from your own data with replacement, so some values appear twice or more and others not at all. Each resample gives a slightly different statistic.' : 'The spread of the green histogram is the bootstrap estimate of how much the statistic would wobble from sample to sample. No formula was used.';
      showVals(ctl, { rate: (v) => v });
    };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#data').onchange = () => { reset(); draw(); };
    q(ctl, '#stat').onchange = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && stats.length < 20000) {
        if (hold > 0) { hold -= dt; if (hold <= 0) { hold = 0; picks = 0; counts.fill(0); last = -1; } dirty = true; }
        else { const k = tick(dt, +q(ctl, '#rate').value * data.length); for (let i = 0; i < k; i++) pick(); }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
