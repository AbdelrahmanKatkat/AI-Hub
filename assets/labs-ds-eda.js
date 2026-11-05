/* Data Science, Module 2 (exploratory data analysis) labs.
   ds-eda-summary, ds-eda-hist, ds-eda-scatter, ds-eda-corr, ds-eda-crosstab, ds-eda-simpson, ds-eda-groups, ds-eda-outliers, ds-eda-log.
   All data are small seeded simulations or the classic published tables quoted in the lessons; every lab is labelled illustrative.
   Animations run on dt (slow defaults, speed slider, play / pause / step / reset) and redraw only when something changed. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = (step) => `<div class="row"><button class="btn primary" id="play">▶ Play</button>${step ? '<button class="btn" id="step">Step</button>' : ''}<button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const ease = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const quant = (s, p) => { if (!s.length) return NaN; const i = p * (s.length - 1), l = Math.floor(i), h = Math.min(l + 1, s.length - 1); return s[l] + (s[h] - s[l]) * (i - l); };
  const tickList = (lo, hi, n) => { const st = AI.niceStep(hi - lo, n || 5), a = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) a.push(+v.toFixed(6)); return a; };

  /* summary statistics; quartiles by linear interpolation (the NumPy default); fences at k x IQR */
  function stats(a, k) {
    k = k === undefined ? 1.5 : k;
    const n = a.length; if (!n) return null;
    const s = a.slice().sort((x, y) => x - y), m = mean(a);
    let ss = 0, s3 = 0; a.forEach((x) => { ss += (x - m) ** 2; s3 += (x - m) ** 3; });
    const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : 0, sdp = Math.sqrt(ss / n);
    const q1 = quant(s, 0.25), med = quant(s, 0.5), q3 = quant(s, 0.75), iqr = q3 - q1, lo = q1 - k * iqr, hi = q3 + k * iqr;
    const inside = s.filter((x) => x >= lo && x <= hi);
    return { n, s, mean: m, sd, med, q1, q3, iqr, lo, hi, min: s[0], max: s[n - 1], wlo: inside.length ? inside[0] : s[0], whi: inside.length ? inside[inside.length - 1] : s[n - 1], skew: sdp > 0 ? s3 / n / sdp ** 3 : 0 };
  }
  const pearson = (x, y) => { const mx = mean(x), my = mean(y); let a = 0, b = 0, c = 0; for (let i = 0; i < x.length; i++) { a += (x[i] - mx) * (y[i] - my); b += (x[i] - mx) ** 2; c += (y[i] - my) ** 2; } return b > 0 && c > 0 ? a / Math.sqrt(b * c) : NaN; };
  const ranks = (a) => { const idx = a.map((v, i) => [v, i]).sort((p, r) => p[0] - r[0]), out = Array(a.length); let i = 0; while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; const rk = (i + j) / 2 + 1; for (let k = i; k <= j; k++) out[idx[k][1]] = rk; i = j + 1; } return out; };
  const spearman = (x, y) => pearson(ranks(x), ranks(y));

  /* the 200-row customer table used by the outlier and workflow labs (seeded, so every run is the same) */
  function customers() {
    const r = AI.rng(2024), N = () => AI.randn(r), rows = [];
    for (let i = 0; i < 200; i++) {
      const age = clamp(Math.round(38 + 11 * N()), 18, 80), tenure = Math.round(clamp(24 + 14 * N(), 1, 72));
      const income = Math.round(Math.exp(3.7 + 0.5 * N()) * 10) / 10;
      const spend = Math.round(Math.max(5, 0.6 * income + 12 * N()) * 10) / 10;
      const churn = r() < 1 / (1 + Math.exp(0.07 * (tenure - 20))) ? 1 : 0;
      rows.push({ age, tenure, income, spend, churn });
    }
    [[17, 999], [88, 999], [140, 0]].forEach(([i, v]) => { rows[i].age = v; });        // data-entry errors
    const idx = Array.from({ length: 200 }, (_, i) => i); for (let i = 199; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    idx.slice(0, 20).forEach((i) => { rows[i].income = null; });                          // 20 missing incomes
    return rows;
  }
  /* regularised incomplete beta -> Student t two-sided p-value (Numerical-Recipes continued fraction) */
  function lgam(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const k of c) s += k / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  function betacf(a, b, x) { const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d; let h = d; for (let m = 1; m <= 200; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d; h *= d * c; aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30; c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break; } return h; }
  function ibeta(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
  const pT = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));
  function welch(a, b) {
    const na = a.length, nb = b.length; if (na < 2 || nb < 2) return null;
    const ma = mean(a), mb = mean(b), va = a.reduce((s, x) => s + (x - ma) ** 2, 0) / (na - 1), vb = b.reduce((s, x) => s + (x - mb) ** 2, 0) / (nb - 1);
    const se = Math.sqrt(va / na + vb / nb), t = (mb - ma) / se, df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1));
    const sp = Math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2));
    return { ma, mb, diff: mb - ma, se, t, df, p: pT(t, df), d: (mb - ma) / sp };
  }
  AI._dsEda = { stats, pearson, spearman, customers, welch };   // exposed so the lesson numbers can be re-computed in node

  /* horizontal box plot of a stats() object on pixel mapping X, centred on y */
  function hbox(g, st, X, y, h, c) {
    if (!st) return;
    g.strokeStyle = c.fg; g.fillStyle = AI.mix(c.bg2, c.blue, 0.45); g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(X(st.wlo), y); g.lineTo(X(st.q1), y); g.moveTo(X(st.q3), y); g.lineTo(X(st.whi), y);
    g.moveTo(X(st.wlo), y - h / 3); g.lineTo(X(st.wlo), y + h / 3); g.moveTo(X(st.whi), y - h / 3); g.lineTo(X(st.whi), y + h / 3); g.stroke();
    g.fillRect(X(st.q1), y - h / 2, Math.max(1, X(st.q3) - X(st.q1)), h); g.strokeRect(X(st.q1), y - h / 2, Math.max(1, X(st.q3) - X(st.q1)), h);
    g.strokeStyle = c.green; g.lineWidth = 2.5; g.beginPath(); g.moveTo(X(st.med), y - h / 2); g.lineTo(X(st.med), y + h / 2); g.stroke();
    g.fillStyle = c.red; st.s.forEach((v) => { if (v < st.wlo || v > st.whi) { g.beginPath(); g.arc(X(v), y, 3.5, 0, 7); g.fill(); } });
    g.lineWidth = 1;
  }

  /* ---------- 1. One variable ---------- */
  AI.labs['ds-eda-summary'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One variable: centre, spread and shape', 'coffees sold per day at a café; days arrive one at a time (illustrative)',
      `<canvas style="height:320px"></canvas>`,
      `${sl('v', 'Sales on day 9 (the last value)', 10, 130, 1, 60)}${sl('sp', 'Speed (days per second)', 0.2, 3, 0.1, 0.6)}${btns(true)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const base = [15, 21, 12, 16, 14, 19, 15, 18], START = 3, LO = 0, HI = 130;
    const vIn = q(ctl, '#v'), spIn = q(ctl, '#sp'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let shown = START, acc = 0, playing = false, dirty = true, mD = null, dD = null; const prog = Array(9).fill(0);
    for (let i = 0; i < START; i++) prog[i] = 1;
    const data = () => base.concat([+vIn.value]).slice(0, shown);
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const x0 = 28, x1 = W - 22, X = (v) => x0 + (v - LO) / (HI - LO) * (x1 - x0), ay = Hh * 0.56, d = data(), st = stats(d);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, ay); g.lineTo(x1, ay); g.stroke();
      tickList(LO, HI, 7).forEach((t) => { g.beginPath(); g.moveTo(X(t), ay); g.lineTo(X(t), ay + 4); g.stroke(); txt(g, t, X(t), ay + 17, c.muted, '11px sans-serif', 'center'); });
      txt(g, 'coffees sold per day', x1, ay + 32, c.muted, '11px sans-serif', 'right');
      // dots
      const R = 8, placed = [];
      d.forEach((v, i) => {
        let lvl = 0; placed.forEach((p) => { if (Math.abs(X(p.v) - X(v)) < R * 1.7) lvl = Math.max(lvl, p.l + 1); });
        placed.push({ v, l: lvl });
        const pr = ease(prog[i]), y = ay - R - lvl * R * 1.9 - (1 - pr) * 30;
        g.globalAlpha = pr; g.fillStyle = st.n > 3 && (v < st.lo || v > st.hi) ? c.red : c.blue;
        g.beginPath(); g.arc(X(v), y, R * (0.4 + 0.6 * pr), 0, 7); g.fill(); g.globalAlpha = 1;
      });
      // mean and median markers (tweened)
      const md = dD === null ? st.med : dD, mn = mD === null ? st.mean : mD;
      g.lineWidth = 2; g.strokeStyle = c.orange; g.beginPath(); g.moveTo(X(mn), 34); g.lineTo(X(mn), ay); g.stroke();
      g.strokeStyle = c.green; g.beginPath(); g.moveTo(X(md), 52); g.lineTo(X(md), ay); g.stroke(); g.lineWidth = 1;
      const right = X(mn) < x1 - 90;
      txt(g, `mean ${AI.fmt(mn, 2)}`, X(mn) + (right ? 5 : -5), 40, c.orange, 'bold 12px sans-serif', right ? 'left' : 'right');
      txt(g, `median ${AI.fmt(md, 2)}`, X(md) + (X(md) < x1 - 100 ? 5 : -5), 58, c.green, 'bold 12px sans-serif', X(md) < x1 - 100 ? 'left' : 'right');
      // box plot
      const by = Hh - 36; txt(g, 'box plot', x0, by - 24, c.muted, '11px sans-serif'); hbox(g, st, X, by, 24, c);
      txt(g, 'red dot = beyond 1.5 × IQR fence', x1, by - 24, c.muted, '11px sans-serif', 'right');
    };
    const refreshText = () => {
      const d = data(), st = stats(d);
      out.innerHTML = `n = ${st.n}, mean ${AI.fmt(st.mean, 2)}, median ${AI.fmt(st.med, 2)}<br>sd ${AI.fmt(st.sd, 2)}, Q1 ${AI.fmt(st.q1, 2)}, Q3 ${AI.fmt(st.q3, 2)}, IQR ${AI.fmt(st.iqr, 2)}<br>skew ${AI.fmt(st.skew, 2)}, five numbers: ${[st.min, st.q1, st.med, st.q3, st.max].map((v) => AI.fmt(v, 2)).join(' · ')}`;
      const gap = st.mean - st.med;
      cap.textContent = shown < 9 ? `Day ${shown} is in. Watch the green median stay put while the orange mean shifts a little with each new day.`
        : `All 9 days are in. The mean is ${AI.fmt(Math.abs(gap), 2)} ${gap > 0 ? 'above' : 'below'} the median: ${Math.abs(gap) > 2 * st.iqr / 4 ? 'one extreme day is dragging the mean, the median and IQR barely notice' : 'the two centres agree, so there is no strong pull from either tail'}. Drag the day-9 slider to see this live.`;
    };
    const touch = () => { dirty = true; refreshText(); };
    vIn.oninput = () => { q(ctl, '#vv').textContent = vIn.value; touch(); };
    spIn.oninput = () => { q(ctl, '#spv').textContent = spIn.value; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { if (shown < 9) { shown++; touch(); } };
    q(ctl, '#reset').onclick = () => { shown = START; acc = 0; prog.fill(0); for (let i = 0; i < START; i++) prog[i] = 1; mD = dD = null; touch(); };
    o.onresize = () => { dirty = true; };
    q(ctl, '#vv').textContent = vIn.value; q(ctl, '#spv').textContent = spIn.value; refreshText();
    AI.loop(root, (dt) => {
      if (playing && shown < 9) { acc += dt * +spIn.value; if (acc >= 1) { acc -= 1; shown++; touch(); } }
      for (let i = 0; i < shown; i++) if (prog[i] < 1) { prog[i] = Math.min(1, prog[i] + dt * 1.5); dirty = true; }
      const st = stats(data()), k = 1 - Math.exp(-dt * 4);
      if (mD === null) { mD = st.mean; dD = st.med; }
      if (Math.abs(mD - st.mean) > 0.01 || Math.abs(dD - st.med) > 0.01) { mD += (st.mean - mD) * k; dD += (st.med - dD) * k; dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 2. Histogram, bin width and transforms ---------- */
  AI.labs['ds-eda-hist'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Histograms, bin width and transforms', 'incomes of 400 people, simulated log-normal data (illustrative)',
      `<canvas style="height:330px"></canvas>`,
      `${sl('nb', 'Number of bins', 3, 60, 1, 12)}${sl('sp', 'Speed (rows per second)', 5, 120, 5, 30)}
       <label>Transform <select id="tf"><option value="raw">none (income)</option><option value="log">log₁₀ income</option><option value="sqrt">square root</option></select></label>
       ${btns(false)}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const r = AI.rng(7), raw = Array.from({ length: 400 }, () => Math.round(Math.exp(3.5 + 0.7 * AI.randn(r)) * 10) / 10);
    const F = { raw: (x) => x, log: Math.log10, sqrt: Math.sqrt };
    const nbIn = q(ctl, '#nb'), spIn = q(ctl, '#sp'), tfIn = q(ctl, '#tf'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let m = 100, acc = 0, playing = false, dirty = true, disp = null, edges = null, tfKey = 'raw';
    const vals = () => raw.slice(0, Math.floor(m)).map(F[tfKey]);
    const full = () => raw.map(F[tfKey]);
    const hist = (a, lo, hi, nb) => { const h = Array(nb).fill(0), w = (hi - lo) / nb; a.forEach((v) => { h[Math.min(nb - 1, Math.floor((v - lo) / w))]++; }); return h; };
    let tgt = [], ymax = 1, lo = 0, hi = 1;
    const retarget = (resample) => {
      const nb = +nbIn.value, fa = full(); lo = Math.min(...fa); hi = Math.max(...fa);
      tgt = hist(vals(), lo, hi, nb); ymax = Math.max(...hist(fa, lo, hi, nb)) * 1.12;
      const old = disp, oe = edges, nl = lo, w = (hi - lo) / nb;
      if (!old || resample === 'zero') disp = Array(nb).fill(0);
      else if (old.length !== nb) { disp = Array.from({ length: nb }, (_, i) => { const cx = nl + (i + 0.5) * w, j = Math.min(old.length - 1, Math.max(0, Math.floor((cx - oe.lo) / ((oe.hi - oe.lo) / old.length)))); return old[j] || 0; }); }
      edges = { lo, hi }; dirty = true;
    };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const x0 = 44, x1 = W - 16, yb = Hh - 64, yt = 18, X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0), Y = (v) => yb - v / ymax * (yb - yt), nb = disp.length, w = (x1 - x0) / nb;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, yb); g.lineTo(x1, yb); g.moveTo(x0, yb); g.lineTo(x0, yt); g.stroke();
      tickList(0, ymax, 4).forEach((t) => { g.strokeStyle = c.border; g.globalAlpha = 0.4; g.beginPath(); g.moveTo(x0, Y(t)); g.lineTo(x1, Y(t)); g.stroke(); g.globalAlpha = 1; txt(g, t, x0 - 5, Y(t) + 4, c.muted, '11px sans-serif', 'right'); });
      for (let i = 0; i < nb; i++) { g.fillStyle = c.blue; g.globalAlpha = 0.85; g.fillRect(x0 + i * w + 0.5, Y(disp[i]), Math.max(1, w - 1), yb - Y(disp[i])); g.globalAlpha = 1; }
      const lab = tfKey === 'log' ? (t) => AI.fmt(10 ** t, 3) : (t) => AI.fmt(t, 3);
      tickList(lo, hi, 7).forEach((t) => { g.beginPath(); g.strokeStyle = c.border; g.moveTo(X(t), yb); g.lineTo(X(t), yb + 4); g.stroke(); txt(g, tfKey === 'log' ? AI.fmt(t, 2) : AI.fmt(t, 3), X(t), yb + 16, c.muted, '11px sans-serif', 'center'); });
      txt(g, tfKey === 'log' ? 'log₁₀(income in k) - tick labels are the log values' : tfKey === 'sqrt' ? '√income' : 'income (thousands)', x1, yb + 30, c.muted, '11px sans-serif', 'right');
      txt(g, 'count', x0 - 5, yt - 6, c.muted, '11px sans-serif', 'right');
      hbox(g, stats(vals()), X, Hh - 14, 16, c);
    };
    const text = () => {
      const a = vals(), st = stats(a), nb = +nbIn.value, w = (hi - lo) / nb;
      q(ctl, '#nbv').textContent = nb; q(ctl, '#spv').textContent = spIn.value;
      out.innerHTML = `n = ${st.n} of 400 · bins ${nb}, width ${AI.fmt(w, 3)}${tfKey === 'log' ? ' (log units)' : ''}<br>mean ${AI.fmt(st.mean, 3)}, median ${AI.fmt(st.med, 3)}, skew ${AI.fmt(st.skew, 2)}`;
      cap.textContent = st.n < 400 ? `Rows are arriving: the shape firms up as n grows (${st.n} now).` : tfKey === 'raw' ? `Raw incomes: skew ${AI.fmt(st.skew, 2)}, a long right tail. Few bins hide it; many bins make it ragged. Try the log transform.` : tfKey === 'log' ? `After the log, skew is ${AI.fmt(st.skew, 2)}: roughly symmetric, a bell shape. Mean ≈ median.` : `The square root pulls the tail in part of the way: skew ${AI.fmt(st.skew, 2)}.`;
    };
    nbIn.oninput = () => { retarget(); text(); };
    spIn.oninput = text;
    tfIn.onchange = () => { tfKey = tfIn.value; disp = null; retarget('zero'); text(); };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { m = 100; acc = 0; retarget(); text(); };
    o.onresize = () => { dirty = true; };
    retarget('zero'); text();
    AI.loop(root, (dt) => {
      if (playing && m < 400) { m = Math.min(400, m + dt * +spIn.value); const nm = Math.floor(m); if (nm !== acc) { acc = nm; tgt = hist(vals(), lo, hi, +nbIn.value); dirty = true; text(); } }
      const k = 1 - Math.exp(-dt * 7);
      for (let i = 0; i < disp.length; i++) { const d = tgt[i] - disp[i]; if (Math.abs(d) > 0.02) { disp[i] += d * k; dirty = true; } else if (d !== 0) { disp[i] = tgt[i]; dirty = true; } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 3. Scatter plot, Pearson and Spearman ---------- */
  AI.labs['ds-eda-scatter'] = function (root) {
    const AX = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5];
    const DS = {
      'Anscombe I': { x: AX, y: [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], xr: [0, 20], yr: [2, 14] },
      'Anscombe II': { x: AX, y: [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74], xr: [0, 20], yr: [2, 14] },
      'Anscombe III': { x: AX, y: [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], xr: [0, 20], yr: [2, 14] },
      'Anscombe IV': { x: [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], y: [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89], xr: [0, 20], yr: [2, 14] },
      'Parabola y = x²': { x: [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], y: [9, 6.25, 4, 2.25, 1, 0.25, 0, 0.25, 1, 2.25, 4, 6.25, 9], xr: [-4, 4], yr: [-1, 10] },
      'Hours vs score': { x: [1, 2, 3, 4, 5, 6], y: [52, 55, 61, 60, 70, 74], xr: [0, 8], yr: [40, 80] },
      'Hours vs score + one bad row': { x: [1, 2, 3, 4, 5, 6, 7], y: [52, 55, 61, 60, 70, 74, 20], xr: [0, 8], yr: [10, 80] }
    };
    const { view, ctl } = AI.labShell(root, 'Scatter plot, Pearson r and Spearman ρ', 'Anscombe\'s quartet and friends; points arrive one by one, then click to add your own (illustrative)',
      `<canvas style="height:340px;cursor:crosshair"></canvas>`,
      `<label>Dataset <select id="ds">${Object.keys(DS).map((k) => `<option>${k}</option>`).join('')}</select></label>${sl('sp', 'Speed (points per second)', 0.3, 4, 0.1, 0.8)}${btns(true)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const dsIn = q(ctl, '#ds'), spIn = q(ctl, '#sp'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let cur = DS[dsIn.value], shown = 3, acc = 0, playing = false, dirty = true, mine = []; const prog = Array(40).fill(1);
    const pts = () => { const n = Math.min(shown, cur.x.length); const x = cur.x.slice(0, n), y = cur.y.slice(0, n); return { x: x.concat(mine.map((p) => p[0])), y: y.concat(mine.map((p) => p[1])), n }; };
    const view_ = () => { const pad = { l: 42, r: 16, t: 14, b: 32 }; return { pad, X: (v) => pad.l + (v - cur.xr[0]) / (cur.xr[1] - cur.xr[0]) * (o.w - pad.l - pad.r), Y: (v) => o.h - pad.b - (v - cur.yr[0]) / (cur.yr[1] - cur.yr[0]) * (o.h - pad.t - pad.b) }; };
    const fit = (x, y) => { const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0; x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; }); const s = sxx > 0 ? sxy / sxx : NaN; return { s, b: my - s * mx, mx, my }; };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const V = view_(), P = pts();
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(V.pad.l, Hh - V.pad.b); g.lineTo(W - V.pad.r, Hh - V.pad.b); g.moveTo(V.pad.l, V.pad.t); g.lineTo(V.pad.l, Hh - V.pad.b); g.stroke();
      tickList(cur.xr[0], cur.xr[1], 6).forEach((t) => { txt(g, t, V.X(t), Hh - V.pad.b + 15, c.muted, '11px sans-serif', 'center'); });
      tickList(cur.yr[0], cur.yr[1], 5).forEach((t) => { g.globalAlpha = 0.35; g.beginPath(); g.moveTo(V.pad.l, V.Y(t)); g.lineTo(W - V.pad.r, V.Y(t)); g.stroke(); g.globalAlpha = 1; txt(g, t, V.pad.l - 5, V.Y(t) + 4, c.muted, '11px sans-serif', 'right'); });
      if (P.x.length >= 3) { const f = fit(P.x, P.y); if (Number.isFinite(f.s)) { g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(V.X(cur.xr[0]), V.Y(f.s * cur.xr[0] + f.b)); g.lineTo(V.X(cur.xr[1]), V.Y(f.s * cur.xr[1] + f.b)); g.stroke(); g.lineWidth = 1; } }
      for (let i = 0; i < P.x.length; i++) { const isMine = i >= P.n, pr = ease(prog[Math.min(i, 39)]); g.fillStyle = isMine ? c.pink : c.blue; g.globalAlpha = pr; g.beginPath(); g.arc(V.X(P.x[i]), V.Y(P.y[i]), 3 + 4 * pr, 0, 7); g.fill(); g.globalAlpha = 1; }
    };
    const text = () => {
      const P = pts(), f = P.x.length >= 3 ? fit(P.x, P.y) : null;
      const r = P.x.length >= 3 ? pearson(P.x, P.y) : NaN, rho = P.x.length >= 3 ? spearman(P.x, P.y) : NaN;
      out.innerHTML = `n = ${P.x.length}${mine.length ? ` (${mine.length} yours)` : ''}<br>Pearson r = <b>${AI.fmt(r, 3)}</b>, Spearman ρ = <b>${AI.fmt(rho, 3)}</b><br>${f ? `line: y = ${AI.fmt(f.s, 3)}·x + ${AI.fmt(f.b, 3)}; mean x ${AI.fmt(f.mx, 2)}, mean y ${AI.fmt(f.my, 2)}` : 'needs 3 points for a line'}`;
      q(ctl, '#spv').textContent = spIn.value;
      const notes = { 'Anscombe I': 'A fairly ordinary noisy straight line.', 'Anscombe II': 'A smooth curve: a straight line is the wrong shape.', 'Anscombe III': 'A perfect line plus one outlier that tilts the fit.', 'Anscombe IV': 'Ten points at the same x; one far-away point alone creates the slope.', 'Parabola y = x²': 'y is fully determined by x, yet r and ρ are about zero: they only see a monotone trend.', 'Hours vs score': 'A clear upward relationship.', 'Hours vs score + one bad row': 'One mis-entered row flips the sign of r. Compare ρ.' };
      cap.textContent = (P.n < cur.x.length ? `Point ${P.n} of ${cur.x.length} just arrived; r and the line update as evidence accumulates. ` : 'All points are in. Click the plot to add a point of your own and watch r move. ') + notes[dsIn.value];
    };
    const reset = () => { cur = DS[dsIn.value]; shown = 3; acc = 0; mine = []; prog.fill(1); for (let i = 3; i < 40; i++) prog[i] = 0; dirty = true; text(); };
    dsIn.onchange = reset; spIn.oninput = text;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { if (shown < cur.x.length) { shown++; dirty = true; text(); } };
    q(ctl, '#reset').onclick = reset;
    cv.addEventListener('click', (e) => {
      const R = cv.getBoundingClientRect(), V = view_(), px = e.clientX - R.left, py = e.clientY - R.top;
      const x = cur.xr[0] + (px - V.pad.l) / (R.width - V.pad.l - V.pad.r) * (cur.xr[1] - cur.xr[0]);
      const y = cur.yr[0] + (R.height - V.pad.b - py) / (R.height - V.pad.t - V.pad.b) * (cur.yr[1] - cur.yr[0]);
      if (x < cur.xr[0] || x > cur.xr[1] || y < cur.yr[0] || y > cur.yr[1] || mine.length > 20) return;
      mine.push([x, y]); prog[Math.min(39, cur.x.length + mine.length - 1)] = 0; dirty = true; text();
    });
    o.onresize = () => { dirty = true; };
    reset();
    AI.loop(root, (dt) => {
      if (playing && shown < cur.x.length) { acc += dt * +spIn.value; if (acc >= 1) { acc -= 1; shown++; text(); dirty = true; } }
      for (let i = 0; i < 40; i++) if (prog[i] < 1 && i < shown + mine.length + 12) { prog[i] = Math.min(1, prog[i] + dt * 2); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 4. Correlation matrix filling in ---------- */
  AI.labs['ds-eda-corr'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A correlation matrix filling in', 'six simulated housing columns, 120 rows; click a cell to see its scatter (illustrative)',
      `<canvas style="height:330px;cursor:pointer"></canvas>`,
      `${sl('th', 'Highlight |r| at or above', 0, 0.95, 0.05, 0.5)}${sl('sp', 'Speed (cells per second)', 0.3, 4, 0.1, 1)}
       <label>Method <select id="me"><option value="p">Pearson r</option><option value="s">Spearman ρ</option></select></label>${btns(false)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const rg = AI.rng(11), N = () => AI.randn(rg), n = 120, cols = { size: [], rooms: [], age: [], dist: [], price: [], 'dist²': [] };
    for (let i = 0; i < n; i++) { const s = N(), rm = 0.8 * s + 0.6 * N(), ag = N(), ds = N(); cols.size.push(s); cols.rooms.push(rm); cols.age.push(ag); cols.dist.push(ds); cols.price.push(0.6 * s + 0.25 * rm - 0.35 * ds + 0.4 * N()); cols['dist²'].push(ds * ds + 0.3 * N()); }
    const names = Object.keys(cols), K = names.length, pairs = []; for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) pairs.push([i, j]);
    const R = { p: {}, s: {} }; pairs.forEach(([i, j]) => { R.p[i + ',' + j] = pearson(cols[names[i]], cols[names[j]]); R.s[i + ',' + j] = spearman(cols[names[i]], cols[names[j]]); });
    const thIn = q(ctl, '#th'), spIn = q(ctl, '#sp'), meIn = q(ctl, '#me'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let filled = 0, acc = 0, playing = false, dirty = true, sel = [0, 1]; const prog = Array(pairs.length).fill(0);
    const val = (i, j) => (i === j ? 1 : R[meIn.value][Math.min(i, j) + ',' + Math.max(i, j)]);
    const pairIndex = (i, j) => pairs.findIndex((p) => p[0] === Math.min(i, j) && p[1] === Math.max(i, j));
    const geom = () => { const W = o.w, Hh = o.h, ml = Math.min(W * 0.58, Hh - 40), cell = (ml - 46) / K; return { cell, x0: 46, y0: 30, ml }; };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const G = geom(), th = +thIn.value;
      names.forEach((nm, i) => { txt(g, nm, G.x0 + (i + 0.5) * G.cell, G.y0 - 8, c.muted, '10px sans-serif', 'center'); txt(g, nm, G.x0 - 4, G.y0 + (i + 0.5) * G.cell + 3, c.muted, '10px sans-serif', 'right'); });
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const x = G.x0 + j * G.cell, y = G.y0 + i * G.cell, k = i === j ? -1 : pairIndex(i, j), pr = i === j ? 1 : ease(prog[k]);
        g.fillStyle = c.bg3; g.fillRect(x + 1, y + 1, G.cell - 2, G.cell - 2);
        if (pr > 0) {
          const v = val(i, j); g.globalAlpha = pr; g.fillStyle = AI.mix(c.bg3, v >= 0 ? c.blue : c.orange, Math.min(1, Math.abs(v)) * 0.95); g.fillRect(x + 1, y + 1, G.cell - 2, G.cell - 2);
          txt(g, v.toFixed(2), x + G.cell / 2, y + G.cell / 2 + 4, c.fg, '11px sans-serif', 'center'); g.globalAlpha = 1;
          if (i !== j && Math.abs(v) >= th) { g.strokeStyle = c.yellow; g.lineWidth = 2.5; g.strokeRect(x + 2, y + 2, G.cell - 4, G.cell - 4); g.lineWidth = 1; }
        }
        if ((i === sel[0] && j === sel[1]) || (i === sel[1] && j === sel[0])) { g.strokeStyle = c.fg; g.lineWidth = 1.5; g.strokeRect(x + 0.5, y + 0.5, G.cell - 1, G.cell - 1); g.lineWidth = 1; }
      }
      // scatter of the selected pair
      const sx = G.x0 + K * G.cell + 22, sw = W - sx - 12, sh = Math.min(sw, 170), sy = G.y0 + 10;
      if (sw > 80) {
        const a = cols[names[sel[0]]], b = cols[names[sel[1]]], k = pairIndex(sel[0], sel[1]), visible = prog[k] > 0.5;
        g.strokeStyle = c.border; g.strokeRect(sx, sy, sw, sh);
        const A = [Math.min(...a), Math.max(...a)], B = [Math.min(...b), Math.max(...b)];
        if (visible) { g.fillStyle = c.blue; g.globalAlpha = 0.6; a.forEach((v, t) => { g.beginPath(); g.arc(sx + 4 + (v - A[0]) / (A[1] - A[0]) * (sw - 8), sy + sh - 4 - (b[t] - B[0]) / (B[1] - B[0]) * (sh - 8), 2.5, 0, 7); g.fill(); }); g.globalAlpha = 1; }
        txt(g, `${names[sel[0]]} (x) vs ${names[sel[1]]} (y)`, sx, sy - 8, c.fg, '11px sans-serif');
        if (visible) txt(g, `r = ${pearson(a, b).toFixed(2)}, ρ = ${spearman(a, b).toFixed(2)}`, sx, sy + sh + 16, c.muted, '11px sans-serif');
      }
    };
    const text = () => {
      const th = +thIn.value; q(ctl, '#thv').textContent = th.toFixed(2); q(ctl, '#spv').textContent = spIn.value;
      const hit = pairs.filter(([i, j], k) => k < filled && Math.abs(val(i, j)) >= th).map(([i, j]) => `${names[i]}–${names[j]} (${val(i, j).toFixed(2)})`);
      out.innerHTML = `${Math.min(filled, pairs.length)} of ${pairs.length} pairs computed<br>${hit.length ? 'at or above the threshold: ' + hit.join(', ') : 'no pair at or above the threshold yet'}`;
      cap.textContent = 'Blue = positive, orange = negative; the yellow frame marks |value| above your threshold. Note dist and dist² sit near 0 even though dist² is made from dist: Pearson sees only straight-line association. Switch to Spearman: it ranks the values first, but a U-shape still defeats it.';
    };
    thIn.oninput = () => { dirty = true; text(); }; spIn.oninput = text; meIn.onchange = () => { dirty = true; text(); };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { filled = 0; acc = 0; prog.fill(0); dirty = true; text(); };
    cv.addEventListener('click', (e) => {
      const Rr = cv.getBoundingClientRect(), G = geom(), j = Math.floor((e.clientX - Rr.left - G.x0) / G.cell), i = Math.floor((e.clientY - Rr.top - G.y0) / G.cell);
      if (i >= 0 && j >= 0 && i < K && j < K && i !== j) { sel = [Math.min(i, j), Math.max(i, j)]; dirty = true; }
    });
    o.onresize = () => { dirty = true; };
    text();
    AI.loop(root, (dt) => {
      if (playing && filled < pairs.length) { acc += dt * +spIn.value; if (acc >= 1) { acc -= 1; filled++; text(); } }
      for (let k = 0; k < pairs.length; k++) { const t = k < filled ? 1 : 0; if (prog[k] !== t) { prog[k] += Math.sign(t - prog[k]) * Math.min(Math.abs(t - prog[k]), dt * 1.6); dirty = true; } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 5. Cross-tab filling in ---------- */
  AI.labs['ds-eda-crosstab'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A cross-tab filling in', '200 sign-ups by channel and whether they converted; rows arrive one at a time (illustrative)',
      `<div id="tbl" style="overflow-x:auto"></div>`,
      `${sl('sp', 'Speed (sign-ups per second)', 2, 60, 2, 10)}${sl('mg', 'Merge categories under this share of rows (%)', 0, 30, 1, 0)}
       <label>Cells show <select id="md"><option value="n">counts</option><option value="r">row %</option><option value="c">column %</option></select></label>${btns(false)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const spec = [['Ad', 80, 8], ['Organic', 70, 14], ['Referral', 48, 24], ['Partner', 2, 1]], rg = AI.rng(5), rows = [];
    spec.forEach(([ch, n, k]) => { for (let i = 0; i < n; i++) rows.push([ch, i < k ? 1 : 0]); });
    for (let i = rows.length - 1; i > 0; i--) { const j = Math.floor(rg() * (i + 1)); [rows[i], rows[j]] = [rows[j], rows[i]]; }
    const spIn = q(ctl, '#sp'), mgIn = q(ctl, '#mg'), mdIn = q(ctl, '#md'), out = q(ctl, '#out'), cap = q(ctl, '#cap'), tbl = q(view, '#tbl');
    let m = 20, playing = false, last = -1;
    const wilson = (x, n) => { if (!n) return [0, 1]; const z = 1.96, p = x / n, d = 1 + z * z / n, cc = p + z * z / (2 * n), a = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(cc - a) / d, (cc + a) / d]; };
    const render = () => {
      const n = Math.floor(m), sub = rows.slice(0, n), tot = {}; sub.forEach(([ch, y]) => { (tot[ch] = tot[ch] || [0, 0])[y]++; });
      const thr = +mgIn.value / 100; let grp = {}; let merged = [];
      Object.keys(tot).forEach((ch) => { const sz = tot[ch][0] + tot[ch][1]; const key = sz / n < thr ? 'Other (merged)' : ch; if (key !== ch) merged.push(ch); (grp[key] = grp[key] || [0, 0]); grp[key][0] += tot[ch][0]; grp[key][1] += tot[ch][1]; });
      const keys = Object.keys(grp).sort((a, b) => (grp[b][0] + grp[b][1]) - (grp[a][0] + grp[a][1])), ctot = [0, 1].map((j) => keys.reduce((s, k) => s + grp[k][j], 0)), md = mdIn.value;
      const cell = (k, j) => { const v = grp[k][j], rs = grp[k][0] + grp[k][1]; return md === 'n' ? v : md === 'r' ? (rs ? (100 * v / rs).toFixed(1) + '%' : '-') : (ctot[j] ? (100 * v / ctot[j]).toFixed(1) + '%' : '-'); };
      let h = `<table class="tbl" style="width:100%;border-collapse:collapse;font-size:13px"><tr><th style="text-align:left">Channel</th><th>No</th><th>Yes</th><th>Total</th><th>Rate (Yes)</th><th>95% interval</th></tr>`;
      keys.forEach((k) => {
        const t = grp[k][0] + grp[k][1], rate = grp[k][1] / t, w = wilson(grp[k][1], t), thin = t < 30;
        h += `<tr style="${thin ? 'color:var(--yellow)' : ''}"><td>${k}${thin ? ' ⚠' : ''}</td><td>${cell(k, 0)}</td><td>${cell(k, 1)}</td><td>${t}</td><td><span style="display:inline-block;height:9px;width:${Math.round(rate * 90)}px;background:var(--blue);border-radius:2px"></span> ${(100 * rate).toFixed(1)}%</td><td>${(100 * w[0]).toFixed(0)}% to ${(100 * w[1]).toFixed(0)}%</td></tr>`;
      });
      h += `<tr style="border-top:1px solid var(--border);font-weight:600"><td>All</td><td>${md === 'n' ? ctot[0] : md === 'c' ? '100%' : (100 * ctot[0] / n).toFixed(1) + '%'}</td><td>${md === 'n' ? ctot[1] : md === 'c' ? '100%' : (100 * ctot[1] / n).toFixed(1) + '%'}</td><td>${n}</td><td>${(100 * ctot[1] / n).toFixed(1)}%</td><td></td></tr></table>`;
      tbl.innerHTML = h;
      q(ctl, '#spv').textContent = spIn.value; q(ctl, '#mgv').textContent = mgIn.value + '%';
      out.innerHTML = `${n} of 200 sign-ups so far${merged.length ? `<br>merged into Other: ${merged.join(', ')}` : ''}`;
      cap.textContent = n < 200 ? 'Each new row lands in one cell; the rate and its interval settle as counts grow. ⚠ marks rows with fewer than 30 sign-ups.' : 'Final table. Partner has 2 sign-ups and a rate of 50%, but its interval runs from 9% to 91%: too little data to trust. Raise the merge slider to fold it into Other.';
    };
    spIn.oninput = render; mgIn.oninput = render; mdIn.onchange = render;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { m = 20; last = -1; render(); };
    render();
    AI.loop(root, (dt) => {
      if (!playing || m >= 200) return;
      m = Math.min(200, m + dt * +spIn.value); if (Math.floor(m) !== last) { last = Math.floor(m); render(); }
    }, { fps: 20 });
  };

  /* ---------- 6. Simpson's paradox ---------- */
  AI.labs['ds-eda-simpson'] = function (root) {
    const rate = { A: { s: 81 / 87, l: 192 / 263 }, B: { s: 234 / 270, l: 55 / 80 } }, NT = 350;
    const { view, ctl } = AI.labShell(root, 'Simpson\'s paradox', 'two kidney-stone treatments; the published counts (Charig et al. 1986). Change who gets the hard cases (illustrative sweep)',
      `<canvas style="height:320px"></canvas>`,
      `${sl('sa', 'Share of A patients with large stones', 0, 1, 0.01, 0.75)}${sl('sb', 'Share of B patients with large stones', 0, 1, 0.01, 0.23)}${sl('sp', 'Sweep speed', 0.02, 0.3, 0.01, 0.06)}
       <div class="row"><button class="btn primary" id="play">▶ Sweep A's share</button><button class="btn" id="real">Real data</button><button class="btn" id="reset">Equal mix</button></div>
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const saIn = q(ctl, '#sa'), sbIn = q(ctl, '#sb'), spIn = q(ctl, '#sp'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let sa = 263 / 350, sb = 80 / 350, playing = false, dir = -1, dirty = true; let disp = null;
    const target = () => ({ As: rate.A.s, Al: rate.A.l, Bs: rate.B.s, Bl: rate.B.l, Ao: (1 - sa) * rate.A.s + sa * rate.A.l, Bo: (1 - sb) * rate.B.s + sb * rate.B.l });
    const setSliders = () => { saIn.value = sa.toFixed(2); sbIn.value = sb.toFixed(2); };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const d = disp, yb = Hh - 44, yt = 26, Y = (v) => yb - (v - 0.5) / 0.5 * (yb - yt);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(30, yb); g.lineTo(W - 10, yb); g.stroke();
      [0.5, 0.6, 0.7, 0.8, 0.9, 1].forEach((t) => { g.globalAlpha = 0.35; g.beginPath(); g.moveTo(30, Y(t)); g.lineTo(W - 10, Y(t)); g.stroke(); g.globalAlpha = 1; txt(g, Math.round(t * 100) + '%', 26, Y(t) + 4, c.muted, '11px sans-serif', 'right'); });
      const groups = [['Small stones', d.As, d.Bs], ['Large stones', d.Al, d.Bl], ['Everyone (pooled)', d.Ao, d.Bo]], gw = (W - 50) / 3;
      groups.forEach(([nm, a, b], i) => {
        const cx = 36 + gw * i + gw / 2, bw = Math.min(46, gw * 0.3); g.globalAlpha = i === 2 ? 1 : 0.85;
        g.fillStyle = c.blue; g.fillRect(cx - bw - 4, Y(a), bw, yb - Y(a)); g.fillStyle = c.orange; g.fillRect(cx + 4, Y(b), bw, yb - Y(b)); g.globalAlpha = 1;
        txt(g, (100 * a).toFixed(1) + '%', cx - bw / 2 - 4, Y(a) - 5, c.blue, 'bold 11px sans-serif', 'center'); txt(g, (100 * b).toFixed(1) + '%', cx + bw / 2 + 4, Y(b) - 5, c.orange, 'bold 11px sans-serif', 'center');
        txt(g, nm, cx, yb + 15, c.fg, '12px sans-serif', 'center'); txt(g, a > b ? 'A wins' : 'B wins', cx, yb + 31, a > b ? c.blue : c.orange, 'bold 11px sans-serif', 'center');
      });
      g.fillStyle = c.blue; g.fillRect(40, 6, 10, 10); txt(g, 'Treatment A', 54, 15, c.fg, '11px sans-serif'); g.fillStyle = c.orange; g.fillRect(140, 6, 10, 10); txt(g, 'Treatment B', 154, 15, c.fg, '11px sans-serif');
    };
    const text = () => {
      const t = target(), nAl = Math.round(sa * NT), nBl = Math.round(sb * NT);
      q(ctl, '#sav').textContent = sa.toFixed(2); q(ctl, '#sbv').textContent = sb.toFixed(2); q(ctl, '#spv').textContent = spIn.value;
      const win = (a, b) => (a > b ? 'A' : 'B'), para = win(t.As, t.Bs) === win(t.Al, t.Bl) && win(t.As, t.Bs) !== win(t.Ao, t.Bo);
      out.innerHTML = `${NT} patients per treatment. A: ${NT - nAl} small + ${nAl} large; B: ${NT - nBl} small + ${nBl} large<br>small stones: A ${(100 * t.As).toFixed(1)}% vs B ${(100 * t.Bs).toFixed(1)}%<br>large stones: A ${(100 * t.Al).toFixed(1)}% vs B ${(100 * t.Bl).toFixed(1)}%<br><b>pooled: A ${(100 * t.Ao).toFixed(1)}% vs B ${(100 * t.Bo).toFixed(1)}%</b>`;
      cap.innerHTML = para ? '<b>Paradox:</b> A is better in both kinds of stone, yet B looks better overall, because A was given far more of the hard (large) cases.' : win(t.As, t.Bs) === win(t.Ao, t.Bo) ? 'The pooled comparison agrees with the groups: the two treatments are given a similar mix of cases, so pooling does no harm.' : 'Pooled and split comparisons disagree.';
    };
    const upd = () => { setSliders(); text(); dirty = true; };
    saIn.oninput = () => { sa = +saIn.value; text(); dirty = true; }; sbIn.oninput = () => { sb = +sbIn.value; text(); dirty = true; }; spIn.oninput = text;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Sweep A\'s share'; };
    q(ctl, '#real').onclick = () => { sa = 263 / 350; sb = 80 / 350; upd(); };
    q(ctl, '#reset').onclick = () => { sa = 0.5; sb = 0.5; upd(); };
    o.onresize = () => { dirty = true; };
    disp = target(); upd();
    AI.loop(root, (dt) => {
      if (playing) { sa += dir * dt * +spIn.value; if (sa <= 0.05) { sa = 0.05; dir = 1; } if (sa >= 0.95) { sa = 0.95; dir = -1; } setSliders(); text(); dirty = true; }
      const t = target(), k = 1 - Math.exp(-dt * 6);
      Object.keys(t).forEach((key) => { const dd = t[key] - disp[key]; if (Math.abs(dd) > 0.0005) { disp[key] += dd * k; dirty = true; } });
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 7. Groups, effect size and sample size ---------- */
  AI.labs['ds-eda-groups'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Comparing groups: effect size versus sample size', 'two groups of simulated scores (mean 50, sd 10); new samples drawn again and again (illustrative)',
      `<canvas style="height:340px"></canvas>`,
      `${sl('d', 'True effect size d (difference in sd units)', 0, 1.5, 0.05, 0.5)}${sl('n', 'Sample size per group n', 4, 400, 1, 8)}${sl('sp', 'Speed (pairs per second)', 1, 20, 1, 3)}${btns(true)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const dIn = q(ctl, '#d'), nIn = q(ctl, '#n'), spIn = q(ctl, '#sp'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let draws = 0, za = [], zb = [], shown = 0, acc = 0, hold = 0, playing = false, dirty = true, hist = [];
    const gen = () => { const r = AI.rng(1000 + draws * 37), N = () => AI.randn(r); za = Array.from({ length: 400 }, N); zb = Array.from({ length: 400 }, N); };
    const A = () => za.slice(0, Math.min(shown, +nIn.value)).map((z) => 50 + 10 * z), B = () => zb.slice(0, Math.min(shown, +nIn.value)).map((z) => 50 + 10 * (+dIn.value) + 10 * z);
    const jit = (i) => ((i * 0.6180339887) % 1) - 0.5;
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const a = A(), b = B(), lw = W * 0.62, yb = Hh - 28, yt = 14, lo = 5, hi = 100, Y = (v) => yb - (v - lo) / (hi - lo) * (yb - yt);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(34, yb); g.lineTo(lw, yb); g.moveTo(34, yb); g.lineTo(34, yt); g.stroke();
      tickList(lo, hi, 6).forEach((t) => { g.globalAlpha = 0.3; g.beginPath(); g.moveTo(34, Y(t)); g.lineTo(lw, Y(t)); g.stroke(); g.globalAlpha = 1; txt(g, t, 30, Y(t) + 4, c.muted, '11px sans-serif', 'right'); });
      [[a, c.blue, 'Group A', 34 + (lw - 34) * 0.27], [b, c.orange, 'Group B', 34 + (lw - 34) * 0.73]].forEach(([v, col, nm, cx]) => {
        const st = stats(v), sp = Math.min(60, (lw - 34) * 0.2);
        txt(g, nm, cx, yb + 16, c.fg, '12px sans-serif', 'center');
        g.fillStyle = col; g.globalAlpha = 0.55; v.forEach((x, i) => { g.beginPath(); g.arc(cx + jit(i) * sp * 1.6, Y(x), 3, 0, 7); g.fill(); }); g.globalAlpha = 1;
        if (st && v.length >= 2) {
          g.strokeStyle = c.fg; g.lineWidth = 1.5; g.fillStyle = AI.mix(c.bg2, col, 0.3); const bx = cx - 14, bw = 28;
          g.beginPath(); g.moveTo(cx + 36, Y(st.wlo)); g.lineTo(cx + 36, Y(st.whi)); g.stroke();
          g.fillRect(cx + 22, Y(st.q3), 28, Y(st.q1) - Y(st.q3)); g.strokeRect(cx + 22, Y(st.q3), 28, Y(st.q1) - Y(st.q3));
          g.strokeStyle = c.green; g.lineWidth = 2.5; g.beginPath(); g.moveTo(cx + 22, Y(st.med)); g.lineTo(cx + 50, Y(st.med)); g.stroke(); g.lineWidth = 1;
        }
      });
      // p-value history on the right (log scale)
      const hx = lw + 24, hw = W - hx - 10, py = (p) => yt + (Math.log10(Math.max(p, 1e-4)) / -4) * (yb - yt);
      txt(g, 'p-value of each finished draw', hx, yt - 2, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.strokeRect(hx, yt + 4, hw, yb - yt - 4);
      [1, 0.1, 0.01, 0.001, 0.0001].forEach((p) => { g.globalAlpha = 0.35; g.beginPath(); g.moveTo(hx, py(p) + 4); g.lineTo(hx + hw, py(p) + 4); g.stroke(); g.globalAlpha = 1; txt(g, p, hx + hw - 2, py(p) + 2, c.muted, '10px sans-serif', 'right'); });
      g.strokeStyle = c.red; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(hx, py(0.05) + 4); g.lineTo(hx + hw, py(0.05) + 4); g.stroke(); g.setLineDash([]); txt(g, '0.05', hx + 3, py(0.05) + 1, c.red, '10px sans-serif');
      hist.forEach((h, i) => { g.fillStyle = h.p < 0.05 ? c.green : c.muted; g.beginPath(); g.arc(hx + 10 + i * ((hw - 20) / 13), py(h.p) + 4, 4, 0, 7); g.fill(); });
    };
    const finish = () => { const w = welch(A(), B()); if (w) { hist.push({ p: w.p, d: w.d }); if (hist.length > 14) hist.shift(); } };
    const text = () => {
      const a = A(), b = B(), w = welch(a, b); q(ctl, '#dv').textContent = (+dIn.value).toFixed(2); q(ctl, '#nv').textContent = nIn.value; q(ctl, '#spv').textContent = spIn.value;
      if (!w) { out.textContent = `draw ${draws + 1}: waiting for at least 2 pairs`; cap.textContent = 'Points land in each group; the box shows Q1, median (green) and Q3.'; return; }
      out.innerHTML = `draw ${draws + 1}, n = ${a.length} per group<br>mean A ${AI.fmt(w.ma, 2)}, mean B ${AI.fmt(w.mb, 2)}, difference ${AI.fmt(w.diff, 2)}<br>sample effect size d = ${AI.fmt(w.d, 2)}<br>Welch t = ${AI.fmt(w.t, 2)}, p = ${w.p < 0.0001 ? '< 0.0001' : AI.fmt(w.p, 4)}`;
      cap.textContent = a.length < +nIn.value ? 'Pairs are still arriving: watch the estimated d and p wander.' : (+dIn.value < 0.2 && +nIn.value > 150) ? 'A tiny true effect with a big sample: p can be very small although the groups barely differ. Statistical significance is not practical importance.' : +nIn.value < 15 ? 'With few observations p jumps around between draws even though the true effect is fixed. Press Play for several draws.' : 'Larger n makes p stable: it reflects whether an effect exists, d says how big it is.';
    };
    const newDraw = () => { draws++; gen(); shown = 0; acc = 0; hold = 0; dirty = true; text(); };
    dIn.oninput = () => { dirty = true; text(); }; nIn.oninput = () => { dirty = true; text(); }; spIn.oninput = text;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { if (shown < +nIn.value) { shown++; if (shown >= +nIn.value) finish(); } else newDraw(); dirty = true; text(); };
    q(ctl, '#reset').onclick = () => { hist = []; draws = 0; gen(); shown = 0; acc = 0; hold = 0; dirty = true; text(); };
    o.onresize = () => { dirty = true; };
    gen(); shown = 4; text();
    AI.loop(root, (dt) => {
      const n = +nIn.value;
      if (playing) {
        if (shown < n) { acc += dt * +spIn.value * Math.max(1, n / 40); const k = Math.floor(acc); if (k > 0) { acc -= k; shown = Math.min(n, shown + k); if (shown >= n) finish(); dirty = true; text(); } }
        else { hold += dt; if (hold > 2.5) newDraw(); }
      }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 8. Outliers: threshold moves, points light up ---------- */
  AI.labs['ds-eda-outliers'] = function (root) {
    const rows = customers();
    const { view, ctl } = AI.labShell(root, 'Outliers: where is the line?', '200 simulated customers; points arrive one by one and are flagged by your rule (illustrative)',
      `<canvas style="height:300px"></canvas>`,
      `<label>Column <select id="col"><option value="age">age (years)</option><option value="income">income (thousands)</option><option value="spend">monthly spend</option></select></label>
       <label>Rule <select id="ru"><option value="iqr">IQR fence: k × IQR beyond the quartiles</option><option value="z">z-score: |z| above k</option></select></label>
       ${sl('k', 'Threshold k', 0.5, 5, 0.1, 1.5)}${sl('sp', 'Speed (rows per second)', 5, 80, 5, 25)}<label style="flex-direction:row;align-items:center;gap:8px"><input type="checkbox" id="zm"> zoom the axis to the middle 98%</label>${btns(false)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const colIn = q(ctl, '#col'), ruIn = q(ctl, '#ru'), kIn = q(ctl, '#k'), spIn = q(ctl, '#sp'), zmIn = q(ctl, '#zm'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let shown = 30, playing = false, dirty = true, fa = null, fb = null;
    const vals = () => rows.slice(0, Math.floor(shown)).map((r) => r[colIn.value]).filter((v) => v !== null);
    const allv = () => rows.map((r) => r[colIn.value]).filter((v) => v !== null);
    const rule = (a) => { const st = stats(a, +kIn.value); if (ruIn.value === 'iqr') return { lo: st.lo, hi: st.hi, st }; return { lo: st.mean - +kIn.value * st.sd, hi: st.mean + +kIn.value * st.sd, st }; };
    const range = () => { const a = allv().sort((x, y) => x - y); return zmIn.checked ? [quant(a, 0.01) - 2, quant(a, 0.99) + 2] : [Math.min(0, a[0]) - (a[0] < 0 ? 5 : 0), a[a.length - 1] * 1.03]; };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const [lo, hi] = range(), x0 = 20, x1 = W - 20, X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0), a = vals(), R = rule(a), ay = Hh - 60;
      if (fa === null) { fa = R.lo; fb = R.hi; }
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, ay); g.lineTo(x1, ay); g.stroke();
      tickList(lo, hi, 7).forEach((t) => { g.beginPath(); g.moveTo(X(t), ay); g.lineTo(X(t), ay + 4); g.stroke(); txt(g, AI.fmt(t, 3), X(t), ay + 17, c.muted, '11px sans-serif', 'center'); });
      // fences (tweened)
      g.fillStyle = c.red; g.globalAlpha = 0.1; g.fillRect(x0, 20, clamp(X(fa), x0, x1) - x0, ay - 20); g.fillRect(clamp(X(fb), x0, x1), 20, x1 - clamp(X(fb), x0, x1), ay - 20); g.globalAlpha = 1;
      g.strokeStyle = c.red; g.setLineDash([5, 4]); [fa, fb].forEach((f) => { if (X(f) >= x0 && X(f) <= x1) { g.beginPath(); g.moveTo(X(f), 20); g.lineTo(X(f), ay); g.stroke(); } }); g.setLineDash([]);
      const col = colIn.value; let n = 0, fl = 0;
      rows.slice(0, Math.floor(shown)).forEach((r, i) => { const v = r[col]; if (v === null) return; n++; const bad = v < R.lo || v > R.hi; if (bad) fl++; const x = X(v); if (x < x0 - 2 || x > x1 + 2) return; const y = 30 + ((i * 0.6180339887) % 1) * (ay - 50); g.fillStyle = bad ? c.red : c.blue; g.globalAlpha = bad ? 1 : 0.6; g.beginPath(); g.arc(x, y, bad ? 4.5 : 3.2, 0, 7); g.fill(); g.globalAlpha = 1; });
      txt(g, `${fl} of ${n} flagged`, x1, 14, c.red, 'bold 12px sans-serif', 'right');
    };
    const text = () => {
      const a = vals(), R = rule(a), st = R.st, bad = a.filter((v) => v < R.lo || v > R.hi), keep = a.filter((v) => v >= R.lo && v <= R.hi), s2 = keep.length ? stats(keep) : st;
      q(ctl, '#kv').textContent = (+kIn.value).toFixed(1); q(ctl, '#spv').textContent = spIn.value;
      const miss = rows.slice(0, Math.floor(shown)).length - a.length;
      out.innerHTML = `rows shown ${Math.floor(shown)}${miss ? ` (${miss} missing ignored)` : ''}; flagged ${bad.length}<br>rule bounds: ${AI.fmt(R.lo, 4)} to ${AI.fmt(R.hi, 4)}<br>mean ${AI.fmt(st.mean, 3)}, sd ${AI.fmt(st.sd, 3)} with all; mean ${AI.fmt(s2.mean, 3)}, sd ${AI.fmt(s2.sd, 3)} without the flagged${bad.length ? `<br>largest flagged: ${AI.fmt(Math.max(...bad.map((v) => Math.abs(v))), 5)}` : ''}`;
      cap.textContent = colIn.value === 'age' ? 'Age has data-entry errors (999 twice, 0 once): any rule catches them, and they are errors to fix, not rare customers.' : colIn.value === 'income' ? 'Income is right-skewed: a symmetric rule flags many honest high earners. Moving k trades missed errors against false alarms. Consider the log scale first.' : 'Spend is noisy and skewed; the flagged points are unusual, not necessarily wrong. Ask why before deleting.';
    };
    const chg = () => { fa = fb = null; dirty = true; text(); };
    colIn.onchange = chg; ruIn.onchange = chg; zmIn.onchange = chg; kIn.oninput = () => { dirty = true; text(); }; spIn.oninput = text;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { shown = 30; chg(); };
    o.onresize = () => { dirty = true; };
    text();
    AI.loop(root, (dt) => {
      if (playing && shown < 200) { const prev = Math.floor(shown); shown = Math.min(200, shown + dt * +spIn.value); if (Math.floor(shown) !== prev) { dirty = true; text(); } }
      const R = rule(vals()), k = 1 - Math.exp(-dt * 7);
      if (fa === null) { fa = R.lo; fb = R.hi; dirty = true; }
      if (Math.abs(fa - R.lo) > 1e-3 * (Math.abs(R.lo) + 1) || Math.abs(fb - R.hi) > 1e-3 * (Math.abs(R.hi) + 1)) { fa += (R.lo - fa) * k; fb += (R.hi - fb) * k; dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ---------- 9. The EDA log: a workflow that ends in a decision ---------- */
  AI.labs['ds-eda-log'] = function (root) {
    const rows = customers();
    const { view, ctl } = AI.labShell(root, 'An EDA run, step by step, with a written log', '200 simulated customers; six checks and a decision (illustrative)',
      `<canvas style="height:250px"></canvas><div id="log" style="font-size:13px;margin-top:8px;max-height:230px;overflow:auto"></div>`,
      `${sl('sec', 'Seconds per step', 2, 10, 1, 4)}<label style="flex-direction:row;align-items:center;gap:8px"><input type="checkbox" id="dr" checked> drop rows with impossible ages</label><label style="flex-direction:row;align-items:center;gap:8px"><input type="checkbox" id="im" checked> fill missing income with the median</label>${btns(true)}
       <div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col(), logEl = q(view, '#log');
    const secIn = q(ctl, '#sec'), drIn = q(ctl, '#dr'), imIn = q(ctl, '#im'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const bad = (r) => r.age < 18 || r.age > 100, STEPS = 6;
    const inc = rows.filter((r) => r.income !== null).map((r) => r.income), logInc = inc.map(Math.log10);
    const sRaw = stats(inc), sLog = stats(logInc);
    const goodRows = rows.filter((r) => !bad(r)); const nBad = rows.length - goodRows.length, nMiss = rows.filter((r) => r.income === null).length;
    const cc = (rs, a, b) => pearson(rs.filter((r) => r.income !== null).map((r) => r[a]), rs.filter((r) => r.income !== null).map((r) => r[b]));
    const pairsC = [['income', 'spend'], ['tenure', 'churn'], ['age', 'income'], ['age', 'spend'], ['tenure', 'spend'], ['income', 'churn']].map(([a, b]) => [a, b, cc(goodRows, a, b)]).sort((x, y) => Math.abs(y[2]) - Math.abs(x[2]));
    const medInc = stats(inc).med;
    let step = -1, sweep = [0, 0, 0, 0, 0, 0], tStep = 0, playing = false, dirty = true; const lines = [];
    const logLine = (i) => {
      const keepN = drIn.checked ? rows.length - nBad : rows.length;
      return [
        `<b>1 Shape.</b> Looked at: row and column counts. Found: ${rows.length} rows, 5 columns (age, tenure, income, spend, churn). Changed: nothing.`,
        `<b>2 Missing.</b> Looked at: blank cells per column. Found: income is missing in ${nMiss} rows (${(100 * nMiss / rows.length).toFixed(0)}%), others complete. Changed: nothing yet; to decide how to fill.`,
        `<b>3 Ranges.</b> Looked at: min and max of age. Found: ${nBad} impossible ages (999, 999, 0). Changed: marked these rows as data-entry errors.`,
        `<b>4 Shapes.</b> Looked at: income histogram. Found: skew ${AI.fmt(sRaw.skew, 2)} raw, ${AI.fmt(sLog.skew, 2)} after log₁₀; mean ${AI.fmt(sRaw.mean, 1)} vs median ${AI.fmt(sRaw.med, 1)}. Changed: will model log income.`,
        `<b>5 Relationships.</b> Looked at: pairwise Pearson r on the clean rows. Found: strongest are ${pairsC.slice(0, 2).map((p) => `${p[0]}–${p[1]} r = ${p[2].toFixed(2)}`).join(' and ')}. Changed: nothing; correlation is not cause.`,
        `<b>6 Decision.</b> ${drIn.checked ? `Drop the ${nBad} bad-age rows` : 'Keep every row (bad ages stay in: age features will be wrong)'}, ${imIn.checked ? `fill ${nMiss} missing incomes with the median ${AI.fmt(medInc, 1)} and add a "was missing" flag` : 'leave missing incomes blank'}, use log income. ${keepN} rows go to the model; mean age after = ${AI.fmt(mean((drIn.checked ? goodRows : rows).map((r) => r.age)), 1)}.`
      ][i];
    };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh); const cols = 25, rws = 8, cw = Math.min((W - 20) / cols, 24), ch = Math.min((Hh - 40) / rws, 24), ox = (W - cols * cw) / 2;
      rows.forEach((r, i) => {
        const cx = ox + (i % cols) * cw, cy = 28 + Math.floor(i / cols) * ch, vis = sweep[0] > i; if (!vis) return;
        let col = c.bg3, ring = null; const dropped = bad(r) && step >= 2 && drIn.checked && step >= 5;
        if (r.income === null && step >= 1 && sweep[1] > i) col = (step >= 5 && imIn.checked) ? c.teal : c.yellow;
        if (bad(r) && step >= 2 && sweep[2] > i) col = c.red;
        if (step >= 3 && sweep[3] > i && r.income !== null && r.income > sRaw.q3 + 1.5 * sRaw.iqr) ring = c.purple;
        g.fillStyle = col; g.globalAlpha = dropped ? 0.35 : 1; g.fillRect(cx + 1, cy + 1, cw - 2, ch - 2); g.globalAlpha = 1;
        if (ring) { g.strokeStyle = ring; g.lineWidth = 2; g.strokeRect(cx + 2, cy + 2, cw - 4, ch - 4); g.lineWidth = 1; }
        if (dropped) { g.strokeStyle = c.fg; g.beginPath(); g.moveTo(cx + 3, cy + 3); g.lineTo(cx + cw - 3, cy + ch - 3); g.stroke(); }
      });
      const key = [[c.bg3, 'customer row'], [c.yellow, 'income missing'], [c.red, 'impossible age'], [c.teal, 'filled'], [c.purple, 'high-income (IQR fence)']]; let x = 8;
      key.forEach(([col, nm]) => { g.fillStyle = col; g.fillRect(x, 8, 10, 10); txt(g, nm, x + 14, 17, c.muted, '11px sans-serif'); x += 22 + nm.length * 6; });
    };
    const text = () => {
      q(ctl, '#secv').textContent = secIn.value + ' s';
      const names = ['Shape', 'Missing values', 'Ranges and errors', 'Distribution shapes', 'Relationships', 'Decision'];
      out.innerHTML = step < 0 ? 'Press Play or Step to start the EDA.' : `step ${step + 1} of ${STEPS}: <b>${names[step]}</b>`;
      cap.textContent = 'Each step writes one line: what was looked at, what was found, what changed. The decision line responds to the two checkboxes.';
      logEl.innerHTML = lines.map((i) => `<div style="padding:5px 0;border-top:1px solid var(--border)">${logLine(i)}</div>`).join('');
      logEl.scrollTop = logEl.scrollHeight;
    };
    const advance = () => { if (step < STEPS - 1) { step++; if (!lines.includes(step)) lines.push(step); tStep = 0; text(); dirty = true; } };
    drIn.onchange = () => { text(); dirty = true; }; imIn.onchange = () => { text(); dirty = true; }; secIn.oninput = text;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; if (playing && step < 0) advance(); };
    q(ctl, '#step').onclick = advance;
    q(ctl, '#reset').onclick = () => { step = -1; sweep = [0, 0, 0, 0, 0, 0]; lines.length = 0; tStep = 0; text(); dirty = true; };
    o.onresize = () => { dirty = true; };
    text();
    AI.loop(root, (dt) => {
      if (sweep[0] < 200) { sweep[0] = Math.min(200, sweep[0] + dt * 60); dirty = true; }
      for (let s = 1; s <= 5; s++) if (step >= s && sweep[s] < 200) { sweep[s] = Math.min(200, sweep[s] + dt * 90); dirty = true; }
      if (playing && step >= 0) { tStep += dt; if (tStep > +secIn.value) { if (step < STEPS - 1) advance(); else { playing = false; q(ctl, '#play').textContent = '▶ Play'; } } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };
  /* readouts wrap instead of scrolling sideways; every lab here gets the class */
  const st = document.createElement('style'); st.textContent = '.dseda .readout{white-space:normal;overflow-wrap:anywhere}'; document.head.appendChild(st);
  Object.keys(AI.labs).filter((k) => k.indexOf('ds-eda-') === 0).forEach((k) => { const f = AI.labs[k]; AI.labs[k] = (root) => { root.classList.add('dseda'); return f(root); }; });
})();
