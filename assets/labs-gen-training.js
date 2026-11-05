/* Generative AI Module 3 labs: training large language models.
   gen-tn-objective, gen-tn-packing, gen-tn-datapipe, gen-tn-contam, gen-tn-fit, gen-tn-compute,
   gen-tn-schedule, gen-tn-precision, gen-tn-memory, gen-tn-bubble, gen-tn-run.
   All simulations are illustrative toys: seeded random numbers (AI.rng), time-based (dt), slow by default, with speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => { t = Math.max(0, Math.min(1, t)); return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const sci = (v) => (v === 0 ? '0' : (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : String(+v.toPrecision(4))));

  /* ================= pure helpers (also exercised from node) ================= */
  const logsum = (a) => Math.log(a.reduce((s, x) => s + Math.exp(x), 0));
  // round to a binary float with `bits` stored mantissa bits (ties to even); exponent range handled only for fp16 overflow
  const roundEven = (x) => { const f = Math.floor(x), d = x - f; if (d < 0.5) return f; if (d > 0.5) return f + 1; return f % 2 === 0 ? f : f + 1; };
  const roundMant = (x, bits, maxVal) => {
    if (!isFinite(x) || x === 0) return x;
    const a = Math.abs(x), e = Math.floor(Math.log2(a)), s = Math.pow(2, e - bits);
    let r = roundEven(a / s) * s;
    if (maxVal && r > maxVal) r = Infinity;
    return x < 0 ? -r : r;
  };
  const FMT = { bf16: { bits: 7, name: 'bfloat16', max: 3.38e38 }, fp16: { bits: 10, name: 'float16', max: 65504 }, fp32: { bits: 23, name: 'float32', max: 3.4e38 } };
  // power-law fit  L = E + A N^-alpha  by grid over E and log-linear regression
  const fitPower = (xs, ys) => {
    const n = xs.length, lx = xs.map((x) => Math.log(x)), ymin = Math.min(...ys);
    let best = null;
    for (let i = 0; i <= 160; i++) {
      const E = (ymin - 0.002) * (i / 160);
      const z = ys.map((y) => Math.log(y - E));
      const mx = lx.reduce((s, v) => s + v, 0) / n, mz = z.reduce((s, v) => s + v, 0) / n;
      let sxx = 0, sxz = 0;
      for (let k = 0; k < n; k++) { sxx += (lx[k] - mx) ** 2; sxz += (lx[k] - mx) * (z[k] - mz); }
      const slope = sxz / sxx, icpt = mz - slope * mx;
      let sse = 0;
      for (let k = 0; k < n; k++) { const p = E + Math.exp(icpt + slope * lx[k]); sse += (ys[k] - p) ** 2; }
      if (!best || sse < best.sse) best = { E, A: Math.exp(icpt), al: -slope, sse };
    }
    return best;
  };
  const predictPow = (p, x) => p.E + p.A * Math.pow(x, -p.al);
  const bootFit = (xs, ys, nb, seed) => {
    const p = fitPower(xs, ys), n = xs.length, r = AI.rng(seed);
    const sigma = Math.max(0.003, Math.sqrt(p.sse / Math.max(1, n - 3)));
    const fits = [];
    for (let b = 0; b < nb; b++) {
      const yb = xs.map((x) => predictPow(p, x) + sigma * AI.randn(r));
      fits.push(fitPower(xs, yb));
    }
    return { p, sigma, fits };
  };
  const pct = (a, f) => { const s = a.slice().sort((x, y) => x - y); return s[clamp(Math.round(f * (s.length - 1)), 0, s.length - 1)]; };
  // toy two-term loss, compute-optimal sizing
  const TOY = { E: 1.5, A: 300, B: 500, a: 0.30, b: 0.25 };
  const toyLoss = (N, D) => TOY.E + TOY.A / Math.pow(N, TOY.a) + TOY.B / Math.pow(D, TOY.b);
  const toyOptN = (C) => Math.pow((TOY.a * TOY.A) / (TOY.b * TOY.B), 1 / (TOY.a + TOY.b)) * Math.pow(C / 6, TOY.b / (TOY.a + TOY.b));
  const lrAt = (t, T, W, peak, minR, shape) => {
    if (t < W) return peak * (t / Math.max(1, W));
    const pr = clamp((t - W) / Math.max(1, T - W), 0, 1);
    if (shape === 'const') return peak;
    if (shape === 'linear') return peak * (minR + (1 - minR) * (1 - pr));
    return peak * (minR + (1 - minR) * 0.5 * (1 + Math.cos(Math.PI * pr)));
  };
  const memStates = (N, Nd, stage) => {
    const w = 2 * N, g = 2 * N, o = 12 * N;
    return { w: stage >= 3 ? w / Nd : w, g: stage >= 2 ? g / Nd : g, mas: stage >= 1 ? 4 * N / Nd : 4 * N, m: stage >= 1 ? 4 * N / Nd : 4 * N, v: stage >= 1 ? 4 * N / Nd : 4 * N, o };
  };
  const packLayout = (lens, L) => {
    // returns {packed:[{row,col,doc,eos}], padded:[...], rowsPacked, rowsPadded, docEnd:[token index of end]}
    const toks = []; lens.forEach((n, d) => { for (let i = 0; i < n; i++) toks.push({ doc: d, eos: false }); toks.push({ doc: d, eos: true }); });
    const packed = toks.map((t, i) => ({ row: Math.floor(i / L), col: i % L, doc: t.doc, eos: t.eos }));
    const rowsPacked = Math.ceil(toks.length / L);
    const padded = []; let row = 0;
    lens.forEach((n, d) => { const tl = n + 1; for (let i = 0; i < tl; i++) padded.push({ row: row + Math.floor(i / L), col: i % L, doc: d, eos: i === tl - 1 }); row += Math.ceil(tl / L); });
    return { toks, packed, padded, rowsPacked, rowsPadded: row };
  };
  AI._gtn = { roundMant, fitPower, predictPow, bootFit, pct, toyLoss, toyOptN, TOY, lrAt, memStates, packLayout, logsum };

  /* =====================================================================
     1. The next-token loss on a six-token sentence
     ===================================================================== */
  const V = ['the', 'cat', 'sat', 'on', 'mat', '⟨eos⟩'];
  const SEQ = ['the', 'cat', 'sat', 'on', 'the', 'mat', '⟨eos⟩'];
  const TRUE_IDX = [1, 2, 3, 0, 4, 5]; // true next token index at each of the 6 positions
  const BASE = [
    [0.03, 0.30, 0.07, 0.10, 0.45, 0.05],
    [0.03, 0.02, 0.50, 0.25, 0.05, 0.15],
    [0.05, 0.02, 0.03, 0.60, 0.05, 0.25],
    [0.70, 0.05, 0.02, 0.03, 0.15, 0.05],
    [0.02, 0.62, 0.03, 0.05, 0.10, 0.18],
    [0.10, 0.05, 0.05, 0.10, 0.30, 0.40],
  ];
  AI.labs['gen-tn-objective'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The next-token loss, one position at a time', 'illustrative: a fixed table of model probabilities for the sentence “the cat sat on the mat”',
      `<canvas style="height:400px"></canvas>`,
      `${sl('sharp', 'Model confidence (power s: p ∝ p^s)', 0, 3, 0.05, 1)}
       <label>Edit the true-token probability at position <select id="epos"><option value="-1">none</option>${[1, 2, 3, 4, 5, 6].map((i) => `<option value="${i - 1}">${i}</option>`).join('')}</select></label>
       ${sl('ptrue', 'p(true next token) at that position', 0.01, 0.99, 0.01, 0.5)}
       ${sl('rate', 'Positions per second', 0.2, 2, 0.1, 0.4)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let pos = 0, hold = 0, dirty = true, ov = [null, null, null, null, null, null];
    const rows = () => {
      const s = +q(ctl, '#sharp').value;
      return BASE.map((r, k) => {
        let w = r.map((p) => Math.pow(p, s)); const z = w.reduce((a, b) => a + b, 0); w = w.map((x) => x / z);
        if (ov[k] !== null) { const t = TRUE_IDX[k], rest = 1 - w[t]; const nr = 1 - ov[k]; w = w.map((x, i) => (i === t ? ov[k] : rest > 0 ? x * nr / rest : nr / 5)); }
        return w;
      });
    };
    const reset = () => { pos = 0; hold = 0; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const R = rows(), k = Math.min(5, Math.floor(pos)), f = pos >= 6 ? 1 : pos - k;
      const ploss = R.map((r, i) => -Math.log(r[TRUE_IDX[i]]));
      // tokens
      const tw = Math.min(70, (W - 20) / 7 - 6), tx = (i) => 10 + i * (tw + 6) + tw / 2;
      txt(g, 'sentence (7 tokens → 6 predictions)', 10, 14, c.muted, '11px sans-serif');
      SEQ.forEach((t, i) => {
        const inCtx = i <= k, isTarget = i === k + 1;
        g.fillStyle = inCtx ? c.blue : 'transparent'; g.globalAlpha = inCtx ? 0.8 : 1;
        g.fillRect(tx(i) - tw / 2, 24, tw, 26); g.globalAlpha = 1;
        g.strokeStyle = isTarget ? c.green : c.border; g.lineWidth = isTarget ? 2.5 : 1; g.strokeRect(tx(i) - tw / 2, 24, tw, 26); g.lineWidth = 1;
        txt(g, t, tx(i), 42, inCtx ? '#fff' : c.fg, '12px sans-serif', 'center');
      });
      txt(g, 'context →', 10, 66, c.blue, '11px sans-serif'); txt(g, `target = “${SEQ[k + 1]}”`, tx(k + 1), 66, c.green, 'bold 11px sans-serif', 'center');
      // probability bars for the current position
      const top = 98, bh = 112, bw = Math.min(54, (W - 40) / 6 - 8), prow = R[k], gf = ease(f / 0.5);
      txt(g, `model's probability for the next token after “${SEQ.slice(0, k + 1).join(' ')}”`, 10, top - 6, c.muted, '11px sans-serif');
      for (let i = 0; i < 6; i++) {
        const x = 14 + i * (bw + 10), hh = prow[i] * bh * gf, isT = i === TRUE_IDX[k];
        g.fillStyle = isT ? c.green : c.blue; g.globalAlpha = isT ? 0.95 : 0.55; g.fillRect(x, top + bh - hh, bw, hh); g.globalAlpha = 1;
        g.strokeStyle = c.border; g.strokeRect(x, top, bw, bh);
        txt(g, V[i], x + bw / 2, top + bh + 13, isT ? c.green : c.fg, (isT ? 'bold ' : '') + '11px sans-serif', 'center');
        if (gf > 0.6) txt(g, prow[i].toFixed(2), x + bw / 2, Math.min(top + bh - 3, top + bh - hh - 3), c.fg, '11px sans-serif', 'center');
      }
      // per-position surprise bars
      const t2 = top + bh + 42, h2 = H - t2 - 22, ymax = 3.2, cw = Math.min(54, (W - 40) / 6 - 8);
      txt(g, 'surprise  −ln p(true token)  at each position (nats)', 10, t2 - 6, c.muted, '11px sans-serif');
      let sum = 0, cnt = 0;
      for (let i = 0; i < 6; i++) {
        const x = 14 + i * (cw + 10), done = i < k || (i === k && f > 0.5);
        const a = i < k ? 1 : i === k ? ease((f - 0.5) / 0.5) : 0, hh = Math.min(1, ploss[i] / ymax) * h2 * a;
        g.strokeStyle = c.border; g.strokeRect(x, t2, cw, h2);
        g.fillStyle = c.orange; g.globalAlpha = 0.85; g.fillRect(x, t2 + h2 - hh, cw, hh); g.globalAlpha = 1;
        txt(g, 'pos ' + (i + 1), x + cw / 2, t2 + h2 + 13, i === k ? c.fg : c.muted, '11px sans-serif', 'center');
        if (a > 0.9) txt(g, ploss[i].toFixed(3), x + cw / 2, Math.min(t2 + h2 - 3, t2 + h2 - hh - 3) , c.fg, '11px sans-serif', 'center');
        if (done) { sum += ploss[i]; cnt++; }
      }
      const meanAll = ploss.reduce((a, b) => a + b, 0) / 6;
      const run = cnt ? sum / cnt : NaN;
      q(ctl, '#out').innerHTML = `position ${k + 1}: p(true) = ${prow[TRUE_IDX[k]].toFixed(3)}, surprise = ${ploss[k].toFixed(3)} nats<br>running mean over ${cnt} scored positions: ${cnt ? run.toFixed(4) : '–'}<br><b>loss over all 6 positions: ${meanAll.toFixed(4)} nats</b>, perplexity e^loss = ${Math.exp(meanAll).toFixed(3)}<br>a model that guesses uniformly over 6 tokens scores ln 6 = ${Math.log(6).toFixed(4)}`;
      q(ctl, '#cap').textContent = pos >= 6 ? 'All six positions scored. The loss is simply the average surprise. Change the confidence slider: sharper probabilities lower the loss at positions where the model was right, and raise it a lot where it was confidently wrong (position 5).' : `Position ${k + 1}: given the context in blue, the true next token is “${SEQ[k + 1]}”. The model gave it probability ${prow[TRUE_IDX[k]].toFixed(2)}; the surprise ${ploss[k].toFixed(2)} is −ln of that. Every position is scored in the same pass.`;
      showVals(ctl, { sharp: (v) => v.toFixed(2), ptrue: (v) => v.toFixed(2), rate: (v) => v });
    };
    const step = () => { pos = Math.min(6, Math.floor(pos) + 1); dirty = true; };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#sharp').oninput = () => { ov = [null, null, null, null, null, null]; dirty = true; };
    q(ctl, '#epos').onchange = () => { const k = +q(ctl, '#epos').value; if (k >= 0) { const r = rows()[k]; q(ctl, '#ptrue').value = r[TRUE_IDX[k]].toFixed(2); ov[k] = +q(ctl, '#ptrue').value; } dirty = true; };
    q(ctl, '#ptrue').oninput = () => { const k = +q(ctl, '#epos').value; if (k >= 0) ov[k] = +q(ctl, '#ptrue').value; dirty = true; };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        if (pos < 6) { pos = Math.min(6, pos + dt * +q(ctl, '#rate').value); dirty = true; }
        else { hold += dt; if (hold > 3) { reset(); } }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Packing documents into fixed windows
     ===================================================================== */
  AI.labs['gen-tn-packing'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Packing documents into training windows', 'illustrative: each cell is one token; the dark cell is the end-of-document token',
      `<canvas style="height:420px"></canvas>`,
      `<label>Document lengths in tokens (comma separated) <input type="text" id="lens" value="5,3,6,2,7,4,3,8" style="width:100%"></label>
       ${sl('L', 'Window length L (tokens)', 4, 16, 1, 8)}
       <label><input type="checkbox" id="mask"> mask attention at document boundaries</label>
       ${sl('rate', 'Tokens per second', 2, 40, 1, 6)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let n = 0, lay = null, dirty = true, tick = AI.ticker();
    const parse = () => { const a = q(ctl, '#lens').value.split(/[\s,]+/).map(Number).filter((v) => v >= 1 && v <= 40).slice(0, 14); return a.length ? a.map(Math.round) : [5, 3, 6]; };
    const rebuild = () => { lay = packLayout(parse(), +q(ctl, '#L').value); n = 0; dirty = true; };
    const hue = (d) => `hsl(${(d * 47) % 360},55%,52%)`;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = +q(ctl, '#L').value, half = W / 2, rowsMax = Math.max(lay.rowsPacked, lay.rowsPadded);
      const cs = Math.max(8, Math.min(26, (half - 24) / L, (H - 70) / rowsMax));
      const grid = (title, cells, rows, ox) => {
        txt(g, title, ox, 16, c.fg, 'bold 12px sans-serif');
        for (let r = 0; r < rows; r++) for (let k = 0; k < L; k++) { g.strokeStyle = c.border; g.strokeRect(ox + k * cs + 0.5, 30 + r * (cs + 6) + 0.5, cs - 1, cs - 1); }
        const nDocsDone = (d) => { let e = 0; for (let i = 0; i < lay.toks.length; i++) if (lay.toks[i].doc === d) e = i; return n > e; };
        cells.forEach((cl, i) => {
          if (i >= n) return;
          g.fillStyle = cl.eos ? c.fg : hue(cl.doc); g.globalAlpha = cl.eos ? 0.85 : 0.9; g.fillRect(ox + cl.col * cs + 1, 30 + cl.row * (cs + 6) + 1, cs - 2, cs - 2); g.globalAlpha = 1;
        });
        // under-row attention spans
        const mask = q(ctl, '#mask').checked;
        for (let r = 0; r < rows; r++) {
          const inRow = cells.map((cl, i) => ({ cl, i })).filter((x) => x.cl.row === r && x.i < n);
          if (!inRow.length) continue;
          const y = 30 + r * (cs + 6) + cs + 2;
          g.strokeStyle = c.muted; g.lineWidth = 2;
          let s = inRow[0];
          const flush = (a, b) => { g.beginPath(); g.moveTo(ox + a.cl.col * cs + 1, y); g.lineTo(ox + b.cl.col * cs + cs - 1, y); g.stroke(); };
          if (!mask) flush(inRow[0], inRow[inRow.length - 1]);
          else { let a = inRow[0]; for (let j = 0; j < inRow.length; j++) { const x = inRow[j]; if (x.cl.eos || j === inRow.length - 1) { flush(a, x); a = inRow[j + 1]; } } }
          g.lineWidth = 1;
        }
        return nDocsDone;
      };
      grid('Packed', lay.packed, lay.rowsPacked, 12);
      // padded grid + pad cells
      grid('One document per window', lay.padded, lay.rowsPadded, half + 6);
      // pad cells in padded layout (grey hatch) for documents already finished
      let end = -1;
      for (let i = 0; i < lay.padded.length; i++) {
        const cl = lay.padded[i];
        if (cl.eos && n > i) { const startCol = cl.col + 1; for (let k = startCol; k < L; k++) { g.fillStyle = c.border; g.globalAlpha = 0.6; g.fillRect(half + 6 + k * cs + 1, 30 + cl.row * (cs + 6) + 1, cs - 2, cs - 2); g.globalAlpha = 1; } }
      }
      const total = lay.toks.length, useP = total / (lay.rowsPacked * L), usePad = total / (lay.rowsPadded * L);
      q(ctl, '#out').innerHTML = `${lay.toks.length} real tokens (documents + ${parse().length} end markers)<br>packed: ${lay.rowsPacked} windows, ${(useP * 100).toFixed(1)}% of positions carry real tokens<br>padded: ${lay.rowsPadded} windows, ${(usePad * 100).toFixed(1)}% carry real tokens (the rest are padding that costs compute)`;
      q(ctl, '#cap').textContent = q(ctl, '#mask').checked ? 'Attention is masked at the dark end-of-document cell: each line under a row is a separate attention span, so a token never reads the unrelated document packed next to it.' : 'Without a mask, one line spans the whole window: tokens can attend to the previous, unrelated document in the same window. Many pipelines accept this; the mask removes it.';
      showVals(ctl, { L: (v) => v, rate: (v) => v });
    };
    const step = () => { if (n < lay.toks.length) n++; dirty = true; };
    wire(ctl, st, step, rebuild, draw);
    q(ctl, '#lens').onchange = rebuild; q(ctl, '#L').oninput = rebuild; q(ctl, '#mask').onchange = () => { dirty = true; };
    o.onresize = draw; rebuild(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < lay.toks.length) { n = Math.min(lay.toks.length, n + tick(dt, +q(ctl, '#rate').value)); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. A data-filtering pipeline on a toy corpus
     ===================================================================== */
  AI.labs['gen-tn-datapipe'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'From raw crawl to training corpus', 'illustrative: 48 made-up documents; each filter is applied to one document at a time',
      `<canvas style="height:400px"></canvas>`,
      `${sl('qthr', 'Quality-score threshold (drop below)', 0, 0.9, 0.05, 0.3)}
       <label><input type="checkbox" id="dedup" checked> remove exact duplicates (keep the first copy)</label>
       <label><input type="checkbox" id="optout" checked> exclude documents whose owner opted out or whose licence forbids training</label>
       ${sl('rate', 'Documents checked per second', 1, 20, 1, 5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const SRC = ['web', 'books', 'code', 'forum'];
    const r = AI.rng(7), docs = [];
    for (let i = 0; i < 48; i++) {
      const src = SRC[Math.floor(r() * 4)];
      const qlt = clamp((src === 'books' ? 0.75 : src === 'code' ? 0.6 : src === 'web' ? 0.4 : 0.3) + 0.22 * AI.randn(r), 0.02, 0.99);
      const dup = i > 10 && r() < 0.2 ? Math.floor(r() * i) : -1;
      docs.push({ i, src, tok: Math.round(2 + 8 * r()), q: qlt, dupOf: dup, restricted: r() < (src === 'books' ? 0.35 : 0.15) });
    }
    docs.forEach((d) => { if (d.dupOf >= 0) { d.src = docs[d.dupOf].src; d.tok = docs[d.dupOf].tok; d.q = docs[d.dupOf].q; d.restricted = docs[d.dupOf].restricted; } });
    const st = { playing: true };
    let n = 0, dirty = true, tick = AI.ticker(), done = 0;
    const eval1 = () => {
      const thr = +q(ctl, '#qthr').value, dd = q(ctl, '#dedup').checked, oo = q(ctl, '#optout').checked, seen = new Set(), res = [];
      docs.forEach((d) => {
        let why = '';
        if (d.q < thr) why = 'quality';
        else if (dd && d.dupOf >= 0 && !res[d.dupOf].why) why = 'duplicate';
        else if (oo && d.restricted) why = 'restricted';
        res.push({ why });
      });
      return res;
    };
    const reset = () => { n = 0; dirty = true; };
    const colOf = (c) => ({ web: c.blue, books: c.purple, code: c.green, forum: c.orange });
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const res = eval1(), cm = colOf(c), cols = 12, cw = Math.min(46, (W - 24) / cols), ch = 30;
      txt(g, 'square = one document (k tokens); cross = removed', 10, 14, c.muted, '11px sans-serif');
      docs.forEach((d, i) => {
        const x = 10 + (i % cols) * cw, y = 24 + Math.floor(i / cols) * (ch + 4), seen = i < Math.floor(n), cur = i === Math.floor(n);
        g.fillStyle = cm[d.src]; g.globalAlpha = seen && res[i].why ? 0.18 : 0.8; g.fillRect(x, y, cw - 3, ch); g.globalAlpha = 1;
        g.strokeStyle = cur ? c.fg : c.border; g.lineWidth = cur ? 2.5 : 1; g.strokeRect(x, y, cw - 3, ch); g.lineWidth = 1;
        txt(g, d.tok, x + (cw - 3) / 2, y + 13, seen && res[i].why ? c.muted : '#fff', 'bold 11px sans-serif', 'center');
        if (seen && res[i].why) {
          g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(x + 3, y + 3); g.lineTo(x + cw - 6, y + ch - 3); g.moveTo(x + cw - 6, y + 3); g.lineTo(x + 3, y + ch - 3); g.stroke(); g.lineWidth = 1;
          txt(g, res[i].why.slice(0, 4), x + (cw - 3) / 2, y + ch - 3, c.fg, '9px sans-serif', 'center');
        } else txt(g, 'q' + d.q.toFixed(1), x + (cw - 3) / 2, y + ch - 4, '#fff', '9px sans-serif', 'center');
      });
      // mixture bars
      const by = 24 + 4 * (ch + 4) + 22, bh = 18, bw = W - 130;
      txt(g, 'mixture of tokens by source', 10, by - 6, c.muted, '11px sans-serif');
      const tot = { before: {}, after: {} }; SRC.forEach((s) => { tot.before[s] = 0; tot.after[s] = 0; });
      let nb = 0, na = 0, rq = 0, rd = 0, ro = 0;
      docs.forEach((d, i) => { tot.before[d.src] += d.tok; nb += d.tok; if (!res[i].why) { tot.after[d.src] += d.tok; na += d.tok; } else if (res[i].why === 'quality') rq += d.tok; else if (res[i].why === 'duplicate') rd += d.tok; else ro += d.tok; });
      [['raw', tot.before, nb], ['kept', tot.after, na]].forEach(([lab, t, tt], k) => {
        let x = 60; const y = by + k * (bh + 6);
        txt(g, lab, 52, y + 13, c.fg, '11px sans-serif', 'right');
        SRC.forEach((s) => { const w = tt ? (t[s] / nb) * bw : 0; g.fillStyle = cm[s]; g.globalAlpha = 0.85; g.fillRect(x, y, w, bh); g.globalAlpha = 1; if (w > 30) txt(g, s + ' ' + Math.round(t[s] / (tt || 1) * 100) + '%', x + 4, y + 13, '#fff', '10px sans-serif'); x += w; });
        txt(g, tt + 'k', x + 6, y + 13, c.fg, '11px sans-serif');
      });
      let seenQ = 0;
      q(ctl, '#out').innerHTML = `raw: ${nb}k tokens in ${docs.length} documents<br>removed for quality ${rq}k, duplicates ${rd}k, restricted ${ro}k (in the order the filters run)<br><b>kept: ${na}k tokens (${(na / nb * 100).toFixed(1)}%)</b>`;
      q(ctl, '#cap').textContent = 'Every filter is a trade. A higher quality threshold removes noise but also the dialects and topics the scorer has never seen praised. Removing duplicates cuts memorisation. Excluding restricted documents respects owners’ wishes and licences but shrinks and skews the corpus. Whether and how data may be used for training is contested and differs between countries; the lab only shows the arithmetic cost of each choice.';
      showVals(ctl, { qthr: (v) => v.toFixed(2), rate: (v) => v });
    };
    const step = () => { n = Math.min(docs.length, Math.floor(n) + 1); dirty = true; };
    wire(ctl, st, step, reset, draw);
    ['qthr', 'dedup', 'optout'].forEach((id) => { q(ctl, '#' + id).oninput = q(ctl, '#' + id).onchange = reset; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < docs.length) { n = Math.min(docs.length, n + dt * +q(ctl, '#rate').value); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. Benchmark contamination
     ===================================================================== */
  AI.labs['gen-tn-contam'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Benchmark contamination', 'illustrative: 100 test questions; a model answers unseen ones with probability p0 and questions it saw in training with probability p1',
      `<canvas style="height:340px"></canvas>`,
      `${sl('f', 'Fraction of the benchmark that leaked into the training data', 0, 1, 0.05, 0.2)}
       ${sl('p0', 'p0: chance of a correct answer on an unseen question', 0.05, 0.95, 0.05, 0.4)}
       ${sl('p1', 'p1: chance of a correct answer on a seen question', 0.05, 1, 0.05, 0.95)}
       ${sl('rate', 'Questions graded per second', 2, 40, 1, 8)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const r = AI.rng(21), N = 100, order = [], u = [];
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    idx.forEach((qi, rank) => (order[qi] = rank));
    for (let i = 0; i < N; i++) u.push(r());
    const st = { playing: true };
    let n = 0, dirty = true, tick = AI.ticker();
    const reset = () => { n = 0; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const f = +q(ctl, '#f').value, p0 = +q(ctl, '#p0').value, p1 = +q(ctl, '#p1').value, nleak = Math.round(f * N);
      const cell = Math.min(30, (W - 24) / 10, (H - 30) / 10 - 2), ox = 12, oy = 14;
      let cs = [0, 0], tot = [0, 0];
      for (let i = 0; i < N; i++) {
        const leaked = order[i] < nleak, correct = u[i] < (leaked ? p1 : p0), x = ox + (i % 10) * cell, y = oy + Math.floor(i / 10) * cell, gr = i < n;
        g.fillStyle = gr ? (correct ? c.green : c.red) : c.border; g.globalAlpha = gr ? 0.85 : 0.35; g.fillRect(x, y, cell - 3, cell - 3); g.globalAlpha = 1;
        if (leaked) { g.strokeStyle = c.yellow; g.lineWidth = 2; g.strokeRect(x + 1, y + 1, cell - 5, cell - 5); g.lineWidth = 1; }
        if (gr) { tot[leaked ? 1 : 0]++; if (correct) cs[leaked ? 1 : 0]++; }
      }
      const px = ox + 10 * cell + 14;
      const accAll = n ? (cs[0] + cs[1]) / n : NaN, accClean = tot[0] ? cs[0] / tot[0] : NaN, accLeak = tot[1] ? cs[1] / tot[1] : NaN;
      const bx = Math.max(px, W - 150), bw = Math.min(40, (W - bx - 20) / 3), by = 20, bh = H - 70;
      [['all', accAll, c.blue], ['clean', accClean, c.green], ['leak', accLeak, c.yellow]].forEach(([lab, v, col], k) => {
        const x = bx + k * (bw + 8); g.strokeStyle = c.border; g.strokeRect(x, by, bw, bh);
        if (!isNaN(v)) { g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x, by + bh * (1 - v), bw, bh * v); g.globalAlpha = 1; txt(g, Math.round(v * 100) + '%', x + bw / 2, by + bh * (1 - v) - 4, c.fg, '11px sans-serif', 'center'); }
        txt(g, lab, x + bw / 2, by + bh + 14, c.fg, '11px sans-serif', 'center');
      });
      txt(g, 'yellow border = question was in the training data', ox, H - 6, c.muted, '11px sans-serif');
      const expected = (1 - nleak / N) * p0 + (nleak / N) * p1;
      q(ctl, '#out').innerHTML = `graded ${Math.floor(n)} of 100<br>expected score = (1 − f)·p0 + f·p1 = ${(1 - nleak / N).toFixed(2)}·${p0.toFixed(2)} + ${(nleak / N).toFixed(2)}·${p1.toFixed(2)} = <b>${expected.toFixed(3)}</b><br>honest score on clean questions only: ${p0.toFixed(2)} (inflation ${(expected - p0).toFixed(3)})<br>with 100 questions the sampling noise alone is about ±${(1.96 * Math.sqrt(expected * (1 - expected) / N)).toFixed(2)}`;
      q(ctl, '#cap').textContent = 'Green = correct, red = wrong. The headline number mixes two different skills: answering new questions and recalling seen ones. Splitting the benchmark into “clean” and “leaked” parts (if you know which is which) shows the gap; with few questions the bars wobble from chance alone.';
      showVals(ctl, { f: (v) => Math.round(v * 100) + '%', p0: (v) => v.toFixed(2), p1: (v) => v.toFixed(2), rate: (v) => v });
    };
    const step = () => { n = Math.min(N, Math.floor(n) + 1); dirty = true; };
    wire(ctl, st, step, reset, draw);
    ['f', 'p0', 'p1'].forEach((id) => (q(ctl, '#' + id).oninput = () => { dirty = true; }));
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < N) { n = Math.min(N, n + tick(dt, +q(ctl, '#rate').value)); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5. Fitting a power law and extrapolating
     ===================================================================== */
  AI.labs['gen-tn-fit'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fit a power law to small runs, then extrapolate', 'illustrative: synthetic losses, not a real training study; the dashed curve is the hidden truth',
      `<canvas style="height:420px"></canvas>`,
      `${sl('k', 'Largest number of runs to use', 4, 9, 1, 6)}
       ${sl('sig', 'Measurement noise (std of the loss)', 0, 0.06, 0.005, 0.015)}
       ${sl('kap', 'Bend in the true curve (0 = exact power law)', -0.15, 0.2, 0.01, 0)}
       ${sl('rate', 'New runs finishing per second', 0.1, 1.5, 0.05, 0.4)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="seed">New noise</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const TR = { E: 1.8, A: 190, al: 0.30 }, NPT = 9;
    const Ns = Array.from({ length: NPT }, (_, i) => Math.pow(10, 6 + 0.5 * i));
    const truth = (N, kap) => TR.E + TR.A * Math.pow(N, -TR.al) * (1 + kap * (Math.log10(N) - 6));
    const st = { playing: true };
    let seed = 5, zs, rr = 4, dirty = true, band = null, fit = null, disp = null, stats = null, tick = 0;
    const regen = () => { const r = AI.rng(seed); zs = Ns.map(() => AI.randn(r)); };
    const ys = () => { const s = +q(ctl, '#sig').value, kp = +q(ctl, '#kap').value; return Ns.map((N, i) => truth(N, kp) + s * zs[i]); };
    const refit = () => {
      const r = Math.min(Math.floor(rr), +q(ctl, '#k').value), y = ys();
      const bf = bootFit(Ns.slice(0, r), y.slice(0, r), 160, 99);
      fit = bf.p; const xg = []; for (let i = 0; i <= 40; i++) xg.push(Math.pow(10, 5.6 + (11 - 5.6) * i / 40));
      band = xg.map((x) => { const a = bf.fits.map((p) => predictPow(p, x)); return { x, lo: pct(a, 0.025), hi: pct(a, 0.975) }; });
      const al = bf.fits.map((p) => p.al), p11 = bf.fits.map((p) => predictPow(p, 1e10));
      stats = { r, sigma: bf.sigma, alMean: al.reduce((a, b) => a + b, 0) / al.length, alLo: pct(al, 0.025), alHi: pct(al, 0.975), pLo: pct(p11, 0.025), pHi: pct(p11, 0.975), y };
      if (!disp) disp = Object.assign({}, fit);
      dirty = true;
    };
    const reset = () => { rr = 4; refit(); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const v = AI.view(o, 5.6, 11, 1.5, 5.2, { l: 44, r: 12, t: 14, b: 40 });
      AI.axes(g, v);
      txt(g, 'log10 of parameter count N', v.pad.l + v.pw / 2, H - 6, c.muted, '11px sans-serif', 'center'); txt(g, 'loss', 4, 12, c.muted, '11px sans-serif');
      const kap = +q(ctl, '#kap').value, r = stats.r, y = stats.y;
      // band
      g.fillStyle = c.purple; g.globalAlpha = 0.2; g.beginPath();
      band.forEach((b, i) => { const X = v.X(Math.log10(b.x)), Y = v.Y(clamp(b.hi, 1.5, 5.2)); i ? g.lineTo(X, Y) : g.moveTo(X, Y); });
      for (let i = band.length - 1; i >= 0; i--) g.lineTo(v.X(Math.log10(band[i].x)), v.Y(clamp(band[i].lo, 1.5, 5.2)));
      g.closePath(); g.fill(); g.globalAlpha = 1;
      // truth
      g.strokeStyle = c.muted; g.setLineDash([5, 4]); g.beginPath();
      for (let i = 0; i <= 80; i++) { const lx = 5.6 + (11 - 5.6) * i / 80, Y = v.Y(truth(Math.pow(10, lx), kap)); i ? g.lineTo(v.X(lx), Y) : g.moveTo(v.X(lx), Y); }
      g.stroke(); g.setLineDash([]);
      // fit
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 80; i++) { const lx = 5.6 + (11 - 5.6) * i / 80, Y = v.Y(predictPow(disp, Math.pow(10, lx))); i ? g.lineTo(v.X(lx), Y) : g.moveTo(v.X(lx), Y); }
      g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(v.X(5.6), v.Y(disp.E)); g.lineTo(v.X(11), v.Y(disp.E)); g.setLineDash([2, 4]); g.stroke(); g.setLineDash([]);
      txt(g, 'fitted floor E', v.X(5.7), v.Y(disp.E) + 14, c.red, '11px sans-serif');
      // points
      for (let i = 0; i < NPT; i++) {
        const X = v.X(Math.log10(Ns[i])), Y = v.Y(y[i]);
        if (i < r) { g.fillStyle = c.blue; g.beginPath(); g.arc(X, Y, 5, 0, 7); g.fill(); }
        else { g.strokeStyle = i < +q(ctl, '#k').value ? c.blue : c.muted; g.lineWidth = 2; g.beginPath(); g.arc(X, Y, 5, 0, 7); g.stroke(); g.lineWidth = 1; }
      }
      const lg = [[c.blue, 'runs used in the fit'], [c.muted, 'later runs (not used)'], [c.purple, 'fitted curve + 95% band'], [c.muted, 'dashed = hidden truth']];
      lg.forEach((l, i) => txt(g, l[1], v.X(8.3), v.pad.t + 14 + i * 14, l[0], '11px sans-serif'));
      const tr = truth(1e10, kap);
      q(ctl, '#out').innerHTML = `runs in fit: ${r} (N from 1e6 to ${sci(Ns[r - 1])})<br>fitted exponent α̂ = ${fit.al.toFixed(3)}, 95% range [${stats.alLo.toFixed(3)}, ${stats.alHi.toFixed(3)}] (the truth used to make the data is ${TR.al})<br>fitted floor Ê = ${fit.E.toFixed(2)} (truth ${TR.E})<br>prediction at N = 1e10: ${predictPow(fit, 1e10).toFixed(3)}, 95% band [${stats.pLo.toFixed(3)}, ${stats.pHi.toFixed(3)}]<br>hidden truth at 1e10: ${tr.toFixed(3)} → ${tr >= stats.pLo && tr <= stats.pHi ? 'inside' : '<b>outside</b>'} the band`;
      q(ctl, '#cap').textContent = r < 5 ? 'Four runs, three unknowns: the fit can bend almost any way, and the band is wide. Let more runs finish.' : (Math.abs(kap) > 0.04 ? 'The true curve is not an exact power law (bend ≠ 0). The band only measures noise, so the truth can leave it: extrapolation assumes the pattern continues.' : 'With an exact power law and small noise the truth sits inside the band, but the band still widens quickly beyond the data.');
      showVals(ctl, { k: (x) => x, sig: (x) => x.toFixed(3), kap: (x) => x.toFixed(2), rate: (x) => x.toFixed(2) });
    };
    const step = () => { rr = Math.min(+q(ctl, '#k').value, Math.floor(rr) + 1); refit(); };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#seed').onclick = () => { seed++; regen(); refit(); };
    ['k', 'sig', 'kap'].forEach((id) => (q(ctl, '#' + id).oninput = () => { rr = Math.min(Math.max(rr, 4), +q(ctl, '#k').value); refit(); }));
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; regen(); refit(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && rr < +q(ctl, '#k').value) { const before = Math.floor(rr); rr += dt * +q(ctl, '#rate').value; if (Math.floor(rr) !== before) refit(); }
      const kk = 1 - Math.exp(-dt * 4); let moving = false;
      ['E', 'A', 'al'].forEach((key) => { const d = fit[key] - disp[key]; if (Math.abs(d) > 1e-5 * (Math.abs(fit[key]) + 1)) { disp[key] += d * kk; moving = true; } });
      if (moving || dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. Compute-optimal sizing for a fixed budget
     ===================================================================== */
  AI.labs['gen-tn-compute'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One compute budget, many ways to spend it', 'illustrative toy law L = E + A/N^a + B/D^b with made-up constants (E 1.5, A 300, a 0.30, B 500, b 0.25) and C = 6ND',
      `<canvas style="height:380px"></canvas>`,
      `${sl('lc', 'log10 of the compute budget C (FLOPs)', 19, 24, 0.1, 21)}
       ${sl('ln', 'log10 of the model size N (parameters)', 7, 12, 0.01, 9)}
       <label><input type="checkbox" id="auto" checked> sweep N automatically</label>
       ${sl('thr', 'Sustained throughput per device (TFLOP/s)', 20, 600, 10, 100)}
       ${sl('rate', 'Sweep speed (decades per second)', 0.05, 0.5, 0.05, 0.15)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let dir = 1, dirty = true;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const C = Math.pow(10, +q(ctl, '#lc').value), lnn = +q(ctl, '#ln').value, N = Math.pow(10, lnn), D = C / (6 * N);
      const Nopt = toyOptN(C), Dopt = C / (6 * Nopt), Lopt = toyLoss(Nopt, Dopt), L = toyLoss(N, D);
      const v = AI.view(o, 7, 12, Lopt - 0.05, Lopt + 1.2, { l: 46, r: 12, t: 14, b: 40 });
      AI.axes(g, v);
      txt(g, 'log10 N  (D = C / 6N shrinks as N grows)', v.pad.l + v.pw / 2, H - 6, c.muted, '11px sans-serif', 'center'); txt(g, 'loss', 4, 12, c.muted, '11px sans-serif');
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 120; i++) { const lx = 7 + 5 * i / 120, n = Math.pow(10, lx), Y = v.Y(toyLoss(n, C / (6 * n))); i ? g.lineTo(v.X(lx), Y) : g.moveTo(v.X(lx), Y); }
      g.stroke(); g.lineWidth = 1; g.restore();
      g.fillStyle = c.green; g.beginPath(); g.arc(v.X(Math.log10(Nopt)), v.Y(Lopt), 6, 0, 7); g.fill();
      txt(g, 'best split', v.X(Math.log10(Nopt)), v.Y(Lopt) + 20, c.green, 'bold 11px sans-serif', 'center');
      const my = clamp(L, v.ymin, v.ymax);
      g.fillStyle = c.orange; g.beginPath(); g.arc(v.X(lnn), v.Y(my), 6, 0, 7); g.fill();
      g.strokeStyle = c.orange; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(v.X(lnn), v.Y(my)); g.lineTo(v.X(lnn), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      txt(g, 'your choice', v.X(lnn), v.Y(my) - 10, c.orange, 'bold 11px sans-serif', 'center');
      const thr = +q(ctl, '#thr').value * 1e12, secs = C / thr;
      q(ctl, '#out').innerHTML = `C = ${sci(C)} FLOPs · N = ${sci(N)} · D = C/6N = ${sci(D)} tokens (${(D / N).toFixed(1)} tokens per parameter)<br>loss at your choice: ${L.toFixed(3)}<br>best split: N* = ${sci(Nopt)}, D* = ${sci(Dopt)} (${(Dopt / Nopt).toFixed(1)} tokens per parameter), loss ${Lopt.toFixed(3)}<br>mis-sizing costs ${(L - Lopt).toFixed(3)} nats<br>device time at ${(thr / 1e12).toFixed(0)} TFLOP/s: ${sci(secs)} device-seconds = ${sci(secs / 3600)} device-hours`;
      q(ctl, '#cap').textContent = 'Same budget, different spending. A very large model trained on few tokens, or a small model trained on very many, both lose to a balanced pair. Raise C and watch the best N move right: this toy law says both N* and D* grow with the budget (here N* ∝ C^0.45).';
      showVals(ctl, { lc: (x) => x.toFixed(1), ln: (x) => x.toFixed(2), thr: (x) => x, rate: (x) => x.toFixed(2) });
    };
    wire(ctl, st, () => { const e = q(ctl, '#ln'); e.value = clamp(+e.value + 0.1, 7, 12); dirty = true; }, () => { q(ctl, '#ln').value = 9; dirty = true; }, draw);
    ['lc', 'ln', 'thr'].forEach((id) => (q(ctl, '#' + id).oninput = () => { if (id === 'ln') q(ctl, '#auto').checked = false; dirty = true; }));
    q(ctl, '#auto').onchange = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing && q(ctl, '#auto').checked) {
        const e = q(ctl, '#ln'); let x = +e.value + dir * dt * +q(ctl, '#rate').value;
        if (x > 12) { x = 12; dir = -1; } if (x < 7) { x = 7; dir = 1; }
        e.value = x; dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. Warm-up and decay schedules on a noisy toy problem
     ===================================================================== */
  AI.labs['gen-tn-schedule'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Learning-rate schedule and AdamW on a noisy toy problem', 'illustrative: one noisy parameter minimising ½θ² with real AdamW arithmetic (β1 0.9, β2 0.95, ε 1e-8, weight decay 0); the schedule is plotted against the step',
      `<canvas style="height:420px"></canvas>`,
      `<label>Shape <select id="shape"><option value="cos">warm-up then cosine decay</option><option value="linear">warm-up then linear decay</option><option value="const">warm-up then constant</option></select></label>
       ${sl('W', 'Warm-up steps', 0, 300, 10, 100)}
       ${sl('peak', 'Peak learning rate η', 0.01, 0.3, 0.01, 0.1)}
       ${sl('minr', 'Final learning rate as a fraction of the peak', 0, 1, 0.05, 0.1)}
       ${sl('rate', 'Steps per second', 10, 300, 10, 60)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const T = 1000, st = { playing: true };
    let t = 0, dirty = true, tick = AI.ticker(), tr = null;
    const sim = () => {
      const W = +q(ctl, '#W').value, pk = +q(ctl, '#peak').value, mr = +q(ctl, '#minr').value, sh = q(ctl, '#shape').value, r = AI.rng(3);
      let th = 3, m = 0, vv = 0; const loss = [], lrs = [];
      for (let s = 0; s < T; s++) {
        const lr = lrAt(s, T, W, pk, mr, sh), g = th + 1.0 * AI.randn(r);
        m = 0.9 * m + 0.1 * g; vv = 0.95 * vv + 0.05 * g * g;
        const mh = m / (1 - Math.pow(0.9, s + 1)), vh = vv / (1 - Math.pow(0.95, s + 1));
        th -= lr * mh / (Math.sqrt(vh) + 1e-8);
        lrs.push(lr); loss.push(0.5 * th * th);
      }
      // smooth the loss for display (moving average over 20 steps)
      const sm = loss.map((_, i) => { let a = 0, k = 0; for (let j = Math.max(0, i - 19); j <= i; j++) { a += loss[j]; k++; } return a / k; });
      return { lrs, loss, sm };
    };
    const reset = () => { t = 0; tr = sim(); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pk = +q(ctl, '#peak').value, h1 = Math.round(H * 0.38);
      const v1 = AI.view({ w: W, h: h1 }, 0, T, 0, pk * 1.1, { l: 46, r: 12, t: 20, b: 20 });
      const v2 = AI.view({ w: W, h: H - h1 }, 0, T, -3, 1.2, { l: 46, r: 12, t: 20, b: 30 });
      txt(g, 'learning rate η(t)', 46, 12, c.muted, '11px sans-serif');
      g.save(); AI.axes(g, v1, { noGrid: true }); g.restore();
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); tr.lrs.forEach((l, i) => { i ? g.lineTo(v1.X(i), v1.Y(l)) : g.moveTo(v1.X(i), v1.Y(l)); }); g.stroke(); g.lineWidth = 1;
      const tt = Math.min(T - 1, Math.floor(t));
      g.strokeStyle = c.orange; g.beginPath(); g.moveTo(v1.X(t), 20); g.lineTo(v1.X(t), h1 - 20); g.stroke();
      g.fillStyle = c.orange; g.beginPath(); g.arc(v1.X(t), v1.Y(tr.lrs[tt]), 5, 0, 7); g.fill();
      g.save(); g.translate(0, h1);
      txt(g, 'loss ½θ² (log10 scale; gray = each step, blue = 20-step average)', 46, 12, c.muted, '11px sans-serif');
      const lg = (x) => Math.log10(Math.max(1e-6, x));
      const Yl = (x) => v2.Y(clamp(lg(x), -3, 1.2));
      g.strokeStyle = c.border; for (let e = -3; e <= 1; e++) { g.beginPath(); g.moveTo(46, Yl(Math.pow(10, e))); g.lineTo(W - 12, Yl(Math.pow(10, e))); g.stroke(); txt(g, '1e' + e, 42, Yl(Math.pow(10, e)) + 4, c.muted, '10px sans-serif', 'right'); }
      g.strokeStyle = c.muted; g.globalAlpha = 0.5; g.beginPath(); for (let i = 0; i <= tt; i++) { i ? g.lineTo(v2.X(i), Yl(tr.loss[i])) : g.moveTo(v2.X(i), Yl(tr.loss[i])); } g.stroke(); g.globalAlpha = 1;
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= tt; i++) { i ? g.lineTo(v2.X(i), Yl(tr.sm[i])) : g.moveTo(v2.X(i), Yl(tr.sm[i])); } g.stroke(); g.lineWidth = 1;
      txt(g, 'step', W / 2, H - h1 - 4, c.muted, '11px sans-serif', 'center');
      g.restore();
      const endAvg = tr.sm[T - 1], pkloss = Math.max(...tr.sm.slice(0, tt + 1));
      q(ctl, '#out').innerHTML = `step ${tt}: η = ${tr.lrs[tt].toExponential(2)}<br>average loss over the last 20 steps: ${tr.sm[tt].toExponential(2)}<br>average over the last 20 steps of the whole run (final value): ${endAvg.toExponential(2)}`;
      q(ctl, '#cap').textContent = 'The marker moves along the schedule. While η is large the parameter keeps being kicked by gradient noise, so the loss stalls at a floor. As η decays, the floor falls. In this tiny problem warm-up changes little; it matters at scale, where early updates on a freshly initialised network can be wildly large (see the lesson text).';
      showVals(ctl, { W: (x) => x, peak: (x) => x.toFixed(2), minr: (x) => x.toFixed(2), rate: (x) => x });
    };
    wire(ctl, st, () => { t = Math.min(T - 1, t + 10); }, reset, draw);
    ['shape', 'W', 'peak', 'minr'].forEach((id) => (q(ctl, '#' + id).oninput = q(ctl, '#' + id).onchange = () => { tr = sim(); dirty = true; }));
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && t < T - 1) { t = Math.min(T - 1, t + dt * +q(ctl, '#rate').value); dirty = true; }
      else if (st.playing && t >= T - 1) { st.holdT = (st.holdT || 0) + dt; if (st.holdT > 3) { st.holdT = 0; t = 0; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     8. Low-precision accumulation
     ===================================================================== */
  AI.labs['gen-tn-precision'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Adding small numbers in 16-bit and 32-bit formats', 'illustrative: rounding to the stated number of stored mantissa bits, ties to even; exponent range ignored except the float16 maximum 65504',
      `<canvas style="height:360px"></canvas>`,
      `${sl('x0', 'Start value', 1, 1000, 1, 1)}
       ${sl('dl', 'Increment added each step', 0.001, 2, 0.001, 1)}
       ${sl('rate', 'Additions per second', 5, 200, 5, 25)}
       <label>Try rounding this number <input type="text" id="num" value="1.0009765625" style="width:100%"></label>
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, NMAX = 1000;
    let n = 0, dirty = true, tick = AI.ticker(), acc = null, hist = null;
    const keys = ['fp32', 'fp16', 'bf16'];
    const reset = () => { n = 0; const x0 = +q(ctl, '#x0').value; acc = {}; hist = { exact: [x0] }; keys.forEach((k) => { acc[k] = roundMant(x0, FMT[k].bits, FMT[k].max); hist[k] = [acc[k]]; }); dirty = true; };
    const add = () => { const d = +q(ctl, '#dl').value; keys.forEach((k) => { acc[k] = roundMant(acc[k] + d, FMT[k].bits, FMT[k].max); hist[k].push(acc[k]); }); hist.exact.push(hist.exact[hist.exact.length - 1] + d); n++; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const x0 = +q(ctl, '#x0').value, d = +q(ctl, '#dl').value, top = x0 + d * NMAX;
      const v = AI.view(o, 0, NMAX, x0 - 0.02 * (top - x0), top + 0.04 * (top - x0), { l: 56, r: 12, t: 14, b: 30 });
      AI.axes(g, v);
      txt(g, 'step', v.pad.l + v.pw / 2, H - 6, c.muted, '11px sans-serif', 'center');
      const line = (arr, col, w, dash) => { g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath(); arr.forEach((y, i) => { const Y = v.Y(isFinite(y) ? y : v.ymax); i ? g.lineTo(v.X(i), Y) : g.moveTo(v.X(i), Y); }); g.stroke(); g.setLineDash([]); g.lineWidth = 1; };
      line(hist.exact, c.fg, 1, [4, 4]); line(hist.fp32, c.blue, 2); line(hist.fp16, c.green, 2); line(hist.bf16, c.orange, 2);
      [[c.fg, 'exact (dashed)'], [c.blue, 'float32 (23 mantissa bits)'], [c.green, 'float16 (10)'], [c.orange, 'bfloat16 (7)']].forEach((l, i) => txt(g, l[1], v.pad.l + 8, v.pad.t + 14 + i * 14, l[0], '11px sans-serif'));
      const ex = hist.exact[n];
      const fmt = (k) => `${k}: ${acc[k]}  (error ${(acc[k] - ex).toPrecision(3)})`;
      const nm = +q(ctl, '#num').value;
      let rows = '';
      if (isFinite(nm)) rows = '<br>' + keys.map((k) => `${k}(${nm}) = ${roundMant(nm, FMT[k].bits, FMT[k].max)}`).join(' · ');
      const sp = (k) => Math.pow(2, Math.floor(Math.log2(Math.max(1e-30, Math.abs(acc[k])))) - FMT[k].bits);
      q(ctl, '#out').innerHTML = `after ${n} additions of ${d}:<br>exact ${ex.toPrecision(7)}<br>${fmt('fp32')}<br>${fmt('fp16')}<br>${fmt('bf16')}<br>gap between neighbouring bf16 numbers near the sum: ${sp('bf16')}${rows}`;
      q(ctl, '#cap').textContent = 'Each format can only store numbers on a grid, and the grid gets coarser as the number grows. When the increment is smaller than half the gap, the sum stops moving. This is why the master copy of the weights, which receives many small updates, is kept in float32.';
      showVals(ctl, { x0: (x) => x, dl: (x) => x, rate: (x) => x });
    };
    wire(ctl, st, () => { if (n < NMAX) add(); }, reset, draw);
    ['x0', 'dl'].forEach((id) => (q(ctl, '#' + id).oninput = () => { reset(); }));
    q(ctl, '#num').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < NMAX) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k && n < NMAX; i++) add(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     9. Memory per device and sharding
     ===================================================================== */
  AI.labs['gen-tn-memory'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Where the bytes go: memory per GPU', 'illustrative: mixed-precision AdamW, N ≈ 12·layers·d², activations ≈ (34 or 2·layers+34)·b·s·d bytes; 1 GB = 10⁹ bytes; real systems add buffers and fragmentation',
      `<canvas style="height:330px"></canvas>`,
      `${sl('lay', 'Layers ℓ', 8, 96, 2, 24)}
       ${sl('d', 'Hidden size d', 512, 8192, 128, 2048)}
       ${sl('b', 'Micro-batch b (sequences per device)', 1, 16, 1, 2)}
       ${sl('s', 'Sequence length s', 512, 8192, 256, 2048)}
       <label>Sharding <select id="stage"><option value="0">none (every device holds everything)</option><option value="1">stage 1: shard optimiser state</option><option value="2">stage 2: + shard gradients</option><option value="3">stage 3: + shard weights</option></select></label>
       ${sl('nd', 'Devices sharing the state', 1, 64, 1, 8)}
       <label><input type="checkbox" id="rc"> recompute activations in the backward pass</label>
       ${sl('cap', 'Memory per device (GB)', 16, 192, 8, 80)}
       <div class="row"><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap2"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const segs = ['w', 'g', 'mas', 'm', 'v', 'act'], names = { w: 'weights (bf16)', g: 'gradients (bf16)', mas: 'fp32 master weights', m: 'Adam m (fp32)', v: 'Adam v (fp32)', act: 'activations' };
    let cur = { w: 0, g: 0, mas: 0, m: 0, v: 0, act: 0, base: 0 }, dirty = true;
    const target = () => {
      const l = +q(ctl, '#lay').value, d = +q(ctl, '#d').value, b = +q(ctl, '#b').value, s = +q(ctl, '#s').value, nd = +q(ctl, '#nd').value, stage = +q(ctl, '#stage').value, rc = q(ctl, '#rc').checked;
      const N = 12 * l * d * d, ms = memStates(N, nd, stage), m0 = memStates(N, 1, 0);
      const act = (rc ? 2 * l + 34 : 34 * l) * b * s * d;
      return { N, l, d, b, s, nd, stage, act, t: Object.assign({}, ms, { act }), base: Object.assign({}, m0, { act }) };
    };
    const reset = () => { cur = { w: 0, g: 0, mas: 0, m: 0, v: 0, act: 0 }; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const T = target(), capB = +q(ctl, '#cap').value * 1e9, colr = [c.blue, c.teal, c.purple, c.orange, c.yellow, c.pink];
      const sum = (o2) => segs.reduce((a, k) => a + o2[k], 0), tot = sum(cur), totT = sum(T.t), totB = sum(T.base);
      const scale = (W - 190) / Math.max(capB * 1.15, totB, totT);
      const bar = (label, vals, y, bh) => {
        let x = 130; txt(g, label, 124, y + bh / 2 + 4, c.fg, '11px sans-serif', 'right');
        segs.forEach((k, i) => { const w = vals[k] * scale; g.fillStyle = colr[i]; g.globalAlpha = 0.85; g.fillRect(x, y, w, bh); g.globalAlpha = 1; x += w; });
        return x;
      };
      const y1 = 40, y2 = 150, bh = 52;
      txt(g, 'no sharding, one device', 130, y1 - 8, c.muted, '11px sans-serif');
      const e1 = bar('stage 0', T.base, y1, bh); txt(g, (totB / 1e9).toFixed(1) + ' GB', e1 + 6, y1 + bh / 2 + 4, c.fg, 'bold 12px sans-serif');
      txt(g, `your setting: stage ${T.stage}, ${T.nd} devices`, 130, y2 - 8, c.muted, '11px sans-serif');
      const e2 = bar('per device', cur, y2, bh); txt(g, (tot / 1e9).toFixed(1) + ' GB', e2 + 6, y2 + bh / 2 + 4, c.fg, 'bold 12px sans-serif');
      const cx = 130 + capB * scale; g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, 18); g.lineTo(cx, y2 + bh + 22); g.stroke(); g.lineWidth = 1;
      txt(g, 'device memory ' + (capB / 1e9) + ' GB', cx - 4, 14, c.red, 'bold 11px sans-serif', 'right');
      segs.forEach((k, i) => { const lx = 12 + (i % 2) * (W / 2), ly = y2 + bh + 52 + Math.floor(i / 2) * 18; g.fillStyle = colr[i]; g.fillRect(lx, ly - 9, 10, 10); txt(g, names[k] + ' ' + (T.t[k] / 1e9).toFixed(2) + ' GB', lx + 14, ly, c.fg, '11px sans-serif'); });
      const fits = totT <= capB;
      q(ctl, '#out').innerHTML = `N ≈ 12·ℓ·d² = ${sci(T.N)} parameters<br>weights+gradients+optimiser state, no sharding: 16N = ${(16 * T.N / 1e9).toFixed(2)} GB<br>with stage ${T.stage} over ${T.nd} devices: ${((T.t.w + T.t.g + T.t.mas + T.t.m + T.t.v) / 1e9).toFixed(2)} GB<br>activations: ${(T.act / 1e9).toFixed(2)} GB<br><b>total per device: ${(totT / 1e9).toFixed(2)} GB → ${fits ? 'fits' : 'does NOT fit'} in ${(capB / 1e9)} GB</b>`;
      q(ctl, '#cap2').textContent = 'Bars slide to their new size when you change a setting. Sharding divides the persistent state by the number of devices, but only the parts you shard; activations are not shared this way, so recomputation or a smaller micro-batch is what shrinks them. Stage 3 must gather each layer’s weights on the fly, which costs communication.';
      showVals(ctl, { lay: (x) => x, d: (x) => x, b: (x) => x, s: (x) => x, nd: (x) => x, cap: (x) => x });
    };
    q(ctl, '#reset').onclick = reset;
    ['lay', 'd', 'b', 's', 'stage', 'nd', 'rc', 'cap'].forEach((id) => (q(ctl, '#' + id).oninput = q(ctl, '#' + id).onchange = () => { dirty = true; }));
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      const T = target(), k = 1 - Math.exp(-dt * 3.5); let moving = false;
      segs.forEach((s) => { const d = T.t[s] - cur[s]; if (Math.abs(d) > 1e5) { cur[s] += d * k; moving = true; } else cur[s] = T.t[s]; });
      if (moving || dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     10. Pipeline parallelism and the bubble
     ===================================================================== */
  AI.labs['gen-tn-bubble'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Pipeline parallelism: stages, micro-batches and the bubble', 'illustrative: forward pass takes 1 time unit per stage per micro-batch, backward takes 2; all forwards, then all backwards',
      `<canvas style="height:360px"></canvas>`,
      `${sl('p', 'Pipeline stages p (devices)', 2, 8, 1, 4)}
       ${sl('m', 'Micro-batches m per step', 1, 24, 1, 4)}
       ${sl('rate', 'Time units per second', 0.5, 6, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let t = 0, dirty = true;
    const sched = () => {
      const p = +q(ctl, '#p').value, m = +q(ctl, '#m').value, blocks = [], Ff = m + p - 1;
      for (let s = 0; s < p; s++) for (let i = 0; i < m; i++) {
        blocks.push({ s, i, t0: i + s, t1: i + s + 1, bwd: false });
        blocks.push({ s, i, t0: Ff + 2 * (i + (p - 1 - s)), t1: Ff + 2 * (i + (p - 1 - s)) + 2, bwd: true });
      }
      return { p, m, blocks, total: Ff + 2 * Ff };
    };
    const reset = () => { t = 0; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const S = sched(), L = 70, R = W - 14, T0 = 28, rh = Math.min(44, (H - 70) / S.p), X = (u) => L + u / S.total * (R - L);
      for (let s = 0; s < S.p; s++) { txt(g, 'device ' + (s + 1), L - 8, T0 + s * (rh + 4) + rh / 2 + 4, c.fg, '11px sans-serif', 'right'); g.strokeStyle = c.border; g.strokeRect(L, T0 + s * (rh + 4), R - L, rh); }
      S.blocks.forEach((b) => {
        const e = Math.max(0, Math.min(t, b.t1) - b.t0); if (e <= 0) return;
        const hue = (b.i * 37) % 360;
        g.fillStyle = `hsl(${hue},${b.bwd ? 60 : 50}%,${b.bwd ? 38 : 58}%)`; g.fillRect(X(b.t0), T0 + b.s * (rh + 4) + 1, X(b.t0 + e) - X(b.t0), rh - 2);
        if (e >= b.t1 - b.t0 && X(b.t1) - X(b.t0) > 14) txt(g, (b.bwd ? 'B' : 'F') + (b.i + 1), (X(b.t0) + X(b.t1)) / 2, T0 + b.s * (rh + 4) + rh / 2 + 4, '#fff', '10px sans-serif', 'center');
      });
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(X(t), T0 - 6); g.lineTo(X(t), T0 + S.p * (rh + 4)); g.stroke(); g.lineWidth = 1;
      txt(g, 'F = forward (light), B = backward (dark, 2x as long)', L, H - 22, c.muted, '11px sans-serif'); txt(g, 'the number is the micro-batch', L, H - 8, c.muted, '11px sans-serif');
      const bub = (S.p - 1) / (S.m + S.p - 1), busy = S.m * 3, idle = 1 - busy / S.total;
      q(ctl, '#out').innerHTML = `step takes ${S.total} time units; each device is busy ${busy} of them<br>idle fraction (measured from the chart) = ${(idle * 100).toFixed(1)}%<br>formula (p − 1)/(m + p − 1) = (${S.p} − 1)/(${S.m} + ${S.p} − 1) = ${(bub * 100).toFixed(1)}%<br>time: ${t.toFixed(1)} / ${S.total}`;
      q(ctl, '#cap').textContent = 'Device 2 cannot start micro-batch 1 until device 1 has finished it, so the first stages wait at the start and the last wait at the end: that is the bubble. More micro-batches keep devices busy longer relative to the fill and drain time.';
      showVals(ctl, { p: (x) => x, m: (x) => x, rate: (x) => x });
    };
    wire(ctl, st, () => { t = Math.min(sched().total, Math.floor(t) + 1); }, reset, draw);
    ['p', 'm'].forEach((id) => (q(ctl, '#' + id).oninput = reset));
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      const tot = sched().total;
      if (st.playing) { if (t < tot) { t = Math.min(tot, t + dt * +q(ctl, '#rate').value); dirty = true; } else { st.hold = (st.hold || 0) + dt; if (st.hold > 3) { st.hold = 0; t = 0; dirty = true; } } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     11. A simulated training run: loss, gradient norm, spikes, checkpoints
     ===================================================================== */
  AI.labs['gen-tn-run'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Operating a training run', 'illustrative: a made-up loss curve with noise; spikes and divergence follow simple rules, not a real model',
      `<canvas style="height:440px"></canvas>`,
      `${sl('lr', 'Learning-rate multiplier (higher = riskier)', 0.5, 4, 0.1, 1.5)}
       <label><input type="checkbox" id="clip" checked> gradient clipping at norm 1</label>
       <label><input type="checkbox" id="rnd" checked> random gradient spikes can occur</label>
       ${sl('ck', 'Checkpoint every (steps)', 100, 1000, 50, 250)}
       ${sl('rate', 'Steps per second', 10, 400, 10, 60)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step 10</button><button class="btn" id="reset">Reset</button></div>
       <div class="row"><button class="btn" id="spike">Inject a spike now</button><button class="btn" id="restart">Restart from last checkpoint</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const T = 3000, st = { playing: true };
    let t, dmg, hist, rng, seedN = 1, diverged, events, tick = AI.ticker(), dirty = true, lastCk, msg;
    const base = (s) => 1.9 + 32.5 / Math.sqrt(s + 50);
    const reset = () => { t = 0; dmg = 0; hist = []; rng = AI.rng(100 + seedN); diverged = false; events = []; lastCk = 0; msg = ''; dirty = true; };
    const spike = () => {
      const m = +q(ctl, '#lr').value, gn = (6 + 4 * rng()) * m, clipOn = q(ctl, '#clip').checked, eff = clipOn ? Math.min(gn, 1) : gn;
      dmg += 0.35 * eff; events.push(t); return gn;
    };
    const advance = () => {
      if (diverged || t >= T) return;
      const m = +q(ctl, '#lr').value; let gn = Math.max(0.15, (0.45 + 0.06 * AI.randn(rng)) * (1 + 0.1 * (m - 1)) * (1 - 0.3 * t / T));
      if (q(ctl, '#rnd').checked && rng() < 0.0002 * Math.pow(m, 3.5)) gn = spike();
      dmg *= Math.exp(-1 / 150);
      const train = base(t) + 0.05 * AI.randn(rng) * (1 + 0.5 * dmg) + dmg;
      const evalL = t % 100 === 0 ? base(t) + 0.1 + 0.02 * AI.randn(rng) + 0.8 * dmg : null;
      hist.push({ train, gn, evalL, ck: t % (+q(ctl, '#ck').value) === 0 });
      if (t % (+q(ctl, '#ck').value) === 0) lastCk = t;
      t++;
      if (dmg > 5) { diverged = true; msg = 'The loss blew up: damage from a large update that was not clipped. Restart from the last checkpoint.'; }
      dirty = true;
    };
    const restart = () => { const k = +q(ctl, '#ck').value, back = Math.floor((t > 0 ? t - 1 : 0) / k) * k; hist.length = back; t = back; dmg = 0; diverged = false; seedN++; rng = AI.rng(100 + seedN * 7 + t); msg = `Rewound to the checkpoint at step ${back}, with a different data order.`; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const h1 = Math.round(H * 0.64);
      const v1 = AI.view({ w: W, h: h1 }, 0, T, 1.5, 7, { l: 46, r: 12, t: 22, b: 8 });
      const v2 = AI.view({ w: W, h: H - h1 }, 0, T, 0, 4, { l: 46, r: 12, t: 16, b: 26 });
      txt(g, 'loss: train (gray), 50-step mean (blue), eval every 100 steps (green)', 46, 12, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; for (let y = 2; y <= 7; y++) { g.beginPath(); g.moveTo(46, v1.Y(y)); g.lineTo(W - 12, v1.Y(y)); g.stroke(); txt(g, y, 42, v1.Y(y) + 4, c.muted, '10px sans-serif', 'right'); }
      // checkpoints
      const k = +q(ctl, '#ck').value; g.strokeStyle = c.teal; g.setLineDash([2, 5]); for (let s = 0; s <= T; s += k) { g.beginPath(); g.moveTo(v1.X(s), 22); g.lineTo(v1.X(s), h1); g.stroke(); } g.setLineDash([]);
      const n = hist.length; if (!n) { return; }
      g.strokeStyle = c.muted; g.globalAlpha = 0.45; g.beginPath(); hist.forEach((h, i) => { const Y = v1.Y(clamp(h.train, 1.5, 7)); i ? g.lineTo(v1.X(i), Y) : g.moveTo(v1.X(i), Y); }); g.stroke(); g.globalAlpha = 1;
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); let a = 0; const win = []; hist.forEach((h, i) => { win.push(h.train); a += h.train; if (win.length > 50) a -= win.shift(); const Y = v1.Y(clamp(a / win.length, 1.5, 7)); i ? g.lineTo(v1.X(i), Y) : g.moveTo(v1.X(i), Y); }); g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.green; hist.forEach((h, i) => { if (h.evalL !== null) { g.beginPath(); g.arc(v1.X(i), v1.Y(clamp(h.evalL, 1.5, 7)), 3.5, 0, 7); g.fill(); } });
      // gradient norm
      txt(g, 'gradient norm before clipping (red = spike above 3)', 46, h1 + 10, c.muted, '11px sans-serif');
      g.save(); g.translate(0, h1);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(46, v2.Y(0)); g.lineTo(W - 12, v2.Y(0)); g.stroke();
      g.strokeStyle = c.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(46, v2.Y(1)); g.lineTo(W - 12, v2.Y(1)); g.stroke(); g.setLineDash([]); txt(g, 'clip 1', 42, v2.Y(1) + 4, c.muted, '10px sans-serif', 'right');
      hist.forEach((h, i) => { g.strokeStyle = h.gn > 3 ? c.red : c.blue; g.globalAlpha = h.gn > 3 ? 1 : 0.55; g.beginPath(); g.moveTo(v2.X(i), v2.Y(0)); g.lineTo(v2.X(i), v2.Y(Math.min(4, h.gn))); g.stroke(); }); g.globalAlpha = 1;
      txt(g, 'step', W / 2, H - h1 - 4, c.muted, '11px sans-serif', 'center'); g.restore();
      const last = hist[n - 1], ev = hist.filter((h) => h.evalL !== null).pop();
      q(ctl, '#out').innerHTML = `step ${t} of ${T}${diverged ? ' · <b style="color:var(--red)">DIVERGED</b>' : ''}<br>train loss ${last.train.toFixed(3)}, grad norm ${last.gn.toFixed(2)}, last evaluation loss ${ev ? ev.evalL.toFixed(3) : '–'}<br>last checkpoint: step ${lastCk}; spikes so far: ${events.length}`;
      q(ctl, '#cap').textContent = msg || 'The gradient norm is the early warning: a spike shows up there before the loss moves. With clipping, a spike moves the weights only a little. Without it (or at a high learning rate) one bad step can send the run into a region it never recovers from.';
      showVals(ctl, { lr: (x) => x.toFixed(1), ck: (x) => x, rate: (x) => x });
    };
    q(ctl, '#play').onclick = () => { st.playing = !st.playing; q(ctl, '#play').textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; q(ctl, '#play').textContent = '▶ Play'; for (let i = 0; i < 10; i++) advance(); draw(); };
    q(ctl, '#reset').onclick = () => { reset(); draw(); };
    q(ctl, '#spike').onclick = () => { if (!diverged && t < T) { spike(); hist.length && (hist[hist.length - 1].gn = 6 * (+q(ctl, '#lr').value)); msg = 'A spike was injected.'; dirty = true; } };
    q(ctl, '#restart').onclick = () => { restart(); draw(); };
    ['lr', 'clip', 'rnd', 'ck', 'rate'].forEach((id) => (q(ctl, '#' + id).oninput = q(ctl, '#' + id).onchange = () => { dirty = true; }));
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && !diverged && t < T) { const kk = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < kk; i++) advance(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
