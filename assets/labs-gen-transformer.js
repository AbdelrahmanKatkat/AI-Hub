/* Generative AI Module 2 labs: attention and the transformer.
   gen-tr-lookup, gen-tr-sdpa, gen-tr-scale, gen-tr-heads, gen-tr-posenc, gen-tr-order, gen-tr-rope,
   gen-tr-block, gen-tr-mask, gen-tr-gen, gen-tr-kv, gen-tr-params.
   All simulations are illustrative toy models with seeded random numbers (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const softmax = (a) => { const m = Math.max(...a); const e = a.map((x) => Math.exp(x - m)); const s = e.reduce((x, y) => x + y, 0); return e.map((x) => x / s); };
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const ent = (p) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0);
  const f2 = (v, d = 2) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d);
  const fmtN = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + 'G' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : '' + Math.round(n));
  const fmtB = (b) => (b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(b % 1024 ? 1 : 0) + ' KiB' : b < 1073741824 ? (b / 1048576).toFixed(b % 1048576 ? 2 : 0) + ' MiB' : (b / 1073741824).toFixed(2) + ' GiB');
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const arrow = (g, x0, y0, x1, y1, col, w) => {
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w || 2;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0), h = 8;
    if (Math.hypot(x1 - x0, y1 - y0) > 3) { g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4)); g.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4)); g.closePath(); g.fill(); }
    g.lineWidth = 1;
  };
  const rrect = (g, x, y, w, h, fill, stroke, lw) => { g.fillStyle = fill; g.fillRect(x, y, w, h); if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); g.lineWidth = 1; } };

  /* =====================================================================
     1. Attention as a soft lookup
     ===================================================================== */
  AI.labs['gen-tr-lookup'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A soft lookup: query, keys, values', 'illustrative: 3 keys and 3 values in 2-D; the query arrow sweeps slowly around a circle',
      `<canvas style="height:380px"></canvas>`,
      `${sl('len', 'Query length |q| (bigger = sharper lookup)', 0, 3, 0.1, 2)}
       ${sl('ang', 'Query direction (degrees)', 0, 360, 1, 0)}
       ${sl('spd', 'Sweep speed (degrees per second)', 0, 40, 1, 10)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const K = [[1, 0], [0, 1], [1, 1]], V = [[10, 0], [0, 10], [4, 4]], names = ['A', 'B', 'C'];
    const st = { playing: true };
    let ang = 0, dirty = true, trail = [], tacc = 0;
    const calc = () => {
      const L = +q(ctl, '#len').value, a = ang * Math.PI / 180, qv = [L * Math.cos(a), L * Math.sin(a)];
      const s = K.map((k) => dot(qv, k)), w = softmax(s), out = [0, 1].map((j) => w.reduce((t, wi, i) => t + wi * V[i][j], 0));
      return { qv, s, w, out };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const r = calc(), cols = [c.blue, c.green, c.orange];
      const pw = (W - 36) / 2, side = Math.min(pw, H - 110), top = 26;
      const bx = (px0, lo, hi) => ({ X: (x) => px0 + (x - lo) / (hi - lo) * side, Y: (y) => top + side - (y - lo) / (hi - lo) * side });
      const x0 = 12 + (pw - side) / 2, x1 = 24 + pw + (pw - side) / 2;
      const A = bx(x0, -3.2, 3.2), B = bx(x1, -1, 11);
      txt(g, 'key space', x0, 16, c.muted); txt(g, 'value space', x1, 16, c.muted);
      for (const [M, px] of [[A, x0], [B, x1]]) { g.strokeStyle = c.border; g.strokeRect(px + 0.5, top + 0.5, side - 1, side - 1); }
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(A.X(-3.2), A.Y(0)); g.lineTo(A.X(3.2), A.Y(0)); g.moveTo(A.X(0), A.Y(-3.2)); g.lineTo(A.X(0), A.Y(3.2)); g.stroke();
      K.forEach((k, i) => { arrow(g, A.X(0), A.Y(0), A.X(k[0]), A.Y(k[1]), cols[i], 2); txt(g, 'k' + names[i], A.X(k[0]) + 5, A.Y(k[1]) - 5, cols[i], 'bold 12px sans-serif'); });
      arrow(g, A.X(0), A.Y(0), A.X(r.qv[0]), A.Y(r.qv[1]), c.fg, 3); txt(g, 'q', A.X(r.qv[0]) + 6, A.Y(r.qv[1]) + 4, c.fg, 'bold 12px sans-serif');
      const hard = r.s.indexOf(Math.max(...r.s));
      V.forEach((v, i) => { g.fillStyle = cols[i]; g.beginPath(); g.arc(B.X(v[0]), B.Y(v[1]), 6, 0, 7); g.fill(); const nr = B.X(v[0]) > x1 + side - 62; txt(g, 'v' + names[i] + ' (' + v[0] + ',' + v[1] + ')', B.X(v[0]) + (nr ? -9 : 9), B.Y(v[1]) + (nr ? -9 : 4), cols[i], '11px sans-serif', nr ? 'right' : 'left'); });
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.arc(B.X(V[hard][0]), B.Y(V[hard][1]), 10, 0, 7); g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.purple; g.globalAlpha = 0.5; g.beginPath(); trail.forEach((p, i) => (i ? g.lineTo(B.X(p[0]), B.Y(p[1])) : g.moveTo(B.X(p[0]), B.Y(p[1])))); g.stroke(); g.globalAlpha = 1;
      g.fillStyle = c.purple; g.beginPath(); g.arc(B.X(r.out[0]), B.Y(r.out[1]), 7, 0, 7); g.fill(); txt(g, 'output', B.X(r.out[0]) + 10, B.Y(r.out[1]) - 8, c.purple, 'bold 12px sans-serif');
      txt(g, 'ring = hard lookup', x1, top + side + 14, c.muted, '11px sans-serif');
      const by = top + side + 24;
      r.w.forEach((w, i) => { const yy = by + i * 17; txt(g, 'weight ' + names[i], 12, yy + 11, cols[i], '11px sans-serif'); rrect(g, 78, yy, (W - 150) * w, 12, cols[i]); txt(g, f2(w, 3), 82 + (W - 150) * w, yy + 11, c.fg, '11px sans-serif'); });
      q(ctl, '#out').textContent = `q = (${f2(r.qv[0])}, ${f2(r.qv[1])})\nscores q·k = ${r.s.map((s) => f2(s)).join(', ')}\nweights = ${r.w.map((w) => f2(w, 3)).join(', ')}\noutput = (${f2(r.out[0])}, ${f2(r.out[1])})`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = +q(ctl, '#len').value < 0.2 ? 'A tiny query matches every key about equally, so the weights are almost uniform and the output is the plain average of the values.' : 'The query is compared with every key by a dot product, the scores become weights by softmax, and the output is the weighted average of the values. Lengthen the query and the lookup sharpens toward the ring; shorten it and it blurs.';
      showVals(ctl, { len: (v) => v.toFixed(1), ang: (v) => Math.round(v) + '°', spd: (v) => v });
    };
    const reset = () => { ang = 0; q(ctl, '#ang').value = 0; trail = []; dirty = true; };
    const push = () => { const r = calc(); trail.push(r.out); if (trail.length > 160) trail.shift(); };
    wire(ctl, st, () => { ang = (ang + 10) % 360; q(ctl, '#ang').value = ang; push(); dirty = true; }, reset, draw);
    q(ctl, '#ang').oninput = () => { ang = +q(ctl, '#ang').value; trail = []; dirty = true; };
    q(ctl, '#len').oninput = () => { trail = []; dirty = true; };
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { ang = (ang + +q(ctl, '#spd').value * dt) % 360; q(ctl, '#ang').value = ang; tacc += dt; if (tacc > 0.05) { tacc = 0; push(); } dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Scaled dot-product attention, step by step with editable Q, K, V
     ===================================================================== */
  AI.labs['gen-tr-sdpa'] = function (root) {
    const DEF = { Q: [[1, 2, 0, 1], [2, 0, 1, 3], [0, 1, 2, 1]], K: [[2, 1, 0, 1], [0, 3, 1, 0], [1, 0, 2, 2]], V: [[1, 0], [0, 2], [3, 1]] };
    const toks = ['the', 'cat', 'sat'];
    const mat = (id, label, cols) => `<div style="margin:4px 10px 4px 0"><b>${label}</b><table style="border-collapse:collapse">${DEF[id].map((row, i) => `<tr><td style="padding:0 6px 0 0;font-size:11px;color:var(--muted)">${toks[i]}</td>${row.map((v, j) => `<td><input type="number" step="0.5" data-m="${id}" data-i="${i}" data-j="${j}" value="${v}" style="width:3.6em;background:var(--bg2);color:var(--fg);border:1px solid var(--border)"></td>`).join('')}</tr>`).join('')}</table></div>`;
    const { view, ctl } = AI.labShell(root, 'Scaled dot-product attention, one stage at a time', 'illustrative: 3 tokens, d_k = 4, d_v = 2; edit any number in Q, K or V and the stages replay',
      `<canvas style="height:340px"></canvas><div style="display:flex;flex-wrap:wrap;margin-top:6px">${mat('Q', 'Q (3×4)')}${mat('K', 'K (3×4)')}${mat('V', 'V (3×2)')}</div>`,
      `<label><input type="checkbox" id="scale" checked> divide the scores by √d_k</label>
       ${sl('spd', 'Cells revealed per second', 0.5, 8, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let M = null, stage = 0, rv = 0, hold = 0, dirty = true;
    const readM = () => { const r = { Q: [[], [], []], K: [[], [], []], V: [[], [], []] }; view.querySelectorAll('input[data-m]').forEach((e) => { r[e.dataset.m][+e.dataset.i][+e.dataset.j] = +e.value || 0; }); return r; };
    const compute = () => {
      const m = readM(), dk = 4, sc = q(ctl, '#scale').checked;
      const S = m.Q.map((qq) => m.K.map((k) => dot(qq, k)));
      const Ss = S.map((r) => r.map((x) => (sc ? x / Math.sqrt(dk) : x)));
      const Wt = Ss.map(softmax), Wn = S.map(softmax), Ws = S.map((r) => softmax(r.map((x) => x / Math.sqrt(dk))));
      const mulV = (W) => W.map((w) => [0, 1].map((j) => w.reduce((t, wi, i) => t + wi * m.V[i][j], 0)));
      M = { m, S, Ss, Wt, Wn, Ws, out: mulV(Wt), sc };
    };
    const restart = () => { compute(); stage = 0; rv = 0; hold = 0; dirty = true; };
    const nCells = () => (stage === 3 ? 6 : 9);
    const draw = () => {
      if (!M) compute();
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const titles = ['① raw scores S = Q Kᵀ', '② scaled scores S / √d_k' + (M.sc ? '' : '  (skipped: scale is off)'), '③ weights = softmax of each row', '④ output = weights × V'];
      txt(g, titles[stage], 12, 18, c.fg, 'bold 13px sans-serif');
      const cs = Math.min(78, (W - 130) / 3), x0 = 70, y0 = 56, nc = stage === 3 ? 2 : 3;
      const data = [M.S, M.Ss, M.Wt, M.out][stage];
      const hdr = stage === 3 ? ['dim 1', 'dim 2'] : toks;
      hdr.forEach((t, j) => txt(g, (stage === 3 ? '' : 'key ') + t, x0 + j * cs + cs / 2, y0 - 8, c.muted, '11px sans-serif', 'center'));
      toks.forEach((t, i) => txt(g, 'query ' + t, x0 - 6, y0 + i * cs + cs / 2 + 4, c.muted, '11px sans-serif', 'right'));
      const mx = Math.max(1e-9, ...data.flat().map(Math.abs));
      let cur = -1;
      for (let i = 0; i < 3; i++) for (let j = 0; j < nc; j++) {
        const idx = i * nc + j, a = clamp(rv - idx, 0, 1);
        const v = data[i][j], t = stage === 2 ? v : Math.abs(v) / mx;
        const colr = stage === 2 ? c.green : stage === 3 ? c.purple : (v < 0 ? c.red : c.blue);
        rrect(g, x0 + j * cs + 2, y0 + i * cs + 2, cs - 4, cs - 4, a > 0 ? AI.mix(c.bg2, colr, 0.15 + 0.7 * t * a) : c.bg2, c.border);
        if (a > 0) txt(g, stage === 2 ? f2(v, 3) : f2(v, stage === 3 ? 3 : 2), x0 + j * cs + cs / 2, y0 + i * cs + cs / 2 + 5, c.fg, 'bold 13px sans-serif', 'center');
        if (a > 0 && a < 1 || (idx === Math.floor(rv) && rv < nCells())) cur = idx;
      }
      // formula for the cell being revealed
      let ln1 = '', ln2 = '';
      const ci = cur >= 0 ? cur : nCells() - 1, ri = Math.floor(ci / nc), cj = ci % nc;
      if (stage === 0) { const a = M.m.Q[ri], b = M.m.K[cj]; ln1 = `S[${toks[ri]}, ${toks[cj]}] = q·k = ${a.map((x, i) => x + '×' + b[i]).join(' + ')}`; ln2 = `= ${f2(M.S[ri][cj], 0)}`; }
      if (stage === 1) { ln1 = `${f2(M.S[ri][cj], 0)} / √4 = ${f2(M.S[ri][cj], 0)} / 2 = ${f2(M.Ss[ri][cj])}` + (M.sc ? '' : '  (no scaling: unchanged)'); }
      if (stage === 2) { const row = M.Ss[ri], mxr = Math.max(...row), den = row.reduce((s, x) => s + Math.exp(x - mxr), 0); ln1 = `row ${toks[ri]}: e^${f2(row[cj])} / (${row.map((x) => 'e^' + f2(x)).join(' + ')})`; ln2 = `= ${f2(Math.exp(row[cj] - mxr) / den, 4)}`; }
      if (stage === 3) { const w = M.Wt[ri]; ln1 = `out[${toks[ri]}, dim ${cj + 1}] = ${w.map((x, i) => f2(x, 3) + '×' + M.m.V[i][cj]).join(' + ')}`; ln2 = `= ${f2(M.out[ri][cj], 3)}`; }
      txt(g, ln1, 12, y0 + 3 * cs + 22, c.fg, '12px monospace'); txt(g, ln2, 12, y0 + 3 * cs + 40, c.fg, '12px monospace');
      const rowsum = M.Wt.map((r) => r.reduce((s, x) => s + x, 0));
      q(ctl, '#out').textContent = `Largest weight in each row\n  with scale   : ${M.Ws.map((r) => f2(Math.max(...r), 3)).join(', ')}\n  without scale: ${M.Wn.map((r) => f2(Math.max(...r), 3)).join(', ')}\nEntropy in bits (3 equal weights = 1.585)\n  with scale   : ${M.Ws.map((r) => f2(ent(r), 3)).join(', ')}\n  without scale: ${M.Wn.map((r) => f2(ent(r), 3)).join(', ')}`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = ['Each cell is the dot product of one query row with one key row.', M.sc ? 'Every score is divided by √d_k = 2 so that scores stay a moderate size.' : 'Scale switched off: the scores go straight into softmax, which makes the weights sharper.', 'Softmax runs along each row: every row of weights is positive and sums to 1 (' + rowsum.map((s) => f2(s, 3)).join(', ') + ').', 'Each output row is a weighted average of the rows of V, using that query\'s weights.'][stage];
      showVals(ctl, { spd: (v) => v });
    };
    const advance = () => { if (stage < 3) { stage++; rv = 0; hold = 0; } else { stage = 0; rv = 0; hold = 0; } dirty = true; };
    wire(ctl, st, () => { if (rv < nCells()) rv = nCells(); else advance(); }, restart, draw);
    view.querySelectorAll('input[data-m]').forEach((e) => { e.oninput = restart; });
    q(ctl, '#scale').onchange = restart;
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; restart(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        if (rv < nCells()) { rv = Math.min(nCells(), rv + dt * +q(ctl, '#spd').value); dirty = true; }
        else { hold += dt; if (hold > (stage === 3 ? 3 : 1.6)) advance(); }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Why the scale: score size grows with d_k
     ===================================================================== */
  AI.labs['gen-tr-scale'] = function (root) {
    const DS = [1, 4, 16, 64, 256], NK = 8;
    const { view, ctl } = AI.labShell(root, 'Why divide by √d_k', 'illustrative: random queries and keys with independent standard-normal entries, 8 keys per trial, seeded',
      `<canvas style="height:400px"></canvas>`,
      `${sl('di', 'Key dimension d_k (1, 4, 16, 64, 256)', 0, 4, 1, 3)}
       ${sl('rate', 'Trials per second', 0.5, 20, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let rs, n, sq, cnt, sU, sS, hU, hS, last, dirty = true, tick = AI.ticker();
    const d = () => DS[+q(ctl, '#di').value];
    const reset = () => { rs = AI.rng(7); n = 0; sq = 0; cnt = 0; sU = 0; sS = 0; hU = []; hS = []; last = null; dirty = true; };
    const add = () => {
      const D = d(), qv = Array.from({ length: D }, () => AI.randn(rs)), sc = [];
      for (let i = 0; i < NK; i++) { let s = 0; for (let j = 0; j < D; j++) s += qv[j] * AI.randn(rs); sc.push(s); }
      sq += sc.reduce((a, x) => a + x * x, 0); cnt += NK;
      const wu = softmax(sc), ws = softmax(sc.map((x) => x / Math.sqrt(D)));
      sU += Math.max(...wu); sS += Math.max(...ws); n++; hU.push(sU / n); hS.push(sS / n);
      if (hU.length > 400) { hU.shift(); hS.shift(); }
      last = { wu, ws, sc };
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const half = (W - 36) / 2, bt = 24, bb = 150;
      const panel = (x0, title, w, col) => {
        txt(g, title, x0, 14, c.muted);
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, bb); g.lineTo(x0 + half, bb); g.stroke();
        if (w) w.forEach((v, i) => { const bw = half / NK; rrect(g, x0 + i * bw + 3, bb - v * (bb - bt), bw - 6, v * (bb - bt), col); txt(g, f2(v, 2), x0 + i * bw + bw / 2, bb - v * (bb - bt) - 3, c.fg, '10px sans-serif', 'center'); });
        g.setLineDash([3, 3]); g.strokeStyle = c.muted; const yu = bb - (1 / NK) * (bb - bt); g.beginPath(); g.moveTo(x0, yu); g.lineTo(x0 + half, yu); g.stroke(); g.setLineDash([]);
      };
      panel(12, 'latest trial: no scale', last && last.wu, c.red);
      panel(24 + half, 'latest trial: scaled', last && last.ws, c.green);
      txt(g, 'dashed = 1/8, an even split', 12, bb + 14, c.muted, '10px sans-serif');
      const t0 = 200, t1 = H - 26, L = 40, R = W - 12, xm = Math.max(20, hU.length);
      txt(g, 'running mean of the largest weight', L, 190, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      txt(g, '1', L - 5, t0 + 8, c.muted, '10px sans-serif', 'right'); txt(g, '0', L - 5, t1, c.muted, '10px sans-serif', 'right');
      const line = (h, col) => { if (!h.length) return; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); h.forEach((v, i) => { const x = L + i / (xm - 1) * (R - L), y = t1 - v * (t1 - t0); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.lineWidth = 1; };
      line(hU, c.red); line(hS, c.green);
      txt(g, `trials: ${n}`, R, H - 8, c.muted, '11px sans-serif', 'right');
      txt(g, 'red: no scale', R - 4, t1 - 22, c.red, 'bold 11px sans-serif', 'right'); txt(g, 'green: scaled', R - 4, t1 - 8, c.green, 'bold 11px sans-serif', 'right');
      const D = d();
      q(ctl, '#out').textContent = n ? `d_k = ${D}\nstandard deviation of a raw score: measured ${f2(Math.sqrt(sq / cnt))}  (theory √d_k = ${f2(Math.sqrt(D))})\nafter dividing by √d_k: ${f2(Math.sqrt(sq / cnt) / Math.sqrt(D))}\nmean largest weight, no scale: ${f2(sU / n, 3)}   with scale: ${f2(sS / n, 3)}` : `d_k = ${D}. Press play to draw random queries and keys.`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = D <= 1 ? 'With d_k = 1 a score is a product of two numbers and scaling changes nothing (divide by 1).' : 'Raise d_k: the raw scores spread out like √d_k, so unscaled softmax piles almost all weight on one key (red climbs toward 1) while the scaled version stays at a level that does not depend on d_k (green). A saturated softmax has almost zero gradient, which is why training stalls without the scale.';
      showVals(ctl, { di: (v) => DS[v], rate: (v) => v });
    };
    wire(ctl, st, add, reset, draw);
    q(ctl, '#di').oninput = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < 3000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) add(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. Multi-head attention: heads look at different places
     ===================================================================== */
  AI.labs['gen-tr-heads'] = function (root) {
    const toks = ['the', 'cat', 'sat', 'on', 'the', 'mat'], ids = [0, 1, 2, 3, 0, 4], T = 6, D = 8;
    const { view, ctl } = AI.labShell(root, 'Several heads, several attention patterns', 'illustrative: random (not trained) projections of 8-dimensional embeddings; a real model learns its heads',
      `<canvas style="height:520px"></canvas>`,
      `${sl('h', 'Number of heads h (d_model = 8)', 0, 3, 1, 1)}
       ${sl('spd', 'Query-token switches per second', 0.1, 2, 0.1, 0.5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const HS = [1, 2, 4, 8], st = { playing: true };
    const re = AI.rng(21), E = Array.from({ length: 5 }, () => Array.from({ length: D }, () => AI.randn(re)));
    const X = ids.map((i) => E[i]);
    let maps = [], cur = 0, tacc = 0, dirty = true;
    const build = () => {
      const h = HS[+q(ctl, '#h').value], dh = D / h; maps = [];
      for (let j = 0; j < h; j++) {
        const rw = AI.rng(100 * h + j), WQ = Array.from({ length: D }, () => Array.from({ length: dh }, () => AI.randn(rw) / Math.sqrt(D))), WK = Array.from({ length: D }, () => Array.from({ length: dh }, () => AI.randn(rw) / Math.sqrt(D)));
        const proj = (W) => X.map((x) => Array.from({ length: dh }, (_, c) => x.reduce((s, v, r) => s + v * W[r][c], 0)));
        const Qm = proj(WQ), Km = proj(WK);
        maps.push(Qm.map((qq) => softmax(Km.map((k) => 1.6 * dot(qq, k) / Math.sqrt(dh)))));
      }
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const h = maps.length, cols = h <= 2 ? h : (W < 640 ? 2 : 4), rows = Math.ceil(h / cols);
      const pw = (W - 12) / cols, ph = (H - 8) / rows, cell = Math.min((pw - 60) / T, (ph - 44) / T, 38);
      maps.forEach((M, j) => {
        const px = 6 + (j % cols) * pw + (pw - cell * T - 30) / 2 + 30, py = 6 + Math.floor(j / cols) * ph + 24;
        txt(g, `head ${j + 1} (width ${D / h})`, px, py - 8, c.fg, 'bold 11px sans-serif');
        for (let i = 0; i < T; i++) {
          txt(g, toks[i], px - 4, py + i * cell + cell / 2 + 3, i === cur ? c.fg : c.muted, (i === cur ? 'bold ' : '') + '10px sans-serif', 'right');
          for (let k = 0; k < T; k++) rrect(g, px + k * cell, py + i * cell, cell - 1, cell - 1, AI.mix(c.bg2, c.blue, M[i][k] * 1.05));
        }
        for (let k = 0; k < T; k++) txt(g, toks[k][0], px + k * cell + cell / 2, py + T * cell + 11, c.muted, '10px sans-serif', 'center');
        g.strokeStyle = c.orange; g.lineWidth = 2; g.strokeRect(px - 1, py + cur * cell - 1, cell * T + 1, cell + 1); g.lineWidth = 1;
      });
      const lines = maps.map((M, j) => { const r = M[cur], b = r.indexOf(Math.max(...r)); return `head ${j + 1}: "${toks[cur]}" looks mostly at "${toks[b]}" (${f2(r[b], 2)})`; });
      const h2 = maps.length, dh = D / h2;
      q(ctl, '#out').textContent = lines.join('\n') + `\n\nParameters (no biases): W_Q, W_K, W_V, W_O are each 8×8 = 64, so the layer has ${4 * D * D} whatever h is. Each head owns ${D}×${dh} = ${D * dh} entries of each of W_Q, W_K, W_V.`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = `Row = the token asking (orange frame = the current one), column = the token it reads. Each row sums to 1. Compare the heads: each puts its weight on different columns (see the list of where each head looks). With more heads each head is narrower (d_head = 8/h), but the total parameter count stays ${4 * D * D}.`;
      showVals(ctl, { h: (v) => HS[v], spd: (v) => v });
    };
    wire(ctl, st, () => { cur = (cur + 1) % T; }, () => { cur = 0; tacc = 0; }, draw);
    q(ctl, '#h').oninput = () => { build(); draw(); };
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; build(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { tacc += dt * +q(ctl, '#spd').value; if (tacc >= 1) { tacc -= 1; cur = (cur + 1) % T; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5a. Sinusoidal position encodings
     ===================================================================== */
  AI.labs['gen-tr-posenc'] = function (root) {
    const NP = 32;
    const { view, ctl } = AI.labShell(root, 'Sinusoidal position encodings', 'illustrative: PE(p, 2i) = sin(p·ω_i), PE(p, 2i+1) = cos(p·ω_i) with ω_i = base^(−2i/d); a cursor scans positions',
      `<canvas style="height:400px"></canvas>`,
      `${sl('d', 'Dimension d (even)', 4, 64, 4, 16)}
       ${sl('base', 'log10 of the base', 1, 5, 0.5, 4)}
       ${sl('spd', 'Scan speed (positions per second)', 0.25, 4, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let p = 0, dirty = true;
    const pe = (pos, d, base) => { const r = []; for (let i = 0; i < d / 2; i++) { const w = Math.pow(base, -2 * i / d); r.push(Math.sin(pos * w), Math.cos(pos * w)); } return r; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const d = +q(ctl, '#d').value, base = Math.pow(10, +q(ctl, '#base').value), pi = Math.round(p);
      const hw = W - 50, cw = hw / d, ch = 5.5, x0 = 36, y0 = 24;
      txt(g, 'rows = positions 0..31, columns = dimensions (red < 0 < blue)', 12, 14, c.muted, '11px sans-serif');
      for (let r = 0; r < NP; r++) { const v = pe(r, d, base); for (let k = 0; k < d; k++) rrect(g, x0 + k * cw, y0 + r * ch, cw, ch, v[k] >= 0 ? AI.mix(c.bg2, c.blue, v[k]) : AI.mix(c.bg2, c.red, -v[k])); }
      g.strokeStyle = c.orange; g.lineWidth = 2; g.strokeRect(x0 - 1, y0 + pi * ch - 1, d * cw + 2, ch + 2); g.lineWidth = 1;
      txt(g, '0', x0 - 4, y0 + 6, c.muted, '10px sans-serif', 'right'); txt(g, '31', x0 - 4, y0 + NP * ch, c.muted, '10px sans-serif', 'right');
      // similarity plot
      const L = x0, R = W - 14, t0 = y0 + NP * ch + 36, t1 = H - 30;
      txt(g, 'PE(' + pi + ') · PE(q) / (d/2) against q', L, t0 - 8, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      const ym = (v) => t1 - (v + 1) / 2 * (t1 - t0), xq = (k) => L + k / (NP - 1) * (R - L);
      g.setLineDash([3, 3]); g.strokeStyle = c.muted; g.beginPath(); g.moveTo(L, ym(0)); g.lineTo(R, ym(0)); g.stroke(); g.setLineDash([]);
      const a = pe(pi, d, base); g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath();
      for (let k = 0; k < NP; k++) { const s = dot(a, pe(k, d, base)) / (d / 2); k ? g.lineTo(xq(k), ym(s)) : g.moveTo(xq(k), ym(s)); } g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.orange; g.beginPath(); g.arc(xq(pi), ym(1), 5, 0, 7); g.fill();
      txt(g, 'q = 0', L, t1 + 14, c.muted, '10px sans-serif'); txt(g, 'q = 31', R, t1 + 14, c.muted, '10px sans-serif', 'right');
      const v = pe(pi, d, base);
      q(ctl, '#out').textContent = `position ${pi}: first four numbers of PE = ${v.slice(0, 4).map((x) => f2(x, 4)).join(', ')}\nfastest pair wavelength 2π ≈ 6.3 positions; slowest wavelength 2π·base^((d−2)/d) ≈ ${f2(2 * Math.PI * Math.pow(base, (d - 2) / d), 0)} positions`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = 'Fast pairs (left columns) change every position, slow pairs (right) barely move: together they are like the digits of a clock. As the cursor moves, the similarity curve keeps its shape and slides with it: the dot product of two encodings depends only on the offset q − p.';
      showVals(ctl, { d: (x) => x, base: (x) => (x % 1 ? x.toFixed(1) : x) + '  (base ' + Math.round(Math.pow(10, x)) + ')', spd: (x) => x });
    };
    wire(ctl, st, () => { p = (p + 1) % NP; }, () => { p = 0; }, draw);
    ['d', 'base', 'spd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { p += dt * +q(ctl, '#spd').value; if (p >= NP - 0.5) p = 0; dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5b. Attention ignores order unless positions are added
     ===================================================================== */
  AI.labs['gen-tr-order'] = function (root) {
    const { view, ctl } = AI.labShell(root, '"man bites dog" or "dog bites man"?', 'illustrative: 2-D token vectors, identity projections, one head; positions are added with a strength α',
      `<canvas style="height:260px"></canvas>`,
      `${sl('a', 'Position strength α', 0, 1.5, 0.05, 0)}
       ${sl('spd', 'Order swaps per second', 0.1, 1, 0.05, 0.3)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const TK = { man: [1, 0], bites: [0, 1], dog: [1, 1] }, P = [[0, 1], [1, 0], [1, 1]];
    const orders = [['man', 'bites', 'dog'], ['dog', 'bites', 'man']];
    const st = { playing: true };
    let u = 0, target = 0, tacc = 0, dirty = true;
    const attn = (ord, al) => {
      const X = ord.map((t, s) => TK[t].map((v, i) => v + al * P[s][i])), qv = X[1];
      const w = softmax(X.map((k) => dot(qv, k) / Math.sqrt(2))), out = [0, 1].map((j) => w.reduce((t, wi, i) => t + wi * X[i][j], 0));
      return { w, out };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const al = +q(ctl, '#a').value, slotW = (W - 40) / 3, bw = Math.min(110, slotW - 20), y = 70, cols = { man: c.blue, bites: c.green, dog: c.orange };
      const ord = u < 0.5 ? orders[0] : orders[1], r = attn(ord, al);
      for (let s = 0; s < 3; s++) { g.strokeStyle = c.border; g.setLineDash([3, 3]); g.strokeRect(20 + s * slotW + (slotW - bw) / 2 + 0.5, y + 0.5, bw, 44); g.setLineDash([]); txt(g, 'position ' + (s + 1), 20 + s * slotW + slotW / 2, y - 8, c.muted, '11px sans-serif', 'center'); }
      const e = ease(u);
      const slotOf = (t) => { const a = orders[0].indexOf(t), b = orders[1].indexOf(t); return a + (b - a) * e; };
      for (const t of ['man', 'bites', 'dog']) { const s = slotOf(t), x = 20 + s * slotW + (slotW - bw) / 2; rrect(g, x, y, bw, 44, AI.mix(c.bg2, cols[t], 0.35), cols[t], 2); txt(g, t, x + bw / 2, y + 19, c.fg, 'bold 14px sans-serif', 'center'); txt(g, '(' + TK[t].join(', ') + ')', x + bw / 2, y + 36, c.muted, '11px sans-serif', 'center'); }
      ord.forEach((t, s) => { const x = 20 + s * slotW + slotW / 2; rrect(g, x - bw / 2, y + 70, bw * r.w[s], 12, cols[t]); txt(g, 'weight ' + f2(r.w[s], 3), x - bw / 2, y + 100, c.fg, '11px sans-serif'); });
      txt(g, '"bites" attends to its neighbours with these weights:', 20, y + 62, c.muted, '11px sans-serif');
      const A = attn(orders[0], al), B = attn(orders[1], al);
      q(ctl, '#out').textContent = `output for "bites":\n  man bites dog: (${f2(A.out[0], 4)}, ${f2(A.out[1], 4)})\n  dog bites man: (${f2(B.out[0], 4)}, ${f2(B.out[1], 4)})\n  difference: ${f2(Math.hypot(A.out[0] - B.out[0], A.out[1] - B.out[1]), 4)}`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = al === 0 ? 'No position signal: swapping the first and last word leaves the output for "bites" unchanged, so attention cannot tell the two sentences apart.' : 'With positions added the two orders give different outputs, so the model can distinguish who bites whom. Set α back to 0 to see the difference vanish.';
      showVals(ctl, { a: (v) => v.toFixed(2), spd: (v) => v });
    };
    wire(ctl, st, () => { target = 1 - target; }, () => { u = 0; target = 0; tacc = 0; }, draw);
    q(ctl, '#a').oninput = () => { dirty = true; }; q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { tacc += dt * +q(ctl, '#spd').value; if (tacc >= 1) { tacc -= 1; target = 1 - target; } }
      if (u !== target) { u += Math.sign(target - u) * Math.min(Math.abs(target - u), dt * 0.7); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5c. Rotary position embedding: rotation by position, dot depends on offset
     ===================================================================== */
  AI.labs['gen-tr-rope'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Rotary embedding: rotate by position, compare by offset', 'illustrative: one 2-D pair; the query sits at position m = n + Δ, the key at n; both slide along the text together',
      `<canvas style="height:400px"></canvas>`,
      `${sl('qx', 'q, first number', -3, 3, 0.5, 2)}${sl('qy', 'q, second number', -3, 3, 0.5, 1)}
       ${sl('kx', 'k, first number', -3, 3, 0.5, 1)}${sl('ky', 'k, second number', -3, 3, 0.5, 2)}
       ${sl('th', 'Rotation per position θ (radians)', 0.1, 1.5, 0.05, 0.5)}
       ${sl('dl', 'Offset Δ = m − n', -6, 6, 1, 2)}
       ${sl('spd', 'Slide speed (positions per second)', 0.1, 2, 0.1, 0.4)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, NMAX = 24;
    let n = 3, hist = [], tacc = 0, dirty = true;
    const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
    const vals = () => ({ qv: [+q(ctl, '#qx').value, +q(ctl, '#qy').value], kv: [+q(ctl, '#kx').value, +q(ctl, '#ky').value], th: +q(ctl, '#th').value, dl: +q(ctl, '#dl').value });
    const sample = () => { const { qv, kv, th, dl } = vals(), m = n + dl; hist.push([dot(rot(qv, m * th), rot(kv, n * th)), dot(rot(qv, m * th), kv)]); if (hist.length > 160) hist.shift(); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const { qv, kv, th, dl } = vals(), m = n + dl, qr = rot(qv, m * th), kr = rot(kv, n * th);
      const side = Math.min(W * 0.42, 190), cx = 14 + side / 2, cy = 28 + side / 2, sc = side / 2 / 4;
      txt(g, 'the 2-D plane', 14, 16, c.muted);
      g.strokeStyle = c.border; g.strokeRect(14.5, 28.5, side, side); g.beginPath(); g.moveTo(14, cy); g.lineTo(14 + side, cy); g.moveTo(cx, 28); g.lineTo(cx, 28 + side); g.stroke();
      g.setLineDash([3, 3]); arrow(g, cx, cy, cx + qv[0] * sc, cy - qv[1] * sc, c.blue, 1); arrow(g, cx, cy, cx + kv[0] * sc, cy - kv[1] * sc, c.orange, 1); g.setLineDash([]);
      arrow(g, cx, cy, cx + qr[0] * sc, cy - qr[1] * sc, c.blue, 3); arrow(g, cx, cy, cx + kr[0] * sc, cy - kr[1] * sc, c.orange, 3);
      txt(g, "q'", cx + qr[0] * sc + 4, cy - qr[1] * sc - 4, c.blue, 'bold 12px sans-serif'); txt(g, "k'", cx + kr[0] * sc + 4, cy - kr[1] * sc + 12, c.orange, 'bold 12px sans-serif');
      txt(g, 'dashed: before rotation', 14, 28 + side + 14, c.muted, '10px sans-serif');
      const L = side + 44, R = W - 12, a0 = 40, a1 = 28 + side, nq = Math.hypot(...qv) * Math.hypot(...kv);
      const lim = Math.max(1, nq), Y = (v) => a1 - (v / lim + 1) / 2 * (a1 - a0), X = (d) => L + (d + 8) / 16 * (R - L);
      txt(g, "q'·k' against offset Δ", L, 16, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.strokeRect(L + 0.5, a0 + 0.5, R - L, a1 - a0); g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(R, Y(0)); g.stroke();
      const phq = Math.atan2(qv[1], qv[0]), phk = Math.atan2(kv[1], kv[0]);
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 160; i++) { const d = -8 + i * 0.1, v = nq * Math.cos(phk - phq - d * th); i ? g.lineTo(X(d), Y(v)) : g.moveTo(X(d), Y(v)); } g.stroke(); g.lineWidth = 1;
      const cur = nq * Math.cos(phk - phq - dl * th); g.fillStyle = c.orange; g.beginPath(); g.arc(X(dl), Y(cur), 6, 0, 7); g.fill();
      txt(g, '−8', L, a1 + 12, c.muted, '10px sans-serif'); txt(g, '+8', R, a1 + 12, c.muted, '10px sans-serif', 'right');
      const b0 = 28 + side + 56, b1 = H - 40, L2 = 40;
      txt(g, 'dot product while the pair slides along the text', 14, b0 - 8, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.strokeRect(L2 + 0.5, b0 + 0.5, R - L2, b1 - b0); const Y2 = (v) => b1 - (v / lim + 1) / 2 * (b1 - b0); g.beginPath(); g.moveTo(L2, Y2(0)); g.lineTo(R, Y2(0)); g.stroke();
      const tr = (k, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => { const x = L2 + i / 159 * (R - L2); i ? g.lineTo(x, Y2(h[k])) : g.moveTo(x, Y2(h[k])); }); g.stroke(); g.lineWidth = 1; };
      tr(1, c.red); tr(0, c.green);
      txt(g, 'green: q rotated by mθ, k by nθ (flat)', 14, b1 + 16, c.green, '11px sans-serif'); txt(g, 'red: only q rotated (wanders)', 14, b1 + 32, c.red, '11px sans-serif');
      const direct = dot(qr, kr), formula = nq * Math.cos(phk - phq - dl * th);
      q(ctl, '#out').textContent = `n = ${f2(n, 2)}, m = ${f2(m, 2)}\nq′·k′ computed by rotating both = ${f2(direct, 5)}\n|q||k|·cos(φ_k − φ_q − Δθ)     = ${f2(formula, 5)}\nunrotated q·k = ${f2(dot(qv, kv), 3)}`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = 'Slide the pair along the text (play): both arrows turn, yet their dot product stays exactly on the same value because only the angle between them, Δθ plus the original gap, matters. Change Δ and the dot product moves along the green cosine curve.';
      showVals(ctl, { qx: (v) => v, qy: (v) => v, kx: (v) => v, ky: (v) => v, th: (v) => v.toFixed(2), dl: (v) => v, spd: (v) => v });
    };
    const reset = () => { n = 3; hist = []; sample(); dirty = true; };
    wire(ctl, st, () => { n += 1; sample(); }, reset, draw);
    ['qx', 'qy', 'kx', 'ky', 'th', 'dl', 'spd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { hist = []; sample(); dirty = true; }; });
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { n += dt * +q(ctl, '#spd').value; if (n > NMAX) { n = 3; hist = []; } tacc += dt; if (tacc > 0.06) { tacc = 0; sample(); } dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. The transformer block: the residual stream
     ===================================================================== */
  AI.labs['gen-tr-block'] = function (root) {
    const D = 8, T = 4, F = 32, NAMES = ['LayerNorm 1', 'Attention', 'add (+)', 'LayerNorm 2', 'MLP 8→32→8', 'add (+)'];
    const { view, ctl } = AI.labShell(root, 'A pre-norm transformer block on the residual stream', 'illustrative: random (not trained) weights; d_model = 8, 4 tokens, 1 attention head, MLP 8 → 32 → 8 with ReLU',
      `<canvas style="height:420px"></canvas>`,
      `${sl('nb', 'Number of blocks', 1, 6, 1, 2)}
       ${sl('tok', 'Token whose vector is drawn (1 to 4)', 1, 4, 1, 4)}
       ${sl('inp', 'Input scale', 0.2, 6, 0.2, 1)}
       <label><input type="checkbox" id="ln" checked> use layer normalisation before each sublayer</label>
       ${sl('spd', 'Stages per second', 0.1, 1.5, 0.1, 0.4)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    const mkW = (r, rows, cols, gain) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => AI.randn(r) * gain / Math.sqrt(rows)));
    const lnorm = (x) => { const mu = x.reduce((a, b) => a + b, 0) / x.length, va = x.reduce((a, b) => a + (b - mu) * (b - mu), 0) / x.length; return x.map((v) => (v - mu) / Math.sqrt(va + 1e-5)); };
    const vm = (x, W) => W[0].map((_, j) => x.reduce((s, v, i) => s + v * W[i][j], 0));
    const Ws = Array.from({ length: 6 }, (_, b) => { const r = AI.rng(40 + b); return { Q: mkW(r, D, D, 1), K: mkW(r, D, D, 1), V: mkW(r, D, D, 1), O: mkW(r, D, D, 0.6), W1: mkW(r, D, F, 1), W2: mkW(r, F, D, 0.5) }; });
    const rx = AI.rng(9), X0 = Array.from({ length: T }, () => Array.from({ length: D }, () => AI.randn(rx)));
    let steps = [], stage = 0, prog = 0, dirty = true;
    const compute = () => {
      const s = +q(ctl, '#inp').value, L = +q(ctl, '#nb').value, useLN = q(ctl, '#ln').checked; let X = X0.map((r) => r.map((v) => v * s)); steps = [];
      for (let b = 0; b < L; b++) {
        const w = Ws[b], pre = (x) => (useLN ? lnorm(x) : x);
        const H1 = X.map(pre), Qm = H1.map((h) => vm(h, w.Q)), Km = H1.map((h) => vm(h, w.K)), Vm = H1.map((h) => vm(h, w.V));
        const A = Qm.map((qq) => { const wt = softmax(Km.map((k) => dot(qq, k) / Math.sqrt(D))); return vm(Vm[0].map((_, j) => wt.reduce((t, wi, i) => t + wi * Vm[i][j], 0)), w.O); });
        steps.push({ before: X, seen: H1, delta: A }); const X1 = X.map((x, i) => x.map((v, j) => v + A[i][j]));
        steps.push({ before: X, seen: X, delta: A, after: X1 }); // add stage reuses delta
        const H2 = X1.map(pre), M = H2.map((h) => vm(h.length ? vm(h, w.W1).map((v) => Math.max(0, v)) : h, w.W2));
        steps.push({ before: X1, seen: H2, delta: M }); const X2 = X1.map((x, i) => x.map((v, j) => v + M[i][j]));
        steps.push({ before: X1, seen: X1, delta: M, after: X2 }); X = X2;
      }
    };
    // stage layout per block: 0 LN1 (seen), 1 attention (delta), 2 add, 3 LN2, 4 MLP (delta), 5 add
    const info = (S, t) => { // returns {x: stream vector, side: vector shown below, sideLabel}
      const b = Math.floor(S / 6), k = S % 6, L = steps.length / 4; if (b >= L) return null;
      const blk = steps.slice(b * 4, b * 4 + 4), tk = t;
      const pick = [() => ({ x: blk[0].before[tk], side: blk[0].seen[tk], label: 'input to attention = LN(x)' }),
        () => ({ x: blk[0].before[tk], side: blk[0].delta[tk], label: 'attention output (update)' }),
        () => ({ x: blk[1].after.length ? blk[1].before[tk] : null, to: blk[1].after[tk], side: blk[1].delta[tk], label: 'update being added' }),
        () => ({ x: blk[2].before[tk], side: blk[2].seen[tk], label: 'input to MLP = LN(x)' }),
        () => ({ x: blk[2].before[tk], side: blk[2].delta[tk], label: 'MLP output (update)' }),
        () => ({ x: blk[3].before[tk], to: blk[3].after[tk], side: blk[3].delta[tk], label: 'update being added' })][k]();
      return pick;
    };
    const nstages = () => 6 * +q(ctl, '#nb').value;
    const bars = (g, c, x0, y0, w, h, v, col, mx, lab, ghost) => {
      const bw = w / D; g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, y0 + h / 2); g.lineTo(x0 + w, y0 + h / 2); g.stroke();
      txt(g, lab, x0, y0 - 5, c.muted, '11px sans-serif');
      for (let i = 0; i < D; i++) { const hh = clamp(v[i] / mx, -1, 1) * h / 2; rrect(g, x0 + i * bw + 3, hh >= 0 ? y0 + h / 2 - hh : y0 + h / 2, bw - 6, Math.abs(hh), col);
        if (ghost) { const gh = clamp(ghost[i] / mx, -1, 1) * h / 2; g.strokeStyle = c.fg; g.strokeRect(x0 + i * bw + 3.5, gh >= 0 ? y0 + h / 2 - gh : y0 + h / 2, bw - 7, Math.abs(gh)); } }
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const S = Math.min(stage, nstages() - 1), k = S % 6, bidx = Math.floor(S / 6), tk = +q(ctl, '#tok').value - 1, inf = info(S, tk);
      if (!inf) return;
      // diagram
      const dx = 14, dw = Math.min(170, W * 0.36), yb = 18, bh = 30, gap = 15;
      txt(g, `block ${bidx + 1} of ${q(ctl, '#nb').value}`, dx + dw / 2 + 12, 12, c.muted, 'bold 11px sans-serif', 'center');
      const ys = NAMES.map((_, i) => yb + 10 + i * (bh + gap));
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); g.moveTo(dx + 8, yb); g.lineTo(dx + 8, ys[5] + bh + 10); g.stroke(); g.lineWidth = 1;
      txt(g, 'stream', dx + 8, ys[5] + bh + 24, c.purple, '10px sans-serif', 'center');
      NAMES.forEach((nm, i) => {
        const on = i === k, isAdd = i === 2 || i === 5, x = dx + 24, w = dw - 12;
        if (isAdd) { g.fillStyle = on ? c.orange : c.bg3; g.beginPath(); g.arc(dx + 8, ys[i] + bh / 2, 11, 0, 7); g.fill(); txt(g, '+', dx + 8, ys[i] + bh / 2 + 5, c.fg, 'bold 15px sans-serif', 'center'); }
        rrect(g, x, ys[i], w, bh, on ? AI.mix(c.bg2, c.orange, 0.45) : c.bg2, on ? c.orange : c.border, on ? 2 : 1);
        txt(g, nm, x + w / 2, ys[i] + bh / 2 + 4, c.fg, (on ? 'bold ' : '') + '12px sans-serif', 'center');
        if (!isAdd) { g.strokeStyle = c.muted; g.beginPath(); g.moveTo(dx + 8, ys[i] + bh / 2); g.lineTo(x, ys[i] + bh / 2); g.stroke(); }
        else { g.strokeStyle = c.muted; g.beginPath(); g.moveTo(dx + 19, ys[i] + bh / 2); g.lineTo(x, ys[i] + bh / 2); g.stroke(); }
      });
      // bars
      const rx0 = dx + dw + 30, rw = W - rx0 - 14, mx = 3;
      const t = ease(clamp(prog, 0, 1)); let xv = inf.x;
      if (inf.to) xv = inf.x.map((v, i) => v + (inf.to[i] - v) * t);
      bars(g, c, rx0, 40, rw, 120, xv, c.purple, mx, `residual stream, token ${tk + 1}`, inf.to && t < 1 ? inf.x : null);
      bars(g, c, rx0, 215, rw, 100, inf.side, k === 0 || k === 3 ? c.teal : c.orange, mx, inf.label);
      const nx = Math.hypot(...inf.x), ns = Math.hypot(...inf.side);
      txt(g, `|x| = ${f2(nx)}   |lower| = ${f2(ns)}`, rx0, 340, c.fg, '12px sans-serif'); if (k === 1 || k === 4 || k === 2 || k === 5) txt(g, `update/stream = ${f2(ns / nx)}`, rx0, 356, c.fg, '12px sans-serif');
      txt(g, 'bar scale ±3, larger values clipped', rx0, 374, c.muted, '10px sans-serif');
      const dd = 8, params = { att: 4 * dd * dd + 4 * dd, mlp: 8 * dd * dd + 5 * dd, ln: 4 * dd };
      q(ctl, '#out').textContent = `stage ${S + 1} of ${nstages()}: ${NAMES[k]}\nParameters of this toy block (d = 8, with biases and LayerNorm scale and shift): attention ${params.att}, MLP ${params.mlp}, norms ${params.ln}, total ${params.att + params.mlp + params.ln} = 12·8² + 13·8.\nThe MLP holds ${f2(params.mlp / (params.att + params.mlp + params.ln) * 100, 0)}% of them.`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = ['A pre-norm block first rescales each token vector to mean 0 and variance 1 (when the box is ticked); this normalised copy, not the stream itself, goes into attention.', 'Attention mixes information from all four tokens into one update vector for this token.', 'The update is added to the stream: the stream keeps everything it had and gains the update.', 'The same rescaling again before the MLP.', 'The MLP expands the 8 numbers to 32, applies ReLU and projects back to 8, one token at a time.', 'The second update is added. The result is the stream handed to the next block.'][k];
      showVals(ctl, { nb: (v) => v, tok: (v) => v, inp: (v) => v.toFixed(1), spd: (v) => v });
    };
    const reset = () => { stage = 0; prog = 0; compute(); dirty = true; };
    wire(ctl, st, () => { stage = (stage + 1) % nstages(); prog = 0; }, reset, draw);
    ['nb', 'inp', 'ln'].forEach((id) => { q(ctl, '#' + id).oninput = () => { stage = Math.min(stage, nstages() - 1); compute(); dirty = true; }; });
    q(ctl, '#ln').onchange = () => { compute(); dirty = true; };
    ['tok', 'spd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    o.onresize = draw; compute(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { prog += dt * +q(ctl, '#spd').value * 1.0; dirty = true; if (prog >= 1.6) { prog = 0; stage = (stage + 1) % nstages(); } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. The causal mask
     ===================================================================== */
  AI.labs['gen-tr-mask'] = function (root) {
    const T = 5, toks = ['t1', 't2', 't3', 't4', 't5'];
    const { view, ctl } = AI.labShell(root, 'The causal mask', 'illustrative: 5 positions, random scores, one number per token as its value; toggle the mask and perturb the last token',
      `<canvas style="height:400px"></canvas>`,
      `<label><input type="checkbox" id="mask" checked> apply the causal mask</label>
       <label><input type="checkbox" id="pert"> add +1 to the value of the last token</label>
       ${sl('spd', 'Row highlight per second', 0.1, 1.5, 0.1, 0.4)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">New scores</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let S, v, seed = 3, m = 1, row = 0, tacc = 0, dirty = true;
    const make = () => { const r = AI.rng(seed); S = Array.from({ length: T }, () => Array.from({ length: T }, () => AI.randn(r) * 1.3)); v = Array.from({ length: T }, () => Math.round(AI.randn(r) * 20) / 10); dirty = true; };
    const weights = (mm) => S.map((rw, i) => softmax(rw.map((x, j) => (j > i ? (mm >= 1 ? -Infinity : x - mm * 12) : x))));
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const Wt = weights(m), cs = Math.min(52, (W - 70) / (2 * T + 0.6)), x0 = 26, y0 = 52, x1 = x0 + T * cs + 20;
      txt(g, 'scores S', x0, 18, c.fg, 'bold 12px sans-serif'); txt(g, 'weights (softmax of rows)', x1, 18, c.fg, 'bold 12px sans-serif');
      for (let j = 0; j < T; j++) { txt(g, 'k' + (j + 1), x0 + j * cs + cs / 2, y0 - 6, c.muted, '10px sans-serif', 'center'); txt(g, 'k' + (j + 1), x1 + j * cs + cs / 2, y0 - 6, c.muted, '10px sans-serif', 'center'); }
      for (let i = 0; i < T; i++) {
        txt(g, 'r' + (i + 1), x0 - 4, y0 + i * cs + cs / 2 + 4, i === row ? c.orange : c.muted, (i === row ? 'bold ' : '') + '11px sans-serif', 'right');
        for (let j = 0; j < T; j++) {
          const fut = j > i, a = fut ? 1 - m : 1;
          rrect(g, x0 + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2, fut && m >= 1 ? c.bg3 : AI.mix(c.bg2, S[i][j] >= 0 ? c.blue : c.red, Math.min(0.8, Math.abs(S[i][j]) / 3) * a), c.border);
          txt(g, fut && m >= 1 ? '−∞' : f2(S[i][j], 1), x0 + j * cs + cs / 2, y0 + i * cs + cs / 2 + 4, fut && m >= 1 ? c.muted : c.fg, '10px sans-serif', 'center');
          rrect(g, x1 + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2, AI.mix(c.bg2, c.green, Wt[i][j] * 0.95), c.border);
          txt(g, f2(Wt[i][j], 2), x1 + j * cs + cs / 2, y0 + i * cs + cs / 2 + 4, c.fg, '10px sans-serif', 'center');
        }
      }
      g.strokeStyle = c.orange; g.lineWidth = 2; g.strokeRect(x0 - 1, y0 + row * cs, T * cs + 2, cs); g.strokeRect(x1 - 1, y0 + row * cs, T * cs + 2, cs); g.lineWidth = 1;
      const out = Wt.map((w) => w.reduce((s, x, j) => s + x * v[j], 0)), pert = q(ctl, '#pert').checked;
      const out2 = Wt.map((w) => w.reduce((s, x, j) => s + x * (v[j] + (j === T - 1 && pert ? 1 : 0)), 0));
      const by = y0 + T * cs + 30;
      txt(g, 'value of each token (one number)', 12, by, c.muted, '11px sans-serif'); v.forEach((x, j) => txt(g, f2(x, 1), x0 + j * cs + cs / 2, by + 16, c.fg, '11px sans-serif', 'center'));
      txt(g, pert ? 'change of each row output after +1' : 'output of each row = weights · values', 12, by + 40, c.muted, '11px sans-serif');
      for (let i = 0; i < T; i++) { const d = pert ? out2[i] - out[i] : out[i]; txt(g, f2(d, 2), x0 + i * cs + cs / 2, by + 58, pert && Math.abs(d) < 1e-9 ? c.green : c.fg, 'bold 11px sans-serif', 'center'); txt(g, 'r' + (i + 1), x0 + i * cs + cs / 2, by + 72, c.muted, '10px sans-serif', 'center'); }
      q(ctl, '#out').textContent = pert ? `Rows changed by the last token: ${out2.map((x, i) => (Math.abs(x - out[i]) > 1e-9 ? i + 1 : null)).filter(Boolean).join(', ') || 'none'}.` : `Row ${row + 1} may read keys 1..${m >= 1 ? row + 1 : T}; its weights sum to ${f2(Wt[row].reduce((a, b) => a + b, 0), 3)}.`;
      q(ctl, '#cap').textContent = m >= 1 ? 'With the mask, every score above the diagonal is replaced by −∞ before softmax, so its weight is exactly 0. Row i can only read keys 1 to i, and the top-left weight is always 1.' : 'Without the mask every row reads every key, including the future ones. In next-token training that would let the model copy the answer.';
      showVals(ctl, { spd: (x) => x });
    };
    const setMask = () => { dirty = true; };
    wire(ctl, st, () => { row = (row + 1) % T; }, () => { seed++; make(); row = 0; }, draw);
    q(ctl, '#mask').onchange = setMask; q(ctl, '#pert').onchange = setMask; q(ctl, '#spd').oninput = setMask;
    o.onresize = draw; make(); draw();
    AI.loop(root, (dt) => {
      const tg = q(ctl, '#mask').checked ? 1 : 0;
      if (m !== tg) { m += Math.sign(tg - m) * Math.min(Math.abs(tg - m), dt * 0.6); dirty = true; }
      if (st.playing) { tacc += dt * +q(ctl, '#spd').value; if (tacc >= 1) { tacc -= 1; row = (row + 1) % T; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     8. Training in parallel versus generating sequentially
     ===================================================================== */
  AI.labs['gen-tr-gen'] = function (root) {
    const seq = ['the', 'cat', 'sat', 'on', 'the', 'mat', '.', '<end>'], T = seq.length;
    const { view, ctl } = AI.labShell(root, 'Training: one pass for all positions. Generation: one pass per token', 'illustrative timeline of forward passes for an 8-token sequence',
      `<canvas style="height:360px"></canvas>`,
      `${sl('spd', 'Generation speed (tokens per second)', 0.2, 3, 0.1, 0.7)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let k = 0, frac = 0, hold = 0, dirty = true;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const bw = Math.min(80, (W - 40) / (T + 0.3)), x0 = 20;
      const pulse = clamp(1 - frac * 1.6, 0, 1);
      txt(g, 'TRAINING (teacher forcing)', 12, 18, c.fg, 'bold 12px sans-serif');
      for (let i = 0; i < T; i++) {
        rrect(g, x0 + i * bw + 2, 30, bw - 4, 26, AI.mix(c.bg2, c.blue, 0.35 + 0.4 * pulse), c.blue); txt(g, i ? seq[i - 1] : '<start>', x0 + i * bw + bw / 2, 47, c.fg, '12px sans-serif', 'center');
        rrect(g, x0 + i * bw + 2, 100, bw - 4, 26, AI.mix(c.bg2, c.green, 0.35 + 0.4 * pulse), c.green); txt(g, seq[i], x0 + i * bw + bw / 2, 117, c.fg, '12px sans-serif', 'center');
        arrow(g, x0 + i * bw + bw / 2, 58, x0 + i * bw + bw / 2, 98, c.muted, 1);
      }
      txt(g, 'inputs (true text, shifted right)', 12, 72, c.muted, '10px sans-serif'); txt(g, 'one masked pass, a loss at all 8 positions', 12, 88, c.muted, '10px sans-serif'); txt(g, 'targets', 12, 140, c.muted, '10px sans-serif');
      txt(g, 'GENERATION: one token per pass', 12, 180, c.fg, 'bold 12px sans-serif');
      for (let i = 0; i < T; i++) {
        const done = i < k, act = i === k;
        rrect(g, x0 + i * bw + 2, 196, bw - 4, 26, done ? AI.mix(c.bg2, c.orange, 0.5) : c.bg2, act ? c.orange : done ? c.orange : c.border, act ? 2 : 1);
        txt(g, done ? seq[i] : act ? '?' : '', x0 + i * bw + bw / 2, 213, c.fg, '12px sans-serif', 'center');
      }
      for (let i = 0; i < k; i++) { const xa = x0 + i * bw + bw / 2; g.strokeStyle = c.orange; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(xa, 224); g.lineTo(xa, 240); g.lineTo(x0 + k * bw + bw / 2, 240); g.lineTo(x0 + k * bw + bw / 2, 224); g.stroke(); g.globalAlpha = 1; }
      txt(g, k < T ? `pass ${k + 1}: read tokens 1..${k}, predict ${k + 1}` : 'finished', 12, 262, c.muted, '11px sans-serif');
      const maxP = T, bx = 12, bwid = W - 24;
      txt(g, 'forward passes needed', 12, 292, c.fg, 'bold 12px sans-serif');
      rrect(g, bx, 300, bwid * 1 / maxP, 14, c.blue); txt(g, 'training: 1', bx + bwid / maxP + 6, 312, c.blue, '11px sans-serif');
      rrect(g, bx, 320, bwid * Math.min(k + frac, T) / maxP, 14, c.orange); txt(g, `generation: ${Math.min(k + 1, T)} of ${T}`, bx + bwid * Math.min(k + frac, T) / maxP + 6, 332, c.orange, '11px sans-serif');
      q(ctl, '#out').textContent = `Training the whole 8-token sequence: 1 pass, 8 predictions, 8 loss terms.\nGenerating the same 8 tokens: 8 passes, one after another.\nThe passes cannot be run side by side because token k+1 is an input of pass k+1.`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = 'The mask is what makes the top picture legal: position i never sees targets after i, so predicting all positions at once is the same as predicting them one at a time. At generation time the model feeds on its own samples, one per pass.';
      showVals(ctl, { spd: (x) => x });
    };
    const reset = () => { k = 0; frac = 0; hold = 0; dirty = true; };
    wire(ctl, st, () => { k = (k + 1) % (T + 1); frac = 0; }, reset, draw);
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { if (k < T) { frac += dt * +q(ctl, '#spd').value; if (frac >= 1) { frac = 0; k++; } } else { hold += dt; if (hold > 2.5) reset(); } dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     9. KV cache growing during generation
     ===================================================================== */
  AI.labs['gen-tr-kv'] = function (root) {
    const NT = 48;
    const { view, ctl } = AI.labShell(root, 'The KV cache grows one token at a time', 'illustrative toy configuration: every cell stores the K and V vectors of one token in one layer',
      `<canvas style="height:420px"></canvas>`,
      `${sl('L', 'Layers L', 1, 8, 1, 4)}${sl('h', 'Key/value heads h', 1, 8, 1, 4)}${sl('dh', 'Head dimension d_head', 4, 64, 4, 16)}
       <label>Bytes per number b <select id="b"><option value="1">1 (8-bit)</option><option value="2" selected>2 (16-bit)</option><option value="4">4 (32-bit)</option></select></label>
       ${sl('spd', 'Tokens generated per second', 0.5, 12, 0.5, 3)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let t = 0, dirty = true, tick = AI.ticker(), hold = 0;
    const cfg = () => ({ L: +q(ctl, '#L').value, h: +q(ctl, '#h').value, dh: +q(ctl, '#dh').value, b: +q(ctl, '#b').value });
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const { L, h, dh, b } = cfg(), per = 2 * L * h * dh * b;
      txt(g, 'one column per token, one row per layer (blue K, orange V)', 12, 16, c.muted, '11px sans-serif');
      const gx = 60, gy = 26, cw = (W - gx - 14) / NT, ch = Math.min(22, 120 / L);
      for (let l = 0; l < L; l++) { txt(g, 'layer ' + (l + 1), gx - 6, gy + l * ch + ch / 2 + 3, c.muted, '10px sans-serif', 'right'); for (let i = 0; i < NT; i++) { const on = i < t; rrect(g, gx + i * cw + 0.5, gy + l * ch + 0.5, cw - 1, ch / 2 - 0.5, on ? c.blue : c.bg2); rrect(g, gx + i * cw + 0.5, gy + l * ch + ch / 2, cw - 1, ch / 2 - 0.5, on ? c.orange : c.bg2); } }
      const my = gy + L * ch + 24;
      txt(g, 'bytes/token = 2·L·h·d_head·b = ' + per, 12, my, c.fg, 'bold 12px sans-serif');
      txt(g, `cache now (${t} tokens) = ${fmtB(per * t)}`, 12, my + 18, c.fg, '12px sans-serif');
      // projections chart
      const L0 = 52, R = W - 14, t0 = my + 56, t1 = H - 22, xm = Math.max(8, t);
      txt(g, 'K, V vectors computed per layer (cumulative)', L0, my + 42, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L0, t0); g.lineTo(L0, t1); g.lineTo(R, t1); g.stroke();
      const ymax = xm * (xm + 1) / 2, X = (i) => L0 + i / xm * (R - L0), Y = (v) => t1 - v / ymax * (t1 - t0);
      g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= t; i++) { const v = i * (i + 1) / 2; i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(0), Y(0)); } g.stroke();
      g.strokeStyle = c.green; g.beginPath(); for (let i = 0; i <= t; i++) { i ? g.lineTo(X(i), Y(i)) : g.moveTo(X(0), Y(0)); } g.stroke(); g.lineWidth = 1;
      txt(g, 'no cache: t(t+1)/2 = ' + (t * (t + 1) / 2), R, t0 + 12, c.red, 'bold 11px sans-serif', 'right'); txt(g, 'with cache: t = ' + t, R, t0 + 28, c.green, 'bold 11px sans-serif', 'right');
      txt(g, `after ${t} tokens`, R, H - 6, c.muted, '10px sans-serif', 'right');
      q(ctl, '#out').textContent = `For this configuration, 1 token costs ${per} bytes of cache.\nAt 128 tokens: ${fmtB(per * 128)}.  At 2 048: ${fmtB(per * 2048)}.  At 32 768: ${fmtB(per * 32768)}.\nThe cache is linear in length (unlike the T×T score matrix).`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = 'Every new token adds one column: its K and V vectors for each layer are computed once and kept. Without a cache the model would recompute K and V of all earlier tokens at every step (red); with it, each step does one new token (green). The price is memory, which grows in proportion to length, layers, heads, head size and bytes per number.';
      showVals(ctl, { L: (v) => v, h: (v) => v, dh: (v) => v, spd: (v) => v });
    };
    const reset = () => { t = 0; hold = 0; dirty = true; };
    wire(ctl, st, () => { t = Math.min(NT, t + 1); }, reset, draw);
    ['L', 'h', 'dh', 'spd'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    q(ctl, '#b').onchange = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { if (t < NT) { const n = tick(dt, +q(ctl, '#spd').value); if (n) { t = Math.min(NT, t + n); dirty = true; } } else { hold += dt; if (hold > 2.5) reset(); } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     10. Parameter and cost calculator
     ===================================================================== */
  AI.labs['gen-tr-params'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Where the parameters and the compute go', 'a calculator for a decoder-only transformer: edit the configuration and compare the explicit tensor count with the formula',
      `<canvas style="height:400px"></canvas>`,
      `${sl('d', 'Model width d', 8, 256, 8, 64)}${sl('L', 'Layers L', 1, 12, 1, 4)}${sl('V', 'Vocabulary V', 1000, 50000, 1000, 1000)}
       ${sl('f', 'MLP expansion factor f', 2, 8, 1, 4)}${sl('h', 'Heads h', 1, 16, 1, 4)}${sl('T', 'Sequence length T', 16, 4096, 16, 128)}
       <label><input type="checkbox" id="bias" checked> biases and LayerNorm shift</label>
       <label><input type="checkbox" id="tie" checked> output layer shares the embedding table</label>
       <div class="row"><button class="btn" id="reset">Reset to the toy configuration</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    let disp = null, dirty = true;
    const calc = () => {
      const d = +q(ctl, '#d').value, L = +q(ctl, '#L').value, V = +q(ctl, '#V').value, f = +q(ctl, '#f').value, h = +q(ctl, '#h').value, T = +q(ctl, '#T').value, bias = q(ctl, '#bias').checked, tie = q(ctl, '#tie').checked;
      const att = 4 * d * d + (bias ? 4 * d : 0), mlp = 2 * f * d * d + (bias ? (f + 1) * d : 0), norm = (bias ? 4 : 2) * d, emb = V * d, out = tie ? 0 : V * d, fin = bias ? 2 * d : d;
      const total = emb + out + L * (att + mlp + norm) + fin;
      // explicit tensor list
      const tens = [[V, d]]; if (!tie) tens.push([d, V]);
      for (let l = 0; l < L; l++) { tens.push([d, d], [d, d], [d, d], [d, d], [d, f * d], [f * d, d], [d], [d]); if (bias) tens.push([d], [d], [d], [d], [f * d], [d], [d], [d]); }
      tens.push([d]); if (bias) tens.push([d]);
      const explicit = tens.reduce((s, t) => s + t.reduce((a, b) => a * b, 1), 0);
      const dense = (8 + 4 * f) * T * d * d, quad = 4 * T * T * d, scoreEntries = h * T * T;
      return { d, L, V, f, h, T, att, mlp, norm, emb, out, fin, total, explicit, dense, quad, scoreEntries, tens: tens.length };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const r = calc(); if (!disp) disp = { ...r };
      const parts = [['embedding table', 'emb', c.blue, 1], ['attention (all layers)', 'att', c.green, r.L], ['MLP (all layers)', 'mlp', c.orange, r.L], ['norms and output', 'norm', c.purple, 0]];
      const vals = [disp.emb + disp.out, disp.att * r.L, disp.mlp * r.L, disp.norm * r.L + disp.fin], tot = vals.reduce((a, b) => a + b, 0) || 1;
      txt(g, 'parameters: ' + r.total.toLocaleString('en-US') + ' in total', 12, 16, c.fg, 'bold 12px sans-serif');
      let x = 12; const bw = W - 24;
      vals.forEach((v, i) => { const w = v / tot * bw; rrect(g, x, 26, Math.max(0, w), 26, parts[i][2]); x += w; });
      let ly = 74; const real = [r.emb + r.out, r.att * r.L, r.mlp * r.L, r.norm * r.L + r.fin];
      vals.forEach((v, i) => { rrect(g, 12, ly - 10, 12, 12, parts[i][2]); txt(g, parts[i][0] + ': ' + Math.round(real[i]).toLocaleString('en-US') + ' (' + f2(v / tot * 100, 1) + '%)', 30, ly, c.fg, '12px sans-serif'); ly += 18; });
      const cy = 170; txt(g, 'FLOPs of one layer at T = ' + r.T, 12, cy, c.fg, 'bold 12px sans-serif');
      const mx = Math.max(disp.dense, disp.quad, 1);
      rrect(g, 12, cy + 12, Math.max(2, (W - 24) * disp.dense / mx), 20, c.orange); txt(g, 'dense (8+4f)·T·d²: ' + fmtN(r.dense), 18, cy + 27, c.fg, '11px sans-serif');
      rrect(g, 12, cy + 38, Math.max(2, (W - 24) * disp.quad / mx), 20, c.red); txt(g, 'attention 4·T²·d: ' + fmtN(r.quad), 18, cy + 53, c.fg, '11px sans-serif');
      txt(g, 'score entries h·T² = ' + r.scoreEntries.toLocaleString('en-US'), 12, cy + 80, c.muted, '11px sans-serif');
      const ok = r.explicit === r.total;
      txt(g, (ok ? '✓ explicit count = formula' : '✗ explicit count differs'), 12, cy + 104, ok ? c.green : c.red, 'bold 12px sans-serif');
      q(ctl, '#out').textContent = 'per block: attention ' + r.att.toLocaleString('en-US') + ' + MLP ' + r.mlp.toLocaleString('en-US') + ' + norms ' + r.norm + ' = ' + (r.att + r.mlp + r.norm).toLocaleString('en-US') + (q(ctl, '#bias').checked && r.f === 4 ? '  (= 12d² + 13d = ' + (12 * r.d * r.d + 13 * r.d) + ')' : '') + '\nembedding table V·d = ' + r.emb.toLocaleString('en-US') + '\ntotal = V·d' + (r.out ? ' + V·d' : '') + ' + L·(per block) + final norm = ' + r.total.toLocaleString('en-US') + '\nexplicit count of ' + r.tens + ' tensors listed one by one: ' + r.explicit.toLocaleString('en-US') + '\nattention FLOPs equal the dense FLOPs at T = (2 + f)·d = ' + (2 + r.f) * r.d;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      q(ctl, '#cap').textContent = 'Drag T up: the red bar overtakes the orange one near T = (2 + f)·d, but the parameter count does not change, because attention weights have no parameters. Drag d up: the parameter count grows with d² while the embedding table grows only with d.';
      showVals(ctl, { d: (v) => v, L: (v) => v, V: (v) => v, f: (v) => v, h: (v) => v, T: (v) => v });
      disp._t = r;
    };
    q(ctl, '#reset').onclick = () => { const set = { d: 64, L: 4, V: 1000, f: 4, h: 4, T: 128 }; for (const k in set) q(ctl, '#' + k).value = set[k]; q(ctl, '#bias').checked = true; q(ctl, '#tie').checked = true; dirty = true; };
    ctl.querySelectorAll('input').forEach((e) => { e.oninput = () => { dirty = true; }; e.onchange = () => { dirty = true; }; });
    o.onresize = () => { dirty = true; };
    draw();
    AI.loop(root, (dt) => {
      const r = calc(), k = 1 - Math.exp(-dt * 6);
      for (const key of ['att', 'mlp', 'norm', 'emb', 'out', 'fin', 'dense', 'quad']) { const dv = r[key] - disp[key]; if (Math.abs(dv) > 0.5) { disp[key] += dv * k; dirty = true; } else disp[key] = r[key]; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
