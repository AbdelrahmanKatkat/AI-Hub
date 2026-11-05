/* Generative AI Module 5 labs: efficient adaptation and inference.
   gen-ef-lora-count, gen-ef-lora-fit, gen-ef-quant, gen-ef-quant-group, gen-ef-roofline, gen-ef-batching,
   gen-ef-spec, gen-ef-spec-speed, gen-ef-distill, gen-ef-moe-router, gen-ef-moe-params.
   All simulations are illustrative, seeded (AI.rng), dt-based, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const ni = (id, label, val, min, max, step) => `<label>${label} <input type="number" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}" style="width:7em"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const fmtN = (n) => Math.round(n).toLocaleString('en-US');
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const softmax = (z) => { const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
  const heat = (g, v, mx, c) => { const a = clamp(Math.abs(v) / mx, 0, 1), rgb = AI.rgb(v >= 0 ? c.blue : c.red); return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.08 + 0.85 * a})`; };

  /* =====================================================================
     1. LoRA parameter counts: shapes drawn to scale
     ===================================================================== */
  AI.labs['gen-ef-lora-count'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'LoRA: how many numbers are trained?', 'illustrative: the three shapes are drawn to scale; the counts are exact formulas',
      `<canvas style="height:300px"></canvas>`,
      `${ni('din', 'd_in (columns of W)', 1024, 1, 100000, 1)}${ni('dout', 'd_out (rows of W)', 1024, 1, 100000, 1)}
       ${sl('r', 'Rank r', 1, 128, 1, 1)}${sl('rate', 'Sweep speed (ranks per second)', 0.5, 12, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let rShown = 1, rTarget = 1, dirty = true, acc = 0;
    const rMaxOf = () => Math.max(1, Math.min(128, Math.floor(Math.min(+q(ctl, '#din').value || 1, +q(ctl, '#dout').value || 1))));
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const din = Math.max(1, Math.round(+q(ctl, '#din').value || 1)), dout = Math.max(1, Math.round(+q(ctl, '#dout').value || 1));
      const r = Math.round(rShown);
      const sc = Math.min((W * 0.2) / din, (H - 90) / dout), y0 = 44;
      const wW = din * sc, wH = dout * sc;
      const rw = Math.max(2, r * sc), aH = Math.max(2, r * sc);
      txt(g, 'frozen W', 14, 22, c.muted, 'bold 12px sans-serif'); txt(g, `${dout} × ${din}`, 14, 36, c.muted, '11px sans-serif');
      g.fillStyle = c.bg3; g.strokeStyle = c.border; g.fillRect(14, y0, wW, wH); g.strokeRect(14, y0, wW, wH);
      txt(g, `${fmtN(din * dout)} numbers`, 14, y0 + wH + 16, c.fg, 'bold 12px sans-serif');
      const bx = 14 + wW + 56;
      txt(g, 'trainable B', bx - 2, 22, c.blue, 'bold 12px sans-serif'); txt(g, `${dout} × ${r}`, bx - 2, 36, c.muted, '11px sans-serif');
      g.fillStyle = c.blue; g.globalAlpha = 0.85; g.fillRect(bx, y0, rw, wH); g.globalAlpha = 1;
      const ax = bx + Math.max(rw, 2) + 96;
      txt(g, 'trainable A', ax, 22, c.orange, 'bold 12px sans-serif'); txt(g, `${r} × ${din}`, ax, 36, c.muted, '11px sans-serif');
      g.fillStyle = c.orange; g.globalAlpha = 0.85; g.fillRect(ax, y0, wW, aH); g.globalAlpha = 1;
      txt(g, 'versus', 14 + wW + 28, y0 + wH / 2 + 4, c.muted, '12px sans-serif', 'center');
      txt(g, '×', bx + Math.max(rw, 2) + 48, y0 + wH / 2 + 4, c.muted, 'bold 16px sans-serif', 'center');
      const full = din * dout, lora = r * (din + dout), ratio = lora / full, rstar = full / (din + dout);
      const by = H - 30, bwid = W - 28;
      g.fillStyle = c.bg3; g.fillRect(14, by, bwid, 12);
      g.fillStyle = lora > full ? c.red : c.green; g.fillRect(14, by, Math.min(1, ratio) * bwid, 12);
      txt(g, `LoRA parameters as a share of W: ${(ratio * 100).toFixed(ratio < 0.1 ? 4 : 2)}%`, 14, by - 6, c.muted, '11px sans-serif');
      q(ctl, '#r').max = rMaxOf();
      q(ctl, '#out').innerHTML = `full W: d_in·d_out = ${fmtN(din)}×${fmtN(dout)} = <b>${fmtN(full)}</b><br>LoRA: r(d_in+d_out) = ${r}×(${fmtN(din)}+${fmtN(dout)}) = <b>${fmtN(lora)}</b> (${(ratio * 100).toFixed(4)}% of W)<br>break-even rank r* = d_in·d_out/(d_in+d_out) = ${rstar.toFixed(2)}`;
      q(ctl, '#cap').textContent = lora > full ? `Rank ${r} is above the break-even ${rstar.toFixed(2)}: here B and A together hold MORE numbers than W, so LoRA saves nothing.` : (r === 1 ? 'Rank 1: B is a single column and A a single row. Press play to let the rank grow slowly and watch B and A thicken while W stays put.' : `B and A are thin slabs next to the big W. The saving is the ratio ${(ratio * 100).toFixed(2)}%.`);
      showVals(ctl, { r: (v) => v, rate: (v) => v });
    };
    const sync = () => { q(ctl, '#r').value = Math.round(rTarget); };
    const step = () => { rTarget = rTarget + 1 > rMaxOf() ? 1 : rTarget + 1; sync(); dirty = true; };
    const reset = () => { rTarget = rShown = 1; acc = 0; sync(); dirty = true; };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#r').oninput = () => { rTarget = +q(ctl, '#r').value; dirty = true; };
    ['#din', '#dout'].forEach((s) => (q(ctl, s).oninput = () => { rTarget = Math.min(rTarget, rMaxOf()); dirty = true; }));
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { rTarget += dt * +q(ctl, '#rate').value; if (rTarget > rMaxOf() + 0.999) rTarget = 1; sync(); dirty = true; }
      if (Math.abs(rShown - rTarget) > 0.01) { rShown += (rTarget - rShown) * Math.min(1, dt * 8); dirty = true; } else if (rShown !== rTarget) { rShown = rTarget; dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. LoRA fit: train B (6 x r) and A (r x 6) to match a target weight change
     ===================================================================== */
  AI.labs['gen-ef-lora-fit'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Training a low-rank update B·A (B starts at zero)', 'illustrative: a 6×6 target change that is rank 2 plus a little noise; plain gradient descent',
      `<canvas style="height:400px"></canvas>`,
      `${sl('r', 'Rank r', 1, 6, 1, 1)}${sl('lr', 'Learning rate', 0.01, 0.15, 0.01, 0.04)}${sl('rate', 'Gradient steps per second', 2, 60, 1, 12)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true }, n = 6;
    const rt = AI.rng(3), vec = () => { const v = []; for (let i = 0; i < n; i++) v.push(AI.randn(rt)); const s = Math.hypot(...v); return v.map((x) => x / s); };
    const u1 = vec(), v1 = vec(), u2 = vec(), v2 = vec();
    const T = [];
    for (let i = 0; i < n; i++) { T.push([]); for (let j = 0; j < n; j++) T[i].push(2 * u1[i] * v1[j] + 1 * u2[i] * v2[j] + 0.05 * AI.randn(rt)); }
    // singular values by power iteration with deflation on T^T T
    const G = []; for (let i = 0; i < n; i++) { G.push([]); for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += T[k][i] * T[k][j]; G[i].push(s); } }
    const sv = [];
    for (let k = 0; k < n; k++) {
      let v = Array.from({ length: n }, (_, i) => 1 + 0.1 * i);
      for (let it = 0; it < 400; it++) { const w = G.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)); const nm = Math.hypot(...w) || 1; v = w.map((x) => x / nm); }
      const Gv = G.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)), lam = v.reduce((s, x, i) => s + x * Gv[i], 0);
      sv.push(Math.sqrt(Math.max(0, lam)));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i][j] -= lam * v[i] * v[j];
    }
    const best = (r) => Math.sqrt(sv.slice(r).reduce((s, x) => s + x * x, 0));
    let B, A, hist, steps, dirty = true, tick = AI.ticker();
    const reset = () => {
      const r = +q(ctl, '#r').value, ra = AI.rng(7);
      A = []; B = [];
      for (let i = 0; i < r; i++) { A.push([]); for (let j = 0; j < n; j++) A[i].push(0.5 * AI.randn(ra)); }
      for (let i = 0; i < n; i++) B.push(new Array(r).fill(0));
      hist = []; steps = 0; hist.push(loss()); dirty = true;
    };
    const prod = () => { const r = A.length; return T.map((_, i) => T[0].map((__, j) => { let s = 0; for (let k = 0; k < r; k++) s += B[i][k] * A[k][j]; return s; })); };
    const loss = () => { const P = prod(); let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += (P[i][j] - T[i][j]) ** 2; return Math.sqrt(s); };
    const stepGD = () => {
      const r = A.length, lr = +q(ctl, '#lr').value, P = prod();
      const R = P.map((row, i) => row.map((x, j) => x - T[i][j]));
      const dB = B.map((row, i) => row.map((_, k) => { let s = 0; for (let j = 0; j < n; j++) s += R[i][j] * A[k][j]; return s; }));
      const dA = A.map((row, k) => row.map((_, j) => { let s = 0; for (let i = 0; i < n; i++) s += B[i][k] * R[i][j]; return s; }));
      for (let i = 0; i < n; i++) for (let k = 0; k < r; k++) B[i][k] -= lr * dB[i][k];
      for (let k = 0; k < r; k++) for (let j = 0; j < n; j++) A[k][j] -= lr * dA[k][j];
      steps++; hist.push(loss()); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const P = prod(), R = P.map((row, i) => row.map((x, j) => x - T[i][j]));
      const pw = (W - 40) / 3, cell = Math.min(34, (pw - 6) / n), top = 26;
      const panels = [['target change ΔW*', T], ['learned B·A', P], ['residual (B·A − ΔW*)', R]];
      panels.forEach(([name, M], p) => {
        const x0 = 10 + p * (pw + 10);
        txt(g, name, x0, 16, c.muted, 'bold 11px sans-serif');
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          g.fillStyle = heat(g, M[i][j], 2, c); g.fillRect(x0 + j * cell, top + i * cell, cell - 1, cell - 1);
          if (cell > 26) txt(g, M[i][j].toFixed(2), x0 + j * cell + cell / 2, top + i * cell + cell / 2 + 3, c.fg, '9px sans-serif', 'center');
        }
      });
      const cy0 = top + n * 34 + 36, cy1 = H - 24, L = 46, Rr = W - 14;
      txt(g, 'error ‖B·A − ΔW*‖ (Frobenius) against gradient step', L, cy0 - 10, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, cy0); g.lineTo(L, cy1); g.lineTo(Rr, cy1); g.stroke();
      const ymax = Math.max(2, hist[0]) * 1.05, xm = Math.max(60, Math.ceil(hist.length / 60) * 60);
      const X = (k) => L + (k / xm) * (Rr - L), Y = (v) => cy1 - (v / ymax) * (cy1 - cy0);
      txt(g, '0', L - 6, cy1 + 4, c.muted, '11px sans-serif', 'right'); txt(g, ymax.toFixed(1), L - 6, cy0 + 8, c.muted, '11px sans-serif', 'right');
      txt(g, `step ${steps}`, Rr, H - 6, c.muted, '11px sans-serif', 'right');
      const r = A.length, bst = best(r);
      g.strokeStyle = c.green; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(L, Y(bst)); g.lineTo(Rr, Y(bst)); g.stroke(); g.setLineDash([]);
      txt(g, `best possible at rank ${r}: ${bst.toFixed(3)}`, Rr - 4, Y(bst) - 5, c.green, 'bold 11px sans-serif', 'right');
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(0), Y(v)))); g.stroke(); g.lineWidth = 1;
      const e = hist[hist.length - 1];
      q(ctl, '#out').innerHTML = `trainable numbers: r(d_in+d_out) = ${r}×12 = <b>${r * 12}</b> versus ${n * n} in a full 6×6 update<br>error now <b>${e.toFixed(3)}</b>, best any rank-${r} matrix can do <b>${bst.toFixed(3)}</b> (from the singular values ${sv.map((x) => x.toFixed(2)).join(', ')})`;
      q(ctl, '#cap').textContent = r < 2 ? 'Rank 1 can only copy the strongest direction of the target; the residual panel keeps the second one. The error settles onto the green line, the best a rank-1 matrix can do.' : (r === 2 ? 'Rank 2 matches the structure of the target, so the error falls to the noise floor.' : 'More rank than the target needs: the extra directions can only chase noise, so the error barely improves.');
      showVals(ctl, { r: (v) => v, lr: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, stepGD, reset, draw);
    q(ctl, '#r').oninput = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && steps < 3000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) stepGD(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     quantisation helpers (uniform asymmetric, range always includes 0)
     ===================================================================== */
  const quantise = (w, bits) => {
    const mn = Math.min(0, ...w), mx = Math.max(0, ...w), L = Math.pow(2, bits) - 1;
    const s = mx > mn ? (mx - mn) / L : 1, z = Math.round(-mn / s);
    const rows = w.map((x) => { const qq = clamp(Math.round(x / s) + z, 0, L), d = s * (qq - z); return { x, q: qq, d, e: d - x }; });
    return { s, z, L, rows, mse: rows.reduce((a, r) => a + r.e * r.e, 0) / w.length, maxe: Math.max(...rows.map((r) => Math.abs(r.e))) };
  };

  /* =====================================================================
     3. Quantisation grid: weights snap to the nearest level
     ===================================================================== */
  AI.labs['gen-ef-quant'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Uniform quantisation: weights snap to a grid', 'illustrative: q = clamp(round(x/s) + z), x̂ = s(q − z); one weight snaps at a time',
      `<canvas style="height:260px"></canvas>`,
      `<label>Weights (comma separated, at most 12)<input type="text" id="w" value="-0.62, -0.31, -0.05, 0, 0.12, 0.38, 0.47, 0.90" style="width:100%"></label>
       ${sl('bits', 'Bits per weight', 2, 8, 1, 4)}${sl('rate', 'Seconds per weight (slower is easier to follow)', 0.3, 3, 0.1, 1.2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em;overflow-x:auto"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let t = 0, dirty = true, qd, w, lastKey = '';
    const parse = () => { w = q(ctl, '#w').value.split(',').map((s) => parseFloat(s)).filter((x) => isFinite(x)).slice(0, 12); if (w.length < 2) w = [-0.5, 0.5]; qd = quantise(w, +q(ctl, '#bits').value); };
    const dur = () => +q(ctl, '#rate').value;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const mn = Math.min(0, ...w) - 0.05, mx = Math.max(0, ...w) + 0.05, L = 20, R = W - 20, X = (v) => L + (v - mn) / (mx - mn) * (R - L), ly = H * 0.62;
      txt(g, `${qd.L + 1} levels (${q(ctl, '#bits').value} bits), step size s = ${qd.s.toFixed(5)}, zero point z = ${qd.z}`, L, 16, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, ly); g.lineTo(R, ly); g.stroke();
      for (let k = 0; k <= qd.L; k++) { const x = X(qd.s * (k - qd.z)); if (x < L - 1 || x > R + 1) continue; g.strokeStyle = k === qd.z ? c.green : c.border; g.lineWidth = k === qd.z ? 2 : 1; g.beginPath(); g.moveTo(x, ly - (k === qd.z ? 24 : 14)); g.lineTo(x, ly + (k === qd.z ? 24 : 14)); g.stroke(); }
      g.lineWidth = 1;
      txt(g, 'level z (exactly 0.0)', X(0), ly + 40, c.green, '11px sans-serif', 'center');
      const prog = (i) => clamp(t / dur() - i, 0, 1);
      w.forEach((x, i) => {
        const p = ease(prog(i)), xq = qd.rows[i].d, y = ly - 54 - (i % 2) * 22 + p * 54 + (i % 2) * 22 * p;
        g.strokeStyle = c.muted; g.setLineDash([2, 3]); g.beginPath(); g.arc(X(x), ly - 54 - (i % 2) * 22, 5, 0, 7); g.stroke(); g.setLineDash([]);
        const xx = X(x + (xq - x) * p);
        g.fillStyle = prog(i) >= 1 ? c.orange : c.blue; g.beginPath(); g.arc(xx, y, 5, 0, 7); g.fill();
        if (prog(i) > 0) { g.strokeStyle = c.red; g.beginPath(); g.moveTo(X(x), ly + 6); g.lineTo(X(xq), ly + 6); g.stroke(); }
        txt(g, i + 1, X(x), ly - 62 - (i % 2) * 22, c.muted, '10px sans-serif', 'center');
      });
      const done = Math.min(w.length, Math.floor(t / dur()));
      const key = done + '|' + q(ctl, '#w').value + '|' + q(ctl, '#bits').value;
      if (key !== lastKey) {
        lastKey = key;
        let h = `<table class="tbl" style="font-size:12px"><thead><tr><th>i</th><th>x</th><th>q</th><th>x̂ = s(q−z)</th><th>error</th></tr></thead><tbody>`;
        for (let i = 0; i < done; i++) { const r = qd.rows[i]; h += `<tr><td>${i + 1}</td><td>${r.x.toFixed(2)}</td><td>${r.q}</td><td>${r.d.toFixed(5)}</td><td>${r.e >= 0 ? '+' : ''}${r.e.toFixed(5)}</td></tr>`; }
        h += '</tbody></table>';
        if (done === w.length) h += `mean squared error <b>${qd.mse.toExponential(4)}</b>, largest |error| <b>${qd.maxe.toFixed(5)}</b> (never more than s/2 = ${(qd.s / 2).toFixed(5)} inside the range)`;
        q(ctl, '#out').innerHTML = h;
      }
      q(ctl, '#cap').textContent = done < w.length ? `Weight ${done + 1}: the dot slides to the nearest grid line; the red segment is its rounding error.` : 'Every weight now sits on a grid line (orange). Lower the bits and press Reset: the grid thins out and the errors grow.';
      showVals(ctl, { bits: (v) => v, rate: (v) => v + ' s' });
    };
    const total = () => w.length * dur();
    const step = () => { t = Math.min(total(), (Math.floor(t / dur() + 1e-9) + 1) * dur()); dirty = true; };
    const reset = () => { t = 0; lastKey = ''; dirty = true; };
    wire(ctl, st, step, reset, draw);
    const re = () => { parse(); reset(); draw(); };
    q(ctl, '#w').oninput = re; q(ctl, '#bits').oninput = re; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; parse(); draw();
    AI.loop(root, (dt) => { if (st.playing && t < total()) { t = Math.min(total(), t + dt); dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     4. Group-wise scales and an outlier
     ===================================================================== */
  AI.labs['gen-ef-quant-group'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One scale per group: containing an outlier', 'illustrative: 64 seeded weights ~ N(0, 0.1²) plus one outlier; each group gets its own scale and zero point',
      `<canvas style="height:340px"></canvas>`,
      `${sl('gs', 'Group size (weights sharing one scale)', 0, 5, 1, 0)}${sl('bits', 'Bits per weight', 2, 8, 1, 4)}${sl('out', 'Outlier size (weight 21)', 0, 3, 0.1, 1.2)}${sl('rate', 'Groups per second', 0.5, 8, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out2" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true }, N = 64, sizes = [64, 32, 16, 8, 4, 2];
    const rw = AI.rng(21), base = []; for (let i = 0; i < N; i++) base.push(0.1 * AI.randn(rw));
    let t = 0, dirty = true, res;
    const compute = () => {
      const gsz = sizes[+q(ctl, '#gs').value], bits = +q(ctl, '#bits').value, w = base.slice(); w[20] = +q(ctl, '#out').value;
      const err = new Array(N).fill(0), scales = []; let se = 0, me = 0;
      for (let s0 = 0; s0 < N; s0 += gsz) { const qd = quantise(w.slice(s0, s0 + gsz), bits); scales.push(qd.s); qd.rows.forEach((r, k) => { err[s0 + k] = r.e; se += r.e * r.e; me = Math.max(me, Math.abs(r.e)); }); }
      res = { gsz, bits, w, err, scales, mse: se / N, maxe: me, groups: Math.ceil(N / gsz), eff: bits + 32 / gsz };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 40, R = W - 12, bw = (R - L) / N, doneG = Math.min(res.groups, Math.floor(t));
      const mid = 100, vmax = Math.max(0.5, ...res.w.map(Math.abs)) * 1.05;
      txt(g, 'weights (bars) and the rounding error each one picks up (below)', L, 14, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, mid); g.lineTo(R, mid); g.stroke();
      res.w.forEach((v, i) => { g.fillStyle = i === 20 ? c.red : c.blue; g.globalAlpha = 0.8; const h = v / vmax * 70; g.fillRect(L + i * bw + 1, h >= 0 ? mid - h : mid, bw - 2, Math.abs(h)); g.globalAlpha = 1; });
      const emax = Math.max(0.02, ...res.err.map(Math.abs)) * 1.1, mid2 = 250;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, mid2); g.lineTo(R, mid2); g.stroke();
      for (let k = 0; k < res.groups; k++) {
        const a = k * res.gsz, b = Math.min(N, a + res.gsz), on = k < doneG;
        g.fillStyle = k % 2 ? c.bg3 : c.bg2; g.globalAlpha = 0.6; g.fillRect(L + a * bw, 188, (b - a) * bw, 124); g.globalAlpha = 1;
        if (k === doneG && st.playing !== undefined) { g.strokeStyle = c.orange; g.lineWidth = 2; g.strokeRect(L + a * bw, 22, (b - a) * bw, 292); g.lineWidth = 1; }
        for (let i = a; i < b; i++) { if (!on) continue; const h = res.err[i] / emax * 60; g.fillStyle = c.orange; g.fillRect(L + i * bw + 1, h >= 0 ? mid2 - h : mid2, bw - 2, Math.abs(h)); }
      }
      txt(g, `+${emax.toFixed(2)}`, L - 4, 192, c.muted, '10px sans-serif', 'right'); txt(g, `-${emax.toFixed(2)}`, L - 4, 312, c.muted, '10px sans-serif', 'right');
      let se = 0, cnt = 0; for (let i = 0; i < Math.min(N, doneG * res.gsz); i++) { se += res.err[i] ** 2; cnt++; }
      q(ctl, '#out2').innerHTML = `group size <b>${res.gsz}</b> -> ${res.groups} scales; effective cost = ${res.bits} + 32/${res.gsz} = <b>${res.eff.toFixed(2)}</b> bits per weight (assuming one 16-bit scale and one 16-bit zero point per group)<br>mean squared error so far: ${cnt ? (se / cnt).toExponential(3) : '-'}; final over all 64: <b>${res.mse.toExponential(3)}</b>, largest |error| ${res.maxe.toFixed(4)}`;
      q(ctl, '#cap').textContent = doneG < res.groups ? `Quantising group ${doneG + 1} of ${res.groups}. Its scale is set by its own largest and smallest values.` : (res.gsz === 64 ? 'One scale for all 64 weights: the outlier stretches the grid, so the small weights are rounded very coarsely.' : 'Only the group that contains the outlier gets a coarse grid; the other groups keep a fine one.');
      showVals(ctl, { gs: (v) => sizes[v], bits: (v) => v, out: (v) => v, rate: (v) => v });
    };
    const step = () => { t = Math.min(res.groups, Math.floor(t) + 1); dirty = true; };
    const reset = () => { t = 0; dirty = true; };
    wire(ctl, st, step, reset, draw);
    const re = () => { compute(); reset(); draw(); };
    ['#gs', '#bits', '#out'].forEach((s) => (q(ctl, s).oninput = re)); q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; compute(); draw();
    AI.loop(root, (dt) => { if (st.playing && t < res.groups) { t = Math.min(res.groups, t + dt * +q(ctl, '#rate').value); dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     5. Roofline and timeline: prefill versus decode
     ===================================================================== */
  AI.labs['gen-ef-roofline'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Prefill is compute-bound, decode is memory-bound', 'illustrative toy: round numbers, not any real chip or model; time of a phase = max(compute time, memory time)',
      `<canvas style="height:430px"></canvas>`,
      `${ni('P', 'Parameters (billions)', 1, 0.1, 100, 0.1)}${ni('peak', 'Peak compute (TFLOP/s)', 100, 1, 10000, 1)}${ni('bw', 'Memory bandwidth (TB/s)', 1, 0.1, 100, 0.1)}
       ${sl('T', 'Prompt length T (tokens)', 1, 2048, 1, 512)}${sl('B', 'Decode batch size B', 1, 256, 1, 1)}${sl('ctx', 'Context length per sequence', 0, 4096, 64, 512)}
       ${sl('rate', 'Animation speed (model ms per second)', 1, 30, 1, 4)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true }, KV = 65536, NS = 10;
    let clock = 0, dirty = true, m;
    const calc = () => {
      const P = (+q(ctl, '#P').value || 1) * 1e9, pk = (+q(ctl, '#peak').value || 1) * 1e12, bw = (+q(ctl, '#bw').value || 1) * 1e12;
      const T = +q(ctl, '#T').value, B = +q(ctl, '#B').value, ctx = +q(ctl, '#ctx').value;
      const pf = 2 * P * T, pb = 2 * P, df = 2 * P * B, db = 2 * P + B * ctx * KV;
      m = { P, pk, bw, T, B, ctx, pf, pb, df, db, ptc: pf / pk * 1e3, ptm: pb / bw * 1e3, dtc: df / pk * 1e3, dtm: db / bw * 1e3 };
      m.pt = Math.max(m.ptc, m.ptm); m.dt = Math.max(m.dtc, m.dtm); m.ridge = pk / bw;
    };
    const total = () => m.pt + NS * m.dt;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 56, R = W - 16, rt = 26, rb = 190;
      txt(g, 'roofline: speed against FLOP per byte (log scales)', L, 16, c.muted, '11px sans-serif');
      const x0 = Math.log10(0.5), x1 = Math.log10(5000), y1 = Math.log10(m.pk * 1.5), y0 = y1 - 4.2;
      const X = (v) => L + (Math.log10(v) - x0) / (x1 - x0) * (R - L), Y = (v) => rb - (Math.log10(Math.max(v, 1)) - y0) / (y1 - y0) * (rb - rt);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, rt); g.lineTo(L, rb); g.lineTo(R, rb); g.stroke();
      for (const v of [1, 10, 100, 1000]) txt(g, v, X(v), rb + 13, c.muted, '10px sans-serif', 'center');
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(X(0.5), Y(m.bw * 0.5)); g.lineTo(X(m.ridge), Y(m.pk)); g.lineTo(R, Y(m.pk)); g.stroke(); g.lineWidth = 1;
      txt(g, `ridge ${m.ridge.toFixed(0)} FLOP/byte`, X(m.ridge) + 4, Y(m.pk) + 14, c.muted, '10px sans-serif');
      const ai = (f, b) => f / b;
      const pts = [[ai(m.pf, m.pb), 'prefill', c.blue], [ai(m.df, m.db), 'decode', c.orange]];
      pts.forEach(([a, lab, col]) => { const px = X(clamp(a, 0.5, 5000)), py = Y(Math.min(m.pk, m.bw * a)); g.fillStyle = col; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill(); txt(g, `${lab} (${a.toFixed(a < 10 ? 2 : 0)})`, px + 9, py + (lab === 'prefill' ? -6 : 14), col, 'bold 11px sans-serif'); });
      // timeline
      const ty = 232, lane = 34, tx0 = L, tx1 = R, S = (ms) => tx0 + ms / total() * (tx1 - tx0);
      txt(g, 'one request: prefill, then decode steps', L, ty - 12, c.muted, '11px sans-serif');
      txt(g, 'compute', tx0 - 6, ty + lane * 0.6, c.blue, '10px sans-serif', 'right'); txt(g, 'memory', tx0 - 6, ty + lane * 1.6 + 6, c.orange, '10px sans-serif', 'right');
      const phases = [[0, m.ptc, m.ptm, m.pt]]; for (let i = 0; i < NS; i++) phases.push([m.pt + i * m.dt, m.dtc, m.dtm, m.dt]);
      phases.forEach(([s0, tc, tm, d], k) => {
        const e = clamp(clock - s0, 0, d), f = (x) => Math.min(x, e) / total() * (tx1 - tx0);
        g.fillStyle = c.bg3; g.fillRect(S(s0), ty, S(s0 + d) - S(s0) - 1, lane * 2 + 8);
        g.fillStyle = c.blue; g.globalAlpha = 0.85; g.fillRect(S(s0), ty + 2, f(tc), lane - 4); g.globalAlpha = 1;
        g.fillStyle = c.orange; g.globalAlpha = 0.85; g.fillRect(S(s0), ty + lane + 6, f(tm), lane - 4); g.globalAlpha = 1;
        if (k === 0) txt(g, 'prefill', S(s0) + 3, ty + lane * 2 + 22, c.muted, '10px sans-serif');
      });
      txt(g, 'decode steps', S(m.pt) + 3, ty + lane * 2 + 22, c.muted, '10px sans-serif');
      g.strokeStyle = c.fg; g.beginPath(); g.moveTo(S(Math.min(clock, total())), ty - 4); g.lineTo(S(Math.min(clock, total())), ty + lane * 2 + 12); g.stroke();
      const stepsDone = clock <= m.pt ? 0 : Math.min(NS, Math.floor((clock - m.pt) / m.dt));
      txt(g, `model time ${Math.min(clock, total()).toFixed(2)} ms · decode steps done ${stepsDone}`, L, H - 22, c.fg, '11px sans-serif'); txt(g, `tokens produced across the batch: ${stepsDone * m.B}`, L, H - 8, c.fg, '11px sans-serif');
      const bound = (tc, tm) => (tc > tm ? 'compute-bound' : 'memory-bound');
      const tput = m.B / (m.dt / 1e3);
      q(ctl, '#out').innerHTML = `<b>prefill</b> (T = ${m.T}): FLOPs 2PT = ${m.pf.toExponential(2)}, bytes ≈ 2P = ${m.pb.toExponential(2)}, intensity ${ai(m.pf, m.pb).toFixed(1)}; compute ${m.ptc.toFixed(3)} ms, memory ${m.ptm.toFixed(3)} ms -> ${bound(m.ptc, m.ptm)}, ${m.pt.toFixed(3)} ms<br><b>one decode step</b> (B = ${m.B}, context ${m.ctx}): FLOPs 2PB = ${m.df.toExponential(2)}, bytes 2P + B·ctx·kv = ${m.db.toExponential(2)}, intensity ${ai(m.df, m.db).toFixed(2)}; compute ${m.dtc.toFixed(3)} ms, memory ${m.dtm.toFixed(3)} ms -> ${bound(m.dtc, m.dtm)}, ${m.dt.toFixed(3)} ms<br>decode throughput B/step time = <b>${fmtN(tput)}</b> tokens/s across the batch; latency per token for each sequence = ${m.dt.toFixed(3)} ms`;
      q(ctl, '#cap').textContent = m.dtm > m.dtc ? 'Each decode step must stream all the weights (and every cached key and value) from memory to do very little arithmetic: the orange lane is the long one. A bigger batch reuses the same weights for more tokens, which is why batching raises throughput.' : 'At this batch size the decode step has crossed the ridge: compute, not memory, now limits it, so a still bigger batch no longer helps throughput.';
      showVals(ctl, { T: (v) => v, B: (v) => v, ctx: (v) => v, rate: (v) => v + ' ms/s' });
    };
    const step = () => { clock = Math.min(total(), clock + (clock < m.pt ? m.pt : m.dt)); dirty = true; };
    const reset = () => { clock = 0; dirty = true; };
    wire(ctl, st, step, reset, draw);
    const re = () => { calc(); dirty = true; };
    ['#P', '#peak', '#bw', '#T', '#B', '#ctx'].forEach((s) => (q(ctl, s).oninput = re)); q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; calc(); draw();
    AI.loop(root, (dt) => { if (st.playing) { clock += dt * +q(ctl, '#rate').value; if (clock > total() + 2) clock = 0; dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     6. Static versus continuous batching
     ===================================================================== */
  AI.labs['gen-ef-batching'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Static versus continuous batching', 'illustrative: 12 requests with different output lengths; every decode step takes the same time (the memory-bound assumption)',
      `<canvas style="height:380px"></canvas>`,
      `${sl('S', 'Slots in the batch', 2, 8, 1, 4)}${sl('mx', 'Longest output (tokens)', 4, 24, 1, 16)}${sl('rate', 'Decode steps per second', 1, 12, 0.5, 3)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true }, NR = 12;
    let cur = 0, dirty = true, tick = AI.ticker(), sim;
    const build = () => {
      const S = +q(ctl, '#S').value, mx = +q(ctl, '#mx').value, rr = AI.rng(9), len = [];
      for (let i = 0; i < NR; i++) len.push(2 + Math.floor(rr() * (mx - 1)));
      const stat = Array.from({ length: S }, () => []), cont = Array.from({ length: S }, () => []);
      // static: groups of S requests, each group runs max length
      for (let b = 0; b < NR; b += S) { const grp = len.slice(b, b + S), steps = Math.max(...grp); for (let s = 0; s < S; s++) for (let k = 0; k < steps; k++) stat[s].push(s < grp.length && k < grp[s] ? b + s : -1); }
      // continuous: a freed slot immediately takes the next waiting request
      let next = 0; const rem = new Array(S).fill(0), id = new Array(S).fill(-1);
      for (let k = 0; k < 400; k++) {
        for (let s = 0; s < S; s++) if (rem[s] === 0 && next < NR) { id[s] = next; rem[s] = len[next]; next++; }
        if (rem.every((x) => x === 0)) break;
        for (let s = 0; s < S; s++) { if (rem[s] > 0) { cont[s].push(id[s]); rem[s]--; } else cont[s].push(-1); }
      }
      const ss = stat[0].length, cs = cont[0].length, tot = len.reduce((a, b) => a + b, 0);
      sim = { S, len, stat, cont, ss, cs, tot, us: tot / (S * ss), uc: tot / (S * cs), max: Math.max(ss, cs) };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pal = [c.blue, c.green, c.orange, c.purple, c.teal, c.pink, c.yellow, c.red], L = 70, R = W - 12, cw = (R - L) / sim.max;
      const panel = (name, grid, y0, h, steps) => {
        txt(g, name, 10, y0 - 8, c.fg, 'bold 12px sans-serif');
        const rh = Math.min(26, h / sim.S);
        for (let s = 0; s < sim.S; s++) { txt(g, 'slot ' + (s + 1), L - 6, y0 + s * rh + rh * 0.65, c.muted, '10px sans-serif', 'right'); for (let k = 0; k < steps; k++) { const v = grid[s][k]; g.strokeStyle = c.border; g.strokeRect(L + k * cw + 0.5, y0 + s * rh + 0.5, cw - 1, rh - 1); if (k < cur && v >= 0) { g.fillStyle = pal[v % pal.length]; g.globalAlpha = 0.85; g.fillRect(L + k * cw + 1, y0 + s * rh + 1, cw - 2, rh - 2); g.globalAlpha = 1; } } }
        return y0 + sim.S * rh;
      };
      const h = (H - 100) / 2;
      const e1 = panel(`static batching: ${sim.ss} steps`, sim.stat, 28, h, sim.ss);
      panel(`continuous batching: ${sim.cs} steps`, sim.cont, e1 + 40, h, sim.cs);
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(L + cur * cw, 18); g.lineTo(L + cur * cw, H - 14); g.stroke(); g.lineWidth = 1;
      txt(g, `step ${Math.min(cur, sim.max)} · colour = a request · empty = idle slot`, L, H - 4, c.muted, '11px sans-serif');
      q(ctl, '#out').innerHTML = `request lengths (tokens): ${sim.len.join(', ')} (total ${sim.tot})<br>static: ${sim.ss} steps, utilisation ${(sim.us * 100).toFixed(1)}%<br>continuous: ${sim.cs} steps, utilisation ${(sim.uc * 100).toFixed(1)}%, ${(sim.ss / sim.cs).toFixed(2)}× faster for the whole set`;
      q(ctl, '#cap').textContent = 'Static batching holds every slot until the longest request in the group finishes. Continuous batching refills a slot the moment its request ends, so the batch stays full.';
      showVals(ctl, { S: (v) => v, mx: (v) => v, rate: (v) => v });
    };
    const step = () => { if (cur < sim.max) cur++; dirty = true; };
    const reset = () => { cur = 0; dirty = true; };
    wire(ctl, st, step, reset, draw);
    ['#S', '#mx'].forEach((s) => (q(ctl, s).oninput = () => { build(); reset(); draw(); })); q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; build(); draw();
    AI.loop(root, (dt) => { if (st.playing && cur < sim.max) { const k = tick(dt, +q(ctl, '#rate').value); if (k) { cur = Math.min(sim.max, cur + k); dirty = true; } } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     7. Speculative decoding: accept or resample on a 3-token vocabulary
     ===================================================================== */
  AI.labs['gen-ef-spec'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Speculative decoding on a 3-token vocabulary', 'illustrative: the draft model q proposes k tokens, the target model p verifies them; the same p and q at every position',
      `<canvas style="height:400px"></canvas>`,
      `<b>Target p (weights)</b>${sl('p0', 'p: token A', 0.5, 10, 0.5, 5)}${sl('p1', 'p: token B', 0.5, 10, 0.5, 3)}${sl('p2', 'p: token C', 0.5, 10, 0.5, 2)}
       <b>Draft q (weights)</b>${sl('q0', 'q: token A', 0.5, 10, 0.5, 3)}${sl('q1', 'q: token B', 0.5, 10, 0.5, 2)}${sl('q2', 'q: token C', 0.5, 10, 0.5, 5)}
       ${sl('k', 'Draft length k', 1, 6, 1, 4)}${sl('rate', 'Animation speed (events per second)', 0.5, 6, 0.5, 1.5)}
       ${btns}<div class="row"><button class="btn" id="bulk">Simulate 20 000 steps instantly</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true }, NAMES = ['A', 'B', 'C'];
    let rnd, evs, ei, acc, counts, nTok, nSteps, hist, dirty = true, shown, drafted, verdicts, emitted, bulkNote = '';
    const dists = () => { const pw = [0, 1, 2].map((i) => +q(ctl, '#p' + i).value), qw = [0, 1, 2].map((i) => +q(ctl, '#q' + i).value), ps = pw.reduce((a, b) => a + b), qs = qw.reduce((a, b) => a + b); return { p: pw.map((x) => x / ps), q: qw.map((x) => x / qs) }; };
    const sample = (d) => { let u = rnd(), s = 0; for (let i = 0; i < d.length; i++) { s += d[i]; if (u < s) return i; } return d.length - 1; };
    const oneStep = (k, d) => {
      const out = [], ev = []; let allOk = true;
      const dr = []; for (let i = 0; i < k; i++) dr.push(sample(d.q));
      dr.forEach((x, i) => ev.push({ t: 'draft', i, x }));
      for (let i = 0; i < k; i++) {
        const x = dr[i], ratio = Math.min(1, d.p[x] / d.q[x]), u = rnd(), ok = u < ratio;
        ev.push({ t: 'verify', i, x, ratio, u, ok });
        if (ok) out.push(x); else {
          const res = d.p.map((pv, j) => Math.max(0, pv - d.q[j])), s = res.reduce((a, b) => a + b, 0);
          const y = sample(res.map((v) => v / s)); out.push(y); ev.push({ t: 'fix', i, x: y }); allOk = false; break;
        }
      }
      if (allOk) { const y = sample(d.p); out.push(y); ev.push({ t: 'bonus', x: y }); }
      return { out, ev };
    };
    const reset = () => { rnd = AI.rng(17); counts = [0, 0, 0]; nTok = 0; nSteps = 0; hist = []; evs = []; ei = 0; acc = 0; drafted = []; verdicts = []; emitted = []; bulkNote = ''; dirty = true; };
    const nextStep = () => { const r = oneStep(+q(ctl, '#k').value, dists()); evs = r.ev; ei = 0; drafted = []; verdicts = []; emitted = []; evs.pending = r.out; };
    const advance = () => {
      if (ei >= evs.length) { nextStep(); }
      const e = evs[ei++];
      if (e.t === 'draft') drafted.push(e.x); else if (e.t === 'verify') { verdicts[e.i] = e; if (e.ok) emitted.push(e.x); } else emitted.push(e.x);
      if (e.t === 'fix' || e.t === 'bonus') { const out = evs.pending; out.forEach((x) => { counts[x]++; }); nTok += out.length; nSteps++; hist.push(out.slice()); if (hist.length > 6) hist.shift(); }
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const d = dists(), k = +q(ctl, '#k').value, cols = [c.blue, c.orange, c.purple];
      const alpha = d.p.reduce((s, pv, i) => s + Math.min(pv, d.q[i]), 0), Et = (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
      // top: distributions
      txt(g, 'p filled, q outline', 10, 16, c.muted, '11px sans-serif');
      const half = (W - 24) / 2, colw = half / 3, bw = (colw - 8) / 2, bx = 10, by = 120, bh = 80;
      for (let i = 0; i < 3; i++) {
        const x = bx + i * colw;
        g.fillStyle = cols[i]; g.globalAlpha = 0.8; g.fillRect(x, by - d.p[i] * bh, bw, d.p[i] * bh); g.globalAlpha = 1;
        g.strokeStyle = c.fg; g.lineWidth = 2; g.strokeRect(x + bw + 2, by - d.q[i] * bh, bw, d.q[i] * bh); g.lineWidth = 1;
        txt(g, `${NAMES[i]} p${d.p[i].toFixed(2).slice(1)} q${d.q[i].toFixed(2).slice(1)}`, x, by + 14, c.muted, '10px sans-serif');
      }
      txt(g, `α = Σ min(p,q) = ${alpha.toFixed(3)}`, bx, 150, c.fg, 'bold 11px sans-serif');
      // histogram of emitted tokens vs p
      const hx = 10 + half + 14, hw = bw;
      txt(g, `emitted (${nTok}): share vs p`, hx, 16, c.muted, '11px sans-serif');
      for (let i = 0; i < 3; i++) {
        const x = hx + i * colw, fr = nTok ? counts[i] / nTok : 0;
        g.fillStyle = cols[i]; g.globalAlpha = 0.8; g.fillRect(x, by - fr * bh, hw, fr * bh); g.globalAlpha = 1;
        g.strokeStyle = c.fg; g.setLineDash([4, 3]); g.strokeRect(x + hw + 2, by - d.p[i] * bh, hw, d.p[i] * bh); g.setLineDash([]);
        txt(g, `${NAMES[i]} ${fr.toFixed(2)}`, x, by + 14, c.muted, '10px sans-serif');
      }
      // step diagram
      const sy = 190, cw = Math.min(54, (W - 130) / (k + 1)), mk = (x, y, label, col, sub) => { g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x, y, cw - 6, 30); g.globalAlpha = 1; txt(g, label, x + (cw - 6) / 2, y + 20, '#fff', 'bold 14px sans-serif', 'center'); if (sub) txt(g, sub, x + (cw - 6) / 2, y + 44, c.muted, '10px sans-serif', 'center'); };
      txt(g, `step ${nSteps + 1}: draft row (what q proposed) then verdicts from the target`, 12, sy - 6, c.muted);
      txt(g, 'draft', 12, sy + 24, c.fg, 'bold 11px sans-serif'); txt(g, 'target', 12, sy + 84, c.fg, 'bold 11px sans-serif');
      for (let i = 0; i < k; i++) {
        const x = 70 + i * cw;
        g.strokeStyle = c.border; g.strokeRect(x, sy + 6, cw - 6, 30); g.strokeRect(x, sy + 66, cw - 6, 30);
        if (i < drafted.length) mk(x, sy + 6, NAMES[drafted[i]], cols[drafted[i]], `q ${d.q[drafted[i]].toFixed(2)}`);
        const v = verdicts[i];
        if (v) { mk(x, sy + 66, v.ok ? '✓' : '✗', v.ok ? c.green : c.red); txt(g, `p/q = ${(d.p[v.x] / d.q[v.x]).toFixed(2)}`, x + (cw - 6) / 2, sy + 110, c.muted, '10px sans-serif', 'center'); txt(g, `u = ${v.u.toFixed(2)}`, x + (cw - 6) / 2, sy + 122, v.ok ? c.green : c.red, '10px sans-serif', 'center'); }
      }
      const last = evs && evs.length ? evs[Math.min(ei, evs.length) - 1] : null;
      if (last && (last.t === 'fix' || last.t === 'bonus')) { const x = 70 + Math.min(last.t === 'fix' ? last.i : k, k) * cw; mk(x, sy + 66, NAMES[last.x], cols[last.x]); txt(g, last.t === 'fix' ? 'resampled' : 'bonus from p', x + (cw - 6) / 2, sy + 110, c.orange, '10px sans-serif', 'center'); }
      // emitted history
      txt(g, 'tokens emitted, newest step last:', 12, sy + 150, c.muted);
      hist.forEach((h, r) => { const y = sy + 160 + r * 22; h.forEach((x, i) => { g.fillStyle = cols[x]; g.globalAlpha = 0.85; g.fillRect(12 + i * 26, y, 22, 17); g.globalAlpha = 1; txt(g, NAMES[x], 23 + i * 26, y + 13, '#fff', 'bold 11px sans-serif', 'center'); }); });
      const spd = nSteps ? nTok / nSteps : 0;
      q(ctl, '#out').innerHTML = `α = Σ min(p, q) = <b>${alpha.toFixed(4)}</b>; formula E[tokens per step] = (1 − α^(k+1))/(1 − α) = <b>${Et.toFixed(4)}</b><br>simulated so far: ${nTok} tokens in ${nSteps} steps = <b>${spd.toFixed(4)}</b> per step${bulkNote ? '<br>' + bulkNote : ''}`;
      const cp = ei > 0 ? evs[ei - 1] : null;
      q(ctl, '#cap').textContent = !cp ? 'Press play. Each step: the draft proposes k tokens, then the target checks them from left to right.' : cp.t === 'draft' ? 'The draft model proposes tokens cheaply, one after another.' : cp.t === 'verify' ? `Token ${NAMES[cp.x]}: accept with probability min(1, p/q) = ${cp.ratio.toFixed(2)}. Random number u = ${cp.u.toFixed(2)} -> ${cp.ok ? 'accepted' : 'rejected, and everything after it is thrown away'}.` : cp.t === 'fix' ? 'Rejection: one replacement token is drawn from the leftover mass max(0, p − q), renormalised. The step ends here.' : 'All drafts accepted, so the target adds one extra token for free from its own distribution.';
      showVals(ctl, { p0: (v) => v, p1: (v) => v, p2: (v) => v, q0: (v) => v, q1: (v) => v, q2: (v) => v, k: (v) => v, rate: (v) => v });
    };
    const step = () => { advance(); };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#bulk').onclick = () => {
      const d = dists(), k = +q(ctl, '#k').value, r2 = AI.rng(99); const keep = rnd; rnd = r2;
      const cnt = [0, 0, 0]; let tk = 0; const n = 20000;
      for (let s = 0; s < n; s++) { const r = oneStep(k, d); r.out.forEach((x) => { cnt[x]++; }); tk += r.out.length; }
      rnd = keep; const alpha = d.p.reduce((s, pv, i) => s + Math.min(pv, d.q[i]), 0);
      bulkNote = `20 000 steps: ${tk} tokens = ${(tk / n).toFixed(4)} per step (formula ${((1 - Math.pow(alpha, k + 1)) / (1 - alpha)).toFixed(4)}); shares A/B/C = ${cnt.map((x) => (x / tk).toFixed(4)).join(' / ')} against p = ${d.p.map((x) => x.toFixed(4)).join(' / ')}`; dirty = true;
    };
    ['p0', 'p1', 'p2', 'q0', 'q1', 'q2', 'k'].forEach((id) => (q(ctl, '#' + id).oninput = () => { reset(); draw(); })); q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing) { acc += dt * +q(ctl, '#rate').value; while (acc >= 1) { acc -= 1; advance(); } } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     8. Speculative decoding speed-up curve
     ===================================================================== */
  AI.labs['gen-ef-spec-speed'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'How much faster? Tokens per step and speed-up against k', 'illustrative: independent acceptance with the same probability α at every position; draft cost c is a fraction of one target step',
      `<canvas style="height:320px"></canvas>`,
      `${sl('al', 'Acceptance probability α', 0.05, 0.98, 0.01, 0.7)}${sl('c', 'Draft cost c (one draft step, as a share of a target step)', 0, 0.5, 0.01, 0.1)}${sl('k', 'Draft length k', 1, 12, 1, 4)}${sl('rate', 'Sweep speed (α per second)', 0.02, 0.3, 0.01, 0.06)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: false };
    let dirty = true, dir = 1;
    const E = (a, k) => (1 - Math.pow(a, k + 1)) / (1 - a);
    const simE = (a, k) => { const r = AI.rng(5); let tk = 0; const n = 20000; for (let s = 0; s < n; s++) { let m = 0; while (m < k && r() < a) m++; tk += m + 1; } return tk / n; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const a = +q(ctl, '#al').value, cc = +q(ctl, '#c').value, k = +q(ctl, '#k').value, L = 46, R = W - 14, t0 = 24, t1 = H - 34, KM = 12;
      const ymax = Math.max(4, E(a, KM)) * 1.05, X = (kk) => L + (kk / KM) * (R - L), Y = (v) => t1 - v / ymax * (t1 - t0);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      for (let kk = 0; kk <= KM; kk += 2) txt(g, kk, X(kk), t1 + 14, c.muted, '11px sans-serif', 'center');
      txt(g, 'draft length k', (L + R) / 2, H - 6, c.muted, '11px sans-serif', 'center');
      for (let v = 0; v <= Math.floor(ymax); v++) txt(g, v, L - 6, Y(v) + 4, c.muted, '11px sans-serif', 'right');
      const curve = (f, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let kk = 0; kk <= KM; kk++) { if (kk) g.lineTo(X(kk), Y(f(kk))); else g.moveTo(X(kk), Y(f(kk))); } g.stroke(); g.lineWidth = 1; };
      curve((kk) => E(a, kk), c.blue); curve((kk) => E(a, kk) / (1 + cc * kk), c.orange);
      txt(g, 'expected tokens per target step', X(KM) - 4, Y(E(a, KM)) - 6, c.blue, 'bold 11px sans-serif', 'right');
      txt(g, 'speed-up over plain decoding', X(KM) - 4, Y(E(a, KM) / (1 + cc * KM)) + 14, c.orange, 'bold 11px sans-serif', 'right');
      let bk = 1, bs = 0; for (let kk = 1; kk <= KM; kk++) { const s = E(a, kk) / (1 + cc * kk); if (s > bs) { bs = s; bk = kk; } }
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(k), t0); g.lineTo(X(k), t1); g.stroke(); g.setLineDash([]);
      g.fillStyle = c.green; g.beginPath(); g.arc(X(k), Y(simE(a, k)), 5, 0, 7); g.fill();
      txt(g, 'green dot = simulated', X(k) + 8, Y(simE(a, k)) - 8, c.green, '11px sans-serif');
      q(ctl, '#out').innerHTML = `k = ${k}: E[tokens] = (1 − ${a.toFixed(2)}^${k + 1})/(1 − ${a.toFixed(2)}) = <b>${E(a, k).toFixed(4)}</b>, simulated ${simE(a, k).toFixed(4)}<br>time per step = 1 + c·k = ${(1 + cc * k).toFixed(2)} target-steps, so speed-up = <b>${(E(a, k) / (1 + cc * k)).toFixed(3)}×</b><br>best k for these α and c is ${bk} (${bs.toFixed(3)}×); the limit as k grows is ${(1 / (1 - a)).toFixed(2)} tokens per step`;
      q(ctl, '#cap').textContent = 'Longer drafts help less and less (the blue curve flattens toward 1/(1−α)) while every extra draft step costs time, so the orange curve peaks and then falls.';
      showVals(ctl, { al: (v) => v, c: (v) => v, k: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => { const e = q(ctl, '#al'); e.value = Math.min(0.98, +e.value + 0.03); dirty = true; }, () => { q(ctl, '#al').value = 0.7; q(ctl, '#c').value = 0.1; q(ctl, '#k').value = 4; dirty = true; }, draw);
    st.playing = false; q(ctl, '#play').textContent = '▶ Play';
    ['#al', '#c', '#k', '#rate'].forEach((s) => (q(ctl, s).oninput = () => { dirty = true; }));
    o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing) { const e = q(ctl, '#al'); let v = +e.value + dir * dt * +q(ctl, '#rate').value; if (v > 0.98) { v = 0.98; dir = -1; } if (v < 0.05) { v = 0.05; dir = 1; } e.value = v; dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     9. Distillation on a 4-token vocabulary
     ===================================================================== */
  AI.labs['gen-ef-distill'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Distillation: a student learns the teacher\'s soft targets', 'illustrative: the student is a free vector of 4 logits trained by gradient descent on T²·KL(p_T ‖ q_T)',
      `<canvas style="height:340px"></canvas>`,
      `<b>Teacher logits</b>${sl('t0', 'token 1', -4, 6, 0.5, 4)}${sl('t1', 'token 2', -4, 6, 0.5, 2)}${sl('t2', 'token 3', -4, 6, 0.5, 1)}${sl('t3', 'token 4', -4, 6, 0.5, -1)}
       ${sl('T', 'Temperature T', 1, 5, 0.5, 2)}
       <label>Train on <select id="mode"><option value="soft">teacher soft targets (KL at temperature T)</option><option value="hard">hard label only (teacher's top token)</option></select></label>
       ${sl('lr', 'Learning rate', 0.01, 0.4, 0.01, 0.1)}${sl('rate', 'Gradient steps per second', 2, 60, 1, 10)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let z, steps, hist, dirty = true, tick = AI.ticker();
    const tl = () => [0, 1, 2, 3].map((i) => +q(ctl, '#t' + i).value), T = () => +q(ctl, '#T').value;
    const kl = (a, b) => a.reduce((s, x, i) => s + (x > 1e-12 ? x * Math.log(x / Math.max(b[i], 1e-12)) : 0), 0);
    const metric = () => kl(softmax(tl()), softmax(z));
    const reset = () => { z = [0, 0, 0, 0]; steps = 0; hist = [metric()]; dirty = true; };
    const stepGD = () => {
      const lr = +q(ctl, '#lr').value, t = tl();
      if (q(ctl, '#mode').value === 'soft') { const TT = T(), pt = softmax(t.map((x) => x / TT)), qs = softmax(z.map((x) => x / TT)); for (let i = 0; i < 4; i++) z[i] -= lr * TT * (qs[i] - pt[i]); }
      else { const qs = softmax(z), top = t.indexOf(Math.max(...t)); for (let i = 0; i < 4; i++) z[i] -= lr * (qs[i] - (i === top ? 1 : 0)); }
      steps++; hist.push(metric()); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const TT = T(), soft = q(ctl, '#mode').value === 'soft', Tv = soft ? TT : 1;
      const pt = softmax(tl().map((x) => x / Tv)), qs = softmax(z.map((x) => x / Tv));
      txt(g, `temperature ${Tv}: student filled, teacher outline`, 10, 16, c.muted, '11px sans-serif');
      const colw = (W - 24) / 4, bw = (colw - 8) / 2, by = 130, bh = 100, bx = 10;
      for (let i = 0; i < 4; i++) { const x = bx + i * colw; g.fillStyle = c.blue; g.globalAlpha = 0.8; g.fillRect(x, by - qs[i] * bh, bw, qs[i] * bh); g.globalAlpha = 1; g.strokeStyle = c.orange; g.lineWidth = 2; g.strokeRect(x + bw + 2, by - pt[i] * bh, bw, pt[i] * bh); g.lineWidth = 1; txt(g, `${i + 1}: S${qs[i].toFixed(2).slice(1)} T${pt[i].toFixed(2).slice(1)}`, x, by + 14, c.muted, '10px sans-serif'); }
      const L = 46, R = W - 14, cy0 = 190, cy1 = H - 24, ymax = Math.max(0.5, hist[0]) * 1.05, xm = Math.max(60, Math.ceil(hist.length / 60) * 60);
      txt(g, 'KL(teacher ‖ student) at T = 1, against step', L, cy0 - 8, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, cy0); g.lineTo(L, cy1); g.lineTo(R, cy1); g.stroke();
      const X = (k) => L + k / xm * (R - L), Y = (v) => cy1 - Math.min(v, ymax) / ymax * (cy1 - cy0);
      txt(g, '0', L - 6, cy1 + 4, c.muted, '11px sans-serif', 'right'); txt(g, ymax.toFixed(2), L - 6, cy0 + 8, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(0), Y(v)))); g.stroke(); g.lineWidth = 1;
      const t1 = softmax(tl()), s1 = softmax(z);
      q(ctl, '#out').innerHTML = `student logits: ${z.map((x) => x.toFixed(2)).join(', ')}<br>at T = 1: teacher ${t1.map((x) => x.toFixed(3)).join(', ')} · student ${s1.map((x) => x.toFixed(3)).join(', ')}<br>KL(teacher ‖ student) at T = 1: <b>${metric().toFixed(4)}</b> after ${steps} steps; ratio token 2 / token 3 — teacher ${(t1[1] / t1[2]).toFixed(2)}, student ${(s1[1] / s1[2]).toFixed(2)}`;
      q(ctl, '#cap').textContent = soft ? (TT > 1 ? `At T = ${TT} the teacher's small probabilities are lifted, so the student can see which wrong tokens are "less wrong". The student ends up with the teacher's whole ranking.` : 'At T = 1 the teacher is quite confident, so the small probabilities give a weak signal. Raise T to soften it.') : 'With only the hard label the student is pushed to put all its probability on one token: it learns nothing about how the other tokens rank, and the KL to the teacher stays large.';
      showVals(ctl, { t0: (v) => v, t1: (v) => v, t2: (v) => v, t3: (v) => v, T: (v) => v, lr: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, stepGD, reset, draw);
    ['#t0', '#t1', '#t2', '#t3', '#T', '#mode'].forEach((s) => (q(ctl, s).oninput = () => { reset(); draw(); })); q(ctl, '#mode').onchange = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && steps < 4000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) stepGD(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     10. MoE router with a load-balancing loss
     ===================================================================== */
  AI.labs['gen-ef-moe-router'] = function (root) {
    const E = 4, NT = 96;
    const { view, ctl } = AI.labShell(root, 'A router sends each token to its top-k experts', 'illustrative: 96 tokens in a 2-D plane, 4 experts, a linear router that starts out lopsided; only the balancing loss is trained here',
      `<canvas style="height:380px"></canvas>`,
      `${sl('lam', 'Balancing strength (0 = off)', 0, 3, 0.1, 1)}${sl('k', 'Experts per token (top-k)', 1, 2, 1, 1)}${sl('rate', 'Gradient steps per second', 2, 40, 1, 6)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const rt = AI.rng(4), toks = [], ctr = [[-1.2, -1.1], [1.3, -1], [-1, 1.2], [1.1, 1.0]];
    for (let i = 0; i < NT; i++) { const cc = ctr[i % 4]; toks.push([cc[0] + 0.5 * AI.randn(rt), cc[1] + 0.5 * AI.randn(rt)]); }
    let Wt, bt, steps, dirty = true, tick = AI.ticker();
    const reset = () => { const r = AI.rng(8); Wt = []; for (let j = 0; j < E; j++) Wt.push([0.3 * AI.randn(r), 0.3 * AI.randn(r)]); bt = [1.6, 0.0, 0.0, -0.4]; steps = 0; dirty = true; };
    const probs = (x) => softmax(Wt.map((w, j) => w[0] * x[0] + w[1] * x[1] + bt[j]));
    const topk = (p, k) => p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((a) => a[1]);
    const stats = () => {
      const k = +q(ctl, '#k').value, P = toks.map(probs), f = new Array(E).fill(0), Pm = new Array(E).fill(0);
      P.forEach((p) => { topk(p, k).forEach((j) => f[j]++); p.forEach((v, j) => (Pm[j] += v)); });
      for (let j = 0; j < E; j++) { f[j] /= NT * k; Pm[j] /= NT; }
      return { P, f, Pm, aux: E * f.reduce((s, x, j) => s + x * Pm[j], 0), k };
    };
    const stepGD = () => {
      const s = stats(), lr = 1 * +q(ctl, '#lam').value;
      const gW = Wt.map(() => [0, 0]), gb = new Array(E).fill(0);
      s.P.forEach((p, t) => { const dotfp = s.f.reduce((a, fv, i) => a + fv * p[i], 0); for (let j = 0; j < E; j++) { const gj = (E / NT) * p[j] * (s.f[j] - dotfp); gW[j][0] += gj * toks[t][0]; gW[j][1] += gj * toks[t][1]; gb[j] += gj; } });
      for (let j = 0; j < E; j++) { Wt[j][0] -= lr * gW[j][0]; Wt[j][1] -= lr * gW[j][1]; bt[j] -= lr * gb[j]; }
      steps++; dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const s = stats(), cols = [c.blue, c.orange, c.green, c.purple], pw = Math.min(W * 0.58, H - 30), L = 10, top = 22;
      const X = (v) => L + (v + 3) / 6 * pw, Y = (v) => top + (v + 3) / 6 * (H - 40);
      txt(g, 'colour = top expert, ring = second choice', L, 14, c.muted, '11px sans-serif');
      const gs = 24;
      for (let i = 0; i < gs; i++) for (let j = 0; j < gs; j++) { const x = -3 + (i + 0.5) / gs * 6, y = -3 + (j + 0.5) / gs * 6; const e = topk(probs([x, y]), 1)[0]; const rgb = AI.rgb(cols[e]); g.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.16)`; g.fillRect(X(x) - pw / gs / 2, Y(y) - (H - 40) / gs / 2, pw / gs + 1, (H - 40) / gs + 1); }
      toks.forEach((x, t) => { const tk = topk(s.P[t], s.k); g.fillStyle = cols[tk[0]]; g.beginPath(); g.arc(X(x[0]), Y(x[1]), 4, 0, 7); g.fill(); if (tk[1] !== undefined) { g.strokeStyle = cols[tk[1]]; g.lineWidth = 1.5; g.beginPath(); g.arc(X(x[0]), Y(x[1]), 6.5, 0, 7); g.stroke(); g.lineWidth = 1; } });
      const bx = L + pw + 40, bw = (W - bx - 14) / E, by = H - 50, bh = H - 120;
      txt(g, 'f filled, P outline', bx, 14, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(bx, by - 0.25 * bh * 2); g.lineTo(bx + bw * E, by - 0.25 * bh * 2); g.stroke(); txt(g, '1/E', bx - 4, by - 0.5 * bh + 4, c.muted, '10px sans-serif', 'right');
      for (let j = 0; j < E; j++) { const x = bx + j * bw; g.fillStyle = cols[j]; g.globalAlpha = 0.85; g.fillRect(x + 4, by - s.f[j] * bh * 2, bw * 0.4, s.f[j] * bh * 2); g.globalAlpha = 1; g.strokeStyle = cols[j]; g.lineWidth = 2; g.strokeRect(x + bw * 0.5, by - s.Pm[j] * bh * 2, bw * 0.4, s.Pm[j] * bh * 2); g.lineWidth = 1; txt(g, 'E' + (j + 1), x + bw / 2, by + 14, c.muted, '11px sans-serif', 'center'); txt(g, s.f[j].toFixed(2), x + 4, by - s.f[j] * bh * 2 - 4, c.fg, '10px sans-serif'); }
      q(ctl, '#out').innerHTML = `f = ${s.f.map((x) => x.toFixed(3)).join(', ')}<br>P = ${s.Pm.map((x) => x.toFixed(3)).join(', ')}<br>balancing loss E·Σ fᵢPᵢ = <b>${s.aux.toFixed(4)}</b> (1.0000 when every expert gets exactly 1/${E}); step ${steps}`;
      q(ctl, '#cap').textContent = +q(ctl, '#lam').value === 0 ? 'Balancing is off, so nothing moves: one expert keeps most of the tokens and the others are nearly idle ("routing collapse").' : (s.aux < 1.05 ? 'The loads are close to equal: the regions have been pushed to share the plane.' : 'The gradient lowers the probability of the overloaded expert and raises the others, so the coloured regions slowly shift.');
      showVals(ctl, { lam: (v) => v, k: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, stepGD, reset, draw);
    q(ctl, '#k').oninput = () => { dirty = true; }; q(ctl, '#lam').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => { if (st.playing && steps < 3000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) stepGD(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     11. MoE total versus active parameters
     ===================================================================== */
  AI.labs['gen-ef-moe-params'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Total versus active parameters in one MoE layer', 'illustrative arithmetic: attention 4d², each expert a feed-forward block 8d² (two matrices d×4d), router d×E',
      `<canvas style="height:260px"></canvas>`,
      `${ni('d', 'd_model', 512, 8, 16384, 8)}${sl('E', 'Number of experts E', 1, 64, 1, 8)}${sl('k', 'Experts used per token k', 1, 8, 1, 2)}${sl('rate', 'Sweep speed (experts per second)', 0.5, 8, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let Es = 8, Et = 8, dirty = true;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const d = Math.max(8, +q(ctl, '#d').value || 8), E = Math.round(Es), k = Math.min(+q(ctl, '#k').value, E);
      const A = 4 * d * d, F = 8 * d * d, Rr = d * E, tot = A + E * F + Rr, act = A + k * F + Rr, dense = A + F;
      const mx = Math.max(tot, 1), L = 14, bw = W - 28;
      const row = (y, label, parts, val) => { txt(g, label, L, y - 6, c.muted, '11px sans-serif'); let x = L; parts.forEach(([v, col]) => { const w = v / mx * bw; g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x, y, w, 26); g.globalAlpha = 1; x += w; }); txt(g, fmtN(val), x + 6 > W - 90 ? W - 10 : x + 6, y + 18, c.fg, 'bold 11px sans-serif', x + 6 > W - 90 ? 'right' : 'left'); };
      row(40, 'total parameters stored (attention + router + all E experts)', [[A, c.blue], [Rr, c.yellow], [E * F, c.orange]], tot);
      row(110, 'active parameters per token (attention + router + k experts)', [[A, c.blue], [Rr, c.yellow], [k * F, c.orange]], act);
      row(180, 'a dense model with one feed-forward block (for compute comparison)', [[dense, c.blue]], dense);
      txt(g, 'blue attention · yellow router · orange experts', L, H - 8, c.muted, '11px sans-serif');
      q(ctl, '#out').innerHTML = `attention 4d² = ${fmtN(A)}; one expert 8d² = ${fmtN(F)}; router dE = ${fmtN(Rr)}<br>total = 4d² + E·8d² + dE = <b>${fmtN(tot)}</b><br>active = 4d² + k·8d² + dE = <b>${fmtN(act)}</b> (${(act / tot * 100).toFixed(1)}% of total); arithmetic per token is close to a dense model with k feed-forward blocks, but all ${fmtN(tot)} numbers must still be held in memory`;
      q(ctl, '#cap').textContent = 'Adding experts grows the orange bar of the top row (what you store) without changing the middle row (what each token uses). That is the whole point, and also the cost: memory grows with E.';
      showVals(ctl, { E: (v) => v, k: (v) => v, rate: (v) => v });
    };
    const step = () => { Et = Et + 1 > 64 ? 1 : Et + 1; Es = Et; q(ctl, '#E').value = Et; dirty = true; };
    const reset = () => { Et = Es = 8; q(ctl, '#E').value = 8; dirty = true; };
    wire(ctl, st, step, reset, draw);
    q(ctl, '#E').oninput = () => { Es = Et = +q(ctl, '#E').value; dirty = true; };
    ['#d', '#k', '#rate'].forEach((s) => (q(ctl, s).oninput = () => { dirty = true; }));
    o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing) { Et += dt * +q(ctl, '#rate').value; if (Et > 64.99) Et = 1; Es = Et; q(ctl, '#E').value = Math.round(Et); dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };
})();
