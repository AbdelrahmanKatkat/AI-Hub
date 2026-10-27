/* Module 1 (Data preparation) labs: prepipe, edaexplorer, missinglab, encodelab, scalelab, featlab, imbalancelab, leaklab, splitlab.
 * All animations are dt-based, slow by default, with speed slider + pause/step/reset and captions. Simulations are illustrative. */
(function () {
  const $ = (root, s) => root.querySelector(s);
  const FONT = '12px ui-monospace,Menlo,Consolas,monospace';
  const playCtl = (speedDefault = 1) => `<div class="row"><button class="btn primary" data-a="play">⏸ Pause</button><button class="btn" data-a="step">Step</button><button class="btn" data-a="reset">Reset</button></div>
    <label>Speed <span class="v" data-a="spv"></span><input type="range" data-a="sp" min="0.25" max="3" step="0.05" value="${speedDefault}"></label>`;
  const A = (root, n) => root.querySelector(`[data-a="${n}"]`);
  /* generic play/pause/step wiring: returns state object {playing, speed()} */
  const wire = (root, onStep, onReset) => {
    const st = { playing: true, speed: () => +A(root, 'sp').value };
    A(root, 'play').onclick = () => { st.playing = !st.playing; A(root, 'play').textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    A(root, 'step').onclick = () => { st.playing = false; A(root, 'play').textContent = '▶ Play'; onStep(); };
    A(root, 'reset').onclick = () => { onReset(); };
    A(root, 'sp').oninput = () => { A(root, 'spv').textContent = '×' + (+A(root, 'sp').value).toFixed(2); };
    A(root, 'spv').textContent = '×' + (+A(root, 'sp').value).toFixed(2);
    return st;
  };
  const rr = (g, x, y, w, h, r) => { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };
  const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
  const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN; };
  const caption = (g, w, text, y = 4) => { g.save(); g.font = '12px sans-serif'; const C = AI.col(); const tw = Math.min(w - 8, g.measureText(text).width + 16); g.fillStyle = C.bg2; g.globalAlpha = 0.93; g.fillRect(4, y, tw, 22); g.globalAlpha = 1; g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(text, 12, y + 15); g.restore(); };
  const pearson = (x, y) => { const px = [], py = []; for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) { px.push(x[i]); py.push(y[i]); } const mx = mean(px), my = mean(py); let a = 0, b = 0, c = 0; px.forEach((v, i) => { a += (v - mx) * (py[i] - my); b += (v - mx) ** 2; c += (py[i] - my) ** 2; }); return b && c ? a / Math.sqrt(b * c) : 0; };

  /* ======================= prepipe: fit -> store -> apply ======================= */
  AI.labs.prepipe = function (root) {
    const { view, ctl } = AI.labShell(root, 'Preprocessing pipeline: fit on train, store, apply', 'one row travelling through both paths',
      `<canvas style="height:350px"></canvas><div class="caption">Illustrative numbers from the worked examples of this module. Top lane = training (parameters are learned and stored). Bottom lane = inference (stored parameters are only read).</div>`,
      playCtl(1) + `<div class="readout" data-a="out"></div>`);
    const c = AI.canvas(view.querySelector('canvas'));
    const stages = [
      { n: 'raw row', train: 'income = ?   city = C', inf: 'income = ?   city = C', store: '' },
      { n: 'impute', train: 'income ← 39.5', inf: 'income ← 39.5', store: 'fill = 39.5', cap: 'median of the training income column' },
      { n: 'encode', train: 'city C → 0.333', inf: 'city C → 0.333', store: 'A:0.6 B:0.5 C:0.333', cap: 'smoothed target table (+ unknown fallback 0.5)' },
      { n: 'scale', train: 'z = −0.016', inf: 'z = −0.016', store: 'μ=40  σ=31.6', cap: 'mean & std of the training column' },
      { n: 'model', train: 'fit weights', inf: 'predict', store: 'weights w, b', cap: 'the model itself' },
    ];
    const NS = stages.length;
    let pos = 0, tgt = 0; // pos in [0, 2*NS)
    const DUR = 1.8;
    const st = wire(root, () => { tgt = Math.floor(pos + 0.001) + 1; if (tgt > 2 * NS - 0.0001) tgt = 0; pos = Math.min(pos, tgt); }, () => { pos = 0; tgt = 0; st.playing = true; A(root, 'play').textContent = '⏸ Pause'; });
    A(root, 'sp').value = 0.6; A(root, 'spv').textContent = '×0.60'; c.onresize = () => (dirty = true);
    let dirty = true, held = 0;
    AI.loop(root, (dt) => {
      const sp = st.speed();
      if (st.playing) { pos += (dt * sp) / DUR; if (pos >= 2 * NS + 0.6) pos = 0; dirty = true; }
      else if (Math.abs(pos - tgt) > 1e-3) { pos += Math.sign(tgt - pos) * Math.min(Math.abs(tgt - pos), (dt * 1.2) / DUR * 1.5); dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const m = 14, bw = Math.max(40, (w - 2 * m) / NS - 8), gap = ((w - 2 * m) - NS * bw) / (NS - 1 || 1);
      const lanes = [{ y: 66, label: 'TRAINING  (fit → store)', col: C.green, key: 'train' }, { y: 232, label: 'INFERENCE  (read stored → apply)', col: C.blue, key: 'inf' }];
      const bh = 44;
      g.font = FONT;
      lanes.forEach((L, li) => {
        g.fillStyle = L.col; g.textAlign = 'left'; g.fillText(L.label, m, L.y - 10);
        stages.forEach((s, i) => {
          const x = m + i * (bw + gap), active = Math.floor(pos) === li * NS + i;
          g.fillStyle = C.bg2; g.strokeStyle = active ? L.col : C.border; g.lineWidth = active ? 2.5 : 1;
          rr(g, x, L.y, bw, bh, 8); g.fill(); g.stroke();
          g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(s.n, x + bw / 2, L.y + 18);
          g.fillStyle = C.muted; g.font = (bw < 100 ? 8 : 10) + 'px ui-monospace,monospace'; g.fillText(li ? s.inf : s.train, x + bw / 2, L.y + 34, bw - 4); g.font = FONT;
          if (i < NS - 1) AI.arrow(g, x + bw + 1, L.y + bh / 2, x + bw + gap - 1, L.y + bh / 2, C.border, 1.5);
          // store boxes (shared between lanes): drawn in the middle band
          if (li === 0 && s.store) {
            const sy = 136, learned = pos >= i + 0.5 || pos >= NS;
            g.fillStyle = learned ? C.bg3 : 'transparent'; g.strokeStyle = learned ? C.orange : C.border; g.lineWidth = 1; g.setLineDash(learned ? [] : [4, 4]);
            rr(g, x, sy, bw, 34, 6); g.fill(); g.stroke(); g.setLineDash([]);
            g.fillStyle = learned ? C.orange : C.muted; g.font = (bw < 100 ? 8 : 10) + 'px ui-monospace,monospace'; g.fillText(learned ? s.store : 'not learned yet', x + bw / 2, sy + 21, bw - 4); g.font = FONT;
            g.fillStyle = C.muted; g.fillText('stored', x + bw / 2, sy + 48);
            // arrow down from train stage to store
            const a = learned ? 1 : 0.25; g.globalAlpha = a; AI.arrow(g, x + bw / 2, L.y + bh + 2, x + bw / 2, sy - 2, C.green, 1.5); g.globalAlpha = 1;
            // arrow from store down to inference stage (read)
            const reading = pos >= NS + i && pos < NS + i + 1; g.globalAlpha = pos >= NS ? 1 : 0.15; AI.arrow(g, x + bw / 2 + 12, sy + 52, x + bw / 2 + 12, lanes[1].y - 2, reading ? C.blue : C.border, reading ? 2.5 : 1.2); g.globalAlpha = 1;
          }
        });
      });
      // token (the row) moving along its lane
      const p = Math.min(pos, 2 * NS - 0.001), li = p >= NS ? 1 : 0, local = p - li * NS, i0 = Math.min(NS - 1, Math.floor(local)), f = AI.smooth(local - i0);
      const xa = m + i0 * (bw + gap) + bw / 2, xb = m + Math.min(NS - 1, i0 + 1) * (bw + gap) + bw / 2;
      const tx = AI.lerp(xa, xb, i0 < NS - 1 ? f : 0), ty = lanes[li].y + bh / 2;
      g.fillStyle = C.yellow; g.globalAlpha = 0.95; g.beginPath(); g.arc(tx, ty, 9, 0, 7); g.fill(); g.globalAlpha = 1;
      g.fillStyle = C.bg; g.textAlign = 'center'; g.fillText('●', tx, ty + 4);
      const s = stages[i0];
      const cap = li === 0
        ? (i0 === 0 ? '① Training: a raw row enters the pipeline.' : i0 === NS - 1 ? '⑤ Training: the model is fit on the transformed training data.' : `② Training: FIT the ${s.n} on training rows only — learn ${s.cap} — and STORE it.`)
        : (i0 === 0 ? '⑥ Inference: a brand-new row arrives. Nothing is re-learned.' : i0 === NS - 1 ? '⑩ Inference: the stored model predicts.' : `⑦ Inference: APPLY the stored ${s.n} (${s.store}) — never re-fit on this row.`);
      caption(g, w, cap);
      A(root, 'out').innerHTML = `<span class="m">phase</span> ${li ? 'inference' : 'training'}   <span class="m">stage</span> ${s.n}\n<span class="m">stored so far</span> ${stages.slice(1, NS).filter((q, k) => pos >= k + 1.5 || pos >= NS).map((q) => q.store).join('  |  ') || '–'}`;
    }, { fps: 30 });
  };

  /* ======================= EDA explorer ======================= */
  AI.labs.edaexplorer = function (root) {
    const COLS = ['age', 'tenure', 'income', 'spend', 'visits', 'churn'];
    const { view, ctl } = AI.labShell(root, 'EDA explorer', 'generated customer table (illustrative, 300 rows)',
      `<canvas data-a="h" style="height:150px"></canvas><canvas data-a="s" style="height:230px"></canvas><canvas data-a="m" style="height:290px;cursor:pointer"></canvas>
       <div class="caption">Top: histogram of x. Middle: scatter x vs y. Bottom: correlation heat-map (click a cell) and missing-value bars.</div>`,
      `<label>x column <select data-a="x">${COLS.map((k) => `<option>${k}</option>`).join('')}</select></label>
       <label>y column <select data-a="y">${COLS.map((k, i) => `<option${i === 3 ? ' selected' : ''}>${k}</option>`).join('')}</select></label>
       <label>Bins <span class="v" data-a="bv"></span><input type="range" data-a="bins" min="4" max="50" step="1" value="20"></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" data-a="log"> log transform x</label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" data-a="miss"> highlight rows with missing income</label>
       <div class="readout" data-a="out"></div>`);
    const N = 300, r = AI.rng(11), D = {};
    COLS.forEach((k) => (D[k] = []));
    const miss = [];
    for (let i = 0; i < N; i++) {
      const age = AI.clamp(40 + 12 * AI.randn(r), 18, 80), ten = AI.clamp((age - 18) * 0.45 + 3 * AI.randn(r), 0, 40);
      let inc = Math.exp(Math.log(38) + 0.55 * AI.randn(r) + 0.006 * (age - 40)); if (i % 97 === 5) inc *= 6;
      const spend = Math.max(0, 0.9 * inc + 8 * AI.randn(r) + 0.2 * ten), visits = Math.max(0, Math.round(3 + 0.1 * (50 - age) + 2 * AI.randn(r)));
      const churn = r() < AI.clamp(0.55 - 0.012 * ten + (visits < 2 ? 0.2 : 0), 0.05, 0.9) ? 1 : 0;
      const isMiss = r() < AI.clamp(0.04 + 0.012 * (age - 30), 0, 0.5);
      D.age.push(age); D.tenure.push(ten); D.income.push(isMiss ? NaN : inc); D.spend.push(spend); D.visits.push(visits); D.churn.push(churn); miss.push(isMiss);
    }
    const q = (s) => $(root, s);
    const hc = AI.canvas(q('[data-a=h]')), sc = AI.canvas(q('[data-a=s]')), mc = AI.canvas(q('[data-a=m]'));
    const corr = COLS.map((a) => COLS.map((b) => pearson(D[a], D[b])));
    let hist = null, hshow = [], ptsX = new Array(N).fill(0.5), ptsY = new Array(N).fill(0.5), tx = [], ty = [], dirty = true, sx = '', sy = '';
    const val = (k, i) => (q('[data-a=log]').checked && k === q('[data-a=x]').value ? Math.log(Math.max(D[k][i], 1e-9)) : D[k][i]);
    const recompute = () => {
      const kx = q('[data-a=x]').value, ky = q('[data-a=y]').value, nb = +q('[data-a=bins]').value; q('[data-a=bv]').textContent = nb;
      const xs = D[kx].map((_, i) => val(kx, i)), ys = D[ky];
      const fx = xs.filter(Number.isFinite), fy = ys.filter(Number.isFinite);
      const lx = Math.min(...fx), hx = Math.max(...fx), ly = Math.min(...fy), hy = Math.max(...fy);
      const h = new Array(nb).fill(0); fx.forEach((v) => h[Math.min(nb - 1, Math.floor(((v - lx) / (hx - lx || 1)) * nb))]++);
      hist = { lo: lx, hi: hx, h, mx: Math.max(...h) };
      if (hshow.length !== nb) hshow = new Array(nb).fill(0);
      tx = xs.map((v) => (Number.isFinite(v) ? (v - lx) / (hx - lx || 1) : NaN)); ty = ys.map((v) => (Number.isFinite(v) ? (v - ly) / (hy - ly || 1) : NaN));
      const s = [...fx].sort((a, b) => a - b), n = s.length, qn = (p) => { const pos = p * (n - 1), lo = Math.floor(pos); return s[lo] + (pos - lo) * ((s[Math.min(n - 1, lo + 1)] || s[lo]) - s[lo]); };
      const q1 = qn(0.25), q3 = qn(0.75), iqr = q3 - q1, out = fx.filter((v) => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length;
      const st = AI.stats(fx);
      q('[data-a=out]').innerHTML = `<span class="m">x =</span> ${kx}${q('[data-a=log]').checked ? ' (log)' : ''}   <span class="m">n</span> ${n}   <span class="m">missing</span> ${N - n}\n<span class="m">mean</span> ${AI.fmt(st.mean, 3)}   <span class="m">median</span> ${AI.fmt(qn(0.5), 3)}   <span class="m">std</span> ${AI.fmt(st.std, 3)}\n<span class="m">Q1</span> ${AI.fmt(q1, 3)}  <span class="m">Q3</span> ${AI.fmt(q3, 3)}  <span class="o">IQR outliers: ${out}</span>   <span class="m">skew</span> ${AI.fmt(st.skew, 2)}\n<span class="g">r(x,y) = ${AI.fmt(pearson(xs, ys), 3)}</span>   <span class="m">y =</span> ${ky}`;
      dirty = true;
    };
    ['x', 'y', 'bins', 'log'].forEach((n) => (q(`[data-a=${n}]`).oninput = recompute)); q('[data-a=miss]').oninput = () => (dirty = true);
    mc.cv.onclick = (e) => {
      const rect = mc.cv.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top, n = COLS.length, L = Math.min(mc.w * 0.55, mc.h - 50), cell = L / n, ox = 56, oy = 30;
      const i = Math.floor((x - ox) / cell), j = Math.floor((y - oy) / cell);
      if (i >= 0 && i < n && j >= 0 && j < n) { q('[data-a=x]').value = COLS[i]; q('[data-a=y]').value = COLS[j]; recompute(); }
    };
    recompute(); hc.onresize = sc.onresize = mc.onresize = () => (dirty = true);
    AI.loop(root, (dt) => {
      const k = 1 - Math.exp(-dt * 3);
      let moving = false;
      for (let i = 0; i < hshow.length; i++) { const t = hist.h[i] / hist.mx, d = t - hshow[i]; if (Math.abs(d) > 1e-3) { hshow[i] += d * k; moving = true; } }
      for (let i = 0; i < N; i++) { if (Number.isFinite(tx[i])) { const a = tx[i] - ptsX[i], b = ty[i] - ptsY[i]; if (Math.abs(a) > 1e-3 || Math.abs(b) > 1e-3) { ptsX[i] += a * k; ptsY[i] += b * k; moving = true; } } }
      if (!moving && !dirty) return; dirty = false;
      const C = AI.col(), kx = q('[data-a=x]').value, ky = q('[data-a=y]').value, hm = q('[data-a=miss]').checked;
      // histogram
      let g = hc.ctx; g.clearRect(0, 0, hc.w, hc.h);
      const pad = { l: 36, r: 10, t: 22, b: 22 }, pw = hc.w - pad.l - pad.r, ph = hc.h - pad.t - pad.b, nb = hshow.length, bw = pw / nb;
      g.fillStyle = C.blue; hshow.forEach((v, i) => { g.globalAlpha = 0.85; g.fillRect(pad.l + i * bw + 1, pad.t + ph - v * ph, bw - 2, v * ph); }); g.globalAlpha = 1;
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(pad.l, pad.t + ph + 0.5); g.lineTo(pad.l + pw, pad.t + ph + 0.5); g.stroke();
      g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center';
      const stp = AI.niceStep(hist.hi - hist.lo, 5); for (let t = Math.ceil(hist.lo / stp) * stp; t <= hist.hi; t += stp) g.fillText(AI.fmt(+t.toFixed(6), 2), pad.l + ((t - hist.lo) / (hist.hi - hist.lo || 1)) * pw, pad.t + ph + 14);
      g.textAlign = 'left'; g.fillStyle = C.blue; g.fillText('histogram of ' + kx + (q('[data-a=log]').checked ? ' (log)' : '') + '   (max bin = ' + hist.mx + ' rows)', pad.l, 14);
      // scatter
      g = sc.ctx; g.clearRect(0, 0, sc.w, sc.h);
      const sp = { l: 36, r: 10, t: 22, b: 22 }, spw = sc.w - sp.l - sp.r, sph = sc.h - sp.t - sp.b;
      g.strokeStyle = C.border; g.strokeRect(sp.l + 0.5, sp.t + 0.5, spw, sph);
      for (let i = 0; i < N; i++) {
        if (!Number.isFinite(tx[i])) { if (Number.isFinite(ty[i]) && hm) { g.fillStyle = C.red; g.globalAlpha = 0.8; g.fillRect(sp.l + 2, sp.t + sph - ptsY[i] * sph - 1.5, 5, 3); g.globalAlpha = 1; } continue; }
        const isM = miss[i];
        g.fillStyle = hm && isM ? C.red : D.churn[i] ? C.orange : C.blue; g.globalAlpha = hm && !isM ? 0.25 : 0.75;
        g.beginPath(); g.arc(sp.l + ptsX[i] * spw, sp.t + sph - ptsY[i] * sph, 3, 0, 7); g.fill();
      }
      g.globalAlpha = 1; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(`scatter: ${kx} (→) vs ${ky} (↑)   colour = churn` + (hm ? '   red = income missing' : ''), sp.l, 14);
      // heatmap + missing bars
      g = mc.ctx; g.clearRect(0, 0, mc.w, mc.h);
      const n = COLS.length, L = Math.min(mc.w * 0.55, mc.h - 50), cell = L / n, ox = 56, oy = 30;
      g.font = '10px ui-monospace,monospace'; g.textAlign = 'center';
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = corr[j][i], sel = COLS[i] === kx && COLS[j] === ky;
        g.fillStyle = v >= 0 ? AI.mix(C.bg2, C.orange, Math.abs(v)) : AI.mix(C.bg2, C.blue, Math.abs(v));
        g.fillRect(ox + i * cell, oy + j * cell, cell - 1, cell - 1);
        if (sel) { g.strokeStyle = C.fg; g.lineWidth = 2; g.strokeRect(ox + i * cell, oy + j * cell, cell - 1, cell - 1); g.lineWidth = 1; }
        g.fillStyle = C.fg; g.fillText(v.toFixed(2), ox + i * cell + cell / 2, oy + j * cell + cell / 2 + 3);
      }
      g.fillStyle = C.muted; COLS.forEach((k, i) => { g.textAlign = 'center'; g.fillText(k.slice(0, 6), ox + i * cell + cell / 2, oy - 6); g.textAlign = 'right'; g.fillText(k, ox - 4, oy + i * cell + cell / 2 + 3); });
      g.textAlign = 'left'; g.font = FONT; g.fillStyle = C.muted; g.fillText('correlation matrix (click)', ox, 14);
      const bx = ox + L + 28, bwid = Math.max(40, mc.w - bx - 14); g.fillText('missing %', bx, 14);
      COLS.forEach((k, i) => { const m = D[k].filter((v) => !Number.isFinite(v)).length / N, y = oy + i * cell; g.fillStyle = C.bg3; g.fillRect(bx, y + 6, bwid, cell - 14); g.fillStyle = m ? C.red : C.green; g.fillRect(bx, y + 6, bwid * Math.min(1, m * 4), cell - 14); g.fillStyle = C.fg; g.font = '10px ui-monospace,monospace'; g.fillText((m * 100).toFixed(1) + '%', bx + 4, y + cell / 2 + 3); });
    }, { fps: 30 });
  };

  /* ======================= missing-data lab ======================= */
  AI.labs.missinglab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Missing-data lab', 'gaps fill in cell by cell; the lab knows the hidden truth (illustrative simulation)',
      `<div data-a="tbl"></div><canvas data-a="c" style="height:230px"></canvas><div class="caption">Histogram: grey = true income, blue = observed, orange = imputed cells. The stored fill value(s) are learned from observed rows only.</div>`,
      `<label>Why is income missing? <select data-a="mech"><option value="mcar">MCAR — at random</option><option value="mar">MAR — older customers skip it</option><option value="mnar">MNAR — high incomes skip it</option></select></label>
       <label>Imputation <select data-a="meth"><option value="mean">mean</option><option value="median">median</option><option value="zero">constant 0</option><option value="reg">regression on age</option><option value="knn">k-NN on age (k=3)</option><option value="drop">drop rows (no fill)</option></select></label>
       <label>Fraction missing <span class="v" data-a="fv"></span><input type="range" data-a="frac" min="0.1" max="0.5" step="0.05" value="0.3"></label>
       ${playCtl(1)}<button class="btn" data-a="new">New data</button><div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), c = AI.canvas(q('[data-a=c]'));
    const N = 40; let seed = 5, rows = [], missIdx = [], filled = 0, fillVals = [], acc = 0, dirtyT = true;
    const gen = () => {
      const r = AI.rng(seed), mech = q('[data-a=mech]').value, frac = +q('[data-a=frac]').value; q('[data-a=fv]').textContent = Math.round(frac * 100) + '%';
      rows = Array.from({ length: N }, (_, i) => { const age = 20 + Math.round(45 * r()), inc = Math.max(8, 14 + 0.9 * age + 6 * AI.randn(r)), spend = 2 * inc + 10 * AI.randn(r); return { id: i + 1, age, inc, spend }; });
      const m = Math.round(frac * N), key = rows.map((o, i) => { const wgt = mech === 'mcar' ? 1 : mech === 'mar' ? Math.exp(0.12 * (o.age - 40)) : Math.exp(0.22 * (o.inc - 50)); return [Math.pow(r(), 1 / wgt), i]; }).sort((a, b) => b[0] - a[0]);
      missIdx = key.slice(0, m).map((k) => k[1]).sort((a, b) => a - b); rows.forEach((o) => (o.miss = false)); missIdx.forEach((i) => (rows[i].miss = true));
      reset();
    };
    const fit = () => {
      const obs = rows.filter((o) => !o.miss), meth = q('[data-a=meth]').value; let info = '';
      const xs = obs.map((o) => o.age), ys = obs.map((o) => o.inc), mx = mean(xs), my = mean(ys); let sxy = 0, sxx = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
      const w = sxy / sxx, b = my - w * mx;
      const f = (o) => meth === 'mean' ? my : meth === 'median' ? median(ys) : meth === 'zero' ? 0 : meth === 'reg' ? w * o.age + b : meth === 'knn' ? mean(obs.map((p) => [Math.abs(p.age - o.age), p.inc]).sort((u, v) => u[0] - v[0]).slice(0, 3).map((p) => p[1])) : NaN;
      info = meth === 'mean' ? `stored fill = ${AI.fmt(my, 2)}` : meth === 'median' ? `stored fill = ${AI.fmt(median(ys), 2)}` : meth === 'zero' ? 'stored fill = 0' : meth === 'reg' ? `stored model: income = ${AI.fmt(w, 2)}·age + ${AI.fmt(b, 2)}` : meth === 'knn' ? 'stored: the training rows (k-NN is lazy)' : 'nothing stored: rows are dropped';
      fillVals = missIdx.map((i) => f(rows[i])); return info;
    };
    let info = '';
    const reset = () => { filled = 0; acc = 0; info = fit(); dirtyT = true; };
    const stepOnce = () => { if (filled < missIdx.length) { filled++; dirtyT = true; } };
    const st = wire(root, stepOnce, reset);
    A(root, 'sp').value = 1;
    q('[data-a=new]').onclick = () => { seed += 7; gen(); };
    ['mech', 'frac'].forEach((n) => (q(`[data-a=${n}]`).onchange = gen)); q('[data-a=frac]').oninput = gen; q('[data-a=meth]').onchange = reset;
    gen(); c.onresize = () => (dirtyT = true);
    let cur = 0;
    const metrics = () => {
      const drop = q('[data-a=meth]').value === 'drop';
      const vals = [], xs = [], ys = [];
      rows.forEach((o) => { if (!o.miss) { vals.push(o.inc); xs.push(o.inc); ys.push(o.spend); } });
      let sse = 0, nf = 0; if (!drop) missIdx.forEach((i, k) => { if (k < filled) { vals.push(fillVals[k]); xs.push(fillVals[k]); ys.push(rows[i].spend); sse += (fillVals[k] - rows[i].inc) ** 2; nf++; } });
      const slope = (x, y) => { const mx = mean(x), my = mean(y); let a = 0, b = 0; x.forEach((v, i) => { a += (v - mx) * (y[i] - my); b += (v - mx) ** 2; }); return a / b; };
      return { vals, rmse: nf ? Math.sqrt(sse / nf) : NaN, m: mean(vals), s: std(vals), slope: slope(xs, ys), tm: mean(rows.map((o) => o.inc)), ts: std(rows.map((o) => o.inc)), ts2: slope(rows.map((o) => o.inc), rows.map((o) => o.spend)), nf };
    };
    AI.loop(root, (dt) => {
      if (st.playing && filled < missIdx.length) { acc += dt * st.speed() / 0.9; while (acc >= 1 && filled < missIdx.length) { acc -= 1; stepOnce(); } }
      if (!dirtyT) return; dirtyT = false;
      const C = AI.col(), drop = q('[data-a=meth]').value === 'drop', m = metrics();
      // table: first rows containing a gap (up to 6) + context
      const show = missIdx.slice(0, 5).map((i) => rows[i]);
      q('[data-a=tbl]').innerHTML = `<table class="tbl" style="font-size:12px"><tr><th>row</th><th>age</th><th>income (k$)</th><th>true (hidden)</th><th>spend</th></tr>${show.map((o) => { const k = missIdx.indexOf(o.id - 1), done = k < filled && !drop; return `<tr><td>${o.id}</td><td>${o.age}</td><td style="color:${done ? C.orange : C.red}">${done ? AI.fmt(fillVals[k], 1) : '?'}</td><td style="color:${C.muted}">${AI.fmt(o.inc, 1)}</td><td>${AI.fmt(o.spend, 1)}</td></tr>`; }).join('')}<tr><td colspan="5" style="color:${C.muted}">… ${missIdx.length} rows have a gap; filled so far: ${drop ? 0 : filled}</td></tr></table>`;
      const g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const lo = 0, hi = 130, nb = 26, pad = { l: 30, r: 10, t: 26, b: 22 }, pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b, bw = pw / nb;
      const hs = (arr) => { const h = new Array(nb).fill(0); arr.forEach((v) => { if (Number.isFinite(v)) h[AI.clamp(Math.floor(((v - lo) / (hi - lo)) * nb), 0, nb - 1)]++; }); return h; };
      const ht = hs(rows.map((o) => o.inc)), ho = hs(rows.filter((o) => !o.miss).map((o) => o.inc)), hf = hs(drop ? [] : missIdx.slice(0, filled).map((_, k) => fillVals[k])), mx = Math.max(...ht, ...ho.map((v, i) => v + hf[i]));
      ht.forEach((v, i) => { g.fillStyle = C.muted; g.globalAlpha = 0.35; g.fillRect(pad.l + i * bw, pad.t + ph - (v / mx) * ph, bw - 1, (v / mx) * ph); });
      g.globalAlpha = 0.9; ho.forEach((v, i) => { g.fillStyle = C.blue; g.fillRect(pad.l + i * bw + 2, pad.t + ph - (v / mx) * ph, bw - 5, (v / mx) * ph); });
      hf.forEach((v, i) => { g.fillStyle = C.orange; g.fillRect(pad.l + i * bw + 2, pad.t + ph - ((ho[i] + v) / mx) * ph, bw - 5, (v / mx) * ph); });
      g.globalAlpha = 1; g.fillStyle = C.muted; g.font = FONT; g.textAlign = 'center'; for (let t = 0; t <= hi; t += 20) g.fillText(t, pad.l + ((t - lo) / (hi - lo)) * pw, pad.t + ph + 14);
      g.textAlign = 'left'; g.fillText('income distribution', pad.l, 14);
      const tmx = pad.l + ((m.tm - lo) / (hi - lo)) * pw, emx = pad.l + ((m.m - lo) / (hi - lo)) * pw;
      g.strokeStyle = C.muted; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(tmx, pad.t); g.lineTo(tmx, pad.t + ph); g.stroke(); g.strokeStyle = C.green; g.beginPath(); g.moveTo(emx, pad.t); g.lineTo(emx, pad.t + ph); g.stroke(); g.setLineDash([]);
      q('[data-a=out]').innerHTML = `<span class="m">fit on observed rows:</span> ${info}\n<span class="m">mean income</span> true ${AI.fmt(m.tm, 2)} → estimated <span class="g">${AI.fmt(m.m, 2)}</span>\n<span class="m">std income</span> true ${AI.fmt(m.ts, 2)} → <span class="o">${AI.fmt(m.s, 2)}</span> (variance shrinks with constant fills)\n<span class="m">fill RMSE vs hidden truth</span> ${Number.isFinite(m.rmse) ? AI.fmt(m.rmse, 2) : '–'}   <span class="m">filled</span> ${drop ? 0 : m.nf}/${missIdx.length}\n<span class="m">slope spend~income</span> true ${AI.fmt(m.ts2, 2)} → ${AI.fmt(m.slope, 2)}`;
    }, { fps: 30 });
  };

  /* ======================= encoding lab ======================= */
  AI.labs.encodelab = function (root) {
    const MODES = [['onehot', 'one-hot'], ['ordinal', 'ordinal (A=1,B=2…)'], ['freq', 'frequency'], ['te_naive', 'target — naive'], ['te_smooth', 'target — smoothed (m=2)'], ['te_oof', 'target — leave-one-out'], ['hash', 'hashing (4 buckets)']];
    const { view, ctl } = AI.labShell(root, 'Encoding lab', 'the table rewrites itself; then test target-encoding leakage (illustrative simulation)',
      `<canvas data-a="t" style="height:290px"></canvas><canvas data-a="b" style="height:120px"></canvas><div class="caption">Top: the 8-row table (city, bought). Bottom: how many columns each encoding creates for K categories.</div>`,
      `<label>Encoding <select data-a="mode">${MODES.map((m) => `<option value="${m[0]}">${m[1]}</option>`).join('')}</select></label>
       <label>Categories K (cardinality) <span class="v" data-a="kv"></span><input type="range" data-a="k" min="0" max="100" step="1" value="12"></label>
       ${playCtl(1)}
       <div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), tc = AI.canvas(q('[data-a=t]')), bc = AI.canvas(q('[data-a=b]'));
    const city = ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C'], y = [1, 0, 1, 0, 1, 0, 1, 1], cats = ['A', 'B', 'C', 'D'], gm = mean(y);
    const stat = (c, excl) => { let n = 0, s = 0; city.forEach((v, i) => { if (v === c && i !== excl) { n++; s += y[i]; } }); return { n, s }; };
    const enc = (mode) => city.map((c, i) => {
      const f = (x) => AI.fmt(x, 3);
      if (mode === 'onehot') return cats.map((k) => (k === c ? '1' : '0'));
      if (mode === 'ordinal') return [String(cats.indexOf(c) + 1)];
      if (mode === 'freq') return [f(city.filter((v) => v === c).length / city.length)];
      if (mode === 'te_naive') { const s = stat(c, -1); return [f(s.s / s.n)]; }
      if (mode === 'te_smooth') { const s = stat(c, -1); return [f((s.s + 2 * gm) / (s.n + 2))]; }
      if (mode === 'te_oof') { const s = stat(c, i); return [s.n ? f(s.s / s.n) : f(gm) + '*']; }
      return [String((c.charCodeAt(0) * 7 + 3) % 4)];
    });
    const colNames = (mode) => (mode === 'onehot' ? cats : [mode === 'hash' ? 'bucket' : 'x_city']);
    let mode = 'onehot', cells = [], t0 = 0, time = 0, dirty = true, acc = 0, auto = true;
    const setMode = (m, anim = true) => { mode = m; cells = enc(m).map((row, r) => ({ vals: row, t: anim ? time + 0.5 + r * 0.35 / Math.max(0.3, st.speed()) : -1 })); dirty = true; };
    const st = wire(root, () => { const i = MODES.findIndex((m) => m[0] === mode); q('[data-a=mode]').value = MODES[(i + 1) % MODES.length][0]; setMode(q('[data-a=mode]').value); }, () => setMode(mode));
    A(root, 'sp').value = 1; setMode('onehot', false);
    q('[data-a=mode]').onchange = () => { auto = false; setMode(q('[data-a=mode]').value); };
    q('[data-a=k]').oninput = () => (dirty = true);
    let prevVals = enc('onehot'), prevCols = 4;
    // leakage test
    const leak = (K, kind) => {
      const r = AI.rng(31 + K), n = 60, tr = [], te = [];
      for (let i = 0; i < n; i++) tr.push([Math.floor(r() * K), r() < 0.5 ? 1 : 0]); for (let i = 0; i < n; i++) te.push([Math.floor(r() * K), r() < 0.5 ? 1 : 0]);
      const S = new Map(), G = mean(tr.map((v) => v[1])); tr.forEach(([c, l]) => { const s = S.get(c) || [0, 0]; s[0]++; s[1] += l; S.set(c, s); });
      const e = ([c, l]) => { const s = S.get(c); return kind === 'naive' ? s[1] / s[0] : kind === 'smooth' ? (s[1] + 2 * G) / (s[0] + 2) : s[0] > 1 ? (s[1] - l) / (s[0] - 1) : G; };
      const trE = tr.map((row) => e(row)), cand = [...new Set(trE)].sort((a, b) => a - b); let best = 0, bt = 0.5;
      cand.forEach((t, k) => { const t2 = k + 1 < cand.length ? (t + cand[k + 1]) / 2 : t + 1; const a = tr.filter((row, i) => (trE[i] >= t2 ? 1 : 0) === row[1]).length / n; if (a > best) { best = a; bt = t2; } });
      const teA = te.filter(([c, l]) => { const s = S.get(c), v = s ? (kind === 'naive' ? s[1] / s[0] : (s[1] + (kind === 'smooth' ? 2 * G : 0)) / (s[0] + (kind === 'smooth' ? 2 : 0))) : G; return (v >= bt ? 1 : 0) === l; }).length / n;
      return { tr: best, te: teA };
    };
    AI.loop(root, (dt) => {
      time += dt; const speed = st.speed();
      if (st.playing && auto) { acc += dt * speed; if (acc > 5) { acc = 0; const i = MODES.findIndex((m) => m[0] === mode); const nm = MODES[(i + 1) % MODES.length][0]; q('[data-a=mode]').value = nm; setMode(nm); } }
      const animating = cells.some((c) => c.t >= 0 && time < c.t + 0.6);
      if (!dirty && !animating) return; dirty = false;
      const C = AI.col(), g = tc.ctx, w = tc.w, h = tc.h; g.clearRect(0, 0, w, h); g.font = FONT;
      const names = colNames(mode), nc = names.length, cw = Math.min(90, (w - 200) / (nc + 2)), rh = 26, ox = 14, oy = 40;
      g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('city', ox + 25, oy - 8); g.fillText('bought', ox + 85, oy - 8);
      names.forEach((nm, j) => g.fillText(nm, ox + 170 + j * cw + cw / 2, oy - 8));
      cells.forEach((rw, r) => {
        const yy = oy + r * rh; g.fillStyle = C.bg2; g.fillRect(ox, yy, 50, rh - 3); g.fillRect(ox + 60, yy, 50, rh - 3);
        g.fillStyle = C.fg; g.fillText(city[r], ox + 25, yy + 17); g.fillStyle = y[r] ? C.green : C.muted; g.fillText(y[r], ox + 85, yy + 17);
        g.fillStyle = C.muted; g.fillText('→', ox + 135, yy + 17);
        rw.vals.forEach((v, j) => {
          const p = rw.t < 0 ? 1 : AI.clamp((time - rw.t) / 0.5 + 0.0, 0, 1) , flip = rw.t < 0 ? 1 : p, x = ox + 170 + j * cw;
          g.save(); g.translate(x + cw / 2 - 2, yy + (rh - 3) / 2); g.scale(1, Math.max(0.05, Math.abs(Math.cos((1 - flip) * Math.PI / 2))));
          const warm = rw.t >= 0 && time < rw.t + 1.4;
          g.fillStyle = flip < 1 || warm ? AI.mix(C.bg2, C.orange, flip < 1 ? 0.1 : 0.35) : C.bg3; g.fillRect(-cw / 2 + 2, -(rh - 3) / 2, cw - 4, rh - 3);
          g.fillStyle = (mode === 'te_naive' && city.filter((cc) => cc === city[r]).length === 1) ? C.red : C.fg; g.textAlign = 'center'; g.fillText(flip < 0.5 ? (prevVals[r] || [''])[j] ?? '' : v, 0, 4); g.restore();
        });
      });
      const info = { onehot: `${nc} columns for 4 categories; stored: category list`, ordinal: 'imposes order A<B<C<D — false for cities', freq: 'collides when counts are equal; stored: counts', te_naive: 'row 7 (D) is the only D: its encoding is its own label (red) — LEAK', te_smooth: '(n·mean + 2·0.625)/(n+2): rare categories shrink toward the global mean 0.625', te_oof: 'each row excludes itself; * = category seen once → global mean fallback', hash: 'no table to store; collisions possible (e.g. A and D?)' };
      caption(g, w, info[mode], h - 28);
      const mm = q('[data-a=mode]').value; if (mm === mode && !animating) prevVals = enc(mode);
      // cardinality bars
      const K = Math.round(Math.pow(10, 1 + (+q('[data-a=k]').value) / 100 * 4)); q('[data-a=kv]').textContent = K.toLocaleString();
      const b = bc.ctx; b.clearRect(0, 0, bc.w, bc.h);
      const bars = [['one-hot', K, C.red], ['ordinal', 1, C.blue], ['frequency', 1, C.blue], ['target', 1, C.blue], ['hash (1024)', Math.min(K, 1024), C.orange]], bx = 90, bwid = bc.w - bx - 70, L = Math.log10(100000);
      b.font = FONT; bars.forEach(([n, v, col], i) => { const y0 = 8 + i * 22; b.fillStyle = C.muted; b.textAlign = 'right'; b.fillText(n, bx - 6, y0 + 12); b.fillStyle = C.bg3; b.fillRect(bx, y0, bwid, 16); b.fillStyle = col; b.fillRect(bx, y0, Math.max(2, (Math.log10(v + 1) / L) * bwid), 16); b.fillStyle = C.fg; b.textAlign = 'left'; b.fillText(v.toLocaleString(), bx + bwid + 6, y0 + 12); });
      const lk = ['naive', 'smooth', 'oof'].map((k) => leak(Math.min(K, 40), k));
      q('[data-a=out]').innerHTML = `<span class="m">columns created for K =</span> ${K.toLocaleString()}: one-hot <span class="r">${K.toLocaleString()}</span>, others 1\n<span class="m">memory, 1M rows × one-hot (1 byte)</span> ${(K).toLocaleString()} MB\n<span class="m">Leakage test</span> (coin-flip labels, ${Math.min(K, 40)} categories, 60 train / 60 new rows):\n<span class="m">naive target enc.</span> train <span class="o">${(lk[0].tr * 100).toFixed(0)}%</span> → new rows <span class="r">${(lk[0].te * 100).toFixed(0)}%</span>\n<span class="m">smoothed</span> train ${(lk[1].tr * 100).toFixed(0)}% → new ${(lk[1].te * 100).toFixed(0)}%\n<span class="m">leave-one-out</span> train ${(lk[2].tr * 100).toFixed(0)}% → new ${(lk[2].te * 100).toFixed(0)}%\n<span class="m">truth: labels are random, honest accuracy ≈ 50%</span>`;
    }, { fps: 30 });
    tc.onresize = () => (dirty = true); bc.onresize = () => (dirty = true);
  };

  /* ======================= scaling lab ======================= */
  AI.labs.scalelab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Scaling lab: contours, gradient descent and distances', 'same learning-rate rule on both sides (illustrative simulation)',
      `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start"><canvas data-a="l" style="display:block;flex:1;min-width:200px;height:260px"></canvas><canvas data-a="r" style="display:block;flex:1;min-width:200px;height:260px"></canvas></div>
       <canvas data-a="k" style="height:210px"></canvas>
       <div class="caption">Top: loss contours in weight space for raw features (left) and standardised features (right); dots = gradient-descent path. Bottom: nearest neighbours of a query, raw vs scaled space.</div>`,
      `<label>Scale ratio s (feature 2 is s× smaller in raw units) <span class="v" data-a="sv"></span><input type="range" data-a="s" min="1" max="8" step="0.5" value="5"></label>
       <label>Learning rate = fraction of the stability limit 2/λmax <span class="v" data-a="lv"></span><input type="range" data-a="lr" min="0.1" max="0.98" step="0.02" value="0.9"></label>
       ${playCtl(1)}<div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), cL = AI.canvas(q('[data-a=l]')), cR = AI.canvas(q('[data-a=r]')), cK = AI.canvas(q('[data-a=k]'));
    const N = 80, rr0 = AI.rng(4); const F1 = [], F2 = [], Y = [];
    for (let i = 0; i < N; i++) { const a = AI.randn(rr0), b = 0.5 * a + 0.866 * AI.randn(rr0); F1.push(a); F2.push(b); Y.push(2 * a + 2 * b + 0.3 * AI.randn(rr0)); }
    const build = (s) => { // returns problem for features [f1, f2/s]
      let h11 = 0, h12 = 0, h22 = 0, c1 = 0, c2 = 0, yy = 0; for (let i = 0; i < N; i++) { const x1 = F1[i], x2 = F2[i] / s; h11 += x1 * x1; h12 += x1 * x2; h22 += x2 * x2; c1 += x1 * Y[i]; c2 += x2 * Y[i]; yy += Y[i] * Y[i]; }
      const H = [h11 / N, h12 / N, h22 / N], c = [c1 / N, c2 / N], det = H[0] * H[2] - H[1] * H[1], opt = [(H[2] * c[0] - H[1] * c[1]) / det, (H[0] * c[1] - H[1] * c[0]) / det];
      const tr = H[0] + H[2], disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), lmax = tr / 2 + disc, lmin = tr / 2 - disc;
      const L = (w) => 0.5 * (H[0] * w[0] * w[0] + 2 * H[1] * w[0] * w[1] + H[2] * w[1] * w[1]) - c[0] * w[0] - c[1] * w[1] + 0.5 * yy / N;
      return { H, c, opt, lmax, lmin, L, Lmin: L(opt) };
    };
    let P = {}, S = {}, path = { raw: [], sc: [] }, steps = 0, acc = 0, done = { raw: -1, sc: -1 }, imgs = {}, dirty = true;
    const view2 = (cv, key) => {
      const o = key === 'raw' ? P.raw.opt : P.sc.opt; const e = key === 'raw' ? [-1.2, 4.2, -2.5, Math.max(4.2, o[1] * 1.35 + 2)] : [-1, 3.5, -1, 3.5];
      return { x0: e[0], x1: e[1], y0: e[2], y1: e[3] };
    };
    const reset = () => {
      const s = +q('[data-a=s]').value; q('[data-a=sv]').textContent = '×' + s; const lrf = +q('[data-a=lr]').value; q('[data-a=lv]').textContent = lrf.toFixed(2) + ' × limit';
      P.raw = build(s); P.sc = build(1.0000001);
      // standardised: feature 2 scaled to unit std
      const sd2 = std(F2); let h = [0, 0, 0], cc = [0, 0], yy = 0; for (let i = 0; i < N; i++) { const x2 = F2[i] / sd2; h[0] += F1[i] * F1[i]; h[1] += F1[i] * x2; h[2] += x2 * x2; cc[0] += F1[i] * Y[i]; cc[1] += x2 * Y[i]; yy += Y[i] * Y[i]; }
      const H = [h[0] / N, h[1] / N, h[2] / N], c = [cc[0] / N, cc[1] / N], det = H[0] * H[2] - H[1] * H[1], opt = [(H[2] * c[0] - H[1] * c[1]) / det, (H[0] * c[1] - H[1] * c[0]) / det], tr = H[0] + H[2], disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      P.sc = { H, c, opt, lmax: tr / 2 + disc, lmin: tr / 2 - disc, L: (w) => 0.5 * (H[0] * w[0] * w[0] + 2 * H[1] * w[0] * w[1] + H[2] * w[1] * w[1]) - c[0] * w[0] - c[1] * w[1] + 0.5 * yy / N }; P.sc.Lmin = P.sc.L(opt);
      // raw model uses feature2 = F2/s with s relative to std: ratio of units = s (std of raw f2 ~ 1/s)
      P.lr = { raw: lrf * 2 / P.raw.lmax, sc: lrf * 2 / P.sc.lmax };
      S = { raw: [0, 0], sc: [0, 0] }; path = { raw: [[0, 0]], sc: [[0, 0]] }; steps = 0; acc = 0; done = { raw: -1, sc: -1 }; imgs = {}; dirty = true; knnPrep();
    };
    const stepOnce = () => {
      ['raw', 'sc'].forEach((k) => { const p = P[k], w = S[k], g0 = p.H[0] * w[0] + p.H[1] * w[1] - p.c[0], g1 = p.H[1] * w[0] + p.H[2] * w[1] - p.c[1]; w[0] -= P.lr[k] * g0; w[1] -= P.lr[k] * g1; path[k].push([w[0], w[1]]); if (path[k].length > 400) path[k].shift(); if (done[k] < 0 && p.L(w) - p.Lmin < 1e-3 * (p.L([0, 0]) - p.Lmin)) done[k] = steps + 1; });
      steps++; dirty = true;
    };
    const st = wire(root, stepOnce, reset); A(root, 'sp').value = 1;
    ['s', 'lr'].forEach((n) => (q(`[data-a=${n}]`).oninput = reset));
    const imageFor = (key, cv) => {
      if (imgs[key] && imgs[key].w === cv.w && imgs[key].h === cv.h) return imgs[key].c;
      const C = AI.col(), v = view2(cv, key), res = 90, off = document.createElement('canvas'); off.width = off.height = res; const g = off.getContext('2d'), im = g.createImageData(res, res), p = P[key], bg = AI.rgb(C.bg2), fg = AI.rgb(C.blue), L0 = p.L([0, 0]) - p.Lmin;
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) { const w0 = v.x0 + ((i + 0.5) / res) * (v.x1 - v.x0), w1 = v.y1 - ((j + 0.5) / res) * (v.y1 - v.y0), d = Math.max(0, p.L([w0, w1]) - p.Lmin) / L0, band = 0.5 + 0.5 * Math.cos(Math.log(d + 1e-4) * 3.2), a = 0.15 + 0.5 * band * Math.max(0, 1 - Math.log10(d + 1) * 0.9); for (let k = 0; k < 3; k++) im.data[(j * res + i) * 4 + k] = bg[k] + (fg[k] - bg[k]) * a; im.data[(j * res + i) * 4 + 3] = 255; }
      g.putImageData(im, 0, 0); imgs[key] = { c: off, w: cv.w, h: cv.h }; return off;
    };
    // kNN panel
    let kd = null, qi = 0, qt = 0;
    const knnPrep = () => {
      const s = +q('[data-a=s]').value, r = AI.rng(9), n = 60, X = [];
      for (let i = 0; i < n; i++) { const c = i % 2; X.push({ a: (c ? 1 : -1) * 1 + 0.9 * AI.randn(r), b: 3 * s * AI.randn(r), c }); }
      const sa = std(X.map((p) => p.a)), sb = std(X.map((p) => p.b)), loo = (sc) => { let ok = 0; X.forEach((p, i) => { let bd = 1e18, bj = 0; X.forEach((o, j) => { if (j === i) return; const d = ((p.a - o.a) / (sc ? sa : 1)) ** 2 + ((p.b - o.b) / (sc ? sb : 1)) ** 2; if (d < bd) { bd = d; bj = j; } }); if (X[bj].c === p.c) ok++; }); return ok / n; };
      kd = { X, sa, sb, accRaw: loo(false), accSc: loo(true), s }; dirty = true;
    };
    reset();
    AI.loop(root, (dt) => {
      qt += dt; if (qt > 3.2 / st.speed()) { qt = 0; qi = (qi + 1) % 12; dirty = true; }
      if (st.playing) { acc += dt * 8 * st.speed(); const n = Math.min(10, Math.floor(acc)); acc -= n; for (let i = 0; i < n; i++) stepOnce(); }
      if (!dirty) return; dirty = false;
      const C = AI.col();
      [['raw', cL, 'RAW features'], ['sc', cR, 'STANDARDISED']].forEach(([key, cv, title]) => {
        const g = cv.ctx; g.clearRect(0, 0, cv.w, cv.h); const v = view2(cv, key); g.drawImage(imageFor(key, cv), 0, 0, cv.w, cv.h);
        const X = (x) => ((x - v.x0) / (v.x1 - v.x0)) * cv.w, Y = (y) => (1 - (y - v.y0) / (v.y1 - v.y0)) * cv.h, pa = path[key];
        g.strokeStyle = C.orange; g.lineWidth = 1.5; g.beginPath(); pa.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.stroke();
        const last = pa[pa.length - 1]; g.fillStyle = C.orange; g.beginPath(); g.arc(X(last[0]), Y(last[1]), 4, 0, 7); g.fill();
        const o = P[key].opt; g.fillStyle = C.green; g.beginPath(); g.arc(X(o[0]), Y(o[1]), 5, 0, 7); g.fill();
        g.font = FONT; g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(title + ' — axes: w₁ → , w₂ ↑ (equal units)', 6, 14);
        caption(g, cv.w, key === 'raw' ? 'steep across the valley → zig-zag; flat along it → crawl' : 'round bowl → straight to the minimum', cv.h - 26);
      });
      const kg = cK.ctx, g = kg; g.clearRect(0, 0, cK.w, cK.h);
      const half = cK.w / 2;
      [[0, false, 'RAW distances (b axis drawn compressed)'], [half, true, 'SCALED distances']].forEach(([ox, sc, label]) => {
        const sa = sc ? kd.sa : 1, sb = sc ? kd.sb : 1, rawW = half - 40, bsc = 3 * kd.s;
        const px = (p) => ox + 20 + AI.clamp(p.a / 1.35 / 3 + 0.5, 0.02, 0.98) * rawW, py = (p) => 28 + (1 - AI.clamp(p.b / bsc / 6 + 0.5, 0.02, 0.98)) * (cK.h - 56);
        g.fillStyle = C.fg; g.font = FONT; g.textAlign = 'left'; g.fillText(label, ox + 8, 14);
        const qp = kd.X[qi], dd = kd.X.map((p, j) => [((p.a - qp.a) / sa) ** 2 + ((p.b - qp.b) / sb) ** 2, j]).sort((u, v) => u[0] - v[0]).slice(1, 4);
        kd.X.forEach((p) => { g.fillStyle = p.c ? C.orange : C.blue; g.globalAlpha = 0.7; g.beginPath(); g.arc(px(p), py(p), 3, 0, 7); g.fill(); }); g.globalAlpha = 1;
        dd.forEach(([, j]) => { g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px(qp), py(qp)); g.lineTo(px(kd.X[j]), py(kd.X[j])); g.stroke(); });
        g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.arc(px(qp), py(qp), 6, 0, 7); g.stroke(); g.lineWidth = 1;
        const votes = dd.map(([, j]) => kd.X[j].c), correct = (votes.filter((v) => v === qp.c).length >= 2);
        g.fillStyle = correct ? C.green : C.red; g.fillText(`3-NN vote ${correct ? 'correct' : 'WRONG'}   nearest: Δa=${AI.fmt(Math.abs(kd.X[dd[0][1]].a - qp.a), 2)}, Δb=${AI.fmt(Math.abs(kd.X[dd[0][1]].b - qp.b), 2)}`, ox + 8, cK.h - 8);
      });
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(half, 0); g.lineTo(half, cK.h); g.stroke();
      const f = (n) => (n < 0 ? 'not yet' : n + ' steps');
      const gap = (k) => AI.fmt(P[k].L(S[k]) - P[k].Lmin, 3);
      q('[data-a=out]').innerHTML = `<span class="m">condition number λmax/λmin</span> raw <span class="r">${AI.fmt(P.raw.lmax / P.raw.lmin, 1)}</span>   scaled <span class="g">${AI.fmt(P.sc.lmax / P.sc.lmin, 1)}</span>\n<span class="m">step</span> ${steps}   <span class="m">loss gap</span> raw ${gap('raw')}   scaled ${gap('sc')}\n<span class="m">to reach 0.1% of start gap:</span> raw ${f(done.raw)}, scaled ${f(done.sc)}\n<span class="m">leave-one-out 1-NN accuracy (feature 1 informative, feature 2 noise):</span> raw <span class="r">${(kd.accRaw * 100).toFixed(0)}%</span> → scaled <span class="g">${(kd.accSc * 100).toFixed(0)}%</span>`;
    }, { fps: 30 });
    [cL, cR, cK].forEach((c) => (c.onresize = () => { imgs = {}; dirty = true; }));
  };

  /* ======================= feature-engineering lab ======================= */
  AI.labs.featlab = function (root) {
    const FEATS = [['x', 'x'], ['y', 'y'], ['x2', 'x²'], ['y2', 'y²'], ['xy', 'x·y'], ['bins', 'bins of x,y (4 each)'], ['noise', '6 noise columns']];
    const { view, ctl } = AI.labShell(root, 'Feature-engineering lab', 'a linear model (logistic regression) trained live on the columns you tick (illustrative)',
      `<canvas data-a="c" style="height:360px"></canvas><div class="caption">Colour = model probability. Filled dots = training points, squares = held-out test points. The model is always linear in the ticked columns.</div>`,
      `<label>Dataset <select data-a="ds"><option value="circles">circles</option><option value="xor">xor</option><option value="moons">moons</option><option value="checker">checker (hard)</option></select></label>
       <div class="chips" data-a="fs">${FEATS.map((f, i) => `<label class="chip" style="cursor:pointer"><input type="checkbox" data-f="${f[0]}" ${i < 2 ? 'checked' : ''}> ${f[1]}</label>`).join('')}</div>
       ${playCtl(1)}<div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), cv = AI.canvas(q('[data-a=c]'));
    let D = null, tr = [], te = [], teSet = new Set(), noise = [], W = [], b = 0, mu = [], sd = [], Xtr = [], Xte = [], steps = 0, dirty = true;
    const on = () => FEATS.filter((f) => q(`[data-f=${f[0]}]`).checked).map((f) => f[0]);
    const phi = (x, y, nz, fl) => { const o = []; if (fl.includes('x')) o.push(x); if (fl.includes('y')) o.push(y); if (fl.includes('x2')) o.push(x * x); if (fl.includes('y2')) o.push(y * y); if (fl.includes('xy')) o.push(x * y); if (fl.includes('bins')) { for (const v of [x, y]) for (let k = 0; k < 4; k++) o.push(v >= -1.3 + k * 0.65 && v < -1.3 + (k + 1) * 0.65 ? 1 : 0); } if (fl.includes('noise')) for (let k = 0; k < 6; k++) o.push(nz ? nz[k] : 0); return o; };
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const rebuild = () => {
      const fl = on(); const raw = tr.map((i) => phi(D.X[i][0], D.X[i][1], noise[i], fl)), m = raw.length && raw[0] ? raw[0].length : 0;
      mu = []; sd = []; for (let j = 0; j < m; j++) { const col = raw.map((r2) => r2[j]); mu.push(mean(col)); sd.push(std(col) || 1); }
      Xtr = raw.map((r2) => r2.map((v, j) => (v - mu[j]) / sd[j])); Xte = te.map((i) => phi(D.X[i][0], D.X[i][1], noise[i], fl).map((v, j) => (v - mu[j]) / sd[j]));
      W = new Array(m).fill(0); b = 0; steps = 0; dirty = true;
    };
    const setup = () => {
      const kind = q('[data-a=ds]').value; D = ML.data(kind, 260, 3, { noise: 0.07 }); const r = AI.rng(5);
      noise = D.X.map(() => Array.from({ length: 6 }, () => AI.randn(r)));
      const idx = D.X.map((_, i) => i).sort((a, b2) => ((a * 7919) % 263) - ((b2 * 7919) % 263)); tr = idx.slice(0, 180); te = idx.slice(180); teSet = new Set(te);
      rebuild();
    };
    const prob = (x) => sig(x.reduce((s, v, j) => s + W[j] * v, b));
    const stepOnce = () => { const m = W.length; if (!m) return; const g = new Array(m).fill(0); let gb = 0; Xtr.forEach((x, i) => { const e = prob(x) - D.y[tr[i]]; for (let j = 0; j < m; j++) g[j] += e * x[j]; gb += e; }); for (let j = 0; j < m; j++) W[j] -= 0.8 * (g[j] / Xtr.length + 0.002 * W[j]); b -= (0.8 * gb) / Xtr.length; steps++; dirty = true; };
    const st = wire(root, stepOnce, rebuild); A(root, 'sp').value = 1;
    q('[data-a=ds]').onchange = setup; q('[data-a=fs]').onchange = rebuild; setup();
    const accOf = (X, idx) => X.filter((x, i) => (prob(x) > 0.5 ? 1 : 0) === D.y[idx[i]]).length / (X.length || 1);
    const tk = AI.ticker();
    AI.loop(root, (dt) => {
      if (st.playing && steps < 4000) { const n = tk(dt, 12 * st.speed()); for (let i = 0; i < n; i++) stepOnce(); }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h, fl = on(); g.clearRect(0, 0, w, h);
      if (W.length) ML.boundary(g, w, h, (x, y) => { const p = prob(phi(x, y, null, fl).map((v, j) => (v - mu[j]) / sd[j])); return [1 - p, p]; }, { range: 1.3, res: 52 });
      const m = ML.toPx(w, h, 1.3), cols = ML.COLORS(2);
      D.X.forEach((p, i) => { g.fillStyle = cols[D.y[i]]; g.strokeStyle = C.fg; g.lineWidth = 1; if (teSet.has(i)) { g.fillRect(m.X(p[0]) - 3.5, m.Y(p[1]) - 3.5, 7, 7); g.strokeRect(m.X(p[0]) - 3.5, m.Y(p[1]) - 3.5, 7, 7); } else { g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), 3.5, 0, 7); g.fill(); g.globalAlpha = 0.6; g.stroke(); g.globalAlpha = 1; } });
      caption(g, w, W.length ? `step ${steps}: gradient step on ${W.length} input column(s) — the boundary moves toward a better fit` : 'tick at least one column');
      const ta = W.length ? accOf(Xtr, tr) : 0, ea = W.length ? accOf(Xte, te) : 0;
      q('[data-a=out]').innerHTML = `<span class="m">columns</span> ${on().join(', ') || '–'} → ${W.length} inputs\n<span class="m">train accuracy</span> <span class="g">${(ta * 100).toFixed(0)}%</span>   <span class="m">test accuracy</span> <span class="o">${(ea * 100).toFixed(0)}%</span>\n<span class="m">step</span> ${steps}   (columns standardised with train mean/std — stored)`;
    }, { fps: 20 });
    cv.onresize = () => (dirty = true);
  };

  /* ======================= imbalance lab ======================= */
  AI.labs.imbalancelab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Imbalanced-data lab', 'weighted logistic regression; resampling animates (illustrative simulation)',
      `<canvas data-a="c" style="height:330px"></canvas><canvas data-a="m" style="height:90px"></canvas><div class="caption">Orange = minority (positive), blue = majority. White line = decision threshold. Hollow orange = duplicated or synthetic points.</div>`,
      `<label>Minority share <span class="v" data-a="shv"></span><input type="range" data-a="sh" min="0.02" max="0.5" step="0.01" value="0.1"></label>
       <label>Remedy <select data-a="m"><option value="none">none</option><option value="cw">class weights</option><option value="over">random oversampling</option><option value="under">undersampling</option><option value="smote">SMOTE</option></select></label>
       <label>Decision threshold <span class="v" data-a="thv"></span><input type="range" data-a="th" min="0.05" max="0.95" step="0.01" value="0.5"></label>
       ${playCtl(1)}<div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), cv = AI.canvas(q('[data-a=c]')), mc = AI.canvas(q('[data-a=m]')), R = 1.3, NT = 400;
    let base = [], syn = [], shown = 0, kept = [], W = [0, 0], b = 0, steps = 0, dirty = true, test = []; const tk = AI.ticker();
    const sig = (z) => 1 / (1 + Math.exp(-z)), draw = (r, c) => (c ? [0.45 + 0.3 * AI.randn(r), 0.3 + 0.3 * AI.randn(r)] : [-0.35 + 0.4 * AI.randn(r), -0.2 + 0.4 * AI.randn(r)]);
    const resample = () => {
      const m = q('[data-a=m]').value, r = AI.rng(5), mins = base.filter((o) => o.c), majs = base.filter((o) => !o.c); syn = []; kept = base.slice();
      if (m === 'over') for (let i = 0; i < majs.length - mins.length; i++) syn.push({ p: mins[Math.floor(r() * mins.length)].p, c: 1 });
      if (m === 'under') kept = mins.concat(majs.slice().sort(() => r() - 0.5).slice(0, mins.length));
      if (m === 'smote') for (let i = 0; i < majs.length - mins.length; i++) { const a = mins[Math.floor(r() * mins.length)], nn = mins.map((o) => [(o.p[0] - a.p[0]) ** 2 + (o.p[1] - a.p[1]) ** 2, o]).sort((u, v) => u[0] - v[0]).slice(1, 6), bb = nn.length ? nn[Math.floor(r() * nn.length)][1] : a, l = r(); syn.push({ p: [a.p[0] + l * (bb.p[0] - a.p[0]), a.p[1] + l * (bb.p[1] - a.p[1])], c: 1, from: a.p, to: bb.p }); }
      shown = 0; W = [0, 0]; b = 0; steps = 0; dirty = true;
    };
    const build = () => {
      const sh = +q('[data-a=sh]').value, r = AI.rng(21), nm = Math.max(4, Math.round(NT * sh)); q('[data-a=shv]').textContent = Math.round(sh * 100) + '%';
      base = []; for (let i = 0; i < NT - nm; i++) base.push({ p: draw(r, 0), c: 0 }); for (let i = 0; i < nm; i++) base.push({ p: draw(r, 1), c: 1 });
      const rt = AI.rng(77); test = []; for (let i = 0; i < 3000; i++) { const c = rt() < sh ? 1 : 0; test.push({ p: draw(rt, c), c }); }
      resample();
    };
    const wts = () => { if (q('[data-a=m]').value !== 'cw') return [1, 1]; const n1 = base.filter((o) => o.c).length, n0 = base.length - n1; return [base.length / (2 * n0), base.length / (2 * n1)]; };
    const stepOnce = () => {
      const tset = kept.concat(syn.slice(0, Math.floor(shown))), wc = wts(); let g0 = 0, g1 = 0, gb = 0, sw = 0;
      tset.forEach((o) => { const e = (sig(W[0] * o.p[0] + W[1] * o.p[1] + b) - o.c) * wc[o.c]; g0 += e * o.p[0]; g1 += e * o.p[1]; gb += e; sw += wc[o.c]; });
      const lr = 1.2; W[0] -= (lr * g0) / sw; W[1] -= (lr * g1) / sw; b -= (lr * gb) / sw; steps++; dirty = true;
    };
    const st = wire(root, stepOnce, resample); A(root, 'sp').value = 1;
    q('[data-a=sh]').oninput = build; q('[data-a=m]').onchange = resample; q('[data-a=th]').oninput = () => (dirty = true);
    build();
    AI.loop(root, (dt) => {
      if (st.playing) { if (shown < syn.length) { shown = Math.min(syn.length, shown + dt * 10 * st.speed()); dirty = true; } const n = tk(dt, 25 * st.speed()); for (let i = 0; i < n && steps < 3000; i++) stepOnce(); }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h, th = +q('[data-a=th]').value; q('[data-a=thv]').textContent = th.toFixed(2);
      g.clearRect(0, 0, w, h); ML.boundary(g, w, h, (x, y) => { const p = sig(W[0] * x + W[1] * y + b); return [1 - p, p]; }, { range: R, res: 44 });
      const m = ML.toPx(w, h, R), cols = ML.COLORS(2), meth = q('[data-a=m]').value;
      if (meth === 'under') { const ks = new Set(kept); base.forEach((o) => { if (ks.has(o)) return; g.globalAlpha = 0.15; g.fillStyle = cols[0]; g.beginPath(); g.arc(m.X(o.p[0]), m.Y(o.p[1]), 3, 0, 7); g.fill(); }); g.globalAlpha = 1; }
      kept.forEach((o) => { g.fillStyle = cols[o.c]; g.strokeStyle = C.fg; g.globalAlpha = 0.85; g.beginPath(); g.arc(m.X(o.p[0]), m.Y(o.p[1]), o.c ? 4 : 3, 0, 7); g.fill(); g.globalAlpha = 0.5; g.stroke(); g.globalAlpha = 1; });
      const ns = Math.floor(shown);
      syn.slice(0, ns).forEach((o) => { g.strokeStyle = cols[1]; g.lineWidth = 1.5; g.beginPath(); g.arc(m.X(o.p[0]) + (o.from ? 0 : 3), m.Y(o.p[1]) - (o.from ? 0 : 3), 3.5, 0, 7); g.stroke(); });
      if (meth === 'smote' && ns < syn.length) { const o = syn[ns]; g.strokeStyle = C.yellow; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(m.X(o.from[0]), m.Y(o.from[1])); g.lineTo(m.X(o.to[0]), m.Y(o.to[1])); g.stroke(); g.setLineDash([]); g.fillStyle = C.yellow; g.beginPath(); g.arc(m.X(o.p[0]), m.Y(o.p[1]), 4, 0, 7); g.fill(); }
      const lg = Math.log(th / (1 - th)), pts = []; if (Math.abs(W[1]) > 1e-6) { [-R, R].forEach((x) => pts.push([x, (lg - b - W[0] * x) / W[1]])); } else if (Math.abs(W[0]) > 1e-6) { [-R, R].forEach((y) => pts.push([(lg - b - W[1] * y) / W[0], y])); }
      if (pts.length) { g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.beginPath(); g.moveTo(m.X(pts[0][0]), m.Y(pts[0][1])); g.lineTo(m.X(pts[1][0]), m.Y(pts[1][1])); g.stroke(); }
      caption(g, w, meth === 'smote' && ns < syn.length ? 'SMOTE: pick a minority point, a minority neighbour, then a point part-way between (yellow)' : meth === 'cw' ? 'class weights: each minority error counts N/(K·n₁) times more' : meth === 'under' ? 'undersampling: most majority points (faded) are not used' : `step ${steps}: the boundary settles; minority share ${Math.round(+q('[data-a=sh]').value * 100)}%`);
      let tp = 0, fp = 0, fn = 0, tn = 0; test.forEach((o) => { const pr = sig(W[0] * o.p[0] + W[1] * o.p[1] + b) >= th ? 1 : 0; if (o.c) { pr ? tp++ : fn++; } else { pr ? fp++ : tn++; } });
      const accv = (tp + tn) / test.length, P = tp + fp ? tp / (tp + fp) : 0, Rc = tp + fn ? tp / (tp + fn) : 0, F = P + Rc ? (2 * P * Rc) / (P + Rc) : 0, base0 = (tn + fp) / test.length;
      const bm = mc.ctx, bw = mc.w; bm.clearRect(0, 0, bw, mc.h); bm.font = FONT;
      [['accuracy', accv, C.blue], ['always-neg', base0, C.muted], ['precision', P, C.green], ['recall', Rc, C.orange], ['F1', F, C.purple]].forEach(([n, v, col], i) => { const x0 = 8 + (i * (bw - 16)) / 5, ww = (bw - 16) / 5 - 8; bm.fillStyle = C.bg3; bm.fillRect(x0, 20, ww, 46); bm.fillStyle = col; bm.fillRect(x0, 66 - v * 46, ww, v * 46); bm.fillStyle = C.fg; bm.textAlign = 'center'; bm.fillText((v * 100).toFixed(0) + '%', x0 + ww / 2, 14); bm.fillStyle = C.muted; bm.font = '10px ui-monospace,monospace'; bm.fillText(n, x0 + ww / 2, 82); bm.font = FONT; });
      q('[data-a=out]').innerHTML = `<span class="m">test set (natural ratio, 3000 rows)</span>  TP ${tp}  FN ${fn}  FP ${fp}  TN ${tn}\n<span class="m">accuracy</span> ${(accv * 100).toFixed(1)}%  vs always-negative <span class="r">${(base0 * 100).toFixed(1)}%</span>\n<span class="m">precision</span> ${(P * 100).toFixed(0)}%  <span class="m">recall</span> <span class="o">${(Rc * 100).toFixed(0)}%</span>  <span class="m">F1</span> ${(F * 100).toFixed(0)}%`;
    }, { fps: 20 });
    cv.onresize = () => (dirty = true); mc.onresize = () => (dirty = true);
  };

  /* ======================= leakage lab ======================= */
  AI.labs.leaklab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Leakage lab: honest vs leaky pipeline', 'pure-noise features, random labels: the true accuracy is 50% (illustrative simulation)',
      `<canvas data-a="c" style="height:290px"></canvas><canvas data-a="h" style="height:110px"></canvas><div class="caption">Top: the pipeline stages; red arrow = test rows influencing a fitted step. Bottom: score of every run (dots) and their running average.</div>`,
      `<label>Pipeline <select data-a="mode"><option value="honest">honest: split → select → fit</option><option value="leakSel">leaky: select features on ALL data, then split</option><option value="leakScale">leaky: scale on ALL data, then split (mild)</option></select></label>
       <label>Noise features p <span class="v" data-a="pv"></span><input type="range" data-a="p" min="10" max="1000" step="10" value="300"></label>
       <label>Rows n <span class="v" data-a="nv"></span><input type="range" data-a="n" min="30" max="120" step="10" value="60"></label>
       ${playCtl(1)}<button class="btn" data-a="b30">Run 30 now</button><div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), cv = AI.canvas(q('[data-a=c]')), hc = AI.canvas(q('[data-a=h]'));
    let scores = [], pos = 0, seed = 1, dirty = true; const K = 10;
    const once = (mode, n, p, sd) => {
      const r = AI.rng(sd), X = [], y = []; for (let i = 0; i < n; i++) { X.push(Array.from({ length: p }, () => AI.randn(r))); y.push(i % 2); }
      const idx = [...Array(n).keys()].sort(() => r() - 0.5), ntr = Math.round(n * 0.7), tr = idx.slice(0, ntr), te = idx.slice(ntr), all = [...Array(n).keys()];
      const sel = (rows) => { const sc = []; for (let j = 0; j < p; j++) { let m1 = 0, m0 = 0, c1 = 0, c0 = 0; rows.forEach((i) => { if (y[i]) { m1 += X[i][j]; c1++; } else { m0 += X[i][j]; c0++; } }); sc.push([Math.abs(m1 / c1 - m0 / c0), j]); } return sc.sort((a, b) => b[0] - a[0]).slice(0, K).map((a) => a[1]); };
      if (mode === 'leakScale') { for (let j = 0; j < p; j++) { const col = X.map((rw) => rw[j]), mu = mean(col), s = std(col) || 1; X.forEach((rw) => (rw[j] = (rw[j] - mu) / s)); } }
      const cols = mode === 'leakSel' ? sel(all) : sel(tr);
      const cen = [0, 1].map((c) => cols.map((j) => mean(tr.filter((i) => y[i] === c).map((i) => X[i][j]))));
      return te.filter((i) => { const d = [0, 1].map((c) => cols.reduce((s, j, k) => s + (X[i][j] - cen[c][k]) ** 2, 0)); return (d[1] < d[0] ? 1 : 0) === y[i]; }).length / te.length;
    };
    const params = () => { const p = +q('[data-a=p]').value, n = +q('[data-a=n]').value; q('[data-a=pv]').textContent = p; q('[data-a=nv]').textContent = n; return [q('[data-a=mode]').value, n, p]; };
    const reset = () => { scores = []; pos = 0; seed = 1; dirty = true; params(); };
    const st = wire(root, () => { const [m, n, p] = params(); scores.push(once(m, n, p, seed++)); dirty = true; }, reset); A(root, 'sp').value = 1;
    ['mode', 'p', 'n'].forEach((k) => (q(`[data-a=${k}]`).oninput = reset)); q('[data-a=b30]').onclick = () => { const [m, n, p] = params(); for (let i = 0; i < 30; i++) scores.push(once(m, n, p, seed++)); dirty = true; };
    params();
    const DUR = 3.2;
    AI.loop(root, (dt) => {
      if (st.playing) { pos += (dt * st.speed()) / DUR; dirty = true; if (pos >= 1) { pos = 0; const [m, n, p] = params(); scores.push(once(m, n, p, seed++)); } }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h, mode = q('[data-a=mode]').value; g.clearRect(0, 0, w, h); g.font = FONT;
      const order = mode === 'leakSel' ? ['data', 'select top-10', 'split', 'fit model', 'score on test'] : mode === 'leakScale' ? ['data', 'scale (fit)', 'split', 'fit model', 'score on test'] : ['data', 'split', 'select (train)', 'fit model', 'score on test'];
      const bw = Math.max(70, (w - 20) / 5 - 14), gap = (w - 20 - 5 * bw) / 4, y0 = 110, bh = 50, cur = Math.min(4.999, pos * 5), ti = Math.floor(cur), splitIdx = order.indexOf('split');
      order.forEach((nm, i) => {
        const x = 10 + i * (bw + gap), active = ti === i, learns = /select|scale|fit/.test(nm);
        g.fillStyle = C.bg2; g.strokeStyle = active ? C.yellow : C.border; g.lineWidth = active ? 3 : 1; rr(g, x, y0, bw, bh, 8); g.fill(); g.stroke();
        g.fillStyle = C.fg; g.textAlign = 'center'; g.font = '11px ui-monospace,monospace'; g.fillText(nm, x + bw / 2, y0 + 22); g.fillStyle = C.muted; g.font = '10px ui-monospace,monospace'; g.fillText(learns ? '(learns params)' : nm === 'split' ? '70% / 30%' : '', x + bw / 2, y0 + 38); g.font = FONT;
        if (i < 4) AI.arrow(g, x + bw + 1, y0 + bh / 2, x + bw + gap - 1, y0 + bh / 2, C.border, 1.5);
      });
      const xl = (i) => 10 + i * (bw + gap) + bw / 2;
      if (mode !== 'honest') { g.strokeStyle = C.red; g.lineWidth = 2.5; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(xl(4), y0 + bh + 6); g.bezierCurveTo(xl(4), y0 + bh + 70, xl(1), y0 + bh + 70, xl(1), y0 + bh + 4); g.stroke(); g.setLineDash([]); g.fillStyle = C.red; g.textAlign = 'center'; g.fillText(mode === 'leakSel' ? 'LEAK: test labels helped choose the features' : 'mild leak: test rows shaped mean/std', (xl(1) + xl(4)) / 2, y0 + bh + 88); }
      else { g.fillStyle = C.green; g.textAlign = 'center'; g.fillText('no arrow from test rows into any fitted step', w / 2, y0 + bh + 88); }
      const tf = cur - ti, x1 = xl(ti), x2 = xl(Math.min(4, ti + 1)); g.fillStyle = C.yellow; g.beginPath(); g.arc(AI.lerp(x1, x2, AI.smooth(tf)), y0 - 14, 7, 0, 7); g.fill();
      const learnsNow = /select|scale|fit/.test(order[ti]), allRows = mode !== 'honest' && ti < splitIdx;
      caption(g, w, `run ${scores.length + 1}: "${order[ti]}" — ${learnsNow ? 'parameters are LEARNED here from ' + (allRows ? 'ALL rows, including the test rows' : 'training rows only') : 'no learning at this stage'}`);
      const b = hc.ctx; b.clearRect(0, 0, hc.w, hc.h); const pad = { l: 36, r: 10, t: 10, b: 20 }, pw = hc.w - pad.l - pad.r, ph = hc.h - pad.t - pad.b, Y = (v) => pad.t + (1 - v) * ph;
      b.strokeStyle = C.border; b.fillStyle = C.muted; b.font = FONT; b.textAlign = 'right'; [0, 0.5, 1].forEach((v) => { b.beginPath(); b.moveTo(pad.l, Y(v)); b.lineTo(pad.l + pw, Y(v)); b.stroke(); b.fillText(v * 100 + '%', pad.l - 4, Y(v) + 4); });
      b.strokeStyle = C.red; b.setLineDash([5, 4]); b.beginPath(); b.moveTo(pad.l, Y(0.5)); b.lineTo(pad.l + pw, Y(0.5)); b.stroke(); b.setLineDash([]);
      const nmax = Math.max(30, scores.length); scores.forEach((s, i) => { b.fillStyle = C.blue; b.globalAlpha = 0.6; b.beginPath(); b.arc(pad.l + ((i + 0.5) / nmax) * pw, Y(s), 2.5, 0, 7); b.fill(); }); b.globalAlpha = 1;
      b.strokeStyle = C.orange; b.lineWidth = 2; b.beginPath(); let run = 0; scores.forEach((s, i) => { run += s; const v = run / (i + 1); i ? b.lineTo(pad.l + ((i + 0.5) / nmax) * pw, Y(v)) : b.moveTo(pad.l + (0.5 / nmax) * pw, Y(v)); }); b.stroke(); b.lineWidth = 1;
      q('[data-a=out]').innerHTML = `<span class="m">runs</span> ${scores.length}   <span class="m">average test accuracy</span> <span class="${mode === 'leakSel' ? 'r' : 'g'}">${scores.length ? (mean(scores) * 100).toFixed(1) + '%' : '–'}</span>   <span class="m">true accuracy</span> 50% (features are noise)`;
    }, { fps: 20 });
    cv.onresize = () => (dirty = true); hc.onresize = () => (dirty = true);
  };

  /* ======================= split lab ======================= */
  AI.labs.splitlab = function (root) {
    const { view, ctl } = AI.labShell(root, 'Splitting lab: stratified counts and time', '20 rows with 5 positives; then a drifting time series (illustrative)',
      `<canvas data-a="a" style="height:230px"></canvas><canvas data-a="t" style="height:230px"></canvas><div class="caption">Top: repeated 80/20 splits. Bottom: forecasting a drifting series with a random split vs a time split.</div>`,
      `<label>Split type <select data-a="kind"><option value="rand">random</option><option value="strat">stratified</option></select></label>
       <label>Series drift <span class="v" data-a="dv"></span><input type="range" data-a="dr" min="0" max="3" step="0.1" value="1.5"></label>
       <label>Time panel split <select data-a="tm"><option value="rand">random split (leaky)</option><option value="time">time split (honest)</option></select></label>
       ${playCtl(1)}<button class="btn" data-a="b200">Run 200 splits</button><div class="readout" data-a="out"></div>`);
    const q = (s) => $(root, s), ca = AI.canvas(q('[data-a=a]')), ct = AI.canvas(q('[data-a=t]'));
    const pos = [1, 1, 1, 1, 1].concat(new Array(15).fill(0)); let counts = { rand: [0, 0, 0, 0, 0], strat: [0, 0, 0, 0, 0] }, cur = null, dirty = true, seed = 1, acc = 0;
    const shuffle = (a, r) => a.map((v) => [r(), v]).sort((u, v) => u[0] - v[0]).map((u) => u[1]);
    const split = (kind, sd) => { const r = AI.rng(sd); if (kind === 'rand') return shuffle([...Array(20).keys()], r).slice(0, 4); return shuffle([0, 1, 2, 3, 4], r).slice(0, 1).concat(shuffle([...Array(15).keys()].map((i) => i + 5), r).slice(0, 3)); };
    const addCount = (kind, idx) => counts[kind][idx.filter((i) => pos[i]).length]++;
    const stepOnce = () => { const k = q('[data-a=kind]').value; cur = split(k, seed++); addCount(k, cur); dirty = true; };
    const st = wire(root, stepOnce, () => { counts = { rand: [0, 0, 0, 0, 0], strat: [0, 0, 0, 0, 0] }; cur = null; dirty = true; }); A(root, 'sp').value = 1;
    q('[data-a=b200]').onclick = () => { const k = q('[data-a=kind]').value; for (let i = 0; i < 200; i++) addCount(k, split(k, 1000 + seed++)); dirty = true; };
    ['kind', 'dr', 'tm'].forEach((k) => (q(`[data-a=${k}]`).oninput = () => (dirty = true)));
    const T = 60, series = (dr) => { const r = AI.rng(3); return Array.from({ length: T }, (_, i) => 1.5 + dr * 0.06 * i + 0.8 * Math.sin(i / 3) + 0.25 * AI.randn(r)); };
    AI.loop(root, (dt) => {
      if (st.playing) { acc += (dt * st.speed()) / 0.9; if (acc >= 1) { acc = 0; stepOnce(); } }
      if (!dirty) return; dirty = false; const C = AI.col(), k = q('[data-a=kind]').value, g = ca.ctx; g.clearRect(0, 0, ca.w, ca.h); g.font = FONT;
      const cw = Math.min(34, (ca.w - 20) / 20); for (let i = 0; i < 20; i++) { const x = 10 + i * cw, isT = cur && cur.includes(i); g.fillStyle = pos[i] ? C.orange : C.blue; g.globalAlpha = isT ? 1 : 0.55; g.fillRect(x, 28, cw - 3, 26); g.globalAlpha = 1; if (isT) { g.strokeStyle = C.fg; g.lineWidth = 2.5; g.strokeRect(x, 26, cw - 3, 30); } }
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('20 rows (orange = positive); outlined = the 4 chosen for the test set', 10, 18);
      const tot = counts[k].reduce((a, b) => a + b, 0), exp = [1365, 2275, 1050, 150, 5].map((v) => v / 4845);
      const pad = { l: 20, r: 12, t: 92, b: 22 }, pw = ca.w - pad.l - pad.r, phh = ca.h - pad.t - pad.b, bw = pw / 5, mx = Math.max(0.5, ...counts[k].map((v) => (tot ? v / tot : 0)));
      for (let j = 0; j < 5; j++) { const f = tot ? counts[k][j] / tot : 0, hh = (f / mx) * phh; g.fillStyle = k === 'rand' ? C.red : C.green; g.fillRect(pad.l + j * bw + 8, pad.t + phh - hh, bw - 16, hh); g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(tot ? (f * 100).toFixed(0) + '%' : '', pad.l + j * bw + bw / 2, pad.t + phh - hh - 4); g.fillStyle = C.muted; g.fillText(j + ' pos. in test', pad.l + j * bw + bw / 2, pad.t + phh + 14); if (k === 'rand') { g.strokeStyle = C.yellow; g.lineWidth = 2; const yy = pad.t + phh - (exp[j] / mx) * phh; g.beginPath(); g.moveTo(pad.l + j * bw + 4, yy); g.lineTo(pad.l + j * bw + bw - 4, yy); g.stroke(); } }
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(k === 'rand' ? 'yellow = exact probability: 28.2%, 46.9%, 21.7%, 3.1%, 0.1%' : 'stratified: always exactly 1 positive in the test set', 10, 76);
      const dr = +q('[data-a=dr]').value, S = series(dr), mode = q('[data-a=tm]').value; q('[data-a=dv]').textContent = dr.toFixed(1);
      const r = AI.rng(8), test = mode === 'time' ? Array.from({ length: 18 }, (_, i) => T - 18 + i) : shuffle([...Array(T).keys()], r).slice(0, 18).sort((a, b) => a - b), tset = new Set(test), trn = S.map((v, i) => [i, v]).filter((p) => !tset.has(p[0]));
      const pred = (i) => mean(trn.map((p) => [Math.abs(p[0] - i), p[1]]).sort((a, b) => a[0] - b[0]).slice(0, 2).map((p) => p[1]));
      const g2 = ct.ctx; g2.clearRect(0, 0, ct.w, ct.h); const v = AI.view(ct, 0, T, 0, 8, { l: 34, r: 10, t: 24, b: 22 }); AI.axes(g2, v);
      let mae = 0; test.forEach((i) => { mae += Math.abs(pred(i) - S[i]); }); mae /= test.length;
      if (mode === 'time') { g2.fillStyle = C.orange; g2.globalAlpha = 0.12; g2.fillRect(v.X(T - 18), v.pad.t, v.X(T) - v.X(T - 18), v.ph); g2.globalAlpha = 1; }
      S.forEach((val, i) => { const t = tset.has(i); g2.fillStyle = t ? C.orange : C.blue; g2.beginPath(); g2.arc(v.X(i), v.Y(val), t ? 4 : 3, 0, 7); g2.fill(); if (t) { g2.strokeStyle = C.red; g2.lineWidth = 1; g2.beginPath(); g2.moveTo(v.X(i), v.Y(val)); g2.lineTo(v.X(i), v.Y(pred(i))); g2.stroke(); g2.fillStyle = C.green; g2.fillRect(v.X(i) - 2, v.Y(pred(i)) - 2, 4, 4); } });
      g2.fillStyle = C.muted; g2.font = FONT; g2.textAlign = 'left'; g2.fillText('blue = train, orange = test, green = prediction (mean of 2 nearest training points in time)', v.pad.l, 14);
      q('[data-a=out]').innerHTML = `<span class="m">splits recorded (${k})</span> ${tot}   <span class="m">with 0 positives in test</span> <span class="${k === 'rand' ? 'r' : 'g'}">${tot ? ((counts[k][0] / tot) * 100).toFixed(1) : '–'}%</span>\n<span class="m">time panel (${mode})</span> mean abs error <span class="${mode === 'rand' ? 'g' : 'o'}">${AI.fmt(mae, 3)}</span>  ${mode === 'rand' ? '— looks great: neighbours in time are in training (the future leaks)' : '— the honest cost of extrapolating into the future under drift'}`;
    }, { fps: 20 });
    ca.onresize = () => (dirty = true); ct.onresize = () => (dirty = true);
  };
})();
