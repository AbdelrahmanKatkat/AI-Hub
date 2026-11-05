/* Data Science Module 7 labs: predictive modelling in practice.
   ds-md-split, ds-md-kfold, ds-md-baseline, ds-md-learning, ds-md-threshold, ds-md-tuning, ds-md-search, ds-md-explain.
   All simulations are illustrative: seeded random numbers, dt-based animation, slow defaults, pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const spd = () => sl('spd', 'Speed', 0.25, 3, 0.25, 1);
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  const sd = (a) => { const m = mean(a); return a.length > 1 ? Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - 1)) : 0; };
  const shuffle = (arr, r) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  // linear map helper: mk(x0, y0, w, h, xmin, xmax, ymin, ymax) -> {X, Y}
  const mk = (x0, y0, w, h, xa, xb, ya, yb) => ({ X: (x) => x0 + ((x - xa) / (xb - xa)) * w, Y: (y) => y0 + h - ((y - ya) / (yb - ya)) * h });
  const frame = (g, c, x0, y0, w, h) => { g.strokeStyle = c.border; g.lineWidth = 1; g.strokeRect(x0, y0, w, h); };
  const themed = (draw) => window.addEventListener('themechange', () => draw());

  /* ================= 1. Train / validation / test split ================= */
  AI.labs['ds-md-split'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Splitting 100 rows', 'illustrative: each dot is one labelled row; orange dots are the rare positive class',
      `<canvas style="height:260px"></canvas>`,
      `<label>Split method <select id="mode"><option value="random">random</option><option value="strat">stratified (keep the class mix)</option><option value="time">time-based (oldest rows train)</option></select></label>
       ${sl('tr', 'Train share %', 40, 80, 5, 60)}
       ${sl('va', 'Validation share %', 5, 30, 5, 20)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="shuf">Reshuffle now</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. 100 rows with about 10 positives (positives become a little more common in later rows, as in real data that drifts). Dots glide to their new set; the test share is whatever is left over. A new random split is drawn every few seconds while playing.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const N = 100, rl = AI.rng(11);
    const lab = []; for (let i = 0; i < N; i++) lab.push(rl() < 0.04 + 0.12 * (i / (N - 1)) ? 1 : 0);
    const dots = lab.map((l, i) => ({ l, i, x: 0, y: 0, sx: 0, sy: 0, tx: 0, ty: 0 }));
    let count = 0, prog = 1, wait = 0, playing = true, dirty = true, hist = [], inited = false;
    const modeEl = q(ctl, '#mode'), trEl = q(ctl, '#tr'), vaEl = q(ctl, '#va'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    const assign = () => {
      const tr = +trEl.value / 100, va = +vaEl.value / 100, mode = modeEl.value, a = new Array(N).fill(2);
      const r = AI.rng(1000 + count * 7919);
      const cut = (idx, f) => { const nt = Math.round(idx.length * tr), nv = Math.round(idx.length * va); idx.forEach((id, k) => { a[id] = k < nt ? 0 : k < nt + nv ? 1 : 2; }); };
      if (mode === 'time') cut(dots.map((d) => d.i));
      else if (mode === 'random') cut(shuffle(dots.map((d) => d.i), r));
      else { cut(shuffle(dots.filter((d) => d.l).map((d) => d.i), r)); cut(shuffle(dots.filter((d) => !d.l).map((d) => d.i), r)); }
      return a;
    };
    let asg = assign();
    const layout = () => {
      const W = o.w, shares = [0, 1, 2].map((s) => Math.max(8, asg.filter((v) => v === s).length)), tot = shares.reduce((a, b) => a + b, 0);
      const avail = W - 40, boxes = []; let x = 12;
      shares.forEach((s) => { const bw = (avail * s) / tot; boxes.push({ x, w: bw }); x += bw + 8; });
      const R = 6, step = 2 * R + 4;
      for (let s = 0; s < 3; s++) {
        const cols = Math.max(1, Math.floor((boxes[s].w - 12) / step)); let k = 0;
        dots.forEach((d) => { if (asg[d.i] === s) { d.tx = boxes[s].x + 8 + R + (k % cols) * step; d.ty = 52 + R + Math.floor(k / cols) * step; k++; } });
      }
      return boxes;
    };
    let boxes = layout();
    const start = () => { dots.forEach((d) => { d.sx = d.x; d.sy = d.y; }); boxes = layout(); if (!inited) { dots.forEach((d) => { d.x = d.sx = d.tx; d.y = d.sy = d.ty; }); inited = true; prog = 1; } else prog = 0; dirty = true; };
    const reshuffle = () => { count++; asg = assign(); start(); };
    const counts = () => [0, 1, 2].map((s) => { const m = dots.filter((d) => asg[d.i] === s); return { n: m.length, p: m.filter((d) => d.l).length }; });
    const draw = () => {
      const g = o.ctx, c = AI.col(); g.clearRect(0, 0, o.w, o.h);
      const names = ['Train', 'Validation', 'Test'], cs = counts();
      boxes.forEach((b, s) => {
        g.strokeStyle = [c.blue, c.purple, c.green][s]; g.lineWidth = 2; g.strokeRect(b.x, 36, b.w, o.h - 80);
        txt(g, `${names[s]}: ${cs[s].n} rows`, b.x + 4, 28, [c.blue, c.purple, c.green][s], 'bold 12px sans-serif');
        txt(g, `${cs[s].p} positive (${cs[s].n ? Math.round((100 * cs[s].p) / cs[s].n) : 0}%)`, b.x + 4, o.h - 26, c.fg, '12px sans-serif');
      });
      dots.forEach((d) => { g.fillStyle = d.l ? c.orange : c.muted; g.globalAlpha = d.l ? 1 : 0.55; g.beginPath(); g.arc(d.x, d.y, 6, 0, 7); g.fill(); });
      g.globalAlpha = 1; g.lineWidth = 1;
      const m = modeEl.value, cap = m === 'time' ? 'Time-based: the oldest rows train, the newest test. Nothing to reshuffle; test sees the future.' : m === 'random' ? 'Random: positives can land unevenly, so a small set may hold few or none.' : 'Stratified: every set keeps the same share of positives.';
      txt(g, cap, 12, o.h - 6, c.muted, '11px sans-serif');
      const tp = cs[2].p; if (!hist.length || hist[hist.length - 1] !== tp || false) { /* history is pushed in reshuffle */ }
      out.innerHTML = `Positives in train / validation / test: <b>${cs[0].p} / ${cs[1].p} / ${cs[2].p}</b>. Test-set positives over the last random splits: ${hist.slice(-8).join(', ') || 'none yet'}.`;
    };
    const apply = () => { reshuffle(); hist.push(counts()[2].p); };
    [modeEl, trEl, vaEl].forEach((e) => { e.oninput = () => { if (+trEl.value + +vaEl.value > 90) vaEl.value = 90 - +trEl.value; q(ctl, '#trv').textContent = trEl.value; q(ctl, '#vav').textContent = vaEl.value; hist = []; apply(); wait = 0; }; });
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#shuf').onclick = () => { apply(); wait = 0; };
    q(ctl, '#reset').onclick = () => { count = 0; hist = []; asg = assign(); start(); wait = 0; };
    o.onresize = () => { boxes = layout(); dots.forEach((d) => { d.x = d.tx; d.y = d.ty; }); dirty = true; };
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#trv').textContent = trEl.value; q(ctl, '#vav').textContent = vaEl.value; q(ctl, '#spdv').textContent = '1×';
    start(); themed(() => { dirty = true; });
    AI.loop(root, (dt) => {
      const s = +spdEl.value;
      if (prog < 1) {
        prog = Math.min(1, prog + (dt * s) / 1.8);
        dots.forEach((d) => { const t = AI.smooth(clamp(prog * 1.5 - (d.i / N) * 0.5, 0, 1)); d.x = d.sx + (d.tx - d.sx) * t; d.y = d.sy + (d.ty - d.sy) * t; });
        dirty = true;
      } else if (playing && modeEl.value !== 'time') { wait += dt * s; if (wait > 3.2) { wait = 0; apply(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    hist.push(counts()[2].p);
  };

  /* ================= 2. k-fold cross-validation ================= */
  AI.labs['ds-md-kfold'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'k-fold cross-validation', 'illustrative: the held-out fold rotates; the "model" predicts the training mean',
      `<canvas style="height:330px"></canvas>`,
      `<label>Ten target values (edit, comma separated)<input type="text" id="y" value="12,15,11,18,14,20,13,17,16,19" style="width:100%"></label>
       ${sl('k', 'Folds k', 2, 10, 1, 5)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next fold</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Each row is one data point. In fold s (orange) the model is "fit" on the blue rows (it predicts their mean) and scored on the orange rows with mean absolute error. After k rounds every row has been held out exactly once. Set k = number of rows for leave-one-out.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const yEl = q(ctl, '#y'), kEl = q(ctl, '#k'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    let y = [], k = 5, res = [], step = 0, t = 0, playing = true, done = 0, dirty = true;
    const fold = (i) => Math.floor((i * k) / y.length);
    const compute = () => {
      y = yEl.value.split(/[ ,;]+/).map(Number).filter((v) => Number.isFinite(v)).slice(0, 20);
      if (y.length < 4) y = [12, 15, 11, 18, 14, 20, 13, 17, 16, 19];
      kEl.max = y.length; k = clamp(+kEl.value, 2, y.length); kEl.value = k; q(ctl, '#kv').textContent = k;
      res = []; for (let f = 0; f < k; f++) {
        const tr = y.filter((_, i) => fold(i) !== f), te = y.filter((_, i) => fold(i) === f), m = mean(tr);
        res.push({ m, mae: mean(te.map((v) => Math.abs(v - m))), n: te.length });
      }
      step = 0; t = 0; done = 0; dirty = true;
    };
    const finish = () => { done = Math.max(done, step + 1); step++; t = 0; dirty = true; if (step >= k) { step = k; } };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const n = y.length, cw = Math.min(54, (W - 24) / n), x0 = (W - cw * n) / 2, ymax = Math.max(...y) * 1.1;
      const f = Math.min(step, k - 1), cur = step < k ? res[f] : null, revealed = t > 0.5 || step >= k;
      txt(g, step < k ? `Round ${f + 1} of ${k}: fold ${f + 1} is held out` : `All ${k} rounds done`, 12, 18, c.fg, 'bold 13px sans-serif');
      for (let i = 0; i < n; i++) {
        const held = step < k && fold(i) === f, h = (y[i] / ymax) * 90, x = x0 + i * cw;
        g.fillStyle = held ? c.orange : c.blue; g.globalAlpha = held ? 1 : 0.55; g.fillRect(x + 3, 130 - h, cw - 6, h); g.globalAlpha = 1;
        txt(g, String(y[i]), x + cw / 2, 124 - h, c.fg, '11px sans-serif', 'center');
        txt(g, 'f' + (fold(i) + 1), x + cw / 2, 146, c.muted, '10px sans-serif', 'center');
      }
      if (cur) {
        const ym = 130 - (cur.m / ymax) * 90; g.strokeStyle = c.green; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(x0, ym); g.lineTo(x0 + cw * n, ym); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
        txt(g, `training mean = ${cur.m.toFixed(3)} (prediction)`, 12, 36, c.green, '12px sans-serif');
        if (revealed) txt(g, `fold ${f + 1} error (MAE) = ${cur.mae.toFixed(3)}`, 12, 52, c.orange, 'bold 12px sans-serif');
      }
      // bars of fold errors so far
      const by = 175, bh = 110, mm = Math.max(1, ...res.map((r) => r.mae)) * 1.2, bw = Math.min(60, (W - 60) / k);
      txt(g, 'fold errors so far (MAE)', 12, by - 6, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(30, by); g.lineTo(30, by + bh); g.lineTo(W - 10, by + bh); g.stroke();
      const got = res.slice(0, done).map((r) => r.mae);
      for (let j = 0; j < k; j++) {
        const x = 40 + j * bw;
        if (j < done) { const h = (res[j].mae / mm) * bh; g.fillStyle = c.purple; g.fillRect(x, by + bh - h, bw - 8, h); txt(g, res[j].mae.toFixed(2), x + (bw - 8) / 2, by + bh - h - 4, c.fg, '10px sans-serif', 'center'); }
        else { g.strokeStyle = c.border; g.setLineDash([3, 3]); g.strokeRect(x, by + bh - 6, bw - 8, 6); g.setLineDash([]); }
        txt(g, 'f' + (j + 1), x + (bw - 8) / 2, by + bh + 14, c.muted, '10px sans-serif', 'center');
      }
      if (got.length) { const m = mean(got); const yy = by + bh - (m / mm) * bh; g.strokeStyle = c.yellow; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(30, yy); g.lineTo(W - 10, yy); g.stroke(); g.setLineDash([]); }
      out.innerHTML = got.length ? `Mean of ${got.length} fold error${got.length > 1 ? 's' : ''} = <b>${mean(got).toFixed(3)}</b>` + (got.length > 1 ? `, spread (sd) = ${sd(got).toFixed(3)}, standard error ≈ ${(sd(got) / Math.sqrt(got.length)).toFixed(3)}` : '') : 'No fold scored yet.';
    };
    [yEl, kEl].forEach((e) => (e.oninput = compute));
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { if (step < k) finish(); else compute(); };
    q(ctl, '#reset').onclick = compute;
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#spdv').textContent = '1×';
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    compute();
    let hold = 0;
    AI.loop(root, (dt) => {
      const s = +spdEl.value;
      if (playing) {
        if (step < k) { const before = t; t += (dt * s) / 3; if (before <= 0.5 && t > 0.5) dirty = true; if (t >= 1) finish(); }
        else { hold += dt * s; if (hold > 5) { hold = 0; compute(); } }
      }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 3. Baselines under class imbalance ================= */
  AI.labs['ds-md-baseline'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Baseline versus model on rare cases', 'illustrative: patients arrive one at a time; the baseline always says "healthy"',
      `<canvas style="height:330px"></canvas>`,
      `${sl('pr', 'Share of sick patients %', 1, 40, 1, 6)}
       ${sl('se', 'Model sensitivity % (sick found)', 0, 100, 5, 60)}
       ${sl('sp', 'Model specificity % (healthy cleared)', 70, 100, 1, 95)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Top strip: the last 60 arrivals (orange = sick; a ring means the model flagged them). Chart: running accuracy of the do-nothing baseline (grey) and of the model (blue). Changing a slider restarts the stream with the same seed.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const prEl = q(ctl, '#pr'), seEl = q(ctl, '#se'), spEl = q(ctl, '#sp'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    const MAX = 400; let r, n, st, accB, accM, strip, playing = true, dirty = true, acc = 0;
    const reset = () => { r = AI.rng(5); n = 0; st = { tp: 0, fp: 0, fn: 0, tn: 0 }; accB = []; accM = []; strip = []; acc = 0; dirty = true;
      [['pr', '%'], ['se', '%'], ['sp', '%']].forEach(([id, u]) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value + u)); q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    const arrive = () => {
      const sick = r() < +prEl.value / 100, flag = sick ? r() < +seEl.value / 100 : r() < 1 - +spEl.value / 100;
      n++; if (sick) flag ? st.tp++ : st.fn++; else flag ? st.fp++ : st.tn++;
      accB.push((st.tn + st.fp) / n); accM.push((st.tp + st.tn) / n); strip.push({ sick, flag }); if (strip.length > 60) strip.shift();
    };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const cw = Math.min(14, (W - 24) / 60);
      strip.forEach((d, i) => { const x = 12 + i * cw + cw / 2; g.fillStyle = d.sick ? c.orange : c.muted; g.globalAlpha = d.sick ? 1 : 0.5; g.beginPath(); g.arc(x, 22, cw * 0.35, 0, 7); g.fill(); g.globalAlpha = 1; if (d.flag) { g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); g.arc(x, 22, cw * 0.48, 0, 7); g.stroke(); g.lineWidth = 1; } });
      const m = mk(44, 50, W - 60, H - 150, 0, MAX, 0, 1);
      frame(g, c, 44, 50, W - 60, H - 150);
      [0, 0.5, 1].forEach((v) => { txt(g, v.toFixed(1), 38, m.Y(v) + 4, c.muted, '10px sans-serif', 'right'); });
      txt(g, 'patients seen', W - 16, H - 92, c.muted, '10px sans-serif', 'right'); txt(g, 'accuracy', 44, 44, c.muted, '10px sans-serif');
      [[accB, c.muted], [accM, c.blue]].forEach(([a, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); a.forEach((v, i) => { const x = m.X(i + 1), yy = m.Y(v); i ? g.lineTo(x, yy) : g.moveTo(x, yy); }); g.stroke(); g.lineWidth = 1; });
      const pos = st.tp + st.fn, precision = st.tp + st.fp ? st.tp / (st.tp + st.fp) : NaN, recall = pos ? st.tp / pos : NaN;
      const f = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–'), y0 = H - 76;
      txt(g, `n = ${n}`, 12, y0, c.fg, 'bold 12px monospace');
      txt(g, `baseline  accuracy ${f(n ? (st.tn + st.fp) / n : NaN)}   recall ${n ? '0.000' : '–'}   precision undefined (never says "sick")`, 12, y0 + 18, c.muted, '12px monospace');
      txt(g, `model     accuracy ${f(n ? (st.tp + st.tn) / n : NaN)}   recall ${f(recall)}   precision ${f(precision)}`, 12, y0 + 36, c.blue, '12px monospace');
      txt(g, `model counts: TP ${st.tp}  FP ${st.fp}  FN ${st.fn}  TN ${st.tn}`, 12, y0 + 54, c.fg, '12px monospace');
      out.innerHTML = n ? `Do-nothing baseline: <b>${((100 * (st.tn + st.fp)) / n).toFixed(1)}%</b> accurate while finding <b>0</b> of ${pos} sick patients. The model is ${((100 * (st.tp + st.tn)) / n).toFixed(1)}% accurate and finds ${st.tp} of ${pos}.` : 'Waiting for the first patient.';
    };
    [prEl, seEl, spEl].forEach((e) => (e.oninput = reset)); spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = reset;
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    reset();
    AI.loop(root, (dt) => {
      if (playing && n < MAX) { acc += dt * 8 * +spdEl.value; let k = Math.min(Math.floor(acc), 10); acc -= Math.floor(acc); while (k-- > 0 && n < MAX) { arrive(); dirty = true; } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 4. Learning curve ================= */
  AI.labs['ds-md-learning'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A learning curve filling in', 'illustrative: more training rows, a flexible polynomial versus a straight-line baseline',
      `<canvas style="height:340px"></canvas>`,
      `${sl('d', 'Flexible model: polynomial degree', 2, 10, 1, 6)}
       ${sl('sg', 'Noise level σ', 0.05, 0.6, 0.05, 0.25)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">+10 rows</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Left: the first m training rows, the flexible fit (orange) and the straight-line baseline (grey dashed). Right: error on the training rows (orange) and on 300 unseen validation rows (blue); the grey curve is the straight line's validation error. The dotted line is the noise floor σ: no model can score below it. Errors above 2 are clipped.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const dEl = q(ctl, '#d'), sEl = q(ctl, '#sg'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    const F = (x) => Math.sin(2.4 * x) + 0.4 * x, MMAX = 120, MIN = 4;
    let X, Y, VX, VY, cur, curves, m = MIN, playing = true, dirty = true;
    const fit = (xs, ys, deg) => {
      const n = deg + 1, A = Array.from({ length: n }, () => new Array(n + 1).fill(0));
      xs.forEach((x, k) => { const p = []; for (let i = 0; i < n; i++) p.push(Math.pow(x, i)); for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) A[i][j] += p[i] * p[j]; A[i][n] += p[i] * ys[k]; } });
      for (let i = 0; i < n; i++) A[i][i] += 1e-6;
      for (let i = 0; i < n; i++) { let b = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[b][i])) b = r; [A[i], A[b]] = [A[b], A[i]];
        for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let cc = i; cc <= n; cc++) A[r][cc] -= f * A[i][cc]; } }
      const w = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = A[i][n]; for (let j = i + 1; j < n; j++) s -= A[i][j] * w[j]; w[i] = s / A[i][i]; }
      return w;
    };
    const ev = (w, x) => { let s = 0, p = 1; for (let i = 0; i < w.length; i++) { s += w[i] * p; p *= x; } return s; };
    const rmse = (w, xs, ys) => Math.sqrt(mean(xs.map((x, i) => (ev(w, x) - ys[i]) ** 2)));
    const build = () => {
      const sg = +sEl.value, deg = +dEl.value, r = AI.rng(21), rn = () => AI.randn(r);
      X = []; Y = []; for (let i = 0; i < MMAX; i++) { const x = r() * 2 - 1; X.push(x); Y.push(F(x) + sg * rn()); }
      VX = []; VY = []; for (let i = 0; i < 300; i++) { const x = r() * 2 - 1; VX.push(x); VY.push(F(x) + sg * rn()); }
      curves = []; for (let mm = MIN; mm <= MMAX; mm++) { const xs = X.slice(0, mm), ys = Y.slice(0, mm), w = fit(xs, ys, deg), w1 = fit(xs, ys, 1);
        curves.push({ m: mm, tr: rmse(w, xs, ys), va: rmse(w, VX, VY), vb: rmse(w1, VX, VY), w, w1 }); }
      q(ctl, '#dv').textContent = deg; q(ctl, '#sgv').textContent = sg.toFixed(2); q(ctl, '#spdv').textContent = spdEl.value + '×'; dirty = true;
    };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const half = W / 2, k = curves[clamp(Math.floor(m) - MIN, 0, curves.length - 1)], mm = Math.floor(m), sg = +sEl.value;
      // left: data and fits
      const L = mk(34, 28, half - 50, H - 70, -1, 1, -2.2, 2.2); frame(g, c, 34, 28, half - 50, H - 70);
      txt(g, `first ${mm} training rows`, 36, 20, c.fg, 'bold 12px sans-serif');
      g.save(); g.beginPath(); g.rect(34, 28, half - 50, H - 70); g.clip();
      g.strokeStyle = c.border; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = -1 + i / 50; i ? g.lineTo(L.X(x), L.Y(F(x))) : g.moveTo(L.X(x), L.Y(F(x))); } g.stroke();
      g.fillStyle = c.blue; for (let i = 0; i < mm; i++) { g.beginPath(); g.arc(L.X(X[i]), L.Y(Y[i]), 3, 0, 7); g.fill(); }
      g.strokeStyle = c.muted; g.setLineDash([5, 4]); g.beginPath(); for (let i = 0; i <= 60; i++) { const x = -1 + i / 30; i ? g.lineTo(L.X(x), L.Y(ev(k.w1, x))) : g.moveTo(L.X(x), L.Y(ev(k.w1, x))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = c.orange; g.lineWidth = 2.5; g.beginPath(); for (let i = 0; i <= 120; i++) { const x = -1 + i / 60; i ? g.lineTo(L.X(x), L.Y(ev(k.w, x))) : g.moveTo(L.X(x), L.Y(ev(k.w, x))); } g.stroke(); g.lineWidth = 1; g.restore();
      txt(g, 'grey line = true pattern (no noise)', 38, H - 28, c.muted, '10px sans-serif');
      // right: learning curves
      const x0 = half + 24, w = half - 40, R = mk(x0, 28, w, H - 70, 0, MMAX, 0, 2); frame(g, c, x0, 28, w, H - 70);
      txt(g, 'error (RMSE) vs training rows', x0 + 2, 20, c.fg, 'bold 12px sans-serif');
      [0, 0.5, 1, 1.5, 2].forEach((v) => txt(g, String(v), x0 - 4, R.Y(v) + 4, c.muted, '10px sans-serif', 'right'));
      [0, 40, 80, 120].forEach((v) => txt(g, String(v), R.X(v), H - 24, c.muted, '10px sans-serif', 'center'));
      g.strokeStyle = c.green; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(x0, R.Y(sg)); g.lineTo(x0 + w, R.Y(sg)); g.stroke(); g.setLineDash([]);
      txt(g, 'noise floor', x0 + w - 4, R.Y(sg) - 4, c.green, '10px sans-serif', 'right');
      const line = (key, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); let first = true; curves.forEach((cv2) => { if (cv2.m > mm) return; const v = Math.min(2, cv2[key]), xx = R.X(cv2.m), yy = R.Y(v); first ? g.moveTo(xx, yy) : g.lineTo(xx, yy); first = false; }); g.stroke(); g.lineWidth = 1; };
      line('vb', c.muted); line('tr', c.orange); line('va', c.blue);
      txt(g, 'validation (flexible)', x0 + 6, H - 8, c.blue, '10px sans-serif'); txt(g, 'training', x0 + 120, H - 8, c.orange, '10px sans-serif'); txt(g, 'validation (line)', x0 + 176, H - 8, c.muted, '10px sans-serif');
      const gap = k.va - k.tr;
      out.innerHTML = `m = <b>${mm}</b> rows. Flexible model: training ${k.tr.toFixed(3)}, validation ${k.va.toFixed(3)} (gap ${gap.toFixed(3)}). Straight-line baseline validation: ${k.vb.toFixed(3)}. ${k.va < k.vb ? 'The flexible model is ahead of the baseline.' : 'The simple baseline is still at least as good.'}`;
    };
    [dEl, sEl].forEach((e) => (e.oninput = () => { build(); }));
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { m = Math.min(MMAX, m + 10); dirty = true; };
    q(ctl, '#reset').onclick = () => { m = MIN; dirty = true; };
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    build();
    AI.loop(root, (dt) => {
      if (playing && m < MMAX) { m = Math.min(MMAX, m + dt * 5 * +spdEl.value); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 5. Threshold and confusion matrix ================= */
  AI.labs['ds-md-threshold'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A threshold moving the confusion matrix', 'illustrative: 200 scored cases; everything at or above the threshold is flagged positive',
      `<canvas style="height:380px"></canvas>`,
      `${sl('th', 'Threshold', 0.05, 0.95, 0.01, 0.5)}
       ${sl('pv', 'Share of positives %', 5, 50, 1, 30)}
       ${sl('sep', 'Class separation', 0.1, 0.6, 0.05, 0.3)}
       <label>Cost of one false alarm (FP) <input type="number" id="cfp" value="1" min="0" step="1" style="width:5em"></label>
       <label>Cost of one miss (FN) <input type="number" id="cfn" value="5" min="0" step="1" style="width:5em"></label>
       ${spd()}
       <div class="row"><button class="btn primary" id="play">▶ Sweep</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Histogram of model scores: positives orange, negatives grey. Sliding the threshold changes only the decision, not the model. Bottom right: precision (green) and recall (blue) at every threshold; the dot is where you are. Costs are per case.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const thEl = q(ctl, '#th'), pvEl = q(ctl, '#pv'), sepEl = q(ctl, '#sep'), spdEl = q(ctl, '#spd'), cfp = q(ctl, '#cfp'), cfn = q(ctl, '#cfn'), out = q(ctl, '#out');
    const N = 200, r = AI.rng(33), U = [], Z = []; for (let i = 0; i < N; i++) { U.push(r()); Z.push(AI.randn(r)); }
    let lab, sc, th = 0.5, dir = -1, playing = false, dirty = true;
    const build = () => { const p = +pvEl.value / 100, d = +sepEl.value; lab = U.map((u) => (u < p ? 1 : 0)); sc = Z.map((z, i) => clamp(0.5 + (lab[i] ? d / 2 : -d / 2) + 0.18 * z, 0, 0.999)); dirty = true; };
    const cm = (t) => { let tp = 0, fp = 0, fn = 0, tn = 0; for (let i = 0; i < N; i++) { const f = sc[i] >= t; if (f && lab[i]) tp++; else if (f) fp++; else if (lab[i]) fn++; else tn++; } return { tp, fp, fn, tn }; };
    const met = (m) => { const P = m.tp + m.fp ? m.tp / (m.tp + m.fp) : 1, R = m.tp + m.fn ? m.tp / (m.tp + m.fn) : 0; return { P, R, F: P + R ? (2 * P * R) / (P + R) : 0, A: (m.tp + m.tn) / N }; };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const bins = 20, hp = new Array(bins).fill(0), hn = new Array(bins).fill(0);
      sc.forEach((s, i) => { const b = Math.min(bins - 1, Math.floor(s * bins)); (lab[i] ? hp : hn)[b]++; });
      const mx = Math.max(...hp.map((v, i) => v + hn[i]), 1), hx = 30, hw = W - 44, hy = 22, hh = 150;
      const bw = hw / bins; frame(g, c, hx, hy, hw, hh);
      for (let b = 0; b < bins; b++) { const flagged = (b + 1) / bins > th - 1e-9 && b / bins >= th - 1e-9 || b / bins >= th;
        const h1 = (hn[b] / mx) * hh, h2 = (hp[b] / mx) * hh; g.globalAlpha = b / bins >= th - 1e-9 ? 1 : 0.4;
        g.fillStyle = c.muted; g.fillRect(hx + b * bw + 1, hy + hh - h1, bw - 2, h1); g.fillStyle = c.orange; g.fillRect(hx + b * bw + 1, hy + hh - h1 - h2, bw - 2, h2); g.globalAlpha = 1; }
      const tx = hx + th * hw; g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(tx, hy - 4); g.lineTo(tx, hy + hh); g.stroke(); g.lineWidth = 1;
      txt(g, `threshold ${th.toFixed(2)}: flagged → right of the line`, Math.min(tx + 6, W - 250), hy + 12, c.red, 'bold 11px sans-serif');
      txt(g, '0', hx, hy + hh + 14, c.muted, '10px sans-serif', 'center'); txt(g, 'score', hx + hw / 2, hy + hh + 14, c.muted, '10px sans-serif', 'center'); txt(g, '1', hx + hw, hy + hh + 14, c.muted, '10px sans-serif', 'center');
      // confusion matrix
      const m = cm(th), me = met(m), cx = 30, cy = 215, cs = 64;
      txt(g, 'actual ↓   predicted →', cx, cy - 8, c.muted, '10px sans-serif');
      const cells = [[m.tp, 'TP', c.green, 0, 0], [m.fn, 'FN', c.red, 0, 1], [m.fp, 'FP', c.orange, 1, 0], [m.tn, 'TN', c.blue, 1, 1]];
      txt(g, 'flagged', cx + 40 + cs / 2, cy + 6, c.fg, '11px sans-serif', 'center'); txt(g, 'not flagged', cx + 40 + cs * 1.5, cy + 6, c.fg, '11px sans-serif', 'center');
      txt(g, 'positive', cx + 36, cy + 14 + cs / 2, c.fg, '11px sans-serif', 'right'); txt(g, 'negative', cx + 36, cy + 14 + cs * 1.5, c.fg, '11px sans-serif', 'right');
      cells.forEach(([v, nm, col, rr, cc]) => { g.globalAlpha = 0.25; g.fillStyle = col; g.fillRect(cx + 40 + cc * cs, cy + 14 + rr * cs, cs - 3, cs - 3); g.globalAlpha = 1; txt(g, String(v), cx + 40 + cc * cs + cs / 2, cy + 14 + rr * cs + 30, c.fg, 'bold 18px sans-serif', 'center'); txt(g, nm, cx + 40 + cc * cs + cs / 2, cy + 14 + rr * cs + 48, col, '11px sans-serif', 'center'); });
      // PR curve
      const px = 270, pw = W - px - 16, py = 215, ph = 130, P = mk(px, py, pw, ph, 0, 1, 0, 1); frame(g, c, px, py, pw, ph);
      txt(g, 'precision and recall vs threshold', px, py - 6, c.muted, '10px sans-serif');
      [['P', c.green], ['R', c.blue]].forEach(([k, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 50; i++) { const t = 0.02 + (0.96 * i) / 50, e = met(cm(t)); const xx = P.X(t), yy = P.Y(e[k]); i ? g.lineTo(xx, yy) : g.moveTo(xx, yy); } g.stroke(); g.lineWidth = 1; });
      g.fillStyle = c.green; g.beginPath(); g.arc(P.X(th), P.Y(me.P), 4, 0, 7); g.fill(); g.fillStyle = c.blue; g.beginPath(); g.arc(P.X(th), P.Y(me.R), 4, 0, 7); g.fill();
      txt(g, 'precision', px + 6, py + ph - 6, c.green, '10px sans-serif'); txt(g, 'recall', px + 60, py + ph - 6, c.blue, '10px sans-serif');
      const fpc = +cfp.value || 0, fnc = +cfn.value || 0, cost = m.fp * fpc + m.fn * fnc; let bt = 0.05, bc = 1e9;
      for (let t = 0.05; t <= 0.951; t += 0.01) { const mm = cm(t), cc = mm.fp * fpc + mm.fn * fnc; if (cc < bc - 1e-9) { bc = cc; bt = t; } }
      out.innerHTML = `TP ${m.tp}, FP ${m.fp}, FN ${m.fn}, TN ${m.tn}.<br>Accuracy ${me.A.toFixed(3)} · precision ${me.P.toFixed(3)} · recall ${me.R.toFixed(3)} · F1 ${me.F.toFixed(3)}<br>Cost = ${m.fp}×${fpc} + ${m.fn}×${fnc} = <b>${cost}</b>; cheapest threshold ≈ <b>${bt.toFixed(2)}</b> (cost ${bc}).`;
      thEl.value = th.toFixed(2); q(ctl, '#thv').textContent = th.toFixed(2);
    };
    thEl.oninput = () => { th = +thEl.value; playing = false; q(ctl, '#play').textContent = '▶ Sweep'; dirty = true; };
    [pvEl, sepEl].forEach((e) => (e.oninput = () => { q(ctl, '#pvv').textContent = pvEl.value + '%'; q(ctl, '#sepv').textContent = (+sepEl.value).toFixed(2); build(); }));
    [cfp, cfn].forEach((e) => (e.oninput = () => { dirty = true; }));
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Sweep'; };
    q(ctl, '#reset').onclick = () => { th = 0.5; dir = -1; playing = false; q(ctl, '#play').textContent = '▶ Sweep'; dirty = true; };
    q(ctl, '#pvv').textContent = '30%'; q(ctl, '#sepv').textContent = '0.30'; q(ctl, '#spdv').textContent = '1×';
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    build();
    AI.loop(root, (dt) => {
      if (playing) { th += dir * dt * 0.05 * +spdEl.value; if (th <= 0.05) { th = 0.05; dir = 1; } if (th >= 0.95) { th = 0.95; dir = -1; } dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 6. Overfitting the validation set ================= */
  AI.labs['ds-md-tuning'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The winner of a tuning run', 'illustrative: configurations are tried one at a time; the best validation score is picked',
      `<canvas style="height:330px"></canvas>`,
      `${sl('nc', 'Configurations to try', 2, 100, 1, 40)}
       ${sl('nv', 'Validation rows', 50, 1000, 50, 100)}
       ${sl('spr', 'Real quality spread between configurations', 0, 0.03, 0.005, 0)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Reset</button><button class="btn" id="seed">New seed</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative (normal approximation to a binomial score). Every configuration has a real accuracy near 0.70 plus the spread you set. Blue dots: validation score of each configuration. Orange rings: score of the same configuration on 2000 fresh test rows. The line is the best validation score so far. With spread 0, all configurations are equal and the "winner" is pure luck.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const ncEl = q(ctl, '#nc'), nvEl = q(ctl, '#nv'), spEl = q(ctl, '#spr'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    let seed = 7, cfg = [], shown = 0, playing = true, dirty = true, gap = 0;
    const build = () => {
      const r = AI.rng(seed), nc = +ncEl.value, nv = +nvEl.value, spr = +spEl.value; cfg = [];
      for (let j = 0; j < nc; j++) { const a = clamp(0.7 + spr * AI.randn(r), 0.05, 0.95), zv = AI.randn(r), zt = AI.randn(r);
        cfg.push({ a, v: a + Math.sqrt((a * (1 - a)) / nv) * zv, t: a + Math.sqrt((a * (1 - a)) / 2000) * zt }); }
      // average optimism over 200 repeated studies (fresh seeds), using the same settings
      const rr = AI.rng(999), gaps = []; for (let s = 0; s < 200; s++) { let best = -1, bt = 0; for (let j = 0; j < nc; j++) { const a = clamp(0.7 + spr * AI.randn(rr), 0.05, 0.95), v = a + Math.sqrt((a * (1 - a)) / nv) * AI.randn(rr); if (v > best) { best = v; bt = a; } } gaps.push(best - bt); }
      gap = mean(gaps); shown = 0; dirty = true;
      [['nc', ''], ['nv', ''], ['spr', '']].forEach(([id]) => (q(ctl, '#' + id + 'v').textContent = (id === 'spr' ? (+q(ctl, '#' + id).value).toFixed(3) : q(ctl, '#' + id).value)));
      q(ctl, '#spdv').textContent = spdEl.value + '×';
    };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const n = cfg.length, k = Math.floor(shown), lo = 0.5, hi = 0.9, m = mk(44, 24, W - 60, H - 90, 0, n + 1, lo, hi); frame(g, c, 44, 24, W - 60, H - 90);
      [0.5, 0.6, 0.7, 0.8, 0.9].forEach((v) => { txt(g, v.toFixed(1), 38, m.Y(v) + 4, c.muted, '10px sans-serif', 'right'); });
      g.strokeStyle = c.green; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(44, m.Y(0.7)); g.lineTo(W - 16, m.Y(0.7)); g.stroke(); g.setLineDash([]);
      txt(g, 'real accuracy ≈ 0.70', W - 20, m.Y(0.7) + 14, c.green, '10px sans-serif', 'right');
      let best = -1, bi = -1; g.strokeStyle = c.yellow; g.lineWidth = 2; g.beginPath();
      for (let j = 0; j < k; j++) { if (cfg[j].v > best) { best = cfg[j].v; bi = j; } const x = m.X(j + 1), y = m.Y(best); j ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); g.lineWidth = 1;
      for (let j = 0; j < k; j++) { const x = m.X(j + 1); g.fillStyle = c.blue; g.beginPath(); g.arc(x, m.Y(clamp(cfg[j].v, lo, hi)), 3.5, 0, 7); g.fill();
        g.strokeStyle = c.orange; g.beginPath(); g.arc(x, m.Y(clamp(cfg[j].t, lo, hi)), 3.5, 0, 7); g.stroke(); }
      if (bi >= 0) { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.arc(m.X(bi + 1), m.Y(cfg[bi].v), 8, 0, 7); g.stroke(); g.lineWidth = 1; }
      txt(g, 'configuration number', 44 + (W - 60) / 2, H - 48, c.muted, '10px sans-serif', 'center');
      txt(g, 'validation score', 8, 16, c.muted, '10px sans-serif');
      if (bi >= 0) {
        const b = cfg[bi];
        txt(g, `after ${k} of ${n}: chosen = #${bi + 1}, validation ${b.v.toFixed(3)}, test ${b.t.toFixed(3)}, optimism ${(b.v - b.t >= 0 ? '+' : '') + (b.v - b.t).toFixed(3)}`, 44, H - 26, c.fg, '12px sans-serif');
        txt(g, `Average optimism over 200 repeated studies: +${gap.toFixed(3)}`, 44, H - 8, c.muted, '11px sans-serif');
        out.innerHTML = `Best validation score so far: <b>${b.v.toFixed(3)}</b>. Fresh test rows give ${b.t.toFixed(3)}. The validation score is the maximum of ${k} noisy numbers, so it is biased upwards; the test score is not.`;
      } else out.textContent = 'Waiting for the first configuration.';
    };
    [ncEl, nvEl, spEl].forEach((e) => (e.oninput = build));
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = build; q(ctl, '#seed').onclick = () => { seed += 13; build(); };
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    build();
    AI.loop(root, (dt) => {
      if (playing && shown < cfg.length) { shown = Math.min(cfg.length, shown + dt * 2.5 * +spdEl.value); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 7. Grid versus random search ================= */
  AI.labs['ds-md-search'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Grid search versus random search', 'illustrative: the score depends strongly on x and only a little on y',
      `<canvas style="height:300px"></canvas>`,
      `${sl('bud', 'Budget (number of trials)', 4, 36, 1, 9)}
       ${sl('imp', 'How much y matters', 0, 1, 0.1, 0.1)}
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Restart</button><button class="btn" id="seed">New seed</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Bright = better score. Both panels get the same budget. Grid uses a fixed lattice, so only a few distinct x values are ever tried; random draws a fresh x and y each time, so it tries as many distinct x values as it has trials. Squares: grid. Circles: random.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const bEl = q(ctl, '#bud'), iEl = q(ctl, '#imp'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    let seed = 3, G = [], Rn = [], shown = 0, playing = true, dirty = true;
    const score = (x, y, w) => 0.6 + 0.3 * Math.exp(-(((x - 0.63) / 0.1) ** 2)) + w * 0.1 * Math.exp(-(((y - 0.4) / 0.2) ** 2)) - (w * 0.05);
    const build = () => {
      const b = +bEl.value, side = Math.max(1, Math.floor(Math.sqrt(b))), r = AI.rng(seed); G = []; Rn = [];
      for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) G.push([(i + 0.5) / side, (j + 0.5) / side]);
      const extra = b - side * side; // budget not fitting the lattice is unused by grid (it cannot extend a lattice)
      for (let i = 0; i < b; i++) Rn.push([r(), r()]);
      shown = 0; dirty = true; build.extra = extra;
      q(ctl, '#budv').textContent = b; q(ctl, '#impv').textContent = (+iEl.value).toFixed(1); q(ctl, '#spdv').textContent = spdEl.value + '×';
    };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H); const w = +iEl.value, k = Math.floor(shown);
      const pw = (W - 40) / 2, ph = H - 70, res = [];
      [['Grid search', G, 0], ['Random search', Rn, 1]].forEach(([name, pts, pi]) => {
        const x0 = 14 + pi * (pw + 12), y0 = 24, cells = 36, cw = pw / cells, chh = ph / cells;
        for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) { const s = score((i + 0.5) / cells, 1 - (j + 0.5) / cells, w), t = clamp((s - 0.55) / 0.4, 0, 1); g.fillStyle = `rgba(88,166,255,${0.08 + 0.7 * t})`; g.fillRect(x0 + i * cw, y0 + j * chh, cw + 0.5, chh + 0.5); }
        frame(g, c, x0, y0, pw, ph); txt(g, name, x0, y0 - 6, c.fg, 'bold 12px sans-serif');
        let best = -1, bp = null; const used = pts.slice(0, Math.min(k, pts.length)), xs = new Set();
        used.forEach((p) => { const s = score(p[0], p[1], w); xs.add(p[0].toFixed(4)); if (s > best) { best = s; bp = p; } });
        used.forEach((p) => { const X = x0 + p[0] * pw, Y = y0 + (1 - p[1]) * ph; g.fillStyle = c.orange; g.strokeStyle = c.bg; if (pi === 0) { g.fillRect(X - 4, Y - 4, 8, 8); } else { g.beginPath(); g.arc(X, Y, 4.5, 0, 7); g.fill(); } });
        if (bp) { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.arc(x0 + bp[0] * pw, y0 + (1 - bp[1]) * ph, 9, 0, 7); g.stroke(); g.lineWidth = 1; }
        txt(g, 'hyperparameter x (important) →', x0 + pw / 2, y0 + ph + 14, c.muted, '10px sans-serif', 'center');
        txt(g, `trials ${used.length} · distinct x ${xs.size} · best score ${best > 0 ? best.toFixed(3) : '–'}`, x0, y0 + ph + 30, c.fg, '11px sans-serif');
        res.push({ best, xs: xs.size });
      });
      out.innerHTML = `Peak possible score is ${score(0.63, 0.4, w).toFixed(3)} (at x = 0.63). After ${Math.min(k, Rn.length)} trials: grid best <b>${res[0].best > 0 ? res[0].best.toFixed(3) : '–'}</b> from ${res[0].xs} distinct x values; random best <b>${res[1].best > 0 ? res[1].best.toFixed(3) : '–'}</b> from ${res[1].xs}.` + (build.extra > 0 ? ` (The grid uses ${G.length} of the ${Rn.length} trials: a lattice needs a square number.)` : '');
    };
    [bEl, iEl].forEach((e) => (e.oninput = build));
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = build; q(ctl, '#seed').onclick = () => { seed += 17; build(); };
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    build();
    AI.loop(root, (dt) => {
      if (playing && shown < Rn.length) { shown = Math.min(Rn.length, shown + dt * 1.6 * +spdEl.value); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ================= 8. Permutation importance and partial dependence ================= */
  AI.labs['ds-md-explain'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Permutation importance and partial dependence', 'illustrative: a fixed model f(a,b,c) = 4a + 6b² + w·c on 120 rows; the truth never uses c',
      `<canvas style="height:360px"></canvas>`,
      `${sl('wc', 'Weight w the model puts on c', 0, 4, 0.5, 0)}
       <label>Partial dependence for feature <select id="feat"><option value="0">a</option><option value="1" selected>b</option><option value="2">c</option></select></label>
       ${spd()}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">One shuffle</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Illustrative. Left: each round shuffles one column at a time and records how much the model's mean absolute error (MAE) rises; bars show the running average over rounds, dots are single rounds. Right: partial dependence of the chosen feature, found by setting it to each value for every row and averaging the predictions (thin lines are 20 individual rows; the thick line is the average). The model is fixed; only w changes what it reads.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const wEl = q(ctl, '#wc'), fEl = q(ctl, '#feat'), spdEl = q(ctl, '#spd'), out = q(ctl, '#out');
    const N = 120, r0 = AI.rng(41), D = []; for (let i = 0; i < N; i++) D.push([r0(), r0(), r0()]);
    const Y = D.map((d) => 4 * d[0] + 6 * d[1] * d[1] + 0.4 * AI.randn(r0));
    const f = (a, b, c, w) => 4 * a + 6 * b * b + w * c;
    let rounds, sums, last, playing = true, dirty = true, t = 0, sweep = 0, rr;
    const mae = (rows, w) => mean(rows.map((d, i) => Math.abs(Y[i] - f(d[0], d[1], d[2], w))));
    const reset = () => { rounds = 0; sums = [0, 0, 0]; last = [[], [], []]; rr = AI.rng(77); t = 0; sweep = 0; dirty = true; q(ctl, '#wcv').textContent = (+wEl.value).toFixed(1); q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    const round = () => {
      const w = +wEl.value, base = mae(D, w);
      for (let j = 0; j < 3; j++) { const p = shuffle(D.map((d) => d[j]), rr), rows = D.map((d, i) => { const e = d.slice(); e[j] = p[i]; return e; }); const inc = mae(rows, w) - base; sums[j] += inc; last[j].push(inc); }
      rounds++; dirty = true;
    };
    const draw = () => {
      const g = o.ctx, c = AI.col(), W = o.w, H = o.h; g.clearRect(0, 0, W, H); const w = +wEl.value, base = mae(D, w), cols = [c.blue, c.orange, c.purple], names = ['a', 'b', 'c'];
      const half = W * 0.42, avg = sums.map((s) => (rounds ? s / rounds : 0)), mx = Math.max(1, ...avg, ...last.flat()) * 1.1;
      txt(g, `Permutation importance · ${rounds} rounds`, 12, 18, c.fg, 'bold 12px sans-serif');
      const bx = 40, bw = half - 56, by = 40, bh = 36;
      for (let j = 0; j < 3; j++) { const y = by + j * (bh + 16); txt(g, names[j], bx - 8, y + bh / 2 + 4, c.fg, 'bold 13px sans-serif', 'right');
        g.fillStyle = cols[j]; g.globalAlpha = 0.8; g.fillRect(bx, y, (avg[j] / mx) * bw, bh); g.globalAlpha = 1;
        last[j].forEach((v) => { g.fillStyle = c.fg; g.beginPath(); g.arc(bx + (v / mx) * bw, y + bh / 2, 2.5, 0, 7); g.fill(); });
        txt(g, '+' + avg[j].toFixed(3), bx + (avg[j] / mx) * bw + 6, y + bh / 2 + 4, c.fg, '11px monospace'); }
      txt(g, `error rise when shuffled (baseline MAE ${base.toFixed(3)})`, bx - 30, by + 3 * (bh + 16) + 6, c.muted, '10px sans-serif');
      txt(g, avg[2] > 0.02 ? 'The model leans on c although c carries no real signal.' : 'Shuffling c changes nothing: the model ignores c.', 12, H - 26, avg[2] > 0.02 ? c.red : c.green, '11px sans-serif');
      // PDP panel
      const j = +fEl.value, x0 = half + 40, pw = W - x0 - 14, py = 34, ph = H - 80, vals = D.map((d) => d[j]);
      const gmin = 0, gmax = 1, G = 25, pdp = [], ice = [];
      for (let k = 0; k <= G; k++) { const v = gmin + ((gmax - gmin) * k) / G; const preds = D.map((d) => { const e = d.slice(); e[j] = v; return f(e[0], e[1], e[2], w); }); pdp.push(mean(preds)); ice.push(preds); }
      const lo = Math.min(...pdp, ...ice.flat()), hi = Math.max(...pdp, ...ice.flat()), P = mk(x0, py, pw, ph, 0, 1, lo, hi); frame(g, c, x0, py, pw, ph);
      txt(g, `Partial dependence of the prediction on ${names[j]}`, x0, py - 8, c.fg, 'bold 12px sans-serif');
      const upto = Math.floor(sweep);
      g.strokeStyle = c.muted; g.globalAlpha = 0.35; for (let i = 0; i < 20; i++) { g.beginPath(); for (let k = 0; k <= Math.min(upto, G); k++) { const xx = P.X(k / G), yy = P.Y(ice[k][i]); k ? g.lineTo(xx, yy) : g.moveTo(xx, yy); } g.stroke(); } g.globalAlpha = 1;
      g.strokeStyle = cols[j]; g.lineWidth = 3; g.beginPath(); for (let k = 0; k <= Math.min(upto, G); k++) { const xx = P.X(k / G), yy = P.Y(pdp[k]); k ? g.lineTo(xx, yy) : g.moveTo(xx, yy); } g.stroke(); g.lineWidth = 1;
      const sx = P.X(Math.min(upto, G) / G); g.strokeStyle = c.red; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(sx, py); g.lineTo(sx, py + ph); g.stroke(); g.setLineDash([]);
      txt(g, `${names[j]} set to ${(Math.min(upto, G) / G).toFixed(2)} for every row → average prediction ${pdp[Math.min(upto, G)].toFixed(2)}`, x0, py + ph + 18, c.fg, '11px sans-serif');
      txt(g, `${names[j]} →`, x0 + pw, py + ph - 4, c.muted, '10px sans-serif', 'right');
      out.innerHTML = `Importance (extra error): a ${avg[0].toFixed(3)}, b ${avg[1].toFixed(3)}, c ${avg[2].toFixed(3)}. Importance says what the <i>model</i> uses; it does not say what changes the outcome in the world.`;
    };
    wEl.oninput = reset; fEl.oninput = () => { sweep = 0; dirty = true; };
    spdEl.oninput = () => { q(ctl, '#spdv').textContent = spdEl.value + '×'; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { round(); };
    q(ctl, '#reset').onclick = reset;
    o.onresize = () => { dirty = true; }; themed(() => { dirty = true; });
    reset();
    AI.loop(root, (dt) => {
      const s = +spdEl.value;
      if (playing) { t += dt * s; if (t > 1.5 && rounds < 30) { t = 0; round(); } if (sweep < 25) { sweep = Math.min(25.99, sweep + dt * 4 * s); dirty = true; } else { sweep += dt * s * 0.0; } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
  };
})();
