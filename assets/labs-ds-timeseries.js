/* Data Science Module 8 labs: time series & forecasting.
   All series are simulated with seeded random numbers, so every run can be repeated. The labs are ILLUSTRATIVE:
   they stand in for real demand, energy or traffic data. Animations are dt-based, slow by default, and have
   play/pause, step, reset and a speed slider. Lab names are prefixed ds-ts-. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const sel = (id, label, opts) => `<label>${label}<select id="${id}">${opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select></label>`;
  const chk = (id, label, on) => `<label style="flex-direction:row;align-items:center;gap:8px"><input type="checkbox" id="${id}"${on ? ' checked' : ''}> ${label}</label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  const nf = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  const tx = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '11px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const SH0 = [-0.6, -0.9, -0.4, 0.2, 0.6, 0.9, 1, 0.7, 0.1, -0.3, -0.2, -0.8];
  const SH = SH0.map((v) => v - mean(SH0));            // a seasonal shape that sums to zero over 12 steps
  const noiseArr = (n, seed) => { const r = AI.rng(seed), a = []; for (let i = 0; i < n; i++) a.push(AI.randn(r)); return a; };

  /* a framed plot area; returns data->pixel maps */
  function frame(g, c, b, x0, x1, y0, y1, label) {
    const X = (v) => b.l + ((v - x0) / (x1 - x0)) * b.w, Y = (v) => b.t + b.h - ((v - y0) / (y1 - y0)) * b.h;
    g.strokeStyle = c.border; g.lineWidth = 1; g.strokeRect(b.l + 0.5, b.t + 0.5, b.w, b.h);
    for (let i = 0; i <= 2; i++) { const v = y0 + ((y1 - y0) * i) / 2; tx(g, nf(v), b.l - 4, Y(v) + 3, c.muted, '10px sans-serif', 'right'); if (i === 1) { g.strokeStyle = c.border; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(b.l, Y(v)); g.lineTo(b.l + b.w, Y(v)); g.stroke(); g.globalAlpha = 1; } }
    if (label) tx(g, label, b.l + 6, b.t + 13, c.muted, '11px sans-serif');
    return { X, Y };
  }
  /* polyline of arr[i] at x=i for i0..i1, breaking at null/NaN */
  function ln(g, arr, X, Y, col, lw, i0, i1, dash) {
    g.strokeStyle = col; g.lineWidth = lw || 1.5; g.setLineDash(dash || []); g.beginPath(); let pen = false;
    for (let i = i0; i <= i1 && i < arr.length; i++) { const v = arr[i]; if (v == null || !Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); pen = true; }
    g.stroke(); g.setLineDash([]); g.lineWidth = 1;
  }
  const dot = (g, x, y, r, col) => { g.fillStyle = col; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); };
  const range = (arrs, pad) => { let lo = Infinity, hi = -Infinity; arrs.forEach((a) => a.forEach((v) => { if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } })); const p = (hi - lo) * (pad == null ? 0.08 : pad) || 1; return [lo - p, hi + p]; };
  /* sample autocorrelation at lag k (mean removed, divided by the lag-0 sum of squares) */
  function acf(x, k) { const m = mean(x); let n0 = 0, nk = 0; x.forEach((v) => (n0 += (v - m) * (v - m))); for (let i = k; i < x.length; i++) nk += (x[i] - m) * (x[i - k] - m); return n0 > 0 ? nk / n0 : 0; }
  /* generic play/step/reset binding. st = {playing, speed}; step(n) advances n items; reset() restarts */
  function bindPlay(ctl, st, stepOnce, reset) {
    const pb = q(ctl, '#play');
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; stepOnce(); };
    q(ctl, '#reset').onclick = () => { reset(); };
  }
  const pauseUI = (ctl, st) => { st.playing = false; q(ctl, '#play').textContent = '▶ Play'; };

  /* ===================== 1a. decomposition ===================== */
  AI.labs['ds-ts-decompose'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Decompose a series: trend, season, residual', 'a monthly series is rebuilt as it arrives, then split into three parts (illustrative)',
      `<canvas style="height:440px"></canvas>`,
      `${sel('mode', 'Model', [['add', 'Additive: y = trend + season + noise'], ['mul', 'Multiplicative: y = trend × season × noise']])}
       ${sl('slope', 'Trend slope per month', 0, 1.2, 0.05, 0.5)}
       ${sl('amp', 'Season strength', 0, 10, 0.5, 5)}
       ${sl('sd', 'Noise level', 0, 5, 0.25, 1.5)}
       ${sl('sp', 'Speed (months per second)', 0.5, 8, 0.5, 3)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 72, Z = noiseArr(N, 11);
    const st = { playing: true }; let pos = 0, y, T, S, E, valid;
    const g$ = (id) => +q(ctl, '#' + id).value;
    const build = () => {
      const mode = q(ctl, '#mode').value, sl_ = g$('slope'), am = g$('amp'), sd = g$('sd');
      y = []; for (let t = 0; t < N; t++) { const base = (mode === 'add' ? 20 : 30) + sl_ * t; y.push(mode === 'add' ? base + am * SH[t % 12] * 1 + sd * Z[t] : base * (1 + 0.05 * am * SH[t % 12]) * (1 + 0.01 * sd * Z[t])); }
      T = y.map((_, i) => { if (i < 6 || i > N - 7) return null; let s = 0.5 * y[i - 6] + 0.5 * y[i + 6]; for (let j = i - 5; j <= i + 5; j++) s += y[j]; return s / 12; });
      const raw = [], cnt = []; for (let p = 0; p < 12; p++) { raw[p] = 0; cnt[p] = 0; }
      y.forEach((v, i) => { if (T[i] != null) { raw[i % 12] += mode === 'add' ? v - T[i] : v / T[i]; cnt[i % 12]++; } });
      const ph = raw.map((v, p) => v / cnt[p]), m = mean(ph);
      const sidx = ph.map((v) => (mode === 'add' ? v - m : v / m));
      S = y.map((_, i) => sidx[i % 12]);
      E = y.map((v, i) => (T[i] == null ? null : mode === 'add' ? v - T[i] - S[i] : v / (T[i] * S[i])));
      valid = E.filter((v) => v != null);
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h, mode = q(ctl, '#mode').value; g.clearRect(0, 0, W, H);
      const r = Math.floor(pos), l = 44, w = W - l - 12, hs = [0.34, 0.2, 0.2, 0.2];
      const tops = []; let acc = 8; hs.forEach((h) => { tops.push(acc); acc += h * (H - 30) + 6; });
      const Y0 = range([y]), x1 = N - 1;
      let f = frame(g, c, { l, t: tops[0], w, h: hs[0] * (H - 30) }, 0, x1, Y0[0], Y0[1], 'observed series (blue) and its trend estimate (orange)');
      ln(g, y, f.X, f.Y, c.blue, 1.6, 0, r); ln(g, T, f.X, f.Y, c.orange, 2.2, 0, r - 6);
      dot(g, f.X(Math.min(r, N - 1)), f.Y(y[Math.min(r, N - 1)]), 3.5, c.blue);
      const Tr = range([T], 0.08);
      f = frame(g, c, { l, t: tops[1], w, h: hs[1] * (H - 30) }, 0, x1, Tr[0], Tr[1], 'trend: centred 12-month average');
      ln(g, T, f.X, f.Y, c.orange, 2, 0, r - 6);
      const Sv = S.map((v) => (mode === 'add' ? v : v - 1)), Sm = Math.max(...Sv.map(Math.abs), 0.01) * 1.25;
      f = frame(g, c, { l, t: tops[2], w, h: hs[2] * (H - 30) }, 0, x1, (mode === 'add' ? 0 : 1) - Sm, (mode === 'add' ? 0 : 1) + Sm, 'seasonal pattern: the same 12 numbers repeating');
      ln(g, S, f.X, f.Y, c.green, 2, 0, r - 6);
      const Em = Math.max(...valid.map((v) => Math.abs(v - (mode === 'add' ? 0 : 1))), 0.01) * 1.25, e0 = mode === 'add' ? 0 : 1;
      f = frame(g, c, { l, t: tops[3], w, h: hs[3] * (H - 30) }, 0, x1, e0 - Em, e0 + Em, 'residual: what is left (should look like noise)');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, f.Y(e0)); g.lineTo(l + w, f.Y(e0)); g.stroke();
      E.forEach((v, i) => { if (v != null && i <= r - 6) dot(g, f.X(i), f.Y(v), 2.5, c.purple); });
      tx(g, 'month', l + w, H - 6, c.muted, '10px sans-serif', 'right');
      const m = mean(valid.map((v) => (mode === 'add' ? v : v - 1))), s = Math.sqrt(mean(valid.map((v) => (v - (mode === 'add' ? 0 : 1)) ** 2)));
      q(ctl, '#out').textContent = `month ${r + 1} of ${N} revealed\nseasonal range: ${(mode === 'add' ? Math.min(...S).toFixed(2) + ' to ' + Math.max(...S).toFixed(2) : Math.min(...S).toFixed(3) + ' to ' + Math.max(...S).toFixed(3) + ' (ratio)')}\nresidual spread (sd): ${s.toFixed(3)}`;
      q(ctl, '#cap').textContent = r < 6 ? 'The series arrives one month at a time. A centred average needs 6 months on each side, so the trend appears 6 months behind.' : (mode === 'add' ? 'Additive: the seasonal swing has the same size at every level, so the three parts add.' : 'Multiplicative: the seasonal swing grows with the trend, so the parts multiply (see the widening wiggles).');
      ['slope', 'amp', 'sd', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['mode', 'slope', 'amp', 'sd'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { pos = Math.min(N - 1, Math.floor(pos) + 1); draw(); }, () => { pos = 0; st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= N - 1) return; pos = Math.min(N - 1, pos + dt * g$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 1b. autocorrelation ===================== */
  AI.labs['ds-ts-acf'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Autocorrelation: how a series resembles its own past', 'the lag k slides along; the scatter and the ACF bar answer the same question (illustrative)',
      `<canvas style="height:420px"></canvas>`,
      `${sel('type', 'Series', [['noise', 'Pure noise'], ['trend', 'Upward trend + noise'], ['season', 'Season of length 12 + noise'], ['ar1', 'AR(1): each value remembers the last']])}
       ${sl('phi', 'AR(1) memory φ (used by the AR(1) series)', 0, 0.95, 0.05, 0.8)}
       ${sl('lag', 'Lag k', 1, 24, 1, 1)}
       ${sl('sp', 'Sweep speed (lags per 10 s)', 1, 12, 1, 4)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 96, Z = noiseArr(N + 50, 5);
    const st = { playing: true }; let kf = 1, x = [];
    const build = () => {
      const t = q(ctl, '#type').value, phi = +q(ctl, '#phi').value; x = [];
      for (let i = 0; i < N; i++) {
        if (t === 'noise') x.push(50 + 2 * Z[i]);
        else if (t === 'trend') x.push(30 + 0.3 * i + 2 * Z[i]);
        else if (t === 'season') x.push(50 + 6 * SH[i % 12] * 1.6 + 1.5 * Z[i]);
        else x.push(50);
      }
      if (t === 'ar1') { let p = 0; x = []; for (let i = 0; i < N + 20; i++) { p = phi * p + 2 * Z[i]; if (i >= 20) x.push(50 + p); } }
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h, k = +q(ctl, '#lag').value; g.clearRect(0, 0, W, H);
      const l = 40, topH = H * 0.36, [y0, y1] = range([x]);
      let f = frame(g, c, { l, t: 6, w: W - l - 10, h: topH }, 0, N - 1, y0, y1, 'the series; orange = value now, blue = value k steps earlier');
      ln(g, x, f.X, f.Y, c.muted, 1.2, 0, N - 1);
      const show = [N - 1 - 18, N - 1 - 9, N - 1];                                  // three example pairs near the end
      show.forEach((i) => { if (i - k >= 0) { dot(g, f.X(i), f.Y(x[i]), 4, c.orange); dot(g, f.X(i - k), f.Y(x[i - k]), 4, c.blue); g.strokeStyle = c.border; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(f.X(i - k), f.Y(x[i - k])); g.lineTo(f.X(i), f.Y(x[i])); g.stroke(); g.setLineDash([]); } });
      // scatter
      const sb = { l: l, t: topH + 34, w: (W - l - 30) * 0.42, h: H - topH - 70 };
      const xs = x.slice(0, N - k), ys = x.slice(k), sx = range([x]);
      const fs = frame(g, c, sb, sx[0], sx[1], sx[0], sx[1], 'y(t−k) across, y(t) up');
      g.globalAlpha = 0.5; for (let i = 0; i < xs.length; i++) dot(g, fs.X(xs[i]), fs.Y(ys[i]), 2.5, c.blue); g.globalAlpha = 1;
      const r = acf(x, k), m = mean(x), sd2 = Math.sqrt(mean(x.map((v) => (v - m) ** 2)));
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(fs.X(sx[0]), fs.Y(m + r * (sx[0] - m))); g.lineTo(fs.X(sx[1]), fs.Y(m + r * (sx[1] - m))); g.stroke(); g.lineWidth = 1;
      // ACF bars
      const ab = { l: sb.l + sb.w + 50, t: sb.t, w: W - (sb.l + sb.w + 50) - 10, h: sb.h };
      const fa = frame(g, c, ab, 0.5, 24.5, -1, 1, 'ACF: bar k = correlation at lag k');
      const band = 1.96 / Math.sqrt(N);
      g.fillStyle = c.border; g.globalAlpha = 0.5; g.fillRect(ab.l, fa.Y(band), ab.w, fa.Y(-band) - fa.Y(band)); g.globalAlpha = 1;
      const bw = Math.max(2, ab.w / 24 - 3);
      for (let j = 1; j <= 24; j++) { const rj = acf(x, j); g.fillStyle = j === k ? c.orange : c.blue; g.fillRect(fa.X(j) - bw / 2, Math.min(fa.Y(0), fa.Y(rj)), bw, Math.abs(fa.Y(rj) - fa.Y(0))); }
      tx(g, 'lag', ab.l + ab.w, ab.t + ab.h + 14, c.muted, '10px sans-serif', 'right');
      q(ctl, '#out').textContent = `lag k = ${k}\ncorrelation r(${k}) = ${r.toFixed(3)}\nnoise band ±${band.toFixed(3)} (grey)`;
      q(ctl, '#cap').textContent = Math.abs(r) < band ? `At lag ${k} the bar is inside the grey band: no more resemblance than pure chance would give.` : r > 0 ? `At lag ${k} the series tends to be on the same side of its average as ${k} steps ago (the scatter slopes up).` : `At lag ${k} the series tends to be on the opposite side of its average from ${k} steps ago (the scatter slopes down).`;
      ['phi', 'lag', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    const setK = (v) => { kf = Math.max(1, Math.min(24.99, v)); q(ctl, '#lag').value = Math.floor(kf); };
    q(ctl, '#lag').oninput = () => { kf = +q(ctl, '#lag').value; pauseUI(ctl, st); draw(); };
    ['type', 'phi'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { setK(Math.floor(kf) + 1 > 24 ? 1 : Math.floor(kf) + 1); draw(); }, () => { setK(1); st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing) return; const before = Math.floor(kf); kf += dt * (+q(ctl, '#sp').value) / 10; if (kf >= 25) kf = 1; q(ctl, '#lag').value = Math.floor(kf); if (Math.floor(kf) !== before) draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 2. smoothing ===================== */
  AI.labs['ds-ts-smooth'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Moving average and exponential smoothing', 'points arrive one at a time; three smoothers chase the signal (illustrative)',
      `<canvas style="height:380px"></canvas>`,
      `${sel('sc', 'The world is', [['level', 'Steady level + noise'], ['step', 'Level that jumps at t = 50'], ['trend', 'A steady upward trend']])}
       ${sl('w', 'Moving-average window w', 1, 20, 1, 5)}
       ${sl('a', 'Smoothing weight α (level)', 0.05, 1, 0.05, 0.3)}
       ${sl('b', 'Trend weight β (Holt)', 0, 1, 0.05, 0.2)}
       ${sl('sd', 'Noise level', 0, 6, 0.5, 3)}
       ${sl('sp', 'Speed (points per second)', 1, 12, 1, 4)}
       ${chk('shMA', 'Show moving average (orange)', true)}${chk('shSES', 'Show simple exponential smoothing (green)', true)}${chk('shH', 'Show Holt trend method (purple)', false)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 100, Z = noiseArr(N, 21), st = { playing: true }; let pos = 2, y = [];
    const v$ = (id) => +q(ctl, '#' + id).value;
    const build = () => { const sc = q(ctl, '#sc').value, sd = v$('sd'); y = Z.map((z, t) => (sc === 'level' ? 50 : sc === 'step' ? (t < 50 ? 50 : 60) : 40 + 0.25 * t) + sd * z); };
    const compute = (r) => {
      const w = v$('w'), a = v$('a'), be = v$('b'), MA = [], L = [], H = [], HF = [];
      let l = y[0], hl = y[0], hb = y[1] - y[0];
      for (let t = 0; t < r; t++) {
        MA.push(t >= w - 1 ? mean(y.slice(t - w + 1, t + 1)) : null);
        if (t > 0) { l = a * y[t] + (1 - a) * l; }
        L.push(l);
        if (t === 0) { H.push(y[0]); HF.push(null); } else { const pred = hl + hb; HF.push(pred); const nl = a * y[t] + (1 - a) * (hl + hb); hb = be * (nl - hl) + (1 - be) * hb; hl = nl; H.push(hl); }
      }
      return { MA, L, H, HF, hl, hb, l };
    };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h, r = Math.floor(pos); g.clearRect(0, 0, W, Hh);
      const k = compute(r), [y0, y1] = range([y]);
      const f = frame(g, c, { l: 40, t: 8, w: W - 52, h: Hh - 34 }, 0, N + 10, y0, y1, '');
      for (let i = 0; i < r; i++) dot(g, f.X(i), f.Y(y[i]), 2.6, c.muted);
      if (q(ctl, '#shMA').checked) ln(g, k.MA, f.X, f.Y, c.orange, 2.2, 0, r - 1);
      if (q(ctl, '#shSES').checked) { ln(g, k.L, f.X, f.Y, c.green, 2.2, 0, r - 1); g.strokeStyle = c.green; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(f.X(r - 1), f.Y(k.l)); g.lineTo(f.X(r + 9), f.Y(k.l)); g.stroke(); g.setLineDash([]); }
      if (q(ctl, '#shH').checked) { ln(g, k.H, f.X, f.Y, c.purple, 2.2, 0, r - 1); g.strokeStyle = c.purple; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(f.X(r - 1), f.Y(k.hl)); g.lineTo(f.X(r + 9), f.Y(k.hl + 10 * k.hb)); g.stroke(); g.setLineDash([]); }
      tx(g, 'time', 40 + W - 52, Hh - 6, c.muted, '10px sans-serif', 'right'); tx(g, 'dashed = forecast of the next 10 steps', 48, 22, c.muted);
      const err = (arr, from) => { let s = 0, n = 0; for (let t = from; t < r; t++) if (arr[t] != null) { s += Math.abs(y[t] - arr[t]); n++; } return n ? (s / n).toFixed(2) : '–'; };
      const w = v$('w'), maPred = k.MA.map((v, i) => (i > 0 ? k.MA[i - 1] : null)), sesPred = k.L.map((v, i) => (i > 0 ? k.L[i - 1] : null));
      q(ctl, '#out').textContent = `t = ${r}, latest value ${y[r - 1].toFixed(1)}\nMA(${w}) now ${k.MA[r - 1] == null ? '–' : k.MA[r - 1].toFixed(2)}   one-step MAE ${err(maPred, 1)}\nSES now ${k.l.toFixed(2)}   one-step MAE ${err(sesPred, 1)}\nHolt level ${k.hl.toFixed(2)}, trend ${k.hb.toFixed(3)}/step   MAE ${err(k.HF, 1)}`;
      const sc = q(ctl, '#sc').value;
      q(ctl, '#cap').textContent = sc === 'step' ? 'After the jump at t = 50 the moving average needs about w steps to catch up; SES catches up at a pace set by α. Large α reacts fast but also chases noise.' : sc === 'trend' ? 'With a trend, a flat smoother (orange, green) always lags below the data. Switch on Holt: it estimates a slope and keeps up.' : 'With a steady level, a small α or a long window gives a calm estimate; a large α or a short window jitters with the noise.';
      ['w', 'a', 'b', 'sd', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['sc', 'sd'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    ['w', 'a', 'b', 'sp', 'shMA', 'shSES', 'shH'].forEach((id) => (q(ctl, '#' + id).oninput = draw));
    bindPlay(ctl, st, () => { pos = Math.min(N, Math.floor(pos) + 1); draw(); }, () => { pos = 2; st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= N) return; pos = Math.min(N, pos + dt * v$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 3. AR(1), differencing and the ACF ===================== */
  AI.labs['ds-ts-arima'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Autoregression and differencing', 'simulate a series, forecast each next value from the last one, and read the ACF (illustrative)',
      `<canvas style="height:420px"></canvas>`,
      `${sel('type', 'Series', [['ar', 'Stationary AR(1) around 50'], ['rw', 'Random walk with upward drift']])}
       ${sl('phi', 'True memory φ (AR(1) series)', -0.9, 0.95, 0.05, 0.7)}
       ${chk('d', 'Difference once (d = 1): plot y(t) − y(t−1)', false)}
       ${sl('sp', 'Speed (points per second)', 1, 12, 1, 4)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 120, Z = noiseArr(N + 30, 33), st = { playing: true }; let pos = 12, base = [];
    const v$ = (id) => +q(ctl, '#' + id).value;
    const build = () => {
      const t = q(ctl, '#type').value, phi = v$('phi'); base = []; let p = 0, lv = 50;
      for (let i = 0; i < N + 20; i++) { p = phi * p + 2 * Z[i]; lv += 0.4 + 1.2 * Z[i]; if (i >= 20) base.push(t === 'ar' ? 50 + p : lv); }
    };
    const plotted = () => (q(ctl, '#d').checked ? base.slice(1).map((v, i) => v - base[i]) : base);
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h, x = plotted(), r = Math.min(Math.floor(pos), x.length); g.clearRect(0, 0, W, H);
      const rev = x.slice(0, r), mu = mean(rev), phat = acf(rev, 1);
      const [y0, y1] = range([x]);
      const f = frame(g, c, { l: 40, t: 8, w: W - 52, h: H * 0.5 }, 0, x.length - 1, y0, y1, q(ctl, '#d').checked ? 'differenced series  y(t) − y(t−1)' : 'series  y(t)   (grey dots = one-step forecasts)');
      ln(g, x, f.X, f.Y, c.blue, 1.8, 0, r - 1);
      let se = 0, sn = 0, n = 0;
      for (let i = 1; i < r; i++) { const fc = mu + phat * (x[i - 1] - mu); dot(g, f.X(i), f.Y(fc), 2.4, c.orange); se += (x[i] - fc) ** 2; sn += (x[i] - x[i - 1]) ** 2; n++; }
      const nextFc = mu + phat * (x[r - 1] - mu); g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.arc(f.X(r), f.Y(nextFc), 5, 0, 7); g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(f.X(0), f.Y(mu)); g.lineTo(f.X(x.length - 1), f.Y(mu)); g.stroke(); g.setLineDash([]);
      const ab = { l: 40, t: H * 0.5 + 38, w: W - 52, h: H - (H * 0.5 + 38) - 22 };
      const fa = frame(g, c, ab, 0.5, 20.5, -1, 1, 'ACF of what is plotted (blue bars); orange line = φ^k if AR(1)');
      const band = 1.96 / Math.sqrt(Math.max(r, 2)); g.fillStyle = c.border; g.globalAlpha = 0.5; g.fillRect(ab.l, fa.Y(band), ab.w, fa.Y(-band) - fa.Y(band)); g.globalAlpha = 1;
      const bw = Math.max(3, ab.w / 20 - 5);
      for (let k = 1; k <= 20; k++) { const rk = r > k + 2 ? acf(rev, k) : 0; g.fillStyle = c.blue; g.fillRect(fa.X(k) - bw / 2, Math.min(fa.Y(0), fa.Y(rk)), bw, Math.abs(fa.Y(rk) - fa.Y(0))); }
      if (q(ctl, '#type').value === 'ar' && !q(ctl, '#d').checked) { const ph = v$('phi'); g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); for (let k = 1; k <= 20; k++) { const px = fa.X(k), py = fa.Y(Math.pow(ph, k)); k === 1 ? g.moveTo(px, py) : g.lineTo(px, py); } g.stroke(); g.lineWidth = 1; }
      tx(g, 'lag', ab.l + ab.w, ab.t + ab.h + 14, c.muted, '10px sans-serif', 'right');
      const rm = n ? Math.sqrt(se / n) : 0, rn = n ? Math.sqrt(sn / n) : 0;
      q(ctl, '#out').textContent = `${r} points so far\nestimated φ̂ = r(1) = ${phat.toFixed(3)}, mean ${mu.toFixed(2)}\nnext-step forecast (ring) ${nextFc.toFixed(2)}\none-step RMSE: AR ${rm.toFixed(2)}  vs  "same as last" ${rn.toFixed(2)}`;
      const ty = q(ctl, '#type').value, dd = q(ctl, '#d').checked;
      q(ctl, '#cap').textContent = ty === 'rw' && !dd ? 'A random walk wanders: its ACF stays high for many lags and its mean is not fixed. That slow decay is the sign it needs differencing.' : ty === 'rw' && dd ? 'After differencing, the random walk becomes steady noise around its drift: the ACF bars fall inside the band. This is the "I" in ARIMA.' : dd ? 'Differencing an already steady series over-corrects: r(1) turns negative. Only difference what needs it.' : 'For AR(1) the ACF decays like φ^k (orange). The forecast leans from the last value back toward the mean, with strength φ.';
      ['phi', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['type', 'phi', 'd'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { pos = Math.min(N - 1, Math.floor(pos) + 1); draw(); }, () => { pos = 12; st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= N - 1) return; pos = Math.min(N - 1, pos + dt * v$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 4a. rolling-origin backtest ===================== */
  AI.labs['ds-ts-backtest'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Rolling-origin backtest', 'the forecast origin moves forward; each step trains on the past only (illustrative series)',
      `<canvas style="height:420px"></canvas>`,
      `${sel('m', 'Forecast method', [['naive', 'Naive: repeat the last value'], ['snaive', 'Seasonal naive: repeat the same month last year'], ['drift', 'Drift: last value plus average slope'], ['mean', 'Mean of all history']])}
       ${sl('init', 'First training size (months)', 24, 84, 6, 48)}
       ${sl('h', 'Horizon h (months ahead)', 1, 12, 1, 6)}
       ${sl('sp', 'Origins per second', 0.25, 4, 0.25, 1)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 120, Z = noiseArr(N, 41), st = { playing: true };
    const y = Z.map((z, t) => 40 + 0.35 * t + 8 * SH[t % 12] * 1.5 + 2.5 * z);
    let origins = [], fcs = [], pos = 0, errSum = [], errCnt = [];
    const v$ = (id) => +q(ctl, '#' + id).value;
    const forecast = (m, o_, h) => {                    // forecast y[o_+h-1] using y[0..o_-1] only
      const hist = y.slice(0, o_), last = hist[o_ - 1];
      if (m === 'naive') return last; if (m === 'mean') return mean(hist);
      if (m === 'drift') return last + (h * (last - hist[0])) / (o_ - 1);
      return hist[o_ - 12 + ((h - 1) % 12)];
    };
    const build = () => {
      const m = q(ctl, '#m').value, i0 = v$('init'), h = v$('h'); origins = []; fcs = [];
      for (let o_ = i0; o_ <= N - 1; o_++) { const path = []; for (let k = 1; k <= h; k++) path.push(o_ + k - 1 < N ? forecast(m, o_, k) : null); origins.push(o_); fcs.push(path); }
      pos = 0; recount();
    };
    const recount = () => { const h = v$('h'); errSum = Array(h).fill(0); errCnt = Array(h).fill(0); const r = Math.floor(pos); for (let i = 0; i <= Math.min(r, origins.length - 1); i++) for (let k = 0; k < h; k++) { const t = origins[i] + k; if (t < N) { errSum[k] += Math.abs(y[t] - fcs[i][k]); errCnt[k]++; } } };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const r = Math.min(Math.floor(pos), origins.length - 1), org = origins[r], h = v$('h'), [y0, y1] = range([y]);
      recount();
      const f = frame(g, c, { l: 40, t: 8, w: W - 52, h: H * 0.55 }, 0, N - 1, y0, y1, '');
      g.fillStyle = c.blue; g.globalAlpha = 0.1; g.fillRect(f.X(0), f.Y(y1), f.X(org - 1) - f.X(0), f.Y(y0) - f.Y(y1)); g.globalAlpha = 1;
      g.strokeStyle = c.orange; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(f.X(org - 0.5), f.Y(y1)); g.lineTo(f.X(org - 0.5), f.Y(y0)); g.stroke(); g.setLineDash([]);
      ln(g, y, f.X, f.Y, c.muted, 1.2, org, N - 1); ln(g, y, f.X, f.Y, c.blue, 1.8, 0, org - 1);
      const fp = fcs[r].map((v, k) => (v == null ? null : v)); const arr = []; for (let k = 0; k < h; k++) arr[org + k] = fp[k];
      ln(g, arr, f.X, f.Y, c.orange, 2.4, org, org + h - 1);
      for (let k = 0; k < h; k++) if (org + k < N) { g.strokeStyle = c.red; g.beginPath(); g.moveTo(f.X(org + k), f.Y(y[org + k])); g.lineTo(f.X(org + k), f.Y(fp[k])); g.stroke(); dot(g, f.X(org + k), f.Y(y[org + k]), 3, c.fg); }
      tx(g, 'train on the past (blue)', f.X(1), 22, c.blue); tx(g, 'origin', f.X(org) + 4, 22, c.orange); tx(g, 'red = error at each horizon', f.X(org) + 4, 36, c.red);
      const bb = { l: 40, t: H * 0.55 + 42, w: W - 52, h: H - (H * 0.55 + 42) - 24 };
      const maes = errSum.map((s, k) => (errCnt[k] ? s / errCnt[k] : 0)), top = Math.max(...maes, 1) * 1.15;
      const fb = frame(g, c, bb, 0.5, h + 0.5, 0, top, 'mean absolute error so far, by horizon');
      const bw = Math.max(6, bb.w / h - 8);
      maes.forEach((v, k) => { g.fillStyle = c.orange; g.fillRect(fb.X(k + 1) - bw / 2, fb.Y(v), bw, fb.Y(0) - fb.Y(v)); tx(g, v.toFixed(1), fb.X(k + 1), fb.Y(v) - 3, c.fg, '10px sans-serif', 'center'); tx(g, 'h=' + (k + 1), fb.X(k + 1), bb.t + bb.h + 13, c.muted, '10px sans-serif', 'center'); });
      const tot = errSum.reduce((a, b) => a + b, 0), cn = errCnt.reduce((a, b) => a + b, 0);
      q(ctl, '#out').textContent = `origin ${r + 1} of ${origins.length}: training on the first ${org} months\nforecasting months ${org + 1} to ${Math.min(N, org + h)}\noverall MAE so far (${cn} forecasts): ${(tot / Math.max(1, cn)).toFixed(2)}`;
      q(ctl, '#cap').textContent = 'Each origin may use only data before the dashed line. The error bars on the right grow with horizon: the further ahead, the worse the forecast.';
      ['init', 'h', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['m', 'init', 'h'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { pos = Math.min(origins.length - 1, Math.floor(pos) + 1); draw(); }, () => { build(); st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= origins.length - 1) return; pos = Math.min(origins.length - 1, pos + dt * v$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 4b. shuffled versus time split ===================== */
  AI.labs['ds-ts-shuffle'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Why time must never be shuffled', 'the same model, scored two ways; only one score is honest (illustrative series)',
      `<canvas style="height:340px"></canvas>`,
      `${sel('sp_', 'How the test points are chosen', [['time', 'Honest: the last 20% of the timeline'], ['shuffle', 'Wrong: 20% picked at random (shuffled)']])}
       ${sl('sd', 'Noise level', 0, 5, 0.5, 2)}
       ${sl('seed', 'Random seed of the shuffle', 1, 20, 1, 3)}
       ${sl('sp', 'Test points per second', 0.5, 6, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 60, Z = noiseArr(N, 51), st = { playing: true };
    let y, test, trainIdx, res, pos = 0;
    const v$ = (id) => +q(ctl, '#' + id).value;
    const build = () => {
      const sd = v$('sd'); y = Z.map((z, t) => 30 + 0.5 * t + 6 * SH[t % 12] + sd * z);
      const nT = 12, mode = q(ctl, '#sp_').value;
      if (mode === 'time') test = Array.from({ length: nT }, (_, i) => N - nT + i);
      else { const r = AI.rng(v$('seed') * 97), idx = Array.from({ length: N }, (_, i) => i); for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } test = idx.slice(0, nT).sort((a, b) => a - b); }
      const ts = new Set(test); trainIdx = []; for (let i = 0; i < N; i++) if (!ts.has(i)) trainIdx.push(i);
      res = test.map((t) => {                           // the "model": a straight line through the nearest training points
        const before = trainIdx.filter((i) => i < t), after = trainIdx.filter((i) => i > t);
        let pred;
        if (before.length && after.length) { const a = before[before.length - 1], b = after[0]; pred = y[a] + ((y[b] - y[a]) * (t - a)) / (b - a); }       // sees the future: interpolation
        else { const a = before[before.length - 1], a2 = before[Math.max(0, before.length - 10)]; pred = y[a] + ((y[a] - y[a2]) / Math.max(1, a - a2)) * (t - a); }                            // only the past: extrapolation
        return { t, pred, err: Math.abs(y[t] - pred) };
      });
      pos = 0;
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h, r = Math.floor(pos); g.clearRect(0, 0, W, H);
      const [y0, y1] = range([y]), f = frame(g, c, { l: 40, t: 8, w: W - 52, h: H - 34 }, 0, N - 1, y0, y1, '');
      const ts = new Set(test);
      ln(g, y, f.X, f.Y, c.border, 1, 0, N - 1);
      for (let i = 0; i < N; i++) dot(g, f.X(i), f.Y(y[i]), ts.has(i) ? 3.2 : 2.6, ts.has(i) ? c.red : c.blue);
      let s = 0;
      for (let k = 0; k < Math.min(r, res.length); k++) { const e = res[k]; g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(f.X(e.t), f.Y(y[e.t])); g.lineTo(f.X(e.t), f.Y(e.pred)); g.stroke(); g.lineWidth = 1; dot(g, f.X(e.t), f.Y(e.pred), 3, c.orange); s += e.err; }
      tx(g, 'blue = train, red = test, orange = prediction and its error', 48, 22, c.muted);
      const n = Math.min(r, res.length), all = mean(res.map((e) => e.err));
      q(ctl, '#out').textContent = `test points scored: ${n} of ${res.length}\nmean absolute error so far: ${n ? (s / n).toFixed(2) : '–'}\nfinal error when finished: ${all.toFixed(2)}`;
      q(ctl, '#cap').textContent = q(ctl, '#sp_').value === 'time' ? 'Honest split: every test point lies after all training points, so the model must extrapolate, as it will in real use. The error is larger and real.' : 'Shuffled split: each test point sits between two training points, so the model interpolates between a past and a future neighbour. The error looks small, but in production there is no future neighbour.';
      ['sd', 'seed', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['sp_', 'sd', 'seed'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { pos = Math.min(res.length, Math.floor(pos) + 1); draw(); }, () => { build(); st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= res.length) return; pos = Math.min(res.length, pos + dt * v$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };

  /* ===================== 5. forecast fan ===================== */
  AI.labs['ds-ts-fan'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The forecast fan and its calibration', 'possible futures arrive one by one; the interval should catch the stated share of them (illustrative)',
      `<canvas style="height:400px"></canvas>`,
      `${sl('ts', 'True noise per step σ (what the world does)', 0.5, 3, 0.1, 1.5)}
       ${sl('as', 'Assumed noise per step σ (what your forecast uses)', 0.5, 3, 0.1, 1.5)}
       ${sel('lv', 'Interval level', [['0.5', '50%'], ['0.8', '80%'], ['0.95', '95%']])}
       ${sl('sp', 'New futures per second', 0.5, 8, 0.5, 2)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), HN = 30, NH = 40, st = { playing: true }; // HN horizon, NH history length
    const ZQ = { 0.5: 0.6745, 0.8: 1.2816, 0.95: 1.96 };
    const hist = (() => { const z = noiseArr(NH, 61), a = [60]; for (let i = 1; i < NH; i++) a.push(a[i - 1] + 0.3 + 1.5 * z[i]); return a; })();
    const last = hist[NH - 1], drift = 0.3;
    let paths = [], rnd = AI.rng(71), tot = 0, inside = 0, insideAll = 0, cellsAll = 0, acc = 0;
    const v$ = (id) => +q(ctl, '#' + id).value;
    const reset = () => { paths = []; tot = 0; inside = 0; insideAll = 0; cellsAll = 0; rnd = AI.rng(71); };
    const addPath = () => {
      const ts = v$('ts'), as = v$('as'), z = ZQ[q(ctl, '#lv').value], p = [last]; let all = 0;
      for (let h = 1; h <= HN; h++) { p.push(p[h - 1] + drift + ts * AI.randn(rnd)); const half = z * as * Math.sqrt(h), ok = Math.abs(p[h] - (last + drift * h)) <= half; if (ok) { insideAll++; all++; } cellsAll++; }
      tot++; if (Math.abs(p[HN] - (last + drift * HN)) <= z * as * Math.sqrt(HN)) inside++;
      paths.push(p); if (paths.length > 40) paths.shift();
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const as = v$('as'), lvl = q(ctl, '#lv').value, ys = [hist, [last - 3.5 * as * Math.sqrt(HN), last + drift * HN + 3.5 * as * Math.sqrt(HN)]];
      const [y0, y1] = range(ys, 0.04), f = frame(g, c, { l: 40, t: 8, w: W - 52, h: H - 34 }, 0, NH - 1 + HN, y0, y1, '');
      const bands = [['0.95', 0.12], ['0.8', 0.16], ['0.5', 0.22]];
      bands.forEach(([lv, al]) => { const z = ZQ[lv]; g.fillStyle = c.blue; g.globalAlpha = al; g.beginPath(); for (let h = 0; h <= HN; h++) { const px = f.X(NH - 1 + h), py = f.Y(last + drift * h + z * as * Math.sqrt(h)); h ? g.lineTo(px, py) : g.moveTo(px, py); } for (let h = HN; h >= 0; h--) g.lineTo(f.X(NH - 1 + h), f.Y(last + drift * h - z * as * Math.sqrt(h))); g.closePath(); g.fill(); g.globalAlpha = 1; });
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(f.X(NH - 1), f.Y(last)); g.lineTo(f.X(NH - 1 + HN), f.Y(last + drift * HN)); g.stroke(); g.lineWidth = 1;
      const z = ZQ[lvl]; g.strokeStyle = c.blue; g.lineWidth = 1.6; [1, -1].forEach((sg) => { g.beginPath(); for (let h = 0; h <= HN; h++) { const px = f.X(NH - 1 + h), py = f.Y(last + drift * h + sg * z * as * Math.sqrt(h)); h ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke(); }); g.lineWidth = 1;
      paths.forEach((p, i) => { g.globalAlpha = 0.15 + 0.5 * (i / paths.length); ln(g, p, (v) => f.X(NH - 1 + v), f.Y, c.green, 1.2, 0, HN); });
      g.globalAlpha = 1; ln(g, hist, f.X, f.Y, c.fg, 2, 0, NH - 1);
      tx(g, 'history', f.X(2), 22, c.fg); tx(g, 'orange = point forecast, shaded = 50 / 80 / 95% fan, green = simulated futures', f.X(NH + 1), 22, c.muted);
      const cov = tot ? inside / tot : 0, covAll = cellsAll ? insideAll / cellsAll : 0, nom = +lvl;
      const diff = tot > 20 ? covAll - nom : 0;
      q(ctl, '#out').textContent = `futures simulated: ${tot}\nhalf-width at h=${HN}: ±${(z * as * Math.sqrt(HN)).toFixed(2)} (grows like √h)\ninside the interval at h=${HN}: ${tot ? (100 * cov).toFixed(0) : '–'}%\ninside across all horizons: ${cellsAll ? (100 * covAll).toFixed(0) : '–'}%   target ${(100 * nom).toFixed(0)}%`;
      q(ctl, '#cap').textContent = tot < 20 ? 'Possible futures are being simulated one at a time. Watch the fraction caught by the interval settle.' : Math.abs(diff) < 0.04 ? 'Well calibrated: the share of futures inside the interval is close to its stated level.' : diff < 0 ? 'Over-confident: the interval is too narrow, so it catches fewer futures than it promises. Assumed σ is below the true σ.' : 'Under-confident: the interval is wider than needed, so it catches more than promised. Safe, but less useful for planning.';
      ['ts', 'as', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['ts', 'as', 'lv'].forEach((id) => (q(ctl, '#' + id).oninput = () => { reset(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { addPath(); draw(); }, () => { reset(); st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || tot >= 3000) return; acc += dt * v$('sp'); if (acc >= 1) { const n = Math.min(Math.floor(acc), 5); acc -= Math.floor(acc); for (let i = 0; i < n; i++) addPath(); draw(); } }, { fps: 30 });
    draw();
  };

  /* ===================== 6. lag features, windows and leakage ===================== */
  AI.labs['ds-ts-lags'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Lag features: turning a series into a table', 'a window slides along the series, one table row per step; then a model is scored on the future (illustrative)',
      `<canvas style="height:430px"></canvas>`,
      `${sl('k', 'Number of lags k', 1, 8, 1, 3)}
       ${sl('sd', 'Noise level', 0, 8, 0.5, 3)}
       ${chk('leak', 'Add a leaky feature: the 3-day average centred on the day being predicted', false)}
       ${sl('sp', 'Window steps per second', 0.5, 6, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const o = AI.canvas(q(view, 'canvas')), c = AI.col(), N = 56, Z = noiseArr(N, 81), st = { playing: true }; let y, pos = 0, rows;
    const v$ = (id) => +q(ctl, '#' + id).value;
    const solve = (A, b) => { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < n; i++) { let p = i; for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[p][i])) p = j; [M[i], M[p]] = [M[p], M[i]]; for (let j = i + 1; j < n; j++) { const f = M[j][i] / M[i][i]; for (let l = i; l <= n; l++) M[j][l] -= f * M[i][l]; } } const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x; };
    const ols = (X, t) => { const p = X[0].length + 1, A = Array.from({ length: p }, () => Array(p).fill(0)), b = Array(p).fill(0); X.forEach((r, i) => { const v = [1, ...r]; for (let a = 0; a < p; a++) { b[a] += v[a] * t[i]; for (let d = 0; d < p; d++) A[a][d] += v[a] * v[d]; } }); for (let a = 0; a < p; a++) A[a][a] += 1e-6; return solve(A, b); };
    const build = () => {
      const k = v$('k'), sd = v$('sd'), leak = q(ctl, '#leak').checked; y = Z.map((z, t) => 100 + 0.4 * t + 10 * Math.sin((2 * Math.PI * t) / 7) + sd * z);
      rows = []; for (let t = k; t < N - 1; t++) { const f = []; for (let j = 1; j <= k; j++) f.push(y[t - j]); if (leak) f.push((y[t - 1] + y[t] + y[t + 1]) / 3); rows.push({ t, f, target: y[t] }); }
      const split = Math.floor(rows.length * 0.7), tr = rows.slice(0, split), te = rows.slice(split);
      const w = ols(tr.map((r) => r.f), tr.map((r) => r.target)), pr = (r) => w[0] + r.f.reduce((s, v, i) => s + v * w[i + 1], 0);
      rows.forEach((r) => (r.pred = pr(r)));
      rows.split = split; rows.maeModel = mean(te.map((r) => Math.abs(r.target - r.pred))); rows.maeNaive = mean(te.map((r) => Math.abs(r.target - y[r.t - 1])));
      pos = 0;
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h, k = v$('k'), leak = q(ctl, '#leak').checked, r = Math.min(Math.floor(pos), rows.length - 1); g.clearRect(0, 0, W, H);
      const [y0, y1] = range([y]), f = frame(g, c, { l: 40, t: 8, w: W - 52, h: H * 0.38 }, 0, N - 1, y0, y1, '');
      ln(g, y, f.X, f.Y, c.border, 1, 0, N - 1);
      const split = rows.split, trainEnd = rows[split - 1].t;
      g.fillStyle = c.blue; g.globalAlpha = 0.07; g.fillRect(f.X(0), f.Y(y1), f.X(trainEnd + 0.5) - f.X(0), f.Y(y0) - f.Y(y1)); g.globalAlpha = 1;
      g.fillStyle = c.red; g.globalAlpha = 0.07; g.fillRect(f.X(trainEnd + 0.5), f.Y(y1), f.X(N - 1) - f.X(trainEnd + 0.5), f.Y(y0) - f.Y(y1)); g.globalAlpha = 1;
      tx(g, 'train rows', f.X(1), 22, c.blue); tx(g, 'test rows (the future)', f.X(trainEnd + 1), 22, c.red);
      for (let i = 0; i < N; i++) dot(g, f.X(i), f.Y(y[i]), 2, c.muted);
      const cur = rows[r]; for (let j = 1; j <= k; j++) dot(g, f.X(cur.t - j), f.Y(y[cur.t - j]), 4.5, c.blue); dot(g, f.X(cur.t), f.Y(y[cur.t]), 5, c.orange);
      if (leak) { dot(g, f.X(cur.t + 1), f.Y(y[cur.t + 1]), 4.5, c.red); tx(g, 'leak: uses tomorrow', f.X(cur.t + 1) + 6, f.Y(y[cur.t + 1]) - 6, c.red); }
      // table of up to 7 latest rows
      const tl = 40, tt = H * 0.38 + 40, rh = 20, show = Math.min(7, r + 1), cols = k + (leak ? 1 : 0) + 2, cw = Math.min(88, (W - tl - 12) / cols);
      const heads = []; for (let j = k; j >= 1; j--) heads.push('y(t−' + j + ')'); if (leak) heads.push('centred avg*'); heads.push('target y(t)', 'model');
      heads.forEach((h, i) => tx(g, h, tl + i * cw + cw - 6, tt, c.muted, '10px sans-serif', 'right'));
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(tl, tt + 5); g.lineTo(tl + heads.length * cw, tt + 5); g.stroke();
      for (let i = 0; i < show; i++) {
        const rr = rows[r - show + 1 + i], yy = tt + 20 + i * rh, isTest = r - show + 1 + i >= split, last = i === show - 1;
        g.fillStyle = last ? c.orange : isTest ? c.red : c.blue; g.globalAlpha = last ? 0.15 : 0.06; g.fillRect(tl, yy - 14, heads.length * cw, rh); g.globalAlpha = 1;
        const vals = []; for (let j = k; j >= 1; j--) vals.push(y[rr.t - j]); if (leak) vals.push(rr.f[k]); vals.push(rr.target, rr.pred);
        vals.forEach((v, j) => tx(g, v.toFixed(1), tl + j * cw + cw - 6, yy, j === vals.length - 2 ? c.fg : c.muted, '11px monospace', 'right'));
      }
      tx(g, 'each row is one window position: the k previous values are the inputs, the next value is the label', tl, H - 8, c.muted, '10px sans-serif');
      q(ctl, '#out').textContent = `window at t = ${cur.t}: inputs y(t−${k}) … y(t−1), label y(t)\nrows ${r + 1} of ${rows.length}; first ${split} train, last ${rows.length - split} test\ntest MAE: model ${rows.maeModel.toFixed(2)}   naive "same as yesterday" ${rows.maeNaive.toFixed(2)}`;
      q(ctl, '#cap').textContent = leak ? 'The centred average includes the value being predicted (and tomorrow). The test error collapses, which looks wonderful and is worthless: that feature cannot exist at forecast time.' : 'Every feature in a row comes strictly from the past of its label. The split is by time: train rows first, test rows after.';
      ['k', 'sd', 'sp'].forEach((id) => (q(ctl, '#' + id + 'v').textContent = q(ctl, '#' + id).value));
    };
    ['k', 'sd', 'leak'].forEach((id) => (q(ctl, '#' + id).oninput = () => { build(); draw(); }));
    q(ctl, '#sp').oninput = draw;
    bindPlay(ctl, st, () => { pos = Math.min(rows.length - 1, Math.floor(pos) + 1); draw(); }, () => { pos = 0; st.playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); });
    o.onresize = draw;
    AI.loop(root, (dt) => { if (!st.playing || pos >= rows.length - 1) return; pos = Math.min(rows.length - 1, pos + dt * v$('sp')); draw(); }, { fps: 30 });
    build(); draw();
  };
})();
