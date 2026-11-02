/* Robotics · Module 3 (state estimation & mapping) labs. All worlds are synthetic and every simulation is illustrative.
 * bayes-corridor    <div data-lab="bayes-corridor"></div>      histogram Bayes filter on a ring corridor
 * kalman-track      <div data-lab="kalman-track"></div>       1-D Kalman filter following a moving robot
 * ekf-range        <div data-lab="ekf-range"></div>          a belief cloud pushed through range and bearing
 * particle-corridor <div data-lab="particle-corridor"></div>  predict, weight and resample a particle cloud
 * occupancy-grid    <div data-lab="occupancy-grid"></div>     log-odds mapping with simulated beams
 * pose-graph        <div data-lab="pose-graph"></div>         dead reckoning drift, loop closure and relaxation
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const TAU = Math.PI * 2;
  const wrap = (v, n) => ((v % n) + n) % n;
  const rrect = (g, x, y, w, h, r) => { g.beginPath(); if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h); };

  /* speed slider: returns a getter for the current multiplier (0.25x to 2x) */
  const speedRow = (ctl) => {
    const l = document.createElement('label');
    l.innerHTML = '<span>Speed <span class="v"></span></span><input type="range" min="0.25" max="2" step="0.05" value="1">';
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = (+inp.value).toFixed(2) + '×');
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  /* generic slider with a live value label; cb runs on every change; returns a getter */
  const slider = (ctl, label, min, max, step, value, unit, cb) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const dec = String(step).includes('.') ? String(step).split('.')[1].length : 0;
    const upd = () => (v.textContent = (+inp.value).toFixed(dec) + (unit ? ' ' + unit : ''));
    inp.oninput = () => { upd(); if (cb) cb(+inp.value); };
    upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  const buttons = (ctl, labels) => {
    const d = document.createElement('div'); d.className = 'row';
    d.innerHTML = labels.map((t) => `<button class="btn small">${t}</button>`).join('');
    ctl.appendChild(d);
    return [...d.querySelectorAll('button')];
  };
  const readoutEl = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; };

  /* ---------- 1 · histogram Bayes filter on a corridor ---------- */
  const WORLD = ['D', 'W', 'W', 'D', 'W', 'W', 'W', 'D', 'W', 'W'];
  AI.labs['bayes-corridor'] = function (root) {
    const N = WORLD.length;
    const { view, ctl } = AI.labShell(root, 'Where am I? A belief over ten cells',
      'illustrative ring corridor: D = door, W = wall; the robot never sees its own cell number',
      '<canvas style="height:280px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const accS = slider(ctl, 'Sensor accuracy', 0.55, 0.99, 0.01, 0.8, '', null);
    const relS = slider(ctl, 'Move reliability (+1 cell)', 0.5, 1, 0.05, 0.8, '', null);
    const btns = buttons(ctl, ['▶ Auto', 'Sense', 'Move', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const START = 'Press Sense or Move, or Auto. At the start every cell is equally likely: the belief is uniform.';
    let rnd, truth, bel, shown, auto, acc, nextSense, dirty, note;
    const norm = (a) => { const s = a.reduce((p, q) => p + q, 0); return a.map((v) => v / s); };
    const reset = () => {
      rnd = AI.rng(5); truth = 1; bel = Array(N).fill(1 / N); shown = bel.slice();
      auto = false; acc = 0; nextSense = true; dirty = true; note = START;
    };
    const sync = () => (btns[0].textContent = auto ? '❚❚ Pause' : '▶ Auto');
    const sense = () => {
      const a = accS(), real = WORLD[truth];
      const z = rnd() < a ? real : (real === 'D' ? 'W' : 'D');
      bel = norm(bel.map((p, i) => p * (WORLD[i] === z ? a : 1 - a)));
      note = `Sense: the sensor reports "${z === 'D' ? 'door' : 'wall'}". Cells that match get multiplied by ${a.toFixed(2)}, the others by ${(1 - a).toFixed(2)}, then every value is divided by the total so the belief sums to 1.`;
      dirty = true;
    };
    const move = () => {
      const q = relS(), st = (1 - q) / 2, nb = Array(N).fill(0);
      bel.forEach((p, i) => { nb[(i + 1) % N] += q * p; nb[i] += st * p; nb[(i + 2) % N] += st * p; });
      const u = rnd();
      truth = (truth + (u < q ? 1 : u < q + st ? 0 : 2)) % N;
      bel = nb;
      note = `Move: the command was "+1 cell". The wheels did +1 with probability ${q.toFixed(2)}, 0 with ${st.toFixed(2)} and +2 with ${st.toFixed(2)}. The belief spreads out: moving makes the robot less sure where it is.`;
      dirty = true;
    };
    reset();
    btns[0].onclick = () => { auto = !auto; acc = 0; sync(); };
    btns[1].onclick = () => { auto = false; sync(); sense(); };
    btns[2].onclick = () => { auto = false; sync(); move(); };
    btns[3].onclick = () => { auto = false; sync(); reset(); sync(); };
    c.onresize = () => { dirty = true; };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const L = 40, R = 12, T = 18, B = 58, bw = (W - L - R) / N, y0 = H - B, ph = H - T - B;
      g.font = `11px ${MONO}`; g.lineWidth = 1;
      for (let v = 0; v <= 1.0001; v += 0.25) {
        const y = y0 - v * ph;
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(L, Math.round(y) + 0.5); g.lineTo(W - R, Math.round(y) + 0.5); g.stroke();
        g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(v.toFixed(2), L - 6, y + 4);
      }
      g.textAlign = 'center';
      for (let i = 0; i < N; i++) {
        const x = L + i * bw, bx = x + bw * 0.15, bwid = bw * 0.7, hgt = ph * shown[i];
        const door = WORLD[i] === 'D';
        g.fillStyle = C.blue; g.fillRect(bx, y0 - hgt, bwid, hgt);
        g.fillStyle = door ? C.orange : C.muted;
        g.fillText(door ? 'door' : 'wall', x + bw / 2, y0 + 16);
        g.fillStyle = C.fg;
        g.fillText(String(i), x + bw / 2, y0 + 30);
        if (shown[i] > 0.005) g.fillText(shown[i].toFixed(2), x + bw / 2, y0 - hgt - 5);
      }
      const rx = L + (truth + 0.5) * bw;
      g.fillStyle = C.green; g.beginPath(); g.arc(rx, y0 + 46, 5, 0, TAU); g.fill();
      g.fillStyle = C.green; g.textAlign = 'left'; g.fillText('true cell (simulator only)', L, y0 + 50);
    };
    AI.loop(root, (dt) => {
      if (auto) {
        acc += dt * speed();
        if (acc >= 2.4) { acc = 0; if (nextSense) sense(); else move(); nextSense = !nextSense; }
      }
      const k = 1 - Math.exp(-dt * 6);
      for (let i = 0; i < N; i++) {
        const nv = shown[i] + (bel[i] - shown[i]) * k;
        if (Math.abs(nv - shown[i]) > 1e-5) dirty = true;
        shown[i] = nv;
      }
      if (!dirty) return;
      dirty = false;
      draw();
      cap.textContent = note;
      let best = 0; bel.forEach((p, i) => { if (p > bel[best]) best = i; });
      const H2 = -bel.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
      out.textContent = `most likely cell: ${best} (belief ${bel[best].toFixed(3)}) · entropy H = ${H2.toFixed(2)} bits (max ${Math.log2(N).toFixed(2)} bits for ten equal cells) · true cell ${truth} is hidden from the robot`;
    }, { fps: 30 });
  };

  /* ---------- 2 · one-dimensional Kalman filter ---------- */
  AI.labs['kalman-track'] = function (root) {
    const T = 40;
    const { view, ctl } = AI.labShell(root, 'Kalman filter: a noisy reading and a model fused',
      'illustrative: the robot truly moves +1 m per second; each second it gets one noisy position reading',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let rnd, run, playing, dirty, acc;
    const btns = buttons(ctl, ['❚❚ Pause', 'Step ▸', 'Restart']);
    const sync = () => (btns[0].textContent = playing ? '❚❚ Pause' : '▶ Play');
    const restart = () => {
      rnd = AI.rng(21);
      run = { truth: [0], z: [null], mu: [0], P: [4], K: [null] };
      playing = true; dirty = true; acc = 0;
      if (btns) sync();
    };
    const sRs = slider(ctl, 'Measurement noise σ_R', 0.2, 3, 0.1, 1.0, 'm', () => restart());
    const sQs = slider(ctl, 'Process noise σ_Q', 0, 1.5, 0.05, 0.3, 'm', () => restart());
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const step = () => {
      const k = run.truth.length, sR = sRs(), sQ = sQs();
      const x = run.truth[k - 1] + 1 + sQ * AI.randn(rnd);
      const z = x + sR * AI.randn(rnd);
      let mu = run.mu[k - 1] + 1, P = run.P[k - 1] + sQ * sQ;      /* predict */
      const K = P / (P + sR * sR);                                  /* gain */
      mu = mu + K * (z - mu); P = (1 - K) * P;                      /* update */
      run.truth.push(x); run.z.push(z); run.mu.push(mu); run.P.push(P); run.K.push(K);
      dirty = true;
    };
    restart();
    const tick = AI.ticker();
    btns[0].onclick = () => { if (run.truth.length > T) restart(); playing = !playing; sync(); };
    btns[1].onclick = () => { playing = false; sync(); if (run.truth.length <= T) step(); };
    btns[2].onclick = () => restart();
    c.onresize = () => { dirty = true; };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const v = AI.view(c, 0, T, -2, 46, { l: 46, r: 14, t: 14, b: 34 });
      AI.axes(g, v);
      const n = run.truth.length;
      g.beginPath();
      for (let k = 0; k < n; k++) g.lineTo(v.X(k), v.Y(run.mu[k] + 2 * Math.sqrt(run.P[k])));
      for (let k = n - 1; k >= 0; k--) g.lineTo(v.X(k), v.Y(run.mu[k] - 2 * Math.sqrt(run.P[k])));
      g.closePath(); g.globalAlpha = 0.18; g.fillStyle = C.blue; g.fill(); g.globalAlpha = 1;
      g.fillStyle = C.muted;
      for (let k = 1; k < n; k++) { g.beginPath(); g.arc(v.X(k), v.Y(run.z[k]), 3, 0, TAU); g.fill(); }
      const line = (arr, col, w, dash) => {
        g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
        for (let k = 0; k < n; k++) (k ? g.lineTo : g.moveTo).call(g, v.X(k), v.Y(arr[k]));
        g.stroke(); g.setLineDash([]);
      };
      line(run.truth, C.green, 1.5, [5, 4]);
      line(run.mu, C.blue, 2.5);
      g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'center';
      g.fillText('time k (s, one step per second)', (v.pad.l + W - v.pad.r) / 2, H - 6);
      g.save(); g.translate(12, (v.pad.t + H - v.pad.b) / 2); g.rotate(-Math.PI / 2); g.fillText('position (m)', 0, 0); g.restore();
    };
    AI.loop(root, (dt) => {
      if (playing && run.truth.length <= T) {
        const m = tick(dt, 0.8 * speed());
        for (let i = 0; i < m && run.truth.length <= T; i++) step();
        if (run.truth.length > T) { playing = false; sync(); }
      }
      if (!dirty) return;
      dirty = false;
      draw();
      const k = run.truth.length - 1, sR = sRs(), sQ = sQs();
      if (k === 0) cap.textContent = 'Start: the estimate is 0 m with uncertainty σ = 2 m (P = 4 m²). No reading has arrived yet.';
      else cap.textContent = `Predict: the model says "+1 m", and the uncertainty grows by σ_Q² = ${(sQ * sQ).toFixed(3)} m². Update: the gain K = P / (P + R) = ${run.K[k].toFixed(3)} (R = σ_R² = ${(sR * sR).toFixed(2)} m²), so the estimate moves ${(run.K[k] * 100).toFixed(0)}% of the way from the prediction to the reading.`;
      if (k >= T) cap.textContent += ' The run is finished: change a slider or press Restart.';
      out.textContent = k === 0 ? 'k = 0' : `k = ${k} s · reading z = ${run.z[k].toFixed(2)} m · estimate μ = ${run.mu[k].toFixed(2)} m · σ = √P = ${Math.sqrt(run.P[k]).toFixed(2)} m · truth ${run.truth[k].toFixed(2)} m (shown dashed)`;
    }, { fps: 30 });
  };

  /* ---------- 3 · nonlinear measurement: EKF linearisation versus the true cloud ---------- */
  AI.labs['ekf-range'] = function (root) {
    const LM = [9, 7], MU = [3, 4], NS = 500;
    const { view, ctl } = AI.labShell(root, 'Range and bearing: the EKF straightens a curve',
      'illustrative 2-D world: a Gaussian belief over the robot position is pushed through h(p) = (range, bearing) to a landmark',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const sigS = slider(ctl, 'Belief spread σ (each axis)', 0.2, 2.5, 0.1, 1.0, 'm', () => { dirty = true; });
    const btns = buttons(ctl, ['Replay reveal']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const rr = AI.rng(4), Z = [];
    for (let i = 0; i < NS; i++) Z.push([AI.randn(rr), AI.randn(rr)]);
    let shownN = 0, dirty = true;
    const hOf = (p) => { const dx = LM[0] - p[0], dy = LM[1] - p[1]; return [Math.hypot(dx, dy), Math.atan2(dy, dx)]; };
    const jac = (p) => { const dx = LM[0] - p[0], dy = LM[1] - p[1], r = Math.hypot(dx, dy), r2 = r * r; return [[-dx / r, -dy / r], [dy / r2, -dx / r2]]; };
    const ellipse = (cen, Cm, k) => {
      const a = Cm[0][0], b = Cm[0][1], d = Cm[1][1];
      const tr = (a + d) / 2, disc = Math.sqrt(((a - d) / 2) ** 2 + b * b);
      const l1 = tr + disc, l2 = Math.max(1e-12, tr - disc), th = 0.5 * Math.atan2(2 * b, a - d);
      const pts = [];
      for (let i = 0; i <= 72; i++) {
        const t = (i / 72) * TAU, x = k * Math.sqrt(l1) * Math.cos(t), y = k * Math.sqrt(l2) * Math.sin(t);
        pts.push([cen[0] + x * Math.cos(th) - y * Math.sin(th), cen[1] + x * Math.sin(th) + y * Math.cos(th)]);
      }
      return pts;
    };
    btns[0].onclick = () => { shownN = 0; dirty = true; };
    c.onresize = () => { dirty = true; };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const sig = sigS(), n = Math.floor(shownN), half = W / 2;
      const pts = Z.map(([a, b]) => [MU[0] + sig * a, MU[1] + sig * b]);
      /* left panel: the world */
      const vl = AI.view({ w: half - 6, h: c.h }, 0, 12, 0, 10, { l: 34, r: 10, t: 14, b: 28 });
      AI.axes(g, vl);
      g.fillStyle = C.blue; g.globalAlpha = 0.45;
      for (let i = 0; i < n; i++) { g.beginPath(); g.arc(vl.X(pts[i][0]), vl.Y(pts[i][1]), 2, 0, TAU); g.fill(); }
      g.globalAlpha = 1;
      g.fillStyle = C.orange; g.beginPath(); g.arc(vl.X(LM[0]), vl.Y(LM[1]), 6, 0, TAU); g.fill();
      g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'left'; g.fillText('landmark', vl.X(LM[0]) + 9, vl.Y(LM[1]) + 4);
      g.strokeStyle = C.fg; g.lineWidth = 1.5; const cx = vl.X(MU[0]), cy = vl.Y(MU[1]);
      g.beginPath(); g.moveTo(cx - 6, cy); g.lineTo(cx + 6, cy); g.moveTo(cx, cy - 6); g.lineTo(cx, cy + 6); g.stroke();
      g.fillText('belief mean μ', cx + 8, cy - 8);
      g.fillText('position (m)', vl.pad.l + 4, vl.pad.t + 12);
      /* right panel: measurement space */
      const hs = pts.slice(0, Math.max(n, 1)).map(hOf);
      const all = pts.map(hOf);
      let r0 = Infinity, r1 = -Infinity, b0 = Infinity, b1 = -Infinity;
      all.forEach(([r, b]) => { r0 = Math.min(r0, r); r1 = Math.max(r1, r); b0 = Math.min(b0, b); b1 = Math.max(b1, b); });
      const hm = hOf(MU), padR = 0.08 * (r1 - r0 + 0.2), padB = 0.08 * (b1 - b0 + 0.02);
      const vr = AI.view({ w: W - half - 6, h: c.h }, r0 - padR, r1 + padR, b0 - padB, b1 + padB, { l: 44, r: 12, t: 14, b: 28 });
      g.save(); g.translate(half + 6, 0);
      AI.axes(g, vr);
      g.fillStyle = C.green; g.globalAlpha = 0.45;
      for (let i = 0; i < n; i++) { g.beginPath(); g.arc(vr.X(all[i][0]), vr.Y(all[i][1]), 2, 0, TAU); g.fill(); }
      g.globalAlpha = 1;
      const J = jac(MU), Cm = [[0, 0], [0, 0]];
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) Cm[i][j] = sig * sig * (J[i][0] * J[j][0] + J[i][1] * J[j][1]);
      const el = ellipse(hm, Cm, 2);
      g.strokeStyle = C.yellow; g.lineWidth = 2; g.setLineDash([6, 4]); g.globalAlpha = n ? Math.min(1, shownN / NS) : 0;
      g.beginPath(); el.forEach(([r, b], i) => (i ? g.lineTo : g.moveTo).call(g, vr.X(r), vr.Y(b))); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
      g.fillStyle = C.fg; g.beginPath(); g.arc(vr.X(hm[0]), vr.Y(hm[1]), 4, 0, TAU); g.fill();
      if (n) {
        const mc = hs.reduce((s, q) => [s[0] + q[0] / hs.length, s[1] + q[1] / hs.length], [0, 0]);
        g.fillStyle = C.pink; g.beginPath(); g.arc(vr.X(mc[0]), vr.Y(mc[1]), 5, 0, TAU); g.fill();
        g.fillText('mean of cloud', vr.X(mc[0]) + 8, vr.Y(mc[1]) + 14);
      }
      g.fillText('h(μ): EKF centre', vr.X(hm[0]) + 8, vr.Y(hm[1]) - 6);
      g.fillText('range (m)', vr.pad.l + vr.pw / 2 - 20, vr.pad.t + vr.ph + 26);
      g.fillText('bearing (rad)', 4, vr.pad.t + 12);
      g.restore();
      /* readout */
      const sdR = Math.sqrt(Cm[0][0]), mcSd = n > 1 ? Math.sqrt(hs.reduce((s, q) => s + (q[0] - hs.reduce((a, t) => a + t[0], 0) / n) ** 2, 0) / n) : 0;
      return { hm, J, Cm, sdR, mcSd, n, hs, sig };
    };
    AI.loop(root, (dt) => {
      if (shownN < NS) { shownN = Math.min(NS, shownN + dt * 60 * speed()); dirty = true; }
      if (!dirty) return;
      dirty = false;
      const s = draw();
      const mcMean = s.n ? s.hs.reduce((a, q) => a + q[0], 0) / s.n : NaN;
      cap.textContent = 'Left: possible robot positions drawn from a Gaussian belief. Right: each of the same positions pushed through h(p) = (range, bearing), with the landmark at (9, 7) m. The dashed ellipse is the EKF picture: a straight-line stand-in for h at the mean. Where the right-hand cloud curves or is lopsided, the ellipse is wrong.';
      out.textContent = `h(μ) = (range ${s.hm[0].toFixed(3)} m, bearing ${s.hm[1].toFixed(3)} rad) · Jacobian row for range = (${s.J[0][0].toFixed(3)}, ${s.J[0][1].toFixed(3)}) · EKF sd of range = ${s.sdR.toFixed(3)} m · sample sd of range = ${s.mcSd.toFixed(3)} m (${s.n} of ${NS} samples) · sample mean of range minus h(μ) = ${s.n ? (mcMean - s.hm[0]).toFixed(3) : '–'} m`;
    }, { fps: 30 });
  };

  /* ---------- 4 · particle filter on the ring corridor ---------- */
  AI.labs['particle-corridor'] = function (root) {
    const PH = 1.8; /* seconds per phase */
    const { view, ctl } = AI.labShell(root, 'Particle filter: a crowd of guesses on the corridor',
      'illustrative ring of ten cells (the corridor wraps round); each dot is one particle, each particle is one hypothesis',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const nS = slider(ctl, 'Particles N', 10, 400, 10, 60, '', () => reset());
    const accS = slider(ctl, 'Sensor accuracy', 0.55, 0.99, 0.01, 0.8, '', null);
    const btns = buttons(ctl, ['▶ Play', 'Next phase', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    let rnd, P, truth, truthS, phase, clock, playing, dirty, note, reading, neff, est;
    const motion = () => { const u = rnd(); return (u < 0.8 ? 1 : u < 0.9 ? 0 : 2) + 0.05 * AI.randn(rnd); };
    const sync = () => (btns[0].textContent = playing ? '❚❚ Pause' : '▶ Play');
    const cellOf = (x) => Math.floor(wrap(x, 10));
    const reset = () => {
      rnd = AI.rng(9);
      const n = nS();
      P = Array.from({ length: n }, () => ({ x: rnd() * 10, s: 0, w: 1 / n }));
      truth = 6.5; truthS = 0; phase = 2; clock = 0; playing = false; reading = null; neff = n; est = null;
      note = 'Press Play. The cloud starts spread over the whole corridor: no idea where the robot is.';
      dirty = true; sync();
    };
    /* run the action of a phase when it starts */
    const doPhase = (p) => {
      if (p === 0) {
        truthS = motion(); truth = wrap(truth + truthS, 10);
        P.forEach((q) => { q.s = motion(); q.x = wrap(q.x + q.s, 10); });
        note = 'Predict: the wheels moved the robot. Every particle gets its own random motion sample (+1 cell 80% of the time, 0 cells 10%, +2 cells 10%). The cloud spreads out.';
      } else if (p === 1) {
        const a = accS(), real = ['D', 'W', 'W', 'D', 'W', 'W', 'W', 'D', 'W', 'W'][cellOf(truth)];
        reading = rnd() < a ? real : (real === 'D' ? 'W' : 'D');
        const W10 = ['D', 'W', 'W', 'D', 'W', 'W', 'W', 'D', 'W', 'W'];
        let sum = 0;
        P.forEach((q) => { q.w = W10[cellOf(q.x)] === reading ? a : 1 - a; sum += q.w; });
        P.forEach((q) => (q.w /= sum));
        neff = 1 / P.reduce((s, q) => s + q.w * q.w, 0);
        note = `Sense: the reading is "${reading === 'D' ? 'door' : 'wall'}". A particle standing on a matching cell gets weight ${a.toFixed(2)}, otherwise ${(1 - a).toFixed(2)}; the weights are then normalised to sum to 1.`;
      } else {
        const n = P.length, cum = []; let acc = 0;
        P.forEach((q) => { acc += q.w; cum.push(acc); });
        cum[n - 1] = 1;
        const u0 = rnd() / n, nx = []; let j = 0;
        for (let i = 0; i < n; i++) {
          const u = (u0 + i) / n;
          while (j < n - 1 && u > cum[j]) j++;
          nx.push(P[j].x);
        }
        P.forEach((q, i) => { q.x = nx[i]; q.s = 0; q.w = 1 / n; });
        const counts = Array(10).fill(0);
        P.forEach((q) => counts[cellOf(q.x)]++);
        est = counts.indexOf(Math.max(...counts));
        note = `Resample: draw ${n} new particles in proportion to their weights (systematic resampling: one random offset, then evenly spaced pointers). Light particles disappear and heavy ones are copied. The estimate is the cell with the most particles: ${est}.`;
      }
      clock = 0; dirty = true;
    };
    reset();
    btns[0].onclick = () => { playing = !playing; sync(); };
    btns[1].onclick = () => { playing = false; sync(); phase = (phase + 1) % 3; doPhase(phase); };
    btns[2].onclick = () => reset();
    c.onresize = () => { dirty = true; };
    const draw = (k) => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const X0 = 26, CW = (W - 2 * X0) / 10, bandY = H * 0.36, bandH = 26;
      const xOf = (x) => X0 + x * CW;
      const W10 = ['D', 'W', 'W', 'D', 'W', 'W', 'W', 'D', 'W', 'W'];
      /* corridor cells */
      for (let i = 0; i < 10; i++) {
        g.fillStyle = W10[i] === 'D' ? AI.mix(C.bg2, C.orange, 0.5) : C.bg2;
        g.fillRect(X0 + i * CW + 1, bandY, CW - 2, bandH);
        g.fillStyle = W10[i] === 'D' ? C.orange : C.muted; g.font = `11px ${MONO}`; g.textAlign = 'center';
        g.fillText(W10[i] === 'D' ? 'door' : 'wall', X0 + (i + 0.5) * CW, bandY + bandH + 14);
        g.fillStyle = C.muted; g.fillText(String(i), X0 + (i + 0.5) * CW, bandY + bandH + 27);
      }
      /* particles, drawn at their display positions */
      const disp = (q) => wrap(q.x - q.s * (1 - k), 10);
      g.fillStyle = C.blue;
      P.forEach((q, i) => {
        const lane = (i * 7) % 5;
        g.globalAlpha = 0.75; g.beginPath(); g.arc(xOf(disp(q)), bandY - 8 - lane * 5, 3, 0, TAU); g.fill();
      });
      g.globalAlpha = 1;
      /* histogram: particle counts, or weight totals while sensing */
      const bins = Array(10).fill(0);
      P.forEach((q) => { bins[cellOf(disp(q))] += phase === 1 && k >= 1 ? q.w : 1 / P.length; });
      const mx = Math.max(...bins, 1e-9), hb = H * 0.9 - (bandY + bandH + 36), hy = bandY + bandH + 40;
      g.fillStyle = C.blue; g.globalAlpha = 0.35;
      for (let i = 0; i < 10; i++) g.fillRect(X0 + i * CW + CW * 0.2, hy + hb * (1 - bins[i] / mx), CW * 0.6, hb * (bins[i] / mx));
      g.globalAlpha = 1;
      /* true robot */
      const tx = xOf(wrap(truth - truthS * (1 - k), 10));
      g.fillStyle = C.red; g.beginPath(); g.arc(tx, bandY + bandH / 2, 6, 0, TAU); g.fill();
      g.fillStyle = C.red; g.textAlign = 'center'; g.font = `11px ${MONO}`;
      g.fillText('true robot', tx, bandY + bandH / 2 - 11);
      /* estimate */
      if (est !== null && phase === 2) {
        g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([4, 3]);
        const ex = X0 + (est + 0.5) * CW; g.beginPath(); g.moveTo(ex, bandY - 60); g.lineTo(ex, hy + hb); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.green; g.fillText('estimate', ex, bandY - 64);
      }
    };
    AI.loop(root, (dt) => {
      if (playing) {
        clock += dt * speed();
        if (clock >= PH) { phase = (phase + 1) % 3; doPhase(phase); }
        dirty = true;
      }
      if (!dirty) return;
      dirty = false;
      const k = phase === 0 ? AI.smooth(Math.min(1, clock / PH)) : 1;
      draw(k);
      cap.textContent = note;
      const N = P.length;
      const counts = Array(10).fill(0); P.forEach((q) => counts[cellOf(q.x)]++);
      const mode = counts.indexOf(Math.max(...counts));
      out.textContent = `particles N = ${N} · effective sample size N_eff = ${neff.toFixed(1)} (equal weights would give N) · most particles in cell ${mode} · true cell ${cellOf(truth)} (simulator only)`;
    }, { fps: 30 });
  };

  /* ---------- 5 · occupancy grid with simulated beams ---------- */
  AI.labs['occupancy-grid'] = function (root) {
    const WW = 24, HH = 16, NB = 48, MAXR = 18, RX = 4.5, RY = 8.5;
    const wall = new Uint8Array(WW * HH);
    const setW = (x, y) => { if (x >= 0 && y >= 0 && x < WW && y < HH) wall[y * WW + x] = 1; };
    for (let x = 0; x < WW; x++) { setW(x, 0); setW(x, HH - 1); }
    for (let y = 0; y < HH; y++) { setW(0, y); setW(WW - 1, y); }
    for (let y = 1; y < 12; y++) if (y < 6 || y > 7) setW(8, y);
    for (let x = 12; x < 23; x++) if (x !== 16) setW(x, 10);
    for (let x = 14; x <= 15; x++) for (let y = 3; y <= 4; y++) setW(x, y);
    setW(19, 4); setW(20, 4); setW(20, 5);
    const { view, ctl } = AI.labShell(root, 'Occupancy grid: a robot maps a room with beams',
      'illustrative: a 2-D lidar scans from one fixed pose; each cell stores log-odds of being occupied',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const pS = slider(ctl, 'Sensor accuracy p', 0.55, 0.95, 0.01, 0.7, '', () => {});
    const btns = buttons(ctl, ['▶ Scan', 'Clear map']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    let rnd, L, beam, scanning, acc, dirty, drawnBeams, note;
    const reset = () => {
      rnd = AI.rng(12); L = new Float32Array(WW * HH); beam = 0; scanning = false; acc = 0; dirty = true;
      drawnBeams = []; note = 'Press Scan. Each beam walks from the robot outwards; cells before the return are evidence for "free", the return cell is evidence for "occupied".';
    };
    const sync = () => (btns[0].textContent = scanning ? '❚❚ Pause' : '▶ Scan');
    reset();
    btns[0].onclick = () => { scanning = !scanning; if (beam >= NB) { beam = 0; drawnBeams = []; } sync(); };
    btns[1].onclick = () => { reset(); sync(); };
    c.onresize = () => { dirty = true; };
    const castBeam = (ang) => {
      const dx = Math.cos(ang), dy = Math.sin(ang), cells = [];
      let hit = null, d = 0;
      for (d = 0.05; d <= MAXR; d += 0.05) {
        const x = RX + dx * d, y = RY + dy * d, cx = Math.floor(x), cy = Math.floor(y);
        if (cx < 0 || cy < 0 || cx >= WW || cy >= HH) break;
        const id = cy * WW + cx;
        if (cells[cells.length - 1] !== id) cells.push(id);
        if (wall[id]) { hit = id; break; }
      }
      return { cells, hit, end: Math.min(d, MAXR) };
    };
    const doBeam = () => {
      const p = pS(), lO = Math.log(p / (1 - p)), lF = -lO;
      const ang = (beam / NB) * TAU + 0.5 / NB * TAU;
      const b = castBeam(ang);
      let reported = null;
      const visited = b.hit !== null ? b.cells.filter((id) => id !== b.hit) : b.cells;
      visited.forEach((id) => (L[id] = Math.max(-8, Math.min(8, L[id] + lF))));
      if (b.hit !== null) {
        if (rnd() < p) reported = b.hit;
        else {
          const hx = b.hit % WW, hy = Math.floor(b.hit / WW);
          const nb = [[1, 0], [-1, 0], [0, 1], [0, -1]][Math.floor(rnd() * 4)];
          const nx = Math.min(WW - 1, Math.max(0, hx + nb[0])), ny = Math.min(HH - 1, Math.max(0, hy + nb[1]));
          reported = ny * WW + nx;
        }
        L[reported] = Math.max(-8, Math.min(8, L[reported] + lO));
      }
      drawnBeams.push({ ang, end: b.end, hit: b.hit !== null });
      beam++;
      note = `Beam ${beam} of ${NB}: ${visited.length} cell(s) updated as free (log-odds ${lF.toFixed(3)} each). ` +
        (b.hit !== null ? `Return reported in cell (${reported % WW}, ${Math.floor(reported / WW)}) and updated as occupied (+${lO.toFixed(3)}). ` : 'No return within range: every cell along the beam is free. ') +
        (p < 0.9 ? 'Some returns land on a neighbouring cell, which is why the map gets blurry near walls.' : '');
      if (beam >= NB) { scanning = false; sync(); note += ' Scan complete. Press Scan again to add more evidence.'; }
      dirty = true;
    };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const s = Math.min((W - 10) / WW, (H - 10) / HH), ox = (W - s * WW) / 2, oy = (H - s * HH) / 2;
      for (let y = 0; y < HH; y++) for (let x = 0; x < WW; x++) {
        const pr = 1 / (1 + Math.exp(-L[y * WW + x]));
        g.fillStyle = AI.mix(C.bg, C.orange, pr);
        g.fillRect(ox + x * s, oy + y * s, s + 0.5, s + 0.5);
      }
      g.strokeStyle = C.border; g.lineWidth = 0.5;
      for (let x = 0; x <= WW; x++) { g.beginPath(); g.moveTo(ox + x * s, oy); g.lineTo(ox + x * s, oy + HH * s); g.stroke(); }
      for (let y = 0; y <= HH; y++) { g.beginPath(); g.moveTo(ox, oy + y * s); g.lineTo(ox + WW * s, oy + y * s); g.stroke(); }
      const rx = ox + RX * s, ry = oy + RY * s;
      g.strokeStyle = C.yellow; g.lineWidth = 1; g.globalAlpha = 0.35;
      drawnBeams.forEach((bm) => { g.beginPath(); g.moveTo(rx, ry); g.lineTo(rx + Math.cos(bm.ang) * bm.end * s, ry + Math.sin(bm.ang) * bm.end * s); g.stroke(); });
      g.globalAlpha = 1;
      g.fillStyle = C.green; g.beginPath(); g.arc(rx, ry, 6, 0, TAU); g.fill();
    };
    AI.loop(root, (dt) => {
      if (scanning) { acc += dt * speed() * 3; while (acc >= 1 && scanning) { acc -= 1; doBeam(); } }
      if (!dirty) return;
      dirty = false;
      draw();
      cap.textContent = note;
      let tw = 0, twFound = 0, tf = 0, tfFree = 0, occ = 0;
      for (let y = 0; y < HH; y++) for (let x = 0; x < WW; x++) {
        const id = y * WW + x, pr = 1 / (1 + Math.exp(-L[id]));
        if (wall[id]) { tw++; if (pr > 0.5) twFound++; }
        else if (x > 0 && y > 0 && x < WW - 1 && y < HH - 1) { tf++; if (pr < 0.5) tfFree++; }
        if (pr > 0.5) occ++;
      }
      out.textContent = `beams cast ${beam} of ${NB} in this scan · cells with P(occupied) > 0.5: ${occ} · true wall cells marked occupied: ${twFound} of ${tw} · free interior cells marked free: ${tfFree} of ${tf} · sensor p = ${pS().toFixed(2)}, log-odds per hit ±${Math.log(pS() / (1 - pS())).toFixed(3)}`;
    }, { fps: 30 });
  };

  /* ---------- 6 · pose graph: drift, loop closure and relaxation ---------- */
  AI.labs['pose-graph'] = function (root) {
    const DIRS = [[1, 0], [0, 1], [-1, 0], [0, -1]], MAXIT = 400;
    const TRUE = [[0, 0]];
    DIRS.forEach(([dx, dy]) => { for (let s = 0; s < 4; s++) { const l = TRUE[TRUE.length - 1]; TRUE.push([l[0] + dx, l[1] + dy]); } });
    const { view, ctl } = AI.labShell(root, 'Drift and loop closure: relaxing a pose graph',
      'illustrative 2-D graph: a square of 16 one-metre steps; poses are points, not frames',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let z, D, Pp, disp, iter, playing, dirty, loopOn = true;
    const biasS = slider(ctl, 'Gyro bias β (extra heading error per step)', 0, 5, 0.5, 2, '°', () => rebuild());
    const btns = buttons(ctl, ['Loop closure: on', '▶ Relax', 'Skip to end', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const tick = AI.ticker();
    const rebuild = () => {
      const b = biasS() * Math.PI / 180;
      z = [];
      for (let k = 0; k < 16; k++) {
        const th = b * (k + 1), cb = Math.cos(th), sb = Math.sin(th);   /* the heading error accumulates */
        const sx = TRUE[k + 1][0] - TRUE[k][0], sy = TRUE[k + 1][1] - TRUE[k][1];
        z.push([cb * sx - sb * sy, sb * sx + cb * sy]);
      }
      D = [[0, 0]];
      for (let k = 0; k < 16; k++) D.push([D[k][0] + z[k][0], D[k][1] + z[k][1]]);
      Pp = D.map((p) => p.slice()); disp = D.map((p) => p.slice());
      iter = 0; playing = false; dirty = true; btns[1].textContent = '▶ Relax'; btns[0].textContent = loopOn ? 'Loop closure: on' : 'Loop closure: off';
      note = 'Dead reckoning: each step is turned by β more than the one before (a gyro bias that accumulates). The steps add up to a path that drifts and no longer closes.';
    };
    let note = '';
    rebuild();
    btns[0].onclick = () => { loopOn = !loopOn; rebuild(); };
    btns[1].onclick = () => { playing = !playing; btns[1].textContent = playing ? '❚❚ Pause' : '▶ Relax'; };
    btns[2].onclick = () => { let moved = 1; for (let i = 0; i < 4000 && moved > 1e-9; i++) { moved = sweep(); iter++; } playing = false; btns[1].textContent = '▶ Relax'; disp = Pp.map((p) => p.slice()); dirty = true; };
    btns[3].onclick = () => { rebuild(); };
    c.onresize = () => { dirty = true; };
    /* one sweep over all edges: each edge pulls its two poses towards the measured step; node 0 is anchored */
    function sweep() {
      const edges = [];
      for (let k = 0; k < 16; k++) edges.push([k, k + 1, z[k][0], z[k][1]]);
      if (loopOn) edges.push([16, 0, 0, 0]);
      let moved = 0;
      edges.forEach(([i, j, zx, zy]) => {
        const ex = (Pp[j][0] - Pp[i][0]) - zx, ey = (Pp[j][1] - Pp[i][1]) - zy;
        const fi = i === 0 ? 0 : j === 0 ? 1 : 0.5, fj = j === 0 ? 0 : i === 0 ? 1 : 0.5;
        if (i !== 0) { Pp[i][0] += fi * ex; Pp[i][1] += fi * ey; moved += Math.abs(ex) + Math.abs(ey); }
        if (j !== 0) { Pp[j][0] -= fj * ex; Pp[j][1] -= fj * ey; moved += Math.abs(ex) + Math.abs(ey); }
      });
      return moved;
    }
    /* total squared edge error: the quantity relaxation minimises */
    const edgeErr = (P) => {
      let e2 = 0;
      for (let k = 0; k < 16; k++) e2 += ((P[k + 1][0] - P[k][0]) - z[k][0]) ** 2 + ((P[k + 1][1] - P[k][1]) - z[k][1]) ** 2;
      if (loopOn) e2 += (P[16][0] - P[0][0]) ** 2 + (P[16][1] - P[0][1]) ** 2;
      return e2;
    };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const sc = Math.min((W - 60) / 6, (H - 60) / 6);
      const X = (x) => W / 2 + (x - 2) * sc, Y = (y) => H / 2 - (y - 2) * sc;
      const path = (P, col, w, dash) => {
        g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
        P.forEach((p, k) => (k ? g.lineTo : g.moveTo).call(g, X(p[0]), Y(p[1])));
        g.stroke(); g.setLineDash([]);
      };
      path(TRUE, C.muted, 1, [2, 3]);
      path(D, C.orange, 1.5, [6, 4]);
      if (loopOn) { g.strokeStyle = C.green; g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(disp[16][0]), Y(disp[16][1])); g.lineTo(X(disp[0][0]), Y(disp[0][1])); g.stroke(); g.setLineDash([]); }
      path(disp, C.blue, 2.5);
      g.fillStyle = C.blue; disp.forEach((p) => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 3, 0, TAU); g.fill(); });
      g.fillStyle = C.orange; D.forEach((p, k) => { if (k % 4 === 0) { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 3, 0, TAU); g.fill(); } });
      g.fillStyle = C.fg; g.font = `11px ${MONO}`; g.textAlign = 'left';
      g.fillText('start (anchored)', X(0) + 8, Y(0) + 16);
      g.fillText('pose 16', X(disp[16][0]) + 8, Y(disp[16][1]) - 8);
      g.fillStyle = C.muted; g.fillText('dotted grey: true square', 12, H - 12);
      g.fillStyle = C.orange; g.fillText('dashed orange: dead reckoning', 12, H - 28);
      g.fillStyle = C.blue; g.fillText('solid blue: current estimate', 12, H - 44);
    };
    AI.loop(root, (dt) => {
      if (playing && iter < MAXIT) {
        const m = tick(dt, 12 * speed());
        for (let i = 0; i < m && iter < MAXIT; i++) { sweep(); iter++; }
        if (iter >= MAXIT) { playing = false; btns[1].textContent = '▶ Relax'; }
        dirty = true;
      }
      let moved = 0;
      for (let k = 0; k < Pp.length; k++) {
        const nx = disp[k][0] + (Pp[k][0] - disp[k][0]) * (1 - Math.exp(-dt * 5));
        const ny = disp[k][1] + (Pp[k][1] - disp[k][1]) * (1 - Math.exp(-dt * 5));
        moved = Math.max(moved, Math.abs(nx - disp[k][0]) + Math.abs(ny - disp[k][1]));
        disp[k][0] = nx; disp[k][1] = ny;
      }
      if (moved > 1e-5) dirty = true;
      if (!dirty) return;
      dirty = false;
      draw();
      const gapD = Math.hypot(D[16][0] - D[0][0], D[16][1] - D[0][1]), gapP = Math.hypot(Pp[16][0] - Pp[0][0], Pp[16][1] - Pp[0][1]);
      if (!loopOn) cap.textContent = 'Loop closure is off: no edge says that pose 16 is pose 0, so relaxing cannot pull the drift back. The odometry edges are already satisfied by the dead-reckoned poses, so nothing moves.';
      else if (iter === 0) cap.textContent = 'Relax: each edge says "pose j minus pose i should equal the measured step". The loop-closure edge says pose 16 and pose 0 are the same place. Press Relax: each sweep moves the poses a little to reduce the total squared error.';
      else cap.textContent = `Iteration ${iter}: the error of the loop closure is spread back along the path, leg by leg. The optimised path is a compromise that satisfies all edges as well as possible.`;
      out.textContent = `iteration ${iter} · start-to-end gap: dead reckoning ${gapD.toFixed(3)} m, now ${gapP.toFixed(3)} m · sum of squared edge errors: ${edgeErr(D).toFixed(3)} m² before, ${edgeErr(Pp).toFixed(3)} m² now · gyro bias β = ${biasS().toFixed(1)}° per step`;
    }, { fps: 30 });
  };
})();
