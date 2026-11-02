/* Robotics · Module 1 (sensors & actuators) labs. Illustrative simulations: no real hardware is read.
 * encoder-disc   <div data-lab="encoder-disc"></div>   quadrature, absolute code, gear ratio
 * imu-drift      <div data-lab="imu-drift"></div>      gyro bias drift, accelerometer tilt, complementary filter
 * stereo-depth   <div data-lab="stereo-depth"></div>   disparity, depth Z = fB/d and its error band
 * lidar-scan     <div data-lab="lidar-scan"></div>     beam sweep, ray casting, time of flight
 * noise-fusion   <div data-lab="noise-fusion"></div>   two noisy sensors, bias, inverse-variance fusion
 * force-contact  <div data-lab="force-contact"></div>  spring contact, threshold, grip friction
 */
(function () {
  const AI = window.AI;
  if (!AI) return;
  const TAU = 2 * Math.PI, DEG = 180 / Math.PI, C_LIGHT = 299792458;
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const fx = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');

  /* ---------- shared controls ---------- */
  function slider(ctl, label, min, max, step, val, unit, dec) {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = (+inp.value).toFixed(dec) + (unit ? ' ' + unit : ''));
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return () => +inp.value;
  }
  /* expression input: returns a getter for the compiled function f(scope) */
  function expr(ctl, label, src, vars) {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span> <input type="text" value="${esc(src)}" spellcheck="false" style="font-family:${MONO};min-width:220px"> <span class="err" style="color:var(--red);font-size:12px"></span>`;
    ctl.appendChild(l);
    const inp = l.querySelector('input'), err = l.querySelector('.err');
    let fn = AI.compile(AI.parse(src), vars);
    AI.exprInput(inp, err, { allowed: vars, onChange: (f) => (fn = f) });
    return () => fn;
  }
  /* transport row: play/pause, one step, reset. Returns a state object. */
  function transport(ctl, onReset) {
    const d = document.createElement('div'); d.className = 'row';
    d.innerHTML = '<button class="btn small">⏸ Pause</button><button class="btn small">Step ▸</button><button class="btn small">Reset</button>';
    ctl.appendChild(d);
    const [pp, st, rs] = d.querySelectorAll('button');
    const s = { playing: true, stepReq: 0 };
    pp.onclick = () => { s.playing = !s.playing; pp.textContent = s.playing ? '⏸ Pause' : '▶ Play'; };
    st.onclick = () => { s.playing = false; pp.textContent = '▶ Play'; s.stepReq++; };
    rs.onclick = () => { onReset(); s.stepReq = 0; s.playing = true; pp.textContent = '⏸ Pause'; };
    return s;
  }
  /* standard lab frame: header, canvas and caption. Returns pieces and a redraw flag. */
  function frame(root, title, sub, canvasStyle) {
    const { view, ctl } = AI.labShell(root, title, sub,
      `<canvas style="${canvasStyle}"></canvas><div class="caption"></div>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const cap = AI.$('.caption', view);
    const st = { dirty: true };
    cv.onresize = () => (st.dirty = true);
    ctl.addEventListener('input', () => (st.dirty = true));
    ctl.addEventListener('click', () => (st.dirty = true));
    const readout = document.createElement('div'); readout.className = 'readout'; view.appendChild(readout);
    return { view, ctl, cv, cap, st, readout };
  }
  function clearCanvas(c) {
    const C = AI.col(), g = c.ctx;
    g.fillStyle = C.bg2; g.fillRect(0, 0, c.w, c.h);
    return C;
  }
  /* seeded Gaussian generator */
  const gauss = (rng) => AI.randn(rng);

  /* ================= 1 · encoder disc ================= */
  AI.labs['encoder-disc'] = function (root) {
    const F = frame(root, 'Encoder disc and quadrature',
      'A disc turns; two channels read its slits. Illustrative: ideal sensor, no friction.', 'height:320px');
    const { ctl, cv, cap, st, readout } = F;
    const getL = slider(ctl, 'Slits on the disc L', 4, 500, 1, 500, '', 0);
    const getN = slider(ctl, 'Gear ratio N (motor turns per joint turn)', 1, 100, 1, 1, '', 0);
    const ef = expr(ctl, 'Motor speed ω(t) in rev/s, t in s', '0.4', ['t']);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    const modeRow = document.createElement('div'); modeRow.className = 'row';
    modeRow.innerHTML = '<button class="btn small">Reading: incremental</button>';
    ctl.appendChild(modeRow);
    const modeBtn = modeRow.querySelector('button');
    let absMode = false;
    modeBtn.onclick = () => { absMode = !absMode; modeBtn.textContent = 'Reading: ' + (absMode ? 'absolute (12-bit)' : 'incremental'); };
    let T = 0, theta = 0;
    const tr = transport(ctl, () => { T = 0; theta = 0; });
    const tick = AI.ticker();

    AI.loop(root, (dt) => {
      let steps = 0;
      if (tr.playing) steps = tick(dt, 100 * getSpeed());
      else if (tr.stepReq > 0) { tr.stepReq--; steps = 10; }
      const fn = ef();
      for (let i = 0; i < steps; i++) { T += 0.01; const w = fn({ t: T }); theta += (Number.isFinite(w) ? w : 0) * TAU * 0.01; }
      if (!(steps > 0 || st.dirty)) return;
      st.dirty = false;
      draw(fn);
    });

    function draw(fn) {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H = cv.h;
      const L = Math.round(getL()), N = getN(), CPR = 4 * L;
      const w = fn({ t: T });
      const dir = w > 0 ? 'forward' : w < 0 ? 'reverse' : 'stopped';
      // disc on the left
      const R = Math.min(H * 0.38, W * 0.17), cx = R + 18, cy = H * 0.5;
      g.beginPath(); g.arc(cx, cy, R, 0, TAU); g.fillStyle = C.blue; g.globalAlpha = 0.35; g.fill(); g.globalAlpha = 1;
      g.fillStyle = C.bg2;
      for (let k = 0; k < L; k++) {
        const a0 = theta + (k / L) * TAU, a1 = a0 + (0.5 / L) * TAU;
        g.beginPath(); g.arc(cx, cy, R, a0, a1); g.arc(cx, cy, R * 0.55, a1, a0, true); g.closePath(); g.fill();
      }
      g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + R * Math.cos(theta), cy + R * Math.sin(theta)); g.stroke();
      g.fillStyle = C.yellow; g.beginPath(); g.arc(cx, cy - R - 8, 4, 0, TAU); g.fill();
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.textAlign = 'center';
      g.fillText('light sensor', cx, cy - R - 14);
      // waveform strip: the window shows six slits, the right edge is the current angle
      const x0 = W * 0.42, x1 = W - 12, span = (6 / L) * TAU, ya = H * 0.3, yb = H * 0.7, amp = H * 0.12;
      g.textAlign = 'left'; g.fillStyle = C.muted;
      g.fillText('A', x0 - 2, ya - amp - 4); g.fillText('B', x0 - 2, yb - amp - 4);
      const state = (ang) => { const s = Math.floor((ang / TAU) * 4 * L); return ((s % 4) + 4) % 4; };
      const pathFor = (chan, base) => {
        g.beginPath();
        const n = Math.max(2, Math.round(x1 - x0));
        for (let i = 0; i <= n; i++) {
          const ang = theta - ((n - i) / n) * span;
          const s = state(ang);
          const v = chan === 'A' ? (s === 1 || s === 2 ? 1 : 0) : (s === 2 || s === 3 ? 1 : 0);
          const x = x0 + (i / n) * (x1 - x0), y = base - v * amp;
          if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
        }
        g.strokeStyle = chan === 'A' ? C.green : C.purple; g.lineWidth = 2; g.stroke();
      };
      pathFor('A', ya); pathFor('B', yb);
      g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(x1, 8); g.lineTo(x1, H - 8); g.stroke();
      g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText('now', x1 - 4, H - 10);
      // state table
      const s = state(theta);
      const names = ['00', '10', '11', '01'];
      g.textAlign = 'left'; g.font = '12px ' + MONO;
      g.fillStyle = C.fg; g.fillText('quadrature state:  ' + names.map((n, i) => (i === s ? '[' + n + ']' : ' ' + n + ' ')).join(' '), x0, H * 0.06);

      const motorDeg = (((theta * DEG) % 360) + 360) % 360;
      const counts = Math.floor((theta / TAU) * CPR);
      const frac = ((theta / TAU) % 1 + 1) % 1;
      const code = Math.floor(frac * 4096);
      const bin = code.toString(2).padStart(12, '0');
      const jointDeg = ((((theta / N) * DEG) % 360) + 360) % 360;
      cap.textContent = absMode
        ? 'Absolute reading: the 12-bit code is one word per angle, so the position is known at power-up. Motor angle ' + fx(motorDeg, 1) + '°, joint ' + fx(jointDeg, 2) + '°.'
        : 'Incremental reading: channel A leads B when turning forward. Direction now: ' + dir + '. Counts since power-up: ' + counts + ' (zero is arbitrary).';
      readout.innerHTML =
        'CPR = 4 × ' + L + ' = <b>' + CPR + '</b> counts/rev · motor resolution <b>' + fx(360 / CPR, 4) + '°</b> per count<br>' +
        'joint resolution (÷ N = ' + N + ') <b>' + fx(360 / (CPR * N), 5) + '°</b> per count · joint angle <b>' + fx(jointDeg, 2) + '°</b><br>' +
        (absMode ? '12-bit absolute code <b>' + bin + '</b> (' + code + ' of 4096)' : 'incremental count <b>' + counts + '</b>') +
        ' · speed ω(t) = ' + fx(w, 3) + ' rev/s';
    }
  };

  /* ================= 2 · IMU drift and complementary filter ================= */
  AI.labs['imu-drift'] = function (root) {
    const F = frame(root, 'IMU: gyro drift and the complementary filter',
      'Illustrative: a simulated tilt, a biased gyro and a noisy accelerometer. Not real sensor data.', 'height:300px');
    const { ctl, cv, cap, st, readout } = F;
    const ef = expr(ctl, 'True tilt θ(t) in rad, t in s', '0.4*sin(0.25*t)', ['t']);
    const getBias = slider(ctl, 'Gyro bias b', 0, 2, 0.05, 0.5, '°/s', 2);
    const getNoise = slider(ctl, 'Gyro noise', 0, 0.5, 0.01, 0.1, '°/s', 2);
    const getAlpha = slider(ctl, 'Filter weight α', 0.8, 0.999, 0.001, 0.98, '', 3);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 0.5, '×', 2);
    let T = 0, thTrue = 0, gyroEst = 0, fused = 0, rng = AI.rng(11), hist = [], k = 0;
    const init = () => {
      T = 0; rng = AI.rng(11); hist = []; k = 0;
      const th0 = ef()({ t: 0 }); thTrue = th0; gyroEst = th0; fused = th0;
    };
    init();
    const tr = transport(ctl, init);
    const tick = AI.ticker();
    const SIM_DT = 0.01, H = 0.001;

    function simStep() {
      const fn = ef();
      const th = fn({ t: T });
      const dth = (fn({ t: T + H }) - fn({ t: T - H })) / (2 * H);
      const bias = getBias() / DEG, sn = getNoise() / DEG;
      const wm = dth + bias + sn * gauss(rng);
      const ax = 9.81 * Math.sin(th) + 0.05 * gauss(rng);
      const az = 9.81 * Math.cos(th) + 0.05 * gauss(rng);
      const tacc = Math.atan2(ax, az);
      const a = getAlpha();
      gyroEst += wm * SIM_DT;
      fused = a * (fused + wm * SIM_DT) + (1 - a) * tacc;
      T += SIM_DT; k++;
      thTrue = fn({ t: T });
      if (k % 5 === 0) { hist.push({ t: T, tr: thTrue, g: gyroEst, f: fused }); if (hist.length > 1200) hist.shift(); }
    }

    AI.loop(root, (dt) => {
      let steps = 0;
      if (tr.playing) steps = tick(dt, 100 * getSpeed());
      else if (tr.stepReq > 0) { tr.stepReq--; steps = 100; }
      for (let i = 0; i < steps; i++) simStep();
      if (!(steps > 0 || st.dirty)) return;
      st.dirty = false;
      draw();
    });

    function draw() {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H2 = cv.h;
      const sx = W * 0.22, sy = H2 * 0.52, Lb = Math.min(W * 0.16, H2 * 0.35);
      // body drawings
      const bodyLine = (ang, col, wdt) => {
        g.strokeStyle = col; g.lineWidth = wdt; g.beginPath();
        g.moveTo(sx, sy); g.lineTo(sx + Lb * Math.sin(ang), sy - Lb * Math.cos(ang)); g.stroke();
      };
      g.beginPath(); g.arc(sx, sy, 3, 0, TAU); g.fillStyle = C.muted; g.fill();
      bodyLine(thTrue, C.muted, 8);
      bodyLine(gyroEst, C.orange, 3);
      bodyLine(fused, C.green, 3);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'center';
      g.fillText('body (top view of tilt)', sx, H2 - 8);
      // time plot
      const px0 = W * 0.44, px1 = W - 14, py0 = 14, py1 = H2 - 26, YR = 45;
      const tMax = Math.max(60, T), tMin = tMax - 60;
      const X = (t) => px0 + ((t - tMin) / (tMax - tMin)) * (px1 - px0);
      const Y = (d) => py0 + (1 - (AI.clamp(d, -YR, YR) + YR) / (2 * YR)) * (py1 - py0);
      g.strokeStyle = C.border; g.lineWidth = 1;
      for (let d = -40; d <= 40; d += 20) { g.beginPath(); g.moveTo(px0, Y(d)); g.lineTo(px1, Y(d)); g.stroke(); g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(d + '°', px0 - 6, Y(d) + 4); }
      g.textAlign = 'center'; g.fillText('time (s), last 60 s', (px0 + px1) / 2, H2 - 8);
      const trace = (key, col, wdt) => {
        g.strokeStyle = col; g.lineWidth = wdt; g.beginPath();
        hist.forEach((p, i) => { const x = X(p.t), y = Y(p[key] * DEG); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
        g.stroke();
      };
      trace('tr', C.muted, 3); trace('g', C.orange, 2); trace('f', C.green, 2);
      const eg = (gyroEst - thTrue) * DEG, ef2 = (fused - thTrue) * DEG;
      const biasDeg = getBias() * T;
      cap.textContent = 'Grey: truth. Orange: gyro only, which drifts by bias × time = ' + fx(biasDeg, 1) + '° so far. Green: filtered, which the accelerometer keeps pulling back.';
      readout.innerHTML =
        'true tilt <b>' + fx(thTrue * DEG, 2) + '°</b> · gyro-only error <b>' + fx(eg, 2) + '°</b> · fused error <b>' + fx(ef2, 2) + '°</b><br>' +
        'time <b>' + fx(T, 1) + ' s</b> · filter weight α = <b>' + getAlpha().toFixed(3) + '</b> (time constant ≈ ' + fx(SIM_DT / (1 - getAlpha()), 2) + ' s)';
    }
  };

  /* ================= 3 · stereo disparity and depth ================= */
  AI.labs['stereo-depth'] = function (root) {
    const F = frame(root, 'Stereo depth from disparity',
      'Top view (cameras at the bottom) and the two image strips. Illustrative geometry with exact pinhole formulas.', 'height:360px');
    const { ctl, cv, cap, st, readout } = F;
    const ef = expr(ctl, 'Depth Z(t) in m, t in s', '5.5 + 4.5*sin(0.15*t)', ['t']);
    const getX = slider(ctl, 'Lateral position X of the object', -0.5, 0.5, 0.01, 0.25, 'm', 2);
    const getB = slider(ctl, 'Baseline B', 4, 40, 0.5, 12, 'cm', 1);
    const getF = slider(ctl, 'Focal length f', 200, 1000, 10, 600, 'px', 0);
    const getErr = slider(ctl, 'Disparity error ±', 0, 2, 0.05, 0.5, 'px', 2);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    let T = 0;
    const tr = transport(ctl, () => { T = 0; });
    const IMG_W = 640, CX = 320;

    AI.loop(root, (dt) => {
      let moved = false;
      if (tr.playing) { T += dt * getSpeed() * 0.5; moved = true; }
      else if (tr.stepReq > 0) { tr.stepReq--; T += 0.5; moved = true; }
      if (!(moved || st.dirty)) return;
      st.dirty = false;
      draw();
    });

    function draw() {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H = cv.h;
      const Z = AI.clamp(ef()({ t: T }), 0.5, 10);
      const X = getX(), B = getB() / 100, f = getF();
      const dTrue = (f * B) / Z;
      const uL = CX + (f * X) / Z, uR = CX + (f * (X - B)) / Z;
      const dMeas = dTrue + getErr() * (2 * AI.rng(Math.floor(T * 5) + 1)() - 1);
      const Zest = (f * B) / Math.max(dMeas, 1e-6);
      const band = (Z * Z * getErr()) / (f * B);
      // top view
      const tx0 = 30, tx1 = W - 20, ty0 = 12, ty1 = H * 0.58;
      const PX = (xm) => tx0 + ((xm + 0.6) / 1.5) * (tx1 - tx0);
      const PY = (zm) => ty1 - (zm / 10) * (ty1 - ty0);
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '11px ' + MONO; g.textAlign = 'right';
      for (let zz = 2; zz <= 10; zz += 2) { g.beginPath(); g.moveTo(tx0, PY(zz)); g.lineTo(tx1, PY(zz)); g.stroke(); g.fillText(zz + ' m', tx0 - 4, PY(zz) + 4); }
      // rays from both cameras to the object
      g.setLineDash([4, 4]); g.strokeStyle = C.muted;
      [0, B].forEach((cxm) => { g.beginPath(); g.moveTo(PX(cxm), PY(0)); g.lineTo(PX(X), PY(Z)); g.stroke(); });
      g.setLineDash([]);
      [0, B].forEach((cxm, i) => {
        g.fillStyle = i === 0 ? C.blue : C.orange;
        g.fillRect(PX(cxm) - 8, PY(0) - 6, 16, 10);
      });
      g.fillStyle = C.fg; g.textAlign = 'center'; g.font = '11px ' + MONO;
      g.fillText('left cam', PX(0), PY(0) + 18); g.fillText('right cam', PX(B), PY(0) + 18);
      g.fillStyle = C.green; g.beginPath(); g.arc(PX(X), PY(Z), 6, 0, TAU); g.fill();
      g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText('object Z = ' + fx(Z, 2) + ' m', PX(X) + 10, PY(Z) + 4);
      // image strips: a 640 px wide sensor, the centre column at CX
      const stripX0 = 30, stripW = W - 60;
      const sy = [H * 0.66, H * 0.84], sh = H * 0.1;
      const ux = (u) => stripX0 + (u / IMG_W) * stripW;
      ['left image', 'right image'].forEach((name, i) => {
        g.fillStyle = C.bg; g.fillRect(stripX0, sy[i], stripW, sh);
        g.strokeStyle = C.border; g.strokeRect(stripX0, sy[i], stripW, sh);
        g.fillStyle = C.muted; g.textAlign = 'left'; g.font = '11px ' + MONO; g.fillText(name, stripX0 + 4, sy[i] - 4);
      });
      const uu = [uL, uR];
      uu.forEach((u, i) => {
        const inside = u >= 0 && u <= IMG_W;
        const xx = ux(AI.clamp(u, 0, IMG_W));
        g.fillStyle = i === 0 ? C.blue : C.orange;
        g.beginPath(); g.arc(xx, sy[i] + sh / 2, 6, 0, TAU); g.fill();
        if (!inside) { g.fillStyle = C.red; g.fillText('out of view', xx + 8, sy[i] + sh / 2 + 4); }
      });
      g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(ux(CX), sy[0]); g.lineTo(ux(CX), sy[1] + sh); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.textAlign = 'left';
      g.fillText('disparity d = u_L − u_R = ' + fx(dTrue, 2) + ' px', ux(0), sy[1] + sh + 14);
      cap.textContent = 'Depth from disparity: Z = fB/d. A far object has a small disparity, and half a pixel of disparity error then moves the depth by a wide band.';
      readout.innerHTML =
        'true Z = <b>' + fx(Z, 2) + ' m</b> · true d = fB/Z = <b>' + fx(dTrue, 2) + ' px</b> · measured d = <b>' + fx(dMeas, 2) + ' px</b><br>' +
        'estimated Z = fB/d = <b>' + fx(Zest, 2) + ' m</b> · depth error band ±<b>' + fx(band, 3) + ' m</b> (= Z²Δd/(fB)) · B = ' + fx(B * 100, 1) + ' cm';
    }
  };

  /* ================= 4 · LiDAR sweep and time of flight ================= */
  AI.labs['lidar-scan'] = function (root) {
    const F = frame(root, 'LiDAR sweep and time of flight',
      'Illustrative 2-D slice of a 3-D scanner: walls and two boxes, a beam sweeping slowly around the robot.', 'height:340px');
    const { ctl, cv, cap, st, readout } = F;
    const getN = slider(ctl, 'Beams per turn', 12, 360, 1, 72, '', 0);
    const getRate = slider(ctl, 'Sweep rate', 0.02, 0.5, 0.01, 0.1, 'rev/s', 2);
    const getSig = slider(ctl, 'Range noise σ', 0, 0.2, 0.005, 0.03, 'm', 3);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    // world: robot at the origin; walls and two boxes, all in metres
    const segs = [
      [-8, -6, 8, -6], [8, -6, 8, 6], [8, 6, -8, 6], [-8, 6, -8, -6],
      [2, 1, 3, 1], [3, 1, 3, 2.5], [3, 2.5, 2, 2.5], [2, 2.5, 2, 1],
      [-4, -3, -3, -3], [-3, -3, -3, -1.5], [-3, -1.5, -4, -1.5], [-4, -1.5, -4, -3],
    ];
    const RMAX = 30;
    const cast = (a) => {
      const dx = Math.cos(a), dy = Math.sin(a);
      let best = Infinity;
      for (const [x1, y1, x2, y2] of segs) {
        const ex = x2 - x1, ey = y2 - y1;
        const cde = dx * ey - dy * ex;
        if (Math.abs(cde) < 1e-12) continue;
        const t = (x1 * ey - y1 * ex) / cde;
        const s = (x1 * dy - y1 * dx) / cde;
        if (t > 0 && s >= 0 && s <= 1 && t < best) best = t;
      }
      return best <= RMAX ? best : null;
    };
    let T = 0, shots = 0, lastN = -1, ptR = [], lastShot = null, rng = AI.rng(3);
    const reset = () => { T = 0; shots = 0; ptR = []; lastShot = null; rng = AI.rng(3); };
    const tr = transport(ctl, reset);
    const tick = AI.ticker();

    AI.loop(root, (dt) => {
      const N = Math.round(getN());
      if (N !== lastN) { lastN = N; reset(); st.dirty = true; ptR = new Array(N).fill(undefined); }
      if (tr.playing) T += dt * getSpeed();
      else if (tr.stepReq > 0) { tr.stepReq--; T += 0.5; }
      const target = Math.floor(T * getRate() * N);
      let fired = 0;
      while (shots < target && fired < 400) {
        const k = shots % N, a = (TAU * k) / N;
        const r0 = cast(a);
        const r = r0 === null ? null : Math.max(0, r0 + getSig() * gauss(rng));
        ptR[k] = r;
        lastShot = { k, a, r: r, r0 };
        shots++; fired++;
      }
      if (!(fired > 0 || st.dirty)) return;
      st.dirty = false;
      draw(N);
    });

    function draw(N) {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H = cv.h;
      const S = Math.min(W, H) / 19.5, ox = W / 2, oy = H / 2;
      const P = (x, y) => [ox + x * S, oy - y * S];
      // range rings
      g.strokeStyle = C.border; g.lineWidth = 1; g.font = '11px ' + MONO; g.fillStyle = C.muted;
      [10, 20, 30].forEach((R) => { g.beginPath(); g.arc(ox, oy, R * S, 0, TAU); g.stroke(); g.fillText(R + ' m', ox + R * S + 3, oy - 3); });
      // walls and boxes
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath();
      segs.forEach(([x1, y1, x2, y2]) => { const [a, b] = P(x1, y1), [c, d] = P(x2, y2); g.moveTo(a, b); g.lineTo(c, d); });
      g.stroke();
      // points from the latest sweep
      g.fillStyle = C.teal || C.blue;
      for (let k = 0; k < ptR.length; k++) {
        const r = ptR[k]; if (r === undefined || r === null) continue;
        const a = (TAU * k) / N, [px, py] = P(r * Math.cos(a), r * Math.sin(a));
        g.fillRect(px - 1.5, py - 1.5, 3, 3);
      }
      // current beam
      g.fillStyle = C.green; g.beginPath(); g.arc(ox, oy, 5, 0, TAU); g.fill();
      if (lastShot) {
        const reach = lastShot.r === null ? RMAX : lastShot.r;
        const [ex, ey] = P(reach * Math.cos(lastShot.a), reach * Math.sin(lastShot.a));
        g.strokeStyle = C.yellow; g.lineWidth = 1.5; g.beginPath(); g.moveTo(ox, oy); g.lineTo(ex, ey); g.stroke();
        if (lastShot.r !== null) { g.fillStyle = C.yellow; g.beginPath(); g.arc(ex, ey, 4, 0, TAU); g.fill(); }
      }
      const got = ptR.filter((v) => v !== undefined && v !== null).length;
      const rr = lastShot && lastShot.r !== null ? lastShot.r : null;
      cap.textContent = 'Each beam measures the echo delay Δt and converts it to a range r = cΔt/2, then to a point. The cloud keeps the latest return for every beam, so it fills in as the sweep turns.';
      readout.innerHTML = rr === null
        ? 'current beam: <b>no return</b> within ' + RMAX + ' m · ' + N + ' beams per turn · ' + got + ' points'
        : 'current beam at ' + fx((lastShot.a * 180) / Math.PI, 1) + '° · range <b>' + fx(rr, 3) + ' m</b> · echo delay Δt = 2r/c = <b>' + fx((2 * rr / C_LIGHT) * 1e9, 2) + ' ns</b> · ' + got + ' points';
    }
  };

  /* ================= 5 · two sensors and fusion ================= */
  AI.labs['noise-fusion'] = function (root) {
    const F = frame(root, 'Two noisy sensors and their fusion',
      'Illustrative: a quantity measured with Gaussian noise and a bias. Each dot is one reading.', 'height:320px');
    const { ctl, cv, cap, st, readout } = F;
    const ef = expr(ctl, 'True value x(t), t in s', '1.0', ['t']);
    const getS1 = slider(ctl, 'σ sensor 1 (blue)', 0.05, 1, 0.01, 0.3, 'm', 2);
    const getS2 = slider(ctl, 'σ sensor 2 (orange)', 0.05, 1, 0.01, 0.1, 'm', 2);
    const getB1 = slider(ctl, 'Bias of sensor 1', -0.5, 0.5, 0.01, 0, 'm', 2);
    const getRate = slider(ctl, 'Reading pairs per second', 0.5, 5, 0.1, 2, 'Hz', 1);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    let T = 0, rng = AI.rng(5), S = null, hist = [], wsum = 0, n = 0;
    const reset = () => { T = 0; rng = AI.rng(5); hist = []; wsum = 0; n = 0; };
    const tr = transport(ctl, reset);
    const tick = AI.ticker();

    AI.loop(root, (dt) => {
      let pairs = 0;
      if (tr.playing) pairs = tick(dt, getRate() * getSpeed());
      else if (tr.stepReq > 0) { tr.stepReq--; pairs = 1; }
      for (let i = 0; i < pairs; i++) onePair();
      if (!(pairs > 0 || st.dirty)) return;
      st.dirty = false;
      draw();
    });

    function onePair() {
      const x = ef()({ t: T });
      const s1 = getS1(), s2 = getS2(), b1 = getB1();
      const z1 = x + b1 + s1 * gauss(rng), z2 = x + s2 * gauss(rng);
      const w1 = 1 / (s1 * s1), w2 = 1 / (s2 * s2);
      wsum += w1 * z1 + w2 * z2; n += 1;
      const est = wsum / (n * (w1 + w2));
      const sd = 1 / Math.sqrt(n * (w1 + w2));
      hist.push({ t: T, z1, z2, est, sd, x });
      T += 1 / Math.max(getRate(), 0.1);
      while (hist.length && hist[0].t < T - 40) hist.shift();
      S = { z1, z2, est, sd, x, s1, s2, w1, w2 };
    }

    function draw() {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H = cv.h;
      const p0 = 40, p1 = W - 12, q0 = 12, q1 = H - 24;
      const tMax = Math.max(40, T), tMin = tMax - 40;
      const xc = S ? S.x : ef()({ t: T });
      const R = Math.max(0.4, 3 * Math.max(getS1(), getS2()) + Math.abs(getB1()));
      const X = (t) => p0 + ((t - tMin) / (tMax - tMin)) * (p1 - p0);
      const Y = (v) => q0 + (1 - (AI.clamp(v - xc, -R, R) + R) / (2 * R)) * (q1 - q0);
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '11px ' + MONO; g.textAlign = 'right';
      [-R, 0, R].forEach((d) => { g.beginPath(); g.moveTo(p0, Y(xc + d)); g.lineTo(p1, Y(xc + d)); g.stroke(); g.fillText(fx(xc + d, 2) + ' m', p0 - 4, Y(xc + d) + 4); });
      // band and fused line
      if (hist.length > 1) {
        g.fillStyle = C.green; g.globalAlpha = 0.15; g.beginPath();
        hist.forEach((h, i) => { const y = Y(h.est + 2 * h.sd); if (i === 0) g.moveTo(X(h.t), y); else g.lineTo(X(h.t), y); });
        for (let i = hist.length - 1; i >= 0; i--) { const h = hist[i]; g.lineTo(X(h.t), Y(h.est - 2 * h.sd)); }
        g.closePath(); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath();
        hist.forEach((h, i) => { const x = X(h.t), y = Y(h.est); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
        g.stroke();
      }
      // truth line
      g.strokeStyle = C.muted; g.lineWidth = 1.5; g.setLineDash([6, 4]); g.beginPath();
      for (let i = 0; i <= 80; i++) { const t = tMin + (i / 80) * (tMax - tMin); const x = X(t), y = Y(ef()({ t })); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
      g.stroke(); g.setLineDash([]);
      // dots
      hist.forEach((h) => {
        g.fillStyle = C.blue; g.globalAlpha = 0.7; g.beginPath(); g.arc(X(h.t), Y(h.z1), 2.5, 0, TAU); g.fill();
        g.fillStyle = C.orange; g.beginPath(); g.arc(X(h.t), Y(h.z2), 2.5, 0, TAU); g.fill(); g.globalAlpha = 1;
      });
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'center';
      g.fillText('time (s), last 40 s · dashed grey: true value · green: fused estimate with ±2 std band', (p0 + p1) / 2, H - 6);
      if (!S) { cap.textContent = 'Press play to start the readings.'; readout.textContent = ''; return; }
      const m1 = hist.length ? hist.reduce((a, h) => a + h.z1, 0) / hist.length : NaN;
      const m2 = hist.length ? hist.reduce((a, h) => a + h.z2, 0) / hist.length : NaN;
      cap.textContent = 'Each pair of dots is one reading from each sensor. The green fused estimate weights each by 1/σ², and its band shrinks as the readings accumulate (the band is ±2 × the fused standard error).';
      readout.innerHTML =
        'pairs <b>' + n + '</b> · fused estimate <b>' + fx(S.est, 3) + ' m</b> · fused std <b>' + fx(S.sd, 4) + ' m</b> (≈ 1/√(n(w₁+w₂)))<br>' +
        'sensor 1 mean <b>' + fx(m1, 3) + ' m</b> (bias ' + fx(getB1(), 2) + ' m) · sensor 2 mean <b>' + fx(m2, 3) + ' m</b> · true value <b>' + fx(S.x, 3) + ' m</b>';
    }
  };

  /* ================= 6 · force sensing at contact ================= */
  AI.labs['force-contact'] = function (root) {
    const F = frame(root, 'Force sensing at contact',
      'Illustrative: a finger presses a soft surface. The spring law is editable; noise and grip are simulated.', 'height:340px');
    const { ctl, cv, cap, st, readout } = F;
    const law = expr(ctl, 'Force law F(x) in N, x = penetration in m', '5000*x', ['x']);
    const getNoise = slider(ctl, 'Force noise σ', 0, 2, 0.05, 0.5, 'N', 2);
    const getThr = slider(ctl, 'Contact threshold', 0.5, 5, 0.1, 1.5, 'N', 1);
    const getNg = slider(ctl, 'Grip force per finger N_g', 5, 100, 1, 40, 'N', 0);
    const getMu = slider(ctl, 'Friction coefficient μ', 0.1, 1, 0.01, 0.4, '', 2);
    const getMass = slider(ctl, 'Part mass m', 0.5, 10, 0.1, 3, 'kg', 1);
    const getSpeed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    const barRow = document.createElement('div'); barRow.className = 'bar'; barRow.innerHTML = '<i></i>'; ctl.appendChild(barRow);
    const bar = barRow.querySelector('i');
    const PS = 0.005, DT = 0.02;
    let T = 0, rng = AI.rng(9), hist = [], falseFlags = 0, quiet = 0;
    const reset = () => { T = 0; rng = AI.rng(9); hist = []; falseFlags = 0; quiet = 0; };
    const tr = transport(ctl, reset);
    const tick = AI.ticker();
    let cur = null;
    const finger = (t) => 0.01 * (1 + Math.sin(0.3 * t));  // metres, 0 to 20 mm

    AI.loop(root, (dt) => {
      let steps = 0;
      if (tr.playing) steps = tick(dt, 50 * getSpeed());
      else if (tr.stepReq > 0) { tr.stepReq--; steps = 5; }
      const fn = law();
      for (let i = 0; i < steps; i++) {
        T += DT;
        const p = finger(T), delta = Math.max(0, p - PS);
        const truth = Math.max(0, fn({ x: delta }));
        const reading = truth + getNoise() * gauss(rng);
        const flag = reading > getThr();
        if (delta === 0 && flag) falseFlags++;
        cur = { t: T, p, delta, truth, reading, flag };
        hist.push(cur); if (hist.length > 1500) hist.shift();
      }
      if (!(steps > 0 || st.dirty)) return;
      st.dirty = false;
      draw();
    });

    function draw() {
      const C = clearCanvas(cv), g = cv.ctx, W = cv.w, H = cv.h;
      // scene: finger moves right into a surface, top third of the canvas
      const sceneH = H * 0.34, yb = sceneH - 10;
      const xOf = (p) => W * 0.12 + (p / 0.02) * W * 0.3;
      const surfX = xOf(PS);
      g.fillStyle = C.muted; g.globalAlpha = 0.35; g.fillRect(surfX, 14, W - surfX - 10, yb - 14); g.globalAlpha = 1;
      g.fillStyle = C.bg3 || C.border; g.fillRect(surfX - 2, 14, 2, yb - 14);
      const p = cur ? cur.p : finger(T);
      g.fillStyle = C.blue; g.fillRect(xOf(p) - 16, 20, 16, yb - 20);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('surface', surfX + 6, 30);
      g.fillText('finger', xOf(p) - 56, 30);
      // plot: force against time, bottom two thirds
      const q0 = sceneH + 14, q1 = H - 22, p0 = 46, p1 = W - 12;
      const tMax = Math.max(30, T), tMin = tMax - 30;
      const YMAX = 90;
      const X = (t) => p0 + ((t - tMin) / (tMax - tMin)) * (p1 - p0);
      const Y = (v) => q1 - (AI.clamp(v, 0, YMAX) / YMAX) * (q1 - q0);
      g.strokeStyle = C.border; g.lineWidth = 1;
      for (let v = 0; v <= YMAX; v += 30) { g.beginPath(); g.moveTo(p0, Y(v)); g.lineTo(p1, Y(v)); g.stroke(); g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(v + ' N', p0 - 4, Y(v) + 4); }
      const thr = getThr();
      g.strokeStyle = C.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(p0, Y(thr)); g.lineTo(p1, Y(thr)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath();
      hist.forEach((h, i) => { const x = X(h.t), y = Y(h.truth); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); });
      g.stroke();
      g.fillStyle = C.orange; g.globalAlpha = 0.75;
      hist.forEach((h) => { const x = X(h.t); if (x < p0) return; g.beginPath(); g.arc(x, Y(h.reading), 2, 0, TAU); g.fill(); });
      g.globalAlpha = 1;
      g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('time (s), last 30 s · grey: true force · dots: sensor reading · red: threshold', (p0 + p1) / 2, H - 6);
      // grip
      const Ng = getNg(), mu = getMu(), m = getMass();
      const cap2 = 2 * mu * Ng, load = m * 9.81;
      bar.style.width = (cap2 > 0 ? AI.clamp((load / cap2) * 100, 0, 100) : 100).toFixed(1) + '%';
      bar.style.background = load > cap2 ? C.red : C.green;
      const pen = cur ? cur.delta : 0;
      cap.textContent = 'The finger presses only when it passes the surface (delta > 0). Noise makes readings cross the threshold even with no contact: those are false flags. Grip: the bar is the load mg over the friction limit 2μN_g; the part slips when it is full.';
      readout.innerHTML =
        'penetration δ = <b>' + fx(pen * 1000, 2) + ' mm</b> · true force <b>' + fx(cur ? cur.truth : 0, 2) + ' N</b> · reading <b>' + fx(cur ? cur.reading : 0, 2) + ' N</b> · flag <b>' + (cur && cur.flag ? 'CONTACT' : 'none') + '</b> · false flags <b>' + falseFlags + '</b><br>' +
        'grip capacity 2μN_g = <b>' + fx(cap2, 1) + ' N</b> · load mg = <b>' + fx(load, 1) + ' N</b> · ' + (load > cap2 ? '<span style="color:var(--red)">slips</span>' : 'holds, margin ' + fx(cap2 - load, 1) + ' N');
    }
  };
})();
