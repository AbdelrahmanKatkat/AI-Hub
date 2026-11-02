/* Robotics · Module 0 · Robot foundations — labs.
 * Every simulation is ILLUSTRATIVE: a teaching model in code, not a real robot. Numbers are computed live.
 * sense-think-act   <div data-lab="sense-think-act"></div>    one joint, one loop; the control rate decides stability
 * frame-chain       <div data-lab="frame-chain"></div>        frames attached to a two-link planar arm
 * config-space      <div data-lab="config-space"></div>       workspace collisions mapped to configuration space
 * reach-2link       <div data-lab="reach-2link"></div>        reachable ring and the two elbow solutions
 * redundancy-3link  <div data-lab="redundancy-3link"></div>   three joints, one spare degree of freedom
 * stack-flow        <div data-lab="stack-flow"></div>         publish rates, bus latency and data age
 * stack-timeline    <div data-lab="stack-timeline"></div>     fixed-priority schedule of control tasks
 */
(function () {
  const D = Math.PI / 180, TAU = Math.PI * 2;
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const f = (v, d = 2) => (Math.abs(v) < 5e-10 ? 0 : v).toFixed(d);
  const theme = () => document.documentElement.getAttribute('data-theme') || '';

  /* ---------- small UI helpers ---------- */
  const slider = (ctl, label, min, max, step, val, unit = '', dec = 2) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const show = () => (v.textContent = (+inp.value).toFixed(dec) + unit);
    const o = {
      get: () => +inp.value,
      set: (x) => { inp.value = x; show(); },
      onchange: null,
    };
    inp.oninput = () => { show(); if (o.onchange) o.onchange(); };
    show();
    ctl.appendChild(l);
    return o;
  };
  const buttons = (ctl, labels) => {
    const d = document.createElement('div');
    d.className = 'row';
    d.innerHTML = labels.map((t) => `<button class="btn small">${t}</button>`).join('');
    ctl.appendChild(d);
    return [...d.querySelectorAll('button')];
  };
  const readout = (ctl) => {
    const d = document.createElement('div');
    d.className = 'readout';
    ctl.appendChild(d);
    return d;
  };
  const text = (g, s, x, y, color, align = 'left', size = 11) => {
    g.save(); g.font = size + 'px ' + MONO; g.fillStyle = color; g.textAlign = align; g.fillText(s, x, y); g.restore();
  };
  const seg = (g, x0, y0, x1, y1, color, w = 2, dash = null) => {
    g.save(); g.strokeStyle = color; g.lineWidth = w; if (dash) g.setLineDash(dash);
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.restore();
  };
  const arrow = (g, x0, y0, x1, y1, color, w = 2) => {
    seg(g, x0, y0, x1, y1, color, w);
    const a = Math.atan2(y1 - y0, x1 - x0), hl = 8;
    g.beginPath(); g.moveTo(x1, y1);
    g.lineTo(x1 - hl * Math.cos(a - 0.4), y1 - hl * Math.sin(a - 0.4));
    g.lineTo(x1 - hl * Math.cos(a + 0.4), y1 - hl * Math.sin(a + 0.4));
    g.closePath(); g.fillStyle = color; g.fill();
  };
  const disc = (g, x, y, r, color) => { g.beginPath(); g.arc(x, y, r, 0, TAU); g.fillStyle = color; g.fill(); };
  const gauss = () => { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * Math.random()); };

  /* planar homogeneous transforms (3x3) */
  const Rm = (a) => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
  const Tr = (x, y) => [[1, 0, x], [0, 1, y], [0, 0, 1]];
  const mul = (A, B) => A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
  const app = (A, p) => [0, 1, 2].map((i) => A[i][0] * p[0] + A[i][1] * p[1] + A[i][2] * p[2]);

  /* two-link planar arm, link lengths L1, L2 */
  const pdist = (px, py, ax, ay, bx, by) => {
    const abx = bx - ax, aby = by - ay;
    const t = clamp(((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby), 0, 1);
    return Math.hypot(px - (ax + t * abx), py - (ay + t * aby));
  };
  const arm2 = (q1, q2, L1 = 1, L2 = 1) => {
    const ex = L1 * Math.cos(q1), ey = L1 * Math.sin(q1);
    return { ex, ey, tx: ex + L2 * Math.cos(q1 + q2), ty: ey + L2 * Math.sin(q1 + q2) };
  };
  const hitArm = (q1, q2, cx, cy, r) => {
    const a = arm2(q1, q2);
    return pdist(cx, cy, 0, 0, a.ex, a.ey) < r || pdist(cx, cy, a.ex, a.ey, a.tx, a.ty) < r;
  };

  /* ================= 1 · sense, think, act ================= */
  AI.labs['sense-think-act'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One joint, one loop (illustrative)',
      'sense → think → act, repeated at a fixed control rate · simulated time, not wall-clock',
      '<canvas style="height:250px"></canvas><canvas style="height:170px"></canvas><div class="caption"></div>', '');
    const [arm, strip] = [...view.querySelectorAll('canvas')].map((cv) => AI.canvas(cv));
    const cap = AI.$('.caption', view);
    const rate = slider(ctl, 'Control rate', 1, 200, 1, 10, ' Hz', 0);
    const kp = slider(ctl, 'Gain Kp (speed per radian of error)', 0.5, 8, 0.1, 5, ' 1/s', 1);
    const tgt = slider(ctl, 'Target angle', -90, 90, 1, 30, '°', 0);
    const noise = slider(ctl, 'Sensor noise (standard deviation)', 0, 3, 0.1, 0, '°', 1);
    const speed = slider(ctl, 'Speed (simulated s per real s)', 0.05, 2, 0.05, 0.5, '×', 2);
    const [pp, step, reset] = buttons(ctl, ['Pause', 'Step one tick', 'Reset']);
    const rd = readout(ctl);

    const S = { th: 0, t: 0, acc: 0, hist: [[0, 0]], run: true, div: false, meas: 0, cmd: 0, move: 0 };
    const restart = () => { S.th = 0; S.t = 0; S.acc = 0; S.hist = [[0, 0]]; S.div = false; };
    const tick = () => {
      if (S.div) return;
      const dtc = 1 / rate.get();
      S.meas = S.th + noise.get() * D * gauss();          // 1 sense: the encoder reading, with noise
      const err = tgt.get() * D - S.meas;                  // 2 think: error and command
      S.cmd = kp.get() * err;                              // rad/s
      S.move = S.cmd * dtc;                                // 3 act: the motor runs at that speed for one period
      S.th += S.move;
      S.t += dtc;
      S.hist.push([S.t, S.th]);
      if (Math.abs(S.th) > 1000) S.div = true;
    };
    const advance = (dt) => {
      if (S.run) {
        S.acc += dt * speed.get();
        const dtc = 1 / rate.get();
        let n = 0;
        while (S.acc >= dtc && n < 400) { tick(); S.acc -= dtc; n++; }
        if (S.acc > dtc) S.acc = 0;
      }
      const cut = S.t - 8;
      while (S.hist.length > 2 && S.hist[0][0] < cut) S.hist.shift();
    };

    const draw = () => {
      const C = AI.col(), g = arm.ctx, w = arm.w, h = arm.h;
      g.clearRect(0, 0, w, h);
      const px = w * 0.28, py = h * 0.55, L = Math.min(w, h) * 0.36;
      const at = (a) => [px + L * Math.cos(a), py - L * Math.sin(a)];
      seg(g, px, py, px + L * 1.15 * Math.cos(tgt.get() * D), py - L * 1.15 * Math.sin(tgt.get() * D), C.green, 1.5, [5, 4]);
      const ms = at(S.meas);
      seg(g, px, py, ms[0], ms[1], C.orange, 1.5, [2, 3]);
      const th = S.div ? 0 : S.th;
      const tp = at(th);
      seg(g, px, py, tp[0], tp[1], C.blue, 8);
      disc(g, px, py, 7, C.fg);
      disc(g, tp[0], tp[1], 5, C.green);
      text(g, 'pivot', px, py + 20, C.muted, 'center');
      text(g, '① last measured angle (orange, dotted)', 10, 16, C.orange);
      text(g, '③ joint angle (blue)', 10, 30, C.blue);
      text(g, '— — target direction', 10, 44, C.green);

      const sg = strip.ctx, sw = strip.w, sh = strip.h;
      sg.clearRect(0, 0, sw, sh);
      const t1 = Math.max(S.t, 1e-3), t0 = Math.max(0, t1 - 8);
      const Ymax = Math.max(45, Math.abs(tgt.get()) * 1.4);
      const v = AI.view(strip, t0, t0 + 8 > t1 ? t0 + 8 : t1, -Ymax, Ymax, { l: 46, r: 12, t: 10, b: 24 });
      seg(sg, v.pad.l, v.Y(0), sw - v.pad.r, v.Y(0), C.border, 1);
      seg(sg, v.pad.l, v.Y(tgt.get()), sw - v.pad.r, v.Y(tgt.get()), C.green, 1.5, [6, 4]);
      sg.save(); sg.beginPath(); sg.rect(v.pad.l, 0, sw - v.pad.l - v.pad.r, sh); sg.clip();
      sg.beginPath();
      S.hist.forEach((p, i) => {
        const x = v.X(p[0]), y = v.Y(clamp(p[1] / D, -Ymax * 3, Ymax * 3));
        if (i === 0) sg.moveTo(x, y); else sg.lineTo(x, y);
      });
      sg.strokeStyle = C.blue; sg.lineWidth = 2; sg.stroke();
      sg.restore();
      text(sg, Ymax.toFixed(0) + '°', 6, v.Y(Ymax) + 4, C.muted);
      text(sg, '0', 36, v.Y(0) + 4, C.muted, 'right');
      text(sg, '-' + Ymax.toFixed(0) + '°', 6, v.Y(-Ymax) + 4, C.muted);
      text(sg, 'θ(t), simulated time (s)', sw - 12, sh - 6, C.muted, 'right');

      const dtc = 1 / rate.get(), kdt = kp.get() * dtc;
      rd.textContent =
        `control rate        ${rate.get()} Hz   →  period Δt = ${f(dtc * 1000, 1)} ms\n` +
        `loop gain           Kp·Δt = ${f(kdt, 3)}   (stable for Kp·Δt < 2)\n` +
        `target / angle      ${tgt.get()} °  /  ${f(S.th / D, 1)} °\n` +
        `① sense (noisy)     ${f(S.meas / D, 2)} °\n` +
        `② think: command    ${f(S.cmd / D, 1)} °/s\n` +
        `③ act: moved        ${f(S.move / D, 2)} ° during this Δt\n` +
        `simulated time      ${f(S.t, 2)} s`;

      let msg;
      if (S.div) msg = `The joint has run away. Kp·Δt = ${f(kdt, 2)} is above 2, so every tick overshoots more than the last. Raise the rate (shorter Δt) or lower Kp.`;
      else if (kdt >= 2) msg = `Kp·Δt = ${f(kdt, 2)} ≥ 2: each tick overshoots by more than the error it removes, so the joint oscillates with growing swings.`;
      else if (kdt > 1) msg = `Kp·Δt = ${f(kdt, 2)} is between 1 and 2: the joint overshoots and rings back to the target.`;
      else msg = `Kp·Δt = ${f(kdt, 2)} < 1: the error shrinks smoothly, by a factor (1 − Kp·Δt) = ${f(1 - kdt, 2)} each tick.`;
      cap.textContent = msg + (noise.get() > 0 ? ' Sensor noise makes every command jitter even when the loop is stable.' : '');
    };

    let last = '';
    pp.onclick = () => { S.run = !S.run; pp.textContent = S.run ? 'Pause' : 'Play'; };
    step.onclick = () => { S.run = false; pp.textContent = 'Play'; tick(); };
    reset.onclick = () => restart();
    AI.loop(root, (dt) => {
      advance(dt);
      const k = [S.t.toFixed(4), S.th.toFixed(5), S.div, tgt.get(), kp.get(), rate.get(), noise.get(), arm.w, arm.h, strip.w, theme()].join('|');
      if (k !== last) { last = k; draw(); }
    }, { fps: 60 });
  };

  /* ================= 2 · frames on a two-link arm ================= */
  AI.labs['frame-chain'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Frames on a two-link arm',
      'each link carries a frame; multiplying the transforms places the gripper (illustrative 2-D arm)',
      '<canvas style="height:330px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const a1 = slider(ctl, 'Joint 1 angle a₁', -180, 180, 1, 30, '°', 0);
    const a2 = slider(ctl, 'Joint 2 angle a₂', -180, 180, 1, 45, '°', 0);
    const L1 = slider(ctl, 'Link 1 length L₁', 0.5, 1.5, 0.05, 1, ' m', 2);
    const L2 = slider(ctl, 'Link 2 length L₂', 0.5, 1.5, 0.05, 0.8, ' m', 2);
    const px = slider(ctl, 'Point p in gripper frame, x', -0.5, 0.8, 0.05, 0.2, ' m', 2);
    const py = slider(ctl, 'Point p in gripper frame, y', -0.5, 0.5, 0.05, 0, ' m', 2);
    const rd = readout(ctl);
    const cur = { a1: 30 * D, a2: 45 * D };

    const chain = () => {
      const l1 = L1.get(), l2 = L2.get();
      const T1 = Rm(cur.a1);                 // frame 1: at the base, rotated by a1
      const T2 = mul(mul(T1, Tr(l1, 0)), Rm(cur.a2)); // frame 2: at the elbow, rotated by a1 + a2
      const T3 = mul(T2, Tr(l2, 0));         // frame 3: at the gripper
      return { T1, T2, T3, l1, l2 };
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const { T1, T2, T3, l1, l2 } = chain();
      const R = l1 + l2 + 0.3;
      const s = Math.min((w * 0.5) / R, (h * 0.55) / R);
      const X = (x) => w * 0.5 + x * s, Y = (y) => h * 0.58 - y * s;
      const axes = (M, label, len, dim) => {
        const o = app(M, [0, 0, 1]), ex = app(M, [len, 0, 0]), ey = app(M, [0, len, 0]);
        const a = dim ? C.muted : C.blue, b = dim ? C.muted : C.green;
        arrow(g, X(o[0]), Y(o[1]), X(ex[0]), Y(ex[1]), a, 2);
        arrow(g, X(o[0]), Y(o[1]), X(ey[0]), Y(ey[1]), b, 2);
        text(g, label, X(o[0]) + 6, Y(o[1]) + 14, dim ? C.muted : C.fg);
      };
      axes([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'F0 base', 0.4, true);
      axes(T1, 'F1 joint 1', 0.25, false);
      const e = app(T2, [0, 0, 1]), t = app(T3, [0, 0, 1]);
      seg(g, X(0), Y(0), X(e[0]), Y(e[1]), C.muted, 10);
      seg(g, X(e[0]), Y(e[1]), X(t[0]), Y(t[1]), C.purple, 10);
      axes(T2, 'F2 elbow', 0.25, false);
      axes(T3, 'F3 gripper', 0.25, false);
      disc(g, X(0), Y(0), 5, C.fg);
      disc(g, X(e[0]), Y(e[1]), 4, C.fg);
      const p = app(T3, [px.get(), py.get(), 1]);
      disc(g, X(p[0]), Y(p[1]), 6, C.orange);
      text(g, 'p', X(p[0]) + 8, Y(p[1]) - 6, C.orange);

      const M = T3;
      rd.textContent =
        `T₀³ (base ← gripper), rounded to 3 decimals:\n` +
        M.map((r) => '  [ ' + r.map((v) => f(v, 3).padStart(7)).join(' ') + ' ]').join('\n') + '\n' +
        `gripper origin in base frame   (${f(t[0], 3)}, ${f(t[1], 3)}) m\n` +
        `p = (${f(px.get(), 2)}, ${f(py.get(), 2)}) m in F3 → base frame (${f(p[0], 3)}, ${f(p[1], 3)}) m\n` +
        `angles a₁ = ${f(cur.a1 / D, 1)}°, a₂ = ${f(cur.a2 / D, 1)}°`;
      cap.textContent = 'Each frame is the previous frame times one rotation and one translation. The last column of the matrix is where that frame’s origin sits in the base frame. Move a₁ alone and the whole arm swings about the base; move a₂ and only the gripper side moves.';
    };

    let last = '';
    AI.loop(root, (dt) => {
      const k = 1 - Math.exp(-dt * 5);
      cur.a1 += (a1.get() * D - cur.a1) * k;
      cur.a2 += (a2.get() * D - cur.a2) * k;
      const key = [cur.a1.toFixed(5), cur.a2.toFixed(5), L1.get(), L2.get(), px.get(), py.get(), cv.w, cv.h, theme()].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 60 });
  };

  /* ================= 3 · configuration space ================= */
  AI.labs['config-space'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Configuration space: the arm as one point',
      'top: the arm and obstacle in the workspace · bottom: the same arm as a point (θ₁, θ₂); red = the arm touches the obstacle (illustrative, L₁ = L₂ = 1 m)',
      '<canvas style="height:250px"></canvas><canvas style="height:250px"></canvas><div class="caption"></div>', '');
    const [wsCv, csCv] = [...view.querySelectorAll('canvas')].map((cv) => AI.canvas(cv));
    const cap = AI.$('.caption', view);
    const ox = slider(ctl, 'Obstacle centre x', -2, 2, 0.05, 1.0, ' m', 2);
    const oy = slider(ctl, 'Obstacle centre y', -2, 2, 0.05, 0.0, ' m', 2);
    const orr = slider(ctl, 'Obstacle radius', 0.1, 0.6, 0.05, 0.3, ' m', 2);
    const speed = slider(ctl, 'Speed of the path', 0.25, 2, 0.05, 1, '×', 2);
    const [pp] = buttons(ctl, ['Pause']);
    const rd = readout(ctl);

    const QS = [-2.0, 0.3], QG = [1.0, -1.5];   // start and goal configurations (radians)
    const NG = 150;
    const S = { tau: 0, run: true };
    let gkey = '', off = null, grid = null;
    const buildGrid = () => {
      const cx = ox.get(), cy = oy.get(), r = orr.get();
      const k = cx + ',' + cy + ',' + r + ',' + theme();
      if (k === gkey) return;
      gkey = k;
      grid = new Uint8Array(NG * NG);
      for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) {
        const q1 = -Math.PI + ((i + 0.5) / NG) * TAU, q2 = -Math.PI + ((j + 0.5) / NG) * TAU;
        grid[j * NG + i] = hitArm(q1, q2, cx, cy, r) ? 1 : 0;
      }
      off = document.createElement('canvas'); off.width = NG; off.height = NG;
      const og = off.getContext('2d'), im = og.createImageData(NG, NG);
      const free = AI.rgb(AI.col().bg3), bad = AI.rgb(AI.col().red);
      for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) {
        const row = NG - 1 - j, p = (row * NG + i) * 4, c = grid[j * NG + i] ? bad : free;
        im.data[p] = c[0]; im.data[p + 1] = c[1]; im.data[p + 2] = c[2]; im.data[p + 3] = grid[j * NG + i] ? 230 : 255;
      }
      og.putImageData(im, 0, 0);
    };
    const pathHit = () => {
      for (let k = 0; k <= 400; k++) {
        const s = k / 400, q1 = QS[0] + s * (QG[0] - QS[0]), q2 = QS[1] + s * (QG[1] - QS[1]);
        if (hitArm(q1, q2, ox.get(), oy.get(), orr.get())) return s;
      }
      return -1;
    };
    const s = () => (1 - Math.cos(TAU * S.tau)) / 2;   // 0 → 1 → 0, smooth

    const draw = () => {
      buildGrid();
      const C = AI.col();
      const sv = s(), q1 = QS[0] + sv * (QG[0] - QS[0]), q2 = QS[1] + sv * (QG[1] - QS[1]);
      const inHit = hitArm(q1, q2, ox.get(), oy.get(), orr.get());
      // workspace
      const g = wsCv.ctx, w = wsCv.w, h = wsCv.h, R = 2.3, sc = Math.min(w, h) / (2 * R);
      const X = (x) => w / 2 + x * sc, Y = (y) => h / 2 - y * sc;
      g.clearRect(0, 0, w, h);
      seg(g, X(-R), Y(0), X(R), Y(0), C.border, 1);
      seg(g, X(0), Y(-R), X(0), Y(R), C.border, 1);
      g.beginPath(); g.arc(X(0), Y(0), 2 * sc, 0, TAU); g.strokeStyle = C.border; g.setLineDash([4, 4]); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
      g.beginPath(); g.arc(X(ox.get()), Y(oy.get()), orr.get() * sc, 0, TAU); g.fillStyle = 'rgba(248,81,73,0.35)'; g.fill();
      const a = arm2(q1, q2);
      const col = inHit ? C.red : C.blue;
      seg(g, X(0), Y(0), X(a.ex), Y(a.ey), col, 7);
      seg(g, X(a.ex), Y(a.ey), X(a.tx), Y(a.ty), col, 7);
      disc(g, X(0), Y(0), 6, C.fg); disc(g, X(a.ex), Y(a.ey), 4, C.fg); disc(g, X(a.tx), Y(a.ty), 5, C.green);
      text(g, 'obstacle', X(ox.get()), Y(oy.get()) + 4, C.fg, 'center');
      text(g, 'reach ring r = 2 m (dashed)', 10, 16, C.muted);

      // configuration space
      const c = csCv.ctx, cw = csCv.w, ch = csCv.h, pad = 34;
      const pw = cw - pad * 2, ph = ch - pad * 2;
      const P = (a1, a2) => [pad + ((a1 + Math.PI) / TAU) * pw, pad + ((Math.PI - a2) / TAU) * ph];
      c.clearRect(0, 0, cw, ch);
      c.imageSmoothingEnabled = false;
      c.drawImage(off, pad, pad, pw, ph);
      c.strokeStyle = C.border; c.lineWidth = 1; c.strokeRect(pad, pad, pw, ph);
      // straight path in configuration space
      seg(c, ...P(QS[0], QS[1]), ...P(QG[0], QG[1]), C.yellow, 2, [6, 4]);
      const sP = P(QS[0], QS[1]), gP = P(QG[0], QG[1]), dP = P(q1, q2);
      disc(c, sP[0], sP[1], 5, C.fg); text(c, 'S', sP[0] + 8, sP[1] - 6, C.fg);
      disc(c, gP[0], gP[1], 5, C.fg); text(c, 'G', gP[0] + 8, gP[1] - 6, C.fg);
      disc(c, dP[0], dP[1], 6, inHit ? C.red : C.green);
      text(c, 'θ₁ →  (−180° … 180°)', pad + pw / 2, ch - 8, C.muted, 'center');
      c.save(); c.translate(12, pad + ph / 2); c.rotate(-Math.PI / 2); text(c, 'θ₂ →', 0, 0, C.muted, 'center'); c.restore();
      text(c, 'configuration space', pad + 4, pad - 10, C.muted);

      const hitS = pathHit();
      rd.textContent =
        `configuration q = (θ₁, θ₂) = (${f(q1 / D, 1)}°, ${f(q2 / D, 1)}°)\n` +
        `state: ${inHit ? 'IN COLLISION (red)' : 'free'}\n` +
        `straight line S → G in configuration space: ` + (hitS < 0 ? 'free all the way\n' : `hits red at s = ${f(hitS, 2)} (fraction of the way)\n`) +
        `links L₁ = L₂ = 1.00 m; obstacle radius ${f(orr.get(), 2)} m`;
      cap.textContent = inHit
        ? 'The dot is red: some point on one of the links lies inside the obstacle. The workspace shows the touch; the configuration space shows it as a point in a red region.'
        : 'Every point of the arm is clear of the obstacle at this configuration. A planner works in the grid on the right: it must find a path of free (dark) cells from S to G. The yellow line is a straight move, which crosses red when it hits the obstacle.';
    };

    let last = '';
    pp.onclick = () => { S.run = !S.run; pp.textContent = S.run ? 'Pause' : 'Play'; };
    AI.loop(root, (dt) => {
      if (S.run) S.tau = (S.tau + dt * speed.get() / 14) % 1;   // one full back-and-forth every 14 s at 1×
      const key = [S.tau.toFixed(5), ox.get(), oy.get(), orr.get(), speed.get(), wsCv.w, wsCv.h, csCv.w, theme()].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 60 });
  };

  /* ================= 4 · reachable workspace and IK ================= */
  AI.labs['reach-2link'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Reachable workspace and the two elbow solutions',
      'a two-link planar arm reaches a ring, not a disc · inverse kinematics by the law of cosines (illustrative)',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const La = slider(ctl, 'Link 1 length L₁', 0.5, 1.5, 0.05, 1, ' m', 2);
    const Lb = slider(ctl, 'Link 2 length L₂', 0.3, 1.5, 0.05, 0.6, ' m', 2);
    const tx = slider(ctl, 'Target x', -2, 2, 0.02, 0.866, ' m', 3);
    const ty = slider(ctl, 'Target y', -2, 2, 0.02, 0.5, ' m', 3);
    const [sw] = buttons(ctl, ['Switch solution (animate)']);
    const rd = readout(ctl);
    const S = { th1: null, th2: null, pick: 0 };   // pick 0 = θ₂ > 0, 1 = θ₂ < 0

    const solve = (x, y, a, b) => {
      const d2 = x * x + y * y;
      const c2 = (d2 - a * a - b * b) / (2 * a * b);
      if (c2 < -1 || c2 > 1) return { c2, sols: null, r: Math.sqrt(d2) };
      const t2 = Math.acos(c2);
      const sols = [t2, -t2].map((u) => ({ t2: u, t1: Math.atan2(y, x) - Math.atan2(b * Math.sin(u), a + b * Math.cos(u)) }));
      return { c2, sols, r: Math.sqrt(d2) };
    };
    sw.onclick = () => { S.pick = 1 - S.pick; };

    const draw = (sol) => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const a = La.get(), b = Lb.get(), R = a + b + 0.2;
      const sc = Math.min(w, h) / (2 * R);
      const X = (x) => w / 2 + x * sc, Y = (y) => h / 2 - y * sc;
      g.beginPath(); g.arc(X(0), Y(0), (a + b) * sc, 0, TAU); g.fillStyle = C.blue; g.globalAlpha = 0.14; g.fill();
      g.beginPath(); g.arc(X(0), Y(0), Math.abs(a - b) * sc, 0, TAU); g.fillStyle = C.bg2; g.globalAlpha = 1; g.fill();
      g.globalAlpha = 1;
      seg(g, X(-R), Y(0), X(R), Y(0), C.border, 1);
      seg(g, X(0), Y(-R), X(0), Y(R), C.border, 1);
      const drawArm = (t1, t2, color, width, dash) => {
        const e = arm2(t1, t2, a, b);
        g.save(); g.setLineDash(dash || []);
        seg(g, X(0), Y(0), X(e.ex), Y(e.ey), color, width);
        seg(g, X(e.ex), Y(e.ey), X(e.tx), Y(e.ty), color, width);
        g.restore();
        disc(g, X(e.ex), Y(e.ey), 4, C.fg);
        return e;
      };
      if (sol.sols) {
        const [A, Bs] = sol.sols;
        drawArm(A.t1, A.t2, C.blue, 2, [6, 4]);
        drawArm(Bs.t1, Bs.t2, C.orange, 2, [6, 4]);
      }
      if (S.th1 !== null) {
        const e = drawArm(S.th1, S.th2, C.fg, 6);
        disc(g, X(0), Y(0), 6, C.fg);
        disc(g, X(e.tx), Y(e.ty), 4, C.green);
      }
      const tgx = tx.get(), tgy = ty.get();
      disc(g, X(tgx), Y(tgy), 6, C.green);
      text(g, 'target', X(tgx) + 9, Y(tgy) - 8, C.green);
      text(g, 'inner hole |L₁−L₂|', X(0) + 6, Y(0) - 8, C.muted);
      text(g, 'reach ring L₁+L₂', 10, 16, C.muted);
    };

    let last = '';
    AI.loop(root, (dt) => {
      const a = La.get(), b = Lb.get(), x = tx.get(), y = ty.get();
      const sol = solve(x, y, a, b);
      if (sol.sols) {
        const goal = sol.sols[S.pick];
        if (S.th1 === null) { S.th1 = goal.t1; S.th2 = goal.t2; }
        const k = 1 - Math.exp(-dt * 1.8);  // ease toward the chosen solution: joint space, not a straight line of the tip
        S.th1 += (goal.t1 - S.th1) * k;
        S.th2 += (goal.t2 - S.th2) * k;
      }
      const key = [a, b, x, y, S.pick, S.th1 === null ? 0 : S.th1.toFixed(4), S.th2 === null ? 0 : S.th2.toFixed(4), cv.w, cv.h, theme()].join('|');
      if (key === last) return;
      last = key;
      draw(sol);
      const r = sol.r, lo = Math.abs(a - b), hi = a + b;
      if (!sol.sols) {
        const why = r > hi
          ? `Unreachable: the target is r = ${f(r, 3)} m from the base, but the longest reach is L₁ + L₂ = ${f(hi, 2)} m.`
          : `Unreachable: the target lies inside the hole, r = ${f(r, 3)} m < |L₁ − L₂| = ${f(lo, 2)} m.`;
        cap.textContent = why + ' No joint angles put the tip there; the arm keeps its last pose.';
        rd.textContent = `reach window  ${f(lo, 2)} m ≤ r ≤ ${f(hi, 2)} m\n` + `target distance r = ${f(r, 3)} m   → no solution\n` + `cos θ₂ = ${f(sol.c2, 3)}  (must lie in [−1, 1])`;
        return;
      }
      const A = sol.sols[0], Bs = sol.sols[1];
      const act = S.pick === 0 ? A : Bs;
      const det = a * b * Math.sin(act.t2);
      rd.textContent =
        `reach window        ${f(lo, 2)} m ≤ r ≤ ${f(hi, 2)} m\n` +
        `target distance r   ${f(r, 3)} m\n` +
        `cos θ₂ = (r²−L₁²−L₂²)/(2L₁L₂) = ${f(sol.c2, 3)}\n` +
        `solution A (θ₂ > 0): θ₂ = ${f(A.t2 / D, 2)}°, θ₁ = ${f(A.t1 / D, 2)}°\n` +
        `solution B (θ₂ < 0): θ₂ = ${f(Bs.t2 / D, 2)}°, θ₁ = ${f(Bs.t1 / D, 2)}°\n` +
        `active pose: det J = L₁L₂ sin θ₂ = ${f(det, 3)} m²`;
      cap.textContent = Math.abs(det) < 0.05
        ? 'Both solutions sit close to a singular pose (the arm is almost straight or folded flat): the Jacobian loses rank here, so tiny tip motions need large joint motions.'
        : 'Two solutions reach the same tip: A (blue dashed) bends the elbow one way, B (orange dashed) the other. The solid arm is the active one. Press the button to fold it from one to the other; during the swing the tip wanders off the target, because the joints move linearly, not the tip.';
    }, { fps: 60 });
  };

  /* ================= 5 · redundancy in a three-link arm ================= */
  AI.labs['redundancy-3link'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Three joints, two task numbers: one spare degree of freedom',
      'the tip is held at one point while the gripper angle φ changes · links L₁ = L₂ = L₃ = 1 m (illustrative, elbow-up solutions shown)',
      '<canvas style="height:330px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const tx = slider(ctl, 'Tip target x', -2.5, 2.5, 0.05, 1.5, ' m', 2);
    const ty = slider(ctl, 'Tip target y', -2.5, 2.5, 0.05, 1.0, ' m', 2);
    const phi = slider(ctl, 'Gripper angle φ (absolute)', -180, 180, 1, 0, '°', 0);
    const [sweep] = buttons(ctl, ['Sweep φ']);
    const rd = readout(ctl);
    const LL = [1, 1, 1];
    let sweeping = false;
    sweep.onclick = () => { sweeping = !sweeping; sweep.textContent = sweeping ? 'Stop sweep' : 'Sweep φ'; };

    const ik3 = (x, y, p) => {
      const wx = x - LL[2] * Math.cos(p), wy = y - LL[2] * Math.sin(p);
      const dd = Math.hypot(wx, wy);
      const c = (dd * dd - LL[0] * LL[0] - LL[1] * LL[1]) / (2 * LL[0] * LL[1]);
      if (c < -1 - 1e-9 || c > 1 + 1e-9) return null;
      const t2 = Math.acos(clamp(c, -1, 1));
      const t1 = Math.atan2(wy, wx) - Math.atan2(LL[1] * Math.sin(t2), LL[0] + LL[1] * Math.cos(t2));
      return { t1, t2, t3: p - t1 - t2, a1: t1, a2: t1 + t2, a3: p };
    };
    const joints = (sol) => {
      const p0 = [0, 0];
      const p1 = [LL[0] * Math.cos(sol.a1), LL[0] * Math.sin(sol.a1)];
      const p2 = [p1[0] + LL[1] * Math.cos(sol.a2), p1[1] + LL[1] * Math.sin(sol.a2)];
      const p3 = [p2[0] + LL[2] * Math.cos(sol.a3), p2[1] + LL[2] * Math.sin(sol.a3)];
      return [p0, p1, p2, p3];
    };

    const draw = (x, y, p, sol) => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const R = 3.3, sc = Math.min(w, h) / (2 * R);
      const X = (q) => w / 2 + q * sc, Y = (q) => h / 2 - q * sc;
      seg(g, X(-R), Y(0), X(R), Y(0), C.border, 1);
      seg(g, X(0), Y(-R), X(0), Y(R), C.border, 1);
      // ghosts: the same tip with other gripper angles
      for (let d = -180; d < 180; d += 30) {
        const gs = ik3(x, y, d * D);
        if (!gs) continue;
        const J = joints(gs);
        for (let i = 0; i < 3; i++) seg(g, X(J[i][0]), Y(J[i][1]), X(J[i + 1][0]), Y(J[i + 1][1]), C.muted, 2);
      }
      if (sol) {
        const J = joints(sol);
        for (let i = 0; i < 3; i++) seg(g, X(J[i][0]), Y(J[i][1]), X(J[i + 1][0]), Y(J[i + 1][1]), C.blue, 7);
        for (let i = 1; i < 3; i++) disc(g, X(J[i][0]), Y(J[i][1]), 4, C.fg);
        disc(g, X(0), Y(0), 6, C.fg);
        seg(g, X(J[3][0]), Y(J[3][1]), X(J[3][0] + 0.25 * Math.cos(p)), Y(J[3][1] + 0.25 * Math.sin(p)), C.orange, 3);
      }
      disc(g, X(x), Y(y), 6, C.green);
      text(g, 'tip target', X(x) + 9, Y(y) - 8, C.green);
      text(g, 'grey: other joint vectors giving the same tip (every 30° of φ)', 10, 16, C.muted);
    };

    let last = '';
    AI.loop(root, (dt) => {
      if (sweeping) {
        let v = phi.get() + dt * 25;
        if (v > 180) v -= 360;
        phi.set(Math.round(v));
      }
      const x = tx.get(), y = ty.get(), pd = phi.get();
      const sol = ik3(x, y, pd * D);
      const key = [x, y, pd, cv.w, cv.h, theme()].join('|');
      if (key === last) return;
      last = key;
      draw(x, y, pd * D, sol);
      if (!sol) {
        cap.textContent = `With φ = ${pd}° no arm reaches this tip: the wrist point is out of reach. Other gripper angles may still work.`;
        rd.textContent = `joints: 3 revolute   task: 2 (tip x, y)   redundancy: 1\n` + `this φ: unreachable`;
        return;
      }
      const J = joints(sol);
      rd.textContent =
        `joints: 3 revolute   task: 2 (tip x, y)   redundancy = 3 − 2 = 1\n` +
        `θ₁ (base joint, absolute)  ${f(sol.t1 / D, 2)}°\n` +
        `θ₂ (elbow, relative)       ${f(sol.t2 / D, 2)}°\n` +
        `θ₃ (wrist, relative)       ${f(sol.t3 / D, 2)}°\n` +
        `gripper angle φ            ${f(pd, 0)}°   (= θ₁ + θ₂ + θ₃)\n` +
        `tip reached                (${f(J[3][0], 3)}, ${f(J[3][1], 3)}) m`;
      cap.textContent = 'The tip stays on the target while φ changes: three joints, but the tip needs only two numbers, so one degree of freedom is spare. The grey arms are all valid solutions; the blue one is the current choice. A planner can use the spare freedom to avoid obstacles or joint limits.';
    }, { fps: 60 });
  };

  /* ================= 6 · the stack: rates, latency and data age ================= */
  AI.labs['stack-flow'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Rates, bus latency and the age of data (illustrative)',
      'each stage publishes on its own clock; the controller acts on the newest estimate that has arrived',
      '<canvas style="height:280px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const rs = slider(ctl, 'Sensor rate', 10, 200, 1, 100, ' Hz', 0);
    const re = slider(ctl, 'Estimator rate', 10, 200, 1, 50, ' Hz', 0);
    const rp = slider(ctl, 'Planner rate', 1, 50, 1, 10, ' Hz', 0);
    const rc = slider(ctl, 'Controller rate', 50, 1000, 10, 500, ' Hz', 0);
    const lat = slider(ctl, 'Bus latency per hop', 0, 20, 0.5, 2, ' ms', 1);
    const speed = slider(ctl, 'Speed (simulated s per real s)', 0.05, 1, 0.05, 0.25, '×', 2);
    const [pp] = buttons(ctl, ['Pause']);
    const rd = readout(ctl);
    const S = { t: 0, run: true };
    const W = 0.25;   // window shown, seconds
    const fl = (x) => Math.floor(x + 1e-9);

    const draw = () => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h, t = S.t;
      g.clearRect(0, 0, w, h);
      const pad = { l: 110, r: 14, t: 14, b: 26 };
      const v = AI.view(cv, t - W, t, 0, 4, pad);
      const lanes = [['sensor', rs.get(), C.muted], ['estimator', re.get(), C.green], ['planner', rp.get(), C.purple], ['controller', rc.get(), C.blue]];
      const L = lat.get() / 1000;
      const tc = fl(t * rc.get()) / rc.get();                 // newest controller tick
      const tp = Math.max(-1, fl((tc - L) * re.get()) / re.get()); // newest estimate that has ARRIVED by tc
      lanes.forEach(([name, rate, color], i) => {
        const y = v.Y(3.5 - i);
        seg(g, pad.l, y, w - pad.r, y, C.border, 1);
        text(g, name + ' ' + rate.toFixed(0) + ' Hz', 8, y + 4, color);
        const k0 = Math.ceil((t - W) * rate - 1e-9), k1 = Math.floor(t * rate + 1e-9);
        for (let k = k0; k <= k1; k++) disc(g, v.X(k / rate), y, 4, color);
      });
      // controller tick and the estimate it uses
      seg(g, v.X(tc), pad.t, v.X(tc), h - pad.b, C.blue, 2);
      if (tp >= 0) {
        seg(g, v.X(tp), pad.t, v.X(tp), h - pad.b, C.green, 2, [5, 4]);
        seg(g, v.X(tp + L), pad.t, v.X(tp + L), h - pad.b, C.orange, 1, [2, 3]);
        g.fillStyle = 'rgba(88,166,255,0.10)'; g.fillRect(v.X(tp), pad.t, v.X(tc) - v.X(tp), h - pad.t - pad.b);
      }
      for (let k = 0; k <= 4; k++) text(g, (t - W + (k * W) / 4).toFixed(2) + ' s', v.X(t - W + (k * W) / 4), h - 8, C.muted, 'center', 10);
      text(g, 'orange dotted: when that estimate arrives (publish + bus latency)', pad.l, 12, C.orange, 'left', 10);

      const age = tp >= 0 ? tc - tp : NaN;
      const bound = L + 1 / re.get();
      rd.textContent =
        `now                        t = ${f(t, 3)} s\n` +
        `controller tick used       t_c = ${f(tc, 3)} s   (rate ${rc.get()} Hz)\n` +
        `newest estimate that arrived  published t_p = ${tp >= 0 ? f(tp, 3) : 'none'} s\n` +
        `data age at the controller    t_c − t_p = ${isNaN(age) ? '—' : f(age * 1000, 1) + ' ms'}\n` +
        `bound: latency + 1/estimator rate = ${f(bound * 1000, 1)} ms`;
      cap.textContent = 'The controller never sees the latest estimate, only the latest one that arrived. Its age is at least the bus latency and at most latency + one estimator period. Here the rates are not from any real robot: they show how the clocks interact.';
    };

    let last = '';
    pp.onclick = () => { S.run = !S.run; pp.textContent = S.run ? 'Pause' : 'Play'; };
    AI.loop(root, (dt) => {
      if (S.run) S.t += dt * speed.get();
      const key = [S.t.toFixed(4), rs.get(), re.get(), rp.get(), rc.get(), lat.get(), cv.w, cv.h, theme()].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 60 });
  };

  /* ================= 7 · a fixed-priority schedule ================= */
  AI.labs['stack-timeline'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A fixed-priority schedule of three control tasks (illustrative)',
      'shorter period = higher priority · each job must finish before its next release · time in milliseconds',
      '<canvas style="height:240px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const cT = slider(ctl, 'Controller period Tc', 0.5, 2, 0.05, 1.0, ' ms', 2);
    const cC = slider(ctl, 'Controller compute C', 0.05, 0.6, 0.05, 0.2, ' ms', 2);
    const eT = slider(ctl, 'Estimator period', 5, 20, 0.5, 10, ' ms', 1);
    const eC = slider(ctl, 'Estimator compute', 0.5, 5, 0.1, 2, ' ms', 1);
    const pT = slider(ctl, 'Planner period', 50, 200, 5, 100, ' ms', 0);
    const pC = slider(ctl, 'Planner compute', 5, 50, 1, 20, ' ms', 0);
    const speed = slider(ctl, 'Cursor speed', 5, 60, 1, 20, ' ms per s', 0);
    const [pp] = buttons(ctl, ['Pause']);
    const rd = readout(ctl);
    const H = 200, Q = 0.05;           // horizon 200 ms, integer ticks of 0.05 ms
    const S = { cur: 0, run: true, key: '', sch: null, tasks: null };

    const schedule = (tasks) => {
      const n = tasks.length, rem = tasks.map(() => 0), nextRel = tasks.map(() => 0);
      const blocks = tasks.map(() => []), misses = [], rels = tasks.map(() => []), HT = Math.round(H / Q);
      for (let t = 0; t < HT; t++) {
        for (let i = 0; i < n; i++) if (t === nextRel[i]) {
          if (rem[i] > 0) misses.push({ i, t });    // previous job still unfinished at its deadline
          rem[i] = tasks[i].C; rels[i].push(t); nextRel[i] += tasks[i].T;
        }
        let run = -1;
        for (let i = 0; i < n; i++) if (rem[i] > 0) { run = i; break; }
        if (run >= 0) {
          rem[run]--;
          const b = blocks[run], last = b[b.length - 1];
          if (last && last[1] === t) last[1] = t + 1; else b.push([t, t + 1]);
        }
      }
      return { blocks, misses, rels, HT };
    };
    const refresh = () => {
      const k = [cT.get(), cC.get(), eT.get(), eC.get(), pT.get(), pC.get()].join(',');
      if (k === S.key) return;
      S.key = k;
      const tk = (ms) => Math.round(ms / Q);
      S.tasks = [
        { name: 'controller', T: tk(cT.get()), C: tk(cC.get()), Tms: cT.get(), Cms: cC.get(), col: 'blue' },
        { name: 'estimator', T: tk(eT.get()), C: tk(eC.get()), Tms: eT.get(), Cms: eC.get(), col: 'green' },
        { name: 'planner', T: tk(pT.get()), C: tk(pC.get()), Tms: pT.get(), Cms: pC.get(), col: 'orange' },
      ];
      S.sch = schedule(S.tasks);
    };

    const draw = () => {
      refresh();
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const pad = { l: 130, r: 16 }, rowH = 44, top = 42;
      const xT = (ms) => pad.l + (ms / H) * (w - pad.l - pad.r);
      const sch = S.sch, tasks = S.tasks;
      const U = tasks.reduce((a, t) => a + t.Cms / t.Tms, 0);
      const B = tasks.length * (Math.pow(2, 1 / tasks.length) - 1);
      tasks.forEach((tk, i) => {
        const y0 = top + i * rowH, color = C[tk.col];
        text(g, tk.name, 8, y0 + 16, color);
        text(g, `T=${tk.Tms.toFixed(2)} C=${tk.Cms.toFixed(2)} ms`, 8, y0 + 30, C.muted, 'left', 10);
        seg(g, pad.l, y0 + rowH - 6, w - pad.r, y0 + rowH - 6, C.border, 1);
        sch.rels[i].forEach((t) => seg(g, xT(t * Q), y0 + 4, xT(t * Q), y0 + 10, C.muted, 1));
        sch.blocks[i].forEach(([s0, e0]) => { g.fillStyle = color; g.fillRect(xT(s0 * Q), y0 + 12, Math.max(1, xT(e0 * Q) - xT(s0 * Q)), 20); });
      });
      sch.misses.forEach((m) => {
        const x = xT(m.t * Q), y = top + m.i * rowH + 22;
        seg(g, x - 5, y - 9, x + 5, y + 1, C.red, 3); seg(g, x - 5, y + 1, x + 5, y - 9, C.red, 3);
      });
      // axis
      const ay = top + tasks.length * rowH + 6;
      seg(g, pad.l, ay, w - pad.r, ay, C.border, 1);
      for (let ms = 0; ms <= H; ms += 20) { seg(g, xT(ms), ay, xT(ms), ay + 4, C.muted, 1); text(g, ms + '', xT(ms), ay + 16, C.muted, 'center', 10); }
      text(g, 'time (ms)', w - pad.r, ay + 16, C.muted, 'right', 10);
      // cursor
      S.cur = S.cur % H;
      const cx = xT(S.cur);
      seg(g, cx, top - 6, cx, ay, C.fg, 1.5, [4, 3]);
      let running = 'idle';
      const tick = Math.floor(S.cur / Q + 1e-9);
      sch.blocks.forEach((bl, i) => { if (bl.some(([s0, e0]) => tick >= s0 && tick < e0)) running = tasks[i].name; });
      rd.textContent =
        `cursor t = ${S.cur.toFixed(1)} ms, running: ${running}\n` +
        `utilisation U = Σ C/T = ${U.toFixed(3)}   bound for ${tasks.length} tasks = ${B.toFixed(3)}\n` +
        `deadline misses within ${H} ms: ${sch.misses.length}`;
      if (sch.misses.length) cap.textContent = `Red crosses mark jobs still unfinished when the next job of the same task arrives: the deadline is missed. Total compute demand is ${U.toFixed(2)}, which is ${U > 1 ? 'above 1: no scheduler can finish all the work' : 'high enough that the lower-priority tasks are starved at some moments'}.`;
      else if (U <= B) cap.textContent = `U = ${U.toFixed(3)} is below the bound ${B.toFixed(3)} for three tasks, so this fixed-priority schedule is guaranteed to meet every deadline (a sufficient test: a failure of the test would not prove a miss). Ticks mark each release; the controller runs first whenever it is ready.`;
      else cap.textContent = `U = ${U.toFixed(3)} is above the bound ${B.toFixed(3)}. The simple test cannot promise schedulability, but this 200 ms run shows no miss. Watch for jobs that slip near the end of a period.`;
    };

    let last = '';
    pp.onclick = () => { S.run = !S.run; pp.textContent = S.run ? 'Pause' : 'Play'; };
    AI.loop(root, (dt) => {
      if (S.run) S.cur += dt * speed.get();
      const key = [S.cur.toFixed(2), cT.get(), cC.get(), eT.get(), eC.get(), pT.get(), pC.get(), cv.w, cv.h, theme()].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 60 });
  };
})();
