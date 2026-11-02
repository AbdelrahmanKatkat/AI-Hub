/* Robotics · Module 2 (kinematics & dynamics) labs. Planar arms and wheeled bases are simulated, not hardware.
 * fk-arm        <div data-lab="fk-arm"></div>        planar two-link forward kinematics, trajectory expressions
 * ik-arm        <div data-lab="ik-arm"></div>        drag a target; two elbow solutions; damped numerical solver
 * jac-map       <div data-lab="jac-map"></div>       joint-speed circle mapped to tip velocity; torques J^T F
 * dyn-pendulum  <div data-lab="dyn-pendulum"></div>  one-link torque, inertia and gravity, editable torque
 * odo-drive     <div data-lab="odo-drive"></div>     differential drive: true arc versus dead reckoning
 * sim-integrators <div data-lab="sim-integrators"></div> explicit versus semi-implicit Euler on a spring
 * stick-slip    <div data-lab="stick-slip"></div>    Coulomb friction: stuck or slipping block
 */
(function () {
  const D = Math.PI / 180, G = 9.81;
  const F = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const safe = (v) => (Number.isFinite(v) ? v : 0);

  /* ---------- small control helpers ---------- */
  const slider = (ctl, label, min, max, step, value, unit, dec, onInput) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = (+inp.value).toFixed(dec) + (unit ? ' ' + unit : ''));
    inp.addEventListener('input', () => { upd(); if (onInput) onInput(); });
    upd();
    ctl.appendChild(l);
    return { get: () => +inp.value, set: (x) => { inp.value = x; upd(); } };
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
    d.style.whiteSpace = 'pre-line';
    ctl.appendChild(d);
    return d;
  };
  const exprBox = (ctl, label, value, allowed, onChange) => {
    const w = document.createElement('div');
    w.innerHTML = `<div style="font-size:12px;color:var(--muted);margin-top:6px">${label}</div>`
      + `<input type="text" spellcheck="false" value="${value}" style="width:100%;font-family:ui-monospace,Menlo,Consolas,monospace">`
      + `<div class="expr-err" style="color:var(--red);font-size:12px;min-height:16px"></div>`;
    ctl.appendChild(w);
    const inp = w.querySelector('input'), err = w.querySelector('.expr-err');
    const ei = AI.exprInput(inp, err, { allowed, onChange });
    ei.run();
    return ei;
  };
  const seg = (g, x0, y0, x1, y1, color, width) => {
    g.strokeStyle = color; g.lineWidth = width; g.lineCap = 'round';
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
  };
  const dot = (g, x, y, r, color) => { g.fillStyle = color; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill(); };
  const label = (g, text, x, y, color, align = 'left') => {
    g.fillStyle = color; g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = align; g.fillText(text, x, y);
  };

  /* ---------- 1 · forward kinematics ---------- */
  AI.labs['fk-arm'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Forward kinematics of a planar arm',
      'joint angles in, tip position out; the elbow angle is measured relative to the upper arm',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const sL1 = slider(ctl, 'Upper arm l₁', 0.5, 1.5, 0.05, 1.0, 'm', 2, () => (dirty = true));
    const sL2 = slider(ctl, 'Forearm l₂', 0.2, 1.0, 0.05, 0.5, 'm', 2, () => (dirty = true));
    const sT1 = slider(ctl, 'Shoulder θ₁', -180, 180, 1, 30, '°', 0, () => (dirty = true));
    const sT2 = slider(ctl, 'Elbow θ₂ (relative)', -180, 180, 1, 45, '°', 0, () => (dirty = true));
    const sSpeed = slider(ctl, 'Trajectory speed', 0.1, 1, 0.05, 0.3, '×', 2);
    const [bTraj, bPause, bReset] = buttons(ctl, ['Trajectory: off', '❚❚ Pause', 'Clear trail']);
    let fn1 = () => 0, fn2 = () => 0;
    exprBox(ctl, 'θ₁(t) in radians, used in trajectory mode (t in seconds)', 'pi/3*sin(0.3*t)', ['t'], (f) => (fn1 = f));
    exprBox(ctl, 'θ₂(t) in radians, used in trajectory mode', 'pi/2*sin(0.2*t)', ['t'], (f) => (fn2 = f));
    const rd = readout(ctl);

    let trajOn = false, running = true, tt = 0;
    let cur1 = sT1.get() * D, cur2 = sT2.get() * D;
    let trail = [];
    const tipOf = (a1, a2) => {
      const L1 = sL1.get(), L2 = sL2.get();
      const ex = L1 * Math.cos(a1), ey = L1 * Math.sin(a1);
      return { ex, ey, tx: ex + L2 * Math.cos(a1 + a2), ty: ey + L2 * Math.sin(a1 + a2) };
    };

    bTraj.onclick = () => {
      trajOn = !trajOn; tt = 0; trail = []; dirty = true;
      bTraj.textContent = trajOn ? 'Trajectory: on' : 'Trajectory: off';
    };
    bPause.onclick = () => { running = !running; bPause.textContent = running ? '❚❚ Pause' : '▶ Play'; };
    bReset.onclick = () => { trail = []; dirty = true; };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      const L1 = sL1.get(), L2 = sL2.get();
      const ox = W / 2, oy = H * 0.74, S = Math.min((W - 24) / 5.2, (H * 0.74 - 14) / 2.6);
      const P = (x, y) => [ox + x * S, oy - y * S];
      g.clearRect(0, 0, W, H);
      g.setLineDash([5, 5]); g.strokeStyle = C.border; g.lineWidth = 1;
      for (const r of [L1 + L2, Math.abs(L1 - L2)]) { g.beginPath(); g.arc(ox, oy, r * S, Math.PI, 2 * Math.PI); g.stroke(); }
      g.setLineDash([]);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(12, oy); g.lineTo(W - 12, oy); g.stroke();
      if (trail.length > 1) {
        g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath();
        trail.forEach((p, i) => { const [X, Y] = P(p[0], p[1]); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); });
        g.stroke();
      }
      const k = tipOf(cur1, cur2);
      const [X0, Y0] = P(0, 0), [X1, Y1] = P(k.ex, k.ey), [X2, Y2] = P(k.tx, k.ty);
      seg(g, X0, Y0, X1, Y1, C.blue, 8);
      seg(g, X1, Y1, X2, Y2, C.green, 8);
      dot(g, X0, Y0, 6, C.fg); dot(g, X1, Y1, 5, C.fg); dot(g, X2, Y2, 6, C.orange);
      label(g, 'shoulder', X0 + 10, Y0 + 16, C.muted);
      label(g, 'elbow', X1 + 8, Y1 - 8, C.muted);
      label(g, `tip (${F(k.tx, 3)}, ${F(k.ty, 3)}) m`, X2 + 8, Y2 - 8, C.orange);
      const abs = (cur1 + cur2) / D, r = Math.hypot(k.tx, k.ty);
      rd.textContent =
        `θ₁ = ${F(cur1 / D, 1)}°    θ₂ = ${F(cur2 / D, 1)}° (relative to upper arm)    absolute forearm = θ₁+θ₂ = ${F(abs, 1)}°\n`
        + `elbow = (${F(k.ex, 3)}, ${F(k.ey, 3)}) m    tip = (${F(k.tx, 3)}, ${F(k.ty, 3)}) m\n`
        + `distance from shoulder r = ${F(r, 3)} m    reach band [${F(Math.abs(L1 - L2), 2)}, ${F(L1 + L2, 2)}] m`;
      cap.textContent = trajOn
        ? 'Trajectory mode: θ₁ and θ₂ follow your expressions. The purple trail is the path of the tip, and it never leaves the reach band.'
        : 'Forward kinematics: the elbow sits at l₁(cos θ₁, sin θ₁). The forearm points at the absolute angle θ₁+θ₂, so the tip adds l₂(cos(θ₁+θ₂), sin(θ₁+θ₂)).';
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running && trajOn) { tt += dt * sSpeed.get(); changed = true; }
      let g1, g2;
      if (trajOn) { g1 = safe(fn1({ t: tt })); g2 = safe(fn2({ t: tt })); }
      else { g1 = sT1.get() * D; g2 = sT2.get() * D; }
      const k = 1 - Math.exp(-dt * 5);
      const n1 = cur1 + (g1 - cur1) * k, n2 = cur2 + (g2 - cur2) * k;
      if (Math.abs(n1 - cur1) + Math.abs(n2 - cur2) > 1e-7) changed = true;
      cur1 = n1; cur2 = n2;
      if (changed) {
        const t = tipOf(cur1, cur2), last = trail[trail.length - 1];
        if (!last || Math.hypot(t.tx - last[0], t.ty - last[1]) > 0.004) trail.push([t.tx, t.ty]);
        if (trail.length > 400) trail.shift();
        draw();
      }
    }, { fps: 60 });
  };

  /* ---------- 2 · inverse kinematics ---------- */
  AI.labs['ik-arm'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Inverse kinematics: reaching a target',
      'drag the target; inside the reach band there are two elbow solutions (links fixed at 1.00 m and 0.50 m)',
      '<canvas style="height:320px;touch-action:none;cursor:crosshair"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), cvEl = AI.$('canvas', view), c = AI.canvas(cvEl);
    const L1 = 1.0, L2 = 0.5;
    let tx = 1.0, ty = 0.8, dirty = true, dragging = false;
    let elbow = 'up', goalCF = true, solving = false, iters = 0, solveAcc = 0, solveErr = NaN;
    let path = [];
    const solveCF = (x, y) => {
      const r2 = x * x + y * y, cq = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), ok = Math.abs(cq) <= 1;
      const alpha = Math.atan2(y, x), q2u = ok ? Math.acos(cq) : 0;
      const beta = (q2) => Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
      return { ok, r: Math.sqrt(r2), cq, alpha, up: ok ? [alpha - beta(q2u), q2u] : null, down: ok ? [alpha - beta(-q2u), -q2u] : null };
    };
    const fk = (a1, a2) => [L1 * Math.cos(a1) + L2 * Math.cos(a1 + a2), L1 * Math.sin(a1) + L2 * Math.sin(a1 + a2)];
    let cur1, cur2;
    { const s = solveCF(tx, ty); [cur1, cur2] = s.up; }

    const lam = slider(ctl, 'Solver damping λ (numerical only)', 0.01, 0.5, 0.01, 0.1, 'm', 2, null);
    const [bUp, bDown, bSolve, bReset] = buttons(ctl, ['Elbow up', 'Elbow down', 'Solve numerically', 'Reset target']);
    const rd = readout(ctl);

    const geom = () => {
      const W = c.w, H = c.h;
      return { ox: W * 0.5, oy: H * 0.82, S: Math.min((W - 30) / 3.2, (H * 0.82 - 16) / 1.7) };
    };
    const setTarget = (e) => {
      const r = cvEl.getBoundingClientRect(), G = geom();
      tx = AI.clamp((e.clientX - r.left - G.ox) / G.S, -1.6, 1.6);
      ty = AI.clamp((G.oy - (e.clientY - r.top)) / G.S, 0, 1.65);
      goalCF = true; solving = false; dirty = true; path = [];
    };
    cvEl.addEventListener('pointerdown', (e) => { dragging = true; cvEl.setPointerCapture(e.pointerId); setTarget(e); });
    cvEl.addEventListener('pointermove', (e) => { if (dragging) setTarget(e); });
    const stopDrag = () => (dragging = false);
    cvEl.addEventListener('pointerup', stopDrag);
    cvEl.addEventListener('pointercancel', stopDrag);

    bUp.onclick = () => { elbow = 'up'; goalCF = true; solving = false; path = []; dirty = true; };
    bDown.onclick = () => { elbow = 'down'; goalCF = true; solving = false; path = []; dirty = true; };
    bSolve.onclick = () => {
      cur1 = 0; cur2 = 0; iters = 0; solveAcc = 0; solveErr = NaN;
      solving = true; goalCF = false; path = []; dirty = true;
    };
    bReset.onclick = () => { tx = 1.0; ty = 0.8; elbow = 'up'; goalCF = true; solving = false; path = []; dirty = true; };

    /* one damped least-squares step: dq = J^T (J J^T + λ² I)^-1 e */
    const iterate = () => {
      const s1 = Math.sin(cur1), c1 = Math.cos(cur1), s12 = Math.sin(cur1 + cur2), c12 = Math.cos(cur1 + cur2);
      const J11 = -L1 * s1 - L2 * s12, J12 = -L2 * s12, J21 = L1 * c1 + L2 * c12, J22 = L2 * c12;
      const [fx, fy] = fk(cur1, cur2);
      const ex = tx - fx, ey = ty - fy, l2 = lam.get() ** 2;
      const a = J11 * J11 + J12 * J12 + l2, b = J11 * J21 + J12 * J22, d = J21 * J21 + J22 * J22 + l2;
      const det = a * d - b * b;
      const y1 = (d * ex - b * ey) / det, y2 = (-b * ex + a * ey) / det;
      cur1 += J11 * y1 + J21 * y2;
      cur2 += J12 * y1 + J22 * y2;
      iters++;
      const [nx, ny] = fk(cur1, cur2);
      solveErr = Math.hypot(tx - nx, ty - ny);
      path.push([nx, ny]);
      if (solveErr < 1e-3 || iters >= 60) solving = false;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      const { ox, oy, S } = geom();
      const P = (x, y) => [ox + x * S, oy - y * S];
      g.clearRect(0, 0, W, H);
      /* reach band */
      g.fillStyle = C.bg3; g.beginPath();
      g.arc(ox, oy, (L1 + L2) * S, Math.PI, 2 * Math.PI); g.arc(ox, oy, Math.abs(L1 - L2) * S, 2 * Math.PI, Math.PI, true);
      g.closePath(); g.fill();
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.lineWidth = 1;
      for (const r of [L1 + L2, Math.abs(L1 - L2)]) { g.beginPath(); g.arc(ox, oy, r * S, Math.PI, 2 * Math.PI); g.stroke(); }
      g.setLineDash([]);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(10, oy); g.lineTo(W - 10, oy); g.stroke();
      const s = solveCF(tx, ty);
      const reachable = s.ok;
      if (reachable) {
        for (const [name, sol] of [['up', s.up], ['down', s.down]]) {
          const e1 = L1 * Math.cos(sol[0]), e2 = L1 * Math.sin(sol[0]);
          const [X0, Y0] = P(0, 0), [X1, Y1] = P(e1, e2), [X2, Y2] = P(e1 + L2 * Math.cos(sol[0] + sol[1]), e2 + L2 * Math.sin(sol[0] + sol[1]));
          const chosen = name === elbow;
          if (!chosen) { g.setLineDash([5, 5]); seg(g, X0, Y0, X1, Y1, C.muted, 2); seg(g, X1, Y1, X2, Y2, C.muted, 2); g.setLineDash([]); }
          label(g, name === 'up' ? 'elbow up' : 'elbow down', X1 + 6, Y1 - 6, chosen ? C.blue : C.muted);
        }
      }
      if (path.length > 1) {
        for (const p of path) { const [X, Y] = P(p[0], p[1]); dot(g, X, Y, 2.5, C.purple); }
      }
      const e1 = L1 * Math.cos(cur1), e2 = L1 * Math.sin(cur1);
      const [X0, Y0] = P(0, 0), [X1, Y1] = P(e1, e2), [X2, Y2] = P(e1 + L2 * Math.cos(cur1 + cur2), e2 + L2 * Math.sin(cur1 + cur2));
      seg(g, X0, Y0, X1, Y1, C.blue, 8); seg(g, X1, Y1, X2, Y2, C.green, 8);
      dot(g, X0, Y0, 6, C.fg); dot(g, X1, Y1, 5, C.fg);
      const [TX, TY] = P(tx, ty);
      g.strokeStyle = reachable ? C.green : C.red; g.lineWidth = 2;
      g.beginPath(); g.arc(TX, TY, 9, 0, Math.PI * 2); g.stroke();
      g.beginPath(); g.moveTo(TX - 14, TY); g.lineTo(TX + 14, TY); g.moveTo(TX, TY - 14); g.lineTo(TX, TY + 14); g.stroke();
      label(g, `target (${F(tx, 2)}, ${F(ty, 2)}) m`, TX + 14, TY - 12, reachable ? C.green : C.red);
      const rLine = `target distance r = ${F(s.r, 3)} m, reach band [${F(Math.abs(L1 - L2), 2)}, ${F(L1 + L2, 2)}] m`;
      if (reachable) {
        rd.textContent = `${rLine}\n`
          + `cos θ₂ = (r²−l₁²−l₂²)/(2l₁l₂) = ${F(s.cq, 3)}  →  θ₂ = ±${F(Math.acos(s.cq) / D, 2)}°\n`
          + `elbow up:   θ₁ = ${F(s.up[0] / D, 2)}°, θ₂ = ${F(s.up[1] / D, 2)}°\n`
          + `elbow down: θ₁ = ${F(s.down[0] / D, 2)}°, θ₂ = ${F(s.down[1] / D, 2)}°\n`
          + `arm now: θ₁ = ${F(cur1 / D, 2)}°, θ₂ = ${F(cur2 / D, 2)}°   tip error = ${F(Math.hypot(tx - fk(cur1, cur2)[0], ty - fk(cur1, cur2)[1]) * 1000, 2)} mm`
          + (solving || iters ? `\nsolver: iteration ${iters}, error = ${F(solveErr * 1000, 2)} mm` : '');
      } else {
        rd.textContent = `${rLine}\ncos θ₂ = ${F(s.cq, 3)}, which is outside [−1, 1], so there is no joint solution.`;
      }
      if (!reachable) cap.textContent = `No solution: the target is ${F(s.r, 2)} m from the shoulder, but the arm reaches only up to ${F(L1 + L2, 2)} m. The arm points toward it as the closest pose.`;
      else if (solving || (iters && !goalCF)) cap.textContent = 'Numerical solver: each step moves the joints along the damped Jacobian direction. The purple dots are the tip positions after each iteration. The solver settles on the solution nearest its start, which is not always the elbow you chose.';
      else if (Math.abs(s.cq) > 0.97) cap.textContent = 'Near the edge of the band the two elbow solutions merge, and small target changes need large joint changes. This is where closed-form solutions are most sensitive.';
      else cap.textContent = `Two solutions reach the target. Solid: the chosen ${elbow} elbow. Dashed: the other one. Both satisfy forward kinematics exactly.`;
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (solving) {
        solveAcc += dt * 2;
        let n = Math.min(Math.floor(solveAcc), 10); solveAcc -= Math.floor(solveAcc);
        for (let i = 0; i < n && solving; i++) iterate();
        changed = true;
      }
      if (!solving && goalCF) {
        const s = solveCF(tx, ty);
        let g1, g2;
        if (s.ok) { const p = elbow === 'up' ? s.up : s.down; g1 = p[0]; g2 = p[1]; }
        else { g1 = s.alpha; g2 = 0; }
        const k = 1 - Math.exp(-dt * 4);
        const n1 = cur1 + (g1 - cur1) * k, n2 = cur2 + (g2 - cur2) * k;
        if (Math.abs(n1 - cur1) + Math.abs(n2 - cur2) > 1e-6) changed = true;
        cur1 = n1; cur2 = n2;
      }
      if (changed) draw();
    }, { fps: 60 });
  };

  /* ---------- 3 · Jacobian ---------- */
  AI.labs['jac-map'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The Jacobian: joint speeds to tip velocity',
      'left: the arm and its tip velocity; right: the tip velocity of every unit joint speed (an ellipse)',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const L1 = 1.0, L2 = 0.5;
    const sT1 = slider(ctl, 'Shoulder θ₁', -180, 180, 1, 30, '°', 0, () => (dirty = true));
    const sT2 = slider(ctl, 'Elbow θ₂', -180, 180, 1, 45, '°', 0, () => (dirty = true));
    const sFx = slider(ctl, 'Tip force Fx', -10, 10, 0.5, 10, 'N', 1, () => (dirty = true));
    const sFy = slider(ctl, 'Tip force Fy', -10, 10, 0.5, 0, 'N', 1, () => (dirty = true));
    const [bPlay] = buttons(ctl, ['❚❚ Pause']);
    const rd = readout(ctl);
    let running = true, phi = 0, cur1 = sT1.get() * D, cur2 = sT2.get() * D;
    bPlay.onclick = () => { running = !running; bPlay.textContent = running ? '❚❚ Pause' : '▶ Play'; };

    const jac = (a1, a2) => {
      const s1 = Math.sin(a1), c1 = Math.cos(a1), s12 = Math.sin(a1 + a2), c12 = Math.cos(a1 + a2);
      return [[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]];
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const J = jac(cur1, cur2), det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      /* left: arm */
      const ox = W * 0.24, oy = H * 0.66, S = Math.min((W * 0.24 - 8) / 1.6, (H * 0.66 - 14) / 1.6);
      const P = (x, y) => [ox + x * S, oy - y * S];
      const e1 = L1 * Math.cos(cur1), e2 = L1 * Math.sin(cur1);
      const tx = e1 + L2 * Math.cos(cur1 + cur2), ty = e2 + L2 * Math.sin(cur1 + cur2);
      const [X0, Y0] = P(0, 0), [X1, Y1] = P(e1, e2), [X2, Y2] = P(tx, ty);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(6, oy); g.lineTo(W * 0.5 - 6, oy); g.stroke();
      seg(g, X0, Y0, X1, Y1, C.blue, 8); seg(g, X1, Y1, X2, Y2, C.green, 8);
      dot(g, X0, Y0, 6, C.fg); dot(g, X1, Y1, 5, C.fg); dot(g, X2, Y2, 5, C.orange);
      /* velocity arrow for the current joint-speed direction (0.4 m per m/s) */
      const vx = J[0][0] * Math.cos(phi) + J[0][1] * Math.sin(phi), vy = J[1][0] * Math.cos(phi) + J[1][1] * Math.sin(phi);
      AI.arrow(g, X2, Y2, X2 + vx * 0.4 * S, Y2 - vy * 0.4 * S, C.orange, 2.5);
      /* joint-speed circle inset */
      const ix = 34, iy = H - 34, ir = 24;
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.arc(ix, iy, ir, 0, Math.PI * 2); g.stroke();
      dot(g, ix + ir * Math.cos(phi), iy - ir * Math.sin(phi), 4, C.purple);
      label(g, 'q̇ (rad/s)', ix + ir + 6, iy + 4, C.muted);
      /* right: task-space ellipse (image of the unit joint-speed circle) */
      const cx = W * 0.74, cy = H * 0.5, T = Math.min((W * 0.5 - 16) / 3.2, (H * 0.78 - 16) / 3.2);
      const Q = (x, y) => [cx + x * T, cy - y * T];
      g.strokeStyle = C.border; g.lineWidth = 1;
      g.beginPath(); g.moveTo(cx - 1.6 * T, cy); g.lineTo(cx + 1.6 * T, cy); g.moveTo(cx, cy - 1.6 * T); g.lineTo(cx, cy + 1.6 * T); g.stroke();
      g.fillStyle = C.bg3; g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 120; i++) {
        const a = (i / 120) * Math.PI * 2, x = J[0][0] * Math.cos(a) + J[0][1] * Math.sin(a), y = J[1][0] * Math.cos(a) + J[1][1] * Math.sin(a);
        const [X, Y] = Q(x, y); if (i) g.lineTo(X, Y); else g.moveTo(X, Y);
      }
      g.closePath(); g.fill(); g.stroke();
      const [DX, DY] = Q(vx, vy); dot(g, DX, DY, 5, C.orange);
      label(g, 'ẋ (m/s)', cx + 1.6 * T - 4, cy - 6, C.muted, 'right');
      label(g, 'ẏ (m/s)', cx + 6, cy - 1.6 * T + 10, C.muted);
      /* torques */
      const Fx = sFx.get(), Fy = sFy.get();
      const t1 = J[0][0] * Fx + J[1][0] * Fy, t2 = J[0][1] * Fx + J[1][1] * Fy;
      rd.textContent =
        `J = [[${F(J[0][0], 3)}, ${F(J[0][1], 3)}], [${F(J[1][0], 3)}, ${F(J[1][1], 3)}]] m/rad\n`
        + `det J = ${F(det, 4)} m²  (= l₁l₂ sin θ₂ = ${F(L1 * L2 * Math.sin(cur2), 4)})\n`
        + `joint speed (cos φ, sin φ) = (${F(Math.cos(phi), 2)}, ${F(Math.sin(phi), 2)}) rad/s → tip velocity (${F(vx, 3)}, ${F(vy, 3)}) m/s\n`
        + `tip force F = (${F(Fx, 1)}, ${F(Fy, 1)}) N → joint torques τ = Jᵀ F = (${F(t1, 2)}, ${F(t2, 2)}) N·m`;
      cap.textContent = Math.abs(det) < 0.05
        ? 'Near-singular: the ellipse has collapsed toward a line, so the arm cannot move the tip across it, and joint speeds for a sideways move become very large.'
        : 'The moving dot is one unit joint-speed direction; its image on the right is the tip velocity J·q̇. The ellipse is the image of all of them, and it flattens to a line exactly when det J = 0.';
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running) { phi += dt * 0.5; changed = true; }
      const g1 = sT1.get() * D, g2 = sT2.get() * D, k = 1 - Math.exp(-dt * 4);
      const n1 = cur1 + (g1 - cur1) * k, n2 = cur2 + (g2 - cur2) * k;
      if (Math.abs(n1 - cur1) + Math.abs(n2 - cur2) > 1e-6) changed = true;
      cur1 = n1; cur2 = n2;
      if (changed) draw();
    }, { fps: 30 });
  };

  /* ---------- 4 · manipulator dynamics (one link) ---------- */
  AI.labs['dyn-pendulum'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One-link arm: torque, inertia and gravity',
      'I_o q̈ = τ − m g lc cos q − b q̇, simulated (illustrative: one link, not a real motor)',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const sM = slider(ctl, 'Mass m', 0.5, 5, 0.1, 2, 'kg', 1, () => (dirty = true));
    const sLc = slider(ctl, 'Centre distance lc (from joint)', 0.1, 0.5, 0.01, 0.25, 'm', 2, () => (dirty = true));
    const sI = slider(ctl, 'Centroidal inertia I', 0, 0.2, 0.005, 0.05, 'kg·m²', 3, () => (dirty = true));
    const sB = slider(ctl, 'Viscous friction b', 0, 2, 0.01, 0, 'N·m·s', 2, () => (dirty = true));
    const sSpeed = slider(ctl, 'Simulation speed', 0.1, 1, 0.05, 0.5, '× real time', 2);
    const [bPlay, bStep, bReset, bComp] = buttons(ctl, ['❚❚ Pause', 'Step 0.1 s', 'Reset (horizontal, at rest)', 'Gravity compensation']);
    let fnT = () => 0;
    const ei = exprBox(ctl, 'Motor torque τ in N·m; variables t (s), q (rad, from horizontal, up positive), w (rad/s), m, lc, g', '0', ['t', 'q', 'w', 'm', 'lc', 'g'], (f) => (fnT = f));
    const rd = readout(ctl);

    const H = 0.004;
    let q = 0, w = 0, t = 0, running = true, hist = [], nextS = 0;
    const Io = () => sM.get() * sLc.get() ** 2 + sI.get();
    const acc = (tt, qq, ww) => {
      const m = sM.get(), lc = sLc.get();
      let tau = safe(fnT({ t: tt, q: qq, w: ww, m, lc, g: G }));
      return (tau - m * G * lc * Math.cos(qq) - sB.get() * ww) / Io();
    };
    const rk = (h) => {
      const a1 = acc(t, q, w);
      const q2 = q + 0.5 * h * w, w2 = w + 0.5 * h * a1, a2 = acc(t + 0.5 * h, q2, w2);
      const q3 = q + 0.5 * h * w2, w3 = w + 0.5 * h * a2, a3 = acc(t + 0.5 * h, q3, w3);
      const q4 = q + h * w3, w4 = w + h * a3, a4 = acc(t + h, q4, w4);
      q += (h / 6) * (w + 2 * w2 + 2 * w3 + w4);
      w += (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
      t += h;
      if (t >= nextS) { hist.push([t, q / D]); nextS += 0.05; if (hist.length > 240) hist.shift(); }
    };
    const tick = AI.ticker();
    bPlay.onclick = () => { running = !running; bPlay.textContent = running ? '❚❚ Pause' : '▶ Play'; dirty = true; };
    bStep.onclick = () => { for (let i = 0; i < 25; i++) rk(H); dirty = true; };
    bReset.onclick = () => { q = 0; w = 0; t = 0; nextS = 0; hist = []; dirty = true; };
    bComp.onclick = () => { ei.set('m*g*lc*cos(q)'); dirty = true; };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h;
      g.clearRect(0, 0, W, Hh);
      const m = sM.get(), lc = sLc.get(), Lr = 2 * lc;
      const px = W * 0.5, py = Hh * 0.42, S = Math.min(W * 0.3, Hh * 0.38);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(px - S * 1.1, py); g.lineTo(px + S * 1.1, py); g.stroke();
      const tipX = px + Lr * S * Math.cos(q), tipY = py - Lr * S * Math.sin(q);
      seg(g, px, py, tipX, tipY, C.blue, 9);
      dot(g, px, py, 7, C.fg);
      const cX = px + lc * S * Math.cos(q), cY = py - lc * S * Math.sin(q);
      dot(g, cX, cY, 5, C.orange);
      /* gravity arrow at the centre of mass */
      AI.arrow(g, cX, cY, cX, cY + 34, C.red, 2);
      label(g, 'mg', cX + 8, cY + 30, C.red);
      label(g, 'joint', px + 10, py + 18, C.muted);
      label(g, 'centre of mass', cX + 10, cY - 10, C.orange);
      /* strip chart of q for the last 12 s */
      const sy0 = Hh * 0.86, sy1 = Hh * 0.97, sx0 = 36, sx1 = W - 12, tEnd = t, tStart = Math.max(0, tEnd - 12);
      g.strokeStyle = C.border; g.lineWidth = 1;
      for (const v of [-90, 0, 90]) { const yy = sy1 - ((v + 180) / 360) * (sy1 - sy0); g.beginPath(); g.moveTo(sx0, yy); g.lineTo(sx1, yy); g.stroke(); label(g, v + '°', sx0 - 4, yy + 4, C.muted, 'right'); }
      g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath();
      hist.forEach((p, i) => {
        const X = sx0 + ((p[0] - tStart) / 12) * (sx1 - sx0), Y = sy1 - ((p[1] + 180) / 360) * (sy1 - sy0);
        if (i) g.lineTo(X, Y); else g.moveTo(X, Y);
      });
      g.stroke();
      /* readouts */
      const tauG = m * G * lc * Math.cos(q), tau = safe(fnT({ t, q, w, m, lc, g: G })), qdd = acc(t, q, w);
      rd.textContent =
        `q = ${F(q / D, 1)}°   q̇ = ${F(w / D, 1)}°/s   t = ${F(t, 2)} s\n`
        + `τ (motor) = ${F(tau, 3)} N·m   gravity m·g·lc·cos q = ${F(tauG, 3)} N·m\n`
        + `I_o = m·lc² + I = ${F(Io(), 4)} kg·m²   q̈ = ${F(qdd, 2)} rad/s²\n`
        + `holding torque at horizontal, m·g·lc = ${F(m * G * lc, 3)} N·m`;
      const net = tau - tauG - sB.get() * w;
      if (Math.abs(net) < 1e-3 && Math.abs(w) < 1e-3) cap.textContent = 'Balanced: the motor torque equals the gravity torque, so nothing accelerates. Try the gravity compensation button, then change the torque.';
      else cap.textContent = `Net torque τ − m g lc cos q − b q̇ = ${F(net, 3)} N·m, so q̈ = net / I_o = ${F(qdd, 2)} rad/s². Positive angular acceleration lifts the link.`;
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running) {
        const n = tick(dt, sSpeed.get() / H);
        for (let i = 0; i < n; i++) rk(H);
        if (n > 0) changed = true;
      }
      if (changed) draw();
    }, { fps: 60 });
  };

  /* ---------- 5 · differential drive and odometry ---------- */
  AI.labs['odo-drive'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Differential drive: true motion and odometry',
      'illustrative: a simulated robot. The orange ghost is dead reckoning with a wrong wheel-radius scale and coarse steps.',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const mark = () => (dirty = true);
    const sWR = slider(ctl, 'Right wheel ωR', -10, 10, 0.5, 10, 'rad/s', 1, mark);
    const sWL = slider(ctl, 'Left wheel ωL', -10, 10, 0.5, 8, 'rad/s', 1, mark);
    const sR = slider(ctl, 'Wheel radius r', 0.03, 0.08, 0.001, 0.05, 'm', 3, mark);
    const sL = slider(ctl, 'Track width L', 0.2, 0.4, 0.01, 0.30, 'm', 2, mark);
    const sScale = slider(ctl, 'Odometry radius scale (wrong r)', 0.9, 1.1, 0.01, 1.0, '×', 2, mark);
    const sH = slider(ctl, 'Odometry step h', 0.05, 1.0, 0.05, 1.0, 's', 2, mark);
    const sSpeed = slider(ctl, 'Speed', 0.1, 2, 0.05, 0.5, '×', 2);
    const [bPlay, bStep, bReset] = buttons(ctl, ['❚❚ Pause', 'Step 0.5 s', 'Reset pose']);
    const rd = readout(ctl);

    let running = true, t = 0, odoAcc = 0;
    const T = { x: 0, y: 0, th: 0, pr: 0, pl: 0 }, O = { x: 0, y: 0, th: 0 };
    let trueTr = [], odoTr = [];
    const cmd = () => { const r = sR.get(), L = sL.get(), wr = sWR.get(), wl = sWL.get(); return { v: (r * (wr + wl)) / 2, w: (r * (wr - wl)) / L }; };
    const advTrue = (dt) => {
      const { v, w } = cmd(), th = T.th, th2 = th + w * dt;
      if (Math.abs(w) < 1e-9) { T.x += v * Math.cos(th) * dt; T.y += v * Math.sin(th) * dt; }
      else { T.x += (v / w) * (Math.sin(th2) - Math.sin(th)); T.y += (v / w) * (Math.cos(th) - Math.cos(th2)); }
      T.th = th2;
      T.pr += sWR.get() * dt; T.pl += sWL.get() * dt;
    };
    const odoStep = (h) => {
      const { v, w } = cmd(), s = sScale.get();
      O.x += s * v * Math.cos(O.th) * h; O.y += s * v * Math.sin(O.th) * h; O.th += s * w * h;
    };
    const advance = (ds) => {
      advTrue(ds);
      odoAcc += ds;
      const h = sH.get();
      while (odoAcc >= h) { odoStep(h); odoAcc -= h; }
      t += ds;
      const lt = trueTr[trueTr.length - 1];
      if (!lt || Math.hypot(T.x - lt[0], T.y - lt[1]) > 0.01) trueTr.push([T.x, T.y]);
      const lo = odoTr[odoTr.length - 1];
      if (!lo || Math.hypot(O.x - lo[0], O.y - lo[1]) > 0.01) odoTr.push([O.x, O.y]);
      if (trueTr.length > 1500) trueTr.shift();
      if (odoTr.length > 1500) odoTr.shift();
    };
    bPlay.onclick = () => { running = !running; bPlay.textContent = running ? '❚❚ Pause' : '▶ Play'; mark(); };
    bStep.onclick = () => { advance(0.5); mark(); };
    bReset.onclick = () => {
      Object.assign(T, { x: 0, y: 0, th: 0, pr: 0, pl: 0 }); Object.assign(O, { x: 0, y: 0, th: 0 });
      t = 0; odoAcc = 0; trueTr = []; odoTr = []; mark();
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h;
      const S = Math.min(W / 9, Hh / 4.5), ox = W * 0.3, oy = Hh * 0.6;
      const P = (x, y) => [ox + x * S, oy - y * S];
      g.clearRect(0, 0, W, Hh);
      g.strokeStyle = C.border; g.lineWidth = 1; g.globalAlpha = 0.5;
      for (let i = -6; i <= 12; i++) { const [X] = P(i * 0.5, 0); g.beginPath(); g.moveTo(X, 0); g.lineTo(X, Hh); g.stroke(); }
      for (let j = -4; j <= 6; j++) { const [, Y] = P(0, j * 0.5); g.beginPath(); g.moveTo(0, Y); g.lineTo(W, Y); g.stroke(); }
      g.globalAlpha = 1;
      /* scale bar */
      seg(g, 14, Hh - 14, 14 + S, Hh - 14, C.fg, 2); label(g, '1 m', 14, Hh - 20, C.muted);
      /* trails */
      const trail = (pts, color, dash) => {
        if (pts.length < 2) return;
        g.strokeStyle = color; g.lineWidth = 2; g.setLineDash(dash);
        g.beginPath(); pts.forEach((p, i) => { const [X, Y] = P(p[0], p[1]); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); });
        g.stroke(); g.setLineDash([]);
      };
      trail(odoTr, C.orange, [6, 4]);
      trail(trueTr, C.blue, []);
      /* robot body and wheels */
      const L = sL.get(), r = sR.get();
      const body = (x, y, th, color, dashed, phases) => {
        const W2 = (bx, by) => P(x + bx * Math.cos(th) - by * Math.sin(th), y + bx * Math.sin(th) + by * Math.cos(th));
        g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2; g.setLineDash(dashed ? [4, 3] : []);
        const pts = [W2(0.14, 0), W2(-0.08, 0.1), W2(-0.08, -0.1)];
        g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath(); g.stroke();
        g.setLineDash([]);
        if (!phases) return;
        /* wheel = a bar along the heading (drawn larger than life for legibility); the yellow rim marker moves with the wheel angle */
        const fwx = Math.cos(th), fwy = -Math.sin(th), hl = Math.max(6, r * S * 1.5);
        for (const [by, ph] of [[-L / 2, phases.pr], [L / 2, phases.pl]]) {
          const [wx, wy] = W2(0, by);
          g.strokeStyle = C.muted; g.lineWidth = 4; g.beginPath();
          g.moveTo(wx - fwx * hl, wy - fwy * hl); g.lineTo(wx + fwx * hl, wy + fwy * hl); g.stroke();
          dot(g, wx + fwx * hl * Math.sin(ph), wy + fwy * hl * Math.sin(ph), 3, C.yellow);
        }
      };
      body(O.x, O.y, O.th, C.orange, true, null);
      body(T.x, T.y, T.th, C.blue, false, { pr: T.pr, pl: T.pl });
      const { v, w } = cmd();
      const d = Math.hypot(T.x - O.x, T.y - O.y);
      rd.textContent =
        `true pose   (x, y, θ) = (${F(T.x, 3)} m, ${F(T.y, 3)} m, ${F(T.th / D, 1)}°)\n`
        + `odometry  (x, y, θ) = (${F(O.x, 3)} m, ${F(O.y, 3)} m, ${F(O.th / D, 1)}°)   position error = ${F(d, 3)} m\n`
        + `v = r(ωR+ωL)/2 = ${F(v, 3)} m/s     ω = r(ωR−ωL)/L = ${F(w, 3)} rad/s     t = ${F(t, 2)} s`
        + (Math.abs(w) > 1e-6 ? `\nturn radius R = v/ω = ${F(v / w, 3)} m` : '');
      if (Math.abs(w) < 1e-6) cap.textContent = `Straight line at v = ${F(v, 2)} m/s. Blue: true motion. Orange ghost: odometry with the wrong radius scale and step h = ${F(sH.get(), 2)} s.`;
      else if (Math.abs(v) < 1e-6) cap.textContent = 'Spinning in place: v = 0, so the position stays fixed while the heading changes. Odometry error here is pure heading drift.';
      else cap.textContent = `Curve: the robot circles a point on the axle line, radius R = v/ω = ${F(v / w, 2)} m. Coarse straight-line steps cut the corners, so the orange ghost drifts away from the true arc.`;
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running) { advance(Math.min(dt, 0.1) * sSpeed.get()); changed = true; }
      if (changed) draw();
    }, { fps: 60 });
  };

  /* ---------- 6 · integrators on a spring ---------- */
  AI.labs['sim-integrators'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Simulating a spring: explicit versus semi-implicit Euler',
      'illustrative: m = 1 kg, k = 100 N/m, ω = 10 rad/s, x₀ = 1 m, E₀ = 50 J. Energy should stay flat.',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const OMEGA2 = 100, E0 = 50;
    const sH = slider(ctl, 'Time step h', 0.005, 0.25, 0.005, 0.01, 's', 3, () => (dirty = true));
    const sSpeed = slider(ctl, 'Simulation speed', 0.1, 1, 0.05, 0.5, '× real time', 2);
    const [bMethod, bPlay, bReset] = buttons(ctl, ['Method: explicit Euler', '❚❚ Pause', 'Reset']);
    const rd = readout(ctl);
    let method = 'explicit', running = true, x = 1, v = 0, t = 0, nextS = 0, hist = [], phase = [], diverged = false;
    const tick = AI.ticker();
    const reset = () => { x = 1; v = 0; t = 0; nextS = 0; hist = []; phase = []; diverged = false; dirty = true; };
    const step = (h) => {
      if (method === 'explicit') { const xn = x + h * v, vn = v - h * OMEGA2 * x; x = xn; v = vn; }
      else { v = v - h * OMEGA2 * x; x = x + h * v; }
      t += h;
      if (Math.abs(x) > 1e6 || !Number.isFinite(x) || !Number.isFinite(v)) { diverged = true; return; }
      if (t >= nextS) {
        const ratio = (0.5 * v * v + 0.5 * OMEGA2 * x * x) / E0;
        hist.push([t, Math.log10(Math.max(ratio, 1e-6))]);
        phase.push([x, v / 10]);
        nextS += 0.05;
        if (hist.length > 200) hist.shift();
        if (phase.length > 200) phase.shift();
      }
    };
    bMethod.onclick = () => {
      method = method === 'explicit' ? 'semi' : 'explicit';
      bMethod.textContent = method === 'explicit' ? 'Method: explicit Euler' : 'Method: semi-implicit Euler';
      reset();
    };
    bPlay.onclick = () => { running = !running; bPlay.textContent = running ? '❚❚ Pause' : '▶ Play'; dirty = true; };
    bReset.onclick = reset;

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h;
      g.clearRect(0, 0, W, Hh);
      const h = sH.get();
      /* spring */
      const wallX = W * 0.04, restX = W * 0.27, Sm = Math.min(W * 0.2, 70), yS = Hh * 0.2;
      g.strokeStyle = C.muted; g.lineWidth = 3; g.beginPath(); g.moveTo(wallX, yS - 26); g.lineTo(wallX, yS + 26); g.stroke();
      const mx = restX + Math.max(-2, Math.min(2, x)) * Sm;
      g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); g.moveTo(wallX, yS);
      const nZ = 14, segW = (mx - 16 - wallX) / nZ;
      for (let i = 1; i <= nZ; i++) g.lineTo(wallX + i * segW, yS + (i % 2 ? -9 : 9));
      g.lineTo(mx - 16, yS); g.stroke();
      g.fillStyle = C.blue; g.fillRect(mx - 16, yS - 16, 32, 32);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(restX, yS + 30); g.lineTo(restX, yS + 40); g.stroke();
      label(g, 'rest', restX, yS + 54, C.muted, 'center');
      if (Math.abs(x) > 2) label(g, `x = ${F(x, 1)} m (off scale)`, mx, yS - 26, C.red, 'center');
      /* phase plot: (x, v/ω), a circle when energy is conserved */
      const pcx = W * 0.2, pcy = Hh * 0.7, pr = Math.min(W * 0.14, Hh * 0.2);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.arc(pcx, pcy, pr, 0, Math.PI * 2); g.stroke();
      g.beginPath(); g.moveTo(pcx - pr - 8, pcy); g.lineTo(pcx + pr + 8, pcy); g.moveTo(pcx, pcy - pr - 8); g.lineTo(pcx, pcy + pr + 8); g.stroke();
      const sc = pr / 1.2;
      g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath();
      phase.forEach((p, i) => { const X = pcx + AI.clamp(p[0], -3, 3) * sc, Y = pcy - AI.clamp(p[1], -3, 3) * sc; if (i) g.lineTo(X, Y); else g.moveTo(X, Y); });
      g.stroke();
      label(g, 'phase: x (m) vs v/ω (m)', pcx, pcy + pr + 22, C.muted, 'center');
      /* energy plot: log10(E/E0) from -1 to 5 */
      const ex0 = W * 0.55, ex1 = W * 0.96, ey0 = Hh * 0.1, ey1 = Hh * 0.88;
      const Yv = (lv) => ey1 - ((AI.clamp(lv, -1, 5) + 1) / 6) * (ey1 - ey0);
      g.strokeStyle = C.border; g.lineWidth = 1;
      for (const lv of [-1, 0, 1, 2, 3, 4, 5]) { const yy = Yv(lv); g.beginPath(); g.moveTo(ex0, yy); g.lineTo(ex1, yy); g.stroke(); label(g, lv === 0 ? '1' : lv === 1 ? '10' : lv < 0 ? '0.1' : '10^' + lv, ex0 - 4, yy + 4, C.muted, 'right'); }
      g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath();
      hist.forEach((p, i) => { const X = ex0 + ((p[0] - (hist[0][0])) / 10) * (ex1 - ex0), Y = Yv(p[1]); if (i) g.lineTo(X, Y); else g.moveTo(X, Y); });
      g.stroke();
      label(g, 'E / E₀ (log scale)', ex0, ey0 - 2, C.muted);
      label(g, 'time (s, last 10 s)', ex1, ey1 + 16, C.muted, 'right');
      /* readouts */
      const ratio = (0.5 * v * v + 0.5 * OMEGA2 * x * x) / E0;
      const limit = 2 / 10;
      rd.textContent =
        `t = ${F(t, 2)} s   x = ${F(x, 3)} m   v = ${F(v, 3)} m/s   E/E₀ = ${F(ratio, 3)}\n`
        + `time step h = ${F(h, 3)} s,  stability limit for semi-implicit: h < 2/ω = ${F(limit, 2)} s\n`
        + (method === 'explicit'
          ? `explicit Euler multiplies E by (1 + ω²h²) = ${F(1 + OMEGA2 * h * h, 4)} each step, for any h`
          : `semi-implicit: ${h < limit ? 'stable at this step (h < 0.2 s)' : 'UNSTABLE at this step (h ≥ 0.2 s)'}`);
      if (diverged) cap.textContent = 'Diverged: the simulated energy grew without limit. This is a numerical artefact of the method and step, not physics. Switch method or shrink the step.';
      else if (method === 'explicit') cap.textContent = 'Explicit Euler: each step uses the old velocity and old position, so the energy grows a little every step. The green phase orbit spirals outward.';
      else cap.textContent = 'Semi-implicit Euler: the velocity is updated first and the new velocity moves the position. The energy wobbles inside a band and the phase orbit stays closed.';
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running && !diverged) {
        const n = tick(dt, sSpeed.get() / sH.get());
        for (let i = 0; i < n && !diverged; i++) step(sH.get());
        if (n > 0) changed = true;
      }
      if (changed) draw();
    }, { fps: 60 });
  };

  /* ---------- 7 · Coulomb friction: stick or slip ---------- */
  AI.labs['stick-slip'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Coulomb friction: stick or slip',
      'illustrative: a block on a table. Friction is at most μN; beyond that the block slides.',
      '<canvas style="height:180px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    let dirty = true; c.onresize = () => (dirty = true);
    const sF = slider(ctl, 'Applied force F', -10, 10, 0.1, 4, 'N', 1, () => (dirty = true));
    const sMu = slider(ctl, 'Friction coefficient μ', 0, 1, 0.01, 0.5, '', 2, () => (dirty = true));
    const sN = slider(ctl, 'Normal force N', 1, 20, 0.5, 10, 'N', 1, () => (dirty = true));
    const sM = slider(ctl, 'Mass m', 0.5, 5, 0.1, 1, 'kg', 1, () => (dirty = true));
    const sSpeed = slider(ctl, 'Simulation speed', 0.1, 1, 0.05, 0.5, '× real time', 2);
    const [bPlay, bReset] = buttons(ctl, ['❚❚ Pause', 'Reset block']);
    const rd = readout(ctl);
    const H = 0.002, tick = AI.ticker();
    let running = true, x = 0, v = 0, t = 0, ff = 0, status = 'stuck', a = 0;
    bPlay.onclick = () => { running = !running; bPlay.textContent = running ? '❚❚ Pause' : '▶ Play'; dirty = true; };
    bReset.onclick = () => { x = 0; v = 0; t = 0; dirty = true; };

    const stepOnce = (h) => {
      const lim = sMu.get() * sN.get(), Fa = sF.get(), m = sM.get();
      if (v === 0 && Math.abs(Fa) <= lim) { status = 'stuck'; ff = -Fa; a = 0; }
      else {
        const dir = v !== 0 ? Math.sign(v) : Math.sign(Fa);
        status = 'slipping';
        ff = -lim * dir;
        a = (Fa + ff) / m;
        const vOld = v;
        v += a * h;
        if (vOld !== 0 && Math.sign(v) !== Math.sign(vOld)) v = 0;
        x += v * h; t += h;
        return;
      }
      t += h;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h;
      g.clearRect(0, 0, W, Hh);
      const S = 60, cx = W * 0.5, by = Hh * 0.5;
      const span = W - 80, bx = ((((cx + x * S - 40) % span) + span) % span) + 40;
      g.strokeStyle = C.border; g.lineWidth = 4; g.beginPath(); g.moveTo(10, by + 26); g.lineTo(W - 10, by + 26); g.stroke();
      g.fillStyle = C.bg2; g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.roundRect(bx - 30, by - 30, 60, 60, 8); g.fill(); g.stroke();
      const Fa = sF.get();
      if (Math.abs(Fa) > 0.05) AI.arrow(g, bx + (Fa > 0 ? 34 : -34), by, bx + (Fa > 0 ? 34 : -34) + Fa * 6, by, C.orange, 3);
      if (Math.abs(ff) > 0.05) AI.arrow(g, bx + (ff > 0 ? 34 : -34), by + 14, bx + (ff > 0 ? 34 : -34) + ff * 6, by + 14, C.green, 3);
      label(g, `F = ${F(Fa, 1)} N (orange)`, 12, 18, C.orange);
      label(g, `friction = ${F(ff, 2)} N (green)`, 12, 36, C.green);
      const lim = sMu.get() * sN.get();
      rd.textContent =
        `status: ${status}    limit μN = ${F(lim, 2)} N    friction = ${F(ff, 2)} N\n`
        + `net force F + f = ${F(Fa + ff, 2)} N    acceleration a = ${F(a, 2)} m/s²    velocity = ${F(v, 3)} m/s`;
      if (status === 'stuck') cap.textContent = `Stuck: |F| = ${F(Math.abs(Fa), 2)} N ≤ μN = ${F(lim, 2)} N, so static friction equals −F and the net force is zero.`;
      else cap.textContent = `Slipping: |F| > μN, so friction is μN = ${F(lim, 2)} N, and the block accelerates at (F − μN)/m = ${F(a, 2)} m/s². Make F smaller than μN to see it stop.`;
    };

    AI.loop(root, (dt) => {
      let changed = dirty; dirty = false;
      if (running) {
        const n = tick(dt, sSpeed.get() / H);
        for (let i = 0; i < n; i++) stepOnce(H);
        if (n > 0) changed = true;
      }
      if (changed) draw();
    }, { fps: 60 });
  };
})();
