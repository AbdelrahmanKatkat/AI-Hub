/* Robotics · Module 6 (Control) labs. Illustrative 1-D and small-state models, not hardware data.
 * fb-step      <div data-lab="fb-step"></div>      proportional feedback on a first-order plant, with delay
 * pid-terms    <div data-lab="pid-terms"></div>    P, I and D terms, actuator limit, anti-windup
 * lqr-cart     <div data-lab="lqr-cart"></div>     LQR gains for a cart (double integrator)
 * lyap         <div data-lab="lyap"></div>         damped pendulum, energy as a Lyapunov function
 * mpc          <div data-lab="mpc"></div>          receding-horizon control of a 1-D point mass
 * impedance    <div data-lab="impedance"></div>    mass-spring-damper pressing on a stiff wall
 */
(function () {
  const AI = window.AI;
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const col = (k) => AI.col()[k] || '#888888';
  const f = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  /* ---------- controls: sliders and checkboxes; onChange(key) runs after each edit ---------- */
  function sliders(ctl, defs, onChange) {
    const v = {};
    defs.forEach((d) => {
      const l = document.createElement('label');
      if (d.type === 'check') {
        l.style.flexDirection = 'row'; l.style.alignItems = 'center'; l.style.gap = '8px'; l.style.color = 'var(--fg)';
        l.innerHTML = `<input type="checkbox"${d.v ? ' checked' : ''}><span>${d.l}</span>`;
        const inp = l.querySelector('input');
        v[d.k] = !!d.v;
        inp.onchange = () => { v[d.k] = inp.checked; onChange(d.k); };
      } else {
        const dec = d.dec === undefined ? 2 : d.dec;
        l.innerHTML = `<span>${d.l} <span class="v"></span></span><input type="range" min="${d.min}" max="${d.max}" step="${d.step}" value="${d.v}">`;
        const inp = l.querySelector('input'), out = l.querySelector('.v');
        const show = (x) => (out.textContent = (+x).toFixed(dec) + (d.unit ? ' ' + d.unit : ''));
        v[d.k] = +d.v; show(d.v);
        inp.oninput = () => { v[d.k] = +inp.value; show(inp.value); onChange(d.k); };
      }
      ctl.appendChild(l);
    });
    return { v };
  }

  /* ---------- playback: simulated time t advances at `speed` sim-seconds per real second ---------- */
  function player(ctl, T, step, onRestart) {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<button class="btn small">⏸ Pause</button><button class="btn small">Step ▸</button><button class="btn small">Restart</button>';
    ctl.appendChild(row);
    const [bp, bs, br] = row.querySelectorAll('button');
    const sl = document.createElement('label');
    sl.innerHTML = '<span>Speed <span class="v"></span></span><input type="range" min="0.25" max="2" step="0.05" value="1">';
    ctl.appendChild(sl);
    const inp = sl.querySelector('input'), sv = sl.querySelector('.v');
    const showSp = () => (sv.textContent = (+inp.value).toFixed(2) + '× (1× = one simulated second per real second)');
    inp.oninput = showSp; showSp();
    const P = { t: 0, playing: true, changed: true, T };
    const sync = () => (bp.textContent = P.playing ? '⏸ Pause' : '▶ Play');
    const restart = () => { P.t = 0; P.playing = true; P.changed = true; sync(); if (onRestart) onRestart(); };
    bp.onclick = () => { if (!P.playing && P.t >= T) P.t = 0; P.playing = !P.playing; P.changed = true; sync(); };
    bs.onclick = () => { P.playing = false; P.t = Math.min(T, P.t + step); P.changed = true; sync(); };
    br.onclick = restart;
    P.restart = restart;
    /* returns true when the on-screen state changed (so the caller redraws) */
    P.adv = (dt) => {
      if (P.playing) {
        P.t = Math.min(T, P.t + dt * (+inp.value));
        if (P.t >= T) { P.playing = false; sync(); }
        P.changed = true;
      }
      const c = P.changed; P.changed = false; return c;
    };
    sync();
    return P;
  }

  /* ---------- plot: data-space axes, series, horizontal reference lines, cursor ---------- */
  function plot(c, o) {
    const ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    const v = AI.view(c, o.xmin, o.xmax, o.ymin, o.ymax, { l: 54, r: 14, t: 14, b: 30 });
    AI.axes(ctx, v);
    ctx.save();
    ctx.beginPath(); ctx.rect(v.pad.l, v.pad.t, v.pw, v.ph); ctx.clip();
    (o.hlines || []).forEach((h) => {
      const py = v.Y(h.y);
      ctx.strokeStyle = col(h.color); ctx.lineWidth = 1.3; ctx.setLineDash(h.dash || []);
      ctx.beginPath(); ctx.moveTo(v.pad.l, py); ctx.lineTo(v.pad.l + v.pw, py); ctx.stroke();
      if (h.label) { ctx.setLineDash([]); ctx.fillStyle = col(h.color); ctx.font = '11px ' + MONO; ctx.textAlign = 'left'; ctx.fillText(h.label, v.pad.l + 6, py - 5); }
    });
    ctx.setLineDash([]);
    (o.series || []).forEach((s) => {
      if (s.pts.length < 2) return;
      ctx.strokeStyle = col(s.color); ctx.lineWidth = s.w || 2; ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      s.pts.forEach(([x, y], i) => { const px = v.X(x), py = v.Y(y); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
    });
    ctx.setLineDash([]);
    (o.dots || []).forEach((d) => {
      ctx.fillStyle = col(d.color); ctx.beginPath(); ctx.arc(v.X(d.x), v.Y(d.y), d.r || 3, 0, 6.2832); ctx.fill();
    });
    if (o.cursor !== undefined) {
      const px = v.X(o.cursor);
      ctx.strokeStyle = col('fg'); ctx.globalAlpha = 0.45; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, v.pad.t); ctx.lineTo(px, v.pad.t + v.ph); ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.restore();
    ctx.font = '11px ' + MONO;
    ctx.fillStyle = col('muted');
    if (o.xl) { ctx.textAlign = 'right'; ctx.fillText(o.xl, v.pad.l + v.pw, v.pad.t + v.ph + 27); }
    if (o.yl) { ctx.textAlign = 'left'; ctx.fillText(o.yl, v.pad.l + 4, v.pad.t + 11); }
    // legend in the top-right corner
    const named = (o.series || []).filter((s) => s.name);
    ctx.textAlign = 'right';
    named.forEach((s, i) => {
      const y = v.pad.t + 12 + i * 14, x = v.pad.l + v.pw - 6;
      ctx.fillStyle = col(s.color); ctx.fillText(s.name, x, y);
      const tw = ctx.measureText(s.name).width;
      ctx.strokeStyle = col(s.color); ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
      ctx.beginPath(); ctx.moveTo(x - tw - 26, y - 4); ctx.lineTo(x - tw - 6, y - 4); ctx.stroke(); ctx.setLineDash([]);
    });
  }

  /* horizontal bars around zero, one per item: {name, val, color} */
  function bars(c, items, range) {
    const ctx = c.ctx;
    ctx.clearRect(0, 0, c.w, c.h);
    const x0 = 118, x1 = c.w - 14, mid = (x0 + x1) / 2, sc = (x1 - x0) / 2 / range;
    const rowH = (c.h - 10) / items.length;
    ctx.font = '11px ' + MONO;
    items.forEach((it, i) => {
      const y = 6 + i * rowH + rowH * 0.2, h = rowH * 0.6;
      ctx.fillStyle = col('muted'); ctx.textAlign = 'right'; ctx.fillText(it.name, x0 - 8, y + h * 0.75);
      ctx.fillStyle = col(it.color);
      const w = clamp(it.val, -range, range) * sc;
      ctx.fillRect(Math.min(mid, mid + w), y, Math.abs(w), h);
      ctx.fillStyle = col('fg'); ctx.textAlign = it.val >= 0 ? 'left' : 'right';
      ctx.fillText(f(it.val, 2), it.val >= 0 ? mid + w + 5 : mid + w - 5, y + h * 0.75);
    });
    ctx.strokeStyle = col('muted'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(mid + 0.5, 4); ctx.lineTo(mid + 0.5, c.h - 4); ctx.stroke();
  }

  /* ================= Lesson 6.1 · feedback ================= */
  AI.labs['fb-step'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Proportional feedback on a first-order plant',
      'τ·ẏ = −y + u,  u = K·(r − y(t − L)),  b = 1 · illustrative',
      '<canvas style="height:260px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    let dirty = true, traj = null, pl;
    cv.onresize = () => (dirty = true);
    const H = 0.01, TT = 12, R = 1;
    const S = sliders(ctl, [
      { k: 'K', l: 'Gain K', min: 0, max: 8, step: 0.1, v: 3, dec: 1 },
      { k: 'tau', l: 'Plant time constant τ', min: 0.5, max: 5, step: 0.1, v: 2, unit: 's', dec: 1 },
      { k: 'L', l: 'Sensor / comms delay L', min: 0, max: 2, step: 0.1, v: 0, unit: 's', dec: 1 },
    ], () => { compute(); pl.restart(); });
    pl = player(ctl, TT, 0.5, () => (dirty = true));
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    function compute() {
      const { K, tau, L } = S.v, n = Math.round(TT / H), d = Math.round(L / H);
      const y = new Float64Array(n + 1);
      for (let k = 0; k < n; k++) {
        const yd = k - d >= 0 ? y[k - d] : 0;          // the controller only sees the delayed output
        y[k + 1] = y[k] + H * (-y[k] + K * (R - yd)) / tau;
      }
      traj = y;
    }
    function draw() {
      const { K, tau, L } = S.v, n = Math.round(TT / H), k = Math.min(n, Math.round(pl.t / H));
      const pts = [];
      for (let i = 0; i <= k; i += 4) pts.push([i * H, traj[i]]);
      const yInf = K / (1 + K), tcl = tau / (1 + K), yNow = traj[k];
      plot(cv, {
        xmin: 0, xmax: TT, ymin: -0.5, ymax: 2, xl: 't (s)', yl: 'y',
        hlines: [{ y: R, color: 'orange', dash: [6, 4], label: 'setpoint r = 1' },
                 { y: yInf, color: 'muted', dash: [2, 4], label: 'predicted steady state' }],
        series: [{ pts, color: 'blue', w: 2.5, name: 'y(t)' }], cursor: pl.t,
      });
      let mn = Infinity, mx = -Infinity;
      for (let i = Math.max(0, n - 200); i <= n; i++) { mn = Math.min(mn, traj[i]); mx = Math.max(mx, traj[i]); }
      const amp = mx - mn;
      const state = Math.abs(yNow) > 20 ? 'diverging: the loop is unstable at these settings'
        : amp > 0.02 ? `still ringing (peak-to-peak ${f(amp)} over the last 2 s)` : 'settled';
      cap.textContent = `t = ${f(pl.t, 1)} s: y = ${f(yNow)} (error ${f(R - yNow)}). ` +
        (L > 0 ? 'The controller reacts to an old reading, so it keeps correcting past the target.' : 'With no delay, y creeps to its steady state.');
      out.textContent = [
        `K = ${f(K, 1)},  τ = ${f(tau, 1)} s,  delay L = ${f(L, 1)} s,  r = 1`,
        `steady state  y∞ = K/(1+K) = ${f(yInf)}`,
        `steady error  e∞ = 1/(1+K) = ${f(1 - yInf)}`,
        `closed-loop time constant  τ/(1+K) = ${f(tcl)} s`,
        `last 2 s: ${state}`,
      ].join('\n');
    }
    compute();
    AI.loop(root, (dt) => { if (pl.adv(dt) || dirty) { dirty = false; draw(); } }, { fps: 60 });
  };

  /* ================= Lesson 6.2 · PID ================= */
  AI.labs['pid-terms'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'PID with an actuator limit',
      'plant τ·ẏ = −y + u, τ = 2 s;  setpoint r = 1 from t = 0 · illustrative',
      '<canvas style="height:200px"></canvas><canvas style="height:150px"></canvas><canvas style="height:120px"></canvas><div class="caption"></div>', '');
    const cs = AI.$$('canvas', view).map(AI.canvas), cap = AI.$('.caption', view);
    let dirty = true, D = null, pl;
    cs.forEach((c) => (c.onresize = () => (dirty = true)));
    const TT = 15, DT = 0.01, R = 1, TAU = 2, KAW = 2;
    const S = sliders(ctl, [
      { k: 'Kp', l: 'Proportional Kp', min: 0, max: 5, step: 0.1, v: 2, dec: 1 },
      { k: 'Ki', l: 'Integral Ki', min: 0, max: 2, step: 0.05, v: 0.6, unit: '1/s', dec: 2 },
      { k: 'Kd', l: 'Derivative Kd', min: 0, max: 2, step: 0.05, v: 0.2, unit: 's', dec: 2 },
      { k: 'umax', l: 'Actuator limit |u| ≤', min: 0.5, max: 5, step: 0.1, v: 1.5, unit: 'u-units', dec: 1 },
      { k: 'aw', l: 'anti-windup (back-calculation)', type: 'check', v: false },
    ], () => { compute(); pl.restart(); });
    pl = player(ctl, TT, 0.5, () => (dirty = true));
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    function compute() {
      const { Kp, Ki, Kd, umax, aw } = S.v, n = Math.round(TT / DT);
      D = { y: new Float64Array(n + 1), u: new Float64Array(n), raw: new Float64Array(n), P: new Float64Array(n), I: new Float64Array(n), Dd: new Float64Array(n), n };
      let y = 0, z = 0, ePrev = R, ymax = R;
      for (let k = 0; k < n; k++) {
        const e = R - y, de = (e - ePrev) / DT; ePrev = e;
        const p = Kp * e, it = Ki * z, dd = Kd * de, ur = p + it + dd, us = clamp(ur, -umax, umax);
        D.y[k] = y; D.u[k] = us; D.raw[k] = ur; D.P[k] = p; D.I[k] = it; D.Dd[k] = dd;
        z += DT * (e + (aw ? KAW * (us - ur) : 0));  // back-calculation drains the integrator while saturated
        y += DT * (-y + us) / TAU;
        ymax = Math.max(ymax, y);
      }
      D.y[n] = y; D.ymax = ymax;
    }
    function draw() {
      const k = Math.min(D.n - 1, Math.round(pl.t / DT));
      const ys = [], us = [], ur = [];
      for (let i = 0; i <= k; i += 4) { ys.push([i * DT, D.y[i]]); us.push([i * DT, D.u[i]]); ur.push([i * DT, D.raw[i]]); }
      const umax = S.v.umax, sat = Math.abs(D.raw[k]) > umax;
      plot(cs[0], {
        xmin: 0, xmax: TT, ymin: -0.2, ymax: Math.max(1.5, D.ymax * 1.1), xl: 't (s)', yl: 'y',
        hlines: [{ y: R, color: 'orange', dash: [6, 4], label: 'setpoint r = 1' }],
        series: [{ pts: ys, color: 'blue', w: 2.5, name: 'y(t)' }], cursor: pl.t,
      });
      plot(cs[1], {
        xmin: 0, xmax: TT, ymin: -umax * 1.4, ymax: umax * 1.4, xl: 't (s)', yl: 'u',
        hlines: [{ y: umax, color: 'red', dash: [4, 4], label: '+limit' }, { y: -umax, color: 'red', dash: [4, 4] }],
        series: [{ pts: ur, color: 'muted', w: 1.5, dash: [3, 3], name: 'u before limit' }, { pts: us, color: 'purple', w: 2.2, name: 'u applied' }],
        cursor: pl.t,
      });
      bars(cs[2], [
        { name: 'P = Kp·e', val: D.P[k], color: 'blue' },
        { name: 'I = Ki·∫e', val: D.I[k], color: 'green' },
        { name: 'D = Kd·ė', val: D.Dd[k], color: 'orange' },
        { name: 'sum (raw u)', val: D.raw[k], color: 'purple' },
      ], 3);
      const overshoot = (D.ymax - R) / R * 100;
      cap.textContent = `t = ${f(pl.t, 1)} s: P reacts to the error now, I to the error summed so far, D to how fast the error changes. ` +
        (sat ? 'The actuator is saturated: the applied u is clipped.' : 'The actuator is inside its limit.');
      out.textContent = [
        `Kp = ${f(S.v.Kp, 1)},  Ki = ${f(S.v.Ki, 2)} /s,  Kd = ${f(S.v.Kd, 2)} s,  |u| ≤ ${f(umax, 1)}`,
        `y(t) = ${f(D.y[k])},  e = ${f(R - D.y[k])}`,
        `peak overshoot so far ≈ ${f(Math.max(0, overshoot), 1)} %`,
        `anti-windup: ${S.v.aw ? 'on' : 'off'}`,
      ].join('\n');
    }
    compute();
    AI.loop(root, (dt) => { if (pl.adv(dt) || dirty) { dirty = false; draw(); } }, { fps: 60 });
  };

  /* ================= Lesson 6.3 · state space and LQR ================= */
  AI.labs['lqr-cart'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'LQR state feedback on a cart',
      'ẍ = u (unit mass)  ·  u = −K·[x, v]ᵀ  ·  K from the Riccati equation · illustrative',
      '<canvas style="height:110px"></canvas><canvas style="height:220px"></canvas><div class="caption"></div>', '');
    const cs = AI.$$('canvas', view).map(AI.canvas), cap = AI.$('.caption', view);
    let dirty = true, D = null, pl;
    cs.forEach((c) => (c.onresize = () => (dirty = true)));
    const TT = 8, DT = 0.005;
    const S = sliders(ctl, [
      { k: 'q1', l: 'Position weight q₁', min: 0.1, max: 10, step: 0.1, v: 1, dec: 1 },
      { k: 'q2', l: 'Velocity weight q₂', min: 0.1, max: 10, step: 0.1, v: 1, dec: 1 },
      { k: 'r', l: 'Effort weight r', min: 0.1, max: 10, step: 0.1, v: 1, dec: 1 },
      { k: 'x0', l: 'Start position x(0)', min: -2, max: 2, step: 0.1, v: 1, unit: 'm', dec: 1 },
    ], () => { compute(); pl.restart(); });
    pl = player(ctl, TT, 0.5, () => (dirty = true));
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    /* closed form for Q = diag(q1, q2), R = r on the double integrator (solves the 2x2 Riccati equation) */
    function gains() {
      const { q1, q2, r } = S.v;
      return { k1: Math.sqrt(q1 / r), k2: Math.sqrt((q2 + 2 * Math.sqrt(q1 * r)) / r) };
    }
    function compute() {
      const { k1, k2 } = gains(), n = Math.round(TT / DT);
      const X = new Float64Array(n + 1), V = new Float64Array(n + 1), U = new Float64Array(n + 1);
      let x = S.v.x0, v = 0;
      for (let k = 0; k <= n; k++) {
        const u = -(k1 * x + k2 * v);
        X[k] = x; V[k] = v; U[k] = u;
        if (k < n) { v += DT * u; x += DT * v; }
      }
      let last = 0;
      for (let k = 0; k <= n; k++) if (Math.abs(X[k]) >= 0.05) last = k;
      D = { X, V, U, n, k1, k2, settle: (last + 1) * DT };
    }
    function draw() {
      const { k1, k2, n } = D, k = Math.min(n, Math.round(pl.t / DT));
      /* rail: cart drawn from the simulated position, with a trail and the force arrow */
      const c = cs[0], ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const xpx = (x) => 40 + ((x + 3) / 6) * (c.w - 80), yc = c.h * 0.55;
      ctx.strokeStyle = col('border'); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(xpx(-3), yc + 22); ctx.lineTo(xpx(3), yc + 22); ctx.stroke();
      ctx.strokeStyle = col('orange'); ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(xpx(0), 8); ctx.lineTo(xpx(0), c.h - 8); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = '11px ' + MONO; ctx.fillStyle = col('orange'); ctx.textAlign = 'left'; ctx.fillText('target x = 0', xpx(0) + 5, 16);
      const tr0 = Math.max(0, k - 80);
      for (let i = tr0; i <= k; i += 4) { ctx.globalAlpha = 0.08 + 0.5 * (i - tr0) / Math.max(1, k - tr0); ctx.fillStyle = col('blue'); ctx.fillRect(xpx(D.X[i]) - 1.5, yc - 18, 3, 3); }
      ctx.globalAlpha = 1;
      const cx = xpx(D.X[k]), w = 0.5 * (c.w - 80) / 6;
      ctx.fillStyle = col('blue'); ctx.globalAlpha = 0.85; ctx.fillRect(cx - w / 2, yc - 16, w, 36); ctx.globalAlpha = 1;
      const ua = D.U[k], al = clamp(ua * 14, -90, 90);
      if (Math.abs(al) > 2) AI.arrow(ctx, cx, yc + 2, cx + al, yc + 2, col('purple'), 2.5);
      ctx.fillStyle = col('purple'); ctx.textAlign = 'center';
      ctx.fillText(`u = ${f(ua, 2)} m/s²`, cx + al / 2, yc - 26);
      /* traces */
      const ys = [], vs = [], us = [];
      for (let i = 0; i <= k; i += 4) { ys.push([i * DT, D.X[i]]); vs.push([i * DT, D.V[i]]); us.push([i * DT, D.U[i]]); }
      let m = 0.5; for (let i = 0; i <= n; i += 10) m = Math.max(m, Math.abs(D.X[i]), Math.abs(D.V[i]), 0.5 * Math.abs(D.U[i]));
      plot(cs[1], {
        xmin: 0, xmax: TT, ymin: -m * 1.1, ymax: m * 1.1, xl: 't (s)', yl: 'x, v, u/2',
        hlines: [{ y: 0, color: 'muted', dash: [2, 4] }],
        series: [{ pts: ys, color: 'blue', w: 2.5, name: 'x (m)' }, { pts: vs, color: 'green', w: 2, name: 'v (m/s)' }, { pts: us, color: 'purple', w: 1.5, dash: [4, 3], name: 'u/2 (m/s²)' }],
        cursor: pl.t,
      });
      const zeta = k2 / (2 * Math.sqrt(k1)), wn = Math.sqrt(k1);
      cap.textContent = `t = ${f(pl.t, 1)} s. The gain pulls x toward 0 like a spring (k₁) and damps v like a brake (k₂). Both are chosen by the weights, not by hand.`;
      out.textContent = [
        `K = [k₁, k₂] = [${f(k1)} s⁻², ${f(k2)} s⁻¹]  (unit mass)`,
        `closed loop: s² + ${f(k2)} s + ${f(k1)} = 0`,
        `ωn = √k₁ = ${f(wn)} rad/s,   ζ = k₂/(2√k₁) = ${f(zeta)}`,
        `settles within 0.05 m at t ≈ ${f(D.settle, 2)} s`,
        `now: x = ${f(D.X[k])} m,  v = ${f(D.V[k])} m/s,  u = ${f(D.U[k])} m/s²`,
      ].join('\n');
    }
    compute();
    AI.loop(root, (dt) => { if (pl.adv(dt) || dirty) { dirty = false; draw(); } }, { fps: 60 });
  };

  /* ================= Lesson 6.4 · Lyapunov stability ================= */
  AI.labs.lyap = function (root) {
    const { view, ctl } = AI.labShell(root, 'Energy as a Lyapunov function (damped pendulum)',
      'θ̇ = ω,  ω̇ = −sin θ − c·ω   (unit length, g/L = 1, angles in rad)  ·  V = 1 − cos θ + ω²/2',
      '<canvas style="height:170px"></canvas><canvas style="height:200px"></canvas><canvas style="height:120px"></canvas><div class="caption"></div>', '');
    const cs = AI.$$('canvas', view).map(AI.canvas), cap = AI.$('.caption', view);
    let dirty = true, D = null, pl;
    cs.forEach((c) => (c.onresize = () => (dirty = true)));
    const TT = 20, H = 0.01, N = Math.round(TT / H);
    const S = sliders(ctl, [
      { k: 'c', l: 'Damping c (negative = energy pumped in)', min: -0.2, max: 1, step: 0.02, v: 0.3, dec: 2 },
      { k: 'th0', l: 'Start angle θ(0)', min: 0.1, max: 3, step: 0.05, v: 1, unit: 'rad', dec: 2 },
      { k: 'w0', l: 'Start speed ω(0)', min: -2, max: 2, step: 0.05, v: 0, unit: 'rad/s', dec: 2 },
    ], () => { compute(); pl.restart(); });
    pl = player(ctl, TT, 1, () => (dirty = true));
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    const rhs = (th, w, c) => [w, -Math.sin(th) - c * w];
    function compute() {
      const { c } = S.v;
      const TH = new Float64Array(N + 1), W = new Float64Array(N + 1), VV = new Float64Array(N + 1);
      let th = S.v.th0, w = S.v.w0;
      for (let k = 0; k <= N; k++) {
        TH[k] = th; W[k] = w; VV[k] = 1 - Math.cos(th) + w * w / 2;
        const k1 = rhs(th, w, c);
        const k2 = rhs(th + H / 2 * k1[0], w + H / 2 * k1[1], c);
        const k3 = rhs(th + H / 2 * k2[0], w + H / 2 * k2[1], c);
        const k4 = rhs(th + H * k3[0], w + H * k3[1], c);
        th += H / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        w += H / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      }
      let vmax = 0; for (let k = 0; k <= N; k++) vmax = Math.max(vmax, VV[k]);
      D = { TH, W, VV, vmax };
    }
    function draw() {
      const { TH, W, VV, vmax } = D, k = Math.min(N, Math.round(pl.t / H)), step = 4;
      /* pendulum */
      const c0 = cs[0], x0c = c0.ctx; x0c.clearRect(0, 0, c0.w, c0.h);
      const px = c0.w / 2, py = 22, L = Math.min(c0.h - 50, c0.w * 0.35);
      const bob = (i) => [px + L * Math.sin(TH[i]), py + L * Math.cos(TH[i])];
      x0c.strokeStyle = col('muted'); x0c.lineWidth = 1; x0c.setLineDash([3, 4]);
      x0c.beginPath(); x0c.moveTo(px, py); x0c.lineTo(px, py + L + 10); x0c.stroke(); x0c.setLineDash([]);
      x0c.fillStyle = col('muted'); x0c.beginPath(); x0c.arc(px, py, 4, 0, 6.2832); x0c.fill();
      const t0 = Math.max(0, k - 60);
      for (let i = t0; i <= k; i += 3) { const [tx, ty] = bob(i); x0c.globalAlpha = 0.05 + 0.4 * (i - t0) / Math.max(1, k - t0); x0c.fillStyle = col('blue'); x0c.beginPath(); x0c.arc(tx, ty, 2.5, 0, 6.2832); x0c.fill(); }
      x0c.globalAlpha = 1;
      const [bx, by] = bob(k);
      x0c.strokeStyle = col('fg'); x0c.lineWidth = 2; x0c.beginPath(); x0c.moveTo(px, py); x0c.lineTo(bx, by); x0c.stroke();
      x0c.fillStyle = col(S.v.c < 0 ? 'red' : 'blue'); x0c.beginPath(); x0c.arc(bx, by, 11, 0, 6.2832); x0c.fill();
      x0c.font = '11px ' + MONO; x0c.fillStyle = col('muted'); x0c.textAlign = 'left';
      x0c.fillText('θ = ' + f(TH[k], 2) + ' rad', 10, c0.h - 10);
      /* phase portrait with level curves of V */
      const ylim = 3.2, xlim = Math.PI + 0.3;
      const lc = [];
      [0.25, 0.6, 1.0, 1.5, 2.0].forEach((Lv) => {
        const a = [], b = [];
        for (let th = -xlim; th <= xlim; th += 0.02) {
          const rest = Lv - (1 - Math.cos(th));
          if (rest >= 0) { const w = Math.sqrt(2 * rest); a.push([th, w]); b.push([th, -w]); }
        }
        lc.push(a, b);
      });
      const pts = [];
      for (let i = 0; i <= k; i += step) pts.push([TH[i], W[i]]);
      plot(cs[1], {
        xmin: -xlim, xmax: xlim, ymin: -ylim, ymax: ylim, xl: 'θ (rad)', yl: 'ω (rad/s)',
        hlines: [{ y: 0, color: 'muted', dash: [2, 4] }],
        series: [...lc.map((p) => ({ pts: p, color: 'border', w: 1 })),
                 { pts, color: 'blue', w: 2.2, name: 'trajectory' }],
        dots: [{ x: TH[k], y: W[k], color: 'orange', r: 4 }],
      });
      /* V(t) */
      const pv = []; for (let i = 0; i <= k; i += step) pv.push([i * H, VV[i]]);
      plot(cs[2], {
        xmin: 0, xmax: TT, ymin: 0, ymax: vmax * 1.1 + 0.05, xl: 't (s)', yl: 'V (energy)',
        series: [{ pts: pv, color: 'green', w: 2.2, name: 'V(t)' }], cursor: pl.t,
      });
      const Vdot = -S.v.c * W[k] * W[k];
      const meaning = S.v.c > 0 ? 'V decreases: energy leaves the pendulum (V̇ = −c·ω² ≤ 0).'
        : S.v.c < 0 ? 'V increases: negative damping pumps energy in, so the bottom is unstable.'
        : 'V is constant: no damping, the pendulum swings forever.';
      cap.textContent = `t = ${f(pl.t, 1)} s. ${meaning}`;
      out.textContent = [
        `c = ${f(S.v.c, 2)},  V(0) = ${f(VV[0])}`,
        `now:  θ = ${f(TH[k])} rad,  ω = ${f(W[k])} rad/s`,
        `V(t) = 1 − cos θ + ω²/2 = ${f(VV[k])}`,
        `V̇(t) = −c·ω² = ${f(Vdot)} per s`,
      ].join('\n');
    }
    compute();
    AI.loop(root, (dt) => { if (pl.adv(dt) || dirty) { dirty = false; draw(); } }, { fps: 60 });
  };

  /* ================= Lesson 6.5 · MPC ================= */
  AI.labs.mpc = function (root) {
    const { view, ctl } = AI.labShell(root, 'Receding-horizon control of a point mass',
      'p(k+1) = p + Ts·v + Ts²/2·u,   v(k+1) = v + Ts·u,   Ts = 0.5 s · illustrative, no noise',
      '<canvas style="height:230px"></canvas><canvas style="height:90px"></canvas><div class="caption"></div>', '');
    const cs = AI.$$('canvas', view).map(AI.canvas), cap = AI.$('.caption', view);
    let dirty = true, pl, st, disp = 0, lastCost = 0;
    cs.forEach((c) => (c.onresize = () => (dirty = true)));
    const TS = 0.5, NSTEPS = 40, TT = NSTEPS * TS, h = 0.5;
    const S = sliders(ctl, [
      { k: 'N', l: 'Horizon N (steps)', min: 1, max: 12, step: 1, v: 6, dec: 0 },
      { k: 'umax', l: 'Input limit |u| ≤', min: 0.2, max: 3, step: 0.1, v: 1, unit: 'm/s²', dec: 1 },
      { k: 'rho', l: 'Effort weight ρ', min: 0, max: 0.5, step: 0.01, v: 0.05, dec: 2 },
      { k: 'r', l: 'Target position r', min: -1, max: 2, step: 0.1, v: 1, unit: 'm', dec: 1 },
    ], () => pl.restart());
    pl = player(ctl, TT, TS, () => reset());
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    function reset() { st = { p: 0, v: 0, k: 0, plan: null, hist: [], warm: null, u: 0 }; disp = 0; dirty = true; }
    /* roll the model forward for a whole input sequence U from (p0, v0): predicted positions and cost */
    function predict(p0, v0, U) {
      const { r, rho } = S.v;
      let p = p0, v = v0, J = 0;
      const P = new Array(U.length);
      for (let k = 0; k < U.length; k++) {
        p = p + TS * v + 0.5 * TS * TS * U[k];
        v = v + TS * U[k];
        P[k] = p;
        J += (p - r) * (p - r) + rho * U[k] * U[k];
      }
      return { P, J };
    }
    /* projected gradient on the box |u| <= umax; the cost is quadratic, so the Hessian is found exactly by finite differences */
    function solve() {
      const N = Math.round(S.v.N), umax = S.v.umax, W = st.warm;
      const cost = (U) => predict(st.p, st.v, U).J;
      const U = Array.from({ length: N }, (_, k) => clamp(W && k + 1 < W.length ? W[k + 1] : 0, -umax, umax));
      const z = new Array(N).fill(0), f0 = cost(z);
      const fi = Array.from({ length: N }, (_, i) => { const a = z.slice(); a[i] += h; return cost(a); });
      let L = 0;
      for (let i = 0; i < N; i++) {
        let row = 0;
        for (let j = 0; j < N; j++) {
          const a = z.slice(); a[i] += h; a[j] += h;
          row += Math.abs((cost(a) - fi[i] - fi[j] + f0) / (h * h));
        }
        L = Math.max(L, row);
      }
      const alpha = L > 0 ? 1 / L : 1;
      for (let it = 0; it < 300; it++) {
        let moved = 0;
        for (let i = 0; i < N; i++) {
          const up = U.slice(), dn = U.slice(); up[i] += h; dn[i] -= h;
          const g = (cost(up) - cost(dn)) / (2 * h);
          const nu = clamp(U[i] - alpha * g, -umax, umax);
          moved = Math.max(moved, Math.abs(nu - U[i])); U[i] = nu;
        }
        if (moved < 1e-7) break;
      }
      return U;
    }
    function mpcStep() {
      const U = solve();
      const pr = predict(st.p, st.v, U);
      st.plan = { t0: st.k * TS, P: pr.P };
      lastCost = pr.J;
      st.warm = U;
      const u = U[0];
      st.hist.push([st.k * TS, st.p]);
      st.p = st.p + TS * st.v + 0.5 * TS * TS * u;
      st.v = st.v + TS * u;
      st.u = u;
      st.k++;
    }
    function draw() {
      const N = Math.round(S.v.N), r = S.v.r, umax = S.v.umax, t = pl.t;
      const c0 = cs[0];
      const xmin = t - 6, xmax = t + N * TS + 0.5;
      const plan = st.plan;
      const data = st.hist.concat([[st.k * TS, st.p]]);
      let ymin = Math.min(-0.2, r - 0.2), ymax = Math.max(r + 0.3, 0.5);
      data.forEach(([, p]) => { ymin = Math.min(ymin, p - 0.1); ymax = Math.max(ymax, p + 0.1); });
      if (plan) plan.P.forEach((p) => { ymin = Math.min(ymin, p - 0.1); ymax = Math.max(ymax, p + 0.1); });
      const past = data.filter(([tt]) => tt >= xmin - 0.01);
      const predPts = plan ? [[st.k * TS, st.p]].concat(plan.P.map((p, j) => [plan.t0 + (j + 1) * TS, p])) : [];
      plot(c0, {
        xmin, xmax, ymin, ymax, xl: 't (s)', yl: 'p (m)',
        hlines: [{ y: r, color: 'orange', dash: [6, 4], label: 'target r' }],
        series: [
          { pts: past, color: 'blue', w: 2.5, name: 'actual p(t)' },
          { pts: predPts, color: 'purple', w: 2, dash: [5, 4], name: 'planned p over horizon' },
        ],
        dots: predPts.slice(1).map(([x, y]) => ({ x, y, color: 'purple', r: 2.8 })),
        cursor: t,
      });
      /* rail with eased cart position, target and the applied input as an arrow */
      const c1 = cs[1], ctx = c1.ctx; ctx.clearRect(0, 0, c1.w, c1.h);
      const xpx = (p) => 40 + ((p + 0.5) / 3.5) * (c1.w - 60), yc = c1.h * 0.55;
      ctx.strokeStyle = col('border'); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(xpx(-0.5), yc + 18); ctx.lineTo(xpx(3), yc + 18); ctx.stroke();
      ctx.strokeStyle = col('orange'); ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(xpx(r), 6); ctx.lineTo(xpx(r), c1.h - 6); ctx.stroke(); ctx.setLineDash([]);
      const cx = xpx(disp), bw = 26;
      ctx.fillStyle = col('blue'); ctx.fillRect(cx - bw / 2, yc - 14, bw, 30);
      const al = clamp(st.u * 40, -70, 70);
      if (Math.abs(al) > 2) AI.arrow(ctx, cx, yc - 22, cx + al, yc - 22, col('purple'), 2.5);
      ctx.font = '11px ' + MONO; ctx.fillStyle = col('purple'); ctx.textAlign = 'center';
      ctx.fillText(`u₀ = ${f(st.u, 2)} m/s²`, cx + al / 2, yc - 30);
      ctx.fillStyle = col('muted'); ctx.textAlign = 'left';
      ctx.fillText(`p = ${f(disp)} m`, 8, c1.h - 8);
      const sec = (st.k * TS).toFixed(1);
      cap.textContent = `① measure (p, v) at t = ${sec} s   ② solve for ${N} inputs that bring p near r without breaking |u| ≤ ${f(umax, 1)}   ③ apply only u₀   ④ slide the horizon and repeat.`;
      out.textContent = [
        `t = ${f(t, 1)} s,  p = ${f(st.p)} m,  v = ${f(st.v)} m/s`,
        `applied u₀ = ${f(st.u, 3)} m/s²   (limit ±${f(umax, 1)})`,
        `horizon N·Ts = ${f(N * TS, 1)} s,  plan cost J = ${f(lastCost, 3)}`,
        `planned p over horizon: ${plan ? plan.P.slice(0, 4).map((p) => f(p, 2)).join(', ') + (N > 4 ? ', …' : '') : '–'} m`,
      ].join('\n');
    }
    reset();
    AI.loop(root, (dt) => {
      while (st.k < NSTEPS && st.k * TS <= pl.t + 1e-9) mpcStep();
      const before = disp;
      disp += (st.p - disp) * Math.min(1, dt * 6);
      const moved = Math.abs(disp - before) > 1e-5;
      if (pl.adv(dt) || moved || dirty) { dirty = false; draw(); }
    }, { fps: 60 });
  };

  /* ================= Lesson 6.6 · force and impedance ================= */
  AI.labs.impedance = function (root) {
    const { view, ctl } = AI.labShell(root, 'Impedance control against a stiff wall',
      'M·ẍ + B·ẋ + K·(x − x_d) = −F_wall,   F_wall = k_e·max(0, x − x_w),  x_w = 0 · illustrative',
      '<canvas style="height:100px"></canvas><canvas style="height:150px"></canvas><canvas style="height:110px"></canvas><div class="caption"></div>', '');
    const cs = AI.$$('canvas', view).map(AI.canvas), cap = AI.$('.caption', view);
    let dirty = true, D = null, pl;
    cs.forEach((c) => (c.onresize = () => (dirty = true)));
    const TT = 3, H = 0.0005, EVERY = 20, X0 = -0.2;
    const S = sliders(ctl, [
      { k: 'M', l: 'Effective mass M', min: 0.2, max: 5, step: 0.1, v: 1, unit: 'kg', dec: 1 },
      { k: 'B', l: 'Damping B', min: 0, max: 60, step: 1, v: 20, unit: 'N·s/m', dec: 0 },
      { k: 'K', l: 'Stiffness K', min: 10, max: 400, step: 5, v: 100, unit: 'N/m', dec: 0 },
      { k: 'xd', l: 'Desired position x_d (beyond the wall)', min: 0.02, max: 0.3, step: 0.01, v: 0.1, unit: 'm', dec: 2 },
      { k: 'ke', l: 'Wall stiffness k_e', min: 500, max: 5000, step: 50, v: 2000, unit: 'N/m', dec: 0 },
      { k: 'wall', l: 'wall present', type: 'check', v: true },
    ], () => { compute(); pl.restart(); });
    pl = player(ctl, TT, 0.25, () => (dirty = true));
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);

    /* semi-implicit Euler at 0.5 ms: stable for the stiffest wall in the sliders */
    function run(M, B, K, xd, ke, wall) {
      const n = Math.round(TT / H), xs = [], Fs = [];
      let x = X0, v = 0, peak = 0;
      for (let k = 0; k <= n; k++) {
        const Fw = wall && x > 0 ? ke * x : 0;
        if (Fw > peak) peak = Fw;
        if (k % EVERY === 0) { xs.push(x); Fs.push(Fw); }
        const a = (-B * v - K * (x - xd) - Fw) / M;
        v += a * H; x += v * H;
      }
      return { xs, Fs, peak };
    }
    function compute() {
      const { M, B, K, xd, ke, wall } = S.v;
      D = run(M, B, K, xd, ke, wall);
    }
    function draw() {
      const { M, B, K, xd, ke, wall } = S.v;
      const k = Math.min(D.xs.length - 1, Math.round(pl.t / (H * EVERY)));
      const xNow = D.xs[k], Fnow = D.Fs[k];
      /* scene: the robot block on a line, the wall at x = 0, the target as a dashed line */
      const c0 = cs[0], ctx = c0.ctx; ctx.clearRect(0, 0, c0.w, c0.h);
      const xmin = -0.25, xmax = 0.2, xpx = (x) => 20 + (x - xmin) / (xmax - xmin) * (c0.w - 40), yc = c0.h * 0.5;
      ctx.fillStyle = col('red'); ctx.globalAlpha = 0.14; ctx.fillRect(xpx(0), 8, c0.w - xpx(0) - 20, c0.h - 16); ctx.globalAlpha = 1;
      ctx.strokeStyle = col('red'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xpx(0), 8); ctx.lineTo(xpx(0), c0.h - 8); ctx.stroke();
      ctx.strokeStyle = col('orange'); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(xpx(xd), 8); ctx.lineTo(xpx(xd), c0.h - 8); ctx.stroke(); ctx.setLineDash([]);
      const bw = 0.05 * (c0.w - 40) / (xmax - xmin);
      ctx.fillStyle = col('blue'); ctx.fillRect(xpx(xNow) - bw / 2, yc - 20, bw, 40);
      if (Fnow > 0.5) AI.arrow(ctx, xpx(xNow) - bw / 2, yc, xpx(xNow) - bw / 2 - clamp(Fnow * 0.5, 0, 90), yc, col('purple'), 2.5);
      ctx.font = '11px ' + MONO; ctx.fillStyle = col('muted'); ctx.textAlign = 'left';
      ctx.fillText('robot', xpx(-0.24), c0.h - 8);
      ctx.fillStyle = col('orange'); ctx.fillText('target x_d', xpx(xd) + 4, 20);
      ctx.fillStyle = col('red'); ctx.fillText(wall ? 'wall' : 'no wall', xpx(0) + 4, c0.h - 8);
      /* position and force over time */
      const pts = [], fp = [];
      for (let i = 0; i <= k; i++) { pts.push([i * H * EVERY, D.xs[i]]); fp.push([i * H * EVERY, D.Fs[i]]); }
      plot(cs[1], {
        xmin: 0, xmax: TT, ymin: -0.25, ymax: Math.max(xd, ...D.xs) + 0.03, xl: 't (s)', yl: 'x (m)',
        hlines: [{ y: xd, color: 'orange', dash: [5, 4], label: 'x_d' }, { y: 0, color: 'red', dash: [2, 3], label: 'wall x_w = 0' }],
        series: [{ pts, color: 'blue', w: 2.4, name: 'x(t)' }], cursor: pl.t,
      });
      plot(cs[2], {
        xmin: 0, xmax: TT, ymin: 0, ymax: Math.max(1, D.peak * 1.15), xl: 't (s)', yl: 'F (N)',
        series: [{ pts: fp, color: 'purple', w: 2.4, name: 'contact force F(t)' }], cursor: pl.t,
      });
      const wn = Math.sqrt(K / M), zeta = B / (2 * Math.sqrt(K * M));
      const Feq = wall ? ke * K * xd / (K + ke) : 0;
      const Fstiff = wall ? ke * 10000 * xd / (10000 + ke) : 0;
      cap.textContent = Fnow > 0 ? `t = ${f(pl.t, 2)} s: touching the wall, pushing with ${f(Fnow, 1)} N (F = k_e·x).`
        : `t = ${f(pl.t, 2)} s: no contact yet, x = ${f(xNow, 3)} m. Illustrative: one axis, no friction.`;
      out.textContent = [
        `M = ${f(M, 1)} kg,  B = ${f(B, 0)} N·s/m,  K = ${f(K, 0)} N/m`,
        `ωn = √(K/M) = ${f(wn, 2)} rad/s,   ζ = B/(2√(KM)) = ${f(zeta, 2)}  (1 = critical)`,
        `contact equilibrium F = k_e·K·x_d/(K+k_e) = ${f(Feq, 2)} N`,
        `same target with stiff K = 10 000 N/m: F = ${f(Fstiff, 1)} N`,
        `peak contact force in this run = ${f(D.peak, 1)} N`,
        `now: t = ${f(pl.t, 2)} s,  x = ${f(xNow, 3)} m,  F = ${f(Fnow, 1)} N`,
      ].join('\n');
    }
    compute();
    AI.loop(root, (dt) => { if (pl.adv(dt) || dirty) { dirty = false; draw(); } }, { fps: 60 });
  };
})();
