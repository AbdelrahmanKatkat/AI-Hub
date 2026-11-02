/* Robotics Module 9 (robo-production.html) labs: stopdist, watchdog, safefilter, latency, fleet.
 * Every simulation is ILLUSTRATIVE: a point mass, a single joint or a synthetic fleet, not a real robot or a certified safety function.
 * Animations run in real time units (dt via AI.loop), slow by default, with speed, pause, step and restart; they redraw only when the state changed. */
(function () {
  const dpOf = (s) => (String(s).split('.')[1] || '').length;
  const field = (id, label, min, max, step, v) =>
    `<label><span>${label} <b class="v" data-for="${id}"></b></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${v}"></label>`;
  const transport = (speedHTML) =>
    `<div class="row"><button class="btn small primary" data-act="play">Pause</button><button class="btn small" data-act="step">Step</button><button class="btn small" data-act="reset">Restart</button></div>` + speedHTML;
  // wires value read-outs for every slider in ctl; onChange runs on any slider move
  const wire = (ctl, onChange) => {
    ctl.querySelectorAll('input[type=range]').forEach((i) => {
      const o = ctl.querySelector(`[data-for="${i.id}"]`);
      const upd = () => { if (o) o.textContent = (+i.value).toFixed(dpOf(i.step)); };
      i.addEventListener('input', () => { upd(); if (onChange) onChange(); });
      upd();
    });
  };
  const val = (ctl, id) => +ctl.querySelector('#' + id).value;
  // play / step / restart buttons; st.play toggles, st.step asks for one fixed step
  const bind = (ctl, st, restart) => {
    const b = (a) => ctl.querySelector(`[data-act="${a}"]`);
    b('play').onclick = () => { st.play = !st.play; b('play').textContent = st.play ? 'Pause' : 'Play'; };
    b('step').onclick = () => { st.play = false; b('play').textContent = 'Play'; st.step = true; };
    b('reset').onclick = () => restart();
  };
  const sp = (min, max, step, v) => field('sp', 'Playback speed', min, max, step, v);
  const hline = (g, y0, x0, x1, col, dash) => { g.save(); g.strokeStyle = col; g.setLineDash(dash || []); g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y0); g.stroke(); g.restore(); };
  const vline = (g, x, y0, y1, col, dash) => { g.save(); g.strokeStyle = col; g.setLineDash(dash || []); g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke(); g.restore(); };
  const txt = (g, s, x, y, col, font, align) => { g.save(); g.fillStyle = col; g.font = font || '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = align || 'left'; g.fillText(s, x, y); g.restore(); };
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : '–');

  /* ================================================================
   * 9.1 · Protective stop: stopping distance against the hazard boundary
   * ================================================================ */
  AI.labs.stopdist = function (root) {
    const { view, ctl } = AI.labShell(root, 'Protective stop on a rail (illustrative)',
      'a point mass moving toward a hazard zone; the sensor trips at D, the robot reacts for t_r, then brakes at a',
      '<canvas style="height:200px"></canvas><div class="readout" id="ro" style="margin-top:8px;font-size:13px"></div><div class="caption" id="cap" style="margin-top:6px"></div>',
      field('dd', 'Sensor trips at D (m)', 0.2, 1.5, 0.05, 0.6) +
      field('hz', 'Hazard zone starts at H (m)', 0.6, 1.6, 0.05, 1.0) +
      field('v', 'Speed v (m/s)', 0.2, 1.5, 0.05, 1.0) +
      field('tr', 'Reaction time t_r (s)', 0, 0.3, 0.01, 0.10) +
      field('a', 'Braking deceleration a (m/s²)', 0.5, 4, 0.1, 2.0) +
      transport(sp(0.1, 1, 0.05, 0.3)));
    const c = AI.canvas(AI.$('canvas', view));
    const st = { play: true, step: false, x: 0, ph: 'approach', tt: 0, v0: 1, dirty: true };
    const X0 = -0.1, X1 = 1.7;
    const reset = () => { Object.assign(st, { x: 0, ph: 'approach', tt: 0, v0: val(ctl, 'v'), dirty: true }); };
    const sim = (secs) => {
      const D = val(ctl, 'dd'), v = val(ctl, 'v'), tr = val(ctl, 'tr'), a = val(ctl, 'a');
      for (let s = 0; s < secs; s += 0.005) {
        if (st.ph === 'approach') { st.x += v * 0.005; if (st.x >= D) { st.ph = 'react'; st.tt = 0; } }
        else if (st.ph === 'react') { st.x += v * 0.005; st.tt += 0.005; if (st.tt >= tr) { st.ph = 'brake'; st.tt = 0; st.v0 = v; } }
        else if (st.ph === 'brake') {
          const vn = st.v0 - a * st.tt;
          if (vn <= 0) { st.ph = 'stopped'; }
          else { st.x += vn * 0.005; st.tt += 0.005; }
        }
        if (st.ph === 'stopped') break;
      }
      st.dirty = true;
    };
    wire(ctl, () => { reset(); });
    bind(ctl, st, reset);
    c.onresize = () => { st.dirty = true; };
    AI.loop(root, (dt) => {
      const sp_ = val(ctl, 'sp');
      if (st.play && st.ph !== 'stopped') sim(dt * sp_);
      else if (st.step) { st.step = false; if (st.ph !== 'stopped') sim(0.25); }
      if (!st.dirty) return;
      st.dirty = false;
      const D = val(ctl, 'dd'), H = val(ctl, 'hz'), v = val(ctl, 'v'), tr = val(ctl, 'tr'), a = val(ctl, 'a');
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      const X = (x) => 40 + ((x - X0) / (X1 - X0)) * (w - 60);
      const yR = h * 0.5;
      g.clearRect(0, 0, w, h);
      // hazard zone
      g.fillStyle = 'rgba(248,81,73,0.16)';
      g.fillRect(X(H), 30, X(X1) - X(H), h - 60);
      txt(g, 'hazard zone', X(H) + 6, 46, C.red);
      // rail and ticks
      g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(X(X0), yR + 26); g.lineTo(X(X1), yR + 26); g.stroke();
      for (let m = 0; m <= 1.6 + 1e-9; m += 0.2) { txt(g, m.toFixed(1) + ' m', X(m), yR + 44, C.muted, '11px ui-monospace,monospace', 'center'); }
      // sensor line
      vline(g, X(D), 30, h - 40, C.yellow, [5, 4]);
      txt(g, 'sensor trips', X(D) + 4, h - 44, C.yellow, '11px ui-monospace,monospace');
      // predicted stop marker
      const pred = D + v * tr + (v * v) / (2 * a);
      vline(g, X(pred), 30, h - 40, C.muted, [2, 3]);
      txt(g, 'predicted stop', X(pred) + 4, 70, C.muted, '11px ui-monospace,monospace');
      // robot block (front edge at x)
      const bw = 0.12 * (w - 60) / (X1 - X0); // block length in px
      g.fillStyle = st.ph === 'brake' ? C.orange : st.ph === 'stopped' ? C.green : C.blue;
      g.beginPath(); g.roundRect(X(st.x) - bw, yR - 22, Math.max(18, bw), 44, 6); g.fill();
      const inHaz = st.x > H;
      const margin = H - st.x;
      txt(g, 'x = ' + st.x.toFixed(3) + ' m', X(st.x) + 8, yR - 30, C.fg, '12px ui-monospace,monospace', 'left');
      const cap = {
        approach: 'Approaching at v. The sensor has not tripped yet, so the robot keeps going.',
        react: 'Sensor tripped. The robot still moves at v during the reaction time t_r, because the stop command takes time to act.',
        brake: 'Braking at a. The distance needed to stop from v is v² / (2a).',
        stopped: 'Stopped at x = ' + st.x.toFixed(3) + ' m.',
      }[st.ph];
      view.querySelector('#cap').textContent = (inHaz ? 'The robot is inside the hazard zone. ' : '') + cap + ' (illustrative)';
      view.querySelector('#ro').innerHTML =
        `stop position x<sub>stop</sub> = <b>${st.x.toFixed(3)} m</b> &nbsp; predicted x = <b>${pred.toFixed(3)} m</b><br>` +
        `margin to hazard H − x = <b style="color:var(--${margin >= 0 ? 'green' : 'red'})">${margin.toFixed(3)} m</b> &nbsp; ${margin >= 0 ? 'safe' : 'NOT safe'}`;
    });
  };

  /* ================================================================
   * 9.2 · Watchdog heartbeat and a sensor residual check
   * ================================================================ */
  AI.labs.watchdog = function (root) {
    const { view, ctl } = AI.labShell(root, 'Heartbeat watchdog and residual check (illustrative)',
      'a controller sends a heartbeat every P ms; the watchdog trips after T ms of silence and stops the motor',
      '<canvas style="height:230px"></canvas><div class="readout" id="ro" style="margin-top:8px;font-size:13px"></div><div class="caption" id="cap" style="margin-top:6px"></div>',
      field('per', 'Heartbeat period P (ms)', 5, 50, 1, 10) +
      field('tmo', 'Watchdog timeout T (ms)', 5, 80, 1, 30) +
      field('vel', 'Robot speed v (m/s)', 0.1, 1.0, 0.05, 0.5) +
      field('tf', 'Heartbeats stop at t_f (ms; 200 = never)', 0, 200, 1, 52) +
      transport(sp(5, 100, 5, 20)) +
      '<hr style="border:0;border-top:1px solid var(--border);margin:10px 0">' +
      field('zm', 'Measured range z (m)', 0.5, 2.5, 0.01, 1.85) +
      field('zp', 'Predicted range ẑ (m)', 0.5, 2.5, 0.01, 1.20) +
      field('sg', 'Sensor noise σ (m)', 0.01, 0.2, 0.01, 0.05) +
      field('kk', 'Gate multiplier k', 1, 5, 0.5, 3) +
      '<div class="readout" id="res" style="font-size:13px"></div>');
    const c = AI.canvas(AI.$('canvas', view));
    const st = { play: true, step: false, t: 0, dirty: true };
    const TMAX = 200;
    const lastBeat = (t, P, tf) => Math.floor(Math.min(t, tf) / P) * P;
    // first time the watchdog trips, scanning the elapsed silence in 0.25 ms steps
    const tripAt = (P, T, tf, tnow) => {
      for (let t = 0; t <= tnow + 1e-9; t += 0.25) { if (t - lastBeat(t, P, tf) > T + 1e-9) return t; }
      return null;
    };
    const restart = () => { st.t = 0; st.dirty = true; };
    wire(ctl, restart);
    bind(ctl, st, restart);
    c.onresize = () => { st.dirty = true; };
    AI.loop(root, (dt) => {
      if (st.play) st.t = Math.min(TMAX, st.t + dt * val(ctl, 'sp'));
      else if (st.step) { st.step = false; st.t = Math.min(TMAX, st.t + 2); }
      if (st.play || st.step) st.dirty = true;
      if (!st.dirty) return;
      st.dirty = false;
      const P = val(ctl, 'per'), T = val(ctl, 'tmo'), v = val(ctl, 'vel'), tf = val(ctl, 'tf');
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h, pl = 90, pr = 16;
      const X = (ms) => pl + (ms / TMAX) * (w - pl - pr);
      g.clearRect(0, 0, w, h);
      const rows = { beat: 46, timer: 118, motor: 190 };
      txt(g, 'controller', 8, rows.beat + 4, C.muted); txt(g, 'watchdog', 8, rows.timer + 4, C.muted); txt(g, 'motor', 8, rows.motor + 4, C.muted);
      // axis
      for (let ms = 0; ms <= TMAX; ms += 25) { txt(g, ms + '', X(ms), h - 6, C.muted, '10px ui-monospace,monospace', 'center'); vline(g, X(ms), 30, h - 18, C.border); }
      const tripT = tripAt(P, T, tf, st.t);
      // heartbeats sent so far
      for (let k = 0; k * P <= Math.min(st.t, tf); k++) {
        g.fillStyle = C.green; g.beginPath(); g.arc(X(k * P), rows.beat, 4, 0, 7); g.fill();
      }
      if (tf < TMAX) vline(g, X(tf), 30, h - 18, C.red, [4, 3]);
      if (tf < TMAX) txt(g, 'fault t_f = ' + tf + ' ms', X(tf) + 4, 36, C.red, '11px ui-monospace,monospace');
      // timer bar: elapsed silence
      const el = st.t - lastBeat(st.t, P, tf);
      g.fillStyle = C.bg3 || C.border; g.fillRect(X(0), rows.timer - 8, X(TMAX) - X(0), 16);
      g.fillStyle = el > T ? C.red : C.blue; g.fillRect(X(0), rows.timer - 8, Math.max(0, X(Math.min(el, TMAX)) - X(0)), 16);
      vline(g, X(T), rows.timer - 14, rows.timer + 14, C.orange, [3, 3]);
      txt(g, 'T = ' + T + ' ms', X(T) + 4, rows.timer - 14, C.orange, '11px ui-monospace,monospace');
      // motor
      const tripped = tripT !== null;
      const moveEnd = tripped ? Math.min(st.t, tripT) : st.t;
      g.fillStyle = tripped ? C.red : C.green; g.fillRect(X(0), rows.motor - 8, X(moveEnd) - X(0), 16);
      if (tripped) { txt(g, 'STOP', X(tripT) + 6, rows.motor + 4, C.red); vline(g, X(tripT), rows.motor - 14, rows.motor + 14, C.red); }
      const coast = tf < st.t && tripped ? v * (tripT - tf) / 1000 : null;
      const statusTxt = tripped ? 'TRIPPED: motor stopped at ' + tripT.toFixed(2) + ' ms' : 'running: heartbeat healthy';
      view.querySelector('#cap').textContent = tripped
        ? (tf < st.t ? 'Heartbeats stopped at t_f. The watchdog tripped after T of silence, so the robot coasted for the time between the two events.' : 'The watchdog tripped even though beats arrive. With T ≤ P, a normal gap between beats is already too long (trap).')
        : 'Heartbeats arrive every P ms. Silence never exceeds T, so the motor keeps running. (illustrative)';
      // residual check
      const z = val(ctl, 'zm'), zp = val(ctl, 'zp'), sg = val(ctl, 'sg'), kk = val(ctl, 'kk');
      const r = z - zp, gate = kk * sg;
      ctl.querySelector('#res').innerHTML = `residual r = z − ẑ = <b>${r.toFixed(3)} m</b>, gate k·σ = <b>${gate.toFixed(3)} m</b> → ` +
        (Math.abs(r) > gate ? '<b style="color:var(--red)">fault flagged (|r| > kσ)</b>' : '<b style="color:var(--green)">consistent (|r| ≤ kσ)</b>');
      view.querySelector('#ro').innerHTML =
        `${statusTxt}<br>last beat at ${fmt(lastBeat(st.t, P, tf), 1)} ms; elapsed silence now ${fmt(el, 1)} ms` +
        (coast !== null ? `<br>coast distance v·(t_trip − t_f) = <b>${coast.toFixed(4)} m</b> (${(coast * 1000).toFixed(1)} mm)` : '');
    });
  };

  /* ================================================================
   * 9.3 · Safety filter on a learned joint command (1-D joint, control-barrier style limit)
   * ================================================================ */
  AI.labs.safefilter = function (root) {
    const QMAX = 1.5, QMIN = -1.5, DT = 0.01, Q0 = 1.2;
    const { view, ctl } = AI.labShell(root, 'Safety filter on a learned joint command (illustrative)',
      'a 1-D joint; the policy proposes a velocity; the filter applies v_safe = min(v_prop, v_max, γ(q_max − q)) and its lower twin',
      '<canvas style="height:300px"></canvas><div class="readout" id="ro" style="margin-top:8px;font-size:13px"></div><div class="caption" id="cap" style="margin-top:6px"></div>',
      `<label><span>Learned policy: v_prop(t, q) in rad/s</span><input type="text" id="pol" value="0.9 + 0.6*sin(0.8*t)" spellcheck="false"></label><div class="err" id="perr"></div>` +
      field('vm', 'Velocity limit v_max (rad/s)', 0.1, 1.0, 0.05, 0.5) +
      field('gm', 'Barrier gain γ (1/s)', 0.5, 5, 0.1, 2) +
      transport(sp(0.1, 1, 0.05, 0.3)));
    const c = AI.canvas(AI.$('canvas', view));
    let policy = null;
    const st = { play: true, step: false, t: 0, q: Q0, acc: 0, hist: [], dirty: true, lastSample: -1 };
    const restart = () => { Object.assign(st, { t: 0, q: Q0, acc: 0, hist: [], dirty: true, lastSample: -1 }); };
    AI.exprInput(ctl.querySelector('#pol'), ctl.querySelector('#perr'), { allowed: ['t', 'q'], onChange: (f) => { policy = f; restart(); } }).run();
    const stepSim = () => {
      const vmax = val(ctl, 'vm'), gam = val(ctl, 'gm');
      let vp = policy ? policy({ t: st.t, q: st.q }) : 0;
      if (!Number.isFinite(vp)) vp = 0;
      const up = Math.min(vmax, gam * (QMAX - st.q)), lo = Math.max(-vmax, -gam * (st.q - QMIN));
      const vs = AI.clamp(vp, lo, up);
      if (st.t - st.lastSample >= 0.05 - 1e-9 || st.lastSample < 0) {
        st.hist.push({ t: st.t, q: st.q, vp, vs }); if (st.hist.length > 600) st.hist.shift(); st.lastSample = st.t;
      }
      st.q += vs * DT; st.t += DT;
    };
    wire(ctl, restart);
    bind(ctl, st, restart);
    c.onresize = () => { st.dirty = true; };
    AI.loop(root, (dt) => {
      if (st.play) st.acc += dt * val(ctl, 'sp');
      else if (st.step) { st.step = false; st.acc += 0.25; }
      let n = 0;
      while (st.acc >= DT && n < 60) { stepSim(); st.acc -= DT; n++; }
      if (n) st.dirty = true;
      if (!st.dirty) return;
      st.dirty = false;
      const vmax = val(ctl, 'vm'), gam = val(ctl, 'gm');
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const x0 = Math.max(0, st.t - 15), x1 = Math.max(15, st.t);
      const pl = 52, pr = 12, top = { y0: 14, y1: h / 2 - 14, lo: -1.7, hi: 1.7 }, bot = { y0: h / 2 + 14, y1: h - 22, lo: -1.2, hi: 1.8 };
      const X = (t) => pl + ((t - x0) / (x1 - x0)) * (w - pl - pr);
      const Yt = (y) => top.y1 - ((y - top.lo) / (top.hi - top.lo)) * (top.y1 - top.y0);
      const Yb = (y) => bot.y1 - ((y - bot.lo) / (bot.hi - bot.lo)) * (bot.y1 - bot.y0);
      // top: joint angle with limits
      txt(g, 'joint q (rad)', pl, top.y0 + 10, C.fg, '11px ui-monospace,monospace');
      [QMAX, QMIN, 0].forEach((y) => hline(g, Yt(y), pl, w - pr, y === 0 ? C.border : C.red, y === 0 ? [] : [5, 4]));
      txt(g, '+1.5 limit', w - pr - 4, Yt(QMAX) - 4, C.red, '10px ui-monospace,monospace', 'right');
      txt(g, '−1.5 limit', w - pr - 4, Yt(QMIN) + 12, C.red, '10px ui-monospace,monospace', 'right');
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      st.hist.forEach((s, i) => { const px = X(s.t), py = Yt(s.q); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
      // bottom: proposed vs applied velocity
      txt(g, 'velocity (rad/s): proposed (orange, dashed) vs applied (green)', pl, bot.y0 + 10, C.fg, '11px ui-monospace,monospace');
      hline(g, Yb(0), pl, w - pr, C.border);
      hline(g, Yb(vmax), pl, w - pr, C.muted, [3, 3]); hline(g, Yb(-vmax), pl, w - pr, C.muted, [3, 3]);
      g.strokeStyle = C.orange; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath();
      st.hist.forEach((s, i) => { const px = X(s.t), py = Yb(AI.clamp(s.vp, bot.lo, bot.hi)); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.green; g.lineWidth = 2.5; g.beginPath();
      st.hist.forEach((s, i) => { const px = X(s.t), py = Yb(s.vs); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
      txt(g, 'time t (s)', w - pr, h - 6, C.muted, '10px ui-monospace,monospace', 'right');
      // read-outs at the current state
      const q = st.q, hb = QMAX - q, cap = gam * hb;
      const vpNow = policy ? policy({ t: st.t, q }) : 0;
      const up = Math.min(vmax, cap), lo = Math.max(-vmax, -gam * (q - QMIN));
      const vsNow = AI.clamp(Number.isFinite(vpNow) ? vpNow : 0, lo, up);
      const clipped = Math.abs(vsNow - vpNow) > 1e-9;
      view.querySelector('#ro').innerHTML =
        `t = <b>${st.t.toFixed(2)} s</b> &nbsp; q = <b>${q.toFixed(3)} rad</b> &nbsp; h = q_max − q = <b>${hb.toFixed(3)} rad</b><br>` +
        `γ·h = <b>${cap.toFixed(3)} rad/s</b> &nbsp; v_max = <b>${vmax.toFixed(2)} rad/s</b> &nbsp; proposed v_prop = <b>${(Number.isFinite(vpNow) ? vpNow : 0).toFixed(3)} rad/s</b><br>` +
        `applied v_safe = <b style="color:var(--${clipped ? 'orange' : 'green'})">${vsNow.toFixed(3)} rad/s</b> ${clipped ? '(clipped by the filter)' : '(unchanged)'}`;
      view.querySelector('#cap').textContent = 'Near the limit the allowed speed γ·h shrinks toward 0, so the joint slows down before it reaches +1.5 rad. (illustrative, one joint, Euler step 0.01 s)';
    });
  };

  /* ================================================================
   * 9.5 · Latency budget for a control loop
   * ================================================================ */
  AI.labs.latency = function (root) {
    const { view, ctl } = AI.labShell(root, 'Latency budget for one loop cycle (illustrative)',
      'each cycle must finish inside its period 1000/f ms; stage times add up; jitter adds a random spread',
      '<canvas style="height:200px"></canvas><div class="readout" id="ro" style="margin-top:8px;font-size:13px"></div><div class="caption" id="cap" style="margin-top:6px"></div>',
      field('hz', 'Loop rate f (Hz)', 1, 1000, 1, 10) +
      field('cam', 'Sensor capture (ms)', 0, 50, 0.5, 33) +
      field('per', 'Perception compute (ms)', 0, 80, 0.5, 40) +
      field('pl', 'Planning (ms)', 0, 40, 0.5, 12) +
      field('net', 'Network hop (ms)', 0, 50, 0.5, 8) +
      field('act', 'Actuation command (ms)', 0, 10, 0.1, 2) +
      field('jit', 'Jitter σ (fraction of nominal)', 0, 0.3, 0.01, 0.10) +
      transport(sp(0.2, 3, 0.1, 1)));
    const c = AI.canvas(AI.$('canvas', view));
    const st = { play: true, step: false, k: 0, acc: 0, dirty: true, totals: [], key: '' };
    const STAGES = [['cam', 'capture', 'blue'], ['per', 'perceive', 'purple'], ['pl', 'plan', 'teal'], ['net', 'network', 'orange'], ['act', 'actuate', 'green']];
    const build = () => {
      const nom = STAGES.map(([id]) => val(ctl, id));
      const sig = val(ctl, 'jit'), rnd = AI.rng(11);
      st.totals = [];
      for (let k = 0; k < 400; k++) {
        let tot = 0;
        nom.forEach((m) => { tot += Math.max(0, m * (1 + sig * AI.randn(rnd))); });
        st.totals.push(tot);
      }
      st.key = nom.join(',') + sig;
    };
    const restart = () => { st.k = 0; st.acc = 0; st.dirty = true; };
    wire(ctl, () => { build(); restart(); });
    build();
    bind(ctl, st, restart);
    c.onresize = () => { st.dirty = true; };
    AI.loop(root, (dt) => {
      if (st.play) st.acc += dt * val(ctl, 'sp');
      else if (st.step) { st.step = false; st.acc += 1; }
      while (st.acc >= 1) { st.acc -= 1; st.k = (st.k + 1) % st.totals.length; st.dirty = true; }
      if (!st.dirty) return;
      st.dirty = false;
      const hz = val(ctl, 'hz'), budget = 1000 / hz;
      const nomTotal = STAGES.reduce((s, [id]) => s + val(ctl, id), 0);
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const scale = Math.max(budget, nomTotal, 1) * 1.15;
      const pl = 16, pr = 16, bw = w - pl - pr;
      const X = (ms) => pl + (ms / scale) * bw;
      // nominal stacked bar
      let acc = 0;
      txt(g, 'nominal stage times, stacked', pl, 22, C.fg, '12px ui-monospace,monospace');
      STAGES.forEach(([id, name, col]) => {
        const m = val(ctl, id);
        g.fillStyle = C[col]; g.fillRect(X(acc), 34, Math.max(0, X(acc + m) - X(acc)), 34);
        if (X(acc + m) - X(acc) > 46) txt(g, name + ' ' + m.toFixed(1), X(acc) + 4, 56, C.bg, '11px ui-monospace,monospace');
        acc += m;
      });
      vline(g, X(budget), 26, 82, C.red, [5, 4]);
      txt(g, 'budget ' + budget.toFixed(1) + ' ms', Math.min(X(budget) + 4, w - 120), 98, C.red, '11px ui-monospace,monospace');
      txt(g, 'nominal total ' + nomTotal.toFixed(1) + ' ms', pl, 118, C.fg, '11px ui-monospace,monospace');
      // history strip: last 60 cycles
      const hist = 60, gx0 = pl, gy1 = h - 26, gy0 = 136, gw = bw / hist;
      hline(g, gy1, gx0, w - pr, C.border);
      const yOf = (ms) => gy1 - (Math.min(ms, scale) / scale) * (gy1 - gy0);
      let misses = 0, worst = 0;
      for (let i = 0; i < hist; i++) {
        const idx = (st.k - hist + i + st.totals.length * 2) % st.totals.length;
        const tot = st.totals[idx];
        if (tot > budget) misses++;
        worst = Math.max(worst, tot);
        g.fillStyle = tot > budget ? C.red : C.green;
        g.fillRect(gx0 + i * gw + 1, yOf(tot), Math.max(1, gw - 2), gy1 - yOf(tot));
      }
      hline(g, yOf(budget), gx0, w - pr, C.red, [4, 3]);
      const cur = st.totals[st.k], margin = budget - cur;
      view.querySelector('#ro').innerHTML =
        `cycle <b>${st.k + 1}</b>: total <b>${cur.toFixed(2)} ms</b> vs budget <b>${budget.toFixed(2)} ms</b> → margin <b style="color:var(--${margin >= 0 ? 'green' : 'red'})">${margin.toFixed(2)} ms</b><br>` +
        `last 60 cycles: <b>${misses}</b> over budget; worst total <b>${worst.toFixed(2)} ms</b>`;
      view.querySelector('#cap').textContent = (margin >= 0 ? 'This cycle finished inside its period.' : 'This cycle MISSED its deadline; the command goes out late.') +
        ' Green bars: met. Red bars: missed. Jitter makes some cycles longer than the nominal sum. (illustrative)';
    });
  };

  /* ================================================================
   * 9.6 · Fleet: calibration drift against a tolerance, and a canary rollout
   * ================================================================ */
  AI.labs.fleet = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fleet: calibration drift and a canary rollout (illustrative)',
      'a force sensor drifts by r N per day; recalibrate when the offset reaches the tolerance; then roll an update out in stages',
      '<canvas style="height:220px"></canvas><div class="readout" id="ro" style="margin-top:8px;font-size:13px"></div><div class="caption" id="cap" style="margin-top:6px"></div>',
      field('r', 'Drift rate r (N/day)', 0.005, 0.1, 0.001, 0.02) +
      field('tol', 'Tolerance (N)', 0.2, 1.0, 0.05, 0.5) +
      field('cn', 'Canary robots n', 5, 200, 1, 10) +
      field('cf', 'Canary failures f', 0, 10, 1, 1) +
      transport(sp(1, 10, 0.5, 4)));
    const c = AI.canvas(AI.$('canvas', view));
    const st = { play: true, step: false, day: 0, cal: 0, dirty: true };
    const HORIZON = 60;
    const restart = () => { Object.assign(st, { day: 0, cal: 0, dirty: true }); };
    wire(ctl, restart);
    bind(ctl, st, restart);
    c.onresize = () => { st.dirty = true; };
    const wilson = (f, n, z = 1.96) => {
      const p = f / n, d = 1 + (z * z) / n, cen = (p + (z * z) / (2 * n)) / d;
      const hw = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
      return [Math.max(0, cen - hw), Math.min(1, cen + hw)];
    };
    AI.loop(root, (dt) => {
      const spd = val(ctl, 'sp');
      if (st.play) st.day = Math.min(HORIZON, st.day + dt * spd);
      else if (st.step) { st.step = false; st.day = Math.min(HORIZON, st.day + 1); }
      const r = val(ctl, 'r'), tol = val(ctl, 'tol');
      const P = tol / r; // days between recalibrations: the offset reaches tol after P days
      const cal = Math.floor(st.day / P);
      if (cal !== st.cal) { st.cal = cal; st.dirty = true; }
      if (st.day !== st._lastDay) { st._lastDay = st.day; st.dirty = true; }
      if (!st.dirty) return;
      st.dirty = false;
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const pl = 46, pr = 14, pt = 16, pb = 26, YMAX = 1.2;
      const X = (d) => pl + (d / HORIZON) * (w - pl - pr), Y = (n) => pt + (1 - n / YMAX) * (h - pt - pb);
      for (let n = 0; n <= YMAX + 1e-9; n += 0.3) { txt(g, n.toFixed(1), pl - 6, Y(n) + 4, C.muted, '10px ui-monospace,monospace', 'right'); hline(g, Y(n), pl, w - pr, C.border); }
      for (let d = 0; d <= HORIZON; d += 10) txt(g, d + ' d', X(d), h - 8, C.muted, '10px ui-monospace,monospace', 'center');
      hline(g, Y(tol), pl, w - pr, C.red, [5, 4]);
      txt(g, 'tolerance ' + tol.toFixed(2) + ' N', w - pr - 4, Y(tol) - 5, C.red, '10px ui-monospace,monospace', 'right');
      // offset sawtooth up to the current day
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      const steps = 240;
      for (let i = 0; i <= steps; i++) {
        const d = (st.day * i) / steps, o = r * (d - Math.floor(d / P) * P);
        i ? g.lineTo(X(d), Y(o)) : g.moveTo(X(d), Y(o));
      }
      g.stroke();
      vline(g, X(st.day), pt, h - pb, C.yellow, [2, 3]);
      // rollout strip: canary (stage 1), 10 %, 100 %
      const cn = val(ctl, 'cn'), cf = val(ctl, 'cf');
      const [lo, hi] = wilson(cf, cn);
      const p = cf / cn;
      const verdict = hi < 0.05 ? 'failure rate clearly below 5 %: continue' : lo > 0.05 ? 'failure rate clearly above 5 %: halt' : 'undecided at 5 %: keep the canary running';
      const off = r * (st.day - cal * P);
      view.querySelector('#ro').innerHTML =
        `day <b>${st.day.toFixed(1)}</b> &nbsp; offset now <b>${off.toFixed(3)} N</b> &nbsp; recalibrations so far <b>${st.cal}</b> &nbsp; days between recalibrations = tol / r = <b>${P.toFixed(1)} d</b><br>` +
        `canary failure rate f / n = <b>${(100 * p).toFixed(1)} %</b>, 95 % interval [<b>${(100 * lo).toFixed(1)} %</b>, <b>${(100 * hi).toFixed(1)} %</b>]<br>` +
        `rule: ${verdict}`;
      view.querySelector('#cap').textContent = 'Drift grows as a straight line until the offset reaches the tolerance, then the sensor is recalibrated (drops to 0). The canary check below is a separate question, not part of the drift. (illustrative)';
    });
  };
})();
