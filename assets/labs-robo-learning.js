/* Robotics · Module 7 (Learning for robots) labs. All are illustrative toys, not robot data; each states its simplifications.
 * bc-rollout       <div data-lab="bc-rollout"></div>       behaviour cloning: fitted policy versus expert on a lateral-error toy
 * dagger-loop      <div data-lab="dagger-loop"></div>      DAgger: label the learner's own visited states and refit
 * mdp-values       <div data-lab="mdp-values"></div>       value-iteration sweeps on a six-cell corridor
 * ppo-clip         <div data-lab="ppo-clip"></div>         the clipped PPO objective as a function of the ratio r
 * bandit-pg        <div data-lab="bandit-pg"></div>        REINFORCE on a three-armed bandit, with or without a baseline
 * sim-random       <div data-lab="sim-random"></div>       domain randomisation on a one-dimensional reaching task
 * reward-routes    <div data-lab="reward-routes"></div>    proxy versus true reward, with a hazard budget
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';

  /* ---------- shared helpers ---------- */
  const slider = (ctl, name, min, max, step, val, unit, dec = 2) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = AI.fmt(+inp.value, dec) + (unit ? ' ' + unit : ''));
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const readout = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; };

  /* transport: play/pause, step, reset and a speed multiplier. Step adds one pending step for the lab loop. */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 2, 0.05, speed, '×', 2);
    T.setPlaying = (p) => { T.playing = p; pp.textContent = p ? 'Pause' : 'Play'; };
    pp.onclick = () => T.setPlaying(!T.playing);
    st.onclick = () => { T.setPlaying(false); T.pending += 1; };
    rs.onclick = () => onReset();
    return T;
  };

  /* redraw only when the state key, canvas size or theme changes */
  const live = (root, canvases, key, update, draw) => {
    let last = null;
    AI.loop(root, (dt) => {
      update(dt);
      const k = key() + '|' + canvases.map((c) => c.w + 'x' + c.h).join(',') + '|' + document.documentElement.dataset.theme;
      if (k === last) return;
      last = k; draw();
    });
  };

  const curve = (g, v, f, color, w = 2.5, dash = null) => {
    g.save(); g.strokeStyle = color; g.lineWidth = w; if (dash) g.setLineDash(dash);
    g.beginPath(); let pen = false;
    for (let i = 0; i <= 300; i++) {
      const x = v.xmin + ((v.xmax - v.xmin) * i) / 300, y = f(x);
      if (!Number.isFinite(y)) { pen = false; continue; }
      const px = v.X(x), py = v.Y(AI.clamp(y, v.ymin - 1, v.ymax + 1));
      if (!pen) { g.moveTo(px, py); pen = true; } else g.lineTo(px, py);
    }
    g.stroke(); g.restore();
  };

  /* least-squares polynomial fit (normal equations, Gaussian elimination) */
  const fitPoly = (S, U, deg) => {
    const A = Array.from({ length: deg + 1 }, () => Array(deg + 1).fill(0)), b = Array(deg + 1).fill(0);
    S.forEach((e, i) => {
      const r = []; for (let k = 0; k <= deg; k++) r.push(e ** k);
      for (let p = 0; p <= deg; p++) { b[p] += r[p] * U[i]; for (let q = 0; q <= deg; q++) A[p][q] += r[p] * r[q]; }
    });
    const M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i <= deg; i++) {
      const piv = M[i][i];
      for (let j = i + 1; j <= deg; j++) { const f = M[j][i] / piv; for (let k = i; k <= deg + 1; k++) M[j][k] -= f * M[i][k]; }
    }
    const c = Array(deg + 1).fill(0);
    for (let i = deg; i >= 0; i--) { let s = M[i][deg + 1]; for (let j = i + 1; j <= deg; j++) s -= M[i][j] * c[j]; c[i] = s / M[i][i]; }
    return { c, f: (e) => { let s = 0; for (let k = 0; k <= deg; k++) s += c[k] * e ** k; return s; } };
  };

  /* ---------- lateral-error toy shared by bc-rollout and dagger-loop ----------
     state e (m) = lateral error; expert command u*(e) = -1.5 sin(e) (m/s); step dt = 0.5 s; states clipped to +-4 m. */
  const EXPERT = (e) => -1.5 * Math.sin(e);
  const rollout = (pol, e0, K = 12, dt = 0.5) => {
    const xs = [e0]; let e = e0;
    for (let t = 0; t < K; t++) { e = AI.clamp(e + dt * pol(e), -4, 4); xs.push(e); }
    return xs;
  };
  const DEMO = (() => {
    const S = [], U = [];
    for (const e0 of [0.6, -0.6, 0.3]) { let e = e0; for (let t = 0; t < 8; t++) { S.push(e); U.push(EXPERT(e)); e += 0.5 * EXPERT(e); } }
    return { S, U };
  })();

  /* one horizontal row per rollout; k = number of steps shown */
  const drawTrack = (cT, rows, k, band) => {
    const C = AI.col(), g = cT.ctx, v = AI.view(cT, -4.5, 4.5, -1, 1, { l: 78, r: 12, t: 8, b: 22 });
    g.clearRect(0, 0, cT.w, cT.h);
    if (band) { g.globalAlpha = 0.12; g.fillStyle = C.blue; g.fillRect(v.X(band[0]), v.pad.t, v.X(band[1]) - v.X(band[0]), v.ph); g.globalAlpha = 1; }
    AI.axes(g, v, { noY: true });
    g.strokeStyle = C.muted; g.lineWidth = 1; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(v.X(0) + 0.5, v.pad.t); g.lineTo(v.X(0) + 0.5, v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
    g.font = '11px ' + MONO;
    rows.forEach((r) => {
      const y = v.Y(r.y), n = Math.min(k, r.xs.length - 1);
      g.strokeStyle = r.color; g.globalAlpha = 0.4; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= n; i++) { const px = v.X(AI.clamp(r.xs[i], -4.5, 4.5)); if (i) g.lineTo(px, y); else g.moveTo(px, y); }
      g.stroke(); g.globalAlpha = 1;
      g.fillStyle = r.color; g.beginPath(); g.arc(v.X(AI.clamp(r.xs[n], -4.5, 4.5)), y, 6, 0, 6.2832); g.fill();
      g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(r.label, 6, y + 4);
    });
    g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('lateral error e (m)', v.X(0), cT.h - 4);
  };

  /* ================= 7.1 behaviour cloning ================= */
  AI.labs['bc-rollout'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Behaviour cloning on a lateral-error toy',
      'illustrative · a least-squares polynomial copies the expert on the states it was shown, then drives on its own',
      `<canvas class="pol" style="height:240px"></canvas><canvas class="trk" style="height:130px;margin-top:10px"></canvas>`, '');
    const cP = AI.canvas(AI.$('canvas.pol', view)), cT = AI.canvas(AI.$('canvas.trk', view));
    const deg = slider(ctl, 'Learner polynomial degree', 1, 3, 1, 3, '', 0);
    const e0s = slider(ctl, 'Start error e₀', -3, 3, 0.1, 2.5, 'm', 1);
    const out = readout(ctl);
    const K = 12, tk = AI.ticker();
    let k = 0, fit = null, coef = [], pathE = [], pathL = [], pKey = '';
    const refresh = () => {
      const d = deg(), s0 = e0s(), key = d + '|' + s0;
      if (key === pKey) return;
      pKey = key;
      const r = fitPoly(DEMO.S, DEMO.U, d);
      fit = r.f; coef = r.c;
      pathE = rollout(EXPERT, s0, K); pathL = rollout(fit, s0, K);
    };
    const T = transport(ctl, () => { k = 0; T.setPlaying(true); });
    const update = (dt) => {
      refresh();
      let n = T.playing ? tk(dt, 0.8 * T.speed()) : 0;
      n += T.pending; T.pending = 0;
      k = Math.min(K, k + n);
      if (k >= K && T.playing) T.setPlaying(false);
    };
    const draw = () => {
      refresh();
      const C = AI.col(), g = cP.ctx, v = AI.view(cP, -3.5, 3.5, -3, 3);
      g.clearRect(0, 0, cP.w, cP.h); AI.axes(g, v);
      g.globalAlpha = 0.12; g.fillStyle = C.blue; g.fillRect(v.X(-0.6), v.pad.t, v.X(0.6) - v.X(-0.6), v.ph); g.globalAlpha = 1;
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      curve(g, v, EXPERT, C.green, 2.5);
      curve(g, v, fit, C.orange, 2.5);
      g.fillStyle = C.blue; DEMO.S.forEach((s, i) => { g.beginPath(); g.arc(v.X(s), v.Y(DEMO.U[i]), 3.5, 0, 6.2832); g.fill(); });
      g.restore();
      g.font = '11px ' + MONO; g.textAlign = 'left';
      g.fillStyle = C.green; g.fillText('expert u*(e) = −1.5 sin e', v.pad.l + 8, v.pad.t + 14);
      g.fillStyle = C.orange; g.fillText('learner u(e), fitted to demonstrations', v.pad.l + 8, v.pad.t + 28);
      g.fillStyle = C.blue; g.fillText('demonstrated states (blue band: |e| ≤ 0.6 m)', v.pad.l + 8, v.pad.t + 42);
      drawTrack(cT, [
        { xs: pathE, y: 0.45, color: C.green, label: 'expert' },
        { xs: pathL, y: -0.45, color: C.orange, label: 'learner' },
      ], k, [-0.6, 0.6]);
      const eE = pathE[k], eL = pathL[k];
      const clipped = pathL.some((x) => Math.abs(x) >= 4);
      out.textContent =
        `step ${k} of ${K}   (each step 0.5 s, so t = ${AI.fmt(k * 0.5, 2)} s)\n` +
        `expert   e = ${AI.fmt(eE, 3)} m   u* = ${AI.fmt(EXPERT(eE), 3)} m/s\n` +
        `learner  e = ${AI.fmt(eL, 3)} m   u  = ${AI.fmt(fit(eL), 3)} m/s\n` +
        `learner coefficients c0..cd = ${coef.map((c) => AI.fmt(c, 3)).join(', ')}\n` +
        (clipped ? 'the learner left the demonstrated region and was clipped at ±4 m' : 'the learner stays within ±4 m');
    };
    live(root, [cP, cT], () => k + '|' + pKey, update, draw);
  };

  /* ================= 7.2 DAgger ================= */
  AI.labs['dagger-loop'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'DAgger: ask the expert about the learner’s own states',
      'illustrative · same toy, cubic learner · each iteration rolls the learner out, labels every visited state with the expert, refits',
      `<canvas class="pol" style="height:240px"></canvas><canvas class="trk" style="height:130px;margin-top:10px"></canvas>`, '');
    const cP = AI.canvas(AI.$('canvas.pol', view)), cT = AI.canvas(AI.$('canvas.trk', view));
    const e0s = slider(ctl, 'Test start e₀', -3, 3, 0.1, 2.5, 'm', 1);
    const out = readout(ctl);
    const STARTS = [-2.5, -1.5, 1.5, 2.5, 3.0], MAXIT = 8, DEG = 3;
    let S = [], U = [], added = [], iter = 0, fit = null, coef = [], auto = 0, prog = 0, pathE = [], pathL = [], pKey = '';
    const refit = () => { const r = fitPoly(S, U, DEG); fit = r.f; coef = r.c; };
    const reset = () => { S = DEMO.S.slice(); U = DEMO.U.slice(); added = []; iter = 0; auto = 0; prog = 0; refit(); pKey = ''; T.setPlaying(true); };
    const iterate = () => {
      if (iter >= MAXIT) return;
      for (const s0 of STARTS) for (const e of rollout(fit, s0, 12)) { S.push(e); U.push(EXPERT(e)); added.push(e); }
      iter++; refit(); pKey = '';
    };
    const T = transport(ctl, reset);
    reset();
    const refreshPath = () => {
      const s0 = e0s(), key = iter + '|' + s0;
      if (key === pKey) return;
      pKey = key; pathE = rollout(EXPERT, s0, 12); pathL = rollout(fit, s0, 12);
    };
    const update = (dt) => {
      if (T.playing && iter < MAXIT) { auto += dt * T.speed(); if (auto >= 3) { auto = 0; iterate(); } }
      if (iter >= MAXIT && T.playing) T.setPlaying(false);
      if (T.pending) { T.pending = 0; iterate(); }
      if (T.playing) prog = (prog + dt * T.speed()) % 14;
      refreshPath();
    };
    const draw = () => {
      refreshPath();
      const C = AI.col(), g = cP.ctx, v = AI.view(cP, -3.5, 3.5, -3, 3);
      g.clearRect(0, 0, cP.w, cP.h); AI.axes(g, v);
      g.globalAlpha = 0.12; g.fillStyle = C.blue; g.fillRect(v.X(-0.6), v.pad.t, v.X(0.6) - v.X(-0.6), v.ph); g.globalAlpha = 1;
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      curve(g, v, EXPERT, C.green, 2.5);
      curve(g, v, fit, C.orange, 2.5);
      g.fillStyle = C.purple; added.forEach((s) => { g.beginPath(); g.arc(v.X(s), v.Y(EXPERT(s)), 2, 0, 6.2832); g.fill(); });
      g.fillStyle = C.blue; DEMO.S.forEach((s, i) => { g.beginPath(); g.arc(v.X(s), v.Y(DEMO.U[i]), 3.5, 0, 6.2832); g.fill(); });
      g.restore();
      g.font = '11px ' + MONO; g.textAlign = 'left';
      g.fillStyle = C.green; g.fillText('expert u*(e)', v.pad.l + 8, v.pad.t + 14);
      g.fillStyle = C.orange; g.fillText('learner after iteration ' + iter, v.pad.l + 8, v.pad.t + 28);
      g.fillStyle = C.purple; g.fillText('states the learner visited, labelled by the expert', v.pad.l + 8, v.pad.t + 42);
      g.fillStyle = C.blue; g.fillText('original demonstrations', v.pad.l + 8, v.pad.t + 56);
      const kk = Math.min(12, Math.floor(prog));
      drawTrack(cT, [
        { xs: pathE, y: 0.45, color: C.green, label: 'expert' },
        { xs: pathL, y: -0.45, color: C.orange, label: 'learner' },
      ], kk, [-0.6, 0.6]);
      const errs = [-2.5, 2.5, 3.0].map((s) => rollout(fit, s, 12).slice(-1)[0]);
      out.textContent =
        `iteration ${iter} of ${MAXIT} (iteration 0 = plain behaviour cloning)\n` +
        `labelled states: ${DEMO.S.length} demonstrated + ${added.length} visited = ${S.length}\n` +
        `learner e after 12 steps from e0 = −2.5, 2.5, 3.0 m: ${errs.map((x) => AI.fmt(x, 3)).join(', ')} m\n` +
        `cubic coefficients c0..c3 = ${coef.map((c) => AI.fmt(c, 3)).join(', ')}\n` +
        (iter === 0 ? 'Press Play or Step: each iteration adds the learner’s own visited states.' : 'Each iteration adds states the learner actually reaches, labelled by the expert.');
    };
    live(root, [cP, cT], () => iter + '|' + S.length + '|' + pKey + '|' + Math.floor(prog) + '|' + T.playing, update, draw);
  };

  /* ================= 7.3 value iteration (RL basics) ================= */
  AI.labs['mdp-values'] = function (root) {
    const N = 6, GOAL = 5, PIT = 0;
    const { view, ctl } = AI.labShell(root, 'Value iteration on a six-cell corridor',
      'illustrative · a robot moves left or right; entering the goal (cell 5) pays +1, entering the pit (cell 0) pays −1; both end the episode',
      `<canvas style="height:250px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const gam = slider(ctl, 'Discount γ', 0.5, 0.99, 0.01, 0.9, '', 2);
    const cost = slider(ctl, 'Cost per move c', 0, 0.2, 0.01, 0.04, '', 2);
    const out = readout(ctl);
    let V = [], sweeps = 0, conv = false, timer = 0, prm = '';
    const reset = () => { V = Array(N).fill(0); sweeps = 0; conv = false; timer = 0; prm = gam() + '|' + cost(); };
    const q = (s, d, Vs, g, c) => {
      const s2 = s + d;
      const r = (s2 === GOAL ? 1 : s2 === PIT ? -1 : 0) - c;
      const v2 = (s2 === GOAL || s2 === PIT) ? 0 : Vs[s2];
      return r + g * v2;
    };
    const backup = (Vs, g, c) => {
      const NV = Vs.slice();
      for (let s = 1; s < N - 1; s++) NV[s] = Math.max(q(s, -1, Vs, g, c), q(s, 1, Vs, g, c));
      return NV;
    };
    reset();
    const T = transport(ctl, reset);
    const update = (dt) => {
      if (gam() + '|' + cost() !== prm) reset();
      let go = 0;
      if (T.playing && !conv) { timer += dt * T.speed(); if (timer >= 1.2) { timer = 0; go = 1; } }
      if (T.pending) { go += T.pending; T.pending = 0; }
      for (let i = 0; i < go && !conv; i++) {
        const NV = backup(V, gam(), cost());
        const dmax = Math.max(...NV.map((x, j) => Math.abs(x - V[j])));
        V = NV; sweeps++;
        if (dmax < 1e-6) conv = true;
      }
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, v = AI.view(cv, -0.5, N - 0.5, -1.35, 1.35, { l: 40, r: 12, t: 12, b: 40 });
      const cw = v.pw / N;
      g.clearRect(0, 0, cv.w, cv.h);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(v.pad.l, v.Y(0)); g.lineTo(v.pad.l + v.pw, v.Y(0)); g.stroke();
      g.font = '11px ' + MONO; g.textAlign = 'center';
      for (let s = 0; s < N; s++) {
        const x = v.X(s), terminal = s === GOAL || s === PIT;
        const col = s === GOAL ? C.green : s === PIT ? C.red : (V[s] >= 0 ? C.blue : C.orange);
        const top = v.Y(V[s]), base = v.Y(0);
        g.fillStyle = col; g.globalAlpha = terminal ? 0.55 : 0.6;
        g.fillRect(x - cw * 0.28, Math.min(top, base), cw * 0.56, Math.abs(base - top) || 1);
        g.globalAlpha = 1;
        g.fillStyle = C.fg;
        g.fillText(AI.fmt(V[s], 3), x, Math.min(top, base) - 6 < v.pad.t + 10 ? v.pad.t + 12 : Math.min(top, base) - 6);
        g.fillStyle = C.muted; g.fillText(s === GOAL ? 'goal' : s === PIT ? 'pit' : 'cell ' + s, x, v.pad.t + v.ph + 16);
        if (!terminal) {
          const left = q(s, -1, V, gam(), cost()), right = q(s, 1, V, gam(), cost());
          const d = right >= left ? 1 : -1;
          const y = v.Y(-1.12);
          AI.arrow(g, x, y, x + d * cw * 0.3, y, C.purple, 2);
        }
      }
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText(`sweep ${sweeps}: each cell takes the best of  (reward − c) + γ · neighbour value`, v.pad.l + 4, v.pad.t + 12);
      out.textContent =
        `sweep ${sweeps}   γ = ${AI.fmt(gam(), 2)}   cost per move c = ${AI.fmt(cost(), 2)}\n` +
        `V = [${V.map((x) => AI.fmt(x, 3)).join(', ')}]\n` +
        (conv ? 'converged: no value changed by more than 1e-6' : 'purple arrows show the greedy move; values spread outward from the goal one sweep at a time');
    };
    live(root, [cv], () => V.map((x) => x.toFixed(6)).join(',') + '|' + sweeps + '|' + prm, update, draw);
  };

  /* ================= 7.4 PPO clipped objective ================= */
  AI.labs['ppo-clip'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The clipped PPO objective',
      'illustrative · one sampled action: r = π_new(a|s) / π_old(a|s) is the probability ratio and A its advantage estimate',
      `<canvas style="height:270px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const R = slider(ctl, 'Probability ratio r', 0, 2, 0.01, 1.3, '', 2);
    const A = slider(ctl, 'Advantage A', -3, 3, 0.1, 2, '', 1);
    const EPS = slider(ctl, 'Clip ε', 0.05, 0.5, 0.01, 0.2, '', 2);
    const out = readout(ctl);
    const obj = (r, a, e) => Math.min(r * a, AI.clamp(r, 1 - e, 1 + e) * a);
    const draw = () => {
      const r = R(), a = A(), e = EPS(), C = AI.col(), g = cv.ctx, v = AI.view(cv, 0, 2, -6, 6);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      g.globalAlpha = 0.12; g.fillStyle = C.green; g.fillRect(v.X(1 - e), v.pad.t, v.X(1 + e) - v.X(1 - e), v.ph); g.globalAlpha = 1;
      curve(g, v, (x) => x * a, C.muted, 1.5, [6, 5]);
      curve(g, v, (x) => obj(x, a, e), C.blue, 3);
      const y = obj(r, a, e);
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(r), v.Y(AI.clamp(y, -6, 6)), 6, 0, 6.2832); g.fill();
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('dashed: unclipped r·A', v.pad.l + 8, v.pad.t + 14);
      g.fillStyle = C.blue; g.fillText('solid: L = min(r·A, clip(r, 1−ε, 1+ε)·A)', v.pad.l + 8, v.pad.t + 28);
      g.fillStyle = C.green; g.fillText('green band: the clip range [1−ε, 1+ε]', v.pad.l + 8, v.pad.t + 42);
      const clipped = (a > 0 && r > 1 + e) || (a < 0 && r < 1 - e);
      out.textContent =
        `r·A            = ${AI.fmt(r, 2)} × ${AI.fmt(a, 2)} = ${AI.fmt(r * a, 3)}\n` +
        `clip(r)·A      = ${AI.fmt(AI.clamp(r, 1 - e, 1 + e), 3)} × ${AI.fmt(a, 2)} = ${AI.fmt(AI.clamp(r, 1 - e, 1 + e) * a, 3)}\n` +
        `L = min(…)     = ${AI.fmt(y, 3)}\n` +
        (clipped ? 'gradient with respect to r: 0 (the clip is active, so this sample no longer pushes the policy)'
                 : 'gradient with respect to r: A = ' + AI.fmt(a, 2) + ' (the unclipped term is the minimum)');
    };
    live(root, [cv], () => R() + '|' + A() + '|' + EPS(), () => {}, draw);
  };

  /* ================= 7.4 REINFORCE on a bandit ================= */
  AI.labs['bandit-pg'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'REINFORCE on a three-armed bandit',
      'illustrative · three slot machines with unknown mean rewards; the policy is a softmax over three logits θ, learned from sampled rewards',
      `<canvas class="bars" style="height:190px"></canvas><canvas class="hist" style="height:150px;margin-top:10px"></canvas>`, '');
    const cB = AI.canvas(AI.$('canvas.bars', view)), cH = AI.canvas(AI.$('canvas.hist', view));
    const mu = [slider(ctl, 'Mean reward arm 1 μ₁', 0, 3, 0.1, 1.0, '', 1), slider(ctl, 'Mean reward arm 2 μ₂', 0, 3, 0.1, 2.0, '', 1), slider(ctl, 'Mean reward arm 3 μ₃', 0, 3, 0.1, 0.5, '', 1)];
    const sig = slider(ctl, 'Reward noise σ', 0, 1, 0.05, 0.5, '', 2);
    const lr = slider(ctl, 'Learning rate α', 0.01, 1, 0.01, 0.2, '', 2);
    const lab = document.createElement('label');
    lab.innerHTML = '<span>Running-mean baseline b</span><input type="checkbox" checked>';
    ctl.appendChild(lab);
    const useB = () => lab.querySelector('input').checked;
    const out = readout(ctl);
    const softmax = (t) => { const m = Math.max(...t), e = t.map((x) => Math.exp(x - m)), z = e[0] + e[1] + e[2]; return e.map((x) => x / z); };
    let th = [0, 0, 0], rnd = AI.rng(7), nb = 0, rbar = 0, hist = [], steps = 0;
    const reset = () => { th = [0, 0, 0]; rnd = AI.rng(7); nb = 0; rbar = 0; hist = [softmax(th)]; steps = 0; };
    reset();
    const T = transport(ctl, reset);
    const tk = AI.ticker();
    const stepOnce = () => {
      const p = softmax(th);
      let u = rnd(), a = 0, c = p[0];
      while (u > c && a < 2) { a++; c += p[a]; }
      const m = [mu[0](), mu[1](), mu[2]()];
      const R = m[a] + sig() * AI.randn(rnd);
      const b = useB() ? rbar : 0;
      for (let k = 0; k < 3; k++) th[k] += lr() * (R - b) * ((k === a ? 1 : 0) - p[k]);
      nb++; rbar += (R - rbar) / nb;
      steps++;
      hist.push(softmax(th)); if (hist.length > 400) hist.shift();
    };
    const update = (dt) => {
      let n = T.playing ? tk(dt, 2 * T.speed()) : 0;
      n += T.pending; T.pending = 0;
      for (let i = 0; i < n; i++) stepOnce();
    };
    const draw = () => {
      const C = AI.col(), p = softmax(th), cols = [C.blue, C.green, C.orange];
      let g = cB.ctx, v = AI.view(cB, 0, 3, 0, 1, { l: 40, r: 12, t: 12, b: 30 });
      g.clearRect(0, 0, cB.w, cB.h); AI.axes(g, v, { noGrid: true });
      for (let i = 0; i < 3; i++) {
        const x = i + 0.5, w = 0.4;
        g.fillStyle = cols[i]; g.globalAlpha = 0.75;
        g.fillRect(v.X(x - w / 2), v.Y(p[i]), v.X(x + w / 2) - v.X(x - w / 2), v.Y(0) - v.Y(p[i]));
        g.globalAlpha = 1; g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'center';
        g.fillText('π(arm ' + (i + 1) + ') = ' + AI.fmt(p[i], 3), v.X(x), v.Y(p[i]) - 6);
      }
      g = cH.ctx; v = AI.view(cH, 0, Math.max(10, hist.length - 1), 0, 1, { l: 40, r: 12, t: 12, b: 26 });
      g.clearRect(0, 0, cH.w, cH.h); AI.axes(g, v);
      g.font = '11px ' + MONO; g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText('probability of each arm over the last 400 steps', v.pad.l + 6, v.pad.t + 12);
      for (let i = 0; i < 3; i++) {
        g.strokeStyle = cols[i]; g.lineWidth = 2; g.beginPath();
        hist.forEach((h, j) => { const px = v.X(j), py = v.Y(h[i]); if (j) g.lineTo(px, py); else g.moveTo(px, py); });
        g.stroke();
      }
      out.textContent =
        `steps ${steps}   baseline ${useB() ? 'on' : 'off'}   running-mean reward ${AI.fmt(rbar, 3)}\n` +
        `θ = (${th.map((x) => AI.fmt(x, 3)).join(', ')})   π = (${p.map((x) => AI.fmt(x, 3)).join(', ')})\n` +
        `update: θ_k += α (R − b)(1[k = a] − π_k)`;
    };
    live(root, [cB, cH], () => steps + '|' + useB(), update, draw);
  };

  /* ================= 7.5 domain randomisation ================= */
  AI.labs['sim-random'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Domain randomisation on a one-dimensional reaching task',
      'illustrative · each row is one simulated world; the controller is fixed (tuned for 1.0 kg) and the target is 1 m away',
      `<canvas style="height:330px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const wid = slider(ctl, 'Mass randomisation ±', 0, 0.8, 0.02, 0.4, 'kg', 2);
    const mReal = slider(ctl, 'Real robot mass m', 0.5, 2.5, 0.05, 1.0, 'kg', 2);
    const out = readout(ctl);
    const NW = 40, TT = 1.5, DT = 0.01, KP = 16, KD = 6, UMAX = 6, TGT = 1;
    const rnd = AI.rng(11), U = Array.from({ length: NW }, () => [rnd(), rnd()]);
    const sim = (m, c) => {
      const xs = [0]; let x = 0, v = 0, mx = 0;
      for (let t = 0; t < TT / DT; t++) {
        const u = AI.clamp(KP * (TGT - x) - KD * v, -UMAX, UMAX);
        v += ((u - c * v) / m) * DT; x += v * DT; mx = Math.max(mx, x); xs.push(x);
      }
      return { xs, x, ok: Math.abs(x - TGT) < 0.05 && mx < 1.1 };
    };
    let tA = 0, worlds = [], real = null, wKey = '';
    const refresh = () => {
      const w = wid(), m0 = mReal(), key = w + '|' + m0;
      if (key === wKey) return;
      wKey = key;
      worlds = U.map(([u1, u2]) => { const m = 1 + w * (2 * u1 - 1), c = 0.2 + 0.8 * u2; return { m, c, ...sim(m, c) }; });
      real = { m: m0, ...sim(m0, 0.5) };
    };
    const T = transport(ctl, () => { tA = 0; T.setPlaying(true); });
    const update = (dt) => {
      refresh();
      if (T.playing) tA += dt * 0.5 * T.speed();
      if (tA > TT + 0.8) tA = 0;
      if (T.pending) { T.pending = 0; tA = Math.min(TT, tA + 0.1); }
    };
    const draw = () => {
      refresh();
      const C = AI.col(), g = cv.ctx;
      const v = AI.view(cv, -0.05, 1.35, 0, NW + 1.6, { l: 12, r: 12, t: 22, b: 26 });
      g.clearRect(0, 0, cv.w, cv.h);
      g.globalAlpha = 0.12; g.fillStyle = C.green; g.fillRect(v.X(0.95), v.pad.t, v.X(1.05) - v.X(0.95), v.ph); g.globalAlpha = 1;
      g.strokeStyle = C.green; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(1), v.pad.t); g.lineTo(v.X(1), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      g.font = '11px ' + MONO; g.textAlign = 'center'; g.fillStyle = C.muted;
      for (let x = 0; x <= 1.3; x += 0.2) g.fillText(AI.fmt(x, 1) + ' m', v.X(x), v.pad.t + v.ph + 16);
      g.textAlign = 'left'; g.fillText('green band: within 0.05 m of the 1 m target', v.pad.l + 4, v.pad.t - 8);
      const pos = (s) => s.xs[Math.min(s.xs.length - 1, Math.round(tA / DT))];
      worlds.forEach((s, i) => {
        const y = v.Y(NW - i + 0.5), px = v.X(pos(s));
        g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(v.X(0), y); g.lineTo(v.X(1.3), y); g.stroke();
        g.fillStyle = s.ok ? C.green : C.red; g.beginPath(); g.arc(px, y, 4.5, 0, 6.2832); g.fill();
      });
      const yR = v.Y(0.5);
      g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(0), yR); g.lineTo(v.X(1.3), yR); g.stroke();
      g.fillStyle = C.blue; g.beginPath(); g.arc(v.X(pos(real)), yR, 7, 0, 6.2832); g.fill();
      g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText('real robot', v.X(1.31) - 2, yR + 4);
      const k = worlds.filter((s) => s.ok).length;
      out.textContent =
        `simulated worlds that succeed: ${k} of ${NW}   (mass ${AI.fmt(1 - wid(), 2)}…${AI.fmt(1 + wid(), 2)} kg, damping 0.2…1.0 N·s/m)\n` +
        `real robot  m = ${AI.fmt(real.m, 2)} kg, damping 0.5 N·s/m: x(1.5 s) = ${AI.fmt(real.x, 3)} m → ${real.ok ? 'success' : 'fails'}\n` +
        `success: |x(1.5 s) − 1 m| < 0.05 m and overshoot < 0.1 m.  Controller PD: Kp = ${KP} N/m, Kd = ${KD} N·s/m, limit ±${UMAX} N`;
    };
    live(root, [cv], () => wid() + '|' + mReal() + '|' + Math.round(tA / 0.02) + '|' + wKey, update, draw);
  };

  /* ================= 7.6 reward design and hazards ================= */
  AI.labs['reward-routes'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Proxy reward versus true reward',
      'illustrative · a 9 × 5 grid: start on the left, goal on the right; the fast route crosses two hazard cells, the safe route takes a detour',
      `<canvas style="height:270px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const harm = slider(ctl, 'Harm per hazard cell h (true reward only)', 0, 5, 0.1, 3, '', 1);
    const budget = slider(ctl, 'Hazard budget (most hazard cells allowed)', 0, 2, 1, 2, '', 0);
    const mode = document.createElement('label');
    mode.innerHTML = '<span>Reward the robot was trained on</span><select><option value="proxy">proxy: +20 at goal, −1 per move</option><option value="true">true: proxy minus h per hazard cell</option></select>';
    ctl.appendChild(mode);
    const sel = mode.querySelector('select');
    const out = readout(ctl);
    const GOAL_R = 20, STEP_C = 1;
    const ROUTES = [
      { name: 'fast', color: 'orange', cells: [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2]] },
      { name: 'safe', color: 'green', cells: [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [8, 1], [8, 2]] },
    ];
    const HAZ = new Set(['3,2', '5,2']);
    const info = () => ROUTES.map((r) => {
      const moves = r.cells.length - 1, hz = r.cells.filter(([x, y]) => HAZ.has(x + ',' + y)).length;
      const proxy = GOAL_R - STEP_C * moves, trueR = proxy - harm() * hz;
      return { ...r, moves, hz, proxy, trueR, ok: hz <= budget() };
    });
    const choose = (inf) => {
      const key = sel.value === 'true' ? 'trueR' : 'proxy';
      const feas = inf.filter((r) => r.ok);
      if (!feas.length) return null;
      return feas.reduce((a, b) => (b[key] > a[key] ? b : a)).name;
    };
    let prog = 0, lastChoice = '';
    const T = transport(ctl, () => { prog = 0; T.setPlaying(true); });
    sel.addEventListener('change', () => (prog = 0));
    const update = (dt) => {
      const ch = choose(info());
      if (ch !== lastChoice) { lastChoice = ch; prog = 0; }
      if (T.playing && ch) {
        prog += dt * T.speed();
        const n = ROUTES.find((r) => r.name === ch).cells.length - 1;
        if (prog > n + 1.2) prog = 0;
      }
      if (T.pending) { T.pending = 0; prog += 1; }
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, inf = info(), ch = choose(inf);
      const cell = Math.min(cv.w / 10.5, cv.h / 6.5), ox = (cv.w - 9 * cell) / 2, oy = (cv.h - 5 * cell) / 2;
      const cx = (x) => ox + (x + 0.5) * cell, cy = (y) => oy + (y + 0.5) * cell;
      g.clearRect(0, 0, cv.w, cv.h);
      for (let x = 0; x < 9; x++) for (let y = 0; y < 5; y++) {
        const hz = HAZ.has(x + ',' + y);
        g.fillStyle = hz ? C.red : C.bg2; g.globalAlpha = hz ? 0.35 : 1;
        g.fillRect(ox + x * cell + 1, oy + y * cell + 1, cell - 2, cell - 2);
        g.globalAlpha = 1;
      }
      g.font = '11px ' + MONO; g.textAlign = 'center'; g.textBaseline = 'middle';
      inf.forEach((r) => {
        const col = C[r.color], chosen = r.name === ch;
        g.strokeStyle = col; g.lineWidth = chosen ? 4 : 1.5; g.globalAlpha = chosen ? 0.95 : 0.35;
        g.beginPath(); r.cells.forEach(([x, y], i) => (i ? g.lineTo(cx(x), cy(y)) : g.moveTo(cx(x), cy(y)))); g.stroke(); g.globalAlpha = 1;
      });
      g.fillStyle = C.red; g.fillText('!', cx(3), cy(2) - cell * 0.3); g.fillText('!', cx(5), cy(2) - cell * 0.3);
      g.fillStyle = C.fg; g.fillText('S', cx(0), cy(2)); g.fillText('G', cx(8), cy(2));
      if (ch) {
        const r = ROUTES.find((q) => q.name === ch), i = Math.min(r.cells.length - 1, Math.floor(prog));
        const f = Math.min(1, prog - Math.floor(prog)), a = r.cells[i], b = r.cells[Math.min(r.cells.length - 1, i + 1)];
        const px = cx(a[0]) + (cx(b[0]) - cx(a[0])) * f, py = cy(a[1]) + (cy(b[1]) - cy(a[1])) * f;
        g.fillStyle = C.blue; g.beginPath(); g.arc(px, py, cell * 0.26, 0, 6.2832); g.fill();
      }
      g.textBaseline = 'alphabetic';
      const lines = inf.map((r) => `${r.name.padEnd(5)} moves ${r.moves}, hazards ${r.hz}: proxy ${AI.fmt(r.proxy, 2)}, true ${AI.fmt(r.trueR, 2)}${r.ok ? '' : '  (over budget)'}${r.name === ch ? '  ← chosen' : ''}`);
      out.textContent = lines.join('\n') + '\n' +
        (ch ? `the robot follows the ${ch} route (hazard cells are red)` : 'no route satisfies the hazard budget: the robot stays put');
    };
    live(root, [cv], () => harm() + '|' + budget() + '|' + sel.value + '|' + Math.floor(prog * 4), update, draw);
  };
})();
