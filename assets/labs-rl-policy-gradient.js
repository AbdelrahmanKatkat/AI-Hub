/* AI Learning Hub — labs for Reinforcement Learning Module 6 (policy gradients and actor–critic).
 * All labs are illustrative toys with fixed seeds, so runs are reproducible.
 *
 * pg-softmax      <div data-lab="pg-softmax"></div>      softmax policy whose logits are formulas of a state feature x
 * pg-gauss        <div data-lab="pg-gauss"></div>        Gaussian policy over a continuous action, with score values
 * pg-estimate     <div data-lab="pg-estimate"></div>     score-function estimates converging to the exact policy gradient
 * pg-reinforce    <div data-lab="pg-reinforce"></div>    REINFORCE runs on a three-armed bandit (no baseline)
 * pg-baseline     <div data-lab="pg-baseline"></div>     a baseline keeps the mean gradient and shrinks its spread
 * pg-actor-critic <div data-lab="pg-actor-critic"></div> one-step actor–critic episodes with TD error
 * pg-ppo          <div data-lab="pg-ppo"></div>          the clipped PPO ratio over K gradient steps on one batch
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const SUB = ['₁', '₂', '₃'];

  /* ---------- shared helpers ---------- */
  // slider returns a getter; getter.set(v) moves it programmatically
  const slider = (ctl, name, min, max, step, val, unit, dec = 2) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = AI.fmt(+inp.value, dec) + (unit ? ' ' + unit : ''));
    inp.oninput = upd; upd(); ctl.appendChild(l);
    const g = () => +inp.value;
    g.set = (x) => { inp.value = x; upd(); };
    return g;
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const readout = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; d.style.whiteSpace = 'pre-line'; d.style.fontFamily = MONO; d.style.fontSize = '12px'; ctl.appendChild(d); return d; };

  // expression field: a formula of the named variables, validated with the shared safe parser
  const exprField = (ctl, label, def, allowed) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span><input type="text" spellcheck="false" value="${def}"><small style="color:var(--red)"></small>`;
    ctl.appendChild(l);
    const inp = l.querySelector('input'), err = l.querySelector('small');
    const H = { fn: () => 0, text: () => inp.value };
    AI.exprInput(inp, err, { allowed, onChange: (fn) => (H.fn = fn) }).run();
    return H;
  };

  // transport: play/pause, step (one whole step when paused), reset, speed
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 3, 0.05, speed, '×', 2);
    pp.onclick = () => { T.playing = !T.playing; pp.textContent = T.playing ? 'Pause' : 'Play'; };
    st.onclick = () => { T.playing = false; pp.textContent = 'Play'; T.pending += 1; };
    rs.onclick = () => onReset();
    return T;
  };

  // advance an animation by `step(dt)`, redraw only when the state key or canvas size or theme changed
  const drive = (root, canvases, key, step, draw) => {
    let last = null;
    AI.loop(root, (dt) => {
      step(dt);
      const k = key() + '|' + canvases.map((c) => c.w + 'x' + c.h).join(',') + '|' + document.documentElement.dataset.theme;
      if (k === last) return;
      last = k; draw();
    });
  };

  // redraw only when state changes (no animation): same key test
  const reactive = (root, canvases, key, draw) => drive(root, canvases, key, () => {}, draw);

  const softmax = (th, tau = 1) => {
    const m = Math.max(...th);
    const z = th.map((t) => Math.exp((t - m) / tau));
    const s = z.reduce((a, b) => a + b, 0);
    return z.map((v) => v / s);
  };

  /* ================= 6.1 softmax policy ================= */
  AI.labs['pg-softmax'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A softmax policy over three actions',
      'illustrative · each logit θₐ(x) is a formula of one state feature x; π(a|x) = e^(θₐ/τ) / Σ_b e^(θ_b/τ)',
      `<canvas class="bars" style="height:170px"></canvas><canvas class="curves" style="height:200px"></canvas>`, '');
    const cB = AI.canvas(AI.$('canvas.bars', view)), cC = AI.canvas(AI.$('canvas.curves', view));
    const X = slider(ctl, 'State feature x', -2, 2, 0.01, 0.5, '', 2);
    const TAU = slider(ctl, 'Temperature τ', 0.2, 3, 0.05, 1, '', 2);
    const F = [exprField(ctl, 'Logit θ₁(x) =', '2 - x', ['x']), exprField(ctl, 'Logit θ₂(x) =', 'x', ['x']), exprField(ctl, 'Logit θ₃(x) =', '0', ['x'])];
    const row = rowOf(ctl);
    const acts = [button(row, 'Sample action 1'), button(row, 'Sample action 2'), button(row, 'Sample action 3')];
    let pick = 0;
    acts.forEach((b, i) => (b.onclick = () => { pick = i; }));
    const out = readout(ctl);
    const C = () => AI.col();
    const logits = (x) => F.map((h) => h.fn({ x }));
    const draw = () => {
      const x = X(), tau = TAU(), C0 = C(), g = cB.ctx;
      const th = logits(x), p = softmax(th, tau);
      g.clearRect(0, 0, cB.w, cB.h);
      const colors = [C0.blue, C0.green, C0.orange];
      const bw = cB.w / 3;
      p.forEach((v, i) => {
        const h = v * (cB.h - 40), x0 = i * bw + bw * 0.2, w = bw * 0.6;
        g.fillStyle = colors[i]; g.globalAlpha = 0.85; g.fillRect(x0, cB.h - 24 - h, w, h); g.globalAlpha = 1;
        g.fillStyle = C0.fg; g.font = '12px ' + MONO; g.textAlign = 'center';
        g.fillText(`π(${i + 1}|x) = ${AI.fmt(v, 3)}`, x0 + w / 2, cB.h - 24 - h - 6);
        g.fillStyle = C0.muted; g.fillText(`action ${i + 1}  (θ = ${AI.fmt(th[i], 2)})`, x0 + w / 2, cB.h - 8);
      });
      // curves over x
      const v = AI.view(cC, -2, 2, 0, 1);
      const cg = cC.ctx;
      cg.clearRect(0, 0, cC.w, cC.h); AI.axes(cg, v);
      [colors[0], colors[1], colors[2]].forEach((col, i) => {
        cg.strokeStyle = col; cg.lineWidth = 2.2; cg.beginPath();
        for (let k = 0; k <= 200; k++) {
          const xx = -2 + (4 * k) / 200, pp = softmax(logits(xx), tau)[i];
          const px = v.X(xx), py = v.Y(pp);
          k ? cg.lineTo(px, py) : cg.moveTo(px, py);
        }
        cg.stroke();
      });
      cg.strokeStyle = C0.muted; cg.setLineDash([4, 4]); cg.beginPath(); cg.moveTo(v.X(x), v.pad.t); cg.lineTo(v.X(x), v.pad.t + v.ph); cg.stroke(); cg.setLineDash([]);
      cg.font = '11px ' + MONO; cg.textAlign = 'left';
      cg.fillStyle = C0.muted; cg.fillText('π(a|x) against x; dashed line = current x', v.pad.l + 8, v.pad.t + 14);
      const a = pick, gr = [0, 1, 2].map((b) => ((a === b ? 1 : 0) - p[b]) / tau);
      out.textContent =
        `logits θ(x)   = (${th.map((t) => AI.fmt(t, 2)).join(', ')})\n` +
        `π(·|x)        = (${p.map((t) => AI.fmt(t, 3)).join(', ')})   sum = ${AI.fmt(p.reduce((s, t) => s + t, 0), 3)}\n` +
        `sampled action: ${a + 1}\n` +
        `∇θ log π(${a + 1}|x) = (${gr.map((t) => AI.fmt(t, 3)).join(', ')})  = (e_${a + 1} − π) / τ\n` +
        `Positive entries raise that logit when the return is positive.`;
    };
    drive(root, [cB, cC], () => [X(), TAU(), F.map((h) => h.text()).join('|'), pick].join('|'), () => {}, draw);
  };

  /* ================= 6.1 Gaussian policy ================= */
  AI.labs['pg-gauss'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A Gaussian policy over a continuous action',
      'illustrative · a ~ N(μ(x), σ²); the dots are 40 fixed draws (seed 7) that move as μ and σ change',
      `<canvas style="height:250px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const X = slider(ctl, 'State feature x', -2, 2, 0.01, 0, '', 2);
    const SIG = slider(ctl, 'Standard deviation σ', 0.1, 2, 0.01, 0.5, '', 2);
    const A = slider(ctl, 'Chosen action a', -3, 3, 0.05, 1.2, '', 2);
    const M = exprField(ctl, 'Mean μ(x) =', '0.8 + 0.5*x', ['x']);
    const out = readout(ctl);
    const R = AI.rng(7);
    const Z = Array.from({ length: 40 }, () => AI.randn(R));
    const dens = (a, mu, s) => Math.exp(-((a - mu) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
    const draw = () => {
      const x = X(), s = SIG(), a = A(), mu = M.fn({ x }), C = AI.col(), g = cv.ctx;
      const peak = dens(mu, mu, s);
      const v = AI.view(cv, -3, 3, 0, peak * 1.25);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      g.fillStyle = C.blue; g.globalAlpha = 0.18; g.beginPath(); g.moveTo(v.X(-3), v.Y(0));
      for (let k = 0; k <= 240; k++) { const aa = -3 + (6 * k) / 240; g.lineTo(v.X(aa), v.Y(dens(aa, mu, s))); }
      g.lineTo(v.X(3), v.Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      for (let k = 0; k <= 240; k++) { const aa = -3 + (6 * k) / 240, px = v.X(aa), py = v.Y(dens(aa, mu, s)); k ? g.lineTo(px, py) : g.moveTo(px, py); }
      g.stroke();
      g.fillStyle = C.green;
      Z.forEach((z, i) => { const aa = mu + s * z; if (aa > -3 && aa < 3) { g.beginPath(); g.arc(v.X(aa), v.Y(0) - 6 - (i % 5) * 4, 3, 0, 6.2832); g.fill(); } });
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(a), v.Y(0)); g.lineTo(v.X(a), v.Y(dens(a, mu, s))); g.stroke();
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(a), v.Y(dens(a, mu, s)), 5, 0, 6.2832); g.fill();
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('blue: density N(μ, σ²); green dots: 40 fixed samples; orange: chosen action', v.pad.l + 8, v.pad.t + 14);
      const z = (a - mu) / s, logp = -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * z * z;
      out.textContent =
        `μ(x) = ${AI.fmt(mu, 3)}    σ = ${AI.fmt(s, 3)}    a = ${AI.fmt(a, 3)}\n` +
        `z = (a − μ)/σ = ${AI.fmt(z, 3)}\n` +
        `log π(a|x) = −½ log 2π − log σ − z²/2 = ${AI.fmt(logp, 4)}\n` +
        `∂ log π / ∂μ = (a − μ)/σ² = ${AI.fmt((a - mu) / (s * s), 3)}\n` +
        `∂ log π / ∂σ = ((a − μ)² − σ²)/σ³ = ${AI.fmt(((a - mu) ** 2 - s * s) / s ** 3, 3)}`;
    };
    drive(root, [cv], () => [X(), SIG(), A(), M.text()].join('|'), () => {}, draw);
  };

  /* ================= 6.2 score-function estimate ================= */
  AI.labs['pg-estimate'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The score-function estimate converges to the exact gradient',
      'illustrative · a one-state, two-action bandit; π₁ = σ(θ); each sample g = Q(a)·∇θ log π(a) and the running mean is plotted',
      `<canvas style="height:280px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const Q1 = slider(ctl, 'Action 1 value Q(1)', -2, 4, 0.1, 1, '', 1);
    const Q2 = slider(ctl, 'Action 2 value Q(2)', -2, 4, 0.1, 3, '', 1);
    const TH = slider(ctl, 'Logit θ (π₁ = σ(θ))', -3, 3, 0.05, 0, '', 2);
    const N = 400, R = AI.rng(11), U = Array.from({ length: N }, () => R());
    const T = transport(ctl, () => { t = 0; }, 1);
    const out = readout(ctl);
    let t = 0;
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const samples = () => {
      const th = TH(), p1 = sig(th), q1 = Q1(), q2 = Q2(), g = [];
      for (let i = 0; i < N; i++) {
        const a = U[i] < p1 ? 0 : 1, q = a === 0 ? q1 : q2;
        g.push(q * ((a === 0 ? 1 : 0) - p1));
      }
      return { p1, q1, q2, g, exact: p1 * (1 - p1) * (q1 - q2) };
    };
    const step = (dt) => {
      if (T.playing) t += dt * T.speed() * 6;
      if (T.pending) { t = Math.floor(t) + T.pending; T.pending = 0; }
      t = Math.min(t, N);
    };
    const draw = () => {
      const S = samples(), n = Math.floor(t), C = AI.col(), g = cv.ctx;
      let run = 0; const means = [];
      for (let i = 0; i < n; i++) { run += S.g[i]; means.push(run / (i + 1)); }
      const span = Math.max(1, Math.abs(S.q1), Math.abs(S.q2)) * 1.2;
      const v = AI.view(cv, 0, N, -span, span);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      g.strokeStyle = C.green; g.setLineDash([6, 5]); g.lineWidth = 1.8; g.beginPath(); g.moveTo(v.X(0), v.Y(S.exact)); g.lineTo(v.X(N), v.Y(S.exact)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      means.forEach((m, i) => (i ? g.lineTo(v.X(i + 1), v.Y(m)) : g.moveTo(v.X(1), v.Y(m))));
      g.stroke();
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('blue: running mean of the sampled estimates; green dashed: exact ∂J/∂θ = π₁(1−π₁)(Q(1)−Q(2))', v.pad.l + 8, v.pad.t + 14);
      g.fillText(`n = ${n} samples (x-axis: sample count)`, v.pad.l + 8, v.pad.t + 28);
      const last = n ? S.g[n - 1] : NaN;
      out.textContent =
        `π₁ = σ(θ) = ${AI.fmt(S.p1, 4)}     π₂ = ${AI.fmt(1 - S.p1, 4)}\n` +
        `exact ∂J/∂θ = π₁π₂(Q₁ − Q₂) = ${AI.fmt(S.exact, 4)}\n` +
        `running mean after ${n} samples = ${n ? AI.fmt(means[n - 1], 4) : '–'}\n` +
        `last sample g = ${n ? AI.fmt(last, 3) : '–'}   (g = Q(a)(1[a=1] − π₁))\n` +
        `Each sample is a single action and a single reward; only the mean is the gradient.`;
    };
    drive(root, [cv], () => [Q1(), Q2(), TH(), Math.floor(t)].join('|'), step, draw);
  };

  /* ================= 6.3 REINFORCE ================= */
  AI.labs['pg-reinforce'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'REINFORCE on a three-armed bandit',
      'illustrative · no baseline; 12 runs with fixed seeds start from θ = (0, 0, 0); the reward is μₐ + σ·noise; each line is P(best arm) per episode',
      `<canvas style="height:280px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const ALPHA = slider(ctl, 'Learning rate α', 0.01, 0.5, 0.01, 0.1, '', 2);
    const SIG = slider(ctl, 'Reward noise σ', 0, 1.5, 0.05, 0.5, '', 2);
    const MU = [slider(ctl, 'Mean reward μ₁', -1, 3, 0.1, 1.0, '', 1), slider(ctl, 'Mean reward μ₂', -1, 3, 0.1, 2.0, '', 1), slider(ctl, 'Mean reward μ₃', -1, 3, 0.1, 0.5, '', 1)];
    const NRUN = 12, NEP = 200;
    let t = 0, runs = null, runKey = '';
    const T = transport(ctl, () => { t = 0; }, 1);
    const out = readout(ctl);
    const simulate = () => {
      const a = ALPHA(), s = SIG(), mu = MU.map((f) => f());
      const best = mu.indexOf(Math.max(...mu));
      runs = [];
      for (let r = 0; r < NRUN; r++) {
        const R = AI.rng(100 + r * 7), th = [0, 0, 0], hist = [];
        for (let e = 0; e < NEP; e++) {
          const p = softmax(th);
          hist.push(p[best]);
          const u = R(); let acc = 0, act = 2;
          for (let i = 0; i < 3; i++) { acc += p[i]; if (u < acc) { act = i; break; } }
          const rew = mu[act] + s * AI.randn(R);
          for (let i = 0; i < 3; i++) th[i] += a * rew * ((i === act ? 1 : 0) - p[i]);
        }
        hist.push(softmax(th)[best]);
        runs.push(hist);
      }
      runKey = [a, s, ...mu].join('|');
      return best;
    };
    let best = simulate();
    const step = (dt) => {
      if (T.playing) t += dt * T.speed() * 6;
      if (T.pending) { t = Math.floor(t) + T.pending; T.pending = 0; }
      t = Math.min(t, NEP);
      const key = [ALPHA(), SIG(), ...MU.map((f) => f())].join('|');
      if (key !== runKey) best = simulate();
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, e = Math.floor(t);
      const v = AI.view(cv, 0, NEP, 0, 1);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v, { noY: false });
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(0), v.Y(1 / 3)); g.lineTo(v.X(NEP), v.Y(1 / 3)); g.stroke(); g.setLineDash([]);
      g.lineWidth = 1.2; g.strokeStyle = C.orange; g.globalAlpha = 0.55;
      runs.forEach((h) => { g.beginPath(); for (let i = 0; i <= e && i < h.length; i++) (i ? g.lineTo(v.X(i), v.Y(h[i])) : g.moveTo(v.X(i), v.Y(h[i]))); g.stroke(); });
      g.globalAlpha = 1;
      const mean = [];
      for (let i = 0; i <= e && i < NEP; i++) mean.push(runs.reduce((s, h) => s + h[i], 0) / NRUN);
      g.strokeStyle = C.blue; g.lineWidth = 3; g.beginPath();
      mean.forEach((m, i) => (i ? g.lineTo(v.X(i), v.Y(m)) : g.moveTo(v.X(i), v.Y(m))));
      g.stroke();
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('orange: 12 runs · blue: mean · dashed: uniform policy 1/3', v.pad.l + 8, v.pad.t + 14);
      const vals = runs.map((h) => h[Math.min(e, NEP)]);
      const mu = MU.map((f) => f());
      out.textContent =
        `episode ${e} of ${NEP} · best arm = ${best + 1} (μ = ${AI.fmt(mu[best], 1)})\n` +
        `P(best arm) across runs: min ${AI.fmt(Math.min(...vals), 3)}, max ${AI.fmt(Math.max(...vals), 3)}, mean ${AI.fmt(mean[mean.length - 1], 3)}\n` +
        `Update per episode: θ ← θ + α · r · (e_a − π). With all rewards positive, even a bad arm is reinforced when sampled.`;
    };
    drive(root, [cv], () => [Math.floor(t), runKey].join('|'), step, draw);
  };

  /* ================= 6.3 / 6.4 baseline and variance ================= */
  AI.labs['pg-baseline'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A baseline keeps the mean and shrinks the spread',
      'illustrative · three arms with logits θ; N = 1000 fixed draws (seed 5); the sample estimate for component θ₁ is (r − b)(1[a=1] − π₁)',
      `<canvas style="height:260px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas', view));
    const TH = [slider(ctl, 'Logit θ₁', -2, 2, 0.05, 0.4, '', 2), slider(ctl, 'Logit θ₂', -2, 2, 0.05, 0, '', 2), slider(ctl, 'Logit θ₃', -2, 2, 0.05, -0.4, '', 2)];
    const MU = [slider(ctl, 'Mean reward μ₁', 0, 3, 0.1, 1.0, '', 1), slider(ctl, 'Mean reward μ₂', 0, 3, 0.1, 2.0, '', 1), slider(ctl, 'Mean reward μ₃', 0, 3, 0.1, 0.5, '', 1)];
    const SIG = slider(ctl, 'Reward noise σ', 0, 2, 0.05, 1, '', 2);
    const B = slider(ctl, 'Baseline b', 0, 3, 0.01, 0, '', 2);
    const row = rowOf(ctl);
    const setV = button(row, 'Set b = V(θ) (exact)');
    const N = 1000, R = AI.rng(5), U = Array.from({ length: N }, () => R()), Zs = Array.from({ length: N }, () => AI.randn(R));
    const out = readout(ctl);
    setV.onclick = () => { const p = softmax(TH.map((f) => f())), V = p.reduce((s, q, i) => s + q * MU[i](), 0); B.set(+V.toFixed(2)); };
    const estimates = (b) => {
      const th = TH.map((f) => f()), p = softmax(th), mu = MU.map((f) => f()), s = SIG(), gs = [];
      for (let i = 0; i < N; i++) {
        let acc = 0, a = 2;
        for (let k = 0; k < 3; k++) { acc += p[k]; if (U[i] < acc) { a = k; break; } }
        const r = mu[a] + s * Zs[i];
        gs.push((r - b) * ((a === 0 ? 1 : 0) - p[0]));
      }
      return gs;
    };
    const exact = () => {
      const p = softmax(TH.map((f) => f())), mu = MU.map((f) => f());
      return p.reduce((s, q, a) => s + q * mu[a] * ((a === 0 ? 1 : 0) - p[0]), 0);
    };
    const V = () => { const p = softmax(TH.map((f) => f())); return p.reduce((s, q, i) => s + q * MU[i](), 0); };
    const mom = (xs) => { const m = xs.reduce((a, b) => a + b, 0) / xs.length; const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length; return { m, v }; };
    const draw = () => {
      const C = AI.col(), g = cv.ctx;
      const b = B(), e = exact();
      const g0 = estimates(0), gb = estimates(b);
      const lo = Math.min(...g0, ...gb, e) - 0.2, hi = Math.max(...g0, ...gb, e) + 0.2;
      const bins = 40, binsOf = (xs) => { const h = Array(bins).fill(0); xs.forEach((x) => { const k = AI.clamp(Math.floor(((x - lo) / (hi - lo)) * bins), 0, bins - 1); h[k]++; }); return h; };
      const h0 = binsOf(g0), hb = binsOf(gb), hm = Math.max(...h0, ...hb);
      const v = AI.view(cv, lo, hi, 0, hm * 1.15);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v, { noY: true });
      const bw = (hi - lo) / bins;
      g.globalAlpha = 0.45; g.fillStyle = C.orange;
      h0.forEach((c, k) => g.fillRect(v.X(lo + k * bw), v.Y(c), v.X(lo + (k + 1) * bw) - v.X(lo + k * bw) - 1, v.Y(0) - v.Y(c)));
      g.fillStyle = C.blue;
      hb.forEach((c, k) => g.fillRect(v.X(lo + k * bw), v.Y(c), v.X(lo + (k + 1) * bw) - v.X(lo + k * bw) - 1, v.Y(0) - v.Y(c)));
      g.globalAlpha = 1;
      g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v.X(e), v.Y(0)); g.lineTo(v.X(e), v.Y(hm * 1.1)); g.stroke(); g.setLineDash([]);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('orange: estimates with b = 0 · blue: estimates with the current b · green dashed: exact ∂J/∂θ₁', v.pad.l + 8, v.pad.t + 14);
      const s0 = mom(g0), sb = mom(gb), Vv = V();
      out.textContent =
        `V(θ) = Σ π μ = ${AI.fmt(Vv, 3)}    advantage of arm 1: μ₁ − V = ${AI.fmt(MU[0]() - Vv, 3)}\n` +
        `exact ∂J/∂θ₁ = ${AI.fmt(e, 4)}\n` +
        `b = 0:    mean ${AI.fmt(s0.m, 4)}   variance ${AI.fmt(s0.v, 4)}\n` +
        `b = ${AI.fmt(b, 2)}:  mean ${AI.fmt(sb.m, 4)}   variance ${AI.fmt(sb.v, 4)}\n` +
        `The mean moves only by sampling noise for any b; the variance depends on b.`;
    };
    reactive(root, [cv], () => [...TH, ...MU, SIG, B].map((f) => f()).join('|'), draw);
  };

  /* ================= 6.5 one-step actor–critic ================= */
  AI.labs['pg-actor-critic'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One-step actor–critic on a two-step decision',
      'illustrative · from S choose Up (to M, reward 0) or Down (to Done, reward r_D); at M one step gives reward +1 and ends. Actor: logits at S. Critic: V(S), V(M).',
      `<canvas class="map" style="height:200px"></canvas><canvas class="plot" style="height:210px;margin-top:10px"></canvas>`, '');
    const cM = AI.canvas(AI.$('canvas.map', view)), cP = AI.canvas(AI.$('canvas.plot', view));
    const GAM = slider(ctl, 'Discount γ', 0, 1, 0.01, 0.9, '', 2);
    const RD = slider(ctl, 'Reward for Down r_D', 0, 1.5, 0.05, 0.3, '', 2);
    const AC = slider(ctl, 'Critic step α_c', 0.01, 1, 0.01, 0.3, '', 2);
    const AA = slider(ctl, 'Actor step α_a', 0, 1, 0.01, 0.3, '', 2);
    const T = transport(ctl, () => resetAll(), 1);
    const out = readout(ctl);
    let model, hist, events, ei, clock, epi, endSnap;
    const resetAll = () => {
      model = { z: [0, 0], VS: 0, VM: 0 };
      hist = []; events = []; ei = 0; clock = 0; epi = 0; endSnap = null;
    };
    resetAll();
    const piUp = () => softmax(model.z)[0];
    // play one episode now and store its events (each event holds the state after that step)
    const runEpisode = () => {
      const R = AI.rng(1000 + epi), g = GAM(), rD = RD(), ac = AC(), aa = AA();
      const pi = softmax(model.z);
      const a = R() < pi[0] ? 0 : 1;
      const ev = [];
      const push = (cap, hi, td) => ev.push({ cap, hi, td, snap: { z: model.z.slice(), VS: model.VS, VM: model.VM, pi: softmax(model.z)[0] } });
      push(`① Sample at S with π(Up|S) = ${AI.fmt(pi[0], 3)}: the agent takes ${a === 0 ? 'Up' : 'Down'}.`, a === 0 ? 'Up' : 'Down', null);
      if (a === 1) {
        const d = rD + g * 0 - model.VS;
        push(`② Down: reward r_D = ${AI.fmt(rD, 2)}, then Done. TD error δ = r + γV(Done) − V(S) = ${AI.fmt(rD, 2)} + 0 − ${AI.fmt(model.VS, 3)} = ${AI.fmt(d, 3)}`, 'Down', d);
        model.VS += ac * d;
        const pS = softmax(model.z);
        model.z = model.z.map((zz, k) => zz + aa * d * ((k === 1 ? 1 : 0) - pS[k]));
        push(`③ Critic: V(S) ← V(S) + α_c δ = ${AI.fmt(model.VS, 3)}.  Actor: logits at S move by α_a·δ·∇log π(Down|S).`, 'Down', null);
      } else {
        const d = 0 + g * model.VM - model.VS;
        push(`② Up: reward 0, next state M. TD error δ_S = 0 + γV(M) − V(S) = 0 + ${AI.fmt(g, 2)}·${AI.fmt(model.VM, 3)} − ${AI.fmt(model.VS, 3)} = ${AI.fmt(d, 3)}`, 'Up', d);
        model.VS += ac * d;
        const pS = softmax(model.z);
        model.z = model.z.map((zz, k) => zz + aa * d * ((k === 0 ? 1 : 0) - pS[k]));
        push(`③ Critic at S: V(S) ← ${AI.fmt(model.VS, 3)}.  Actor at S: logits move by α_a·δ·∇log π(Up|S).`, 'Up', null);
        const dM = 1 + 0 - model.VM;
        push(`④ At M: reward +1, then Done. TD error δ_M = 1 + 0 − V(M) = ${AI.fmt(dM, 3)}`, 'M', dM);
        model.VM += ac * dM;
        push(`⑤ Critic at M: V(M) ← ${AI.fmt(model.VM, 3)}. M has one action, so no actor update.`, 'M', null);
      }
      endSnap = { pi: piUp(), VS: model.VS, VM: model.VM };
      epi += 1;
      events = ev; ei = 0;
    };
    runEpisode();
    const step = (dt) => {
      if (T.playing) clock += dt * T.speed();
      if (T.pending) { clock += 1.5; T.pending = 0; }
      while (clock >= 1.5) {
        clock -= 1.5;
        if (ei < events.length - 1) ei += 1;
        else if (epi < 300) {
          // the finished episode joins the plot only after its last event has been shown
          if (endSnap) hist.push(endSnap);
          runEpisode();
        }
      }
    };
    const diagram = (g, C, cur, snap) => {
      const W = cM.w, H = cM.h;
      g.clearRect(0, 0, W, H);
      const nodes = { S: [W * 0.16, H * 0.5], M: [W * 0.5, H * 0.5], Done: [W * 0.84, H * 0.5] };
      const lab = { S: `S  V = ${AI.fmt(snap.VS, 3)}`, M: `M  V = ${AI.fmt(snap.VM, 3)}`, Done: 'Done' };
      const edge = (a, b, color, width, text, ty) => {
        g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width;
        AI.arrow(g, a[0] + 26, a[1], b[0] - 26, b[1], color, width);
        g.font = '12px ' + MONO; g.textAlign = 'center'; g.fillText(text, (a[0] + b[0]) / 2, ty);
      };
      const hiUp = cur === 'Up' || cur === 'M', hiDown = cur === 'Down';
      edge(nodes.S, nodes.M, hiUp ? C.orange : C.muted, hiUp ? 3 : 1.5, `Up, r = 0  π = ${AI.fmt(snap.pi, 2)}`, H * 0.3);
      // curved down edge from S to Done
      g.strokeStyle = hiDown ? C.orange : C.muted; g.lineWidth = hiDown ? 3 : 1.5;
      g.beginPath(); g.moveTo(nodes.S[0], nodes.S[1] + 22); g.quadraticCurveTo(W * 0.5, H * 0.95, nodes.Done[0] - 26, nodes.Done[1] + 14); g.stroke();
      g.fillStyle = hiDown ? C.orange : C.muted; g.font = '12px ' + MONO; g.textAlign = 'center';
      g.fillText(`Down, r = ${AI.fmt(RD(), 2)}  π = ${AI.fmt(1 - snap.pi, 2)}`, W * 0.5, H * 0.9);
      edge(nodes.M, nodes.Done, cur === 'M' ? C.orange : C.muted, cur === 'M' ? 3 : 1.5, 'r = +1', H * 0.3 + 0);
      [['S', C.blue], ['M', C.green], ['Done', C.muted]].forEach(([k, col]) => {
        const [x, y] = nodes[k];
        g.fillStyle = col; g.globalAlpha = 0.25; g.beginPath(); g.arc(x, y, 24, 0, 6.2832); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 24, 0, 6.2832); g.stroke();
        g.fillStyle = C.fg; g.font = '12px ' + MONO; g.textAlign = 'center'; g.fillText(k === 'Done' ? 'Done' : k, x, y + 4);
      });
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.fillText(lab.S, nodes.S[0], nodes.S[1] + 44);
      g.fillText(lab.M, nodes.M[0], nodes.M[1] + 44);
    };
    const plot = (g, C) => {
      const W = cP.w, H = cP.h, n = Math.max(40, hist.length);
      const v = AI.view(cP, 0, n, 0, 1.1);
      g.clearRect(0, 0, W, H); AI.axes(g, v);
      const pTarget = GAM() > RD() ? 1 : 0;
      g.setLineDash([5, 4]); g.lineWidth = 1.2; g.strokeStyle = C.blue;
      g.beginPath(); g.moveTo(v.X(0), v.Y(pTarget)); g.lineTo(v.X(n), v.Y(pTarget)); g.stroke(); g.setLineDash([]);
      const line = (key, col) => {
        g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath();
        hist.forEach((h, i) => { const val = AI.clamp(h[key], 0, 1.1); i ? g.lineTo(v.X(i + 1), v.Y(val)) : g.moveTo(v.X(1), v.Y(val)); });
        g.stroke();
      };
      line('pi', C.blue); line('VS', C.green); line('VM', C.orange);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('blue: π(Up|S) · green: V(S) · orange: V(M) (clipped at 1.1) · dashed: greedy target for π', v.pad.l + 8, v.pad.t + 14);
    };
    const draw = () => {
      const C = AI.col();
      const ev = events[Math.min(ei, events.length - 1)];
      diagram(cM.ctx, C, ev.hi, ev.snap);
      plot(cP.ctx, C);
      out.textContent =
        `episode ${epi}   event ${ei + 1} of ${events.length}\n${ev.cap}\n` +
        `after this event: V(S) = ${AI.fmt(ev.snap.VS, 3)}   V(M) = ${AI.fmt(ev.snap.VM, 3)}   π(Up|S) = ${AI.fmt(ev.snap.pi, 3)}\n` +
        `Targets: V*(S) = max(γ·1, r_D) = ${AI.fmt(Math.max(GAM(), RD()), 2)},  V*(M) = 1`;
    };
    drive(root, [cM, cP], () => [epi, ei, GAM(), RD(), AC(), AA()].join('|'), step, draw);
  };

  /* ================= 6.6 PPO clipped ratio over K steps ================= */
  AI.labs['pg-ppo'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The clipped ratio over K gradient steps on one batch',
      'illustrative · one state, two actions, π_old(a) = 0.5; one sampled action with advantage A; logit θ moves by lr·g per step; the same batch is reused',
      `<canvas class="p" style="height:200px"></canvas><canvas class="r" style="height:160px;margin-top:10px"></canvas>`, '');
    const cP = AI.canvas(AI.$('canvas.p', view)), cR = AI.canvas(AI.$('canvas.r', view));
    const A = slider(ctl, 'Advantage A', -2, 2, 0.1, 2, '', 1);
    const EPS = slider(ctl, 'Clip ε', 0.05, 0.5, 0.01, 0.2, '', 2);
    const LR = slider(ctl, 'Step size lr', 0.05, 1, 0.01, 0.25, '', 2);
    const K = slider(ctl, 'Gradient steps K', 4, 30, 1, 12, '', 0);
    const T = transport(ctl, () => { t = 0; }, 1);
    const out = readout(ctl);
    const POLD = 0.5, MAXK = 30;
    const sig = (x) => 1 / (1 + Math.exp(-x));
    let t = 0;
    const traj = (clipOn, a, eps, lr) => {
      let th = 0; const p = [POLD], r = [1], act = [false];
      for (let s = 0; s < MAXK; s++) {
        const pc = sig(th), ratio = pc / POLD;
        const active = clipOn && ((a > 0 && ratio > 1 + eps) || (a < 0 && ratio < 1 - eps));
        const grad = active ? 0 : a * ratio * (1 - pc);
        th += lr * grad;
        p.push(sig(th)); r.push(sig(th) / POLD); act.push(active);
      }
      return { p, r, act };
    };
    const step = (dt) => {
      if (T.playing) t += dt * T.speed() * 2;
      if (T.pending) { t = Math.floor(t) + T.pending; T.pending = 0; }
      t = Math.min(t, K());
    };
    const draw = () => {
      const a = A(), eps = EPS(), lr = LR(), k = Math.min(Math.floor(t), K()), kmax = K();
      const C = AI.col();
      const on = traj(true, a, eps, lr), off = traj(false, a, eps, lr);
      // p plot
      let g = cP.ctx; const v = AI.view(cP, 0, kmax, 0, 1);
      g.clearRect(0, 0, cP.w, cP.h); AI.axes(g, v);
      g.setLineDash([5, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(v.X(0), v.Y(POLD)); g.lineTo(v.X(kmax), v.Y(POLD)); g.stroke(); g.setLineDash([]);
      const path = (arr, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); for (let i = 0; i <= k; i++) (i ? g.lineTo(v.X(i), v.Y(arr[i])) : g.moveTo(v.X(0), v.Y(arr[0]))); g.stroke(); };
      path(off.p, C.muted, 2); path(on.p, C.blue, 3);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('π(a|s) per step: blue = clipped objective, grey = unclipped (same lr)', v.pad.l + 8, v.pad.t + 14);
      // ratio plot
      g = cR.ctx; const w = AI.view(cR, 0, kmax, 0, 2.2);
      g.clearRect(0, 0, cR.w, cR.h); AI.axes(g, w);
      g.globalAlpha = 0.15; g.fillStyle = C.green; g.fillRect(w.pad.l, w.Y(1 + eps), w.pw, w.Y(1 - eps) - w.Y(1 + eps)); g.globalAlpha = 1;
      const rpath = (arr, col, wd) => { g.strokeStyle = col; g.lineWidth = wd; g.beginPath(); for (let i = 0; i <= k; i++) (i ? g.lineTo(w.X(i), w.Y(arr[i])) : g.moveTo(w.X(0), w.Y(arr[0]))); g.stroke(); };
      rpath(off.r, C.muted, 2); rpath(on.r, C.blue, 3);
      g.font = '11px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText(`ratio r = π/π_old; green band = [1−ε, 1+ε] = [${AI.fmt(1 - eps, 2)}, ${AI.fmt(1 + eps, 2)}]`, w.pad.l + 8, w.pad.t + 14);
      // act[i] is true when the update that produced step i was blocked by the clip
      const firstClip = on.act.findIndex((x, i) => x && i >= 1 && i <= kmax);
      const rr = on.r[k], ro = off.r[k];
      out.textContent =
        `step ${k} of K = ${kmax}\n` +
        `clipped run:   π(a) = ${AI.fmt(on.p[k], 4)}   r = ${AI.fmt(rr, 3)}   ${k >= 1 && on.act[k] ? 'last update blocked by the clip (gradient 0)' : 'gradient flows'}\n` +
        `unclipped run: π(a) = ${AI.fmt(off.p[k], 4)}   r = ${AI.fmt(ro, 3)}\n` +
        (firstClip >= 1 ? `the clip first blocks the update that produces step ${firstClip}.` : 'the clip never becomes active within K steps.') + '\n' +
        `L^CLIP for this sample = min(r·A, clip(r)·A) = ${AI.fmt(Math.min(rr * a, AI.clamp(rr, 1 - eps, 1 + eps) * a), 4)}`;
    };
    drive(root, [cP, cR], () => [Math.floor(t * 4) / 4, A(), EPS(), LR(), K()].join('|'), step, draw);
  };
})();
