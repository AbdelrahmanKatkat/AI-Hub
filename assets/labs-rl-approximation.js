/* Reinforcement Learning · Module 5 (Function approximation & deep Q-learning) labs.
 * All are small illustrative toys in plain JS (a linear model, tile features, a single weight). None is a large experiment.
 * rl-linear-fit    <div data-lab="rl-linear-fit"></div>    semi-gradient TD(0) with linear features on a 10-state random walk
 * rl-tile-coding   <div data-lab="rl-tile-coding"></div>   tile coding fitting an editable target f(s) on [0, 1]
 * rl-divergence    <div data-lab="rl-divergence"></div>    expected bootstrapped updates on a two-state toy: shared weight versus tabular
 * rl-replay        <div data-lab="rl-replay"></div>        a replay buffer filled by a random-walk agent, with uniform or no replay
 * rl-target-sync   <div data-lab="rl-target-sync"></div>   one weight chasing a frozen copy that is synced every K updates
 * rl-max-bias      <div data-lab="rl-max-bias"></div>      the upward bias of a max over noisy estimates, and the double estimator
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
    const get = () => +inp.value;
    get.input = inp;
    return get;
  };
  const select = (ctl, name, opts, val) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name}</span><select>${opts.map(([v, t]) => `<option value="${v}"${v === val ? ' selected' : ''}>${t}</option>`).join('')}</select>`;
    const sel = l.querySelector('select');
    ctl.appendChild(l);
    const get = () => sel.value;
    get.input = sel;
    return get;
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const readout = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; };
  const onInput = (get, fn) => get.input.addEventListener('input', fn);

  /* transport: play/pause, step (one pending step for the lab loop), reset, speed */
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

  const plotFn = (g, v, fn, color, w = 2.5) => {
    g.save(); g.strokeStyle = color; g.lineWidth = w; g.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = v.xmin + ((v.xmax - v.xmin) * i) / 200;
      const y = fn(x);
      const px = v.X(x), py = v.Y(AI.clamp(y, v.ymin, v.ymax));
      if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
    }
    g.stroke(); g.restore();
  };
  const text = (g, color, s, x, y) => { g.fillStyle = color; g.font = '11px ' + MONO; g.textAlign = 'left'; g.fillText(s, x, y); };
  const dot = (g, x, y, r, color) => { g.fillStyle = color; g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill(); };

  /* ================= 5.1 linear value fit (semi-gradient TD(0)) ================= */
  AI.labs['rl-linear-fit'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Linear value fit on a 10-state random walk',
      'illustrative · semi-gradient TD(0), γ = 1, start in state 6 · left exit pays 0, right exit pays +1 · true values v*(s) = s/11',
      `<canvas class="lf" style="height:260px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.lf', view));
    const N = 10, START = 6;
    const TRUE = (s) => s / 11;
    const X = {
      constant: () => [1],
      line: (s) => [1, s / 10],
      pairs: (s) => { const x = [0, 0, 0, 0, 0]; x[Math.ceil(s / 2) - 1] = 1; return x; },
      tabular: (s) => { const x = new Array(N).fill(0); x[s - 1] = 1; return x; },
    };
    const T = transport(ctl, () => reset(), 1);
    const feat = select(ctl, 'Features', [
      ['constant', 'constant: one weight (the mean)'], ['line', 'line: [1, s/10]'],
      ['pairs', 'pairs: one weight per pair of states'], ['tabular', 'tabular: one weight per state'],
    ], 'line');
    const alpha = slider(ctl, 'Step size α', 0.01, 0.3, 0.01, 0.05, '', 2);
    const out = readout(ctl);
    const tick = AI.ticker();
    const R0 = 7;
    let R, w, s, ep, steps, lastFeat, lastDelta;
    const dot2 = (a, b) => a.reduce((p, q, i) => p + q * b[i], 0);
    const V = (st) => (st === 0 || st === N + 1 ? 0 : dot2(w, X[feat()](st)));
    const reset = () => {
      R = AI.rng(R0); lastFeat = feat();
      w = new Array(X[lastFeat](1).length).fill(0);
      s = START; ep = 0; steps = 0; lastDelta = 0;
    };
    onInput(feat, () => reset());
    reset();
    const stepOnce = () => {
      const a = R() < 0.5 ? -1 : 1;
      const s2 = s + a;
      const terminal = s2 === 0 || s2 === N + 1;
      const reward = s2 === N + 1 ? 1 : 0;
      const target = reward + (terminal ? 0 : V(s2));
      const x = X[feat()](s);
      const delta = target - V(s);
      const al = alpha();
      w = w.map((wi, i) => wi + al * delta * x[i]);
      lastDelta = delta;
      steps++;
      if (terminal) { ep++; s = START; } else s = s2;
    };
    const update = (dt) => {
      if (feat() !== lastFeat) reset();
      let n = T.playing ? tick(dt, 6 * T.speed()) : 0;
      if (T.pending) { n += 1; T.pending = 0; }
      for (let i = 0; i < n; i++) stepOnce();
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx;
      const v = AI.view(cv, 0.5, N + 0.5, -0.05, 1.05);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      const bw = (v.pw / N) * 0.6, y0 = v.Y(0);
      let se = 0;
      for (let st = 1; st <= N; st++) {
        const vh = V(st);
        se += (vh - TRUE(st)) ** 2;
        const x = v.X(st), y = v.Y(AI.clamp(vh, v.ymin, v.ymax));
        g.globalAlpha = 0.55; g.fillStyle = C.blue; g.fillRect(x - bw / 2, Math.min(y, y0), bw, Math.abs(y0 - y)); g.globalAlpha = 1;
        dot(g, x, v.Y(TRUE(st)), 4, C.green);
      }
      dot(g, v.X(s), v.Y(-0.03), 4.5, C.orange);
      text(g, C.muted, 'v(s) for states 1…10 (x-axis = state s)', v.pad.l + 8, v.pad.t + 14);
      text(g, C.blue, 'blue bars: learned v̂(s)', v.pad.l + 8, v.pad.t + 28);
      text(g, C.green, 'green dots: true v*(s) = s/11', v.pad.l + 8, v.pad.t + 42);
      text(g, C.orange, 'orange: state of the walker now', v.pad.l + 8, v.pad.t + 56);
      const rmse = Math.sqrt(se / N);
      out.textContent =
        `features: ${feat()} · weights w = [${w.map((z) => AI.fmt(z, 3)).join(', ')}]\n` +
        `episodes ${ep} · transitions ${steps} · current state ${s} · last TD error δ = ${AI.fmt(lastDelta, 3)}\n` +
        `RMSE of v̂ against v* over the 10 states = ${AI.fmt(rmse, 3)}\n` +
        'line and tabular can match v* exactly (v* is linear in s); constant and pairs cannot.';
    };
    live(root, [cv], () => feat() + '|' + steps + '|' + alpha() + '|' + ep, update, draw);
  };

  /* ================= 5.2 tile coding ================= */
  AI.labs['rl-tile-coding'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Tile coding: many shifted staircases fitting one curve',
      'illustrative · one state s in [0, 1] · m tilings are shifted copies of one grid of width 1/n · each update moves one active tile per tiling',
      `<canvas class="tcf" style="height:230px"></canvas><canvas class="tct" style="height:150px;margin-top:10px"></canvas>`, '');
    const cF = AI.canvas(AI.$('canvas.tcf', view)), cT = AI.canvas(AI.$('canvas.tct', view));
    const T = transport(ctl, () => reset(), 1);
    const exRow = document.createElement('div');
    exRow.innerHTML = '<label><span>Target f(s)</span></label>';
    const exInp = document.createElement('input');
    exInp.type = 'text'; exInp.value = 'sin(2*pi*s)'; exInp.spellcheck = false;
    exRow.querySelector('label').appendChild(exInp);
    ctl.appendChild(exRow);
    const exErr = readout(ctl);
    const mT = slider(ctl, 'Tilings m', 1, 6, 1, 3, '', 0);
    const nT = slider(ctl, 'Tiles per tiling n', 2, 16, 1, 4, '', 0);
    const alpha = slider(ctl, 'Step size α', 0.05, 1, 0.05, 0.5, '', 2);
    const probe = slider(ctl, 'Probe state s', 0, 1, 0.01, 0.3, '', 2);
    const out = readout(ctl);
    const tick = AI.ticker();
    let f = null, W = [], R, updates = 0;
    const tileOf = (st, j, m, n) => { const wt = 1 / n, off = (j * wt) / m; return Math.floor((st + off) / wt); };
    const vhat = (st) => { let sum = 0; for (let j = 0; j < W.length; j++) sum += W[j][tileOf(st, j, W.length, nT())]; return sum; };
    const reset = () => {
      const m = mT(), n = nT();
      W = Array.from({ length: m }, () => new Array(n + 1).fill(0));
      R = AI.rng(11); updates = 0;
    };
    AI.exprInput(exInp, exErr, {
      allowed: ['s'],
      check: (fn) => { for (let i = 0; i <= 10; i++) if (!Number.isFinite(fn({ s: i / 10 }))) throw new Error('f must be finite on [0, 1]'); },
      onChange: (fn) => { f = fn; reset(); },
    }).run();
    onInput(mT, reset); onInput(nT, reset);
    reset();
    const update = (dt) => {
      let n = T.playing ? tick(dt, 20 * T.speed()) : 0;
      if (T.pending) { n += 1; T.pending = 0; }
      const m = W.length, al = alpha();
      for (let i = 0; i < n; i++) {
        const st = R();
        const err = f({ s: st }) - vhat(st);
        for (let j = 0; j < m; j++) W[j][tileOf(st, j, m, nT())] += (al / m) * err;
        updates++;
      }
    };
    const draw = () => {
      const C = AI.col(), g = cF.ctx;
      const v = AI.view(cF, 0, 1, -1.4, 1.4);
      g.clearRect(0, 0, cF.w, cF.h); AI.axes(g, v);
      plotFn(g, v, (x) => f({ s: x }), C.green, 2.5);
      plotFn(g, v, (x) => vhat(x), C.orange, 2.5);
      const ps = probe();
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(ps), v.pad.t); g.lineTo(v.X(ps), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      text(g, C.green, 'target f(s)', v.pad.l + 8, v.pad.t + 14);
      text(g, C.orange, 'tile-coded v̂(s), current weights', v.pad.l + 8, v.pad.t + 28);
      let se = 0; const K = 200;
      for (let i = 0; i <= K; i++) { const x = i / K; const d = vhat(x) - f({ s: x }); se += d * d; }
      const rms = Math.sqrt(se / (K + 1));
      const m = W.length, n = nT(), wt = 1 / n;
      const rowH = Math.min(22, (cT.h - 8) / Math.max(m, 1));
      cT.ctx.clearRect(0, 0, cT.w, cT.h);
      const gt = cT.ctx;
      const x0 = 40, pw = cT.w - x0 - 12;
      for (let j = 0; j < m; j++) {
        const yj = 4 + j * rowH, off = (j * wt) / m, act = tileOf(ps, j, m, n);
        text(gt, C.muted, 'tiling ' + (j + 1), 4, yj + rowH * 0.7);
        for (let k = 0; k <= n; k++) {
          const a = AI.clamp(k * wt - off, 0, 1), b = AI.clamp((k + 1) * wt - off, 0, 1);
          if (b <= a) continue;
          const xa = x0 + a * pw, xb = x0 + b * pw;
          gt.globalAlpha = k === act ? 0.85 : 0.12; gt.fillStyle = k === act ? C.orange : C.blue;
          gt.fillRect(xa + 1, yj + 1, xb - xa - 2, rowH - 3); gt.globalAlpha = 1;
        }
      }
      text(gt, C.muted, 'probe s = ' + AI.fmt(ps, 2) + ': one lit tile per tiling', x0, cT.h - 4);
      const act = [];
      for (let j = 0; j < m; j++) act.push(`tiling ${j + 1} tile ${tileOf(ps, j, m, n)} (w = ${AI.fmt(W[j][tileOf(ps, j, m, n)], 3)})`);
      out.textContent =
        `m = ${m} tilings, n = ${n} tiles each, ${m * (n + 1)} weights in total\n` +
        `sampled updates ${updates} · RMS error on 201 grid points = ${AI.fmt(rms, 3)}\n` +
        `v̂(probe) = ${AI.fmt(vhat(ps), 3)} from ${act.join('; ')}`;
    };
    live(root, [cF, cT], () => updates + '|' + mT() + '|' + nT() + '|' + Math.round(probe() * 100) + '|' + f, update, draw);
  };

  /* ================= 5.3 divergence: shared weight versus tabular ================= */
  AI.labs['rl-divergence'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bootstrapping with a shared weight: does the error grow?',
      'illustrative · two-state toy, reward 0 · s1 → s2 and s2 → s2 always · expected semi-gradient TD update · plotted: log₁₀ of the size of the weight',
      `<canvas class="dv" style="height:280px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.dv', view));
    const T = transport(ctl, () => reset(), 1);
    const gamma = slider(ctl, 'Discount γ', 0.5, 0.99, 0.01, 0.9, '', 2);
    const alpha = slider(ctl, 'Step size α', 0.05, 1, 0.05, 0.5, '', 2);
    const d1s = slider(ctl, 'State weight d(s1)', 0, 1, 0.01, 0.5, '', 2);
    const out = readout(ctl);
    const tick = AI.ticker();
    const MAXIT = 120;
    let w, t1, t2, hist, it;
    const reset = () => {
      w = 1; t1 = 1; t2 = 2; it = 0;
      hist = [[Math.log10(Math.abs(w)), Math.log10(Math.max(Math.abs(t1), Math.abs(t2)))]];
    };
    [gamma, alpha, d1s].forEach((g) => onInput(g, reset));
    reset();
    const iterate = () => {
      const g = gamma(), a = alpha(), d1 = d1s(), d2 = 1 - d1;
      // shared weight: phi(s1) = 1, phi(s2) = 2
      w = w + a * (d1 * (g * 2 * w - w) * 1 + d2 * (g * 2 * w - 2 * w) * 2);
      // tabular: one weight per state
      const n1 = t1 + a * d1 * (g * t2 - t1);
      const n2 = t2 + a * d2 * (g * t2 - t2);
      t1 = n1; t2 = n2;
      it++;
      hist.push([Math.log10(Math.abs(w) + 1e-12), Math.log10(Math.max(Math.abs(t1), Math.abs(t2)) + 1e-12)]);
    };
    const update = (dt) => {
      let n = T.playing ? tick(dt, 4 * T.speed()) : 0;
      if (T.pending) { n += 1; T.pending = 0; }
      for (let i = 0; i < n && it < MAXIT; i++) iterate();
      if (it >= MAXIT && T.playing) T.setPlaying(false);
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx;
      const v = AI.view(cv, 0, MAXIT, -4, 6);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      const drawLine = (idx, color) => {
        g.save(); g.strokeStyle = color; g.lineWidth = 2.5; g.beginPath();
        hist.forEach((h, i) => {
          const px = v.X(i), py = v.Y(AI.clamp(h[idx], v.ymin, v.ymax));
          if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
        });
        g.stroke(); g.restore();
      };
      drawLine(0, C.orange);
      drawLine(1, C.green);
      text(g, C.orange, 'shared weight w: log₁₀|w| (φ(s1)=1, φ(s2)=2)', v.pad.l + 8, v.pad.t + 14);
      text(g, C.green, 'tabular: log₁₀ of the largest |θ| (one weight per state)', v.pad.l + 8, v.pad.t + 28);
      text(g, C.muted, 'y-axis: each gridline is a factor of 10', v.pad.l + 8, v.pad.t + 42);
      const a = alpha(), gm = gamma(), d1 = d1s();
      const factor = 1 + a * (d1 * (2 * gm - 1) + (1 - d1) * (4 * gm - 4));
      const dstar = (4 - 4 * gm) / (3 - 2 * gm);
      out.textContent =
        `iteration ${it} of ${MAXIT} · w = ${AI.fmt(w, 4)} · θ = (${AI.fmt(t1, 4)}, ${AI.fmt(t2, 4)})\n` +
        `shared-weight growth factor per iteration = 1 + α[d1(2γ−1) + d2(4γ−4)] = ${AI.fmt(factor, 4)}\n` +
        `${Math.abs(factor) > 1 ? 'the shared weight grows: divergence' : 'the shared weight shrinks: stable here'}; ` +
        `threshold d(s1) = ${dstar > 0 && dstar <= 1 ? AI.fmt(dstar, 3) : 'outside [0, 1]'}`;
    };
    live(root, [cv], () => it + '|' + gamma() + '|' + alpha() + '|' + d1s(), update, draw);
  };

  /* ================= 5.4 experience replay ================= */
  AI.labs['rl-replay'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Experience replay: which transitions does the learner see?',
      'illustrative · random walk on states 0…10, start 5, reset to 5 at either end · one transition per step · the buffer keeps the newest transitions',
      `<canvas class="rb" style="height:120px"></canvas><canvas class="rh" style="height:170px;margin-top:10px"></canvas>`, '');
    const cB = AI.canvas(AI.$('canvas.rb', view)), cH = AI.canvas(AI.$('canvas.rh', view));
    const T = transport(ctl, () => reset(), 1);
    const cap = slider(ctl, 'Buffer capacity', 5, 200, 5, 40, '', 0);
    const B = slider(ctl, 'Batch size', 1, 16, 1, 4, '', 0);
    const mode = select(ctl, 'Learner sees', [['uniform', 'uniform replay (sample the buffer)'], ['latest', 'no replay (the last B transitions)']], 'uniform');
    const out = readout(ctl);
    const tick = AI.ticker();
    let R, buf, stepsN, s, ages, last;
    const reset = () => { R = AI.rng(21); buf = []; stepsN = 0; s = 5; ages = []; last = []; };
    [cap, B, mode].forEach((g) => onInput(g, reset));
    reset();
    const stepOnce = () => {
      const a = R() < 0.5 ? -1 : 1, s2 = s + a;
      buf.push({ t: stepsN, s, s2 });
      while (buf.length > cap()) buf.shift();
      stepsN++;
      s = s2 === 0 || s2 === 10 ? 5 : s2;
      const b = B();
      let picks;
      if (mode() === 'latest') picks = buf.slice(-b);
      else picks = Array.from({ length: b }, () => buf[Math.floor(R() * buf.length)]);
      last = picks;
      for (const p of picks) ages.push(stepsN - 1 - p.t);
      if (ages.length > 400) ages.splice(0, ages.length - 400);
    };
    const update = (dt) => {
      let n = T.playing ? tick(dt, 4 * T.speed()) : 0;
      if (T.pending) { n += 1; T.pending = 0; }
      for (let i = 0; i < n; i++) stepOnce();
    };
    const draw = () => {
      const C = AI.col();
      // buffer grid: 20 columns, one cell per slot, oldest at top-left
      const cols = 20, c = cap(), rows = Math.ceil(c / cols);
      const gB = cB.ctx; gB.clearRect(0, 0, cB.w, cB.h);
      const cw = cB.w / cols, ch = Math.min(18, cB.h / rows);
      for (let i = 0; i < c; i++) {
        const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
        const tr = buf[i];
        if (!tr) { gB.strokeStyle = C.border; gB.strokeRect(x + 2, y + 2, cw - 4, ch - 4); continue; }
        const age = buf.length - 1 - i;
        gB.globalAlpha = 0.25 + 0.7 * (1 - age / Math.max(1, c));
        gB.fillStyle = C.blue; gB.fillRect(x + 2, y + 2, cw - 4, ch - 4); gB.globalAlpha = 1;
        if (last.includes(tr)) { gB.strokeStyle = C.orange; gB.lineWidth = 2; gB.strokeRect(x + 2, y + 2, cw - 4, ch - 4); }
      }
      // histogram of sampled ages
      const gH = cH.ctx; gH.clearRect(0, 0, cH.w, cH.h);
      const v = AI.view(cH, 0, c, 0, 1);
      AI.axes(gH, v, { noY: true });
      const bins = 20, counts = new Array(bins).fill(0);
      for (const a of ages) counts[Math.min(bins - 1, Math.floor((a / c) * bins))]++;
      const mx = Math.max(1, ...counts);
      const bw = v.pw / bins;
      counts.forEach((k, i) => {
        const h = (k / mx) * v.ph;
        gH.fillStyle = C.orange; gH.globalAlpha = 0.8;
        gH.fillRect(v.pad.l + i * bw + 1, v.pad.t + v.ph - h, bw - 2, h); gH.globalAlpha = 1;
      });
      text(gH, C.muted, 'histogram of sampled ages (0 = newest transition), last 400 samples', v.pad.l + 6, v.pad.t + 14);
      const mean = ages.length ? ages.reduce((p, q) => p + q, 0) / ages.length : NaN;
      out.textContent =
        `steps ${stepsN} · buffer ${buf.length}/${c} · batch ${B()} · ${mode() === 'latest' ? 'no replay' : 'uniform replay'}\n` +
        `mean age of the last ${ages.length} sampled transitions = ${AI.fmt(mean, 3)} steps\n` +
        `uniform sampling from a full buffer of size N has mean age (N−1)/2 = ${AI.fmt((buf.length - 1) / 2, 3)} steps;\n` +
        `no replay with batch B has mean age (B−1)/2 = ${AI.fmt((B() - 1) / 2, 3)} steps and consecutive batches overlap in time.`;
    };
    live(root, [cB, cH], () => stepsN + '|' + cap() + '|' + B() + '|' + mode(), update, draw);
  };

  /* ================= 5.5 target network (scalar toy) ================= */
  AI.labs['rl-target-sync'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A frozen target: one weight chasing a copy synced every K updates',
      'illustrative · one weight, not a neural network · y = r + γ·c·w̄ with r = 1, γ = 0.9, c = 1 · w̄ is the frozen copy · fixed point w* = r/(1 − γc) = 10',
      `<canvas class="ts" style="height:260px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.ts', view));
    const T = transport(ctl, () => reset(), 1);
    const K = slider(ctl, 'Sync period K (updates)', 1, 50, 1, 10, '', 0);
    const alpha = slider(ctl, 'Step size α', 0.05, 0.5, 0.01, 0.2, '', 2);
    const out = readout(ctl);
    const tick = AI.ticker();
    const WSTAR = 10, TMAX = 120, GAMMA = 0.9, C_ = 1, RW = 1;
    let w, wbar, t, hist, y, hit;
    const reset = () => { w = 0; wbar = 0; t = 0; hist = [{ w, wbar }]; y = RW + GAMMA * C_ * wbar; hit = null; };
    [K, alpha].forEach((g) => onInput(g, reset));
    reset();
    const iterate = () => {
      const a = alpha();
      y = RW + GAMMA * C_ * wbar;
      w = w + a * (y - w);
      t++;
      if (t % K() === 0) wbar = w;
      hist.push({ w, wbar });
      if (hit === null && Math.abs(w - WSTAR) < 0.5) hit = t;
    };
    const update = (dt) => {
      let n = T.playing ? tick(dt, 4 * T.speed()) : 0;
      if (T.pending) { n += 1; T.pending = 0; }
      for (let i = 0; i < n && t < TMAX; i++) iterate();
      if (t >= TMAX && T.playing) T.setPlaying(false);
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx;
      const v = AI.view(cv, 0, TMAX, 0, 11.5);
      g.clearRect(0, 0, cv.w, cv.h); AI.axes(g, v);
      g.save(); g.strokeStyle = C.green; g.setLineDash([6, 5]); g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(v.pad.l, v.Y(WSTAR)); g.lineTo(v.pad.l + v.pw, v.Y(WSTAR)); g.stroke(); g.restore();
      g.save(); g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath();
      hist.forEach((h, i) => { const px = v.X(i), py = v.Y(h.wbar); if (i === 0) g.moveTo(px, py); else { g.lineTo(px, hist[i - 1] ? v.Y(hist[i - 1].wbar) : py); g.lineTo(px, py); } });
      g.stroke(); g.restore();
      g.save(); g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      hist.forEach((h, i) => { const px = v.X(i), py = v.Y(AI.clamp(h.w, 0, 11.5)); if (i === 0) g.moveTo(px, py); else g.lineTo(px, py); });
      g.stroke(); g.restore();
      text(g, C.blue, 'online weight w (learned every update)', v.pad.l + 8, v.pad.t + 14);
      text(g, C.orange, 'frozen copy w̄ (replaced every K updates)', v.pad.l + 8, v.pad.t + 28);
      text(g, C.green, 'fixed point w* = 10', v.pad.l + 8, v.pad.t + 42);
      out.textContent =
        `update ${t} of ${TMAX} · w = ${AI.fmt(w, 3)} · frozen w̄ = ${AI.fmt(wbar, 3)} · target y = ${AI.fmt(y, 3)}\n` +
        `distance to w* = ${AI.fmt(Math.abs(WSTAR - w), 3)}` +
        (hit === null ? ' · not yet within 0.5 of w*' : ` · first within 0.5 of w* at update ${hit}`) + '\n' +
        'Illustrative: the frozen copy only makes each block of updates regress toward a fixed number; it does not show DQN’s empirical benefit.';
    };
    live(root, [cv], () => t + '|' + K() + '|' + alpha(), update, draw);
  };

  /* ================= 5.6 max bias and double estimation ================= */
  AI.labs['rl-max-bias'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The max of noisy estimates is biased upward; double estimation removes it',
      'illustrative · every action has true value Q* = 0 · each trial draws two independent noisy estimate vectors (online and target) with Gaussian noise σ',
      `<canvas class="mb" style="height:260px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.mb', view));
    const T = transport(ctl, () => reset(), 1);
    const K = slider(ctl, 'Number of actions K', 2, 10, 1, 4, '', 0);
    const sig = slider(ctl, 'Noise σ', 0.1, 2, 0.1, 1, '', 1);
    const out = readout(ctl);
    const tick = AI.ticker();
    const NTR = 3000;
    const THEORY = { 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 6: 1.267, 7: 1.352, 8: 1.424, 9: 1.485, 10: 1.539 };
    let shownDQN, shownDbl, seen, dqn, dbl;
    const reset = () => {
      const k = K(), s = sig(), R = AI.rng(31);
      const g = () => AI.randn(R);
      dqn = new Float64Array(NTR); dbl = new Float64Array(NTR);
      for (let i = 0; i < NTR; i++) {
        const A = Array.from({ length: k }, () => s * g());
        const B = Array.from({ length: k }, () => s * g());
        let am = 0; for (let j = 1; j < k; j++) if (A[j] > A[am]) am = j;
        dqn[i] = Math.max(...B);
        dbl[i] = B[am];
      }
      seen = 0;
    };
    [K, sig].forEach((g) => onInput(g, reset));
    reset();
    const update = (dt) => {
      let n = T.playing ? tick(dt, 300 * T.speed()) : 0;
      if (T.pending) { n += 50; T.pending = 0; }
      seen = Math.min(NTR, seen + n);
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx;
      const s = sig(), lim = 3 * s;
      const v = AI.view(cv, -lim, lim, 0, 1);
      g.clearRect(0, 0, cv.w, cv.h);
      const bins = 40, bw = (2 * lim) / bins;
      const hD = new Array(bins).fill(0), hB = new Array(bins).fill(0);
      let sD = 0, sB = 0;
      for (let i = 0; i < seen; i++) {
        sD += dqn[i]; sB += dbl[i];
        const bd = Math.min(bins - 1, Math.max(0, Math.floor((dqn[i] + lim) / bw)));
        const bb = Math.min(bins - 1, Math.max(0, Math.floor((dbl[i] + lim) / bw)));
        hD[bd]++; hB[bb]++;
      }
      const mx = Math.max(1, ...hD, ...hB);
      AI.axes(g, v, { noY: true });
      const draw1 = (h, color) => {
        g.globalAlpha = 0.45; g.fillStyle = color;
        h.forEach((c, i) => { const hh = (c / mx) * v.ph; g.fillRect(v.X(-lim + i * bw) + 1, v.pad.t + v.ph - hh, v.pw / bins - 2, hh); });
        g.globalAlpha = 1;
      };
      draw1(hD, C.orange);
      draw1(hB, C.blue);
      const mD = seen ? sD / seen : NaN, mB = seen ? sB / seen : NaN;
      g.save(); g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(0), v.pad.t); g.lineTo(v.X(0), v.pad.t + v.ph); g.stroke(); g.restore();
      if (seen) {
        g.save(); g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(mD), v.pad.t); g.lineTo(v.X(mD), v.pad.t + v.ph); g.stroke(); g.restore();
        g.save(); g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(mB), v.pad.t); g.lineTo(v.X(mB), v.pad.t + v.ph); g.stroke(); g.restore();
      }
      text(g, C.orange, 'max over the target net (DQN-style): orange', v.pad.l + 8, v.pad.t + 14);
      text(g, C.blue, 'double: argmax from online net, value from target net: blue', v.pad.l + 8, v.pad.t + 28);
      text(g, C.muted, 'dashed line: true Q* = 0', v.pad.l + 8, v.pad.t + 42);
      const th = THEORY[K()] !== undefined ? AI.fmt(THEORY[K()] * s, 3) : 'n/a';
      out.textContent =
        `trials shown ${seen} of ${NTR} · K = ${K()} · σ = ${AI.fmt(s, 2)}\n` +
        `mean DQN-style estimate = ${AI.fmt(mD, 3)} (bias +${AI.fmt(mD, 3)}); theory for normal noise: E[max] = ${th}\n` +
        `mean double estimate = ${AI.fmt(mB, 3)} (expected 0: the value is picked by one net and read from the other)`;
    };
    live(root, [cv], () => seen + '|' + K() + '|' + sig(), update, draw);
  };
})();
