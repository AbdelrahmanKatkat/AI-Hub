/* Reinforcement learning · Module 2 (Dynamic programming) labs. All are illustrative gridworld toys; every number is computed in code.
 * Gridworld: 4×4 cells numbered 0–15 row by row, goals at cells 0 and 15, every move costs −1, a move into a wall stays put.
 * dp-bellman            <div data-lab="dp-bellman"></div>            two-state choice problem: the Bellman backup as a fixed-point map
 * dp-evaluation         <div data-lab="dp-evaluation"></div>         iterative policy evaluation, sweep by sweep, with a stopping tolerance
 * dp-improvement        <div data-lab="dp-improvement"></div>        action values in one cell, greedy improvement, then re-evaluation
 * dp-policy-iteration   <div data-lab="dp-policy-iteration"></div>   evaluation and improvement alternating until the arrows settle
 * dp-value-iteration    <div data-lab="dp-value-iteration"></div>    max-backup sweeps, greedy arrows and the stopping bound (slippery grid)
 * dp-gpi                <div data-lab="dp-gpi"></div>                generalised policy iteration: k evaluation sweeps per improvement
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const DR = [-1, 1, 0, 0], DC = [0, 0, -1, 1]; // actions: 0 up, 1 down, 2 left, 3 right
  const DX = [0, 0, -1, 1], DY = [-1, 1, 0, 0];
  const NAMES = ['up', 'down', 'left', 'right'];
  const ARW = ['↑', '↓', '←', '→'];

  /* ---------- model and Bellman operators (synchronous: every cell reads the previous sweep) ---------- */
  const isGoal = (s) => s === 0 || s === 15;
  const nxt = (s, a) => {
    const r = s >> 2, c = s & 3, nr = r + DR[a], nc = c + DC[a];
    return nr < 0 || nr > 3 || nc < 0 || nc > 3 ? s : nr * 4 + nc;
  };
  // intended move with probability 1 - slip; each of the other three moves with probability slip / 3
  const outs = (s, a, slip) => {
    if (!slip) return [[1, nxt(s, a)]];
    const o = [[1 - slip, nxt(s, a)]];
    for (let b = 0; b < 4; b++) if (b !== a) o.push([slip / 3, nxt(s, b)]);
    return o;
  };
  const qOf = (V, s, a, gm, slip) => {
    let q = 0;
    for (const [p, s2] of outs(s, a, slip)) q += p * (-1 + gm * V[s2]); // V of a goal is always 0
    return q;
  };
  const UNI = () => Array(16).fill(null); // null in a cell = uniform random over the four moves
  const greedy = (V, gm, slip) => {
    const pi = [];
    for (let s = 0; s < 16; s++) {
      if (isGoal(s)) { pi.push(-1); continue; }
      let best = -Infinity, ba = 0;
      for (let a = 0; a < 4; a++) { const q = qOf(V, s, a, gm, slip); if (q > best + 1e-9) { best = q; ba = a; } }
      pi.push(ba);
    }
    return pi;
  };
  const countChanged = (a, b) => { let n = 0; for (let s = 0; s < 16; s++) if (!isGoal(s) && a[s] !== b[s]) n++; return n; };
  const sweepPol = (V, pi, gm, slip) => {
    const NV = V.slice(); let d = 0;
    for (let s = 0; s < 16; s++) {
      if (isGoal(s)) continue;
      let v = 0;
      if (pi[s] === null) for (let a = 0; a < 4; a++) v += 0.25 * qOf(V, s, a, gm, slip);
      else v = qOf(V, s, pi[s], gm, slip);
      NV[s] = v; d = Math.max(d, Math.abs(v - V[s]));
    }
    return { V: NV, d };
  };
  const sweepOpt = (V, gm, slip) => {
    const NV = V.slice(); let d = 0;
    for (let s = 0; s < 16; s++) {
      if (isGoal(s)) continue;
      let b = -Infinity;
      for (let a = 0; a < 4; a++) b = Math.max(b, qOf(V, s, a, gm, slip));
      NV[s] = b; d = Math.max(d, Math.abs(b - V[s]));
    }
    return { V: NV, d };
  };
  const solvePol = (pi, gm, slip) => {
    let V = Array(16).fill(0);
    for (let i = 0; i < 200000; i++) { const r = sweepPol(V, pi, gm, slip); V = r.V; if (r.d < 1e-10) break; }
    return V;
  };
  const solveOpt = (gm, slip) => {
    let V = Array(16).fill(0);
    for (let i = 0; i < 200000; i++) { const r = sweepOpt(V, gm, slip); V = r.V; if (r.d < 1e-10) break; }
    return V;
  };

  /* ---------- lab chrome helpers (same pattern as the other lab files) ---------- */
  const slider = (ctl, name, min, max, step, val, unit, dec = 2) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = AI.fmt(+inp.value, dec) + (unit ? ' ' + unit : ''));
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  const select = (ctl, name, opts, val) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name}</span>`;
    const s = document.createElement('select');
    opts.forEach(([v, t]) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = t; if (String(v) === String(val)) o.selected = true;
      s.appendChild(o);
    });
    l.appendChild(s); ctl.appendChild(l);
    return () => s.value;
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const readout = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; };
  /* transport: play/pause, step (one unit of work), reset and a speed multiplier */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 4, 0.05, speed, '×', 2);
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

  /* ---------- drawing ---------- */
  const gridLayout = (cv) => {
    const cell = Math.min(cv.w, cv.h) / 4;
    return { cell, ox: (cv.w - 4 * cell) / 2, oy: (cv.h - 4 * cell) / 2 };
  };
  const cellAt = (cv, e) => {
    const rect = cv.cv.getBoundingClientRect(), L = gridLayout(cv);
    const x = e.clientX - rect.left - L.ox, y = e.clientY - rect.top - L.oy;
    if (x < 0 || y < 0 || x >= 4 * L.cell || y >= 4 * L.cell) return -1;
    return Math.floor(y / L.cell) * 4 + Math.floor(x / L.cell);
  };
  const drawArrows = (g, cx, cy, cell, dirs, color, alpha) => {
    g.save(); g.globalAlpha = alpha;
    for (const a of dirs) {
      AI.arrow(g, cx + DX[a] * cell * 0.3, cy + DY[a] * cell * 0.3, cx + DX[a] * cell * 0.46, cy + DY[a] * cell * 0.46, color, 2.2);
    }
    g.restore();
  };
  // pi: policy array (null = no arrows at all); cells with null inside pi are uniform; o.sel = highlighted cell; o.proposed = orange arrows
  const drawGrid = (cv, V, pi, o = {}) => {
    const C = AI.col(), g = cv.ctx, L = gridLayout(cv), cell = L.cell;
    g.clearRect(0, 0, cv.w, cv.h);
    let vmin = 0;
    for (let s = 0; s < 16; s++) vmin = Math.min(vmin, V[s]);
    const scale = Math.max(1, -vmin);
    for (let s = 0; s < 16; s++) {
      const x = L.ox + (s & 3) * cell, y = L.oy + (s >> 2) * cell, cx = x + cell / 2, cy = y + cell / 2;
      g.fillStyle = isGoal(s) ? AI.mix(C.bg2, C.green, 0.45) : AI.mix(C.bg2, C.blue, 0.08 + 0.4 * AI.clamp(-V[s] / scale, 0, 1));
      g.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (o.sel === s) { g.strokeStyle = C.yellow; g.lineWidth = 3; g.strokeRect(x + 2, y + 2, cell - 4, cell - 4); }
      g.font = '10px ' + MONO; g.fillStyle = C.muted; g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillText(String(s), x + 4, y + 4);
      g.font = '13px ' + MONO; g.fillStyle = C.fg; g.textAlign = 'center'; g.textBaseline = 'middle';
      if (isGoal(s)) { g.fillText('goal', cx, cy); continue; }
      g.fillText(AI.fmt(V[s], 2), cx, cy);
      if (o.proposed) {
        drawArrows(g, cx, cy, cell, o.proposed[s] === null ? [0, 1, 2, 3] : [o.proposed[s]], C.orange, 1);
      } else if (pi) {
        if (pi[s] === null) drawArrows(g, cx, cy, cell, [0, 1, 2, 3], C.muted, 0.35);
        else drawArrows(g, cx, cy, cell, [pi[s]], C.purple, 1);
      }
    }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  };
  // generic plot in plot units. o: {xmax, ymin, ymax, ticks:[{v,label}], xlabel, series:[{pts:[[x,y]...], color, dash, w}]}
  const plot = (cv, o) => {
    const C = AI.col(), g = cv.ctx;
    g.clearRect(0, 0, cv.w, cv.h);
    const v = AI.view(cv, 0, o.xmax, o.ymin, o.ymax, { l: 52, r: 14, t: 10, b: 26 });
    g.font = '11px ' + MONO; g.lineWidth = 1;
    g.strokeStyle = C.border; g.strokeRect(v.pad.l + 0.5, v.pad.t + 0.5, v.pw, v.ph);
    g.fillStyle = C.muted; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const t of o.ticks || []) {
      const y = v.Y(t.v);
      g.save(); g.globalAlpha = 0.25; g.beginPath(); g.moveTo(v.pad.l, y); g.lineTo(v.pad.l + v.pw, y); g.strokeStyle = C.border; g.stroke(); g.restore();
      g.fillText(t.label, v.pad.l - 6, y);
    }
    g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    if (o.xlabel) g.fillText(o.xlabel, v.pad.l + v.pw / 2, cv.h - 6);
    for (const s of o.series) {
      if (!s.pts.length) continue;
      g.strokeStyle = s.color; g.lineWidth = s.w || 2; g.setLineDash(s.dash || []);
      g.beginPath();
      s.pts.forEach(([x, y], i) => {
        const X = v.X(x), Y = v.Y(AI.clamp(y, o.ymin, o.ymax));
        if (i) g.lineTo(X, Y); else g.moveTo(X, Y);
      });
      g.stroke(); g.setLineDash([]);
    }
    return v;
  };

  /* ================= 2.1 Bellman equations: the backup as a fixed-point map ================= */
  AI.labs['dp-bellman'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A fixed point you can watch',
      'illustrative · two states, one choice, exact arithmetic; each backup uses the previous numbers and the values settle on the Bellman solution',
      '<canvas class="dg" style="height:170px"></canvas><canvas class="dp" style="height:200px"></canvas>', '');
    const cd = AI.canvas(AI.$('canvas.dg', view)), cp = AI.canvas(AI.$('canvas.dp', view));
    const mode = select(ctl, 'Policy being backed up', [['opt', 'Optimal (max)'], ['stay', 'Always stay'], ['go', 'Always go']], 'opt');
    const gam = slider(ctl, 'Discount γ', 0.5, 0.99, 0.01, 0.9, '', 2);
    const rs = slider(ctl, 'Stay reward r', 0, 3, 0.1, 1, '', 1);
    const out = readout(ctl);
    let VA = 0, VB = 0, n = 0, hist = [], prm = '';
    const key = () => mode() + '|' + gam() + '|' + rs();
    const reset = () => { VA = 0; VB = 0; n = 0; hist = []; prm = key(); };
    const exact = () => {
      const gm = gam(), r = rs(), st = r / (1 - gm), go = 3 / (1 - gm * gm);
      const a = mode() === 'stay' ? st : mode() === 'go' ? go : Math.max(st, go);
      return { A: a, B: gm * a };
    };
    reset();
    const T = transport(ctl, reset, 1);
    const tk = AI.ticker();
    // states: A (stay: reward r, stays in A; go: reward 3, moves to B); B (only go: reward 0, moves to A)
    const backup = () => {
      const gm = gam(), r = rs();
      const qS = r + gm * VA, qG = 3 + gm * VB, qB = gm * VA;
      const nA = mode() === 'stay' ? qS : mode() === 'go' ? qG : Math.max(qS, qG);
      VA = nA; VB = qB; n++;
      hist.push([n, VA, VB]);
    };
    const update = (dt) => {
      if (key() !== prm) reset();
      const e = exact();
      if (Math.abs(VA - e.A) < 1e-7 && Math.abs(VB - e.B) < 1e-7) { T.pending = 0; return; } // converged: stop stepping
      let steps = T.playing ? tk(dt, 2 * T.speed()) : 0;
      steps += T.pending; T.pending = 0;
      for (let i = 0; i < steps && n < 6000; i++) backup();
    };
    const drawD = () => {
      const C = AI.col(), g = cd.ctx, w = cd.w, h = cd.h;
      g.clearRect(0, 0, w, h);
      const ax = w * 0.28, bx = w * 0.72, y = h * 0.56, R = Math.min(34, h * 0.2);
      g.font = '12px ' + MONO; g.textAlign = 'center'; g.textBaseline = 'middle';
      // stay: a loop on A
      g.strokeStyle = C.blue; g.lineWidth = 1.8;
      g.beginPath(); g.ellipse(ax, y - R - 12, 13, 10, 0, 0, Math.PI * 2); g.stroke();
      g.fillStyle = C.fg; g.fillText(`stay, +${AI.fmt(rs(), 1)}`, ax, y - R - 36);
      // go: A to B (upper) and B to A (lower)
      AI.arrow(g, ax + R, y - 9, bx - R, y - 9, C.purple, 2);
      AI.arrow(g, bx - R, y + 9, ax + R, y + 9, C.teal, 2);
      g.fillStyle = C.fg; g.fillText('go, +3', w / 2, y - 22);
      g.fillText('go, +0', w / 2, y + 24);
      // nodes
      [[ax, 'A', VA, C.blue], [bx, 'B', VB, C.teal]].forEach(([x, name, val, col]) => {
        g.beginPath(); g.arc(x, y, R, 0, Math.PI * 2); g.fillStyle = C.bg3; g.fill(); g.strokeStyle = col; g.lineWidth = 2; g.stroke();
        g.fillStyle = C.fg; g.font = 'bold 13px ' + MONO; g.fillText(name, x, y - 7);
        g.font = '11px ' + MONO; g.fillText('V = ' + AI.fmt(val, 3), x, y + 9);
      });
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    };
    const drawP = () => {
      const C = AI.col(), e = exact();
      const xmax = Math.max(30, Math.ceil(n * 1.1)), ymax = Math.max(1, e.A * 1.15);
      const series = [
        { pts: hist.map((p) => [p[0], p[1]]), color: C.blue, w: 2 },
        { pts: hist.map((p) => [p[0], p[2]]), color: C.teal, w: 1.5 },
        { pts: [[0, e.A], [xmax, e.A]], color: C.purple, dash: [6, 4], w: 1.5 },
      ];
      plot(cp, { xmax, ymin: 0, ymax, xlabel: 'backups', ticks: [0, ymax / 2, ymax].map((v) => ({ v, label: AI.fmt(v, 1) })), series });
    };
    const draw = () => {
      drawD(); drawP();
      const e = exact(), gm = gam(), r = rs();
      const qS = r + gm * VA, qG = 3 + gm * VB;
      const pname = mode() === 'opt' ? 'optimal (max)' : mode() === 'stay' ? 'always stay' : 'always go';
      const conv = Math.abs(VA - e.A) < 1e-7 && Math.abs(VB - e.B) < 1e-7;
      out.textContent =
        `backup ${n}   γ = ${AI.fmt(gm, 2)}   r = ${AI.fmt(r, 1)}   backing up: ${pname}\n` +
        `Q(A, stay) = r + γ·V(A) = ${AI.fmt(qS, 3)}     Q(A, go) = 3 + γ·V(B) = ${AI.fmt(qG, 3)}\n` +
        `V(A) = ${AI.fmt(VA, 3)}   V(B) = ${AI.fmt(VB, 3)}   exact V(A) = ${AI.fmt(e.A, 3)} (dashed)\n` +
        (conv ? 'converged: the backup no longer changes the values, so they satisfy the Bellman equation.'
              : 'Each backup replaces V by the right-hand side of the Bellman equation, using the previous numbers.');
    };
    live(root, [cd, cp], () => n + '|' + key() + '|' + VA.toFixed(9), update, draw);
  };

  /* ================= 2.2 Iterative policy evaluation ================= */
  AI.labs['dp-evaluation'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Iterative policy evaluation, sweep by sweep',
      'illustrative · 4×4 gridworld, goals in cells 0 and 15, every move costs −1; each sweep replaces every cell using the previous sweep',
      '<canvas class="g" style="height:300px"></canvas><canvas class="p" style="height:130px"></canvas>', '');
    const cg = AI.canvas(AI.$('canvas.g', view)), cp = AI.canvas(AI.$('canvas.p', view));
    const pol = select(ctl, 'Policy', [['uni', 'Uniform random (¼ each)'], ['down', 'Always down']], 'uni');
    const gam = slider(ctl, 'Discount γ', 0.5, 1, 0.01, 1, '', 2);
    const tol = select(ctl, 'Stop when largest change < θ', [['0.01', '0.01'], ['0.001', '0.001'], ['0.000001', '0.000001']], '0.01');
    const out = readout(ctl);
    let V, pi, sw, d, deltas, done, diverging, prm;
    const key = () => pol() + '|' + gam() + '|' + tol();
    const reset = () => {
      V = Array(16).fill(0);
      pi = pol() === 'down' ? Array(16).fill(1) : UNI();
      sw = 0; d = Infinity; deltas = []; done = false; diverging = false; prm = key();
    };
    reset();
    const T = transport(ctl, reset, 1);
    const tk = AI.ticker();
    const update = (dt) => {
      if (key() !== prm) reset();
      if (done || diverging) return;
      let steps = T.playing ? tk(dt, 3 * T.speed()) : 0;
      steps += T.pending; T.pending = 0;
      const gm = gam(), th = +tol();
      for (let i = 0; i < steps && !done && !diverging; i++) {
        const r = sweepPol(V, pi, gm, 0);
        V = r.V; d = r.d; sw++; deltas.push(d);
        if (d < th) done = true;
        else if (gm === 1 && sw >= 50 && d > 0.5 && Math.min(...V) < -50) diverging = true;
      }
    };
    const draw = () => {
      const gm = gam(), th = +tol();
      drawGrid(cg, V, pi);
      const ymax = 1, ymin = -7;
      const pts = deltas.map((x, i) => [i + 1, Math.log10(Math.max(x, 1e-12))]);
      const C = AI.col();
      plot(cp, {
        xmax: Math.max(20, sw), ymin, ymax, xlabel: 'sweep',
        ticks: [-6, -4, -2, 0].map((v) => ({ v, label: '1e' + v })),
        series: [
          { pts, color: C.blue, w: 2 },
          { pts: [[0, Math.log10(th)], [Math.max(20, sw), Math.log10(th)]], color: C.green, dash: [5, 4], w: 1.2 },
        ],
      });
      let cap;
      if (diverging) cap = 'diverging: with γ = 1 and always down, the cells that never reach a goal keep losing −1 per sweep, so the values have no finite limit.';
      else if (done) cap = `stopped after ${sw} sweeps: the largest change fell below θ. The table is accurate to about the bound below.`;
      else if (sw === 0) cap = 'sweep 0: every value starts at 0. Press Play or Step.';
      else cap = `sweep ${sw}: every cell now equals its one-step expectation, using the previous sweep's values.`;
      const bound = gm < 1 && sw > 0 ? AI.fmt((gm * d) / (1 - gm), 4) : (gm === 1 ? 'none (γ = 1)' : '–');
      out.textContent =
        `sweep ${sw}   largest change Δ = ${sw ? AI.fmt(d, 6) : '–'}   θ = ${AI.fmt(th, 6)}\n` +
        `V(cell 1) = ${AI.fmt(V[1], 4)}   error bound γΔ/(1−γ) = ${bound}\n` + cap;
    };
    live(root, [cg, cp], () => sw + '|' + key() + '|' + done + diverging, update, draw);
  };

  /* ================= 2.3 Policy improvement ================= */
  AI.labs['dp-improvement'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Action values and greedy improvement',
      'illustrative · click a cell to see its four action values; Improve makes every cell greedy; Evaluate runs sweeps for the new arrows',
      '<canvas class="g" style="height:300px"></canvas><canvas class="b" style="height:170px"></canvas>', '');
    const cg = AI.canvas(AI.$('canvas.g', view)), cb = AI.canvas(AI.$('canvas.b', view));
    const gam = select(ctl, 'Discount γ', [['1', '1 (undiscounted)'], ['0.9', '0.9']], '1');
    const row = rowOf(ctl);
    const bImp = button(row, 'Improve'), bEval = button(row, 'Evaluate'), bReset = button(row, 'Reset to uniform');
    const speed = slider(ctl, 'Evaluation speed', 0.5, 8, 0.5, 3, 'sweeps/s', 1);
    const out = readout(ctl);
    let V, pi, sel = 1, evaluating = false, evalSweeps = 0, msg = '', prm = '';
    const reset = () => {
      const gm = +gam();
      pi = UNI();
      V = solvePol(pi, gm, 0); // the converged uniform-policy values for this γ
      evaluating = false; evalSweeps = 0; prm = gam();
      msg = 'Uniform policy: each cell averages its four action values. Click a cell, then press Improve.';
    };
    reset();
    bReset.onclick = reset;
    bImp.onclick = () => {
      const np = greedy(V, +gam(), 0);
      const ch = countChanged(pi, np);
      pi = np; evaluating = false;
      msg = `Improve: ${ch} cells changed their arrow. The values are still the old ones; press Evaluate to see what the new arrows are worth.`;
    };
    bEval.onclick = () => {
      evaluating = true; evalSweeps = 0;
      msg = 'Evaluate: sweeps update the values for the current arrows until the largest change is below 1e-6.';
    };
    cg.cv.addEventListener('click', (e) => { const s = cellAt(cg, e); if (s >= 0 && !isGoal(s)) sel = s; });
    const tk = AI.ticker();
    const update = (dt) => {
      if (gam() !== prm) reset();
      if (!evaluating) return;
      const steps = tk(dt, +speed());
      for (let i = 0; i < steps && evaluating; i++) {
        const r = sweepPol(V, pi, +gam(), 0);
        V = r.V; evalSweeps++;
        if (r.d < 1e-6) {
          evaluating = false;
          msg = `Evaluate: converged after ${evalSweeps} sweeps. Press Improve again: if no arrow changes, the policy is greedy for its own values.`;
        }
      }
    };
    const draw = () => {
      const C = AI.col(), gm = +gam(), s = sel;
      drawGrid(cg, V, pi, { sel: s });
      const cbg = cb.ctx; cbg.clearRect(0, 0, cb.w, cb.h);
      if (isGoal(s)) {
        out.textContent = msg + '\n' + 'Cell ' + s + ' is a goal: no decision is made there. Click another cell.';
        return;
      }
      const Q = [0, 1, 2, 3].map((a) => qOf(V, s, a, gm, 0));
      const lo = Math.min(-1, Math.min(...Q) * 1.25);
      const v = AI.view(cb, -0.6, 3.6, lo, 0, { l: 52, r: 12, t: 12, b: 26 });
      cbg.font = '11px ' + MONO; cbg.textAlign = 'center';
      cbg.strokeStyle = C.border; cbg.strokeRect(v.pad.l + 0.5, v.pad.t + 0.5, v.pw, v.ph);
      cbg.fillStyle = C.muted; cbg.textAlign = 'right'; cbg.textBaseline = 'middle';
      cbg.fillText('0', v.pad.l - 6, v.Y(0));
      cbg.fillText(AI.fmt(lo, 1), v.pad.l - 6, v.Y(lo));
      cbg.textBaseline = 'alphabetic';
      const best = Q.indexOf(Math.max(...Q));
      const chosen = pi[s];
      const mean = chosen === null ? Q.reduce((x, y) => x + y, 0) / 4 : Q[chosen];
      for (let a = 0; a < 4; a++) {
        const x = v.X(a), y0 = v.Y(0), y1 = v.Y(Q[a]);
        const hi = pi[s] === null ? C.blue : (a === chosen ? C.purple : C.muted);
        cbg.globalAlpha = 0.65; cbg.fillStyle = hi; cbg.fillRect(x - v.pw * 0.09, Math.min(y0, y1), v.pw * 0.18, Math.abs(y1 - y0) || 1);
        cbg.globalAlpha = 1;
        cbg.fillStyle = C.fg; cbg.textAlign = 'center';
        cbg.fillText(`${NAMES[a]} ${AI.fmt(Q[a], 2)}`, x, v.pad.t + v.ph + 16);
        if (a === best) { cbg.strokeStyle = C.green; cbg.lineWidth = 2; cbg.strokeRect(x - v.pw * 0.1, Math.min(y0, y1) - 1, v.pw * 0.2, Math.abs(y1 - y0) + 2); }
      }
      cbg.textAlign = 'left';
      out.textContent =
        `${msg}\n` +
        `cell ${s}: Q(up) = ${AI.fmt(Q[0], 3)}   Q(down) = ${AI.fmt(Q[1], 3)}   Q(left) = ${AI.fmt(Q[2], 3)}   Q(right) = ${AI.fmt(Q[3], 3)}\n` +
        (chosen === null
          ? `uniform policy: V(${s}) = average of the four = ${AI.fmt(mean, 3)}   (table shows ${AI.fmt(V[s], 3)})`
          : `current arrow ${ARW[chosen]}: V(${s}) = Q(${s}, ${NAMES[chosen]}) = ${AI.fmt(mean, 3)}   best action: ${NAMES[best]}`) +
        `\n${gm === 1 ? 'γ = 1' : 'γ = ' + AI.fmt(gm, 2)}   evaluation sweeps so far: ${evalSweeps}`;
    };
    live(root, [cg, cb], () => sel + '|' + (pi ? pi.join(',') : '') + '|' + V.map((x) => x.toFixed(6)).join(',') + '|' + msg + '|' + gam(), update, draw);
  };

  /* ================= 2.4 Policy iteration ================= */
  AI.labs['dp-policy-iteration'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Policy iteration: evaluate, then improve, until the arrows settle',
      'illustrative · discounted 4×4 gridworld with deterministic moves; each round evaluates the current arrows from V = 0, then improves',
      '<canvas class="g" style="height:320px"></canvas><canvas class="p" style="height:120px"></canvas>', '');
    const cg = AI.canvas(AI.$('canvas.g', view)), cp = AI.canvas(AI.$('canvas.p', view));
    const start = select(ctl, 'Start policy', [['down', 'Always down'], ['uni', 'Uniform random']], 'down');
    const gam = select(ctl, 'Discount γ', [['0.9', '0.9'], ['0.99', '0.99']], '0.9');
    const theta = select(ctl, 'Evaluation tolerance θ', [['0.001', '0.001'], ['0.0001', '0.0001']], '0.001');
    const out = readout(ctl);
    let pi, V, phase, k, swRound, d, deltas, roundLog, holdT, proposed, prm, msg;
    const key = () => start() + '|' + gam() + '|' + theta();
    const reset = () => {
      pi = start() === 'down' ? Array(16).fill(1) : UNI();
      V = Array(16).fill(0); phase = 'eval'; k = 0; swRound = 0; d = Infinity; deltas = [];
      roundLog = []; holdT = 0; proposed = null; prm = key();
      msg = 'Round 1 · evaluation: sweeps compute the values of the current arrows, starting from V = 0.';
    };
    reset();
    const T = transport(ctl, reset, 1);
    const tk = AI.ticker();
    const applyImprove = (gm) => {
      const ch = countChanged(pi, proposed);
      roundLog.push({ r: k + 1, sweeps: swRound, changed: ch });
      pi = proposed; proposed = null;
      if (ch === 0) {
        phase = 'done';
        msg = `Stop: round ${k + 1} changed no arrow, so the policy is greedy for its own values and the values are optimal.`;
      } else {
        k++; phase = 'eval'; V = Array(16).fill(0); swRound = 0; d = Infinity; deltas = [];
        msg = `Round ${k + 1} · evaluation: the new arrows are evaluated from V = 0.`;
      }
    };
    const update = (dt) => {
      if (key() !== prm) reset();
      const gm = +gam(), th = +theta();
      if (phase === 'done') return;
      if (phase === 'hold') {
        if (T.pending) { holdT = 99; T.pending = 0; }
        if (T.playing) holdT += dt * T.speed();
        if (holdT >= 2.2) applyImprove(gm);
        return;
      }
      let steps = T.playing ? tk(dt, 6 * T.speed()) : 0;
      steps += T.pending; T.pending = 0;
      for (let i = 0; i < steps && phase === 'eval'; i++) {
        const r = sweepPol(V, pi, gm, 0);
        V = r.V; d = r.d; swRound++; deltas.push(d);
        if (d < th) {
          phase = 'hold'; holdT = 0;
          proposed = greedy(V, gm, 0);
          msg = `Improve: orange arrows are greedy for these evaluated values. They replace the purple arrows if they differ.`;
        }
      }
    };
    const draw = () => {
      const C = AI.col();
      drawGrid(cg, V, pi, { proposed: phase === 'hold' ? proposed : null });
      const xmax = Math.max(20, swRound);
      const pts = deltas.map((x, i) => [i + 1, Math.log10(Math.max(x, 1e-12))]);
      plot(cp, {
        xmax, ymin: -7, ymax: 1, xlabel: 'evaluation sweep in this round',
        ticks: [-6, -4, -2, 0].map((v) => ({ v, label: '1e' + v })),
        series: [
          { pts, color: C.blue, w: 2 },
          { pts: [[0, Math.log10(+theta())], [xmax, Math.log10(+theta())]], color: C.green, dash: [5, 4], w: 1.2 },
        ],
      });
      const log = roundLog.map((r) => `round ${r.r}: ${r.sweeps} evaluation sweeps, ${r.changed} arrows changed`).join('\n');
      out.textContent =
        `${msg}\n` +
        `round ${k + 1}   sweep ${swRound}   Δ = ${swRound ? AI.fmt(d, 5) : '–'}   θ = ${theta()}   γ = ${gam()}\n` +
        (log ? log : 'no round finished yet');
    };
    live(root, [cg, cp], () => phase + '|' + k + '|' + swRound + '|' + key() + '|' + (proposed ? proposed.join(',') : '') + '|' + (pi ? pi.join(',') : ''), update, draw);
  };

  /* ================= 2.5 Value iteration ================= */
  AI.labs['dp-value-iteration'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Value iteration: max-backup sweeps and the stopping bound',
      'illustrative · 4×4 gridworld with slippery moves: the intended move succeeds with probability 1 − slip, otherwise one of the other three',
      '<canvas class="g" style="height:300px"></canvas><canvas class="p" style="height:130px"></canvas>', '');
    const cg = AI.canvas(AI.$('canvas.g', view)), cp = AI.canvas(AI.$('canvas.p', view));
    const slip = select(ctl, 'Slip probability', [['0', '0 (deterministic)'], ['0.2', '0.2'], ['0.4', '0.4']], '0.2');
    const gam = slider(ctl, 'Discount γ', 0.5, 0.99, 0.01, 0.9, '', 2);
    const theta = select(ctl, 'Stop when largest change < θ', [['0.01', '0.01'], ['0.001', '0.001'], ['0.0001', '0.0001']], '0.001');
    const out = readout(ctl);
    let V, pi, sw, d, deltas, done, prm;
    const key = () => slip() + '|' + gam() + '|' + theta();
    const reset = () => { V = Array(16).fill(0); pi = null; sw = 0; d = Infinity; deltas = []; done = false; prm = key(); };
    reset();
    const T = transport(ctl, reset, 1);
    const tk = AI.ticker();
    const update = (dt) => {
      if (key() !== prm) reset();
      if (done) return;
      let steps = T.playing ? tk(dt, 2 * T.speed()) : 0;
      steps += T.pending; T.pending = 0;
      const gm = gam(), sp = +slip(), th = +theta();
      for (let i = 0; i < steps && !done; i++) {
        const r = sweepOpt(V, gm, sp);
        V = r.V; d = r.d; sw++; deltas.push(d);
        pi = greedy(V, gm, sp);
        if (d < th) done = true;
      }
    };
    const draw = () => {
      const C = AI.col(), gm = gam(), sp = +slip(), th = +theta();
      drawGrid(cg, V, pi);
      const C2 = AI.col();
      plot(cp, {
        xmax: Math.max(20, sw), ymin: -6.5, ymax: 0.5, xlabel: 'sweep',
        ticks: [-6, -4, -2, 0].map((v) => ({ v, label: '1e' + v })),
        series: [
          { pts: deltas.map((x, i) => [i + 1, Math.log10(Math.max(x, 1e-12))]), color: C.blue, w: 2 },
          { pts: [[0, Math.log10(th)], [Math.max(20, sw), Math.log10(th)]], color: C2.green, dash: [5, 4], w: 1.2 },
        ],
      });
      const bound = sw ? (gm * d) / (1 - gm) : NaN;
      const terms = 14 * 4 * (sp > 0 ? 4 : 1);
      let cap;
      if (sw === 0) cap = 'sweep 0: V = 0 everywhere and there are no arrows yet. Press Play or Step.';
      else if (done) cap = `stopped after ${sw} sweeps: the largest change is below θ, so the greedy arrows are now the policy to follow.`;
      else cap = `sweep ${sw}: each cell takes the best action of its one-step look-ahead. The arrows are recomputed after every sweep.`;
      out.textContent =
        `sweep ${sw}   largest change Δ = ${sw ? AI.fmt(d, 6) : '–'}   θ = ${th}\n` +
        `V(cell 1) = ${AI.fmt(V[1], 4)}   greedy arrow in cell 1: ${pi ? ARW[pi[1]] : '–'}\n` +
        `error bound if you stop now, γΔ/(1−γ) = ${sw ? AI.fmt(bound, 4) : '–'}   work per sweep: 14 cells × 4 actions × ${sp > 0 ? 4 : 1} outcomes = ${terms} terms\n` +
        cap;
    };
    live(root, [cg, cp], () => sw + '|' + key() + '|' + done, update, draw);
  };

  /* ================= 2.6 Generalised policy iteration ================= */
  AI.labs['dp-gpi'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Generalised policy iteration: how much evaluation per improvement?',
      'illustrative · slippery 4×4 gridworld; k evaluation sweeps per improvement step; dashed curves show k = 1 and k = ∞ for comparison',
      '<canvas class="g" style="height:280px"></canvas><canvas class="p" style="height:170px"></canvas>', '');
    const cg = AI.canvas(AI.$('canvas.g', view)), cp = AI.canvas(AI.$('canvas.p', view));
    const kSel = select(ctl, 'Evaluation sweeps per improvement, k',
      [['1', '1 (value-iteration-like)'], ['2', '2'], ['3', '3'], ['5', '5'], ['10', '10'], ['inf', '∞ (policy iteration)']], '1');
    const slip = select(ctl, 'Slip probability', [['0', '0 (deterministic)'], ['0.2', '0.2']], '0.2');
    const gam = select(ctl, 'Discount γ', [['0.9', '0.9'], ['0.99', '0.99']], '0.9');
    const out = readout(ctl);
    const CAP = 1500, MAXROUNDS = 60;
    const cache = {};
    const optCache = {};
    const optimum = (gm, sp) => { const k = gm + '|' + sp; if (!optCache[k]) optCache[k] = solveOpt(gm, sp); return optCache[k]; };
    const errOf = (V, Vs) => { let m = 0; for (let s = 0; s < 16; s++) if (!isGoal(s)) m = Math.max(m, Math.abs(V[s] - Vs[s])); return m; };
    // The whole run is cheap on a 4×4 grid, so the trajectory is computed once per setting and then replayed.
    const trajectory = (kv, gm, sp) => {
      const key = kv + '|' + gm + '|' + sp;
      if (cache[key]) return cache[key];
      const Vs = optimum(gm, sp), kLim = kv === 'inf' ? Infinity : +kv;
      let V = Array(16).fill(0), pi = UNI(), sw = 0, round = 0;
      const frames = [{ kind: 'start', V, pi, sw, round, err: errOf(V, Vs) }];
      while (sw < CAP && round < MAXROUNDS) {
        let j = 0;
        while (j < kLim && sw < CAP) {
          const r = sweepPol(V, pi, gm, sp);
          V = r.V; sw++; j++;
          frames.push({ kind: 'eval', V, pi, sw, round, err: errOf(V, Vs) });
          if (kLim === Infinity && r.d < 1e-7) break;
        }
        const np = greedy(V, gm, sp);
        const ch = countChanged(pi, np);
        pi = np; round++;
        frames.push({ kind: 'improve', V, pi, sw, round, err: errOf(V, Vs), changed: ch });
        if (ch === 0 && errOf(V, Vs) < 1e-6) break;
      }
      return (cache[key] = frames);
    };
    const sweepsTo = (frames, th) => { for (const f of frames) if (f.kind !== 'improve' && f.err < th) return f.sw; return null; };
    let frames, idx = 0, prm = '';
    const T = transport(ctl, () => { idx = 0; }, 1);
    const tk = AI.ticker();
    const advance = (n) => { while (n > 0 && idx < frames.length - 1) { idx++; if (frames[idx].kind === 'eval') n--; } };
    const update = (dt) => {
      const key = kSel() + '|' + gam() + '|' + slip();
      if (key !== prm) { prm = key; frames = trajectory(kSel(), +gam(), +slip()); idx = 0; }
      if (T.playing) advance(tk(dt, 4 * T.speed()));
      if (T.pending) { advance(1); T.pending = 0; }
      if (idx >= frames.length - 1) T.setPlaying(false);
    };
    const draw = () => {
      const C = AI.col();
      const f = frames[idx];
      const gm = +gam(), sp = +slip();
      drawGrid(cg, f.V, f.pi);
      const ref1 = trajectory('1', gm, sp), refI = trajectory('inf', gm, sp);
      const xmax = Math.max(20, ...ref1.map((x) => x.sw), ...refI.map((x) => x.sw), f.sw);
      const path = frames.slice(0, idx + 1).filter((x) => x.kind !== 'improve').map((x) => [x.sw, Math.log10(Math.max(x.err, 1e-6))]);
      const tick = [-6, -4, -2, 0].map((v) => ({ v, label: '1e' + v }));
      plot(cp, {
        xmax, ymin: -6.2, ymax: 0.3, xlabel: 'total evaluation sweeps', ticks: tick,
        series: [
          { pts: ref1.filter((x) => x.kind !== 'improve').map((x) => [x.sw, Math.log10(Math.max(x.err, 1e-6))]), color: C.teal, dash: [5, 4], w: 1.5 },
          { pts: refI.filter((x) => x.kind !== 'improve').map((x) => [x.sw, Math.log10(Math.max(x.err, 1e-6))]), color: C.purple, dash: [5, 4], w: 1.5 },
          { pts: path, color: C.blue, w: 2.5 },
        ],
      });
      const to = (fr) => { const s = sweepsTo(fr, 1e-3); return s === null ? 'not reached' : s; };
      const kLabel = kSel() === 'inf' ? '∞' : kSel();
      const cap = f.kind === 'start' ? 'Start: V = 0 and the uniform policy. Each round below does k evaluation sweeps, then one greedy improvement.'
        : f.kind === 'improve' ? `Improvement: the arrows are made greedy for the current values (${f.changed} arrows changed). The next sweep evaluates these arrows.`
        : `Evaluation sweep ${f.sw} of round ${f.round + 1}: the values move toward the value of the current arrows.`;
      out.textContent =
        `k = ${kLabel}   sweep ${f.sw}   round ${f.round}   error max|V − V*| = ${AI.fmt(f.err, 4)}\n` +
        `sweeps to reach error < 0.001:  this run ${to(frames)}   k = 1: ${to(ref1)}   k = ∞: ${to(refI)}\n` +
        cap;
    };
    live(root, [cg, cp], () => (frames ? idx : -1) + '|' + kSel() + '|' + gam() + '|' + slip(), update, draw);
  };
})();
