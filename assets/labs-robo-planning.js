/* Robotics · Module 5 (Planning & motion) labs. Illustrative simulations: no real robot is involved.
 * plan-cspace    <div data-lab="plan-cspace"></div>    two-link arm in workspace and in joint-angle space
 * plan-astar     <div data-lab="plan-astar"></div>     Dijkstra, A* and weighted A* on an editable grid
 * plan-rrt       <div data-lab="plan-rrt"></div>       RRT tree growing around obstacles (seeded)
 * plan-poly      <div data-lab="plan-poly"></div>      quintic / cubic rest-to-rest moves with limits
 * plan-trajopt   <div data-lab="plan-trajopt"></div>   a path bent by gradient descent on smoothness + clearance
 * plan-strips    <div data-lab="plan-strips"></div>    STRIPS-style task plan, state by state
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const F = (v, d = 3) => AI.fmt(+(+v).toFixed(d), d);

  /* shared control helpers */
  const sliderRow = (ctl, label, o) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${o.min}" max="${o.max}" step="${o.step}" value="${o.value}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const dec = o.dec === undefined ? 2 : o.dec;
    const upd = () => (v.textContent = (+inp.value).toFixed(dec) + (o.unit ? ' ' + o.unit : ''));
    inp.oninput = () => { upd(); if (o.onChange) o.onChange(+inp.value); };
    upd();
    ctl.appendChild(l);
    return { get: () => +inp.value, set: (x) => { inp.value = x; upd(); }, el: inp };
  };
  const selectRow = (ctl, label, options, value, onChange) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span><select>${options.map(([k, t]) => `<option value="${k}"${k === value ? ' selected' : ''}>${t}</option>`).join('')}</select>`;
    const sel = l.querySelector('select');
    sel.onchange = () => onChange(sel.value);
    ctl.appendChild(l);
    return sel;
  };
  const btnRow = (ctl, labels) => {
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
  const dark = () => document.documentElement.dataset.theme;

  /* ================= 5.1 · two-link arm: workspace and configuration space ================= */
  AI.labs['plan-cspace'] = function (root) {
    const L1 = 1, L2 = 1, N = 90;
    const { view, ctl } = AI.labShell(root, 'Configuration space of a two-link arm',
      'illustrative · L₁ = L₂ = 1 m · joint angles in radians',
      '<canvas style="height:250px"></canvas><canvas style="height:250px;margin-top:10px"></canvas><div class="caption"></div>', '');
    const wsC = AI.canvas(AI.$$('canvas', view)[0]), csC = AI.canvas(AI.$$('canvas', view)[1]);
    const cap = AI.$('.caption', view);
    const t1 = sliderRow(ctl, 'θ₁ shoulder', { min: -Math.PI, max: Math.PI, step: 0.01, value: 0, unit: 'rad', dec: 3 });
    const t2 = sliderRow(ctl, 'θ₂ elbow (relative)', { min: -Math.PI, max: Math.PI, step: 0.01, value: 0, unit: 'rad', dec: 3 });
    const cx = sliderRow(ctl, 'obstacle Cx', { min: -0.5, max: 2.5, step: 0.05, value: 0.5, unit: 'm' });
    const cy = sliderRow(ctl, 'obstacle Cy', { min: 0, max: 2.5, step: 0.05, value: 1.2, unit: 'm' });
    const rr = sliderRow(ctl, 'obstacle radius R', { min: 0.1, max: 0.6, step: 0.01, value: 0.3, unit: 'm' });
    const sp = sliderRow(ctl, 'sweep speed', { min: 0.1, max: 1, step: 0.05, value: 0.4, unit: 'rad/s' });
    const [bSweep, bReset] = btnRow(ctl, ['▶ Sweep θ₁', 'Reset']);
    const ro = readout(ctl);

    const segDist = (ax, ay, bx, by, px, py) => {
      const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
      let t = L ? ((px - ax) * dx + (py - ay) * dy) / L : 0;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    };
    const pose = (a, b) => {
      const ex = L1 * Math.cos(a), ey = L1 * Math.sin(a);
      return { ex, ey, tx: ex + L2 * Math.cos(a + b), ty: ey + L2 * Math.sin(a + b) };
    };
    // clearance = distance from the obstacle centre to the nearest link, minus R (metres). Negative = inside.
    const gapOf = (a, b) => {
      const p = pose(a, b), ox = cx.get(), oy = cy.get();
      return Math.min(segDist(0, 0, p.ex, p.ey, ox, oy), segDist(p.ex, p.ey, p.tx, p.ty, ox, oy)) - rr.get();
    };

    let blocked = null, gridKey = '';
    const buildGrid = () => {
      const k = [cx.get(), cy.get(), rr.get()].join();
      if (k === gridKey) return;
      gridKey = k;
      blocked = new Uint8Array(N * N);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const a = -Math.PI + ((i + 0.5) * 2 * Math.PI) / N, b = -Math.PI + ((j + 0.5) * 2 * Math.PI) / N;
        blocked[j * N + i] = gapOf(a, b) < 0 ? 1 : 0;
      }
    };

    let sweeping = false, trail = [], lastKey = '', dirty = true;
    const drawWS = () => {
      const C = AI.col(), g = wsC.ctx, W = wsC.w, H = wsC.h;
      g.clearRect(0, 0, W, H);
      const s = Math.min(W / 4.6, H / 3.8), ox = W / 2, oy = H * 0.82;
      const X = (x) => ox + x * s, Y = (y) => oy - y * s;
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(W, Y(0)); g.stroke();
      g.globalAlpha = 0.35; g.fillStyle = C.red; g.beginPath(); g.arc(X(cx.get()), Y(cy.get()), rr.get() * s, 0, Math.PI * 2); g.fill(); g.globalAlpha = 1;
      g.strokeStyle = C.red; g.lineWidth = 1.5; g.stroke();
      // tip trail
      g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath();
      trail.forEach((q, i) => { const p = pose(q[0], q[1]); if (i === 0) g.moveTo(X(p.tx), Y(p.ty)); else g.lineTo(X(p.tx), Y(p.ty)); });
      g.stroke();
      const p = pose(t1.get(), t2.get());
      g.lineCap = 'round';
      g.strokeStyle = C.blue; g.lineWidth = 6; g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(p.ex), Y(p.ey)); g.stroke();
      g.strokeStyle = C.orange; g.beginPath(); g.moveTo(X(p.ex), Y(p.ey)); g.lineTo(X(p.tx), Y(p.ty)); g.stroke();
      g.lineCap = 'butt';
      g.fillStyle = C.fg; g.beginPath(); g.arc(X(0), Y(0), 5, 0, Math.PI * 2); g.fill();
      g.fillStyle = C.orange; g.beginPath(); g.arc(X(p.ex), Y(p.ey), 4, 0, Math.PI * 2); g.fill();
      g.fillStyle = C.green; g.beginPath(); g.arc(X(p.tx), Y(p.ty), 5, 0, Math.PI * 2); g.fill();
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.fillText('workspace, metres', 6, 14);
    };

    const drawCS = () => {
      const C = AI.col(), g = csC.ctx, W = csC.w, H = csC.h;
      g.clearRect(0, 0, W, H);
      const m = { l: 44, r: 10, t: 10, b: 26 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
      const X = (a) => m.l + ((a + Math.PI) / (2 * Math.PI)) * pw, Y = (b) => m.t + (1 - (b + Math.PI) / (2 * Math.PI)) * ph;
      g.fillStyle = C.bg2; g.fillRect(m.l, m.t, pw, ph);
      const cw = pw / N, ch = ph / N;
      g.fillStyle = C.red; g.globalAlpha = 0.6;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) if (blocked[j * N + i]) g.fillRect(m.l + i * cw, m.t + (N - 1 - j) * ch, cw + 0.6, ch + 0.6);
      g.globalAlpha = 1;
      g.strokeStyle = C.border; g.strokeRect(m.l, m.t, pw, ph);
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.textAlign = 'center';
      [[-Math.PI, '−π'], [0, '0'], [Math.PI, 'π']].forEach(([v, t]) => g.fillText(t, X(v), m.t + ph + 15));
      g.textAlign = 'right';
      [[-Math.PI, '−π'], [0, '0'], [Math.PI, 'π']].forEach(([v, t]) => g.fillText(t, m.l - 5, Y(v) + 4));
      g.textAlign = 'left'; g.fillText('θ₁ →', m.l + pw - 34, m.t + ph - 6); g.fillText('θ₂ ↑', m.l + 4, m.t + 12);
      g.strokeStyle = C.green; g.lineWidth = 1.5; g.beginPath();
      trail.forEach((q, i) => {
        const px = X(q[0]), py = Y(q[1]);
        if (i === 0 || Math.abs(q[0] - trail[i - 1][0]) > Math.PI) g.moveTo(px, py); else g.lineTo(px, py);
      });
      g.stroke();
      g.fillStyle = C.blue; g.beginPath(); g.arc(X(t1.get()), Y(t2.get()), 5, 0, Math.PI * 2); g.fill();
      g.strokeStyle = C.fg; g.stroke();
    };

    bSweep.onclick = () => {
      if (sweeping) sweeping = false;
      else {
        if (t1.get() >= Math.PI - 1e-6) t1.set(-Math.PI);
        sweeping = true; trail = [];
      }
      bSweep.textContent = sweeping ? '❚❚ Pause' : '▶ Sweep θ₁';
    };
    bReset.onclick = () => { sweeping = false; t1.set(0); t2.set(0); trail = []; bSweep.textContent = '▶ Sweep θ₁'; dirty = true; };
    wsC.onresize = csC.onresize = () => (dirty = true);
    ['t1', 't2', 'cx', 'cy', 'rr'].forEach((k) => { const el = { t1, t2, cx, cy, rr }[k].el; el.addEventListener('input', () => (dirty = true)); });

    AI.loop(root, (dt) => {
      if (sweeping) {
        let v = t1.get() + dt * sp.get();
        if (v >= Math.PI) { v = Math.PI; sweeping = false; bSweep.textContent = '▶ Sweep θ₁'; }
        t1.set(v);
        trail.push([t1.get(), t2.get()]);
        if (trail.length > 900) trail.shift();
        dirty = true;
      }
      buildGrid();
      const key = [t1.get(), t2.get(), cx.get(), cy.get(), rr.get(), wsC.w, wsC.h, csC.w, csC.h, dark(), trail.length].join();
      if (!dirty && key === lastKey) return;
      dirty = false; lastKey = key;
      drawWS(); drawCS();
      const p = pose(t1.get(), t2.get()), gap = gapOf(t1.get(), t2.get());
      cap.textContent = sweeping
        ? 'Sweeping θ₁ at ' + sp.get().toFixed(2) + ' rad/s: the marker follows the green trail through the dark map. Each dark crossing is a collision.'
        : 'Left: the arm in the plane (metres). Right: the same arm in joint angles (radians); dark cells are configurations where a link touches the obstacle.';
      ro.textContent =
        `elbow  (${F(p.ex)}, ${F(p.ey)}) m\n` +
        `tip    (${F(p.tx)}, ${F(p.ty)}) m\n` +
        `clearance = distance to nearest link − R = ${F(gap)} m\n` +
        `status: ${gap < 0 ? 'COLLISION (dark cell)' : 'free'}`;
    });
  };

  /* ================= 5.2 · grid search: Dijkstra, A*, weighted A* ================= */
  AI.labs['plan-astar'] = function (root) {
    const MAP = [
      '................',
      '....##..........',
      '....#...###.....',
      '....#.....#.....',
      '....####..#.....',
      '..........#.....',
      '.#####.....###..',
      '.....#.........#',
      '.....#....#....#',
      '.....#....#.....',
      '................',
    ];
    const W = 16, H = 11;
    const { view, ctl } = AI.labShell(root, 'Grid search: Dijkstra, A* and weighted A*',
      'illustrative · 4-connected grid · every step costs 1 · h = Manhattan distance',
      '<canvas style="height:340px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const wall = new Uint8Array(W * H);
    MAP.forEach((row, y) => [...row.padEnd(W, '.').slice(0, W)].forEach((ch, x) => { wall[y * W + x] = ch === '#' ? 1 : 0; }));
    const START = (H - 1) * W, GOAL = W - 1;
    const hOf = (i) => Math.abs((i % W) - (GOAL % W)) + Math.abs(Math.floor(i / W) - Math.floor(GOAL / W));

    let mode = 'astar', wWeight = 2, playing = false, dirty = true;
    let gS, parent, closed, openSet, expanded, done, found, pathCells, lastExp;
    const reset = () => {
      gS = new Float64Array(W * H).fill(Infinity); parent = new Int32Array(W * H).fill(-1);
      closed = new Uint8Array(W * H); openSet = new Set([START]); gS[START] = 0;
      expanded = 0; done = false; found = false; pathCells = null; lastExp = null; dirty = true;
    };
    const keyOf = (i) => (mode === 'dijkstra' ? gS[i] : mode === 'astar' ? gS[i] + hOf(i) : gS[i] + wWeight * hOf(i));
    const step = () => {
      if (done) return;
      if (openSet.size === 0) { done = true; found = false; playing = false; dirty = true; return; }
      let best = -1, bk = Infinity, bh = Infinity;
      for (const i of openSet) {
        const k = keyOf(i), h = hOf(i);
        if (k < bk || (k === bk && h < bh)) { best = i; bk = k; bh = h; }
      }
      openSet.delete(best); closed[best] = 1; expanded++;
      lastExp = { i: best, g: gS[best], f: bk, h: hOf(best) };
      dirty = true;
      if (best === GOAL) {
        done = true; found = true; playing = false;
        pathCells = [];
        for (let c = GOAL; c !== -1; c = parent[c]) pathCells.unshift(c);
        return;
      }
      const x = best % W, y = Math.floor(best / W);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const n = ny * W + nx;
        if (wall[n] || closed[n]) continue;
        const ng = gS[best] + 1;
        if (ng < gS[n]) { gS[n] = ng; parent[n] = best; openSet.add(n); }
      }
    };

    const modeSel = selectRow(ctl, 'method', [['dijkstra', 'Dijkstra (h = 0)'], ['astar', 'A*'], ['wastar', 'weighted A*']], 'astar', (v) => { mode = v; reset(); });
    sliderRow(ctl, 'weight w (weighted A*)', { min: 1, max: 3, step: 0.1, value: 2, dec: 1, onChange: (v) => { wWeight = v; reset(); } });
    const rate = sliderRow(ctl, 'expansions per second', { min: 1, max: 10, step: 1, value: 2, dec: 0 });
    const [bPlay, bStep, bReset] = btnRow(ctl, ['▶ Play', 'Step ▸', 'Reset']);
    const ro = readout(ctl);
    const tick = AI.ticker();
    reset();

    bPlay.onclick = () => { if (done) reset(); playing = !playing; bPlay.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
    bStep.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; step(); };
    bReset.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; reset(); };
    cv.onresize = () => (dirty = true);
    const geom = () => {
      const w = cv.w, h = cv.h, cs = Math.min(w / W, h / H);
      return { cs, ox: (w - cs * W) / 2, oy: (h - cs * H) / 2 };
    };
    cv.cv.addEventListener('click', (e) => {
      const r = cv.cv.getBoundingClientRect(), { cs, ox, oy } = geom();
      const x = Math.floor((e.clientX - r.left - ox) / cs), y = Math.floor((e.clientY - r.top - oy) / cs);
      if (x < 0 || y < 0 || x >= W || y >= H) return;
      const i = y * W + x;
      if (i === START || i === GOAL) return;
      wall[i] = wall[i] ? 0 : 1;
      playing = false; bPlay.textContent = '▶ Play';
      reset();
    });

    const draw = () => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const { cs, ox, oy } = geom();
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        let fill = C.bg2, alpha = 1;
        if (wall[i]) fill = C.muted;
        else if (closed[i]) { fill = C.blue; alpha = 0.3; }
        else if (openSet.has(i)) { fill = C.yellow; alpha = 0.45; }
        g.globalAlpha = alpha; g.fillStyle = fill;
        g.fillRect(ox + x * cs, oy + y * cs, cs, cs);
        g.globalAlpha = 1;
      }
      if (pathCells) {
        g.fillStyle = C.green; g.globalAlpha = 0.8;
        pathCells.forEach((i) => g.fillRect(ox + (i % W) * cs + cs * 0.25, oy + Math.floor(i / W) * cs + cs * 0.25, cs * 0.5, cs * 0.5));
        g.globalAlpha = 1;
      }
      g.strokeStyle = C.border; g.lineWidth = 1;
      for (let x = 0; x <= W; x++) { g.beginPath(); g.moveTo(ox + x * cs + 0.5, oy); g.lineTo(ox + x * cs + 0.5, oy + H * cs); g.stroke(); }
      for (let y = 0; y <= H; y++) { g.beginPath(); g.moveTo(ox, oy + y * cs + 0.5); g.lineTo(ox + W * cs, oy + y * cs + 0.5); g.stroke(); }
      g.fillStyle = C.green; g.beginPath(); g.arc(ox + (START % W + 0.5) * cs, oy + (Math.floor(START / W) + 0.5) * cs, cs * 0.32, 0, 7); g.fill();
      g.fillStyle = C.red; g.beginPath(); g.arc(ox + (GOAL % W + 0.5) * cs, oy + (Math.floor(GOAL / W) + 0.5) * cs, cs * 0.32, 0, 7); g.fill();
      g.fillStyle = C.fg; g.font = 'bold 11px ' + MONO; g.textAlign = 'center';
      g.fillText('S', ox + (START % W + 0.5) * cs, oy + (Math.floor(START / W) + 0.5) * cs + 4);
      g.fillText('G', ox + (GOAL % W + 0.5) * cs, oy + (Math.floor(GOAL / W) + 0.5) * cs + 4);
      g.textAlign = 'left';
    };

    AI.loop(root, (dt) => {
      if (playing && !done) {
        const n = tick(dt, rate.get());
        for (let k = 0; k < n && !done; k++) step();
        if (done) { playing = false; bPlay.textContent = '▶ Play'; }
      }
      if (!dirty) return;
      dirty = false;
      draw();
      const name = mode === 'dijkstra' ? 'Dijkstra ranks by g (cost so far)' : mode === 'astar' ? 'A* ranks by f = g + h' : 'weighted A* ranks by g + w·h, w = ' + wWeight.toFixed(1);
      if (done && found) {
        const len = pathCells.length - 1;
        cap.textContent = `Goal reached: path of ${len} steps, cost ${len} (unit steps). ${mode === 'wastar' ? 'With w > 1 this path may not be the shortest.' : 'Dijkstra and A* with this heuristic return an optimal path.'}`;
      } else if (done) {
        cap.textContent = 'No route: the goal is walled off. Click a grey wall cell to open a gap, then press reset.';
      } else if (lastExp) {
        const cell = `(${lastExp.i % W}, ${Math.floor(lastExp.i / W)})`;
        cap.textContent = `Expanding cell ${cell}: g = ${lastExp.g} (cost so far), h = ${lastExp.h} (estimate to go), f = ${lastExp.f}. ${name}.`;
      } else {
        cap.textContent = 'Press play. Blue cells have been expanded; yellow cells are waiting in the frontier. Click a grey cell to add or remove a wall.';
      }
      ro.textContent = `method: ${mode === 'dijkstra' ? 'Dijkstra' : mode === 'astar' ? 'A*' : 'weighted A* (w = ' + wWeight.toFixed(1) + ')'}\n` +
        `cells expanded: ${expanded}\n` +
        `frontier (yellow): ${openSet.size}\n` +
        `cost of goal so far: ${gS[GOAL] === Infinity ? '–' : gS[GOAL]}\n` +
        (found ? `path: ${pathCells.length - 1} steps` : done ? 'path: none' : 'path: searching');
    });
    void modeSel;
  };

  /* ================= 5.3 · RRT in a plane with disc obstacles (seeded) ================= */
  AI.labs['plan-rrt'] = function (root) {
    const DISCS = [[3, 3, 1.1], [6, 5, 1.3], [3, 7.4, 0.9], [7.6, 8.0, 0.9], [5.2, 1.6, 0.8]];
    const START = [0.8, 0.8], GOAL = [9.2, 9.2], SIZE = 10, MAXN = 1500;
    const { view, ctl } = AI.labShell(root, 'RRT: a tree grows toward random samples',
      'illustrative · 10 m × 10 m plane · fixed seed so runs repeat',
      '<canvas style="height:340px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const free = (x, y) => x >= 0 && y >= 0 && x <= SIZE && y <= SIZE && DISCS.every(([cx, cy, r]) => Math.hypot(x - cx, y - cy) > r);
    const segFree = (a, b) => { for (let k = 0; k <= 12; k++) { const t = k / 12; if (!free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)) return false; } return true; };

    const eta = sliderRow(ctl, 'step size η', { min: 0.2, max: 2, step: 0.05, value: 0.8, unit: 'm' });
    const bias = sliderRow(ctl, 'goal bias', { min: 0, max: 0.3, step: 0.01, value: 0.05, dec: 2 });
    const rate = sliderRow(ctl, 'samples per second', { min: 1, max: 12, step: 1, value: 3, dec: 0 });
    const [bPlay, bStep, bReset, bSeed] = btnRow(ctl, ['▶ Play', 'Step ▸', 'Reset', 'New seed']);
    const ro = readout(ctl);
    const tick = AI.ticker();

    let seed = 7, rng, nodes, parentOf, found, pathIdx, iters, last, playing = false, dirty = true, stopped = '';
    const reset = () => {
      rng = AI.rng(seed);
      nodes = [START.slice()]; parentOf = [-1]; found = false; pathIdx = null; iters = 0; last = null; stopped = ''; dirty = true;
    };
    const pathLen = () => { let L = 0; for (let i = pathIdx[0]; i >= 0 && parentOf[i] >= 0; i = parentOf[i]) L += Math.hypot(nodes[i][0] - nodes[parentOf[i]][0], nodes[i][1] - nodes[parentOf[i]][1]); return L; };
    const iter = () => {
      if (found || stopped) return;
      iters++;
      const target = rng() < bias.get() ? GOAL.slice() : [rng() * SIZE, rng() * SIZE];
      let near = 0, nd = Infinity;
      for (let i = 0; i < nodes.length; i++) { const d = Math.hypot(nodes[i][0] - target[0], nodes[i][1] - target[1]); if (d < nd) { nd = d; near = i; } }
      const e = eta.get(), dd = Math.hypot(target[0] - nodes[near][0], target[1] - nodes[near][1]);
      const k = Math.min(e, dd) / (dd || 1);
      const nw = [nodes[near][0] + (target[0] - nodes[near][0]) * k, nodes[near][1] + (target[1] - nodes[near][1]) * k];
      const ok = segFree(nodes[near], nw);
      last = { target, near: nodes[near].slice(), nw, ok };
      if (ok) {
        nodes.push(nw); parentOf.push(near);
        const idx = nodes.length - 1;
        if (Math.hypot(GOAL[0] - nw[0], GOAL[1] - nw[1]) <= e && segFree(nw, GOAL)) {
          nodes.push(GOAL.slice()); parentOf.push(idx); found = true; pathIdx = [nodes.length - 1];
          playing = false;
        }
      }
      if (nodes.length >= MAXN && !found) { stopped = 'budget'; playing = false; }
      dirty = true;
    };

    bPlay.onclick = () => { if (found || stopped) reset(); playing = !playing; bPlay.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
    bStep.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; iter(); };
    bReset.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; reset(); };
    bSeed.onclick = () => { seed = ((seed * 7919 + 13) % 100000) + 1; playing = false; bPlay.textContent = '▶ Play'; reset(); };
    cv.onresize = () => (dirty = true);
    eta.el.addEventListener('input', () => (dirty = true));
    reset();

    const geo = () => { const w = cv.w, h = cv.h, s = Math.min(w, h) / (SIZE + 1); return { s, ox: (w - s * SIZE) / 2, oy: (h - s * SIZE) / 2 }; };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const { s, ox, oy } = geo();
      const X = (x) => ox + x * s, Y = (y) => oy + (SIZE - y) * s;
      g.fillStyle = C.bg2; g.fillRect(X(0), Y(SIZE), SIZE * s, SIZE * s);
      g.globalAlpha = 0.4; g.fillStyle = C.red;
      DISCS.forEach(([cx, cy, r]) => { g.beginPath(); g.arc(X(cx), Y(cy), r * s, 0, 7); g.fill(); });
      g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.globalAlpha = 0.55; g.lineWidth = 1;
      g.beginPath();
      for (let i = 1; i < nodes.length; i++) { const p = nodes[i], q = nodes[parentOf[i]]; g.moveTo(X(p[0]), Y(p[1])); g.lineTo(X(q[0]), Y(q[1])); }
      g.stroke(); g.globalAlpha = 1;
      g.fillStyle = C.blue;
      for (let i = 0; i < nodes.length; i++) { g.beginPath(); g.arc(X(nodes[i][0]), Y(nodes[i][1]), 1.8, 0, 7); g.fill(); }
      if (last) {
        g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X(last.near[0]), Y(last.near[1])); g.lineTo(X(last.target[0]), Y(last.target[1])); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.muted; g.beginPath(); g.arc(X(last.target[0]), Y(last.target[1]), 3, 0, 7); g.fill();
        g.strokeStyle = last.ok ? C.green : C.red; g.lineWidth = 2.5; g.beginPath(); g.moveTo(X(last.near[0]), Y(last.near[1])); g.lineTo(X(last.nw[0]), Y(last.nw[1])); g.stroke();
      }
      if (found) {
        g.strokeStyle = C.orange; g.lineWidth = 4; g.beginPath();
        for (let i = pathIdx[0]; i >= 0; i = parentOf[i]) { const p = nodes[i]; if (i === pathIdx[0]) g.moveTo(X(p[0]), Y(p[1])); else g.lineTo(X(p[0]), Y(p[1])); }
        g.stroke();
      }
      g.fillStyle = C.green; g.beginPath(); g.arc(X(START[0]), Y(START[1]), 6, 0, 7); g.fill();
      g.fillStyle = C.red; g.beginPath(); g.arc(X(GOAL[0]), Y(GOAL[1]), 6, 0, 7); g.fill();
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.fillText('start', X(START[0]) + 8, Y(START[1]) + 4); g.fillText('goal', X(GOAL[0]) - 34, Y(GOAL[1]) - 8);
    };

    AI.loop(root, (dt) => {
      if (playing && !found && !stopped) {
        const n = tick(dt, rate.get());
        for (let k = 0; k < n && !found && !stopped; k++) iter();
      }
      if (!dirty) return;
      dirty = false;
      draw();
      if (found) cap.textContent = `Found: the orange path links start to goal. Length ${F(pathLen(), 2)} m. It is not the shortest possible: RRT* or smoothing (Lesson 5.5) would shorten it.`;
      else if (stopped) cap.textContent = 'Budget of 1500 nodes used without reaching the goal. Try a larger step size or a higher goal bias, then reset.';
      else if (last) cap.textContent = `Sample (dashed target) · nearest node in blue · new node one step η = ${F(eta.get(), 2)} m toward it. ${last.ok ? 'Accepted: the segment is free (green).' : 'Rejected: the segment would cross an obstacle (red).'}`;
      else cap.textContent = 'Press play. The tree starts at the green start point; each step adds one node toward a random sample.';
      ro.textContent = `seed: ${seed}\nnodes in tree: ${nodes.length}\niterations: ${iters}\ngoal connected: ${found ? 'yes' : 'no'}` + (found ? `\npath length: ${F(pathLen(), 2)} m` : '');
    });
  };

  /* ================= 5.4 · rest-to-rest trajectories with limits ================= */
  AI.labs['plan-poly'] = function (root) {
    const SHAPES = {
      quintic: { s: (u) => 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, d: (u) => 30 * u * u - 60 * u ** 3 + 30 * u ** 4, dd: (u) => 60 * u - 180 * u * u + 120 * u ** 3 },
      cubic: { s: (u) => 3 * u * u - 2 * u ** 3, d: (u) => 6 * u - 6 * u * u, dd: (u) => 6 - 12 * u },
    };
    // peak factors of the shape, found numerically on a fine grid
    const peaks = {};
    for (const k of Object.keys(SHAPES)) {
      let pv = 0, pa = 0;
      for (let i = 0; i <= 4000; i++) { const u = i / 4000; pv = Math.max(pv, Math.abs(SHAPES[k].d(u))); pa = Math.max(pa, Math.abs(SHAPES[k].dd(u))); }
      peaks[k] = { kv: pv, ka: pa };
    }
    const { view, ctl } = AI.labShell(root, 'Rest-to-rest trajectories with limits',
      'illustrative · joint in radians · time in seconds · real-time playback',
      '<canvas style="height:360px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    let shape = 'quintic';
    selectRow(ctl, 'shape', [['quintic', 'quintic (smooth at both ends)'], ['cubic', 'cubic (acceleration jumps)']], 'quintic', (v) => { shape = v; dirty = true; });
    const q0 = sliderRow(ctl, 'start q₀', { min: -3, max: 3, step: 0.05, value: 0, unit: 'rad' });
    const qf = sliderRow(ctl, 'end q_f', { min: -3, max: 3, step: 0.05, value: 1, unit: 'rad' });
    const T = sliderRow(ctl, 'duration T', { min: 0.5, max: 6, step: 0.1, value: 2, unit: 's', dec: 1 });
    const vmax = sliderRow(ctl, 'limit |q̇| max', { min: 0.1, max: 3, step: 0.05, value: 1, unit: 'rad/s' });
    const amax = sliderRow(ctl, 'limit |q̈| max', { min: 0.2, max: 10, step: 0.1, value: 2, unit: 'rad/s²', dec: 1 });
    const speed = sliderRow(ctl, 'playback speed', { min: 0.25, max: 2, step: 0.05, value: 1, unit: '×' });
    const [bPlay, bRestart] = btnRow(ctl, ['▶ Play', 'Restart']);
    const ro = readout(ctl);

    let tt = 0, playing = false, dirty = true;
    const sh = () => SHAPES[shape];
    const at = (t) => {
      const Tt = T.get(), dq = qf.get() - q0.get(), u = Math.min(1, Math.max(0, t / Tt));
      return { q: q0.get() + dq * sh().s(u), v: (dq * sh().d(u)) / Tt, a: (dq * sh().dd(u)) / (Tt * Tt) };
    };
    bPlay.onclick = () => { if (tt >= T.get()) tt = 0; playing = !playing; bPlay.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
    bRestart.onclick = () => { tt = 0; playing = false; bPlay.textContent = '▶ Play'; dirty = true; };
    [q0, qf, T, vmax, amax].forEach((s) => s.el.addEventListener('input', () => (dirty = true)));
    cv.onresize = () => (dirty = true);

    const drawPanel = (g, C, top, h, W, title, unit, f, lo, hi, limits) => {
      const m = { l: 56, r: 10 }, pw = W - m.l - m.r;
      const Y = (y) => top + (1 - (y - lo) / (hi - lo)) * h;
      const X = (t) => m.l + (t / T.get()) * pw;
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(m.l, top, pw, h);
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.textAlign = 'right';
      g.fillText(AI.fmt(+hi.toFixed(2), 2), m.l - 5, top + 10); g.fillText(AI.fmt(+lo.toFixed(2), 2), m.l - 5, top + h);
      g.textAlign = 'left'; g.fillText(title + ' (' + unit + ')', m.l + 4, top + 12);
      if (lo < 0 && hi > 0) { g.strokeStyle = C.border; g.beginPath(); g.moveTo(m.l, Y(0)); g.lineTo(m.l + pw, Y(0)); g.stroke(); }
      if (limits) limits.forEach(([y, col]) => {
        g.setLineDash([5, 4]); g.strokeStyle = col; g.beginPath(); g.moveTo(m.l, Y(y)); g.lineTo(m.l + pw, Y(y)); g.stroke(); g.setLineDash([]);
      });
      g.strokeStyle = C.blue; g.lineWidth = 2.2; g.beginPath();
      for (let i = 0; i <= 240; i++) { const t = (T.get() * i) / 240, p = f(t); if (i === 0) g.moveTo(X(t), Y(p)); else g.lineTo(X(t), Y(p)); }
      g.stroke();
      const tc = Math.min(tt, T.get()), pc = f(tc);
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X(tc), top); g.lineTo(X(tc), top + h); g.stroke();
      g.fillStyle = C.orange; g.beginPath(); g.arc(X(tc), Y(pc), 4.5, 0, 7); g.fill();
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, W = cv.w, H = cv.h;
      g.clearRect(0, 0, W, H);
      const dq = Math.abs(qf.get() - q0.get()), Tt = T.get(), pk = peaks[shape];
      const vpk = pk.kv * dq / Tt, apk = pk.ka * dq / (Tt * Tt);
      const span = Math.max(0.2, dq);
      const lo0 = Math.min(q0.get(), qf.get()) - 0.1 * span - 0.05, hi0 = Math.max(q0.get(), qf.get()) + 0.1 * span + 0.05;
      const vLim = Math.max(vpk, vmax.get()) * 1.15, aLim = Math.max(apk, amax.get()) * 1.15;
      const ph = (H - 30) / 3 - 14;
      drawPanel(g, C, 6, ph, W, 'position q', 'rad', (t) => at(t).q, lo0, hi0, null);
      drawPanel(g, C, 6 + ph + 14 + 4, ph, W, 'velocity q̇', 'rad/s', (t) => at(t).v, -vLim, vLim, [[vmax.get(), C.red], [-vmax.get(), C.red]]);
      drawPanel(g, C, 6 + 2 * (ph + 18), ph, W, 'acceleration q̈', 'rad/s²', (t) => at(t).a, -aLim, aLim, [[amax.get(), C.red], [-amax.get(), C.red]]);
    };

    AI.loop(root, (dt) => {
      if (playing) {
        tt += dt * speed.get();
        if (tt >= T.get()) { tt = T.get(); playing = false; bPlay.textContent = '▶ Play'; }
        dirty = true;
      }
      if (!dirty) return;
      dirty = false;
      draw();
      const dq = qf.get() - q0.get(), Tt = T.get(), pk = peaks[shape], now = at(Math.min(tt, Tt));
      const vpk = pk.kv * Math.abs(dq) / Tt, apk = pk.ka * Math.abs(dq) / (Tt * Tt);
      const vOver = vpk > vmax.get(), aOver = apk > amax.get();
      cap.textContent = `${shape === 'quintic' ? 'Quintic' : 'Cubic'} move: the top panel is the position, the middle the velocity and the bottom the acceleration. Red dashed lines are the limits. ${shape === 'cubic' ? 'The cubic acceleration jumps at the ends.' : 'The quintic acceleration is zero at both ends.'}`;
      ro.textContent =
        `Δq = ${F(dq, 3)} rad\n` +
        `peak |q̇| = ${F(vpk, 3)} rad/s   limit ${F(vmax.get(), 2)} → ${vOver ? 'EXCEEDS' : 'ok'}\n` +
        `peak |q̈| = ${F(apk, 3)} rad/s²  limit ${F(amax.get(), 1)} → ${aOver ? 'EXCEEDS' : 'ok'}\n` +
        `at t = ${F(Math.min(tt, Tt), 2)} s: q = ${F(now.q, 3)} rad, q̇ = ${F(now.v, 3)} rad/s, q̈ = ${F(now.a, 3)} rad/s²`;
    });
  };

  /* ================= 5.5 · trajectory optimisation: smoothness + clearance ================= */
  AI.labs['plan-trajopt'] = function (root) {
    const N = 24, OBS = { cx: 5, cy: 0.3, R: 1.0 }, X0 = [0, 0], X1 = [10, 0];
    const { view, ctl } = AI.labShell(root, 'Bending a path away from an obstacle',
      'illustrative · 2-D plane, metres · gradient descent on the cost J',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const ws = sliderRow(ctl, 'smoothness weight w_s', { min: 0, max: 2, step: 0.05, value: 1, dec: 2, onChange: () => refresh() });
    const wc = sliderRow(ctl, 'clearance weight w_c', { min: 0, max: 4, step: 0.1, value: 2, dec: 1, onChange: () => refresh() });
    const ds = sliderRow(ctl, 'safety margin d_safe', { min: 0.1, max: 1, step: 0.05, value: 0.5, unit: 'm' , onChange: () => refresh() });
    const al = sliderRow(ctl, 'step size α', { min: 0.01, max: 0.1, step: 0.005, value: 0.04, dec: 3 });
    const sp = sliderRow(ctl, 'iterations per second', { min: 1, max: 12, step: 1, value: 6, dec: 0 });
    const [bPlay, bStep, bReset] = btnRow(ctl, ['▶ Play', 'Step ▸', 'Reset']);
    const ro = readout(ctl);
    const tick = AI.ticker();

    let P = [], J = 0, iter = 0, backtracks = 0, playing = false, dirty = true, stalled = 0, settled = false;
    const gapOf = (p) => Math.hypot(p[0] - OBS.cx, p[1] - OBS.cy) - OBS.R;
    const cost = (Q) => {
      let s = 0, c = 0;
      for (let i = 1; i < N - 1; i++) {
        const ax = Q[i + 1][0] - 2 * Q[i][0] + Q[i - 1][0], ay = Q[i + 1][1] - 2 * Q[i][1] + Q[i - 1][1];
        s += ax * ax + ay * ay;
        const v = Math.max(0, ds.get() - gapOf(Q[i]));
        c += v * v;
      }
      return ws.get() * s + wc.get() * c;
    };
    const grad = () => {
      const G = P.map(() => [0, 0]), eps = 1e-5;
      for (let i = 1; i < N - 1; i++) for (let k = 0; k < 2; k++) {
        const old = P[i][k];
        P[i][k] = old + eps; const jp = cost(P);
        P[i][k] = old - eps; const jm = cost(P);
        P[i][k] = old;
        G[i][k] = (jp - jm) / (2 * eps);
      }
      return G;
    };
    const init = () => {
      P = []; for (let i = 0; i < N; i++) { const t = i / (N - 1); P.push([X0[0] + (X1[0] - X0[0]) * t, X0[1] + (X1[1] - X0[1]) * t]); }
      iter = 0; backtracks = 0; stalled = 0; settled = false; J = cost(P); dirty = true;
    };
    const iterate = () => {
      if (settled) return;
      const G = grad(); const prevJ = J;
      let a = al.get();
      for (let tries = 0; tries < 8; tries++) {
        const Q = P.map((p, i) => {
          if (i === 0 || i === N - 1) return p;
          let dx = -a * G[i][0], dy = -a * G[i][1];
          const m = Math.hypot(dx, dy); if (m > 0.25) { dx *= 0.25 / m; dy *= 0.25 / m; }
          return [p[0] + dx, p[1] + dy];
        });
        const Jn = cost(Q);
        if (Jn <= J) { P = Q; J = Jn; break; }
        a *= 0.5; backtracks++;
      }
      iter++;
      stalled = prevJ - J < 1e-9 ? stalled + 1 : 0;
      if (stalled > 6) { settled = true; playing = false; }
      dirty = true;
    };
    const refresh = () => { J = cost(P); dirty = true; };
    init();

    bPlay.onclick = () => { if (settled) init(); playing = !playing; bPlay.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
    bStep.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; iterate(); };
    bReset.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; init(); };
    cv.onresize = () => (dirty = true);

    const minGap = () => Math.min(...P.slice(1, N - 1).map(gapOf));
    const draw = () => {
      const C = AI.col(), g = cv.ctx, W = cv.w, H = cv.h;
      g.clearRect(0, 0, W, H);
      const s = Math.min(W / 11, H / 4), X = (x) => W / 2 + (x - 5) * s, Y = (y) => H / 2 - (y - 0.2) * s;
      g.globalAlpha = 0.35; g.fillStyle = C.red; g.beginPath(); g.arc(X(OBS.cx), Y(OBS.cy), OBS.R * s, 0, 7); g.fill(); g.globalAlpha = 1;
      g.strokeStyle = C.red; g.lineWidth = 1.5; g.beginPath(); g.arc(X(OBS.cx), Y(OBS.cy), OBS.R * s, 0, 7); g.stroke();
      g.setLineDash([5, 5]); g.strokeStyle = C.orange; g.beginPath(); g.arc(X(OBS.cx), Y(OBS.cy), (OBS.R + ds.get()) * s, 0, 7); g.stroke(); g.setLineDash([]);
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X(X0[0]), Y(X0[1])); g.lineTo(X(X1[0]), Y(X1[1])); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.blue; g.lineWidth = 3; g.beginPath();
      P.forEach((p, i) => { if (i === 0) g.moveTo(X(p[0]), Y(p[1])); else g.lineTo(X(p[0]), Y(p[1])); });
      g.stroke();
      g.fillStyle = C.blue; P.forEach((p) => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.5, 0, 7); g.fill(); });
      g.fillStyle = C.green; g.beginPath(); g.arc(X(X0[0]), Y(X0[1]), 6, 0, 7); g.fill();
      g.fillStyle = C.red; g.beginPath(); g.arc(X(X1[0]), Y(X1[1]), 6, 0, 7); g.fill();
      g.fillStyle = C.muted; g.font = '11px ' + MONO; g.fillText('dashed grey: start path · dashed orange: safety margin', 8, H - 8);
    };

    AI.loop(root, (dt) => {
      if (playing && !settled) {
        const n = tick(dt, sp.get());
        for (let k = 0; k < n && !settled; k++) iterate();
        if (n > 0) dirty = true;
        if (settled) { playing = false; bPlay.textContent = '▶ Play'; }
      }
      if (!dirty) return;
      dirty = false;
      draw();
      const mg = minGap();
      cap.textContent = `iteration ${iter} · cost J = ${F(J, 3)} · smallest clearance gap = ${F(mg, 3)} m (margin ${F(ds.get(), 2)} m) · backtracks ${backtracks}` +
        (mg < 0 ? ' · the path is inside the obstacle' : mg < ds.get() ? ' · inside the safety margin' : '') +
        (settled ? ' · settled (a local minimum, not necessarily the global best)' : '');
      ro.textContent = `J smoothness + clearance terms\n` +
        `cost J: ${F(J, 4)}\nsmallest gap: ${F(mg, 3)} m\nbacktracks (halved α): ${backtracks}`;
    });
  };

  /* ================= 5.6 · STRIPS-style task planning ================= */
  AI.labs['plan-strips'] = function (root) {
    const LOCS = ['table', 'shelf', 'sink'];
    const OPS = [];
    LOCS.forEach((a) => LOCS.forEach((b) => { if (a !== b) OPS.push({ n: `move(${a}, ${b})`, pre: [`at_robot(${a})`], add: [`at_robot(${b})`], del: [`at_robot(${a})`] }); }));
    LOCS.forEach((l) => {
      OPS.push({ n: `pick(cup, ${l})`, pre: [`at_robot(${l})`, `at_cup(${l})`, 'handempty'], add: ['holding_cup'], del: [`at_cup(${l})`, 'handempty'] });
      OPS.push({ n: `place(cup, ${l})`, pre: [`at_robot(${l})`, 'holding_cup'], add: [`at_cup(${l})`, 'handempty'], del: ['holding_cup'] });
    });
    const STARTS = {
      table: { label: 'robot and cup on the table', facts: ['at_robot(table)', 'at_cup(table)', 'handempty'] },
      sink: { label: 'robot at shelf, cup in sink', facts: ['at_robot(shelf)', 'at_cup(sink)', 'handempty'] },
    };
    const GOALS = {
      shelf: { label: 'cup on the shelf', facts: ['at_cup(shelf)'] },
      tt: { label: 'cup on table, robot at table', facts: ['at_cup(table)', 'at_robot(table)'] },
      sink: { label: 'cup in the sink', facts: ['at_cup(sink)'] },
    };
    const UNIVERSE = [...new Set(OPS.flatMap((o) => [...o.pre, ...o.add, ...o.del]))].sort();
    const applies = (s, o) => o.pre.every((p) => s.has(p));
    const apply = (s, o) => { const n = new Set(s); o.del.forEach((d) => n.delete(d)); o.add.forEach((a) => n.add(a)); return n; };
    const keyOf = (s) => [...s].sort().join('|');
    const solve = (start, goal) => {
      const q = [{ s: start, plan: [], states: [start] }], seen = new Set([keyOf(start)]);
      let explored = 0;
      while (q.length) {
        const cur = q.shift(); explored++;
        if (goal.every((f) => cur.s.has(f))) return { plan: cur.plan, states: cur.states, explored };
        for (let i = 0; i < OPS.length; i++) {
          if (!applies(cur.s, OPS[i])) continue;
          const ns = apply(cur.s, OPS[i]), k = keyOf(ns);
          if (!seen.has(k)) { seen.add(k); q.push({ s: ns, plan: [...cur.plan, i], states: [...cur.states, ns] }); }
        }
      }
      return null;
    };

    const { view, ctl } = AI.labShell(root, 'Task planning: pick, move, place',
      'illustrative · symbolic facts · breadth-first search gives a shortest plan',
      '<canvas style="height:260px"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const startSel = selectRow(ctl, 'start', [['table', STARTS.table.label], ['sink', STARTS.sink.label]], 'table', () => replan());
    const goalSel = selectRow(ctl, 'goal', [['shelf', GOALS.shelf.label], ['tt', GOALS.tt.label], ['sink', GOALS.sink.label]], 'shelf', () => replan());
    const speed = sliderRow(ctl, 'speed', { min: 0.25, max: 2, step: 0.05, value: 1, unit: '×' });
    const [bPlay, bStep, bRestart] = btnRow(ctl, ['▶ Play', 'Step ▸', 'Restart']);
    const ro = readout(ctl);

    let plan = null, states = [], explored = 0, k = 0, p = 0, playing = false, dirty = true;
    const STEP_T = 2.0;
    function replan() {
      const r = solve(new Set(STARTS[startSel.value].facts), GOALS[goalSel.value].facts);
      plan = r ? r.plan : null; states = r ? r.states : [new Set(STARTS[startSel.value].facts)]; explored = r ? r.explored : 0;
      k = 0; p = 0; playing = false; bPlay.textContent = '▶ Play'; dirty = true;
    }
    bPlay.onclick = () => { if (!plan || k >= plan.length) { k = 0; p = 0; } playing = !playing; bPlay.textContent = playing ? '❚❚ Pause' : '▶ Play'; };
    bStep.onclick = () => { playing = false; bPlay.textContent = '▶ Play'; if (plan && k < plan.length) { p = 0; k++; dirty = true; } };
    bRestart.onclick = () => { k = 0; p = 0; playing = false; bPlay.textContent = '▶ Play'; dirty = true; };
    cv.onresize = () => (dirty = true);
    replan();

    const ROOM = { table: 0.18, shelf: 0.5, sink: 0.82 };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, W = cv.w, H = cv.h;
      g.clearRect(0, 0, W, H);
      const done = !plan || k >= plan.length;
      const sk = states[Math.min(k, states.length - 1)] || new Set(), sn = states[Math.min(k + 1, states.length - 1)] || sk;
      const e = done ? 1 : AI.smooth(p);
      // rooms
      LOCS.forEach((l) => {
        const cx = ROOM[l] * W, bw = W * 0.24, bh = H * 0.5, top = H * 0.18;
        g.fillStyle = C.bg2; g.strokeStyle = C.border; g.lineWidth = 1.5;
        g.fillRect(cx - bw / 2, top, bw, bh); g.strokeRect(cx - bw / 2, top, bw, bh);
        g.fillStyle = C.muted; g.font = '12px ' + MONO; g.textAlign = 'center'; g.fillText(l, cx, top + bh + 16);
      });
      // robot location: moves smoothly during a move action
      const rs = [...sk].find((f) => f.startsWith('at_robot(')), rn = [...sn].find((f) => f.startsWith('at_robot('));
      const locOf = (f) => f.slice(9, -1);
      const isMove = !done && plan && OPS[plan[k]].n.startsWith('move');
      const rFrom = locOf(rs), rTo = locOf(rn);
      const rx = isMove ? ROOM[rFrom] + (ROOM[rTo] - ROOM[rFrom]) * e : ROOM[rFrom];
      const robotX = rx * W, robotY = H * 0.36;
      g.fillStyle = C.blue; g.globalAlpha = 0.9; g.fillRect(robotX - 30, robotY - 16, 60, 32); g.globalAlpha = 1;
      g.fillStyle = C.fg; g.font = 'bold 12px ' + MONO; g.textAlign = 'center'; g.fillText('robot', robotX, robotY + 4);
      // cup: in a room, or in the gripper
      const holdBefore = sk.has('holding_cup'), holdAfter = sn.has('holding_cup');
      const cupRoomBefore = [...sk].find((f) => f.startsWith('at_cup(')), cupRoomAfter = [...sn].find((f) => f.startsWith('at_cup('));
      let cx, cy = H * 0.58;
      const gripper = { x: robotX, y: robotY + 30 };
      if (!done && plan && OPS[plan[k]].n.startsWith('pick')) {
        const room = ROOM[locOf(cupRoomBefore)] * W; cx = room + (gripper.x - room) * e; cy = cy + (gripper.y - cy) * e;
      } else if (!done && plan && OPS[plan[k]].n.startsWith('place')) {
        const room = ROOM[locOf(cupRoomAfter)] * W; cx = gripper.x + (room - gripper.x) * e; cy = gripper.y + (cy - gripper.y) * e;
      } else if (holdAfter || holdBefore) { cx = gripper.x; cy = gripper.y; }
      else { cx = ROOM[locOf(cupRoomBefore)] * W; }
      g.fillStyle = C.orange; g.beginPath(); g.arc(cx, cy, 11, 0, 7); g.fill();
      g.fillStyle = C.fg; g.font = '11px ' + MONO; g.fillText('cup', cx, cy + 4);
      g.textAlign = 'left';
    };

    AI.loop(root, (dt) => {
      if (playing && plan && k < plan.length) {
        p += (dt * speed.get()) / STEP_T;
        if (p >= 1) { p = 0; k++; if (k >= plan.length) { playing = false; bPlay.textContent = '▶ Play'; } }
        dirty = true;
      }
      if (!dirty) return;
      dirty = false;
      draw();
      if (!plan) {
        cap.textContent = 'No plan: the goal cannot be reached with these actions from this start.';
        ro.textContent = 'no plan found';
        return;
      }
      if (plan.length === 0) {
        cap.textContent = 'The start state already satisfies the goal: the empty plan is enough.';
      } else if (k >= plan.length) {
        cap.textContent = `Plan finished in ${plan.length} actions. Every precondition held when its action ran.`;
      } else {
        const o = OPS[plan[k]];
        const cur = states[Math.min(k, states.length - 1)] || new Set();
        const pre = o.pre.map((f) => (cur.has(f) ? f + ' ✓' : f + ' ✗')).join(', ');
        cap.textContent = `Step ${k + 1} of ${plan.length}: ${o.n}. Preconditions: ${pre}. ${p < 0.5 ? 'Checking the preconditions.' : 'Applying the effects: delete ' + (o.del.join(', ') || 'nothing') + '; add ' + o.add.join(', ') + '.'}`;
      }
      const shown = states[Math.min(k, states.length - 1)] || new Set();
      const after = states[Math.min(k + 1, states.length - 1)] || shown;
      const transitioning = plan && k < plan.length && p >= 0.5;
      const lines = UNIVERSE.map((f) => {
        const inK = shown.has(f), inN = after.has(f);
        if (transitioning && inK && !inN) return '✗ ' + f + '   (deleted)';
        if (transitioning && !inK && inN) return '+ ' + f + '   (added)';
        return (inK ? '● ' : '  ') + f;
      });
      ro.textContent = (plan ? 'plan: ' + plan.map((i, j) => (j === k ? '▸ ' : '  ') + (j + 1) + '. ' + OPS[i].n).join('\n') + '\n' : '') +
        'facts (● true):\n' + lines.join('\n') + `\n\nstates explored by search: ${explored}`;
    });
  };
})();
