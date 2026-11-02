/* Reinforcement Learning · Module 8 (Advanced topics) labs. All are illustrative toys, not experiments; each caption states its simplifications.
 * intrinsic-grid   <div data-lab="intrinsic-grid"></div>   count-based exploration bonus on a 9×9 grid with a far goal
 * hier-rooms       <div data-lab="hier-rooms"></div>       options (doors, goal) chosen by a high-level policy in two rooms
 * ma-games         <div data-lab="ma-games"></div>         two independent learners in a repeated 2×2 game
 * offline-pess     <div data-lab="offline-pess"></div>     a fixed dataset: naive versus pessimistic action choice
 * irl-arc          <div data-lab="irl-arc"></div>          the reward directions consistent with an expert's choices
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';

  /* ---------- shared helpers (same shape as the other lab files) ---------- */
  const slider = (ctl, name, min, max, step, val, unit, dec = 2) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = AI.fmt(+inp.value, dec) + (unit ? ' ' + unit : ''));
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return { get: () => +inp.value, el: inp };
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const readout = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; d.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6'; ctl.appendChild(d); return d; };
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  /* transport: pause/play, step (one event per click while paused), reset and a speed multiplier. */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 2, 0.05, speed, '×', 2).get;
    pp.onclick = () => { T.playing = !T.playing; pp.textContent = T.playing ? 'Pause' : 'Play'; };
    st.onclick = () => { T.playing = false; pp.textContent = 'Play'; T.pending += 1; };
    rs.onclick = () => { T.pending = 0; onReset(); };
    return T;
  };

  /* redraw only when the key (state, size or theme) changes */
  const keyOf = (canvases, extra) => extra + '|' + canvases.map((c) => c.w + 'x' + c.h).join(',') + '|' + document.documentElement.dataset.theme;

  /* ==========================================================================
     1. intrinsic-grid — count-based bonus β/√(N+1) for moving into a cell
     ========================================================================== */
  AI.labs['intrinsic-grid'] = function (root) {
    const W = 9, H = 9, START = [0, 0], GOAL = [8, 8], EPS = 0.1, RATE = 4;
    const v = AI.labShell(root, 'Count-based exploration on a 9×9 grid',
      'Illustrative toy. The only environment reward is +1 at the goal G (8,8). β = 0 is a plain random walk.',
      '<canvas style="height:360px"></canvas><div class="readout cap" style="margin-top:8px"></div>', '');
    const cv = v.view.querySelector('canvas'), cap = v.view.querySelector('.cap');
    cap.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6';
    const c = AI.canvas(cv);
    const ctl = v.ctl;
    const beta = slider(ctl, 'Bonus scale β', 0, 2, 0.05, 1.0, '', 2);
    const T = transport(ctl, () => reset(), 1);
    const info = readout(ctl);
    info.innerHTML = `<p style="margin:0;color:var(--muted)">Each move, the agent looks at the four neighbours and picks the one with the largest bonus β/√(N+1), where N is how often that cell has been visited so far. With probability ε = ${EPS} it moves at random. Ties are broken at random with a fixed seed (7).</p>`;

    const idx = (p) => p[1] * W + p[0];
    let N, pos, steps, rnd, goalStep, goalCount, tick = AI.ticker();
    const reset = () => {
      N = new Array(W * H).fill(0); pos = START.slice(); steps = 0; goalStep = null; goalCount = 0; rnd = AI.rng(7);
      N[idx(pos)] = 1;
    };
    reset();

    const MOVES = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const doStep = () => {
      const opts = MOVES.map(([dx, dy]) => [pos[0] + dx, pos[1] + dy]).filter(([x, y]) => x >= 0 && y >= 0 && x < W && y < H);
      let next;
      if (beta.get() === 0 || rnd() < EPS) next = opts[Math.floor(rnd() * opts.length)];
      else {
        let best = -Infinity, cands = [];
        for (const p of opts) {
          const val = beta.get() / Math.sqrt(N[idx(p)] + 1);
          if (val > best + 1e-12) { best = val; cands = [p]; }
          else if (Math.abs(val - best) <= 1e-12) cands.push(p);
        }
        next = cands[Math.floor(rnd() * cands.length)];
      }
      pos = next; N[idx(pos)] += 1; steps += 1;
      if (pos[0] === GOAL[0] && pos[1] === GOAL[1]) {
        if (goalStep === null) goalStep = steps;
        goalCount += 1; pos = START.slice(); N[idx(pos)] += 1;
      }
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const cell = Math.min((c.w - 20) / W, (c.h - 20) / H), ox = (c.w - cell * W) / 2, oy = 10;
      let maxN = 1;
      for (const n of N) if (n > maxN) maxN = n;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const n = N[y * W + x], t = Math.min(1, Math.log(1 + n) / Math.log(1 + Math.max(8, maxN)));
        const px = ox + x * cell, py = oy + (H - 1 - y) * cell;
        g.fillStyle = n ? AI.mix(C.bg2, C.blue, 0.15 + 0.75 * t) : C.bg2;
        g.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        if (cell > 30 && n) { g.fillStyle = C.fg; g.font = `${Math.round(cell * 0.3)}px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(n), px + cell / 2, py + cell / 2); }
      }
      const gx = ox + GOAL[0] * cell, gy = oy + (H - 1 - GOAL[1]) * cell;
      g.fillStyle = C.orange; g.fillRect(gx + 1, gy + 1, cell - 2, cell - 2);
      g.fillStyle = C.bg; g.font = `bold ${Math.round(cell * 0.4)}px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('G', gx + cell / 2, gy + cell / 2);
      const sx = ox + START[0] * cell, sy = oy + (H - 1 - START[1]) * cell;
      g.fillStyle = C.muted; g.font = `${Math.round(cell * 0.3)}px ${MONO}`; g.fillText('S', sx + cell * 0.2, sy + cell * 0.8);
      const ax = ox + (pos[0] + 0.5) * cell, ay = oy + (H - 1 - pos[1] + 0.5) * cell;
      g.beginPath(); g.arc(ax, ay, cell * 0.28, 0, Math.PI * 2); g.fillStyle = C.yellow; g.fill();
      g.strokeStyle = C.bg; g.lineWidth = 2; g.stroke();
      g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    };

    const caption = () => {
      const covered = N.filter((n) => n > 0).length, here = N[idx(pos)];
      const bonusNext = beta.get() / Math.sqrt(here + 1);
      cap.innerHTML = `<b>Step ${steps}</b> · agent at (${pos[0]}, ${pos[1]}) · visits to this cell so far: ${here}<br>` +
        `Next-move bonus here: β/√(N+1) = ${AI.fmt(beta.get(), 2)}/√${here + 1} = ${AI.fmt(bonusNext, 3)}<br>` +
        `Cells visited: ${covered} of ${W * H} · goal reached ${goalCount} time(s) · ` +
        (goalStep === null ? 'goal not yet found' : `goal first found at step ${goalStep}`);
    };

    let last = null;
    AI.loop(root, (dt) => {
      if (T.playing) { const n = tick(dt, RATE * T.speed()); for (let i = 0; i < n; i++) doStep(); }
      else while (T.pending > 0) { doStep(); T.pending -= 1; }
      const k = keyOf([c], steps + '|' + beta.get() + '|' + goalCount);
      if (k === last) return;
      last = k; draw(); caption();
    });
    c.onresize = () => { last = null; };
  };

  /* ==========================================================================
     2. hier-rooms — a high-level policy chooses options; each runs a low-level path
     ========================================================================== */
  AI.labs['hier-rooms'] = function (root) {
    const W = 11, H = 7, START = [1, 3], GOAL = [9, 3];
    const isWall = (x, y) => x === 5 && y !== 2 && y !== 4;
    const OPTS = [
      { name: 'door top', target: [5, 2], room: 'A', q: 0.2 },
      { name: 'door bottom', target: [5, 4], room: 'A', q: 0.5 },
      { name: 'goal', target: GOAL, room: 'any', q: 0.1 },
    ];
    const v = AI.labShell(root, 'Options in two rooms: a high-level and a low-level policy',
      'Illustrative toy. The high level picks one option by its value estimate Q; the low level takes one primitive step per tick toward that option’s target. Door options start only in room A.',
      '<canvas style="height:300px"></canvas><div class="readout cap" style="margin-top:8px"></div>', '');
    const cv = v.view.querySelector('canvas'), cap = v.view.querySelector('.cap');
    cap.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6';
    const c = AI.canvas(cv);
    const ctl = v.ctl;
    const qs = OPTS.map((o) => slider(ctl, `Q(${o.name}) — value estimate`, 0, 1, 0.05, o.q, '', 2));
    const T = transport(ctl, () => reset(), 0.6);
    const info = readout(ctl);
    info.innerHTML = '<p style="margin:0;color:var(--muted)">Changing a Q value takes effect at the next high-level decision, not in the middle of an option. Each option runs until its target is reached; the high level then decides again.</p>';

    let pos, cur, prim, decisions, pause, log, tick = AI.ticker();
    const reset = () => { pos = START.slice(); cur = null; prim = 0; decisions = 0; pause = 0; log = []; };
    reset();

    const wallOK = (x, y) => x >= 0 && y >= 0 && x < W && y < H && !isWall(x, y);
    const bfs = (from, to) => {
      const key = (p) => p[0] + ',' + p[1], prev = new Map([[key(from), null]]), q = [from];
      while (q.length) {
        const p = q.shift();
        if (p[0] === to[0] && p[1] === to[1]) break;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const n = [p[0] + dx, p[1] + dy];
          if (wallOK(n[0], n[1]) && !prev.has(key(n))) { prev.set(key(n), p); q.push(n); }
        }
      }
      const path = []; let p = to;
      while (p && key(p) !== key(from)) { path.unshift(p); p = prev.get(key(p)); }
      return path;
    };
    const available = () => OPTS.map((o, i) => ({ o, i, q: qs[i].get() })).filter(({ o }) => o.room === 'any' || (o.room === 'A' && pos[0] < 5));
    const choose = () => {
      const av = available();
      let best = av[0];
      for (const a of av) if (a.q > best.q + 1e-12) best = a;
      decisions += 1;
      cur = { name: best.o.name, target: best.o.target, path: bfs(pos, best.o.target), k: 0, q: best.q, idx: best.i };
    };

    const doStep = () => {
      if (pause > 0) return;
      if (!cur) choose();
      if (cur.path.length) { pos = cur.path.shift(); prim += 1; cur.k += 1; }
      if (!cur.path.length) {
        log.push({ name: cur.name, k: cur.k }); if (log.length > 8) log.shift();
        if (cur.target[0] === GOAL[0] && cur.target[1] === GOAL[1]) pause = 2.5;
        cur = null;
      }
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const cell = Math.min((c.w - 20) / W, (c.h - 20) / H), ox = (c.w - cell * W) / 2, oy = 10;
      const X = (x) => ox + x * cell, Y = (y) => oy + (H - 1 - y) * cell;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        g.fillStyle = isWall(x, y) ? C.border : C.bg2;
        g.fillRect(X(x) + 1, Y(y) + 1, cell - 2, cell - 2);
      }
      // room labels
      g.fillStyle = C.muted; g.font = `${Math.max(10, Math.round(cell * 0.3))}px ${MONO}`; g.textAlign = 'center';
      g.fillText('room A', X(2.5), Y(H - 1) - 1); g.fillText('room B', X(7.5), Y(H - 1) - 1);
      // doors
      [[5, 2], [5, 4]].forEach(([x, y]) => { g.fillStyle = C.yellow; g.globalAlpha = 0.35; g.fillRect(X(x) + 1, Y(y) + 1, cell - 2, cell - 2); g.globalAlpha = 1; });
      // planned path of the current option
      if (cur && cur.path.length) {
        g.strokeStyle = C.blue; g.lineWidth = 2; g.setLineDash([4, 4]); g.beginPath();
        g.moveTo(X(pos[0] + 0.5), Y(pos[1] + 0.5));
        cur.path.forEach((p) => g.lineTo(X(p[0] + 0.5), Y(p[1] + 0.5)));
        g.stroke(); g.setLineDash([]);
      }
      // goal and current target
      g.fillStyle = C.orange; g.fillRect(X(GOAL[0]) + 1, Y(GOAL[1]) + 1, cell - 2, cell - 2);
      g.fillStyle = C.bg; g.font = `bold ${Math.round(cell * 0.38)}px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText('G', X(GOAL[0]) + cell / 2, Y(GOAL[1]) + cell / 2);
      if (cur) { g.strokeStyle = C.purple; g.lineWidth = 3; g.strokeRect(X(cur.target[0]) + 3, Y(cur.target[1]) + 3, cell - 6, cell - 6); }
      g.beginPath(); g.arc(X(pos[0] + 0.5), Y(pos[1] + 0.5), cell * 0.28, 0, Math.PI * 2); g.fillStyle = C.yellow; g.fill();
      g.textBaseline = 'alphabetic'; g.textAlign = 'left';
    };

    const caption = () => {
      const av = available().map((a) => a.i);
      const lines = OPTS.map((o, i) => {
        const ok = av.includes(i), chosen = cur && cur.idx === i;
        return `<span style="color:${ok ? 'var(--fg)' : 'var(--muted)'}">${chosen ? '▶ ' : '&nbsp;&nbsp;'}${esc(o.name)}</span> Q=${AI.fmt(qs[i].get(), 2)}${ok ? '' : ' (not available here)'}`;
      }).join('<br>');
      const now = cur ? `Low level: ${cur.path.length} primitive step(s) left to reach (${cur.target[0]}, ${cur.target[1]}).` : (pause > 0 ? 'Goal reached. Restarting in a moment.' : 'High level: choosing an option.');
      const hist = log.map((l) => `${esc(l.name)} (${l.k})`).join(' → ');
      cap.innerHTML = `${lines}<br><b>${now}</b><br>Primitive steps: ${prim} · high-level decisions: ${decisions}<br>Recent options (primitive steps each): ${hist || '–'}`;
    };

    const reset2 = reset;
    let last = null;
    AI.loop(root, (dt) => {
      if (pause > 0 && T.playing) { pause -= dt; if (pause <= 0) { pause = 0; reset2(); } }
      if (T.playing) { const n = tick(dt, 2.5 * T.speed()); for (let i = 0; i < n; i++) doStep(); }
      else while (T.pending > 0) { doStep(); T.pending -= 1; }
      const k = keyOf([c], `${prim}|${decisions}|${cur ? cur.name : '-'}|${pos}|${pause > 0}|${qs.map((s) => s.get()).join(',')}`);
      if (k === last) return;
      last = k; draw(); caption();
    });
  };

  /* ==========================================================================
     3. ma-games — two independent Q-learners in a repeated 2×2 game
     ========================================================================== */
  const PRESETS = {
    coordination: { name: 'Coordination', M1: [[1, 0], [0, 2]], M2: [[1, 0], [0, 2]] },
    pennies: { name: 'Matching pennies', M1: [[1, -1], [-1, 1]], M2: [[-1, 1], [1, -1]] },
    dilemma: { name: 'Prisoner’s dilemma (A = cooperate)', M1: [[3, 0], [5, 1]], M2: [[3, 5], [0, 1]] },
  };
  AI.labs['ma-games'] = function (root) {
    const ALPHA = 0.1, EPS = 0.1, HIST = 300;
    const v = AI.labShell(root, 'Two independent learners in a repeated game',
      'Illustrative toy. Each agent runs stateless Q-learning (α = 0.1, ε = 0.1) and treats the other agent as part of the environment. Payoffs are editable.',
      '<canvas style="height:260px"></canvas><div class="readout cap" style="margin-top:8px"></div>', '');
    const cv = v.view.querySelector('canvas'), cap = v.view.querySelector('.cap');
    cap.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6';
    const c = AI.canvas(cv);
    const ctl = v.ctl;

    const pr = document.createElement('div'); pr.style.cssText = 'color:var(--muted)'; pr.textContent = 'Presets (fills the payoff tables below)'; ctl.appendChild(pr);
    const prow = rowOf(ctl);
    const presetBtns = Object.keys(PRESETS).map((k) => { const b = button(prow, PRESETS[k].name.split(' (')[0]); b.onclick = () => setPreset(k); return b; });

    // editable payoff tables: rows = agent 1's action (A, B), columns = agent 2's action (A, B)
    const grid = document.createElement('div');
    grid.style.cssText = 'display:grid;grid-template-columns:auto 1fr 1fr;gap:4px 6px;align-items:center;font-family:' + MONO + ';font-size:12px';
    ctl.appendChild(grid);
    const inputs = { M1: [], M2: [] };
    const table = (key, title) => {
      grid.insertAdjacentHTML('beforeend', `<span style="color:var(--muted);grid-column:1/4">${title}: agent 1 row, agent 2 column</span><span></span><span style="color:var(--muted)">A</span><span style="color:var(--muted)">B</span>`);
      [0, 1].forEach((i) => {
        grid.insertAdjacentHTML('beforeend', `<span style="color:var(--muted)">${i === 0 ? 'A' : 'B'}</span>`);
        inputs[key][i] = [0, 1].map(() => { const inp = document.createElement('input'); inp.type = 'number'; inp.step = '0.5'; grid.appendChild(inp); inp.oninput = () => { dirty = true; }; return inp; });
      });
    };
    let dirty = false;
    table('M1', 'Agent 1 payoff');
    table('M2', 'Agent 2 payoff');
    const readPayoffs = (key) => inputs[key].map((row) => row.map((inp) => { const x = parseFloat(inp.value); return Number.isFinite(x) ? x : 0; }));
    const setPayoffs = (P) => { ['M1', 'M2'].forEach((key) => { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) inputs[key][i][j].value = String(P[key][i][j]); }); };

    const T = transport(ctl, () => { restart(); }, 1);
    const info = readout(ctl);
    info.innerHTML = '<p style="margin:0;color:var(--muted)">Both learners update only the action they played, with the reward from the payoff table. Changing a payoff takes effect from the next round. Matching pennies and the prisoner’s dilemma are the two classic cases to compare.</p>';

    let P1, P2, Q1, Q2, rounds, hist, tick = AI.ticker(), rnd, lastP;
    const restart = () => { Q1 = [0, 0]; Q2 = [0, 0]; rounds = 0; hist = []; rnd = AI.rng(3); };
    const setPreset = (k) => { setPayoffs(PRESETS[k]); restart(); };
    setPreset('coordination');

    // probability of playing A under ε-greedy on Q (ties split evenly)
    const pA = (Q) => (Q[0] > Q[1] ? 1 - EPS / 2 : Q[0] < Q[1] ? EPS / 2 : 0.5);
    const pick = (Q) => (rnd() < pA(Q) ? 0 : 1);
    const round = () => {
      const a1 = pick(Q1), a2 = pick(Q2);
      const M1 = readPayoffs('M1'), M2 = readPayoffs('M2');
      const r1 = M1[a1][a2], r2 = M2[a1][a2];
      Q1[a1] += ALPHA * (r1 - Q1[a1]);
      Q2[a2] += ALPHA * (r2 - Q2[a2]);
      rounds += 1;
      hist.push([pA(Q1), pA(Q2)]); if (hist.length > HIST) hist.shift();
      lastP = { a1, a2, r1, r2 };
    };

    // best-reply threshold: own action A is better when the opponent's P(A) is above or below x
    const reply = (M, own) => {
      // own = 'row' for agent 1 (payoffs M[own][opp]), 'col' for agent 2 (payoffs M[opp][own])
      const u = (me, opp) => (own === 'row' ? M[me][opp] : M[opp][me]);
      const d0 = u(0, 1) - u(1, 1), d1 = u(0, 0) - u(1, 0); // d(q) = EU(A) - EU(B) at opponent P(A) = q
      if (d0 >= 0 && d1 >= 0) return 'A is the better reply whatever the other agent does (or ties)';
      if (d0 <= 0 && d1 <= 0) return 'B is the better reply whatever the other agent does (or ties)';
      const x = d0 / (d0 - d1);
      return `A is the better reply when the other agent’s P(A) is ${d1 > d0 ? 'above' : 'below'} ${AI.fmt(x, 3)}`;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const view = AI.view(c, 0, HIST, 0, 1, { l: 44, r: 12, t: 12, b: 26 });
      AI.axes(g, view, { noGrid: false });
      g.fillStyle = C.muted; g.font = `11px ${MONO}`; g.textAlign = 'left';
      g.fillText('P(A)', 4, view.pad.t + 10);
      g.textAlign = 'right'; g.fillText('round →', view.pad.l + view.pw, view.pad.t + view.ph + 22);
      const plot = (idx, color) => {
        if (hist.length < 2) return;
        g.strokeStyle = color; g.lineWidth = 2.5; g.beginPath();
        hist.forEach((h, i) => {
          const x = view.X(HIST - hist.length + i), y = view.Y(h[idx]);
          if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
        });
        g.stroke();
      };
      plot(0, C.blue); plot(1, C.orange);
    };

    const caption = () => {
      const M1 = readPayoffs('M1'), M2 = readPayoffs('M2');
      const p1 = pA(Q1), p2 = pA(Q2);
      const last = lastP ? `Last round: agent 1 played ${lastP.a1 ? 'B' : 'A'} (payoff ${lastP.r1}), agent 2 played ${lastP.a2 ? 'B' : 'A'} (payoff ${lastP.r2}).` : 'No rounds yet.';
      cap.innerHTML = `<span style="color:var(--blue)">■</span> agent 1 P(A) = ${AI.fmt(p1, 2)} · <span style="color:var(--orange)">■</span> agent 2 P(A) = ${AI.fmt(p2, 2)} · round ${rounds}<br>` +
        `Estimates Q: agent 1 (A, B) = (${AI.fmt(Q1[0], 2)}, ${AI.fmt(Q1[1], 2)}); agent 2 = (${AI.fmt(Q2[0], 2)}, ${AI.fmt(Q2[1], 2)})<br>` +
        `${last}<br>` +
        `Agent 1: ${reply(M1, 'row')}.<br>Agent 2: ${reply(M2, 'col')}.<br>` +
        `<span style="color:var(--muted)">Each agent's best reply moves whenever the other agent's P(A) moves, so both learners chase a target that keeps shifting (non-stationarity).</span>`;
    };

    let last = null;
    AI.loop(root, (dt) => {
      if (T.playing) { const n = tick(dt, 3 * T.speed()); for (let i = 0; i < n; i++) round(); }
      else while (T.pending > 0) { round(); T.pending -= 1; }
      const k = keyOf([c], `${rounds}|${dirty}|${Q1.join(',')}|${Q2.join(',')}|${readPayoffs('M1').join(',')}|${readPayoffs('M2').join(',')}`);
      if (k === last) return;
      last = k; dirty = false; draw(); caption();
    });
  };

  /* ==========================================================================
     4. offline-pess — a fixed dataset: naive versus pessimistic action choice
     ========================================================================== */
  AI.labs['offline-pess'] = function (root) {
    const Qh = [[1.0, 0.8, 3.0], [0.4, 1.6, 1.2], [1.5, 2.6, 1.8], [0.9, 0.7, 0.6]];
    const N = [[50, 20, 0], [4, 60, 2], [10, 0, 5], [12, 9, 15]];
    const ACT = ['a1', 'a2', 'a3'];
    const v = AI.labShell(root, 'Choosing actions from a fixed dataset',
      'Illustrative toy: four states, three actions, a logged dataset with counts n, and learned estimates Q̂ (given). Pessimistic score = Q̂ − κ/√(n+1). No new data is collected.',
      '<canvas style="height:440px"></canvas><div class="readout cap" style="margin-top:8px"></div>', '');
    const cv = v.view.querySelector('canvas'), cap = v.view.querySelector('.cap');
    cap.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6';
    const c = AI.canvas(cv);
    const ctl = v.ctl;
    const kap = slider(ctl, 'Pessimism κ (target)', 0, 4, 0.1, 3.0, '', 1);
    const info = readout(ctl);
    info.innerHTML = '<p style="margin:0;color:var(--muted)">Key: grey bar = naive Q̂; red segment = penalty κ/√(n+1); blue tick = pessimistic score; green box = chosen action. κ = 0 is the naive rule: pick the largest Q̂, even for an action the data never tried. Raising κ subtracts more from rarely seen actions. The shown κ eases toward the slider, so changes are visible.</p>';
    const tbl = document.createElement('table'); tbl.className = 'tbl'; tbl.style.fontSize = '12px'; ctl.appendChild(tbl);
    tbl.innerHTML = '<tr><th>state</th><th>n (a1, a2, a3)</th></tr>' + N.map((r, s) => `<tr><td>s${s}</td><td>${r.join(', ')}</td></tr>`).join('');

    let shown = 3.0, lastDraw = null;
    const best = (s, k) => {
      let bi = 0, bv = -Infinity;
      for (let a = 0; a < 3; a++) { const sc = Qh[s][a] - k / Math.sqrt(N[s][a] + 1); if (sc > bv + 1e-12) { bv = sc; bi = a; } }
      return bi;
    };
    const naive = (s) => { let bi = 0; for (let a = 1; a < 3; a++) if (Qh[s][a] > Qh[s][bi]) bi = a; return bi; };

    const draw = () => {
      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const top = 16, rowH = (c.h - top - 18) / 4;
      const lo = -2, hi = 3.5, X = (x) => 52 + (x - lo) / (hi - lo) * (c.w - 62);
      // zero line
      g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(X(0), top); g.lineTo(X(0), top + rowH * 4); g.stroke();
      g.fillStyle = C.muted; g.font = `11px ${MONO}`; g.textAlign = 'center';
      g.fillText('0', X(0), top + rowH * 4 + 12);
      g.fillText(String(lo), X(lo), top + rowH * 4 + 12); g.fillText(String(hi), X(hi), top + rowH * 4 + 12);
      for (let s = 0; s < 4; s++) {
        const y0 = top + s * rowH, bh = rowH / 3.6;
        g.textAlign = 'left'; g.fillStyle = C.fg; g.font = `bold 12px ${MONO}`; g.fillText('s' + s, 6, y0 + 14); g.font = `11px ${MONO}`;
        for (let a = 0; a < 3; a++) {
          const yc = y0 + rowH * (0.2 + a * 0.27);
          const q = Qh[s][a], pen = shown / Math.sqrt(N[s][a] + 1), pess = q - pen;
          // naive bar
          g.fillStyle = C.border; g.fillRect(Math.min(X(0), X(q)), yc, Math.abs(X(q) - X(0)), bh);
          // penalty segment (from naive to pessimistic)
          g.fillStyle = AI.mix(C.bg2, C.red, 0.55); g.fillRect(Math.min(X(q), X(pess)), yc + bh * 0.3, Math.abs(X(q) - X(pess)), bh * 0.4);
          // pessimistic marker
          g.fillStyle = C.blue; g.fillRect(X(pess) - 2, yc - 2, 4, bh + 4);
          g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(ACT[a] + ` n=${N[s][a]}`, 46, yc + bh * 0.8);
          g.textAlign = 'left';
          const ch = best(s, shown) === a;
          if (ch) { g.strokeStyle = C.green; g.lineWidth = 2; g.strokeRect(X(lo) - 2, yc - 3, c.w - 62 + 4, bh + 6); }
        }
      }
      g.textAlign = 'left';

    };

    const caption = () => {
      const lines = [];
      for (let s = 0; s < 4; s++) {
        const nb = naive(s), pb = best(s, shown);
        const flag = (a) => (N[s][a] === 0 ? ', never in data' : '');
        lines.push(`s${s}: naive picks ${ACT[nb]} (n=${N[s][nb]}${flag(nb)}); pessimistic picks ${ACT[pb]} (n=${N[s][pb]})${nb !== pb ? ' <b>— differs</b>' : ''}`);
      }
      cap.innerHTML = `κ shown = ${AI.fmt(shown, 2)} (target ${AI.fmt(kap.get(), 1)})<br>${lines.join('<br>')}`;
    };

    let last = null;
    AI.loop(root, (dt) => {
      const target = kap.get();
      shown += (target - shown) * Math.min(1, dt * 2.5);
      if (Math.abs(target - shown) < 1e-3) shown = target;
      const k = keyOf([c], shown.toFixed(3));
      if (k === last) return;
      last = k; draw(); caption();
    });
  };

  /* ==========================================================================
     5. irl-arc — reward directions consistent with the expert's choices
     ========================================================================== */
  AI.labs['irl-arc'] = function (root) {
    const P = [[0.9, 0.2], [0.2, 0.9], [0.6, 0.6], [0.85, 0.55], [0.3, 0.3]];
    const NAMES = ['a', 'b', 'c', 'd', 'e'];
    const v = AI.labShell(root, 'Which reward explains the expert?',
      'Illustrative toy. Rewards are linear, r(a) = w·φ(a), with a 2-D feature φ(a). Only the direction of w is identifiable here, so the right panel is a circle of directions.',
      '<canvas style="height:300px"></canvas><div class="readout cap" style="margin-top:8px"></div>', '');
    const cv = v.view.querySelector('canvas'), cap = v.view.querySelector('.cap');
    cap.style.cssText = 'white-space:normal;font-family:inherit;font-size:13px;line-height:1.6';
    const c = AI.canvas(cv);
    const ctl = v.ctl;
    const th = slider(ctl, 'Expert’s true direction θ*', 0, 360, 1, 30, '°', 0);
    const T = transport(ctl, () => restart(), 1);
    const info = readout(ctl);
    info.innerHTML = '<p style="margin:0;color:var(--muted)">Each demonstration offers the expert a random set of 3 of the 5 actions. The expert picks its best one under its hidden w*. A learner keeps every direction that would have made each choice the best. Changing θ* restarts the demonstrations.</p>';

    let demos, rnd, tick = AI.ticker(), feas = null, feasKey = '';
    const restart = () => { demos = []; rnd = AI.rng(11); feas = null; feasKey = ''; };
    restart();
    let lastTheta = th.get();
    const expertW = () => [Math.cos(th.get() * Math.PI / 180), Math.sin(th.get() * Math.PI / 180)];
    const dot = (w, f) => w[0] * f[0] + w[1] * f[1];
    const MAX = 60;

    const demo = () => {
      if (demos.length >= MAX) return;
      const idx = [0, 1, 2, 3, 4];
      for (let i = 4; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      const offered = idx.slice(0, 3);
      const w = expertW();
      let chosen = offered[0];
      for (const a of offered) if (dot(w, P[a]) > dot(w, P[chosen]) + 1e-12) chosen = a;
      demos.push({ offered, chosen });
    };

    const feasible = () => {
      const key = demos.length + '|' + th.get();
      if (feasKey === key) return feas;
      feasKey = key;
      const out = [];
      for (let d = 0; d < 360; d++) {
        const r = d * Math.PI / 180, w = [Math.cos(r), Math.sin(r)];
        let ok = true;
        for (const dm of demos) {
          for (const b of dm.offered) if (b !== dm.chosen && dot(w, [P[dm.chosen][0] - P[b][0], P[dm.chosen][1] - P[b][1]]) < -1e-12) { ok = false; break; }
          if (!ok) break;
        }
        if (ok) out.push(d);
      }
      feas = out;
      return feas;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const half = c.w / 2, S = Math.min(half - 30, c.h - 40), ox = 20, oy = (c.h - S) / 2 + S;
      const fx = (x) => ox + x * S, fy = (y) => oy - y * S;
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(ox, oy - S, S, S);
      g.fillStyle = C.muted; g.font = `11px ${MONO}`; g.textAlign = 'left'; g.fillText('feature plane φ = (φ₁, φ₂)', ox, oy - S - 6);
      const latest = demos.length ? demos[demos.length - 1] : null;
      P.forEach((p, i) => {
        g.beginPath(); g.arc(fx(p[0]), fy(p[1]), latest && latest.chosen === i ? 7 : 5, 0, Math.PI * 2);
        g.fillStyle = latest && latest.chosen === i ? C.yellow : C.blue; g.fill();
        g.fillStyle = C.fg; g.fillText(NAMES[i], fx(p[0]) + 8, fy(p[1]) - 6);
      });
      // expert direction arrow from the centre of the plane
      const w = expertW(), cx = fx(0.5), cy = fy(0.5);
      AI.arrow(g, cx, cy, cx + w[0] * S * 0.35, cy - w[1] * S * 0.35, C.orange, 2);
      g.fillStyle = C.orange; g.fillText('w*', cx + w[0] * S * 0.37, cy - w[1] * S * 0.37);

      // right panel: circle of directions
      const rc = half + (c.w - half) / 2, rr = Math.min(c.w - half - 30, c.h - 60) / 2, rcy = c.h / 2 + 4;
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.arc(rc, rcy, rr, 0, Math.PI * 2); g.stroke();
      g.fillStyle = C.muted; g.fillText('directions of w', rc - 54, rcy - rr - 10);
      const fe = feasible();
      g.fillStyle = C.green;
      fe.forEach((d) => { const r = d * Math.PI / 180; g.beginPath(); g.arc(rc + Math.cos(r) * rr, rcy - Math.sin(r) * rr, 3, 0, Math.PI * 2); g.fill(); });
      const tr = th.get() * Math.PI / 180;
      AI.arrow(g, rc, rcy, rc + Math.cos(tr) * rr * 0.92, rcy - Math.sin(tr) * rr * 0.92, C.orange, 2);
      if (fe.length && demos.length) {
        let sx = 0, sy = 0; fe.forEach((d) => { sx += Math.cos(d * Math.PI / 180); sy += Math.sin(d * Math.PI / 180); });
        const m = Math.atan2(sy, sx);
        AI.arrow(g, rc, rcy, rc + Math.cos(m) * rr * 0.7, rcy - Math.sin(m) * rr * 0.7, C.blue, 2);
      }
      g.textAlign = 'left';
    };

    const caption = () => {
      const fe = feasible();
      const last = demos.length ? demos[demos.length - 1] : null;
      const offeredTxt = last ? last.offered.map((i) => NAMES[i]).sort().join(', ') : '–';
      const lastTxt = last ? `Latest demo: offered {${offeredTxt}}, expert chose <b>${NAMES[last.chosen]}</b>.` : 'No demonstrations yet.';
      cap.innerHTML = `<span style="color:var(--orange)">■</span> expert’s w* at θ* = ${AI.fmt(th.get(), 0)}° · <span style="color:var(--green)">●</span> directions still consistent with all demos: ${fe.length} of 360 (${AI.fmt(fe.length / 3.6, 1)}%) · <span style="color:var(--blue)">■</span> learner’s centre estimate<br>` +
        `Demonstrations: ${demos.length} (of up to ${MAX}). ${lastTxt}<br>` +
        `<span style="color:var(--muted)">The feasible set always contains θ*. Many directions explain the same choices, so the recovered reward is ambiguous. Scale is not recovered at all: 2w gives the same choices as w.</span>`;
    };

    let last = null;
    AI.loop(root, (dt) => {
      if (th.get() !== lastTheta) { lastTheta = th.get(); restart(); }
      if (T.playing) { const n = tick(dt, 0.7 * T.speed()); for (let i = 0; i < n; i++) demo(); }
      else while (T.pending > 0) { demo(); T.pending -= 1; }
      const k = keyOf([c], demos.length + '|' + th.get());
      if (k === last) return;
      last = k; draw(); caption();
    });
  };
})();
