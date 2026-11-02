/* Reinforcement Learning · Module 0 (Foundations of sequential decisions) labs. All are illustrative toys; each states its simplifications.
 * rl-loop        <div data-lab="rl-loop"></div>        agent and environment loop on a five-cell corridor
 * rl-returns     <div data-lab="rl-returns"></div>     discounted returns computed backwards through a reward list
 * rl-gridworld   <div data-lab="rl-gridworld"></div>   slippery 3x3 gridworld: exact transition probabilities and sampling
 * rl-values      <div data-lab="rl-values"></div>      policy evaluation sweeps on a three-state chain
 * rl-explore     <div data-lab="rl-explore"></div>     greedy versus epsilon-greedy on three noisy arms
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
    return { get: () => +inp.value, set: (x) => { inp.value = x; upd(); }, input: inp };
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };
  const block = (ctl) => { const d = document.createElement('div'); d.style.cssText = `font-family:${MONO};font-size:12px;line-height:1.6;color:var(--muted)`; ctl.appendChild(d); return d; };
  const mount = (root, title, sub, height) => {
    const { view, ctl } = AI.labShell(root, title, sub, `<canvas style="height:${height}px"></canvas>`, '');
    const c = AI.canvas(AI.$('canvas', view));
    return { c, ctl };
  };
  /* transport: play/pause, step, reset, speed. Play runs the lab in real time; Step requests one unit. */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, stepReq: false };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 2, 0.05, speed, '×', 2).get;
    T.set = (p) => { T.playing = p; pp.textContent = p ? 'Pause' : 'Play'; };
    pp.onclick = () => T.set(!T.playing);
    st.onclick = () => { T.set(false); T.stepReq = true; };
    rs.onclick = () => onReset();
    return T;
  };
  const C = () => AI.col();
  const txt = (ctx, s, x, y, color, size = 13, align = 'left', weight = '') => {
    ctx.save(); ctx.fillStyle = color; ctx.font = `${weight} ${size}px ${MONO}`.trim(); ctx.textAlign = align; ctx.fillText(s, x, y); ctx.restore();
  };

  /* ================= 1. agent and environment loop ================= */
  AI.labs['rl-loop'] = function (root) {
    const { c, ctl } = mount(root, 'The agent and environment loop', 'five-cell corridor · illustrative · γ = 1 (undiscounted)', 300);
    const N = 5, PIT = 0, GOAL = 4, START = 2, PH = 1.2;
    const pl = document.createElement('label');
    pl.innerHTML = '<span>Policy π</span><select><option value="random">random: 50% left, 50% right</option><option value="right">always right</option><option value="left">always left</option></select>';
    ctl.appendChild(pl);
    const sel = pl.querySelector('select');
    const info = block(ctl);
    let T = null, rng, ep, s, pos, phase, timer, a, r, done, wait, steps, ret, log;

    const startEpisode = () => {
      s = START; pos = START; phase = 0; timer = 0; a = null; r = null; done = false; wait = 0; steps = 0; ret = 0;
    };
    const reset = () => { rng = AI.rng(11); ep = 1; log = []; startEpisode(); };
    T = transport(ctl, reset, 1);
    reset();

    const pick = () => {
      if (sel.value === 'random') return rng() < 0.5 ? 'left' : 'right';
      return sel.value;
    };
    const nextPhase = () => {
      if (phase === 0) { a = pick(); phase = 1; }
      else if (phase === 1) {
        const sPrev = s, s2 = s + (a === 'right' ? 1 : -1);
        r = s2 === GOAL ? 1 : s2 === PIT ? -1 : 0;
        s = s2; steps++; ret += r;
        log.unshift(`S_t=${sPrev}  A_t=${a}  R_t+1=${r}  S_t+1=${s}`);
        if (log.length > 4) log.pop();
        phase = 2;
        if (s === PIT || s === GOAL) done = true;
      } else phase = 0;
    };
    const tick = (dt) => {
      if (done) {
        if (T.stepReq) { T.stepReq = false; ep++; startEpisode(); return; }
        if (T.playing) { wait += dt * T.speed(); if (wait >= 2.5) { ep++; startEpisode(); } }
        return;
      }
      let adv = false;
      if (T.stepReq) { T.stepReq = false; adv = true; }
      else if (T.playing) { timer += dt * T.speed(); if (timer >= PH) { timer = 0; adv = true; } }
      if (adv) nextPhase();
    };
    const caption = () => {
      if (phase === 0) return `1. Observe: the agent sees state S_t = ${s}.`;
      if (phase === 1) return `2. Act: the policy chooses A_t = ${a}.`;
      if (done) return s === GOAL ? '3. Environment: the goal is reached, R = +1. Episode ends.' : '3. Environment: the pit is reached, R = -1. Episode ends.';
      return `3. Environment: moves to S_t+1 = ${s} and returns R_t+1 = ${r}.`;
    };
    const draw = () => {
      const ctx = c.ctx, W = c.w, H = c.h, col = C();
      ctx.clearRect(0, 0, W, H);
      const cw = Math.min(110, (W - 40) / N), x0 = (W - cw * N) / 2, cy = H * 0.42;
      for (let i = 0; i < N; i++) {
        const x = x0 + i * cw;
        ctx.save(); ctx.globalAlpha = i === PIT ? 0.28 : i === GOAL ? 0.28 : 1;
        ctx.fillStyle = i === PIT ? col.red : i === GOAL ? col.green : col.bg2;
        ctx.fillRect(x + 2, cy - cw * 0.5, cw - 4, cw); ctx.restore();
        ctx.strokeStyle = col.border; ctx.strokeRect(x + 2, cy - cw * 0.5, cw - 4, cw);
        txt(ctx, `${i}`, x + cw / 2, cy + cw * 0.62 + 16, col.muted, 12, 'center');
        if (i === PIT) txt(ctx, 'pit  R=-1', x + cw / 2, cy + cw * 0.62 + 32, col.red, 11, 'center');
        if (i === GOAL) txt(ctx, 'goal  R=+1', x + cw / 2, cy + cw * 0.62 + 32, col.green, 11, 'center');
        if (i === START) txt(ctx, 'start', x + cw / 2, cy - cw * 0.62, col.muted, 11, 'center');
      }
      const ax = x0 + (pos + 0.5) * cw;
      if (phase === 1 && a) {
        const tx = ax + (a === 'right' ? 0.42 : -0.42) * cw;
        AI.arrow(ctx, ax, cy, tx, cy, col.orange, 3);
      }
      ctx.beginPath(); ctx.arc(ax, cy, cw * 0.22, 0, Math.PI * 2); ctx.fillStyle = col.blue; ctx.fill();
      if (phase === 2 && r !== null) txt(ctx, `R = ${r > 0 ? '+' : ''}${r}`, ax, cy - cw * 0.5 - 8, r > 0 ? col.green : r < 0 ? col.red : col.muted, 14, 'center', 'bold');
      txt(ctx, caption(), 14, 24, col.fg, 14);
      const steps3 = ['1 observe', '2 act', '3 respond'];
      steps3.forEach((t, i) => txt(ctx, t, W * 0.2 + i * W * 0.2, H - 14, i === phase ? col.blue : col.muted, 13, 'center', i === phase ? 'bold' : ''));
      info.innerHTML = `Episode ${ep} · step ${steps} · return so far ${ret} (undiscounted)<br>` +
        (log.length ? log.map((l) => l).join('<br>') : 'no transitions yet');
    };
    let lastKey = null;
    const key = () => [ep, s, phase, a, r, done, steps, Math.round(pos * 200), Math.round(timer * 20), Math.round(wait * 10), T.playing].join('|');
    c.onresize = () => { lastKey = null; };
    window.addEventListener('themechange', () => { lastKey = null; });
    AI.loop(root, (dt) => {
      tick(dt);
      pos += (s - pos) * Math.min(1, dt * 5);
      if (Math.abs(pos - s) < 1e-4) pos = s;
      const k = key();
      if (k !== lastKey) { draw(); lastKey = k; }
    }, { fps: 30 });
    sel.addEventListener('change', () => { startEpisode(); lastKey = null; });
  };

  /* ================= 2. rewards, returns and discounting ================= */
  AI.labs['rl-returns'] = function (root) {
    const { c, ctl } = mount(root, 'Returns, computed backwards', 'G_t = R_t+1 + γ G_t+1 · exact arithmetic · illustrative reward list', 300);
    const rl = document.createElement('label');
    rl.innerHTML = '<span>Rewards R_1, R_2, … (comma-separated, 1 to 8 numbers)</span><input type="text" spellcheck="false" value="1, 0, 2, 0, 10">';
    ctl.appendChild(rl);
    const rin = rl.querySelector('input');
    const err = document.createElement('div'); err.className = 'err'; ctl.appendChild(err);
    const g = slider(ctl, 'Discount γ', 0, 1, 0.01, 0.9, '', 2);
    const info = block(ctl);
    let R = [1, 0, 2, 0, 10], N = R.length, G = [], cur = N, wait = 0;

    const parse = (t) => {
      const parts = t.split(',').map((x) => x.trim()).filter((x) => x !== '');
      if (!parts.length || parts.length > 8) throw new Error('give between 1 and 8 numbers');
      return parts.map((p) => { if (!/^-?\d+(\.\d+)?$/.test(p)) throw new Error(`"${p}" is not a number`); return +p; });
    };
    const recompute = () => {
      const gam = g.get();
      G = new Array(N + 1).fill(0);
      for (let t = N - 1; t >= 0; t--) G[t] = R[t] + gam * G[t + 1];
    };
    const restart = () => { cur = N; wait = 0; recompute(); lastKey = null; };
    const T = transport(ctl, restart, 1);
    rin.addEventListener('input', () => {
      try { R = parse(rin.value); N = R.length; rin.classList.remove('bad'); err.textContent = ''; restart(); }
      catch (e) { rin.classList.add('bad'); err.textContent = e.message; }
    });
    g.input.addEventListener('input', restart);
    recompute();

    const draw = () => {
      const ctx = c.ctx, W = c.w, H = c.h, col = C(), gam = g.get();
      ctx.clearRect(0, 0, W, H);
      const cw = (W - 40) / N, base = H * 0.55;
      let maxAbs = 0.5;
      for (let t = 0; t < N; t++) { maxAbs = Math.max(maxAbs, Math.abs(R[t])); if (t >= cur) maxAbs = Math.max(maxAbs, Math.abs(G[t])); }
      const sc = (H * 0.36) / maxAbs;
      ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(20, base); ctx.lineTo(W - 20, base); ctx.stroke();
      for (let t = 0; t < N; t++) {
        const xc = 20 + (t + 0.5) * cw, bw = cw * 0.28;
        const hr = R[t] * sc;
        ctx.fillStyle = col.blue; ctx.fillRect(xc - bw - 2, base - hr, bw, hr);
        txt(ctx, `R=${R[t]}`, xc - bw / 2 - 2, base + (hr >= 0 ? 16 : -6) + (hr < 0 ? -0 : 0), col.blue, 11, 'center');
        if (t >= cur) {
          const hg = G[t] * sc;
          ctx.fillStyle = col.orange; ctx.fillRect(xc + 2, base - hg, bw, hg);
          txt(ctx, `G=${AI.fmt(+G[t].toFixed(4), 4)}`, xc + bw / 2 + 2, base - hg - 6, col.orange, 11, 'center');
        }
        txt(ctx, `t=${t}`, xc, H - 12, col.muted, 11, 'center');
      }
      let cap;
      if (cur >= N) cap = 'Press Play: the backward pass starts at the last reward, where nothing lies beyond it (G_N = 0).';
      else if (cur > 0) cap = `G_${cur} = R_${cur + 1} + γ·G_${cur + 1} = ${R[cur]} + ${gam}·${AI.fmt(+G[cur + 1].toFixed(4), 4)} = ${AI.fmt(+G[cur].toFixed(4), 4)}`;
      else cap = `Done: G_0 = ${AI.fmt(+G[0].toFixed(4), 4)} is the discounted return from the start.`;
      txt(ctx, cap, 14, 24, col.fg, 13);
      const horizon = gam < 1 ? AI.fmt(1 / (1 - gam), 3) : '∞ (undiscounted)';
      const w5 = [];
      for (let k = 0; k < Math.min(N, 5); k++) w5.push(AI.fmt(+Math.pow(gam, k).toFixed(4), 4));
      info.innerHTML = `γ^t weights: ${w5.join(', ')}${N > 5 ? ', …' : ''}<br>effective horizon 1/(1−γ) = ${horizon}<br>` +
        `check: Σ γ^t R_t+1 = ${AI.fmt(+R.reduce((s, r, k) => s + Math.pow(gam, k) * r, 0).toFixed(4), 4)} = G_0`;
    };
    let lastKey = null, lastCur = null;
    c.onresize = () => { lastKey = null; };
    window.addEventListener('themechange', () => { lastKey = null; });
    let timer = 0;
    const tk = AI.ticker();
    AI.loop(root, (dt) => {
      if (cur > 0 && T.playing) {
        if (cur === N && timer === 0) { /* start */ }
        timer += dt * T.speed();
        if (timer >= 1.2) { timer = 0; cur--; }
      } else if (cur > 0 && T.stepReq) { T.stepReq = false; cur--; }
      else if (cur === 0 && T.playing) { wait += dt * T.speed(); if (wait >= 3) { wait = 0; restart(); } }
      if (T.stepReq && cur === 0) { T.stepReq = false; }
      const k = [cur, N, g.get(), R.join(','), T.playing].join('|');
      if (k !== lastKey) { draw(); lastKey = k; }
    }, { fps: 30 });
    restart();
  };

  /* ================= 3. slippery gridworld: the transition model ================= */
  AI.labs['rl-gridworld'] = function (root) {
    const { c, ctl } = mount(root, 'A slippery 3×3 gridworld: the transition model', 'P(s′ | s, a): the environment decides where the agent lands · illustrative', 340);
    const GOAL = 2, PIT = 1, START = 6, TERM = [1, 2];
    const NAMES = ['top-left', 'top-middle (pit)', 'top-right (goal)', 'middle-left', 'centre', 'middle-right', 'bottom-left (start)', 'bottom-middle', 'bottom-right'];
    const DIR = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
    const SIDE = { up: ['left', 'right'], down: ['left', 'right'], left: ['up', 'down'], right: ['up', 'down'] };
    const p = slider(ctl, 'Slip probability p', 0, 0.5, 0.01, 0.2, '', 2);
    const br = rowOf(ctl);
    const ab = {};
    ['up', 'left', 'down', 'right'].forEach((d) => { ab[d] = button(br, { up: 'Action: up', left: 'Action: left', down: 'Action: down', right: 'Action: right' }[d]); });
    const cr = rowOf(ctl);
    const walkBtn = button(cr, 'Walk (repeat action)'), countBtn = button(cr, 'Count 200 samples'), resetBtn = button(cr, 'Reset');
    const spd = slider(ctl, 'Walk speed', 0.25, 2, 0.05, 1, '×', 2).get;
    const info = block(ctl);
    let s = START, pos = [2, 0], chosen = 'right', counts = new Array(9).fill(0), total = 0, rng = AI.rng(5), walking = false, walkT = 0, walkSteps = 0, lastMsg = 'Choose an action, then move or count.';

    const move = (st, d) => {
      const r = Math.floor(st / 3), cc = st % 3, [dr, dc] = DIR[d];
      const nr = Math.min(2, Math.max(0, r + dr)), nc = Math.min(2, Math.max(0, cc + dc));
      return nr * 3 + nc;
    };
    const rew = (s2) => (s2 === GOAL ? 1 : s2 === PIT ? -1 : 0);
    const outcomes = (st, a, q) => {
      if (TERM.includes(st)) return [{ s2: st, prob: 1 }];
      const m = new Map();
      [[a, 1 - q], [SIDE[a][0], q / 2], [SIDE[a][1], q / 2]].forEach(([d, pr]) => {
        const s2 = move(st, d); m.set(s2, (m.get(s2) || 0) + pr);
      });
      return [...m.entries()].map(([s2, prob]) => ({ s2, prob })).filter((o) => o.prob > 1e-12);
    };
    const sampleFrom = (outs) => {
      const u = rng(); let cum = 0;
      for (const o of outs) { cum += o.prob; if (u < cum) return o.s2; }
      return outs[outs.length - 1].s2;
    };
    const clearCounts = () => { counts = new Array(9).fill(0); total = 0; };
    const setChosen = (d) => {
      chosen = d; clearCounts();
      Object.keys(ab).forEach((k) => ab[k].classList.toggle('on', k === d));
    };
    const doMove = () => {
      if (TERM.includes(s)) return false;
      const outs = outcomes(s, chosen, p.get());
      const s2 = sampleFrom(outs);
      lastMsg = `Sampled: ${NAMES[s]} → ${NAMES[s2]}, reward R = ${rew(s2) > 0 ? '+' : ''}${rew(s2)}`;
      s = s2; clearCounts();
      if (TERM.includes(s)) { walking = false; lastMsg += '. Terminal: press Reset.'; }
      return true;
    };
    const reset = () => { s = START; pos = [2, 0]; walking = false; clearCounts(); lastMsg = 'Reset to the start. Choose an action, then move or count.'; };
    Object.keys(ab).forEach((k) => (ab[k].onclick = () => { setChosen(k); }));
    walkBtn.onclick = () => { walking = !walking; walkT = 0; walkBtn.textContent = walking ? 'Stop walk' : 'Walk (repeat action)'; if (walking && TERM.includes(s)) reset(); };
    countBtn.onclick = () => {
      if (TERM.includes(s)) { lastMsg = 'Terminal state: press Reset before counting.'; return; }
      const outs = outcomes(s, chosen, p.get());
      for (let i = 0; i < 200; i++) { counts[sampleFrom(outs)]++; total++; }
      lastMsg = `Counted 200 samples of ${chosen} from ${NAMES[s]} (the agent did not move).`;
    };
    resetBtn.onclick = () => { reset(); walkBtn.textContent = 'Walk (repeat action)'; };
    p.input.addEventListener('input', clearCounts);
    setChosen(chosen);

    const draw = () => {
      const ctx = c.ctx, W = c.w, H = c.h, col = C(), q = p.get();
      ctx.clearRect(0, 0, W, H);
      const cs = Math.min((W - 40) / 3, (H - 110) / 3), ox = (W - cs * 3) / 2, oy = 40;
      const cellXY = (st) => [ox + (st % 3 + 0.5) * cs, oy + (Math.floor(st / 3) + 0.5) * cs];
      for (let st = 0; st < 9; st++) {
        const x = ox + (st % 3) * cs, y = oy + Math.floor(st / 3) * cs;
        ctx.save(); ctx.globalAlpha = st === PIT || st === GOAL ? 0.3 : 1;
        ctx.fillStyle = st === PIT ? col.red : st === GOAL ? col.green : col.bg2;
        ctx.fillRect(x + 2, y + 2, cs - 4, cs - 4); ctx.restore();
        ctx.strokeStyle = col.border; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
        const tag = st === PIT ? 'PIT −1' : st === GOAL ? 'GOAL +1' : st === START ? 'START' : '';
        if (tag) txt(ctx, tag, x + cs / 2, y + 16, col.muted, 11, 'center');
      }
      const outs = TERM.includes(s) ? [] : outcomes(s, chosen, q);
      outs.forEach((o) => {
        const [x, y] = cellXY(o.s2);
        ctx.beginPath(); ctx.arc(x, y, cs * (0.12 + 0.14 * o.prob), 0, Math.PI * 2);
        ctx.fillStyle = col.orange; ctx.globalAlpha = 0.25 + 0.5 * o.prob; ctx.fill(); ctx.globalAlpha = 1;
        txt(ctx, o.prob.toFixed(2), x, y + cs * 0.36, col.orange, 12, 'center', 'bold');
      });
      const pr = Math.floor(pos[0]), pc = pos[1];
      const px = ox + (pc + 0.5) * cs, py = oy + (pr + 0.5) * cs;
      ctx.beginPath(); ctx.arc(px, py, cs * 0.17, 0, Math.PI * 2); ctx.fillStyle = col.blue; ctx.fill();
      const [sx, sy] = cellXY(s);
      const dd = DIR[chosen];
      AI.arrow(ctx, sx, sy, sx + dd[1] * cs * 0.38, sy + dd[0] * cs * 0.38, col.yellow, 3);
      txt(ctx, `Action: ${chosen}   Markov: these probabilities depend only on the cell and the action.`, 14, 22, col.fg, 13);
      txt(ctx, `intended ${AI.fmt(1 - q, 2)} · each side slip ${AI.fmt(q / 2, 2)} · into a wall: stays put`, 14, H - 12, col.muted, 12);
      let lines = `Current cell: ${NAMES[s]}<br>`;
      if (TERM.includes(s)) lines += 'Episode over (terminal).<br>';
      else {
        lines += 'exact P(s′|s,a) vs observed:<br>';
        outs.forEach((o) => {
          const obs = total ? (counts[o.s2] / total).toFixed(3) : '–';
          lines += `→ ${NAMES[o.s2]}: exact ${o.prob.toFixed(3)} · observed ${obs}<br>`;
        });
        if (total) lines += `samples so far: ${total}<br>`;
      }
      lines += lastMsg;
      info.innerHTML = lines;
    };

    let timer = 0, lastKey = null;
    c.onresize = () => { lastKey = null; };
    window.addEventListener('themechange', () => { lastKey = null; });
    AI.loop(root, (dt) => {
      if (walking) {
        timer += dt * spd();
        if (timer >= 1.0) { timer = 0; walkSteps++; if (!doMove() || walkSteps > 40) walking = false; if (!walking) { walkBtn.textContent = 'Walk (repeat action)'; } }
      }
      const tr = Math.floor(s / 3), tc = s % 3;
      pos[0] += (tr - pos[0]) * Math.min(1, dt * 4);
      pos[1] += (tc - pos[1]) * Math.min(1, dt * 4);
      if (Math.abs(pos[0] - tr) < 1e-3) pos[0] = tr;
      if (Math.abs(pos[1] - tc) < 1e-3) pos[1] = tc;
      const k = [s, chosen, Math.round(p.get() * 100), total, counts.join(','), pos[0].toFixed(3), pos[1].toFixed(3), walking, lastMsg].join('|');
      if (k !== lastKey) { draw(); lastKey = k; }
    }, { fps: 30 });
  };

  /* ================= 4. policy evaluation: values from sweeps ================= */
  AI.labs['rl-values'] = function (root) {
    const { c, ctl } = mount(root, 'Policy evaluation: values from sweeps', 'three-state chain · fixed policy · exact answer as dashed marks · illustrative', 340);
    const P = slider(ctl, 'Probability of going right, p', 0, 1, 0.05, 0.5, '', 2);
    const G = slider(ctl, 'Discount γ', 0, 1, 0.01, 0.9, '', 2);
    const RR = slider(ctl, 'Reward for exiting right, r', -2, 2, 0.1, 1, '', 1);
    const sr = rowOf(ctl);
    const stLbl = document.createElement('span'); stLbl.textContent = 'Inspect state:'; sr.appendChild(stLbl);
    const stSel = document.createElement('select');
    stSel.innerHTML = '<option value="0">state 1</option><option value="1">state 2</option><option value="2">state 3</option>';
    sr.appendChild(stSel);
    const info = block(ctl);
    let V = [0, 0, 0], sweeps = 0, timer = 0;
    const exact = (p, g, rR) => {
      const M = [[1, -p * g, 0, 0], [-(1 - p) * g, 1, -p * g, 0], [0, -(1 - p) * g, 1, p * rR]];
      for (let i = 0; i < 3; i++) {
        for (let k = i + 1; k < 3; k++) {
          const f = M[k][i] / M[i][i];
          for (let j = i; j < 4; j++) M[k][j] -= f * M[i][j];
        }
      }
      const x = [0, 0, 0];
      for (let i = 2; i >= 0; i--) {
        let v = M[i][3];
        for (let j = i + 1; j < 3; j++) v -= M[i][j] * x[j];
        x[i] = v / M[i][i];
      }
      return x;
    };
    const parts = (old, s, p, g, rR) => {
      const rightR = s === 2 ? rR : 0, rightNext = s === 2 ? 0 : old[s + 1];
      const leftR = 0, leftNext = s === 0 ? 0 : old[s - 1];
      return { QR: rightR + g * rightNext, QL: leftR + g * leftNext };
    };
    const sweep = () => {
      const p = P.get(), g = G.get(), rR = RR.get();
      const nv = [0, 0, 0];
      for (let s = 0; s < 3; s++) { const q = parts(V, s, p, g, rR); nv[s] = p * q.QR + (1 - p) * q.QL; }
      V = nv; sweeps++;
    };
    const restart = () => { V = [0, 0, 0]; sweeps = 0; timer = 0; lastKey = null; };
    const T = transport(ctl, restart, 1);
    [P, G, RR].forEach((sl) => sl.input.addEventListener('input', restart));
    stSel.addEventListener('change', () => { lastKey = null; });

    const draw = () => {
      const ctx = c.ctx, W = c.w, H = c.h, col = C();
      const p = P.get(), g = G.get(), rR = RR.get();
      const Ex = exact(p, g, rR);
      ctx.clearRect(0, 0, W, H);
      const cy = H * 0.22, xs = [0.3, 0.5, 0.7].map((f) => f * W), xL = 0.1 * W, xR = 0.9 * W;
      const nodeR = 18;
      xs.forEach((x, i) => {
        ctx.beginPath(); ctx.arc(x, cy, nodeR, 0, Math.PI * 2); ctx.fillStyle = col.bg2; ctx.fill(); ctx.strokeStyle = col.blue; ctx.stroke();
        txt(ctx, `s${i + 1}`, x, cy + 4, col.fg, 13, 'center', 'bold');
      });
      ctx.strokeStyle = col.muted; ctx.fillStyle = col.muted; ctx.lineWidth = 1;
      AI.arrow(ctx, xs[0] - nodeR, cy - 6, xs[1] - nodeR - 2, cy - 6, col.green, 1.5);
      AI.arrow(ctx, xs[1] + nodeR, cy - 6, xs[2] - nodeR - 2, cy - 6, col.green, 1.5);
      AI.arrow(ctx, xs[1] - nodeR, cy + 8, xs[0] + nodeR, cy + 8, col.orange, 1.5);
      AI.arrow(ctx, xs[2] - nodeR, cy + 8, xs[1] + nodeR, cy + 8, col.orange, 1.5);
      txt(ctx, `right ${AI.fmt(p, 2)}`, (xs[0] + xs[1]) / 2, cy - 14, col.green, 11, 'center');
      txt(ctx, `left ${AI.fmt(1 - p, 2)}`, (xs[0] + xs[1]) / 2, cy + 24, col.orange, 11, 'center');
      txt(ctx, `right ${AI.fmt(p, 2)}`, (xs[1] + xs[2]) / 2, cy - 14, col.green, 11, 'center');
      txt(ctx, `left ${AI.fmt(1 - p, 2)}`, (xs[1] + xs[2]) / 2, cy + 24, col.orange, 11, 'center');
      AI.arrow(ctx, xs[0] - nodeR, cy + 8, xL + 30, cy + 8, col.orange, 1.5);
      AI.arrow(ctx, xs[2] + nodeR, cy - 6, xR - 30, cy - 6, col.green, 1.5);
      txt(ctx, 'exit left: 0', xL, cy + 46, col.orange, 11, 'center');
      txt(ctx, `exit right: ${AI.fmt(rR, 2)}`, xR, cy + 46, col.green, 11, 'center');
      const top = H * 0.55, bot = H - 40, ymax = Math.max(0.1, ...V.map(Math.abs), ...Ex.map(Math.abs)) * 1.2;
      const Y = (v) => top + (1 - (v + ymax) / (2 * ymax)) * (bot - top);
      ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(40, Y(0)); ctx.lineTo(W - 20, Y(0)); ctx.stroke();
      [0, 1, 2].forEach((i) => {
        const x = W * (0.25 + i * 0.25), bw = W * 0.12;
        const y0 = Y(0), yv = Y(V[i]);
        ctx.fillStyle = col.blue; ctx.fillRect(x - bw / 2, Math.min(y0, yv), bw, Math.abs(y0 - yv));
        txt(ctx, `V(s${i + 1}) = ${AI.fmt(+V[i].toFixed(4), 4)}`, x, bot + 16, col.fg, 12, 'center');
        ctx.setLineDash([5, 4]); ctx.strokeStyle = col.orange; ctx.beginPath(); ctx.moveTo(x - bw * 0.8, Y(Ex[i])); ctx.lineTo(x + bw * 0.8, Y(Ex[i])); ctx.stroke(); ctx.setLineDash([]);
      });
      const err = Math.max(...V.map((v, i) => Math.abs(v - Ex[i])));
      const cap = sweeps === 0 ? 'Press Play: sweep 0 starts every value at 0.'
        : `Sweep ${sweeps}: each value becomes its one-step lookahead under the policy. Orange dashes = exact V^π.`;
      txt(ctx, cap, 14, 22, col.fg, 13);
      const s0 = +stSel.value, q = parts(V, s0, p, g, rR);
      info.innerHTML = `max |V_k − V^π| = ${AI.fmt(err, 6)} after ${sweeps} sweeps<br>` +
        `inspect s${s0 + 1}: Q(right) = ${AI.fmt(+q.QR.toFixed(4), 4)}, Q(left) = ${AI.fmt(+q.QL.toFixed(4), 4)}<br>` +
        `check: p·Q(right) + (1−p)·Q(left) = ${AI.fmt(+(p * q.QR + (1 - p) * q.QL).toFixed(4), 4)} = V(s${s0 + 1})`;
      return err;
    };
    let lastKey = null;
    c.onresize = () => { lastKey = null; };
    window.addEventListener('themechange', () => { lastKey = null; });
    AI.loop(root, (dt) => {
      if (T.stepReq) { T.stepReq = false; if (sweeps < 300) sweep(); }
      else if (T.playing && sweeps < 300) { timer += dt * T.speed(); if (timer >= 0.8) { timer = 0; sweep(); } }
      const k = [V.map((v) => v.toFixed(8)).join(','), sweeps, P.get(), G.get(), RR.get(), stSel.value].join('|');
      if (k !== lastKey) { draw(); lastKey = k; }
    }, { fps: 30 });
  };

  /* ================= 5. greedy versus epsilon-greedy ================= */
  AI.labs['rl-explore'] = function (root) {
    const { c, ctl } = mount(root, 'Greedy versus epsilon-greedy on three noisy arms', 'illustrative: Gaussian arms · both agents see the same noise · estimates start at 0', 340);
    const means = [slider(ctl, 'Arm A mean', -1, 3, 0.1, 0.5, '', 1), slider(ctl, 'Arm B mean', -1, 3, 0.1, 2, '', 1), slider(ctl, 'Arm C mean', -1, 3, 0.1, 1.2, '', 1)];
    const sd = slider(ctl, 'Noise σ', 0, 1.5, 0.05, 0.4, '', 2);
    const eps = slider(ctl, 'Exploration ε (second agent)', 0, 0.5, 0.01, 0.1, '', 2);
    const info = block(ctl);
    const NMAX = 300, NAMES = ['A', 'B', 'C'], ARM_COL = ['blue', 'green', 'orange'];
    const noise = []; { const r = AI.rng(3); for (let i = 0; i < NMAX; i++) noise.push(AI.randn(r)); }
    let agents = [], t = 0;
    const newAgent = (e) => ({ e, Q: [0, 0, 0], n: [0, 0, 0], hist: [], ret: 0, rng: AI.rng(42) });
    const restart = () => { agents = [newAgent(0), newAgent(eps.get())]; t = 0; lastKey = null; };
    const T = transport(ctl, restart, 1);
    const tk = AI.ticker();
    [...means, sd, eps].forEach((sl) => sl.input.addEventListener('input', restart));
    const argmax = (Q) => { let b = 0; for (let i = 1; i < Q.length; i++) if (Q[i] > Q[b]) b = i; return b; };
    const pull = (ag) => {
      const arm = ag.rng() < ag.e ? Math.floor(ag.rng() * 3) : argmax(ag.Q);
      const rew = means[arm].get() + sd.get() * noise[t];
      ag.n[arm]++; ag.Q[arm] += (rew - ag.Q[arm]) / ag.n[arm]; ag.ret += rew; ag.hist.push(arm);
      return { arm, rew };
    };
    let lastRes = null;
    const stepAll = () => {
      if (t >= NMAX) return;
      const res = agents.map((ag) => pull(ag));
      lastRes = res; t++;
    };
    const draw = () => {
      const ctx = c.ctx, W = c.w, H = c.h, col = C();
      ctx.clearRect(0, 0, W, H);
      const mv = means.map((m) => m.get());
      const best = argmax(mv);
      const panelW = (W - 30) / 2;
      const titles = ['Greedy (ε = 0)', `ε-greedy (ε = ${AI.fmt(eps.get(), 2)})`];
      agents.forEach((ag, pi) => {
        const x0 = 10 + pi * (panelW + 10), top = 60, bot = H * 0.62, ymin = -1, ymax = 3;
        const Y = (v) => bot - ((v - ymin) / (ymax - ymin)) * (bot - top);
        txt(ctx, titles[pi], x0 + 6, 46, col.fg, 14, 'left', 'bold');
        ctx.strokeStyle = col.border; ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x0 + panelW, Y(0)); ctx.stroke();
        [0, 1, 2].forEach((i) => {
          const x = x0 + panelW * (0.2 + i * 0.3), bw = panelW * 0.16;
          const y0 = Y(0), yv = Y(ag.Q[i]);
          ctx.fillStyle = col[ARM_COL[i]]; ctx.globalAlpha = 0.85;
          ctx.fillRect(x - bw / 2, Math.min(y0, yv), bw, Math.abs(y0 - yv)); ctx.globalAlpha = 1;
          ctx.setLineDash([5, 4]); ctx.strokeStyle = col.muted; ctx.beginPath(); ctx.moveTo(x - bw * 0.9, Y(mv[i])); ctx.lineTo(x + bw * 0.9, Y(mv[i])); ctx.stroke(); ctx.setLineDash([]);
          txt(ctx, `${NAMES[i]}: Q=${AI.fmt(+ag.Q[i].toFixed(3), 3)}`, x, bot + 18, col.fg, 12, 'center');
          txt(ctx, `pulls ${ag.n[i]}`, x, bot + 34, col.muted, 11, 'center');
        });
        const sx0 = x0 + 6, sw = panelW - 12, sy = H - 60;
        ctx.strokeStyle = col.border; ctx.strokeRect(sx0, sy - 8, sw, 16);
        ag.hist.forEach((arm, k) => {
          ctx.fillStyle = col[ARM_COL[arm]]; ctx.fillRect(sx0 + (k / NMAX) * sw, sy - 6, Math.max(1.5, sw / NMAX), 12);
        });
        const share = ag.hist.length ? Math.round((100 * ag.n[best]) / ag.hist.length) : 0;
        txt(ctx, `pulls: ${ag.hist.length} · total reward ${AI.fmt(+ag.ret.toFixed(2), 2)} · best arm ${NAMES[best]} share ${share}%`, x0 + 6, H - 14, col.muted, 11);
      });
      let cap;
      if (t === 0) cap = 'Press Play: each agent pulls one arm per step. Untried arms keep their starting estimate of 0.';
      else if (t >= NMAX) cap = `Stopped after ${NMAX} pulls. Press Reset to run again.`;
      else {
        const g = agents[0], e = agents[1], gl = lastRes[0], el = lastRes[1];
        const untried = g.n.map((v, i) => (v === 0 ? NAMES[i] : null)).filter(Boolean);
        cap = `Pull ${t}: greedy pulled ${NAMES[gl.arm]} (reward ${AI.fmt(+gl.rew.toFixed(2), 2)})` +
          (untried.length ? `; untried arms ${untried.join(', ')} still sit at 0.` : '.') +
          ` ε-agent pulled ${NAMES[el.arm]}.`;
      }
      txt(ctx, cap, 14, 24, col.fg, 13);
      info.innerHTML = `true means: A ${AI.fmt(mv[0], 1)}, B ${AI.fmt(mv[1], 1)}, C ${AI.fmt(mv[2], 1)} · best arm ${NAMES[best]}<br>` +
        `Greedy locks onto an arm whose estimate is above 0 and never tests the others; ε-greedy keeps tests going at rate ε.`;
    };
    let lastKey = null;
    c.onresize = () => { lastKey = null; };
    window.addEventListener('themechange', () => { lastKey = null; });
    restart();
    AI.loop(root, (dt) => {
      let n = 0;
      if (T.stepReq) { T.stepReq = false; n = 1; }
      else if (T.playing) n = tk(dt, 1 * T.speed());
      for (let i = 0; i < n; i++) stepAll();
      const k = [t, sd.get(), eps.get(), means.map((m) => m.get()).join(',')].join('|');
      if (k !== lastKey) { draw(); lastKey = k; }
    }, { fps: 30 });
  };
})();
