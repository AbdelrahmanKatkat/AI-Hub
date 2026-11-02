/* Reinforcement Learning · Module 4 (Temporal-difference learning) labs. All are illustrative toys with fixed seeds.
 * td-ripple       <div data-lab="td-ripple"></div>       TD(0) on a five-state random walk; the TD error moves back through values
 * cliff-sarsa     <div data-lab="cliff-sarsa"></div>     SARSA learning to walk a cliff-edge gridworld, one step at a time
 * cliff-compare   <div data-lab="cliff-compare"></div>   SARSA and Q-learning on the same cliff, same seed
 * target-spread   <div data-lab="target-spread"></div>   sampled SARSA bootstrap values versus the Expected SARSA average
 * nstep-target    <div data-lab="nstep-target"></div>    n-step targets built term by term on a three-step episode
 * lambda-traces   <div data-lab="lambda-traces"></div>   eligibility traces and the lambda return on the same episode
 */
(function () {
  const AI = window.AI;
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const F = (v, d = 3) => AI.fmt(v, d);

  /* ---------- shared UI helpers ---------- */
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

  /* transport: play/pause, step and reset. Step adds one pending step while paused. */
  const controls = (ctl, onReset) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    pp.onclick = () => { T.playing = !T.playing; pp.textContent = T.playing ? 'Pause' : 'Play'; };
    st.onclick = () => { T.playing = false; pp.textContent = 'Play'; T.pending += 1; };
    rs.onclick = () => onReset();
    return T;
  };

  /* drive a lab: rate() gives steps per second, stepFn() does one step, keyFn(dt) returns a state key and drawFn() runs when it changes */
  const pump = (root, T, rate, stepFn, keyFn, drawFn) => {
    const tk = AI.ticker();
    let lastKey = null;
    AI.loop(root, (dt) => {
      let n = 0;
      if (T.playing) n = tk(dt, rate());
      else if (T.pending) { n = T.pending; T.pending = 0; }
      for (let i = 0; i < n; i++) stepFn();
      const k = keyFn(dt);
      if (k !== lastKey) { lastKey = k; drawFn(); }
    });
  };
  const theme = () => document.documentElement.dataset.theme;

  /* ================= cliff-walking environment (Sutton & Barto, Example 6.6) =================
     4 rows x 12 columns, indexed s = 12*row + col. Start (3,0)=36, goal (3,11)=47, cliff (3,1)..(3,10) = 37..46.
     Every move costs -1; stepping onto the cliff costs -100 and sends the agent back to the start; reaching the goal ends the episode. */
  const CL = { START: 36, GOAL: 47, MAXT: 200 };
  const isCliff = (s) => s >= 37 && s <= 46;
  const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1], ANAME = ['up', 'right', 'down', 'left'];
  const envStep = (s, a) => {
    const r = AI.clamp(Math.floor(s / 12) + DR[a], 0, 3), c = AI.clamp((s % 12) + DC[a], 0, 11);
    const s2 = r * 12 + c;
    if (isCliff(s2)) return { s2: CL.START, r: -100, done: false, fell: true };
    if (s2 === CL.GOAL) return { s2, r: -1, done: true, fell: false };
    return { s2, r: -1, done: false, fell: false };
  };
  const cellName = (s) => `(${Math.floor(s / 12)},${s % 12})`;

  /* tabular agent. kind = 'sarsa' (on-policy) or 'q' (off-policy Q-learning). Epsilon-greedy behaviour. */
  const makeAgent = (kind, seed, eps, alpha) => ({
    kind, Q: new Float64Array(48 * 4), rng: AI.rng(seed), eps, alpha, gamma: 1,
    s: CL.START, a: null, t: 0, ret: 0, rets: [], trail: [CL.START], last: null,
  });
  const bestA = (ag, s) => {
    let best = -Infinity, cand = [];
    for (let a = 0; a < 4; a++) {
      const v = ag.Q[s * 4 + a];
      if (v > best + 1e-12) { best = v; cand = [a]; } else if (Math.abs(v - best) <= 1e-12) cand.push(a);
    }
    return cand[Math.floor(ag.rng() * cand.length)];
  };
  const choose = (ag, s) => (ag.rng() < ag.eps ? Math.floor(ag.rng() * 4) : bestA(ag, s));
  const maxQ = (ag, s) => Math.max(ag.Q[s * 4], ag.Q[s * 4 + 1], ag.Q[s * 4 + 2], ag.Q[s * 4 + 3]);

  /* one environment transition plus one update; returns what happened for the caption */
  const stepAgent = (ag) => {
    if (ag.a === null) ag.a = choose(ag, ag.s);
    const s = ag.s, a = ag.a, tr = envStep(s, a), i = s * 4 + a;
    let target, a2 = null;
    if (tr.done) target = tr.r;
    else if (ag.kind === 'sarsa') { a2 = choose(ag, tr.s2); target = tr.r + ag.gamma * ag.Q[tr.s2 * 4 + a2]; }
    else target = tr.r + ag.gamma * maxQ(ag, tr.s2);
    const old = ag.Q[i];
    ag.Q[i] = old + ag.alpha * (target - old);
    ag.ret += tr.r; ag.t++;
    ag.last = { s, a, r: tr.r, s2: tr.s2, a2, target, old, newQ: ag.Q[i], fell: tr.fell, done: tr.done, ended: false };
    ag.s = tr.s2; ag.trail.push(tr.s2);
    if (tr.done || ag.t >= CL.MAXT) {
      ag.rets.push(ag.ret); ag.ret = 0; ag.t = 0; ag.s = CL.START; ag.a = null; ag.trail = [CL.START];
      ag.last.ended = true;
    } else {
      ag.a = ag.kind === 'sarsa' ? a2 : null;
    }
    return ag.last;
  };

  /* greedy policy read off the table (first index wins ties, so it is deterministic) */
  const greedyPath = (Q) => {
    let s = CL.START; const path = [s]; let reached = false, fell = false;
    for (let k = 0; k < 80; k++) {
      let best = -Infinity, a = 0;
      for (let b = 0; b < 4; b++) if (Q[s * 4 + b] > best) { best = Q[s * 4 + b]; a = b; }
      const tr = envStep(s, a);
      if (tr.fell) fell = true;
      s = tr.s2; path.push(s);
      if (tr.done) { reached = true; break; }
    }
    return { path, reached, fell, steps: path.length - 1 };
  };

  /* grid drawing shared by the cliff labs */
  const gridGeom = (w, h) => {
    const cell = Math.min((w - 16) / 12, (h - 12) / 4);
    return { cell, x0: (w - 12 * cell) / 2, y0: (h - 4 * cell) / 2 };
  };
  const drawGrid = (cv, ag, label) => {
    const g = cv.ctx, C = AI.col(), w = cv.w, h = cv.h, G = gridGeom(w, h), cell = G.cell;
    g.clearRect(0, 0, w, h);
    g.font = `11px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle';
    for (let s = 0; s < 48; s++) {
      const x = G.x0 + (s % 12) * cell, y = G.y0 + Math.floor(s / 12) * cell;
      g.fillStyle = C.bg2; g.fillRect(x, y, cell, cell);
      if (isCliff(s)) { g.globalAlpha = 0.45; g.fillStyle = C.red; g.fillRect(x, y, cell, cell); g.globalAlpha = 1; }
      if (s === CL.GOAL) { g.globalAlpha = 0.5; g.fillStyle = C.green; g.fillRect(x, y, cell, cell); g.globalAlpha = 1; }
      g.strokeStyle = C.border; g.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (s === CL.START) { g.fillStyle = C.muted; g.fillText('S', x + cell / 2, y + cell / 2); }
      if (s === CL.GOAL) { g.fillStyle = C.muted; g.fillText('G', x + cell / 2, y + cell / 2); }
      if (isCliff(s) && s === 37) { g.fillStyle = C.fg; g.fillText('cliff', x + cell * 5, y + cell / 2); }
      if (!isCliff(s) && s !== CL.GOAL) {
        let best = -Infinity, a = -1;
        for (let b = 0; b < 4; b++) if (ag.Q[s * 4 + b] > best + 1e-9) { best = ag.Q[s * 4 + b]; a = b; }
        if (a >= 0 && best !== 0) {
          const mx = x + cell / 2, my = y + cell / 2, L = cell * 0.3;
          AI.arrow(g, mx - DC[a] * L * 0.6, my - DR[a] * L * 0.6, mx + DC[a] * L, my + DR[a] * L, C.blue, 1.6);
        }
      }
    }
    // trail of the current episode
    if (ag.trail.length > 1) {
      g.strokeStyle = C.orange; g.lineWidth = 2; g.globalAlpha = 0.55; g.beginPath();
      ag.trail.forEach((s, k) => {
        const px = G.x0 + (s % 12) * cell + cell / 2, py = G.y0 + Math.floor(s / 12) * cell + cell / 2;
        if (k === 0) g.moveTo(px, py); else g.lineTo(px, py);
      });
      g.stroke(); g.globalAlpha = 1;
    }
    const s = ag.s;
    g.fillStyle = C.orange; g.strokeStyle = C.fg; g.lineWidth = 1.5;
    g.beginPath(); g.arc(G.x0 + (s % 12) * cell + cell / 2, G.y0 + Math.floor(s / 12) * cell + cell / 2, cell * 0.2, 0, 6.2832); g.fill(); g.stroke();
    if (label) { g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(label, G.x0, G.y0 - 8); }
  };

  /* return per episode, with a 10-episode moving average; values below -200 are clipped */
  const drawRets = (cv, series) => {
    const g = cv.ctx, C = AI.col(), w = cv.w, h = cv.h;
    g.clearRect(0, 0, w, h);
    const n = Math.max(50, ...series.map((s) => s.rets.length));
    const v = AI.view(cv, 0, n, -200, 0);
    AI.axes(g, v);
    for (const s of series) {
      const r = s.rets;
      if (r.length < 2) continue;
      g.strokeStyle = s.color; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i < r.length; i++) {
        let sum = 0, c = 0;
        for (let j = Math.max(0, i - 9); j <= i; j++) { sum += r[j]; c++; }
        const px = v.X(i + 1), py = v.Y(AI.clamp(sum / c, -200, 0));
        if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();
    }
    g.fillStyle = C.muted; g.font = `11px ${MONO}`; g.textAlign = 'left';
    series.forEach((s, k) => { g.fillStyle = s.color; g.fillText(s.name, v.pad.l + 8 + k * 150, v.pad.t + 12); });
  };
  const baseline = (g, v, color) => { g.save(); g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(v.pad.l, v.Y(0)); g.lineTo(v.pad.l + v.pw, v.Y(0)); g.stroke(); g.restore(); };
  const yticks = (g, v, vals, color) => { g.save(); g.fillStyle = color; g.font = "11px " + MONO; g.textAlign = "right"; vals.forEach((y) => g.fillText(F(y, 2), v.pad.l - 5, v.Y(y) + 4)); g.restore(); };
  const avgLast = (r, k = 20) => {
    if (!r.length) return NaN;
    const xs = r.slice(-k); return xs.reduce((a, b) => a + b, 0) / xs.length;
  };

  /* ================= 1. td-ripple ================= */
  const N = 5, RIPPLE_SEED = 7;
  const trueValues = (g) => {
    const x = [0, 0, 0, 0, 0, 0, 0];
    for (let it = 0; it < 4000; it++) for (let i = 1; i <= N; i++) x[i] = g * (0.5 * x[i - 1] + 0.5 * x[i + 1]) + (i === N ? 0.5 : 0);
    return x;
  };
  AI.labs['td-ripple'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'TD(0) on a five-state random walk',
      'illustrative · the agent starts in state 3 and steps left or right at random; the right exit pays +1, the left exit pays 0 · dashed green = true values for this γ',
      `<canvas style="height:230px"></canvas><div class="cap" style="font-size:13px;min-height:52px;margin-top:8px"></div>`, '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const alpha = slider(ctl, 'Step size α', 0.01, 0.5, 0.01, 0.1, '', 2);
    const gam = slider(ctl, 'Discount γ', 0.5, 1, 0.01, 1, '', 2);
    const rate = slider(ctl, 'Transitions per second', 0.5, 8, 0.5, 2, '/s', 1);
    const out = readout(ctl);
    let V, disp, pos, rng, msg, lastBubble, trail, episodes, epoch, tg, tgG;
    const reset = () => {
      V = [0, 0, 0, 0, 0, 0, 0]; disp = V.slice(); pos = null; rng = AI.rng(RIPPLE_SEED);
      msg = 'Press Play. Each step the agent moves one cell at random, and TD(0) updates the value of the cell it left at every step, not at the end of the episode.';
      lastBubble = null; trail = []; episodes = 0; epoch = (epoch || 0) + 1;
    };
    reset();
    tgG = gam(); tg = trueValues(tgG);
    const T = controls(ctl, reset);
    const stepFn = () => {
      if (pos === null) { pos = 3; trail = [3]; msg = 'New episode: the agent starts in state 3.'; return; }
      const g = gam(), a = alpha(), s = pos, s2 = s + (rng() < 0.5 ? -1 : 1);
      const r = s2 === N + 1 ? 1 : 0, vNext = (s2 === 0 || s2 === N + 1) ? 0 : V[s2], vOld = V[s];
      const d = r + g * vNext - vOld;
      V[s] = vOld + a * d;
      lastBubble = { s, d, age: 0 };
      msg = `Moved ${s} → ${s2}, reward r = ${r}. TD error δ = r + γV(s′) − V(s) = ${r} + ${F(g, 2)}·${F(vNext, 3)} − ${F(vOld, 3)} = ${F(d, 3)}. Then V(${s}) ← ${F(vOld, 3)} + α·δ = ${F(V[s], 3)}.`;
      if (s2 === 0 || s2 === N + 1) {
        pos = null; episodes++;
        msg += r ? ' The episode ends at the right exit (reward 1).' : ' The episode ends at the left exit (reward 0).';
      } else { pos = s2; trail.push(s2); }
    };
    const keyFn = (dt) => {
      const g = gam();
      if (g !== tgG) { tg = trueValues(g); tgG = g; }
      let settled = true;
      for (let i = 0; i <= N + 1; i++) {
        disp[i] += (V[i] - disp[i]) * Math.min(1, dt * 6);
        if (Math.abs(V[i] - disp[i]) > 1e-3) settled = false;
      }
      if (settled) disp = V.slice();
      if (lastBubble) lastBubble.age += dt;
      let err = 0;
      for (let i = 1; i <= N; i++) err += (V[i] - tg[i]) ** 2;
      out.innerHTML = `episodes: <b>${episodes}</b> · α = ${F(alpha(), 2)} · γ = ${F(g, 2)}<br>RMS error of V against the true values: <b>${F(Math.sqrt(err / N), 3)}</b><br>true values (for this γ): ${[1, 2, 3, 4, 5].map((i) => F(tg[i], 3)).join(', ')}`;
      return [epoch, pos, episodes, disp.map((v) => v.toFixed(3)).join(), lastBubble ? Math.min(2.5, lastBubble.age).toFixed(1) : '', cv.w, cv.h, theme(), g.toFixed(3)].join('|');
    };
    const draw = () => {
      const C = AI.col(), gx = cv.ctx, w = cv.w, h = cv.h;
      gx.clearRect(0, 0, w, h);
      const bw = Math.min(90, (w - 40) / 7), gap = 8, x0 = (w - 7 * bw) / 2;
      const top = 30, bot = h - 40, H = bot - top;
      const X = (i) => x0 + i * bw + gap / 2, BW = bw - gap;
      const Y = (v) => bot - AI.clamp(v, 0, 1) * H;
      gx.font = `12px ${MONO}`; gx.textAlign = 'center'; gx.textBaseline = 'middle';
      for (let i = 0; i <= N + 1; i++) {
        const x = X(i);
        if (i === 0 || i === N + 1) {
          gx.fillStyle = C.bg3; gx.fillRect(x, top, BW, H);
          gx.fillStyle = C.muted;
          gx.fillText(i === 0 ? 'exit left' : 'exit right', x + BW / 2, top + H / 2 - 8);
          gx.fillText(i === 0 ? 'r = 0' : 'r = +1', x + BW / 2, top + H / 2 + 10);
          continue;
        }
        gx.strokeStyle = C.border; gx.lineWidth = 1; gx.strokeRect(x + 0.5, top + 0.5, BW, H);
        gx.globalAlpha = 0.55; gx.fillStyle = C.blue; gx.fillRect(x, Y(disp[i]), BW, bot - Y(disp[i])); gx.globalAlpha = 1;
        gx.setLineDash([5, 4]); gx.strokeStyle = C.green; gx.lineWidth = 2;
        gx.beginPath(); gx.moveTo(x, Y(tg[i])); gx.lineTo(x + BW, Y(tg[i])); gx.stroke(); gx.setLineDash([]); gx.lineWidth = 1;
        gx.fillStyle = C.fg; gx.fillText(`V(${i}) = ${F(V[i], 2)}`, x + BW / 2, bot + 14);
        gx.fillStyle = C.muted; gx.fillText(`state ${i}`, x + BW / 2, bot + 28);
      }
      if (pos !== null) {
        gx.fillStyle = C.orange; gx.beginPath(); gx.arc(X(pos) + BW / 2, top + 14, 7, 0, 6.2832); gx.fill();
      }
      if (lastBubble && lastBubble.age < 2.5) {
        const a = 1 - lastBubble.age / 2.5;
        gx.globalAlpha = a; gx.fillStyle = lastBubble.d >= 0 ? C.green : C.red;
        gx.fillText(`δ = ${F(lastBubble.d, 3)}`, X(lastBubble.s) + BW / 2, top + H * 0.5);
        gx.globalAlpha = 1;
      }
      cap.textContent = msg;
    };
    pump(root, T, () => rate(), stepFn, keyFn, draw);
    cv.onresize = draw;
  };

  /* ================= 2. cliff-sarsa ================= */
  AI.labs['cliff-sarsa'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'SARSA on a cliff-edge gridworld',
      'illustrative · ε-greedy SARSA with γ = 1 · cells are (row, column) from the top-left; the cliff is the red row at the bottom · each episode is capped at 200 steps',
      `<canvas style="height:150px"></canvas><canvas style="height:150px;margin-top:10px"></canvas><div class="cap" style="font-size:13px;min-height:56px;margin-top:8px"></div>`, '');
    const cvG = AI.canvas(AI.$('canvas', view)), cvR = AI.canvas(view.querySelectorAll('canvas')[1]), cap = AI.$('.cap', view);
    const eps = slider(ctl, 'Exploration ε', 0, 0.5, 0.01, 0.1, '', 2);
    const alpha = slider(ctl, 'Step size α', 0.05, 1, 0.05, 0.5, '', 2);
    const rate = slider(ctl, 'Steps per second', 1, 300, 1, 10, '/s', 0);
    const out = readout(ctl);
    let ag, epoch = 0;
    const reset = () => { ag = makeAgent('sarsa', 11, eps(), alpha()); epoch++; cap.textContent = 'Fresh table: every Q-value is 0, so every choice is a tie and the agent wanders. Press Play.'; };
    reset();
    const T = controls(ctl, reset);
    const stepFn = () => {
      ag.eps = eps(); ag.alpha = alpha();
      const L = stepAgent(ag), n = ag.rets.length + 1;
      let txt = `Episode ${n}, step ${ag.t || (L.ended ? 'end' : '')}: from ${cellName(L.s)} the agent took "${ANAME[L.a]}" → ${cellName(L.s2)}, reward ${L.r}.`;
      if (L.fell) txt += ' It fell off the cliff and is back at S.';
      if (L.ended && !L.done && !L.fell) txt += ' The episode hit the 200-step cap.';
      if (L.done) txt += ' Reached the goal.';
      if (!L.done) txt += ` SARSA target: r + γ·Q(s′, a′) with a′ = "${ANAME[L.a2]}" (really taken next) = ${F(L.target, 2)}. Q moves ${F(L.old, 2)} → ${F(L.newQ, 2)}.`;
      else txt += ` Target = r = ${L.r}; Q moves ${F(L.old, 2)} → ${F(L.newQ, 2)}.`;
      cap.textContent = txt;
    };
    const keyFn = () => {
      const k = [epoch, ag.rets.length, ag.t, ag.s, ag.trail.length, cvG.w, cvG.h, cvR.w, cvR.h, theme()].join('|');
      if (k !== undefined) {
        const gp = greedyPath(ag.Q);
        out.innerHTML = `episodes: <b>${ag.rets.length}</b> · average return, last 20 episodes: <b>${F(avgLast(ag.rets), 1)}</b><br>greedy path now: <b>${gp.steps}</b> steps, ${gp.reached ? 'reaches the goal' : 'does not reach the goal'}${gp.fell ? ', but its moves touch the cliff' : ''}`;
      }
      return k;
    };
    const draw = () => {
      drawGrid(cvG, ag, 'Greedy arrows and this episode’s path');
      drawRets(cvR, [{ rets: ag.rets, color: AI.col().blue, name: 'SARSA return (10-episode average)' }]);
    };
    pump(root, T, () => rate(), stepFn, keyFn, draw);
    cvG.onresize = draw; cvR.onresize = draw;
  };

  /* ================= 3. cliff-compare ================= */
  AI.labs['cliff-compare'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'SARSA and Q-learning on the same cliff',
      'illustrative · identical seed, ε and α · blue = SARSA (learns the safe, longer route), orange = Q-learning (learns the edge route, then pays for exploring near the cliff)',
      `<canvas style="height:130px"></canvas><canvas style="height:130px;margin-top:10px"></canvas><canvas style="height:150px;margin-top:10px"></canvas><div class="cap" style="font-size:13px;min-height:44px;margin-top:8px"></div>`, '');
    const cvS = AI.canvas(view.querySelectorAll('canvas')[0]), cvQ = AI.canvas(view.querySelectorAll('canvas')[1]), cvR = AI.canvas(view.querySelectorAll('canvas')[2]);
    const cap = AI.$('.cap', view);
    const eps = slider(ctl, 'Exploration ε', 0, 0.5, 0.01, 0.1, '', 2);
    const alpha = slider(ctl, 'Step size α', 0.05, 1, 0.05, 0.5, '', 2);
    const rate = slider(ctl, 'Steps per second (each agent)', 1, 300, 1, 10, '/s', 0);
    const out = readout(ctl);
    let aS, aQ, epoch = 0;
    const reset = () => { aS = makeAgent('sarsa', 11, eps(), alpha()); aQ = makeAgent('q', 11, eps(), alpha()); epoch++; cap.textContent = 'Both agents start from the same empty table and the same random seed. Press Play.'; };
    reset();
    const T = controls(ctl, reset);
    const stepFn = () => {
      aS.eps = aQ.eps = eps(); aS.alpha = aQ.alpha = alpha();
      const LS = stepAgent(aS), LQ = stepAgent(aQ);
      const fmtL = (name, L) => `${name}: ${cellName(L.s)} → ${cellName(L.s2)} via "${ANAME[L.a]}", r = ${L.r}${L.fell ? ' (fell off)' : ''}${L.done ? ' (goal)' : ''}`;
      cap.textContent = fmtL('SARSA', LS) + '.  ' + fmtL('Q-learning', LQ) + '.  SARSA bootstraps from the action it chose; Q-learning from the best action at the next state.';
    };
    const keyFn = () => {
      const gS = greedyPath(aS.Q), gQ = greedyPath(aQ.Q);
      out.innerHTML = `average return, last 20 episodes: SARSA <b>${F(avgLast(aS.rets), 1)}</b> · Q-learning <b>${F(avgLast(aQ.rets), 1)}</b><br>episodes: SARSA ${aS.rets.length} · Q-learning ${aQ.rets.length}<br>greedy route length: SARSA <b>${gS.steps}</b> · Q-learning <b>${gQ.steps}</b>`;
      return [epoch, aS.rets.length, aS.t, aS.s, aQ.rets.length, aQ.t, aQ.s, cvS.w, cvS.h, cvQ.w, cvQ.h, cvR.w, cvR.h, theme()].join('|');
    };
    const draw = () => {
      drawGrid(cvS, aS, 'SARSA');
      drawGrid(cvQ, aQ, 'Q-learning');
      drawRets(cvR, [{ rets: aS.rets, color: AI.col().blue, name: 'SARSA' }, { rets: aQ.rets, color: AI.col().orange, name: 'Q-learning' }]);
    };
    pump(root, T, () => rate(), stepFn, keyFn, draw);
    cvS.onresize = draw; cvQ.onresize = draw; cvR.onresize = draw;
  };

  /* ================= 4. target-spread ================= */
  AI.labs['target-spread'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Sampled SARSA values versus the Expected SARSA average',
      'exact for one next state s′ · three actions · the bootstrap value X = Q(s′, a′), with a′ drawn from the ε-greedy policy at s′',
      `<canvas style="height:150px"></canvas><canvas style="height:150px;margin-top:10px"></canvas><div class="cap" style="font-size:13px;min-height:44px;margin-top:8px"></div>`, '');
    const cvP = AI.canvas(view.querySelectorAll('canvas')[0]), cvN = AI.canvas(view.querySelectorAll('canvas')[1]), cap = AI.$('.cap', view);
    const NAMES = ['a₁ safe', 'a₂ edge', 'a₃ cliff'];
    const q1 = slider(ctl, 'Q(s′, a₁ safe)', -10, 0, 0.5, -1, '', 1);
    const q2 = slider(ctl, 'Q(s′, a₂ edge)', -10, 0, 0.5, -2, '', 1);
    const q3 = slider(ctl, 'Q(s′, a₃ cliff)', -10, 0, 0.5, -10, '', 1);
    const eps = slider(ctl, 'Exploration ε', 0, 0.5, 0.01, 0.3, '', 2);
    const rate = slider(ctl, 'Samples per second', 0.5, 10, 0.5, 3, '/s', 1);
    const out = readout(ctl);
    const dist = () => {
      const Q = [q1(), q2(), q3()], e = eps(), A = 3;
      let gi = 0; for (let i = 1; i < A; i++) if (Q[i] > Q[gi]) gi = i;
      const p = Q.map((_, i) => (i === gi ? 1 - e : 0) + e / A);
      const E = Q.reduce((s, q, i) => s + p[i] * q, 0);
      const v = Q.reduce((s, q, i) => s + p[i] * (q - E) ** 2, 0);
      return { Q, p, E, v, mx: Math.max(...Q), gi };
    };
    let rng, samples, key = '', epoch = 0, D;
    const restart = () => { rng = AI.rng(5); samples = []; epoch++; };
    restart();
    D = dist();
    const T = controls(ctl, restart);
    const stepFn = () => {
      const u = rng(); let acc = 0, pick = 2;
      for (let i = 0; i < 3; i++) { acc += D.p[i]; if (u < acc) { pick = i; break; } }
      samples.push({ a: pick, x: D.Q[pick] });
      cap.textContent = `Sample ${samples.length}: the policy picked ${NAMES[pick]}, so the SARSA bootstrap value is Q = ${F(D.Q[pick], 1)}. Each point is one possible target; their average drifts toward E, the Expected SARSA value.`;
    };
    const keyFn = () => {
      const d = dist();
      const k = [d.Q.join(), eps(), epoch].join('|');
      if (k !== key) { key = k; D = d; samples = []; }
      D = d;
      const n = samples.length;
      const mean = n ? samples.reduce((s, z) => s + z.x, 0) / n : NaN;
      const svar = n > 1 ? samples.reduce((s, z) => s + (z.x - mean) ** 2, 0) / (n - 1) : NaN;
      out.innerHTML = `exact E = Σ p·Q = <b>${F(d.E, 3)}</b> · exact variance of X = <b>${F(d.v, 3)}</b><br>max Q = <b>${F(d.mx, 1)}</b> (Q-learning would bootstrap from this) · greedy action: ${NAMES[d.gi]}<br>samples so far: ${n} · running mean: <b>${isNaN(mean) ? '–' : F(mean, 3)}</b> · sample variance: <b>${isNaN(svar) ? '–' : F(svar, 3)}</b>`;
      return [k, samples.length, cvP.w, cvP.h, cvN.w, cvN.h, theme()].join('|');
    };
    const draw = () => {
      const C = AI.col(), g = cvP.ctx, w = cvP.w, h = cvP.h;
      g.clearRect(0, 0, w, h);
      const v = AI.view(cvP, 0, 3, 0, 1, { l: 40, r: 12, t: 14, b: 40 });
      baseline(g, v, C.border); yticks(g, v, [0, 0.5, 1], C.muted);
      for (let i = 0; i < 3; i++) {
        const bw = v.pw / 3 * 0.4, cx = v.X(i + 0.5);
        g.fillStyle = i === D.gi ? C.blue : C.muted; g.globalAlpha = 0.6;
        g.fillRect(cx - bw / 2, v.Y(D.p[i]), bw, v.Y(0) - v.Y(D.p[i])); g.globalAlpha = 1;
        g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'center';
        g.fillText(`${NAMES[i]}: p = ${F(D.p[i], 2)}, Q = ${F(D.Q[i], 1)}`, cx, v.Y(D.p[i]) - 6);
      }
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('probability of the action the policy picks at s′', v.pad.l + 4, v.pad.t + 12);

      const n2 = cvN.ctx, w2 = cvN.w, h2 = cvN.h;
      n2.clearRect(0, 0, w2, h2);
      const vn = AI.view(cvN, -11, 1, 0, 1, { l: 30, r: 16, t: 14, b: 30 });
      AI.axes(n2, vn, { noGrid: true, noY: true });
      const line = (x, color, dash, label) => {
        n2.save(); n2.strokeStyle = color; n2.lineWidth = 1.5; n2.setLineDash(dash); n2.beginPath();
        n2.moveTo(vn.X(x), vn.pad.t); n2.lineTo(vn.X(x), vn.pad.t + vn.ph); n2.stroke(); n2.restore();
        n2.fillStyle = color; n2.font = '11px ' + MONO; n2.textAlign = 'center'; n2.fillText(label, vn.X(x), vn.pad.t + 10);
      };
      samples.forEach((z, k) => {
        const jy = 0.2 + 0.6 * ((k * 0.618) % 1);
        n2.globalAlpha = 0.55; n2.fillStyle = C.blue;
        n2.beginPath(); n2.arc(vn.X(z.x), vn.Y(jy), 3, 0, 6.2832); n2.fill(); n2.globalAlpha = 1;
      });
      const n = samples.length;
      if (n) {
        const mean = samples.reduce((s, z) => s + z.x, 0) / n;
        n2.strokeStyle = C.blue; n2.lineWidth = 2.5; n2.beginPath(); n2.moveTo(vn.X(mean), vn.pad.t + 22); n2.lineTo(vn.X(mean), vn.pad.t + vn.ph); n2.stroke();
      }
      line(D.E, C.green, [5, 4], 'E');
      line(D.mx, C.orange, [2, 3], 'max');
      n2.fillStyle = C.muted; n2.font = '11px ' + MONO; n2.textAlign = 'left';
      n2.fillText('X = Q(s′, a′) → (blue dots: sampled; solid blue: running mean)', vn.pad.l + 4, vn.pad.t + vn.ph - 4);
    };
    pump(root, T, () => rate(), stepFn, keyFn, draw);
    cvP.onresize = draw; cvN.onresize = draw;
  };

  /* ================= shared three-step episode for lessons 4.5 and 4.6 =================
     Chain S1 -> S2 -> S3 -> terminal, rewards R1 = 0, R2 = 0, R3 = 1, gamma = 0.9.
     Value estimates V = (V(S1), V(S2), V(S3)) = (0, 0.225, 0.75): the values after two TD(0) episodes (lesson 4.1). */
  const CHAIN = { R: [0, 0, 1], V: [0, 0.225, 0.75], g: 0.9, trueS1: 0.81 };

  /* ================= 5. nstep-target ================= */
  AI.labs['nstep-target'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Building an n-step target term by term',
      'illustrative numbers from a three-step chain · γ = 0.9 · the estimates V(S2) = 0.225 and V(S3) = 0.75 are frozen for this lab',
      `<canvas style="height:190px"></canvas><canvas style="height:150px;margin-top:10px"></canvas><div class="cap" style="font-size:13px;min-height:44px;margin-top:8px"></div>`, '');
    const cvT = AI.canvas(view.querySelectorAll('canvas')[0]), cvB = AI.canvas(view.querySelectorAll('canvas')[1]), cap = AI.$('.cap', view);
    const nS = slider(ctl, 'Steps before bootstrapping, n', 1, 3, 1, 2, '', 0);
    const out = readout(ctl);
    const G = CHAIN;
    // term k (k = 0..2) is gamma^k R_{k+1}; the bootstrap term for window n is gamma^n V(S_{n+1})
    const Gn = (n) => {
      let s = 0;
      for (let k = 0; k < n; k++) s += Math.pow(G.g, k) * G.R[k];
      const vb = n < 3 ? G.V[n] : 0; // V(S2) after 1 step, V(S3) after 2 steps, V(terminal) = 0 after 3
      return s + Math.pow(G.g, n) * vb;
    };
    let prog = 0, lastN = -1;
    const reset = () => { prog = 0; lastN = -1; };
    const T = controls(ctl, reset);
    const stepFn = () => { prog = Math.min(nS() + 1, prog + 1); };
    const keyFn = (dt) => {
      const n = nS();
      if (T.playing) prog = Math.min(n + 1, prog + dt * 1.0);
      const done = prog >= n + 1;
      const g = Gn(n);
      out.innerHTML = `G<sup>(${n})</sup> = <b>${F(g, 4)}</b> · true value V*(S1) = γ² = <b>${F(G.trueS1, 3)}</b> · error of this target = <b>${F(g - G.trueS1, 4)}</b><br>${n === 3 ? 'with n = 3 nothing is bootstrapped: this is the Monte Carlo return.' : 'the bootstrap term uses V(S' + (n + 2) + ') = ' + F(G.V[n], 3) + '.'}`;
      return [n, prog.toFixed(2), done, cvT.w, cvT.h, cvB.w, cvB.h, theme()].join('|');
    };
    const draw = () => {
      const C = AI.col(), n = nS(), g = cvT.ctx, w = cvT.w, h = cvT.h;
      g.clearRect(0, 0, w, h);
      g.font = '12px ' + MONO; g.textAlign = 'center'; g.textBaseline = 'middle';
      const xs = [w * 0.12, w * 0.38, w * 0.62, w * 0.88], ys = h * 0.3;
      const names = ['S1', 'S2', 'S3', 'end'];
      for (let i = 0; i < 4; i++) {
        g.fillStyle = i === 3 ? C.bg3 : (i < n ? C.blue : C.bg2); g.globalAlpha = i < n ? 0.55 : 1;
        g.beginPath(); g.arc(xs[i], ys, 20, 0, 6.2832); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = C.border; g.stroke();
        g.fillStyle = C.fg; g.fillText(names[i], xs[i], ys);
      }
      for (let k = 0; k < 3; k++) {
        const inWin = k < n;
        g.strokeStyle = inWin ? C.orange : C.muted; g.lineWidth = inWin ? 2.5 : 1; g.globalAlpha = inWin ? 1 : 0.5;
        g.beginPath(); g.moveTo(xs[k] + 22, ys); g.lineTo(xs[k + 1] - 22, ys); g.stroke(); g.globalAlpha = 1;
        g.fillStyle = inWin ? C.orange : C.muted;
        g.fillText(`R${k + 1} = ${G.R[k]}`, (xs[k] + xs[k + 1]) / 2, ys - 16);
      }
      g.fillStyle = C.muted; g.font = '11px ' + MONO;
      g.fillText(`bootstrap after n = ${n}: γ^${n} · V(S${n + 1}) = ${F(Math.pow(G.g, n) * (n < 3 ? G.V[n] : 0), 4)}`, w / 2, h * 0.7);
      g.fillText('orange = the reward terms inside the n-step window', w / 2, h * 0.82);

      // bar chart: contributions of the target for the chosen n, building term by term
      const b = cvB.ctx, bw = cvB.w, bh = cvB.h;
      b.clearRect(0, 0, bw, bh);
      const v = AI.view(cvB, 0, 4, -0.1, 1, { l: 40, r: 12, t: 12, b: 26 });
      baseline(b, v, C.border); yticks(b, v, [0, 0.5, 1], C.muted);
      let run = 0;
      for (let k = 0; k <= n; k++) {
        const term = k < n ? Math.pow(G.g, k) * G.R[k] : Math.pow(G.g, n) * (n < 3 ? G.V[n] : 0);
        const frac = AI.clamp(prog - k, 0, 1);
        if (frac <= 0) continue;
        const y0 = v.Y(run), y1 = v.Y(run + term * frac);
        b.fillStyle = k < n ? C.orange : C.blue; b.globalAlpha = 0.65;
        b.fillRect(v.X(k + 0.15), Math.min(y0, y1), v.X(k + 0.85) - v.X(k + 0.15), Math.abs(y1 - y0));
        b.globalAlpha = 1;
        run += term * frac;
      }
      b.strokeStyle = C.green; b.setLineDash([5, 4]); b.lineWidth = 2;
      b.beginPath(); b.moveTo(v.pad.l, v.Y(G.trueS1)); b.lineTo(v.pad.l + v.pw, v.Y(G.trueS1)); b.stroke(); b.setLineDash([]);
      b.fillStyle = C.fg; b.font = '11px ' + MONO; b.textAlign = 'left';
      b.fillText('target so far: ' + F(run, 3), v.pad.l + 6, v.pad.t + 12);
      b.fillStyle = C.green; b.fillText('true 0.81', v.pad.l + 6, v.Y(G.trueS1) - 6);
      b.fillStyle = C.muted; b.textAlign = 'center';
      ['term 1', 'term 2', 'term 3', 'bootstrap'].forEach((t, k) => b.fillText(t, v.X(k + 0.5), bh - 8));

      const steps = Math.floor(prog);
      cap.textContent = prog >= n + 1
        ? `n = ${n}: the target is G^(${n}) = ${F(Gn(n), 4)}. Each extra step replaces an estimate with a real reward, so the target depends less on V and more on the rewards.`
        : `Building G^(${n}) one term at a time: ${Math.min(steps, n)} of ${n} reward terms (plus ${steps >= n ? 'the bootstrap' : 'no bootstrap yet'}). Change n to compare.`;
    };
    pump(root, T, () => 0, stepFn, keyFn, draw);
    cvT.onresize = draw; cvB.onresize = draw;
  };

  /* ================= 6. lambda-traces ================= */
  AI.labs['lambda-traces'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Eligibility traces on a three-step episode',
      'the backward view: each step adds δ times a decaying trace · γ = 0.9, and V frozen at (0, 0.225, 0.75) during the episode (offline)',
      `<canvas style="height:200px"></canvas><div class="cap" style="font-size:13px;min-height:44px;margin-top:8px"></div>`, '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const lam = slider(ctl, 'Trace decay λ', 0, 1, 0.05, 0.5, '', 2);
    const alpha = slider(ctl, 'Step size α', 0.05, 1, 0.05, 0.5, '', 2);
    const rate = slider(ctl, 'Time steps per second', 0.2, 2, 0.1, 0.5, '/s', 1);
    const out = readout(ctl);
    const G = CHAIN;
    let t, e, dV, dls, epoch = 0;
    const reset = () => { t = 0; e = [0, 0, 0]; dV = [0, 0, 0]; dls = []; epoch++; cap.textContent = 'Time 0: the agent is in S1. Press Play to watch the trace for each state move through the episode.'; };
    reset();
    const T = controls(ctl, reset);
    const deltaAt = (k) => G.R[k] + G.g * (k < 2 ? G.V[k + 1] : 0) - G.V[k];
    const stepFn = () => {
      if (t >= 3) return;
      const l = lam(), a = alpha(), gl = G.g * l;
      for (let s = 0; s < 3; s++) e[s] *= gl;
      e[t] += 1;
      const d = deltaAt(t);
      for (let s = 0; s < 3; s++) dV[s] += a * d * e[s];
      dls.push(d);
      cap.textContent = `Time ${t}: at S${t + 1}, δ${t} = R + γV(next) − V = ${F(G.R[t], 2)} + ${F(G.g, 2)}·${F(k2(t), 3)} − ${F(G.V[t], 3)} = ${F(d, 4)}. Every trace is multiplied by γλ = ${F(gl, 3)}, S${t + 1} gets +1, and each state moves by α·δ·e.`;
      t++;
      if (t === 3) cap.textContent += ' Episode finished: the traces are used up.';
    };
    const k2 = (k) => (k < 2 ? G.V[k + 1] : 0);
    const keyFn = () => {
      const l = lam(), a = alpha();
      const Gl = (() => {
        const G1 = G.R[0] + G.g * G.V[1], G2 = G.R[0] + G.g * G.R[1] + G.g * G.g * G.V[2], G3 = G.R[0] + G.g * G.R[1] + G.g * G.g * G.R[2];
        return (1 - l) * G1 + (1 - l) * l * G2 + l * l * G3;
      })();
      const fwd = a * (Gl - G.V[0]);
      out.innerHTML = `forward view: λ-return G<sup>λ</sup><sub>0</sub> = <b>${F(Gl, 4)}</b> · α(G<sup>λ</sup> − V(S1)) = <b>${F(fwd, 4)}</b><br>backward view, so far: ΔV(S1) = <b>${F(dV[0], 4)}</b> · ΔV(S2) = ${F(dV[1], 4)} · ΔV(S3) = ${F(dV[2], 4)}<br>weights on the forward view: n = 1: ${F(1 - l, 2)} · n = 2: ${F((1 - l) * l, 2)} · full return: ${F(l * l, 2)}`;
      return [epoch, t, l.toFixed(3), a.toFixed(3), cv.w, cv.h, theme()].join('|');
    };
    const draw = () => {
      const C = AI.col(), g = cv.ctx, w = cv.w, h = cv.h;
      g.clearRect(0, 0, w, h);
      const half = w / 2;
      g.font = '12px ' + MONO; g.textAlign = 'center'; g.textBaseline = 'middle';
      // left: traces e(S1..S3)
      const v = AI.view(cv, 0, 3, 0, 1.05, { l: 40, r: half, t: 26, b: 34 });
      const bwid = v.pw / 3 * 0.5;
      g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText('eligibility traces e(s)', 40, 12);
      baseline(g, v, C.border); yticks(g, v, [0, 0.5, 1], C.muted);
      for (let s = 0; s < 3; s++) {
        const cx = v.X(s + 0.5);
        g.fillStyle = s === t - 1 ? C.orange : C.blue; g.globalAlpha = 0.6;
        g.fillRect(cx - bwid / 2, v.Y(e[s]), bwid, v.Y(0) - v.Y(e[s])); g.globalAlpha = 1;
        g.fillStyle = C.fg; g.textAlign = 'center';
        g.fillText(`S${s + 1}: ${F(e[s], 3)}`, cx, v.Y(e[s]) - 9);
      }
      // right: accumulated update and the last delta
      const x2 = half + 10;
      g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText('accumulated α·δ·e', x2, 12);
      const bars = dV;
      const vb = AI.view(cv, 0, 3, -0.05, 0.35, { l: x2, r: 12, t: 26, b: 34 });
      for (let s = 0; s < 3; s++) {
        const cx = vb.X(s + 0.5), bw2 = (vb.pw / 3) * 0.5;
        g.fillStyle = C.purple; g.globalAlpha = 0.7;
        const y0 = vb.Y(0), y1 = vb.Y(bars[s]);
        g.fillRect(cx - bw2 / 2, Math.min(y0, y1), bw2, Math.abs(y1 - y0)); g.globalAlpha = 1;
        g.fillStyle = C.fg; g.textAlign = 'center';
        g.fillText(`S${s + 1}: ${F(bars[s], 3)}`, cx, Math.min(y0, y1) - 9);
      }
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText(`δ so far: ${dls.map((d) => F(d, 3)).join(', ') || '–'}`, x2, h - 12);
    };
    pump(root, T, () => rate(), stepFn, keyFn, draw);
    cv.onresize = draw;
  };

  /* expose the pure simulation pieces for testing */
  AI.rlTD = { envStep, makeAgent, stepAgent, greedyPath, trueValues, CHAIN, CL };
})();
