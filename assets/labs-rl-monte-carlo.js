/* Reinforcement learning · Module 3 (Monte Carlo methods) labs. All run on a five-cell corridor (cells 0 and 4 are the ends,
 * entering 4 pays r, every other step pays 0). Illustrative toys; every run is seeded so results are reproducible.
 * mc-walk        <div data-lab="mc-walk"></div>        Monte Carlo prediction: walk, carry returns back, average
 * mc-control     <div data-lab="mc-control"></div>     Monte Carlo control with exploring starts versus a fixed start
 * mc-onpolicy    <div data-lab="mc-onpolicy"></div>    on-policy control with epsilon-soft policies
 * mc-is          <div data-lab="mc-is"></div>          importance sampling: ordinary versus weighted estimates
 * mc-offpolicy   <div data-lab="mc-offpolicy"></div>   off-policy weighted importance-sampling control and its stopping rule
 * mc-error       <div data-lab="mc-error"></div>       error of the Monte Carlo estimate versus number of episodes
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const SEED = 20240601;
  const B = 0.5; // behaviour probability of each action in the uniform examples

  /* ---------- helpers ---------- */
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
  const modeButtons = (row, labels, init, onPick) => {
    const bs = labels.map((l, i) => { const b = button(row, l); b.onclick = () => { bs.forEach((x, j) => x.classList.toggle('on', j === i)); onPick(i); }; return b; });
    bs.forEach((x, j) => x.classList.toggle('on', j === init));
    return bs;
  };

  /* transport: play/pause, step, reset, speed. Step queues one discrete event for the lab loop. */
  const transport = (ctl, onReset) => {
    const r = rowOf(ctl);
    const Tr = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    Tr.speed = slider(ctl, 'Speed', 0.25, 2, 0.05, 1, '×', 2);
    Tr.setPlaying = (p) => { Tr.playing = p; pp.textContent = p ? 'Pause' : 'Play'; };
    pp.onclick = () => Tr.setPlaying(!Tr.playing);
    st.onclick = () => { Tr.setPlaying(false); Tr.pending += 1; };
    rs.onclick = () => onReset();
    return Tr;
  };

  /* redraw only when the state key, canvas size or theme changes */
  const live = (root, cvs, key, update, draw) => {
    let last = null;
    AI.loop(root, (dt) => {
      update(dt);
      const k = key() + '|' + cvs.map((c) => c.w + 'x' + c.h).join(',') + '|' + document.documentElement.dataset.theme;
      if (k === last) return;
      last = k; draw();
    });
  };

  /* ---------- corridor model ---------- */
  const move = (s, a, rR) => {
    const s2 = s + a;
    if (s2 === 0) return { s: 0, r: 0, done: true };
    if (s2 === 4) return { s: 4, r: rR, done: true };
    return { s: s2, r: 0, done: false };
  };

  // one episode: P = positions (T+1, last is terminal), A = actions (T), R = rewards (T), G = returns (T)
  const makeEp = (s0, actFn, rR, gamma) => {
    const P = [s0], A = [], R = [];
    let s = s0;
    for (let t = 0; t < 60; t++) {
      const a = actFn(s, t);
      const m = move(s, a, rR);
      A.push(a); R.push(m.r); P.push(m.s); s = m.s;
      if (m.done) break;
    }
    const T = A.length, G = new Array(T).fill(0);
    let acc = 0;
    for (let t = T - 1; t >= 0; t--) { acc = R[t] + gamma * acc; G[t] = acc; }
    return { P, A, R, G, T };
  };

  // exact state values for interior cells 1..3 under a policy with P(R | s) = pR(s)
  const exactV = (gamma, rR, pR) => {
    const V = { 1: 0, 2: 0, 3: 0 };
    for (let i = 0; i < 1000; i++) {
      const nv = {};
      for (const s of [1, 2, 3]) {
        let v = 0;
        for (const a of [-1, 1]) {
          const p = a === 1 ? pR(s) : 1 - pR(s);
          const m = move(s, a, rR);
          v += p * (m.r + gamma * (m.done ? 0 : V[m.s]));
        }
        nv[s] = v;
      }
      Object.assign(V, nv);
    }
    return V;
  };

  // optimal action values q*(s,a) for interior cells, reward r on exiting at 4
  const optimalQ = (gamma, rR) => {
    const V = { 1: 0, 2: 0, 3: 0 };
    const q = (s, a) => { const m = move(s, a, rR); return m.r + gamma * (m.done ? 0 : V[m.s]); };
    for (let i = 0; i < 1000; i++) for (const s of [1, 2, 3]) V[s] = Math.max(q(s, -1), q(s, 1));
    return { 1: [q(1, -1), q(1, 1)], 2: [q(2, -1), q(2, 1)], 3: [q(3, -1), q(3, 1)] };
  };

  /* episode animation: walk the path, carry the returns back, then commit. Phases: hold -> walk -> back -> hold. */
  const runner = (nextEp, commit) => {
    const X = { ep: null, phase: 'hold', k: 0, shown: 0, acc: 1.2, epNo: 0 };
    const dur = () => (X.phase === 'hold' ? 1.2 : 0.6);
    const event = () => {
      if (X.phase === 'hold') { X.ep = nextEp(); X.epNo++; X.phase = 'walk'; X.k = 0; X.shown = 0; }
      else if (X.phase === 'walk') { if (X.k < X.ep.T) X.k++; else { X.phase = 'back'; X.shown = 0; } }
      else if (X.shown < X.ep.T) X.shown++;
      else { commit(X.ep); X.phase = 'hold'; }
    };
    X.tick = (dt, Tr) => {
      if (Tr.pending) { for (let i = 0; i < Tr.pending; i++) event(); Tr.pending = 0; }
      if (!Tr.playing) return;
      X.acc += dt * Tr.speed();
      let guard = 0;
      while (X.acc >= dur() && guard++ < 20) { X.acc -= dur(); event(); }
    };
    X.key = () => X.epNo + '|' + X.phase + '|' + X.k + '|' + X.shown;
    X.reset = () => { X.ep = null; X.phase = 'hold'; X.epNo = 0; X.k = 0; X.shown = 0; X.acc = 1.2; };
    return X;
  };

  /* corridor drawing. o: fill{s:color}, top{s:text}, bot{s:text}, endLabel[2], trail[], walker, note */
  const drawCorridor = (c, o) => {
    const C = AI.col(), g = c.ctx;
    g.clearRect(0, 0, c.w, c.h);
    const pad = 10, gap = 6, y0 = 26, cw = (c.w - 2 * pad - 4 * gap) / 5;
    const ch = Math.max(56, c.h - y0 - 44);
    const cx = (s) => pad + s * (cw + gap) + cw / 2;
    g.font = '11px ' + MONO; g.textAlign = 'center';
    for (let s = 0; s < 5; s++) {
      const x = pad + s * (cw + gap), end = s === 0 || s === 4;
      g.fillStyle = end ? C.bg3 : C.bg2; g.fillRect(x, y0, cw, ch);
      if (o.fill && o.fill[s]) { g.globalAlpha = 0.28; g.fillStyle = o.fill[s]; g.fillRect(x, y0, cw, ch); g.globalAlpha = 1; }
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x + 0.5, y0 + 0.5, cw - 1, ch - 1);
      g.fillStyle = C.muted; g.fillText(end ? 'end ' + s : 'cell ' + s, x + cw / 2, y0 - 9);
      if (end) {
        g.fillStyle = C.fg; g.font = 'bold 12px ' + MONO;
        g.fillText(o.endLabel ? o.endLabel[s === 0 ? 0 : 1] : '', x + cw / 2, y0 + ch * 0.5 + 4);
        g.font = '11px ' + MONO;
      } else {
        g.fillStyle = C.fg; g.font = 'bold 12px ' + MONO;
        g.fillText((o.top && o.top[s]) || '', x + cw / 2, y0 + ch * 0.42);
        g.font = '11px ' + MONO; g.fillStyle = C.muted;
        g.fillText((o.bot && o.bot[s]) || '', x + cw / 2, y0 + ch * 0.72);
      }
    }
    const band = y0 + ch + 14;
    if (o.trail && o.trail.length > 1) {
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.globalAlpha = 0.8; g.beginPath();
      o.trail.forEach((s, i) => (i ? g.lineTo(cx(s), band) : g.moveTo(cx(s), band)));
      g.stroke(); g.globalAlpha = 1;
    }
    if (o.walker !== null && o.walker !== undefined) {
      g.fillStyle = C.orange; g.beginPath(); g.arc(cx(o.walker), band, 6, 0, 6.2832); g.fill();
    }
    g.fillStyle = C.fg; g.textAlign = 'left';
    g.fillText(o.note || '', pad, c.h - 8);
  };

  /* bar chart. items: {label, value, sub, color, mark}; o: vmax, step, target, targetLabel, title */
  const drawBars = (c, items, o) => {
    const C = AI.col(), g = c.ctx;
    g.clearRect(0, 0, c.w, c.h);
    const pad = { l: 40, r: 10, t: 24, b: 40 };
    const pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b;
    const vmax = o.vmax || 1, step = o.step || 0.25, bw = pw / items.length;
    g.font = '11px ' + MONO; g.strokeStyle = C.border; g.fillStyle = C.muted; g.textAlign = 'right';
    for (let v = 0; v <= vmax + 1e-9; v += step) {
      const y = pad.t + ph - (v / vmax) * ph;
      g.beginPath(); g.moveTo(pad.l, Math.round(y) + 0.5); g.lineTo(pad.l + pw, Math.round(y) + 0.5); g.stroke();
      g.fillText(AI.fmt(+v.toFixed(3), 2), pad.l - 5, y + 4);
    }
    items.forEach((it, i) => {
      const x = pad.l + i * bw + bw * 0.18, w = bw * 0.64;
      const h = AI.clamp(it.value / vmax, 0, 1) * ph;
      g.fillStyle = it.color || C.blue; g.globalAlpha = 0.85; g.fillRect(x, pad.t + ph - h, w, h); g.globalAlpha = 1;
      g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(AI.fmt(it.value, 3), x + w / 2, pad.t + ph - h - 5);
      if (it.mark !== undefined) {
        const my = pad.t + ph - (it.mark / vmax) * ph;
        g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); g.moveTo(x - 4, my); g.lineTo(x + w + 4, my); g.stroke();
      }
      g.fillStyle = C.muted; g.fillText(it.label, x + w / 2, pad.t + ph + 14);
      if (it.sub) g.fillText(it.sub, x + w / 2, pad.t + ph + 28);
    });
    if (o.target !== undefined) {
      const y = pad.t + ph - (o.target / vmax) * ph;
      g.setLineDash([5, 4]); g.strokeStyle = C.green; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + pw, y); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.green; g.textAlign = 'left'; g.fillText(o.targetLabel || '', pad.l + 4, y - 4);
    }
    if (o.title) { g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(o.title, pad.l, 13); }
  };

  /* first-visit update of action values; q[s] = [Q(s,L), Q(s,R)], n[s] = counts */
  const firstVisitUpdate = (ep, q, n) => {
    const seen = new Set();
    for (let t = 0; t < ep.T; t++) {
      const s = ep.P[t];
      if (s === 0 || s === 4) continue;
      const ai = ep.A[t] === 1 ? 1 : 0, key = s * 2 + ai;
      if (seen.has(key)) continue;
      seen.add(key);
      n[s][ai] += 1;
      q[s][ai] += (ep.G[t] - q[s][ai]) / n[s][ai];
    }
  };
  const zeroQ = () => ({ 1: [0, 0], 2: [0, 0], 3: [0, 0] });

  /* ================= 3.1 Monte Carlo prediction ================= */
  AI.labs['mc-walk'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Monte Carlo prediction: walk, collect, average',
      'illustrative · corridor, uniform random policy, start in cell 2 · returns are carried back from the end and averaged',
      `<canvas class="cor" style="height:170px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.cor', view));
    const gS = slider(ctl, 'Discount γ', 0.5, 1, 0.05, 0.9, '', 2);
    const rS = slider(ctl, 'Reward for exiting at cell 4 (r)', 0, 2, 0.1, 1, '', 1);
    const out = readout(ctl);
    const mode = { v: 0 };
    const mrow = rowOf(ctl);
    modeButtons(mrow, ['First-visit', 'Every-visit'], 0, (i) => { mode.v = i; });
    let rng, fvS, fvN, evS, evN, EX = null, cfg = '';
    const resetStats = () => {
      rng = AI.rng(SEED);
      fvS = { 1: 0, 2: 0, 3: 0 }; fvN = { 1: 0, 2: 0, 3: 0 };
      evS = { 1: 0, 2: 0, 3: 0 }; evN = { 1: 0, 2: 0, 3: 0 };
    };
    resetStats();
    const commit = (ep) => {
      const seen = new Set();
      for (let t = 0; t < ep.T; t++) {
        const s = ep.P[t];
        if (s === 0 || s === 4) continue;
        evS[s] += ep.G[t]; evN[s] += 1;
        if (!seen.has(s)) { seen.add(s); fvS[s] += ep.G[t]; fvN[s] += 1; }
      }
    };
    const Tr = transport(ctl, () => { resetStats(); X.reset(); });
    const X = runner(() => makeEp(2, () => (rng() < 0.5 ? -1 : 1), rS(), gS()), commit);
    const refreshCfg = () => {
      const k = gS() + '|' + rS();
      if (k === cfg) return;
      cfg = k; EX = exactV(gS(), rS(), () => 0.5); resetStats(); X.reset();
    };
    const mean = (sum, n, s) => (n[s] ? sum[s] / n[s] : null);
    const update = (dt) => { refreshCfg(); X.tick(dt, Tr); };
    const draw = () => {
      const ep = X.ep, first = mode.v === 0;
      const top = {}, bot = {}, fill = {};
      const C = AI.col();
      for (const s of [1, 2, 3]) {
        const m = first ? mean(fvS, fvN, s) : mean(evS, evN, s);
        top[s] = 'V ≈ ' + (m === null ? '–' : AI.fmt(m, 3));
        bot[s] = 'n = ' + (first ? fvN[s] : evN[s]);
      }
      let trail = null, walker = null, note = 'press Play: an episode is walked, then its returns are carried back';
      let endLabel = ['r = 0', 'r = ' + AI.fmt(rS(), 1)];
      if (ep) {
        const k = X.phase === 'walk' ? X.k : ep.T;
        trail = ep.P.slice(0, k + 1);
        walker = X.phase === 'walk' ? ep.P[X.k] : ep.P[ep.T];
        for (const s of ep.P.slice(0, k + 1)) if (s >= 1 && s <= 3) fill[s] = C.orange;
        if (X.phase === 'back') {
          for (let t = ep.T - 1; t >= 0 && t >= ep.T - X.shown; t--) {
            const s = ep.P[t];
            if (s >= 1 && s <= 3) bot[s] = 'G = ' + AI.fmt(ep.G[t], 3) + ' (t=' + t + ')';
          }
          note = 'returns carried back: G_t = R_{t+1} + γ G_{t+1}, from the last step to the first';
        } else {
          note = 'walking: cell ' + ep.P[X.k] + ' at step ' + X.k + ' of ' + ep.T;
        }
      }
      drawCorridor(cv, { top, bot, fill, trail, walker, endLabel, note });
      const row = (s) => {
        const fm = mean(fvS, fvN, s), em = mean(evS, evN, s);
        return `cell ${s}: first-visit ${fm === null ? '–' : AI.fmt(fm, 3)} (n=${fvN[s]}) · every-visit ${em === null ? '–' : AI.fmt(em, 3)} (n=${evN[s]}) · exact ${AI.fmt(EX[s], 3)}`;
      };
      out.textContent =
        `episode ${X.epNo} · mode ${first ? 'first-visit' : 'every-visit'} · γ = ${AI.fmt(gS(), 2)} · r = ${AI.fmt(rS(), 1)}\n` +
        (ep ? `path: ${ep.P.join(' → ')}   rewards: ${ep.R.map((r) => AI.fmt(r, 1)).join(', ')}\n` : '') +
        [1, 2, 3].map(row).join('\n');
    };
    live(root, [cv], () => X.key() + '|' + mode.v + '|' + cfg, update, draw);
  };

  /* ================= 3.2 Monte Carlo control with exploring starts ================= */
  AI.labs['mc-control'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Monte Carlo control: exploring starts versus a fixed start',
      'illustrative · corridor · greedy policy ties go to R · bars show Q(s,a) with the optimal q* as a green tick',
      `<canvas class="cor" style="height:170px"></canvas><canvas class="bars" style="height:150px;margin-top:10px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.cor', view)), cb = AI.canvas(AI.$('canvas.bars', view));
    const gS = slider(ctl, 'Discount γ', 0.5, 1, 0.05, 0.9, '', 2);
    const out = readout(ctl);
    const mode = { v: 0 };
    modeButtons(rowOf(ctl), ['Exploring starts', 'Fixed start (2, R)'], 0, (i) => { mode.v = i; resetAll(); });
    let rng, Q, N, cfg = '', QS = null;
    const greedy = (s) => (Q[s][1] >= Q[s][0] ? 1 : -1);
    const resetStats = () => { rng = AI.rng(SEED + 1); Q = zeroQ(); N = zeroQ(); };
    resetStats();
    const commit = (ep) => firstVisitUpdate(ep, Q, N);
    const Tr = transport(ctl, () => resetAll());
    const resetAll = () => { resetStats(); if (X) X.reset(); };
    const X = runner(() => {
      let s0 = 2, a0 = 1;
      if (mode.v === 0) { s0 = 1 + Math.floor(rng() * 3); a0 = rng() < 0.5 ? -1 : 1; }
      return makeEp(s0, (s, t) => (t === 0 ? a0 : greedy(s)), 1, gS());
    }, commit);
    const update = (dt) => {
      const k = gS();
      if (k !== cfg) { cfg = k; QS = optimalQ(k, 1); resetAll(); }
      X.tick(dt, Tr);
    };
    const draw = () => {
      const ep = X.ep, C = AI.col();
      const top = {}, bot = {}, fill = {};
      for (const s of [1, 2, 3]) {
        top[s] = 'π: ' + (greedy(s) === 1 ? 'R' : 'L');
        bot[s] = 'L ' + AI.fmt(Q[s][0], 3) + ' R ' + AI.fmt(Q[s][1], 3);
      }
      let trail = null, walker = null, note = 'press Play: each episode starts from a new pair, then follows the greedy policy';
      if (ep) {
        const k = X.phase === 'walk' ? X.k : ep.T;
        trail = ep.P.slice(0, k + 1);
        walker = X.phase === 'walk' ? ep.P[X.k] : ep.P[ep.T];
        for (const s of ep.P.slice(0, k + 1)) if (s >= 1 && s <= 3) fill[s] = C.orange;
        note = mode.v === 0
          ? `exploring start (${ep.P[0]}, ${ep.A[0] === 1 ? 'R' : 'L'}); later steps follow the greedy policy`
          : 'fixed start (2, R): only the R actions are ever tried';
      }
      drawCorridor(cv, { top, bot, fill, trail, walker, endLabel: ['r = 0', 'r = 1'], note });
      const items = [];
      for (const s of [1, 2, 3]) {
        items.push({ label: s + 'L', value: Q[s][0], sub: 'n=' + N[s][0], color: C.purple, mark: QS[s][0] });
        items.push({ label: s + 'R', value: Q[s][1], sub: 'n=' + N[s][1], color: C.blue, mark: QS[s][1] });
      }
      drawBars(cb, items, { vmax: 1.1, step: 0.25, title: 'Q(s,a) bars · green tick = optimal q* (γ shown above)' });
      out.textContent =
        `episode ${X.epNo} · mode ${mode.v === 0 ? 'exploring starts' : 'fixed start (2, R)'} · γ = ${AI.fmt(gS(), 2)}\n` +
        [1, 2, 3].map((s) => `cell ${s}: greedy ${greedy(s) === 1 ? 'R' : 'L'}   Q(L) ${AI.fmt(Q[s][0], 3)} vs q* ${AI.fmt(QS[s][0], 3)}   Q(R) ${AI.fmt(Q[s][1], 3)} vs q* ${AI.fmt(QS[s][1], 3)}`).join('\n');
    };
    live(root, [cv, cb], () => X.key() + '|' + mode.v + '|' + cfg, update, draw);
  };

  /* ================= 3.3 On-policy control with epsilon-soft policies ================= */
  AI.labs['mc-onpolicy'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'On-policy Monte Carlo with an ε-greedy policy',
      'illustrative · corridor · start in a random interior cell · the same policy generates and is improved by the data',
      `<canvas class="cor" style="height:170px"></canvas><canvas class="bars" style="height:150px;margin-top:10px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.cor', view)), cb = AI.canvas(AI.$('canvas.bars', view));
    const eS = slider(ctl, 'Exploration ε', 0, 0.5, 0.01, 0.2, '', 2);
    const out = readout(ctl);
    let rng, Q, N;
    const resetStats = () => { rng = AI.rng(SEED + 3); Q = zeroQ(); N = zeroQ(); };
    resetStats();
    const greedy = (s) => (Q[s][1] >= Q[s][0] ? 1 : -1);
    const pR = (s, e) => (greedy(s) === 1 ? 1 - e + e / 2 : e / 2);
    const Tr = transport(ctl, () => { resetStats(); X.reset(); });
    const X = runner(() => {
      const e = eS(), s0 = 1 + Math.floor(rng() * 3);
      return makeEp(s0, (s) => (rng() < pR(s, e) ? 1 : -1), 1, 0.9);
    }, (ep) => firstVisitUpdate(ep, Q, N));
    const update = (dt) => X.tick(dt, Tr);
    const draw = () => {
      const ep = X.ep, C = AI.col(), e = eS();
      const top = {}, bot = {}, fill = {};
      for (const s of [1, 2, 3]) {
        top[s] = 'π(R) = ' + AI.fmt(pR(s, e), 3);
        bot[s] = 'Q L ' + AI.fmt(Q[s][0], 2) + ' R ' + AI.fmt(Q[s][1], 2);
      }
      let trail = null, walker = null, note = 'ε-greedy: the greedy action gets 1 − ε + ε/2, the other gets ε/2';
      if (ep) {
        const k = X.phase === 'walk' ? X.k : ep.T;
        trail = ep.P.slice(0, k + 1);
        walker = X.phase === 'walk' ? ep.P[X.k] : ep.P[ep.T];
        for (const s of ep.P.slice(0, k + 1)) if (s >= 1 && s <= 3) fill[s] = C.orange;
        note = 'episode ' + X.epNo + ': actions ' + ep.A.map((a) => (a === 1 ? 'R' : 'L')).join(' ') + ' · G0 = ' + AI.fmt(ep.G[0], 3);
      }
      drawCorridor(cv, { top, bot, fill, trail, walker, endLabel: ['r = 0', 'r = 1'], note });
      const items = [1, 2, 3].map((s) => ({ label: 'π(R|' + s + ')', value: pR(s, e), color: C.blue }));
      drawBars(cb, items, { vmax: 1, step: 0.25, target: 1 - e / 2, targetLabel: 'limit 1 − ε/2 = ' + AI.fmt(1 - e / 2, 3), title: 'probability of R in each cell' });
      const v2 = (1 - pR(2, e)) * Q[2][0] + pR(2, e) * Q[2][1];
      out.textContent =
        `episode ${X.epNo} · ε = ${AI.fmt(e, 2)}\n` +
        `policy in cell 2: π(L) = ${AI.fmt(1 - pR(2, e), 3)}, π(R) = ${AI.fmt(pR(2, e), 3)}\n` +
        `expected Q in cell 2 under the policy = ${AI.fmt(v2, 4)}\n` +
        `best ε-soft limit for π(R): 1 − ε/2 = ${AI.fmt(1 - e / 2, 3)} (R is the better action in every cell)`;
    };
    live(root, [cv, cb], () => X.key() + '|' + eS(), update, draw);
  };

  /* ================= 3.4 Importance sampling ================= */
  AI.labs['mc-is'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Importance sampling: reweighting logged episodes',
      'illustrative · corridor from cell 2 · behaviour: R and L each 0.5 · target: π(R|s) = p in every cell · true value of cell 2 under π shown in green',
      `<canvas class="cor" style="height:170px"></canvas><canvas class="plt" style="height:230px;margin-top:10px"></canvas>`, '');
    const cC = AI.canvas(AI.$('canvas.cor', view)), cP = AI.canvas(AI.$('canvas.plt', view));
    const pS = slider(ctl, 'Target probability of R, p', 0.5, 1, 0.05, 1, '', 2);
    const out = readout(ctl);
    let rng, EPS;
    const ratio = (a, p) => (a === 1 ? p : 1 - p) / B;
    const weight = (ep, p, upto) => { let w = 1; for (let t = 0; t < upto; t++) w *= ratio(ep.A[t], p); return w; };
    const reset = () => { rng = AI.rng(SEED + 4); EPS = []; X.reset(); };
    const Tr = transport(ctl, reset);
    const X = runner(() => makeEp(2, () => (rng() < B ? 1 : -1), 1, 0.9), (ep) => EPS.push(ep));
    reset();
    const series = (p) => {
      let num = 0, den = 0;
      const ord = [], wt = [];
      EPS.forEach((ep, i) => {
        const w = weight(ep, p, ep.T);
        num += w * ep.G[0]; den += w;
        ord.push(num / (i + 1)); wt.push(den > 0 ? num / den : null);
      });
      return { ord, wt };
    };
    const update = (dt) => X.tick(dt, Tr);
    const draw = () => {
      const p = pS(), C = AI.col(), ep = X.ep;
      const truth = exactV(0.9, 1, () => p)[2];
      const top = {}, bot = {}, fill = {};
      let trail = null, walker = null, note = 'press Play: episodes are drawn from the behaviour policy';
      if (ep) {
        const k = X.phase === 'walk' ? X.k : ep.T;
        trail = ep.P.slice(0, k + 1);
        walker = X.phase === 'walk' ? ep.P[X.k] : ep.P[ep.T];
        for (const s of ep.P.slice(0, k + 1)) if (s >= 1 && s <= 3) fill[s] = C.orange;
        note = X.phase === 'walk'
          ? `weight so far after ${X.k} step(s): ρ = ${AI.fmt(weight(ep, p, X.k), 4)}`
          : `returned G0 = ${AI.fmt(ep.G[0], 3)} · full weight ρ = ${AI.fmt(weight(ep, p, ep.T), 4)}`;
      }
      drawCorridor(cC, { top, bot, fill, trail, walker, endLabel: ['r = 0', 'r = 1'], note });
      const N = Math.max(60, EPS.length + 5);
      const v = AI.view(cP, 0, N, -0.5, 2.5);
      const g = cP.ctx;
      g.clearRect(0, 0, cP.w, cP.h); AI.axes(g, v);
      const sr = series(p);
      const Y = (y) => v.Y(AI.clamp(y, v.ymin, v.ymax));
      g.strokeStyle = C.green; g.setLineDash([6, 4]); g.lineWidth = 2; g.beginPath();
      g.moveTo(v.X(0), Y(truth)); g.lineTo(v.X(N), Y(truth)); g.stroke(); g.setLineDash([]);
      const line = (arr, color) => {
        g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); let pen = false;
        arr.forEach((y, i) => { if (y === null) { pen = false; return; } const px = v.X(i + 1), py = Y(y); if (!pen) { g.moveTo(px, py); pen = true; } else g.lineTo(px, py); });
        g.stroke();
      };
      line(sr.ord, C.orange);
      line(sr.wt, C.blue);
      g.font = '11px ' + MONO; g.textAlign = 'left'; g.fillStyle = C.orange; g.fillText('ordinary (unbiased, volatile)', v.pad.l + 8, v.pad.t + 14);
      g.fillStyle = C.blue; g.fillText('weighted (biased for small n, steadier)', v.pad.l + 8, v.pad.t + 28);
      g.fillStyle = C.green; g.fillText('true v_π(2) = ' + AI.fmt(truth, 3), v.pad.l + 8, v.pad.t + 42);
      g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('episodes n', v.X(N / 2), v.pad.t + v.ph + 26);
      const n = EPS.length, last = sr.ord.length ? sr.ord[n - 1] : null, lastW = sr.wt.length ? sr.wt[n - 1] : null;
      out.textContent =
        `episodes n = ${n} · p = ${AI.fmt(p, 2)} · behaviour 0.5 for each action\n` +
        `ordinary estimate = ${last === null ? '–' : AI.fmt(last, 4)}\n` +
        `weighted estimate = ${lastW === null ? '–' : AI.fmt(lastW, 4)}\n` +
        `true value of cell 2 under π = ${AI.fmt(truth, 4)}`;
    };
    live(root, [cC, cP], () => X.key() + '|' + pS() + '|' + EPS.length, update, draw);
  };

  /* ================= 3.5 Off-policy Monte Carlo control ================= */
  AI.labs['mc-offpolicy'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Off-policy control: learn the greedy policy from random episodes',
      'illustrative · corridor · behaviour: R and L each 0.5 · target: greedy with ties to R · the backward pass stops at the first action the target would not take',
      `<canvas class="cor" style="height:170px"></canvas><canvas class="bars" style="height:150px;margin-top:10px"></canvas>`, '');
    const cv = AI.canvas(AI.$('canvas.cor', view)), cb = AI.canvas(AI.$('canvas.bars', view));
    const out = readout(ctl);
    let rng, Q, C2, QS;
    const greedy = (s) => (Q[s][1] >= Q[s][0] ? 1 : -1);
    const reset = () => { rng = AI.rng(SEED + 5); Q = zeroQ(); C2 = zeroQ(); X.reset(); };
    const Tr = transport(ctl, reset);
    // weighted incremental update run on copies; returns the trace used for the animation
    const plan = (ep) => {
      const Qc = zeroQ(), Cc = zeroQ();
      for (const s of [1, 2, 3]) { Qc[s] = Q[s].slice(); Cc[s] = C2[s].slice(); }
      const trace = [];
      let W = 1;
      for (let t = ep.T - 1; t >= 0; t--) {
        const s = ep.P[t], a = ep.A[t], ai = a === 1 ? 1 : 0;
        Cc[s][ai] += W;
        Qc[s][ai] += (W / Cc[s][ai]) * (ep.G[t] - Qc[s][ai]);
        const gr = Qc[s][1] >= Qc[s][0] ? 1 : -1;
        const stop = a !== gr;
        trace.push({ t, s, a, G: ep.G[t], W, C: Cc[s][ai], Qv: Qc[s][ai], gr, stop });
        if (stop) break;
        W = W / B;
      }
      return { trace, Qc, Cc };
    };
    const X = runner(() => {
      const ep = makeEp(1 + Math.floor(rng() * 3), () => (rng() < B ? 1 : -1), 1, 0.9);
      ep.plan = plan(ep);
      return ep;
    }, (ep) => { Q = ep.plan.Qc; C2 = ep.plan.Cc; });
    reset();
    const update = (dt) => X.tick(dt, Tr);
    const draw = () => {
      const ep = X.ep, C = AI.col();
      const top = {}, bot = {}, fill = {};
      for (const s of [1, 2, 3]) {
        top[s] = 'greedy: ' + (greedy(s) === 1 ? 'R' : 'L');
        bot[s] = 'Q L ' + AI.fmt(Q[s][0], 2) + ' R ' + AI.fmt(Q[s][1], 2);
      }
      let trail = null, walker = null, note = 'press Play: a random episode, then its backward pass';
      const lines = [];
      if (ep) {
        const k = X.phase === 'walk' ? X.k : ep.T;
        trail = ep.P.slice(0, k + 1);
        walker = X.phase === 'walk' ? ep.P[X.k] : ep.P[ep.T];
        for (const s of ep.P.slice(0, k + 1)) if (s >= 1 && s <= 3) fill[s] = C.orange;
        if (X.phase === 'back') {
          const shown = Math.min(X.shown, ep.plan.trace.length);
          ep.plan.trace.slice(0, shown).forEach((e) => lines.push(
            `t=${e.t} (${e.s},${e.a === 1 ? 'R' : 'L'}) G=${AI.fmt(e.G, 3)} W=${AI.fmt(e.W, 2)} C=${AI.fmt(e.C, 2)} Q=${AI.fmt(e.Qv, 3)} greedy ${e.gr === 1 ? 'R' : 'L'}` + (e.stop ? '  -> action differs: STOP' : '  -> continue')));
          note = shown < ep.plan.trace.length ? 'backward pass: updating the step with weight W' : 'pass finished: earlier steps are not updated';
        } else {
          note = 'episode walked under the behaviour policy: ' + ep.A.map((a) => (a === 1 ? 'R' : 'L')).join(' ');
        }
      }
      drawCorridor(cv, { top, bot, fill, trail, walker, endLabel: ['r = 0', 'r = 1'], note });
      const items = [];
      for (const s of [1, 2, 3]) {
        items.push({ label: s + 'L', value: Q[s][0], sub: 'C=' + AI.fmt(C2[s][0], 2), color: C.purple });
        items.push({ label: s + 'R', value: Q[s][1], sub: 'C=' + AI.fmt(C2[s][1], 2), color: C.blue });
      }
      drawBars(cb, items, { vmax: 1.1, step: 0.25, title: 'weighted Q(s,a) bars and cumulative weights C(s,a)' });
      out.textContent =
        `episode ${X.epNo} · greedy policy: cell 1 ${greedy(1) === 1 ? 'R' : 'L'}, cell 2 ${greedy(2) === 1 ? 'R' : 'L'}, cell 3 ${greedy(3) === 1 ? 'R' : 'L'}\n` +
        (lines.length ? lines.join('\n') : 'backward pass: none yet');
    };
    live(root, [cv, cb], () => X.key() + '|' + X.epNo, update, draw);
  };

  /* ================= 3.6 Strengths and limits of Monte Carlo ================= */
  AI.labs['mc-error'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Monte Carlo error versus number of episodes',
      'illustrative · corridor, uniform random policy, estimate of V(2) from first returns · 30 seeded runs, log-log axes',
      `<canvas class="plt" style="height:240px"></canvas><canvas class="strip" style="height:110px;margin-top:10px"></canvas>`, '');
    const cP = AI.canvas(AI.$('canvas.plt', view)), cS = AI.canvas(AI.$('canvas.strip', view));
    const gS = slider(ctl, 'Discount γ', 0.5, 0.95, 0.05, 0.9, '', 2);
    const out = readout(ctl);
    const NMAX = 300, SEEDS = 30;
    let cfg = '', V2 = 0, err = null, est = null, sdAt = null, meanAt = null;
    const build = (g) => {
      V2 = exactV(g, 1, () => 0.5)[2];
      est = [];
      for (let i = 0; i < SEEDS; i++) {
        const r = AI.rng(1000 + i);
        let sum = 0; const e = new Array(NMAX + 1).fill(0);
        for (let n = 1; n <= NMAX; n++) { sum += makeEp(2, () => (r() < 0.5 ? -1 : 1), 1, g).G[0]; e[n] = sum / n; }
        est.push(e);
      }
      err = new Array(NMAX + 1).fill(0); sdAt = new Array(NMAX + 1).fill(0); meanAt = new Array(NMAX + 1).fill(0);
      for (let n = 1; n <= NMAX; n++) {
        let se = 0, m = 0;
        for (let i = 0; i < SEEDS; i++) { se += Math.abs(est[i][n] - V2); m += est[i][n]; }
        m /= SEEDS;
        let v = 0; for (let i = 0; i < SEEDS; i++) v += (est[i][n] - m) ** 2;
        err[n] = se / SEEDS; meanAt[n] = m; sdAt[n] = Math.sqrt(v / SEEDS);
      }
    };
    const Tr = transport(ctl, () => { X.n = 0; });
    const X = { n: 0, acc: 0, tk: AI.ticker() };
    const update = (dt) => {
      const k = gS();
      if (k !== cfg) { cfg = k; build(k); X.n = 1; }
      if (Tr.pending) { X.n = Math.min(NMAX, X.n + Tr.pending); Tr.pending = 0; }
      if (Tr.playing) X.n = Math.min(NMAX, X.n + X.tk(dt, 8 * Tr.speed()));
    };
    const draw = () => {
      const C = AI.col(), n = X.n, g = cP.ctx;
      const v = AI.view(cP, 0, 2.5, -2.5, 0.5);
      g.clearRect(0, 0, cP.w, cP.h);
      // grid and labels on log axes
      g.font = '11px ' + MONO; g.strokeStyle = C.border; g.fillStyle = C.muted;
      [[1, '1'], [10, '10'], [100, '100']].forEach(([nn, lab]) => {
        const x = v.X(Math.log10(nn));
        g.beginPath(); g.moveTo(x + 0.5, v.pad.t); g.lineTo(x + 0.5, v.pad.t + v.ph); g.stroke();
        g.textAlign = 'center'; g.fillText(lab, x, v.pad.t + v.ph + 15);
      });
      [[0.01, '0.01'], [0.1, '0.1'], [1, '1']].forEach(([e, lab]) => {
        const y = v.Y(Math.log10(e));
        g.beginPath(); g.moveTo(v.pad.l, y + 0.5); g.lineTo(v.pad.l + v.pw, y + 0.5); g.stroke();
        g.textAlign = 'right'; g.fillText(lab, v.pad.l - 5, y + 4);
      });
      g.textAlign = 'center'; g.fillText('episodes n (log scale)', v.X(1.25), v.pad.t + v.ph + 30);
      g.save(); g.translate(12, v.pad.t + v.ph / 2); g.rotate(-Math.PI / 2); g.fillText('mean |error| (log scale)', 0, 0); g.restore();
      // guide with slope -1/2 through the error at n = 10
      const e10 = err[10];
      g.setLineDash([5, 4]); g.strokeStyle = C.green; g.beginPath();
      for (let i = 0; i <= 40; i++) {
        const lx = (i / 40) * 2.5; const ly = Math.log10(e10) - 0.5 * (lx - 1);
        const px = v.X(lx), py = v.Y(AI.clamp(ly, v.ymin, v.ymax));
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      }
      g.stroke(); g.setLineDash([]);
      g.fillStyle = C.green; g.textAlign = 'left'; g.fillText('guide: slope −1/2 (error ∝ 1/√n)', v.pad.l + 8, v.pad.t + 14);
      // error curve drawn up to n
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath();
      for (let k = 1; k <= Math.max(1, n); k++) {
        const px = v.X(Math.log10(k)), py = v.Y(AI.clamp(Math.log10(Math.max(err[k], 1e-9)), v.ymin, v.ymax));
        k === 1 ? g.moveTo(px, py) : g.lineTo(px, py);
      }
      g.stroke();
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(Math.log10(Math.max(1, n))), v.Y(AI.clamp(Math.log10(Math.max(err[Math.max(1, n)], 1e-9)), v.ymin, v.ymax)), 4, 0, 6.2832); g.fill();
      g.fillStyle = C.orange; g.textAlign = 'left'; g.fillText('mean |error| over 30 seeds', v.pad.l + 8, v.pad.t + 28);
      // strip: each seed's estimate after n episodes
      const s = cS.ctx;
      s.clearRect(0, 0, cS.w, cS.h);
      const sv = AI.view(cS, 0, 0.9, 0, 1, { l: 24, r: 14, t: 16, b: 26 });
      s.strokeStyle = C.border; s.fillStyle = C.muted; s.font = '11px ' + MONO; s.textAlign = 'center';
      for (let x = 0; x <= 0.9 + 1e-9; x += 0.1) { s.beginPath(); s.moveTo(sv.X(x) + 0.5, sv.pad.t); s.lineTo(sv.X(x) + 0.5, sv.pad.t + sv.ph); s.stroke(); s.fillText(AI.fmt(+x.toFixed(1), 1), sv.X(x), sv.pad.t + sv.ph + 15); }
      s.fillStyle = C.orange; s.globalAlpha = 0.75;
      for (let i = 0; i < SEEDS; i++) { s.beginPath(); s.arc(sv.X(AI.clamp(est[i][Math.max(1, n)], 0, 0.9)), sv.Y(0.5 + ((i % 5) - 2) * 0.12), 4, 0, 6.2832); s.fill(); }
      s.globalAlpha = 1;
      s.strokeStyle = C.green; s.lineWidth = 2; s.beginPath(); s.moveTo(sv.X(V2), sv.pad.t); s.lineTo(sv.X(V2), sv.pad.t + sv.ph); s.stroke();
      s.fillStyle = C.green; s.textAlign = 'left'; s.fillText('exact V(2) = ' + AI.fmt(V2, 3), sv.X(V2) + 6, sv.pad.t + 12);
      s.fillStyle = C.muted; s.textAlign = 'left'; s.fillText('each dot: one seed’s estimate after n episodes', sv.pad.l, sv.pad.t - 4);
      const nn = Math.max(1, n);
      out.textContent =
        `episodes n = ${n} · γ = ${AI.fmt(gS(), 2)} · exact V(2) = ${AI.fmt(V2, 4)}\n` +
        `mean |error| over 30 seeds = ${AI.fmt(err[nn], 4)}\n` +
        `average estimate over seeds = ${AI.fmt(meanAt[nn], 4)}  (unbiased: close to the exact value)\n` +
        `spread across seeds: sd = ${AI.fmt(sdAt[nn], 4)}, sd × √n = ${AI.fmt(sdAt[nn] * Math.sqrt(nn), 3)} (roughly constant)`;
    };
    live(root, [cP, cS], () => X.n + '|' + cfg, update, draw);
  };
})();
