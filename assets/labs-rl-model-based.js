/* Reinforcement Learning · Module 7 (Model-based RL) labs. All are illustrative toys with fixed seeds; each states its simplifications.
 * rlmb-dyna   <div data-lab="rlmb-dyna"></div>    Dyna-Q on a seven-cell corridor: real steps, model, planning sweeps
 * rlmb-mcts   <div data-lab="rlmb-mcts"></div>    UCT tree search on a two-decision tree: selection, expansion, simulation, backup
 * rlmb-error  <div data-lab="rlmb-error"></div>   pendulum with a wrong gravity: open-loop versus branched model rollouts
 * rlmb-mpc    <div data-lab="rlmb-mpc"></div>     random-shooting MPC on a 2-D point mass with a learned (wrong) gain
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';

  /* ---------- shared helpers (same pattern as the other lab files) ---------- */
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

  /* transport: play/pause, step, reset and a speed multiplier. */
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

  /* ================= 7.1 Dyna-Q ================= */
  AI.labs['rlmb-dyna'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Dyna-Q on a seven-cell corridor',
      'illustrative · tabular Dyna-Q, start at s0, goal s6 (reward +1 on entering), γ = 0.95, fixed seed 7 · each dot is one real step',
      `<canvas class="cor" style="height:150px"></canvas><canvas class="cur" style="height:210px;margin-top:10px"></canvas>`, '');
    const cC = AI.canvas(AI.$('canvas.cor', view)), cU = AI.canvas(AI.$('canvas.cur', view));
    const nS = slider(ctl, 'Planning steps n per real step', 0, 50, 1, 10, '', 0);
    const aS = slider(ctl, 'Step size α', 0.1, 1, 0.05, 0.5, '', 2);
    const eS = slider(ctl, 'Exploration ε', 0, 0.5, 0.05, 0.1, '', 2);
    const out = readout(ctl);
    const GAMMA = 0.95, NS_ = 7, GOAL = 6, EPIS = 30, CAP = 200, SEED = 7;

    /* one full run; when record is true, every real step is stored for animation */
    function runDyna(n, alpha, eps, record) {
      const rng = AI.rng(SEED);
      const Q = Array.from({ length: NS_ }, () => [0, 0]);
      const model = new Map(), seen = [];
      const events = [], ends = [], steps = [];
      for (let ep = 0; ep < EPIS; ep++) {
        let s = 0, k = 0;
        while (s !== GOAL && k < CAP) {
          let a;
          if (rng() < eps) a = rng() < 0.5 ? 0 : 1;
          else a = Q[s][0] > Q[s][1] ? 0 : Q[s][1] > Q[s][0] ? 1 : (rng() < 0.5 ? 0 : 1);
          const s2 = a === 1 ? Math.min(s + 1, GOAL) : Math.max(s - 1, 0);
          const r = s2 === GOAL ? 1 : 0;
          const term = s2 === GOAL;
          /* direct Q-learning update */
          Q[s][a] += alpha * (r + (term ? 0 : GAMMA * Math.max(Q[s2][0], Q[s2][1])) - Q[s][a]);
          /* model update: store the last observed (r, s') for the pair */
          const key = s * 2 + a;
          if (!model.has(key)) seen.push(key);
          model.set(key, [r, s2]);
          k++;
          /* n planning updates, each replaying a uniformly sampled stored pair */
          for (let i = 0; i < n; i++) {
            const pk = seen[Math.floor(rng() * seen.length)];
            const [pr, ps2] = model.get(pk);
            const ps = pk >> 1, pa = pk & 1;
            Q[ps][pa] += alpha * (pr + (ps2 === GOAL ? 0 : GAMMA * Math.max(Q[ps2][0], Q[ps2][1])) - Q[ps][pa]);
          }
          if (record) {
            events.push({
              ep, k, s, a, s2, r, plan: n,
              qmax: Q.map((q) => Math.max(q[0], q[1])),
              arg: Q.map((q) => (q[1] > q[0] ? 1 : q[0] > q[1] ? 0 : -1)),
            });
          }
          s = s2;
        }
        steps.push(k);
        if (record) ends.push(events.length);
      }
      return { steps, events, ends };
    }

    let R = null, B = null, key = '', idx = 0;
    const refresh = () => {
      const n = Math.round(nS()), al = aS(), ep = eS();
      const k = n + '|' + al + '|' + ep;
      if (k === key) return;
      key = k;
      R = runDyna(n, al, ep, true);
      B = runDyna(0, al, ep, false);
      idx = 0;
    };
    const T = transport(ctl, () => { idx = 0; T.setPlaying(true); });
    const skipBtn = button(rowOf(ctl), 'Skip to end of episode');
    skipBtn.onclick = () => { refresh(); idx = R.ends.find((e) => e > idx) ?? R.events.length; T.setPlaying(false); };
    const tk = AI.ticker();

    const update = (dt) => {
      refresh();
      const len = R.events.length;
      if (T.playing) idx = Math.min(len, idx + tk(dt, 1.2 * T.speed()));
      if (T.pending) { idx = Math.min(len, idx + T.pending); T.pending = 0; }
      if (idx >= len && T.playing) T.setPlaying(false);
    };

    const draw = () => {
      refresh();
      const C = AI.col();
      const last = idx > 0 ? R.events[idx - 1] : null;
      /* corridor */
      const g = cC.ctx, W = cC.w, H = cC.h;
      g.clearRect(0, 0, W, H);
      const qmax = last ? last.qmax : new Array(NS_).fill(0);
      const arg = last ? last.arg : new Array(NS_).fill(-1);
      const pad = 10, cw = (W - 2 * pad) / NS_;
      const bx = (i) => pad + i * cw;
      g.font = '12px ' + MONO; g.textAlign = 'center';
      for (let i = 0; i < NS_; i++) {
        const cx = bx(i) + cw / 2;
        if (arg[i] === 1) AI.arrow(g, cx - 12, 24, cx + 12, 24, C.blue, 2);
        else if (arg[i] === 0) AI.arrow(g, cx + 12, 24, cx - 12, 24, C.blue, 2);
        g.fillStyle = i === GOAL ? AI.mix(C.bg2, C.green, 0.5) : C.bg2;
        g.fillRect(bx(i) + 4, 40, cw - 8, 34);
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(bx(i) + 4.5, 40.5, cw - 9, 33);
        g.fillStyle = C.fg; g.fillText('s' + i, cx, 62);
        const h = Math.max(0, qmax[i]) * 50, base = H - 8;
        g.fillStyle = C.blue; g.globalAlpha = 0.7; g.fillRect(cx - 9, base - h, 18, h); g.globalAlpha = 1;
        g.fillStyle = C.muted; g.fillText(AI.fmt(qmax[i], 2), cx, base - h - 4);
      }
      const ai = last ? last.s2 : 0;
      g.fillStyle = C.orange; g.beginPath(); g.arc(bx(ai) + cw / 2, 40, 6, 0, 6.2832); g.fill();
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('bars: max Q(s,·)   arrows: greedy action   dot: agent', 10, 12);

      /* steps per episode */
      const c = cU, v = AI.view(c, 0.5, EPIS + 0.5, 0, CAP, { l: 46, r: 12, t: 12, b: 28 }), cg = c.ctx;
      cg.clearRect(0, 0, c.w, c.h);
      AI.axes(cg, v);
      cg.fillStyle = C.muted; cg.textAlign = 'center';
      cg.fillText('episode', v.X((EPIS + 1) / 2), c.h - 6);
      const line = (arr, color, dash, upTo) => {
        cg.save(); cg.strokeStyle = color; cg.lineWidth = 2; if (dash) cg.setLineDash(dash);
        cg.beginPath();
        arr.forEach((y, i) => { if (i >= upTo) return; const px = v.X(i + 1), py = v.Y(AI.clamp(y, 0, CAP)); if (i) cg.lineTo(px, py); else cg.moveTo(px, py); });
        cg.stroke(); cg.restore();
        arr.forEach((y, i) => { if (i < upTo) { cg.fillStyle = color; cg.beginPath(); cg.arc(v.X(i + 1), v.Y(AI.clamp(y, 0, CAP)), 3, 0, 6.2832); cg.fill(); } });
      };
      line(B.steps, C.muted, [5, 4], EPIS);
      const doneEps = R.ends.filter((e) => e <= idx).length;
      line(R.steps, C.blue, null, doneEps);
      cg.fillStyle = C.muted; cg.textAlign = 'left';
      cg.fillText('grey dashed: n = 0 (Q-learning only) · blue: current n', v.pad.l + 8, v.pad.t + 14);

      const avg = (arr, m) => arr.slice(0, m).reduce((a, b) => a + b, 0) / Math.min(m, arr.length);
      const n = Math.round(nS());
      const cap = !last
        ? 'Press play. The agent starts at s0 and each real step is followed by n simulated updates.'
        : `Real step ${last.k} of episode ${last.ep + 1}: s${last.s} → s${last.s2} (${last.a ? 'R' : 'L'}), reward ${last.r}. ` +
          `The model stores this pair. Then ${last.plan} planning updates replay stored pairs.`;
      out.textContent =
        `${cap}\n` +
        `updates per real step: ${1 + n}  (1 direct + ${n} planning)\n` +
        `episode ${last ? last.ep + 1 : 1}  ·  events shown ${idx} of ${R.events.length}\n` +
        `steps to goal, first 5 episodes: current n = [${R.steps.slice(0, 5).join(', ')}]   n = 0 = [${B.steps.slice(0, 5).join(', ')}]\n` +
        `mean steps over the first 10 episodes: current n ${AI.fmt(avg(R.steps, 10), 1)}, n = 0 ${AI.fmt(avg(B.steps, 10), 1)}`;
    };

    live(root, [cC, cU], () => key + '|' + idx, update, draw);
  };

  /* ================= 7.2 Monte Carlo tree search ================= */
  AI.labs['rlmb-mcts'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'UCT tree search on a two-decision tree',
      'illustrative · decision 1 picks A, B or C; decision 2 picks x or y; leaf payoffs A: 0.2 / 0.8, B: 0.5 / 0.4, C: 0.9 / 0.1 · random completion in simulation',
      `<canvas class="tr" style="height:330px"></canvas>`, '');
    const cT = AI.canvas(AI.$('canvas.tr', view));
    const cExp = slider(ctl, 'Exploration constant c', 0, 3, 0.05, Math.SQRT2, '', 2);
    const out = readout(ctl);

    const PAY = { A: { x: 0.2, y: 0.8 }, B: { x: 0.5, y: 0.4 }, C: { x: 0.9, y: 0.1 } };
    const ACT1 = ['A', 'B', 'C'], ACT2 = ['x', 'y'];
    const PH = 0.8, CYCLE = 3.2, MAXIT = 300;
    const numActs = (d) => (d === 0 ? 3 : 2);
    let rng, rootN, iters, last, elapsed, sigKey = '';

    const mk = (act, depth, parent) => ({ act, depth, parent, children: [], N: 0, W: 0, payoff: depth === 2 ? PAY[parent.act][act] : null });
    const labelOf = (nd) => (nd.depth === 0 ? 'root' : nd.depth === 1 ? nd.act : nd.parent.act + nd.act);
    const uct = (p, ch, cc) => ch.W / ch.N + cc * Math.sqrt(Math.log(p.N) / ch.N);

    function iterate() {
      const cc = cExp();
      const path = [rootN];
      let node = rootN, newNode = null, sim = null, payoff = 0, scores = null;
      while (true) {
        if (node.depth === 2) { payoff = node.payoff; sim = { leaf: labelOf(node), rolled: null }; break; }
        if (node.children.length < numActs(node.depth)) {
          const ch = mk(node.depth === 0 ? ACT1[node.children.length] : ACT2[node.children.length], node.depth + 1, node);
          node.children.push(ch); newNode = ch; path.push(ch);
          if (ch.depth === 2) { payoff = ch.payoff; sim = { leaf: labelOf(ch), rolled: null }; }
          else {
            const j = rng() < 0.5 ? 0 : 1;
            payoff = PAY[ch.act][ACT2[j]];
            sim = { leaf: ch.act + ACT2[j], rolled: ACT2[j] };
          }
          break;
        }
        if (node === rootN) scores = node.children.map((c) => ({ label: labelOf(c), v: uct(node, c, cc) }));
        let best = null, bv = -Infinity;
        for (const c of node.children) { const v = uct(node, c, cc); if (v > bv + 1e-12) { bv = v; best = c; } }
        node = best; path.push(node);
      }
      for (const p of path) { p.N++; p.W += payoff; }
      iters++;
      last = { path, newNode, sim, payoff, scores };
    }

    const reset = () => {
      rng = AI.rng(5);
      rootN = mk('', 0, null); iters = 0; last = null; elapsed = CYCLE;
      T.setPlaying(true);
    };
    const T = transport(ctl, reset);
    reset();

    const allNodes = () => {
      const out = [];
      const walk = (nd) => { out.push(nd); nd.children.forEach(walk); };
      walk(rootN);
      return out;
    };
    const posOf = (nd, w, h) => {
      if (nd.depth === 0) return [w / 2, 38];
      const i = ACT1.indexOf(nd.depth === 1 ? nd.act : nd.parent.act);
      const x1 = w * (0.2 + 0.3 * i);
      if (nd.depth === 1) return [x1, 140];
      const j = ACT2.indexOf(nd.act);
      return [x1 + (j === 0 ? -0.09 : 0.09) * w, 250];
    };

    const update = (dt) => {
      if (T.pending) { T.pending = 0; if (iters < MAXIT) { iterate(); elapsed = 0; } }
      if (!T.playing) return;
      elapsed += dt * T.speed();
      if (elapsed >= CYCLE) {
        if (iters >= MAXIT) { T.setPlaying(false); return; }
        iterate(); elapsed = 0;
      }
    };

    const draw = () => {
      const C = AI.col(), g = cT.ctx, w = cT.w, h = cT.h;
      g.clearRect(0, 0, w, h);
      const ph = Math.min(3, Math.floor(elapsed / PH));
      const inCycle = elapsed < CYCLE && last;
      const onPath = new Set(last ? last.path : []);
      const nodes = allNodes();
      /* edges */
      g.lineWidth = 2;
      for (const nd of nodes) {
        if (!nd.parent) continue;
        const [x0, y0] = posOf(nd.parent, w, h), [x1, y1] = posOf(nd, w, h);
        const hot = inCycle && onPath.has(nd) && nd.parent && onPath.has(nd.parent);
        g.strokeStyle = hot && ph < 3 ? C.orange : C.border;
        g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      }
      /* unexpanded slots (dashed) */
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.fillStyle = C.muted; g.font = '12px ' + MONO; g.textAlign = 'center';
      const slots = [rootN];
      if (rootN.children.length < 3) {
        for (let i = rootN.children.length; i < 3; i++) {
          const x = w * (0.2 + 0.3 * i);
          g.beginPath(); g.arc(x, 140, 16, 0, 6.2832); g.stroke();
          g.fillText(ACT1[i], x, 144);
        }
      }
      for (const a of rootN.children) {
        if (a.children.length < 2) {
          const i = ACT1.indexOf(a.act);
          for (let j = a.children.length; j < 2; j++) {
            const x = w * (0.2 + 0.3 * i) + (j === 0 ? -0.09 : 0.09) * w;
            g.beginPath(); g.arc(x, 250, 13, 0, 6.2832); g.stroke();
          }
        }
      }
      g.setLineDash([]);
      /* nodes */
      for (const nd of nodes) {
        const [x, y] = posOf(nd, w, h);
        const isNew = inCycle && last.newNode === nd;
        const r = nd.depth === 2 ? 13 : 16;
        let fill = C.bg2;
        if (isNew && ph >= 1) fill = C.orange;
        else if (inCycle && onPath.has(nd) && ph === 3) fill = C.blue;
        g.fillStyle = fill; g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill();
        g.strokeStyle = onPath.has(nd) && inCycle ? C.orange : C.border; g.lineWidth = 2; g.stroke();
        g.fillStyle = C.fg; g.textAlign = 'center';
        g.fillText(labelOf(nd), x, y + 4);
        g.fillStyle = C.muted; g.font = '11px ' + MONO;
        const txt = nd.depth === 2 ? `payoff ${AI.fmt(nd.payoff, 2)}` : `N=${nd.N}  Q=${nd.N ? AI.fmt(nd.W / nd.N, 2) : '–'}`;
        g.fillText(txt, x, y + (nd.depth === 2 ? 27 : 30));
        g.font = '12px ' + MONO;
      }
      /* simulation: dashed purple path from the new node to the rolled-out leaf */
      if (inCycle && ph >= 2 && last.sim && last.sim.rolled && last.newNode) {
        const [x0, y0] = posOf(last.newNode, w, h);
        const leaf = last.newNode.children.find((c) => c.act === last.sim.rolled) || null;
        const j = ACT2.indexOf(last.sim.rolled);
        const i = ACT1.indexOf(last.newNode.act);
        const x1 = w * (0.2 + 0.3 * i) + (j === 0 ? -0.09 : 0.09) * w;
        g.save(); g.strokeStyle = C.purple; g.setLineDash([5, 4]); g.lineWidth = 2;
        g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, 250); g.stroke(); g.restore();
        g.fillStyle = C.purple; g.textAlign = 'center';
        g.fillText('rollout → ' + last.sim.leaf + ' = ' + AI.fmt(last.payoff, 2), w * 0.5, 300);
      }
      /* caption and table */
      let cap;
      if (!last) cap = 'Press play. Each iteration: selection, expansion, simulation, backup.';
      else if (!inCycle) cap = `Iteration ${iters} finished. The next one starts shortly.`;
      else if (ph === 0) cap = `Iteration ${iters} · selection: from the root, follow the highest UCT score`
        + (last.scores ? ' (' + last.scores.map((s) => `${s.label} ${AI.fmt(s.v, 3)}`).join(', ') + ')' : ' (the root still has unexpanded children, so expansion comes first)') + '.';
      else if (ph === 1) cap = `Iteration ${iters} · expansion: add one new child ${last.newNode ? labelOf(last.newNode) : ''}.`;
      else if (ph === 2) cap = `Iteration ${iters} · simulation: ${last.sim.rolled ? 'play the second decision at random to a leaf' : 'the new node is a leaf, so the payoff is read directly'}; payoff ${AI.fmt(last.payoff, 2)}.`;
      else cap = `Iteration ${iters} · backup: add 1 visit and payoff ${AI.fmt(last.payoff, 2)} to every node on the path.`;
      const rows = rootN.children.map((a) => `${a.act}: N=${a.N}, W=${AI.fmt(a.W, 2)}, Q=${a.N ? AI.fmt(a.W / a.N, 3) : '–'}`).join('\n');
      const most = rootN.children.length ? rootN.children.reduce((b, c) => (c.N > b.N ? c : b)) : null;
      out.textContent = cap + '\n' + rows +
        `\niterations ${iters} · recommended move (most visited) ${most ? most.act : '–'} · true best leaf: C then x = 0.9`;
    };

    live(root, [cT], () => iters + '|' + Math.floor(elapsed * 10) + '|' + cExp().toFixed(2), update, draw);
  };

  /* ================= 7.3 Model error and rollouts ================= */
  AI.labs['rlmb-error'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Pendulum with a wrong gravity: how model rollouts drift',
      'illustrative · pendulum θ″ = −(g/L) sin θ, L = 1 m, θ₀ = 1 rad, ω₀ = 0 · true g = 9.81 m/s² · the model uses ĝ = g(1 + error)',
      `<canvas class="ang" style="height:190px"></canvas><canvas class="err" style="height:130px;margin-top:10px"></canvas>`, '');
    const cA = AI.canvas(AI.$('canvas.ang', view)), cE = AI.canvas(AI.$('canvas.err', view));
    const gErr = slider(ctl, 'Gravity error of the model', -20, 20, 1, 5, '%', 0);
    const kS = slider(ctl, 'Branched rollout length k', 0.2, 4, 0.2, 1.0, 's', 1);
    const out = readout(ctl);
    const G = 9.81, L = 1, TH0 = 1.0, TT = 4, DTR = 0.02, SUB = 4;
    const sub = (th, w, g, dt) => { const w2 = w - (g / L) * Math.sin(th) * dt; return [th + w2 * dt, w2]; };

    function simulate(gHat, k) {
      const N = Math.round(TT / DTR);
      const tr = [TH0], wt = [0], op = [TH0], br = [TH0];
      let s = [TH0, 0];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < SUB; j++) s = sub(s[0], s[1], G, DTR / SUB);
        tr.push(s[0]); wt.push(s[1]);
      }
      let m = [TH0, 0];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < SUB; j++) m = sub(m[0], m[1], gHat, DTR / SUB);
        op.push(m[0]);
      }
      const kIdx = Math.max(1, Math.round(k / DTR));
      let mb = [TH0, 0];
      for (let i = 0; i < N; i++) {
        if (i % kIdx === 0) mb = [tr[i], wt[i]];
        for (let j = 0; j < SUB; j++) mb = sub(mb[0], mb[1], gHat, DTR / SUB);
        br.push(mb[0]);
      }
      const eO = op.map((x, i) => Math.abs(x - tr[i])), eB = br.map((x, i) => Math.abs(x - tr[i]));
      return { tr, op, br, eO, eB, N, kIdx };
    }

    let S = null, key = '', tc = 0;
    const refresh = () => {
      const e = gErr(), k = kS(), kk = e + '|' + k;
      if (kk === key) return;
      key = kk;
      S = simulate(G * (1 + e / 100), k);
    };
    const T = transport(ctl, () => { tc = 0; T.setPlaying(true); });
    T.setPlaying(true);
    const update = (dt) => {
      refresh();
      if (T.playing) tc += dt * 0.5 * T.speed();
      if (T.pending) { tc += 0.2; T.pending = 0; }
      if (tc >= TT) { tc = TT; T.setPlaying(false); }
    };

    const draw = () => {
      refresh();
      const C = AI.col();
      const i = Math.min(S.N, Math.round(tc / DTR));
      /* angle plot */
      const cg = cA.ctx, va = AI.view(cA, 0, TT, -1.3, 1.3, { l: 46, r: 12, t: 10, b: 24 });
      cg.clearRect(0, 0, cA.w, cA.h);
      AI.axes(cg, va);
      cg.save(); cg.setLineDash([4, 4]); cg.strokeStyle = C.muted;
      for (let m = 1; m * S.kIdx * DTR < TT - 1e-9; m++) {
        const x = va.X(m * S.kIdx * DTR);
        cg.beginPath(); cg.moveTo(x, va.pad.t); cg.lineTo(x, va.pad.t + va.ph); cg.stroke();
      }
      cg.restore();
      const plot = (arr, color, dash) => {
        cg.save(); cg.strokeStyle = color; cg.lineWidth = 2; if (dash) cg.setLineDash(dash);
        cg.beginPath(); arr.forEach((y, j) => { const px = va.X(j * DTR), py = va.Y(y); if (j) cg.lineTo(px, py); else cg.moveTo(px, py); });
        cg.stroke(); cg.restore();
      };
      plot(S.tr, C.green, null);
      plot(S.op, C.orange, [6, 4]);
      plot(S.br, C.blue, null);
      cg.strokeStyle = C.fg; cg.lineWidth = 1; cg.beginPath(); cg.moveTo(va.X(tc), va.pad.t); cg.lineTo(va.X(tc), va.pad.t + va.ph); cg.stroke();
      cg.font = '11px ' + MONO; cg.textAlign = 'left';
      cg.fillStyle = C.green; cg.fillText('true θ(t)', va.pad.l + 8, va.pad.t + 14);
      cg.fillStyle = C.orange; cg.fillText('open-loop model, full horizon', va.pad.l + 8, va.pad.t + 28);
      cg.fillStyle = C.blue; cg.fillText(`branched model, restart every ${AI.fmt(S.kIdx * DTR, 2)} s (dotted lines)`, va.pad.l + 8, va.pad.t + 42);
      cg.fillStyle = C.muted; cg.textAlign = 'center'; cg.fillText('time t (s)', va.X(TT / 2), cA.h - 6);

      /* error plot */
      const ce = cE.ctx;
      let yMax = 0.05;
      for (let j = 0; j <= S.N; j++) yMax = Math.max(yMax, S.eO[j], S.eB[j]);
      yMax *= 1.1;
      const ve = AI.view(cE, 0, TT, 0, yMax, { l: 46, r: 12, t: 10, b: 24 });
      ce.clearRect(0, 0, cE.w, cE.h);
      AI.axes(ce, ve);
      const ploterr = (arr, color, dash) => {
        ce.save(); ce.strokeStyle = color; ce.lineWidth = 2; if (dash) ce.setLineDash(dash);
        ce.beginPath(); arr.forEach((y, j) => { const px = ve.X(j * DTR), py = ve.Y(y); if (j) ce.lineTo(px, py); else ce.moveTo(px, py); });
        ce.stroke(); ce.restore();
      };
      ploterr(S.eO, C.orange, [6, 4]);
      ploterr(S.eB, C.blue, null);
      ce.strokeStyle = C.fg; ce.beginPath(); ce.moveTo(ve.X(tc), ve.pad.t); ce.lineTo(ve.X(tc), ve.pad.t + ve.ph); ce.stroke();
      ce.font = '11px ' + MONO; ce.fillStyle = C.muted; ce.textAlign = 'left';
      ce.fillText('|model − true| (rad): orange open-loop, blue branched', ve.pad.l + 8, ve.pad.t + 14);

      /* readout */
      let maxO = 0, tO = 0, maxB = 0;
      for (let j = 0; j <= S.N; j++) {
        if (S.eO[j] > maxO) { maxO = S.eO[j]; tO = j * DTR; }
        if (S.eB[j] > maxB) maxB = S.eB[j];
      }
      const first = S.eO.findIndex((x) => x > 0.1);
      out.textContent =
        `t = ${AI.fmt(i * DTR, 2)} s   true θ = ${AI.fmt(S.tr[i], 3)} rad   open-loop θ = ${AI.fmt(S.op[i], 3)}   branched θ = ${AI.fmt(S.br[i], 3)}\n` +
        `error now: open-loop ${AI.fmt(S.eO[i], 4)} rad   branched ${AI.fmt(S.eB[i], 4)} rad\n` +
        `over 0–4 s: max open-loop error ${AI.fmt(maxO, 3)} rad (at t = ${AI.fmt(tO, 2)} s); max branched error ${AI.fmt(maxB, 3)} rad\n` +
        `open-loop error first exceeds 0.1 rad at ${first < 0 ? 'never in 4 s' : AI.fmt(first * DTR, 2) + ' s'}\n` +
        `Illustrative: the same integrator runs the true and model pendulums, so the only difference is the gravity constant.`;
    };

    live(root, [cA, cE], () => key + '|' + Math.round(tc / DTR), update, draw);
  };

  /* ================= 7.4 Model-predictive control with a learned model ================= */
  AI.labs['rlmb-mpc'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Random-shooting MPC with a learned gain',
      'illustrative · 2-D point mass, true gain g = 1, cup at the origin · each step: sample sequences, score with the model ĝ, apply the first action, replan',
      `<canvas class="mpc" style="height:330px"></canvas>`, '');
    const cM = AI.canvas(AI.$('canvas.mpc', view));
    const gh = slider(ctl, 'Model gain ĝ (true g = 1)', 0.2, 1.6, 0.05, 0.6, '', 2);
    const hS = slider(ctl, 'Planning horizon H', 2, 16, 1, 8, '', 0);
    const nS = slider(ctl, 'Sampled sequences N', 50, 400, 50, 200, '', 0);
    const out = readout(ctl);
    const TRUE_G = 1, LAM = 0.05, AMAX = 0.5, START = [2, -1.5], NSHOW = 40, MAXSTEPS = 300;
    let rng, p, disp, trail, plan, planStart, shown, steps, cumCost, lastU, lastPred, bestJ;

    function reset() {
      rng = AI.rng(11);
      p = START.slice(); disp = START.slice();
      trail = [START.slice()]; plan = []; planStart = START.slice(); shown = [];
      steps = 0; cumCost = 0; lastU = null; lastPred = null; bestJ = null;
      T.setPlaying(true);
    }
    const T = transport(ctl, reset);
    reset();
    const tk = AI.ticker();

    function planStep() {
      const g = gh(), H = Math.round(hS()), N = Math.round(nS());
      const from = p.slice();
      let bestSeq = null, best = Infinity;
      shown = [];
      for (let j = 0; j < N; j++) {
        let x = from[0], y = from[1], J = 0;
        const us = new Array(2 * H), path = [];
        for (let t = 0; t < H; t++) {
          const ux = (rng() * 2 - 1) * AMAX, uy = (rng() * 2 - 1) * AMAX;
          us[2 * t] = ux; us[2 * t + 1] = uy;
          x += g * ux; y += g * uy;
          J += x * x + y * y + LAM * (ux * ux + uy * uy);
          if (j < NSHOW) path.push([x, y]);
        }
        if (j < NSHOW) shown.push(path);
        if (J < best) { best = J; bestSeq = us; }
      }
      /* the best sequence's predicted path from the current state */
      let x = from[0], y = from[1];
      const bpath = [];
      for (let t = 0; t < H; t++) { x += g * bestSeq[2 * t]; y += g * bestSeq[2 * t + 1]; bpath.push([x, y]); }
      plan = bpath; planStart = from; bestJ = best;
      /* execute only the first action with the real gain */
      const ux = bestSeq[0], uy = bestSeq[1];
      lastU = [ux, uy];
      lastPred = [g * ux, g * uy];
      p = [from[0] + TRUE_G * ux, from[1] + TRUE_G * uy];
      cumCost += p[0] * p[0] + p[1] * p[1] + LAM * (ux * ux + uy * uy);
      trail.push(p.slice()); if (trail.length > 200) trail.shift();
      steps++;
    }

    const update = (dt) => {
      if (T.pending) { T.pending = 0; if (steps < MAXSTEPS) planStep(); }
      if (T.playing) {
        const n = tk(dt, 0.8 * T.speed());
        for (let i = 0; i < n && steps < MAXSTEPS; i++) planStep();
        if (steps >= MAXSTEPS) T.setPlaying(false);
      }
      const a = 1 - Math.exp(-dt * 5);
      disp = [disp[0] + (p[0] - disp[0]) * a, disp[1] + (p[1] - disp[1]) * a];
    };

    const draw = () => {
      const C = AI.col(), g = cM.ctx, v = AI.view(cM, -3, 3, -3, 3, { l: 36, r: 12, t: 12, b: 28 });
      g.clearRect(0, 0, cM.w, cM.h);
      AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      /* sampled candidate paths (model predictions) */
      g.strokeStyle = C.muted; g.globalAlpha = 0.18; g.lineWidth = 1;
      for (const path of shown) {
        g.beginPath(); g.moveTo(v.X(planStart[0]), v.Y(planStart[1]));
        for (const [px, py] of path) g.lineTo(v.X(px), v.Y(py));
        g.stroke();
      }
      g.globalAlpha = 1;
      /* best planned path under the model */
      if (plan.length) {
        g.strokeStyle = C.orange; g.lineWidth = 2.5; g.setLineDash([6, 4]);
        g.beginPath(); g.moveTo(v.X(planStart[0]), v.Y(planStart[1]));
        for (const [px, py] of plan) g.lineTo(v.X(px), v.Y(py));
        g.stroke(); g.setLineDash([]);
      }
      /* real trail */
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      trail.forEach(([px, py], i) => { if (i) g.lineTo(v.X(px), v.Y(py)); else g.moveTo(v.X(px), v.Y(py)); });
      g.stroke();
      g.restore();
      /* cup and hand */
      g.fillStyle = C.green; g.beginPath(); g.arc(v.X(0), v.Y(0), 7, 0, 6.2832); g.fill();
      g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'left';
      g.fillText('cup (origin)', v.X(0) + 10, v.Y(0) + 4);
      g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(disp[0]), v.Y(disp[1]), 6, 0, 6.2832); g.fill();
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('blue: real path · orange dashed: best plan under the model · grey: sampled plans', v.pad.l + 8, v.pad.t + 14);

      const r = Math.hypot(p[0], p[1]);
      const cap = steps === 0
        ? 'Press play. Each step replans from the observed position and applies only the first action.'
        : `Step ${steps}: planned with ĝ = ${AI.fmt(gh(), 2)}; the first command was applied with the true gain g = 1.`;
      out.textContent = cap + '\n' +
        `distance to cup |p| = ${AI.fmt(r, 3)}   position p = (${AI.fmt(p[0], 2)}, ${AI.fmt(p[1], 2)})\n` +
        (lastU
          ? `last command u = (${AI.fmt(lastU[0], 3)}, ${AI.fmt(lastU[1], 3)})   model predicted move ĝu = (${AI.fmt(lastPred[0], 3)}, ${AI.fmt(lastPred[1], 3)})   real move gu = (${AI.fmt(TRUE_G * lastU[0], 3)}, ${AI.fmt(TRUE_G * lastU[1], 3)})\n`
          : 'no command yet\n') +
        `best model cost J over H = ${Math.round(hS())} steps: ${bestJ === null ? '–' : AI.fmt(bestJ, 3)}   accumulated real stage cost: ${AI.fmt(cumCost, 2)}\n` +
        'Changing ĝ or H takes effect at the next replanning.';
    };

    live(root, [cM], () => steps + '|' + Math.round(disp[0] * 200) + ',' + Math.round(disp[1] * 200) + '|' + gh().toFixed(2) + Math.round(hS()) + Math.round(nS()), update, draw);
  };
})();
