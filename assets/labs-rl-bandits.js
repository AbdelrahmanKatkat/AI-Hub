/* Reinforcement Learning · Module 1 (Bandits and exploration) labs. Illustrative toys, not experiments on real systems.
 * rl-arena         <div data-lab="rl-arena"></div>          pulling three noisy arms by hand or at random
 * rl-sample-mean   <div data-lab="rl-sample-mean"></div>    the running average as rewards arrive, with a rough band
 * rl-eps-greedy    <div data-lab="rl-eps-greedy"></div>     epsilon-greedy on a 10-armed testbed, live run and averaged curves
 * rl-ucb           <div data-lab="rl-ucb"></div>           UCB bonuses shrinking as an arm is pulled
 * rl-gradient      <div data-lab="rl-gradient"></div>      gradient bandit: softmax preferences, with or without a baseline
 * rl-drift         <div data-lab="rl-drift"></div>         a drifting arm tracked by sample averages versus a constant step size
 */
(function () {
  const AI = window.AI;
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
  const note = (ctl, html) => { const d = document.createElement('p'); d.style.cssText = 'color:var(--muted);font-size:12px;margin:0'; d.innerHTML = html; ctl.appendChild(d); return d; };

  /* play / pause, single step, reset and a speed multiplier. take() returns how many steps to run this frame. */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0, tick: AI.ticker() };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 2, 0.05, speed, '×', 2);
    pp.onclick = () => { T.playing = !T.playing; pp.textContent = T.playing ? 'Pause' : 'Play'; };
    st.onclick = () => { T.playing = false; pp.textContent = 'Play'; T.pending += 1; };
    rs.onclick = () => onReset();
    T.take = (dt, perSec) => { let n = T.pending; T.pending = 0; if (T.playing) n += T.tick(dt, perSec * T.speed()); return n; };
    return T;
  };

  /* redraw only when the state key, canvas sizes or theme change */
  const live = (root, cvs, key, update, draw) => {
    let last = null;
    AI.loop(root, (dt) => {
      update(dt);
      const k = key() + '|' + cvs.map((c) => c.w + 'x' + c.h).join(',') + '|' + document.documentElement.dataset.theme;
      if (k === last) return;
      last = k; draw();
    });
  };

  /* reproducible standard-normal noise: the same seed always gives the same rewards */
  const noise = (seed, n) => { const r = AI.rng(seed); return Array.from({ length: n }, () => AI.randn(r)); };

  const ARM_COL = () => { const C = AI.col(); return [C.blue, C.orange, C.green, C.purple, C.pink, C.teal]; };

  /* =====================================================================
     1. bandit-problem · the arena
     ===================================================================== */
  AI.labs['rl-arena'] = function (root) {
    const Q = [1.0, 2.0, 1.5];           // true action values q*(a), hidden from the learner
    const OPT = 2.0, SIG = 1.0;           // best value and noise standard deviation
    const NZ = noise(7, 400);
    const { view, ctl } = AI.labShell(root, 'Bandit arena: three slot machines',
      'illustrative · 3 arms · reward = q*(a) + noise, noise ~ N(0, 1²), seed 7',
      '<canvas style="height:280px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const reveal = document.createElement('label');
    reveal.innerHTML = '<span>Show the hidden true means q*</span><input type="checkbox">';
    reveal.style.flexDirection = 'row'; reveal.style.alignItems = 'center'; reveal.style.gap = '8px';
    ctl.appendChild(reveal);
    const chk = reveal.querySelector('input');
    const hist = []; let ver = 0;
    const pick = AI.rng(99);
    const pull = (a) => { hist.push({ a, r: Q[a] + SIG * NZ[hist.length % NZ.length] }); ver++; };
    const row = rowOf(ctl);
    const manual = [0, 1, 2].map((a) => { const b = button(row, 'Pull arm ' + (a + 1)); b.onclick = () => pull(a); return b; });
    const tr = transport(ctl, () => { hist.length = 0; ver++; }, 1);
    const ro = readout(ctl);
    note(ctl, 'The learner sees only the rewards. “Random pull” (Play or Step) chooses an arm uniformly at random.');

    const ARM = ARM_COL();
    const stats = () => {
      const cnt = [0, 0, 0], sum = [0, 0, 0];
      let total = 0, regret = 0;
      hist.forEach((h) => { cnt[h.a]++; sum[h.a] += h.r; total += h.r; regret += OPT - Q[h.a]; });
      return { cnt, sum, total, regret };
    };
    const draw = () => {
      const g = cv.ctx, W = cv.w, H = cv.h, C = AI.col();
      g.clearRect(0, 0, W, H);
      const L = 78, R = 14, T = 12, B = 30, pw = W - L - R, ph = H - T - B;
      const X = (r) => L + ((AI.clamp(r, -2.5, 4.5) + 2.5) / 7) * pw;
      const rowH = ph / 3, S = stats();
      g.font = '11px ' + MONO; g.textAlign = 'center';
      for (let r = -2; r <= 4; r++) {
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(X(r) + 0.5, T); g.lineTo(X(r) + 0.5, T + ph); g.stroke();
        g.fillStyle = C.muted; g.fillText(String(r), X(r), T + ph + 16);
      }
      g.fillText('reward', L + pw / 2, T + ph + 28);
      for (let a = 0; a < 3; a++) {
        const yc = T + (a + 0.5) * rowH;
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(L, yc + 0.5); g.lineTo(L + pw, yc + 0.5); g.stroke();
        g.textAlign = 'left'; g.fillStyle = ARM[a]; g.font = 'bold 12px ' + MONO;
        g.fillText('arm ' + (a + 1), 8, yc - 4);
        g.font = '11px ' + MONO; g.fillStyle = C.muted;
        g.fillText('n = ' + S.cnt[a], 8, yc + 12);
        let k = 0;
        hist.forEach((h, i) => {
          if (h.a !== a) return;
          k++;
          const jit = ((i * 37) % 11 - 5) * 1.1;
          g.globalAlpha = 0.85; g.fillStyle = ARM[a];
          g.beginPath(); g.arc(X(h.r), yc + jit, 4, 0, 2 * Math.PI); g.fill();
        });
        g.globalAlpha = 1;
        if (S.cnt[a]) {
          const m = S.sum[a] / S.cnt[a];
          g.strokeStyle = ARM[a]; g.lineWidth = 3;
          g.beginPath(); g.moveTo(X(m), yc - rowH * 0.36); g.lineTo(X(m), yc + rowH * 0.36); g.stroke(); g.lineWidth = 1;
        }
        if (chk.checked) {
          g.setLineDash([5, 4]); g.strokeStyle = C.fg; g.beginPath(); g.moveTo(X(Q[a]), yc - rowH * 0.42); g.lineTo(X(Q[a]), yc + rowH * 0.42); g.stroke(); g.setLineDash([]);
          g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText('q* = ' + Q[a].toFixed(1), X(Q[a]), yc - rowH * 0.46);
        }
      }
      cap.textContent = 'Each dot is one pull: its horizontal position is the noisy reward it returned. The short bar is that arm’s sample mean. Pulling an arm many times makes its bar settle on q*(a).';
      ro.innerHTML = `pulls <b>${hist.length}</b> · total reward <b>${S.total.toFixed(2)}</b> · average <b>${hist.length ? (S.total / hist.length).toFixed(2) : '–'}</b><br>regret so far <b>${S.regret.toFixed(2)}</b> = Σ (q* best − q*(arm pulled)), which only the designer can compute`;
    };
    live(root, [cv], () => ver + '|' + chk.checked, (dt) => {
      const n = tr.take(dt, 1);
      for (let i = 0; i < n; i++) pull(Math.floor(pick() * 3));
    }, draw);
    chk.onchange = () => ver++;
  };

  /* =====================================================================
     2. sample-average · the running mean
     ===================================================================== */
  AI.labs['rl-sample-mean'] = function (root) {
    const NZ = noise(21, 400);
    const { view, ctl } = AI.labShell(root, 'Running average Q_n as rewards arrive',
      'illustrative · one arm · R_n = μ + σ·ε_n, ε_n ~ N(0, 1), seed 21',
      '<canvas style="height:280px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const mu = slider(ctl, 'True mean μ', -1, 4, 0.1, 2.0, '', 1);
    const sg = slider(ctl, 'Noise σ', 0.2, 2, 0.1, 1.0, '', 1);
    let n = 0, Qv = 0, prev = 0, R = 0, ver = 0;
    const tr = transport(ctl, () => { n = 0; Qv = 0; prev = 0; R = 0; ver++; }, 1);
    const ro = readout(ctl);
    const step = () => {
      if (n >= NZ.length) return;
      R = mu() + sg() * NZ[n];
      prev = Qv; n++;
      Qv = prev + (R - prev) / n;   // incremental sample average: Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n
      ver++;
    };
    const draw = () => {
      const g = cv.ctx, W = cv.w, H = cv.h, C = AI.col();
      g.clearRect(0, 0, W, H);
      const m = mu(), s = sg();
      const L = 52, Rr = 14, T = 12, B = 28, pw = W - L - Rr, ph = H - T - B;
      const xmax = Math.max(20, n + 2);
      const ymin = m - 3.2 * s - 0.5, ymax = m + 3.2 * s + 0.5;
      const v = AI.view(cv, 0, xmax, ymin, ymax, { l: L, r: Rr, t: T, b: B });
      AI.axes(g, v);
      const qs = [0];
      for (let k = 1; k <= n; k++) qs.push(qs[k - 1] + (m + s * NZ[k - 1] - qs[k - 1]) / k);
      // rough band Q_k ± 2σ/√k around the running estimate
      if (n > 0) {
        g.globalAlpha = 0.18; g.fillStyle = C.blue;
        g.beginPath();
        for (let k = 1; k <= n; k++) g.lineTo(v.X(k), v.Y(qs[k] + (2 * s) / Math.sqrt(k)));
        for (let k = n; k >= 1; k--) g.lineTo(v.X(k), v.Y(qs[k] - (2 * s) / Math.sqrt(k)));
        g.closePath(); g.fill(); g.globalAlpha = 1;
      }
      // true mean
      g.setLineDash([6, 5]); g.strokeStyle = C.fg; g.beginPath(); g.moveTo(v.X(0), v.Y(m)); g.lineTo(v.X(xmax), v.Y(m)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.font = '11px ' + MONO; g.textAlign = 'right'; g.fillText('μ = ' + m.toFixed(1), v.X(xmax) - 4, v.Y(m) - 5);
      // samples
      for (let k = 1; k <= n; k++) { g.fillStyle = C.muted; g.globalAlpha = 0.55; g.beginPath(); g.arc(v.X(k), v.Y(m + sg() * NZ[k - 1]), 2.6, 0, 7); g.fill(); }
      g.globalAlpha = 1;
      // running estimate
      if (n > 0) {
        g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath(); g.moveTo(v.X(1), v.Y(qs[1]));
        for (let k = 2; k <= n; k++) g.lineTo(v.X(k), v.Y(qs[k]));
        g.stroke(); g.lineWidth = 1;
        g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(n), v.Y(Qv), 5, 0, 7); g.fill();
      }
      cap.textContent = 'Grey dots are the rewards R_n. The orange line is Q_n, the running average, and the blue band is a rough Q_n ± 2σ/√n. The average wanders at first and then settles near μ.';
      ro.innerHTML = `n = <b>${n}</b> · R_n = <b>${n ? R.toFixed(2) : '–'}</b> · Q_(n−1) = <b>${n ? prev.toFixed(3) : '–'}</b> · Q_n = <b>${n ? Qv.toFixed(3) : '–'}</b><br>update: Q_n = Q_(n−1) + (R_n − Q_(n−1)) / n = ${n ? prev.toFixed(3) + ' + (' + (R - prev).toFixed(3) + ') / ' + n : '–'}<br>error μ − Q_n = <b>${n ? (m - Qv).toFixed(3) : '–'}</b>; standard error σ/√n = <b>${n ? (s / Math.sqrt(n)).toFixed(3) : '–'}</b>`;
    };
    live(root, [cv], () => ver + '|' + mu() + '|' + sg(), (dt) => {
      const k = tr.take(dt, 0.8);
      for (let i = 0; i < k; i++) step();
    }, draw);
  };

  /* =====================================================================
     3. epsilon-greedy · 10-armed testbed (200 runs, common random numbers)
     ===================================================================== */
  const TB = (() => {
    let cache = null;
    return () => {
      if (cache) return cache;
      const K = 10, RUNS = 200, STEPS = 1000;
      const QS = new Float64Array(RUNS * K), NZ = new Float64Array(RUNS * STEPS);
      const UE = new Float64Array(RUNS * STEPS), UA = new Float64Array(RUNS * STEPS);
      for (let r = 0; r < RUNS; r++) {
        const g1 = AI.rng(1000 + r), g2 = AI.rng(5000 + r), g3 = AI.rng(9000 + r), g4 = AI.rng(13000 + r);
        for (let a = 0; a < K; a++) QS[r * K + a] = AI.randn(g1);
        for (let t = 0; t < STEPS; t++) { NZ[r * STEPS + t] = AI.randn(g2); UE[r * STEPS + t] = g3(); UA[r * STEPS + t] = g4(); }
      }
      const best = new Int32Array(RUNS);
      let bestAvg = 0;
      for (let r = 0; r < RUNS; r++) {
        let b = 0;
        for (let a = 1; a < K; a++) if (QS[r * K + a] > QS[r * K + b]) b = a;
        best[r] = b; bestAvg += QS[r * K + b] / RUNS;
      }
      cache = { K, RUNS, STEPS, QS, NZ, UE, UA, best, bestAvg };
      return cache;
    };
  })();

  /* one run of epsilon-greedy with sample-average estimates; ties go to the lowest index */
  const simRun = (T, r, eps, rec) => {
    const { K, STEPS, QS, NZ, UE, UA, best } = T;
    const Q = new Float64Array(K), N = new Int32Array(K);
    const R = new Float64Array(STEPS), O = new Uint8Array(STEPS);
    const snaps = rec ? new Float64Array((STEPS + 1) * K) : null;
    const ch = rec ? new Int8Array(STEPS) : null, ex = rec ? new Uint8Array(STEPS) : null;
    for (let t = 0; t < STEPS; t++) {
      if (rec) snaps.set(Q, t * K);
      const explore = UE[r * STEPS + t] < eps;
      let a;
      if (explore) a = Math.min(K - 1, Math.floor(UA[r * STEPS + t] * K));
      else { a = 0; for (let i = 1; i < K; i++) if (Q[i] > Q[a]) a = i; }
      const reward = QS[r * K + a] + NZ[r * STEPS + t];
      N[a]++; Q[a] += (reward - Q[a]) / N[a];
      R[t] = reward; O[t] = a === best[r] ? 1 : 0;
      if (rec) { ch[t] = a; ex[t] = explore ? 1 : 0; }
    }
    if (rec) snaps.set(Q, STEPS * K);
    return { R, O, snaps, ch, ex };
  };

  const avgCurves = (eps) => {
    const T = TB(), sumR = new Float64Array(T.STEPS), sumO = new Float64Array(T.STEPS);
    for (let r = 0; r < T.RUNS; r++) {
      const o = simRun(T, r, eps, false);
      for (let t = 0; t < T.STEPS; t++) { sumR[t] += o.R[t] / T.RUNS; sumO[t] += o.O[t] / T.RUNS; }
    }
    return { R: sumR, O: sumO };
  };

  AI.labs['rl-eps-greedy'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Epsilon-greedy on a 10-armed testbed',
      'illustrative · 10 arms, q*(a) ~ N(0,1), noise ~ N(0,1) · 200 runs · seeds fixed',
      '<canvas style="height:230px"></canvas><canvas style="height:120px;margin-top:10px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const [cA, cB] = AI.$$('canvas', view).map((c) => AI.canvas(c));
    const cap = AI.$('.cap', view);
    const eps = slider(ctl, 'Exploration rate ε', 0, 0.5, 0.01, 0.1, '', 2);
    let t = 0, ver = 0, lastE = null, live1 = null, cur = null;
    const fixed = {};
    const fixedCurve = (e) => (fixed[e] || (fixed[e] = avgCurves(e)));
    const tr = transport(ctl, () => { t = 0; ver++; }, 1);
    const ro = readout(ctl);
    note(ctl, 'Each run uses the same 10 hidden arms and the same random numbers for every ε, so the curves differ only because of ε.');
    const Tb = TB();
    const update = (dt) => {
      const e = eps();
      if (e !== lastE) { lastE = e; cur = avgCurves(e); live1 = simRun(Tb, 0, e, true); ver++; }
      const n = tr.take(dt, 4);
      if (n) { t = Math.min(Tb.STEPS, t + n); }
    };
    const draw = () => {
      const C = AI.col(), e = lastE, g = cA.ctx, W = cA.w, H = cA.h;
      g.clearRect(0, 0, W, H);
      const v = AI.view(cA, 0, 1000, 0, 1.8, { l: 46, r: 12, t: 12, b: 28 });
      AI.axes(g, v);
      g.fillStyle = C.muted; g.font = '11px ' + ('ui-monospace,Menlo,Consolas,monospace'); g.textAlign = 'left';
      g.fillText('steps t', v.pad.l + v.pw - 40, v.pad.t + v.ph + 24);
      const line = (arr, color, w, dash) => {
        g.save(); g.strokeStyle = color; g.lineWidth = w; if (dash) g.setLineDash(dash);
        g.beginPath();
        for (let i = 0; i < t; i++) { const x = v.X(i + 1), y = v.Y(arr[i]); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
        g.stroke(); g.restore();
      };
      line(fixedCurve(0).R, C.muted, 1.5);
      line(fixedCurve(0.1).R, C.blue, 1.5);
      line(cur.R, C.orange, 2.5);
      g.setLineDash([6, 5]); g.strokeStyle = C.fg; g.beginPath(); g.moveTo(v.X(0), v.Y(Tb.bestAvg)); g.lineTo(v.X(1000), v.Y(Tb.bestAvg)); g.stroke(); g.setLineDash([]);
      g.font = '11px ' + 'ui-monospace,Menlo,Consolas,monospace'; g.fillStyle = C.fg; g.textAlign = 'right';
      g.fillText('oracle ≈ ' + Tb.bestAvg.toFixed(2), v.X(1000) - 4, v.Y(Tb.bestAvg) - 5);
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('grey ε=0 · blue ε=0.1 · orange ε=' + e.toFixed(2), v.pad.l + 6, v.pad.t + 14);

      // live run bars: estimates Q_t(a) against true q*(a) for run 0
      const K = Tb.K, sn = live1.snaps, k0 = Math.min(t, Tb.STEPS);
      const w = cB.w, h = cB.h, gb = cB.ctx; gb.clearRect(0, 0, w, h);
      const L = 46, R = 12, T = 8, B = 22, pw = w - L - R, ph = h - T - B, bw = pw / K;
      const yq = (q) => T + (1 - (AI.clamp(q, -2, 2) + 2) / 4) * ph;
      gb.font = '10px ' + 'ui-monospace,Menlo,Consolas,monospace';
      for (let a = 0; a < K; a++) {
        const q = Tb.QS[a], est = sn[k0 * K + a], x = L + a * bw;
        gb.fillStyle = AI.mix(C.blue, C.blue, 0); gb.globalAlpha = 0.7; gb.fillStyle = C.blue;
        const y0 = yq(0), y1 = yq(est);
        gb.fillRect(x + bw * 0.2, Math.min(y0, y1), bw * 0.6, Math.abs(y1 - y0)); gb.globalAlpha = 1;
        gb.strokeStyle = C.fg; gb.setLineDash([3, 2]); gb.beginPath(); gb.moveTo(x + bw * 0.1, yq(q)); gb.lineTo(x + bw * 0.9, yq(q)); gb.stroke(); gb.setLineDash([]);
        gb.fillStyle = C.muted; gb.textAlign = 'center'; gb.fillText(String(a + 1), x + bw / 2, h - 6);
      }
      gb.fillStyle = C.muted; gb.textAlign = 'right';
      gb.fillText('+2', L - 4, T + 8); gb.fillText('0', L - 4, yq(0) + 4); gb.fillText('−2', L - 4, T + ph);
      gb.textAlign = 'left'; gb.fillText('bars: estimate Q(a) · dashed: true q*(a)', L + 4, T + 10);

      // coloured tally of the last 40 choices: blue = greedy, orange = explore
      const tallyN = Math.min(40, k0);
      let chips = '';
      for (let i = k0 - tallyN; i < k0; i++) {
        if (i < 0) continue;
        chips += `<span style="color:${live1.ex[i] ? C.orange : C.blue}">■</span>`;
      }
      const greedyPct = k0 ? 100 * (1 - (Array.from(live1.ex.slice(0, k0)).reduce((s, x) => s + x, 0) / k0)) : 0;
      cap.innerHTML = `Last ${tallyN} choices of run 0 (blue = greedy, orange = explore): ${chips || '–'}`;
      ro.innerHTML = `step t = <b>${k0}</b> / ${Tb.STEPS} · average reward now <b>${k0 ? cur.R[k0 - 1].toFixed(3) : '–'}</b> · oracle ${Tb.bestAvg.toFixed(2)}<br>% optimal action now <b>${k0 ? (100 * cur.O[k0 - 1]).toFixed(1) : '–'}</b>% (ε = ${e.toFixed(2)} ⇒ about ${(100 * (1 - e + e / Tb.K)).toFixed(1)}% when the estimates are right) · run 0 greedy share ${greedyPct.toFixed(1)}%`;
    };
    live(root, [cA, cB], () => ver + '|' + t + '|' + eps(), update, draw);
  };

  /* =====================================================================
     4. ucb · optimism under uncertainty, four arms
     ===================================================================== */
  AI.labs['rl-ucb'] = function (root) {
    const QT = [0.3, 0.6, 0.9, 0.5], K = 4, SIG = 1.0, NMAX = 400;
    const NZ = new Float64Array(K * NMAX);
    const g = AI.rng(31);
    for (let i = 0; i < K * NMAX; i++) NZ[i] = AI.randn(g);
    const { view, ctl } = AI.labShell(root, 'Upper confidence bounds: optimism in the face of uncertainty',
      'illustrative · 4 arms, q* = (0.3, 0.6, 0.9, 0.5), noise ~ N(0, 1²), seed 31',
      '<canvas style="height:300px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const c = slider(ctl, 'Exploration constant c', 0, 3, 0.05, 2, '', 2);
    const tr = transport(ctl, () => reset(), 1);
    const reveal = document.createElement('label');
    reveal.innerHTML = '<span>Show the hidden true q*</span><input type="checkbox">';
    reveal.style.flexDirection = 'row'; reveal.style.alignItems = 'center'; reveal.style.gap = '8px';
    ctl.appendChild(reveal);
    const chk = reveal.querySelector('input');
    const ro = readout(ctl);
    let t = 0, N, S, Q, last = -1, ver = 0;
    const reset = () => { t = 0; N = [0, 0, 0, 0]; S = [0, 0, 0, 0]; Q = [0, 0, 0, 0]; last = -1; ver++; };
    reset();
    const bonus = (a) => (N[a] === 0 ? Infinity : c() * Math.sqrt(Math.log(t) / N[a]));
    const choose = () => {
      let pick = 0, bv = -Infinity;
      for (let a = 0; a < K; a++) {
        const v = N[a] === 0 ? Infinity : Q[a] + bonus(a);
        if (v > bv) { bv = v; pick = a; }
      }
      return pick;
    };
    const step = () => {
      const a = choose();
      const r = QT[a] + SIG * NZ[a * NMAX + (N[a] % NMAX)];
      N[a]++; Q[a] += (r - Q[a]) / N[a];
      t++; last = a; ver++;
    };
    const draw = () => {
      const C = AI.col(), gg = cv.ctx, W = cv.w, H = cv.h;
      gg.clearRect(0, 0, W, H);
      const L = 58, R = 16, T = 16, B = 40, pw = W - L - R, ph = H - T - B;
      const yv = (y) => T + (1 - (AI.clamp(y, -1, 5) + 1) / 6) * ph;
      gg.font = '11px ' + MONO; gg.fillStyle = C.muted; gg.textAlign = 'right';
      for (let y = -1; y <= 5; y += 1) { gg.strokeStyle = C.border; gg.beginPath(); gg.moveTo(L, yv(y) + 0.5); gg.lineTo(L + pw, yv(y) + 0.5); gg.stroke(); gg.fillText(String(y), L - 6, yv(y) + 4); }
      const COL = [C.blue, C.orange, C.green, C.purple];
      const bw = pw / K;
      for (let a = 0; a < K; a++) {
        const x = L + a * bw + bw * 0.2, wd = bw * 0.6;
        const base = yv(0);
        if (N[a] > 0) {
          const q = Q[a], ub = Q[a] + bonus(a);
          gg.fillStyle = COL[a]; gg.globalAlpha = 0.85;
          gg.fillRect(x, Math.min(base, yv(q)), wd, Math.abs(base - yv(q)));
          gg.globalAlpha = 0.3;
          gg.fillRect(x, yv(ub), wd, Math.abs(yv(q) - yv(ub)));
          gg.globalAlpha = 1;
          gg.strokeStyle = COL[a]; gg.setLineDash([4, 3]); gg.strokeRect(x, yv(ub), wd, Math.abs(yv(q) - yv(ub))); gg.setLineDash([]);
          gg.fillStyle = C.fg; gg.textAlign = 'center';
          gg.fillText('UCB ' + ub.toFixed(2), x + wd / 2, yv(ub) - 6);
        } else {
          gg.fillStyle = C.fg; gg.textAlign = 'center'; gg.fillText('∞ (untried)', x + wd / 2, T + 14);
        }
        if (a === last) { gg.strokeStyle = C.fg; gg.lineWidth = 2.5; gg.strokeRect(x - 4, T - 4, wd + 8, ph + 8); gg.lineWidth = 1; }
        if (chk.checked) { gg.strokeStyle = C.fg; gg.setLineDash([5, 4]); gg.beginPath(); gg.moveTo(x - 4, yv(QT[a])); gg.lineTo(x + wd + 4, yv(QT[a])); gg.stroke(); gg.setLineDash([]); }
        gg.fillStyle = C.muted; gg.textAlign = 'center'; gg.fillText('arm ' + (a + 1) + ' · N=' + N[a], x + wd / 2, H - 22);
      }
      cap.textContent = 'Solid bar: the estimate Q(a). Faded bar on top: the bonus c√(ln t / N(a)), which shrinks each time the arm is pulled. The chosen arm (outlined) has the highest total.';
      const lines = [0, 1, 2, 3].map((a) => N[a] ? `arm ${a + 1}: Q = ${Q[a].toFixed(3)}, bonus = ${bonus(a).toFixed(3)}` : `arm ${a + 1}: untried, index = ∞`);
      ro.innerHTML = `pulls so far t = <b>${t}</b> · c = <b>${c().toFixed(2)}</b> · last choice: <b>${last >= 0 ? 'arm ' + (last + 1) : '–'}</b><br>${lines.join('<br>')}`;
    };
    live(root, [cv], () => ver + '|' + c() + '|' + chk.checked, (dt) => {
      const n = tr.take(dt, 1.5);
      for (let i = 0; i < n; i++) step();
    }, draw);
  };

  /* =====================================================================
     5. gradient-bandit · softmax preferences, with and without a baseline
     ===================================================================== */
  AI.labs['rl-gradient'] = function (root) {
    const QT = [1.0, 2.0, 1.5], T_MAX = 600, BEST = 1;
    const UU = new Float64Array(T_MAX), NZ = new Float64Array(T_MAX);
    const g1 = AI.rng(41), g2 = AI.rng(42);
    for (let i = 0; i < T_MAX; i++) { UU[i] = g1(); NZ[i] = AI.randn(g2); }
    const softmax = (h) => { const m = Math.max(...h), e = h.map((x) => Math.exp(x - m)), s = e[0] + e[1] + e[2]; return e.map((x) => x / s); };
    const run = (alpha, off, useBase) => {
      const H = [0, 0, 0], pis = new Float64Array((T_MAX + 1) * 3);
      let Rb = 0;
      for (let t = 0; t < T_MAX; t++) {
        const pi = softmax(H);
        pis.set(pi, t * 3);
        let a = 0, cum = pi[0];
        while (UU[t] > cum && a < 2) { a++; cum += pi[a]; }
        const R = QT[a] + off + NZ[t];
        const base = useBase ? Rb : 0;
        for (let i = 0; i < 3; i++) H[i] += alpha * (R - base) * ((i === a ? 1 : 0) - pi[i]);
        if (useBase) Rb += (R - Rb) / (t + 1);
      }
      pis.set(softmax(H), T_MAX * 3);
      return pis;
    };
    const { view, ctl } = AI.labShell(root, 'Gradient bandit: preferences move, probabilities follow',
      'illustrative · 3 arms, q* = (1.0, 2.0, 1.5), noise ~ N(0, 1²), seed 41/42 · same random numbers for both learners',
      '<canvas style="height:150px"></canvas><canvas style="height:200px;margin-top:10px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const [cBars, cCurve] = AI.$$('canvas', view).map((c) => AI.canvas(c));
    const cap = AI.$('.cap', view);
    const alpha = slider(ctl, 'Step size α', 0.01, 1, 0.01, 0.1, '', 2);
    const off = slider(ctl, 'Reward offset added to every arm', 0, 10, 0.5, 4, '', 1);
    const tr = transport(ctl, () => { t = 0; ver++; }, 1);
    const ro = readout(ctl);
    note(ctl, 'Blue learner uses the baseline R̄ (average reward so far). Orange learner uses no baseline. Both see identical rewards.');
    let t = 0, ver = 0, key = null, A = null, B = null;
    const recompute = () => { A = run(alpha(), off(), true); B = run(alpha(), off(), false); };
    const update = (dt) => {
      const k = alpha() + '|' + off();
      if (k !== key) { key = k; recompute(); ver++; }
      const n = tr.take(dt, 4);
      if (n) t = Math.min(T_MAX, t + n);
    };
    const draw = () => {
      const C = AI.col();
      const gb = cBars.ctx, w = cBars.w, h = cBars.h;
      gb.clearRect(0, 0, w, h);
      const L = 120, R = 16, rows = [['baseline R̄', A, C.blue], ['no baseline', B, C.orange]];
      rows.forEach((row, ri) => {
        const y = 14 + ri * 64, arr = row[1], o = t * 3;
        gb.font = '12px ' + MONO; gb.fillStyle = row[2]; gb.textAlign = 'left';
        gb.fillText(row[0], 8, y + 14);
        const bw = w - L - R;
        let x = L;
        for (let a = 0; a < 3; a++) {
          const p = arr[o + a], seg = bw * p;
          gb.fillStyle = [C.blue, C.orange, C.green][a]; gb.globalAlpha = 0.25 + 0.6 * p;
          gb.fillRect(x, y, seg, 26); gb.globalAlpha = 1;
          gb.fillStyle = C.fg; gb.font = '11px ' + MONO; gb.textAlign = 'center';
          if (seg > 40) gb.fillText(`arm ${a + 1}: ${(100 * p).toFixed(1)}%`, x + seg / 2, y + 17);
          x += seg;
        }
      });
      const gc = cCurve.ctx, W = cCurve.w, H = cCurve.h;
      gc.clearRect(0, 0, W, H);
      const v = AI.view(cCurve, 0, T_MAX, 0, 1, { l: 46, r: 12, t: 10, b: 28 });
      AI.axes(gc, v);
      const line = (arr, color) => {
        gc.strokeStyle = color; gc.lineWidth = 2.5; gc.beginPath();
        for (let i = 0; i <= t; i++) { const x = v.X(i), y = v.Y(arr[i * 3 + BEST]); if (i === 0) gc.moveTo(x, y); else gc.lineTo(x, y); }
        gc.stroke();
      };
      line(A, C.blue); line(B, C.orange);
      gc.fillStyle = C.muted; gc.font = '11px ' + MONO; gc.textAlign = 'left';
      gc.fillText('P(best arm 2) vs step t', v.pad.l + 6, v.pad.t + 14);
      cap.textContent = 'Top: the probability of each arm at step t. Bottom: the probability of the best arm (arm 2) over time. Larger offsets make the no-baseline learner noisier.';
      ro.innerHTML = `step t = <b>${t}</b> / ${T_MAX} · α = <b>${alpha().toFixed(2)}</b> · offset = <b>${off().toFixed(1)}</b><br>P(arm 2) baseline = <b>${(A[t * 3 + BEST]).toFixed(3)}</b> · no baseline = <b>${(B[t * 3 + BEST]).toFixed(3)}</b>`;
    };
    live(root, [cBars, cCurve], () => ver + '|' + t, update, draw);
  };

  /* =====================================================================
     6. nonstationary-bandits · sample averages versus a constant step size
     ===================================================================== */
  AI.labs['rl-drift'] = function (root) {
    const T_MAX = 300, Z = new Float64Array(T_MAX), EPS = new Float64Array(T_MAX);
    const g1 = AI.rng(77), g2 = AI.rng(78);
    for (let i = 0; i < T_MAX; i++) { Z[i] = AI.randn(g1); EPS[i] = AI.randn(g2); }
    const { view, ctl } = AI.labShell(root, 'A drifting arm: averages lag, constant steps track',
      'illustrative · one arm whose true mean drifts as a random walk · noise σ = 1 · seed 77/78',
      '<canvas style="height:280px"></canvas><p class="cap" style="font-size:13px;color:var(--muted);margin:8px 0 0"></p>', '');
    const cv = AI.canvas(AI.$('canvas', view)), cap = AI.$('.cap', view);
    const drift = slider(ctl, 'Drift per step d', 0, 0.2, 0.01, 0.05, '', 2);
    const alpha = slider(ctl, 'Constant step size α', 0.01, 1, 0.01, 0.1, '', 2);
    const tr = transport(ctl, () => { t = 0; ver++; }, 1);
    const ro = readout(ctl);
    let t = 0, ver = 0, key = null, qTrue, rew, avgQ, cstQ;
    const recompute = () => {
      const d = drift(), a = alpha();
      qTrue = new Float64Array(T_MAX); rew = new Float64Array(T_MAX);
      avgQ = new Float64Array(T_MAX); cstQ = new Float64Array(T_MAX);
      let q = 0;
      for (let i = 0; i < T_MAX; i++) {
        qTrue[i] = q; rew[i] = q + EPS[i];
        const n = i + 1, prevA = i ? avgQ[i - 1] : 0, prevC = i ? cstQ[i - 1] : 0;
        avgQ[i] = prevA + (rew[i] - prevA) / n;      // sample average, step 1/n
        cstQ[i] = prevC + a * (rew[i] - prevC);      // constant step size
        q += d * Z[i];                                // the true mean drifts after each pull
      }
    };
    const update = (dt) => {
      const k = drift() + '|' + alpha();
      if (k !== key) { key = k; recompute(); ver++; }
      const n = tr.take(dt, 5);
      if (n) t = Math.min(T_MAX, t + n);
    };
    const draw = () => {
      const C = AI.col(), gg = cv.ctx, W = cv.w, H = cv.h;
      gg.clearRect(0, 0, W, H);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < T_MAX; i++) { lo = Math.min(lo, qTrue[i], rew[i]); hi = Math.max(hi, qTrue[i], rew[i]); }
      const pad = 0.5, v = AI.view(cv, 0, T_MAX, lo - pad, hi + pad, { l: 52, r: 14, t: 12, b: 28 });
      AI.axes(gg, v);
      for (let i = 0; i < t; i++) { gg.fillStyle = C.muted; gg.globalAlpha = 0.5; gg.beginPath(); gg.arc(v.X(i + 0.5), v.Y(rew[i]), 2, 0, 7); gg.fill(); }
      gg.globalAlpha = 1;
      const line = (arr, color, w) => {
        gg.strokeStyle = color; gg.lineWidth = w; gg.beginPath();
        for (let i = 0; i < t; i++) { const x = v.X(i + 1), y = v.Y(arr[i]); if (i === 0) gg.moveTo(x, y); else gg.lineTo(x, y); }
        gg.stroke(); gg.lineWidth = 1;
      };
      line(qTrue, C.fg, 2.5);
      line(avgQ, C.green, 2);
      line(cstQ, C.blue, 2.5);
      gg.font = '11px ' + MONO; gg.fillStyle = C.muted; gg.textAlign = 'left';
      gg.fillText('black: true mean q*(t) · green: sample average · blue: constant α · grey dots: rewards', v.pad.l + 6, v.pad.t + 14);
      cap.textContent = 'The sample average weights every past reward equally, so it carries the old level of the drift. The constant step size forgets gradually and follows the black line closer.';
      const half = Math.log(0.5) / Math.log(1 - alpha());
      const i = Math.max(0, t - 1);
      ro.innerHTML = `step t = <b>${t}</b> / ${T_MAX} · true q* = <b>${t ? qTrue[i].toFixed(2) : '–'}</b><br>sample average Q = <b>${t ? avgQ[i].toFixed(2) : '–'}</b> (error ${t ? (avgQ[i] - qTrue[i]).toFixed(2) : '–'})<br>constant α Q = <b>${t ? cstQ[i].toFixed(2) : '–'}</b> (error ${t ? (cstQ[i] - qTrue[i]).toFixed(2) : '–'})<br>α = ${alpha().toFixed(2)}: a reward's weight halves every <b>${half.toFixed(1)}</b> pulls; effective memory ≈ 1/α = <b>${(1 / alpha()).toFixed(1)}</b> pulls`;
    };
    live(root, [cv], () => ver + '|' + t, update, draw);
  };
})();
