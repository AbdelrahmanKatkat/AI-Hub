/* Reinforcement Learning · Module 9 (Applied RL, alignment & evaluation) labs. All are illustrative toys, not measured results.
 * rlhfKL           <div data-lab="rlhfKL"></div>             KL-regularised tuning of a policy over four canned responses
 * rewardHack       <div data-lab="rewardHack"></div>         a learner climbing a proxy reward that over-rewards a gameable behaviour
 * fictitiousPlay   <div data-lab="fictitiousPlay"></div>     self-play by fictitious play in rock, paper, scissors
 * seedCI           <div data-lab="seedCI"></div>             learning curves over seeds with a 95% t-interval band
 * canaryRollout    <div data-lab="canaryRollout"></div>      staged traffic rollout with a noisy guardrail
 * oversightGate    <div data-lab="oversightGate"></div>      risk-scored actions routed to auto, human review or block
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';

  /* ---------- shared helpers ---------- */
  const slider = (ctl, name, min, max, step, val, unit = '', dec = 2, onchg = null) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = AI.fmt(+inp.value, dec) + (unit ? ' ' + unit : ''));
    inp.oninput = () => { upd(); if (onchg) onchg(+inp.value); };
    upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  const rowOf = (ctl) => { const d = document.createElement('div'); d.className = 'row'; ctl.appendChild(d); return d; };
  const button = (row, text) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = text; row.appendChild(b); return b; };

  /* transport: play/pause, step (one pending unit) and reset, plus a speed multiplier */
  const transport = (ctl, onReset, speed = 1) => {
    const r = rowOf(ctl);
    const T = { playing: true, pending: 0 };
    const pp = button(r, 'Pause'), st = button(r, 'Step'), rs = button(r, 'Reset');
    T.speed = slider(ctl, 'Speed', 0.25, 3, 0.05, speed, '×', 2);
    T.setPlaying = (p) => { T.playing = p; pp.textContent = p ? 'Pause' : 'Play'; };
    pp.onclick = () => T.setPlaying(!T.playing);
    st.onclick = () => { T.setPlaying(false); T.pending += 1; };
    rs.onclick = () => onReset();
    T.take = () => { const p = T.pending; T.pending = 0; return p; };
    return T;
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

  const CANVAS = (h) => `<canvas style="width:100%;height:${h}px;display:block"></canvas>`;
  const READ = '<div class="rl-read" style="font:13px ' + MONO + ';margin-top:8px;line-height:1.6"></div>';
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /* ======================================================================
   * 9.1  rlhf-kl: KL-regularised tuning of a policy
   * ==================================================================== */
  AI.labs.rlhfKL = function (root) {
    const names = ['A helpful', 'B wordy', 'C unsafe', 'D refusal'];
    const ref = [0.25, 0.25, 0.25, 0.25];
    const defaults = [1.0, 0.4, -0.5, 0.0];
    const { view, ctl } = AI.labShell(root, 'Preference tuning with a KL leash',
      'Illustrative: four canned responses, not a language model. The reference policy is uniform. The rewards are set by you.',
      CANVAS(300) + READ, '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    const cap = document.createElement('p');
    cap.style.cssText = 'margin:0 0 6px;color:var(--muted);font-size:13px';
    view.insertBefore(cap, cv);

    let p = ref.slice();
    const reset = () => { p = ref.slice(); };
    const beta = slider(ctl, 'KL strength β', 0.05, 2, 0.01, 0.5, '', 2);
    const rw = defaults.map((d, i) => slider(ctl, 'reward r(' + names[i].split(' ')[0] + ')', -2, 2, 0.05, d, '', 2));
    const T = transport(ctl, reset, 1);

    // Closed form optimum of E[r] - beta * KL(pi || ref):  pi*(y) ∝ ref(y) exp(r(y)/beta)
    const target = () => {
      const r = rw.map((g) => g()), b = beta();
      const mx = Math.max(...r);
      const w = ref.map((q, i) => q * Math.exp((r[i] - mx) / b));
      const Z = w.reduce((a, x) => a + x, 0);
      return { t: w.map((x) => x / Z), r, b };
    };

    const update = (dt) => {
      let k = 0;
      if (T.playing) k = 1 - Math.exp(-dt * T.speed() * 2.5);
      if (T.take()) k = 1 - Math.exp(-0.5 * 2.5);
      if (k === 0) return;
      const tg = target().t;
      p = p.map((x, i) => x + (tg[i] - x) * k);
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col();
      g.clearRect(0, 0, W, H);
      const { r, b } = target();
      const L = 40, R = W - 10, Tp = 14, B = H - 50;
      g.strokeStyle = col.border; g.lineWidth = 1;
      g.beginPath(); g.moveTo(L, Tp); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'right';
      [0, 0.5, 1].forEach((v) => { const y = B - (B - Tp) * v; g.fillText(v.toFixed(1), L - 5, y + 4); });
      const gw = (R - L) / 4;
      g.textAlign = 'center';
      for (let i = 0; i < 4; i++) {
        const x0 = L + gw * i + gw * 0.18, bw = gw * 0.3;
        const yr = B - (B - Tp) * ref[i];
        g.globalAlpha = 0.45; g.fillStyle = col.muted; g.fillRect(x0, yr, bw, B - yr); g.globalAlpha = 1;
        const yp = B - (B - Tp) * p[i];
        g.fillStyle = col.blue; g.fillRect(x0 + bw + 3, yp, bw, B - yp);
        g.fillStyle = col.fg; g.font = '11px ' + MONO;
        g.fillText(p[i].toFixed(3), x0 + bw + 3 + bw / 2, yp - 5);
        g.fillStyle = col.muted; g.font = '12px ' + MONO;
        g.fillText(names[i], L + gw * i + gw / 2, B + 16);
        g.fillText('r = ' + r[i].toFixed(2), L + gw * i + gw / 2, B + 32);
      }
      g.textAlign = 'left'; g.fillStyle = col.muted; g.font = '11px ' + MONO;
      g.fillText('grey = reference π_ref', L + 4, Tp + 10);
      g.fillStyle = col.blue; g.fillText('blue = current π', L + 150, Tp + 10);

      const E = p.reduce((s, q, i) => s + q * r[i], 0);
      const KL = p.reduce((s, q, i) => (q > 0 ? s + q * Math.log(q / ref[i]) : s), 0);
      const J = E - b * KL;
      const mxI = r.indexOf(Math.max(...r));
      const gap = r[0] - r[2];
      const pref = sigmoid(gap), loss = -Math.log(pref);
      cap.textContent = b < 0.2
        ? 'Small β: the policy piles onto the highest-reward response and moves far from the reference (large KL).'
        : b > 1
          ? 'Large β: the KL leash dominates, so π stays close to the reference and the reward barely moves it.'
          : 'Medium β: a compromise. The policy shifts toward reward but keeps some mass on every response.';
      rd.innerHTML =
        `E<sub>π</sub>[r] = ${AI.fmt(E, 4)} &nbsp; KL(π‖π_ref) = ${AI.fmt(KL, 4)} &nbsp; objective J = E − βKL = ${AI.fmt(J, 4)}<br>` +
        `reward-only optimum puts all mass on "${names[mxI]}" (E = ${AI.fmt(Math.max(...r), 2)}, KL = ln 4 = ${AI.fmt(Math.log(4), 3)}).<br>` +
        `Pairwise reward model, A vs C: P(A preferred) = σ(r<sub>A</sub> − r<sub>C</sub>) = σ(${AI.fmt(gap, 2)}) = ${AI.fmt(pref, 4)}, loss −ln σ = ${AI.fmt(loss, 4)}`;
    };

    live(root, [c], () => p.map((x) => x.toFixed(4)).join(',') + '|' + target().r.join(',') + '|' + target().b, update, draw);
  };

  /* ======================================================================
   * 9.2  reward-hack: Goodhart in one dimension
   * T(x) = 1 - (x - 0.3)^2 ;  P(x) = T(x) + a·x  (a = how much the proxy over-rewards the gameable share x)
   * ==================================================================== */
  AI.labs.rewardHack = function (root) {
    const { view, ctl } = AI.labShell(root, 'Goodhart in one dimension',
      'Illustrative toy: x is the share of time spent on a behaviour the proxy over-rewards. True value T(x) = 1 − (x − 0.3)². Proxy P(x) = T(x) + a·x.',
      CANVAS(340) + READ, '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    const Tf = (x) => 1 - (x - 0.3) ** 2;
    let x, hist, n;
    const reset = () => { x = 0.1; hist = []; n = 0; };
    const A = slider(ctl, 'Gaming weight a', 0, 1.5, 0.05, 1.0, '', 2, () => reset());
    const T = transport(ctl, reset, 1);
    reset();
    const tick = AI.ticker();

    const update = (dt) => {
      let steps = T.take();
      if (T.playing) steps += tick(dt, 2 * T.speed());
      const a = A();
      for (let s = 0; s < steps && n < 200; s++) {
        const g = -2 * (x - 0.3) + a;      // gradient of the proxy P(x) with respect to x
        x = AI.clamp(x + 0.1 * g, 0, 1);   // learning rate 0.1 per step
        n++;
        hist.push([Tf(x) + a * x, Tf(x)]);
      }
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col(), a = A();
      g.clearRect(0, 0, W, H);
      const yLo = 0.3, yHi = 2.2;
      const L = 44, R = W - 12;
      const topT = 12, topB = H * 0.5 - 16;
      const botT = H * 0.5 + 12, botB = H - 22;
      const Y = (v, t, b) => b - (b - t) * (v - yLo) / (yHi - yLo);
      // top panel: curves over x
      g.strokeStyle = col.border; g.beginPath(); g.moveTo(L, topT); g.lineTo(L, topB); g.lineTo(R, topB); g.stroke();
      const X = (v) => L + (R - L) * v;
      const curve = (f, color, dash) => {
        g.save(); g.strokeStyle = color; g.lineWidth = 2.2; if (dash) g.setLineDash(dash);
        g.beginPath();
        for (let i = 0; i <= 200; i++) { const xv = i / 200, yv = f(xv); i ? g.lineTo(X(xv), Y(yv, topT, topB)) : g.moveTo(X(xv), Y(yv, topT, topB)); }
        g.stroke(); g.restore();
      };
      curve((xv) => Tf(xv) + a * xv, col.orange);
      curve(Tf, col.green);
      const xs = AI.clamp(0.3 + a / 2, 0, 1);
      g.setLineDash([4, 4]); g.strokeStyle = col.orange; g.beginPath(); g.moveTo(X(xs), topT); g.lineTo(X(xs), topB); g.stroke(); g.setLineDash([]);
      g.fillStyle = col.orange; g.beginPath(); g.arc(X(x), Y(Tf(x) + a * x, topT, topB), 5, 0, 7); g.fill();
      g.fillStyle = col.green; g.beginPath(); g.arc(X(x), Y(Tf(x), topT, topB), 5, 0, 7); g.fill();
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'center';
      [0, 0.5, 1].forEach((v) => g.fillText(v.toFixed(1), X(v), topB + 13));
      g.textAlign = 'left'; g.fillStyle = col.orange; g.fillText('proxy P(x)', L + 6, topT + 12);
      g.fillStyle = col.green; g.fillText('true T(x)', L + 6, topT + 26);
      g.fillStyle = col.muted; g.fillText('x = share on gameable behaviour', L + 110, topB + 13);
      // bottom panel: time series
      g.strokeStyle = col.border; g.beginPath(); g.moveTo(L, botT); g.lineTo(L, botB); g.lineTo(R, botB); g.stroke();
      const S = (s) => L + (R - L) * s / 200;
      const series = (idx, color) => {
        g.save(); g.strokeStyle = color; g.lineWidth = 2;
        g.beginPath();
        hist.forEach((h, i) => { const px = S(i + 1), py = Y(h[idx], botT, botB); i ? g.lineTo(px, py) : g.moveTo(px, py); });
        g.stroke(); g.restore();
      };
      if (hist.length) { series(0, col.orange); series(1, col.green); }
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'center';
      [0, 100, 200].forEach((s) => g.fillText(s + ' steps', S(s), botB + 13));
      g.textAlign = 'left';
      g.fillText('learner over time (learning rate 0.1 per step, stops at 200)', L + 6, botT + 12);

      const last = hist.length ? hist[hist.length - 1] : [Tf(x) + a * x, Tf(x)];
      const optT = Tf(xs);
      rd.innerHTML =
        `step ${n} · x = ${AI.fmt(x, 3)} · proxy P = ${AI.fmt(last[0], 3)} · true T = ${AI.fmt(last[1], 3)}<br>` +
        `proxy optimum x* = 0.3 + a/2 = ${AI.fmt(xs, 3)} (clipped to [0,1]) gives true value T(x*) = ${AI.fmt(optT, 3)}.` +
        (n >= 200 ? '<br>Stopped at 200 steps; press Reset to run again.' : '');
    };

    live(root, [c], () => x.toFixed(5) + '|' + n + '|' + A(), update, draw);
  };

  /* ======================================================================
   * 9.3  fictitious-play: self-play in rock, paper, scissors
   * ==================================================================== */
  AI.labs.fictitiousPlay = function (root) {
    const MOVES = ['Rock', 'Paper', 'Scissors'];
    const { view, ctl } = AI.labShell(root, 'Self-play by fictitious play',
      'Illustrative: two learners each best-respond to the other\'s past move frequencies. Zero-sum: win +1, lose −1, tie 0. Fixed seed.',
      CANVAS(330) + READ, '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    // u(my, opp): +1 if my move beats opp, -1 if it loses, 0 on a tie. Moves are 0=Rock, 1=Paper, 2=Scissors.
    const U = (m, o) => (m === o ? 0 : ((m - o + 3) % 3 === 1 ? 1 : -1));

    let rng, cA, cB, rounds, trail, last;
    const reset = () => {
      rng = AI.rng(7);
      cA = [1, 1, 1];                      // A's move counts (uniform start)
      cB = [1 + BIAS(), 1, 1];             // B's history is pre-loaded with Rock pseudo-counts
      rounds = 0; trail = []; last = null;
    };
    const BIAS = slider(ctl, 'B history starts with Rock pseudo-counts', 0, 20, 1, 0, '', 0, () => reset());
    const T = transport(ctl, reset, 1);
    reset();
    const tick = AI.ticker();

    const bestResp = (q) => {
      const u = [0, 1, 2].map((m) => q[0] * U(m, 0) + q[1] * U(m, 1) + q[2] * U(m, 2));
      const mx = Math.max(...u);
      const ties = [0, 1, 2].filter((i) => Math.abs(u[i] - mx) < 1e-12);
      return ties[Math.floor(rng() * ties.length)];
    };
    const roundOnce = () => {
      const sB = cB[0] + cB[1] + cB[2], sA = cA[0] + cA[1] + cA[2];
      const a = bestResp(cB.map((v) => v / sB));   // A best-responds to B's history
      const b = bestResp(cA.map((v) => v / sA));   // B best-responds to A's history
      cA[a]++; cB[b]++; rounds++;
      last = { a, b, res: U(a, b) };
      trail.push(cA.map((v) => v / (sA + 1)));
      if (trail.length > 600) trail.shift();
    };

    const update = (dt) => {
      let steps = T.take();
      if (T.playing) steps += tick(dt, 2 * T.speed());
      for (let s = 0; s < steps; s++) roundOnce();
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col();
      g.clearRect(0, 0, W, H);
      const s = Math.min(W * 0.62, (H - 50) * 2 / Math.sqrt(3));
      const hgt = s * Math.sqrt(3) / 2;
      const cx = W / 2, base = H / 2 + hgt / 2 - 8;
      const V = [[cx - s / 2, base], [cx + s / 2, base], [cx, base - hgt]];
      const P = (q) => [0, 1, 2].reduce((acc, i) => [acc[0] + q[i] * V[i][0], acc[1] + q[i] * V[i][1]], [0, 0]);
      g.strokeStyle = col.border; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(V[0][0], V[0][1]); g.lineTo(V[1][0], V[1][1]); g.lineTo(V[2][0], V[2][1]); g.closePath(); g.stroke();
      g.fillStyle = col.fg; g.font = '12px ' + MONO;
      g.textAlign = 'right'; g.fillText('Rock', V[0][0] - 6, V[0][1] + 4);
      g.textAlign = 'left'; g.fillText('Paper', V[1][0] + 6, V[1][1] + 4);
      g.textAlign = 'center'; g.fillText('Scissors', V[2][0], V[2][1] - 8);
      // trail of A's empirical frequencies
      g.save(); g.lineWidth = 1.6;
      trail.forEach((q, i) => {
        if (i === 0) return;
        const pa = P(trail[i - 1]), pb = P(q);
        g.globalAlpha = 0.15 + 0.85 * (i / trail.length);
        g.strokeStyle = col.blue; g.beginPath(); g.moveTo(pa[0], pa[1]); g.lineTo(pb[0], pb[1]); g.stroke();
      });
      g.restore();
      const centre = P([1 / 3, 1 / 3, 1 / 3]);
      g.fillStyle = col.green; g.beginPath(); g.arc(centre[0], centre[1], 4, 0, 7); g.fill();
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'left';
      g.fillText('green = equilibrium (1/3 each)', 10, 16);
      if (trail.length) {
        const now = P(trail[trail.length - 1]);
        g.fillStyle = col.blue; g.beginPath(); g.arc(now[0], now[1], 6, 0, 7); g.fill();
      }
      const frq = trail.length ? trail[trail.length - 1] : [1 / 3, 1 / 3, 1 / 3];
      const tv = 0.5 * frq.reduce((acc, v) => acc + Math.abs(v - 1 / 3), 0);
      const lastTxt = last
        ? `A played ${MOVES[last.a]}, B played ${MOVES[last.b]} → ${last.res > 0 ? 'A wins' : last.res < 0 ? 'B wins' : 'tie'}`
        : 'no rounds yet';
      rd.innerHTML =
        `round ${rounds} · ${lastTxt}<br>` +
        `A's move frequencies: Rock ${AI.fmt(100 * frq[0], 1)}% · Paper ${AI.fmt(100 * frq[1], 1)}% · Scissors ${AI.fmt(100 * frq[2], 1)}%<br>` +
        `distance from equilibrium (total variation) = ${AI.fmt(tv, 4)}`;
    };

    live(root, [c], () => rounds + '|' + BIAS(), update, draw);
  };

  /* ======================================================================
   * 9.4  seed-ci: learning curves over seeds with a 95% t-interval
   * ==================================================================== */
  AI.labs.seedCI = function (root) {
    const { view, ctl } = AI.labShell(root, 'Learning curves over seeds',
      'Illustrative: a made-up learning-curve model with seed-to-seed spread, not measured results. Band = 95% t-interval for the mean across the seeds shown.',
      CANVAS(320) + READ, '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    const TCRIT = [null, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262];
    const RMAX = slider(ctl, 'Asymptote R_max', 50, 400, 5, 200, '', 0, () => reset());
    const TAU = slider(ctl, 'Time scale τ (k steps)', 5, 80, 1, 40, '', 0, () => reset());
    const NOISE = slider(ctl, 'Seed spread (relative sd)', 0, 0.4, 0.01, 0.15, '', 2, () => reset());
    const NMAX = slider(ctl, 'Seeds to run', 1, 10, 1, 10, '', 0);
    // ten fixed seeds, drawn once from AI.rng(11)
    const rr = AI.rng(11);
    const SEEDS = Array.from({ length: 10 }, () => ({ z: AI.randn(rr), y: AI.randn(rr) }));
    let revealed = 1;
    const reset = () => { revealed = 1; };
    const T = transport(ctl, reset, 1);

    const curve = (sd, t) => {
      const amp = Math.max(0, RMAX() * (1 + NOISE() * sd.z));
      const tau = Math.max(1, TAU() * (1 + 0.25 * sd.y));
      return amp * (1 - Math.exp(-t / tau));
    };

    const update = (dt) => {
      if (T.playing) revealed = Math.min(NMAX(), revealed + dt * T.speed() * 0.8);
      if (T.take()) revealed = Math.min(NMAX(), Math.floor(revealed) + 1);
      if (revealed > NMAX()) revealed = NMAX();
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col();
      g.clearRect(0, 0, W, H);
      const n = Math.max(1, Math.min(NMAX(), Math.floor(revealed)));
      const L = 46, R = W - 12, Tp = 12, B = H - 26;
      const yMax = RMAX() * 1.3;
      const X = (t) => L + (R - L) * t / 100, Y = (v) => B - (B - Tp) * v / yMax;
      g.strokeStyle = col.border; g.beginPath(); g.moveTo(L, Tp); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'right';
      [0, 0.5, 1].forEach((f) => g.fillText(AI.fmt(yMax * f, 0), L - 5, Y(yMax * f) + 4));
      g.textAlign = 'center';
      [0, 50, 100].forEach((t) => g.fillText(t + 'k steps', X(t), B + 15));
      const ts = Array.from({ length: 61 }, (_, i) => (i * 100) / 60);
      // individual seeds (faint)
      for (let k = 0; k < n; k++) {
        g.save(); g.strokeStyle = col.muted; g.globalAlpha = 0.35; g.lineWidth = 1;
        g.beginPath(); ts.forEach((t, i) => { const v = curve(SEEDS[k], t); i ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); });
        g.stroke(); g.restore();
      }
      // mean and band
      const mean = [], half = [];
      ts.forEach((t) => {
        const vals = SEEDS.slice(0, n).map((sd) => curve(sd, t));
        const m = vals.reduce((a, b) => a + b, 0) / n;
        const s2 = n > 1 ? vals.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1) : 0;
        mean.push(m); half.push(n > 1 ? TCRIT[n - 1] * Math.sqrt(s2) / Math.sqrt(n) : 0);
      });
      if (n > 1) {
        g.save(); g.globalAlpha = 0.22; g.fillStyle = col.blue; g.beginPath();
        ts.forEach((t, i) => (i ? g.lineTo(X(t), Y(mean[i] + half[i])) : g.moveTo(X(t), Y(mean[i] + half[i]))));
        for (let i = ts.length - 1; i >= 0; i--) g.lineTo(X(ts[i]), Y(mean[i] - half[i]));
        g.closePath(); g.fill(); g.restore();
      }
      g.save(); g.strokeStyle = col.blue; g.lineWidth = 2.8; g.beginPath();
      ts.forEach((t, i) => (i ? g.lineTo(X(t), Y(mean[i])) : g.moveTo(X(t), Y(mean[i]))));
      g.stroke(); g.restore();
      const finals = SEEDS.slice(0, n).map((sd) => curve(sd, 100));
      const fm = finals.reduce((a, b) => a + b, 0) / n;
      const fs = n > 1 ? Math.sqrt(finals.reduce((a, b) => a + (b - fm) ** 2, 0) / (n - 1)) : 0;
      const hw = n > 1 ? TCRIT[n - 1] * fs / Math.sqrt(n) : 0;
      const best = Math.max(...finals);
      rd.innerHTML =
        `seeds run: ${n} of ${NMAX()} · grey = single seeds · blue = mean with 95% t-band<br>` +
        `final return at 100k steps: mean ${AI.fmt(fm, 1)}${n > 1 ? ' ± ' + AI.fmt(hw, 1) + ' (95% CI half-width, df = ' + (n - 1) + ')' : ' (one seed: no interval)'}<br>` +
        `best single seed: ${AI.fmt(best, 1)}. Reporting the best seed overstates the method by ${AI.fmt(100 * (best / fm - 1), 1)}%.`;
    };

    live(root, [c], () => Math.floor(revealed) + '|' + Math.round(revealed * 100) + '|' + RMAX() + '|' + TAU() + '|' + NOISE() + '|' + NMAX(), update, draw);
  };

  /* ======================================================================
   * 9.5  canary-rollout: staged traffic with a noisy guardrail
   * ==================================================================== */
  AI.labs.canaryRollout = function (root) {
    const STAGES = [1, 5, 25, 100];
    const SD = 0.3;           // per-episode relative sd of return (illustrative)
    const STAGE_SECONDS = 4;  // simulated seconds per stage at speed 1
    const { view, ctl } = AI.labShell(root, 'Staged rollout with a guardrail',
      'Illustrative: candidate policy receives 1%, 5%, 25% then 100% of traffic. Each stage measures the candidate against the baseline and rolls back if the measured drop exceeds the guard.',
      CANVAS(260) + '<div class="rl-read" style="font:13px ' + MONO + ';margin-top:8px;line-height:1.6"></div>', '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    const TRUE = slider(ctl, 'True change in candidate return (%)', -20, 10, 1, -8, '%', 0, () => reset());
    const GUARD = slider(ctl, 'Guard: roll back if measured drop exceeds (%)', 1, 15, 0.5, 5, '%', 1, () => reset());
    let rng, stage, clock, state, meas, lines;
    const se = (pct) => SD / Math.sqrt(Math.max(1, 2000 * pct / 100));   // standard error of relative change
    const startStage = () => {
      const pct = STAGES[stage];
      meas = TRUE() / 100 + se(pct) * AI.randn(rng);
      clock = 0;
    };
    const reset = () => {
      rng = AI.rng(5); stage = 0; state = 'running'; lines = []; startStage();
    };
    const T = transport(ctl, reset, 1);
    reset();

    const finishStage = () => {
      const pct = STAGES[stage], N = Math.round(2000 * pct / 100);
      const caught = meas < -GUARD() / 100;
      lines.push(`${pct}% · N = ${N} · measured change ${AI.fmt(100 * meas, 2)}% ${caught ? '→ ROLL BACK' : '→ pass'}`);
      if (caught) { state = 'rolled'; return; }
      stage++;
      if (stage >= STAGES.length) { state = 'done'; return; }
      startStage();
    };

    const update = (dt) => {
      if (state !== 'running') { T.take(); return; }
      if (T.take()) { finishStage(); return; }
      if (T.playing) {
        clock += dt * T.speed();
        if (clock >= STAGE_SECONDS) finishStage();
      }
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col();
      g.clearRect(0, 0, W, H);
      const L = 44, R = W - 12, Tp = 16, B = H - 42;
      g.strokeStyle = col.border; g.beginPath(); g.moveTo(L, Tp); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      const gw = (R - L) / STAGES.length;
      g.font = '12px ' + MONO; g.textAlign = 'center';
      STAGES.forEach((pct, i) => {
        const x0 = L + gw * i + gw * 0.2, bw = gw * 0.6;
        const full = B - Tp;
        const hgt = full * (pct / 100) ** 0.5;   // square-root height so 1% is still visible
        g.globalAlpha = 0.25; g.fillStyle = col.muted; g.fillRect(x0, B - hgt, bw, hgt); g.globalAlpha = 1;
        let fill = 0, color = col.muted;
        if (i < stage) { fill = 1; color = col.green; }
        else if (i === stage) { fill = state === 'running' ? clock / STAGE_SECONDS : 1; color = state === 'rolled' ? col.red : col.blue; }
        if (i === stage && state === 'done') { fill = 1; color = col.green; }
        if (fill > 0) { g.fillStyle = color; g.fillRect(x0, B - hgt, bw, hgt * Math.min(1, fill)); }
        g.fillStyle = col.fg; g.fillText(pct + '%', L + gw * i + gw / 2, B + 16);
        g.fillStyle = col.muted; g.font = '11px ' + MONO;
        g.fillText('SE ' + AI.fmt(100 * se(pct), 1) + '%', L + gw * i + gw / 2, B + 30);
        g.font = '12px ' + MONO;
      });
      g.textAlign = 'left'; g.fillStyle = col.muted; g.font = '11px ' + MONO;
      g.fillText('bar height ∝ √(traffic); grey = not reached, blue = running, green = passed, red = rolled back', L + 4, Tp + 10);
      const status = state === 'running' ? `stage ${stage + 1} of ${STAGES.length} (${STAGES[stage]}% traffic)` : state === 'rolled' ? 'rolled back' : 'full rollout';
      rd.innerHTML =
        `status: ${status}. Guard ${AI.fmt(GUARD(), 1)}%. True change ${AI.fmt(TRUE(), 0)}%.<br>` +
        `standard error at 1% traffic = ${AI.fmt(100 * se(1), 2)}% · guard / SE = ${AI.fmt(GUARD() / (100 * se(1)), 2)}. ` +
        `A guard close to the noise rolls back good releases and passes bad ones.<br>` +
        (lines.length ? lines.map((l) => '· ' + l).join('<br>') : '· no stage finished yet');
    };

    live(root, [c], () => stage + '|' + state + '|' + Math.floor(clock * 10) + '|' + TRUE() + '|' + GUARD() + '|' + lines.length, update, draw);
  };

  /* ======================================================================
   * 9.6  oversight-gate: risk-scored actions routed to auto, review or block
   * ==================================================================== */
  AI.labs.oversightGate = function (root) {
    const HARM = 0.7;            // hidden severity at or above which an action is harmful (illustrative)
    const { view, ctl } = AI.labShell(root, 'Risk-scored actions with human oversight',
      'Illustrative: each proposed action has a hidden severity and a noisy risk score. Score below a: auto-approved. Between a and b: human review. At or above b: blocked. Harmful means severity ≥ 0.7.',
      CANVAS(250) + '<div class="rl-read" style="font:13px ' + MONO + ';margin-top:8px;line-height:1.6"></div>', '');
    const cv = view.querySelector('canvas'), c = AI.canvas(cv), rd = view.querySelector('.rl-read');
    // 400 pre-generated proposals from a fixed seed: severity s = U^2 (mostly low), score = s + noise, reviewer draw u
    const rr = AI.rng(3);
    const PROP = Array.from({ length: 400 }, () => {
      const s = rr() ** 2;
      const score = AI.clamp(s + 0.1 * AI.randn(rr), 0, 1);
      return { s, score, u: rr() };
    });
    let shown = 0, clock = 0;
    const reset = () => { shown = 0; clock = 0; };
    const AUTO = slider(ctl, 'Auto-approve below risk score a', 0, 0.9, 0.01, 0.3, '', 2, () => reset());
    const BLOCK = slider(ctl, 'Block at or above risk score b', 0.05, 1, 0.01, 0.7, '', 2, () => reset());
    const CATCH = slider(ctl, 'Reviewer catch rate', 0, 1, 0.01, 0.9, '', 2, () => reset());
    const T = transport(ctl, reset, 1);

    const decide = (p, a, b) => {
      const bb = Math.max(a, b);
      const route = p.score < a ? 'auto' : p.score >= bb ? 'block' : 'review';
      const harmful = p.s >= HARM;
      let executed = false;
      if (route === 'auto') executed = true;
      else if (route === 'review') executed = !harmful || p.u >= CATCH();  // an uncaught harm is executed
      return { route, harmful, executed };
    };

    const update = (dt) => {
      let steps = T.take();
      if (T.playing) { clock += dt * T.speed() * 2; steps += Math.floor(clock); clock -= Math.floor(clock); }
      shown = Math.min(PROP.length, shown + steps);
    };

    const draw = () => {
      const g = c.ctx, W = c.w, H = c.h, col = AI.col();
      g.clearRect(0, 0, W, H);
      const a = AUTO(), b = Math.max(BLOCK(), a);
      const L = 20, R = W - 20, top = 26, bot = H - 36;
      const X = (v) => L + (R - L) * v;
      // zones
      g.globalAlpha = 0.18;
      g.fillStyle = col.green; g.fillRect(X(0), top, X(a) - X(0), bot - top);
      g.fillStyle = col.yellow; g.fillRect(X(a), top, X(b) - X(a), bot - top);
      g.fillStyle = col.red; g.fillRect(X(b), top, X(1) - X(b), bot - top);
      g.globalAlpha = 1;
      g.strokeStyle = col.border; g.beginPath(); g.moveTo(L, bot); g.lineTo(R, bot); g.stroke();
      g.fillStyle = col.muted; g.font = '11px ' + MONO; g.textAlign = 'center';
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => g.fillText(v.toFixed(2), X(v), bot + 14));
      g.fillStyle = col.green; g.fillText('auto', (X(0) + X(a)) / 2, top - 8);
      g.fillStyle = col.yellow; g.fillText('review', (X(a) + X(b)) / 2, top - 8);
      g.fillStyle = col.red; g.fillText('block', (X(b) + X(1)) / 2, top - 8);
      g.fillStyle = col.muted; g.fillText('risk score', (L + R) / 2, bot + 28);
      // last 120 proposals as dots
      const from = Math.max(0, shown - 120);
      let stats = { auto: 0, review: 0, block: 0, harmTotal: 0, harmExec: 0, falseBlock: 0, reviewsDone: 0 };
      for (let i = 0; i < shown; i++) {
        const p = PROP[i], d = decide(p, a, b);
        if (d.route === 'auto') stats.auto++;
        if (d.route === 'review') stats.review++;
        if (d.route === 'block') stats.block++;
        if (d.harmful) stats.harmTotal++;
        if (d.harmful && d.executed) stats.harmExec++;
        if (d.route === 'block' && !d.harmful) stats.falseBlock++;
        if (d.route === 'review') stats.reviewsDone++;
        if (i >= from) {
          const y = top + 20 + (bot - top - 40) * (0.15 + 0.7 * ((i * 0.618) % 1));
          const x = X(p.score);
          g.beginPath(); g.arc(x, y, 4, 0, 7);
          if (d.harmful) g.fillStyle = col.red; else g.fillStyle = col.blue;
          if (d.executed) { g.globalAlpha = 0.95; g.fill(); } else { g.globalAlpha = 0.9; g.lineWidth = 1.5; g.strokeStyle = g.fillStyle; g.stroke(); }
          g.globalAlpha = 1;
        }
      }
      g.textAlign = 'left'; g.fillStyle = col.muted; g.font = '11px ' + MONO;
      g.fillText('dot = proposal (red = harmful, blue = not); filled = executed, hollow = blocked', L, H - 8);
      const n = shown || 1;
      const last = shown > 0 ? PROP[shown - 1] : null;
      const lastD = last ? decide(last, a, b) : null;
      rd.innerHTML =
        `proposals seen: ${shown} · auto ${AI.fmt(100 * stats.auto / n, 1)}% · review ${AI.fmt(100 * stats.review / n, 1)}% · blocked ${AI.fmt(100 * stats.block / n, 1)}%<br>` +
        `harmful proposals: ${stats.harmTotal} · harmful actions executed: ${stats.harmExec} (${AI.fmt(100 * stats.harmExec / Math.max(1, stats.harmTotal), 1)}% of harmful) · false blocks: ${stats.falseBlock}<br>` +
        `reviews needed per 100 proposals: ${AI.fmt(100 * stats.reviewsDone / n, 1)}<br>` +
        (last ? `latest #${shown}: score ${AI.fmt(last.score, 2)} → ${lastD.route}${lastD.executed ? ', executed' : ', not executed'}${lastD.harmful ? ' (harmful)' : ''}` : 'no proposals yet');
    };

    live(root, [c], () => shown + '|' + AUTO() + '|' + Math.max(BLOCK(), AUTO()) + '|' + CATCH(), update, draw);
  };
})();
