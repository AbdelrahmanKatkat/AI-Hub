/* Data Science Module 5 labs: experiments and causal inference.
   Every simulation is illustrative (seeded random numbers, so a run can be replayed). Animations run on dt, start slowly,
   and have speed, pause, step and reset controls. Colours come from AI.col() so both themes work. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = (more) => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Replay</button>${more || ''}</div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const invPhi = (p) => { let lo = -9, hi = 9; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (Phi(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const pTwo = (z) => 2 * (1 - Phi(Math.abs(z)));
  const f = (v, d) => (Number.isFinite(v) ? v.toFixed(d === undefined ? 3 : d) : '–');
  const pct = (v, d) => (v * 100).toFixed(d === undefined ? 1 : d) + '%';

  /* Shared wiring: play/pause button, step, replay. Returns helpers. */
  function wire(ctl, api) {
    const pb = q(ctl, '#play');
    const show = () => { pb.textContent = api.playing ? '⏸ Pause' : '▶ Play'; };
    pb.onclick = () => { if (!api.playing && api.finished && api.finished()) api.reset(); api.playing = !api.playing; show(); };
    q(ctl, '#step').onclick = () => { api.playing = false; show(); api.step(); };
    q(ctl, '#reset').onclick = () => { api.reset(); api.playing = true; show(); };
    return show;
  }

  /* ---- 1. Randomised A/B test: two groups from one population, the difference settling ---- */
  AI.labs['ds-ex-ab'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A/B test: users arrive and are randomised (illustrative)', 'one population, two groups, a difference that settles',
      `<canvas style="height:400px"></canvas>`,
      `${sl('lift', 'True effect of B (outcome units)', 0, 4, 0.1, 1)}
       ${sl('sd', 'Noise: spread of users (sd)', 1, 10, 0.5, 4)}
       ${sl('speed', 'Users per second', 1, 60, 1, 5)}
       <label>Who gets B?<select id="mode"><option value="coin">A coin flip (randomised)</option><option value="keen">Keen users opt in to B</option></select></label>
       ${btns('<button class="btn" id="newd">New draw</button>')}
       <div class="readout" id="out"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const MU = 10, NMAX = 800;
    let seed = 11, rnd, users, st, hist, tick = AI.ticker(), dirty = true;
    const api = { playing: true, finished: () => st.n >= NMAX };
    const read = () => ({ lift: +q(ctl, '#lift').value, sd: +q(ctl, '#sd').value, mode: q(ctl, '#mode').value });
    const addUser = () => {
      const P = read();
      const y0 = MU + P.sd * AI.randn(rnd);
      const pT = P.mode === 'coin' ? 0.5 : 1 / (1 + Math.exp(-1.6 * (y0 - MU) / P.sd));
      const t = rnd() < pT ? 1 : 0, y = y0 + (t ? P.lift : 0), jit = rnd();
      users.push({ t, y, jit });
      const s = st[t]; s.n++; s.s += y; s.ss += y * y; st.n++;
      if (st[0].n >= 2 && st[1].n >= 2) {
        const v = (s0) => Math.max(0, (s0.ss - s0.s * s0.s / s0.n) / (s0.n - 1));
        const se = Math.sqrt(v(st[0]) / st[0].n + v(st[1]) / st[1].n), d = st[1].s / st[1].n - st[0].s / st[0].n;
        hist.push({ n: st.n, d, se });
      }
      dirty = true;
    };
    const reset = () => { rnd = AI.rng(seed); users = []; st = [{ n: 0, s: 0, ss: 0 }, { n: 0, s: 0, ss: 0 }]; st.n = 0; hist = []; dirty = true; };
    api.reset = reset; api.step = () => { if (st.n < NMAX) addUser(); };
    const show = wire(ctl, api);
    const ids = ['lift', 'sd', 'speed', 'mode'];
    ids.forEach((id) => (q(ctl, '#' + id).oninput = () => { if (id !== 'speed') reset(); dirty = true; }));
    q(ctl, '#mode').onchange = () => { reset(); };
    q(ctl, '#newd').onclick = () => { seed++; reset(); api.playing = true; show(); };
    const draw = () => {
      const P = read(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#liftv').textContent = P.lift.toFixed(1); q(ctl, '#sdv').textContent = P.sd; q(ctl, '#speedv').textContent = q(ctl, '#speed').value;
      const l = 46, r = W - 14, top = 22, mid = H * 0.5;
      const lo = MU - 3 * P.sd, hi = MU + 3 * P.sd + P.lift, X = (v) => l + ((v - lo) / (hi - lo)) * (r - l);
      txt(g, 'each dot is one user\'s outcome; line = group mean', l, 14, c.muted, '11px sans-serif');
      const rowsY = [top + 40, top + 100], rowH = 44, names = ['A (control)', 'B (treatment)'], cols = [c.orange, c.blue];
      for (let k = 0; k < 2; k++) {
        const cy = rowsY[k];
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, cy + rowH / 2); g.lineTo(r, cy + rowH / 2); g.stroke();
        g.fillStyle = cols[k]; g.globalAlpha = 0.5;
        for (const u of users) if (u.t === k) { g.beginPath(); g.arc(X(u.y), cy + (u.jit - 0.5) * rowH, 2.6, 0, 7); g.fill(); }
        g.globalAlpha = 1;
        txt(g, names[k] + `  n = ${st[k].n}`, l, cy - rowH / 2 - 2, cols[k], 'bold 11px sans-serif');
        if (st[k].n) { const mx = X(st[k].s / st[k].n); g.strokeStyle = cols[k]; g.lineWidth = 2.5; g.beginPath(); g.moveTo(mx, cy - rowH / 2); g.lineTo(mx, cy + rowH / 2); g.stroke(); g.lineWidth = 1; }
      }
      // difference history
      const py0 = mid + 52, py1 = H - 26, yr = P.lift + P.sd * 0.9;
      const Y = (v) => py1 - ((v + yr) / (2 * yr)) * (py1 - py0), nmax = Math.max(60, Math.ceil(st.n / 100) * 100), Xn = (n) => l + (n / nmax) * (r - l);
      txt(g, 'difference B − A as users arrive (band = ±1.96 standard errors)', l, py0 - 8, c.muted, '11px sans-serif');
      g.save(); g.beginPath(); g.rect(l, py0, r - l, py1 - py0); g.clip();
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, Y(0)); g.lineTo(r, Y(0)); g.stroke();
      if (hist.length > 1) {
        g.fillStyle = c.blue; g.globalAlpha = 0.18; g.beginPath();
        hist.forEach((h, i) => (i ? g.lineTo(Xn(h.n), Y(h.d + 1.96 * h.se)) : g.moveTo(Xn(h.n), Y(h.d + 1.96 * h.se))));
        for (let i = hist.length - 1; i >= 0; i--) g.lineTo(Xn(hist[i].n), Y(hist[i].d - 1.96 * hist[i].se));
        g.closePath(); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => (i ? g.lineTo(Xn(h.n), Y(h.d)) : g.moveTo(Xn(h.n), Y(h.d)))); g.stroke(); g.lineWidth = 1;
      }
      g.strokeStyle = c.green; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(l, Y(P.lift)); g.lineTo(r, Y(P.lift)); g.stroke(); g.setLineDash([]);
      g.restore();
      txt(g, 'true effect', r - 2, Y(P.lift) - 4, c.green, '11px sans-serif', 'right');
      txt(g, '0', l - 6, Y(0) + 4, c.muted, '11px sans-serif', 'right');
      txt(g, `users: 0 … ${nmax}`, r, py1 + 16, c.muted, '11px sans-serif', 'right');
      const h = hist[hist.length - 1];
      q(ctl, '#out').textContent = h
        ? `mean A = ${f(st[0].s / st[0].n, 2)}   mean B = ${f(st[1].s / st[1].n, 2)}\ndifference = ${f(h.d, 2)}   SE = ${f(h.se, 2)}\ndifference / SE = ${f(h.d / h.se, 2)}`
        : 'waiting for users…';
      q(ctl, '#cap').textContent = P.mode === 'coin'
        ? 'Each user is assigned by a coin flip, so the two groups are alike on average and the gap estimates the effect of B. The band narrows roughly like 1/√n.'
        : 'Keen users (high outcome) opt in to B. Even with effect 0 the gap is positive: selection, not B, creates it. Randomising removes this.';
    };
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && st.n < NMAX) { const k = tick(dt, +q(ctl, '#speed').value); for (let i = 0; i < k && st.n < NMAX; i++) addUser(); if (st.n >= NMAX) { api.playing = false; show(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    reset(); draw();
  };

  /* ---- 2. Two-proportion z-test and interval for the lift ---- */
  AI.labs['ds-ex-ztest'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Conversion test: z-test and interval for the lift (illustrative)', 'both arms receive users at the same pace; the interval shrinks around the truth',
      `<canvas style="height:340px"></canvas>`,
      `${sl('pa', 'Control conversion rate (%)', 1, 20, 0.5, 5)}
       ${sl('lift', 'True lift of B (percentage points)', -2, 4, 0.1, 1.2)}
       ${sl('nmax', 'Planned users per arm', 500, 20000, 500, 4000)}
       ${sl('alpha', 'Significance level α', 0.01, 0.2, 0.01, 0.05)}
       ${sl('speed', 'Users per arm per second', 10, 400, 10, 100)}
       ${btns('<button class="btn" id="newd">New draw</button>')}
       <div class="readout" id="out"></div>
       <p class="caption">The vertical dotted line is the planned sample size for 80% power at the true lift (when the lift is not zero). A small interval needs many users.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    let seed = 5, rnd, n, xa, xb, hist, tick = AI.ticker(), dirty = true;
    const api = { playing: true, finished: () => n >= +q(ctl, '#nmax').value };
    const P = () => ({ pa: +q(ctl, '#pa').value / 100, lift: +q(ctl, '#lift').value / 100, N: +q(ctl, '#nmax').value, al: +q(ctl, '#alpha').value });
    const stats = () => {
      const p = P(), pa = xa / n, pb = xb / n, pp = (xa + xb) / (2 * n), d = pb - pa;
      const sep = Math.sqrt(pp * (1 - pp) * 2 / n), z = sep > 0 ? d / sep : 0, se = Math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n), zc = invPhi(1 - p.al / 2);
      return { pa, pb, pp, d, sep, z, pv: pTwo(z), se, lo: d - zc * se, hi: d + zc * se, zc };
    };
    const addUsers = (k) => {
      const p = P(), pb = Math.min(0.999, Math.max(0.0005, p.pa + p.lift));
      for (let i = 0; i < k && n < p.N; i++) { n++; if (rnd() < p.pa) xa++; if (rnd() < pb) xb++; if (n % 25 === 0 && n >= 100) { const s = stats(); hist.push({ n, d: s.d, lo: s.lo, hi: s.hi }); } }
      dirty = true;
    };
    const reset = () => { rnd = AI.rng(seed); n = 0; xa = 0; xb = 0; hist = []; dirty = true; };
    api.reset = reset; api.step = () => addUsers(100);
    const show = wire(ctl, api);
    ['pa', 'lift', 'nmax', 'alpha'].forEach((id) => (q(ctl, '#' + id).oninput = () => { if (id === 'alpha') { dirty = true; return; } reset(); }));
    q(ctl, '#speed').oninput = () => { dirty = true; };
    q(ctl, '#newd').onclick = () => { seed++; reset(); api.playing = true; show(); };
    const draw = () => {
      const p = P(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#pav').textContent = (p.pa * 100).toFixed(1) + '%'; q(ctl, '#liftv').textContent = (p.lift * 100).toFixed(1) + ' pp';
      q(ctl, '#nmaxv').textContent = p.N; q(ctl, '#alphav').textContent = p.al.toFixed(2); q(ctl, '#speedv').textContent = q(ctl, '#speed').value;
      const l = 50, r = W - 16, top = 26, bot = H - 30, yr = Math.max(2.5, Math.abs(p.lift * 100) + 2.5);
      const Y = (v) => bot - ((v * 100 + yr) / (2 * yr)) * (bot - top), Xn = (k) => l + (k / p.N) * (r - l);
      txt(g, `lift of B over A with a ${((1 - p.al) * 100).toFixed(0)}% interval (percentage points)`, l, 16, c.muted, '11px sans-serif');
      g.save(); g.beginPath(); g.rect(l, top, r - l, bot - top); g.clip();
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(l, Y(0)); g.lineTo(r, Y(0)); g.stroke();
      if (hist.length > 1) {
        g.fillStyle = c.blue; g.globalAlpha = 0.2; g.beginPath();
        hist.forEach((h, i) => (i ? g.lineTo(Xn(h.n), Y(h.hi)) : g.moveTo(Xn(h.n), Y(h.hi))));
        for (let i = hist.length - 1; i >= 0; i--) g.lineTo(Xn(hist[i].n), Y(hist[i].lo));
        g.closePath(); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => (i ? g.lineTo(Xn(h.n), Y(h.d)) : g.moveTo(Xn(h.n), Y(h.d)))); g.stroke(); g.lineWidth = 1;
      }
      g.strokeStyle = c.green; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(l, Y(p.lift)); g.lineTo(r, Y(p.lift)); g.stroke(); g.setLineDash([]);
      g.restore();
      txt(g, 'no difference (0)', r, Y(0) - 4, c.red, '11px sans-serif', 'right');
      txt(g, 'true lift', r, Y(p.lift) - 4, c.green, '11px sans-serif', 'right');
      const pb = p.pa + p.lift;
      if (Math.abs(p.lift) > 1e-9 && pb > 0 && pb < 1) {
        const pbar = (p.pa + pb) / 2, zc = invPhi(1 - p.al / 2), need = Math.pow(zc * Math.sqrt(2 * pbar * (1 - pbar)) + 0.8416 * Math.sqrt(p.pa * (1 - p.pa) + pb * (1 - pb)), 2) / (p.lift * p.lift);
        if (need <= p.N) { g.strokeStyle = c.muted; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(Xn(need), top); g.lineTo(Xn(need), bot); g.stroke(); g.setLineDash([]); txt(g, `planned n ≈ ${Math.round(need)}`, Xn(need) + 4, top + 12, c.muted, '11px sans-serif'); }
      }
      txt(g, `users per arm: 0 … ${p.N}`, r, H - 8, c.muted, '11px sans-serif', 'right');
      if (n >= 30) {
        const s = stats();
        q(ctl, '#out').textContent =
          `control    ${xa} / ${n} = ${pct(s.pa, 2)}\ntreatment  ${xb} / ${n} = ${pct(s.pb, 2)}\nlift = ${f(s.d * 100, 2)} pp\npooled rate = ${pct(s.pp, 2)}, SE = ${f(s.sep * 100, 3)} pp\nz = ${f(s.z, 3)}   p = ${f(s.pv, 4)}\n${((1 - p.al) * 100).toFixed(0)}% interval: [${f(s.lo * 100, 2)}, ${f(s.hi * 100, 2)}] pp\n${s.pv < p.al ? 'p < α: evidence the arms differ' : 'p ≥ α: the data are compatible with no difference'}`;
      } else q(ctl, '#out').textContent = 'collecting users…';
    };
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && n < P().N) { addUsers(tick(dt, +q(ctl, '#speed').value)); if (n >= P().N) { api.playing = false; show(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    reset(); draw();
  };

  /* ---- 3. Peeking and sequential boundaries over many simulated experiments ---- */
  AI.labs['ds-ex-peek'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Peeking at 4000 simulated experiments (illustrative)', 'each line is one experiment; the test statistic z wanders as users arrive',
      `<canvas style="height:400px"></canvas>`,
      `<label>Number of looks<select id="K"><option value="1">1 (look once, at the end)</option><option value="2">2</option><option value="4">4</option><option value="5" selected>5</option><option value="10">10</option><option value="20">20</option></select></label>
       <label>Stopping rule<select id="rule"><option value="fixed">Stop when |z| &gt; 1.96 at any look</option><option value="pocock">Pocock: one constant boundary</option><option value="obf">O'Brien–Fleming: strict early, relaxed late</option></select></label>
       ${sl('eff', 'True effect (z at the end; 0 = none)', 0, 4, 0.1, 0)}
       ${sl('speed', 'Speed (grid steps per second)', 0.2, 3, 0.1, 0.6)}
       ${btns()}
       <div class="readout" id="out"></div>
       <p class="caption">Thin lines: 40 of the 4000 simulated experiments. A line turns red once it has crossed a boundary at a look. The percentages count all 4000. With effect 0 every red line is a false positive.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const NP = 4000, G = 20, Z0 = new Float32Array(NP * G);
    { const r = AI.rng(2024); for (let i = 0; i < NP; i++) { let s = 0; for (let k = 1; k <= G; k++) { s += AI.randn(r); Z0[i * G + k - 1] = s / Math.sqrt(k); } } }
    const pctile = (arr) => { const a = Float64Array.from(arr).sort(); return a[Math.floor(0.95 * a.length)]; };
    let t = 0, dirty = true, cfg = null, first, cum, finalSig, bnd, looks;
    const api = { playing: true, finished: () => t >= G };
    const recompute = () => {
      const K = +q(ctl, '#K').value, rule = q(ctl, '#rule').value, eff = +q(ctl, '#eff').value, every = G / K;
      looks = []; for (let j = 1; j <= K; j++) looks.push(j * every);
      let cst = 1.96;
      if (rule === 'pocock' || rule === 'obf') {
        const m = new Float64Array(NP);
        for (let i = 0; i < NP; i++) { let mx = 0; looks.forEach((L, j) => { const a = Math.abs(Z0[i * G + L - 1]) * (rule === 'obf' ? Math.sqrt((j + 1) / K) : 1); if (a > mx) mx = a; }); m[i] = mx; }
        cst = pctile(m);
      }
      bnd = looks.map((L, j) => (rule === 'obf' ? cst * Math.sqrt(K / (j + 1)) : cst));
      first = new Int16Array(NP).fill(-1); cum = new Float64Array(K + 1); finalSig = 0;
      const Zv = (i, L) => Z0[i * G + L - 1] + eff * Math.sqrt(L / G);
      for (let i = 0; i < NP; i++) {
        for (let j = 0; j < K; j++) if (Math.abs(Zv(i, looks[j])) > bnd[j]) { first[i] = j; break; }
        if (first[i] >= 0) for (let j = first[i] + 1; j <= K; j++) cum[j]++;
        if (Math.abs(Zv(i, G)) > 1.96) finalSig++;
      }
      cfg = { K, rule, eff, cst };
      dirty = true;
    };
    const reset = () => { t = 0; dirty = true; };
    api.reset = reset; api.step = () => { const e = G / cfg.K; t = Math.min(G, (Math.floor(t / e + 1e-9) + 1) * e); dirty = true; };
    const show = wire(ctl, api);
    ['K', 'rule', 'eff'].forEach((id) => { q(ctl, '#' + id).oninput = () => { recompute(); reset(); }; q(ctl, '#' + id).onchange = () => { recompute(); reset(); }; });
    q(ctl, '#speed').oninput = () => { dirty = true; };
    const zAt = (i, k) => (k <= 0 ? 0 : Z0[i * G + k - 1] + cfg.eff * Math.sqrt(k / G));
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#effv').textContent = cfg.eff.toFixed(1); q(ctl, '#speedv').textContent = (+q(ctl, '#speed').value).toFixed(1);
      const l = 40, r = W - 14, top = 22, bot = H - 30, ZR = 5, X = (k) => l + (k / G) * (r - l), Y = (z) => top + ((ZR - z) / (2 * ZR)) * (bot - top);
      txt(g, 'z statistic of the difference  (x axis: users collected, left to right)', l, 14, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, Y(0)); g.lineTo(r, Y(0)); g.stroke();
      [-4, -2, 2, 4].forEach((v) => txt(g, String(v), l - 6, Y(v) + 4, c.muted, '10px sans-serif', 'right'));
      // boundaries and looks
      g.strokeStyle = c.muted; g.setLineDash([2, 4]);
      looks.forEach((L) => { g.beginPath(); g.moveTo(X(L), top); g.lineTo(X(L), bot); g.stroke(); });
      g.setLineDash([]);
      g.strokeStyle = c.yellow; g.lineWidth = 2;
      looks.forEach((L, j) => { [1, -1].forEach((sg) => { g.beginPath(); g.moveTo(X(L) - 9, Y(sg * bnd[j])); g.lineTo(X(L) + 9, Y(sg * bnd[j])); g.stroke(); }); });
      g.lineWidth = 1;
      // the 40 sample paths up to time t
      const kf = Math.floor(t), frac = t - kf;
      for (let i = 0; i < 40; i++) {
        let crossed = false; for (let j = 0; j < looks.length; j++) if (looks[j] <= t + 1e-9 && first[i] >= 0 && first[i] <= j) { crossed = true; break; }
        g.strokeStyle = crossed ? c.red : c.blue; g.globalAlpha = crossed ? 0.9 : 0.35; g.lineWidth = crossed ? 1.6 : 1;
        g.beginPath(); g.moveTo(X(0), Y(0));
        for (let k = 1; k <= kf; k++) g.lineTo(X(k), Y(zAt(i, k)));
        if (frac > 0 && kf < G) { const a = zAt(i, kf), b = zAt(i, kf + 1); g.lineTo(X(t), Y(a + (b - a) * frac)); }
        g.stroke();
      }
      g.globalAlpha = 1; g.lineWidth = 1;
      const row = looks.filter((L) => L <= t + 1e-9).length;
      const stopped = cum[row] / NP;
      const final = finalSig / NP;
      q(ctl, '#out').textContent =
        `looks so far: ${row} of ${cfg.K}\nboundary constant: ${f(cfg.cst, 3)}${cfg.rule === 'obf' ? ' × √(K/j)' : ''}\ncrossed a boundary by now: ${pct(stopped)}\n(one look at the end: ${pct(final)})\n${cfg.eff === 0 ? 'effect 0: all crossings are false' : 'effect > 0: crossings mix signal and luck'}`;
    };
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && t < G) { t = Math.min(G, t + dt * +q(ctl, '#speed').value); dirty = true; if (t >= G) { api.playing = false; show(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    recompute(); draw();
  };

  /* ---- 4. Multiple comparisons: Bonferroni and Benjamini-Hochberg ---- */
  AI.labs['ds-ex-multi'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Many tests, one threshold? Bonferroni and Benjamini–Hochberg (illustrative)', 'p-values arrive one at a time and take their sorted rank',
      `<canvas style="height:400px"></canvas>`,
      `${sl('m', 'Number of tests m', 5, 60, 1, 20)}
       ${sl('frac', 'Share of tests with a real effect', 0, 0.5, 0.05, 0.2)}
       ${sl('eff', 'Size of a real effect (z)', 1, 5, 0.25, 3)}
       ${sl('alpha', 'α', 0.01, 0.1, 0.01, 0.05)}
       ${sl('speed', 'Tests per second', 0.5, 8, 0.5, 2)}
       ${btns('<button class="btn" id="newd">New draw</button><button class="btn" id="avg">Average over 500 draws</button>')}
       <div class="readout" id="out"></div>
       <p class="caption">Sorted p-values on a log axis. Lines: α (flat, orange), Bonferroni α/m (flat, red), Benjamini–Hochberg iα/m (rising, green). Green dots are real effects, grey dots are nulls; a BH rejection fills the dot.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    let seed = 3, pts = [], arrived = 0, tick = AI.ticker(), dirty = true, avgText = '';
    const api = { playing: true, finished: () => arrived >= pts.length };
    const P = () => ({ m: +q(ctl, '#m').value, frac: +q(ctl, '#frac').value, eff: +q(ctl, '#eff').value, al: +q(ctl, '#alpha').value });
    const gen = (sd) => {
      const p = P(), r = AI.rng(sd), nReal = Math.round(p.m * p.frac), real = [];
      for (let i = 0; i < p.m; i++) real.push(i < nReal);
      for (let i = p.m - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [real[i], real[j]] = [real[j], real[i]]; }
      return real.map((rl) => { const z = (rl ? p.eff : 0) + AI.randn(r); return { real: rl, p: Math.max(1e-12, pTwo(z)), x: 0, age: 0 }; });
    };
    const decide = (list, p) => {
      const s = list.slice().sort((a, b) => a.p - b.p), m = list.length; let kstar = 0;
      s.forEach((u, i) => { if (u.p <= ((i + 1) * p.al) / m) kstar = i + 1; });
      s.forEach((u, i) => { u.bh = i < kstar; u.bon = u.p <= p.al / m; u.unc = u.p < p.al; });
      return s;
    };
    const tally = (list, key) => { let rej = 0, fp = 0; list.forEach((u) => { if (u[key]) { rej++; if (!u.real) fp++; } }); return { rej, fp, tp: rej - fp }; };
    const reset = () => { pts = gen(seed); arrived = 0; dirty = true; avgText = ''; };
    api.reset = reset; api.step = () => { if (arrived < pts.length) { arrived++; dirty = true; } };
    const show = wire(ctl, api);
    ['m', 'frac', 'eff', 'alpha'].forEach((id) => (q(ctl, '#' + id).oninput = () => { reset(); }));
    q(ctl, '#speed').oninput = () => { dirty = true; };
    q(ctl, '#newd').onclick = () => { seed++; reset(); api.playing = true; show(); };
    q(ctl, '#avg').onclick = () => {
      const p = P(), acc = { unc: [0, 0], bon: [0, 0], bh: [0, 0] }, N = 500; let nullsTot = 0;
      for (let d = 0; d < N; d++) { const L = gen(1000 + d); decide(L, p); L.forEach((u) => { if (!u.real) nullsTot++; }); ['unc', 'bon', 'bh'].forEach((k) => { const t = tally(L, k); acc[k][0] += t.fp; acc[k][1] += t.rej; }); }
      const fdr = (k) => (acc[k][1] ? acc[k][0] / acc[k][1] : 0);
      avgText = `\naverage of 500 draws (false positives per draw | false discovery proportion):\n  uncorrected ${f(acc.unc[0] / N, 2)} | ${pct(fdr('unc'))}\n  Bonferroni  ${f(acc.bon[0] / N, 2)} | ${pct(fdr('bon'))}\n  BH          ${f(acc.bh[0] / N, 2)} | ${pct(fdr('bh'))}`;
      dirty = true;
    };
    const draw = () => {
      const p = P(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#mv').textContent = p.m; q(ctl, '#fracv').textContent = (p.frac * 100).toFixed(0) + '% (' + Math.round(p.m * p.frac) + ' tests)'; q(ctl, '#effv').textContent = p.eff.toFixed(2);
      q(ctl, '#alphav').textContent = p.al.toFixed(2); q(ctl, '#speedv').textContent = (+q(ctl, '#speed').value).toFixed(1);
      const l = 52, r = W - 14, top = 22, bot = H - 30, LO = -4;
      const X = (k) => l + ((k - 0.5) / p.m) * (r - l), Y = (v) => top + ((0 - Math.log10(Math.max(v, 1e-4))) / -LO) * (bot - top);
      txt(g, 'p-value (log scale) against rank', l, 14, c.muted, '11px sans-serif');
      [1, 0.1, 0.01, 0.001, 0.0001].forEach((v) => { g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, Y(v)); g.lineTo(r, Y(v)); g.stroke(); txt(g, String(v), l - 6, Y(v) + 4, c.muted, '10px sans-serif', 'right'); });
      g.lineWidth = 2;
      g.strokeStyle = c.orange; g.beginPath(); g.moveTo(l, Y(p.al)); g.lineTo(r, Y(p.al)); g.stroke();
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(l, Y(p.al / p.m)); g.lineTo(r, Y(p.al / p.m)); g.stroke();
      g.strokeStyle = c.green; g.beginPath(); for (let i = 1; i <= p.m; i++) { const px = X(i), py = Y((i * p.al) / p.m); i > 1 ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke();
      g.lineWidth = 1;
      txt(g, 'α', r, Y(p.al) - 4, c.orange, '11px sans-serif', 'right'); txt(g, 'α/m', r, Y(p.al / p.m) - 4, c.red, '11px sans-serif', 'right'); txt(g, 'BH: iα/m', X(p.m) - 6, Y(p.al) + 14, c.green, '11px sans-serif', 'right');
      const done = arrived >= pts.length, shown = pts.slice(0, arrived);
      const sorted = decide(shown.length ? shown : [], p);
      let moving = false;
      sorted.forEach((u, i) => { const tx = X(i + 1); u.x = u.x ? u.x + (tx - u.x) * 0.25 : tx; if (Math.abs(tx - u.x) > 0.4) moving = true; });
      let full = null;
      if (done) { full = decide(pts, p); full.forEach((u, i) => (u.x = u.x ? u.x + (X(i + 1) - u.x) * 0.25 : X(i + 1))); }
      sorted.forEach((u) => {
        const rad = 5 * Math.min(1, u.age / 0.5);
        const col = u.real ? c.green : c.muted;
        g.beginPath(); g.arc(u.x, Y(u.p), Math.max(rad, 0.1), 0, 7);
        if (done && u.bh) { g.fillStyle = col; g.fill(); } else if (!done && u.unc) { g.fillStyle = c.orange; g.fill(); } else { g.strokeStyle = col; g.lineWidth = 2; g.stroke(); g.lineWidth = 1; }
        if (done && u.bon) { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.arc(u.x, Y(u.p), 8, 0, 7); g.stroke(); g.lineWidth = 1; }
      });
      if (done) {
        const rs = [['uncorrected (p < α)', 'unc'], ['Bonferroni', 'bon'], ['Benjamini–Hochberg', 'bh']].map(([nm, k]) => { const t = tally(pts, k); return `${nm.padEnd(20)} rejects ${String(t.rej).padStart(2)}  false ${String(t.fp).padStart(2)}  real found ${String(t.tp).padStart(2)}`; });
        const nReal = pts.filter((u) => u.real).length;
        q(ctl, '#out').textContent = `${p.m} tests, ${nReal} real, ${p.m - nReal} null\n` + rs.join('\n') + avgText;
      } else q(ctl, '#out').textContent = `p-values received: ${arrived} of ${p.m}\n(orange = p < α uncorrected)\nwait for all m before correcting`;
      return moving;
    };
    let moving = false;
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && arrived < pts.length) { const k = tick(dt, +q(ctl, '#speed').value); if (k) { arrived = Math.min(pts.length, arrived + k); dirty = true; } if (arrived >= pts.length) { api.playing = false; show(); } }
      let growing = false; pts.forEach((u, i) => { if (i < arrived && u.age < 0.5) { u.age += dt; growing = true; } });
      if (dirty || moving || growing) { dirty = false; moving = draw(); }
    }, { fps: 30 });
    reset(); moving = draw();
  };

  /* ---- 5. Confounding: naive difference versus stratified ---- */
  AI.labs['ds-ex-confound'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A confounder shifts the apparent effect (illustrative)', 'power users both adopt a feature more and retain more, whatever the feature does',
      `<canvas style="height:420px"></canvas>`,
      `${sl('tau', 'True effect of the feature on retention', 0, 0.2, 0.01, 0.05)}
       ${sl('s', 'Confounding strength s (0 = random assignment)', 0, 1, 0.05, 0.75)}
       ${sl('speed', 'Users per second', 5, 200, 5, 50)}
       ${btns('<button class="btn" id="newd">New draw</button>')}
       <div class="readout" id="out"></div>
       <p class="caption">Power users (40% of users) retain far more and, when s &gt; 0, adopt the feature far more. The red line compares all adopters with all non-adopters. The green line compares inside each group and averages by group size (stratification).</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const NMAX = 4000;
    let seed = 8, rnd, N, cell, hist, tick = AI.ticker(), dirty = true;
    const api = { playing: true, finished: () => N >= NMAX };
    const P = () => ({ tau: +q(ctl, '#tau').value, s: +q(ctl, '#s').value });
    const est = () => {
      const r = (z, t) => (cell[z][t].n ? cell[z][t].y / cell[z][t].n : NaN);
      const n1 = cell[0][1].n + cell[1][1].n, n0 = cell[0][0].n + cell[1][0].n;
      const naive = (cell[0][1].y + cell[1][1].y) / n1 - (cell[0][0].y + cell[1][0].y) / n0;
      let adj = 0; for (let z = 0; z < 2; z++) adj += ((cell[z][0].n + cell[z][1].n) / N) * (r(z, 1) - r(z, 0));
      return { naive, adj };
    };
    const add = () => {
      const p = P(), z = rnd() < 0.4 ? 1 : 0, pT = z ? 0.5 + 0.4 * p.s : 0.5 - 0.4 * p.s, t = rnd() < pT ? 1 : 0;
      const y = rnd() < 0.25 + 0.6 * z + p.tau * t ? 1 : 0;
      cell[z][t].n++; cell[z][t].y += y; N++;
      if (N % 20 === 0 && cell.every((row) => row.every((x) => x.n >= 3))) { const e = est(); hist.push({ n: N, naive: e.naive, adj: e.adj }); }
      dirty = true;
    };
    const reset = () => { rnd = AI.rng(seed); N = 0; cell = [[{ n: 0, y: 0 }, { n: 0, y: 0 }], [{ n: 0, y: 0 }, { n: 0, y: 0 }]]; hist = []; dirty = true; };
    api.reset = reset; api.step = () => { for (let i = 0; i < 20 && N < NMAX; i++) add(); };
    const show = wire(ctl, api);
    ['tau', 's'].forEach((id) => (q(ctl, '#' + id).oninput = () => reset()));
    q(ctl, '#speed').oninput = () => { dirty = true; };
    q(ctl, '#newd').onclick = () => { seed++; reset(); api.playing = true; show(); };
    const arrow = (g, x0, y0, x1, y1, w, col) => {
      const a = Math.atan2(y1 - y0, x1 - x0), ex = x1 - Math.cos(a) * 26, ey = y1 - Math.sin(a) * 26, sx = x0 + Math.cos(a) * 26, sy = y0 + Math.sin(a) * 26;
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(sx, sy); g.lineTo(ex, ey); g.stroke();
      g.beginPath(); g.moveTo(ex, ey); g.lineTo(ex - 9 * Math.cos(a - 0.4), ey - 9 * Math.sin(a - 0.4)); g.lineTo(ex - 9 * Math.cos(a + 0.4), ey - 9 * Math.sin(a + 0.4)); g.closePath(); g.fill(); g.lineWidth = 1;
    };
    const draw = () => {
      const p = P(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#tauv').textContent = p.tau.toFixed(2); q(ctl, '#sv').textContent = p.s.toFixed(2); q(ctl, '#speedv').textContent = q(ctl, '#speed').value;
      // DAG
      const cx = W / 2, zx = cx, zy = 28, tx = cx - Math.min(150, W * 0.28), ty = 98, yx = cx + Math.min(150, W * 0.28), yy = 98;
      arrow(g, zx, zy, tx, ty, 1 + 5 * p.s, c.purple); arrow(g, zx, zy, yx, yy, 4, c.purple); arrow(g, tx, ty, yx, yy, 1 + p.tau * 20, c.blue);
      [[zx, zy, 'Z power user', c.purple], [tx, ty, 'T uses feature', c.blue], [yx, yy, 'Y retained', c.green]].forEach(([x, y, s, col]) => { g.fillStyle = c.bg2; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 22, 0, 7); g.fill(); g.stroke(); g.lineWidth = 1; txt(g, s, x, y + 38, col, '11px sans-serif', 'center'); });
      txt(g, 'width = strength', 8, 14, c.muted, '10px sans-serif');
      // chart
      const l = 48, r = W - 16, top = 168, bot = H - 26, lo = -0.1, hi = 0.6, Y = (v) => bot - ((v - lo) / (hi - lo)) * (bot - top), Xn = (n) => l + (n / NMAX) * (r - l);
      txt(g, 'estimated effect on retention (difference in retention rate)', l, top - 8, c.muted, '11px sans-serif');
      [0, 0.2, 0.4, 0.6].forEach((v) => { g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, Y(v)); g.lineTo(r, Y(v)); g.stroke(); txt(g, v.toFixed(1), l - 6, Y(v) + 4, c.muted, '10px sans-serif', 'right'); });
      g.strokeStyle = c.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(l, Y(p.tau)); g.lineTo(r, Y(p.tau)); g.stroke(); g.setLineDash([]);
      txt(g, 'true effect', r, Y(p.tau) + 14, c.muted, '11px sans-serif', 'right');
      const line = (key, col) => { if (hist.length < 2) return; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => (i ? g.lineTo(Xn(h.n), Y(h[key])) : g.moveTo(Xn(h.n), Y(h[key])))); g.stroke(); g.lineWidth = 1; };
      line('naive', c.red); line('adj', c.green);
      txt(g, `users: 0 … ${NMAX}`, r, H - 6, c.muted, '11px sans-serif', 'right');
      const rt = (z, t) => (cell[z][t].n ? pct(cell[z][t].y / cell[z][t].n, 0) + ` (${cell[z][t].n})` : '–');
      let out = `            used feature   did not use\npower users  ${rt(1, 1).padEnd(13)}  ${rt(1, 0)}\ncasual users ${rt(0, 1).padEnd(13)}  ${rt(0, 0)}\n(cells: retention rate (users))\n`;
      if (hist.length) { const e = est(); out += `naive difference      ${f(e.naive, 3)}\nstratified difference ${f(e.adj, 3)}\ntrue effect           ${f(p.tau, 3)}`; } else out += 'collecting users…';
      q(ctl, '#out').textContent = out;
    };
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && N < NMAX) { const k = tick(dt, +q(ctl, '#speed').value); for (let i = 0; i < k && N < NMAX; i++) add(); if (N >= NMAX) { api.playing = false; show(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    reset(); draw();
  };

  /* ---- 6. Difference-in-differences ---- */
  AI.labs['ds-ex-did'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Difference-in-differences (illustrative)', 'weeks arrive one at a time; the policy starts in week 6 for the treated group only',
      `<canvas style="height:400px"></canvas>`,
      `<label>Shared trend g(t), edit it (t = week)<input type="text" id="expr" value="0.8*t" spellcheck="false"></label><div class="caption" id="err" style="color:var(--red);margin:0"></div>
       ${sl('eff', 'True effect of the policy', 0, 15, 0.5, 6)}
       ${sl('gap', 'Starting gap: treated minus control', 0, 25, 1, 12)}
       ${sl('slope', 'Extra trend in treated group (breaks parallel trends)', -1, 1, 0.1, 0)}
       ${sl('sd', 'Weekly noise (sd)', 0, 4, 0.25, 1)}
       ${sl('speed', 'Weeks per second', 0.2, 3, 0.1, 0.7)}
       ${btns('<button class="btn" id="newd">New draw</button>')}
       <div class="readout" id="out"></div>
       <p class="caption">Solid lines are the observed weekly means. The dashed line is the treated group's counterfactual under parallel trends: its own pre-period level moved by the control group's change.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const W_N = 12, T0 = 6;
    let seed = 21, noise = [], t = 0, g_fn = (s) => 0.8 * s.t, dirty = true;
    const api = { playing: true, finished: () => t >= W_N - 1 };
    const P = () => ({ eff: +q(ctl, '#eff').value, gap: +q(ctl, '#gap').value, slope: +q(ctl, '#slope').value, sd: +q(ctl, '#sd').value });
    const mkNoise = () => { const r = AI.rng(seed); noise = []; for (let i = 0; i < W_N; i++) noise.push([AI.randn(r), AI.randn(r)]); };
    const series = () => {
      const p = P(), C = [], Tr = [];
      for (let w = 0; w < W_N; w++) { const gv = g_fn({ t: w }); C.push(50 + gv + p.sd * noise[w][0]); Tr.push(50 + p.gap + gv + p.slope * w + (w >= T0 ? p.eff : 0) + p.sd * noise[w][1]); }
      return { C, Tr };
    };
    const ei = AI.exprInput(q(ctl, '#expr'), q(ctl, '#err'), {
      allowed: ['t'],
      check: (fn) => { for (let w = 0; w < W_N; w++) if (!Number.isFinite(fn({ t: w }))) throw new Error('g(t) must be a finite number for every week'); },
      onChange: (fn) => { g_fn = fn; dirty = true; },
    });
    const reset = () => { mkNoise(); t = 0; dirty = true; };
    api.reset = reset; api.step = () => { t = Math.min(W_N - 1, Math.floor(t + 1e-9) + 1); dirty = true; };
    const show = wire(ctl, api);
    ['eff', 'gap', 'slope', 'sd'].forEach((id) => (q(ctl, '#' + id).oninput = () => { dirty = true; }));
    q(ctl, '#speed').oninput = () => { dirty = true; };
    q(ctl, '#newd').onclick = () => { seed++; reset(); api.playing = true; show(); };
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const draw = () => {
      const p = P(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#effv').textContent = p.eff.toFixed(1); q(ctl, '#gapv').textContent = p.gap; q(ctl, '#slopev').textContent = p.slope.toFixed(1); q(ctl, '#sdv').textContent = p.sd.toFixed(2); q(ctl, '#speedv').textContent = (+q(ctl, '#speed').value).toFixed(1);
      const { C, Tr } = series(), kf = Math.floor(t + 1e-9);
      let vmin = Infinity, vmax = -Infinity; for (let w = 0; w < W_N; w++) { vmin = Math.min(vmin, C[w], Tr[w]); vmax = Math.max(vmax, C[w], Tr[w]); }
      vmin -= 3; vmax += 3;
      const l = 48, r = W - 16, top = 26, bot = H - 30, X = (w) => l + ((w + 0.5) / W_N) * (r - l), Y = (v) => bot - ((v - vmin) / (vmax - vmin)) * (bot - top);
      txt(g, 'weekly outcome (e.g. orders per store)', l, 16, c.muted, '11px sans-serif');
      g.fillStyle = c.bg3; g.globalAlpha = 0.5; g.fillRect(X(T0 - 0.5), top, r - X(T0 - 0.5), bot - top); g.globalAlpha = 1;
      txt(g, 'policy starts (week 6)', X(T0 - 0.5) + 4, top + 12, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(l, bot); g.lineTo(r, bot); g.stroke();
      for (let w = 0; w < W_N; w += 2) txt(g, String(w), X(w), bot + 14, c.muted, '10px sans-serif', 'center');
      const trace = (arr, col, lab) => {
        g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2.2; g.beginPath(); g.moveTo(X(0), Y(arr[0]));
        for (let w = 1; w <= kf; w++) g.lineTo(X(w), Y(arr[w]));
        const fr = t - kf; if (fr > 0 && kf < W_N - 1) g.lineTo(X(t), Y(arr[kf] + (arr[kf + 1] - arr[kf]) * fr));
        g.stroke(); g.lineWidth = 1;
        for (let w = 0; w <= kf; w++) { g.beginPath(); g.arc(X(w), Y(arr[w]), 3.2, 0, 7); g.fill(); }
        txt(g, lab, X(Math.min(kf, W_N - 1)) + 6, Y(arr[Math.min(kf, W_N - 1)]) - 8, col, 'bold 11px sans-serif');
      };
      trace(C, c.orange, 'control'); trace(Tr, c.blue, 'treated');
      let out = 'waiting for weeks…';
      const preN = Math.min(kf + 1, T0), postN = Math.max(0, kf + 1 - T0);
      if (preN === T0 && postN >= 1) {
        const tp = mean(Tr.slice(0, T0)), cp = mean(C.slice(0, T0)), tq = mean(Tr.slice(T0, kf + 1)), cq = mean(C.slice(T0, kf + 1));
        // counterfactual
        g.strokeStyle = c.blue; g.setLineDash([5, 4]); g.beginPath();
        for (let w = T0; w <= kf; w++) { const cf = tp + (C[w] - cp); w === T0 ? g.moveTo(X(w), Y(cf)) : g.lineTo(X(w), Y(cf)); } g.stroke(); g.setLineDash([]);
        const cfEnd = tp + (C[kf] - cp); g.strokeStyle = c.green; g.lineWidth = 3; g.beginPath(); g.moveTo(X(kf) + 8, Y(cfEnd)); g.lineTo(X(kf) + 8, Y(Tr[kf])); g.stroke(); g.lineWidth = 1;
        const did = (tq - tp) - (cq - cp), tbar = mean(Array.from({ length: postN }, (_, i) => T0 + i)), bias = p.slope * (tbar - (T0 - 1) / 2);
        out = `treated: ${f(tp, 2)} before, ${f(tq, 2)} after (change ${f(tq - tp, 2)})\ncontrol: ${f(cp, 2)} before, ${f(cq, 2)} after (change ${f(cq - cp, 2)})\nDiD = ${f(tq - tp, 2)} − ${f(cq - cp, 2)} = ${f(did, 2)}\ntrue effect ${f(p.eff, 2)}\nnaive before/after (treated only): ${f(tq - tp, 2)}\nnaive after-only gap: ${f(tq - cq, 2)}` + (Math.abs(p.slope) > 1e-9 ? `\nparallel trends violated: DiD is off by about ${f(bias, 2)}` : '');
      } else if (preN < T0) out = `pre-period: week ${kf} of ${T0 - 1}\nthe policy has not started yet`;
      q(ctl, '#out').textContent = out;
    };
    o.onresize = () => { dirty = true; };
    AI.loop(root, (dt) => {
      if (api.playing && t < W_N - 1) { t = Math.min(W_N - 1, t + dt * +q(ctl, '#speed').value); dirty = true; if (t >= W_N - 1) { api.playing = false; show(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    mkNoise(); ei.run(); draw();
  };
})();
