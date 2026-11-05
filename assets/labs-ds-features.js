/* Data Science Module 6 labs: feature engineering and data pipelines.
   All labs are illustrative: small tables and seeded simulations, not real products. Animations run on dt (slow by default),
   each has speed / pause / step / reset, and redraws only when the state changed. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const ease = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const sdev = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };   // population sd (ddof 0, as scikit-learn)
  const f = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : 'undefined');
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const box = (g, x, y, w, h, fill, stroke, dash) => {
    g.beginPath(); g.rect(x, y, w, h);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); }
  };
  const line = (g, x1, y1, x2, y2, col, w, dash) => { g.strokeStyle = col; g.lineWidth = w || 1; g.setLineDash(dash || []); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.setLineDash([]); g.lineWidth = 1; };
  const dot = (g, x, y, r, col) => { g.fillStyle = col; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); };
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const ctlTail = (spd) => `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.25" value="${spd || 1}"></label>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>
     <div class="readout" id="out"></div><p class="caption" id="cap"></p>`;

  /* Generic player: a time value t in [0, T] that advances with dt. o = {T, rate (units/s at speed 1), stepSize, loop, draw(t), startPlaying} */
  function player(root, ctl, o) {
    const S = { t: 0, playing: o.startPlaying !== false, speed: 1 };
    const play = q(ctl, '#play'), sp = q(ctl, '#sp'), spv = q(ctl, '#spv');
    const lab = () => { play.textContent = S.playing ? '⏸ Pause' : (S.t >= o.T && !o.loop ? '▶ Replay' : '▶ Play'); };
    S.redraw = () => o.draw(S.t);
    S.set = (t, keep) => { S.t = clamp(t, 0, o.T); if (!keep) S.playing = false; lab(); o.draw(S.t); };
    play.onclick = () => { if (!S.playing && S.t >= o.T && !o.loop) S.t = 0; S.playing = !S.playing; lab(); o.draw(S.t); };
    q(ctl, '#step').onclick = () => { S.playing = false; const st = o.stepSize || 1; S.t = S.t >= o.T ? 0 : Math.min(o.T, (Math.floor(S.t / st + 1e-9) + 1) * st); lab(); o.draw(S.t); };
    q(ctl, '#reset').onclick = () => { S.t = 0; S.playing = true; lab(); o.draw(0); };
    sp.oninput = () => { S.speed = +sp.value; spv.textContent = S.speed + '×'; };
    sp.oninput();
    AI.loop(root, (dt) => {
      if (!S.playing) return;
      S.t += dt * o.rate * S.speed;
      if (S.t >= o.T) { if (o.loop) S.t -= o.T; else { S.t = o.T; S.playing = false; lab(); } }
      o.draw(S.t);
    }, { fps: 30 });
    lab();
    return S;
  }
  const setup = (root, title, sub, h, ctl) => {
    const { view, ctl: c } = AI.labShell(root, title, sub, `<canvas style="height:${h}px"></canvas>`, ctl);
    return { view, ctl: c, o: AI.canvas(q(view, 'canvas')) };
  };

  /* ====================================================================== 6.1 encoding */
  const TRAIN = [[1, 'Cairo', 'Basic', 1], [2, 'Giza', 'Plus', 0], [3, 'Cairo', 'Basic', 1], [4, 'Alex', 'Pro', 0],
    [5, 'Giza', 'Basic', 1], [6, 'Cairo', 'Plus', 0], [7, 'Alex', 'Plus', 0], [8, 'Giza', 'Basic', 0]];
  const CITIES = ['Alex', 'Cairo', 'Giza'], PLANS = ['Basic', 'Plus', 'Pro'];
  const GM = mean(TRAIN.map((r) => r[3]));
  const cityEnc = (city, m) => { const r = TRAIN.filter((x) => x[1] === city); if (!r.length) return { n: 0, mean: NaN, te: GM }; const mu = mean(r.map((x) => x[3])); return { n: r.length, mean: mu, te: (r.length * mu + m * GM) / (r.length + m) }; };

  AI.labs['ds-ft-record'] = function (root) {
    const { ctl, o } = setup(root, 'A record becomes a feature vector', 'illustrative: one transform at a time; fit on the 8-row training table',
      360,
      `<label>City <select id="city"><option>Cairo</option><option>Giza</option><option>Alex</option><option value="Luxor">Luxor (never seen in training)</option></select></label>
       <label>Plan <select id="plan"><option>Basic</option><option>Plus</option><option>Pro</option></select></label>
       ${sl('m', 'Target-encoding smoothing m (0 = raw mean)', 0, 10, 1, 2)}
       ${ctlTail(1)}`);
    const c = () => AI.col();
    const cityEl = q(ctl, '#city'), planEl = q(ctl, '#plan'), mIn = q(ctl, '#m'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const names = ['city=Alex', 'city=Cairo', 'city=Giza', 'city=unseen', 'plan_ord', 'city_te'];
    const caps = ['The raw record: two text fields. A model cannot multiply text, so each field must become numbers.',
      'Step 1, one-hot: city becomes one column per known city. Exactly one is 1; a city never seen in training lights the "unseen" column instead.',
      'Step 2, ordinal: the plan has a natural order Basic < Plus < Pro, so one column with 0, 1, 2 keeps that order.',
      'Step 3, target encoding: the city is replaced by its smoothed churn rate learned from the training rows only. An unseen city falls back to the overall rate.'];
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = c(); g.clearRect(0, 0, W, H);
      const city = cityEl.value, plan = planEl.value, m = +mIn.value, ce = cityEnc(city, m);
      q(ctl, '#mv').textContent = m;
      const vals = [0, 1, 2].map((i) => (CITIES[i] === city ? 1 : 0)).concat([CITIES.includes(city) ? 0 : 1, PLANS.indexOf(plan), ce.te]);
      const grp = [0, 0, 0, 0, 1, 2], gcol = [col.blue, col.green, col.purple];
      txt(g, 'raw record', 16, 20, col.muted, '12px sans-serif');
      box(g, 16, 30, 170, 34, col.bg2, col.border); txt(g, `city: ${city}`, 28, 52, col.fg, 'bold 13px sans-serif');
      box(g, 16, 72, 170, 34, col.bg2, col.border); txt(g, `plan: ${plan}`, 28, 94, col.fg, 'bold 13px sans-serif');
      const cw = (W - 32) / 6, cy = 200, ch = 70;
      const alpha = [ease(t - 0), ease(t - 1), ease(t - 2)];   // group k appears while t runs from k to k+1
      for (let i = 0; i < 6; i++) {
        const a = alpha[grp[i]], x = 16 + i * cw;
        if (a > 0) {
          const src = i === 4 ? [186, 89] : [186, 47];
          g.globalAlpha = 0.55 * a; line(g, src[0], src[1], x + cw / 2 - 3, cy, gcol[grp[i]], 1.5); g.globalAlpha = 1;
        }
        box(g, x, cy, cw - 6, ch, null, a > 0 ? col.border : col.muted, a > 0 ? [] : [4, 4]);
        if (a > 0) {
          g.globalAlpha = a;
          const on = i < 4 ? vals[i] === 1 : true;
          box(g, x + 1, cy + 1, cw - 8, ch - 2, on ? gcol[grp[i]] + '33' : null);
          txt(g, i < 4 ? String(vals[i]) : (i === 4 ? String(vals[i]) : vals[i].toFixed(3)), x + (cw - 6) / 2, cy + 46, col.fg, 'bold 16px sans-serif', 'center');
          g.globalAlpha = 1;
        } else txt(g, '?', x + (cw - 6) / 2, cy + 44, col.muted, '16px sans-serif', 'center');
        txt(g, names[i], x + (cw - 6) / 2, cy + 20, a > 0 ? gcol[grp[i]] : col.muted, '11px sans-serif', 'center');
      }
      const step = Math.min(3, Math.ceil(t));
      txt(g, 'feature vector (6 numbers)', 16, cy - 12, col.muted, '12px sans-serif');
      cap.textContent = caps[step];
      txt(g, t >= 3.3 ? 'done: this vector goes to the model' : `step ${step} of 3`, 16, H - 28, col.fg, 'bold 12px sans-serif');
      out.innerHTML = `city ${city}: n = ${ce.n}, raw churn rate ${ce.n ? f(ce.mean) : 'none'}, overall rate ${f(GM)}<br>smoothed encoding = (n·mean + m·overall) / (n + m) = <b>${f(ce.te)}</b>`;
    };
    const S = player(root, ctl, { T: 3.4, rate: 0.4, stepSize: 1, draw });
    cityEl.onchange = planEl.onchange = () => { S.t = 0; S.playing = true; S.set(0, true); };
    mIn.oninput = S.redraw; o.onresize = S.redraw; draw(0);
  };

  /* ====================================================================== 6.2 scaling */
  AI.labs['ds-ft-scale'] = function (root) {
    const { ctl, o } = setup(root, 'Fit the scaler on training data only', 'illustrative: a new value arrives and is scaled with training statistics',
      330,
      `<label>Training values (comma separated)<input type="text" id="tr" value="20, 25, 30, 35, 90" style="width:100%"></label>
       ${sl('x', 'New (test) value x', 0, 120, 1, 60)}
       <label class="row"><input type="checkbox" id="wrong"> Fit the scaler on train + test (wrong)</label>
       ${ctlTail(1)}`);
    const trIn = q(ctl, '#tr'), xIn = q(ctl, '#x'), wr = q(ctl, '#wrong'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const parse = () => { const a = trIn.value.split(/[ ,;]+/).map(Number).filter((v) => Number.isFinite(v)); return a.length >= 2 ? a : null; };
    let S;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const tr = parse(); if (!tr) { out.textContent = 'Enter at least two numbers.'; return; }
      const x = Math.round(t); xIn.value = x; q(ctl, '#xv').textContent = x;
      const m = mean(tr), s = sdev(tr), mn = Math.min(...tr), mx = Math.max(...tr), all = tr.concat([x]);
      const m2 = mean(all), s2 = sdev(all), wrong = wr.checked;
      const z = (x - m) / s, zw = (x - m2) / s2, mm = (x - mn) / (mx - mn);
      const L = 90, R = W - 24, rows = [70, 160, 250];
      const ax = (lo, hi, y, label, ticks) => {
        line(g, L, y, R, y, col.border, 1.5); txt(g, label, 10, y - 22, col.muted, '12px sans-serif');
        ticks.forEach((v) => { const px = L + (R - L) * (v - lo) / (hi - lo); line(g, px, y - 3, px, y + 3, col.muted); txt(g, String(v), px, y + 16, col.muted, '10px sans-serif', 'center'); });
        return (v) => L + (R - L) * (clamp(v, lo, hi) - lo) / (hi - lo);
      };
      const px = ax(0, 120, rows[0], 'raw value', [0, 30, 60, 90, 120]);
      const pz = ax(-3, 5, rows[1], 'standardised z', [-2, 0, 2, 4]);
      const pm = ax(-0.5, 1.6, rows[2], 'min-max (train min..max → 0..1)', [0, 0.5, 1]);
      tr.forEach((v) => { dot(g, px(v), rows[0], 5, col.blue); dot(g, pz((v - m) / s), rows[1], 5, col.blue); dot(g, pm((v - mn) / (mx - mn)), rows[2], 5, col.blue); });
      line(g, pz(0), rows[1] - 10, pz(0), rows[1] + 10, col.fg);
      dot(g, px(x), rows[0], 7, col.orange); dot(g, pz(z), rows[1], 7, col.orange); dot(g, pm(mm), rows[2], 7, col.orange);
      if (wrong) { g.strokeStyle = col.red; g.lineWidth = 2; g.beginPath(); g.arc(pz(zw), rows[1], 9, 0, 7); g.stroke(); g.lineWidth = 1; txt(g, 'z with test peeking', pz(zw), rows[1] - 16, col.red, '11px sans-serif', 'center'); }
      line(g, px(x), rows[0] + 8, pz(z), rows[1] - 8, col.orange, 1, [3, 4]); line(g, pz(z), rows[1] + 8, pm(mm), rows[2] - 8, col.orange, 1, [3, 4]);
      txt(g, `blue = training rows, orange = the new value (${x})`, 10, H - 8, col.muted, '11px sans-serif');
      out.innerHTML = `train: mean ${f(m, 2)}, sd ${f(s, 4)}, min ${mn}, max ${mx}<br>z = (${x} − ${f(m, 2)}) / ${f(s, 4)} = <b>${f(z, 4)}</b> &nbsp; min-max = <b>${f(mm, 4)}</b>`
        + (wrong ? `<br><span style="color:var(--red)">peeking: mean ${f(m2, 2)}, sd ${f(s2, 4)} → z = ${f(zw, 4)}</span>` : '');
      cap.textContent = wrong ? 'Wrong: the scaler has seen the test value, so its mean and sd already contain it and the z moves. Real test data would not be available at fit time.'
        : 'Correct: the scaler learned mean and sd from the blue training rows only, then the same numbers transform whatever arrives. The new value is allowed to land outside the 0..1 range.';
    };
    S = player(root, ctl, { T: 120, rate: 6, stepSize: 10, loop: true, draw });
    xIn.oninput = () => S.set(+xIn.value);
    trIn.oninput = S.redraw; wr.onchange = S.redraw; o.onresize = S.redraw; S.t = 60; draw(60);
  };

  AI.labs['ds-ft-transform'] = function (root) {
    const { ctl, o } = setup(root, 'Skewed numbers and transforms', 'illustrative: 400 simulated order values arriving one by one; type your own transform',
      340,
      `<label>Transform of x<input type="text" id="expr" value="log(x)" style="width:100%"></label><div class="err" id="err" style="color:var(--red);font-size:12px"></div>
       <div class="row" id="presets"></div>${ctlTail(1)}`);
    const rnd = AI.rng(11), X = Array.from({ length: 400 }, () => Math.exp(3.3 + 0.8 * AI.randn(rnd)));
    const st = AI.stats(X), sc = { mean: st.mean, std: st.std, min: st.min, max: st.max };
    let Y = X.slice(), fn = null, S;
    const out = q(ctl, '#out'), cap = q(ctl, '#cap'), BINS = 24;
    const hist = (v, lo, hi, n) => { const h = new Array(BINS).fill(0); for (let i = 0; i < n; i++) { const y = v[i]; if (Number.isFinite(y)) h[clamp(Math.floor((y - lo) / (hi - lo) * BINS), 0, BINS - 1)]++; } return h; };
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const n = Math.max(1, Math.floor(t)), fin = Y.filter(Number.isFinite);
      if (!fin.length) return;
      const lo2 = Math.min(...fin), hi2 = Math.max(...fin), lo1 = st.min, hi1 = st.max;
      const h1 = hist(X, lo1, hi1, n), h2 = hist(Y, lo2, hi2 === lo2 ? lo2 + 1 : hi2, n), top = Math.max(...h1, ...h2, 1);
      [[h1, 'raw x', col.blue, 34, lo1, hi1], [h2, 'transformed', col.green, 190, lo2, hi2]].forEach(([h, name, cc, y0, lo, hi]) => {
        const bw = (W - 40) / BINS, ht = 110;
        txt(g, name, 20, y0 - 8, col.muted); line(g, 20, y0 + ht, W - 20, y0 + ht, col.border);
        h.forEach((c2, i) => box(g, 20 + i * bw + 1, y0 + ht - (c2 / top) * ht, bw - 2, (c2 / top) * ht, cc));
        txt(g, f(lo, 1), 20, y0 + ht + 14, col.muted, '10px sans-serif'); txt(g, f(hi, 1), W - 20, y0 + ht + 14, col.muted, '10px sans-serif', 'right');
      });
      const sk1 = AI.stats(X.slice(0, n)).skew, sk2 = AI.stats(Y.slice(0, n).filter(Number.isFinite)).skew;
      txt(g, `n = ${n}`, W - 20, 26, col.fg, 'bold 12px sans-serif', 'right');
      const skf = (v) => (n >= 3 && Number.isFinite(v) ? f(v, 2) : '–');
      out.innerHTML = `rows seen: <b>${n}</b> &nbsp; skewness raw ${skf(sk1)} → transformed <b>${skf(sk2)}</b> (0 means symmetric)<br>invalid results (NaN, ∞): ${Y.slice(0, n).filter((v) => !Number.isFinite(v)).length}`;
      cap.textContent = 'Each row arrives and falls into a bin. A long right tail (a few huge orders) makes a linear model chase the outliers; a log pulls the tail in.';
    };
    const ei = AI.exprInput(q(ctl, '#expr'), q(ctl, '#err'), {
      allowed: ['x', 'mean', 'std', 'min', 'max'],
      check: (f2) => { f2({ ...sc, x: 5 }); },
      onChange: (f2) => { fn = f2; Y = X.map((x) => f2({ ...sc, x })); if (S) S.redraw(); },
    });
    [['log(x)', 'log(x)'], ['sqrt(x)', 'sqrt(x)'], ['z-score', '(x-mean)/std'], ['min-max', '(x-min)/(max-min)'], ['x² (worse)', 'x^2']].forEach(([k, v]) => {
      const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => ei.set(v); q(ctl, '#presets').appendChild(b);
    });
    S = player(root, ctl, { T: 400, rate: 14, stepSize: 20, draw });
    ei.run(); o.onresize = S.redraw; draw(1);
  };

  /* ====================================================================== 6.3 time */
  AI.labs['ds-ft-cyclic'] = function (root) {
    const { ctl, o } = setup(root, 'Hour of day on a circle', 'illustrative: why 23:00 and 01:00 are neighbours',
      320,
      `${sl('a', 'Fixed hour A', 0, 23, 1, 23)}${ctlTail(1)}`);
    const aIn = q(ctl, '#a'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const pt = (h) => [Math.sin(2 * Math.PI * h / 24), Math.cos(2 * Math.PI * h / 24)];
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const A = +aIn.value, B = t % 24; q(ctl, '#av').textContent = A;
      // left: raw hour line
      const L = 20, R = W * 0.45, ly = 70, px = (h) => L + (R - L) * h / 24;
      line(g, L, ly, R, ly, col.border, 2); txt(g, 'raw hour as one number', L, 30, col.muted);
      for (let h = 0; h <= 24; h += 6) { line(g, px(h), ly - 4, px(h), ly + 4, col.muted); txt(g, String(h), px(h), ly + 18, col.muted, '10px sans-serif', 'center'); }
      dot(g, px(A), ly, 6, col.blue); dot(g, px(B), ly, 6, col.orange);
      line(g, px(A), ly + 34, px(B), ly + 34, col.red, 2);
      const rawd = Math.abs(A - B); txt(g, `raw distance ${f(rawd, 1)} h`, (px(A) + px(B)) / 2, ly + 54, col.red, '12px sans-serif', 'center');
      // right: circle
      const cx = W * 0.72, cy = H / 2 + 6, r = Math.min(W * 0.22, H * 0.36);
      g.strokeStyle = col.border; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.stroke();
      for (let h = 0; h < 24; h++) { const p = pt(h); dot(g, cx + r * p[0], cy - r * p[1], h % 6 === 0 ? 3 : 1.5, col.muted); if (h % 6 === 0) txt(g, String(h), cx + (r + 14) * p[0], cy - (r + 14) * p[1] + 4, col.muted, '10px sans-serif', 'center'); }
      const pa = pt(A), pb = pt(B);
      line(g, cx + r * pa[0], cy - r * pa[1], cx + r * pb[0], cy - r * pb[1], col.green, 2);
      dot(g, cx + r * pa[0], cy - r * pa[1], 6, col.blue); dot(g, cx + r * pb[0], cy - r * pb[1], 6, col.orange);
      txt(g, '(sin, cos) of the hour angle', cx, 22, col.muted, '12px sans-serif', 'center');
      const chord = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]), circ = Math.min(rawd, 24 - rawd);
      out.innerHTML = `A = ${A}:00 → (sin, cos) = (${f(pa[0])}, ${f(pa[1])})<br>B = ${f(B, 1)}:00 → (${f(pb[0])}, ${f(pb[1])})<br>raw gap ${f(rawd, 1)} h · true gap on the clock ${f(circ, 1)} h · chord between points <b>${f(chord)}</b>`;
      cap.textContent = 'B sweeps around the day. On the raw line 23 and 1 look 22 hours apart; on the circle they are close, and the chord (a distance a model can use) is small.';
    };
    const S = player(root, ctl, { T: 24, rate: 1.2, stepSize: 1, loop: true, draw });
    aIn.oninput = S.redraw; o.onresize = S.redraw; draw(1);
  };

  AI.labs['ds-ft-roll'] = function (root) {
    const { ctl, o } = setup(root, 'Lags and rolling means', 'illustrative: a window slides over daily sales; each day it may use only the past',
      340,
      `<label>Daily sales<input type="text" id="ser" value="12, 15, 11, 18, 20, 17, 25, 28, 22, 30" style="width:100%"></label>
       <label>Holiday days<input type="text" id="hol" value="6, 7" style="width:100%"></label>
       ${sl('w', 'Window length w (days)', 1, 5, 1, 3)}
       <label class="row"><input type="checkbox" id="leak"> Include today in the window (leakage)</label>
       ${ctlTail(1)}`);
    const serIn = q(ctl, '#ser'), holIn = q(ctl, '#hol'), wIn = q(ctl, '#w'), leak = q(ctl, '#leak'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const nums = (s) => s.split(/[ ,;]+/).map(Number).filter((v, i, a) => s.trim() && Number.isFinite(v));
    let S;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const y = nums(serIn.value).slice(0, 14), hol = nums(holIn.value), N = y.length, w = +wIn.value; q(ctl, '#wv').textContent = w;
      if (N < 3) { out.textContent = 'Enter at least 3 sales values.'; return; }
      const d = clamp(Math.floor(t) + 1, 1, N), fr = t - Math.floor(t);
      const lk = leak.checked, ymax = Math.max(...y) * 1.15, base = H - 80, ht = base - 50, bw = (W - 40) / N;
      const wl = lk ? d - w + 1 : d - w, wh = lk ? d : d - 1;                    // window = days wl..wh (1-indexed)
      const sm = ease(fr * 3), prev = d > 1 ? d - 1 : d;
      const xl = 20 + bw * (((wl - 1) - (d - prev) * (1 - sm)) ), xr = 20 + bw * (wh + 0 - (d - prev) * (1 - sm));
      if (wl >= 1) box(g, xl, 30, xr - xl, base - 30, (lk ? col.red : col.blue) + '22', lk ? col.red : col.blue, [4, 3]);
      y.forEach((v, i) => {
        const day = i + 1, inW = day >= wl && day <= wh, x = 20 + i * bw, h = (v / ymax) * ht;
        const cc = day === d ? col.orange : inW ? (lk && day === d ? col.red : col.blue) : day > d ? col.border : col.muted;
        box(g, x + 4, base - h, bw - 8, h, cc);
        txt(g, String(v), x + bw / 2, base - h - 4, day > d ? col.muted : col.fg, '11px sans-serif', 'center');
        txt(g, 'd' + day, x + bw / 2, base + 14, col.muted, '10px sans-serif', 'center');
        if (hol.includes(day)) txt(g, '★', x + bw / 2, base + 32, col.yellow, '14px sans-serif', 'center');
      });
      txt(g, 'today = orange bar; grey-outlined bars are the future (not yet known)', 20, 18, col.muted, '11px sans-serif');
      const full = wl >= 1, win = y.slice(Math.max(0, wl - 1), wh), rm = full ? mean(win) : NaN, lag1 = d > 1 ? y[d - 2] : NaN;
      const nxt = hol.filter((h) => h >= d).sort((a, b) => a - b)[0];
      out.innerHTML = `today = day ${d} (sales ${y[d - 1]})<br>lag-1 = ${Number.isFinite(lag1) ? lag1 : 'none yet'} &nbsp; rolling mean of ${w} days = <b>${full ? f(rm) : 'not enough history'}</b>${lk ? ' <span style="color:var(--red)">(contains today)</span>' : ''}<br>is_holiday = ${hol.includes(d) ? 1 : 0}, days to next holiday = ${nxt === undefined ? 'none' : nxt - d}`;
      cap.textContent = lk ? 'The window includes today\'s own sales, which is the value we want to predict. The feature is partly the answer: leakage.'
        : 'The window covers only the w days before today, so the feature could have been computed that morning. Early days have too little history, so the feature is missing.';
    };
    const cfg = { T: 10, rate: 0.5, stepSize: 1, draw };
    S = player(root, ctl, cfg);
    serIn.oninput = () => { cfg.T = Math.max(1, Math.min(14, nums(serIn.value).length)); S.t = Math.min(S.t, cfg.T); S.redraw(); };
    holIn.oninput = wIn.oninput = leak.onchange = S.redraw; o.onresize = S.redraw; draw(0);
  };

  /* ====================================================================== 6.4 aggregates */
  AI.labs['ds-ft-window'] = function (root) {
    const { ctl, o } = setup(root, 'A window sliding over an event log', 'illustrative: one customer\'s purchases; "now" moves forward one day at a time',
      330,
      `<label>Purchases (day:amount, ...)<input type="text" id="ev" value="1:30, 3:45, 4:20, 8:60, 9:25, 12:40, 15:35, 16:50, 19:20" style="width:100%"></label>
       ${sl('w', 'Window length (days)', 1, 14, 1, 7)}
       ${sl('la', 'Look-ahead past "now" (days; 0 = correct)', 0, 5, 1, 0)}
       ${ctlTail(1)}`);
    const evIn = q(ctl, '#ev'), wIn = q(ctl, '#w'), laIn = q(ctl, '#la'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const parse = () => evIn.value.split(/[,;]+/).map((s) => s.split(':').map(Number)).filter((a) => a.length === 2 && a.every(Number.isFinite));
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const ev = parse(), w = +wIn.value, la = +laIn.value, now = Math.floor(t); q(ctl, '#wv').textContent = w; q(ctl, '#lav').textContent = la;
      const D = 20, L = 24, R = W - 24, base = H - 90, ht = base - 60, px = (d) => L + (R - L) * d / D, amax = Math.max(60, ...ev.map((e) => e[1]));
      const lo = now - w, hi = now + la;                                   // window = [lo, hi)
      box(g, px(lo), 40, Math.max(2, px(hi) - px(lo)), base - 40, (la ? col.red : col.blue) + '1f', la ? col.red : col.blue, [4, 3]);
      line(g, L, base, R, base, col.border, 1.5);
      for (let d = 0; d <= D; d += 2) txt(g, String(d), px(d), base + 16, col.muted, '10px sans-serif', 'center');
      ev.forEach(([d, a]) => {
        const inW = d >= lo && d < hi, fut = d >= now, h = (a / amax) * ht;
        const cc = inW && fut ? col.red : inW ? col.blue : d < now ? col.muted : col.border;
        box(g, px(d) - 6, base - h, 12, h, cc); txt(g, String(a), px(d), base - h - 4, d < now ? col.fg : col.muted, '10px sans-serif', 'center');
      });
      line(g, px(now), 30, px(now), base + 4, col.orange, 2); txt(g, `now = day ${now} (events on day ${now} or later are not yet known)`, px(now) + 6, 32, col.orange, '11px sans-serif', now > 12 ? 'right' : 'left');
      const inw = ev.filter((e) => e[0] >= lo && e[0] < hi), past = ev.filter((e) => e[0] < now);
      const cnt = inw.length, sum = inw.reduce((s, e) => s + e[1], 0), rec = past.length ? now - Math.max(...past.map((e) => e[0])) : NaN;
      out.innerHTML = `window [${lo}, ${hi}) days: <b>count ${cnt}</b>, sum ${sum}, mean ${cnt ? f(sum / cnt, 2) : 'none'}<br>recency = ${Number.isFinite(rec) ? rec + ' day(s) since last purchase' : 'no purchase yet'} · frequency so far = ${past.length} purchases`;
      cap.textContent = la ? `Look-ahead ${la}: the window now reaches into days the model could not know at "now" (red bars). The feature looks stronger than it will ever be in production.`
        : 'The window ends exactly at "now". Blue bars are inside it, grey ones are older, outlined bars are the future. Move on and the window slides, dropping old days and adding new ones.';
    };
    const S = player(root, ctl, { T: 20, rate: 0.6, stepSize: 1, draw });
    evIn.oninput = wIn.oninput = laIn.oninput = S.redraw; o.onresize = S.redraw; S.t = 4; draw(4);
  };

  /* ====================================================================== 6.5 leakage */
  const dsFeat = (AI.dsFeat = {});
  // Leak 1: contacts with support, counted in a window that may reach past the prediction time.
  dsFeat.leakTime = (seed, n = 300) => {
    const r = AI.rng(seed), rows = [];
    for (let i = 0; i < n; i++) {
      const y = r() < 0.3 ? 1 : 0, past = [], fut = [];
      for (let d = 0; d < 30; d++) past.push(r() < 0.03 + 0.02 * y ? 1 : 0);
      for (let d = 0; d < 14; d++) fut.push(r() < 0.02 + 0.35 * y ? 1 : 0);
      rows.push({ y, past: past.reduce((a, b) => a + b, 0), fut });
    }
    const half = n / 2, acc = [];
    for (let s = 0; s <= 14; s++) {
      const x = rows.map((rw) => rw.past + rw.fut.slice(0, s).reduce((a, b) => a + b, 0));
      let bestT = 0, bestA = -1;
      for (let th = 0; th <= 8; th++) { let ok = 0; for (let i = 0; i < half; i++) ok += ((x[i] >= th ? 1 : 0) === rows[i].y) ? 1 : 0; if (ok / half > bestA) { bestA = ok / half; bestT = th; } }
      let ok = 0; for (let i = half; i < n; i++) ok += ((x[i] >= bestT ? 1 : 0) === rows[i].y) ? 1 : 0;
      acc.push({ s, th: bestT, test: ok / (n - half), train: bestA });
    }
    const base = 1 - rows.slice(half).reduce((a, b) => a + b.y, 0) / (n - half);
    return { acc, base };
  };
  // Leak 2: choose the best features on ALL rows, then cross-validate (wrong) versus choose inside each fold (right). Labels and features are pure noise.
  dsFeat.selectSim = (seed, P, k, reps = 10, n = 40, folds = 5) => {
    let leakyHits = 0, rightHits = 0, tot = 0;
    for (let rep = 0; rep < reps; rep++) {
      const r = AI.rng(seed * 1000 + rep), rnd = () => AI.randn(r);
      const y = Array.from({ length: n }, (_, i) => (i % 2 ? 1 : -1)); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [y[i], y[j]] = [y[j], y[i]]; }
      const X = Array.from({ length: n }, () => Array.from({ length: P }, rnd));
      const corr = (idx) => Array.from({ length: P }, (_, j) => { let s = 0; for (const i of idx) s += X[i][j] * y[i]; return s / idx.length; });
      const top = (c) => c.map((v, j) => [Math.abs(v), j]).sort((a, b) => b[0] - a[0]).slice(0, k).map((a) => a[1]);
      const all = Array.from({ length: n }, (_, i) => i), leakySel = top(corr(all));
      for (let fo = 0; fo < folds; fo++) {
        const test = all.filter((i) => i % folds === fo), train = all.filter((i) => i % folds !== fo), c = corr(train);
        for (const [sel, kind] of [[leakySel, 0], [top(c), 1]]) {
          const mu = sel.map((j) => mean(train.map((i) => X[i][j])));
          for (const i of test) { let sc = 0; sel.forEach((j, a) => { sc += c[j] * (X[i][j] - mu[a]); }); const ok = (sc >= 0 ? 1 : -1) === y[i] ? 1 : 0; if (kind === 0) leakyHits += ok; else rightHits += ok; }
        }
        tot += test.length;
      }
    }
    return { leaky: leakyHits / tot, right: rightHits / tot };
  };

  AI.labs['ds-ft-leak-time'] = function (root) {
    const { ctl, o } = setup(root, 'A future value creeping into the past', 'illustrative: 300 simulated customers; support contacts counted in a window that may reach past the prediction day',
      340,
      `<div class="row"><button class="btn" id="seed">New random customers</button></div>${ctlTail(1)}`);
    let seed = 7, D = dsFeat.leakTime(seed), S;
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const s = Math.min(14, Math.floor(t)), L = 30, R = W - 20;
      // top: timeline from day -30 to +14
      const px = (d) => L + (R - L) * (d + 30) / 44, ty = 60;
      line(g, L, ty, R, ty, col.border, 2);
      box(g, px(-30), ty - 14, px(0) - px(-30), 28, col.blue + '30', col.blue); box(g, px(0), ty - 14, Math.max(0.1, px(s) - px(0)), 28, col.red + '40', col.red);
      line(g, px(0), ty - 24, px(0), ty + 24, col.orange, 2); txt(g, 'prediction day (all we really know)', px(0) - 6, ty - 28, col.orange, '11px sans-serif', 'right');
      txt(g, 'past 30 days', px(-15), ty + 38, col.blue, '11px sans-serif', 'center'); if (s > 0) txt(g, `+${s} future days`, px(s / 2), ty + 38, col.red, '11px sans-serif', 'center');
      txt(g, 'window used to count contacts', L, 22, col.muted, '12px sans-serif');
      // bottom: accuracy curve
      const cy0 = 130, cy1 = H - 36, ax = (v) => cy1 - (v - 0.5) / 0.5 * (cy1 - cy0);
      line(g, L, cy1, R, cy1, col.border); line(g, L, cy0, L, cy1, col.border);
      txt(g, 'test accuracy', L, cy0 - 8, col.muted, '12px sans-serif'); txt(g, '1.0', L - 4, cy0 + 4, col.muted, '10px sans-serif', 'right'); txt(g, '0.5', L - 4, cy1 + 4, col.muted, '10px sans-serif', 'right');
      line(g, L, ax(D.base), R, ax(D.base), col.muted, 1, [4, 4]); txt(g, `always say "stays": ${f(D.base, 2)}`, R, ax(D.base) - 5, col.muted, '10px sans-serif', 'right');
      const xs = (i) => L + (R - L) * i / 14;
      g.strokeStyle = col.red; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= s; i++) { const y = ax(D.acc[i].test); i ? g.lineTo(xs(i), y) : g.moveTo(xs(i), y); } g.stroke(); g.lineWidth = 1;
      for (let i = 0; i <= s; i++) dot(g, xs(i), ax(D.acc[i].test), 3, col.red);
      for (let i = 0; i <= 14; i += 2) txt(g, String(i), xs(i), cy1 + 16, col.muted, '10px sans-serif', 'center');
      txt(g, 'days of the future included', (L + R) / 2, H - 6, col.muted, '11px sans-serif', 'center');
      const a = D.acc[s];
      out.innerHTML = `future days included: <b>${s}</b> → best threshold: contacts ≥ ${a.th}<br>train accuracy ${f(a.train, 3)} · <b>test accuracy ${f(a.test, 3)}</b> (no future: ${f(D.acc[0].test, 3)})`;
      cap.textContent = s === 0 ? 'With a strictly past window the feature is weak: the model barely beats the always-"stays" baseline. That is the honest number.'
        : 'Churners contact support just before leaving, and those contacts happen after the prediction day. Each extra future day adds label information the model will never have in production.';
    };
    S = player(root, ctl, { T: 14.99, rate: 0.35, stepSize: 1, draw });
    q(ctl, '#seed').onclick = () => { seed++; D = dsFeat.leakTime(seed); S.t = 0; S.playing = true; S.redraw(); };
    o.onresize = S.redraw; draw(0);
  };

  AI.labs['ds-ft-leak-select'] = function (root) {
    const GRID = [5, 10, 20, 50, 100, 200, 500];
    const { ctl, o } = setup(root, 'Leakage through preprocessing', 'illustrative: 40 rows, every feature is pure noise; average of 10 simulated datasets',
      320,
      `${sl('k', 'Features kept by the selector', 1, 10, 1, 5)}
       <div class="row"><button class="btn" id="seed">New random datasets</button></div>${ctlTail(1)}`);
    let seed = 1, R = null, S;
    const out = q(ctl, '#out'), cap = q(ctl, '#cap'), kIn = q(ctl, '#k');
    const compute = () => { q(ctl, '#kv').textContent = kIn.value; R = GRID.map((p) => dsFeat.selectSim(seed, p, +kIn.value)); };
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const i = Math.min(GRID.length - 1, Math.floor(t)), L = 44, Rr = W - 24, y0 = 30, y1 = H - 52, ax = (v) => y1 - (v - 0.3) / 0.7 * (y1 - y0), xs = (j) => L + (Rr - L) * j / (GRID.length - 1);
      line(g, L, y1, Rr, y1, col.border); line(g, L, y0, L, y1, col.border);
      [0.3, 0.5, 0.7, 1.0].forEach((v) => txt(g, v.toFixed(1), L - 6, ax(v) + 4, col.muted, '10px sans-serif', 'right'));
      line(g, L, ax(0.5), Rr, ax(0.5), col.muted, 1, [4, 4]); txt(g, 'chance 0.5 (labels are random)', Rr, ax(0.5) + 14, col.muted, '10px sans-serif', 'right');
      GRID.forEach((p, j) => txt(g, String(p), xs(j), y1 + 16, col.muted, '10px sans-serif', 'center'));
      txt(g, 'number of noise features available', (L + Rr) / 2, H - 8, col.muted, '11px sans-serif', 'center');
      [['leaky', col.red, 'select on all rows, then CV'], ['right', col.green, 'select inside each fold']].forEach(([key, cc, lab], n) => {
        g.strokeStyle = cc; g.lineWidth = 2; g.beginPath();
        for (let j = 0; j <= i; j++) { const y = ax(R[j][key]); j ? g.lineTo(xs(j), y) : g.moveTo(xs(j), y); } g.stroke(); g.lineWidth = 1;
        for (let j = 0; j <= i; j++) dot(g, xs(j), ax(R[j][key]), 3.5, cc);
        txt(g, lab, L + 8, y0 + 4 + n * 16, cc, '12px sans-serif');
      });
      out.innerHTML = `${GRID[i]} noise features, keep ${kIn.value}:<br>leaky CV accuracy <b style="color:var(--red)">${f(R[i].leaky, 3)}</b> · correct CV accuracy <b style="color:var(--green)">${f(R[i].right, 3)}</b>`;
      cap.textContent = 'There is nothing to learn here, so an honest score sits near 0.5. Choosing features with the held-out rows included lets the selector peek at them, and the more candidates it can choose from, the better it looks.';
    };
    S = player(root, ctl, { T: GRID.length - 0.01, rate: 0.4, stepSize: 1, draw, startPlaying: true });
    kIn.oninput = () => { compute(); S.redraw(); };
    q(ctl, '#seed').onclick = () => { seed++; compute(); S.t = 0; S.playing = true; S.redraw(); };
    compute(); o.onresize = S.redraw; draw(0);
  };

  /* ====================================================================== 6.6 pipelines */
  AI.labs['ds-ft-pipe'] = function (root) {
    const { ctl, o } = setup(root, 'Fit once, transform everywhere', 'illustrative: a three-step pipeline fitted on 5 training rows, then applied to a new record',
      360,
      `${sl('age', 'New record: age', 15, 80, 1, 60)}
       <label class="row"><input type="checkbox" id="miss"> age is missing</label>
       <label>City <select id="city"><option>Cairo</option><option>Giza</option><option>Alex</option><option value="Luxor">Luxor (unseen)</option></select></label>
       <label class="row"><input type="checkbox" id="wrong"> Refit on the serving data (wrong)</label>
       ${ctlTail(1)}`);
    const AGE = [22, 35, 41, 29, 48], CT = ['Cairo', 'Giza', 'Cairo', 'Alex', 'Giza'], MU = mean(AGE), SD = sdev(AGE), CATS = ['Alex', 'Cairo', 'Giza'];
    const ageIn = q(ctl, '#age'), miss = q(ctl, '#miss'), cityEl = q(ctl, '#city'), wr = q(ctl, '#wrong'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h, col = AI.col(); g.clearRect(0, 0, W, H);
      const age = +ageIn.value, isM = miss.checked, city = cityEl.value, wrong = wr.checked; q(ctl, '#agev').textContent = isM ? '–' : age;
      // fit panel
      const a0 = ease(t);
      box(g, 14, 14, W - 28, 92, null, col.border); txt(g, 'FIT: runs once, on training rows only', 24, 32, col.purple, 'bold 12px sans-serif');
      txt(g, `training ages ${AGE.join(', ')}   cities ${CT.join(', ')}`, 24, 50, col.muted, '11px sans-serif');
      g.globalAlpha = a0;
      txt(g, `stored: age mean = ${f(MU, 1)} · age sd = ${f(SD, 4)} · categories = [${CATS.join(', ')}]`, 24, 76, col.fg, 'bold 12px sans-serif');
      txt(g, 'these numbers are the fitted state of the pipeline', 24, 94, col.muted, '11px sans-serif'); g.globalAlpha = 1;
      // transform panel
      const stg = ['1. impute age', '2. standardise age', '3. one-hot city', 'vector'], bw = (W - 28 - 3 * 14) / 4, y = 150, bh = 110;
      txt(g, 'TRANSFORM: every new record, the same stored numbers', 14, 132, col.green, 'bold 12px sans-serif');
      const mu = wrong ? (isM ? NaN : age) : MU, sd = wrong ? 0 : SD;
      const aImp = isM ? mu : age, z = (aImp - mu) / sd, oh = CATS.map((c) => (c === city ? 1 : 0)), unk = CATS.includes(city) ? 0 : 1;
      const lines = [
        [`raw age: ${isM ? 'missing' : age}`, isM ? `fill with ${f(mu, 1)}` : 'nothing to fill', `→ ${f(aImp, 1)}`],
        [`(${f(aImp, 1)} − ${f(mu, 1)})`, `/ ${f(sd, 4)}`, `→ ${f(z, 4)}`],
        [`city ${city}`, `cols ${CATS.join(' ')}`, `→ ${oh.join(' ')}${unk ? '  (unseen: all 0)' : ''}`],
        [`[ ${f(z, 3)}, ${oh.join(', ')} ]`, '4 numbers', 'to the model'],
      ];
      for (let i = 0; i < 4; i++) {
        const a = ease(t - (i + 1) * 1), x = 14 + i * (bw + 14);
        box(g, x, y, bw, bh, i === 3 ? col.green + '22' : null, a > 0 ? col.border : col.muted, a > 0 ? [] : [4, 4]);
        txt(g, stg[i], x + 8, y + 18, a > 0 ? (i === 3 ? col.green : col.blue) : col.muted, 'bold 11px sans-serif');
        g.globalAlpha = a; lines[i].forEach((s, k) => txt(g, s, x + 8, y + 42 + k * 20, k === 2 ? col.fg : col.muted, k === 2 ? 'bold 11px sans-serif' : '11px sans-serif')); g.globalAlpha = 1;
        if (i < 3) txt(g, '→', x + bw + 7, y + bh / 2, col.muted, '14px sans-serif', 'center');
      }
      txt(g, wrong ? 'refit on a single serving row: mean = its own age, sd = 0, so z is 0 / 0 ...' : 'the stored mean and sd are reused unchanged, so training and serving agree', 14, H - 32, wrong ? col.red : col.muted, '12px sans-serif');
      out.innerHTML = `training z-score of age 60 would be ${f((60 - MU) / SD, 4)}<br>this record: z = <b${wrong ? ' style="color:var(--red)"' : ''}>${f(z, 4)}</b>, one-hot = [${oh.join(', ')}]${unk ? ', unseen city' : ''}`;
      cap.textContent = t < 1 ? 'Fit: the pipeline learns its numbers from the training rows (the only time it looks at data).'
        : wrong ? 'Wrong: refitting at serving time changes the numbers the model was trained with. The model gets inputs from a different scale than it learned.'
        : 'Transform: each step applies stored numbers to the record. No statistics are recomputed, so a record gets the same features in training and in production.';
    };
    const S = player(root, ctl, { T: 5.2, rate: 0.35, stepSize: 1, draw });
    ageIn.oninput = () => { miss.checked = false; S.redraw(); }; miss.onchange = wr.onchange = cityEl.onchange = S.redraw;
    o.onresize = S.redraw; draw(0);
  };
})();
