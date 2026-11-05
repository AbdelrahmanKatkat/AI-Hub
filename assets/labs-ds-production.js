/* Data Science Module 9 labs: reproducible reports, subgroup evaluation, drift monitoring, fairness metrics, k-anonymity, scoping.
   All simulations are ILLUSTRATIVE (seeded synthetic data) and are not legal or compliance advice.
   Every animation runs on dt, starts slowly, and has speed / pause / step / reset controls. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const speedCtl = `${sl('spd', 'Speed', 0.25, 3, 0.25, 1)}`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const wilson = (k, n) => { if (!n) return [0, 1]; const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [c - h, c + h]; };
  const rowBtns = (extra) => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button>${extra || ''}</div>`;
  const themeRedraw = (fn) => window.addEventListener('themechange', () => setTimeout(fn, 0));

  /* ---------------------------------------------------------------- 1. Re-run from raw */
  AI.labs['ds-pr-repro'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Re-run from raw data', 'a published number must come out again when the script runs on the raw file (illustrative)',
      `<canvas style="height:300px"></canvas>`,
      `<label>Raw value in row 5 (published file has 150) <input type="number" id="v5" value="150" step="5"></label>
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="rule" checked> Script drops refunds (negative amounts)</label>
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="patch"> Hand-type the number in the report (105)</label>
       ${speedCtl}
       <div class="row"><button class="btn primary" id="run">▶ Re-run from raw</button><button class="btn" id="pause">Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div>
       <p class="caption">Published report: mean order value 102.50 from a file with fingerprint shown above. A packet walks raw → clean → summarise → report; then the new number is compared with the published one.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const BASE = [120, 80, -30, 95, 150, -45, 110, 60];
    const fp = (a) => { let h = 0x811c9dc5; const s = a.join(','); for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h.toString(16).padStart(8, '0'); };
    const PUB = { fp: fp(BASE), mean: 102.5 };
    let p = 0, running = true, res = null, dirty = true;
    const names = ['raw file', 'clean', 'summarise', 'report'];
    const compute = () => {
      const raw = BASE.slice(); raw[4] = +q(ctl, '#v5').value || 0;
      const rule = q(ctl, '#rule').checked, patch = q(ctl, '#patch').checked;
      const cl = rule ? raw.filter((x) => x > 0) : raw;
      const mean = cl.reduce((s, x) => s + x, 0) / cl.length;
      res = { fp: fp(raw), n: raw.length, nc: cl.length, mean, rep: patch ? 105 : mean, patch, rule };
    };
    const start = () => { compute(); p = 0; running = true; dirty = true; };
    const verdict = () => {
      if (!res) return '';
      const ok = Math.abs(res.rep - PUB.mean) < 1e-9;
      const why = [];
      if (res.fp !== PUB.fp) why.push('raw data fingerprint differs (' + res.fp + ' vs ' + PUB.fp + '): the file changed since publication');
      if (!res.rule) why.push('the cleaning rule differs: refunds were kept, so the mean is ' + res.mean.toFixed(2));
      if (res.patch) why.push('the report number was typed by hand; the script gives ' + res.mean.toFixed(2));
      return ok ? 'REPRODUCED: ' + res.rep.toFixed(2) + ' equals the published 102.50.' : 'NOT REPRODUCED: ' + res.rep.toFixed(2) + ' vs published 102.50.\n- ' + why.join('\n- ');
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const bw = Math.min(150, (W - 60) / 4 - 16), gap = (W - 4 * bw) / 5, y0 = 70, bh = 120;
      for (let i = 0; i < 4; i++) {
        const x = gap + i * (bw + gap), act = p >= i && p < i + 1 && running, done = p >= i + 1 || (p >= 4);
        g.fillStyle = act ? c.bg3 : c.bg2; g.strokeStyle = act ? c.blue : done ? c.green : c.border; g.lineWidth = act ? 2 : 1;
        g.beginPath(); g.roundRect(x, y0, bw, bh, 10); g.fill(); g.stroke(); g.lineWidth = 1;
        txt(g, names[i], x + bw / 2, y0 + 22, c.fg, 'bold 13px sans-serif', 'center');
        const show = (res && p >= i + 1) || (res && p >= 4);
        const L = [];
        if (i === 0) L.push('8 rows', res ? 'fp ' + res.fp : '');
        if (i === 1 && show) L.push(res.rule ? 'drop refunds' : 'keep all rows', res.nc + ' rows kept');
        if (i === 2 && show) L.push('mean of amount', res.mean.toFixed(2));
        if (i === 3 && show) L.push(res.patch ? 'typed by hand:' : 'from script:', res.rep.toFixed(2));
        L.forEach((s, k) => txt(g, s, x + bw / 2, y0 + 50 + k * 22, c.muted, '12px ui-monospace,monospace', 'center'));
        if (i < 3) { g.strokeStyle = c.border; g.beginPath(); g.moveTo(x + bw + 4, y0 + bh / 2); g.lineTo(x + bw + gap - 4, y0 + bh / 2); g.stroke(); }
      }
      if (running && p < 4) { const px = gap + p * (bw + gap) + bw / 2 + (p % 1 < 0.5 ? 0 : 0); g.fillStyle = c.orange; g.beginPath(); g.arc(px, y0 - 22, 8, 0, 7); g.fill(); txt(g, 'running ' + names[Math.floor(p)], px, y0 - 38, c.orange, '12px sans-serif', 'center'); }
      const done = res && p >= 4, ok = done && Math.abs(res.rep - PUB.mean) < 1e-9;
      txt(g, done ? (ok ? '✓ matches the published number' : '✗ does not match the published number') : (res ? 'running…' : 'press Re-run'), W / 2, y0 + bh + 44, done ? (ok ? c.green : c.red) : c.muted, 'bold 14px sans-serif', 'center');
      txt(g, 'published: mean 102.50, fingerprint ' + PUB.fp, W / 2, y0 + bh + 68, c.muted, '12px sans-serif', 'center');
      out.textContent = p >= 4 ? verdict() : (res ? 'Packet at stage ' + (Math.floor(p) + 1) + ' of 4…' : '');
    };
    q(ctl, '#run').onclick = start;
    q(ctl, '#pause').onclick = (e) => { running = !running; e.target.textContent = running ? 'Pause' : 'Resume'; dirty = true; };
    q(ctl, '#step').onclick = () => { if (!res) compute(); running = false; p = Math.min(4, Math.floor(p) + 1); q(ctl, '#pause').textContent = 'Resume'; dirty = true; };
    q(ctl, '#reset').onclick = () => { q(ctl, '#v5').value = 150; q(ctl, '#rule').checked = true; q(ctl, '#patch').checked = false; q(ctl, '#pause').textContent = 'Pause'; start(); };
    const spd = q(ctl, '#spd'), sv = q(ctl, '#spdv'); spd.oninput = () => (sv.textContent = spd.value + '×'); sv.textContent = '1×';
    o.onresize = draw; themeRedraw(draw);
    AI.loop(root, (dt) => {
      if (running && res && p < 4) { p = Math.min(4, p + dt * 0.4 * +spd.value); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    start();
  };

  /* ---------------------------------------------------------------- 2. Subgroup evaluation */
  AI.labs['ds-pr-subgroup'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Evaluation by subgroup', 'test cases arrive one by one; overall accuracy hides the weaker group (illustrative)',
      `<canvas style="height:300px"></canvas>`,
      `${sl('aa', 'True accuracy, group A', 0.5, 1, 0.01, 0.9)}${sl('ab', 'True accuracy, group B', 0.5, 1, 0.01, 0.7)}${sl('sh', 'Share of test cases from group B', 0.05, 0.5, 0.01, 0.25)}${speedCtl}
       ${rowBtns()}
       <div class="readout" id="out"></div>
       <p class="caption">Each case is drawn from group B with the chosen share and is correct with that group's true accuracy (seeded random numbers). Whiskers are 95% Wilson intervals; they shrink as cases arrive, and the smaller group's stays wide for longer.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const aa = q(ctl, '#aa'), ab = q(ctl, '#ab'), sh = q(ctl, '#sh'), spd = q(ctl, '#spd');
    const N = 400; let rnd, n, kA, kB, nA, nB, playing = true, acc = 0, dirty = true;
    const reset = () => { rnd = AI.rng(11); n = kA = kB = nA = nB = 0; acc = 0; dirty = true; };
    const add = () => {
      if (n >= N) return; n++;
      if (rnd() < +sh.value) { nB++; if (rnd() < +ab.value) kB++; } else { nA++; if (rnd() < +aa.value) kA++; }
    };
    const labels = () => { q(ctl, '#aav').textContent = (+aa.value).toFixed(2); q(ctl, '#abv').textContent = (+ab.value).toFixed(2); q(ctl, '#shv').textContent = Math.round(+sh.value * 100) + '%'; q(ctl, '#spdv').textContent = spd.value + '×'; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); labels();
      const x0 = 150, x1 = W - 24, X = (v) => x0 + (x1 - x0) * clamp((v - 0.3) / 0.7, 0, 1);
      g.strokeStyle = c.border; g.fillStyle = c.muted; g.font = '11px sans-serif'; g.textAlign = 'center';
      for (let v = 0.3; v <= 1.001; v += 0.1) { g.beginPath(); g.moveTo(X(v), 30); g.lineTo(X(v), H - 40); g.stroke(); g.fillText(v.toFixed(1), X(v), H - 24); }
      txt(g, 'accuracy', x1, H - 6, c.muted, '11px sans-serif', 'right');
      const rows = [['group A', kA, nA, c.blue, +aa.value], ['group B', kB, nB, c.orange, +ab.value], ['overall', kA + kB, n, c.purple, null]];
      rows.forEach(([nm, k, m, col, tr], i) => {
        const y = 62 + i * 68;
        txt(g, nm, 12, y + 4, c.fg, 'bold 13px sans-serif'); txt(g, 'n = ' + m, 12, y + 22, c.muted, '11px sans-serif');
        if (tr !== null) { g.strokeStyle = c.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(tr), y - 20); g.lineTo(X(tr), y + 20); g.stroke(); g.setLineDash([]); }
        if (m) {
          const [lo, hi] = wilson(k, m), p = k / m;
          g.strokeStyle = col; g.lineWidth = 3; g.beginPath(); g.moveTo(X(lo), y); g.lineTo(X(hi), y); g.stroke(); g.lineWidth = 1;
          g.fillStyle = col; g.beginPath(); g.arc(X(p), y, 7, 0, 7); g.fill();
          txt(g, p.toFixed(3), X(p), y - 14, c.fg, '12px ui-monospace,monospace', 'center');
        }
      });
      txt(g, 'dashed line = true accuracy of the group', x1, 18, c.muted, '11px sans-serif', 'right');
      const f = (k, m) => { if (!m) return 'no cases yet'; const [lo, hi] = wilson(k, m); return (k / m).toFixed(3) + ' (95% CI ' + lo.toFixed(2) + ' to ' + hi.toFixed(2) + ', n=' + m + ')'; };
      out.textContent = 'MODEL CARD, evaluation by subgroup\nA: ' + f(kA, nA) + '\nB: ' + f(kB, nB) + '\nOverall: ' + f(kA + kB, n) + '\n' + (n >= N ? 'Test set complete (' + N + ' cases).' : 'Cases so far: ' + n + ' of ' + N);
    };
    [aa, ab, sh, spd].forEach((e) => (e.oninput = () => { if (e !== spd) reset(); dirty = true; }));
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; add(); dirty = true; };
    q(ctl, '#reset').onclick = reset;
    o.onresize = draw; themeRedraw(draw);
    const tick = AI.ticker();
    AI.loop(root, (dt) => {
      if (playing && n < N) { const m = tick(dt, 6 * +spd.value); for (let i = 0; i < m; i++) add(); if (m) dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    reset(); draw();
  };

  /* ---------------------------------------------------------------- 3. Drift monitoring */
  AI.labs['ds-pr-drift'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Drift monitor: PSI, accuracy and an alert', 'live inputs slide away from the training distribution one day at a time (illustrative simulation)',
      `<canvas style="height:470px"></canvas>`,
      `${sl('rate', 'Drift: shift of the input mean per day', 0, 0.08, 0.005, 0.03)}${sl('thr', 'PSI alert threshold', 0.05, 0.5, 0.01, 0.2)}${speedCtl}
       ${rowBtns()}
       <div class="readout" id="out"></div>
       <p class="caption">One feature x. Training: x ~ N(0,1). The model says "yes" when x &gt; 0; in the training range that was right, but the true rule is "yes only for 0 &lt; x &lt; 2.5", with 8% label noise. Each day 300 new inputs arrive. PSI uses 10 bins cut at the training deciles and needs no labels; accuracy needs labels, which usually arrive late.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const rate = q(ctl, '#rate'), thr = q(ctl, '#thr'), spd = q(ctl, '#spd');
    const EDGES = [-1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816];
    const DAYS = 80, PER = 300;
    let rnd, mu, day, psi, accs, mus, hist, playing = true, acc = 0, dirty = true;
    const reset = () => { rnd = AI.rng(5); mu = 0; day = 0; psi = []; accs = []; mus = []; hist = new Array(48).fill(0); acc = 0; dirty = true; };
    const binOf = (x) => { let b = 0; while (b < 9 && x > EDGES[b]) b++; return b; };
    const nextDay = () => {
      if (day >= DAYS) return;
      mu += +rate.value; if (day === 0) mu = 0;
      const cnt = new Array(10).fill(0); hist = new Array(48).fill(0); let ok = 0;
      for (let i = 0; i < PER; i++) {
        const x = mu + AI.randn(rnd); cnt[binOf(x)]++;
        let y = x > 0 && x < 2.5 ? 1 : 0; if (rnd() < 0.08) y = 1 - y;
        if ((x > 0 ? 1 : 0) === y) ok++;
        const hb = Math.floor((x + 4) / 0.25); if (hb >= 0 && hb < 48) hist[hb]++;
      }
      let s = 0; cnt.forEach((k) => { const a = Math.max(k / PER, 1e-4); s += (a - 0.1) * Math.log(a / 0.1); });
      psi.push(s); accs.push(ok / PER); mus.push(mu); day++; dirty = true;
    };
    const alertDay = () => { for (let i = 0; i < psi.length; i++) if (psi[i] > +thr.value) return i + 1; return 0; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#ratev').textContent = (+rate.value).toFixed(3); q(ctl, '#thrv').textContent = (+thr.value).toFixed(2); q(ctl, '#spdv').textContent = spd.value + '×';
      const L = 54, R = W - 14, ad = alertDay();
      // top: histogram vs training density
      const t0 = 28, t1 = 170, X = (x) => L + ((x + 4) / 12) * (R - L);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t1); g.lineTo(R, t1); g.stroke();
      const bw = (R - L) / 48, mx = 0.4 * PER * 0.25 / 1;
      hist.forEach((v, i) => { const h = (v / (PER * 0.25 * 0.4)) * (t1 - t0 - 14); g.fillStyle = c.orange; g.globalAlpha = 0.75; g.fillRect(L + i * bw + 1, t1 - h, bw - 2, h); g.globalAlpha = 1; });
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 120; i++) { const x = -4 + 12 * i / 120, d = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), y = t1 - (d / 0.4) * (t1 - t0 - 14); i ? g.lineTo(X(x), y) : g.moveTo(X(x), y); }
      g.stroke(); g.lineWidth = 1;
      g.setLineDash([4, 4]); g.strokeStyle = c.muted; [0, 2.5].forEach((x) => { g.beginPath(); g.moveTo(X(x), t0); g.lineTo(X(x), t1); g.stroke(); }); g.setLineDash([]);
      txt(g, 'model: yes if x > 0', X(0) + 4, t0 + 8, c.muted, '11px sans-serif'); txt(g, 'truth: yes only if x < 2.5', X(2.5) + 4, t0 + 8, c.muted, '11px sans-serif');
      txt(g, 'blue: training distribution · orange: today\'s inputs (day ' + day + ', mean ' + (mus.length ? mus[mus.length - 1].toFixed(2) : '0.00') + ')', L, 16, c.fg, '12px sans-serif');
      [-4, -2, 0, 2, 4, 6, 8].forEach((x) => txt(g, String(x), X(x), t1 + 14, c.muted, '10px sans-serif', 'center'));
      // charts below
      const DX = (d) => L + (d / DAYS) * (R - L);
      const chart = (y0, y1, vals, lo, hi, col, label, line) => {
        g.strokeStyle = c.border; g.strokeRect(L, y0, R - L, y1 - y0);
        const Y = (v) => y1 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (y1 - y0);
        if (line !== undefined) { g.setLineDash([5, 4]); g.strokeStyle = c.red; g.beginPath(); g.moveTo(L, Y(line)); g.lineTo(R, Y(line)); g.stroke(); g.setLineDash([]); }
        g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); vals.forEach((v, i) => (i ? g.lineTo(DX(i + 1), Y(v)) : g.moveTo(DX(i + 1), Y(v)))); g.stroke(); g.lineWidth = 1;
        txt(g, label, L + 6, y0 + 14, col, 'bold 12px sans-serif'); txt(g, hi.toFixed(1), L - 6, y0 + 8, c.muted, '10px sans-serif', 'right'); txt(g, lo.toFixed(1), L - 6, y1, c.muted, '10px sans-serif', 'right');
        if (ad) { g.strokeStyle = c.red; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(DX(ad), y0); g.lineTo(DX(ad), y1); g.stroke(); g.globalAlpha = 1; }
      };
      const pmax = Math.max(0.5, ...psi) * 1.05;
      chart(206, 316, psi, 0, pmax, c.purple, 'PSI (needs no labels)', +thr.value);
      chart(340, 440, accs, 0.4, 1, c.green, 'accuracy (needs labels)');
      txt(g, 'day', R, 462, c.muted, '10px sans-serif', 'right');
      if (ad) { g.fillStyle = c.red; g.beginPath(); g.roundRect(R - 250, 30, 250, 22, 6); g.fill(); txt(g, 'ALERT day ' + ad + ': PSI ' + psi[ad - 1].toFixed(2) + ' > ' + (+thr.value).toFixed(2), R - 125, 45, '#fff', 'bold 12px sans-serif', 'center'); }
      const base = accs.length ? accs[0] : 0;
      out.textContent = 'day ' + day + ' of ' + DAYS + '\nPSI now: ' + (psi.length ? psi[psi.length - 1].toFixed(3) : '-') + '\naccuracy now: ' + (accs.length ? accs[accs.length - 1].toFixed(3) : '-') + (accs.length ? ' (day 1: ' + base.toFixed(3) + ')' : '') + '\n' + (ad ? 'Alert raised on day ' + ad + (accs[ad - 1] > base - 0.03 ? '\nwhen accuracy was still near its start.' : '.') : 'No alert yet.');
    };
    [rate, thr, spd].forEach((e) => (e.oninput = () => (dirty = true)));
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; nextDay(); };
    q(ctl, '#reset').onclick = () => { reset(); nextDay(); };
    o.onresize = draw; themeRedraw(draw);
    AI.loop(root, (dt) => {
      if (playing && day < DAYS) { acc += dt * 1.0 * +spd.value; while (acc >= 1) { acc -= 1; nextDay(); } }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    reset(); nextDay(); draw();
  };

  /* ---------------------------------------------------------------- 4. Fairness metrics */
  AI.labs['ds-pr-fair'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fairness metrics and a threshold', 'two groups, one score: move the threshold and watch the gaps (illustrative, not legal advice)',
      `<canvas style="height:500px"></canvas>`,
      `${sl('t', 'Threshold (group A, and B if linked)', 0.1, 0.9, 0.01, 0.5)}
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="link" checked> Same threshold for both groups</label>
       ${sl('tb', 'Threshold for group B (when unlinked)', 0.1, 0.9, 0.01, 0.5)}
       ${sl('sh', 'Score shift applied to group B', -0.15, 0.15, 0.01, 0)}
       ${speedCtl}
       ${rowBtns()}
       <div class="readout" id="out"></div>
       <p class="caption">Synthetic scores, seeded. Green squares are people who truly have the outcome (for example repay a loan), orange do not. Group A has base rate about 55%, group B about 30%. Play sweeps the threshold slowly back and forth.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const t = q(ctl, '#t'), tb = q(ctl, '#tb'), sh = q(ctl, '#sh'), link = q(ctl, '#link'), spd = q(ctl, '#spd');
    const rnd = AI.rng(23), mk = (n, base) => Array.from({ length: n }, () => { const y = rnd() < base ? 1 : 0; return { y, s: clamp(0.5 + (y ? 0.14 : -0.14) + 0.17 * AI.randn(rnd), 0.01, 0.99) }; });
    const DA = mk(150, 0.55), DB = mk(150, 0.3);
    let playing = false, dir = 1, dirty = true;
    const metrics = (D, th, shift) => {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      D.forEach((r) => { const pred = clamp(r.s + shift, 0, 1) >= th; if (r.y) (pred ? tp++ : fn++); else (pred ? fp++ : tn++); });
      const n = D.length; return { sel: (tp + fp) / n, tpr: tp / Math.max(1, tp + fn), fpr: fp / Math.max(1, fp + tn), ppv: tp + fp ? tp / (tp + fp) : 0, base: (tp + fn) / n };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const th = +t.value, thB = link.checked ? th : +tb.value, shift = +sh.value;
      q(ctl, '#tv').textContent = th.toFixed(2); q(ctl, '#tbv').textContent = thB.toFixed(2); q(ctl, '#shv').textContent = shift.toFixed(2); q(ctl, '#spdv').textContent = spd.value + '×'; tb.disabled = link.checked;
      const L = 60, R = W - 20, X = (s) => L + s * (R - L), bins = 25, bw = (R - L) / bins;
      const fitSq = (D, sf) => { const c2 = new Array(bins).fill(0); D.forEach((r) => c2[Math.floor(clamp(r.s + sf, 0, 0.999) * bins)]++); return Math.max(2, Math.min(bw - 1, 7, 88 / Math.max(...c2) - 1)); };
      [[DA, th, 0, 'group A', 24], [DB, thB, shift, 'group B', 128]].forEach(([D, tt, sf, nm, y0]) => {
        const base = y0 + 90, sq = fitSq(D, sf), cnt = new Array(bins).fill(0);
        txt(g, nm, 8, y0 + 50, c.fg, 'bold 12px sans-serif');
        g.fillStyle = c.bg2; g.fillRect(X(tt), y0, R - X(tt), 92);
        D.slice().sort((a, b) => a.s - b.s).forEach((r) => { const s = clamp(r.s + sf, 0, 0.999), b = Math.floor(s * bins), k = cnt[b]++; g.fillStyle = r.y ? c.green : c.orange; g.fillRect(L + b * bw + (bw - sq) / 2, base - (k + 1) * (sq + 1), sq, sq); });
        g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(X(tt), y0 - 2); g.lineTo(X(tt), base + 2); g.stroke(); g.lineWidth = 1;
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, base + 2); g.lineTo(R, base + 2); g.stroke();
      });
      txt(g, 'score → (shaded: predicted positive, i.e. approved)', L, 254, c.muted, '11px sans-serif');
      [0, 0.5, 1].forEach((s) => txt(g, s.toFixed(1), X(s), 238, c.muted, '10px sans-serif', 'center'));
      const mA = metrics(DA, th, 0), mB = metrics(DB, thB, shift);
      const rows = [['selection rate', 'sel', 'demographic parity gap'], ['true-positive rate', 'tpr', 'equal opportunity gap'], ['false-positive rate', 'fpr', 'FPR gap'], ['precision (PPV)', 'ppv', 'PPV gap']];
      const bx = 150, bwid = R - bx - 110;
      rows.forEach(([nm, key, gn], i) => {
        const y = 272 + i * 54; txt(g, nm, 8, y + 14, c.fg, '12px sans-serif');
        [[mA, c.blue, 0], [mB, c.purple, 1]].forEach(([m, col, j]) => { g.fillStyle = col; g.fillRect(bx, y + j * 20, m[key] * bwid, 16); txt(g, m[key].toFixed(3), bx + m[key] * bwid + 6, y + j * 20 + 12, c.fg, '11px ui-monospace,monospace'); });
        const gap = Math.abs(mA[key] - mB[key]);
        txt(g, gn, R, y + 12, c.muted, '10px sans-serif', 'right'); txt(g, gap.toFixed(3), R, y + 30, gap < 0.02 ? c.green : c.red, 'bold 13px ui-monospace,monospace', 'right');
      });
      txt(g, 'blue = group A, purple = group B', bx, H - 6, c.muted, '11px sans-serif');
      const nz = (a) => a.toFixed(3);
      out.textContent = 'base rate A ' + nz(mA.base) + ', B ' + nz(mB.base) + '\nDP gap |sel A - sel B| = ' + nz(Math.abs(mA.sel - mB.sel)) + '\nEO gap |TPR A - TPR B| = ' + nz(Math.abs(mA.tpr - mB.tpr)) + '\nFPR gap = ' + nz(Math.abs(mA.fpr - mB.fpr)) + ', PPV gap = ' + nz(Math.abs(mA.ppv - mB.ppv)) + '\nBase rates differ and the score is imperfect:\nnot every gap can be zero at once.';
    };
    [t, tb, sh, link, spd].forEach((e) => (e.oninput = e.onchange = () => (dirty = true)));
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#play').textContent = '▶ Play';
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; t.value = clamp(+t.value + 0.05, 0.1, 0.9).toFixed(2); dirty = true; };
    q(ctl, '#reset').onclick = () => { t.value = 0.5; tb.value = 0.5; sh.value = 0; link.checked = true; playing = false; q(ctl, '#play').textContent = '▶ Play'; dirty = true; };
    o.onresize = draw; themeRedraw(draw);
    AI.loop(root, (dt) => {
      if (playing) { let v = +t.value + dir * dt * 0.05 * +spd.value; if (v >= 0.9) { v = 0.9; dir = -1; } if (v <= 0.1) { v = 0.1; dir = 1; } t.value = v.toFixed(3); dirty = true; }
      if (dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    draw();
  };

  /* ---------------------------------------------------------------- 5. k-anonymity */
  AI.labs['ds-pr-kanon'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'k-anonymity: groups merging as detail is removed', 'every dot is a person; boxes are groups that share the same quasi-identifiers (synthetic, illustrative)',
      `<canvas style="height:470px"></canvas>`,
      `${sl('lvl', 'Generalisation level (0 = exact detail)', 0, 4, 1, 0)}${sl('kt', 'Target k', 2, 6, 1, 3)}${speedCtl}
       ${rowBtns()}
       <div class="readout" id="out"></div>
       <p class="caption">Quasi-identifiers: age, ZIP code, sex. Level 0: exact age, 5-digit ZIP, sex. 1: 5-year age band, 4-digit ZIP. 2: 10-year band, 3-digit ZIP. 3: 20-year band. 4: also drop sex and keep a 2-digit ZIP. Dot colour is a sensitive value (diagnosis). Red box: someone is unique. Dashed outline: everyone in the box has the same diagnosis.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const lvl = q(ctl, '#lvl'), kt = q(ctl, '#kt'), spd = q(ctl, '#spd');
    const rnd = AI.rng(31), DX = ['flu', 'asthma', 'diabetes', 'none'], N = 80;
    const rec = Array.from({ length: N }, () => ({ age: 20 + Math.floor(rnd() * 50), zip: '10' + (1 + Math.floor(rnd() * 3)) + String(Math.floor(rnd() * 12)).padStart(2, '0').slice(0, 2), sex: rnd() < 0.5 ? 'F' : 'M', dx: DX[Math.floor(rnd() * 4)], x: 0, y: 0 }));
    const key = (r, l) => [
      () => r.age + '|' + r.zip + '|' + r.sex,
      () => 'age ' + Math.floor(r.age / 5) * 5 + '-' + (Math.floor(r.age / 5) * 5 + 4) + '|' + r.zip.slice(0, 4) + '*|' + r.sex,
      () => 'age ' + Math.floor(r.age / 10) * 10 + 's|' + r.zip.slice(0, 3) + '**|' + r.sex,
      () => 'age ' + (r.age < 40 ? '20-39' : r.age < 60 ? '40-59' : '60-69') + '|' + r.zip.slice(0, 3) + '**|' + r.sex,
      () => 'age ' + (r.age < 40 ? '20-39' : r.age < 60 ? '40-59' : '60-69') + '|' + r.zip.slice(0, 2) + '***',
    ][l]();
    const dxCol = (c) => ({ flu: c.blue, asthma: c.orange, diabetes: c.purple, none: c.muted });
    let layout = null, playing = false, acc = 0, dirty = true, shownLvl = -1;
    const compute = (l) => {
      const m = {}; rec.forEach((r) => (m[key(r, l)] = (m[key(r, l)] || [])).push(r));
      const groups = Object.entries(m).map(([k, rs]) => ({ k, rs })).sort((a, b) => a.rs.length - b.rs.length || (a.k < b.k ? -1 : 1));
      return groups;
    };
    const place = (groups, W) => {
      // shelf packing: each group box is cols x rows dots
      const D = 12, pad = 8, gapx = 8, gapy = 22; let x = 8, y = 8, rowH = 0; const boxes = [];
      groups.forEach((gr) => {
        const n = gr.rs.length, cols = Math.min(n, 8), rows = Math.ceil(n / cols), bw = Math.max(cols * D + pad * 2, 36), bh = rows * D + pad * 2;
        if (x + bw > W - 8) { x = 8; y += rowH + gapy; rowH = 0; }
        gr.bx = x; gr.by = y; gr.bw = bw; gr.bh = bh;
        gr.rs.forEach((r, i) => { r.tx = x + pad + (i % cols) * D + D / 2; r.ty = y + pad + Math.floor(i / cols) * D + D / 2; });
        x += bw + gapx; rowH = Math.max(rowH, bh); boxes.push(gr);
      });
      return boxes;
    };
    const rebuild = () => {
      const l = +lvl.value; layout = place(compute(l), o.w); shownLvl = l;
      rec.forEach((r) => { if (!r.init) { r.x = r.tx; r.y = r.ty; r.init = true; } });
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      q(ctl, '#lvlv').textContent = lvl.value; q(ctl, '#ktv').textContent = kt.value; q(ctl, '#spdv').textContent = spd.value + '×';
      const K = +kt.value, cols = dxCol(c);
      layout.forEach((gr) => {
        const n = gr.rs.length, homog = gr.rs.every((r) => r.dx === gr.rs[0].dx);
        g.strokeStyle = n === 1 ? c.red : n < K ? c.yellow : c.green; g.lineWidth = n < K ? 2 : 1; if (homog && n > 1) g.setLineDash([4, 3]);
        g.fillStyle = c.bg2; g.beginPath(); g.roundRect(gr.bx, gr.by, gr.bw, gr.bh, 5); g.fill(); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
        if (gr.bw >= 64 || n > 1) txt(g, 'n=' + n, gr.bx + 2, gr.by + gr.bh + 11, c.muted, '9px sans-serif');
      });
      rec.forEach((r) => { g.fillStyle = cols[r.dx]; g.beginPath(); g.arc(r.x, r.y, 4.4, 0, 7); g.fill(); });
      const sizes = layout.map((gr) => gr.rs.length), kMin = Math.min(...sizes), small = layout.filter((gr) => gr.rs.length < K).reduce((s, gr) => s + gr.rs.length, 0);
      const uniq = layout.filter((gr) => gr.rs.length === 1).length, hom = layout.filter((gr) => gr.rs.length > 1 && gr.rs.every((r) => r.dx === gr.rs[0].dx)).reduce((s, gr) => s + gr.rs.length, 0);
      txt(g, 'flu', W - 190, H - 8, cols.flu, '11px sans-serif'); txt(g, 'asthma', W - 150, H - 8, cols.asthma, '11px sans-serif'); txt(g, 'diabetes', W - 100, H - 8, cols.diabetes, '11px sans-serif'); txt(g, 'none', W - 44, H - 8, cols.none, '11px sans-serif');
      out.textContent = 'level ' + lvl.value + ': ' + layout.length + ' groups, k = smallest group = ' + kMin + '\nunique people (k = 1): ' + uniq + '\nin groups smaller than ' + K + ': ' + small + (kMin >= K ? ' → ' + K + '-anonymous' : ' → not ' + K + '-anonymous') + '\nin same-diagnosis groups: ' + hom + ' (still leaks)';
    };
    const sync = () => { if (shownLvl !== +lvl.value) rebuild(); dirty = true; };
    lvl.oninput = sync; kt.oninput = () => (dirty = true); spd.oninput = () => (dirty = true);
    q(ctl, '#play').textContent = '▶ Play'; q(ctl, '#play').onclick = (e) => { if (!playing && +lvl.value >= 4) { lvl.value = 0; sync(); } playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; lvl.value = Math.min(4, +lvl.value + 1); sync(); };
    q(ctl, '#reset').onclick = () => { playing = false; acc = 0; q(ctl, '#play').textContent = '▶ Play'; lvl.value = 0; sync(); };
    o.onresize = () => { rebuild(); draw(); }; themeRedraw(draw);
    AI.loop(root, (dt) => {
      if (playing) { acc += dt * 0.3 * +spd.value; if (acc >= 1) { acc = 0; if (+lvl.value < 4) { lvl.value = +lvl.value + 1; sync(); } else { playing = false; q(ctl, '#play').textContent = '▶ Play'; } } }
      if (!layout) return;
      const f = 1 - Math.exp(-dt * 3.5); let moving = false;
      rec.forEach((r) => { const dx = r.tx - r.x, dy = r.ty - r.y; if (Math.abs(dx) + Math.abs(dy) > 0.15) { r.x += dx * f; r.y += dy * f; moving = true; } else { r.x = r.tx; r.y = r.ty; } });
      if (moving || dirty) { dirty = false; draw(); }
    }, { fps: 30 });
    rebuild(); draw();
  };

  /* ---------------------------------------------------------------- 6. Scoping a project */
  AI.labs['ds-pr-scope'] = function (root) {
    const ST = ['Ask & agree', 'Collect', 'Clean', 'Explore', 'Model/test', 'Communicate', 'Deploy & monitor'];
    const DEF = [1, 3, 5, 3, 4, 2, 2];
    const { view, ctl } = AI.labShell(root, 'Scoping a project: where the time goes', 'plan the six-stage loop in working days and see what an unclear question costs (illustrative planning toy)',
      `<canvas style="height:330px"></canvas>`,
      `${ST.map((s, i) => sl('d' + i, s + ' (days)', 0, 15, 1, DEF[i])).join('')}${sl('cl', 'How clear is the question? (assumption)', 0, 1, 0.05, 0.5)}${speedCtl}
       ${rowBtns()}
       <div class="readout" id="out"></div>
       <p class="caption">Days are YOUR plan, not benchmarks. Assumption in this toy, not a measured rule: every unit of unclarity adds rework equal to 60% of the work that came after the Ask stage. The playhead walks the plan; the striped bar is the rework.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), out = q(ctl, '#out');
    const dIn = ST.map((_, i) => q(ctl, '#d' + i)), cl = q(ctl, '#cl'), spd = q(ctl, '#spd');
    const cols = (c) => [c.blue, c.teal, c.green, c.yellow, c.orange, c.purple, c.pink];
    let t = 0, playing = true, dirty = true;
    const plan = () => { const d = dIn.map((e) => +e.value), down = d.slice(1).reduce((s, x) => s + x, 0), rework = (1 - +cl.value) * 0.6 * down, total = d.reduce((s, x) => s + x, 0) + rework; return { d, rework, total, down }; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      ST.forEach((s, i) => (q(ctl, '#d' + i + 'v').textContent = dIn[i].value)); q(ctl, '#clv').textContent = Math.round(+cl.value * 100) + '%'; q(ctl, '#spdv').textContent = spd.value + '×';
      const P = plan(), L = 14, R = W - 14, scale = (R - L) / Math.max(1, P.total), cc = cols(c);
      let x = L; const ph = L + Math.min(t, P.total) * scale;
      P.d.forEach((d, i) => {
        const w = d * scale; g.fillStyle = cc[i]; g.globalAlpha = ph >= x + w ? 0.95 : 0.35; g.fillRect(x, 50 + (i % 2) * 40, Math.max(0, w - 1), 34); g.globalAlpha = 1;
        if (w > 18) txt(g, d + 'd', x + w / 2, 72 + (i % 2) * 40, c.bg, 'bold 11px sans-serif', 'center');
        txt(g, ST[i], x, i % 2 ? 136 : 44, c.muted, '10px sans-serif'); x += w;
      });
      // rework striped bar
      const rw = P.rework * scale; if (rw > 0) { g.fillStyle = c.red; g.globalAlpha = ph > x ? 0.7 : 0.25; for (let k = 0; k < rw; k += 6) g.fillRect(x + k, 160, Math.min(3, rw - k), 34); g.globalAlpha = 1; txt(g, 'rework ' + P.rework.toFixed(1) + 'd', x, 212, c.red, '11px sans-serif'); }
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(ph, 30); g.lineTo(ph, 200); g.stroke(); g.lineWidth = 1;
      // what the stakeholder meets now
      const where = (() => { let a = 0; for (let i = 0; i < 7; i++) { a += P.d[i]; if (t < a) return i; } return t < P.total ? 7 : 8; })();
      const say = ['Agree the decision, the metric, who owns it, and what "done" looks like.', 'Find the data, check access and licence, fix the extract.', 'Handle missing values, joins, duplicates; log every rule.', 'Look before modelling; share surprises early.', 'Baseline first, then models; evaluate honestly.', 'Report method, limits and uncertainty; ask for decisions.', 'Ship, monitor drift, plan the retrain.', 'Rework: the question changed late, so earlier stages are redone.', 'Plan complete.'][where];
      txt(g, where < 7 ? 'Now: ' + ST[where] : where === 7 ? 'Now: rework' : 'Done', L, H - 46, c.fg, 'bold 13px sans-serif'); txt(g, say, L, H - 26, c.muted, '12px sans-serif');
      out.textContent = 'planned work: ' + (P.total - P.rework) + ' days\nexpected rework: ' + P.rework.toFixed(1) + ' days\ntotal: ' + P.total.toFixed(1) + ' days   (rework is ' + (P.total ? Math.round(P.rework / P.total * 100) : 0) + '% of total)\ntime on the Ask stage: ' + P.d[0] + ' day(s)';
    };
    [...dIn, cl, spd].forEach((e) => (e.oninput = () => (dirty = true)));
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; t = Math.min(plan().total, Math.floor(t) + 1); dirty = true; };
    q(ctl, '#reset').onclick = () => { t = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; dirty = true; };
    o.onresize = draw; themeRedraw(draw);
    AI.loop(root, (dt) => { if (playing) { const T = plan().total; if (t < T) { t = Math.min(T, t + dt * 1.5 * +spd.value); dirty = true; } } if (dirty) { dirty = false; draw(); } }, { fps: 30 });
    draw();
  };
})();
