/* Data Science Module 4 labs: visual encodings, choosing charts, honest axes, colour, dashboards, storytelling.
   Every lab animates on dt (not per frame), starts slowly, and has speed / pause / reset. Datasets are small and seeded;
   all simulations are illustrative. Colours come from AI.col() except in the palette labs, where the colours are the subject. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const nice = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m))), f = m / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; };
  const hexRGB = (h) => { h = h.replace('#', ''); if (h.length === 3) h = h.replace(/./g, '$&$&'); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
  const toHex = (a) => '#' + a.map((v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')).join('').toUpperCase();
  const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = (a) => 0.2126 * lin(a[0]) + 0.7152 * lin(a[1]) + 0.0722 * lin(a[2]);
  const gam = (v) => { v = clamp(v, 0, 1); return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055)); };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const mixA = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const rgbs = (a) => `rgb(${a.map((v) => Math.round(v)).join(',')})`;
  const parseList = (s) => s.split(/[,\s;]+/).filter(Boolean).map(Number);
  // shared play / step / reset row
  const btns = (step) => `<div class="row"><button class="btn primary" id="play">▶ Play</button>${step ? '<button class="btn" id="step">Step ▸</button>' : ''}<button class="btn" id="reset">Reset</button></div>`;
  const errBox = (id) => `<div id="${id}" style="color:var(--red);font-size:12px;min-height:14px"></div>`;

  /* ================= 1. a chart built one encoding at a time ================= */
  AI.labs['ds-vz-encode'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A chart built one encoding at a time', 'position, then length, then colour, then area (illustrative)',
      `<canvas style="height:410px"></canvas>`,
      `<label>Weekly sales (thousands), comma-separated<input type="text" id="vals" value="62, 55, 48, 41, 34"></label>${errBox('err')}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       <label>Circle size rule <select id="rule"><option value="area">area proportional to value (correct)</option><option value="radius">radius proportional to value (common bug)</option></select></label>
       ${btns(true)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const vIn = q(ctl, '#vals'), spd = q(ctl, '#spd'), rule = q(ctl, '#rule'), out = q(ctl, '#out'), cap = q(ctl, '#cap'), err = q(ctl, '#err');
    let vals = [62, 55, 48, 41, 34], t = 0, playing = false, last = '';
    const CAPS = ['Stage 1 of 5. We have a table of five shops and nothing drawn yet: only an axis and the names. Position along the horizontal axis says WHICH shop.',
      'Stage 2 of 5. Position on a common scale: each value becomes a dot at a height. You can compare any two dots against the same gridlines. This is the most accurate encoding.',
      'Stage 3 of 5. Length: the dot grows a bar from the zero line. Bars add a second cue (length) on top of position, which is why bar charts are so readable.',
      'Stage 4 of 5. Colour: one accent hue picks out the largest shop and the rest fade. Colour is good for category and emphasis, poor for reading off quantities.',
      'Stage 5 of 5. Area: the same numbers drawn as circles. Notice how hard it is to say by how much the largest is bigger than the smallest.'];
    const parse = () => {
      const v = parseList(vIn.value);
      if (v.length < 2 || v.length > 8 || v.some((x) => !(x > 0) || !isFinite(x))) { err.textContent = 'Enter 2 to 8 positive numbers.'; return; }
      err.textContent = ''; vals = v; last = '';
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const n = vals.length, mx = Math.max(...vals), mn = Math.min(...vals), ymax = nice(mx * 1.08);
      const x0 = 46, x1 = W - 14, top = 28, base = 250, ht = base - top, slot = (x1 - x0) / n, bw = Math.min(60, slot * 0.5);
      const p = [0, 1, 2, 3, 4].map((k) => ease((t - k) / 0.6));
      g.globalAlpha = p[0];
      g.strokeStyle = c.border; g.lineWidth = 1; g.fillStyle = c.muted;
      for (let i = 0; i <= 4; i++) {
        const v = (ymax * i) / 4, y = base - (v / ymax) * ht;
        g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
        txt(g, AI.fmt(v, 3), x0 - 6, y + 4, c.muted, '11px sans-serif', 'right');
      }
      for (let i = 0; i < n; i++) txt(g, 'Shop ' + String.fromCharCode(65 + i), x0 + slot * (i + 0.5), base + 16, c.fg, '12px sans-serif', 'center');
      g.globalAlpha = 1;
      const big = vals.indexOf(mx);
      for (let i = 0; i < n; i++) {
        const cx = x0 + slot * (i + 0.5), full = (vals[i] / ymax) * ht;
        const col = p[3] > 0 ? AI.mix(c.blue, i === big ? c.orange : c.muted, p[3]) : c.blue;
        if (p[2] > 0) { g.globalAlpha = 0.85; g.fillStyle = col; g.fillRect(cx - bw / 2, base - full * p[2], bw, full * p[2]); g.globalAlpha = 1; }
        if (p[1] > 0) {
          const y = base - full * p[1];
          g.fillStyle = p[2] > 0 ? c.fg : col; g.beginPath(); g.arc(cx, y, 5, 0, 7); g.fill();
          if (p[2] > 0.9) txt(g, String(vals[i]), cx, y - 10, c.fg, '12px sans-serif', 'center');
        }
      }
      // area row
      if (p[4] > 0) {
        const cy = 336, rmax = 34, byRadius = rule.value === 'radius';
        txt(g, 'size = ' + (byRadius ? 'radius ∝ value' : 'area ∝ value'), x0, cy - 48, c.muted, '11px sans-serif');
        for (let i = 0; i < n; i++) {
          const cx = x0 + slot * (i + 0.5), r = rmax * (byRadius ? vals[i] / mx : Math.sqrt(vals[i] / mx)) * p[4];
          g.globalAlpha = 0.85; g.fillStyle = i === big ? c.orange : c.muted; g.beginPath(); g.arc(cx, cy, Math.max(r, 0.5), 0, 7); g.fill(); g.globalAlpha = 1;
          txt(g, String(vals[i]), cx, cy + rmax + 14, c.fg, '12px sans-serif', 'center');
        }
      }
      const r = mx / mn, s = rule.value === 'radius' ? r * r : r;
      out.innerHTML = `Largest ÷ smallest = ${mx} ÷ ${mn} = <b>${r.toFixed(3)}</b>. Perceived ratio (illustrative, Stevens exponents): by position or length ≈ ${r.toFixed(2)}; by area ≈ ${Math.pow(s, 0.7).toFixed(2)}${rule.value === 'radius' ? ` (the drawn area ratio is ${s.toFixed(2)}, the bug inflates it)` : ''}.`;
      cap.textContent = CAPS[clamp(Math.floor(t + 1e-9), 0, 4)];
    };
    const redraw = () => { last = ''; };
    vIn.oninput = parse; rule.onchange = redraw; spd.oninput = redraw;
    q(ctl, '#play').onclick = (e) => { if (t >= 4.99) t = 0; playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; t = Math.min(5, Math.floor(t + 1e-9) + 1); last = ''; };
    q(ctl, '#reset').onclick = () => { t = 0; playing = false; q(ctl, '#play').textContent = '▶ Play'; last = ''; };
    o.onresize = redraw;
    AI.loop(root, (dt) => {
      if (playing) { t = Math.min(5, t + dt * 0.25 * +spd.value); if (t >= 5) { playing = false; q(ctl, '#play').textContent = '▶ Play'; } }
      q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
      const key = [t.toFixed(3), vals.join(), rule.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 2. choosing the right chart ================= */
  AI.labs['ds-vz-choose'] = function (root) {
    const r = AI.rng(7);
    const wait = Array.from({ length: 60 }, () => Math.round((4 + -Math.log(1 - r()) * 8) * 10) / 10);
    const pairs = Array.from({ length: 24 }, () => { const x = 1 + r() * 9, y = 100 + 5 * x + AI.randn(r) * 6; return [x, y]; });
    const DATA = {
      comparison: { lab: ['A', 'B', 'C', 'D', 'E'], v: [62, 55, 48, 41, 34], ask: 'Which shop sells the most?' },
      composition: { lab: ['Search', 'Social', 'Email', 'Direct'], v: [40, 25, 20, 15], ask: 'What share of visits comes from each channel?' },
      distribution: { lab: null, v: wait, ask: 'How are customer waiting times spread out?' },
      relationship: { lab: null, v: pairs.map((p) => p[1]), pairs, ask: 'Do weeks with more ad spend have more sales?' },
      trend: { lab: ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], v: Array.from({ length: 12 }, (_, m) => Math.round(120 + 3.2 * m + 8 * Math.sin((m / 12) * 6.283) + (r() - 0.5) * 6)), ask: 'How are monthly sales moving over the year?' }
    };
    const FIT = {
      comparison: { bar: ['good', 'Bars compare a few categories by length from a common zero line.'], line: ['poor', 'A line implies the points are ordered and connected; shops are not.'], hist: ['poor', 'A histogram bins one variable; it does not compare named items.'], scatter: ['poor', 'A scatter needs two numeric variables; here one axis is just the shop name.'], pie: ['ok', 'Works for 3 to 5 slices, but angles are harder to compare than lengths.'] },
      composition: { bar: ['good', 'Bars (or one stacked bar) make the parts easy to compare and add to 100.'], line: ['poor', 'Channels have no order, so a line has no meaning.'], hist: ['poor', 'A histogram shows how values are spread, not parts of a whole.'], scatter: ['poor', 'No second variable to plot.'], pie: ['ok', 'A pie is acceptable here: few parts, and the whole is the point.'] },
      distribution: { bar: ['poor', 'One bar per customer is a table in disguise; the shape is not visible.'], line: ['poor', 'The order of customers is arbitrary, so the line zigzags for no reason.'], hist: ['good', 'Binning shows the shape: most waits are short, a few are long (right-skewed).'], scatter: ['poor', 'Plots waiting time against row number, which carries no meaning.'], pie: ['poor', 'A pie of 60 customers is unreadable.'] },
      relationship: { bar: ['poor', 'Bars of one variable hide the other variable entirely.'], line: ['poor', 'Joins points by their order, not by ad spend.'], hist: ['poor', 'Shows only the sales values, not how they relate to spend.'], scatter: ['good', 'One point per week: ad spend across, sales up. The trend and the outliers are visible.'], pie: ['poor', 'Pies cannot show two variables.'] },
      trend: { bar: ['ok', 'Bars work for a few periods, but a line shows direction and slope better.'], line: ['good', 'Time runs left to right and the line shows direction, slope and seasonality.'], hist: ['poor', 'Throws away the time order, the very thing the question asks about.'], scatter: ['ok', 'Fine as dots, but without the line the eye has to join them.'], pie: ['poor', 'Months are not parts of a whole.'] }
    };
    const { view, ctl } = AI.labShell(root, 'Same question, different charts', 'pick the question, then try every chart type (illustrative data)',
      `<canvas style="height:340px"></canvas>`,
      `<label>The question <select id="qn"><option value="comparison">comparison</option><option value="composition">composition</option><option value="distribution">distribution</option><option value="relationship">relationship</option><option value="trend" selected>trend</option></select></label>
       <label>Chart type <select id="ch"><option value="bar">bar</option><option value="line" selected>line</option><option value="hist">histogram</option><option value="scatter">scatter</option><option value="pie">pie</option></select></label>
       ${sl('bins', 'Histogram bins', 3, 20, 1, 8)}
       <label>Your own numbers (comma-separated, blank = default; ignored for relationship)<input type="text" id="mine" placeholder="e.g. 3, 8, 5, 12, 9"></label>
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(false)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const qn = q(ctl, '#qn'), ch = q(ctl, '#ch'), bins = q(ctl, '#bins'), mine = q(ctl, '#mine'), spd = q(ctl, '#spd'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let p = 0, playing = true, last = '';
    const getData = () => {
      const d = DATA[qn.value], mv = parseList(mine.value);
      if (qn.value !== 'relationship' && mv.length >= 3 && mv.length <= 60 && mv.every((x) => isFinite(x) && x >= 0)) return { lab: mv.length === d.v.length ? d.lab : null, v: mv, pairs: null };
      return { lab: d.lab, v: d.v, pairs: d.pairs || null };
    };
    const frame = (g, x0, x1, top, base, ymax, ymin, name) => {
      g.strokeStyle = c.border; g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { const v = ymin + ((ymax - ymin) * i) / 4, y = base - ((v - ymin) / (ymax - ymin)) * (base - top); g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); txt(g, AI.fmt(v, 3), x0 - 6, y + 4, c.muted, '11px sans-serif', 'right'); }
    };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const d = getData(), v = d.v, n = v.length, kind = ch.value, e = ease(p);
      const x0 = 50, x1 = W - 16, top = 34, base = H - 36, ht = base - top;
      txt(g, DATA[qn.value].ask, x0, 18, c.fg, 'bold 13px sans-serif');
      const labelAt = (i, x) => { const s = d.lab ? d.lab[i] : (n <= 14 ? String(i + 1) : ''); if (s) txt(g, s, x, base + 16, c.muted, '11px sans-serif', 'center'); };
      if (kind === 'bar') {
        const ymax = nice(Math.max(...v) * 1.08); frame(g, x0, x1, top, base, ymax, 0); const slot = (x1 - x0) / n;
        v.forEach((y, i) => { const h = (y / ymax) * ht * e, bw = Math.max(1, slot * 0.7); g.fillStyle = c.blue; g.fillRect(x0 + slot * i + (slot - bw) / 2, base - h, bw, h); labelAt(i, x0 + slot * (i + 0.5)); });
      } else if (kind === 'line') {
        const lo = Math.min(...v), hi = Math.max(...v), ymax = nice(hi * 1.05); frame(g, x0, x1, top, base, ymax, 0);
        const px = (i) => x0 + ((x1 - x0) * (i + 0.5)) / n, py = (y) => base - (y / ymax) * ht, k = e * (n - 1);
        g.strokeStyle = c.blue; g.lineWidth = 2.5; g.beginPath();
        for (let i = 0; i <= Math.floor(k); i++) i ? g.lineTo(px(i), py(v[i])) : g.moveTo(px(i), py(v[i]));
        if (Math.floor(k) < n - 1) { const f = k - Math.floor(k), i = Math.floor(k); g.lineTo(px(i) + (px(i + 1) - px(i)) * f, py(v[i] + (v[i + 1] - v[i]) * f)); }
        g.stroke(); v.forEach((_, i) => { if (i <= k) labelAt(i, px(i)); });
        void lo;
      } else if (kind === 'hist') {
        const nb = +bins.value, lo = Math.min(...v), hi = Math.max(...v), w = (hi - lo) / nb || 1, cnt = Array(nb).fill(0);
        v.forEach((y) => { cnt[Math.min(nb - 1, Math.floor((y - lo) / w))]++; });
        const ymax = nice(Math.max(...cnt) * 1.1); frame(g, x0, x1, top, base, ymax, 0); const slot = (x1 - x0) / nb;
        cnt.forEach((k, i) => { const h = (k / ymax) * ht * e; g.fillStyle = c.blue; g.fillRect(x0 + slot * i + 1, base - h, slot - 2, h); });
        txt(g, `${AI.fmt(lo, 3)}`, x0, base + 16, c.muted, '11px sans-serif', 'center'); txt(g, `${AI.fmt(hi, 3)}`, x1, base + 16, c.muted, '11px sans-serif', 'center');
        txt(g, `${nb} bins of width ${AI.fmt(w, 3)}`, (x0 + x1) / 2, base + 16, c.muted, '11px sans-serif', 'center');
      } else if (kind === 'scatter') {
        const pts = d.pairs || v.map((y, i) => [i + 1, y]), xs = pts.map((a) => a[0]), ys = pts.map((a) => a[1]);
        const xlo = Math.min(...xs), xhi = Math.max(...xs), ylo = Math.floor(Math.min(...ys) / 10) * 10, yhi = Math.max(...ys) * 1.03;
        const yl = d.pairs ? ylo : 0; frame(g, x0, x1, top, base, nice(yhi) > yhi ? yhi : yhi, yl);
        const k = Math.floor(e * pts.length);
        pts.forEach((a, i) => { if (i < k) { g.fillStyle = c.blue; g.globalAlpha = 0.85; g.beginPath(); g.arc(x0 + ((a[0] - xlo) / (xhi - xlo || 1)) * (x1 - x0 - 10) + 5, base - ((a[1] - yl) / (yhi - yl)) * ht, 4, 0, 7); g.fill(); g.globalAlpha = 1; } });
        txt(g, d.pairs ? 'ad spend (thousand)' : 'row number', (x0 + x1) / 2, base + 16, c.muted, '11px sans-serif', 'center');
      } else {
        const vv = v.slice(0, 8), tot = vv.reduce((a, b) => a + b, 0), cx = (x0 + x1) / 2, cy = (top + base) / 2 + 6, R = Math.min(ht, x1 - x0) / 2 - 8;
        const cols = [c.blue, c.orange, c.green, c.purple, c.yellow, c.pink, c.teal, c.red];
        let a0 = -Math.PI / 2; const sweep = e * Math.PI * 2;
        vv.forEach((y, i) => { const a1 = a0 + (y / tot) * Math.PI * 2, lim = -Math.PI / 2 + sweep; if (a0 < lim) { g.fillStyle = cols[i]; g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, Math.min(a1, lim)); g.closePath(); g.fill(); g.strokeStyle = c.bg; g.lineWidth = 2; g.stroke(); if (a1 <= lim && vv.length <= 6) { const m = (a0 + a1) / 2; txt(g, (d.lab ? d.lab[i] + ' ' : '') + Math.round((y / tot) * 100) + '%', cx + Math.cos(m) * R * 0.62, cy + Math.sin(m) * R * 0.62 + 4, c.bg, 'bold 12px sans-serif', 'center'); } } a0 = a1; });
        if (v.length > 8) txt(g, 'only the first 8 values shown', cx, base + 16, c.muted, '11px sans-serif', 'center');
      }
      const f = FIT[qn.value][kind], colr = f[0] === 'good' ? c.green : f[0] === 'ok' ? c.yellow : c.red;
      out.innerHTML = `Fit for a <b>${qn.value}</b> question: <b style="color:${colr}">${f[0]}</b>. ${f[1]}`;
      cap.textContent = p < 1 ? 'Drawing the chart: marks appear at a steady pace so you can see how this chart type builds up.' : 'Finished. Change the chart type or the question and compare how easily the answer can be read.';
      q(ctl, '#binsv').textContent = bins.value; q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    const redo = () => { p = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; last = ''; };
    qn.onchange = redo; ch.onchange = redo; mine.oninput = redo; bins.oninput = () => { last = ''; };
    spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { if (p >= 1) p = 0; playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = redo;
    o.onresize = () => { last = ''; };
    q(ctl, '#play').textContent = '⏸ Pause';
    AI.loop(root, (dt) => {
      if (playing && p < 1) { p = Math.min(1, p + dt * 0.3 * +spd.value); if (p >= 1) { playing = false; q(ctl, '#play').textContent = '▶ Play'; } }
      const key = [p.toFixed(3), qn.value, ch.value, bins.value, mine.value, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 3a. the same data with a truncated and a full axis ================= */
  AI.labs['ds-vz-axis'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'One difference, two axes', 'the same two bars on a truncated axis (left) and a zero-based axis (right)',
      `<canvas style="height:360px"></canvas>`,
      `<label>Value A<input type="text" id="va" value="52"></label><label>Value B<input type="text" id="vb" value="55"></label>${errBox('err')}
       ${sl('base', 'Baseline of the left chart', 0, 51, 0.5, 50)}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(false)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Play slides the left baseline from 0 up to just under the smaller value and back, slowly. The right chart never moves. Heights are proportional to (value − baseline).</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const va = q(ctl, '#va'), vb = q(ctl, '#vb'), base = q(ctl, '#base'), spd = q(ctl, '#spd'), out = q(ctl, '#out'), err = q(ctl, '#err');
    let A = 52, B = 55, b = 50, dir = 1, playing = false, last = '';
    const parse = () => {
      const a = +va.value, bb = +vb.value;
      if (!(a > 0 && bb > 0 && isFinite(a) && isFinite(bb))) { err.textContent = 'Enter two positive numbers.'; return; }
      err.textContent = ''; A = a; B = bb; const m = Math.min(A, B); base.max = String(Math.floor(m * 0.98 * 2) / 2); b = Math.min(b, +base.max); base.value = b; last = '';
    };
    const panel = (g, x0, x1, top, bs, title, accent) => {
      const bot = top + 250, ymax = Math.max(A, B) + (Math.max(A, B) - bs) * 0.15, H = bot - top, f = (v) => ((v - bs) / (ymax - bs)) * H;
      txt(g, title, x0, top - 12, accent, 'bold 12px sans-serif');
      g.strokeStyle = c.border; g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { const v = bs + ((ymax - bs) * i) / 4, y = bot - f(v); g.beginPath(); g.moveTo(x0 + 44, y); g.lineTo(x1, y); g.stroke(); txt(g, v.toFixed(1), x0 + 40, y + 4, c.muted, '11px sans-serif', 'right'); }
      const w = Math.min(70, (x1 - x0 - 44) / 4), cx = [x0 + 44 + (x1 - x0 - 44) * 0.3, x0 + 44 + (x1 - x0 - 44) * 0.7];
      [[A, c.blue, 'A'], [B, c.orange, 'B']].forEach(([v, col, nm], i) => {
        const h = Math.max(0, f(v)); g.fillStyle = col; g.fillRect(cx[i] - w / 2, bot - h, w, h);
        txt(g, nm + ' = ' + AI.fmt(v, 4), cx[i], bot + 16, c.fg, '12px sans-serif', 'center');
      });
      return [f(A), f(B)];
    };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const mid = W / 2;
      const [hl1, hl2] = panel(g, 6, mid - 10, 40, b, `Truncated axis (starts at ${AI.fmt(b, 3)})`, c.red);
      const [hr1, hr2] = panel(g, mid + 10, W - 6, 40, 0, 'Full axis (starts at 0)', c.green);
      void hr1; void hr2;
      const shown = hl1 > 0 ? (hl2 - hl1) / hl1 : Infinity, truth = (B - A) / A, lie = A / (A - b);
      out.innerHTML = `Truth: B ÷ A = ${(B / A).toFixed(4)}, a difference of <b>${(truth * 100).toFixed(2)}%</b>.<br>Left chart: bar B is ${hl1 > 0 ? (hl2 / hl1).toFixed(3) : '∞'} times as tall as bar A, so it <b>looks</b> ${(shown * 100).toFixed(1)}% bigger. Lie factor = A ÷ (A − baseline) = ${A} ÷ ${AI.fmt(A - b, 4)} = <b>${lie.toFixed(2)}</b>.`;
      q(ctl, '#basev').textContent = b.toFixed(1); q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    va.oninput = parse; vb.oninput = parse;
    base.oninput = () => { b = +base.value; last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; b = Math.min(50, +base.max); base.value = b; dir = 1; last = ''; };
    o.onresize = () => { last = ''; };
    AI.loop(root, (dt) => {
      if (playing) { const m = +base.max; b += dir * dt * (m / 8) * +spd.value; if (b >= m) { b = m; dir = -1; } if (b <= 0) { b = 0; dir = 1; } base.value = b; }
      const key = [b.toFixed(3), A, B, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 3b. linear versus log scale, with an editable formula ================= */
  AI.labs['ds-vz-scale'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Linear and log scales', 'type any growth formula in t and watch it plotted both ways',
      `<canvas style="height:340px"></canvas>`,
      `<label>y(t) = <input type="text" id="f" value="100*1.5^t"></label>${errBox('err')}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(false)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">t runs 0 to 10 and the points appear one per second at speed 1. On the linear axis steps of equal size mean equal additions; on the log axis steps of equal size mean equal multiples, so constant percentage growth becomes a straight line.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const spd = q(ctl, '#spd'), out = q(ctl, '#out');
    let fn = null, tr = 0, playing = true, last = '';
    const ei = AI.exprInput(q(ctl, '#f'), q(ctl, '#err'), { allowed: ['t'], onChange: (f) => { fn = f; tr = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; last = ''; } });
    const Y = () => Array.from({ length: 11 }, (_, t) => fn({ t }));
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      if (!fn) return;
      const ys = Y(), ok = ys.every((y) => isFinite(y)), pos = ys.every((y) => y > 0);
      const top = 40, bot = Hh - 34, half = W / 2;
      const draw1 = (x0, x1, log, title) => {
        txt(g, title, x0, top - 14, c.fg, 'bold 12px sans-serif');
        g.strokeStyle = c.border; g.lineWidth = 1;
        if (!ok || (log && !pos)) { txt(g, log && ok ? 'log axis needs y > 0 everywhere' : 'y(t) is not finite', x0 + 40, (top + bot) / 2, c.red); return; }
        let f, ticks = [];
        if (log) {
          const lo = Math.floor(Math.log10(Math.min(...ys))), hi = Math.ceil(Math.log10(Math.max(...ys)) - 1e-9), span = Math.max(1, hi - lo);
          f = (y) => bot - ((Math.log10(y) - lo) / span) * (bot - top);
          for (let k = lo; k <= hi; k++) ticks.push([Math.pow(10, k), f(Math.pow(10, k))]);
        } else {
          const lo = Math.min(0, ...ys), hi = nice(Math.max(...ys, 1e-9) * 1.05);
          f = (y) => bot - ((y - lo) / (hi - lo)) * (bot - top);
          for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; ticks.push([v, f(v)]); }
        }
        ticks.forEach(([v, y]) => { g.beginPath(); g.moveTo(x0 + 44, y); g.lineTo(x1, y); g.stroke(); txt(g, AI.fmt(v, 3), x0 + 38, y + 4, c.muted, '11px sans-serif', 'right'); });
        const px = (t) => x0 + 44 + ((x1 - x0 - 54) * t) / 10;
        for (let t = 0; t <= 10; t += 2) txt(g, String(t), px(t), bot + 16, c.muted, '11px sans-serif', 'center');
        g.strokeStyle = log ? c.orange : c.blue; g.lineWidth = 2.5; g.beginPath();
        const k = tr;
        for (let s = 0; s <= 100; s++) { const t = (s / 100) * 10; if (t > k) break; const y = fn({ t }); s ? g.lineTo(px(t), f(y)) : g.moveTo(px(t), f(y)); }
        g.stroke();
        ys.forEach((y, t) => { if (t <= k) { g.fillStyle = log ? c.orange : c.blue; g.beginPath(); g.arc(px(t), f(y), 3.5, 0, 7); g.fill(); } });
      };
      draw1(4, half - 8, false, 'Linear axis');
      draw1(half + 8, W - 4, true, 'Log axis (powers of 10)');
      let msg = '';
      if (ok && pos) {
        const gr = Math.pow(ys[10] / ys[0], 0.1), sl2 = Math.log10(ys[10] / ys[0]) / 10;
        msg = `y(0) = ${AI.fmt(ys[0], 4)}, y(10) = ${AI.fmt(ys[10], 4)}. Average growth per step = ${gr.toFixed(4)}× (${((gr - 1) * 100).toFixed(1)}%). Slope on the log axis = log₁₀(${gr.toFixed(4)}) = ${sl2.toFixed(4)} decades per step${gr > 1 ? `; doubling time ≈ ${(Math.log(2) / Math.log(gr)).toFixed(2)} steps` : ''}.`;
      } else msg = 'Use a formula that is positive for t from 0 to 10, such as 100*1.5^t, 50*2^t or 5*t^2+1.';
      out.textContent = msg; q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { if (tr >= 10) tr = 0; playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { tr = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; last = ''; };
    q(ctl, '#play').textContent = '⏸ Pause';
    o.onresize = () => { last = ''; };
    ei.run();
    AI.loop(root, (dt) => {
      if (playing && tr < 10) { tr = Math.min(10, tr + dt * 1 * +spd.value); if (tr >= 10) { playing = false; q(ctl, '#play').textContent = '▶ Play'; } }
      const key = [tr.toFixed(2), q(ctl, '#f').value, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 3c. dual axes that imply a relationship ================= */
  AI.labs['ds-vz-dual'] = function (root) {
    const A = [20, 22, 25, 30, 34, 40, 44, 43, 37, 30, 24, 21], B = [310, 300, 330, 420, 430, 520, 520, 560, 480, 400, 330, 350];
    const M = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
    const { view, ctl } = AI.labShell(root, 'Dual axes: you choose where the lines cross', 'park visitors (thousand, left axis) and ice-cream sales (units, right axis), illustrative',
      `<canvas style="height:340px"></canvas>`,
      `${sl('rmin', 'Right axis minimum', 0, 300, 10, 0)}
       ${sl('rmax', 'Right axis maximum', 400, 2000, 20, 700)}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(false)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Left axis is fixed at 0 to 50. Play slowly sweeps the right-axis maximum; the correlation r never changes, only how the two lines sit against each other.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const rmin = q(ctl, '#rmin'), rmax = q(ctl, '#rmax'), spd = q(ctl, '#spd'), out = q(ctl, '#out');
    let mx = 700, dir = 1, playing = false, last = '';
    const rr = (() => { const n = 12, ma = A.reduce((a, b) => a + b) / n, mb = B.reduce((a, b) => a + b) / n; let s = 0, p = 0, qq = 0; A.forEach((x, i) => { s += (x - ma) * (B[i] - mb); p += (x - ma) ** 2; qq += (B[i] - mb) ** 2; }); return s / Math.sqrt(p * qq); })();
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const x0 = 46, x1 = W - 56, top = 24, bot = H - 36, mn = +rmin.value, ymaxL = 50;
      const px = (i) => x0 + ((x1 - x0) * (i + 0.5)) / 12, pl = (v) => bot - (v / ymaxL) * (bot - top), pr = (v) => bot - ((v - mn) / (mx - mn)) * (bot - top);
      g.strokeStyle = c.border; g.lineWidth = 1;
      for (let i = 0; i <= 5; i++) { const y = bot - ((bot - top) * i) / 5; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); txt(g, String(i * 10), x0 - 6, y + 4, c.blue, '11px sans-serif', 'right'); txt(g, String(Math.round(mn + ((mx - mn) * i) / 5)), x1 + 6, y + 4, c.orange, '11px sans-serif', 'left'); }
      M.forEach((m, i) => txt(g, m, px(i), bot + 16, c.muted, '11px sans-serif', 'center'));
      const line = (arr, f, col) => { g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath(); arr.forEach((v, i) => (i ? g.lineTo(px(i), f(v)) : g.moveTo(px(i), f(v)))); g.stroke(); };
      line(A, pl, c.blue); line(B, pr, c.orange);
      txt(g, 'visitors', x0, top - 8, c.blue, 'bold 12px sans-serif'); txt(g, 'ice-cream sales', x1, top - 8, c.orange, 'bold 12px sans-serif', 'right');
      let cross = [];
      for (let i = 0; i < 11; i++) { const d0 = pl(A[i]) - pr(B[i]), d1 = pl(A[i + 1]) - pr(B[i + 1]); if (d0 === 0 || d0 * d1 < 0) cross.push(M[i] + '–' + M[i + 1]); }
      out.innerHTML = `Right axis ${AI.fmt(mn, 3)} to ${AI.fmt(mx, 3)}. Correlation r = <b>${rr.toFixed(3)}</b> (unchanged by any axis choice). The lines cross ${cross.length} time${cross.length === 1 ? '' : 's'} on this chart${cross.length ? ' (between ' + cross.join(', ') + ')' : ''}. The crossings come from the axes, not from the data.`;
      q(ctl, '#rminv').textContent = rmin.value; q(ctl, '#rmaxv').textContent = String(Math.round(mx)); q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    rmin.oninput = () => { last = ''; }; rmax.oninput = () => { mx = +rmax.value; last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; mx = 700; dir = 1; rmax.value = 700; rmin.value = 0; last = ''; };
    o.onresize = () => { last = ''; };
    AI.loop(root, (dt) => {
      if (playing) { mx += dir * dt * 90 * +spd.value; if (mx >= 1400) { mx = 1400; dir = -1; } if (mx <= 500) { mx = 500; dir = 1; } rmax.value = mx; }
      const key = [mx.toFixed(1), rmin.value, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 4a. colour-blind simulation ================= */
  const MAT = {
    deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
    protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]]
  };
  const simulate = (rgb, k) => { if (k === 'normal') return rgb.slice(); const l = rgb.map(lin); return MAT[k].map((r) => gam(r[0] * l[0] + r[1] * l[1] + r[2] * l[2])); };
  const toLab = (rgb) => {
    const [R, G, B] = rgb.map(lin), X = 0.4124 * R + 0.3576 * G + 0.1805 * B, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = 0.0193 * R + 0.1192 * G + 0.9505 * B;
    const t = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116), fx = t(X / 0.95047), fy = t(Y), fz = t(Z / 1.08883);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  };
  const dE = (a, b) => { const x = toLab(a), y = toLab(b); return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); };
  const STD = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd'];
  const OKI = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00'];
  const SNAME = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo'];
  const minGap = (hexes, k) => { const S = hexes.map((h) => simulate(hexRGB(h), k)); let m = 1e9, pr = [0, 1]; for (let i = 0; i < S.length; i++) for (let j = i + 1; j < S.length; j++) { const d = dE(S[i], S[j]); if (d < m) { m = d; pr = [i, j]; } } return { m, pr }; };

  AI.labs['ds-vz-cvd'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Does your palette survive colour blindness?', 'five lines, two palettes, four ways of seeing (Machado et al. simulation, illustrative)',
      `<canvas style="height:400px"></canvas>`,
      `<label>How the viewer sees <select id="sim"><option value="normal">typical vision</option><option value="deutan" selected>deuteranopia (no green cones)</option><option value="protan">protanopia (no red cones)</option><option value="tritan">tritanopia (no blue cones)</option></select></label>
       <label>Redundant encoding <select id="red"><option value="off">off: colour only</option><option value="on">on: markers + direct labels</option></select></label>
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       <div class="row"><button class="btn primary" id="play">▶ Tour</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Top: a common default palette (red, green, blue, orange, purple). Bottom: the Okabe-Ito palette (orange, sky blue, bluish green, blue, vermillion). Colours morph slowly into the simulated view. Tour cycles through the four vision types.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const sim = q(ctl, '#sim'), red = q(ctl, '#red'), spd = q(ctl, '#spd'), out = q(ctl, '#out');
    let m = 1, cur = sim.value, tour = false, tt = 0, last = '';
    const Y = SNAME.map((_, k) => Array.from({ length: 9 }, (_, i) => 0.5 + 0.34 * Math.sin(i * 0.55 + k * 1.25) * (0.6 + 0.1 * k) + 0.05 * k - 0.1));
    const chart = (g, x0, y0, w, h, hexes, title) => {
      txt(g, title, x0, y0 - 8, c.fg, 'bold 12px sans-serif');
      g.strokeStyle = c.border; g.lineWidth = 1; g.strokeRect(x0, y0, w, h);
      const px = (i) => x0 + 12 + ((w - 110) * i) / 8, py = (v) => y0 + h - 10 - v * (h - 20);
      hexes.forEach((hx, k) => {
        const a = mixA(hexRGB(hx), simulate(hexRGB(hx), cur), ease(m)), col = rgbs(a);
        g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2.5; g.beginPath(); Y[k].forEach((v, i) => (i ? g.lineTo(px(i), py(v)) : g.moveTo(px(i), py(v)))); g.stroke();
        if (red.value === 'on') {
          Y[k].forEach((v, i) => { if (i % 2 === 0) { const x = px(i), y = py(v); g.beginPath(); if (k === 0) g.arc(x, y, 4, 0, 7); else if (k === 1) g.rect(x - 4, y - 4, 8, 8); else if (k === 2) { g.moveTo(x, y - 5); g.lineTo(x + 5, y + 4); g.lineTo(x - 5, y + 4); g.closePath(); } else if (k === 3) { g.moveTo(x, y - 5); g.lineTo(x + 5, y); g.lineTo(x, y + 5); g.lineTo(x - 5, y); g.closePath(); } else { g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4); g.lineWidth = 2; g.strokeStyle = col; g.stroke(); return; } g.fill(); } });
          txt(g, SNAME[k], px(8) + 8, py(Y[k][8]) + 4, c.fg, '11px sans-serif');
        } else {
          g.fillRect(x0 + w - 86, y0 + 12 + k * 20, 12, 12); txt(g, SNAME[k], x0 + w - 68, y0 + 22 + k * 20, c.fg, '11px sans-serif');
        }
      });
    };
    const draw = () => {
      const g = o.ctx, W = o.w; g.clearRect(0, 0, W, o.h);
      chart(g, 10, 24, W - 20, 160, STD, 'Common default palette'); chart(g, 10, 214, W - 20, 160, OKI, 'Okabe-Ito palette');
      const a = minGap(STD, cur), b = minGap(OKI, cur), nm = (r) => SNAME[r.pr[0]] + ' and ' + SNAME[r.pr[1]];
      out.innerHTML = `Smallest colour gap (CIE76 ΔE; below about 10 is hard to tell apart) under <b>${sim.options[sim.selectedIndex].text}</b>:<br>default palette <b>${a.m.toFixed(1)}</b> (${nm(a)}) · Okabe-Ito <b>${b.m.toFixed(1)}</b> (${nm(b)}).`;
      q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    const go = (v) => { cur = v; m = 0; last = ''; };
    sim.onchange = () => { tour = false; q(ctl, '#play').textContent = '▶ Tour'; go(sim.value); };
    red.onchange = () => { last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { tour = !tour; tt = 0; e.target.textContent = tour ? '⏸ Pause tour' : '▶ Tour'; };
    q(ctl, '#reset').onclick = () => { tour = false; q(ctl, '#play').textContent = '▶ Tour'; sim.value = 'deutan'; red.value = 'off'; go('deutan'); };
    o.onresize = () => { last = ''; };
    const order = ['normal', 'deutan', 'protan', 'tritan'];
    m = 0;
    AI.loop(root, (dt) => {
      if (m < 1) { m = Math.min(1, m + dt * 0.5 * +spd.value); last = ''; }
      if (tour) { tt += dt * +spd.value; if (tt > 5) { tt = 0; const nx = order[(order.indexOf(cur) + 1) % 4]; sim.value = nx; go(nx); } }
      const key = [m.toFixed(3), cur, red.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 4b. sequential and diverging ramps ================= */
  AI.labs['ds-vz-ramp'] = function (root) {
    const RAMPS = {
      viridis: { name: 'viridis (sequential)', stops: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'], div: false },
      rainbow: { name: 'rainbow / jet (do not use)', stops: ['#00007f', '#0000ff', '#00ffff', '#ffff00', '#ff0000', '#7f0000'], div: false },
      grey: { name: 'grey (sequential)', stops: ['#000000', '#ffffff'], div: false },
      diverge: { name: 'blue-white-red (diverging)', stops: ['#2166ac', '#f7f7f7', '#b2182b'], div: true }
    };
    const at = (st, t) => { const n = st.length - 1, i = Math.min(n - 1, Math.floor(t * n)), u = t * n - i; return mixA(hexRGB(st[i]), hexRGB(st[i + 1]), u); };
    const { view, ctl } = AI.labShell(root, 'Sequential and diverging colour ramps', 'a field coloured four ways, with its greyscale twin and the ramp luminance (illustrative)',
      `<canvas style="height:420px"></canvas>`,
      `<label>Ramp <select id="rp"><option value="viridis">viridis (sequential)</option><option value="rainbow">rainbow / jet</option><option value="grey">grey</option><option value="diverge">blue-white-red (diverging)</option></select></label>
       <label>Field z(x, y) = <input type="text" id="f" value="exp(-(x^2+y^2)) - 0.6*exp(-((x-1)^2+(y+1)^2))"></label>${errBox('err')}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(false)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">x and y run from −2 to 2. Play slides a probe along the ramp: cells whose value matches the probe get a white outline, so you see where that colour lives in the field. Left: coloured field. Right: the same field in greyscale. Bottom: the ramp and its luminance curve.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const rp = q(ctl, '#rp'), spd = q(ctl, '#spd'), out = q(ctl, '#out');
    let fn = null, grid = null, tp = 0.5, playing = false, last = '';
    const NX = 36, NY = 24;
    const ei = AI.exprInput(q(ctl, '#f'), q(ctl, '#err'), { allowed: ['x', 'y'], onChange: (f) => { fn = f; compute(); last = ''; } });
    const compute = () => {
      const v = []; let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) { const z = fn({ x: -2 + (4 * (i + 0.5)) / NX, y: 2 - (4 * (j + 0.5)) / NY }); v.push(isFinite(z) ? z : 0); }
      v.forEach((z) => { lo = Math.min(lo, z); hi = Math.max(hi, z); });
      grid = { v, lo, hi };
    };
    const norm = (z, div) => { if (div) { const m = Math.max(Math.abs(grid.lo), Math.abs(grid.hi)) || 1; return clamp(0.5 + (0.5 * z) / m, 0, 1); } return grid.hi === grid.lo ? 0.5 : (z - grid.lo) / (grid.hi - grid.lo); };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); if (!grid) return;
      const R = RAMPS[rp.value], pw = (W - 30) / 2, ph = Math.min(230, pw * 0.67), cw = pw / NX, ch = ph / NY, y0 = 26;
      txt(g, 'Coloured field', 8, 16, c.fg, 'bold 12px sans-serif'); txt(g, 'Same field in greyscale', 22 + pw, 16, c.fg, 'bold 12px sans-serif');
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        const t = norm(grid.v[j * NX + i], R.div), a = at(R.stops, t);
        g.fillStyle = rgbs(a); g.fillRect(8 + i * cw, y0 + j * ch, cw + 0.5, ch + 0.5);
        const gv = gam(lum(a)); g.fillStyle = `rgb(${gv},${gv},${gv})`; g.fillRect(22 + pw + i * cw, y0 + j * ch, cw + 0.5, ch + 0.5);
        if (Math.abs(t - tp) < 0.025) { g.strokeStyle = '#ffffff'; g.lineWidth = 1; g.strokeRect(8 + i * cw + 0.5, y0 + j * ch + 0.5, cw - 1, ch - 1); g.strokeStyle = '#000000'; g.strokeRect(8 + i * cw + 1.5, y0 + j * ch + 1.5, cw - 3, ch - 3); }
      }
      // ramp strip + luminance curve
      const sy = y0 + ph + 26, sh = 24, sx0 = 8, sx1 = W - 8, N = 96; let rev = 0, prev = 0, ls = [];
      for (let k = 0; k < N; k++) { const t = k / (N - 1), a = at(R.stops, t); g.fillStyle = rgbs(a); g.fillRect(sx0 + ((sx1 - sx0) * k) / N, sy, (sx1 - sx0) / N + 1, sh); ls.push(lum(a)); }
      for (let k = 1; k < N; k++) { const d = ls[k] - ls[k - 1]; if (Math.abs(d) > 1e-4) { const s = Math.sign(d); if (prev && s !== prev) rev++; prev = s; } }
      const cy0 = sy + sh + 14, cht = H - cy0 - 14;
      txt(g, 'luminance along the ramp (0 dark, 1 light)', sx0, cy0 - 3, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.lineWidth = 1; g.strokeRect(sx0, cy0, sx1 - sx0, cht);
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); ls.forEach((l, k) => { const x = sx0 + ((sx1 - sx0) * k) / (N - 1), y = cy0 + cht - l * cht; k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
      const px = sx0 + (sx1 - sx0) * tp; g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(px, sy - 4); g.lineTo(px, cy0 + cht); g.stroke();
      const pa = at(R.stops, tp);
      out.innerHTML = `${R.name}. Probe at t = ${tp.toFixed(2)}: colour ${toHex(pa)}, luminance ${lum(pa).toFixed(3)}. Luminance direction changes: <b>${rev}</b> ${R.div ? '(a diverging ramp is dark, light, dark by design: one change)' : rev === 0 ? '(monotonic: lighter always means larger)' : '(NOT monotonic: in greyscale or to a colour-blind viewer, different values look alike)'}.`;
      q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    rp.onchange = () => { last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { playing = false; tp = 0.5; q(ctl, '#play').textContent = '▶ Play'; last = ''; };
    o.onresize = () => { last = ''; };
    ei.run();
    AI.loop(root, (dt) => {
      if (playing) { tp += dt * 0.08 * +spd.value; if (tp > 1) tp = 0; }
      const key = [tp.toFixed(3), rp.value, q(ctl, '#f').value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 4c. contrast checker ================= */
  AI.labs['ds-vz-contrast'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Contrast ratio checker', 'WCAG 2.x relative luminance: ratio = (L1 + 0.05) / (L2 + 0.05)',
      `<canvas style="height:300px"></canvas>`,
      `<label>Foreground (text or mark) hex<input type="text" id="fg" value="#E69F00"></label>
       <label>Background hex<input type="text" id="bg" value="#FFFFFF"></label>${errBox('err')}
       <div class="chips" id="sw"></div>
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       <div class="row"><button class="btn primary" id="play">▶ Sweep</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption">Click an Okabe-Ito swatch to test it. Sweep slowly fades the foreground from the background colour towards black or white and shows where it passes 3:1 (graphics, large text) and 4.5:1 (normal text).</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const fg = q(ctl, '#fg'), bg = q(ctl, '#bg'), spd = q(ctl, '#spd'), out = q(ctl, '#out'), err = q(ctl, '#err'), sw = q(ctl, '#sw');
    const OI = [['black', '#000000'], ['orange', '#E69F00'], ['sky blue', '#56B4E9'], ['bluish green', '#009E73'], ['yellow', '#F0E442'], ['blue', '#0072B2'], ['vermillion', '#D55E00'], ['reddish purple', '#CC79A7']];
    sw.innerHTML = OI.map(([n, h]) => `<span class="chip" data-h="${h}" style="border-left:12px solid ${h}">${n}</span>`).join('');
    let u = -1, playing = false, last = '';
    const valid = (s) => /^#[0-9a-f]{6}$/i.test(s.trim());
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      if (!valid(fg.value) || !valid(bg.value)) { err.textContent = 'Use six-digit hex colours like #E69F00.'; return; }
      err.textContent = '';
      const B = hexRGB(bg.value.trim()), far = lum(B) > 0.18 ? [0, 0, 0] : [255, 255, 255];
      const F = u >= 0 ? mixA(B, far, u) : hexRGB(fg.value.trim());
      const r = ratio(F, B);
      g.fillStyle = rgbs(B); g.fillRect(0, 0, W, H);
      g.fillStyle = rgbs(F); g.font = 'bold 30px sans-serif'; g.textAlign = 'left'; g.fillText('Weekly revenue is up 12%', 18, 56);
      g.font = '15px sans-serif'; g.fillText('Normal-size body text: axis labels, legends and footnotes need 4.5 : 1.', 18, 92);
      g.fillRect(18, 118, 70, 40); g.beginPath(); g.arc(130, 138, 20, 0, 7); g.fill(); g.lineWidth = 4; g.strokeStyle = rgbs(F); g.beginPath(); g.moveTo(170, 158); g.lineTo(215, 118); g.lineTo(260, 146); g.lineTo(310, 112); g.stroke();
      g.font = '12px sans-serif'; g.fillStyle = rgbs(F); g.fillText('bar, dot and line marks need 3 : 1', 18, 184);
      const pass = (t) => (r >= t ? 'pass' : 'FAIL');
      // meter
      const mx0 = 18, mx1 = W - 18, my = H - 54, sc = (v) => mx0 + ((mx1 - mx0) * Math.min(v, 21)) / 21;
      g.fillStyle = 'rgba(128,128,128,0.35)'; g.fillRect(mx0, my, mx1 - mx0, 12);
      g.fillStyle = r >= 4.5 ? '#2da44e' : r >= 3 ? '#d4a72c' : '#cf222e'; g.fillRect(mx0, my, sc(r) - mx0, 12);
      [3, 4.5].forEach((v) => { g.fillStyle = rgbs(far.map((x) => x)); g.fillRect(sc(v) - 1, my - 4, 2, 20); });
      g.font = '11px sans-serif'; g.fillStyle = rgbs(far); g.fillText('3', sc(3) - 3, my + 30); g.fillText('4.5', sc(4.5) - 6, my + 30); g.fillText('21', mx1 - 12, my + 30);
      out.innerHTML = `Foreground ${toHex(F)} on ${toHex(B)}: ratio <b>${r.toFixed(2)} : 1</b>. Normal text (4.5) <b>${pass(4.5)}</b> · large text and graphics (3) <b>${pass(3)}</b>.`;
      q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    sw.onclick = (e) => { const h = e.target.getAttribute && e.target.getAttribute('data-h'); if (h) { fg.value = h; u = -1; playing = false; q(ctl, '#play').textContent = '▶ Sweep'; last = ''; } };
    fg.oninput = () => { u = -1; last = ''; }; bg.oninput = () => { last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { if (u < 0 || u >= 1) u = 0; playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Sweep'; };
    q(ctl, '#reset').onclick = () => { u = -1; playing = false; fg.value = '#E69F00'; bg.value = '#FFFFFF'; q(ctl, '#play').textContent = '▶ Sweep'; last = ''; };
    o.onresize = () => { last = ''; };
    AI.loop(root, (dt) => {
      if (playing) { u += dt * 0.1 * +spd.value; if (u >= 1) { u = 1; playing = false; q(ctl, '#play').textContent = '▶ Sweep'; } }
      const key = [u.toFixed(3), fg.value, bg.value, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 5. a dashboard refreshing as a filter changes ================= */
  AI.labs['ds-vz-dash'] = function (root) {
    const REG = ['North', 'South', 'East'], MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const REV = { North: [50, 48, 45, 43, 40, 38], South: [20, 28, 36, 44, 52, 60], East: [30, 31, 29, 32, 30, 31] };
    const HW = { North: 0.7, South: 0.4, East: 0.5 }, AOV = { North: 250, South: 125, East: 200 };
    const { view, ctl } = AI.labShell(root, 'A dashboard that follows its filters', 'revenue in thousands of dollars, 6 months, 3 regions, 2 product lines (illustrative)',
      `<canvas style="height:380px"></canvas>`,
      `<div><b style="font-size:12px">Region</b> <span class="chips" id="rg"></span></div>
       <div style="margin-top:6px"><b style="font-size:12px">Product</b> <span class="chips" id="pr"></span></div>
       <label>Monthly revenue goal (thousand)<input type="text" id="goal" value="110"></label>${errBox('err')}
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       <div class="row"><button class="btn primary" id="play">▶ Tour</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption" id="cap">Click a region or product chip (or a region bar) to filter. Tour walks through filters, one every five seconds at speed 1, and every tile and chart eases to its new values.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const spd = q(ctl, '#spd'), out = q(ctl, '#out'), cap = q(ctl, '#cap'), goalIn = q(ctl, '#goal'), err = q(ctl, '#err');
    const mkChips = (el, items) => { el.innerHTML = items.map((n) => `<span class="chip" data-v="${n}">${n}</span>`).join(''); };
    mkChips(q(ctl, '#rg'), ['All'].concat(REG)); mkChips(q(ctl, '#pr'), ['All', 'Hardware', 'Software']);
    let reg = 'All', prd = 'All', goal = 110, tour = false, tt = 0, ti = 0, last = '', moving = true;
    const TOUR = [['All', 'All'], ['North', 'All'], ['South', 'All'], ['East', 'All'], ['All', 'Hardware'], ['All', 'Software'], ['South', 'Software']];
    const cf = (r) => (prd === 'All' ? 1 : prd === 'Hardware' ? HW[r] : 1 - HW[r]);
    const target = () => {
      const mon = MON.map((_, m) => REG.reduce((s, r) => s + (reg === 'All' || reg === r ? REV[r][m] * cf(r) : 0), 0));
      const rev = mon.reduce((a, b) => a + b, 0), ord = REG.reduce((s, r) => s + (reg === 'All' || reg === r ? REV[r].reduce((a, b) => a + b, 0) * cf(r) * 1000 / AOV[r] : 0), 0);
      const byReg = REG.map((r) => REV[r].reduce((a, b) => a + b, 0) * cf(r));
      return { mon, rev, ord, aov: ord ? (rev * 1000) / ord : 0, gr: mon[0] ? (mon[5] / mon[0] - 1) * 100 : 0, byReg };
    };
    let T = target(), D = { mon: T.mon.slice(), rev: T.rev, ord: T.ord, aov: T.aov, gr: T.gr, byReg: T.byReg.slice() };
    const sync = () => {
      root.querySelectorAll('#rg .chip').forEach((e) => (e.style.background = e.dataset.v === reg ? c.blue : '', e.style.color = e.dataset.v === reg ? '#fff' : ''));
      root.querySelectorAll('#pr .chip').forEach((e) => (e.style.background = e.dataset.v === prd ? c.blue : '', e.style.color = e.dataset.v === prd ? '#fff' : ''));
    };
    const apply = (r, p) => { reg = r; prd = p; T = target(); moving = true; sync(); };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const tw = (W - 30) / 4, tiles = [['Revenue', '$' + Math.round(D.rev) + 'k', 'six months'], ['Orders', Math.round(D.ord).toLocaleString('en-US'), 'six months'], ['Average order', '$' + Math.round(D.aov), 'revenue ÷ orders'], ['Growth', (D.gr >= 0 ? '+' : '') + D.gr.toFixed(0) + '%', 'June vs January']];
      tiles.forEach((tl, i) => {
        const x = 6 + i * (tw + 6); g.fillStyle = c.bg2; g.fillRect(x, 6, tw, 70); g.strokeStyle = c.border; g.strokeRect(x, 6, tw, 70);
        txt(g, tl[0], x + 10, 24, c.muted, '11px sans-serif');
        txt(g, tl[1], x + 10, 54, i === 3 ? (D.gr >= 0 ? c.green : c.red) : c.fg, 'bold 22px sans-serif'); txt(g, tl[2], x + 10, 70, c.muted, '10px sans-serif');
      });
      // line chart
      const lx0 = 40, lx1 = W * 0.6 - 10, top = 108, bot = H - 40, ymax = nice(Math.max(...T.mon, ...D.mon, goal, 10) * 1.1);
      txt(g, 'Revenue by month (thousand)', lx0, top - 14, c.fg, 'bold 12px sans-serif');
      g.strokeStyle = c.border; g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { const y = bot - ((bot - top) * i) / 4; g.beginPath(); g.moveTo(lx0, y); g.lineTo(lx1, y); g.stroke(); txt(g, AI.fmt((ymax * i) / 4, 3), lx0 - 6, y + 4, c.muted, '11px sans-serif', 'right'); }
      const px = (i) => lx0 + 14 + ((lx1 - lx0 - 28) * i) / 5, py = (v) => bot - (v / ymax) * (bot - top);
      g.setLineDash([5, 4]); g.strokeStyle = c.yellow; g.beginPath(); g.moveTo(lx0, py(goal)); g.lineTo(lx1, py(goal)); g.stroke(); g.setLineDash([]);
      txt(g, 'goal ' + goal, lx1, py(goal) - 4, c.yellow, '11px sans-serif', 'right');
      g.strokeStyle = c.blue; g.lineWidth = 2.5; g.beginPath(); D.mon.forEach((v, i) => (i ? g.lineTo(px(i), py(v)) : g.moveTo(px(i), py(v)))); g.stroke();
      D.mon.forEach((v, i) => { g.fillStyle = c.blue; g.beginPath(); g.arc(px(i), py(v), 3.5, 0, 7); g.fill(); txt(g, MON[i], px(i), bot + 16, c.muted, '11px sans-serif', 'center'); });
      // region bars
      const bx0 = W * 0.6 + 30, bx1 = W - 14, byMax = nice(Math.max(...D.byReg, ...T.byReg, 10) * 1.1), slot = (bx1 - bx0) / 3;
      txt(g, 'By region (click to drill down)', bx0 - 20, top - 14, c.fg, 'bold 12px sans-serif');
      REG.forEach((r, i) => {
        const h = (D.byReg[i] / byMax) * (bot - top), sel = reg === 'All' || reg === r;
        g.fillStyle = sel ? c.orange : c.muted; g.globalAlpha = sel ? 1 : 0.5; g.fillRect(bx0 + slot * i + slot * 0.15, bot - h, slot * 0.7, h); g.globalAlpha = 1;
        txt(g, r, bx0 + slot * (i + 0.5), bot + 16, c.fg, '12px sans-serif', 'center'); txt(g, String(Math.round(D.byReg[i])), bx0 + slot * (i + 0.5), bot - h - 6, c.fg, '11px sans-serif', 'center');
      });
      const above = T.mon.filter((v) => v >= goal).length;
      out.innerHTML = `Filter: <b>${reg}</b> / <b>${prd}</b>. Revenue $${AI.fmt(T.rev, 4)}k over six months; ${above} of 6 months at or above the $${goal}k goal. Growth January to June: <b>${(T.gr >= 0 ? '+' : '') + T.gr.toFixed(1)}%</b>.`;
      q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    const chipClick = (e) => { const v = e.target.dataset && e.target.dataset.v; if (!v) return; tour = false; q(ctl, '#play').textContent = '▶ Tour'; if (e.currentTarget.id === 'rg') apply(v, prd); else apply(reg, v); };
    q(ctl, '#rg').onclick = chipClick; q(ctl, '#pr').onclick = chipClick;
    cv.addEventListener('click', (e) => {
      const rc = cv.getBoundingClientRect(), x = e.clientX - rc.left, W = o.w, bx0 = W * 0.6 + 30, slot = (W - 14 - bx0) / 3, i = Math.floor((x - bx0) / slot);
      if (i >= 0 && i < 3 && e.clientY - rc.top > 100) { tour = false; q(ctl, '#play').textContent = '▶ Tour'; apply(reg === REG[i] ? 'All' : REG[i], prd); }
    });
    goalIn.oninput = () => { const v = +goalIn.value; if (!(v > 0)) { err.textContent = 'Enter a positive number.'; return; } err.textContent = ''; goal = v; T = target(); moving = true; };
    q(ctl, '#play').onclick = (e) => { tour = !tour; tt = 0; e.target.textContent = tour ? '⏸ Pause tour' : '▶ Tour'; if (tour) { ti = 0; apply(...TOUR[0]); } };
    q(ctl, '#reset').onclick = () => { tour = false; q(ctl, '#play').textContent = '▶ Tour'; apply('All', 'All'); };
    spd.oninput = () => { moving = true; };
    const NOTE = { 'All/All': 'All regions: revenue is rising, up 29% from January to June.', 'North/All': 'Filter to North and the story reverses: revenue is falling.', 'South/All': 'South is the growth: it tripled from 20 to 60.', 'East/All': 'East is flat.', 'All/Hardware': 'Hardware only: the totals shift because the product mix differs by region.', 'All/Software': 'Software only: a different mix of regions, so a different shape.', 'South/Software': 'Drill down: South software, a small slice of the business.' };
    cap.textContent = NOTE['All/All'];
    o.onresize = () => { moving = true; };
    sync();
    AI.loop(root, (dt) => {
      if (tour) { tt += dt * +spd.value; if (tt > 5) { tt = 0; ti = (ti + 1) % TOUR.length; apply(...TOUR[ti]); } }
      if (moving) {
        const k = 1 - Math.exp(-dt * 1.6 * +spd.value); let d = 0;
        const st = (a, b) => { const n = a + (b - a) * k; d = Math.max(d, Math.abs(b - n)); return Math.abs(b - n) < 0.01 ? b : n; };
        D.mon = D.mon.map((v, i) => st(v, T.mon[i])); D.byReg = D.byReg.map((v, i) => st(v, T.byReg[i]));
        D.rev = st(D.rev, T.rev); D.ord = st(D.ord, T.ord); D.aov = st(D.aov, T.aov); D.gr = st(D.gr, T.gr);
        cap.textContent = NOTE[reg + '/' + prd] || ('Filter ' + reg + ' / ' + prd + '.');
        if (d < 0.01) moving = false; last = '';
      }
      const key = [moving, D.rev.toFixed(2), D.gr.toFixed(2), reg, prd, goal, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };

  /* ================= 6. from chart to argument ================= */
  AI.labs['ds-vz-story'] = function (root) {
    const P1 = 0.048, P2 = 0.061;
    const { view, ctl } = AI.labShell(root, 'From chart to argument', 'an A/B test of a sign-up button: conversion rate 4.8% (A) versus 6.1% (B), illustrative',
      `<canvas style="height:400px"></canvas>`,
      `${sl('n', 'Users per variant (n)', 200, 8000, 100, 1000)}
       <label>Your headline (blank = automatic)<input type="text" id="hl" placeholder="type a headline"></label>
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 1)}
       ${btns(true)}
       <div class="readout" id="out" style="white-space:normal"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const nIn = q(ctl, '#n'), hl = q(ctl, '#hl'), spd = q(ctl, '#spd'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let t = 0, playing = false, last = '';
    const CAPS = ['Stage 1 of 5. The chart as a tool leaves it: a generic title, two equal-looking bars. The reader must work out what matters.',
      'Stage 2 of 5. Emphasis: colour only the bar the argument is about; the other fades to the background.',
      'Stage 3 of 5. Annotation: put the difference on the chart in numbers, so nobody has to subtract bar heights.',
      'Stage 4 of 5. Headline: replace the topic title with the finding, in a sentence, that the data really supports.',
      'Stage 5 of 5. Uncertainty: add the 95% intervals. The difference panel shows whether zero is a plausible value.'];
    const stats = () => {
      const n = +nIn.value, se1 = Math.sqrt(P1 * (1 - P1) / n), se2 = Math.sqrt(P2 * (1 - P2) / n), d = P2 - P1, se = Math.sqrt(se1 * se1 + se2 * se2);
      return { n, se1, se2, d, se, lo: d - 1.96 * se, hi: d + 1.96 * se };
    };
    const pp = (v) => (v * 100).toFixed(1);
    const headline = (s) => (s.lo > 0 ? `Variant B converts ${pp(s.d)} points better than A (95% CI ${pp(s.lo)} to ${pp(s.hi)})` : s.hi < 0 ? `Variant B converts worse than A` : `Too early to call: B is ${pp(s.d)} points ahead, but the interval (${pp(s.lo)} to ${pp(s.hi)}) includes zero`);
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const s = stats(), p = [0, 1, 2, 3, 4].map((k) => ease((t - k) / 0.6));
      const title = hl.value.trim() || headline(s);
      g.globalAlpha = 1 - p[3]; txt(g, 'Conversion rate by variant', 12, 24, c.fg, 'bold 16px sans-serif'); g.globalAlpha = p[3];
      let fs = 16; g.font = `bold ${fs}px sans-serif`; while (g.measureText(title).width > W - 24 && fs > 10) { fs--; g.font = `bold ${fs}px sans-serif`; }
      txt(g, title, 12, 24, c.fg, `bold ${fs}px sans-serif`); g.globalAlpha = 1;
      const lw = W * 0.58, x0 = 44, x1 = lw - 10, top = 60, bot = H - 40, ymax = 0.08;
      const py = (v) => bot - (v / ymax) * (bot - top);
      g.strokeStyle = c.border; g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) { const v = (ymax * i) / 4; g.beginPath(); g.moveTo(x0, py(v)); g.lineTo(x1, py(v)); g.stroke(); txt(g, (v * 100).toFixed(0) + '%', x0 - 6, py(v) + 4, c.muted, '11px sans-serif', 'right'); }
      const cx = [x0 + (x1 - x0) * 0.27, x0 + (x1 - x0) * 0.73], bw = (x1 - x0) * 0.28, vals = [P1, P2], ses = [s.se1, s.se2];
      vals.forEach((v, i) => {
        const col = i === 1 ? AI.mix(c.muted, c.orange, p[1]) : AI.mix(c.blue, c.muted, p[1]);
        g.fillStyle = col; g.globalAlpha = 0.9; g.fillRect(cx[i] - bw / 2, py(v), bw, bot - py(v)); g.globalAlpha = 1;
        txt(g, 'Variant ' + 'AB'[i], cx[i], bot + 16, c.fg, '12px sans-serif', 'center'); txt(g, (v * 100).toFixed(1) + '%', cx[i], py(v) + 18, c.bg, 'bold 13px sans-serif', 'center');
        if (p[4] > 0) { const a = py(v + 1.96 * ses[i] * p[4]), b2 = py(Math.max(0, v - 1.96 * ses[i] * p[4])); g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(cx[i], a); g.lineTo(cx[i], b2); g.moveTo(cx[i] - 8, a); g.lineTo(cx[i] + 8, a); g.moveTo(cx[i] - 8, b2); g.lineTo(cx[i] + 8, b2); g.stroke(); }
      });
      if (p[2] > 0) {
        g.globalAlpha = p[2]; g.strokeStyle = c.orange; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(cx[0] + bw / 2, py(P1)); g.lineTo(cx[1] + bw / 2, py(P1)); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(cx[1] + bw / 2 + 8, py(P1)); g.lineTo(cx[1] + bw / 2 + 8, py(P2)); g.stroke();
        txt(g, '+' + pp(s.d) + ' points', cx[1] + bw / 2 - 6, py(P2) - 22 - (p[4] > 0 ? 0 : 0), c.orange, 'bold 13px sans-serif', 'center'); g.globalAlpha = 1;
      }
      // difference panel
      const dx0 = lw + 24, dx1 = W - 16, dm = 0.045, dx = (v) => dx0 + ((v + dm) / (2 * dm)) * (dx1 - dx0), my = (top + bot) / 2;
      g.globalAlpha = p[4]; txt(g, 'Difference B − A (percentage points)', dx0, top - 14, c.fg, 'bold 12px sans-serif');
      g.strokeStyle = c.border; g.lineWidth = 1; g.beginPath(); g.moveTo(dx0, my + 30); g.lineTo(dx1, my + 30); g.stroke();
      [-0.04, -0.02, 0, 0.02, 0.04].forEach((v) => { g.beginPath(); g.moveTo(dx(v), my + 26); g.lineTo(dx(v), my + 34); g.stroke(); txt(g, (v * 100).toFixed(0), dx(v), my + 48, c.muted, '11px sans-serif', 'center'); });
      g.strokeStyle = c.red; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(dx(0), top + 10); g.lineTo(dx(0), my + 30); g.stroke(); g.setLineDash([]);
      txt(g, 'no difference', dx(0), top + 6, c.red, '11px sans-serif', 'center');
      const half = 1.96 * s.se * p[4], okc = s.lo > 0 ? c.green : c.yellow;
      g.strokeStyle = okc; g.lineWidth = 3; g.beginPath(); g.moveTo(dx(s.d - half), my); g.lineTo(dx(s.d + half), my); g.moveTo(dx(s.d - half), my - 7); g.lineTo(dx(s.d - half), my + 7); g.moveTo(dx(s.d + half), my - 7); g.lineTo(dx(s.d + half), my + 7); g.stroke();
      g.fillStyle = c.fg; g.beginPath(); g.arc(dx(s.d), my, 5, 0, 7); g.fill();
      txt(g, `${pp(s.lo)} to ${pp(s.hi)}`, (dx(s.lo) + dx(s.hi)) / 2, my - 14, c.fg, '12px sans-serif', 'center'); g.globalAlpha = 1;
      out.innerHTML = `n = ${s.n} per variant. Difference = ${P2} − ${P1} = ${s.d.toFixed(3)}; standard error = ${s.se.toFixed(5)}; 95% interval ${s.lo.toFixed(4)} to ${s.hi.toFixed(4)} (${(s.lo > 0 ? 'excludes' : s.hi < 0 ? 'excludes' : 'includes')} zero).`;
      cap.textContent = CAPS[clamp(Math.floor(t + 1e-9), 0, 4)];
      q(ctl, '#nv').textContent = String(s.n); q(ctl, '#spdv').textContent = (+spd.value).toFixed(2) + '×';
    };
    nIn.oninput = () => { last = ''; }; hl.oninput = () => { last = ''; }; spd.oninput = () => { last = ''; };
    q(ctl, '#play').onclick = (e) => { if (t >= 4.99) t = 0; playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; t = Math.min(5, Math.floor(t + 1e-9) + 1); last = ''; };
    q(ctl, '#reset').onclick = () => { t = 0; playing = false; q(ctl, '#play').textContent = '▶ Play'; last = ''; };
    o.onresize = () => { last = ''; };
    AI.loop(root, (dt) => {
      if (playing) { t = Math.min(5, t + dt * 0.25 * +spd.value); if (t >= 5) { playing = false; q(ctl, '#play').textContent = '▶ Play'; } }
      const key = [t.toFixed(3), nIn.value, hl.value, spd.value, o.w].join('|');
      if (key !== last) { last = key; draw(); }
    }, { fps: 30 });
  };
})();
