/* Data Science Module 0 labs: the loop, data types, tidy data and unit of analysis, seeds and defaults, answerable questions.
   All animations run on dt, start slowly, and have speed, pause/step and reset controls. Simulations are illustrative. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const qa = (r, s) => Array.from(r.querySelectorAll(s));
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const sm = (t) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const wrap = (g, s, maxW) => {
    const words = s.split(/\s+/), lines = []; let cur = '';
    words.forEach((w) => { const t = cur ? cur + ' ' + w : w; if (g.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; });
    if (cur) lines.push(cur); return lines;
  };
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

  /* ---- 1. The data-science loop ---- */
  AI.labs['ds-fnd-loop'] = function (root) {
    const STG = [
      { n: 'Ask & collect', c: 'Turn the question into columns you can measure, then gather the rows. Everything later depends on this step.' },
      { n: 'Clean & prepare', c: 'Fix types, units, duplicates and gaps so each row means one thing. Most of the real time goes here.' },
      { n: 'Explore', c: 'Plot and summarise before modelling: ranges, shapes, surprises, relationships. Look for mistakes in the data.' },
      { n: 'Test & model', c: {
        descriptive: 'Summarise: counts, means and spreads, each with its sample size.',
        diagnostic: 'Compare groups and ask whether the difference is bigger than chance alone would produce.',
        predictive: 'Fit a model and measure its error on rows it has not seen.',
        prescriptive: 'Predict the outcome of each possible action, then pick the best one under a stated cost.' } },
      { n: 'Communicate', c: 'State the answer, its uncertainty and what you assumed, in words the decision-maker can act on.' },
      { n: 'Deploy & monitor', c: 'The decision is made. Keep measuring: if the world changes the answer goes stale, and a new question starts the loop again.' }];
    const EX = {
      descriptive: 'How many iced drinks did we sell each week this summer?',
      diagnostic: 'Why did iced-drink sales jump in weeks 5 to 8?',
      predictive: 'How many iced drinks will we sell next week if it is 26 °C?',
      prescriptive: 'How many cups of ice should we order for next week?' };
    const { view, ctl } = AI.labShell(root, 'The data-science loop', 'one question travels round six stages (illustrative)',
      `<canvas style="height:380px"></canvas>`,
      `<label>Kind of question <select id="ty"><option value="descriptive">descriptive: what happened?</option><option value="diagnostic">diagnostic: why?</option><option value="predictive" selected>predictive: what will happen?</option><option value="prescriptive">prescriptive: what should we do?</option></select></label>
       <label>Your question (edit it) <input type="text" id="qs" style="width:100%"></label>
       ${sl('sp', 'Speed', 0.2, 2, 0.1, 0.6)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" style="white-space:normal" id="out"></div>
       <p class="caption">The token is the question. It waits at each stage while the caption says what happens there, then moves on. After deployment the loop starts again with a new question.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const ty = q(ctl, '#ty'), qs = q(ctl, '#qs'), sp = q(ctl, '#sp'), out = q(ctl, '#out');
    qs.value = EX.predictive;
    let ph = 0, tgt = Infinity, playing = true, laps = 0;
    const stageText = (k) => { const s = STG[k]; return typeof s.c === 'string' ? s.c : s.c[ty.value]; };
    const pos = (k, W, H) => { const R = Math.min(W * 0.18, H * 0.34), a = -Math.PI / 2 + (k * Math.PI) / 3; return [W / 2 + R * Math.cos(a), H / 2 + R * Math.sin(a), R, a]; };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const k = Math.floor(ph) % 6, f = ph - Math.floor(ph), s = f < 0.6 ? 0 : sm((f - 0.6) / 0.4);
      const R = pos(0, W, H)[2];
      g.strokeStyle = c.border; g.lineWidth = 2; g.beginPath(); g.arc(W / 2, H / 2, R, 0, 7); g.stroke();
      for (let i = 0; i < 6; i++) {
        const [x, y, , a] = pos(i, W, H), act = i === k && s === 0;
        g.fillStyle = act ? c.blue : c.bg3; g.strokeStyle = act ? c.blue : c.border; g.lineWidth = 2;
        g.beginPath(); g.arc(x, y, 22, 0, 7); g.fill(); g.stroke();
        txt(g, String(i + 1), x, y + 5, act ? '#fff' : c.fg, 'bold 14px sans-serif', 'center');
        const lx = x + Math.cos(a) * 36, ly = y + Math.sin(a) * 36 + 4, al = Math.abs(Math.cos(a)) < 0.2 ? 'center' : Math.cos(a) > 0 ? 'left' : 'right';
        txt(g, STG[i].n, lx, ly + (Math.sin(a) > 0.5 ? 8 : 0), act ? c.blue : c.fg, (act ? 'bold ' : '') + '11px sans-serif', al);
      }
      const a0 = pos(k, W, H), a1 = pos((k + 1) % 6, W, H);
      const ang = a0[3] + s * (Math.PI / 3), tx = W / 2 + R * Math.cos(ang), ty2 = H / 2 + R * Math.sin(ang);
      g.fillStyle = c.orange; g.beginPath(); g.arc(tx, ty2, 9, 0, 7); g.fill();
      txt(g, '?', tx, ty2 + 4, '#fff', 'bold 12px sans-serif', 'center');
      g.font = '13px sans-serif';
      const lines = wrap(g, qs.value || '(type a question)', R * 1.6), y0 = H / 2 - (lines.length * 17) / 2 + 6;
      txt(g, ty.value.toUpperCase(), W / 2, y0 - 22, c.muted, 'bold 11px sans-serif', 'center');
      lines.forEach((l, i) => txt(g, l, W / 2, y0 + i * 17, c.fg, '13px sans-serif', 'center'));
      txt(g, `loops completed: ${laps}`, 10, H - 10, c.muted, '11px sans-serif');
      const moving = s > 0;
      out.innerHTML = moving ? `<b>Hand-over:</b> the result of “${STG[k].n}” moves on to “${STG[(k + 1) % 6].n}”.`
        : `<b>${k + 1}. ${STG[k].n}.</b> ${stageText(k)}`;
    };
    ty.onchange = () => { qs.value = EX[ty.value]; draw(); };
    qs.oninput = draw;
    q(ctl, '#play').onclick = (e) => { playing = !playing; tgt = playing ? Infinity : ph; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; tgt = Math.floor(ph) + 1; };
    q(ctl, '#reset').onclick = () => { ph = 0; laps = 0; tgt = Infinity; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); };
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (!(playing || ph < tgt)) return;
      ph += dt * +sp.value * 0.3;
      if (!playing && ph >= tgt) ph = tgt;
      if (ph >= 6) { ph -= 6; tgt = tgt === Infinity ? tgt : tgt - 6; laps++; }
      draw();
    }, { fps: 30 });
    draw();
  };

  /* ---- 2. Types of data ---- */
  AI.labs['ds-fnd-types'] = function (root) {
    const COLS = ['customer_id', 'age', 'plan', 'satisfaction', 'signup', 'review'];
    const ROWS = [[1041, 34, 'Free', 4, '2025-01-06', 'love the app'], [1042, 52, 'Pro', 2, '2025-01-09', 'too slow on my phone'], [1043, 29, 'Free', 5, '2025-01-15', 'simple and quick'],
      [1044, 41, 'Team', 4, '2025-01-21', 'good, needs export'], [1045, 38, 'Free', 3, '2025-02-03', 'ok']];
    const KEY = ['identifier', 'numeric', 'categorical', 'ordinal', 'datetime', 'text'];
    const WHY = ['A label that only says which customer this is. Arithmetic on it means nothing.', 'A measured quantity: gaps are equal and averages make sense.',
      'A few named groups with no order.', 'Ordered levels (1 to 5), but the gap between 2 and 3 may not equal the gap between 4 and 5.', 'A point in time: you can order dates and subtract them to get a duration.',
      'Free text, an unstructured column: it must be turned into numbers (length, word counts, embeddings) before it can be summarised.'];
    const TYPES = ['identifier', 'numeric', 'categorical', 'ordinal', 'datetime', 'text'];
    const NO = (w) => ({ l: 'no', w });
    const V = {
      '?': { mean: NO('Pick a type for this column first.'), median: NO('Pick a type for this column first.'), mode: NO('Pick a type for this column first.'), range: NO('Pick a type for this column first.') },
      identifier: { mean: NO('An ID is a label; the average of labels is not a customer.'), median: NO('IDs have no meaningful middle.'), mode: NO('Every ID is unique, so there is no most common one.'), range: NO('The gap between two labels measures nothing.') },
      numeric: { mean: { l: 'ok', w: 'Equal gaps, so a mean is meaningful.' }, median: { l: 'ok', w: 'The middle value of the sorted numbers.' }, mode: { l: 'caution', w: 'Measured values rarely repeat, so a mode is weak.' }, range: { l: 'ok', w: 'Largest minus smallest.' } },
      categorical: { mean: NO('Categories have no numeric size to average.'), median: NO('With no order there is no middle.'), mode: { l: 'ok', w: 'The most frequent category is the natural summary.' }, range: NO('Names cannot be subtracted.') },
      ordinal: { mean: { l: 'caution', w: 'Computable, but it assumes the steps between levels are equal.' }, median: { l: 'ok', w: 'Order is all a median needs.' }, mode: { l: 'ok', w: 'The most common level.' }, range: { l: 'caution', w: 'Counts steps between levels; do not read it as a distance.' } },
      datetime: { mean: { l: 'caution', w: 'A midpoint date; rarely useful.' }, median: { l: 'ok', w: 'The middle date.' }, mode: { l: 'caution', w: 'Exact timestamps seldom repeat.' }, range: { l: 'ok', w: 'Latest minus earliest is a duration.' } },
      text: { mean: NO('Text must first be turned into numbers.'), median: NO('Text must first be turned into numbers.'), mode: NO('Whole sentences rarely repeat.'), range: NO('Text must first be turned into numbers.') } };
    const day = (s) => Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10)) / 864e5;
    const dstr = (d) => new Date(d * 864e5).toISOString().slice(0, 10);
    const val = (ci, op) => {
      const raw = ROWS.map((r) => r[ci]);
      if (op === 'mode') {
        const m = {}; raw.forEach((x) => (m[x] = (m[x] || 0) + 1)); const mx = Math.max(...Object.values(m));
        return mx === 1 ? 'no value repeats' : Object.keys(m).filter((k) => m[k] === mx).join(' / ') + ` (${mx} of ${raw.length})`;
      }
      let a = raw;
      if (ci === 4) a = raw.map(day);
      if (a.some((x) => typeof x !== 'number')) return 'cannot compute: the values are words';
      const s = [...a].sort((x, y) => x - y);
      let v = op === 'mean' ? mean(a) : op === 'median' ? s[(s.length - 1) >> 1] : s[s.length - 1] - s[0];
      if (ci === 4) return op === 'range' ? `${v} days` : dstr(v);
      return Number.isInteger(v) ? String(v) : v.toFixed(2);
    };
    const { view, ctl } = AI.labShell(root, 'Types of data', 'give each column a type, then see which summaries make sense (illustrative table)',
      `<div style="overflow-x:auto"><table class="tbl" id="tb" style="font-size:11px"></table></div><div class="readout" id="score" style="white-space:normal;margin-top:8px"></div>`,
      `<b>Compute a summary</b>
       <label>Column <select id="col">${COLS.map((c2, i) => `<option value="${i}">${c2}</option>`).join('')}</select></label>
       <label>Summary <select id="op"><option value="mean">mean</option><option value="median">median</option><option value="mode">mode</option><option value="range">range (max − min)</option></select></label>
       <div class="readout" style="white-space:normal" id="res"></div>
       ${sl('sp', 'Tour speed', 0.3, 2, 0.1, 0.7)}
       <div class="row"><button class="btn primary" id="tour">▶ Guided tour</button><button class="btn" id="clr">Clear labels</button></div>
       <p class="caption">Choose a type in each column header. The summary verdict follows <i>your</i> label, so a wrong label shows its consequence. The tour labels the columns one by one and explains why.</p>`);
    const tb = q(view, '#tb'), score = q(view, '#score'), res = q(ctl, '#res'), sp = q(ctl, '#sp');
    tb.innerHTML = `<thead><tr>${COLS.map((c2) => `<th>${c2}</th>`).join('')}</tr><tr>${COLS.map((c2, i) => `<th data-c="${i}"><select data-i="${i}" style="font-size:10px;max-width:78px;padding:2px"><option value="?">type?</option>${TYPES.map((t) => `<option>${t}</option>`).join('')}</select></th>`).join('')}</tr></thead>
      <tbody>${ROWS.map((r) => `<tr>${r.map((v, i) => `<td data-c="${i}">${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody>`;
    const sels = qa(tb, 'select');
    let hi = -1, tour = -1, tt = 0;
    const colour = (i) => qa(tb, `[data-c="${i}"]`).forEach((e) => (e.style.background = i === hi ? 'rgba(88,166,255,.18)' : ''));
    const upd = () => {
      COLS.forEach((x, i) => colour(i));
      const got = sels.filter((s, i) => s.value === KEY[i]).length, set = sels.filter((s) => s.value !== '?').length;
      let msg = `${set} of 6 columns labelled; ${got} match the answer key.`;
      if (tour >= 0) msg = `<b>${COLS[tour]}: ${KEY[tour]}.</b> ${WHY[tour]}`;
      else if (set === 6 && got === 6) msg += ' All correct. Note that “text” is the only unstructured column here.';
      else if (set > 0) { const bad = sels.findIndex((s, i) => s.value !== '?' && s.value !== KEY[i]); if (bad >= 0) msg += ` Look again at <b>${COLS[bad]}</b>: ${WHY[bad]}`; }
      score.innerHTML = msg;
      const ci = +q(ctl, '#col').value, op = q(ctl, '#op').value, t = sels[ci].value, v = V[t][op];
      const icon = v.l === 'ok' ? '✔ sensible' : v.l === 'caution' ? '⚠ use with care' : '✘ not meaningful';
      res.innerHTML = `${op} of <b>${COLS[ci]}</b> labelled “${t === '?' ? 'no type' : t}”: <b>${icon}</b>. ${v.w}<br>Arithmetic result: ${esc(val(ci, op))}`;
    };
    sels.forEach((s) => (s.onchange = () => { tour = -1; upd(); }));
    q(ctl, '#col').onchange = upd; q(ctl, '#op').onchange = upd;
    q(ctl, '#clr').onclick = () => { sels.forEach((s) => (s.value = '?')); tour = -1; hi = -1; q(ctl, '#tour').textContent = '▶ Guided tour'; upd(); };
    q(ctl, '#tour').onclick = (e) => {
      if (tour >= 0) { tour = -1; hi = -1; e.target.textContent = '▶ Guided tour'; upd(); return; }
      sels.forEach((s) => (s.value = '?')); tour = 0; tt = 0; hi = 0; sels[0].value = KEY[0]; e.target.textContent = '■ Stop tour'; upd();
    };
    AI.loop(root, (dt) => {
      if (tour < 0) return;
      tt += dt * +sp.value;
      if (tt >= 4) {
        tt = 0; tour++;
        if (tour >= 6) { tour = -1; hi = -1; q(ctl, '#tour').textContent = '▶ Guided tour'; upd(); return; }
        hi = tour; sels[tour].value = KEY[tour]; upd();
      }
    }, { fps: 20 });
    upd();
  };

  /* ---- 3. Tidy data: wide to long ---- */
  AI.labs['ds-fnd-tidy'] = function (root) {
    const PN = ['Tea', 'Coffee', 'Cake', 'Juice'], MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const BASE = [[120, 135, 150], [200, 210, 260], [80, 95, 90]];
    const V = (p, m) => (p < 3 && m < 3 ? BASE[p][m] : 70 + ((p * 41 + m * 29 + 13) % 120));
    const { view, ctl } = AI.labShell(root, 'From wide to tidy (long)', 'each month column becomes values of one “month” variable (illustrative shop sales)',
      `<canvas style="height:400px"></canvas>`,
      `${sl('np', 'Products', 2, 4, 1, 3)}${sl('nm', 'Month columns', 2, 6, 1, 3)}${sl('sp', 'Speed', 0.2, 2, 0.1, 0.8)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Replay</button></div>
       <div class="readout" style="white-space:normal" id="out"></div>
       <p class="caption">Left: the wide table, where the month is hidden in the column headers. Right: the tidy table, one row per product and month. One value flies across at a time; its column header becomes data.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), npI = q(ctl, '#np'), nmI = q(ctl, '#nm'), sp = q(ctl, '#sp');
    let P, M, N, pr = 0, tgt = Infinity, playing = true;
    const dims = () => { P = +npI.value; M = +nmI.value; N = P * M; q(ctl, '#npv').textContent = P; q(ctl, '#nmv').textContent = M; };
    const restart = () => { dims(); pr = 0; tgt = Infinity; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); };
    const draw = () => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const cw = Math.min(54, (W * 0.5 - 24 - 62) / M), ch = 30, gx = 10, gy = 40;
      const x0 = W * 0.5 + 14, cwl = (W - x0 - 10) / 3, rh = Math.min(26, (H - 70) / (N + 1));
      const k = Math.min(N - 1, Math.floor(pr)), f = pr >= N ? 1 : pr - Math.floor(pr), e = sm(f);
      txt(g, 'wide: one row per product', gx, 18, c.muted, 'bold 12px sans-serif');
      txt(g, 'tidy: one row per observation', x0, 18, c.muted, 'bold 12px sans-serif');
      const done = pr >= N ? N : k;
      const wideXY = (p, m) => [gx + 62 + m * cw + cw / 2, gy + ch + p * ch + ch / 2];
      const longXY = (r) => [x0 + cwl * 2.5, 40 + (r + 1) * rh + rh / 2];
      g.font = '11px sans-serif';
      for (let m = 0; m < M; m++) txt(g, MN[m], gx + 62 + m * cw + cw / 2, gy + ch / 2 + 4, c.blue, 'bold 12px sans-serif', 'center');
      for (let p = 0; p < P; p++) txt(g, PN[p], gx, gy + ch + p * ch + ch / 2 + 4, c.fg, 'bold 12px sans-serif');
      ['product', 'month', 'sales'].forEach((h, i) => txt(g, h, x0 + cwl * (i + 0.5), 40 + rh / 2 + 4, c.blue, 'bold 12px sans-serif', 'center'));
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, 40 + rh); g.lineTo(W - 10, 40 + rh); g.stroke();
      for (let p = 0; p < P; p++) for (let m = 0; m < M; m++) {
        const r = p * M + m, [cx, cy] = wideXY(p, m), gone = r < done;
        g.strokeStyle = c.border; g.strokeRect(cx - cw / 2 + 1, cy - ch / 2 + 1, cw - 2, ch - 2);
        if (r === k && pr < N) { g.fillStyle = 'rgba(219,109,40,.25)'; g.fillRect(cx - cw / 2 + 1, cy - ch / 2 + 1, cw - 2, ch - 2); }
        txt(g, String(V(p, m)), cx, cy + 4, gone ? c.border : c.fg, '12px sans-serif', 'center');
      }
      for (let r = 0; r < done; r++) {
        const p = Math.floor(r / M), m = r % M, y = 40 + (r + 1) * rh + rh / 2 + 4;
        txt(g, PN[p], x0 + cwl * 0.5, y, c.fg, '11px sans-serif', 'center'); txt(g, MN[m], x0 + cwl * 1.5, y, c.fg, '11px sans-serif', 'center'); txt(g, String(V(p, m)), x0 + cwl * 2.5, y, c.fg, '11px sans-serif', 'center');
      }
      let msg;
      if (pr < N) {
        const p = Math.floor(k / M), m = k % M, [ax, ay] = wideXY(p, m), [bx, by] = longXY(k), x = ax + (bx - ax) * e, y = ay + (by - ay) * e;
        const y4 = 40 + (k + 1) * rh + rh / 2 + 4;
        g.globalAlpha = e; txt(g, PN[p], x0 + cwl * 0.5, y4, c.fg, '11px sans-serif', 'center'); txt(g, MN[m], x0 + cwl * 1.5, y4, c.orange, 'bold 11px sans-serif', 'center'); g.globalAlpha = 1;
        g.fillStyle = c.orange; g.fillRect(x - 17, y - 10, 34, 20);
        txt(g, String(V(p, m)), x, y + 4, '#fff', 'bold 12px sans-serif', 'center');
        msg = `Moving <b>${PN[p]} / ${MN[m]} = ${V(p, m)}</b> into row ${k + 1}: the header “${MN[m]}” becomes a value in the <b>month</b> column.`;
      } else {
        let tot = 0; for (let p = 0; p < P; p++) for (let m = 0; m < M; m++) tot += V(p, m);
        msg = `Done: ${P} × ${M} = <b>${N}</b> values became ${N} rows with 3 columns. Total sales are ${tot} in both layouts, so nothing was lost.`;
      }
      out.innerHTML = msg + `<br>Rows moved: ${done} of ${N}.`;
    };
    npI.oninput = restart; nmI.oninput = restart;
    q(ctl, '#play').onclick = (ev) => { playing = !playing; tgt = playing ? Infinity : pr; ev.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; tgt = Math.min(N, Math.floor(pr) + 1); };
    q(ctl, '#reset').onclick = restart;
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (pr >= N || !(playing || pr < tgt)) return;
      pr += dt * +sp.value * 0.8;
      if (!playing && pr >= tgt) pr = tgt;
      if (pr > N) pr = N;
      draw();
    }, { fps: 30 });
    dims(); draw();
  };

  /* ---- 3b. Unit of analysis ---- */
  AI.labs['ds-fnd-unit'] = function (root) {
    const ORD = [['Ana', 'Mon', 12], ['Ben', 'Mon', 8], ['Ana', 'Tue', 20], ['Cy', 'Tue', 5], ['Ben', 'Wed', 15], ['Ana', 'Wed', 10], ['Cy', 'Thu', 7], ['Ana', 'Thu', 6]];
    const { view, ctl } = AI.labShell(root, 'Choose the unit of analysis', 'the same 8 orders, regrouped: what is one row? (illustrative)',
      `<div style="display:flex;gap:14px;flex-wrap:wrap"><div style="flex:1;min-width:210px"><b>Source: one row per order</b><table class="tbl" id="src"></table></div><div style="flex:1;min-width:210px"><b id="rt">Result</b><table class="tbl" id="dst"></table></div></div>`,
      `<label>One row should be a… <select id="un"><option value="order">order (no grouping)</option><option value="cust" selected>customer</option><option value="day">day</option></select></label>
       ${sl('sp', 'Speed', 0.2, 2, 0.1, 0.7)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Replay</button></div>
       <div class="readout" style="white-space:normal" id="out"></div>
       <p class="caption">Each order is absorbed into its group, one at a time. Watch how many rows the result has and what “average per row” means for each choice.</p>`);
    const src = q(view, '#src'), dst = q(view, '#dst'), out = q(ctl, '#out'), sp = q(ctl, '#sp'), un = q(ctl, '#un');
    let i = 0, acc = 0, playing = true;
    const key = (r, idx) => (un.value === 'order' ? 'order ' + (idx + 1) : un.value === 'cust' ? r[0] : r[1]);
    const render = () => {
      const groups = [], map = {};
      for (let a = 0; a < i; a++) { const kk = key(ORD[a], a); if (!(kk in map)) { map[kk] = { k: kk, s: 0, n: 0 }; groups.push(map[kk]); } map[kk].s += ORD[a][2]; map[kk].n++; }
      src.innerHTML = '<tr><th>customer</th><th>day</th><th>amount</th></tr>' + ORD.map((r, a) => `<tr style="${a === i ? 'background:rgba(219,109,40,.25)' : a < i ? 'opacity:.4' : ''}"><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
      dst.innerHTML = `<tr><th>${un.value === 'order' ? 'order' : un.value === 'cust' ? 'customer' : 'day'}</th><th>orders</th><th>total spent</th></tr>` + groups.map((x) => `<tr><td>${x.k}</td><td>${x.n}</td><td>${x.s}</td></tr>`).join('');
      q(view, '#rt').textContent = `Result: one row per ${un.value === 'order' ? 'order' : un.value === 'cust' ? 'customer' : 'day'} (${groups.length} rows so far)`;
      const total = ORD.reduce((s, r) => s + r[2], 0), G = un.value === 'order' ? 8 : new Set(ORD.map((r) => (un.value === 'cust' ? r[0] : r[1]))).size;
      out.innerHTML = i < 8 ? `Absorbing order ${i + 1}: ${ORD[i][0]}, ${ORD[i][1]}, ${ORD[i][2]} into <b>${key(ORD[i], i)}</b>.`
        : `Finished: 8 order rows became <b>${G}</b> rows. Total ${total} in every layout, but the average amount <i>per row</i> is ${total} / ${G} = <b>${(total / G).toFixed(2)}</b>. Choose the unit to match the question: “how much does a typical customer spend?” needs one row per customer.`;
    };
    const restart = () => { i = 0; acc = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; render(); };
    un.onchange = restart;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { playing = false; q(ctl, '#play').textContent = '▶ Play'; if (i < 8) { i++; render(); } };
    q(ctl, '#reset').onclick = restart;
    AI.loop(root, (dt) => {
      if (!playing || i >= 8) return;
      acc += dt * +sp.value / 1.3;
      if (acc >= 1) { acc -= 1; i++; render(); }
    }, { fps: 20 });
    render();
  };

  /* ---- 4. Seeds ---- */
  AI.labs['ds-fnd-seed'] = function (root) {
    const POP = [12, 15, 9, 22, 18, 14, 30, 11, 16, 20, 13, 25, 17, 10, 19, 21, 8, 24, 15, 27];
    const { view, ctl } = AI.labShell(root, 'Seeds make randomness repeatable', 'draw a few of 20 weekly sales figures at random (illustrative)',
      `<canvas style="height:300px"></canvas>`,
      `<label>Seed <input type="number" id="sd" value="42" style="width:90px"></label>
       <label><input type="checkbox" id="us" checked> use the seed (untick for a different draw every time)</label>
       ${sl('k', 'Sample size', 3, 10, 1, 5)}${sl('sp', 'Speed', 0.2, 2, 0.1, 0.7)}
       <div class="row"><button class="btn primary" id="run">Run again</button><button class="btn" id="play">⏸ Pause</button></div>
       <div class="readout" style="white-space:normal" id="out"></div><div class="readout" id="hist" style="white-space:normal;max-height:120px;overflow:auto"></div>
       <p class="caption">Bars are the 20 values (population mean 17.3, dashed). Each run picks without replacement. The history lists finished runs: with a seed they repeat exactly.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col(), out = q(ctl, '#out'), hist = q(ctl, '#hist'), sp = q(ctl, '#sp');
    let idx = [], shown = 0, acc = 0, playing = true, runs = 0, logs = [], last = null;
    const pick = () => {
      const K = +q(ctl, '#k').value, r = q(ctl, '#us').checked ? AI.rng(+q(ctl, '#sd').value || 0) : Math.random, a = POP.map((v, i) => i), res = [];
      for (let i = 0; i < K; i++) { const j = i + Math.floor(r() * (a.length - i)); const t = a[i]; a[i] = a[j]; a[j] = t; res.push(a[i]); }
      return res;
    };
    const start = () => { idx = pick(); shown = 0; acc = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); };
    const draw = () => {
      q(ctl, '#kv').textContent = q(ctl, '#k').value;
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const bw = (W - 40) / 20, base = H - 50, ht = H - 100, sel = idx.slice(0, shown);
      POP.forEach((v, i) => {
        const on = sel.includes(i), cur = shown > 0 && idx[shown - 1] === i;
        g.fillStyle = cur ? c.orange : on ? c.blue : c.bg3; g.fillRect(20 + i * bw + 2, base - (v / 32) * ht, bw - 4, (v / 32) * ht);
        txt(g, String(v), 20 + i * bw + bw / 2, base - (v / 32) * ht - 4, on ? c.fg : c.muted, '10px sans-serif', 'center');
        txt(g, String(i), 20 + i * bw + bw / 2, base + 13, c.muted, '10px sans-serif', 'center');
      });
      const my = base - (17.3 / 32) * ht; g.strokeStyle = c.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(20, my); g.lineTo(W - 20, my); g.stroke(); g.setLineDash([]);
      txt(g, 'index in the population', W / 2, H - 10, c.muted, '11px sans-serif', 'center');
      const vals = sel.map((i) => POP[i]);
      if (shown === idx.length && idx.length) {
        last = vals; const m = mean(vals);
        out.innerHTML = `Sample indices [${idx.join(', ')}] → values [${vals.join(', ')}], <b>mean ${m.toFixed(2)}</b>.`;
      } else out.innerHTML = shown ? `Picked ${shown} of ${idx.length}: [${vals.join(', ')}], running mean ${mean(vals).toFixed(2)}.` : 'Ready.';
    };
    const finish = () => { runs++; const m = mean(idx.map((i) => POP[i])); logs.unshift(`run ${runs}: ${q(ctl, '#us').checked ? 'seed ' + q(ctl, '#sd').value : 'no seed'} → [${idx.map((i) => POP[i]).join(', ')}], mean ${m.toFixed(2)}`); logs = logs.slice(0, 8); hist.innerHTML = logs.map(esc).join('<br>'); };
    q(ctl, '#run').onclick = start;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#k').oninput = start; q(ctl, '#sd').onchange = start; q(ctl, '#us').onchange = start;
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (!playing || shown >= idx.length) return;
      acc += dt * +sp.value / 1.1;
      if (acc >= 1) { acc -= 1; shown++; draw(); if (shown === idx.length) finish(); }
    }, { fps: 20 });
    start();
  };

  /* ---- 4b. Same data, different default ---- */
  AI.labs['ds-fnd-pin'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Same data, two defaults', 'a standard deviation whose divisor depends on the tool (illustrative)',
      `<canvas style="height:330px"></canvas>`,
      `<label>Data (2 to 14 numbers, edit them) <input type="text" id="dt" value="4, 8, 6, 5, 3, 2, 8, 9, 2, 5" style="width:100%"></label>
       ${sl('sp', 'Speed', 0.2, 2, 0.1, 0.7)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Replay</button></div>
       <div class="readout" style="white-space:normal" id="out"></div>
       <p class="caption">Stage 1 values arrive, stage 2 the mean, stage 3 each squared deviation is added to the sum of squares, stage 4 the sum is divided: by n (NumPy's default, ddof = 0) or by n − 1 (pandas' default, ddof = 1). Pin the version and state the setting.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col(), out = q(ctl, '#out'), sp = q(ctl, '#sp');
    let D = [], t = 0, playing = true;
    const parse = () => { const a = q(ctl, '#dt').value.split(/[,\s]+/).filter(Boolean).map(Number); return a.length >= 2 && a.length <= 14 && a.every(Number.isFinite) ? a : null; };
    const restart = () => { const a = parse(); q(ctl, '#dt').classList.toggle('bad', !a); if (a) D = a; t = 0; playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); };
    const draw = () => {
      const n = D.length, m = mean(D), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const T1 = n * 0.6, T2 = T1 + 1.6, T3 = T2 + n * 0.9, T4 = T3 + 2;
      const mx = Math.max(...D.map(Math.abs), 1) * 1.15, base = H - 70, ht = H - 130, bw = (W - 50) / n;
      const Y = (v) => base - (v / mx) * ht;
      const nb = t < T1 ? Math.floor(t / 0.6) + 1 : n, dev = t < T2 ? 0 : Math.min(n, Math.floor((t - T2) / 0.9) + 1);
      let ss = 0;
      for (let i = 0; i < dev; i++) ss += (D[i] - m) ** 2;
      for (let i = 0; i < n; i++) {
        if (i >= nb) break;
        const x = 30 + i * bw + 4;
        g.fillStyle = i === dev - 1 && t < T3 ? c.orange : c.blue; g.fillRect(x, Y(D[i]), bw - 8, base - Y(D[i]));
        txt(g, String(D[i]), x + (bw - 8) / 2, Y(D[i]) - 4, c.fg, '11px sans-serif', 'center');
        if (i < dev) { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(x + (bw - 8) / 2, Y(D[i])); g.lineTo(x + (bw - 8) / 2, Y(m)); g.stroke(); g.lineWidth = 1; txt(g, ((D[i] - m) ** 2).toFixed(2), x + (bw - 8) / 2, base + 14, c.red, '10px sans-serif', 'center'); }
      }
      let msg = `Stage 1: ${nb} of ${n} values have arrived.`;
      if (t >= T1) {
        g.strokeStyle = c.green; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(24, Y(m)); g.lineTo(W - 10, Y(m)); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
        txt(g, `mean = ${m.toFixed(2)}`, W - 12, Y(m) - 5, c.green, 'bold 12px sans-serif', 'right');
        msg = `Stage 2: the mean is ${D.reduce((s, x) => s + x, 0)} / ${n} = ${m.toFixed(2)}.`;
      }
      if (t >= T2) msg = `Stage 3: squared deviations added so far (${dev} of ${n}): sum of squares = <b>${ss.toFixed(2)}</b>.`;
      if (t >= T3) {
        const a = Math.sqrt(ss / n), b = Math.sqrt(ss / (n - 1));
        msg = `Stage 4: sum of squares ${ss.toFixed(2)}.<br>Divide by n = ${n}: sd = <b>${a.toFixed(4)}</b> (NumPy default). Divide by n − 1 = ${n - 1}: sd = <b>${b.toFixed(4)}</b> (pandas default).<br>The second is ${((b / a - 1) * 100).toFixed(1)}% larger from identical data. Neither is a bug: the setting is part of the result, so record it.`;
      }
      out.innerHTML = msg;
    };
    q(ctl, '#dt').oninput = restart;
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = restart;
    o.onresize = draw;
    AI.loop(root, (dt) => {
      const n = D.length, T4 = n * 0.6 + 1.6 + n * 0.9 + 2;
      if (!playing || t >= T4) return;
      t += dt * +sp.value; draw();
    }, { fps: 30 });
    restart();
  };

  /* ---- 5. An answerable question ---- */
  AI.labs['ds-fnd-question'] = function (root) {
    const NMAX = 6000;
    const { view, ctl } = AI.labShell(root, 'Does the data answer the decision?', 'orders arrive one by one; is the complaint rate above the threshold? (illustrative simulation)',
      `<canvas style="height:340px"></canvas>`,
      `${sl('tr', 'True complaint rate (unknown in real life)', 0.02, 0.15, 0.005, 0.09)}${sl('th', 'Decision threshold (hire support above this)', 0.04, 0.12, 0.005, 0.08)}${sl('sp', 'Orders per second', 10, 400, 10, 80)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Restart</button><button class="btn" id="new">New run</button></div>
       <div class="readout" style="white-space:normal" id="out"></div>
       <p class="caption">Orange line: the observed rate with its 95% interval (normal approximation, p̂ ± 1.96·√(p̂(1−p̂)/n)). Dashed line: the threshold from the decision. The question is only answered once the whole interval is on one side.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col(), out = q(ctl, '#out');
    let n = 0, k = 0, pts = [], seed = 11, r, playing = true, tick = AI.ticker(), decided = null;
    const restart = () => { n = 0; k = 0; pts = []; decided = null; r = AI.rng(seed); playing = true; q(ctl, '#play').textContent = '⏸ Pause'; draw(); };
    const stat = () => { const p = n ? k / n : 0, se = n ? Math.sqrt(p * (1 - p) / n) : 0; return { p, lo: p - 1.96 * se, hi: p + 1.96 * se }; };
    const draw = () => {
      q(ctl, '#trv').textContent = (+q(ctl, '#tr').value * 100).toFixed(1) + '%'; q(ctl, '#thv').textContent = (+q(ctl, '#th').value * 100).toFixed(1) + '%'; q(ctl, '#spv').textContent = q(ctl, '#sp').value;
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const V = AI.view(o, 0, NMAX, 0, 0.2, { l: 48, r: 14, t: 14, b: 32 }), th = +q(ctl, '#th').value, s = stat();
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(V.X(0), V.Y(0)); g.lineTo(V.X(NMAX), V.Y(0)); g.moveTo(V.X(0), V.Y(0)); g.lineTo(V.X(0), V.Y(0.2)); g.stroke();
      [0, 0.05, 0.1, 0.15, 0.2].forEach((v) => txt(g, (v * 100).toFixed(0) + '%', 42, V.Y(v) + 4, c.muted, '11px sans-serif', 'right'));
      [0, 2000, 4000, 6000].forEach((v) => txt(g, String(v), V.X(v), V.Y(0) + 16, c.muted, '11px sans-serif', 'center'));
      txt(g, 'orders observed (n)', V.X(NMAX / 2), H - 4, c.muted, '11px sans-serif', 'center');
      g.strokeStyle = c.red; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(V.X(0), V.Y(th)); g.lineTo(V.X(NMAX), V.Y(th)); g.stroke(); g.setLineDash([]);
      txt(g, `threshold ${(th * 100).toFixed(1)}%`, V.X(NMAX) - 4, V.Y(th) - 5, c.red, '11px sans-serif', 'right');
      if (pts.length > 1) {
        const cl = (v) => V.Y(Math.max(0, Math.min(0.2, v)));
        g.fillStyle = 'rgba(219,109,40,.22)'; g.beginPath(); pts.forEach((pt, i) => (i ? g.lineTo(V.X(pt.n), cl(pt.hi)) : g.moveTo(V.X(pt.n), cl(pt.hi))));
        for (let i = pts.length - 1; i >= 0; i--) g.lineTo(V.X(pts[i].n), cl(pts[i].lo)); g.closePath(); g.fill();
        g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); pts.forEach((pt, i) => (i ? g.lineTo(V.X(pt.n), cl(pt.p)) : g.moveTo(V.X(pt.n), cl(pt.p)))); g.stroke(); g.lineWidth = 1;
      }
      let v = `n = ${n}: ${k} complaints, observed rate ${(s.p * 100).toFixed(2)}%.`;
      if (n >= 30) {
        v += ` 95% interval ${(Math.max(0, s.lo) * 100).toFixed(2)}% to ${(s.hi * 100).toFixed(2)}%. `;
        if (s.lo > th) v += '<b>Whole interval above the threshold: the data supports hiring.</b>';
        else if (s.hi < th) v += '<b>Whole interval below the threshold: the data supports not hiring.</b>';
        else v += '<b>The interval still contains the threshold: the data cannot answer the decision yet. Keep collecting or accept the uncertainty.</b>';
        if (decided && (s.lo > th || s.hi < th)) v += `<br>The verdict first became clear at n = ${decided}.`;
      } else v += ' Too few orders for an interval.';
      out.innerHTML = v;
    };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = restart;
    q(ctl, '#new').onclick = () => { seed++; restart(); };
    ['#tr', '#th', '#sp'].forEach((id) => (q(ctl, id).oninput = () => { if (id !== '#sp') decided = null; draw(); }));
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (!playing || n >= NMAX) return;
      const m = tick(dt, +q(ctl, '#sp').value), tr = +q(ctl, '#tr').value;
      for (let i = 0; i < m && n < NMAX; i++) {
        n++; if (r() < tr) k++;
        if (n % 20 === 0) { const s = stat(); pts.push({ n, p: s.p, lo: s.lo, hi: s.hi }); }
        if (n >= 30 && !decided) { const s = stat(), th = +q(ctl, '#th').value; if (s.lo > th || s.hi < th) decided = n; }
      }
      draw();
    }, { fps: 30 });
    restart();
  };
})();
