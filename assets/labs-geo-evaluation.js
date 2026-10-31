/* GeoAI · Module 8 (evaluation, accuracy & ground truth) labs.
 *
 * accsampler   sample points checked one by one against a reference, confusion matrix filling, UA/PA/OA live
 * areaweight   class areas as stacked bars; raw vs area-adjusted accuracy; rare classes move the estimate
 * objmatch     predicted vs reference objects on a pixel grid; IoU, Dice and matching at a threshold
 * labelnoise   reference labels flipped at a rate; measured accuracy falls below the true accuracy
 * errormap     per-pixel error map scanned row by row; filters by region and class
 * uncmap       per-pixel class probabilities -> entropy map -> flagged review area
 *
 * Synthetic maps are seeded and labelled as illustrative. Every number shown is computed in the page.
 */
(function () {
  const G = (AI.geoEval = {});
  const CNAME = ['Forest', 'Crop', 'Urban'];
  const CL = ['green', 'yellow', 'purple']; // class colour keys (AI.col())
  const W = 40, H = 28; // synthetic map size in pixels
  G.W = W; G.H = H;

  /* ---------- small helpers ---------- */
  const rgba = (hex, a) => { const [r, g, b] = AI.rgb(hex); return `rgba(${r},${g},${b},${a})`; };
  const ramp = (stops, t) => {
    t = AI.clamp(t, 0, 1);
    const n = stops.length - 1, f = t * n, i = Math.min(n - 1, Math.floor(f)), u = f - i;
    const A = AI.rgb(stops[i]), B = AI.rgb(stops[i + 1]);
    return `rgb(${A.map((v, k) => Math.round(v + (B[k] - v) * u)).join(',')})`;
  };
  const pct = (v, d = 0) => (Number.isFinite(v) ? (100 * v).toFixed(d) + '%' : '–');
  const r3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–');
  const shuffle = (arr, rnd) => { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; } return arr; };
  G.rgba = rgba; G.ramp = ramp;

  /* seeded smooth field on a W×H grid, values in [-1, 1] (bilinear value noise with smoothstep) */
  G.field = (seed, w, h, cell) => {
    const r = AI.rng(seed);
    const nx = Math.ceil(w / cell) + 2, ny = Math.ceil(h / cell) + 2;
    const lat = new Float64Array(nx * ny);
    for (let i = 0; i < lat.length; i++) lat[i] = r() * 2 - 1;
    const out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const fx = x / cell, fy = y / cell, ix = Math.floor(fx), iy = Math.floor(fy);
      const tx = AI.smooth(fx - ix), ty = AI.smooth(fy - iy);
      const a = lat[iy * nx + ix], b = lat[iy * nx + ix + 1], c = lat[(iy + 1) * nx + ix], d = lat[(iy + 1) * nx + ix + 1];
      out[y * w + x] = AI.lerp(AI.lerp(a, b, tx), AI.lerp(c, d, tx), ty);
    }
    return out;
  };

  /* synthetic land-cover map: reference (0 Forest, 1 Crop, 2 Urban) and a product with clustered errors */
  G.makeMap = (seed, rate) => {
    const N = W * H;
    const f1 = G.field(seed * 7 + 1, W, H, 9);
    const ref = new Uint8Array(N);
    for (let i = 0; i < N; i++) { const s = f1[i]; ref[i] = s > 0.22 ? 0 : s < -0.28 ? 2 : 1; }
    const e = G.field(seed * 7 + 2, W, H, 6), g = G.field(seed * 7 + 3, W, H, 5);
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => e[a] - e[b]);
    const k = Math.round(rate * N);
    const prod = Uint8Array.from(ref);
    const isErr = new Uint8Array(N);
    for (let j = 0; j < k; j++) { const i = order[j]; isErr[i] = 1; prod[i] = (ref[i] + 1 + (g[i] > 0 ? 1 : 0)) % 3; }
    return { W, H, N, ref, prod, isErr, nErr: k, seed };
  };
  G.region = (x, y) => (y < H / 2 ? 'N' : 'S') + (x < W / 2 ? 'W' : 'E');

  /* sample order: stratified (equal allocation across map classes, interleaved) or simple random without replacement */
  G.sampleOrder = (map, N, mode, seed) => {
    const rnd = AI.rng(seed * 31 + 5);
    const idx = Array.from({ length: map.N }, (_, i) => i);
    if (mode === 'random') return shuffle(idx, rnd).slice(0, N);
    const pools = [[], [], []];
    idx.forEach((i) => pools[map.prod[i]].push(i));
    pools.forEach((p) => shuffle(p, rnd));
    const out = [];
    for (let k = 0; k < N; k++) out.push(pools[k % 3][Math.floor(k / 3)]);
    return out.filter((v) => v !== undefined);
  };

  /* ---------- control builders ---------- */
  function sliderRow(ctl, label, min, max, step, val, fmt, onChange) {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => { v.textContent = fmt ? fmt(+inp.value) : AI.fmt(+inp.value, 2); };
    inp.addEventListener('input', () => { upd(); onChange(+inp.value); });
    upd(); ctl.appendChild(l);
    return { set(x) { inp.value = x; upd(); }, get() { return +inp.value; } };
  }
  function selectRow(ctl, label, options, val, onChange) {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span><select>${options.map(([v, t]) => `<option value="${v}"${String(v) === String(val) ? ' selected' : ''}>${t}</option>`).join('')}</select>`;
    const s = l.querySelector('select');
    s.addEventListener('change', () => onChange(s.value));
    ctl.appendChild(l);
    return { set(x) { s.value = String(x); }, get() { return s.value; } };
  }
  function buttons(ctl, list) {
    const row = document.createElement('div'); row.className = 'row';
    const b = {};
    list.forEach(([k, t, fn]) => {
      const el = document.createElement('button'); el.className = 'btn small'; el.textContent = t;
      el.addEventListener('click', () => fn(el));
      row.appendChild(el); b[k] = el;
    });
    ctl.appendChild(row);
    return b;
  }
  const readoutBox = (ctl, html) => { const o = document.createElement('div'); o.className = 'readout'; o.style.whiteSpace = 'normal'; if (html) o.innerHTML = html; ctl.appendChild(o); return o; };
  const CLS_OPTS = [['all', 'all classes'], ['0', 'Forest'], ['1', 'Crop'], ['2', 'Urban']];

  /* =====================================================================
   * 8.1 accuracy sampler
   * ===================================================================== */
  AI.labs.accsampler = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Sampling a land-cover product', d.sub || 'one point at a time, compared with the reference',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const st = { seed: 7, rate: 0.2, N: 30, mode: 'stratified', speed: 0.6, playing: true, done: 0, flash: 0 };
    let map = null, order = [], key = '';
    const tick = AI.ticker();
    const rebuild = () => { map = G.makeMap(st.seed, st.rate); order = G.sampleOrder(map, st.N, st.mode, st.seed); st.done = 0; st.flash = 0; st.playing = true; key = ''; };
    c.onresize = () => { key = ''; };

    const confusion = () => {
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let s = 0; s < st.done; s++) { const i = order[s]; M[map.prod[i]][map.ref[i]]++; }
      return M;
    };

    const drawMap = () => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const gap = 16, pw = (w - gap) / 2, cell = Math.min(pw / W, (h - 6) / H);
      const ox = [(pw - cell * W) / 2, pw + gap + (pw - cell * W) / 2], oy = (h - cell * H) / 2;
      const titles = ['Reference (truth)', 'Product (the map)'];
      for (let p = 0; p < 2; p++) {
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const i = y * W + x, k = p === 0 ? map.ref[i] : map.prod[i];
          g.fillStyle = C[CL[k]]; g.fillRect(ox[p] + x * cell, oy + y * cell, cell + 0.4, cell + 0.4);
        }
        g.fillStyle = C.muted; g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
        g.fillText(titles[p], ox[p] + (cell * W) / 2, oy - 4 < 12 ? 12 : oy - 4);
      }
      for (let s = 0; s < st.done; s++) {
        const i = order[s], x = i % W, y = (i / W) | 0, agree = map.prod[i] === map.ref[i];
        const r = Math.max(2.6, cell * 0.32);
        for (let p = 0; p < 2; p++) {
          g.beginPath(); g.arc(ox[p] + (x + 0.5) * cell, oy + (y + 0.5) * cell, r, 0, 6.2832);
          g.fillStyle = agree ? C.green : C.red; g.fill(); g.lineWidth = 1; g.strokeStyle = C.bg; g.stroke();
        }
      }
      if (st.done > 0 && st.flash > 0) {
        const i = order[st.done - 1], x = i % W, y = (i / W) | 0;
        for (let p = 0; p < 2; p++) {
          g.beginPath(); g.arc(ox[p] + (x + 0.5) * cell, oy + (y + 0.5) * cell, cell * (1 + (1 - st.flash) * 2), 0, 6.2832);
          g.strokeStyle = C.fg; g.globalAlpha = st.flash; g.lineWidth = 2; g.stroke(); g.globalAlpha = 1;
        }
      }
    };

    const update = (M) => {
      const rowT = [0, 1, 2].map((i) => M[i].reduce((a, b) => a + b, 0));
      const colT = [0, 1, 2].map((j) => M[0][j] + M[1][j] + M[2][j]);
      const diag = M[0][0] + M[1][1] + M[2][2];
      const OA = st.done ? diag / st.done : NaN;
      const UA = [0, 1, 2].map((i) => (rowT[i] ? M[i][i] / rowT[i] : NaN));
      const PA = [0, 1, 2].map((j) => (colT[j] ? M[j][j] / colT[j] : NaN));
      const cells = [0, 1, 2].map((i) => `<tr><td><b>${CNAME[i]}</b></td>${[0, 1, 2].map((j) => `<td style="${i === j ? 'color:var(--green);font-weight:600' : ''}">${M[i][j]}</td>`).join('')}<td>${rowT[i]}</td><td>${pct(UA[i])}</td></tr>`).join('');
      out.innerHTML = `<table class="tbl" style="margin:0 0 8px"><tr><th>Map ↓ / Ref →</th>${CNAME.map((n) => `<th>${n}</th>`).join('')}<th>row n<sub>i·</sub></th><th>UA</th></tr>${cells}`
        + `<tr><td>column n<sub>·j</sub></td>${colT.map((v) => `<td>${v}</td>`).join('')}<td>${st.done}</td><td></td></tr>`
        + `<tr><td>PA</td>${PA.map((v) => `<td>${pct(v)}</td>`).join('')}<td></td><td></td></tr></table>`
        + `OA = diagonal / n = ${diag} / ${st.done} = <b>${Number.isFinite(OA) ? OA.toFixed(3) : '–'}</b>`;
      if (st.done === 0) cap.textContent = 'Press Play. Each point asks the map and the reference the same question.';
      else {
        const i = order[st.done - 1], p = map.prod[i], r = map.ref[i];
        cap.textContent = `Point ${st.done} of ${st.N} (${st.mode}): map says ${CNAME[p]}, reference says ${CNAME[r]}: ${p === r ? 'agreement' : 'disagreement'}.` + (st.done === st.N ? ' Sample complete: read UA and PA in the table.' : '');
      }
    };

    const out = readoutBox(ctl);
    const btn = buttons(ctl, [
      ['play', 'Pause', () => { if (st.done >= st.N) rebuild(); st.playing = !st.playing; }],
      ['step', 'Step', () => { if (st.done < st.N) { st.done++; st.flash = 1; key = ''; } st.playing = false; }],
      ['reset', 'Reset', () => rebuild()],
      ['map', 'New map', () => { st.seed = (st.seed % 97) + 3; rebuild(); }],
    ]);
    selectRow(ctl, 'sampling design', [['stratified', 'stratified (by map class)'], ['random', 'simple random']], st.mode, (v) => { st.mode = v; rebuild(); });
    sliderRow(ctl, 'product error rate', 0, 0.4, 0.01, st.rate, (v) => AI.fmt(v, 2), (v) => { st.rate = v; rebuild(); });
    sliderRow(ctl, 'sample size N', 10, 100, 1, st.N, (v) => String(v), (v) => { st.N = v; rebuild(); });
    sliderRow(ctl, 'speed', 0.2, 3, 0.1, st.speed, (v) => v.toFixed(1) + ' points/s', (v) => { st.speed = v; });

    rebuild();
    AI.loop(root, (dt) => {
      if (st.playing && st.done < st.N) {
        const n = tick(dt, st.speed);
        if (n > 0) { st.done = Math.min(st.N, st.done + n); st.flash = 1; if (st.done >= st.N) st.playing = false; }
      }
      if (st.flash > 0) st.flash = Math.max(0, st.flash - dt * 1.2);
      btn.play.textContent = st.playing ? 'Pause' : (st.done >= st.N ? 'Replay' : 'Play');
      const k = `${st.done}|${Math.round(st.flash * 30)}|${c.w}x${c.h}|${document.documentElement.dataset.theme}|${st.seed}|${st.rate}|${st.N}|${st.mode}`;
      if (k === key) return;
      key = k;
      drawMap();
      update(confusion());
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8.2 area weighting
   * ===================================================================== */
  const AW_M = [[10, 2, 1], [1, 8, 1], [0, 1, 6]]; // the matrix of lesson 8.1 (rows: map, columns: reference)
  const AW_rowT = AW_M.map((r) => r.reduce((a, b) => a + b, 0));
  const AW_colT = [0, 1, 2].map((j) => AW_M[0][j] + AW_M[1][j] + AW_M[2][j]);
  const AW_N = AW_rowT.reduce((a, b) => a + b, 0);
  G.areaEstimates = (A) => {
    const tot = A[0] + A[1] + A[2], Wt = A.map((a) => a / tot);
    const p = AW_M.map((r, i) => r.map((v) => (Wt[i] * v) / AW_rowT[i]));
    const colP = [0, 1, 2].map((j) => p[0][j] + p[1][j] + p[2][j]);
    return {
      W: Wt,
      OA: p[0][0] + p[1][1] + p[2][2],
      PA: [0, 1, 2].map((j) => p[j][j] / colP[j]),
      UA: [0, 1, 2].map((i) => AW_M[i][i] / AW_rowT[i]),
      PAraw: [0, 1, 2].map((j) => AW_M[j][j] / AW_colT[j]),
      OAraw: (AW_M[0][0] + AW_M[1][1] + AW_M[2][2]) / AW_N,
      p,
    };
  };

  AI.labs.areaweight = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Area weights and the estimate', d.sub || 'the same sample, different areas',
      '<canvas style="height:330px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const st = { A: [700, 250, 50], speed: 3, playing: true, replay: false };
    const tgt = () => G.areaEstimates(st.A);
    const disp = { W: tgt().W, PA: tgt().PA, OA: tgt().OA };
    let key = '';
    c.onresize = () => { key = ''; };
    const out = readoutBox(ctl);

    const draw = () => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h, L = 60, R = w - 14;
      g.clearRect(0, 0, w, h);
      g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textBaseline = 'middle';
      // stacked bars: map area shares (animated) and sample shares (fixed)
      const bar = (y, parts, label) => {
        let x = L; g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(label, L - 6, y + 9);
        parts.forEach(([frac, col, txt]) => {
          const bw = (R - L) * frac;
          g.fillStyle = col; g.fillRect(x, y, Math.max(0, bw), 18);
          if (bw > 70) { g.fillStyle = C.bg; g.textAlign = 'center'; g.fillText(txt, x + bw / 2, y + 9); }
          x += bw;
        });
        g.strokeStyle = C.border; g.strokeRect(L + 0.5, y + 0.5, R - L - 1, 17);
      };
      bar(12, [0, 1, 2].map((i) => [disp.W[i], C[CL[i]], `${CNAME[i]} ${pct(disp.W[i])}`]), 'map area');
      bar(40, [0, 1, 2].map((i) => [AW_rowT[i] / AW_N, C[CL[i]], `${pct(AW_rowT[i] / AW_N)}`]), 'sample');
      // grouped bars: producer's accuracy raw vs area-adjusted
      const top = 96, bottom = h - 40, ph = bottom - top, yv = (v) => bottom - v * ph;
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.textAlign = 'right';
      [0, 0.5, 1].forEach((v) => { g.beginPath(); g.moveTo(L, yv(v) + 0.5); g.lineTo(R, yv(v) + 0.5); g.stroke(); g.fillText(v.toFixed(1), L - 6, yv(v)); });
      const gw = (R - L) / 3;
      for (let j = 0; j < 3; j++) {
        const x0 = L + j * gw + gw * 0.2, bw = gw * 0.26;
        const raw = AW_M[j][j] / AW_colT[j], adj = disp.PA[j];
        g.fillStyle = rgba(C.muted, 0.45); g.fillRect(x0, yv(raw), bw, bottom - yv(raw));
        g.fillStyle = C[CL[j]]; g.fillRect(x0 + bw + 6, yv(adj), bw, bottom - yv(adj));
        g.fillStyle = C.fg; g.textAlign = 'center';
        g.fillText(pct(raw), x0 + bw / 2, yv(raw) - 9);
        g.fillText(pct(adj), x0 + bw * 1.5 + 6, yv(adj) - 9);
        g.fillStyle = C.muted; g.fillText(`${CNAME[j]} PA`, x0 + bw + 3, bottom + 14);
      }
      g.textAlign = 'left'; g.fillStyle = C.muted;
      g.fillText('grey = raw counts     coloured = area-adjusted', L, bottom + 28);
    };

    const btn = buttons(ctl, [
      ['play', 'Pause', () => { st.playing = !st.playing; }],
      ['replay', 'Replay', () => { disp.W = [0, 0, 0]; disp.PA = [0, 0, 0]; disp.OA = 0; st.playing = true; }],
    ]);
    const sl = [0, 1, 2].map((i) => sliderRow(ctl, `${CNAME[i]} area (km²)`, 5, 1000, 5, st.A[i], (v) => `${v} km²`, (v) => { st.A[i] = v; }));
    sliderRow(ctl, 'easing speed', 0.5, 8, 0.1, st.speed, (v) => v.toFixed(1), (v) => { st.speed = v; });

    AI.loop(root, (dt) => {
      const t = tgt();
      if (st.playing) {
        const k = 1 - Math.exp(-dt * st.speed);
        for (let i = 0; i < 3; i++) { disp.W[i] += (t.W[i] - disp.W[i]) * k; disp.PA[i] += (t.PA[i] - disp.PA[i]) * k; }
        disp.OA += (t.OA - disp.OA) * k;
      }
      const settled = Math.abs(disp.OA - t.OA) < 1e-4 && disp.W.every((v, i) => Math.abs(v - t.W[i]) < 1e-4);
      btn.play.textContent = st.playing ? 'Pause' : 'Play';
      const k2 = `${disp.W.map((v) => v.toFixed(3)).join()}|${disp.PA.map((v) => v.toFixed(3)).join()}|${c.w}x${c.h}|${document.documentElement.dataset.theme}`;
      if (k2 === key && settled) return;
      key = k2;
      draw();
      const Wt = t.W;
      const w3 = Wt[2], nU = AW_rowT[2];
      cap.textContent = `Urban is ${pct(w3)} of the area but ${pct(nU / AW_N)} of the sample. Its Forest-to-Urban error carries weight ${(Wt[0] / AW_rowT[0]).toFixed(3)} on its own; each correct Urban point carries ${(w3 / nU).toFixed(3)}.`;
      out.innerHTML = `areas: Forest ${st.A[0]} · Crop ${st.A[1]} · Urban ${st.A[2]} km²  →  W = ${Wt.map((v) => v.toFixed(3)).join(', ')}<br>`
        + `overall accuracy: raw <b>${r3(t.OAraw)}</b> · area-adjusted <b>${r3(t.OA)}</b><br>`
        + `producer's accuracy (raw → adjusted): ${[0, 1, 2].map((j) => `${CNAME[j]} ${r3(t.PAraw[j])} → <b>${r3(t.PA[j])}</b>`).join(' · ')}<br>`
        + `user's accuracy (unchanged by area): ${t.UA.map((v, i) => `${CNAME[i]} ${r3(v)}`).join(' · ')}`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8.3 object matcher
   * ===================================================================== */
  const OM_COLS = 10, OM_ROWS = 8;
  const OM_REF_ROWS = ['..........', '.AAA......', '.AAA......', '..........', '.....BB...', '.....BB.CC', '.....BB.CC', '........CC'];
  const OM_PRED_ROWS = ['..........', '.1111.....', '.1111.....', '..........', '.....22...', '.....22...', '......3333', '......3333'];
  const omSet = (rows, ch) => { const s = new Set(); rows.forEach((r, y) => [...r].forEach((c, x) => { if (c === ch) s.add(y * OM_COLS + x); })); return s; };
  const OM_REFS = [{ id: 'A', set: omSet(OM_REF_ROWS, 'A') }, { id: 'B', set: omSet(OM_REF_ROWS, 'B') }, { id: 'C', set: omSet(OM_REF_ROWS, 'C') }];
  const OM_PREDS = [{ id: '1', set: omSet(OM_PRED_ROWS, '1') }, { id: '2', set: omSet(OM_PRED_ROWS, '2') }, { id: '3', set: omSet(OM_PRED_ROWS, '3') }];
  const OM_NB = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const omAdjust = (set, b, others) => {
    if (b === 0) return new Set(set);
    if (b > 0) {
      const out = new Set(set);
      for (const i of set) {
        const x = i % OM_COLS, y = (i / OM_COLS) | 0;
        for (const [dx, dy] of OM_NB) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= OM_COLS || ny >= OM_ROWS) continue;
          const j = ny * OM_COLS + nx;
          if (!others.has(j)) out.add(j);
        }
      }
      return out;
    }
    const out = new Set();
    for (const i of set) {
      const x = i % OM_COLS, y = (i / OM_COLS) | 0;
      let edge = false;
      for (const [dx, dy] of OM_NB) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= OM_COLS || ny >= OM_ROWS) continue;
        if (!set.has(ny * OM_COLS + nx)) edge = true;
      }
      if (!edge) out.add(i);
    }
    return out;
  };
  G.omEvaluate = (preds, thr) => {
    const pairs = [];
    for (const r of OM_REFS) for (const p of preds) {
      let inter = 0; for (const i of p.set) if (r.set.has(i)) inter++;
      if (!inter) continue;
      const uni = r.set.size + p.set.size - inter;
      pairs.push({ r, p, inter, uni, iou: inter / uni, dice: (2 * inter) / (r.set.size + p.set.size) });
    }
    pairs.sort((a, b) => b.iou - a.iou);
    const usedR = new Set(), usedP = new Set(), matches = [];
    for (const pr of pairs) {
      if (usedR.has(pr.r.id) || usedP.has(pr.p.id)) continue;
      if (pr.iou >= thr - 1e-9) { usedR.add(pr.r.id); usedP.add(pr.p.id); matches.push(pr); }
    }
    const rows = OM_REFS.map((r) => {
      const cand = pairs.filter((pr) => pr.r === r);
      const best = cand.length ? cand[0] : null; // pairs are sorted by IoU
      const m = matches.find((x) => x.r === r) || null;
      return { r, best, match: m };
    });
    const TP = matches.length, FN = OM_REFS.length - TP, FP = preds.length - TP;
    const P = TP + FP ? TP / (TP + FP) : 0, Rc = TP + FN ? TP / (TP + FN) : 0;
    return { pairs, matches, rows, TP, FP, FN, P, R: Rc, F1: P + Rc ? (2 * P * Rc) / (P + Rc) : 0 };
  };

  AI.labs.objmatch = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Predicted vs reference buildings', d.sub || 'IoU per pair, match threshold, one-pixel boundary change',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const st = { thr: 0.5, b: 0, sweep: false, sweepT: 0, speed: 1 };
    const N = OM_COLS * OM_ROWS;
    const predAlpha = new Float32Array(N), predTarget = new Float32Array(N);
    let preds = [], ev = null, key = '', lastB = null;
    c.onresize = () => { key = ''; };
    const refCol = { A: 'blue', B: 'purple', C: 'teal' };
    const out = readoutBox(ctl);

    const recompute = () => {
      const base = OM_PREDS.map((p) => p.set);
      preds = OM_PREDS.map((p, k) => ({ id: p.id, set: omAdjust(p.set, st.b, new Set([].concat(...base.filter((_, j) => j !== k).map((s) => [...s])))) }));
      predTarget.fill(0);
      preds.forEach((p) => p.set.forEach((i) => (predTarget[i] = 1)));
      ev = G.omEvaluate(preds, st.thr);
      lastB = st.b;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const cs = Math.min((w - 20) / OM_COLS, (h - 36) / OM_ROWS), ox = (w - cs * OM_COLS) / 2, oy = 14;
      for (let y = 0; y < OM_ROWS; y++) for (let x = 0; x < OM_COLS; x++) {
        const i = y * OM_COLS + x, px = ox + x * cs, py = oy + y * cs;
        g.fillStyle = C.bg3; g.fillRect(px, py, cs, cs);
        const ref = OM_REFS.find((r) => r.set.has(i));
        if (ref) { g.fillStyle = rgba(C[refCol[ref.id]], 0.4); g.fillRect(px, py, cs, cs); }
        const a = predAlpha[i];
        if (a > 0.01) { g.fillStyle = rgba(C.orange, 0.55 * a); g.fillRect(px + cs * 0.12, py + cs * 0.12, cs * 0.76, cs * 0.76); }
      }
      g.strokeStyle = C.border;
      for (let x = 0; x <= OM_COLS; x++) { g.beginPath(); g.moveTo(ox + x * cs + 0.5, oy); g.lineTo(ox + x * cs + 0.5, oy + OM_ROWS * cs); g.stroke(); }
      for (let y = 0; y <= OM_ROWS; y++) { g.beginPath(); g.moveTo(ox, oy + y * cs + 0.5); g.lineTo(ox + OM_COLS * cs, oy + y * cs + 0.5); g.stroke(); }
      // reference outlines and labels
      g.font = '600 12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
      OM_REFS.forEach((r, k) => {
        let sx = 0, sy = 0; r.set.forEach((i) => { sx += i % OM_COLS; sy += (i / OM_COLS) | 0; });
        const cx = ox + ((sx / r.set.size) + 0.5) * cs, cy = oy + ((sy / r.set.size) + 0.5) * cs;
        const row = ev.rows[k], ok = !!row.match;
        const txt = `${r.id} ${row.best ? row.best.iou.toFixed(2) : '–'}`;
        const tw = g.measureText(txt).width + 10;
        g.fillStyle = C.bg; g.globalAlpha = 0.85; g.fillRect(cx - tw / 2, cy - 9, tw, 18); g.globalAlpha = 1;
        g.fillStyle = ok ? C.green : C.red; g.fillText(txt, cx, cy);
      });
      g.fillStyle = C.muted; g.textAlign = 'left'; g.font = '11px ui-monospace,Menlo,Consolas,monospace';
      g.fillText(`blue/purple/teal = reference A/B/C   orange = predicted   label = best IoU (green = match)`, ox, oy + OM_ROWS * cs + 14);
    };

    buttons(ctl, [['sweep', 'Sweep', (el) => { st.sweep = !st.sweep; el.textContent = st.sweep ? 'Stop sweep' : 'Sweep'; if (st.sweep) st.sweepT = 0; }],
      ['reset', 'Reset', () => { st.b = 0; bSl.set(0); thrSl.set(0.5); }]]);
    const thrSl = sliderRow(ctl, 'match threshold (IoU ≥)', 0.05, 0.95, 0.05, st.thr, (v) => v.toFixed(2), (v) => { st.thr = v; });
    const bSl = sliderRow(ctl, 'predicted boundary (−1 shrink, +1 grow, px)', -1, 1, 1, st.b, (v) => (v > 0 ? '+' : '') + v + ' px', (v) => { st.b = v; });
    sliderRow(ctl, 'sweep speed', 0.3, 2, 0.1, st.speed, (v) => v.toFixed(1) + '×', (v) => { st.speed = v; });

    recompute();
    for (let i = 0; i < N; i++) predAlpha[i] = predTarget[i];
    AI.loop(root, (dt) => {
      if (st.sweep) {
        st.sweepT += dt * st.speed;
        const seq = [0, 1, 0, -1][Math.floor(st.sweepT / 2.2) % 4];
        if (seq !== st.b) { st.b = seq; bSl.set(seq); }
      }
      if (st.b !== lastB) recompute();
      let moving = false;
      const k = 1 - Math.exp(-dt * 5);
      for (let i = 0; i < N; i++) {
        const dd = predTarget[i] - predAlpha[i];
        if (Math.abs(dd) > 0.005) { predAlpha[i] += dd * k; moving = true; } else predAlpha[i] = predTarget[i];
      }
      const k2 = `${st.thr}|${st.b}|${c.w}x${c.h}|${document.documentElement.dataset.theme}`;
      if (k2 === key && !moving) return;
      key = k2;
      draw();
      const e = ev;
      const misses = e.rows.filter((rw) => !rw.match).map((rw) => (rw.best ? `${rw.r.id} best IoU ${rw.best.iou.toFixed(3)} is below ${st.thr.toFixed(2)}` : `${rw.r.id} has no overlap`));
      cap.textContent = `Predicted boundary ${st.b > 0 ? '+' : ''}${st.b} px. ${e.TP} of 3 pairs match at IoU ≥ ${st.thr.toFixed(2)}.` + (misses.length ? ' Misses: ' + misses.join('; ') + '.' : ' Every reference object is matched.');
      const rowsHtml = e.rows.map((rw) => {
        const b = rw.best;
        if (!b) return `<tr><td>${rw.r.id}</td><td>–</td><td colspan="5">no overlap</td></tr>`;
        const status = rw.match ? '<b style="color:var(--green)">match</b>' : '<span style="color:var(--red)">miss</span>';
        return `<tr><td>${rw.r.id} vs ${b.p.id}</td><td>${b.inter}</td><td>${b.uni}</td><td>${b.iou.toFixed(3)}</td><td>${b.dice.toFixed(3)}</td><td>${rw.r.set.size * 100} / ${b.p.set.size * 100}</td><td>${status}</td></tr>`;
      }).join('');
      out.innerHTML = `<table class="tbl" style="margin:0 0 8px"><tr><th>best pair</th><th>∩ px</th><th>∪ px</th><th>IoU</th><th>Dice</th><th>area m² ref / pred</th><th></th></tr>${rowsHtml}</table>`
        + `TP = ${e.TP} · FP = ${e.FP} · FN = ${e.FN}<br>precision = ${e.P.toFixed(3)} · recall = ${e.R.toFixed(3)} · <b>F1 = ${e.F1.toFixed(3)}</b>`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8.4 label noise
   * ===================================================================== */
  AI.labs.labelnoise = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Label noise and measured accuracy', d.sub || 'a point counts as wrong when model and reference disagree',
      '<canvas style="height:340px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const NP = 200, GW = 20, GH = 10;
    const st = { a: 0.9, etaT: 0.2, eta: 0.2, seed: 11, speed: 1, sweep: false, playing: true };
    let rank1 = null, rank2 = null, key = '';
    c.onresize = () => { key = ''; };
    const build = () => {
      const r = AI.rng(st.seed * 97 + 3);
      const p1 = shuffle(Array.from({ length: NP }, (_, i) => i), r), p2 = shuffle(Array.from({ length: NP }, (_, i) => i), r);
      rank1 = new Int32Array(NP); rank2 = new Int32Array(NP);
      p1.forEach((i, k) => (rank1[i] = k)); p2.forEach((i, k) => (rank2[i] = k));
      key = '';
    };
    const measure = () => {
      const m = Math.round((1 - st.a) * NP), k = Math.round(st.eta * NP);
      let agree = 0, flips = 0;
      for (let i = 0; i < NP; i++) {
        const inW = rank1[i] < m, inF = rank2[i] < k;
        if (inF) flips++;
        if (inW === inF) agree++; // model and reference both wrong, or both right: they agree
      }
      return { agree: agree / NP, flips, m, k };
    };
    const out = readoutBox(ctl);

    const draw = (ms) => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const plotTop = h * 0.58;
      const cell = Math.min((w - 20) / GW, (plotTop - 46) / GH), ox = (w - cell * GW) / 2, oy = 6;
      const m = ms.m, k = ms.k;
      for (let i = 0; i < NP; i++) {
        const x = i % GW, y = (i / GW) | 0;
        const inW = rank1[i] < m, inF = rank2[i] < k;
        const agreeRef = inW === inF;
        const cx = ox + (x + 0.5) * cell, cy = oy + (y + 0.5) * cell;
        g.beginPath(); g.arc(cx, cy, cell * 0.3, 0, 6.2832);
        g.fillStyle = agreeRef ? C.green : C.red; g.fill();
        if (inF) { g.beginPath(); g.arc(cx, cy, cell * 0.46, 0, 6.2832); g.strokeStyle = C.purple; g.lineWidth = 2; g.stroke(); }
      }
      // accuracy plot (expected curve, dashed truth, measured point)
      const v = AI.view(c, 0, 0.4, 0.5, 1.0, { l: 44, r: 12, t: plotTop, b: 26 });
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '11px ui-monospace,Menlo,Consolas,monospace';
      for (let yv = 0.5; yv <= 1.0001; yv += 0.1) { g.beginPath(); g.moveTo(v.pad.l, v.Y(yv) + 0.5); g.lineTo(v.pad.l + v.pw, v.Y(yv) + 0.5); g.stroke(); g.textAlign = 'right'; g.fillText(yv.toFixed(1), v.pad.l - 5, v.Y(yv) + 4); }
      for (let xv = 0; xv <= 0.4001; xv += 0.1) { g.textAlign = 'center'; g.fillText(xv.toFixed(1), v.X(xv), v.pad.t + v.ph + 14); }
      g.fillText('flip rate η', v.pad.l + v.pw / 2, v.pad.t + v.ph + 14 + 12);
      g.save(); g.setLineDash([6, 5]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(v.X(0), v.Y(st.a)); g.lineTo(v.X(0.4), v.Y(st.a)); g.stroke(); g.restore();
      g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath();
      for (let s = 0; s <= 100; s++) { const e = (0.4 * s) / 100, y = (1 - e) * st.a + e * (1 - st.a); s ? g.lineTo(v.X(e), v.Y(y)) : g.moveTo(v.X(e), v.Y(y)); }
      g.stroke(); g.lineWidth = 1;
      const expv = (1 - ms.eta) * st.a + ms.eta * (1 - st.a);
      g.beginPath(); g.arc(v.X(ms.eta), v.Y(expv), 4, 0, 6.2832); g.strokeStyle = C.blue; g.stroke();
      g.beginPath(); g.arc(v.X(ms.eta), v.Y(ms.agree), 5, 0, 6.2832); g.fillStyle = C.orange; g.fill();
      g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('dashed = true accuracy a', v.pad.l + 8, v.pad.t + 10);
      g.fillStyle = C.blue; g.fillText('blue = expected (1−η)a + η(1−a)', v.pad.l + 8, v.pad.t + 24);
    };

    buttons(ctl, [
      ['sweep', 'Sweep', (el) => { st.sweep = !st.sweep; el.textContent = st.sweep ? 'Stop sweep' : 'Sweep'; if (st.sweep && st.etaT >= 0.4) st.etaT = 0; }],
      ['play', 'Pause', (el) => { st.playing = !st.playing; el.textContent = st.playing ? 'Pause' : 'Play'; }],
      ['new', 'New labels', () => { st.seed = (st.seed % 97) + 3; build(); }],
    ]);
    sliderRow(ctl, 'model true accuracy a', 0.5, 1, 0.01, st.a, (v) => v.toFixed(2), (v) => { st.a = v; key = ''; });
    sliderRow(ctl, 'reference flip rate η', 0, 0.4, 0.01, st.etaT, (v) => v.toFixed(2), (v) => { st.etaT = v; st.sweep = false; });
    sliderRow(ctl, 'speed (sweep and easing)', 0.3, 2, 0.1, st.speed, (v) => v.toFixed(1) + '×', (v) => { st.speed = v; });

    build();
    AI.loop(root, (dt) => {
      if (st.playing) {
        if (st.sweep) { st.etaT = Math.min(0.4, st.etaT + dt * 0.03 * st.speed); if (st.etaT >= 0.4) st.sweep = false; }
        st.eta += (st.etaT - st.eta) * Math.min(1, dt * 2.5 * st.speed);
        if (Math.abs(st.etaT - st.eta) < 1e-4) st.eta = st.etaT;
      }
      const ms = measure();
      ms.eta = st.eta;
      const k2 = `${ms.k}|${st.a}|${c.w}x${c.h}|${document.documentElement.dataset.theme}`;
      if (k2 === key) return;
      key = k2;
      draw(ms);
      const expv = (1 - st.eta) * st.a + st.eta * (1 - st.a);
      cap.textContent = `Ring = the reference label was flipped (${ms.k} of ${NP} points). Red = model and reference disagree, so the point counts as an error.`;
      out.innerHTML = `flip rate shown η = ${st.eta.toFixed(3)} → ${ms.k} reference labels flipped<br>`
        + `true accuracy a = ${st.a.toFixed(2)}  (the model's accuracy against the truth)<br>`
        + `measured accuracy = 1 − |W △ F| / N = <b style="color:var(--orange)">${ms.agree.toFixed(3)}</b><br>`
        + `expected (1−η)a + η(1−a) = <b style="color:var(--blue)">${expv.toFixed(3)}</b><br>`
        + `score lost to noise = a − measured = ${(st.a - ms.agree).toFixed(3)}`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8.5 error map
   * ===================================================================== */
  AI.labs.errormap = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Error map of a land-cover product', d.sub || 'error type, class and region filters',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const st = { rate: 0.2, seed: 7, mode: 'type', region: 'all', cls: 'all', rows: 0, playing: true, speed: 1.5 };
    let map = null, key = '';
    const tick = AI.ticker();
    c.onresize = () => { key = ''; };
    const rebuild = () => { map = G.makeMap(st.seed, st.rate); st.rows = 0; st.playing = true; key = ''; };
    const out = readoutBox(ctl);

    const active = (i) => {
      const x = i % W, y = (i / W) | 0;
      if (st.region !== 'all' && G.region(x, y) !== st.region) return false;
      if (st.cls !== 'all' && map.ref[i] !== +st.cls && map.prod[i] !== +st.cls) return false;
      return true;
    };

    const draw = () => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const cell = Math.min((w - 10) / W, (h - 30) / H), ox = (w - cell * W) / 2, oy = 20;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, px = ox + x * cell, py = oy + y * cell;
        const shown = y < st.rows;
        const on = active(i);
        g.globalAlpha = on ? 1 : 0.16;
        if (!shown) g.fillStyle = C.bg3;
        else if (st.mode === 'type') g.fillStyle = map.isErr[i] ? C.red : rgba(C.green, 0.4);
        else g.fillStyle = C[CL[map.ref[i]]];
        g.fillRect(px, py, cell + 0.4, cell + 0.4);
        if (shown && st.mode === 'class' && map.isErr[i]) { g.beginPath(); g.arc(px + cell / 2, py + cell / 2, cell * 0.28, 0, 6.2832); g.fillStyle = C.red; g.fill(); }
        g.globalAlpha = 1;
      }
      g.strokeStyle = C.muted; g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(ox + (W / 2) * cell, oy); g.lineTo(ox + (W / 2) * cell, oy + H * cell); g.moveTo(ox, oy + (H / 2) * cell); g.lineTo(ox + W * cell, oy + (H / 2) * cell); g.stroke();
      g.setLineDash([]);
      if (st.rows < H) { const sy = oy + st.rows * cell; g.fillStyle = C.fg; g.fillRect(ox, sy - 1, W * cell, 2); }
      g.fillStyle = C.muted; g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'left';
      g.fillText(st.mode === 'type' ? 'red = wrong pixel, green = correct (revealed rows)' : 'colour = reference class; red dot = wrong pixel', ox, 13);
    };

    const stats = () => {
      const regs = { NW: [0, 0], NE: [0, 0], SW: [0, 0], SE: [0, 0] };
      let revealed = 0, omis = [0, 0, 0], comm = [0, 0, 0], ref = [0, 0, 0], prodC = [0, 0, 0];
      for (let i = 0; i < map.N; i++) {
        const x = i % W, y = (i / W) | 0, rk = (y < H / 2 ? 'N' : 'S') + (x < W / 2 ? 'W' : 'E');
        regs[rk][0]++; if (map.isErr[i]) regs[rk][1]++;
        if (y < st.rows && map.isErr[i]) revealed++;
        ref[map.ref[i]]++; prodC[map.prod[i]]++;
        if (map.isErr[i]) { omis[map.ref[i]]++; comm[map.prod[i]]++; }
      }
      return { regs, revealed, omis, comm, ref, prodC };
    };

    buttons(ctl, [
      ['play', 'Pause', (el) => { st.playing = !st.playing; el.textContent = st.playing ? 'Pause' : (st.rows >= H ? 'Replay' : 'Play'); if (st.rows >= H) { st.rows = 0; st.playing = true; } }],
      ['step', 'Step row', () => { if (st.rows < H) st.rows++; st.playing = false; }],
      ['all', 'Reveal all', () => { st.rows = H; st.playing = false; }],
      ['new', 'New map', () => { st.seed = (st.seed % 97) + 3; rebuild(); }],
    ]);
    selectRow(ctl, 'colour by', [['type', 'error type (correct / wrong)'], ['class', 'reference class']], st.mode, (v) => { st.mode = v; key = ''; });
    selectRow(ctl, 'region', [['all', 'whole map'], ['NW', 'north-west'], ['NE', 'north-east'], ['SW', 'south-west'], ['SE', 'south-east']], st.region, (v) => { st.region = v; key = ''; });
    selectRow(ctl, 'class (ref or map)', CLS_OPTS, st.cls, (v) => { st.cls = v; key = ''; });
    sliderRow(ctl, 'product error rate', 0, 0.4, 0.01, st.rate, (v) => v.toFixed(2), (v) => { st.rate = v; rebuild(); });
    sliderRow(ctl, 'scan speed', 0.5, 6, 0.5, st.speed, (v) => v.toFixed(1) + ' rows/s', (v) => { st.speed = v; });

    rebuild();
    AI.loop(root, (dt) => {
      if (st.playing && st.rows < H) {
        const n = tick(dt, st.speed);
        if (n > 0) st.rows = Math.min(H, st.rows + n);
        if (st.rows >= H) st.playing = false;
      }
      const k2 = `${st.rows}|${st.mode}|${st.region}|${st.cls}|${st.seed}|${st.rate}|${c.w}x${c.h}|${document.documentElement.dataset.theme}`;
      if (k2 === key) return;
      key = k2;
      draw();
      const s = stats();
      const shownErr = s.revealed;
      cap.textContent = `Scanning row ${st.rows} of ${H}. Wrong pixels revealed so far: ${shownErr}. Look for blocks of red, not for single dots.`;
      const rr = (k) => { const [n, e] = s.regs[k]; return `<tr><td>${k === 'NW' ? 'north-west' : k === 'NE' ? 'north-east' : k === 'SW' ? 'south-west' : 'south-east'}</td><td>${n}</td><td>${e}</td><td><b>${pct(1 - e / n, 0)}</b></td></tr>`; };
      out.innerHTML = `<table class="tbl" style="margin:0 0 8px"><tr><th>region (full map)</th><th>pixels</th><th>wrong</th><th>accuracy</th></tr>${['NW', 'NE', 'SW', 'SE'].map(rr).join('')}</table>`
        + `<table class="tbl" style="margin:0"><tr><th>class</th><th>reference n</th><th>omission (missed)</th><th>commission (false alarm)</th></tr>`
        + [0, 1, 2].map((k) => `<tr><td>${CNAME[k]}</td><td>${s.ref[k]}</td><td>${s.omis[k]}</td><td>${s.comm[k]}</td></tr>`).join('') + `</table>`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8.6 uncertainty map
   * ===================================================================== */
  AI.labs.uncmap = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Probability maps, entropy and review', d.sub || 'per-pixel class probabilities, entropy and flagged area',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const c = AI.canvas(AI.$('canvas', view)), cap = AI.$('.caption', view);
    const st = { beta: 2.2, tau: 0.9, cls: 0, seed: 5, px: 10, cur: 0, speed: 18, playing: true, step: false };
    let P = null, Hn = null, flag = null, key = '';
    const tick = AI.ticker();
    const LOG3 = Math.log2(3);
    c.onresize = () => { key = ''; };
    const compute = () => {
      const N = W * H;
      const f = [0, 1, 2].map((k) => G.field(st.seed * 5 + k + 11, W, H, 9));
      P = new Float32Array(N * 3); Hn = new Float32Array(N); flag = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        const z = [0, 1, 2].map((k) => st.beta * f[k][i]);
        const mx = Math.max(z[0], z[1], z[2]);
        const e = z.map((v) => Math.exp(v - mx)), s = e[0] + e[1] + e[2];
        let h = 0;
        for (let k = 0; k < 3; k++) { const p = e[k] / s; P[i * 3 + k] = p; if (p > 0) h -= p * Math.log2(p); }
        Hn[i] = h; flag[i] = h > st.tau ? 1 : 0;
      }
      key = '';
    };
    const out = readoutBox(ctl);

    const draw = () => {
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      g.clearRect(0, 0, w, h);
      const gap = 16, pw = (w - gap) / 2, cell = Math.min(pw / W, (h - 30) / H);
      const ox = [(pw - cell * W) / 2, pw + gap + (pw - cell * W) / 2], oy = 22;
      const titles = [`P(${CNAME[st.cls]}) per pixel`, 'entropy H (bits), outlined = flagged'];
      g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center'; g.fillStyle = C.muted;
      g.fillText(titles[0], ox[0] + (cell * W) / 2, 14); g.fillText(titles[1], ox[1] + (cell * W) / 2, 14);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        g.fillStyle = ramp([C.bg, C[CL[st.cls]]], P[i * 3 + st.cls]);
        g.fillRect(ox[0] + x * cell, oy + y * cell, cell + 0.4, cell + 0.4);
        g.fillStyle = ramp([C.blue, C.yellow, C.red], Hn[i] / LOG3);
        g.fillRect(ox[1] + x * cell, oy + y * cell, cell + 0.4, cell + 0.4);
        if (flag[i]) { g.strokeStyle = C.fg; g.lineWidth = 1.2; g.strokeRect(ox[1] + x * cell + 0.6, oy + y * cell + 0.6, cell - 1.2, cell - 1.2); }
      }
      const i = Math.floor(st.cur) % (W * H), cx = i % W, cy = (i / W) | 0;
      g.strokeStyle = C.fg; g.lineWidth = 2;
      for (let p = 0; p < 2; p++) g.strokeRect(ox[p] + cx * cell, oy + cy * cell, cell, cell);
      g.lineWidth = 1;
    };

    buttons(ctl, [
      ['play', 'Pause', (el) => { st.playing = !st.playing; el.textContent = st.playing ? 'Pause' : 'Play'; }],
      ['step', 'Step', () => { st.cur = Math.floor(st.cur) + 1; st.playing = false; key = ''; }],
      ['new', 'New map', () => { st.seed = (st.seed % 97) + 3; compute(); }],
    ]);
    sliderRow(ctl, 'sharpness β (higher = more confident)', 0.5, 6, 0.1, st.beta, (v) => v.toFixed(1), (v) => { st.beta = v; compute(); });
    sliderRow(ctl, 'flag if H above (bits)', 0, 1.58, 0.01, st.tau, (v) => v.toFixed(2) + ' bits', (v) => { st.tau = v; compute(); });
    selectRow(ctl, 'class shown', [['0', 'Forest'], ['1', 'Crop'], ['2', 'Urban']], st.cls, (v) => { st.cls = +v; key = ''; });
    sliderRow(ctl, 'pixel size (m)', 5, 30, 5, st.px, (v) => v + ' m', (v) => { st.px = v; key = ''; });
    sliderRow(ctl, 'cursor speed', 5, 60, 1, st.speed, (v) => v + ' px/s', (v) => { st.speed = v; });

    compute();
    AI.loop(root, (dt) => {
      if (st.playing) st.cur += dt * st.speed;
      const idx = Math.floor(st.cur) % (W * H);
      const k2 = `${st.beta}|${st.tau}|${st.cls}|${st.seed}|${st.px}|${idx}|${c.w}x${c.h}|${document.documentElement.dataset.theme}`;
      if (k2 === key) return;
      key = k2;
      draw();
      let nF = 0; for (let i = 0; i < W * H; i++) nF += flag[i];
      const i = idx, x = i % W, y = (i / W) | 0;
      const p = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
      const am = p.indexOf(Math.max(p[0], p[1], p[2]));
      const areaKm2 = (nF * st.px * st.px) / 1e6;
      cap.textContent = `Cursor (${x}, ${y}): most likely ${CNAME[am]} at ${(p[am] * 100).toFixed(0)}%, entropy ${Hn[i].toFixed(3)} bits, ${flag[i] ? 'above' : 'below'} the ${st.tau.toFixed(2)}-bit flag. Lower sharpness raises entropy; the most likely class can stay the same.`;
      out.innerHTML = `p = (${p.map((v) => v.toFixed(3)).join(', ')}) for (Forest, Crop, Urban) · H = <b>${Hn[i].toFixed(3)}</b> bits · max possible ${LOG3.toFixed(3)}<br>`
        + `flagged pixels: <b>${nF}</b> of ${W * H} (${pct(nF / (W * H), 1)}) · flagged area = ${nF} × ${st.px}² m² = <b>${areaKm2.toFixed(4)} km²</b>`;
    }, { fps: 30 });
  };
})();
