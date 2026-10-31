/* GeoAI Module 7 labs (foundation models & embeddings). Requires lib.js.
 *   mimreconstruct   <div data-lab="mimreconstruct"></div>   masked patches -> baseline reconstruction -> error map
 *   tsstack          <div data-lab="tsstack"></div>          NDVI time series, clouds, linear gap filling, editable phenology
 *   embedsphere      <div data-lab="embedsphere"></div>      tile embeddings on a sphere, cosine top-k search from a click
 *   finetunedial     <div data-lab="finetunedial"></div>     freeze depth / adapters -> trainable parameter count, schematic curves
 *   zoofilter        <div data-lab="zoofilter"></div>        synthetic model catalogue, staged filters, compatibility check
 * All records, tiles and curves are synthetic and illustrative; the parameter counts are exact for a ViT-B/16-like encoder.
 */
(function () {
  const $ = (s, r) => AI.$(s, r);
  const $$ = (s, r) => AI.$$(s, r);
  const TAU = Math.PI * 2;

  /* ---------- shared transport: lab time with play / pause / step / restart / speed ---------- */
  function clock(root, ctl, o) {
    const st = { t: 0, playing: true, speed: o.speed || 1, total: o.total, ms: o.milestones || [], onReset: o.onReset };
    ctl.insertAdjacentHTML('beforeend',
      `<label>Speed <span class="v" data-sv>×${st.speed}</span><input type="range" min="0.25" max="3" step="0.25" value="${st.speed}" data-sp></label>
       <div class="row"><button class="btn primary" data-play>⏸ Pause</button><button class="btn" data-step>Step</button><button class="btn" data-reset>Restart</button></div>`);
    const sp = $('[data-sp]', ctl), sv = $('[data-sv]', ctl), pb = $('[data-play]', ctl);
    const setP = (p) => { st.playing = p; pb.textContent = p ? '⏸ Pause' : '▶ Play'; };
    sp.oninput = () => { st.speed = +sp.value; sv.textContent = '×' + st.speed; };
    pb.onclick = () => { if (st.t >= st.total) st.t = 0; setP(!st.playing); };
    $('[data-step]', ctl).onclick = () => {
      setP(false);
      const nx = st.ms.find((m) => m > st.t + 1e-6);
      st.t = nx !== undefined ? nx : st.total;
    };
    $('[data-reset]', ctl).onclick = () => { st.t = 0; setP(true); if (st.onReset) st.onReset(); };
    st.advance = (dt) => { if (st.playing) { st.t += dt * st.speed; if (st.t >= st.total) { st.t = st.total; setP(false); } } };
    st.restart = () => { st.t = 0; setP(true); };
    return st;
  }

  function scaledImage(g, cw, px, x, y, size, data, N) {
    // data: Uint8ClampedArray RGBA of N x N; drawn with nearest-neighbour so patches stay crisp
    const im = new ImageData(data, N, N);
    cw.putImageData(im, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(cw.canvas, x, y, size, size);
  }
  function offscreen(N) {
    const c = document.createElement('canvas'); c.width = c.height = N;
    return c.getContext('2d');
  }

  /* =====================================================================
   * 1. Masked-patch reconstruction
   * ===================================================================== */
  AI.labs.mimreconstruct = function (root) {
    const N = 32;
    const { view, ctl } = AI.labShell(root, 'Mask, hide, reconstruct',
      'a 32 × 32 tile with three bands shown as false colour (NIR, red, green); masked values are scored, visible ones are not',
      `<canvas style="height:300px"></canvas><div class="caption" data-cap></div>`,
      `<label>Mask ratio <span class="v" data-mv></span><input type="range" min="0" max="95" step="5" value="75" data-mr></label>
       <label>Patch size <select data-ps><option value="2">2 × 2 px (256 patches)</option><option value="4" selected>4 × 4 px (64 patches)</option><option value="8">8 × 8 px (16 patches)</option></select></label>
       <label>Scene texture (noise) <span class="v" data-tv></span><input type="range" min="0" max="0.3" step="0.01" value="0.08" data-tx></label>
       <div class="row"><button class="btn" data-seed>New scene &amp; new mask</button></div>`);
    const cv = $('canvas', view), c = AI.canvas(cv), cw = offscreen(N);
    const q = (s) => $(s, ctl);
    let seed = 11, S = null, dirty = true, lastKey = '';

    function makeScene(sd, tex) {
      const r = AI.rng(sd), base = [0.18, 0.11, 0.48], bands = [0, 1, 2].map(() => new Float32Array(N * N));
      for (let b = 0; b < 3; b++) {
        const terms = [0, 1, 2].map(() => ({ a: 0.05 + r() * 0.07, kx: 1 + Math.floor(r() * 3), ky: 1 + Math.floor(r() * 3), px: r() * TAU, py: r() * TAU }));
        const blocks = Array.from({ length: 9 }, () => ({ x: Math.floor(r() * 26), y: Math.floor(r() * 26), w: 4 + Math.floor(r() * 7), h: 4 + Math.floor(r() * 7), d: (r() - 0.5) * 0.25 }));
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          let v = base[b];
          for (const t of terms) v += t.a * Math.sin(TAU * t.kx * x / N + t.px) * Math.sin(TAU * t.ky * y / N + t.py);
          for (const k of blocks) if (x >= k.x && x < k.x + k.w && y >= k.y && y < k.y + k.h) v += k.d;
          v += tex * AI.randn(r);
          bands[b][y * N + x] = AI.clamp(v, 0, 1);
        }
      }
      return bands;
    }
    function rgbOf(bands, idx) {
      // false colour: NIR -> red channel, red -> green, green -> blue
      return [bands[2][idx] * 255 * 1.1, bands[0][idx] * 255 * 1.9, bands[1][idx] * 255 * 2.6];
    }

    function rebuildAll() {
      const ratio = +q('[data-mr]').value / 100, P = +q('[data-ps]').value, G = N / P, n = G * G, tex = +q('[data-tx]').value;
      q('[data-mv]').textContent = Math.round(ratio * 100) + '%';
      q('[data-tv]').textContent = tex.toFixed(2);
      const bands = makeScene(seed, tex);
      // patch means per band
      const pm = [0, 1, 2].map(() => new Float32Array(n));
      for (let gi = 0; gi < G; gi++) for (let gj = 0; gj < G; gj++) for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let y = gi * P; y < gi * P + P; y++) for (let x = gj * P; x < gj * P + P; x++) s += bands[b][y * N + x];
        pm[b][gi * G + gj] = s / (P * P);
      }
      // hiding order (nested: a higher ratio hides a superset)
      const r = AI.rng(seed + 101), perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      const order = new Int32Array(n).fill(-1); perm.forEach((p, i) => (order[p] = i));
      const count = Math.round(ratio * n), masked = (p) => order[p] < count;
      // baseline: average of visible patch values in the smallest square around the hidden patch
      const pred = [0, 1, 2].map(() => new Float32Array(n));
      let se = 0, nv = 0;
      for (let p = 0; p < n; p++) {
        if (!masked(p)) continue;
        const gi = Math.floor(p / G), gj = p % G;
        for (let R = 1; R <= G; R++) {
          let cnt = 0; const acc = [0, 0, 0];
          for (let i = Math.max(0, gi - R); i <= Math.min(G - 1, gi + R); i++)
            for (let j = Math.max(0, gj - R); j <= Math.min(G - 1, gj + R); j++) {
              const k = i * G + j; if (masked(k)) continue;
              cnt++; for (let b = 0; b < 3; b++) acc[b] += pm[b][k];
            }
          if (cnt) { for (let b = 0; b < 3; b++) pred[b][p] = acc[b] / cnt; break; }
        }
        for (let y = gi * P; y < gi * P + P; y++) for (let x = gj * P; x < gj * P + P; x++)
          for (let b = 0; b < 3; b++) { const e = bands[b][y * N + x] - pred[b][p]; se += e * e; nv++; }
      }
      S = { bands, P, G, n, count, masked, order, pred, mse: nv ? se / nv : NaN, nv, ratio };
      dirty = true;
    }

    function panel(g, x0, y0, size, fn) {
      const d = new Uint8ClampedArray(N * N * 4);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const [r, gg, b] = fn(x, y), o = (y * N + x) * 4;
        d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = 255;
      }
      scaledImage(g, cw, 0, x0, y0, size, d, N);
    }

    const cap = $('[data-cap]', view);
    const clk = clock(root, ctl, { total: 14, milestones: [3, 5, 11, 14], onReset: () => (dirty = true) });
    $$('input,select', ctl).forEach((el) => el.addEventListener('input', rebuildAll));
    $$('select', ctl).forEach((el) => el.addEventListener('change', rebuildAll));
    $('[data-seed]', ctl).onclick = () => { seed = Math.floor(Math.random() * 9999); rebuildAll(); };
    rebuildAll();

    const grey = [120, 120, 130];
    AI.loop(root, (dt) => {
      clk.advance(dt);
      const t = clk.t, hideP = AI.clamp((t - 3) / 2, 0, 1), recP = AI.clamp((t - 5) / 6, 0, 1), errP = AI.clamp((t - 11) / 1.5, 0, 1);
      const key = [Math.round(t * 20), S.P, S.ratio, seed, cv.width, cv.height, document.documentElement.dataset.theme].join('|');
      if (key === lastKey && !dirty) return;
      lastKey = key; dirty = false;
      const g = c.ctx, W = c.w, H = c.h, gap = 10, size = Math.min((W - 3 * gap) / 4, H - 24);
      g.clearRect(0, 0, W, H);
      const x0 = (W - (4 * size + 3 * gap)) / 2;
      const titles = ['① full tile', '② input: hidden patches', '③ model reconstruction', '④ error on hidden values'];
      const { bands, P, G, count, order, pred, masked } = S, rc = recP * count, full = Math.floor(rc), frac = rc - full;
      const hideCount = Math.floor(hideP * count);
      const panelX = (i) => x0 + i * (size + gap);
      // panel 1: truth
      panel(g, panelX(0), 22, size, (x, y) => rgbOf(bands, y * N + x));
      // panel 2: input
      panel(g, panelX(1), 22, size, (x, y) => {
        const p = Math.floor(y / P) * G + Math.floor(x / P);
        if (masked(p) && order[p] < hideCount) return grey;
        return rgbOf(bands, y * N + x);
      });
      // panel 3: reconstruction (predictions use the same false-colour mapping as the truth)
      const predRGB = (p) => [pred[2][p] * 255 * 1.1, pred[0][p] * 255 * 1.9, pred[1][p] * 255 * 2.6];
      panel(g, panelX(2), 22, size, (x, y) => {
        const p = Math.floor(y / P) * G + Math.floor(x / P), i = y * N + x;
        if (!masked(p) || order[p] >= hideCount) return rgbOf(bands, i);
        const o = order[p];
        if (o < full) return predRGB(p);
        if (o === full) { const pc = predRGB(p); return [0, 1, 2].map((k) => grey[k] + (pc[k] - grey[k]) * frac); }
        return grey;
      });
      // panel 4: error map (after reconstruction)
      panel(g, panelX(3), 22, size, (x, y) => {
        const p = Math.floor(y / P) * G + Math.floor(x / P), i = y * N + x;
        if (!masked(p) || errP <= 0) return [22, 24, 30];
        let e = 0;
        for (let b = 0; b < 3; b++) e += Math.abs(bands[b][i] - pred[b][p]);
        e = (e / 3) * 3.0 * errP;
        return [AI.clamp(e * 255, 0, 255), AI.clamp(e * 140, 0, 255), 30];
      });
      g.fillStyle = AI.col().muted; g.font = '11px sans-serif'; g.textAlign = 'center';
      for (let i = 0; i < 4; i++) g.fillText(titles[i], panelX(i) + size / 2, 14);
      const caps = [
        `① The full tile: every value is known. In real pretraining these are unlabelled archive tiles.`,
        `② Hide ${Math.round(S.ratio * 100)}% of the patches (${count} of ${S.n}). The encoder will see only the visible ones.`,
        `③ The model fills each hidden patch from its visible neighbours. This baseline averages them, so every hidden patch becomes flat.`,
        `④ Score only the hidden values: the error map and the masked mean squared error. Visible patches are never scored.`,
      ];
      const phase = t < 3 ? 0 : t < 5 ? 1 : t < 11 ? 2 : t < 14 ? 3 : 3;
      cap.textContent = caps[phase] + (t >= 14 ? ' (finished: change a slider or restart)' : '');
      const mse = errP > 0 ? S.mse : NaN;
      const ro = ctl.querySelector('.readout') || (() => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; })();
      ro.innerHTML = `patches: <b>${S.n}</b> (${S.G} × ${S.G}) · hidden <b class="o">${count}</b> · visible <b class="g">${S.n - count}</b>\n` +
        `values scored (hidden × 3 bands): <b>${S.nv}</b> of ${N * N * 3} (${Math.round(100 * S.nv / (N * N * 3))}%)\n` +
        `masked MSE of the baseline: <b class="r">${Number.isFinite(mse) ? mse.toFixed(4) : '—'}</b>` +
        `\n<span class="m">A trained decoder replaces the flat average with learned texture and context.</span>`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 2. Time series of NDVI: clouds, observations and linear gap filling
   * ===================================================================== */
  AI.labs.tsstack = function (root) {
    const { view, ctl } = AI.labShell(root, 'One pixel through a season',
      'observations (dots) on a true phenology curve; clouds remove some; the orange line fills the gaps by linear interpolation in time',
      `<canvas style="height:300px"></canvas><div class="caption" data-cap></div>`,
      `<label>True NDVI curve y(t), t in days <input type="text" spellcheck="false" value="0.2+0.5*exp(-((t-200)/60)^2)" data-f></label>
       <div class="err" data-err></div>
       <label>Dates in the season <span class="v" data-nv></span><input type="range" min="4" max="12" step="1" value="4" data-n></label>
       <label>Cloud probability <span class="v" data-pv></span><input type="range" min="0" max="0.9" step="0.05" value="0.25" data-pc></label>
       <div class="row"><button class="btn" data-seed>New clouds</button></div>`);
    const cv = $('canvas', view), c = AI.canvas(cv);
    const q = (s) => $(s, ctl);
    let f = null, seed = 5, D = null, dirty = true, lastKey = '';
    const YMIN = 0, YMAX = 1, XMIN = 0, XMAX = 365;
    const fin = q('[data-f]');
    const formula = AI.exprInput(fin, q('[data-err]'), { allowed: ['t'], onChange: (fn) => { f = fn; rebuild(); } });

    function rebuild() {
      if (!f) return;
      const n = +q('[data-n]').value, pc = +q('[data-pc]').value;
      q('[data-nv]').textContent = n; q('[data-pv]').textContent = Math.round(pc * 100) + '%';
      const r = AI.rng(seed), step = 365 / n, days = [];
      for (let i = 0; i < n; i++) days.push(Math.round(AI.clamp((i + 0.5) * step + (r() - 0.5) * 8, 0, 365)));
      const clouded = days.map(() => r() < pc);
      let kept = days.map((d, i) => i).filter((i) => !clouded[i]);
      if (kept.length < 2) kept = [0, n - 1].filter((v, i, a) => a.indexOf(v) === i);
      const obs = days.map((d) => ({ d, y: Math.round(clampY(f({ t: d })) * 100) / 100 }));
      const keptObs = kept.map((i) => ({ d: days[i], y: obs[i].y })).sort((a, b) => a.d - b.d);
      // linear interpolation in time over a daily grid; hold the nearest observation at the two ends
      const fill = [];
      for (let t = 0; t <= 365; t++) {
        let v;
        if (t <= keptObs[0].d) v = keptObs[0].y;
        else if (t >= keptObs[keptObs.length - 1].d) v = keptObs[keptObs.length - 1].y;
        else {
          let k = 0; while (keptObs[k + 1].d < t) k++;
          const a = keptObs[k], b = keptObs[k + 1];
          v = a.y + (b.y - a.y) * (t - a.d) / (b.d - a.d);
        }
        fill.push(v);
      }
      let se = 0, big = 0;
      for (let t = 0; t <= 365; t++) { const e = fill[t] - clampY(f({ t })); se += e * e; }
      for (let k = 1; k < keptObs.length; k++) big = Math.max(big, keptObs[k].d - keptObs[k - 1].d);
      D = { days, clouded, obs, keptObs, fill, rmse: Math.sqrt(se / 366), big, n, kept: keptObs.length };
      clk.total = n * 0.9 + 4 + 1; clk.ms = [n * 0.9, n * 0.9 + 4];
      dirty = true;
    }
    const clampY = (v) => (Number.isFinite(v) ? AI.clamp(v, YMIN, YMAX) : 0);

    const cap = $('[data-cap]', view);
    const clk = clock(root, ctl, { total: 10, milestones: [3.6, 7.6], onReset: () => (dirty = true) });
    $$('input', ctl).forEach((el) => el.addEventListener('input', rebuild));
    $('[data-seed]', ctl).onclick = () => { seed = Math.floor(Math.random() * 9999); rebuild(); };

    AI.loop(root, (dt) => {
      clk.advance(dt);
      if (!D) return;
      const t = clk.t, n = D.n, revealN = AI.clamp(Math.floor(t / 0.9), 0, n);
      const fillP = AI.clamp((t - n * 0.9) / 4, 0, 1);
      const key = [Math.round(t * 20), seed, f && f.toString().length, +q('[data-n]').value, +q('[data-pc]').value, cv.width, cv.height, document.documentElement.dataset.theme].join('|');
      if (key === lastKey && !dirty) return;
      lastKey = key; dirty = false;
      const g = c.ctx, W = c.w, H = c.h, Cc = AI.col();
      const v = AI.view(c, XMIN, XMAX, YMIN, YMAX);
      g.clearRect(0, 0, W, H); AI.axes(g, v);
      g.fillStyle = Cc.muted; g.font = '11px sans-serif'; g.textAlign = 'center';
      g.fillText('day of season →', v.pad.l + v.pw / 2, H - 4);
      // true curve
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.strokeStyle = Cc.blue; g.lineWidth = 2.5; g.beginPath();
      for (let k = 0; k <= 365; k++) { const x = v.X(k), y = v.Y(clampY(f({ t: k }))); k ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      // gap-filled line, drawn left to right as fillP grows
      const upto = Math.floor(fillP * 365);
      if (upto > 0) {
        g.strokeStyle = Cc.orange; g.lineWidth = 2; g.setLineDash([]); g.beginPath();
        for (let k = 0; k <= upto; k++) { const x = v.X(k), y = v.Y(D.fill[k]); k ? g.lineTo(x, y) : g.moveTo(x, y); }
        g.stroke();
      }
      g.restore();
      // observations
      D.days.forEach((d, i) => {
        if (i >= revealN) return;
        const x = v.X(d), y = v.Y(D.obs[i].y);
        if (D.clouded[i]) {
          g.strokeStyle = Cc.muted; g.lineWidth = 1.5; g.beginPath();
          g.moveTo(x - 5, y - 5); g.lineTo(x + 5, y + 5); g.moveTo(x + 5, y - 5); g.lineTo(x - 5, y + 5); g.stroke();
        } else {
          g.fillStyle = Cc.green; g.beginPath(); g.arc(x, y, 5, 0, TAU); g.fill();
        }
      });
      g.fillStyle = Cc.muted; g.font = '10px sans-serif'; g.textAlign = 'left';
      g.fillText('● observed   ✕ cloud (no value)   — orange: linear fill', v.pad.l + 6, v.pad.t + 12);
      const left = D.days.filter((d, i) => i < revealN && !D.clouded[i]).length;
      const cp = t < n * 0.9 ? `dates appear one at a time: ${revealN} of ${n} shown; ✕ marks a cloud` : fillP < 1 ? 'linear interpolation between neighbouring observations, in time (not in index)' : 'filled curve complete: compare it with the blue true curve';
      cap.textContent = cp;
      const ro = ctl.querySelector('.readout') || (() => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; })();
      ro.innerHTML = `observed (kept): <b class="g">${D.kept}</b> of ${n} dates · clouded: <b class="r">${n - D.kept}</b>\n` +
        `largest gap between kept dates: <b class="o">${D.big}</b> days\n` +
        `RMSE of filled curve vs true, days 0–365: <b>${D.rmse.toFixed(3)}</b>\n` +
        `<span class="m">A 30-day gap is filled with a line; a long winter gap would invent a green-up.</span>`;
    }, { fps: 30 });
    formula.run();
  };

  /* =====================================================================
   * 3. Embeddings on a sphere, cosine top-k search from a click
   * ===================================================================== */
  AI.labs.embedsphere = function (root) {
    const { view, ctl } = AI.labShell(root, 'Tiles as points on a sphere',
      'each dot is a tile embedding normalised to length 1; click a dot to use it as the query',
      `<canvas style="height:340px;cursor:pointer"></canvas><div class="caption" data-cap></div>`,
      `<label>k (matches) <span class="v" data-kv></span><input type="range" min="2" max="10" step="1" value="5" data-k></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" data-raw> show raw lengths and rank by raw dot product</label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" data-spin checked> slow rotation</label>
       <div class="row"><button class="btn" data-clear>Clear query</button></div>`);
    const cv = $('canvas', view), c = AI.canvas(cv);
    const q = (s) => $(s, ctl);
    const GROUPS = [{ name: 'forest', col: 'green', c: [0.35, 0.8, 0.45] }, { name: 'water', col: 'blue', c: [-0.6, -0.25, 0.75] }, { name: 'urban', col: 'orange', c: [0.2, -0.75, -0.6] }];
    const r = AI.rng(21), N = 90, pts = [];
    for (let i = 0; i < N; i++) {
      const g = i % 3, cc = GROUPS[g].c;
      let v = [cc[0] + AI.randn(r) * 0.28, cc[1] + AI.randn(r) * 0.28, cc[2] + AI.randn(r) * 0.28];
      const L = Math.hypot(...v); const u = v.map((x) => x / L);
      pts.push({ g, u, len: 0.7 + 0.6 * r(), raw: u.map((x, k) => x * (0.7 + 0.6 * r())) });
    }
    pts.forEach((p) => (p.raw = p.u.map((x) => x * p.len)));
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let yaw = 0.4, drag = null, query = -1, qT = 0, t = 0, matches = [], simOf = [], spinOn = true, rawOn = false, dirty = true, lastKey = '';
    const TILT = 0.35;
    const project = (p, R, cx, cy, sc) => {
      const [x, y, z] = p;
      const x1 = x * Math.cos(yaw) + z * Math.sin(yaw), z1 = -x * Math.sin(yaw) + z * Math.cos(yaw);
      const y2 = y * Math.cos(TILT) - z1 * Math.sin(TILT), z2 = y * Math.sin(TILT) + z1 * Math.cos(TILT);
      return { sx: cx + R * sc * x1, sy: cy - R * sc * y2, depth: z2 };
    };
    const slerp = (a, b, s) => {
      const cosT = AI.clamp(dot(a, b), -1, 1), th = Math.acos(cosT);
      if (th < 1e-6) return a;
      const A = Math.sin((1 - s) * th) / Math.sin(th), B = Math.sin(s * th) / Math.sin(th);
      return [0, 1, 2].map((k) => A * a[k] + B * b[k]);
    };
    function setQuery(i) {
      query = i; qT = t;
      const P = rawOn ? pts.map((p) => p.raw) : pts.map((p) => p.u);
      const sims = P.map((v) => dot(P[i], v));
      simOf = sims;
      const k = +q('[data-k]').value;
      matches = sims.map((s, j) => j).filter((j) => j !== i).sort((a, b) => sims[b] - sims[a]).slice(0, k);
      dirty = true;
    }
    cv.addEventListener('pointerdown', (e) => {
      const rc = cv.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
      const R = Math.min(c.w, c.h) * 0.42, cx = c.w / 2, cy = c.h / 2;
      let best = -1, bd = 14;
      pts.forEach((p, i) => {
        const s = project(p.u, R, cx, cy, 1);
        const d = Math.hypot(s.sx - mx, s.sy - my);
        if (d < bd && s.depth > -0.3) { bd = d; best = i; }
      });
      if (best >= 0) setQuery(best); else drag = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('pointermove', (e) => { if (drag) { yaw += (e.clientX - drag.x) * 0.01; drag = { x: e.clientX, y: e.clientY }; dirty = true; } });
    window.addEventListener('pointerup', () => (drag = null));
    q('[data-k]').oninput = () => { q('[data-kv]').textContent = q('[data-k]').value; if (query >= 0) setQuery(query); };
    q('[data-kv]').textContent = q('[data-k]').value;
    q('[data-raw]').onchange = (e) => { rawOn = e.target.checked; if (query >= 0) setQuery(query); };
    q('[data-spin]').onchange = (e) => (spinOn = e.target.checked);
    q('[data-clear]').onclick = () => { query = -1; matches = []; dirty = true; };
    const cap = $('[data-cap]', view);
    const clk = clock(root, ctl, { total: 1e9, milestones: [], onReset: () => { t = 0; query = -1; matches = []; dirty = true; } });

    AI.loop(root, (dt) => {
      clk.advance(dt);
      if (clk.playing && spinOn) { yaw += dt * 0.15 * clk.speed; dirty = true; }
      t = clk.t;
      const reveal = query >= 0 ? Math.min(matches.length, Math.floor((t - qT) / 0.6) + 1) : 0;
      const key = [Math.round(yaw * 60), query, reveal, rawOn, +q('[data-k]').value, c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (key === lastKey && !dirty) return;
      lastKey = key; dirty = false;
      const g = c.ctx, W = c.w, H = c.h, Cc = AI.col();
      g.clearRect(0, 0, W, H);
      const R = Math.min(W, H) * 0.42, cx = W / 2, cy = H / 2;
      g.strokeStyle = Cc.border; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, R, 0, TAU); g.stroke();
      // draw points: back hemisphere faint
      const proj = pts.map((p) => project(rawOn ? p.raw : p.u, R, cx, cy, 1));
      pts.forEach((p, i) => {
        const s = proj[i], front = s.depth >= 0;
        g.globalAlpha = front ? 0.95 : 0.22;
        g.fillStyle = Cc[GROUPS[p.g].col];
        g.beginPath(); g.arc(s.sx, s.sy, i === query ? 3 : 4.2, 0, TAU); g.fill();
      });
      g.globalAlpha = 1;
      if (query >= 0) {
        const qs = proj[query];
        // arcs along the great circle from query to each revealed match, brightness ~ cosine
        for (let m = 0; m < reveal; m++) {
          const j = matches[m], cosv = dot(pts[query].u, pts[j].u);
          g.strokeStyle = Cc.fg; g.lineWidth = 1.6; g.globalAlpha = 0.25 + 0.7 * AI.clamp(cosv, 0, 1);
          g.beginPath();
          for (let s = 0; s <= 24; s++) {
            const pt = slerp(pts[query].u, pts[j].u, s / 24);
            const pr = project(pt, R, cx, cy, 1);
            s ? g.lineTo(pr.sx, pr.sy) : g.moveTo(pr.sx, pr.sy);
          }
          g.stroke();
          const sj = proj[j];
          g.globalAlpha = 1; g.strokeStyle = Cc.fg; g.lineWidth = 2;
          g.beginPath(); g.arc(sj.sx, sj.sy, 7, 0, TAU); g.stroke();
        }
        g.globalAlpha = 1; g.strokeStyle = Cc.fg; g.lineWidth = 3;
        g.beginPath(); g.arc(qs.sx, qs.sy, 9, 0, TAU); g.stroke();
        g.fillStyle = Cc.fg; g.font = '11px sans-serif'; g.textAlign = 'left';
        g.fillText('query', qs.sx + 12, qs.sy - 8);
      }
      g.globalAlpha = 1;
      g.fillStyle = Cc.muted; g.font = '11px sans-serif'; g.textAlign = 'left';
      GROUPS.forEach((gr, k) => { g.fillStyle = Cc[gr.col]; g.fillText('● ' + gr.name, 10, 16 + k * 14); });
      if (query < 0) cap.textContent = 'Click a dot (drag the background to turn the sphere). Its neighbours will be found by cosine similarity.';
      else {
        const same = matches.slice(0, reveal).filter((j) => pts[j].g === pts[query].g).length;
        cap.textContent = `query is a ${GROUPS[pts[query].g].name} tile; matches appear one by one, brighter arcs = higher cosine (${rawOn ? 'raw dot product' : 'unit vectors'}).`;
        const ro = ctl.querySelector('.readout') || (() => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; })();
        const lines = matches.slice(0, reveal).map((j, m) => `${m + 1}. tile ${j} (${GROUPS[pts[j].g].name}) ${rawOn ? 'dot' : 'cos'} = ${(rawOn ? simOf[j] : dot(pts[query].u, pts[j].u)).toFixed(4)}`);
        ro.innerHTML = `k = ${matches.length} · same land-cover group as the query: <b class="g">${same}</b> of ${reveal}\n` + (lines.join('\n') || '<span class="m">waiting for matches…</span>');
      }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4. Fine-tuning dial: freeze depth, adapters, parameter counts, schematic curves
   * ===================================================================== */
  AI.labs.finetunedial = function (root) {
    const { view, ctl } = AI.labShell(root, 'How much of the encoder to train',
      'ViT-B/16-like encoder: 12 blocks, D = 768, MLP 3072, 10-class head; parameter counts are exact, the learning curves are schematic',
      `<canvas style="height:300px"></canvas><div class="readout" data-ro></div><div class="caption" data-cap></div>`,
      `<label>Mode <select data-mode><option value="partial" selected>partial fine-tuning (freeze first blocks)</option><option value="adapters">adapters (all original weights frozen)</option></select></label>
       <label>Frozen blocks <span class="v" data-kv></span><input type="range" min="0" max="12" step="1" value="8" data-k></label>
       <label>Adapter bottleneck r <select data-r><option>4</option><option>8</option><option>16</option><option>32</option><option selected>64</option><option>128</option><option>256</option></select></label>
       <label>Labelled tiles N <select data-n><option>50</option><option selected>200</option><option>2000</option></select></label>`);
    const cv = $('canvas', view), c = AI.canvas(cv);
    const q = (s) => $(s, ctl);
    const TOTAL = 85798656 + 7690, D = 768, BLOCK = 7087872, PE = 590592, POS = 151296, CLS = 768, FN = 1536, HEAD = 7690, L = 12;
    let lastKey = '', dirty = true;
    const counts = () => {
      const mode = q('[data-mode]').value, k = +q('[data-k]').value, r = +q('[data-r]').value;
      let trainable, adapter = 0, frozenPart;
      if (mode === 'adapters') {
        adapter = L * (2 * D * r + r + D);
        trainable = adapter + HEAD; frozenPart = TOTAL - HEAD;
      } else if (k === 0) {
        trainable = TOTAL; frozenPart = 0;
      } else {
        trainable = (L - k) * BLOCK + HEAD; frozenPart = TOTAL - trainable;
      }
      return { mode, k, r, trainable, adapter, frozenPart, share: trainable / TOTAL };
    };
    const curve = (share, N, e) => {
      const cap = Math.sqrt(share), ceiling = 0.62 + 0.28 * cap * Math.min(1, Math.sqrt(N / 400));
      const tau = 3 + 8 * (1 - cap), over = 0.12 * cap * cap * Math.max(0, 1 - N / 1000) * (e / 30);
      return ceiling - (ceiling - 0.5) * Math.exp(-e / tau) - over;
    };
    const fmtN = (v) => Math.round(v).toLocaleString('en-US');
    const cap = $('[data-cap]', view), ro = $('[data-ro]', view);
    const clk = clock(root, ctl, { total: 30, milestones: [], onReset: () => (dirty = true) });
    $$('input,select', ctl).forEach((el) => el.addEventListener('input', () => { q('[data-kv]').textContent = q('[data-k]').value; dirty = true; }));
    q('[data-kv]').textContent = q('[data-k]').value;

    AI.loop(root, (dt) => {
      clk.advance(dt);
      const e = clk.t, ct = counts(), N = +q('[data-n]').value;
      const key = [Math.round(e * 10), ct.trainable, N, c.w, c.h, document.documentElement.dataset.theme].join('|');
      if (key === lastKey && !dirty) return;
      lastKey = key; dirty = false;
      q('[data-kv]').textContent = ct.k;
      const g = c.ctx, W = c.w, H = c.h, Cc = AI.col();
      g.clearRect(0, 0, W, H);
      // left: stacked bar of parameter groups; right: schematic curves
      const bw = W * 0.36, bx = 14, by = 24, bh = H - 56;
      const trainFrac = ct.trainable / TOTAL;
      g.fillStyle = Cc.bg2; g.fillRect(bx, by, bw, bh);
      g.fillStyle = Cc.green; g.fillRect(bx, by + bh * (1 - trainFrac), bw, bh * trainFrac);
      g.fillStyle = Cc.muted; g.fillRect(bx, by, bw, bh * (1 - trainFrac));
      g.fillStyle = Cc.fg; g.font = '11px sans-serif'; g.textAlign = 'left';
      g.fillText('trainable (green) vs frozen', bx, 14);
      g.fillStyle = Cc.fg; g.font = 'bold 12px sans-serif';
      g.fillText(`${(trainFrac * 100).toFixed(2)}%`, bx + 6, by + bh * (1 - trainFrac) + 16);
      const px = W * 0.46, pw = W - px - 16, ph = H - 56, py = 24;
      const vv = { X: (x) => px + (x / 30) * pw, Y: (y) => py + (1 - (y - 0.4) / 0.6) * ph };
      g.strokeStyle = Cc.border; g.lineWidth = 1; g.beginPath(); g.moveTo(px, py); g.lineTo(px, py + ph); g.lineTo(px + pw, py + ph); g.stroke();
      g.fillStyle = Cc.muted; g.font = '10px sans-serif'; g.textAlign = 'center';
      g.fillText('epoch', px + pw / 2, py + ph + 16); g.textAlign = 'left';
      g.fillText('score (schematic)', px + 4, py + 10);
      const plot = (fn, col, w, alpha) => {
        g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = w; g.beginPath();
        for (let x = 0; x <= Math.min(30, e); x += 0.25) { const y = fn(x); x ? g.lineTo(vv.X(x), vv.Y(y)) : g.moveTo(vv.X(x), vv.Y(y)); }
        g.stroke(); g.globalAlpha = 1;
      };
      const probeShare = HEAD / TOTAL, fullShare = 1, ourShare = ct.share;
      plot((x) => curve(probeShare, N, x), Cc.muted, 1.5, 0.7);
      plot((x) => curve(fullShare, N, x), Cc.purple, 1.5, 0.7);
      plot((x) => curve(ourShare, N, x), Cc.green, 2.8, 1);
      g.fillStyle = Cc.muted; g.font = '10px sans-serif'; g.textAlign = 'left';
      g.fillText('grey: linear probe · purple: full · green: your setting', px + 6, py + ph - 6);
      cap.textContent = ct.mode === 'adapters'
        ? `Adapters: each block gains a D→r→D bottleneck; the 85.8 M original weights never change. Try r from 4 to 256.`
        : ct.k === 0 ? 'Full fine-tuning: every weight is trainable, including the patch embedding that must match your bands.'
        : `The first ${ct.k} block${ct.k > 1 ? 's' : ''} stay frozen; the other ${12 - ct.k} blocks and the head learn from your tiles.`;
      const pc = (x) => (x * 100).toFixed(4) + '%';
      ro.innerHTML = `trainable: <b class="g">${fmtN(ct.trainable)}</b> of ${fmtN(TOTAL)} (${pc(trainFrac)})\n` +
        (ct.mode === 'adapters'
          ? `adapters: 12 × (2·768·${ct.r} + ${ct.r} + 768) = <b>${fmtN(ct.adapter)}</b> · head ${fmtN(HEAD)}`
          : ct.k === 0 ? `every block + patch embedding + head: <b>${fmtN(TOTAL)}</b>`
          : `${12 - ct.k} trainable blocks: ${12 - ct.k} × ${fmtN(BLOCK)} = <b>${fmtN((12 - ct.k) * BLOCK)}</b> · head ${fmtN(HEAD)}`) +
        `\nfrozen: ${fmtN(ct.frozenPart)} · patch embedding ${fmtN(PE)} · positions ${fmtN(POS)} · class token ${CLS} · final norm ${FN} (frozen unless full)` +
        `\n<span class="m">Curves are illustrative shapes only: they show how capacity and labels change the trade-off, not measured results.</span>`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 5. Model zoo: synthetic STAC MLM records, staged filters, compatibility
   * ===================================================================== */
  AI.labs.zoofilter = function (root) {
    const BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'];
    const NATIVE = { B02: 10, B03: 10, B04: 10, B08: 10, B11: 20, B12: 20 };
    const RECS = [
      { id: 'synth-lc-seg-a', task: 'semantic-segmentation', bands: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'], gsd: 10, params: 24100000, arch: 'U-Net', scale: 'reflectance' },
      { id: 'synth-scene-cls-b', task: 'scene-classification', bands: ['B02', 'B03', 'B04', 'B08'], gsd: 10, params: 11200000, arch: 'ResNet-style CNN', scale: 'reflectance' },
      { id: 'synth-lc-seg-c', task: 'semantic-segmentation', bands: ['B04', 'B08'], gsd: 10, params: 6800000, arch: 'U-Net (small)', scale: 'reflectance' },
      { id: 'synth-lc-seg-d', task: 'semantic-segmentation', bands: ['B02', 'B03', 'B04', 'B08'], gsd: 30, params: 24100000, arch: 'U-Net', scale: 'reflectance' },
      { id: 'synth-lc-seg-e', task: 'semantic-segmentation', bands: ['B02', 'B03', 'B04', 'B08'], gsd: 10, params: 41500000, arch: 'Transformer decoder', scale: 'reflectance' },
    ];
    const { view, ctl } = AI.labShell(root, 'A model catalogue, filtered in stages',
      'synthetic records (invented for this lesson), shown as simplified STAC MLM fields; click a row to see its record',
      `<div class="zoo-table" data-table></div><div class="readout" data-json></div><div class="caption" data-cap></div>`,
      `<label>Task <select data-task><option value="any">any</option><option value="semantic-segmentation" selected>semantic-segmentation</option><option value="scene-classification">scene-classification</option></select></label>
       <label>Search: model must accept these bands</label>
       <div class="row" data-req>${BANDS.map((b) => `<label style="flex-direction:row;gap:4px;align-items:center;color:var(--fg)"><input type="checkbox" value="${b}" ${['B02', 'B03', 'B04', 'B08'].includes(b) ? 'checked' : ''}>${b}</label>`).join('')}</div>
       <label>GSD limit <select data-gsd><option value="any">any</option><option value="10" selected>≤ 10 m</option><option value="20">≤ 20 m</option><option value="30">≤ 30 m</option></select></label>
       <label>My data: bands I have <span class="v">native GSD</span></label>
       <div class="row" data-have>${BANDS.map((b) => `<label style="flex-direction:row;gap:4px;align-items:center;color:var(--fg)"><input type="checkbox" value="${b}" checked>${b} <span class="m">${NATIVE[b]} m</span></label>`).join('')}</div>
       <label>My data: input scale <select data-scale><option value="reflectance" selected>reflectance 0–1</option><option value="dn">digital numbers 0–10,000</option></select></label>`);
    const q = (s) => $(s, ctl);
    const tbl = $('[data-table]', view), jsonEl = $('[data-json]', view), cap = $('[data-cap]', view);
    let sel = 0, lastKey = '', dirty = true, stageOK = [];

    const reqBands = () => $$('[data-req] input', ctl).filter((i) => i.checked).map((i) => i.value);
    const haveBands = () => $$('[data-have] input', ctl).filter((i) => i.checked).map((i) => i.value);
    const gsdLimit = () => (q('[data-gsd]').value === 'any' ? Infinity : +q('[data-gsd]').value);

    function stages() {
      const task = q('[data-task]').value, req = reqBands(), lim = gsdLimit();
      const s1 = RECS.map((r) => task === 'any' || r.task === task);
      const s2 = RECS.map((r, i) => s1[i] && req.every((b) => r.bands.includes(b)));
      const s3 = RECS.map((r, i) => s2[i] && r.gsd <= lim);
      return [s1, s2, s3];
    }
    function verdict(r) {
      const have = haveBands();
      const missing = r.bands.filter((b) => !have.includes(b));
      if (missing.length) return { cls: 'r', txt: 'incompatible: missing ' + missing.join(', ') };
      if (q('[data-scale]').value !== r.scale) return { cls: 'r', txt: 'incompatible: scale (model expects reflectance 0–1)' };
      const rs = r.bands.filter((b) => NATIVE[b] !== r.gsd);
      if (rs.length) return { cls: 'o', txt: `needs resampling: ${rs.join(', ')} (native ${NATIVE[rs[0]]} m → ${r.gsd} m)` };
      return { cls: 'g', txt: 'compatible' };
    }
    function drawTable(stage) {
      // stage: number of filters applied (0..3) during the animation; shown rows fade as they are removed
      const [s1, s2, s3] = stageOK;
      const pass = [s1, s2, s3];
      const rows = RECS.map((r, i) => {
        const alive = stage >= 3 ? s3[i] : stage >= 2 ? s2[i] : stage >= 1 ? s1[i] : true;
        const ok = alive ? 1 : 0.22;
        const cls = i === sel ? 'sel' : '';
        const v = verdict(r);
        return `<tr class="${cls}" data-i="${i}" style="opacity:${ok}">
          <td>${r.id}</td><td>${r.task}</td><td>${r.bands.join(' ')}</td><td>${r.gsd}</td><td>${(r.params / 1e6).toFixed(1)} M</td>
          <td><span style="color:var(--${v.cls === 'o' ? 'orange' : v.cls === 'g' ? 'green' : 'red'})">${v.txt}</span></td></tr>`;
      }).join('');
      tbl.innerHTML = `<table class="tbl"><tr><th>record</th><th>task</th><th>bands</th><th>GSD</th><th>params</th><th>compatibility with my data</th></tr>${rows}</table>`;
      $$('tr[data-i]', tbl).forEach((tr) => (tr.onclick = () => { sel = +tr.dataset.i; dirty = true; }));
      const counts = pass.map((arr) => arr.filter(Boolean).length);
      cap.textContent = stage === 0 ? 'All five records in the catalogue.'
        : stage === 1 ? `filter 1, task: ${counts[0]} records remain`
        : stage === 2 ? `filter 2, bands: ${counts[1]} remain (filter 1 gave ${counts[0]})`
        : `filter 3, GSD: ${counts[2]} remain. Open them, then check compatibility with your data.`;
      const r = RECS[sel];
      jsonEl.innerHTML = `<b>${r.id}</b> (simplified STAC MLM fields)\n` +
        `"mlm:architecture": "${r.arch}",\n"mlm:tasks": ["${r.task}"],\n"mlm:total_parameters": ${r.params},\n"gsd": ${r.gsd},\n` +
        `"mlm:input": { "bands": [${r.bands.map((b) => '"' + b + '"').join(', ')}], "scale": "${r.scale} 0–1" }`;
    }

    const clk = clock(root, ctl, { total: 6.5, milestones: [2, 4, 6], onReset: () => (dirty = true) });
    $$('input,select', ctl).forEach((el) => el.addEventListener('change', () => { clk.restart(); dirty = true; }));
    $$('input[type=checkbox]', ctl).forEach((el) => el.addEventListener('input', () => { clk.restart(); dirty = true; }));

    AI.loop(root, (dt) => {
      clk.advance(dt);
      stageOK = stages();
      const stage = clk.t < 2 ? 0 : clk.t < 4 ? 1 : clk.t < 6 ? 2 : 3;
      const key = [stage, sel, reqBands().join(), haveBands().join(), q('[data-task]').value, q('[data-gsd]').value, q('[data-scale]').value].join('|');
      if (key === lastKey && !dirty) return;
      lastKey = key; dirty = false;
      drawTable(stage);
    }, { fps: 30 });
  };
})();
