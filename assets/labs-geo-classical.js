/* GeoAI · Module 4 labs (spectral indices & classical GeoAI).
 *   geoSignatures  spectral reflectance curves, a band window, a noisy pixel matched to the nearest signature
 *   geoNdvi        editable red / NIR per pixel, NDVI painted cell by cell, threshold mask and area
 *   geoPixelClass  k-nearest-neighbour classifier on a two-band synthetic scene, feature plane and accuracy
 *   geoObjects     SLIC-style superpixels grown iteration by iteration, object features on click
 *   geoZonal       a polygon over a raster, exact cell coverage weights, weighted statistics
 *   geoChange      two NDVI dates, difference map, threshold, noise slider and changed-area counter
 * Values marked illustrative are not measurements. All motion is dt-based; redraws happen only when state changes.
 */
(function () {
  const clamp = AI.clamp;
  const ff = (v, d = 4) => (Number.isFinite(v) ? String(+v.toFixed(d)) : '–');
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const ramp = (stops, t) => {
    t = clamp(t, 0, 1) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t));
    return AI.mix(stops[i], stops[i + 1], t - i);
  };
  // NDVI colour scale: blue (water, negative) -> pale (bare) -> green (dense)
  const NDVI_STOPS = ['#2563a8', '#e9e3c4', '#c8c26a', '#7fbf5a', '#1d7a3a', '#0b4a24'];
  const ndviColor = (v) => ramp(NDVI_STOPS, (v + 0.5) / 1.4);
  const hue = (k) => `hsl(${(k * 137.5) % 360} 70% 55%)`;

  /* ---------- shared controls ---------- */
  function transport(ctl, speed0) {
    const st = { playing: true, speed: speed0 || 1, stepReq: 0, restartReq: false };
    const row = el('div', 'row');
    row.innerHTML = '<button class="btn small" data-a="play">⏸ Pause</button><button class="btn small" data-a="step">Step</button><button class="btn small" data-a="rep">↺ Replay</button>';
    const sp = el('label', '', '<span>Speed <span class="v"></span></span><input type="range" min="0.25" max="3" step="0.05" value="' + st.speed + '" style="width:120px">');
    sp.style.flexDirection = 'row'; sp.style.alignItems = 'center'; sp.style.gap = '8px';
    const si = sp.querySelector('input'), sv = sp.querySelector('.v');
    const upd = () => { st.speed = +si.value; sv.textContent = (+si.value).toFixed(2) + '×'; };
    si.oninput = upd; upd();
    row.appendChild(sp); ctl.appendChild(row);
    const pb = row.querySelector('[data-a=play]');
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    row.querySelector('[data-a=step]').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; st.stepReq++; };
    row.querySelector('[data-a=rep]').onclick = () => { st.restartReq = true; st.playing = true; pb.textContent = '⏸ Pause'; };
    return st;
  }
  function slider(ctl, name, min, max, step, val, fmt, onChange) {
    const l = el('label', '', `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`);
    const inp = l.querySelector('input'), out = l.querySelector('.v');
    const show = () => { out.textContent = fmt ? fmt(+inp.value) : ff(+inp.value, 3); };
    inp.oninput = () => { show(); if (onChange) onChange(+inp.value); };
    show(); ctl.appendChild(l);
    return { get: () => +inp.value, set: (x) => { inp.value = x; show(); }, el: l };
  }
  const readout = (ctl) => { const r = el('div', 'readout'); ctl.appendChild(r); return r; };

  /* =====================================================================
   * 4.1  Spectral signatures
   * ===================================================================== */
  const WL = [490, 560, 665, 740, 842, 1000, 1610, 2190];
  // illustrative surface reflectance at the wavelengths of WL (0 to 1)
  const MATS = [
    { n: 'Healthy vegetation', c: 'green', v: [0.02, 0.08, 0.04, 0.20, 0.45, 0.46, 0.20, 0.09] },
    { n: 'Crop', c: 'yellow', v: [0.04, 0.12, 0.10, 0.16, 0.25, 0.30, 0.22, 0.12] },
    { n: 'Bare soil', c: 'orange', v: [0.12, 0.20, 0.25, 0.27, 0.30, 0.32, 0.38, 0.34] },
    { n: 'Clear water', c: 'blue', v: [0.08, 0.10, 0.05, 0.02, 0.02, 0.01, 0.005, 0.002] },
    { n: 'Concrete', c: 'purple', v: [0.16, 0.17, 0.19, 0.20, 0.22, 0.25, 0.27, 0.25] },
  ];
  const S2 = [{ n: 'B2', w: 490, i: 0 }, { n: 'B3', w: 560, i: 1 }, { n: 'B4', w: 665, i: 2 }, { n: 'B8', w: 842, i: 4 }];
  const curveAt = (m, wl) => {
    if (wl <= WL[0]) return m.v[0];
    for (let i = 0; i < WL.length - 1; i++) {
      if (wl <= WL[i + 1]) return AI.lerp(m.v[i], m.v[i + 1], (wl - WL[i]) / (WL[i + 1] - WL[i]));
    }
    return m.v[WL.length - 1];
  };
  const windowMean = (m, c, w) => { let s = 0; for (let k = 0; k <= 8; k++) s += curveAt(m, c - w / 2 + (w * k) / 8); return s / 9; };

  AI.labs.geoSignatures = function (root) {
    const { view, ctl } = AI.labShell(root, 'Spectral signatures',
      'reflectance by wavelength · a band window · one noisy pixel matched to the nearest signature',
      '<canvas style="height:240px"></canvas><div class="caption"></div><canvas style="height:150px;margin-top:8px"></canvas>', '');
    const cTop = AI.canvas(view.querySelectorAll('canvas')[0]), cBot = AI.canvas(view.querySelectorAll('canvas')[1]);
    const cap = view.querySelector('.caption');
    const st = transport(ctl, 1);
    let dirty = true, lastKey = '', seq = 0, pix = null, acc = 0, shown = 0;
    const sWin = slider(ctl, 'Band window centre (100 nm wide)', 400, 2300, 10, 842, (v) => v + ' nm', () => (dirty = true));
    const sNoise = slider(ctl, 'Sensor noise on the pixel', 0, 0.06, 0.005, 0.02, null, () => (dirty = true));
    const ro = readout(ctl);
    const row = el('div', 'row');
    row.innerHTML = '<button class="btn small" data-a="new">New pixel</button><select style="width:auto"><option value="-1">random surface</option>'
      + MATS.map((m, i) => `<option value="${i}">${m.n}</option>`).join('') + '</select>';
    ctl.appendChild(row);
    const selMat = row.querySelector('select');
    const newPixel = () => {
      seq++;
      const r = AI.rng(seq * 7919 + 13);
      const truth = +selMat.value >= 0 ? +selMat.value : Math.floor(r() * MATS.length);
      pix = { truth, z: S2.map(() => AI.randn(r)) };
      acc = 0; shown = 0; dirty = true;
    };
    row.querySelector('[data-a=new]').onclick = newPixel;
    selMat.onchange = newPixel;
    const obsOf = () => S2.map((b, k) => clamp(MATS[pix.truth].v[b.i] + sNoise.get() * pix.z[k], 0, 1));
    const distances = (obs) => MATS.map((m) => Math.sqrt(S2.reduce((a, b, k) => a + (obs[k] - m.v[b.i]) ** 2, 0)));
    newPixel();

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; acc = 0; dirty = true; }
      if (st.stepReq) { acc = Math.floor(acc) + st.stepReq; st.stepReq = 0; dirty = true; }
      if (st.playing && acc < 4) { acc += dt * st.speed * 0.8; } // about 1.25 s per band at 1x speed
      const ns = Math.min(4, Math.floor(acc));
      if (ns !== shown) { shown = ns; dirty = true; }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), c = sWin.get(), obs = obsOf(), d = distances(obs);
      const nearest = d.indexOf(Math.min(...d));

      /* top: signatures */
      const g = cTop.ctx;
      g.clearRect(0, 0, cTop.w, cTop.h);
      const v = AI.view(cTop, 400, 2300, 0, 0.5, { l: 40, r: 12, t: 12, b: 26 });
      AI.axes(g, v);
      g.globalAlpha = 0.22; g.fillStyle = C.blue;
      g.fillRect(v.X(c - 50), v.pad.t, v.X(c + 50) - v.X(c - 50), v.ph);
      g.globalAlpha = 1;
      g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
      S2.forEach((b) => {
        g.setLineDash([3, 3]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(v.X(b.w), v.pad.t); g.lineTo(v.X(b.w), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.muted; g.fillText(b.n, v.X(b.w), v.pad.t + 10);
      });
      MATS.forEach((m) => {
        g.strokeStyle = C[m.c]; g.lineWidth = 2; g.beginPath();
        for (let wl = 490; wl <= 2300; wl += 10) { const x = v.X(wl), y = v.Y(curveAt(m, wl)); wl === 490 ? g.moveTo(x, y) : g.lineTo(x, y); }
        g.stroke();
        g.fillStyle = C[m.c];
        WL.forEach((x, k) => { g.beginPath(); g.arc(v.X(x), v.Y(m.v[k]), 2.5, 0, 7); g.fill(); });
      });
      g.textAlign = 'left'; g.fillStyle = C.muted;
      g.fillText('shaded: band window (illustrative 100 nm)', v.pad.l + 6, v.pad.t + v.ph - 6);

      /* bottom: the pixel, band by band */
      const gb = cBot.ctx;
      gb.clearRect(0, 0, cBot.w, cBot.h);
      const vb = AI.view(cBot, 0, 4, 0, 0.5, { l: 40, r: 10, t: 10, b: 26 });
      gb.font = '11px ui-monospace,Menlo,Consolas,monospace';
      gb.fillStyle = C.muted; gb.textAlign = 'right';
      for (let yy = 0; yy <= 0.5001; yy += 0.1) gb.fillText(yy.toFixed(1), vb.pad.l - 5, vb.Y(yy) + 4);
      S2.forEach((b, k) => {
        const xc = vb.X(k + 0.5), bw = vb.pw / 4 * 0.42;
        if (k < shown) { gb.fillStyle = C.fg; gb.globalAlpha = 0.8; gb.fillRect(xc - bw / 2, vb.Y(obs[k]), bw, vb.Y(0) - vb.Y(obs[k])); gb.globalAlpha = 1; }
        else { gb.fillStyle = C.muted; gb.textAlign = 'center'; gb.fillText('…', xc, vb.Y(0.02)); }
        MATS.forEach((m) => { gb.fillStyle = C[m.c]; gb.beginPath(); gb.arc(xc, vb.Y(m.v[b.i]), 3.5, 0, 7); gb.fill(); });
        gb.fillStyle = C.muted; gb.textAlign = 'center'; gb.fillText(`${b.n} ${b.w} nm`, xc, vb.Y(0) + 15);
      });

      /* readout + caption */
      const lines = [`window ${c - 50}–${c + 50} nm, mean reflectance:`];
      MATS.forEach((m) => lines.push(`  ${m.n.padEnd(18, ' ')} ${ff(windowMean(m, c, 100), 3)}`));
      lines.push('');
      lines.push(`pixel bands read: ${shown} of 4` + (shown > 0 ? '  ' + S2.slice(0, shown).map((b, k) => `${b.n}=${ff(obs[k], 3)}`).join(' ') : ''));
      if (shown >= 4) {
        lines.push('distance to each signature:');
        MATS.forEach((m, i) => lines.push(`  ${m.n.padEnd(18, ' ')} ${ff(d[i], 4)}`));
        lines.push(`nearest: ${MATS[nearest].n}   (true surface: ${MATS[pix.truth].n})`);
      }
      ro.textContent = lines.join('\n');
      if (shown < 4) {
        cap.textContent = `Reading band ${S2[shown].n} (${S2[shown].w} nm). A pixel is four numbers, one per band; the next band arrives in about a second.`;
      } else {
        cap.textContent = `All four bands read. Each signature's distance is the square root of the summed squared band differences; the closest signature wins: ${MATS[nearest].n}.`;
      }
      const key = [shown, c, sNoise.get(), pix.truth, seq].join('|');
      if (key === lastKey) return; lastKey = key;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4.2  NDVI calculator
   * ===================================================================== */
  AI.labs.geoNdvi = function (root) {
    const { view, ctl } = AI.labShell(root, 'NDVI calculator',
      'edit one pixel’s red and NIR · the NDVI image is painted cell by cell · the threshold sets the vegetation mask',
      '<canvas style="height:200px;cursor:pointer"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(view.querySelector('canvas')), cap = view.querySelector('.caption');
    const N = 6;
    const LAYOUT = ['FFFCCC', 'FFFCCC', 'FFCFFS', 'SSCWWS', 'SSSWWS', 'SSSWWC'];   // F forest, C crop, S soil, W water
    const TYPE = { F: [0.04, 0.45], C: [0.10, 0.25], S: [0.25, 0.30], W: [0.05, 0.02] };
    const r = AI.rng(11);
    const red = [], nir = [];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const t = TYPE[LAYOUT[j][i]];
      red.push(+clamp(t[0] + (r() - 0.5) * 0.02, 0.01, 0.6).toFixed(3));
      nir.push(+clamp(t[1] + (r() - 0.5) * 0.06, 0.01, 0.7).toFixed(3));
    }
    const ndvi = (k) => { const s = nir[k] + red[k]; return s > 0 ? (nir[k] - red[k]) / s : 0; };
    const st = transport(ctl, 1);
    let sel = 0, paint = 0, pacc = 0, dirty = true, lastKey = '';
    const sRed = slider(ctl, 'Red of the selected cell', 0.01, 0.6, 0.005, red[0], null, (v) => { red[sel] = v; paint = Math.min(paint, sel); dirty = true; });
    const sNir = slider(ctl, 'NIR of the selected cell', 0.01, 0.7, 0.005, nir[0], null, (v) => { nir[sel] = v; paint = Math.min(paint, sel); dirty = true; });
    const sThr = slider(ctl, 'Vegetation threshold (NDVI)', -0.2, 0.9, 0.01, 0.3, null, () => (dirty = true));
    const ro = readout(ctl);
    const geom = () => {
      const W = cv.w, H = cv.h, gap = 16;
      const ps = Math.min((W - 2 * gap) / 3, H - 26), cell = ps / N;
      const x0 = (W - (3 * ps + 2 * gap)) / 2, y0 = 6;
      return { ps, cell, panels: [0, 1, 2].map((p) => x0 + p * (ps + gap)), y0 };
    };
    cv.cv.addEventListener('click', (e) => {
      const G = geom(), x = e.offsetX, y = e.offsetY;
      for (let p = 0; p < 3; p++) {
        const px = x - G.panels[p], py = y - G.y0;
        if (px >= 0 && px < G.ps && py >= 0 && py < G.ps) {
          const col = Math.floor(px / G.cell), row = Math.floor(py / G.cell);
          sel = row * N + col;
          sRed.set(red[sel]); sNir.set(nir[sel]);
          dirty = true;
          return;
        }
      }
    });
    cv.onresize = () => (dirty = true);

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; paint = 0; pacc = 0; dirty = true; }
      if (st.stepReq) { paint = Math.min(N * N, paint + st.stepReq); st.stepReq = 0; dirty = true; }
      if (st.playing && paint < N * N) { pacc += dt * st.speed * 5; const n = Math.floor(pacc); if (n) { pacc -= n; paint = Math.min(N * N, paint + n); dirty = true; } }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), g = cv.ctx, G = geom(), thr = sThr.get();
      g.clearRect(0, 0, cv.w, cv.h);
      const titles = ['Red (B4)', 'NIR (B8)', 'NDVI'];
      g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
      for (let p = 0; p < 3; p++) {
        const X = G.panels[p], Y = G.y0;
        for (let k = 0; k < N * N; k++) {
          const cx = X + (k % N) * G.cell, cy = Y + Math.floor(k / N) * G.cell;
          if (p === 0) { const gv = Math.round(255 * clamp(red[k] / 0.5, 0, 1)); g.fillStyle = `rgb(${gv},${gv},${gv})`; }
          else if (p === 1) { const gv = Math.round(255 * clamp(nir[k] / 0.6, 0, 1)); g.fillStyle = `rgb(${gv},${gv},${gv})`; }
          else if (k < paint) g.fillStyle = ndviColor(ndvi(k));
          else g.fillStyle = C.bg2;
          g.fillRect(cx, cy, G.cell + 0.5, G.cell + 0.5);
          if (p === 2 && k < paint && ndvi(k) < thr) { g.globalAlpha = 0.55; g.fillStyle = C.bg; g.fillRect(cx, cy, G.cell + 0.5, G.cell + 0.5); g.globalAlpha = 1; }
        }
        g.fillStyle = C.muted; g.fillText(titles[p], X + G.ps / 2, G.y0 + G.ps + 14);
        g.strokeStyle = C.fg; g.lineWidth = 2;
        g.strokeRect(X + (sel % N) * G.cell + 1, G.y0 + Math.floor(sel / N) * G.cell + 1, G.cell - 2, G.cell - 2);
      }
      // readout
      const rr = red[sel], nn = nir[sel], nv = ndvi(sel);
      const mask = [...Array(N * N).keys()].filter((k) => ndvi(k) >= thr).length;
      const area = mask * 100 / 10000;
      ro.textContent = `selected cell: row ${Math.floor(sel / N)}, col ${sel % N}\n`
        + `NDVI = (NIR − Red) / (NIR + Red)\n`
        + `     = (${ff(nn, 3)} − ${ff(rr, 3)}) / (${ff(nn, 3)} + ${ff(rr, 3)}) = ${ff(nv, 4)}\n`
        + `mask (NDVI ≥ ${ff(thr, 2)}): ${mask} of 36 cells\n`
        + `vegetated area: ${mask} × 100 m² = ${ff(area, 3)} ha  (10 m pixels)`;
      if (paint < N * N) {
        const k = paint;
        cap.textContent = `Painting cell (row ${Math.floor(k / N)}, col ${k % N}): NDVI = ${ff(ndvi(k), 3)}. Each cell is one pixel; the mask is applied to the painted cells.`;
      } else {
        cap.textContent = 'All 36 cells painted. Click any cell in the three panels to select it, then drag its red or NIR to see the NDVI change.';
      }
      const key = [paint, sel, thr, red[sel], nir[sel]].join('|');
      lastKey = key;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4.3  Pixel classifier (k-nearest neighbours on two bands)
   * ===================================================================== */
  AI.labs.geoPixelClass = function (root) {
    const { view, ctl } = AI.labShell(root, 'Pixel classifier',
      'k-nearest neighbours on red and NIR · the scene is painted row by row · the feature plane shows the boundary',
      '<canvas style="height:230px"></canvas><canvas style="height:190px;margin-top:8px"></canvas><div class="caption"></div>', '');
    const cMap = AI.canvas(view.querySelectorAll('canvas')[0]), cFeat = AI.canvas(view.querySelectorAll('canvas')[1]);
    const cap = view.querySelector('.caption');
    const SZ = 40, NPIX = SZ * SZ;
    const CLS = [
      { n: 'water', c: 'blue', R: 0.05, N: 0.02 },
      { n: 'vegetation', c: 'green', R: 0.04, N: 0.45 },
      { n: 'bare soil', c: 'orange', R: 0.25, N: 0.30 },
    ];
    const st = transport(ctl, 1);
    let dirty = true, seed = 1, rowAcc = 0, rows = 0, lastKey = '', model = null;
    const sK = slider(ctl, 'k (neighbours that vote)', 1, 25, 1, 5, null, () => (dirty = true, recompute()));
    const sNoise = slider(ctl, 'Spectral noise (sd, reflectance)', 0, 0.08, 0.005, 0.03, null, () => (dirty = true, recompute()));
    const row = el('div', 'row');
    row.innerHTML = '<button class="btn small" data-a="new">New scene</button>';
    ctl.appendChild(row);
    row.querySelector('[data-a=new]').onclick = () => { seed++; scene = build(); recompute(); rows = 0; rowAcc = 0; dirty = true; };
    const ro = readout(ctl);

    /* scene: water pond bottom-left, bare-soil block top-right, vegetation elsewhere */
    function build(sd0) {
      const r = AI.rng(seed * 101 + 7), rt = AI.rng(seed * 313 + 5);
      const truth = new Int8Array(NPIX), R = new Float64Array(NPIX), Nn = new Float64Array(NPIX), zR = new Float64Array(NPIX), zN = new Float64Array(NPIX);
      for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
        const i = y * SZ + x;
        let t = 1;
        if ((x - 12) ** 2 + (y - 30) ** 2 < 40) t = 0;
        else if (x >= 24 && y <= 18) t = 2;
        truth[i] = t;
        zR[i] = AI.randn(r); zN[i] = AI.randn(r);
      }
      // training pixels: a random 8% sample (the lesson's trap: random pixels are neighbours of test pixels)
      const train = [];
      const isTrain = new Uint8Array(NPIX);
      while (train.length < NPIX * 0.08) { const i = Math.floor(rt() * NPIX); if (!isTrain[i]) { isTrain[i] = 1; train.push(i); } }
      return { truth, zR, zN, train, isTrain };
    }
    let scene = build();
    function recompute() {
      const sd = sNoise.get(), k = sK.get();
      const Rv = new Float64Array(NPIX), Nv = new Float64Array(NPIX);
      for (let i = 0; i < NPIX; i++) {
        const c = CLS[scene.truth[i]];
        Rv[i] = clamp(c.R + sd * scene.zR[i], 0, 0.6); Nv[i] = clamp(c.N + sd * scene.zN[i], 0, 0.8);
      }
      const knn = (x, y) => {
        const dd = scene.train.map((i) => ({ i, d: (Rv[i] - x) ** 2 + (Nv[i] - y) ** 2 }));
        dd.sort((a, b) => a.d - b.d);
        const votes = [0, 0, 0];
        for (let j = 0; j < Math.min(k, dd.length); j++) votes[scene.truth[dd[j].i]]++;
        let best = 0;
        for (let c = 1; c < 3; c++) if (votes[c] > votes[best]) best = c;
        return { cls: best, votes, first: dd.slice(0, k).map((e) => scene.truth[e.i]) };
      };
      const pred = new Int8Array(NPIX);
      for (let i = 0; i < NPIX; i++) pred[i] = knn(Rv[i], Nv[i]).cls;
      // accuracy on pixels not used for training
      let ok = 0, n = 0; const conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < NPIX; i++) if (!scene.isTrain[i]) { n++; conf[scene.truth[i]][pred[i]]++; if (pred[i] === scene.truth[i]) ok++; }
      // feature-plane boundary on a grid
      const G = 48, grid = new Int8Array(G * G);
      for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) grid[gy * G + gx] = knn(gx / G * 0.35, gy / G * 0.6).cls;
      model = { Rv, Nv, pred, knn, acc: ok / Math.max(1, n), n, conf, G, grid, k, sd };
      lastKey = '';
    }
    recompute();

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; rows = 0; rowAcc = 0; dirty = true; }
      if (st.stepReq) { rows = Math.min(SZ, rows + st.stepReq); st.stepReq = 0; dirty = true; }
      if (st.playing && rows < SZ) { rowAcc += dt * st.speed * 2.5; const n = Math.floor(rowAcc); if (n) { rowAcc -= n; rows = Math.min(SZ, rows + n); dirty = true; } }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), M = model;
      /* scene map */
      const g = cMap.ctx; g.clearRect(0, 0, cMap.w, cMap.h);
      const cs = Math.min(cMap.w * 0.6 / SZ, (cMap.h - 10) / SZ), ox = 10, oy = 6;
      for (let i = 0; i < NPIX; i++) {
        const x = i % SZ, y = Math.floor(i / SZ);
        if (y < rows) { g.fillStyle = C[CLS[M.pred[i]].c]; g.globalAlpha = 0.9; } else { g.fillStyle = C.bg2; g.globalAlpha = 1; }
        g.fillRect(ox + x * cs, oy + y * cs, cs + 0.5, cs + 0.5);
      }
      g.globalAlpha = 1;
      // test-pixel error marks on painted rows
      g.strokeStyle = C.fg; g.lineWidth = 1;
      if (rows > 0 && rows <= SZ) { g.strokeRect(ox - 0.5, oy - 0.5, SZ * cs + 1, Math.min(rows, SZ) * cs + 1); }
      // legend text on the right of the map
      const lx = ox + SZ * cs + 16;
      g.font = '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'left';
      CLS.forEach((c, i) => { g.fillStyle = C[c.c]; g.fillRect(lx, oy + 6 + i * 20, 12, 12); g.fillStyle = C.fg; g.fillText(c.n, lx + 18, oy + 16 + i * 20); });
      g.fillStyle = C.muted; g.fillText('painted: ' + rows + ' of ' + SZ + ' rows', lx, oy + 90);

      /* feature plane */
      const f = cFeat.ctx; f.clearRect(0, 0, cFeat.w, cFeat.h);
      const v = AI.view(cFeat, 0, 0.35, 0, 0.6, { l: 40, r: 12, t: 10, b: 26 });
      const gw = (v.pw) / M.G, gh = (v.ph) / M.G;
      f.globalAlpha = 0.28;
      for (let gy = 0; gy < M.G; gy++) for (let gx = 0; gx < M.G; gx++) {
        f.fillStyle = C[CLS[M.grid[gy * M.G + gx]].c];
        f.fillRect(v.pad.l + gx * gw, v.pad.t + v.ph - (gy + 1) * gh, gw + 0.5, gh + 0.5);
      }
      f.globalAlpha = 1;
      AI.axes(f, v);
      f.font = '11px ui-monospace,Menlo,Consolas,monospace'; f.fillStyle = C.muted; f.textAlign = 'center';
      f.fillText('red reflectance', v.pad.l + v.pw / 2, v.pad.t + v.ph + 22);
      f.save(); f.translate(12, v.pad.t + v.ph / 2); f.rotate(-Math.PI / 2); f.fillText('NIR', 0, 0); f.restore();
      for (let i = 0; i < NPIX; i++) {
        if (scene.isTrain[i]) continue;
        f.fillStyle = C[CLS[scene.truth[i]].c]; f.globalAlpha = 0.35;
        f.fillRect(v.X(M.Rv[i]) - 1, v.Y(M.Nv[i]) - 1, 2, 2);
      }
      f.globalAlpha = 1;
      scene.train.forEach((i) => {
        f.fillStyle = C[CLS[scene.truth[i]].c]; f.strokeStyle = C.fg; f.lineWidth = 1;
        f.beginPath(); f.arc(v.X(M.Rv[i]), v.Y(M.Nv[i]), 3, 0, 7); f.fill(); f.stroke();
      });

      /* readout and caption */
      const lines = [
        `k = ${M.k}, noise sd = ${ff(M.sd, 3)}`,
        `training pixels (random 8%): ${scene.train.length}`,
        `test pixels: ${M.n}`,
        `overall accuracy on test pixels: ${(M.acc * 100).toFixed(1)}%`,
        'confusion (rows = truth, cols = predicted):',
      ];
      CLS.forEach((c, i) => lines.push(`  ${c.n.padEnd(11, ' ')} ${M.conf[i].map((x) => String(x).padStart(5, ' ')).join(' ')}`));
      lines.push('(random pixel split: test pixels have near twins in training, so this is optimistic)');
      ro.textContent = lines.join('\n');
      if (rows < SZ) {
        const y = Math.min(SZ - 1, rows), i = y * SZ, x0 = 0;
        const nb = M.knn(M.Rv[i + x0], M.Nv[i + x0]);
        const cnt = [0, 0, 0]; nb.first.forEach((c) => cnt[c]++);
        cap.textContent = `Row ${y + 1}: the first pixel has red ${ff(M.Rv[i], 3)}, NIR ${ff(M.Nv[i], 3)}. Its ${M.k} nearest training points vote ${cnt.map((c, q) => c + ' ' + CLS[q].n).join(', ')}, so it is labelled ${CLS[nb.cls].n}.`;
      } else {
        cap.textContent = `Scene complete. Accuracy on the test pixels is ${(M.acc * 100).toFixed(1)}%. Increase k to smooth the boundary, or raise the noise to make the classes overlap.`;
      }
      lastKey = 'x';
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4.4  Object-based: SLIC-style superpixels
   * ===================================================================== */
  AI.labs.geoObjects = function (root) {
    const { view, ctl } = AI.labShell(root, 'Object-based segmentation (SLIC)',
      'superpixels grow on a 16 × 16 lightness image · click a superpixel to read its object features',
      '<canvas style="height:300px;cursor:pointer"></canvas><div class="caption"></div>', '');
    const cv = AI.canvas(view.querySelector('canvas')), cap = view.querySelector('.caption');
    const W = 16, S = 4, MAXIT = 10;
    const r = AI.rng(5);
    const L = new Float64Array(W * W);
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const blob = ((x - 10) ** 2) / 18 + ((y - 6) ** 2) / 10 < 1;
      const base = blob ? 72 : (x < 6 ? 30 : 46);
      L[y * W + x] = clamp(base + (r() - 0.5) * 16, 0, 100);
    }
    const st = transport(ctl, 1);
    let dirty = true, iter = 1, acc = 0, sel = -1, res = null, lastM = -1;
    const sM = slider(ctl, 'Compactness m (weight of position)', 1, 40, 1, 10, null, () => (dirty = true));
    const ro = readout(ctl);
    const seeds = [];
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) seeds.push([2 + 4 * i, 2 + 4 * j]);
    const disp = { x: seeds.map((s) => s[0]), y: seeds.map((s) => s[1]) };

    function run(iters, m) {
      const K = seeds.length;
      const cx = seeds.map((s) => s[0]), cy = seeds.map((s) => s[1]), cl = seeds.map(([x, y]) => L[y * W + x]);
      const lab = new Int32Array(W * W).fill(0);
      for (let it = 0; it < iters; it++) {
        const dist = new Float64Array(W * W).fill(Infinity);
        for (let k = 0; k < K; k++) {
          const x0 = Math.max(0, cx[k] - 2 * S), x1 = Math.min(W - 1, cx[k] + 2 * S);
          const y0 = Math.max(0, cy[k] - 2 * S), y1 = Math.min(W - 1, cy[k] + 2 * S);
          for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
            const idx = y * W + x, dc = L[idx] - cl[k], ds = Math.hypot(x - cx[k], y - cy[k]);
            const D = Math.sqrt(dc * dc + (ds / S) ** 2 * m * m);
            if (D < dist[idx]) { dist[idx] = D; lab[idx] = k; }
          }
        }
        const sx = new Float64Array(K), sy = new Float64Array(K), sl = new Float64Array(K), n = new Int32Array(K);
        for (let idx = 0; idx < W * W; idx++) { const k = lab[idx]; sx[k] += idx % W; sy[k] += Math.floor(idx / W); sl[k] += L[idx]; n[k]++; }
        for (let k = 0; k < K; k++) if (n[k]) { cx[k] = sx[k] / n[k]; cy[k] = sy[k] / n[k]; cl[k] = sl[k] / n[k]; }
      }
      return { lab, cx, cy };
    }

    cv.cv.addEventListener('click', (e) => {
      const G = geomO();
      const x = Math.floor((e.offsetX - G.ox) / G.cs), y = Math.floor((e.offsetY - G.oy) / G.cs);
      if (x >= 0 && x < W && y >= 0 && y < W && res) { sel = res.lab[y * W + x]; dirty = true; }
    });
    function geomO() {
      const cs = Math.min(cv.w, cv.h - 14) / W;
      return { cs, ox: (cv.w - W * cs) / 2, oy: 6 };
    }
    cv.onresize = () => (dirty = true);

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; iter = 1; acc = 0; dirty = true; }
      if (st.stepReq) { iter = Math.min(MAXIT, iter + st.stepReq); st.stepReq = 0; dirty = true; }
      if (st.playing && iter < MAXIT) {
        acc += dt * st.speed * 0.8; // one iteration about every 1.25 s at 1x
        while (acc >= 1 && iter < MAXIT) { acc -= 1; iter++; dirty = true; }
      }
      const m = sM.get();
      if (m !== lastM) { lastM = m; dirty = true; }
      if (!res || dirty) { res = run(iter, m); }
      // centres glide to their targets (dt-based, so speed does not depend on frame rate)
      let moving = false;
      for (let k = 0; k < seeds.length; k++) {
        const a = 1 - Math.exp(-dt * 5);
        const dx = res.cx[k] - disp.x[k], dy = res.cy[k] - disp.y[k];
        if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) { disp.x[k] += dx * a; disp.y[k] += dy * a; moving = true; }
      }
      if (!dirty && !moving) return;
      dirty = false;
      const C = AI.col(), g = cv.ctx, G = geomO();
      g.clearRect(0, 0, cv.w, cv.h);
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
        const idx = y * W + x, gv = Math.round(L[idx] / 100 * 255);
        g.fillStyle = `rgb(${gv},${gv},${gv})`;
        g.fillRect(G.ox + x * G.cs, G.oy + y * G.cs, G.cs + 0.5, G.cs + 0.5);
        g.globalAlpha = 0.3; g.fillStyle = idx >= 0 && res.lab[idx] === sel ? C.blue : hue(res.lab[idx]);
        g.fillRect(G.ox + x * G.cs, G.oy + y * G.cs, G.cs + 0.5, G.cs + 0.5);
        g.globalAlpha = 1;
      }
      g.strokeStyle = C.fg; g.lineWidth = 1.5;
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
        const idx = y * W + x, lb = res.lab[idx];
        const X = G.ox + x * G.cs, Y = G.oy + y * G.cs;
        if (x + 1 < W && res.lab[idx + 1] !== lb) { g.beginPath(); g.moveTo(X + G.cs, Y); g.lineTo(X + G.cs, Y + G.cs); g.stroke(); }
        if (y + 1 < W && res.lab[idx + W] !== lb) { g.beginPath(); g.moveTo(X, Y + G.cs); g.lineTo(X + G.cs, Y + G.cs); g.stroke(); }
      }
      g.fillStyle = C.fg;
      for (let k = 0; k < seeds.length; k++) { g.beginPath(); g.arc(G.ox + (disp.x[k] + 0.5) * G.cs, G.oy + (disp.y[k] + 0.5) * G.cs, 3, 0, 7); g.fill(); }

      // object features of the selected superpixel
      const pxs = [];
      for (let idx = 0; idx < W * W; idx++) if (res.lab[idx] === sel) pxs.push(idx);
      if (sel >= 0 && pxs.length) {
        const n = pxs.length, mean = pxs.reduce((a, i) => a + L[i], 0) / n;
        const sd = Math.sqrt(pxs.reduce((a, i) => a + (L[i] - mean) ** 2, 0) / n);
        const cxo = pxs.reduce((a, i) => a + (i % W), 0) / n, cyo = pxs.reduce((a, i) => a + Math.floor(i / W), 0) / n;
        const mn = Math.min(...pxs.map((i) => L[i])), mx = Math.max(...pxs.map((i) => L[i]));
        g.strokeStyle = C.blue; g.lineWidth = 3;
        pxs.forEach((i) => { const X = G.ox + (i % W) * G.cs, Y = G.oy + Math.floor(i / W) * G.cs; g.strokeRect(X + 1.5, Y + 1.5, G.cs - 3, G.cs - 3); });
        ro.textContent = `superpixel ${sel + 1} of 16 · iteration ${iter} of ${MAXIT}\n`
          + `object features\n`
          + `  area        ${n} pixels\n`
          + `  mean L      ${ff(mean, 2)}\n`
          + `  std L       ${ff(sd, 2)}  (population)\n`
          + `  centroid    (${ff(cxo, 2)}, ${ff(cyo, 2)})\n`
          + `  pixel range ${ff(mn, 1)} … ${ff(mx, 1)}\n`
          + `a single pixel varies by the whole range; the object mean is steadier.`;
      } else {
        ro.textContent = `iteration ${iter} of ${MAXIT} · m = ${m}\nclick a superpixel to read its object features`;
      }
      if (iter >= MAXIT) cap.textContent = 'Ten iterations: the boundaries have settled. Replay to watch the process again.';
      else if (iter === 1) cap.textContent = 'Iteration 1: each seed claims nearby pixels; the distance D adds lightness difference to spatial distance, weighted by m.';
      else cap.textContent = `Iteration ${iter}: every pixel rejoins its nearest centre under D; each centre then moves to the mean position and lightness of its pixels (dots glide).`;
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4.5  Zonal statistics
   * ===================================================================== */
  function clipPoly(poly, x0, y0, x1, y1) {
    let out = poly;
    const fns = [(p) => p[0] - x0, (p) => x1 - p[0], (p) => p[1] - y0, (p) => y1 - p[1]];
    for (const f of fns) {
      const inp = out; out = [];
      for (let i = 0; i < inp.length; i++) {
        const A = inp[i], B = inp[(i + 1) % inp.length];
        const fa = f(A), fb = f(B);
        if (fa >= 0) { out.push(A); if (fb < 0) { const t = fa / (fa - fb); out.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); } }
        else if (fb >= 0) { const t = fa / (fa - fb); out.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); }
      }
      if (!out.length) break;
    }
    return out;
  }
  const polyArea = (p) => { if (p.length < 3) return 0; let s = 0; for (let i = 0; i < p.length; i++) { const a = p[i], b = p[(i + 1) % p.length]; s += a[0] * b[1] - b[0] * a[1]; } return Math.abs(s) / 2; };

  AI.labs.geoZonal = function (root) {
    const { view, ctl } = AI.labShell(root, 'Zonal statistics over a polygon',
      'drag a corner of the polygon · each touched cell is weighted by the share of its area inside the polygon',
      '<canvas style="height:260px;touch-action:none;cursor:crosshair"></canvas><div class="caption"></div><canvas style="height:130px;margin-top:8px"></canvas>', '');
    const cMap = AI.canvas(view.querySelectorAll('canvas')[0]), cBar = AI.canvas(view.querySelectorAll('canvas')[1]);
    const cap = view.querySelector('.caption');
    const N = 8;
    // the top-left 3×3 block is the worked example of lesson 4.5
    const LESSON = [[0.71, 0.64, 0.22], [0.66, 0.58, 0.31], [0.12, 0.41, 0.09]];
    const LESSON_LC = [['crop', 'forest', 'crop'], ['crop', 'forest', 'crop'], ['soil', 'crop', 'soil']];
    const r = AI.rng(21);
    const ndvi = [], lc = [];
    for (let row = 0; row < N; row++) for (let col = 0; col < N; col++) {
      const v = clamp(0.5 + 0.28 * Math.sin(col / 2.1 + 0.4) * Math.cos(row / 2.6) + 0.06 * (r() - 0.5) * 2, 0.05, 0.92);
      ndvi.push(row < 3 && col < 3 ? LESSON[row][col] : +v.toFixed(2));
      lc.push(row < 3 && col < 3 ? LESSON_LC[row][col] : (v > 0.6 ? 'forest' : v > 0.4 ? 'crop' : 'soil'));
    }
    const PRESETS = {
      'Lesson farm (triangle)': [[0.5, 0.5], [2.5, 0.5], [1, 2]],
      'Wide parcel': [[1.2, 1.0], [6.8, 1.4], [6.2, 6.6], [1.0, 5.8]],
      'Small plot': [[4.2, 3.6], [5.9, 3.9], [5.2, 5.5]],
    };
    let poly = PRESETS['Lesson farm (triangle)'].map((p) => p.slice());
    let cov = null, scan = 0, pacc = 0, dirty = true, drag = -1, lastKey = '';
    const st = transport(ctl, 1);
    const row1 = el('div', 'row');
    row1.innerHTML = '<select style="width:auto">' + Object.keys(PRESETS).map((k) => `<option>${k}</option>`).join('') + '</select>'
      + '<label style="flex-direction:row;align-items:center;gap:6px"><input type="checkbox" style="width:auto"> all-touched rule (not coverage)</label>';
    ctl.appendChild(row1);
    const selP = row1.querySelector('select'), chkAT = row1.querySelector('input[type=checkbox]');
    const ro = readout(ctl);
    function recompute() {
      cov = new Float64Array(N * N);
      for (let row = 0; row < N; row++) for (let col = 0; col < N; col++) {
        cov[row * N + col] = polyArea(clipPoly(poly, col, row, col + 1, row + 1));
      }
      scan = 0; pacc = 0; dirty = true;
    }
    selP.onchange = () => { poly = PRESETS[selP.value].map((p) => p.slice()); recompute(); };
    chkAT.onchange = () => { dirty = true; };
    recompute();

    const geom = () => { const cs = Math.min(cMap.w, cMap.h - 6) / N; return { cs, ox: (cMap.w - N * cs) / 2, oy: 4 }; };
    const toCell = (e) => { const G = geom(); return [(e.offsetX - G.ox) / G.cs, (e.offsetY - G.oy) / G.cs]; };
    const cvEl = cMap.cv;
    cvEl.addEventListener('pointerdown', (e) => {
      const G = geom(), [x, y] = toCell(e);
      let best = -1, bd = 14 / G.cs;
      poly.forEach((p, k) => { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd) { bd = d; best = k; } });
      if (best >= 0) { drag = best; cvEl.setPointerCapture(e.pointerId); }
    });
    cvEl.addEventListener('pointermove', (e) => {
      if (drag < 0) return;
      const [x, y] = toCell(e);
      poly[drag] = [clamp(x, 0, N), clamp(y, 0, N)];
      recompute();
    });
    const endDrag = () => { drag = -1; };
    cvEl.addEventListener('pointerup', endDrag);
    cvEl.addEventListener('pointercancel', endDrag);

    function stats(upto) {
      const AT = chkAT.checked;
      let sw = 0, swv = 0, sn = 0, snv = 0, mn = Infinity, mx = -Infinity;
      const areaBy = {}, countBy = {};
      for (let k = 0; k < upto; k++) {
        if (cov[k] <= 1e-9) continue;
        const w = AT ? 1 : cov[k], v = ndvi[k];
        sw += w; swv += w * v; sn++; snv += v;
        mn = Math.min(mn, v); mx = Math.max(mx, v);
        areaBy[lc[k]] = (areaBy[lc[k]] || 0) + w; countBy[lc[k]] = (countBy[lc[k]] || 0) + 1;
      }
      const modeOf = (o) => Object.keys(o).sort((a, b) => o[b] - o[a])[0] || '–';
      return { sw, swv, sn, snv, mn, mx, wmean: sw ? swv / sw : NaN, umean: sn ? snv / sn : NaN, modeA: modeOf(areaBy), modeC: modeOf(countBy), areaBy, countBy };
    }

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; scan = 0; pacc = 0; dirty = true; }
      if (st.stepReq) { scan = Math.min(N * N, scan + st.stepReq); st.stepReq = 0; dirty = true; }
      if (st.playing && scan < N * N) { pacc += dt * st.speed * 4; const n = Math.floor(pacc); if (n) { pacc -= n; scan = Math.min(N * N, scan + n); dirty = true; } }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), g = cMap.ctx, G = geom(), AT = chkAT.checked;
      g.clearRect(0, 0, cMap.w, cMap.h);
      for (let k = 0; k < N * N; k++) {
        const row = Math.floor(k / N), col = k % N, X = G.ox + col * G.cs, Y = G.oy + row * G.cs;
        g.fillStyle = k < scan ? ndviColor(ndvi[k]) : C.bg2;
        g.fillRect(X, Y, G.cs + 0.5, G.cs + 0.5);
        if (k < scan && cov[k] > 1e-9) {
          // the part of the polygon inside this cell
          const piece = clipPoly(poly, col, row, col + 1, row + 1);
          g.beginPath();
          piece.forEach((p, i) => { const px = G.ox + p[0] * G.cs, py = G.oy + p[1] * G.cs; i ? g.lineTo(px, py) : g.moveTo(px, py); });
          g.closePath(); g.fillStyle = C.orange; g.globalAlpha = 0.45; g.fill(); g.globalAlpha = 1;
          g.font = '10px ui-monospace,Menlo,Consolas,monospace'; g.fillStyle = C.fg; g.textAlign = 'center';
          g.fillText(ff(AT ? 1 : cov[k], 2), X + G.cs / 2, Y + G.cs / 2 + 3);
        }
        g.strokeStyle = C.border; g.lineWidth = 0.5; g.strokeRect(X, Y, G.cs, G.cs);
      }
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath();
      poly.forEach((p, i) => { const px = G.ox + p[0] * G.cs, py = G.oy + p[1] * G.cs; i ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.closePath(); g.stroke();
      g.fillStyle = C.fg;
      poly.forEach((p) => { g.beginPath(); g.arc(G.ox + p[0] * G.cs, G.oy + p[1] * G.cs, 4, 0, 7); g.fill(); });

      /* statistics bars (over the cells scanned so far) */
      const S = stats(scan), b = cBar.ctx;
      b.clearRect(0, 0, cBar.w, cBar.h);
      const rows = [
        ['area-weighted mean', S.wmean, C.green],
        ['unweighted mean (cells touched)', S.umean, C.teal || C.blue],
        ['minimum', S.mn, C.orange],
        ['maximum', S.mx, C.green],
      ];
      b.font = '12px ui-monospace,Menlo,Consolas,monospace';
      rows.forEach(([name, val, col], i) => {
        const y = 14 + i * 28, x0 = 190, bw = cBar.w - x0 - 60;
        b.fillStyle = C.muted; b.textAlign = 'left'; b.fillText(name, 8, y + 10);
        b.fillStyle = C.bg3; b.fillRect(x0, y, bw, 12);
        if (Number.isFinite(val)) { b.fillStyle = col; b.fillRect(x0, y, bw * clamp(val, 0, 1), 12); b.fillStyle = C.fg; b.fillText(ff(val, 4), x0 + bw + 8, y + 10); }
        else { b.fillStyle = C.muted; b.fillText('–', x0 + 4, y + 10); }
      });

      const area = cov.reduce((a, c) => a + c, 0);
      ro.textContent = `polygon area = ${ff(polyArea(poly), 3)} cells · touched cells = ${[...cov].filter((c) => c > 1e-9).length}\n`
        + `scanned ${scan} of 64 cells · weight sum Σw = ${ff(S.sw, 4)}\n`
        + `Σ w·NDVI = ${ff(S.swv, 4)}  →  mean = ${ff(S.wmean, 4)}\n`
        + `mode (area-weighted): ${S.modeA}     mode (count): ${S.modeC}\n`
        + `area by class: ${Object.keys(S.areaBy).map((k) => k + ' ' + ff(S.areaBy[k], 3)).join(', ') || '–'}\n`
        + `count by class: ${Object.keys(S.countBy).map((k) => k + ' ' + S.countBy[k]).join(', ') || '–'}`;
      if (scan < N * N) {
        const k = Math.min(scan, N * N - 1), row = Math.floor(k / N), col = k % N;
        cap.textContent = cov[k] > 1e-9
          ? `Cell (row ${row}, col ${col}): coverage w = ${ff(AT ? 1 : cov[k], 4)}, NDVI ${ff(ndvi[k], 2)}. It adds w × NDVI = ${ff((AT ? 1 : cov[k]) * ndvi[k], 4)} to the weighted sum.`
          : `Cell (row ${row}, col ${col}) lies outside the polygon, so its weight is 0.`;
      } else {
        cap.textContent = 'Scan complete. Drag a corner to reshape the polygon; the scan restarts. Try the all-touched rule to see how it changes small polygons.';
      }
      lastKey = 'z';
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4.6  Change detection by differencing
   * ===================================================================== */
  AI.labs.geoChange = function (root) {
    const { view, ctl } = AI.labShell(root, 'Change detection by differencing',
      'two NDVI dates on a 24 × 16 grid (10 m pixels) · difference, threshold, noise and the changed-area counter',
      '<canvas style="height:210px"></canvas><div class="caption"></div><canvas style="height:120px;margin-top:8px"></canvas>', '');
    const cMap = AI.canvas(view.querySelectorAll('canvas')[0]), cHist = AI.canvas(view.querySelectorAll('canvas')[1]);
    const cap = view.querySelector('.caption');
    const W = 24, H = 16, NP = W * H;
    const r = AI.rng(31);
    const base = new Float64Array(NP), drop = new Float64Array(NP), truth = new Uint8Array(NP), z1 = new Float64Array(NP), z2 = new Float64Array(NP);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      base[i] = clamp(0.55 + 0.22 * Math.sin(x / 3.4) * Math.cos(y / 2.9), 0.05, 0.9);
      const clear = (x - 16) ** 2 + (y - 6) ** 2 < 14;
      const minor = x >= 3 && x <= 7 && y >= 10 && y <= 13;
      drop[i] = clear ? 0.45 : minor ? 0.12 : 0;
      truth[i] = clear ? 1 : 0;
      z1[i] = AI.randn(r); z2[i] = AI.randn(r);
    }
    const st = transport(ctl, 1);
    let dirty = true, rowAcc = 0, rowsDone = 0;
    const sSig = slider(ctl, 'Noise per date (σ, NDVI units)', 0, 0.12, 0.005, 0.04, null, () => (dirty = true));
    const sTau = slider(ctl, 'Threshold τ on |D|', 0.02, 0.5, 0.01, 0.2, null, () => (dirty = true));
    const ro = readout(ctl);
    const T1 = new Float64Array(NP), T2 = new Float64Array(NP), D = new Float64Array(NP);

    AI.loop(root, (dt) => {
      if (st.restartReq) { st.restartReq = false; rowsDone = 0; rowAcc = 0; dirty = true; }
      if (st.stepReq) { rowsDone = Math.min(H, rowsDone + st.stepReq); st.stepReq = 0; dirty = true; }
      if (st.playing && rowsDone < H) { rowAcc += dt * st.speed * 1.5; const n = Math.floor(rowAcc); if (n) { rowAcc -= n; rowsDone = Math.min(H, rowsDone + n); dirty = true; } }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), sg = sSig.get(), tau = sTau.get(), sigD = Math.SQRT2 * sg;
      let flagged = 0, tp = 0, fp = 0, fn = 0;
      const flag = new Uint8Array(NP);
      for (let i = 0; i < NP; i++) {
        T1[i] = clamp(base[i] + sg * z1[i], -1, 1);
        T2[i] = clamp(base[i] - drop[i] + sg * z2[i], -1, 1);
        D[i] = T2[i] - T1[i];
        flag[i] = Math.abs(D[i]) > tau ? 1 : 0;
        if (flag[i]) flagged++;
        if (flag[i] && truth[i]) tp++; else if (flag[i]) fp++; else if (truth[i]) fn++;
      }
      /* difference map */
      const g = cMap.ctx; g.clearRect(0, 0, cMap.w, cMap.h);
      const cs = Math.min(cMap.w / W, cMap.h / H), ox = (cMap.w - W * cs) / 2, oy = 4;
      for (let i = 0; i < NP; i++) {
        const x = i % W, y = Math.floor(i / W), X = ox + x * cs, Y = oy + y * cs;
        if (y >= rowsDone) { g.fillStyle = C.bg2; }
        else {
          const t = clamp(Math.abs(D[i]) / 0.5, 0, 1);
          g.fillStyle = D[i] < 0 ? AI.mix(C.bg3, '#d1495b', t) : AI.mix(C.bg3, '#2b7bd6', t);
        }
        g.fillRect(X, Y, cs + 0.5, cs + 0.5);
        if (y < rowsDone && flag[i]) { g.strokeStyle = C.fg; g.lineWidth = 1.5; g.strokeRect(X + 2, Y + 2, cs - 4, cs - 4); }
      }
      g.setLineDash([4, 3]); g.strokeStyle = C.yellow; g.lineWidth = 1.5;
      for (let i = 0; i < NP; i++) if (truth[i]) { const x = i % W, y = Math.floor(i / W); g.strokeRect(ox + x * cs + 0.5, oy + y * cs + 0.5, cs - 1, cs - 1); }
      g.setLineDash([]);

      /* histogram of D over the painted rows */
      const h = cHist.ctx; h.clearRect(0, 0, cHist.w, cHist.h);
      const v = AI.view(cHist, -0.6, 0.6, 0, 1, { l: 36, r: 10, t: 8, b: 22 });
      const bins = new Array(40).fill(0);
      let painted = 0;
      for (let i = 0; i < NP; i++) {
        if (Math.floor(i / W) >= rowsDone) continue;
        painted++;
        const bi = clamp(Math.floor((D[i] + 0.6) / 0.03), 0, 39); bins[bi]++;
      }
      const mx = Math.max(1, ...bins);
      const bw = v.pw / 40;
      bins.forEach((c, bi) => {
        const bh = (c / mx) * v.ph * 0.95;
        h.fillStyle = (-0.6 + bi * 0.03) < -tau || (-0.6 + bi * 0.03) > tau ? C.orange : C.muted;
        h.globalAlpha = 0.85; h.fillRect(v.pad.l + bi * bw, v.pad.t + v.ph - bh, bw - 1, bh); h.globalAlpha = 1;
      });
      h.setLineDash([4, 3]); h.strokeStyle = C.yellow; h.lineWidth = 1.5;
      [-tau, tau].forEach((t) => { h.beginPath(); h.moveTo(v.X(t), v.pad.t); h.lineTo(v.X(t), v.pad.t + v.ph); h.stroke(); });
      h.setLineDash([]);
      h.font = '11px ui-monospace,Menlo,Consolas,monospace'; h.fillStyle = C.muted; h.textAlign = 'center';
      [-0.6, -0.3, 0, 0.3, 0.6].forEach((t) => h.fillText(String(t), v.X(t), v.pad.t + v.ph + 14));
      h.textAlign = 'left'; h.fillText('histogram of D = T2 − T1 (painted rows only)', v.pad.l + 4, v.pad.t + 12);

      const haFlag = flagged * 100 / 10000;
      ro.textContent = `σ_D = √2 · σ = ${ff(sigD, 4)}     τ / σ_D = ${ff(tau / sigD, 2)}\n`
        + `changed area: ${flagged} pixels × 100 m² = ${ff(haFlag, 3)} ha\n`
        + `true clear-cut pixels (dashed): ${truth.reduce((a, c) => a + c, 0)}\n`
        + `caught: ${tp}   missed: ${fn}   false alarms: ${fp}`;
      // caption for the row being painted
      const row = Math.min(H - 1, Math.floor(rowsDone));
      if (rowsDone < H) {
        const rowFlags = [];
        for (let x = 0; x < W; x++) { const i = row * W + x; if (Math.abs(D[i]) > tau) rowFlags.push(x); }
        const ex = rowFlags.length ? rowFlags[0] : -1;
        const exText = ex >= 0
          ? ` First flagged pixel (col ${ex}): T1 = ${ff(T1[row * W + ex], 2)}, T2 = ${ff(T2[row * W + ex], 2)}, D = ${ff(D[row * W + ex], 2)}.`
          : '';
        cap.textContent = `Row ${row + 1} of 16: ${rowFlags.length} of 24 pixels have |D| > ${ff(tau, 2)}.` + exText;
      } else {
        cap.textContent = 'All rows compared. Change the noise: the unchanged ground spreads, so the threshold must rise to keep false alarms down.';
      }
    }, { fps: 30 });
  };
})();
