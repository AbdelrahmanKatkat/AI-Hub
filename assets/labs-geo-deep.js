/* GeoAI Module 5 (deep learning for Earth observation) labs. Requires assets/lib.js.
 *
 *   geoadapter   <div data-lab="geoadapter"></div>     first conv layer: 3 bands vs 12 bands, output preserved?
 *   unetwalk     <div data-lab="unetwalk"></div>       U-Net walk: tile shrinks, decoder paints the mask, skips
 *   dice2x2      <div data-lab="dice2x2"></div>        soft Dice, cross-entropy and hard Dice on a 2x2 mask
 *   tilesize     <div data-lab="tilesize"></div>       tile size vs object size vs resolution, detection on a grid
 *   polygonize   <div data-lab="polygonize"></div>     mask -> pixel contour -> Douglas-Peucker polygon
 *   changesiam   <div data-lab="changesiam"></div>     siamese change map on two dates, changed-area counter
 *   sarview      <div data-lab="sarview"></div>        SAR linear vs dB, optical view, cloud overlay, editable dB formula
 *   slidewin     <div data-lab="slidewin"></div>       sliding-window inference: hard overwrite vs blended overlap
 */
(function () {
  /* ---------- shared helpers ---------- */
  const fmtv = (v) => String(+(+v).toFixed(3));
  const sl = (id, label, min, max, step, value) => `<label><span>${label} <span class="v" data-for="${id}"></span></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
  const sel = (id, label, opts, value) => `<label><span>${label}</span><select id="${id}">${opts.map(([v, t]) => `<option value="${v}"${String(v) === String(value) ? ' selected' : ''}>${t}</option>`).join('')}</select></label>`;
  const showVals = (root) => AI.$$('input[type=range]', root).forEach((i) => { const s = AI.$(`[data-for="${i.id}"]`, root); if (s) s.textContent = fmtv(i.value); });
  const wire = (root, fn) => { showVals(root); const h = () => { showVals(root); fn(); }; root.addEventListener('input', h); root.addEventListener('change', h); return h; };
  const num = (n) => Math.round(n).toLocaleString('en-US');
  const clamp = AI.clamp, lerp = AI.lerp;
  const pal = () => {
    const C = AI.col(), g = (k, d) => (C[k] && C[k][0] === '#' ? AI.rgb(C[k]) : d);
    return { bg: g('bg', [13, 17, 23]), bg2: g('bg2', [22, 27, 34]), bg3: g('bg3', [33, 38, 45]), fg: g('fg', [230, 237, 243]), muted: g('muted', [139, 148, 158]),
      blue: g('blue', [47, 129, 247]), green: g('green', [63, 185, 80]), orange: g('orange', [219, 109, 40]), purple: g('purple', [163, 113, 247]),
      yellow: g('yellow', [210, 153, 34]), red: g('red', [248, 81, 73]), teal: g('teal', [57, 197, 207]), pink: g('pink', [219, 97, 162]), border: g('border', [48, 54, 61]) };
  };
  const mixc = (a, b, t) => { t = clamp(t, 0, 1); return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)); };
  const css = (c, a = 1) => (a === 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`);
  const ramp = (t, stops) => { t = clamp(t, 0, 1) * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(t)); return mixc(stops[i], stops[i + 1], t - i); };
  const HEAT = [[16, 32, 58], [120, 60, 90], [219, 109, 40], [250, 230, 120]];
  const GREY = (v) => { const g = Math.round(clamp(v, 0, 1) * 255); return [g, g, g]; };
  const CLASS_COL = [[13, 17, 23], [47, 129, 247], [63, 185, 80], [219, 109, 40], [163, 113, 247], [57, 197, 207], [210, 153, 34], [219, 97, 162]];

  /* =====================================================================
   * 1. Multi-band input adapter
   * ===================================================================== */
  AI.labs.geoadapter = function (root) {
    AI.labShell(root, 'Adapting the first layer to more bands', 'the same 64 filters now read more channels: the weights grow, and on grey input the output can stay the same',
      '<canvas style="height:380px"></canvas><div class="caption" id="cap"></div>',
      [sl('bands', 'Input bands', 3, 12, 1, 12), sel('k', 'Kernel size', [[3, '3 × 3'], [5, '5 × 5'], [7, '7 × 7']], 7),
        sel('init', 'Initialise the new bands', [['replicate', 'mean of RGB × 3/N (keeps output)'], ['copy', 'copy RGB into bands 1–3, zero the rest'], ['random', 'random (He initialisation)']], 'replicate'),
        sl('grey', 'Grey input x (same value in every band)', 0, 1, 0.05, 0.5),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button><button class="btn small" id="grow">Grow 3 → N ▶</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), cv = q('canvas'), c = AI.canvas(cv);
    const R = AI.rng(7), G = () => AI.randn(R);
    // pretrained RGB kernels (7x7) for 8 displayed filters, plus random draws for new bands
    const PRE = [], NEW = [];
    for (let f = 0; f < 8; f++) {
      PRE.push([0, 1, 2].map(() => Float64Array.from({ length: 49 }, () => G() * 0.25)));
      NEW.push(Array.from({ length: 12 }, () => Float64Array.from({ length: 49 }, () => G())));
    }
    let bands = 12, k = 7, init = 'replicate', grey = 0.5, shown = 3, paused = false, dirty = true, lastKey = '';
    const at = (a, i, j) => a[(i + (7 - k) / 2) * 7 + (j + (7 - k) / 2)];
    const wgt = (f, b, i, j) => {
      if (init === 'replicate') return (3 / bands) * (at(PRE[f][0], i, j) + at(PRE[f][1], i, j) + at(PRE[f][2], i, j)) / 3;
      if (init === 'copy') return b < 3 ? at(PRE[f][b], i, j) : 0;
      return at(NEW[f][b], i, j) * Math.sqrt(2 / (bands * k * k));
    };
    const sums = () => {
      let pre = 0, now = 0;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
        for (let ch = 0; ch < 3; ch++) pre += at(PRE[0][ch], i, j);
        for (let b = 0; b < bands; b++) now += wgt(0, b, i, j);
      }
      return { pre: pre * grey, now: now * grey };
    };
    const cell = (g, x, y, s, val, m) => { const v = val / m; g.fillStyle = css(v >= 0 ? mixc(PAL.bg2, PAL.orange, v) : mixc(PAL.bg2, PAL.blue, -v)); g.fillRect(x, y, s / k - 0.6, s / k - 0.6); };
    let PAL = pal();
    const draw = () => {
      PAL = pal();
      const g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const x1 = W * 0.36, rowH = (H - 46) / 8, slot = (W - x1 - 10) / 12, s = Math.min(rowH - 6, slot - 3, 34);
      g.font = '12px sans-serif'; g.fillStyle = css(PAL.muted); g.fillText('pretrained: 3 channels (R, G, B)', 8, 16);
      g.fillText(`adapted: ${bands} channels`, x1, 16);
      for (let f = 0; f < 8; f++) {
        const y = 30 + f * rowH + (rowH - s) / 2;
        for (let ch = 0; ch < 3; ch++) {
          const x = 8 + ch * (s + 4);
          g.save(); g.beginPath(); g.rect(x, y, s, s); g.clip();
          for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) cell(g, x + (j * s) / k, y + (i * s) / k, s, at(PRE[f][ch], i, j), 0.6);
          g.restore();
        }
      }
      for (let f = 0; f < 8; f++) {
        const y = 30 + f * rowH + (rowH - s) / 2;
        for (let b = 0; b < 12; b++) {
          const a = clamp(shown - b, 0, 1); if (a <= 0) continue;
          const x = x1 + b * slot; g.save(); g.globalAlpha = a; g.beginPath(); g.rect(x, y, s, s); g.clip();
          for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) cell(g, x + (j * s) / k, y + (i * s) / k, s, wgt(f, b, i, j), 0.6);
          g.restore(); g.globalAlpha = 1;
          if (b >= 3) { g.strokeStyle = css(PAL.purple, 0.6); g.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1); }
        }
      }
    };
    const update = () => {
      const S = sums(), same = Math.abs(S.pre - S.now) < 1e-9;
      const cap = { replicate: 'Mean × 3/N: each new band carries 1/N of the RGB mean, so on grey input the sum over bands equals the RGB sum. The output does not change.',
        copy: 'Copy RGB into bands 1–3 and zero the rest: the RGB bands reproduce the old response on grey input; the new bands start silent and learn from zero.',
        random: 'Random new kernels: the response changes, so the pretrained features are partly lost and must be re-learned.' };
      q('#cap').textContent = cap[init];
      q('#out').innerHTML = `3-band layer:   64 × 3 × ${k} × ${k} = <b class="b">${num(64 * 3 * k * k)}</b> weights\n${bands}-band layer:   64 × ${bands} × ${k} × ${k} = <b class="o">${num(64 * bands * k * k)}</b> weights (+64 biases)\nper band:          64 × ${k} × ${k} = <b>${num(64 * k * k)}</b>\nparameters now:    <b>${num(64 * bands * k * k + 64)}</b>\n\ngrey input x = ${fmtv(grey)} in every band, filter 1:\npretrained output  <b>${AI.fmt(S.pre, 4)}</b>\nadapted output     <b>${AI.fmt(S.now, 4)}</b>  ${same ? '<span class="g">identical</span>' : '<span class="r">differs</span>'}`;
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    q('#grow').onclick = () => { shown = 3; paused = false; q('#pp').textContent = '⏸ Pause'; };
    wire(root, () => {
      bands = +q('#bands').value; k = +q('#k').value; init = q('#init').value; grey = +q('#grey').value; dirty = true;
    });
    c.onresize = () => (dirty = true);
    AI.loop(root, (dt) => {
      if (!paused && Math.abs(shown - bands) > 0.01) { shown += (bands - shown) * Math.min(1, dt * 2.5); dirty = true; }
      if (!paused && Math.abs(shown - bands) <= 0.01 && shown !== bands) { shown = bands; dirty = true; }
      const key = [bands, k, init, grey, Math.round(shown * 100), c.w, c.h].join();
      if (key !== lastKey || dirty) { lastKey = key; dirty = false; draw(); update(); }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 2. U-Net walk
   * ===================================================================== */
  // Pure shape/parameter model of a U-Net (3x3 convs with bias, 2x2 transposed up-convs, 1x1 head).
  AI.geoDeep = AI.geoDeep || {};
  AI.geoDeep.unetSpec = function (bands, tile, depth, base, K) {
    const conv = (a, b) => a * b * 9 + b, up = (a, b) => 4 * a * b + b;
    const enc = [], dec = [];
    let total = 0;
    for (let l = 0; l <= depth; l++) {
      const c = base * 2 ** l, res = tile >> l;
      const p = l === 0 ? conv(bands, c) + conv(c, c) : conv(base * 2 ** (l - 1), c) + conv(c, c);
      enc.push({ l, c, res, p }); total += p;
    }
    for (let l = depth - 1; l >= 0; l--) {
      const c = base * 2 ** l, res = tile >> l, pu = up(base * 2 ** (l + 1), c), pc = conv(2 * c, c) + conv(c, c);
      dec.push({ l, c, res, pu, p: pu + pc }); total += pu + pc;
    }
    const head = base * K + K; total += head;
    const peak = Math.max(...enc.map((e) => e.c * e.res * e.res), ...dec.map((d) => d.c * d.res * d.res)) * 4;
    return { enc, dec, head, total, peak, bottleneck: enc[depth].p };
  };
  const maskLabel = (u, v) => {
    if ((u - 0.38) ** 2 + (v - 0.55) ** 2 < 0.05) return 1;
    if (u > 0.62 && v < 0.5) return 2;
    if (v > 0.8) return 3;
    return 0;
  };

  AI.labs.unetwalk = function (root) {
    AI.labShell(root, 'Walk through a U-Net', 'the tile shrinks on the way down; the decoder paints the mask back up, and skips carry detail across',
      '<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>',
      [sl('bands', 'Input bands', 3, 12, 1, 12), sel('tile', 'Tile size (px)', [[128, '128'], [256, '256'], [512, '512']], 256),
        sl('depth', 'Depth (down-sampling levels)', 2, 4, 1, 4), sel('base', 'Base width (channels)', [[32, '32'], [64, '64']], 64),
        sl('K', 'Classes K', 2, 8, 1, 5), sl('sp', 'Speed', 0.3, 2, 0.1, 1),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button><button class="btn small" id="st">Step ▶</button><button class="btn small" id="rs">↺ Restart</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), cv = q('canvas'), c = AI.canvas(cv);
    let S = null, bands = 12, tile = 256, depth = 4, base = 64, K = 5, t = 0, paused = false, lastStep = -1, PAL = pal();
    const DUR = 1.3;
    const rebuild = () => { S = AI.geoDeep.unetSpec(bands, tile, depth, base, K); const L = []; S.enc.forEach((e) => L.push({ type: 'enc', ...e })); S.dec.forEach((d) => L.push({ type: 'dec', ...d })); L.push({ type: 'head' }); S.steps = L; };
    const caption = (st) => {
      if (st.type === 'enc') return st.l === 0 ? `Encoder level 0: two 3×3 convolutions keep the ${st.res}×${st.res} size and turn ${bands} bands into ${st.c} channels.`
        : st.l === depth ? `Bottleneck: ${st.c} channels on a ${st.res}×${st.res} map. It knows what is in the tile best, and where it is least well.`
          : `Encoder level ${st.l}: max-pool halves the map to ${st.res}×${st.res} and two convolutions widen it to ${st.c} channels.`;
      if (st.type === 'dec') return `Decoder level ${st.l}: upsample to ${st.res}×${st.res}, concatenate the skip from encoder level ${st.l} (${2 * st.c} channels), two convolutions back to ${st.c}. The mask paints in.`;
      return `Head: a 1×1 convolution maps ${base} channels to ${K} class scores per pixel; softmax, then arg-max, gives the label map.`;
    };
    const readout = () => {
      q('#out').innerHTML = `parameters     <b class="o">${num(S.total)}</b>  <span class="m">(≈ ${AI.fmt(S.total * 4 / 2 ** 20, 1)} MiB as float32)</span>\nbottleneck     ${num(S.bottleneck)}  <span class="m">(${AI.fmt(100 * S.bottleneck / S.total, 3)}% of all)</span>\nlargest tensor ${S.enc[0].c}×${S.enc[0].res}×${S.enc[0].res} = <b class="b">${AI.fmt(S.peak / 2 ** 20, 1)} MiB</b> per image\nskips          ${depth} (encoder level l → decoder level l)`;
    };
    const drawAll = (step, f) => {
      PAL = pal(); const g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const n = S.steps.length, X0 = 46, X1 = W - 40, cy = H * 0.46, span = (X1 - X0) / (n - 1), Hmax = Math.min(H * 0.6, span * 0.85);
      const xs = S.steps.map((_, i) => X0 + i * span);
      const side = (res) => Math.max(8, Hmax * res / tile);
      const nn = (res) => Math.min(res, 12);
      // skip arcs (behind blocks)
      S.dec.forEach((d) => {
        const si = S.enc.length + (depth - 1 - d.l), a = S.enc[d.l], ax = xs[d.l], bx = xs[si], active = step === si;
        const top = cy - Hmax / 2 - 10 - (depth - d.l) * 6;
        g.strokeStyle = css(active ? PAL.green : PAL.border, active ? 1 : 0.8); g.lineWidth = active ? 2.5 : 1.2; g.setLineDash(active ? [] : [4, 4]);
        g.beginPath(); g.moveTo(ax, cy - side(a.res) / 2 - 4); g.bezierCurveTo(ax, top, bx, top, bx, cy - side(d.res) / 2 - 4); g.stroke(); g.setLineDash([]);
        if (active) { const tt = f, px = (1 - tt) * (1 - tt) * ax + 2 * (1 - tt) * tt * ax + tt * tt * bx, py = (1 - tt) * (1 - tt) * (cy - side(a.res) / 2 - 4) + 2 * (1 - tt) * tt * top + tt * tt * (cy - side(d.res) / 2 - 4); g.fillStyle = css(PAL.green); g.beginPath(); g.arc(px, py, 4, 0, 7); g.fill(); }
      });
      // blocks
      S.steps.forEach((st, i) => {
        const x = xs[i], done = i < step, now = i === step, dim = i > step;
        const res = st.res || 1, sd = st.type === 'head' ? Hmax * 0.4 : side(res), w = Math.max(10, Math.min(sd, span * 0.7));
        const y = cy - sd / 2, thick = st.c ? 2 + 2 * Math.log2(st.c / base) : 6;
        g.globalAlpha = dim ? 0.35 : 1;
        // depth slab (channels)
        g.fillStyle = css(PAL.bg3); g.fillRect(x - w / 2 + thick, y - thick, w, sd);
        if (st.type === 'head') {
          for (let kk = 0; kk < K; kk++) { g.fillStyle = css(CLASS_COL[kk % CLASS_COL.length].map((v, ii) => v)); g.fillRect(x - w / 2, y + (kk * sd) / K, w, sd / K - 1); }
        } else {
          const m = nn(res), cw = w / m;
          for (let r = 0; r < m; r++) for (let cc = 0; cc < m; cc++) {
            const u = (cc + 0.5) / m, v = (r + 0.5) / m;
            let col;
            if (st.type === 'enc') col = mixc(PAL.bg2, PAL.blue, 0.25 + 0.35 * (0.5 + 0.5 * Math.sin(u * 9) * Math.cos(v * 7)));
            else if (done || (now && r / m < f)) col = CLASS_COL[maskLabel(u, v) % CLASS_COL.length];   // mask painted
            else col = mixc(PAL.bg2, PAL.blue, 0.2);                                                    // not yet painted
            g.fillStyle = css(col); g.fillRect(x - w / 2 + cc * cw, y + r * (sd / m), cw + 0.3, sd / m + 0.3);
          }
        }
        g.globalAlpha = 1;
        g.strokeStyle = css(now ? PAL.green : PAL.border, now ? 1 : 0.7); g.lineWidth = now ? 2.5 : 1;
        g.strokeRect(x - w / 2, y, w, sd);
        // labels
        g.fillStyle = css(now ? PAL.fg : PAL.muted); g.font = `${now ? '600 ' : ''}10.5px ui-monospace,Menlo,Consolas,monospace`; g.textAlign = 'center';
        const lab = st.type === 'head' ? `K=${K}` : `${st.c}`, sub = st.type === 'head' ? `${base}→${K}` : `${res}²`;
        g.fillText(st.type === 'enc' ? `E${st.l}` : st.type === 'dec' ? `D${st.l}` : 'head', x, cy + Hmax / 2 + 16);
        g.fillText(lab + ' ch', x, cy + Hmax / 2 + 30); g.fillText(sub, x, cy + Hmax / 2 + 44);
        if (st.p || st.pu) { g.fillStyle = css(PAL.orange); g.fillText(num((st.p || 0) + (st.pu || 0)), x, cy + Hmax / 2 + 58); }
        if (st.type === 'head') { g.fillStyle = css(PAL.orange); g.fillText(num(S.head), x, cy + Hmax / 2 + 58); }
      });
    };
    const update = () => {
      rebuild(); readout();
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    q('#st').onclick = () => { paused = true; q('#pp').textContent = '▶ Play'; t = (Math.floor(t / DUR) + 1) * DUR; };
    q('#rs').onclick = () => { t = 0; paused = false; q('#pp').textContent = '⏸ Pause'; };
    wire(root, () => { bands = +q('#bands').value; tile = +q('#tile').value; depth = +q('#depth').value; base = +q('#base').value; K = +q('#K').value; update(); lastStep = -1; });
    update();
    c.onresize = () => (lastStep = -2);
    AI.loop(root, (dt) => {
      const n = S.steps.length, total = n * DUR;
      if (!paused) { t += dt * (+q('#sp').value); if (t >= total + 1.6) t = 0; }
      const step = Math.min(n - 1, Math.floor(t / DUR)), f = clamp((t - step * DUR) / DUR, 0, 1), eased = AI.smooth(f);
      if (step !== lastStep) { lastStep = step; q('#cap').textContent = `Step ${step + 1} of ${n}: ${caption(S.steps[step])}`; }
      drawAll(step, eased);
    }, { fps: 30 });
  };

  /* =====================================================================
   * 3. Dice and cross-entropy on a 2 x 2 mask
   * ===================================================================== */
  AI.labs.dice2x2 = function (root) {
    AI.labShell(root, 'Dice and cross-entropy on a 2 × 2 mask', 'one class ("built-up"): set the predicted probabilities and the truth; the losses update',
      '<canvas style="height:250px"></canvas><div class="caption" id="cap">Bars: cross-entropy of each pixel (taller = worse). Green outline = true built-up pixel.</div>',
      [sl('p1', 'p, pixel 1 (top-left)', 0, 1, 0.05, 0.9), sl('p2', 'p, pixel 2 (top-right)', 0, 1, 0.05, 0.2), sl('p3', 'p, pixel 3 (bottom-left)', 0, 1, 0.05, 0.6), sl('p4', 'p, pixel 4 (bottom-right)', 0, 1, 0.05, 0.1),
        sel('y1', 'truth, pixel 1', [[1, 'built-up'], [0, 'other']], 1), sel('y2', 'truth, pixel 2', [[1, 'built-up'], [0, 'other']], 0),
        sel('y3', 'truth, pixel 3', [[1, 'built-up'], [0, 'other']], 1), sel('y4', 'truth, pixel 4', [[1, 'built-up'], [0, 'other']], 0),
        sl('thr', 'Threshold for hard mask', 0.05, 0.95, 0.05, 0.5), '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', root));
    const bar = [0, 0, 0, 0];
    let lastDraw = 0, PAL = pal();
    const read = () => {
      const p = [1, 2, 3, 4].map((i) => +q('#p' + i).value), y = [1, 2, 3, 4].map((i) => +q('#y' + i).value), thr = +q('#thr').value;
      const sp = p.reduce((a, b) => a + b, 0), sy = y.reduce((a, b) => a + b, 0), inter = p.reduce((a, v, i) => a + v * y[i], 0);
      const ce = p.map((v, i) => { const pc = clamp(v, 1e-6, 1 - 1e-6); return y[i] ? -Math.log(pc) : -Math.log(1 - pc); });
      const ceMean = ce.reduce((a, b) => a + b, 0) / 4;
      const dice = sp + sy > 0 ? (2 * inter) / (sp + sy) : NaN;
      const ph = p.map((v) => (v >= thr ? 1 : 0));
      const TP = ph.reduce((a, v, i) => a + v * y[i], 0), FP = ph.reduce((a, v, i) => a + v * (1 - y[i]), 0), FN = y.reduce((a, v, i) => a + v * (1 - ph[i]), 0);
      const hard = 2 * TP + FP + FN > 0 ? (2 * TP) / (2 * TP + FP + FN) : NaN;
      q('#out').innerHTML = `soft overlap  Σp·y = <b>${AI.fmt(inter, 3)}</b>,  Σp = <b>${AI.fmt(sp, 3)}</b>,  Σy = <b>${sy}</b>\nsoft Dice    = 2·Σpy / (Σp + Σy) = <b class="g">${AI.fmt(dice, 4)}</b>   loss = <b class="o">${AI.fmt(1 - dice, 4)}</b>\ncross-entropy per pixel: ${ce.map((v) => AI.fmt(v, 3)).join(', ')}\nmean cross-entropy = <b class="o">${AI.fmt(ceMean, 4)}</b>\nhard mask at ${AI.fmt(thr, 2)}: TP ${TP}, FP ${FP}, FN ${FN} → Dice <b class="b">${AI.fmt(hard, 4)}</b>`;
      return { ce };
    };
    const draw = (ce, dt) => {
      PAL = pal(); const g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const p = [1, 2, 3, 4].map((i) => +q('#p' + i).value), y = [1, 2, 3, 4].map((i) => +q('#y' + i).value);
      const gs = Math.min(H * 0.55, W * 0.3), gx = W * 0.15, gy = 14;
      for (let i = 0; i < 4; i++) {
        const r = Math.floor(i / 2), cc = i % 2, x = gx + cc * gs, yy = gy + r * gs;
        g.fillStyle = css(mixc(PAL.bg2, PAL.blue, p[i])); g.fillRect(x + 3, yy + 3, gs - 6, gs - 6);
        g.fillStyle = css(PAL.fg); g.font = '600 13px sans-serif'; g.textAlign = 'center';
        g.fillText(p[i].toFixed(2), x + gs / 2, yy + gs / 2 + 5);
        if (y[i]) { g.strokeStyle = css(PAL.green); g.lineWidth = 3; g.strokeRect(x + 5, yy + 5, gs - 10, gs - 10); }
      }
      // CE bars (eased toward target)
      const bx = W * 0.56, bw = (W - bx - 20) / 4, base = H - 26;
      g.font = '11px sans-serif'; g.fillStyle = css(PAL.muted); g.textAlign = 'center'; g.fillText('cross-entropy per pixel', (bx + W - 10) / 2, 14);
      for (let i = 0; i < 4; i++) {
        bar[i] += (ce[i] - bar[i]) * Math.min(1, dt * 4);
        const hgt = Math.min(bar[i], 3) / 3 * (base - 30);
        g.fillStyle = css(mixc(PAL.green, PAL.red, clamp(bar[i] / 2.5, 0, 1)));
        g.fillRect(bx + i * bw + 6, base - hgt, bw - 12, hgt);
        g.fillStyle = css(PAL.fg); g.fillText(bar[i].toFixed(2), bx + i * bw + bw / 2, base - hgt - 4);
        g.fillStyle = css(PAL.muted); g.fillText('px ' + (i + 1), bx + i * bw + bw / 2, base + 12);
      }
    };
    let ceT = [0, 0, 0, 0];
    wire(root, () => { lastDraw = 0; });
    c.onresize = () => (lastDraw = 0);
    AI.loop(root, (dt) => {
      const key = [1, 2, 3, 4].map((i) => q('#p' + i).value + q('#y' + i).value).join() + q('#thr').value + c.w;
      if (key !== lastDraw) { lastDraw = key; ceT = read().ce; }
      const settled = bar.every((v, i) => Math.abs(v - ceT[i]) < 1e-3);
      if (!settled || key === lastDraw) draw(ceT, dt);
    }, { fps: 30 });
  };

  /* =====================================================================
   * 4. Tiles, objects and resolution
   * ===================================================================== */
  AI.labs.tilesize = function (root) {
    AI.labShell(root, 'Tile size, object size and resolution', 'a 2 km × 2 km harbour cut into tiles: a ship is found only if it is big enough in pixels and lies whole inside one tile',
      '<canvas style="height:380px"></canvas><div class="caption" id="cap"></div>',
      [sel('gsd', 'Ground sample distance', [[0.3, '0.3 m'], [0.5, '0.5 m'], [1, '1 m'], [2, '2 m'], [5, '5 m'], [10, '10 m']], 0.5),
        sel('tile', 'Tile size (px)', [[128, '128'], [256, '256'], [512, '512'], [1024, '1024']], 512),
        sl('ov', 'Overlap (px, at most tile − 16)', 0, 256, 8, 64), sl('L', 'Ship length (m)', 5, 60, 1, 20),
        sl('minpx', 'Smallest detectable object (px)', 2, 32, 1, 8),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button><button class="btn small" id="st">Next tile ▶</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', root));
    const SIDE = 2000, R = AI.rng(11);
    const SHIPS = Array.from({ length: 30 }, () => ({ x: 20 + R() * 1960, y: 20 + R() * 1960, f: 0.8 + R() * 0.4 }));
    let scan = 0, paused = false, lastKey = '', PAL = pal(), plan = null;
    const compute = () => {
      const g = +q('#gsd').value, T = +q('#tile').value, O = Math.min(+q('#ov').value, T - 16), st = T - O, Lm = +q('#L').value, minpx = +q('#minpx').value;
      const sideP = Math.round(SIDE / g), starts = [];
      let a = 0;
      for (;;) { starts.push(a); if (a + T >= sideP) break; a += st; }
      starts[starts.length - 1] = Math.max(0, Math.min(starts[starts.length - 1], sideP - T));
      const inside = (lo, hi) => starts.some((s0) => s0 <= lo && hi <= s0 + T);
      const ships = SHIPS.map((s) => {
        const len = Lm * s.f, px = len / g, bx0 = (s.x - len / 2) / g, bx1 = (s.x + len / 2) / g, by0 = (s.y - len / 2) / g, by1 = (s.y + len / 2) / g;
        const whole = inside(bx0, bx1) && inside(by0, by1);
        const status = px < minpx ? 'small' : whole ? 'ok' : 'cut';
        return { ...s, len, px, status };
      });
      const N = starts.length;
      const counts = { ok: 0, cut: 0, small: 0 }; ships.forEach((s) => (counts[s.status] += 1));
      const sMean = (Lm * 1.0) / g;
      return { g, T, O, st, sideP, starts, N, ships, counts, pInside: Math.min(1, (T - sMean) / (T - O)), tileM: T * g, Lm, minpx };
    };
    const draw = () => {
      PAL = pal(); const cx = c.ctx, W = c.w, H = c.h; cx.clearRect(0, 0, W, H);
      const P = plan, sc = (W - 4) / SIDE, nT = P.N * P.N, cur = scan % nT, ci = Math.floor(cur / P.N), cj = cur % P.N;
      cx.fillStyle = css(PAL.bg2); cx.fillRect(2, 2, W - 4, H - 4);
      // tile outlines
      if (nT <= 900) {
        cx.strokeStyle = css(PAL.border); cx.lineWidth = 1;
        P.starts.forEach((x0) => P.starts.forEach((y0) => cx.strokeRect(2 + x0 * P.g * sc, 2 + y0 * P.g * sc, P.T * P.g * sc, P.T * P.g * sc)));
      }
      // current tile
      const x0 = P.starts[cj], y0 = P.starts[ci];
      cx.fillStyle = css(PAL.blue, 0.18); cx.fillRect(2 + x0 * P.g * sc, 2 + y0 * P.g * sc, P.T * P.g * sc, P.T * P.g * sc);
      cx.strokeStyle = css(PAL.blue); cx.lineWidth = 2; cx.strokeRect(2 + x0 * P.g * sc, 2 + y0 * P.g * sc, P.T * P.g * sc, P.T * P.g * sc);
      // ships
      P.ships.forEach((s) => {
        const col = s.status === 'ok' ? PAL.green : s.status === 'cut' ? PAL.red : PAL.muted;
        const d = Math.max(4, s.len * sc);
        cx.fillStyle = css(col); cx.fillRect(2 + s.x * sc - d / 2, 2 + s.y * sc - d / 2, d, d);
      });
      q('#cap').textContent = `Tile ${cur + 1} of ${nT} (origin ${x0}, ${y0} px). Green: whole ship inside a tile and large enough. Red: cut by every tile edge. Grey: too small to see.`;
      q('#out').innerHTML = `ship length ${AI.fmt(P.Lm, 3)} m = <b class="b">${AI.fmt(P.Lm / P.g, 3)} px</b> at ${P.g} m (detector minimum ${P.minpx} px)\ntile ${P.T} px = ${AI.fmt(P.tileM, 4)} m on the ground; stride ${P.st} px\nscene ${P.sideP} px per side → <b>${P.N} × ${P.N} = ${nT} tiles</b>\nships: <b class="g">${P.counts.ok} found</b>, <b class="r">${P.counts.cut} cut</b>, <b>${P.counts.small} too small</b> (of ${P.ships.length})\nP(whole inside) for an average ship = min(1, (T − s)/(T − O)) = <b>${AI.fmt(P.pInside, 3)}</b>`;
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    q('#st').onclick = () => { scan += 1; lastKey = ''; };
    wire(root, () => { plan = compute(); lastKey = ''; });
    plan = compute();
    c.onresize = () => (lastKey = '');
    AI.loop(root, (dt) => {
      if (!paused) scan += dt * 1.5;
      const key = [Math.floor(scan), q('#gsd').value, q('#tile').value, q('#ov').value, q('#L').value, q('#minpx').value, c.w].join();
      if (key !== lastKey) { lastKey = key; draw(); }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 5. Mask to polygon (pure geometry, exported for testing)
   * ===================================================================== */
  const GRID_ROWS = 20, GRID_COLS = 28;
  const makeGrid = () => {
    const g = Array.from({ length: GRID_ROWS }, () => Array(GRID_COLS).fill('.'));
    const rect = (x0, y0, x1, y1, ch) => { for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) g[y][x] = ch; };
    rect(2, 2, 10, 8, 'A'); rect(7, 2, 10, 4, '.');          // A: an L-shaped house
    rect(16, 2, 22, 6, 'B');                                  // B: a rectangular shed
    const stair = [[14, 17], [14, 18], [15, 19], [15, 20], [16, 21]];   // C: a staircase roof (rows 11..15)
    stair.forEach(([x0, x1], i) => { for (let x = x0; x <= x1; x++) g[11 + i][x] = 'C'; });
    return g;
  };
  // Trace the outer boundary of one label as a closed ring of pixel corners (clockwise on screen).
  const traceRing = (g, ch) => {
    const R = g.length, C = g[0].length, fg = (x, y) => y >= 0 && y < R && x >= 0 && x < C && g[y][x] === ch;
    const out = new Map(), key = (x, y) => x + ',' + y;
    const add = (x0, y0, x1, y1) => { const k = key(x0, y0); if (!out.has(k)) out.set(k, []); out.get(k).push([x1, y1]); };
    let first = null;
    for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) {
      if (!fg(x, y)) continue;
      if (!fg(x, y - 1)) { add(x, y, x + 1, y); first = first || [x, y]; }
      if (!fg(x + 1, y)) add(x + 1, y, x + 1, y + 1);
      if (!fg(x, y + 1)) add(x + 1, y + 1, x, y + 1);
      if (!fg(x - 1, y)) add(x, y + 1, x, y);
    }
    if (!first) return [];
    const ring = [first.slice()];
    let cur = [first[0] + 1, first[1]];
    for (let guard = 0; guard < 100000; guard++) {
      if (cur[0] === first[0] && cur[1] === first[1]) break;
      ring.push(cur.slice());
      const outs = out.get(key(cur[0], cur[1]));
      if (!outs || !outs.length) break;
      cur = outs.shift();
    }
    return ring;
  };
  const dpOpen = (P, eps) => {
    const keep = new Uint8Array(P.length); keep[0] = 1; keep[P.length - 1] = 1;
    const rec = (i, j) => {
      if (j <= i + 1) return;
      const ax = P[i][0], ay = P[i][1], dx = P[j][0] - ax, dy = P[j][1] - ay, L = Math.hypot(dx, dy) || 1e-12;
      let best = -1, bd = -1;
      for (let k = i + 1; k < j; k++) { const d = Math.abs(dx * (P[k][1] - ay) - dy * (P[k][0] - ax)) / L; if (d > bd) { bd = d; best = k; } }
      if (bd > eps) { keep[best] = 1; rec(i, best); rec(best, j); }
    };
    rec(0, P.length - 1);
    return P.filter((_, i) => keep[i]);
  };
  // Closed ring: split at the vertex farthest from the first one, simplify both halves.
  const simplifyRing = (ring, eps) => {
    const n = ring.length;
    if (n < 4) return ring.slice();
    let far = 0, fd = -1;
    for (let i = 1; i < n; i++) { const d = (ring[i][0] - ring[0][0]) ** 2 + (ring[i][1] - ring[0][1]) ** 2; if (d > fd) { fd = d; far = i; } }
    const A = ring.slice(0, far + 1), B = ring.slice(far).concat([ring[0]]);
    const a = dpOpen(A, eps), b = dpOpen(B, eps);
    return a.concat(b.slice(1, -1));
  };
  const shoelace = (P) => { let s = 0; for (let i = 0; i < P.length; i++) { const j = (i + 1) % P.length; s += P[i][0] * P[j][1] - P[j][0] * P[i][1]; } return Math.abs(s) / 2; };
  AI.geoDeep.makeGrid = makeGrid; AI.geoDeep.traceRing = traceRing; AI.geoDeep.simplifyRing = simplifyRing; AI.geoDeep.shoelace = shoelace;

  AI.labs.polygonize = function (root) {
    AI.labShell(root, 'From a building mask to a polygon', 'the outline of each pixel mask is traced edge by edge (a staircase), then simplified with Douglas–Peucker',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div>',
      [sel('b', 'Building', [['A', 'A · L-shaped house'], ['B', 'B · rectangular shed'], ['C', 'C · staircase roof']], 'A'),
        sl('eps', 'Tolerance ε (pixels)', 0, 2, 0.05, 0.5),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', root));
    const G = makeGrid();
    let ring = [], simp = [], tp = 0, paused = false, lastKey = '', lastB = '', PAL = pal(), pixels = 0;
    const build = () => {
      const ch = q('#b').value; ring = traceRing(G, ch); pixels = G.flat().filter((v) => v === ch).length; lastB = ch; tp = 0;
    };
    const simplify = () => { simp = simplifyRing(ring, +q('#eps').value); };
    const draw = () => {
      PAL = pal(); const g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const cs = Math.min(W / GRID_COLS, H / GRID_ROWS), ox = (W - cs * GRID_COLS) / 2, oy = (H - cs * GRID_ROWS) / 2 + 4;
      const cols = { A: PAL.blue, B: PAL.green, C: PAL.orange };
      for (let y = 0; y < GRID_ROWS; y++) for (let x = 0; x < GRID_COLS; x++) {
        const ch = G[y][x]; g.fillStyle = ch === '.' ? css(PAL.bg2) : css(cols[ch], ch === q('#b').value ? 0.85 : 0.25);
        g.fillRect(ox + x * cs, oy + y * cs, cs, cs);
      }
      g.strokeStyle = css(PAL.border, 0.5); g.lineWidth = 0.5;
      for (let x = 0; x <= GRID_COLS; x++) { g.beginPath(); g.moveTo(ox + x * cs, oy); g.lineTo(ox + x * cs, oy + GRID_ROWS * cs); g.stroke(); }
      for (let y = 0; y <= GRID_ROWS; y++) { g.beginPath(); g.moveTo(ox, oy + y * cs); g.lineTo(ox + GRID_COLS * cs, oy + y * cs); g.stroke(); }
      const n = Math.floor(tp);
      if (ring.length) {
        g.strokeStyle = css(PAL.fg, 0.8); g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath();
        ring.slice(0, Math.max(1, n)).forEach(([x, y], i) => (i ? g.lineTo(ox + x * cs, oy + y * cs) : g.moveTo(ox + x * cs, oy + y * cs)));
        if (n >= ring.length) g.closePath(); g.stroke(); g.setLineDash([]);
      }
      if (n >= ring.length && simp.length) {
        g.strokeStyle = css(PAL.yellow); g.lineWidth = 3; g.beginPath();
        simp.forEach(([x, y], i) => (i ? g.lineTo(ox + x * cs, oy + y * cs) : g.moveTo(ox + x * cs, oy + y * cs)));
        g.closePath(); g.stroke();
        g.fillStyle = css(PAL.yellow); simp.forEach(([x, y]) => { g.beginPath(); g.arc(ox + x * cs, oy + y * cs, 3, 0, 7); g.fill(); });
      }
      const area = simp.length ? shoelace(simp) : 0, err = pixels ? (100 * (area - pixels)) / pixels : 0;
      q('#cap').textContent = n < ring.length ? `Tracing the outline: ${n} of ${ring.length} pixel corners.` : `Simplified to ${simp.length} vertices with ε = ${fmtv(q('#eps').value)} px (dashed = full staircase, yellow = polygon).`;
      q('#out').innerHTML = `pixels in mask        <b>${pixels}</b>\npixel-corner ring    <b>${ring.length}</b> vertices (every edge)\npolygon vertices     <b class="g">${simp.length}</b>\nshoelace area        <b class="b">${AI.fmt(area, 4)}</b> px²   (error ${AI.fmt(err, 3)}%)\n<span class="m">At ε = 0 the area equals the pixel count exactly.</span>`;
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    wire(root, () => { if (q('#b').value !== lastB) build(); simplify(); lastKey = ''; });
    build(); simplify();
    c.onresize = () => (lastKey = '');
    AI.loop(root, (dt) => {
      if (!paused) tp += dt * 14;
      if (tp > ring.length + 30) { tp = 0; }
      const key = [Math.floor(tp), q('#b').value, q('#eps').value, c.w].join();
      if (key !== lastKey) { lastKey = key; draw(); }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 6. Siamese change map
   * ===================================================================== */
  AI.labs.changesiam = function (root) {
    AI.labShell(root, 'Siamese change map', 'two dates go through the same encoder; the distance between features is the change score',
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><canvas style="height:190px"></canvas><canvas style="height:190px"></canvas><canvas style="height:190px"></canvas><canvas style="height:190px"></canvas></div><div class="caption" id="cap"></div>',
      [sl('tau', 'Threshold τ on the change distance', 0, 0.6, 0.02, 0.2),
        sel('shift', 'Seasonal shift (brighter season on date 2)', [['off', 'off'], ['on', 'on (+0.30)']], 'off'),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button><button class="btn small" id="rs">↺ Replay</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), cvs = AI.$$('canvas', root), cs = cvs.map((cv) => AI.canvas(cv));
    const N = 20, R = AI.rng(5), NZ = Float64Array.from({ length: N * N }, () => (R() - 0.5) * 0.04);
    const idx = (x, y) => y * N + x;
    const inRect = (x, y, x0, y0, x1, y1) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
    let v1 = null, v2 = null, F = null, rows = 0, paused = false, lastKey = '', PAL = pal(), tot = null;
    const build = () => {
      v1 = new Float64Array(N * N); v2 = new Float64Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const b = 0.5 + 0.1 * Math.sin(x * 0.7) * Math.cos(y * 0.5) + (Math.floor(y / 4) % 2 === 0 ? 0.08 : -0.08) + NZ[idx(x, y)];
        v1[idx(x, y)] = b;
        let nb = b;
        if (inRect(x, y, 4, 4, 8, 7) || inRect(x, y, 13, 11, 16, 14)) nb += 0.35;   // two new buildings on date 2
        v2[idx(x, y)] = nb + (q('#shift').value === 'on' ? 0.3 : 0);
      }
      const med = (a) => { const s = Array.from(a).sort((p, r) => p - r); return s[s.length >> 1]; };
      const m1 = med(v1), m2 = med(v2);
      const tex = (v, x, y) => { let s = 0, n = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = clamp(x + dx, 0, N - 1), yy = clamp(y + dy, 0, N - 1); s += v[idx(xx, yy)]; n++; } return v[idx(x, y)] - s / n; };
      F = { d: new Float64Array(N * N), raw: new Float64Array(N * N) };
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = idx(x, y), f1 = v1[i] - m1, f2 = v2[i] - m2, g1 = tex(v1, x, y), g2 = tex(v2, x, y);
        F.d[i] = Math.hypot(f2 - f1, g2 - g1); F.raw[i] = Math.abs(v2[i] - v1[i]);
      }
    };
    const panel = (cx, x0, y0, w, h, fn, label) => {
      const sx = w / N, sy = h / N;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { cx.fillStyle = css(fn(x, y)); cx.fillRect(x0 + x * sx, y0 + y * sy, sx + 0.4, sy + 0.4); }
      cx.fillStyle = css(PAL.fg); cx.font = '11px sans-serif'; cx.textAlign = 'left'; cx.fillText(label, x0 + 4, y0 + 13);
    };
    const draw = () => {
      PAL = pal(); const tau = +q('#tau').value, shown = Math.floor(rows);
      const heat = (t) => ramp(clamp(t, 0, 1), HEAT);
      const panels = [
        [cs[0], (x, y) => GREY((v1[idx(x, y)] - 0.2) / 0.8), 'date 1 (bands → grey)'],
        [cs[1], (x, y) => GREY((v2[idx(x, y)] - 0.2) / 0.8), 'date 2'],
        [cs[2], (x, y) => heat(F.raw[idx(x, y)] / 0.5), 'raw difference |date2 − date1|'],
        [cs[3], (x, y) => { const i = idx(x, y); const ch = y < shown && F.d[i] > tau; return ch ? [248, 81, 73] : heat(F.d[i] / 0.5); }, 'siamese distance (red = changed)'],
      ];
      panels.forEach(([cc, fn, label]) => { const g = cc.ctx; g.clearRect(0, 0, cc.w, cc.h); panel(g, 0, 0, cc.w, cc.h, fn, label); });
      let cnt = 0, rawCnt = 0, sCnt = 0;
      for (let i = 0; i < N * N; i++) { if (F.raw[i] > tau) rawCnt++; if (F.d[i] > tau) sCnt++; }
      for (let y = 0; y < shown; y++) for (let x = 0; x < N; x++) if (F.d[idx(x, y)] > tau) cnt++;
      q('#cap').textContent = `Scanning row ${Math.min(N, shown)} of ${N}: both dates pass through the same encoder. Changed pixels are red.`;
      q('#out').innerHTML = `changed (siamese, full scene): <b class="g">${sCnt} pixels = ${num(sCnt * 100)} m²</b> at 10 m per pixel\nraw difference at the same τ flags: <b class="r">${rawCnt} pixels</b>${q('#shift').value === 'on' ? ' (the seasonal shift lights the scene)' : ''}\nchanged so far (rows scanned): <b>${cnt}</b> pixels = ${num(cnt * 100)} m²`;
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    q('#rs').onclick = () => { rows = 0; paused = false; q('#pp').textContent = '⏸ Pause'; };
    wire(root, () => { build(); lastKey = ''; });
    build();
    cs.forEach((cc) => (cc.onresize = () => (lastKey = '')));
    AI.loop(root, (dt) => {
      if (!paused) { rows += dt * 5; if (rows > N + 12) rows = 0; }
      const key = [Math.min(N, Math.floor(rows)), q('#tau').value, q('#shift').value, cs[0].w].join();
      if (key !== lastKey) { lastKey = key; draw(); }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 7. SAR and optical, linear and dB
   * ===================================================================== */
  AI.labs.sarview = function (root) {
    AI.labShell(root, 'SAR and optical, linear and decibels', 'the same scene: radar sees the lake and the city through cloud; log scaling spreads the dark values out',
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><canvas style="height:250px"></canvas><canvas style="height:250px"></canvas></div><canvas style="height:110px;margin-top:8px"></canvas><div class="caption" id="cap"></div>',
      [sel('mode', 'SAR display', [['lin', 'linear σ⁰ (0 … 1)'], ['db', 'decibels (dB)']], 'lin'), sel('cloud', 'Cloud over the optical image', [['on', 'on'], ['off', 'off']], 'on'),
        sl('spk', 'Speckle strength', 0, 1, 0.05, 0.6), sl('dmin', 'dB floor', -40, -10, 1, -25), sl('dmax', 'dB ceiling', 0, 10, 1, 5),
        '<label><span>Display formula, dB as a function of linear x (natural log is log)</span><input type="text" id="expr" value="10*log(x)/log(10)" spellcheck="false"></label><div class="err" id="err"></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), cvs = AI.$$('canvas', root), cO = AI.canvas(cvs[0]), cS = AI.canvas(cvs[1]), cH = AI.canvas(cvs[2]);
    const N = 40, R = AI.rng(21), U = Float64Array.from({ length: N * N }, () => Math.max(1e-6, R())), E = Float64Array.from(U, (u) => -Math.log(u));
    const reg = (x, y) => {
      if ((x - 12) ** 2 + (y - 28) ** 2 < 64) return 'water';
      if (x >= 26 && x <= 38 && y >= 4 && y <= 16) return (x * 7 + y * 3) % 5 < 2 ? ((x % 4 === 0 && y % 4 === 0) ? 'bright' : 'city') : 'street';
      if (x < 22 && y < 12) return 'forest';
      return (x + y) % 6 < 3 ? 'field' : 'field2';
    };
    const SIG = { water: 0.01, city: 0.5, bright: 1.0, street: 0.02, forest: 0.1, field: 0.05, field2: 0.03 };
    const OPT = { water: [30, 80, 150], city: [150, 150, 150], bright: [215, 212, 200], street: [85, 85, 92], forest: [40, 100, 50], field: [150, 170, 70], field2: [118, 150, 58] };
    const cloudAt = (x, y) => (x - 30) ** 2 + (y - 30) ** 2 < 81;
    let DBF = (s) => 10 * Math.log10(s.x), morph = 0, lastKey = '', paused = false, PAL = pal(), cur = null;
    const compute = () => {
      const spk = +q('#spk').value;
      const I = new Float64Array(N * N), regs = new Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x, r = reg(x, y); regs[i] = r;
        I[i] = SIG[r] * ((1 - spk) + spk * E[i]);
      }
      return { I, regs };
    };
    const dbOf = (v) => { try { const d = DBF({ x: Math.max(v, 1e-4) }); return Number.isFinite(d) ? d : -40; } catch (e) { return -40; } };
    const stats = (C, mode) => {
      const names = ['water', 'field', 'forest', 'city'], groups = { water: ['water'], field: ['field', 'field2'], forest: ['forest'], city: ['city', 'bright'] };
      return names.map((nm) => {
        let s = 0, n = 0;
        for (let i = 0; i < N * N; i++) if (groups[nm].includes(C.regs[i])) { const v = mode === 'db' ? dbOf(C.I[i]) : C.I[i]; s += v; n++; }
        return [nm, n ? s / n : NaN];
      });
    };
    const draw = () => {
      PAL = pal(); const mode = q('#mode').value, cloud = q('#cloud').value === 'on', dmin = +q('#dmin').value, dmax = +q('#dmax').value;
      const target = mode === 'db' ? 1 : 0;
      const oc = cO.ctx, sc = cS.ctx, hc = cH.ctx;
      oc.clearRect(0, 0, cO.w, cO.h); sc.clearRect(0, 0, cS.w, cS.h); hc.clearRect(0, 0, cH.w, cH.h);
      const cw = cO.w / N, ch = cO.h / N, sw = cS.w / N, sh = cS.h / N;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x, r = cur.regs[i], o = OPT[r], shade = 0.92 + 0.08 * Math.sin(x * 1.7 + y * 2.3);
        let col = o.map((v) => Math.round(v * shade));
        if (cloud && cloudAt(x, y)) col = mixc(col, [240, 240, 245], 0.95);
        oc.fillStyle = css(col); oc.fillRect(x * cw, y * ch, cw + 0.4, ch + 0.4);
        const lin = clamp(cur.I[i], 0, 1), dbn = clamp((dbOf(cur.I[i]) - dmin) / (dmax - dmin), 0, 1);
        const v = lerp(lin, dbn, morph);
        sc.fillStyle = css(GREY(v)); sc.fillRect(x * sw, y * sh, sw + 0.4, sh + 0.4);
      }
      oc.fillStyle = css(PAL.fg); oc.font = '11px sans-serif'; oc.fillText('optical (true colour)' + (cloud ? ', cloud on' : ''), 6, 14);
      sc.fillStyle = css(PAL.fg); sc.font = '11px sans-serif'; sc.fillText(mode === 'db' ? 'SAR, dB' : 'SAR, linear σ⁰', 6, 14);
      // histogram of the SAR values in the current unit
      const bins = 30, lo = mode === 'db' ? dmin : 0, hi = mode === 'db' ? dmax : 1, cnt = new Array(bins).fill(0);
      for (let i = 0; i < N * N; i++) { const v = mode === 'db' ? dbOf(cur.I[i]) : cur.I[i], b = Math.floor(((clamp(v, lo, hi - 1e-9) - lo) / (hi - lo)) * bins); cnt[b]++; }
      const mx = Math.max(...cnt, 1), bw = cH.w / bins;
      hc.fillStyle = css(PAL.blue, 0.8); cnt.forEach((n, b) => hc.fillRect(b * bw + 1, cH.h - 16 - (n / mx) * (cH.h - 30), bw - 2, (n / mx) * (cH.h - 30)));
      hc.fillStyle = css(PAL.muted); hc.font = '10.5px sans-serif'; hc.fillText(mode === 'db' ? `${dmin} dB` : '0', 4, cH.h - 3); hc.fillText(mode === 'db' ? `${dmax} dB` : '1.0', cH.w - 40, cH.h - 3);
      hc.fillText('histogram of SAR pixel values (speckle included)', 60, 12);
      const sLin = stats(cur, 'lin'), sDb = stats(cur, 'db'), st = mode === 'db' ? sDb : sLin;
      const unit = (v) => (mode === 'db' ? `${AI.fmt(v, 3)} dB` : AI.fmt(v, 3));
      const contrast = mode === 'db' ? `fields − water = ${AI.fmt(sDb[1][1] - sDb[0][1], 3)} dB` : `fields ÷ water = ${AI.fmt(sLin[1][1] / sLin[0][1], 3)}×`;
      q('#out').innerHTML = `region means (${mode === 'db' ? 'mean of dB values' : 'mean of linear values'}):\n${st.map(([n, v]) => `  ${n.padEnd(7)} <b>${unit(v)}</b>`).join('\n')}\nwater vs fields: <b class="g">${contrast}</b>\n<span class="m">Power ratio 5× is 7 dB. Linear values hide the dark end; dB spreads it.</span>`;
      q('#cap').textContent = mode === 'db' ? 'dB: each 10 dB is a factor of 10 in power, so the dark water and the fields separate. Speckle is now additive noise.' : 'Linear σ⁰: the bright city fills the scale and the dark water is squeezed to black. Switch to dB.';
    };
    const parseDB = () => {
      const inp = q('#expr'), err = q('#err');
      try { const ast = AI.parse(inp.value); const fn = AI.compile(ast, ['x']); fn({ x: 1 }); DBF = fn; err.textContent = ''; inp.classList.remove('bad'); }
      catch (e) { inp.classList.add('bad'); err.textContent = e.message; }
    };
    q('#expr').addEventListener('input', () => { parseDB(); lastKey = ''; });
    q('#expr').addEventListener('change', () => { parseDB(); lastKey = ''; });
    wire(root, () => { cur = compute(); lastKey = ''; });
    parseDB(); cur = compute();
    [cO, cS, cH].forEach((cc) => (cc.onresize = () => (lastKey = '')));
    AI.loop(root, (dt) => {
      const target = q('#mode').value === 'db' ? 1 : 0;
      if (Math.abs(morph - target) > 0.001) { morph += Math.sign(target - morph) * Math.min(Math.abs(target - morph), dt / 1.5); }
      else morph = target;
      const key = [q('#mode').value, q('#cloud').value, q('#dmin').value, q('#dmax').value, Math.round(morph * 200), cO.w, cS.w, q('#spk').value, q('#expr').value].join();
      if (key !== lastKey) { lastKey = key; draw(); }
    }, { fps: 30 });
  };

  /* =====================================================================
   * 8. Sliding-window inference
   * ===================================================================== */
  AI.labs.slidewin = function (root) {
    AI.labShell(root, 'Sliding-window inference and stitching', 'a 928-pixel scene predicted tile by tile; the model is least reliable at tile edges, so overlap and blending hide the seams',
      '<canvas style="height:330px"></canvas><canvas style="height:120px;margin-top:8px"></canvas><div class="caption" id="cap"></div>',
      [sel('tile', 'Tile size (px)', [[128, '128'], [256, '256'], [512, '512']], 256), sl('ov', 'Overlap (px)', 0, 120, 8, 32),
        sel('mode', 'Combining overlaps', [['hard', 'hard: newest tile overwrites'], ['blend', 'blend: triangular weights']], 'hard'),
        sl('sp', 'Speed', 0.3, 2, 0.1, 1),
        '<div class="row"><button class="btn small primary" id="pp">⏸ Pause</button><button class="btn small" id="rs">↺ Restart</button></div>',
        '<div class="readout" id="out"></div>'].join(''));
    const q = (s) => AI.$(s, root), cvs = AI.$$('canvas', root), cM = AI.canvas(cvs[0]), cX = AI.canvas(cvs[1]);
    const W = 928, GR = 4, NC = W / GR;     // scene 928 px, drawn on a 4-px grid
    const truth = (x, y) => 0.5 + 0.4 * Math.sin(x / 160) * Math.cos(y / 210);
    let T = 256, O = 32, MODE = 'hard', tiles = [], t = 0, paused = false, lastKey = '', PAL = pal(), res = null;
    const RB = AI.rng(99);
    const build = () => {
      T = +q('#tile').value; O = Math.min(+q('#ov').value, T - 16); MODE = q('#mode').value;
      const st = T - O, starts = []; let a = 0;
      for (;;) { starts.push(a); if (a + T >= W) break; a += st; }
      starts[starts.length - 1] = Math.max(0, Math.min(starts[starts.length - 1], W - T));
      tiles = []; starts.forEach((b) => starts.forEach((a2) => tiles.push({ a: a2, b, bias: (AI.rng(a2 * 31 + b * 7 + 3)() - 0.5) * 0.5 })));
      void RB; t = 0; res = null; acc = null;
    };
    const tilePred = (tl, x, y) => {
      const d = Math.min(x - tl.a, tl.a + T - 1 - x, y - tl.b, tl.b + T - 1 - y);
      const e = Math.max(0, 1 - d / 40);
      return clamp(truth(x, y) + tl.bias * e, 0, 1);
    };
    // accumulators: weighted sums and weights (blend), last written value (hard); updated tile by tile
    let acc = null;
    const addTile = (tl) => {
      const r0 = Math.max(0, Math.floor(tl.b / GR)), r1 = Math.min(NC - 1, Math.floor((tl.b + T - 1) / GR));
      const c0 = Math.max(0, Math.floor(tl.a / GR)), c1 = Math.min(NC - 1, Math.floor((tl.a + T - 1) / GR));
      for (let r = r0; r <= r1; r++) for (let cc = c0; cc <= c1; cc++) {
        const x = cc * GR + GR / 2, y = r * GR + GR / 2;
        if (x < tl.a || x >= tl.a + T || y < tl.b || y >= tl.b + T) continue;
        const j = x - tl.a, kk = y - tl.b, w = Math.min(j + 1, T - j) * Math.min(kk + 1, T - kk), p = tilePred(tl, x, y), i = r * NC + cc;
        acc.S[i] += w * p; acc.Wt[i] += w; acc.last[i] = p; acc.cov[i] = 1;
      }
    };
    const compute = (nDone) => {
      if (!acc || acc.n > nDone) acc = { S: new Float64Array(NC * NC), Wt: new Float64Array(NC * NC), last: new Float64Array(NC * NC).fill(NaN), cov: new Uint8Array(NC * NC), n: 0 };
      while (acc.n < nDone) { addTile(tiles[acc.n]); acc.n++; }
      const out = new Float64Array(NC * NC);
      for (let i = 0; i < NC * NC; i++) out[i] = MODE === 'blend' ? (acc.Wt[i] ? acc.S[i] / acc.Wt[i] : NaN) : acc.last[i];
      return { out, cov: acc.cov };
    };
    const ramp3 = (v) => ramp(v, [[27, 42, 74], [120, 150, 110], [242, 204, 96]]);
    const draw = (ci) => {
      PAL = pal(); const cx = cM.ctx, Wc = cM.w, Hc = cM.h; cx.clearRect(0, 0, Wc, Hc);
      const sx = Wc / NC, sy = Hc / NC, nDone = Math.min(ci, tiles.length);
      if (!res || res.n !== nDone || res.mode !== MODE || res.T !== T || res.O !== O) res = { ...compute(nDone), n: nDone, mode: MODE, T, O };
      for (let r = 0; r < NC; r++) for (let cc = 0; cc < NC; cc++) {
        const i = r * NC + cc;
        cx.fillStyle = res.cov[i] ? css(ramp3(res.out[i])) : css(PAL.bg2);
        cx.fillRect(cc * sx, r * sy, sx + 0.5, sy + 0.5);
      }
      if (nDone < tiles.length) {
        const tl = tiles[nDone]; cx.strokeStyle = css(PAL.fg); cx.lineWidth = 2;
        cx.strokeRect(tl.a * sx / GR, tl.b * sy / GR, T * sx / GR, T * sy / GR);
      }
      // cross-section through the middle row
      const gx = cX.ctx, Wx = cX.w, Hx = cX.h; gx.clearRect(0, 0, Wx, Hx);
      const row = Math.floor(NC / 2) + 0, yv = row * GR + GR / 2;
      gx.strokeStyle = css(PAL.muted, 0.7); gx.setLineDash([4, 4]); gx.beginPath();
      for (let cc = 0; cc < NC; cc++) { const x = cc * Wx / NC, yy = Hx - 8 - truth(cc * GR + 2, yv) * (Hx - 16); cc ? gx.lineTo(x, yy) : gx.moveTo(x, yy); }
      gx.stroke(); gx.setLineDash([]);
      gx.strokeStyle = css(PAL.orange); gx.lineWidth = 2; gx.beginPath();
      let started = false, maxJump = 0, prev = NaN, errSum = 0, errN = 0;
      for (let cc = 0; cc < NC; cc++) {
        const i = row * NC + cc, v = res.out[i];
        if (!res.cov[i]) { started = false; prev = NaN; continue; }
        const x = cc * Wx / NC, yy = Hx - 8 - v * (Hx - 16);
        if (!started) { gx.moveTo(x, yy); started = true; } else gx.lineTo(x, yy);
        if (!Number.isNaN(prev)) maxJump = Math.max(maxJump, Math.abs(v - prev));
        prev = v; errSum += Math.abs(v - truth(cc * GR + 2, yv)); errN++;
      }
      gx.stroke();
      gx.fillStyle = css(PAL.muted); gx.font = '10.5px sans-serif'; gx.fillText('dashed: truth · orange: output along one row (values, 0 at bottom, 1 at top)', 6, 12);
      const tot = tiles.length, cur = Math.min(nDone, tot - 1);
      q('#cap').textContent = nDone >= tot ? `All ${tot} tiles done. ${MODE === 'hard' ? 'Hard overwrite: vertical steps show where a newer tile starts.' : 'Blended: the overlaps are weighted toward each tile centre, so the steps shrink.'}`
        : `Tile ${cur + 1} of ${tot} (origin x = ${tiles[cur].a}, y = ${tiles[cur].b}). Each tile's own border error is strongest at its edge.`;
      const wsum = res.cov.reduce((a, b) => a + b, 0);
      q('#out').innerHTML = `tiles: ${Math.round(Math.sqrt(tot))} per side = <b>${tot}</b>, stride ${T - O} px, overlap ${O} px\nlargest step between neighbouring samples on the row: <b class="${maxJump > 0.12 ? 'r' : 'g'}">${AI.fmt(maxJump, 3)}</b>\nmean |output − truth| on the row: <b>${errN ? AI.fmt(errSum / errN, 3) : '–'}</b>\naccumulators: 2 scene-sized arrays (sum, weight) = <b>${num(2 * W * W * 4 / 2 ** 20 * 1)}</b> MiB at float32 for the full ${W}² scene`;
    };
    q('#pp').onclick = () => { paused = !paused; q('#pp').textContent = paused ? '▶ Play' : '⏸ Pause'; };
    q('#rs').onclick = () => { t = 0; paused = false; q('#pp').textContent = '⏸ Pause'; lastKey = ''; };
    wire(root, () => { build(); lastKey = ''; });
    build();
    [cM, cX].forEach((cc) => (cc.onresize = () => { lastKey = ''; res = null; }));
    AI.loop(root, (dt) => {
      if (!paused) { t += dt * (+q('#sp').value) * 0.9; if (t > tiles.length * 1.0 + 2.5) t = 0; }
      const ci = Math.min(tiles.length, Math.floor(t));
      const key = [ci, q('#tile').value, q('#ov').value, q('#mode').value, cM.w].join();
      if (key !== lastKey) { lastKey = key; draw(ci); }
    }, { fps: 30 });
  };
})();
