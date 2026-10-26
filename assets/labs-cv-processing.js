/* Computer Vision · Module 1 (classical image processing) labs.
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes).
 *   pointops  histeq  filterlab  cannylab  morpho  interp  alias  pyramid  warp  freq  sines
 * All animate in real time (dt), start slow, have a speed slider + pause/step, and only redraw when something changed.
 */
(function () {
  const CV = AI.cv, $ = AI.$, clamp = AI.clamp, lerp = AI.lerp, smooth = AI.smooth;

  /* ---------- small shared helpers ---------- */
  const sl = (id, label, min, max, val, step) => `<label>${label} <span class="v" data-v="${id}"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step || 1}" value="${val}"></label>`;
  const sel = (id, label, opts, val) => `<label>${label} <select id="${id}">${Object.entries(opts).map(([k, v]) => `<option value="${k}"${k == val ? ' selected' : ''}>${v}</option>`).join('')}</select></label>`;
  const chk = (id, label, on) => `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="${id}"${on ? ' checked' : ''}> ${label}</label>`;
  const PLAY = (sp) => `<label>Speed <span class="v" data-v="sp"></span><input type="range" id="sp" min="0.1" max="3" step="0.05" value="${sp || 1}"></label>
    <div class="row"><button class="btn" id="back">◀ Step</button><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">↺ Restart</button></div>`;
  const ERR = (id) => `<div id="${id}" style="color:var(--red);font-size:12px;min-height:0"></div>`;
  const chipRow = (id) => `<div class="chips" id="${id}" style="margin-top:4px"></div>`;
  const chip = (parent, label, on) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = on; parent.appendChild(b); return b; };
  function vals(root) { const upd = () => root.querySelectorAll('[data-v]').forEach((s) => { const i = root.querySelector('#' + s.dataset.v); if (i) s.textContent = i.value; }); root.addEventListener('input', upd); upd(); return upd; }
  function player(q) {
    const P = { playing: true, sp: () => +q('#sp').value };
    P.set = (v) => { P.playing = v; q('#play').textContent = v ? '⏸ Pause' : '▶ Play'; };
    q('#play').onclick = () => P.set(!P.playing);
    q('#step').onclick = () => { P.set(false); P.onstep && P.onstep(1); };
    q('#back').onclick = () => { P.set(false); P.onstep && P.onstep(-1); };
    q('#reset').onclick = () => { P.onreset && P.onreset(); };
    return P;
  }
  function mkImg(w, h) {
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d'), id = ctx.createImageData(w, h), d = id.data;
    return {
      cv, w, h, d,
      gray(a) { for (let i = 0, n = w * h; i < n; i++) { const v = a[i]; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255; } ctx.putImageData(id, 0, 0); return this; },
      rgb(f) { for (let i = 0, n = w * h; i < n; i++) { const c = f(i); d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255; } ctx.putImageData(id, 0, 0); return this; },
      flush() { ctx.putImageData(id, 0, 0); return this; },
    };
  }
  const blit = (g, im, x, y, w, h, smoothing) => { g.imageSmoothingEnabled = !!smoothing; g.drawImage(im.cv, x, y, w, h); };
  const text = (g, s, x, y, col, size, align) => { g.fillStyle = col; g.font = (size || 12) + 'px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = align || 'left'; g.textBaseline = 'alphabetic'; g.fillText(s, x, y); };
  const label = (g, s, x, y, C) => text(g, s, x, y, C.muted, 12, 'left');
  const border = (g, x, y, w, h, col, lw) => { g.strokeStyle = col; g.lineWidth = lw || 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); };
  const pingpong = (u) => { u = ((u % 1) + 1) % 1; return u < 0.25 ? 0 : u < 0.5 ? smooth((u - 0.25) / 0.25) : u < 0.75 ? 1 : 1 - smooth((u - 0.75) / 0.25); };
  const hexA = (hex, a) => { const c = AI.rgb(hex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
  const grayScene = (k, n) => CV.gray(CV.resample(k, n));
  const SC5 = { house: 'house', shapes: 'colour shapes', digit: 'handwritten 7', edges: 'edges & stripes', checker: 'checkerboard' };
  function setup(root, c, draw) { // common dirty-flag plumbing
    const S = { dirty: true };
    c.onresize = () => (S.dirty = true);
    window.addEventListener('themechange', () => (S.dirty = true));
    root.addEventListener('input', () => (S.dirty = true));
    root.addEventListener('click', () => (S.dirty = true));
    return S;
  }
  const hist = (arr, bins, lo, hi) => { const h = new Float32Array(bins), s = bins / (hi - lo); for (let i = 0; i < arr.length; i++) { const b = clamp(Math.floor((arr[i] - lo) * s), 0, bins - 1); h[b]++; } return h; };
  function bars(g, x, y, w, h, hs, col, C, title, ticks) {
    const mx = Math.max(1, ...hs), bw = w / hs.length;
    g.fillStyle = C.bg2; g.fillRect(x, y, w, h);
    g.fillStyle = col;
    for (let i = 0; i < hs.length; i++) { const bh = (hs[i] / mx) * (h - 4); g.fillRect(x + i * bw, y + h - bh, Math.max(1, bw - 0.5), bh); }
    border(g, x, y, w, h, C.border);
    text(g, title, x + 6, y + 14, C.muted, 11);
    if (ticks !== false) { text(g, '0', x, y + h + 12, C.muted, 10, 'left'); text(g, '255', x + w, y + h + 12, C.muted, 10, 'right'); }
  }

  /* ======================================================================== */
  /* pointops: editable per-pixel formula, live histograms and LUT curve       */
  /* ======================================================================== */
  AI.labs.pointops = function (root) {
    const SCN = { house: 'house', shapes: 'colour shapes', digit: 'handwritten 7', gradient: 'gradient', dim: 'house, low contrast (dull)', bright: 'house, overexposed' };
    const PRE = { identity: 'x', brighten: 'x+60', 'more contrast': '128+(x-128)*1.6', 'gamma (slider g)': '255*(x/255)^g', invert: '255-x', 'threshold (slider t)': '255*sigmoid((x-t)*40)', 'gain & bias': 'a*x+b', posterize: '85*floor(x/85)' };
    const { view } = AI.labShell(root, 'Point operations: one formula for every pixel', 'edit f(x); the image, the look-up table and both histograms follow',
      '<canvas style="height:440px"></canvas><div class="caption">Top: before and after. Bottom: histogram of the original, the look-up table x → f(x) (white dot scans every input value), histogram of the result.</div>',
      `${sel('scene', 'Image', SCN, 'house')}
       <div>Presets${chipRow('pre')}</div>
       <label>New value f(x) — x is the old value 0…255<input type="text" id="f" value="255*(x/255)^g" spellcheck="false"></label>${ERR('err')}
       ${sl('a', 'gain a', 0.2, 3, 1, 0.05)}${sl('b', 'bias b', -120, 120, 0, 1)}${sl('g', 'gamma g', 0.2, 3, 0.5, 0.05)}${sl('t', 'threshold t', 0, 255, 128, 1)}
       ${chk('morph', 'animate before → after', true)}
       ${PLAY(1)}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), upd = vals(root), S = setup(root, c);
    const imA = mkImg(120, 120), imB = mkImg(120, 120), N = 14400;
    let base = new Uint8Array(N), h0 = null, fn = null, u = 0, sweep = 0;
    const lut = new Float32Array(256), lm = new Float32Array(256), outA = new Uint8ClampedArray(N);
    const load = () => {
      const k = q('#scene').value, g = grayScene(k === 'dim' || k === 'bright' ? 'house' : k, 120), col = k === 'shapes' ? null : null;
      for (let i = 0; i < N; i++) { let v = g[i]; if (k === 'dim') v = 0.3 * v + 70; else if (k === 'bright') v = Math.min(255, v * 1.4 + 95); base[i] = Math.round(v); }
      h0 = hist(base, 64, 0, 256); imA.gray(base); S.dirty = true;
    };
    const rebuild = () => { if (!fn) return; const s = { a: +q('#a').value, b: +q('#b').value, g: +q('#g').value, t: +q('#t').value }; for (let x = 0; x < 256; x++) { s.x = x; const v = fn(s); lut[x] = Number.isFinite(v) ? clamp(Math.round(v), 0, 255) : 0; } S.dirty = true; };
    const ex = AI.exprInput(q('#f'), q('#err'), { allowed: ['x', 'a', 'b', 'g', 't'], onChange: (f) => { fn = f; rebuild(); } });
    Object.entries(PRE).forEach(([n, f]) => chip(q('#pre'), n, () => { ex.set(f); }));
    ['#a', '#b', '#g', '#t'].forEach((s) => q(s).addEventListener('input', rebuild));
    q('#scene').onchange = load;
    P.onstep = () => { u = u >= 0.5 && u < 0.75 ? 0 : 0.5; S.dirty = true; };
    P.onreset = () => { u = 0; sweep = 0; P.set(true); };
    load(); ex.run();
    AI.loop(root, (dt) => {
      const sp = P.sp();
      if (P.playing) { if (q('#morph').checked) u += dt * 0.125 * sp; sweep = (sweep + dt * 25 * sp) % 256; S.dirty = true; }
      if (!S.dirty || !fn) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const m = q('#morph').checked ? pingpong(u) : 1;
      for (let x = 0; x < 256; x++) lm[x] = lerp(x, lut[x], m);
      for (let i = 0; i < N; i++) outA[i] = lm[base[i]];
      imB.gray(outA);
      const h1 = hist(outA, 64, 0, 256), pad = 10, s = Math.min((W - 3 * pad) / 2, H * 0.5 - 6), x0 = (W - (2 * s + pad)) / 2;
      label(g, 'before', x0, 12, C); label(g, m >= 0.999 ? 'after f(x)' : `after f(x) — blend ${Math.round(m * 100)}%`, x0 + s + pad, 12, C);
      blit(g, imA, x0, 18, s, s); blit(g, imB, x0 + s + pad, 18, s, s);
      const y0 = 18 + s + 20, ph = H - y0 - 20, pw = (W - 4 * pad) / 3, xp = Math.floor(sweep), yp = lm[xp];
      bars(g, pad, y0, pw, ph, h0, C.blue, C, 'histogram before');
      bars(g, 3 * pad + 2 * pw, y0, pw, ph, h1, C.orange, C, 'histogram after');
      // LUT plot
      const lx = 2 * pad + pw, X = (v) => lx + 4 + (v / 255) * (pw - 8), Y = (v) => y0 + ph - 4 - (v / 255) * (ph - 8);
      g.fillStyle = C.bg2; g.fillRect(lx, y0, pw, ph); border(g, lx, y0, pw, ph, C.border);
      g.strokeStyle = C.border; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(255), Y(255)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.green; g.lineWidth = 2.5; g.beginPath(); for (let x = 0; x < 256; x += 2) { const px = X(x), py = Y(lm[x]); x ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke(); g.lineWidth = 1;
      g.strokeStyle = C.fg; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(xp), Y(0)); g.lineTo(X(xp), Y(yp)); g.lineTo(X(0), Y(yp)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.beginPath(); g.arc(X(xp), Y(yp), 5, 0, 7); g.fill();
      text(g, 'look-up table  x → f(x)', lx + 6, y0 + 14, C.muted, 11);
      // probe lines on histograms
      g.strokeStyle = C.fg; g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath();
      g.moveTo(pad + (xp / 256) * pw, y0); g.lineTo(pad + (xp / 256) * pw, y0 + ph); g.moveTo(3 * pad + 2 * pw + (yp / 256) * pw, y0); g.lineTo(3 * pad + 2 * pw + (yp / 256) * pw, y0 + ph); g.stroke(); g.setLineDash([]);
      const st = AI.stats(Array.from(outA).filter((_, i) => i % 7 === 0)), st0 = AI.stats(Array.from(base).filter((_, i) => i % 7 === 0));
      q('#out').innerHTML = `scanning input <b class="b">x = ${xp}</b>\nstored output <b class="o">f(x) = ${yp}</b> ${m < 1 ? '<span class="m">(blended)</span>' : ''}\n<span class="m">mean  ${AI.fmt(st0.mean, 1)} → ${AI.fmt(st.mean, 1)}\nspread ${AI.fmt(st0.std, 1)} → ${AI.fmt(st.std, 1)}</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* histeq: histogram equalisation, one stage at a time                       */
  /* ======================================================================== */
  AI.labs.histeq = function (root) {
    const SCN = { dim: 'house, low contrast', bright: 'house, overexposed', dimshapes: 'colour shapes, hazy', digit: 'handwritten 7, grey' };
    const DURS = [4.5, 5, 6, 6], CAP = [
      '① Count: how many pixels have each brightness? The bars are the histogram h(v). Notice they are squeezed into a narrow band — the picture is dull.',
      '② Accumulate: the CDF adds the bars up from left to right. cdf(v) = number of pixels with brightness ≤ v. It climbs steeply exactly where many pixels live.',
      '③ Map: the new brightness of a pixel with old value v is T(v) = 255·(cdf(v) − cdf_min)/(N − cdf_min). Read it off the CDF: input v → height → output.',
      '④ Apply: replace every pixel by T(v). The dense band is stretched over the whole range, and the new histogram becomes roughly flat.'];
    const { view } = AI.labShell(root, 'Histogram equalisation, step by step', 'histogram → CDF → mapping → new image',
      '<canvas style="height:400px"></canvas><div class="caption" id="cap"></div>',
      `${sel('scene', 'Image', SCN, 'dim')}${PLAY(1)}<div>Jump to stage<div class="chips" id="stg"></div></div><div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const N = 14400, im = mkImg(120, 120), base = new Uint8Array(N), eq = new Uint8Array(N), cur = new Uint8ClampedArray(N), mapT = new Float32Array(256), cdf = new Float32Array(256);
    let h = null, T = 0, last = -1, cdfMin = 0;
    const START = DURS.map((_, i) => DURS.slice(0, i).reduce((a, b) => a + b, 0)), TOT = DURS.reduce((a, b) => a + b, 0);
    const load = () => {
      const k = q('#scene').value, g = grayScene(k === 'digit' ? 'digit' : k === 'dimshapes' ? 'shapes' : 'house', 120);
      for (let i = 0; i < N; i++) { let v = g[i]; if (k === 'dim' || k === 'dimshapes') v = 0.3 * v + 70; else if (k === 'bright') v = Math.min(255, v * 1.4 + 95); else v = 0.35 * v + 40; base[i] = Math.round(v); }
      h = hist(base, 256, 0, 256); let s = 0; cdfMin = 0;
      for (let v = 0; v < 256; v++) { s += h[v]; cdf[v] = s; if (!cdfMin && h[v]) cdfMin = s; }
      for (let v = 0; v < 256; v++) mapT[v] = clamp(Math.round((255 * (cdf[v] - cdfMin)) / (N - cdfMin)), 0, 255);
      for (let i = 0; i < N; i++) eq[i] = mapT[base[i]];
      T = 0; S.dirty = true;
    };
    q('#scene').onchange = load;
    const stageButtons = DURS.map((_, i) => chip(q('#stg'), ['count', 'CDF', 'map', 'apply'][i], () => { P.set(false); T = START[i] + 0.01; S.dirty = true; }));
    P.onstep = (d) => { const i = DURS.findIndex((_, k) => T < START[k] + DURS[k]); T = START[(i + d + 4) % 4] + 0.01; S.dirty = true; };
    P.onreset = () => { T = 0; P.set(true); S.dirty = true; };
    load();
    AI.loop(root, (dt) => {
      if (P.playing) { T = (T + dt * P.sp()) % TOT; S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      let st = DURS.findIndex((_, k) => T < START[k] + DURS[k]); if (st < 0) st = 3; const u = (T - START[st]) / DURS[st];
      if (st !== last) { last = st; q('#cap').textContent = CAP[st]; stageButtons.forEach((b, i) => (b.style.borderColor = i === st ? 'var(--blue)' : '')); }
      const m = st === 3 ? smooth(u * 1.4 - 0.1) : st < 3 ? 0 : 1;
      for (let i = 0; i < N; i++) cur[i] = lerp(base[i], eq[i], m);
      im.gray(cur);
      const pad = 12, s = Math.min(W * 0.34, H - 60), px = pad + s + 22, pw = W - px - pad, ph = H - 64, py = 28;
      label(g, m > 0 ? 'image (blending to equalised)' : 'original image', pad, 16, C); blit(g, im, pad, 28, s, s);
      // histogram (current) + CDF
      const hc = m > 0 ? hist(cur, 256, 0, 256) : h, mx = Math.max(...h) * 1.05;
      g.fillStyle = C.bg2; g.fillRect(px, py, pw, ph); border(g, px, py, pw, ph, C.border);
      const X = (v) => px + 2 + (v / 255) * (pw - 4), Yh = (n) => py + ph - 2 - (n / mx) * (ph - 8), Yc = (n) => py + ph - 2 - (n / N) * (ph - 8);
      const reveal = st === 0 ? u * 1.15 * 255 : 255;
      g.fillStyle = C.blue; for (let v = 0; v <= Math.min(255, reveal); v++) { const y = Yh(Math.min(hc[v], mx)); g.fillRect(X(v), y, Math.max(1, (pw - 4) / 256), py + ph - 2 - y); }
      if (m > 0 && m < 1) { g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.beginPath(); for (let v = 0; v < 256; v++) { const y = Yh(Math.min(h[v], mx)); v ? g.lineTo(X(v), y) : g.moveTo(X(v), y); } g.stroke(); g.setLineDash([]); }
      let cv = 0;
      if (st === 0) { cv = Math.min(255, Math.floor(reveal)); g.strokeStyle = C.fg; g.beginPath(); g.moveTo(X(cv), py); g.lineTo(X(cv), py + ph); g.stroke(); }
      if (st === 1 || st === 2) {
        cv = st === 1 ? Math.min(255, Math.floor(u * 1.1 * 255)) : Math.min(255, Math.floor(u * 1.1 * 255));
        g.strokeStyle = C.orange; g.lineWidth = 3; g.beginPath(); for (let v = 0; v <= (st === 1 ? cv : 255); v++) { const y = Yc(cdf[v]); v ? g.lineTo(X(v), y) : g.moveTo(X(v), y); } g.stroke(); g.lineWidth = 1;
        g.strokeStyle = C.fg; g.beginPath(); g.moveTo(X(cv), py); g.lineTo(X(cv), py + ph); g.stroke();
        g.fillStyle = C.orange; g.beginPath(); g.arc(X(cv), Yc(cdf[cv]), 5, 0, 7); g.fill();
        if (st === 2) { const tv = mapT[cv]; g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(X(cv), Yc(cdf[cv])); g.lineTo(X(tv), Yc(cdf[cv])); g.lineTo(X(tv), py + ph); g.stroke(); g.setLineDash([]); g.lineWidth = 1; g.fillStyle = C.green; g.beginPath(); g.arc(X(tv), py + ph - 3, 5, 0, 7); g.fill(); text(g, `new value ${tv}`, clamp(X(tv), px + 50, px + pw - 50), py + ph - 12, C.green, 11, 'center'); }
      }
      if (st === 3) { g.strokeStyle = C.orange; g.globalAlpha = 0.5; g.lineWidth = 2; g.beginPath(); for (let v = 0; v < 256; v++) { const y = Yc(cdf[v]); v ? g.lineTo(X(v), y) : g.moveTo(X(v), y); } g.stroke(); g.globalAlpha = 1; g.lineWidth = 1; }
      text(g, st < 3 ? 'blue = histogram h(v)   orange = CDF' : 'blue = histogram of the current image   dashed = original   orange = CDF', px + 6, py + 14, C.muted, 11);
      text(g, '0', px, py + ph + 13, C.muted, 10); text(g, '255', px + pw, py + ph + 13, C.muted, 10, 'right'); text(g, 'brightness v', px + pw / 2, py + ph + 13, C.muted, 10, 'center');
      const sd = st < 3 ? Math.round(AI.stats(Array.from(base).filter((_, i) => i % 5 === 0)).std) : Math.round(AI.stats(Array.from(cur).filter((_, i) => i % 5 === 0)).std);
      q('#out').innerHTML = `stage <b class="b">${st + 1}/4</b>${st > 0 ? `\nbrightness v = <b class="o">${cv}</b>\nh(v) = ${h[cv]}   cdf(v) = ${cdf[cv]}` : ''}${st === 2 ? `\nT(v) = 255·(${cdf[cv]}−${cdfMin})/(${N}−${cdfMin}) = <b class="g">${mapT[cv]}</b>` : ''}\n<span class="m">spread (std of brightness): ${sd}</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* filterlab: larger image, blur / sharpen / size / padding                  */
  /* ======================================================================== */
  AI.labs.filterlab = function (root) {
    const FILT = { gauss: 'Gaussian blur', box: 'box blur', sharpen: 'sharpen (unsharp mask)', detail: 'high-pass: image − blur', motion: 'motion blur (horizontal)', emboss: 'emboss (3×3)' };
    const { view } = AI.labShell(root, 'Filtering a real picture', 'a k×k window sweeps over the image; the result is revealed behind it',
      '<canvas style="height:360px"></canvas><div class="caption">Left: input with the sliding k×k window. Middle: output revealed as the window passes. Right: the kernel (green = positive weights, orange = negative).</div>',
      `${sel('scene', 'Image', SC5, 'house')}${sel('type', 'Filter', FILT, 'gauss')}
       ${sl('size', 'kernel size k (=2·v+1)', 1, 7, 2, 1)}${sl('sigma', 'σ (Gaussian / sharpen width)', 0.5, 5, 1.5, 0.1)}${sl('amt', 'sharpen amount', 0, 4, 1.5, 0.1)}
       ${sel('pad', 'Border handling', { zero: 'zero padding', replicate: 'replicate edge', reflect: 'reflect' }, 'replicate')}
       ${PLAY(1.5)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const n = 120, NN = n * n, im0 = mkImg(n, n), im1 = mkImg(n, n);
    let img = grayScene('house', n), K = null, k = 3, res = new Float32Array(NN), centered = false, pos = 0, tot = 0, hold = 0;
    const pad = new Float32Array((n + 16) * (n + 16));
    const compute = () => {
      img = grayScene(q('#scene').value, n);
      const type = q('#type').value; k = 2 * +q('#size').value + 1; const r = (k - 1) / 2, sg = +q('#sigma').value, am = +q('#amt').value;
      const G = new Float32Array(k * k); let sum = 0;
      for (let j = 0; j < k; j++) for (let i = 0; i < k; i++) { const v = Math.exp(-((i - r) ** 2 + (j - r) ** 2) / (2 * sg * sg)); G[j * k + i] = v; sum += v; }
      for (let i = 0; i < G.length; i++) G[i] /= sum;
      K = new Float32Array(k * k); centered = false;
      if (type === 'box') K.fill(1 / (k * k));
      else if (type === 'gauss') K.set(G);
      else if (type === 'motion') for (let i = 0; i < k; i++) K[r * k + i] = 1 / k;
      else if (type === 'sharpen') { for (let i = 0; i < K.length; i++) K[i] = -am * G[i]; K[r * k + r] += 1 + am; }
      else if (type === 'detail') { for (let i = 0; i < K.length; i++) K[i] = -G[i]; K[r * k + r] += 1; centered = true; }
      else { k = 3; K = new Float32Array([-2, -1, 0, -1, 1, 1, 0, 1, 2]); centered = true; }
      const R = (k - 1) / 2, M = n + 2 * R, mode = q('#pad').value;
      const idx = (i) => (i < 0 ? (mode === 'reflect' ? -i - 1 : 0) : i >= n ? (mode === 'reflect' ? 2 * n - i - 1 : n - 1) : i);
      for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) { const sx = x - R, sy = y - R; pad[y * M + x] = mode === 'zero' && (sx < 0 || sy < 0 || sx >= n || sy >= n) ? 0 : img[idx(sy) * n + idx(sx)]; }
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let s = 0; for (let j = 0; j < k; j++) { const row = (y + j) * M + x; for (let i = 0; i < k; i++) s += K[j * k + i] * pad[row + i]; }
        res[y * n + x] = centered ? (type === 'emboss' ? 128 + s : 128 + 2.5 * s) : s;
      }
      im0.gray(img); im1.gray(res); S.dirty = true;
    };
    ['#scene', '#type', '#size', '#sigma', '#amt', '#pad'].forEach((s) => q(s).addEventListener('input', compute));
    P.onstep = (d) => { pos = clamp(pos + d * n * 6, 0, NN); S.dirty = true; };
    P.onreset = () => { pos = 0; hold = 0; P.set(true); };
    compute();
    AI.loop(root, (dt) => {
      if (P.playing) { if (pos >= NN) { hold += dt; if (hold > 1.8) { pos = 0; hold = 0; } } else pos = Math.min(NN, pos + dt * 600 * P.sp()); S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const gap = 12, kw = Math.min(150, W * 0.2), s = Math.min((W - kw - 4 * gap) / 2, H - 30), x1 = gap, x2 = 2 * gap + s, x3 = 3 * gap + 2 * s, y0 = 22, f = s / n;
      label(g, 'input', x1, 14, C); label(g, 'output', x2, 14, C); label(g, `kernel ${k}×${k}`, x3, 14, C);
      blit(g, im0, x1, y0, s, s); border(g, x1, y0, s, s, C.border);
      g.fillStyle = C.bg2; g.fillRect(x2, y0, s, s);
      const rows = Math.floor(pos / n), part = pos - rows * n;
      g.save(); g.beginPath(); g.rect(x2, y0, s, rows * f); g.rect(x2, y0 + rows * f, part * f, f); g.clip(); blit(g, im1, x2, y0, s, s); g.restore(); border(g, x2, y0, s, s, C.border);
      const cx = Math.floor(part), cy = Math.min(n - 1, rows), r = (k - 1) / 2;
      if (pos < NN) {
        g.fillStyle = hexA(C.blue, 0.25); g.fillRect(x1 + (cx - r) * f, y0 + (cy - r) * f, k * f, k * f);
        g.strokeStyle = C.blue; g.lineWidth = 2; g.strokeRect(x1 + (cx - r) * f, y0 + (cy - r) * f, k * f, k * f);
        g.strokeStyle = C.orange; g.strokeRect(x2 + cx * f - 1, y0 + cy * f - 1, f + 2, f + 2); g.lineWidth = 1;
      }
      const ks = Math.min(kw, s * 0.7) / k, mxK = Math.max(...K.map(Math.abs)) || 1;
      for (let j = 0; j < k; j++) for (let i = 0; i < k; i++) {
        const w = K[j * k + i]; g.fillStyle = w >= 0 ? AI.mix(C.bg2, C.green, Math.min(1, w / mxK)) : AI.mix(C.bg2, C.orange, Math.min(1, -w / mxK)); g.fillRect(x3 + i * ks, y0 + j * ks, ks, ks);
        if (ks > 20) text(g, AI.fmt(w, 3), x3 + (i + 0.5) * ks, y0 + (j + 0.6) * ks, C.fg, Math.min(10, ks / 3.2), 'center');
      }
      border(g, x3, y0, ks * k, ks * k, C.border);
      const type = q('#type').value, sep = type === 'box' || type === 'gauss' || type === 'motion';
      q('#out').innerHTML = `kernel <b>${k}×${k}</b>   Σ weights = <b class="o">${AI.fmt(K.reduce((a, b) => a + b, 0), 3)}</b>\nmults per pixel: <b class="b">${k * k}</b>${sep ? `  (separable: ${2 * k})` : ''}\nmults for the image: ${AI.fmt(k * k * NN / 1e6, 2)} M\n<span class="m">${centered ? 'grey 128 = zero response' : 'sum 1 keeps brightness'}</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* cannylab: Sobel -> Canny, stage by stage                                  */
  /* ======================================================================== */
  AI.labs.cannylab = function (root) {
    const NAMES = ['1 gray', '2 blur', '3 Gx', '4 Gy', '5 |G|', '6 angle', '7 NMS', '8 thresh', '9 hyster.'];
    const CAP = [
      '① The input: a 60×60 gray image. Every pixel is one brightness number.',
      '② Blur with a Gaussian (width σ). A single noisy pixel would otherwise create a fake edge. Bigger σ = fewer but smoother edges.',
      '③ Gx = Sobel-x ⊛ image. White where brightness rises to the right, black where it falls, mid-gray = flat. Only vertical edges respond.',
      '④ Gy = Sobel-y ⊛ image. The same for vertical changes: horizontal edges respond.',
      '⑤ Magnitude |G| = √(Gx² + Gy²): edge strength in every direction. Edges are visible but thick, and noisy-faint ones are mixed with real ones.',
      '⑥ Orientation θ = atan2(Gy, Gx): hue shows the direction of steepest brightening. The edge itself runs perpendicular to this arrow.',
      '⑦ Non-maximum suppression: keep a pixel only if |G| ≥ both neighbours along the gradient direction (rounded to 0°, 45°, 90°, 135°). Thick ridges become 1-pixel lines.',
      '⑧ Double threshold: ≥ high is a strong edge (white); between low and high is weak (gray); below low is thrown away.',
      '⑨ Hysteresis: a weak pixel survives only if it is chain-connected to a strong one. Watch strong edges grow outwards one ring (orange) at a time; unconnected weak specks die.'];
    const { view } = AI.labShell(root, 'Sobel → Canny: an edge detector in nine stages', 'slow cross-fades between stages; click a thumbnail to jump; hover the big image to probe a pixel',
      '<canvas style="height:420px;cursor:crosshair"></canvas><div class="caption" id="cap" style="min-height:3em"></div>',
      `${sel('scene', 'Image', SC5, 'shapes')}${sl('sg', 'blur σ (0 = none)', 0, 3, 1, 0.1)}${sl('lo', 'low threshold (% of max |G|)', 1, 60, 8, 1)}${sl('hi', 'high threshold (% of max |G|)', 2, 90, 25, 1)}
       ${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const n = 60, NN = n * n, imgs = NAMES.map(() => mkImg(n, n));
    let A, B, Gx, Gy, Mg, An, Bn, Nm, Cl, Gn, maxGen = 0, T = 0, last = -1, lastG = -1, DUR = [], START = [], TOT = 1, probe = null, rects = [], mainR = null, maxMag = 1;
    const at = (a, x, y) => a[clamp(y, 0, n - 1) * n + clamp(x, 0, n - 1)];
    const compute = () => {
      A = grayScene(q('#scene').value, n); const sg = +q('#sg').value;
      if (sg < 0.25) B = Float32Array.from(A); else {
        const r = Math.ceil(2.5 * sg), k = []; let s = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sg * sg)); k.push(v); s += v; }
        const t = new Float32Array(NN); B = new Float32Array(NN);
        for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * at(A, x + i, y); t[y * n + x] = a / s; }
        for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * at(t, x, y + i); B[y * n + x] = a / s; }
      }
      Gx = new Float32Array(NN); Gy = new Float32Array(NN); Mg = new Float32Array(NN); An = new Float32Array(NN); Bn = new Uint8Array(NN); maxMag = 1; let maxG = 1;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const p = (i, j) => at(B, x + i, y + j), i = y * n + x;
        Gx[i] = p(1, -1) + 2 * p(1, 0) + p(1, 1) - p(-1, -1) - 2 * p(-1, 0) - p(-1, 1);
        Gy[i] = p(-1, 1) + 2 * p(0, 1) + p(1, 1) - p(-1, -1) - 2 * p(0, -1) - p(1, -1);
        Mg[i] = Math.hypot(Gx[i], Gy[i]); An[i] = Math.atan2(Gy[i], Gx[i]);
        maxMag = Math.max(maxMag, Mg[i]); maxG = Math.max(maxG, Math.abs(Gx[i]), Math.abs(Gy[i]));
        const a = ((An[i] * 180) / Math.PI + 180) % 180; Bn[i] = a < 22.5 || a >= 157.5 ? 0 : a < 67.5 ? 45 : a < 112.5 ? 90 : 135;
      }
      Nm = new Float32Array(NN);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x, b = Bn[i], d = b === 0 ? [1, 0] : b === 45 ? [1, 1] : b === 90 ? [0, 1] : [-1, 1];
        const m1 = x + d[0] < 0 || y + d[1] < 0 || x + d[0] >= n || y + d[1] >= n ? 0 : Mg[(y + d[1]) * n + x + d[0]], m2 = x - d[0] < 0 || y - d[1] < 0 || x - d[0] >= n || y - d[1] >= n ? 0 : Mg[(y - d[1]) * n + x - d[0]];
        Nm[i] = Mg[i] >= m1 && Mg[i] >= m2 ? Mg[i] : 0;
      }
      let lo = (+q('#lo').value / 100) * maxMag, hi = (+q('#hi').value / 100) * maxMag; if (lo > hi) lo = hi;
      Cl = new Uint8Array(NN); Gn = new Int16Array(NN).fill(-1); const qu = [];
      for (let i = 0; i < NN; i++) { Cl[i] = Nm[i] >= hi ? 2 : Nm[i] >= lo && Nm[i] > 0 ? 1 : 0; if (Cl[i] === 2) { Gn[i] = 0; qu.push(i); } }
      maxGen = 0;
      for (let h = 0; h < qu.length; h++) { const i = qu[h], x = i % n, y = (i / n) | 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= n || Y >= n) continue; const j = Y * n + X; if (Cl[j] === 1 && Gn[j] < 0) { Gn[j] = Gn[i] + 1; maxGen = Math.max(maxGen, Gn[j]); qu.push(j); } } }
      imgs[0].gray(A); imgs[1].gray(B);
      imgs[2].gray(Gx.map((v) => 128 + (v / maxG) * 127)); imgs[3].gray(Gy.map((v) => 128 + (v / maxG) * 127));
      imgs[4].gray(Mg.map((v) => (v / maxMag) * 255));
      imgs[5].rgb((i) => { const hh = (An[i] / (2 * Math.PI) + 0.5) * 360, v = Math.sqrt(Mg[i] / maxMag), cc = v, xx = cc * (1 - Math.abs(((hh / 60) % 2) - 1)), m = 0, [r, g, b] = hh < 60 ? [cc, xx, 0] : hh < 120 ? [xx, cc, 0] : hh < 180 ? [0, cc, xx] : hh < 240 ? [0, xx, cc] : hh < 300 ? [xx, 0, cc] : [cc, 0, xx]; return [(r + m) * 255, (g + m) * 255, (b + m) * 255]; });
      imgs[6].gray(Nm.map((v) => Math.min(255, (v / maxMag) * 255 * 1.6)));
      imgs[7].gray(Cl.map((v) => (v === 2 ? 255 : v === 1 ? 110 : 0)));
      DUR = [3, 3.8, 3.8, 3.8, 3.8, 3.8, 4.2, 4.2, Math.max(5, (maxGen + 2) * 0.8)]; START = DUR.map((_, i) => DUR.slice(0, i).reduce((a, b) => a + b, 0)); TOT = START[8] + DUR[8] + 2;
      lastG = -2; paintHyst(0); S.dirty = true;
    };
    const paintHyst = (g) => { // g = generation currently shown
      const d = imgs[8].d;
      for (let i = 0; i < NN; i++) {
        let r = 0, gg = 0, b = 0;
        if (Cl[i] === 2) r = gg = b = 255;
        else if (Cl[i] === 1) { if (Gn[i] >= 0 && Gn[i] <= g) { if (Gn[i] === g && g <= maxGen) { r = 255; gg = 170; b = 40; } else r = gg = b = 255; } else { r = gg = b = g > maxGen ? 0 : 80; } }
        d[i * 4] = r; d[i * 4 + 1] = gg; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
      }
      imgs[8].flush();
    };
    ['#scene', '#sg', '#lo', '#hi'].forEach((s) => q(s).addEventListener('input', compute));
    const jump = (i) => { T = START[i] + 0.02; S.dirty = true; };
    P.onstep = (d) => { let i = START.findIndex((s, k) => T < s + DUR[k]); if (i < 0) i = 8; jump((i + d + 9) % 9); };
    P.onreset = () => { T = 0; P.set(true); };
    const evpos = (e) => { const r = c.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.cv.addEventListener('pointermove', (e) => { const [x, y] = evpos(e); if (mainR && x >= mainR.x && y >= mainR.y && x < mainR.x + mainR.s && y < mainR.y + mainR.s) probe = [Math.floor(((x - mainR.x) / mainR.s) * n), Math.floor(((y - mainR.y) / mainR.s) * n)]; else probe = null; S.dirty = true; });
    c.cv.addEventListener('pointerleave', () => { probe = null; S.dirty = true; });
    c.cv.addEventListener('click', (e) => { const [x, y] = evpos(e); rects.forEach((r, i) => { if (x >= r.x && x < r.x + r.s && y >= r.y && y < r.y + r.s) { P.set(false); jump(i); } }); });
    compute();
    AI.loop(root, (dt) => {
      if (P.playing) { T = (T + dt * P.sp()) % TOT; S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      let st = START.findIndex((s, k) => T < s + DUR[k]); if (st < 0) st = 8; const u = T - START[st];
      if (st !== last) { last = st; q('#cap').textContent = CAP[st]; }
      let gen = -1; if (st === 8) { gen = Math.min(maxGen + 1, Math.floor(u / 0.8)); if (gen !== lastG) { lastG = gen; paintHyst(gen); } }
      const s = Math.min(H - 30, W * 0.56), x0 = 10, y0 = 24; mainR = { x: x0, y: y0, s };
      label(g, NAMES[st], x0, 16, C);
      g.imageSmoothingEnabled = false;
      const fade = st > 0 ? smooth(u / 0.9) : 1;
      if (fade < 1) blit(g, imgs[st - 1], x0, y0, s, s);
      g.globalAlpha = fade; blit(g, imgs[st], x0, y0, s, s); g.globalAlpha = 1; border(g, x0, y0, s, s, C.border);
      const tx = x0 + s + 16, ts = Math.min(110, (W - tx - 10 - 2 * 8) / 3); rects = [];
      NAMES.forEach((nm, i) => { const X = tx + (i % 3) * (ts + 8), Y = y0 + Math.floor(i / 3) * (ts + 24); rects.push({ x: X, y: Y, s: ts }); blit(g, imgs[i], X, Y, ts, ts); border(g, X, Y, ts, ts, i === st ? C.blue : C.border, i === st ? 3 : 1); text(g, nm, X, Y + ts + 13, i === st ? C.fg : C.muted, 10); });
      if (probe) { g.strokeStyle = C.yellow; g.lineWidth = 2; g.strokeRect(x0 + (probe[0] * s) / n, y0 + (probe[1] * s) / n, s / n, s / n); g.lineWidth = 1; }
      const cnt = (v) => Cl.reduce((a, b) => a + (b === v ? 1 : 0), 0); let kept = 0; for (let i = 0; i < NN; i++) if (Cl[i] === 2 || (Cl[i] === 1 && Gn[i] >= 0)) kept++;
      let pr = ''; if (probe) { const i = probe[1] * n + probe[0]; pr = `\n<b>pixel (${probe[0]}, ${probe[1]})</b>\nGx = ${AI.fmt(Gx[i], 1)}   Gy = ${AI.fmt(Gy[i], 1)}\n|G| = <b class="o">${AI.fmt(Mg[i], 1)}</b>   θ = <b class="b">${AI.fmt((An[i] * 180) / Math.PI, 1)}°</b> → ${Bn[i]}°\nNMS keeps: ${Nm[i] > 0 ? 'yes' : 'no'}`; }
      q('#out').innerHTML = `max |G| = ${AI.fmt(maxMag, 0)}\nstrong ${cnt(2)}   weak ${cnt(1)}\nfinal edge pixels <b class="g">${kept}</b>${pr}`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* morpho: structuring element sliding over a binary image                   */
  /* ======================================================================== */
  AI.labs.morpho = function (root) {
    const W = 14, H = 10;
    const PICS = {
      'blob + noise': ['..............', '.....#........', '..#####...#...', '.#######......', '.########.....', '.#######..##..', '..#####...##..', '..............', '.#............', '..............'],
      'two touching blobs': ['..............', '.####....####.', '.#####..#####.', '.######.#####.', '.#############', '.####..#####..', '.####....####.', '..............', '.....#...#....', '..............'],
      'broken ring': ['..............', '...########...', '..##......##..', '..#........#..', '..#...........', '..#........#..', '..##......##..', '...###.####...', '..............', '..............'],
      'letters': ['..............', '.#...#.####...', '.#...#.#...#..', '.#####.####...', '.#...#.#...#..', '.#...#.####...', '..............', '..#.#..#.#....', '..............', '..............'] };
    const SES = { 'square 3×3': ['00000', '01110', '01110', '01110', '00000'], 'cross (plus)': ['00000', '00100', '01110', '00100', '00000'], 'horizontal 1×3': ['00000', '00000', '01110', '00000', '00000'], 'vertical 3×1': ['00000', '00100', '00100', '00100', '00000'], 'disc 5×5': ['00100', '01110', '11111', '01110', '00100'] };
    const OPS = { erode: 'erosion', dilate: 'dilation', open: 'opening (erode → dilate)', close: 'closing (dilate → erode)', comp: 'connected components' };
    const { view } = AI.labShell(root, 'Morphology: a shape probing a binary image', 'click cells of the input to draw; click the 5×5 shape grid to edit the structuring element',
      '<canvas style="height:310px;cursor:pointer"></canvas><div class="caption" id="cap" style="min-height:3em"></div>',
      `${sel('pic', 'Image', Object.fromEntries(Object.keys(PICS).map((k) => [k, k])), 'blob + noise')}${sel('op', 'Operation', OPS, 'erode')}
       <div>Structuring element (click to edit)<div id="se" style="display:grid;grid-template-columns:repeat(5,26px);gap:3px;margin-top:4px"></div>${chipRow('sepre')}</div>
       ${sel('conn', 'Components: connectivity', { 4: '4-connected', 8: '8-connected' }, 8)}
       ${sl('rate', 'positions per second', 1, 60, 8, 1)}${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const bits = new Uint8Array(W * H), se = new Uint8Array(25), seBtn = [];
    let phases = [], ph = 0, pos = 0, hold = 0, labels = null, order = [], nComp = 0, rects = null;
    const loadPic = () => { const p = PICS[q('#pic').value]; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) bits[y * W + x] = p[y][x] === '#' ? 1 : 0; compute(); };
    for (let i = 0; i < 25; i++) { const b = document.createElement('button'); b.style.cssText = 'width:26px;height:26px;border:1px solid var(--border);border-radius:4px;cursor:pointer;padding:0'; b.onclick = () => { se[i] ^= 1; paintSE(); compute(); }; q('#se').appendChild(b); seBtn.push(b); }
    const paintSE = () => seBtn.forEach((b, i) => { b.style.background = se[i] ? 'var(--blue)' : 'var(--bg2)'; b.style.outline = i === 12 ? '2px solid var(--orange)' : ''; });
    Object.entries(SES).forEach(([nm, rows]) => chip(q('#sepre'), nm, () => { rows.join('').split('').forEach((v, i) => (se[i] = +v)); paintSE(); compute(); }));
    SES['square 3×3'].join('').split('').forEach((v, i) => (se[i] = +v)); paintSE();
    const seList = () => { const l = []; for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) if (se[j * 5 + i]) l.push([i - 2, j - 2]); return l; };
    const apply = (src, kind) => { // erode / dilate
      const l = seList(), o = new Uint8Array(W * H);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let hit = 0, fit = 1;
        for (const [dx, dy] of l) { const X = x + dx, Y = y + dy, v = X < 0 || Y < 0 || X >= W || Y >= H ? 0 : src[Y * W + X]; if (v) hit = 1; else fit = 0; }
        o[y * W + x] = kind === 'erode' ? (l.length ? fit : 0) : hit;
      }
      return o;
    };
    function compute() {
      const op = q('#op').value; pos = 0; ph = 0; hold = 0; S.dirty = true;
      if (op === 'comp') {
        const con = +q('#conn').value; labels = new Int16Array(W * H); order = []; nComp = 0;
        for (let i = 0; i < W * H; i++) if (bits[i] && !labels[i]) {
          nComp++; const stk = [i]; labels[i] = nComp;
          while (stk.length) { const j = stk.pop(), x = j % W, y = (j / W) | 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { if (!dx && !dy) continue; if (con === 4 && dx && dy) continue; const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= H) continue; const k = Y * W + X; if (bits[k] && !labels[k]) { labels[k] = nComp; stk.push(k); } } }
          order.push(i);
        }
        phases = []; return;
      }
      if (op === 'erode') phases = [{ kind: 'erode', src: bits }];
      else if (op === 'dilate') phases = [{ kind: 'dilate', src: bits }];
      else if (op === 'open') phases = [{ kind: 'erode', src: bits }];
      else phases = [{ kind: 'dilate', src: bits }];
      phases[0].out = apply(phases[0].src, phases[0].kind);
      if (op === 'open' || op === 'close') phases.push({ kind: op === 'open' ? 'dilate' : 'erode', src: phases[0].out, out: apply(phases[0].out, op === 'open' ? 'dilate' : 'erode') });
    }
    ['#pic'].forEach((s) => q(s).addEventListener('input', loadPic)); q('#op').addEventListener('input', compute); q('#conn').addEventListener('input', compute);
    P.onstep = (d) => { pos = clamp(pos + d, 0, W * H); S.dirty = true; };
    P.onreset = () => { compute(); P.set(true); };
    c.cv.addEventListener('click', (e) => { if (!rects) return; const r = c.cv.getBoundingClientRect(), x = e.clientX - r.left - rects.x1, y = e.clientY - r.top - rects.y; const cx = Math.floor(x / rects.cs), cy = Math.floor(y / rects.cs); if (cx >= 0 && cy >= 0 && cx < W && cy < H) { bits[cy * W + cx] ^= 1; compute(); } });
    loadPic();
    AI.loop(root, (dt) => {
      const op = q('#op').value, total = W * H;
      if (P.playing) {
        if (op === 'comp') { if (pos >= total) { hold += dt; if (hold > 2.5) { pos = 0; hold = 0; } } else pos = Math.min(total, pos + dt * +q('#rate').value * P.sp() * 3); }
        else if (pos >= total) { hold += dt; if (hold > 1.5) { hold = 0; if (ph < phases.length - 1) { ph++; pos = 0; } else { ph = 0; pos = 0; } } } else pos = Math.min(total, pos + dt * +q('#rate').value * P.sp());
        S.dirty = true;
      }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const cs = Math.min((w - 50) / (2 * W), (h - 52) / H), gx1 = 12, gx2 = gx1 + W * cs + 26, y0 = 26; rects = { x1: gx1, y: y0, cs };
      const ph0 = op === 'comp' ? null : phases[Math.min(ph, phases.length - 1)], cur = Math.min(total - 1, Math.floor(pos)), cx = cur % W, cy = (cur / W) | 0;
      const src = ph0 ? ph0.src : bits, l = seList();
      label(g, op === 'comp' ? 'input (click to toggle)' : ph === 0 ? 'input (click cells to toggle)' : `after ${phases[0].kind === 'erode' ? 'erosion' : 'dilation'} = input to ${ph0.kind}`, gx1, 16, C);
      label(g, op === 'comp' ? 'labelled components' : `output of ${ph0.kind === 'erode' ? 'erosion' : 'dilation'}${phases.length > 1 ? ` (phase ${ph + 1}/2)` : ''}`, gx2, 16, C);
      const pal = [C.blue, C.orange, C.green, C.purple, C.pink, C.teal, C.yellow, C.red];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x; g.fillStyle = src[i] ? C.fg : C.bg2; g.globalAlpha = src[i] ? 0.85 : 1; g.fillRect(gx1 + x * cs, y0 + y * cs, cs - 1, cs - 1); g.globalAlpha = 1;
        const done = i < pos; let col = C.bg2;
        if (op === 'comp') { if (bits[i] && labels[i]) { const seedIdx = order[labels[i] - 1]; if (seedIdx <= pos) col = pal[(labels[i] - 1) % pal.length]; else col = C.fg; } if (i === cur && pos < total) col = col; g.fillStyle = bits[i] ? col : C.bg2; g.globalAlpha = bits[i] && col === C.fg ? 0.4 : 1; }
        else { g.fillStyle = done ? (ph0.out[i] ? C.green : C.bg2) : C.bg2; }
        g.fillRect(gx2 + x * cs, y0 + y * cs, cs - 1, cs - 1); g.globalAlpha = 1;
      }
      border(g, gx1, y0, W * cs, H * cs, C.border); border(g, gx2, y0, W * cs, H * cs, C.border);
      let cap = '';
      if (op === 'comp') {
        const seen = order.filter((s) => s <= pos).length;
        g.strokeStyle = C.yellow; g.lineWidth = 2; g.strokeRect(gx1 + cx * cs, y0 + cy * cs, cs, cs); g.lineWidth = 1;
        cap = `Scan left→right, top→bottom. When the scan meets a foreground pixel with no label, start a new label and flood-fill every connected pixel (${q('#conn').value}-connectivity). Components found so far: ${seen}.`;
        q('#out').innerHTML = `scan position (${cx}, ${cy})\ncomponents so far <b class="o">${seen}</b> of ${nComp}\n<span class="m">small specks become their own components — filter them by size</span>`;
      } else {
        // structuring element overlay
        for (const [dx, dy] of l) { const X = cx + dx, Y = cy + dy; if (X < 0 || Y < 0 || X >= W || Y >= H) { g.strokeStyle = C.red; g.setLineDash([2, 2]); g.strokeRect(gx1 + X * cs + 1, y0 + Y * cs + 1, cs - 3, cs - 3); g.setLineDash([]); continue; } g.fillStyle = src[Y * W + X] ? hexA(C.green, 0.6) : hexA(C.red, 0.55); g.fillRect(gx1 + X * cs, y0 + Y * cs, cs - 1, cs - 1); }
        g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(gx1 + cx * cs, y0 + cy * cs, cs - 1, cs - 1); g.lineWidth = 1;
        if (pos < total) { g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(gx2 + cx * cs, y0 + cy * cs, cs - 1, cs - 1); g.lineWidth = 1; }
        let ones = 0; for (const [dx, dy] of l) { const X = cx + dx, Y = cy + dy; if (X >= 0 && Y >= 0 && X < W && Y < H && src[Y * W + X]) ones++; }
        const res = ph0.kind === 'erode' ? (ones === l.length && l.length ? 1 : 0) : (ones > 0 ? 1 : 0);
        cap = ph0.kind === 'erode' ? `Erosion: put the shape's centre (orange) on pixel (${cx}, ${cy}). The shape fits only if all ${l.length} covered pixels are foreground (green). Covered foreground: ${ones}/${l.length} → output ${res}. Red = background under the shape.` : `Dilation: put the shape's centre (orange) on pixel (${cx}, ${cy}). The output is 1 if the shape touches at least one foreground pixel (green). Covered foreground: ${ones}/${l.length} → output ${res}.`;
        q('#out').innerHTML = `centre (${cx}, ${cy})\ncovered foreground <b class="o">${ones}/${l.length}</b>\noutput <b class="g">${res}</b>\n<span class="m">foreground: ${src.reduce((a, b) => a + b, 0)} → ${ph0.out.reduce((a, b) => a + b, 0)} pixels</span>`;
      }
      q('#cap').textContent = cap;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* interp: nearest / bilinear / bicubic upscaling with the sample point      */
  /* ======================================================================== */
  AI.labs.interp = function (root) {
    const { view } = AI.labShell(root, 'Interpolation: inventing values between pixels', 'a tiny image is enlarged; follow the sample point and the pixels that vote',
      '<canvas style="height:400px"></canvas><div class="caption" id="cap" style="min-height:3em"></div>',
      `${sel('scene', 'Image', { house: 'house', shapes: 'colour shapes', digit: 'handwritten 7', edges: 'edges & stripes' }, 'digit')}
       ${sel('ns', 'Source size', { 6: '6×6', 8: '8×8', 10: '10×10' }, 6)}${sl('fac', 'enlarge by', 2, 8, 3, 1)}
       ${sel('mode', 'Method', { nearest: 'nearest neighbour', bilinear: 'bilinear (4 pixels)', bicubic: 'bicubic (16 pixels)' }, 'bilinear')}
       ${sl('rate', 'output pixels per second', 2, 120, 20, 1)}${PLAY(1)}<div class="row"><button class="btn" id="all">Show all</button></div><div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    let src, N, F, O, outv, pos = 0, hold = 0, imO = null;
    const cr = (t) => { const a = -0.5, r = []; for (const d of [1 + t, t, 1 - t, 2 - t]) { const x = Math.abs(d); r.push(x <= 1 ? (a + 2) * x ** 3 - (a + 3) * x * x + 1 : a * x ** 3 - 5 * a * x * x + 8 * a * x - 4 * a); } return r; };
    const px = (x, y) => src[clamp(y, 0, N - 1) * N + clamp(x, 0, N - 1)];
    const sample = (xs, ys, mode) => {
      if (mode === 'nearest') return px(Math.floor(xs + 0.5), Math.floor(ys + 0.5));
      const x0 = Math.floor(xs), y0 = Math.floor(ys), fx = xs - x0, fy = ys - y0;
      if (mode === 'bilinear') return (1 - fy) * ((1 - fx) * px(x0, y0) + fx * px(x0 + 1, y0)) + fy * ((1 - fx) * px(x0, y0 + 1) + fx * px(x0 + 1, y0 + 1));
      const wx = cr(fx), wy = cr(fy); let s = 0; for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += wx[i] * wy[j] * px(x0 - 1 + i, y0 - 1 + j); return clamp(s, 0, 255);
    };
    const compute = () => {
      N = +q('#ns').value; F = +q('#fac').value; O = N * F; src = grayScene(q('#scene').value, N).map(Math.round);
      outv = new Float32Array(O * O); const mode = q('#mode').value;
      for (let y = 0; y < O; y++) for (let x = 0; x < O; x++) outv[y * O + x] = sample((x + 0.5) / F - 0.5, (y + 0.5) / F - 0.5, mode);
      imO = mkImg(O, O).gray(outv.map(Math.round)); pos = 0; hold = 0; S.dirty = true;
    };
    ['#scene', '#ns', '#fac', '#mode'].forEach((s) => q(s).addEventListener('input', compute));
    q('#all').onclick = () => { pos = O * O; S.dirty = true; };
    P.onstep = (d) => { pos = clamp(Math.floor(pos) + d, 0, O * O); S.dirty = true; };
    P.onreset = () => { pos = 0; hold = 0; P.set(true); };
    compute();
    AI.loop(root, (dt) => {
      if (P.playing) { if (pos >= O * O) { hold += dt; if (hold > 2.5) { pos = 0; hold = 0; } } else pos = Math.min(O * O, pos + dt * +q('#rate').value * P.sp()); S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const s = Math.min((W - 50) / 2, H - 30), x0 = 12, x1 = x0 + s + 26, y0 = 24, cs = s / N, co = s / O, mode = q('#mode').value;
      label(g, `source ${N}×${N} (numbers = brightness)`, x0, 16, C); label(g, `output ${O}×${O}`, x1, 16, C);
      const cur = Math.min(O * O - 1, Math.floor(pos)), ox = cur % O, oy = (cur / O) | 0, xs = (ox + 0.5) / F - 0.5, ys = (oy + 0.5) / F - 0.5;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const v = src[y * N + x]; g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x0 + x * cs, y0 + y * cs, cs, cs); if (cs >= 26) text(g, v, x0 + (x + 0.5) * cs, y0 + (y + 0.5) * cs + 4, v > 128 ? '#000' : '#fff', Math.min(12, cs / 3), 'center'); }
      g.strokeStyle = 'rgba(128,128,128,.5)'; for (let i = 0; i <= N; i++) { g.beginPath(); g.moveTo(x0 + i * cs, y0); g.lineTo(x0 + i * cs, y0 + s); g.moveTo(x0, y0 + i * cs); g.lineTo(x0 + s, y0 + i * cs); g.stroke(); }
      // support
      const x0i = Math.floor(xs), y0i = Math.floor(ys), fx = xs - x0i, fy = ys - y0i; let wts = [];
      if (mode === 'nearest') wts = [[Math.floor(xs + 0.5), Math.floor(ys + 0.5), 1]];
      else if (mode === 'bilinear') wts = [[x0i, y0i, (1 - fx) * (1 - fy)], [x0i + 1, y0i, fx * (1 - fy)], [x0i, y0i + 1, (1 - fx) * fy], [x0i + 1, y0i + 1, fx * fy]];
      else { const wx = cr(fx), wy = cr(fy); for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) wts.push([x0i - 1 + i, y0i - 1 + j, wx[i] * wy[j]]); }
      for (const [X, Y, w] of wts) { if (X < 0 || Y < 0 || X >= N || Y >= N) continue; g.fillStyle = w >= 0 ? hexA(C.green, Math.min(0.75, 0.2 + Math.abs(w) * 1.6)) : hexA(C.red, 0.6); g.fillRect(x0 + X * cs, y0 + Y * cs, cs, cs); g.strokeStyle = C.fg; g.strokeRect(x0 + X * cs + 0.5, y0 + Y * cs + 0.5, cs - 1, cs - 1); if (mode !== 'bicubic' || cs > 36) text(g, w.toFixed(2), x0 + (X + 0.5) * cs, y0 + (Y + 0.5) * cs + (cs >= 26 ? 17 : 4), C.fg, Math.min(11, cs / 3.4), 'center'); }
      const sx = x0 + (xs + 0.5) * cs, sy = y0 + (ys + 0.5) * cs;
      // output
      g.fillStyle = C.bg2; g.fillRect(x1, y0, s, s);
      const full = Math.floor(pos / O), part = pos - full * O;
      g.save(); g.beginPath(); g.rect(x1, y0, s, full * co); g.rect(x1, y0 + full * co, part * co, co); g.clip(); blit(g, imO, x1, y0, s, s); g.restore();
      border(g, x0, y0, s, s, C.border); border(g, x1, y0, s, s, C.border);
      if (pos < O * O) {
        const cx = x1 + (ox + 0.5) * co, cy = y0 + (oy + 0.5) * co;
        g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(x1 + ox * co, y0 + oy * co, co, co); g.lineWidth = 1;
        g.strokeStyle = C.orange; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(cx, cy); g.lineTo(sx, sy); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.orange; g.beginPath(); g.arc(sx, sy, 5, 0, 7); g.fill(); g.strokeStyle = C.fg; g.stroke();
      }
      let f = '';
      if (mode === 'nearest') f = `nearest pixel (${wts[0][0]}, ${wts[0][1]}) → value ${px(wts[0][0], wts[0][1])}`;
      else if (mode === 'bilinear') f = `fx=${fx.toFixed(2)} fy=${fy.toFixed(2)}\ntop    = ${px(x0i, y0i)}+${fx.toFixed(2)}(${px(x0i + 1, y0i)}−${px(x0i, y0i)}) = ${((1 - fx) * px(x0i, y0i) + fx * px(x0i + 1, y0i)).toFixed(1)}\nbottom = ${px(x0i, y0i + 1)}+${fx.toFixed(2)}(${px(x0i + 1, y0i + 1)}−${px(x0i, y0i + 1)}) = ${((1 - fx) * px(x0i, y0i + 1) + fx * px(x0i + 1, y0i + 1)).toFixed(1)}`;
      else f = `16 taps, weights from the cubic\nkernel; they sum to 1,\nsome are negative (red)`;
      q('#out').innerHTML = `output pixel (${ox}, ${oy})\nx_s = (${ox}+0.5)/${F} − 0.5 = <b class="b">${xs.toFixed(3)}</b>\ny_s = (${oy}+0.5)/${F} − 0.5 = <b class="b">${ys.toFixed(3)}</b>\n${f}\nvalue = <b class="o">${AI.fmt(outv[cur], 1)}</b>`;
      q('#cap').textContent = mode === 'nearest' ? 'Each output pixel centre is mapped back into the source (orange dot); the nearest source pixel simply copies its value — blocky but exact.' : mode === 'bilinear' ? 'The 4 surrounding source pixels vote, weighted by how close the orange dot is to each (bigger weight = nearer). Result: smooth, but slightly blurry.' : 'A 4×4 neighbourhood votes with cubic weights (some negative). Sharper than bilinear; it can overshoot near edges (ringing).';
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* alias: downscaling with and without pre-filtering                          */
  /* ======================================================================== */
  AI.labs.alias = function (root) {
    const { view } = AI.labShell(root, 'Aliasing: why you must blur before you shrink', 'moving stripes are shrunk by picking pixels vs by averaging first',
      '<canvas style="height:330px"></canvas><div class="caption">Left: the 120×120 original (dots = the pixels point-sampling keeps). Middle: keep one pixel in f. Right: average every f×f block first. Both are shown at the same size.</div>',
      `${sel('pat', 'Pattern', { stripes: 'moving stripes', zone: 'zone plate (frequency grows outward)', checker: 'checkerboard', house: 'house' }, 'stripes')}
       ${sl('f', 'shrink factor f', 1, 10, 4, 1)}${sl('w', 'stripe frequency ω (cycles per pixel)', 0.02, 0.5, 0.2, 0.005)}${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const n = 120, srcA = new Float32Array(n * n), house = grayScene('house', n), imS = mkImg(n, n), imP = mkImg(n, n), imF = mkImg(n, n), pa = new Float32Array(n * n), fa = new Float32Array(n * n);
    let t = 0;
    const pattern = (kind, w, shift) => {
      const cx = n / 2;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        let v;
        if (kind === 'stripes') v = 0.5 + 0.5 * Math.sin(2 * Math.PI * w * (x - shift));
        else if (kind === 'zone') v = 0.5 + 0.5 * Math.cos(Math.PI * 0.0059 * ((x - cx) ** 2 + (y - cx) ** 2));
        else if (kind === 'checker') { const p = Math.max(1, Math.round(1 / (2 * w))); v = (Math.floor((x - shift) / p) + Math.floor(y / p)) & 1 ? 0.95 : 0.05; }
        else v = house[y * n + x] / 255;
        srcA[y * n + x] = v * 255;
      }
    };
    P.onreset = () => { t = 0; P.set(true); S.dirty = true; };
    P.onstep = (d) => { t += d * 0.25; S.dirty = true; };
    AI.loop(root, (dt) => {
      const kind = q('#pat').value;
      if (P.playing && kind !== 'house' && kind !== 'zone') { t += dt * P.sp(); S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, f = +q('#f').value, w = +q('#w').value; g.clearRect(0, 0, W, H);
      pattern(kind, w, t * 5); imS.gray(srcA);
      const m = Math.floor(n / f), ds = Math.min((W - 4 * 12) / 3, H - 40);
      const pim = mkImg(m, m), fim = mkImg(m, m), pv = new Float32Array(m * m), fv = new Float32Array(m * m);
      for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { pv[y * m + x] = srcA[(y * f + (f >> 1)) * n + x * f + (f >> 1)]; let s = 0; for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) s += srcA[(y * f + j) * n + x * f + i]; fv[y * m + x] = s / (f * f); }
      pim.gray(pv); fim.gray(fv);
      const xs = [12, 24 + ds, 36 + 2 * ds], y0 = 24;
      label(g, 'original 120×120', xs[0], 16, C); label(g, `point sample → ${m}×${m}`, xs[1], 16, C); label(g, `average then sample → ${m}×${m}`, xs[2], 16, C);
      blit(g, imS, xs[0], y0, ds, ds); blit(g, pim, xs[1], y0, ds, ds); blit(g, fim, xs[2], y0, ds, ds);
      xs.forEach((x) => border(g, x, y0, ds, ds, C.border));
      if (f >= 3) { g.fillStyle = C.red; for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { g.fillRect(xs[0] + ((x * f + (f >> 1) + 0.5) / n) * ds - 1.5, y0 + ((y * f + (f >> 1) + 0.5) / n) * ds - 1.5, 3, 3); } }
      const wo = w * f, alias = Math.abs(wo - Math.round(wo));
      q('#out').innerHTML = kind === 'stripes' || kind === 'checker' ? `after ÷${f}: Nyquist limit = 0.5 cycle/pixel\nstripes now ${AI.fmt(wo, 2)} cycles per output pixel\n${wo > 0.5 ? `<b class="r">above Nyquist → aliased to ${AI.fmt(alias, 2)} cycles/px</b>\n(coarse false pattern, may move backwards)` : '<b class="g">below Nyquist → shown truthfully</b>'}` : `shrink ÷${f}: only frequencies below ${AI.fmt(0.5 / f, 3)} cycles/px survive\n<span class="m">anything faster folds into fake coarse patterns</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* pyramid: blur -> drop every other row/column, repeated                    */
  /* ======================================================================== */
  AI.labs.pyramid = function (root) {
    const { view } = AI.labShell(root, 'Gaussian pyramid', 'blur, then throw away every second row and column — repeat',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>',
      `${sel('scene', 'Image', SC5, 'house')}${chk('lap', 'show Laplacian levels (what each level lost) after building', false)}${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const sizes = [120, 60, 30, 15], DUR = 5.5, TOT = DUR * 3 + 3;
    let lv = [], bl = [], ims = [], imsB = [], imsL = [], T = 0;
    const blur = (a, n) => { const k = [1, 4, 6, 4, 1], t = new Float32Array(n * n), o = new Float32Array(n * n), p = (arr, x, y) => arr[clamp(y, 0, n - 1) * n + clamp(x, 0, n - 1)]; for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += k[i + 2] * p(a, x + i, y); t[y * n + x] = s / 16; } for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let s = 0; for (let i = -2; i <= 2; i++) s += k[i + 2] * p(t, x, y + i); o[y * n + x] = s / 16; } return o; };
    const compute = () => {
      lv = [grayScene(q('#scene').value, 120)]; bl = [];
      for (let l = 0; l < 3; l++) { const n = sizes[l], b = blur(lv[l], n), m = sizes[l + 1], d = new Float32Array(m * m); bl.push(b); for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) d[y * m + x] = b[2 * y * n + 2 * x]; lv.push(d); }
      ims = lv.map((a, i) => mkImg(sizes[i], sizes[i]).gray(a)); imsB = bl.map((a, i) => mkImg(sizes[i], sizes[i]).gray(a));
      imsL = [0, 1, 2].map((l) => { const n = sizes[l], m = sizes[l + 1], up = new Float32Array(n * n); for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const fx = (x + 0.5) / 2 - 0.5, fy = (y + 0.5) / 2 - 0.5, x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, p = (X, Y) => lv[l + 1][clamp(Y, 0, m - 1) * m + clamp(X, 0, m - 1)]; up[y * n + x] = (1 - ay) * ((1 - ax) * p(x0, y0) + ax * p(x0 + 1, y0)) + ay * ((1 - ax) * p(x0, y0 + 1) + ax * p(x0 + 1, y0 + 1)); } return mkImg(n, n).gray(lv[l].map((v, i) => clamp(128 + 2 * (v - up[i]), 0, 255))); });
      S.dirty = true;
    };
    q('#scene').addEventListener('input', compute); q('#lap').addEventListener('input', () => (S.dirty = true));
    P.onstep = (d) => { T = ((Math.floor(T / DUR) + d + 4) % 4) * DUR + 0.01; S.dirty = true; };
    P.onreset = () => { T = 0; P.set(true); };
    compute();
    AI.loop(root, (dt) => {
      if (P.playing) { T = (T + dt * P.sp()) % TOT; S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const L = Math.min(2, Math.floor(T / DUR)), done = T >= DUR * 3, u = done ? 1 : (T - L * DUR) / DUR, sc = Math.min(1.7, (W - 70) / 225, (H - 40) / 120), y1 = H - 24, gap = 16;
      const xo = [12]; for (let i = 1; i < 4; i++) xo.push(xo[i - 1] + sizes[i - 1] * sc + gap);
      const built = done ? 3 : L, showL = q('#lap').checked && done;
      for (let l = 0; l <= 3; l++) {
        if (l > built + (done ? 0 : 0) && !(l === built + 1)) continue;
        const sz = sizes[l] * sc, x = xo[l], y = y1 - sz; if (l === built + 1 && !done) continue;
        g.globalAlpha = 1; blit(g, showL && l < 3 ? imsL[l] : ims[l], x, y, sz, sz); border(g, x, y, sz, sz, C.border);
        text(g, `level ${l}: ${sizes[l]}×${sizes[l]}`, x, y - 6, l === built && !done ? C.fg : C.muted, 11);
      }
      let cap;
      if (!done) {
        // active level L: blur crossfade (u<0.45), then keep-every-2nd highlight & shrink into level L+1 slot
        const sz = sizes[L] * sc, x = xo[L], y = y1 - sz;
        if (u < 0.45) { const a = smooth(u / 0.4); g.globalAlpha = a; blit(g, imsB[L], x, y, sz, sz); g.globalAlpha = 1; cap = `Step ${L + 1}a: blur level ${L} with the 5-tap Gaussian [1 4 6 4 1]/16 (fading in) — removes detail finer than the next level can store.`; }
        else {
          blit(g, imsB[L], x, y, sz, sz); const k = (u - 0.45) / 0.55, kk = smooth(k * 1.3 - 0.15), sz2 = sizes[L + 1] * sc, x2 = xo[L + 1], y2 = y1 - sz2;
          g.fillStyle = hexA(C.orange, 0.9); const dot = Math.max(1.5, sc * 1.2); if (sizes[L] <= 60) for (let j = 0; j < sizes[L]; j += 2) for (let i = 0; i < sizes[L]; i += 2) g.fillRect(x + (i + 0.5) * sc - dot / 2, y + (j + 0.5) * sc - dot / 2, dot, dot);
          g.globalAlpha = kk; blit(g, ims[L + 1], lerp(x, x2, kk), lerp(y, y2, kk), lerp(sz, sz2, kk), lerp(sz, sz2, kk)); g.globalAlpha = 1; border(g, x2, y2, sz2, sz2, C.border);
          cap = `Step ${L + 1}b: keep every 2nd row and column (orange dots = survivors) → ${sizes[L + 1]}×${sizes[L + 1]}, a quarter of the pixels.`;
        }
      } else cap = showL ? 'Laplacian levels: level − (blurred, enlarged next level) = the detail that level holds that the coarser one cannot. Mid-gray = zero.' : 'Done: each level has a quarter of the pixels of the one below. Objects that are big in level 0 are small in level 3 — a multi-scale view of the same scene.';
      q('#cap').textContent = cap;
      q('#out').innerHTML = `pixels per level: ${sizes.map((s) => s * s).join(' → ')}\ntotal ≈ <b class="o">${AI.fmt(sizes.reduce((a, s) => a + s * s, 0) / (120 * 120), 3)}×</b> the original\n<span class="m">(infinite pyramid → 4/3)</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* warp: editable 3x3 matrix, inverse mapping, grid, morph                   */
  /* ======================================================================== */
  AI.labs.warp = function (root) {
    const PRE = { identity: [0, 1, 1, 0, 0, 0, 0, 0], 'rotate 30°': [30, 1, 1, 0, 0, 0, 0, 0], 'scale ×1.5': [0, 1.5, 1.5, 0, 0, 0, 0, 0], 'shear': [0, 1, 1, 0.5, 0, 0, 0, 0], 'rotate + scale + shift': [20, 0.8, 0.8, 0, 0.3, -0.2, 0, 0], 'flip horizontal': [0, -1, 1, 0, 0, 0, 0, 0], 'perspective (tilt)': [0, 1, 1, 0, 0, 0, 0.45, 0.15] };
    const { view } = AI.labShell(root, 'Geometric transforms: warping with a 3×3 matrix', 'edit the matrix or use the sliders; the grid shows where every source point lands',
      '<canvas style="height:390px"></canvas><div class="caption" id="cap" style="min-height:3em"></div>',
      `<div>Presets${chipRow('pre')}</div>
       <div>Matrix M (editable)<div id="mat" style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px"></div></div>
       ${sl('rot', 'rotation (degrees)', -180, 180, 0, 1)}${sl('sx', 'scale x', -2, 2.5, 1, 0.05)}${sl('sy', 'scale y', -2, 2.5, 1, 0.05)}${sl('sh', 'shear', -1, 1, 0, 0.05)}
       ${sl('tx', 'shift x', -1, 1, 0, 0.02)}${sl('ty', 'shift y', -1, 1, 0, 0.02)}${sl('p1', 'perspective h31', -0.6, 0.6, 0, 0.01)}${sl('p2', 'perspective h32', -0.6, 0.6, 0, 0.01)}
       ${chk('morph', 'animate identity → M', true)}${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c); vals(root);
    const R = 150, SRC = CV.scene('house'), outIm = mkImg(R, R), M = [1, 0, 0, 0, 1, 0, 0, 0, 1], inp = [];
    let u = 0, tp = 0;
    for (let i = 0; i < 9; i++) { const e = document.createElement('input'); e.type = 'number'; e.step = 0.05; e.style.textAlign = 'center'; if (i > 5) e.style.borderColor = 'var(--orange)'; e.oninput = () => { M[i] = +e.value || 0; S.dirty = true; }; q('#mat').appendChild(e); inp.push(e); }
    const fromSliders = () => {
      const th = (+q('#rot').value * Math.PI) / 180, sx = +q('#sx').value, sy = +q('#sy').value, sh = +q('#sh').value, co = Math.cos(th), si = Math.sin(th);
      // A = R * Shear * S
      const a = co * sx, b = (co * sh - si) * sy, d = si * sx, e = (si * sh + co) * sy;
      M.splice(0, 9, a, b, +q('#tx').value, d, e, +q('#ty').value, +q('#p1').value, +q('#p2').value, 1); inp.forEach((e2, i) => (e2.value = +M[i].toFixed(3)));
    };
    ['#rot', '#sx', '#sy', '#sh', '#tx', '#ty', '#p1', '#p2'].forEach((s) => q(s).addEventListener('input', fromSliders));
    Object.entries(PRE).forEach(([n, v]) => chip(q('#pre'), n, () => { ['rot', 'sx', 'sy', 'sh', 'tx', 'ty', 'p1', 'p2'].forEach((id, i) => (q('#' + id).value = v[i])); vals(root)(); fromSliders(); S.dirty = true; }));
    fromSliders(); q('#rot').value = 25; q('#sx').value = 0.9; q('#sy').value = 0.9; vals(root)(); fromSliders();
    P.onstep = () => { u = u >= 0.5 && u < 0.75 ? 0 : 0.5; S.dirty = true; };
    P.onreset = () => { u = 0; tp = 0; P.set(true); };
    const inv3 = (m) => { const [a, b, c2, d, e, f, g, h, i] = m, A = e * i - f * h, B = -(d * i - f * g), Cc = d * h - e * g, det = a * A + b * B + c2 * Cc; if (Math.abs(det) < 1e-6) return null; return [A / det, -(b * i - c2 * h) / det, (b * f - c2 * e) / det, B / det, (a * i - c2 * g) / det, -(a * f - c2 * d) / det, Cc / det, -(a * h - b * g) / det, (a * e - b * d) / det]; };
    AI.loop(root, (dt) => {
      if (P.playing) { if (q('#morph').checked) u += dt * 0.1 * P.sp(); tp += dt * 0.35 * P.sp(); S.dirty = true; }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const m = q('#morph').checked ? pingpong(u) : 1, Mt = M.map((v, i) => lerp(i % 4 === 0 ? 1 : 0, v, m)), iv = inv3(Mt), V = 1.5;
      const s = Math.min((W - 36) / 2, H - 30), x0 = 12, x1 = x0 + s + 12, y0 = 24, d = outIm.d;
      // inverse mapping render
      for (let cy = 0; cy < R; cy++) for (let cx = 0; cx < R; cx++) {
        const o = (cy * R + cx) * 4; let r = 0, gg = 0, b = 0;
        if (iv) {
          const xo = ((cx + 0.5) / R * 2 - 1) * V, yo = ((cy + 0.5) / R * 2 - 1) * V, w = iv[6] * xo + iv[7] * yo + iv[8];
          if (w > 1e-4) { const xs = (iv[0] * xo + iv[1] * yo + iv[2]) / w, ys = (iv[3] * xo + iv[4] * yo + iv[5]) / w, fx = xs * 60 + 59.5, fy = ys * 60 + 59.5;
            if (fx >= 0 && fy >= 0 && fx <= 119 && fy <= 119) { const X = Math.floor(fx), Y = Math.floor(fy), ax = fx - X, ay = fy - Y, i0 = (Y * 120 + X) * 4, X1 = X < 119 ? 4 : 0, Y1 = Y < 119 ? 480 : 0;
              const bl = (k) => (1 - ay) * ((1 - ax) * SRC[i0 + k] + ax * SRC[i0 + X1 + k]) + ay * ((1 - ax) * SRC[i0 + Y1 + k] + ax * SRC[i0 + Y1 + X1 + k]); r = bl(0); gg = bl(1); b = bl(2); } else { r = gg = b = 24; } } else { r = gg = b = 24; }
        } else r = gg = b = 24;
        d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = 255;
      }
      outIm.flush();
      const Xs = (x) => x0 + ((x / V + 1) / 2) * s, Ys = (y) => y0 + ((y / V + 1) / 2) * s, Xo = (x) => x1 + ((x / V + 1) / 2) * s, Yo = (y) => y0 + ((y / V + 1) / 2) * s;
      label(g, 'source (image = [−1,1]²)', x0, 16, C);
      g.fillStyle = C.bg2; g.fillRect(x0, y0, s, s); g.imageSmoothingEnabled = true;
      const srcImg = mkSrc(); g.drawImage(srcImg, Xs(-1), Ys(-1), Xs(1) - Xs(-1), Ys(1) - Ys(-1));
      label(g, 'output = M · source', x1, 16, C);
      g.drawImage(outIm.cv, x1, y0, s, s); border(g, x0, y0, s, s, C.border); border(g, x1, y0, s, s, C.border);
      const fw = (x, y) => { const w = Mt[6] * x + Mt[7] * y + Mt[8]; return w > 1e-3 ? [(Mt[0] * x + Mt[1] * y + Mt[2]) / w, (Mt[3] * x + Mt[4] * y + Mt[5]) / w] : null; };
      // grids
      g.lineWidth = 1; for (let k = -4; k <= 4; k++) { const v = k / 4; g.strokeStyle = k === 0 ? 'rgba(255,255,255,.8)' : 'rgba(255,255,255,.35)';
        g.beginPath(); g.moveTo(Xs(v), Ys(-1)); g.lineTo(Xs(v), Ys(1)); g.moveTo(Xs(-1), Ys(v)); g.lineTo(Xs(1), Ys(v)); g.stroke();
        g.strokeStyle = k === 0 ? 'rgba(255,200,60,.95)' : 'rgba(255,200,60,.5)';
        for (const horiz of [0, 1]) { const a = fw(horiz ? -1 : v, horiz ? v : -1), b = fw(horiz ? 1 : v, horiz ? v : 1); if (a && b) { g.beginPath(); g.moveTo(Xo(a[0]), Yo(a[1])); g.lineTo(Xo(b[0]), Yo(b[1])); g.stroke(); } } }
      const cn = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => fw(x, y));
      if (cn.every(Boolean)) { g.strokeStyle = C.orange; g.lineWidth = 3; g.beginPath(); cn.forEach((p, i) => (i ? g.lineTo(Xo(p[0]), Yo(p[1])) : g.moveTo(Xo(p[0]), Yo(p[1])))); g.closePath(); g.stroke(); g.lineWidth = 1; }
      // probe: an output pixel and where it comes from
      let pr = 'matrix is singular: the image collapses and cannot be un-done';
      if (iv) {
        const xo = 0.55 * Math.sin(tp * 0.9), yo = 0.55 * Math.sin(tp * 0.61 + 1), w = iv[6] * xo + iv[7] * yo + iv[8], xs = (iv[0] * xo + iv[1] * yo + iv[2]) / w, ys = (iv[3] * xo + iv[4] * yo + iv[5]) / w;
        g.fillStyle = C.blue; g.beginPath(); g.arc(Xo(xo), Yo(yo), 6, 0, 7); g.fill(); g.strokeStyle = C.fg; g.stroke();
        g.fillStyle = C.green; g.beginPath(); g.arc(Xs(xs), Ys(ys), 6, 0, 7); g.fill(); g.stroke();
        g.strokeStyle = C.blue; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(Xo(xo), Yo(yo)); g.lineTo(Xs(xs), Ys(ys)); g.stroke(); g.setLineDash([]);
        pr = `output pixel (${xo.toFixed(2)}, ${yo.toFixed(2)}) ● blue\nasks M⁻¹ → source (${xs.toFixed(2)}, ${ys.toFixed(2)}) ● green`;
      }
      const det = Mt[0] * (Mt[4] * Mt[8] - Mt[5] * Mt[7]) - Mt[1] * (Mt[3] * Mt[8] - Mt[5] * Mt[6]) + Mt[2] * (Mt[3] * Mt[7] - Mt[4] * Mt[6]);
      q('#out').innerHTML = `M${m < 1 ? ` (blend ${Math.round(m * 100)}%)` : ''} =\n${[0, 1, 2].map((r) => '[ ' + [0, 1, 2].map((k) => Mt[r * 3 + k].toFixed(2).padStart(6)).join(' ') + ' ]').join('\n')}\ndet = <b class="o">${AI.fmt(det, 3)}</b>${Mt[6] || Mt[7] ? '  <span class="m">(homography)</span>' : Math.abs(Mt[1] + Mt[3]) < 1e-9 && Mt[0] === Mt[4] ? '' : ''}\n${pr}`;
      q('#cap').textContent = 'Gold grid = the source grid pushed through M (forward). The picture itself is drawn the other way round: for each output pixel, M⁻¹ finds where in the source to read (blue dot asks, green dot answers) — so no holes appear. Edit the bottom row for perspective.';
    }, { fps: 30 });
    let srcCv = null; function mkSrc() { if (!srcCv) { srcCv = document.createElement('canvas'); srcCv.width = srcCv.height = 120; const ctx = srcCv.getContext('2d'), id = ctx.createImageData(120, 120); id.data.set(SRC); ctx.putImageData(id, 0, 0); } return srcCv; }
  };

  /* ======================================================================== */
  /* freq: 2-D spectrum, circular masks, reconstruction                        */
  /* ======================================================================== */
  AI.labs.freq = function (root) {
    const N = 60, NN = N * N;
    const { view } = AI.labShell(root, 'The frequency view of an image', 'every picture is a sum of waves; keep some, discard others, rebuild',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap" style="min-height:3em"></div>',
      `${sel('scene', 'Image', { house: 'house', shapes: 'colour shapes', digit: 'handwritten 7', edges: 'edges & stripes', checker: 'checkerboard' }, 'house')}
       ${sel('mode', 'Keep', { low: 'low frequencies (blur)', high: 'high frequencies (edges, detail)', band: 'a ring (band-pass)' }, 'low')}
       ${sl('rad', 'radius (frequency index)', 0, 42, 8, 1)}${chk('soft', 'soft Gaussian mask (no ringing)', false)}${chk('auto', 'sweep the radius automatically', true)}${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c), upd = vals(root);
    const cosT = new Float32Array(N * N), sinT = new Float32Array(N * N);
    for (let k = 0; k < N; k++) for (let n = 0; n < N; n++) { const a = (2 * Math.PI * ((k * n) % N)) / N; cosT[k * N + n] = Math.cos(a); sinT[k * N + n] = Math.sin(a); }
    const Fr = new Float32Array(NN), Fi = new Float32Array(NN), Tr = new Float32Array(NN), Ti = new Float32Array(NN), Mr = new Float32Array(NN), Mi = new Float32Array(NN), rec = new Float32Array(NN), recOut = new Float32Array(NN);
    const imO = mkImg(N, N), imS = mkImg(N, N), imR = mkImg(N, N); let gray, total = 1, lastKey = '', maxLog = 1, tt = 0;
    const dft = (inR, inI, outR, outI, sign) => { // separable 2D DFT, sign -1 forward, +1 inverse (unnormalised)
      for (let y = 0; y < N; y++) for (let u = 0; u < N; u++) { let sr = 0, si = 0; for (let x = 0; x < N; x++) { const co = cosT[u * N + x], sn = sign * sinT[u * N + x], a = inR[y * N + x], b = inI ? inI[y * N + x] : 0; sr += a * co - b * sn; si += a * sn + b * co; } Tr[y * N + u] = sr; Ti[y * N + u] = si; }
      for (let u = 0; u < N; u++) for (let v = 0; v < N; v++) { let sr = 0, si = 0; for (let y = 0; y < N; y++) { const co = cosT[v * N + y], sn = sign * sinT[v * N + y], a = Tr[y * N + u], b = Ti[y * N + u]; sr += a * co - b * sn; si += a * sn + b * co; } outR[v * N + u] = sr; outI[v * N + u] = si; }
    };
    const load = () => {
      gray = grayScene(q('#scene').value, N); imO.gray(gray); dft(gray, null, Fr, Fi, -1);
      total = 0; maxLog = 1; const lg = new Float32Array(NN);
      for (let i = 0; i < NN; i++) { const m = Math.hypot(Fr[i], Fi[i]); total += m * m; lg[i] = Math.log(1 + m); maxLog = Math.max(maxLog, lg[i]); }
      imS.gray(lg.map((v, i) => 0)); // placeholder size
      const d = imS.d; for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) { const du = (u + N / 2) % N, dv = (v + N / 2) % N, val = (lg[v * N + u] / maxLog) * 255; const o = (dv * N + du) * 4; d[o] = d[o + 1] = d[o + 2] = Math.min(255, val * 1.1); d[o + 3] = 255; }
      lastKey = ''; S.dirty = true;
    };
    q('#scene').addEventListener('input', load);
    P.onstep = (d) => { q('#auto').checked = false; q('#rad').value = clamp(+q('#rad').value + d * 2, 0, 42); upd(); S.dirty = true; };
    P.onreset = () => { tt = 0; q('#auto').checked = true; P.set(true); };
    load();
    AI.loop(root, (dt) => {
      if (P.playing && q('#auto').checked) { tt += dt * 0.05 * P.sp(); q('#rad').value = Math.round(pingpong(tt * 1) * 0 + (0.5 - 0.5 * Math.cos(tt * 2 * Math.PI)) * 42); upd(); S.dirty = true; }
      const key = [q('#scene').value, q('#mode').value, q('#rad').value, q('#soft').checked].join();
      if (!S.dirty && key === lastKey) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, mode = q('#mode').value, rad = +q('#rad').value, soft = q('#soft').checked;
      if (key !== lastKey) {
        lastKey = key; let kept = 0, en = 0;
        for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
          const du = u <= N / 2 ? u : u - N, dv = v <= N / 2 ? v : v - N, r = Math.hypot(du, dv), i = v * N + u; let w;
          if (mode === 'low') w = soft ? Math.exp(-(r * r) / (2 * Math.max(0.5, rad / 1.5) ** 2)) : r <= rad ? 1 : 0;
          else if (mode === 'high') w = soft ? 1 - Math.exp(-(r * r) / (2 * Math.max(0.5, rad / 1.5) ** 2)) : r > rad ? 1 : 0;
          else w = soft ? Math.exp(-((r - rad) ** 2) / (2 * 3 * 3)) : Math.abs(r - rad) <= 3 ? 1 : 0;
          Mr[i] = Fr[i] * w; Mi[i] = Fi[i] * w; if (w > 0.5) kept++; en += (Mr[i] ** 2 + Mi[i] ** 2);
        }
        dft(Mr, Mi, rec, recOut, 1);
        const add = mode === 'low' ? 0 : 128; for (let i = 0; i < NN; i++) rec[i] = clamp(rec[i] / NN + add, 0, 255);
        imR.gray(rec); q('#out').innerHTML = `coefficients kept: <b class="o">${AI.fmt((100 * kept) / NN, 1)}%</b>\nenergy kept: <b class="g">${AI.fmt((100 * en) / total, 1)}%</b>\n<span class="m">${mode === 'low' ? 'few coefficients carry most energy: natural images are mostly low-frequency' : 'high frequencies hold fine detail and edges but little energy'}</span>`;
      }
      g.clearRect(0, 0, W, H);
      const s = Math.min((W - 4 * 12) / 3, H - 30), xs = [12, 24 + s, 36 + 2 * s], y0 = 24;
      label(g, 'image', xs[0], 16, C); label(g, 'spectrum (log |F|), DC at centre', xs[1], 16, C); label(g, 'rebuilt from kept waves', xs[2], 16, C);
      blit(g, imO, xs[0], y0, s, s); blit(g, imS, xs[1], y0, s, s); blit(g, imR, xs[2], y0, s, s);
      xs.forEach((x) => border(g, x, y0, s, s, C.border));
      const cx = xs[1] + s / 2, cy = y0 + s / 2, f = s / N;
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.setLineDash(soft ? [5, 4] : []);
      if (mode === 'band') { g.beginPath(); g.arc(cx, cy, Math.max(0, rad - 3) * f, 0, 7); g.stroke(); g.beginPath(); g.arc(cx, cy, (rad + 3) * f, 0, 7); g.stroke(); } else { g.beginPath(); g.arc(cx, cy, Math.max(0.5, rad) * f, 0, 7); g.stroke(); }
      g.setLineDash([]); g.lineWidth = 1;
      q('#cap').textContent = mode === 'low' ? 'Inside the circle = slow brightness changes (blobs, gradients). Keep only these and the picture blurs. A hard circle causes ringing near edges; tick "soft" to see it disappear.' : mode === 'high' ? 'Outside the circle = fast changes (edges, texture, noise). Keep only these and just outlines remain, centred on gray.' : 'A ring keeps one band of detail sizes — the idea behind multi-scale filters.';
    }, { fps: 30 });
  };

  /* ======================================================================== */
  /* sines: 1-D sum-of-sines builder                                           */
  /* ======================================================================== */
  AI.labs.sines = function (root) {
    const K = 8;
    const PRE = { 'square wave': (k) => (k % 2 ? 4 / (Math.PI * k) : 0), 'sawtooth': (k) => (2 / (Math.PI * k)) * (k % 2 ? 1 : -1), 'triangle': (k) => (k % 2 ? (8 / (Math.PI * Math.PI * k * k)) * (((k - 1) / 2) % 2 ? -1 : 1) : 0), 'single sine': (k) => (k === 1 ? 1 : 0), 'sharp pulse': (k) => 0.25 };
    const { view } = AI.labShell(root, 'Building a signal from sine waves', 'each slider is the amount of one frequency; the sum is the signal',
      '<canvas style="height:360px"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>',
      `<div>Presets${chipRow('pre')}</div><div id="amps" style="display:grid;grid-template-columns:1fr 1fr;gap:2px 10px"></div>
       ${sl('nh', 'use only the first n harmonics', 1, K, K, 1)}<div class="row"><button class="btn" id="build">▶ build up one term at a time</button></div>${PLAY(1)}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), P = player(q), S = setup(root, c), upd = vals(root);
    const A = new Float32Array(K + 1);
    for (let k = 1; k <= K; k++) { const l = document.createElement('label'); l.innerHTML = `<span>A${k} <span class="v" data-v="a${k}"></span></span><input type="range" id="a${k}" min="-1.3" max="1.3" step="0.01" value="0">`; q('#amps').appendChild(l); q('#a' + k).oninput = () => { A[k] = +q('#a' + k).value; S.dirty = true; }; }
    const setPre = (f) => { for (let k = 1; k <= K; k++) { A[k] = +f(k).toFixed(3); q('#a' + k).value = A[k]; } upd(); S.dirty = true; };
    Object.entries(PRE).forEach(([n, f]) => chip(q('#pre'), n, () => { q('#nh').value = K; setPre(f); }));
    setPre(PRE['square wave']);
    let ph = 0, building = false, bt = 0;
    q('#build').onclick = () => { building = true; bt = 0; q('#nh').value = 1; upd(); P.set(true); };
    P.onstep = (d) => { q('#nh').value = clamp(+q('#nh').value + d, 1, K); upd(); S.dirty = true; };
    P.onreset = () => { ph = 0; building = false; q('#nh').value = K; upd(); P.set(true); S.dirty = true; };
    AI.loop(root, (dt) => {
      if (P.playing) { ph += dt * 0.4 * P.sp(); S.dirty = true; if (building) { bt += dt * P.sp(); const n = Math.min(K, 1 + Math.floor(bt / 1.6)); if (n !== +q('#nh').value) { q('#nh').value = n; upd(); } if (n >= K && bt > K * 1.6 + 1.5) building = false; } }
      if (!S.dirty) return; S.dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const nh = +q('#nh').value, pw = W - 50, top = 14, wh = H * 0.58, by = top + wh + 28, bh = H - by - 22;
      const mx = Math.max(1.3, ...A.map(Math.abs)) * 1.15, X = (t) => 36 + (t / (2 * Math.PI)) * pw, Y = (v) => top + wh / 2 - (v / (K > 0 ? 1.8 : 1)) * (wh / 2 / 1);
      g.fillStyle = C.bg2; g.fillRect(36, top, pw, wh); border(g, 36, top, pw, wh, C.border);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(36, Y(0)); g.lineTo(36 + pw, Y(0)); g.stroke();
      const cols = [C.blue, C.green, C.purple, C.teal, C.pink, C.yellow, C.red, C.orange];
      for (let k = 1; k <= nh; k++) { if (!A[k]) continue; g.strokeStyle = cols[(k - 1) % 8]; g.globalAlpha = 0.45; g.lineWidth = 1.5; g.beginPath(); for (let i = 0; i <= 200; i++) { const t = (i / 200) * 2 * Math.PI, v = A[k] * Math.sin(k * (t - ph)); i ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); } g.stroke(); }
      g.globalAlpha = 1; g.strokeStyle = C.fg; g.lineWidth = 3; g.beginPath(); let hi = -9, lo = 9;
      for (let i = 0; i <= 400; i++) { const t = (i / 400) * 2 * Math.PI; let v = 0; for (let k = 1; k <= nh; k++) v += A[k] * Math.sin(k * (t - ph)); hi = Math.max(hi, v); lo = Math.min(lo, v); i ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); } g.stroke(); g.lineWidth = 1;
      text(g, 'thick = sum of the first n terms; thin = each term', 42, top + 14, C.muted, 11);
      // spectrum
      g.fillStyle = C.bg2; g.fillRect(36, by, pw, bh); border(g, 36, by, pw, bh, C.border); const bw = pw / K, my = by + bh / 2;
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(36, my); g.lineTo(36 + pw, my); g.stroke();
      for (let k = 1; k <= K; k++) { const hgt = (A[k] / mx) * (bh / 2 - 4); g.fillStyle = k <= nh ? cols[(k - 1) % 8] : C.border; g.fillRect(36 + (k - 0.5) * bw - bw * 0.25, my - Math.max(hgt, 0), bw * 0.5, Math.abs(hgt)); text(g, k, 36 + (k - 0.5) * bw, by + bh + 14, C.muted, 10, 'center'); }
      text(g, 'spectrum: amplitude of each harmonic (frequency 1…8)', 42, by - 6, C.muted, 11);
      q('#out').innerHTML = `terms used: <b class="o">${nh}</b> of ${K}\nsum ranges ${AI.fmt(lo, 2)} … ${AI.fmt(hi, 2)}\n<span class="m">${nh >= 3 && PRE['square wave'] && Math.abs(A[1] - 4 / Math.PI) < 0.02 ? 'overshoot ≈ 9% of the jump at a sharp edge (Gibbs ringing)' : 'smooth shapes need few terms; sharp corners need many'}</span>`;
      q('#cap').textContent = 'Slow shapes use low harmonics; sharp corners need high ones. The same idea in 2-D is the image spectrum: edges = high frequencies.';
    }, { fps: 30 });
  };
})();
