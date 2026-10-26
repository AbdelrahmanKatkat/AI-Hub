/* Computer Vision · Module 0 labs (Images as numbers).
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes).
 *   cvchannels  RGB planes sliding apart and merging, toggle channels, probe a pixel
 *   cvcolor     colour-space explorer (RGB / HSV wheel / luminance / YCbCr / Lab) with editable colour
 *   cvlayout    HWC vs CHW memory order, strides, batch dimension, animated memory walk
 *   cvhist      histogram + brightness/contrast/gamma + editable per-pixel formula
 *   cvbayer     Bayer mosaic -> demosaicing animation
 *   cvjpeg      JPEG stepper: 8x8 block -> DCT -> quantise -> reconstruct, quality slider
 * All animation is dt-based, slow by default, with a speed slider and pause/step.
 */
(function () {
  const CV = AI.cv, $ = (s, r) => AI.$(s, r);
  const CH = ['#e5484d', '#30a46c', '#3e63dd', '#8b8d98'];
  const sceneOpts = (list) => (list || Object.keys(CV.SCENES)).map((k) => `<option value="${k}">${CV.SCENES[k]}</option>`).join('');
  const themeKey = () => document.documentElement.dataset.theme || '';
  const chipRow = (parent, opts, cur, cb) => {
    const bs = Object.entries(opts).map(([k, v]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = v; b.onclick = () => { cur = k; mark(); cb(k); }; parent.appendChild(b); return [k, b]; });
    const mark = () => bs.forEach(([k, b]) => (b.style.borderColor = k === cur ? 'var(--blue)' : ''));
    mark();
    return { set(k) { cur = k; mark(); } };
  };
  const clamp = AI.clamp, smooth = AI.smooth;
  const rgb2hsv = (r, g, b, h0) => {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d > 0) { h = mx === r ? 60 * (((g - b) / d + 6) % 6) : mx === g ? 60 * ((b - r) / d + 2) : 60 * ((r - g) / d + 4); } else h = h0 || 0;
    return [h, mx ? d / mx : 0, mx];
  };
  const hsv2rgb = (h, s, v) => {
    const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
  };
  const toLab = (R, G, B) => {
    const lin = (u) => { u /= 255; return u <= 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4); };
    const r = lin(R), g = lin(G), b = lin(B);
    const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047, Y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b, Z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
    const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
    return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
  };
  const hex2 = (v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0');

  /* ================= cvchannels ================= */
  AI.labs.cvchannels = function (root) {
    const { ctl } = AI.labShell(root, 'Split & recombine the colour channels', 'an RGB image is three grey planes added together — toggle channels, click to probe a pixel',
      '<canvas id="cv" style="height:340px;cursor:crosshair"></canvas><div class="caption" id="cap"></div>',
      `<label>Image <select id="scene">${sceneOpts()}</select></label>
       <div>Channels switched on<div class="row">${['R', 'G', 'B'].map((n, i) => `<label style="flex-direction:row;gap:4px;align-items:center;color:${CH[i]}"><input type="checkbox" id="c${i}" checked> ${n}</label>`).join('')}</div></div>
       <label>Plane display<select id="disp"><option value="tint">tinted by colour (light adds up)</option><option value="gray">grey values 0–255</option></select></label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="2" step="0.05" value="0.6"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next phase ▶</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    const N = 40, planes = [0, 1, 2].map(() => { const c = document.createElement('canvas'); c.width = c.height = N; return c; });
    const grays = [0, 1, 2].map(() => { const c = document.createElement('canvas'); c.width = c.height = N; return c; });
    const stage = document.createElement('canvas');
    let im, ph = 0, t = 0, playing = true, probe = { x: 20, y: 14 }, last = '', geo = null;
    const D = [1.6, 3.4, 2.6, 3.4];
    const CAP = ['① One RGB image. Every pixel stores three numbers (R, G, B). Here the three colour planes sit on top of each other and their light adds up.',
      '② The planes slide apart. Each one is an ordinary grey picture: how much red / green / blue light every pixel carries.',
      '③ Separated: three H×W arrays of numbers. Stacked together they are the "C" of a C×H×W tensor. Click a plane to probe a pixel.',
      '④ Adding the three planes back together restores the colour image (red + green light = yellow, all three = white).'];
    const build = () => {
      im = CV.resample(q('#scene').value, N);
      const arr = [im.r, im.g, im.b];
      for (let c = 0; c < 3; c++) {
        const a = planes[c].getContext('2d').createImageData(N, N), b = grays[c].getContext('2d').createImageData(N, N);
        for (let i = 0; i < N * N; i++) {
          const v = arr[c][i];
          a.data[i * 4 + c] = v; a.data[i * 4 + 3] = 255;
          b.data[i * 4] = b.data[i * 4 + 1] = b.data[i * 4 + 2] = v; b.data[i * 4 + 3] = 255;
        }
        planes[c].getContext('2d').putImageData(a, 0, 0); grays[c].getContext('2d').putImageData(b, 0, 0);
      }
      last = '';
    };
    const pval = () => (ph === 0 ? 0 : ph === 1 ? smooth(t / D[1]) : ph === 2 ? 1 : 1 - smooth(t / D[3]));
    const on = () => [0, 1, 2].map((i) => q('#c' + i).checked);
    const layout = (p) => {
      const s = Math.min(cv.h - 52, cv.w / 3.3), cx = cv.w / 2, sp = s * 1.1, y = 14;
      return { s, y, rects: [0, 1, 2].map((i) => ({ x: cx - s / 2 + (i - 1) * sp * p, y })) };
    };
    const pick = (e) => {
      if (!geo) return;
      const r = q('#cv').getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      for (let i = 2; i >= 0; i--) { const R = geo.rects[i]; if (mx >= R.x && mx < R.x + geo.s && my >= R.y && my < R.y + geo.s) { probe = { x: clamp(Math.floor((mx - R.x) / geo.s * N), 0, N - 1), y: clamp(Math.floor((my - R.y) / geo.s * N), 0, N - 1) }; last = ''; return; }
      }
    };
    let drag = false;
    q('#cv').addEventListener('pointerdown', (e) => { drag = true; q('#cv').setPointerCapture(e.pointerId); pick(e); });
    q('#cv').addEventListener('pointermove', (e) => drag && pick(e));
    q('#cv').addEventListener('pointerup', () => (drag = false));
    q('#scene').onchange = build; q('#disp').onchange = () => (last = '');
    [0, 1, 2].forEach((i) => (q('#c' + i).onchange = () => (last = '')));
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { ph = (ph + 1) % 4; t = 0; last = ''; };
    build();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp.toFixed(2) + '×';
      if (playing) { t += dt * sp; if (t >= D[ph]) { ph = (ph + 1) % 4; t = 0; } }
      const p = pval(), en = on(), disp = q('#disp').value;
      const key = [ph, Math.round(p * 500), probe.x, probe.y, en.join(), disp, cv.w, cv.h, themeKey()].join();
      if (key === last) return; last = key;
      if (stage.width !== Math.round(cv.w) || stage.height !== Math.round(cv.h)) { stage.width = Math.round(cv.w); stage.height = Math.round(cv.h); }
      geo = layout(p);
      const g = stage.getContext('2d'), C = AI.col();
      g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.fillStyle = '#000'; g.fillRect(0, 0, stage.width, stage.height);
      g.imageSmoothingEnabled = false; g.globalCompositeOperation = 'lighter';
      const k = disp === 'gray' ? p : 0;
      for (let i = 0; i < 3; i++) if (en[i]) {
        const R = geo.rects[i];
        if (k < 1) { g.globalAlpha = 1 - k; g.drawImage(planes[i], R.x, R.y, geo.s, geo.s); }
        if (k > 0) { g.globalAlpha = k; g.drawImage(grays[i], R.x, R.y, geo.s, geo.s); }
      }
      g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
      const c = cv.ctx; c.clearRect(0, 0, cv.w, cv.h); c.drawImage(stage, 0, 0, cv.w, cv.h);
      c.font = '12px ui-monospace,Menlo,Consolas,monospace'; c.textAlign = 'center';
      for (let i = 0; i < 3; i++) {
        const R = geo.rects[i];
        c.strokeStyle = en[i] ? CH[i] : C.muted; c.lineWidth = 1.5; c.setLineDash(en[i] ? [] : [5, 4]); c.strokeRect(R.x, R.y, geo.s, geo.s); c.setLineDash([]);
        if (p > 0.15) { c.globalAlpha = clamp((p - 0.15) * 3, 0, 1); c.fillStyle = CH[i]; c.fillText(['R plane', 'G plane', 'B plane'][i] + (en[i] ? '' : ' (off)'), R.x + geo.s / 2, R.y + geo.s + 18); c.globalAlpha = 1; }
        const px = R.x + (probe.x + 0.5) / N * geo.s, py = R.y + (probe.y + 0.5) / N * geo.s;
        if (p < 0.02 && i > 0) continue;
        c.strokeStyle = '#fff'; c.lineWidth = 1; c.strokeRect(R.x + probe.x / N * geo.s, R.y + probe.y / N * geo.s, geo.s / N, geo.s / N);
      }
      if (p < 0.02) { c.fillStyle = '#fff'; c.fillText('R + G + B', cv.w / 2, geo.y + geo.s + 18); }
      q('#cap').textContent = CAP[ph];
      const i = probe.y * N + probe.x, v = [im.r[i], im.g[i], im.b[i]].map(Math.round);
      const mix = v.map((x, j) => (en[j] ? x : 0));
      q('#out').innerHTML = `probe pixel (x=${probe.x}, y=${probe.y})\n<span class="r">R</span> = ${v[0]}   <span class="g">G</span> = ${v[1]}   <span class="b">B</span> = ${v[2]}\nshown colour = (${mix.join(', ')})\nluma 0.299R+0.587G+0.114B = <b class="o">${AI.fmt(0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2], 1)}</b>\n<span class="m">image ${N}×${N} here → 3 planes × ${N * N} numbers</span>`;
    }, { fps: 30 });
  };

  /* ================= cvcolor ================= */
  AI.labs.cvcolor = function (root) {
    const sl = (id, label, mn, mx, v, st) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${mn}" max="${mx}" step="${st || 1}" value="${v}"></label>`;
    const { ctl } = AI.labShell(root, 'Colour-space explorer', 'one colour, many coordinate systems — edit RGB or HSV and watch every number follow',
      '<canvas id="cv" style="height:320px"></canvas><div class="caption" id="cap">The disc is the HSV wheel at the current brightness V: angle = hue H, distance from centre = saturation S. The white ring is your colour.</div>',
      `<div>Presets<div class="chips" id="pre" style="margin-top:4px"></div></div>
       ${sl('r', '<b style="color:#e5484d">R</b>', 0, 255, 200)}${sl('g', '<b style="color:#30a46c">G</b>', 0, 255, 100)}${sl('b', '<b style="color:#3e63dd">B</b>', 0, 255, 50)}
       ${sl('h', 'Hue H (°)', 0, 359, 20)}${sl('s', 'Saturation S (%)', 0, 100, 75)}${sl('v', 'Value V (%)', 0, 100, 78)}
       <label>Hex <input type="text" id="hex" value="#c86432" spellcheck="false"></label>
       <label>Hue-sweep speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="2" step="0.05" value="0.6"></label>
       <div class="row"><button class="btn primary" id="sweep">▶ Sweep the hue</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    let R = 200, G = 100, B = 50, H = 20, S = 0.75, V = 200 / 255, sweeping = false, last = '', discV = -1;
    const disc = document.createElement('canvas'); disc.width = disc.height = 200;
    const setRGB = (r, g, b) => { R = clamp(Math.round(r), 0, 255); G = clamp(Math.round(g), 0, 255); B = clamp(Math.round(b), 0, 255); [H, S, V] = rgb2hsv(R, G, B, H); sync(); };
    const setHSV = (h, s, v) => { H = (h + 360) % 360; S = clamp(s, 0, 1); V = clamp(v, 0, 1); [R, G, B] = hsv2rgb(H, S, V).map(Math.round); sync(); };
    const sync = () => {
      const set = (id, v) => { if (document.activeElement !== q('#' + id)) q('#' + id).value = v; };
      set('r', R); set('g', G); set('b', B); set('h', Math.round(H) % 360); set('s', Math.round(S * 100)); set('v', Math.round(V * 100));
      if (document.activeElement !== q('#hex')) { q('#hex').value = '#' + hex2(R) + hex2(G) + hex2(B); q('#hex').classList.remove('bad'); }
      last = '';
    };
    ['r', 'g', 'b'].forEach((id) => (q('#' + id).oninput = () => setRGB(+q('#r').value, +q('#g').value, +q('#b').value)));
    ['h', 's', 'v'].forEach((id) => (q('#' + id).oninput = () => setHSV(+q('#h').value, +q('#s').value / 100, +q('#v').value / 100)));
    q('#hex').oninput = () => { const m = /^#?([0-9a-f]{6})$/i.exec(q('#hex').value.trim()); if (!m) return q('#hex').classList.add('bad'); q('#hex').classList.remove('bad'); setRGB(parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4), 16)); };
    const PRE = { red: [230, 40, 40], orange: [200, 100, 50], olive: [128, 128, 0], teal: [0, 150, 150], purple: [140, 60, 200], gray: [128, 128, 128], white: [255, 255, 255] };
    Object.entries(PRE).forEach(([k, v]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => { sweeping = false; q('#sweep').textContent = '▶ Sweep the hue'; setRGB(...v); }; q('#pre').appendChild(b); });
    q('#sweep').onclick = () => { sweeping = !sweeping; q('#sweep').textContent = sweeping ? '⏸ Stop sweep' : '▶ Sweep the hue'; if (sweeping && S < 0.2) setHSV(H, 0.9, Math.max(V, 0.8)); };
    const drawDisc = () => {
      const g = disc.getContext('2d'), id = g.createImageData(200, 200);
      for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) {
        const dx = (x - 99.5) / 100, dy = (99.5 - y) / 100, r = Math.hypot(dx, dy), i = (y * 200 + x) * 4;
        if (r > 1) { id.data[i + 3] = 0; continue; }
        const h = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360, c = hsv2rgb(h, r, V);
        id.data[i] = c[0]; id.data[i + 1] = c[1]; id.data[i + 2] = c[2]; id.data[i + 3] = 255;
      }
      g.putImageData(id, 0, 0);
    };
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp.toFixed(2) + '×';
      if (sweeping) setHSV(H + dt * sp * 20, S, V); // 20°/s at 1×: a full turn takes 18 s
      ['r', 'g', 'b'].forEach((id) => (q('#' + id + 'v').textContent = q('#' + id).value));
      q('#hv').textContent = Math.round(H) % 360 + '°'; q('#sv').textContent = Math.round(S * 100) + '%'; q('#vv').textContent = Math.round(V * 100) + '%';
      const key = [R, G, B, Math.round(H * 4), cv.w, cv.h, themeKey()].join();
      if (key === last) return; last = key;
      const vq = Math.round(V * 64); if (vq !== discV) { discV = vq; drawDisc(); }
      const c = cv.ctx, C = AI.col(); c.clearRect(0, 0, cv.w, cv.h);
      const D = Math.min(cv.h - 24, cv.w * 0.5), cx = 12 + D / 2, cy = cv.h / 2;
      c.imageSmoothingEnabled = true; c.drawImage(disc, cx - D / 2, cy - D / 2, D, D);
      c.font = '11px ui-monospace,Menlo,Consolas,monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
      [['0° red', 0], ['120° green', 120], ['240° blue', 240]].forEach(([t, a]) => { c.fillStyle = C.muted; const r = D / 2 - 4; const x = cx + Math.cos((a * Math.PI) / 180) * (r - 2) * 0.82, y = cy - Math.sin((a * Math.PI) / 180) * (r - 2) * 0.82; c.fillStyle = V > 0.55 ? '#000' : '#fff'; c.fillText(t, x, y); });
      const mx = cx + Math.cos((H * Math.PI) / 180) * S * D / 2, my = cy - Math.sin((H * Math.PI) / 180) * S * D / 2;
      c.lineWidth = 3; c.strokeStyle = '#000'; c.beginPath(); c.arc(mx, my, 8, 0, 7); c.stroke(); c.lineWidth = 2; c.strokeStyle = '#fff'; c.beginPath(); c.arc(mx, my, 8, 0, 7); c.stroke();
      // bars + swatches
      const x0 = cx + D / 2 + 24, bw = Math.max(40, cv.w - x0 - 14), y0 = cy - D / 2 + 4;
      c.textAlign = 'left';
      [['R', R, CH[0]], ['G', G, CH[1]], ['B', B, CH[2]]].forEach(([n, v, col], i) => {
        const y = y0 + i * 30; c.fillStyle = C.bg3 || C.border; c.fillRect(x0 + 18, y, bw - 18, 16); c.fillStyle = col; c.fillRect(x0 + 18, y, (bw - 18) * v / 255, 16);
        c.fillStyle = C.fg; c.fillText(n, x0, y + 8); c.fillText(String(v), x0 + 22, y + 8);
      });
      const Y = Math.round(0.299 * R + 0.587 * G + 0.114 * B), M = Math.round((R + G + B) / 3), sy = y0 + 108, sw = Math.min(70, (bw - 12) / 3);
      [['colour', `rgb(${R},${G},${B})`], ['luma ' + Y, `rgb(${Y},${Y},${Y})`], ['mean ' + M, `rgb(${M},${M},${M})`]].forEach(([t, f], i) => { c.fillStyle = f; c.fillRect(x0 + i * (sw + 6), sy, sw, 46); c.strokeStyle = C.border; c.strokeRect(x0 + i * (sw + 6) + 0.5, sy + 0.5, sw, 46); c.fillStyle = C.muted; c.fillText(t, x0 + i * (sw + 6), sy + 58); });
      c.fillStyle = C.muted; c.fillText('Two greys from the same colour: which one feels as bright as the colour?', x0, sy + 82, bw);
      const [Ly, a, bb] = toLab(R, G, B);
      const Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B, Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
      q('#out').innerHTML = `<b>RGB</b> (${R}, ${G}, ${B})   hex ${'#' + hex2(R) + hex2(G) + hex2(B)}\n<b>HSV</b> (${AI.fmt(H, 0)}°, ${AI.fmt(S * 100, 0)}%, ${AI.fmt(V * 100, 0)}%)\n<b>Luma</b> 0.299·${R} + 0.587·${G} + 0.114·${B}\n     = <b class="o">${AI.fmt(0.299 * R + 0.587 * G + 0.114 * B, 1)}</b>   (plain mean ${AI.fmt((R + G + B) / 3, 1)})\n<b>YCbCr</b> (${AI.fmt(0.299 * R + 0.587 * G + 0.114 * B, 1)}, ${AI.fmt(Cb, 1)}, ${AI.fmt(Cr, 1)})\n<b>Lab</b> (${AI.fmt(Ly, 1)}, ${AI.fmt(a, 1)}, ${AI.fmt(bb, 1)})`;
    }, { fps: 30 });
  };

  /* ================= cvlayout ================= */
  AI.labs.cvlayout = function (root) {
    const sl = (id, label, mn, mx, v) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${mn}" max="${mx}" value="${v}"></label>`;
    const { ctl } = AI.labShell(root, 'How a tensor sits in memory', 'the same numbers, two orders — walk through memory, query an element, change the shape',
      '<canvas id="cv" style="height:430px"></canvas><div class="caption" id="cap"></div>',
      `<div>Memory layout<div class="row" id="lay"></div></div>
       <div class="row"><div style="flex:1">${sl('N', 'Batch N', 1, 3, 2)}</div><div style="flex:1">${sl('C', 'Channels C', 1, 4, 3)}</div></div>
       <div class="row"><div style="flex:1">${sl('H', 'Height H', 2, 4, 3)}</div><div style="flex:1">${sl('W', 'Width W', 2, 6, 4)}</div></div>
       <div><b>Query one element</b> <span style="color:var(--muted)">(white outline)</span></div>
       <div class="row"><div style="flex:1">${sl('qn', 'n', 0, 2, 0)}</div><div style="flex:1">${sl('qc', 'c', 0, 2, 0)}</div></div>
       <div class="row"><div style="flex:1">${sl('qh', 'h', 0, 2, 1)}</div><div style="flex:1">${sl('qw', 'w', 0, 3, 2)}</div></div>
       <label>Memory-walk speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.5" max="8" step="0.5" value="2.5"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step +1</button><button class="btn" id="rst">Restart</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    let lay = 'NCHW', playing = true, k = 0, acc = 0, last = '';
    const tick = AI.ticker();
    const chip = chipRow(q('#lay'), { NCHW: 'NCHW  (PyTorch)', NHWC: 'NHWC  (TensorFlow)' }, lay, (v) => { lay = v; last = ''; });
    const dims = () => ({ N: +q('#N').value, C: +q('#C').value, H: +q('#H').value, W: +q('#W').value });
    const strides = (d) => (lay === 'NCHW' ? { n: d.C * d.H * d.W, c: d.H * d.W, h: d.W, w: 1 } : { n: d.H * d.W * d.C, c: 1, h: d.W * d.C, w: d.C });
    const inv = (d, o) => { const s = strides(d); const r = { n: 0, c: 0, h: 0, w: 0 }; let rem = o; for (const key of Object.keys(s).sort((a, b) => s[b] - s[a])) { r[key] = Math.floor(rem / s[key]); rem -= r[key] * s[key]; } return r; };
    ['N', 'C', 'H', 'W', 'qn', 'qc', 'qh', 'qw'].forEach((id) => (q('#' + id).oninput = () => (last = '')));
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; k++; last = ''; };
    q('#rst').onclick = () => { k = 0; last = ''; };
    AI.loop(root, (dt) => {
      const d = dims(), T = d.N * d.C * d.H * d.W;
      [['qn', d.N], ['qc', d.C], ['qh', d.H], ['qw', d.W]].forEach(([id, mx]) => { const e = q('#' + id); e.max = mx - 1; if (+e.value > mx - 1) e.value = mx - 1; });
      ['N', 'C', 'H', 'W', 'qn', 'qc', 'qh', 'qw'].forEach((id) => (q('#' + id + 'v').textContent = q('#' + id).value));
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' cells/s';
      if (playing) { const n = tick(dt, sp); if (n) { k += n; last = ''; } }
      k = ((k % T) + T) % T;
      const key = [lay, T, d.N, d.C, d.H, d.W, q('#qn').value, q('#qc').value, q('#qh').value, q('#qw').value, k, cv.w, cv.h, themeKey()].join();
      if (key === last) return; last = key;
      const qn = +q('#qn').value, qc = +q('#qc').value, qh = +q('#qh').value, qw = +q('#qw').value, st = strides(d);
      const qo = qn * st.n + qc * st.c + qh * st.h + qw * st.w, sc = inv(d, k);
      const c = cv.ctx, C = AI.col(); c.clearRect(0, 0, cv.w, cv.h);
      c.font = '11px ui-monospace,Menlo,Consolas,monospace'; c.textBaseline = 'middle';
      // logical view
      c.fillStyle = C.fg; c.textAlign = 'left'; c.fillText(`Logical view — what you index: tensor[n][c][h][w], shape (${lay === 'NCHW' ? `${d.N}, ${d.C}, ${d.H}, ${d.W}` : `${d.N}, ${d.H}, ${d.W}, ${d.C}`})`, 8, 12);
      const lh = 168, cs = Math.max(14, Math.min(38, (cv.w / d.N - 26) / d.W, (lh - 30) / d.H));
      for (let n = 0; n < d.N; n++) {
        const ox = 10 + n * (d.W * cs + 22), oy = 38;
        c.fillStyle = C.muted; c.textAlign = 'left'; c.fillText('n = ' + n, ox, oy - 10);
        for (let h = 0; h < d.H; h++) for (let w = 0; w < d.W; w++) {
          c.strokeStyle = C.border; c.strokeRect(ox + w * cs + 0.5, oy + h * cs + 0.5, cs, cs);
          for (let ch = 0; ch < d.C; ch++) {
            const bw = (cs - 4) / d.C; c.fillStyle = CH[Math.min(ch, 3)]; c.globalAlpha = 0.85; c.fillRect(ox + w * cs + 2 + ch * bw, oy + h * cs + 2, bw - 0.5, cs - 4); c.globalAlpha = 1;
            const isQ = n === qn && ch === qc && h === qh && w === qw, isS = n === sc.n && ch === sc.c && h === sc.h && w === sc.w;
            if (isS) { c.strokeStyle = '#ff9b3d'; c.lineWidth = 2.5; c.strokeRect(ox + w * cs + 2 + ch * bw - 1, oy + h * cs + 1, bw + 1, cs - 2); }
            if (isQ) { c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.strokeRect(ox + w * cs + 2 + ch * bw - 2, oy + h * cs, bw + 3, cs); }
            c.lineWidth = 1;
          }
        }
      }
      c.textAlign = 'left'; c.fillStyle = C.muted; c.fillText('each cell = one pixel; its coloured slivers = its C channel values', 10, 38 + d.H * cs + 14);
      // memory strip
      const my = lh + 54; c.fillStyle = C.fg; c.fillText('Memory — one long line of numbers, wrapped into rows (offset 0 at top-left). Orange = being read, white = your query.', 8, my - 16);
      const cw = Math.max(16, Math.min(30, Math.floor((cv.w - 16) / Math.min(T, 30)))), cols = Math.max(1, Math.min(T, Math.floor((cv.w - 16) / cw)));
      const ch2 = Math.min(cw, 26), pos = (o) => ({ x: 8 + (o % cols) * cw, y: my + Math.floor(o / cols) * (ch2 + 3) });
      c.textAlign = 'center';
      for (let o = 0; o < T; o++) {
        const e = inv(d, o), p = pos(o);
        c.fillStyle = CH[Math.min(e.c, 3)]; c.globalAlpha = 0.8; c.fillRect(p.x, p.y, cw - 2, ch2); c.globalAlpha = 1;
        if (cw >= 18) { c.fillStyle = '#fff'; c.fillText(String(o), p.x + (cw - 2) / 2, p.y + ch2 / 2); }
      }
      // pixel's channels highlighted
      const pxO = []; for (let ch = 0; ch < d.C; ch++) pxO.push(qn * st.n + ch * st.c + qh * st.h + qw * st.w);
      c.strokeStyle = '#ffe14d'; c.lineWidth = 2;
      pxO.forEach((o) => { const p = pos(o); c.strokeRect(p.x + 1, p.y + ch2 + 1, cw - 4, 0); c.beginPath(); c.moveTo(p.x + 1, p.y + ch2 + 1.5); c.lineTo(p.x + cw - 3, p.y + ch2 + 1.5); c.stroke(); });
      const sp2 = pos(k), qp = pos(qo);
      c.strokeStyle = '#ff9b3d'; c.lineWidth = 3; c.strokeRect(sp2.x - 1, sp2.y - 1, cw, ch2 + 2);
      c.strokeStyle = '#fff'; c.lineWidth = 2; c.strokeRect(qp.x - 2, qp.y - 2, cw + 2, ch2 + 4); c.lineWidth = 1;
      const rows = Math.ceil(T / cols);
      c.textAlign = 'left'; c.fillStyle = '#ffe14d'; c.fillText('yellow ticks under the cells: the C channel values of the SAME pixel as your query', 8, my + rows * (ch2 + 3) + 10);
      const nbr = pxO.map((o) => o).sort((a, b) => a - b), gap = nbr.length > 1 ? nbr[1] - nbr[0] : 0;
      q('#cap').innerHTML = `Orange cursor: memory offset <b>${k}</b> holds element (n=${sc.n}, c=${sc.c}, h=${sc.h}, w=${sc.w}). In <b>${lay}</b> the channels of one pixel are ${gap === 1 ? 'next to each other (good for per-pixel work)' : `${gap} cells apart (each channel is a contiguous plane)`}.`;
      const f = lay === 'NCHW' ? `n·${st.n} + c·${st.c} + h·${st.h} + w·${st.w}` : `n·${st.n} + h·${st.h} + w·${st.w} + c·${st.c}`;
      q('#out').innerHTML = `layout <b>${lay}</b>   elements = <b class="o">${T}</b>\nstrides (n,c,h,w) = (${st.n}, ${st.c}, ${st.h}, ${st.w})\noffset = ${f}\n       = ${qn}·${st.n} + ${qc}·${st.c} + ${qh}·${st.h} + ${qw}·${st.w} = <b class="o">${qo}</b>\nbytes (float32) = ${T} × 4 = <b>${T * 4}</b>`;
    }, { fps: 30 });
  };

  /* ================= cvhist ================= */
  AI.labs.cvhist = function (root) {
    const sl = (id, label, mn, mx, v, st) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${mn}" max="${mx}" step="${st}" value="${v}"></label>`;
    const DEF = 'max(c*(v-0.5)+0.5+b,0)^g';
    const { ctl } = AI.labShell(root, 'Histogram & per-pixel formula', 'out = f(v) is applied to every pixel value v∈[0,1] — brightness b, contrast c, gamma g, or your own formula',
      '<canvas id="im" style="height:190px"></canvas><canvas id="hist" style="height:200px"></canvas><div class="caption" id="cap"></div>',
      `<label>Image <select id="scene">${sceneOpts()}</select></label>
       <label>Original exposure <select id="expo"><option value="norm">as is</option><option value="dark">under-exposed (dark)</option><option value="flat">low contrast (foggy)</option><option value="bright">over-exposed (bright)</option></select></label>
       <label>Histogram of <select id="mode"><option value="lum">luminance</option><option value="rgb">R, G, B channels</option></select></label>
       ${sl('b', 'Brightness b', -0.5, 0.5, 0, 0.01)}${sl('c', 'Contrast c', 0.2, 3, 1, 0.05)}${sl('g', 'Gamma g', 0.3, 3, 1, 0.05)}
       <label>Formula  out = f(v)   (use v, b, c, g)<input type="text" id="f" value="${DEF}" spellcheck="false"></label><div class="readout" id="err" style="color:var(--red);min-height:0;padding:0;border:0;background:none"></div>
       <div class="row"><button class="btn" id="rst">Reset sliders</button><button class="btn primary" id="sweep">▶ Auto-sweep b, c, g</button></div>
       <label>Sweep speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="1.5" step="0.05" value="0.5"></label>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cim = AI.canvas(q('#im')), chs = AI.canvas(q('#hist'));
    const S = 120, base = [new Uint8Array(S * S), new Uint8Array(S * S), new Uint8Array(S * S)], outp = [new Uint8Array(S * S), new Uint8Array(S * S), new Uint8Array(S * S)];
    const cvB = document.createElement('canvas'), cvA = document.createElement('canvas'); cvB.width = cvB.height = cvA.width = cvA.height = S;
    const lut = new Uint8Array(256), NB = 64;
    const cur = [0, 1, 2, 3].map(() => new Float32Array(NB)), tgt = [0, 1, 2, 3].map(() => new Float32Array(NB)), bef = [0, 1, 2, 3].map(() => new Float32Array(NB));
    let fn = null, dirty = true, last = '', sweep = false, st = 0, stats = null;
    const expoMap = { norm: (v) => v, dark: (v) => v * 0.3, flat: (v) => 0.4 + 0.2 * v, bright: (v) => 0.65 + 0.35 * v };
    const loadBase = () => {
      const d = CV.scene(q('#scene').value), e = expoMap[q('#expo').value];
      for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) base[c][i] = Math.round(255 * e(d[i * 4 + c] / 255));
      const id = cvB.getContext('2d').createImageData(S, S);
      for (let i = 0; i < S * S; i++) { id.data[i * 4] = base[0][i]; id.data[i * 4 + 1] = base[1][i]; id.data[i * 4 + 2] = base[2][i]; id.data[i * 4 + 3] = 255; }
      cvB.getContext('2d').putImageData(id, 0, 0); dirty = true;
    };
    const ex = AI.exprInput(q('#f'), q('#err'), { allowed: ['v', 'b', 'c', 'g'], check: (f) => { const t = f({ v: 0.5, b: 0, c: 1, g: 1 }); if (!Number.isFinite(t)) throw new Error('f(0.5) is not a number'); }, onChange: (f) => { fn = f; dirty = true; } });
    ex.run();
    ['scene', 'expo', 'mode'].forEach((id) => (q('#' + id).onchange = () => { id === 'mode' ? (dirty = true) : loadBase(); }));
    ['b', 'c', 'g'].forEach((id) => (q('#' + id).oninput = () => { dirty = true; }));
    q('#rst').onclick = () => { sweep = false; q('#sweep').textContent = '▶ Auto-sweep b, c, g'; q('#b').value = 0; q('#c').value = 1; q('#g').value = 1; q('#f').value = DEF; ex.run(); dirty = true; };
    q('#sweep').onclick = () => { sweep = !sweep; st = 0; q('#sweep').textContent = sweep ? '⏸ Stop sweep' : '▶ Auto-sweep b, c, g'; if (sweep) { q('#b').value = 0; q('#c').value = 1; q('#g').value = 1; q('#f').value = DEF; ex.run(); } };
    const histo = (arrs, dst, lum) => {
      dst.forEach((a) => a.fill(0));
      for (let i = 0; i < S * S; i++) {
        dst[3][((0.299 * arrs[0][i] + 0.587 * arrs[1][i] + 0.114 * arrs[2][i]) / 256 * NB) | 0]++;
        for (let c = 0; c < 3; c++) dst[c][(arrs[c][i] / 256 * NB) | 0]++;
      }
    };
    const lumStats = (arrs) => { const l = new Float32Array(S * S); for (let i = 0; i < S * S; i++) l[i] = 0.299 * arrs[0][i] + 0.587 * arrs[1][i] + 0.114 * arrs[2][i]; return AI.stats(l); };
    let sB = null;
    const recompute = () => {
      const b = +q('#b').value, c = +q('#c').value, g = +q('#g').value;
      for (let i = 0; i < 256; i++) { let o = fn({ v: i / 255, b, c, g }); if (!Number.isFinite(o)) o = 0; lut[i] = Math.round(clamp(o, 0, 1) * 255); }
      for (let ch = 0; ch < 3; ch++) for (let i = 0; i < S * S; i++) outp[ch][i] = lut[base[ch][i]];
      const id = cvA.getContext('2d').createImageData(S, S);
      for (let i = 0; i < S * S; i++) { id.data[i * 4] = outp[0][i]; id.data[i * 4 + 1] = outp[1][i]; id.data[i * 4 + 2] = outp[2][i]; id.data[i * 4 + 3] = 255; }
      cvA.getContext('2d').putImageData(id, 0, 0);
      histo(base, bef); histo(outp, tgt);
      sB = lumStats(base); stats = lumStats(outp);
      let clipped = 0; for (let i = 0; i < S * S; i++) { const l = 0.299 * outp[0][i] + 0.587 * outp[1][i] + 0.114 * outp[2][i]; if (l < 1 || l > 254) clipped++; }
      stats.clip = clipped / (S * S) * 100;
    };
    loadBase();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp.toFixed(2) + '×';
      if (sweep) {
        st += dt * sp; const u = (st % 10) / 10, which = Math.floor(st / 10) % 3, s = Math.sin(2 * Math.PI * u);
        q('#b').value = which === 0 ? 0.3 * s : 0; q('#c').value = which === 1 ? 1 + 0.8 * s : 1; q('#g').value = which === 2 ? Math.exp(0.7 * s) : 1; dirty = true;
      }
      ['b', 'c', 'g'].forEach((id) => (q('#' + id + 'v').textContent = (+q('#' + id).value).toFixed(2)));
      if (dirty && fn) { dirty = false; recompute(); }
      let moving = false; const mode = q('#mode').value;
      for (let k = 0; k < 4; k++) for (let i = 0; i < NB; i++) { const dlt = tgt[k][i] - cur[k][i]; if (Math.abs(dlt) > 0.5) { cur[k][i] += dlt * Math.min(1, dt * 7); moving = true; } else cur[k][i] = tgt[k][i]; }
      const key = [moving ? Math.random() : 0, q('#b').value, q('#c').value, q('#g').value, q('#f').value, q('#scene').value, q('#expo').value, mode, cim.w, chs.w, themeKey(), sweep ? st : 0].join();
      if (key === last && !moving) return; last = key;
      const C = AI.col(), c = cim.ctx; c.clearRect(0, 0, cim.w, cim.h);
      const s = Math.min((cim.w - 40) / 3, cim.h - 28), gap = (cim.w - 3 * s) / 4;
      c.imageSmoothingEnabled = false; c.font = '11px ui-monospace,Menlo,Consolas,monospace'; c.textAlign = 'center'; c.fillStyle = C.muted;
      c.drawImage(cvB, gap, 4, s, s); c.fillText('before', gap + s / 2, s + 16);
      c.drawImage(cvA, 2 * gap + s, 4, s, s); c.fillText('after  out = f(v)', 2 * gap + s * 1.5, s + 16);
      const px = 3 * gap + 2 * s;
      c.fillStyle = C.bg2; c.fillRect(px, 4, s, s); c.strokeStyle = C.border; c.strokeRect(px + 0.5, 4.5, s, s);
      c.strokeStyle = C.muted; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(px, 4 + s); c.lineTo(px + s, 4); c.stroke(); c.setLineDash([]);
      c.strokeStyle = C.orange; c.lineWidth = 2; c.beginPath(); for (let i = 0; i < 256; i++) { const X = px + (i / 255) * s, Y = 4 + s - (lut[i] / 255) * s; i ? c.lineTo(X, Y) : c.moveTo(X, Y); } c.stroke(); c.lineWidth = 1;
      c.fillStyle = C.muted; c.fillText('curve: in → out', px + s / 2, s + 16);
      // histogram
      const h = chs.ctx; h.clearRect(0, 0, chs.w, chs.h);
      const pad = { l: 8, r: 8, t: 8, b: 22 }, pw = chs.w - pad.l - pad.r, ph2 = chs.h - pad.t - pad.b, bw = pw / NB;
      const set = mode === 'lum' ? [3] : [0, 1, 2]; let mx = 1; for (const k of set) for (let i = 0; i < NB; i++) mx = Math.max(mx, bef[k][i], tgt[k][i]);
      mx = Math.max(mx * 0.7, 1);
      const colr = mode === 'lum' ? [C.fg] : CH.slice(0, 3);
      set.forEach((k, j) => {
        h.fillStyle = colr[j]; h.globalAlpha = mode === 'lum' ? 0.75 : 0.45;
        for (let i = 0; i < NB; i++) { const v = Math.min(1.05, cur[k][i] / mx) * ph2; h.fillRect(pad.l + i * bw, pad.t + ph2 - v, Math.max(1, bw - 1), v); }
        h.globalAlpha = 1; h.strokeStyle = mode === 'lum' ? C.orange : colr[j]; h.lineWidth = 1.5; h.setLineDash([4, 3]); h.beginPath();
        for (let i = 0; i < NB; i++) { const v = Math.min(1.05, bef[k][i] / mx) * ph2, x = pad.l + i * bw; i ? h.lineTo(x, pad.t + ph2 - v) : h.moveTo(x, pad.t + ph2 - v); h.lineTo(x + bw, pad.t + ph2 - v); } h.stroke(); h.setLineDash([]);
      });
      h.strokeStyle = C.muted; h.beginPath(); h.moveTo(pad.l, pad.t + ph2 + 0.5); h.lineTo(pad.l + pw, pad.t + ph2 + 0.5); h.stroke();
      h.fillStyle = C.muted; h.font = '11px ui-monospace,Menlo,Consolas,monospace'; h.textAlign = 'center';
      [0, 64, 128, 192, 255].forEach((v) => h.fillText(v, pad.l + (v / 256) * pw + (v === 255 ? -6 : v === 0 ? 6 : 0), pad.t + ph2 + 15));
      if (stats) {
        const mX = pad.l + stats.mean / 256 * pw; h.strokeStyle = C.blue; h.lineWidth = 2; h.beginPath(); h.moveTo(mX, pad.t); h.lineTo(mX, pad.t + ph2); h.stroke(); h.lineWidth = 1;
        h.fillStyle = C.blue; h.textAlign = 'left'; h.fillText('mean', clamp(mX + 4, 0, chs.w - 36), pad.t + 10);
      }
      h.textAlign = 'right'; h.fillStyle = C.muted; h.fillText('filled = after, dashed = before   (x: value 0 → 255, y: how many pixels)', chs.w - 8, pad.t + 10 + 14);
      if (stats) {
        q('#out').innerHTML = `            before     after\nmean      ${AI.fmt(sB.mean, 1).padStart(7)}   <b class="o">${AI.fmt(stats.mean, 1).padStart(7)}</b>\nstd       ${AI.fmt(sB.std, 1).padStart(7)}   <b class="o">${AI.fmt(stats.std, 1).padStart(7)}</b>\nmin–max   ${(AI.fmt(sB.min, 0) + '–' + AI.fmt(sB.max, 0)).padStart(7)}   ${(AI.fmt(stats.min, 0) + '–' + AI.fmt(stats.max, 0)).padStart(7)}\nclipped (0 or 255)  <b class="${stats.clip > 5 ? 'r' : 'g'}">${AI.fmt(stats.clip, 1)}%</b>`;
        q('#cap').textContent = sweep ? ['Sweeping brightness b: the whole histogram slides left/right — std stays the same.', 'Sweeping contrast c: the histogram stretches/squeezes around 0.5 — std changes, mean barely moves.', 'Sweeping gamma g: shadows and highlights move differently — the shape bends, 0 and 255 stay put.'][Math.floor(st / 10) % 3] : 'Move a slider or edit the formula: the histogram is a summary of all pixel values, ignoring where they are.';
      }
    }, { fps: 30 });
  };

  /* ================= cvbayer ================= */
  AI.labs.cvbayer = function (root) {
    const { ctl } = AI.labShell(root, 'From Bayer mosaic to a colour image', 'each photosite records ONE colour; the other two are interpolated from neighbours (demosaicing)',
      '<canvas id="cv" style="height:300px"></canvas><div class="caption" id="cap"></div>',
      `<label>Scene <select id="scene">${sceneOpts(['shapes', 'house', 'edges', 'checker', 'digit'])}</select></label>
       <label>Grid size <select id="n"><option value="8">8 × 8</option><option value="12" selected>12 × 12</option><option value="15">15 × 15</option><option value="20">20 × 20</option></select></label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="1" max="40" value="6"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="rst">Restart</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    let im, n = 12, rec, k = 0, playing = true, last = '';
    const tick = AI.ticker();
    const filt = (x, y) => ((y & 1) === 0 ? ((x & 1) === 0 ? 0 : 1) : ((x & 1) === 0 ? 1 : 2));
    const build = () => {
      n = +q('#n').value; im = CV.resample(q('#scene').value, n); const arr = [im.r, im.g, im.b];
      rec = [new Float32Array(n * n), new Float32Array(n * n), new Float32Array(n * n)];
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 3; c++) {
        if (filt(x, y) === c) { rec[c][y * n + x] = arr[c][y * n + x]; continue; }
        let s = 0, m = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < n && Y < n && filt(X, Y) === c) { s += arr[c][Y * n + X]; m++; } }
        rec[c][y * n + x] = s / m;
      }
      let e = 0; for (let c = 0; c < 3; c++) for (let i = 0; i < n * n; i++) e += Math.abs(rec[c][i] - arr[c][i]); im.err = e / (3 * n * n);
      k = 0; last = '';
    };
    q('#scene').onchange = build; q('#n').onchange = build;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; k = Math.min(n * n, k + 1); last = ''; };
    q('#rst').onclick = () => { k = 0; last = ''; };
    build();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' px/s';
      if (playing && k < n * n) { const m = tick(dt, sp); if (m) { k = Math.min(n * n, k + m); last = ''; } }
      const key = [k, n, cv.w, cv.h, themeKey()].join(); if (key === last) return; last = key;
      const c = cv.ctx, C = AI.col(); c.clearRect(0, 0, cv.w, cv.h);
      const s = Math.min((cv.w - 40) / 3, cv.h - 40), gap = (cv.w - 3 * s) / 4, cs = s / n, oy = 22;
      c.font = '11px ui-monospace,Menlo,Consolas,monospace'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = C.muted;
      ['① what the scene really is', '② what the sensor records', '③ reconstructed (demosaiced)'].forEach((t, i) => c.fillText(t, gap * (i + 1) + s * (i + 0.5), 10, s + gap * 0.9));
      const cur = Math.min(k, n * n - 1), cx = cur % n, cy = Math.floor(cur / n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x, f = filt(x, y), v = [im.r[i], im.g[i], im.b[i]], X = [gap + x * cs, 2 * gap + s + x * cs, 3 * gap + 2 * s + x * cs], Y = oy + y * cs;
        c.fillStyle = `rgb(${Math.round(v[0])},${Math.round(v[1])},${Math.round(v[2])})`; c.fillRect(X[0], Y, cs + 0.6, cs + 0.6);
        const t = [0, 0, 0]; t[f] = v[f]; c.fillStyle = `rgb(${Math.round(t[0])},${Math.round(t[1])},${Math.round(t[2])})`; c.fillRect(X[1], Y, cs + 0.6, cs + 0.6);
        if (cs >= 10) { c.strokeStyle = 'rgba(0,0,0,.35)'; c.strokeRect(X[1] + 0.5, Y + 0.5, cs, cs); }
        const done = i < k;
        if (done) c.fillStyle = `rgb(${Math.round(rec[0][i])},${Math.round(rec[1][i])},${Math.round(rec[2][i])})`;
        else { const g = Math.round(0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]); c.fillStyle = `rgb(${g},${g},${g})`; c.globalAlpha = 0.35; }
        c.fillRect(X[2], Y, cs + 0.6, cs + 0.6); c.globalAlpha = 1;
      }
      c.strokeStyle = '#ffe14d'; c.lineWidth = 2.5; c.strokeRect(2 * gap + s + (cx - 1) * cs, oy + (cy - 1) * cs, 3 * cs, 3 * cs);
      c.strokeStyle = '#fff'; c.strokeRect(3 * gap + 2 * s + cx * cs, oy + cy * cs, cs, cs); c.lineWidth = 1;
      const f = filt(cx, cy), nm = ['R', 'G', 'B'], i = cy * n + cx, arr = [im.r, im.g, im.b];
      const lines = [];
      for (let ch = 0; ch < 3; ch++) {
        if (ch === f) { lines.push(`${nm[ch]}: measured directly = <b>${Math.round(rec[ch][i])}</b>`); continue; }
        const vals = []; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = cx + dx, Y = cy + dy; if (X >= 0 && Y >= 0 && X < n && Y < n && filt(X, Y) === ch) vals.push(Math.round(arr[ch][Y * n + X])); }
        lines.push(`${nm[ch]}: mean of ${vals.length} neighbours (${vals.join(', ')}) = <b class="o">${Math.round(rec[ch][i])}</b>`);
      }
      const finished = k >= n * n;
      q('#out').innerHTML = `pixel ${Math.min(k, n * n)} of ${n * n}: (x=${cx}, y=${cy}) behind a <b>${nm[f]}</b> filter\n${lines.join('\n')}\ntrue colour = (${Math.round(im.r[i])}, ${Math.round(im.g[i])}, ${Math.round(im.b[i])})\n<span class="m">mean abs. error when finished: ${AI.fmt(im.err, 1)} of 255</span>`;
      q('#cap').textContent = finished ? 'Done. Two thirds of all colour numbers were guessed — smooth areas come out right, sharp edges and fine patterns (try the checkerboard) get colour fringes.' : 'The yellow 3×3 window slides over the raw mosaic. The missing colours of the centre pixel are averages of the neighbours that DID measure that colour.';
    }, { fps: 30 });
  };

  /* ================= cvjpeg ================= */
  AI.labs.cvjpeg = function (root) {
    const QB = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]].flat();
    const TEXT = [52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144, 104, 66, 73, 63, 58, 71, 122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79, 65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61, 65, 83, 87, 79, 69, 68, 65, 76, 78, 94];
    const TM = new Float64Array(64); for (let u = 0; u < 8; u++) for (let x = 0; x < 8; x++) TM[u * 8 + x] = (u ? 0.5 : Math.SQRT1_2 / 2) * Math.cos(((2 * x + 1) * u * Math.PI) / 16);
    const mm = (A, B, tA, tB) => { const R = new Float64Array(64); for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { let s = 0; for (let k = 0; k < 8; k++) s += (tA ? A[k * 8 + i] : A[i * 8 + k]) * (tB ? B[j * 8 + k] : B[k * 8 + j]); R[i * 8 + j] = s; } return R; };
    const dct = (b) => mm(mm(TM, b, 0, 0), TM, 0, 1), idct = (F) => mm(mm(TM, F, 1, 0), TM, 0, 0);
    const qtab = (qual) => { const sc = qual < 50 ? 5000 / qual : 200 - 2 * qual; return QB.map((v) => clamp(Math.floor((v * sc + 50) / 100), 1, 255)); };
    const STAGES = ['① pixels', '② minus 128', '③ DCT', '④ ÷Q, round', '⑤ ×Q', '⑥ inverse DCT'];
    const { ctl } = AI.labShell(root, 'JPEG pipeline on one 8×8 block', 'pixels → DCT → quantise → reconstruct — change the quality and see what is thrown away',
      '<canvas id="cv" style="height:430px;cursor:pointer;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Image <select id="scene">${sceneOpts(['house', 'shapes', 'edges', 'digit', 'checker'])}</select></label>
       <div class="row"><button class="btn small" id="text">Use textbook block</button></div>
       <label>JPEG quality <span class="v" id="qv"></span><input type="range" id="qual" min="1" max="100" value="50"></label>
       <div>Stage<div class="chips" id="stg" style="margin-top:4px"></div></div>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="2" step="0.05" value="0.6"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next stage ▶</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    const S = 120, NBK = 15;
    let gray, comp = new Uint8Array(S * S), blk = Float64Array.from(TEXT), blkPos = null, stage = 0, ts = 0, playing = true, last = '', qt = null, qual = 50, imgDirty = true, geo = null, cached = null;
    const stageBtns = STAGES.map((t, i) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = t; b.onclick = () => { stage = i; ts = 0; last = ''; }; q('#stg').appendChild(b); return b; });
    const imgs = [0, 1].map(() => { const c = document.createElement('canvas'); c.width = c.height = S; return c; });
    const loadScene = () => { gray = CV.gray(CV.resample(q('#scene').value, S)); imgDirty = true; blkPos = null; blk = Float64Array.from(TEXT); ts = 0; last = ''; };
    const compress = () => {
      qt = qtab(qual); let se = 0, nz = 0;
      for (let by = 0; by < NBK; by++) for (let bx = 0; bx < NBK; bx++) {
        const b = new Float64Array(64); for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) b[y * 8 + x] = gray[(by * 8 + y) * S + bx * 8 + x] - 128;
        const F = dct(b); for (let i = 0; i < 64; i++) { const qv = Math.round(F[i] / qt[i]); if (qv) nz++; F[i] = qv * qt[i]; }
        const R = idct(F); for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { const o = (by * 8 + y) * S + bx * 8 + x, v = clamp(Math.round(R[y * 8 + x] + 128), 0, 255); comp[o] = v; se += (v - gray[o]) ** 2; }
      }
      cached = { psnr: 10 * Math.log10(255 * 255 / Math.max(se / (S * S), 1e-9)), nzFrac: nz / (NBK * NBK * 64) };
      [gray, comp].forEach((a, j) => { const g = imgs[j].getContext('2d'), id = g.createImageData(S, S); for (let i = 0; i < S * S; i++) { const v = a[i]; id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255; } g.putImageData(id, 0, 0); });
      imgDirty = false;
    };
    const stageData = () => {
      const Qt = qt, sh = Float64Array.from(blk, (v) => v - 128), F = dct(sh), qn = F.map((v, i) => Math.round(v / Qt[i])), dq = qn.map((v, i) => v * Qt[i]), R = idct(dq).map((v) => clamp(Math.round(v + 128), 0, 255));
      return [blk, sh, F, qn, dq, R];
    };
    q('#scene').onchange = loadScene;
    q('#text').onclick = () => { blk = Float64Array.from(TEXT); blkPos = null; ts = 0; last = ''; };
    q('#qual').oninput = () => { qual = +q('#qual').value; imgDirty = true; last = ''; };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { stage = (stage + 1) % 6; ts = 0; last = ''; };
    q('#cv').addEventListener('pointerdown', (e) => {
      if (!geo) return; const r = q('#cv').getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const I = geo.img; if (mx >= I.x && mx < I.x + I.s && my >= I.y && my < I.y + I.s) { const bx = clamp(Math.floor((mx - I.x) / I.s * NBK), 0, NBK - 1), by = clamp(Math.floor((my - I.y) / I.s * NBK), 0, NBK - 1); blkPos = { bx, by }; blk = new Float64Array(64); for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = Math.round(gray[(by * 8 + y) * S + bx * 8 + x]); ts = 0; last = ''; return; }
      const G = geo.grid; if ((stage === 0 || stage === 1) && mx >= G.x && my >= G.y && mx < G.x + 8 * G.cs && my < G.y + 8 * G.cs) { const i = Math.floor((my - G.y) / G.cs) * 8 + Math.floor((mx - G.x) / G.cs); blk[i] = (blk[i] + 32) % 256; ts = 0; last = ''; }
    });
    loadScene();
    const DUR = 4.5;
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp.toFixed(2) + '×'; q('#qv').textContent = qual;
      if (playing) { ts += dt * sp; if (ts > DUR) { stage = (stage + 1) % 6; ts = 0; } } else ts = Math.max(ts, DUR * 0.6);
      if (imgDirty) compress();
      const rev = clamp(ts / (DUR * 0.5), 0, 1), key = [stage, Math.round(rev * 64), qual, blk.join('').length, blkPos && blkPos.bx + ',' + blkPos.by, cv.w, cv.h, themeKey(), q('#scene').value, Array.from(blk).reduce((a, b) => a + b, 0)].join();
      if (key === last) return; last = key;
      const c = cv.ctx, C = AI.col(); c.clearRect(0, 0, cv.w, cv.h);
      const data = stageData()[stage], Qt = qt;
      stageBtns.forEach((b, i) => (b.style.borderColor = i === stage ? 'var(--blue)' : ''));
      const lw = Math.min(190, cv.w * 0.33), is = lw - 10, grid = { x: lw + 18, y: 26, cs: Math.min((cv.w - lw - 30) / 8, (cv.h - 48) / 8) };
      geo = { img: { x: 6, y: 20, s: is }, grid };
      c.font = '11px ui-monospace,Menlo,Consolas,monospace'; c.textBaseline = 'middle'; c.textAlign = 'left'; c.fillStyle = C.muted;
      c.fillText('original', 6, 10); c.fillText(`after JPEG q=${qual}`, 6, 20 + is + 20 + 12 - 10 + 0);
      c.imageSmoothingEnabled = false; c.drawImage(imgs[0], 6, 20, is, is); c.drawImage(imgs[1], 6, 20 + is + 34 - 4 - 0, is, is);
      if (blkPos) { c.strokeStyle = '#ffe14d'; c.lineWidth = 2; const b = is / NBK; c.strokeRect(6 + blkPos.bx * b, 20 + blkPos.by * b, b, b); c.strokeRect(6 + blkPos.bx * b, 20 + is + 30 + blkPos.by * b, b, b); c.lineWidth = 1; }
      c.fillStyle = C.fg; c.textAlign = 'left'; c.fillText(`${STAGES[stage]}${blkPos ? `   (block ${blkPos.bx},${blkPos.by})` : '   (textbook block)'}`, grid.x, 12);
      const mxv = Math.max(1, ...Array.from(data, Math.abs));
      c.textAlign = 'center';
      for (let i = 0; i < 64; i++) {
        const x = grid.x + (i % 8) * grid.cs, y = grid.y + Math.floor(i / 8) * grid.cs, shown = i < rev * 64, v = data[i];
        c.globalAlpha = shown ? 1 : 0.12;
        const frac = stage === 0 || stage === 5 ? v / 255 : Math.abs(v) / mxv;
        if (stage === 0 || stage === 5) c.fillStyle = `rgb(${v},${v},${v})`; else c.fillStyle = v >= 0 ? `rgba(255,155,61,${Math.min(1, Math.sqrt(frac)) * 0.85})` : `rgba(62,99,221,${Math.min(1, Math.sqrt(frac)) * 0.85})`;
        c.fillRect(x, y, grid.cs - 1, grid.cs - 1);
        const lum = stage === 0 || stage === 5 ? v : 128; c.fillStyle = stage === 0 || stage === 5 ? (lum > 128 ? '#000' : '#fff') : C.fg;
        c.font = `${Math.max(9, Math.min(14, grid.cs / 2.9))}px ui-monospace,Menlo,Consolas,monospace`;
        c.fillText(stage === 2 ? String(Math.round(v)) : String(Math.round(v)), x + grid.cs / 2, y + grid.cs / 2);
        if (stage === 3 && grid.cs > 30) { c.globalAlpha *= 0.7; c.fillStyle = C.muted; c.font = '9px ui-monospace,Menlo,Consolas,monospace'; c.fillText('/' + Qt[i], x + grid.cs - 12, y + 8); }
        c.globalAlpha = 1;
      }
      const F = stageData(); const nz = F[3].reduce((a, v) => a + (v ? 1 : 0), 0), errs = F[5].map((v, i) => v - blk[i]);
      const mae = errs.reduce((a, v) => a + Math.abs(v), 0) / 64;
      q('#cap').textContent = ['Stage ① — the block as the camera stored it: 64 brightness values. (Click a cell to change a pixel by +32, or click the image to pick another block.)', 'Stage ② — subtract 128 so values are centred on 0 (like a signal that swings up and down).', 'Stage ③ — the 2-D DCT: 64 new numbers saying "how much of each wave pattern is in this block". Top-left = average brightness; further right/down = finer detail.', `Stage ④ — divide each coefficient by the Q-table entry and round. Most high-frequency numbers become 0. Only ${nz} of 64 survive: this is the lossy step.`, 'Stage ⑤ — multiply back by Q. The rounded-off information is gone for good.', 'Stage ⑥ — inverse DCT and +128: the reconstructed block. Compare with ① — close, but not identical.'][stage];
      q('#out').innerHTML = `quality ${qual}  →  Q scaled by ${qual < 50 ? AI.fmt(5000 / qual / 100, 2) : AI.fmt((200 - 2 * qual) / 100, 2)}\nnon-zero coefficients (block): <b class="o">${nz} / 64</b>\nmean |error| (block): <b>${AI.fmt(mae, 1)}</b> grey levels\nwhole image: ${AI.fmt(cached.nzFrac * 100, 0)}% non-zero, PSNR <b>${AI.fmt(cached.psnr, 1)} dB</b>\n<span class="m">${nz === 64 ? '' : 'zeros are cheap to store (run-length + Huffman)'}</span>`;
    }, { fps: 30 });
  };
})();
