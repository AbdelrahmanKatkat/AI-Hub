/* Computer-vision labs: pixels (zoom into numbers), convolve (animated sliding kernel), archviz (architecture animation).
 * Requires assets/lib.js and assets/arch.js.
 *
 *   <div data-lab="pixels"></div>
 *   <div data-lab="convolve" data-kernel="blur"></div>
 *   <div data-lab="archviz" data-preset="resnet18" data-presets="resnet18,resnet50" data-edit data-resolution></div>
 *   <div data-lab="archviz"><script type="text/plain">input 3 32 32 …</script></div>        (inline spec)
 */
(function () {
  const CV = (AI.cv = {});

  /* ---------- procedural test scenes (120x120 RGB, cached) ---------- */
  const BASE = 120, cache = {};
  CV.SCENES = { shapes: 'shapes', house: 'house', digit: 'handwritten 7', edges: 'edges & stripes', gradient: 'gradient', checker: 'checkerboard' };
  CV.scene = function (kind) {
    if (cache[kind]) return cache[kind];
    const c = document.createElement('canvas'); c.width = c.height = BASE;
    const g = c.getContext('2d', { willReadFrequently: true });
    const lin = (x0, y0, x1, y1, a, b) => { const gr = g.createLinearGradient(x0, y0, x1, y1); gr.addColorStop(0, a); gr.addColorStop(1, b); return gr; };
    if (kind === 'shapes') {
      g.fillStyle = lin(0, 0, 0, BASE, '#cfe3f5', '#f4efe1'); g.fillRect(0, 0, BASE, BASE);
      g.fillStyle = '#d8402f'; g.beginPath(); g.arc(38, 40, 22, 0, 7); g.fill();
      g.fillStyle = '#2f6fd8'; g.fillRect(66, 62, 38, 38);
      g.fillStyle = '#2e9d4f'; g.beginPath(); g.moveTo(26, 108); g.lineTo(54, 108); g.lineTo(40, 74); g.closePath(); g.fill();
      g.fillStyle = '#f2c230'; g.beginPath(); g.arc(94, 24, 11, 0, 7); g.fill();
    } else if (kind === 'house') {
      g.fillStyle = lin(0, 0, 0, 80, '#7fb6ee', '#d6ecff'); g.fillRect(0, 0, BASE, BASE);
      g.fillStyle = '#4c9a3d'; g.fillRect(0, 84, BASE, 36);
      g.fillStyle = '#c98b52'; g.fillRect(30, 50, 60, 42);
      g.fillStyle = '#a03a2a'; g.beginPath(); g.moveTo(24, 52); g.lineTo(60, 22); g.lineTo(96, 52); g.closePath(); g.fill();
      g.fillStyle = '#5a3a22'; g.fillRect(52, 66, 14, 26);
      g.fillStyle = '#f7e9a0'; g.fillRect(36, 58, 12, 12); g.fillRect(72, 58, 12, 12);
      g.fillStyle = '#ffd23f'; g.beginPath(); g.arc(104, 18, 9, 0, 7); g.fill();
    } else if (kind === 'digit') {
      g.fillStyle = '#000'; g.fillRect(0, 0, BASE, BASE);
      g.strokeStyle = '#fff'; g.lineWidth = 11; g.lineCap = 'round'; g.lineJoin = 'round';
      g.beginPath(); g.moveTo(30, 30); g.lineTo(90, 28); g.lineTo(52, 96); g.stroke();
      g.lineWidth = 6; g.beginPath(); g.moveTo(44, 62); g.lineTo(80, 60); g.stroke();
    } else if (kind === 'edges') {
      g.fillStyle = '#202020'; g.fillRect(0, 0, BASE, BASE);
      g.fillStyle = '#f0f0f0'; g.fillRect(12, 12, 40, 96);
      g.fillStyle = '#9a9a9a'; g.fillRect(52, 12, 30, 96);
      g.fillStyle = '#f0f0f0'; g.beginPath(); g.moveTo(82, 108); g.lineTo(108, 12); g.lineTo(120, 12); g.lineTo(120, 108); g.closePath(); g.fill();
    } else if (kind === 'gradient') {
      g.fillStyle = lin(0, 0, BASE, 0, '#000', '#fff'); g.fillRect(0, 0, BASE, BASE);
    } else { // checker
      for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { g.fillStyle = (i + j) % 2 ? '#101010' : '#f2f2f2'; g.fillRect(i * 15, j * 15, 15, 15); }
    }
    return (cache[kind] = g.getImageData(0, 0, BASE, BASE).data);
  };
  /** Block-average the scene to n x n; returns {r,g,b} Float32Arrays (0..255). n must divide 120. */
  CV.resample = function (kind, n) {
    const d = CV.scene(kind), f = BASE / n, r = new Float32Array(n * n), g = new Float32Array(n * n), b = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let sr = 0, sg = 0, sb = 0;
      for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) { const k = ((y * f + j) * BASE + x * f + i) * 4; sr += d[k]; sg += d[k + 1]; sb += d[k + 2]; }
      const m = f * f; r[y * n + x] = sr / m; g[y * n + x] = sg / m; b[y * n + x] = sb / m;
    }
    return { r, g, b, n };
  };
  CV.gray = (im) => im.r.map((v, i) => 0.299 * v + 0.587 * im.g[i] + 0.114 * im.b[i]);

  const chip = (parent, label, on) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = on; parent.appendChild(b); return b; };

  /* ================= pixels ================= */
  AI.labs.pixels = function (root) {
    const SIZES = [4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120];
    const { view, ctl } = AI.labShell(root, 'Zoom into the pixels', 'every image is a grid of numbers — drag the window, change the resolution and bit depth',
      '<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:10px"><canvas id="full" style="height:300px;cursor:crosshair;touch-action:none"></canvas><canvas id="zoom" style="height:300px"></canvas></div><div class="caption">Left: the whole image. Drag the white window. Right: those pixels magnified, with the number the computer stores for each.</div>',
      `<label>Image <select id="scene">${Object.entries(CV.SCENES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></label>
       <div>Show <div class="row" id="mode"></div></div>
       <label>Resolution: <b id="resv"></b> pixels across<input type="range" id="res" min="0" max="${SIZES.length - 1}" value="${SIZES.indexOf(30)}"></label>
       <label>Bit depth: <b id="bitv"></b><input type="range" id="bits" min="1" max="8" value="8"></label>
       <label>Window size <span class="v" id="winv"></span><input type="range" id="win" min="3" max="10" value="6"></label>
       <div class="row"><button class="btn primary" id="zoomanim">▶ Animate: photo → pixels</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cf = AI.canvas(q('#full')), cz = AI.canvas(q('#zoom'));
    let mode = 'gray', win = { x: 5, y: 6 }, drag = false, anim = null, last = '';
    const modeBox = q('#mode'); const modes = { gray: 'Gray', rgb: 'RGB', r: 'R', g: 'G', b: 'B' };
    const modeBtns = Object.entries(modes).map(([k, v]) => chip(modeBox, v, () => { mode = k; last = ''; paint(); }));
    const paint = () => modeBtns.forEach((b, i) => { b.style.borderColor = Object.keys(modes)[i] === mode ? 'var(--blue)' : ''; });
    paint();
    q('#zoomanim').onclick = () => { q('#res').value = SIZES.length - 1; q('#win').value = 5; anim = { t: 0 }; };
    const place = (e) => { const r = q('#full').getBoundingClientRect(), n = SIZES[+q('#res').value], w = +q('#win').value, s = Math.min(r.width, r.height); win = { x: AI.clamp(Math.round(((e.clientX - r.left) / s) * n - w / 2), 0, n - w), y: AI.clamp(Math.round(((e.clientY - r.top) / s) * n - w / 2), 0, n - w) }; };
    q('#full').addEventListener('pointerdown', (e) => { drag = true; q('#full').setPointerCapture(e.pointerId); place(e); });
    q('#full').addEventListener('pointermove', (e) => drag && place(e));
    q('#full').addEventListener('pointerup', () => (drag = false));
    AI.loop(root, (dt) => {
      if (anim) { anim.t += dt * 0.35; const k = Math.min(SIZES.length - 1, SIZES.length - 1 - Math.floor(anim.t * (SIZES.length - 4))); q('#res').value = Math.max(3, k); if (anim.t > 1.05) anim = null; }
      const n = SIZES[+q('#res').value], bits = +q('#bits').value, w = Math.min(+q('#win').value, n), sc = q('#scene').value;
      q('#resv').textContent = `${n}×${n}`; q('#bitv').textContent = `${bits} bit → ${2 ** bits} levels`; q('#winv').textContent = `${w}×${w}`;
      win.x = AI.clamp(win.x, 0, n - w); win.y = AI.clamp(win.y, 0, n - w);
      const key = [n, bits, w, sc, mode, win.x, win.y, cf.w, cz.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const im = CV.resample(sc, n), step = 255 / (2 ** bits - 1), qz = (v) => Math.round(v / step) * step;
      const chan = { r: im.r, g: im.g, b: im.b, gray: CV.gray(im) }, C = AI.col();
      const val = (arr, i) => Math.round(qz(arr[i]));
      // full image
      let g = cf.ctx; g.clearRect(0, 0, cf.w, cf.h);
      const s = Math.min(cf.w, cf.h), px = s / n;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x;
        g.fillStyle = mode === 'rgb' ? `rgb(${val(im.r, i)},${val(im.g, i)},${val(im.b, i)})` : (() => { const v = val(chan[mode === 'gray' ? 'gray' : mode], i); return `rgb(${v},${v},${v})`; })();
        g.fillRect(Math.floor(x * px), Math.floor(y * px), Math.ceil(px), Math.ceil(px));
      }
      g.strokeStyle = '#fff'; g.lineWidth = 2.5; g.strokeRect(win.x * px, win.y * px, w * px, w * px);
      g.strokeStyle = '#000'; g.lineWidth = 1; g.strokeRect(win.x * px - 1.5, win.y * px - 1.5, w * px + 3, w * px + 3);
      // magnified numbers
      g = cz.ctx; g.clearRect(0, 0, cz.w, cz.h);
      const cell = Math.min(cz.w, cz.h) / w, fs = Math.max(8, Math.min(15, cell / (mode === 'rgb' ? 3.4 : 2.4)));
      g.textAlign = 'center'; g.textBaseline = 'middle'; g.font = `${fs}px ui-monospace,Menlo,Consolas,monospace`;
      for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) {
        const i = (win.y + y) * n + win.x + x, X = x * cell, Y = y * cell;
        const rgbv = [val(im.r, i), val(im.g, i), val(im.b, i)], gv = val(chan.gray, i), cv = mode === 'rgb' ? rgbv : mode === 'gray' ? [gv, gv, gv] : (() => { const v = val(chan[mode], i); return [v, v, v]; })();
        g.fillStyle = `rgb(${cv[0]},${cv[1]},${cv[2]})`; g.fillRect(X, Y, cell, cell);
        const lum = 0.299 * cv[0] + 0.587 * cv[1] + 0.114 * cv[2]; g.fillStyle = lum > 128 ? '#000' : '#fff';
        if (cell >= 22) {
          if (mode === 'rgb') { g.fillText(rgbv[0], X + cell / 2, Y + cell * 0.25); g.fillText(rgbv[1], X + cell / 2, Y + cell * 0.5); g.fillText(rgbv[2], X + cell / 2, Y + cell * 0.75); }
          else g.fillText(mode === 'gray' ? gv : val(chan[mode], i), X + cell / 2, Y + cell / 2);
        }
        g.strokeStyle = 'rgba(128,128,128,.5)'; g.strokeRect(X, Y, cell, cell);
      }
      const total = n * n * (mode === 'rgb' ? 3 : 1);
      q('#out').innerHTML = `image: <b>${n}×${n}</b> = ${n * n} pixels${mode === 'rgb' ? ' × 3 channels' : ''}\nnumbers stored: <b class="o">${total}</b>\nbits needed: <b>${total * bits}</b>  (${AI.fmt(total * bits / 8, 1)} bytes)\n<span class="m">${mode === 'rgb' ? 'each pixel = (R, G, B), top to bottom in each cell' : 'each cell = brightness 0 (black) … 255 (white)'}</span>`;
    }, { fps: 30 });
  };

  /* ================= convolve ================= */
  AI.labs.convolve = function (root) {
    const KERNELS = {
      identity: [0, 0, 0, 0, 1, 0, 0, 0, 0], 'box blur': [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9), gaussian: [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16),
      sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0], 'sobel x': [-1, 0, 1, -2, 0, 2, -1, 0, 1], 'sobel y': [-1, -2, -1, 0, 0, 0, 1, 2, 1], laplacian: [0, 1, 0, 1, -4, 1, 0, 1, 0], emboss: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
    };
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, 'Convolution: a kernel sliding over an image', 'edit the 3×3 numbers; watch one output pixel built per step',
      '<canvas style="height:380px"></canvas><div class="caption" id="cap"></div>',
      `<label>Image <select id="scene">${['digit', 'shapes', 'edges', 'house', 'checker'].map((k) => `<option value="${k}">${CV.SCENES[k]}</option>`).join('')}</select></label>
       <div>Kernel presets<div class="chips" id="pre" style="margin-top:4px"></div></div>
       <div>Kernel weights (editable)<div id="kern" style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px"></div></div>
       <label>Image size <span class="v" id="nv"></span><input type="range" id="n" min="0" max="5" value="2"></label>
       <div class="row"><label style="flex-direction:row;gap:5px;align-items:center"><input type="checkbox" id="pad"> zero padding (same size)</label><label style="flex-direction:row;gap:5px;align-items:center">stride <select id="stride" style="width:auto"><option>1</option><option>2</option></select></label></div>
       <label>Speed <span class="v">windows / second</span><input type="range" id="sp" min="1" max="40" value="3"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Restart</button></div>
       <div class="readout" id="out" style="font-size:12px"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), tick = AI.ticker();
    const NS = [8, 10, 12, 15, 20, 24];
    let K = KERNELS[d.kernel] || KERNELS['box blur'], img = null, out = [], pos = 0, playing = true, N = 12, P = 0, S = 1, O = 0, dirty = true;
    const kin = [...Array(9)].map((_, i) => { const e = document.createElement('input'); e.type = 'number'; e.step = 0.1; e.style.cssText = 'text-align:center'; e.oninput = () => { K[i] = +e.value || 0; recompute(); }; q('#kern').appendChild(e); return e; });
    const setK = (k) => { K = [...k]; K.forEach((v, i) => (kin[i].value = +v.toFixed(3))); recompute(); };
    Object.entries(KERNELS).forEach(([n, k]) => chip(q('#pre'), n, () => { setK(k); pos = 0; }));
    const load = () => {
      N = NS[+q('#n').value]; q('#nv').textContent = `${N}×${N}`;
      const im = CV.resample(q('#scene').value, N); img = CV.gray(im); P = q('#pad').checked ? 1 : 0; S = +q('#stride').value;
      O = Math.floor((N + 2 * P - 3) / S) + 1; recompute(); pos = 0;
    };
    const px = (x, y) => (x < 0 || y < 0 || x >= N || y >= N ? 0 : img[y * N + x]);
    function recompute() {
      out = [];
      for (let oy = 0; oy < O; oy++) for (let ox = 0; ox < O; ox++) { let s = 0; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s += K[j * 3 + i] * px(ox * S - P + i, oy * S - P + j); out.push(s); }
      dirty = true;
    }
    ['#scene', '#n', '#pad', '#stride'].forEach((s) => q(s).addEventListener('input', load));
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; pos = (pos + 1) % (O * O + 1); };
    q('#reset').onclick = () => (pos = 0);
    setK(K); load();

    AI.loop(root, (dt) => {
      if (playing) { pos += tick(dt, +q('#sp').value); if (pos > O * O + 3) pos = 0; }
      const t = Math.min(O * O, Math.floor(pos)), C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const sumK = K.reduce((a, b) => a + b, 0), norm = Math.abs(sumK - 1) < 0.05 ? 'clamp' : 'center';
      const maxAbs = Math.max(1e-6, ...out.map(Math.abs)), shade = (v) => (norm === 'clamp' ? AI.clamp(v, 0, 255) : AI.clamp(128 + (127 * v) / maxAbs, 0, 255));
      const gap = 26, kw = 96, W = c.w, avail = W - kw - gap * 4;
      const inCells = N + 2 * P, cs = Math.min(avail * 0.52 / inCells, (c.h - 40) / inCells), os = Math.min(avail * 0.4 / O, (c.h - 40) / O);
      const x0 = gap, y0 = 28, kx = x0 + inCells * cs + gap, ox0 = kx + kw + gap;
      g.font = '12px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText(`input ${N}×${N}${P ? ' + zero padding' : ''}`, x0, 16); g.fillText('kernel', kx, 16); g.fillText(`output ${O}×${O}`, ox0, 16);
      const showNum = cs >= 24, showNumO = os >= 24;
      g.textAlign = 'center'; g.textBaseline = 'middle'; g.font = `${Math.min(11, cs / 2.4)}px ui-monospace,Menlo,monospace`;
      for (let y = -P; y < N + P; y++) for (let x = -P; x < N + P; x++) {
        const v = px(x, y), X = x0 + (x + P) * cs, Y = y0 + (y + P) * cs, pad = x < 0 || y < 0 || x >= N || y >= N;
        g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(X, Y, cs, cs);
        if (pad) { g.strokeStyle = C.muted; g.setLineDash([2, 2]); g.strokeRect(X + 0.5, Y + 0.5, cs - 1, cs - 1); g.setLineDash([]); }
        if (showNum) { g.fillStyle = v > 128 ? '#000' : '#fff'; g.fillText(Math.round(v), X + cs / 2, Y + cs / 2); }
      }
      // current window
      const cur = Math.min(t, O * O - 1), oy = Math.floor(cur / O), ox = cur % O;
      if (O * O > 0) {
        const wx = x0 + (ox * S) * cs, wy = y0 + (oy * S) * cs;
        g.fillStyle = C.blue; g.globalAlpha = 0.22; g.fillRect(wx, wy, 3 * cs, 3 * cs); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 3; g.strokeRect(wx, wy, 3 * cs, 3 * cs);
        // kernel panel
        const ks = kw / 3;
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
          const w = K[j * 3 + i], X = kx + i * ks, Y = y0 + j * ks;
          g.fillStyle = w > 0 ? AI.mix(C.bg2, C.green, Math.min(1, Math.abs(w) / 2)) : w < 0 ? AI.mix(C.bg2, C.orange, Math.min(1, Math.abs(w) / 2)) : C.bg2; g.fillRect(X, Y, ks, ks);
          g.strokeStyle = C.muted; g.lineWidth = 1; g.strokeRect(X, Y, ks, ks);
          g.fillStyle = C.fg; g.font = '12px ui-monospace,Menlo,monospace'; g.fillText(AI.fmt(w, 2), X + ks / 2, Y + ks / 2);
        }
        // output panel (revealed so far)
        for (let k = 0; k < O * O; k++) {
          const X = ox0 + (k % O) * os, Y = y0 + Math.floor(k / O) * os;
          if (k < t) { const v = shade(out[k]); g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(X, Y, os, os); if (showNumO) { g.fillStyle = v > 128 ? '#000' : '#fff'; g.font = `${Math.min(11, os / 2.4)}px ui-monospace,Menlo,monospace`; g.fillText(Math.round(out[k]), X + os / 2, Y + os / 2); } }
          else { g.strokeStyle = C.border; g.strokeRect(X + 0.5, Y + 0.5, os - 1, os - 1); }
        }
        g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(ox0 + ox * os, y0 + oy * os, os, os);
        // arrow window -> output
        g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.lineWidth = 1; g.beginPath(); g.moveTo(wx + 3 * cs, wy + 1.5 * cs); g.lineTo(kx, y0 + 1.5 * ks); g.moveTo(kx + kw, y0 + 1.5 * ks); g.lineTo(ox0 + ox * os, y0 + oy * os + os / 2); g.stroke(); g.setLineDash([]);
        // readout
        const terms = []; let s = 0;
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) { const p = px(ox * S - P + i, oy * S - P + j), w = K[j * 3 + i]; terms.push(`${AI.fmt(w, 2)}×${Math.round(p)}`); s += w * p; }
        q('#out').innerHTML = `<span class="m">output pixel (${ox}, ${oy})</span>\n${terms.slice(0, 3).join(' + ')}\n+ ${terms.slice(3, 6).join(' + ')}\n+ ${terms.slice(6).join(' + ')}\n= <b class="o">${AI.fmt(s, 2)}</b>${norm === 'center' ? '\n<span class="m">grey 128 = zero response; brighter = positive</span>' : ''}`;
        q('#cap').textContent = `Step ${Math.min(t + 1, O * O)} of ${O * O}: multiply the 9 pixels under the blue window by the 9 kernel weights, add them up → one output pixel. Then slide ${S === 2 ? 'two pixels' : 'one pixel'} and repeat.`;
      }
    }, { fps: 30 });
  };

  /* ================= architecture visualiser ================= */
  const TYPE_COLOR = { input: 'muted', conv: 'blue', dwconv: 'teal', maxpool: 'red', avgpool: 'red', gap: 'red', flatten: 'muted', fc: 'orange', res: 'purple', bottleneck: 'purple', dwsep: 'teal', invres: 'teal', dense: 'pink', up: 'green', concat: 'green', add: 'green', patch: 'pink', cls: 'pink', pos: 'pink', norm: 'muted', tblock: 'pink', seqfc: 'orange' };
  const EXPLAIN = {
    input: 'The raw image tensor enters: channels × height × width.',
    conv: 'Slides learned filters over the image; each filter produces one feature map. More filters = more channels.',
    dwconv: 'One filter per channel (no mixing between channels) — cheap spatial filtering.',
    maxpool: 'Keeps the strongest response in each window: halves height and width, adds a little shift tolerance.',
    avgpool: 'Averages each window: smoothly downsamples the feature maps.',
    gap: 'Global average pooling: collapses each feature map to a single number — one number per channel.',
    flatten: 'Unrolls the tensor into one long vector so fully connected layers can read it.',
    fc: 'Fully connected: every input number connects to every output number (a big matrix multiply).',
    res: 'Residual block: output = F(x) + x. The skip path lets gradients and information flow straight through, so very deep nets train.',
    bottleneck: 'Bottleneck residual block: 1×1 reduces channels, 3×3 does the spatial work cheaply, 1×1 restores channels; skip adds the input back.',
    dwsep: 'Depthwise-separable: spatial filtering per channel, then a 1×1 conv to mix channels — ~8–9× cheaper than a full 3×3 conv.',
    invres: 'Inverted residual: expand channels with 1×1, depthwise 3×3, then project back down (linear). Skip when shapes match.',
    dense: 'Dense block: every layer receives the concatenation of all earlier feature maps, so features are reused instead of recomputed.',
    up: 'Transposed convolution: learns to double height and width (the decoder grows the image back).',
    concat: 'Skip connection by concatenation: copies high-resolution detail from the encoder and stacks it channel-wise.',
    add: 'Skip connection by addition.',
    patch: 'Cuts the image into patches and linearly projects each to a vector: an image becomes a sequence of tokens.',
    cls: 'Adds a learnable class token whose final state summarises the image.',
    pos: 'Adds position embeddings so the transformer knows where each patch came from.',
    norm: 'Layer normalisation.',
    tblock: 'Transformer encoder blocks: self-attention lets every patch look at every other patch, then an MLP processes each token.',
    seqfc: 'Classification head applied to the class token.',
  };
  const fmtN = (v) => (v >= 1e9 ? (v / 1e9).toFixed(2) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' K' : String(Math.round(v)));

  AI.labs.archviz = function (root) {
    const d = root.dataset, inline = AI.$('script[type="text/plain"]', root), presetKeys = (d.presets || '').split(',').map((s) => s.trim()).filter(Boolean);
    const P = AIArch.PRESETS;
    let spec = inline ? inline.textContent.trim() : (P[d.preset] || P['small-cnn']).spec;
    root.classList.add('lab');
    root.innerHTML = `<div class="lab-head"><i class="live-dot"></i><b>${d.title || 'Architecture explorer'}</b><span>a data pulse flows through the network — click any block to inspect it</span></div>
      <div class="arch-bar">
        ${presetKeys.length ? `<label>Model <select id="preset">${presetKeys.map((k) => `<option value="${k}" ${k === d.preset ? 'selected' : ''}>${P[k].title}</option>`).join('')}</select></label>` : ''}
        ${d.resolution !== undefined ? `<label>Input resolution <b id="resv"></b><input type="range" id="res" min="32" max="512" step="32" value="${d.res || 224}"></label>` : ''}
        <label>Speed<input type="range" id="sp" min="0.1" max="1.5" step="0.05" value="0.4"></label>
        <button class="btn small" id="prev">◀ Prev</button><button class="btn small primary" id="play">⏸ Pause</button><button class="btn small" id="next">Next ▶</button>
      </div>
      <div class="arch-scroll"><canvas style="height:${d.height || 330}px"></canvas></div>
      <div class="caption arch-cap" id="cap"></div>
      <div class="arch-bottom"><div id="detail"></div><div><div class="readout" id="sum"></div>${d.edit !== undefined ? `<label style="display:block;margin-top:8px;font-size:12px;color:var(--muted)">Edit the architecture (one layer per line)<textarea id="spec" rows="12" spellcheck="false">${spec.replace(/</g, '&lt;')}</textarea></label><div class="err" id="err"></div>` : ''}</div></div>`;
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', root), c = AI.canvas(cv);
    let B = null, active = 0, pos = 0, playing = true, pinned = -1, hover = -1, shownKey = '', lastKey = '';
    const STEP = 92, X0 = 56;
    const build = () => {
      const res = q('#res') ? +q('#res').value : 0;
      if (q('#resv')) q('#resv').textContent = `${res}×${res}`;
      const r = AIArch.build(spec, res ? { resolution: res } : {});
      if (q('#err')) q('#err').textContent = r.errors.join('\n');
      if (!r.blocks.length) return;
      B = r; pos = Math.min(pos, B.blocks.length - 1); lastKey = '';
      cv.style.width = Math.max(AI.$('.arch-scroll', root).clientWidth, X0 * 2 + B.blocks.length * STEP) + 'px';
      showSummary();
    };
    const showSummary = () => {
      const t = B.totals, imgIn = B.blocks[0] && B.blocks[0].out;
      q('#sum').innerHTML = `<span class="m">whole network</span>\nparameters  <b class="o">${fmtN(t.params)}</b>  <span class="m">(${fmtN(t.params * 4)}B as float32)</span>\ncompute     <b class="b">${fmtN(t.macs)}</b> MACs  <span class="m">(≈ ${fmtN(t.macs * 2)}FLOPs)</span>\nactivations <b class="g">${fmtN(t.act)}</b> values  <span class="m">(sum over blocks)</span>\nlargest tensor ${fmtN(t.peak)} values\nfinal receptive field ${t.rf ? t.rf + ' px' : '—'}`;
    };
    const showDetail = (i) => {
      if (!B || i < 0 || i >= B.blocks.length) return; const b = B.blocks[i];
      q('#detail').innerHTML = `<h4 style="margin:0 0 4px">${i + 1}. ${b.name}${b.rep ? ' ×' + b.rep : ''} <small style="color:var(--muted)">${b.tags.join(' · ')}</small></h4>
        <div class="caption" style="margin:0 0 6px">${EXPLAIN[b.type] || ''}</div>
        <div class="mono" style="font-size:12px">${b.in ? AIArch.shapeStr(b.in) : '—'} <span style="color:var(--muted)">→</span> <b>${b.outStr}</b> &nbsp; params <b>${fmtN(b.params)}</b> &nbsp; MACs <b>${fmtN(b.macs)}</b></div>
        <table class="tbl" style="font-size:12px"><tr><th>layer</th><th>output</th><th>params</th><th>MACs</th><th>RF</th></tr>${b.sub.map((s) => `<tr><td>${s.name}</td><td class="mono">${s.out}</td><td>${fmtN(s.params)}</td><td>${fmtN(s.macs)}</td><td>${s.rf ?? ''}</td></tr>`).join('')}</table>`;
      q('#cap').textContent = `Block ${i + 1} of ${B.blocks.length}: ${EXPLAIN[b.type] || b.name}`;
    };
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    const go = (i) => { playing = false; q('#play').textContent = '▶ Play'; pos = AI.clamp(i, 0, B.blocks.length - 1); pinned = pos; shownKey = ''; };
    q('#prev').onclick = () => go(Math.round(pos) - 1); q('#next').onclick = () => go(Math.round(pos) + 1);
    const idxAt = (e) => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left; return B ? AI.clamp(Math.round((x - X0) / STEP), 0, B.blocks.length - 1) : -1; };
    cv.addEventListener('mousemove', (e) => (hover = idxAt(e))); cv.addEventListener('mouseleave', () => (hover = -1));
    cv.addEventListener('click', (e) => go(idxAt(e)));
    if (q('#preset')) q('#preset').onchange = () => { spec = P[q('#preset').value].spec; if (q('#spec')) q('#spec').value = spec; pos = 0; pinned = -1; build(); };
    if (q('#res')) q('#res').oninput = build;
    if (q('#spec')) { let tm; q('#spec').addEventListener('input', () => { clearTimeout(tm); tm = setTimeout(() => { spec = q('#spec').value; build(); }, 250); }); }
    build();
    new ResizeObserver(() => B && build()).observe(AI.$('.arch-scroll', root));

    AI.loop(root, (dt, time) => {
      if (!B) return;
      const n = B.blocks.length, speed = +q('#sp').value;
      if (playing) { pos += dt * speed; if (pos >= n - 0.01) pos = 0; }
      const i = Math.min(n - 1, Math.floor(pos)), f = pos - i, dwell = f < 0.55, cur = pinned >= 0 && !playing ? pinned : i;
      if (playing) pinned = -1;
      const key = [shownKey, cur].join(); if (key !== lastKey) { lastKey = key; shownKey = cur; showDetail(cur); }
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const cy = c.h * 0.46, geo = B.blocks.map((b, k) => {
        const s = b.out, h = s.k === 'img' ? 10 + 13 * Math.log2(Math.max(s.H, 1)) : s.k === 'seq' ? 14 + 10 * Math.log2(Math.max(s.N, 2)) : 14 + 8 * Math.log2(Math.max(s.D, 2));
        const t = s.k === 'img' ? 6 + 4.5 * Math.log2(Math.max(s.C, 1)) : s.k === 'seq' ? 8 + 3.5 * Math.log2(Math.max(s.D, 2)) : 8;
        return { x: X0 + k * STEP, h: Math.min(h, c.h * 0.5), t: Math.min(t, 60) };
      });
      const topY = cy - Math.max(...geo.map((q2) => q2.h)) / 2;
      // skip arcs
      B.blocks.forEach((b, k) => {
        const G = geo[k]; let a = null;
        if (b.skip === 'add' || b.skip === 'concat') a = [G.x - G.t / 2 - 22, G.x + G.t / 2 + 22, 18];
        else if (b.skip && b.skip.from !== undefined) a = [geo[b.skip.from].x, G.x, 26 + Math.min(70, (k - b.skip.from) * 7)];
        if (!a) return;
        const y0 = (b.skip && b.skip.from !== undefined) ? topY - 12 : cy - G.h / 2 - 6, h = a[2], hot = k === cur;
        g.strokeStyle = hot ? C.green : C.border; g.lineWidth = hot ? 3 : 2; g.setLineDash(b.skip && b.skip.kind === 'concat' ? [6, 4] : []); g.beginPath(); g.moveTo(a[0], y0); g.bezierCurveTo(a[0], y0 - h, a[1], y0 - h, a[1], y0); g.stroke(); g.setLineDash([]);
        g.fillStyle = hot ? C.green : C.muted; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText(b.skip === 'concat' || (b.skip && b.skip.kind === 'concat') ? 'concat' : '+', (a[0] + a[1]) / 2, y0 - h * 0.75 - 2);
        if (hot && dwell === (k === i)) { const tt = (time * 0.6) % 1, bx = AI.lerp(a[0], a[1], tt), by = y0 - h * 0.75 * Math.sin(Math.PI * tt); g.fillStyle = C.green; g.beginPath(); g.arc(bx, by, 4.5, 0, 7); g.fill(); }
      });
      // main flow line
      g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(geo[0].x, cy); g.lineTo(geo[n - 1].x, cy); g.stroke();
      // blocks
      B.blocks.forEach((b, k) => {
        const G = geo[k], col = C[TYPE_COLOR[b.type] || 'muted'], hot = k === cur, hov = k === hover, dx = 8, dy = -8;
        const x = G.x - G.t / 2, y = cy - G.h / 2;
        if (hot) { g.shadowColor = col; g.shadowBlur = 16; }
        g.fillStyle = col; g.globalAlpha = hot ? 0.95 : 0.62;
        g.fillRect(x, y, G.t, G.h);
        g.globalAlpha = hot ? 0.8 : 0.4; g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx, y + dy); g.lineTo(x + G.t + dx, y + dy); g.lineTo(x + G.t, y); g.closePath(); g.fill();
        g.beginPath(); g.moveTo(x + G.t, y); g.lineTo(x + G.t + dx, y + dy); g.lineTo(x + G.t + dx, y + G.h + dy); g.lineTo(x + G.t, y + G.h); g.closePath(); g.fill();
        g.globalAlpha = 1; g.shadowBlur = 0;
        g.strokeStyle = hot || hov ? C.fg : col; g.lineWidth = hot ? 2 : 1; g.strokeRect(x, y, G.t, G.h);
        // labels
        g.textAlign = 'center'; g.fillStyle = hot ? C.fg : C.muted; g.font = `${hot ? '600 ' : ''}11px sans-serif`;
        const ly = c.h - 62; g.fillText(b.name + (b.rep ? ' ×' + b.rep : ''), G.x, ly);
        g.font = '10.5px ui-monospace,Menlo,monospace'; g.fillStyle = hot ? C.fg : C.muted; g.fillText(b.outStr, G.x, ly + 15);
        if (b.params) { g.fillStyle = C.orange; g.fillText(fmtN(b.params), G.x, ly + 29); }
        if (b.tags.length) { g.fillStyle = C.muted; g.font = '9.5px sans-serif'; g.fillText(b.tags.filter((t) => t !== 'input').join('·'), G.x, ly - 13); }
      });
      // pulse
      const x0 = geo[i].x, x1 = geo[Math.min(n - 1, i + 1)].x, mv = f < 0.55 ? 0 : AI.smooth((f - 0.55) / 0.45), px = AI.lerp(x0, x1, mv), col = C[TYPE_COLOR[B.blocks[i].type] || 'blue'];
      g.shadowColor = C.fg; g.shadowBlur = 14; g.fillStyle = C.fg; g.beginPath(); g.arc(px, cy, dwell ? 6 + 2 * Math.sin(f * 24) : 5, 0, 7); g.fill(); g.shadowBlur = 0;
      if (playing && dwell) { /* keep detail synced with the pulse */ }
    }, { fps: 30 });
  };
})();
