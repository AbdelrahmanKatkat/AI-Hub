/* Computer Vision · Module 7 labs (segmentation & dense prediction).
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes).
 *   segpixel   per-pixel classification scan (editable threshold, noise, temperature)
 *   upsample   nearest / bilinear / transposed conv / pixel shuffle on a 3x3 map (editable kernel, overlap + checkerboards)
 *   unetanim   encoder shrinks, decoder grows, skip connections can be cut
 *   dilated    dilated-convolution receptive-field map with editable rates
 *   diceiou    paint two masks, live IoU / Dice / accuracy / cross-entropy
 *   samprompt  promptable segmentation (points, box) with slow captioned growth; 3 ambiguity masks
 *   panoptic   instance matching and panoptic quality
 *   heatpose   heatmap keypoints decoded by argmax (+ sub-pixel refinement)
 */
(function () {
  const PROTO = [
    { n: 'background', c: [225, 233, 235], col: [139, 148, 158] },
    { n: 'red disc', c: [216, 64, 47], col: [248, 81, 73] },
    { n: 'blue square', c: [47, 111, 216], col: [88, 166, 255] },
    { n: 'green triangle', c: [46, 157, 79], col: [63, 185, 80] },
    { n: 'yellow disc', c: [242, 194, 48], col: [210, 153, 34] },
  ];
  const rgbs = (a) => `rgb(${a[0]},${a[1]},${a[2]})`;
  const PLAY = (def, min, max) => `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="${min}" max="${max}" value="${def}"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Restart</button></div>`;
  function player(q, unit, onStep, onReset) {
    const st = { playing: true };
    const set = () => (q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play');
    q('#play').onclick = () => { st.playing = !st.playing; set(); };
    q('#step').onclick = () => { st.playing = false; set(); onStep(); };
    q('#reset').onclick = () => { st.playing = true; set(); onReset(); };
    st.rate = () => { const r = +q('#sp').value / 10; q('#spv').textContent = r + ' ' + unit; return r; };
    return st;
  }
  const chipBtn = (parent, label, fn) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = fn; parent.appendChild(b); return b; };
  const ptr = (e, el) => { const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  const theme = () => document.documentElement.dataset.theme;
  const softmax5 = (d2, T, out, o) => {
    let mx = -1e9; const l = [];
    for (let k = 0; k < 5; k++) { l[k] = -d2[k] / (2 * T * T); if (l[k] > mx) mx = l[k]; }
    let s = 0; for (let k = 0; k < 5; k++) { l[k] = Math.exp(l[k] - mx); s += l[k]; }
    for (let k = 0; k < 5; k++) out[o + k] = l[k] / s;
  };
  const dist2 = (r, g, b) => PROTO.map((p) => (r - p.c[0]) ** 2 + (g - p.c[1]) ** 2 + (b - p.c[2]) ** 2);

  /* ================= segpixel ================= */
  AI.labs.segpixel = function (root) {
    const N = 20, TOT = N * N;
    const { view, ctl } = AI.labShell(root, 'Per-pixel classification', 'a "network" scans the image and gives every pixel a class',
      '<canvas style="height:400px"></canvas><div class="caption" id="cap"></div>',
      `<label>Temperature T <span class="v" id="Tv"></span><input type="range" id="T" min="10" max="90" value="35"></label>
       <label>Pixel noise <span class="v" id="nzv"></span><input type="range" id="nz" min="0" max="80" value="10"></label>
       <label>Confidence threshold <span class="v" id="thrv"></span><input type="range" id="thr" min="0" max="95" value="50"></label>
       ${PLAY(30, 1, 400)}
       <div class="row"><button class="btn" id="all">Show all at once</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), tick = AI.ticker();
    const im = AI.cv.resample('shapes', N), rnd = AI.rng(7);
    const nz = Float32Array.from({ length: TOT * 3 }, () => AI.randn(rnd));
    const pr = new Float32Array(TOT * 5), best = new Int8Array(TOT), conf = new Float32Array(TOT), outc = new Int8Array(TOT), gt = new Int8Array(TOT);
    for (let i = 0; i < TOT; i++) { const d = dist2(im.r[i], im.g[i], im.b[i]); gt[i] = d.indexOf(Math.min(...d)); }
    let pos = 0, last = '';
    function compute() {
      const T = +q('#T').value, sg = +q('#nz').value, thr = +q('#thr').value / 100;
      q('#Tv').textContent = T; q('#nzv').textContent = sg; q('#thrv').textContent = thr.toFixed(2);
      for (let i = 0; i < TOT; i++) {
        const d = dist2(im.r[i] + nz[i * 3] * sg, im.g[i] + nz[i * 3 + 1] * sg, im.b[i] + nz[i * 3 + 2] * sg);
        softmax5(d, T, pr, i * 5);
        let b = 0; for (let k = 1; k < 5; k++) if (pr[i * 5 + k] > pr[i * 5 + b]) b = k;
        best[i] = b; conf[i] = pr[i * 5 + b]; outc[i] = conf[i] < thr ? -1 : b;
      }
      last = '';
    }
    ['#T', '#nz', '#thr'].forEach((s) => q(s).addEventListener('input', compute));
    q('#all').onclick = () => (pos = TOT);
    const st = player(q, 'pixels / second ÷ 10 ×10', () => (pos = Math.min(TOT, Math.floor(pos) + 1)), () => (pos = 0));
    compute();
    AI.loop(root, (dt) => {
      const rate = st.rate() * 10; q('#spv').textContent = rate + ' pixels / s';
      if (st.playing && pos < TOT) pos = Math.min(TOT, pos + tick(dt, rate));
      const n = Math.floor(pos), cur = Math.min(TOT - 1, n);
      const key = [n, last, c.w, theme()].join() + [q('#T').value, q('#nz').value, q('#thr').value].join();
      if (key === last + '!') return;
      const g = c.ctx, C = AI.col(); g.clearRect(0, 0, c.w, c.h);
      const s = Math.min((c.w - 50) / 2, c.h - 120), cs = s / N, x0 = 15, x1 = x0 + s + 20, y0 = 24;
      g.font = '12px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('input image (20×20 pixels)', x0, 15); g.fillText('label map: one class per pixel', x1, 15);
      let ok = 0, unsure = 0, known = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x;
        g.fillStyle = `rgb(${im.r[i]},${im.g[i]},${im.b[i]})`; g.fillRect(x0 + x * cs, y0 + y * cs, cs + 0.6, cs + 0.6);
        if (i < n) {
          if (outc[i] < 0) { g.fillStyle = C.bg3; g.fillRect(x1 + x * cs, y0 + y * cs, cs + 0.6, cs + 0.6); g.fillStyle = C.muted; g.font = `${cs * 0.7}px sans-serif`; g.textAlign = 'center'; g.fillText('?', x1 + (x + 0.5) * cs, y0 + (y + 0.75) * cs); unsure++; }
          else { g.fillStyle = rgbs(PROTO[outc[i]].col); g.fillRect(x1 + x * cs, y0 + y * cs, cs + 0.6, cs + 0.6); known++; if (outc[i] === gt[i]) ok++; }
        } else { g.fillStyle = C.bg2; g.fillRect(x1 + x * cs, y0 + y * cs, cs, cs); g.strokeStyle = C.border; g.strokeRect(x1 + x * cs + 0.5, y0 + y * cs + 0.5, cs - 1, cs - 1); }
      }
      if (n < TOT) {
        const cx = cur % N, cy = Math.floor(cur / N);
        g.strokeStyle = '#fff'; g.lineWidth = 2.5; g.strokeRect(x0 + cx * cs, y0 + cy * cs, cs, cs); g.strokeRect(x1 + cx * cs, y0 + cy * cs, cs, cs); g.lineWidth = 1;
      }
      // probability bars of the current pixel
      const by = y0 + s + 26, bh = c.h - by - 34, bw = Math.min(70, (c.w - 40) / 5 - 12);
      g.textAlign = 'center'; g.font = '11px sans-serif';
      for (let k = 0; k < 5; k++) {
        const bx = 20 + k * (bw + 14 + (c.w - 40 - 5 * (bw + 14)) / 5), p = pr[cur * 5 + k];
        g.fillStyle = C.bg2; g.fillRect(bx, by, bw, bh);
        g.fillStyle = rgbs(PROTO[k].col); g.globalAlpha = k === best[cur] ? 1 : 0.5; g.fillRect(bx, by + bh * (1 - p), bw, bh * p); g.globalAlpha = 1;
        g.fillStyle = C.fg; g.fillText(p.toFixed(2), bx + bw / 2, by + bh - 4 > by + bh * (1 - p) ? by + bh - 4 : by + bh * (1 - p) - 3);
        g.fillStyle = C.muted; g.fillText(PROTO[k].n, bx + bw / 2, by + bh + 13);
      }
      g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText(`class probabilities of the white pixel (softmax of −distance² to each class colour)`, 20, by - 6);
      const d = dist2(im.r[cur] + nz[cur * 3] * +q('#nz').value, im.g[cur] + nz[cur * 3 + 1] * +q('#nz').value, im.b[cur] + nz[cur * 3 + 2] * +q('#nz').value);
      q('#out').innerHTML = `pixel <b>(${cur % N}, ${Math.floor(cur / N)})</b>  colour <b>(${Math.round(im.r[cur])}, ${Math.round(im.g[cur])}, ${Math.round(im.b[cur])})</b>\narg-max class: <b class="o">${PROTO[best[cur]].n}</b>  p = <b>${conf[cur].toFixed(2)}</b>\n${outc[cur] < 0 ? '<span class="r">below threshold → "unsure"</span>' : '<span class="g">above threshold → labelled</span>'}\nscanned: ${n}/${TOT}   labelled: ${known}   unsure: ${unsure}\naccuracy on labelled pixels: <b>${known ? ((100 * ok) / known).toFixed(1) : '–'}%</b>\n<span class="m">truth = class of the noise-free pixel</span>`;
      q('#cap').textContent = n >= TOT ? 'Done. A real fully convolutional net does all 400 pixels in ONE forward pass — the scan is slowed down only so you can watch.' : `① take pixel (${cur % N},${Math.floor(cur / N)}) → ② score every class → ③ softmax → ④ arg-max (or "unsure" if the top probability is below ${(+q('#thr').value / 100).toFixed(2)}) → ⑤ colour the label map.`;
      last = last; void d;
    }, { fps: 30 });
  };

  /* ================= upsample ================= */
  AI.labs.upsample = function (root) {
    const { view, ctl } = AI.labShell(root, 'Upsampling: four ways to grow a feature map', 'edit the 3×3 input and the kernel; watch every write and every overlap',
      '<canvas style="height:420px"></canvas><div class="caption" id="cap"></div>',
      `<label>Method <select id="m"><option value="nearest">nearest-neighbour ×2</option><option value="bilinear">bilinear ×2</option><option value="tconv" selected>transposed convolution</option><option value="shuffle">pixel shuffle ×2</option></select></label>
       <div class="row" id="tc"><label style="flex:1">kernel K <select id="K"><option>2</option><option selected>3</option><option>4</option></select></label><label style="flex:1">stride S <select id="S"><option>1</option><option selected>2</option></select></label></div>
       <div id="kwrap">Kernel weights (editable)<div id="kern" style="display:grid;gap:4px;margin-top:4px"></div></div>
       <div>Input 3×3 (editable)<div id="inp" style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px"></div><div class="chips" id="pre" style="margin-top:6px"></div></div>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="cntv"> colour output by overlap count</label>
       ${PLAY(8, 2, 60)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), tick = AI.ticker();
    const X = [1, 2, 3, 4, 5, 6, 7, 8, 9], ker = { 2: [1, 1, 1, 1], 3: Array(9).fill(1), 4: Array(16).fill(1) }, sw = [1, 0.5, 0.75, 0.25];
    let steps = [], OH = 6, pos = 0, ocache = null, ckey = '', maxabs = 1;
    const inin = X.map((v, i) => { const e = document.createElement('input'); e.type = 'number'; e.step = 1; e.value = v; e.style.textAlign = 'center'; e.oninput = () => { X[i] = +e.value || 0; build(); }; q('#inp').appendChild(e); return e; });
    const swin = []; // shuffle weights live in the kernel box when method = shuffle
    function kernUI() {
      const m = q('#m').value, K = +q('#K').value, box = q('#kern'); box.innerHTML = '';
      q('#tc').style.display = m === 'tconv' ? '' : 'none';
      q('#kwrap').style.display = m === 'tconv' || m === 'shuffle' ? '' : 'none';
      q('#kwrap').firstChild.textContent = m === 'shuffle' ? '4 channel weights (1×1 conv that makes 4 channels from the input)' : 'Kernel weights (editable)';
      const arr = m === 'shuffle' ? sw : ker[K], n = m === 'shuffle' ? 2 : K;
      box.style.gridTemplateColumns = `repeat(${n},1fr)`;
      arr.forEach((v, i) => { const e = document.createElement('input'); e.type = 'number'; e.step = 0.25; e.value = v; e.style.textAlign = 'center'; e.oninput = () => { arr[i] = +e.value || 0; build(); }; box.appendChild(e); });
    }
    const presets = { 'all ones': () => X.fill(1), ramp: () => X.forEach((_, i) => (X[i] = i + 1)), 'one bright pixel': () => X.forEach((_, i) => (X[i] = i === 4 ? 9 : 0)) };
    Object.entries(presets).forEach(([n, f]) => chipBtn(q('#pre'), n, () => { f(); inin.forEach((e, i) => (e.value = X[i])); build(); }));
    function build() {
      const m = q('#m').value, K = +q('#K').value, S = +q('#S').value, A = (j, i) => X[j * 3 + i]; steps = [];
      if (m === 'nearest') { OH = 6; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) steps.push({ src: [[j, i, 1]], w: [0, 1].flatMap((a) => [0, 1].map((b) => ({ o: [2 * j + a, 2 * i + b], v: A(j, i) }))), cap: `Copy input pixel (${j},${i}) = ${A(j, i)} into a 2×2 block. Nothing is learned, nothing overlaps — but the result is blocky.` }); }
      else if (m === 'bilinear') {
        OH = 6;
        for (let oy = 0; oy < 6; oy++) for (let ox = 0; ox < 6; ox++) {
          const sy = AI.clamp((oy + 0.5) / 2 - 0.5, 0, 2), sx = AI.clamp((ox + 0.5) / 2 - 0.5, 0, 2), y0 = Math.floor(sy), x0 = Math.floor(sx), y1 = Math.min(2, y0 + 1), x1 = Math.min(2, x0 + 1), fy = sy - y0, fx = sx - x0;
          const wts = [[y0, x0, (1 - fy) * (1 - fx)], [y0, x1, (1 - fy) * fx], [y1, x0, fy * (1 - fx)], [y1, x1, fy * fx]];
          const v = wts.reduce((a, [j, i, w]) => a + w * A(j, i), 0);
          steps.push({ src: wts, w: [{ o: [oy, ox], v }], cap: `Output (${oy},${ox}) lies at input coordinate (${sy.toFixed(2)}, ${sx.toFixed(2)}): blend the 4 nearest input pixels with weights ${wts.map((t) => t[2].toFixed(2)).join(', ')} → ${AI.fmt(v, 2)}.` });
        }
      } else if (m === 'tconv') {
        OH = (3 - 1) * S + K; const kk = ker[K];
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
          const w = []; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) w.push({ o: [j * S + a, i * S + b], v: A(j, i) * kk[a * K + b], add: true });
          steps.push({ src: [[j, i, 1]], w, cap: `Stamp: multiply the whole ${K}×${K} kernel by input pixel (${j},${i}) = ${A(j, i)} and ADD it into the output at offset (${j * S}, ${i * S}). Where stamps overlap, values add.` });
        }
      } else {
        OH = 6;
        for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) steps.push({ src: [[j, i, 1]], w: [0, 1].flatMap((a) => [0, 1].map((b) => ({ o: [2 * j + a, 2 * i + b], v: A(j, i) * sw[2 * a + b] }))), cap: `Pixel (${j},${i}) = ${A(j, i)} became 4 channels (×${sw.join(', ×')}); pixel shuffle lays channel (a,b) at sub-position (a,b) of its own 2×2 block — no overlap, each output cell written exactly once.` });
      }
      const f = apply(steps.length); maxabs = Math.max(1e-6, ...f.out.map(Math.abs)); ckey = '';
    }
    function apply(upto) {
      const out = new Float32Array(OH * OH), cnt = new Uint8Array(OH * OH);
      for (let k = 0; k < upto; k++) for (const w of steps[k].w) { const i = w.o[0] * OH + w.o[1]; if (w.add) out[i] += w.v; else out[i] = w.v; cnt[i]++; }
      return { out, cnt };
    }
    ['#m', '#K', '#S'].forEach((s) => q(s).addEventListener('input', () => {
      kernUI(); build(); pos = 0;
      const m = q('#m').value; q('#sp').value = m === 'bilinear' ? 40 : 8;
    }));
    q('#cntv').addEventListener('input', () => (ckey = ''));
    const st = player(q, 'steps / s', () => (pos = Math.min(steps.length, Math.floor(pos) + 1)), () => (pos = 0));
    kernUI(); build();
    AI.loop(root, (dt) => {
      const rate = st.rate(); if (st.playing && pos < steps.length) pos = Math.min(steps.length, pos + dt * rate);
      const idx = Math.min(steps.length - 1, Math.floor(pos)), frac = pos - idx, done = pos >= steps.length;
      const upto = done ? steps.length : frac > 0.3 ? idx + 1 : idx;
      const key = [upto, idx, done, q('#cntv').checked, c.w, theme(), q('#m').value, X.join(), ker[3].join(), ker[2].join(), ker[4].join(), sw.join(), q('#K').value, q('#S').value].join();
      if (key === ckey && !(!done && st.playing)) { if (key === ckey) return; }
      ckey = key;
      const m = q('#m').value, K = +q('#K').value, S = +q('#S').value, g = c.ctx, C = AI.col(); g.clearRect(0, 0, c.w, c.h);
      const { out, cnt } = apply(upto), kn = m === 'tconv' ? K : m === 'shuffle' ? 2 : 0;
      const u = Math.min(56, (c.w - 28 - (kn ? 44 : 22)) / (3 + (kn ? kn * 0.7 : 0) + OH), (c.h - 60) / OH), cs = u, os = u, ks = u * 0.7;
      const ix = 14, iy = 40, kx = ix + 3 * cs + 22, ox = kx + (kn ? kn * ks + 22 : 0), oy = 40;
      g.font = '12px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('input 3×3', ix, 22); if (kn) g.fillText(m === 'shuffle' ? 'channel weights' : `kernel ${K}×${K}`, kx, 22);
      g.fillText(`output ${OH}×${OH}`, ox, 22);
      const cur = done ? null : steps[idx], srcSet = new Map(); if (cur) cur.src.forEach(([j, i, w]) => srcSet.set(j * 3 + i, w));
      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
        const v = X[j * 3 + i], px = ix + i * cs, py = iy + j * cs;
        g.fillStyle = AI.mix(C.bg2, C.blue, Math.min(1, Math.abs(v) / 10)); g.fillRect(px, py, cs, cs);
        if (srcSet.has(j * 3 + i)) { g.strokeStyle = C.blue; g.lineWidth = 3; g.strokeRect(px + 1, py + 1, cs - 2, cs - 2); g.lineWidth = 1; }
        else { g.strokeStyle = C.border; g.strokeRect(px + 0.5, py + 0.5, cs - 1, cs - 1); }
        g.fillStyle = C.fg; g.font = `${Math.min(15, cs / 3)}px ui-monospace,Menlo,monospace`; g.fillText(AI.fmt(v, 2), px + cs / 2, py + cs / 2);
        if (m === 'bilinear' && srcSet.has(j * 3 + i)) { g.fillStyle = C.orange; g.font = '10px sans-serif'; g.fillText('w=' + srcSet.get(j * 3 + i).toFixed(2), px + cs / 2, py + cs - 8); }
      }
      if (kn) {
        const arr = m === 'shuffle' ? sw : ker[K];
        for (let a = 0; a < kn; a++) for (let b = 0; b < kn; b++) { const v = arr[a * kn + b]; g.fillStyle = v > 0 ? AI.mix(C.bg2, C.green, Math.min(1, v / 2)) : v < 0 ? AI.mix(C.bg2, C.orange, Math.min(1, -v / 2)) : C.bg2; g.fillRect(kx + b * ks, iy + a * ks, ks, ks); g.strokeStyle = C.border; g.strokeRect(kx + b * ks + 0.5, iy + a * ks + 0.5, ks - 1, ks - 1); g.fillStyle = C.fg; g.font = '10px ui-monospace,Menlo,monospace'; g.fillText(AI.fmt(v, 2), kx + (b + 0.5) * ks, iy + (a + 0.5) * ks); }
      }
      const mxc = Math.max(1, ...cnt), tgt = new Set(cur ? cur.w.map((w) => w.o[0] * OH + w.o[1]) : []);
      for (let y = 0; y < OH; y++) for (let x = 0; x < OH; x++) {
        const k = y * OH + x, px = ox + x * os, py = oy + y * os, show = cnt[k] > 0;
        g.fillStyle = !show ? C.bg2 : q('#cntv').checked ? AI.mix(C.bg2, C.orange, cnt[k] / (Math.max(2, mxc) + 0.5)) : AI.mix(C.bg2, out[k] >= 0 ? C.green : C.orange, Math.min(1, Math.abs(out[k]) / maxabs));
        g.fillRect(px, py, os, os);
        g.strokeStyle = tgt.has(k) ? C.orange : C.border; g.lineWidth = tgt.has(k) ? 2.5 : 1; g.strokeRect(px + 0.5, py + 0.5, os - 1, os - 1); g.lineWidth = 1;
        if (show) { g.fillStyle = C.fg; g.font = `${Math.min(13, os / 2.8)}px ui-monospace,Menlo,monospace`; g.fillText(q('#cntv').checked ? '×' + cnt[k] : AI.fmt(out[k], 1), px + os / 2, py + os / 2); }
      }
      const fin = apply(steps.length).cnt, mn = Math.min(...fin), mx = Math.max(...fin);
      let info = '';
      if (m === 'tconv') info = `O = (I − 1)·S + K\n  = (3 − 1)·${S} + ${K} = <b class="o">${OH}</b>\noverlap count: min <b>${mn}</b>, max <b>${mx}</b>\n${K % S === 0 && S > 1 ? '<span class="g">K % S = 0 → even overlap</span>' : S === 1 ? '<span class="m">S = 1: smooth overlap</span>' : '<span class="r">K % S ≠ 0 → uneven overlap (checkerboard risk)</span>'}\nparams: K·K·C_in·C_out = ${K * K}·C_in·C_out`;
      else if (m === 'shuffle') info = 'output (C, 2H, 2W) = rearranged from\n(4C, H, W): <b class="o">no multiply, no overlap</b>\nevery output cell written once\nthe learning happens in the conv\nthat makes the 4C channels';
      else if (m === 'bilinear') info = 'fixed weights from distance;\n<b class="o">no parameters</b>, smooth result.\nrule: out = Σ w·neighbour,\nw from fractional position';
      else info = 'copy each value to a 2×2 block;\n<b class="o">no parameters</b>, blocky edges.';
      q('#out').innerHTML = info;
      q('#cap').textContent = done ? 'All steps done. Tick "colour by overlap count": with kernel 3 and stride 2 some cells are written 1×, others 2× — the uneven pattern is the checkerboard.' : `Step ${idx + 1} of ${steps.length}: ${cur.cap}`;
      g.textBaseline = 'alphabetic';
    }, { fps: 30 });
  };

  /* ================= unetanim ================= */
  AI.labs.unetanim = function (root) {
    const N = 40, LV = [40, 20, 10, 5], T0 = 35;
    const { view, ctl } = AI.labShell(root, 'U-Net: shrink, grow, and skip', 'encoder (left) shrinks, decoder (right) grows; cut the skips and watch the edges blur',
      '<canvas style="height:430px"></canvas><div class="caption" id="cap"></div>',
      `<div>Skip connections (carry detail across)<div style="display:flex;flex-direction:column;gap:5px;margin-top:5px">
         <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="s0" checked> skip at 40×40 (finest)</label>
         <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="s1" checked> skip at 20×20</label>
         <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="s2" checked> skip at 10×10</label></div>
         <div class="chips" id="pre" style="margin-top:6px"></div></div>
       ${PLAY(4, 1, 20)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    const im = AI.cv.resample('shapes', N), feats = [{ r: im.r, g: im.g, b: im.b }];
    for (let l = 1; l < 4; l++) { const p = feats[l - 1], n = LV[l - 1], m = LV[l], o = { r: new Float32Array(m * m), g: new Float32Array(m * m), b: new Float32Array(m * m) }; for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) for (const k of ['r', 'g', 'b']) o[k][y * m + x] = (p[k][2 * y * n + 2 * x] + p[k][2 * y * n + 2 * x + 1] + p[k][(2 * y + 1) * n + 2 * x] + p[k][(2 * y + 1) * n + 2 * x + 1]) / 4; feats.push(o); }
    const Lk = feats.map((f, l) => { const n = LV[l], o = new Float32Array(n * n * 5); for (let i = 0; i < n * n; i++) softmax5(dist2(f.r[i], f.g[i], f.b[i]), T0, o, i * 5); return o; });
    const gt = Int8Array.from({ length: N * N }, (_, i) => { let b = 0; for (let k = 1; k < 5; k++) if (Lk[0][i * 5 + k] > Lk[0][i * 5 + b]) b = k; return b; });
    const up2 = (a, n) => { const m = 2 * n, o = new Float32Array(m * m * 5); for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) { const sy = AI.clamp((y + 0.5) / 2 - 0.5, 0, n - 1), sx = AI.clamp((x + 0.5) / 2 - 0.5, 0, n - 1), y0 = Math.floor(sy), x0 = Math.floor(sx), y1 = Math.min(n - 1, y0 + 1), x1 = Math.min(n - 1, x0 + 1), fy = sy - y0, fx = sx - x0; for (let k = 0; k < 5; k++) o[(y * m + x) * 5 + k] = (1 - fy) * (1 - fx) * a[(y0 * n + x0) * 5 + k] + (1 - fy) * fx * a[(y0 * n + x1) * 5 + k] + fy * (1 - fx) * a[(y1 * n + x0) * 5 + k] + fy * fx * a[(y1 * n + x1) * 5 + k]; } return o; };
    const mk = (n) => { const cv = document.createElement('canvas'); cv.width = cv.height = n; return cv; };
    const enc = LV.map((n) => mk(n)), dec = LV.map((n) => mk(n));
    enc.forEach((cv, l) => { const g = cv.getContext('2d'), d = g.createImageData(LV[l], LV[l]); for (let i = 0; i < LV[l] ** 2; i++) { d.data[i * 4] = feats[l].r[i]; d.data[i * 4 + 1] = feats[l].g[i]; d.data[i * 4 + 2] = feats[l].b[i]; d.data[i * 4 + 3] = 255; } g.putImageData(d, 0, 0); });
    let D = [], res = { acc: 0, edge: 0, nEdge: 0 };
    const paint = (l, a) => { const n = LV[l], g = dec[l].getContext('2d'), d = g.createImageData(n, n); for (let i = 0; i < n * n; i++) { let b = 0; for (let k = 1; k < 5; k++) if (a[i * 5 + k] > a[i * 5 + b]) b = k; const cf = a[i * 5 + b], col = PROTO[b].col, w = 0.25 + 0.75 * cf; for (let k = 0; k < 3; k++) d.data[i * 4 + k] = 120 + (col[k] - 120) * w; d.data[i * 4 + 3] = 255; } g.putImageData(d, 0, 0); };
    function decode() {
      const sk = [q('#s0').checked, q('#s1').checked, q('#s2').checked]; D = [];
      D[3] = Lk[3]; paint(3, D[3]);
      for (let l = 2; l >= 0; l--) {
        const U = up2(D[l + 1], LV[l + 1]), n = LV[l], o = new Float32Array(n * n * 5);
        for (let i = 0; i < n * n; i++) { let s = 0; for (let k = 0; k < 5; k++) { const v = sk[l] ? Math.sqrt(U[i * 5 + k]) * Lk[l][i * 5 + k] : U[i * 5 + k]; o[i * 5 + k] = v; s += v; } for (let k = 0; k < 5; k++) o[i * 5 + k] = s > 1e-9 ? o[i * 5 + k] / s : 0.2; }
        D[l] = o; paint(l, o);
      }
      let ok = 0, eok = 0, en = 0;
      for (let i = 0; i < N * N; i++) {
        let b = 0; for (let k = 1; k < 5; k++) if (D[0][i * 5 + k] > D[0][i * 5 + b]) b = k;
        if (b === gt[i]) ok++;
        const x = i % N, y = Math.floor(i / N); let edge = false;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < N && yy < N && gt[yy * N + xx] !== gt[i]) edge = true; }
        if (edge) { en++; if (b === gt[i]) eok++; }
      }
      res = { acc: ok / (N * N), edge: en ? eok / en : 1, nEdge: en };
    }
    ['#s0', '#s1', '#s2'].forEach((s) => q(s).addEventListener('input', decode));
    chipBtn(q('#pre'), 'cut all skips', () => { ['#s0', '#s1', '#s2'].forEach((s) => (q(s).checked = false)); decode(); });
    chipBtn(q('#pre'), 'restore all', () => { ['#s0', '#s1', '#s2'].forEach((s) => (q(s).checked = true)); decode(); });
    decode();
    let pos = 0, clock = 0;
    const st = player(q, 'phases / s', () => (pos = Math.min(8, Math.floor(pos) + 1)), () => (pos = 0));
    const TP = [0, 1, 2, 3, 4, 5, 6, 7], FROM = [0, 0, 1, 2, 3, 3, 3, 2]; // tile t appears at phase TP[t], flying from the place of tile FROM[t]
    const PL = [0, 1, 2, 3, 3, 2, 1, 0]; // level of tile t (column placement); tiles 5-7 are decoder side
    const CAP = [
      'Input image enters (toy 40×40; the real net uses e.g. 128×128 with 64 channels after two 3×3 convs).',
      'Encoder: max-pool halves the image → 20×20. Channels double in a real net, so the map holds fewer positions but richer features.',
      'Pool again → 10×10. Each cell now summarises a 4×4 block and "sees" a larger neighbourhood (growing receptive field).',
      'Bottleneck 5×5: very large receptive field — it knows WHAT is in the image, but one cell covers 8×8 pixels of WHERE.',
      'Classify the bottleneck: a coarse, blocky guess. Mixed cells at object borders are uncertain (pale colours).',
      '', '', '',
    ];
    AI.loop(root, (dt) => {
      const rate = st.rate(); clock += dt; if (st.playing && pos < 8) pos = Math.min(8, pos + dt * rate);
      const sk = [q('#s0').checked, q('#s1').checked, q('#s2').checked], g = c.ctx, C = AI.col(); g.clearRect(0, 0, c.w, c.h);
      const s0 = Math.min(150, (c.w - 7 * 16) / 3.7), top = 34 + s0 / 2, bot = c.h - 62, W = c.w, dx = 14;
      const sizeOf = (l) => s0 * (LV[l] / 40);
      const colw = (W - 2 * dx - 2 * s0) / 6;
      const place = (t) => { const l = PL[t], size = sizeOf(l), enc = t < 5, ycl = top + ((bot - top) * l) / 3; let cx; if (enc) cx = dx + s0 / 2 + l * (colw + (s0 / 2 - sizeOf(l) / 2) * 0 + 0) + l * (0); else cx = 0; return { l, size, enc, cy: ycl }; };
      // columns: encoder level l at x = dx + s0/2 + l*step, decoder mirrored
      const step = (W - 2 * dx - s0) / 6, cxEnc = (l) => dx + s0 / 2 + l * step, cxDec = (l) => W - dx - s0 / 2 - l * step;
      const P = (t) => { const p = place(t); return { x: p.enc ? cxEnc(p.l) : cxDec(p.l), y: p.cy, s: p.size, l: p.l, enc: p.enc }; };
      g.font = '12px sans-serif'; g.textAlign = 'center'; g.fillStyle = C.muted;
      g.fillText('ENCODER (contracting path)', cxEnc(1) + 10, 14); g.fillText('DECODER (expanding path)', cxDec(1) - 10, 14);
      g.imageSmoothingEnabled = false;
      const lab = (p, txt) => { g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText(txt, p.x, p.y + p.s / 2 + 13); };
      // skips (under tiles)
      for (let l = 0; l < 3; l++) {
        const tD = 7 - l, u = AI.smooth(pos - TP[tD] - 0.6);
        if (u <= 0) continue;
        const a = P(l), b = P(tD), x0 = a.x + a.s / 2 + 3, x1 = b.x - b.s / 2 - 3;
        g.globalAlpha = Math.min(1, u * 1.5);
        if (sk[l]) {
          g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); g.moveTo(x0, a.y); g.lineTo(x1, b.y); g.stroke(); g.lineWidth = 1;
          for (let k = 0; k < 3; k++) { const f = ((clock * 0.45 + k / 3) % 1), px = x0 + (x1 - x0) * f; g.fillStyle = C.green; g.beginPath(); g.arc(px, a.y, 3.5, 0, 7); g.fill(); }
          g.fillStyle = C.green; g.font = '11px sans-serif'; g.fillText('skip: copy + concat', (x0 + x1) / 2, a.y - 6);
        } else {
          g.strokeStyle = C.red; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(x0, a.y); g.lineTo(x1, b.y); g.stroke(); g.setLineDash([]);
          const mx = (x0 + x1) / 2; g.fillStyle = C.red; g.font = 'bold 16px sans-serif'; g.fillText('✕', mx, a.y + 5); g.font = '11px sans-serif'; g.fillText('skip cut', mx, a.y - 10);
        }
        g.globalAlpha = 1;
      }
      for (let t = 0; t < 8; t++) {
        const u = AI.smooth(pos - TP[t]); if (u <= 0) continue;
        const to = P(t), fromP = P(FROM[t] === undefined ? t : FROM[t]), isSame = t === 4;
        const x = t === 0 ? to.x : AI.lerp(fromP.x, to.x, u), y = t === 0 ? to.y : AI.lerp(fromP.y, to.y, u), sz = t === 0 ? to.s : AI.lerp(fromP.s, to.s, u);
        const cv = t < 4 ? enc[to.l] : dec[to.l];
        if (t === 3 && pos > 4) { /* crossfade handled by tile 4 drawn on top */ }
        g.globalAlpha = Math.min(1, u * 2.5) * (t === 4 ? 1 : 1);
        g.drawImage(cv, x - sz / 2, y - sz / 2, sz, sz); g.globalAlpha = 1;
        g.strokeStyle = t < 4 ? C.blue : C.green; g.lineWidth = 2; g.strokeRect(x - sz / 2, y - sz / 2, sz, sz); g.lineWidth = 1;
        if (u > 0.95 && !isSame) lab(to, `${LV[to.l]}×${LV[to.l]}${t === 3 ? ' bottleneck' : ''}`);
        if (isSame && u > 0.95) { g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('class map', to.x, to.y + to.s / 2 + 26); }
        if (t > 0 && u > 0.02 && u < 0.98 && !isSame) { g.strokeStyle = C.muted; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(fromP.x, fromP.y); g.lineTo(to.x, to.y); g.stroke(); g.setLineDash([]); }
      }
      const ph = Math.min(7, Math.floor(pos));
      let cap = CAP[ph];
      if (ph >= 5) { const l = 7 - ph; cap = `Decoder step ${ph - 4}: up-convolution doubles the map to ${LV[l]}×${LV[l]}. ` + (sk[l] ? 'The skip connection concatenates the encoder features of the same size, so fine detail returns and the class map is sharpened.' : 'The skip is CUT: the decoder only has the upsampled coarse guess, so edges stay blurry/blocky.'); }
      if (pos >= 8) cap = `Done. Pixel accuracy ${(100 * res.acc).toFixed(1)}%; accuracy on the ${res.nEdge} boundary pixels ${(100 * res.edge).toFixed(1)}%. Toggle skips and press Restart to compare.`;
      q('#cap').textContent = cap;
      q('#out').innerHTML = `result @40×40 (all phases done):\npixel accuracy   <b class="o">${(100 * res.acc).toFixed(1)}%</b>\nboundary pixels  <b class="o">${(100 * res.edge).toFixed(1)}%</b>  (n=${res.nEdge})\nskips on: <b>${sk.filter(Boolean).length}/3</b>\n<span class="m">truth = per-pixel classification of the full-resolution input</span>`;
    }, { fps: 30 });
  };

  /* ================= dilated ================= */
  AI.labs.dilated = function (root) {
    const { view, ctl } = AI.labShell(root, 'Dilated convolution: receptive field map', 'four stacked 3×3 layers; edit each layer\'s dilation rate',
      '<canvas style="height:420px"></canvas><div class="caption" id="cap"></div>',
      `<div>Dilation rate of each layer (1 = ordinary conv)<div id="rates" style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:5px"></div><div class="chips" id="pre" style="margin-top:6px"></div></div>
       ${PLAY(4, 1, 20)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    const R = [1, 2, 4, 8], rin = R.map((v, i) => { const e = document.createElement('input'); e.type = 'number'; e.min = 1; e.max = 8; e.value = v; e.style.textAlign = 'center'; e.oninput = () => { R[i] = AI.clamp(Math.round(+e.value) || 1, 1, 8); build(); pos = 0; }; q('#rates').appendChild(e); return e; });
    const setR = (a) => { a.forEach((v, i) => { R[i] = v; rin[i].value = v; }); build(); pos = 0; };
    [[1, 1, 1, 1], [1, 2, 4, 8], [2, 2, 2, 2], [1, 2, 5, 1], [4, 4, 4, 4]].forEach((a) => chipBtn(q('#pre'), a.join(','), () => setR(a)));
    let stages = [], G = 9, rf = 9, holes = 0, covered = 0, mx = 1, pos = 0;
    function build() {
      rf = 1 + 2 * R.reduce((a, b) => a + b, 0); G = rf; stages = [];
      let cur = new Float32Array(G * G); cur[(G >> 1) * G + (G >> 1)] = 1; stages.push(cur);
      for (let l = 3; l >= 0; l--) { const nx = new Float32Array(G * G), r = R[l]; for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { const v = cur[y * G + x]; if (!v) continue; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const yy = y + a * r, xx = x + b * r; if (yy >= 0 && xx >= 0 && yy < G && xx < G) nx[yy * G + xx] += v; } } cur = nx; stages.push(cur); }
      const f = stages[4]; covered = 0; mx = 0; for (const v of f) { if (v > 0) covered++; if (v > mx) mx = v; }
      holes = rf * rf - covered;
    }
    const st = player(q, 'layers / s', () => (pos = Math.min(4, Math.floor(pos) + 1)), () => (pos = 0));
    build();
    AI.loop(root, (dt) => {
      const rate = st.rate(); if (st.playing && pos < 4) pos = Math.min(4, pos + dt * rate);
      const s = Math.min(Math.floor(pos), 4), fr = pos - s, g = c.ctx, C = AI.col(); g.clearRect(0, 0, c.w, c.h);
      const side = Math.min(c.w * 0.62, c.h - 40), cs = side / G, x0 = 12, y0 = 28;
      g.font = '12px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(`input pixels that influence ONE output pixel (field ${rf}×${rf})`, x0, 17);
      const A = stages[s], B = stages[Math.min(4, s + 1)];
      for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
        const i = y * G + x, v = A[i], n = B[i];
        if (v > 0) { g.fillStyle = AI.mix(C.bg2, C.blue, 0.35 + 0.65 * Math.sqrt(v / Math.max(1, Math.max(...A.slice(0, 1), mx)) )); }
        else if (s < 4 && n > 0) { g.globalAlpha = 1; g.fillStyle = AI.mix(C.bg2, C.orange, AI.smooth(fr) * 0.9); }
        else g.fillStyle = C.bg2;
        g.fillRect(x0 + x * cs, y0 + y * cs, cs, cs);
      }
      if (cs >= 4) { g.strokeStyle = C.border; g.lineWidth = 0.5; for (let k = 0; k <= G; k++) { g.beginPath(); g.moveTo(x0 + k * cs, y0); g.lineTo(x0 + k * cs, y0 + side); g.moveTo(x0, y0 + k * cs); g.lineTo(x0 + side, y0 + k * cs); g.stroke(); } }
      const ctr = G >> 1; g.strokeStyle = C.fg; g.lineWidth = 2; g.strokeRect(x0 + ctr * cs, y0 + ctr * cs, cs, cs); g.lineWidth = 1;
      if (s === 4) for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) if (stages[4][y * G + x] === 0) { g.fillStyle = C.red; g.globalAlpha = 0.35; g.fillRect(x0 + x * cs, y0 + y * cs, cs, cs); g.globalAlpha = 1; }
      // stage list on the right
      const rx = x0 + side + 24; g.textAlign = 'left'; g.font = '12px sans-serif';
      for (let l = 0; l < 4; l++) {
        const stage = 4 - l, y = y0 + 8 + l * 48, active = s === l || (s === 4 && l === 3), done = s > l;
        g.fillStyle = active ? C.orange : done ? C.blue : C.muted; g.fillText(`layer ${stage}: 3×3, rate ${R[stage - 1]}`, rx, y);
        g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText(`taps ${R[stage - 1]} px apart → +${2 * R[stage - 1]} field`, rx, y + 15); g.font = '12px sans-serif';
        // mini kernel
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { g.fillStyle = active ? C.orange : done ? C.blue : C.bg3; g.fillRect(c.w - 30 + b * 7, y - 10 + a * 7, 5, 5); }
      }
      const ph = s;
      q('#cap').textContent = s === 0 ? 'Start at ONE output pixel (white box) and trace backwards through the stack.' : s < 4 ? `Layer ${5 - s} reads 9 pixels spaced ${R[4 - s]} apart. Orange = new pixels that now influence the output (previous layers\' pixels are blue).` : `Done. Field = 1 + 2·(${R.join('+')}) = ${rf}. Red cells are blind spots inside the field that NO path reaches (gridding).`;
      q('#out').innerHTML = `rates <b>${R.join(', ')}</b>\nreceptive field = 1 + 2·Σr = <b class="o">${rf} × ${rf}</b>\nplain 4×(3×3): 9 × 9\npixels reached: ${covered} / ${rf * rf}\nblind spots: <b class="${holes ? 'r' : 'g'}">${holes}</b>\nweights per layer: 9·C² (same for every rate!)`;
      void ph;
    }, { fps: 30 });
  };

  /* ================= diceiou ================= */
  AI.labs.diceiou = function (root) {
    const N = 12, TOT = N * N;
    const { view, ctl } = AI.labShell(root, 'Segmentation metrics: IoU, Dice, accuracy, cross-entropy', 'click or drag to paint — blue = prediction, orange = ground truth',
      '<canvas style="height:380px;touch-action:none;cursor:crosshair"></canvas><div class="caption" id="cap"></div>',
      `<div>Paint which mask?<div class="row" style="margin-top:4px"><button class="btn primary" id="mp">prediction</button><button class="btn" id="mt">truth</button></div><div class="chips" id="pre" style="margin-top:6px"></div></div>
       <label>Prediction softness s <span class="v" id="sv"></span><input type="range" id="soft" min="1" max="45" value="5"></label>
       ${PLAY(10, 2, 80)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv), tick = AI.ticker();
    const P = new Uint8Array(TOT), Tm = new Uint8Array(TOT);
    const box = (a, x0, y0, x1, y1) => { a.fill(0); for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) a[y * N + x] = 1; };
    const setAll = (fp, ft) => { fp(P); ft(Tm); pos = 0; };
    const PRE = {
      'shifted': () => setAll((a) => box(a, 5, 3, 9, 8), (a) => box(a, 4, 3, 8, 8)),
      'too big': () => setAll((a) => box(a, 2, 2, 10, 10), (a) => box(a, 4, 4, 8, 8)),
      'too small': () => setAll((a) => box(a, 5, 5, 6, 6), (a) => box(a, 3, 3, 8, 8)),
      'empty pred': () => setAll((a) => a.fill(0), (a) => box(a, 5, 5, 6, 6)),
      'perfect': () => setAll((a) => box(a, 3, 3, 8, 8), (a) => box(a, 3, 3, 8, 8)),
    };
    let which = 'P', pos = 0, painting = null, mode = 1;
    Object.entries(PRE).forEach(([n, f]) => chipBtn(q('#pre'), n, f));
    q('#mp').onclick = () => { which = 'P'; q('#mp').classList.add('primary'); q('#mt').classList.remove('primary'); };
    q('#mt').onclick = () => { which = 'T'; q('#mt').classList.add('primary'); q('#mp').classList.remove('primary'); };
    PRE.shifted();
    const geo = () => { const s = Math.min(c.w * 0.55, c.h - 30); return { s, cs: s / N, x0: 12, y0: 14 }; };
    const cell = (e) => { const p = ptr(e, cv), { cs, x0, y0 } = geo(), x = Math.floor((p.x - x0) / cs), y = Math.floor((p.y - y0) / cs); return x >= 0 && y >= 0 && x < N && y < N ? y * N + x : -1; };
    const paint = (e) => { const i = cell(e); if (i < 0) return; (which === 'P' ? P : Tm)[i] = painting; };
    cv.addEventListener('pointerdown', (e) => { const i = cell(e); if (i < 0) return; cv.setPointerCapture(e.pointerId); painting = (which === 'P' ? P : Tm)[i] ? 0 : 1; paint(e); });
    cv.addEventListener('pointermove', (e) => painting !== null && paint(e));
    cv.addEventListener('pointerup', () => (painting = null));
    const st = player(q, 'pixels / s', () => (pos = Math.min(TOT, Math.floor(pos) + 1)), () => (pos = 0));
    AI.loop(root, (dt) => {
      const rate = st.rate() * 10; q('#spv').textContent = rate + ' cells / s'; if (st.playing && pos < TOT) pos = Math.min(TOT, pos + tick(dt, rate));
      const n = Math.floor(pos), g = c.ctx, C = AI.col(), { s, cs, x0, y0 } = geo(); g.clearRect(0, 0, c.w, c.h);
      let tp = 0, fp = 0, fn = 0, tn = 0, TP = 0, FP = 0, FN = 0, TN = 0;
      for (let i = 0; i < TOT; i++) {
        const p = P[i], t = Tm[i], x = i % N, y = Math.floor(i / N), X = x0 + x * cs, Y = y0 + y * cs;
        if (p && t) TP++; else if (p) FP++; else if (t) FN++; else TN++;
        const scanned = i < n;
        if (scanned) { if (p && t) tp++; else if (p) fp++; else if (t) fn++; else tn++; }
        g.fillStyle = scanned ? (p && t ? C.green : p ? C.blue : t ? C.orange : C.bg2) : C.bg2; g.globalAlpha = scanned && (p || t) ? 0.85 : 1; g.fillRect(X, Y, cs, cs); g.globalAlpha = 1;
        if (!scanned) { if (p) { g.fillStyle = C.blue; g.globalAlpha = 0.45; g.fillRect(X + cs * 0.15, Y + cs * 0.15, cs * 0.7, cs * 0.7); g.globalAlpha = 1; } if (t) { g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(X + 2, Y + 2, cs - 4, cs - 4); g.lineWidth = 1; } }
        g.strokeStyle = C.border; g.strokeRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
      }
      if (n < TOT) { g.strokeStyle = '#fff'; g.lineWidth = 3; g.strokeRect(x0 + (n % N) * cs, y0 + Math.floor(n / N) * cs, cs, cs); g.lineWidth = 1; }
      const iou = TP + FP + FN ? TP / (TP + FP + FN) : 1, dice = 2 * TP + FP + FN ? (2 * TP) / (2 * TP + FP + FN) : 1, acc = (TP + TN) / TOT, prec = TP + FP ? TP / (TP + FP) : 0, rec = TP + FN ? TP / (TP + FN) : 0, iouB = TN + FP + FN ? TN / (TN + FP + FN) : 1;
      // bars
      const bx = x0 + s + 24, bw = Math.max(40, c.w - bx - 20), items = [['IoU (object)', iou, C.green], ['Dice (object)', dice, C.blue], ['pixel accuracy', acc, C.orange], ['precision', prec, C.purple], ['recall', rec, C.teal]];
      g.font = '12px sans-serif'; g.textAlign = 'left';
      items.forEach(([nm, v, col], k) => { const y = y0 + 8 + k * 56; g.fillStyle = C.muted; g.fillText(nm, bx, y); g.fillStyle = C.bg2; g.fillRect(bx, y + 6, bw, 16); g.fillStyle = col; g.fillRect(bx, y + 6, bw * v, 16); g.fillStyle = C.fg; g.fillText(v.toFixed(3), bx + 6, y + 19); });
      const sf = +q('#soft').value / 100; q('#sv').textContent = sf.toFixed(2);
      let ce = 0, sp = 0, sy = 0, si = 0;
      for (let i = 0; i < TOT; i++) { const pp = P[i] ? 1 - sf : sf, y = Tm[i]; ce += -(y ? Math.log(pp) : Math.log(1 - pp)); sp += pp; sy += y; si += pp * y; }
      ce /= TOT; const sd = sp + sy ? (2 * si) / (sp + sy) : 1;
      const mIoU = (iou + iouB) / 2;
      q('#out').innerHTML = `${n >= TOT ? '<span class="g">all cells counted</span>' : `counting… ${n}/${TOT}: TP ${tp} FP ${fp} FN ${fn}`}\nTP <b class="g">${TP}</b>  FP <b class="b">${FP}</b>  FN <b class="o">${FN}</b>  TN ${TN}\nIoU  = TP/(TP+FP+FN) = ${TP}/${TP + FP + FN} = <b>${iou.toFixed(3)}</b>\nDice = 2TP/(2TP+FP+FN) = ${2 * TP}/${2 * TP + FP + FN} = <b>${dice.toFixed(3)}</b>\ncheck: 2·IoU/(1+IoU) = ${(2 * iou / (1 + iou)).toFixed(3)}\nIoU(background) = ${iouB.toFixed(3)}  →  mIoU = <b class="o">${mIoU.toFixed(3)}</b>\nmean cross-entropy (soft p) = <b>${ce.toFixed(3)}</b>\nsoft Dice = <b>${sd.toFixed(3)}</b>  (loss ${(1 - sd).toFixed(3)})`;
      q('#cap').textContent = 'Green = TP (both), blue = FP (prediction only), orange = FN (truth only). Try "empty pred": pixel accuracy stays high while IoU and Dice collapse — why accuracy alone is a trap.';
    }, { fps: 30 });
  };

  /* ================= samprompt ================= */
  AI.labs.samprompt = function (root) {
    const N = 60, TOT = N * N;
    const PART = [
      { n: 'sky', c: [170, 210, 245], grp: 'sky', obj: 'sky' }, { n: 'grass', c: [76, 154, 61], grp: 'grass', obj: 'grass' }, { n: 'wall', c: [201, 139, 82], grp: 'facade', obj: 'house' },
      { n: 'roof', c: [160, 58, 42], grp: 'roof', obj: 'house' }, { n: 'door', c: [90, 58, 34], grp: 'facade', obj: 'house' }, { n: 'window', c: [247, 233, 160], grp: 'facade', obj: 'house' }, { n: 'sun', c: [255, 210, 63], grp: 'sun', obj: 'sun' },
    ];
    const { view, ctl } = AI.labShell(root, 'Promptable segmentation (SAM-style)', 'click a point or drag a box; three candidate masks grow — ambiguity is a feature',
      '<canvas style="height:440px;touch-action:none;cursor:crosshair"></canvas><div class="caption" id="cap"></div>',
      `<div>Prompt type<div class="chips" id="mode" style="margin-top:4px"></div></div>
       <div class="row"><button class="btn" id="clr">Clear prompts</button><button class="btn" id="demo">Demo: click a window</button></div>
       ${PLAY(10, 2, 30)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    const im = AI.cv.resample('house', N);
    let lab = new Uint8Array(TOT);
    for (let i = 0; i < TOT; i++) { let b = 0, bd = 1e9; PART.forEach((p, k) => { const d = (im.r[i] - p.c[0]) ** 2 + (im.g[i] - p.c[1]) ** 2 + (im.b[i] - p.c[2]) ** 2; if (d < bd) { bd = d; b = k; } }); lab[i] = b; }
    { const o = new Uint8Array(TOT); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const cnt = new Uint8Array(PART.length); for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const yy = AI.clamp(y + a, 0, N - 1), xx = AI.clamp(x + b, 0, N - 1); cnt[lab[yy * N + xx]]++; } let b = lab[y * N + x]; cnt.forEach((v, k) => { if (v > cnt[b]) b = k; }); o[y * N + x] = b; } lab = o; }
    const sceneC = document.createElement('canvas'); sceneC.width = sceneC.height = N;
    { const g = sceneC.getContext('2d'), d = g.createImageData(N, N); for (let i = 0; i < TOT; i++) { d.data[i * 4] = im.r[i]; d.data[i * 4 + 1] = im.g[i]; d.data[i * 4 + 2] = im.b[i]; d.data[i * 4 + 3] = 255; } g.putImageData(d, 0, 0); }
    const key = [(k) => k, (k) => PART[k].grp, (k) => PART[k].obj];
    function comp(seed, lvl) { const f = key[lvl], tgt = f(lab[seed]), m = new Uint8Array(TOT), stack = [seed]; m[seed] = 1; while (stack.length) { const i = stack.pop(), x = i % N, y = (i / N) | 0; for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue; const j = yy * N + xx; if (!m[j] && f(lab[j]) === tgt) { m[j] = 1; stack.push(j); } } } return m; }
    const prompts = [], names = ['1 · sub-part', '2 · part', '3 · whole object'], mini = [0, 1, 2].map(() => { const cv2 = document.createElement('canvas'); cv2.width = cv2.height = N; return cv2; });
    let masks = [], dist = [], maxD = [0, 0, 0], area = [0, 0, 0], fit = [0, 0, 0], sel = 0, mode = 'pos', t = 0, lastR = -1, drag = null, seeds = [], lastRing = -1;
    const modeBtns = { pos: '+ point', neg: '− point', box: 'box' }, mb = {};
    Object.entries(modeBtns).forEach(([k, v]) => (mb[k] = chipBtn(q('#mode'), v, () => { mode = k; Object.entries(mb).forEach(([kk, b]) => (b.style.borderColor = kk === k ? 'var(--blue)' : '')); })));
    mb.pos.style.borderColor = 'var(--blue)';
    function solve() {
      masks = [0, 1, 2].map(() => new Uint8Array(TOT)); seeds = []; fit = [0, 0, 0];
      const pos = prompts.filter((p) => p.t === 'pos'), neg = prompts.filter((p) => p.t === 'neg'), bx = prompts.find((p) => p.t === 'box');
      const sd = pos.map((p) => p.y * N + p.x); if (bx) sd.push(Math.floor((bx.y0 + bx.y1) / 2) * N + Math.floor((bx.x0 + bx.x1) / 2));
      sd.forEach((s) => [0, 1, 2].forEach((l) => { const m = comp(s, l); for (let i = 0; i < TOT; i++) if (m[i]) masks[l][i] = 1; }));
      neg.forEach((p) => { const m = comp(p.y * N + p.x, 0); [0, 1, 2].forEach((l) => { for (let i = 0; i < TOT; i++) if (m[i]) masks[l][i] = 0; }); });
      if (bx) [0, 1, 2].forEach((l) => { let mnx = 99, mny = 99, mxx = -1, mxy = -1; for (let i = 0; i < TOT; i++) { const x = i % N, y = (i / N) | 0; if (x < bx.x0 || x > bx.x1 || y < bx.y0 || y > bx.y1) masks[l][i] = 0; else if (masks[l][i]) { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mny = Math.min(mny, y); mxy = Math.max(mxy, y); } } if (mxx >= 0) { const ai = (mxx - mnx + 1) * (mxy - mny + 1), ab = (bx.x1 - bx.x0 + 1) * (bx.y1 - bx.y0 + 1); fit[l] = ai / ab; } });
      sd.forEach((s) => s); seeds = sd;
      dist = masks.map((m) => { const d = new Int16Array(TOT).fill(-1), qd = []; seeds.forEach((s) => { if (m[s] && d[s] < 0) { d[s] = 0; qd.push(s); } }); for (let h = 0; h < qd.length; h++) { const i = qd[h], x = i % N, y = (i / N) | 0; for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue; const j = yy * N + xx; if (m[j] && d[j] < 0) { d[j] = d[i] + 1; qd.push(j); } } } return d; });
      maxD = dist.map((d) => d.reduce((a, b) => Math.max(a, b), 0)); area = masks.map((m) => m.reduce((a, b) => a + b, 0));
      if (bx) { let bi = 0; fit.forEach((v, k) => { if (v > fit[bi]) bi = k; }); sel = bi; }
      t = 0; lastRing = -1;
    }
    const geo = () => { const s = Math.min(c.w * 0.6, c.h - 40); const rx = 12 + s + 16, mh = (c.h - 40 - 10) / 3; return { s, x0: 12, y0: 34, rx, mh, mw: Math.min(mh, c.w - rx - 10) }; };
    const toCell = (e) => { const p = ptr(e, cv), { s, x0, y0 } = geo(); return { x: AI.clamp(Math.floor(((p.x - x0) / s) * N), 0, N - 1), y: AI.clamp(Math.floor(((p.y - y0) / s) * N), 0, N - 1), px: p.x, py: p.y, inside: p.x >= x0 && p.y >= y0 && p.x < x0 + s && p.y < y0 + s }; };
    cv.addEventListener('pointerdown', (e) => {
      const cc = toCell(e), { rx, y0, mh, mw } = geo();
      if (cc.px >= rx) { const k = Math.floor((cc.py - y0) / (mh + 5)); if (k >= 0 && k < 3) sel = k; lastRing = -1; return; }
      if (!cc.inside) return; cv.setPointerCapture(e.pointerId);
      if (mode === 'box') drag = { x0: cc.x, y0: cc.y, x1: cc.x, y1: cc.y }; else { prompts.push({ t: mode, x: cc.x, y: cc.y }); solve(); }
    });
    cv.addEventListener('pointermove', (e) => { if (drag) { const cc = toCell(e); drag.x1 = cc.x; drag.y1 = cc.y; } });
    cv.addEventListener('pointerup', () => { if (drag) { const b = { t: 'box', x0: Math.min(drag.x0, drag.x1), x1: Math.max(drag.x0, drag.x1), y0: Math.min(drag.y0, drag.y1), y1: Math.max(drag.y0, drag.y1) }; drag = null; if (b.x1 - b.x0 > 2 && b.y1 - b.y0 > 2) { for (let i = prompts.length - 1; i >= 0; i--) if (prompts[i].t === 'box') prompts.splice(i, 1); prompts.push(b); solve(); } } });
    q('#clr').onclick = () => { prompts.length = 0; solve(); };
    q('#demo').onclick = () => { prompts.length = 0; prompts.push({ t: 'pos', x: 18, y: 31 }); mode = 'pos'; solve(); };
    const st = player(q, '× speed', () => { if (t < 4.2) t = [0, 1.2, 2.6, 4.2].find((b) => b > t + 1e-6) || 4.2; else t += 1 / 8; }, () => { t = 0; lastRing = -1; });
    solve();
    const STAGES = ['image encoder (cached)', 'prompt encoder', 'mask decoder', '3 masks'];
    AI.loop(root, (dt) => {
      const mult = st.rate(); if (st.playing && prompts.length) t += (dt * mult) * (t < 4.2 ? 1 / 1.0 : 1);
      const ring = t < 4.2 ? -1 : Math.floor((t - 4.2) * 8), g = c.ctx, C = AI.col(), { s, x0, y0, rx, mh, mw } = geo();
      if (ring !== lastRing) { lastRing = ring; masks.forEach((m, k) => { const g2 = mini[k].getContext('2d'), d = g2.createImageData(N, N), col = AI.rgb(C.blue); for (let i = 0; i < TOT; i++) { if (m[i] && dist[k][i] >= 0 && dist[k][i] <= ring) { d.data[i * 4] = col[0]; d.data[i * 4 + 1] = col[1]; d.data[i * 4 + 2] = col[2]; d.data[i * 4 + 3] = 255; } } g2.putImageData(d, 0, 0); }); }
      g.clearRect(0, 0, c.w, c.h);
      // pipeline strip
      const stg = !prompts.length ? -1 : t < 1.2 ? 0 : t < 2.6 ? 1 : t < 4.2 ? 2 : 3; g.font = '11px sans-serif'; g.textAlign = 'center';
      const sw = (c.w - 24) / 4; STAGES.forEach((nm, k) => { g.fillStyle = k === stg ? C.orange : C.bg2; g.globalAlpha = k === stg ? 0.9 : 1; g.fillRect(12 + k * sw, 4, sw - 6, 20); g.globalAlpha = 1; g.fillStyle = k === stg ? '#000' : C.muted; g.fillText(nm, 12 + k * sw + (sw - 6) / 2, 18); });
      g.imageSmoothingEnabled = false; g.drawImage(sceneC, x0, y0, s, s);
      if (prompts.length) { g.globalAlpha = 0.6; g.drawImage(mini[sel], x0, y0, s, s); g.globalAlpha = 1; }
      const cs = s / N; g.lineWidth = 2;
      prompts.forEach((p) => { if (p.t === 'box') { g.strokeStyle = C.orange; g.strokeRect(x0 + p.x0 * cs, y0 + p.y0 * cs, (p.x1 - p.x0 + 1) * cs, (p.y1 - p.y0 + 1) * cs); } else { g.fillStyle = p.t === 'pos' ? C.green : C.red; g.beginPath(); g.arc(x0 + (p.x + 0.5) * cs, y0 + (p.y + 0.5) * cs, 6, 0, 7); g.fill(); g.strokeStyle = '#fff'; g.stroke(); g.fillStyle = '#fff'; g.font = 'bold 11px sans-serif'; g.fillText(p.t === 'pos' ? '+' : '−', x0 + (p.x + 0.5) * cs, y0 + (p.y + 0.5) * cs + 4); } });
      if (drag) { g.strokeStyle = C.orange; g.setLineDash([4, 4]); const a = Math.min(drag.x0, drag.x1), b = Math.min(drag.y0, drag.y1); g.strokeRect(x0 + a * cs, y0 + b * cs, (Math.abs(drag.x1 - drag.x0) + 1) * cs, (Math.abs(drag.y1 - drag.y0) + 1) * cs); g.setLineDash([]); }
      g.lineWidth = 1;
      for (let k = 0; k < 3; k++) {
        const y = y0 + k * (mh + 5); g.globalAlpha = 0.35; g.drawImage(sceneC, rx, y, mw, mh); g.globalAlpha = 1; if (prompts.length) g.drawImage(mini[k], rx, y, mw, mh);
        g.strokeStyle = k === sel ? C.orange : C.border; g.lineWidth = k === sel ? 3 : 1; g.strokeRect(rx, y, mw, mh); g.lineWidth = 1;
        g.fillStyle = C.fg; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText(names[k], rx + 4, y + 13);
      }
      const pr = prompts.length;
      q('#out').innerHTML = pr ? `prompts: ${prompts.filter((p) => p.t === 'pos').length} positive, ${prompts.filter((p) => p.t === 'neg').length} negative, ${prompts.some((p) => p.t === 'box') ? 1 : 0} box\n` + [0, 1, 2].map((k) => `mask ${k + 1}: area <b>${area[k]}</b> px (${((100 * area[k]) / TOT).toFixed(1)}%)${prompts.some((p) => p.t === 'box') ? ` box-fit ${fit[k].toFixed(2)}` : ''}${k === sel ? ' ◀' : ''}`).join('\n') + (area[0] === area[1] && area[1] === area[2] ? '\n<span class="m">3 identical masks: no ambiguity here</span>' : '\n<span class="o">3 different sizes = ambiguity</span>') : '<span class="m">click on the picture to place a prompt</span>';
      q('#cap').textContent = !pr ? 'Pick a prompt type, then click the picture. Try the roof, the window, the sky — and then a window with a box around the whole house.' : t < 1.2 ? '① The heavy image encoder already ran ONCE for this picture and cached a 256×64×64 embedding — every new prompt reuses it (that is why it is interactive).' : t < 2.6 ? '② The prompt encoder turns your click into a small vector: a position code for (x, y) plus a learned "positive/negative" embedding (a box = two corner points).' : t < 4.2 ? '③ The light mask decoder lets the prompt tokens and the image embedding exchange information (2 two-way attention layers), then predicts 3 mask logits.' : ring <= Math.max(...maxD) ? `④ The masks grow outward from the prompt (ring ${ring}) — a pedagogical stand-in for the decoder's output. Three granularities: part, group, whole.` : '④ Done. Click a candidate on the right to view it; add a − point to carve something out, or a box to pick the granularity.';
    }, { fps: 30 });
  };

  /* ================= panoptic ================= */
  AI.labs.panoptic = function (root) {
    const { view, ctl } = AI.labShell(root, 'Instance matching and panoptic quality', 'drag the blue predictions onto the orange ground-truth segments (boxes stand in for masks)',
      '<canvas style="height:400px;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Resize selected prediction <span class="v" id="szv">1.0×</span><input type="range" id="sz" min="40" max="180" value="100"></label>
       <div class="chips" id="pre"></div>
       ${PLAY(5, 1, 20)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    const GT = [{ n: 'car A', k: 'car', x: 8, y: 12, w: 28, h: 18 }, { n: 'car B', k: 'car', x: 52, y: 14, w: 30, h: 20 }, { n: 'person', k: 'person', x: 20, y: 58, w: 12, h: 30 }];
    const mkP = () => [{ n: 'P1', k: 'car', x: 10, y: 14, w: 28, h: 18, w0: 28, h0: 18 }, { n: 'P2', k: 'car', x: 70, y: 22, w: 28, h: 20, w0: 28, h0: 20 }, { n: 'P3', k: 'person', x: 20, y: 61, w: 12, h: 30, w0: 12, h0: 30 }, { n: 'P4', k: 'car', x: 55, y: 62, w: 26, h: 18, w0: 26, h0: 18 }];
    let PR = mkP(), sel = 0, drag = null, pos = 0, scale = 1;
    const inter = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const iou = (a, b) => { const i = inter(a, b); return i / (a.w * a.h + b.w * b.h - i); };
    const geo = () => { const s = Math.min(c.h - 20, c.w * 0.62); return { s, k: s / 100, x0: 10, y0: 10 }; };
    cv.addEventListener('pointerdown', (e) => { const p = ptr(e, cv), { k, x0, y0 } = geo(), x = (p.x - x0) / k, y = (p.y - y0) / k; for (let i = PR.length - 1; i >= 0; i--) { const r = PR[i]; if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) { sel = i; drag = { dx: x - r.x, dy: y - r.y }; cv.setPointerCapture(e.pointerId); q('#sz').value = Math.round((r.w / r.w0) * 100); pos = 0; break; } } });
    cv.addEventListener('pointermove', (e) => { if (!drag) return; const p = ptr(e, cv), { k, x0, y0 } = geo(), r = PR[sel]; r.x = AI.clamp((p.x - x0) / k - drag.dx, 0, 100 - r.w); r.y = AI.clamp((p.y - y0) / k - drag.dy, 0, 100 - r.h); });
    cv.addEventListener('pointerup', () => { drag = null; pos = 0; });
    q('#sz').addEventListener('input', () => { const r = PR[sel], f = +q('#sz').value / 100; r.w = r.w0 * f; r.h = r.h0 * f; r.x = AI.clamp(r.x, 0, 100 - r.w); r.y = AI.clamp(r.y, 0, 100 - r.h); pos = 0; });
    chipBtn(q('#pre'), 'reset', () => { PR = mkP(); pos = 0; });
    chipBtn(q('#pre'), 'make P2 match car B', () => { PR[1].x = 53; PR[1].y = 15; PR[1].w = 30; PR[1].h = 20; pos = 0; });
    const st = player(q, 'steps / s', () => (pos = Math.min(5, Math.floor(pos) + 1)), () => (pos = 0));
    const ev = () => { const ms = GT.map((gt) => { let b = -1, bi = 0; PR.forEach((p, i) => { const v = p.k === gt.k ? iou(gt, p) : 0; if (v > bi) { bi = v; b = i; } }); return { b, v: bi, tp: bi > 0.5 }; }); return ms; };
    AI.loop(root, (dt) => {
      const rate = st.rate() * 0.5; if (st.playing && pos < 5) pos = Math.min(5, pos + dt * rate);
      const g = c.ctx, C = AI.col(), { s, k, x0, y0 } = geo(); g.clearRect(0, 0, c.w, c.h);
      g.fillStyle = C.bg2; g.fillRect(x0, y0, s, s); g.strokeStyle = C.border; g.strokeRect(x0, y0, s, s);
      const ms = ev(), step = Math.floor(pos), shown = Math.min(3, step);
      GT.forEach((r, i) => { g.strokeStyle = C.orange; g.lineWidth = i === step && step < 3 ? 4 : 2; g.setLineDash(i < shown ? [] : [5, 4]); g.strokeRect(x0 + r.x * k, y0 + r.y * k, r.w * k, r.h * k); g.setLineDash([]); g.fillStyle = C.orange; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('GT ' + r.n, x0 + r.x * k + 3, y0 + r.y * k + 12); });
      PR.forEach((r, i) => { const matched = ms.some((m, j) => m.tp && m.b === i && j < shown); g.fillStyle = C.blue; g.globalAlpha = matched ? 0.55 : 0.3; g.fillRect(x0 + r.x * k, y0 + r.y * k, r.w * k, r.h * k); g.globalAlpha = 1; g.strokeStyle = i === sel ? C.fg : C.blue; g.lineWidth = i === sel ? 2.5 : 1.5; g.strokeRect(x0 + r.x * k, y0 + r.y * k, r.w * k, r.h * k); g.lineWidth = 1; g.fillStyle = C.fg; g.fillText(`${r.n} (${r.k})`, x0 + r.x * k + 3, y0 + (r.y + r.h) * k - 4); });
      let tp = 0, sum = 0, fnn = 0; const usedP = new Set(); const lines = [];
      ms.forEach((m, j) => { if (j >= shown) return; if (m.tp) { tp++; sum += m.v; usedP.add(m.b); lines.push(`GT ${GT[j].n}: best ${PR[m.b].n}, IoU ${m.v.toFixed(2)} > 0.5 → <b class="g">TP</b>`); } else { fnn++; lines.push(`GT ${GT[j].n}: best IoU ${m.v.toFixed(2)} ≤ 0.5 → <b class="r">FN</b>`); } });
      let fp = 0; if (step >= 3) { PR.forEach((_, i) => { if (!usedP.has(i)) fp++; }); }
      const den = tp + 0.5 * fp + 0.5 * fnn, pq = step >= 4 && den ? sum / den : null, sq = tp ? sum / tp : 0, rq = den ? tp / den : 0;
      q('#out').innerHTML = (lines.join('\n') || '<span class="m">press Play / Step to match</span>') + (step >= 3 ? `\nunmatched predictions → <b class="r">FP = ${fp}</b>` : '') + (pq !== null ? `\nSQ = ΣIoU/TP = ${sum.toFixed(2)}/${tp} = <b>${sq.toFixed(3)}</b>\nRQ = TP/(TP+½FP+½FN) = ${tp}/${den.toFixed(1)} = <b>${rq.toFixed(3)}</b>\nPQ = SQ·RQ = <b class="o">${(sq * rq).toFixed(3)}</b>` : '');
      q('#cap').textContent = step < 3 ? `Step ${step + 1}: take ground-truth segment "${GT[step].n}", find the same-class prediction with the highest IoU; if IoU > 0.5 the match is valid (and unique).` : step === 3 ? 'Every prediction that matched nothing is a false positive.' : 'PQ = (mean IoU of matches) × (a detection F1-like term). Drag the blue boxes and watch PQ change.';
      g.textAlign = 'left'; g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText(`IoU(P${sel + 1}, nearest GT same class) = ${Math.max(0, ...GT.filter((r) => r.k === PR[sel].k).map((r) => iou(r, PR[sel]))).toFixed(2)}`, x0, y0 + s + 18);
      void scale;
    }, { fps: 30 });
  };

  /* ================= heatpose ================= */
  AI.labs.heatpose = function (root) {
    const J = [['head', 64, 18], ['neck', 64, 34], ['l-elbow', 44, 46], ['l-hand', 30, 60], ['r-elbow', 84, 46], ['r-hand', 98, 60], ['hip', 64, 70], ['l-foot', 50, 108], ['r-foot', 78, 108]];
    const BONES = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [1, 6], [6, 7], [6, 8]], MIR = { 2: 4, 3: 5, 4: 2, 5: 3, 7: 8, 8: 7 };
    const { view, ctl } = AI.labShell(root, 'Heatmap keypoints: decode by arg-max', 'the net outputs one heatmap per joint; we find the peak and scale back to the image',
      '<canvas style="height:380px;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Heatmap blob width σ <span class="v" id="sgv"></span><input type="range" id="sg" min="2" max="14" value="6"></label>
       <label>Noise <span class="v" id="nzv"></span><input type="range" id="nz" min="0" max="60" value="12"></label>
       <label>Stride (heatmap = 128/stride) <select id="S"><option>2</option><option selected>4</option><option>8</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="dis" checked> left/right confusion (distractor blob)</label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="sub" checked> sub-pixel refinement (¼-stride shift)</label>
       ${PLAY(10, 1, 40)}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    const gt = J.map((j) => ({ x: j[1], y: j[2] })), pred = J.map(() => null);
    let H = [], Hn = 16, ji = 0, scan = 0, phase = 0, hold = 0, dragJ = -1, hmC = document.createElement('canvas'), best = null, dirty = true;
    const rnd = (j, i) => { const v = Math.sin((j + 1) * 91.7 + i * 12.9898) * 43758.5453; return v - Math.floor(v); };
    function heat(j) {
      const S = +q('#S').value, n = 128 / S, sg = +q('#sg').value, nz = +q('#nz').value / 100, h = new Float32Array(n * n), dis = q('#dis').checked && MIR[j] !== undefined;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const cx = (x + 0.5) * S, cy = (y + 0.5) * S; let v = Math.exp(-((cx - gt[j].x) ** 2 + (cy - gt[j].y) ** 2) / (2 * sg * sg)); if (dis) { const m = gt[MIR[j]]; v += 0.7 * Math.exp(-((cx - m.x) ** 2 + (cy - m.y) ** 2) / (2 * sg * sg)); } v += nz * rnd(j, y * n + x) * 0.6; h[y * n + x] = v; }
      return h;
    }
    function prep() {
      const S = +q('#S').value; Hn = 128 / S; H = J.map((_, j) => heat(j)); q('#sgv').textContent = q('#sg').value; q('#nzv').textContent = (+q('#nz').value / 100).toFixed(2);
      hmC.width = hmC.height = Hn; const g = hmC.getContext('2d'), d = g.createImageData(Hn, Hn), h = H[ji]; let mx = 0; h.forEach((v) => (mx = Math.max(mx, v)));
      for (let i = 0; i < Hn * Hn; i++) { const v = h[i] / mx, r = Math.min(255, 40 + 600 * v * v), gg = Math.min(255, 255 * Math.max(0, v * 1.5 - 0.4)), b = Math.max(0, 200 - 400 * v); d.data[i * 4] = r; d.data[i * 4 + 1] = gg; d.data[i * 4 + 2] = b; d.data[i * 4 + 3] = 255; } g.putImageData(d, 0, 0);
      scan = 0; phase = 0; best = null; pred[ji] = null; dirty = true;
    }
    ['#sg', '#nz', '#S', '#dis'].forEach((s) => q(s).addEventListener('input', () => { pred.fill(null); ji = 0; prep(); }));
    const decodeAll = () => {
      const S = +q('#S').value, h = H[ji]; let bi = 0; for (let i = 1; i < Hn * Hn; i++) if (h[i] > h[bi]) bi = i;
      const bx = bi % Hn, by = (bi / Hn) | 0; let ox = 0, oy = 0;
      if (q('#sub').checked) { const L = bx > 0 ? h[by * Hn + bx - 1] : -1e9, R = bx < Hn - 1 ? h[by * Hn + bx + 1] : -1e9, U = by > 0 ? h[(by - 1) * Hn + bx] : -1e9, D = by < Hn - 1 ? h[(by + 1) * Hn + bx] : -1e9; ox = (R > L ? 0.25 : -0.25) * S; oy = (D > U ? 0.25 : -0.25) * S; }
      return { bi, bx, by, v: h[bi], px: (bx + 0.5) * S + ox, py: (by + 0.5) * S + oy, rx: (bx + 0.5) * S, ry: (by + 0.5) * S };
    };
    const geo = () => { const s = Math.min((c.w - 36) / 2, c.h - 50); return { s, k: s / 128, x0: 12, y0: 30, hx: 12 + s + 20 }; };
    cv.addEventListener('pointerdown', (e) => { const p = ptr(e, cv), { k, x0, y0 } = geo(); let b = -1, bd = 14; gt.forEach((j, i) => { const d = Math.hypot(x0 + j.x * k - p.x, y0 + j.y * k - p.y); if (d < bd) { bd = d; b = i; } }); if (b >= 0) { dragJ = b; cv.setPointerCapture(e.pointerId); } });
    cv.addEventListener('pointermove', (e) => { if (dragJ < 0) return; const p = ptr(e, cv), { k, x0, y0 } = geo(); gt[dragJ].x = AI.clamp((p.x - x0) / k, 2, 126); gt[dragJ].y = AI.clamp((p.y - y0) / k, 2, 126); });
    cv.addEventListener('pointerup', () => { if (dragJ >= 0) { dragJ = -1; pred.fill(null); ji = 0; prep(); } });
    const adv = () => { phase = 5; };
    const st = player(q, 'speed', () => { if (phase < 4) { scan = Hn * Hn; phase = 4; best = decodeAll(); pred[ji] = best; } else { ji = (ji + 1) % J.length; prep(); } }, () => { pred.fill(null); ji = 0; prep(); });
    prep();
    AI.loop(root, (dt) => {
      const rate = st.rate(), g = c.ctx, C = AI.col(), { s, k, x0, y0, hx } = geo(), S = +q('#S').value, cs = s / Hn;
      if (st.playing) {
        if (phase === 0) { scan += dt * rate * 12 * (Hn / 16); if (scan >= Hn * Hn) { scan = Hn * Hn; phase = 1; hold = 0; best = decodeAll(); } }
        else if (phase < 4) { hold += dt * rate; if (hold > 1.1) { hold = 0; phase++; if (phase === 4) pred[ji] = best; } }
        else { hold += dt * rate; if (hold > 1.3) { hold = 0; if (ji < J.length - 1) { ji++; prep(); } else { phase = 5; } } }
      }
      // heat canvas refresh when joint changes handled in prep; also refresh on selection change
      g.clearRect(0, 0, c.w, c.h); g.font = '12px sans-serif'; g.fillStyle = C.muted; g.textAlign = 'left';
      g.fillText('image 128×128 (drag orange joints)', x0, 18); g.fillText(`heatmap of "${J[ji][0]}" ${Hn}×${Hn} (stride ${S})`, hx, 18);
      g.fillStyle = C.bg2; g.fillRect(x0, y0, s, s);
      g.strokeStyle = C.orange; g.lineWidth = 3; BONES.forEach(([a, b]) => { g.beginPath(); g.moveTo(x0 + gt[a].x * k, y0 + gt[a].y * k); g.lineTo(x0 + gt[b].x * k, y0 + gt[b].y * k); g.stroke(); });
      g.strokeStyle = C.blue; g.lineWidth = 2; BONES.forEach(([a, b]) => { if (pred[a] && pred[b]) { g.beginPath(); g.moveTo(x0 + pred[a].px * k, y0 + pred[a].py * k); g.lineTo(x0 + pred[b].px * k, y0 + pred[b].py * k); g.stroke(); } }); g.lineWidth = 1;
      gt.forEach((j, i) => { g.fillStyle = i === ji ? '#fff' : C.orange; g.beginPath(); g.arc(x0 + j.x * k, y0 + j.y * k, i === ji ? 6 : 4.5, 0, 7); g.fill(); });
      pred.forEach((p, i) => { if (p) { g.fillStyle = C.blue; g.beginPath(); g.arc(x0 + p.px * k, y0 + p.py * k, 4, 0, 7); g.fill(); } });
      g.imageSmoothingEnabled = false; g.drawImage(hmC, hx, y0, s, s); g.strokeStyle = C.border; g.strokeRect(hx, y0, s, s);
      const n = Math.floor(scan);
      if (phase === 0) { // scanned area veil + cursor
        g.fillStyle = C.bg; g.globalAlpha = 0.85; const cr = Math.floor(n / Hn); g.fillRect(hx, y0 + (cr + 1) * cs, s, s - (cr + 1) * cs); g.fillRect(hx + ((n % Hn) + 1) * cs, y0 + cr * cs, s - ((n % Hn) + 1) * cs, cs); g.globalAlpha = 1;
        g.strokeStyle = '#fff'; g.lineWidth = 2; g.strokeRect(hx + (n % Hn) * cs, y0 + Math.floor(n / Hn) * cs, cs, cs); g.lineWidth = 1;
        const h = H[ji]; let bi = 0; for (let i = 1; i < Math.min(n, Hn * Hn); i++) if (h[i] > h[bi]) bi = i; if (n > 0) { g.strokeStyle = C.green; g.lineWidth = 2.5; g.strokeRect(hx + (bi % Hn) * cs, y0 + Math.floor(bi / Hn) * cs, cs, cs); g.lineWidth = 1; }
      } else if (best) { g.strokeStyle = C.green; g.lineWidth = 3; g.strokeRect(hx + best.bx * cs, y0 + best.by * cs, cs, cs); g.lineWidth = 1; }
      const errTrue = best ? Math.hypot(best.px - gt[ji].x, best.py - gt[ji].y) : 0, errNo = best ? Math.hypot(best.rx - gt[ji].x, best.ry - gt[ji].y) : 0;
      q('#out').innerHTML = best && phase >= 1 ? `joint <b>${J[ji][0]}</b>\narg-max cell = (<b class="g">${best.bx}</b>, <b class="g">${best.by}</b>), value ${best.v.toFixed(2)}\n× stride: (${best.bx}+½)·${S} = ${best.rx}, (${best.by}+½)·${S} = ${best.ry}\n${q('#sub').checked ? `sub-pixel: ${best.px.toFixed(1)}, ${best.py.toFixed(1)}\n` : ''}true position (${gt[ji].x.toFixed(1)}, ${gt[ji].y.toFixed(1)})\nerror <b class="o">${errTrue.toFixed(1)} px</b> ${q('#sub').checked ? `(without refinement ${errNo.toFixed(1)})` : ''}\n<span class="m">quantisation error ≤ stride/2 = ${S / 2} px</span>` : `joint <b>${J[ji][0]}</b>\nscanning ${n} / ${Hn * Hn} cells…\nbest so far kept in green`;
      q('#cap').textContent = phase === 0 ? `① scan every heatmap cell and keep the largest value (green box). A GPU does this with one arg-max.` : phase === 1 ? '② peak found. With a left/right distractor the wrong blob can win if σ is large and noise high.' : phase === 2 ? '③ sub-pixel refinement: move ¼ of a stride towards the larger neighbour.' : phase === 3 ? '④ multiply cell coordinates by the stride to get image pixels.' : phase === 4 ? '⑤ the decoded joint joins the predicted skeleton (blue); next joint follows.' : 'All joints decoded. Drag an orange joint, increase noise or stride, and compare orange vs blue.';
      void adv;
    }, { fps: 30 });
  };
})();
