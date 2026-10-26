/* Computer-vision Module 5 labs: patchify, posembed, attnlab, attncost, swinwin, swinstages, mae, clip, rfgrowth.
 * Also registers extra ViT presets for the archviz lab. Requires lib.js, arch.js, labs-cv.js.
 */
(function () {
  const CV = AI.cv;
  const SCENES = ['house', 'shapes', 'digit', 'edges', 'checker'];
  const sceneOpts = () => SCENES.map((k) => `<option value="${k}">${CV.SCENES[k]}</option>`).join('');

  /* ---------- extra presets ---------- */
  if (window.AIArch) {
    const vit = (P, D, h, m, L) => `input 3 224 224\npatch ${P} ${D}\ncls\npos\ntblock ${h} ${m} x${L}\nnorm\nseqfc 1000`;
    Object.assign(AIArch.PRESETS, {
      'vit-ti16': { title: 'ViT-Ti/16 (DeiT-Tiny)', spec: vit(16, 192, 3, 768, 12) },
      'vit-s16': { title: 'ViT-S/16 (DeiT-Small)', spec: vit(16, 384, 6, 1536, 12) },
      'vit-b32': { title: 'ViT-B/32', spec: vit(32, 768, 12, 3072, 12) },
      'vit-l16': { title: 'ViT-L/16', spec: vit(16, 1024, 16, 4096, 24) },
      'hybrid-r50-vit': { title: 'Hybrid: ResNet-50 stages 1-3 + ViT-B encoder', spec: `input 3 224 224\nconv 64 k7 s2\nbn\nrelu\nmaxpool k3 s2 p1\nbottleneck 256 mid64\nbottleneck 256 mid64 x2\nbottleneck 512 mid128 s2\nbottleneck 512 mid128 x3\nbottleneck 1024 mid256 s2\nbottleneck 1024 mid256 x5\npatch 1 768\ncls\npos\ntblock 12 3072 x12\nnorm\nseqfc 1000` },
    });
  }

  /* ---------- shared helpers ---------- */
  const scv = {};
  const sceneCanvas = (k) => {
    if (scv[k]) return scv[k];
    const d = CV.scene(k), c = document.createElement('canvas'); c.width = c.height = 120;
    const g = c.getContext('2d'), im = g.createImageData(120, 120); im.data.set(d); g.putImageData(im, 0, 0);
    return (scv[k] = c);
  };
  const gmc = {};
  /** mean colour (0..255, interleaved rgb) of each cell of a g x g grid over the 120px scene */
  function gridMean(kind, g) {
    const key = kind + g; if (gmc[key]) return gmc[key];
    const d = CV.scene(kind), out = new Float32Array(g * g * 3);
    for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) {
      const x0 = Math.floor((x * 120) / g), x1 = Math.max(x0 + 1, Math.floor(((x + 1) * 120) / g)), y0 = Math.floor((y * 120) / g), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * 120) / g));
      let r = 0, gg = 0, b = 0, n = 0;
      for (let j = y0; j < y1; j++) for (let i = x0; i < x1; i++) { const k = (j * 120 + i) * 4; r += d[k]; gg += d[k + 1]; b += d[k + 2]; n++; }
      const o = (y * g + x) * 3; out[o] = r / n; out[o + 1] = gg / n; out[o + 2] = b / n;
    }
    return (gmc[key] = out);
  }
  const rgbs = (a, i) => `rgb(${Math.round(a[i * 3])},${Math.round(a[i * 3 + 1])},${Math.round(a[i * 3 + 2])})`;
  const drawScene = (g, k, x, y, w, h) => { g.imageSmoothingEnabled = true; g.drawImage(sceneCanvas(k), x, y, w, h); };
  const playHTML = (def, max) => `<label>Speed <span class="v" id="spv">×${def}</span><input type="range" id="sp" min="0.1" max="${max || 3}" step="0.05" value="${def}"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Restart</button></div>`;
  function wirePlay(root, st, onStep, onReset) {
    const q = (s) => AI.$(s, root);
    st.playing = true; st.speed = +q('#sp').value;
    q('#sp').oninput = () => { st.speed = +q('#sp').value; q('#spv').textContent = '×' + st.speed; };
    const setP = (p) => { st.playing = p; q('#play').textContent = p ? '⏸ Pause' : '▶ Play'; };
    q('#play').onclick = () => setP(!st.playing);
    q('#step').onclick = () => { setP(false); onStep(); };
    q('#reset').onclick = () => onReset();
    st.setPlay = setP;
  }
  const sm = (a) => { const m = Math.max(...a); let s = 0; const e = a.map((x) => { const v = x === -Infinity ? 0 : Math.exp(x - m); s += v; return v; }); return e.map((v) => v / s); };
  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
  const gfmt = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' G' : n >= 1e6 ? (n / 1e6).toFixed(2) + ' M' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' k' : String(Math.round(n)));
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };

  /* ================= patchify ================= */
  AI.labs.patchify = function (root) {
    const PS = [8, 14, 16, 28, 32, 56];
    const { view } = AI.labShell(root, 'Image → patches → tokens', 'each patch is flattened, projected by one shared matrix, and joins the sequence',
      '<canvas style="height:500px"></canvas><div class="caption" id="cap"></div>',
      `<label>Scene <select id="scene">${sceneOpts()}</select></label>
       <label>Image side H = W <select id="H"><option>96</option><option>112</option><option>160</option><option selected>224</option><option>384</option></select></label>
       <label>Patch size P <span class="v" id="pv"></span><input type="range" id="P" min="0" max="${PS.length - 1}" value="4"></label>
       <label>Token width D <select id="D"><option>192</option><option>384</option><option selected>768</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="cls"> add a [CLS] token</label>
       ${playHTML(1, 12)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    const rg = AI.rng(7), R = Array.from({ length: 12 }, () => Array.from({ length: 108 }, () => rg() * 2 - 1));
    let H, P, g, n, D, scene, pos = 0, hold = 0, cls = false;
    function setup() {
      H = +q('#H').value; P = PS[+q('#P').value]; g = Math.floor(H / P); n = g * g; D = +q('#D').value; scene = q('#scene').value; cls = q('#cls').checked; pos = 0; hold = 0;
      q('#pv').textContent = P + '×' + P;
      const note = H % P ? `\n<span class="r">H is not divisible by P: ${g}·${P} = ${g * P} px used (crop)</span>` : '';
      q('#out').innerHTML = `tokens N = (H/P)² = (${H}/${P})² = <b class="o">${n}</b>${cls ? ' (+1 CLS = ' + (n + 1) + ')' : ''}\nflattened patch: ${P}·${P}·3 = <b>${P * P * 3}</b> numbers\nprojection E: ${P * P * 3} × ${D}\n  = <b>${fmtInt(P * P * 3 * D + D)}</b> params\n  MACs = N·P²·3·D = ${gfmt(n * P * P * 3 * D)}\nsequence: ${n + (cls ? 1 : 0)} × ${D} = ${fmtInt((n + (cls ? 1 : 0)) * D)} numbers\nimage: ${H}·${H}·3 = ${fmtInt(H * H * 3)} numbers\nattention pairs N² = ${fmtInt(n * n)}${note}`;
    }
    ['#scene', '#H', '#P', '#D', '#cls'].forEach((s) => q(s).addEventListener('input', setup));
    wirePlay(root, st, () => { pos = Math.min(n, Math.floor(pos) + 1); hold = 0; }, () => { pos = 0; hold = 0; });
    setup();
    const smp = (px, py, gx, gy) => { // 6x6 samples of patch (gx,gy)
      const d = CV.scene(scene), o = [];
      for (let j = 0; j < 6; j++) for (let i = 0; i < 6; i++) {
        const X = Math.min(119, Math.floor(((gx + (i + 0.5) / 6) / g) * 120)), Y = Math.min(119, Math.floor(((gy + (j + 0.5) / 6) / g) * 120)), k = (Y * 120 + X) * 4;
        o.push([d[k], d[k + 1], d[k + 2]]);
      }
      return o;
    };
    AI.loop(root, (dt) => {
      if (st.playing) { if (pos < n) pos = Math.min(n, pos + dt * 2 * st.speed); else { hold += dt; if (hold > 4) { pos = 0; hold = 0; } } }
      const C = AI.col(), gx = c.ctx, W = c.w, Ht = c.h; gx.clearRect(0, 0, W, Ht);
      const topH = Math.round(Ht * 0.6), s = Math.min(W * 0.4, topH - 30), ix = 10, iy = 24, cell = s / g;
      txt(gx, `image ${H}×${H}×3  →  ${g}×${g} patches`, ix, 16, C.muted);
      drawScene(gx, scene, ix, iy, s, s);
      const done = Math.floor(pos), cur = done < n ? done : -1, fr = pos - done;
      if (cell >= 3) {
        gx.strokeStyle = 'rgba(160,160,160,.8)'; gx.lineWidth = 1; gx.beginPath();
        for (let i = 0; i <= g; i++) { gx.moveTo(ix + i * cell, iy); gx.lineTo(ix + i * cell, iy + s); gx.moveTo(ix, iy + i * cell); gx.lineTo(ix + s, iy + i * cell); }
        gx.stroke();
      }
      gx.fillStyle = 'rgba(46,160,67,.28)';
      for (let k = 0; k < done; k++) gx.fillRect(ix + (k % g) * cell, iy + Math.floor(k / g) * cell, cell, cell);
      // tray
      const trayY = topH + 20, tot = n + (cls ? 1 : 0), area = (W - 20) * (Ht - trayY - 6), t = Math.max(3, Math.min(26, Math.floor(Math.sqrt(area / tot)) - 2)), per = Math.max(1, Math.floor((W - 20) / (t + 2)));
      const slot = (k) => [10 + (k % per) * (t + 2), trayY + Math.floor(k / per) * (t + 2)];
      txt(gx, `token sequence (${done + (cls ? 1 : 0)} of ${tot} placed)`, 10, trayY - 6, C.muted);
      const gm = gridMean(scene, g), off = cls ? 1 : 0;
      if (cls) { const [x, y] = slot(0); gx.fillStyle = C.orange; gx.fillRect(x, y, t, t); }
      for (let k = 0; k < done; k++) { const [x, y] = slot(k + off); gx.fillStyle = rgbs(gm, k); gx.fillRect(x, y, t, t); }
      let cap = 'All patches are now tokens: an N×D matrix, ready for the transformer.';
      if (cur >= 0) {
        const px = cur % g, py = Math.floor(cur / g), sx = ix + px * cell, sy = iy + py * cell;
        gx.strokeStyle = C.blue; gx.lineWidth = 3; gx.strokeRect(sx, sy, cell, cell);
        const dx = ix + s + 26, dw = W - dx - 10, ps = 64, pyy = iy + 6, px0 = dx, samp = smp(0, 0, px, py);
        // patch position (flying in first 0.3)
        const u = AI.smooth(fr / 0.3), X = AI.lerp(sx, px0, u), Y = AI.lerp(sy, pyy, u), S = AI.lerp(cell, ps, u);
        for (let j = 0; j < 6; j++) for (let i = 0; i < 6; i++) { const v = samp[j * 6 + i]; gx.fillStyle = `rgb(${v[0] | 0},${v[1] | 0},${v[2] | 0})`; gx.fillRect(X + (i * S) / 6, Y + (j * S) / 6, S / 6 + 0.5, S / 6 + 0.5); }
        gx.strokeStyle = C.blue; gx.lineWidth = 2; gx.strokeRect(X, Y, S, S);
        txt(gx, `patch ${cur + 1}: ${P}×${P}×3`, px0 + ps + 10, pyy + 14, C.fg);
        txt(gx, `row ${py}, col ${px}`, px0 + ps + 10, pyy + 30, C.muted);
        // strip
        const sy2 = pyy + ps + 30, cw = Math.max(3, Math.min(14, (dw - 6) / 36));
        txt(gx, `① flatten → ${P * P * 3} numbers (first 36 shown)`, dx, sy2 - 6, C.muted);
        const nShow = fr < 0.3 ? 0 : fr < 0.55 ? Math.ceil(((fr - 0.3) / 0.25) * 36) : 36;
        for (let k = 0; k < 36; k++) { const v = samp[k]; gx.fillStyle = k < nShow ? `rgb(${v[0] | 0},${v[1] | 0},${v[2] | 0})` : C.bg2; gx.fillRect(dx + k * cw, sy2, cw - 1, 16); }
        if (nShow < 36 && nShow > 0) { const k = nShow - 1, i = k % 6, j = Math.floor(k / 6); gx.strokeStyle = C.orange; gx.lineWidth = 2; gx.strokeRect(px0 + (i * ps) / 6, pyy + (j * ps) / 6, ps / 6, ps / 6); }
        // projection
        const by = sy2 + 40;
        txt(gx, `② multiply by E (${P * P * 3} × ${D})`, dx, by - 6, C.muted);
        gx.fillStyle = fr >= 0.55 && fr < 0.8 ? AI.mix(C.bg2, C.purple, 0.6) : C.bg2; gx.fillRect(dx, by, Math.min(dw, 210), 22); gx.strokeStyle = C.purple; gx.lineWidth = 1.5; gx.strokeRect(dx, by, Math.min(dw, 210), 22);
        txt(gx, `token = x·E + b`, dx + 8, by + 15, C.fg, '12px ui-monospace,Menlo,monospace');
        // token bar
        const ty = by + 52, f = [];
        for (let k = 0; k < 12; k++) { let a = 0; for (let j = 0; j < 36; j++) for (let ch = 0; ch < 3; ch++) a += R[k][j * 3 + ch] * (samp[j][ch] / 127.5 - 1); f.push(Math.tanh(a / 6)); }
        txt(gx, `③ token: a ${D}-vector (12 shown)`, dx, ty - 6, C.muted);
        const nT = fr < 0.55 ? 0 : fr < 0.8 ? Math.ceil(((fr - 0.55) / 0.25) * 12) : 12, tw = Math.min(18, (dw - 6) / 12);
        for (let k = 0; k < 12; k++) {
          const v = f[k]; if (fr >= 0.8) break; gx.fillStyle = k < nT ? (v > 0 ? AI.mix(C.bg2, C.orange, Math.min(1, v * 1.5)) : AI.mix(C.bg2, C.blue, Math.min(1, -v * 1.5))) : C.bg2;
          gx.fillRect(dx + k * tw, ty, tw - 1, 16);
        }
        if (fr >= 0.8) { const e = AI.smooth((fr - 0.8) / 0.2), sl = slot(cur + off); gx.fillStyle = rgbs(gm, cur); gx.fillRect(AI.lerp(dx, sl[0], e), AI.lerp(ty, sl[1], e), AI.lerp(tw * 12, t, e), AI.lerp(16, t, e)); }
        cap = fr < 0.3 ? `Patch ${cur + 1} of ${n} is cut out of the image.` : fr < 0.55 ? `① Flatten: the ${P}×${P}×3 block is unrolled into one vector of ${P * P * 3} numbers.` : fr < 0.8 ? `② Project: the vector is multiplied by the shared matrix E (${P * P * 3}×${D}) — the same E for every patch.` : `③ The ${D}-dimensional token joins the sequence at position ${cur + 1 + off}.`;
      }
      if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
    }, { fps: 30 });
  };

  /* ================= posembed ================= */
  AI.labs.posembed = function (root) {
    const { view } = AI.labShell(root, 'Position embeddings', 'attention alone cannot tell where a patch came from; click a patch to see how similar its position code is to the others',
      '<canvas style="height:380px;cursor:pointer"></canvas><div class="caption" id="cap"></div>',
      `<label>Scene <select id="scene">${sceneOpts()}</select></label>
       <label>Grid <span class="v" id="gv"></span><input type="range" id="g" min="4" max="12" value="8"></label>
       <label>Position code <select id="mode"><option value="none">none</option><option value="rand">random (untrained table)</option><option value="sin1d">1-D sinusoidal (raster index)</option><option value="sin2d" selected>2-D sinusoidal (row + column)</option></select></label>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    const DIM = 32; let g, scene, mode, sel = 0, perm = [], shuffled = false, u = 0, tmr = 0, seed = 1, content = [], pes = [], delta = 0;
    const Wc = Array.from({ length: DIM }, (_, i) => { const r = AI.rng(100 + i); return [AI.randn(r), AI.randn(r), AI.randn(r)]; });
    function peVec(slot) {
      const v = new Array(DIM).fill(0); if (mode === 'none') return v;
      if (mode === 'rand') { const r = AI.rng(slot * 7919 + 13); return v.map(() => AI.randn(r)); }
      if (mode === 'sin1d') { for (let k = 0; k < 16; k++) { const f = 1 / Math.pow(1.45, k); v[k] = Math.sin(slot * f); v[16 + k] = Math.cos(slot * f); } return v; }
      const r = Math.floor(slot / g), cc = slot % g;
      for (let k = 0; k < 8; k++) { const f = 1 / Math.pow(1.8, k); v[k] = Math.sin(r * f); v[8 + k] = Math.cos(r * f); v[16 + k] = Math.sin(cc * f); v[24 + k] = Math.cos(cc * f); }
      return v;
    }
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    function attnOut(order, a) { // order[slot] = patch content index; query = patch a
      const X = order.map((p, s) => content[p].map((x, i) => x + pes[s][i])), sa = order.indexOf(a), w = sm(X.map((x) => dot(X[sa], x) / Math.sqrt(DIM)));
      const o = new Array(DIM).fill(0); X.forEach((x, j) => x.forEach((v, i) => (o[i] += w[j] * v))); return o;
    }
    function setup() {
      g = +q('#g').value; scene = q('#scene').value; mode = q('#mode').value; q('#gv').textContent = g + '×' + g; sel = Math.min(sel, g * g - 1);
      const gm = gridMean(scene, g); content = []; for (let i = 0; i < g * g; i++) { const f = [gm[i * 3] / 127.5 - 1, gm[i * 3 + 1] / 127.5 - 1, gm[i * 3 + 2] / 127.5 - 1]; const v = Wc.map((w) => dot(w, f)); const nrm = Math.sqrt(dot(v, v)) || 1; content.push(v.map((x) => (x / nrm) * 3)); }
      pes = []; for (let i = 0; i < g * g; i++) pes.push(peVec(i));
      newPerm(); refresh();
    }
    function newPerm() { const r = AI.rng(seed++ * 31 + 5); perm = [...Array(g * g).keys()]; for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; } }
    function refresh() {
      const id = [...Array(g * g).keys()], a = attnOut(id, sel), b = attnOut(perm, sel);
      delta = Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));
      q('#out').innerHTML = `selected patch: row ${Math.floor(sel / g)}, col ${sel % g}\nattention output of that patch,\nordered vs shuffled image:\n  Δ = <b class="${delta < 1e-9 ? 'g' : 'o'}">${AI.fmt(delta, 3)}</b>\n<span class="m">${mode === 'none' ? 'no position code: output identical,\nattention sees an unordered set' : 'position codes change the output:\nthe model can tell the arrangement'}</span>`;
    }
    ['#scene', '#g', '#mode'].forEach((s) => q(s).addEventListener('input', setup));
    wirePlay(root, st, () => { if (u <= 0.01 || !shuffled) { newPerm(); refresh(); } shuffled = !shuffled; }, () => { shuffled = false; tmr = 0; });
    const geom = () => { const ps = Math.min(c.w * 0.47, c.h - 44); return { ps, x1: 8, x2: c.w - ps - 8, y: 26 }; };
    c.cv.addEventListener('pointerdown', (e) => {
      const r = c.cv.getBoundingClientRect(), { ps, x1, x2, y } = geom(), mx = e.clientX - r.left, my = e.clientY - r.top;
      for (const x0 of [x1, x2]) if (mx >= x0 && mx < x0 + ps && my >= y && my < y + ps) { sel = Math.floor((my - y) / (ps / g)) * g + Math.floor((mx - x0) / (ps / g)); refresh(); }
    });
    setup();
    AI.loop(root, (dt) => {
      if (st.playing) { tmr += dt * st.speed; const cyc = tmr % 12; const want = cyc >= 3 && cyc < 9; if (want !== shuffled) { if (want) { newPerm(); refresh(); } shuffled = want; } }
      u += ((shuffled ? 1 : 0) - u) * Math.min(1, dt * 1.6 * st.speed);
      const C = AI.col(), gx = c.ctx, { ps, x1, x2, y } = geom(), cell = ps / g, gm = gridMean(scene, g); gx.clearRect(0, 0, c.w, c.h);
      txt(gx, shuffled || u > 0.05 ? 'patches shuffled' : 'image patches in order', x1, 16, C.muted);
      txt(gx, mode === 'none' ? 'position code: none' : 'similarity of position codes to the selected slot', x2, 16, C.muted);
      const ue = AI.smooth(u), slotOf = []; perm.forEach((p, s) => (slotOf[p] = s));
      for (let k = 0; k < g * g; k++) {
        const s1 = k, s2 = slotOf[k], X = AI.lerp(s1 % g, s2 % g, ue) * cell, Y = AI.lerp(Math.floor(s1 / g), Math.floor(s2 / g), ue) * cell;
        gx.fillStyle = rgbs(gm, k); gx.fillRect(x1 + X, y + Y, cell + 0.5, cell + 0.5);
      }
      gx.strokeStyle = 'rgba(150,150,150,.7)'; gx.lineWidth = 1; gx.beginPath(); for (let i = 0; i <= g; i++) { gx.moveTo(x1 + i * cell, y); gx.lineTo(x1 + i * cell, y + ps); gx.moveTo(x1, y + i * cell); gx.lineTo(x1 + ps, y + i * cell); } gx.stroke();
      const nrm = (v) => Math.sqrt(dot(v, v)) || 1, ps0 = pes[sel];
      for (let k = 0; k < g * g; k++) {
        const X = x2 + (k % g) * cell, Y = y + Math.floor(k / g) * cell;
        if (mode === 'none') { gx.fillStyle = C.bg2; } else { const cs = dot(ps0, pes[k]) / (nrm(ps0) * nrm(pes[k])); gx.fillStyle = cs >= 0 ? AI.mix(C.bg2, C.orange, Math.min(1, cs)) : AI.mix(C.bg2, C.blue, Math.min(1, -cs)); if (cell >= 30) { gx.fillRect(X, Y, cell, cell); txt(gx, cs.toFixed(2), X + cell / 2, Y + cell / 2 + 4, C.fg, '10px ui-monospace,monospace', 'center'); continue; } }
        gx.fillRect(X, Y, cell, cell);
      }
      for (const x0 of [x1, x2]) { gx.strokeStyle = 'rgba(150,150,150,.5)'; gx.strokeRect(x0, y, ps, ps); gx.strokeStyle = C.blue; gx.lineWidth = 3; gx.strokeRect(x0 + (sel % g) * cell, y + Math.floor(sel / g) * cell, cell, cell); gx.lineWidth = 1; }
      const cap = mode === 'none' ? 'Without position codes, shuffling the patches changes nothing for attention: the model sees a bag of patches.' : 'With position codes, each token carries its (row, column); nearby positions have similar codes (orange), distant ones do not.';
      if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
    }, { fps: 30 });
  };

  /* ================= attnlab ================= */
  AI.labs.attnlab = function (root) {
    const { view } = AI.labShell(root, 'Attention over patches', 'click a patch: score → softmax → weighted sum; change temperature, heads, masks, layers',
      '<canvas style="height:430px;cursor:pointer;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `<label>Scene <select id="scene">${sceneOpts()}</select></label>
       <label>Grid <span class="v" id="gv"></span><input type="range" id="g" min="5" max="12" value="8"></label>
       <label>View <select id="view"><option value="0">head 1: content (similar colour)</option><option value="1">head 2: local neighbourhood</option><option value="2">head 3: same row</option><option value="3">head 4: salient (edge) patches</option><option value="4">mean of the 4 heads</option><option value="5">4 heads side by side</option><option value="6">attention rollout (layers)</option></select></label>
       <label>Temperature τ <span class="v" id="tv"></span><input type="range" id="tau" min="-1.3" max="1" step="0.05" value="0"></label>
       <label>Mask <select id="mask"><option value="g">global (all patches)</option><option value="w">window M×M</option><option value="r">local radius</option></select></label>
       <label>Window / radius <span class="v" id="mv"></span><input type="range" id="m" min="1" max="6" value="2"></label>
       <label>Rollout layers <span class="v" id="lv"></span><input type="range" id="L" min="1" max="8" value="4"></label>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    let g, N, scene, tau, S = [], A = [], ok = null, sal = [], col = null, query = 0, phase = 2, ptime = 0, disp = [], cm = 0, key = '', rollKey = '', Ahat = [], panels = [], dirty = true;
    function build() {
      g = +q('#g').value; N = g * g; scene = q('#scene').value; tau = Math.pow(10, +q('#tau').value);
      q('#gv').textContent = g + '×' + g; q('#tv').textContent = AI.fmt(tau, 2); q('#lv').textContent = q('#L').value;
      const kind = q('#mask').value, m = +q('#m').value; q('#mv').textContent = kind === 'w' ? m + '×' + m : kind === 'r' ? 'radius ' + m : '–';
      col = gridMean(scene, g); sal = new Array(N).fill(0);
      const cc = (i) => [col[i * 3] / 255, col[i * 3 + 1] / 255, col[i * 3 + 2] / 255];
      for (let i = 0; i < N; i++) { const x = i % g, y = Math.floor(i / g); let s = 0, n = 0; for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= g || Y >= g) continue; const a = cc(i), b = cc(Y * g + X); s += Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]); n++; } sal[i] = s / n; }
      const mx = Math.max(1e-6, ...sal); sal = sal.map((v) => v / mx);
      S = [0, 1, 2, 3].map(() => new Float32Array(N * N));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const dx = (i % g) - (j % g), dy = Math.floor(i / g) - Math.floor(j / g), a = cc(i), b = cc(j), d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
        S[0][i * N + j] = -40 * d2; S[1][i * N + j] = -(dx * dx + dy * dy) / 1.1; S[2][i * N + j] = -1.5 * dy * dy - 0.05 * dx * dx; S[3][i * N + j] = 5 * sal[j] - 0.02 * (dx * dx + dy * dy);
      }
      ok = new Uint8Array(N * N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const xi = i % g, yi = Math.floor(i / g), xj = j % g, yj = Math.floor(j / g);
        ok[i * N + j] = kind === 'g' ? 1 : kind === 'w' ? (Math.floor(xi / m) === Math.floor(xj / m) && Math.floor(yi / m) === Math.floor(yj / m) ? 1 : 0) : (Math.abs(xi - xj) <= m && Math.abs(yi - yj) <= m ? 1 : 0);
      }
      A = S.map((s) => softAll(s, tau));
      rollKey = ''; query = Math.min(query, N - 1); dirty = true;
    }
    function softAll(s, t) { const o = new Float32Array(N * N), row = new Array(N); for (let i = 0; i < N; i++) { for (let j = 0; j < N; j++) row[j] = ok[i * N + j] ? s[i * N + j] / t : -Infinity; const w = sm(row); for (let j = 0; j < N; j++) o[i * N + j] = w[j]; } return o; }
    function rollout() {
      const L = +q('#L').value, k = [g, scene, tau, L, q('#mask').value, q('#m').value].join(); if (k === rollKey) return; rollKey = k;
      Ahat = []; for (let l = 0; l < L; l++) { const a = softAll(S[l % 4], tau * (1 + 0.25 * l)), h = new Float32Array(N * N); for (let i = 0; i < N * N; i++) h[i] = 0.5 * a[i]; for (let i = 0; i < N; i++) h[i * N + i] += 0.5; Ahat.push(h); }
    }
    function rowFor(v, qi) {
      const w = new Float32Array(N);
      if (v <= 3) { for (let j = 0; j < N; j++) w[j] = A[v][qi * N + j]; }
      else if (v === 4) { for (let h = 0; h < 4; h++) for (let j = 0; j < N; j++) w[j] += A[h][qi * N + j] / 4; }
      else { rollout(); let vec = new Float32Array(N); vec[qi] = 1; for (let l = Ahat.length - 1; l >= 0; l--) { const nv = new Float32Array(N), M = Ahat[l]; for (let i = 0; i < N; i++) { const vi = vec[i]; if (vi) for (let j = 0; j < N; j++) nv[j] += vi * M[i * N + j]; } vec = nv; } return vec; }
      return w;
    }
    ['#scene', '#g', '#tau', '#mask', '#m', '#L'].forEach((s) => q(s).addEventListener('input', build));
    q('#view').addEventListener('input', () => { dirty = true; ptime = 0; });
    wirePlay(root, st, () => { phase = (phase + 1) % 4; ptime = 0; dirty = true; }, () => { phase = 0; ptime = 0; });
    const DUR = [1.6, 2.4, 2.6, 3];
    const layout = () => {
      const v = +q('#view').value, W = c.w, H = c.h;
      if (v === 5) { const ps = Math.min(W / 2 - 10, (H - 60) / 2); return [0, 1, 2, 3].map((h) => ({ h, x: 6 + (h % 2) * (ps + 8), y: 22 + Math.floor(h / 2) * (ps + 20), ps })); }
      const ps = Math.min(W - 160, H - 36); return [{ h: v, x: 6, y: 22, ps }];
    };
    c.cv.addEventListener('pointerdown', (e) => {
      const r = c.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      for (const p of layout()) if (mx >= p.x && mx < p.x + p.ps && my >= p.y && my < p.y + p.ps) { query = Math.floor((my - p.y) / (p.ps / g)) * g + Math.floor((mx - p.x) / (p.ps / g)); phase = 0; ptime = 0; dirty = true; }
    });
    build();
    AI.loop(root, (dt) => {
      if (st.playing) { ptime += dt * st.speed; if (ptime > DUR[phase]) { ptime = 0; phase = (phase + 1) % 4; } }
      const v = +q('#view').value, pl = layout(), C = AI.col(), gx = c.ctx; gx.clearRect(0, 0, c.w, c.h);
      const target = [], rows = [];
      pl.forEach((p, pi) => {
        const w = rowFor(p.h, query), wmax = Math.max(1e-9, ...w); rows.push(w);
        const raw = []; if (p.h <= 3) { let mn = Infinity, mxs = -Infinity; for (let j = 0; j < N; j++) if (ok[query * N + j]) { const s = S[p.h][query * N + j]; mn = Math.min(mn, s); mxs = Math.max(mxs, s); } for (let j = 0; j < N; j++) raw[j] = ok[query * N + j] ? (S[p.h][query * N + j] - mn) / Math.max(1e-6, mxs - mn) : 0; }
        target[pi] = new Float32Array(N); for (let j = 0; j < N; j++) target[pi][j] = phase <= 1 ? (phase === 0 ? 0 : raw.length ? raw[j] * 0.8 : w[j] / wmax) : w[j] / wmax;
        if (!disp[pi] || disp[pi].length !== N) disp[pi] = new Float32Array(N);
        const f = Math.min(1, dt * 5); for (let j = 0; j < N; j++) disp[pi][j] += (target[pi][j] - disp[pi][j]) * f;
      });
      cm += ((phase >= 2 ? 1 : 0) - cm) * Math.min(1, dt * 4);
      pl.forEach((p, pi) => {
        const cell = p.ps / g, w = rows[pi];
        txt(gx, p.h <= 3 ? ['head 1 · content', 'head 2 · local', 'head 3 · row', 'head 4 · salient'][p.h] : p.h === 4 ? 'mean of 4 heads' : `rollout, ${q('#L').value} layers`, p.x, p.y - 6, C.muted);
        drawScene(gx, scene, p.x, p.y, p.ps, p.ps);
        for (let j = 0; j < N; j++) {
          const a = Math.min(1, Math.pow(disp[pi][j], 0.7)) * 0.82; if (a < 0.01) continue;
          gx.fillStyle = `rgba(${Math.round(AI.lerp(60, 255, cm))},${Math.round(AI.lerp(140, 150, cm))},${Math.round(AI.lerp(255, 30, cm))},${a})`; gx.fillRect(p.x + (j % g) * cell, p.y + Math.floor(j / g) * cell, cell, cell);
          if (!ok[query * N + j] && p.h <= 4) { gx.fillStyle = 'rgba(0,0,0,.45)'; gx.fillRect(p.x + (j % g) * cell, p.y + Math.floor(j / g) * cell, cell, cell); }
        }
        gx.strokeStyle = 'rgba(150,150,150,.45)'; gx.lineWidth = 1; gx.beginPath(); for (let i = 0; i <= g; i++) { gx.moveTo(p.x + i * cell, p.y); gx.lineTo(p.x + i * cell, p.y + p.ps); gx.moveTo(p.x, p.y + i * cell); gx.lineTo(p.x + p.ps, p.y + i * cell); } gx.stroke();
        if (cell >= 36 && phase >= 2) for (let j = 0; j < N; j++) if (w[j] > 0.005) txt(gx, (w[j] * 100).toFixed(0) + '%', p.x + (j % g) * cell + cell / 2, p.y + Math.floor(j / g) * cell + cell / 2 + 4, '#fff', 'bold 11px sans-serif', 'center');
        const qx = p.x + (query % g) * cell, qy = p.y + Math.floor(query / g) * cell; gx.strokeStyle = C.blue; gx.lineWidth = 3; gx.strokeRect(qx, qy, cell, cell);
        if (phase >= 2) {
          const idx = [...Array(N).keys()].sort((a, b) => w[b] - w[a]).slice(0, 6), tt = performance.now() / 1000;
          idx.forEach((j) => { if (j === query) return; const x = p.x + (j % g + 0.5) * cell, y = p.y + (Math.floor(j / g) + 0.5) * cell, X = qx + cell / 2, Y = qy + cell / 2; gx.strokeStyle = `rgba(255,170,40,${0.4 + 0.5 * w[j] / (w[idx[0]] || 1)})`; gx.lineWidth = 1 + 8 * w[j]; gx.beginPath(); gx.moveTo(X, Y); gx.lineTo(x, y); gx.stroke(); if (phase === 3) { const u = (tt * 0.4 * st.speed + j * 0.13) % 1; gx.fillStyle = C.yellow; gx.beginPath(); gx.arc(AI.lerp(x, X, u), AI.lerp(y, Y, u), 3.5, 0, 7); gx.fill(); } });
        }
        // output swatch
        if (pl.length === 1) {
          const sx = p.x + p.ps + 16; let r = 0, gg = 0, b = 0; for (let j = 0; j < N; j++) { r += w[j] * col[j * 3]; gg += w[j] * col[j * 3 + 1]; b += w[j] * col[j * 3 + 2]; }
          txt(gx, 'query colour', sx, p.y + 14, C.muted); gx.fillStyle = rgbs(col, query); gx.fillRect(sx, p.y + 20, 44, 28); gx.strokeStyle = C.border; gx.strokeRect(sx, p.y + 20, 44, 28);
          if (phase === 3) { txt(gx, 'after attention', sx, p.y + 70, C.muted); gx.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`; gx.fillRect(sx, p.y + 76, 44, 28); gx.strokeRect(sx, p.y + 76, 44, 28); txt(gx, 'Σ weight × colour', sx, p.y + 120, C.muted, '11px sans-serif'); }
        }
      });
      const w0 = rows[0], ent = -[...w0].reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0), top = [...Array(N).keys()].sort((a, b) => w0[b] - w0[a]).slice(0, 3);
      const allowed = ok.slice(query * N, query * N + N).reduce((s, x) => s + x, 0);
      const k2 = [query, phase, v, g, tau, q('#mask').value, q('#m').value, scene].join();
      if (k2 !== key) {
        key = k2;
        q('#out').innerHTML = `query: row ${Math.floor(query / g)}, col ${query % g}\nallowed keys: ${allowed} of ${N}\nΣ weights = ${AI.fmt(w0.reduce((s, x) => s + x, 0), 3)}\ntop 3: ${top.map((j) => `(${Math.floor(j / g)},${j % g}) ${(w0[j] * 100).toFixed(1)}%`).join('  ')}\nentropy ${ent.toFixed(2)} bits ≈ ${AI.fmt(Math.pow(2, ent), 1)} effective keys`;
        q('#cap').textContent = ['① The blue patch is the query: it asks "who is relevant to me?"', '② Score every key: s = q·k/√d, then divide by τ (blue = low, strong blue = high).', '③ Softmax turns scores into weights that sum to 1 (orange). Masked patches get exactly 0.', '④ The output is the weighted sum of the value vectors: here shown as a blended colour.'][phase];
      }
    }, { fps: 30 });
  };

  /* ================= attncost ================= */
  AI.labs.attncost = function (root) {
    const { view } = AI.labShell(root, 'Attention cost vs image resolution', 'tokens N = (H/P)²; linear layers cost 12·N·D², attention scores+mix cost 2·N²·D (per block)',
      '<canvas style="height:470px"></canvas><div class="caption" id="cap"></div>',
      `<label>Image side H <span class="v" id="hv"></span><input type="range" id="H" min="64" max="2048" step="16" value="224"></label>
       <label>Patch size P <select id="P"><option>4</option><option>8</option><option selected>16</option><option>32</option></select></label>
       <label>Width D <select id="D"><option>192</option><option>384</option><option selected>768</option><option>1024</option></select></label>
       <label>Windowed attention <select id="w"><option value="0">off (global)</option><option value="7">window 7×7 tokens</option><option value="8">window 8×8</option><option value="14">window 14×14</option></select></label>
       <div class="row"><button class="btn primary" id="sweep">▶ Sweep resolution</button><button class="btn" id="stop">⏸ Pause</button></div>
       <label>Speed <span class="v" id="spv">×1</span><input type="range" id="sp" min="0.1" max="3" step="0.05" value="1"></label>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let sweeping = false, paused = false, speed = 1, lin = 0, att = 0, win = 0, dN = 196, bars = [0, 0, 0], key = '';
    q('#sp').oninput = () => { speed = +q('#sp').value; q('#spv').textContent = '×' + speed; };
    q('#sweep').onclick = () => { sweeping = true; paused = false; q('#H').value = 224; q('#stop').textContent = '⏸ Pause'; };
    q('#stop').onclick = () => { paused = !paused; q('#stop').textContent = paused ? '▶ Resume' : '⏸ Pause'; };
    q('#H').oninput = () => { sweeping = false; };
    const calc = (H, P, D, w) => { const N = Math.pow(H / P, 2); return { N, lin: 12 * N * D * D, att: 2 * N * N * D, win: w ? 2 * N * w * w * D : 0 }; };
    AI.loop(root, (dt) => {
      if (sweeping && !paused) { const h = +q('#H').value + dt * 40 * speed; q('#H').value = Math.min(2048, h); if (h >= 2048) sweeping = false; }
      const H = Math.round(+q('#H').value / 16) * 16, P = +q('#P').value, D = +q('#D').value, w = +q('#w').value, r = calc(H, P, D, w);
      q('#hv').textContent = `${H}×${H}`;
      const f = Math.min(1, dt * 5); dN += (Math.log(r.N) - Math.log(dN)) * f; const Ns = Math.exp(dN);
      bars[0] += (r.lin - bars[0]) * f; bars[1] += (r.att - bars[1]) * f; bars[2] += (r.win - bars[2]) * f;
      const C = AI.col(), gx = c.ctx, W = c.w; gx.clearRect(0, 0, W, c.h);
      const base = calc(224, 16, D, 0), vmax = Math.max(bars[0], bars[1], base.lin, 1);
      txt(gx, `MACs per block (×12 blocks for ViT-B): N = ${fmtInt(r.N)} tokens`, 10, 16, C.muted);
      const items = [['linear (QKV, proj, MLP) 12·N·D²', bars[0], C.blue], ['attention QKᵀ and ·V   2·N²·D', bars[1], C.red]].concat(w ? [[`windowed attention 2·N·${w}²·D`, bars[2], C.green]] : []);
      items.forEach((it, i) => { const y = 28 + i * 44, bw = (W - 30) * Math.min(1, it[1] / vmax); gx.fillStyle = C.bg2; gx.fillRect(10, y + 14, W - 30, 18); gx.fillStyle = it[2]; gx.fillRect(10, y + 14, bw, 18); txt(gx, it[0] + '  =  ' + gfmt(it[1]), 10, y + 10, C.fg); });
      const bx = 10 + (W - 30) * Math.min(1, base.lin / vmax); gx.strokeStyle = C.yellow; gx.setLineDash([4, 3]); gx.beginPath(); gx.moveTo(bx, 38); gx.lineTo(bx, 28 + items.length * 44); gx.stroke(); gx.setLineDash([]); txt(gx, 'linear cost at 224/16', bx - 4, 28 + items.length * 44 + 10, C.yellow, '10px sans-serif', 'right');
      // log-log chart
      const cy0 = 190, ch = c.h - cy0 - 30, cx0 = 54, cw = W - cx0 - 14, lx0 = Math.log10(16), lx1 = Math.log10(30000), ly0 = 4, ly1 = 13.5;
      const X = (n) => cx0 + ((Math.log10(n) - lx0) / (lx1 - lx0)) * cw, Y = (m) => cy0 + ch - ((Math.log10(Math.max(1, m)) - ly0) / (ly1 - ly0)) * ch;
      gx.strokeStyle = C.border; gx.lineWidth = 1; gx.strokeRect(cx0, cy0, cw, ch);
      for (let e = 5; e <= 13; e += 2) { gx.beginPath(); gx.moveTo(cx0, Y(10 ** e)); gx.lineTo(cx0 + cw, Y(10 ** e)); gx.stroke(); txt(gx, '1e' + e, cx0 - 4, Y(10 ** e) + 4, C.muted, '10px sans-serif', 'right'); }
      [100, 1000, 10000].forEach((n) => { txt(gx, String(n), X(n), cy0 + ch + 14, C.muted, '10px sans-serif', 'center'); });
      txt(gx, 'tokens N (log)', cx0 + cw / 2, cy0 + ch + 27, C.muted, '10px sans-serif', 'center'); txt(gx, 'MACs per block (log)', cx0, cy0 - 6, C.muted);
      const curve = (fn, color, dash) => { gx.strokeStyle = color; gx.lineWidth = 2; gx.setLineDash(dash || []); gx.beginPath(); for (let i = 0; i <= 60; i++) { const n = 16 * Math.pow(30000 / 16, i / 60); i ? gx.lineTo(X(n), Y(fn(n))) : gx.moveTo(X(n), Y(fn(n))); } gx.stroke(); gx.setLineDash([]); };
      curve((n) => 12 * n * D * D, C.blue); curve((n) => 2 * n * n * D, C.red); if (w) curve((n) => 2 * n * w * w * D, C.green, [5, 4]);
      gx.strokeStyle = C.yellow; gx.setLineDash([2, 3]); gx.beginPath(); gx.moveTo(X(6 * D), cy0); gx.lineTo(X(6 * D), cy0 + ch); gx.stroke(); gx.setLineDash([]); txt(gx, `crossover N = 6D = ${6 * D}`, X(6 * D) + 4, cy0 + 12, C.yellow, '10px sans-serif');
      const dot = (m, color) => { gx.fillStyle = color; gx.beginPath(); gx.arc(X(Ns), Y(m), 5, 0, 7); gx.fill(); };
      dot(12 * Ns * D * D, C.blue); dot(2 * Ns * Ns * D, C.red); if (w) dot(2 * Ns * w * w * D, C.green);
      const k = [H, P, D, w].join();
      if (k !== key) {
        key = k; const blocks = 12, heads = D / 64, tot = (r.lin + (w ? r.win : r.att)) * blocks, tb = tot / 1e9;
        q('#out').innerHTML = `N = (${H}/${P})² = <b class="o">${fmtInt(r.N)}</b> tokens\nattention matrix N² = ${fmtInt(r.N * r.N)} per head\n× ${heads} heads, fp16: ${AI.fmt((r.N * r.N * heads * 2) / 1e6, 1)} MB per layer\nper block: linear ${gfmt(r.lin)}, attention ${gfmt(w ? r.win : r.att)}\nattention share: <b>${AI.fmt(100 * (w ? r.win : r.att) / (r.lin + (w ? r.win : r.att)), 1)}%</b>\n12 blocks ≈ <b>${AI.fmt(tb, 1)} GMACs</b>\nvs 224/16: ×${AI.fmt(tot / ((base.lin + base.att) * 12), 1)}`;
        q('#cap').textContent = r.N < 6 * D ? 'Below N = 6D the linear layers dominate: attention is a small slice.' : 'Above N = 6D the N² attention term dominates — this is why high-resolution ViTs use windows or other tricks.';
      }
    }, { fps: 30 });
  };

  /* ================= swinwin ================= */
  AI.labs.swinwin = function (root) {
    const { view } = AI.labShell(root, 'Swin: windows, shifted windows, reach', 'attention only inside M×M windows; the next block shifts the grid by M/2 so information crosses the borders',
      '<canvas style="height:420px;cursor:pointer"></canvas><div class="caption" id="cap"></div>',
      `<label>Feature map <select id="G"><option value="8" selected>8×8 tokens</option><option value="12">12×12</option><option value="16">16×16</option></select></label>
       <label>Window M <select id="M"></select></label>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    let G = 8, M = 4, ph = 0, t = 0, probe = 27, r1 = [], r2 = [];
    const DURS = [4, 3.2, 4, 5], PAL = ['#2f81f7', '#3fb950', '#d29922', '#a371f7', '#f85149', '#39c5cf'];
    const win = (x, y, s) => Math.floor((y + s) / M) * 100 + Math.floor((x + s) / M);
    function setupM() { const sel = q('#M'), opts = [2, 3, 4, 6].filter((m) => G % m === 0 && G / m >= 2); sel.innerHTML = opts.map((m) => `<option ${m === M ? 'selected' : ''}>${m}</option>`).join(''); if (!opts.includes(M)) M = opts.includes(4) ? 4 : opts[0]; sel.value = M; }
    function reach() {
      const s = Math.floor(M / 2); r1 = []; r2 = new Set();
      const px = probe % G, py = Math.floor(probe / G), w1 = win(px, py, 0);
      for (let i = 0; i < G * G; i++) if (win(i % G, Math.floor(i / G), 0) === w1) r1.push(i);
      const ws = new Set(r1.map((i) => win(i % G, Math.floor(i / G), s)));
      for (let i = 0; i < G * G; i++) if (ws.has(win(i % G, Math.floor(i / G), s))) r2.add(i);
      r2 = [...r2];
      q('#out').innerHTML = `tokens N = ${G * G}, window M = ${M} (${M * M} tokens)\nwindows: ${(G / M) ** 2}\nattention pairs/layer:\n  global  N² = <b>${fmtInt(G ** 4)}</b>\n  window  N·M² = <b class="g">${fmtInt(G * G * M * M)}</b>  (÷${AI.fmt(G * G / (M * M), 1)})\nreach of marked token:\n  layer 1: <b class="b">${r1.length}</b> tokens\n  layer 2 (shifted): <b class="o">${r2.length}</b> tokens\n  global: ${G * G}`;
    }
    q('#G').addEventListener('input', () => { G = +q('#G').value; setupM(); probe = Math.min(probe, G * G - 1); probe = (G / 2 - 1) * G + G / 2 - 1; reach(); });
    q('#M').addEventListener('input', () => { M = +q('#M').value; reach(); });
    wirePlay(root, st, () => { ph = (ph + 1) % 4; t = 0; }, () => { ph = 0; t = 0; });
    const geom = () => { const cs = Math.min((c.w - 190) / G, (c.h - 40) / G); return { cs, x0: 10, y0: 26 }; };
    c.cv.addEventListener('pointerdown', (e) => { const r = c.cv.getBoundingClientRect(), { cs, x0, y0 } = geom(), x = Math.floor((e.clientX - r.left - x0) / cs), y = Math.floor((e.clientY - r.top - y0) / cs); if (x >= 0 && y >= 0 && x < G && y < G) { probe = y * G + x; reach(); } });
    setupM(); probe = (G / 2 - 1) * G + G / 2 - 1; reach();
    const CAPS = ['Block ℓ: tokens attend only inside their own M×M window (all windows run in parallel).', 'The window grid moves by M/2 in both directions (really: the feature map is rolled cyclically).', 'Block ℓ+1: attention inside the shifted windows. Old borders now lie inside windows, so information crosses them.', 'Reach of the marked token: blue after block ℓ, orange after block ℓ+1. Repeating the pair spreads information over the whole map.'];
    AI.loop(root, (dt) => {
      if (st.playing) { t += dt * st.speed; if (t > DURS[ph]) { t = 0; ph = (ph + 1) % 4; } }
      const C = AI.col(), gx = c.ctx, { cs, x0, y0 } = geom(); gx.clearRect(0, 0, c.w, c.h);
      const s = Math.floor(M / 2), u = ph === 1 ? AI.smooth(t / DURS[1]) : ph >= 2 ? 1 : 0, shifted = ph >= 2 || (ph === 1 && u > 0.5);
      txt(gx, `${G}×${G} feature map, window ${M}×${M}${shifted ? ', shifted by ' + s : ''}`, x0, 16, C.muted);
      for (let i = 0; i < G * G; i++) { const x = i % G, y = Math.floor(i / G), w = win(x, y, shifted ? s : 0), k = (Math.floor(w / 100) * 3 + (w % 100)) % 6; gx.fillStyle = PAL[k] + '40'; gx.fillRect(x0 + x * cs, y0 + y * cs, cs, cs); }
      // reach
      if (ph === 0 || ph === 1) { gx.fillStyle = 'rgba(47,129,247,.30)'; r1.forEach((i) => gx.fillRect(x0 + (i % G) * cs, y0 + Math.floor(i / G) * cs, cs, cs)); }
      if (ph >= 2) { if (ph === 3 || t > 1) { gx.fillStyle = 'rgba(240,136,62,.28)'; r2.forEach((i) => gx.fillRect(x0 + (i % G) * cs, y0 + Math.floor(i / G) * cs, cs, cs)); } gx.fillStyle = 'rgba(47,129,247,.30)'; r1.forEach((i) => gx.fillRect(x0 + (i % G) * cs, y0 + Math.floor(i / G) * cs, cs, cs)); }
      gx.strokeStyle = 'rgba(150,150,150,.4)'; gx.lineWidth = 1; gx.beginPath(); for (let i = 0; i <= G; i++) { gx.moveTo(x0 + i * cs, y0); gx.lineTo(x0 + i * cs, y0 + G * cs); gx.moveTo(x0, y0 + i * cs); gx.lineTo(x0 + G * cs, y0 + i * cs); } gx.stroke();
      // window borders
      const bl = (off, color, dash, alpha) => { gx.save(); gx.beginPath(); gx.rect(x0, y0, G * cs, G * cs); gx.clip(); gx.strokeStyle = color; gx.globalAlpha = alpha; gx.lineWidth = 3; gx.setLineDash(dash); gx.beginPath(); for (let k = 0; k <= G / M + 1; k++) { const p = (k * M - off) * cs; gx.moveTo(x0 + p, y0); gx.lineTo(x0 + p, y0 + G * cs); gx.moveTo(x0, y0 + p); gx.lineTo(x0 + G * cs, y0 + p); } gx.stroke(); gx.restore(); };
      bl(0, C.blue, [], ph === 1 ? 1 - u : ph >= 2 ? 0.0 : 1); if (ph >= 1) bl(s * u, C.orange, [7, 5], 1);
      // star lines
      if (ph === 0 || ph === 2) {
        const sh = ph === 2 ? s : 0, pulse = 0.5 + 0.5 * Math.sin(t * 3 * st.speed);
        const groups = {}; for (let i = 0; i < G * G; i++) { const w = win(i % G, Math.floor(i / G), sh); (groups[w] = groups[w] || []).push(i); }
        gx.strokeStyle = (ph === 0 ? C.blue : C.orange); gx.globalAlpha = 0.25 + 0.35 * pulse; gx.lineWidth = 1; gx.beginPath();
        Object.values(groups).forEach((m) => { let mx = 0, my = 0; m.forEach((i) => { mx += i % G; my += Math.floor(i / G); }); mx = x0 + (mx / m.length + 0.5) * cs; my = y0 + (my / m.length + 0.5) * cs; m.forEach((i) => { gx.moveTo(mx, my); gx.lineTo(x0 + ((i % G) + 0.5) * cs, y0 + (Math.floor(i / G) + 0.5) * cs); }); });
        gx.stroke(); gx.globalAlpha = 1;
      }
      const px = probe % G, py = Math.floor(probe / G); gx.fillStyle = C.yellow; gx.beginPath(); gx.arc(x0 + (px + 0.5) * cs, y0 + (py + 0.5) * cs, Math.max(4, cs * 0.22), 0, 7); gx.fill(); gx.strokeStyle = '#000'; gx.lineWidth = 1.5; gx.stroke();
      const lx = x0 + G * cs + 18; txt(gx, 'phase', lx, 40, C.muted);
      ['① regular windows', '② shift the grid', '③ shifted windows', '④ reach grows'].forEach((l, i) => txt(gx, l, lx, 60 + i * 20, i === ph ? C.fg : C.muted, i === ph ? 'bold 12px sans-serif' : '12px sans-serif'));
      txt(gx, 'click a token to move the marker', lx, 160, C.muted, '11px sans-serif');
      if (q('#cap').textContent !== CAPS[ph]) q('#cap').textContent = CAPS[ph];
    }, { fps: 30 });
  };

  /* ================= swinstages ================= */
  AI.labs.swinstages = function (root) {
    const CFG = { 'Swin-T': [96, [2, 2, 6, 2], [3, 6, 12, 24]], 'Swin-S': [96, [2, 2, 18, 2], [3, 6, 12, 24]], 'Swin-B': [128, [2, 2, 18, 2], [4, 8, 16, 32]] };
    const { view } = AI.labShell(root, 'Swin pyramid and patch merging', 'every stage merges 2×2 tokens: height and width halve, channels double',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div><div class="readout" id="tab" style="margin-top:8px"></div>',
      `<label>Model <select id="mod">${Object.keys(CFG).map((k) => `<option>${k}</option>`).join('')}</select></label>
       <label>Input size <select id="res"><option>224</option><option>448</option><option>896</option></select></label>
       <label>Window M <select id="M"><option>7</option><option>14</option></select></label>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    let rows = [], tot = {}, t = 0;
    function calc() {
      const [C0, dep, hd] = CFG[q('#mod').value], res = +q('#res').value, M = +q('#M').value, P = 4; let g = res / P, params = P * P * 3 * C0 + C0 + 2 * C0, macs = g * g * P * P * 3 * C0, cc = C0, gl = 0, wl = 0; rows = [];
      for (let s = 0; s < 4; s++) {
        let mp = 0, mm = 0; if (s > 0) { mp = 2 * 4 * cc + 4 * cc * 2 * cc; mm = (g / 2) * (g / 2) * 4 * cc * 2 * cc; params += mp; macs += mm; cc *= 2; g /= 2; }
        const n = g * g, m = Math.min(M, g), bp = 4 * cc * cc + 4 * cc + (2 * m - 1) ** 2 * hd[s] + 8 * cc * cc + 5 * cc + 4 * cc, bm = 4 * n * cc * cc + 2 * n * m * m * cc + 8 * n * cc * cc;
        params += dep[s] * bp; macs += dep[s] * bm; gl += dep[s] * 2 * n * n * cc; wl += dep[s] * 2 * n * m * m * cc;
        rows.push({ s: s + 1, g, c: cc, n, w: Math.ceil(g / m) ** 2, d: dep[s], h: hd[s], bp, mp, bm });
      }
      params += 2 * cc + cc * 1000 + 1000; macs += cc * 1000; tot = { params, macs, gl, wl };
      q('#tab').innerHTML = `<span class="m">stage  tokens      width  blocks heads windows  params/block</span>\n` + rows.map((r) => `${r.s}      ${(r.g + '×' + r.g).padEnd(8)} ${String(r.c).padEnd(6)} ${String(r.d).padEnd(6)} ${String(r.h).padEnd(5)} ${String(r.w).padEnd(8)} ${fmtInt(r.bp)}`).join('\n');
      q('#out').innerHTML = `total params <b class="o">${AI.fmt(params / 1e6, 2)} M</b>\ntotal MACs <b class="o">${AI.fmt(macs / 1e9, 2)} G</b>\nattention-product MACs:\n  windowed ${gfmt(wl)}\n  if global ${gfmt(gl)} (×${AI.fmt(gl / wl, 1)})\npatch merging 1→2: ${fmtInt(rows[1].mp)} params`;
    }
    ['#mod', '#res', '#M'].forEach((s) => q(s).addEventListener('input', calc));
    wirePlay(root, st, () => { t = (Math.floor(t / 3) + 1) * 3; }, () => { t = 0; });
    calc();
    AI.loop(root, (dt) => {
      if (st.playing) t += dt * st.speed; const T = t % 12;
      const C = AI.col(), gx = c.ctx, W = c.w, H = c.h; gx.clearRect(0, 0, W, H);
      // left: merging animation
      const lw = Math.min(W * 0.45, 260), ph = Math.floor(T / 3), u = AI.smooth((T % 3) / 3), C0 = 6;
      txt(gx, 'patch merging of one 2×2 group', 10, 16, C.muted);
      const barH = (cc) => 12 + cc * 3, cols = [C.blue, C.green, C.orange, C.purple], bx = 22, by = 150;
      for (let k = 0; k < 4; k++) {
        const startX = 22 + (k % 2) * 44, startY = 40 + Math.floor(k / 2) * 40, stackX = 40 + k * 26;
        const X = ph === 0 ? startX : ph === 1 ? AI.lerp(startX, stackX, u) : stackX, Y = ph === 0 ? startY : ph === 1 ? AI.lerp(startY, by + 20, u) : by + 20;
        const sq = ph <= 1 ? 30 : 22, hh = ph === 2 ? AI.lerp(barH(C0), barH(C0) * 0.5, u) : ph === 3 ? barH(C0) * 0.5 : ph === 0 ? 30 : AI.lerp(30, barH(C0), u);
        if (ph <= 0) { gx.fillStyle = cols[k] + 'aa'; gx.fillRect(X, Y, 30, 30); txt(gx, 'C', X + 12, Y + 19, '#fff'); }
        else { gx.fillStyle = cols[k] + 'cc'; gx.fillRect(X, Y, 22, hh); }
      }
      const caps = ['Four neighbouring tokens, each a vector of C numbers.', 'Concatenate them: one vector of 4C numbers.', 'LayerNorm + linear layer 4C → 2C: the stack is squeezed to half its length.', 'One new token of width 2C replaces four: the map has half the height and width.'];
      gx.fillStyle = C.bg2; if (ph === 3) { gx.fillStyle = C.teal + 'cc'; gx.fillRect(150, 70, 44, 44); txt(gx, '2C', 162, 96, '#fff'); }
      txt(gx, ['4 × C', '4C', '4C → 2C', '1 × 2C'][ph], 10, H - 40, C.fg, 'bold 13px sans-serif');
      // right: pyramid boxes
      const x0 = lw + 20, aw = W - x0 - 10, maxS = H - 60;
      txt(gx, `feature pyramid (${q('#res').value}×${q('#res').value} input)`, x0, 16, C.muted);
      let cx = x0 + 4; const wsum = rows.reduce((a, r) => a + 14 + Math.log2(r.c / 48) * 8 + 10, 0), sc = Math.min(1, aw / (wsum + 40)), pulse = (T / 12) * 4;
      rows.forEach((r, i) => {
        const hgt = Math.max(24, (maxS * Math.log2(r.g * 2)) / Math.log2(240) * 0.9), th = (14 + Math.log2(r.c / 48) * 8) * sc, yy = 30 + (maxS - hgt) / 2;
        const glow = Math.max(0, 1 - Math.abs(pulse - i - 0.5)); gx.fillStyle = AI.mix(C.bg2, C.blue, 0.25 + 0.6 * glow); gx.fillRect(cx, yy, th, hgt); gx.strokeStyle = C.blue; gx.lineWidth = 1.5; gx.strokeRect(cx, yy, th, hgt);
        const nw = Math.sqrt(r.w); gx.strokeStyle = 'rgba(255,255,255,.35)'; gx.lineWidth = 1; gx.beginPath(); for (let k = 1; k < nw && nw <= 16; k++) { gx.moveTo(cx, yy + (hgt * k) / nw); gx.lineTo(cx + th, yy + (hgt * k) / nw); } gx.stroke();
        txt(gx, `S${r.s}`, cx + th / 2, yy - 6, C.fg, 'bold 11px sans-serif', 'center'); txt(gx, `${r.g}²`, cx + th / 2, yy + hgt + 13, C.muted, '10px sans-serif', 'center'); txt(gx, `C=${r.c}`, cx + th / 2, yy + hgt + 25, C.muted, '10px sans-serif', 'center');
        cx += th + 16 * sc + 10;
      });
      const cap = caps[ph]; if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
    }, { fps: 30 });
  };

  /* ================= mae ================= */
  AI.labs.mae = function (root) {
    const { view } = AI.labShell(root, 'Masked autoencoder', 'hide most patches, encode only the visible ones, paint the rest back',
      '<canvas style="height:440px"></canvas><div class="caption" id="cap"></div>',
      `<label>Scene <select id="scene">${sceneOpts()}</select></label>
       <label>Patches per side <select id="g"><option>8</option><option>10</option><option selected>14</option></select></label>
       <label>Mask ratio <span class="v" id="rv"></span><input type="range" id="ratio" min="10" max="95" step="5" value="75"></label>
       <div class="row"><button class="btn" id="reroll">🎲 Re-roll mask</button></div>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    let g, N, scene, gm, masked = [], order = [], nv = 0, rank = [], recon = [], mse = 0, ph = 0, t = 0, seed = 3;
    const DUR = [2, 3.5, 3, 3, 3, 3.5];
    function setup() {
      g = +q('#g').value; N = g * g; scene = q('#scene').value; gm = gridMean(scene, g);
      const r = AI.rng(seed * 97 + 1); order = [...Array(N).keys()]; for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      ratio();
    }
    function ratio() {
      const rt = +q('#ratio').value / 100; q('#rv').textContent = Math.round(rt * 100) + '%'; const nm = Math.round(rt * N); nv = N - nm;
      masked = new Array(N).fill(false); order.slice(0, nm).forEach((i) => (masked[i] = true));
      rank = []; let k = 0; for (let i = 0; i < N; i++) if (!masked[i]) rank[i] = k++;
      const vis = [...Array(N).keys()].filter((i) => !masked[i]); recon = new Array(N); mse = 0;
      for (let i = 0; i < N; i++) {
        if (!masked[i]) { recon[i] = [gm[i * 3], gm[i * 3 + 1], gm[i * 3 + 2]]; continue; }
        const x = i % g, y = Math.floor(i / g), ds = vis.map((j) => [j, (x - (j % g)) ** 2 + (y - Math.floor(j / g)) ** 2]).sort((a, b) => a[1] - b[1]).slice(0, 4);
        let wsum = 0, r = 0, gg = 0, b = 0; ds.forEach(([j, d]) => { const w = 1 / (d + 0.5); wsum += w; r += w * gm[j * 3]; gg += w * gm[j * 3 + 1]; b += w * gm[j * 3 + 2]; });
        recon[i] = ds.length ? [r / wsum, gg / wsum, b / wsum] : [128, 128, 128];
        mse += ([0, 1, 2].reduce((s, ch) => s + (recon[i][ch] / 255 - gm[i * 3 + ch] / 255) ** 2, 0) / 3);
      }
      mse = nm ? mse / nm : 0;
      q('#out').innerHTML = `patches N = ${N}, masked ${nm}, visible <b class="g">${nv}</b>\nencoder sees ${nv} tokens (${AI.fmt((100 * nv) / N, 0)}%)\nlinear cost ×${AI.fmt(nv / N, 2)}\nattention pairs ${fmtInt(nv * nv)} vs ${fmtInt(N * N)}\n  → ×${AI.fmt((nv * nv) / (N * N), 3)}\nMSE on masked patches: <b class="o">${AI.fmt(mse, 4)}</b>\n<span class="m">stand-in decoder: neighbour average</span>`;
    }
    ['#scene', '#g'].forEach((s) => q(s).addEventListener('input', setup)); q('#ratio').addEventListener('input', ratio);
    q('#reroll').onclick = () => { seed++; setup(); t = 0; ph = 0; };
    wirePlay(root, st, () => { ph = (ph + 1) % 6; t = 0; }, () => { ph = 0; t = 0; });
    setup();
    const CAPS = ['① The image is cut into patches (the tokens).', '② Randomly hide most patches (75% in the real MAE). Hidden patches are removed, not painted grey.', '③ Only the visible patches go into the encoder: a short sequence, so cheap.', '④ Mask tokens (grey, one shared learned vector) are inserted at the hidden positions; position embeddings tell the decoder where each belongs.', '⑤ A light decoder predicts the pixels of every masked patch.', '⑥ Compare to the original: the loss is the mean squared error on masked patches only.'];
    AI.loop(root, (dt) => {
      if (st.playing) { t += dt * st.speed; if (t > DUR[ph]) { t = 0; ph = (ph + 1) % 6; } }
      const C = AI.col(), gx = c.ctx, W = c.w, H = c.h; gx.clearRect(0, 0, W, H);
      const ps = Math.min(W * 0.46, H - 36), x1 = 8, y1 = 26, cell = ps / g, x2 = W - ps - 8, u = AI.smooth(t / DUR[ph]);
      txt(gx, ph >= 4 ? 'image + reconstruction' : 'image', x1, 16, C.muted); txt(gx, ph <= 1 ? '' : ph === 2 ? `encoder input: ${nv} tokens` : ph === 3 ? `decoder input: ${N} tokens` : 'decoder output', x2, 16, C.muted);
      const appear = ph === 1 ? Math.floor(u * N) : ph > 1 ? N : 0;
      for (let i = 0; i < N; i++) {
        const X = x1 + (i % g) * cell, Y = y1 + Math.floor(i / g) * cell; let hidden = masked[i] && order.indexOf(i) < appear;
        if (ph >= 4 && masked[i]) { const f = ph === 4 ? u : 1; gx.fillStyle = `rgb(${Math.round(AI.lerp(128, recon[i][0], f))},${Math.round(AI.lerp(128, recon[i][1], f))},${Math.round(AI.lerp(128, recon[i][2], f))})`; }
        else gx.fillStyle = hidden ? C.bg3 : rgbs(gm, i);
        gx.fillRect(X, Y, cell + 0.5, cell + 0.5);
        if (ph >= 2 && !masked[i]) { gx.strokeStyle = C.green; gx.lineWidth = 1; gx.strokeRect(X + 0.5, Y + 0.5, cell - 1, cell - 1); }
      }
      if (cell >= 3) { gx.strokeStyle = 'rgba(150,150,150,.5)'; gx.lineWidth = 1; gx.beginPath(); for (let i = 0; i <= g; i++) { gx.moveTo(x1 + i * cell, y1); gx.lineTo(x1 + i * cell, y1 + ps); gx.moveTo(x1, y1 + i * cell); gx.lineTo(x1 + ps, y1 + i * cell); } gx.stroke(); }
      if (ph >= 2) {
        const compact = (i) => { const per = Math.max(1, Math.round(Math.sqrt(nv))), r = rank[i]; return [x2 + (r % per) * cell, y1 + Math.floor(r / per) * cell]; }, raster = (i) => [x2 + (i % g) * cell, y1 + Math.floor(i / g) * cell];
        for (let i = 0; i < N; i++) {
          let X, Y, fill;
          if (!masked[i]) {
            const a = [x1 + (i % g) * cell, y1 + Math.floor(i / g) * cell], b = compact(i), r = raster(i);
            if (ph === 2) { X = AI.lerp(a[0], b[0], u); Y = AI.lerp(a[1], b[1], u); } else if (ph === 3) { X = AI.lerp(b[0], r[0], u); Y = AI.lerp(b[1], r[1], u); } else { X = r[0]; Y = r[1]; }
            fill = rgbs(gm, i);
          } else {
            if (ph === 2) continue; const r = raster(i); X = r[0]; Y = r[1];
            const f = ph === 3 ? u : 1; fill = ph >= 4 ? `rgb(${Math.round(AI.lerp(128, recon[i][0], ph === 4 ? u : 1))},${Math.round(AI.lerp(128, recon[i][1], ph === 4 ? u : 1))},${Math.round(AI.lerp(128, recon[i][2], ph === 4 ? u : 1))})` : C.bg3; gx.globalAlpha = f;
          }
          gx.fillStyle = fill; gx.fillRect(X, Y, cell - 1, cell - 1); gx.globalAlpha = 1;
          if (masked[i] && ph === 3) { gx.strokeStyle = C.muted; gx.setLineDash([2, 2]); gx.strokeRect(X + 0.5, Y + 0.5, cell - 2, cell - 2); gx.setLineDash([]); }
        }
      }
      if (ph === 5) txt(gx, `MSE (masked) = ${AI.fmt(mse, 4)}`, x2, y1 + ps + 14, C.orange, 'bold 12px sans-serif');
      if (q('#cap').textContent !== CAPS[ph]) q('#cap').textContent = CAPS[ph];
    }, { fps: 30 });
  };

  /* ================= clip ================= */
  AI.labs.clip = function (root) {
    const simclr = root.dataset.mode === 'simclr';
    const A_NAMES = simclr ? ['view a1', 'view a2', 'view a3'] : ['🐱 image 1', '🐶 image 2', '🚗 image 3'], B_NAMES = simclr ? ['view b1', 'view b2', 'view b3'] : ['"a cat"', '"a dog"', '"a car"'];
    const ZS = [['a photo of a cat', [0.9, 0.3, 0.2]], ['a photo of a dog', [0.3, 0.9, 0.2]], ['a photo of a car', [-0.2, -0.3, 0.9]], ['a photo of a plane', [-0.7, 0.2, 0.7]]];
    const IM = [['cat photo', [0.8, 0.45, 0.25]], ['dog photo', [0.4, 0.8, 0.25]], ['blurry cat/dog', [0.65, 0.65, 0.2]], ['car photo', [-0.15, -0.35, 0.9]], ['sketch, unfamiliar', [-0.6, -0.6, -0.4]]];
    const { view } = AI.labShell(root, simclr ? 'Contrastive pairs on the sphere (SimCLR-style)' : 'CLIP: images and text on one sphere', simclr ? 'two views of each image are pulled together, other images pushed away' : 'train the dual encoders contrastively, then classify zero-shot',
      '<canvas style="height:420px;cursor:grab;touch-action:none"></canvas><div class="caption" id="cap"></div>',
      `${simclr ? '' : '<label>Mode <select id="mode"><option value="train">contrastive training</option><option value="zs">zero-shot classification</option></select></label>'}
       <label>Temperature τ <span class="v" id="tv"></span><input type="range" id="tau" min="-1.3" max="0.3" step="0.02" value="-0.7"></label>
       <div id="match"><div style="color:var(--muted);margin-bottom:4px">${simclr ? 'which view matches' : 'which text matches'} (editable)</div>${[0, 1, 2].map((i) => `<div class="row" style="margin-bottom:3px"><span style="width:90px">${A_NAMES[i]} ↔</span><select id="m${i}" style="width:auto">${[0, 1, 2].map((j) => `<option value="${j}" ${i === j ? 'selected' : ''}>${B_NAMES[j]}</option>`).join('')}</select></div>`).join('')}</div>
       <div class="row"><button class="btn" id="scatter">🎲 Re-scatter</button><button class="btn" id="nextimg">Next test image</button></div>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {}, tick = AI.ticker();
    const nrm = (v) => { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); }, dotp = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let U = [], V = [], tau = 0.2, perm = [0, 1, 2], seed = 5, hist = [], ang = 0.6, drag = null, zsImg = 0, zsFrom = null, zsTo = null, zsT = 1, zsPos = nrm(IM[0][1]), zsTimer = 0, probs = [], mode = 'train';
    const scatter = () => { const r = AI.rng(seed++ * 13 + 1); const rv = () => nrm([r() - 0.5, r() - 0.5, r() - 0.5]); U = [rv(), rv(), rv()]; V = [rv(), rv(), rv()]; hist = []; };
    scatter();
    const mats = () => { const S = U.map((u) => V.map((v) => dotp(u, v))), L = S.map((r) => r.map((x) => x / tau)), Pr = L.map(sm), Pc = [0, 1, 2].map((j) => sm(L.map((r) => r[j]))); return { S, L, Pr, Pc }; };
    // target T[i][j] = 1 if text j matches image i
    const lossOf = () => { const { Pr, Pc } = mats(); let lr = 0, lc = 0; for (let i = 0; i < 3; i++) { lr -= Math.log(Pr[i][perm[i]] + 1e-12); lc -= Math.log(Pc[perm[i]][i] + 1e-12); } return [lr / 3, lc / 3, (lr + lc) / 6]; };
    function trainStep() {
      const { Pr, Pc } = mats(), lr = 0.08, gU = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], gV = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        const T = perm[i] === j ? 1 : 0, dS = (Pr[i][j] - T + (Pc[j][i] - T)) / 6;
        for (let k = 0; k < 3; k++) { gU[i][k] += (dS * V[j][k]) / tau; gV[j][k] += (dS * U[i][k]) / tau; }
      }
      const upd = (P, G) => P.map((p, i) => { const d = dotp(G[i], p), t = G[i].map((x, k) => x - d * p[k]); return nrm(p.map((x, k) => x - lr * t[k])); });
      U = upd(U, gU); V = upd(V, gV); hist.push(lossOf()[2]); if (hist.length > 160) hist.shift();
    }
    const setTau = () => { tau = Math.pow(10, +q('#tau').value); q('#tv').textContent = AI.fmt(tau, 3); hist = []; };
    q('#tau').addEventListener('input', setTau); setTau();
    [0, 1, 2].forEach((i) => q('#m' + i).addEventListener('input', () => { const v = +q('#m' + i).value, j = perm.indexOf(v); if (j >= 0 && j !== i) { perm[j] = perm[i]; q('#m' + j).value = perm[j]; } perm[i] = v; hist = []; }));
    q('#scatter').onclick = () => scatter();
    const nextImg = () => { zsImg = (zsImg + 1) % IM.length; zsFrom = zsPos; zsTo = nrm(IM[zsImg][1]); zsT = 0; };
    q('#nextimg').onclick = nextImg;
    if (!simclr) q('#mode').addEventListener('input', () => { mode = q('#mode').value; q('#match').style.display = mode === 'zs' ? 'none' : ''; q('#scatter').style.display = mode === 'zs' ? 'none' : ''; q('#nextimg').style.display = mode === 'zs' ? '' : 'none'; hist = []; });
    else q('#nextimg').style.display = 'none';
    q('#nextimg').style.display = 'none';
    wirePlay(root, st, () => { if (mode === 'zs') nextImg(); else trainStep(); }, () => { if (mode === 'zs') { zsImg = 0; zsPos = nrm(IM[0][1]); zsT = 1; } else scatter(); });
    c.cv.addEventListener('pointerdown', (e) => { drag = e.clientX; c.cv.setPointerCapture(e.pointerId); });
    c.cv.addEventListener('pointermove', (e) => { if (drag !== null) { ang += (e.clientX - drag) * 0.01; drag = e.clientX; } });
    c.cv.addEventListener('pointerup', () => (drag = null));
    const proj = (p, cx, cy, R) => { const ca = Math.cos(ang), sa = Math.sin(ang), x = p[0] * ca + p[2] * sa, z = -p[0] * sa + p[2] * ca, tl = 0.45, y = p[1] * Math.cos(tl) - z * Math.sin(tl), z2 = p[1] * Math.sin(tl) + z * Math.cos(tl); return [cx + R * x, cy - R * y, z2]; };
    AI.loop(root, (dt) => {
      const C = AI.col(), gx = c.ctx, W = c.w, H = c.h; gx.clearRect(0, 0, W, H);
      if (st.playing && drag === null) ang += dt * 0.25 * st.speed;
      const zs = mode === 'zs';
      if (!zs && st.playing) { const n = tick(dt, 6 * st.speed); for (let i = 0; i < n; i++) trainStep(); }
      if (zs) {
        if (zsT < 1) { zsT = Math.min(1, zsT + (dt * st.speed) / 3.5); const e = AI.smooth(zsT), om = Math.acos(AI.clamp(dotp(zsFrom, zsTo), -1, 1)) || 1e-4; const a = Math.sin((1 - e) * om) / Math.sin(om), b = Math.sin(e * om) / Math.sin(om); zsPos = nrm(zsFrom.map((x, k) => a * x + b * zsTo[k])); }
        else if (st.playing) { zsTimer += dt * st.speed; if (zsTimer > 3.5) { zsTimer = 0; nextImg(); } }
        const lg = ZS.map((z) => dotp(zsPos, nrm(z[1])) / tau); probs = sm(lg);
      }
      const R = Math.min(H * 0.38, W * 0.2), cx = R + 18, cy = H / 2 - 6;
      gx.strokeStyle = C.border; gx.lineWidth = 1.5; gx.beginPath(); gx.arc(cx, cy, R, 0, 7); gx.stroke();
      gx.strokeStyle = C.border; gx.globalAlpha = 0.5; for (let la = -60; la <= 60; la += 30) { const rr = Math.cos((la * Math.PI) / 180); gx.beginPath(); for (let a = 0; a <= 64; a++) { const th = (a / 64) * 2 * Math.PI, p = proj([rr * Math.cos(th), Math.sin((la * Math.PI) / 180), rr * Math.sin(th)], cx, cy, R); a ? gx.lineTo(p[0], p[1]) : gx.moveTo(p[0], p[1]); } gx.stroke(); } gx.globalAlpha = 1;
      const pt = (p, col, shape, label, big) => { const s = proj(p, cx, cy, R), a = s[2] > 0 ? 1 : 0.35; gx.globalAlpha = a; gx.fillStyle = col; const r = big || 7; if (shape === 'c') { gx.beginPath(); gx.arc(s[0], s[1], r, 0, 7); gx.fill(); } else gx.fillRect(s[0] - r, s[1] - r, r * 2, r * 2); gx.fillStyle = C.fg; gx.font = '11px sans-serif'; gx.textAlign = 'left'; gx.fillText(label, s[0] + 10, s[1] + 4); gx.globalAlpha = 1; return s; };
      const colsA = [C.blue, C.green, C.orange];
      if (!zs) {
        for (let i = 0; i < 3; i++) { const a = proj(U[i], cx, cy, R), b = proj(V[perm[i]], cx, cy, R); gx.strokeStyle = colsA[i]; gx.globalAlpha = 0.6; gx.setLineDash([4, 3]); gx.beginPath(); gx.moveTo(a[0], a[1]); gx.lineTo(b[0], b[1]); gx.stroke(); gx.setLineDash([]); gx.globalAlpha = 1; }
        for (let i = 0; i < 3; i++) { pt(U[i], colsA[i], 'c', A_NAMES[i]); pt(V[i], colsA[perm.indexOf(i)], 's', B_NAMES[i]); }
        const { S, Pr } = mats(), [lr, lc, lt] = lossOf(), mx = cx + R + 56, my = 56, cs = Math.min(64, (W - mx - 16) / 3.6);
        txt(gx, 'cosine similarity (small) and row-softmax (big)', mx, 18, C.muted, '11px sans-serif'); txt(gx, '↓ image   → ' + (simclr ? 'view b' : 'text'), mx, 34, C.muted, '11px sans-serif');
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { const X = mx + j * cs, Y = my + i * cs; gx.fillStyle = AI.mix(C.bg2, C.orange, Pr[i][j]); gx.fillRect(X, Y, cs - 2, cs - 2); if (perm[i] === j) { gx.strokeStyle = C.green; gx.lineWidth = 3; gx.strokeRect(X, Y, cs - 2, cs - 2); } txt(gx, (Pr[i][j] * 100).toFixed(0) + '%', X + cs / 2, Y + cs / 2 + 4, C.fg, 'bold 14px sans-serif', 'center'); txt(gx, 's=' + S[i][j].toFixed(2), X + cs / 2, Y + cs - 10, '#e6edf3', '10px sans-serif', 'center'); }
        // sparkline
        const sy = my + 3 * cs + 28, sh = Math.max(30, H - sy - 14); txt(gx, 'loss history', mx, sy - 6, C.muted, '11px sans-serif'); gx.strokeStyle = C.border; gx.strokeRect(mx, sy, cs * 3, sh);
        if (hist.length > 1) { gx.strokeStyle = C.red; gx.lineWidth = 2; gx.beginPath(); hist.forEach((v, k) => { const X = mx + (k / 159) * cs * 3, Y = sy + sh - AI.clamp(v / 1.8, 0, 1) * sh; k ? gx.lineTo(X, Y) : gx.moveTo(X, Y); }); gx.stroke(); }
        q('#out').innerHTML = `row loss (image→${simclr ? 'view' : 'text'}) <b>${AI.fmt(lr, 3)}</b>\ncolumn loss              <b>${AI.fmt(lc, 3)}</b>\ntotal = average          <b class="o">${AI.fmt(lt, 3)}</b>\nuniform guess ln 3 = 1.099\n<span class="m">green frames = true pairs</span>`;
        const cap = lt > 0.8 ? 'Pairs are scattered: the diagonal is not the largest entry in its row. The gradient pulls matching points together and pushes the others apart.' : lt > 0.25 ? 'Matching pairs are approaching; non-matching ones are being pushed away on the sphere.' : 'Each row and column now peaks on the green (true) pair: the loss is small. Try lowering τ or changing the matches.';
        if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      } else {
        ZS.forEach((z, k) => pt(nrm(z[1]), [C.blue, C.green, C.orange, C.purple][k], 's', z[0].replace('a photo of a ', '"') + '"'));
        const s = pt(zsPos, C.yellow, 'c', IM[zsImg][0], 8); gx.strokeStyle = '#000'; gx.lineWidth = 1; gx.beginPath(); gx.arc(s[0], s[1], 8, 0, 7); gx.stroke();
        ZS.forEach((z, k) => { const b = proj(nrm(z[1]), cx, cy, R); gx.strokeStyle = [C.blue, C.green, C.orange, C.purple][k]; gx.globalAlpha = 0.25 + 0.7 * probs[k]; gx.lineWidth = 1 + 6 * probs[k]; gx.beginPath(); gx.moveTo(s[0], s[1]); gx.lineTo(b[0], b[1]); gx.stroke(); gx.globalAlpha = 1; });
        const mx = cx + R + 56, my = 50, bw = W - mx - 20; txt(gx, `softmax(cos / τ), τ = ${AI.fmt(tau, 3)}`, mx, 24, C.muted, '12px sans-serif');
        ZS.forEach((z, k) => { const y = my + k * 54; txt(gx, z[0], mx, y + 12, C.fg); gx.fillStyle = C.bg2; gx.fillRect(mx, y + 18, bw, 16); gx.fillStyle = [C.blue, C.green, C.orange, C.purple][k]; gx.fillRect(mx, y + 18, bw * probs[k], 16); txt(gx, (probs[k] * 100).toFixed(1) + '%  (cos ' + dotp(zsPos, nrm(z[1])).toFixed(2) + ')', mx + 4, y + 31, '#fff', '11px sans-serif'); });
        q('#out').innerHTML = `test image: <b>${IM[zsImg][0]}</b>\nprediction: <b class="o">${ZS[probs.indexOf(Math.max(...probs))][0].replace('a photo of a ', '')}</b>  (${(Math.max(...probs) * 100).toFixed(1)}%)\n<span class="m">no classifier trained: prompts = classes</span>`;
        const cap = 'The image embedding travels over the sphere; each prompt is a text embedding. Probability = softmax of cosine / τ. Ambiguous images give split probabilities; lower τ makes the choice sharper.'; if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      }
    }, { fps: 30 });
    if (!simclr) q('#mode').dispatchEvent(new Event('input'));
  };

  /* ================= rfgrowth ================= */
  AI.labs.rfgrowth = function (root) {
    const G = 16;
    const { view } = AI.labShell(root, 'How far can one token see?', 'receptive field after L layers: 3×3 convolutions, windowed attention (Swin, M=4), global attention (ViT)',
      '<canvas style="height:300px;cursor:pointer"></canvas><div class="caption" id="cap"></div>',
      `<label>Layers L <span class="v" id="lv"></span><input type="range" id="L" min="1" max="8" value="1"></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="auto" checked> auto-grow depth</label>
       ${playHTML(1, 3)}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), st = {};
    let probe = 8 * G + 8, t = 0, key = '';
    const M = 4, win = (x, y, s) => Math.floor((y + s) / M) * 100 + Math.floor((x + s) / M);
    function sets(L) {
      const px = probe % G, py = Math.floor(probe / G), cnn = new Set(), sw = new Set();
      for (let i = 0; i < G * G; i++) if (Math.max(Math.abs((i % G) - px), Math.abs(Math.floor(i / G) - py)) <= L) cnn.add(i);
      let cur = new Set([probe]);
      for (let l = 0; l < L; l++) { const s = l % 2 ? M / 2 : 0, ws = new Set([...cur].map((i) => win(i % G, Math.floor(i / G), s))), nx = new Set(); for (let i = 0; i < G * G; i++) if (ws.has(win(i % G, Math.floor(i / G), s))) nx.add(i); cur = nx; }
      return [cnn, cur, new Set([...Array(G * G).keys()])];
    }
    wirePlay(root, st, () => { q('#auto').checked = false; q('#L').value = (+q('#L').value % 8) + 1; }, () => { q('#L').value = 1; t = 0; });
    const geom = () => { const pw = (c.w - 20) / 3, cs = Math.min((pw - 14) / G, (c.h - 70) / G); return { pw, cs }; };
    c.cv.addEventListener('pointerdown', (e) => { const r = c.cv.getBoundingClientRect(), { pw, cs } = geom(), mx = e.clientX - r.left, my = e.clientY - r.top; const k = Math.floor((mx - 8) / pw), x = Math.floor((mx - 8 - k * pw) / cs), y = Math.floor((my - 40) / cs); if (k >= 0 && k < 3 && x >= 0 && y >= 0 && x < G && y < G) probe = y * G + x; });
    AI.loop(root, (dt) => {
      if (st.playing && q('#auto').checked) { t += dt * st.speed; if (t > 1.6) { t = 0; q('#L').value = (+q('#L').value % 8) + 1; } }
      const L = +q('#L').value; q('#lv').textContent = L;
      const C = AI.col(), gx = c.ctx, { pw, cs } = geom(); gx.clearRect(0, 0, c.w, c.h); const S = sets(L);
      const names = ['CNN, stacked 3×3 convs', 'Swin-style windows (M=4)', 'ViT, global attention'], cols = [C.blue, C.green, C.orange];
      S.forEach((set, k) => {
        const x0 = 8 + k * pw; txt(gx, names[k], x0, 18, C.fg, 'bold 12px sans-serif'); txt(gx, `sees ${set.size} of ${G * G} tokens`, x0, 34, C.muted, '11px sans-serif');
        gx.fillStyle = C.bg2; gx.fillRect(x0, 40, G * cs, G * cs); gx.fillStyle = cols[k] + '88'; set.forEach((i) => gx.fillRect(x0 + (i % G) * cs, 40 + Math.floor(i / G) * cs, cs, cs));
        gx.strokeStyle = 'rgba(150,150,150,.35)'; gx.lineWidth = 1; gx.beginPath(); for (let i = 0; i <= G; i++) { gx.moveTo(x0 + i * cs, 40); gx.lineTo(x0 + i * cs, 40 + G * cs); gx.moveTo(x0, 40 + i * cs); gx.lineTo(x0 + G * cs, 40 + i * cs); } gx.stroke();
        gx.fillStyle = C.yellow; gx.beginPath(); gx.arc(x0 + ((probe % G) + 0.5) * cs, 40 + (Math.floor(probe / G) + 0.5) * cs, Math.max(3, cs * 0.3), 0, 7); gx.fill();
      });
      const k = [L, probe].join();
      if (k !== key) { key = k; q('#out').innerHTML = `after L = ${L} layers\nCNN: (2L+1)² → side ${2 * L + 1}, sees <b class="b">${S[0].size}</b>\nwindows: <b class="g">${S[1].size}</b>\nViT: <b class="o">${S[2].size}</b> (all)\n<span class="m">click any grid to move the token</span>`; q('#cap').textContent = 'The CNN grows one ring per layer; windowed attention grows in steps as windows alternate; global attention covers everything after one layer.'; }
    }, { fps: 30 });
  };
})();
