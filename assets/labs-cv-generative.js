/* Computer-vision Module 8 labs: aeLab, ganGame, diffLab, srLab, nerfLab, videoLab, trackLab.
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes). All motion is dt-based, slow by default, with speed + pause/step.
 */
(function () {
  const q = (r, s) => AI.$(s, r);
  const PLAY = '<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">↺ Reset</button></div><label>Speed <b id="spv"></b><input type="range" id="sp" min="0.1" max="4" step="0.05" value="1"></label>';
  function ui(root, def, hooks) {
    const s = { playing: true }, sp = q(root, '#sp');
    sp.value = def;
    const upd = () => (q(root, '#spv').textContent = '×' + (+sp.value).toFixed(2));
    sp.oninput = upd; upd();
    q(root, '#play').onclick = () => { s.playing = !s.playing; q(root, '#play').textContent = s.playing ? '⏸ Pause' : '▶ Play'; };
    q(root, '#step').onclick = () => { s.playing = false; q(root, '#play').textContent = '▶ Play'; hooks.step && hooks.step(); };
    q(root, '#reset').onclick = () => hooks.reset && hooks.reset();
    s.speed = () => +sp.value;
    return s;
  }
  function mkImg(n) {
    const c = document.createElement('canvas'); c.width = c.height = n;
    const g = c.getContext('2d'), id = g.createImageData(n, n), d = id.data;
    return {
      n,
      gray(v, lo, hi) { for (let i = 0; i < n * n; i++) { const t = AI.clamp((v[i] - lo) / (hi - lo), 0, 1) * 255; d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = t; d[4 * i + 3] = 255; } g.putImageData(id, 0, 0); },
      div(v, s) { for (let i = 0; i < n * n; i++) { const t = AI.clamp(v[i] / s, -1, 1), a = Math.abs(t); const base = 40; if (t >= 0) { d[4 * i] = base + a * 200; d[4 * i + 1] = base + a * 100; d[4 * i + 2] = base - a * 10; } else { d[4 * i] = base - a * 10; d[4 * i + 1] = base + a * 90; d[4 * i + 2] = base + a * 200; } d[4 * i + 3] = 255; } g.putImageData(id, 0, 0); },
      draw(ctx, x, y, w, h) { ctx.imageSmoothingEnabled = false; ctx.drawImage(c, x, y, w, h === undefined ? w : h); },
    };
  }
  const label = (g, txt, x, y, col, al, font) => { g.fillStyle = col; g.textAlign = al || 'left'; g.font = font || '11px sans-serif'; g.fillText(txt, x, y); };
  const frame = (g, x, y, w, h, col, lw) => { g.strokeStyle = col; g.lineWidth = lw || 1; g.strokeRect(x + 0.5, y + 0.5, w, h); };
  const PSNR = (a, b, peak) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; const m = s / a.length; return { mse: m, psnr: m < 1e-9 ? 99 : 10 * Math.log10((peak * peak) / m) }; };

  /* ======================= autoencoder ======================= */
  AI.labs.aeLab = function (root) {
    const N = 20, KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 576], NAMES = ['circle', 'square', 'triangle', 'cross'];
    const protos = [0, 1, 2, 3].map((k) => {
      const a = new Float32Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const px = ((x + 0.5) / N) * 2 - 1, py = ((y + 0.5) / N) * 2 - 1; let d;
        if (k === 0) d = Math.hypot(px, py) - 0.62;
        else if (k === 1) d = Math.max(Math.abs(px), Math.abs(py)) - 0.55;
        else if (k === 2) d = Math.max(py - 0.55, Math.abs(px) - (py + 0.65) * 0.54);
        else d = Math.min(Math.max(Math.abs(px) - 0.2, Math.abs(py) - 0.65), Math.max(Math.abs(px) - 0.65, Math.abs(py) - 0.2));
        a[y * N + x] = AI.clamp(0.5 - (d * N) / 2, 0, 1);
      }
      return a;
    });
    const CEN = [[-0.6, -0.5], [0.6, -0.5], [-0.6, 0.5], [0.6, 0.5]];
    const rnd = AI.rng(7), pts = [];
    CEN.forEach((c, k) => { for (let i = 0; i < 16; i++) pts.push({ k, x: c[0] + 0.15 * AI.randn(rnd), y: c[1] + 0.15 * AI.randn(rnd) }); });
    const { view, ctl } = AI.labShell(root, 'Autoencoder: latent space and bottleneck', 'drag the latent point, shrink the bottleneck',
      '<canvas id="a" style="height:270px;touch-action:none;cursor:grab"></canvas><div class="caption" id="capa"></div><canvas id="b" style="height:210px"></canvas><div class="caption" id="capb"></div>',
      PLAY + `<label>Image for the bottleneck test <select id="scene"><option value="house">house</option><option value="shapes">shapes</option><option value="digit">handwritten 7</option><option value="edges">edges</option></select></label>
      <label>Bottleneck size: <b id="kv"></b> numbers<input type="range" id="k" min="0" max="9" step="1" value="5"></label>
      <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="sweep"> auto-sweep the bottleneck</label><div class="readout" id="out"></div>`);
    const ca = AI.canvas(q(root, '#a')), cb = AI.canvas(q(root, '#b'));
    const imgs = protos.map(() => mkImg(N)), dec = mkImg(N), I0 = mkImg(24), I1 = mkImg(24), I2 = mkImg(24);
    let z = [-0.2, 0.1], t = 0, drag = false, ps = 200, px0 = 24, py0 = 20, sweepPos = 5, sweepDir = 1, kIdx = 5, dirty = true;
    protos.forEach((p, i) => imgs[i].gray(p, 0, 1));
    const w = new Float32Array(4), out = new Float32Array(N * N);
    const decode = (zz) => {
      let s = 0; for (let k = 0; k < 4; k++) { const d2 = (zz[0] - CEN[k][0]) ** 2 + (zz[1] - CEN[k][1]) ** 2; w[k] = Math.exp(-d2 / (2 * 0.45 * 0.45)); s += w[k]; }
      for (let k = 0; k < 4; k++) w[k] /= s || 1;
      out.fill(0); for (let k = 0; k < 4; k++) for (let i = 0; i < N * N; i++) out[i] += w[k] * protos[k][i];
      dec.gray(out, 0, 1);
    };
    // DCT bottleneck
    const n = 24, B = [];
    for (let u = 0; u < n; u++) { B.push([]); for (let x = 0; x < n; x++) B[u].push((u ? Math.sqrt(2 / n) : Math.sqrt(1 / n)) * Math.cos((Math.PI * (2 * x + 1) * u) / (2 * n))); }
    const order = []; for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) order.push([u, v]);
    order.sort((a, b) => a[0] + a[1] - (b[0] + b[1]) || a[0] - b[0]);
    let orig, F, rec = new Float32Array(n * n), err = new Float32Array(n * n), stat = { mse: 0, psnr: 0 };
    const loadScene = () => {
      orig = CV().gray(CV().resample(q(root, '#scene').value, n));
      F = new Float32Array(n * n);
      for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) { let s = 0; for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) s += B[u][y] * B[v][x] * orig[y * n + x]; F[u * n + v] = s; }
      I0.gray(orig, 0, 255); recon();
    };
    const CV = () => AI.cv;
    const recon = () => {
      const k = KS[kIdx]; q(root, '#kv').textContent = k;
      rec.fill(0);
      for (let m = 0; m < k; m++) { const [u, v] = order[m], f = F[u * n + v]; for (let y = 0; y < n; y++) { const by = B[u][y] * f; for (let x = 0; x < n; x++) rec[y * n + x] += by * B[v][x]; } }
      for (let i = 0; i < n * n; i++) err[i] = Math.abs(rec[i] - orig[i]);
      I1.gray(rec, 0, 255); I2.gray(err, 0, 128);
      stat = PSNR(rec, orig, 255); dirty = true;
      q(root, '#out').innerHTML = `<span class="m">bottleneck</span> <b class="o">${k}</b> numbers\n<span class="m">compression</span> 576 / ${k} = <b class="b">${AI.fmt(576 / k, 3)} : 1</b>\n<span class="m">MSE</span> ${AI.fmt(stat.mse, 4)}   <span class="m">PSNR</span> ${AI.fmt(stat.psnr, 3)} dB`;
    };
    q(root, '#scene').onchange = loadScene;
    q(root, '#k').oninput = () => { kIdx = +q(root, '#k').value; sweepPos = kIdx; recon(); };
    const S = ui(root, 0.6, { step: () => { t += 0.4; }, reset: () => { z = [-0.2, 0.1]; t = 0; } });
    const toZ = (e) => { const r = q(root, '#a').getBoundingClientRect(); return [AI.clamp(((e.clientX - r.left - px0) / ps) * 2 - 1, -1.1, 1.1), AI.clamp(1 - ((e.clientY - r.top - py0) / ps) * 2, -1.1, 1.1)]; };
    const ea = q(root, '#a');
    ea.addEventListener('pointerdown', (e) => { drag = true; ea.setPointerCapture(e.pointerId); S.playing = false; q(root, '#play').textContent = '▶ Play'; z = toZ(e); });
    ea.addEventListener('pointermove', (e) => { if (drag) z = toZ(e); });
    ea.addEventListener('pointerup', () => (drag = false));
    const start = () => { if (AI.cv) loadScene(); else setTimeout(start, 50); };
    start();
    AI.loop(root, (dt) => {
      if (S.playing && !drag) t += dt * 0.5 * S.speed();
      if (S.playing && !drag) z = [0.85 * Math.sin(t), 0.8 * Math.sin(0.7 * t + 1)];
      if (q(root, '#sweep').checked && S.playing) {
        sweepPos += sweepDir * dt * 0.6 * S.speed(); if (sweepPos > 9) { sweepPos = 9; sweepDir = -1; } if (sweepPos < 0) { sweepPos = 0; sweepDir = 1; }
        const ni = Math.round(sweepPos); if (ni !== kIdx) { kIdx = ni; q(root, '#k').value = ni; if (orig) recon(); }
      }
      const C = AI.col();
      // panel A
      let g = ca.ctx; g.clearRect(0, 0, ca.w, ca.h);
      ps = Math.min(ca.h - 40, ca.w * 0.46); px0 = 24; py0 = 22;
      const X = (v) => px0 + ((v + 1) / 2) * ps, Y = (v) => py0 + (1 - (v + 1) / 2) * ps;
      g.fillStyle = C.bg2; g.fillRect(px0, py0, ps, ps); frame(g, px0, py0, ps, ps, C.border);
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(X(0), py0); g.lineTo(X(0), py0 + ps); g.moveTo(px0, Y(0)); g.lineTo(px0 + ps, Y(0)); g.stroke();
      const cols = [C.red, C.blue, C.green, C.purple];
      pts.forEach((p) => { g.fillStyle = cols[p.k]; g.globalAlpha = 0.7; g.beginPath(); g.arc(X(p.x), Y(p.y), 3, 0, 7); g.fill(); });
      g.globalAlpha = 1;
      CEN.forEach((c, k) => { imgs[k].draw(g, X(c[0]) - 15, Y(c[1]) - 15, 30); frame(g, X(c[0]) - 15, Y(c[1]) - 15, 30, 30, cols[k], 2); label(g, NAMES[k], X(c[0]), Y(c[1]) + 28, C.muted, 'center', '10px sans-serif'); });
      label(g, 'latent plane (z₁ →, z₂ ↑): each training image is one dot', px0, 14, C.muted);
      decode(z);
      const zx = X(z[0]), zy = Y(z[1]);
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.arc(zx, zy, 8, 0, 7); g.moveTo(zx - 13, zy); g.lineTo(zx + 13, zy); g.moveTo(zx, zy - 13); g.lineTo(zx, zy + 13); g.stroke();
      const ds = Math.min(ps * 0.8, ca.w - ps - 130), dx = px0 + ps + 70;
      AI.arrow(g, px0 + ps + 8, py0 + ps / 2, dx - 6, py0 + ps / 2, C.orange, 3); label(g, 'decoder', px0 + ps + 36, py0 + ps / 2 - 8, C.orange, 'center');
      if (ds > 20) {
        dec.draw(g, dx, py0 + 10, ds); frame(g, dx, py0 + 10, ds, ds, C.fg);
        label(g, `z = (${AI.fmt(z[0], 2)}, ${AI.fmt(z[1], 2)})`, dx, py0 + ds + 28, C.fg, 'left', '12px ui-monospace,monospace');
        label(g, '2 numbers → 400 pixels', dx, py0 + ds + 44, C.muted);
        NAMES.forEach((nm, k) => { label(g, `${nm} ${Math.round(w[k] * 100)}%`, dx, py0 + ds + 60 + k * 13, cols[k], 'left'); });
      }
      const best = w.indexOf(Math.max(...w));
      q(root, '#capa').textContent = Math.max(...w) > 0.9 ? `Decoder output: a clean ${NAMES[best]}, the point sits on its cluster.` : `Between clusters the decoder blends the nearest shapes (${NAMES.map((nm, k) => `${nm} ${Math.round(w[k] * 100)}%`).join(', ')}): a smooth morph.`;
      // panel B
      if (!orig) return;
      g = cb.ctx; g.clearRect(0, 0, cb.w, cb.h);
      const ts = Math.min(cb.h - 62, (cb.w - 60) / 3), gx = (cb.w - 3 * ts) / 4;
      [[I0, 'original 24×24 = 576'], [I1, `decoded from ${KS[kIdx]} numbers`], [I2, 'error (brighter = worse)']].forEach(([im, tx], i) => { const x = gx + i * (ts + gx); im.draw(g, x, 22, ts); frame(g, x, 22, ts, ts, C.border); label(g, tx, x + ts / 2, 15, C.muted, 'center'); });
      const bw = cb.w - 40; g.fillStyle = C.border; g.fillRect(20, 22 + ts + 12, bw, 8); g.fillStyle = C.orange; g.fillRect(20, 22 + ts + 12, Math.max(2, (bw * KS[kIdx]) / 576), 8);
      label(g, `code size ${KS[kIdx]} of 576 (${AI.fmt(576 / KS[kIdx], 3)}:1)   PSNR ${AI.fmt(stat.psnr, 3)} dB`, 20, 22 + ts + 36, C.fg, 'left', '12px ui-monospace,monospace');
      q(root, '#capb').textContent = KS[kIdx] < 16 ? 'Tiny bottleneck: only the coarse brightness layout survives. The kept numbers are the lowest-frequency (smoothest) components: a linear stand-in for what an encoder learns.' : KS[kIdx] < 256 ? 'Medium bottleneck: shapes and large edges return, fine detail is still missing.' : 'Big bottleneck: near-perfect copy, but there is hardly any compression, so little has been learned.';
    }, { fps: 30 });
  };

  /* ======================= GAN game ======================= */
  AI.labs.ganGame = function (root) {
    const { view, ctl } = AI.labShell(root, 'The GAN game in 1-D', 'discriminator D (blue) and generator fakes (orange) against the real data (green)',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap"></div>',
      PLAY + `<label>Real data <select id="real"><option value="2">two modes (−2 and +2)</option><option value="1">one mode (at +1.5)</option><option value="3">three modes</option></select></label>
      <label>Generator loss <select id="loss"><option value="ns">non-saturating: −log D(G(z))</option><option value="mm">minimax: log(1−D(G(z)))</option></select></label>
      <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="col"> mode collapse (generator can only shift all fakes together)</label><div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), H = 12, NF = 80, NR = 80, LD = 0.4, LG = 0.35;
    const CENT = { 1: [1.5], 2: [-2, 2], 3: [-2.6, 0, 2.6] }, SD = 0.45;
    let r, w1, b1, w2, b2, xs, step, acc, dMean = [0, 0], lastMove = [], dirty = true, pulse = 0;
    const sg = (a) => 1 / (1 + Math.exp(-a));
    const fwd = (x) => { let a = b2; for (let j = 0; j < H; j++) a += w2[j] * Math.tanh(w1[j] * x + b1[j]); return a; };
    const da = (x) => { let s = 0; for (let j = 0; j < H; j++) { const h = Math.tanh(w1[j] * x + b1[j]); s += w2[j] * (1 - h * h) * w1[j]; } return s; };
    const reset = () => {
      r = AI.rng(5); w1 = []; b1 = []; w2 = []; b2 = 0; xs = []; step = 0; acc = 0; lastMove = new Array(NF).fill(0);
      for (let j = 0; j < H; j++) { w1.push(AI.randn(r)); b1.push(AI.randn(r)); w2.push(AI.randn(r) * 0.5); }
      for (let i = 0; i < NF; i++) xs.push(AI.randn(r) * 0.35);
      dirty = true;
    };
    const real = () => { const C = CENT[q(root, '#real').value]; return C[Math.floor(r() * C.length)] + AI.randn(r) * SD; };
    const one = () => {
      const gw1 = new Array(H).fill(0), gb1 = new Array(H).fill(0), gw2 = new Array(H).fill(0); let gb2 = 0, dr = 0, df = 0;
      const add = (x, g) => { for (let j = 0; j < H; j++) { const h = Math.tanh(w1[j] * x + b1[j]); gw2[j] += g * h; const dp = g * w2[j] * (1 - h * h); gw1[j] += dp * x; gb1[j] += dp; } gb2 += g; };
      for (let i = 0; i < NR; i++) { const x = real(), D = sg(fwd(x)); dr += D / NR; add(x, -(1 - D) / NR); }
      for (const x of xs) { const D = sg(fwd(x)); df += D / NF; add(x, D / NF); }
      for (let j = 0; j < H; j++) { w1[j] -= LD * gw1[j]; b1[j] -= LD * gb1[j]; w2[j] -= LD * gw2[j]; } b2 -= LD * gb2;
      dMean = [dr, df];
      const ns = q(root, '#loss').value === 'ns', col = q(root, '#col').checked, mv = []; let mean = 0;
      for (const x of xs) { const D = sg(fwd(x)), g = (ns ? 1 - D : D) * da(x); mv.push(g); mean += g / NF; }
      let mx = 0;
      for (let i = 0; i < NF; i++) { const m = col ? mean : mv[i]; xs[i] += LG * m; lastMove[i] = LG * m; mx += xs[i] / NF; }
      if (col) for (let i = 0; i < NF; i++) xs[i] += (mx - xs[i]) * 0.05;
      step++; dirty = true;
    };
    const S = ui(root, 0.6, { step: one, reset });
    q(root, '#real').onchange = reset; q(root, '#loss').onchange = reset; q(root, '#col').onchange = reset;
    reset();
    AI.loop(root, (dt, time) => {
      if (S.playing) { acc += dt * 8 * S.speed(); const k = Math.floor(acc); acc -= k; for (let i = 0; i < Math.min(k, 6); i++) one(); }
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h, x0 = 36, x1 = W - 36, top = 28, bot = Hh - 56;
      const X = (v) => x0 + ((v + 4.5) / 9) * (x1 - x0), Y = (p) => bot - p * (bot - top);
      g.clearRect(0, 0, W, Hh);
      g.fillStyle = C.bg2; g.fillRect(x0, top, x1 - x0, bot - top); frame(g, x0, top, x1 - x0, bot - top, C.border);
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(x0, Y(0.5)); g.lineTo(x1, Y(0.5)); g.stroke(); g.setLineDash([]);
      label(g, 'D = 0.5', x1 + 2, Y(0.5) + 4, C.muted, 'left', '10px sans-serif'); label(g, '1', x1 + 2, Y(1) + 4, C.muted, 'left', '10px sans-serif'); label(g, '0', x1 + 2, Y(0) + 4, C.muted, 'left', '10px sans-serif');
      const cen = CENT[q(root, '#real').value], G = 90, rp = [], fp = []; let mxr = 0, mxf = 0;
      for (let i = 0; i <= G; i++) {
        const v = -4.5 + (9 * i) / G; let a = 0; for (const m of cen) a += Math.exp(-((v - m) ** 2) / (2 * SD * SD)) / (SD * Math.sqrt(2 * Math.PI) * cen.length); rp.push(a); mxr = Math.max(mxr, a);
        let b = 0; for (const x of xs) b += Math.exp(-((v - x) ** 2) / (2 * 0.25 * 0.25)) / (0.25 * Math.sqrt(2 * Math.PI) * NF); fp.push(b); mxf = Math.max(mxf, b);
      }
      const sc = 0.75 / Math.max(mxr, mxf, 0.01);
      const fill = (arr, col) => { g.fillStyle = col; g.globalAlpha = 0.35; g.beginPath(); g.moveTo(X(-4.5), Y(0)); arr.forEach((p, i) => g.lineTo(X(-4.5 + (9 * i) / G), Y(p * sc))); g.lineTo(X(4.5), Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); arr.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(-4.5 + (9 * i) / G), Y(p * sc))); g.stroke(); };
      fill(rp, C.green); fill(fp, C.orange);
      g.strokeStyle = C.blue; g.lineWidth = 3; g.beginPath(); for (let i = 0; i <= 120; i++) { const v = -4.5 + (9 * i) / 120, d = sg(fwd(v)); (i ? g.lineTo : g.moveTo).call(g, X(v), Y(d)); } g.stroke();
      label(g, 'real density', x0 + 6, top + 14, C.green); label(g, 'fake density', x0 + 96, top + 14, C.orange); label(g, 'D(x): "is it real?"', x0 + 190, top + 14, C.blue);
      // particles + arrows
      const py = bot + 18; g.strokeStyle = C.border; g.beginPath(); g.moveTo(x0, py); g.lineTo(x1, py); g.stroke();
      xs.forEach((x, i) => { g.fillStyle = C.orange; g.beginPath(); g.arc(X(AI.clamp(x, -4.5, 4.5)), py + (i % 3) * 5 - 5, 3, 0, 7); g.fill(); if (i % 4 === 0 && Math.abs(lastMove[i]) > 1e-4) AI.arrow(g, X(x), py + 16, X(x + lastMove[i] * 6), py + 16, C.orange, 1.5); });
      label(g, 'fakes G(z) ·  arrow = this G step (move up the slope of D)', x0, py + 34, C.muted, 'left', '10px sans-serif');
      const sorted = xs.slice().sort((a, b) => a - b), mean = xs.reduce((a, b) => a + b, 0) / NF, sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / NF), left = xs.filter((x) => x < 0).length;
      q(root, '#out').innerHTML = `<span class="m">step</span> ${step}\n<span class="m">mean D(real)</span> <b class="g">${AI.fmt(dMean[0], 2)}</b>   <span class="m">mean D(fake)</span> <b class="o">${AI.fmt(dMean[1], 2)}</b>\n<span class="m">fakes</span> mean ${AI.fmt(mean, 2)}, spread ${AI.fmt(sd, 2)}`;
      const colOn = q(root, '#col').checked, nm = cen.length;
      q(root, '#cap').textContent = step < 2 ? '① D steps: tilt the blue curve up where real data lives and down where fakes are. ② G steps: every fake moves up the slope of D. Press play.'
        : colOn ? (sd < 0.3 ? `Collapsed: all fakes sit in one spot (spread ${AI.fmt(sd, 2)}). D learns that spot is fake, G hops elsewhere, and the other real mode is never covered.` : 'G can only move all fakes together, so the cloud shrinks toward one point.')
        : Math.abs(dMean[0] - dMean[1]) < 0.12 ? `D can barely tell real from fake (D(real) ${AI.fmt(dMean[0], 2)} vs D(fake) ${AI.fmt(dMean[1], 2)}): near equilibrium. Fakes now follow the real shape.`
        : 'D (blue) is winning: it is high where real data is and low elsewhere; the fakes slide up that slope, which moves them toward the real modes.';
    }, { fps: 30 });
  };

  /* ======================= diffusion ======================= */
  AI.labs.diffLab = function (root) {
    const n = 16, D = n * n, T = 50, NAMES = ['circle', 'square', 'triangle', 'cross'];
    const x0s = [0, 1, 2, 3].map((k) => {
      const a = new Float32Array(D);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const px = ((x + 0.5) / n) * 2 - 1, py = ((y + 0.5) / n) * 2 - 1; let d;
        if (k === 0) d = Math.hypot(px, py) - 0.62; else if (k === 1) d = Math.max(Math.abs(px), Math.abs(py)) - 0.55; else if (k === 2) d = Math.max(py - 0.55, Math.abs(px) - (py + 0.65) * 0.54); else d = Math.min(Math.max(Math.abs(px) - 0.2, Math.abs(py) - 0.65), Math.max(Math.abs(px) - 0.65, Math.abs(py) - 0.2));
        a[y * n + x] = 2 * AI.clamp(0.5 - (d * n) / 2, 0, 1) - 1;
      }
      return a;
    });
    const ab = []; { let p = 1; for (let t = 0; t < T; t++) { const beta = 0.004 + (0.2 * t) / (T - 1) ** 1 * 1.0; p *= 1 - Math.min(beta, 0.5); ab.push(p); } }
    const { view, ctl } = AI.labShell(root, 'Diffusion: noise in, picture out', 'forward = add noise (fixed formula); reverse = predict the noise and remove it',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div><div class="readout" id="probe"></div>',
      PLAY + `<div><div class="row" id="mode"><button class="chip" data-m="f">Forward: add noise</button><button class="chip" data-m="r">Reverse: sample</button></div></div>
      <label>Prompt <select id="cls">${NAMES.map((a, i) => `<option value="${i}">${a}</option>`).join('')}</select></label>
      <label>Guidance scale w: <b id="wv"></b><input type="range" id="w" min="0" max="8" step="0.5" value="4"></label>
      <label>Sampling steps: <b id="sv"></b><input type="range" id="S" min="5" max="40" step="1" value="20"></label>
      <label>Forward step t (drag to scrub): <b id="tv"></b><input type="range" id="t" min="0" max="${T - 1}" step="1" value="0"></label>
      <div class="row"><button class="btn" id="newn">New noise</button></div>`);
    const c = AI.canvas(AI.$('canvas', view)), tiles = Array.from({ length: 6 }, () => mkImg(n)), thumbs = Array.from({ length: 12 }, () => mkImg(n));
    let mode = 'f', seed = 3, eps0, x, ts, si, phase, ph, hist, hold, tF = 0, ready = false;
    let eu = new Float32Array(D), ec = new Float32Array(D), eg = new Float32Array(D), x0h = new Float32Array(D), xn = new Float32Array(D), xt = new Float32Array(D), info = {};
    const wts = new Float32Array(4);
    const newNoise = () => { const r = AI.rng(seed++); eps0 = new Float32Array(D); for (let i = 0; i < D; i++) eps0[i] = AI.randn(r); };
    const restartRev = () => {
      x = Float32Array.from(eps0); const S = +q(root, '#S').value; ts = []; for (let i = 0; i < S; i++) ts.push(Math.round(((S - 1 - i) * (T - 1)) / Math.max(1, S - 1)));
      si = 0; phase = 0; ph = 0; hold = 0; hist = [Float32Array.from(x)]; computeStep();
    };
    const computeStep = () => {
      const t = ts[si], a = ab[t], sa = Math.sqrt(a), sn = Math.sqrt(1 - a), w = +q(root, '#w').value, k = +q(root, '#cls').value;
      let mx = -1e9; for (let j = 0; j < 4; j++) { let d2 = 0; for (let i = 0; i < D; i++) d2 += (x[i] - sa * x0s[j][i]) ** 2; wts[j] = -d2 / (2 * (1 - a)); mx = Math.max(mx, wts[j]); }
      let s = 0; for (let j = 0; j < 4; j++) { wts[j] = Math.exp(wts[j] - mx); s += wts[j]; } for (let j = 0; j < 4; j++) wts[j] /= s;
      const tp = si + 1 < ts.length ? ts[si + 1] : -1, ap = tp < 0 ? 1 : ab[tp];
      for (let i = 0; i < D; i++) {
        let mu = 0; for (let j = 0; j < 4; j++) mu += wts[j] * x0s[j][i];
        eu[i] = (x[i] - sa * mu) / sn; ec[i] = (x[i] - sa * x0s[k][i]) / sn; eg[i] = eu[i] + w * (ec[i] - eu[i]);
        x0h[i] = AI.clamp((x[i] - sn * eg[i]) / sa, -1, 1);
        const ek = (x[i] - sa * x0h[i]) / sn; // noise implied by the clipped x0 estimate (standard clipping practice)
        xn[i] = Math.sqrt(ap) * x0h[i] + Math.sqrt(1 - ap) * ek;
      }
      info = { t, a, tp, ap };
      tiles[0].gray(x, -1.8, 1.8); tiles[1].div(eu, 3); tiles[2].div(ec, 3); tiles[3].div(eg, 3); tiles[4].gray(x0h, -1, 1); tiles[5].gray(xn, -1.8, 1.8);
    };
    const advance = () => { x = Float32Array.from(xn); hist.push(Float32Array.from(x)); si++; phase = 0; ph = 0; if (si >= ts.length) { si = ts.length - 1; hold = 0.01; phase = 5; } else computeStep(); };
    const forwardImg = (t) => { const k = +q(root, '#cls').value, a = ab[t], sa = Math.sqrt(a), sn = Math.sqrt(1 - a); for (let i = 0; i < D; i++) xt[i] = sa * x0s[k][i] + sn * eps0[i]; tiles[0].gray(x0s[k], -1, 1); tiles[1].div(eps0, 3); tiles[2].gray(xt, -1.8, 1.8); };
    const setMode = (m) => { mode = m; AI.$$('#mode button', root).forEach((b) => b.classList.toggle('primary', b.dataset.m === m)); if (m === 'r') restartRev(); else forwardImg(tF); };
    AI.$$('#mode button', root).forEach((b) => (b.onclick = () => setMode(b.dataset.m)));
    q(root, '#newn').onclick = () => { newNoise(); setMode(mode); };
    ['#cls', '#w', '#S'].forEach((s) => q(root, s).addEventListener('input', () => { q(root, '#wv').textContent = q(root, '#w').value; q(root, '#sv').textContent = q(root, '#S').value; if (s !== '#w' || true) setMode(mode); }));
    q(root, '#t').oninput = () => { tF = +q(root, '#t').value; S.playing = false; q(root, '#play').textContent = '▶ Play'; if (mode === 'f') forwardImg(tF); };
    const S = ui(root, 1.2, { step: () => { if (mode === 'f') { tF = Math.min(T - 1, tF + 1); q(root, '#t').value = tF; forwardImg(tF); } else { if (phase < 3) phase++; else advance(); ph = 0; } }, reset: () => { tF = 0; q(root, '#t').value = 0; newNoise(); setMode(mode); } });
    q(root, '#wv').textContent = q(root, '#w').value; q(root, '#sv').textContent = q(root, '#S').value;
    newNoise(); setMode('f');
    const PH = 0.55;
    AI.loop(root, (dt) => {
      if (S.playing) {
        if (mode === 'f') { tF += dt * 4 * S.speed(); if (tF > T - 1 + 8) tF = 0; const ti = Math.min(T - 1, Math.floor(tF)); if (+q(root, '#t').value !== ti) { q(root, '#t').value = ti; forwardImg(ti); } }
        else if (phase < 4) { ph += dt * S.speed() / PH * 1.0; if (ph >= 1) { ph = 0; phase++; if (phase === 4) advance(); } }
        else if (phase === 5) { hold += dt; if (hold > 4 / Math.max(0.3, S.speed())) { newNoise(); restartRev(); } }
      }
      const C = AI.col(), g = c.ctx, W = c.w; g.clearRect(0, 0, W, c.h);
      q(root, '#tv').textContent = Math.round(tF > T - 1 ? T - 1 : tF);
      if (mode === 'f') {
        const t = Math.min(T - 1, Math.floor(tF)), a = ab[t], ts2 = Math.min(150, (W - 160) / 3), gx = (W - 3 * ts2) / 4, y = 40;
        [['x₀ clean image', 0], ['ε noise (fixed draw)', 1], ['x_t noisy', 2]].forEach(([tx, i], k) => { const xx = gx + k * (ts2 + gx); tiles[i].draw(g, xx, y, ts2); frame(g, xx, y, ts2, ts2, k === 2 ? C.orange : C.border, k === 2 ? 3 : 1); label(g, tx, xx + ts2 / 2, y - 8, C.muted, 'center'); });
        label(g, '×' + AI.fmt(Math.sqrt(a), 3), gx + ts2 + gx / 2 - 8, y + ts2 / 2, C.blue, 'center', '13px ui-monospace,monospace'); label(g, '+ ×' + AI.fmt(Math.sqrt(1 - a), 3), gx * 2 + 2 * ts2 + gx / 2 - 14, y + ts2 / 2, C.orange, 'center', '13px ui-monospace,monospace');
        // noise level bar
        const by = y + ts2 + 30, bw = W - 80; g.fillStyle = C.border; g.fillRect(40, by, bw, 10); g.fillStyle = C.blue; g.fillRect(40, by, bw * a, 10); g.fillStyle = C.orange; g.fillRect(40 + bw * a, by, bw * (1 - a), 10);
        label(g, `signal share √ᾱ = ${AI.fmt(Math.sqrt(a), 3)}   noise share √(1−ᾱ) = ${AI.fmt(Math.sqrt(1 - a), 3)}`, 40, by + 28, C.fg, 'left', '12px ui-monospace,monospace');
        const k = +q(root, '#cls').value, p = 8 * n + 8;
        q(root, '#cap').textContent = t < 3 ? 'Step 0 to 3: almost no noise. x_t = √ᾱ·x₀ + √(1−ᾱ)·ε, with ᾱ near 1.' : a > 0.15 ? `t = ${t}: the shape is fading under the noise. Coarse layout survives longest, fine detail goes first.` : `t = ${t}: almost pure noise (ᾱ = ${AI.fmt(a, 3)}). This is where sampling will start.`;
        q(root, '#probe').innerHTML = `<span class="m">probe pixel (8,8)</span>  x₀ = ${AI.fmt(x0s[k][p], 3)}  ε = ${AI.fmt(eps0[p], 3)}  ᾱ = ${AI.fmt(a, 3)}  →  x_t = ${AI.fmt(Math.sqrt(a), 3)}·x₀ + ${AI.fmt(Math.sqrt(1 - a), 3)}·ε = <b class="o">${AI.fmt(xt[p], 3)}</b>`;
        for (let i = 0; i < 12; i++) { const tt = Math.round((i * (T - 1)) / 11), aa = Math.sqrt(ab[tt]), bb = Math.sqrt(1 - ab[tt]), tmp = new Float32Array(D); for (let j = 0; j < D; j++) tmp[j] = aa * x0s[k][j] + bb * eps0[j]; thumbs[i].gray(tmp, -1.8, 1.8); const sz = Math.min(40, (W - 40) / 12 - 3), xx = 20 + i * (sz + 3), yy = c.h - sz - 18; thumbs[i].draw(g, xx, yy, sz); frame(g, xx, yy, sz, sz, tt <= t && (i === 11 || Math.round(((i + 1) * (T - 1)) / 11) > t) ? C.orange : C.border, 2); }
        label(g, 'strip: t = 0 … ' + (T - 1), 20, c.h - 4, C.muted);
      } else {
        const ts2 = Math.min(84, (W - 40) / 6.6), gap = (W - 6 * ts2) / 7, y = 52;
        const names = ['x_t input', 'ε_u none', 'ε_c prompt', 'ε guided', 'x̂₀ guess', 'x_(t−1)'];
        const show = [true, phase >= 1, phase >= 1, phase >= 2, phase >= 3, phase >= 4 || phase === 5];
        const hot = [0, 1, 1, 2, 3, 4][Math.min(phase, 4)];
        for (let i = 0; i < 6; i++) {
          const xx = gap + i * (ts2 + gap);
          if (show[i]) { tiles[i].draw(g, xx, y, ts2); } else { g.fillStyle = C.bg2; g.fillRect(xx, y, ts2, ts2); }
          frame(g, xx, y, ts2, ts2, (i === 0 && phase === 0) || (phase === 1 && (i === 1 || i === 2)) || (phase === 2 && i === 3) || (phase === 3 && i === 4) || (phase >= 4 && i === 5) ? C.orange : C.border, 3);
          label(g, names[i], xx + ts2 / 2, y - 8, show[i] ? C.fg : C.muted, 'center', '10px sans-serif');
          if (i < 5) AI.arrow(g, xx + ts2 + 2, y + ts2 / 2, xx + ts2 + gap - 2, y + ts2 / 2, show[i + 1] ? C.orange : C.border, 2);
        }
        const k = +q(root, '#cls').value, w = +q(root, '#w').value, p = 8 * n + 8, t = info.t, a = info.a;
        const cap = ['① The network sees the noisy x_t and the step t (via the time embedding).', '② It predicts the noise twice: ε_u with no prompt (a mix of all four shapes) and ε_c with the prompt "' + NAMES[k] + '".', `③ Guidance: ε = ε_u + ${w}·(ε_c − ε_u). Higher w pushes harder toward the prompt.`, '④ Estimate the clean image x̂₀ = (x_t − √(1−ᾱ)·ε)/√ᾱ, then re-noise it to the next, lower level: x_(t−1).', 'Next step.', 'Finished: the noise has become a ' + NAMES[k] + ' (w = ' + w + '). New noise starts automatically.'];
        q(root, '#cap').textContent = `Step ${si + 1} of ${ts.length} (t = ${t}).  ` + cap[phase];
        q(root, '#probe').innerHTML = `<span class="m">probe pixel (8,8)</span>  t=${t} ᾱ=${AI.fmt(a, 3)}  x_t=${AI.fmt(x[p], 3)}  ε_u=${AI.fmt(eu[p], 3)}  ε_c=${AI.fmt(ec[p], 3)}  ε=<b class="o">${AI.fmt(eg[p], 3)}</b>  x̂₀=${AI.fmt(x0h[p], 3)}  x_(t−1)=${AI.fmt(xn[p], 3)}\n<span class="m">shape belief of the unprompted network:</span> ${NAMES.map((nm, j) => `${nm} ${Math.round(wts[j] * 100)}%`).join(', ')}`;
        // history strip
        const sz = Math.min(36, (W - 40) / Math.max(12, hist.length) - 3);
        hist.forEach((h, i) => { if (i >= 24) return; const im = thumbs[i % 12]; im.gray(h, -1.8, 1.8); const xx = 20 + i * (sz + 3), yy = c.h - sz - 18; if (xx + sz < W) { im.draw(g, xx, yy, sz); frame(g, xx, yy, sz, sz, i === si ? C.orange : C.border, 2); } });
        label(g, 'history of sampling steps (left = pure noise)', 20, c.h - 4, C.muted);
      }
    }, { fps: 30 });
  };

  /* ======================= super-resolution ======================= */
  AI.labs.srLab = function (root) {
    const HR = 60;
    const { view, ctl } = AI.labShell(root, 'Super-resolution: nearest, bilinear, learned', 'a 5×5 filter is trained by gradient descent to sharpen the bilinear upsample',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap"></div>',
      PLAY + `<label>Image <select id="scene"><option value="house">house</option><option value="shapes">shapes</option><option value="digit">handwritten 7</option><option value="edges">edges</option></select></label>
      <label>Scale factor <select id="f"><option value="2">2× (30→60)</option><option value="3">3× (20→60)</option><option value="4" selected>4× (15→60)</option><option value="5">5× (12→60)</option></select></label>
      <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="err"> show error</label><div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), tl = [0, 1, 2, 3].map(() => mkImg(HR));
    let hr, lr, f, nn, bl, K, V, b0, bv, out, iter, acc, hist = [], dirty = true, pn, pb;
    const bil = () => { const L = HR / f; bl = new Float32Array(HR * HR); nn = new Float32Array(HR * HR); for (let y = 0; y < HR; y++) for (let x = 0; x < HR; x++) { const u = (x + 0.5) / f - 0.5, v = (y + 0.5) / f - 0.5, x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0, P = (i, j) => lr[AI.clamp(j, 0, L - 1) * L + AI.clamp(i, 0, L - 1)]; bl[y * HR + x] = (1 - fy) * ((1 - fx) * P(x0, y0) + fx * P(x0 + 1, y0)) + fy * ((1 - fx) * P(x0, y0 + 1) + fx * P(x0 + 1, y0 + 1)); nn[y * HR + x] = lr[Math.floor(y / f) * L + Math.floor(x / f)]; } };
    const load = () => {
      if (!AI.cv) return setTimeout(load, 50);
      f = +q(root, '#f').value; hr = AI.cv.gray(AI.cv.resample(q(root, '#scene').value, HR)).map((v) => v / 255);
      const L = HR / f; lr = new Float32Array(L * L); for (let y = 0; y < HR; y++) for (let x = 0; x < HR; x++) lr[Math.floor(y / f) * L + Math.floor(x / f)] += hr[y * HR + x] / (f * f);
      bil(); reset(); pn = PSNR(nn, hr, 1); pb = PSNR(bl, hr, 1);
    };
    const reset = () => { K = new Float32Array(25); K[12] = 1; V = new Float32Array(25); b0 = 0; bv = 0; iter = 0; acc = 0; hist = []; out = new Float32Array(HR * HR); apply(); dirty = true; };
    const apply = () => { for (let y = 0; y < HR; y++) for (let x = 0; x < HR; x++) { let s = b0; for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) s += K[j * 5 + i] * bl[AI.clamp(y + j - 2, 0, HR - 1) * HR + AI.clamp(x + i - 2, 0, HR - 1)]; out[y * HR + x] = s; } };
    const train = () => {
      const gk = new Float32Array(25); let gb = 0;
      for (let y = 0; y < HR; y++) for (let x = 0; x < HR; x++) { const e = (2 * (out[y * HR + x] - hr[y * HR + x])) / (HR * HR); gb += e; for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) gk[j * 5 + i] += e * bl[AI.clamp(y + j - 2, 0, HR - 1) * HR + AI.clamp(x + i - 2, 0, HR - 1)]; }
      for (let i = 0; i < 25; i++) { V[i] = 0.9 * V[i] - 0.18 * gk[i]; K[i] += V[i]; } bv = 0.9 * bv - 0.18 * gb; b0 += bv;
      apply(); iter++; if (iter % 5 === 0) hist.push(PSNR(out, hr, 1).psnr); dirty = true;
    };
    q(root, '#scene').onchange = load; q(root, '#f').onchange = load; q(root, '#err').onchange = () => (dirty = true);
    const S = ui(root, 1, { step: () => { for (let i = 0; i < 5; i++) train(); }, reset });
    load();
    const grad = (a) => { let s = 0; for (let y = 0; y < HR - 1; y++) for (let x = 0; x < HR - 1; x++) s += Math.abs(a[y * HR + x + 1] - a[y * HR + x]) + Math.abs(a[(y + 1) * HR + x] - a[y * HR + x]); return s / (HR * HR); };
    AI.loop(root, (dt) => {
      if (!hr) return;
      if (S.playing) { acc += dt * 6 * S.speed(); const k = Math.floor(acc); acc -= k; for (let i = 0; i < Math.min(k, 8); i++) { for (let m = 0; m < 10; m++) train(); } }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w; g.clearRect(0, 0, W, c.h);
      const ts2 = Math.min(150, (W - 50) / 4), gap = (W - 4 * ts2) / 5, y = 28, showErr = q(root, '#err').checked;
      const po = PSNR(out, hr, 1);
      const arr = [[nn, 'nearest (low-res blocks)', pn], [bl, 'bilinear (smooth)', pb], [out, 'learned 5×5 filter', po], [hr, 'true high-res', null]];
      arr.forEach(([a, tx, p], i) => {
        const xx = gap + i * (ts2 + gap);
        if (showErr && p) { const e = new Float32Array(HR * HR); for (let j = 0; j < e.length; j++) e[j] = Math.abs(a[j] - hr[j]) * 4; tl[i].gray(e, 0, 1); } else tl[i].gray(a, 0, 1);
        tl[i].draw(g, xx, y, ts2); frame(g, xx, y, ts2, ts2, i === 2 ? C.orange : C.border, i === 2 ? 3 : 1);
        label(g, tx, xx + ts2 / 2, y - 8, C.fg, 'center'); if (p) label(g, `PSNR ${AI.fmt(p.psnr, 3)} dB`, xx + ts2 / 2, y + ts2 + 16, C.muted, 'center', '11px ui-monospace,monospace');
      });
      // kernel
      const ky = y + ts2 + 46, ks = Math.min(34, (c.h - ky - 10) / 5), kx = 24; let mx = 0.01; for (let i = 0; i < 25; i++) mx = Math.max(mx, Math.abs(K[i]));
      label(g, 'learned 5×5 filter', kx, ky - 6, C.fg);
      for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) { const v = K[j * 5 + i]; g.fillStyle = v >= 0 ? AI.mix(C.bg2, C.green, Math.min(1, v / mx)) : AI.mix(C.bg2, C.orange, Math.min(1, -v / mx)); g.fillRect(kx + i * ks, ky + j * ks, ks, ks); g.strokeStyle = C.border; g.strokeRect(kx + i * ks + 0.5, ky + j * ks + 0.5, ks, ks); if (ks > 26) label(g, AI.fmt(v, 2), kx + i * ks + ks / 2, ky + j * ks + ks / 2 + 4, C.fg, 'center', '10px ui-monospace,monospace'); }
      // curve
      const cx = kx + 5 * ks + 40, cw = W - cx - 24, ch = 5 * ks;
      g.fillStyle = C.bg2; g.fillRect(cx, ky, cw, ch); frame(g, cx, ky, cw, ch, C.border); label(g, 'PSNR while training (dB)', cx, ky - 6, C.fg);
      if (hist.length > 1) { const lo = Math.min(pb.psnr - 1, ...hist), hi = Math.max(...hist, pb.psnr + 1); g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => (i ? g.lineTo : g.moveTo).call(g, cx + (i / Math.max(hist.length - 1, 1)) * cw * Math.min(1, hist.length / 60) + 0, ky + ch - ((v - lo) / (hi - lo)) * ch)); g.stroke(); g.setLineDash([4, 4]); g.strokeStyle = C.muted; const yb = ky + ch - ((pb.psnr - lo) / (hi - lo)) * ch; g.beginPath(); g.moveTo(cx, yb); g.lineTo(cx + cw, yb); g.stroke(); g.setLineDash([]); label(g, 'bilinear', cx + cw - 4, yb - 3, C.muted, 'right', '10px sans-serif'); }
      q(root, '#out').innerHTML = `<span class="m">training steps</span> ${iter * 1}\n<span class="m">sharpness (mean |gradient|)</span> bilinear ${AI.fmt(grad(bl), 3)} → learned <b class="o">${AI.fmt(grad(out), 3)}</b> (truth ${AI.fmt(grad(hr), 3)})`;
      q(root, '#cap').textContent = iter < 5 ? `The 5×5 filter starts as "copy the centre" (= bilinear). Each training step nudges the 25 weights to reduce the squared error against the true image.` : `Filter now has a bright centre with negative surroundings: it boosts the difference between a pixel and its neighbours, i.e. it re-adds edge contrast that the downscaling lost. Real SR nets do this with millions of weights trained on thousands of images; this demo trains on one image, so it flatters itself.`;
    }, { fps: 30 });
  };

  /* ======================= NeRF ray marching ======================= */
  AI.labs.nerfLab = function (root) {
    const COLS = [[0.9, 0.25, 0.2], [0.2, 0.7, 0.3], [0.25, 0.4, 0.9]], BG = [0.55, 0.65, 0.8], NEAR = 2, FAR = 6;
    const { view, ctl } = AI.labShell(root, 'NeRF: volume rendering along one ray', 'samples march from the camera; transmittance T falls as the ray meets matter',
      '<canvas style="height:380px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out"></div>',
      PLAY + `<label>Samples N: <b id="nv"></b><input type="range" id="N" min="4" max="24" step="1" value="8"></label>
      ${[0, 1, 2].map((i) => `<label>blob ${i + 1} density σ: <b id="d${i}v"></b><input type="range" id="d${i}" min="0" max="12" step="0.25" value="${[2, 3, 2.5][i]}"></label><label>blob ${i + 1} position t: <b id="p${i}v"></b><input type="range" id="p${i}" min="2.3" max="5.7" step="0.05" value="${[2.9, 3.9, 5][i]}"></label>`).join('')}
      <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="jit"> random sample offsets (stratified)</label>`);
    const c = AI.canvas(AI.$('canvas', view)), WD = 0.28;
    let pos = 0, acc = 0, dirty = true, rows = [], jr = AI.rng(1), offs = [];
    const dens = (t) => { let s = 0, cr = [0, 0, 0]; for (let i = 0; i < 3; i++) { const d = +q(root, '#d' + i).value * Math.exp(-((t - +q(root, '#p' + i).value) ** 2) / (2 * WD * WD)); s += d; for (let k = 0; k < 3; k++) cr[k] += d * COLS[i][k]; } return { s, c: s > 1e-6 ? cr.map((v) => v / s) : [0.5, 0.5, 0.5] }; };
    const compute = () => {
      const N = +q(root, '#N').value, dl = (FAR - NEAR) / N; q(root, '#nv').textContent = N;
      for (let i = 0; i < 3; i++) { q(root, '#d' + i + 'v').textContent = q(root, '#d' + i).value; q(root, '#p' + i + 'v').textContent = q(root, '#p' + i).value; }
      if (offs.length !== N) offs = Array.from({ length: N }, () => jr());
      rows = []; let T = 1, cum = [0, 0, 0];
      for (let i = 0; i < N; i++) {
        const t = NEAR + (i + (q(root, '#jit').checked ? offs[i] : 0.5)) * dl, d = dens(t), al = 1 - Math.exp(-d.s * dl), w = T * al;
        for (let k = 0; k < 3; k++) cum[k] += w * d.c[k];
        rows.push({ t, sig: d.s, col: d.c, al, T, w, cum: cum.slice(), dl }); T *= 1 - al;
      }
      rows.Tend = T; rows.final = cum.map((v, k) => v + T * BG[k]); dirty = true;
    };
    ctl.addEventListener('input', () => { compute(); });
    const S = ui(root, 0.8, { step: () => { pos = Math.min(rows.length, Math.floor(pos) + 1); }, reset: () => { pos = 0; } });
    compute();
    AI.loop(root, (dt) => {
      if (S.playing) { pos += dt * 1.2 * S.speed(); if (pos > rows.length + 2.5) pos = 0; }
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h; g.clearRect(0, 0, W, Hh);
      const N = rows.length, cur = Math.min(N, Math.floor(pos)), x0 = 66, x1 = W - 110, X = (t) => x0 + ((t - NEAR) / (FAR - NEAR)) * (x1 - x0), rayY = 62;
      const rgb = (a, al) => `rgba(${Math.round(a[0] * 255)},${Math.round(a[1] * 255)},${Math.round(a[2] * 255)},${al === undefined ? 1 : al})`;
      // scene strip
      g.fillStyle = C.bg2; g.fillRect(x0, 22, x1 - x0, 80); frame(g, x0, 22, x1 - x0, 80, C.border);
      for (let i = 0; i < 3; i++) { const sg = +q(root, '#d' + i).value, px = X(+q(root, '#p' + i).value), rad = WD * 2.2 * ((x1 - x0) / (FAR - NEAR)); const gr = g.createRadialGradient(px, 62, 2, px, 62, rad); gr.addColorStop(0, rgb(COLS[i], Math.min(0.95, sg / 6))); gr.addColorStop(1, rgb(COLS[i], 0)); g.fillStyle = gr; g.fillRect(px - rad, 22, 2 * rad, 80); }
      g.strokeStyle = C.fg; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x0 - 30, rayY); g.lineTo(x1, rayY); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.beginPath(); g.arc(x0 - 36, rayY, 6, 0, 7); g.fill(); label(g, 'camera', x0 - 40, rayY - 12, C.muted, 'center', '10px sans-serif');
      rows.forEach((r, i) => { g.fillStyle = i < cur ? rgb(r.col) : C.muted; g.beginPath(); g.arc(X(r.t), rayY, i === cur ? 6 : 3.5, 0, 7); g.fill(); if (i === cur) { g.strokeStyle = C.fg; g.lineWidth = 2; g.stroke(); } });
      label(g, 'ray through the scene (3 blobs of coloured fog); dots = sample points', x0, 16, C.muted);
      // charts
      const cy0 = 128, ch = 78; const chart = (y, title, f, mx, col, stepMode) => {
        g.fillStyle = C.bg2; g.fillRect(x0, y, x1 - x0, ch); frame(g, x0, y, x1 - x0, ch, C.border); label(g, title, x0 + 4, y + 12, C.muted, 'left', '10px sans-serif');
        return (v) => y + ch - 6 - (v / mx) * (ch - 24);
      };
      const mxS = Math.max(6, ...Array.from({ length: 40 }, (_, i) => dens(NEAR + (i / 39) * (FAR - NEAR)).s)) * 1.05;
      const YS = chart(cy0, 'density σ(t) with samples', 0, mxS);
      g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 120; i++) { const t = NEAR + (i / 120) * (FAR - NEAR); (i ? g.lineTo : g.moveTo).call(g, X(t), YS(dens(t).s)); } g.stroke();
      rows.forEach((r, i) => { g.strokeStyle = i < cur ? C.fg : C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(X(r.t), YS(0)); g.lineTo(X(r.t), YS(r.sig)); g.stroke(); });
      const YT = chart(cy0 + ch + 8, 'transmittance T (light that survives to each sample)', 0, 1.05);
      rows.forEach((r, i) => { const bw = ((x1 - x0) / N) * 0.8, bx = X(r.t) - bw / 2; g.fillStyle = i < cur ? C.blue : C.border; g.globalAlpha = i < cur ? 0.85 : 0.5; g.fillRect(bx, YT(r.T), bw, YT(0) - YT(r.T)); g.globalAlpha = 1; });
      const YW = chart(cy0 + 2 * (ch + 8), 'weight w = T·α (how much each sample contributes), coloured by its colour', 0, 1.05);
      rows.forEach((r, i) => { const bw = ((x1 - x0) / N) * 0.8, bx = X(r.t) - bw / 2; g.fillStyle = i < cur ? rgb(r.col) : C.border; g.fillRect(bx, YW(Math.min(1, r.w)), bw, YW(0) - YW(Math.min(1, r.w))); });
      // pixel swatch
      const run = cur > 0 ? rows[cur - 1].cum : [0, 0, 0], Tcur = cur > 0 ? rows[cur - 1].T * (1 - rows[cur - 1].al) : 1, sx = x1 + 26;
      label(g, 'pixel so far', sx + 28, cy0 + 10, C.muted, 'center', '10px sans-serif');
      g.fillStyle = rgb(run.map((v, k) => v + Tcur * BG[k])); g.fillRect(sx, cy0 + 16, 56, 56); frame(g, sx, cy0 + 16, 56, 56, C.fg, 2);
      label(g, 'final', sx + 28, cy0 + 92, C.muted, 'center', '10px sans-serif'); g.fillStyle = rgb(rows.final); g.fillRect(sx, cy0 + 98, 56, 56); frame(g, sx, cy0 + 98, 56, 56, C.border, 1);
      const r = rows[Math.min(cur, N - 1)];
      q(root, '#out').innerHTML = `<span class="m">sample ${Math.min(cur + 1, N)} of ${N}</span>  t = ${AI.fmt(r.t, 3)}  δ = ${AI.fmt(r.dl, 3)}  σ = <b class="o">${AI.fmt(r.sig, 3)}</b>\nα = 1−e^(−σδ) = ${AI.fmt(r.al, 4)}   T = ${AI.fmt(r.T, 4)}   w = T·α = <b class="g">${AI.fmt(r.w, 4)}</b>\nsum of weights so far ${AI.fmt(rows.slice(0, Math.min(cur + 1, N)).reduce((a, b) => a + b.w, 0), 3)}   light left at the end ${AI.fmt(rows.Tend, 4)}`;
      q(root, '#cap').textContent = cur >= N ? `Ray finished: pixel colour = Σ w·c + (leftover light ${AI.fmt(rows.Tend, 3)}) × background. Opaque early blobs leave little light for the ones behind.` : `Sample ${cur + 1}: network gives density σ and colour; α is the share of light this slab blocks; T is how much light even reached it; the product w adds its colour to the pixel.`;
    }, { fps: 30 });
  };

  /* ======================= video ======================= */
  AI.labs.videoLab = function (root) {
    const W = 14, TT = 8, OW = 12, OT = 6, NPOS = OW * OW * OT;
    const PRE = {
      'moving right': [[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]].map((a) => a.map((v) => v / 3)),
      'moving left': [[0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]].map((a) => a.map((v) => v / 3)),
      'temporal difference': [new Array(9).fill(-1 / 9), new Array(9).fill(0), new Array(9).fill(1 / 9)],
      'static blur (no time)': [new Array(9).fill(0), new Array(9).fill(1 / 9), new Array(9).fill(0)],
    };
    const r = AI.rng(11), frames = [];
    for (let t = 0; t < TT; t++) { const f = new Float32Array(W * W); for (let i = 0; i < f.length; i++) f[i] = 0.08 + 0.04 * r(); for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) f[(5 + y) * W + 1 + t + x] = 1; for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) f[(1 + y) * W + 10 + x] = 0.9; frames.push(f); }
    const { view, ctl } = AI.labShell(root, 'Video: a 3-D kernel over space and time', 'the square moves right; the dot at top right is static',
      '<canvas style="height:300px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out"></div>',
      PLAY + '<div>Kernel presets <div class="row" id="pre"></div></div><div>Kernel weights (3 frames × 3×3; edit them)<div id="kg"></div></div><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="flow" checked> show motion arrows (block matching)</label>');
    const c = AI.canvas(AI.$('canvas', view));
    let K = PRE['moving right'].map((a) => a.slice()), O = new Float32Array(NPOS), pos = 0, acc = 0, inputs = [];
    const flows = []; for (let t = 0; t < TT - 1; t++) { const fl = []; for (let by = 0; by < W; by += 2) for (let bx = 0; bx < W; bx += 2) { let best = 1e9, bdx = 0, bdy = 0, base = 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { let s = 0; for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const x = AI.clamp(bx + i + dx, 0, W - 1), y = AI.clamp(by + j + dy, 0, W - 1); s += (frames[t][(by + j) * W + bx + i] - frames[t + 1][y * W + x]) ** 2; } if (dx === 0 && dy === 0) base = s; if (s < best - 1e-9 || (Math.abs(s - best) < 1e-9 && Math.abs(dx) + Math.abs(dy) < Math.abs(bdx) + Math.abs(bdy))) { best = s; bdx = dx; bdy = dy; } } if ((bdx || bdy) && base - best > 0.3) fl.push({ x: bx + 1, y: by + 1, dx: bdx, dy: bdy }); } flows.push(fl); }
    const kg = q(root, '#kg'); kg.style.cssText = 'display:grid;grid-template-columns:1fr;gap:4px';
    for (let t = 0; t < 3; t++) { const d = document.createElement('div'); d.innerHTML = `<div style="font-size:10px;color:var(--muted)">frame t+${t}</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px">${Array.from({ length: 9 }, (_, i) => `<input type="number" step="0.1" data-t="${t}" data-i="${i}" style="width:100%;min-width:0;padding:2px 4px;font-size:11px">`).join('')}</div>`; kg.appendChild(d); inputs.push(...AI.$$('input', d)); }
    const setK = (k) => { K = k.map((a) => a.slice()); inputs.forEach((inp) => (inp.value = AI.fmt(K[+inp.dataset.t][+inp.dataset.i], 3))); recompute(); pos = 0; };
    inputs.forEach((inp) => inp.addEventListener('input', () => { K[+inp.dataset.t][+inp.dataset.i] = +inp.value || 0; recompute(); }));
    Object.keys(PRE).forEach((nm) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = nm; b.onclick = () => setK(PRE[nm]); q(root, '#pre').appendChild(b); });
    function recompute() { for (let t = 0; t < OT; t++) for (let y = 0; y < OW; y++) for (let x = 0; x < OW; x++) { let s = 0; for (let a = 0; a < 3; a++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s += K[a][j * 3 + i] * frames[t + a][(y + j) * W + x + i]; O[(t * OW + y) * OW + x] = s; } }
    const S = ui(root, 0.8, { step: () => { pos = (Math.floor(pos) + 1) % (NPOS + 1); }, reset: () => { pos = 0; } });
    setK(K);
    AI.loop(root, (dt) => {
      if (S.playing) { acc += dt * 30 * S.speed(); pos += Math.floor(acc); acc -= Math.floor(acc); if (pos > NPOS + 20) pos = 0; }
      const C = AI.col(), g = c.ctx, w = c.w; g.clearRect(0, 0, w, c.h);
      const cs = Math.min(11, w / 56), p = Math.min(NPOS - 1, Math.floor(pos)), t = Math.floor(p / (OW * OW)), rr = p % (OW * OW), oy = Math.floor(rr / OW), ox = rr % OW;
      const sx = 16, sy = 36 + 7 * cs * 0.8, dxs = cs * 1.15, dys = -cs * 0.85;
      label(g, `video: ${TT} frames of ${W}×${W}`, sx, 16, C.muted);
      for (let f = TT - 1; f >= 0; f--) {
        const X0 = sx + f * dxs, Y0 = sy + f * dys + 3 * cs;
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const v = Math.round(frames[f][y * W + x] * 255); g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(X0 + x * cs, Y0 + y * cs, cs, cs); }
        const inWin = f >= t && f < t + 3; frame(g, X0, Y0, W * cs, W * cs, inWin ? C.blue : C.border, inWin ? 2 : 1);
        if (inWin) { g.fillStyle = C.blue; g.globalAlpha = 0.35; g.fillRect(X0 + ox * cs, Y0 + oy * cs, 3 * cs, 3 * cs); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 2; g.strokeRect(X0 + ox * cs, Y0 + oy * cs, 3 * cs, 3 * cs); }
        if (f === t) label(g, 't = ' + f, X0 - 2, Y0 + W * cs + 11, C.blue, 'left', '10px sans-serif');
      }
      // output
      const ox0 = sx + (TT - 1) * dxs + W * cs + 28, oy0 = 36 + 4 * cs; let mx = 0.2; for (let i = 0; i < NPOS; i++) mx = Math.max(mx, Math.abs(O[i]));
      label(g, `output slice t = ${t} (${OW}×${OW})`, ox0, 16 + 0, C.muted);
      for (let y = 0; y < OW; y++) for (let x = 0; x < OW; x++) { const k = (t * OW + y) * OW + x, rev = k <= p; if (rev) { const v = O[k] / mx; g.fillStyle = v >= 0 ? `rgb(${Math.round(40 + 200 * v)},${Math.round(40 + 90 * v)},40)` : `rgb(40,${Math.round(40 + 80 * -v)},${Math.round(40 + 200 * -v)})`; g.fillRect(ox0 + x * cs, oy0 + y * cs, cs, cs); } else { g.strokeStyle = C.border; g.strokeRect(ox0 + x * cs + 0.5, oy0 + y * cs + 0.5, cs, cs); } }
      frame(g, ox0, oy0, OW * cs, OW * cs, C.border); g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(ox0 + ox * cs, oy0 + oy * cs, cs, cs);
      AI.arrow(g, sx + (TT - 1) * dxs + W * cs + 2, oy0 + 6 * cs, ox0 - 4, oy0 + 6 * cs, C.muted, 2);
      // flow
      const fx = ox0 + OW * cs + 24; if (fx + W * cs < w + 6 && q(root, '#flow').checked) {
        const ft = Math.min(TT - 2, t + 1), fy = oy0; label(g, `motion t=${ft}→${ft + 1}`, fx, 16, C.muted);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { const v = Math.round(frames[ft][y * W + x] * 255); g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(fx + x * cs, fy + y * cs, cs, cs); }
        flows[ft].forEach((a) => AI.arrow(g, fx + a.x * cs, fy + a.y * cs, fx + (a.x + a.dx * 1.6) * cs, fy + (a.y + a.dy * 1.6) * cs, C.orange, 2));
      }
      let s = [0, 0, 0]; for (let a = 0; a < 3; a++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s[a] += K[a][j * 3 + i] * frames[t + a][(oy + j) * W + ox + i];
      q(root, '#out').innerHTML = `<span class="m">output (x=${ox}, y=${oy}, t=${t})</span>  = frame t: ${AI.fmt(s[0], 3)}  + frame t+1: ${AI.fmt(s[1], 3)}  + frame t+2: ${AI.fmt(s[2], 3)}  =  <b class="o">${AI.fmt(s[0] + s[1] + s[2], 3)}</b>`;
      q(root, '#cap').textContent = `Position ${p + 1} of ${NPOS}: the 3×3×3 cube multiplies 27 pixels (3×3 in each of three consecutive frames) by 27 weights and adds them: one output number. Scan x, then y, then move on in time. Bright output = the pattern encoded by the weights is present here.`;
    }, { fps: 30 });
  };

  /* ======================= tracking ======================= */
  function hungarian(cost) {
    const n = cost.length, m = n ? cost[0].length : 0; if (!n || !m) return [];
    if (n > m) { const tr = Array.from({ length: m }, (_, j) => cost.map((row) => row[j])), a = hungarian(tr), res = new Array(n).fill(-1); a.forEach((i, j) => { if (i >= 0) res[i] = j; }); return res; }
    const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
    for (let i = 1; i <= n; i++) {
      p[0] = i; let j0 = 0; const minv = new Array(m + 1).fill(Infinity), used = new Array(m + 1).fill(false);
      do {
        used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
        for (let j = 1; j <= m; j++) if (!used[j]) { const cur = cost[i0 - 1][j - 1] - u[i0] - v[j]; if (cur < minv[j]) { minv[j] = cur; way[j] = j0; } if (minv[j] < delta) { delta = minv[j]; j1 = j; } }
        for (let j = 0; j <= m; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const ans = new Array(n).fill(-1); for (let j = 1; j <= m; j++) if (p[j]) ans[p[j] - 1] = j - 1; return ans;
  }
  AI.hungarian = AI.hungarian || hungarian;
  AI.labs.trackLab = function (root) {
    const WW = 320, WH = 220, BS = 28;
    const { view, ctl } = AI.labShell(root, 'Tracking: Kalman filter + IoU + Hungarian', 'each frame: ① move ② predict ③ detect ④ associate ⑤ update',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out"></div>',
      PLAY + `<label>Scenario <select id="scn"><option value="b">3 bouncing objects</option><option value="x">two objects crossing</option></select></label>
      <label>Measurement noise σ: <b id="nv"></b> px<input type="range" id="nz" min="0" max="10" step="0.5" value="4"></label>
      <label>Process noise q: <b id="qv"></b><input type="range" id="qn" min="0.01" max="4" step="0.01" value="0.3"></label>
      <label>Missed-detection probability: <b id="mv"></b><input type="range" id="pm" min="0" max="0.6" step="0.05" value="0.1"></label>
      <label>False detections per frame: <b id="fv"></b><input type="range" id="pf" min="0" max="0.8" step="0.05" value="0.1"></label>
      <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="greedy"> greedy matching instead of Hungarian</label>`);
    const c = AI.canvas(AI.$('canvas', view));
    const COLS = ['blue', 'green', 'orange', 'purple', 'pink', 'teal', 'red', 'yellow'];
    let rng, objs, tracks, dets, frameNo, phase, ph, nextId, matches, unmatchedDet, switches, hit, truthPrev, S2, nextObj;
    const sl = (id) => +q(root, '#' + id).value;
    const reset = () => {
      rng = AI.rng(21); frameNo = 0; phase = 0; ph = 0; nextId = 1; tracks = []; dets = []; matches = []; switches = 0; unmatchedDet = [];
      const sc = q(root, '#scn').value;
      objs = sc === 'x' ? [{ x: 40, y: 60, vx: 11, vy: 6.5, s: 28, id: 0 }, { x: 280, y: 160, vx: -11, vy: -6.5, s: 28, id: 1 }] : [{ x: 60, y: 60, vx: 8, vy: 3, s: 30, id: 0 }, { x: 250, y: 70, vx: -6, vy: 6, s: 26, id: 1 }, { x: 150, y: 170, vx: 5, vy: -7, s: 28, id: 2 }];
      objs.forEach((o) => { o.px = o.x; o.py = o.y; });
      ['nz', 'qn', 'pm', 'pf'].forEach((id) => (q(root, '#' + { nz: 'nv', qn: 'qv', pm: 'mv', pf: 'fv' }[id]).textContent = q(root, '#' + id).value));
    };
    ctl.addEventListener('input', () => ['nz', 'qn', 'pm', 'pf'].forEach((id) => (q(root, '#' + { nz: 'nv', qn: 'qv', pm: 'mv', pf: 'fv' }[id]).textContent = q(root, '#' + id).value)));
    const iou = (a, b) => { const ax0 = a.x - a.s / 2, ax1 = a.x + a.s / 2, ay0 = a.y - a.s / 2, ay1 = a.y + a.s / 2, bx0 = b.x - b.s / 2, bx1 = b.x + b.s / 2, by0 = b.y - b.s / 2, by1 = b.y + b.s / 2; const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)), ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0)), I = iw * ih, U = a.s * a.s + b.s * b.s - I; return U > 0 ? I / U : 0; };
    // per-axis KF: state [p, v], P = [a b; b d]
    const kfPred = (P, q2) => { const a = P[0] + 2 * P[1] + P[3] + q2 / 4, b = P[1] + P[3] + q2 / 2, d = P[3] + q2; return [a, b, b, d]; };
    const step1 = () => { // move truth
      objs.forEach((o) => { o.px = o.x; o.py = o.y; o.vx += AI.randn(rng) * 0.6; o.vy += AI.randn(rng) * 0.6; o.x += o.vx; o.y += o.vy; if (o.x < 20 || o.x > WW - 20) { o.vx *= -1; o.x = AI.clamp(o.x, 20, WW - 20); } if (o.y < 20 || o.y > WH - 20) { o.vy *= -1; o.y = AI.clamp(o.y, 20, WH - 20); } });
      frameNo++;
    };
    const step2 = () => { // predict
      const qn = sl('qn'); tracks.forEach((t) => { t.oldPx = t.x; t.oldPy = t.y; t.oldPX = t.Px.slice(); t.oldPY = t.Py.slice(); t.x += t.vx; t.y += t.vy; t.Px = kfPred(t.Px, qn * 4); t.Py = kfPred(t.Py, qn * 4); });
    };
    const step3 = () => { // detect
      const nz = sl('nz'); dets = [];
      objs.forEach((o) => { if (rng() >= sl('pm')) dets.push({ x: o.x + AI.randn(rng) * nz, y: o.y + AI.randn(rng) * nz * 0.6, s: o.s * (1 + 0.08 * AI.randn(rng)), src: o.id }); });
      if (rng() < sl('pf')) dets.push({ x: 30 + rng() * (WW - 60), y: 30 + rng() * (WH - 60), s: 26, src: -1 });
    };
    const step4 = () => { // associate
      matches = []; unmatchedDet = [];
      const nT = tracks.length, nD = dets.length; const M = tracks.map((t) => dets.map((d) => iou({ x: t.x, y: t.y, s: t.s }, d)));
      let asg = new Array(nT).fill(-1);
      if (nT && nD) {
        if (q(root, '#greedy').checked) { const used = new Set(), pairs = []; M.forEach((row, i) => row.forEach((v, j) => pairs.push([v, i, j]))); pairs.sort((a, b) => b[0] - a[0]); const ti = new Set(); pairs.forEach(([v, i, j]) => { if (!ti.has(i) && !used.has(j)) { asg[i] = j; ti.add(i); used.add(j); } }); }
        else asg = hungarian(M.map((row) => row.map((v) => 1 - v)));
      }
      const usedD = new Set();
      tracks.forEach((t, i) => { const j = asg[i]; t.matched = null; if (j >= 0 && M[i][j] >= 0.2) { matches.push({ t, d: dets[j], iou: M[i][j] }); t.matched = dets[j]; usedD.add(j); } });
      dets.forEach((d, j) => { if (!usedD.has(j)) unmatchedDet.push(d); });
    };
    const step5 = () => { // update
      const nz = sl('nz');
      matches.forEach(({ t, d }) => {
        [['x', 'Px', 'vx', nz * nz], ['y', 'Py', 'vy', (nz * 0.6) ** 2]].forEach(([pk, Pk, vk, R]) => { const P = t[Pk], S = P[0] + Math.max(R, 0.25), K0 = P[0] / S, K1 = P[1] / S, y = d[pk] - t[pk]; t[pk] += K0 * y; t[vk] += K1 * y; t[Pk] = [(1 - K0) * P[0], (1 - K0) * P[1], (1 - K0) * P[1], P[3] - K1 * P[1]]; t.K = [K0, K1]; });
        t.hits++; t.miss = 0; t.s = 0.7 * t.s + 0.3 * d.s;
        if (d.src >= 0) { if (t.src >= 0 && t.src !== d.src) switches++; t.src = d.src; }
      });
      tracks.forEach((t) => { if (!t.matched) t.miss++; t.trail.push([t.x, t.y]); if (t.trail.length > 30) t.trail.shift(); });
      unmatchedDet.forEach((d) => { const R = Math.max(nz, 0.5); tracks.push({ id: nextId++, x: d.x, y: d.y, vx: 0, vy: 0, s: d.s, Px: [R * R, 0, 0, 25], Py: [(R * 0.6) ** 2, 0, 0, 25], hits: 1, miss: 0, src: d.src, trail: [[d.x, d.y]], oldPx: d.x, oldPy: d.y, oldPX: [R * R, 0, 0, 25], oldPY: [(R * 0.6) ** 2, 0, 0, 25] }); });
      tracks = tracks.filter((t) => (t.hits >= 3 ? t.miss <= 4 : t.miss <= 1));
    };
    const advance = (n) => { if (n === 0) step1(); else if (n === 1) { step2(); } else if (n === 2) step3(); else if (n === 3) step4(); else step5(); };
    const S = ui(root, 0.8, { step: () => { phase = (phase + 1) % 5; ph = 0; advance(phase); }, reset });
    q(root, '#scn').onchange = reset;
    reset();
    const PHN = ['① The world moves on: objects glide to their new positions.', '② PREDICT: each track moves its estimate forward with its velocity; the dashed ellipse (2σ) grows because we are less sure.', '③ DETECT: the detector reports noisy boxes (grey). Some objects are missed, some false boxes appear.', '④ ASSOCIATE: IoU between predicted boxes and detections, solved by the Hungarian algorithm; coloured lines show the matches.', '⑤ UPDATE: the Kalman gain blends prediction and detection; the ellipse shrinks. Unmatched detections start new (dashed) tracks.'];
    const DUR = 0.9;
    AI.loop(root, (dt) => {
      if (S.playing) { ph += (dt / DUR) * S.speed(); if (ph >= 1) { ph = 0; phase = (phase + 1) % 5; advance(phase); } }
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h, sc = Math.min(w / WW, h / WH), ox = (w - WW * sc) / 2, oy = (h - WH * sc) / 2, X = (v) => ox + v * sc, Y = (v) => oy + v * sc;
      g.clearRect(0, 0, w, h); g.fillStyle = C.bg2; g.fillRect(ox, oy, WW * sc, WH * sc); frame(g, ox, oy, WW * sc, WH * sc, C.border);
      const e = AI.smooth(ph);
      const box = (cx, cy, s, col, lw, dash) => { g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.strokeRect(X(cx - s / 2), Y(cy - s / 2), s * sc, s * sc); g.setLineDash([]); };
      const ell = (cx, cy, P1, P2, col, lw, dash, fillA) => { g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.beginPath(); g.ellipse(X(cx), Y(cy), Math.max(2, 2 * Math.sqrt(Math.max(P1, 0)) * sc), Math.max(2, 2 * Math.sqrt(Math.max(P2, 0)) * sc), 0, 0, 7); g.stroke(); g.setLineDash([]); if (fillA) { g.globalAlpha = fillA; g.fillStyle = col; g.fill(); g.globalAlpha = 1; } };
      // truth (glides during phase 0)
      objs.forEach((o) => { const k = phase === 0 ? e : 1, px = AI.lerp(o.px, o.x, k), py = AI.lerp(o.py, o.y, k); g.fillStyle = C.fg; g.globalAlpha = 0.25; g.fillRect(X(px - o.s / 2), Y(py - o.s / 2), o.s * sc, o.s * sc); g.globalAlpha = 1; });
      // tracks
      tracks.forEach((t) => {
        const col = C[COLS[(t.id - 1) % COLS.length]], conf = t.hits >= 3;
        g.strokeStyle = col; g.lineWidth = 1.5; g.globalAlpha = 0.5; g.beginPath(); t.trail.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(p[0]), Y(p[1]))); g.stroke(); g.globalAlpha = 1;
        const predicted = phase === 1 || phase === 2 || phase === 3 || (phase === 4 && ph < 1);
        if (phase === 1) { const px = AI.lerp(t.oldPx, t.x, e), py = AI.lerp(t.oldPy, t.y, e); ell(px, py, AI.lerp(t.oldPX[0], t.Px[0], e), AI.lerp(t.oldPY[0], t.Py[0], e), col, 2, [5, 4]); box(px, py, t.s, col, 2, [5, 4]); }
        else if (phase === 2 || phase === 3) { ell(t.x, t.y, t.Px[0], t.Py[0], col, 2, [5, 4]); box(t.x, t.y, t.s, col, 2, [5, 4]); }
        else if (phase === 4) { const m = matches.find((q3) => q3.t === t); const pr = { x: t.x, y: t.y }; ell(t.x, t.y, t.Px[0], t.Py[0], col, 2.5, conf ? [] : [3, 3], 0.08); box(t.x, t.y, t.s, col, 2.5, conf ? [] : [3, 3]); }
        else { ell(t.x, t.y, t.Px[0], t.Py[0], col, 2.5, conf ? [] : [3, 3], 0.08); box(t.x, t.y, t.s, col, 2.5, conf ? [] : [3, 3]); }
        g.fillStyle = col; g.beginPath(); g.arc(X(t.x), Y(t.y), 3, 0, 7); g.fill(); if (conf || phase === 4) AI.arrow(g, X(t.x), Y(t.y), X(t.x + t.vx * 1.5), Y(t.y + t.vy * 1.5), col, 1.5);
        label(g, '#' + t.id + (t.miss ? ' (missed ' + t.miss + ')' : ''), X(t.x - t.s / 2), Y(t.y - t.s / 2) - 4, col, 'left', 'bold 11px sans-serif');
      });
      // detections
      if (phase >= 2) { g.globalAlpha = phase === 2 ? e : 1; dets.forEach((d) => { box(d.x, d.y, d.s, C.muted, 2); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X(d.x) - 4, Y(d.y)); g.lineTo(X(d.x) + 4, Y(d.y)); g.moveTo(X(d.x), Y(d.y) - 4); g.lineTo(X(d.x), Y(d.y) + 4); g.stroke(); }); g.globalAlpha = 1; }
      if (phase === 3 || phase === 4) matches.forEach(({ t, d, iou: v }) => { const col = C[COLS[(t.id - 1) % COLS.length]], k = phase === 3 ? e : 1; g.strokeStyle = col; g.lineWidth = 3; g.beginPath(); g.moveTo(X(t.x), Y(t.y)); g.lineTo(AI.lerp(X(t.x), X(d.x), k), AI.lerp(Y(t.y), Y(d.y), k)); g.stroke(); if (k > 0.8) label(g, 'IoU ' + AI.fmt(v, 2), (X(t.x) + X(d.x)) / 2 + 4, (Y(t.y) + Y(d.y)) / 2 - 4, col, 'left', '10px ui-monospace,monospace'); });
      if (phase === 4) unmatchedDet.forEach((d) => label(g, 'new', X(d.x), Y(d.y) - d.s * sc / 2 - 4, C.fg, 'center', '10px sans-serif'));
      label(g, 'grey solid = ground truth blocks · dashed = prediction · grey boxes = detections', ox + 4, oy + WH * sc + 12 > h ? h - 2 : oy + WH * sc + 12, C.muted, 'left', '10px sans-serif');
      const t0 = tracks[0], conf = tracks.filter((t) => t.hits >= 3).length;
      q(root, '#out').innerHTML = `<span class="m">frame</span> ${frameNo}   <span class="m">tracks</span> ${conf} confirmed, ${tracks.length - conf} tentative   <span class="m">detections</span> ${dets.length}   <span class="m">matches</span> ${matches.length}   <span class="m">ID switches</span> <b class="o">${switches}</b>` + (t0 && t0.K ? `\n<span class="m">track #${t0.id}</span> Kalman gain (x): position ${AI.fmt(t0.K[0], 2)}, velocity ${AI.fmt(t0.K[1], 2)}   position std ${AI.fmt(Math.sqrt(Math.max(0, t0.Px[0])), 2)} px` : '');
      q(root, '#cap').textContent = 'Frame ' + frameNo + '. ' + PHN[phase];
    }, { fps: 30 });
  };
})();
