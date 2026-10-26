/* Computer-vision Module 2 labs: cvCorner, cvMatch, cvRansac, cvPinhole, cvStereo, cvFlow.
 * Requires assets/lib.js and assets/labs-cv.js (AI.cv scenes). All animation is dt-based, slow by default,
 * with a speed slider and pause/step, and redraws only when something changed. */
(function () {
  const CV = AI.cv;
  const q$ = (root, s) => AI.$(s, root);
  const PLAY = `<label>Speed <span class="v" id="spv">×1</span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Restart</button></div>`;

  /** Shared play/pause/step/speed wiring. st = {playing, dirty, speed()}. */
  function bindPlay(root, st, onStep, onReset) {
    const q = (s) => q$(root, s), sp = q('#sp');
    st.playing = true; st.dirty = true; st.speed = () => +sp.value;
    const lab = () => { q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play'; st.dirty = true; };
    q('#play').onclick = () => { st.playing = !st.playing; lab(); };
    q('#step').onclick = () => { st.playing = false; lab(); onStep(); };
    q('#reset').onclick = () => { onReset(); st.dirty = true; };
    sp.addEventListener('input', () => { q('#spv').textContent = '×' + (+sp.value).toFixed(2).replace(/\.?0+$/, ''); });
    root.addEventListener('input', () => (st.dirty = true));
    root.addEventListener('change', () => (st.dirty = true));
    window.addEventListener('themechange', () => (st.dirty = true));
  }
  const mkCanvas = (root, view, st) => { const c = AI.canvas(AI.$('canvas', view)); c.onresize = () => (st.dirty = true); return c; };
  const ptr = (cv, e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const T = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const grayCanvas = (n, fn) => { const c = document.createElement('canvas'); c.width = c.height = n; const g = c.getContext('2d'); const d = g.createImageData(n, n); for (let i = 0; i < n * n; i++) { const v = fn(i); d.data[i * 4] = v[0]; d.data[i * 4 + 1] = v[1]; d.data[i * 4 + 2] = v[2]; d.data[i * 4 + 3] = 255; } g.putImageData(d, 0, 0); return c; };

  /* ======================================================================= */
  /* cvCorner: sliding window, E(theta) shape, Harris heatmap                  */
  /* ======================================================================= */
  AI.labs.cvCorner = function (root) {
    const N = 60;
    const { view } = AI.labShell(root, 'Corner detector: how much does the window change?', 'slide a window, shift it in every direction, and turn the change into one number (Harris R)',
      '<canvas style="height:290px;touch-action:none;cursor:crosshair"></canvas><div class="caption" id="cap"></div>',
      `<label>Window size <span class="v" id="wv"></span><input type="range" id="w" min="3" max="9" step="2" value="5"></label>
       <label>Harris k <span class="v" id="kv"></span><input type="range" id="k" min="0.02" max="0.12" step="0.005" value="0.05"></label>
       <label>Corner threshold (% of max R) <span class="v" id="tv"></span><input type="range" id="th" min="1" max="60" value="15"></label>
       ${PLAY}<div class="readout" id="out"></div>`);
    const q = (s) => q$(root, s), st = {}, cv = AI.$('canvas', view), c = mkCanvas(root, view, st);
    const I = Float32Array.from(CV.gray(CV.resample('house', N)), (v) => v / 255);
    const Ix = new Float32Array(N * N), Iy = new Float32Array(N * N);
    for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) { const i = y * N + x; Ix[i] = (I[i + 1] - I[i - 1]) / 2; Iy[i] = (I[i + N] - I[i - N]) / 2; }
    const cl = (v, a, b) => Math.min(b, Math.max(a, v));
    const samp = (x, y) => { x = cl(x, 0, N - 1.001); y = cl(y, 0, N - 1.001); const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * N + x0; return I[i] * (1 - fx) * (1 - fy) + I[i + 1] * fx * (1 - fy) + I[i + N] * (1 - fx) * fy + I[i + N + 1] * fx * fy; };
    const tensor = (x, y, r) => { let a = 0, b = 0, d = 0; for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) { const k = cl(y + j, 0, N - 1) * N + cl(x + i, 0, N - 1); a += Ix[k] * Ix[k]; b += Ix[k] * Iy[k]; d += Iy[k] * Iy[k]; } return [a, b, d]; };
    const harris = (a, b, d, k) => { const tr = a + d, det = a * d - b * b, h = tr / 2, s = Math.sqrt(Math.max(0, h * h - det)); return { l1: h + s, l2: h - s, R: det - k * tr * tr, det, tr }; };
    const Eth = (x, y, r, th, s) => { const dx = s * Math.cos(th), dy = s * Math.sin(th); let e = 0; for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) { const d = samp(x + i + dx, y + j + dy) - samp(x + i, y + j); e += d * d; } return e; };
    const ic = grayCanvas(N, (i) => { const v = Math.round(I[i] * 255); return [v, v, v]; });
    let hc = null, Rm = new Float32Array(N * N), maxR = 1, maxL = 1, Escale = 1, dets = [];
    const NDIR = 24;
    function recompute() {
      const r = (+q('#w').value - 1) / 2, k = +q('#k').value, thr = +q('#th').value / 100;
      q('#wv').textContent = `${2 * r + 1}×${2 * r + 1}`; q('#kv').textContent = k.toFixed(3); q('#tv').textContent = (thr * 100).toFixed(0) + '%';
      maxR = 1e-9; maxL = 1e-9; let bx = 0, by = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const t = tensor(x, y, r), h = harris(t[0], t[1], t[2], k); Rm[y * N + x] = h.R; if (h.R > maxR) { maxR = h.R; bx = x; by = y; } if (h.l1 > maxL) maxL = h.l1; }
      dets = [];
      for (let y = 2; y < N - 2; y++) for (let x = 2; x < N - 2; x++) { const v = Rm[y * N + x]; if (v < thr * maxR) continue; let ok = true; for (let j = -2; j <= 2 && ok; j++) for (let i = -2; i <= 2; i++) if ((i || j) && Rm[(y + j) * N + x + i] > v) { ok = false; break; } if (ok) dets.push([x, y]); }
      hc = grayCanvas(N, (i) => { const v = Rm[i] / maxR; if (v >= 0) { const t = Math.sqrt(Math.min(1, v)); return [30 + 225 * t, 30 + 100 * t, 30]; } const t = Math.sqrt(Math.min(1, -v)); return [30, 30 + 90 * t, 30 + 200 * t]; });
      Escale = 1e-9; for (let a = 0; a < NDIR; a++) Escale = Math.max(Escale, Eth(bx, by, r, (a / NDIR) * 2 * Math.PI, 2));
      st.dirty = true;
    }
    ['#w', '#k', '#th'].forEach((s) => q(s).addEventListener('input', recompute));
    const WP = [[8, 8], [6, 42], [18, 29], [45, 35], [30, 11], [26, 33]];
    let pos = { x: WP[0][0], y: WP[0][1] }, tour = { i: 0, from: WP[0], to: WP[0], t: 0, phase: 'dwell' }, tourOn = true, ang = 0, lastCap = '';
    const nextWP = () => { tour.i = (tour.i + 1) % WP.length; tour.from = [pos.x, pos.y]; tour.to = WP[tour.i]; tour.t = 0; tour.phase = 'move'; tourOn = true; };
    bindPlay(root, st, nextWP, () => { tour.i = -1; pos = { x: WP[0][0], y: WP[0][1] }; tour.from = tour.to = WP[0]; tour.phase = 'dwell'; tour.t = 0; tourOn = true; ang = 0; });
    recompute();
    let layout = null;
    const place = (e) => {
      const [mx, my] = ptr(cv, e); if (!layout) return;
      for (const P of [layout.A, layout.B]) if (mx >= P.x && mx <= P.x + P.s && my >= P.y && my <= P.y + P.s) { pos = { x: cl(((mx - P.x) / P.s) * N, 0, N - 1), y: cl(((my - P.y) / P.s) * N, 0, N - 1) }; tourOn = false; st.dirty = true; }
    };
    let drag = false;
    cv.addEventListener('pointerdown', (e) => { drag = true; cv.setPointerCapture(e.pointerId); place(e); });
    cv.addEventListener('pointermove', (e) => drag && place(e));
    cv.addEventListener('pointerup', () => (drag = false));

    AI.loop(root, (dt) => {
      const sp = st.speed();
      if (st.playing) {
        ang += dt * 0.5 * sp;
        if (tourOn) {
          tour.t += dt * sp;
          if (tour.phase === 'move') { const u = tour.t / 2.4; pos = { x: AI.lerp(tour.from[0], tour.to[0], AI.smooth(u)), y: AI.lerp(tour.from[1], tour.to[1], AI.smooth(u)) }; if (u >= 1) { tour.phase = 'dwell'; tour.t = 0; } }
          else if (tour.t >= 6) nextWP();
        }
      }
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      const r = (+q('#w').value - 1) / 2, k = +q('#k').value, thr = +q('#th').value / 100;
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(), s = Math.min(H - 34, (W - 30) * 0.37), cell = s / N;
      const A = { x: 8, y: 24, s }, B = { x: 8 + s + 10, y: 24, s }; layout = { A, B };
      const ex0 = B.x + s + 10, ecx = ex0 + (W - ex0 - 8) / 2, ecy = 24 + s / 2, er = Math.max(10, Math.min((W - ex0 - 8) / 2, s / 2) - 4);
      g.clearRect(0, 0, W, H);
      T(g, 'image + window', A.x, 16, C.muted); T(g, 'Harris response R', B.x, 16, C.muted); T(g, 'change E(θ) per shift', ex0, 16, C.muted);
      g.imageSmoothingEnabled = false; g.drawImage(ic, A.x, A.y, s, s); g.drawImage(hc, B.x, B.y, s, s);
      const px = Math.round(pos.x), py = Math.round(pos.y);
      const t = tensor(px, py, r), h = harris(t[0], t[1], t[2], k);
      for (const [dx, dy] of dets) { for (const P of [A, B]) { g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath(); g.arc(P.x + (dx + 0.5) * cell, P.y + (dy + 0.5) * cell, 4, 0, 7); g.stroke(); } }
      for (const P of [A, B]) { g.strokeStyle = C.blue; g.lineWidth = 2.5; g.strokeRect(P.x + (px - r) * cell, P.y + (py - r) * cell, (2 * r + 1) * cell, (2 * r + 1) * cell); }
      // shifted ghost window along probe direction
      const sh = 2, gx = Math.cos(ang) * sh * cell, gy = Math.sin(ang) * sh * cell;
      g.save(); g.setLineDash([4, 3]); g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(A.x + (px - r) * cell + gx, A.y + (py - r) * cell + gy, (2 * r + 1) * cell, (2 * r + 1) * cell); g.restore();
      AI.arrow(g, A.x + (px + 0.5) * cell, A.y + (py + 0.5) * cell, A.x + (px + 0.5) * cell + gx * 1.4, A.y + (py + 0.5) * cell + gy * 1.4, C.orange, 2);
      // E(theta) polygon
      g.strokeStyle = C.border; g.lineWidth = 1; for (const f of [0.5, 1]) { g.beginPath(); g.arc(ecx, ecy, er * f, 0, 7); g.stroke(); }
      g.beginPath(); g.moveTo(ecx - er, ecy); g.lineTo(ecx + er, ecy); g.moveTo(ecx, ecy - er); g.lineTo(ecx, ecy + er); g.stroke();
      g.beginPath();
      for (let a = 0; a <= NDIR; a++) { const th = (a / NDIR) * 2 * Math.PI, e = Math.min(1.15, Eth(px, py, r, th, sh) / Escale), rr = er * e; const X = ecx + rr * Math.cos(th), Y = ecy + rr * Math.sin(th); a ? g.lineTo(X, Y) : g.moveTo(X, Y); }
      g.closePath(); g.globalAlpha = 0.35; g.fillStyle = C.green; g.fill(); g.globalAlpha = 1; g.strokeStyle = C.green; g.lineWidth = 2; g.stroke();
      const ec = Math.min(1.15, Eth(px, py, r, ang, sh) / Escale);
      g.fillStyle = C.orange; g.beginPath(); g.arc(ecx + er * ec * Math.cos(ang), ecy + er * ec * Math.sin(ang), 4, 0, 7); g.fill();
      T(g, 'ring = change of a strong corner', ex0, 24 + s + 14, C.muted, '10px sans-serif');
      // verdict
      let kind, txt;
      if (h.l1 < 0.03 * maxL) { kind = 'FLAT'; txt = 'Flat: both eigenvalues are tiny, so shifting the window in any direction changes almost nothing. Nothing to lock onto.'; }
      else if (h.R > thr * maxR) { kind = 'CORNER'; txt = 'Corner: shifting in every direction changes the window (both λ large, R large and positive). A reliable landmark.'; }
      else if (h.R < 0) { kind = 'EDGE'; txt = 'Edge: big change across the edge, almost none along it (one λ large, one small, R negative). Position is known in one direction only.'; }
      else { kind = 'WEAK'; txt = 'Weak structure: some change, but R is below the corner threshold.'; }
      if (txt !== lastCap) { q('#cap').textContent = txt; lastCap = txt; }
      q('#out').innerHTML = `window at (${px}, ${py}) → <b class="${kind === 'CORNER' ? 'g' : kind === 'EDGE' ? 'o' : 'm'}">${kind}</b>\nΣIx²=${AI.fmt(t[0], 3)}  ΣIxIy=${AI.fmt(t[1], 3)}  ΣIy²=${AI.fmt(t[2], 3)}\nλ₁=${AI.fmt(h.l1, 3)}  λ₂=${AI.fmt(h.l2, 3)}\ndet=${AI.fmt(h.det, 4)}  tr=${AI.fmt(h.tr, 3)}\nR = det − k·tr² = <b>${AI.fmt(h.R, 4)}</b>\n<span class="m">detected corners: ${dets.length} (white rings)</span>`;
    }, { fps: 30 });
  };

  /* ======================================================================= */
  /* cvMatch: descriptors, nearest neighbour, ratio test                       */
  /* ======================================================================= */
  AI.labs.cvMatch = function (root) {
    const KP = [[60, 22], [24, 52], [96, 52], [36, 58], [48, 58], [36, 70], [48, 70], [72, 58], [84, 58], [72, 70], [84, 70], [52, 66], [66, 66], [52, 92], [66, 92], [30, 92], [90, 92], [104, 18]];
    const NAMES = ['roof apex', 'roof left', 'roof right', 'window-1 TL', 'window-1 TR', 'window-1 BL', 'window-1 BR', 'window-2 TL', 'window-2 TR', 'window-2 BL', 'window-2 BR', 'door TL', 'door TR', 'door BL', 'door BR', 'wall left', 'wall right', 'sun'];
    const TWIN = { 7: 3, 8: 4, 9: 5, 10: 6 }, MISSED = new Set([12, 17]), D = 8, n = KP.length;
    const { view } = AI.labShell(root, 'Matching descriptors between two views', 'nearest neighbour in descriptor space, then the Lowe ratio test throws out ambiguous matches',
      '<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>',
      `<label>Descriptor noise (view change) <span class="v" id="nv"></span><input type="range" id="noise" min="0" max="0.5" step="0.01" value="0.08"></label>
       <label>Distance threshold τ <span class="v" id="tv"></span><input type="range" id="tau" min="0.1" max="2.5" step="0.05" value="1.2"></label>
       <label>Ratio ρ = d₁/d₂ must be below <span class="v" id="rv"></span><input type="range" id="rho" min="0.3" max="1" step="0.01" value="0.8"></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="ron" checked> use ratio test</label>
       ${PLAY}<div class="row"><button class="btn" id="all">Match all now</button></div><div class="readout" id="out"></div>`);
    const q = (s) => q$(root, s), st = {}, c = mkCanvas(root, view, st);
    const r = AI.rng(11), base = KP.map(() => Array.from({ length: D }, () => r()));
    for (const [i, j] of Object.entries(TWIN)) base[i] = base[j].map((v) => v + 0.07 * AI.randn(r));
    const zB = KP.map(() => Array.from({ length: D }, () => AI.randn(r)));
    const clutter = [[20, 20], [100, 100], [14, 96], [96, 90]].map((p) => ({ p, d: Array.from({ length: D }, () => r()), id: -1 }));
    const img = (() => { const cv = document.createElement('canvas'); cv.width = cv.height = 120; cv.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(CV.scene('house')), 120, 120), 0, 0); return cv; })();
    const ROT = 0.38, SC = 0.82, tB = (p) => { const x = p[0] - 60, y = p[1] - 60, c0 = Math.cos(ROT), s0 = Math.sin(ROT); return [SC * (c0 * x - s0 * y), SC * (s0 * x + c0 * y)]; };
    let Bk = [], M = [], pos = 0;
    function build() {
      const noise = +q('#noise').value; q('#nv').textContent = noise.toFixed(2);
      Bk = [];
      KP.forEach((p, i) => { if (!MISSED.has(i)) Bk.push({ id: i, p, d: base[i].map((v, k) => v + noise * zB[i][k]) }); });
      clutter.forEach((cl) => Bk.push(cl));
      M = KP.map((_, i) => { let b1 = 1e9, b2 = 1e9, j1 = -1, j2 = -1; Bk.forEach((b, j) => { let s = 0; for (let k = 0; k < D; k++) s += (base[i][k] - b.d[k]) ** 2; s = Math.sqrt(s); if (s < b1) { b2 = b1; j2 = j1; b1 = s; j1 = j; } else if (s < b2) { b2 = s; j2 = j; } }); return { d1: b1, d2: b2, j1, j2 }; });
      st.dirty = true;
    }
    q('#noise').addEventListener('input', build);
    bindPlay(root, st, () => { pos = Math.min(n, Math.floor(pos) + 1); }, () => { pos = 0; });
    q('#all').onclick = () => { pos = n; st.dirty = true; };
    build();
    const verdict = (i) => { const m = M[i], tau = +q('#tau').value, rho = +q('#rho').value, on = q('#ron').checked; const ratio = m.d1 / m.d2; const acc = m.d1 <= tau && (!on || ratio < rho); return { acc, ratio, ok: Bk[m.j1].id === i }; };
    AI.loop(root, (dt) => {
      if (st.playing) { pos += dt * 0.4 * st.speed(); if (pos > n + 1.5) pos = 0; }
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      q('#tv').textContent = (+q('#tau').value).toFixed(2); q('#rv').textContent = (+q('#rho').value).toFixed(2);
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(), s = Math.min((W - 24) / 2, H - 112), A = { x: 8, y: 22 }, Bp = { x: W - 8 - s, y: 22 };
      g.clearRect(0, 0, W, H);
      T(g, 'view A (reference)', A.x, 14, C.muted); T(g, 'view B (rotated, zoomed out)', Bp.x, 14, C.muted);
      const kA = (p) => [A.x + (p[0] / 120) * s, A.y + (p[1] / 120) * s];
      const kB = (p) => { const t = tB(p); return [Bp.x + s / 2 + (t[0] / 120) * s, Bp.y + s / 2 + (t[1] / 120) * s]; };
      g.fillStyle = C.bg2; g.fillRect(A.x, A.y, s, s); g.fillRect(Bp.x, Bp.y, s, s);
      g.drawImage(img, A.x, A.y, s, s);
      g.save(); g.beginPath(); g.rect(Bp.x, Bp.y, s, s); g.clip(); g.translate(Bp.x + s / 2, Bp.y + s / 2); g.rotate(ROT); g.scale((SC * s) / 120, (SC * s) / 120); g.translate(-60, -60); g.drawImage(img, 0, 0); g.restore();
      KP.forEach((p) => { const [x, y] = kA(p); g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.stroke(); });
      Bk.forEach((b) => { const [x, y] = kB(b.p); g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.stroke(); });
      const cur = Math.min(n - 1, Math.floor(pos)), f = pos - cur, done = Math.min(n, Math.floor(pos));
      let acc = 0, good = 0, bad = 0, avail = 0, found = 0;
      for (let i = 0; i < n; i++) if (!MISSED.has(i)) avail++;
      const line = (i, j, col, w, dash, frac) => { const a = kA(KP[i]), b = kB(Bk[j].p); g.save(); g.strokeStyle = col; g.lineWidth = w; if (dash) g.setLineDash(dash); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac); g.stroke(); g.restore(); };
      for (let i = 0; i < done; i++) {
        const v = verdict(i), m = M[i];
        if (v.acc) { acc++; if (v.ok) { good++; found++; } else bad++; line(i, m.j1, v.ok ? C.green : C.red, 2, null, 1); }
        else line(i, m.j1, C.muted, 1, [3, 4], 1);
      }
      let cap = 'Press play: each keypoint of view A searches view B for the closest descriptor.';
      if (pos < n && pos >= 0) {
        const m = M[cur], v = verdict(cur);
        const stage = f < 0.25 ? 0 : f < 0.65 ? 1 : 2, a = kA(KP[cur]);
        g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath(); g.arc(a[0], a[1], 8, 0, 7); g.stroke();
        if (stage >= 1) { const e = AI.smooth((f - 0.25) / 0.4); line(cur, m.j1, C.orange, 2.5, null, e); if (m.j2 >= 0) line(cur, m.j2, C.yellow, 1.2, [4, 3], e); }
        if (stage === 2) { line(cur, m.j1, v.acc ? (v.ok ? C.green : C.red) : C.muted, v.acc ? 2.5 : 1.5, v.acc ? null : [3, 4], 1); }
        cap = `Keypoint ${cur + 1}/${n} (${NAMES[cur]}): nearest d₁=${AI.fmt(m.d1, 2)}, second d₂=${AI.fmt(m.d2, 2)}, ratio ${AI.fmt(v.ratio, 2)}`;
        if (stage === 2) cap += v.acc ? (v.ok ? ' → accepted (correct match).' : ' → accepted but WRONG: an outlier for RANSAC.') : (m.d1 > +q('#tau').value ? ' → rejected: too far.' : ' → rejected: ambiguous (ratio too high).');
        else cap += stage === 0 ? '  ① look at this keypoint’s descriptor' : '  ② find nearest (orange) and second nearest (yellow, dashed)';
        // descriptor bars
        const by = Bp.y + s + 14, bh = 46, gw = (W - 24) / 3;
        [['A descriptor', base[cur], C.blue], ['nearest in B', Bk[m.j1].d, C.orange], ['2nd nearest', m.j2 >= 0 ? Bk[m.j2].d : base[cur], C.yellow]].forEach(([lab, d, col], gi) => {
          const x0 = 12 + gi * gw; T(g, lab, x0, by - 2, C.muted, '11px sans-serif');
          const bw = (gw - 30) / D; d.forEach((v, k) => { const hh = AI.clamp(v, 0, 1.3) / 1.3 * bh; g.fillStyle = col; g.globalAlpha = gi && stage < 1 ? 0.15 : 0.9; g.fillRect(x0 + k * bw, by + bh + 4 - hh, bw - 3, hh); g.globalAlpha = 1; });
        });
        T(g, 'd = √Σ(pᵢ−qᵢ)²   (8 numbers each)', 12, H - 4, C.muted, '11px sans-serif');
      }
      if (txt(cap)) q('#cap').textContent = cap;
      const prec = acc ? good / acc : 0;
      q('#out').innerHTML = `matched so far: ${done}/${n}\naccepted: <b>${acc}</b>  (<span class="g">${good} correct</span>, <span class="r">${bad} wrong</span>)\nprecision: <b>${AI.fmt(prec * 100, 0)}%</b>\nrecall: <b>${AI.fmt(avail && done ? (found / avail) * 100 : 0, 0)}%</b> of ${avail} possible\n<span class="m">dark circles in B near nothing = clutter keypoints</span>`;
    }, { fps: 30 });
    let lastTxt = '';
    function txt(s) { if (s === lastTxt) return false; lastTxt = s; return true; }
  };

  /* ======================================================================= */
  /* cvRansac: line and homography                                              */
  /* ======================================================================= */
  function gauss(A, b) {
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      if (Math.abs(A[p][i]) < 1e-9) return null;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let k = i; k < n; k++) A[r][k] -= f * A[i][k]; b[r] -= f * b[i]; }
    }
    const x = Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= A[i][k] * x[k]; x[i] = s / A[i][i]; }
    return x;
  }
  AI.labs.cvRansac = function (root) {
    const NP = 40;
    const { view } = AI.labShell(root, 'RANSAC: guess from a minimal sample, then count the votes', 'each iteration: ① pick  ② fit  ③ count inliers  ④ keep if best',
      '<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>',
      `<label>Model <select id="mode"><option value="line">line (s = 2)</option><option value="homog">homography (s = 4)</option></select></label>
       <label>Outlier ratio <span class="v" id="ov"></span><input type="range" id="out" min="0" max="80" value="40"></label>
       <label>Inlier threshold t <span class="v" id="tv"></span><input type="range" id="thr" min="0.1" max="2" step="0.05" value="0.4"></label>
       <div class="row"><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="lsq"> least squares on all points</label><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="truth"> reveal true outliers</label></div>
       ${PLAY}
       <div class="row"><button class="btn" id="fast">Run 50 fast</button><button class="btn" id="refit">Refit on inliers</button><button class="btn" id="newd">New data</button></div>
       <div class="readout" id="rd"></div>`);
    const q = (s) => q$(root, s), st = {}, c = mkCanvas(root, view, st);
    let seed = 3, mode = 'line', P = null, rs, it, ph, tph, sample, model, inl, cnt, best, refit, degenerate, msg;
    const S = () => (mode === 'line' ? 2 : 4);
    const Htrue = [0.9, 0.25, 1.0, -0.15, 0.95, 0.5, 0.012, 0.006];
    const applyH = (h, x, y) => { const w = h[6] * x + h[7] * y + 1; if (Math.abs(w) < 1e-6) return [1e9, 1e9]; return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w]; };
    function gen() {
      mode = q('#mode').value; const r = AI.rng(seed * 7919), nOut = Math.round((NP * +q('#out').value) / 100);
      P = { ax: [], ay: [], bx: [], by: [], gt: [] };
      for (let i = 0; i < NP; i++) {
        const out = i < nOut;
        const x = r() * 10, y = r() * 10;
        if (mode === 'line') { P.ax.push(x); P.ay.push(out ? -1 + r() * 10 : 0.5 * x + 1 + 0.15 * AI.randn(r)); P.bx.push(0); P.by.push(0); }
        else { P.ax.push(x); P.ay.push(y); if (out) { P.bx.push(r() * 12); P.by.push(r() * 10); } else { const p = applyH(Htrue, x, y); P.bx.push(p[0] + 0.12 * AI.randn(r)); P.by.push(p[1] + 0.12 * AI.randn(r)); } }
        P.gt.push(!out);
      }
      // shuffle order so outliers are not first
      const idx = [...Array(NP).keys()]; for (let i = NP - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      for (const k of ['ax', 'ay', 'bx', 'by', 'gt']) P[k] = idx.map((i) => P[k][i]);
      reset();
    }
    function reset() { rs = AI.rng(seed * 31 + 5); it = 0; ph = 0; tph = 0; sample = []; model = null; inl = null; cnt = 0; best = { cnt: -1, model: null, inl: null }; refit = null; msg = ''; startIter(); st.dirty = true; }
    function pick() { const s = []; while (s.length < S()) { const i = Math.floor(rs() * NP); if (!s.includes(i)) s.push(i); } return s; }
    function fit(s) {
      if (mode === 'line') { const [i, j] = s, x1 = P.ax[i], y1 = P.ay[i], x2 = P.ax[j], y2 = P.ay[j], a = y1 - y2, b = x2 - x1, nn = Math.hypot(a, b); if (nn < 1e-6) return null; return [a / nn, b / nn, (x1 * y2 - x2 * y1) / nn]; }
      const A = [], b = []; for (const i of s) { const x = P.ax[i], y = P.ay[i], u = P.bx[i], v = P.by[i]; A.push([x, y, 1, 0, 0, 0, -x * u, -y * u], [0, 0, 0, x, y, 1, -x * v, -y * v]); b.push(u, v); }
      if (s.length === 4) return gauss(A, b);
      const AtA = Array.from({ length: 8 }, () => Array(8).fill(0)), Atb = Array(8).fill(0);
      A.forEach((row, r2) => { for (let a = 0; a < 8; a++) { Atb[a] += row[a] * b[r2]; for (let d = 0; d < 8; d++) AtA[a][d] += row[a] * row[d]; } });
      return gauss(AtA, Atb);
    }
    const err = (m, i) => (mode === 'line' ? Math.abs(m[0] * P.ax[i] + m[1] * P.ay[i] + m[2]) : (() => { const p = applyH(m, P.ax[i], P.ay[i]); return Math.hypot(p[0] - P.bx[i], p[1] - P.by[i]); })());
    function count(m) { const t = +q('#thr').value, a = [], res = []; let k = 0; for (let i = 0; i < NP; i++) { const e = err(m, i); res.push(e); const ok = e < t; a.push(ok); if (ok) k++; } return { a, k, res }; }
    function startIter() { sample = pick(); model = null; inl = null; cnt = 0; degenerate = false; ph = 0; tph = 0; }
    function enter(p) {
      ph = p; tph = 0;
      if (p === 1) { model = fit(sample); if (!model) { degenerate = true; } }
      if (p === 2) { if (model) { const r = count(model); inl = r.a; cnt = r.k; } else { inl = null; cnt = 0; } }
      if (p === 3) { if (model && cnt > best.cnt) { best = { cnt, model, inl }; msg = `new best: ${cnt} inliers`; } else msg = model ? `${cnt} ≤ best (${best.cnt}): discard` : 'degenerate sample: discard'; }
    }
    function advance() { if (ph < 3) enter(ph + 1); else { it++; startIter(); } st.dirty = true; }
    function fast(n) { for (let k = 0; k < n; k++) { const s = pick(), m = fit(s); it++; if (!m) continue; const r = count(m); if (r.k > best.cnt) best = { cnt: r.k, model: m, inl: r.a }; } startIter(); ph = 0; tph = 0; sample = []; model = null; inl = null; msg = `ran 50 iterations: best has ${best.cnt} inliers`; st.dirty = true; }
    function doRefit() {
      if (!best.model) return; const idx = []; best.inl.forEach((v, i) => v && idx.push(i)); if (idx.length < S()) return;
      if (mode === 'line') { let mx = 0, my = 0; idx.forEach((i) => { mx += P.ax[i]; my += P.ay[i]; }); mx /= idx.length; my /= idx.length; let sxx = 0, sxy = 0, syy = 0; idx.forEach((i) => { const dx = P.ax[i] - mx, dy = P.ay[i] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }); const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), dx = Math.cos(th), dy = Math.sin(th); refit = [-dy, dx, dy * mx - dx * my]; }
      else refit = fit(idx);
      st.dirty = true;
    }
    ['#mode', '#out'].forEach((s) => q(s).addEventListener('input', () => { if (s === '#mode') q('#thr').value = q('#mode').value === 'line' ? 0.4 : 0.5; gen(); }));
    q('#thr').addEventListener('input', () => { if (model && ph >= 2) enter(2); if (best.model) { const r = count(best.model); best.cnt = r.k; best.inl = r.a; } });
    q('#fast').onclick = () => fast(50); q('#refit').onclick = doRefit; q('#newd').onclick = () => { seed++; gen(); };
    bindPlay(root, st, advance, reset);
    gen();
    const DUR = [1.3, 1.3, 1.5, 1.4];
    AI.loop(root, (dt) => {
      if (st.playing) { tph += dt * st.speed(); if (tph >= DUR[ph]) advance(); }
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      const t = +q('#thr').value; q('#tv').textContent = t.toFixed(2); q('#ov').textContent = q('#out').value + '%';
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(), two = mode === 'homog'; g.clearRect(0, 0, W, H);
      const pw = two ? W / 2 : W, nOut = P.gt.filter((v) => !v).length, w = (NP - nOut) / NP, s = S();
      const panel = (k) => {
        const ph2 = H - 8, V = AI.view({ w: pw, h: ph2 }, two ? -0.5 : -0.5, two ? 12.5 : 10.5, two ? -0.5 : -1.5, two ? 10.5 : 9.5, { l: 28, r: 8, t: 20, b: 22 });
        g.save(); g.translate(k * pw, 0); AI.axes(g, V);
        g.beginPath(); g.rect(V.pad.l, V.pad.t, V.pw, V.ph); g.clip();
        return { V, close: () => g.restore() };
      };
      const X = (V, i, b) => V.X(b ? P.bx[i] : P.ax[i]), Y = (V, i, b) => V.Y(b ? P.by[i] : P.ay[i]);
      const dot = (V, i, b, col, r0, hollow) => { g.beginPath(); g.arc(X(V, i, b), Y(V, i, b), r0, 0, 7); if (hollow) { g.strokeStyle = col; g.lineWidth = 2; g.stroke(); } else { g.fillStyle = col; g.fill(); } };
      const drawLine = (V, m, col, lw, dash, band) => {
        const px = -m[0] * m[2], py = -m[1] * m[2], dx = -m[1], dy = m[0], L = 40;
        if (band) { g.fillStyle = col; g.globalAlpha = 0.12; g.beginPath(); g.moveTo(V.X(px + m[0] * band - dx * L), V.Y(py + m[1] * band - dy * L)); g.lineTo(V.X(px + m[0] * band + dx * L), V.Y(py + m[1] * band + dy * L)); g.lineTo(V.X(px - m[0] * band + dx * L), V.Y(py - m[1] * band + dy * L)); g.lineTo(V.X(px - m[0] * band - dx * L), V.Y(py - m[1] * band - dy * L)); g.closePath(); g.fill(); g.globalAlpha = 1; }
        g.save(); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.beginPath(); g.moveTo(V.X(px - dx * L), V.Y(py - dy * L)); g.lineTo(V.X(px + dx * L), V.Y(py + dy * L)); g.stroke(); g.restore();
      };
      const quad = (V, h, col, lw, dash) => { const pts = [[1, 1], [9, 1], [9, 9], [1, 9]].map((p) => (h ? applyH(h, p[0], p[1]) : p)); g.save(); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(V.X(p[0]), V.Y(p[1])) : g.moveTo(V.X(p[0]), V.Y(p[1])))); g.closePath(); g.stroke(); g.restore(); };
      const showInl = (inl && ph >= 2), bi = !showInl && ph === 0 && it > 0 && best.inl, colOf = (i) => (showInl ? (inl[i] ? C.green : C.muted) : bi ? (bi[i] ? C.green : C.muted) : C.muted);
      if (!two) {
        const { V, close } = panel(0);
        if (q('#lsq').checked) { let mx = 0, my = 0; for (let i = 0; i < NP; i++) { mx += P.ax[i]; my += P.ay[i]; } mx /= NP; my /= NP; let sxy = 0, sxx = 0; for (let i = 0; i < NP; i++) { sxy += (P.ax[i] - mx) * (P.ay[i] - my); sxx += (P.ax[i] - mx) ** 2; } const sl = sxy / sxx; drawLine(V, (() => { const nn = Math.hypot(sl, 1); return [sl / nn, -1 / nn, (my - sl * mx) / nn]; })(), C.purple, 2, [2, 4]); }
        if (best.model) { drawLine(V, best.model, C.green, 3, null, t); }
        if (refit) drawLine(V, refit, C.blue, 2.5, null);
        if (model && ph >= 1) drawLine(V, model, C.orange, 2, [6, 4], ph >= 2 ? t : 0);
        for (let i = 0; i < NP; i++) { dot(V, i, false, colOf(i), 4); if (q('#truth').checked && !P.gt[i]) { g.strokeStyle = C.red; g.lineWidth = 1.5; const x = X(V, i), y = Y(V, i); g.beginPath(); g.moveTo(x - 6, y - 6); g.lineTo(x + 6, y + 6); g.moveTo(x + 6, y - 6); g.lineTo(x - 6, y + 6); g.stroke(); } }
        if (ph === 0 || ph >= 1) sample.forEach((i) => dot(V, i, false, C.orange, 8, true));
        close();
        T(g, 'orange dashed = current guess · green = best so far' + (refit ? ' · blue = refit' : ''), 30, 14, C.muted, '11px sans-serif');
      } else {
        const A = panel(0), pa = A.V;
        quad(pa, null, C.muted, 1.5);
        for (let i = 0; i < NP; i++) dot(pa, i, false, colOf(i), 3.5);
        sample.forEach((i) => dot(pa, i, false, C.orange, 8, true));
        A.close();
        const B = panel(1), pb = B.V;
        if (best.model) quad(pb, best.model, C.green, 3);
        if (refit) quad(pb, refit, C.blue, 2.5);
        if (model && ph >= 1) quad(pb, model, C.orange, 2, [6, 4]);
        for (let i = 0; i < NP; i++) {
          if (model && ph >= 2) { const p = applyH(model, P.ax[i], P.ay[i]); g.strokeStyle = inl && inl[i] ? C.green : C.red; g.globalAlpha = 0.6; g.lineWidth = 1; g.beginPath(); g.moveTo(pb.X(p[0]), pb.Y(p[1])); g.lineTo(X(pb, i, true), Y(pb, i, true)); g.stroke(); g.globalAlpha = 1; g.beginPath(); g.arc(pb.X(p[0]), pb.Y(p[1]), 3, 0, 7); g.strokeStyle = C.orange; g.stroke(); }
          dot(pb, i, true, colOf(i), 3.5); if (q('#truth').checked && !P.gt[i]) { const x = X(pb, i, true), y = Y(pb, i, true); g.strokeStyle = C.red; g.beginPath(); g.moveTo(x - 6, y - 6); g.lineTo(x + 6, y + 6); g.moveTo(x + 6, y - 6); g.lineTo(x - 6, y + 6); g.stroke(); }
        }
        sample.forEach((i) => dot(pb, i, true, C.orange, 8, true));
        B.close();
        T(g, 'photo 1 points (square = region)', 30, 14, C.muted, '11px sans-serif'); T(g, 'photo 2: points, predicted (hollow) and residuals', pw + 30, 14, C.muted, '11px sans-serif');
      }
      const caps = [`① Pick ${s} random points (orange rings): the smallest set that defines a ${mode === 'line' ? 'line' : 'homography'}.`, degenerate ? '② These points cannot define a model (degenerate). Skip.' : `② Fit the model to only those ${s} points.`, `③ Count inliers: points within t = ${t.toFixed(2)} of the model (green). ${cnt} of ${NP}.`, `④ ${msg || 'Keep the model if it beats the best so far.'}`];
      const cap = `Iteration ${it + 1}. ` + caps[ph]; if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      const ws = Math.pow(w, s), P1 = 1 - Math.pow(1 - ws, it), Nreq = ws > 0 && ws < 1 ? Math.ceil(Math.log(0.01) / Math.log(1 - ws)) : 1;
      const wb = best.cnt > 0 ? best.cnt / NP : 0;
      q('#rd').innerHTML = `true inlier fraction w = <b>${AI.fmt(w, 2)}</b>  (${NP - nOut}/${NP})\nw^s = ${AI.fmt(ws, 4)}  per-iteration success\niterations so far: <b>${it}</b>\nP(≥1 clean sample) = 1−(1−w^s)^N = <b class="g">${AI.fmt(P1 * 100, 1)}%</b>\nN needed for 99%: <b class="o">${Nreq}</b>\nbest inliers: <b>${Math.max(0, best.cnt)}</b> (w̃ = ${AI.fmt(wb, 2)})`;
    }, { fps: 30 });
  };

  /* ======================================================================= */
  /* cvPinhole                                                                  */
  /* ======================================================================= */
  AI.labs.cvPinhole = function (root) {
    const { view } = AI.labShell(root, 'The pinhole camera: rays from 3-D points to pixels', 'top view (X right, Z forward) and the image the camera records; pulses travel along the rays',
      '<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>',
      `<label>Focal length f <span class="v" id="fv"></span><input type="range" id="f" min="250" max="1600" step="10" value="800"></label>
       <label>Camera X position <span class="v" id="cxv"></span><input type="range" id="cx" min="-3" max="3" step="0.1" value="0"></label>
       <label>Camera yaw <span class="v" id="yv"></span><input type="range" id="yaw" min="-35" max="35" step="1" value="0"></label>
       <label>Cube distance Z <span class="v" id="zv"></span><input type="range" id="zc" min="3" max="12" step="0.1" value="6"></label>
       <label>Lens distortion k₁ <span class="v" id="kv"></span><input type="range" id="k1" min="-0.3" max="0.3" step="0.01" value="0"></label>
       <label>Follow vertex <select id="vtx">${[...Array(8).keys()].map((k) => `<option value="${k}" ${k === 1 ? 'selected' : ''}>${k}</option>`).join('')}</select></label>
       ${PLAY}<div class="readout" id="out"></div>`);
    const q = (s) => q$(root, s), st = {}, c = mkCanvas(root, view, st);
    let tt = 0;
    const edges = []; for (let a = 0; a < 8; a++) for (let b = 0; b < 3; b++) { const o = a ^ (1 << b); if (o > a) edges.push([a, o]); }
    bindPlay(root, st, () => { tt += 1; }, () => { tt = 0; });
    AI.loop(root, (dt) => {
      if (st.playing) tt += dt * st.speed();
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      const f = +q('#f').value, Cx = +q('#cx').value, psi = (+q('#yaw').value * Math.PI) / 180, Zc = +q('#zc').value, k1 = +q('#k1').value, sel = +q('#vtx').value;
      q('#fv').textContent = f + ' px'; q('#cxv').textContent = Cx.toFixed(1) + ' m'; q('#yv').textContent = q('#yaw').value + '°'; q('#zv').textContent = Zc.toFixed(1) + ' m'; q('#kv').textContent = k1.toFixed(2);
      const cs = Math.cos(psi), sn = Math.sin(psi), X0 = 0.6;
      const V = [...Array(8).keys()].map((k) => [X0 + (k & 1 ? 1 : -1), k & 2 ? 1 : -1, Zc + (k & 4 ? 1 : -1)]);
      const toCam = (p) => { const dx = p[0] - Cx, dz = p[2]; return [cs * dx - sn * dz, p[1], sn * dx + cs * dz]; };
      const pix = (p) => { if (p[2] < 0.05) return null; const xn = p[0] / p[2], yn = p[1] / p[2], d = 1 + k1 * (xn * xn + yn * yn); return [f * xn * d + 320, f * yn * d + 240]; };
      const Cam = V.map(toCam), Px = Cam.map(pix);
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(); g.clearRect(0, 0, W, H);
      // top view
      const tw = W * 0.5, Vw = AI.view({ w: tw, h: H }, -7, 7, -1.5, 12, { l: 22, r: 6, t: 20, b: 18 });
      AI.axes(g, Vw);
      T(g, 'top view', 24, 14, C.muted);
      const fwd = [sn, cs], rt = [cs, -sn], dd = 1.6, hw = (dd * 320) / f;
      const pc = [Cx, 0];
      const plane = (a) => [pc[0] + dd * fwd[0] + a * rt[0], pc[1] + dd * fwd[1] + a * rt[1]];
      // FoV wedge
      const tanH = 320 / f; g.save(); g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.lineWidth = 1;
      for (const sg of [-1, 1]) { const dx = fwd[0] + sg * tanH * rt[0], dz = fwd[1] + sg * tanH * rt[1]; g.beginPath(); g.moveTo(Vw.X(pc[0]), Vw.Y(pc[1])); g.lineTo(Vw.X(pc[0] + dx * 14), Vw.Y(pc[1] + dz * 14)); g.stroke(); }
      g.restore();
      // cube footprint
      g.strokeStyle = C.fg; g.lineWidth = 1.5; g.strokeRect(Vw.X(X0 - 1), Vw.Y(Zc + 1), Vw.X(X0 + 1) - Vw.X(X0 - 1), Vw.Y(Zc - 1) - Vw.Y(Zc + 1));
      // image plane
      const p1 = plane(-hw), p2 = plane(hw); g.strokeStyle = C.blue; g.lineWidth = 4; g.beginPath(); g.moveTo(Vw.X(p1[0]), Vw.Y(p1[1])); g.lineTo(Vw.X(p2[0]), Vw.Y(p2[1])); g.stroke();
      // camera glyph
      g.fillStyle = C.fg; g.beginPath(); g.arc(Vw.X(pc[0]), Vw.Y(pc[1]), 5, 0, 7); g.fill();
      T(g, 'pinhole', Vw.X(pc[0]) + 8, Vw.Y(pc[1]) + 4, C.muted, '11px sans-serif'); T(g, 'image plane', Vw.X(p2[0]) + 4, Vw.Y(p2[1]) - 4, C.blue, '11px sans-serif');
      const rayIdx = [0, 1, 4, 5, sel].filter((v, i, a) => a.indexOf(v) === i);
      rayIdx.forEach((k, ri) => {
        const cp = Cam[k]; if (cp[2] < 0.05) return; const isSel = k === sel, col = isSel ? C.orange : C.yellow;
        const w0 = [V[k][0], V[k][2]];
        g.strokeStyle = col; g.globalAlpha = isSel ? 1 : 0.55; g.lineWidth = isSel ? 2 : 1; g.beginPath(); g.moveTo(Vw.X(w0[0]), Vw.Y(w0[1])); g.lineTo(Vw.X(pc[0]), Vw.Y(pc[1])); g.stroke(); g.globalAlpha = 1;
        const ip = plane((dd * cp[0]) / cp[2]); g.fillStyle = col; g.beginPath(); g.arc(Vw.X(ip[0]), Vw.Y(ip[1]), 4, 0, 7); g.fill();
        const s0 = ((tt / 4) + ri * 0.2) % 1, px = AI.lerp(w0[0], pc[0], s0), pz = AI.lerp(w0[1], pc[1], s0);
        g.fillStyle = '#fff'; g.beginPath(); g.arc(Vw.X(px), Vw.Y(pz), 3.5, 0, 7); g.fill();
        g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
      });
      // image panel
      const ix0 = tw + 12, iw = W - ix0 - 8, ih = Math.min(H - 56, (iw * 3) / 4), iwd = (ih * 4) / 3, sc = iwd / 640, ox = ix0 + (iw - iwd) / 2, oy = 24;
      T(g, 'image (640×480 px)', ox, 16, C.muted);
      g.fillStyle = C.bg2; g.fillRect(ox, oy, iwd, ih); g.strokeStyle = C.border; g.strokeRect(ox, oy, iwd, ih);
      g.save(); g.beginPath(); g.rect(ox, oy, iwd, ih); g.clip();
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(ox + iwd / 2, oy); g.lineTo(ox + iwd / 2, oy + ih); g.moveTo(ox, oy + ih / 2); g.lineTo(ox + iwd, oy + ih / 2); g.stroke();
      g.strokeStyle = C.fg; g.lineWidth = 2;
      edges.forEach(([a, b]) => { g.beginPath(); let ok = false; for (let i = 0; i <= 14; i++) { const t = i / 14, P3 = [0, 1, 2].map((d) => Cam[a][d] + (Cam[b][d] - Cam[a][d]) * t), p = pix(P3); if (!p) { ok = false; continue; } const X = ox + p[0] * sc, Y = oy + p[1] * sc; if (!ok) { g.moveTo(X, Y); ok = true; } else g.lineTo(X, Y); } g.stroke(); });
      Px.forEach((p, k) => { if (!p) return; g.fillStyle = k === sel ? C.orange : C.yellow; g.beginPath(); g.arc(ox + p[0] * sc, oy + p[1] * sc, k === sel ? 5 : 3, 0, 7); g.fill(); });
      g.restore();
      T(g, '(0,0)', ox + 2, oy + ih + 12, C.muted, '10px sans-serif'); T(g, '(640,480)', ox + iwd, oy + ih + 12, C.muted, '10px sans-serif', 'right');
      const fov = (2 * Math.atan(320 / f) * 180) / Math.PI;
      const cap = Cam[sel][2] < 0.05 ? 'The followed vertex is behind the camera: it has no pixel.' : 'Every point on a ray lands on the same pixel; f scales the whole image, 1/Z shrinks distant things.';
      if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      const w = V[sel], cm = Cam[sel], p = Px[sel], n3 = (a) => a.map((v) => AI.fmt(v, 2)).join(', ');
      q('#out').innerHTML = `K = [[${f},0,320],[0,${f},240],[0,0,1]]\nFoV(x) = <b>${AI.fmt(fov, 1)}°</b>\nvertex ${sel}: X_w = (${n3(w)})\nX_c = R(X_w−C) = (${n3(cm)})\n` + (p ? `u = f·x/z + 320 = <b class="o">${AI.fmt(p[0], 1)}</b>\nv = f·y/z + 240 = <b class="o">${AI.fmt(p[1], 1)}</b>${Math.abs(k1) > 1e-9 ? '\n<span class="m">(includes distortion k₁)</span>' : ''}` : '<span class="r">behind camera</span>');
    }, { fps: 30 });
  };

  /* ======================================================================= */
  /* cvStereo                                                                   */
  /* ======================================================================= */
  AI.labs.cvStereo = function (root) {
    const { view } = AI.labShell(root, 'Stereo: disparity tells you depth', 'two cameras, one baseline; the nearer the point, the bigger the shift between left and right images',
      '<canvas style="height:380px"></canvas><div class="caption" id="cap"></div>',
      `<label>Baseline B <span class="v" id="bv"></span><input type="range" id="B" min="0.1" max="1" step="0.01" value="0.54"></label>
       <label>Focal length f <span class="v" id="fv"></span><input type="range" id="f" min="300" max="1400" step="10" value="700"></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="auto" checked> auto-move the orange point</label>
       <label>Point depth Z (when not auto) <span class="v" id="zv"></span><input type="range" id="Z" min="1.5" max="25" step="0.1" value="6"></label>
       ${PLAY}<div class="readout" id="out"></div>`);
    const q = (s) => q$(root, s), st = {}, c = mkCanvas(root, view, st);
    let ph = 0;
    const OBJ = [{ X: -1.2, Z: 3, col: 'blue' }, { X: 0.9, Z: 6, col: 'green' }, { X: -0.4, Z: 12, col: 'purple' }];
    bindPlay(root, st, () => { ph += 0.5; }, () => { ph = 0; });
    AI.loop(root, (dt) => {
      if (st.playing) ph += dt * 0.35 * st.speed();
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      const B = +q('#B').value, f = +q('#f').value, auto = q('#auto').checked;
      const Z = auto ? Math.exp(AI.lerp(Math.log(2), Math.log(22), 0.5 + 0.5 * Math.sin(ph))) : +q('#Z').value;
      q('#bv').textContent = B.toFixed(2) + ' m'; q('#fv').textContent = f + ' px'; q('#zv').textContent = Z.toFixed(1) + ' m';
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(); g.clearRect(0, 0, W, H);
      const objs = OBJ.map((o) => ({ ...o, c: C[o.col] })).concat([{ X: 0.3, Z, c: C.orange, main: true }]);
      // top view
      const tw = Math.max(150, W * 0.36), Vt = AI.view({ w: tw, h: H }, -3, 3, -1, 24, { l: 22, r: 6, t: 20, b: 14 });
      AI.axes(g, Vt); T(g, 'top view', 24, 14, C.muted);
      const cams = [-B / 2, B / 2];
      objs.forEach((o) => { cams.forEach((cx) => { g.strokeStyle = o.c; g.globalAlpha = o.main ? 0.9 : 0.35; g.lineWidth = o.main ? 1.5 : 1; g.beginPath(); g.moveTo(Vt.X(o.X), Vt.Y(o.Z)); g.lineTo(Vt.X(cx), Vt.Y(0)); g.stroke(); g.globalAlpha = 1; }); g.fillStyle = o.c; g.beginPath(); g.arc(Vt.X(o.X), Vt.Y(o.Z), o.main ? 6 : 4, 0, 7); g.fill(); });
      cams.forEach((cx, i) => { g.fillStyle = C.fg; g.fillRect(Vt.X(cx) - 5, Vt.Y(0) - 3, 10, 8); T(g, i ? 'R' : 'L', Vt.X(cx) - 3, Vt.Y(0) + 18, C.fg, 'bold 11px sans-serif'); });
      g.strokeStyle = C.fg; g.beginPath(); g.moveTo(Vt.X(cams[0]), Vt.Y(0) - 8); g.lineTo(Vt.X(cams[1]), Vt.Y(0) - 8); g.stroke(); T(g, 'B', Vt.X(0) - 3, Vt.Y(0) - 11, C.fg, '11px sans-serif');
      // scanlines
      const rx = tw + 14, rw = W - rx - 8, sy = [28, 28 + 78], su = (u) => rx + ((u + 320) / 640) * rw;
      const xs = objs.map((o) => ({ ...o, xl: (f * (o.X + B / 2)) / o.Z, xr: (f * (o.X - B / 2)) / o.Z }));
      ['left image (row y)', 'right image (same row)'].forEach((lab, k) => {
        T(g, lab, rx, sy[k] - 6, C.muted, '11px sans-serif'); g.fillStyle = C.bg2; g.fillRect(rx, sy[k], rw, 36); g.strokeStyle = C.border; g.strokeRect(rx, sy[k], rw, 36);
        g.beginPath(); g.moveTo(su(0), sy[k]); g.lineTo(su(0), sy[k] + 36); g.stroke();
        xs.forEach((o) => { const u = k ? o.xr : o.xl; if (Math.abs(u) > 320) return; g.fillStyle = o.c; g.beginPath(); g.arc(su(u), sy[k] + 18, o.main ? 7 : 5, 0, 7); g.fill(); });
      });
      xs.forEach((o) => { if (Math.abs(o.xl) > 320 || Math.abs(o.xr) > 320) return; g.strokeStyle = o.c; g.globalAlpha = o.main ? 0.9 : 0.4; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(su(o.xl), sy[0] + 18); g.lineTo(su(o.xr), sy[1] + 18); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; });
      const m = xs[xs.length - 1], d = m.xl - m.xr;
      if (Math.abs(m.xl) < 320 && Math.abs(m.xr) < 320) { g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath(); g.moveTo(su(m.xr), sy[1] + 42); g.lineTo(su(m.xl), sy[1] + 42); g.stroke(); T(g, `d = ${AI.fmt(d, 1)} px`, (su(m.xr) + su(m.xl)) / 2, sy[1] + 56, C.orange, 'bold 12px sans-serif', 'center'); }
      // curve
      const cy0 = 28 + 78 + 70, ch = H - cy0 - 4, Vc = AI.view({ w: rw + 40, h: ch }, 0, 24, 0, Math.max(40, (f * B) / 1.5), { l: 36, r: 8, t: 14, b: 20 });
      g.save(); g.translate(rx - 30, cy0);
      AI.axes(g, Vc); T(g, 'disparity d (px) vs depth Z (m): d = fB / Z', 36, 10, C.muted, '11px sans-serif');
      g.fillStyle = C.orange; g.globalAlpha = 0.15; g.beginPath(); let first = true;
      for (let z = 1.5; z <= 24; z += 0.25) { const dd = (f * B) / z + 1; first ? g.moveTo(Vc.X(z), Vc.Y(dd)) : g.lineTo(Vc.X(z), Vc.Y(dd)); first = false; }
      for (let z = 24; z >= 1.5; z -= 0.25) g.lineTo(Vc.X(z), Vc.Y(Math.max(0, (f * B) / z - 1)));
      g.closePath(); g.fill(); g.globalAlpha = 1;
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); for (let z = 1.5; z <= 24; z += 0.25) { const y = (f * B) / z; z === 1.5 ? g.moveTo(Vc.X(z), Vc.Y(y)) : g.lineTo(Vc.X(z), Vc.Y(y)); } g.stroke();
      objs.forEach((o) => { g.fillStyle = o.c; g.beginPath(); g.arc(Vc.X(o.Z), Vc.Y((f * B) / o.Z), o.main ? 6 : 4, 0, 7); g.fill(); });
      g.restore();
      const fb = f * B, dz = fb / Math.max(0.5, d - 1) - Z, dzm = Z - fb / (d + 1);
      const cap = Z < 5 ? 'Close: big disparity, depth is measured precisely.' : Z > 14 ? 'Far: the disparity is a few pixels, so one pixel of error is metres of depth.' : 'Move the point nearer and the gap between its left and right positions grows.';
      if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      q('#out').innerHTML = `f·B = ${AI.fmt(fb, 1)} px·m\nZ = <b>${AI.fmt(Z, 2)}</b> m  →  d = fB/Z = <b class="o">${AI.fmt(d, 2)}</b> px\nback: Z = fB/d = ${AI.fmt(fb / d, 2)} m\n±1 px of disparity → Z in [${AI.fmt(fb / (d + 1), 2)}, ${AI.fmt(fb / Math.max(0.5, d - 1), 2)}] m\nΔZ ≈ Z²/(fB) = ${AI.fmt((Z * Z) / fb, 2)} m per pixel`;
    }, { fps: 30 });
  };

  /* ======================================================================= */
  /* cvFlow                                                                     */
  /* ======================================================================= */
  AI.labs.cvFlow = function (root) {
    const N = 48;
    const { view } = AI.labShell(root, 'Optical flow: Lucas–Kanade and the aperture problem', 'a patch moves between two frames; each window solves a 2×2 system for its own arrow',
      '<canvas style="height:360px;touch-action:none;cursor:crosshair"></canvas><div class="caption" id="cap"></div>',
      `<label>True motion u (px) <span class="v" id="uv"></span><input type="range" id="u" min="-3" max="3" step="0.1" value="1.5"></label>
       <label>True motion v (px) <span class="v" id="vv"></span><input type="range" id="v" min="-3" max="3" step="0.1" value="1"></label>
       <label>Window size <span class="v" id="wv"></span><input type="range" id="w" min="3" max="11" step="2" value="5"></label>
       <div class="row"><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="tex" checked> texture inside</label><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="ap"> peek through an aperture</label></div>
       ${PLAY}<div class="readout" id="out"></div>`);
    const q = (s) => q$(root, s), st = {}, cv = AI.$('canvas', view), c = mkCanvas(root, view, st);
    const r = AI.rng(5), blobs = Array.from({ length: 9 }, () => ({ x: (r() - 0.5) * 18, y: (r() - 0.5) * 18, a: (r() - 0.5) * 0.7, s: 1.6 + r() * 1.2 }));
    const f = (x, y, ox, oy, tex) => { const dx = x - 24 - ox, dy = y - 24 - oy, m = 1 / (1 + Math.exp(-(11 - Math.max(Math.abs(dx), Math.abs(dy))) / 0.9)); let t = 0.75; if (tex) for (const b of blobs) t += b.a * Math.exp(-((dx - b.x) ** 2 + (dy - b.y) ** 2) / (2 * b.s * b.s)); return 0.25 + 0.1 * (x / N) + m * (t - 0.25 - 0.1 * (x / N)); };
    const I1 = new Float32Array(N * N), I2 = new Float32Array(N * N), Ix = new Float32Array(N * N), Iy = new Float32Array(N * N), It = new Float32Array(N * N);
    let field = [], probe = { x: 35, y: 24 }, tt = 0, U = 1.5, V = 1, rd = null;
    const cl = (v) => Math.min(N - 1, Math.max(0, v));
    const solve = (x, y, w) => {
      const rr = (w - 1) / 2; let a = 0, b = 0, d = 0, bx = 0, by = 0;
      for (let j = -rr; j <= rr; j++) for (let i = -rr; i <= rr; i++) { const k = cl(y + j) * N + cl(x + i); a += Ix[k] * Ix[k]; b += Ix[k] * Iy[k]; d += Iy[k] * Iy[k]; bx += Ix[k] * It[k]; by += Iy[k] * It[k]; }
      const tr = a + d, det = a * d - b * b, h = tr / 2, sq = Math.sqrt(Math.max(0, h * h - det)), l1 = h + sq, l2 = h - sq;
      let type = 0, u = 0, v = 0, ex = 1, ey = 0;
      if (l1 > 0.02) {
        if (l2 < 0.08 * l1) { type = 1; if (Math.abs(b) > 1e-12) { ex = l1 - d; ey = b; } else if (a >= d) { ex = 1; ey = 0; } else { ex = 0; ey = 1; } const nn = Math.hypot(ex, ey); ex /= nn; ey /= nn; const s = -(bx * ex + by * ey) / l1; u = s * ex; v = s * ey; }
        else { type = 2; u = -(d * bx - b * by) / det; v = -(a * by - b * bx) / det; }
      }
      return { a, b, d, bx, by, l1, l2, type, u, v };
    };
    function frames() {
      U = +q('#u').value; V = +q('#v').value; const tex = q('#tex').checked, w = +q('#w').value;
      q('#uv').textContent = U.toFixed(1); q('#vv').textContent = V.toFixed(1); q('#wv').textContent = `${w}×${w}`;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = y * N + x; I1[i] = f(x, y, 0, 0, tex); I2[i] = f(x, y, U, V, tex); }
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = y * N + x, xp = cl(x + 1), xm = cl(x - 1), yp = cl(y + 1), ym = cl(y - 1); Ix[i] = (I1[y * N + xp] - I1[y * N + xm]) / 2; Iy[i] = (I1[yp * N + x] - I1[ym * N + x]) / 2; It[i] = I2[i] - I1[i]; }
      field = []; for (let y = 4; y < N - 2; y += 4) for (let x = 4; x < N - 2; x += 4) { const s = solve(x, y, w); if (s.type) field.push({ x, y, ...s }); }
      st.dirty = true;
    }
    ['#u', '#v', '#w', '#tex'].forEach((s) => q(s).addEventListener('input', frames));
    bindPlay(root, st, () => { tt = tt < 4 ? 4 : 0; }, () => { tt = 0; });
    frames();
    let layout = null, drag = false;
    const place = (e) => { const [mx, my] = ptr(cv, e); if (!layout) return; for (const P of layout) if (mx >= P.x && mx <= P.x + P.s && my >= P.y && my <= P.y + P.s) { probe = { x: Math.round(((mx - P.x) / P.s) * N - 0.5), y: Math.round(((my - P.y) / P.s) * N - 0.5) }; st.dirty = true; } };
    cv.addEventListener('pointerdown', (e) => { drag = true; cv.setPointerCapture(e.pointerId); place(e); });
    cv.addEventListener('pointermove', (e) => drag && place(e));
    cv.addEventListener('pointerup', () => (drag = false));
    const mid = document.createElement('canvas'); mid.width = mid.height = N; const mg = mid.getContext('2d'), mdat = mg.createImageData(N, N);
    AI.loop(root, (dt) => {
      if (st.playing) { tt += dt * st.speed(); if (tt >= 8) tt -= 8; }
      if (!(st.playing || st.dirty)) return; st.dirty = false;
      const s = tt < 2 ? 0 : tt < 4 ? AI.smooth((tt - 2) / 2) : tt < 6 ? 1 : 1 - AI.smooth((tt - 6) / 2);
      const tex = q('#tex').checked, w = +q('#w').value;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const v = Math.round(255 * Math.min(1, Math.max(0, f(x, y, U * s, V * s, tex)))), k = (y * N + x) * 4; mdat.data[k] = mdat.data[k + 1] = mdat.data[k + 2] = v; mdat.data[k + 3] = 255; }
      mg.putImageData(mdat, 0, 0);
      const W = c.w, H = c.h, g = c.ctx, C = AI.col(); g.clearRect(0, 0, W, H);
      const sz = Math.min(H - 30, (W - 24) / 2), cell = sz / N, A = { x: 8, y: 22, s: sz }, B = { x: W - 8 - sz, y: 22, s: sz }; layout = [A, B];
      T(g, tt < 3 ? 'frame t' : tt < 5 ? (tt < 4 ? 'moving…' : 'frame t+1') : 'back…', A.x, 14, C.muted); T(g, 'Lucas–Kanade flow (arrows ×3)', B.x, 14, C.muted);
      g.imageSmoothingEnabled = false; g.drawImage(mid, A.x, A.y, sz, sz);
      g.globalAlpha = 0.55; g.drawImage(mid, B.x, B.y, sz, sz); g.globalAlpha = 1;
      g.fillStyle = C.bg; g.globalAlpha = 0.55; g.fillRect(B.x, B.y, sz, sz); g.globalAlpha = 1;
      for (const a of field) { const x0 = B.x + (a.x + 0.5) * cell, y0 = B.y + (a.y + 0.5) * cell; AI.arrow(g, x0, y0, x0 + a.u * cell * 3, y0 + a.v * cell * 3, a.type === 2 ? C.green : C.orange, 1.8); }
      // true motion legend
      AI.arrow(g, B.x + 14, B.y + sz - 14, B.x + 14 + U * cell * 3, B.y + sz - 14 + V * cell * 3, '#ffffff', 2); T(g, 'true', B.x + 20, B.y + sz - 4, C.fg, '10px sans-serif');
      // aperture overlay
      const px = probe.x, py = probe.y, rr = ((w - 1) / 2 + 0.5) * cell;
      if (q('#ap').checked) { g.save(); g.fillStyle = 'rgba(0,0,0,0.88)'; g.beginPath(); g.rect(A.x, A.y, sz, sz); g.arc(A.x + (px + 0.5) * cell, A.y + (py + 0.5) * cell, rr + 3, 0, 7); g.fill('evenodd'); g.restore(); }
      for (const P of [A, B]) { g.strokeStyle = C.blue; g.lineWidth = 2.5; g.strokeRect(P.x + (px - (w - 1) / 2) * cell, P.y + (py - (w - 1) / 2) * cell, w * cell, w * cell); }
      const e = solve(cl(px), cl(py), w);
      if (e.type) AI.arrow(g, B.x + (px + 0.5) * cell, B.y + (py + 0.5) * cell, B.x + (px + 0.5) * cell + e.u * cell * 3, B.y + (py + 0.5) * cell + e.v * cell * 3, C.blue, 3);
      const kind = e.type === 2 ? 'full flow (both directions known)' : e.type === 1 ? 'normal flow only (aperture problem)' : 'no information (flat)';
      const cap = (q('#ap').checked ? 'Through the hole you see only an edge sliding: many motions look the same. ' : '') + (tt < 2 ? 'Frame t: brightness pattern.' : tt < 4 ? 'The pattern moves by (u, v): brightness constancy says each patch keeps its brightness.' : tt < 6 ? 'Frame t+1.' : 'Back to frame t.');
      if (q('#cap').textContent !== cap) q('#cap').textContent = cap;
      q('#out').innerHTML = `window at (${px}, ${py}): <b class="${e.type === 2 ? 'g' : 'o'}">${kind}</b>\nM = [[${AI.fmt(e.a, 3)}, ${AI.fmt(e.b, 3)}], [${AI.fmt(e.b, 3)}, ${AI.fmt(e.d, 3)}]]\nb = (${AI.fmt(e.bx, 3)}, ${AI.fmt(e.by, 3)})\nλ₁=${AI.fmt(e.l1, 3)}  λ₂=${AI.fmt(e.l2, 3)}\nsolve M·(u,v) = −b\nestimate (u, v) = <b>(${AI.fmt(e.u, 2)}, ${AI.fmt(e.v, 2)})</b>\ntrue     (u, v) = (${U.toFixed(1)}, ${V.toFixed(1)})`;
    }, { fps: 30 });
  };
})();
