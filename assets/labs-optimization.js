/* Optimization labs (Module 6): landscapes, convexity, gradient descent (1D, quadratic, step size), SGD, momentum, Adam table,
   learning-rate schedules, constrained optimisation (Lagrange, L1 vs L2).
   All animations run on dt (real time), start slow, and have a speed slider + pause / step buttons. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btn = (id, t, cls) => `<button class="${cls || 'btn'}" id="${id}">${t}</button>`;
  const mkChips = (box, items, fn) => items.forEach(([label, val]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = () => fn(val, label); box.appendChild(b); });
  const drag = (cv, fn) => {
    let on = false;
    const p = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('pointerdown', (e) => { on = true; try { cv.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ } fn(...p(e)); });
    cv.addEventListener('pointermove', (e) => { if (on) fn(...p(e)); });
    const up = () => (on = false);
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  };
  /** isotropic mapping: data rectangle fitted and centred in the canvas */
  const iso = (c, x0, x1, y0, y1, pad = 8) => {
    const s = Math.min((c.w - 2 * pad) / (x1 - x0), (c.h - 2 * pad) / (y1 - y0)), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return { s, X: (x) => c.w / 2 + (x - cx) * s, Y: (y) => c.h / 2 - (y - cy) * s, ix: (px) => cx + (px - c.w / 2) / s, iy: (py) => cy - (py - c.h / 2) / s };
  };
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const dot = (g, x, y, r, fill, stroke) => { g.beginPath(); g.arc(x, y, r, 0, 7); g.fillStyle = fill; g.fill(); if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); } };
  const num = (v, d = 4) => (Number.isFinite(v) ? (Math.abs(v) >= 1e5 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : String(+v.toFixed(d))) : v > 0 ? '∞' : Number.isNaN(v) ? 'NaN' : '−∞');
  const playBtn = (root, state) => { const b = q(root, '#play'); b.onclick = () => { state.playing = !state.playing; b.textContent = state.playing ? '⏸ Pause' : '▶ Play'; }; };
  const setPlay = (root, state, v) => { state.playing = v; q(root, '#play').textContent = v ? '⏸ Pause' : '▶ Play'; };
  const eig2 = (H) => { const m = (H.a + H.d) / 2, r = Math.hypot((H.a - H.d) / 2, H.b); return { l1: m + r, l2: m - r, th: 0.5 * Math.atan2(2 * H.b, H.a - H.d) }; };

  /* ===================================================================
     Heat-map helper shared by landscapes
     =================================================================== */
  function paintHeat(off, c, fn, V) {
    const W = 110, H = Math.max(40, Math.round((110 * c.h) / c.w));
    off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H), vals = new Float64Array(W * H), [x0, x1, y0, y1] = V;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) vals[j * W + i] = fn({ x: x0 + ((i + 0.5) / W) * (x1 - x0), y: y1 - ((j + 0.5) / H) * (y1 - y0) });
    const sorted = [...vals].filter(Number.isFinite).sort((a, b) => a - b);
    const lo = sorted[0] ?? 0, hi = AI.quantile(sorted, 0.95) || lo + 1, C = AI.col(), A = AI.rgb(C.bg), B = AI.rgb(C.purple), Cc = AI.rgb(C.orange);
    for (let k = 0; k < vals.length; k++) {
      const tt = AI.clamp(Math.pow(AI.clamp((vals[k] - lo) / (hi - lo || 1), 0, 1), 0.45), 0, 1), band = (tt * 14) % 1, edge = band < 0.12 ? 0.18 : 0;
      for (let ch = 0; ch < 3; ch++) im.data[k * 4 + ch] = Math.min(255, (tt < 0.5 ? A[ch] + (B[ch] - A[ch]) * tt * 2 : B[ch] + (Cc[ch] - B[ch]) * (tt - 0.5) * 2) * (0.85 + edge) + edge * 60);
      im.data[k * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }

  /* ===================================================================
     1. Landscapes: critical points + Hessian classification
     =================================================================== */
  AI.labs.landscapes = function (root) {
    const { view } = AI.labShell(root, 'Loss landscape explorer', 'edit f(x, y); critical points are found and classified from the Hessian eigenvalues',
      `<canvas style="height:430px;cursor:crosshair;touch-action:none"></canvas><div class="caption" id="cap">Click the map to probe a point, then roll a ball downhill from it.</div>`,
      `<label>f(x, y) = <input type="text" id="expr" value="x^3/3 - x + y^2" spellcheck="false"></label><div class="err" id="err"></div>
       <div class="chips" id="presets"></div>
       ${sl('sp', 'Ball speed (steps / second)', 1, 20, 1, 4)}
       <div class="row">${btn('roll', '▶ Roll a ball from the probe', 'btn primary')}${btn('play', '⏸ Pause')}${btn('clr', 'Clear')}</div>
       <div class="legend"><span><i style="background:var(--green)"></i>minimum</span><span><i style="background:var(--red)"></i>maximum</span><span><i style="background:var(--orange)"></i>saddle</span></div>
       <div class="readout" id="out"></div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv), off = document.createElement('canvas'), st = { playing: true };
    const PRE = {
      'Hill + valley': { e: 'x^3/3 - x + y^2', v: [-3, 3, -2.5, 2.5], p: [-0.4, 1.6], lr: 0.12 },
      Bowl: { e: 'x^2 + 2*y^2', v: [-3, 3, -3, 3], p: [2.4, 2], lr: 0.2 },
      Saddle: { e: 'x^2 - y^2', v: [-3, 3, -3, 3], p: [2, 0.4], lr: 0.1 },
      'Sum of squares?': { e: 'x^2 + 4*x*y + y^2', v: [-3, 3, -3, 3], p: [2, -0.4], lr: 0.06 },
      'Egg box': { e: 'sin(x)*sin(y)', v: [-3.5, 3.5, -3.5, 3.5], p: [0.6, 1.1], lr: 0.3 },
      Himmelblau: { e: '(x^2+y-11)^2 + (x+y^2-7)^2', v: [-5, 5, -5, 5], p: [0.5, 0.5], lr: 0.01 },
      'Flat in y': { e: 'x^2 + y^4', v: [-2, 2, -2, 2], p: [1.5, 1.4], lr: 0.1 },
    };
    let fn = null, V = PRE['Hill + valley'].v, probe = [...PRE['Hill + valley'].p], lr = PRE['Hill + valley'].lr, crit = [], ball = null, dirty = true, imgDirty = true, key = '';
    const tick = AI.ticker();
    const F = (x, y) => fn({ x, y });
    const G = (x, y) => { const h = 1e-4; return [(F(x + h, y) - F(x - h, y)) / (2 * h), (F(x, y + h) - F(x, y - h)) / (2 * h)]; };
    const Hs = (x, y) => { const h = 1e-3, f0 = F(x, y); return { a: (F(x + h, y) - 2 * f0 + F(x - h, y)) / (h * h), d: (F(x, y + h) - 2 * f0 + F(x, y - h)) / (h * h), b: (F(x + h, y + h) - F(x + h, y - h) - F(x - h, y + h) + F(x - h, y - h)) / (4 * h * h) }; };
    const TOL = 2e-3;
    const classify = (e) => (e.l1 > TOL && e.l2 > TOL ? 'min' : e.l1 < -TOL && e.l2 < -TOL ? 'max' : e.l1 > TOL && e.l2 < -TOL ? 'saddle' : 'flat');
    const NAME = { min: 'minimum (bowl: curves up in every direction)', max: 'maximum (cap: curves down in every direction)', saddle: 'saddle (up in one direction, down in another)', flat: 'degenerate (a curvature is ≈ 0: Hessian cannot decide)' };
    const COLOR = { min: 'green', max: 'red', saddle: 'orange', flat: 'muted' };
    const findCrit = () => {
      crit = []; const [x0, x1, y0, y1] = V, n = 10;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let p = [x0 + ((i + 0.5) / n) * (x1 - x0), y0 + ((j + 0.5) / n) * (y1 - y0)], ok = false;
        for (let it = 0; it < 60; it++) {
          const g = G(p[0], p[1]);
          if (!g.every(Number.isFinite)) break;
          if (Math.hypot(g[0], g[1]) < 1e-6) { ok = true; break; }
          const H = Hs(p[0], p[1]), det = H.a * H.d - H.b * H.b;
          if (!Number.isFinite(det) || Math.abs(det) < 1e-12) break;
          let dx = -(H.d * g[0] - H.b * g[1]) / det, dy = -(-H.b * g[0] + H.a * g[1]) / det;
          const L = Math.hypot(dx, dy); if (L > 1) { dx /= L; dy /= L; }
          p = [p[0] + dx, p[1] + dy];
        }
        if (!ok || p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1) continue;
        if (crit.some((k) => Math.hypot(k.x - p[0], k.y - p[1]) < 2e-3)) continue;
        const e = eig2(Hs(p[0], p[1])); crit.push({ x: p[0], y: p[1], e, kind: classify(e) });
      }
    };
    const ei = AI.exprInput(q(root, '#expr'), q(root, '#err'), {
      allowed: ['x', 'y'], check: (f) => { if (!Number.isFinite(f({ x: 0.5, y: 0.5 }))) throw new Error('f(x, y) is not finite'); },
      onChange: (f) => { fn = f; imgDirty = true; dirty = true; ball = null; findCrit(); },
    });
    mkChips(q(root, '#presets'), Object.entries(PRE).map(([k, p]) => [k, p]), (p) => { V = p.v; probe = [...p.p]; lr = p.lr; ei.set(p.e); });
    ei.run();
    q(root, '#roll').onclick = () => { ball = { p: [...probe], path: [[...probe]], k: 0, done: false }; setPlay(root, st, true); dirty = true; };
    q(root, '#clr').onclick = () => { ball = null; dirty = true; };
    playBtn(root, st);
    drag(cv, (px, py) => { const [x0, x1, y0, y1] = V; probe = [x0 + (px / c.w) * (x1 - x0), y1 - (py / c.h) * (y1 - y0)]; ball = null; dirty = true; });
    c.onresize = () => { imgDirty = true; dirty = true; };
    window.addEventListener('themechange', () => { imgDirty = true; dirty = true; });

    AI.loop(root, (dt) => {
      q(root, '#spv').textContent = q(root, '#sp').value;
      if (ball && st.playing && !ball.done) {
        const n = tick(dt, +q(root, '#sp').value);
        for (let i = 0; i < n && !ball.done; i++) {
          const g = G(ball.p[0], ball.p[1]); let sx = lr * g[0], sy = lr * g[1]; const L = Math.hypot(sx, sy), cap = (V[1] - V[0]) * 0.06;
          if (L > cap) { sx *= cap / L; sy *= cap / L; }
          ball.p = [ball.p[0] - sx, ball.p[1] - sy]; ball.k++; ball.path.push([...ball.p]);
          if (Math.hypot(...g) < 1e-3 || ball.k > 400 || ball.p[0] < V[0] || ball.p[0] > V[1] || ball.p[1] < V[2] || ball.p[1] > V[3]) ball.done = true;
        }
        dirty = true;
      }
      if (imgDirty) { paintHeat(off, c, fn, V); imgDirty = false; }
      if (!dirty) return;
      dirty = false;
      const C = AI.col(), g = c.ctx, [x0, x1, y0, y1] = V, X = (x) => ((x - x0) / (x1 - x0)) * c.w, Y = (y) => ((y1 - y) / (y1 - y0)) * c.h;
      g.clearRect(0, 0, c.w, c.h); g.drawImage(off, 0, 0, c.w, c.h);
      for (const k of crit) {
        const px = X(k.x), py = Y(k.y), col = C[COLOR[k.kind]];
        g.fillStyle = col; g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath();
        if (k.kind === 'saddle') { g.moveTo(px, py - 8); g.lineTo(px + 8, py); g.lineTo(px, py + 8); g.lineTo(px - 8, py); g.closePath(); } else if (k.kind === 'max') g.rect(px - 6, py - 6, 12, 12); else g.arc(px, py, 7, 0, 7);
        g.fill(); g.stroke();
      }
      // probe + curvature axes
      const H = Hs(probe[0], probe[1]), e = eig2(H), len = 46;
      const dirs = [[Math.cos(e.th), Math.sin(e.th), e.l1], [-Math.sin(e.th), Math.cos(e.th), e.l2]];
      for (const [dx, dy, lam] of dirs) {
        g.strokeStyle = lam >= 0 ? C.blue : C.orange; g.lineWidth = 3; g.beginPath(); g.moveTo(X(probe[0]) - dx * len, Y(probe[1]) + dy * len); g.lineTo(X(probe[0]) + dx * len, Y(probe[1]) - dy * len); g.stroke();
      }
      dot(g, X(probe[0]), Y(probe[1]), 6, C.fg, C.bg);
      if (ball) {
        g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); ball.path.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.stroke();
        g.fillStyle = C.fg; ball.path.forEach(([x, y]) => { g.beginPath(); g.arc(X(x), Y(y), 2.2, 0, 7); g.fill(); });
        dot(g, X(ball.p[0]), Y(ball.p[1]), 7, C.yellow, '#fff');
      }
      txt(g, 'blue bar: curves UP along this direction   orange bar: curves DOWN', 10, 16, C.fg, '11px sans-serif');
      const gr = G(probe[0], probe[1]), kind = classify(e);
      q(root, '#cap').textContent = ball ? `Ball: step ${ball.k}, f = ${num(F(ball.p[0], ball.p[1]), 3)}${ball.done ? ' — settled (gradient ≈ 0) or left the map' : ' — each dot is one gradient step'}` : 'Click the map to probe a point, then roll a ball downhill from it.';
      const cnt = { min: 0, max: 0, saddle: 0, flat: 0 }; crit.forEach((k) => cnt[k.kind]++);
      q(root, '#out').innerHTML = `probe (${num(probe[0], 2)}, ${num(probe[1], 2)})   f = ${num(F(probe[0], probe[1]), 3)}\n∇f = (${num(gr[0], 3)}, ${num(gr[1], 3)})\nHessian = [[${num(H.a, 3)}, ${num(H.b, 3)}],[${num(H.b, 3)}, ${num(H.d, 3)}]]\nλ₁ = <span class="b">${num(e.l1, 3)}</span>   λ₂ = <span class="o">${num(e.l2, 3)}</span>\n<span class="m">${Math.hypot(...gr) < 0.05 ? 'gradient ≈ 0 here → ' + NAME[kind] : 'gradient ≠ 0: not a critical point; Hessian shows local shape'}</span>\n\ncritical points in view: ${cnt.min} min, ${cnt.max} max, ${cnt.saddle} saddle${cnt.flat ? ', ' + cnt.flat + ' degenerate' : ''}`;
    }, { fps: 30 });
  };

  /* ===================================================================
     2. Convexity: the chord test
     =================================================================== */
  AI.labs.convexity = function (root) {
    const { view } = AI.labShell(root, 'Convex or not? The chord test', 'a function is convex when every chord lies on or above its graph',
      `<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>`,
      `<label>f(x) = <input type="text" id="expr" value="x^2" spellcheck="false"></label><div class="err" id="err"></div>
       <div class="chips" id="presets"></div>
       ${sl('a', 'Chord end a', -3, 3, 0.05, -2)}${sl('b', 'Chord end b', -3, 3, 0.05, 2)}
       ${sl('sp', 'Slide speed', 0.1, 1.5, 0.05, 0.3)}
       <div class="row">${btn('play', '⏸ Pause')}${btn('rand', 'Test 400 random chords', 'btn primary')}</div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), st = { playing: true };
    let fn = null, t = 0, dirn = 1, yr = [-1, 5], test = null;
    const ei = AI.exprInput(q(root, '#expr'), q(root, '#err'), { allowed: ['x'], check: (f) => { if (!Number.isFinite(f({ x: 0.7 }))) throw new Error('not finite'); }, onChange: (f) => { fn = f; test = null; } });
    mkChips(q(root, '#presets'), [['x^2 (convex)', 'x^2'], ['|x| (convex, kink)', 'abs(x)'], ['exp(x) (convex)', 'exp(x)'], ['double well (non-convex)', 'x^4/4 - x^2 + 0.3*x'], ['sin(x) (non-convex)', '1.5*sin(x)'], ['−x^2 (concave)', '-x^2'], ['relu(x) (convex)', 'relu(x)']], (p) => { ei.set(p); });
    ei.run(); playBtn(root, st);
    const F = (x) => fn({ x });
    q(root, '#rand').onclick = () => {
      let bad = 0, worst = 0;
      for (let i = 0; i < 400; i++) { const a = -3 + Math.random() * 6, b = -3 + Math.random() * 6, s = Math.random(), x = (1 - s) * a + s * b, gap = F(x) - ((1 - s) * F(a) + s * F(b)); if (gap > 1e-9) { bad++; worst = Math.max(worst, gap); } }
      test = bad ? `${bad} of 400 random chords dip BELOW the graph (worst gap ${num(worst, 3)}) → NOT convex` : 'all 400 chords lie on/above the graph → consistent with convex';
    };
    AI.loop(root, (dt) => {
      const a = +q(root, '#a').value, b = +q(root, '#b').value; q(root, '#av').textContent = num(a, 2); q(root, '#bv').textContent = num(b, 2); q(root, '#spv').textContent = q(root, '#sp').value;
      if (st.playing) { t += dirn * dt * +q(root, '#sp').value; if (t > 1) { t = 1; dirn = -1; } if (t < 0) { t = 0; dirn = 1; } }
      let lo = Infinity, hi = -Infinity; for (let i = 0; i <= 80; i++) { const y = F(-3.5 + (7 * i) / 80); if (Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }
      lo = Math.max(lo, -30); hi = Math.min(hi, 30); const pd = (hi - lo) * 0.15 + 0.4; yr[0] += (lo - pd - yr[0]) * 0.2; yr[1] += (hi + pd - yr[1]) * 0.2;
      const C = AI.col(), g = c.ctx, v = AI.view(c, -3.5, 3.5, yr[0], yr[1]);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const fa = F(a), fb = F(b);
      // shaded gap between chord and curve (red where curve is above chord)
      const lo2 = Math.min(a, b), hi2 = Math.max(a, b);
      for (let i = 0; i < 160; i++) {
        const x = lo2 + ((hi2 - lo2) * (i + 0.5)) / 160, s = hi2 === lo2 ? 0 : (x - a) / (b - a), ch = (1 - s) * fa + s * fb, fx = F(x);
        g.fillStyle = fx > ch + 1e-9 ? C.red : C.green; g.globalAlpha = 0.22; g.fillRect(v.X(x - (hi2 - lo2) / 320), v.Y(Math.max(fx, ch)), (v.X(hi2) - v.X(lo2)) / 160 + 1, Math.abs(v.Y(fx) - v.Y(ch)));
      }
      g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); let pen = false;
      for (let i = 0; i <= 300; i++) { const x = -3.5 + (7 * i) / 300, y = F(x); if (!Number.isFinite(y)) { pen = false; continue; } pen ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); pen = true; }
      g.stroke();
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(a), v.Y(fa)); g.lineTo(v.X(b), v.Y(fb)); g.stroke();
      const xt = (1 - t) * a + t * b, ft = F(xt), ct = (1 - t) * fa + t * fb, ok = ft <= ct + 1e-9;
      g.strokeStyle = ok ? C.green : C.red; g.lineWidth = 3; g.beginPath(); g.moveTo(v.X(xt), v.Y(ft)); g.lineTo(v.X(xt), v.Y(ct)); g.stroke();
      dot(g, v.X(a), v.Y(fa), 5, C.fg); dot(g, v.X(b), v.Y(fb), 5, C.fg); dot(g, v.X(xt), v.Y(ft), 6, C.blue, '#fff'); dot(g, v.X(xt), v.Y(ct), 6, C.orange, '#fff');
      g.restore();
      txt(g, 'blue = f(x)   orange = chord   green gap: chord above (good)   red gap: curve above chord (not convex)', v.pad.l + 6, v.pad.t + 14, C.fg, '11px sans-serif');
      q(root, '#cap').textContent = `Point on the segment x = (1−t)a + tb with t = ${num(t, 2)}:   f(x) = ${num(ft, 3)}   chord = (1−t)f(a) + t f(b) = ${num(ct, 3)}   →   ${ok ? 'f(x) ≤ chord ✓' : 'f(x) > chord ✗ (convexity broken)'}`;
      // sampled verdict on [a,b]
      let viol = 0; for (let i = 0; i < 200; i++) { const s = i / 199, x = (1 - s) * a + s * b; if (F(x) > (1 - s) * fa + s * fb + 1e-9) viol++; }
      q(root, '#out').innerHTML = `this chord (a=${num(a, 2)}, b=${num(b, 2)}): ${viol ? `<span class="r">${viol}/200 sample points above the chord</span>` : '<span class="g">curve stays below the chord</span>'}\n${test ? test : '<span class="m">press “Test 400 random chords” for a wider check</span>'}\n<span class="m">One failing chord is a proof of non-convexity;\npassing chords never prove convexity.</span>`;
    }, { fps: 30 });
  };

  /* ===================================================================
     3a. Gradient descent in 1D: tangent, step, move
     =================================================================== */
  AI.labs.gd1d = function (root) {
    const { view } = AI.labShell(root, 'Gradient descent in one dimension', 'f(w) = ½·L·(w − 3)² + 1 — drag on the plot to choose the start; every step is shown in three phases',
      `<canvas style="height:400px;cursor:ew-resize;touch-action:none"></canvas><div class="caption" id="cap" style="min-height:3em"></div>`,
      `${sl('L', 'Curvature L (= f″)', 0.5, 5, 0.5, 2)}${sl('eta', 'Learning rate η', 0.01, 2, 0.01, 0.1)}
       ${sl('sp', 'Speed ×', 0.25, 4, 0.25, 1)}
       <div class="chips" id="presets"></div>
       <div class="row">${btn('play', '⏸ Pause')}${btn('step', 'Next phase')}${btn('reset', 'Restart')}</div>
       <div class="readout" id="out"></div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv), st = { playing: true };
    const MAXK = 40, DUR = [1.1, 1.1, 1.3], X0 = -4, X1 = 10;
    let w0 = 0, w = 0, wn = 0, k = 0, phase = 0, t = 0, trail = [], over = '', lastCap = '';
    const L = () => +q(root, '#L').value, eta = () => +q(root, '#eta').value;
    const f = (x) => 0.5 * L() * (x - 3) ** 2 + 1, df = (x) => L() * (x - 3);
    const reset = () => { w = w0; wn = w0; k = 0; phase = 0; t = 0; trail = [w0]; over = ''; setPlay(root, st, true); };
    mkChips(q(root, '#presets'), [['too small (ηL = 0.05)', 0.05], ['steady (ηL = 0.5)', 0.5], ['perfect (ηL = 1)', 1], ['overshoot (ηL = 1.7)', 1.7], ['diverges (ηL = 2.1)', 2.1]], (r) => { q(root, '#eta').value = AI.clamp(r / L(), 0.01, 2); reset(); });
    q(root, '#reset').onclick = reset; playBtn(root, st);
    q(root, '#step').onclick = () => { setPlay(root, st, false); t = DUR[phase]; };
    q(root, '#L').oninput = reset; q(root, '#eta').oninput = reset;
    drag(cv, (px) => { const v = AI.view(c, X0, X1, 0, 1); w0 = AI.clamp(v.ix(px), X0 + 0.3, X1 - 0.3); reset(); });
    reset();
    AI.loop(root, (dt) => {
      q(root, '#Lv').textContent = num(L(), 1); q(root, '#etav').textContent = num(eta(), 2); q(root, '#spv').textContent = '×' + q(root, '#sp').value;
      if (!over && (st.playing || t >= DUR[phase])) {
        if (st.playing) t += dt * +q(root, '#sp').value;
        if (t >= DUR[phase]) {
          t = 0;
          if (phase === 0) { phase = 1; } else if (phase === 1) { wn = w - eta() * df(w); phase = 2; } else { w = wn; k++; trail.push(w); phase = 0; if (k >= MAXK) { over = `stopped after ${MAXK} steps`; setPlay(root, st, false); } if (Math.abs(w) > 1e4) { over = 'diverged — the iterate is flying away'; setPlay(root, st, false); } }
        }
      }
      const slope = df(w), step = -eta() * slope;
      if (phase === 1 && t === 0) wn = w + step;
      const C = AI.col(), g = c.ctx, ymax = 0.5 * L() * 25 + 1, v = AI.view(c, X0, X1, -ymax * 0.06, ymax);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath();
      for (let i = 0; i <= 240; i++) { const x = X0 + ((X1 - X0) * i) / 240; i ? g.lineTo(v.X(x), v.Y(f(x))) : g.moveTo(v.X(x), v.Y(f(x))); } g.stroke();
      // minimum marker
      dot(g, v.X(3), v.Y(1), 4, C.green);
      // trail
      trail.forEach((x, i) => { if (x > X0 && x < X1) { dot(g, v.X(x), v.Y(f(x)), i === trail.length - 1 ? 3 : 3.5, C.muted); if (i < 12) txt(g, String(i), v.X(x) + 5, v.Y(f(x)) - 7, C.muted, '10px sans-serif'); } });
      const wb = phase === 2 ? AI.lerp(w, wn, AI.smooth(t / DUR[2])) : w, yb = f(wb);
      if (phase <= 1) {
        const al = phase === 0 ? AI.smooth(t / 0.8) : 1;
        g.globalAlpha = al; g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(X0), v.Y(f(w) + slope * (X0 - w))); g.lineTo(v.X(X1), v.Y(f(w) + slope * (X1 - w))); g.stroke(); g.globalAlpha = 1;
      }
      if (phase >= 1) {
        const p = phase === 1 ? AI.smooth(t / DUR[1]) : 1, y0 = v.Y(0) - 10;
        AI.arrow(g, v.X(w), y0, v.X(w + step * p), y0, C.orange, 4);
        g.setLineDash([3, 4]); g.strokeStyle = C.orange; g.lineWidth = 1; g.beginPath(); g.moveTo(v.X(w), v.Y(f(w))); g.lineTo(v.X(w), y0); g.stroke(); g.setLineDash([]);
      }
      dot(g, v.X(wb), v.Y(yb), 8, C.yellow, '#fff');
      g.restore();
      txt(g, 'yellow ball = current w   red line = tangent (slope f′)   orange arrow = step', v.pad.l + 6, v.pad.t + 14, C.fg, '11px sans-serif');
      const cap = over || (phase === 0 ? `① Measure the slope at w = ${num(w, 4)}: f′(w) = L(w − 3) = ${num(slope, 4)} (${slope > 0 ? 'uphill to the right' : 'uphill to the left'}).` : phase === 1 ? `② Step = −η·f′(w) = −${num(eta(), 2)} × ${num(slope, 4)} = ${num(step, 4)}  (go opposite to the slope).` : `③ Move: w ← ${num(w, 4)} + (${num(step, 4)}) = ${num(wn, 4)};  loss ${num(f(w), 4)} → ${num(f(wn), 4)}.`);
      const cap2 = `step ${k}  ${cap}`;
      if (cap2 !== lastCap) { q(root, '#cap').textContent = cap2; lastCap = cap2; }
      const r = eta() * L();
      q(root, '#out').innerHTML = `w = ${num(w, 5)}   f(w) = ${num(f(w), 5)}\nηL = <b>${num(r, 3)}</b>  →  error multiplies by (1 − ηL) = <span class="o">${num(1 - r, 3)}</span> per step\n${r < 2 ? '<span class="g">η &lt; 2/L = ' + num(2 / L(), 3) + ': converges' + (r > 1 ? ' (zig-zagging over the minimum)' : '') + '</span>' : '<span class="r">η &gt; 2/L = ' + num(2 / L(), 3) + ': diverges</span>'}`;
    }, { fps: 30 });
  };

  /* ===================================================================
     3b. Gradient descent on a 2-D quadratic: conditioning and zig-zag
     =================================================================== */
  AI.labs.gdquad = function (root) {
    const { view } = AI.labShell(root, 'Zig-zag and the condition number', 'f(x, y) = ½(x² + κy²): eigenvalues 1 and κ, so the condition number is κ. Click the map to restart elsewhere.',
      `<canvas style="height:420px;cursor:crosshair;touch-action:none"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>`,
      `${sl('kap', 'Condition number κ', 1, 60, 1, 10)}${sl('fr', 'Step size as a fraction of 2/L', 0.05, 1.1, 0.01, 0.9)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="best"> use the best fixed step η = 2/(1+κ)</label>
       ${sl('sp', 'Speed (steps / second)', 1, 30, 1, 3)}
       <div class="row">${btn('play', '⏸ Pause')}${btn('step', 'One step')}${btn('reset', 'Restart')}</div>
       <div class="readout" id="out"></div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv), st = { playing: true }, tick = AI.ticker();
    let start = [8, 2.4], p = [...start], path = [[...start]], k = 0, hist = [], frozen = false;
    const kap = () => +q(root, '#kap').value, eta = () => (q(root, '#best').checked ? 2 / (1 + kap()) : (+q(root, '#fr').value * 2) / kap());
    const f = (x, y) => 0.5 * (x * x + kap() * y * y);
    const reset = () => { p = [...start]; path = [[...start]]; k = 0; hist = []; frozen = false; };
    const doStep = () => {
      if (frozen) return;
      const g = [p[0], kap() * p[1]], e = eta(); hist = [{ g, e, from: [...p] }];
      p = [p[0] - e * g[0], p[1] - e * g[1]]; k++; path.push([...p]); if (path.length > 500) path.shift();
      if (!p.every(Number.isFinite) || Math.hypot(...p) > 1e7) frozen = true;
    };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { setPlay(root, st, false); doStep(); };
    ['kap', 'fr', 'best'].forEach((id) => (q(root, '#' + id).oninput = reset)); playBtn(root, st);
    drag(cv, (px, py) => { const m = iso(c, -10, 10, -6, 6); start = [AI.clamp(m.ix(px), -9.5, 9.5), AI.clamp(m.iy(py), -5.5, 5.5)]; reset(); });
    AI.loop(root, (dt) => {
      q(root, '#kapv').textContent = kap(); q(root, '#frv').textContent = num(+q(root, '#fr').value, 2); q(root, '#spv').textContent = q(root, '#sp').value;
      if (st.playing) { const n = tick(dt, +q(root, '#sp').value); for (let i = 0; i < n; i++) doStep(); }
      const C = AI.col(), g = c.ctx, m = iso(c, -10, 10, -6, 6), f0 = f(...start);
      g.clearRect(0, 0, c.w, c.h);
      for (let j = 0; j < 9; j++) { // contours: f = f0 * 0.45^j
        const lv = f0 * Math.pow(0.45, j), a = Math.sqrt(2 * lv), b = Math.sqrt((2 * lv) / kap());
        g.strokeStyle = C.purple; g.globalAlpha = 0.25 + 0.5 * (1 - j / 9); g.lineWidth = 1.5; g.beginPath();
        for (let i = 0; i <= 90; i++) { const th = (i / 90) * 2 * Math.PI; i ? g.lineTo(m.X(a * Math.cos(th)), m.Y(b * Math.sin(th))) : g.moveTo(m.X(a * Math.cos(th)), m.Y(b * Math.sin(th))); }
        g.stroke();
      }
      g.globalAlpha = 1;
      dot(g, m.X(0), m.Y(0), 5, C.green);
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); path.forEach(([x, y], i) => (i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)))); g.stroke();
      g.fillStyle = C.orange; path.forEach(([x, y]) => { g.beginPath(); g.arc(m.X(x), m.Y(y), 2.6, 0, 7); g.fill(); });
      dot(g, m.X(start[0]), m.Y(start[1]), 5, C.fg);
      if (!frozen) { const gr = [p[0], kap() * p[1]], n = Math.hypot(...gr) || 1; AI.arrow(g, m.X(p[0]), m.Y(p[1]), m.X(p[0]) - (gr[0] / n) * 34, m.Y(p[1]) + (gr[1] / n) * 34, C.red, 2.5); dot(g, m.X(p[0]), m.Y(p[1]), 6.5, C.yellow, '#fff'); }
      txt(g, 'purple = level sets of f   red arrow = −∇f at the ball   green dot = minimum', 10, 16, C.fg, '11px sans-serif');
      const e = eta(), kk = kap(), rho = Math.max(Math.abs(1 - e), Math.abs(1 - e * kk)), need = rho < 1 ? Math.ceil(Math.log(1e-3) / Math.log(rho)) : Infinity;
      const h = hist[0];
      q(root, '#cap').textContent = h ? `step ${k}: ∇f = (${num(h.g[0], 3)}, ${num(h.g[1], 3)}) at (${num(h.from[0], 3)}, ${num(h.from[1], 3)}) → x ← x − η·∇f with η = ${num(h.e, 4)}  →  (${num(p[0], 3)}, ${num(p[1], 3)})` : `Press play. The ball starts at (${num(start[0], 2)}, ${num(start[1], 2)}).`;
      q(root, '#out').innerHTML = `η = <b>${num(e, 4)}</b>   (2/L = ${num(2 / kk, 4)}, L = κ = ${kk})\nper-coordinate factors: x: |1−η| = ${num(Math.abs(1 - e), 3)}   y: |1−ηκ| = ${num(Math.abs(1 - e * kk), 3)}\nrate ρ = max = <span class="o">${num(rho, 4)}</span>   best possible (κ−1)/(κ+1) = ${num((kk - 1) / (kk + 1), 4)}\nsteps to shrink error ×1000: ${Number.isFinite(need) ? '≈ ' + need : '<span class="r">never — it diverges</span>'}\nf = ${num(f(p[0], p[1]), 4)}`;
    }, { fps: 30 });
  };

  /* ===================================================================
     3c. Learning-rate regimes: too small / right / too large / diverging
     =================================================================== */
  AI.labs.gdlr = function (root) {
    const { view } = AI.labShell(root, 'Four learning rates on the same parabola', 'f(w) = ½·L·w², start w₀ = 1; w ← (1 − ηL)·w, so only the product r = ηL matters',
      `<canvas style="height:340px"></canvas><div class="caption" id="cap"></div>`,
      `${sl('r', 'Your own r = ηL (white curve)', 0.05, 2.5, 0.05, 1.5)}${sl('sp', 'Speed (steps / second)', 1, 20, 1, 3)}
       <div class="legend"><span><i style="background:var(--blue)"></i>r = 0.1 too small</span><span><i style="background:var(--green)"></i>r = 0.9 good</span><span><i style="background:var(--orange)"></i>r = 1.9 oscillates, still converges</span><span><i style="background:var(--red)"></i>r = 2.1 diverges</span></div>
       <div class="row">${btn('play', '⏸ Pause')}${btn('step', 'One step')}${btn('reset', 'Restart')}</div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), st = { playing: true }, tick = AI.ticker(), N = 30;
    const runs = [['blue', 0.1], ['green', 0.9], ['orange', 1.9], ['red', 2.1], ['fg', 1.5]];
    let k = 0, W = runs.map(() => [1]);
    const reset = () => { k = 0; W = runs.map(() => [1]); };
    const adv = () => { if (k >= N) return; k++; runs.forEach(([, r0], i) => { const r = i === 4 ? +q(root, '#r').value : r0; W[i].push(W[i][k - 1] * (1 - r)); }); };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { setPlay(root, st, false); adv(); }; q(root, '#r').oninput = reset; playBtn(root, st);
    AI.loop(root, (dt) => {
      q(root, '#rv').textContent = num(+q(root, '#r').value, 2); q(root, '#spv').textContent = q(root, '#sp').value;
      if (st.playing) { const n = tick(dt, +q(root, '#sp').value); for (let i = 0; i < n; i++) adv(); }
      const C = AI.col(), g = c.ctx, v = AI.view(c, 0, N, -1.6, 1.6);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(0), v.Y(0)); g.lineTo(v.X(N), v.Y(0)); g.stroke(); g.setLineDash([]);
      runs.forEach(([col], i) => {
        const w = W[i]; g.strokeStyle = C[col]; g.fillStyle = C[col]; g.lineWidth = i === 4 ? 3 : 2; g.beginPath();
        w.forEach((y, j) => (j ? g.lineTo(v.X(j), v.Y(y)) : g.moveTo(v.X(j), v.Y(y)))); g.stroke();
        w.forEach((y, j) => { g.beginPath(); g.arc(v.X(j), v.Y(y), 2.8, 0, 7); g.fill(); });
      });
      g.restore();
      txt(g, 'w_k against step k (the minimum is w = 0)', v.pad.l + 6, v.pad.t + 14, C.fg, '11px sans-serif');
      q(root, '#cap').textContent = `step ${k} of ${N}: each step multiplies w by (1 − r). |1 − r| < 1 → shrink; 1 − r < 0 → flip sides (zig-zag); |1 − r| > 1 → grow.`;
      q(root, '#out').innerHTML = runs.map(([col, r0], i) => { const r = i === 4 ? +q(root, '#r').value : r0, w = W[i][k]; return `<span class="${col === 'blue' ? 'b' : col === 'green' ? 'g' : col === 'orange' ? 'o' : col === 'red' ? 'r' : ''}">r=${num(r, 2)}</span>  factor ${num(1 - r, 2).padStart(5)}  w_${k} = ${num(w, 4)}`; }).join('\n');
    }, { fps: 30 });
  };

  /* ===================================================================
     4. SGD: full batch vs mini-batch on a dataset whose mean loss we minimise
     =================================================================== */
  AI.labs.sgdlab = function (root) {
    const N = 24;
    const { view } = AI.labShell(root, 'Full-batch gradient vs noisy mini-batches', 'each data point a_i defines a loss ½(θ−a_i)ᵀD(θ−a_i); we minimise the MEAN loss, whose minimum is the mean ā of the points',
      `<canvas style="height:430px"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>`,
      `${sl('B', 'Batch size B (of 24)', 1, N, 1, 2)}${sl('eta', 'Learning rate η', 0.02, 0.6, 0.01, 0.18)}${sl('spread', 'Data spread σ', 0.3, 2.2, 0.1, 1.2)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="decay"> decay η as 1/(1 + 0.03·k)</label>
       ${sl('sp', 'Speed (steps / second)', 1, 20, 1, 3)}
       <div class="legend"><span><i style="background:var(--blue)"></i>full batch</span><span><i style="background:var(--orange)"></i>SGD runs (3 random seeds)</span></div>
       <div class="row">${btn('play', '⏸ Pause')}${btn('step', 'One step')}${btn('reset', 'Restart')}</div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), st = { playing: true }, tick = AI.ticker(), r0 = AI.rng(11), Z = [];
    for (let i = 0; i < N; i++) Z.push([AI.randn(r0), AI.randn(r0)]);
    const D = [1, 3], TH0 = [-4.6, 3.1], COLS = ['orange', 'pink', 'purple'];
    let A = [], abar = [0, 0], trC = 0, full, runs, k = 0, lastBatch = [], errSum = 0, errN = 0, lastStep = null;
    const build = () => { const s = +q(root, '#spread').value; A = Z.map(([a, b]) => [s * a + 0.4, s * 0.8 * b - 0.3]); abar = [A.reduce((u, a) => u + a[0], 0) / N, A.reduce((u, a) => u + a[1], 0) / N]; trC = A.reduce((u, a) => u + D[0] ** 2 * (a[0] - abar[0]) ** 2 + D[1] ** 2 * (a[1] - abar[1]) ** 2, 0) / N; };
    const loss = (t) => A.reduce((u, a) => u + 0.5 * (D[0] * (t[0] - a[0]) ** 2 + D[1] * (t[1] - a[1]) ** 2), 0) / N;
    const reset = () => { build(); full = { p: [...TH0], path: [[...TH0]] }; runs = COLS.map((c0) => ({ c: c0, p: [...TH0], path: [[...TH0]] })); k = 0; errSum = 0; errN = 0; lastBatch = []; lastStep = null; };
    const pick = (B) => { const idx = [...Array(N).keys()]; for (let i = 0; i < B; i++) { const j = i + Math.floor(Math.random() * (N - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; } return idx.slice(0, B); };
    const doStep = () => {
      const B = +q(root, '#B').value, eta = +q(root, '#eta').value * (q(root, '#decay').checked ? 1 / (1 + 0.03 * k) : 1);
      const gf = [D[0] * (full.p[0] - abar[0]), D[1] * (full.p[1] - abar[1])];
      full.p = [full.p[0] - eta * gf[0], full.p[1] - eta * gf[1]]; full.path.push([...full.p]);
      runs.forEach((r, ri) => {
        const tg = [D[0] * (r.p[0] - abar[0]), D[1] * (r.p[1] - abar[1])], idx = pick(B); let mx = 0, my = 0; idx.forEach((i) => { mx += A[i][0]; my += A[i][1]; }); mx /= B; my /= B;
        const gb = [D[0] * (r.p[0] - mx), D[1] * (r.p[1] - my)]; errSum += (gb[0] - tg[0]) ** 2 + (gb[1] - tg[1]) ** 2; errN++;
        if (ri === 0) { lastBatch = idx; lastStep = { from: [...r.p], tg, gb, eta, m: [mx, my] }; }
        r.p = [r.p[0] - eta * gb[0], r.p[1] - eta * gb[1]]; r.path.push([...r.p]); if (r.path.length > 400) r.path.shift();
      });
      if (full.path.length > 400) full.path.shift(); k++;
    };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { setPlay(root, st, false); doStep(); }; ['spread'].forEach((id) => (q(root, '#' + id).oninput = reset)); playBtn(root, st);
    reset();
    AI.loop(root, (dt) => {
      const B = +q(root, '#B').value; q(root, '#Bv').textContent = B; q(root, '#etav').textContent = num(+q(root, '#eta').value, 2); q(root, '#spreadv').textContent = num(+q(root, '#spread').value, 1); q(root, '#spv').textContent = q(root, '#sp').value;
      if (st.playing) { const n = tick(dt, +q(root, '#sp').value); for (let i = 0; i < n; i++) doStep(); }
      const C = AI.col(), g = c.ctx, m = iso(c, -7, 7, -5, 5);
      g.clearRect(0, 0, c.w, c.h);
      [0.4, 1.2, 2.8, 5.5, 10, 18, 30].forEach((lv, j) => {
        g.strokeStyle = C.purple; g.globalAlpha = 0.65 - j * 0.06; g.lineWidth = 1.3; g.beginPath(); const a = Math.sqrt(2 * lv / D[0]), b = Math.sqrt(2 * lv / D[1]);
        for (let i = 0; i <= 80; i++) { const th = (i / 80) * 2 * Math.PI, x = m.X(abar[0] + a * Math.cos(th)), y = m.Y(abar[1] + b * Math.sin(th)); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke();
      });
      g.globalAlpha = 1;
      A.forEach((a, i) => dot(g, m.X(a[0]), m.Y(a[1]), 3.2, C.muted));
      lastBatch.forEach((i) => { g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); g.arc(m.X(A[i][0]), m.Y(A[i][1]), 6.5, 0, 7); g.stroke(); });
      if (lastStep) { dot(g, m.X(lastStep.m[0]), m.Y(lastStep.m[1]), 4.5, C.yellow); }
      const drawPath = (r, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); r.path.forEach(([x, y], i) => (i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)))); g.stroke(); };
      runs.forEach((r) => drawPath(r, C[r.c], 1.6)); drawPath(full, C.blue, 2.8);
      runs.forEach((r) => dot(g, m.X(r.p[0]), m.Y(r.p[1]), 5, C[r.c], '#fff')); dot(g, m.X(full.p[0]), m.Y(full.p[1]), 6.5, C.blue, '#fff');
      if (lastStep) { const s = lastStep, f0 = lastStep.from; AI.arrow(g, m.X(f0[0]), m.Y(f0[1]), m.X(f0[0] - s.eta * s.tg[0]), m.Y(f0[1] - s.eta * s.tg[1]), C.blue, 2); AI.arrow(g, m.X(f0[0]), m.Y(f0[1]), m.X(f0[0] - s.eta * s.gb[0]), m.Y(f0[1] - s.eta * s.gb[1]), C.yellow, 2); }
      g.fillStyle = C.green; g.beginPath(); const sx = m.X(abar[0]), sy = m.Y(abar[1]);
      for (let i = 0; i < 10; i++) { const rr = i % 2 ? 4 : 10, a = (Math.PI * i) / 5 - Math.PI / 2; g.lineTo(sx + rr * Math.cos(a), sy + rr * Math.sin(a)); } g.closePath(); g.fill();
      txt(g, 'grey dots = data a_i   ★ = mean (the minimiser)   yellow rings = current batch of the first SGD run', 10, 16, C.fg, '11px sans-serif');
      const th = Math.sqrt(Math.max(0, (trC / B) * ((N - B) / (N - 1)))), emp = errN ? Math.sqrt(errSum / errN) : NaN;
      q(root, '#cap').textContent = lastStep ? `step ${k}: blue arrow = true step (η·mean gradient over all 24), yellow arrow = this batch's step (batch of ${B}); the difference is the noise.` : 'Press play: every step each SGD run draws a fresh random batch.';
      q(root, '#out').innerHTML = `mean loss  full ${num(loss(full.p), 3)} | SGD ${runs.map((r) => num(loss(r.p), 3)).join(' | ')}\ngradient-noise size  measured ${num(emp, 3)}   theory ${num(th, 3)}\n<span class="m">noise ∝ σ/√B (and exactly 0 when B = 24 = full batch)</span>`;
    }, { fps: 30 });
  };

  /* ===================================================================
     5a. Momentum: gradients accumulate into a velocity (heavy ball)
     =================================================================== */
  AI.labs.momentumlab = function (root) {
    const { view } = AI.labShell(root, 'Momentum on a ravine', 'f(x, y) = ½(x² + κy²) — gradient descent zig-zags; the heavy ball averages the zig-zag away',
      `<canvas style="height:430px;cursor:crosshair;touch-action:none"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>`,
      `${sl('kap', 'Steepness κ of the walls', 2, 40, 1, 15)}${sl('eta', 'Learning rate η', 0.01, 0.2, 0.005, 0.1)}${sl('beta', 'Momentum β', 0, 0.98, 0.01, 0.9)}${sl('sp', 'Speed (steps / second)', 0.5, 12, 0.5, 2)}
       <div class="legend"><span><i style="background:var(--blue)"></i>plain GD</span><span><i style="background:var(--green)"></i>momentum</span><span><i style="background:var(--orange)"></i>recent gradients, weighted β^j</span></div>
       <div class="row">${btn('play', '⏸ Pause')}${btn('step', 'One step')}${btn('reset', 'Restart')}</div>
       <div class="readout" id="out"></div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv), st = { playing: true }, tick = AI.ticker();
    let start = [-8, 3], gd, mo, k = 0, hitG = null, hitM = null;
    const kap = () => +q(root, '#kap').value, f = (p) => 0.5 * (p[0] ** 2 + kap() * p[1] ** 2), gr = (p) => [p[0], kap() * p[1]];
    const reset = () => { gd = { p: [...start], path: [[...start]] }; mo = { p: [...start], v: [0, 0], path: [[...start]], gs: [], prev: [...start] }; k = 0; hitG = hitM = null; };
    const doStep = () => {
      const eta = +q(root, '#eta').value, beta = +q(root, '#beta').value;
      let g = gr(gd.p); gd.p = [gd.p[0] - eta * g[0], gd.p[1] - eta * g[1]]; gd.path.push([...gd.p]);
      g = gr(mo.p); mo.v = [beta * mo.v[0] + g[0], beta * mo.v[1] + g[1]]; mo.gs.unshift(g); if (mo.gs.length > 60) mo.gs.pop();
      mo.prev = [...mo.p]; mo.p = [mo.p[0] - eta * mo.v[0], mo.p[1] - eta * mo.v[1]]; mo.path.push([...mo.p]);
      if (gd.path.length > 400) gd.path.shift(); if (mo.path.length > 400) mo.path.shift(); k++;
      if (hitG === null && f(gd.p) < 0.01) hitG = k; if (hitM === null && f(mo.p) < 0.01) hitM = k;
      if (!gd.p.every(Number.isFinite) || Math.hypot(...gd.p) > 1e6) gd.p = [0, 0];
      if (!mo.p.every(Number.isFinite) || Math.hypot(...mo.p) > 1e6) mo.p = [0, 0];
    };
    q(root, '#reset').onclick = reset; q(root, '#step').onclick = () => { setPlay(root, st, false); doStep(); }; ['kap', 'eta', 'beta'].forEach((id) => (q(root, '#' + id).oninput = reset)); playBtn(root, st);
    drag(cv, (px, py) => { const m = iso(c, -10, 10, -5, 5); start = [AI.clamp(m.ix(px), -9.5, 9.5), AI.clamp(m.iy(py), -4.8, 4.8)]; reset(); });
    reset();
    AI.loop(root, (dt) => {
      q(root, '#kapv').textContent = kap(); q(root, '#etav').textContent = num(+q(root, '#eta').value, 3); q(root, '#betav').textContent = num(+q(root, '#beta').value, 2); q(root, '#spv').textContent = q(root, '#sp').value;
      if (st.playing) { const n = tick(dt, +q(root, '#sp').value); for (let i = 0; i < n; i++) doStep(); }
      const C = AI.col(), g = c.ctx, m = iso(c, -10, 10, -5, 5), eta = +q(root, '#eta').value, beta = +q(root, '#beta').value, f0 = f(start);
      g.clearRect(0, 0, c.w, c.h);
      for (let j = 0; j < 8; j++) { const lv = f0 * Math.pow(0.42, j), a = Math.sqrt(2 * lv), b = Math.sqrt((2 * lv) / kap()); g.strokeStyle = C.purple; g.globalAlpha = 0.65 - j * 0.06; g.lineWidth = 1.3; g.beginPath(); for (let i = 0; i <= 90; i++) { const th = (i / 90) * 2 * Math.PI, x = m.X(a * Math.cos(th)), y = m.Y(b * Math.sin(th)); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); }
      g.globalAlpha = 1;
      const draw = (r, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); r.path.forEach(([x, y], i) => (i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)))); g.stroke(); g.fillStyle = col; r.path.forEach(([x, y]) => { g.beginPath(); g.arc(m.X(x), m.Y(y), 2.4, 0, 7); g.fill(); }); };
      draw(gd, C.blue, 2); draw(mo, C.green, 2.5);
      // chain of weighted gradients that built the last momentum step
      if (k > 0) {
        let cur = [...mo.prev];
        const terms = Math.min(8, mo.gs.length);
        for (let j = 0; j < terms; j++) {
          const w = Math.pow(beta, j), d = [-eta * w * mo.gs[j][0], -eta * w * mo.gs[j][1]], nx = [cur[0] + d[0], cur[1] + d[1]];
          g.globalAlpha = 0.95 - j * 0.07; if (Math.hypot(d[0], d[1]) * m.s > 3) AI.arrow(g, m.X(cur[0]), m.Y(cur[1]), m.X(nx[0]), m.Y(nx[1]), C.orange, 2.5); cur = nx;
        }
        g.globalAlpha = 1;
        if (Math.hypot(mo.p[0] - cur[0], mo.p[1] - cur[1]) * m.s > 3) { g.setLineDash([4, 3]); AI.arrow(g, m.X(cur[0]), m.Y(cur[1]), m.X(mo.p[0]), m.Y(mo.p[1]), C.muted, 2); g.setLineDash([]); }
      }
      dot(g, m.X(0), m.Y(0), 5, C.fg); dot(g, m.X(start[0]), m.Y(start[1]), 4, C.muted);
      dot(g, m.X(gd.p[0]), m.Y(gd.p[1]), 6.5, C.blue, '#fff'); dot(g, m.X(mo.p[0]), m.Y(mo.p[1]), 7, C.green, '#fff');
      txt(g, 'orange chain = −η·β^j·g (newest gradient first); chain ends where the momentum ball lands', 10, 16, C.fg, '11px sans-serif');
      q(root, '#cap').textContent = k ? `step ${k}: v = β·v_prev + g = (${num(mo.v[0], 2)}, ${num(mo.v[1], 2)}); the green ball moves by −η·v. Note how the y-components of old gradients (alternating signs) cancel in the sum while the x-components pile up.` : 'Press play. Both balls start at the same point; click the map to move the start.';
      q(root, '#out').innerHTML = `plain GD   f = ${num(f(gd.p), 4)}   ${hitG ? 'reached f<0.01 at step ' + hitG : '…'}\nmomentum   f = ${num(f(mo.p), 4)}   ${hitM ? 'reached f<0.01 at step ' + hitM : '…'}\n<span class="m">effective step along a steady slope ≈ η/(1−β) = ${num(eta / (1 - beta), 3)}</span>`;
    }, { fps: 30 });
  };

  /* ===================================================================
     5b. Adam / RMSProp by hand: row-by-row table
     =================================================================== */
  AI.labs.adamtable = function (root) {
    const { view } = AI.labShell(root, 'Adam and RMSProp, one row at a time', 'a single parameter θ with a sequence of gradients you choose — reveal the rows slowly and check each number by hand',
      `<div id="tbl" style="overflow-x:auto;min-height:150px"></div><div class="caption" id="cap" style="min-height:4.2em;font-family:ui-monospace,Menlo,Consolas,monospace"></div>`,
      `<label>Optimizer <select id="opt"><option value="adam">Adam (β₁, β₂)</option><option value="rms">RMSProp (β₂ only)</option></select></label>
       <label>Gradients g₁, g₂, … (up to 8) <input type="text" id="gs" value="4, 2, -1" spellcheck="false"></label>
       ${sl('eta', 'Learning rate η', 0.01, 0.5, 0.01, 0.1)}${sl('b1', 'β₁', 0, 0.99, 0.01, 0.9)}${sl('b2', 'β₂', 0.5, 0.999, 0.001, 0.999)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="bc" checked> bias correction (m̂ = m/(1−β₁ᵏ), v̂ = v/(1−β₂ᵏ))</label>
       <div class="row">${btn('next', 'Reveal next row', 'btn primary')}${btn('all', 'Show all')}${btn('reset', 'Reset')}<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="auto"> auto (one row / 3 s)</label></div>`);
    let shown = 0, acc = 0, rows = [];
    const th0 = 1, eps = 1e-8;
    const f4 = (v) => (v === 0 ? '0' : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(3) : String(+v.toPrecision(5)));
    const compute = () => {
      const gs = q(root, '#gs').value.split(/[,\s]+/).filter(Boolean).map(Number).filter(Number.isFinite).slice(0, 8);
      const eta = +q(root, '#eta').value, b1 = +q(root, '#b1').value, b2 = +q(root, '#b2').value, bc = q(root, '#bc').checked, rms = q(root, '#opt').value === 'rms';
      let m = 0, v = 0, th = th0; rows = [];
      gs.forEach((g, i) => {
        const k = i + 1; if (!rms) m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
        const mh = rms ? g : bc ? m / (1 - b1 ** k) : m, vh = bc ? v / (1 - b2 ** k) : v, u = (eta * mh) / (Math.sqrt(vh) + eps); th -= u;
        rows.push({ k, g, m, v, mh, vh, u, th, rms, eta, b1, b2, bc });
      });
      shown = Math.min(shown, rows.length); render();
    };
    const render = () => {
      const rms = q(root, '#opt').value === 'rms', bc = q(root, '#bc').checked;
      let h = `<table class="tbl"><tr><th>k</th><th>g<sub>k</sub></th>${rms ? '' : '<th>m<sub>k</sub></th>'}<th>v<sub>k</sub></th>${rms ? '' : '<th>m̂<sub>k</sub></th>'}${bc ? '<th>v̂<sub>k</sub></th>' : ''}<th>update</th><th>θ<sub>k</sub></th></tr><tr><td>0</td><td colspan="${rms ? 2 : 3}" class="m">θ₀ = ${th0}</td>${rms ? '' : '<td></td>'}${bc ? '<td></td>' : ''}<td></td><td><b>${th0}</b></td></tr>`;
      rows.slice(0, shown).forEach((r, i) => { h += `<tr${i === shown - 1 ? ' style="background:var(--bg3)"' : ''}><td>${r.k}</td><td>${f4(r.g)}</td>${rms ? '' : `<td>${f4(r.m)}</td>`}<td>${f4(r.v)}</td>${rms ? '' : `<td>${f4(r.mh)}</td>`}${bc ? `<td>${f4(r.vh)}</td>` : ''}<td>${f4(r.u)}</td><td><b>${f4(r.th)}</b></td></tr>`; });
      q(root, '#tbl').innerHTML = h + '</table>';
      const r = rows[shown - 1];
      q(root, '#cap').textContent = !r ? 'Press “Reveal next row”. Each row applies the update rule once; the highlighted row is the newest.' : r.rms
        ? `k=${r.k}: v = ${r.b2}·v_prev + ${f4(1 - r.b2)}·${f4(r.g)}² = ${f4(r.v)}${r.bc ? ' → v̂ = v/(1−β₂^k) = ' + f4(r.vh) : ''};  update = η·g/(√v${r.bc ? '̂' : ''} + ε) = ${r.eta}·${f4(r.g)}/${f4(Math.sqrt(r.vh))} = ${f4(r.u)};  θ = ${f4(r.th)}`
        : `k=${r.k}: m = ${r.b1}·m_prev + ${f4(1 - r.b1)}·${f4(r.g)} = ${f4(r.m)};  v = ${r.b2}·v_prev + ${f4(1 - r.b2)}·${f4(r.g)}² = ${f4(r.v)}\n  m̂ = ${f4(r.mh)}, v̂ = ${f4(r.vh)};  update = η·m̂/(√v̂ + ε) = ${r.eta}·${f4(r.mh)}/${f4(Math.sqrt(r.vh))} = ${f4(r.u)};  θ = ${f4(r.th)}`;
    };
    ['gs', 'eta', 'b1', 'b2', 'bc', 'opt'].forEach((id) => (q(root, '#' + id).oninput = () => { q(root, '#etav').textContent = q(root, '#eta').value; q(root, '#b1v').textContent = q(root, '#b1').value; q(root, '#b2v').textContent = q(root, '#b2').value; compute(); }));
    q(root, '#next').onclick = () => { if (shown < rows.length) shown++; render(); };
    q(root, '#all').onclick = () => { shown = rows.length; render(); };
    q(root, '#reset').onclick = () => { shown = 0; acc = 0; render(); };
    q(root, '#etav').textContent = q(root, '#eta').value; q(root, '#b1v').textContent = q(root, '#b1').value; q(root, '#b2v').textContent = q(root, '#b2').value;
    compute();
    AI.loop(root, (dt) => { if (q(root, '#auto').checked) { acc += dt; if (acc >= 3) { acc = 0; if (shown < rows.length) { shown++; render(); } } } }, { fps: 10 });
  };

  /* ===================================================================
     6. Learning-rate schedules + toy training run
     =================================================================== */
  AI.labs.schedules = function (root) {
    const { view } = AI.labShell(root, 'Learning-rate schedules and what they do', 'top: η(t) for five schedules; bottom: loss of the same noisy toy problem trained with each (log scale)',
      `<canvas style="height:520px"></canvas><div class="caption" id="cap" style="min-height:2.6em"></div>`,
      `${sl('mx', 'Peak learning rate η_max', 0.01, 0.12, 0.005, 0.07)}${sl('T', 'Total steps T', 100, 400, 20, 200)}${sl('W', 'Warm-up steps W', 0, 60, 2, 20)}${sl('mn', 'Final fraction η_min/η_max', 0.01, 0.3, 0.01, 0.02)}${sl('gam', 'Step-decay factor γ (drop every T/4)', 0.1, 0.8, 0.05, 0.3)}${sl('noise', 'Gradient noise σ', 0, 1.5, 0.05, 0.4)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="sharp" checked> early sharpness (loss is 2.5× steeper at the start, toy model of early training)</label>
       <div class="chips" id="pick"></div>
       ${sl('sp', 'Speed (steps / second)', 2, 60, 1, 10)}
       <div class="row">${btn('play', '⏸ Pause')}${btn('reset', 'Replay')}</div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), st = { playing: true };
    const NAMES = ['constant', 'step decay', 'cosine', 'warm-up + cosine', 'exponential'], COLS = ['muted', 'blue', 'green', 'orange', 'purple'];
    const D = 10, LAM = Array.from({ length: D }, (_, i) => 0.5 * Math.pow(40, i / (D - 1))), noiseZ = [];
    const rn = AI.rng(5); for (let i = 0; i < 400 * D; i++) noiseZ.push(AI.randn(rn));
    let sel = 3, k = 0, acc = 0, curves = [], losses = [], sig = '';
    mkChips(q(root, '#pick'), NAMES.map((n, i) => [n, i]), (i) => { sel = i; });
    const P = () => ({ mx: +q(root, '#mx').value, T: +q(root, '#T').value, W: +q(root, '#W').value, mn: +q(root, '#mn').value, gam: +q(root, '#gam').value });
    const sched = (i, t, p) => {
      const lo = p.mx * p.mn;
      if (i === 0) return p.mx;
      if (i === 1) return p.mx * Math.pow(p.gam, Math.floor((4 * t) / p.T));
      if (i === 2) return lo + 0.5 * (p.mx - lo) * (1 + Math.cos((Math.PI * t) / p.T));
      if (i === 3) return t < p.W ? (p.mx * (t + 1)) / p.W : lo + 0.5 * (p.mx - lo) * (1 + Math.cos((Math.PI * (t - p.W)) / Math.max(1, p.T - p.W)));
      return p.mx * Math.pow(p.mn, t / p.T);
    };
    const simulate = () => {
      const p = P(), sig2 = +q(root, '#noise').value, sharp = q(root, '#sharp').checked ? 1.5 : 0;
      curves = NAMES.map((_, i) => Array.from({ length: p.T + 1 }, (_, t) => sched(i, Math.min(t, p.T), p)));
      losses = NAMES.map((_, i) => {
        const x = new Float64Array(D).fill(1), out = []; let dead = false;
        for (let t = 0; t <= p.T; t++) {
          let L = 0; for (let d = 0; d < D; d++) L += 0.5 * LAM[d] * x[d] * x[d];
          if (!Number.isFinite(L) || L > 1e4) { dead = true; L = 1e4; }
          out.push(L);
          if (t === p.T) break;
          const mult = 1 + sharp * Math.exp(-t / 20), eta = curves[i][t];
          for (let d = 0; d < D; d++) { x[d] -= eta * (LAM[d] * mult * x[d] + sig2 * noiseZ[t * D + d]); if (dead) x[d] = 0; }
        }
        return out;
      });
    };
    q(root, '#reset').onclick = () => { k = 0; acc = 0; setPlay(root, st, true); }; playBtn(root, st);
    AI.loop(root, (dt) => {
      ['mx', 'W', 'mn', 'gam', 'noise'].forEach((id) => (q(root, '#' + id + 'v').textContent = num(+q(root, '#' + id).value, 3))); q(root, '#Tv').textContent = q(root, '#T').value; q(root, '#spv').textContent = q(root, '#sp').value;
      const key = ['mx', 'T', 'W', 'mn', 'gam', 'noise'].map((id) => q(root, '#' + id).value).join(',') + q(root, '#sharp').checked;
      if (key !== sig) { sig = key; simulate(); }
      const p = P(); if (st.playing) { acc += dt * +q(root, '#sp').value; if (acc >= 1) { const n = Math.floor(acc); acc -= n; k = Math.min(p.T, k + n); if (k >= p.T) setPlay(root, st, false); } }
      const C = AI.col(), g = c.ctx, h1 = Math.round(c.h * 0.38);
      g.clearRect(0, 0, c.w, c.h);
      // top: schedules
      const pad = { l: 48, r: 12, t: 22, b: 22 };
      const v1 = AI.view({ w: c.w, h: h1 }, 0, p.T, 0, p.mx * 1.1, pad);
      AI.axes(g, v1);
      NAMES.forEach((_, i) => {
        g.strokeStyle = C[COLS[i]]; g.globalAlpha = i === sel ? 1 : 0.45; g.lineWidth = i === sel ? 3 : 1.6; g.beginPath();
        curves[i].forEach((y, t) => (t ? g.lineTo(v1.X(t), v1.Y(y)) : g.moveTo(v1.X(t), v1.Y(y)))); g.stroke();
      });
      g.globalAlpha = 1; g.strokeStyle = C.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v1.X(k), v1.pad.t); g.lineTo(v1.X(k), v1.pad.t + v1.ph); g.stroke(); g.setLineDash([]);
      NAMES.forEach((_, i) => dot(g, v1.X(k), v1.Y(curves[i][k]), i === sel ? 5 : 3, C[COLS[i]]));
      txt(g, 'learning rate η(t)', pad.l + 6, 14, C.fg, '11px sans-serif');
      // bottom: loss, log scale
      const y0 = h1 + 18, h2 = c.h - y0, v2 = AI.view({ w: c.w, h: h2 }, 0, p.T, -3, 3, { l: 48, r: 12, t: 16, b: 24 }), Y2 = (L) => y0 + v2.Y(AI.clamp(Math.log10(Math.max(L, 1e-9)), -3, 3));
      g.save(); g.translate(0, y0);
      g.restore();
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '10px monospace'; g.textAlign = 'right';
      for (let e = -3; e <= 3; e++) { const yy = Y2(Math.pow(10, e)); g.beginPath(); g.moveTo(v2.pad.l, yy); g.lineTo(v2.pad.l + v2.pw, yy); g.stroke(); g.fillText('1e' + e, v2.pad.l - 4, yy + 3); }
      g.textAlign = 'center'; for (let t = 0; t <= p.T; t += AI.niceStep(p.T, 5)) g.fillText(String(t), v2.X(t), c.h - 6);
      g.save(); g.beginPath(); g.rect(v2.pad.l, y0 + v2.pad.t, v2.pw, v2.ph); g.clip();
      NAMES.forEach((_, i) => {
        g.strokeStyle = C[COLS[i]]; g.globalAlpha = i === sel ? 1 : 0.5; g.lineWidth = i === sel ? 3 : 1.6; g.beginPath();
        for (let t = 0; t <= k; t++) (t ? g.lineTo(v2.X(t), Y2(losses[i][t])) : g.moveTo(v2.X(t), Y2(losses[i][t]))); g.stroke();
      });
      g.restore(); g.globalAlpha = 1;
      txt(g, 'training loss (log scale; flat bottom = noise floor, spike = unstable step)', v2.pad.l + 6, y0 + 10, C.fg, '11px sans-serif');
      const fin = (i) => losses[i][k];
      q(root, '#cap').textContent = `step ${k} / ${p.T}. Selected: ${NAMES[sel]} (η now = ${num(curves[sel][k], 4)}). Warm-up avoids the early blow-up of high η when the loss is sharp; decay shrinks the noise floor.`;
      q(root, '#out').innerHTML = NAMES.map((n, i) => `<span class="${i === sel ? 'o' : 'm'}">${n.padEnd(17)}</span> η=${num(curves[i][k], 4).padEnd(8)} loss=${fin(i) >= 1e4 ? '<span class="r">blew up</span>' : num(fin(i), 4)}`).join('\n');
    }, { fps: 30 });
  };

  /* ===================================================================
     7. Constraints: level sets touching L2 / L1 balls, Lagrange condition
     =================================================================== */
  AI.labs.constraints = function (root) {
    const { view } = AI.labShell(root, 'Constrained minimum: where the loss contour first touches the ball', 'the loss wants to be at ★ (drag it); the allowed region is a ball ‖w‖ ≤ r. Grow the contour until it touches.',
      `<canvas style="height:460px;cursor:grab;touch-action:none"></canvas><div class="caption" id="cap" style="min-height:2.8em"></div>`,
      `<div class="row">${btn('l2', 'L2 ball (circle)', 'btn primary')}${btn('l1', 'L1 ball (diamond)')}</div>
       ${sl('r', 'Radius r', 0.5, 4, 0.1, 2)}${sl('q', 'Loss stretch (q: how much steeper in the 2nd axis)', 1, 8, 0.25, 1)}${sl('phi', 'Loss tilt (degrees)', 0, 180, 5, 0)}${sl('sp', 'Speed ×', 0.25, 3, 0.25, 1)}
       <div class="row">${btn('replay', '↻ Replay growth', 'btn primary')}${btn('play', '⏸ Pause')}</div>
       <div class="chips" id="pre"></div>
       <div class="readout" id="out"></div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv), st = { playing: true };
    let norm = 2, ctr = [3, 0.5], u = 0, sol = null, sigK = '';
    const R = 5.6;
    const r = () => +q(root, '#r').value, qq = () => +q(root, '#q').value, phi = () => (+q(root, '#phi').value * Math.PI) / 180;
    const Qm = () => { const co = Math.cos(phi()), si = Math.sin(phi()), s = qq(); return { a: co * co + s * si * si, b: co * si * (1 - s), d: si * si + s * co * co }; };
    const fval = (p) => { const Q = Qm(), dx = p[0] - ctr[0], dy = p[1] - ctr[1]; return 0.5 * (Q.a * dx * dx + 2 * Q.b * dx * dy + Q.d * dy * dy); };
    const bpt = (s) => {
      if (norm === 2) return [r() * Math.cos(s), r() * Math.sin(s)];
      const m = ((s % 4) + 4) % 4, e = Math.floor(m), t = m - e, R0 = r(), V = [[R0, 0], [0, R0], [-R0, 0], [0, -R0]], A0 = V[e % 4], B0 = V[(e + 1) % 4];
      return [A0[0] + (B0[0] - A0[0]) * t, A0[1] + (B0[1] - A0[1]) * t];
    };
    const inside = (p) => (norm === 2 ? Math.hypot(p[0], p[1]) <= r() + 1e-9 : Math.abs(p[0]) + Math.abs(p[1]) <= r() + 1e-9);
    const solve = () => {
      if (inside(ctr)) return { p: [...ctr], fs: 0, free: true };
      const span = norm === 2 ? 2 * Math.PI : 4, N = 1600; let best = 0, bf = Infinity;
      for (let i = 0; i < N; i++) { const f0 = fval(bpt((i / N) * span)); if (f0 < bf) { bf = f0; best = (i / N) * span; } }
      let lo = best - span / N, hi = best + span / N;
      for (let it = 0; it < 50; it++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (fval(bpt(m1)) < fval(bpt(m2))) hi = m2; else lo = m1; }
      const p = bpt((lo + hi) / 2); return { p, fs: fval(p), free: false };
    };
    const ell = (g, m, s, col, w) => {
      const Q = Qm(), co = Math.cos(phi()), si = Math.sin(phi()), a = Math.sqrt(2 * s), b = Math.sqrt((2 * s) / qq());
      g.strokeStyle = col; g.lineWidth = w; g.beginPath();
      for (let i = 0; i <= 120; i++) { const th = (i / 120) * 2 * Math.PI, w1 = a * Math.cos(th), w2 = b * Math.sin(th), x = ctr[0] + co * w1 - si * w2, y = ctr[1] + si * w1 + co * w2; i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)); }
      g.stroke(); return Q;
    };
    const setNorm = (n) => { norm = n; q(root, '#l2').className = n === 2 ? 'btn primary' : 'btn'; q(root, '#l1').className = n === 1 ? 'btn primary' : 'btn'; u = 0; };
    q(root, '#l2').onclick = () => setNorm(2); q(root, '#l1').onclick = () => setNorm(1);
    q(root, '#replay').onclick = () => { u = 0; setPlay(root, st, true); }; playBtn(root, st);
    ['r', 'q', 'phi'].forEach((id) => (q(root, '#' + id).oninput = () => { u = 0; }));
    mkChips(q(root, '#pre'), [['L1: lands on an axis (sparse)', () => { setNorm(1); ctr = [3, 0.5]; q(root, '#r').value = 2; q(root, '#q').value = 1; q(root, '#phi').value = 0; }], ['L2: both weights non-zero', () => { setNorm(2); ctr = [3, 0.5]; q(root, '#r').value = 2; q(root, '#q').value = 1; q(root, '#phi').value = 0; }], ['centre inside → constraint inactive', () => { ctr = [0.6, 0.4]; u = 0; }]], (fnc) => fnc());
    drag(cv, (px, py) => { const m = iso(c, -R, R, -R, R); ctr = [AI.clamp(m.ix(px), -R + 0.2, R - 0.2), AI.clamp(m.iy(py), -R + 0.2, R - 0.2)]; u = 0; });
    AI.loop(root, (dt) => {
      q(root, '#rv').textContent = num(r(), 1); q(root, '#qv').textContent = num(qq(), 2); q(root, '#phiv').textContent = q(root, '#phi').value + '°'; q(root, '#spv').textContent = '×' + q(root, '#sp').value;
      const key = [norm, r(), qq(), phi(), ctr[0].toFixed(3), ctr[1].toFixed(3)].join(); if (key !== sigK) { sigK = key; sol = solve(); }
      if (st.playing) u = Math.min(1, u + (dt * +q(root, '#sp').value) / 5);
      const C = AI.col(), g = c.ctx, m = iso(c, -R, R, -R, R), Q = Qm(), s = sol.fs * AI.smooth(u);
      g.clearRect(0, 0, c.w, c.h);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(m.X(-R), m.Y(0)); g.lineTo(m.X(R), m.Y(0)); g.moveTo(m.X(0), m.Y(-R)); g.lineTo(m.X(0), m.Y(R)); g.stroke();
      txt(g, 'w₁', m.X(R) - 18, m.Y(0) - 6, C.muted, '12px sans-serif'); txt(g, 'w₂', m.X(0) + 6, m.Y(R) + 12, C.muted, '12px sans-serif');
      // ball
      g.beginPath(); for (let i = 0; i <= 400; i++) { const [x, y] = bpt((i / 400) * (norm === 2 ? 2 * Math.PI : 4)); i ? g.lineTo(m.X(x), m.Y(y)) : g.moveTo(m.X(x), m.Y(y)); } g.closePath();
      g.fillStyle = C.blue; g.globalAlpha = 0.14; g.fill(); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 3; g.stroke();
      // faint contours + growing contour
      if (!sol.free) [0.25, 0.5, 1.8, 3.5].forEach((fr) => { g.globalAlpha = 0.35; ell(g, m, sol.fs * fr, C.purple, 1.2); g.globalAlpha = 1; });
      const done = u >= 1 && !sol.free;
      if (!sol.free && s > 1e-6) ell(g, m, s, done ? C.green : C.orange, 3);
      dot(g, m.X(ctr[0]), m.Y(ctr[1]), 7, C.red, '#fff'); txt(g, '★ loss minimum (drag me)', m.X(ctr[0]) + 10, m.Y(ctr[1]) - 8, C.red, '12px sans-serif');
      let msg = '';
      if (sol.free) { dot(g, m.X(sol.p[0]), m.Y(sol.p[1]), 5, C.green); msg = 'The unconstrained minimum is already inside the ball → the constraint is inactive (multiplier λ = 0). Drag ★ outside the ball.'; }
      else {
        const p = sol.p, Q0 = Qm(), gx = Q0.a * (p[0] - ctr[0]) + Q0.b * (p[1] - ctr[1]), gy = Q0.b * (p[0] - ctr[0]) + Q0.d * (p[1] - ctr[1]), d = [-gx, -gy], dl = Math.hypot(d[0], d[1]) || 1;
        if (u > 0.98) {
          const L = 70, tol = 1e-3 * r(); let ok = true, extra = '';
          if (norm === 2) { const n = [p[0] / r(), p[1] / r()]; AI.arrow(g, m.X(p[0]), m.Y(p[1]), m.X(p[0]) + n[0] * L, m.Y(p[1]) - n[1] * L, C.blue, 3); ok = Math.abs(d[0] * n[1] - d[1] * n[0]) / dl < 0.02; }
          else {
            const vx = Math.abs(p[1]) < tol * 4, vy = Math.abs(p[0]) < tol * 4;
            if (vx || vy) {
              const sx = Math.sign(p[0]) || 1, sy = Math.sign(p[1]) || 1, n1 = vx ? [sx, 1] : [1, sy], n2 = vx ? [sx, -1] : [-1, sy];
              [n1, n2].forEach((n) => { const nl = Math.hypot(...n); g.setLineDash([5, 4]); AI.arrow(g, m.X(p[0]), m.Y(p[1]), m.X(p[0]) + (n[0] / nl) * L, m.Y(p[1]) - (n[1] / nl) * L, C.blue, 2.5); g.setLineDash([]); });
              const det = n1[0] * n2[1] - n1[1] * n2[0], al = (d[0] * n2[1] - d[1] * n2[0]) / det, be = (n1[0] * d[1] - n1[1] * d[0]) / det; ok = al >= -1e-6 && be >= -1e-6; extra = ' (corner: the normal is a whole cone between the two dashed arrows)';
            } else { const n = [Math.sign(p[0]), Math.sign(p[1])], nl = Math.SQRT2; AI.arrow(g, m.X(p[0]), m.Y(p[1]), m.X(p[0]) + (n[0] / nl) * L, m.Y(p[1]) - (n[1] / nl) * L, C.blue, 3); ok = Math.abs(d[0] * n[1] - d[1] * n[0]) / dl < 0.02; }
            msg = extra;
          }
          AI.arrow(g, m.X(p[0]), m.Y(p[1]), m.X(p[0]) + (d[0] / dl) * L * 1.05, m.Y(p[1]) - (d[1] / dl) * L * 1.05, C.orange, 3);
          msg = `③ Touching point w* = (${num(p[0], 3)}, ${num(p[1], 3)}). Orange −∇f and blue outward normal point the same way${msg} → Lagrange condition ∇f = −λ∇g ${ok ? 'holds ✓' : '(not satisfied: refine)'}.`;
        } else msg = u < 0.05 ? '① The loss is lowest at ★. But we may only use points inside the blue region.' : `② Grow the contour of constant loss (orange, level f = ${num(s, 3)}) until it first touches the blue region.`;
        dot(g, m.X(p[0]), m.Y(p[1]), done ? 8 : 4, done ? C.green : C.muted, '#fff');
      }
      q(root, '#cap').textContent = msg;
      const sparse = !sol.free && (Math.abs(sol.p[0]) < 5e-3 * r() || Math.abs(sol.p[1]) < 5e-3 * r());
      q(root, '#out').innerHTML = `ball: ${norm === 2 ? 'L2  w₁² + w₂² ≤ r²' : 'L1  |w₁| + |w₂| ≤ r'}   r = ${num(r(), 2)}\nsolution w* = (${num(sol.p[0], 3)}, ${num(sol.p[1], 3)})\nloss at w* = ${num(sol.fs, 4)}   ‖∇f‖ = ${num(Math.hypot(Q.a * (sol.p[0] - ctr[0]) + Q.b * (sol.p[1] - ctr[1]), Q.b * (sol.p[0] - ctr[0]) + Q.d * (sol.p[1] - ctr[1])), 3)}\n${sol.free ? '<span class="m">constraint inactive</span>' : sparse ? '<span class="g">one weight is exactly 0 → SPARSE solution</span>' : '<span class="o">both weights non-zero (dense)</span>'}`;
    }, { fps: 30 });
  };
})();
