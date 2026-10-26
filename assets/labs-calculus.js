/* Labs for Module 2 · Calculus.
 *   slopezoom   zoom into a curve until it looks straight (limits & slope)
 *   rules       f and f' side by side (autodiff) + "check my derivative"
 *   chainwiggle a wiggle propagating through two nested machines
 *   gradmap     contour map, partial slopes, gradient, directional derivative, walk uphill/downhill
 *   jacmap      tiny circle -> ellipse under a 2D->2D map (the Jacobian)
 *   hessian     curvature along a direction (osculating parabola), eigen-directions
 *   taylor      order 0..6 polynomial approximations around a point
 *   riemann     rectangles refining into an integral
 *   ftc         fundamental theorem: area function and its slope
 */
(function () {
  const TAU = Math.PI * 2;
  const chips = (el, list, cb) => list.forEach((p) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = p.name || p; b.onclick = () => cb(p); el.appendChild(b); });
  const plotFn = (g, v, f, col, w, dash, x0, x1, n = 300) => {
    g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
    let pen = false, py = 0;
    for (let i = 0; i <= n; i++) {
      const x = x0 + ((x1 - x0) * i) / n, y = f(x);
      if (!Number.isFinite(y) || Math.abs(y) > 1e6) { pen = false; continue; }
      const jump = pen && Math.abs(v.Y(y) - py) > 4000;
      pen && !jump ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); pen = true; py = v.Y(y);
    }
    g.stroke(); g.setLineDash([]);
  };
  const autoY = (f, x0, x1, cap = 30) => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 160; i++) { const y = f(x0 + ((x1 - x0) * i) / 160); if (Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    lo = Math.max(lo, -cap); hi = Math.min(hi, cap);
    if (hi - lo < 0.5) { const m = (hi + lo) / 2; lo = m - 0.25; hi = m + 0.25; }
    const pd = (hi - lo) * 0.12; return [lo - pd, hi + pd];
  };
  const text = (g, s, x, y, col, align = 'left', font = '12px sans-serif') => { g.font = font; g.fillStyle = col; g.textAlign = align; g.fillText(s, x, y); };
  const drag = (cv, fn) => {
    const mv = (e) => { const r = cv.getBoundingClientRect(); fn(e.clientX - r.left, e.clientY - r.top, e); };
    cv.style.touchAction = 'none';
    cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); mv(e); cv._d = true; });
    cv.addEventListener('pointermove', (e) => { if (cv._d) mv(e); });
    cv.addEventListener('pointerup', () => (cv._d = false));
  };
  const gradAt = (ast, x, y) => { const r = AI.autodiff(ast, { x, y }); return [r.varGrad.x || 0, r.varGrad.y || 0, r.value]; };
  const hessAt = (ast, x, y) => {
    const h = 1e-3, a = gradAt(ast, x + h, y), b = gradAt(ast, x - h, y), c = gradAt(ast, x, y + h), d = gradAt(ast, x, y - h);
    const hxx = (a[0] - b[0]) / (2 * h), hyy = (c[1] - d[1]) / (2 * h), hxy = ((a[1] - b[1]) / (2 * h) + (c[0] - d[0]) / (2 * h)) / 2;
    return [hxx, hxy, hyy];
  };
  const eig2 = (a, b, c) => { // symmetric [[a,b],[b,c]]
    const m = (a + c) / 2, d = Math.hypot((a - c) / 2, b), l1 = m + d, l2 = m - d;
    const th = 0.5 * Math.atan2(2 * b, a - c); // angle of eigenvector for l1
    return { l1, l2, v1: [Math.cos(th), Math.sin(th)], v2: [-Math.sin(th), Math.cos(th)] };
  };

  /* ---------- truncated Taylor series ("jets") of an expression AST around x = a ---------- */
  AI.calcJet = function (ast, a, N) {
    const Z = () => new Array(N + 1).fill(0), K = (v) => { const r = Z(); r[0] = v; return r; };
    const add = (p, q) => p.map((v, i) => v + q[i]), sub = (p, q) => p.map((v, i) => v - q[i]);
    const mul = (p, q) => { const r = Z(); for (let i = 0; i <= N; i++) for (let j = 0; i + j <= N; j++) r[i + j] += p[i] * q[j]; return r; };
    const div = (p, q) => { const r = Z(); for (let n = 0; n <= N; n++) { let s = p[n]; for (let k = 1; k <= n; k++) s -= q[k] * r[n - k]; r[n] = s / q[0]; } return r; };
    const exp = (p) => { const r = Z(); r[0] = Math.exp(p[0]); for (let n = 1; n <= N; n++) { let s = 0; for (let k = 1; k <= n; k++) s += k * p[k] * r[n - k]; r[n] = s / n; } return r; };
    const log = (p) => { const r = Z(); r[0] = Math.log(p[0]); for (let n = 1; n <= N; n++) { let s = p[n]; for (let k = 1; k < n; k++) s -= (k * r[k] * p[n - k]) / n; r[n] = s / p[0]; } return r; };
    const sincos = (p) => { const s = Z(), c = Z(); s[0] = Math.sin(p[0]); c[0] = Math.cos(p[0]); for (let n = 1; n <= N; n++) { let a1 = 0, a2 = 0; for (let k = 1; k <= n; k++) { a1 += k * p[k] * c[n - k]; a2 += k * p[k] * s[n - k]; } s[n] = a1 / n; c[n] = -a2 / n; } return [s, c]; };
    const ipow = (p, n) => { let r = K(1); for (let i = 0; i < n; i++) r = mul(r, p); return r; };
    const pw = (p, q) => {
      const cst = q.slice(1).every((v) => v === 0), e = q[0];
      if (cst && Number.isInteger(e) && Math.abs(e) <= 40) return e >= 0 ? ipow(p, e) : div(K(1), ipow(p, -e));
      if (!(p[0] > 0)) return K(NaN);
      return exp(mul(q, log(p)));
    };
    const ev = (n) => {
      switch (n.t) {
        case 'num': return K(n.v);
        case 'var': if (n.n === 'e') return K(Math.E); if (n.n === 'pi') return K(Math.PI); { const r = Z(); r[0] = a; r[1] = 1; return r; }
        case 'neg': return ev(n.a).map((v) => -v);
        case 'bin': { const p = ev(n.a), q = ev(n.b); return n.op === '+' ? add(p, q) : n.op === '-' ? sub(p, q) : n.op === '*' ? mul(p, q) : n.op === '/' ? div(p, q) : pw(p, q); }
        case 'fn2': { const p = ev(n.a), q = ev(n.b); return (n.n === 'max' ? p[0] >= q[0] : p[0] <= q[0]) ? p : q; }
        case 'fn': {
          const p = ev(n.a);
          switch (n.n) {
            case 'exp': return exp(p);
            case 'log': case 'ln': return log(p);
            case 'sin': return sincos(p)[0];
            case 'cos': return sincos(p)[1];
            case 'tan': { const [s, c] = sincos(p); return div(s, c); }
            case 'sqrt': return pw(p, K(0.5));
            case 'abs': return p[0] >= 0 ? p : p.map((v) => -v);
            case 'relu': return p[0] > 0 ? p : Z();
            case 'tanh': { const e2 = exp(p.map((v) => 2 * v)); return div(sub(e2, K(1)), add(e2, K(1))); }
            case 'sigmoid': return div(K(1), add(K(1), exp(p.map((v) => -v))));
            case 'atan': {
              const dp = Z(); for (let k = 0; k < N; k++) dp[k] = (k + 1) * p[k + 1];
              const q = div(dp, add(K(1), mul(p, p))), r = Z(); r[0] = Math.atan(p[0]); for (let k = 1; k <= N; k++) r[k] = q[k - 1] / k; return r;
            }
          }
        }
      }
      return K(NaN);
    };
    return ev(ast);
  };

  /* =============================================================== slopezoom */
  AI.labs.slopezoom = function (root) {
    const { view, ctl } = AI.labShell(root, 'Zoom into a curve', 'keep zooming — every smooth curve turns into its tangent line',
      `<canvas style="height:380px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px"></div>`,
      `<label>f(x) = <input type="text" id="expr" value="sin(x) + x^2/4" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>Point x₀ <span class="v" id="xv"></span><input type="range" id="x0" min="-3" max="3" step="0.05" value="1"></label>
       <label>Window half-width w <span class="v" id="wv"></span><input type="range" id="z" min="-4" max="0.7" step="0.01" value="0.7"></label>
       <label>Zoom speed <span class="v">decades / second</span><input type="range" id="sp" min="0.03" max="0.5" step="0.01" value="0.12"></label>
       <div class="row"><button class="btn primary" id="play">▶ Zoom in</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    c.onresize = () => (key = '');
    let ast = null, fn = null, playing = false, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f, a) => { fn = f; ast = a; key = ''; } });
    chips(q('#pre'), ['sin(x) + x^2/4', 'x^3 - 3*x', 'exp(x)', 'abs(x)', 'sqrt(abs(x))', 'sigmoid(3*x)'], (p) => ei.set(p));
    ei.run();
    q('#play').onclick = () => { if (+q('#z').value <= -3.99) q('#z').value = 0.7; playing = !playing; key = ''; };
    q('#reset').onclick = () => { playing = false; q('#z').value = 0.7; };
    AI.loop(root, (dt) => {
      if (playing) { const z = +q('#z').value - dt * +q('#sp').value; q('#z').value = Math.max(-4, z); if (z <= -4) playing = false; }
      q('#play').textContent = playing ? '⏸ Pause' : '▶ Zoom in';
      const x0 = +q('#x0').value, z = +q('#z').value, k2 = [x0, z, c.w, c.h, AI.col().bg, q('#expr').value].join('|');
      if (k2 === key) return; key = k2;
      const w = Math.pow(10, z), h = w / 2, C = AI.col(), g = c.ctx;
      q('#xv').textContent = AI.fmt(x0, 2); q('#wv').textContent = AI.fmt(w, 4);
      const f0 = fn({ x: x0 }), f1 = fn({ x: x0 + h }), m = AI.autodiff(ast, { x: x0 }).varGrad.x ?? NaN;
      const pad = { l: 58, r: 14, t: 14, b: 30 }, pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b, yh = (w * ph) / pw;
      const v = AI.view(c, x0 - w, x0 + w, f0 - yh, f0 + yh, pad);
      g.clearRect(0, 0, c.w, c.h);
      const sx = AI.niceStep(2 * w, 5), sy = AI.niceStep(2 * yh, 5);
      g.lineWidth = 1; g.font = '11px ui-monospace,Menlo,monospace';
      for (let d = -Math.floor(w / sx) * sx; d <= w + 1e-12; d += sx) { const X = v.X(x0 + d); g.strokeStyle = Math.abs(d) < sx / 1e3 ? C.muted : C.border; g.beginPath(); g.moveTo(X, pad.t); g.lineTo(X, pad.t + ph); g.stroke(); text(g, AI.fmt(+d.toPrecision(3), 3), X, c.h - 14, C.muted, 'center', '11px ui-monospace,monospace'); }
      for (let d = -Math.floor(yh / sy) * sy; d <= yh + 1e-12; d += sy) { const Y = v.Y(f0 + d); g.strokeStyle = Math.abs(d) < sy / 1e3 ? C.muted : C.border; g.beginPath(); g.moveTo(pad.l, Y); g.lineTo(pad.l + pw, Y); g.stroke(); text(g, AI.fmt(+d.toPrecision(3), 3), pad.l - 5, Y + 4, C.muted, 'right', '11px ui-monospace,monospace'); }
      text(g, 'Δx = x − x₀', pad.l + pw, c.h - 2, C.muted, 'right', '11px sans-serif'); text(g, 'Δy = y − f(x₀)', 4, pad.t + 8, C.muted, 'left', '11px sans-serif');
      g.save(); g.beginPath(); g.rect(pad.l, pad.t, pw, ph); g.clip();
      const line = (s, col, lw, dash) => { g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.beginPath(); g.moveTo(v.X(x0 - w), v.Y(f0 - s * w)); g.lineTo(v.X(x0 + w), v.Y(f0 + s * w)); g.stroke(); g.setLineDash([]); };
      if (Number.isFinite(f1)) line((f1 - f0) / h, C.orange, 2);
      if (Number.isFinite(m)) line(m, C.red, 1.5, [7, 5]);
      plotFn(g, v, (x) => fn({ x }), C.blue, 2.6, null, x0 - w, x0 + w, 240);
      g.fillStyle = C.fg; [[x0, f0], [x0 + h, f1]].forEach(([x, y]) => { if (Number.isFinite(y)) { g.beginPath(); g.arc(v.X(x), v.Y(y), 5, 0, 7); g.fill(); } });
      g.restore();
      text(g, 'blue f(x)   orange secant (x₀ to x₀+w/2)   red dashed tangent', pad.l + 8, pad.t + 28, C.muted, 'left', '11px sans-serif');
      let bend = 0; for (let i = 0; i <= 60; i++) { const x = x0 - w + (2 * w * i) / 60, y = fn({ x }); if (Number.isFinite(y)) bend = Math.max(bend, Math.abs(y - f0 - m * (x - x0))); }
      const ratio = bend / w, sec = (f1 - f0) / h;
      q('#out').innerHTML = `window w     = ${AI.fmt(w, 5)}\nstep h = w/2  = ${AI.fmt(h, 5)}\nΔy = f(x₀+h)−f(x₀) = ${AI.fmt(f1 - f0, 6)}\nsecant slope  = <span class="o">${AI.fmt(sec, 6)}</span>\ntangent slope = <span class="r">${AI.fmt(m, 6)}</span>\ngap           = ${AI.fmt(Math.abs(sec - m), 6)}\nbend / w      = ${AI.fmt(ratio, 5)}`;
      q('#cap').textContent = !Number.isFinite(m) || ratio > 0.2 ? 'Zoomed out: the curve clearly bends, so the secant slope and the tangent slope disagree.'
        : ratio > 0.01 ? 'Zooming in: the bend shrinks faster than the window. The secant is closing in on the tangent.'
        : ratio > 1e-4 ? 'Almost straight: the curve is now hard to tell from its tangent line. The secant slope ≈ the derivative.'
        : 'Indistinguishable from a straight line. This is what “differentiable” means: locally the curve is a line, and its slope is f′(x₀).';
    });
  };

  /* =============================================================== rules */
  AI.labs.rules = function (root) {
    const { view, ctl } = AI.labShell(root, 'Derivative rules, checked by machine', 'type f(x) and your guess for f′(x); the computer differentiates exactly and compares',
      `<canvas style="height:440px"></canvas>`,
      `<label>f(x) = <input type="text" id="expr" value="x^2*sin(x)" spellcheck="false"></label><div class="err" id="err"></div>
       <label>My guess for f′(x) = <input type="text" id="guess" value="2*x*sin(x) + x^2*cos(x)" spellcheck="false"></label><div class="err" id="err2"></div>
       <div class="chips" id="pre"></div>
       <label>Point x₀ <span class="v" id="xv"></span><input type="range" id="x0" min="-3" max="3" step="0.05" value="1"></label>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    c.onresize = () => (key = '');
    let ast = null, fn = null, gf = null, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f, a) => { fn = f; ast = a; key = ''; } });
    const eg = AI.exprInput(q('#guess'), q('#err2'), { allowed: ['x'], onChange: (f) => { gf = f; key = ''; } });
    const PRE = [['x^3', '3*x^2'], ['x^2*sin(x)', '2*x*sin(x) + x^2*cos(x)'], ['sin(x)/x', '(x*cos(x) - sin(x))/x^2'], ['exp(x)', 'exp(x)'], ['log(x)', '1/x'],
      ['sigmoid(x)', 'sigmoid(x)*(1 - sigmoid(x))'], ['tanh(x)', '1 - tanh(x)^2'], ['relu(x)', '(1 + x/abs(x))/2'], ['sin(x^2)', '2*x*cos(x^2)'], ['exp(-x^2)', '-2*x*exp(-x^2)']];
    chips(q('#pre'), PRE.map((p) => ({ name: p[0], p })), (o) => { ei.set(o.p[0]); eg.set(o.p[1]); });
    ei.run(); eg.run();
    AI.loop(root, () => {
      const x0 = +q('#x0').value, k2 = [x0, c.w, AI.col().bg, q('#expr').value, q('#guess').value].join('|');
      if (k2 === key) return; key = k2; q('#xv').textContent = AI.fmt(x0, 2);
      const C = AI.col(), g = c.ctx, half = c.h / 2, d = (x) => AI.autodiff(ast, { x }).varGrad.x ?? NaN;
      g.clearRect(0, 0, c.w, c.h);
      const panel = (oy, f, col, title, tangent) => {
        const cc = { w: c.w, h: half }, [lo, hi] = autoY(f, -5, 5), v = AI.view(cc, -5, 5, lo, hi, { l: 44, r: 12, t: 20, b: 22 });
        g.save(); g.translate(0, oy); AI.axes(g, v); text(g, title, 50, 14, C.muted, 'left', '11px sans-serif');
        g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
        plotFn(g, v, f, col, 2.4, null, -5, 5);
        if (tangent) { const y0 = f(x0), m = d(x0); g.strokeStyle = C.red; g.setLineDash([6, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(x0 - 1.5), v.Y(y0 - 1.5 * m)); g.lineTo(v.X(x0 + 1.5), v.Y(y0 + 1.5 * m)); g.stroke(); g.setLineDash([]); }
        const yy = f(x0); if (Number.isFinite(yy)) { g.fillStyle = C.fg; g.beginPath(); g.arc(v.X(x0), v.Y(yy), 5, 0, 7); g.fill(); }
        g.restore(); return v;
      };
      panel(0, (x) => fn({ x }), C.blue, 'f(x) with its tangent at x₀', true);
      const v2 = panel(half, d, C.green, 'f′(x) from automatic differentiation (green) vs your guess (orange dots)', false);
      g.save(); g.translate(0, half); g.beginPath(); g.rect(v2.pad.l, v2.pad.t, v2.pw, v2.ph); g.clip(); g.fillStyle = C.orange;
      for (let i = 0; i <= 50; i++) { const x = -5 + (10 * i) / 50, y = gf({ x }); if (Number.isFinite(y)) { g.beginPath(); g.arc(v2.X(x), v2.Y(y), 3, 0, 7); g.fill(); } }
      g.restore();
      const ad = d(x0), h = 1e-5, fd = (fn({ x: x0 + h }) - fn({ x: x0 - h })) / (2 * h), gu = gf({ x: x0 });
      let worst = 0, n = 0; for (let i = 0; i <= 200; i++) { const x = -5 + (10 * i) / 200, a = d(x), b = gf({ x }); if (Number.isFinite(a) && Number.isFinite(b)) { worst = Math.max(worst, Math.abs(a - b)); n++; } }
      q('#out').innerHTML = `f(x₀)                = ${AI.fmt(fn({ x: x0 }), 5)}\nexact f′(x₀) (autodiff)= <span class="g">${AI.fmt(ad, 6)}</span>\ncentral difference   = ${AI.fmt(fd, 6)}\nyour guess at x₀      = <span class="o">${AI.fmt(gu, 6)}</span>\nmax |guess−exact| = ${AI.fmt(worst, 5)}\n${worst < 1e-6 ? '<span class="g">✔ guess matches exact</span>' : '<span class="r">✘ differs — compare curves</span>'}`;
    });
  };

  /* =============================================================== chainwiggle */
  AI.labs.chainwiggle = function (root) {
    const { view, ctl } = AI.labShell(root, 'The chain rule as a wiggle passing through machines', 'x → g → u → f → y : each machine stretches the wiggle by its own local rate',
      `<canvas style="height:400px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:3em"></div>`,
      `<label>Inner machine u = g(x) <input type="text" id="g" value="x^2" spellcheck="false"></label><div class="err" id="e1"></div>
       <label>Outer machine y = f(u) <input type="text" id="f" value="sin(u)" spellcheck="false"></label><div class="err" id="e2"></div>
       <label>Base point x₀ <span class="v" id="xv"></span><input type="range" id="x0" min="-2" max="2" step="0.05" value="0.8"></label>
       <label>Wiggle size Δx <span class="v" id="dv"></span><input type="range" id="dx" min="0.01" max="0.5" step="0.01" value="0.2"></label>
       <label>Speed <span class="v">wiggles per second</span><input type="range" id="sp" min="0.03" max="0.5" step="0.01" value="0.12"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ¼ wiggle</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    c.onresize = () => (key = '');
    let gf = null, ga = null, ff = null, fa = null, playing = true, ph = 0, target = null, key = '';
    const e1 = AI.exprInput(q('#g'), q('#e1'), { allowed: ['x'], onChange: (f, a) => { gf = f; ga = a; key = ''; } });
    const e2 = AI.exprInput(q('#f'), q('#e2'), { allowed: ['u'], onChange: (f, a) => { ff = f; fa = a; key = ''; } });
    e1.run(); e2.run();
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; target = null; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = '▶ Play'; target = (Math.floor(ph / (TAU / 4) + 1e-6) + 1) * (TAU / 4); };
    AI.loop(root, (dt) => {
      if (playing) ph += dt * TAU * +q('#sp').value;
      else if (target !== null) { ph += Math.min(dt * TAU * 0.15, target - ph); if (ph >= target - 1e-9) { ph = target; target = null; } }
      const x0 = +q('#x0').value, dx = +q('#dx').value, k2 = [ph.toFixed(3), x0, dx, c.w, AI.col().bg, q('#g').value, q('#f').value].join('|');
      if (k2 === key) return; key = k2;
      q('#xv').textContent = AI.fmt(x0, 2); q('#dv').textContent = AI.fmt(dx, 2);
      const C = AI.col(), g = c.ctx, u0 = gf({ x: x0 }), y0 = ff({ u: u0 });
      const X = x0 + dx * Math.sin(ph), U = gf({ x: X }), Y = ff({ u: U });
      const dX = X - x0, dU = U - u0, dY = Y - y0;
      const amp = (a, b) => Math.max(Math.abs(a), Math.abs(b));
      const aU = amp(gf({ x: x0 + dx }) - u0, gf({ x: x0 - dx }) - u0), aY = amp(ff({ u: gf({ x: x0 + dx }) }) - y0, ff({ u: gf({ x: x0 - dx }) }) - y0);
      const mx = Math.max(dx, aU, aY) || 1, halfH = (c.h - 110) / 2, sc = (0.9 * halfH) / mx, cy = 60 + halfH;
      const gu = AI.autodiff(ga, { x: x0 }).varGrad.x, fu = AI.autodiff(fa, { u: u0 }).varGrad.u;
      g.clearRect(0, 0, c.w, c.h);
      const cols = [c.w * 0.12, c.w * 0.5, c.w * 0.88], boxes = [c.w * 0.31, c.w * 0.69], names = [['x', C.blue, x0, dX, dx], ['u = g(x)', C.orange, u0, dU, aU], ['y = f(u)', C.green, y0, dY, aY]];
      const dots = [X, U, Y], ys = [dX, dU, dY].map((d) => cy - Math.max(-halfH, Math.min(halfH, d * sc)));
      names.forEach(([nm, col, base, d, a], i) => {
        const gx = cols[i];
        g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(gx, cy - halfH); g.lineTo(gx, cy + halfH); g.stroke();
        g.fillStyle = col + '33'; g.fillStyle = C.bg3; g.fillRect(gx - 9, cy - a * sc, 18, 2 * a * sc);
        g.strokeStyle = C.muted; g.lineWidth = 1; g.beginPath(); g.moveTo(gx - 14, cy); g.lineTo(gx + 14, cy); g.stroke();
        g.fillStyle = col; g.beginPath(); g.arc(gx, ys[i], 8, 0, 7); g.fill();
        text(g, nm, gx, 22, col, 'center', 'bold 13px sans-serif'); text(g, 'base ' + AI.fmt(base, 3), gx, 38, C.muted, 'center', '11px ui-monospace,monospace');
        text(g, (d >= 0 ? 'Δ = +' : 'Δ = ') + AI.fmt(d, 4), gx, c.h - 22, col, 'center', '12px ui-monospace,monospace');
        text(g, '±' + AI.fmt(a, 3) + ' range', gx, c.h - 6, C.muted, 'center', '11px ui-monospace,monospace');
      });
      [0, 1].forEach((i) => {
        const bx = boxes[i], by = cy - 26;
        g.strokeStyle = C.muted; g.lineWidth = 1.5; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(cols[i] + 10, ys[i]); g.lineTo(cols[i + 1] - 10, ys[i + 1]); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.bg2; g.strokeStyle = C.border; g.lineWidth = 1.5; g.beginPath(); g.roundRect ? g.roundRect(bx - 48, by, 96, 52, 10) : g.rect(bx - 48, by, 96, 52); g.fill(); g.stroke();
        text(g, i ? 'machine f' : 'machine g', bx, by + 20, C.fg, 'center', 'bold 13px sans-serif');
        text(g, 'local rate ' + AI.fmt(i ? fu : gu, 3), bx, by + 39, i ? C.green : C.orange, 'center', '12px ui-monospace,monospace');
      });
      const r1 = Math.abs(dX) > 1e-9 ? dU / dX : gu, r2 = Math.abs(dU) > 1e-9 ? dY / dU : fu;
      q('#out').innerHTML = `Δx = ${AI.fmt(dX, 5)}
Δu = ${AI.fmt(dU, 5)}
Δy = ${AI.fmt(dY, 5)}\nΔu/Δx = <span class="o">${AI.fmt(r1, 4)}</span>   (g′ = ${AI.fmt(gu, 4)})\nΔy/Δu = <span class="g">${AI.fmt(r2, 4)}</span>   (f′ = ${AI.fmt(fu, 4)})\nlocal rates g′·f′ = <span class="b">${AI.fmt(gu * fu, 5)}</span>\nmeasured  Δy/Δx = ${AI.fmt(Math.abs(dX) > 1e-9 ? dY / dX : gu * fu, 5)}\n<span class="m">chain rule: dy/dx = f′(g(x))·g′(x)</span>`;
      q('#cap').textContent = Math.abs(dX) < 1e-9 ? 'At rest: every gauge sits at its base value.' : `① the input is nudged by Δx = ${AI.fmt(dX, 3)} ② machine g turns that into Δu ≈ ${AI.fmt(gu, 2)} × Δx = ${AI.fmt(dU, 3)} ③ machine f turns Δu into Δy ≈ ${AI.fmt(fu, 2)} × Δu = ${AI.fmt(dY, 3)}. Overall stretch ≈ ${AI.fmt(gu, 2)} × ${AI.fmt(fu, 2)} = ${AI.fmt(gu * fu, 3)}.`;
    });
  };

  /* ---------- contour image helper (shared by gradmap and hessian) ---------- */
  const contourImage = (off, W, H, fn, x0, x1, y0, y1) => {
    off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H), vals = new Float64Array(W * H), dx = (x1 - x0) / W, dy = (y1 - y0) / H;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) vals[j * W + i] = fn({ x: x0 + (i + 0.5) * dx, y: y1 - (j + 0.5) * dy });
    const s = [...vals].filter(Number.isFinite).sort((a, b) => a - b), lo = s[0] ?? 0, hi = AI.quantile(s, 0.96) > lo ? AI.quantile(s, 0.96) : lo + 1, step = (hi - lo) / 14;
    const C = AI.col(), A = AI.rgb(C.bg), B = AI.rgb(C.purple), D = AI.rgb(C.orange), F = AI.rgb(C.fg), gm = [];
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const k = j * W + i, v = vals[k], gx = ((vals[j * W + Math.min(W - 1, i + 1)] - vals[j * W + Math.max(0, i - 1)]) / (2 * dx)), gy = ((vals[Math.max(0, j - 1) * W + i] - vals[Math.min(H - 1, j + 1) * W + i]) / (2 * dy));
      const gn = Math.hypot(gx, gy) + 1e-12; if (i % 7 === 0 && j % 7 === 0 && Number.isFinite(gn)) gm.push(gn);
      const t = AI.clamp(Math.pow(AI.clamp((v - lo) / (hi - lo || 1), 0, 1), 0.55), 0, 1), sN = v / step, fr = Math.abs(sN - Math.round(sN));
      const a = AI.clamp(1 - ((fr * step) / gn / dx) / 0.9, 0, 1) * 0.55;
      for (let ch = 0; ch < 3; ch++) { const base = t < 0.5 ? A[ch] + (B[ch] - A[ch]) * t * 2 : B[ch] + (D[ch] - B[ch]) * (t - 0.5) * 2; im.data[k * 4 + ch] = Number.isFinite(v) ? base * (1 - a) + F[ch] * a : 40; }
      im.data[k * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0); gm.sort((a, b) => a - b); return gm.length ? gm[Math.floor(gm.length / 2)] : 1;
  };
  const MAP_PRE = [
    { name: 'Bowl x²+3y²', e: 'x^2 + 3*y^2', c: [0, 0, 6] }, { name: 'Saddle x²−y²', e: 'x^2 - y^2', c: [0, 0, 6] },
    { name: 'Hills', e: '2*sin(x)*cos(y) + 0.1*(x^2+y^2)', c: [0, 0, 8] }, { name: 'Valley', e: '(1-x)^2 + 5*(y-x^2)^2', c: [0.5, 1, 5] },
    { name: 'Peak exp(−r²)', e: 'exp(-(x^2+y^2)/4)', c: [0, 0, 7] },
  ];

  /* =============================================================== gradmap */
  AI.labs.gradmap = function (root) {
    const { view, ctl } = AI.labShell(root, 'Gradient map', 'drag the dot — see partial slopes, the gradient, and the slope in any direction',
      `<canvas style="height:460px;cursor:crosshair"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>f(x, y) = <input type="text" id="expr" value="x^2 + 3*y^2" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>Direction angle θ <span class="v" id="tv"></span><input type="range" id="th" min="0" max="360" step="1" value="30"></label>
       <div class="row"><label style="flex-direction:row;gap:6px"><input type="checkbox" id="sPart" checked> partials</label><label style="flex-direction:row;gap:6px"><input type="checkbox" id="sField" checked> field</label><label style="flex-direction:row;gap:6px"><input type="checkbox" id="sDir" checked> direction</label></div>
       <div class="row"><button class="btn" id="sweep">↻ Sweep θ</button><button class="btn primary" id="up">▲ Walk uphill</button><button class="btn" id="down">▼ Walk downhill</button></div>
       <label>Walk speed <span class="v">steps / second</span><input type="range" id="sp" min="1" max="12" step="1" value="3"></label>
       <label>Step size η <span class="v" id="ev"></span><input type="range" id="eta" min="0.01" max="0.3" step="0.01" value="0.08"></label>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv), off = document.createElement('canvas'), tick = AI.ticker();
    c.onresize = () => (key = '');
    let ast = null, fn = null, V = { cx: 0, cy: 0, s: 6 }, p = [1.5, 1], dirty = true, gmed = 1, sweeping = false, walk = 0, trail = [], key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x', 'y'], check: (f) => { if (!Number.isFinite(f({ x: 0.3, y: 0.7 }))) throw new Error('f is not finite here'); }, onChange: (f, a) => { fn = f; ast = a; dirty = true; trail = []; } });
    chips(q('#pre'), MAP_PRE, (o) => { V = { cx: o.c[0], cy: o.c[1], s: o.c[2] }; p = [o.c[0] + o.c[2] * 0.25, o.c[1] + o.c[2] * 0.15]; walk = 0; ei.set(o.e); });
    ei.run();
    const geom = () => { const sp = V.s, yh = (sp * c.h) / c.w; return { x0: V.cx - sp / 2, x1: V.cx + sp / 2, y0: V.cy - yh / 2, y1: V.cy + yh / 2, k: c.w / sp }; };
    drag(cv, (px, py) => { const G = geom(); p = [G.x0 + px / G.k, G.y1 - py / G.k]; walk = 0; trail = []; });
    q('#sweep').onclick = () => (sweeping = !sweeping);
    q('#up').onclick = () => { walk = walk === 1 ? 0 : 1; trail = [[...p]]; };
    q('#down').onclick = () => { walk = walk === -1 ? 0 : -1; trail = [[...p]]; };
    window.addEventListener('themechange', () => (dirty = true));
    AI.loop(root, (dt) => {
      if (sweeping) q('#th').value = (+q('#th').value + dt * 25) % 360;
      if (walk) { const n = tick(dt, +q('#sp').value), eta = +q('#eta').value; for (let i = 0; i < n; i++) { const [gx, gy] = gradAt(ast, p[0], p[1]); const gn = Math.hypot(gx, gy); if (gn < 1e-9) break; p = [p[0] + walk * eta * gx, p[1] + walk * eta * gy]; trail.push([...p]); if (trail.length > 300) trail.shift(); } }
      q('#up').textContent = walk === 1 ? '⏸ Stop' : '▲ Walk uphill'; q('#down').textContent = walk === -1 ? '⏸ Stop' : '▼ Walk downhill'; q('#sweep').textContent = sweeping ? '⏸ Stop sweep' : '↻ Sweep θ';
      const th = (+q('#th').value * Math.PI) / 180, flags = ['sPart', 'sField', 'sDir'].map((i) => q('#' + i).checked).join('');
      const k2 = [p[0].toFixed(4), p[1].toFixed(4), th.toFixed(3), flags, c.w, c.h, V.s, dirty, trail.length, q('#eta').value].join('|');
      if (k2 === key) return; key = k2;
      q('#tv').textContent = Math.round((th * 180) / Math.PI) + '°'; q('#ev').textContent = q('#eta').value;
      const G = geom(), C = AI.col(), g = c.ctx, X = (x) => (x - G.x0) * G.k, Y = (y) => (G.y1 - y) * G.k;
      if (dirty) { gmed = contourImage(off, 190, Math.max(40, Math.round((190 * c.h) / c.w)), (s) => fn(s), G.x0, G.x1, G.y0, G.y1); dirty = false; }
      g.clearRect(0, 0, c.w, c.h); g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, c.w, c.h);
      g.font = '10px ui-monospace,monospace';
      for (let i = 0; i <= 6; i++) { const x = G.x0 + ((G.x1 - G.x0) * i) / 6; text(g, AI.fmt(x, 1), X(x), c.h - 3, C.fg, 'center', '10px ui-monospace,monospace'); }
      for (let i = 1; i <= 4; i++) { const y = G.y0 + ((G.y1 - G.y0) * i) / 5; text(g, AI.fmt(y, 1), 3, Y(y) + 3, C.fg, 'left', '10px ui-monospace,monospace'); }
      const [gx, gy, f0] = gradAt(ast, p[0], p[1]), gn = Math.hypot(gx, gy), ks = 38 / gmed, cap = (L) => (L > 130 ? 130 / L : 1);
      if (q('#sField').checked) for (let i = 0; i < 11; i++) for (let j = 0; j < Math.max(4, Math.round((11 * c.h) / c.w)); j++) {
        const x = G.x0 + ((i + 0.5) / 11) * (G.x1 - G.x0), y = G.y1 - ((j + 0.5) / Math.max(4, Math.round((11 * c.h) / c.w))) * (G.y1 - G.y0), [a, b] = gradAt(ast, x, y), n = Math.hypot(a, b);
        if (n > 1e-9 && Number.isFinite(n)) AI.arrow(g, X(x), Y(y), X(x) + (a / n) * 16, Y(y) - (b / n) * 16, C.fg + '99', 1.2);
      }
      if (trail.length > 1) { g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); trail.forEach((t, i) => (i ? g.lineTo(X(t[0]), Y(t[1])) : g.moveTo(X(t[0]), Y(t[1])))); g.stroke(); }
      const px = X(p[0]), py = Y(p[1]), L = gn * ks, cp = cap(L);
      if (gn > 1e-9) {
        const ux = gx / gn, uy = gy / gn; // contour tangent is perpendicular
        g.strokeStyle = C.fg; g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(px + uy * 45, py + ux * 45); g.lineTo(px - uy * 45, py - ux * 45); g.stroke(); g.setLineDash([]);
        g.lineWidth = 1; g.beginPath(); const s = 9; g.moveTo(px + ux * s, py - uy * s); g.lineTo(px + ux * s + uy * s, py - uy * s + ux * s); g.lineTo(px + uy * s, py + ux * s); g.stroke();
      }
      if (q('#sPart').checked) { AI.arrow(g, px, py, px + gx * ks * cp, py, C.orange, 2.5); AI.arrow(g, px, py, px, py - gy * ks * cp, C.blue, 2.5); }
      if (gn > 1e-9) { AI.arrow(g, px, py, px + gx * ks * cp, py - gy * ks * cp, C.green, 3.5); AI.arrow(g, px, py, px - gx * ks * cp * 0.6, py + gy * ks * cp * 0.6, C.red, 2); }
      if (q('#sDir').checked) AI.arrow(g, px, py, px + Math.cos(th) * 55, py - Math.sin(th) * 55, C.pink, 3);
      g.fillStyle = C.fg; g.strokeStyle = C.bg; g.lineWidth = 2; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill(); g.stroke();
      const du = gx * Math.cos(th) + gy * Math.sin(th);
      // mini plot of D_u f versus theta
      const mx = c.w - 168, my = 10, mw = 156, mh = 74; g.fillStyle = C.bg + 'dd'; g.fillRect(mx, my, mw, mh); g.strokeStyle = C.border; g.strokeRect(mx, my, mw, mh);
      const amp = Math.max(gn, 1e-9); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(mx, my + mh / 2); g.lineTo(mx + mw, my + mh / 2); g.stroke();
      g.strokeStyle = C.pink; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 90; i++) { const a = (i / 90) * TAU, y = gx * Math.cos(a) + gy * Math.sin(a); const X2 = mx + (a / TAU) * mw, Y2 = my + mh / 2 - (y / amp) * (mh / 2 - 8); i ? g.lineTo(X2, Y2) : g.moveTo(X2, Y2); } g.stroke();
      g.fillStyle = C.fg; g.beginPath(); g.arc(mx + (th / TAU) * mw, my + mh / 2 - (du / amp) * (mh / 2 - 8), 4, 0, 7); g.fill();
      text(g, 'slope D_u f vs angle θ (0°…360°)', mx + 4, my + 11, C.muted, 'left', '10px sans-serif');
      q('#out').innerHTML = `point  (${AI.fmt(p[0], 3)}, ${AI.fmt(p[1], 3)})   f = ${AI.fmt(f0, 4)}\n∂f/∂x = <span class="o">${AI.fmt(gx, 4)}</span>   ∂f/∂y = <span class="b">${AI.fmt(gy, 4)}</span>\n∇f = <span class="g">(${AI.fmt(gx, 3)}, ${AI.fmt(gy, 3)})</span>   |∇f| = ${AI.fmt(gn, 4)}\nu = (cos θ, sin θ) = (${AI.fmt(Math.cos(th), 3)}, ${AI.fmt(Math.sin(th), 3)})\nD_u f = ∇f·u = <span class="r">${AI.fmt(du, 4)}</span>  (max = |∇f|)`;
      q('#cap').textContent = gn < 1e-6 ? 'Flat spot: the gradient is zero, so no direction goes up or down to first order.' : `green = ∇f (steepest uphill), red = −∇f (steepest downhill), orange/blue = the two partial slopes, dashed line = the contour (always ⟂ ∇f), pink = chosen direction. Slope that way: ${AI.fmt(du, 3)}.`;
    });
  };

  /* =============================================================== jacmap */
  AI.labs.jacmap = function (root) {
    const { view, ctl } = AI.labShell(root, 'A tiny circle becomes an ellipse', 'the Jacobian is the best linear stand-in for a map, close to a point',
      `<canvas style="height:400px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>u(x, y) = <input type="text" id="u" value="x*cos(y)" spellcheck="false"></label><div class="err" id="e1"></div>
       <label>v(x, y) = <input type="text" id="v" value="x*sin(y)" spellcheck="false"></label><div class="err" id="e2"></div>
       <div class="chips" id="pre"></div>
       <label>Circle radius r <span class="v" id="rv"></span><input type="range" id="r" min="-2.3" max="0.1" step="0.01" value="-0.15"></label>
       <label>Tracer speed <span class="v">radians / second</span><input type="range" id="sp" min="0.1" max="1" step="0.05" value="0.4"></label>
       <div class="row"><button class="btn primary" id="shrink">▶ Shrink circle</button><button class="btn" id="pause">⏸ Pause tracer</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    c.onresize = () => (key = '');
    let ua = null, va = null, uf = null, vf = null, p = [1.5, 0.8], shrinking = false, tracer = 0, running = true, key = '', zoomOut = 4, lastOut = 0;
    const e1 = AI.exprInput(q('#u'), q('#e1'), { allowed: ['x', 'y'], onChange: (f, a) => { uf = f; ua = a; key = ''; } });
    const e2 = AI.exprInput(q('#v'), q('#e2'), { allowed: ['x', 'y'], onChange: (f, a) => { vf = f; va = a; key = ''; } });
    chips(q('#pre'), [{ name: 'polar → cartesian', u: 'x*cos(y)', v: 'x*sin(y)', p: [1.5, 0.8] }, { name: 'complex square', u: 'x^2 - y^2', v: '2*x*y', p: [1, 0.7] }, { name: 'tanh layer', u: 'tanh(x + 0.5*y)', v: 'tanh(y - x)', p: [0.3, 0.2] }, { name: 'linear', u: '2*x + y', v: 'x - y', p: [0, 0] }, { name: 'singular (det 0)', u: 'x + y', v: '2*x + 2*y', p: [0.5, 0.5] }],
      (o) => { p = [...o.p]; e1.set(o.u); e2.set(o.v); });
    e1.run(); e2.run();
    q('#shrink').onclick = () => { q('#r').value = 0.1; shrinking = true; };
    q('#r').oninput = () => (shrinking = false);
    q('#pause').onclick = () => { running = !running; q('#pause').textContent = running ? '⏸ Pause tracer' : '▶ Resume tracer'; };
    drag(cv, (px, py) => { if (px < c.w / 2) { const s = c.w / 2 / 6; p = [(px - c.w / 4) / s, (c.h / 2 - py) / s]; } });
    AI.loop(root, (dt) => {
      if (shrinking) { const r = +q('#r').value - dt * 0.12; q('#r').value = Math.max(-2.3, r); if (r <= -2.3) shrinking = false; }
      if (running) tracer = (tracer + dt * +q('#sp').value) % TAU;
      const r = Math.pow(10, +q('#r').value), k2 = [p[0].toFixed(3), p[1].toFixed(3), r.toFixed(5), tracer.toFixed(3), c.w, c.h, AI.col().bg, q('#u').value, q('#v').value].join('|');
      if (k2 === key) return; key = k2;
      q('#rv').textContent = AI.fmt(r, 4);
      const C = AI.col(), g = c.ctx, F = (x, y) => [uf({ x, y }), vf({ x, y })];
      const [ux, uy] = gradAt(ua, p[0], p[1]), [vx, vy] = gradAt(va, p[0], p[1]), c0 = F(p[0], p[1]);
      const N = 72, circ = [], lin = []; let area = 0, dev = 0;
      for (let i = 0; i < N; i++) { const a = (i / N) * TAU, dx = r * Math.cos(a), dy = r * Math.sin(a), o = F(p[0] + dx, p[1] + dy), l = [c0[0] + ux * dx + uy * dy, c0[1] + vx * dx + vy * dy]; circ.push(o); lin.push(l); dev = Math.max(dev, Math.hypot(o[0] - l[0], o[1] - l[1])); }
      for (let i = 0; i < N; i++) { const a = circ[i], b = circ[(i + 1) % N]; area += a[0] * b[1] - b[0] * a[1]; }
      area = area / 2 / (Math.PI * r * r);
      const det = ux * vy - uy * vx, tr = ux * ux + uy * uy + vx * vx + vy * vy, sv = Math.sqrt(Math.max(0, (tr + Math.sqrt(Math.max(0, tr * tr - 4 * det * det))) / 2)), sv2 = sv > 1e-12 ? Math.abs(det) / sv : 0;
      // left panel
      const half = c.w / 2, S = half / 6, lx = (x) => half / 2 + x * S, ly = (y) => c.h / 2 - y * S;
      g.clearRect(0, 0, c.w, c.h);
      const panel = (ox, title) => { g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(ox + 0.5, 0.5, half - 1, c.h - 1); text(g, title, ox + 8, 16, C.muted, 'left', '11px sans-serif'); };
      panel(0, 'input plane (x, y) — drag the centre'); panel(half, 'output plane (u, v) — auto-zoomed to fit');
      for (let i = -5; i <= 5; i++) { g.strokeStyle = C.border; g.beginPath(); g.moveTo(lx(i), 0); g.lineTo(lx(i), c.h); g.moveTo(0, ly(i)); g.lineTo(half, ly(i)); g.stroke(); }
      const dot = (X, Y, rad, col) => { g.fillStyle = col; g.beginPath(); g.arc(X, Y, rad, 0, 7); g.fill(); };
      const hue = (i) => `hsl(${Math.round((i / N) * 360)},75%,60%)`;
      for (let i = 0; i < N; i++) dot(lx(p[0] + r * Math.cos((i / N) * TAU)), ly(p[1] + r * Math.sin((i / N) * TAU)), Math.max(1.5, Math.min(3, r * S * 0.1)), hue(i));
      AI.arrow(g, lx(p[0]), ly(p[1]), lx(p[0] + Math.max(r, 0.06)), ly(p[1]), C.orange, 2); AI.arrow(g, lx(p[0]), ly(p[1]), lx(p[0]), ly(p[1] + Math.max(r, 0.06)), C.blue, 2);
      dot(lx(p[0]), ly(p[1]), 4, C.fg);
      const ti = (tracer / TAU) * N, a0 = Math.floor(ti) % N; dot(lx(p[0] + r * Math.cos(tracer)), ly(p[1] + r * Math.sin(tracer)), 6, C.fg);
      // right panel: auto zoom
      let ext = 0; circ.forEach((o) => (ext = Math.max(ext, Math.abs(o[0] - c0[0]), Math.abs(o[1] - c0[1])))); lin.forEach((o) => (ext = Math.max(ext, Math.abs(o[0] - c0[0]), Math.abs(o[1] - c0[1]))));
      const want = Math.max(ext * 1.35, 1e-7); zoomOut += (want - zoomOut) * 0.12; if (Math.abs(want - zoomOut) > want * 0.003) key = ''; const S2 = Math.min(half, c.h) / 2 / zoomOut * 0.92, rx = (x) => half + half / 2 + (x - c0[0]) * S2, ry = (y) => c.h / 2 - (y - c0[1]) * S2;
      const ste = AI.niceStep(zoomOut * 2, 6);
      g.strokeStyle = C.border; for (let t = -Math.floor(zoomOut / ste) * ste; t <= zoomOut; t += ste) { g.beginPath(); g.moveTo(rx(c0[0] + t), 0); g.lineTo(rx(c0[0] + t), c.h); g.moveTo(half, ry(c0[1] + t)); g.lineTo(2 * half, ry(c0[1] + t)); g.stroke(); }
      text(g, 'grid step ' + AI.fmt(ste, 3) + ' · centre (' + AI.fmt(c0[0], 3) + ', ' + AI.fmt(c0[1], 3) + ')', half + 8, c.h - 8, C.muted, 'left', '10px ui-monospace,monospace');
      g.strokeStyle = C.fg; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); lin.forEach((l, i) => (i ? g.lineTo(rx(l[0]), ry(l[1])) : g.moveTo(rx(l[0]), ry(l[1])))); g.closePath(); g.stroke(); g.setLineDash([]);
      circ.forEach((o, i) => dot(rx(o[0]), ry(o[1]), 3, hue(i)));
      AI.arrow(g, rx(c0[0]), ry(c0[1]), rx(c0[0] + ux * Math.max(r, 1e-9) * 1), ry(c0[1] + vx * Math.max(r, 1e-9)), C.orange, 2.5); AI.arrow(g, rx(c0[0]), ry(c0[1]), rx(c0[0] + uy * r), ry(c0[1] + vy * r), C.blue, 2.5);
      dot(rx(c0[0]), ry(c0[1]), 4, C.fg);
      const to = F(p[0] + r * Math.cos(tracer), p[1] + r * Math.sin(tracer)); g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.moveTo(rx(c0[0]), ry(c0[1])); g.lineTo(rx(to[0]), ry(to[1])); g.stroke(); dot(rx(to[0]), ry(to[1]), 6, C.fg);
      q('#out').innerHTML = `J = [ ∂u/∂x ∂u/∂y ] = [ <span class="o">${AI.fmt(ux, 3)}</span>  <span class="b">${AI.fmt(uy, 3)}</span> ]\n    [ ∂v/∂x ∂v/∂y ]   [ <span class="o">${AI.fmt(vx, 3)}</span>  <span class="b">${AI.fmt(vy, 3)}</span> ]\ndet J = ${AI.fmt(det, 4)}  area ratio = ${AI.fmt(area, 4)}\nstretch factors = ${AI.fmt(sv, 3)}, ${AI.fmt(sv2, 3)}
gap to ellipse = ${AI.fmt(dev, 5)}  (${AI.fmt(dev / r, 3)} r)`;
      q('#cap').textContent = dev / r > 0.05 ? 'The circle is big, so the true image (dots) is a curved blob: the map is not linear over this much room.' : 'The circle is small: the true image (dots) sits on the dashed ellipse predicted by J. Close to a point, every smooth map is a matrix. Orange/blue arrows = where J sends the x- and y-steps.';
    });
  };

  /* =============================================================== hessian */
  AI.labs.hessian = function (root) {
    const { view, ctl } = AI.labShell(root, 'Curvature: second derivatives', 'slice the surface along a direction and compare with the best parabola',
      `<canvas style="height:520px;cursor:crosshair"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>f(x, y) = <input type="text" id="expr" value="x^2 - 0.5*y^2 + 0.2*x*y" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>Slice direction θ <span class="v" id="tv"></span><input type="range" id="th" min="0" max="360" step="1" value="0"></label>
       <label>Sweep speed <span class="v">degrees / second</span><input type="range" id="sp" min="5" max="60" step="1" value="20"></label>
       <div class="row"><button class="btn primary" id="sweep">▶ Rotate slice</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv), off = document.createElement('canvas');
    c.onresize = () => (key = '');
    let ast = null, fn = null, V = { cx: 0, cy: 0, s: 6 }, p = [0.5, 0.5], dirty = true, sweeping = false, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x', 'y'], check: (f) => { if (!Number.isFinite(f({ x: 0.3, y: 0.7 }))) throw new Error('f is not finite here'); }, onChange: (f, a) => { fn = f; ast = a; dirty = true; } });
    chips(q('#pre'), [{ name: 'saddle', e: 'x^2 - 0.5*y^2 + 0.2*x*y', c: [0, 0, 4], p: [0.5, 0.5] }, { name: 'bowl: stiff & flat', e: '3*x^2 + 0.3*y^2', c: [0, 0, 4], p: [0.8, 0.8] }, { name: 'hilltop', e: '-x^2 - y^2 + 0.5*x*y', c: [0, 0, 4], p: [0.5, 0.5] }, { name: 'wavy', e: 'sin(x)*cos(y) + 0.05*(x^2+y^2)', c: [0, 0, 8], p: [1, 0.5] }, { name: 'trough', e: 'x^2', c: [0, 0, 4], p: [1, 0.5] }],
      (o) => { V = { cx: o.c[0], cy: o.c[1], s: o.c[2] }; p = [...o.p]; ei.set(o.e); });
    ei.run();
    const MAPH = 330, SL = 150, geom = () => { const sp = V.s, yh = (sp * MAPH) / c.w; return { x0: V.cx - sp / 2, x1: V.cx + sp / 2, y0: V.cy - yh / 2, y1: V.cy + yh / 2, k: c.w / sp }; };
    drag(cv, (px, py) => { if (py < MAPH) { const G = geom(); p = [G.x0 + px / G.k, G.y1 - py / G.k]; } });
    q('#sweep').onclick = () => (sweeping = !sweeping);
    window.addEventListener('themechange', () => (dirty = true));
    AI.loop(root, (dt) => {
      if (sweeping) q('#th').value = (+q('#th').value + dt * +q('#sp').value) % 360;
      q('#sweep').textContent = sweeping ? '⏸ Stop' : '▶ Rotate slice';
      const th = (+q('#th').value * Math.PI) / 180, k2 = [p[0].toFixed(4), p[1].toFixed(4), th.toFixed(3), c.w, c.h, V.s, dirty, AI.col().bg].join('|');
      if (k2 === key) return; key = k2; q('#tv').textContent = Math.round(+q('#th').value) + '°';
      const G = geom(), C = AI.col(), g = c.ctx, X = (x) => (x - G.x0) * G.k, Y = (y) => (G.y1 - y) * G.k;
      if (dirty) { contourImage(off, 190, Math.round((190 * MAPH) / c.w), (s) => fn(s), G.x0, G.x1, G.y0, G.y1); dirty = false; }
      g.clearRect(0, 0, c.w, c.h); g.drawImage(off, 0, 0, c.w, MAPH);
      const [gx, gy, f0] = gradAt(ast, p[0], p[1]), [hxx, hxy, hyy] = hessAt(ast, p[0], p[1]), E = eig2(hxx, hxy, hyy), u = [Math.cos(th), Math.sin(th)];
      const kap = hxx * u[0] * u[0] + 2 * hxy * u[0] * u[1] + hyy * u[1] * u[1], d1 = gx * u[0] + gy * u[1], px = X(p[0]), py = Y(p[1]);
      [[E.l1, E.v1], [E.l2, E.v2]].forEach(([l, v]) => { const col = l >= 0 ? C.blue : C.orange, L = 70; g.strokeStyle = col; g.lineWidth = 3; g.beginPath(); g.moveTo(px - v[0] * L, py + v[1] * L); g.lineTo(px + v[0] * L, py - v[1] * L); g.stroke(); text(g, 'λ=' + AI.fmt(l, 2), px + v[0] * (L + 6) + (v[0] < 0 ? -36 : 4), py - v[1] * (L + 6) + 4, col, 'left', 'bold 11px ui-monospace,monospace'); });
      AI.arrow(g, px, py, px + u[0] * 60, py - u[1] * 60, C.pink, 3); g.fillStyle = C.fg; g.strokeStyle = C.bg; g.lineWidth = 2; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill(); g.stroke();
      text(g, 'drag the dot · blue axis = curves up (λ>0) · orange axis = curves down (λ<0) · pink = slice direction', 8, 15, C.fg, 'left', '11px sans-serif');
      // slice panel
      const T = 1.5 * (V.s / 6), oy = MAPH + 8, sh = c.h - oy - 8, gF = (t) => fn({ x: p[0] + t * u[0], y: p[1] + t * u[1] }), par = (t) => f0 + d1 * t + 0.5 * kap * t * t;
      let lo = Infinity, hi = -Infinity; for (let i = 0; i <= 80; i++) { const t = -T + (2 * T * i) / 80; [gF(t), par(t)].forEach((y) => { if (Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }); }
      const pd = (hi - lo) * 0.12 + 1e-6, sv = AI.view({ w: c.w, h: sh }, -T, T, lo - pd, hi + pd, { l: 52, r: 12, t: 14, b: 22 });
      g.save(); g.translate(0, oy); g.fillStyle = C.bg; g.fillRect(0, 0, c.w, sh); AI.axes(g, sv); text(g, 'slice g(t) = f(point + t·u)   (blue)   vs   f + g′t + ½κt²   (orange dashed)', 56, 11, C.muted, 'left', '11px sans-serif');
      g.beginPath(); g.rect(sv.pad.l, sv.pad.t, sv.pw, sv.ph); g.clip(); plotFn(g, sv, gF, C.blue, 2.8, null, -T, T, 160); plotFn(g, sv, par, C.orange, 2, [6, 4], -T, T, 80);
      g.fillStyle = C.fg; g.beginPath(); g.arc(sv.X(0), sv.Y(f0), 5, 0, 7); g.fill(); g.restore();
      const kind = Math.hypot(gx, gy) > 0.15 ? 'not a critical point (∇f ≠ 0)' : E.l1 > 1e-6 && E.l2 > 1e-6 ? 'local MINIMUM (both λ > 0)' : E.l1 < -1e-6 && E.l2 < -1e-6 ? 'local MAXIMUM (both λ < 0)' : E.l1 * E.l2 < 0 ? 'SADDLE (λ of opposite sign)' : 'degenerate (λ ≈ 0)';
      q('#out').innerHTML = `∇f = (${AI.fmt(gx, 3)}, ${AI.fmt(gy, 3)})\nH = [ ${AI.fmt(hxx, 3)}  ${AI.fmt(hxy, 3)} ]\n    [ ${AI.fmt(hxy, 3)}  ${AI.fmt(hyy, 3)} ]\neigenvalues λ₁ = <span class="b">${AI.fmt(E.l1, 3)}</span>, λ₂ = <span class="o">${AI.fmt(E.l2, 3)}</span>\ncurvature along u: κ = uᵀHu = <span class="r">${AI.fmt(kap, 4)}</span>\n(between λ₂ and λ₁)\n${kind}`;
      q('#cap').textContent = 'The curvature along the pink direction is κ = uᵀHu. It is largest along the eigenvector of the biggest eigenvalue and smallest along the other: rotate the slice and watch the parabola open and close.';
    });
  };

  /* =============================================================== taylor */
  AI.labs.taylor = function (root) {
    const { view, ctl } = AI.labShell(root, 'Taylor approximation', 'build a better and better polynomial from the derivatives at one point',
      `<canvas style="height:420px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>f(x) = <input type="text" id="expr" value="sin(x)" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>Expansion point a <span class="v" id="av"></span><input type="range" id="a" min="-3" max="3" step="0.05" value="0"></label>
       <label>Order n <span class="v" id="nv"></span><input type="range" id="n" min="0" max="6" step="1" value="1"></label>
       <label>Test point x <span class="v" id="xv"></span><input type="range" id="x" min="-5" max="5" step="0.05" value="1.5"></label>
       <label>Build-up speed <span class="v">seconds per order</span><input type="range" id="sp" min="1" max="6" step="0.5" value="2.5"></label>
       <div class="row"><button class="btn primary" id="play">▶ Build up 0 → 6</button><label style="flex-direction:row;gap:6px"><input type="checkbox" id="old" checked> lower orders</label></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    c.onresize = () => (key = '');
    let ast = null, fn = null, playing = false, acc = 0, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f, a) => { fn = f; ast = a; key = ''; } });
    chips(q('#pre'), ['sin(x)', 'exp(x)', 'cos(x)', 'log(1+x)', '1/(1-x)', 'atan(x)', 'sqrt(1+x)', 'tanh(x)', 'abs(x)'], (p) => { ei.set(p); if (p === 'log(1+x)' || p === 'sqrt(1+x)') q('#a').value = 0; });
    ei.run();
    q('#play').onclick = () => { playing = !playing; if (playing) { q('#n').value = 0; acc = 0; } key = ''; };
    AI.loop(root, (dt) => {
      if (playing) { acc += dt / +q('#sp').value; if (acc >= 1) { acc = 0; const n = +q('#n').value + 1; if (n > 6) playing = false; else q('#n').value = n; } }
      q('#play').textContent = playing ? '⏸ Stop' : '▶ Build up 0 → 6';
      const a = +q('#a').value, n = +q('#n').value, xt = +q('#x').value, old = q('#old').checked, k2 = [a, n, xt, old, c.w, AI.col().bg, q('#expr').value].join('|');
      if (k2 === key) return; key = k2; q('#av').textContent = AI.fmt(a, 2); q('#nv').textContent = n; q('#xv').textContent = AI.fmt(xt, 2);
      const C = AI.col(), g = c.ctx, co = AI.calcJet(ast, a, 6), T = (m) => (x) => { let s = 0, pw = 1; for (let k = 0; k <= m; k++) { s += co[k] * pw; pw *= x - a; } return s; };
      const f = (x) => fn({ x }), [lo, hi] = autoY(f, -5, 5, 12), v = AI.view(c, -5, 5, lo, hi, { l: 44, r: 12, t: 14, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      const Tn = T(n); g.fillStyle = C.red + '26'; g.beginPath(); let started = false;
      for (let i = 0; i <= 200; i++) { const x = -5 + (10 * i) / 200, y = Tn(x), fy = f(x); if (Number.isFinite(y) && Number.isFinite(fy)) { started ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); started = true; } }
      for (let i = 200; i >= 0; i--) { const x = -5 + (10 * i) / 200, fy = f(x); if (Number.isFinite(fy)) g.lineTo(v.X(x), v.Y(fy)); } g.closePath(); g.fill();
      if (old) for (let m = 0; m < n; m++) plotFn(g, v, T(m), C.orange + '66', 1.2, null, -5, 5, 160);
      plotFn(g, v, f, C.blue, 2.8, null, -5, 5, 300); plotFn(g, v, Tn, C.orange, 2.6, null, -5, 5, 200);
      g.strokeStyle = C.muted; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(v.X(a), v.pad.t); g.lineTo(v.X(a), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; [[a, f(a)], [xt, f(xt)]].forEach(([x, y]) => { if (Number.isFinite(y)) { g.beginPath(); g.arc(v.X(x), v.Y(y), 5, 0, 7); g.fill(); } });
      g.fillStyle = C.orange; if (Number.isFinite(Tn(xt))) { g.beginPath(); g.arc(v.X(xt), v.Y(Tn(xt)), 5, 0, 7); g.fill(); }
      g.restore(); text(g, '— f(x)   — order-n polynomial   shaded = error   dashed line = a', v.pad.l + 8, v.pad.t + 12, C.muted, 'left', '11px sans-serif');
      const poly = co.slice(0, n + 1).map((cf, k) => `${AI.fmt(cf, 4)}${k ? '(x−a)' + (k > 1 ? '^' + k : '') : ''}`).join(' + ');
      const err = f(xt) - Tn(xt);
      q('#out').innerHTML = `coefficients f⁽ᵏ⁾(a)/k! :\n${co.map((cf, k) => `  k=${k}: ${AI.fmt(cf, 5)}${k <= n ? '' : '  <span class="m">(not used)</span>'}`).join('\n')}\nT${n}(x) = ${poly}\nf(x) = ${AI.fmt(f(xt), 5)}   T${n}(x) = <span class="o">${AI.fmt(Tn(xt), 5)}</span>\nerror = <span class="r">${AI.fmt(err, 5)}</span>   (distance |x−a| = ${AI.fmt(Math.abs(xt - a), 3)})`;
      q('#cap').textContent = ['Order 0: a flat line at the starting height f(a).', 'Order 1: the tangent line — the “linear approximation” that gradient descent relies on.', 'Order 2: adds curvature (½f″(a)(x−a)²) — the parabola Newton’s method uses.', 'Higher orders keep bending the polynomial to hug f for longer.', 'More orders: the good region around a widens.', 'More orders: the good region around a widens.', 'Order 6: close over a wide window — but never forever for functions like 1/(1−x) or log(1+x).'][n];
    });
  };

  /* =============================================================== riemann */
  AI.labs.riemann = function (root) {
    const { view, ctl } = AI.labShell(root, 'Area as a sum of thin rectangles', 'more, thinner rectangles → the exact area under the curve',
      `<canvas style="height:400px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>f(x) = <input type="text" id="expr" value="x^2/4 + 1" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>from a <span class="v" id="av"></span><input type="range" id="a" min="-3" max="3" step="0.1" value="0"></label>
       <label>to b <span class="v" id="bv"></span><input type="range" id="b" min="-3" max="5" step="0.1" value="4"></label>
       <label>Rectangles n <span class="v" id="nv"></span><input type="range" id="n" min="1" max="200" step="1" value="4"></label>
       <label>Rule <select id="m"><option value="L">left endpoints</option><option value="R">right endpoints</option><option value="M">midpoints</option><option value="T">trapezoids</option></select></label>
       <label>Speed <span class="v">sweep length in seconds</span><input type="range" id="sp" min="1" max="8" step="0.5" value="4"></label>
       <div class="row"><button class="btn primary" id="play">▶ Refine automatically</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    c.onresize = () => (key = '');
    let fn = null, auto = false, reveal = 1, phase = 'sweep', hold = 0, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f) => { fn = f; key = ''; } });
    chips(q('#pre'), ['x^2/4 + 1', 'sin(x) + 1.5', 'exp(-x^2)*3', 'x*(4 - x)', 'cos(2*x)'], (p) => ei.set(p));
    ei.run();
    q('#play').onclick = () => { auto = !auto; if (auto) { q('#n').value = 1; reveal = 0; phase = 'sweep'; } else reveal = 1; key = ''; };
    q('#n').oninput = () => { auto = false; reveal = 1; };
    AI.loop(root, (dt) => {
      if (auto) {
        const sp = +q('#sp').value;
        if (phase === 'sweep') { reveal += dt / sp; if (reveal >= 1) { reveal = 1; phase = 'hold'; hold = 0; } }
        else { hold += dt; if (hold > sp * 0.3) { const n = +q('#n').value; if (n >= 128) { auto = false; } else { q('#n').value = Math.min(200, n * 2); reveal = 0; phase = 'sweep'; } } }
      }
      q('#play').textContent = auto ? '⏸ Stop' : '▶ Refine automatically';
      const a = Math.min(+q('#a').value, +q('#b').value - 0.1), b = +q('#b').value, n = +q('#n').value, m = q('#m').value, k2 = [a, b, n, m, reveal.toFixed(3), c.w, AI.col().bg, q('#expr').value].join('|');
      if (k2 === key) return; key = k2; q('#av').textContent = AI.fmt(a, 1); q('#bv').textContent = AI.fmt(b, 1); q('#nv').textContent = n;
      const C = AI.col(), g = c.ctx, f = (x) => fn({ x }), [lo, hi] = autoY(f, -3.5, 5.5, 20), v = AI.view(c, -3.5, 5.5, Math.min(lo, -0.5), Math.max(hi, 0.5), { l: 44, r: 12, t: 14, b: 24 });
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      const dx = (b - a) / n, shown = reveal * n; let sum = 0, full = 0;
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      for (let i = 0; i < n; i++) {
        const xl = a + i * dx, xr = xl + dx, h = m === 'L' ? f(xl) : m === 'R' ? f(xr) : m === 'M' ? f((xl + xr) / 2) : null, tl = f(xl), tr = f(xr);
        const val = m === 'T' ? ((tl + tr) / 2) * dx : h * dx; if (!Number.isFinite(val)) continue; full += val;
        const frac = AI.clamp(shown - i, 0, 1); if (frac <= 0) continue; sum += val * frac;
        g.fillStyle = (m === 'T' ? (tl + tr) / 2 : h) >= 0 ? C.green + '55' : C.orange + '66'; g.strokeStyle = C.green; g.lineWidth = n > 60 ? 0.5 : 1.2;
        const x1 = xl + dx * frac;
        g.beginPath(); if (m === 'T') { g.moveTo(v.X(xl), v.Y(0)); g.lineTo(v.X(xl), v.Y(tl)); g.lineTo(v.X(x1), v.Y(tl + (tr - tl) * frac)); g.lineTo(v.X(x1), v.Y(0)); } else g.rect(v.X(xl), v.Y(Math.max(h, 0)), v.X(x1) - v.X(xl), Math.abs(v.Y(h) - v.Y(0)));
        g.closePath(); g.fill(); if (n <= 100) g.stroke();
      }
      plotFn(g, v, f, C.blue, 2.8, null, -3.5, 5.5, 300);
      g.strokeStyle = C.muted; g.setLineDash([3, 4]); [a, b].forEach((x) => { g.beginPath(); g.moveTo(v.X(x), v.pad.t); g.lineTo(v.X(x), v.pad.t + v.ph); g.stroke(); }); g.setLineDash([]);
      g.restore();
      let ex = 0; const S = 2000, hh = (b - a) / S; for (let i = 0; i <= S; i++) { const wgt = i === 0 || i === S ? 1 : i % 2 ? 4 : 2, y = f(a + i * hh); ex += wgt * (Number.isFinite(y) ? y : 0); } ex *= hh / 3;
      q('#out').innerHTML = `width Δx = (b−a)/n = ${AI.fmt(dx, 4)}\nrunning sum so far = <span class="g">${AI.fmt(sum, 5)}</span>\nsum with all n rectangles = ${AI.fmt(full, 5)}\n“exact” ∫ₐᵇ f dx ≈ <span class="b">${AI.fmt(ex, 5)}</span>\nerror = <span class="r">${AI.fmt(full - ex, 5)}</span>`;
      q('#cap').textContent = `${n} rectangle${n > 1 ? 's' : ''} of width ${AI.fmt(dx, 3)}: add up (height × width) for each. Green = above the axis (adds), orange = below (subtracts). Error shrinks as n grows.`;
    });
  };

  /* =============================================================== ftc */
  AI.labs.ftc = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fundamental theorem of calculus', 'the area collected so far, A(t), grows at exactly the rate f(t)',
      `<canvas style="height:460px"></canvas><div class="caption" id="cap" style="padding:0 12px 10px;min-height:2.6em"></div>`,
      `<label>f(x) = <input type="text" id="expr" value="cos(x) + 0.5" spellcheck="false"></label><div class="err" id="err"></div><div class="chips" id="pre"></div>
       <label>Start a <span class="v" id="av"></span><input type="range" id="a" min="-1" max="3" step="0.1" value="0"></label>
       <label>Sweep position t <span class="v" id="tv"></span><input type="range" id="t" min="-1" max="7" step="0.01" value="1.5"></label>
       <label>Sweep speed <span class="v">x-units / second</span><input type="range" id="sp" min="0.1" max="1.5" step="0.05" value="0.4"></label>
       <div class="row"><button class="btn primary" id="play">▶ Sweep t</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    c.onresize = () => (key = '');
    let fn = null, playing = false, key = '';
    const ei = AI.exprInput(q('#expr'), q('#err'), { allowed: ['x'], onChange: (f) => { fn = f; key = ''; } });
    chips(q('#pre'), ['cos(x) + 0.5', 'x*(4 - x)/2', '2*exp(-(x-2)^2)', 'sin(2*x)', '1 + x/3'], (p) => ei.set(p));
    ei.run();
    q('#play').onclick = () => { playing = !playing; if (playing && +q('#t').value > 6.5) q('#t').value = -1; };
    AI.loop(root, (dt) => {
      if (playing) { const t = +q('#t').value + dt * +q('#sp').value; q('#t').value = Math.min(7, t); if (t >= 7) playing = false; }
      q('#play').textContent = playing ? '⏸ Pause' : '▶ Sweep t';
      const a = +q('#a').value, t = Math.max(+q('#t').value, a), k2 = [a, t.toFixed(3), c.w, AI.col().bg, q('#expr').value].join('|');
      if (k2 === key) return; key = k2; q('#av').textContent = AI.fmt(a, 1); q('#tv').textContent = AI.fmt(t, 2);
      const C = AI.col(), g = c.ctx, f = (x) => fn({ x }), half = c.h / 2, X0 = -1, X1 = 7;
      const N = 560, xs = [], A = []; let acc = 0; // cumulative integral from a
      for (let i = 0; i <= N; i++) { const x = X0 + ((X1 - X0) * i) / N; xs.push(x); }
      const Af = (x) => { if (x <= a) return 0; const M = 400, h = (x - a) / M; let s = 0; for (let i = 0; i < M; i++) { const y = f(a + (i + 0.5) * h); s += Number.isFinite(y) ? y : 0; } return s * h; };
      let lo = Infinity, hi = -Infinity; xs.forEach((x, i) => { if (i % 4 === 0) { const y = Af(x); lo = Math.min(lo, y); hi = Math.max(hi, y); } });
      const [flo, fhi] = autoY(f, X0, X1, 20), cc = { w: c.w, h: half };
      const v1 = AI.view(cc, X0, X1, Math.min(flo, -0.3), Math.max(fhi, 0.3), { l: 44, r: 12, t: 18, b: 20 }), v2 = AI.view(cc, X0, X1, Math.min(lo, -0.5) - 0.3, Math.max(hi, 0.5) + 0.3, { l: 44, r: 12, t: 18, b: 22 });
      g.clearRect(0, 0, c.w, c.h);
      g.save(); AI.axes(g, v1); text(g, 'f(x) — shaded: area from a to t', 50, 12, C.muted, 'left', '11px sans-serif');
      g.beginPath(); g.rect(v1.pad.l, v1.pad.t, v1.pw, v1.ph); g.clip();
      g.beginPath(); g.moveTo(v1.X(a), v1.Y(0)); for (let i = 0; i <= 120; i++) { const x = a + ((t - a) * i) / 120, y = f(x); g.lineTo(v1.X(x), v1.Y(Number.isFinite(y) ? y : 0)); } g.lineTo(v1.X(t), v1.Y(0)); g.closePath(); g.fillStyle = C.green + '55'; g.fill();
      plotFn(g, v1, f, C.blue, 2.8, null, X0, X1); g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(v1.X(t), v1.Y(0)); g.lineTo(v1.X(t), v1.Y(f(t))); g.stroke(); g.restore();
      g.save(); g.translate(0, half); AI.axes(g, v2); text(g, 'A(t) = ∫ from a to t of f — the area so far', 50, 12, C.muted, 'left', '11px sans-serif');
      g.beginPath(); g.rect(v2.pad.l, v2.pad.t, v2.pw, v2.ph); g.clip(); plotFn(g, v2, Af, C.green, 2.8, null, X0, X1, 200);
      const At = Af(t), m = f(t), L = 1.1; g.strokeStyle = C.red; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v2.X(t - L), v2.Y(At - m * L)); g.lineTo(v2.X(t + L), v2.Y(At + m * L)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.beginPath(); g.arc(v2.X(t), v2.Y(At), 6, 0, 7); g.fill(); g.restore();
      const h = 1e-3, sl = (Af(t + h) - Af(Math.max(a, t - h))) / (t + h - Math.max(a, t - h));
      q('#out').innerHTML = `t = ${AI.fmt(t, 3)}\nA(t) = area so far = <span class="g">${AI.fmt(At, 5)}</span>\nheight f(t)       = <span class="r">${AI.fmt(m, 5)}</span>\nslope of A at t   = <span class="r">${AI.fmt(sl, 5)}</span>   ← equal!`;
      q('#cap').textContent = m > 0.02 ? 'f(t) > 0: the area is growing, so A slopes upward — steeply where f(t) is tall.' : m < -0.02 ? 'f(t) < 0: new area counts negatively, so A slopes downward.' : 'f(t) ≈ 0: no new area is being added, so A is flat: a peak or valley of A.';
    });
  };
})();
