/* AI Learning Hub — shared helpers: expression parser + autodiff, canvas/plot utilities, UI glue. */
(function () {
  const AI = (window.AI = { labs: {} });
  AI.$ = (s, r = document) => r.querySelector(s);
  AI.$$ = (s, r = document) => [...r.querySelectorAll(s)];
  AI.clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  AI.lerp = (a, b, t) => a + (b - a) * t;
  AI.smooth = (t) => { t = AI.clamp(t, 0, 1); return t * t * (3 - 2 * t); };

  AI.fmt = (v, d = 3) => {
    if (v === undefined || v === null) return '–';
    if (Number.isNaN(v)) return 'NaN';
    if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
    return String(+v.toFixed(d));
  };

  /* ---------- colours (read from CSS variables so light/dark both work) ---------- */
  let colCache = null;
  AI.col = () => {
    if (colCache) return colCache;
    const g = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    colCache = {};
    ['bg', 'bg2', 'bg3', 'border', 'fg', 'muted', 'blue', 'green', 'purple', 'orange', 'yellow', 'red', 'pink', 'teal'].forEach((k) => (colCache[k] = g('--' + k)));
    return colCache;
  };
  window.addEventListener('themechange', () => (colCache = null));
  AI.rgb = (hex) => {
    const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex);
    if (!m) return [128, 128, 128];
    let h = m[1];
    if (h.length === 3) h = h.replace(/./g, '$&$&');
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  AI.mix = (a, b, t) => {
    const A = AI.rgb(a), B = AI.rgb(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
  };

  /* ---------- random ---------- */
  AI.rng = (seed) => () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  AI.randn = (r = Math.random) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

  /* ---------- expression parser ---------- */
  const F1 = {
    sin: [Math.sin, (x) => Math.cos(x)],
    cos: [Math.cos, (x) => -Math.sin(x)],
    tan: [Math.tan, (x) => 1 / Math.cos(x) ** 2],
    exp: [Math.exp, Math.exp],
    log: [Math.log, (x) => 1 / x],
    sqrt: [Math.sqrt, (x) => 0.5 / Math.sqrt(x)],
    abs: [Math.abs, (x) => Math.sign(x)],
    tanh: [Math.tanh, (x) => 1 - Math.tanh(x) ** 2],
    sigmoid: [(x) => 1 / (1 + Math.exp(-x)), (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }],
    relu: [(x) => Math.max(0, x), (x) => (x > 0 ? 1 : 0)],
    atan: [Math.atan, (x) => 1 / (1 + x * x)],
  };
  F1.ln = F1.log;
  const F2 = {
    max: [Math.max, (a, b) => (a >= b ? [1, 0] : [0, 1])],
    min: [Math.min, (a, b) => (a <= b ? [1, 0] : [0, 1])],
  };
  const CONST = { pi: Math.PI, e: Math.E };
  let uid = 0;

  function tokenize(src) {
    const re = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)|([A-Za-z_][A-Za-z_0-9]*)|(.))/y;
    const out = [];
    let m;
    re.lastIndex = 0;
    while (re.lastIndex < src.length && (m = re.exec(src))) {
      if (m[1] !== undefined) out.push({ k: 'num', v: parseFloat(m[1]) });
      else if (m[2] !== undefined) out.push({ k: 'id', v: m[2] });
      else if (m[3] !== undefined) out.push({ k: 'op', v: m[3] });
    }
    return out;
  }

  AI.parse = function (src) {
    const tk = tokenize(src);
    let p = 0;
    const peek = () => tk[p];
    const isOp = (v) => tk[p] && tk[p].k === 'op' && tk[p].v === v;
    const node = (o) => ((o.id = ++uid), o);
    function expr() {
      let a = term();
      while (isOp('+') || isOp('-')) {
        const op = tk[p++].v;
        a = node({ t: 'bin', op, a, b: term() });
      }
      return a;
    }
    function startsFactor() {
      const t = peek();
      return t && (t.k === 'num' || t.k === 'id' || (t.k === 'op' && t.v === '('));
    }
    function term() {
      let a = unary();
      for (;;) {
        if (isOp('*') || isOp('/')) {
          const op = tk[p++].v;
          a = node({ t: 'bin', op, a, b: unary() });
        } else if (startsFactor()) a = node({ t: 'bin', op: '*', a, b: unary() }); // implicit multiplication: 2x, 3(x+1)
        else return a;
      }
    }
    function unary() {
      if (isOp('-')) { p++; return node({ t: 'neg', a: unary() }); }
      if (isOp('+')) { p++; return unary(); }
      return power();
    }
    function power() {
      const a = primary();
      if (isOp('^')) { p++; return node({ t: 'bin', op: '^', a, b: unary() }); }
      return a;
    }
    function primary() {
      const t = tk[p++];
      if (!t) throw new Error('Unexpected end of expression');
      if (t.k === 'num') return node({ t: 'num', v: t.v });
      if (t.k === 'id') {
        if (isOp('(') && (F1[t.v] || F2[t.v])) {
          p++;
          const a = expr();
          if (F2[t.v]) {
            if (!isOp(',')) throw new Error(`${t.v}() needs two arguments`);
            p++;
            const b = expr();
            if (!isOp(')')) throw new Error('Missing )');
            p++;
            return node({ t: 'fn2', n: t.v, a, b });
          }
          if (!isOp(')')) throw new Error('Missing )');
          p++;
          return node({ t: 'fn', n: t.v, a });
        }
        if (F1[t.v] || F2[t.v]) throw new Error(`${t.v} needs parentheses, e.g. ${t.v}(x)`);
        return node({ t: 'var', n: t.v });
      }
      if (t.v === '(') {
        const a = expr();
        if (!isOp(')')) throw new Error('Missing )');
        p++;
        return a;
      }
      throw new Error(`Unexpected "${t.v}"`);
    }
    if (!tk.length) throw new Error('Type an expression');
    const ast = expr();
    if (p < tk.length) throw new Error(`Unexpected "${tk[p].v}"`);
    return ast;
  };

  AI.vars = function (ast, set = new Set()) {
    if (ast.t === 'var' && !(ast.n in CONST)) set.add(ast.n);
    if (ast.a) AI.vars(ast.a, set);
    if (ast.b) AI.vars(ast.b, set);
    return set;
  };

  /** Compile to a fast closure f(scope). `allowed` (optional) restricts variable names. */
  AI.compile = function (ast, allowed) {
    if (allowed) for (const v of AI.vars(ast)) if (!allowed.includes(v)) throw new Error(`Unknown variable "${v}" (use: ${allowed.join(', ')})`);
    const build = (n) => {
      switch (n.t) {
        case 'num': { const v = n.v; return () => v; }
        case 'var': { const k = n.n; if (k in CONST) { const c = CONST[k]; return () => c; } return (s) => s[k]; }
        case 'neg': { const a = build(n.a); return (s) => -a(s); }
        case 'fn': { const f = F1[n.n][0], a = build(n.a); return (s) => f(a(s)); }
        case 'fn2': { const f = F2[n.n][0], a = build(n.a), b = build(n.b); return (s) => f(a(s), b(s)); }
        case 'bin': {
          const a = build(n.a), b = build(n.b);
          switch (n.op) {
            case '+': return (s) => a(s) + b(s);
            case '-': return (s) => a(s) - b(s);
            case '*': return (s) => a(s) * b(s);
            case '/': return (s) => a(s) / b(s);
            default: return (s) => Math.pow(a(s), b(s));
          }
        }
      }
    };
    return build(ast);
  };

  /** Reverse-mode autodiff over the expression tree. Returns node values/grads plus d(out)/d(var). */
  AI.autodiff = function (ast, scope) {
    const val = new Map(), grad = new Map(), varGrad = {};
    const fwd = (n) => {
      let v;
      switch (n.t) {
        case 'num': v = n.v; break;
        case 'var': v = n.n in CONST ? CONST[n.n] : scope[n.n]; break;
        case 'neg': v = -fwd(n.a); break;
        case 'fn': v = F1[n.n][0](fwd(n.a)); break;
        case 'fn2': v = F2[n.n][0](fwd(n.a), fwd(n.b)); break;
        case 'bin': {
          const a = fwd(n.a), b = fwd(n.b);
          v = n.op === '+' ? a + b : n.op === '-' ? a - b : n.op === '*' ? a * b : n.op === '/' ? a / b : Math.pow(a, b);
        }
      }
      val.set(n.id, v);
      return v;
    };
    const bwd = (n, g) => {
      grad.set(n.id, g);
      switch (n.t) {
        case 'var': if (!(n.n in CONST)) varGrad[n.n] = (varGrad[n.n] || 0) + g; break;
        case 'neg': bwd(n.a, -g); break;
        case 'fn': bwd(n.a, g * F1[n.n][1](val.get(n.a.id))); break;
        case 'fn2': { const [da, db] = F2[n.n][1](val.get(n.a.id), val.get(n.b.id)); bwd(n.a, g * da); bwd(n.b, g * db); break; }
        case 'bin': {
          const a = val.get(n.a.id), b = val.get(n.b.id);
          if (n.op === '+') { bwd(n.a, g); bwd(n.b, g); }
          else if (n.op === '-') { bwd(n.a, g); bwd(n.b, -g); }
          else if (n.op === '*') { bwd(n.a, g * b); bwd(n.b, g * a); }
          else if (n.op === '/') { bwd(n.a, g / b); bwd(n.b, (-g * a) / (b * b)); }
          else { bwd(n.a, g * b * Math.pow(a, b - 1)); bwd(n.b, a > 0 ? g * Math.pow(a, b) * Math.log(a) : 0); }
        }
      }
    };
    const value = fwd(ast);
    bwd(ast, 1);
    return { value, val, grad, varGrad };
  };

  /** Wire a text input to an expression: validates, shows errors, calls onChange(fn, ast) when valid. */
  AI.exprInput = function (input, errEl, opts) {
    const o = { fn: null, ast: null };
    const run = () => {
      try {
        const ast = AI.parse(input.value);
        const fn = AI.compile(ast, opts.allowed);
        if (opts.check) opts.check(fn, ast);
        o.fn = fn; o.ast = ast;
        input.classList.remove('bad');
        errEl.textContent = '';
        opts.onChange(fn, ast);
      } catch (e) {
        input.classList.add('bad');
        errEl.textContent = e.message;
      }
    };
    input.addEventListener('input', run);
    o.set = (txt) => { input.value = txt; run(); };
    o.run = run;
    return o;
  };

  /* ---------- canvas / plotting ---------- */
  AI.canvas = function (cv) {
    const ctx = cv.getContext('2d');
    const o = { ctx, cv, w: 0, h: 0 };
    const fit = () => {
      const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      o.w = r.width; o.h = r.height;
      cv.width = Math.max(1, Math.round(r.width * d));
      cv.height = Math.max(1, Math.round(r.height * d));
      ctx.setTransform(d, 0, 0, d, 0, 0);
      if (o.onresize) o.onresize();
    };
    new ResizeObserver(fit).observe(cv);
    fit();
    return o;
  };

  /** Animation loop that only runs while `el` is on screen. fn(dt, time). */
  AI.loop = function (el, fn, opts) {
    let vis = true, last = performance.now();
    const gap = 1000 / ((opts && opts.fps) || 60) - 2; // opts.fps caps the frame rate for heavy labs
    new IntersectionObserver((e) => (vis = e[0].isIntersecting)).observe(el);
    const tick = (t) => {
      requestAnimationFrame(tick);
      if (t - last < gap) return;
      const dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      if (vis && !document.hidden) fn(dt, t / 1000);
    };
    requestAnimationFrame(tick);
  };

  /** Fixed-rate stepping: const n = ticker(dt, perSecond) -> whole steps to run this frame (frame-rate independent). */
  AI.ticker = () => { let acc = 0; return (dt, perSec) => { acc += dt * perSec; const n = Math.floor(acc); acc -= n; return Math.min(n, 40); }; };

  AI.niceStep = (range, target = 6) => {
    const raw = range / target, pow = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / pow;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * pow;
  };

  /** Linear view mapping from data space to pixel space. */
  AI.view = (c, xmin, xmax, ymin, ymax, pad = { l: 40, r: 12, t: 12, b: 28 }) => {
    const pw = c.w - pad.l - pad.r, ph = c.h - pad.t - pad.b;
    return {
      xmin, xmax, ymin, ymax, pad, pw, ph,
      X: (x) => pad.l + ((x - xmin) / (xmax - xmin)) * pw,
      Y: (y) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * ph,
      ix: (px) => xmin + ((px - pad.l) / pw) * (xmax - xmin),
      iy: (py) => ymin + (1 - (py - pad.t) / ph) * (ymax - ymin),
    };
  };

  AI.axes = (ctx, v, opts = {}) => {
    const C = AI.col();
    ctx.save();
    ctx.font = '11px ui-monospace,Menlo,Consolas,monospace';
    ctx.lineWidth = 1;
    const sx = AI.niceStep(v.xmax - v.xmin, Math.max(3, v.pw / 80)), sy = AI.niceStep(v.ymax - v.ymin, Math.max(3, v.ph / 50));
    ctx.fillStyle = C.muted;
    ctx.textAlign = 'center';
    for (let x = Math.ceil(v.xmin / sx) * sx; x <= v.xmax + 1e-9; x += sx) {
      const px = Math.round(v.X(x)) + 0.5;
      ctx.strokeStyle = Math.abs(x) < sx / 1e3 ? C.muted : C.border;
      if (!opts.noGrid) { ctx.beginPath(); ctx.moveTo(px, v.pad.t); ctx.lineTo(px, v.pad.t + v.ph); ctx.stroke(); }
      ctx.fillText(AI.fmt(+x.toFixed(6), 2), px, v.pad.t + v.ph + 15);
    }
    ctx.textAlign = 'right';
    if (!opts.noY) for (let y = Math.ceil(v.ymin / sy) * sy; y <= v.ymax + 1e-9; y += sy) {
      const py = Math.round(v.Y(y)) + 0.5;
      ctx.strokeStyle = Math.abs(y) < sy / 1e3 ? C.muted : C.border;
      if (!opts.noGrid) { ctx.beginPath(); ctx.moveTo(v.pad.l, py); ctx.lineTo(v.pad.l + v.pw, py); ctx.stroke(); }
      ctx.fillText(AI.fmt(+y.toFixed(6), 2), v.pad.l - 5, py + 4);
    }
    ctx.restore();
  };

  AI.arrow = (ctx, x0, y0, x1, y1, color, w = 2) => {
    const a = Math.atan2(y1 - y0, x1 - x0), h = 8;
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * h * 0.6, y1 - Math.sin(a) * h * 0.6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * Math.cos(a - 0.45), y1 - h * Math.sin(a - 0.45));
    ctx.lineTo(x1 - h * Math.cos(a + 0.45), y1 - h * Math.sin(a + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  };

  AI.stats = (arr) => {
    const n = arr.length;
    if (!n) return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN, skew: NaN, kurt: NaN };
    let s = 0, mn = Infinity, mx = -Infinity;
    for (const v of arr) { s += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    const mean = s / n;
    let m2 = 0, m3 = 0, m4 = 0;
    for (const v of arr) { const d = v - mean; m2 += d * d; m3 += d ** 3; m4 += d ** 4; }
    m2 /= n; m3 /= n; m4 /= n;
    const std = Math.sqrt(m2);
    return { n, mean, std, min: mn, max: mx, skew: std ? m3 / std ** 3 : 0, kurt: std ? m4 / m2 ** 2 - 3 : 0 };
  };
  AI.quantile = (sorted, q) => sorted[AI.clamp(Math.floor(q * (sorted.length - 1)), 0, sorted.length - 1)];

  /** Lab chrome: standard header + body grid. Returns {view, ctl}. */
  AI.labShell = (root, title, sub, viewHTML, ctlHTML) => {
    root.classList.add('lab');
    root.innerHTML = `<div class="lab-head"><i class="live-dot"></i><b>${title}</b><span>${sub}</span></div>
      <div class="lab-body"><div class="lab-view">${viewHTML}</div><div class="lab-ctl">${ctlHTML}</div></div>`;
    return { view: root.querySelector('.lab-view'), ctl: root.querySelector('.lab-ctl') };
  };
})();
