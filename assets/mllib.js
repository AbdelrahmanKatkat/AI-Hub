/* Shared machine-learning toolkit for the ML course labs. Pure data/algorithm code + small canvas helpers.
 * Works in the browser (window.ML) and in node (module.exports) for the pure parts.
 *
 *   ML.data(kind, n, seed, opts)     2-D classification sets: moons circles blobs xor spiral linear checker  -> {X:[[x,y]...], y:[int...], K}
 *   ML.reg(kind, n, seed)            1-D regression sets: linear quad sine step outliers                    -> {x:[...], y:[...]}
 *   ML.knn(X,y,k)                    -> {proba(p)->[K probs]}
 *   ML.tree.fit(X,y,opts)            -> tree {root,depth,leaves, predictProba(p), path(p)}   opts: maxDepth, minLeaf, criterion('gini'|'entropy')
 *   ML.tree.bestSplit(X,y,idx,crit)  -> {feature,threshold,gain,left,right} (one greedy step, for animations)
 *   ML.kmeans(X,k,seed)              -> object with .step() (assign+update), .C centroids, .labels, .inertia()
 *   ML.logistic(X,y,opts)            -> object with .step(lr) one gradient step, .w, .b, .loss(), .proba(p)
 *   ML.linreg(x,y)                   -> {w,b} closed form (1-D)
 *   ML.pca2d(X)                      -> {mean, vals:[l1,l2], vecs:[[..],[..]]}
 *   ML.boundary(canvasCtx, w, h, proba(x,y)->[K probs], {range:1.3, res:56, K})   paints a probability heat-map (browser)
 *   ML.points(ctx, w, h, data, {range:1.3, r:4})                                   draws the labelled points (browser)
 *   ML.COLORS(K)                     -> K css colours from the theme (browser)
 */
(function (root) {
  const ML = {};
  const rng = (seed) => { let s = (seed | 0) + 1; return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const randn = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  ML.rng = rng; ML.randn = randn;

  /* ---------- datasets ---------- */
  ML.data = function (kind, n = 200, seed = 1, o = {}) {
    const r = rng(seed), noise = o.noise ?? 0.08, X = [], y = [];
    let K = 2;
    for (let i = 0; i < n; i++) {
      let x, yy, c = i % 2;
      switch (kind) {
        case 'moons': { const t = r() * Math.PI; if (c) { x = 1 - Math.cos(t) - 0.5; yy = 0.3 - Math.sin(t) * 0.8 + 0.1 - 0.5; x = x * 1.0; yy = -Math.sin(t) * 0.8 + 0.1; x = 0.5 - Math.cos(t) * 0.9 + 0; } else { x = Math.cos(t) * 0.9 - 0.5; yy = Math.sin(t) * 0.8 - 0.1; } x += noise * randn(r); yy += noise * randn(r); break; }
        case 'circles': { const a = r() * 6.283, rad = c ? 0.35 : 0.85; x = rad * Math.cos(a) + noise * randn(r); yy = rad * Math.sin(a) + noise * randn(r); break; }
        case 'blobs': { K = o.K || 3; c = i % K; const a = (c / K) * 6.283 + 0.5; x = 0.6 * Math.cos(a) + 0.22 * randn(r); yy = 0.6 * Math.sin(a) + 0.22 * randn(r); break; }
        case 'xor': { x = r() * 2 - 1; yy = r() * 2 - 1; c = x * yy > 0 ? 1 : 0; x += 0.06 * Math.sign(x); yy += 0.06 * Math.sign(yy); break; }
        case 'spiral': { const t = (Math.floor(i / 2) / (n / 2)) * 3 + 0.3, a = t * 2.2 + c * Math.PI; x = (t / 3.3) * Math.cos(a) * 1.15 + noise * 0.4 * randn(r); yy = (t / 3.3) * Math.sin(a) * 1.15 + noise * 0.4 * randn(r); break; }
        case 'checker': { x = r() * 2 - 1; yy = r() * 2 - 1; c = (Math.floor((x + 1) * 2) + Math.floor((yy + 1) * 2)) % 2; break; }
        default: { const s = c ? 0.42 : -0.42; x = s + 0.3 * randn(r); yy = s * 0.7 + 0.3 * randn(r); } // 'linear': two overlapping gaussians
      }
      X.push([x, yy]); y.push(c);
    }
    return { X, y, K };
  };
  ML.reg = function (kind, n = 40, seed = 1) {
    const r = rng(seed), x = [], y = [];
    for (let i = 0; i < n; i++) {
      const xi = -3 + (6 * i) / (n - 1) + (r() - 0.5) * 0.1; let v;
      switch (kind) {
        case 'quad': v = 0.6 * xi * xi - 1; break; case 'sine': v = 2 * Math.sin(1.3 * xi); break; case 'step': v = xi < 0 ? -1 : 1.5; break;
        default: v = 1.5 * xi + 0.5;
      }
      v += 0.35 * randn(r); if (kind === 'outliers' && r() < 0.1) v += 5 * (r() - 0.5) * 2;
      x.push(xi); y.push(v);
    }
    return { x, y };
  };

  /* ---------- k-NN ---------- */
  ML.knn = function (X, y, k = 5) {
    const K = Math.max(...y) + 1;
    return {
      proba(p) {
        const d = X.map((q, i) => [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, i]).sort((a, b) => a[0] - b[0]).slice(0, k), out = new Array(K).fill(0);
        d.forEach(([, i]) => out[y[i]]++); return out.map((v) => v / k);
      },
      neighbours(p) { return X.map((q, i) => [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, i]).sort((a, b) => a[0] - b[0]).slice(0, k).map((d) => d[1]); },
    };
  };

  /* ---------- decision tree (CART) ---------- */
  const impurity = (counts, crit) => {
    const n = counts.reduce((a, b) => a + b, 0); if (!n) return 0;
    if (crit === 'entropy') return -counts.reduce((a, c) => (c ? a + (c / n) * Math.log2(c / n) : a), 0);
    return 1 - counts.reduce((a, c) => a + (c / n) ** 2, 0);
  };
  const tcount = (y, idx, K) => { const c = new Array(K).fill(0); idx.forEach((i) => c[y[i]]++); return c; };
  ML.tree = {
    impurity,
    bestSplit(X, y, idx, crit = 'gini') {
      const K = Math.max(...y) + 1, parent = impurity(tcount(y, idx, K), crit); let best = null;
      for (let f = 0; f < X[0].length; f++) {
        const s = [...idx].sort((a, b) => X[a][f] - X[b][f]), left = new Array(K).fill(0), right = tcount(y, idx, K);
        for (let j = 0; j < s.length - 1; j++) {
          left[y[s[j]]]++; right[y[s[j]]]--;
          if (X[s[j]][f] === X[s[j + 1]][f]) continue;
          const nl = j + 1, nr = s.length - nl, w = (nl * impurity(left, crit) + nr * impurity(right, crit)) / s.length, gain = parent - w;
          if (!best || gain > best.gain + 1e-12) best = { feature: f, threshold: (X[s[j]][f] + X[s[j + 1]][f]) / 2, gain, left: s.slice(0, nl), right: s.slice(nl) };
        }
      }
      return best;
    },
    fit(X, y, o = {}) {
      const maxDepth = o.maxDepth ?? 3, minLeaf = o.minLeaf ?? 1, crit = o.criterion || 'gini', K = Math.max(...y) + 1;
      let leaves = 0, depth = 0;
      const grow = (idx, d) => {
        const counts = tcount(y, idx, K), node = { n: idx.length, counts, imp: impurity(counts, crit), depth: d, idx };
        depth = Math.max(depth, d);
        const sp = d < maxDepth && node.imp > 0 && idx.length >= 2 * minLeaf ? ML.tree.bestSplit(X, y, idx, crit) : null;
        if (!sp || sp.gain <= 1e-12 || sp.left.length < minLeaf || sp.right.length < minLeaf) { node.leaf = true; leaves++; return node; }
        node.feature = sp.feature; node.threshold = sp.threshold; node.gain = sp.gain;
        node.left = grow(sp.left, d + 1); node.right = grow(sp.right, d + 1); return node;
      };
      const rootNode = grow(X.map((_, i) => i), 0);
      const walk = (p) => { let n = rootNode; const path = [n]; while (!n.leaf) { n = p[n.feature] <= n.threshold ? n.left : n.right; path.push(n); } return path; };
      return { root: rootNode, depth, leaves, path: walk, predictProba: (p) => { const c = walk(p).pop().counts, s = c.reduce((a, b) => a + b, 0); return c.map((v) => v / s); } };
    },
  };

  /* ---------- k-means (step-wise, for animation) ---------- */
  ML.kmeans = function (X, k, seed = 1) {
    const r = rng(seed), C = []; const pool = X.map((_, i) => i);
    for (let j = 0; j < k; j++) { const i = pool.splice(Math.floor(r() * pool.length), 1)[0]; C.push([...X[i]]); }
    const o = { C, labels: new Array(X.length).fill(-1), iter: 0, phase: 'assign' };
    o.assign = () => { let ch = 0; X.forEach((p, i) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; b = j; } }); if (o.labels[i] !== b) ch++; o.labels[i] = b; }); return ch; };
    o.update = () => { C.forEach((c, j) => { let sx = 0, sy = 0, n = 0; X.forEach((p, i) => { if (o.labels[i] === j) { sx += p[0]; sy += p[1]; n++; } }); if (n) { c[0] = sx / n; c[1] = sy / n; } }); };
    o.step = () => { if (o.phase === 'assign') { const ch = o.assign(); o.phase = 'update'; return ch; } o.update(); o.phase = 'assign'; o.iter++; return -1; };
    o.inertia = () => X.reduce((s, p, i) => (o.labels[i] < 0 ? s : s + (p[0] - C[o.labels[i]][0]) ** 2 + (p[1] - C[o.labels[i]][1]) ** 2), 0);
    return o;
  };

  /* ---------- logistic regression ---------- */
  ML.logistic = function (X, y, o = {}) {
    const m = { w: [0, 0], b: 0, steps: 0, l2: o.l2 || 0 }, sig = (z) => 1 / (1 + Math.exp(-z));
    m.proba = (p) => { const q = sig(m.w[0] * p[0] + m.w[1] * p[1] + m.b); return [1 - q, q]; };
    m.loss = () => { let s = 0; X.forEach((p, i) => { const q = m.proba(p)[1]; s -= y[i] ? Math.log(q + 1e-12) : Math.log(1 - q + 1e-12); }); return s / X.length + 0.5 * m.l2 * (m.w[0] ** 2 + m.w[1] ** 2); };
    m.step = (lr = 0.5) => { let g0 = 0, g1 = 0, gb = 0; X.forEach((p, i) => { const e = m.proba(p)[1] - y[i]; g0 += e * p[0]; g1 += e * p[1]; gb += e; }); const n = X.length; m.w[0] -= lr * (g0 / n + m.l2 * m.w[0]); m.w[1] -= lr * (g1 / n + m.l2 * m.w[1]); m.b -= lr * gb / n; m.steps++; };
    return m;
  };

  /* ---------- 1-D linear regression & 2-D PCA ---------- */
  ML.linreg = function (x, y) {
    const n = x.length, mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0; x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; });
    const w = sxy / sxx; return { w, b: my - w * mx };
  };
  ML.pca2d = function (X) {
    const n = X.length, mx = X.reduce((a, p) => a + p[0], 0) / n, my = X.reduce((a, p) => a + p[1], 0) / n;
    let a = 0, b = 0, c = 0; X.forEach((p) => { const dx = p[0] - mx, dy = p[1] - my; a += dx * dx; b += dx * dy; c += dy * dy; }); a /= n - 1; b /= n - 1; c /= n - 1;
    const tr = a + c, det = a * c - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const vec = (l) => { let v = Math.abs(b) > 1e-12 ? [l - c, b] : (a >= c ? (l === l1 ? [1, 0] : [0, 1]) : (l === l1 ? [0, 1] : [1, 0])); const m = Math.hypot(v[0], v[1]); return [v[0] / m, v[1] / m]; };
    return { mean: [mx, my], vals: [l1, l2], vecs: [vec(l1), vec(l2)] };
  };

  /* ---------- browser helpers ---------- */
  ML.COLORS = (K = 2) => { const C = window.AI ? AI.col() : {}; return [C.blue, C.orange, C.green, C.purple, C.pink, C.teal].slice(0, K); };
  ML.boundary = function (ctx, w, h, proba, o = {}) {
    const R = o.range || 1.3, res = o.res || 56, K = o.K || 2, cols = ML.COLORS(K).map((c) => AI.rgb(c)), bg = AI.rgb(AI.col().bg);
    const off = ML._off || (ML._off = document.createElement('canvas')); off.width = off.height = res;
    const g = off.getContext('2d'), im = g.createImageData(res, res);
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      const x = ((i + 0.5) / res) * 2 * R - R, y = R - ((j + 0.5) / res) * 2 * R, pr = proba(x, y), top = Math.max(...pr), s = 0.2 + 0.55 * ((top - 1 / K) / (1 - 1 / K));
      for (let c = 0; c < 3; c++) { let v = 0; for (let k = 0; k < K; k++) v += pr[k] * cols[k][c]; im.data[(j * res + i) * 4 + c] = bg[c] + (v - bg[c]) * s; }
      im.data[(j * res + i) * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, w, h);
  };
  ML.toPx = (w, h, R = 1.3) => ({ X: (x) => ((x + R) / (2 * R)) * w, Y: (y) => ((R - y) / (2 * R)) * h, ix: (px) => (px / w) * 2 * R - R, iy: (py) => R - (py / h) * 2 * R });
  ML.points = function (ctx, w, h, data, o = {}) {
    const m = ML.toPx(w, h, o.range || 1.3), cols = ML.COLORS(data.K || 2), C = AI.col();
    data.X.forEach((p, i) => { ctx.fillStyle = cols[data.y[i]]; ctx.strokeStyle = C.fg; ctx.lineWidth = 1; ctx.globalAlpha = o.alpha ?? 1; ctx.beginPath(); ctx.arc(m.X(p[0]), m.Y(p[1]), o.r || 4, 0, 7); ctx.fill(); ctx.globalAlpha = 0.6; ctx.stroke(); ctx.globalAlpha = 1; });
  };

  root.ML = ML;
  if (typeof module !== 'undefined') module.exports = ML;
})(typeof window !== 'undefined' ? window : globalThis);
