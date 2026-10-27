/* Module 6 (Neural networks) labs: perceptron, activations, training loop, init/normalisation, regularisation, optimisers,
 * tabular MLP (embeddings), debugging.  Registered as AI.labs.mln*.  All animation is dt-based, slow by default, captioned. */
(function () {
  const C = () => AI.col();
  const $ = (s, r) => AI.$(s, r);
  const T = (ctx, s, x, y, col, al, font) => { ctx.save(); ctx.fillStyle = col; ctx.textAlign = al || 'left'; ctx.font = font || '12px ui-monospace,Menlo,Consolas,monospace'; ctx.fillText(s, x, y); ctx.restore(); };
  const f3 = (v) => AI.fmt(v, 3);
  const ease = (dt, k) => 1 - Math.exp(-dt * k);
  const SM = '11px ui-monospace,Menlo,Consolas,monospace';

  /* ---------- activations ---------- */
  const sg = (z) => 1 / (1 + Math.exp(-z));
  const SQ = Math.sqrt(2 / Math.PI);
  const ACT = {
    sigmoid: { f: sg, d: (z) => { const s = sg(z); return s * (1 - s); } },
    tanh: { f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
    relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    leaky: { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01) },
    gelu: {
      f: (z) => 0.5 * z * (1 + Math.tanh(SQ * (z + 0.044715 * z ** 3))),
      d: (z) => { const u = SQ * (z + 0.044715 * z ** 3), t = Math.tanh(u); return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * SQ * (1 + 0.134145 * z * z); },
    },
    silu: { f: (z) => z * sg(z), d: (z) => { const s = sg(z); return s * (1 + z * (1 - s)); } },
  };

  /* ---------- tiny MLP (flat arrays, per-example forward/backward, SGD or Adam) ---------- */
  function mkNet(sizes, act, seed, init) {
    const r = AI.rng(seed || 1), L = sizes.length - 1;
    const net = { sizes, act, a: ACT[act] || ACT.tanh, L, W: [], b: [], gW: [], gb: [], mW: [], vW: [], mb: [], vb: [], Z: [], A: [new Float64Array(sizes[0])], M: [], t: 0, gn: [] };
    for (let l = 0; l < L; l++) {
      const ni = sizes[l], no = sizes[l + 1];
      const sd = init === 'zero' ? 0 : init === 'tiny' ? 0.01 : init === 'large' ? 0.5 : init === 'xavier' ? Math.sqrt(2 / (ni + no)) : Math.sqrt(2 / ni);
      const W = new Float64Array(ni * no);
      for (let i = 0; i < W.length; i++) W[i] = sd * AI.randn(r);
      net.W.push(W); net.b.push(new Float64Array(no));
      net.gW.push(new Float64Array(ni * no)); net.gb.push(new Float64Array(no));
      net.mW.push(new Float64Array(ni * no)); net.vW.push(new Float64Array(ni * no));
      net.mb.push(new Float64Array(no)); net.vb.push(new Float64Array(no));
      net.Z.push(new Float64Array(no)); net.A.push(new Float64Array(no)); net.M.push(new Float64Array(no).fill(1));
    }
    net.fwd = function (x, drop, rnd) {
      const A = net.A;
      for (let i = 0; i < x.length; i++) A[0][i] = x[i];
      for (let l = 0; l < L; l++) {
        const ni = sizes[l], no = sizes[l + 1], W = net.W[l], Z = net.Z[l], Ao = A[l + 1], Ai = A[l];
        for (let j = 0; j < no; j++) {
          let s = net.b[l][j];
          for (let i = 0; i < ni; i++) s += W[j * ni + i] * Ai[i];
          Z[j] = s;
          if (l < L - 1) {
            let v = net.a.f(s);
            if (drop > 0) { const m = rnd() < drop ? 0 : 1 / (1 - drop); net.M[l][j] = m; v *= m; } else net.M[l][j] = 1;
            Ao[j] = v;
          } else Ao[j] = s;
        }
      }
      return A[L];
    };
    net.bwd = function (dout) {
      let delta = Float64Array.from(dout);
      for (let l = L - 1; l >= 0; l--) {
        const ni = sizes[l], no = sizes[l + 1], W = net.W[l], gW = net.gW[l], Ai = net.A[l];
        for (let j = 0; j < no; j++) { net.gb[l][j] += delta[j]; for (let i = 0; i < ni; i++) gW[j * ni + i] += delta[j] * Ai[i]; }
        if (l > 0) {
          const nd = new Float64Array(ni);
          for (let i = 0; i < ni; i++) { let s = 0; for (let j = 0; j < no; j++) s += W[j * ni + i] * delta[j]; nd[i] = s * net.M[l - 1][i] * net.a.d(net.Z[l - 1][i]); }
          delta = nd;
        }
      }
    };
    /* o: {opt:'sgd'|'adam', wd, clip}; returns global gradient norm (before clipping); net.gn = per-layer norms */
    net.step = function (lr, n, o) {
      o = o || {};
      let tot = 0;
      for (let l = 0; l < L; l++) {
        let s = 0;
        for (let i = 0; i < net.gW[l].length; i++) { net.gW[l][i] /= n; s += net.gW[l][i] ** 2; }
        for (let i = 0; i < net.gb[l].length; i++) { net.gb[l][i] /= n; s += net.gb[l][i] ** 2; }
        net.gn[l] = Math.sqrt(s); tot += s;
      }
      tot = Math.sqrt(tot);
      const sc = o.clip && tot > o.clip ? o.clip / tot : 1;
      net.t++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
      const upd = (P, G, M, V, decay) => {
        for (let i = 0; i < P.length; i++) {
          const g = G[i] * sc;
          if (o.opt === 'adam') {
            M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g;
            P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + 1e-8) + (decay ? lr * (o.wd || 0) * P[i] : 0);
          } else P[i] -= lr * (g + (decay ? (o.wd || 0) * P[i] : 0));
          G[i] = 0;
        }
      };
      for (let l = 0; l < L; l++) { upd(net.W[l], net.gW[l], net.mW[l], net.vW[l], true); upd(net.b[l], net.gb[l], net.mb[l], net.vb[l], false); }
      return tot;
    };
    net.snap = () => ({ W: net.W.map((w) => Float64Array.from(w)), b: net.b.map((w) => Float64Array.from(w)) });
    net.restore = (s) => { s.W.forEach((w, l) => net.W[l].set(w)); s.b.forEach((w, l) => net.b[l].set(w)); };
    net.meanAbsW = () => { let s = 0, n = 0; net.W.forEach((w) => w.forEach((v) => { s += Math.abs(v); n++; })); return s / n; };
    return net;
  }
  AI.mln = { mkNet, ACT };

  /* play/pause + single step plumbing */
  function transport(root, onStep) {
    const o = { playing: true };
    const pb = $('#play', root), sb = $('#step', root);
    if (pb) pb.onclick = () => { o.playing = !o.playing; pb.textContent = o.playing ? '⏸ Pause' : '▶ Play'; };
    if (sb) sb.onclick = () => { o.playing = false; if (pb) pb.textContent = '▶ Play'; onStep(); };
    return o;
  }
  function clipPoly(poly, a, b, c) { // keep the part with a*x+b*y+c > 0
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length], dp = a * p[0] + b * p[1] + c, dq = a * q[0] + b * q[1] + c;
      if (dp > 0) out.push(p);
      if ((dp > 0) !== (dq > 0)) { const t = dp / (dp - dq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    }
    return out;
  }

  /* ======================================================================================= */
  /* Lab 1: perceptron                                                                       */
  /* ======================================================================================= */
  AI.labs.mlnPerceptron = function (root) {
    AI.labShell(root, 'Perceptron learning rule', 'the line moves only when the perceptron makes a mistake',
      `<canvas id="cv" style="height:340px"></canvas>`,
      `<label>Rule to learn <select id="ds"><option>AND</option><option>OR</option><option>XOR</option></select></label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="3" step="0.1" value="0.6"></label>
       <label>Learning rate η <span class="v" id="etav"></span><input type="range" id="eta" min="0.1" max="2" step="0.1" value="1"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    const DS = { AND: [0, 0, 0, 1], OR: [0, 1, 1, 1], XOR: [0, 1, 1, 0] };
    const XS = [[0, 0], [0, 1], [1, 0], [1, 1]];
    let w, b, dw, db, k, epoch, mist, done, cur, flash, hist, why, tot, ys;
    const reset = () => { ys = DS[q('#ds').value]; w = [0, 0]; b = 0; dw = [0, 0]; db = 0; k = 0; epoch = 1; mist = 0; done = false; cur = -1; flash = 0; hist = []; tot = 0; why = 'Press Play. Start: w = (0,0), b = 0. The perceptron predicts 1 when z = w·x + b > 0.'; };
    const doStep = () => {
      if (done) return;
      const x = XS[k], y = ys[k], eta = +q('#eta').value, z = w[0] * x[0] + w[1] * x[1] + b, yh = z > 0 ? 1 : 0, e = y - yh;
      const zs = `z = ${f3(w[0])}·${x[0]} + ${f3(w[1])}·${x[1]} + ${f3(b)} = ${f3(z)}`;
      cur = k; flash = 1;
      if (e) {
        w = [w[0] + eta * e * x[0], w[1] + eta * e * x[1]]; b += eta * e; mist++; tot++;
        why = `Epoch ${epoch}, example (${x}) label ${y}: ${zs} ⇒ predicts ${yh} ✗. e = ${e > 0 ? '+1' : '−1'}, so w ← w ${e > 0 ? '+' : '−'} η·x and b ← b ${e > 0 ? '+' : '−'} η. New w = (${f3(w[0])}, ${f3(w[1])}), b = ${f3(b)}.`;
      } else why = `Epoch ${epoch}, example (${x}) label ${y}: ${zs} ⇒ predicts ${yh} ✓. Correct, so nothing changes.`;
      k++;
      if (k === 4) {
        hist.push(mist);
        if (mist === 0) { done = true; why += ` ✔ A whole epoch without a mistake: converged.`; }
        else if (epoch >= 40) { done = true; why += ' ✘ 40 epochs and still making mistakes: no straight line separates these classes, so the rule never converges.'; }
        else { epoch++; mist = 0; k = 0; }
      }
    };
    q('#ds').onchange = reset; q('#reset').onclick = reset;
    const tr = transport(root, doStep), tick = AI.ticker();
    reset();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' examples/s'; q('#etav').textContent = q('#eta').value;
      if (tr.playing) { const n = tick(dt, sp); for (let i = 0; i < n; i++) doStep(); }
      const e = ease(dt, 5); dw = [dw[0] + (w[0] - dw[0]) * e, dw[1] + (w[1] - dw[1]) * e]; db += (b - db) * e; flash = Math.max(0, flash - dt * 0.6);
      const c = C(), ctx = cv.ctx, v = AI.view(cv, -0.6, 1.6, -0.6, 1.6), nz = Math.abs(dw[0]) + Math.abs(dw[1]) > 1e-6;
      ctx.clearRect(0, 0, cv.w, cv.h);
      const rect = [[v.xmin, v.ymin], [v.xmax, v.ymin], [v.xmax, v.ymax], [v.xmin, v.ymax]];
      ctx.fillStyle = AI.mix(c.bg2, c.orange, 0.22);
      if (nz) {
        const poly = clipPoly(rect, dw[0], dw[1], db);
        if (poly.length > 2) { ctx.beginPath(); poly.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](v.X(p[0]), v.Y(p[1]))); ctx.fill(); }
      } else if (db > 0) ctx.fillRect(v.pad.l, v.pad.t, v.pw, v.ph);
      AI.axes(ctx, v);
      if (nz) {
        const n = Math.hypot(dw[0], dw[1]), mx = -db * dw[0] / (n * n), my = -db * dw[1] / (n * n), dx = -dw[1] / n, dy = dw[0] / n, L = 6;
        ctx.save(); ctx.beginPath(); ctx.rect(v.pad.l, v.pad.t, v.pw, v.ph); ctx.clip();
        ctx.strokeStyle = c.fg; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(v.X(mx - L * dx), v.Y(my - L * dy)); ctx.lineTo(v.X(mx + L * dx), v.Y(my + L * dy)); ctx.stroke(); ctx.restore();
        AI.arrow(ctx, v.X(mx), v.Y(my), v.X(mx + dw[0] / n * 0.3), v.Y(my + dw[1] / n * 0.3), c.green, 2);
      }
      XS.forEach((x, i) => {
        const px = v.X(x[0]), py = v.Y(x[1]), z = dw[0] * x[0] + dw[1] * x[1] + db, ok = (z > 0 ? 1 : 0) === ys[i];
        if (i === cur && flash > 0) { ctx.strokeStyle = c.yellow; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 20 + 6 * Math.sin(flash * 9), 0, 7); ctx.stroke(); }
        ctx.fillStyle = ys[i] ? c.orange : c.blue; ctx.beginPath(); ctx.arc(px, py, 12, 0, 7); ctx.fill();
        ctx.strokeStyle = ok ? c.green : c.red; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 15, 0, 7); ctx.stroke();
        T(ctx, `(${x}) → ${ys[i]}`, px, py - 24, c.muted, 'center', SM);
      });
      T(ctx, 'orange = 1, blue = 0 · ring green = right, red = wrong · shaded = predicts 1', v.pad.l + 4, 12, c.muted, 'left', '10px ui-monospace,monospace');
      q('#out').innerHTML = `w = (${f3(w[0])}, ${f3(w[1])}) · b = ${f3(b)}<br>epoch ${epoch} · updates so far ${tot}<br>mistakes per epoch: ${hist.join(', ') || '–'}`;
      q('#cap').textContent = why;
    });
  };

  /* ======================================================================================= */
  /* Lab 2: activation gallery                                                               */
  /* ======================================================================================= */
  AI.labs.mlnActivations = function (root) {
    AI.labShell(root, 'Activation gallery', 'function, slope, saturation, dead units, and learning',
      `<canvas id="plot" style="height:250px"></canvas><canvas id="demo" style="height:230px"></canvas>`,
      `<label>Activation <select id="act"><option>sigmoid</option><option>tanh</option><option selected>relu</option><option>leaky</option><option>gelu</option><option>silu</option></select></label>
       <label>Input z <span class="v" id="zv"></span><input type="range" id="z" min="-6" max="6" step="0.05" value="1.5"></label>
       <label>Demo speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="2" step="0.05" value="0.7"></label>
       <div class="row"><button class="btn primary" id="sat">Saturation demo</button><button class="btn" id="dead">Dead ReLU demo</button><button class="btn" id="trn">Train tiny net</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cp = AI.canvas(q('#plot')), cd = AI.canvas(q('#demo'));
    let mode = 'sat', t0 = 0, curves = {}, tr = null, dirty = true;
    const names = Object.keys(ACT), COLS = ['blue', 'orange', 'green', 'purple', 'red', 'teal'];
    const start = (m) => {
      mode = m; t0 = 0; tr = null;
      if (m === 'train') { const nm = q('#act').value; tr = { net: mkNet([1, 8, 8, 1], nm, 3, nm === 'sigmoid' || nm === 'tanh' ? 'xavier' : 'he'), step: 0, loss: [] }; curves[nm] = tr.loss; }
    };
    q('#sat').onclick = () => start('sat'); q('#dead').onclick = () => start('dead'); q('#trn').onclick = () => start('train');
    q('#act').onchange = () => { dirty = true; if (mode !== 'train') start(mode); }; q('#z').oninput = () => (dirty = true);
    cp.onresize = () => (dirty = true);
    const tick = AI.ticker();
    const XT = Array.from({ length: 20 }, (_, i) => -1 + (2 * i) / 19), YT = XT.map((x) => Math.sin(2.5 * x));
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = '×' + sp; q('#zv').textContent = q('#z').value;
      const c = C(), nm = q('#act').value, a = ACT[nm], z = +q('#z').value;
      if (dirty) {
        dirty = false;
        const ctx = cp.ctx, v = AI.view(cp, -6, 6, -1.5, 3);
        ctx.clearRect(0, 0, cp.w, cp.h); AI.axes(ctx, v);
        ctx.fillStyle = AI.mix(c.bg2, c.red, 0.4);
        let on = false, x0 = 0;
        for (let i = 0; i <= 240; i++) { const x = -6 + i * 0.05, flat = a.d(x) < 0.05; if (flat && !on) { on = true; x0 = x; } if ((!flat || i === 240) && on) { on = false; ctx.fillRect(v.X(x0), v.pad.t + v.ph - 8, v.X(x) - v.X(x0), 8); } }
        const plot = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath(); for (let i = 0; i <= 240; i++) { const x = -6 + i * 0.05; ctx[i ? 'lineTo' : 'moveTo'](v.X(x), v.Y(AI.clamp(fn(x), -1.5, 3))); } ctx.stroke(); };
        plot(a.f, c.blue); plot(a.d, c.orange);
        const fz = a.f(z), dz = a.d(z);
        ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted; ctx.beginPath(); ctx.moveTo(v.X(z), v.Y(-1.5)); ctx.lineTo(v.X(z), v.Y(3)); ctx.stroke(); ctx.setLineDash([]);
        [[fz, c.blue], [dz, c.orange]].forEach(([y, col]) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(v.X(z), v.Y(AI.clamp(y, -1.5, 3)), 6, 0, 7); ctx.fill(); });
        T(ctx, 'f(z)', v.pad.l + 8, v.pad.t + 14, c.blue); T(ctx, 'f′(z) slope', v.pad.l + 60, v.pad.t + 14, c.orange);
        T(ctx, 'red band = slope < 0.05', v.pad.l + v.pw - 6, v.pad.t + v.ph - 12, c.red, 'right', SM);
        q('#out').innerHTML = `${nm}(${f3(z)}) = ${f3(fz)}<br>slope f′(${f3(z)}) = ${f3(dz)}${dz < 0.05 ? ' ⚠ almost blocked' : ''}`;
      }
      const ctx = cd.ctx;
      ctx.clearRect(0, 0, cd.w, cd.h);
      if (mode === 'sat') {
        t0 += dt * sp;
        const s = a.d(z), L = 6, per = 1.2, W = Math.max(60, cd.w - 170), p = t0 / per;
        T(ctx, `Gradient 1.0 flows back through ${L} ${nm} layers, × f′(z) = ${f3(s)} each`, 8, 18, c.fg, 'left', SM);
        for (let k = 0; k <= L; k++) {
          const val = s ** k, y = 34 + k * 27, fr = AI.clamp(p - k + 1, 0, 1);
          T(ctx, k === 0 ? 'output' : `layer ${L - k + 1}`, 8, y + 14, c.muted, 'left', SM);
          ctx.fillStyle = c.bg3; ctx.fillRect(70, y, W, 18);
          if (fr > 0) { ctx.fillStyle = val < 0.05 ? c.red : c.green; ctx.fillRect(70, y, Math.max(1.5, W * Math.min(1, val) * AI.smooth(fr)), 18); if (fr >= 1) T(ctx, val.toExponential(1), 76 + W, y + 14, c.fg, 'left', SM); }
          if (p >= k - 0.5 && p < k + 0.5) { ctx.fillStyle = c.yellow; ctx.beginPath(); ctx.arc(66, y + 9, 4, 0, 7); ctx.fill(); }
        }
        q('#cap').textContent = s >= 0.99 ? `Slope ≈ 1 (ReLU on its active side): the signal arrives intact, ${f3(s ** L)} after ${L} layers.` : `Slope ${f3(s)} per layer ⇒ after ${L} layers the gradient is ${(s ** L).toExponential(2)} of the original. Early layers barely learn. Move z toward 0 or change the activation.`;
      } else if (mode === 'dead') {
        t0 += dt * sp;
        const bsh = AI.lerp(0.8, -3.2, AI.smooth(t0 / 7)), N = 9, v = AI.view(cd, -1.2, 1.2, -1.2, 1.5);
        AI.axes(ctx, v);
        ctx.strokeStyle = c.blue; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 100; i++) { const x = -1.2 + i * 0.024; ctx[i ? 'lineTo' : 'moveTo'](v.X(x), v.Y(AI.clamp(a.f(x + bsh), -1.2, 1.5))); } ctx.stroke();
        let g = 0, alive = 0;
        for (let i = 0; i < N; i++) { const x = -1 + (2 * i) / (N - 1), zz = x + bsh, d = a.d(zz); g += d / N; if (d > 0.5) alive++; ctx.fillStyle = d > 0.5 ? c.green : d > 0 ? c.yellow : c.red; ctx.beginPath(); ctx.arc(v.X(x), v.Y(AI.clamp(a.f(zz), -1.2, 1.5)), 6, 0, 7); ctx.fill(); }
        T(ctx, `One neuron a = ${nm}(x + b) on 9 data points; bias b drifts down to ${f3(bsh)}`, 8, 16, c.fg, 'left', SM);
        T(ctx, `mean gradient for b = ${g.toExponential(2)}   (${alive}/9 points active)`, 8, cd.h - 8, g === 0 ? c.red : c.fg, 'left', '12px ui-monospace,monospace');
        q('#cap').textContent = g === 0 ? 'Every point has z < 0 ⇒ slope exactly 0 ⇒ gradient 0 ⇒ the bias and weights never move again: a dead ReLU. Try Leaky/GELU/SiLU: a small slope remains.' : 'As b falls, points slide into the flat negative region (red) where the slope is zero. Green dots still pass gradient.';
      } else if (mode === 'train') {
        const N = 300;
        if (tr && tr.step < N) {
          const n = tick(dt, 30 * sp);
          for (let i = 0; i < n && tr.step < N; i++) {
            let L = 0;
            for (let j = 0; j < XT.length; j++) { const o = tr.net.fwd([XT[j]], 0)[0]; L += (o - YT[j]) ** 2 / XT.length; tr.net.bwd([2 * (o - YT[j])]); }
            tr.net.step(0.02, XT.length, { opt: 'adam' }); tr.loss.push(L); tr.step++;
          }
        }
        const v = AI.view(cd, 0, N, 0, 0.7, { l: 40, r: 100, t: 24, b: 22 });
        AI.axes(ctx, v);
        T(ctx, 'Tiny 1-8-8-1 net learning y = sin(2.5x): training loss per step, same data and optimiser', 8, 14, c.fg, 'left', SM);
        Object.keys(curves).forEach((k) => {
          const col = c[COLS[names.indexOf(k)]], cu = curves[k];
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); cu.forEach((l, i) => ctx[i ? 'lineTo' : 'moveTo'](v.X(i), v.Y(Math.min(0.7, l)))); ctx.stroke();
          if (cu.length) T(ctx, `${k} ${f3(cu[cu.length - 1])}`, v.X(N) + 6, v.Y(Math.min(0.69, cu[cu.length - 1])) + 4, col, 'left', SM);
        });
        q('#cap').textContent = 'Choose another activation and press "Train tiny net" again to add its curve (illustrative: one small seed, 300 Adam steps).';
      }
    }, { fps: 40 });
    q('#sat').click();
  };


  /* ======================================================================================= */
  /* Lab 3: the training loop, phase by phase                                                */
  /* ======================================================================================= */
  AI.labs.mlnTrainLoop = function (root) {
    AI.labShell(root, 'The training loop', 'sample a batch → forward → loss → backward → update (a 2-4-3-1 tanh network on 60 points)',
      `<canvas id="cv" style="height:330px"></canvas><canvas id="lc" style="height:150px"></canvas>`,
      `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>
       <label>Learning rate η <span class="v" id="lrv"></span><input type="range" id="lr" min="0.05" max="3" step="0.05" value="0.5"></label>
       <label>Batch size B <select id="bs"><option>4</option><option selected>12</option><option>20</option><option>60</option></select></label>
       <label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="turbo"> Turbo (skip the animation, ~20 batches/s)</label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next phase ▶</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv')), lc = AI.canvas(q('#lc'));
    const D = ML.data('moons', 60, 3), X = D.X, Y = D.y, N = X.length, R = 1.6;
    const PH = ['① sample a batch', '② forward pass', '③ loss', '④ backward pass', '⑤ update'];
    let net, perm, ptr, epoch, batch, phase, pt, steps, stepLoss, marks, act, gW, bl, p0, oldW;
    const mapOff = document.createElement('canvas'); mapOff.width = mapOff.height = 26;
    const rnd = AI.rng(11);
    const shuffle = () => { const a = [...Array(N).keys()]; for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    const B = () => +q('#bs').value, LR = () => +q('#lr').value;
    const prob = (x, y) => sg(net.fwd([x, y], 0)[0]);
    const fullLoss = () => { let L = 0; for (let i = 0; i < N; i++) { const p = prob(X[i][0], X[i][1]); L -= Y[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12); } return L / N; };
    const updateMap = () => {
      const c = C(), cx = mapOff.getContext('2d'), im = cx.createImageData(26, 26), a = AI.rgb(c.blue), b = AI.rgb(c.orange);
      for (let j = 0; j < 26; j++) for (let i = 0; i < 26; i++) {
        const p = prob(-R + (2 * R * (i + 0.5)) / 26, R - (2 * R * (j + 0.5)) / 26), k = (j * 26 + i) * 4;
        for (let m = 0; m < 3; m++) im.data[k + m] = a[m] + (b[m] - a[m]) * p;
        im.data[k + 3] = 110;
      }
      cx.putImageData(im, 0, 0);
    };
    const reset = () => {
      net = mkNet([2, 4, 3, 1], 'tanh', 5, 'xavier'); perm = shuffle(); ptr = 0; epoch = 1; steps = 0; stepLoss = []; marks = [{ step: 0, loss: fullLoss() }];
      phase = -1; pt = 0; act = null; gW = null; bl = 0; oldW = null; batch = []; updateMap(); next();
    };
    function compute() {
      const n = batch.length; let L = 0;
      for (let j = 0; j < n; j++) {
        const i = batch[j], o = net.fwd(X[i], 0)[0], p = sg(o);
        L -= Y[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12);
        net.bwd([p - Y[i]]);
        if (j === 0) { act = net.A.map((a) => Float64Array.from(a)); p0 = p; }
      }
      bl = L / n; gW = net.gW.map((g) => Float64Array.from(g, (v) => v / n));
    }
    function enter(ph) {
      phase = ph; pt = 0;
      if (ph === 0) { if (ptr >= N) { epoch++; ptr = 0; perm = shuffle(); } batch = perm.slice(ptr, ptr + B()); ptr += batch.length; }
      if (ph === 1) compute();
      if (ph === 4) {
        oldW = net.snap(); net.step(LR(), batch.length, { opt: 'sgd' }); steps++; stepLoss.push(bl); updateMap();
        if (ptr >= N) marks.push({ step: steps, loss: fullLoss() });
      }
    }
    function next() { enter(phase === 4 || phase < 0 ? 0 : phase + 1); }
    q('#reset').onclick = reset; q('#bs').onchange = reset;
    const tr = transport(root, next), tick = AI.ticker();
    reset();
    const maxAbs = (arrs) => { let m = 1e-9; arrs.forEach((a) => a.forEach((v) => { m = Math.max(m, Math.abs(v)); })); return m; };
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = '×' + sp; q('#lrv').textContent = LR();
      const turbo = q('#turbo').checked;
      if (tr.playing) {
        if (turbo) { const n = tick(dt, 20); for (let i = 0; i < n; i++) { enter(0); enter(1); enter(4); } phase = 4; pt = 1; }
        else { pt += (dt * sp) / 1.5; if (pt >= 1) next(); }
      }
      const c = C(), ctx = cv.ctx, W = cv.w, H = cv.h, wl = Math.min(W * 0.4, 330);
      ctx.clearRect(0, 0, W, H);
      /* --- left: data + decision map --- */
      const v = AI.view(cv, -R, R, -R, R, { l: 8, r: W - wl + 8, t: 22, b: 8 });
      ctx.imageSmoothingEnabled = true; ctx.drawImage(mapOff, v.pad.l, v.pad.t, v.pw, v.ph);
      ctx.strokeStyle = c.border; ctx.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      const inB = new Set(batch);
      X.forEach((p, i) => {
        const px = v.X(p[0]), py = v.Y(p[1]);
        ctx.fillStyle = Y[i] ? c.orange : c.blue; ctx.beginPath(); ctx.arc(px, py, inB.has(i) ? 5 : 3.2, 0, 7); ctx.fill();
        if (inB.has(i)) { ctx.strokeStyle = c.yellow; ctx.lineWidth = phase === 0 ? 2 + Math.sin(pt * 14) : 1.5; ctx.beginPath(); ctx.arc(px, py, 8, 0, 7); ctx.stroke(); }
      });
      T(ctx, 'data · decision map', 8, 14, c.muted, 'left', '10.5px ui-monospace,monospace');
      /* --- right: network --- */
      const sizes = net.sizes, x0 = wl + 36, x1 = W - 40, colX = (l) => x0 + ((x1 - x0) * l) / (sizes.length - 1);
      const nodeY = (l, i) => 34 + ((H - 70) * (i + 0.5)) / sizes[l];
      const wShow = net.W.map((w, l) => (phase === 4 && oldW ? Float64Array.from(w, (nv, i) => AI.lerp(oldW.W[l][i], nv, AI.smooth(pt))) : w));
      const mw = maxAbs(wShow), mg = gW ? maxAbs(gW) : 1;
      const Lc = net.L, fwdP = phase === 1 ? pt * (Lc + 0.4) : phase > 1 ? 99 : -1, bwdP = phase === 3 ? pt * (Lc + 0.4) : phase > 3 ? 99 : -1;
      for (let l = 0; l < Lc; l++) {
        const ni = sizes[l], no = sizes[l + 1];
        const fwdOn = fwdP > l, bwdOn = bwdP > Lc - 1 - l;
        for (let j = 0; j < no; j++) for (let i = 0; i < ni; i++) {
          const w = wShow[l][j * ni + i], g = gW ? gW[l][j * ni + i] : 0;
          let col, lw;
          if (phase >= 3 && bwdOn && gW) { col = g > 0 ? c.red : c.blue; lw = 1 + 6 * Math.abs(g) / mg; ctx.globalAlpha = 0.9; }
          else { col = w > 0 ? c.blue : c.orange; lw = 0.6 + 3.2 * Math.abs(w) / mw; ctx.globalAlpha = fwdOn || phase !== 1 ? 0.75 : 0.25; }
          if (phase === 4) { col = c.green; ctx.globalAlpha = 0.5 + 0.4 * Math.sin(pt * 3.14); }
          ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(colX(l), nodeY(l, i)); ctx.lineTo(colX(l + 1), nodeY(l + 1, j)); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      for (let l = 0; l < sizes.length; l++) for (let i = 0; i < sizes[l]; i++) {
        const lit = phase === 1 ? fwdP >= l : phase > 1, a = act && lit ? act[l][i] : 0;
        ctx.fillStyle = lit && act ? (l === Lc ? AI.mix(c.bg3, c.orange, a) : a >= 0 ? AI.mix(c.bg3, c.blue, Math.min(1, a)) : AI.mix(c.bg3, c.orange, Math.min(1, -a))) : c.bg3;
        ctx.strokeStyle = c.fg; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(colX(l), nodeY(l, i), 11, 0, 7); ctx.fill(); ctx.stroke();
      }
      const lab = ['input', 'hidden 1 (tanh)', 'hidden 2 (tanh)', 'output p'];
      for (let l = 0; l < sizes.length; l++) T(ctx, lab[l], colX(l), 16, c.muted, 'center', '10px ui-monospace,monospace');
      T(ctx, phase === 3 ? 'edge width/colour = gradient ∂L/∂w (red +, blue −)' : phase === 4 ? 'green: weights move by −η·gradient' : 'edge width = |weight| (blue +, orange −)', x0 - 20, H - 8, c.muted, 'left', '10.5px ui-monospace,monospace');
      /* --- loss curve --- */
      const c2 = lc.ctx, mx = Math.max(30, steps + 2), v2 = AI.view(lc, 0, mx, 0, 0.9, { l: 40, r: 12, t: 20, b: 22 });
      c2.clearRect(0, 0, lc.w, lc.h); AI.axes(c2, v2);
      T(c2, 'loss vs update step: dots = batch loss, thick line = full-data loss after each epoch', 8, 13, c.muted, 'left', '10.5px ui-monospace,monospace');
      c2.fillStyle = c.muted; stepLoss.forEach((l, i) => { c2.beginPath(); c2.arc(v2.X(i + 1), v2.Y(Math.min(0.9, l)), 2, 0, 7); c2.fill(); });
      c2.strokeStyle = c.green; c2.lineWidth = 2.5; c2.beginPath(); marks.forEach((m, i) => c2[i ? 'lineTo' : 'moveTo'](v2.X(m.step), v2.Y(Math.min(0.9, m.loss)))); c2.stroke();
      const nb = Math.ceil(N / B());
      q('#out').innerHTML = `epoch ${epoch} · batch ${Math.min(nb, Math.ceil(ptr / B()))}/${nb} · update step ${steps}<br>full-data loss ${f3(marks[marks.length - 1].loss)} · batch loss ${steps ? f3(stepLoss[steps - 1]) : '–'}`;
      const ph = Math.max(0, phase), n = batch.length;
      const cap = [
        `${PH[0]}: draw ${n} of the ${N} examples (yellow rings, without replacement) — ${nb} batches make one epoch.`,
        `${PH[1]}: each example flows left→right: z = Wh + b, then tanh; the first batch example gets p = ${p0 === undefined ? '–' : f3(p0)}.`,
        `${PH[2]}: binary cross-entropy averaged over the batch = ${f3(bl)} (about 0.693 means "no idea").`,
        `${PH[3]}: gradients flow right→left, one layer at a time, via δ = Wᵀδ ⊙ tanh′. Largest |∂L/∂w| = ${f3(mg)}.`,
        `${PH[4]}: every weight moves by −η·gradient (η = ${LR()}); the decision map changes slightly.`,
      ];
      q('#cap').textContent = cap[ph] + (turbo ? ' (turbo: phases skipped)' : '');
    }, { fps: 30 });
  };

  /* ======================================================================================= */
  /* Lab 4: initialisation & normalisation                                                   */
  /* ======================================================================================= */
  AI.labs.mlnInit = function (root) {
    AI.labShell(root, 'Activation histograms per layer', 'push 128 random inputs through a deep random network and watch the signal',
      `<canvas id="cv" style="height:360px"></canvas>`,
      `<label>Initialisation <select id="init"><option value="tiny">tiny (σ = 0.01)</option><option value="large">large (σ = 0.3)</option><option value="xavier">Xavier (σ² = 1/n)</option><option value="he">He (σ² = 2/n)</option></select></label>
       <label>Activation <select id="act"><option>relu</option><option>tanh</option></select></label>
       <label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="bn"> BatchNorm before each activation</label>
       <label>Depth <span class="v" id="dv"></span><input type="range" id="depth" min="3" max="10" step="1" value="8"></label>
       <label>Width n <span class="v" id="wv"></span><input type="range" id="width" min="32" max="128" step="16" value="96"></label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="2" step="0.05" value="0.8"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="replay">Replay</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    const BINS = 30, BATCH = 128;
    let res = [], key = '', prog = 0;
    const compute = () => {
      const depth = +q('#depth').value, n = +q('#width').value, init = q('#init').value, an = q('#act').value, bn = q('#bn').checked, f = ACT[an].f;
      const r = AI.rng(7), sd = init === 'tiny' ? 0.01 : init === 'large' ? 0.3 : init === 'xavier' ? Math.sqrt(1 / n) : Math.sqrt(2 / n);
      let a = Array.from({ length: BATCH }, () => Float64Array.from({ length: n }, () => AI.randn(r)));
      res = [];
      for (let l = 0; l < depth; l++) {
        const W = Float64Array.from({ length: n * n }, () => sd * AI.randn(r)), z = a.map((x) => { const o = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += W[j * n + i] * x[i]; o[j] = s; } return o; });
        if (bn) for (let j = 0; j < n; j++) { let m = 0, v = 0; z.forEach((o) => (m += o[j] / BATCH)); z.forEach((o) => (v += (o[j] - m) ** 2 / BATCH)); const s = Math.sqrt(v + 1e-5); z.forEach((o) => (o[j] = (o[j] - m) / s)); }
        a = z.map((o) => o.map((v) => f(v)));
        const lo = an === 'tanh' ? -1 : 0, hi = an === 'tanh' ? 1 : 4, h = new Float64Array(BINS); let s1 = 0, s2 = 0, sat = 0, cnt = 0;
        a.forEach((o) => o.forEach((v) => { s1 += v; s2 += v * v; cnt++; h[AI.clamp(Math.floor(((v - lo) / (hi - lo)) * BINS), 0, BINS - 1)]++; if (an === 'tanh' ? Math.abs(v) > 0.95 : v === 0) sat++; }));
        const mean = s1 / cnt; res.push({ h: h.map((x) => x / cnt), mean, std: Math.sqrt(Math.max(0, s2 / cnt - mean * mean)), rms: Math.sqrt(s2 / cnt), sat: sat / cnt, lo, hi });
      }
      prog = 0;
    };
    ['depth', 'width'].forEach((id) => (q('#' + id).oninput = () => { q('#dv').textContent = q('#depth').value; q('#wv').textContent = q('#width').value; }));
    q('#replay').onclick = () => { prog = 0; };
    const tr = transport(root, () => { prog = Math.min(res.length + 0.5, Math.floor(prog) + 1); });
    q('#dv').textContent = q('#depth').value; q('#wv').textContent = q('#width').value;
    AI.loop(root, (dt) => {
      const k = ['init', 'act', 'depth', 'width'].map((i) => q('#' + i).value).join() + q('#bn').checked;
      if (k !== key) { key = k; compute(); }
      const sp = +q('#sp').value; q('#spv').textContent = '×' + sp;
      if (tr.playing && prog < res.length + 0.5) prog += (dt * sp) / 1.0;
      const c = C(), ctx = cv.ctx, W = cv.w, H = cv.h, L = res.length, cols = Math.min(L, W < 640 ? 3 : 5), rows = Math.ceil(L / cols), pw = W / cols, ph = H / rows;
      ctx.clearRect(0, 0, W, H);
      res.forEach((r, l) => {
        const gx = (l % cols) * pw, gy = Math.floor(l / cols) * ph, fr = AI.smooth(AI.clamp(prog - l, 0, 1));
        const bx = gx + 14, bw = pw - 28, by = gy + ph - 52, bh = ph - 82;
        T(ctx, `layer ${l + 1}`, gx + pw / 2, gy + 16, fr > 0 ? c.fg : c.muted, 'center', SM);
        ctx.strokeStyle = c.border; ctx.beginPath(); ctx.moveTo(bx, by + 0.5); ctx.lineTo(bx + bw, by + 0.5); ctx.stroke();
        if (fr <= 0) return;
        const bad = r.std < 0.05 || r.sat > 0.6 || r.rms > 6, good = r.rms > 0.4 && r.rms < 2.5;
        ctx.fillStyle = bad ? c.red : good ? c.green : c.yellow;
        const bwid = bw / BINS;
        r.h.forEach((p, i) => { const hh = Math.min(1, p / 0.35) * bh * fr; ctx.fillRect(bx + i * bwid + 0.5, by - hh, Math.max(1, bwid - 1), hh); });
        T(ctx, `${r.lo}`, bx, by + 12, c.muted, 'left', '10px ui-monospace,monospace'); T(ctx, `${r.hi}${r.hi === 4 ? '+' : ''}`, bx + bw, by + 12, c.muted, 'right', '10px ui-monospace,monospace');
        T(ctx, `std ${AI.fmt(r.std, 3)}`, gx + pw / 2, by + 26, c.fg, 'center', SM);
        T(ctx, `${q('#act').value === 'tanh' ? '|a|>0.95' : 'zeros'} ${Math.round(r.sat * 100)}%`, gx + pw / 2, by + 40, bad ? c.red : c.muted, 'center', '10.5px ui-monospace,monospace');
      });
      if (L) {
        const a = res[0], z = res[L - 1];
        q('#out').innerHTML = `layer 1 rms ${AI.fmt(a.rms, 3)} → layer ${L} rms ${AI.fmt(z.rms, 3)}<br>(rms of activations; healthy ≈ constant)`;
        const ratio = z.rms / (a.rms || 1e-12), nm = q('#init').value, bn = q('#bn').checked;
        q('#cap').textContent = bn ? 'BatchNorm re-standardises each feature over the batch at every layer, so the histograms keep the same shape even with a bad initialisation.'
          : ratio < 0.05 ? `The signal shrinks layer after layer (×${AI.fmt(Math.pow(ratio, 1 / (L - 1)), 2)} per layer): the output is nearly constant and gradients vanish. Use He (ReLU) or Xavier (tanh).`
          : ratio > 20 || z.sat > 0.6 ? 'The signal explodes (ReLU) or piles up at ±1 (tanh saturation): gradients blow up or vanish. Scale the weights by √(2/n).'
          : nm === 'he' || nm === 'xavier' ? 'Variance is roughly preserved from layer to layer: this is what He / Xavier initialisation is designed for (He for ReLU, Xavier for tanh).' : 'A middling scale: some change per layer; compare with tiny or He.';
      }
    }, { fps: 30 });
  };

  /* ======================================================================================= */
  /* Lab 5: regularisation                                                                   */
  /* ======================================================================================= */
  AI.labs.mlnRegular = function (root) {
    AI.labShell(root, 'The generalisation gap', 'a 1-24-24-1 tanh network trained on 12 noisy points; validation = 40 other noisy points (illustrative)',
      `<canvas id="fit" style="height:190px"></canvas><canvas id="loss" style="height:170px"></canvas><canvas id="int" style="height:120px"></canvas>`,
      `<label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="dr"> Dropout p = 0.3</label>
       <label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="wd"> Weight decay λ = <span id="wdv"></span> (AdamW style, exaggerated)</label><input type="range" id="wds" min="0.05" max="1" step="0.05" value="0.3">
       <label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="es"> Early stopping, patience <span id="pv"></span> epochs</label><input type="range" id="pat" min="5" max="60" step="5" value="25">
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="1" max="30" step="1" value="6"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Restart</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cf = AI.canvas(q('#fit')), cl = AI.canvas(q('#loss')), ci = AI.canvas(q('#int'));
    const r0 = AI.rng(4), tgt = (x) => Math.sin(3 * x);
    const TR = Array.from({ length: 12 }, (_, i) => { const x = -0.95 + (1.9 * i) / 11; return [x, tgt(x) + 0.3 * AI.randn(r0)]; });
    const VA = Array.from({ length: 40 }, (_, i) => { const x = -1 + (2 * i) / 39; return [x, tgt(x) + 0.3 * AI.randn(r0)]; });
    const MAXE = 400, SPE = 10, LR = 0.01;
    let net, ep, trL, vaL, best, bestEp, snap, stopped, stopEp, rr, mask1, mask2;
    const evalSet = (S) => { let s = 0; S.forEach(([x, y]) => (s += (net.fwd([x], 0)[0] - y) ** 2)); return s / S.length; };
    const reset = () => { net = mkNet([1, 24, 24, 1], 'tanh', 9, 'xavier'); ep = 0; trL = []; vaL = []; best = Infinity; bestEp = 0; snap = null; stopped = false; stopEp = 0; rr = AI.rng(21); mask1 = new Float64Array(24).fill(1); mask2 = new Float64Array(24).fill(1); };
    const epoch = () => {
      if (stopped || ep >= MAXE) return;
      const dr = q('#dr').checked ? 0.3 : 0, wd = q('#wd').checked ? +q('#wds').value : 0;
      for (let s = 0; s < SPE; s++) {
        TR.forEach(([x, y]) => { const o = net.fwd([x], dr, rr)[0]; net.bwd([2 * (o - y)]); });
        net.step(LR, TR.length, { opt: 'adam', wd });
      }
      mask1 = Float64Array.from(net.M[0]); mask2 = Float64Array.from(net.M[1]);
      ep++; trL.push(evalSet(TR)); vaL.push(evalSet(VA));
      if (vaL[ep - 1] < best) { best = vaL[ep - 1]; bestEp = ep; snap = net.snap(); }
      if (q('#es').checked && ep - bestEp >= +q('#pat').value) { stopped = true; stopEp = ep; net.restore(snap); }
    };
    ['dr', 'wd', 'es', 'wds', 'pat'].forEach((id) => (q('#' + id).onchange = reset)); q('#reset').onclick = reset;
    const tr = transport(root, epoch), tick = AI.ticker();
    reset();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' epochs/s'; q('#wdv').textContent = q('#wds').value; q('#pv').textContent = q('#pat').value;
      if (tr.playing) { const n = tick(dt, sp); for (let i = 0; i < n; i++) epoch(); }
      const c = C();
      let ctx = cf.ctx, v = AI.view(cf, -1.1, 1.1, -1.8, 1.8, { l: 34, r: 8, t: 18, b: 20 });
      ctx.clearRect(0, 0, cf.w, cf.h); AI.axes(ctx, v);
      ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.beginPath(); for (let i = 0; i <= 80; i++) { const x = -1.1 + i * 0.0275; ctx[i ? 'lineTo' : 'moveTo'](v.X(x), v.Y(tgt(x))); } ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = c.green; ctx.lineWidth = 2.5; ctx.beginPath(); for (let i = 0; i <= 80; i++) { const x = -1.1 + i * 0.0275; ctx[i ? 'lineTo' : 'moveTo'](v.X(x), v.Y(AI.clamp(net.fwd([x], 0)[0], -1.8, 1.8))); } ctx.stroke();
      VA.forEach(([x, y]) => { ctx.strokeStyle = c.orange; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(v.X(x), v.Y(y), 3, 0, 7); ctx.stroke(); });
      TR.forEach(([x, y]) => { ctx.fillStyle = c.blue; ctx.beginPath(); ctx.arc(v.X(x), v.Y(y), 5, 0, 7); ctx.fill(); });
      T(ctx, 'model green · truth dashed · train = filled · val = hollow', 8, 12, c.muted, 'left', '10.5px ui-monospace,monospace');
      ctx = cl.ctx; v = AI.view(cl, 0, MAXE, 0, 0.5, { l: 40, r: 10, t: 18, b: 20 });
      ctx.clearRect(0, 0, cl.w, cl.h); AI.axes(ctx, v);
      if (ep > 1) {
        ctx.fillStyle = AI.mix(c.bg2, c.red, 0.25); ctx.beginPath();
        for (let i = 0; i < ep; i++) ctx[i ? 'lineTo' : 'moveTo'](v.X(i + 1), v.Y(Math.min(0.5, vaL[i])));
        for (let i = ep - 1; i >= 0; i--) ctx.lineTo(v.X(i + 1), v.Y(Math.min(0.5, trL[i])));
        ctx.fill();
        [[trL, c.blue], [vaL, c.orange]].forEach(([a, col]) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); a.forEach((l, i) => ctx[i ? 'lineTo' : 'moveTo'](v.X(i + 1), v.Y(Math.min(0.5, l)))); ctx.stroke(); });
        ctx.strokeStyle = c.green; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(v.X(bestEp), v.pad.t); ctx.lineTo(v.X(bestEp), v.pad.t + v.ph); ctx.stroke(); ctx.setLineDash([]);
        T(ctx, 'best val', v.X(bestEp) + 4, v.pad.t + 12, c.green, 'left', SM);
        if (stopped) { ctx.strokeStyle = c.red; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(v.X(stopEp), v.pad.t); ctx.lineTo(v.X(stopEp), v.pad.t + v.ph); ctx.stroke(); T(ctx, 'STOP', v.X(stopEp) + 4, v.pad.t + 26, c.red, 'left', SM); }
      }
      T(ctx, 'train loss (blue) · validation loss (orange) · red area = generalisation gap', 8, 12, c.muted, 'left', '10.5px ui-monospace,monospace');
      ctx = ci.ctx; ctx.clearRect(0, 0, ci.w, ci.h);
      const dots = (m, y, nm) => { T(ctx, nm, 8, y + 4, c.muted, 'left', '10.5px ui-monospace,monospace'); for (let i = 0; i < 24; i++) { const x = 92 + i * 16; ctx.fillStyle = m[i] === 0 ? c.bg3 : c.purple; ctx.strokeStyle = m[i] === 0 ? c.red : c.purple; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill(); ctx.stroke(); } };
      dots(mask1, 18, 'hidden 1'); dots(mask2, 40, 'hidden 2');
      T(ctx, q('#dr').checked ? 'dropout masks (grey = silenced this step)' : 'no dropout: all units active', 92, 62, c.muted, 'left', '10.5px ui-monospace,monospace');
      const W2 = net.W[1], cx0 = Math.max(100, ci.w - 24 * 3 - 150);
      T(ctx, 'weights hidden1→hidden2', cx0 + 80, 56, c.muted, 'left', '10px ui-monospace,monospace');
      for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) { const w = W2[j * 24 + i]; ctx.fillStyle = w > 0 ? AI.mix(c.bg2, c.blue, Math.min(1, Math.abs(w) * 1.5)) : AI.mix(c.bg2, c.orange, Math.min(1, Math.abs(w) * 1.5)); ctx.fillRect(cx0 + i * 3, 8 + j * 3.6, 3, 3.6); }
      const gap = ep ? vaL[ep - 1] - trL[ep - 1] : 0;
      q('#out').innerHTML = `epoch ${ep} (×${SPE} steps)<br>train ${ep ? f3(trL[ep - 1]) : '–'} · val ${ep ? f3(vaL[ep - 1]) : '–'}<br>gap ${f3(gap)} · mean |w| ${f3(net.meanAbsW())}${stopped ? `<br>stopped at epoch ${stopEp}; restored best (epoch ${bestEp})` : ''}`;
      q('#cap').textContent = ep < 3 ? 'Training starts: both losses fall together.' : stopped ? `Early stopping: validation had not improved for ${q('#pat').value} epochs, so training halted and the weights of epoch ${bestEp} were restored.`
        : gap > 0.1 ? 'Validation loss rises while training loss keeps falling: the network is memorising noise (the curve wiggles through the filled points). Turn on a regulariser and restart.'
        : ep >= MAXE ? 'Finished 400 epochs; compare the gap with and without regularisation.' : 'The gap is still small: the model is fitting the signal. Keep watching.';
    }, { fps: 30 });
  };

  /* ======================================================================================= */
  /* Lab 6: optimisers on one landscape                                                      */
  /* ======================================================================================= */
  AI.labs.mlnOptim = function (root) {
    AI.labShell(root, 'SGD vs momentum vs Adam', 'three balls start at the same point on the same loss f(x, y); edit the formula',
      `<canvas id="map" style="height:300px"></canvas><canvas id="lc" style="height:150px"></canvas>`,
      `<label>Loss f(x, y) <input id="f" value="0.5*x^2+10*y^2" spellcheck="false"></label><div class="caption" id="err" style="color:var(--red)"></div>
       <label>Presets <select id="pre"><option value="0.5*x^2+10*y^2">narrow valley</option><option value="0.5*x^2+0.5*y^2">round bowl</option><option value="(1-x)^2+5*(y-x^2)^2">banana (Rosenbrock-like)</option><option value="0.5*x^2+3*y^2+0.6*sin(3*x)*sin(3*y)">ripples</option></select></label>
       <label>SGD / momentum rate <span class="v" id="l1v"></span><input type="range" id="l1" min="0.01" max="0.3" step="0.01" value="0.04"></label>
       <label>Adam rate <span class="v" id="l2v"></span><input type="range" id="l2" min="0.01" max="0.5" step="0.01" value="0.1"></label>
       <label>Schedule <select id="sch"><option value="c">constant</option><option value="w">warmup (10 steps) + cosine to step 100</option></select></label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="1" max="20" step="0.5" value="4"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cm = AI.canvas(q('#map')), cl = AI.canvas(q('#lc'));
    const VX = [-3, 3], VY = [-2, 2], START = [-2.4, 1.3], OPT = ['SGD', 'Momentum', 'Adam'], CK = ['blue', 'orange', 'green'];
    let fn = null, ast = null, fmin = 0, img = document.createElement('canvas'), runs, t = 0, imgDirty = true;
    img.width = 120; img.height = 80;
    const ex = AI.exprInput(q('#f'), q('#err'), { allowed: ['x', 'y'], check: (f) => { f({ x: 1, y: 1 }); }, onChange: (f, a) => { fn = f; ast = a; imgDirty = true; reset(); } });
    const grad = (x, y) => { const r = AI.autodiff(ast, { x, y }); return [r.varGrad.x || 0, r.varGrad.y || 0, r.value]; };
    const reset = () => {
      t = 0;
      runs = OPT.map(() => ({ p: START.slice(), d: START.slice(), m: [0, 0], v: [0, 0], trail: [START.slice()], loss: [], dead: false }));
    };
    const sched = () => (q('#sch').value === 'c' ? 1 : t < 10 ? (t + 1) / 10 : Math.max(0.02, 0.5 * (1 + Math.cos(Math.PI * Math.min(1, (t - 10) / 90)))));
    const doStep = () => {
      if (!fn) return;
      const s = sched(), l1 = +q('#l1').value * s, l2 = +q('#l2').value * s;
      runs.forEach((r, k) => {
        if (r.dead) return;
        const [gx, gy, val] = grad(r.p[0], r.p[1]); r.loss.push(val);
        if (k === 0) { r.p[0] -= l1 * gx; r.p[1] -= l1 * gy; }
        else if (k === 1) { r.m[0] = 0.9 * r.m[0] + gx; r.m[1] = 0.9 * r.m[1] + gy; r.p[0] -= l1 * r.m[0]; r.p[1] -= l1 * r.m[1]; }
        else {
          const tt = t + 1, g = [gx, gy];
          for (let i = 0; i < 2; i++) { r.m[i] = 0.9 * r.m[i] + 0.1 * g[i]; r.v[i] = 0.999 * r.v[i] + 0.001 * g[i] * g[i]; r.p[i] -= (l2 * (r.m[i] / (1 - 0.9 ** tt))) / (Math.sqrt(r.v[i] / (1 - 0.999 ** tt)) + 1e-8); }
        }
        if (!(Math.abs(r.p[0]) < 30 && Math.abs(r.p[1]) < 30)) { r.dead = true; r.p = [AI.clamp(r.p[0] || 0, -30, 30), AI.clamp(r.p[1] || 0, -30, 30)]; }
        r.trail.push(r.p.slice());
      });
      t++;
    };
    q('#reset').onclick = reset; q('#pre').onchange = () => ex.set(q('#pre').value);
    const tr = transport(root, doStep), tick = AI.ticker();
    ex.run();
    cm.onresize = () => (imgDirty = true);
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' steps/s'; q('#l1v').textContent = q('#l1').value; q('#l2v').textContent = q('#l2').value;
      if (tr.playing) { const n = tick(dt, sp); for (let i = 0; i < n; i++) doStep(); }
      if (!fn) return;
      const c = C(), ctx = cm.ctx, v = AI.view(cm, VX[0], VX[1], VY[0], VY[1], { l: 6, r: 6, t: 18, b: 6 });
      if (imgDirty) {
        imgDirty = false;
        const cx = img.getContext('2d'), im = cx.createImageData(120, 80), vals = new Float64Array(9600);
        let mn = Infinity, mxv = -Infinity;
        for (let j = 0; j < 80; j++) for (let i = 0; i < 120; i++) { const f = fn({ x: VX[0] + ((i + 0.5) / 120) * 6, y: VY[1] - ((j + 0.5) / 80) * 4 }); vals[j * 120 + i] = f; if (isFinite(f)) { mn = Math.min(mn, f); mxv = Math.max(mxv, f); } }
        fmin = mn; const a = AI.rgb(c.bg2), b = AI.rgb(c.purple), L = Math.log1p(Math.max(1e-9, mxv - mn));
        for (let k = 0; k < 9600; k++) {
          const l = Math.log1p(Math.max(0, vals[k] - mn)), u = AI.clamp(l / L, 0, 1), fr = (l * 5) % 1, line = fr < 0.1 ? 0.45 : 0;
          for (let m = 0; m < 3; m++) im.data[k * 4 + m] = a[m] + (b[m] - a[m]) * (0.12 + 0.5 * (1 - u) ** 1.5 + line);
          im.data[k * 4 + 3] = 255;
        }
        cx.putImageData(im, 0, 0);
      }
      ctx.clearRect(0, 0, cm.w, cm.h); ctx.imageSmoothingEnabled = true; ctx.drawImage(img, v.pad.l, v.pad.t, v.pw, v.ph);
      ctx.strokeStyle = c.border; ctx.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      const e = ease(dt, 8);
      runs.forEach((r, k) => {
        r.d[0] += (r.p[0] - r.d[0]) * e; r.d[1] += (r.p[1] - r.d[1]) * e;
        ctx.save(); ctx.beginPath(); ctx.rect(v.pad.l, v.pad.t, v.pw, v.ph); ctx.clip();
        ctx.strokeStyle = c[CK[k]]; ctx.lineWidth = 2; ctx.beginPath(); r.trail.forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](v.X(p[0]), v.Y(p[1]))); ctx.stroke();
        ctx.fillStyle = r.dead ? c.red : c[CK[k]]; ctx.strokeStyle = c.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(v.X(r.d[0]), v.Y(r.d[1]), 7 - k, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
      });
      ctx.fillStyle = c.fg; ctx.beginPath(); ctx.arc(v.X(START[0]), v.Y(START[1]), 3, 0, 7); ctx.fill();
      T(ctx, 'start', v.X(START[0]) + 6, v.Y(START[1]) - 6, c.muted, 'left', SM);
      OPT.forEach((n, k) => T(ctx, n, v.pad.l + 8 + k * 100, 13, c[CK[k]], 'left', '12px ui-monospace,monospace'));
      /* loss plot (log scale) */
      const c2 = cl.ctx, mx = Math.max(40, t + 2), v2 = AI.view(cl, 0, mx, -5, 2, { l: 40, r: 10, t: 16, b: 20 });
      c2.clearRect(0, 0, cl.w, cl.h); AI.axes(c2, v2);
      T(c2, 'log10 (loss − minimum) per step · dashed = learning-rate factor', 8, 11, c.muted, 'left', '10.5px ui-monospace,monospace');
      runs.forEach((r, k) => { c2.strokeStyle = c[CK[k]]; c2.lineWidth = 2; c2.beginPath(); r.loss.forEach((l, i) => c2[i ? 'lineTo' : 'moveTo'](v2.X(i), v2.Y(AI.clamp(Math.log10(Math.max(1e-6, l - fmin)), -5, 2)))); c2.stroke(); });
      if (q('#sch').value === 'w') { c2.setLineDash([4, 4]); c2.strokeStyle = c.muted; c2.beginPath(); for (let i = 0; i <= mx; i++) { const s = i < 10 ? (i + 1) / 10 : Math.max(0.02, 0.5 * (1 + Math.cos(Math.PI * Math.min(1, (i - 10) / 90)))); c2[i ? 'lineTo' : 'moveTo'](v2.X(i), v2.Y(-5 + 7 * s)); } c2.stroke(); c2.setLineDash([]); }
      q('#out').innerHTML = runs.map((r, k) => `<span style="color:${c[CK[k]]}">${OPT[k]}</span>: (${f3(r.p[0])}, ${f3(r.p[1])}) f = ${r.dead ? 'diverged' : f3(grad(r.p[0], r.p[1])[2])}`).join('<br>') + `<br>step ${t} · lr factor ${f3(sched())}`;
      const dd = runs.filter((r) => r.dead).map((r, i) => OPT[runs.indexOf(r)]);
      q('#cap').textContent = dd.length ? `${dd.join(', ')} diverged (red ball): the rate is too large for this curvature. Lower it or add a schedule.`
        : 'SGD follows the raw gradient (zig-zags where the valley is steep, crawls where it is flat). Momentum accumulates a velocity: it speeds along the valley and may overshoot. Adam divides by the running gradient size so every direction moves at ≈ the Adam rate.';
    }, { fps: 30 });
  };

  /* ======================================================================================= */
  /* Lab 7: tabular MLP — embedding lookup + parameter counter                               */
  /* ======================================================================================= */
  AI.labs.mlnTabular = function (root) {
    AI.labShell(root, 'Embeddings for a categorical column', 'a record looks up its category row, concatenates with the numeric features, and enters the MLP',
      `<canvas id="cv" style="height:430px"></canvas>`,
      `<div class="row"><button class="btn" id="p1">baseline</button><button class="btn" id="p2">wide &amp; shallow</button><button class="btn" id="p3">narrow &amp; deep</button></div>
       <label>city: categories <span class="v" id="v1v"></span><input type="range" id="v1" min="50" max="20000" step="50" value="1000"></label>
       <label>city: embedding dim <span class="v" id="d1v"></span><input type="range" id="d1" min="2" max="64" step="1" value="16"></label>
       <label>device: categories / dim <span class="v" id="v2v"></span><input type="range" id="v2" min="5" max="500" step="5" value="50"><input type="range" id="d2" min="2" max="32" step="1" value="8"></label>
       <label>plan: categories / dim <span class="v" id="v3v"></span><input type="range" id="v3" min="2" max="30" step="1" value="7"><input type="range" id="d3" min="2" max="16" step="1" value="4"></label>
       <label>numeric features <span class="v" id="nv"></span><input type="range" id="num" min="0" max="60" step="1" value="10"></label>
       <label>hidden width <span class="v" id="wv"></span><input type="range" id="wd" min="16" max="512" step="16" value="128"></label>
       <label>hidden layers <span class="v" id="hv"></span><input type="range" id="dp" min="1" max="6" step="1" value="2"></label>
       <label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="halve" checked> halve the width after each layer</label>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="2" step="0.05" value="0.7"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv'));
    const CITY = ['Cairo', 'Lima', 'Oslo', 'Pune', 'Kyiv', 'Rome', 'Lagos', 'Seoul'], DEV = ['phone', 'tablet', 'laptop'], PLAN = ['free', 'pro'];
    const rr = AI.rng(5), EMB = CITY.map(() => Array.from({ length: 6 }, () => +(rr() * 2 - 1).toFixed(2)));
    const RECS = [[6, 0, 1, [0.4, -1.2, 0.8]], [2, 2, 0, [-0.7, 0.3, 1.1]], [5, 1, 1, [1.3, 0.9, -0.4]], [0, 0, 0, [0.1, -0.5, -1.0]], [3, 2, 1, [-1.1, 1.4, 0.2]]];
    const set = (o) => Object.keys(o).forEach((k) => (q('#' + k).value = o[k]));
    q('#p1').onclick = () => { set({ v1: 1000, d1: 16, v2: 50, d2: 8, v3: 7, d3: 4, num: 10, wd: 128, dp: 2 }); q('#halve').checked = true; };
    q('#p2').onclick = () => { set({ v1: 1000, d1: 16, v2: 50, d2: 8, v3: 7, d3: 4, num: 10, wd: 512, dp: 1 }); q('#halve').checked = false; };
    q('#p3').onclick = () => { set({ v1: 1000, d1: 16, v2: 50, d2: 8, v3: 7, d3: 4, num: 10, wd: 32, dp: 4 }); q('#halve').checked = false; };
    let rec = 0, phase = 0, pt = 0;
    const doStep = () => { if (phase === 4) { phase = 0; rec = (rec + 1) % RECS.length; } else phase++; pt = 0; };
    const tr = transport(root, doStep);
    const PH = ['① a record arrives: categorical columns are just ids', '② the city id selects one row of the embedding table (no one-hot needed)', '③ the row is copied into the input vector', '④ numeric (standardised) values are appended: this is the MLP input', '⑤ the vector flows through the fully connected layers to a prediction'];
    const sizes = () => {
      const n = +q('#num').value, e = [[+q('#v1').value, +q('#d1').value], [+q('#v2').value, +q('#d2').value], [+q('#v3').value, +q('#d3').value]];
      const inp = n + e.reduce((s, x) => s + x[1], 0), D = +q('#dp').value, hs = []; let w = +q('#wd').value;
      for (let i = 0; i < D; i++) { hs.push(Math.max(4, Math.round(w))); if (q('#halve').checked) w /= 2; }
      const dims = [inp, ...hs, 1]; let mlp = 0; for (let i = 0; i < dims.length - 1; i++) mlp += dims[i] * dims[i + 1] + dims[i + 1];
      return { e, inp, dims, mlp, emb: e.reduce((s, x) => s + x[0] * x[1], 0), n };
    };
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = '×' + sp;
      [['v1v', 'v1'], ['d1v', 'd1'], ['nv', 'num'], ['wv', 'wd'], ['hv', 'dp']].forEach(([a, b]) => (q('#' + a).textContent = q('#' + b).value));
      q('#v2v').textContent = q('#v2').value + ' / ' + q('#d2').value; q('#v3v').textContent = q('#v3').value + ' / ' + q('#d3').value;
      if (tr.playing) { pt += (dt * sp) / 1.4; if (pt >= 1) doStep(); }
      const z = sizes(), c = C(), ctx = cv.ctx, W = cv.w, H = cv.h, R = RECS[rec];
      ctx.clearRect(0, 0, W, H);
      const ncol = AI.clamp(z.e[0][1], 2, 6), cw = 30, ch = 24, tx = 150, ty = 56;
      /* record card */
      ctx.fillStyle = c.bg2; ctx.strokeStyle = c.border; ctx.fillRect(8, 30, 120, 118); ctx.strokeRect(8, 30, 120, 118);
      T(ctx, 'record', 16, 46, c.muted, 'left', SM);
      T(ctx, `city = ${CITY[R[0]]}`, 16, 66, phase <= 1 ? c.yellow : c.fg); T(ctx, `device = ${DEV[R[1]]}`, 16, 84, c.fg); T(ctx, `plan = ${PLAN[R[2]]}`, 16, 102, c.fg);
      T(ctx, `num = ${R[3].join(', ')}`, 16, 120, c.fg, 'left', '10.5px ui-monospace,monospace'); T(ctx, `(city id = ${R[0]})`, 16, 138, c.muted, 'left', SM);
      /* embedding table */
      T(ctx, `city embedding table  ${z.e[0][0]} × ${z.e[0][1]}  (8 rows shown)`, tx, 24, c.fg, 'left', SM);
      CITY.forEach((nm, i) => {
        const y = ty + i * (ch + 4), on = i === R[0] && phase >= 1;
        T(ctx, `${i} ${nm}`, tx + ncol * cw + 8, y + 16, on ? c.yellow : c.muted, 'left', SM);
        for (let j = 0; j < ncol; j++) { const w = EMB[i][j]; ctx.fillStyle = w > 0 ? AI.mix(c.bg2, c.blue, Math.abs(w)) : AI.mix(c.bg2, c.orange, Math.abs(w)); ctx.fillRect(tx + j * cw, y, cw - 2, ch); T(ctx, w.toFixed(2), tx + j * cw + cw / 2 - 1, y + 16, c.fg, 'center', '10px ui-monospace,monospace'); }
        if (on) { ctx.strokeStyle = c.yellow; ctx.lineWidth = 2.5; ctx.strokeRect(tx - 2, y - 2, ncol * cw + 2, ch + 4); }
      });
      if (phase === 1) { const y = ty + R[0] * (ch + 4) + ch / 2, xx = AI.lerp(128, tx - 6, AI.smooth(pt)); AI.arrow(ctx, 100, 66, xx, y, c.yellow, 2); }
      /* input vector */
      const vy = ty + 8 * (ch + 4) + 18, ey = ty + R[0] * (ch + 4);
      T(ctx, `input vector (${z.inp} numbers = ${z.n} numeric + ${z.e.reduce((s, x) => s + x[1], 0)} embedding)`, tx, vy - 8, c.fg, 'left', SM);
      for (let j = 0; j < ncol; j++) {
        const w = EMB[R[0]][j], tgtx = tx + j * cw, tgty = vy, f = phase === 2 ? AI.smooth(pt) : phase > 2 ? 1 : 0;
        if (phase < 2) continue;
        ctx.fillStyle = w > 0 ? AI.mix(c.bg2, c.blue, Math.abs(w)) : AI.mix(c.bg2, c.orange, Math.abs(w)); ctx.fillRect(AI.lerp(tx + j * cw, tgtx, f), AI.lerp(ey, tgty, f), cw - 2, ch); T(ctx, w.toFixed(2), AI.lerp(tx + j * cw, tgtx, f) + cw / 2 - 1, AI.lerp(ey, tgty, f) + 16, c.fg, 'center', '10px ui-monospace,monospace');
      }
      if (phase >= 3) { R[3].forEach((w, j) => { const a = phase === 3 ? AI.smooth(pt) : 1; ctx.globalAlpha = a; ctx.fillStyle = w > 0 ? AI.mix(c.bg2, c.green, Math.min(1, Math.abs(w))) : AI.mix(c.bg2, c.purple, Math.min(1, Math.abs(w))); ctx.fillRect(tx + (ncol + 0.4 + j) * cw, vy, cw - 2, ch); T(ctx, w.toFixed(1), tx + (ncol + 0.4 + j) * cw + cw / 2 - 1, vy + 16, c.fg, 'center', '10px ui-monospace,monospace'); ctx.globalAlpha = 1; }); T(ctx, '… + device, plan, more numerics', tx + (ncol + 4) * cw, vy + 16, c.muted, 'left', '10px ui-monospace,monospace'); }
      /* MLP boxes (a row under the input vector) */
      const labs = z.dims.slice(1), my = vy + 52, bwid = (W - 16) / labs.length;
      T(ctx, 'MLP: fully connected layers (input size → output size)', 8, my - 8, c.fg, 'left', SM);
      labs.forEach((d, i) => {
        const x = 8 + i * bwid, pul = phase === 4 ? AI.clamp(pt * (labs.length + 1) - i, 0, 1) : 0;
        ctx.fillStyle = AI.mix(c.bg2, c.green, 0.7 * Math.sin(pul * Math.PI)); ctx.strokeStyle = c.border; ctx.fillRect(x, my, bwid - 6, 38); ctx.strokeRect(x, my, bwid - 6, 38);
        T(ctx, `fc ${z.dims[i]}→${d}`, x + 4, my + 16, c.fg, 'left', '10px ui-monospace,monospace'); T(ctx, i < labs.length - 1 ? '+ ReLU' : 'output', x + 4, my + 30, c.muted, 'left', '10px ui-monospace,monospace');
      });
      if (phase === 4) T(ctx, `prediction ≈ ${AI.fmt(0.2 + 0.5 * Math.sin(rec * 1.7 + 1), 2)} (illustrative)`, 8, H - 8, c.green, 'left', SM);
      /* counter */
      const tot = z.emb + z.mlp;
      q('#out').innerHTML = `<b>shape</b> (B,${z.dims.join(') → (B,')})<br>embeddings: ${z.emb.toLocaleString()}<br>MLP: ${z.mlp.toLocaleString()}<br><b>total: ${tot.toLocaleString()}</b> (${Math.round((100 * z.emb) / tot)}% in embeddings)`;
      q('#cap').textContent = PH[phase] + (phase === 0 ? ` (the table here is a 4-to-6-column illustration; the counter on the right uses your sliders).` : '');
    }, { fps: 30 });
  };

  /* ======================================================================================= */
  /* Lab 8: a broken training run + checklist                                                */
  /* ======================================================================================= */
  AI.labs.mlnDebug = function (root) {
    const SC = {
      flat: { name: 'Loss is flat', lr: 1e-6, opt: 'sgd', init: 'he', n: 16, noise: 0.1, steps: 400, sym: 'The loss barely moves from its starting value (SGD, lr = 1e-6). Gradient norms are tiny.', hint: 'Cause: the learning rate is far too small (an LR finder would show it). Fix: raise lr to about 0.03.' },
      nan: { name: 'Loss becomes NaN', lr: 1, opt: 'sgd', init: 'he', n: 16, noise: 0.1, steps: 400, sym: 'The loss grows, becomes inf and then NaN (SGD, lr = 1).', hint: 'Cause: the learning rate is too high, so the weights explode. Fix: lower the lr (0.02); clipping the gradient norm also stops the NaN.' },
      one: { name: 'Cannot overfit one batch', lr: 0.1, opt: 'sgd', init: 'zero', n: 8, noise: 0, steps: 400, sym: 'A batch of only 8 noise-free points cannot be memorised: the loss stalls at the variance of the labels.', hint: 'Cause: all weights start at zero, so every hidden unit is identical and gets no gradient (symmetry). Fix: random (He/Xavier) initialisation.' },
      over: { name: 'Train ↓, validation ↑', lr: 0.01, opt: 'adam', init: 'xavier', n: 12, noise: 0.3, steps: 1500, sym: 'Training loss goes to ~0 while the validation loss rises: classic overfitting (12 noisy points, a 1-24-24-1 net, Adam).', hint: 'Cause: memorisation. Fixes: regularise (dropout + weight decay), early stopping, or more data.' },
    };
    const FIX = [
      ['lr_up', 'Raise the learning rate to 0.03'], ['lr_down', 'Lower the learning rate to 0.02'], ['clip', 'Clip the gradient norm at 1'],
      ['init', 'Initialise with He (random) weights'], ['reg', 'Add dropout 0.2 + weight decay'], ['early', 'Use early stopping (patience 15 checks)'],
      ['data', 'Collect 5× more training data'], ['relu', 'Switch the activation to ReLU'], ['big', 'Make the network bigger (64 units)'],
    ];
    AI.labShell(root, 'Debug the broken run', 'pick a symptom, then tick the fix you believe in: the run restarts with that change (real training of a tiny network)',
      `<canvas id="cv" style="height:230px"></canvas><canvas id="gr" style="height:110px"></canvas>`,
      `<label>Symptom <select id="sc">${Object.keys(SC).map((k) => `<option value="${k}">${SC[k].name}</option>`).join('')}</select></label>
       <div>Checklist: tick a fix to try it${FIX.map(([k, t]) => `<label style="display:flex;flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="f_${k}"> ${t}</label>`).join('')}</div>
       <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="20" max="200" step="10" value="60"></label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Restart</button><button class="btn" id="hint">Show hint</button></div>
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => $(s, root), cv = AI.canvas(q('#cv')), cg = AI.canvas(q('#gr'));
    let cfg, net, TR, VA, step, trL, vaL, best, bestStep, snap, done, nanAt, stopAt, rr, hint = false, lastG = [0, 0, 0];
    const tgt = (x) => Math.sin(3 * x);
    const build = () => {
      const k = q('#sc').value; cfg = { ...SC[k], key: k, drop: 0, wd: 0, clip: 0, es: false, act: 'tanh', w: 24 };
      const on = (id) => q('#f_' + id).checked;
      if (on('lr_up')) cfg.lr = 0.03;
      if (on('lr_down')) cfg.lr = 0.02;
      if (on('clip')) cfg.clip = 1;
      if (on('init')) cfg.init = 'he';
      if (on('reg')) { cfg.drop = 0.2; cfg.wd = 0.3; }
      if (on('early')) cfg.es = true;
      if (on('data')) cfg.n *= 5;
      if (on('relu')) cfg.act = 'relu';
      if (on('big')) cfg.w = 64;
    };
    const mse = (S) => { let s = 0; S.forEach(([x, y]) => (s += (net.fwd([x], 0)[0] - y) ** 2)); return s / S.length; };
    const restart = () => {
      build(); const r0 = AI.rng(4);
      TR = Array.from({ length: cfg.n }, (_, i) => { const x = -0.95 + (1.9 * i) / (cfg.n - 1); return [x, tgt(x) + cfg.noise * AI.randn(r0)]; });
      VA = Array.from({ length: 40 }, (_, i) => { const x = -1 + (2 * i) / 39; return [x, tgt(x) + 0.3 * AI.randn(r0)]; });
      net = mkNet([1, cfg.w, cfg.w, 1], cfg.act, 9, cfg.init); rr = AI.rng(3);
      step = 0; trL = []; vaL = []; best = Infinity; bestStep = 0; snap = null; done = false; nanAt = 0; stopAt = 0; lastG = [0, 0, 0];
    };
    const doStep = () => {
      if (done) return;
      let L = 0;
      TR.forEach(([x, y]) => { const o = net.fwd([x], cfg.drop, rr)[0]; L += (o - y) ** 2 / TR.length; net.bwd([2 * (o - y)]); });
      net.step(cfg.lr, TR.length, { opt: cfg.opt, wd: cfg.wd, clip: cfg.clip });
      lastG = net.gn.slice(); step++; trL.push(L);
      if (!isFinite(L)) { done = true; nanAt = step; return; }
      if (step % 10 === 0) {
        const v = mse(VA); vaL.push([step, v]);
        if (v < best) { best = v; bestStep = step; snap = net.snap(); }
        if (cfg.es && step - bestStep >= 150) { done = true; stopAt = step; net.restore(snap); return; }
      }
      if (step >= cfg.steps) done = true;
    };
    q('#sc').onchange = () => { FIX.forEach(([k]) => (q('#f_' + k).checked = false)); hint = false; q('#hint').textContent = 'Show hint'; restart(); };
    FIX.forEach(([k]) => (q('#f_' + k).onchange = restart));
    q('#reset').onclick = restart; q('#hint').onclick = () => { hint = !hint; q('#hint').textContent = hint ? 'Hide hint' : 'Show hint'; };
    const tr = transport(root, doStep), tick = AI.ticker();
    restart();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; q('#spv').textContent = sp + ' steps/s';
      if (tr.playing) { const n = tick(dt, sp); for (let i = 0; i < n; i++) doStep(); }
      const c = C(), ctx = cv.ctx, v = AI.view(cv, 0, cfg.steps, -3, 3, { l: 40, r: 10, t: 18, b: 20 });
      ctx.clearRect(0, 0, cv.w, cv.h); AI.axes(ctx, v);
      const ly = (l) => v.Y(AI.clamp(Math.log10(Math.max(1e-9, l)), -3, 3));
      ctx.strokeStyle = c.blue; ctx.lineWidth = 2; ctx.beginPath(); let pen = false;
      trL.forEach((l, i) => { if (!isFinite(l)) { pen = false; return; } ctx[pen ? 'lineTo' : 'moveTo'](v.X(i + 1), ly(l)); pen = true; }); ctx.stroke();
      if (cfg.key === 'over') {
        ctx.strokeStyle = c.orange; ctx.beginPath(); vaL.forEach(([s, l], i) => ctx[i ? 'lineTo' : 'moveTo'](v.X(s), ly(l))); ctx.stroke();
        if (bestStep) { ctx.strokeStyle = c.green; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(v.X(bestStep), v.pad.t); ctx.lineTo(v.X(bestStep), v.pad.t + v.ph); ctx.stroke(); ctx.setLineDash([]); }
      }
      if (nanAt) { ctx.fillStyle = c.red; ctx.fillRect(v.X(nanAt) - 2, v.pad.t, 4, v.ph); T(ctx, 'NaN / inf', v.X(nanAt) - 8, v.pad.t + 14, c.red, 'right', SM); }
      T(ctx, 'log10 (loss) per step · blue = training loss' + (cfg.key === 'over' ? ' · orange = validation loss (every 10 steps) · green = best' : ''), 8, 12, c.muted, 'left', '10.5px ui-monospace,monospace');
      const g = cg.ctx; g.clearRect(0, 0, cg.w, cg.h);
      T(g, 'gradient norm per layer (log scale, 1e-8 … 1e2): tiny bars = vanishing', 8, 13, c.muted, 'left', '10.5px ui-monospace,monospace');
      ['layer 1 (input side)', 'layer 2', 'layer 3 (output)'].forEach((nm, i) => {
        const val = lastG[i] || 0, f = AI.clamp((Math.log10(Math.max(1e-8, val)) + 8) / 10, 0, 1), y = 24 + i * 26, W = Math.max(40, cg.w - 260);
        T(g, nm, 8, y + 14, c.muted, 'left', SM); g.fillStyle = c.bg3; g.fillRect(140, y, W, 18); g.fillStyle = f < 0.25 ? c.red : f > 0.9 ? c.yellow : c.green; g.fillRect(140, y, Math.max(1, W * f), 18);
        T(g, isFinite(val) ? val.toExponential(1) : 'NaN', 146 + W, y + 14, c.fg, 'left', SM);
      });
      const L = trL.length ? trL[trL.length - 1] : NaN;
      let verdict = 'running…', ok = false;
      if (done) {
        const tv = mse(VA), trE = mse(TR);
        if (nanAt) verdict = `✘ NaN at step ${nanAt}`;
        else if (cfg.key === 'over') { const gap = tv - trE; ok = gap < 0.07 && tv < 0.17; verdict = `${ok ? '✔ healed' : '✘ still overfitting'}: train ${f3(trE)}, val ${f3(tv)}, gap ${f3(gap)}${stopAt ? ` (stopped at step ${stopAt}, best weights from step ${bestStep})` : ''}`; }
        else if (cfg.key === 'nan') { ok = isFinite(L) && L < 0.5; verdict = ok ? `✔ healed: no NaN, loss ${f3(L)}` : `✘ finite but still bad: loss ${f3(L)}`; }
        else { ok = L < 0.1; verdict = ok ? `✔ healed: the loss fell to ${f3(L)}` : `✘ still stuck at loss ${f3(L)}`; }
      }
      q('#out').innerHTML = `${SC[cfg.key].name}<br>step ${step}/${cfg.steps} · lr ${cfg.lr}${cfg.clip ? ' · clip 1' : ''}${cfg.drop ? ' · dropout .2 · wd .3' : ''}<br>train loss ${f3(L)}<br><b style="color:${ok ? c.green : done ? c.red : c.muted}">${verdict}</b>`;
      q('#cap').textContent = hint ? SC[cfg.key].hint : `Symptom: ${SC[cfg.key].sym} Tick the fix you think addresses the cause; wrong fixes (and decoys such as ReLU or a bigger network) do not help.`;
    }, { fps: 30 });
  };
})();
