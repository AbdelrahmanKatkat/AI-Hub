/* Information-theory labs (Mathematics Module 5): surprise and entropy, cross-entropy and KL, mutual information.
   All numbers are in bits (log base 2). Animations run on dt, start slowly, and have pause / reset controls. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const L2 = Math.log2;
  const H = (p) => -p.reduce((s, x) => s + (x > 0 ? x * L2(x) : 0), 0);
  const CE = (p, r) => -p.reduce((s, x, i) => s + (x > 0 ? x * L2(r[i]) : 0), 0);
  const KL = (p, r) => p.reduce((s, x, i) => s + (x > 0 ? x * L2(x / r[i]) : 0), 0);
  const f4 = (v) => (Number.isFinite(v) ? v.toFixed(4) : '∞');
  const bar = (g, x, y, w, h, col) => { g.fillStyle = col; g.fillRect(x, y, w, h); };
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };

  /* ---- 1. Surprise and entropy ---- */
  AI.labs['info-surprise'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Surprise and entropy', 'rare events are surprising; entropy is the average surprise of a coin',
      `<canvas style="height:340px"></canvas>`,
      `${sl('p', 'Probability of heads p', 0.01, 0.99, 0.01, 0.9)}
       <div class="row"><button class="btn primary" id="play">▶ Play</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div>
       <p class="caption">Left: surprise of heads, −log₂ p, and of tails, −log₂(1−p). Right: entropy H(p) = −p log₂ p − (1−p) log₂(1−p), the average surprise. Bits.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const pIn = q(ctl, '#p'), pv = q(ctl, '#pv'), out = q(ctl, '#out');
    let p = +pIn.value, dir = 1, playing = false;
    const set = (v) => { p = Math.min(0.99, Math.max(0.01, v)); pIn.value = p.toFixed(2); };
    const draw = () => {
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const half = W / 2 - 16, top = 28, base = Hh - 36, ht = base - top;
      // left: surprise bars for heads and tails
      const sH = -L2(p), sT = -L2(1 - p), maxS = 7;
      txt(g, 'surprise (bits)', 8, 16, c.muted);
      bar(g, 40, base - (sH / maxS) * ht, half / 2 - 20, (sH / maxS) * ht, c.blue);
      bar(g, half / 2 + 20, base - (sT / maxS) * ht, half / 2 - 20, (sT / maxS) * ht, c.orange);
      txt(g, `heads −log₂p = ${sH.toFixed(3)}`, 40, base + 16, c.fg);
      txt(g, `tails −log₂(1−p) = ${sT.toFixed(3)}`, half / 2 + 20, base + 16, c.fg);
      // right: entropy curve
      const x0 = W / 2 + 12, x1 = W - 16, y = (v) => base - v * ht;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, base); g.lineTo(x1, base); g.moveTo(x0, base); g.lineTo(x0, top); g.stroke();
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 200; i++) { const t = 0.01 + (0.98 * i) / 200, px = x0 + (x1 - x0) * t; i ? g.lineTo(px, y(H([t, 1 - t]))) : g.moveTo(px, y(H([t, 1 - t]))); }
      g.stroke(); g.lineWidth = 1;
      const hp = H([p, 1 - p]), mx = x0 + (x1 - x0) * p;
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(mx, base); g.lineTo(mx, y(hp)); g.stroke(); g.setLineDash([]);
      g.fillStyle = c.purple; g.beginPath(); g.arc(mx, y(hp), 5, 0, 7); g.fill();
      txt(g, 'p', x1 - 4, base + 16, c.muted, '12px sans-serif', 'right');
      txt(g, '0.5', x0 + (x1 - x0) * 0.5, base + 16, c.muted, '12px sans-serif', 'center');
      txt(g, '1 bit', x0 - 4, top + 4, c.muted, '11px sans-serif', 'right');
      txt(g, `H(p) = ${hp.toFixed(3)} bits`, x0 + 6, top - 8, c.purple, 'bold 12px sans-serif');
      pv.textContent = p.toFixed(2);
      out.textContent = `p = ${p.toFixed(2)}: heads is ${sH.toFixed(3)} bits surprising, tails ${sT.toFixed(3)} bits; entropy H = ${hp.toFixed(4)} bits. A fair coin (p = 0.5) gives exactly 1 bit, the maximum for two outcomes.`;
    };
    pIn.oninput = () => { p = +pIn.value; draw(); };
    q(ctl, '#play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { set(0.9); draw(); };
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (!playing) return;
      set(p + dir * dt * 0.08);            // slow sweep across the whole range
      if (p >= 0.99 || p <= 0.01) dir = -dir;
      draw();
    }, { fps: 30 });
    draw();
  };

  /* ---- 2. Cross-entropy and KL divergence ---- */
  AI.labs['info-kl'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Cross-entropy and KL divergence', 'the true distribution p and a model q; the gap between them is the KL divergence',
      `<canvas style="height:300px"></canvas>`,
      `${sl('a', 'q(cat)', 0.05, 0.9, 0.01, 0.5)}
       ${sl('b', 'q(dog)', 0.05, 0.9, 0.01, 0.3)}
       <div class="readout" id="out"></div>
       <p class="caption">True distribution p = (0.7, 0.2, 0.1) over cat, dog, bird. The model q = (a, b, 1−a−b) is the one you set. Bars: p in blue, q in orange.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col(), P = [0.7, 0.2, 0.1];
    const aIn = q(ctl, '#a'), bIn = q(ctl, '#b'), out = q(ctl, '#out');
    const names = ['cat', 'dog', 'bird'];
    const draw = () => {
      let a = +aIn.value, b = +bIn.value;
      if (a + b > 0.97) { b = 0.97 - a; bIn.value = b.toFixed(2); }   // keep the third outcome positive
      const R = [a, b, 1 - a - b];
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      const base = Hh - 40, ht = Hh - 80, gw = W / 3;
      for (let i = 0; i < 3; i++) {
        const x = i * gw + gw / 2;
        bar(g, x - 46, base - P[i] * ht, 40, P[i] * ht, c.blue);
        bar(g, x + 6, base - R[i] * ht, 40, R[i] * ht, c.orange);
        txt(g, names[i], x, base + 18, c.fg, '13px sans-serif', 'center');
        txt(g, `p ${P[i].toFixed(2)}`, x - 26, base - P[i] * ht - 6, c.blue, '11px sans-serif', 'center');
        txt(g, `q ${R[i].toFixed(2)}`, x + 26, base - R[i] * ht - 6, c.orange, '11px sans-serif', 'center');
      }
      const Hp = H(P), ce = CE(P, R), kl = KL(P, R), klr = KL(R, P);
      q(ctl, '#av').textContent = a.toFixed(2); q(ctl, '#bv').textContent = b.toFixed(2);
      out.innerHTML = `H(p) = ${f4(Hp)} bits &nbsp;·&nbsp; cross-entropy H(p,q) = ${f4(ce)} bits<br>`
        + `KL(p‖q) = H(p,q) − H(p) = <b>${f4(kl)}</b> bits &nbsp;·&nbsp; reverse KL(q‖p) = ${f4(klr)} bits`;
    };
    aIn.oninput = draw; bIn.oninput = draw;
    o.onresize = draw;
    draw();
  };

  /* ---- 3. Mutual information ---- */
  AI.labs['info-mi'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Mutual information', 'how much knowing X tells you about Y, in bits',
      `<canvas style="height:300px"></canvas>`,
      `${sl('e', 'Dependence ε (0 = independent, 0.25 = Y copies X)', 0, 0.25, 0.01, 0.15)}
       <div class="row"><button class="btn primary" id="play">▶ Play</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out"></div>
       <p class="caption">Joint table P(X,Y) = [[0.25+ε, 0.25−ε], [0.25−ε, 0.25+ε]] with X and Y each a fair bit. ε slowly sweeps during Play.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const eIn = q(ctl, '#e'), ev = q(ctl, '#ev'), out = q(ctl, '#out');
    let e = +eIn.value, dir = 1, playing = false;
    const MI = (J) => {
      const px = J.map((r) => r[0] + r[1]), py = [0, 1].map((j) => J[0][j] + J[1][j]);
      let s = 0; J.forEach((r, i) => r.forEach((v, j) => { if (v > 0) s += v * L2(v / (px[i] * py[j])); })); return s;
    };
    const draw = () => {
      const J = [[0.25 + e, 0.25 - e], [0.25 - e, 0.25 + e]];
      const g = o.ctx, W = o.w, Hh = o.h; g.clearRect(0, 0, W, Hh);
      // joint heat cells
      const cell = Math.min(110, (W * 0.4) / 2), gx = 60, gy = 40;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const v = J[i][j], t = v / 0.5;
        g.fillStyle = `rgba(88,166,255,${Math.max(0.05, Math.min(1, t))})`;
        g.fillRect(gx + j * cell, gy + i * cell, cell - 4, cell - 4);
        txt(g, v.toFixed(3), gx + j * cell + cell / 2 - 2, gy + i * cell + cell / 2 + 4, c.fg, '13px monospace', 'center');
      }
      txt(g, 'X=0', gx - 8, gy + cell / 2 + 4, c.muted, '12px sans-serif', 'right');
      txt(g, 'X=1', gx - 8, gy + cell * 1.5 + 4, c.muted, '12px sans-serif', 'right');
      txt(g, 'Y=0', gx + cell / 2, gy - 10, c.muted, '12px sans-serif', 'center');
      txt(g, 'Y=1', gx + cell * 1.5, gy - 10, c.muted, '12px sans-serif', 'center');
      // entropy bars
      const HX = H([0.5, 0.5]), HY = HX, HXY = H(J.flat()), I = MI(J);
      const bx = gx + cell * 2 + 60, bw = Math.max(40, W - bx - 30), maxH = 2.2;
      const rows = [['H(X)', HX, c.orange], ['H(Y)', HY, c.orange], ['H(X,Y)', HXY, c.blue], ['I(X;Y)', I, c.purple]];
      rows.forEach(([name, v, col], k) => {
        const y = 30 + k * 46;
        txt(g, name, bx, y + 14, c.fg, '13px sans-serif');
        bar(g, bx + 70, y, (v / maxH) * (bw - 70), 22, col);
        txt(g, v.toFixed(4) + ' bits', bx + 70 + (v / maxH) * (bw - 70) + 6, y + 15, c.fg, '12px monospace');
      });
      ev.textContent = e.toFixed(2);
      out.innerHTML = `I(X;Y) = H(X) + H(Y) − H(X,Y) = ${HX.toFixed(4)} + ${HY.toFixed(4)} − ${HXY.toFixed(4)} = <b>${I.toFixed(4)}</b> bits`
        + `<br>${I < 1e-9 ? 'Independent: knowing X tells you nothing about Y.' : 'Knowing X removes ' + I.toFixed(3) + ' of the 1 bit of uncertainty in Y.'}`;
    };
    eIn.oninput = () => { e = +eIn.value; draw(); };
    q(ctl, '#play').onclick = (ev2) => { playing = !playing; ev2.target.textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#reset').onclick = () => { e = 0.15; eIn.value = e; draw(); };
    o.onresize = draw;
    AI.loop(root, (dt) => {
      if (!playing) return;
      e += dir * dt * 0.04;
      if (e >= 0.25 || e <= 0) dir = -dir;
      e = Math.min(0.25, Math.max(0, e)); eIn.value = e.toFixed(2);
      draw();
    }, { fps: 30 });
    draw();
  };
})();
