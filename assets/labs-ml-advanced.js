/* ML Module 8 labs: label propagation, transfer learning, recommender, time series, Bayesian, A/B test, Simpson's paradox.
 * All animations are dt-based, slow by default, with a speed slider, pause/step and captions. */
(function () {
  const $ = (s, r) => AI.$(s, r);
  const sld = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = (extra = '') => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button>${extra}</div>`;
  const font = (g, s = 12) => (g.font = s + 'px ui-monospace,Menlo,Consolas,monospace');
  const caption = (g, w, txt, prog) => {
    const C = AI.col(); font(g, 12);
    const bw = Math.min(w - 16, g.measureText(txt).width + 20);
    g.fillStyle = C.bg2; g.globalAlpha = 0.93; g.fillRect(8, 6, bw, 22); g.globalAlpha = 1;
    g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(txt, 18, 21);
    if (prog !== undefined) { g.fillStyle = C.border; g.fillRect(8, 29, bw, 3); g.fillStyle = C.blue; g.fillRect(8, 29, bw * AI.clamp(prog, 0, 1), 3); }
  };
  const play = (root, st, onStep) => {
    $('#play', root).onclick = () => { st.playing = !st.playing; $('#play', root).textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    $('#step', root).onclick = () => { st.playing = false; $('#play', root).textContent = '▶ Play'; onStep(); };
  };
  const spd = (root) => { const v = +$('#sp', root).value; $('#spv', root).textContent = '×' + AI.fmt(v, 2); return v; };
  const normCdf = (z) => { // Abramowitz-Stegun
    const t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989423 * Math.exp((-z * z) / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - p : p;
  };

  /* ================= 1. Label propagation ================= */
  AI.labs.mlLabelProp = function (root) {
    const { view, ctl } = AI.labShell(root, 'Label propagation lab', 'a few labelled points colour all the rest (illustrative graph on synthetic data)',
      `<canvas id="cv" style="height:380px;cursor:pointer"></canvas><div class="caption">Rings = labelled seeds. Colour = current score (blue = class +, orange = class −, grey = undecided). Click a point to add or remove a seed.</div>`,
      `<label>Dataset <select id="ds"><option value="blobs">two clusters</option><option value="moons">two moons</option></select></label>
       ${sld('k', 'Neighbours k in the graph', 2, 12, 1, 4)}
       <label>Click adds a seed of class <select id="cls"><option value="0">+ (blue)</option><option value="1">− (orange)</option></select></label>
       ${sld('sp', 'Speed', 0.25, 6, 0.25, 1)}
       ${btns()}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas(q('#cv'));
    const st = { playing: true };
    let D, nb, f, fd, seeds, round = 0, phase = 0, ph = 0, N = 70;
    const PH = 0.8;
    const build = () => {
      const kind = q('#ds').value;
      D = ML.data(kind, N, kind === 'moons' ? 4 : 6, kind === 'blobs' ? { K: 2 } : {});
      graph(); initSeeds();
    };
    const graph = () => {
      const k = +q('#k').value; nb = D.X.map(() => new Set());
      D.X.forEach((p, i) => {
        D.X.map((r, j) => [j, (r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2]).filter((a) => a[0] !== i).sort((a, b) => a[1] - b[1]).slice(0, k).forEach((a) => { nb[i].add(a[0]); nb[a[0]].add(i); });
      });
    };
    const initSeeds = () => {
      seeds = new Map();
      [0, 1].forEach((cl) => { // the point of each class closest to that class's mean-left/right extreme
        const idx = D.y.map((y, i) => (y === cl ? i : -1)).filter((i) => i >= 0);
        const mx = idx.reduce((p, i) => p + D.X[i][0], 0) / idx.length, my = idx.reduce((p, i) => p + D.X[i][1], 0) / idx.length; idx.sort((a, b) => (D.X[a][0] - mx) ** 2 + (D.X[a][1] - my) ** 2 - (D.X[b][0] - mx) ** 2 - (D.X[b][1] - my) ** 2);
        seeds.set(idx[0], cl ? -1 : 1);
      });
      resetF();
    };
    const resetF = () => { f = D.X.map((_, i) => (seeds.has(i) ? seeds.get(i) : 0)); if (!fd) fd = f.slice(); round = 0; phase = 0; ph = 0; };
    const stepRound = () => {
      const g = f.slice();
      for (let i = 0; i < N; i++) { if (seeds.has(i)) { g[i] = seeds.get(i); continue; } let s = 0, n = 0; nb[i].forEach((j) => { s += f[j]; n++; }); g[i] = n ? s / n : 0; }
      f = g; round++;
    };
    q('#reset').onclick = resetF;
    q('#ds').onchange = () => { fd = null; build(); };
    q('#k').oninput = () => { graph(); resetF(); };
    play(root, st, () => stepRound());
    c.cv.onclick = (e) => {
      const r = c.cv.getBoundingClientRect(), m = ML.toPx(c.w, c.h, 1.3), x = m.ix(e.clientX - r.left), y = m.iy(e.clientY - r.top);
      let bi = -1, bd = 0.08;
      D.X.forEach((p, i) => { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; bi = i; } });
      if (bi < 0) return;
      if (seeds.has(bi)) seeds.delete(bi); else seeds.set(bi, q('#cls').value === '0' ? 1 : -1);
      resetF();
    };
    build();
    AI.loop(root, (dt) => {
      const sp = spd(root); q('#kv').textContent = q('#k').value;
      if (st.playing) { ph += (dt * sp) / PH; if (ph >= 1) { ph = 0; phase ^= 1; if (phase === 0) stepRound(); } }
      const k = 1 - Math.exp(-dt * 4);
      for (let i = 0; i < N; i++) fd[i] += (f[i] - fd[i]) * k;
      const C = AI.col(), g = c.ctx, m = ML.toPx(c.w, c.h, 1.3);
      g.clearRect(0, 0, c.w, c.h);
      g.strokeStyle = C.border; g.lineWidth = 1;
      nb.forEach((s, i) => s.forEach((j) => { if (j > i) { g.beginPath(); g.moveTo(m.X(D.X[i][0]), m.Y(D.X[i][1])); g.lineTo(m.X(D.X[j][0]), m.Y(D.X[j][1])); g.stroke(); } }));
      const neutral = C.muted;
      let ok = 0, tot = 0, und = 0;
      D.X.forEach((p, i) => {
        const v = AI.clamp(fd[i], -1, 1), col = v >= 0 ? AI.mix(neutral, C.blue, v) : AI.mix(neutral, C.orange, -v);
        g.fillStyle = col; g.beginPath(); g.arc(m.X(p[0]), m.Y(p[1]), seeds.has(i) ? 7 : 5.5, 0, 7); g.fill();
        if (seeds.has(i)) { g.strokeStyle = C.fg; g.lineWidth = 2.5; g.stroke(); }
        if (!seeds.has(i)) { tot++; if (Math.abs(f[i]) < 0.02) und++; else if ((f[i] > 0) === (D.y[i] === 0)) ok++; }
      });
      caption(g, c.w, phase === 0 ? '① every unlabelled point takes the average of its neighbours' : '② labelled seeds are clamped back to ±1, then repeat', ph);
      q('#out').innerHTML = `<span class="m">round</span> ${round}   <span class="m">seeds</span> ${seeds.size}\n<span class="m">correct (unlabelled)</span> <span class="g">${ok}</span> / ${tot}\n<span class="m">still undecided</span> ${und}`;
    }, { fps: 30 });
  };

  /* ================= 2. Transfer learning ================= */
  AI.labs.mlTransfer = function (root) {
    const { view, ctl } = AI.labShell(root, 'Transfer-learning lab', 'freeze or train each layer; compare data needs (accuracy curves are ILLUSTRATIVE, not measured)',
      `<canvas id="net" style="height:210px;cursor:pointer"></canvas><canvas id="acc" style="height:230px"></canvas><div class="caption">Click a layer to freeze (blue) or train (orange). The parameter counts are exact for this small CNN; the accuracy curves are an illustration of the typical pattern.</div>`,
      `${sld('n', 'Target images (log scale)', 1.3, 3.7, 0.02, 2.3)}
       ${sld('sp', 'Speed', 0.25, 3, 0.25, 1)}
       <div class="row"><button class="btn" id="fe">Feature extraction</button><button class="btn" id="ft1">Fine-tune last block</button><button class="btn" id="ftall">Fine-tune all</button></div>
       ${btns()}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), cn = AI.canvas(q('#net')), ca = AI.canvas(q('#acc'));
    const L = [{ n: 'conv1 3→16', p: 448 }, { n: 'conv2 16→32', p: 4640 }, { n: 'conv3 32→64', p: 18496 }, { n: 'conv4 64→128', p: 73856 }, { n: 'new head 128→10', p: 1290 }];
    const total = L.reduce((a, l) => a + l.p, 0);
    const train = [false, false, false, false, true];
    const st = { playing: true };
    let t = 0; const T = [1.6, 0.8, 1.8, 1.2]; // forward, loss, backward, update (seconds at x1)
    let ph = 0;
    const set = (a) => { a.forEach((v, i) => (train[i] = v)); };
    q('#fe').onclick = () => set([0, 0, 0, 0, 1]); q('#ft1').onclick = () => set([0, 0, 0, 1, 1]); q('#ftall').onclick = () => set([1, 1, 1, 1, 1]);
    q('#reset').onclick = () => { t = 0; ph = 0; };
    play(root, st, () => { t = 0; ph = (ph + 1) % 4; });
    const box = (i) => { const bw = (cn.w - 30) / 5; return { x: 15 + i * bw + 4, y: 60, w: bw - 8, h: 80 }; };
    cn.cv.onclick = (e) => { const r = cn.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; L.forEach((_, i) => { const b = box(i); if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) train[i] = !train[i]; }); };
    const scratch = (n) => 0.1 + 0.78 * (1 - Math.exp(-n / 4000));
    const head = (n) => 0.1 + 0.7 * (1 - Math.exp(-n / 90));
    const full = (n) => 0.1 + 0.88 * (1 - Math.exp(-n / 300)) * (0.55 + 0.45 * (1 - Math.exp(-n / 150)));
    const CAP = ['① forward pass: the image flows through the pretrained layers', '② loss: compare the output with the label', '③ backward pass: gradients flow back — frozen layers get no update', '④ update: only the orange layers change, at their own learning rate'];
    AI.loop(root, (dt) => {
      const sp = spd(root), C = AI.col();
      if (st.playing) { t += dt * sp; if (t >= T[ph]) { t = 0; ph = (ph + 1) % 4; } }
      const trained = L.reduce((a, l, i) => a + (train[i] ? l.p : 0), 0), frac = trained / total, n = Math.pow(10, +q('#n').value);
      q('#nv').textContent = Math.round(n) + ' images';
      // network
      let g = cn.ctx; g.clearRect(0, 0, cn.w, cn.h); font(g, 11);
      L.forEach((l, i) => {
        const b = box(i), on = train[i];
        const glow = ph === 3 && on ? 0.5 + 0.5 * Math.sin((t / T[3]) * Math.PI) : 0;
        g.fillStyle = on ? C.orange : C.blue; g.globalAlpha = 0.25 + 0.5 * glow + (on ? 0.25 : 0.12); g.fillRect(b.x, b.y, b.w, b.h); g.globalAlpha = 1;
        g.strokeStyle = on ? C.orange : C.blue; g.lineWidth = 2; g.strokeRect(b.x, b.y, b.w, b.h);
        g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(l.n, b.x + b.w / 2, b.y + 18); g.fillText(l.p.toLocaleString() + ' params', b.x + b.w / 2, b.y + 36);
        g.fillStyle = on ? C.orange : C.blue; g.fillText(on ? 'TRAINED' : 'FROZEN ❄', b.x + b.w / 2, b.y + 56);
        const dist = 4 - i, lr = on ? 1e-3 * Math.pow(0.3, dist) : 0;
        g.fillStyle = C.muted; g.fillText(on ? 'lr ' + lr.toExponential(1) : 'lr 0', b.x + b.w / 2, b.y + 74);
        if (i < 4) { g.strokeStyle = C.muted; g.beginPath(); g.moveTo(b.x + b.w, b.y + 40); g.lineTo(box(i + 1).x, b.y + 40); g.stroke(); }
      });
      const bx0 = box(0).x, bx1 = box(4).x + box(4).w, yy = 150;
      let lo = -1;
      for (let i = 0; i < 5; i++) if (train[i]) { lo = i; break; }
      if (ph === 0) { const x = AI.lerp(bx0, bx1, t / T[0]); g.fillStyle = C.green; g.beginPath(); g.arc(x, 100, 7, 0, 7); g.fill(); }
      if (ph === 1) { g.fillStyle = C.red; g.textAlign = 'center'; font(g, 13); g.fillText('loss = ' + AI.fmt(1.2 - 0.9 * (frac > 0 ? 0.5 : 0), 2) + '  (example value)', cn.w / 2, yy + 10); }
      if (ph === 2) {
        const x = AI.lerp(bx1, bx0, t / T[2]); g.fillStyle = C.red;
        if (lo >= 0) { const stopX = box(lo).x; if (x >= stopX) { g.beginPath(); g.arc(x, 100, 7, 0, 7); g.fill(); } else { g.textAlign = 'center'; font(g, 11); g.fillText('gradient not needed further left (all frozen)', cn.w / 2, yy + 10); } }
        else { g.textAlign = 'center'; g.fillText('everything frozen: nothing to learn', cn.w / 2, yy + 10); }
      }
      caption(g, cn.w, CAP[ph], t / T[ph]);
      // accuracy
      g = ca.ctx; g.clearRect(0, 0, ca.w, ca.h);
      const v = AI.view(ca, 1.3, 3.7, 0, 1, { l: 44, r: 12, t: 16, b: 30 });
      g.strokeStyle = C.border; font(g, 11); g.fillStyle = C.muted;
      for (let a = 0; a <= 1; a += 0.25) { g.beginPath(); g.moveTo(v.pad.l, v.Y(a)); g.lineTo(v.pad.l + v.pw, v.Y(a)); g.stroke(); g.textAlign = 'right'; g.fillText(Math.round(a * 100) + '%', v.pad.l - 5, v.Y(a) + 4); }
      [[1.3, 20], [2, 100], [3, 1000], [3.7, 5000]].forEach(([lx, tx]) => { g.textAlign = 'center'; g.fillText(tx, v.X(lx), v.pad.t + v.ph + 16); });
      const curves = [['from scratch', scratch, C.muted, 2], ['head only (1.3%)', head, C.blue, 2], ['fine-tune all', full, C.purple, 2], ['your choice', (x) => head(x) * (1 - frac) + full(x) * frac, C.orange, 3.5]];
      curves.forEach(([nm, fn, col, lw], ci) => {
        g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
        for (let i = 0; i <= 80; i++) { const lx = 1.3 + (2.4 * i) / 80, y = fn(Math.pow(10, lx)); i ? g.lineTo(v.X(lx), v.Y(y)) : g.moveTo(v.X(lx), v.Y(y)); }
        g.stroke(); g.fillStyle = col; g.textAlign = 'left'; font(g, 11); g.fillText(nm, v.pad.l + 8 + ci * 118, v.pad.t + 10);
      });
      g.strokeStyle = C.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(+q('#n').value), v.pad.t); g.lineTo(v.X(+q('#n').value), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText('illustrative accuracy vs target images', v.pad.l + v.pw, v.pad.t + v.ph - 6);
      q('#out').innerHTML = `<span class="m">trained</span> <span class="o">${trained.toLocaleString()}</span> / ${total.toLocaleString()}  (<b>${AI.fmt(frac * 100, 1)}%</b>)\n<span class="m">frozen</span> ${(total - trained).toLocaleString()}\n<span class="m">scratch / head / full / yours at ${Math.round(n)} imgs</span>\n${Math.round(scratch(n) * 100)}% / ${Math.round(head(n) * 100)}% / ${Math.round(full(n) * 100)}% / <span class="o">${Math.round((head(n) * (1 - frac) + full(n) * frac) * 100)}%</span>  (illustrative)`;
    }, { fps: 30 });
  };

  /* ================= 3. Recommender ================= */
  AI.labs.mlRecSys = function (root) {
    const { view, ctl } = AI.labShell(root, 'Recommender lab: users and items in one latent space', 'drag the points · train with SGD · add a cold-start user',
      `<canvas id="cv" style="height:340px;cursor:grab"></canvas><div id="mat" style="overflow-x:auto;margin-top:8px"></div><div class="caption">Predicted rating = dot product of a user's vector and an item's vector. Circles = users, squares = items. Click cells of the new user's row (U5) to give ratings.</div>`,
      `${sld('sp', 'Speed', 0.25, 6, 0.25, 1)}
       ${sld('lr', 'Learning rate α', 0.01, 0.15, 0.01, 0.05)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="nu"> Show new user U5 (cold start)</label>
       ${btns('<button class="btn" id="shuf">Randomise vectors</button>')}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas(q('#cv'));
    const R = [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 1, 5, 4], [0, 0, 0, 0]]; // 0 = unobserved
    const P0 = [[0.5, 0.2], [0.4, 0.1], [0.3, 0.6], [0.1, 0.5]], Q0 = [[0.6, 0.1], [0.2, 0.3], [0.1, 0.8], [0.4, 0.5]];
    let P, Q, Pd, Qd, nuOn = false, order = [], oi = 0, cur = null;
    const st = { playing: false }, LAM = 0.02;
    const clone = (a) => a.map((r) => r.slice());
    const reset = () => { P = clone(P0); Q = clone(Q0); const m = [0, 1].map((k) => P0.reduce((a, p) => a + p[k], 0) / 4); P.push(m.slice()); Pd = clone(P); Qd = clone(Q); cur = null; phase = 0; ph = 0; };
    let phase = 0, ph = 0;
    const nU = () => (nuOn ? 5 : 4);
    const obs = () => { const o = []; for (let u = 0; u < nU(); u++) for (let i = 0; i < 4; i++) if (R[u][i]) o.push([u, i]); return o; };
    const pred = (u, i) => P[u][0] * Q[i][0] + P[u][1] * Q[i][1];
    const rmse = () => { const o = obs(); return o.length ? Math.sqrt(o.reduce((a, [u, i]) => a + (R[u][i] - pred(u, i)) ** 2, 0) / o.length) : 0; };
    const nextPair = () => { if (oi >= order.length) { order = obs().sort(() => Math.random() - 0.5); oi = 0; } return order[oi++]; };
    const apply = (u, i) => {
      const a = +q('#lr').value, e = R[u][i] - pred(u, i), pu = P[u].slice(), qi = Q[i].slice();
      for (let k = 0; k < 2; k++) { P[u][k] += a * (e * qi[k] - LAM * pu[k]); Q[i][k] += a * (e * pu[k] - LAM * qi[k]); }
    };
    reset();
    q('#reset').onclick = () => { reset(); };
    q('#shuf').onclick = () => { const r = AI.rng(Math.floor(Math.random() * 1e6)); P = P.map(() => [r() * 1.5, r() * 1.5]); Q = Q.map(() => [r() * 1.5, r() * 1.5]); };
    q('#nu').onchange = () => { nuOn = q('#nu').checked; };
    play(root, st, () => { const [u, i] = nextPair(); apply(u, i); });
    st.playing = true;
    const v0 = () => AI.view(c, -0.3, 3, -0.3, 3, { l: 36, r: 10, t: 10, b: 24 });
    let drag = null;
    const ents = () => { const v = v0(), e = []; for (let u = 0; u < nU(); u++) e.push({ t: 'u', k: u, x: v.X(Pd[u][0]), y: v.Y(Pd[u][1]) }); for (let i = 0; i < 4; i++) e.push({ t: 'i', k: i, x: v.X(Qd[i][0]), y: v.Y(Qd[i][1]) }); return e; };
    c.cv.onpointerdown = (ev) => { const r = c.cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top; let b = null, bd = 18 * 18; ents().forEach((e) => { const d = (e.x - x) ** 2 + (e.y - y) ** 2; if (d < bd) { bd = d; b = e; } }); if (b) { drag = b; c.cv.setPointerCapture(ev.pointerId); } };
    c.cv.onpointermove = (ev) => { if (!drag) return; const r = c.cv.getBoundingClientRect(), v = v0(), x = AI.clamp(v.ix(ev.clientX - r.left), -0.3, 3), y = AI.clamp(v.iy(ev.clientY - r.top), -0.3, 3); const A = drag.t === 'u' ? P : Q, D = drag.t === 'u' ? Pd : Qd; A[drag.k] = [x, y]; D[drag.k] = [x, y]; };
    c.cv.onpointerup = () => { drag = null; };
    let lastKey = '';
    root.addEventListener('click', (ev) => { const td = ev.target.closest('td[data-r]'); if (!td) return; const i = +td.dataset.i; R[4][i] = R[4][i] === 0 ? 5 : R[4][i] === 5 ? 1 : 0; order = []; lastKey = ''; });
    const COLS = ['blue', 'orange', 'green', 'pink', 'purple'];
    AI.loop(root, (dt) => {
      const sp = spd(root); q('#lrv').textContent = q('#lr').value;
      if (st.playing && !drag) {
        ph += dt * sp / 0.7;
        if (phase === 0 && !cur) { cur = nextPair(); }
        if (ph >= 1) { ph = 0; if (phase === 0) { apply(cur[0], cur[1]); phase = 1; } else { phase = 0; cur = null; } }
      }
      const k = 1 - Math.exp(-dt * 5);
      P.forEach((p, u) => { if (!drag || drag.t !== 'u' || drag.k !== u) { Pd[u][0] += (p[0] - Pd[u][0]) * k; Pd[u][1] += (p[1] - Pd[u][1]) * k; } });
      Q.forEach((p, i) => { if (!drag || drag.t !== 'i' || drag.k !== i) { Qd[i][0] += (p[0] - Qd[i][0]) * k; Qd[i][1] += (p[1] - Qd[i][1]) * k; } });
      const C = AI.col(), g = c.ctx, v = v0();
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      if (cur) { g.strokeStyle = C.red; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(v.X(Pd[cur[0]][0]), v.Y(Pd[cur[0]][1])); g.lineTo(v.X(Qd[cur[1]][0]), v.Y(Qd[cur[1]][1])); g.stroke(); g.setLineDash([]); }
      font(g, 12);
      ents().forEach((e) => {
        const col = C[COLS[e.t === 'u' ? e.k % 5 : e.k]] || C.fg; g.fillStyle = col; g.textAlign = 'center';
        if (e.t === 'u') { g.beginPath(); g.arc(e.x, e.y, 9, 0, 7); g.fill(); g.fillStyle = C.bg; g.fillText('U' + (e.k + 1), e.x, e.y + 4); }
        else { g.fillRect(e.x - 9, e.y - 9, 18, 18); g.fillStyle = C.bg; g.fillText('I' + (e.k + 1), e.x, e.y + 4); }
      });
      let cap = !st.playing ? 'Paused — drag points or press Play to train' : '';
      if (st.playing && cur) { const [u, i] = cur, e = R[u][i] - pred(u, i); cap = phase === 0 ? `① pick rating (U${u + 1}, I${i + 1}) = ${R[u][i]}; predicted ${AI.fmt(pred(u, i), 2)}, error e = ${AI.fmt(e, 2)}` : `② nudge both vectors along the error: p ← p + α(e·q − λp), q ← q + α(e·p − λq)`; }
      if (cap) caption(g, c.w, cap, st.playing ? ph : undefined);
      // matrix (HTML) throttled by key
      const key = P.map((p) => p.map((x) => x.toFixed(2))).join() + Q.map((p) => p.map((x) => x.toFixed(2))).join() + nuOn + R[4].join() + (cur ? cur.join() : '');
      if (key !== lastKey) {
        lastKey = key;
        let h = '<table class="tbl"><tr><th></th>' + [1, 2, 3, 4].map((i) => `<th>I${i}</th>`).join('') + '</tr>';
        for (let u = 0; u < nU(); u++) { h += `<tr><td>U${u + 1}</td>`; for (let i = 0; i < 4; i++) { const p = pred(u, i), hi = cur && cur[0] === u && cur[1] === i; h += R[u][i] ? `<td ${u === 4 ? `data-r data-i="${i}" style="cursor:pointer;` : 'style="'}${hi ? 'background:var(--bg3);' : ''}">${R[u][i]} <span style="color:var(--muted)">(${AI.fmt(p, 1)})</span></td>` : `<td ${u === 4 ? `data-r data-i="${i}" style="cursor:pointer;color:var(--orange)"` : 'style="color:var(--orange)"'}>? ${AI.fmt(p, 1)}</td>`; } h += '</tr>'; }
        q('#mat').innerHTML = h + '</table>';
        const t = [...Array(4).keys()].filter((i) => !R[0][i]).map((i) => `I${i + 1}`);
        const rk = [0, 1, 2, 3].filter((i) => !R[nU() - 1][i]).sort((a, b) => pred(nU() - 1, b) - pred(nU() - 1, a)).map((i) => `I${i + 1} (${AI.fmt(pred(nU() - 1, i), 1)})`).join(' > ');
        q('#out').innerHTML = `<span class="m">observed</span> ${obs().length}   <span class="m">RMSE</span> <span class="o">${AI.fmt(rmse(), 3)}</span>\n<span class="m">ranking for U${nU()} (unrated items)</span>\n${rk || '—'}\n<span class="m">black number = true rating (prediction); orange = unseen</span>`;
      }
    }, { fps: 30 });
  };

  /* ================= 4. Time series ================= */
  AI.labs.mlTimeSeries = function (root) {
    const { view, ctl } = AI.labShell(root, 'Time-series lab', 'trend + seasonality + noise · lag window · forecast fan · time-ordered validation (synthetic data)',
      `<canvas id="a" style="height:270px"></canvas><canvas id="b" style="height:190px"></canvas><div class="caption">Top: data (blue), training region (left of the dashed line), forecast (orange) with 95% band, and the unseen future (green). Bottom: decomposition fitted on the training part only.</div>`,
      `${sld('slope', 'Trend slope', -0.5, 1, 0.05, 0.4)}
       ${sld('amp', 'Seasonal amplitude (period 12)', 0, 10, 0.5, 5)}
       ${sld('noise', 'Noise level', 0, 4, 0.1, 1.2)}
       ${sld('split', 'Validation split at t =', 24, 52, 1, 40)}
       <label>Forecast mode <select id="mode"><option value="multi">multi-step (from the split)</option><option value="one">one-step (real lags each time)</option></select></label>
       ${sld('sp', 'Speed', 0.25, 3, 0.25, 1)}
       ${btns('<button class="btn" id="new">New noise</button>')}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), ca = AI.canvas(q('#a')), cb = AI.canvas(q('#b'));
    const N = 60, P = 12, st = { playing: true };
    let seed = 5, eps, y = [], key = '', M = null, w = 4, wph = 0;
    const gen = () => { const r = AI.rng(seed); eps = []; let e = 0; for (let i = 0; i < N; i++) { e = 0.6 * e + 0.8 * AI.randn(r); eps.push(e); } key = ''; };
    gen();
    q('#new').onclick = () => { seed += 7; gen(); };
    q('#reset').onclick = () => { w = 4; wph = 0; };
    const stepW = () => { w++; const s = +q('#split').value; if (w >= s) w = 4; };
    play(root, st, stepW);
    const fit = () => {
      const sl = +q('#slope').value, am = +q('#amp').value, no = +q('#noise').value, s = +q('#split').value;
      y = eps.map((e, t) => 20 + sl * t + am * Math.sin((2 * Math.PI * t) / P) + no * e);
      // linear trend on train
      let sx = 0, sy = 0, sxx = 0, sxy = 0; for (let t = 0; t < s; t++) { sx += t; sy += y[t]; sxx += t * t; sxy += t * y[t]; }
      const n = s, b = (n * sxy - sx * sy) / (n * sxx - sx * sx), a = (sy - b * sx) / n, tbar = sx / n, Sxx = sxx - n * tbar * tbar;
      const tr = (t) => a + b * t;
      const prof = new Array(P).fill(0), cnt = new Array(P).fill(0);
      for (let t = 0; t < s; t++) { prof[t % P] += y[t] - tr(t); cnt[t % P]++; }
      for (let k = 0; k < P; k++) prof[k] /= cnt[k] || 1;
      const m = prof.reduce((p, v) => p + v, 0) / P; for (let k = 0; k < P; k++) prof[k] -= m;
      const res = []; for (let t = 0; t < N; t++) res.push(y[t] - tr(t) - prof[t % P]);
      let n1 = 0, d1 = 0, v = 0; for (let t = 0; t < s; t++) { v += res[t] ** 2; if (t) { n1 += res[t] * res[t - 1]; d1 += res[t - 1] ** 2; } }
      const phi = AI.clamp(d1 ? n1 / d1 : 0, -0.95, 0.95), sd = Math.sqrt(v / s), se = sd * Math.sqrt(1 - phi * phi);
      // forecasts
      const fm = [], fo = [];
      for (let t = s; t < N; t++) {
        const h = t - s + 1, base = tr(t) + prof[t % P];
        let vv = 0; for (let j = 0; j < h; j++) vv += phi ** (2 * j);
        const pv = sd * sd * (1 / n + (t - tbar) ** 2 / Sxx);
        fm.push({ t, f: base + phi ** h * res[s - 1], sd: Math.sqrt(se * se * vv + pv) });
        fo.push({ t, f: base + phi * res[t - 1], sd: Math.sqrt(se * se + pv) });
      }
      const mae = (F) => F.reduce((p, o) => p + Math.abs(o.f - y[o.t]), 0) / F.length;
      M = { a, b, tr, prof, res, phi, sd, se, fm, fo, s, maeM: mae(fm), maeO: mae(fo), naive: y.slice(s).reduce((p, v2, i) => p + Math.abs(v2 - y[s - 1]), 0) / (N - s), snaive: y.slice(s).reduce((p, v2, i) => p + Math.abs(v2 - y[s + i - P]), 0) / (N - s) };
    };
    AI.loop(root, (dt) => {
      const sp = spd(root);
      ['slope', 'amp', 'noise', 'split'].forEach((k) => (q('#' + k + 'v').textContent = q('#' + k).value));
      const k2 = [q('#slope').value, q('#amp').value, q('#noise').value, q('#split').value, seed].join();
      if (k2 !== key) { key = k2; fit(); if (w >= M.s) w = 4; }
      if (st.playing) { wph += (dt * sp) / 0.9; if (wph >= 1) { wph = 0; stepW(); } }
      const C = AI.col(), mode = q('#mode').value, F = mode === 'one' ? M.fo : M.fm, s = M.s;
      let g = ca.ctx; g.clearRect(0, 0, ca.w, ca.h);
      const lo = Math.min(...y, ...F.map((o) => o.f - 2 * o.sd)) - 2, hi = Math.max(...y, ...F.map((o) => o.f + 2 * o.sd)) + 2;
      const v = AI.view(ca, 0, N - 1, lo, hi, { l: 40, r: 10, t: 14, b: 22 }); AI.axes(g, v);
      g.fillStyle = C.blue; g.globalAlpha = 0.07; g.fillRect(v.X(0), v.pad.t, v.X(s - 1) - v.X(0), v.ph); g.globalAlpha = 1;
      // band
      g.fillStyle = C.orange; g.globalAlpha = 0.2; g.beginPath(); F.forEach((o, i) => (i ? g.lineTo(v.X(o.t), v.Y(o.f + 1.96 * o.sd)) : g.moveTo(v.X(o.t), v.Y(o.f + 1.96 * o.sd)))); for (let i = F.length - 1; i >= 0; i--) g.lineTo(v.X(F[i].t), v.Y(F[i].f - 1.96 * F[i].sd)); g.closePath(); g.fill(); g.globalAlpha = 1;
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath(); F.forEach((o, i) => (i ? g.lineTo(v.X(o.t), v.Y(o.f)) : g.moveTo(v.X(o.t), v.Y(o.f)))); g.stroke();
      g.lineWidth = 1.5; g.strokeStyle = C.blue; g.beginPath(); for (let t = 0; t < s; t++) t ? g.lineTo(v.X(t), v.Y(y[t])) : g.moveTo(v.X(t), v.Y(y[t])); g.stroke();
      g.strokeStyle = C.green; g.beginPath(); for (let t = s - 1; t < N; t++) t > s - 1 ? g.lineTo(v.X(t), v.Y(y[t])) : g.moveTo(v.X(t), v.Y(y[t])); g.stroke();
      g.setLineDash([5, 4]); g.strokeStyle = C.fg; g.beginPath(); g.moveTo(v.X(s - 0.5), v.pad.t); g.lineTo(v.X(s - 0.5), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      // lag window
      const x0 = v.X(w - 4), x1 = v.X(w - 1), xt = v.X(w);
      g.fillStyle = C.purple; g.globalAlpha = 0.22; g.fillRect(x0 - 4, v.pad.t, x1 - x0 + 8, v.ph); g.globalAlpha = 1;
      g.strokeStyle = C.purple; g.lineWidth = 2; g.strokeRect(x0 - 4, v.pad.t, x1 - x0 + 8, v.ph);
      g.fillStyle = C.red; g.beginPath(); g.arc(xt, v.Y(y[w]), 6, 0, 7); g.fill();
      font(g, 11); g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('training', v.pad.l + 4, v.pad.t + 24); g.fillText('validation →', v.X(s) + 4, v.pad.t + 24);
      caption(g, ca.w, `lag window: row t=${w}: target ${AI.fmt(y[w], 1)} ← lags ${[1, 2, 3, 4].map((l) => AI.fmt(y[w - l], 1)).join(', ')}`, wph);
      // decomposition
      g = cb.ctx; g.clearRect(0, 0, cb.w, cb.h);
      const panels = [['trend', (t) => M.tr(t), C.green, Math.min(...Array.from({ length: N }, (_, t) => M.tr(t))), Math.max(...Array.from({ length: N }, (_, t) => M.tr(t)))], ['seasonal', (t) => M.prof[t % P], C.purple, -12, 12], ['residual', (t) => M.res[t], C.red, -8, 8]];
      const ph2 = cb.h / 3;
      panels.forEach(([nm, fn, col, mn, mx], i) => {
        const vv = AI.view(cb, 0, N - 1, mn - 0.5, mx + 0.5, { l: 40, r: 10, t: i * ph2 + 6, b: cb.h - (i + 1) * ph2 + 6 });
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(vv.pad.l, vv.pad.t + vv.ph); g.lineTo(vv.pad.l + vv.pw, vv.pad.t + vv.ph); g.stroke();
        g.strokeStyle = col; g.lineWidth = 1.8; g.beginPath(); for (let t = 0; t < s; t++) t ? g.lineTo(vv.X(t), vv.Y(fn(t))) : g.moveTo(vv.X(t), vv.Y(fn(t))); g.stroke();
        g.globalAlpha = 0.4; g.beginPath(); for (let t = s - 1; t < N; t++) t > s - 1 ? g.lineTo(vv.X(t), vv.Y(fn(t))) : g.moveTo(vv.X(t), vv.Y(fn(t))); g.stroke(); g.globalAlpha = 1;
        font(g, 11); g.fillStyle = col; g.textAlign = 'left'; g.fillText(nm, vv.pad.l + 4, vv.pad.t + 10);
      });
      q('#out').innerHTML = `<span class="m">trend</span> ${AI.fmt(M.b, 2)} / step   <span class="m">residual AR φ</span> ${AI.fmt(M.phi, 2)}\n<span class="m">validation MAE</span>\n multi-step ${AI.fmt(M.maeM, 2)}   one-step ${AI.fmt(M.maeO, 2)}\n naive ${AI.fmt(M.naive, 2)}   seasonal-naive ${AI.fmt(M.snaive, 2)}`;
    }, { fps: 30 });
  };

  /* ================= 5. Bayesian ================= */
  AI.labs.mlBayes = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bayesian lab: prior × likelihood = posterior', 'data arrives one point at a time · predictive band widens away from data',
      `<canvas id="a" style="height:210px"></canvas><canvas id="b" style="height:280px"></canvas><div class="caption">Top: belief about the slope w. Bottom: the data and the model's prediction with its 95% band (lesson numbers: τ=1, σ=1, data (1,1.2) (2,1.9) (3,3.2)).</div>`,
      `<label>Data <select id="ds"><option value="lesson">lesson data (3 points)</option><option value="rand">random from true slope (8 points)</option></select></label>
       ${sld('tau', 'Prior sd τ (belief that w is near 0)', 0.2, 3, 0.1, 1)}
       ${sld('sig', 'Noise sd σ assumed by the model', 0.3, 3, 0.1, 1)}
       <label>Model <select id="mode"><option value="lin">Bayesian linear regression (y = w·x)</option><option value="gp">Gaussian process (RBF kernel)</option></select></label>
       ${sld('ell', 'GP length-scale (smoothness)', 0.5, 5, 0.25, 2)}
       ${sld('sp', 'Speed', 0.25, 3, 0.25, 1)}
       ${btns('<button class="btn" id="new">New data</button>')}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), ca = AI.canvas(q('#a')), cb = AI.canvas(q('#b'));
    const st = { playing: true }; let seed = 3, X = [], Y = [], k = 0, phase = 0, ph = 0;
    const gen = () => {
      if (q('#ds').value === 'lesson') { X = [1, 2, 3]; Y = [1.2, 1.9, 3.2]; }
      else { const r = AI.rng(seed); X = [1.5, 3, 5, 6.5, 8, 2, 9.5, 4]; Y = X.map((x) => 1.1 * x + 1.2 * AI.randn(r)); }
      k = 0; phase = 0; ph = 0;
    };
    gen();
    q('#ds').onchange = gen; q('#new').onclick = () => { seed += 5; q('#ds').value = 'rand'; gen(); }; q('#reset').onclick = () => { k = 0; phase = 0; ph = 0; };
    play(root, st, () => { if (k < X.length) k++; phase = 0; ph = 0; });
    const post = (n, tau, sig) => { let p = 1 / (tau * tau), b = 0; for (let i = 0; i < n; i++) { p += (X[i] * X[i]) / (sig * sig); b += (X[i] * Y[i]) / (sig * sig); } return { m: b / p, v: 1 / p }; };
    const pdf = (w, m, v) => Math.exp(-((w - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
    const solve = (A, b) => { const n = b.length, M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; for (let r = i + 1; r < n; r++) { const f = M[r][i] / M[i][i]; for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c]; } } const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c]; x[i] = s / M[i][i]; } return x; };
    const gp = (n, sig, ell, xs) => {
      const kf = (a, b) => 4 * Math.exp(-((a - b) ** 2) / (2 * ell * ell));
      if (!n) return xs.map(() => ({ m: 0, sd: Math.sqrt(4 + sig * sig) }));
      const K = []; for (let i = 0; i < n; i++) { K.push([]); for (let j = 0; j < n; j++) K[i].push(kf(X[i], X[j]) + (i === j ? sig * sig : 0)); }
      const al = solve(K, Y.slice(0, n));
      return xs.map((x) => { const ks = X.slice(0, n).map((xi) => kf(x, xi)), m = ks.reduce((a, v, i) => a + v * al[i], 0), z = solve(K, ks), v = kf(x, x) - ks.reduce((a, v2, i) => a + v2 * z[i], 0); return { m, sd: Math.sqrt(Math.max(v, 0) + sig * sig) }; });
    };
    let cur = null;
    AI.loop(root, (dt) => {
      const sp = spd(root), tau = +q('#tau').value, sig = +q('#sig').value, ell = +q('#ell').value;
      ['tau', 'sig', 'ell'].forEach((id) => (q('#' + id + 'v').textContent = q('#' + id).value));
      if (st.playing && k < X.length) { ph += (dt * sp) / 1.4; if (ph >= 1) { ph = 0; phase++; if (phase > 2) { phase = 0; k++; } } }
      const C = AI.col(), n = k, done = k >= X.length, g = ca.ctx;
      const pr = post(n, tau, sig), po = done ? pr : post(n + 1, tau, sig);
      // top panel
      g.clearRect(0, 0, ca.w, ca.h);
      const v = AI.view(ca, -1, 3, 0, 1, { l: 14, r: 10, t: 26, b: 22 }), W = [];
      for (let i = 0; i <= 160; i++) W.push(-1 + (4 * i) / 160);
      const lik = done ? null : { m: Y[n] / X[n], v: (sig / X[n]) ** 2 };
      const mx = Math.max(pdf(pr.m, pr.m, pr.v), done ? 0 : pdf(po.m, po.m, po.v), lik ? pdf(lik.m, lik.m, lik.v) : 0) * 1.1;
      v.ymax = mx; const Y2 = (p) => v.pad.t + (1 - p / mx) * v.ph;
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(v.pad.l, v.pad.t + v.ph); g.lineTo(v.pad.l + v.pw, v.pad.t + v.ph); g.stroke();
      font(g, 11); g.fillStyle = C.muted; g.textAlign = 'center'; for (let w = -1; w <= 3; w++) g.fillText('w=' + w, v.X(w), v.pad.t + v.ph + 14);
      const curve = (m, vr, col, a, fill) => { g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath(); W.forEach((w, i) => (i ? g.lineTo(v.X(w), Y2(pdf(w, m, vr))) : g.moveTo(v.X(w), Y2(pdf(w, m, vr))))); g.stroke(); if (fill) { g.fillStyle = col; g.globalAlpha = a * 0.18; g.lineTo(v.X(3), Y2(0)); g.lineTo(v.X(-1), Y2(0)); g.fill(); } g.globalAlpha = 1; };
      const ease = AI.smooth(phase === 1 ? ph : phase === 2 ? 1 : 0), ease2 = AI.smooth(phase === 2 ? ph : 0);
      curve(pr.m, pr.v, C.blue, 1, true);
      if (!done && phase >= 1) curve(lik.m, lik.v, C.green, ease * (phase === 2 ? 0.6 : 1));
      if (done) { /* final posterior = prior curve */ } else if (phase === 2) curve(po.m, po.v, C.orange, ease2, true);
      font(g, 11); g.textAlign = 'left';
      g.fillStyle = C.blue; g.fillText('prior' + (n ? ' (= previous posterior)' : ''), v.pad.l + 4, 44);
      g.fillStyle = C.green; g.fillText('likelihood of new point', v.pad.l + 200, 44); g.fillStyle = C.orange; g.fillText('posterior', v.pad.l + 380, 44);
      const lin = q('#mode').value === 'lin';
      const capTxt = done ? 'all points absorbed — the posterior is narrow; now compare the predictive band below' : ['① start from the prior: what we believed before the next point', '② the new point (x=' + X[n] + ', y=' + Y[n] + ') says which w explain it (likelihood)', '③ multiply and renormalise → posterior; it becomes the next prior'][phase];
      caption(g, ca.w, capTxt, done ? undefined : (phase + ph) / 3);
      // bottom panel: data + prediction
      const h = cb.ctx; h.clearRect(0, 0, cb.w, cb.h);
      const xs = []; for (let i = 0; i <= 60; i++) xs.push((12 * i) / 60);
      const nn = lin ? n : n, P2 = lin ? xs.map((x) => ({ m: pr.m * x, sd: Math.sqrt(x * x * pr.v + sig * sig), sp: x * Math.sqrt(pr.v) })) : gp(nn, sig, ell, xs);
      const vb = AI.view(cb, 0, 12, -6, 16, { l: 36, r: 10, t: 14, b: 22 }); AI.axes(h, vb);
      h.fillStyle = C.orange; h.globalAlpha = 0.16; h.beginPath(); xs.forEach((x, i) => (i ? h.lineTo(vb.X(x), vb.Y(P2[i].m + 1.96 * P2[i].sd)) : h.moveTo(vb.X(x), vb.Y(P2[i].m + 1.96 * P2[i].sd)))); for (let i = xs.length - 1; i >= 0; i--) h.lineTo(vb.X(xs[i]), vb.Y(P2[i].m - 1.96 * P2[i].sd)); h.closePath(); h.fill(); h.globalAlpha = 1;
      h.strokeStyle = C.orange; h.lineWidth = 2.5; h.beginPath(); xs.forEach((x, i) => (i ? h.lineTo(vb.X(x), vb.Y(P2[i].m)) : h.moveTo(vb.X(x), vb.Y(P2[i].m)))); h.stroke();
      X.forEach((x, i) => { if (i < n) { h.fillStyle = C.blue; h.beginPath(); h.arc(vb.X(x), vb.Y(Y[i]), 5, 0, 7); h.fill(); } else if (i === n && !done) { h.strokeStyle = C.green; h.lineWidth = 2; h.globalAlpha = phase >= 1 ? 1 : 0.3; h.beginPath(); h.arc(vb.X(x), vb.Y(Y[i]), 7, 0, 7); h.stroke(); h.globalAlpha = 1; } });
      font(h, 11); h.fillStyle = C.muted; h.textAlign = 'left'; h.fillText(lin ? 'mean ± 1.96·√(x²·s + σ²)' : 'GP mean ± 1.96·sd (pinches at data)', vb.pad.l + 6, vb.pad.t + 10);
      const at = (x) => { const i = Math.round((x / 12) * 60); return P2[i]; };
      q('#out').innerHTML = `<span class="m">points absorbed</span> ${n}/${X.length}\n` + (lin ? `<span class="m">posterior</span> mean <span class="o">${AI.fmt(pr.m, 3)}</span>  var ${AI.fmt(pr.v, 4)}  sd ${AI.fmt(Math.sqrt(pr.v), 3)}\n<span class="m">precision</span> ${AI.fmt(1 / pr.v, 2)}\n` : '') + `<span class="m">predictive sd at x=2 / 6 / 10:</span>\n${AI.fmt(at(2).sd, 2)} / ${AI.fmt(at(6).sd, 2)} / ${AI.fmt(at(10).sd, 2)}`;
    }, { fps: 30 });
  };

  /* ================= 6. A/B test ================= */
  AI.labs.mlABTest = function (root) {
    const { view, ctl } = AI.labShell(root, 'A/B test lab: users stream in', 'conversion rates, confidence interval and p-value evolve · the peeking problem (simulated users)',
      `<canvas id="cv" style="height:420px"></canvas><div class="caption">Top: conversion rate of each group. Middle: difference B−A with 95% interval (dashed = planned sample size). Bottom: p-value (red line = 0.05). Each user is randomly assigned to A or B and converts with the true rate you set.</div>`,
      `${sld('pa', 'True conversion rate of A (%)', 2, 30, 0.5, 10)}
       ${sld('lift', 'True lift of B over A (points)', -3, 5, 0.5, 0)}
       ${sld('rate', 'Arrival speed (users / second)', 50, 600, 50, 200)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="peek"> Peek: stop as soon as p &lt; 0.05</label>
       ${btns('<button class="btn" id="sim">Run 300 experiments</button>')}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas(q('#cv'));
    const NMAX = 5000, st = { playing: true }; // per group
    let rnd = AI.rng(11), nA, nB, cA, cB, hist, stopped, sim = null, acc = 0;
    const reset = () => { nA = nB = cA = cB = 0; hist = []; stopped = false; acc = 0; };
    reset();
    const stats = () => {
      if (nA < 30 || nB < 30) return null;
      const a = cA / nA, b = cB / nB, d = b - a, pp = (cA + cB) / (nA + nB), se0 = Math.sqrt(pp * (1 - pp) * (1 / nA + 1 / nB)), se1 = Math.sqrt((a * (1 - a)) / nA + (b * (1 - b)) / nB);
      const z = se0 ? d / se0 : 0; return { a, b, d, se1, z, p: 2 * (1 - normCdf(Math.abs(z))) };
    };
    const addUser = () => {
      const pa = +q('#pa').value / 100, pb = pa + +q('#lift').value / 100;
      if (rnd() < 0.5) { nA++; if (rnd() < pa) cA++; } else { nB++; if (rnd() < AI.clamp(pb, 0, 1)) cB++; }
      const n = Math.min(nA, nB);
      if ((nA + nB) % 40 === 0) { const s = stats(); if (s) { hist.push({ n, ...s }); if (q('#peek').checked && s.p < 0.05 && n >= 100) stopped = true; } }
    };
    q('#reset').onclick = reset;
    ['pa', 'lift'].forEach((id) => (q('#' + id).oninput = reset));
    play(root, st, () => { for (let i = 0; i < 40; i++) addUser(); });
    q('#sim').onclick = () => {
      const pa = +q('#pa').value / 100, pb = AI.clamp(pa + +q('#lift').value / 100, 0, 1), r = AI.rng(Math.floor(Math.random() * 1e9)), E = 300, step = 100;
      let ever = 0, endSig = 0;
      for (let e = 0; e < E; e++) {
        let a = 0, b = 0, hit = false;
        for (let n = 1; n <= NMAX; n++) {
          if (r() < pa) a++; if (r() < pb) b++;
          if (n % step === 0) { const pp = (a + b) / (2 * n), se = Math.sqrt(pp * (1 - pp) * (2 / n)); const z = se ? (b / n - a / n) / se : 0, p = 2 * (1 - normCdf(Math.abs(z))); if (p < 0.05) hit = true; if (n === NMAX && p < 0.05) endSig++; }
        }
        if (hit) ever++;
      }
      sim = { E, ever, endSig, truth: +q('#lift').value };
    };
    AI.loop(root, (dt) => {
      ['pa', 'lift', 'rate'].forEach((id) => (q('#' + id + 'v').textContent = q('#' + id).value));
      if (st.playing && !stopped && Math.min(nA, nB) < NMAX) { acc += dt * +q('#rate').value; const n = Math.min(Math.floor(acc), 60); acc -= n; for (let i = 0; i < n; i++) addUser(); }
      const C = AI.col(), g = c.ctx, s = stats(), H = c.h;
      g.clearRect(0, 0, c.w, H);
      const pa = +q('#pa').value / 100, lift = +q('#lift').value / 100;
      const planned = Math.ceil((2 * (1.96 + 0.8416) ** 2 * (pa + lift / 2) * (1 - pa - lift / 2)) / (lift * lift || 1e-9));
      const panel = (i, y0, y1) => AI.view(c, 0, NMAX, y0, y1, { l: 44, r: 10, t: 14 + i * (H / 3), b: H - (i + 1) * (H / 3) + 14 });
      const v1 = panel(0, Math.max(0, pa - 0.06), pa + 0.1), v2 = panel(1, -0.06, 0.08), v3 = panel(2, 0, 1);
      [v1, v2, v3].forEach((v) => { AI.axes(g, v, { noGrid: false }); });
      const line = (v, key, col, w = 2) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); hist.forEach((o, i) => (i ? g.lineTo(v.X(o.n), v.Y(key(o))) : g.moveTo(v.X(o.n), v.Y(key(o))))); g.stroke(); };
      line(v1, (o) => o.a, C.blue); line(v1, (o) => o.b, C.orange);
      g.fillStyle = C.blue; font(g, 11); g.textAlign = 'left'; g.fillText('A', v1.pad.l + 6, v1.pad.t + 12); g.fillStyle = C.orange; g.fillText('B', v1.pad.l + 24, v1.pad.t + 12);
      g.fillStyle = C.purple; g.globalAlpha = 0.2; g.beginPath(); hist.forEach((o, i) => (i ? g.lineTo(v2.X(o.n), v2.Y(o.d + 1.96 * o.se1)) : g.moveTo(v2.X(o.n), v2.Y(o.d + 1.96 * o.se1)))); for (let i = hist.length - 1; i >= 0; i--) g.lineTo(v2.X(hist[i].n), v2.Y(hist[i].d - 1.96 * hist[i].se1)); g.closePath(); g.fill(); g.globalAlpha = 1;
      line(v2, (o) => o.d, C.purple, 2);
      g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.moveTo(v2.pad.l, v2.Y(0)); g.lineTo(v2.pad.l + v2.pw, v2.Y(0)); g.stroke();
      g.strokeStyle = C.green; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(v2.pad.l, v2.Y(lift)); g.lineTo(v2.pad.l + v2.pw, v2.Y(lift)); g.stroke();
      if (planned <= NMAX && lift !== 0) { g.setLineDash([5, 4]); g.strokeStyle = C.muted; [v1, v2, v3].forEach((v) => { g.beginPath(); g.moveTo(v.X(planned), v.pad.t); g.lineTo(v.X(planned), v.pad.t + v.ph); g.stroke(); }); }
      g.setLineDash([]); g.fillStyle = C.muted; g.fillText('difference B − A, 95% CI (green dotted = true lift)', v2.pad.l + 6, v2.pad.t + 12);
      line(v3, (o) => o.p, C.red, 2);
      g.strokeStyle = C.red; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v3.pad.l, v3.Y(0.05)); g.lineTo(v3.pad.l + v3.pw, v3.Y(0.05)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.muted; g.fillText('p-value (two-sided z-test)', v3.pad.l + 6, v3.pad.t + 12); g.textAlign = 'right'; g.fillText('users per group →', v3.pad.l + v3.pw, v3.pad.t + v3.ph + 14);
      const done = Math.min(nA, nB) >= NMAX;
      const cap = stopped ? 'STOPPED at p < 0.05 — you "declared a winner" (is it real?)' : done ? 'finished: planned sample size reached' : !st.playing ? 'paused' : 'users arrive, are randomly assigned to A or B, and may convert';
      caption(g, c.w, cap, Math.min(nA, nB) / NMAX);
      let t = `<span class="m">users</span> A ${nA}  B ${nB}\n`;
      if (s) t += `<span class="m">rates</span> A ${AI.fmt(s.a * 100, 2)}%  B ${AI.fmt(s.b * 100, 2)}%\n<span class="m">diff</span> <span class="o">${AI.fmt(s.d * 100, 2)} pts</span>  CI [${AI.fmt((s.d - 1.96 * s.se1) * 100, 2)}, ${AI.fmt((s.d + 1.96 * s.se1) * 100, 2)}]\n<span class="m">z</span> ${AI.fmt(s.z, 2)}  <span class="m">p</span> <span class="${s.p < 0.05 ? 'r' : 'g'}">${AI.fmt(s.p, 3)}</span>\n`;
      if (lift) t += `<span class="m">planned n / group (80% power)</span> ${planned}\n`;
      if (sim) t += `<span class="m">${sim.E} simulated experiments (true lift ${sim.truth} pts):</span>\nsignificant at the end only: ${sim.endSig} (${AI.fmt((100 * sim.endSig) / sim.E, 1)}%)\nsignificant at ANY peek (every 100): <span class="r">${sim.ever} (${AI.fmt((100 * sim.ever) / sim.E, 1)}%)</span>`;
      q('#out').innerHTML = t;
    }, { fps: 30 });
  };

  /* ================= 7. Simpson's paradox ================= */
  AI.labs.mlSimpson = function (root) {
    const { view, ctl } = AI.labShell(root, "Confounding lab: Simpson's paradox", 'kidney-stone success counts (Charig et al. 1986) · drag the group sizes',
      `<canvas id="cv" style="height:340px"></canvas><div class="caption">Left: success rate inside each stone-size group (bar width = number of patients). Right: the pooled rate = weighted average of those same rates, stacked by group.</div>`,
      `${sld('as', 'Treatment A · small stones (patients)', 0, 400, 1, 87)}
       ${sld('al', 'Treatment A · large stones (patients)', 0, 400, 1, 263)}
       ${sld('bs', 'Treatment B · small stones (patients)', 0, 400, 1, 270)}
       ${sld('bl', 'Treatment B · large stones (patients)', 0, 400, 1, 80)}
       ${sld('sp', 'Speed', 0.25, 3, 0.25, 1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset animation</button><button class="btn" id="eq">Equal mix (randomised)</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas(q('#cv')), st = { playing: true };
    const rate = { as: 81 / 87, al: 192 / 263, bs: 234 / 270, bl: 55 / 80 };
    let p = 0, hold = 0;
    q('#reset').onclick = () => { p = 0; hold = 0; };
    q('#eq').onclick = () => { ['as', 'bs'].forEach((k) => (q('#' + k).value = 180)); ['al', 'bl'].forEach((k) => (q('#' + k).value = 170)); };
    play(root, st, () => { p = Math.min(1, Math.floor(p * 4) / 4 + 0.25); });
    AI.loop(root, (dt) => {
      const sp = spd(root); ['as', 'al', 'bs', 'bl'].forEach((k) => (q('#' + k + 'v').textContent = q('#' + k).value));
      if (st.playing) { if (p < 1) p = Math.min(1, p + (dt * sp) / 7); else { hold += dt * sp; if (hold > 3) { hold = 0; p = 0; } } }
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const n = { as: +q('#as').value, al: +q('#al').value, bs: +q('#bs').value, bl: +q('#bl').value };
      const nA = n.as + n.al, nB = n.bs + n.bl;
      const oA = nA ? (n.as * rate.as + n.al * rate.al) / nA : 0, oB = nB ? (n.bs * rate.bs + n.bl * rate.bl) / nB : 0;
      const e1 = AI.smooth(p / 0.45), e2 = AI.smooth((p - 0.5) / 0.45);
      const top = 66, bot = H - 40, ph = bot - top, Y = (r) => bot - r * ph;
      // left panel: 2 strata x 2 treatments, widths ~ n
      const Lx = 40, Lw = W * 0.58 - 50, maxW = Math.max(1, n.as + n.bs + n.al + n.bl);
      font(g, 11); g.textAlign = 'center';
      g.strokeStyle = C.border; [0, 0.5, 1].forEach((r) => { g.beginPath(); g.moveTo(Lx - 6, Y(r)); g.lineTo(Lx + Lw + 6, Y(r)); g.stroke(); g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(r * 100 + '%', Lx - 8, Y(r) + 4); });
      let x = Lx; const gap = 14, avail = Lw - gap, wf = avail / maxW;
      const cells = [['A', 'small', n.as, rate.as, C.blue, 'small stones'], ['B', 'small', n.bs, rate.bs, C.orange, ''], ['A', 'large', n.al, rate.al, C.blue, 'large stones'], ['B', 'large', n.bl, rate.bl, C.orange, '']];
      cells.forEach(([t, s, nn, r, col, lab], i) => {
        const w = nn * wf; if (i === 2) x += gap;
        g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x, Y(r * e1), w, r * e1 * ph); g.globalAlpha = 1;
        g.textAlign = 'center'; g.fillStyle = C.fg; if (w > 24) { g.fillText(t, x + w / 2, bot + 14); if (e1 > 0.9) g.fillText(AI.fmt(r * 100, 0) + '%', x + w / 2, Y(r) - 5); g.fillStyle = C.muted; g.fillText(nn, x + w / 2, bot + 28); }
        if (lab) { g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(lab, x, top - 18); }
        x += w;
      });
      // right panel: pooled stack
      const Rx = W * 0.62, bw = (W - Rx - 30) / 2 - 10;
      [['A', nA, [[n.as, rate.as], [n.al, rate.al]], C.blue, oA], ['B', nB, [[n.bs, rate.bs], [n.bl, rate.bl]], C.orange, oB]].forEach(([t, tot, parts, col, ov], i) => {
        const bx = Rx + 10 + i * (bw + 20); let acc = 0;
        parts.forEach(([nn, r], j) => { const hgt = tot ? (nn * r) / tot : 0; g.fillStyle = col; g.globalAlpha = j ? 0.5 : 0.95; g.fillRect(bx, Y(acc + hgt * e2), bw, hgt * e2 * ph); g.globalAlpha = 1; acc += hgt * e2; });
        g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText('overall ' + t, bx + bw / 2, bot + 14); if (e2 > 0.9) { font(g, 13); g.fillText(AI.fmt(ov * 100, 1) + '%', bx + bw / 2, Y(ov) - 6); font(g, 11); }
      });
      g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('pooled: dark = small, light = large stones', Rx, top - 18);
      const winS = rate.as > rate.bs && rate.al > rate.bl ? 'A wins in BOTH groups' : 'mixed';
      const winO = oA > oB ? 'A' : oB > oA ? 'B' : 'tie';
      caption(g, W, p < 0.5 ? '① inside each stone size, compare A with B' : '② pool the patients: weights differ, so the average can flip', p);
      q('#out').innerHTML = `<span class="m">small stones</span> A ${AI.fmt(rate.as * 100, 1)}%  B ${AI.fmt(rate.bs * 100, 1)}%\n<span class="m">large stones</span> A ${AI.fmt(rate.al * 100, 1)}%  B ${AI.fmt(rate.bl * 100, 1)}%\n<span class="m">${winS}</span>\n<span class="m">overall</span> A ${AI.fmt(oA * 100, 1)}%  B ${AI.fmt(oB * 100, 1)}%  → <span class="o">${winO} looks better</span>\n<span class="m">share of large stones</span> A ${AI.fmt(nA ? (100 * n.al) / nA : 0, 0)}%  B ${AI.fmt(nB ? (100 * n.bl) / nB : 0, 0)}%`;
    }, { fps: 30 });
  };
})();
