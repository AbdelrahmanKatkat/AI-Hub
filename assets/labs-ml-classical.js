/* Module 3 labs (classical algorithms): knnlab, nblab, treelab, svmlab, kmeanslab, hdblab, pcalab, isolab.
 * All animation is dt-based, slow by default, captioned, with speed / pause / step / reset. Needs lib.js + mllib.js. */
(function () {
  const ML = window.ML;
  const PB = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>
    <label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>`;
  const CHK = (id, text, on) => `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="${id}" ${on ? 'checked' : ''}> ${text}</label>`;
  const plane = (c, R) => { const s = Math.min(c.w, c.h), ox = (c.w - s) / 2, oy = (c.h - s) / 2; return { s, ox, oy, R, X: (x) => ox + ((x + R) / (2 * R)) * s, Y: (y) => oy + ((R - y) / (2 * R)) * s, ix: (px) => ((px - ox) / s) * 2 * R - R, iy: (py) => R - ((py - oy) / s) * 2 * R }; };
  const dot = (g, x, y, r, fill, stroke, lw) => { g.beginPath(); g.arc(x, y, r, 0, 7); if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1; g.stroke(); } };
  const ease = (t) => AI.smooth(t);
  /* wires play/pause/step/reset + speed; st.playing, st.speed() */
  const wire = (root, st, onReset, onStep) => {
    const q = (s) => AI.$(s, root);
    st.playing = true; st.speed = () => +q('#sp').value;
    const setLbl = () => (q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play');
    q('#play').onclick = () => { st.playing = !st.playing; setLbl(); };
    q('#step').onclick = () => { st.playing = false; setLbl(); onStep && onStep(); };
    q('#reset').onclick = () => { st.playing = true; setLbl(); onReset && onReset(); };
    st.updSpeed = () => (q('#spv').textContent = '×' + AI.fmt(+q('#sp').value, 2));
  };
  const caption = (root, txt) => { const el = AI.$('#cap', root); if (el && el._t !== txt) { el._t = txt; el.innerHTML = txt; } };
  const ctext = (g, txt, x, y, col, font, align) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = align || 'left'; g.fillText(txt, x, y); };

  /* ================= k-NN ================= */
  AI.labs.knnlab = function (root) {
    const { ctl } = AI.labShell(root, 'k-NN lab: neighbours vote', 'click the plane to place a query; neighbours light up nearest-first',
      `<canvas id="cv" style="height:400px;cursor:crosshair"></canvas><div class="caption" id="cap"></div>`,
      `<label>Dataset <select id="ds"><option value="moons">moons</option><option value="circles">circles</option><option value="linear">overlapping blobs</option><option value="xor">xor</option><option value="stretched">stretched units</option></select></label>
       <label>k (neighbours) <span class="v" id="kv"></span><input type="range" id="k" min="1" max="25" step="1" value="5"></label>
       ${CHK('scale', 'scale features (standardise)', true)}${CHK('bd', 'show decision map', true)}${PB}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv')), R = 1.7, st = {};
    let data, W = 1, query = [0.25, 0.2], shown = 0, nb = [], dirty = true, key = '', bc = document.createElement('canvas'), bkey = '';
    bc.width = bc.height = 220;
    c.onresize = () => (dirty = true);
    const mk = () => {
      const ds = q('#ds').value;
      if (ds === 'stretched') { const r = AI.rng(3), X = [], y = []; for (let i = 0; i < 100; i++) { const cl = i % 2; X.push([(r() * 2 - 1) * 1.2, (cl ? 0.4 : -0.4) + 0.2 * AI.randn(r)]); y.push(cl); } data = { X, y, K: 2 }; }
      else data = ML.data(ds, 100, 4, { noise: ds === 'circles' ? 0.12 : 0.16 });
      bkey = ''; resetVote();
    };
    const dist = (a, p) => Math.hypot((a[0] - p[0]) * W, a[1] - p[1]);
    const nbrs = (p, k) => data.X.map((a, i) => ({ i, d: dist(a, p) })).sort((u, v) => u.d - v.d).slice(0, k);
    const resetVote = () => { shown = 0; nb = nbrs(query, +q('#k').value); dirty = true; };
    q('#ds').onchange = mk; q('#k').oninput = () => { resetVote(); }; q('#scale').onchange = () => { W = q('#scale').checked ? 1 : 6; bkey = ''; resetVote(); };
    q('#bd').onchange = () => (dirty = true);
    wire(root, st, resetVote, () => { shown = Math.min(nb.length, Math.floor(shown + 1e-6) + 1); dirty = true; });
    q('#cv').addEventListener('click', (e) => { const r = q('#cv').getBoundingClientRect(), pl = plane(c, R); const x = pl.ix(e.clientX - r.left), y = pl.iy(e.clientY - r.top); if (Math.abs(x) <= R && Math.abs(y) <= R) { query = [x, y]; resetVote(); st.playing = true; q('#play').textContent = '⏸ Pause'; } });
    mk();
    AI.loop(root, (dt) => {
      st.updSpeed(); const k = +q('#k').value; q('#kv').textContent = k;
      if (st.playing && shown < nb.length) { shown = Math.min(nb.length, shown + dt * 0.8 * st.speed() * (k > 8 ? k / 8 : 1)); dirty = true; }
      const bk = [q('#ds').value, k, W].join(); if (q('#bd').checked && bk !== bkey) { bkey = bk; const K = ML.knn; const g = bc.getContext('2d'); g.clearRect(0, 0, 220, 220);
        ML.boundary(g, 220, 220, (x, y) => { const nn = nbrs([x, y], k), o = [0, 0]; nn.forEach((n) => o[data.y[n.i]]++); return [o[0] / k, o[1] / k]; }, { range: R, res: 46, K: 2 }); dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, pl = plane(c, R), cols = ML.COLORS(2);
      g.clearRect(0, 0, c.w, c.h);
      g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      if (q('#bd').checked) g.drawImage(bc, pl.ox, pl.oy, pl.s, pl.s);
      g.strokeStyle = C.border; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      const qx = pl.X(query[0]), qy = pl.Y(query[1]), n = Math.floor(shown + 1e-6), part = shown - n;
      const lit = new Set(nb.slice(0, n).map((o) => o.i));
      data.X.forEach((p, i) => { dot(g, pl.X(p[0]), pl.Y(p[1]), lit.has(i) ? 6 : 4, cols[data.y[i]], lit.has(i) ? C.fg : C.bg, lit.has(i) ? 2 : 1); });
      if (n > 0) { const rad = nb[n - 1].d; g.strokeStyle = C.yellow; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath(); g.ellipse(qx, qy, (rad / W / (2 * R)) * pl.s, (rad / (2 * R)) * pl.s, 0, 0, 7); g.stroke(); g.setLineDash([]); }
      nb.slice(0, n).forEach((o, r) => { const p = data.X[o.i]; g.strokeStyle = cols[data.y[o.i]]; g.lineWidth = 2; g.beginPath(); g.moveTo(qx, qy); g.lineTo(pl.X(p[0]), pl.Y(p[1])); g.stroke(); ctext(g, '#' + (r + 1), pl.X(p[0]) + 8, pl.Y(p[1]) - 6, C.fg, 'bold 11px sans-serif'); });
      if (n < nb.length && part > 0) { const p = data.X[nb[n].i]; g.strokeStyle = C.muted; g.globalAlpha = 0.6; g.lineWidth = 1; g.beginPath(); g.moveTo(qx, qy); g.lineTo(qx + (pl.X(p[0]) - qx) * ease(part), qy + (pl.Y(p[1]) - qy) * ease(part)); g.stroke(); g.globalAlpha = 1; }
      g.fillStyle = C.fg; g.beginPath(); g.moveTo(qx, qy - 10); g.lineTo(qx + 9, qy + 7); g.lineTo(qx - 9, qy + 7); g.closePath(); g.fill();
      // vote panel
      const votes = [0, 0]; nb.slice(0, n).forEach((o) => votes[data.y[o.i]]++); const kk = nb.length;
      const bx = pl.ox + pl.s - 128, by = pl.oy + 8; g.fillStyle = C.bg; g.globalAlpha = 0.85; g.fillRect(bx - 6, by - 4, 124, 64); g.globalAlpha = 1;
      votes.forEach((v, j) => { g.fillStyle = C.border; g.fillRect(bx, by + 8 + j * 24, 100, 12); g.fillStyle = cols[j]; g.fillRect(bx, by + 8 + j * 24, (100 * v) / kk, 12); ctext(g, 'class ' + j + ': ' + v + ' vote' + (v === 1 ? '' : 's'), bx, by + 5 + j * 24, C.fg, '10px sans-serif'); });
      const done = n >= kk, lead = votes[0] === votes[1] ? -1 : votes[0] > votes[1] ? 0 : 1;
      caption(root, n === 0 ? '① Measure the distance from the query (triangle) to every stored point; only the k closest will vote.'
        : !done ? `② Neighbour #${n}: distance ${AI.fmt(nb[n - 1].d, 2)} → votes <b>class ${data.y[nb[n - 1].i]}</b>. Votes so far: ${votes[0]} vs ${votes[1]}.`
        : lead < 0 ? `③ Tie ${votes[0]}–${votes[1]}: with an even k a tie-break rule is needed — use an odd k.` : `③ Majority vote of ${kk}: <b>class ${lead}</b> with probability ${votes[lead]}/${kk} = ${AI.fmt(votes[lead] / kk, 2)}.`);
      q('#out').innerHTML = `<span class="m">k</span> ${k}   <span class="m">stored points</span> ${data.X.length}\n<span class="m">distance computations / query</span> ${data.X.length}\n<span class="m">metric</span> ${W === 1 ? 'standardised Euclidean' : 'raw Euclidean (x counts 6×)'}`;
    }, { fps: 30 });
  };

  /* ================= Naive Bayes ================= */
  AI.labs.nblab = function (root) {
    const { ctl } = AI.labShell(root, 'Naive Bayes lab: word probabilities and the posterior', 'toggle the words of an email, change the smoothing α, watch the evidence accumulate',
      `<canvas id="cv" style="height:400px"></canvas><div class="caption" id="cap"></div>`,
      `<div>Words in the email (click to toggle, order = click order)<div class="chips" id="words"></div></div>
       <label>Laplace smoothing α <span class="v" id="av"></span><input type="range" id="al" min="0" max="5" step="0.25" value="1"></label>${PB}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv')), st = {};
    const V = ['win', 'money', 'now', 'prize', 'free', 'meeting', 'lunch', 'project', 'update'], cS = [2, 3, 1, 1, 1, 0, 0, 0, 0], cH = [0, 0, 2, 0, 0, 2, 1, 1, 1], NS = 8, NH = 7;
    let qw = ['win', 'now'], reveal = 0, dS = V.map(() => 0), dH = V.map(() => 0), dPost = 0.5, dirty = true;
    c.onresize = () => (dirty = true);
    const pS = (i, a) => (cS[i] + a) / (NS + a * V.length), pH = (i, a) => (cH[i] + a) / (NH + a * V.length);
    const chips = () => { const el = q('#words'); el.innerHTML = ''; V.forEach((w) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = w; if (qw.includes(w)) { b.style.borderColor = 'var(--blue)'; b.style.color = 'var(--blue)'; } b.onclick = () => { qw = qw.includes(w) ? qw.filter((x) => x !== w) : [...qw, w]; reveal = 0; chips(); dirty = true; }; el.appendChild(b); }); };
    chips();
    q('#al').oninput = () => { reveal = 0; dirty = true; };
    wire(root, st, () => { reveal = 0; dirty = true; }, () => { reveal = Math.min(qw.length, Math.floor(reveal + 1e-6) + 1); dirty = true; });
    AI.loop(root, (dt) => {
      st.updSpeed(); const a = +q('#al').value; q('#av').textContent = a;
      if (st.playing && reveal < qw.length) { reveal = Math.min(qw.length, reveal + dt * 0.4 * st.speed()); dirty = true; }
      const kk = 1 - Math.exp(-dt * 4);
      // product table
      const idx = qw.map((w) => V.indexOf(w)), prodS = [0.5], prodH = [0.5]; idx.forEach((i, j) => { prodS.push(prodS[j] * pS(i, a)); prodH.push(prodH[j] * pH(i, a)); });
      const step = Math.floor(reveal + 1e-6), post = (j) => (prodS[j] + prodH[j] === 0 ? NaN : prodS[j] / (prodS[j] + prodH[j]));
      const tgtPost = post(step); let moving = false;
      V.forEach((_, i) => { const ts = pS(i, a), th = pH(i, a); if (Math.abs(dS[i] - ts) > 1e-4 || Math.abs(dH[i] - th) > 1e-4) moving = true; dS[i] += (ts - dS[i]) * kk; dH[i] += (th - dH[i]) * kk; });
      if (!Number.isNaN(tgtPost)) { if (Math.abs(dPost - tgtPost) > 1e-3) moving = true; dPost += (tgtPost - dPost) * kk; }
      if (!dirty && !moving) return; dirty = false;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const lw = Math.min(c.w * 0.5, 330), rowH = 30, y0 = 36;
      ctext(g, 'P(word | class), smoothed', 8, 16, C.fg, 'bold 12px sans-serif');
      g.fillStyle = C.orange; g.fillRect(lw - 120, 6, 10, 10); ctext(g, 'spam', lw - 106, 15, C.muted, '11px sans-serif'); g.fillStyle = C.blue; g.fillRect(lw - 66, 6, 10, 10); ctext(g, 'ham', lw - 52, 15, C.muted, '11px sans-serif');
      V.forEach((w, i) => {
        const y = y0 + i * rowH, inq = qw.includes(w);
        if (inq) { g.fillStyle = C.bg3; g.globalAlpha = 0.7; g.fillRect(4, y - 4, lw, rowH - 2); g.globalAlpha = 1; }
        ctext(g, w, 10, y + 15, inq ? C.fg : C.muted, (inq ? 'bold ' : '') + '12px sans-serif');
        const bw = lw - 160, sc = bw / 0.45;
        g.fillStyle = C.orange; g.fillRect(70, y, dS[i] * sc, 10); g.fillStyle = C.blue; g.fillRect(70, y + 12, dH[i] * sc, 10);
        ctext(g, AI.fmt(dS[i], 3), 74 + dS[i] * sc, y + 9, C.muted, '10px monospace'); ctext(g, AI.fmt(dH[i], 3), 74 + dH[i] * sc, y + 21, C.muted, '10px monospace');
      });
      // right panel
      const rx = lw + 24, rw = c.w - rx - 10;
      if (rw > 120) {
        ctext(g, 'Evidence for the email: ' + (qw.length ? qw.join(' + ') : '(no words selected)'), rx, 16, C.fg, 'bold 12px sans-serif');
        const rows = [['prior', 0.5, 0.5]].concat(idx.map((i, j) => ['× P(' + V[i] + '|·)', pS(i, a), pH(i, a)]));
        g.font = '11px monospace';
        ctext(g, 'step', rx, y0 + 4, C.muted, '10px sans-serif'); ctext(g, 'spam running product', rx + 70, y0 + 4, C.orange, '10px sans-serif'); ctext(g, 'ham running product', rx + 70 + rw * 0.45, y0 + 4, C.blue, '10px sans-serif');
        rows.forEach((r, j) => { if (j > step) return; const y = y0 + 20 + j * 22; g.globalAlpha = j === step && step < reveal + 1 && reveal % 1 > 0 ? 0.5 + 0.5 * (reveal % 1) : 1;
          ctext(g, r[0], rx, y, C.fg, '11px monospace'); ctext(g, 'x' + AI.fmt(r[1], 3) + ' = ' + AI.fmt(prodS[j], 5), rx + 70, y, C.orange, '11px monospace'); ctext(g, 'x' + AI.fmt(r[2], 3) + ' = ' + AI.fmt(prodH[j], 5), rx + 70 + rw * 0.45, y, C.blue, '11px monospace'); g.globalAlpha = 1; });
        const by = y0 + 30 + (Math.min(rows.length, step + 1)) * 22 + 14, ph = Math.min(by, c.h - 70);
        ctext(g, 'posterior P(spam | words)', rx, ph, C.fg, '12px sans-serif');
        g.fillStyle = C.blue; g.fillRect(rx, ph + 8, rw - 10, 22); g.fillStyle = C.orange; g.fillRect(rx, ph + 8, (rw - 10) * (Number.isNaN(tgtPost) ? 0.5 : dPost), 22);
        ctext(g, Number.isNaN(tgtPost) ? '0 / 0: undefined' : AI.fmt(tgtPost * 100, 1) + '% spam', rx + 8, ph + 24, '#fff', 'bold 12px sans-serif');
      }
      const veto = idx.length && (prodS[step] === 0 || prodH[step] === 0);
      caption(root, !qw.length ? 'Select some words to classify an email.' : step === 0 ? '① Start from the priors: 3 of 6 training emails are spam, so P(spam) = P(ham) = 0.5. Next: multiply in one word at a time.'
        : `② After <b>${V[idx[step - 1]]}</b>: spam ${AI.fmt(prodS[step], 5)}, ham ${AI.fmt(prodH[step], 5)}.` + (veto ? ' <span style="color:var(--red)">A zero count vetoes a class — raise α above 0.</span>' : '') + (step === qw.length ? ` ③ Posterior: ${Number.isNaN(tgtPost) ? 'undefined (0/0)' : AI.fmt(tgtPost * 100, 1) + '% spam'}.` : ''));
      q('#out').innerHTML = `<span class="m">P(w|c) = (count + α) / (N_c + α·V)</span>\nspam N = ${NS}, ham N = ${NH}, V = ${V.length}\n<span class="m">α</span> ${a}`;
    }, { fps: 30 });
  };

  /* ================= Decision tree ================= */
  AI.labs.treelab = function (root) {
    const { ctl } = AI.labShell(root, 'Decision-tree lab: watch the tree grow', 'each split: scan the candidates, pick the best gain, cut the plane',
      `<canvas id="pl" style="height:340px"></canvas><canvas id="gn" style="height:120px"></canvas><canvas id="tr" style="height:230px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Dataset <select id="ds"><option value="moons">moons</option><option value="circles">circles</option><option value="xor">xor</option><option value="blobs">two blobs</option></select></label>
       <label>Max depth <span class="v" id="dv"></span><input type="range" id="md" min="1" max="8" step="1" value="3"></label>
       <label>Min samples per leaf <span class="v" id="lv"></span><input type="range" id="ml" min="1" max="12" step="1" value="1"></label>
       <label>Impurity <select id="cr"><option value="gini">Gini</option><option value="entropy">entropy</option></select></label>
       ${PB}<div class="row"><button class="btn" id="nd">New data</button></div><div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cp = AI.canvas(q('#pl')), cg = AI.canvas(q('#gn')), ct = AI.canvas(q('#tr')), st = {}, R = 1.7;
    let tr, train, test, events = [], e = 0, ph = 0, t = 0, applied = 0, dirty = true, seed = 5, regions = [];
    [cp, cg, ct].forEach((o) => (o.onresize = () => (dirty = true)));
    const cands = (idx, crit) => {
      const out = [], K = 2, cnt = (ids) => { const a = [0, 0]; ids.forEach((i) => a[train.y[i]]++); return a; }, par = ML.tree.impurity(cnt(idx), crit);
      for (let f = 0; f < 2; f++) { const s = [...idx].sort((a, b) => train.X[a][f] - train.X[b][f]);
        for (let j = 0; j < s.length - 1; j++) { const v0 = train.X[s[j]][f], v1 = train.X[s[j + 1]][f]; if (v0 === v1) continue; const L = s.slice(0, j + 1), Rr = s.slice(j + 1);
          out.push({ f, t: (v0 + v1) / 2, gain: par - (L.length * ML.tree.impurity(cnt(L), crit) + Rr.length * ML.tree.impurity(cnt(Rr), crit)) / s.length }); } }
      return out;
    };
    const build = () => {
      const ds = q('#ds').value, o = { noise: 0.2, K: 2 };
      train = ML.data(ds, 70, seed, o); test = ML.data(ds, 100, seed + 100, o);
      const crit = q('#cr').value;
      tr = ML.tree.fit(train.X, train.y, { maxDepth: +q('#md').value, minLeaf: +q('#ml').value, criterion: crit });
      events = []; let id = 0; const queue = [{ n: tr.root, r: { x0: -R, x1: R, y0: -R, y1: R } }], lay = { leaf: 0 };
      while (queue.length) { const { n, r } = queue.shift(); n.rect = r; if (n.leaf) continue;
        const cd = cands(n.idx, crit), bi = cd.findIndex((x) => x.f === n.feature && Math.abs(x.t - n.threshold) < 1e-9);
        n.ev = events.length; events.push({ node: n, rect: r, cd, best: bi });
        const L = n.feature === 0 ? { x0: r.x0, x1: n.threshold, y0: r.y0, y1: r.y1 } : { x0: r.x0, x1: r.x1, y0: r.y0, y1: n.threshold };
        const Rr = n.feature === 0 ? { x0: n.threshold, x1: r.x1, y0: r.y0, y1: r.y1 } : { x0: r.x0, x1: r.x1, y0: n.threshold, y1: r.y1 };
        queue.push({ n: n.left, r: L }, { n: n.right, r: Rr }); }
      const lay2 = (n, d) => { if (n.leaf) { n.lx = lay.leaf++; n.ly = d; return; } lay2(n.left, d + 1); lay2(n.right, d + 1); n.lx = (n.left.lx + n.right.lx) / 2; n.ly = d; };
      lay2(tr.root, 0); lay.n = lay.leaf; tr.lay = lay;
      e = 0; ph = 0; t = 0; applied = 0; regions = [{ node: tr.root, rect: tr.root.rect }]; dirty = true;
    };
    const durOf = (p) => { const ev = events[e]; return (p === 0 ? Math.min(4, 0.8 + 0.08 * ev.cd.length) : p === 1 ? 1.0 : 1.2) / st.speed(); };
    const finish = () => { if (ph === 2) { const ev = events[e], i = regions.findIndex((r) => r.node === ev.node); regions.splice(i, 1, { node: ev.node.left, rect: ev.node.left.rect }, { node: ev.node.right, rect: ev.node.right.rect }); applied++; e++; ph = 0; t = 0; } else { ph++; t = 0; } dirty = true; };
    ['#ds', '#md', '#ml', '#cr'].forEach((s) => (q(s).onchange = build)); q('#md').oninput = () => (q('#dv').textContent = q('#md').value); q('#ml').oninput = () => (q('#lv').textContent = q('#ml').value);
    q('#nd').onclick = () => { seed += 7; build(); };
    wire(root, st, build, () => { if (e < events.length) { const d = durOf(ph); t = d; finish(); } });
    build();
    const predict = (p) => { let n = tr.root; while (!n.leaf && n.ev < applied) n = p[n.feature] <= n.threshold ? n.left : n.right; return n.counts[1] > n.counts[0] ? 1 : 0; };
    const acc = (d) => d.X.reduce((s, p, i) => s + (predict(p) === d.y[i] ? 1 : 0), 0) / d.X.length;
    AI.loop(root, (dt) => {
      st.updSpeed(); q('#dv').textContent = q('#md').value; q('#lv').textContent = q('#ml').value;
      if (st.playing && e < events.length) { t += dt; dirty = true; if (t >= durOf(ph)) finish(); }
      if (!dirty) return; dirty = false;
      const C = AI.col(), cols = ML.COLORS(2), g = cp.ctx, pl = plane(cp, R), ev = events[e], doneAll = e >= events.length;
      g.clearRect(0, 0, cp.w, cp.h); g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      const prog = ev ? Math.min(1, t / durOf(ph)) : 1;
      const regs = regions.slice(); let newKids = [];
      if (ev && ph === 2) { newKids = [ev.node.left, ev.node.right]; const i = regs.findIndex((r) => r.node === ev.node); regs.splice(i, 1, { node: ev.node.left, rect: ev.node.left.rect, fade: ease(prog) }, { node: ev.node.right, rect: ev.node.right.rect, fade: ease(prog) }); }
      regs.forEach((r) => { const n = r.node, tot = n.counts[0] + n.counts[1], p1 = tot ? n.counts[1] / tot : 0.5, col = AI.mix(cols[0], cols[1], p1); g.fillStyle = col; g.globalAlpha = 0.28 * (r.fade ?? 1) * (0.5 + 0.5 * Math.abs(2 * p1 - 1)); g.fillRect(pl.X(r.rect.x0), pl.Y(r.rect.y1), pl.X(r.rect.x1) - pl.X(r.rect.x0), pl.Y(r.rect.y0) - pl.Y(r.rect.y1)); });
      g.globalAlpha = 1;
      const line = (rc, f, th) => { g.beginPath(); if (f === 0) { g.moveTo(pl.X(th), pl.Y(rc.y0)); g.lineTo(pl.X(th), pl.Y(rc.y1)); } else { g.moveTo(pl.X(rc.x0), pl.Y(th)); g.lineTo(pl.X(rc.x1), pl.Y(th)); } g.stroke(); };
      for (let i = 0; i < applied; i++) { g.strokeStyle = C.fg; g.lineWidth = 1.5; line(events[i].rect, events[i].node.feature, events[i].node.threshold); }
      if (ev) {
        const r = ev.rect; g.strokeStyle = C.yellow; g.lineWidth = 2; g.setLineDash([6, 4]); g.strokeRect(pl.X(r.x0) + 1, pl.Y(r.y1) + 1, pl.X(r.x1) - pl.X(r.x0) - 2, pl.Y(r.y0) - pl.Y(r.y1) - 2); g.setLineDash([]);
        if (ph === 0) { const ci = Math.min(ev.cd.length - 1, Math.floor(prog * ev.cd.length)), cd = ev.cd[ci]; g.strokeStyle = cols[cd.f]; g.lineWidth = 1.5; g.setLineDash([4, 4]); line(r, cd.f, cd.t); g.setLineDash([]); }
        else { const n = ev.node; g.strokeStyle = C.green; g.lineWidth = ph === 1 ? 2 + 2 * Math.sin(prog * Math.PI * 3) ** 2 : 3; line(r, n.feature, n.threshold); }
      }
      train.X.forEach((p, i) => dot(g, pl.X(p[0]), pl.Y(p[1]), 4.5, cols[train.y[i]], C.fg, 1));
      test.X.forEach((p, i) => dot(g, pl.X(p[0]), pl.Y(p[1]), 2.8, null, cols[test.y[i]], 1.2));
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      ctext(g, '● train   ○ test', pl.ox + 6, pl.oy + 14, C.muted, '11px sans-serif');
      // gains
      const gg = cg.ctx; gg.clearRect(0, 0, cg.w, cg.h);
      if (ev) { const cd = ev.cd, mx = Math.max(0.02, ...cd.map((x) => x.gain)), bw = Math.max(2, (cg.w - 20) / cd.length), shownN = ph === 0 ? Math.floor(prog * cd.length) + 1 : cd.length;
        ctext(gg, 'impurity gain of each candidate split (blue: x, orange: y)', 8, 13, C.muted, '11px sans-serif');
        cd.forEach((x, i) => { if (i >= shownN) return; const h = (x.gain / mx) * (cg.h - 40); gg.fillStyle = cols[x.f]; gg.globalAlpha = ph === 0 && i === shownN - 1 ? 1 : 0.65; gg.fillRect(10 + i * bw, cg.h - 8 - h, Math.max(1, bw - 1), h); });
        gg.globalAlpha = 1; if (ph > 0 && ev.best >= 0) { const bi = ev.best, h = (ev.cd[bi].gain / mx) * (cg.h - 40); gg.strokeStyle = C.green; gg.lineWidth = 2; gg.strokeRect(10 + bi * bw - 1, cg.h - 8 - h - 1, Math.max(1, bw - 1) + 2, h + 2); ctext(gg, `best: ${ev.node.feature ? 'y' : 'x'} ≤ ${AI.fmt(ev.node.threshold, 2)}  gain ${AI.fmt(ev.node.gain, 3)}`, Math.min(cg.w - 190, 14 + bi * bw), 28, C.green, 'bold 11px sans-serif'); } }
      else ctext(gg, 'no more splits: every leaf is pure, or depth / min-leaf stops the growth', 8, 20, C.muted, '12px sans-serif');
      // tree diagram
      const gt = ct.ctx; gt.clearRect(0, 0, ct.w, ct.h); const nl = tr.lay.n, ux = (ct.w - 24) / Math.max(1, nl), uy = Math.min(46, (ct.h - 30) / (tr.depth + 1)), small = ux < 34;
      const pos = (n) => [12 + (n.lx + 0.5) * ux, 18 + n.ly * uy];
      const vis = (n, parentOk) => parentOk;
      const walk = (n, parentVis) => { if (!parentVis) return; const [x, y] = pos(n), internalApplied = !n.leaf && n.ev < applied + (ev && ph === 2 ? 1 : 0);
        if (!n.leaf && internalApplied) [n.left, n.right].forEach((ch) => { const [cx, cy] = pos(ch); gt.strokeStyle = C.muted; gt.lineWidth = 1; gt.beginPath(); gt.moveTo(x, y); gt.lineTo(cx, cy); gt.stroke(); });
        const tot = n.counts[0] + n.counts[1], p1 = n.counts[1] / tot, col = AI.mix(cols[0], cols[1], p1), isCur = ev && n === ev.node, asLeaf = n.leaf || !internalApplied;
        dot(gt, x, y, small ? 4 : 9, col, isCur ? C.yellow : C.fg, isCur ? 2.5 : 1);
        if (!small) ctext(gt, asLeaf ? n.counts[0] + '|' + n.counts[1] : (n.feature ? 'y' : 'x') + '≤' + AI.fmt(n.threshold, 1), x, y + 22, C.muted, '9px sans-serif', 'center');
        if (!n.leaf && internalApplied) { walk(n.left, true); walk(n.right, true); } };
      walk(tr.root, true);
      ctext(gt, 'tree so far (counts class0|class1)', 8, ct.h - 6, C.muted, '10px sans-serif');
      const trA = acc(train), teA = acc(test), warn = doneAll && trA - teA > 0.08;
      caption(root, doneAll ? (warn ? `<span style="color:var(--red)">⚠ Overfit warning:</span> train accuracy ${AI.fmt(trA * 100, 0)}% but test only ${AI.fmt(teA * 100, 0)}%. Lower the depth or raise min leaf.` : 'Tree complete. Compare train and test accuracy; change depth or min leaf to see the boundary change.')
        : ph === 0 ? `① Split ${e + 1} of ${events.length}: scan every threshold on x and y inside the dashed region (${ev.cd.length} candidates) and compute its gain.`
        : ph === 1 ? `② Best candidate: ${ev.node.feature ? 'y' : 'x'} ≤ ${AI.fmt(ev.node.threshold, 2)} with gain ${AI.fmt(ev.node.gain, 3)} (parent impurity ${AI.fmt(ev.node.imp, 3)}).`
        : '③ Apply the split: two child regions take the majority colour of their points.');
      q('#out').innerHTML = `<span class="m">splits done</span> ${applied}/${events.length}   <span class="m">leaves now</span> ${applied + 1}   <span class="m">depth cap</span> ${q('#md').value}\n<span class="m">train acc</span> <b>${AI.fmt(trA * 100, 0)}%</b>   <span class="m">test acc</span> <b>${AI.fmt(teA * 100, 0)}%</b>${warn ? '\n<span class="r">overfit: train ≫ test</span>' : ''}`;
    }, { fps: 30 });
  };
  /* ================= SVM ================= */
  function smo(X, y, K, C) {
    const n = X.length, a = new Array(n).fill(0), r = AI.rng(11); let b = 0, passes = 0, guard = 0;
    const f = (i) => { let s = b; for (let j = 0; j < n; j++) if (a[j]) s += a[j] * y[j] * K[j][i]; return s; };
    while (passes < 8 && guard++ < 200) {
      let ch = 0;
      for (let i = 0; i < n; i++) {
        const Ei = f(i) - y[i];
        if ((y[i] * Ei < -1e-3 && a[i] < C) || (y[i] * Ei > 1e-3 && a[i] > 0)) {
          let j = Math.floor(r() * (n - 1)); if (j >= i) j++;
          const Ej = f(j) - y[j], ai = a[i], aj = a[j];
          const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C), H = y[i] !== y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
          if (L === H) continue; const eta = 2 * K[i][j] - K[i][i] - K[j][j]; if (eta >= 0) continue;
          let ajn = aj - (y[j] * (Ei - Ej)) / eta; ajn = Math.min(H, Math.max(L, ajn)); if (Math.abs(ajn - aj) < 1e-5) continue;
          const ain = ai + y[i] * y[j] * (aj - ajn);
          const b1 = b - Ei - y[i] * (ain - ai) * K[i][i] - y[j] * (ajn - aj) * K[i][j], b2 = b - Ej - y[i] * (ain - ai) * K[i][j] - y[j] * (ajn - aj) * K[j][j];
          a[i] = ain; a[j] = ajn; b = ain > 0 && ain < C ? b1 : ajn > 0 && ajn < C ? b2 : (b1 + b2) / 2; ch++;
        }
      }
      passes = ch ? 0 : passes + 1;
    }
    return { a, b };
  }
  AI.labs.svmlab = function (root) {
    const { ctl } = AI.labShell(root, 'SVM lab: margin, support vectors, C and kernels', 'drag points; double-click a point to flip its class',
      `<canvas id="cv" style="height:420px;cursor:grab"></canvas><div class="caption" id="cap"></div>`,
      `<label>Dataset <select id="ds"><option value="linear">two blobs</option><option value="circles">circles</option><option value="xor">xor</option></select></label>
       <label>Kernel <select id="kn"><option value="lin">linear</option><option value="rbf">RBF</option></select></label>
       <label>C (violation penalty) <span class="v" id="cv2"></span><input type="range" id="c" min="-2" max="2" step="0.1" value="0"></label>
       <label>γ (RBF width) <span class="v" id="gv"></span><input type="range" id="gm" min="-1" max="1" step="0.05" value="0"></label>
       <div class="row"><button class="btn" id="fz">New data</button></div><div class="caption">The optimum is computed exactly with SMO; the easing of the road to it is illustrative.</div><div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv')), R = 1.3; let X = [], y = [], sol = null, dw = [0, 0], db = 0, drag = -1, seed = 2, dirty = true, hm = null, hmOld = null, fade = 1, key = '';
    c.onresize = () => (dirty = true);
    const mk = () => {
      const ds = q('#ds').value; let d;
      if (ds === 'linear') { const r = AI.rng(seed); d = { X: [], y: [] }; for (let i = 0; i < 16; i++) { const cl = i % 2; d.X.push([(cl ? 0.5 : -0.5) + 0.28 * AI.randn(r), (cl ? 0.35 : -0.35) + 0.28 * AI.randn(r)]); d.y.push(cl); } }
      else d = ML.data(ds, 26, seed, { noise: 0.1 });
      X = d.X.map((p) => [AI.clamp(p[0], -1.2, 1.2), AI.clamp(p[1], -1.2, 1.2)]); y = d.y.map((v) => (v ? 1 : -1));
      if (ds !== 'linear') q('#kn').value = 'rbf'; else q('#kn').value = 'lin';
      solve();
    };
    const kern = (a, b) => (q('#kn').value === 'lin' ? a[0] * b[0] + a[1] * b[1] : Math.exp(-Math.pow(10, +q('#gm').value) * 2 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)));
    const solve = () => {
      const n = X.length, Cc = Math.pow(10, +q('#c').value), K = X.map((a) => X.map((b) => kern(a, b))), s = smo(X, y, K, Cc);
      let w = [0, 0]; if (q('#kn').value === 'lin') X.forEach((p, i) => { w[0] += s.a[i] * y[i] * p[0]; w[1] += s.a[i] * y[i] * p[1]; });
      sol = { a: s.a, b: s.b, w, C: Cc, lin: q('#kn').value === 'lin' };
      sol.f = (p) => { let v = sol.b; for (let i = 0; i < n; i++) if (s.a[i] > 1e-6) v += s.a[i] * y[i] * kern(X[i], p); return v; };
      if (!sol.lin) { const old = hm; hmOld = old; hm = document.createElement('canvas'); hm.width = hm.height = 60; const g = hm.getContext('2d'), im = g.createImageData(60, 60), Cc2 = AI.col(), c0 = AI.rgb(Cc2.orange), c1 = AI.rgb(Cc2.blue), bg = AI.rgb(Cc2.bg);
        for (let j = 0; j < 60; j++) for (let i = 0; i < 60; i++) { const v = sol.f([((i + 0.5) / 60) * 2 * R - R, R - ((j + 0.5) / 60) * 2 * R]), t = 1 / (1 + Math.exp(-3 * v)), sc = 0.15 + 0.4 * Math.min(1, Math.abs(v)); for (let k = 0; k < 3; k++) im.data[(j * 60 + i) * 4 + k] = bg[k] + (c0[k] + (c1[k] - c0[k]) * t - bg[k]) * sc; im.data[(j * 60 + i) * 4 + 3] = 255; }
        g.putImageData(im, 0, 0); fade = 0; }
      dirty = true;
    };
    ['#ds', '#kn'].forEach((s) => (q(s).onchange = () => { if (s === '#ds') mk(); else solve(); })); ['#c', '#gm'].forEach((s) => (q(s).oninput = solve)); q('#fz').onclick = () => { seed += 5; mk(); };
    const pt = (e) => { const r = q('#cv').getBoundingClientRect(), pl = plane(c, R); return { px: e.clientX - r.left, py: e.clientY - r.top, pl }; };
    const hit = (e) => { const { px, py, pl } = pt(e); let b = -1, bd = 14; X.forEach((p, i) => { const d = Math.hypot(pl.X(p[0]) - px, pl.Y(p[1]) - py); if (d < bd) { bd = d; b = i; } }); return b; };
    q('#cv').addEventListener('pointerdown', (e) => { drag = hit(e); if (drag >= 0) { q('#cv').setPointerCapture(e.pointerId); q('#cv').style.cursor = 'grabbing'; } });
    q('#cv').addEventListener('pointermove', (e) => { if (drag < 0) return; const { px, py, pl } = pt(e); X[drag] = [AI.clamp(pl.ix(px), -R, R), AI.clamp(pl.iy(py), -R, R)]; solve(); });
    q('#cv').addEventListener('pointerup', () => { drag = -1; q('#cv').style.cursor = 'grab'; });
    q('#cv').addEventListener('dblclick', (e) => { const i = hit(e); if (i >= 0) { y[i] = -y[i]; solve(); } });
    mk(); dw = sol.w.slice(); db = sol.b;
    AI.loop(root, (dt) => {
      const Cc = Math.pow(10, +q('#c').value); q('#cv2').textContent = AI.fmt(Cc, 2); q('#gv').textContent = AI.fmt(Math.pow(10, +q('#gm').value), 2);
      const kk = 1 - Math.exp(-dt * 3.5); let moving = false;
      if (sol.lin) { const dd = Math.abs(dw[0] - sol.w[0]) + Math.abs(dw[1] - sol.w[1]) + Math.abs(db - sol.b); if (dd > 1e-3) { moving = true; dw[0] += (sol.w[0] - dw[0]) * kk; dw[1] += (sol.w[1] - dw[1]) * kk; db += (sol.b - db) * kk; } else { dw = sol.w.slice(); db = sol.b; } }
      if (fade < 1) { fade = Math.min(1, fade + dt / 0.8); moving = true; }
      if (!dirty && !moving) return; dirty = false;
      const C = AI.col(), g = c.ctx, pl = plane(c, R), cols = [C.orange, C.blue]; g.clearRect(0, 0, c.w, c.h);
      g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      g.save(); g.beginPath(); g.rect(pl.ox, pl.oy, pl.s, pl.s); g.clip();
      if (!sol.lin && hm) { g.imageSmoothingEnabled = true; if (hmOld && fade < 1) g.drawImage(hmOld, pl.ox, pl.oy, pl.s, pl.s); g.globalAlpha = fade; g.drawImage(hm, pl.ox, pl.oy, pl.s, pl.s); g.globalAlpha = 1; }
      const lines = (off, st, wd, dsh) => { // w.x + b = off
        const w = sol.lin ? dw : [0, 0], bb = db; if (Math.hypot(w[0], w[1]) < 1e-6) return; g.strokeStyle = st; g.lineWidth = wd; g.setLineDash(dsh); g.beginPath();
        let a, b2; if (Math.abs(w[1]) > Math.abs(w[0])) { a = [-R, (off - bb - w[0] * -R) / w[1]]; b2 = [R, (off - bb - w[0] * R) / w[1]]; } else { a = [(off - bb - w[1] * -R) / w[0], -R]; b2 = [(off - bb - w[1] * R) / w[0], R]; }
        g.moveTo(pl.X(a[0]), pl.Y(a[1])); g.lineTo(pl.X(b2[0]), pl.Y(b2[1])); g.stroke(); g.setLineDash([]); };
      let sv = 0, hingeSum = 0, hinge = [];
      X.forEach((p, i) => { const v = sol.f(p), h = Math.max(0, 1 - y[i] * v); hinge.push(h); hingeSum += h; if (sol.a[i] > 1e-6) sv++; });
      if (sol.lin) { const wn = Math.hypot(dw[0], dw[1]); if (wn > 1e-6) { const wd = Math.abs(dw[1]) > Math.abs(dw[0]); g.fillStyle = C.green; g.globalAlpha = 0.1; g.beginPath();
        const edge = (off) => (wd ? [[-R, (off - db + dw[0] * R) / dw[1]], [R, (off - db - dw[0] * R) / dw[1]]] : [[(off - db + dw[1] * R) / dw[0], -R], [(off - db - dw[1] * R) / dw[0], R]]);
        const e1 = edge(1), e2 = edge(-1); g.moveTo(pl.X(e1[0][0]), pl.Y(e1[0][1])); g.lineTo(pl.X(e1[1][0]), pl.Y(e1[1][1])); g.lineTo(pl.X(e2[1][0]), pl.Y(e2[1][1])); g.lineTo(pl.X(e2[0][0]), pl.Y(e2[0][1])); g.closePath(); g.fill(); g.globalAlpha = 1; }
        lines(0, C.fg, 2.5, []); lines(1, C.fg, 1.2, [6, 4]); lines(-1, C.fg, 1.2, [6, 4]); }
      X.forEach((p, i) => { const x = pl.X(p[0]), yy = pl.Y(p[1]), cl = y[i] > 0 ? 1 : 0; if (sol.a[i] > 1e-6) dot(g, x, yy, 11, null, C.yellow, 2.5); dot(g, x, yy, 5.5, cols[cl], C.fg, 1.2); if (hinge[i] > 1e-3) ctext(g, AI.fmt(hinge[i], 2), x + 9, yy - 8, C.red, '10px monospace'); });
      g.restore(); g.strokeStyle = C.border; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      ctext(g, '○ ring = support vector   red number = hinge loss', pl.ox + 6, pl.oy + pl.s - 8, C.muted, '11px sans-serif');
      const wn = Math.hypot(sol.w[0], sol.w[1]);
      caption(root, sol.lin ? `The solid line is w·x + b = 0; dashed lines are ±1 and the green band is the margin, width 2/‖w‖ = ${wn > 1e-6 ? AI.fmt(2 / wn, 2) : '–'}. Only ringed points (support vectors) hold it in place. Lower C to let points sit inside the band (hinge &gt; 0) for a wider road.`
        : 'RBF kernel: the colour map is the decision function. Small γ → smooth blobs; large γ → each support vector gets its own island (overfit). Ringed points are the support vectors.');
      q('#out').innerHTML = `<span class="m">support vectors</span> ${sv} of ${X.length}\n<span class="m">Σ hinge</span> ${AI.fmt(hingeSum, 2)}` + (sol.lin ? `   <span class="m">‖w‖</span> ${AI.fmt(wn, 3)}\n<span class="m">margin 2/‖w‖</span> <b>${wn > 1e-6 ? AI.fmt(2 / wn, 3) : '–'}</b>\n<span class="m">objective 0.5‖w‖² + C·Σhinge</span> ${AI.fmt(0.5 * wn * wn + Cc * hingeSum, 3)}` : '');
    }, { fps: 30 });
  };

  /* ================= k-means ================= */
  AI.labs.kmeanslab = function (root) {
    const { ctl } = AI.labShell(root, 'k-means lab: assign, update, repeat', 'watch the centroids move and the inertia fall; try a bad start',
      `<canvas id="cv" style="height:360px"></canvas><canvas id="in" style="height:110px"></canvas><canvas id="el" style="height:110px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Dataset <select id="ds"><option value="3">3 blobs</option><option value="4">4 blobs</option><option value="uneven">uneven blobs</option><option value="moons">moons (hard)</option></select></label>
       <label>k (clusters) <span class="v" id="kv"></span><input type="range" id="k" min="1" max="8" step="1" value="3"></label>
       ${PB}<div class="row"><button class="btn" id="rs">Restart (new seed)</button><button class="btn" id="bad">Bad initialisation</button><button class="btn" id="elb">Compute elbow</button></div><div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.canvas(q('#cv')), ci = AI.canvas(q('#in')), ce = AI.canvas(q('#el')), st = {}, R = 1.3;
    let X, km, seed = 1, phase = 'assign', t = 0, prevL = [], from = [], hist = [], elbow = null, conv = false, trail = [], dirty = true, ch = 0, order = [];
    [cv, ci, ce].forEach((o) => (o.onresize = () => (dirty = true)));
    const DUR = 2.2;
    const data = () => { const d = q('#ds').value; if (d === 'moons') return ML.data('moons', 150, 3, { noise: 0.08 }).X; if (d === 'uneven') { const r = AI.rng(9), P = []; [[-0.7, 0.5, 0.25, 70], [0.6, 0.6, 0.12, 25], [0, -0.7, 0.18, 55]].forEach(([cx, cy, s, n]) => { for (let i = 0; i < n; i++) P.push([cx + s * AI.randn(r), cy + s * AI.randn(r)]); }); return P; } return ML.data('blobs', 150, 5, { K: +d }).X; };
    const init = (bad) => {
      const k = +q('#k').value; km = ML.kmeans(X, k, seed);
      if (bad) { const a = X[Math.floor(AI.rng(seed + 3)() * X.length)], s = X.map((p, i) => [Math.hypot(p[0] - a[0], p[1] - a[1]), i]).sort((u, v) => u[0] - v[0]); km.C.forEach((cc, j) => { cc[0] = X[s[j][1]][0]; cc[1] = X[s[j][1]][1]; }); }
      phase = 'assign'; t = 0; prevL = X.map(() => -1); from = km.C.map((c) => [...c]); hist = []; conv = false; trail = km.C.map((c) => [[...c]]); begin(); dirty = true;
    };
    const begin = () => { if (phase === 'assign') { prevL = km.labels.slice(); ch = km.assign(); order = X.map((p, i) => i).sort((a, b) => X[a][0] - X[b][0]); hist.push(km.inertia()); } else { from = km.C.map((c) => [...c]); km.update(); trail.forEach((tr, j) => tr.push([...km.C[j]])); hist.push(km.inertia()); } };
    const finish = () => { if (phase === 'assign') { if (ch === 0) { conv = true; dirty = true; return; } phase = 'update'; } else { phase = 'assign'; km.iter++; } t = 0; begin(); dirty = true; };
    const reseed = () => { seed += 1; X = data(); init(false); };
    q('#ds').onchange = () => { X = data(); init(false); }; q('#k').oninput = () => { q('#kv').textContent = q('#k').value; init(false); };
    q('#rs').onclick = () => { seed += 1; init(false); }; q('#bad').onclick = () => init(true);
    q('#elb').onclick = () => { const out = []; for (let k = 1; k <= 8; k++) { let best = Infinity; for (let s = 0; s < 6; s++) { const m = ML.kmeans(X, k, 100 + s); for (let i = 0; i < 40; i++) { const c = m.assign(); m.update(); if (!c && i) break; } m.assign(); best = Math.min(best, m.inertia()); } out.push(best); } elbow = out; dirty = true; };
    wire(root, st, () => init(false), () => { if (!conv) { t = DUR; finish(); t = DUR * 0.999; } });
    X = data(); init(false);
    AI.loop(root, (dt) => {
      st.updSpeed(); q('#kv').textContent = q('#k').value;
      if (st.playing && !conv) { t += (dt * st.speed()); dirty = true; if (t >= DUR) { t = DUR; finish(); } }
      if (!dirty) return; dirty = false;
      const C = AI.col(), cols = [C.blue, C.orange, C.green, C.purple, C.pink, C.teal, C.yellow, C.red], g = cv.ctx, pl = plane(cv, R), p = Math.min(1, t / DUR);
      g.clearRect(0, 0, cv.w, cv.h); g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      const n = X.length, sweepN = phase === 'assign' ? Math.floor(ease(p) * n) : n;
      const rank = new Array(n); order.forEach((i, r) => (rank[i] = r));
      const cur = km.C.map((c, j) => (phase === 'update' && from[j] ? [AI.lerp(from[j][0], c[0], ease(Math.min(1, p / 0.85))), AI.lerp(from[j][1], c[1], ease(Math.min(1, p / 0.85)))] : [c[0], c[1]]));
      X.forEach((pt, i) => { const lab = phase === 'assign' ? (rank[i] < sweepN ? km.labels[i] : prevL[i]) : km.labels[i]; const col = lab >= 0 ? cols[lab % 8] : C.muted;
        if (phase === 'assign' && lab >= 0 && rank[i] < sweepN && rank[i] > sweepN - 8) { g.strokeStyle = cols[lab % 8]; g.globalAlpha = 0.5; g.lineWidth = 1; g.beginPath(); g.moveTo(pl.X(pt[0]), pl.Y(pt[1])); g.lineTo(pl.X(km.C[lab][0]), pl.Y(km.C[lab][1])); g.stroke(); g.globalAlpha = 1; }
        dot(g, pl.X(pt[0]), pl.Y(pt[1]), 3.6, col, null); });
      trail.forEach((tr, j) => { if (tr.length > 1) { g.strokeStyle = cols[j % 8]; g.lineWidth = 1.5; g.setLineDash([3, 3]); g.beginPath(); tr.forEach((q2, i) => (i ? g.lineTo(pl.X(q2[0]), pl.Y(q2[1])) : g.moveTo(pl.X(q2[0]), pl.Y(q2[1])))); if (phase === 'update') g.lineTo(pl.X(cur[j][0]), pl.Y(cur[j][1])); g.stroke(); g.setLineDash([]); } });
      cur.forEach((cc, j) => { const x = pl.X(cc[0]), y = pl.Y(cc[1]); g.fillStyle = cols[j % 8]; g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.rect(x - 7, y - 7, 14, 14); g.fill(); g.stroke(); });
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      // inertia
      const gi = ci.ctx; gi.clearRect(0, 0, ci.w, ci.h); const v = AI.view(ci, 0, Math.max(6, hist.length), 0, Math.max(0.01, ...hist) * 1.1, { l: 46, r: 10, t: 16, b: 20 }); AI.axes(gi, v);
      gi.strokeStyle = C.purple; gi.lineWidth = 2; gi.beginPath(); hist.forEach((h, i) => (i ? gi.lineTo(v.X(i), v.Y(h)) : gi.moveTo(v.X(i), v.Y(h)))); gi.stroke(); hist.forEach((h, i) => dot(gi, v.X(i), v.Y(h), 2.5, C.purple));
      ctext(gi, 'inertia (sum of squared distances) after each half-step', 48, 11, C.muted, '11px sans-serif');
      // elbow
      const ge = ce.ctx; ge.clearRect(0, 0, ce.w, ce.h); const k = +q('#k').value;
      if (elbow) { const v2 = AI.view(ce, 1, 8, 0, Math.max(...elbow) * 1.1, { l: 46, r: 10, t: 16, b: 20 }); AI.axes(ge, v2); ge.strokeStyle = C.green; ge.lineWidth = 2; ge.beginPath(); elbow.forEach((h, i) => (i ? ge.lineTo(v2.X(i + 1), v2.Y(h)) : ge.moveTo(v2.X(1), v2.Y(h)))); ge.stroke(); elbow.forEach((h, i) => dot(ge, v2.X(i + 1), v2.Y(h), i + 1 === k ? 5 : 3, i + 1 === k ? C.yellow : C.green)); ctext(ge, 'elbow: best-of-6 inertia vs k (look for where the fall flattens)', 48, 11, C.muted, '11px sans-serif'); }
      else ctext(ge, 'Press “Compute elbow” to plot final inertia against k.', 12, 30, C.muted, '12px sans-serif');
      caption(root, conv ? `✔ Converged after ${km.iter + 1} assignment rounds: no point changed cluster. Final inertia ${AI.fmt(km.inertia(), 2)}. Try “Restart” with another seed or “Bad initialisation” and compare.`
        : phase === 'assign' ? `① Assign (iteration ${km.iter + 1}): each point joins the nearest centroid — ${AI.fmt(sweepN, 0)} of ${n} done; ${ch} point${ch === 1 ? '' : 's'} will change cluster.`
        : `② Update (iteration ${km.iter + 1}): every centroid glides to the mean of its points; inertia falls to ${AI.fmt(km.inertia(), 2)}.`);
      q('#out').innerHTML = `<span class="m">k</span> ${k}   <span class="m">iteration</span> ${km.iter + 1}   <span class="m">inertia</span> <b>${AI.fmt(km.inertia(), 2)}</b>\n<span class="m">cost/iter</span> n·k·d = ${n * k * 2}`;
    }, { fps: 30 });
  };
  /* ================= Hierarchical clustering & DBSCAN ================= */
  AI.labs.hdblab = function (root) {
    const { ctl } = AI.labShell(root, 'Hierarchical vs DBSCAN lab', 'merge the closest clusters one by one, or grow clusters by density',
      `<canvas id="cv" style="height:340px"></canvas><canvas id="dn" style="height:150px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Method <select id="me"><option value="h">hierarchical (merge tree)</option><option value="d">DBSCAN (density)</option></select></label>
       <label>Dataset <select id="ds"><option value="moons">moons</option><option value="blobs">3 blobs</option><option value="circles">circles</option></select></label>
       <label>Linkage (hierarchical) <select id="lk"><option value="single">single (nearest pair)</option><option value="average">average</option><option value="complete">complete (farthest pair)</option></select></label>
       <label>Stop at clusters <span class="v" id="kv"></span><input type="range" id="k" min="1" max="8" step="1" value="2"></label>
       <label>ε (radius) <span class="v" id="ev"></span><input type="range" id="eps" min="0.05" max="0.5" step="0.01" value="0.16"></label>
       <label>minPts <span class="v" id="mv"></span><input type="range" id="mp" min="2" max="8" step="1" value="4"></label>
       ${PB}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.canvas(q('#cv')), cd = AI.canvas(q('#dn')), st = {}, R = 1.7;
    let X = [], D = [], merges = [], prog = 0, evs = [], dlab = [], dirty = true, leafOrder = [], posX = [], kids = {};
    [cv, cd].forEach((o) => (o.onresize = () => (dirty = true)));
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const mkdata = () => { const d = q('#ds').value; X = d === 'blobs' ? ML.data('blobs', 90, 6, { K: 3 }).X : ML.data(d, 90, 6, { noise: d === 'moons' ? 0.07 : 0.045 }).X; const n = X.length; D = X.map((a) => X.map((b) => dist(a, b))); };
    const hier = () => {
      const n = X.length, lk = q('#lk').value, cdm = D.map((r) => r.slice()), size = new Array(n).fill(1), id = Array.from({ length: n }, (_, i) => i), act = new Set(id); merges = []; kids = {}; let nid = n;
      while (act.size > 1) { let bi = -1, bj = -1, bd = Infinity; const A = [...act]; for (let x = 0; x < A.length; x++) for (let y = x + 1; y < A.length; y++) { const d = cdm[A[x]][A[y]]; if (d < bd) { bd = d; bi = A[x]; bj = A[y]; } }
        merges.push({ a: id[bi], b: id[bj], d: bd, i: bi, j: bj, cid: nid }); kids[nid] = [id[bi], id[bj]];
        for (const k of act) { if (k === bi || k === bj) continue; const di = cdm[bi][k], dj = cdm[bj][k], nv = lk === 'single' ? Math.min(di, dj) : lk === 'complete' ? Math.max(di, dj) : (di * size[bi] + dj * size[bj]) / (size[bi] + size[bj]); cdm[bi][k] = cdm[k][bi] = nv; }
        size[bi] += size[bj]; act.delete(bj); id[bi] = nid++; }
      leafOrder = []; const walk = (c) => { if (c < n) leafOrder.push(c); else kids[c].forEach(walk); }; walk(nid - 1); posX = new Array(n); leafOrder.forEach((p, i) => (posX[p] = i));
      prog = 0;
    };
    const dbs = () => {
      const eps = +q('#eps').value, mp = +q('#mp').value, n = X.length, lab = new Array(n).fill(-1), vis = new Array(n).fill(false), nb = (i) => { const o = []; for (let j = 0; j < n; j++) if (D[i][j] <= eps) o.push(j); return o; }; evs = []; let cl = -1;
      for (let i = 0; i < n; i++) { if (vis[i]) continue; vis[i] = true; const N = nb(i); if (N.length < mp) { lab[i] = -2; evs.push({ i, type: 'noise', cl: -1, nn: N.length }); continue; }
        cl++; lab[i] = cl; evs.push({ i, type: 'core', cl, nn: N.length }); const queue = [...N];
        while (queue.length) { const j = queue.shift(); if (lab[j] === -2) { lab[j] = cl; evs.push({ i: j, type: 'border', cl, nn: 0 }); continue; } if (lab[j] !== -1) continue; lab[j] = cl; vis[j] = true; const Nj = nb(j); if (Nj.length >= mp) { evs.push({ i: j, type: 'core', cl, nn: Nj.length }); Nj.forEach((m) => queue.push(m)); } else evs.push({ i: j, type: 'border', cl, nn: Nj.length }); } }
      dlab = lab; prog = 0;
    };
    const rebuild = () => { if (q('#me').value === 'h') hier(); else dbs(); dirty = true; };
    ['#me', '#lk', '#k', '#eps', '#mp'].forEach((s) => (q(s).oninput = () => { if (s === '#k') { dirty = true; return; } rebuild(); })); q('#ds').onchange = () => { mkdata(); rebuild(); };
    const total = () => (q('#me').value === 'h' ? Math.max(0, X.length - +q('#k').value) : evs.length);
    wire(root, st, () => { prog = 0; dirty = true; }, () => { prog = Math.min(total(), Math.floor(prog + 1e-6) + 1); dirty = true; });
    mkdata(); rebuild();
    const pal = () => { const C = AI.col(); return [C.blue, C.orange, C.green, C.purple, C.pink, C.teal, C.yellow, C.red]; };
    AI.loop(root, (dt) => {
      st.updSpeed(); q('#kv').textContent = q('#k').value; q('#ev').textContent = q('#eps').value; q('#mv').textContent = q('#mp').value;
      const hm = q('#me').value === 'h'; ['#lk', '#k'].forEach((s) => (q(s).parentNode.style.display = hm ? '' : 'none')); ['#eps', '#mp'].forEach((s) => (q(s).parentNode.style.display = hm ? 'none' : '')); cd.cv.style.display = hm ? '' : 'none';
      const tot = total(); if (st.playing && prog < tot) { prog = Math.min(tot, prog + dt * (hm ? 6 : 7) * st.speed()); dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = cv.ctx, pl = plane(cv, R), P = pal(), n = X.length; g.clearRect(0, 0, cv.w, cv.h); g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      let cap = '', info = '';
      if (hm) {
        const m = Math.floor(prog + 1e-6), par = Array.from({ length: n }, (_, i) => i), find = (x) => { while (par[x] !== x) x = par[x] = par[par[x]]; return x; };
        for (let s = 0; s < m; s++) { par[find(merges[s].i)] = find(merges[s].j); }
        const roots = new Map(); for (let i = 0; i < n; i++) { const r = find(i); roots.set(r, (roots.get(r) || 0) + 1); }
        const big = [...roots.entries()].filter((e) => e[1] > 1).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
        X.forEach((p, i) => { const r = find(i), ci = big.indexOf(r); dot(g, pl.X(p[0]), pl.Y(p[1]), ci < 0 ? 3 : 4.5, ci < 0 ? C.muted : P[ci % 8], null); });
        if (m < merges.length && prog < tot) { const mg = merges[m], a = X[mg.i], b = X[mg.j]; g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(pl.X(a[0]), pl.Y(a[1])); g.lineTo(pl.X(b[0]), pl.Y(b[1])); g.stroke(); g.setLineDash([]); }
        const nc = roots.size; cap = prog >= tot && tot > 0 ? `✔ Stopped at ${nc} clusters: the dendrogram cut below shows the height where merging was halted.` : `Merge ${m + 1} of ${n - 1}: the two closest clusters (${q('#lk').value} linkage, distance ${AI.fmt((merges[Math.min(m, merges.length - 1)] || { d: 0 }).d, 3)}) join. Clusters left: ${nc}.`; info = `<span class="m">clusters</span> ${nc}   <span class="m">merges</span> ${m}/${n - 1}\n<span class="m">cost</span> O(n²) memory, O(n³) naive time`;
        // dendrogram
        const gd = cd.ctx; gd.clearRect(0, 0, cd.w, cd.h); const maxD = merges[merges.length - 1].d, W2 = cd.w - 20, hx = (d) => cd.h - 14 - (d / maxD) * (cd.h - 28), xs = (c) => 10 + ((c < n ? posX[c] : null) + 0.5) * (W2 / n);
        const xc = {}; const px = (c) => { if (c < n) return 10 + (posX[c] + 0.5) * (W2 / n); if (xc[c] === undefined) xc[c] = (px(kids[c][0]) + px(kids[c][1])) / 2; return xc[c]; }; const hc = (c) => (c < n ? 0 : merges[c - n].d);
        merges.forEach((mg, s) => { const x1 = px(mg.a), x2 = px(mg.b), y1 = hx(hc(mg.a)), y2 = hx(hc(mg.b)), yt = hx(mg.d); gd.strokeStyle = s < m ? C.fg : C.border; gd.lineWidth = s < m ? 1.5 : 1; gd.beginPath(); gd.moveTo(x1, y1); gd.lineTo(x1, yt); gd.lineTo(x2, yt); gd.lineTo(x2, y2); gd.stroke(); });
        const cutH = m < merges.length ? merges[Math.max(0, m - 1)].d : maxD; if (m > 0) { gd.strokeStyle = C.yellow; gd.setLineDash([5, 4]); gd.beginPath(); gd.moveTo(10, hx(cutH)); gd.lineTo(cd.w - 10, hx(cutH)); gd.stroke(); gd.setLineDash([]); }
        ctext(gd, 'dendrogram: height = merge distance; solid = merges done', 10, 11, C.muted, '10px sans-serif');
      } else {
        const m = Math.floor(prog + 1e-6), cur = evs[Math.min(m, evs.length - 1)], col = new Array(n).fill(-3); for (let s = 0; s < Math.min(m + 1, evs.length); s++) { if (s >= m && prog >= tot - 1e-6 && false) break; col[evs[s].i] = evs[s].type === 'noise' ? -2 : evs[s].cl; }
        const eps = +q('#eps').value, fin = prog >= tot - 1e-6;
        X.forEach((p, i) => { const c = fin ? dlab[i] : col[i], x = pl.X(p[0]), y = pl.Y(p[1]); if (c >= 0) dot(g, x, y, 4.5, P[c % 8], null); else if (c === -2) { g.strokeStyle = C.red; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x - 3, y - 3); g.lineTo(x + 3, y + 3); g.moveTo(x + 3, y - 3); g.lineTo(x - 3, y + 3); g.stroke(); } else dot(g, x, y, 3, C.muted, null); });
        if (cur && !fin) { const p = X[cur.i]; g.strokeStyle = cur.type === 'core' ? C.green : C.yellow; g.lineWidth = 2; g.beginPath(); g.arc(pl.X(p[0]), pl.Y(p[1]), (eps / (2 * R)) * pl.s, 0, 7); g.stroke(); }
        const nCl = dlab.length ? Math.max(...dlab) + 1 : 0; cap = fin ? `✔ ${nCl} cluster${nCl === 1 ? '' : 's'} found; ${dlab.filter((v) => v === -2).length} points are noise (red ×). Change ε or minPts and watch clusters merge, split or dissolve.` : cur.type === 'noise' ? `Point ${cur.i}: only ${cur.nn} points within ε (needs ${q('#mp').value}) → provisional noise.` : cur.type === 'core' ? `Point ${cur.i}: ${cur.nn} points within ε ≥ minPts → <b>core</b> point of cluster ${cur.cl + 1}; its neighbours join the queue.` : `Point ${cur.i} is a <b>border</b> point: inside a core point's ε-circle but not dense itself; it joins cluster ${cur.cl + 1}.`;
        info = `<span class="m">ε</span> ${eps}   <span class="m">minPts</span> ${q('#mp').value}\n<span class="m">events</span> ${Math.min(m + 1, evs.length)}/${evs.length}`;
      }
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      caption(root, cap); q('#out').innerHTML = info;
    }, { fps: 30 });
  };

  /* ================= PCA ================= */
  AI.labs.pcalab = function (root) {
    const { ctl } = AI.labShell(root, 'PCA lab: project onto the direction of most variance', 'a line sweeps through all angles; the best one keeps the most spread',
      `<canvas id="cv" style="height:360px"></canvas><canvas id="vc" style="height:140px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Correlation ρ <span class="v" id="rv"></span><input type="range" id="rho" min="-0.95" max="0.95" step="0.05" value="0.8"></label>
       <label>x units (stretch) <span class="v" id="sv"></span><input type="range" id="sx" min="0.4" max="3" step="0.1" value="1"></label>${PB}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), cv = AI.canvas(q('#cv')), cc = AI.canvas(q('#vc')), st = {};
    let X, S, pc, th1, tr, phase = 0, t = 0, dirty = true, R = 3; const Z = (() => { const r = AI.rng(4); return Array.from({ length: 60 }, () => [AI.randn(r), AI.randn(r)]); })();
    [cv, cc].forEach((o) => (o.onresize = () => (dirty = true)));
    const gen = () => { const rho = +q('#rho').value, sx = +q('#sx').value; X = Z.map((z) => [sx * z[0], rho * z[0] + Math.sqrt(1 - rho * rho) * z[1]]); const m = [0, 1].map((j) => X.reduce((s, p) => s + p[j], 0) / X.length); X = X.map((p) => [p[0] - m[0], p[1] - m[1]]);
      pc = ML.pca2d(X); const a = pc.vals, n = X.length; let sxx = 0, sxy = 0, syy = 0; X.forEach((p) => { sxx += p[0] * p[0]; sxy += p[0] * p[1]; syy += p[1] * p[1]; }); S = [sxx / (n - 1), sxy / (n - 1), syy / (n - 1)];
      th1 = Math.atan2(pc.vecs[0][1], pc.vecs[0][0]); th1 = ((th1 % Math.PI) + Math.PI) % Math.PI; R = 1.2 * Math.max(...X.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
      phase = 0; t = 0; dirty = true; };
    const vr = (th) => { const c = Math.cos(th), s = Math.sin(th); return S[0] * c * c + 2 * S[1] * c * s + S[2] * s * s; };
    ['#rho', '#sx'].forEach((s) => (q(s).oninput = gen));
    const durs = () => [Math.PI / (0.4 * st.speed()), 1.5 / st.speed(), 2.4 / st.speed()];
    wire(root, st, gen, () => { if (phase < 3) { phase++; t = 0; dirty = true; } });
    gen();
    AI.loop(root, (dt) => {
      st.updSpeed(); q('#rv').textContent = q('#rho').value; q('#sv').textContent = q('#sx').value;
      if (st.playing && phase < 3) { t += dt; dirty = true; if (t >= durs()[phase]) { phase++; t = 0; } }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = cv.ctx, pl = plane(cv, R), p = phase < 3 ? Math.min(1, t / durs()[phase]) : 1;
      const th = phase === 0 ? p * Math.PI : phase === 1 ? Math.PI - (Math.PI - th1) * ease(p) : th1, u = [Math.cos(th), Math.sin(th)], pr = phase === 2 ? ease(p) : phase === 3 ? 1 : 0;
      g.clearRect(0, 0, cv.w, cv.h); g.fillStyle = C.bg2; g.fillRect(pl.ox, pl.oy, pl.s, pl.s);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(pl.X(-R), pl.Y(0)); g.lineTo(pl.X(R), pl.Y(0)); g.moveTo(pl.X(0), pl.Y(-R)); g.lineTo(pl.X(0), pl.Y(R)); g.stroke();
      g.strokeStyle = C.green; g.lineWidth = 2.5; g.beginPath(); g.moveTo(pl.X(-R * u[0] * 1.4), pl.Y(-R * u[1] * 1.4)); g.lineTo(pl.X(R * u[0] * 1.4), pl.Y(R * u[1] * 1.4)); g.stroke();
      if (phase === 3) { g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(pl.X(R * u[1] * 1.4), pl.Y(-R * u[0] * 1.4)); g.lineTo(pl.X(-R * u[1] * 1.4), pl.Y(R * u[0] * 1.4)); g.stroke(); g.setLineDash([]); }
      X.forEach((pt) => { const s = pt[0] * u[0] + pt[1] * u[1], fx = s * u[0], fy = s * u[1], x = AI.lerp(pt[0], fx, pr), y = AI.lerp(pt[1], fy, pr);
        g.strokeStyle = C.orange; g.globalAlpha = 0.3 * (1 - pr * 0.6); g.lineWidth = 1; g.beginPath(); g.moveTo(pl.X(pt[0]), pl.Y(pt[1])); g.lineTo(pl.X(fx), pl.Y(fy)); g.stroke(); g.globalAlpha = 1; dot(g, pl.X(x), pl.Y(y), 3.6, C.blue, null); });
      g.strokeStyle = C.border; g.strokeRect(pl.ox + 0.5, pl.oy + 0.5, pl.s - 1, pl.s - 1);
      const tot = pc.vals[0] + pc.vals[1], v = vr(th); ctext(g, 'orange = what is lost (residual)', pl.ox + 6, pl.oy + 14, C.muted, '11px sans-serif');
      // variance curve
      const gv = cc.ctx; gv.clearRect(0, 0, cc.w, cc.h); const vw = AI.view(cc, 0, Math.PI, 0, tot * 1.05, { l: 46, r: 10, t: 16, b: 22 }); AI.axes(gv, vw);
      gv.strokeStyle = C.border; gv.lineWidth = 1.5; gv.beginPath(); for (let i = 0; i <= 90; i++) { const a = (Math.PI * i) / 90; i ? gv.lineTo(vw.X(a), vw.Y(vr(a))) : gv.moveTo(vw.X(a), vw.Y(vr(a))); } gv.stroke();
      gv.strokeStyle = C.purple; gv.lineWidth = 2.5; gv.beginPath(); const last = phase === 0 ? th : Math.PI; for (let i = 0; i <= 90; i++) { const a = (Math.PI * i) / 90; if (a > last) break; i ? gv.lineTo(vw.X(a), vw.Y(vr(a))) : gv.moveTo(vw.X(a), vw.Y(vr(a))); } gv.stroke();
      dot(gv, vw.X(th % Math.PI), vw.Y(v), 5, C.yellow, C.fg, 1.5); if (phase >= 1) { dot(gv, vw.X(th1), vw.Y(pc.vals[0]), 4, null, C.green, 2); ctext(gv, 'maximum = λ₁ = PC1', vw.X(th1) + 8, vw.Y(pc.vals[0]) + 12, C.green, '11px sans-serif'); }
      ctext(gv, 'variance of the projections vs line angle (0 → 180°)', 48, 11, C.muted, '11px sans-serif');
      const ex = pc.vals[0] / tot;
      caption(root, phase === 0 ? `① Sweep: rotate the line and measure the variance of the points' shadows on it (now ${AI.fmt(v, 2)} at ${AI.fmt((th * 180) / Math.PI, 0)}°).` : phase === 1 ? '② The angle with the largest variance is the first principal component — swing the line to it.' : phase === 2 ? '③ Project: every point slides along its orange segment onto the line. That is the 1-D representation.' : `✔ PC1 keeps ${AI.fmt(ex * 100, 1)}% of the variance; the dashed perpendicular PC2 holds the other ${AI.fmt((1 - ex) * 100, 1)}%.`);
      q('#out').innerHTML = `<span class="m">λ₁ (PC1)</span> ${AI.fmt(pc.vals[0], 3)}   <span class="m">λ₂ (PC2)</span> ${AI.fmt(pc.vals[1], 3)}\n<span class="m">variance explained</span> PC1 <b>${AI.fmt(ex * 100, 1)}%</b>, PC2 ${AI.fmt((1 - ex) * 100, 1)}%\n<span class="m">PC1 direction</span> (${AI.fmt(pc.vecs[0][0], 3)}, ${AI.fmt(pc.vecs[0][1], 3)})`;
    }, { fps: 30 });
  };

  /* ================= Isolation forest intuition ================= */
  AI.labs.isolab = function (root) {
    const { ctl } = AI.labShell(root, 'Isolation lab: random cuts isolate the outlier', 'two points, two sequences of random cuts: count how many cuts until each is alone',
      `<canvas id="cv" style="height:380px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Outlier distance from the crowd <span class="v" id="dv"></span><input type="range" id="od" min="0.3" max="1.2" step="0.05" value="1.05"></label>
       <label>Number of trees <span class="v" id="tv"></span><input type="range" id="nt" min="1" max="40" step="1" value="12"></label>${PB}<div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(q('#cv')), st = {}, R = 1.3;
    let P = [], oi = 0, ni = 0, rnd = AI.rng(21), pan = [], round = 0, sums = [0, 0], tcut = 0, wait = 0, dirty = true;
    c.onresize = () => (dirty = true);
    const gen = () => { const r = AI.rng(8); P = []; for (let i = 0; i < 60; i++) P.push([AI.clamp(0.22 * AI.randn(r), -0.7, 0.7), AI.clamp(0.22 * AI.randn(r), -0.7, 0.7)]); let b = 1e9; P.forEach((p, i) => { const d = Math.hypot(p[0] - 0.1, p[1] - 0.05); if (d < b) { b = d; ni = i; } });
      const d = +q('#od').value; P.push([0.8 * d, 0.6 * d]); oi = P.length - 1; reset(); };
    const newTree = () => { pan = [oi, ni].map((tr) => ({ tr, cell: { x0: -R, x1: R, y0: -R, y1: R }, alive: P.map((_, i) => i), cuts: [], h: 0, done: false, anim: 1 })); tcut = 0; wait = 0; };
    const reset = () => { rnd = AI.rng(21); round = 0; sums = [0, 0]; newTree(); dirty = true; };
    const cut = (pn) => { if (pn.done) return; const f = rnd() < 0.5 ? 0 : 1, vals = pn.alive.map((i) => P[i][f]), lo = Math.min(...vals), hi = Math.max(...vals), v = lo + rnd() * (hi - lo), side = P[pn.tr][f] <= v, rc = pn.cell;
      pn.cuts.push({ f, v, rc: { ...rc } }); if (f === 0) { if (side) rc.x1 = v; else rc.x0 = v; } else { if (side) rc.y1 = v; else rc.y0 = v; }
      pn.alive = pn.alive.filter((i) => (P[i][f] <= v) === side); pn.h++; pn.anim = 0; if (pn.alive.length === 1) pn.done = true; };
    const stepAll = () => { pan.forEach(cut); dirty = true; };
    q('#od').oninput = gen; q('#nt').oninput = () => (dirty = true);
    wire(root, st, reset, () => { if (pan.every((x) => x.done)) { endRound(); } else stepAll(); });
    const endRound = () => { sums[0] += pan[0].h; sums[1] += pan[1].h; round++; if (round < +q('#nt').value) newTree(); else { st.playing = false; q('#play').textContent = '▶ Play'; wait = 0; } dirty = true; };
    const cN = (n) => { let H = 0; for (let i = 1; i <= n - 1; i++) H += 1 / i; return 2 * H - (2 * (n - 1)) / n; };
    gen();
    AI.loop(root, (dt) => {
      st.updSpeed(); q('#dv').textContent = q('#od').value; q('#tv').textContent = q('#nt').value;
      const finished = round >= +q('#nt').value;
      if (st.playing && !finished) { const all = pan.every((x) => x.done); const INT = (round < 2 ? 1.2 : 0.3) / st.speed(); pan.forEach((x) => (x.anim = Math.min(1, x.anim + dt / (INT * 0.5)))); dirty = true;
        if (all) { wait += dt; if (wait > 1.4 / st.speed()) endRound(); } else { tcut += dt; if (tcut >= INT) { tcut = 0; stepAll(); } } }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); const pw = (c.w - 14) / 2, s = Math.min(pw, c.h - 46);
      pan.forEach((pn, k) => { const ox = k * (pw + 14) + (pw - s) / 2, oy = 22, X = (x) => ox + ((x + R) / (2 * R)) * s, Y = (y) => oy + ((R - y) / (2 * R)) * s, col = k ? C.blue : C.orange;
        g.fillStyle = C.bg2; g.fillRect(ox, oy, s, s); const rc = pn.cell; g.fillStyle = col; g.globalAlpha = 0.1; g.fillRect(X(rc.x0), Y(rc.y1), X(rc.x1) - X(rc.x0), Y(rc.y0) - Y(rc.y1)); g.globalAlpha = 1;
        pn.cuts.forEach((ct, i) => { const last = i === pn.cuts.length - 1, a = last ? ease(pn.anim) : 1, r0 = ct.rc; g.strokeStyle = C.fg; g.globalAlpha = last ? 1 : 0.55; g.lineWidth = last ? 2 : 1; g.beginPath();
          if (ct.f === 0) { const y0 = Y(r0.y1), y1 = Y(r0.y0), my = (y0 + y1) / 2; g.moveTo(X(ct.v), my - ((y1 - y0) / 2) * a); g.lineTo(X(ct.v), my + ((y1 - y0) / 2) * a); } else { const x0 = X(r0.x0), x1 = X(r0.x1), mx = (x0 + x1) / 2; g.moveTo(mx - ((x1 - x0) / 2) * a, Y(ct.v)); g.lineTo(mx + ((x1 - x0) / 2) * a, Y(ct.v)); } g.stroke(); g.globalAlpha = 1; });
        const alive = new Set(pn.alive); P.forEach((p, i) => { if (i === pn.tr) return; g.globalAlpha = alive.has(i) ? 1 : 0.22; dot(g, X(p[0]), Y(p[1]), 3.2, C.muted, null); }); g.globalAlpha = 1;
        const tp = P[pn.tr]; dot(g, X(tp[0]), Y(tp[1]), 6.5, col, C.fg, 2);
        g.strokeStyle = C.border; g.strokeRect(ox + 0.5, oy + 0.5, s - 1, s - 1);
        ctext(g, k ? 'a point inside the crowd' : 'the outlier', ox, 15, col, 'bold 12px sans-serif'); ctext(g, pn.done ? `isolated after ${pn.h} cut${pn.h === 1 ? '' : 's'}` : `cuts so far: ${pn.h}  (points left in cell: ${pn.alive.length})`, ox, oy + s + 16, pn.done ? C.green : C.muted, '12px sans-serif'); });
      const n = P.length, cn = cN(n), eO = round ? sums[0] / round : NaN, eN = round ? sums[1] / round : NaN;
      const fin = round >= +q('#nt').value, p0 = pan[0], p1 = pan[1];
      caption(root, fin ? `✔ ${round} trees: the outlier needed ${AI.fmt(eO, 2)} cuts on average vs ${AI.fmt(eN, 2)} for the crowd point. Score s = 2^(−E[h]/c(n)) = ${AI.fmt(Math.pow(2, -eO / cn), 2)} vs ${AI.fmt(Math.pow(2, -eN / cn), 2)} — near 1 means anomaly.`
        : p0.done && p1.done ? `Tree ${round + 1}: outlier isolated in ${p0.h} cuts, crowd point in ${p1.h}. Recording and starting the next tree…`
        : `① Tree ${round + 1} of ${q('#nt').value}: pick a random axis, cut at a random position between the min and max of the points in the cell; keep the side containing the highlighted point. ② Repeat until it is alone. (Trees 3+ run faster.)`);
      q('#out').innerHTML = `<span class="m">trees done</span> ${round}/${q('#nt').value}   <span class="m">n</span> ${n}   <span class="m">c(n)</span> ${AI.fmt(cn, 2)}\n<span class="m">avg cuts, outlier</span> <b>${round ? AI.fmt(eO, 2) : '–'}</b>   <span class="m">score</span> ${round ? AI.fmt(Math.pow(2, -eO / cn), 3) : '–'}\n<span class="m">avg cuts, crowd pt</span> <b>${round ? AI.fmt(eN, 2) : '–'}</b>   <span class="m">score</span> ${round ? AI.fmt(Math.pow(2, -eN / cn), 3) : '–'}`;
    }, { fps: 30 });
  };
  /*__MORE__*/
})();
