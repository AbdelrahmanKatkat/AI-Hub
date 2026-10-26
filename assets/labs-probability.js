/* Module 3 · Probability — interactive labs.
   probdice, bayesdots, bayesupdate, randvar, distgallery, gauss2d, lln
   All animation is dt-based, slow by default, with speed sliders and pause/step. */
(function () {
  const AI = window.AI;
  const $ = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const T = (g, s, x, y, col, al, f) => { g.fillStyle = col; g.textAlign = al || 'left'; g.font = f || '11px ui-sans-serif,system-ui,sans-serif'; g.fillText(s, x, y); };
  const syncV = (root) => AI.$$('input[type=range]', root).forEach((r) => { const s = $(root, '#' + r.id + 'v'); if (s) s.textContent = r.dataset.fmt ? r.dataset.fmt.replace('#', r.value) : r.value; });
  const LF = [0]; for (let i = 1; i < 200; i++) LF[i] = LF[i - 1] + Math.log(i);
  const binomPmf = (n, k, p) => { if (k < 0 || k > n) return 0; if (p <= 0) return k === 0 ? 1 : 0; if (p >= 1) return k === n ? 1 : 0; return Math.exp(LF[n] - LF[k] - LF[n - k] + k * Math.log(p) + (n - k) * Math.log(1 - p)); };
  const poisPmf = (l, k) => (k < 0 ? 0 : Math.exp(-l + k * Math.log(l) - LF[k]));
  const normPdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const U = () => Math.random() * (1 - 2e-12) + 1e-12;
  const plot = (c, xmin, xmax, ymin, ymax, pad) => AI.view(c, xmin, xmax, ymin, ymax, pad || { l: 40, r: 12, t: 14, b: 26 });

  function pip(g, x, y, s, v, C) {
    g.fillStyle = '#f4f4f4'; g.strokeStyle = C.muted; g.lineWidth = 1.5;
    g.beginPath(); g.roundRect ? g.roundRect(x, y, s, s, s * 0.18) : g.rect(x, y, s, s); g.fill(); g.stroke();
    const P = { 1: [[1, 1]], 2: [[0, 0], [2, 2]], 3: [[0, 0], [1, 1], [2, 2]], 4: [[0, 0], [2, 0], [0, 2], [2, 2]], 5: [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]], 6: [[0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [2, 2]] };
    g.fillStyle = '#222';
    (P[v] || []).forEach(([i, j]) => { g.beginPath(); g.arc(x + s * (0.2 + 0.3 * i), y + s * (0.2 + 0.3 * j), s * 0.075, 0, 7); g.fill(); });
  }

  /* ============================== 1. Two dice: sample space ============================== */
  AI.labs.probdice = function (root) {
    const evs = {
      'sum = 7': (a, b) => a + b === 7, 'sum = 8': (a, b) => a + b === 8, 'sum ≥ 10': (a, b) => a + b >= 10, 'sum ≤ 4': (a, b) => a + b <= 4,
      'doubles': (a, b) => a === b, 'at least one 6': (a, b) => a === 6 || b === 6, 'first die is even': (a) => a % 2 === 0,
      'first die = 6': (a) => a === 6, 'sum is even': (a, b) => (a + b) % 2 === 0,
    };
    const names = Object.keys(evs);
    const opts = (extra) => extra.concat(names).map((n) => `<option>${n}</option>`).join('');
    const { view, ctl } = AI.labShell(root, 'Sample space of two dice', '36 equally likely outcomes — probability is counting cells',
      `<canvas id="cg" style="height:330px;cursor:pointer"></canvas><canvas id="cc" style="height:150px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Event A (blue)<select id="sa">${opts(['custom (click cells)'])}</select></label>
       <label>Event B (orange)<select id="sb">${opts(['(none)'])}</select></label>
       ${sl('rate', 'Rolls per second', 0.2, 30, 0.1, 0.8)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Roll once</button><button class="btn" id="b100">+100 now</button><button class="btn" id="reset">Clear</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s);
    $(root, '#sa').value = 'sum = 7'; $(root, '#sb').value = '(none)';
    const cg = AI.canvas(q('#cg')), cc = AI.canvas(q('#cc'));
    let custom = new Array(36).fill(false), A = [], B = [], kA = 0, kB = 0, kAB = 0;
    let counts = new Array(36).fill(0), seq = [], est = [], hits = 0, playing = true, acc = 0, roll = null, last = null, flash = 0, face = [1, 1], fl = 0;
    const set = (name) => { const f = evs[name]; return Array.from({ length: 36 }, (_, i) => (f ? f((i / 6 | 0) + 1, (i % 6) + 1) : false)); };
    const rebuild = () => {
      const na = q('#sa').value, nb = q('#sb').value;
      A = na.startsWith('custom') ? custom.slice() : set(na); B = set(nb);
      kA = A.filter(Boolean).length; kB = B.filter(Boolean).length; kAB = A.filter((v, i) => v && B[i]).length;
      est = []; hits = 0; seq.forEach((c, j) => { if (A[c]) hits++; est.push(hits / (j + 1)); });
    };
    const record = (a, b) => { const c = (a - 1) * 6 + (b - 1); counts[c]++; seq.push(c); if (A[c]) hits++; est.push(hits / seq.length); last = c; flash = 1; };
    const startRoll = (rate) => { const a = 1 + (Math.random() * 6 | 0), b = 1 + (Math.random() * 6 | 0); roll = { a, b, t: 0, dur: rate <= 3 ? 0.5 : 0 }; };
    q('#sa').onchange = q('#sb').onchange = rebuild;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { if (!roll) { startRoll(0.5); } };
    q('#b100').onclick = () => { for (let i = 0; i < 100; i++) record(1 + (Math.random() * 6 | 0), 1 + (Math.random() * 6 | 0)); };
    q('#reset').onclick = () => { counts.fill(0); seq = []; est = []; hits = 0; last = null; roll = null; };
    let L = null;
    cg.cv.addEventListener('click', (e) => {
      if (!L) return; const r = cg.cv.getBoundingClientRect(), x = e.clientX - r.left - L.x0, y = e.clientY - r.top - L.y0;
      const j = Math.floor(x / L.cs), i = Math.floor(y / L.cs); if (i < 0 || i > 5 || j < 0 || j > 5) return;
      if (!q('#sa').value.startsWith('custom')) { custom = A.slice(); q('#sa').selectedIndex = 0; }
      custom[i * 6 + j] = !custom[i * 6 + j]; rebuild();
    });
    rebuild();
    AI.loop(root, (dt) => {
      syncV(root);
      const rate = +q('#rate').value;
      if (playing || roll) {
        if (playing) acc = Math.min(acc + dt * rate, 2);
        if (roll) { roll.t += dt; fl += dt; if (roll.t >= roll.dur) { record(roll.a, roll.b); face = [roll.a, roll.b]; roll = null; } }
        let g = 0;
        while (playing && !roll && acc >= 1 && g++ < 40) { acc -= 1; startRoll(rate); if (!roll.dur) { record(roll.a, roll.b); face = [roll.a, roll.b]; roll = null; } }
      }
      flash = Math.max(0, flash - dt * 1.2);
      const C = AI.col(), g = cg.ctx, w = cg.w, h = cg.h;
      g.clearRect(0, 0, w, h);
      const gs = Math.max(120, Math.min(h - 50, w - 150)), cs = gs / 6, x0 = 34, y0 = 34;
      L = { x0, y0, cs };
      T(g, 'second die →', x0 + gs / 2, 14, C.muted, 'center'); g.save(); g.translate(12, y0 + gs / 2); g.rotate(-Math.PI / 2); T(g, 'first die →', 0, 0, C.muted, 'center'); g.restore();
      const mx = Math.max(1, ...counts);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
        const k = i * 6 + j, a = A[k], b = B[k], x = x0 + j * cs, y = y0 + i * cs;
        g.fillStyle = C.bg2; g.fillRect(x, y, cs, cs);
        if (a || b) { g.globalAlpha = 0.55; g.fillStyle = a && b ? C.green : a ? C.blue : C.orange; g.fillRect(x, y, cs, cs); g.globalAlpha = 1; }
        if (counts[k]) { g.globalAlpha = 0.5; g.fillStyle = C.fg; g.fillRect(x + 3, y + cs - 3 - (counts[k] / mx) * (cs * 0.28), cs - 6, 2.5); g.globalAlpha = 1; }
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
        T(g, String(i + 1 + j + 1), x + cs / 2, y + cs / 2 + 4, C.fg, 'center', `${Math.max(10, cs * 0.3)}px ui-monospace,monospace`);
        if (j === 0) T(g, String(i + 1), x0 - 8, y + cs / 2 + 4, C.muted, 'right');
        if (i === 0) T(g, String(j + 1), x + cs / 2, y0 - 5, C.muted, 'center');
      }
      T(g, 'number in a cell = sum of the two dice', x0, y0 + gs + 16, C.muted);
      if (last !== null && flash > 0) { const i = last / 6 | 0, j = last % 6; g.strokeStyle = C.yellow; g.lineWidth = 3 * flash + 1; g.strokeRect(x0 + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2); }
      const dx = x0 + gs + 24, ds = Math.min(50, cs * 1.1);
      let f1 = face[0], f2 = face[1];
      if (roll && roll.dur) { if (((fl * 14) | 0) % 2 === 0) { f1 = 1 + (Math.random() * 6 | 0); f2 = 1 + (Math.random() * 6 | 0); } else { f1 = (fl * 7 | 0) % 6 + 1; f2 = (fl * 11 | 0) % 6 + 1; } }
      pip(g, dx, y0, ds, f1, C); pip(g, dx, y0 + ds + 12, ds, f2, C);
      T(g, 'die 1', dx + ds + 6, y0 + ds / 2 + 4, C.muted); T(g, 'die 2', dx + ds + 6, y0 + ds * 1.5 + 16, C.muted);
      // convergence
      const g2 = cc.ctx, n = est.length, P = kA / 36;
      g2.clearRect(0, 0, cc.w, cc.h);
      const xm = Math.max(20, n * 1.05), v = plot(cc, 0, xm, 0, Math.min(1, Math.max(0.3, P * 2 + 0.1)), { l: 40, r: 12, t: 16, b: 24 });
      AI.axes(g2, v);
      g2.strokeStyle = C.green; g2.lineWidth = 2; g2.setLineDash([6, 4]); g2.beginPath(); g2.moveTo(v.X(0), v.Y(P)); g2.lineTo(v.X(xm), v.Y(P)); g2.stroke(); g2.setLineDash([]);
      g2.strokeStyle = C.blue; g2.lineWidth = 2; g2.beginPath();
      const stp = Math.max(1, Math.floor(n / 600));
      for (let i = 0; i < n; i += stp) { const px = v.X(i + 1), py = v.Y(est[i]); i ? g2.lineTo(px, py) : g2.moveTo(px, py); }
      if (n) g2.lineTo(v.X(n), v.Y(est[n - 1])); g2.stroke();
      T(g2, 'fraction of rolls in A (blue)  vs  theory k/36 (green dashed)   x-axis: number of rolls', v.pad.l + 2, 11, C.muted);
      // readout
      const pa = kA / 36, pb = kB / 36, pab = kAB / 36;
      let verdict = '';
      if (kB) verdict = kAB === 0 && kA && kB ? '<span class="o">mutually exclusive (A∩B = ∅) → these are NOT independent</span>' : kAB * 36 === kA * kB ? '<span class="g">independent: P(A∩B) = P(A)P(B)</span>' : '<span class="r">dependent: P(A∩B) ≠ P(A)P(B)</span>';
      q('#out').innerHTML = `|A| = ${kA}   P(A) = ${kA}/36 = <b>${AI.fmt(pa, 4)}</b>\n` + (kB ? `|B| = ${kB}   P(B) = ${kB}/36 = ${AI.fmt(pb, 4)}\n|A∩B| = ${kAB}  P(A∩B) = ${AI.fmt(pab, 4)}\nP(A)·P(B) = ${AI.fmt(pa * pb, 4)}\n${verdict}\n` : '') + `\nrolls ${n}   in A: ${hits}\nestimate <b>${n ? AI.fmt(hits / n, 4) : '–'}</b>   error ${n ? AI.fmt(Math.abs(hits / n - pa), 4) : '–'}`;
      q('#cap').textContent = roll ? `Rolling… the two dice are a random pick of one of the 36 cells.` : last !== null ? `Last roll: (${face[0]}, ${face[1]}) → sum ${face[0] + face[1]}. ${A[last] ? 'It falls in A.' : 'Not in A.'}  Click grid cells to build your own event.` : 'Press play. Each roll lights up one cell; the running fraction (blue) drifts toward the counted probability (green).';
    }, { fps: 30 });
  };

  /* ============================== 2. Natural-frequency dots (Bayes) ============================== */
  AI.labs.bayesdots = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bayes with 1000 people', 'split by truth, then by test result — count the ringed dots',
      `<canvas id="cv" style="height:420px"></canvas><div class="caption" id="cap"></div>`,
      `${sl('prev', 'Prevalence P(sick) %', 0.5, 60, 0.5, 1)}${sl('sens', 'Sensitivity P(+ | sick) %', 50, 100, 1, 90)}${sl('fpr', 'False-positive rate P(+ | healthy) %', 0, 50, 1, 10)}
       ${sl('spd', 'Speed ×', 0.25, 3, 0.25, 1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next stage ›</button><button class="btn" id="reset">Restart</button></div>
       <div class="chips"><button class="chip" id="p1">rare disease</button><button class="chip" id="p2">common (30%)</button><button class="chip" id="p3">perfect-ish test</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s);
    const c = AI.canvas(q('#cv'));
    const N = 1000, rnd = AI.rng(11);
    const perm = Array.from({ length: N }, (_, i) => i); for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const X = new Float32Array(N), Y = new Float32Array(N), TX = new Float32Array(N), TY = new Float32Array(N), K = new Float32Array(N).map(() => 0.7 + rnd() * 0.6);
    let stage = 0, playing = true, tmr = 0, key = '', ct = 0, ring = 0, fade = 0, S = 0, TP = 0, FP = 0, lbl = [], dirty = true, init = false, sp = 8;
    const cat = (i) => (i < TP ? 0 : i < S ? 1 : i < S + FP ? 2 : 3);
    const caps = ['① 1000 people, nobody sorted yet. Hold on to your guess for the answer.', '② Sort by the truth: the sick (red) go left, the healthy (blue) right.', '③ Run the test. Everyone who tests positive gets a yellow ring: sick people with probability = sensitivity, healthy people with probability = false-positive rate.', '④ Ignore everybody without a ring. Of the ringed dots, what fraction is red? That fraction is P(sick | positive).'];
    const preset = (a, b, d) => { q('#prev').value = a; q('#sens').value = b; q('#fpr').value = d; stage = 0; tmr = 0; };
    q('#p1').onclick = () => preset(1, 90, 10); q('#p2').onclick = () => preset(30, 90, 10); q('#p3').onclick = () => preset(1, 99, 1);
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = () => { stage = Math.min(3, stage + 1); tmr = 0; };
    q('#reset').onclick = () => { stage = 0; tmr = 0; ct = 0; ring = 0; fade = 0; init = false; };
    const layout = () => {
      const W = c.w, H = c.h, prev = +q('#prev').value / 100, se = +q('#sens').value / 100, fp = +q('#fpr').value / 100;
      S = Math.round(N * prev); TP = Math.round(S * se); FP = Math.round((N - S) * fp);
      const bw = (W - 30) / 2, bh = H - 70; let s = Math.sqrt(bw * bh / N);
      for (; s > 4; s -= 0.25) { const cols = Math.floor(bw / s); if ((Math.ceil(N / cols) + 5) * s <= bh) break; }
      sp = s; lbl = [];
      const cols = Math.floor(bw / s), cols0 = Math.floor((W - 20) / s), top = 44, xs = 10, xh = W / 2 + 5;
      const put = (i, x0, y0, j, cl) => { TX[i] = x0 + (j % cl + 0.5) * s; TY[i] = y0 + (Math.floor(j / cl) + 0.5) * s; };
      const grp = (ids, x0, y0, labels, name, c0) => { // stage 2/3: label row then dots, per group
        let row = 0;
        ids.forEach((g, gi) => {
          lbl.push({ x: x0, y: y0 + (row + 0.75) * s, t: g.t });
          row++;
          g.list.forEach((i, j) => { TX[i] = x0 + (j % cols + 0.5) * s; TY[i] = y0 + (row + Math.floor(j / cols) + 0.5) * s; });
          row += Math.ceil(g.list.length / cols) + 0.4;
        });
      };
      const L0 = [], L1 = [], L2 = [], L3 = [];
      for (let i = 0; i < N; i++) [L0, L1, L2, L3][cat(i)].push(i);
      for (let i = 0; i < N; i++) {
        if (stage === 0) { const j = perm[i], cl = cols0, w0 = cl * s; TX[i] = (W - w0) / 2 + (j % cl + 0.5) * s; TY[i] = top + (Math.floor(j / cl) + 0.5) * s; }
      }
      if (stage === 1) { L0.concat(L1).forEach((i, j) => put(i, xs, top, j, cols)); L2.concat(L3).forEach((i, j) => put(i, xh, top, j, cols)); }
      if (stage >= 2) {
        grp([{ list: L0, t: `tested +: ${TP}` }, { list: L1, t: `tested −: ${S - TP}` }], xs, top);
        grp([{ list: L2, t: `tested +: ${FP}` }, { list: L3, t: `tested −: ${N - S - FP}` }], xh, top);
      }
      if (!init) { for (let i = 0; i < N; i++) { X[i] = W / 2; Y[i] = H / 2; } init = true; }
      dirty = true;
    };
    AI.loop(root, (dt) => {
      syncV(root);
      const spd = +q('#spd').value, k = `${q('#prev').value}|${q('#sens').value}|${q('#fpr').value}|${stage}|${c.w}|${c.h}`;
      if (playing && stage < 3) { tmr += dt * spd; if (tmr > (stage === 0 ? 2.2 : 4.5)) { stage++; tmr = 0; } }
      if (k !== key) { key = k; layout(); }
      let moving = false; const e = 1 - Math.exp(-dt * 2.2 * Math.min(spd, 1.5));
      for (let i = 0; i < N; i++) { const ee = Math.min(1, e * K[i]); const dx = TX[i] - X[i], dy = TY[i] - Y[i]; if (dx * dx + dy * dy > 0.01) { X[i] += dx * ee; Y[i] += dy * ee; moving = true; } }
      const tc = stage >= 1 ? 1 : 0, tr = stage >= 2 ? 1 : 0, tf = stage >= 3 ? 1 : 0, ea = 1 - Math.exp(-dt * 2.5 * Math.min(spd, 1.5));
      if (Math.abs(tc - ct) + Math.abs(tr - ring) + Math.abs(tf - fade) > 0.002) { ct += (tc - ct) * ea; ring += (tr - ring) * ea; fade += (tf - fade) * ea; moving = true; }
      if (!moving && !dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, r = Math.max(1.2, sp * 0.36);
      g.clearRect(0, 0, W, H);
      if (stage >= 1) {
        g.globalAlpha = Math.min(1, ct);
        const bw = (W - 30) / 2; g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(10, 38, bw, H - 62); g.strokeRect(W / 2 + 5, 38, bw, H - 62);
        T(g, `SICK  ${S}`, 14, 30, C.red, 'left', 'bold 13px ui-sans-serif,system-ui,sans-serif'); T(g, `HEALTHY  ${N - S}`, W / 2 + 9, 30, C.blue, 'left', 'bold 13px ui-sans-serif,system-ui,sans-serif');
        g.globalAlpha = 1;
      } else T(g, '1000 people', W / 2, 30, C.muted, 'center', 'bold 13px ui-sans-serif,system-ui,sans-serif');
      const colS = AI.mix(C.muted, C.red, ct), colH = AI.mix(C.muted, C.blue, ct);
      for (let pass = 0; pass < 2; pass++) for (let i = 0; i < N; i++) {
        const ct_ = cat(i), pos = ct_ === 0 || ct_ === 2, sick = ct_ < 2;
        if ((pass === 0) === (ring > 0.3 && pos)) continue;
        g.globalAlpha = 1 - (pos ? 0 : 0.7 * fade); g.fillStyle = sick ? colS : colH;
        g.beginPath(); g.arc(X[i], Y[i], r, 0, 6.2832); g.fill();
        if (pos && ring > 0.01) { g.globalAlpha = ring; g.strokeStyle = C.yellow; g.lineWidth = Math.max(1, r * 0.45); g.beginPath(); g.arc(X[i], Y[i], r + 1, 0, 6.2832); g.stroke(); }
      }
      g.globalAlpha = 1;
      if (stage >= 2) { g.globalAlpha = ring; lbl.forEach((l) => T(g, l.t, l.x + 4, l.y, C.fg, 'left', '12px ui-sans-serif,system-ui,sans-serif')); g.globalAlpha = 1; }
      q('#cap').textContent = caps[stage];
      const pp = TP + FP ? TP / (TP + FP) : NaN;
      q('#out').innerHTML = `sick       S  = ${S}    healthy ${N - S}\ntrue +     TP = ${TP}   (${AI.fmt(+q('#sens').value, 0)}% of ${S})\nfalse +    FP = ${FP}   (${AI.fmt(+q('#fpr').value, 0)}% of ${N - S})\n<span class="m">ringed total = ${TP + FP}</span>\n\nP(sick | +) = TP/(TP+FP)\n            = ${TP}/${TP + FP} = <b class="o">${AI.fmt(pp * 100, 1)} %</b>\nP(+ | sick) = <span class="b">${q('#sens').value} %</span>  ← not the same!`;
    }, { fps: 30 });
  };

  /* ============================== 3. Bayes as sequential updating ============================== */
  AI.labs.bayesupdate = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bayes as updating belief', 'every new test multiplies your odds by a likelihood ratio',
      `<canvas id="cb" style="height:70px"></canvas><canvas id="ch" style="height:200px"></canvas><div class="caption" id="cap"></div>`,
      `${sl('prior', 'Prior P(sick) %', 1, 90, 1, 5)}${sl('sens', 'Sensitivity P(+ | sick) %', 50, 99, 1, 90)}${sl('fpr', 'False-positive rate P(+ | healthy) %', 1, 50, 1, 10)}
       <label>Hidden truth (used by auto-tests)<select id="truth"><option value="1">patient is sick</option><option value="0">patient is healthy</option></select></label>
       <div class="row"><button class="btn" id="pos">Test +</button><button class="btn" id="neg">Test −</button></div>
       <div class="row"><button class="btn primary" id="auto">▶ Auto-test</button><button class="btn" id="reset">Reset</button></div>
       ${sl('spd', 'Auto speed ×', 0.25, 3, 0.25, 1)}
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s), cb = AI.canvas(q('#cb')), ch = AI.canvas(q('#ch'));
    let hist = [], ev = [], disp = null, auto = false, tmr = 0, key = '', msg = '', np = 0, nn = 0;
    const par = () => [+q('#prior').value / 100, +q('#sens').value / 100, +q('#fpr').value / 100];
    const reset = () => { hist = [par()[0]]; ev = []; disp = hist[0]; np = nn = 0; msg = 'Start: before any test, the belief is the prior.'; };
    const add = (pos) => {
      if (hist.length > 14) return; const [, se, fp] = par(), p = hist[hist.length - 1];
      const l1 = pos ? se : 1 - se, l0 = pos ? fp : 1 - fp, post = (p * l1) / (p * l1 + (1 - p) * l0);
      hist.push(post); ev.push(pos); pos ? np++ : nn++;
      msg = `Test ${ev.length} came out ${pos ? 'POSITIVE' : 'NEGATIVE'}: likelihood ratio = ${AI.fmt(l1, 2)}/${AI.fmt(l0, 2)} = ${AI.fmt(l1 / l0, 3)}.  Odds ${AI.fmt(p / (1 - p), 3)} × ${AI.fmt(l1 / l0, 3)} = ${AI.fmt(post / (1 - post), 3)}  →  P(sick) = ${AI.fmt(post * 100, 1)} %`;
    };
    q('#pos').onclick = () => add(true); q('#neg').onclick = () => add(false); q('#reset').onclick = reset;
    q('#auto').onclick = () => { auto = !auto; q('#auto').textContent = auto ? '⏸ Stop auto' : '▶ Auto-test'; };
    reset();
    AI.loop(root, (dt) => {
      syncV(root);
      const k = par().join('|'); if (k !== key) { key = k; reset(); }
      if (auto) { tmr += dt * +q('#spd').value; if (tmr > 2.2) { tmr = 0; const [, se, fp] = par(); add(Math.random() < (q('#truth').value === '1' ? se : fp)); } }
      const p = hist[hist.length - 1]; disp += (p - disp) * (1 - Math.exp(-dt * 3));
      const C = AI.col(), g = cb.ctx, bx = 10, bw = cb.w - 20;
      g.clearRect(0, 0, cb.w, cb.h);
      g.fillStyle = C.blue; g.globalAlpha = 0.7; g.fillRect(bx, 22, bw, 26); g.globalAlpha = 1; g.fillStyle = C.red; g.fillRect(bx, 22, bw * disp, 26);
      T(g, `belief: sick ${AI.fmt(disp * 100, 1)} %`, bx + 6, 40, '#fff', 'left', 'bold 12px ui-sans-serif,system-ui,sans-serif'); T(g, `healthy ${AI.fmt((1 - disp) * 100, 1)} %`, bx + bw - 6, 40, '#fff', 'right', 'bold 12px ui-sans-serif,system-ui,sans-serif');
      T(g, 'how strongly we believe "sick" right now', bx, 14, C.muted);
      const g2 = ch.ctx; g2.clearRect(0, 0, ch.w, ch.h);
      const nmax = Math.max(8, hist.length), v = plot(ch, 0, nmax, 0, 1, { l: 40, r: 14, t: 16, b: 26 });
      AI.axes(g2, v); T(g2, 'P(sick) after each test  (red dot: +, blue dot: −)', v.pad.l + 2, 11, C.muted);
      g2.strokeStyle = C.orange; g2.lineWidth = 2.5; g2.beginPath();
      hist.forEach((h, i) => { const y = i === hist.length - 1 ? disp : h; i ? g2.lineTo(v.X(i), v.Y(y)) : g2.moveTo(v.X(i), v.Y(y)); }); g2.stroke();
      hist.forEach((h, i) => { const y = i === hist.length - 1 ? disp : h; g2.fillStyle = i === 0 ? C.muted : ev[i - 1] ? C.red : C.blue; g2.beginPath(); g2.arc(v.X(i), v.Y(y), 4.5, 0, 7); g2.fill(); });
      const [pr, se, fp] = par(), o = p / (1 - p);
      q('#out').innerHTML = `prior odds       ${AI.fmt(pr / (1 - pr), 3)}\nLR(+) = sens/fpr  <b>${AI.fmt(se / fp, 2)}</b>\nLR(−) = (1−sens)/(1−fpr)  <b>${AI.fmt((1 - se) / (1 - fp), 3)}</b>\ntests so far     ${np} + , ${nn} −\nposterior odds   <b>${AI.fmt(o, 3)}</b>\nP(sick | data)   <b class="o">${AI.fmt(p * 100, 2)} %</b>\n<span class="m">(assumes each test errs independently)</span>`;
      q('#cap').textContent = msg;
    }, { fps: 30 });
  };

  /* ============================== 4. Random variables: PMF/PDF/CDF, E[X], Var ============================== */
  AI.labs.randvar = function (root) {
    const dfn = (vals, p) => { const m = vals.reduce((s, x, i) => s + x * p[i], 0); return { disc: true, vals, p, mean: m, v: vals.reduce((s, x, i) => s + (x - m) ** 2 * p[i], 0), lo: vals[0] - 1, hi: vals[vals.length - 1] + 1 }; };
    const V = {
      'Fair die': { par: null, make: () => dfn([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1].map((x) => x / 6)) },
      'Loaded die (P(6) slider)': { par: ['P(6)', 0.05, 0.9, 0.01, 0.4], make: (a) => dfn([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5].map(() => (1 - a) / 5).concat([a])) },
      'Sum of two dice': { par: null, make: () => dfn([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1].map((x) => x / 36)) },
      'Uniform(0, width)': { par: ['width', 0.5, 4, 0.1, 2], make: (a) => ({ disc: false, lo: -0.2, hi: 4.4, pdf: (x) => (x >= 0 && x <= a ? 1 / a : 0), cdf: (x) => AI.clamp(x / a, 0, 1), mean: a / 2, v: a * a / 12, sample: () => Math.random() * a }) },
      'Exponential(rate λ)': { par: ['rate λ', 0.3, 3, 0.1, 1], make: (a) => ({ disc: false, lo: -0.3, hi: 8, pdf: (x) => (x >= 0 ? a * Math.exp(-a * x) : 0), cdf: (x) => (x >= 0 ? 1 - Math.exp(-a * x) : 0), mean: 1 / a, v: 1 / (a * a), sample: () => -Math.log(U()) / a }) },
    };
    const names = Object.keys(V);
    const { view, ctl } = AI.labShell(root, 'Random variable explorer', 'distribution, balance point, spread, and the long-run average',
      `<canvas id="ca" style="height:210px"></canvas><canvas id="cf" style="height:110px"></canvas><canvas id="cr" style="height:130px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Random variable X<select id="sel">${names.map((n) => `<option>${n}</option>`).join('')}</select></label>
       <label id="plab"><span id="pn">parameter</span> <span class="v" id="pv"></span><input type="range" id="par" min="0" max="1" step="0.01" value="0.5"></label>
       ${sl('rate', 'Samples per second', 1, 200, 1, 3)}${sl('fall', 'Fall speed ×', 0.25, 3, 0.25, 1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Draw one</button><button class="btn" id="reset">Clear</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s), ca = AI.canvas(q('#ca')), cf = AI.canvas(q('#cf')), cr = AI.canvas(q('#cr'));
    let X = null, key = '', samples = [], run = [], sum = 0, drops = [], bins = [], playing = true, acc = 0, hmax = 0;
    const BN = 32;
    const sampleOf = () => { if (X.disc) { let u = Math.random(), i = 0; while (i < X.p.length - 1 && u > X.p[i]) { u -= X.p[i]; i++; } return X.vals[i]; } return X.sample(); };
    const clear = () => { samples = []; run = []; sum = 0; drops = []; bins = new Array(X.disc ? X.vals.length : BN).fill(0); };
    const land = (x) => {
      samples.push(x); sum += x; if (run.length < 20000) run.push(sum / samples.length);
      if (X.disc) { const i = X.vals.indexOf(x); if (i >= 0) bins[i]++; } else { const b = Math.floor(((x - X.lo) / (X.hi - X.lo)) * BN); if (b >= 0 && b < BN) bins[b]++; }
    };
    const spawn = () => { const x = sampleOf(), f = +q('#fall').value; if (+q('#rate').value > 25) land(x); else drops.push({ x, t: 0, dur: 1.1 / f }); };
    const rebuild = () => { const nm = q('#sel').value, d = V[nm], pl = q('#plab'); if (d.par) { pl.style.display = ''; $(root, '#pn').textContent = d.par[0]; const r = q('#par'); if (key.split('|')[0] !== nm) { r.min = d.par[1]; r.max = d.par[2]; r.step = d.par[3]; r.value = d.par[4]; } } else pl.style.display = 'none'; X = d.make(d.par ? +q('#par').value : 0); clear(); key = nm + '|' + q('#par').value; };
    q('#sel').onchange = () => { key = ''; rebuild(); }; q('#par').oninput = rebuild;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = spawn; q('#reset').onclick = clear;
    rebuild();
    AI.loop(root, (dt) => {
      syncV(root); q('#pv').textContent = q('#par').value;
      if (playing) { acc = Math.min(acc + dt * +q('#rate').value, 60); let g = 0; while (acc >= 1 && g++ < 40) { acc--; spawn(); } }
      for (const d of drops) d.t += dt; for (let i = drops.length - 1; i >= 0; i--) if (drops[i].t >= drops[i].dur) { land(drops[i].x); drops.splice(i, 1); }
      const C = AI.col(), n = samples.length, g = ca.ctx;
      g.clearRect(0, 0, ca.w, ca.h);
      let ymax;
      if (X.disc) ymax = Math.max(...X.p, ...bins.map((b) => (n ? b / n : 0))) * 1.25; else { ymax = 0; for (let x = X.lo; x <= X.hi; x += (X.hi - X.lo) / 100) ymax = Math.max(ymax, X.pdf(x)); ymax *= 1.3; const bw = (X.hi - X.lo) / BN; if (n) ymax = Math.max(ymax, Math.max(...bins) / (n * bw) * 1.1); }
      const v = plot(ca, X.lo, X.hi, 0, ymax, { l: 44, r: 12, t: 18, b: 28 });
      const sd = Math.sqrt(X.v);
      g.fillStyle = C.purple; g.globalAlpha = 0.14; g.fillRect(v.X(X.mean - sd), v.pad.t, v.X(X.mean + sd) - v.X(X.mean - sd), v.ph); g.globalAlpha = 1;
      AI.axes(g, v);
      g.fillStyle = C.blue; g.strokeStyle = C.blue; g.lineWidth = 2.5;
      if (X.disc) { const bw = v.pw / (X.hi - X.lo) * 0.7; X.vals.forEach((x, i) => { g.globalAlpha = 0.8; g.fillRect(v.X(x) - bw / 2, v.Y(X.p[i]), bw, v.Y(0) - v.Y(X.p[i])); g.globalAlpha = 1; if (n) { g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(v.X(x) - bw / 2 - 2, v.Y(bins[i] / n), bw + 4, v.Y(0) - v.Y(bins[i] / n)); } }); }
      else {
        const bw = (X.hi - X.lo) / BN;
        if (n) { g.fillStyle = C.orange; g.globalAlpha = 0.35; bins.forEach((b, i) => { const hh = (b / (n * bw)); g.fillRect(v.X(X.lo + i * bw) + 1, v.Y(hh), v.pw / BN - 2, v.Y(0) - v.Y(hh)); }); g.globalAlpha = 1; }
        g.beginPath(); for (let i = 0; i <= 300; i++) { const x = X.lo + ((X.hi - X.lo) * i) / 300, y = X.pdf(x); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke();
      }
      // fulcrum at the mean
      const mx = v.X(X.mean), by = v.Y(0);
      g.fillStyle = C.red; g.beginPath(); g.moveTo(mx, by + 2); g.lineTo(mx - 8, by + 17); g.lineTo(mx + 8, by + 17); g.closePath(); g.fill();
      g.strokeStyle = C.red; g.lineWidth = 1.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(mx, by); g.lineTo(mx, v.pad.t + 4); g.stroke(); g.setLineDash([]);
      T(g, `E[X] = ${AI.fmt(X.mean, 3)} (balance point)`, mx + 6, v.pad.t + 10, C.red);
      T(g, `purple band: mean ± σ`, v.pad.l + 4, 11, C.purple);
      // falling drops
      for (const d of drops) { const k = d.t / d.dur; g.fillStyle = C.fg; g.globalAlpha = 0.9; g.beginPath(); g.arc(v.X(d.x), v.pad.t + k * k * (v.ph - 6) + 3, 3.5, 0, 7); g.fill(); g.globalAlpha = 1; }
      // CDF
      const g2 = cf.ctx; g2.clearRect(0, 0, cf.w, cf.h);
      const v2 = plot(cf, X.lo, X.hi, 0, 1, { l: 44, r: 12, t: 12, b: 22 }); AI.axes(g2, v2);
      const F = (x) => (X.disc ? X.vals.reduce((s, xv, i) => s + (xv <= x ? X.p[i] : 0), 0) : X.cdf(x));
      g2.strokeStyle = C.blue; g2.lineWidth = 2.5; g2.beginPath();
      for (let i = 0; i <= 300; i++) { const x = X.lo + ((X.hi - X.lo) * i) / 300, y = F(x); i ? g2.lineTo(v2.X(x), v2.Y(y)) : g2.moveTo(v2.X(x), v2.Y(y)); } g2.stroke();
      if (n) { const s = samples.slice(-4000).sort((a, b) => a - b); g2.strokeStyle = C.orange; g2.lineWidth = 1.8; g2.beginPath(); for (let i = 0; i <= 160; i++) { const x = X.lo + ((X.hi - X.lo) * i) / 160; let lo = 0, hi = s.length; while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] <= x) lo = m + 1; else hi = m; } const y = lo / s.length; i ? g2.lineTo(v2.X(x), v2.Y(y)) : g2.moveTo(v2.X(x), v2.Y(y)); } g2.stroke(); }
      T(g2, 'CDF  F(x) = P(X ≤ x): blue theory, orange from samples', v2.pad.l + 2, 9, C.muted);
      // running mean
      const g3 = cr.ctx; g3.clearRect(0, 0, cr.w, cr.h);
      const rn = run.length, xm = Math.max(30, rn * 1.05), span = Math.max(sd * 2.2, 0.5), v3 = plot(cr, 0, xm, X.mean - span, X.mean + span, { l: 44, r: 12, t: 14, b: 22 });
      AI.axes(g3, v3);
      g3.strokeStyle = C.red; g3.lineWidth = 2; g3.setLineDash([6, 4]); g3.beginPath(); g3.moveTo(v3.X(0), v3.Y(X.mean)); g3.lineTo(v3.X(xm), v3.Y(X.mean)); g3.stroke(); g3.setLineDash([]);
      g3.strokeStyle = C.orange; g3.lineWidth = 2; g3.beginPath(); const st = Math.max(1, Math.floor(rn / 500));
      for (let i = 0; i < rn; i += st) { const px = v3.X(i + 1), py = AI.clamp(v3.Y(run[i]), v3.pad.t, v3.pad.t + v3.ph); i ? g3.lineTo(px, py) : g3.moveTo(px, py); } g3.stroke();
      T(g3, 'running average of the samples (orange) → E[X] (red dashed)', v3.pad.l + 2, 9, C.muted);
      let m = n ? sum / n : NaN, vs = 0; if (n > 1) { for (const s of samples) vs += (s - m) ** 2; vs /= n; }
      q('#out').innerHTML = `E[X]   = <b>${AI.fmt(X.mean, 4)}</b>   sample mean <b class="o">${AI.fmt(m, 4)}</b>\nVar(X) = <b>${AI.fmt(X.v, 4)}</b>   sample var  <b class="o">${AI.fmt(vs, 4)}</b>\nσ      = ${AI.fmt(sd, 4)}\nsamples n = ${n}`;
      q('#cap').textContent = X.disc ? 'Blue bars: P(X = x). Orange outlines: how often each value actually occurred. The red triangle is where the bars would balance on a seesaw.' : 'Blue curve: density f(x); area under it between two points is a probability. Orange: histogram of samples.';
    }, { fps: 30 });
  };

  /* ============================== 5. Distribution gallery (with Galton board) ============================== */
  AI.labs.distgallery = function (root) {
    const catP = (P) => { const s = P.reduce((a, b) => a + b, 0) || 1; return P.map((x) => x / s); };
    const D = {
      'Bernoulli(p)': { disc: true, par: [['p', 0, 1, 0.01, 0.3]], rng: () => [0, 1], pmf: (P, k) => (k === 1 ? P[0] : k === 0 ? 1 - P[0] : 0), mean: (P) => P[0], v: (P) => P[0] * (1 - P[0]), sample: (P) => (Math.random() < P[0] ? 1 : 0), info: 'One yes/no trial. A sigmoid output unit = a Bernoulli p.' },
      'Binomial(n, p)': { disc: true, par: [['n', 1, 20, 1, 10], ['p', 0, 1, 0.01, 0.5]], rng: (P) => [0, P[0]], pmf: (P, k) => binomPmf(P[0], k, P[1]), mean: (P) => P[0] * P[1], v: (P) => P[0] * P[1] * (1 - P[1]), sample: (P) => { let s = 0; for (let i = 0; i < P[0]; i++) s += Math.random() < P[1] ? 1 : 0; return s; }, info: 'Number of successes in n independent trials: the Galton board.' },
      'Categorical (4 classes)': { disc: true, par: [['weight class 0', 0, 10, 1, 5], ['weight class 1', 0, 10, 1, 3], ['weight class 2', 0, 10, 1, 1], ['weight class 3', 0, 10, 1, 1]], rng: () => [0, 3], pmf: (P, k) => (k >= 0 && k <= 3 ? catP(P)[k] : 0), mean: (P) => catP(P).reduce((s, p, i) => s + p * i, 0), v: (P) => { const p = catP(P), m = p.reduce((s, x, i) => s + x * i, 0); return p.reduce((s, x, i) => s + x * (i - m) ** 2, 0); }, sample: (P) => { const p = catP(P); let u = Math.random(), i = 0; while (i < 3 && u > p[i]) { u -= p[i]; i++; } return i; }, info: 'A softmax layer outputs this: one probability per class (weights are normalised to sum to 1).' },
      'Uniform(a, a+w)': { disc: false, par: [['a', -3, 2, 0.1, -1], ['width w', 0.5, 6, 0.1, 2]], view: [-5, 7], pdf: (P, x) => (x >= P[0] && x <= P[0] + P[1] ? 1 / P[1] : 0), mean: (P) => P[0] + P[1] / 2, v: (P) => P[1] ** 2 / 12, sample: (P) => P[0] + Math.random() * P[1], info: 'Every value in an interval equally likely. Used for weight initialisation.' },
      'Gaussian(μ, σ)': { disc: false, par: [['mean μ', -3, 3, 0.1, 0], ['std σ', 0.3, 2.5, 0.05, 1]], view: [-8, 8], pdf: (P, x) => normPdf(x, P[0], P[1]), mean: (P) => P[0], v: (P) => P[1] ** 2, sample: (P) => P[0] + P[1] * AI.randn(), info: 'The bell curve. Noise model behind squared error; weight priors.' },
      'Exponential(rate λ)': { disc: false, par: [['rate λ', 0.2, 3, 0.1, 1]], view: [-0.5, 8], pdf: (P, x) => (x >= 0 ? P[0] * Math.exp(-P[0] * x) : 0), mean: (P) => 1 / P[0], v: (P) => 1 / P[0] ** 2, sample: (P) => -Math.log(U()) / P[0], info: 'Waiting time until the next event; memoryless.' },
      'Poisson(rate λ)': { disc: true, par: [['rate λ', 0.5, 15, 0.5, 4]], rng: (P) => [0, Math.ceil(P[0] + 5 * Math.sqrt(P[0]) + 4)], pmf: (P, k) => poisPmf(P[0], k), mean: (P) => P[0], v: (P) => P[0], sample: (P) => { const L = Math.exp(-P[0]); let k = 0, p = 1; do { k++; p *= Math.random(); } while (p > L); return k - 1; }, info: 'Number of events in a fixed window (counts: clicks, arrivals, mutations).' },
    };
    const names = Object.keys(D);
    const { view, ctl } = AI.labShell(root, 'Distribution gallery', 'pick a distribution, move its knobs, watch samples pile up on the true shape',
      `<canvas id="cm" style="height:250px"></canvas><canvas id="cg" style="height:280px;display:none"></canvas><div class="caption" id="cap"></div>`,
      `<label>Distribution<select id="sel">${names.map((n) => `<option>${n}</option>`).join('')}</select></label>
       ${[0, 1, 2, 3].map((i) => `<label id="l${i}"><span id="n${i}"></span> <span class="v" id="v${i}"></span><input type="range" id="p${i}" min="0" max="1" step="0.01" value="0"></label>`).join('')}
       ${sl('rate', 'Samples per second', 1, 200, 1, 4)}${sl('fall', 'Fall speed ×', 0.25, 3, 0.25, 1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Draw one</button><button class="btn" id="reset">Clear</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s), cm = AI.canvas(q('#cm')), cgb = AI.canvas(q('#cg'));
    let d = null, P = [], counts = [], n = 0, sum = 0, sum2 = 0, drops = [], balls = [], playing = true, acc = 0, ckey = '';
    const BN = 36;
    const reset = () => { counts = new Array(d.disc ? d.rng(P)[1] - d.rng(P)[0] + 1 : BN).fill(0); n = 0; sum = sum2 = 0; drops = []; balls = []; };
    const pick = () => { d = D[q('#sel').value]; d.par.forEach((p, i) => { const r = q('#p' + i); q('#l' + i).style.display = ''; $(root, '#n' + i).textContent = p[0]; r.min = p[1]; r.max = p[2]; r.step = p[3]; r.value = p[4]; }); for (let i = d.par.length; i < 4; i++) q('#l' + i).style.display = 'none'; readP(); reset(); };
    const readP = () => { P = d.par.map((_, i) => +q('#p' + i).value); };
    const land = (x) => { n++; sum += x; sum2 += x * x; if (d.disc) { const k = x - d.rng(P)[0]; if (k >= 0 && k < counts.length) counts[k]++; } else { const [lo, hi] = d.view, b = Math.floor(((x - lo) / (hi - lo)) * BN); if (b >= 0 && b < BN) counts[b]++; } };
    const isBin = () => q('#sel').value.startsWith('Binomial');
    const spawn = () => {
      const f = +q('#fall').value;
      if (isBin()) { const nn = P[0], path = Array.from({ length: nn }, () => (Math.random() < P[1] ? 1 : 0)); balls.push({ t: 0, path }); return; }
      const x = d.sample(P); if (+q('#rate').value > 25) land(x); else drops.push({ x, t: 0, dur: 1.1 / f });
    };
    q('#sel').onchange = () => { pick(); cgb.cv.style.display = isBin() ? '' : 'none'; };
    AI.$$('input[type=range]', root).forEach((r) => { if (/^p\d$/.test(r.id)) r.oninput = () => { readP(); reset(); }; });
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#step').onclick = spawn; q('#reset').onclick = reset;
    pick();
    AI.loop(root, (dt) => {
      syncV(root); d.par.forEach((p, i) => (q('#v' + i).textContent = q('#p' + i).value));
      const f = +q('#fall').value, rate = +q('#rate').value;
      if (playing) { acc = Math.min(acc + dt * rate, 60); let g = 0; while (acc >= 1 && g++ < 40) { acc--; if (isBin() && rate > 25) { land(d.sample(P)); } else spawn(); } }
      for (const o of drops) o.t += dt; for (let i = drops.length - 1; i >= 0; i--) if (drops[i].t >= drops[i].dur) { land(drops[i].x); drops.splice(i, 1); }
      const nrow = isBin() ? P[0] : 0, Trow = 0.3 / f;
      for (let i = balls.length - 1; i >= 0; i--) { balls[i].t += dt; if (balls[i].t >= nrow * Trow + 0.45 / f) { land(balls[i].path.reduce((a, b) => a + b, 0)); balls.splice(i, 1); } }
      const C = AI.col(), g = cm.ctx; g.clearRect(0, 0, cm.w, cm.h);
      let lo, hi, yv, ymax = 0;
      if (d.disc) { [lo, hi] = d.rng(P); yv = []; for (let k = lo; k <= hi; k++) { yv.push(d.pmf(P, k)); } ymax = Math.max(...yv, ...counts.map((c) => (n ? c / n : 0))) * 1.2 + 1e-9; }
      else { [lo, hi] = d.view; for (let i = 0; i <= 200; i++) ymax = Math.max(ymax, d.pdf(P, lo + ((hi - lo) * i) / 200)); const bw = (hi - lo) / BN; ymax = Math.max(ymax * 1.2, n ? Math.max(...counts) / (n * bw) * 1.1 : 0); }
      const xl = d.disc ? lo - 0.7 : lo, xh = d.disc ? hi + 0.7 : hi, v = plot(cm, xl, xh, 0, ymax, { l: 44, r: 12, t: 16, b: 28 });
      AI.axes(g, v);
      const mean = d.mean(P), sd = Math.sqrt(d.v(P));
      if (d.disc) {
        const bw = Math.min(26, (v.pw / (hi - lo + 1.4)) * 0.7);
        for (let k = lo; k <= hi; k++) { const p = yv[k - lo]; g.fillStyle = C.blue; g.globalAlpha = 0.8; g.fillRect(v.X(k) - bw / 2, v.Y(p), bw, v.Y(0) - v.Y(p)); g.globalAlpha = 1; if (n) { const e = counts[k - lo] / n; g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(v.X(k) - bw / 2 - 2, v.Y(e), bw + 4, v.Y(0) - v.Y(e)); } }
      } else {
        const bw = (hi - lo) / BN;
        if (n) { g.fillStyle = C.orange; g.globalAlpha = 0.35; counts.forEach((c, i) => { const hh = c / (n * bw); g.fillRect(v.X(lo + i * bw) + 1, v.Y(hh), v.pw / BN - 2, v.Y(0) - v.Y(hh)); }); g.globalAlpha = 1; }
        g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); for (let i = 0; i <= 300; i++) { const x = lo + ((hi - lo) * i) / 300, y = d.pdf(P, x); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke();
      }
      g.strokeStyle = C.red; g.lineWidth = 1.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(v.X(mean), v.Y(0)); g.lineTo(v.X(mean), v.pad.t); g.stroke(); g.setLineDash([]);
      for (const o of drops) { const k = o.t / o.dur; g.fillStyle = C.fg; g.beginPath(); g.arc(v.X(o.x), v.pad.t + k * k * (v.ph - 6) + 3, 3.5, 0, 7); g.fill(); }
      T(g, 'blue: theoretical PMF / PDF    orange: samples so far    red dashed: mean', v.pad.l + 2, 11, C.muted);
      // Galton board
      if (isBin()) {
        const gb = cgb.ctx, W = cgb.w, H = cgb.h, nn = P[0]; gb.clearRect(0, 0, W, H);
        const dx = Math.min(30, (W - 30) / (nn + 1)), top = 18, binH = 80, dy = (H - top - binH - 34) / Math.max(1, nn), cx = W / 2;
        const bx = (r, k) => cx + (k - r / 2) * dx, by = (r) => top + r * dy;
        gb.fillStyle = C.muted;
        for (let r = 0; r < nn; r++) for (let k = 0; k <= r; k++) { gb.beginPath(); gb.arc(bx(r, k), by(r), 2.5, 0, 7); gb.fill(); }
        const base = by(nn) + 16, mxc = Math.max(1, ...counts), scale = (binH - 10) / Math.max(mxc, n * Math.max(...Array.from({ length: nn + 1 }, (_, k) => binomPmf(nn, k, P[1]))));
        for (let k = 0; k <= nn; k++) {
          const x = bx(nn, k), h2 = counts[k] * scale; gb.fillStyle = C.orange; gb.globalAlpha = 0.55; gb.fillRect(x - dx * 0.4, base + binH - h2, dx * 0.8, h2); gb.globalAlpha = 1;
          gb.strokeStyle = C.blue; gb.lineWidth = 2; const eh = n * binomPmf(nn, k, P[1]) * scale; gb.beginPath(); gb.moveTo(x - dx * 0.4, base + binH - eh); gb.lineTo(x + dx * 0.4, base + binH - eh); gb.stroke();
          gb.strokeStyle = C.border; gb.lineWidth = 1; gb.beginPath(); gb.moveTo(x - dx / 2, base + 6); gb.lineTo(x - dx / 2, base + binH); gb.stroke();
          if (dx > 12) T(gb, String(k), x, base + binH + 13, C.muted, 'center');
        }
        gb.fillStyle = C.fg;
        for (const b of balls) {
          const tr = b.t / Trow; let x, y;
          if (tr >= nn) { const k = b.path.reduce((a, c) => a + c, 0), u = AI.clamp((b.t - nn * Trow) / (0.45 / f), 0, 1); x = bx(nn, k); y = by(nn) + u * u * (binH + 8); }
          else { const r = Math.floor(tr), fr = AI.smooth(tr - r); let k = 0; for (let i = 0; i < r; i++) k += b.path[i]; x = AI.lerp(bx(r, k), bx(r + 1, k + b.path[r]), fr); y = AI.lerp(by(r) - 5, by(r + 1) - 5, fr); }
          gb.beginPath(); gb.arc(x, y - 1, 4, 0, 7); gb.fill();
        }
        T(gb, 'Galton board: each peg sends the ball right with prob p. Bin = number of rights. Blue ticks: expected counts.', 8, 11, C.muted);
      }
      const m = n ? sum / n : NaN, vs = n ? sum2 / n - m * m : NaN;
      q('#out').innerHTML = `${q('#sel').value}\nmean     <b>${AI.fmt(mean, 4)}</b>   sample <b class="o">${AI.fmt(m, 4)}</b>\nvariance <b>${AI.fmt(sd * sd, 4)}</b>   sample <b class="o">${AI.fmt(vs, 4)}</b>\nn = ${n}`;
      q('#cap').textContent = d.info + ' ' + (isBin() ? 'Watch the balls: most take a mix of left and right turns, so the middle bins fill first.' : 'Samples fall and join the orange histogram; with more of them it matches the blue curve.');
    }, { fps: 30 });
  };

  /* ============================== 6. Joint Gaussian: covariance, marginals, conditional ============================== */
  AI.labs.gauss2d = function (root) {
    const { view, ctl } = AI.labShell(root, 'Two variables together: the 2D Gaussian', 'scatter, covariance ellipse, marginals on the axes, and a conditional slice',
      `<canvas id="cv" style="height:470px"></canvas><div class="caption" id="cap"></div>`,
      `${sl('sx', 'Std of X (σx)', 0.3, 1.8, 0.05, 1.2)}${sl('sy', 'Std of Y (σy)', 0.3, 1.8, 0.05, 0.8)}${sl('rho', 'Correlation ρ', -0.99, 0.99, 0.01, 0.7)}
       ${sl('x0', 'Slice: condition on X ≈ x₀', -4, 4, 0.1, 1.5)}
       <label style="flex-direction:row;align-items:center;gap:6px"><input type="checkbox" id="cond" checked> show conditional slice</label>
       ${sl('rate', 'New points per second', 1, 150, 1, 25)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="reset">Clear</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s), c = AI.canvas(q('#cv'));
    const M = 1200, R = 5, Z1 = new Float32Array(M), Z2 = new Float32Array(M), CX = new Float32Array(M), CY = new Float32Array(M), BT = new Float32Array(M);
    let cnt = 0, head = 0, playing = true, acc = 0, time = 0, fr = 0;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    q('#reset').onclick = () => { cnt = 0; head = 0; };
    const add = (cx, cy) => { const i = head; Z1[i] = AI.randn(); Z2[i] = AI.randn(); CX[i] = cx; CY[i] = cy; BT[i] = time; head = (head + 1) % M; cnt = Math.min(M, cnt + 1); };
    AI.loop(root, (dt) => {
      syncV(root); time += dt; fr++;
      const sx = +q('#sx').value, sy = +q('#sy').value, rho = +q('#rho').value, x0 = +q('#x0').value, showC = q('#cond').checked, s2 = Math.sqrt(1 - rho * rho);
      if (playing) { acc = Math.min(acc + dt * +q('#rate').value, 40); let g = 0; while (acc >= 1 && g++ < 30) { acc--; add(0, 0); const i = (head + M - 1) % M; CX[i] = sx * Z1[i]; CY[i] = sy * (rho * Z1[i] + s2 * Z2[i]); } }
      const e = 1 - Math.exp(-dt * 5);
      for (let i = 0; i < cnt; i++) { CX[i] += (sx * Z1[i] - CX[i]) * e; CY[i] += (sy * (rho * Z1[i] + s2 * Z2[i]) - CY[i]) * e; }
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const side = Math.max(120, Math.min(W - 36 - 90, H - 90 - 24)), pad = { l: 36, t: H - 24 - side, r: W - 36 - side, b: 24 }, v = plot(c, -R, R, -R, R, pad);
      AI.axes(g, v, {});
      // slice band
      const bwid = 0.3;
      if (showC) { g.fillStyle = C.orange; g.globalAlpha = 0.13; g.fillRect(v.X(x0 - bwid), v.pad.t, v.X(x0 + bwid) - v.X(x0 - bwid), v.ph); g.globalAlpha = 1; }
      // points
      const BX = 40, bx = new Array(BX).fill(0), by = new Array(BX).fill(0), bc = new Array(BX).fill(0); let nc = 0, mx = 0, my = 0;
      for (let i = 0; i < cnt; i++) { mx += CX[i]; my += CY[i]; }
      mx /= Math.max(cnt, 1); my /= Math.max(cnt, 1); let sxy = 0, sxx = 0, syy = 0;
      for (let i = 0; i < cnt; i++) {
        const x = CX[i], y = CY[i], inS = showC && Math.abs(x - x0) < bwid, age = time - BT[i], r = 2.4 + 4 * Math.exp(-age * 2.2);
        g.fillStyle = inS ? C.orange : C.blue; g.globalAlpha = inS ? 0.95 : 0.55; g.beginPath(); g.arc(v.X(x), v.Y(y), r, 0, 6.2832); g.fill();
        const a = Math.floor(((x + R) / (2 * R)) * BX), b = Math.floor(((y + R) / (2 * R)) * BX); if (a >= 0 && a < BX) bx[a]++; if (b >= 0 && b < BX) by[b]++;
        if (inS) { nc++; const bb = Math.floor(((y + R) / (2 * R)) * BX); if (bb >= 0 && bb < BX) bc[bb]++; }
        sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2;
      }
      g.globalAlpha = 1;
      // covariance ellipse + principal axes
      const a = sx * sx, b = rho * sx * sy, d = sy * sy, tr = a + d, df = (a - d) / 2, rt = Math.sqrt(df * df + b * b), l1 = tr / 2 + rt, l2 = tr / 2 - rt, th = 0.5 * Math.atan2(2 * b, a - d);
      [1, 2].forEach((k) => { g.strokeStyle = C.red; g.lineWidth = k === 1 ? 2.5 : 1.5; g.globalAlpha = k === 1 ? 1 : 0.6; g.beginPath(); for (let i = 0; i <= 90; i++) { const p = (i / 90) * 6.2832, u = k * Math.sqrt(l1) * Math.cos(p), w = k * Math.sqrt(Math.max(l2, 0)) * Math.sin(p), x = u * Math.cos(th) - w * Math.sin(th), y = u * Math.sin(th) + w * Math.cos(th); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke(); });
      g.globalAlpha = 1; g.strokeStyle = C.red; g.lineWidth = 1; g.setLineDash([5, 4]);
      [[Math.cos(th), Math.sin(th), Math.sqrt(l1)], [-Math.sin(th), Math.cos(th), Math.sqrt(Math.max(l2, 0))]].forEach(([ux, uy, s]) => { g.beginPath(); g.moveTo(v.X(-3 * s * ux), v.Y(-3 * s * uy)); g.lineTo(v.X(3 * s * ux), v.Y(3 * s * uy)); g.stroke(); }); g.setLineDash([]);
      T(g, 'red: 1σ and 2σ contours of Σ; dashed = principal axes (PCA)', v.pad.l + 4, v.pad.t + 12, C.red);
      // marginals
      const topH = v.pad.t - 10, rightW = c.w - v.pad.l - v.pw - 10, pk = 1 / (Math.min(sx, sy) * Math.sqrt(2 * Math.PI)) * 1.05;
      const bwd = v.pw / BX;
      g.fillStyle = C.blue; g.globalAlpha = 0.6;
      for (let i = 0; i < BX; i++) { const dens = cnt ? bx[i] / (cnt * (2 * R / BX)) : 0, h2 = (dens / pk) * topH; g.fillRect(v.pad.l + i * bwd + 1, v.pad.t - 4 - h2, bwd - 2, h2); const dy = cnt ? by[i] / (cnt * (2 * R / BX)) : 0, w2 = (dy / pk) * rightW; g.fillRect(v.pad.l + v.pw + 4, v.pad.t + v.ph - (i + 1) * (v.ph / BX) + 1, w2, v.ph / BX - 2); }
      g.globalAlpha = 1; g.strokeStyle = C.fg; g.lineWidth = 1.8;
      g.beginPath(); for (let i = 0; i <= 200; i++) { const x = -R + (2 * R * i) / 200, h2 = (normPdf(x, 0, sx) / pk) * topH; i ? g.lineTo(v.X(x), v.pad.t - 4 - h2) : g.moveTo(v.X(x), v.pad.t - 4 - h2); } g.stroke();
      g.beginPath(); for (let i = 0; i <= 200; i++) { const y = -R + (2 * R * i) / 200, w2 = (normPdf(y, 0, sy) / pk) * rightW; i ? g.lineTo(v.pad.l + v.pw + 4 + w2, v.Y(y)) : g.moveTo(v.pad.l + v.pw + 4 + w2, v.Y(y)); } g.stroke();
      T(g, 'marginal of X', v.pad.l + 4, 12, C.muted); T(g, 'marginal Y', c.w - 4, 12, C.muted, 'right');
      // conditional
      const cm = rho * (sy / sx) * x0, cs = sy * s2;
      if (showC) {
        g.strokeStyle = C.orange; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(x0), v.pad.t); g.lineTo(v.X(x0), v.pad.t + v.ph); g.stroke();
        g.fillStyle = C.orange; g.globalAlpha = 0.5;
        const pk2 = 1 / (cs * Math.sqrt(2 * Math.PI)) * 1.05;
        for (let i = 0; i < BX; i++) { const dens = nc ? bc[i] / (nc * (2 * R / BX)) : 0, w2 = Math.min(rightW, (dens / pk2) * rightW * 0.9); g.fillRect(v.pad.l + v.pw + 4, v.pad.t + v.ph - (i + 1) * (v.ph / BX) + 1, w2, v.ph / BX - 2); }
        g.globalAlpha = 1; g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 200; i++) { const y = -R + (2 * R * i) / 200, w2 = (normPdf(y, cm, cs) / pk2) * rightW * 0.9; i ? g.lineTo(v.pad.l + v.pw + 4 + w2, v.Y(y)) : g.moveTo(v.pad.l + v.pw + 4 + w2, v.Y(y)); } g.stroke();
        T(g, 'Y | X≈x₀', c.w - 4, v.pad.t + 26, C.orange, 'right');
      }
      const scorr = cnt > 2 ? sxy / Math.sqrt(sxx * syy) : NaN;
      q('#out').innerHTML = `Σ = [ ${AI.fmt(a, 3)}  ${AI.fmt(b, 3)} ]\n    [ ${AI.fmt(b, 3)}  ${AI.fmt(d, 3)} ]\ncov(X,Y) = ρσxσy = <b>${AI.fmt(b, 3)}</b>\ncorr ρ = <b>${AI.fmt(rho, 2)}</b>   sample <b class="o">${AI.fmt(scorr, 2)}</b>\neigenvalues  ${AI.fmt(l1, 3)}, ${AI.fmt(l2, 3)}\n` + (showC ? `\nGiven X = ${AI.fmt(x0, 1)}:\n E[Y|X] = ρσy/σx·x₀ = <b class="o">${AI.fmt(cm, 3)}</b>\n Var(Y|X) = σy²(1−ρ²) = <b class="o">${AI.fmt(cs * cs, 3)}</b>\n<span class="m">(marginal Var(Y) = ${AI.fmt(d, 3)})</span>` : '') + `\npoints ${cnt}`;
      q('#cap').textContent = 'Blue dots are samples of the pair (X, Y). Slide ρ and the cloud tilts; the red ellipse is the covariance matrix drawn as a shape. Orange dots: the slice where X is near x₀, whose Y-values form a narrower bell than the marginal.';
    }, { fps: 30 });
  };

  /* ============================== 7. LLN & CLT with many running-mean paths ============================== */
  AI.labs.lln = function (root) {
    const D = {
      'Fair coin (0 or 1)': { mu: 0.5, sd: 0.5, y: [-0.15, 1.15], s: () => (Math.random() < 0.5 ? 1 : 0) },
      'Fair die (1–6)': { mu: 3.5, sd: Math.sqrt(35 / 12), y: [0.5, 6.5], s: () => 1 + (Math.random() * 6 | 0) },
      'Exponential(1)': { mu: 1, sd: 1, y: [-0.3, 3.2], s: () => -Math.log(U()) },
      'Lottery (100 w.p. 0.01, else 0)': { mu: 1, sd: Math.sqrt(99), y: [-4, 14], s: () => (Math.random() < 0.01 ? 100 : 0) },
      'Cauchy (no mean!)': { mu: 0, sd: NaN, y: [-6, 6], s: () => Math.tan(Math.PI * (U() - 0.5)) },
    };
    const names = Object.keys(D);
    const { view, ctl } = AI.labShell(root, 'Law of large numbers and the CLT', '300 independent experiments, each keeps a running average of its own draws',
      `<canvas id="cv" style="height:400px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Each draw X comes from<select id="sel">${names.map((n) => `<option>${n}</option>`).join('')}</select></label>
       ${sl('rate', 'Draws per second (n grows this fast)', 2, 400, 1, 15)}
       <label style="flex-direction:row;align-items:center;gap:6px"><input type="checkbox" id="zoom"> zoom the histogram to ± 3σ/√n</label>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">+1 draw</button><button class="btn" id="reset">Restart</button></div>
       <div class="row"><span style="color:var(--muted)">jump to n =</span><button class="btn small" data-n="10">10</button><button class="btn small" data-n="100">100</button><button class="btn small" data-n="1000">1000</button><button class="btn small" data-n="5000">5000</button></div>
       <div class="readout" id="out"></div>`);
    const q = (s) => $(root, s), c = AI.canvas(q('#cv'));
    const NP = 300, SHOW = 12, NMAX = 5000;
    let d = D[names[0]], sums = new Float64Array(NP), hist = Array.from({ length: SHOW }, () => new Float32Array(NMAX + 1)), n = 0, playing = true, acc = 0, dirty = true;
    const reset = () => { d = D[q('#sel').value]; sums.fill(0); n = 0; dirty = true; };
    const step = () => { if (n >= NMAX) return; n++; for (let p = 0; p < NP; p++) { sums[p] += d.s(); if (p < SHOW) hist[p][n] = sums[p] / n; } dirty = true; };
    q('#sel').onchange = reset; q('#reset').onclick = reset; q('#step').onclick = step; q('#zoom').onchange = () => (dirty = true);
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    AI.$$('[data-n]', root).forEach((b) => (b.onclick = () => { const t = +b.dataset.n; while (n < t && n < NMAX) step(); }));
    reset();
    let wk = '';
    AI.loop(root, (dt) => {
      syncV(root);
      if (playing) { acc = Math.min(acc + dt * +q('#rate').value, 60); let g = 0; while (acc >= 1 && g++ < 40) { acc--; step(); } }
      const k = c.w + '|' + c.h; if (k !== wk) { wk = k; dirty = true; }
      if (!dirty) return; dirty = false;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const hw = Math.max(110, W * 0.24), pad = { l: 44, r: hw + 10, t: 16, b: 28 }, ly = Math.log10(NMAX), cauchy = isNaN(d.sd);
      let y0 = d.y[0], y1 = d.y[1];
      const v = plot(c, 0, ly, y0, y1, pad), X = (m) => v.X(Math.log10(Math.max(1, m)));
      // axes
      g.strokeStyle = C.border; g.lineWidth = 1; g.fillStyle = C.muted; g.font = '11px ui-monospace,monospace'; g.textAlign = 'center';
      [1, 10, 100, 1000, 5000].forEach((m) => { const x = Math.round(X(m)) + 0.5; g.beginPath(); g.moveTo(x, v.pad.t); g.lineTo(x, v.pad.t + v.ph); g.stroke(); g.fillText(String(m), x, v.pad.t + v.ph + 15); });
      g.textAlign = 'right'; const sy = AI.niceStep(y1 - y0, 5); for (let y = Math.ceil(y0 / sy) * sy; y <= y1; y += sy) { const py = Math.round(v.Y(y)) + 0.5; g.beginPath(); g.moveTo(v.pad.l, py); g.lineTo(v.pad.l + v.pw, py); g.stroke(); g.fillText(AI.fmt(+y.toFixed(4), 2), v.pad.l - 5, py + 4); }
      T(g, 'number of draws n (log scale) →', v.pad.l + v.pw / 2, v.pad.t + v.ph + 26, C.muted, 'center');
      // funnel
      if (!cauchy) {
        g.strokeStyle = C.green; g.lineWidth = 1.5; g.setLineDash([5, 4]);
        [1, -1].forEach((sg) => { g.beginPath(); for (let i = 0; i <= 100; i++) { const m = Math.pow(10, (ly * i) / 100), y = AI.clamp(d.mu + sg * 2 * d.sd / Math.sqrt(m), y0, y1); i ? g.lineTo(X(m), v.Y(y)) : g.moveTo(X(m), v.Y(y)); } g.stroke(); });
        g.setLineDash([]);
      }
      g.strokeStyle = C.red; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.pad.l, v.Y(d.mu)); g.lineTo(v.pad.l + v.pw, v.Y(d.mu)); g.stroke();
      // paths
      const pc = [C.blue, C.orange, C.purple, C.teal, C.pink, C.yellow];
      for (let p = 0; p < SHOW; p++) {
        g.strokeStyle = pc[p % pc.length]; g.lineWidth = 1.6; g.globalAlpha = 0.85; g.beginPath(); let st = Math.max(1, Math.floor(n / 800));
        for (let m = 1; m <= n; m += (m < 200 ? 1 : st)) { const px = X(m), py = AI.clamp(v.Y(hist[p][m]), v.pad.t, v.pad.t + v.ph); m === 1 ? g.moveTo(px, py) : g.lineTo(px, py); }
        if (n > 1) g.lineTo(X(n), AI.clamp(v.Y(hist[p][n]), v.pad.t, v.pad.t + v.ph)); g.stroke();
      }
      g.globalAlpha = 1;
      if (n > 0) { g.strokeStyle = C.fg; g.globalAlpha = 0.6; g.lineWidth = 1; g.beginPath(); g.moveTo(X(n), v.pad.t); g.lineTo(X(n), v.pad.t + v.ph); g.stroke(); g.globalAlpha = 1; }
      T(g, cauchy ? 'running averages — no limit exists' : `12 of 300 running averages   red = true mean μ = ${AI.fmt(d.mu, 3)}   green dashed = μ ± 2σ/√n`, v.pad.l + 4, 11, C.muted);
      // histogram panel at current n
      const hx = v.pad.l + v.pw + 12, hwid = hw - 2, zoom = q('#zoom').checked && !cauchy && n > 0, BN = 30;
      let a0 = y0, a1 = y1; if (zoom) { const w = 3 * d.sd / Math.sqrt(n); a0 = d.mu - w; a1 = d.mu + w; }
      const cnts = new Array(BN).fill(0); let sm = 0, s2 = 0;
      for (let p = 0; p < NP; p++) { const m = n ? sums[p] / n : 0; sm += m; s2 += m * m; const b = Math.floor(((m - a0) / (a1 - a0)) * BN); if (n && b >= 0 && b < BN) cnts[b]++; }
      const bw = (a1 - a0) / BN, th = !cauchy && n ? normPdf(d.mu, d.mu, d.sd / Math.sqrt(n)) : 0, mxc = Math.max(...cnts.map((x) => x / (NP * bw)), th, 1e-9) * 1.08;
      g.fillStyle = C.bg2; g.fillRect(hx, v.pad.t, hwid, v.ph);
      g.fillStyle = C.blue; g.globalAlpha = 0.7;
      cnts.forEach((cn, i) => { const len = (cn / (NP * bw) / mxc) * (hwid - 6); g.fillRect(hx, v.pad.t + v.ph - (i + 1) * (v.ph / BN) + 1, len, v.ph / BN - 2); }); g.globalAlpha = 1;
      if (!cauchy && n) { g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 120; i++) { const y = a0 + ((a1 - a0) * i) / 120, len = (normPdf(y, d.mu, d.sd / Math.sqrt(n)) / mxc) * (hwid - 6), py = v.pad.t + v.ph - (i / 120) * v.ph; i ? g.lineTo(hx + len, py) : g.moveTo(hx + len, py); } g.stroke(); }
      T(g, `means at n = ${n}`, hx + 2, v.pad.t - 3, C.muted);
      if (zoom) { T(g, AI.fmt(a1, 3), hx + hwid - 2, v.pad.t + 10, C.muted, 'right'); T(g, AI.fmt(a0, 3), hx + hwid - 2, v.pad.t + v.ph - 3, C.muted, 'right'); T(g, 'own scale', hx + hwid - 2, v.pad.t + v.ph / 2, C.muted, 'right'); }
      const mean = n ? sm / NP : NaN, sdv = n ? Math.sqrt(Math.max(0, s2 / NP - mean * mean)) : NaN;
      q('#out').innerHTML = `n = <b>${n}</b>\nmean of the 300 averages  <b>${AI.fmt(mean, 3)}</b>\nspread (std) of the averages <b class="o">${AI.fmt(sdv, 3)}</b>\n` + (cauchy ? '<span class="r">Cauchy has no mean: the spread never shrinks.</span>' : `theory σ/√n = <b>${AI.fmt(d.sd / Math.sqrt(Math.max(n, 1)), 3)}</b>   (σ = ${AI.fmt(d.sd, 3)})`);
      q('#cap').textContent = cauchy ? 'Cauchy: averaging does not help. One huge draw can drag any average away, and that keeps happening forever.' : 'Each coloured path is one experiment. The cloud of paths funnels toward the true mean (LLN) and the histogram on the right becomes a bell curve (CLT). Heavy skew (lottery) needs a much larger n.';
    }, { fps: 30 });
  };
})();
