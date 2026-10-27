/* Labs for ML Module 5 · Evaluation & model selection.
 *   confmatrix  threshold over two score distributions, matrix cells fill, cost matrix -> optimal t
 *   rocpr       ROC and PR curves drawn point by point, AUC shading, class-imbalance slider
 *   regmetrics  draggable points/line, residuals, MAE/RMSE/R2/MAPE, outlier sensitivity
 *   cvlab       k-fold / stratified / group / time-series folds rotating with per-fold scores
 *   searchlab   grid vs random vs Bayesian search on a 2-D landscape
 *   learncurve  learning curve + validation curve with a diagnosis
 *   calibration reliability diagram with Platt / isotonic correction
 *   fairness    two groups, group thresholds, parity / equal-opportunity trade-off
 * All animations are dt-based, have a speed slider, pause/step and a caption. Illustrative simulations unless stated.
 */
(function () {
  const E = {};
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  const ease = (dt, k) => 1 - Math.exp(-dt * k);
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const logit = (p) => Math.log(AI.clamp(p, 1e-6, 1 - 1e-6) / (1 - AI.clamp(p, 1e-6, 1 - 1e-6)));
  const fx = (v, d = 3) => AI.fmt(v, d);

  /* play / step / reset / speed block; returns state object */
  E.bar = (label) => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Restart</button></div>
    <label>Speed <span class="v" id="spv">1×</span><input type="range" id="sp" min="0.25" max="3" step="0.05" value="1"></label>`;
  E.controls = (root, cb) => {
    const q = (s) => AI.$(s, root), st = { playing: true, speed: 1 };
    const label = () => (q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play');
    q('#play').onclick = () => { st.playing = !st.playing; label(); if (cb.onPlay) cb.onPlay(st.playing); };
    q('#step').onclick = () => { st.playing = false; label(); cb.onStep && cb.onStep(); };
    q('#reset').onclick = () => { st.playing = true; label(); cb.onReset && cb.onReset(); };
    q('#sp').oninput = () => { st.speed = +q('#sp').value; q('#spv').textContent = fx(st.speed, 2) + '×'; };
    st.pause = () => { st.playing = false; label(); };
    st.play = () => { st.playing = true; label(); };
    return st;
  };
  E.txt = (g, s, x, y, col, align = 'left', font = '11px sans-serif') => { g.fillStyle = col; g.font = font; g.textAlign = align; g.fillText(s, x, y); };
  E.curve = (g, v, from, to, f, color, alpha) => {
    g.fillStyle = color; g.globalAlpha = alpha; g.beginPath(); g.moveTo(v.X(from), v.Y(0));
    for (let x = from; x <= to + 1e-9; x += 0.05) g.lineTo(v.X(x), v.Y(f(x)));
    g.lineTo(v.X(to), v.Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1;
  };
  E.axisLabel = (g, c, v, xl, yl) => {
    const C = AI.col(); E.txt(g, xl, v.pad.l + v.pw / 2, c.h - 3, C.muted, 'center');
    g.save(); g.translate(10, v.pad.t + v.ph / 2); g.rotate(-Math.PI / 2); E.txt(g, yl, 0, 0, C.muted, 'center'); g.restore();
  };
  E.drag = (cv, fn) => {
    let on = false;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('pointerdown', (e) => { on = true; cv.setPointerCapture(e.pointerId); fn('down', ...pos(e)); });
    cv.addEventListener('pointermove', (e) => { if (on) fn('move', ...pos(e)); else fn('hover', ...pos(e)); });
    const up = () => { if (on) fn('up'); on = false; };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  };
  const ro = (c, f) => { c.onresize = f; };

  /* ======================= 1. confusion matrix ======================= */
  AI.labs.confmatrix = function (root) {
    const { ctl } = AI.labShell(root, 'Confusion-matrix lab', 'drag the threshold across the two score distributions; the four cells and every metric follow (illustrative model)',
      `<canvas id="dist" style="height:240px;cursor:ew-resize;touch-action:none"></canvas><canvas id="mx" style="height:210px"></canvas>`,
      `<label>Separation d′ (how far apart the classes score) <span class="v" id="dv"></span><input type="range" id="d" min="0.5" max="4" step="0.1" value="1.8"></label>
       <label>Positives are this % of cases <span class="v" id="pv"></span><input type="range" id="p" min="1" max="50" step="1" value="30"></label>
       <label>Threshold t (alarm if score ≥ t) <span class="v" id="tv"></span><input type="range" id="t" min="-3" max="6" step="0.05" value="1"></label>
       <div class="row"><label style="flex:1">cost of a false alarm<input type="number" id="cfp" value="1" min="0" step="0.5"></label><label style="flex:1">cost of a miss<input type="number" id="cfn" value="5" min="0" step="0.5"></label></div>
       <div class="row"><button class="btn" id="opt">Use min-cost t</button><button class="btn" id="f1b">Use max-F1 t</button></div>
       ${E.bar()}
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#dist')), c2 = AI.canvas(q('#mx'));
    let t = 1, tTarget = null, dir = 1, disp = null, key = '', vv = null, costKey = '', costCurve = [], lastCap = '';
    const P = () => ({ d: +q('#d').value, pi: +q('#p').value / 100, cfp: +q('#cfp').value || 0, cfn: +q('#cfn').value || 0 });
    const counts = (d, pi, tt) => { const N = 1000, tpr = 1 - Phi(tt - d), fpr = 1 - Phi(tt); return [N * pi * tpr, N * (1 - pi) * fpr, N * pi * (1 - tpr), N * (1 - pi) * (1 - fpr)]; };
    const cost = (p, tt) => { const [, fp, fn] = counts(p.d, p.pi, tt); return p.cfp * fp + p.cfn * fn; };
    const f1of = (p, tt) => { const [tp, fp, fn] = counts(p.d, p.pi, tt); return (2 * tp) / (2 * tp + fp + fn || 1); };
    const argbest = (fn, sgn) => { let bt = -3, bv = Infinity; for (let x = -3; x <= 6; x += 0.02) { const v = sgn * fn(x); if (v < bv) { bv = v; bt = x; } } return bt; };
    const setT = (v) => { t = AI.clamp(v, -3, 6); q('#t').value = t; };
    const ctrl = E.controls(root, { onStep: () => { tTarget = null; setT(t + 0.25); }, onReset: () => { tTarget = null; setT(-2.5); dir = 1; } });
    q('#t').oninput = () => { ctrl.pause(); tTarget = null; t = +q('#t').value; };
    q('#opt').onclick = () => { ctrl.pause(); tTarget = argbest((x) => cost(P(), x), 1); };
    q('#f1b').onclick = () => { ctrl.pause(); tTarget = argbest((x) => f1of(P(), x), -1); };
    ['#d', '#p', '#cfp', '#cfn'].forEach((s) => (q(s).oninput = () => { key = ''; }));
    E.drag(q('#dist'), (ev, x) => { if (ev === 'hover' || !vv) return; if (ev === 'down') ctrl.pause(); tTarget = null; setT(vv.ix(x)); });
    ro(c1, () => (key = '')); ro(c2, () => (key = ''));

    AI.loop(root, (dt) => {
      const p = P();
      if (tTarget !== null) { t += (tTarget - t) * ease(dt, 3 * ctrl.speed); q('#t').value = t; if (Math.abs(tTarget - t) < 0.005) { t = tTarget; tTarget = null; } }
      else if (ctrl.playing) { t += dir * 0.35 * ctrl.speed * dt; if (t > 5) { t = 5; dir = -1; } if (t < -2.5) { t = -2.5; dir = 1; } q('#t').value = t; }
      const tg = counts(p.d, p.pi, t);
      if (!disp) disp = tg.slice(); else for (let i = 0; i < 4; i++) disp[i] += (tg[i] - disp[i]) * ease(dt, 6);
      const k = [fx(t, 2), p.d, p.pi, p.cfp, p.cfn, disp.map((v) => Math.round(v * 2)).join()].join('|');
      if (k === key) return; key = k;
      q('#dv').textContent = p.d; q('#pv').textContent = Math.round(p.pi * 100) + '%'; q('#tv').textContent = fx(t, 2);
      const C = AI.col(), [TP, FP, FN, TN] = tg, Np = 1000 * p.pi, Nn = 1000 - Np;
      /* distributions */
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const ymax = Math.max((1 - p.pi) * 0.4, p.pi * 0.4, 0.06) * 1.2, v = (vv = AI.view(c1, -4, 7, 0, ymax, { l: 12, r: 12, t: 22, b: 22 }));
      AI.axes(g, v, { noY: true, noGrid: true });
      const fn0 = (x) => (1 - p.pi) * pdf(x), fn1 = (x) => p.pi * pdf(x - p.d);
      E.curve(g, v, -4, t, fn0, C.muted, 0.5); E.curve(g, v, t, 7, fn0, C.red, 0.55);
      E.curve(g, v, -4, t, fn1, C.orange, 0.6); E.curve(g, v, t, 7, fn1, C.green, 0.6);
      g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v.X(t), 14); g.lineTo(v.X(t), v.Y(0)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.fg; g.beginPath(); g.moveTo(v.X(t), 20); g.lineTo(v.X(t) - 6, 10); g.lineTo(v.X(t) + 6, 10); g.fill();
      E.txt(g, 'predict negative ◂', v.X(t) - 8, 12, C.muted, 'right'); E.txt(g, '▸ predict positive', v.X(t) + 8, 12, C.muted);
      E.txt(g, 'negatives', v.X(0), v.Y(fn0(0)) - 5, C.muted, 'center'); E.txt(g, 'positives', v.X(p.d), v.Y(fn1(p.d)) - 5, C.muted, 'center');
      /* matrix + cost */
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const mw = Math.min(c2.w * 0.5, 330), x0 = 62, y0 = 34, cw = (mw - x0) / 2, ch = (c2.h - y0 - 10) / 2;
      E.txt(g, 'predicted +', x0 + cw / 2, 22, C.muted, 'center'); E.txt(g, 'predicted −', x0 + cw * 1.5, 22, C.muted, 'center');
      E.txt(g, 'actual +', x0 - 6, y0 + ch / 2 + 4, C.muted, 'right'); E.txt(g, 'actual −', x0 - 6, y0 + ch * 1.5 + 4, C.muted, 'right');
      [['TP', C.green, disp[0], Np, 0, 0], ['FN', C.orange, disp[2], Np, 1, 0], ['FP', C.red, disp[1], Nn, 0, 1], ['TN', C.muted, disp[3], Nn, 1, 1]].forEach(([n, col, val, tot, cx, cy]) => {
        const X = x0 + cx * cw, Y = y0 + cy * ch, fr = tot ? AI.clamp(val / tot, 0, 1) : 0;
        g.fillStyle = C.bg2; g.fillRect(X + 2, Y + 2, cw - 4, ch - 4);
        g.fillStyle = col; g.globalAlpha = 0.55; g.fillRect(X + 2, Y + 2 + (ch - 4) * (1 - fr), cw - 4, (ch - 4) * fr); g.globalAlpha = 1;
        g.strokeStyle = col; g.lineWidth = 1.5; g.strokeRect(X + 2, Y + 2, cw - 4, ch - 4);
        E.txt(g, n, X + 8, Y + 17, C.fg, 'left', 'bold 12px sans-serif'); E.txt(g, String(Math.round(val)), X + cw / 2, Y + ch / 2 + 8, C.fg, 'center', 'bold 20px sans-serif');
      });
      if (c2.w > 420) {
        const px = x0 + cw * 2 + 36, vr = AI.view(c2, -3, 6, 0, 1, { l: px, r: 14, t: 24, b: 24 }); // cost plot
        const ck = [p.d, p.pi, p.cfp, p.cfn].join();
        if (ck !== costKey) { costKey = ck; costCurve = []; for (let x = -3; x <= 6.001; x += 0.1) costCurve.push([x, cost(p, x)]); }
        const cmax = Math.max(...costCurve.map((a) => a[1])) || 1, cmin = Math.min(...costCurve.map((a) => a[1]));
        const vc = AI.view(c2, -3, 6, 0, cmax * 1.05, { l: px, r: 14, t: 24, b: 24 });
        g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(vc.pad.l, vc.pad.t, vc.pw, vc.ph);
        g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); costCurve.forEach(([x, y], i) => (i ? g.lineTo(vc.X(x), vc.Y(y)) : g.moveTo(vc.X(x), vc.Y(y)))); g.stroke();
        const best = costCurve.find((a) => a[1] === cmin);
        g.fillStyle = C.green; g.beginPath(); g.arc(vc.X(best[0]), vc.Y(best[1]), 5, 0, 7); g.fill();
        g.fillStyle = C.fg; g.beginPath(); g.arc(vc.X(t), vc.Y(cost(p, t)), 5, 0, 7); g.fill();
        E.txt(g, 'total cost vs threshold (green = minimum)', vc.pad.l, 14, C.muted); E.txt(g, 'threshold t', vc.pad.l + vc.pw / 2, c2.h - 4, C.muted, 'center');
      }
      /* readout */
      const acc = (TP + TN) / 1000, prec = TP / (TP + FP || 1), rec = TP / (TP + FN || 1), spec = TN / (TN + FP || 1), f1 = (2 * prec * rec) / (prec + rec || 1);
      const mcc = (TP * TN - FP * FN) / (Math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) || 1);
      q('#out').innerHTML = `<span class="m">expected counts per 1000 cases</span>\naccuracy    <b>${fx(acc)}</b>   (always-negative: ${fx(1 - p.pi)})\nprecision   <b>${fx(prec)}</b>\nrecall      <b>${fx(rec)}</b>\nspecificity <b>${fx(spec)}</b>\nF1          <b>${fx(f1)}</b>\nMCC         <b>${fx(mcc)}</b>\ntotal cost  <span class="o">${fx(p.cfp * FP + p.cfn * FN, 1)}</span>`;
      const cap = tTarget !== null ? 'Moving the threshold to the chosen optimum…' : t > 3 ? 'Very strict: few alarms, almost no false alarms, but most positives are missed (FN grows).' : t < 0 ? 'Very lenient: nearly every positive is caught, but many negatives raise false alarms (FP grows).' : 'Cells fill as the cut-off moves: TP+FN = all positives, FP+TN = all negatives. Compare accuracy with the always-negative baseline.';
      if (cap !== lastCap) { q('#cap').textContent = cap; lastCap = cap; }
    });
  };

  /* ======================= 2. ROC / PR builder ======================= */
  AI.labs.rocpr = function (root) {
    const { ctl } = AI.labShell(root, 'ROC & precision–recall builder', 'sweep the threshold from high to low: each example passed adds one point to both curves (illustrative scores)',
      `<canvas id="strip" style="height:120px"></canvas><canvas id="plots" style="height:290px"></canvas>`,
      `<label>Separation d′ <span class="v" id="dv"></span><input type="range" id="d" min="0" max="4" step="0.1" value="1.5"></label>
       <label>Positives are this % of cases <span class="v" id="pv"></span><input type="range" id="p" min="1" max="50" step="1" value="50"></label>
       ${E.bar()}
       <div class="readout" id="out"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#strip')), c2 = AI.canvas(q('#plots'));
    const NNEG = 300, ranN = AI.rng(11), ranP = AI.rng(23), baseN = Array.from({ length: NNEG }, () => AI.randn(ranN)), baseP = Array.from({ length: 300 }, () => AI.randn(ranP));
    let items = [], cumTP = [], cumFP = [], np = 0, nn = NNEG, k = 0, key = '', tick = AI.ticker(), auc = 0, ap = 0, aucAt = [], apAt = [];
    const build = () => {
      const d = +q('#d').value, pi = +q('#p').value / 100; np = Math.max(3, Math.round((NNEG * pi) / (1 - pi)));
      items = baseN.map((s) => ({ s, y: 0 })).concat(baseP.slice(0, np).map((s) => ({ s: s + d, y: 1 }))).sort((a, b) => b.s - a.s);
      cumTP = [0]; cumFP = [0]; aucAt = [0]; apAt = [0]; let a = 0, apx = 0;
      items.forEach((it, i) => {
        const tp = cumTP[i] + it.y, fp = cumFP[i] + (1 - it.y); cumTP.push(tp); cumFP.push(fp);
        if (!it.y) a += tp / np / nn; else apx += tp / (tp + fp) / np;
        aucAt.push(a); apAt.push(apx);
      });
      auc = a; ap = apx; k = 0; key = '';
    };
    const ctrl = E.controls(root, { onStep: () => { k = Math.min(items.length, k + 1); }, onReset: () => { k = 0; } });
    q('#d').oninput = () => { build(); ctrl.play(); }; q('#p').oninput = () => { build(); ctrl.play(); };
    ro(c1, () => (key = '')); ro(c2, () => (key = ''));
    build();
    AI.loop(root, (dt) => {
      if (ctrl.playing) { k = Math.min(items.length, k + tick(dt, 22 * ctrl.speed)); }
      const kk = key; const nk = [k, q('#d').value, q('#p').value].join('|'); if (nk === kk) return; key = nk;
      const d = +q('#d').value, pi = np / (np + nn), C = AI.col();
      q('#dv').textContent = d; q('#pv').textContent = Math.round(+q('#p').value) + '%';
      /* strip: histograms by class (counts), threshold at the k-th score */
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const lo = -3.5, hi = 7.5, bins = 44, bn = new Array(bins).fill(0), bp = new Array(bins).fill(0);
      items.forEach((it) => { const b = AI.clamp(Math.floor(((it.s - lo) / (hi - lo)) * bins), 0, bins - 1); (it.y ? bp : bn)[b]++; });
      const my = Math.max(...bn, ...bp, 1), v = AI.view(c1, lo, hi, 0, my * 1.1, { l: 12, r: 12, t: 14, b: 22 });
      AI.axes(g, v, { noY: true, noGrid: true });
      const bw = v.pw / bins;
      for (let b = 0; b < bins; b++) {
        g.fillStyle = C.muted; g.globalAlpha = 0.6; g.fillRect(v.pad.l + b * bw + 1, v.Y(bn[b]), bw / 2 - 1, v.Y(0) - v.Y(bn[b]));
        g.fillStyle = C.green; g.fillRect(v.pad.l + b * bw + bw / 2, v.Y(bp[b]), bw / 2 - 1, v.Y(0) - v.Y(bp[b])); g.globalAlpha = 1;
      }
      const tScore = k === 0 ? hi : items[k - 1].s;
      g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(v.X(tScore), 6); g.lineTo(v.X(tScore), v.Y(0)); g.stroke(); g.setLineDash([]);
      E.txt(g, `grey = negatives (${nn}), green = positives (${np}); threshold sweeps right → left`, v.pad.l, 11, C.muted);
      /* plots */
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const half = c2.w / 2, mk = (x0) => ({ l: x0 + 38, r: c2.w - x0 - half + 14, t: 22, b: 36 });
      const vr = AI.view(c2, 0, 1, 0, 1, { l: 38, r: half + 12, t: 22, b: 36 }), vp = AI.view(c2, 0, 1, 0, 1, { l: half + 34, r: 14, t: 22, b: 36 });
      [vr, vp].forEach((vw) => { g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(vw.pad.l, vw.pad.t, vw.pw, vw.ph); [0.25, 0.5, 0.75].forEach((f) => { g.beginPath(); g.moveTo(vw.X(f), vw.Y(0)); g.lineTo(vw.X(f), vw.Y(1)); g.moveTo(vw.X(0), vw.Y(f)); g.lineTo(vw.X(1), vw.Y(f)); g.stroke(); }); });
      E.txt(g, 'ROC', vr.pad.l, 14, C.fg, 'left', 'bold 12px sans-serif'); E.txt(g, 'Precision–recall', vp.pad.l, 14, C.fg, 'left', 'bold 12px sans-serif');
      E.txt(g, 'false positive rate', vr.pad.l + vr.pw / 2, c2.h - 6, C.muted, 'center'); E.txt(g, 'recall', vp.pad.l + vp.pw / 2, c2.h - 6, C.muted, 'center');
      E.txt(g, '1', vr.pad.l - 5, vr.Y(1) + 4, C.muted, 'right'); E.txt(g, '0', vr.pad.l - 5, vr.Y(0), C.muted, 'right'); E.txt(g, '1', vp.pad.l - 5, vp.Y(1) + 4, C.muted, 'right'); E.txt(g, '0', vp.pad.l - 5, vp.Y(0), C.muted, 'right');
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(vr.X(0), vr.Y(0)); g.lineTo(vr.X(1), vr.Y(1)); g.moveTo(vp.X(0), vp.Y(pi)); g.lineTo(vp.X(1), vp.Y(pi)); g.stroke(); g.setLineDash([]);
      E.txt(g, 'chance', vp.X(0.55), vp.Y(pi) - 4, C.muted);
      const rocPt = (i) => [vr.X(cumFP[i] / nn), vr.Y(cumTP[i] / np)], prPt = (i) => [vp.X(cumTP[i] / np), vp.Y(i === 0 ? 1 : cumTP[i] / (cumTP[i] + cumFP[i]))];
      const ki = Math.floor(k);
      g.fillStyle = C.blue; g.globalAlpha = 0.25; g.beginPath(); g.moveTo(vr.X(0), vr.Y(0)); for (let i = 0; i <= ki; i++) g.lineTo(...rocPt(i)); g.lineTo(vr.X(cumFP[ki] / nn), vr.Y(0)); g.closePath(); g.fill();
      g.fillStyle = C.purple; g.beginPath(); g.moveTo(vp.X(0), vp.Y(0)); for (let i = 1; i <= ki; i++) g.lineTo(...prPt(i)); g.lineTo(vp.X(cumTP[ki] / np), vp.Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1;
      g.lineWidth = 2.2; g.strokeStyle = C.blue; g.beginPath(); for (let i = 0; i <= ki; i++) (i ? g.lineTo(...rocPt(i)) : g.moveTo(...rocPt(i))); g.stroke();
      g.strokeStyle = C.purple; g.beginPath(); for (let i = 1; i <= ki; i++) (i > 1 ? g.lineTo(...prPt(i)) : g.moveTo(...prPt(i))); g.stroke();
      g.fillStyle = C.fg; g.beginPath(); g.arc(...rocPt(ki), 5, 0, 7); g.fill(); if (ki > 0) { g.beginPath(); g.arc(...prPt(ki), 5, 0, 7); g.fill(); }
      const tp = cumTP[ki], fp = cumFP[ki], rec = tp / np, prec = tp + fp ? tp / (tp + fp) : 1;
      q('#out').innerHTML = `<span class="m">examples passed ${ki} / ${items.length}</span>\nTP ${tp}  FP ${fp}  FN ${np - tp}  TN ${nn - fp}\nTPR(recall) <b>${fx(rec)}</b>  FPR <b>${fx(fp / nn)}</b>\nprecision <b>${fx(prec)}</b>\nAUC so far <span class="o">${fx(aucAt[ki])}</span>  (final ${fx(auc)})\nAP  so far <span class="o">${fx(apAt[ki])}</span>  (final ${fx(ap)})\nbaseline precision = ${fx(pi)}`;
      q('#cap').textContent = ki === 0 ? 'Threshold above every score: nothing is flagged — start of both curves.' : ki >= items.length ? 'Done. AUC is the shaded area under ROC; AP the area under PR. Move the imbalance slider: ROC hardly changes, PR drops.' : items[ki - 1].y ? 'A positive was passed → ROC steps up, PR precision stays high.' : 'A negative was passed → ROC steps right, PR precision falls.';
    }, { fps: 30 });
  };

  /* ======================= 3. regression metrics ======================= */
  AI.labs.regmetrics = function (root) {
    AI.labShell(root, 'Regression metrics lab', 'drag the points and the two line handles; residuals and every metric update live',
      `<canvas id="main" style="height:260px;touch-action:none;cursor:grab"></canvas><canvas id="res" style="height:110px"></canvas><canvas id="bars" style="height:140px"></canvas>`,
      `<div class="row"><button class="btn" id="fit">Fit least-squares line</button><button class="btn" id="out">Toggle outlier (point 5)</button><button class="btn" id="rst">Reset all</button></div>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#main')), c2 = AI.canvas(q('#res')), c3 = AI.canvas(q('#bars'));
    const P0 = [[1, 10], [2, 12], [3, 15], [4, 20], [5, 40]];
    let pts = P0.map((p) => p.slice()), key = '', vm = null, drag = null, tgt = null, cap = '', outlier = true, vt = 0;
    const lr = (a) => ML.linreg(a.map((p) => p[0]), a.map((p) => p[1]));
    let L = lr(P0.slice(0, 4)), a = L.w, b = L.b;
    const reset = () => { pts = P0.map((p) => p.slice()); outlier = true; L = lr(P0.slice(0, 4)); tgt = { a: L.w, b: L.b }; cap = 'Reset: the line fits the four tidy points; the fifth is an outlier.'; key = ''; };
    const ctrl = E.controls(root, { onStep: () => { vt = 0.35; }, onReset: reset });
    q('#fit').onclick = () => { const f = lr(pts); tgt = { a: f.w, b: f.b }; cap = 'The line glides to the least-squares fit: it minimises the sum of squared residuals, so R² is maximal.'; };
    q('#out').onclick = () => { outlier = !outlier; pts[4][1] = outlier ? 40 : 24; cap = outlier ? 'Outlier on: watch RMSE jump more than MAE.' : 'Outlier removed.'; key = ''; };
    q('#rst').onclick = reset;
    const yv = (x) => a * x + b;
    const view = (c) => { const ym = Math.max(50, ...pts.map((p) => p[1] + 8)); return AI.view(c, 0, 6, -5, ym, { l: 36, r: 12, t: 12, b: 24 }); };
    E.drag(q('#main'), (ev, px, py) => {
      if (!vm) return;
      if (ev === 'down') {
        const hs = [[0.5, yv(0.5), 'h0'], [5.5, yv(5.5), 'h1']]; let best = null, bd = 18;
        pts.forEach((p, i) => { const d = Math.hypot(vm.X(p[0]) - px, vm.Y(p[1]) - py); if (d < bd) { bd = d; best = i; } });
        hs.forEach((h) => { const d = Math.hypot(vm.X(h[0]) - px, vm.Y(h[1]) - py); if (d < bd) { bd = d; best = h[2]; } });
        drag = best; tgt = null;
      } else if (ev === 'move' && drag !== null) {
        const x = vm.ix(px), y = vm.iy(py);
        if (typeof drag === 'number') { pts[drag][0] = AI.clamp(x, 0.2, 5.8); pts[drag][1] = AI.clamp(y, -4, 90); }
        else { const x0 = drag === 'h0' ? 0.5 : 5.5, x1 = drag === 'h0' ? 5.5 : 0.5, y1 = yv(x1); a = (y1 - y) / (x1 - x0); b = y - a * x0; }
        key = '';
      } else if (ev === 'up') drag = null;
    });
    ro(c1, () => (key = '')); ro(c2, () => (key = '')); ro(c3, () => (key = ''));
    const metrics = (arr) => {
      const n = arr.length, e = arr.map((p) => p[1] - yv(p[0])), my = arr.reduce((s, p) => s + p[1], 0) / n;
      const sse = e.reduce((s, v) => s + v * v, 0), sst = arr.reduce((s, p) => s + (p[1] - my) ** 2, 0);
      return { e, mae: e.reduce((s, v) => s + Math.abs(v), 0) / n, rmse: Math.sqrt(sse / n), r2: 1 - sse / (sst || 1), mape: (100 / n) * arr.reduce((s, p, i) => s + Math.abs(e[i]) / Math.max(Math.abs(p[1]), 1e-9), 0) };
    };
    AI.loop(root, (dt) => {
      const step = ctrl.playing ? dt : vt; vt = 0;
      if (tgt && step > 0) { const k = ease(step, 1.6 * ctrl.speed); a += (tgt.a - a) * k; b += (tgt.b - b) * k; if (Math.abs(tgt.a - a) + Math.abs(tgt.b - b) < 1e-3) { a = tgt.a; b = tgt.b; tgt = null; } }
      const k = [a.toFixed(3), b.toFixed(3), pts.map((p) => p[0].toFixed(2) + p[1].toFixed(2)).join()].join('|');
      if (k === key) return; key = k;
      const C = AI.col(), m = metrics(pts), v = (vm = view(c1));
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h); AI.axes(g, v);
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); g.moveTo(v.X(0), v.Y(yv(0))); g.lineTo(v.X(6), v.Y(yv(6))); g.stroke();
      pts.forEach((p) => { g.strokeStyle = C.red; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(v.X(p[0]), v.Y(p[1])); g.lineTo(v.X(p[0]), v.Y(yv(p[0]))); g.stroke(); g.setLineDash([]); });
      pts.forEach((p, i) => { g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(p[0]), v.Y(p[1]), 7, 0, 7); g.fill(); g.strokeStyle = C.fg; g.lineWidth = 1; g.stroke(); E.txt(g, String(i + 1), v.X(p[0]) + 10, v.Y(p[1]) - 8, C.muted); });
      [0.5, 5.5].forEach((x) => { g.fillStyle = C.blue; g.fillRect(v.X(x) - 6, v.Y(yv(x)) - 6, 12, 12); });
      E.txt(g, 'orange = data (drag), blue handles = line (drag), red = residual y − ŷ', v.pad.l + 4, 10, C.muted);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const em = Math.max(5, ...m.e.map(Math.abs)) * 1.15, vr = AI.view(c2, 0, 6, -em, em, { l: 36, r: 12, t: 12, b: 20 });
      AI.axes(g, vr, { noGrid: true }); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(vr.X(0), vr.Y(0)); g.lineTo(vr.X(6), vr.Y(0)); g.stroke();
      pts.forEach((p, i) => { g.strokeStyle = m.e[i] >= 0 ? C.green : C.red; g.lineWidth = 3; g.beginPath(); g.moveTo(vr.X(p[0]), vr.Y(0)); g.lineTo(vr.X(p[0]), vr.Y(m.e[i])); g.stroke(); g.fillStyle = g.strokeStyle; g.beginPath(); g.arc(vr.X(p[0]), vr.Y(m.e[i]), 4, 0, 7); g.fill(); });
      E.txt(g, 'residuals e = y − ŷ (should look like shapeless noise around 0)', vr.pad.l + 4, 10, C.muted);
      g = c3.ctx; g.clearRect(0, 0, c3.w, c3.h);
      const mx = Math.max(10, m.rmse * 1.2, m.mae * 1.2);
      const rows = [['MAE', m.mae, mx, C.blue, fx(m.mae, 2)], ['RMSE', m.rmse, mx, C.purple, fx(m.rmse, 2)], ['R²', AI.clamp(m.r2, 0, 1), 1, C.green, fx(m.r2, 3)], ['MAPE', Math.min(m.mape, 100), 100, C.orange, fx(m.mape, 1) + '%']];
      rows.forEach(([n, val, mm, col, tx], i) => { const y = 8 + i * 31, w0 = Math.max(40, c3.w - 150); E.txt(g, n, 8, y + 15, C.fg); g.fillStyle = C.bg2; g.fillRect(52, y + 3, w0, 18); g.fillStyle = col; g.fillRect(52, y + 3, w0 * AI.clamp(val / mm, 0, 1), 18); E.txt(g, tx, 58 + w0, y + 16, C.fg, 'left', 'bold 12px sans-serif'); });
      const keep = pts.map((p, i) => [p, Math.abs(m.e[i])]).sort((x, y) => y[1] - x[1]).slice(1).map((x) => x[0]), m2 = metrics(keep);
      q('#o').innerHTML = `<span class="m">line ŷ = ${fx(a, 2)}·x + ${fx(b, 2)}</span>\nMAE   <b>${fx(m.mae)}</b>   RMSE <b>${fx(m.rmse)}</b>\nRMSE/MAE <span class="o">${fx(m.rmse / (m.mae || 1), 2)}</span>  (much above 1 → big misses)\nR²    <b>${fx(m.r2)}</b>   MAPE <b>${fx(m.mape, 1)}%</b>\nmean residual ${fx(m.e.reduce((s, v2) => s + v2, 0) / 5, 2)}  (bias)\n\n<span class="m">dropping the worst point:</span>\nMAE ${fx(m2.mae)} → with it x${fx(m.mae / (m2.mae || 1e-9), 1)}\nRMSE ${fx(m2.rmse)} → with it x${fx(m.rmse / (m2.rmse || 1e-9), 1)}`;
      q('#cap').textContent = cap || 'Drag the fifth point up and down: RMSE reacts much more strongly than MAE because it squares the miss.';
    }, { fps: 30 });
  };

  /* ======================= 4. cross-validation ======================= */
  AI.labs.cvlab = function (root) {
    AI.labShell(root, 'Cross-validation lab', 'each round trains a 3-nearest-neighbour model on the blue blocks and tests it on the orange ones (illustrative data)',
      `<canvas id="grid" style="height:280px"></canvas><canvas id="bars" style="height:150px"></canvas>`,
      `<label>Mode <select id="mode"><option value="kfold">k-fold (shuffled)</option><option value="strat">stratified k-fold</option><option value="group">group k-fold (5 groups)</option><option value="time">time-series (expanding window)</option></select></label>
       <label>Folds k <span class="v" id="kv"></span><input type="range" id="k" min="3" max="10" step="1" value="5"></label>
       <label>Class separation <span class="v" id="sv"></span><input type="range" id="sep" min="0.5" max="3" step="0.1" value="1.6"></label>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#grid')), c2 = AI.canvas(q('#bars'));
    const N = 20, DUR = [1.1, 1.1, 1.8, 0.7], TOT = DUR.reduce((a, b) => a + b, 0), ALL = [...Array(N).keys()];
    let data = [], folds = [], scores = [], round = 0, ph = 0, done = false, key = '', shown = [];
    const base = 15 / N;
    const mkData = () => {
      const r = AI.rng(5), sep = +q('#sep').value, perm = ALL.slice().sort(() => r() - 0.5), pos = new Set(perm.slice(0, 5)), off = Array.from({ length: 5 }, () => [AI.randn(r) * 0.9, AI.randn(r) * 0.9]);
      data = ALL.map((i) => { const y = pos.has(i) ? 1 : 0, g = Math.floor(i / 4); return { y, g, x: [(y ? sep / 2 : -sep / 2) + off[g][0] + 0.8 * AI.randn(r), off[g][1] + 0.8 * AI.randn(r)] }; });
    };
    const knn = (tr, i) => { const d = tr.map((j) => [Math.hypot(data[j].x[0] - data[i].x[0], data[j].x[1] - data[i].x[1]), data[j].y]).sort((a, b) => a[0] - b[0]).slice(0, 3); return d.reduce((s, a) => s + a[1], 0) * 2 > d.length ? 1 : 0; };
    const build = () => {
      const mode = q('#mode').value; let k = +q('#k').value; if (mode === 'group') k = Math.min(k, 5); if (mode === 'time') k = Math.min(k, 9);
      const r = AI.rng(9); mkData(); folds = [];
      if (mode === 'kfold') { const perm = ALL.slice().sort(() => r() - 0.5); for (let j = 0; j < k; j++) { const val = perm.filter((_, i) => i % k === j); folds.push({ val, train: ALL.filter((i) => !val.includes(i)) }); } }
      else if (mode === 'strat') { const sh = (a) => a.sort(() => r() - 0.5), P = sh(ALL.filter((i) => data[i].y)), Nn = sh(ALL.filter((i) => !data[i].y)), fo = new Array(N); let c = 0; P.concat(Nn).forEach((i) => (fo[i] = c++ % k)); for (let j = 0; j < k; j++) folds.push({ val: ALL.filter((i) => fo[i] === j), train: ALL.filter((i) => fo[i] !== j) }); }
      else if (mode === 'group') { const go = [0, 1, 2, 3, 4].sort(() => r() - 0.5); for (let j = 0; j < k; j++) { const gs = go.filter((_, i) => i % k === j); folds.push({ val: ALL.filter((i) => gs.includes(data[i].g)), train: ALL.filter((i) => !gs.includes(data[i].g)) }); } }
      else { const m = Math.floor(N / (k + 1)); for (let j = 0; j < k; j++) { const a = (j + 1) * m, b2 = j === k - 1 ? N : (j + 2) * m; folds.push({ val: ALL.filter((i) => i >= a && i < b2), train: ALL.filter((i) => i < a) }); } }
      folds.forEach((f) => { f.ok = f.val.map((i) => ((f.train.length >= 3 ? knn(f.train, i) : 0) === data[i].y ? 1 : 0)); f.score = f.ok.reduce((s, v) => s + v, 0) / f.val.length; });
      scores = folds.map((f) => f.score); round = 0; ph = 0; done = false; shown = scores.map(() => 0); key = '';
      q('#kv').textContent = k; q('#sv').textContent = q('#sep').value;
    };
    const phaseOf = () => { let t = Math.min(ph, TOT - 1e-6), i = 0; while (i < 3 && t >= DUR[i]) { t -= DUR[i]; i++; } return [i, t / DUR[i]]; };
    const advance = () => { round++; ph = 0; if (round >= folds.length) { done = true; round = folds.length - 1; ph = TOT; } };
    const ctrl = E.controls(root, { onStep: () => { if (done) return; const [i] = phaseOf(); let acc = 0; for (let j = 0; j <= i; j++) acc += DUR[j]; ph = acc + 1e-6; if (ph >= TOT) advance(); }, onReset: () => build() });
    ['#mode', '#k', '#sep'].forEach((s) => (q(s).oninput = () => { build(); ctrl.play(); }));
    q('#mode').onchange = q('#mode').oninput;
    ro(c1, () => (key = '')); ro(c2, () => (key = ''));
    build();
    AI.loop(root, (dt) => {
      if (ctrl.playing && !done) { ph += dt * ctrl.speed; if (ph >= TOT) advance(); }
      const K = folds.length, [pi, pf] = phaseOf();
      for (let j = 0; j < K; j++) { const tg = done || j < round || (j === round && ph >= TOT - 0.7 * 0.5 && pi === 3 && pf > 0.3) ? 1 : 0; shown[j] += (tg - shown[j]) * ease(dt, 4); }
      const k = [round, ph.toFixed(2), done, shown.map((v) => v.toFixed(2)).join(), q('#mode').value].join('|');
      if (k === key) return; key = k;
      const C = AI.col(); let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const left = 66, right = 52, cw = (c1.w - left - right) / N, rh = Math.min(30, (c1.h - 44) / K), top = 30;
      for (let i = 0; i < N; i++) E.txt(g, String(i + 1), left + i * cw + cw / 2, 22, data[i].g % 2 ? C.blue : C.muted, 'center', '10px sans-serif');
      E.txt(g, 'example →   (+ = positive class; number colour = group)', left, 10, C.muted);
      folds.forEach((f, j) => {
        const y = top + j * rh, active = j === round && !done, past = j < round || done, al = past ? 1 : active ? (pi === 0 ? pf : 1) : 0;
        E.txt(g, 'fold ' + (j + 1), left - 8, y + rh / 2 + 4, active ? C.fg : C.muted, 'right');
        for (let i = 0; i < N; i++) {
          const isV = f.val.includes(i), isT = f.train.includes(i), x = left + i * cw;
          g.fillStyle = C.bg2; g.fillRect(x + 1, y + 1, cw - 2, rh - 2);
          if (al > 0) { g.globalAlpha = (isV || isT ? 0.65 : 0.15) * al; g.fillStyle = isV ? C.orange : isT ? C.blue : C.muted; g.fillRect(x + 1, y + 1, cw - 2, rh - 2); g.globalAlpha = 1; }
          if (data[i].y) E.txt(g, '+', x + cw / 2, y + rh / 2 + 4, C.fg, 'center', 'bold 11px sans-serif');
          if (isV && (past || (active && pi >= 2))) { const t = f.val.indexOf(i), reveal = past || pi > 2 ? 99 : pf * f.val.length; if (reveal > t) E.txt(g, f.ok[t] ? '✓' : '✗', x + cw / 2, y + rh - 3, f.ok[t] ? C.green : C.red, 'center', 'bold 11px sans-serif'); }
        }
        if (past || (active && pi === 3)) E.txt(g, fx(f.score, 2), left + N * cw + 8, y + rh / 2 + 4, C.fg, 'left', 'bold 12px sans-serif');
        if (active) { g.strokeStyle = C.fg; g.lineWidth = 1.5; g.strokeRect(left - 0.5, y, cw * N + 1, rh); }
      });
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v = AI.view(c2, 0, K + 1, 0, 1.05, { l: 34, r: 12, t: 16, b: 24 }); AI.axes(g, v, {});
      const got = scores.filter((_, j) => shown[j] > 0.3), n = got.length, mean = n ? got.reduce((s, a) => s + a, 0) / n : 0, sd = n > 1 ? Math.sqrt(got.reduce((s, a) => s + (a - mean) ** 2, 0) / (n - 1)) : 0, bw = (v.pw / (K + 1)) * 0.6;
      scores.forEach((s, j) => { const h = s * shown[j]; g.fillStyle = C.blue; g.globalAlpha = 0.8; g.fillRect(v.X(j + 1) - bw / 2, v.Y(h), bw, v.Y(0) - v.Y(h)); g.globalAlpha = 1; E.txt(g, String(j + 1), v.X(j + 1), c2.h - 8, C.muted, 'center'); });
      if (n) { const hi = Math.min(1.05, mean + sd), lo = Math.max(0, mean - sd); g.fillStyle = C.orange; g.globalAlpha = 0.18; g.fillRect(v.X(0), v.Y(hi), v.pw, v.Y(lo) - v.Y(hi)); g.globalAlpha = 1; g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(v.X(0), v.Y(mean)); g.lineTo(v.X(K + 1), v.Y(mean)); g.stroke(); }
      g.strokeStyle = C.muted; g.lineWidth = 1; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(v.X(0), v.Y(base)); g.lineTo(v.X(K + 1), v.Y(base)); g.stroke(); g.setLineDash([]);
      E.txt(g, 'fold accuracy (bars), mean ± std (orange), always-negative baseline (dashed)', v.pad.l + 4, 11, C.muted);
      const se = n > 1 ? sd / Math.sqrt(n) : 0;
      q('#o').innerHTML = `<span class="m">folds done ${n} / ${K}</span>\nscores  ${scores.map((s, j) => (shown[j] > 0.3 ? fx(s, 2) : '·')).join('  ')}\nmean   <b>${fx(mean)}</b>\nstd    <b>${fx(sd)}</b>   SE = std/√k <span class="o">${fx(se)}</span>\nestimate  <span class="o">${fx(mean, 2)} ± ${fx(se, 2)}</span>`;
      const names = ['① assign: blue blocks become training data, orange the validation fold', '② train on the blue blocks only', '③ evaluate on the orange blocks: ✓ correct, ✗ wrong', '④ record this fold’s score, then rotate to the next fold'];
      q('#cap').textContent = done ? 'All folds used once. Report mean ± standard error (folds overlap, so the true uncertainty is a bit larger). Try another mode.' : `Round ${round + 1} of ${K} — ${names[pi]}.` + (q('#mode').value === 'time' ? ' Time-series: training only ever uses the past.' : '');
    }, { fps: 30 });
  };

  /* ======================= 5. hyper-parameter search ======================= */
  const LAND = {
    one: { name: 'only λ₁ matters', f: (x, y) => 0.2 + 0.7 * Math.exp(-(((x - 0.68) / 0.12) ** 2)) + 0.06 * Math.exp(-(((y - 0.35) / 0.35) ** 2)) },
    peaks: { name: 'two peaks (one narrow)', f: (x, y) => 0.15 + 0.62 * Math.exp(-((x - 0.25) ** 2 + (y - 0.75) ** 2) / 0.03) + 0.8 * Math.exp(-((x - 0.75) ** 2 + (y - 0.3) ** 2) / 0.012) },
    ridge: { name: 'narrow diagonal ridge', f: (x, y) => 0.15 + 0.75 * Math.exp(-((x - y) ** 2) / 0.008) * Math.exp(-((x + y - 1.1) ** 2) / 0.5) },
  };
  AI.labs.searchlab = function (root) {
    AI.labShell(root, 'Search lab: grid vs random vs Bayesian', 'same budget, same landscape; one trial per step on each side (illustrative landscape, brighter = better validation score)',
      `<canvas id="pan" style="height:240px"></canvas><canvas id="cur" style="height:140px"></canvas>`,
      `<label>Landscape <select id="land">${Object.entries(LAND).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}</select></label>
       <label>Trial budget <select id="bud"><option value="9">9</option><option value="16">16</option><option value="25" selected>25</option><option value="36">36</option></select></label>
       <label>Measurement noise σ <span class="v" id="nv"></span><input type="range" id="noise" min="0" max="0.1" step="0.005" value="0.01"></label>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#pan')), c2 = AI.canvas(q('#cur'));
    const GR = 28, off = document.createElement('canvas'); off.width = off.height = GR;
    let T = { grid: [], rand: [], bayes: [] }, k = 0, age = 0, key = '', seed = 1, hmKey = '', tick = AI.ticker(), lastEI = null;
    const chol = (A) => { const n = A.length, L = A.map(() => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = A[i][j]; for (let m = 0; m < j; m++) s -= L[i][m] * L[j][m]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j]; } return L; };
    const solveL = (L, b) => { const n = b.length, x = new Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let j = 0; j < i; j++) s -= L[i][j] * x[j]; x[i] = s / L[i][i]; } return x; };
    const solveLT = (L, b) => { const n = b.length, x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j]; x[i] = s / L[i][i]; } return x; };
    const build = () => {
      const f = LAND[q('#land').value].f, B = +q('#bud').value, sg = +q('#noise').value, r = AI.rng(seed * 17 + 3), rn = AI.rng(seed * 31 + 7), g = Math.round(Math.sqrt(B));
      const ev = (x, y) => ({ x, y, t: f(x, y), o: f(x, y) + sg * AI.randn(rn) });
      T = { grid: [], rand: [], bayes: [] };
      for (let i = 0; i < B; i++) T.grid.push(ev(((i % g) + 0.5) / g, (Math.floor(i / g) + 0.5) / g));
      for (let i = 0; i < B; i++) T.rand.push(ev(r(), r()));
      const kern = (a, b) => Math.exp(-((a.x - b.x) ** 2 + (a.y - b.y) ** 2) / (2 * 0.17 ** 2));
      for (let i = 0; i < B; i++) {
        if (i < 3) { T.bayes.push(ev(r(), r())); continue; }
        const P = T.bayes, n = P.length, mu = P.reduce((s, p) => s + p.o, 0) / n, sd = Math.sqrt(P.reduce((s, p) => s + (p.o - mu) ** 2, 0) / n) || 1, z = P.map((p) => (p.o - mu) / sd);
        const K = P.map((a, ia) => P.map((b, ib) => kern(a, b) + (ia === ib ? (sg / sd) ** 2 + 1e-3 : 0))), L = chol(K), al = solveLT(L, solveL(L, z)), fb = Math.max(...z);
        let best = null, bv = -1;
        for (let a = 0; a < 22; a++) for (let b2 = 0; b2 < 22; b2++) {
          const c = { x: (a + 0.5) / 22, y: (b2 + 0.5) / 22 }, ks = P.map((p) => kern(c, p)), m = ks.reduce((s, v, j) => s + v * al[j], 0), w = solveL(L, ks), s2 = Math.max(1e-9, 1 - w.reduce((s, v) => s + v * v, 0)), s = Math.sqrt(s2), zz = (m - fb) / s, ei = (m - fb) * Phi(zz) + s * pdf(zz);
          if (ei > bv) { bv = ei; best = c; }
        }
        const e = ev(best.x, best.y); e.ei = bv; T.bayes.push(e);
      }
      k = 0; age = 0; key = ''; hmKey = '';
      q('#nv').textContent = sg;
    };
    const ctrl = E.controls(root, { onStep: () => { k = Math.min(+q('#bud').value, k + 1); age = 0; key = ''; }, onReset: () => { seed++; build(); } });
    ['#land', '#bud', '#noise'].forEach((s) => (q(s).oninput = () => { build(); ctrl.play(); }));
    q('#land').onchange = q('#land').oninput; q('#bud').onchange = q('#bud').oninput;
    ro(c1, () => (key = '')); ro(c2, () => (key = ''));
    build();
    const bestOf = (arr, n) => { let b = null; for (let i = 0; i < n; i++) if (!b || arr[i].o > b.o) b = arr[i]; return b; };
    AI.loop(root, (dt) => {
      const B = +q('#bud').value;
      if (ctrl.playing && k < B) { const n = tick(dt, 1.1 * ctrl.speed); if (n) { k = Math.min(B, k + n); age = 0; } }
      age += dt;
      const kk = [k, age < 1 ? age.toFixed(2) : 'x', q('#land').value, B].join('|'); if (kk === key) return; key = kk;
      const C = AI.col(), f = LAND[q('#land').value].f;
      if (hmKey !== q('#land').value) { hmKey = q('#land').value; const g0 = off.getContext('2d'), im = g0.createImageData(GR, GR), a = AI.rgb(C.bg), b = AI.rgb(C.green); for (let j = 0; j < GR; j++) for (let i = 0; i < GR; i++) { const t = AI.clamp(f((i + 0.5) / GR, 1 - (j + 0.5) / GR), 0, 1) ** 1.3; for (let c = 0; c < 3; c++) im.data[(j * GR + i) * 4 + c] = a[c] + (b[c] - a[c]) * t * 0.85; im.data[(j * GR + i) * 4 + 3] = 255; } g0.putImageData(im, 0, 0); }
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const pw = (c1.w - 24) / 3, names = [['grid', 'Grid'], ['rand', 'Random'], ['bayes', 'Bayesian (EI)']], ph = c1.h - 44, gs = Math.round(Math.sqrt(B));
      names.forEach(([key2, nm], pi) => {
        const x0 = 6 + pi * (pw + 6), y0 = 22, w = pw, h = ph, X = (x) => x0 + x * w, Y = (y) => y0 + (1 - y) * h;
        g.imageSmoothingEnabled = true; g.drawImage(off, x0, y0, w, h); g.strokeStyle = C.border; g.strokeRect(x0, y0, w, h);
        E.txt(g, nm, x0, 14, C.fg, 'left', 'bold 12px sans-serif');
        const arr = T[key2], b = bestOf(arr, k);
        for (let i = 0; i < k; i++) {
          const p = arr[i], newest = i === k - 1 && age < 1;
          g.fillStyle = C.orange; g.strokeStyle = C.fg; g.lineWidth = 1; g.beginPath(); g.arc(X(p.x), Y(p.y), newest ? 4 + 5 * (1 - age) : 4, 0, 7); g.fill(); g.stroke();
          if (newest) { g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); g.arc(X(p.x), Y(p.y), 6 + 14 * age, 0, 7); g.globalAlpha = 1 - age; g.stroke(); g.globalAlpha = 1; }
        }
        if (b) { g.strokeStyle = C.yellow; g.lineWidth = 2.5; g.beginPath(); g.arc(X(b.x), Y(b.y), 8, 0, 7); g.stroke(); }
        E.txt(g, 'λ₁ →', x0 + w - 4, y0 + h + 14, C.muted, 'right');
      });
      /* curve */
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const v = AI.view(c2, 0, B, 0, 1, { l: 34, r: 12, t: 16, b: 26 }); AI.axes(g, v, {});
      const cols = [C.blue, C.orange, C.purple];
      names.forEach(([key2], pi) => { g.strokeStyle = cols[pi]; g.lineWidth = 2.4; g.beginPath(); let b = -9; for (let i = 0; i < k; i++) { b = Math.max(b, T[key2][i].o); (i ? g.lineTo : g.moveTo).call(g, v.X(i + 1), v.Y(AI.clamp(b, 0, 1))); } g.stroke(); });
      E.txt(g, 'best score found so far vs trials', v.pad.l + 4, 11, C.muted);
      [['Grid', C.blue], ['Random', C.orange], ['Bayesian', C.purple]].forEach(([n, c], i) => { g.fillStyle = c; g.fillRect(v.pad.l + 10 + i * 78, c2.h - 11, 10, 4); E.txt(g, n, v.pad.l + 24 + i * 78, c2.h - 6, C.muted); });
      const bb = names.map(([k2]) => bestOf(T[k2], k));
      q('#o').innerHTML = `<span class="m">trial ${k} / ${B}</span>\nbest found (observed score)\n grid     <b>${bb[0] ? fx(bb[0].o) : '–'}</b>   distinct λ₁ tried: ${Math.min(k, gs)}\n random   <b>${bb[1] ? fx(bb[1].o) : '–'}</b>   distinct λ₁ tried: ${k}\n bayesian <b>${bb[2] ? fx(bb[2].o) : '–'}</b>   distinct λ₁ tried: ${k}\ntrue peak of the landscape ≈ ${fx(Math.max(...Array.from({ length: 400 }, (_, i) => f(((i % 20) + 0.5) / 20, (Math.floor(i / 20) + 0.5) / 20))), 2)}`;
      q('#cap').textContent = k === 0 ? 'Three searches share the same budget. Grid walks a lattice, random draws anywhere, Bayesian fits a surrogate to the points so far and picks the next trial where expected improvement is highest.' : k < 4 ? 'Early trials: Bayesian search starts with 3 random points to have something to model.' : k >= B ? 'Budget spent. Yellow ring = best trial of each method. Change the landscape or the noise and restart.' : `Trial ${k}: ${T.bayes[k - 1].ei !== undefined ? 'the Bayesian point has the highest expected improvement under its surrogate' : 'exploring'} — compare how many different λ₁ values grid vs random have already tested.`;
    }, { fps: 30 });
  };

  /* ======================= 6. learning / validation curves ======================= */
  AI.labs.learncurve = function (root) {
    AI.labShell(root, 'Diagnosis lab: learning curve & validation curve', 'polynomial models of degree d fitted to noisy data, averaged over 6 resamples (illustrative, computed live)',
      `<canvas id="fit" style="height:210px"></canvas><canvas id="lc" style="height:190px"></canvas><canvas id="vc" style="height:150px"></canvas>`,
      `<label>Model complexity (polynomial degree) <span class="v" id="dv"></span><input type="range" id="deg" min="1" max="12" step="1" value="3"></label>
       <label>Label noise σ <span class="v" id="nv"></span><input type="range" id="noise" min="0.1" max="0.5" step="0.05" value="0.25"></label>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#fit')), c2 = AI.canvas(q('#lc')), c3 = AI.canvas(q('#vc'));
    const SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 200], R = 6, MAXD = 12, f = (x) => 0.8 * Math.sin(4.5 * x) + 0.2 * x;
    let tab = [], si = 0, key = '', coefCache = {}, vx = [], vy = [];
    const basis = (x, d) => { const t = [1, x]; for (let j = 2; j <= d; j++) t.push(2 * x * t[j - 1] - t[j - 2]); return t.slice(0, d + 1); };
    const fitPoly = (xs, ys, d) => {
      const m = d + 1, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
      xs.forEach((x, i) => { const b = basis(x, d); for (let r = 0; r < m; r++) { for (let c = 0; c < m; c++) A[r][c] += b[r] * b[c]; A[r][m] += b[r] * ys[i]; } });
      for (let r = 0; r < m; r++) A[r][r] += 1e-8;
      for (let c = 0; c < m; c++) { let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r; [A[c], A[p]] = [A[p], A[c]]; for (let r = c + 1; r < m; r++) { const fct = A[r][c] / A[c][c]; for (let k2 = c; k2 <= m; k2++) A[r][k2] -= fct * A[c][k2]; } }
      const w = new Array(m).fill(0); for (let r = m - 1; r >= 0; r--) { let s = A[r][m]; for (let c = r + 1; c < m; c++) s -= A[r][c] * w[c]; w[r] = s / A[r][r]; } return w;
    };
    const pred = (w, x) => { const b = basis(x, w.length - 1); return b.reduce((s, v, i) => s + v * w[i], 0); };
    const mse = (w, xs, ys) => xs.reduce((s, x, i) => s + (ys[i] - pred(w, x)) ** 2, 0) / xs.length;
    const dataset = (n, rep, sg) => { const r = AI.rng(rep * 977 + n), xs = [], ys = []; for (let i = 0; i < n; i++) { const x = r() * 2 - 1; xs.push(x); ys.push(f(x) + sg * AI.randn(r)); } return [xs, ys]; };
    const build = () => {
      const sg = +q('#noise').value, r = AI.rng(4242); vx = []; vy = []; for (let i = 0; i < 300; i++) { const x = r() * 2 - 1; vx.push(x); vy.push(f(x) + sg * AI.randn(r)); }
      tab = []; coefCache = {};
      for (let d = 1; d <= MAXD; d++) { tab[d] = SIZES.map((n) => { let tr = 0, va = 0; for (let rep = 0; rep < R; rep++) { const [xs, ys] = dataset(n, rep, sg), w = fitPoly(xs, ys, d); tr += mse(w, xs, ys) / R; va += mse(w, vx, vy) / R; } return { tr, va }; }); }
      si = 0; key = ''; q('#nv').textContent = sg;
    };
    const ctrl = E.controls(root, { onStep: () => { si = Math.min(SIZES.length - 1, Math.floor(si) + 1); }, onReset: () => { si = 0; } });
    q('#deg').oninput = () => (key = ''); q('#noise').oninput = () => { build(); ctrl.play(); };
    ro(c1, () => (key = '')); ro(c2, () => (key = '')); ro(c3, () => (key = ''));
    build();
    AI.loop(root, (dt) => {
      if (ctrl.playing && si < SIZES.length - 1) si = Math.min(SIZES.length - 1, si + dt * 0.55 * ctrl.speed);
      const d = +q('#deg').value, i0 = Math.floor(si), fr = si - i0, kk = [si.toFixed(2), d, q('#noise').value].join('|'); if (kk === key) return; key = kk;
      q('#dv').textContent = d;
      const C = AI.col(), sg = +q('#noise').value, floor = sg * sg;
      /* fit panel: rep 0 at current size, points appear as n grows */
      const n = Math.round(AI.lerp(SIZES[i0], SIZES[Math.min(i0 + 1, SIZES.length - 1)], fr)), [xs, ys] = dataset(Math.max(n, 3), 0, sg), w = fitPoly(xs, ys, d);
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = AI.view(c1, -1, 1, -1.6, 1.6, { l: 34, r: 12, t: 16, b: 22 }); AI.axes(g, v, {});
      g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); for (let x = -1; x <= 1.001; x += 0.02) (x < -0.99 ? g.moveTo : g.lineTo).call(g, v.X(x), v.Y(f(x))); g.stroke(); g.setLineDash([]);
      xs.forEach((x, i) => { g.fillStyle = C.orange; g.globalAlpha = 0.85; g.beginPath(); g.arc(v.X(x), v.Y(ys[i]), 3, 0, 7); g.fill(); }); g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.lineWidth = 2.6; g.beginPath(); for (let x = -1; x <= 1.001; x += 0.02) { const y = AI.clamp(pred(w, x), -1.6, 1.6); (x < -0.99 ? g.moveTo : g.lineTo).call(g, v.X(x), v.Y(y)); } g.stroke();
      E.txt(g, `n = ${xs.length} training points · blue = fitted degree-${d} model · dashed = truth`, v.pad.l + 4, 11, C.muted);
      /* learning curve */
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const ymax = Math.min(1.2, Math.max(floor * 4, ...SIZES.map((_, i) => tab[d][i].va)) * 1.1), vl = AI.view(c2, Math.log(SIZES[0]), Math.log(SIZES[SIZES.length - 1]), 0, ymax, { l: 40, r: 12, t: 16, b: 26 });
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(vl.pad.l, vl.pad.t, vl.pw, vl.ph);
      SIZES.forEach((s) => E.txt(g, String(s), vl.X(Math.log(s)), c2.h - 12, C.muted, 'center', '10px sans-serif'));
      [0, 0.5, 1].forEach((t) => E.txt(g, fx(ymax * t, 2), vl.pad.l - 4, vl.Y(ymax * t) + 4, C.muted, 'right', '10px sans-serif'));
      g.setLineDash([4, 4]); g.strokeStyle = C.green; g.beginPath(); g.moveTo(vl.X(Math.log(SIZES[0])), vl.Y(floor)); g.lineTo(vl.X(Math.log(SIZES[SIZES.length - 1])), vl.Y(floor)); g.stroke(); g.setLineDash([]);
      E.txt(g, 'noise floor σ²', vl.pad.l + vl.pw - 4, vl.Y(floor) - 4, C.green, 'right');
      const pts = (key2) => { const a = []; for (let i = 0; i <= i0; i++) a.push([Math.log(SIZES[i]), tab[d][i][key2]]); if (i0 < SIZES.length - 1 && fr > 0) a.push([AI.lerp(Math.log(SIZES[i0]), Math.log(SIZES[i0 + 1]), fr), AI.lerp(tab[d][i0][key2], tab[d][i0 + 1][key2], fr)]); return a; };
      [['tr', C.blue, 'training error'], ['va', C.red, 'validation error']].forEach(([k2, col, nm], j) => { g.strokeStyle = col; g.lineWidth = 2.4; g.beginPath(); pts(k2).forEach(([x, y], i) => (i ? g.lineTo(vl.X(x), vl.Y(Math.min(y, ymax))) : g.moveTo(vl.X(x), vl.Y(Math.min(y, ymax))))); g.stroke(); E.txt(g, nm, vl.pad.l + 8 + j * 120, vl.pad.t + 14, col); });
      E.txt(g, 'learning curve: error vs training-set size', vl.pad.l + vl.pw / 2, 11, C.muted, 'center');
      /* validation curve at current size */
      g = c3.ctx; g.clearRect(0, 0, c3.w, c3.h);
      const ii = Math.min(i0 + (fr > 0.5 ? 1 : 0), SIZES.length - 1), vals = []; for (let dd = 1; dd <= MAXD; dd++) vals.push(tab[dd][ii]);
      const ym2 = Math.min(1.2, Math.max(floor * 4, ...vals.map((x) => x.va)) * 1.05), vv = AI.view(c3, 1, MAXD, 0, ym2, { l: 40, r: 12, t: 16, b: 24 });
      g.strokeStyle = C.border; g.strokeRect(vv.pad.l, vv.pad.t, vv.pw, vv.ph);
      for (let dd = 1; dd <= MAXD; dd++) E.txt(g, String(dd), vv.X(dd), c3.h - 9, dd === d ? C.fg : C.muted, 'center', '10px sans-serif');
      [['tr', C.blue], ['va', C.red]].forEach(([k2, col]) => { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); vals.forEach((o, i) => (i ? g.lineTo : g.moveTo).call(g, vv.X(i + 1), vv.Y(Math.min(o[k2], ym2)))); g.stroke(); });
      g.strokeStyle = C.fg; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(vv.X(d), vv.pad.t); g.lineTo(vv.X(d), vv.pad.t + vv.ph); g.stroke(); g.setLineDash([]);
      E.txt(g, `validation curve at n = ${SIZES[ii]}: error vs degree`, vv.pad.l + vv.pw / 2, 11, C.muted, 'center');
      /* diagnosis */
      const cur = tab[d][ii], gap = cur.va - cur.tr, bias = cur.tr - floor;
      let diag, act;
      if (gap > Math.max(0.6 * floor, 0.5 * cur.tr) && gap > bias * 0.5) { diag = 'HIGH VARIANCE (overfitting)'; act = 'Gap is large: add data / augmentation, lower the degree or add regularisation, stop earlier.'; }
      else if (bias > 0.6 * floor) { diag = 'HIGH BIAS (underfitting)'; act = 'Both errors are high and close: more data will not help; raise complexity, add features, reduce regularisation.'; }
      else { diag = 'GOOD FIT'; act = 'Training error near the noise floor, small gap: remaining error is mostly irreducible noise.'; }
      const best = vals.reduce((b2, o, i) => (o.va < vals[b2 - 1].va ? i + 1 : b2), 1);
      q('#o').innerHTML = `<span class="m">training set size n = ${SIZES[ii]}</span>\ntrain error <b>${fx(cur.tr)}</b>\nvalid error <b>${fx(cur.va)}</b>\nnoise floor σ² <b>${fx(floor)}</b>\navoidable bias ≈ ${fx(bias)}   gap ≈ ${fx(gap)}\n<span class="o">${diag}</span>\nbest degree at this n: ${best}`;
      q('#cap').textContent = si < SIZES.length - 1 ? `Watching the data grow (n → ${SIZES[SIZES.length - 1]}). ${act}` : act;
    }, { fps: 30 });
  };

  /* ======================= 7. calibration ======================= */
  AI.labs.calibration = function (root) {
    AI.labShell(root, 'Calibration lab: reliability diagram', 'a model that ranks well but is over-confident; slide a Platt / isotonic correction (fitted on one half, evaluated on the other) onto the diagonal (illustrative data)',
      `<canvas id="rel" style="height:300px"></canvas><canvas id="map" style="height:150px"></canvas>`,
      `<label>Over-confidence (1 = calibrated) <span class="v" id="kv"></span><input type="range" id="k" min="0.4" max="3" step="0.05" value="2.2"></label>
       <label>Bias (shifts probabilities) <span class="v" id="bv"></span><input type="range" id="b" min="-1.5" max="1.5" step="0.1" value="0"></label>
       <label>Correction <select id="meth"><option value="platt">Platt scaling (sigmoid)</option><option value="iso">Isotonic regression</option></select></label>
       <label>Correction strength s <span class="v" id="sv"></span><input type="range" id="s" min="0" max="1" step="0.01" value="0"></label>
       <div class="row"><button class="btn" id="apply">Apply correction (animate)</button></div>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#rel')), c2 = AI.canvas(q('#map'));
    const M = 2000, r = AI.rng(77), Q = [], Y = [], Z = [];
    for (let i = 0; i < M; i++) { const z = 1.6 * AI.randn(r); Z.push(z); const p = sig(z); Q.push(p); Y.push(r() < p ? 1 : 0); }
    let ph = [], cal = null, key = '', s = 0, anim = 0, vt = 0;
    const build = () => {
      const k = +q('#k').value, b = +q('#b').value; ph = Q.map((p, i) => sig(k * Z[i] + b));
      const fit = [], ev = []; for (let i = 0; i < M; i++) (i % 2 ? ev : fit).push(i);
      if (q('#meth').value === 'platt') {
        let a = 1, c = 0; for (let it = 0; it < 30; it++) { let g0 = 0, g1 = 0, h00 = 1e-6, h01 = 0, h11 = 1e-6; fit.forEach((i) => { const x = logit(ph[i]), p = sig(a * x + c), e = p - Y[i], w = p * (1 - p); g0 += e * x; g1 += e; h00 += w * x * x; h01 += w * x; h11 += w; }); const det = h00 * h11 - h01 * h01 || 1e-9; a -= (h11 * g0 - h01 * g1) / det; c -= (-h01 * g0 + h00 * g1) / det; }
        cal = { f: (p) => sig(a * logit(p) + c), desc: `p' = σ(${fx(a, 2)}·logit(p̂) + ${fx(c, 2)})` };
      } else {
        const srt = fit.slice().sort((i, j) => ph[i] - ph[j]), bl = []; srt.forEach((i) => { bl.push({ w: 1, m: Y[i], lo: ph[i], hi: ph[i] }); while (bl.length > 1 && bl[bl.length - 2].m > bl[bl.length - 1].m) { const b2 = bl.pop(), b1 = bl.pop(); bl.push({ w: b1.w + b2.w, m: (b1.m * b1.w + b2.m * b2.w) / (b1.w + b2.w), lo: b1.lo, hi: b2.hi }); } });
        cal = { f: (p) => { let lo = 0, hi = bl.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (bl[m].lo <= p) lo = m; else hi = m - 1; } return AI.clamp(bl[lo].m, 0.001, 0.999); }, desc: `staircase of ${bl.length} steps (pool-adjacent-violators)` };
      }
      ev.length && (build.ev = ev); key = '';
      q('#kv').textContent = k; q('#bv').textContent = b;
    };
    const auc = (sc, idx) => { const o = idx.slice().sort((i, j) => sc[i] - sc[j]); let rk = 0, np = 0, sum = 0; for (let i = 0; i < o.length;) { let j = i; while (j < o.length && sc[o[j]] === sc[o[i]]) j++; const avg = (i + j + 1) / 2; for (let t = i; t < j; t++) if (Y[o[t]]) { sum += avg; np++; } i = j; } const nn = o.length - np; return (sum - (np * (np + 1)) / 2) / (np * nn); };
    const ctrl = E.controls(root, { onStep: () => { vt = 0.4; anim = 1; }, onReset: () => { s = 0; anim = 0; q('#s').value = 0; key = ''; } });
    q('#apply').onclick = () => { s = 0; q('#s').value = 0; anim = 1; ctrl.play(); };
    ['#k', '#b', '#meth'].forEach((x) => (q(x).oninput = () => build())); q('#meth').onchange = () => build();
    q('#s').oninput = () => { anim = 0; s = +q('#s').value; key = ''; };
    ro(c1, () => (key = '')); ro(c2, () => (key = ''));
    build();
    AI.loop(root, (dt) => {
      const step = ctrl.playing ? dt : vt; vt = 0;
      if (anim && step > 0) { s = Math.min(1, s + step * 0.3 * ctrl.speed); q('#s').value = s; if (s >= 1) anim = 0; }
      const kk = [s.toFixed(3), q('#k').value, q('#b').value, q('#meth').value].join('|'); if (kk === key) return; key = kk;
      q('#sv').textContent = fx(s, 2);
      const C = AI.col(), ev = build.ev, B = 10, cnt = new Array(B).fill(0), sp = new Array(B).fill(0), sy = new Array(B).fill(0), pc = new Array(M);
      let brier = 0, ll = 0;
      ev.forEach((i) => { const p = AI.clamp((1 - s) * ph[i] + s * cal.f(ph[i]), 0.001, 0.999); pc[i] = p; const b = Math.min(B - 1, Math.floor(p * B)); cnt[b]++; sp[b] += p; sy[b] += Y[i]; brier += (p - Y[i]) ** 2; ll -= Y[i] ? Math.log(p) : Math.log(1 - p); });
      const n = ev.length; let ece = 0; for (let b = 0; b < B; b++) if (cnt[b]) ece += (cnt[b] / n) * Math.abs(sp[b] / cnt[b] - sy[b] / cnt[b]);
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const v = AI.view(c1, 0, 1, 0, 1, { l: 40, r: 14, t: 16, b: 34 }); AI.axes(g, v, {}); E.axisLabel(g, c1, v, 'predicted probability', 'observed frequency');
      g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(v.X(0), v.Y(0)); g.lineTo(v.X(1), v.Y(1)); g.stroke(); g.setLineDash([]);
      const mx = Math.max(...cnt, 1);
      for (let b = 0; b < B; b++) { g.fillStyle = C.blue; g.globalAlpha = 0.3; const h = (cnt[b] / mx) * 40; g.fillRect(v.X(b / B) + 2, v.Y(0) - h, v.pw / B - 4, h); g.globalAlpha = 1; }
      const pts = []; for (let b = 0; b < B; b++) if (cnt[b] > 3) pts.push([sp[b] / cnt[b], sy[b] / cnt[b], cnt[b]]);
      g.strokeStyle = C.purple; g.lineWidth = 2.2; g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)))); g.stroke();
      pts.forEach(([x, y, c]) => { g.strokeStyle = C.red; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(x), v.Y(x)); g.lineTo(v.X(x), v.Y(y)); g.stroke(); g.fillStyle = C.purple; g.beginPath(); g.arc(v.X(x), v.Y(y), 3 + 5 * Math.sqrt(c / mx), 0, 7); g.fill(); });
      E.txt(g, 'below diagonal = over-confident · above = under-confident · bars = how many predictions per bin', v.pad.l + 4, 10, C.muted);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const vm = AI.view(c2, 0, 1, 0, 1, { l: 40, r: 14, t: 16, b: 22 }); AI.axes(g, vm, {});
      g.strokeStyle = C.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(vm.X(0), vm.Y(0)); g.lineTo(vm.X(1), vm.Y(1)); g.stroke(); g.setLineDash([]);
      g.strokeStyle = C.orange; g.lineWidth = 2.4; g.beginPath(); for (let i = 0; i <= 100; i++) { const x = i / 100, y = (1 - s) * x + s * cal.f(Math.min(Math.max(x, 0.0005), 0.9995)); (i ? g.lineTo : g.moveTo).call(g, vm.X(x), vm.Y(y)); } g.stroke();
      E.txt(g, 'correction map: raw score → corrected probability (orange)', vm.pad.l + 4, 11, C.muted);
      const idxs = ev; q('#o').innerHTML = `<span class="m">${cal.desc}</span>\nBrier   <b>${fx(brier / n, 4)}</b>\nECE     <b>${fx(ece, 4)}</b>\nlog-loss <b>${fx(ll / n, 4)}</b>\nAUC     <b>${fx(auc(pc, idxs), 4)}</b>  (rank unchanged)`;
      q('#cap').textContent = s === 0 ? 'Raw model: the purple reliability curve sags below the diagonal — it says 90% where reality is lower. Press "Apply correction".' : s < 1 ? 'The correction is blended in: the purple curve slides toward the diagonal while the ranking (AUC) stays the same.' : 'Fully corrected: points lie on the diagonal; Brier score and ECE dropped. Evaluated on data the calibrator never saw.';
    }, { fps: 30 });
  };

  /* ======================= 8. fairness ======================= */
  AI.labs.fairness = function (root) {
    AI.labShell(root, 'Fairness lab: two groups, group thresholds', 'group B’s scores are shifted down; choose thresholds and watch which fairness gap closes and what it costs (illustrative model, d′ = 2)',
      `<canvas id="dist" style="height:270px;touch-action:none"></canvas><canvas id="bars" style="height:150px"></canvas><canvas id="trade" style="height:150px"></canvas>`,
      `<label>Score shift of group B <span class="v" id="shv"></span><input type="range" id="sh" min="0" max="2" step="0.1" value="0.8"></label>
       <label>Base rate of group B (A = 60%) <span class="v" id="bbv"></span><input type="range" id="bb" min="10" max="60" step="5" value="30"></label>
       <label>Threshold A <span class="v" id="tav"></span><input type="range" id="ta" min="-2" max="4" step="0.05" value="1"></label>
       <label>Threshold B <span class="v" id="tbv"></span><input type="range" id="tb" min="-2" max="4" step="0.05" value="1"></label>
       <label style="flex-direction:row;gap:8px;align-items:center"><input type="checkbox" id="lock" checked> one shared threshold</label>
       <div class="row"><button class="btn" id="mxa">Max accuracy (shared)</button><button class="btn" id="mdp">Match selection rate</button><button class="btn" id="meo">Match TPR</button></div>
       ${E.bar()}
       <div class="readout" id="o"></div><div class="caption" id="cap"></div>`);
    const q = (s) => AI.$(s, root), c1 = AI.canvas(q('#dist')), c2 = AI.canvas(q('#bars')), c3 = AI.canvas(q('#trade'));
    let ta = 1, tb = 1, tgtB = null, key = '', cap = '', tradeKey = '', trade = [];
    const P = () => ({ sh: +q('#sh').value, piA: 0.6, piB: +q('#bb').value / 100 });
    const grp = (p, g, t) => { const pi = g ? p.piB : p.piA, sh = g ? p.sh : 0, tpr = 1 - Phi(t - (2 - sh)), fpr = 1 - Phi(t + sh), TP = pi * tpr, FN = pi * (1 - tpr), FP = (1 - pi) * fpr, TN = (1 - pi) * (1 - fpr); return { pi, tpr, fpr, TP, FN, FP, TN, sel: TP + FP, prec: TP / (TP + FP || 1) }; };
    const acc = (p, a, b) => { const A = grp(p, 0, a), B = grp(p, 1, b); return (A.TP + A.TN + B.TP + B.TN) / 2; };
    const bis = (fn, target) => { let lo = -3, hi = 5; for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (fn(m) > target) lo = m; else hi = m; } return (lo + hi) / 2; };
    const setTA = (v) => { ta = v; q('#ta').value = v; if (q('#lock').checked) { tb = v; q('#tb').value = v; } };
    const ctrl = E.controls(root, { onStep: () => { if (tgtB !== null) { tb = tgtB; tgtB = null; q('#tb').value = tb; } }, onReset: () => { q('#lock').checked = true; setTA(1); tgtB = null; cap = ''; } });
    q('#ta').oninput = () => { tgtB = null; setTA(+q('#ta').value); }; q('#tb').oninput = () => { q('#lock').checked = false; tgtB = null; tb = +q('#tb').value; };
    q('#lock').onchange = () => { if (q('#lock').checked) setTA(ta); };
    q('#sh').oninput = q('#bb').oninput = () => { key = ''; };
    q('#mxa').onclick = () => { const p = P(); let bt = 0, bv = -1; for (let t = -2; t <= 4; t += 0.02) { const a = acc(p, t, t); if (a > bv) { bv = a; bt = t; } } q('#lock').checked = true; setTA(bt); tgtB = null; cap = 'Shared threshold that maximises overall accuracy: usually it leaves group B with fewer selections and a lower TPR.'; };
    q('#mdp').onclick = () => { const p = P(); q('#lock').checked = false; tgtB = bis((t) => grp(p, 1, t).sel, grp(p, 0, ta).sel); cap = 'Group B’s threshold slides until selection rates are equal (demographic parity).'; };
    q('#meo').onclick = () => { const p = P(); q('#lock').checked = false; tgtB = bis((t) => grp(p, 1, t).tpr, grp(p, 0, ta).tpr); cap = 'Group B’s threshold slides until true-positive rates are equal (equal opportunity).'; };
    ro(c1, () => (key = '')); ro(c2, () => (key = '')); ro(c3, () => (key = ''));
    AI.loop(root, (dt) => {
      if (tgtB !== null && ctrl.playing) { tb += (tgtB - tb) * ease(dt, 1.5 * ctrl.speed); q('#tb').value = tb; if (Math.abs(tgtB - tb) < 0.004) { tb = tgtB; tgtB = null; } }
      const p = P(), kk = [fx(ta, 3), fx(tb, 3), p.sh, p.piB].join('|'); if (kk === key) return; key = kk;
      q('#shv').textContent = p.sh; q('#bbv').textContent = Math.round(p.piB * 100) + '%'; q('#tav').textContent = fx(ta, 2); q('#tbv').textContent = fx(tb, 2);
      const C = AI.col(), A = grp(p, 0, ta), B = grp(p, 1, tb);
      let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      [[0, ta, A], [1, tb, B]].forEach(([gi, t, G]) => {
        const sh = gi ? p.sh : 0, pi = G.pi, h = c1.h / 2, y0 = gi * h, ymax = 0.42 * 0.6 * 1.25, v = AI.view({ w: c1.w, h }, -5, 6, 0, ymax, { l: 12, r: 12, t: 20, b: 20 });
        const Y = (y) => y0 + v.Y(y);
        const f0 = (x) => (1 - pi) * pdf(x + sh), f1 = (x) => pi * pdf(x - (2 - sh));
        const cu = (from, to, f, col, al) => { g.fillStyle = col; g.globalAlpha = al; g.beginPath(); g.moveTo(v.X(from), Y(0)); for (let x = from; x <= to + 1e-9; x += 0.05) g.lineTo(v.X(x), Y(f(x))); g.lineTo(v.X(to), Y(0)); g.closePath(); g.fill(); g.globalAlpha = 1; };
        cu(-5, t, f0, C.muted, 0.5); cu(t, 6, f0, C.red, 0.55); cu(-5, t, f1, C.orange, 0.6); cu(t, 6, f1, C.green, 0.6);
        g.strokeStyle = C.fg; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(v.X(t), y0 + 14); g.lineTo(v.X(t), Y(0)); g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(v.X(-5), Y(0)); g.lineTo(v.X(6), Y(0)); g.stroke();
        E.txt(g, `Group ${gi ? 'B' : 'A'} · base rate ${Math.round(pi * 100)}% · t = ${fx(t, 2)}`, 14, y0 + 13, C.fg, 'left', 'bold 12px sans-serif');
        E.txt(g, 'green TP · orange FN · red FP · grey TN', c1.w - 14, y0 + 13, C.muted, 'right');
      });
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h);
      const mets = [['selection rate', A.sel, B.sel], ['TPR (recall)', A.tpr, B.tpr], ['FPR', A.fpr, B.fpr], ['precision', A.prec, B.prec]], bw = (c2.w - 30) / 4;
      mets.forEach(([n, a, b], i) => {
        const x0 = 14 + i * bw, bh = c2.h - 74;
        [[a, C.blue, 0], [b, C.orange, 1]].forEach(([val, col, j]) => { g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x0 + 8 + j * (bw / 2 - 10), 34 + bh * (1 - val), bw / 2 - 16, bh * val); g.globalAlpha = 1; E.txt(g, fx(val, 2), x0 + 8 + j * (bw / 2 - 10) + (bw / 2 - 16) / 2, 31 + bh * (1 - val), C.fg, 'center'); });
        E.txt(g, n, x0 + bw / 2 - 4, c2.h - 22, C.fg, 'center'); E.txt(g, 'gap ' + fx(a - b, 2), x0 + bw / 2 - 4, c2.h - 8, Math.abs(a - b) < 0.02 ? C.green : C.orange, 'center');
      });
      E.txt(g, 'blue = group A, orange = group B', c2.w - 8, 12, C.muted, 'right');
      /* trade-off over tB with tA fixed */
      const tk = [p.sh, p.piB, fx(ta, 2)].join(); if (tk !== tradeKey) { tradeKey = tk; trade = []; for (let t = -2; t <= 4.001; t += 0.1) { const Bg = grp(p, 1, t); trade.push([t, acc(p, ta, t), A.sel - Bg.sel, A.tpr - Bg.tpr]); } }
      g = c3.ctx; g.clearRect(0, 0, c3.w, c3.h);
      const v3 = AI.view(c3, -2, 4, -0.6, 1, { l: 40, r: 12, t: 16, b: 34 }); AI.axes(g, v3, {});
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(v3.X(-2), v3.Y(0)); g.lineTo(v3.X(4), v3.Y(0)); g.stroke();
      [[1, C.purple, 'accuracy'], [2, C.blue, 'selection gap A−B'], [3, C.orange, 'TPR gap A−B']].forEach(([ix, col, nm], j) => { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); trade.forEach((row, i) => (i ? g.lineTo : g.moveTo).call(g, v3.X(row[0]), v3.Y(row[ix]))); g.stroke(); E.txt(g, nm, v3.pad.l + 8 + j * 110, 11, col); });
      g.strokeStyle = C.fg; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(v3.X(tb), v3.pad.t); g.lineTo(v3.X(tb), v3.pad.t + v3.ph); g.stroke(); g.setLineDash([]);
      E.txt(g, 'sweeping threshold B (A fixed): where a gap crosses 0 that fairness notion holds', v3.pad.l + v3.pw, c3.h - 3, C.muted, 'right', '10px sans-serif');
      const a = acc(p, ta, tb) * 1, eo = A.fpr - B.fpr;
      q('#o').innerHTML = `<span class="m">per group of 100 people</span>\naccuracy (overall) <b>${fx(a)}</b>\ndemographic parity gap  <b>${fx(A.sel - B.sel, 3)}</b>\nequal-opportunity gap   <b>${fx(A.tpr - B.tpr, 3)}</b>\nFPR gap (equalised odds) <b>${fx(eo, 3)}</b>\nprecision gap            <b>${fx(A.prec - B.prec, 3)}</b>\nselection ratio B/A      ${fx(B.sel / (A.sel || 1), 2)} (rule of thumb ≥ 0.8)`;
      q('#cap').textContent = cap || 'With one shared threshold the lower-scoring group B is selected less and misses more qualified people. Equalising one metric usually moves another away from equal: try matching selection rate, then look at precision and FPR.';
    }, { fps: 30 });
  };
})();
