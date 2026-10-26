/* Module 1 labs — Linear algebra.
 * lincomb, neuron, unitball, datamatrix, matmulsteps, compose, detinv, linsys, eigenvis, pca2d, svdimage, lsproj
 * Animation rules: dt-based, slow by default, speed sliders, pause/step, captions that name each phase. */
(function () {
  const q = (root, s) => AI.$(s, root);
  const F = (v, d = 2) => AI.fmt(v, d);
  const PI = Math.PI;

  (function css() {
    if (document.getElementById('la-css')) return;
    const s = document.createElement('style');
    s.id = 'la-css';
    s.textContent = `.la-mat{border-collapse:separate;border-spacing:4px;display:inline-table;vertical-align:middle}
.la-mat td{min-width:50px;height:36px;text-align:center;border-radius:6px;background:var(--bg);border:1px solid var(--border);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;padding:0 2px;transition:background .25s,border-color .25s,box-shadow .25s}
.la-mat td.hr{box-shadow:inset 0 0 0 2px var(--blue)}.la-mat td.hc{box-shadow:inset 0 0 0 2px var(--orange)}
.la-mat td.cur{background:var(--bg3);box-shadow:inset 0 0 0 2px var(--green)}
.la-mat td.dn{border-color:var(--green);color:var(--green);font-weight:700}.la-mat td.pt{color:var(--orange);font-style:italic}
.la-mat input{width:46px;background:transparent;border:0;color:var(--fg);text-align:center;font:inherit;outline:none}
.la-op{font-size:22px;color:var(--muted);padding:0 6px}.la-wrap{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:4px;padding:6px 0}
.la-bar{height:4px;background:var(--bg3);border-radius:2px;margin-top:6px;overflow:hidden}.la-bar i{display:block;height:100%;background:var(--blue);width:0}
.la-cap{font-size:13px;color:var(--fg);min-height:3.2em;margin-top:6px;line-height:1.5}
.la-ctl-mat{display:grid;grid-template-columns:1fr 1fr;gap:6px}`;
    document.head.appendChild(s);
  })();

  /* ---------- shared helpers ---------- */
  const worldView = (c, R) => { const s = Math.min(c.w, c.h) / (2 * R), ox = c.w / 2, oy = c.h / 2; return { s, ox, oy, P: (x, y) => [ox + x * s, oy - y * s], W: (px, py) => [(px - ox) / s, -(py - oy) / s] }; };
  const T = (M, x, y) => [M[0] * x + M[1] * y, M[2] * x + M[3] * y];
  const mul2 = (A, B) => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
  const blend = (A, B, t) => A.map((v, i) => v + (B[i] - v) * t);
  const I2 = [1, 0, 0, 1];
  const rot = (deg) => { const a = (deg * PI) / 180; return [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)].map((v) => +v.toFixed(4)); };
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const fmtM = (M) => `[${F(M[0], 2)}  ${F(M[1], 2)}; ${F(M[2], 2)}  ${F(M[3], 2)}]`;
  const gridBack = (g, C, w, R) => {
    g.lineWidth = 1; g.strokeStyle = C.border; g.globalAlpha = 0.7;
    g.beginPath();
    for (let i = -R; i <= R; i++) { g.moveTo(...w.P(i, -R)); g.lineTo(...w.P(i, R)); g.moveTo(...w.P(-R, i)); g.lineTo(...w.P(R, i)); }
    g.stroke(); g.globalAlpha = 1;
    g.strokeStyle = C.muted; g.beginPath(); g.moveTo(...w.P(-R, 0)); g.lineTo(...w.P(R, 0)); g.moveTo(...w.P(0, -R)); g.lineTo(...w.P(0, R)); g.stroke();
  };
  const warpGrid = (g, C, w, M, R, col) => {
    g.lineWidth = 1; g.strokeStyle = col || C.blue; g.globalAlpha = 0.55; g.beginPath();
    for (let i = -R; i <= R; i++) { if (i === 0) continue; g.moveTo(...w.P(...T(M, i, -2 * R))); g.lineTo(...w.P(...T(M, i, 2 * R))); g.moveTo(...w.P(...T(M, -2 * R, i))); g.lineTo(...w.P(...T(M, 2 * R, i))); }
    g.stroke(); g.globalAlpha = 1; g.strokeStyle = C.fg; g.lineWidth = 1.5; g.beginPath();
    g.moveTo(...w.P(...T(M, 0, -2 * R))); g.lineTo(...w.P(...T(M, 0, 2 * R))); g.moveTo(...w.P(...T(M, -2 * R, 0))); g.lineTo(...w.P(...T(M, 2 * R, 0))); g.stroke();
  };
  const poly = (g, w, M, pts, fill, alpha, stroke) => {
    g.beginPath(); pts.forEach(([x, y], i) => { const p = w.P(...T(M, x, y)); i ? g.lineTo(...p) : g.moveTo(...p); }); g.closePath();
    if (fill) { g.globalAlpha = alpha; g.fillStyle = fill; g.fill(); g.globalAlpha = 1; }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); }
  };
  const SQ = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const FG = [[0.2, 0.1], [0.4, 0.1], [0.4, 0.4], [0.7, 0.4], [0.7, 0.55], [0.4, 0.55], [0.4, 0.7], [0.8, 0.7], [0.8, 0.85], [0.2, 0.85]];
  const eig2 = (M) => {
    const [a, b, c, d] = M, tr = a + d, det = a * d - b * c, disc = (tr * tr) / 4 - det;
    if (disc < -1e-12) return { re: false, tr, det, im: Math.sqrt(-disc) };
    const s = Math.sqrt(Math.max(0, disc)), e = [];
    [1, -1].forEach((sg) => {
      const lam = tr / 2 + sg * s; let ex = b, ey = lam - a;
      if (Math.abs(ex) + Math.abs(ey) < 1e-9) { ex = lam - d; ey = c; }
      if (Math.abs(ex) + Math.abs(ey) < 1e-9) { ex = sg > 0 ? 1 : 0; ey = sg > 0 ? 0 : 1; }
      const n = Math.hypot(ex, ey); e.push({ lam, v: [ex / n, ey / n] });
    });
    return { re: true, tr, det, e };
  };
  /** Timeline of phases: o.i = current phase, o.t = time inside it (in "speed-scaled" seconds). */
  const phaser = (durs) => {
    const o = { i: 0, t: 0, run: false, done: false };
    o.reset = () => { o.i = 0; o.t = 0; o.run = false; o.done = false; };
    o.update = (dt, sp) => {
      if (!o.run) return;
      o.t += dt * sp;
      while (o.t >= durs[o.i]) {
        if (o.i >= durs.length - 1) { o.t = durs[o.i]; o.run = false; o.done = true; break; }
        o.t -= durs[o.i]; o.i++;
      }
    };
    o.next = () => { if (o.i < durs.length - 1) { o.i++; o.t = 0; } else { o.t = durs[o.i]; o.done = true; } o.run = false; };
    o.p = () => AI.smooth(o.t / durs[o.i]);
    o.play = () => { if (o.done || (o.i >= durs.length - 1 && o.t >= durs[o.i])) o.reset(); o.run = true; };
    return o;
  };
  const speedHTML = (def, id) => `<label>Speed <span class="v" id="${id || 'spv'}"></span><input type="range" id="${id ? id + 'r' : 'sp'}" min="0.2" max="3" step="0.05" value="${def || 1}"></label>`;
  const matInputs = (host, vals, onchange, step) => {
    const ins = vals.map((v, i) => { const el = document.createElement('input'); el.type = 'number'; el.step = step || 0.5; el.value = v; el.oninput = () => { vals[i] = +el.value || 0; onchange(); }; host.appendChild(el); return el; });
    return (nv) => nv.forEach((v, i) => { vals[i] = v; ins[i].value = v; });
  };
  const chipRow = (host, items, fn) => items.forEach(([name, val]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = name; b.onclick = () => fn(val); host.appendChild(b); });
  const solve2 = (M, b) => { const det = M[0] * M[3] - M[1] * M[2]; return Math.abs(det) < 1e-9 ? null : [(b[0] * M[3] - M[1] * b[1]) / det, (M[0] * b[1] - b[0] * M[2]) / det]; };

  /* =====================================================================
   * lincomb — c1*a + c2*b, span, click to aim
   * ===================================================================== */
  AI.labs.lincomb = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, 'Linear combination lab', 'stretch a, stretch b, add — where can you reach?',
      '<canvas style="height:440px;touch-action:none;cursor:crosshair"></canvas><div class="la-cap" id="cap"></div>',
      `<div class="la-ctl-mat" id="mats"></div>
       <div class="chips" id="pre"></div>
       <label><span style="color:var(--blue)">c₁</span> <span class="v" id="v1"></span><input type="range" id="c1" min="-3" max="3" step="0.05" value="1"></label>
       <label><span style="color:var(--orange)">c₂</span> <span class="v" id="v2"></span><input type="range" id="c2" min="-3" max="3" step="0.05" value="1"></label>
       <div class="row"><button class="btn primary" id="auto">▶ Sweep c₁, c₂</button><button class="btn" id="clr">Clear trail</button></div>
       ${speedHTML(0.6)}
       <div class="readout" id="out"></div>
       <div class="caption">Click anywhere on the plane: the sliders glide to the coefficients that reach that point (if it is reachable).</div>`);
    const c = AI.canvas(AI.$('canvas', view)), cv = AI.$('canvas', view), R = 9;
    const A = (d.a || '2,1').split(',').map(Number), B = (d.b || '-1,3').split(',').map(Number);
    const vals = [A[0], A[1], B[0], B[1]];
    const grid = q(root, '#mats');
    grid.innerHTML = '<div style="color:var(--blue);grid-column:1/3">a = (a₁, a₂)</div>';
    const m = [];
    const mk = (i) => { const el = document.createElement('input'); el.type = 'number'; el.step = 0.5; el.value = vals[i]; el.oninput = () => { vals[i] = +el.value || 0; trail.length = 0; }; grid.appendChild(el); m[i] = el; };
    mk(0); mk(1);
    const lb = document.createElement('div'); lb.style.cssText = 'color:var(--orange);grid-column:1/3'; lb.textContent = 'b = (b₁, b₂)'; grid.appendChild(lb);
    mk(2); mk(3);
    const setVals = (v) => { v.forEach((x, i) => { vals[i] = x; m[i].value = x; }); trail.length = 0; };
    chipRow(q(root, '#pre'), [['independent', [2, 1, -1, 3]], ['b = 2a (collinear)', [2, 1, 4, 2]], ['standard basis', [1, 0, 0, 1]]], setVals);
    const trail = []; let auto = false, tau = 0, goal = null, key = '';
    const s1 = q(root, '#c1'), s2 = q(root, '#c2'), sp = q(root, '#sp');
    q(root, '#auto').onclick = () => { auto = !auto; goal = null; q(root, '#auto').textContent = auto ? '❚❚ Pause sweep' : '▶ Sweep c₁, c₂'; };
    q(root, '#clr').onclick = () => (trail.length = 0);
    const toW = (e) => { const r = cv.getBoundingClientRect(), w = worldView(c, R); return w.W(e.clientX - r.left, e.clientY - r.top); };
    let down = false;
    const aim = (e) => {
      const [px, py] = toW(e), sol = solve2([vals[0], vals[2], vals[1], vals[3]], [px, py]);
      if (sol) { goal = { c1: AI.clamp(sol[0], -3, 3), c2: AI.clamp(sol[1], -3, 3) }; auto = false; q(root, '#auto').textContent = '▶ Sweep c₁, c₂'; }
      else { // collinear: project onto the line
        const L = vals[0] * vals[0] + vals[1] * vals[1] || 1, k = (px * vals[0] + py * vals[1]) / L; goal = { c1: AI.clamp(k, -3, 3), c2: 0 }; auto = false;
      }
    };
    cv.addEventListener('pointerdown', (e) => { down = true; cv.setPointerCapture(e.pointerId); aim(e); });
    cv.addEventListener('pointermove', (e) => { if (down) aim(e); });
    cv.addEventListener('pointerup', () => (down = false));
    AI.loop(root, (dt) => {
      const spd = +sp.value; q(root, '#spv').textContent = F(spd, 2) + '×';
      if (auto) { tau += dt * spd; s1.value = 2.4 * Math.sin(0.5 * tau); s2.value = 2.4 * Math.sin(0.31 * tau + 1.3); }
      if (goal) {
        const k = 1 - Math.exp(-dt * 3 * spd);
        s1.value = +s1.value + (goal.c1 - +s1.value) * k; s2.value = +s2.value + (goal.c2 - +s2.value) * k;
        if (Math.abs(goal.c1 - s1.value) + Math.abs(goal.c2 - s2.value) < 1e-3) goal = null;
      }
      const c1 = +s1.value, c2 = +s2.value;
      const k = [vals.join(), c1.toFixed(3), c2.toFixed(3), c.w, c.h, trail.length].join('|');
      if (k === key) return; key = k;
      q(root, '#v1').textContent = F(c1, 2); q(root, '#v2').textContent = F(c2, 2);
      const a = [vals[0], vals[1]], b = [vals[2], vals[3]], t1 = [c1 * a[0], c1 * a[1]], t2 = [c2 * b[0], c2 * b[1]], res = [t1[0] + t2[0], t1[1] + t2[1]];
      const last = trail[trail.length - 1];
      if (!last || Math.hypot(last[0] - res[0], last[1] - res[1]) > 0.04) { trail.push(res); if (trail.length > 500) trail.shift(); }
      const C = AI.col(), g = c.ctx, w = worldView(c, R);
      g.clearRect(0, 0, c.w, c.h); gridBack(g, C, w, R);
      const indep = Math.abs(a[0] * b[1] - a[1] * b[0]) > 1e-6, anyA = Math.hypot(...a) > 1e-9, anyB = Math.hypot(...b) > 1e-9;
      let spanTxt;
      if (indep) { g.fillStyle = C.purple; g.globalAlpha = 0.08; g.fillRect(0, 0, c.w, c.h); g.globalAlpha = 1; spanTxt = 'span{a,b} = the whole plane (shaded): every point is reachable'; }
      else if (anyA || anyB) {
        const dir = anyA ? a : b, n = Math.hypot(...dir); g.strokeStyle = C.purple; g.globalAlpha = 0.35; g.lineWidth = 12; g.beginPath(); g.moveTo(...w.P(-dir[0] / n * 30, -dir[1] / n * 30)); g.lineTo(...w.P(dir[0] / n * 30, dir[1] / n * 30)); g.stroke(); g.globalAlpha = 1;
        spanTxt = 'a and b point along the same line: span{a,b} is only that line — the second vector adds nothing new';
      } else spanTxt = 'both vectors are zero: the span is just the origin';
      g.fillStyle = C.green; trail.forEach(([x, y], i) => { g.globalAlpha = 0.08 + 0.4 * (i / trail.length); const p = w.P(x, y); g.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3); }); g.globalAlpha = 1;
      // faint originals
      AI.arrow(g, ...w.P(0, 0), ...w.P(...a), C.blue, 1.5); AI.arrow(g, ...w.P(0, 0), ...w.P(...b), C.orange, 1.5);
      txt(g, 'a', ...w.P(a[0], a[1]).map((v, i) => v + (i ? -8 : 8)), C.blue, '600 13px sans-serif'); txt(g, 'b', ...w.P(b[0], b[1]).map((v, i) => v + (i ? -8 : 8)), C.orange, '600 13px sans-serif');
      // parallelogram + tip-to-tail construction
      g.setLineDash([5, 4]); g.strokeStyle = C.muted; g.lineWidth = 1.2; g.beginPath(); g.moveTo(...w.P(0, 0)); g.lineTo(...w.P(...t2)); g.lineTo(...w.P(...res)); g.lineTo(...w.P(...t1)); g.stroke(); g.setLineDash([]);
      AI.arrow(g, ...w.P(0, 0), ...w.P(...t1), C.blue, 4);
      AI.arrow(g, ...w.P(...t1), ...w.P(...res), C.orange, 4);
      AI.arrow(g, ...w.P(0, 0), ...w.P(...res), C.green, 3.5);
      txt(g, 'c₁a', ...w.P(t1[0] / 2, t1[1] / 2).map((v, i) => v + (i ? 14 : -26)), C.blue, '600 12px sans-serif');
      txt(g, 'c₂b', ...w.P(t1[0] + t2[0] / 2, t1[1] + t2[1] / 2).map((v, i) => v + (i ? 14 : 8)), C.orange, '600 12px sans-serif');
      txt(g, 'c₁a + c₂b', ...w.P(res[0], res[1]).map((v, i) => v + (i ? -10 : 10)), C.green, '600 13px sans-serif');
      g.fillStyle = C.green; g.beginPath(); g.arc(...w.P(...res), 5, 0, 7); g.fill();
      q(root, '#cap').innerHTML = `① stretch <b style="color:var(--blue)">a</b> by c₁ = ${F(c1)} (blue). ② From its tip, stretch <b style="color:var(--orange)">b</b> by c₂ = ${F(c2)} (orange). ③ The <b style="color:var(--green)">green arrow</b> from the origin to the end is the sum. <i>${spanTxt}.</i>`;
      q(root, '#out').innerHTML = `c₁a = ${F(c1, 2)}·(${F(a[0])}, ${F(a[1])}) = <span class="b">(${F(t1[0])}, ${F(t1[1])})</span>\nc₂b = ${F(c2, 2)}·(${F(b[0])}, ${F(b[1])}) = <span class="o">(${F(t2[0])}, ${F(t2[1])})</span>\nsum = <span class="g">(${F(res[0])}, ${F(res[1])})</span>`;
    });
  };

  /* =====================================================================
   * neuron — a neuron is a dot product
   * ===================================================================== */
  AI.labs.neuron = function (root) {
    const sl = (id, label, min, max, val) => `<label><span>${label} <span class="v" id="v_${id}"></span></span><input type="range" id="${id}" min="${min}" max="${max}" step="0.1" value="${val}"></label>`;
    const { view, ctl } = AI.labShell(root, 'A neuron is a dot product', 'z = w·x + b, then a squashing function',
      '<canvas style="height:330px"></canvas><div class="la-cap" id="cap"></div>',
      `<div class="la-ctl-mat" style="grid-template-columns:1fr 1fr 1fr">${[1, 2, 3].map((i) => sl('x' + i, 'x' + i, -3, 3, [2, 1, -1][i - 1])).join('')}</div>
       <div class="la-ctl-mat" style="grid-template-columns:1fr 1fr 1fr">${[1, 2, 3].map((i) => sl('w' + i, 'w' + i, -2, 2, [1, -0.5, 1.5][i - 1])).join('')}</div>
       ${sl('b', 'bias b', -3, 3, 0.5)}
       <div class="row"><button class="btn" id="al">Point x along w</button><button class="btn" id="pe">Make x ⟂ w</button><button class="btn" id="op">Point x against w</button></div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), ids = ['x1', 'x2', 'x3', 'w1', 'w2', 'w3', 'b'], S = {}; ids.forEach((i) => (S[i] = q(root, '#' + i)));
    const val = () => { const o = {}; ids.forEach((i) => (o[i] = +S[i].value)); return o; };
    const shown = { p1: 0, p2: 0, p3: 0, z: 0, a: 0 };
    const setX = (v) => { ['x1', 'x2', 'x3'].forEach((id, i) => (S[id].value = AI.clamp(+v[i].toFixed(1), -3, 3))); };
    q(root, '#al').onclick = () => { const o = val(), n = Math.hypot(o.w1, o.w2, o.w3) || 1; setX([o.w1, o.w2, o.w3].map((v) => (v / n) * 2)); };
    q(root, '#op').onclick = () => { const o = val(), n = Math.hypot(o.w1, o.w2, o.w3) || 1; setX([o.w1, o.w2, o.w3].map((v) => (-v / n) * 2)); };
    q(root, '#pe').onclick = () => { const o = val(), n = Math.hypot(o.w2, o.w1) || 1; setX(o.w2 === 0 && o.w1 === 0 ? [2, 0, 0] : [(-o.w2 / n) * 2, (o.w1 / n) * 2, 0]); };
    AI.loop(root, (dt) => {
      const o = val(); ids.forEach((i) => (q(root, '#v_' + i).textContent = F(o[i], 1)));
      const p = [o.x1 * o.w1, o.x2 * o.w2, o.x3 * o.w3], z = p[0] + p[1] + p[2] + o.b, act = 1 / (1 + Math.exp(-z));
      const k = 1 - Math.exp(-dt * 4);
      shown.p1 += (p[0] - shown.p1) * k; shown.p2 += (p[1] - shown.p2) * k; shown.p3 += (p[2] - shown.p3) * k; shown.z += (z - shown.z) * k; shown.a += (act - shown.a) * k;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const rows = [['x₁·w₁', shown.p1, `${F(o.x1, 1)} × ${F(o.w1, 1)}`], ['x₂·w₂', shown.p2, `${F(o.x2, 1)} × ${F(o.w2, 1)}`], ['x₃·w₃', shown.p3, `${F(o.x3, 1)} × ${F(o.w3, 1)}`], ['+ bias', o.b, 'b'], ['z = sum', shown.z, '']];
      const x0 = 120, mid = x0 + (c.w - x0 - 30) / 2, sc = (c.w - x0 - 30) / 2 / 8;
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(mid, 10); g.lineTo(mid, c.h - 62); g.stroke();
      rows.forEach(([lab, v, sub], i) => {
        const y = 22 + i * 48, last = i === 4; g.fillStyle = v >= 0 ? (last ? C.orange : C.green) : C.red; g.globalAlpha = 0.85;
        g.fillRect(v >= 0 ? mid : mid + v * sc, y, Math.abs(v) * sc, 26); g.globalAlpha = 1;
        txt(g, lab, 10, y + 18, C.fg, '600 13px sans-serif'); txt(g, sub, 10, y + 4, C.muted, '11px sans-serif');
        txt(g, F(v, 2), v >= 0 ? mid + v * sc + 6 : mid + v * sc - 6, y + 18, C.fg, '12px ui-monospace,monospace', v >= 0 ? 'left' : 'right');
      });
      const y = 22 + 5 * 48 + 4; g.fillStyle = C.bg3; g.fillRect(x0, y, c.w - x0 - 30, 12); g.fillStyle = C.purple; g.fillRect(x0, y, (c.w - x0 - 30) * shown.a, 12);
      txt(g, 'σ(z)', 10, y + 11, C.fg, '600 13px sans-serif'); txt(g, F(shown.a, 3), x0 + (c.w - x0 - 30) * shown.a + 6, y + 11, C.fg, '12px ui-monospace,monospace');
      const nx = Math.hypot(o.x1, o.x2, o.x3), nw = Math.hypot(o.w1, o.w2, o.w3), dot = p[0] + p[1] + p[2], cs = nx * nw ? dot / (nx * nw) : 0;
      q(root, '#cap').innerHTML = `Each bar is one product <i>xᵢwᵢ</i>: green helps the sum, red hurts it. The neuron outputs σ(z) = 1/(1+e<sup>−z</sup>) = <b>${F(act, 3)}</b>. It “fires” hardest when x points the same way as w.`;
      q(root, '#out').innerHTML = `w·x = ${F(dot, 2)}   (z = w·x + b = <span class="o">${F(z, 2)}</span>)\n|w| = ${F(nw, 2)}   |x| = ${F(nx, 2)}\ncos θ = ${F(cs, 2)}  →  θ = ${F((Math.acos(AI.clamp(cs, -1, 1)) * 180) / PI, 0)}°\n<span class="m">w·x = |w||x| cos θ = ${F(nw * nx * cs, 2)}</span>`;
    });
  };

  /* =====================================================================
   * unitball — L_p unit balls morphing, point norms
   * ===================================================================== */
  AI.labs.unitball = function (root) {
    const { view, ctl } = AI.labShell(root, 'Norms: the shape of “length”', 'the set of points at distance r from the origin',
      '<canvas style="height:420px;touch-action:none;cursor:crosshair"></canvas><div class="la-cap" id="cap"></div>',
      `<label>p <span class="v" id="pv"></span><input type="range" id="p" min="0.5" max="8" step="0.05" value="2"></label>
       <div class="chips" id="pre"></div>
       <label>radius r <span class="v" id="rv"></span><input type="range" id="r" min="0.5" max="3" step="0.1" value="2"></label>
       <div class="row"><button class="btn primary" id="tour">▶ Tour p = 1 → 2 → ∞</button></div>
       ${speedHTML(1)}
       <div class="readout" id="out"></div>
       <div class="caption">Click or drag in the plane to place a point and read its three norms.</div>`);
    const c = AI.canvas(AI.$('canvas', view)), cv = AI.$('canvas', view), R = 3.6, pe = q(root, '#p'), re = q(root, '#r');
    let cur = 2, tgt = 2, pt = [1.2, 1.4], tour = null, key = '';
    const norm = (u, p) => p >= 40 ? Math.max(Math.abs(u[0]), Math.abs(u[1])) : Math.pow(Math.pow(Math.abs(u[0]), p) + Math.pow(Math.abs(u[1]), p), 1 / p);
    chipRow(q(root, '#pre'), [['L1', 1], ['L2', 2], ['L∞', 60], ['p = 0.5', 0.5]], (v) => { tgt = v; tour = null; });
    pe.oninput = () => { tgt = cur = +pe.value; tour = null; };
    q(root, '#tour').onclick = () => { tour = [1, 2, 60]; tgt = 1; };
    const toW = (e) => { const r = cv.getBoundingClientRect(); return worldView(c, R).W(e.clientX - r.left, e.clientY - r.top); };
    let down = false; cv.addEventListener('pointerdown', (e) => { down = true; cv.setPointerCapture(e.pointerId); pt = toW(e); }); cv.addEventListener('pointermove', (e) => { if (down) pt = toW(e); }); cv.addEventListener('pointerup', () => (down = false));
    let hold = 0;
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      if (tour) { // wait at each stop, then move on
        if (Math.abs(cur - tgt) < 0.02 * Math.max(1, cur)) { hold += dt * sp; if (hold > 2.2) { hold = 0; tour.shift(); if (tour.length) tgt = tour[0]; else tour = null; } } else hold = 0;
      }
      const prev = cur;
      if (Math.abs(tgt - cur) > 1e-3) { const lc = Math.log(cur), lt = Math.log(tgt), step = dt * sp * 0.7; cur = Math.exp(lc + AI.clamp(lt - lc, -step, step)); }
      if (Math.abs(cur - tgt) < 1e-3) cur = tgt;
      const r = +re.value, k = [cur.toFixed(3), r, pt[0].toFixed(2), pt[1].toFixed(2), c.w].join('|'); if (k === key) return; key = k;
      pe.value = AI.clamp(cur, 0.5, 8); q(root, '#pv').textContent = cur >= 40 ? '∞' : F(cur, 2); q(root, '#rv').textContent = F(r, 1);
      const C = AI.col(), g = c.ctx, w = worldView(c, R); g.clearRect(0, 0, c.w, c.h); gridBack(g, C, w, R);
      g.setLineDash([5, 4]); g.strokeStyle = C.muted; g.beginPath(); g.arc(...w.P(0, 0), r * w.s, 0, 7); g.stroke(); g.setLineDash([]);
      g.beginPath(); for (let i = 0; i <= 360; i++) { const th = (i / 360) * 2 * PI, u = [Math.cos(th), Math.sin(th)], n = norm(u, cur), p = w.P((u[0] / n) * r, (u[1] / n) * r); i ? g.lineTo(...p) : g.moveTo(...p); }
      g.closePath(); g.fillStyle = C.blue; g.globalAlpha = 0.18; g.fill(); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 3; g.stroke();
      const n = norm(pt, cur) || 1e-9; AI.arrow(g, ...w.P(0, 0), ...w.P(...pt), C.orange, 3);
      g.setLineDash([3, 3]); g.strokeStyle = C.orange; g.beginPath(); g.moveTo(...w.P(...pt)); g.lineTo(...w.P((pt[0] / n) * r, (pt[1] / n) * r)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.orange; g.beginPath(); g.arc(...w.P(...pt), 6, 0, 7); g.fill();
      txt(g, 'solid blue curve: all points with ‖x‖ₚ = r;  dashed grey circle: the L2 ball', 10, 18, C.muted, '12px sans-serif');
      const l1 = Math.abs(pt[0]) + Math.abs(pt[1]), l2 = Math.hypot(...pt), li = Math.max(Math.abs(pt[0]), Math.abs(pt[1]));
      const shape = cur < 0.95 ? 'p < 1: the “ball” caves in — this is not a true norm (triangle inequality fails).' : Math.abs(cur - 1) < 0.06 ? 'p = 1: a diamond. Its corners sit on the axes — the reason L1 produces sparse solutions.' : Math.abs(cur - 2) < 0.06 ? 'p = 2: a round circle — the ordinary straight-line length.' : cur >= 40 ? 'p = ∞: a square. Only the largest coordinate counts.' : cur < 2 ? 'between 1 and 2: the diamond’s corners are rounding off.' : 'between 2 and ∞: the circle is squaring up.';
      q(root, '#cap').innerHTML = shape;
      q(root, '#out').innerHTML = `point = (${F(pt[0], 2)}, ${F(pt[1], 2)})\nL1  = |x|+|y|       = <span class="b">${F(l1, 3)}</span>\nL2  = √(x²+y²)      = <span class="g">${F(l2, 3)}</span>\nL∞  = max(|x|,|y|)  = <span class="o">${F(li, 3)}</span>\nLp (p=${cur >= 40 ? '∞' : F(cur, 2)}) = <span class="r">${F(n, 3)}</span>\n<span class="m">always  L∞ ≤ L2 ≤ L1</span>`;
    });
  };

  /* =====================================================================
   * datamatrix — rows/cols, hover, animated transpose
   * ===================================================================== */
  AI.labs.datamatrix = function (root) {
    const { view, ctl } = AI.labShell(root, 'The data matrix X', 'rows are examples, columns are features',
      '<canvas style="height:330px"></canvas><div class="la-cap" id="cap"></div>',
      `<div class="row"><button class="btn primary" id="tr">⇄ Transpose</button></div>
       ${speedHTML(1)}
       <div class="readout" id="out"></div>
       <div class="caption">Hover over any entry: its row (blue) is one example, its column (orange) is one feature. Transposing swaps the roles: entry (i, j) moves to (j, i), so rows become columns.</div>`);
    const cv = AI.$('canvas', view), c = AI.canvas(cv);
    const X = [[3, 120, 8], [2, 85, 15], [4, 150, 3], [3, 100, 10]], rl = ['house 1', 'house 2', 'house 3', 'house 4'], cl = ['rooms', 'area m²', 'age yr'];
    let p = 0, dir = 0, hi = -1, hj = -1, key = '';
    const cw = 78, ch = 40;
    const ox = () => (c.w - 3 * cw) / 2 + 10, oy = () => 66;
    q(root, '#tr').onclick = () => (dir = p > 0.5 || dir > 0 ? -1 : 1);
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; hi = hj = -1;
      if (p > 0.02 && p < 0.98) return;
      const col = Math.floor((x - ox()) / cw), row = Math.floor((y - oy()) / ch);
      if (p <= 0.02 && row >= 0 && row < 4 && col >= 0 && col < 3) { hi = row; hj = col; }
      if (p >= 0.98 && row >= 0 && row < 3 && col >= 0 && col < 4) { hi = col; hj = row; }
    });
    cv.addEventListener('pointerleave', () => (hi = hj = -1));
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      if (dir) { p = AI.clamp(p + dir * dt * sp / 2.6, 0, 1); if (p === 0 || p === 1) dir = 0; }
      const k = [p.toFixed(3), hi, hj, c.w].join('|'); if (k === key) return; key = k;
      const C = AI.col(), g = c.ctx, e = AI.smooth(p); g.clearRect(0, 0, c.w, c.h);
      const x0 = ox(), y0 = oy(), nr = 4 + (3 - 4) * e, nc = 3 + (4 - 3) * e;
      // labels
      g.globalAlpha = 1 - e; cl.forEach((s, j) => txt(g, s, x0 + j * cw + cw / 2, y0 - 12, C.orange, '600 12px sans-serif', 'center')); rl.forEach((s, i) => txt(g, s, x0 - 14, y0 + i * ch + ch / 2 + 4, C.blue, '600 12px sans-serif', 'right'));
      g.globalAlpha = e; rl.forEach((s, i) => txt(g, s, x0 + i * cw + cw / 2, y0 - 12, C.blue, '600 12px sans-serif', 'center')); cl.forEach((s, j) => txt(g, s, x0 - 14, y0 + j * ch + ch / 2 + 4, C.orange, '600 12px sans-serif', 'right')); g.globalAlpha = 1;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        const px = x0 + AI.lerp(j * cw, i * cw, e), py = y0 + AI.lerp(i * ch, j * ch, e);
        g.fillStyle = (i === hi && j === hj) ? C.bg3 : C.bg; g.fillRect(px + 2, py + 2, cw - 4, ch - 4);
        if (i === hi) { g.fillStyle = C.blue; g.globalAlpha = 0.22; g.fillRect(px + 2, py + 2, cw - 4, ch - 4); g.globalAlpha = 1; }
        if (j === hj) { g.fillStyle = C.orange; g.globalAlpha = 0.22; g.fillRect(px + 2, py + 2, cw - 4, ch - 4); g.globalAlpha = 1; }
        g.strokeStyle = (i === hi && j === hj) ? C.fg : C.border; g.lineWidth = 1; g.strokeRect(px + 2.5, py + 2.5, cw - 5, ch - 5);
        txt(g, String(X[i][j]), px + cw / 2, py + ch / 2 + 5, C.fg, '14px ui-monospace,monospace', 'center');
      }
      const bw = nc * cw, bh = nr * ch; g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(x0 + 6, y0); g.lineTo(x0, y0); g.lineTo(x0, y0 + bh); g.lineTo(x0 + 6, y0 + bh); g.moveTo(x0 + bw - 6, y0); g.lineTo(x0 + bw, y0); g.lineTo(x0 + bw, y0 + bh); g.lineTo(x0 + bw - 6, y0 + bh); g.stroke();
      const shape = e < 0.5 ? '4 × 3' : '3 × 4';
      txt(g, e < 0.5 ? 'X  is  4 × 3  (4 examples, 3 features)' : 'Xᵀ  is  3 × 4  (3 features, 4 examples)', c.w / 2, c.h - 14, C.fg, '600 13px sans-serif', 'center');
      q(root, '#cap').textContent = p === 0 ? 'Press Transpose and watch every entry slide across the diagonal.' : p === 1 ? 'Now rows are features and columns are examples. Press again to undo: (Xᵀ)ᵀ = X.' : 'Entry (i, j) is travelling to position (j, i)…';
      q(root, '#out').innerHTML = hi >= 0 ? `X[${hi + 1},${hj + 1}] = <b class="o">${X[hi][hj]}</b>\n  example ${hi + 1} (${rl[hi]}), feature ${hj + 1} (${cl[hj]})\nXᵀ[${hj + 1},${hi + 1}] = ${X[hi][hj]}  (same number)\nrow ${hi + 1} of X = (${X[hi].join(', ')})\ncolumn ${hj + 1} = (${X.map((r) => r[hj]).join(', ')})` : `shape ${shape}  (rows × columns)\n<span class="m">hover an entry…</span>`;
    });
  };

  /* =====================================================================
   * matmulsteps — animated matrix product
   * ===================================================================== */
  AI.labs.matmulsteps = function (root) {
    const sel = (id, v) => `<select id="${id}">${[1, 2, 3, 4].map((n) => `<option${n === v ? ' selected' : ''}>${n}</option>`).join('')}</select>`;
    const { view, ctl } = AI.labShell(root, 'Matrix multiplication, step by step', 'every entry of C is a row of A dotted with a column of B',
      '<div class="la-wrap" id="tabs"></div><div class="la-bar"><i id="pb"></i></div><div class="la-cap" id="cap"></div>',
      `<div class="row" style="gap:12px;white-space:nowrap">A is <b>m</b> × <b>k</b>, B is <b>k</b> × <b>n</b> &nbsp; m ${sel('m', 2)} k ${sel('k', 3)} n ${sel('n', 2)}</div>
       <label>View <select id="mode"><option value="cell">cell by cell (row · column)</option><option value="col">column by column (combination of A's columns)</option></select></label>
       <div class="row"><button class="btn primary" id="play">▶ Play</button><button class="btn" id="step">Step ▸</button><button class="btn" id="reset">↺ Reset</button><button class="btn" id="rand">Random numbers</button></div>
       ${speedHTML(0.7)}
       <div class="readout" id="out"></div>
       <div class="caption">Edit any number in A or B. The inner sizes must match: A has k columns, B has k rows. C is m×n.</div>`);
    const tabs = q(root, '#tabs');
    let m = 2, k = 3, n = 2, A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]], ev = [], e = 0, tt = 0, run = false, mode = 'cell', tdA, tdB, tdC, dirty = true;
    const dur = (x) => (x.type === 'write' ? 0.9 : mode === 'cell' ? 1.1 : 1.6);
    const buildEvents = () => {
      ev = [];
      if (mode === 'cell') { for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) { for (let l = 0; l < k; l++) ev.push({ type: 'term', i, j, l }); ev.push({ type: 'write', i, j }); } }
      else for (let j = 0; j < n; j++) for (let l = 0; l < k; l++) ev.push({ type: 'term', j, l });
    };
    const matHTML = (id, rows, cols, edit) => `<table class="la-mat" id="${id}">${Array.from({ length: rows }, (_, i) => `<tr>${Array.from({ length: cols }, (_, j) => `<td>${edit ? `<input type="number" data-i="${i}" data-j="${j}">` : ''}</td>`).join('')}</tr>`).join('')}</table>`;
    const build = () => {
      tabs.innerHTML = `<div style="text-align:center"><div style="color:var(--blue);font-size:12px">A (${m}×${k})</div>${matHTML('ta', m, k, true)}</div><span class="la-op">×</span><div style="text-align:center"><div style="color:var(--orange);font-size:12px">B (${k}×${n})</div>${matHTML('tb', k, n, true)}</div><span class="la-op">=</span><div style="text-align:center"><div style="color:var(--green);font-size:12px">C = AB (${m}×${n})</div>${matHTML('tc', m, n, false)}</div>`;
      const grab = (id) => [...tabs.querySelectorAll('#' + id + ' tr')].map((tr) => [...tr.children]);
      tdA = grab('ta'); tdB = grab('tb'); tdC = grab('tc');
      tdA.forEach((r, i) => r.forEach((td, j) => { const inp = td.firstChild; inp.value = A[i][j]; inp.oninput = () => { A[i][j] = +inp.value || 0; restart(); }; }));
      tdB.forEach((r, i) => r.forEach((td, j) => { const inp = td.firstChild; inp.value = B[i][j]; inp.oninput = () => { B[i][j] = +inp.value || 0; restart(); }; }));
      buildEvents(); e = 0; tt = 0; run = false; dirty = true; q(root, '#play').textContent = '▶ Play';
    };
    const restart = () => { e = 0; tt = 0; run = false; dirty = true; q(root, '#play').textContent = '▶ Play'; };
    const resize = () => {
      m = +q(root, '#m').value; k = +q(root, '#k').value; n = +q(root, '#n').value; mode = q(root, '#mode').value;
      const fix = (M, r, cc) => Array.from({ length: r }, (_, i) => Array.from({ length: cc }, (_, j) => (M[i] && M[i][j] !== undefined ? M[i][j] : (i + j + 1) % 5 + 1)));
      A = fix(A, m, k); B = fix(B, k, n); build();
    };
    ['m', 'k', 'n', 'mode'].forEach((id) => (q(root, '#' + id).onchange = resize));
    q(root, '#play').onclick = () => { if (e >= ev.length) { e = 0; tt = 0; } run = !run; q(root, '#play').textContent = run ? '❚❚ Pause' : '▶ Play'; };
    q(root, '#step').onclick = () => { run = false; q(root, '#play').textContent = '▶ Play'; if (e < ev.length) { e++; tt = 0; dirty = true; } };
    q(root, '#reset').onclick = restart;
    q(root, '#rand').onclick = () => { const r = () => Math.floor(Math.random() * 9) - 2; A = A.map((row) => row.map(r)); B = B.map((row) => row.map(r)); build(); };
    const render = () => {
      const cur = ev[e], partial = Array.from({ length: m }, () => new Array(n).fill(0)), cnt = Array.from({ length: m }, () => new Array(n).fill(0)), doneCell = Array.from({ length: m }, () => new Array(n).fill(false));
      for (let x = 0; x < e && x < ev.length; x++) {
        const v = ev[x];
        if (mode === 'cell') { if (v.type === 'term') { partial[v.i][v.j] += A[v.i][v.l] * B[v.l][v.j]; cnt[v.i][v.j]++; } else doneCell[v.i][v.j] = true; }
        else for (let i = 0; i < m; i++) { partial[i][v.j] += A[i][v.l] * B[v.l][v.j]; cnt[i][v.j]++; }
      }
      if (mode === 'col') for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (cnt[i][j] === k) doneCell[i][j] = true;
      tdA.forEach((r) => r.forEach((td) => (td.className = ''))); tdB.forEach((r) => r.forEach((td) => (td.className = '')));
      let cap = 'Press <b>Play</b> (or <b>Step</b>). Matrix C will fill in one dot product at a time.', out = '';
      if (cur) {
        if (mode === 'cell') {
          const { i, j } = cur;
          for (let l = 0; l < k; l++) { tdA[i][l].classList.add('hr'); tdB[l][j].classList.add('hc'); }
          if (cur.type === 'term') {
            const l = cur.l; tdA[i][l].classList.add('cur'); tdB[l][j].classList.add('cur');
            cap = `<b>Entry C<sub>${i + 1}${j + 1}</sub></b>: take row ${i + 1} of A (blue) and column ${j + 1} of B (orange). Term ${l + 1} of ${k}: <b>${F(A[i][l])} × ${F(B[l][j])} = ${F(A[i][l] * B[l][j])}</b>. Running sum so far: ${F(partial[i][j] + A[i][l] * B[l][j])}.`;
          } else cap = `Row ${i + 1} · column ${j + 1} finished: <b>C<sub>${i + 1}${j + 1}</sub> = ${F(partial[i][j])}</b>. It is written into C.`;
          const terms = Array.from({ length: k }, (_, l) => `${F(A[i][l])}·${F(B[l][j])}`).join(' + ');
          out = `C[${i + 1},${j + 1}] = ${terms}\n         = ${Array.from({ length: k }, (_, l) => F(A[i][l] * B[l][j])).join(' + ')} = <span class="g">${F(A[i].reduce((s, v, l) => s + v * B[l][j], 0))}</span>`;
        } else {
          const { j, l } = cur; for (let i = 0; i < m; i++) tdA[i][l].classList.add('hr'); tdB[l][j].classList.add('cur');
          for (let x = 0; x < k; x++) tdB[x][j].classList.add('hc');
          cap = `<b>Column ${j + 1} of C</b> is built from the <i>columns of A</i>, using column ${j + 1} of B as the recipe. Step ${l + 1} of ${k}: add <b>${F(B[l][j])} × (column ${l + 1} of A)</b> = ${F(B[l][j])} × (${A.map((r) => F(r[l])).join(', ')}).`;
          out = `C[:,${j + 1}] = ` + Array.from({ length: k }, (_, x) => `${F(B[x][j])}·A[:,${x + 1}]`).join(' + ');
        }
      } else if (e >= ev.length && ev.length) cap = '<b>Done.</b> Every entry of C was one row-times-column dot product. Edit a number in A or B and replay, or switch the view.';
      tdC.forEach((r, i) => r.forEach((td, j) => {
        td.className = ''; if (doneCell[i][j]) { td.className = 'dn'; td.textContent = F(partial[i][j]); } else if (cnt[i][j] > 0) { td.className = 'pt'; td.textContent = F(partial[i][j]) + '…'; } else td.textContent = '·';
        if (cur && mode === 'cell' && cur.i === i && cur.j === j) td.classList.add('cur'); if (cur && mode === 'col' && cur.j === j) td.classList.add('cur');
      }));
      q(root, '#cap').innerHTML = cap; q(root, '#out').innerHTML = out || `cost: m·n·k = ${m}·${n}·${k} = <b>${m * n * k}</b> multiplications`;
      dirty = false;
    };
    resize();
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      if (run && e < ev.length) { tt += dt * sp; if (tt >= dur(ev[e])) { e++; tt = 0; dirty = true; if (e >= ev.length) { run = false; q(root, '#play').textContent = '▶ Replay'; } } }
      if (dirty) render();
      q(root, '#pb').style.width = e < ev.length ? (100 * tt) / dur(ev[e]) + '%' : '100%';
    });
  };

  /* =====================================================================
   * compose — two transformations applied one after the other
   * ===================================================================== */
  AI.labs.compose = function (root) {
    const { view, ctl } = AI.labShell(root, 'Composition: do M₁, then M₂', 'multiplying matrices = chaining transformations',
      '<canvas style="height:430px"></canvas><div class="la-cap" id="cap"></div>',
      `<div style="color:var(--green)">M₁ (applied first)</div><div class="la-ctl-mat" id="m1"></div>
       <div style="color:var(--red)">M₂ (applied second)</div><div class="la-ctl-mat" id="m2"></div>
       <div class="chips" id="pre"></div>
       <div class="row"><button class="btn primary" id="play">▶ Play</button><button class="btn" id="step">Next phase ▸</button><button class="btn" id="reset">↺ Reset</button><button class="btn" id="swap">⇄ Swap order</button></div>
       ${speedHTML(0.8)}
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), R = 4;
    const M1 = [0, -1, 1, 0], M2 = [1, 1, 0, 1], ph = phaser([1.5, 3.2, 1.8, 3.2, 3]);
    const set1 = matInputs(q(root, '#m1'), M1, () => ph.reset(), 0.5), set2 = matInputs(q(root, '#m2'), M2, () => ph.reset(), 0.5);
    chipRow(q(root, '#pre'), [['rotate 90°, then shear', [[0, -1, 1, 0], [1, 1, 0, 1]]], ['stretch, then rotate 45°', [[2, 0, 0, 0.5], rot(45)]], ['two rotations (commute)', [rot(30), rot(60)]]], ([a, b]) => { set1(a); set2(b); ph.reset(); });
    q(root, '#play').onclick = () => { ph.play(); }; q(root, '#step').onclick = () => ph.next(); q(root, '#reset').onclick = () => ph.reset();
    q(root, '#swap').onclick = () => { const a = M1.slice(), b = M2.slice(); set1(b); set2(a); ph.reset(); };
    ph.i = 4; ph.t = 3; ph.done = true;
    const caps = ['Start: the plane as we know it. Green arrow = î (where (1,0) goes), red arrow = ĵ (where (0,1) goes). The F shows orientation.', '<b>Phase 1.</b> Apply M₁ to everything: the whole plane moves, and î, ĵ land on the columns of M₁.', 'M₁ is done. The plane now looks like “the image under M₁”.', '<b>Phase 2.</b> Now apply M₂ to the already-moved plane.', '<b>Done.</b> The two moves together equal the single matrix M₂M₁: its columns are where î and ĵ finally landed.'];
    let key = '';
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×'; ph.update(dt, sp);
      const P21 = mul2(M2, M1), P12 = mul2(M1, M2), p = ph.p();
      let M; if (ph.i === 0) M = I2; else if (ph.i === 1) M = blend(I2, M1, p); else if (ph.i === 2) M = M1; else if (ph.i === 3) M = mul2(blend(I2, M2, p), M1); else M = P21;
      const k = [ph.i, ph.t.toFixed(2), M1.join(), M2.join(), c.w].join('|'); if (k === key) return; key = k;
      const C = AI.col(), g = c.ctx, w = worldView(c, R); g.clearRect(0, 0, c.w, c.h); gridBack(g, C, w, R); warpGrid(g, C, w, M, R, C.blue);
      poly(g, w, P21, SQ, null, 0, C.muted); g.setLineDash([]);
      poly(g, w, M, SQ, C.purple, 0.25); poly(g, w, M, FG, C.orange, 0.6, C.orange);
      AI.arrow(g, ...w.P(0, 0), ...w.P(M[0], M[2]), C.green, 3.5); AI.arrow(g, ...w.P(0, 0), ...w.P(M[1], M[3]), C.red, 3.5);
      txt(g, 'î', ...w.P(M[0], M[2]).map((v, i) => v + (i ? -6 : 8)), C.green, '600 13px sans-serif'); txt(g, 'ĵ', ...w.P(M[1], M[3]).map((v, i) => v + (i ? -6 : 8)), C.red, '600 13px sans-serif');
      txt(g, 'grey outline = final position of the unit square (M₂M₁)', 10, 18, C.muted, '12px sans-serif');
      q(root, '#cap').innerHTML = caps[ph.i];
      const same = P21.every((v, i) => Math.abs(v - P12[i]) < 1e-3);
      q(root, '#out').innerHTML = `M₂·M₁ = ${fmtM(P21)}  <span class="g">(M₁ first)</span>\nM₁·M₂ = ${fmtM(P12)}  <span class="r">(M₂ first)</span>\n${same ? '<span class="m">equal here: these two commute</span>' : '<b class="o">different: order matters!</b>'}`;
    });
  };

  /* =====================================================================
   * detinv — determinant as area scaling, inverse as undo
   * ===================================================================== */
  AI.labs.detinv = function (root) {
    const { view, ctl } = AI.labShell(root, 'Determinant and inverse', 'how much does A stretch area — and can we undo it?',
      '<canvas style="height:430px"></canvas><div class="la-cap" id="cap"></div>',
      `<div class="la-ctl-mat" id="mat"></div>
       <div class="chips" id="pre"></div>
       <div class="row"><button class="btn primary" id="play">▶ Play A, then A⁻¹</button><button class="btn" id="step">Next phase ▸</button><button class="btn" id="reset">↺ Reset</button></div>
       ${speedHTML(0.8)}
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), R = 4, A = [2, 1, 1, 1], ph = phaser([1.5, 3.5, 2.5, 3.5, 2.5]);
    const setA = matInputs(q(root, '#mat'), A, () => ph.reset(), 0.5);
    chipRow(q(root, '#pre'), [['A = [2 1; 1 1]', [2, 1, 1, 1]], ['stretch ×2.5 in x', [2.5, 0, 0, 1]], ['reflect (det < 0)', [-1, 0, 0, 1]], ['rotate 60° (det = 1)', rot(60)], ['singular [1 2; 0.5 1]', [1, 2, 0.5, 1]], ['almost singular', [1, 1, 1, 1.15]]], (v) => { setA(v); ph.reset(); });
    q(root, '#play').onclick = () => ph.play(); q(root, '#step').onclick = () => ph.next(); q(root, '#reset').onclick = () => ph.reset();
    ph.i = 2; ph.t = 0.5;
    let key = '';
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×'; ph.update(dt, sp);
      const det = A[0] * A[3] - A[1] * A[2], sing = Math.abs(det) < 1e-6, p = ph.p();
      let M; if (ph.i === 0) M = I2; else if (ph.i === 1) M = blend(I2, A, p); else if (ph.i === 2) M = A; else if (ph.i === 3) M = sing ? A : blend(A, I2, p); else M = sing ? A : I2;
      const k = [ph.i, ph.t.toFixed(2), A.join(), c.w].join('|'); if (k === key) return; key = k;
      const C = AI.col(), g = c.ctx, w = worldView(c, R); g.clearRect(0, 0, c.w, c.h); gridBack(g, C, w, R); warpGrid(g, C, w, M, R, C.blue);
      g.setLineDash([5, 4]); poly(g, w, I2, SQ, null, 0, C.muted); g.setLineDash([]);
      const dm = M[0] * M[3] - M[1] * M[2];
      poly(g, w, M, SQ, dm < 0 ? C.pink : C.purple, 0.32, dm < 0 ? C.pink : C.purple); poly(g, w, M, FG, C.orange, 0.65, C.orange);
      AI.arrow(g, ...w.P(0, 0), ...w.P(M[0], M[2]), C.green, 3.5); AI.arrow(g, ...w.P(0, 0), ...w.P(M[1], M[3]), C.red, 3.5);
      txt(g, `area of the image of the unit square = |det| = ${F(Math.abs(dm), 3)}`, 10, 18, C.fg, '600 13px sans-serif'); txt(g, 'dashed = original unit square (area 1)', 10, 36, C.muted, '12px sans-serif');
      const caps = ['Start: the unit square (area 1) with an F inside it, so we can see turning and flipping.',
        `<b>Phase 1.</b> Apply A. Watch the area: it grows or shrinks smoothly from 1 to |det A| = ${F(Math.abs(det), 3)}.`,
        sing ? '<b>A is singular (det = 0).</b> The square has been flattened onto a line: its area is 0. Different points now sit on top of each other.' : `A has turned the square into a parallelogram of area ${F(Math.abs(det), 3)}${det < 0 ? ' and flipped it (negative determinant: the F is mirrored)' : ''}.`,
        sing ? '<b>Phase 2.</b> There is no A⁻¹. Nothing can pull the flattened line back into a square, because the lost dimension cannot be recovered.' : '<b>Phase 2.</b> Apply A⁻¹: the matrix that undoes A. Everything glides back to where it started.',
        sing ? 'Information destroyed by a singular matrix is gone for good.' : '<b>Done.</b> A⁻¹A = I: the original square and F are restored.'];
      q(root, '#cap').innerHTML = caps[ph.i];
      const inv = sing ? null : [A[3] / det, -A[1] / det, -A[2] / det, A[0] / det];
      q(root, '#out').innerHTML = `A = ${fmtM(A)}\ndet A = a·d − b·c = ${F(A[0], 2)}·${F(A[3], 2)} − ${F(A[1], 2)}·${F(A[2], 2)} = <span class="o">${F(det, 4)}</span>\n${sing ? '<span class="r">singular: no inverse</span>' : `A⁻¹ = (1/det)·[d −b; −c a]\n    = ${fmtM(inv)}`}\n${Math.abs(det) > 1e-6 && Math.abs(det) < 0.05 ? '<span class="m">tiny |det|: a nearly-flat square, a huge A⁻¹</span>' : ''}`;
    });
  };

  /* =====================================================================
   * linsys — row picture vs column picture
   * ===================================================================== */
  AI.labs.linsys = function (root) {
    const { view, ctl } = AI.labShell(root, 'Ax = b: two pictures of the same system', 'rows: lines that must meet · columns: a recipe that must reach b',
      '<canvas id="c1" style="height:330px"></canvas><canvas id="c2" style="height:330px"></canvas><div class="la-cap" id="cap"></div>',
      `<div style="color:var(--muted)">A = [a b; c d]  and  b = (e, f)</div><div class="la-ctl-mat" id="mat" style="grid-template-columns:1fr 1fr 1fr"></div>
       <div class="chips" id="pre"></div>
       <div class="row"><button class="btn primary" id="play">▶ Replay column picture</button><button class="btn" id="step">Next phase ▸</button></div>
       ${speedHTML(0.8)}
       <div class="readout" id="out"></div>`);
    const cvs = view.querySelectorAll('canvas'), c1 = AI.canvas(cvs[0]), c2 = AI.canvas(cvs[1]), R = 6;
    const V = [2, 1, 5, 1, -1, 1]; // stored as [a,b,e,c,d,f]
    const mat = q(root, '#mat'), ins = [];
    const order = [0, 1, 2, 3, 4, 5], labs = ['a', 'b', 'e', 'c', 'd', 'f'];
    order.forEach((i) => { const el = document.createElement('input'); el.type = 'number'; el.step = 0.5; el.title = labs[i]; el.value = V[i]; el.oninput = () => { V[i] = +el.value || 0; ph.reset(); ph.i = 3; ph.t = 0; }; mat.appendChild(el); ins.push(el); });
    const ph = phaser([1.2, 2.8, 2.8, 3]); ph.i = 3; ph.t = 0;
    chipRow(q(root, '#pre'), [['one solution', [2, 1, 5, 1, -1, 1]], ['no solution (parallel)', [1, 1, 2, 1, 1, 3]], ['infinitely many', [1, 1, 3, 2, 2, 6]]], (v) => { v.forEach((x, i) => { V[i] = x; ins[i].value = x; }); ph.reset(); });
    q(root, '#play').onclick = () => ph.play(); q(root, '#step').onclick = () => ph.next();
    let key = '';
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×'; ph.update(dt, sp);
      const [a, b, e, cc, d, f] = V, det = a * d - b * cc, p = ph.p();
      const kk = [ph.i, ph.t.toFixed(2), V.join(), c1.w].join('|'); if (kk === key) return; key = kk;
      // solve
      let x, y, rank, status;
      const fro = a * a + b * b + cc * cc + d * d;
      if (Math.abs(det) > 1e-9) { x = (e * d - b * f) / det; y = (a * f - e * cc) / det; rank = 2; status = 'unique'; }
      else if (fro > 1e-12) { rank = 1; x = (a * e + cc * f) / fro; y = (b * e + d * f) / fro; const px = a * x + b * y, py = cc * x + d * y; status = Math.hypot(px - e, py - f) < 1e-6 ? 'inf' : 'none'; }
      else { rank = 0; x = y = 0; status = Math.hypot(e, f) < 1e-9 ? 'inf' : 'none'; }
      const C = AI.col();
      // ----- row picture
      { const g = c1.ctx, w = worldView(c1, R); g.clearRect(0, 0, c1.w, c1.h); gridBack(g, C, w, R);
        const line = (A1, B1, E1, col, wd) => { if (Math.abs(A1) + Math.abs(B1) < 1e-9) return; g.strokeStyle = col; g.lineWidth = wd; g.beginPath(); const n2 = A1 * A1 + B1 * B1, cx = (A1 * E1) / n2, cy = (B1 * E1) / n2, dx = -B1, dy = A1; g.moveTo(...w.P(cx - dx * 30, cy - dy * 30)); g.lineTo(...w.P(cx + dx * 30, cy + dy * 30)); g.stroke(); };
        line(a, b, e, C.blue, status === 'inf' && rank === 1 ? 6 : 3); line(cc, d, f, C.orange, 3);
        if (status === 'unique') { g.fillStyle = C.green; g.beginPath(); g.arc(...w.P(x, y), 7, 0, 7); g.fill(); txt(g, `(${F(x, 2)}, ${F(y, 2)})`, ...w.P(x, y).map((v, i) => v + (i ? -12 : 10)), C.green, '600 13px sans-serif'); }
        txt(g, 'ROW picture: each equation is a line', 10, 18, C.fg, '600 13px sans-serif'); txt(g, `${F(a)}x + ${F(b)}y = ${F(e)}`, 10, 36, C.blue, '12px sans-serif'); txt(g, `${F(cc)}x + ${F(d)}y = ${F(f)}`, 10, 52, C.orange, '12px sans-serif');
      }
      // ----- column picture
      { const g = c2.ctx, w = worldView(c2, R); g.clearRect(0, 0, c2.w, c2.h); gridBack(g, C, w, R);
        const col1 = [a, cc], col2 = [b, d], tgt = [e, f];
        if (rank === 1) { const dir = Math.hypot(...col1) > 1e-9 ? col1 : col2, n = Math.hypot(...dir); g.strokeStyle = C.purple; g.globalAlpha = 0.3; g.lineWidth = 10; g.beginPath(); g.moveTo(...w.P(-dir[0] / n * 30, -dir[1] / n * 30)); g.lineTo(...w.P(dir[0] / n * 30, dir[1] / n * 30)); g.stroke(); g.globalAlpha = 1; }
        else if (rank === 2) { g.fillStyle = C.purple; g.globalAlpha = 0.07; g.fillRect(0, 0, c2.w, c2.h); g.globalAlpha = 1; }
        AI.arrow(g, ...w.P(0, 0), ...w.P(...col1), C.blue, 2); AI.arrow(g, ...w.P(0, 0), ...w.P(...col2), C.orange, 2);
        const s1 = ph.i === 0 ? 0 : ph.i === 1 ? p : 1, s2 = ph.i <= 1 ? 0 : ph.i === 2 ? p : 1;
        const t1 = [x * s1 * col1[0], x * s1 * col1[1]], t2 = [y * s2 * col2[0], y * s2 * col2[1]];
        if (s1 > 0) AI.arrow(g, ...w.P(0, 0), ...w.P(...t1), C.blue, 4.5);
        if (s2 > 0) AI.arrow(g, ...w.P(...t1), ...w.P(t1[0] + t2[0], t1[1] + t2[1]), C.orange, 4.5);
        g.fillStyle = C.red; g.beginPath(); g.arc(...w.P(...tgt), 7, 0, 7); g.fill(); txt(g, 'b', ...w.P(...tgt).map((v, i) => v + (i ? -10 : 10)), C.red, '600 14px sans-serif');
        if (status === 'none') { const px = a * x + b * y, py = cc * x + d * y; g.setLineDash([4, 4]); g.strokeStyle = C.red; g.lineWidth = 1.5; g.beginPath(); g.moveTo(...w.P(...tgt)); g.lineTo(...w.P(px, py)); g.stroke(); g.setLineDash([]); }
        txt(g, 'COLUMN picture: reach b with a recipe', 10, 18, C.fg, '600 13px sans-serif'); txt(g, 'blue = column 1, orange = column 2, red dot = b', 10, 36, C.muted, '12px sans-serif');
      }
      const caps = ['Column picture: here are the two columns of A and the target b (red).', status === 'none' ? `<b>Step 1.</b> Stretch column 1 by x = ${F(x, 2)} (the best we can do).` : `<b>Step 1.</b> Stretch column 1 by x = ${F(x, 2)}.`, `<b>Step 2.</b> From its tip, stretch column 2 by y = ${F(y, 2)}.`,
        status === 'unique' ? `<b>Reached b.</b> The recipe (x, y) = (${F(x, 2)}, ${F(y, 2)}) is unique, because the two columns are independent.` : status === 'inf' ? '<b>b is reachable, but in many ways</b>: the columns are dependent, so different recipes give the same result. The recipe shown is just one of them.' : '<b>b is not reachable.</b> Both columns lie on one line (the purple band), and b is off it. The dashed red gap is the closest we can get — the idea behind least squares.'];
      q(root, '#cap').innerHTML = caps[ph.i];
      q(root, '#out').innerHTML = `det = ${F(det, 3)},  rank A = <b class="o">${rank}</b>\n` + (status === 'unique' ? `<span class="g">exactly one solution</span>\nx = ${F(x, 3)},  y = ${F(y, 3)}` : status === 'inf' ? `<span class="b">infinitely many solutions</span>\n(a whole line of them; null space has dimension ${2 - rank})` : '<span class="r">no solution</span>\n(b is outside the column space)');
    });
  };

  /* =====================================================================
   * eigenvis — repeated multiplication / power iteration
   * ===================================================================== */
  AI.labs.eigenvis = function (root) {
    const { view, ctl } = AI.labShell(root, 'Apply M again and again', 'the direction that survives is an eigenvector',
      '<canvas style="height:440px"></canvas><div class="la-cap" id="cap"></div>',
      `<div class="la-ctl-mat" id="mat"></div>
       <div class="chips" id="pre"></div>
       <label>Starting direction v₀ <span class="v" id="av"></span><input type="range" id="ang" min="0" max="360" step="1" value="25"></label>
       <div class="row"><button class="btn primary" id="play">▶ Multiply again and again</button><button class="btn" id="step">Step ▸</button><button class="btn" id="reset">↺ Reset</button></div>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="norm" checked> rescale so the picture stays visible</label>
       <label>Steps per second <span class="v" id="rv"></span><input type="range" id="rate" min="0.2" max="2" step="0.05" value="0.5"></label>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas(AI.$('canvas', view)), R = 3.4, M = [2, 1, 1, 2], MAXK = 40;
    const set = matInputs(q(root, '#mat'), M, () => reset(), 0.5);
    chipRow(q(root, '#pre'), [['symmetric [2 1; 1 2]', [2, 1, 1, 2]], ['[4 1; 2 3]', [4, 1, 2, 3]], ['shrinking [0.9 0.2; 0.1 0.7]', [0.9, 0.2, 0.1, 0.7]], ['rotation 40° (no real eigenvectors)', rot(40)], ['shear [1 1; 0 1]', [1, 1, 0, 1]]], (v) => { set(v); reset(); });
    let Pk = [], k = 0, tt = 0, run = false, key = '';
    const compute = () => { Pk = [I2.slice()]; for (let i = 0; i < MAXK + 1; i++) Pk.push(mul2(M, Pk[i])); };
    const reset = () => { compute(); k = 0; tt = 0; run = false; q(root, '#play').textContent = '▶ Multiply again and again'; key = ''; };
    reset();
    q(root, '#play').onclick = () => { if (k >= MAXK) reset(); run = !run; q(root, '#play').textContent = run ? '❚❚ Pause' : '▶ Multiply again and again'; };
    q(root, '#step').onclick = () => { run = false; tt = 0; if (k < MAXK) k++; q(root, '#play').textContent = '▶ Multiply again and again'; };
    q(root, '#reset').onclick = reset;
    const fro = (P) => Math.hypot(...P);
    AI.loop(root, (dt) => {
      const rate = +q(root, '#rate').value; q(root, '#rv').textContent = F(rate, 2); const ang = +q(root, '#ang').value; q(root, '#av').textContent = ang + '°';
      if (run) { tt += dt * rate; while (tt >= 1) { tt -= 1; k++; if (k >= MAXK) { k = MAXK; tt = 0; run = false; q(root, '#play').textContent = '↺ Restart'; break; } } }
      const norm = q(root, '#norm').checked, kk = [k, tt.toFixed(3), M.join(), ang, norm, c.w].join('|'); if (kk === key) return; key = kk;
      const P0 = Pk[k], P1 = Pk[Math.min(k + 1, MAXK + 1)], t = AI.smooth(tt), P = blend(P0, P1, t);
      const sc = norm ? 2.4 / Math.max(fro(P), 1e-9) * Math.SQRT2 : 1;
      const C = AI.col(), g = c.ctx, w = worldView(c, R); g.clearRect(0, 0, c.w, c.h); gridBack(g, C, w, R);
      const ev = eig2(M);
      if (ev.re) ev.e.forEach((x, i) => { g.strokeStyle = C.yellow; g.setLineDash([7, 5]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(...w.P(-x.v[0] * 30, -x.v[1] * 30)); g.lineTo(...w.P(x.v[0] * 30, x.v[1] * 30)); g.stroke(); g.setLineDash([]); txt(g, `λ=${F(x.lam, 2)}`, ...w.P(x.v[0] * (R - 0.6), x.v[1] * (R - 0.6)).map((v, j) => v + (j ? 0 : 6)), C.yellow, '600 12px sans-serif'); });
      // image of the unit circle
      g.beginPath(); for (let i = 0; i <= 90; i++) { const th = (i / 90) * 2 * PI, [x, y] = T(P, Math.cos(th), Math.sin(th)), p = w.P(x * sc, y * sc); i ? g.lineTo(...p) : g.moveTo(...p); }
      g.closePath(); g.fillStyle = C.purple; g.globalAlpha = 0.14; g.fill(); g.globalAlpha = 1; g.strokeStyle = C.purple; g.lineWidth = 2; g.stroke();
      const a0 = (ang * PI) / 180, v0 = [Math.cos(a0), Math.sin(a0)];
      // ghost trail of earlier directions
      for (let j = 0; j < k && j < 25; j++) { const s = norm ? 2.4 * Math.SQRT2 / Math.max(fro(Pk[j]), 1e-9) : 1, [x, y] = T(Pk[j], ...v0), n = Math.hypot(x, y) || 1; g.globalAlpha = 0.12 + 0.3 * (j / Math.max(k, 1)); AI.arrow(g, ...w.P(0, 0), ...w.P((x / n) * 2.6, (y / n) * 2.6), C.orange, 1.5); g.globalAlpha = 1; }
      const [vx, vy] = T(P, ...v0), vn = Math.hypot(vx, vy) || 1;
      AI.arrow(g, ...w.P(0, 0), ...w.P(vx * sc, vy * sc), C.orange, 4); g.fillStyle = C.orange; g.beginPath(); g.arc(...w.P(vx * sc, vy * sc), 5, 0, 7); g.fill();
      AI.arrow(g, ...w.P(0, 0), ...w.P(...v0), C.muted, 2); txt(g, 'v₀', ...w.P(...v0).map((v, i) => v + (i ? -6 : 6)), C.muted, '12px sans-serif');
      txt(g, 'purple = where the unit circle goes (Mᵏ applied to it)    orange = Mᵏv₀    yellow = eigen-directions', 10, 18, C.muted, '12px sans-serif');
      const Pn = Pk[k + 1], [ux, uy] = T(Pk[k], ...v0), [wx, wy] = T(Pn, ...v0), ratio = Math.hypot(wx, wy) / (Math.hypot(ux, uy) || 1);
      let ang2 = ''; if (ev.re) { const dom = Math.abs(ev.e[0].lam) >= Math.abs(ev.e[1].lam) ? ev.e[0] : ev.e[1]; const cs = Math.abs(ux * dom.v[0] + uy * dom.v[1]) / (Math.hypot(ux, uy) || 1); ang2 = `angle to dominant eigenvector = ${F((Math.acos(AI.clamp(cs, 0, 1)) * 180) / PI, 1)}°`; }
      q(root, '#cap').innerHTML = k === 0 ? 'Press play. Each step multiplies by M once more: <b>orange</b> is the current vector, faint orange arrows are its history. Watch its direction.' : !ev.re ? `Step ${k}: M rotates every direction, so no direction survives — complex eigenvalues (no real eigenvector).` : `Step ${k}: the vector keeps turning toward the dominant (largest |λ|) eigen-direction, and each step multiplies its length by about <b>${F(ratio, 3)}</b>.`;
      q(root, '#out').innerHTML = `M = ${fmtM(M)}\n` + (ev.re ? `λ₁ = <b class="o">${F(ev.e[0].lam, 3)}</b>, v₁ = (${F(ev.e[0].v[0], 3)}, ${F(ev.e[0].v[1], 3)})\nλ₂ = <b class="o">${F(ev.e[1].lam, 3)}</b>, v₂ = (${F(ev.e[1].v[0], 3)}, ${F(ev.e[1].v[1], 3)})` : `λ = ${F(ev.tr / 2, 3)} ± ${F(ev.im, 3)}i  <span class="m">(rotation-like)</span>`) + `\nstep k = ${k}   |Mᵏv₀| = ${F(vn, 3)}\nstretch this step = ${F(ratio, 3)}   ${ang2}`;
    });
  };

  /* =====================================================================
   * pca2d — principal axes, variance vs direction, projection animation
   * ===================================================================== */
  AI.labs.pca2d = function (root) {
    const { view, ctl } = AI.labShell(root, 'PCA in two dimensions', 'the direction with the most variance is also the direction that loses least',
      '<canvas id="c1" style="height:400px"></canvas><canvas id="c2" style="height:150px"></canvas><div class="la-cap" id="cap"></div>',
      `<label>cloud angle <span class="v" id="thv"></span><input type="range" id="th" min="0" max="180" step="1" value="35"></label>
       <label>long spread <span class="v" id="s1v"></span><input type="range" id="s1" min="0.5" max="3" step="0.05" value="2.2"></label>
       <label>short spread <span class="v" id="s2v"></span><input type="range" id="s2" min="0.1" max="1.5" step="0.05" value="0.6"></label>
       <label>my direction φ <span class="v" id="phv"></span><input type="range" id="ph" min="0" max="180" step="1" value="100"></label>
       <div class="row"><button class="btn primary" id="proj">▶ Project onto my direction</button><button class="btn" id="sweep">↻ Sweep φ</button><button class="btn" id="snap">Snap φ to PC1</button></div>
       ${speedHTML(1)}
       <div class="readout" id="out"></div>`);
    const cvs = view.querySelectorAll('canvas'), c1 = AI.canvas(cvs[0]), c2 = AI.canvas(cvs[1]), R = 6.5, N = 70;
    const r = AI.rng(7), Z = Array.from({ length: N }, () => [AI.randn(r), AI.randn(r)]);
    let X = [], cov = [0, 0, 0], pr = 0, tgtP = 0, sweeping = false, snapping = false, key = '', dk = '';
    const sl = (id) => q(root, '#' + id);
    const gen = () => {
      const th = (+sl('th').value * PI) / 180, s1 = +sl('s1').value, s2 = +sl('s2').value, co = Math.cos(th), si = Math.sin(th);
      X = Z.map(([a, b]) => [co * s1 * a - si * s2 * b, si * s1 * a + co * s2 * b]);
      const mx = X.reduce((s, p) => s + p[0], 0) / N, my = X.reduce((s, p) => s + p[1], 0) / N; X = X.map((p) => [p[0] - mx, p[1] - my]);
      cov = [X.reduce((s, p) => s + p[0] * p[0], 0) / N, X.reduce((s, p) => s + p[0] * p[1], 0) / N, X.reduce((s, p) => s + p[1] * p[1], 0) / N];
    };
    q(root, '#proj').onclick = () => { tgtP = tgtP ? 0 : 1; q(root, '#proj').textContent = tgtP ? '◀ Release points' : '▶ Project onto my direction'; };
    q(root, '#sweep').onclick = () => { sweeping = !sweeping; snapping = false; q(root, '#sweep').textContent = sweeping ? '❚❚ Stop sweep' : '↻ Sweep φ'; };
    q(root, '#snap').onclick = () => { snapping = true; sweeping = false; q(root, '#sweep').textContent = '↻ Sweep φ'; };
    AI.loop(root, (dt) => {
      const sp = +sl('sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      const gk = [sl('th').value, sl('s1').value, sl('s2').value].join(); if (gk !== dk) { dk = gk; gen(); }
      const [sxx, sxy, syy] = cov, tr = sxx + syy, l1 = tr / 2 + Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy), l2 = tr - l1, alpha = 0.5 * Math.atan2(2 * sxy, sxx - syy);
      let phi = +sl('ph').value;
      if (sweeping) phi = (phi + dt * sp * 22) % 180;
      if (snapping) { const aD = ((alpha * 180) / PI + 180) % 180; let df = ((aD - phi + 270) % 180) - 90; phi += df * (1 - Math.exp(-dt * sp * 2.2)); if (Math.abs(df) < 0.2) { phi = aD; snapping = false; } }
      sl('ph').value = phi; pr += (tgtP - pr) * (1 - Math.exp(-dt * sp * 1.2));
      const kk = [phi.toFixed(2), pr.toFixed(3), dk, c1.w].join('|'); if (kk === key) return; key = kk;
      sl('thv').textContent = sl('th').value + '°'; sl('s1v').textContent = F(+sl('s1').value, 2); sl('s2v').textContent = F(+sl('s2').value, 2); sl('phv').textContent = F(phi, 0) + '°';
      const ph = (phi * PI) / 180, u = [Math.cos(ph), Math.sin(ph)], varU = u[0] * u[0] * sxx + 2 * u[0] * u[1] * sxy + u[1] * u[1] * syy;
      const C = AI.col(); let g = c1.ctx, w = worldView(c1, R); g.clearRect(0, 0, c1.w, c1.h); gridBack(g, C, w, R);
      g.strokeStyle = C.orange; g.lineWidth = 2.5; g.beginPath(); g.moveTo(...w.P(-u[0] * 30, -u[1] * 30)); g.lineTo(...w.P(u[0] * 30, u[1] * 30)); g.stroke();
      const pc1 = [Math.cos(alpha), Math.sin(alpha)], pc2 = [-pc1[1], pc1[0]];
      g.setLineDash([6, 4]); AI.arrow(g, ...w.P(0, 0), ...w.P(pc1[0] * 2 * Math.sqrt(l1), pc1[1] * 2 * Math.sqrt(l1)), C.yellow, 2.5); AI.arrow(g, ...w.P(0, 0), ...w.P(pc2[0] * 2 * Math.sqrt(l2), pc2[1] * 2 * Math.sqrt(l2)), C.yellow, 2.5); g.setLineDash([]);
      X.forEach((p) => {
        const d = p[0] * u[0] + p[1] * u[1], q2 = [p[0] + (d * u[0] - p[0]) * pr, p[1] + (d * u[1] - p[1]) * pr];
        if (pr > 0.01) { g.strokeStyle = C.red; g.globalAlpha = 0.35; g.lineWidth = 1; g.beginPath(); g.moveTo(...w.P(...p)); g.lineTo(...w.P(...q2)); g.stroke(); g.globalAlpha = 1; }
        g.fillStyle = pr > 0.5 ? C.green : C.blue; g.beginPath(); g.arc(...w.P(...q2), 3.6, 0, 7); g.fill();
      });
      txt(g, 'orange line = your direction φ     yellow = true principal axes (length 2σ)', 10, 18, C.muted, '12px sans-serif');
      // variance curve
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const v = AI.view(c2, 0, 180, 0, l1 * 1.15 + 0.01, { l: 40, r: 14, t: 14, b: 24 }); AI.axes(g, v);
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); for (let a = 0; a <= 180; a += 2) { const t = (a * PI) / 180, vv = Math.cos(t) ** 2 * sxx + 2 * Math.cos(t) * Math.sin(t) * sxy + Math.sin(t) ** 2 * syy; a ? g.lineTo(v.X(a), v.Y(vv)) : g.moveTo(v.X(a), v.Y(vv)); } g.stroke();
      g.strokeStyle = C.orange; g.beginPath(); g.moveTo(v.X(phi), v.pad.t); g.lineTo(v.X(phi), v.pad.t + v.ph); g.stroke(); g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(phi), v.Y(varU), 6, 0, 7); g.fill();
      txt(g, 'variance of the data projected on direction φ  (x-axis: φ in degrees)', v.pad.l + 6, 12, C.muted, '11px sans-serif');
      const aD = ((alpha * 180) / PI + 180) % 180;
      q(root, '#cap').innerHTML = pr < 0.05 ? `Each dot is a data point. Your orange line has direction φ = ${F(phi, 0)}°. Press <b>Project</b> to slide every point onto it (red = the distance each point travels).` : `Points are being squeezed onto your line (green). Variance kept: <b>${F(varU, 2)}</b>; information lost (mean squared distance): <b>${F(tr - varU, 2)}</b>. They always add up to the total variance ${F(tr, 2)}.`;
      q(root, '#out').innerHTML = `total variance = ${F(tr, 3)}\nvariance along φ = <span class="o">${F(varU, 3)}</span>  (${F((100 * varU) / tr, 1)}%)\nlost (⟂ to φ)    = <span class="r">${F(tr - varU, 3)}</span>\nPC1: angle ${F(aD, 1)}°, λ₁ = ${F(l1, 3)} (${F((100 * l1) / tr, 1)}%)\nPC2: λ₂ = ${F(l2, 3)}`;
    });
  };

  /* =====================================================================
   * svdimage — low-rank approximation of an image
   * ===================================================================== */
  const jacobiEig = (S, n) => {
    const A = S.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j]; if (off < 1e-22) break;
      for (let p = 0; p < n - 1; p++) for (let qq = p + 1; qq < n; qq++) {
        if (Math.abs(A[p][qq]) < 1e-15) continue;
        const th = (A[qq][qq] - A[p][p]) / (2 * A[p][qq]), t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cc = 1 / Math.sqrt(t * t + 1), s = t * cc;
        for (let k = 0; k < n; k++) { const x = A[k][p], y = A[k][qq]; A[k][p] = cc * x - s * y; A[k][qq] = s * x + cc * y; }
        for (let k = 0; k < n; k++) { const x = A[p][k], y = A[qq][k]; A[p][k] = cc * x - s * y; A[qq][k] = s * x + cc * y; }
        for (let k = 0; k < n; k++) { const x = V[k][p], y = V[k][qq]; V[k][p] = cc * x - s * y; V[k][qq] = s * x + cc * y; }
      }
    }
    return { vals: A.map((r, i) => r[i]), V };
  };
  /** Thin SVD of an m×n matrix (array of rows) via Jacobi on AᵀA. Returns {s, U[i][k], V[j][k]} sorted by singular value. */
  const svd = (A) => {
    const m = A.length, n = A[0].length, S = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => { let s = 0; for (let k = 0; k < m; k++) s += A[k][i] * A[k][j]; return s; }));
    const { vals, V } = jacobiEig(S, n), idx = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]);
    const s = idx.map((i) => Math.sqrt(Math.max(vals[i], 0))), Vs = V.map((row) => idx.map((i) => row[i]));
    const U = Array.from({ length: m }, (_, i) => idx.map((_, kk) => { if (s[kk] < 1e-9) return 0; let v = 0; for (let j = 0; j < n; j++) v += A[i][j] * Vs[j][kk]; return v / s[kk]; }));
    return { s, U, V: Vs };
  };
  AI.labs.svdimage = function (root) {
    const N = 24;
    const { view, ctl } = AI.labShell(root, 'SVD: an image from a few rank-1 layers', 'A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + …  keep only the first k',
      '<canvas id="c1" style="height:250px"></canvas><canvas id="c2" style="height:150px"></canvas><div class="la-cap" id="cap"></div>',
      `<label>Picture <select id="img"><option value="face">smiley face</option><option value="wave">waves + blob</option><option value="noise">random noise (no structure)</option></select></label>
       <label>rank k <span class="v" id="kv"></span><input type="range" id="k" min="1" max="${N}" step="1" value="3"></label>
       <div class="row"><button class="btn primary" id="play">▶ Add layers one by one</button><button class="btn" id="reset">↺ k = 1</button></div>
       ${speedHTML(1)}
       <div class="readout" id="out"></div>`);
    const cvs = view.querySelectorAll('canvas'), c1 = AI.canvas(cvs[0]), c2 = AI.canvas(cvs[1]);
    const gens = {
      face: (i, j) => { const x = (j / (N - 1)) * 2 - 1, y = (i / (N - 1)) * 2 - 1, r = Math.hypot(x, y); let v = 0.12; if (Math.abs(r - 0.82) < 0.09) v = 0.95; if (Math.hypot(x + 0.32, y + 0.25) < 0.13 || Math.hypot(x - 0.32, y + 0.25) < 0.13) v = 0.95; if (y > 0.12 && Math.abs(Math.hypot(x, y - 0.05) - 0.5) < 0.08 && r < 0.78) v = 0.95; return v; },
      wave: (i, j) => 0.5 + 0.28 * Math.sin(i * 0.5) * Math.cos(j * 0.35) + 0.3 * Math.exp(-((i - 8) ** 2 + (j - 15) ** 2) / 22) + 0.12 * Math.sin((i + j) * 0.3),
      noise: (() => { const r = AI.rng(5); return () => r(); })(),
    };
    const data = {}, dec = {};
    Object.keys(gens).forEach((nm) => { data[nm] = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => gens[nm](i, j))); dec[nm] = svd(data[nm]); });
    let kk = 3, run = false, acc = 0, key = '';
    const ks = q(root, '#k');
    q(root, '#play').onclick = () => { if (+ks.value >= N) ks.value = 1; run = !run; acc = 0; q(root, '#play').textContent = run ? '❚❚ Pause' : '▶ Add layers one by one'; };
    q(root, '#reset').onclick = () => { ks.value = 1; run = false; q(root, '#play').textContent = '▶ Add layers one by one'; };
    AI.loop(root, (dt) => {
      const sp = +q(root, '#sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      if (run) { acc += dt * sp * 0.9; while (acc >= 1) { acc -= 1; ks.value = +ks.value + 1; if (+ks.value >= N) { run = false; q(root, '#play').textContent = '▶ Add layers one by one'; } } }
      const nm = q(root, '#img').value; kk = +ks.value;
      const kkey = [nm, kk, c1.w].join('|'); if (kkey === key) return; key = kkey; q(root, '#kv').textContent = kk;
      const D = dec[nm], A = data[nm], { s, U, V } = D;
      const Ak = Array.from({ length: N }, () => new Array(N).fill(0));
      for (let t = 0; t < kk; t++) for (let i = 0; i < N; i++) { const ui = s[t] * U[i][t]; for (let j = 0; j < N; j++) Ak[i][j] += ui * V[j][t]; }
      const tot = s.reduce((a, v) => a + v * v, 0), kept = s.slice(0, kk).reduce((a, v) => a + v * v, 0), err = Math.sqrt(Math.max(0, 1 - kept / tot));
      const C = AI.col(); let g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h);
      const px = Math.min((c1.w - 60) / (2 * N), (c1.h - 30) / N), left = (c1.w - 2 * N * px - 30) / 2;
      const draw = (M, x0, title) => { for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const v = AI.clamp(M[i][j], 0, 1) * 255; g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x0 + j * px, 24 + i * px, px + 0.5, px + 0.5); } txt(g, title, x0, 15, C.fg, '600 12px sans-serif'); };
      draw(A, left, 'original (24×24 = 576 numbers)'); draw(Ak, left + N * px + 30, `rank-${kk} approximation`);
      g = c2.ctx; g.clearRect(0, 0, c2.w, c2.h); const v = AI.view(c2, 0, N + 1, 0, s[0] * 1.1, { l: 38, r: 10, t: 14, b: 22 }); AI.axes(g, v);
      for (let i = 0; i < N; i++) { g.fillStyle = i < kk ? C.orange : C.border; const h = (s[i] / (s[0] * 1.1)) * v.ph; g.fillRect(v.X(i + 1) - v.pw / (N + 1) * 0.35, v.pad.t + v.ph - h, (v.pw / (N + 1)) * 0.7, h); }
      txt(g, 'singular values σ₁ ≥ σ₂ ≥ … (orange = the k kept layers)', v.pad.l + 6, 11, C.muted, '11px sans-serif');
      const store = kk * (2 * N + 1);
      q(root, '#cap').innerHTML = nm === 'noise' ? 'Noise has no structure to exploit: the singular values decay slowly, and you need almost all layers before the picture looks right. Compression works on <i>structured</i> data.' : `Each layer σᵢuᵢvᵢᵀ is a rank-1 picture. The first few carry the big shapes; later layers add fine detail. With k = ${kk} you keep <b>${F((100 * kept) / tot, 1)}%</b> of the “energy”.`;
      q(root, '#out').innerHTML = `rank k = ${kk}\nenergy kept Σσᵢ²/Σσ² = <span class="g">${F((100 * kept) / tot, 2)}%</span>\nrelative error ‖A−Aₖ‖/‖A‖ = <span class="r">${F(100 * err, 2)}%</span>\nstorage: k(m+n+1) = ${store} numbers  vs  ${N * N}  <span class="m">(${F((100 * store) / (N * N), 0)}%)</span>`;
    });
  };

  /* =====================================================================
   * lsproj — least squares as projection (3 observations, line y = w0 + w1 x)
   * ===================================================================== */
  AI.labs.lsproj = function (root) {
    const { view, ctl } = AI.labShell(root, 'Least squares = projection', 'left: the line through the data · right: the same problem in observation space',
      '<canvas id="c1" style="height:260px"></canvas><canvas id="c2" style="height:380px;touch-action:none;cursor:grab"></canvas><div class="la-cap" id="cap"></div>',
      `<div style="color:var(--muted)">data y = (y₁, y₂, y₃) at x = (0, 1, 2)</div>
       <div class="la-ctl-mat" style="grid-template-columns:1fr 1fr 1fr">${[1, 2, 3].map((i) => `<label><span>y${i} <span class="v" id="vy${i}"></span></span><input type="range" id="y${i}" min="-1" max="4" step="0.1" value="${[1, 2, 2][i - 1]}"></label>`).join('')}</div>
       <div style="color:var(--muted)">your line: ŷ = w₀ + w₁x</div>
       <label>w₀ (intercept) <span class="v" id="vw0"></span><input type="range" id="w0" min="-1" max="3" step="0.01" value="0.2"></label>
       <label>w₁ (slope) <span class="v" id="vw1"></span><input type="range" id="w1" min="-1" max="2" step="0.01" value="1.3"></label>
       <div class="row"><button class="btn primary" id="snap">Snap to least squares</button><button class="btn" id="perf">y on the plane (perfect fit)</button><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="rot"> slowly rotate</label></div>
       ${speedHTML(1)}
       <div class="readout" id="out"></div>`);
    const cvs = view.querySelectorAll('canvas'), c1 = AI.canvas(cvs[0]), c2 = AI.canvas(cvs[1]), xs = [0, 1, 2];
    const sl = (id) => q(root, '#' + id);
    let yaw = 2.2, pitch = 0.4, snap = false, key = '', drag = null;
    sl('snap').onclick = () => (snap = true);
    sl('perf').onclick = () => { sl('y1').value = 1; sl('y2').value = 2; sl('y3').value = 3; snap = true; };
    ['y1', 'y2', 'y3', 'w0', 'w1'].forEach((id) => sl(id).addEventListener('input', () => { if (id[0] === 'w') snap = false; }));
    cvs[1].addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY, yaw, pitch]; cvs[1].setPointerCapture(e.pointerId); cvs[1].style.cursor = 'grabbing'; });
    cvs[1].addEventListener('pointermove', (e) => { if (drag) { yaw = drag[2] + (e.clientX - drag[0]) * 0.008; pitch = AI.clamp(drag[3] + (e.clientY - drag[1]) * 0.008, -0.2, 1.4); } });
    cvs[1].addEventListener('pointerup', () => { drag = null; cvs[1].style.cursor = 'grab'; });
    const proj3 = (p) => { const x1 = p[0] * Math.cos(yaw) - p[1] * Math.sin(yaw), y1 = p[0] * Math.sin(yaw) + p[1] * Math.cos(yaw); return [x1, p[2] * Math.cos(pitch) + y1 * Math.sin(pitch)]; };
    AI.loop(root, (dt) => {
      const sp = +sl('sp').value; q(root, '#spv').textContent = F(sp, 2) + '×';
      const y = [+sl('y1').value, +sl('y2').value, +sl('y3').value];
      // normal equations: XtX w = Xty with X=[1 x]
      const n = 3, sx = 3, sxx = 5, sy = y[0] + y[1] + y[2], sxy = y[1] + 2 * y[2], det = n * sxx - sx * sx, b0 = (sxx * sy - sx * sxy) / det, b1 = (n * sxy - sx * sy) / det;
      if (snap) { const k = 1 - Math.exp(-dt * sp * 2.5); sl('w0').value = +sl('w0').value + (b0 - +sl('w0').value) * k; sl('w1').value = +sl('w1').value + (b1 - +sl('w1').value) * k; if (Math.abs(+sl('w0').value - b0) + Math.abs(+sl('w1').value - b1) < 2e-3) { sl('w0').value = b0; sl('w1').value = b1; snap = false; } }
      if (sl('rot').checked) yaw += dt * sp * 0.2;
      const w0 = +sl('w0').value, w1 = +sl('w1').value;
      const kk = [y.join(), w0.toFixed(3), w1.toFixed(3), yaw.toFixed(3), pitch.toFixed(3), c1.w, c2.w].join('|'); if (kk === key) return; key = kk;
      ['y1', 'y2', 'y3'].forEach((id, i) => (q(root, '#v' + id).textContent = F(y[i], 1))); q(root, '#vw0').textContent = F(w0, 2); q(root, '#vw1').textContent = F(w1, 2);
      const q3 = xs.map((x) => w0 + w1 * x), e = y.map((v, i) => v - q3[i]), sse = e.reduce((s, v) => s + v * v, 0), pb = xs.map((x) => b0 + b1 * x), eb = y.map((v, i) => v - pb[i]), sseb = eb.reduce((s, v) => s + v * v, 0);
      const C = AI.col();
      // ---- left: data space
      { const g = c1.ctx; g.clearRect(0, 0, c1.w, c1.h); const v = AI.view(c1, -0.5, 2.5, -1, 4.5, { l: 34, r: 12, t: 22, b: 22 }); AI.axes(g, v);
        g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(-0.5), v.Y(b0 - 0.5 * b1)); g.lineTo(v.X(2.5), v.Y(b0 + 2.5 * b1)); g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.green; g.lineWidth = 3; g.beginPath(); g.moveTo(v.X(-0.5), v.Y(w0 - 0.5 * w1)); g.lineTo(v.X(2.5), v.Y(w0 + 2.5 * w1)); g.stroke();
        xs.forEach((x, i) => { g.strokeStyle = C.red; g.lineWidth = 2.5; g.beginPath(); g.moveTo(v.X(x), v.Y(y[i])); g.lineTo(v.X(x), v.Y(q3[i])); g.stroke(); g.fillStyle = C.blue; g.beginPath(); g.arc(v.X(x), v.Y(y[i]), 6, 0, 7); g.fill(); g.fillStyle = C.green; g.beginPath(); g.arc(v.X(x), v.Y(q3[i]), 4, 0, 7); g.fill(); });
        txt(g, 'blue = data, green = your line, red = residuals, grey dashed = least-squares line', 38, 14, C.muted, '11px sans-serif'); }
      // ---- right: 3D observation space
      { const g = c2.ctx, s = Math.min(c2.w, c2.h) / 7.5, ox = c2.w / 2, oy = c2.h * 0.62; g.clearRect(0, 0, c2.w, c2.h);
        const P = (p) => { const [a, b] = proj3(p); return [ox + a * s, oy - b * s]; };
        const c1v = [1, 1, 1], c2v = [0, 1, 2];
        // axes
        [[[4, 0, 0], 'obs 1'], [[0, 4, 0], 'obs 2'], [[0, 0, 4], 'obs 3']].forEach(([e1, lab]) => { g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(...P([0, 0, 0])); g.lineTo(...P(e1)); g.stroke(); txt(g, lab, ...P(e1.map((v) => v * 1.08)), C.muted, '11px sans-serif'); });
        // plane patch spanned by 1 and x
        const u0 = -1.5, u1 = 3, v0 = -1.2, v1 = 1.2, pp = (a, b) => P([a * c1v[0] + b * c2v[0], a * c1v[1] + b * c2v[1], a * c1v[2] + b * c2v[2]]);
        g.beginPath(); [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].forEach(([a, b], i) => (i ? g.lineTo(...pp(a, b)) : g.moveTo(...pp(a, b)))); g.closePath(); g.fillStyle = C.purple; g.globalAlpha = 0.16; g.fill(); g.globalAlpha = 0.7; g.strokeStyle = C.purple; g.lineWidth = 1; g.stroke(); g.globalAlpha = 0.25;
        for (let a = Math.ceil(u0); a <= u1; a++) { g.beginPath(); g.moveTo(...pp(a, v0)); g.lineTo(...pp(a, v1)); g.stroke(); } for (let b = -1; b <= 1; b++) { g.beginPath(); g.moveTo(...pp(u0, b)); g.lineTo(...pp(u1, b)); g.stroke(); } g.globalAlpha = 1;
        txt(g, 'column space = all combinations w₀·1 + w₁·x', 10, c2.h - 10, C.purple, '12px sans-serif');
        AI.arrow(g, ...P([0, 0, 0]), ...P(c1v), C.purple, 2); txt(g, '1', ...P(c1v).map((v, i) => v + (i ? -6 : 6)), C.purple, '600 12px sans-serif');
        AI.arrow(g, ...P([0, 0, 0]), ...P(c2v), C.purple, 2); txt(g, 'x', ...P(c2v).map((v, i) => v + (i ? -6 : 6)), C.purple, '600 12px sans-serif');
        const pbV = [pb[0], pb[1], pb[2]];
        g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.lineWidth = 1.5; g.beginPath(); g.moveTo(...P(y)); g.lineTo(...P(pbV)); g.stroke(); g.setLineDash([]);
        AI.arrow(g, ...P([0, 0, 0]), ...P(y), C.blue, 3); txt(g, 'y (data)', ...P(y).map((v, i) => v + (i ? -8 : 8)), C.blue, '600 13px sans-serif');
        AI.arrow(g, ...P([0, 0, 0]), ...P(q3), C.green, 3); txt(g, 'Xw (yours)', ...P(q3).map((v, i) => v + (i ? 16 : 8)), C.green, '600 13px sans-serif');
        g.strokeStyle = C.red; g.lineWidth = 3; g.beginPath(); g.moveTo(...P(y)); g.lineTo(...P(q3)); g.stroke();
        txt(g, 'drag to rotate', c2.w - 10, 16, C.muted, '11px sans-serif', 'right'); }
      const z0 = (v) => (Math.abs(v) < 1e-6 ? 0 : v), xe = z0(eb.reduce((s, v) => s + v, 0)), xe2 = z0(eb[1] + 2 * eb[2]), opt = Math.abs(w0 - b0) + Math.abs(w1 - b1) < 0.01;
      q(root, '#cap').innerHTML = opt ? '<b>At the least-squares fit</b>: Xw is the point of the plane closest to y, so the residual (red) is exactly perpendicular to the plane: it is orthogonal to both the “1” column and the “x” column.' : 'Slide w₀, w₁: Xw moves around the purple plane (every line you can draw is a point of that plane). The red gap is the residual. It is shortest — and perpendicular to the plane — only at the projection.';
      q(root, '#out').innerHTML = `your SSE = Σ(yᵢ−ŷᵢ)² = <span class="r">${F(sse, 3)}</span>\nbest  SSE = <span class="g">${F(sseb, 3)}</span>   at w = (${F(b0, 3)}, ${F(b1, 3)})\nnormal eq.  XᵀX w = Xᵀy :\n  [3 3; 3 5] w = (${F(sy, 2)}, ${F(sxy, 2)})\nXᵀe at best fit = (${F(xe, 3)}, ${F(xe2, 3)})  <span class="m">→ 0: residual ⟂ columns</span>`;
    });
  };
})();
