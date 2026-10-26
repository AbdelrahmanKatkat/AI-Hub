/* Computer-vision Module 3 labs (CNNs from first principles). Requires assets/lib.js (+ labs-cv.js for AI.cv).
 *   shiftdemo, paramcount, mcconv, outsize, pooling, rfield, normaxes, normlab, dropoutdemo, cnntrain, cnnviz
 * All animations are dt-based, slow by default, with speed slider, pause and step. */
(function () {
  const q = (root, s) => AI.$(s, root);
  const mono = 'ui-monospace,Menlo,Consolas,monospace';
  const sgn = (v) => (v < 0 ? '−' : '') + Math.abs(v);
  const fx = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? '0' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(d));
  const ease = (t) => AI.smooth(t);
  const com = (n) => Math.round(n).toLocaleString('en-US');

  /** canvas with a fixed design size; everything is drawn in design units and scaled to the element width */
  function dcanvas(view, W, H) {
    const cv = AI.$('canvas', view);
    cv.style.aspectRatio = W + ' / ' + H; cv.style.height = 'auto';
    const c = AI.canvas(cv);
    c.W = W; c.H = H;
    c.begin = () => { const g = c.ctx; g.clearRect(0, 0, c.w, c.h); g.save(); g.scale(c.w / W, c.w / W); return g; };
    c.end = () => c.ctx.restore();
    c.pt = (e) => { const r = cv.getBoundingClientRect(); return [((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * H) / r.height]; };
    return c;
  }
  const playHTML = (sp, unit, min = 0.1, max = 4, step = 0.1) =>
    `<label>Speed <span class="v" id="spv">${sp} ${unit}</span><input type="range" id="sp" min="${min}" max="${max}" step="${step}" value="${sp}"></label>
     <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="restart">⟲ Restart</button></div>`;
  function player(root, unit, hooks) {
    const st = { playing: true }, b = q(root, '#play');
    const setp = (v) => { st.playing = v; b.textContent = v ? '⏸ Pause' : '▶ Play'; };
    b.onclick = () => setp(!st.playing);
    q(root, '#step').onclick = () => { setp(false); hooks.onStep && hooks.onStep(); };
    q(root, '#restart').onclick = () => hooks.onRestart && hooks.onRestart();
    q(root, '#sp').addEventListener('input', () => (q(root, '#spv').textContent = q(root, '#sp').value + ' ' + unit));
    st.speed = () => +q(root, '#sp').value; st.pause = () => setp(false);
    return st;
  }
  const slider = (id, label, min, max, val, step = 1) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const chipRow = (parent, items, on, cur) => {
    const bs = items.map(([k, label]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = label; b.onclick = () => { on(k); mark(k); }; parent.appendChild(b); b.dataset.k = k; return b; });
    const mark = (k) => bs.forEach((b) => (b.style.borderColor = b.dataset.k === String(k) ? 'var(--blue)' : ''));
    mark(cur); return mark;
  };
  const divcol = (v, m, C) => { const t = Math.min(1, Math.abs(v) / (m || 1)); return v >= 0 ? AI.mix(C.bg2, C.green, t) : AI.mix(C.bg2, C.orange, t); };
  const heatcol = (t, C) => AI.mix(C.bg2, C.red, Math.min(1, Math.max(0, t)));
  const gray = (v) => { const g = Math.round(255 * AI.clamp(v, 0, 1)); return `rgb(${g},${g},${g})`; };
  const txtOn = (v) => (v > 0.5 ? '#000' : '#fff');
  const hash = (a, b, c, d) => { let h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791) ^ (d * 2654435761 | 0); h = Math.imul(h ^ (h >>> 15), 2246822519); h ^= h >>> 13; return h >>> 0; };
  function label(g, t, x, y, color, size = 11, align = 'left') { g.fillStyle = color; g.font = `${size}px sans-serif`; g.textAlign = align; g.textBaseline = 'alphabetic'; g.fillText(t, x, y); }

  /* ================= 3.1 shiftdemo ================= */
  AI.labs.shiftdemo = function (root) {
    const N = 16, T = [1, 2, 1], PM = N - 3;
    const { view } = AI.labShell(root, 'Why sharing weights matters', 'a pattern slides along a row of pixels: one private-weight unit versus one shared kernel',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `${slider('pos', 'Pattern position', 0, PM, 2)}${slider('tr', 'MLP unit was trained with the pattern at', 0, PM, 5)}
       ${playHTML(0.8, 'moves / s', 0.2, 3, 0.1)}`);
    const c = dcanvas(view, 640, 400), tick = AI.ticker();
    let p = 2, pf = 2, dir = 1, P0 = 5, last = '';
    const st = player(root, 'moves / s', { onStep: () => advance(), onRestart: () => { p = 0; dir = 1; sync(); } });
    const sync = () => { q(root, '#pos').value = p; };
    const advance = () => { if (p + dir > PM || p + dir < 0) dir = -dir; p += dir; sync(); };
    q(root, '#pos').addEventListener('input', () => { p = +q(root, '#pos').value; st.pause(); });
    q(root, '#tr').addEventListener('input', () => { P0 = +q(root, '#tr').value; });
    const xs = (pp) => { const x = new Array(N).fill(0); T.forEach((v, i) => (x[pp + i] = v)); return x; };
    const fc = (pp) => { const x = xs(pp); let s = 0; for (let i = 0; i < 3; i++) s += T[i] * x[P0 + i]; return s; };
    const cv = (x) => { const o = []; for (let j = 0; j <= PM; j++) o.push(T[0] * x[j] + T[1] * x[j + 1] + T[2] * x[j + 2]); return o; };
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, st.speed()); for (let i = 0; i < n; i++) advance(); }
      pf += (p - pf) * Math.min(1, dt * 7);
      q(root, '#posv').textContent = p; q(root, '#trv').textContent = P0;
      const key = [p, P0, pf.toFixed(2), c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin(), x = xs(p), x0 = 40, cw = 35;
      const X = (i) => x0 + i * cw;
      label(g, 'pixels x  (a pattern [1 2 1] on a dark background)', x0, 22, C.muted);
      for (let i = 0; i < N; i++) { g.fillStyle = AI.mix(C.bg2, C.blue, x[i] / 2); g.fillRect(X(i), 30, cw - 2, 30); label(g, x[i], X(i) + cw / 2 - 1, 50, C.fg, 12, 'center'); }
      label(g, 'MLP unit: its OWN weight for every pixel (learned when the pattern sat at position ' + P0 + ')', x0, 86, C.muted);
      for (let i = 0; i < N; i++) { const w = i >= P0 && i < P0 + 3 ? T[i - P0] : 0; g.fillStyle = w ? AI.mix(C.bg2, C.orange, w / 2) : C.bg2; g.fillRect(X(i), 94, cw - 2, 26); label(g, w, X(i) + cw / 2 - 1, 112, C.fg, 11, 'center'); }
      g.fillStyle = C.orange; label(g, 'response = ' + fc(p), x0 + 16 * cw - 4, 86, C.orange, 12, 'right');
      label(g, 'Conv unit: ONE shared kernel [1 2 1] slides over every position', x0, 146, C.muted);
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.strokeRect(X(pf) - 1, 26, 3 * cw, 38);
      g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(pf) + 1.5 * cw, 64); g.lineTo(X(pf + 1) + 0.5 * cw, 162); g.stroke(); g.setLineDash([]);
      const o = cv(x), mx = Math.max(...o);
      for (let j = 0; j <= PM; j++) { const v = o[j]; g.fillStyle = AI.mix(C.bg2, C.green, v / 6); g.fillRect(X(j + 1), 164, cw - 2, 28); label(g, v, X(j + 1) + cw / 2 - 1, 183, C.fg, 11, 'center'); if (j === p) { g.strokeStyle = C.green; g.lineWidth = 3; g.strokeRect(X(j + 1) - 1, 163, cw, 30); } }
      label(g, 'conv map (peak follows the pattern: equivariance); max of map = ' + mx, X(1), 208, C.green, 12);
      // chart
      const cx0 = 60, cy0 = 360, cw2 = 520, ch = 110;
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(cx0, cy0 - ch, cw2, ch);
      label(g, 'response as the pattern moves (position 0 … ' + PM + ')', cx0, cy0 - ch - 6, C.muted);
      const PX = (pp) => cx0 + 10 + (pp / PM) * (cw2 - 20), PY = (v) => cy0 - 8 - (v / 6.5) * (ch - 16);
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); for (let pp = 0; pp <= PM; pp++) { const y = PY(fc(pp)); pp ? g.lineTo(PX(pp), y) : g.moveTo(PX(pp), y); } g.stroke();
      g.strokeStyle = C.blue; g.setLineDash([6, 3]); g.beginPath(); g.moveTo(PX(0), PY(6)); g.lineTo(PX(PM), PY(6)); g.stroke(); g.setLineDash([]);
      label(g, 'MLP unit', PX(P0) + 8, PY(6) + 14, C.orange, 11); label(g, 'conv + max over the map (same everywhere)', PX(PM) - 4, PY(6) - 5, C.blue, 11, 'right');
      g.fillStyle = C.fg; g.beginPath(); g.arc(PX(p), PY(fc(p)), 5, 0, 7); g.fill();
      c.end();
      q(root, '#cap').textContent = `Pattern at ${p}: the MLP unit outputs ${fc(p)} (it only knows position ${P0}); the shared kernel gives a map whose peak (${mx}) sits at the pattern, wherever it is.`;
    }, { fps: 30 });
  };

  /* ================= 3.1 paramcount ================= */
  AI.labs.paramcount = function (root) {
    const opt = (arr, d) => arr.map((v) => `<option ${v === d ? 'selected' : ''}>${v}</option>`).join('');
    const { view } = AI.labShell(root, 'Parameter explosion: dense layer versus convolution', 'bars are on a log scale: each tick is 100× more',
      '<canvas></canvas><div class="caption">The convolution never depends on image size; the dense layer grows with (pixels × outputs).</div>',
      `<label>Image side (square) <select id="h">${opt([28, 32, 64, 128, 224, 512], 224)}</select></label>
       <label>Channels <select id="ci">${opt([1, 3], 3)}</select></label>
       <label>Dense layer <select id="mode"><option value="hid">N hidden units</option><option value="same">same output maps as the conv</option></select></label>
       <label>Hidden units N <select id="n">${opt([100, 1000, 4096], 1000)}</select></label>
       <label>Conv filters C_out <select id="co">${opt([8, 32, 64, 128], 64)}</select></label>
       <label>Kernel k <select id="k">${opt([1, 3, 5, 7], 3)}</select></label>
       <div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 250);
    let cur = [0, 0], tgt = [0, 0], last = '';
    const calc = () => {
      const H = +q(root, '#h').value, Ci = +q(root, '#ci').value, N = +q(root, '#n').value, Co = +q(root, '#co').value, k = +q(root, '#k').value, same = q(root, '#mode').value === 'same';
      const inD = H * H * Ci, outD = same ? Co * H * H : N, dense = inD * outD + outD, convP = Ci * k * k * Co + Co;
      const convM = Ci * k * k * Co * H * H, denseM = inD * outD;
      q(root, '#n').disabled = same;
      tgt = [Math.log10(dense), Math.log10(convP)];
      q(root, '#out').innerHTML = `<span class="m">input values</span> ${H}·${H}·${Ci} = <b>${com(inD)}</b>\n<span class="m">dense</span> ${com(inD)}·${com(outD)} + ${com(outD)}\n  = <b class="o">${com(dense)}</b> weights  (${fmtB(dense * 4)})\n<span class="m">conv</span> ${Ci}·${k}²·${Co} + ${Co} = <b class="b">${com(convP)}</b> weights  (${fmtB(convP * 4)})\n<span class="m">ratio dense / conv</span> ≈ <b>${(dense / convP).toExponential(2)}</b>\n<span class="m">MACs</span> dense ${com(denseM)} | conv ${com(convM)}`;
    };
    const fmtB = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(2) + ' MB' : b >= 1e3 ? (b / 1e3).toFixed(1) + ' kB' : b + ' B');
    ['#h', '#ci', '#n', '#co', '#k', '#mode'].forEach((s) => q(root, s).addEventListener('input', calc));
    calc(); cur = tgt.slice();
    AI.loop(root, (dt) => {
      let mv = false; for (let i = 0; i < 2; i++) { const d = tgt[i] - cur[i]; if (Math.abs(d) > 0.002) { cur[i] += d * Math.min(1, dt * 3); mv = true; } else cur[i] = tgt[i]; }
      const key = [cur.map((v) => v.toFixed(3)), c.w, document.documentElement.dataset.theme].join();
      if (key === last && !mv) return; last = key;
      const C = AI.col(), g = c.begin(), x0 = 40, wd = 570, lo = 1, hi = 14, X = (l) => x0 + ((l - lo) / (hi - lo)) * wd;
      for (let l = 2; l <= 14; l += 2) { g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(X(l), 30); g.lineTo(X(l), 190); g.stroke(); label(g, '10^' + l, X(l), 208, C.muted, 11, 'center'); }
      [['dense (fully connected)', cur[0], C.orange, 45], ['convolution (shared)', cur[1], C.blue, 120]].forEach(([n, v, col, y]) => {
        label(g, n, x0, y - 8, C.fg, 12); g.fillStyle = col; g.fillRect(x0, y, Math.max(2, X(v) - x0), 34);
        label(g, '10^' + v.toFixed(1) + ' = ' + (Math.pow(10, v) >= 1e6 ? (Math.pow(10, v) / 1e6).toFixed(1) + ' M' : com(Math.pow(10, v))), Math.min(X(v) + 6, 440), y + 22, C.fg, 12);
      });
      c.end();
    }, { fps: 30 });
  };

  /* ================= 3.2 mcconv ================= */
  AI.labs.mcconv = function (root) {
    const { view } = AI.labShell(root, 'Multi-channel convolution', 'a C_in × k × k filter slides over the input volume; every position sums across ALL channels into one output number',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `${slider('H', 'Input size H = W', 4, 8, 5)}${slider('ci', 'Input channels C_in', 1, 4, 3)}${slider('co', 'Filters C_out', 1, 4, 2)}
       <div class="row"><label>k <select id="k"><option>1</option><option selected>3</option><option>5</option></select></label>
       <label>stride <select id="s"><option>1</option><option>2</option></select></label>
       <label>padding <select id="p"><option>0</option><option>1</option><option>2</option></select></label></div>
       ${playHTML(1, 'positions / s', 0.2, 4, 0.1)}
       <div class="caption" style="margin:0">Click an input cell or a filter weight to change it.</div>
       <div class="readout" id="out"></div>`);
    const c = dcanvas(view, 680, 440), tick = AI.ticker(), ovr = new Map();
    let H, Ci, Co, K, S, P, O, outs = [], pos = 0, wx = 0, wy = 0, last = '', dirty = true, hold = 0, geo = null;
    const inV = (ch, y, x) => (y < 0 || x < 0 || y >= H || x >= H ? 0 : ovr.has(`i${ch},${y},${x}`) ? ovr.get(`i${ch},${y},${x}`) : hash(ch, y, x, 1) % 4);
    const wV = (o, ch, u, v) => { const k = `w${o},${ch},${u},${v}`; return ovr.has(k) ? ovr.get(k) : (hash(o, ch, u * 5 + v, 2) % 3) - 1; };
    const bV = (o) => { const k = `b${o}`; return ovr.has(k) ? ovr.get(k) : (hash(o, 9, 9, 3) % 3) - 1; };
    const at = (o, oy, ox) => { const parts = []; let tot = 0; for (let ch = 0; ch < Ci; ch++) { let s = 0; for (let u = 0; u < K; u++) for (let v = 0; v < K; v++) s += wV(o, ch, u, v) * inV(ch, oy * S - P + u, ox * S - P + v); parts.push(s); tot += s; } return { parts, tot, out: tot + bV(o) }; };
    const read = () => { H = +q(root, '#H').value; Ci = +q(root, '#ci').value; Co = +q(root, '#co').value; K = +q(root, '#k').value; S = +q(root, '#s').value; P = +q(root, '#p').value; O = Math.floor((H + 2 * P - K) / S) + 1; ['H', 'ci', 'co'].forEach((n) => (q(root, '#' + n + 'v').textContent = q(root, '#' + n).value)); };
    const rebuild = () => { read(); outs = []; if (O >= 1) for (let o = 0; o < Co; o++) { const a = []; for (let y = 0; y < O; y++) for (let x = 0; x < O; x++) a.push(at(o, y, x)); outs.push(a); } dirty = true; };
    ['#H', '#ci', '#co', '#k', '#s', '#p'].forEach((s) => q(root, s).addEventListener('input', () => { rebuild(); pos = 0; }));
    const total = () => Math.max(1, Co * O * O);
    const st = player(root, 'positions / s', { onStep: () => { pos = (Math.floor(pos) + 1) % (total() + 1); dirty = true; }, onRestart: () => { pos = 0; dirty = true; } });
    c.cv = AI.$('canvas', view);
    c.cv.addEventListener('pointerdown', (e) => {
      if (!geo) return; const [mx, my] = c.pt(e), cur = Math.min(total() - 1, Math.floor(pos)), o = Math.floor(cur / (O * O));
      for (let ch = 0; ch < Ci; ch++) {
        const gx = geo.x0 + ch * (geo.gw + geo.gap);
        if (mx >= gx && mx < gx + geo.gw && my >= geo.iy && my < geo.iy + geo.gh) { const x = Math.floor((mx - gx) / geo.cs) - P, y = Math.floor((my - geo.iy) / geo.cs) - P; if (x >= 0 && y >= 0 && x < H && y < H) { ovr.set(`i${ch},${y},${x}`, (inV(ch, y, x) + 1) % 4); rebuild(); } }
        if (mx >= gx && mx < gx + K * geo.cs && my >= geo.fy && my < geo.fy + K * geo.cs) { const v = Math.floor((mx - gx) / geo.cs), u = Math.floor((my - geo.fy) / geo.cs), w = wV(o, ch, u, v); ovr.set(`w${o},${ch},${u},${v}`, w >= 2 ? -1 : w + 1); rebuild(); }
      }
    });
    rebuild();
    AI.loop(root, (dt) => {
      if (O < 1) { if (dirty) { dirty = false; const g = c.begin(); label(g, 'kernel larger than the padded input: no output (O < 1)', 20, 60, AI.col().red, 14); c.end(); q(root, '#out').textContent = 'O = ⌊(H+2p−k)/s⌋+1 = ' + O; } return; }
      if (st.playing) { const n = tick(dt, st.speed()); if (n) { pos += n; hold = 0; } if (pos >= total()) { hold += dt; if (hold > 2.2) { pos = 0; hold = 0; } } }
      const tot = total(), t = Math.min(tot - 1, Math.floor(pos)), done = pos >= tot, o = Math.floor(t / (O * O)), idx = t % (O * O), oy = Math.floor(idx / O), ox = idx % O;
      const prevx = wx, prevy = wy; wx += (ox - wx) * Math.min(1, dt * 7); wy += (oy - wy) * Math.min(1, dt * 7);
      const mv = Math.abs(wx - ox) > 0.01 || Math.abs(wy - oy) > 0.01; if (!mv) { wx = ox; wy = oy; }
      const key = [t, done, c.w, document.documentElement.dataset.theme, dirty].join();
      if (key === last && !mv && !dirty) return; last = key; dirty = false;
      const C = AI.col(), g = c.begin(), Hp = H + 2 * P, gap = 16, x0 = 14;
      const cs = Math.min(26, (386 - (Ci - 1) * gap) / (Ci * Hp), 300 / (Hp + K));
      const gw = Hp * cs, gh = gw, iy = 34, fy = iy + gh + 46; geo = { x0, gw, gh, gap, cs, iy, fy };
      label(g, `input volume ${Ci}×${H}×${H}` + (P ? `  (+ zero padding ${P})` : ''), x0, 20, C.muted, 12);
      const nums = cs >= 15;
      for (let ch = 0; ch < Ci; ch++) {
        const gx = x0 + ch * (gw + gap);
        for (let y = 0; y < Hp; y++) for (let x = 0; x < Hp; x++) {
          const v = inV(ch, y - P, x - P), pad = y < P || x < P || y >= H + P || x >= H + P;
          g.fillStyle = AI.mix(C.bg2, C.blue, v / 3); g.fillRect(gx + x * cs, iy + y * cs, cs - 1, cs - 1);
          if (pad) { g.strokeStyle = C.muted; g.setLineDash([2, 2]); g.strokeRect(gx + x * cs + 0.5, iy + y * cs + 0.5, cs - 2, cs - 2); g.setLineDash([]); }
          if (nums) label(g, v, gx + x * cs + cs / 2 - 0.5, iy + y * cs + cs / 2 + 4, C.fg, Math.min(12, cs * 0.5), 'center');
        }
        label(g, 'channel ' + ch, gx, iy + gh + 14, C.muted, 11);
        const wxp = gx + (wx * S) * cs, wyp = iy + (wy * S) * cs;
        g.fillStyle = C.blue; g.globalAlpha = 0.2; g.fillRect(wxp, wyp, K * cs, K * cs); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 2.5; g.strokeRect(wxp, wyp, K * cs, K * cs);
        // filter slice
        const cur = at(o, oy, ox);
        for (let u = 0; u < K; u++) for (let v = 0; v < K; v++) { const w = wV(o, ch, u, v); g.fillStyle = divcol(w, 1, C); g.fillRect(gx + v * cs, fy + u * cs, cs - 1, cs - 1); if (nums) label(g, sgn(w), gx + v * cs + cs / 2 - 0.5, fy + u * cs + cs / 2 + 4, C.fg, Math.min(12, cs * 0.5), 'center'); }
        g.strokeStyle = C.orange; g.lineWidth = 1.5; g.strokeRect(gx - 1, fy - 1, K * cs + 1, K * cs + 1);
        label(g, `Σ = ${sgn(cur.parts[ch])}`, gx, fy + K * cs + 16, C.green, 12);
        if (ch < Ci - 1) label(g, '+', gx + gw + gap / 2 - 3, fy + K * cs + 16, C.fg, 14, 'center');
      }
      label(g, `filter ${o}: ${Ci}×${K}×${K} weights (one ${K}×${K} slice per channel)`, x0, fy - 8, C.orange, 12);
      const cur = at(o, oy, ox);
      label(g, `Σ over channels = ${sgn(cur.tot)},  + bias ${sgn(bV(o))}  →  output ${sgn(cur.out)}`, x0, fy + K * cs + 40, C.fg, 13);
      // outputs
      const ax = 430, aw = 236, pw = aw / 2;
      label(g, `output volume ${Co}×${O}×${O}`, ax, 20, C.muted, 12);
      const os = Math.min(22, (pw - 14) / O), shn = os >= 15;
      let mxa = 1; outs.forEach((a) => a.forEach((r) => (mxa = Math.max(mxa, Math.abs(r.out)))));
      for (let m = 0; m < Co; m++) {
        const mx0 = ax + (m % 2) * pw, my0 = 34 + Math.floor(m / 2) * (O * os + 52);
        label(g, 'map ' + m, mx0, my0 + O * os + 14, m === o ? C.orange : C.muted, 11);
        for (let k = 0; k < O * O; k++) {
          const X = mx0 + (k % O) * os, Y = my0 + Math.floor(k / O) * os, shown = m < o || (m === o && k < t) || done;
          if (shown) { const v = outs[m][k].out; g.fillStyle = divcol(v, mxa, C); g.fillRect(X, Y, os - 1, os - 1); if (shn) label(g, sgn(v), X + os / 2 - 0.5, Y + os / 2 + 4, C.fg, Math.min(11, os * 0.5), 'center'); }
          else { g.strokeStyle = C.border; g.strokeRect(X + 0.5, Y + 0.5, os - 2, os - 2); }
        }
        if (!done && m === o) { g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(mx0 + ox * os - 1, my0 + oy * os - 1, os + 1, os + 1); }
      }
      c.end();
      const pc = Ci * K * K * Co + Co, mac = Ci * K * K * Co * O * O;
      q(root, '#out').innerHTML = `<span class="m">output size</span> ⌊(${H}+2·${P}−${K})/${S}⌋+1 = <b>${O}</b>\n<span class="m">volume</span> C_out×O×O = <b>${Co}×${O}×${O}</b>\n<span class="m">params</span> C_in·k²·C_out + C_out\n  = ${Ci}·${K * K}·${Co} + ${Co} = <b class="o">${pc}</b>\n<span class="m">MACs</span> ${Ci}·${K * K}·${Co}·${O * O} = <b class="b">${mac}</b>\n<span class="m">now: filter ${o}, output (${oy},${ox})</span>\n${cur.parts.map((p2, i) => `ch${i}: ${sgn(p2)}`).join('  ')}\n= ${sgn(cur.tot)} + ${sgn(bV(o))} = <b class="g">${sgn(cur.out)}</b>`;
      q(root, '#cap').textContent = `Step ${Math.min(t + 1, tot)} of ${tot}: the filter's ${Ci} slices multiply the window in every channel; the ${Ci} partial sums and the bias make ONE number of feature map ${o}.`;
    }, { fps: 30 });
  };

  /* ================= 3.3 outsize ================= */
  AI.labs.outsize = function (root) {
    const { view } = AI.labShell(root, 'Output size, stride, padding, dilation', 'see which input cells each output reads; the formula updates live',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `<div class="row" id="dimrow"></div>${slider('n', 'Input size n', 5, 14, 9)}${slider('k', 'Kernel k', 1, 7, 3)}${slider('s', 'Stride s', 1, 4, 1)}${slider('p', 'Padding p', 0, 3, 0)}${slider('d', 'Dilation d', 1, 3, 1)}
       <div class="row"><button class="btn" id="valid">valid (p=0)</button><button class="btn" id="same">same</button></div>
       ${playHTML(0.8, 'outputs / s', 0.2, 3, 0.1)}<div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 340), tick = AI.ticker();
    let n, k, s, p, d, O, ke, mode = '1d', pos = 0, last = '', pin = -1, geo = {};
    const read = () => { n = +q(root, '#n').value; k = +q(root, '#k').value; s = +q(root, '#s').value; p = +q(root, '#p').value; d = +q(root, '#d').value; ['n', 'k', 's', 'p', 'd'].forEach((x) => (q(root, '#' + x + 'v').textContent = q(root, '#' + x).value)); ke = d * (k - 1) + 1; O = Math.floor((n + 2 * p - ke) / s) + 1; };
    chipRow(q(root, '#dimrow'), [['1d', '1-D row'], ['2d', '2-D grid']], (m) => { mode = m; last = ''; }, mode);
    ['n', 'k', 's', 'p', 'd'].forEach((x) => q(root, '#' + x).addEventListener('input', () => { read(); pos = 0; pin = -1; last = ''; }));
    q(root, '#valid').onclick = () => { q(root, '#p').value = 0; read(); last = ''; };
    q(root, '#same').onclick = () => { read(); q(root, '#s').value = 1; q(root, '#p').value = Math.min(3, Math.floor(ke / 2)); read(); last = ''; };
    const st = player(root, 'outputs / s', { onStep: () => { pos = (Math.floor(pos) + 1) % Math.max(1, O); last = ''; }, onRestart: () => { pos = 0; pin = -1; last = ''; } });
    AI.$('canvas', view).addEventListener('pointerdown', (e) => {
      const [mx, my] = c.pt(e); let hit = -1;
      if (mode === '1d') for (let o = 0; o < O; o++) { const x = geo.x0 + (o * s + (ke - 1) / 2) * geo.cs; if (Math.abs(mx - x - geo.cs / 2) < geo.cs / 2 + 2 && my > 190 && my < 260) hit = o; }
      else for (let o = 0; o < O * O; o++) { const X = geo.ox + (o % O) * geo.os, Y = geo.oy + Math.floor(o / O) * geo.os; if (mx >= X && mx < X + geo.os && my >= Y && my < Y + geo.os) hit = o; }
      if (hit >= 0) { pin = hit; pos = hit; st.pause(); last = ''; }
    });
    read();
    AI.loop(root, (dt) => {
      read();
      const tot = mode === '1d' ? O : O * O;
      if (st.playing && tot > 0) { pos += tick(dt, st.speed()) * 1; if (pos >= tot) pos = 0; }
      const cur = Math.min(Math.max(0, tot - 1), Math.floor(pos));
      const key = [cur, mode, n, k, s, p, d, c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin();
      const f = `O = ⌊(n + 2p − d(k−1) − 1)/s⌋ + 1 = ⌊(${n} + ${2 * p} − ${d * (k - 1)} − 1)/${s}⌋ + 1 = ${O < 1 ? 'no output (kernel too big)' : O}`;
      if (O < 1) { label(g, f, 20, 40, C.red, 13); c.end(); q(root, '#out').textContent = f; return; }
      const lft = Math.max(0, (n + 2 * p - ke) % s);
      if (mode === '1d') {
        const L = n + 2 * p, cs = Math.min(38, 600 / L), x0 = (640 - L * cs) / 2; geo = { x0, cs };
        const taps = []; for (let i = 0; i < k; i++) taps.push(cur * s + i * d);
        const covered = new Array(L).fill(false); for (let o = 0; o < O; o++) for (let i = 0; i < k; i++) covered[o * s + i * d] = true;
        for (let o = 0; o < O; o++) for (let t = 0; t < ke; t++) if (t >= 0) { /* span coverage for "ignored" test is by span, not taps */ }
        const spanCov = new Array(L).fill(false); for (let o = 0; o < O; o++) for (let t = 0; t < ke; t++) spanCov[o * s + t] = true;
        label(g, 'input (dashed = zero padding)', 20, 40, C.muted, 12);
        for (let i = 0; i < L; i++) {
          const pad = i < p || i >= n + p, X = x0 + i * cs, tap = taps.includes(i), ign = !spanCov[i];
          g.fillStyle = tap ? AI.mix(C.bg2, C.blue, 0.8) : pad ? C.bg : C.bg2; g.fillRect(X, 80, cs - 2, cs); g.strokeStyle = ign ? C.red : pad ? C.muted : C.border; g.lineWidth = ign ? 2 : 1; if (pad) g.setLineDash([3, 3]); g.strokeRect(X + 0.5, 80.5, cs - 3, cs - 1); g.setLineDash([]);
          label(g, pad ? '0' : i - p, X + cs / 2 - 1, 80 + cs / 2 + 4, C.fg, 11, 'center');
          if (tap) label(g, 'w' + (taps.indexOf(i) + 1), X + cs / 2 - 1, 74, C.blue, 11, 'center');
        }
        // bracket
        const bx0 = x0 + cur * s * cs, bx1 = bx0 + ke * cs - 2; g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(bx0, 130 + cs * 0.4); g.lineTo(bx0, 140 + cs * 0.4); g.lineTo(bx1, 140 + cs * 0.4); g.lineTo(bx1, 130 + cs * 0.4); g.stroke();
        label(g, `blue bracket: effective kernel d(k−1)+1 = ${ke} cells, only ${k} weights`, 20, 270, C.blue, 12);
        label(g, 'output (each sits over its window)', 620, 196, C.muted, 12, 'right');
        for (let o = 0; o < O; o++) {
          const X = x0 + (o * s + (ke - 1) / 2) * cs, hot = o === cur;
          g.fillStyle = hot ? C.orange : AI.mix(C.bg2, C.green, 0.5); g.fillRect(X, 204, cs - 2, cs); label(g, o, X + cs / 2 - 1, 204 + cs / 2 + 4, hot ? '#000' : C.fg, 11, 'center');
          if (hot) { g.strokeStyle = C.orange; g.lineWidth = 1; taps.forEach((t) => { g.beginPath(); g.moveTo(X + cs / 2 - 1, 204); g.lineTo(x0 + t * cs + cs / 2 - 1, 80 + cs); g.stroke(); }); }
        }
        if (lft && O > 0) label(g, `${lft} trailing cell${lft > 1 ? 's' : ''} never seen (red outline): (n+2p−k_eff) is not a multiple of s`, 20, 292, C.red, 12);
      } else {
        const Lp = n + 2 * p, cs = Math.min(24, 250 / Lp), os = Math.min(26, 240 / O), ix = 24, iy = 50, ox = 350, oy = 50; geo = { ox, oy, os };
        const cy = Math.floor(cur / O), cx = cur % O;
        label(g, 'input (dashed = zero padding)', ix, 36, C.muted, 12); label(g, `output ${O}×${O}`, ox, 36, C.muted, 12);
        const spanR = new Array(Lp).fill(false); for (let o = 0; o < O; o++) for (let t = 0; t < ke; t++) spanR[o * s + t] = true;
        for (let y = 0; y < Lp; y++) for (let x = 0; x < Lp; x++) {
          const pad = y < p || x < p || y >= n + p || x >= n + p, X = ix + x * cs, Y = iy + y * cs;
          let tap = false; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) if (cy * s + u * d === y && cx * s + v * d === x) tap = true;
          g.fillStyle = tap ? AI.mix(C.bg2, C.blue, 0.8) : pad ? C.bg : C.bg2; g.fillRect(X, Y, cs - 1, cs - 1);
          if (pad) { g.strokeStyle = C.muted; g.setLineDash([2, 2]); g.strokeRect(X + 0.5, Y + 0.5, cs - 2, cs - 2); g.setLineDash([]); }
          if (!spanR[y] || !spanR[x]) { g.strokeStyle = C.red; g.lineWidth = 1.5; g.strokeRect(X + 0.5, Y + 0.5, cs - 2, cs - 2); }
        }
        g.strokeStyle = C.blue; g.lineWidth = 2; g.strokeRect(ix + cx * s * cs, iy + cy * s * cs, ke * cs - 1, ke * cs - 1);
        for (let o = 0; o < O * O; o++) { const X = ox + (o % O) * os, Y = oy + Math.floor(o / O) * os; g.fillStyle = o === cur ? C.orange : AI.mix(C.bg2, C.green, 0.5); g.fillRect(X, Y, os - 1, os - 1); }
        if (lft) label(g, 'red outline = rows/columns never covered by any window', ix, 320, C.red, 12);
      }
      label(g, f, 20, 330, C.fg, 12);
      c.end();
      q(root, '#out').innerHTML = `<span class="m">k_eff</span> d(k−1)+1 = <b>${ke}</b>\n<span class="m">padded size</span> n+2p = ${n + 2 * p}\n<span class="m">O</span> = <b class="o">${O}</b>   ${mode === '2d' ? '(' + O + '×' + O + ' = ' + O * O + ' outputs)' : ''}\n<span class="m">params</span> ${mode === '2d' ? k * k : k} weights regardless of d`;
      q(root, '#cap').textContent = `Output ${cur} reads the ${mode === '2d' ? k * k : k} highlighted input cells; the window then hops ${s} cell${s > 1 ? 's' : ''}. Click an output to pin it.`;
    }, { fps: 30 });
  };

  /* ================= 3.4 pooling ================= */
  AI.labs.pooling = function (root) {
    const base = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 9, 0, 0, 0, 0], [0, 0, 8, 9, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6, 7, 0], [0, 0, 0, 0, 0, 8, 9, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]].map((r) => r.slice());
    const { view } = AI.labShell(root, 'Pooling: max versus average', 'the window summarises each neighbourhood; the argmax is outlined; shift the input to test tolerance',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `<div class="row" id="typ"></div><div class="row"><label>window k <select id="k"><option>2</option><option>3</option></select></label><label>stride <select id="s"><option>1</option><option selected>2</option><option>3</option></select></label></div>
       ${slider('dx', 'Shift input right by', -2, 2, 0)}${slider('dy', 'Shift input down by', -2, 2, 0)}
       <div class="row"><button class="btn" id="rand">Random map</button><button class="btn" id="reset">Reset map</button></div>
       ${playHTML(1, 'windows / s', 0.2, 4, 0.1)}<div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 330), tick = AI.ticker();
    let type = 'max', pos = 0, last = '', dx = 0, dy = 0, K = 2, S = 2, M = base.map((r) => r.slice());
    const shifted = (m) => m.map((r, y) => r.map((_, x) => { const yy = y - dy, xx = x - dx; return yy < 0 || xx < 0 || yy > 7 || xx > 7 ? 0 : m[yy][xx]; }));
    const pool = (m) => { const kk = type === 'global' ? 8 : K, ss = type === 'global' ? 8 : S, Oo = Math.floor((8 - kk) / ss) + 1, out = [], arg = []; for (let y = 0; y < Oo; y++) for (let x = 0; x < Oo; x++) { let b = -1, bi = [0, 0], sm = 0; for (let u = 0; u < kk; u++) for (let v = 0; v < kk; v++) { const val = m[y * ss + u][x * ss + v]; sm += val; if (val > b) { b = val; bi = [y * ss + u, x * ss + v]; } } out.push(type === 'avg' || type === 'global' ? sm / (kk * kk) : b); arg.push(bi); } return { out, arg, Oo, kk, ss }; };
    const cosv = (a, b) => { let d = 0, na = 0, nb = 0; a.forEach((v, i) => { d += v * b[i]; na += v * v; nb += b[i] * b[i]; }); return na && nb ? d / Math.sqrt(na * nb) : 0; };
    chipRow(q(root, '#typ'), [['max', 'max'], ['avg', 'average'], ['global', 'global average']], (t) => { type = t; pos = 0; last = ''; }, type);
    ['#k', '#s', '#dx', '#dy'].forEach((s2) => q(root, s2).addEventListener('input', () => { K = +q(root, '#k').value; S = +q(root, '#s').value; dx = +q(root, '#dx').value; dy = +q(root, '#dy').value; q(root, '#dxv').textContent = dx; q(root, '#dyv').textContent = dy; pos = 0; last = ''; }));
    q(root, '#dxv').textContent = 0; q(root, '#dyv').textContent = 0;
    q(root, '#rand').onclick = () => { M = M.map((r) => r.map(() => (Math.random() < 0.35 ? Math.ceil(Math.random() * 9) : 0))); last = ''; };
    q(root, '#reset').onclick = () => { M = base.map((r) => r.slice()); last = ''; };
    const st = player(root, 'windows / s', { onStep: () => { pos = Math.floor(pos) + 1; last = ''; }, onRestart: () => { pos = 0; last = ''; } });
    const cs = 30, ix = 20, iy = 40, ox = 330;
    AI.$('canvas', view).addEventListener('pointerdown', (e) => { const [mx, my] = c.pt(e), x = Math.floor((mx - ix) / cs) - dx, y = Math.floor((my - iy) / cs) - dy; if (x >= 0 && y >= 0 && x < 8 && y < 8 && mx >= ix && my >= iy) { M[y][x] = (M[y][x] + 3) % 10; pos = 0; last = ''; } });
    AI.loop(root, (dt) => {
      const sh = shifted(M), P = pool(sh), tot = P.Oo * P.Oo;
      if (st.playing) { pos += tick(dt, st.speed()); if (pos > tot + 2) pos = 0; }
      const t = Math.min(tot, Math.floor(pos)), cur = Math.min(tot - 1, t);
      const key = [t, type, K, S, dx, dy, JSON.stringify(M), c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin(), os = Math.min(40, 240 / P.Oo);
      label(g, `input 8×8${dx || dy ? ` (shifted by ${dx}, ${dy})` : ''}`, ix, 30, C.muted, 12); label(g, `${type} pool → ${P.Oo}×${P.Oo}`, ox, 30, C.muted, 12);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) { g.fillStyle = AI.mix(C.bg2, C.blue, sh[y][x] / 9); g.fillRect(ix + x * cs, iy + y * cs, cs - 1, cs - 1); label(g, sh[y][x], ix + x * cs + cs / 2 - 0.5, iy + y * cs + cs / 2 + 4, C.fg, 12, 'center'); }
      const oy0 = Math.floor(cur / P.Oo), ox0 = cur % P.Oo;
      g.fillStyle = C.blue; g.globalAlpha = 0.18; g.fillRect(ix + ox0 * P.ss * cs, iy + oy0 * P.ss * cs, P.kk * cs, P.kk * cs); g.globalAlpha = 1; g.strokeStyle = C.blue; g.lineWidth = 2.5; g.strokeRect(ix + ox0 * P.ss * cs, iy + oy0 * P.ss * cs, P.kk * cs, P.kk * cs);
      if (type === 'max') { const a = P.arg[cur]; g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(ix + a[1] * cs, iy + a[0] * cs, cs - 1, cs - 1); }
      for (let k = 0; k < tot; k++) { const X = ox + (k % P.Oo) * os, Y = iy + Math.floor(k / P.Oo) * os; if (k < t) { g.fillStyle = AI.mix(C.bg2, C.green, P.out[k] / 9); g.fillRect(X, Y, os - 1, os - 1); label(g, fx(P.out[k], 1), X + os / 2 - 0.5, Y + os / 2 + 4, C.fg, 12, 'center'); } else { g.strokeStyle = C.border; g.strokeRect(X + 0.5, Y + 0.5, os - 2, os - 2); } }
      g.strokeStyle = C.orange; g.lineWidth = 3; g.strokeRect(ox + ox0 * os - 1, iy + oy0 * os - 1, os + 1, os + 1);
      const vals = []; for (let u = 0; u < P.kk; u++) for (let v = 0; v < P.kk; v++) vals.push(sh[oy0 * P.ss + u][ox0 * P.ss + v]);
      const txt = type === 'max' ? `max(${vals.join(', ')}) = ${Math.max(...vals)}` : `(${vals.join(' + ')}) / ${vals.length} = ${fx(vals.reduce((a, b) => a + b, 0) / vals.length, 2)}`;
      label(g, txt.length > 56 ? txt.slice(0, 56) + '…' : txt, ix, 306, C.fg, 13);
      c.end();
      const P0 = pool(M), raw = cosv(M.flat(), sh.flat()), pl = cosv(P0.out, P.out), same = P0.out.filter((v, i) => Math.abs(v - P.out[i]) < 1e-9).length;
      q(root, '#out').innerHTML = `<span class="m">output size</span> ⌊(8−${P.kk})/${P.ss}⌋+1 = <b>${P.Oo}</b>\n<span class="m">params</span> 0\n<span class="m">shift test (cosine similarity)</span>\nraw pixels   <b class="o">${fx(raw, 3)}</b>\npooled maps  <b class="g">${fx(pl, 3)}</b>\n${same}/${P.out.length} pooled values unchanged`;
      q(root, '#cap').textContent = type === 'global' ? 'Global average pooling: the whole map becomes one number per channel.' : `Window ${cur + 1} of ${tot}: ${type === 'max' ? 'keep only the largest value (orange) — the feature fired somewhere in here.' : 'average the values in the window.'}`;
    }, { fps: 30 });
  };

  /* ================= 3.5 rfield ================= */
  AI.labs.rfield = function (root) {
    const PRE = { '3 convs 3×3': '3 1\n3 1\n3 1', 'conv, pool, conv, pool, conv': '3 1\n2 2\n3 1\n2 2\n3 1', 'LeNet-style': '5 1\n2 2\n5 1\n2 2', 'dilated 1,2,4': '3 1 1\n3 1 2\n3 1 4', 'stride-2 stack': '3 2\n3 2\n3 2' };
    const { view } = AI.labShell(root, 'Receptive-field explorer', 'click any neuron: the blue cone is every cell it depends on, traced back to the input',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `<div>Presets<div class="chips" id="pre" style="margin-top:4px"></div></div>
       <label>Layers, one per line: <b>k s [d]</b> (kernel, stride, optional dilation)<textarea id="spec" rows="5" spellcheck="false" style="width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px;font-family:${mono}">${PRE['conv, pool, conv, pool, conv']}</textarea></label><div class="err" id="err"></div>
       ${playHTML(0.8, 'layers / s', 0.2, 3, 0.1)}<div class="readout" id="out"></div>`);
    const c = dcanvas(view, 680, 470), tick = AI.ticker(), N0 = 41;
    let L = [], sel = null, prog = 0, last = '', cone = [], eff = [], g0 = null;
    const build = () => {
      const err = []; const raw = q(root, '#spec').value.split('\n').map((l) => l.replace(/#.*/, '').trim()).filter(Boolean).slice(0, 7);
      L = []; let n = N0, r = 1, j = 1, start = 0;
      raw.forEach((line, i) => { const t = line.split(/[\s,]+/).map(Number); if (t.some((v) => !(v >= 1)) || t.length < 2) { err.push('line ' + (i + 1) + ': need  k s [d]'); return; } const [k, s, d = 1] = t, ke = d * (k - 1) + 1, p = ke % 2 ? Math.floor(ke / 2) : 0, no = Math.floor((n + 2 * p - ke) / s) + 1; if (no < 1) { err.push('line ' + (i + 1) + ': too small'); return; } const rn = r + (ke - 1) * j, jn = j * s, sn = start + ((ke - 1) / 2 - p) * j; L.push({ k, s, d, ke, p, n: no, nin: n, r: rn, j: jn, start: sn, rin: r, jin: j }); n = no; r = rn; j = jn; start = sn; });
      q(root, '#err').textContent = err.join('\n');
      sel = { l: L.length, i: Math.floor(((L.length ? L[L.length - 1].n : N0) - 1) / 2) }; reset();
    };
    const lay = (l) => (l === 0 ? { n: N0, j: 1, start: 0 } : L[l - 1]);
    const reset = () => {
      if (!sel) return; cone = []; let cur = new Set([sel.i]); cone[sel.l] = cur;
      for (let l = sel.l; l >= 1; l--) { const nx = new Set(), Ly = L[l - 1]; cur.forEach((i) => { for (let t = 0; t < Ly.k; t++) { const idx = i * Ly.s - Ly.p + t * Ly.d; if (idx >= 0 && idx < Ly.nin) nx.add(idx); } }); cone[l - 1] = nx; cur = nx; }
      let w = new Map([[sel.i, 1]]);
      for (let l = sel.l; l >= 1; l--) { const Ly = L[l - 1], nw = new Map(); w.forEach((v, i) => { for (let t = 0; t < Ly.k; t++) { const idx = i * Ly.s - Ly.p + t * Ly.d; if (idx >= 0 && idx < Ly.nin) nw.set(idx, (nw.get(idx) || 0) + v / Ly.k); } }); w = nw; }
      eff = new Array(N0).fill(0); if (sel.l === 0) eff[sel.i] = 1; else if (w) w.forEach((v, i) => { if (i < N0) eff[i] = v; });
      prog = 0; last = '';
    };
    chipRow(q(root, '#pre'), Object.keys(PRE).map((k) => [k, k]), (k) => { q(root, '#spec').value = PRE[k]; build(); }, '');
    q(root, '#spec').addEventListener('input', build);
    const st = player(root, 'layers / s', { onStep: () => { prog = Math.floor(prog) + 1; last = ''; }, onRestart: () => { prog = 0; last = ''; } });
    const rowY = (l) => 40 + (L.length - l) * (290 / Math.max(1, L.length)), U = () => 640 / N0;
    AI.$('canvas', view).addEventListener('pointerdown', (e) => {
      const [mx, my] = c.pt(e); let best = null, bd = 1e9;
      for (let l = 0; l <= L.length; l++) { const Ly = lay(l); for (let i = 0; i < Ly.n; i++) { const x = 20 + (Ly.start + i * Ly.j + 0.5) * U(), y = rowY(l) + 7, dd = Math.hypot(mx - x, (my - y) * 1.5); if (dd < bd && dd < 22) { bd = dd; best = { l, i }; } } }
      if (best) { sel = best; reset(); }
    });
    build();
    AI.loop(root, (dt) => {
      if (st.playing) { prog += tick(dt, st.speed()); if (prog > (sel ? sel.l : 0) + 4) prog = 0; }
      const rev = Math.floor(prog), key = [rev, sel && sel.l, sel && sel.i, q(root, '#spec').value, c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin(), u = U();
      label(g, 'positions along an image row (the 2-D case is identical in each axis); input at the bottom', 20, 16, C.muted, 12);
      for (let l = 0; l <= L.length; l++) {
        const Ly = lay(l), y = rowY(l);
        label(g, l ? `layer ${l}: k=${Ly.k} s=${Ly.s}${Ly.d > 1 ? ' d=' + Ly.d : ''}` : 'input', 20, y - 4, C.muted, 10.5);
        for (let i = 0; i < Ly.n; i++) {
          const x = 20 + (Ly.start + i * Ly.j) * u, w = Math.max(3, u * Ly.j * 0.82), isSel = sel && sel.l === l && sel.i === i;
          const inCone = sel && l < sel.l && cone[l] && cone[l].has(i) && sel.l - l <= rev;
          g.fillStyle = isSel ? C.orange : inCone ? C.blue : C.bg2; g.fillRect(x + (u - w) / 2, y, w, 14);
          g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(x + (u - w) / 2 + 0.5, y + 0.5, w - 1, 13);
        }
      }
      // direct links
      if (sel && sel.l >= 1) for (let l = sel.l; l >= 1 && sel.l - l < rev + 1; l--) { const Ly = L[l - 1]; if (!cone[l]) continue; cone[l].forEach((i) => { if (l < sel.l && sel.l - l > rev) return; const x1 = 20 + (Ly.start + i * Ly.j + 0.5) * u + 0; for (let t = 0; t < Ly.k; t++) { const idx = i * Ly.s - Ly.p + t * Ly.d; if (idx < 0 || idx >= Ly.nin || sel.l - l + 1 > rev) continue; const pr = lay(l - 1); g.strokeStyle = C.blue; g.globalAlpha = 0.35; g.beginPath(); g.moveTo(x1, rowY(l) + 14); g.lineTo(20 + (pr.start + idx * pr.j + 0.5) * u, rowY(l - 1)); g.stroke(); g.globalAlpha = 1; } }); }
      // effective bars
      const by = 410, bh = 40, mxw = Math.max(1e-9, ...eff);
      label(g, 'effective contribution of each input pixel (all kernel weights = 1/k)', 20, by - bh - 8, C.muted, 11);
      for (let i = 0; i < N0; i++) { const hh = (eff[i] / mxw) * bh, x = 20 + i * u; g.fillStyle = C.blue; g.globalAlpha = sel && sel.l - 0 <= rev + 0 ? 1 : 0.25; g.fillRect(x + 1, by - hh, u - 2, hh); }
      g.globalAlpha = 1; g.strokeStyle = C.border; g.beginPath(); g.moveTo(20, by + 0.5); g.lineTo(660, by + 0.5); g.stroke();
      c.end();
      // readout
      let tx = `<span class="m">layer   k  s  j    r = r_in + (k_eff−1)·j_in   cells</span>\ninput   -  -  1    1\n`;
      L.forEach((Ly, i) => { tx += `L${i + 1}      ${Ly.k}  ${Ly.s}  ${String(Ly.j).padEnd(4)} ${Ly.rin} + ${Ly.ke - 1}·${Ly.jin} = <b class="${sel && sel.l === i + 1 ? 'o' : 'b'}">${Ly.r}</b>${' '.repeat(Math.max(1, 12 - String(Ly.r).length))}${Ly.n}\n`; });
      const tot = eff.reduce((a, b) => a + b, 0) || 1; let width = 0; const srt = eff.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]); let acc = 0; for (const [v] of srt) { if (v > 0) width++; acc += v; if (acc / tot >= 0.95) break; }
      const rsel = sel ? (sel.l ? L[sel.l - 1].r : 1) : 1;
      q(root, '#out').innerHTML = tx + `\nselected: layer ${sel ? sel.l : 0}, neuron ${sel ? sel.i : 0}\ntheoretical RF <b class="o">${rsel}</b> px; 95% of the influence lies in <b class="g">${width}</b> px`;
      q(root, '#cap').textContent = sel ? (sel.l === 0 ? 'An input pixel sees only itself.' : `Layer ${sel.l} neuron ${sel.i}: tracing back ${Math.min(rev, sel.l)} of ${sel.l} layers; every cell lit in blue can change its value.`) : '';
    }, { fps: 30 });
  };

  /* ================= 3.6 normaxes ================= */
  AI.labs.normaxes = function (root) {
    const Nn = 4, Cc = 6;
    const { view } = AI.labShell(root, 'Which values share statistics?', 'a tiny activation tensor: rows = images in the batch, columns = channels, each cell a 3×3 patch',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `<div class="row" id="mode"></div><label>GroupNorm groups G <select id="G"><option>1</option><option selected>2</option><option>3</option><option>6</option></select></label>
       ${playHTML(0.7, 'groups / s', 0.2, 3, 0.1)}<div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 330), tick = AI.ticker();
    let mode = 'bn', gi = 0, last = '', G = 2;
    const gid = (n, ch) => (mode === 'bn' ? ch : mode === 'ln' ? n : mode === 'in' ? n * Cc + ch : n * G + Math.floor(ch / (Cc / G)));
    const ng = () => (mode === 'bn' ? Cc : mode === 'ln' ? Nn : mode === 'in' ? Nn * Cc : Nn * G);
    chipRow(q(root, '#mode'), [['bn', 'BatchNorm'], ['ln', 'LayerNorm'], ['in', 'InstanceNorm'], ['gn', 'GroupNorm']], (m) => { mode = m; gi = 0; last = ''; }, mode);
    q(root, '#G').addEventListener('input', () => { G = +q(root, '#G').value; gi = 0; last = ''; });
    const st = player(root, 'groups / s', { onStep: () => { gi = (gi + 1) % ng(); last = ''; }, onRestart: () => { gi = 0; last = ''; } });
    const rg = AI.rng(5), pix = []; for (let i = 0; i < Nn * Cc * 9; i++) pix.push(rg());
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, st.speed()); if (n) gi = (gi + n) % ng(); }
      const key = [gi, mode, G, c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin(), x0 = 90, y0 = 50, cw = 84, rh = 56, hues = [C.blue, C.green, C.purple, C.orange, C.teal, C.pink];
      label(g, 'channels c →', x0, 28, C.muted, 12); label(g, 'batch n ↓', 10, 40, C.muted, 12);
      for (let n = 0; n < Nn; n++) for (let ch = 0; ch < Cc; ch++) {
        const X = x0 + ch * cw, Y = y0 + n * rh, id = gid(n, ch), hot = id === gi, col = hues[id % hues.length];
        g.fillStyle = col; g.globalAlpha = hot ? 0.55 : 0.14; g.fillRect(X, Y, cw - 6, rh - 6); g.globalAlpha = 1;
        g.strokeStyle = hot ? C.fg : col; g.lineWidth = hot ? 2.5 : 1; g.strokeRect(X + 0.5, Y + 0.5, cw - 7, rh - 7);
        for (let a = 0; a < 9; a++) { g.fillStyle = gray(pix[(n * Cc + ch) * 9 + a]); g.fillRect(X + 22 + (a % 3) * 12, Y + 7 + Math.floor(a / 3) * 12, 11, 11); }
      }
      for (let ch = 0; ch < Cc; ch++) label(g, 'c' + ch, x0 + ch * cw + (cw - 6) / 2, 44, C.muted, 11, 'center');
      for (let n = 0; n < Nn; n++) label(g, 'n' + n, x0 - 8, y0 + n * rh + 28, C.muted, 11, 'right');
      const sz = mode === 'bn' ? Nn * 9 : mode === 'ln' ? Cc * 9 : mode === 'in' ? 9 : (Cc / G) * 9;
      const nm = { bn: 'BatchNorm', ln: 'LayerNorm', in: 'InstanceNorm', gn: `GroupNorm (G=${G})` }[mode];
      label(g, `Highlighted group: its ${sz} values share ONE mean and ONE variance.`, 20, 310, C.fg, 13);
      c.end();
      q(root, '#out').innerHTML = `<b>${nm}</b>\nvalues per group  <b class="o">${sz}</b>\ngroups (μ,σ² pairs) <b class="b">${ng()}</b>\nlearnable γ,β: ${Cc} each (per channel)\ndepends on batch size: <b class="${mode === 'bn' ? 'r' : 'g'}">${mode === 'bn' ? 'yes' : 'no'}</b>`;
      q(root, '#cap').textContent = { bn: 'BatchNorm: all images and all pixels of ONE channel. At test time stored running averages replace the batch.', ln: 'LayerNorm: one image, all channels and pixels (transformers, ConvNeXt).', in: 'InstanceNorm: one image and one channel (style transfer).', gn: 'GroupNorm: one image, a group of channels; independent of batch size.' }[mode];
    }, { fps: 30 });
  };

  /* ================= 3.6 normlab ================= */
  AI.labs.normlab = function (root) {
    const { view } = AI.labShell(root, 'Batch normalisation in motion', 'the histogram of one channel is shifted, rescaled, then re-scaled by the learned γ and β',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `${slider('mu', 'Input mean μ₀', -6, 6, 3, 0.5)}${slider('sg', 'Input std σ₀', 0.5, 5, 2.5, 0.1)}${slider('ga', 'γ (scale)', 0.2, 3, 1, 0.1)}${slider('be', 'β (shift)', -3, 3, 0, 0.1)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="relu"> then ReLU</label>
       ${playHTML(0.7, 'phases / s', 0.2, 2, 0.1)}
       <div>Mini-batch of 4 (editable)<div class="row" id="mb" style="margin-top:4px"></div></div><div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 330);
    const r = AI.rng(3), Z = []; for (let i = 0; i < 1500; i++) { Z.push(r() < 0.7 ? -0.6 + 0.8 * AI.randn(r) : 1.4 + 0.6 * AI.randn(r)); }
    { const s = AI.stats(Z); for (let i = 0; i < Z.length; i++) Z[i] = (Z[i] - s.mean) / s.std; }
    let tm = 0, last = '', mb = [2, 4, 6, 8];
    const mbIn = mb.map((v, i) => { const e = document.createElement('input'); e.type = 'number'; e.value = v; e.step = 1; e.style.cssText = 'width:64px'; e.oninput = () => { mb[i] = +e.value || 0; last = ''; }; q(root, '#mb').appendChild(e); return e; });
    const st = player(root, 'phases / s', { onStep: () => { tm = Math.floor(tm) + 1; if (tm > 4) tm = 0; last = ''; }, onRestart: () => { tm = 0; last = ''; } });
    ['#mu', '#sg', '#ga', '#be', '#relu'].forEach((s2) => q(root, s2).addEventListener('input', () => { last = ''; }));
    const BINS = 56, LO = -12, HI = 12;
    const hist = (arr) => { const h = new Float32Array(BINS); for (let i = 0; i < arr.length; i++) { const b = Math.floor(((arr[i] - LO) / (HI - LO)) * BINS); if (b >= 0 && b < BINS) h[b]++; } return h; };
    const tmp = new Float32Array(Z.length);
    AI.loop(root, (dt) => {
      const useRelu = q(root, '#relu').checked, maxT = useRelu ? 4.8 : 3.8;
      if (st.playing) { tm += dt * st.speed(); if (tm > maxT) tm = 0; }
      const mu0 = +q(root, '#mu').value, sg0 = +q(root, '#sg').value, ga = +q(root, '#ga').value, be = +q(root, '#be').value;
      ['mu', 'sg', 'ga', 'be'].forEach((n) => (q(root, '#' + n + 'v').textContent = q(root, '#' + n).value));
      const key = [tm.toFixed(2), mu0, sg0, ga, be, useRelu, mb.join(), c.w, document.documentElement.dataset.theme].join();
      if (key === last) return; last = key;
      const eA = ease(tm), eB = ease(tm - 1), eC = ease(tm - 2), eD = ease(tm - 3);
      const x = Z.map((z) => mu0 + sg0 * z), s = AI.stats(x), sig = Math.sqrt(s.std * s.std + 1e-5);
      for (let i = 0; i < x.length; i++) { let v = x[i] - s.mean * eA; v = v / (1 + (sig - 1) * eB); v = v * (1 + (ga - 1) * eC) + be * eC; if (useRelu) v = v + (Math.max(0, v) - v) * eD; tmp[i] = v; }
      const h0 = hist(x), h1 = hist(tmp), mx = Math.max(...h0, ...h1, 1), cs = AI.stats(Array.from(tmp));
      const C = AI.col(), g = c.begin(), x0 = 30, wd = 580, y0 = 250, hh = 200;
      g.strokeStyle = C.border; g.lineWidth = 1; for (let v = -12; v <= 12; v += 4) { const X = x0 + ((v - LO) / (HI - LO)) * wd; g.beginPath(); g.moveTo(X, 40); g.lineTo(X, y0); g.stroke(); label(g, v, X, y0 + 15, C.muted, 11, 'center'); }
      const X0 = x0 + ((0 - LO) / (HI - LO)) * wd; g.strokeStyle = C.muted; g.beginPath(); g.moveTo(X0, 40); g.lineTo(X0, y0); g.stroke();
      const bw = wd / BINS;
      g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.lineWidth = 1.5; g.beginPath(); for (let b = 0; b < BINS; b++) { const y = y0 - (h0[b] / mx) * hh; b ? g.lineTo(x0 + b * bw, y) : g.moveTo(x0 + b * bw, y); g.lineTo(x0 + (b + 1) * bw, y); } g.stroke(); g.setLineDash([]);
      g.fillStyle = C.blue; g.globalAlpha = 0.75; for (let b = 0; b < BINS; b++) { const hgt = (h1[b] / mx) * hh; g.fillRect(x0 + b * bw + 1, y0 - hgt, bw - 2, hgt); } g.globalAlpha = 1;
      label(g, 'dashed = original activations (before BN); blue = current stage', x0, 28, C.muted, 12);
      const ph = tm < 1 ? '① subtract the batch mean μ' : tm < 2 ? '② divide by the batch std σ' : tm < 3 ? '③ scale by γ and shift by β' : tm < 4 ? (useRelu ? '④ ReLU zeroes the negatives' : 'done: the next layer sees mean β, std γ') : 'done';
      label(g, ph, 320, 290, C.fg, 15, 'center');
      label(g, `now: mean ${fx(cs.mean)}   std ${fx(cs.std)}`, 320, 312, C.orange, 13, 'center');
      c.end();
      const m = mb.reduce((a, b) => a + b, 0) / 4, v = mb.reduce((a, b) => a + (b - m) ** 2, 0) / 4, xh = mb.map((b) => (b - m) / Math.sqrt(v + 1e-5)), y = xh.map((b) => ga * b + be), f = (a) => a.map((q2) => fx(q2, 3).padStart(7)).join('');
      q(root, '#out').innerHTML = `<span class="m">toy batch (ε=1e-5, γ=${ga}, β=${be})</span>\nx    ${f(mb)}\nμ = ${fx(m, 3)}   σ² = ${fx(v, 3)} (÷4)\nx̂    ${f(xh)}\ny    ${f(y)}\n<span class="m">fraction negative now:</span> ${fx(Array.from(tmp).filter((q2) => q2 < 0).length / tmp.length, 2)}`;
      q(root, '#cap').textContent = 'The histogram keeps its shape (BN does not make it Gaussian); only its centre and spread change. γ and β let the network undo the normalisation if that is best.';
    }, { fps: 30 });
  };

  /* ================= 3.6 dropoutdemo ================= */
  AI.labs.dropoutdemo = function (root) {
    const { view } = AI.labShell(root, 'Dropout', 'training: each unit is zeroed with probability p, survivors are scaled by 1/(1−p); evaluation: nothing changes',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `${slider('p', 'Drop probability p', 0.1, 0.9, 0.5, 0.1)}<div class="row" id="md"></div>${playHTML(0.7, 'masks / s', 0.2, 3, 0.1)}<div class="readout" id="out"></div>`);
    const c = dcanvas(view, 640, 260), tick = AI.ticker(), N = 16, A = [3, 5, 2, 6, 4, 1, 7, 3, 5, 2, 4, 6, 1, 3, 5, 4];
    let mask = A.map(() => 1), train = true, sum = 0, cnt = 0, fade = 1, last = '';
    const orig = A.reduce((a, b) => a + b, 0);
    const draw = () => { const p = +q(root, '#p').value; mask = A.map(() => (Math.random() < p ? 0 : 1)); sum += A.reduce((a, v, i) => a + (mask[i] ? v / (1 - p) : 0), 0); cnt++; fade = 0; };
    chipRow(q(root, '#md'), [['t', 'training'], ['e', 'evaluation']], (m) => { train = m === 't'; sum = 0; cnt = 0; mask = A.map(() => 1); last = ''; }, 't');
    q(root, '#p').addEventListener('input', () => { sum = 0; cnt = 0; last = ''; });
    const st = player(root, 'masks / s', { onStep: () => train && draw(), onRestart: () => { sum = 0; cnt = 0; } });
    AI.loop(root, (dt) => {
      const p = +q(root, '#p').value; q(root, '#pv').textContent = p;
      if (st.playing && train) { const n = tick(dt, st.speed()); if (n) draw(); }
      const mv = fade < 1; fade = Math.min(1, fade + dt * 3);
      const key = [mask.join(''), p, train, cnt, c.w, document.documentElement.dataset.theme, fade.toFixed(1)].join();
      if (key === last && !mv) return; last = key;
      const C = AI.col(), g = c.begin(), x0 = 30, bw = 33, base = 225, sc = 13;
      label(g, 'activations of 16 units (grey = original value, coloured = value passed on)', x0, 20, C.muted, 12);
      for (let i = 0; i < N; i++) {
        const X = x0 + i * bw, on = !train || mask[i], v = train && on ? A[i] / (1 - p) : on ? A[i] : 0;
        g.strokeStyle = C.border; g.strokeRect(X + 0.5, base - A[i] * sc + 0.5, bw - 6, A[i] * sc);
        g.fillStyle = on ? C.blue : C.red; g.globalAlpha = on ? 1 : 0.35; const hv = on ? v : A[i];
        g.fillRect(X + 2, base - Math.min(hv, 14) * sc * (on ? fade : 1), bw - 10, Math.min(hv, 14) * sc * (on ? fade : 1)); g.globalAlpha = 1;
        label(g, on ? fx(v, 1) : '✕', X + (bw - 6) / 2, base + 14, on ? C.fg : C.red, 11, 'center');
      }
      c.end();
      const cur = train ? A.reduce((a, v, i) => a + (mask[i] ? v / (1 - p) : 0), 0) : orig;
      q(root, '#out').innerHTML = `<span class="m">original sum</span> <b>${orig}</b>\n<span class="m">this mask's sum</span> <b class="o">${fx(cur, 1)}</b>\n<span class="m">running mean over ${cnt} masks</span> <b class="g">${cnt ? fx(sum / cnt, 2) : '–'}</b>\nscale 1/(1−p) = ${fx(1 / (1 - p), 2)}`;
      q(root, '#cap').textContent = train ? 'Each mask gives a different noisy sum, but the long-run average equals the original: scaling by 1/(1−p) keeps the expected activation unchanged.' : 'Evaluation mode: all units are active and unscaled.';
    }, { fps: 30 });
  };

  /* ================= tiny CNN (shared by cnntrain and cnnviz) ================= */
  const F = 4, NC = 3, CLS = ['horizontal line', 'vertical line', 'cross'];
  const Tiny = {
    makeImg(cls, r) {
      const x = new Float32Array(64); for (let i = 0; i < 64; i++) x[i] = 0.12 * r();
      const ri = (a, b) => a + Math.floor(r() * (b - a + 1)), hl = (row, c0, c1) => { for (let cc = c0; cc <= c1; cc++) x[row * 8 + cc] = 0.85 + 0.15 * r(); }, vl = (col, r0, r1) => { for (let qq = r0; qq <= r1; qq++) x[qq * 8 + col] = 0.85 + 0.15 * r(); };
      if (cls === 0) { const len = ri(5, 8), c0 = ri(0, 8 - len); hl(ri(0, 7), c0, c0 + len - 1); }
      else if (cls === 1) { const len = ri(5, 8), r0 = ri(0, 8 - len); vl(ri(0, 7), r0, r0 + len - 1); }
      else { const rr = ri(2, 5), cc = ri(2, 5); hl(rr, cc - 2, cc + 2); vl(cc, rr - 2, rr + 2); }
      return x;
    },
    net(seed) { const r = AI.rng(seed), n = { Wc: new Float32Array(F * 9), bc: new Float32Array(F), Wf: new Float32Array(NC * F * 16), bf: new Float32Array(NC) }; for (let i = 0; i < n.Wc.length; i++) n.Wc[i] = AI.randn(r) * 0.3; for (let i = 0; i < n.Wf.length; i++) n.Wf[i] = AI.randn(r) * 0.15; return n; },
    zero() { return { Wc: new Float32Array(F * 9), bc: new Float32Array(F), Wf: new Float32Array(NC * F * 16), bf: new Float32Array(NC) }; },
    fwd(n, x) {
      const A = new Float32Array(F * 64), R = new Float32Array(F * 64), P = new Float32Array(F * 16), idx = new Int16Array(F * 16);
      for (let f = 0; f < F; f++) for (let y = 0; y < 8; y++) for (let c = 0; c < 8; c++) { let s = n.bc[f]; for (let ky = 0; ky < 3; ky++) { const yy = y + ky - 1; if (yy < 0 || yy > 7) continue; for (let kx = 0; kx < 3; kx++) { const xx = c + kx - 1; if (xx < 0 || xx > 7) continue; s += n.Wc[f * 9 + ky * 3 + kx] * x[yy * 8 + xx]; } } A[f * 64 + y * 8 + c] = s; R[f * 64 + y * 8 + c] = s > 0 ? s : 0; }
      for (let f = 0; f < F; f++) for (let y = 0; y < 4; y++) for (let c = 0; c < 4; c++) { let b = -1, bi = 0; for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const k = f * 64 + (2 * y + j) * 8 + 2 * c + i; if (b < 0 || R[k] > b) { b = R[k]; bi = k; } } P[f * 16 + y * 4 + c] = b; idx[f * 16 + y * 4 + c] = bi; }
      const z = new Float32Array(NC); for (let o = 0; o < NC; o++) { let s = n.bf[o]; for (let i = 0; i < F * 16; i++) s += n.Wf[o * F * 16 + i] * P[i]; z[o] = s; }
      const m = Math.max(...z); let e = 0; const p = Array.from(z, (v) => { const q2 = Math.exp(v - m); e += q2; return q2; }); for (let o = 0; o < NC; o++) p[o] /= e;
      return { A, R, P, idx, z, p };
    },
    bwd(n, x, c, dz, g) {
      const dP = new Float32Array(F * 16); for (let o = 0; o < NC; o++) for (let i = 0; i < F * 16; i++) dP[i] += n.Wf[o * F * 16 + i] * dz[o];
      if (g) for (let o = 0; o < NC; o++) { g.bf[o] += dz[o]; for (let i = 0; i < F * 16; i++) g.Wf[o * F * 16 + i] += dz[o] * c.P[i]; }
      const dA = new Float32Array(F * 64), dR = new Float32Array(F * 64); for (let i = 0; i < F * 16; i++) { const k = c.idx[i]; dR[k] += dP[i]; if (c.A[k] > 0) dA[k] += dP[i]; }
      const dx = new Float32Array(64);
      for (let f = 0; f < F; f++) for (let y = 0; y < 8; y++) for (let cc = 0; cc < 8; cc++) { const d = dA[f * 64 + y * 8 + cc]; if (!d) continue; if (g) g.bc[f] += d; for (let ky = 0; ky < 3; ky++) { const yy = y + ky - 1; if (yy < 0 || yy > 7) continue; for (let kx = 0; kx < 3; kx++) { const xx = cc + kx - 1; if (xx < 0 || xx > 7) continue; if (g) g.Wc[f * 9 + ky * 3 + kx] += d * x[yy * 8 + xx]; dx[yy * 8 + xx] += d * n.Wc[f * 9 + ky * 3 + kx]; } } }
      return { dA, dR, dx };
    },
    step(n, r, lr, B) {
      const g = Tiny.zero(); let loss = 0;
      for (let b = 0; b < B; b++) { const cl = Math.floor(r() * 3), x = Tiny.makeImg(cl, r), o = Tiny.fwd(n, x); loss -= Math.log(o.p[cl] + 1e-9); Tiny.bwd(n, x, o, o.p.map((v, i) => (v - (i === cl ? 1 : 0)) / B), g); }
      for (const k of ['Wc', 'bc', 'Wf', 'bf']) for (let i = 0; i < n[k].length; i++) n[k][i] -= lr * g[k][i];
      return loss / B;
    },
    testSet() { const r = AI.rng(101), t = []; for (let i = 0; i < 30; i++) { const c = i % 3; t.push([Tiny.makeImg(c, r), c]); } return t; },
    acc(n, t) { let a = 0; t.forEach(([x, c]) => { const p = Tiny.fwd(n, x).p; if (p.indexOf(Math.max(...p)) === c) a++; }); return a / t.length; },
  };
  AI.cv = AI.cv || {}; AI.cv.tiny = Tiny;

  /* ================= 3.7 cnntrain ================= */
  AI.labs.cnntrain = function (root) {
    const { view } = AI.labShell(root, 'Train a tiny CNN', 'conv 3×3 (4 filters) → ReLU → max-pool 2 → FC 3 on 8×8 images; 151 weights learned by backprop',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `${playHTML(4, 'steps / s', 0.5, 60, 0.5)}${slider('lr', 'Learning rate η', 0.02, 0.6, 0.25, 0.01)}<label>Batch size <select id="bs"><option>4</option><option selected>8</option><option>16</option></select></label>
       <div class="row"><button class="btn" id="new">New random start</button></div><div class="readout" id="out"></div>`);
    const c = dcanvas(view, 680, 480), tick = AI.ticker(), test = Tiny.testSet();
    let seed = 11, net = Tiny.net(seed), r = AI.rng(7), steps = 0, hist = [], accs = [], sel = 0, last = '', ema = null, dirty = true;
    const reset = () => { seed++; net = Tiny.net(seed); r = AI.rng(seed * 13); steps = 0; hist = []; accs = [[0, Tiny.acc(net, test)]]; ema = null; dirty = true; };
    const doStep = () => { const l = Tiny.step(net, r, +q(root, '#lr').value, +q(root, '#bs').value); steps++; ema = ema == null ? l : 0.9 * ema + 0.1 * l; hist.push(ema); if (steps % 5 === 0) accs.push([steps, Tiny.acc(net, test)]); dirty = true; };
    accs = [[0, Tiny.acc(net, test)]];
    const st = player(root, 'steps / s', { onStep: () => doStep(), onRestart: () => reset() });
    q(root, '#new').onclick = reset; q(root, '#lr').addEventListener('input', () => (q(root, '#lrv').textContent = q(root, '#lr').value)); q(root, '#lrv').textContent = 0.25;
    AI.$('canvas', view).addEventListener('pointerdown', (e) => { const [mx, my] = c.pt(e); if (my > 20 && my < 70) { const i = Math.floor((mx - 10) / 50); if (i >= 0 && i < 6) { sel = i; dirty = true; } } });
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, st.speed()); for (let i = 0; i < n; i++) doStep(); }
      const key = [steps, sel, dirty, c.w, document.documentElement.dataset.theme].join(); if (!dirty && key === last) return; last = key; dirty = false;
      const C = AI.col(), g = c.begin(), [x, cl] = test[sel], o = Tiny.fwd(net, x);
      label(g, 'test images (click one)', 10, 14, C.muted, 11);
      for (let i = 0; i < 6; i++) { const xi = test[i][0]; for (let k = 0; k < 64; k++) { g.fillStyle = gray(xi[k]); g.fillRect(10 + i * 50 + (k % 8) * 5, 24 + Math.floor(k / 8) * 5, 5, 5); } if (i === sel) { g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(9 + i * 50, 23, 42, 42); } }
      // selected sample
      label(g, 'selected: ' + CLS[cl], 10, 92, C.fg, 12);
      for (let k = 0; k < 64; k++) { g.fillStyle = gray(x[k]); g.fillRect(10 + (k % 8) * 14, 100 + Math.floor(k / 8) * 14, 13, 13); }
      // probabilities
      label(g, 'prediction (softmax)', 10, 232, C.muted, 11);
      for (let k = 0; k < 3; k++) { g.fillStyle = k === cl ? C.green : C.blue; g.fillRect(10, 242 + k * 24, 110 * o.p[k], 14); label(g, `${CLS[k]} ${fx(o.p[k] * 100, 0)}%`, 14, 254 + k * 24, C.fg, 11); }
      // filters, maps, pooled
      label(g, 'filter', 168, 14, C.muted, 11); label(g, 'maps 8×8', 244, 14, C.muted, 11); label(g, 'pool 4×4', 316, 14, C.muted, 11);
      let mxw = 0.01; for (let i = 0; i < net.Wc.length; i++) mxw = Math.max(mxw, Math.abs(net.Wc[i]));
      let mxr = 0.01; for (let i = 0; i < o.R.length; i++) mxr = Math.max(mxr, o.R[i]);
      for (let f = 0; f < F; f++) {
        const y0 = 24 + f * 72;
        for (let k = 0; k < 9; k++) { const w = net.Wc[f * 9 + k]; g.fillStyle = divcol(w, mxw, C); g.fillRect(168 + (k % 3) * 20, y0 + Math.floor(k / 3) * 20, 19, 19); label(g, fx(w, 1), 168 + (k % 3) * 20 + 9.5, y0 + Math.floor(k / 3) * 20 + 14, C.fg, 8.5, 'center'); }
        for (let k = 0; k < 64; k++) { g.fillStyle = AI.mix(C.bg2, C.blue, o.R[f * 64 + k] / mxr); g.fillRect(244 + (k % 8) * 8, y0 + Math.floor(k / 8) * 8, 7, 7); }
        for (let k = 0; k < 16; k++) { g.fillStyle = AI.mix(C.bg2, C.blue, o.P[f * 16 + k] / mxr); g.fillRect(316 + (k % 4) * 12, y0 + Math.floor(k / 4) * 12, 11, 11); }
        label(g, 'f' + f, 150, y0 + 34, C.muted, 11);
      }
      // classifier weights
      label(g, 'classifier weights: class × filter × 4×4 location', 392, 14, C.muted, 11);
      let mxf = 0.01; for (let i = 0; i < net.Wf.length; i++) mxf = Math.max(mxf, Math.abs(net.Wf[i]));
      for (let k = 0; k < 3; k++) { label(g, CLS[k], 392, 44 + k * 78 - 6, k === cl ? C.green : C.fg, 11); for (let f = 0; f < F; f++) for (let t = 0; t < 16; t++) { g.fillStyle = divcol(net.Wf[k * 64 + f * 16 + t], mxf, C); g.fillRect(392 + f * 70 + (t % 4) * 15, 44 + k * 78 + Math.floor(t / 4) * 15, 14, 14); } }
      // curves
      const cx0 = 40, cy0 = 462, cw = 620, chh = 105, xmax = Math.max(100, steps);
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(cx0, cy0 - chh, cw, chh);
      label(g, 'loss (orange, smoothed) and test accuracy (blue) vs training step', cx0, cy0 - chh - 5, C.muted, 11);
      const PX = (s2) => cx0 + (s2 / xmax) * cw, lmax = 1.4;
      g.strokeStyle = C.orange; g.lineWidth = 2; g.beginPath(); hist.forEach((l, i) => { const X = PX(i + 1), Y = cy0 - AI.clamp(l / lmax, 0, 1) * chh; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke();
      g.strokeStyle = C.blue; g.beginPath(); accs.forEach(([s2, a], i) => { const X = PX(s2), Y = cy0 - a * chh; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke();
      label(g, 'step ' + xmax, cx0 + cw, cy0 + 12, C.muted, 10, 'right'); label(g, '0', cx0, cy0 + 12, C.muted, 10);
      c.end();
      const a = accs[accs.length - 1][1];
      q(root, '#out').innerHTML = `<span class="m">step</span> ${steps}\n<span class="m">loss (smoothed)</span> <b class="o">${ema == null ? '…' : fx(ema, 3)}</b>\n<span class="m">test accuracy</span> <b class="b">${fx(a * 100, 0)}%</b>\n<span class="m">params</span> 4·9+4 + 64·3+3 = <b>235</b>`;
      q(root, '#cap').textContent = `Step ${steps}: ① a batch goes forward ② cross-entropy ③ backprop (conv-weight gradients summed over all 64 positions) ④ w ← w − η·gradient. Filters turn from noise into line detectors.`;
    }, { fps: 30 });
  };

  /* ================= 3.8 cnnviz ================= */
  AI.labs.cnnviz = function (root) {
    const { view } = AI.labShell(root, 'Look inside the trained CNN', 'paint an image, then see saliency, Grad-CAM and occlusion explain the prediction',
      '<canvas></canvas><div class="caption" id="cap"></div>',
      `<div>Method<div class="row" id="meth" style="margin-top:4px"></div></div><div>Target class<div class="row" id="tcl" style="margin-top:4px"></div></div>
       <div>Paint<div class="row" id="pt" style="margin-top:4px"></div></div><div>Examples<div class="row" id="ex" style="margin-top:4px"></div></div>
       ${playHTML(1, 'speed ×', 0.3, 3, 0.1)}<div class="readout" id="out"></div>`);
    const net = Tiny.net(11), rr = AI.rng(5); for (let s2 = 0; s2 < 400; s2++) Tiny.step(net, rr, 0.3, 8);
    const c = dcanvas(view, 680, 480), er = AI.rng(77);
    let x = Tiny.makeImg(2, er), method = 'gradcam', target = 2, erase = false, tm = 0, last = '', an = null, drag = false;
    const analyse = () => {
      const o = Tiny.fwd(net, x); target = Math.min(target, 2);
      const dz = [0, 0, 0]; dz[target] = 1; const b = Tiny.bwd(net, x, o, dz, null);
      const alpha = []; for (let f = 0; f < F; f++) { let s2 = 0; for (let k = 0; k < 64; k++) s2 += b.dR[f * 64 + k]; alpha.push(s2 / 64); }
      const contrib = []; for (let f = 0; f < F; f++) contrib.push(Array.from(o.R.subarray(f * 64, f * 64 + 64), (v) => v * alpha[f]));
      const sal = Array.from(b.dx, Math.abs), occ = [];
      for (let y = 0; y < 8; y++) for (let xx = 0; xx < 8; xx++) { const x2 = Float32Array.from(x); for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) { const yy = y + u, cx = xx + v; if (yy >= 0 && yy < 8 && cx >= 0 && cx < 8) x2[yy * 8 + cx] = 0; } occ.push(Math.max(0, o.z[target] - Tiny.fwd(net, x2).z[target])); }
      an = { o, alpha, contrib, sal, occ }; last = ''; tm = 0;
    };
    chipRow(q(root, '#meth'), [['gradcam', 'Grad-CAM'], ['saliency', 'saliency'], ['occlusion', 'occlusion']], (m) => { method = m; tm = 0; last = ''; }, method);
    const markT = chipRow(q(root, '#tcl'), CLS.map((n, i) => [i, n]), (t) => { target = +t; analyse(); }, target);
    chipRow(q(root, '#pt'), [['d', 'draw'], ['e', 'erase'], ['c', 'clear']], (m) => { if (m === 'c') { x = new Float32Array(64); analyse(); } else erase = m === 'e'; }, 'd');
    chipRow(q(root, '#ex'), [[0, 'h-line'], [1, 'v-line'], [2, 'cross']], (m) => { x = Tiny.makeImg(+m, er); analyse(); }, '');
    const st = player(root, 'speed ×', { onStep: () => { tm += 1.6; last = ''; }, onRestart: () => { tm = 0; last = ''; } });
    const gx = 14, gy = 34, cs = 28;
    const paint = (e) => { const [mx, my] = c.pt(e), xx = Math.floor((mx - gx) / cs), yy = Math.floor((my - gy) / cs); if (xx >= 0 && yy >= 0 && xx < 8 && yy < 8) { const v = erase ? 0 : 1; if (x[yy * 8 + xx] !== v) { x[yy * 8 + xx] = v; analyse(); } } };
    const cv = AI.$('canvas', view); cv.style.touchAction = 'none';
    cv.addEventListener('pointerdown', (e) => { drag = true; cv.setPointerCapture(e.pointerId); paint(e); }); cv.addEventListener('pointermove', (e) => drag && paint(e)); cv.addEventListener('pointerup', () => (drag = false));
    analyse();
    AI.loop(root, (dt) => {
      if (st.playing) tm += dt * st.speed();
      const hold = 3, stages = method === 'gradcam' ? F + 2 : 0;
      let heat = new Array(64).fill(0), info = '', stage = 0, cnt = 0, ctot = 0;
      if (method === 'gradcam') { const cyc = (stages * 1.6) + hold; if (tm > cyc) tm = 0; stage = Math.min(stages - 1, Math.floor(tm / 1.6)); const upto = Math.min(F, stage); for (let f = 0; f < upto; f++) for (let k = 0; k < 64; k++) heat[k] += an.contrib[f][k]; if (stage === F + 1) heat = heat.map((v) => Math.max(0, v)); ctot = stages; }
      else if (method === 'saliency') { if (tm > 5) tm = 0; const f2 = Math.min(1, tm / 1.5); heat = an.sal.map((v) => v * f2); }
      else { if (tm > 64 / 6 + hold) tm = 0; cnt = Math.min(64, Math.floor(tm * 6)); heat = an.occ.map((v, i) => (i < cnt ? v : 0)); }
      const key = [stage, cnt, Math.round(tm * 20), method, target, c.w, document.documentElement.dataset.theme, Array.from(x).join('')].join();
      if (key === last) return; last = key;
      const C = AI.col(), g = c.begin(), o = an.o;
      let mx = 1e-9; if (method === 'gradcam') { for (let f = 0; f < F; f++) for (let k = 0; k < 64; k++) mx = Math.max(mx, Math.abs(an.contrib[f][k])); mx = Math.max(mx, ...heat.map(Math.abs)); } else mx = Math.max(1e-9, ...(method === 'saliency' ? an.sal : an.occ));
      const isCam = method === 'gradcam' && stage > 0 && stage < F + 1;
      label(g, 'input (click / drag to paint)', gx, 22, C.muted, 12);
      for (let k = 0; k < 64; k++) { const X = gx + (k % 8) * cs, Y = gy + Math.floor(k / 8) * cs; g.fillStyle = gray(x[k]); g.fillRect(X, Y, cs - 1, cs - 1); const h = heat[k] / mx; if (h > 0 && !isCam) { g.fillStyle = C.red; g.globalAlpha = Math.min(0.85, h * 0.85); g.fillRect(X, Y, cs - 1, cs - 1); g.globalAlpha = 1; } else if (isCam && h !== 0) { g.fillStyle = h > 0 ? C.red : C.blue; g.globalAlpha = Math.min(0.8, Math.abs(h) * 0.8); g.fillRect(X, Y, cs - 1, cs - 1); g.globalAlpha = 1; } }
      if (method === 'occlusion' && cnt > 0 && cnt < 64) { const i = cnt - 1, X = gx + ((i % 8) - 1) * cs, Y = gy + (Math.floor(i / 8) - 1) * cs; g.strokeStyle = C.orange; g.lineWidth = 2.5; g.strokeRect(X, Y, 3 * cs, 3 * cs); }
      label(g, 'prediction', gx, 280, C.muted, 12);
      for (let k = 0; k < 3; k++) { g.fillStyle = k === target ? C.orange : C.blue; g.fillRect(gx, 290 + k * 22, 150 * o.p[k], 13); label(g, `${CLS[k]} ${fx(o.p[k] * 100, 0)}%`, gx + 4, 301 + k * 22, C.fg, 11); }
      markT(target);
      // right
      const rx = 262, colW = 100;
      label(g, 'learned filters → activation maps (after ReLU)' + (method === 'gradcam' ? ' → α·map' : ''), rx, 22, C.muted, 12);
      let mxw = 0.01; for (let i = 0; i < net.Wc.length; i++) mxw = Math.max(mxw, Math.abs(net.Wc[i])); let mxr = 0.01; for (let i = 0; i < o.R.length; i++) mxr = Math.max(mxr, o.R[i]);
      for (let f = 0; f < F; f++) {
        const X = rx + f * colW;
        for (let k = 0; k < 9; k++) { g.fillStyle = divcol(net.Wc[f * 9 + k], mxw, C); g.fillRect(X + (k % 3) * 14, 34 + Math.floor(k / 3) * 14, 13, 13); }
        for (let k = 0; k < 64; k++) { g.fillStyle = AI.mix(C.bg2, C.blue, o.R[f * 64 + k] / mxr); g.fillRect(X + (k % 8) * 11, 90 + Math.floor(k / 8) * 11, 10, 10); }
        const lit = method === 'gradcam' && stage >= 1 + f && stage <= F;
        if (method === 'gradcam') {
          label(g, 'α' + f + ' = ' + fx(an.alpha[f], 3), X, 192, stage >= 0 ? (an.alpha[f] >= 0 ? C.green : C.orange) : C.muted, 11);
          for (let k = 0; k < 64; k++) { const v = an.contrib[f][k] / mx; g.fillStyle = v >= 0 ? AI.mix(C.bg2, C.red, Math.min(1, v)) : AI.mix(C.bg2, C.blue, Math.min(1, -v)); g.globalAlpha = stage >= 1 + f ? 1 : 0.25; g.fillRect(X + (k % 8) * 11, 204 + Math.floor(k / 8) * 11, 10, 10); g.globalAlpha = 1; }
          if (lit) { g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(X - 2, 202, 92, 92); }
        }
      }
      // heat panel
      const hy = 312, hs = 20; label(g, method === 'gradcam' ? (stage === 0 ? 'heatmap (empty: nothing added yet)' : stage <= F ? `heatmap = Σ α·map, first ${stage} map${stage > 1 ? 's' : ''}` : 'heatmap = ReLU(Σ α·map)') : method === 'saliency' ? '|∂score/∂pixel|' : `score drop when a 3×3 patch is erased (${cnt}/64)`, rx, hy - 6, C.fg, 12);
      for (let k = 0; k < 64; k++) { const h = heat[k] / mx; g.fillStyle = h >= 0 ? AI.mix(C.bg2, C.red, Math.min(1, h)) : AI.mix(C.bg2, C.blue, Math.min(1, -h)); g.fillRect(rx + (k % 8) * hs, hy + Math.floor(k / 8) * hs, hs - 1, hs - 1); }
      const expl = method === 'gradcam' ? 'α_k = mean over cells of ∂score/∂A_k. Weighted maps are added; red = evidence for the class, blue = evidence against; ReLU keeps only red.' : method === 'saliency' ? 'Which pixels would change the class score most if nudged (one backward pass).' : 'Erase each 3×3 region and measure the drop of the class score (64 forward passes).';
      const words = expl.split(' '); let line = '', ly = hy + 14; words.forEach((w) => { if ((line + w).length > 36) { label(g, line, rx + 180, ly, C.muted, 11); ly += 14; line = ''; } line += w + ' '; }); label(g, line, rx + 180, ly, C.muted, 11);
      c.end();
      q(root, '#out').innerHTML = `<span class="m">target</span> ${CLS[target]}\n<span class="m">logit</span> ${fx(o.z[target], 2)}\n<span class="m">α per filter</span>\n${an.alpha.map((a, i) => 'f' + i + ': ' + fx(a, 3)).join('  ')}`;
      q(root, '#cap').textContent = method === 'gradcam' ? (stage === 0 ? 'Step 0: the four activation maps and their weights α (average gradient of the class score).' : stage <= F ? `Step ${stage}: add α${stage - 1} × map ${stage - 1} to the heatmap.` : 'Final step: ReLU keeps only the positive evidence.') : method === 'saliency' ? 'Saliency: gradient of the class score with respect to each input pixel.' : 'Occlusion: the orange 3×3 box erases pixels; the heatmap shows how much the class score fell.';
    }, { fps: 30 });
  };
})();
