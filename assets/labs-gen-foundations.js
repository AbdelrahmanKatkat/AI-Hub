/* Generative AI Module 0 labs: foundations of generative modelling.
   gen-fd-joint, gen-fd-mle, gen-fd-inverse, gen-fd-chain, gen-fd-chain-score, gen-fd-race, gen-fd-flow1d, gen-fd-tokens.
   All simulations are illustrative, use seeded random numbers (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = (first) => `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const clamp01 = (t) => Math.max(0, Math.min(1, t));
  const norm = (w) => { const s = w.reduce((a, b) => a + b, 0); return s > 0 ? w.map((x) => x / s) : w.map(() => 1 / w.length); };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const pickCum = (p, u) => { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; };
  const f3 = (x) => x.toFixed(3);

  /* =====================================================================
     1. A joint table: sample pairs (x, y), then condition
     ===================================================================== */
  AI.labs['gen-fd-joint'] = function (root) {
    const cellName = ['rain, umbrella', 'rain, no umbrella', 'dry, umbrella', 'dry, no umbrella'];
    const { view, ctl } = AI.labShell(root, 'One joint table, three questions', 'illustrative: x = does the person carry an umbrella, y = is it raining; dots are drawn from the table by inverse transform',
      `<canvas style="height:340px"></canvas>`,
      `${sl('w0', 'weight of (rain, umbrella)', 0, 1, 0.01, 0.2)}
       ${sl('w1', 'weight of (rain, no umbrella)', 0, 1, 0.01, 0.05)}
       ${sl('w2', 'weight of (dry, umbrella)', 0, 1, 0.01, 0.1)}
       ${sl('w3', 'weight of (dry, no umbrella)', 0, 1, 0.01, 0.65)}
       <label>What to generate <select id="mode"><option value="joint">a whole pair (x, y) from p(x, y)</option><option value="rain">only x, given y = rain: p(x | rain)</option><option value="umb">only y, given x = umbrella: p(y | umbrella)</option></select></label>
       ${sl('rate', 'Draws per second', 0.25, 15, 0.25, 1.5)}
       ${btns()}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let rs, counts, dots, landed, last, tick = AI.ticker();
    const reset = () => { rs = AI.rng(21); counts = [0, 0, 0, 0]; dots = []; landed = []; last = null; };
    const probs = () => norm([0, 1, 2, 3].map((i) => +q(ctl, '#w' + i).value));
    const allowed = () => { const m = q(ctl, '#mode').value; return m === 'joint' ? [0, 1, 2, 3] : m === 'rain' ? [0, 1] : [0, 2]; };
    const geom = () => {
      const W = o.w, H = o.h, L = 96, R = W - 12, top = 78, cw = (R - L) / 2, ch = (H - top - 14) / 2;
      return { W, H, L, R, top, cw, ch, cell: (i) => ({ x: L + (i % 2) * cw, y: top + (i < 2 ? 0 : 1) * ch }) };
    };
    const spawn = () => {
      const p = probs(), al = allowed(), pa = norm(al.map((i) => p[i])), u = rs(), j = pickCum(pa, u), cell = al[j];
      let cum = 0; for (let k = 0; k < j; k++) cum += pa[k];
      last = { u, al, pa, cell, p, lo: cum, hi: cum + pa[j] };
      dots.push({ cell, u, t: 0, jx: rs(), jy: rs() });
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, G = geom(); g.clearRect(0, 0, G.W, G.H);
      const cc = [c.blue, c.teal, c.orange, c.pink], p = probs(), al = allowed(), pa = norm(al.map((i) => p[i]));
      // strip of the unit interval over the allowed cells
      const sx0 = G.L, sx1 = G.R, sy = 30;
      txt(g, 'u in [0, 1): one random number picks a cell', G.L, 14, c.muted);
      let cum = 0;
      al.forEach((i, k) => {
        const x0 = sx0 + cum * (sx1 - sx0), x1 = sx0 + (cum + pa[k]) * (sx1 - sx0);
        g.fillStyle = cc[i]; g.globalAlpha = 0.55; g.fillRect(x0, sy, Math.max(0, x1 - x0), 16); g.globalAlpha = 1;
        if (x1 - x0 > 46) txt(g, pa[k].toFixed(2), (x0 + x1) / 2, sy + 12, c.fg, '11px sans-serif', 'center');
        cum += pa[k];
      });
      g.strokeStyle = c.border; g.strokeRect(sx0, sy, sx1 - sx0, 16);
      // cells
      txt(g, 'x: umbrella', G.L + G.cw * 0.5, G.top - 24, c.muted, '11px sans-serif', 'center'); txt(g, 'x: no umbrella', G.L + G.cw * 1.5, G.top - 24, c.muted, '11px sans-serif', 'center');
      txt(g, 'y: rain', G.L - 8, G.top + G.ch * 0.5, c.muted, '12px sans-serif', 'right'); txt(g, 'y: dry', G.L - 8, G.top + G.ch * 1.5, c.muted, '12px sans-serif', 'right');
      const total = counts.reduce((a, b) => a + b, 0);
      for (let i = 0; i < 4; i++) {
        const ce = G.cell(i), on = al.includes(i);
        g.fillStyle = cc[i]; g.globalAlpha = on ? 0.1 + 0.45 * p[i] : 0.04; g.fillRect(ce.x + 2, ce.y + 2, G.cw - 4, G.ch - 4); g.globalAlpha = 1;
        g.strokeStyle = on ? cc[i] : c.border; g.strokeRect(ce.x + 2, ce.y + 2, G.cw - 4, G.ch - 4);
        txt(g, 'p = ' + p[i].toFixed(3), ce.x + 8, ce.y + 18, on ? c.fg : c.muted, 'bold 12px sans-serif');
        const tot = al.reduce((s, k) => s + counts[k], 0);
        txt(g, counts[i] ? `seen ${counts[i]} = ${(counts[i] / Math.max(1, tot)).toFixed(3)}` : 'seen 0', ce.x + 8, ce.y + G.ch - 10, c.muted, '11px sans-serif');
      }
      // landed dots
      for (const d of landed) { const ce = G.cell(d.cell); g.fillStyle = cc[d.cell]; g.beginPath(); g.arc(ce.x + 10 + d.jx * (G.cw - 20), ce.y + 30 + d.jy * (G.ch - 58), 2.4, 0, 7); g.fill(); }
      // flying dots
      for (const d of dots) {
        const ce = G.cell(d.cell), tx = ce.x + 10 + d.jx * (G.cw - 20), ty = ce.y + 30 + d.jy * (G.ch - 58), e = ease(clamp01(d.t / 0.9));
        const sx = sx0 + d.u * (sx1 - sx0);
        g.fillStyle = c.fg; g.beginPath(); g.arc(sx + (tx - sx) * e, sy + 8 + (ty - sy - 8) * e, 4, 0, 7); g.fill();
      }
      void total;
    };
    const readout = () => {
      const p = probs(), pr = p[0] + p[1], pu = p[0] + p[2];
      let s = `<b>p(rain)</b> = ${f3(p[0])} + ${f3(p[1])} = ${f3(pr)} &nbsp; <b>p(umbrella)</b> = ${f3(p[0])} + ${f3(p[2])} = ${f3(pu)}<br>` +
        `<b>p(umbrella | rain)</b> = ${f3(p[0])} / ${f3(pr)} = ${f3(p[0] / pr)}<br>` +
        `<b>p(rain | umbrella)</b> = ${f3(p[0])} / ${f3(pu)} = ${f3(p[0] / pu)}<br>` +
        `<b>Bayes check:</b> ${f3(p[0] / pr)} × ${f3(pr)} / ${f3(pu)} = ${f3((p[0] / pr) * pr / pu)}`;
      if (last) s += `<br><span class="m">last draw: u = ${last.u.toFixed(4)} falls in [${last.lo.toFixed(3)}, ${last.hi.toFixed(3)}) so the cell is ${cellName[last.cell]}</span>`;
      q(ctl, '#out').innerHTML = s;
      const m = q(ctl, '#mode').value;
      q(ctl, '#cap').textContent = m === 'joint' ? 'Sampling from the joint table produces both x and y. The share of dots in each cell converges to the printed p. This is what a generative model can do.' : m === 'rain' ? 'Conditional generation: only the two "rain" cells are allowed and their probabilities are renormalised to sum to 1. The result is p(x | rain).' : 'This is the discriminative question: given x, how likely is each y? A classifier only ever answers this one. It never produces new umbrellas.';
      showVals(ctl, { w0: (v) => v, w1: (v) => v, w2: (v) => v, w3: (v) => v, rate: (v) => v });
    };
    const resetAll = () => { reset(); readout(); };
    wire(ctl, st, spawn, resetAll, () => { readout(); draw(); });
    ['w0', 'w1', 'w2', 'w3', 'mode'].forEach((id) => { q(ctl, '#' + id).oninput = resetAll; });
    o.onresize = draw; reset(); readout(); draw();
    let shown = 0;
    AI.loop(root, (dt) => {
      if (st.playing) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) spawn(); }
      let changed = dots.length > 0;
      for (const d of dots) d.t += dt;
      const rest = [];
      for (const d of dots) { if (d.t >= 0.9) { counts[d.cell]++; landed.push(d); changed = true; } else rest.push(d); }
      dots = rest; if (landed.length > 900) landed.splice(0, landed.length - 900);
      if (changed || shown !== counts.reduce((a, b) => a + b, 0)) { shown = counts.reduce((a, b) => a + b, 0); readout(); draw(); }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Maximum likelihood: the model q moves toward the data frequencies
     ===================================================================== */
  AI.labs['gen-fd-mle'] = function (root) {
    const tok = ['R', 'G', 'B', 'Y'];
    const { view, ctl } = AI.labShell(root, 'Maximum likelihood is minimising KL', 'illustrative: a four-token model q = softmax(z) is fitted to a tiny dataset by gradient ascent on the log-likelihood',
      `<canvas style="height:360px"></canvas>`,
      `${sl('c0', 'how many R in the data', 0, 20, 1, 5)}${sl('c1', 'how many G in the data', 0, 20, 1, 3)}${sl('c2', 'how many B in the data', 0, 20, 1, 2)}${sl('c3', 'how many Y in the data', 0, 20, 1, 0)}
       ${sl('z0', 'model logit z for R', -4, 4, 0.05, 0)}${sl('z1', 'model logit z for G', -4, 4, 0.05, 0)}${sl('z2', 'model logit z for B', -4, 4, 0.05, 0)}${sl('z3', 'model logit z for Y', -4, 4, 0.05, 0)}
       ${sl('speed', 'Fitting speed', 0.1, 3, 0.1, 0.6)}
       ${btns()}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let z, hist, tAcc, time, dirty = true;
    const reset = () => { z = [0, 0, 0, 0]; hist = [z.slice()]; tAcc = 0; time = 0; dirty = true; };
    const counts = () => [0, 1, 2, 3].map((i) => +q(ctl, '#c' + i).value);
    const phat = () => { const c = counts(), n = c.reduce((a, b) => a + b, 0); return n ? c.map((x) => x / n) : [0.25, 0.25, 0.25, 0.25]; };
    const soft = (zz) => { const m = Math.max(...zz), e = zz.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
    const stats = (zz) => {
      const c = counts(), n = Math.max(1, c.reduce((a, b) => a + b, 0)), qq = soft(zz), ph = phat();
      let ll = 0, H = 0, KL = 0;
      for (let i = 0; i < 4; i++) { if (c[i]) ll += c[i] * Math.log(qq[i]); if (ph[i] > 0) { H -= ph[i] * Math.log(ph[i]); KL += ph[i] * Math.log(ph[i] / qq[i]); } }
      return { q: qq, ll, n, nll: -ll / n, H, KL, c, ph };
    };
    const pushZ = () => { for (let i = 0; i < 4; i++) q(ctl, '#z' + i).value = Math.max(-4, Math.min(4, z[i])); };
    const stepFit = (h) => { const s = stats(z), n = s.ph; for (let i = 0; i < 4; i++) z[i] += 3 * h * (n[i] - s.q[i]); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const s = stats(z), mid = Math.round(W * 0.46);
      // left: bars
      const L = 30, R = mid - 10, T = 44, B = H - 52, bw = (R - L) / 4;
      txt(g, 'p̂ data (blue), q model (orange)', L, 16, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, B); g.lineTo(R, B); g.stroke();
      for (let i = 0; i < 4; i++) {
        const x = L + i * bw, hp = s.ph[i] * (B - T), hq = s.q[i] * (B - T);
        g.fillStyle = c.blue; g.globalAlpha = 0.8; g.fillRect(x + 6, B - hp, bw / 2 - 8, hp); g.fillStyle = c.orange; g.fillRect(x + bw / 2 + 2, B - hq, bw / 2 - 8, hq); g.globalAlpha = 1;
        txt(g, s.ph[i].toFixed(2), x + 6 + (bw / 2 - 8) / 2, B - hp - 4, c.blue, '11px sans-serif', 'center');
        txt(g, s.q[i].toFixed(2), x + bw / 2 + 2 + (bw / 2 - 8) / 2, B - hq - 4, c.orange, '11px sans-serif', 'center');
        txt(g, tok[i], x + bw / 2, B + 15, c.fg, 'bold 12px sans-serif', 'center'); txt(g, 'n=' + s.c[i], x + bw / 2, B + 29, c.muted, '10px sans-serif', 'center');
      }
      // right: curve of mean NLL
      const X0 = mid + 40, X1 = W - 14, Y0 = 44, Y1 = H - 44;
      txt(g, 'mean NLL (nats)', X0 - 20, 36, c.muted);
      const vals = hist.map((zz) => stats(zz).nll), ymax = Math.max(1.5, ...vals) * 1.05, ymin = Math.max(0, s.H - 0.25);
      const tm = Math.max(5, time * 1.05), X = (t) => X0 + (t / tm) * (X1 - X0), Y = (v) => Y1 - ((v - ymin) / (ymax - ymin)) * (Y1 - Y0);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(X0, Y0); g.lineTo(X0, Y1); g.lineTo(X1, Y1); g.stroke();
      txt(g, ymax.toFixed(2), X0 - 4, Y0 + 8, c.muted, '11px sans-serif', 'right'); txt(g, ymin.toFixed(2), X0 - 4, Y1, c.muted, '11px sans-serif', 'right');
      g.setLineDash([5, 4]); g.strokeStyle = c.green; g.beginPath(); g.moveTo(X0, Y(s.H)); g.lineTo(X1, Y(s.H)); g.stroke(); g.setLineDash([]);
      txt(g, `floor: entropy of the data H(p̂) = ${s.H.toFixed(4)}`, X1, Y(s.H) + 14, c.green, '11px sans-serif', 'right');
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath();
      vals.forEach((v, k) => { const t = hist.length > 1 ? (k / (hist.length - 1)) * time : 0; k ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); }); g.stroke(); g.lineWidth = 1;
      const cur = s.nll, cx = X(time);
      g.fillStyle = c.orange; g.beginPath(); g.arc(cx, Y(cur), 4, 0, 7); g.fill();
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(cx + 6, Y(cur)); g.lineTo(cx + 6, Y(s.H)); g.stroke();
      txt(g, `gap = KL = ${s.KL.toFixed(4)}`, cx + 10, (Y(cur) + Y(s.H)) / 2 + 4, c.red, '11px sans-serif');
      const e = (v) => (v < 1e-3 ? v.toExponential(3) : v.toFixed(7));
      let lik = 1; for (let i = 0; i < 4; i++) lik *= Math.pow(s.q[i], s.c[i]);
      q(ctl, '#out').innerHTML = `log-likelihood  Σ n<sub>i</sub> ln q<sub>i</sub> = <b>${s.ll.toFixed(4)}</b> over ${s.n} items<br>likelihood = e<sup>${s.ll.toFixed(4)}</sup> = ${e(lik)}<br>mean NLL = −(1/n) log-likelihood = <b>${s.nll.toFixed(4)}</b><br>entropy of data H(p̂) = ${s.H.toFixed(4)}<br>KL(p̂ ‖ q) = ${s.KL.toFixed(4)}<br>H(p̂) + KL = ${(s.H + s.KL).toFixed(4)} (equals mean NLL)`;
      q(ctl, '#cap').textContent = s.KL < 0.002 ? 'q has reached the empirical frequencies: KL is about 0 and the mean NLL has hit its floor H(p̂). The maximum-likelihood fit and the minimum-KL fit are the same point.' : (s.c[3] === 0 && s.q[3] < 0.05 ? 'Y never appears in the data, so q(Y) keeps being pushed toward 0. Maximum likelihood trusts the data completely: a future Y would get probability near 0.' : 'Each moment the model moves up the gradient of the log-likelihood. Mean NLL falls toward the floor H(p̂); the part still above the floor is exactly KL(p̂ ‖ q).');
      showVals(ctl, { c0: (v) => v, c1: (v) => v, c2: (v) => v, c3: (v) => v, z0: (v) => (+v).toFixed(2), z1: (v) => (+v).toFixed(2), z2: (v) => (+v).toFixed(2), z3: (v) => (+v).toFixed(2), speed: (v) => v });
    };
    const manual = () => { for (let i = 0; i < 4; i++) z[i] = +q(ctl, '#z' + i).value; hist = [z.slice()]; time = 0; dirty = true; };
    wire(ctl, st, () => { stepFit(0.5); time += 0.5; hist.push(z.slice()); pushZ(); dirty = true; }, () => { reset(); pushZ(); }, draw);
    for (let i = 0; i < 4; i++) { q(ctl, '#z' + i).oninput = () => { st.playing = false; q(ctl, '#play').textContent = '▶ Play'; manual(); }; q(ctl, '#c' + i).oninput = () => { dirty = true; }; }
    q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const sp = +q(ctl, '#speed').value; stepFit(dt * sp); time += dt; tAcc += dt;
        if (tAcc >= 0.15) { tAcc = 0; hist.push(z.slice()); if (hist.length > 400) hist.shift(); }
        pushZ(); dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Inverse-transform sampling from a small table
     ===================================================================== */
  AI.labs['gen-fd-inverse'] = function (root) {
    const tok = ['R', 'G', 'B', 'Y'];
    const { view, ctl } = AI.labShell(root, 'Sampling from a table: inverse transform', 'illustrative: a seeded random number u in [0, 1) is dropped onto the cumulative table; the interval it lands in is the sampled token',
      `<canvas style="height:340px"></canvas>`,
      `${sl('w0', 'weight of R', 0, 10, 0.5, 5)}${sl('w1', 'weight of G', 0, 10, 0.5, 3)}${sl('w2', 'weight of B', 0, 10, 0.5, 2)}${sl('w3', 'weight of Y', 0, 10, 0.5, 0)}
       ${sl('rate', 'Draws per second', 0.1, 6, 0.1, 0.4)}
       ${btns()}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let rs, counts, n, cur, t, hold, dirty = true;
    const probs = () => norm([0, 1, 2, 3].map((i) => +q(ctl, '#w' + i).value));
    const newDraw = () => { const u = rs(); const p = probs(); cur = { u, j: pickCum(p, u), p }; t = 0; hold = false; };
    const reset = () => { rs = AI.rng(8); counts = [0, 0, 0, 0]; n = 0; newDraw(); dirty = true; };
    const land = () => { if (!hold) { hold = true; counts[cur.j]++; n++; } };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const cc = [c.red, c.green, c.blue, c.yellow], p = cur.p, L = 40, R = W - 20, sy = 92, sh = 34;
      txt(g, 'the unit interval [0, 1), cut by the cumulative probabilities', L, 18, c.muted);
      let cum = 0; const cums = [];
      for (let i = 0; i < 4; i++) {
        const x0 = L + cum * (R - L), x1 = L + (cum + p[i]) * (R - L); cum += p[i]; cums.push(cum);
        const hit = hold && cur.j === i;
        g.fillStyle = cc[i]; g.globalAlpha = hit ? 0.95 : 0.4; g.fillRect(x0, sy, Math.max(0, x1 - x0), sh); g.globalAlpha = 1;
        if (x1 - x0 > 28) txt(g, tok[i] + ' ' + p[i].toFixed(2), (x0 + x1) / 2, sy + 22, c.fg, 'bold 12px sans-serif', 'center');
        g.strokeStyle = c.fg; g.beginPath(); g.moveTo(x1, sy - 4); g.lineTo(x1, sy + sh + 4); g.stroke();
        if (p[i] > 0) txt(g, cum.toFixed(2), x1, sy + sh + 18, c.muted, '11px sans-serif', 'center');
      }
      txt(g, '0', L, sy + sh + 18, c.muted, '11px sans-serif', 'center');
      // dart
      const period = 1 / +q(ctl, '#rate').value, drop = Math.min(0.9, period * 0.5), e = ease(clamp01(t / drop)), ux = L + cur.u * (R - L), y = 40 + (sy - 6 - 40) * e;
      g.fillStyle = c.fg; g.beginPath(); g.moveTo(ux, y + 10); g.lineTo(ux - 6, y - 4); g.lineTo(ux + 6, y - 4); g.closePath(); g.fill();
      txt(g, 'u = ' + cur.u.toFixed(4), Math.min(W - 70, Math.max(L + 40, ux)), 36, c.fg, 'bold 12px sans-serif', 'center');
      // histogram
      const hy0 = 170, hy1 = H - 24, mx = Math.max(0.5, ...counts.map((v) => v / Math.max(1, n)), ...p), bw = (R - L) / 4;
      txt(g, `draws so far (n = ${n}); white tick = table value`, L, hy0 - 6, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, hy1); g.lineTo(R, hy1); g.stroke();
      for (let i = 0; i < 4; i++) {
        const fr = counts[i] / Math.max(1, n), h = fr / mx * (hy1 - hy0 - 14), x = L + i * bw;
        g.fillStyle = cc[i]; g.globalAlpha = 0.85; g.fillRect(x + 12, hy1 - h, bw - 24, h); g.globalAlpha = 1;
        g.strokeStyle = c.fg; const ty = hy1 - p[i] / mx * (hy1 - hy0 - 14); g.beginPath(); g.moveTo(x + 8, ty); g.lineTo(x + bw - 8, ty); g.stroke();
        txt(g, `${tok[i]}: ${counts[i]}  (${fr.toFixed(2)})`, x + bw / 2, hy1 + 14, c.fg, '11px sans-serif', 'center');
      }
      const lo = cur.j ? cums[cur.j - 1] : 0;
      q(ctl, '#out').innerHTML = `cumulative table: ${tok.map((k, i) => `${k} ${cums[i].toFixed(2)}`).join(' | ')}<br>` + (t >= drop ? `u = ${cur.u.toFixed(4)} satisfies ${lo.toFixed(2)} ≤ u &lt; ${cums[cur.j].toFixed(2)}, so the token is <b>${tok[cur.j]}</b>` : `u = ${cur.u.toFixed(4)}: finding the first cumulative value larger than u ...`);
      q(ctl, '#cap').textContent = 'Wider intervals catch more of the random numbers, so they are drawn more often. A token with weight 0 has an empty interval and can never be drawn. Set Y above 0 and watch a new interval appear.';
      showVals(ctl, { w0: (v) => v, w1: (v) => v, w2: (v) => v, w3: (v) => v, rate: (v) => v });
    };
    const adv = () => { land(); newDraw(); };
    wire(ctl, st, adv, reset, draw);
    for (let i = 0; i < 4; i++) q(ctl, '#w' + i).oninput = () => { reset(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        t += dt; const period = 1 / +q(ctl, '#rate').value, drop = Math.min(0.9, period * 0.5);
        if (t >= drop) land();
        if (t >= period) newDraw();
        dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. The chain rule unrolling one token at a time (shared toy bigram table)
     ===================================================================== */
  const V = ['the', 'cat', 'sat', 'mat'];
  const PREV = ['START', 'the', 'cat', 'sat', 'mat'];
  const TABLE = { START: [0.6, 0.2, 0.1, 0.1], the: [0.05, 0.45, 0.05, 0.45], cat: [0.15, 0.05, 0.7, 0.1], sat: [0.5, 0.1, 0.05, 0.35], mat: [0.3, 0.1, 0.5, 0.1] };

  AI.labs['gen-fd-chain'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The chain rule: one token at a time', 'illustrative: each step reads one row of a toy table (the next-token distribution given the previous token), draws u, and appends a token',
      `<canvas style="height:400px"></canvas>`,
      `${sl('len', 'Sequence length L', 1, 8, 1, 3)}
       ${sl('speed', 'Speed', 0.25, 3, 0.05, 0.7)}
       <label>Edit the row for previous token <select id="row"><option>START</option><option>the</option><option>cat</option><option>sat</option><option>mat</option></select></label>
       ${[0, 1, 2, 3].map((i) => sl('r' + i, 'weight of next = ' + V[i], 0, 1, 0.01, 0)).join('')}
       ${btns()}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    const W = {}; for (const k of PREV) W[k] = TABLE[k].slice();
    let rs, seq, fac, k, tt, curU, committed, fin, holdT, recent;
    const rowP = (prev) => norm(W[prev]);
    const newSeq = () => { seq = []; fac = []; k = 0; tt = 0; committed = false; fin = false; holdT = 0; curU = rs(); };
    const reset = () => { rs = AI.rng(4); recent = []; newSeq(); };
    const D = () => 3.4 / +q(ctl, '#speed').value;
    const prevTok = () => (k ? seq[k - 1] : 'START');
    const commit = () => {
      const p = rowP(prevTok()), j = pickCum(p, curU); seq.push(V[j]); fac.push(p[j]); committed = true;
    };
    const nextStep = () => { if (!committed) commit(); k++; tt = 0; committed = false; const L = +q(ctl, '#len').value; if (k >= L) { fin = true; holdT = 0; } else curU = rs(); };
    const finish = () => { const pr = fac.reduce((a, b) => a * b, 1); recent.unshift(`${seq.join(' ')}  p = ${fac.map((x) => x.toFixed(3)).join(' × ')} = ${pr.toFixed(4)}`); if (recent.length > 4) recent.pop(); newSeq(); };
    const advance = (dt) => {
      if (fin) { holdT += dt; if (holdT > 3) finish(); return; }
      tt += dt; const d = D();
      if (!committed && tt >= 0.7 * d) commit();
      if (tt >= d) nextStep();
    };
    const syncRow = () => { const p = rowP(q(ctl, '#row').value); for (let i = 0; i < 4; i++) q(ctl, '#r' + i).value = p[i].toFixed(2); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, Wd = o.w, H = o.h; g.clearRect(0, 0, Wd, H);
      const L = +q(ctl, '#len').value, cc = [c.blue, c.teal, c.orange, c.pink], d = D();
      // sequence slots
      const n = L + 1, bw = Math.min(92, (Wd - 24) / n - 6), sx = (i) => 12 + i * (bw + 6);
      txt(g, 'the sequence so far (each factor is one conditional probability)', 12, 16, c.muted);
      for (let i = 0; i < n; i++) {
        const label = i === 0 ? 'START' : seq[i - 1] || '', active = !fin && i === k + 1;
        g.strokeStyle = active ? c.fg : c.border; g.lineWidth = active ? 2 : 1; g.strokeRect(sx(i), 26, bw, 30); g.lineWidth = 1;
        if (label) txt(g, label, sx(i) + bw / 2, 46, i === 0 ? c.muted : c.fg, 'bold 13px sans-serif', 'center');
        else if (active) txt(g, '?', sx(i) + bw / 2, 46, c.muted, '13px sans-serif', 'center');
        if (i > 0 && fac[i - 1] !== undefined) txt(g, '× ' + fac[i - 1].toFixed(3), sx(i) + bw / 2, 72, c.green, '11px sans-serif', 'center');
      }
      // current step
      const step = Math.min(k, L - 1), prev = fin ? seq[L - 2 >= 0 ? L - 2 : 0] : prevTok();
      void prev;
      if (!fin) {
        const p = rowP(prevTok()), top = 128, X0 = 40, X1 = Wd - 24;
        txt(g, `step ${step + 1} of ${L}:  p(next | ${prevTok()})`, 12, 94, c.fg, 'bold 12px sans-serif');
        let cum = 0;
        for (let i = 0; i < 4; i++) {
          const x0 = X0 + cum * (X1 - X0), x1 = X0 + (cum + p[i]) * (X1 - X0); cum += p[i];
          const hit = committed && seq[k] === V[i];
          g.fillStyle = cc[i]; g.globalAlpha = hit ? 0.95 : 0.45; g.fillRect(x0, top, Math.max(0, x1 - x0), 34); g.globalAlpha = 1;
          g.strokeStyle = c.fg; g.strokeRect(x0, top, Math.max(0, x1 - x0), 34);
          if (x1 - x0 > 30) txt(g, V[i], (x0 + x1) / 2, top + 22, c.fg, 'bold 12px sans-serif', 'center');
          const bh = p[i] * 120, bx = X0 + i * ((X1 - X0) / 4) + 16, by = 250;
          g.fillStyle = cc[i]; g.globalAlpha = hit ? 0.95 : 0.6; g.fillRect(bx, by - bh, (X1 - X0) / 4 - 32, bh); g.globalAlpha = 1;
          txt(g, p[i].toFixed(3), bx + ((X1 - X0) / 4 - 32) / 2, by - bh - 4, c.fg, '11px sans-serif', 'center');
          txt(g, V[i], bx + ((X1 - X0) / 4 - 32) / 2, by + 15, c.muted, '12px sans-serif', 'center');
        }
        // dart
        const ph = clamp01((tt - 0.3 * d) / (0.4 * d)), ux = X0 + curU * (X1 - X0), y = top - 26 + 22 * ease(ph);
        if (tt >= 0.3 * d) { g.fillStyle = c.fg; g.beginPath(); g.moveTo(ux, y + 14); g.lineTo(ux - 6, y); g.lineTo(ux + 6, y); g.closePath(); g.fill(); txt(g, 'u = ' + curU.toFixed(3), Math.min(ux + 9, Wd - 70), y + 8, c.fg, '11px sans-serif', 'left'); }
      } else {
        const pr = fac.reduce((a, b) => a * b, 1);
        txt(g, `finished: "${seq.join(' ')}"`, 12, 120, c.fg, 'bold 14px sans-serif');
        txt(g, `p = ${fac.map((x) => x.toFixed(3)).join(' × ')} = ${pr.toFixed(4)}`, 12, 146, c.green, '13px sans-serif');
        txt(g, `${L} model calls were needed, one per token, and each had to wait for the one before.`, 12, 172, c.muted, '12px sans-serif');
      }
      txt(g, `model calls so far: ${fin ? L : k + (committed ? 1 : 0)}      possible sequences of this length: 4^${L} = ${Math.pow(4, L).toLocaleString()}`, 12, H - 12, c.muted, '12px sans-serif');
      q(ctl, '#out').innerHTML = recent.length ? '<b>recent samples</b><br>' + recent.join('<br>') : 'Samples will be listed here as they finish.';
      q(ctl, '#cap').textContent = !fin ? (tt < 0.3 * d ? 'Step 1: look up the row for the previous token.' : (tt < 0.7 * d ? 'Step 2: a random number u is dropped on the cumulative strip.' : 'Step 3: the interval that holds u is the next token; it is appended and the next step starts from it.')) : 'The sequence probability is the product of the conditional probabilities along the way (the chain rule).';
      showVals(ctl, { len: (v) => v, speed: (v) => v, r0: (v) => v, r1: (v) => v, r2: (v) => v, r3: (v) => v });
    };
    wire(ctl, st, () => { if (fin) finish(); else nextStep(); }, () => { reset(); }, draw);
    q(ctl, '#len').oninput = () => { newSeq(); };
    q(ctl, '#row').onchange = () => { syncRow(); };
    for (let i = 0; i < 4; i++) q(ctl, '#r' + i).oninput = () => { const arr = [0, 1, 2, 3].map((j) => +q(ctl, '#r' + j).value); W[q(ctl, '#row').value] = arr; };
    o.onresize = draw; reset(); syncRow(); draw();
    AI.loop(root, (dt) => { if (st.playing) advance(dt); draw(); }, { fps: 30 });
  };

  /* score one sequence with the chain rule, and check that all 64 sum to 1 */
  AI.labs['gen-fd-chain-score'] = function (root) {
    const opts = V.map((v) => `<option>${v}</option>`).join('');
    const { view, ctl } = AI.labShell(root, 'Score a sequence with the chain rule', 'illustrative: the same toy table as above; choose three tokens and multiply the three conditionals',
      `<div id="tbl" style="overflow-x:auto"></div>`,
      `<label>x₁ <select id="a">${opts}</select></label><label>x₂ <select id="b">${opts}</select></label><label>x₃ <select id="c">${opts}</select></label>
       <div class="row"><button class="btn" id="best">Find the most probable sequence</button><button class="btn" id="rand">Random choice</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const rs = AI.rng(31);
    const all = []; for (const a of V) for (const b of V) for (const c of V) { const p = TABLE.START[V.indexOf(a)] * TABLE[a][V.indexOf(b)] * TABLE[b][V.indexOf(c)]; all.push({ s: [a, b, c], p }); }
    const total = all.reduce((s, x) => s + x.p, 0);
    const html = '<table class="tbl"><thead><tr><th>previous ↓ / next →</th>' + V.map((v) => `<th>${v}</th>`).join('') + '</tr></thead><tbody>' +
      PREV.map((r) => `<tr><td><b>${r}</b></td>${TABLE[r].map((x) => `<td>${x.toFixed(2)}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
    q(view, '#tbl').innerHTML = html;
    const update = () => {
      const s = ['a', 'b', 'c'].map((i) => q(ctl, '#' + i).value);
      const f1 = TABLE.START[V.indexOf(s[0])], f2 = TABLE[s[0]][V.indexOf(s[1])], f3_ = TABLE[s[1]][V.indexOf(s[2])], p = f1 * f2 * f3_;
      q(ctl, '#out').innerHTML = `p(${s[0]}) = ${f1.toFixed(2)}<br>p(${s[1]} | ${s[0]}) = ${f2.toFixed(2)}<br>p(${s[2]} | ${s[1]}) = ${f3_.toFixed(2)}<br>product = <b>${p.toFixed(4)}</b> &nbsp; (−log₂ = ${(-Math.log2(p)).toFixed(3)} bits)<br><span class="m">sum over all 4³ = 64 sequences = ${total.toFixed(6)}</span>`;
      q(ctl, '#cap').textContent = 'The 64 probabilities add to 1, so the product of conditionals really is a probability distribution over whole sequences. Here the toy table conditions only on the previous token; the chain rule itself allows conditioning on the whole history.';
    };
    ['a', 'b', 'c'].forEach((i) => { q(ctl, '#' + i).onchange = update; });
    q(ctl, '#best').onclick = () => { const b = all.reduce((m, x) => (x.p > m.p ? x : m)); ['a', 'b', 'c'].forEach((i, k) => { q(ctl, '#' + i).value = b.s[k]; }); update(); };
    q(ctl, '#rand').onclick = () => { ['a', 'b', 'c'].forEach((i) => { q(ctl, '#' + i).value = V[Math.floor(rs() * 4)]; }); update(); };
    q(ctl, '#a').value = 'the'; q(ctl, '#b').value = 'cat'; q(ctl, '#c').value = 'sat'; update();
  };

  /* =====================================================================
     5. Cost of one sample: how many network evaluations each family needs
     ===================================================================== */
  AI.labs['gen-fd-race'] = function (root) {
    const fam = [
      { n: 'autoregressive', k: 'L', what: 'one next-token distribution per token', lik: 'exact (chain rule)', col: 'blue' },
      { n: 'latent variable (VAE)', k: 1, what: 'decode one latent z into x', lik: 'lower bound (ELBO)', col: 'teal' },
      { n: 'adversarial (GAN)', k: 1, what: 'generator maps noise z to x', lik: 'none (implicit density)', col: 'orange' },
      { n: 'flow', k: 1, what: 'invertible map from z to x', lik: 'exact (change of variables)', col: 'green' },
      { n: 'diffusion / score', k: 'T', what: 'one denoising step per level', lik: 'bound or ODE-based estimate', col: 'purple' }];
    const { view, ctl } = AI.labShell(root, 'How many network evaluations does one sample need?', 'illustrative: counts of evaluations only; one evaluation of different families costs different amounts of compute',
      `<canvas style="height:300px"></canvas>`,
      `${sl('L', 'Autoregressive: tokens L', 1, 200, 1, 40)}${sl('T', 'Diffusion: denoising steps T', 1, 200, 1, 30)}
       ${sl('rate', 'Evaluations per second (the same clock for all)', 2, 60, 1, 12)}
       ${btns()}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let time, holdT, done;
    const reset = () => { time = 0; holdT = 0; done = false; };
    const need = () => fam.map((f) => (f.k === 'L' ? +q(ctl, '#L').value : f.k === 'T' ? +q(ctl, '#T').value : 1));
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const N = need(), rate = +q(ctl, '#rate').value, ev = time * rate, left = 150, right = W - 14, rh = (H - 34) / 5;
      txt(g, 'each row fills as its sampler runs; a full bar is one finished sample', 12, 16, c.muted);
      fam.forEach((f, i) => {
        const y = 28 + i * rh, done_ = Math.min(N[i], Math.floor(ev)), frac = Math.min(1, ev / N[i]);
        txt(g, f.n, 12, y + rh * 0.45, c.fg, 'bold 12px sans-serif'); txt(g, f.lik, 12, y + rh * 0.45 + 14, c.muted, '10px sans-serif');
        g.strokeStyle = c.border; g.strokeRect(left, y + 6, right - left, rh - 16);
        g.fillStyle = c[f.col]; g.globalAlpha = 0.75; g.fillRect(left, y + 6, (right - left) * frac, rh - 16); g.globalAlpha = 1;
        txt(g, `${done_} / ${N[i]} evaluation${N[i] > 1 ? 's' : ''}${frac >= 1 ? '  ✓ sample ready' : ''}`, left + 8, y + rh * 0.5 + 4, c.fg, '12px sans-serif');
      });
      const N0 = N;
      q(ctl, '#out').innerHTML = fam.map((f, i) => `${f.n}: ${N0[i]} evaluation${N0[i] > 1 ? 's' : ''} per sample (${f.what})`).join('<br>');
      q(ctl, '#cap').textContent = 'Autoregressive cost grows with the length of the output and diffusion cost with the number of steps, while a single-pass sampler needs one evaluation. This is one axis only: it says nothing about sample quality, coverage of the data, or training difficulty (see the lesson text).';
      showVals(ctl, { L: (v) => v, T: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => { time += 1 / +q(ctl, '#rate').value; }, reset, draw);
    ['L', 'T', 'rate'].forEach((i) => { q(ctl, '#' + i).oninput = () => { if (i !== 'rate') reset(); draw(); }; });
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const mx = Math.max(...need()) / +q(ctl, '#rate').value;
        if (time < mx) time += dt; else { holdT += dt; if (holdT > 2.5) reset(); }
        draw();
      }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. A one-dimensional flow: noise turned into data by a monotone map
     ===================================================================== */
  AI.labs['gen-fd-flow1d'] = function (root) {
    const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
    const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
    const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const { view, ctl } = AI.labShell(root, 'A flow: noise pushed onto a data distribution', 'illustrative: x = F⁻¹(Φ(z)) is a monotone map that turns Gaussian noise z into a two-hump distribution',
      `<canvas style="height:360px"></canvas>`,
      `${sl('w', 'Weight of the left hump', 0.1, 0.9, 0.05, 0.5)}${sl('d', 'Distance between the humps', 1, 5, 0.1, 3)}${sl('z0', 'Follow one noise value z₀', -2.5, 2.5, 0.05, 1)}
       ${sl('speed', 'Speed (seconds per full move: 8 / speed)', 0.25, 3, 0.05, 0.6)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    const rz = AI.rng(5), zs = []; for (let i = 0; i < 600; i++) zs.push(AI.randn(rz));
    const SD = 0.6;
    let w, m1, m2, xs, s = 0, dir = 1, dirty = true;
    const F = (x) => w * Phi((x - m1) / SD) + (1 - w) * Phi((x - m2) / SD);
    const Finv = (p) => { let lo = -12, hi = 12; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (F(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
    const T = (z) => Finv(Math.min(1 - 1e-12, Math.max(1e-12, Phi(z))));
    const pdfX = (x) => w * npdf(x, m1, SD) + (1 - w) * npdf(x, m2, SD);
    const params = () => { w = +q(ctl, '#w').value; const d = +q(ctl, '#d').value; m1 = -d / 2; m2 = d / 2; xs = zs.map(T); dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 20, R = W - 20, XMIN = -5, XMAX = 5, X = (v) => L + (v - XMIN) / (XMAX - XMIN) * (R - L);
      const top = 40, base = H - 120, e = ease(s);
      txt(g, `histogram of 600 points at t = ${s.toFixed(2)}: t = 0 is the noise, t = 1 is the data`, L, 16, c.muted);
      const pos = zs.map((z, i) => (1 - e) * z + e * xs[i]);
      const nb = 50, bins = new Array(nb).fill(0), bw = (XMAX - XMIN) / nb;
      pos.forEach((v) => { const b = Math.floor((v - XMIN) / bw); if (b >= 0 && b < nb) bins[b]++; });
      const sc = (base - top) / (0.5 * 600 * bw);
      g.fillStyle = c.blue; g.globalAlpha = 0.65; bins.forEach((n, b) => { const h = n * sc / 1; g.fillRect(X(XMIN + b * bw) + 1, base - Math.min(h, base - top), X(XMIN + bw) - X(XMIN) - 2, Math.min(h, base - top)); }); g.globalAlpha = 1;
      // target density at t=1, noise density at t=0
      g.lineWidth = 2; g.strokeStyle = c.orange; g.beginPath();
      for (let i = 0; i <= 200; i++) { const x = XMIN + (XMAX - XMIN) * i / 200, y = ((1 - e) * npdf(x, 0, 1) + e * pdfX(x)) * 600 * bw * sc; i ? g.lineTo(X(x), base - y) : g.moveTo(X(x), base - y); }
      g.stroke(); g.lineWidth = 1;
      txt(g, 'orange: blend of the noise density and the data density', R, top + 4, c.orange, '11px sans-serif', 'right');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, base); g.lineTo(R, base); g.stroke();
      for (let v = -4; v <= 4; v += 2) txt(g, v, X(v), base + 14, c.muted, '11px sans-serif', 'center');
      // trace of the followed point and the map T
      const z0 = +q(ctl, '#z0').value, x0 = T(z0), pz = (1 - e) * z0 + e * x0;
      const my = base + 34, mh = 70;
      txt(g, 'the map x = T(z) (a monotone curve) and the followed point', L, my - 6, c.muted, '11px sans-serif');
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath();
      for (let i = 0; i <= 120; i++) { const z = -3 + 6 * i / 120, x = Math.max(-5, Math.min(5, T(z))), px = L + (z + 3) / 6 * (R - L), py = my + mh - (x + 5) / 10 * mh; i ? g.lineTo(px, py) : g.moveTo(px, py); } g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.red; g.beginPath(); g.arc(L + (z0 + 3) / 6 * (R - L), my + mh - (Math.max(-5, Math.min(5, x0)) + 5) / 10 * mh, 5, 0, 7); g.fill();
      g.strokeStyle = c.red; g.beginPath(); g.moveTo(X(pz), base - 2); g.lineTo(X(pz), base - 22); g.stroke(); txt(g, 'followed point', X(pz), base - 26, c.red, '11px sans-serif', 'center');
      // change of variables
      const h = 1e-3, slope = (T(z0 + h) - T(z0 - h)) / (2 * h), pzv = npdf(z0, 0, 1), px = pzv / slope;
      q(ctl, '#out').innerHTML = `followed point: z₀ = ${z0.toFixed(2)} → x₀ = T(z₀) = ${x0.toFixed(3)}<br>noise density p<sub>Z</sub>(z₀) = ${pzv.toFixed(4)}<br>slope dx/dz = ${slope.toFixed(4)}<br><b>p<sub>X</sub>(x₀) = p<sub>Z</sub>(z₀) / slope = ${px.toFixed(4)}</b><br>data density at x₀, computed directly: ${pdfX(x0).toFixed(4)}`;
      q(ctl, '#cap').textContent = 'The map stretches where the data are sparse and squeezes where they are dense; dividing the noise density by the slope gives the exact data density (change of variables). Both numbers agree because T was built from the data distribution itself, so this is a flow with exact likelihood. Real flows learn T with neural networks.';
      showVals(ctl, { w: (v) => v, d: (v) => v, z0: (v) => (+v).toFixed(2), speed: (v) => v });
    };
    const reset = () => { s = 0; dir = 1; dirty = true; };
    wire(ctl, st, () => { s = Math.min(1, s + 0.1); dirty = true; }, reset, draw);
    ['w', 'd', 'z0'].forEach((i) => { q(ctl, '#' + i).oninput = () => { if (i !== 'z0') params(); dirty = true; }; });
    q(ctl, '#speed').oninput = () => { dirty = true; };
    o.onresize = draw; params(); draw();
    let hold = 0;
    AI.loop(root, (dt) => {
      if (st.playing) {
        if (hold > 0) hold -= dt; else { s += dir * dt * +q(ctl, '#speed').value / 8; if (s >= 1) { s = 1; dir = -1; hold = 1.5; } if (s <= 0) { s = 0; dir = 1; hold = 1.5; } dirty = true; }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. Patches and tokens: a toy image tokeniser
     ===================================================================== */
  AI.labs['gen-fd-tokens'] = function (root) {
    const S = 32, img = new Float32Array(S * S);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      let v = 50 + 120 * (x / (S - 1));
      const dx = x - 11, dy = y - 12; if (dx * dx + dy * dy < 64) v = 235;
      if (Math.abs((x - 20) - (y - 6)) < 2 && x > 17) v = 20;
      if (y > 26) v = 90 + 60 * Math.sin(x / 2);
      img[y * S + x] = v;
    }
    const { view, ctl } = AI.labShell(root, 'A toy tokeniser for a 32×32 grey image', 'illustrative: each patch is replaced by one number (its mean), rounded to K levels; a real tokeniser learns a codebook over patch content instead',
      `<canvas style="height:330px"></canvas>`,
      `<label>Patch size <select id="ps"><option>2</option><option selected>4</option><option>8</option><option>16</option></select></label>
       <label>Vocabulary K <select id="K"><option value="2">2 levels</option><option value="4">4 levels</option><option value="8">8 levels</option><option value="16" selected>16 levels</option><option value="64">64 levels</option><option value="0">continuous (no rounding)</option></select></label>
       ${sl('rate', 'Patches tokenised per second', 1, 60, 1, 6)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="all">Show all</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let k = 0, acc = 0, tick = AI.ticker(), dirty = true, cache = null;
    const cfg = () => ({ ps: +q(ctl, '#ps').value, K: +q(ctl, '#K').value });
    const tokenise = () => {
      const { ps, K } = cfg(), g = S / ps, tokens = [], rec = new Float32Array(S * S);
      let se = 0;
      for (let by = 0; by < g; by++) for (let bx = 0; bx < g; bx++) {
        let m = 0; for (let y = 0; y < ps; y++) for (let x = 0; x < ps; x++) m += img[(by * ps + y) * S + bx * ps + x]; m /= ps * ps;
        const id = K ? Math.round(m / 255 * (K - 1)) : m, val = K ? id / (K - 1) * 255 : m;
        tokens.push(id);
        for (let y = 0; y < ps; y++) for (let x = 0; x < ps; x++) { const i = (by * ps + y) * S + bx * ps + x; rec[i] = val; se += (img[i] - val) ** 2; }
      }
      return { g, tokens, rec, mse: se / (S * S), ps, K };
    };
    const draw = () => {
      if (!cache) cache = tokenise();
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, tk = cache; g.clearRect(0, 0, W, H);
      const sz = Math.min((W - 40) / 2, H - 70, 240), x1 = 12, x2 = W - 12 - sz, y0 = 32, cell = sz / S;
      const paint = (arr, ox, upto) => { for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { const p = Math.floor(y / tk.ps) * tk.g + Math.floor(x / tk.ps); if (upto !== undefined && p >= upto) { g.fillStyle = c.bg2; } else { const v = Math.round(arr[y * S + x]); g.fillStyle = `rgb(${v},${v},${v})`; } g.fillRect(ox + x * cell, y0 + y * cell, cell + 0.6, cell + 0.6); } };
      txt(g, 'original', x1, 22, c.muted); txt(g, `reconstruction from ${Math.min(k, tk.tokens.length)} of ${tk.tokens.length} tokens`, x2, 22, c.muted);
      paint(img, x1); paint(tk.rec, x2, k);
      g.strokeStyle = c.blue; g.lineWidth = 1; g.globalAlpha = 0.5;
      for (let i = 0; i <= tk.g; i++) { g.beginPath(); g.moveTo(x1 + i * tk.ps * cell, y0); g.lineTo(x1 + i * tk.ps * cell, y0 + sz); g.moveTo(x1, y0 + i * tk.ps * cell); g.lineTo(x1 + sz, y0 + i * tk.ps * cell); g.stroke(); }
      g.globalAlpha = 1;
      if (k < tk.tokens.length) { const bx = k % tk.g, by = Math.floor(k / tk.g); g.strokeStyle = c.red; g.lineWidth = 3; g.strokeRect(x1 + bx * tk.ps * cell, y0 + by * tk.ps * cell, tk.ps * cell, tk.ps * cell); g.lineWidth = 1; }
      // token strip in the middle
      const mx0 = x1 + sz + 12, mx1 = x2 - 12;
      if (mx1 - mx0 > 30) { txt(g, 'token ids', (mx0 + mx1) / 2, 22, c.muted, '11px sans-serif', 'center'); const shown = tk.tokens.slice(Math.max(0, Math.min(k, tk.tokens.length) - 12), Math.min(k, tk.tokens.length)); shown.forEach((id, i) => txt(g, tk.K ? id : id.toFixed(0), (mx0 + mx1) / 2, 40 + i * 16, c.fg, '12px monospace', 'center')); }
      const L = tk.tokens.length, bits = tk.K ? Math.log2(tk.K) : 16, total = L * bits, raw = S * S * 8;
      q(ctl, '#out').innerHTML = `grid ${tk.g} × ${tk.g} = <b>${L} tokens</b> (sequence length L)<br>${tk.K ? `each token has ${Math.log2(tk.K).toFixed(0)} bits (log₂ ${tk.K})` : 'each value stored as a 16-bit number (assumed)'}: total ${total} bits<br>raw image: 32 × 32 × 8 = ${raw} bits → compression ${(raw / total).toFixed(1)}×<br>self-attention pairs L² = ${L * L}<br>reconstruction error (mean squared, 0..255 scale) = <b>${tk.mse.toFixed(1)}</b>`;
      q(ctl, '#cap').textContent = 'Bigger patches give a shorter sequence (cheaper for a model that reads every token) but throw away more detail. Fewer levels K give fewer bits per token and more rounding error. Making the sequence short and the picture sharp at once is the job of a learned tokeniser.';
      showVals(ctl, { rate: (v) => v });
    };
    const redo = () => { cache = null; k = 0; acc = 0; dirty = true; };
    wire(ctl, st, () => { cache = cache || tokenise(); k = Math.min(cache.tokens.length, k + 1); }, redo, draw);
    q(ctl, '#ps').onchange = redo; q(ctl, '#K').onchange = redo;
    q(ctl, '#all').onclick = () => { cache = cache || tokenise(); k = cache.tokens.length; dirty = true; };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      cache = cache || tokenise();
      if (st.playing && k < cache.tokens.length) { const n = tick(dt, +q(ctl, '#rate').value); if (n) { k = Math.min(cache.tokens.length, k + n); dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
