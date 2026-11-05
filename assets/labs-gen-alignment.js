/* Generative AI Module 4 labs: alignment and post-training.
   gen-al-sft, gen-al-bt, gen-al-btcurve, gen-al-kl, gen-al-dpo, gen-al-group, gen-al-refusal, gen-al-redteam-k.
   All simulations are illustrative toys (3-4 answers, a few tokens), use seeded random numbers (AI.rng), run on dt, start slowly,
   and have speed / pause / step / reset. No lab claims anything about a real model. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const num = (id, label, val, step) => `<label>${label} <input type="number" id="${id}" step="${step || 0.1}" value="${val}" style="width:5em"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const sg = (x) => 1 / (1 + Math.exp(-x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  const klf = (p, r) => p.reduce((s, v, i) => s + (v > 0 ? v * Math.log(v / r[i]) : 0), 0);
  const Phi = (z) => { const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2); return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y); };
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const rd = (ctl, s) => { q(ctl, '#out').textContent = s; };

  /* grouped bars for a 3/4-answer distribution: reference outline, policy fill, optional target tick */
  function distBars(g, c, x, y, w, h, labels, ref, pol, star, title) {
    txt(g, title, x, y - 8, c.muted, '11px sans-serif');
    g.strokeStyle = c.border; g.beginPath(); g.moveTo(x, y + h); g.lineTo(x + w, y + h); g.stroke();
    const n = labels.length, gw = w / n, bw = Math.min(34, gw * 0.3);
    for (let k = 0; k < n; k++) {
      const cx = x + gw * (k + 0.5);
      g.fillStyle = c.blue; g.globalAlpha = 0.85; const hp = pol[k] * h; g.fillRect(cx + 2, y + h - hp, bw, hp); g.globalAlpha = 1;
      g.strokeStyle = c.muted; g.lineWidth = 1.5; g.strokeRect(cx - bw - 2, y + h - ref[k] * h, bw, ref[k] * h);
      if (star) { g.strokeStyle = c.orange; g.lineWidth = 2.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(cx - 2, y + h - star[k] * h); g.lineTo(cx + bw + 6, y + h - star[k] * h); g.stroke(); g.setLineDash([]); }
      g.lineWidth = 1;
      txt(g, pol[k].toFixed(3), cx + 2 + bw / 2, y + h - hp - 4, c.fg, '10.5px sans-serif', 'center');
      txt(g, labels[k], cx, y + h + 14, c.fg, '11px sans-serif', 'center');
    }
    txt(g, 'grey outline = reference', x + w, y - 8, c.muted, '10.5px sans-serif', 'right');
  }

  /* =====================================================================
     1. Supervised fine-tuning with a chat template and loss masking
     ===================================================================== */
  AI.labs['gen-al-sft'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Instruction tuning: which tokens carry the loss?', 'illustrative: each token has a probability the toy model gives it; the loss is the mean of -ln p over the tokens that are not masked',
      `<canvas style="height:330px"></canvas>`,
      `<label>Loss on <select id="mask"><option value="resp">response tokens only (masked prompt)</option><option value="all">all tokens (no mask)</option></select></label>
       <label>Response (edit; tokens split on spaces) <input type="text" id="resp" value="It is 4 . &lt;end&gt;" style="width:100%"></label>
       ${sl('lr', 'Learning rate', 0.1, 1.5, 0.05, 0.5)}
       ${sl('rate', 'Updates per second', 0.25, 6, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:5.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const prompt = ['&lt;usr&gt;', 'What', 'is', '2', '+', '2', '?', '&lt;bot&gt;'].map((s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>'));
    const st = { playing: true };
    let toks, inResp, z, upd, dirty = true, tick = AI.ticker();
    const parseResp = () => q(ctl, '#resp').value.trim().split(/\s+/).filter(Boolean).slice(0, 10);
    const reset = () => {
      const r = AI.rng(21); const resp = parseResp();
      toks = prompt.concat(resp.length ? resp : ['…']); inResp = toks.map((_, i) => i >= prompt.length);
      z = toks.map(() => { const p = 0.05 + 0.3 * r(); return Math.log(p / (1 - p)); }); upd = 0; dirty = true;
    };
    const inLoss = (i) => q(ctl, '#mask').value === 'all' || inResp[i];
    const doStep = () => { const lr = +q(ctl, '#lr').value; z = z.map((v, i) => (inLoss(i) ? v + lr * (1 - sg(v)) : v)); upd++; dirty = true; };
    const lossOf = (pred) => { let s = 0, n = 0; toks.forEach((_, i) => { if (pred(i)) { s += -Math.log(sg(z[i])); n++; } }); return n ? s / n : 0; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const per = 8, rows = Math.ceil(toks.length / per), L = 10, cw = (W - 2 * L) / per, rh = (H - 36) / rows, bh = rh - 52;
      toks.forEach((t, i) => {
        const r = Math.floor(i / per), k = i % per, x = L + k * cw, y = 18 + r * rh, p = sg(z[i]), on = inLoss(i);
        g.fillStyle = on ? (inResp[i] ? c.green : c.orange) : c.muted; g.globalAlpha = on ? 0.85 : 0.35;
        g.fillRect(x + 4, y + bh - p * bh + 14, cw - 8, p * bh); g.globalAlpha = 1;
        g.strokeStyle = c.border; g.strokeRect(x + 4, y + 14, cw - 8, bh);
        txt(g, p.toFixed(2), x + cw / 2, y + 10, c.fg, '10.5px sans-serif', 'center');
        txt(g, t, x + cw / 2, y + bh + 30, c.fg, '11px sans-serif', 'center');
        txt(g, on ? 'in loss' : 'masked', x + cw / 2, y + bh + 43, on ? c.green : c.muted, '10px sans-serif', 'center');
      });
      txt(g, 'bar = probability of each token (prompt first)', L, H - 4, c.muted, '10.5px sans-serif');
      const lm = lossOf((i) => inLoss(i)), lr_ = lossOf((i) => inResp[i]), lp = lossOf((i) => !inResp[i]);
      rd(ctl, `updates ${upd}.  Loss being minimised (mean of -ln p over tokens in the loss) = ${lm.toFixed(4)}.\nResponse-token loss ${lr_.toFixed(4)}; prompt-token loss ${lp.toFixed(4)} (${q(ctl, '#mask').value === 'all' ? 'also being trained' : 'untouched'}).`);
      q(ctl, '#cap').textContent = q(ctl, '#mask').value === 'all' ? 'No mask: the model is also trained to predict the user\'s own words. Those updates compete for capacity with the part we want, and the model is being taught to write the question.' : 'Mask on: only the answer tokens receive gradient. The prompt is context the model reads, not text it must learn to produce. Each response token drifts upward: loss falls.';
      showVals(ctl, { lr: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, doStep, reset, draw);
    q(ctl, '#mask').onchange = () => { reset(); draw(); };
    q(ctl, '#resp').onchange = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) doStep(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Reward model trained on noisy pairwise comparisons (Bradley-Terry, SGD)
     ===================================================================== */
  AI.labs['gen-al-bt'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fitting a reward model from pairwise comparisons', 'illustrative: 4 answers with hidden qualities; an annotator picks a winner with probability sigma((q_i - q_j)/T); one pair per update',
      `<canvas style="height:360px"></canvas>`,
      `${sl('noise', 'Annotator noise T (0.2 = reliable, 3 = nearly coin flips)', 0.2, 3, 0.1, 0.6)}
       ${sl('lr', 'Learning rate', 0.05, 1, 0.05, 0.3)}
       ${sl('rate', 'Pairs per second', 0.25, 8, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const names = ['A', 'B', 'C', 'D'], Q = [0, 1, 2, 3], NP = 60;
    const st = { playing: true };
    let data, rwd, hist, k, last, dirty = true, tick = AI.ticker();
    const reset = () => {
      const r = AI.rng(8), T = +q(ctl, '#noise').value; data = [];
      for (let n = 0; n < NP; n++) {
        let i = Math.floor(r() * 4), j; do { j = Math.floor(r() * 4); } while (j === i);
        const win = r() < sg((Q[i] - Q[j]) / T); data.push(win ? [i, j] : [j, i]);
      }
      rwd = [0, 0, 0, 0]; hist = []; k = 0; last = null; dirty = true; hist.push(fullLoss());
    };
    const fullLoss = () => data.reduce((s, [w, l]) => s - Math.log(sg(rwd[w] - rwd[l])), 0) / data.length;
    const acc = () => data.filter(([w, l]) => rwd[w] > rwd[l]).length / data.length;
    const doStep = () => {
      const lr = +q(ctl, '#lr').value, [w, l] = data[k % NP], d = rwd[w] - rwd[l], pw = sg(d);
      last = { w, l, d, loss: -Math.log(pw), g: 1 - pw };
      rwd[w] += lr * (1 - pw); rwd[l] -= lr * (1 - pw); k++; hist.push(fullLoss()); if (hist.length > 400) hist.shift(); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 40, R = W - 12, mid = Math.round(H * 0.56), bt = 24, bb = mid - 26;
      txt(g, 'learned reward r (bars) versus the ideal reward (hidden quality / T, dashed), both centred on 0', L, 14, c.muted, '11px sans-serif');
      const mr = rwd.reduce((a, b) => a + b, 0) / 4, mq = 1.5, span = Math.max(2.5, ...rwd.map((v) => Math.abs(v - mr)), 1.5 / +q(ctl, '#noise').value);
      const Y = (v) => (bt + bb) / 2 - (v / span) * ((bb - bt) / 2), gw = (R - L) / 4;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(R, Y(0)); g.stroke();
      names.forEach((nm, i) => {
        const cx = L + gw * (i + 0.5), v = rwd[i] - mr;
        g.fillStyle = v >= 0 ? c.green : c.red; g.globalAlpha = 0.85; g.fillRect(cx - 18, Math.min(Y(0), Y(v)), 36, Math.abs(Y(v) - Y(0))); g.globalAlpha = 1;
        g.strokeStyle = c.orange; g.lineWidth = 2.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(cx - 26, Y((Q[i] - mq) / +q(ctl, '#noise').value)); g.lineTo(cx + 26, Y((Q[i] - mq) / +q(ctl, '#noise').value)); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
        txt(g, `${nm}  r=${v.toFixed(2)}`, cx, bb + 14, c.fg, '11px sans-serif', 'center');
      });
      // loss curve
      const t0 = mid + 22, t1 = H - 22;
      txt(g, 'mean loss over all 60 pairs (updates →)', L, mid + 12, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      const lm = 0.75, xm = Math.max(60, hist.length - 1), X = (i) => L + (i / xm) * (R - L), YL = (v) => t1 - Math.min(1, v / lm) * (t1 - t0);
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => (i ? g.lineTo(X(i), YL(v)) : g.moveTo(X(i), YL(v)))); g.stroke(); g.lineWidth = 1;
      txt(g, '0.693 = ln 2 (no knowledge)', R, YL(0.693) - 3, c.muted, '10.5px sans-serif', 'right');
      g.strokeStyle = c.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(L, YL(0.693)); g.lineTo(R, YL(0.693)); g.stroke(); g.setLineDash([]);
      txt(g, '0', L - 5, t1 + 3, c.muted, '10px sans-serif', 'right');
      const cur = last ? `last pair: ${names[last.w]} beat ${names[last.l]}; gap r_w - r_l = ${last.d.toFixed(3)}; loss -ln sigma(gap) = ${last.loss.toFixed(3)}; both rewards move by lr x ${last.g.toFixed(3)}.` : 'Press play: one labelled pair per update.';
      rd(ctl, `${cur}\nupdates ${k}; mean loss ${fullLoss().toFixed(4)}; pairs ordered correctly ${(acc() * 100).toFixed(0)}% of the labelled pairs.`);
      q(ctl, '#cap').textContent = +q(ctl, '#noise').value > 1.8 ? 'Very noisy labels: the ranking is still recoverable on average, but slowly and with bigger errors, and the loss cannot go far below ln 2.' : 'Each update lifts the winner and pushes down the loser, by less when the model already agrees (small 1 - sigma). Bars drift toward the dashed hidden qualities.';
      showVals(ctl, { noise: (v) => v, lr: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, doStep, reset, draw);
    q(ctl, '#noise').oninput = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < n; i++) doStep(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Bradley-Terry loss curve: two rewards, a gap, a probability and a loss
     ===================================================================== */
  AI.labs['gen-al-btcurve'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'The pairwise loss as a function of the reward gap', 'illustrative: set two rewards; the point glides to the new gap on the curve -ln sigma(gap)',
      `<canvas style="height:300px"></canvas>`,
      `${sl('rw', 'Reward of the preferred answer r_w', -4, 4, 0.1, 2)}
       ${sl('rl', 'Reward of the rejected answer r_l', -4, 4, 0.1, 1)}
       ${sl('rate', 'Glide speed', 0.3, 3, 0.1, 1)}
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    let cur = 1, dirty = true;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 40, R = W - 14, T = 22, B = H - 28, XM = 6, X = (d) => L + (d + XM) / (2 * XM) * (R - L), Y = (v) => B - Math.min(v, 6.5) / 6.5 * (B - T);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, T); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      for (const d of [-6, -3, 0, 3, 6]) txt(g, d, X(d), B + 14, c.muted, '11px sans-serif', 'center');
      txt(g, 'gap = r_w - r_l', R, B + 27, c.muted, '11px sans-serif', 'right');
      txt(g, 'loss = -ln sigma(gap)', L + 4, T - 6, c.blue, '11px sans-serif');
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 120; i++) { const d = -XM + 2 * XM * i / 120; i ? g.lineTo(X(d), Y(-Math.log(sg(d)))) : g.moveTo(X(d), Y(-Math.log(sg(d)))); } g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.green; g.setLineDash([4, 3]); g.beginPath(); for (let i = 0; i <= 120; i++) { const d = -XM + 2 * XM * i / 120; const y = B - sg(d) * (B - T) * 0.5; i ? g.lineTo(X(d), y) : g.moveTo(X(d), y); } g.stroke(); g.setLineDash([]);
      txt(g, 'P(preferred wins) = sigma(gap), drawn at half height', R - 4, T + 12, c.green, '10.5px sans-serif', 'right');
      const d = cur, l = -Math.log(sg(d));
      g.strokeStyle = c.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(d), B); g.lineTo(X(d), Y(l)); g.stroke(); g.setLineDash([]);
      g.fillStyle = c.orange; g.beginPath(); g.arc(X(d), Y(l), 6, 0, 7); g.fill();
      const tg = +q(ctl, '#rw').value - +q(ctl, '#rl').value;
      rd(ctl, `gap = ${tg.toFixed(2)}   P(preferred wins) = sigma(gap) = ${sg(tg).toFixed(4)}\nloss = -ln sigma(gap) = ${(-Math.log(sg(tg))).toFixed(4)}   slope weight 1 - sigma(gap) = ${(1 - sg(tg)).toFixed(4)}`);
      q(ctl, '#cap').textContent = tg > 0 ? 'The reward model already agrees with the human: loss is small and the gradient weight 1 - sigma(gap) is small, so there is little left to learn from this pair.' : (tg === 0 ? 'No opinion: loss is ln 2 = 0.693.' : 'The reward model disagrees with the human. The loss is large and roughly linear in the mistake: the update is as strong as it can be.');
      showVals(ctl, { rw: (v) => v, rl: (v) => v, rate: (v) => v });
    };
    q(ctl, '#rw').oninput = () => { dirty = true; }; q(ctl, '#rl').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      const tg = +q(ctl, '#rw').value - +q(ctl, '#rl').value;
      if (Math.abs(tg - cur) > 1e-3) { cur += (tg - cur) * Math.min(1, dt * 3 * +q(ctl, '#rate').value); dirty = true; } else if (cur !== tg) { cur = tg; dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. RLHF: policy moving under reward minus beta KL, with a proxy reward that can be hacked
     ===================================================================== */
  AI.labs['gen-al-kl'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Reward plus a KL leash: the policy moves, then stops', 'illustrative: 3 answers, exact gradient ascent on E[proxy reward] - beta x KL(policy || reference); the orange tick is the optimum pi*',
      `<canvas style="height:400px"></canvas>`,
      `${sl('beta', 'KL strength beta', 0.1, 4, 0.1, 1)}
       <div class="row">${num('ra', 'proxy r(A)', 1)}${num('rb', 'proxy r(B)', 1.8)}${num('rc', 'proxy r(C)', 3)}</div>
       <div class="row">${num('ta', 'true r(A)', 1)}${num('tb', 'true r(B)', 1)}${num('tc', 'true r(C)', 0)}</div>
       ${sl('rate', 'Speed (time per second)', 0.25, 6, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const ref = [0.5, 0.3, 0.2], labels = ['A concise, correct', 'B long, correct', 'C padded, flattering, wrong'], st = { playing: true };
    let th, hist, T, dirty = true;
    const rv = () => ['ra', 'rb', 'rc'].map((i) => +q(ctl, '#' + i).value || 0), tv = () => ['ta', 'tb', 'tc'].map((i) => +q(ctl, '#' + i).value || 0);
    const stat = (p) => { const r = rv(), t = tv(), b = +q(ctl, '#beta').value, e = (x) => p.reduce((s, v, i) => s + v * x[i], 0); return { prox: e(r), tru: e(t), kl: klf(p, ref), J: e(r) - b * klf(p, ref) }; };
    const optimum = () => { const r = rv(), b = +q(ctl, '#beta').value, w = ref.map((p, i) => p * Math.exp(r[i] / b)); const Z = w.reduce((a, c) => a + c, 0); return { p: w.map((v) => v / Z), Z }; };
    const reset = () => { th = ref.map(Math.log); hist = []; T = 0; hist.push(Object.assign({ t: 0 }, stat(ref))); dirty = true; };
    const advance = (h) => {
      const r = rv(), b = +q(ctl, '#beta').value, n = 6;
      for (let s = 0; s < n; s++) {
        const p = softmax(th), a = p.map((v, i) => r[i] - b * Math.log(v / ref[i])), ab = p.reduce((s2, v, i) => s2 + v * a[i], 0);
        th = th.map((v, i) => v + (h / n) * 2 * p[i] * (a[i] - ab));
      }
      T += h; const last = hist[hist.length - 1];
      if (T - last.t >= 0.1) { hist.push(Object.assign({ t: T }, stat(softmax(th)))); if (hist.length > 500) hist.shift(); }
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const p = softmax(th), op = optimum(), s = stat(p), L = 12, R = W - 12, split = Math.round(H * 0.5);
      distBars(g, c, L, 28, R - L, split - 70, ['A', 'B', 'C'], ref, p, op.p, 'policy probability of each answer');
      const t0 = split + 22, t1 = H - 20, X0 = 40, X = (t) => X0 + (t / Math.max(8, T)) * (R - X0);
      const mx = Math.max(3.4, ...hist.map((h) => h.prox)), mn = Math.min(0, ...hist.map((h) => h.tru)), Y = (v) => t1 - (v - mn) / (mx - mn) * (t1 - t0);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(X0, t0); g.lineTo(X0, t1); g.lineTo(R, t1); g.stroke();
      txt(g, 'over time:  proxy reward (red), true reward (green), KL (purple)', X0, split + 14, c.muted, '11px sans-serif');
      const line = (key, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); hist.forEach((h, i) => (i ? g.lineTo(X(h.t), Y(h[key])) : g.moveTo(X(h.t), Y(h[key])))); g.stroke(); g.lineWidth = 1; };
      line('prox', c.red); line('tru', c.green); line('kl', c.purple);
      txt(g, 'time →', R, t1 + 14, c.muted, '10.5px sans-serif', 'right');
      const b = +q(ctl, '#beta').value;
      rd(ctl, `time ${T.toFixed(1)}   beta = ${b}\npolicy (A, B, C) = (${p.map((v) => v.toFixed(4)).join(', ')})   optimum pi* = (${op.p.map((v) => v.toFixed(4)).join(', ')})\nE[proxy reward] ${s.prox.toFixed(4)}   E[true reward] ${s.tru.toFixed(4)}   KL ${s.kl.toFixed(4)}   objective ${s.J.toFixed(4)} (maximum beta ln Z = ${(b * Math.log(op.Z)).toFixed(4)})`);
      const r = rv(); const hacked = s.tru < stat(ref).tru - 0.05 && s.prox > stat(ref).prox;
      q(ctl, '#cap').textContent = hacked ? 'Reward hacking: the proxy reward keeps rising while the TRUE reward falls. The policy is shifting mass to the padded, flattering answer C because the reward model over-scores it. A larger beta (a shorter leash) limits the damage.' : (Math.max(...r) === r[0] ? 'The proxy and the truth agree here, so optimising the proxy helps.' : 'The policy climbs toward the proxy optimum but the KL term pulls it back toward the reference, so it settles at pi*, not at "all mass on the best-scored answer".');
      showVals(ctl, { beta: (v) => v, rate: (v) => v });
    };
    const step = () => advance(0.25);
    wire(ctl, st, step, reset, draw);
    ['beta', 'ra', 'rb', 'rc', 'ta', 'tb', 'tc'].forEach((id) => { q(ctl, '#' + id).oninput = () => { if (id === 'beta' || id[0] === 'r') { reset(); } dirty = true; draw(); }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && T < 400) advance(dt * +q(ctl, '#rate').value);
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5. DPO: preference pairs push the policy to pi*, implicit reward
     ===================================================================== */
  AI.labs['gen-al-dpo'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Direct preference optimisation: no reward model, same optimum', 'illustrative: 3 answers, three preference pairs, exact gradient descent on the DPO loss; the orange tick is pi* for the reward behind the labels',
      `<canvas style="height:420px"></canvas>`,
      `${sl('beta', 'DPO beta', 0.1, 3, 0.1, 1)}
       <div class="row">${num('ra', 'hidden r(A)', 1)}${num('rb', 'hidden r(B)', 2)}${num('rc', 'hidden r(C)', 0)}</div>
       <label>Labels <select id="mode"><option value="soft">noisy labels (frequencies follow Bradley-Terry)</option><option value="hard">always the same winner (no noise)</option></select></label>
       ${sl('rate', 'Speed (time per second)', 0.25, 6, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const ref = [0.5, 0.3, 0.2], lref = ref.map(Math.log), pairs = [[1, 0], [0, 2], [1, 2]], nm = ['A', 'B', 'C'], st = { playing: true };
    let th, T, dirty = true;
    const rv = () => ['ra', 'rb', 'rc'].map((i) => +q(ctl, '#' + i).value || 0), beta = () => +q(ctl, '#beta').value;
    const marg = (p, pr) => { const a = th.map((v, i) => v - lref[i]); return beta() * (a[pr[0]] - a[pr[1]]); };
    const target = (pr) => (q(ctl, '#mode').value === 'hard' ? 1 : sg(rv()[pr[0]] - rv()[pr[1]]));
    const optimum = () => { const w = ref.map((p, i) => p * Math.exp(rv()[i] / beta())); const Z = w.reduce((a, b) => a + b, 0); return w.map((v) => v / Z); };
    const loss = () => pairs.reduce((s, pr) => { const m = marg(th, pr), p = target(pr); return s + p * -Math.log(sg(m)) + (1 - p) * -Math.log(sg(-m)); }, 0) / pairs.length;
    const reset = () => { th = lref.slice(); T = 0; dirty = true; };
    const advance = (h) => {
      const n = 6, b = beta();
      for (let s = 0; s < n; s++) { const d = [0, 0, 0]; pairs.forEach((pr) => { const w = (sg(marg(th, pr)) - target(pr)) * b; d[pr[0]] -= w; d[pr[1]] += w; }); th = th.map((v, i) => v + (h / n) * 2 * d[i]); }
      T += h; dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pi = softmax(th), L = 12, R = W - 12, hard = q(ctl, '#mode').value === 'hard', b = beta(), r = rv();
      distBars(g, c, L, 28, R - L, 130, nm, ref, pi, hard ? null : optimum(), 'policy probability of each answer');
      // implicit reward bars, centred
      const imp = pi.map((p, i) => b * Math.log(p / ref[i])), mi = imp.reduce((a, v) => a + v, 0) / 3, mr = r.reduce((a, v) => a + v, 0) / 3;
      const top = 236, h2 = 120, span = Math.max(2.5, ...imp.map((v) => Math.abs(v - mi)), ...r.map((v) => Math.abs(v - mr))), Y = (v) => top + h2 / 2 - (v / span) * (h2 / 2), gw = (R - L) / 3;
      txt(g, 'implicit reward  beta ln(pi / pi_ref)  (bars, centred) versus hidden reward (orange, centred)', L, top - 10, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(R, Y(0)); g.stroke();
      nm.forEach((n, i) => {
        const cx = L + gw * (i + 0.5), v = imp[i] - mi;
        g.fillStyle = v >= 0 ? c.green : c.red; g.globalAlpha = 0.85; g.fillRect(cx - 18, Math.min(Y(0), Y(v)), 36, Math.abs(Y(v) - Y(0))); g.globalAlpha = 1;
        if (!hard) { g.strokeStyle = c.orange; g.lineWidth = 2.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(cx - 28, Y(r[i] - mr)); g.lineTo(cx + 28, Y(r[i] - mr)); g.stroke(); g.setLineDash([]); g.lineWidth = 1; }
        txt(g, `${n}  ${v.toFixed(2)}`, cx, top + h2 + 16, c.fg, '11px sans-serif', 'center');
      });
      const ms = pairs.map((pr) => marg(th, pr));
      rd(ctl, `time ${T.toFixed(1)}   DPO loss ${loss().toFixed(4)} (starts at ln 2 = 0.6931)\nmargins m = beta[(ln pi/pi_ref)_w - (ln pi/pi_ref)_l]:  B>A ${ms[0].toFixed(3)}   A>C ${ms[1].toFixed(3)}   B>C ${ms[2].toFixed(3)}\npolicy (A, B, C) = (${pi.map((v) => v.toFixed(4)).join(', ')})`);
      q(ctl, '#cap').textContent = hard ? 'Labels with no noise: the loss can always be lowered by pushing the margins larger, so the policy keeps drifting away from the reference toward the winners. DPO has no reward model to say "enough"; beta and early stopping are what limit this.' : 'With noisy labels the loss has a finite minimum, and there the policy equals pi*: the implicit reward bars match the hidden reward (up to a constant) without any reward model or sampling.';
      showVals(ctl, { beta: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => advance(0.25), reset, draw);
    ['beta', 'ra', 'rb', 'rc'].forEach((id) => { q(ctl, '#' + id).oninput = () => { if (id === 'beta') reset(); dirty = true; draw(); }; });
    q(ctl, '#mode').onchange = () => { reset(); draw(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && T < 600) advance(dt * +q(ctl, '#rate').value);
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. Group-relative advantages from a checker
     ===================================================================== */
  AI.labs['gen-al-group'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Sample a group, check each answer, standardise within the group', 'illustrative: one question "17 x 3 = ?", a checker gives reward 1 or 0; each round samples G answers, computes advantages, nudges the policy',
      `<canvas style="height:560px"></canvas>`,
      `${sl('G', 'Group size G', 2, 16, 1, 8)}
       ${sl('p0', 'Start: probability of the correct answer', 0.05, 0.9, 0.05, 0.25)}
       ${sl('lr', 'Learning rate', 0.2, 4, 0.1, 1.5)}
       ${sl('rate', 'Speed', 0.25, 4, 0.25, 1)}
       <label><input type="checkbox" id="auto" checked> keep going round after round</label>
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7.5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const ans = ['51', '41', '54', '61'], st = { playing: true };
    let th, rs, round, samples, phase, ph, shown, pol0, hist, dirty = true, adv, mean, sd;
    const G = () => +q(ctl, '#G').value;
    const reset = () => {
      const p0 = +q(ctl, '#p0').value; th = [Math.log(p0), ...[1, 2, 3].map(() => Math.log((1 - p0) / 3))];
      rs = AI.rng(31); round = 0; hist = [p0]; newRound(); dirty = true;
    };
    const newRound = () => {
      const p = softmax(th), g = G(); samples = [];
      for (let i = 0; i < g; i++) { let u = rs(), k = 0; while (k < 3 && u > p[k]) { u -= p[k]; k++; } samples.push(k); }
      phase = 'sample'; ph = 0; shown = 0; adv = null; round++;
    };
    const stats = () => {
      const r = samples.map((k) => (k === 0 ? 1 : 0)); mean = r.reduce((a, b) => a + b, 0) / r.length;
      sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) * (b - mean), 0) / r.length);
      adv = r.map((v) => (sd > 0 ? (v - mean) / sd : 0));
    };
    const applyUpdate = () => {
      const lr = +q(ctl, '#lr').value, g = samples.length, d = [0, 0, 0, 0];
      samples.forEach((k, i) => { for (let j = 0; j < 4; j++) d[j] += adv[i] * ((j === k ? 1 : 0) - softmax(th)[j]) / g; });
      pol0 = th.slice(); th = th.map((v, j) => v + lr * d[j]); hist.push(softmax(th)[0]);
    };
    const advance = (dt) => {
      ph += dt;
      if (phase === 'sample') { const n = Math.min(samples.length, Math.floor(ph / 0.7)); if (n !== shown) { shown = n; dirty = true; } if (shown >= samples.length) { phase = 'stats'; ph = 0; stats(); dirty = true; } }
      else if (phase === 'stats') { if (ph > 2.2) { phase = 'update'; ph = 0; applyUpdate(); } dirty = true; }
      else if (phase === 'update') { if (ph > 1.5) { phase = 'done'; ph = 0; } dirty = true; }
      else if (phase === 'done') { if (q(ctl, '#auto').checked && ph > 1.0) newRound(); dirty = true; }
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 12, R = W - 12, show = phase === 'update' && pol0 ? pol0.map((v, i) => v + (th[i] - v) * ease(Math.min(1, ph / 1.2))) : th, p = softmax(show);
      txt(g, `policy over four candidate answers (round ${round})`, L, 16, c.muted, '11px sans-serif');
      const bw = (R - L) / 4;
      p.forEach((v, k) => {
        const x = L + bw * k + 6, hh = 70, y = 28;
        g.fillStyle = k === 0 ? c.green : c.blue; g.globalAlpha = 0.85; g.fillRect(x, y + hh - v * hh, bw - 12, v * hh); g.globalAlpha = 1;
        g.strokeStyle = c.border; g.strokeRect(x, y, bw - 12, hh);
        txt(g, `${ans[k]}${k === 0 ? ' ✓' : ''}  ${v.toFixed(3)}`, x + (bw - 12) / 2, y + hh + 14, c.fg, '11px sans-serif', 'center');
      });
      // table
      const ty = 138, rh = 20, g_ = samples.length, cols = [L, L + 34, L + 80, L + 140];
      txt(g, '#', cols[0], ty - 6, c.muted, '11px sans-serif'); txt(g, 'sample', cols[1], ty - 6, c.muted, '11px sans-serif'); txt(g, 'reward', cols[2], ty - 6, c.muted, '11px sans-serif'); txt(g, 'advantage = (reward - mean) / std', cols[3], ty - 6, c.muted, '11px sans-serif');
      const bx0 = cols[3] + 70, bxW = R - bx0 - 50, zero = bx0 + bxW / 2;
      for (let i = 0; i < g_; i++) {
        if (i >= shown) break; const y = ty + 8 + i * rh, k = samples[i], rr = k === 0 ? 1 : 0;
        txt(g, i + 1, cols[0], y + 6, c.muted, '11px sans-serif'); txt(g, ans[k], cols[1], y + 6, c.fg, '12px sans-serif'); txt(g, rr, cols[2], y + 6, rr ? c.green : c.red, 'bold 12px sans-serif');
        if (adv) {
          const a = adv[i], w = Math.min(1, Math.abs(a) / 2) * bxW / 2;
          txt(g, a.toFixed(4), cols[3], y + 6, c.fg, '11px sans-serif');
          g.fillStyle = a >= 0 ? c.green : c.red; g.globalAlpha = 0.85; g.fillRect(a >= 0 ? zero : zero - w, y - 4, w, 12); g.globalAlpha = 1;
        }
      }
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(zero, ty); g.lineTo(zero, ty + 8 + g_ * rh); g.stroke();
      // history
      const h0 = ty + 8 + 16 * rh + 24, h1 = H - 18, X0 = 40;
      txt(g, 'probability of the correct answer after each round', L, h0 - 8, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(X0, h0); g.lineTo(X0, h1); g.lineTo(R, h1); g.stroke();
      const xm = Math.max(10, hist.length - 1), X = (i) => X0 + i / xm * (R - X0), Y = (v) => h1 - v * (h1 - h0);
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)))); g.stroke(); g.lineWidth = 1;
      txt(g, '1', X0 - 5, h0 + 4, c.muted, '10px sans-serif', 'right'); txt(g, '0', X0 - 5, h1, c.muted, '10px sans-serif', 'right');
      let msg;
      if (phase === 'sample') msg = `Round ${round}: sampling answers from the policy (${shown} of ${samples.length}).`;
      else {
        const nc = samples.filter((k) => k === 0).length;
        msg = `Round ${round}: ${nc} of ${samples.length} correct. mean reward = ${mean.toFixed(4)}; std = ${sd.toFixed(4)} (divide by ${samples.length}). ` + (sd === 0 ? 'All rewards are equal, so every advantage is 0: this group teaches nothing.' : `Correct answers get advantage ${((1 - mean) / sd).toFixed(4)}, wrong ones ${((0 - mean) / sd).toFixed(4)}; they sum to 0.`);
      }
      rd(ctl, msg);
      q(ctl, '#cap').textContent = phase === 'sample' ? 'The checker marks each sampled answer 1 (correct) or 0.' : (phase === 'stats' ? 'Rewards are compared with the group itself: above-average answers get a positive advantage, below-average a negative one. No separate value network is needed.' : 'The policy is nudged toward answers with a positive advantage and away from those with a negative one (a plain policy-gradient step; no clipping or KL term in this toy).');
      showVals(ctl, { G: (v) => v, p0: (v) => v, lr: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => { if (phase === 'sample') { shown = samples.length; phase = 'stats'; ph = 0; stats(); } else if (phase === 'stats') { phase = 'update'; ph = 0; applyUpdate(); } else if (phase === 'update') { phase = 'done'; ph = 0; } else { newRound(); } dirty = true; }, reset, draw);
    q(ctl, '#G').oninput = () => { reset(); draw(); }; q(ctl, '#p0').oninput = () => { reset(); draw(); };
    q(ctl, '#lr').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) advance(dt * +q(ctl, '#rate').value);
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. Refusal threshold: missed harm versus over-refusal
     ===================================================================== */
  AI.labs['gen-al-refusal'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Where to set the refusal threshold', 'illustrative: abstract "requests" with a risk score from an imperfect classifier; no request text is modelled, only the two score distributions',
      `<canvas style="height:360px"></canvas>`,
      `${sl('t', 'Refuse when the score is above', -2.5, 2.5, 0.1, 0.8)}
       ${sl('d', 'How well the scores separate the two groups', 0.5, 4, 0.1, 2.5)}
       ${sl('prev', 'Share of requests that should be refused (%)', 1, 50, 1, 10)}
       ${sl('cm', 'Cost of one wrongly answered request', 1, 50, 1, 10)}
       ${sl('co', 'Cost of one wrongly refused request', 1, 50, 1, 1)}
       ${sl('rate', 'Requests per second', 1, 30, 1, 6)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let rs, dots, cnt, dirty = true, tick = AI.ticker();
    const reset = () => { rs = AI.rng(41); dots = []; cnt = { tp: 0, fn: 0, fp: 0, tn: 0 }; dirty = true; };
    const par = () => ({ t: +q(ctl, '#t').value, d: +q(ctl, '#d').value, pv: +q(ctl, '#prev').value / 100, cm: +q(ctl, '#cm').value, co: +q(ctl, '#co').value });
    const rates = (t, d) => ({ miss: Phi(t - d / 2), over: 1 - Phi(t + d / 2) });
    const cost = (t, p) => { const r = rates(t, p.d); return 1000 * (p.pv * r.miss * p.cm + (1 - p.pv) * r.over * p.co); };
    const addOne = () => {
      const p = par(), harm = rs() < p.pv, s = (harm ? p.d / 2 : -p.d / 2) + AI.randn(rs), refused = s > p.t;
      cnt[harm ? (refused ? 'tp' : 'fn') : (refused ? 'fp' : 'tn')]++;
      dots.push({ s, harm, refused, age: 0 }); if (dots.length > 160) dots.shift(); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const p = par(), L = 14, R = W - 14, XM = 5.5, X = (v) => L + (v + XM) / (2 * XM) * (R - L), top = 26, bot = Math.round(H * 0.5);
      const pdf = (x, m) => Math.exp(-0.5 * (x - m) * (x - m)) / Math.sqrt(2 * Math.PI);
      const Y = (v) => bot - v * (bot - top) * 2.2;
      txt(g, 'risk score (higher = more likely refused)', L, 14, c.muted, '11px sans-serif');
      const curve = (m, w, col) => { g.beginPath(); g.moveTo(X(-XM), bot); for (let i = 0; i <= 160; i++) { const x = -XM + 2 * XM * i / 160; g.lineTo(X(x), Y(pdf(x, m) * w)); } g.lineTo(X(XM), bot); g.closePath(); g.fillStyle = col; g.globalAlpha = 0.3; g.fill(); g.globalAlpha = 1; g.strokeStyle = col; g.stroke(); };
      curve(-p.d / 2, 1 - p.pv, c.blue); curve(p.d / 2, p.pv * 3, c.red);
      txt(g, 'blue: should answer', L, top + 8, c.blue, 'bold 11px sans-serif'); txt(g, 'red: should refuse (3x taller)', R, top + 8, c.red, 'bold 11px sans-serif', 'right');
      g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(X(p.t), top + 14); g.lineTo(X(p.t), bot + 80); g.stroke(); g.lineWidth = 1;
      txt(g, 'threshold', X(p.t) + 4, bot + 90, c.fg, '10.5px sans-serif');
      let bt = -3, bc = 1e18; for (let t = -3; t <= 3; t += 0.01) { const v = cost(t, p); if (v < bc) { bc = v; bt = t; } }
      g.strokeStyle = c.orange; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(X(bt), bot - 4); g.lineTo(X(bt), bot + 80); g.stroke(); g.setLineDash([]);
      txt(g, 'cheapest threshold', X(bt) - 4, bot + 100, c.orange, '10.5px sans-serif', 'right');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, bot); g.lineTo(R, bot); g.stroke();
      // dots strip
      dots.forEach((dt, i) => { const y = bot + 14 + (i % 8) * 8; g.beginPath(); g.arc(X(Math.max(-XM, Math.min(XM, dt.s))), y, 3, 0, 7); g.fillStyle = dt.harm ? c.red : c.blue; g.globalAlpha = 0.35 + 0.65 * (i / dots.length); g.fill(); g.globalAlpha = 1; if (dt.refused) { g.strokeStyle = c.fg; g.stroke(); } });
      txt(g, 'last 160: colour = should be, outline = refused', L, bot + 118, c.muted, '10.5px sans-serif');
      const r = rates(p.t, p.d), n = cnt.tp + cnt.fn + cnt.fp + cnt.tn;
      // confusion counts
      const cy = bot + 136;
      txt(g, `should refuse: refused ${cnt.tp}, answered ${cnt.fn}`, L, cy, c.fg, '11px sans-serif'); txt(g, `should answer: answered ${cnt.tn}, refused ${cnt.fp}`, L, cy + 15, c.fg, '11px sans-serif');
      rd(ctl, `Theory at this threshold: wrongly answered ${(r.miss * 100).toFixed(1)}% of the requests that should be refused; wrongly refused ${(r.over * 100).toFixed(1)}% of the ones that should be answered.\nExpected cost per 1000 requests = ${cost(p.t, p).toFixed(1)}; the cheapest threshold for these costs is ${bt.toFixed(2)} (cost ${bc.toFixed(1)}).\nSimulated so far: ${n} requests.`);
      q(ctl, '#cap').textContent = 'Moving the threshold right removes wrong refusals but lets more harmful requests through; moving it left does the opposite. Better separation (a better model or classifier) lowers both at once; the "right" threshold depends on how costly each mistake is, which is a policy choice, not a mathematical fact.';
      showVals(ctl, { t: (v) => v, d: (v) => v, prev: (v) => v + '%', cm: (v) => v, co: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, addOne, reset, draw);
    ['t', 'd', 'prev', 'cm', 'co'].forEach((id) => { q(ctl, '#' + id).oninput = () => { if (id !== 'cm' && id !== 'co') reset(); dirty = true; draw(); }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) addOne(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     8. Red-team attempts: probability of at least one success, and "found nothing"
     ===================================================================== */
  AI.labs['gen-al-redteam-k'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'How many red-team attempts does it take to see a failure?', 'illustrative: each attempt independently succeeds with a hidden probability p; the counts are abstract, no content is modelled',
      `<canvas style="height:300px"></canvas>`,
      `${sl('lp', 'Hidden success chance per attempt p (%)', 0.1, 10, 0.1, 2)}
       ${sl('rate', 'Attempts per second', 1, 40, 1, 8)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let rs, tries, hits, firstHit, dirty = true, tick = AI.ticker();
    const reset = () => { rs = AI.rng(13); tries = []; hits = 0; firstHit = -1; dirty = true; };
    const addOne = () => { if (tries.length >= 400) return; const p = +q(ctl, '#lp').value / 100, h = rs() < p; tries.push(h); if (h) { hits++; if (firstHit < 0) firstHit = tries.length; } dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const p = +q(ctl, '#lp').value / 100, L = 40, R = W - 14, T = 22, B = H - 74, KM = 400, X = (k) => L + k / KM * (R - L), Y = (v) => B - v * (B - T);
      txt(g, 'probability that at least one of k attempts succeeds = 1 - (1 - p)^k', L, 14, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, T); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      for (const k of [0, 100, 200, 300, 400]) txt(g, k, X(k), B + 14, c.muted, '10.5px sans-serif', 'center');
      txt(g, '1', L - 5, T + 4, c.muted, '10.5px sans-serif', 'right'); txt(g, '0', L - 5, B, c.muted, '10.5px sans-serif', 'right');
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); for (let k = 0; k <= KM; k += 4) { const v = 1 - Math.pow(1 - p, k); k ? g.lineTo(X(k), Y(v)) : g.moveTo(X(k), Y(v)); } g.stroke(); g.lineWidth = 1;
      const k = tries.length; if (k) { g.fillStyle = c.orange; g.beginPath(); g.arc(X(k), Y(1 - Math.pow(1 - p, k)), 5, 0, 7); g.fill(); }
      // attempt strip
      const sy = B + 34, cw = (R - L) / 100;
      tries.slice(-100).forEach((h, i) => { g.fillStyle = h ? c.red : c.muted; g.globalAlpha = h ? 1 : 0.35; g.fillRect(L + i * cw, sy, Math.max(1.5, cw - 1), 14); g.globalAlpha = 1; });
      txt(g, 'last 100 attempts (red = a failure was found)', L, sy + 28, c.muted, '10.5px sans-serif');
      const upper = k ? 1 - Math.pow(0.05, 1 / k) : 1;
      rd(ctl, `attempts ${k}; failures found ${hits}${firstHit > 0 ? ' (first at attempt ' + firstHit + ')' : ''}.\nP(at least one in ${k} attempts) = ${(1 - Math.pow(1 - p, k)).toFixed(4)} when p = ${(p * 100).toFixed(1)}%.\n` + (hits === 0 && k > 0 ? `Nothing found in ${k} attempts. That only shows p is probably below about ${(upper * 100).toFixed(2)}% (a 95% upper bound, 1 - 0.05^(1/k)); it does not show p = 0.` : (hits > 0 ? `Found ${hits}: the observed rate ${(hits / k * 100).toFixed(2)}% is the estimate; its uncertainty is large when k is small.` : '')));
      q(ctl, '#cap').textContent = 'A red team that finds nothing has measured an upper bound, not safety. Rare failures need many attempts, and attempts must differ enough to cover different weaknesses; repeating one trick k times is not k independent chances.';
      showVals(ctl, { lp: (v) => v + '%', rate: (v) => v });
    };
    wire(ctl, st, addOne, reset, draw);
    q(ctl, '#lp').oninput = () => { reset(); draw(); }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const n = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < n; i++) addOne(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
