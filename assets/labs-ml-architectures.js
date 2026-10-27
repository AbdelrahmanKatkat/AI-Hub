/* ML Module 7 labs: RNN unrolling, LSTM cell, attention, sequence-model comparison, embedding space,
   autoencoder bottleneck, GNN message passing, architecture chooser.
   All animations run on a dt-based clock (seconds), have a speed slider, pause/step/reset, and captions.
   Simulations that stand in for training are labelled "illustrative". */
(function () {
  const MONO = '12px ui-monospace,Menlo,Consolas,monospace', SANS = '12px system-ui,-apple-system,Segoe UI,sans-serif';
  const q = (root, s) => root.querySelector(s);
  const nums = (s) => String(s).split(/[,;\s]+/).map(parseFloat).filter((v) => Number.isFinite(v));
  const mat = (s) => String(s).split(';').map((r) => nums(r)).filter((r) => r.length);
  const sg = (x) => 1 / (1 + Math.exp(-x));
  const f2 = (v, d = 3) => AI.fmt(v, d);
  const ease = (t) => AI.smooth(t);
  function txt(g, s, x, y, col, al, font) { g.fillStyle = col; g.textAlign = al || 'left'; g.font = font || SANS; g.fillText(s, x, y); }

  function transport() {
    return `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="3" step="0.1" value="1"></label>
      <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step ▶</button><button class="btn" id="reset">Reset</button></div>`;
  }
  /* Real-time clock: T advances by dt*speed seconds; one "phase" lasts `phase` seconds. */
  function makeClock(root, o) {
    const c = { T: 0, playing: o.autoplay !== false, speed: 1, dirty: true, phase: o.phase || 1.5, phases: o.phases, hold: o.hold || 0, loop: o.loop !== false };
    const play = q(root, '#play'), sp = q(root, '#sp'), spv = q(root, '#spv');
    const total = () => c.phases * c.phase + c.hold;
    const lab = () => (play.textContent = c.playing ? '⏸ Pause' : '▶ Play');
    sp.oninput = () => { c.speed = +sp.value; spv.textContent = '×' + c.speed.toFixed(1); };
    sp.oninput();
    play.onclick = () => { c.playing = !c.playing; lab(); };
    q(root, '#step').onclick = () => {
      c.playing = false; lab();
      c.T = (Math.floor(c.T / c.phase + 1e-6) + 1) * c.phase;
      if (c.T >= total() - 1e-6) c.T = c.loop ? 0 : c.phases * c.phase;
      c.dirty = true;
    };
    q(root, '#reset').onclick = () => { c.T = 0; c.playing = true; lab(); c.dirty = true; };
    c.update = (dt) => {
      if (!c.playing) return;
      c.T += dt * c.speed; c.dirty = true;
      if (c.T >= total()) { if (c.loop) c.T -= total(); else { c.T = c.phases * c.phase; c.playing = false; lab(); } }
    };
    c.idx = () => Math.min(c.phases - 1, Math.floor(c.T / c.phase));
    c.frac = () => (c.T >= c.phases * c.phase ? 1 : (c.T % c.phase) / c.phase);
    c.done = () => c.T >= c.phases * c.phase;
    return c;
  }
  function arrowLine(g, x0, y0, x1, y1, col, w) { AI.arrow(g, x0, y0, x1, y1, col, w || 1.5); }
  function dot(g, x, y, r, col) { g.fillStyle = col; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); }
  function box(g, x, y, w, h, fill, stroke, r) {
    g.fillStyle = fill; g.strokeStyle = stroke; g.lineWidth = 1.5; g.beginPath();
    if (g.roundRect) g.roundRect(x, y, w, h, r || 6); else g.rect(x, y, w, h);
    g.fill(); g.stroke();
  }

  /* ======================= RNN unrolling ======================= */
  AI.labs.rnnUnroll = function (root) {
    AI.labShell(root, 'RNN unrolling lab', 'hidden state flows forward; the gradient decays backward',
      `<canvas style="height:360px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Inputs x₁ … x_T (comma separated, up to 8)<input type="text" id="xs" value="1, 0.5, -1, 0.8, -0.4" spellcheck="false"></label>
       <label>Recurrent weight w_h <span class="v" id="whv"></span><input type="range" id="wh" min="-2" max="2" step="0.05" value="0.5"></label>
       <label>Input weight w_x <span class="v" id="wxv"></span><input type="range" id="wx" min="-2" max="2" step="0.05" value="1"></label>
       <label>Bias b <span class="v" id="bv"></span><input type="range" id="b" min="-1" max="1" step="0.05" value="0"></label>
       ${transport()}
       <div class="readout" id="out"></div>
       <div class="caption">Cell: h_t = tanh(w_h·h_(t−1) + w_x·x_t + b), h₀ = 0. Gradient bars show ∂h_T/∂h_k on a log scale.</div>`);
    const c = AI.canvas(q(root, 'canvas')), clk = makeClock(root, { phases: 1, phase: 1, hold: 0 });
    let S = null;
    const compute = () => {
      const xs = nums(q(root, '#xs').value).slice(0, 8); if (!xs.length) xs.push(0);
      const wh = +q(root, '#wh').value, wx = +q(root, '#wx').value, b = +q(root, '#b').value;
      q(root, '#whv').textContent = wh; q(root, '#wxv').textContent = wx; q(root, '#bv').textContent = b;
      const hs = [0], loc = [1], pre = [0];
      xs.forEach((x, i) => { const a = wh * hs[i] + wx * x + b; pre.push(a); hs.push(Math.tanh(a)); loc.push(wh * (1 - hs[i + 1] ** 2)); });
      const N = xs.length, g = new Array(N + 1).fill(1);
      for (let k = N - 1; k >= 0; k--) g[k] = g[k + 1] * loc[k + 1];
      S = { xs, hs, loc, pre, g, N, wh };
      clk.phases = 2 * N + 1; clk.phase = 1.2; clk.hold = 1.5; clk.T = 0; clk.dirty = true;
      q(root, '#out').innerHTML = hs.slice(1).map((h, i) => `<span class="m">t=${i + 1}</span> x=${f2(xs[i], 2)} pre=${f2(pre[i + 1], 3)} <span class="b">h=${f2(h, 4)}</span> factor=${f2(loc[i + 1], 4)}`).join('\n') + `\n<span class="o">∂h_T/∂h_0 = ${S.g[0].toExponential(2)}</span>`;
    };
    ['#xs', '#wh', '#wx', '#b'].forEach((s) => (q(root, s).oninput = compute));
    compute(); c.onresize = () => (clk.dirty = true);
    const draw = () => {
      const C = AI.col(), g = c.ctx, N = S.N, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const cw = (W - 40) / N, cx = (k) => 20 + (k - 0.5) * cw, yh = 110, yx = 205, yg = 262;
      const fw = clk.T / clk.phase;
      const fwd = Math.min(N, fw), bwd = Math.max(0, Math.min(N, fw - N));
      const phase = fw < N ? 0 : fw < 2 * N ? 1 : 2;
      const bw0 = Math.min(34, cw / 2 - 6);
      for (let k = 1; k <= N; k++) {
        const x = cx(k), act = fwd >= k - 1 ? Math.min(1, fwd - (k - 1)) : 0, a = ease(act), h = S.hs[k];
        arrowLine(g, x, yx - 8, x, yh + bw0, C.border, 1.5);
        txt(g, 'x' + k + '=' + f2(S.xs[k - 1], 2), x, yx + 10, C.muted, 'center', MONO);
        if (k > 1) {
          const x0 = cx(k - 1) + bw0, x1 = x - bw0; arrowLine(g, x0, yh, x1, yh, C.border, 2);
          if (fwd > k - 2 && fwd < k) { const p = ease(fwd - (k - 2)); dot(g, AI.lerp(x0, x1, p), yh, 5, C.blue); }
        } else txt(g, 'h₀=0', x - bw0, yh - bw0 - 6, C.muted, 'left', MONO);
        box(g, x - bw0, yh - bw0, bw0 * 2, bw0 * 2, a > 0.99 ? AI.mix(C.bg2, h >= 0 ? C.blue : C.orange, Math.min(0.85, Math.abs(h) * 0.9)) : C.bg2, a > 0.01 ? C.blue : C.border);
        txt(g, 'cell', x, yh - bw0 + 15, C.muted, 'center');
        txt(g, a > 0.99 ? 'h' + k + '=' + f2(h, 2) : 'h' + k, x, yh + 5, C.fg, 'center', MONO);
        if (a > 0.99) txt(g, 'slope ' + f2(1 - h * h, 2), x, yh + bw0 - 8, C.muted, 'center', '10px system-ui');
        const reveal = bwd >= N - k ? Math.min(1, bwd - (N - k)) : 0;
        if (reveal > 0) {
          const gk = S.g[k], lg = Math.log10(Math.max(1e-30, Math.abs(gk))), frac = AI.clamp((lg + 12) / 12, 0, 1.2), bw = (cw - 14) * Math.min(1, frac) * ease(reveal);
          g.fillStyle = Math.abs(gk) > 1 ? C.orange : C.red; g.fillRect(x - (cw - 14) / 2, yg, bw, 14);
          g.strokeStyle = C.border; g.strokeRect(x - (cw - 14) / 2, yg, cw - 14, 14);
          txt(g, gk.toExponential(1), x, yg + 30, C.red, 'center', '11px ui-monospace,monospace');
        } else txt(g, '∂/∂h' + k, x, yg + 30, C.border, 'center', '11px ui-monospace,monospace');
      }
      if (bwd > 0 && bwd < N) { const k = N - Math.floor(bwd), p = ease(bwd - Math.floor(bwd)); if (k > 1) dot(g, AI.lerp(cx(k) - bw0, cx(k - 1) + bw0, p), yh + bw0 + 8, 5, C.red); }
      txt(g, 'gradient ∂h_T/∂h_k, log scale: full bar = 1, empty = 1e-12', 20, yg - 8, C.muted, 'left');
      txt(g, phase === 0 ? 'forward in time →' : phase === 1 ? '← backward through time' : 'done', W - 20, 24, phase === 1 ? C.red : C.blue, 'right');
      q(root, '#cap').textContent = phase === 0 ? `① Forward: step ${Math.min(N, Math.floor(fw) + 1)} — the cell mixes the previous state with x and squashes with tanh, passing h to the next step.`
        : phase === 1 ? `② Backward: the error signal is multiplied by w_h·(1−h²) at every step it crosses; the bar shrinks (or grows if |w_h| is large).`
        : `③ Result: ∂h_T/∂h₀ = ${S.g[0].toExponential(2)}. ${Math.abs(S.g[0]) < 1e-2 ? 'Vanishing: early inputs hardly influence the loss.' : Math.abs(S.g[0]) > 10 ? 'Exploding.' : 'Roughly stable.'}`;
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(); } }, { fps: 30 });
  };

  /* ======================= LSTM cell ======================= */
  AI.labs.lstmCell = function (root) {
    const gates = ['f', 'i', 'o', 'g'], names = { f: 'forget', i: 'input', o: 'output', g: 'candidate' };
    const dflt = { f: [0.5, 0.4, 0.1], i: [0.8, -0.3, 0], o: [0.3, 0.9, -0.2], g: [1.0, 0.5, 0] };
    const wRows = gates.map((k) => `<div class="row" style="gap:4px"><b style="width:18px">${k}</b>` + [0, 1, 2].map((j) => `<input type="number" step="0.1" id="w${k}${j}" value="${dflt[k][j]}" style="width:62px">`).join('') + `</div>`).join('');
    AI.labShell(root, 'LSTM cell lab', 'gates open and close with live values',
      `<canvas style="height:380px"></canvas><div class="caption" id="cap"></div>`,
      `<label>x <span class="v" id="xv"></span><input type="range" id="x" min="-3" max="3" step="0.1" value="1"></label>
       <label>h_(t−1) <span class="v" id="hv"></span><input type="range" id="h" min="-1" max="1" step="0.05" value="0.5"></label>
       <label>c_(t−1) <span class="v" id="cv"></span><input type="range" id="c" min="-4" max="4" step="0.1" value="2"></label>
       <div class="caption" style="margin:0">Weights per gate: w_x, w_h, b (scalar cell h = d = 1, 12 parameters)</div>${wRows}
       ${transport()}<div class="readout" id="out"></div>
       <button class="btn small" id="ex">Load the worked example from the lesson</button>`);
    const c = AI.canvas(q(root, 'canvas')), clk = makeClock(root, { phases: 6, phase: 1.6, hold: 1.6 });
    let V = null; const shown = { f: 0, i: 0, o: 0, g: 0 };
    const compute = () => {
      const x = +q(root, '#x').value, h = +q(root, '#h').value, cp = +q(root, '#c').value;
      q(root, '#xv').textContent = x; q(root, '#hv').textContent = h; q(root, '#cv').textContent = cp;
      const w = {}; gates.forEach((k) => (w[k] = [0, 1, 2].map((j) => +q(root, '#w' + k + j).value || 0)));
      const z = {}; gates.forEach((k) => (z[k] = w[k][0] * x + w[k][1] * h + w[k][2]));
      const f = sg(z.f), i = sg(z.i), o = sg(z.o), gg = Math.tanh(z.g), fc = f * cp, ig = i * gg, cn = fc + ig, tc = Math.tanh(cn), hn = o * tc;
      V = { x, h, cp, z, f, i, o, g: gg, fc, ig, cn, tc, hn };
      q(root, '#out').innerHTML = `f=${f2(f)}  i=${f2(i)}  o=${f2(o)}  g=${f2(gg)}\nf·c=${f2(fc)}  i·g=${f2(ig)}\n<span class="g">c_t=${f2(cn)}</span>  tanh(c)=${f2(tc)}\n<span class="b">h_t=${f2(hn)}</span>`;
      clk.dirty = true;
    };
    q(root, '.lab-ctl').addEventListener('input', (e) => { if (e.target.id !== 'sp') compute(); });
    q(root, '#ex').onclick = () => { q(root, '#x').value = 1; q(root, '#h').value = 0.5; q(root, '#c').value = 2; gates.forEach((k) => [0, 1, 2].forEach((j) => (q(root, '#w' + k + j).value = dflt[k][j]))); compute(); clk.T = 0; };
    compute(); c.onresize = () => (clk.dirty = true);
    const draw = (dt) => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, idx = clk.idx(), fr = clk.frac(), done = clk.done();
      g.clearRect(0, 0, W, H);
      const yc = 70, yg = H - 127, gx = { f: 0.17, i: 0.37, g: 0.53, o: 0.8 };
      const pr = (p) => (done ? 1 : idx > p ? 1 : idx === p ? ease(Math.min(1, fr * 1.4)) : 0);
      const upto = done ? 6 : idx;
      const target = { f: V.f * pr(0), i: V.i * pr(0), g: (V.g * 0.5 + 0.5) * pr(1), o: V.o * pr(1) };
      gates.forEach((k) => { const d = target[k] - shown[k]; if (Math.abs(d) > 1e-3) { shown[k] += d * Math.min(1, dt * 6 + 0.02); clk.dirty = true; } else shown[k] = target[k]; });
      g.strokeStyle = C.green; g.lineWidth = 3; g.beginPath(); g.moveTo(14, yc); g.lineTo(W - 70, yc); g.stroke();
      txt(g, 'c_(t−1) = ' + f2(V.cp, 2), 14, yc - 14, C.green, 'left', MONO);
      const xF = W * gx.f, xP = W * 0.66, xIG = W * 0.45, xH = W * 0.8;
      const op = (x, y, sym, on, val) => {
        dot(g, x, y, 15, on ? C.bg3 : C.bg2); g.strokeStyle = on ? C.yellow : C.border; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 15, 0, 7); g.stroke();
        txt(g, sym, x, y + 5, C.fg, 'center', '15px system-ui'); if (val) txt(g, val, x, y - 22, on ? C.yellow : C.muted, 'center', MONO);
      };
      op(xF, yc, '×', idx === 2 && !done, upto > 2 ? f2(V.fc, 2) : '');
      op(xP, yc, '+', idx === 4 && !done, upto > 4 ? f2(V.cn, 2) : '');
      txt(g, upto > 4 ? 'c_t = ' + f2(V.cn, 2) : 'c_t', W - 68, yc - 14, C.green, 'left', MONO);
      op(xIG, yc + 70, '×', idx === 3 && !done, upto > 3 ? f2(V.ig, 2) : '');
      arrowLine(g, xIG + 12, yc + 58, xP - 8, yc + 16, idx === 3 && !done ? C.yellow : C.border, 2);
      const yT = yc + 55, yO = yc + 110;
      g.strokeStyle = C.border; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xH, yc); g.lineTo(xH, yT - 14); g.stroke();
      box(g, xH - 22, yT - 14, 44, 24, C.bg2, idx === 5 && !done ? C.yellow : C.border, 12); txt(g, 'tanh', xH, yT + 3, C.fg, 'center');
      op(xH, yO, '×', idx === 5 && !done, '');
      arrowLine(g, xH, yT + 10, xH, yO - 15, C.border, 1.5);
      arrowLine(g, xH + 15, yO, W - 22, yO, idx === 5 && !done ? C.blue : C.border, 2.5);
      txt(g, upto > 5 ? 'h_t = ' + f2(V.hn, 3) : 'h_t', W - 20, yO - 12, C.blue, 'right', MONO);
      const gp = { f: 0, i: 0, g: 1, o: 1 };
      gates.forEach((k) => {
        const x = W * gx[k], on = gp[k] === idx && !done, bw = 34, bh = 64, y0 = yg - 8;
        box(g, x - bw / 2, y0, bw, bh, C.bg2, on ? C.yellow : C.border, 5);
        const fillH = bh * AI.clamp(shown[k], 0, 1);
        g.fillStyle = k === 'g' ? (V.g >= 0 ? C.purple : C.orange) : C.teal; g.globalAlpha = 0.85; g.fillRect(x - bw / 2 + 2, y0 + bh - fillH - 2, bw - 4, Math.max(0, fillH)); g.globalAlpha = 1;
        const rv = pr(gp[k]) > 0.5; const gv = V[k];
        txt(g, k + (rv ? ' = ' + f2(gv, 3) : ''), x, y0 + bh + 16, rv ? C.fg : C.muted, 'center', MONO);
        txt(g, names[k] + (k === 'g' ? ' (tanh)' : ' (σ)'), x, y0 + bh + 31, C.muted, 'center');
        const tx = k === 'f' ? xF : k === 'o' ? xH : k === 'i' ? xIG - 10 : xIG + 10, ty = k === 'f' ? yc + 15 : k === 'o' ? yO + 15 : yc + 86;
        g.strokeStyle = on ? C.yellow : C.border; g.lineWidth = on ? 2.5 : 1.5; g.beginPath(); g.moveTo(x, y0); g.lineTo(x, ty + 30); g.lineTo(tx, ty); g.stroke();
      });
      txt(g, `x=${f2(V.x, 2)}  h_(t−1)=${f2(V.h, 2)}   gate input = w_x·x + w_h·h + b`, 14, H - 6, C.muted, "left", MONO);
      const caps = ['① Forget and input gates: σ(w_x·x + w_h·h + b) squashes each to (0,1). The valve height is how open it is.',
        '② The candidate g = tanh(…) (new content in −1..1) and the output gate o are computed from the same x and h.',
        `③ Forget: keep f·c_(t−1) = ${f2(V.f, 2)}·${f2(V.cp, 2)} = ${f2(V.fc, 3)} of the old memory.`,
        `④ Write: add i·g = ${f2(V.i, 2)}·${f2(V.g, 2)} = ${f2(V.ig, 3)} of the new content.`,
        `⑤ New cell state c_t = ${f2(V.fc, 3)} + ${f2(V.ig, 3)} = ${f2(V.cn, 3)}.`,
        `⑥ Read: h_t = o·tanh(c_t) = ${f2(V.o, 2)}·${f2(V.tc, 3)} = ${f2(V.hn, 3)}.`];
      q(root, '#cap').textContent = done ? `Done: c_t = ${f2(V.cn, 3)}, h_t = ${f2(V.hn, 3)}. Edit any input or weight to see the gates react.` : caps[idx];
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(dt); } }, { fps: 30 });
  };

  /* ======================= Attention ======================= */
  AI.labs.attnLab = function (root) {
    AI.labShell(root, 'Self-attention lab', 'queries, keys, values, softmax weights as arcs, weighted sum',
      `<canvas style="height:400px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Token names<input type="text" id="nm" value="A, B, C" spellcheck="false"></label>
       <label>Embeddings X (one token per row; rows separated by ;)<input type="text" id="X" value="1,0; 0,1; 1,1" spellcheck="false"></label>
       <label>W_Q (d × d_k)<input type="text" id="WQ" value="1,0; 0,1" spellcheck="false"></label>
       <label>W_K<input type="text" id="WK" value="1,1; 0,1" spellcheck="false"></label>
       <label>W_V<input type="text" id="WV" value="1,0; 1,1" spellcheck="false"></label>
       <div class="err" id="err"></div>
       <div class="row"><button class="btn small" id="ex">Lesson example</button><button class="btn small" id="syn">Two similar tokens</button><button class="btn small" id="sharp">Sharper (big queries)</button><button class="btn small" id="free">Cycle all queries</button></div>
       ${transport()}<div class="readout" id="out"></div>
       <div class="caption">Click a token to fix it as the query. Scores are q·k/√d_k, weights are the softmax of each row, output = Σ weight·v.</div>`);
    const c = AI.canvas(q(root, 'canvas')), clk = makeClock(root, { phases: 12, phase: 1.5, hold: 1 });
    let M = null, fixed = -1;
    const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    const compute = () => {
      try {
        const X = mat(q(root, '#X').value).slice(0, 6), WQ = mat(q(root, '#WQ').value), WK = mat(q(root, '#WK').value), WV = mat(q(root, '#WV').value);
        if (!X.length || !WQ.length || !WK.length || !WV.length) throw new Error('fill in X and the three W matrices');
        const d = X[0].length;
        if (X.some((r) => r.length !== d)) throw new Error('every embedding row needs the same length');
        [WQ, WK, WV].forEach((W) => { if (W.length !== d) throw new Error('each W needs ' + d + ' rows (= embedding size)'); if (W.some((r) => r.length !== W[0].length)) throw new Error('W rows must have equal length'); });
        if (WQ[0].length !== WK[0].length) throw new Error('W_Q and W_K must have the same number of columns (d_k)');
        const n = X.length, dk = WQ[0].length, Q = mm(X, WQ), K = mm(X, WK), V = mm(X, WV);
        const S = Q.map((qq) => K.map((k) => qq.reduce((s, v, i) => s + v * k[i], 0) / Math.sqrt(dk)));
        const A = S.map((r) => { const m = Math.max(...r), e = r.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map((v) => v / s); });
        const O = A.map((a) => V[0].map((_, j) => a.reduce((s, w, t) => s + w * V[t][j], 0)));
        const names = q(root, '#nm').value.split(',').map((s) => s.trim()); const nm = X.map((_, i) => names[i] || 'T' + (i + 1));
        M = { n, d, dk, Q, K, V, S, A, O, nm }; q(root, '#err').textContent = '';
        clk.phases = 4 * n; clk.T = 0; fixed = Math.min(fixed, n - 1);
        q(root, '#out').innerHTML = A.map((r, i) => `<span class="m">${nm[i]}</span> w=(${r.map((v) => f2(v, 3)).join(', ')}) → <span class="b">(${O[i].map((v) => f2(v, 3)).join(', ')})</span>`).join('\n');
        clk.dirty = true;
      } catch (e) { q(root, '#err').textContent = e.message; }
    };
    ['#nm', '#X', '#WQ', '#WK', '#WV'].forEach((s) => (q(root, s).oninput = compute));
    const set = (x, a, b, cc, nmv) => { q(root, '#X').value = x; q(root, '#WQ').value = a; q(root, '#WK').value = b; q(root, '#WV').value = cc; q(root, '#nm').value = nmv || 'A, B, C'; compute(); };
    q(root, '#ex').onclick = () => set('1,0; 0,1; 1,1', '1,0; 0,1', '1,1; 0,1', '1,0; 1,1');
    q(root, '#syn').onclick = () => set('1,0; 0.9,0.1; 0,1', '1,0; 0,1', '1,0; 0,1', '1,0; 0,1', 'cat, kitten, car');
    q(root, '#sharp').onclick = () => set('1,0; 0,1; 1,1', '3,0; 0,3', '1,1; 0,1', '1,0; 1,1');
    q(root, '#free').onclick = () => { fixed = -1; clk.T = 0; clk.dirty = true; };
    compute(); c.onresize = () => (clk.dirty = true);
    const pos = (i, W) => { const L = W * 0.6; return 40 + (M.n === 1 ? L / 2 : (i * (L - 70)) / (M.n - 1)); };
    c.cv.addEventListener('click', (e) => {
      const r = c.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      if (!M || y < c.h - 150) return;
      for (let i = 0; i < M.n; i++) if (Math.abs(x - pos(i, c.w)) < 28) { fixed = i; clk.T = 0; clk.dirty = true; return; }
    });
    const draw = () => {
      if (!M) return;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, n = M.n, idx = clk.idx(), sub = clk.done() ? 3 : idx % 4, fr = clk.done() ? 1 : clk.frac();
      const qi = Math.min(n - 1, fixed >= 0 ? fixed : clk.done() ? n - 1 : Math.floor(idx / 4));
      g.clearRect(0, 0, W, H);
      const yT = H - 130, bw = 52, f1 = (v) => '(' + v.map((a) => f2(a, 2)).join(',') + ')';
      const fe = ease(fr);
      for (let j = 0; j < n; j++) {
        const x0 = pos(qi, W), x1 = pos(j, W), w = M.A[qi][j], sc = M.S[qi][j];
        if (sub === 0) continue;
        const hgt = 30 + Math.abs(x1 - x0) * 0.3;
        const wid = sub === 1 ? 2 : AI.lerp(2, 2 + 14 * w, sub === 2 ? fe : 1);
        g.globalAlpha = sub === 1 ? 0.75 * fe : sub === 2 ? 0.3 + 0.7 * (1 - fe) * 0.4 + 0.7 * fe * (0.3 + w) : 0.3 + 0.7 * w;
        g.globalAlpha = AI.clamp(g.globalAlpha, 0, 1);
        g.strokeStyle = sub === 1 ? C.muted : C.purple; g.lineWidth = wid;
        g.beginPath();
        if (j === qi) g.ellipse(x0, yT - 16, 14, 20, 0, 0, Math.PI * 2);
        else { g.moveTo(x0, yT); g.quadraticCurveTo((x0 + x1) / 2, yT - hgt * 2, x1, yT); }
        g.stroke(); g.globalAlpha = 1;
        const lx = j === qi ? x0 : (x0 + x1) / 2, ly = j === qi ? yT - 44 : yT - hgt - 2;
        if (sub >= 1) txt(g, sub === 1 ? 's=' + f2(sc, 2) : 'w=' + f2(w, 2), lx, ly, sub === 1 ? C.muted : C.purple, 'center', MONO);
        if (sub === 3 && j !== qi) { // value particle travelling back along the arc
          const t = fe, bx = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * ((x0 + x1) / 2) + t * t * x0, by = (1 - t) * (1 - t) * yT + 2 * (1 - t) * t * (yT - hgt * 2) + t * t * yT;
          dot(g, bx, by, 3 + 8 * w, C.orange);
        }
      }
      for (let i = 0; i < n; i++) {
        const x = pos(i, W), isq = i === qi;
        box(g, x - bw / 2, yT + 6, bw, 36, isq ? AI.mix(C.bg2, C.blue, 0.3) : C.bg2, isq ? C.blue : C.border, 8);
        txt(g, M.nm[i], x, yT + 29, C.fg, 'center', '13px system-ui');
        txt(g, 'q' + f1(M.Q[i]), x, yT + 58, isq ? C.blue : C.muted, 'center', '10px ui-monospace,monospace');
        txt(g, 'k' + f1(M.K[i]), x, yT + 72, C.green, 'center', '10px ui-monospace,monospace');
        txt(g, 'v' + f1(M.V[i]), x, yT + 86, C.orange, 'center', '10px ui-monospace,monospace');
      }
      if (sub === 3) { g.globalAlpha = fe; txt(g, 'output for ' + M.nm[qi] + ' = (' + M.O[qi].map((v) => f2(v, 3)).join(', ') + ')', W * 0.3, 44, C.orange, 'center', '13px ui-monospace,monospace'); g.globalAlpha = 1; }
      txt(g, 'query token: ' + M.nm[qi], 12, 14, C.blue, 'left', '12px system-ui');
      const mx0 = W * 0.66, my0 = 40, cs = Math.max(14, Math.min(44, (W * 0.32) / n, 200 / n));
      txt(g, 'attention weights (rows = queries)', mx0, 24, C.muted, 'left', '11px system-ui');
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const cur = i === qi, vis = fixed >= 0 ? (cur ? (sub >= 2 ? (sub === 2 ? fe : 1) : 0) : 0.35) : (i < qi ? 1 : cur ? (sub >= 2 ? (sub === 2 ? fe : 1) : 0) : 0), w = M.A[i][j];
        g.fillStyle = C.bg2; g.fillRect(mx0 + j * cs, my0 + i * cs, cs - 2, cs - 2);
        g.globalAlpha = Math.min(1, vis * 0.9 * (0.15 + w)); g.fillStyle = C.purple; g.fillRect(mx0 + j * cs, my0 + i * cs, cs - 2, cs - 2); g.globalAlpha = 1;
        if (vis > 0.5) txt(g, f2(w, 2), mx0 + j * cs + cs / 2 - 1, my0 + i * cs + cs / 2 + 3, C.fg, 'center', '10px ui-monospace,monospace');
      }
      g.strokeStyle = C.blue; g.lineWidth = 2; g.strokeRect(mx0 - 1, my0 + qi * cs - 1, n * cs, cs);
      for (let j = 0; j < n; j++) txt(g, M.nm[j], mx0 + j * cs + cs / 2, my0 + n * cs + 12, C.muted, 'center', '10px system-ui');
      const caps = [`① Query: ${M.nm[qi]} asks "who is relevant to me?" with q = x·W_Q = (${M.Q[qi].map((v) => f2(v, 2)).join(', ')}).`,
        `② Scores: dot product of q with every key, divided by √d_k = ${f2(Math.sqrt(M.dk), 3)}. Bigger score = more similar.`,
        `③ Softmax: scores become weights that sum to 1 — arc thickness is the weight.`,
        `④ Weighted sum: each value v flows to ${M.nm[qi]} in proportion to its weight; output = (${M.O[qi].map((v) => f2(v, 3)).join(', ')}).`];
      q(root, '#cap').textContent = caps[sub];
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(); } }, { fps: 30 });
  };

  /* ======================= RNN vs transformer ======================= */
  AI.labs.seqCompare = function (root) {
    AI.labShell(root, 'Sequence-model comparison', 'how far must information travel, and what runs in parallel?',
      `<canvas id="a" style="height:150px"></canvas><canvas id="b" style="height:190px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Sequence length n <span class="v" id="nv"></span><input type="range" id="n" min="4" max="14" step="1" value="8"></label>
       <label>Embedding size d (for the cost line) <span class="v" id="dv"></span><input type="range" id="d" min="16" max="1024" step="16" value="64"></label>
       ${transport()}<div class="readout" id="out"></div>
       <div class="caption">Question: how does token 1 reach token n? Orange = the information path from the first token to the last.</div>`);
    const ca = AI.canvas(q(root, '#a')), cb = AI.canvas(q(root, '#b')), clk = makeClock(root, { phases: 10, phase: 0.9, hold: 2 });
    let n = 8;
    const upd = () => {
      n = +q(root, '#n').value; const d = +q(root, '#d').value; q(root, '#nv').textContent = n; q(root, '#dv').textContent = d;
      clk.phases = n + 1; clk.T = 0; clk.dirty = true;
      q(root, '#out').innerHTML = `<span class="b">RNN</span>  path ${n - 1}, ${n} seq. steps\n<span class="g">Attn</span> path 1, 1 step/layer\nn² = ${n * n} pairs\n2·n²·d ≈ ${(2 * n * n * d).toLocaleString()} MACs`;
    };
    q(root, '#n').oninput = upd; q(root, '#d').oninput = upd; upd();
    ca.onresize = cb.onresize = () => (clk.dirty = true);
    const draw = () => {
      const C = AI.col(), s = clk.T / clk.phase;
      const px = (W, i) => 30 + (i * (W - 60)) / (n - 1);
      let g = ca.ctx, W = ca.w, H = ca.h; g.clearRect(0, 0, W, H);
      txt(g, 'RNN / LSTM: strictly sequential — step t needs step t−1', 10, 16, C.blue, 'left');
      const y = 85, pos = Math.min(n - 1, s);
      for (let i = 0; i < n; i++) {
        if (i > 0) arrowLine(g, px(W, i - 1) + 12, y, px(W, i) - 12, y, i <= pos + 0.001 ? C.orange : C.border, 2.5);
        const lit = i <= pos + 1e-6; dot(g, px(W, i), y, 11, lit ? C.orange : C.bg3); g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.arc(px(W, i), y, 11, 0, 7); g.stroke();
        txt(g, '' + (i + 1), px(W, i), y + 4, lit ? '#000' : C.muted, 'center', MONO);
      }
      const fl = Math.floor(pos), pf = pos - fl; if (pos < n - 1) dot(g, AI.lerp(px(W, fl) + 11, px(W, fl + 1) - 11, ease(pf)), y, 5, C.yellow);
      txt(g, 'steps used: ' + Math.min(n - 1, Math.ceil(s)) + ' of ' + (n - 1) + (pos >= n - 1 ? '  ✓ token 1 reached token ' + n : ''), 10, 140, C.fg, 'left', MONO);
      g = cb.ctx; W = cb.w; H = cb.h; g.clearRect(0, 0, W, H);
      txt(g, 'Transformer: every token attends to every token in one parallel step', 10, 16, C.green, 'left');
      const y2 = 140, t = AI.clamp(s, 0, 1), a = ease(t);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const hot = i === 0 && j === n - 1;
        g.globalAlpha = (hot ? 1 : 0.18) * a; g.strokeStyle = hot ? C.orange : C.green; g.lineWidth = hot ? 3 : 1;
        g.beginPath(); g.moveTo(px(W, i), y2 - 11); g.quadraticCurveTo((px(W, i) + px(W, j)) / 2, y2 - 12 - (j - i) * 11, px(W, j), y2 - 11); g.stroke(); g.globalAlpha = 1;
      }
      if (t > 0 && t < 1) { const p = ease(t), x0 = px(W, 0), x1 = px(W, n - 1), cy = y2 - 12 - (n - 1) * 11; dot(g, (1 - p) * (1 - p) * x0 + 2 * (1 - p) * p * ((x0 + x1) / 2) + p * p * x1, (1 - p) * (1 - p) * (y2 - 11) + 2 * (1 - p) * p * cy + p * p * (y2 - 11), 5, C.yellow); }
      for (let i = 0; i < n; i++) { const lit = t >= 1 || (t > 0 && i === 0); dot(g, px(W, i), y2, 11, lit ? C.green : C.bg3); g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.arc(px(W, i), y2, 11, 0, 7); g.stroke(); txt(g, '' + (i + 1), px(W, i), y2 + 4, lit ? '#000' : C.muted, 'center', MONO); }
      txt(g, 'steps used: ' + (s > 0 ? 1 : 0) + ' of 1' + (t >= 1 ? '  ✓ all ' + (n * (n - 1)) / 2 + ' pairs computed together' : ''), 10, H - 8, C.fg, 'left', MONO);
      q(root, '#cap').textContent = s < n - 1 ? (s < 1 ? '① Both start: the RNN hands token 1 to token 2; the transformer already connects all pairs at once.' : `② The RNN pulse is at token ${Math.min(n, Math.floor(s) + 1)} — ${n - 1 - Math.floor(s)} more sequential steps to go; the transformer finished in 1.`) : `③ Done: RNN path length ${n - 1} (the gradient crosses ${n - 1} cells), transformer path length 1 — but it paid n² = ${n * n} pair computations.`;
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(); } }, { fps: 30 });
  };

  /* ======================= Embedding space ======================= */
  AI.labs.embedSpace = function (root) {
    const PTS = [
      ['cat', 0.92, 0.4, 0], ['dog', 0.85, 0.58, 0], ['lion', 0.97, 0.22, 0], ['horse', 0.8, 0.12, 0],
      ['apple', -0.3, 0.92, 1], ['pear', -0.12, 0.8, 1], ['banana', -0.52, 0.82, 1], ['grape', -0.45, 1.0, 1],
      ['car', -0.92, -0.25, 2], ['bus', -0.82, -0.5, 2], ['bike', -0.98, -0.05, 2], ['train', -0.7, -0.72, 2],
      ['man', 0.05, -0.8, 3], ['woman', 0.45, -0.75, 3], ['king', 0.1, -0.35, 3], ['queen', 0.5, -0.3, 3],
    ];
    const pts = PTS.map((p) => ({ n: p[0], x: p[1], y: p[2], g: p[3] }));
    const opts = pts.map((p, i) => `<option value="${i}">${p.n}</option>`).join('');
    AI.labShell(root, 'Embedding space lab', '2-D embeddings (illustrative): drag points, search neighbours, do analogies',
      `<canvas style="height:420px;touch-action:none;cursor:grab"></canvas><div class="caption" id="cap"></div>`,
      `<label>Similarity metric<select id="met"><option value="cos">cosine (angle from origin)</option><option value="euc">Euclidean distance</option></select></label>
       <label>Neighbours k <span class="v" id="kv"></span><input type="range" id="k" min="1" max="6" step="1" value="3"></label>
       <div class="caption" style="margin:0">Click a point to search; drag to move it.</div>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="an" checked> Show analogy  a − b + c</label>
       <div class="row"><select id="sa" style="flex:1">${opts}</select><select id="sb" style="flex:1">${opts}</select><select id="sc" style="flex:1">${opts}</select></div>
       ${transport()}<div class="readout" id="out"></div>`);
    q(root, '#sa').value = 14; q(root, '#sb').value = 12; q(root, '#sc').value = 13;
    const cv = q(root, 'canvas'), c = AI.canvas(cv), clk = makeClock(root, { phases: 5, phase: 1.6, hold: 1.5 });
    let sel = 14, drag = -1;
    const dotp = (a, b) => a[0] * b[0] + a[1] * b[1];
    const sim = (a, b, met) => (met === 'cos' ? dotp(a, b) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]) || 1) : -Math.hypot(a[0] - b[0], a[1] - b[1]));
    const near = (v, excl) => { const met = q(root, '#met').value; return pts.map((p, i) => ({ i, s: sim(v, [p.x, p.y], met) })).filter((o) => !excl.includes(o.i)).sort((a, b) => b.s - a.s).slice(0, +q(root, '#k').value); };
    const geo = () => ({ s: Math.min(c.w, c.h) / 2.7, cx: c.w / 2, cy: c.h / 2 });
    const toPx = (x, y) => { const G = geo(); return [G.cx + x * G.s, G.cy - y * G.s]; };
    const fromPx = (px, py) => { const G = geo(); return [(px - G.cx) / G.s, (G.cy - py) / G.s]; };
    const evt = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const hit = (px, py) => { let b = -1, bd = 18; pts.forEach((p, i) => { const [x, y] = toPx(p.x, p.y), d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; b = i; } }); return b; };
    cv.addEventListener('pointerdown', (e) => { const [px, py] = evt(e), h = hit(px, py); if (h >= 0) { drag = h; sel = h; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; clk.T = 0; clk.dirty = true; } });
    cv.addEventListener('pointermove', (e) => { if (drag < 0) return; const [px, py] = evt(e), [x, y] = fromPx(px, py); pts[drag].x = AI.clamp(x, -1.25, 1.25); pts[drag].y = AI.clamp(y, -1.25, 1.25); clk.dirty = true; });
    const up = () => { drag = -1; cv.style.cursor = 'grab'; clk.dirty = true; };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    ['#met', '#k', '#an', '#sa', '#sb', '#sc'].forEach((s) => (q(root, s).oninput = () => { q(root, '#kv').textContent = q(root, '#k').value; clk.T = 0; clk.dirty = true; }));
    q(root, '#kv').textContent = 3; c.onresize = () => (clk.dirty = true);
    const star = (g, x, y, R, col) => { g.fillStyle = col; g.beginPath(); for (let k = 0; k < 10; k++) { const rad = k % 2 ? R / 2 : R, an = (k * Math.PI) / 5 - Math.PI / 2; g.lineTo(x + rad * Math.cos(an), y + rad * Math.sin(an)); } g.closePath(); g.fill(); };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, G = geo(), met = q(root, '#met').value, GC = [C.blue, C.green, C.orange, C.purple];
      g.clearRect(0, 0, W, H);
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(0, G.cy); g.lineTo(W, G.cy); g.moveTo(G.cx, 0); g.lineTo(G.cx, H); g.stroke();
      const sp = (p) => toPx(p.x, p.y), idx = clk.idx(), fr = clk.frac(), done = clk.done();
      const [sx, sy] = sp(pts[sel]), kk = near([pts[sel].x, pts[sel].y], [sel]);
      const rr = ease(done ? 1 : idx === 0 ? fr : 1);
      if (met === 'euc' && kk.length) { const far = Math.max(...kk.map((o) => Math.hypot(pts[o.i].x - pts[sel].x, pts[o.i].y - pts[sel].y))); g.strokeStyle = C.yellow; g.globalAlpha = 0.5; g.lineWidth = 1.5; g.beginPath(); g.arc(sx, sy, far * G.s * rr, 0, 7); g.stroke(); g.globalAlpha = 1; }
      if (met === 'cos') { g.strokeStyle = C.yellow; g.globalAlpha = 0.45; g.lineWidth = 1.5; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(G.cx, G.cy); g.lineTo(G.cx + (sx - G.cx) * 1.8, G.cy + (sy - G.cy) * 1.8); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
      kk.forEach((o) => { const [x, y] = sp(pts[o.i]); g.strokeStyle = C.yellow; g.lineWidth = 2; g.globalAlpha = 0.8 * rr; g.beginPath(); g.moveTo(sx, sy); g.lineTo(AI.lerp(sx, x, rr), AI.lerp(sy, y, rr)); g.stroke(); g.globalAlpha = 1; });
      let ares = null;
      const ia = +q(root, '#sa').value, ib = +q(root, '#sb').value, ic = +q(root, '#sc').value;
      if (q(root, '#an').checked) {
        const a = pts[ia], b = pts[ib], cc = pts[ic];
        ares = [a.x - b.x + cc.x, a.y - b.y + cc.y];
        const [ax, ay] = sp(a), mid = toPx(a.x - b.x + 0, a.y - b.y), [rx, ry] = toPx(ares[0], ares[1]);
        const origin = toPx(0, 0), midv = [ax - (toPx(b.x, b.y)[0] - origin[0]), ay - (toPx(b.x, b.y)[1] - origin[1])];
        const ph = done ? 4 : idx - 1, f = done ? 1 : fr;
        if (ph >= 0) { const p1 = ph === 0 ? ease(f) : 1; arrowLine(g, ax, ay, AI.lerp(ax, midv[0], p1), AI.lerp(ay, midv[1], p1), C.red, 2.5); txt(g, '− ' + b.n, (ax + midv[0]) / 2 + 6, (ay + midv[1]) / 2, C.red, 'left', '12px system-ui'); }
        if (ph >= 1) { const p2 = ph === 1 ? ease(f) : 1; arrowLine(g, midv[0], midv[1], AI.lerp(midv[0], rx, p2), AI.lerp(midv[1], ry, p2), C.green, 2.5); txt(g, '+ ' + cc.n, (midv[0] + rx) / 2 + 6, (midv[1] + ry) / 2, C.green, 'left', '12px system-ui'); }
        if (ph >= 2) star(g, rx, ry, 9 + 2 * Math.sin(clk.T * 3), C.yellow);
        void mid;
      }
      pts.forEach((p, i) => { const [x, y] = sp(p), isq = i === sel, isn = kk.some((o) => o.i === i); dot(g, x, y, isq ? 8 : isn ? 7 : 5.5, GC[p.g]); if (isq || isn) { g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); g.arc(x, y, isq ? 12 : 10, 0, 7); g.stroke(); } txt(g, p.n, x + 9, y - 8, isq ? C.fg : C.muted, 'left', '11px system-ui'); });
      let out = `query: <span class="b">${pts[sel].n}</span>\n` + kk.map((o, r) => `${r + 1}. ${pts[o.i].n.padEnd(7)} ${met === 'cos' ? 'cos' : '−dist'} = ${f2(o.s, 3)}`).join('\n');
      let cap = `Searching: ${pts[sel].n}'s ${kk.length} nearest neighbours by ${met === 'cos' ? 'cosine' : 'distance'} are highlighted.`;
      if (ares) {
        const an = near(ares, [ia, ib, ic]);
        out += `\n\nanalogy: ${pts[ia].n} − ${pts[ib].n} + ${pts[ic].n} = (${f2(ares[0], 2)}, ${f2(ares[1], 2)})\n<span class="g">nearest: ${an[0] ? pts[an[0].i].n : '-'}</span>`;
        const ph = done ? 4 : idx;
        cap = ['① Search ripple grows from the selected point to its nearest neighbours.', `② Analogy: start at ${pts[ia].n}.`, `③ Subtract ${pts[ib].n}: the arrow is the direction "${pts[ia].n} minus ${pts[ib].n}".`, `④ Add ${pts[ic].n}: move by that vector.`, `⑤ The star is the result; its nearest word is ${an[0] ? pts[an[0].i].n : '-'}.`][ph] || cap;
      }
      q(root, '#out').innerHTML = out; q(root, '#cap').textContent = cap;
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(); } }, { fps: 30 });
  };

  /* ======================= Autoencoder bottleneck ======================= */
  let AEC = null; // cached PCA of the synthetic image set (illustrative)
  function aeData() {
    if (AEC) return AEC;
    const r = AI.rng(11), N = 240, D = 64, X = [];
    for (let s = 0; s < N; s++) {
      const cx = 2 + 3 * r(), cy = 2 + 3 * r(), sd = 1.0 + 0.7 * r(), amp = 0.6 + 0.4 * r(), img = new Float64Array(D);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) img[y * 8 + x] = amp * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sd * sd));
      X.push(img);
    }
    const mean = new Float64Array(D); X.forEach((v) => v.forEach((a, i) => (mean[i] += a / N)));
    const Cm = Array.from({ length: D }, () => new Float64Array(D));
    X.forEach((v) => { for (let i = 0; i < D; i++) { const a = v[i] - mean[i]; for (let j = 0; j < D; j++) Cm[i][j] += (a * (v[j] - mean[j])) / N; } });
    let trace = 0; for (let i = 0; i < D; i++) trace += Cm[i][i];
    const comps = [], lam = [];
    for (let k = 0; k < 16; k++) {
      let v = new Float64Array(D).map((_, i) => Math.sin(i * 1.3 + k * 2.1) + 0.1), l = 0;
      for (let it = 0; it < 80; it++) {
        const w = new Float64Array(D); for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += Cm[i][j] * v[j]; w[i] = s; }
        l = Math.hypot(...w) || 1e-12; v = w.map((a) => a / l);
      }
      comps.push(v); lam.push(l);
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) Cm[i][j] -= l * v[i] * v[j];
    }
    const mse = [trace / D]; let acc = trace;
    for (let k = 0; k < 16; k++) { acc -= lam[k]; mse.push(Math.max(0, acc) / D); }
    AEC = { X, mean, comps, lam, mse, trace };
    return AEC;
  }

  AI.labs.autoencoderLab = function (root) {
    AI.labShell(root, 'Autoencoder bottleneck lab', 'illustrative: the optimal linear autoencoder equals PCA on 8×8 images',
      `<canvas style="height:430px"></canvas><div class="caption" id="cap"></div>`,
      `<label>Bottleneck size k (code length) <span class="v" id="kv"></span><input type="range" id="k" min="1" max="16" step="1" value="3"></label>
       <div class="caption" style="margin:0">Input 64 numbers → code of k numbers → 64 numbers. Compression = 64 / k.</div>
       <label>Latent traversal: component <span class="v" id="jv"></span><input type="range" id="j" min="1" max="16" step="1" value="1"></label>
       <label>Latent value z <span class="v" id="zv"></span><input type="range" id="z" min="-3" max="3" step="0.1" value="0"></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="sw" checked> Auto-sweep z</label>
       ${transport()}<div class="readout" id="out"></div>`);
    const c = AI.canvas(q(root, 'canvas')), clk = makeClock(root, { phases: 4, phase: 1.7, hold: 0.5 });
    let samp = 0, sweepT = 0;
    const D = aeData();
    const upd = () => { clk.dirty = true; };
    ['#k', '#j', '#z'].forEach((s) => (q(root, s).oninput = upd)); q(root, '#sw').onchange = upd; c.onresize = upd;
    q(root, '#z').addEventListener('pointerdown', () => { q(root, '#sw').checked = false; });
    const img = (g, x, y, s, arr, C, label) => {
      const cs = s / 8;
      for (let i = 0; i < 64; i++) { const v = AI.clamp(arr[i], 0, 1); g.fillStyle = AI.mix(C.bg2, C.fg, v); g.fillRect(x + (i % 8) * cs, y + Math.floor(i / 8) * cs, cs - 0.5, cs - 0.5); }
      txt(g, label, x + s / 2, y - 6, C.muted, 'center', '12px system-ui');
    };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, k = +q(root, '#k').value, j = Math.min(k, +q(root, '#j').value) - 1, idx = clk.idx(), fr = clk.frac();
      q(root, '#kv').textContent = k + '  (×' + f2(64 / k, 3) + ')'; q(root, '#jv').textContent = j + 1;
      if (q(root, '#sw').checked) q(root, '#z').value = (2.5 * Math.sin(sweepT * 0.5)).toFixed(1);
      const zv = +q(root, '#z').value; q(root, '#zv').textContent = zv;
      g.clearRect(0, 0, W, H);
      const x = D.X[samp % D.X.length], zs = D.comps.map((v) => v.reduce((s, a, i) => s + a * (x[i] - D.mean[i]), 0));
      const shownK = idx < 2 ? 0 : idx === 2 ? Math.round(k * ease(fr)) : k;
      const rec = Array.from(D.mean); for (let m = 0; m < shownK; m++) for (let i = 0; i < 64; i++) rec[i] += zs[m] * D.comps[m][i];
      const s = Math.max(60, Math.min(120, (W - 80) / 4.4)), y0 = 36;
      img(g, 20, y0, s, x, C, 'input x (64 values)');
      const bx = 20 + s + 36, bw = Math.max(60, Math.min(120, W - 4.4 * s - 40));
      txt(g, 'code z (k = ' + k + ')', bx + bw / 2, y0 - 6, C.muted, 'center');
      const bh = s / k;
      for (let m = 0; m < k; m++) {
        const reveal = idx > 1 ? 1 : idx === 1 ? AI.clamp(fr * (k + 1) - m, 0, 1) : 0, v = zs[m] / (Math.sqrt(D.lam[0]) * 2.5);
        g.fillStyle = C.bg2; g.fillRect(bx, y0 + m * bh + 1, bw, Math.max(1, bh - 2));
        g.fillStyle = zs[m] >= 0 ? C.teal : C.orange; const w = AI.clamp(v, -1, 1) * (bw / 2) * ease(reveal);
        g.fillRect(bx + bw / 2, y0 + m * bh + 1, w, Math.max(1, bh - 2));
      }
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(bx + bw / 2, y0); g.lineTo(bx + bw / 2, y0 + s); g.stroke();
      arrowLine(g, 20 + s + 4, y0 + s / 2, bx - 4, y0 + s / 2, idx === 1 ? C.yellow : C.border, 2);
      const rx = bx + bw + 30; arrowLine(g, bx + bw + 4, y0 + s / 2, rx - 4, y0 + s / 2, idx === 2 ? C.yellow : C.border, 2);
      img(g, rx, y0, s, rec, C, 'reconstruction x̂');
      const tx = rx + s + 20;
      if (tx + s < W - 4) { const tv = Array.from(D.mean); for (let i = 0; i < 64; i++) tv[i] += zv * Math.sqrt(D.lam[j]) * D.comps[j][i]; img(g, tx, y0, s, tv, C, 'latent traversal'); txt(g, 'comp ' + (j + 1) + ', z=' + zv, tx + s / 2, y0 + s + 14, C.muted, 'center', MONO); }
      const cx0 = 50, cy0 = y0 + s + 56, cw = W - 90, ch = Math.max(60, H - cy0 - 36);
      txt(g, 'reconstruction error (MSE per pixel) vs bottleneck size k', cx0, cy0 - 12, C.muted, 'left');
      const maxE = D.mse[0];
      g.strokeStyle = C.border; g.lineWidth = 1; g.strokeRect(cx0, cy0, cw, ch);
      g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath();
      D.mse.forEach((e, m) => { const px = cx0 + (m / 16) * cw, py = cy0 + ch - (e / maxE) * ch; if (m) g.lineTo(px, py); else g.moveTo(px, py); }); g.stroke();
      const mx = cx0 + (k / 16) * cw, my = cy0 + ch - (D.mse[k] / maxE) * ch; dot(g, mx, my, 6, C.orange);
      g.strokeStyle = C.orange; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(mx, cy0 + ch); g.lineTo(mx, my); g.stroke(); g.setLineDash([]);
      for (let m = 0; m <= 16; m += 2) txt(g, '' + m, cx0 + (m / 16) * cw, cy0 + ch + 14, C.muted, 'center', '10px ui-monospace,monospace');
      const err = (D.mse[k] / maxE) * 100;
      q(root, '#out').innerHTML = `sample #${samp % D.X.length + 1}\ncode length k = ${k}\ncompression = 64/${k} = <span class="o">${f2(64 / k, 3)}×</span>\nerror left = <span class="r">${f2(err, 3)}%</span>\nparams (linear AE) = ${64 * k + k + k * 64 + 64}`;
      q(root, '#cap').textContent = ['① Input: an 8×8 image is 64 numbers (a blob at an unknown position and size).', `② Encode: squeeze the 64 numbers through ${k} code values (bars).`, `③ Decode: rebuild 64 numbers from the code, one code direction added at a time.`, `④ Compare x̂ with x. Small k loses detail (see the error curve); the traversal image shows what one code value controls.`][idx];
    };
    AI.loop(root, (dt) => {
      const before = clk.T; clk.update(dt); if (clk.T < before) samp++;
      if (q(root, '#sw').checked) { sweepT += dt * clk.speed; clk.dirty = true; }
      if (clk.dirty) { clk.dirty = false; draw(); }
    }, { fps: 30 });
  };

  /* ======================= GNN message passing ======================= */
  AI.labs.gnnLab = function (root) {
    AI.labShell(root, 'Graph neural network lab', 'message passing rounds (one GCN layer per round) on a graph you can edit',
      `<canvas style="height:420px;touch-action:none"></canvas><div class="caption" id="cap"></div>`,
      `<div class="caption" style="margin:0">Drag nodes. Click a node to edit its 2-D feature. Edge mode: click two nodes to add or remove an edge.</div>
       <div class="row"><label style="flex:1">feature a<input type="number" id="fa" step="0.1" value="1"></label><label style="flex:1">feature b<input type="number" id="fb" step="0.1" value="0"></label></div>
       <label>W (2 × 2, applied as h·W)<input type="text" id="W" value="1,-1; 0.5,1" spellcheck="false"></label><div class="err" id="err"></div>
       <label>Rounds (layers) <span class="v" id="rv"></span><input type="range" id="R" min="1" max="3" step="1" value="1"></label>
       <div class="row"><button class="btn" id="em">Edge mode: off</button><button class="btn small" id="add">+ node</button><button class="btn small" id="del">− node</button><button class="btn small" id="rst">Reset graph</button></div>
       ${transport()}<div class="readout" id="out"></div>`);
    const cv = q(root, 'canvas'), c = AI.canvas(cv), clk = makeClock(root, { phases: 3, phase: 1.8, hold: 2 });
    const init = () => ({ nodes: [{ x: 0.18, y: 0.3, f: [1, 0] }, { x: 0.45, y: 0.5, f: [0, 1] }, { x: 0.72, y: 0.25, f: [1, 1] }, { x: 0.72, y: 0.8, f: [2, 0] }], edges: [[0, 1], [1, 2], [1, 3], [2, 3]] });
    let G = init(), sel = 1, edgeMode = false, pend = -1, drag = -1, moved = 0, H = [], AG = [], coef = () => 1, ok = false;
    const compute = () => {
      const Wm = mat(q(root, '#W').value);
      if (Wm.length !== 2 || Wm.some((r) => r.length !== 2)) { q(root, '#err').textContent = 'W must be 2×2, e.g. 1,-1; 0.5,1'; return; }
      q(root, '#err').textContent = '';
      const n = G.nodes.length, R = +q(root, '#R').value; q(root, '#rv').textContent = R;
      const adj = Array.from({ length: n }, (_, i) => new Set([i])); G.edges.forEach(([a, b]) => { adj[a].add(b); adj[b].add(a); });
      const deg = adj.map((s) => s.size);
      coef = (u, v) => 1 / Math.sqrt(deg[u] * deg[v]);
      H = [G.nodes.map((nd) => nd.f.slice())]; AG = [];
      for (let r = 0; r < R; r++) {
        const prev = H[r], agg = prev.map((_, v) => { const s = [0, 0]; adj[v].forEach((u) => { const cf = coef(u, v); s[0] += cf * prev[u][0]; s[1] += cf * prev[u][1]; }); return s; });
        AG.push(agg); H.push(agg.map((a) => [0, 1].map((o) => Math.max(0, a[0] * Wm[0][o] + a[1] * Wm[1][o]))));
      }
      G.adj = adj; G.deg = deg; clk.phases = 3 * R; clk.T = 0; clk.dirty = true; ok = true;
      sel = Math.min(sel, n - 1);
      q(root, '#out').innerHTML = `degree (with self-loop): ${deg.join(', ')}\n` + H[R].map((h, i) => `<span class="m">node ${i + 1}</span> ${H[0][i].map((v) => f2(v, 2)).join(',')} → <span class="b">${h.map((v) => f2(v, 3)).join(', ')}</span>`).join('\n');
      if (document.activeElement !== q(root, '#fa') && document.activeElement !== q(root, '#fb')) { q(root, '#fa').value = G.nodes[sel].f[0]; q(root, '#fb').value = G.nodes[sel].f[1]; }
    };
    const px = (nd) => [30 + nd.x * (c.w - 60), 40 + nd.y * (c.h - 110)];
    ['#W', '#R'].forEach((s) => (q(root, s).oninput = compute));
    ['#fa', '#fb'].forEach((s) => (q(root, s).oninput = () => { G.nodes[sel].f = [+q(root, '#fa').value || 0, +q(root, '#fb').value || 0]; compute(); }));
    q(root, '#em').onclick = () => { edgeMode = !edgeMode; pend = -1; q(root, '#em').textContent = 'Edge mode: ' + (edgeMode ? 'on' : 'off'); q(root, '#em').classList.toggle('on', edgeMode); clk.dirty = true; };
    q(root, '#add').onclick = () => { if (G.nodes.length < 8) { G.nodes.push({ x: 0.1 + 0.8 * Math.random(), y: 0.1 + 0.8 * Math.random(), f: [1, 1] }); compute(); } };
    q(root, '#del').onclick = () => { if (G.nodes.length > 2) { const n = G.nodes.length - 1; G.nodes.pop(); G.edges = G.edges.filter(([a, b]) => a < n && b < n); sel = Math.min(sel, n - 1); compute(); } };
    q(root, '#rst').onclick = () => { G = init(); sel = 1; compute(); };
    const ev = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const hit = (x, y) => { let b = -1, bd = 26; G.nodes.forEach((nd, i) => { const p = px(nd), d = Math.hypot(p[0] - x, p[1] - y); if (d < bd) { bd = d; b = i; } }); return b; };
    cv.addEventListener('pointerdown', (e) => { const [x, y] = ev(e), h = hit(x, y); if (h >= 0) { drag = h; moved = 0; cv.setPointerCapture(e.pointerId); } });
    cv.addEventListener('pointermove', (e) => { if (drag < 0) return; const [x, y] = ev(e); moved++; const nd = G.nodes[drag]; nd.x = AI.clamp((x - 30) / (c.w - 60), 0, 1); nd.y = AI.clamp((y - 40) / (c.h - 110), 0, 1); clk.dirty = true; });
    cv.addEventListener('pointerup', () => {
      if (drag >= 0 && moved < 4) {
        if (edgeMode) {
          if (pend < 0) pend = drag;
          else if (pend !== drag) { const i = G.edges.findIndex(([a, b]) => (a === pend && b === drag) || (a === drag && b === pend)); if (i >= 0) G.edges.splice(i, 1); else G.edges.push([pend, drag]); pend = -1; compute(); }
          else pend = -1;
        } else { sel = drag; q(root, '#fa').blur(); q(root, '#fb').blur(); compute(); q(root, '#fa').value = G.nodes[sel].f[0]; q(root, '#fb').value = G.nodes[sel].f[1]; }
      }
      drag = -1; clk.dirty = true;
    });
    cv.addEventListener('pointercancel', () => { drag = -1; });
    compute(); c.onresize = () => (clk.dirty = true);
    const draw = () => {
      if (!ok) return;
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h, R = +q(root, '#R').value;
      const idx = clk.idx(), done = clk.done(), r = Math.min(R - 1, done ? R - 1 : Math.floor(idx / 3)), sub = done ? 3 : idx % 3, fr = done ? 1 : clk.frac();
      g.clearRect(0, 0, W, Hh);
      G.edges.forEach(([a, b]) => { const A = px(G.nodes[a]), B = px(G.nodes[b]); g.strokeStyle = C.border; g.lineWidth = 2.5; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke(); });
      const feat = (v) => { if (sub === 3) return H[R][v]; if (sub < 2) return H[r][v]; const a = H[r][v], b = H[r + 1][v], t = ease(fr); return [AI.lerp(a[0], b[0], t), AI.lerp(a[1], b[1], t)]; };
      if (sub === 0) {
        const t = ease(fr);
        G.nodes.forEach((_, v) => G.adj[v].forEach((u) => {
          const cf = coef(u, v), A = px(G.nodes[u]), B = px(G.nodes[v]);
          if (u === v) { g.strokeStyle = C.yellow; g.globalAlpha = 0.6; g.lineWidth = 1.5; g.beginPath(); g.arc(A[0], A[1] - 30, 12 + 4 * Math.sin(t * Math.PI), 0, 7); g.stroke(); g.globalAlpha = 1; txt(g, f2(cf, 2), A[0] + 16, A[1] - 38, C.yellow, 'left', '10px ui-monospace,monospace'); return; }
          const dx = B[1] - A[1], dy = A[0] - B[0], L = Math.hypot(dx, dy) || 1, ox = (dx / L) * 6, oy = (dy / L) * 6;
          const x = AI.lerp(A[0], B[0], t) + ox, y = AI.lerp(A[1], B[1], t) + oy;
          dot(g, x, y, 3 + 3 * cf, C.yellow); if (t > 0.3 && t < 0.9) txt(g, f2(cf, 2), x + 6, y - 6, C.yellow, 'left', '10px ui-monospace,monospace');
        }));
      }
      G.nodes.forEach((nd, v) => {
        const [x, y] = px(nd), f = feat(v), isel = v === sel;
        const pulse = sub === 1 || sub === 2 ? 1 + 0.1 * Math.sin(fr * Math.PI) : 1;
        dot(g, x, y, 22 * pulse, AI.mix(C.bg2, C.blue, AI.clamp(Math.hypot(f[0], f[1]) / 2.5, 0.15, 1)));
        g.strokeStyle = isel ? C.yellow : v === pend ? C.green : C.border; g.lineWidth = isel || v === pend ? 3 : 1.5; g.beginPath(); g.arc(x, y, 22 * pulse, 0, 7); g.stroke();
        txt(g, '' + (v + 1), x, y + 5, C.fg, 'center', '14px system-ui');
        txt(g, '(' + f2(f[0], 2) + ', ' + f2(f[1], 2) + ')', x, y + 40, C.fg, 'center', MONO);
        if (sub === 1 || sub === 2) { const a = AG[r][v]; txt(g, 'Σ=(' + f2(a[0], 2) + ', ' + f2(a[1], 2) + ')', x, y - 32, C.orange, 'center', '11px ui-monospace,monospace'); }
      });
      txt(g, 'round ' + (done ? R : r + 1) + ' of ' + R, 12, 18, C.muted, 'left');
      q(root, '#cap').textContent = done ? `Done after ${R} round(s): each node now mixes features from nodes up to ${R} hop(s) away. Edit the graph, features or W.` : [
        `① Messages: every node sends its feature to its neighbours (and itself), scaled by 1/√(d_u·d_v).`,
        `② Aggregate: each node sums what it received (orange Σ) — the order of neighbours does not matter.`,
        `③ Update: multiply by W and apply ReLU — the new feature replaces the old one.`][sub];
    };
    AI.loop(root, (dt) => { clk.update(dt); if (clk.dirty) { clk.dirty = false; draw(); } }, { fps: 30 });
  };

  /* ======================= Architecture chooser ======================= */
  AI.labs.archChooser = function (root) {
    const sel = (id, label, opts) => `<label>${label}<select id="${id}">${opts.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join('')}</select></label>`;
    AI.labShell(root, 'Architecture chooser', 'answer the questions; get a ranked recommendation with reasons (heuristic)',
      `<canvas style="height:330px"></canvas><div class="readout" id="why" style="white-space:normal;margin-top:10px"></div>`,
      sel('data', 'Data type', [['tab', 'Tabular (rows and columns)'], ['img', 'Images'], ['txt', 'Text'], ['ts', 'Time series / sensor stream'], ['aud', 'Audio'], ['gr', 'Graph (molecules, networks)']]) +
      sel('size', 'Labelled examples', [['xs', 'under 1,000'], ['s', '1,000 – 10,000'], ['m', '10,000 – 1 million'], ['l', 'over 1 million']]) +
      sel('lab', 'Labels available?', [['y', 'Yes, labelled'], ['n', 'No (unlabelled / anomaly / search)']]) +
      sel('pre', 'Pre-trained model for this domain?', [['y', 'Yes, can fine-tune'], ['n', 'No']]) +
      sel('comp', 'Training compute', [['cpu', 'Laptop / CPU'], ['gpu', 'One GPU'], ['cl', 'GPU cluster']]) +
      sel('lat', 'Inference latency', [['b', 'Batch / offline'], ['i', 'Interactive (~100 ms)'], ['r', 'Real-time on a small device']]) +
      sel('int', 'Must explain predictions?', [['n', 'Not critical'], ['y', 'Yes, strongly']]));
    const c = AI.canvas(q(root, 'canvas'));
    const NAMES = ['Gradient boosting / linear model', 'MLP', 'CNN from scratch', 'Fine-tune pre-trained CNN', 'Fine-tune pre-trained transformer', 'Transformer from scratch', 'LSTM / GRU', '1-D CNN / TCN', 'Graph neural network', 'Autoencoder / embeddings + kNN'];
    let target = NAMES.map(() => 0), shown = NAMES.map(() => 0), order = NAMES.map((_, i) => i), moving = true;
    const get = (id) => q(root, '#' + id).value;
    const score = () => {
      const d = get('data'), sz = get('size'), lb = get('lab'), pr = get('pre'), cp = get('comp'), lt = get('lat'), it = get('int');
      const S = NAMES.map(() => 0), R = NAMES.map(() => []);
      const add = (i, v, why) => { S[i] += v; if (why) R[i].push((v > 0 ? '+ ' : '− ') + why); };
      const small = sz === 'xs' || sz === 's', big = sz === 'l';
      if (d === 'tab') { add(0, 5, 'trees and linear models are strongest on tabular data'); add(1, 1.5, 'an MLP works on tabular data but rarely beats boosting'); if (small) add(0, 2, 'small data favours simple models'); if (it === 'y') add(0, 2.5, 'trees and linear models are easy to explain'); if (big) add(1, 1.5, 'lots of data lets MLPs with embeddings compete'); }
      if (d === 'img') {
        if (pr === 'y') add(3, big ? 3.5 : 6, 'pre-trained vision features transfer well; fine-tune instead of training');
        if (sz === 'm' || big) add(2, 3, 'enough images to learn filters from scratch');
        if (small && pr === 'n') add(2, 1.5, 'a small CNN has the right bias for few images');
        if (small) add(2, -1, 'few images: a from-scratch net risks overfitting');
        if (pr === 'y') add(4, 3, 'pre-trained vision transformers also fine-tune well');
        if (lt === 'r') { add(3, 1.5, 'compact CNNs run fast on devices'); add(4, -2, 'transformers are heavy for a small device'); }
      }
      if (d === 'txt') {
        if (pr === 'y') add(4, 6.5, 'fine-tuning a pre-trained language model is the default'); else add(0, small ? 3 : 1, 'no pre-trained model: TF-IDF + a linear model is a solid baseline');
        add(6, pr === 'y' ? 0.5 : 2, 'recurrent models handle word order cheaply');
        if (big && cp === 'cl') add(5, 4, 'a huge corpus and a cluster allow pre-training from scratch');
        if (lt === 'r') add(4, -2.5, 'large transformers rarely fit tiny devices');
      }
      if (d === 'ts') {
        add(6, 4, 'recurrent cells fit ordered streams'); add(7, 3.5, 'dilated 1-D convolutions are fast and parallel'); add(0, small ? 3 : 1.5, 'lag features + boosting form a strong time-series baseline');
        if (lt === 'r') { add(6, 2, 'constant memory per step suits streaming'); add(7, 1.5, 'light 1-D CNNs run in real time'); }
        if (big && cp !== 'cpu') add(4, 2, 'transformers shine on long, large series');
      }
      if (d === 'aud') {
        add(7, 3, '1-D convolutions on waveforms'); add(3, 3, 'CNNs on spectrograms'); add(6, 1.5, 'recurrent layers for streaming speech');
        if (pr === 'y') add(4, 5, 'pre-trained speech/audio transformers transfer well');
        if (lt === 'r') add(7, 1.5, 'small convolution stacks are streaming friendly');
      }
      if (d === 'gr') { add(8, 7, 'relations between items carry the signal'); add(0, small ? 2.5 : 1, 'hand-made graph statistics + trees as a baseline'); } else add(8, -5, 'no natural graph structure');
      if (lb === 'n') { add(9, 6, 'unlabelled data: learn representations or detect anomalies'); [0, 2, 3, 4, 6].forEach((i) => add(i, -2, 'supervised models need labels')); }
      if (small && pr === 'n' && d !== 'tab') { add(5, -5, 'too little data to train a transformer from scratch'); add(2, -1, 'deep nets overfit tiny data'); }
      if (cp === 'cpu') { add(4, -2.5, 'fine-tuning transformers needs a GPU'); add(5, -4, 'training from scratch needs GPUs'); add(2, -1, 'CNN training is slow on a CPU'); add(0, 1.5, 'cheap to train on a laptop'); }
      if (cp !== 'cl') add(5, -2.5, 'pre-training from scratch needs a cluster');
      if (lt === 'r') { add(5, -2, 'too slow for a small device'); add(0, 1, 'tiny and fast at inference'); add(4, -1.5, 'slow inference for a small device'); }
      if (it === 'y') { add(1, -1, 'harder to explain'); add(4, -1.5, 'hard to explain'); add(5, -1.5, 'hard to explain'); add(0, 1, 'interpretable'); }
      target = S; order = S.map((_, i) => i).sort((a, b) => S[b] - S[a]);
      const top = order[0];
      q(root, '#why').innerHTML = `<b>Recommended: <span style="color:var(--green)">${NAMES[top]}</span></b><br>` + R[top].map((s) => `<div>${s}</div>`).join('') + `<br><span style="color:var(--muted)">Next: ${NAMES[order[1]]}, then ${NAMES[order[2]]}. A heuristic first choice — always test a simple baseline.</span>`;
      moving = true;
    };
    root.querySelectorAll('select').forEach((s) => (s.onchange = score)); c.onresize = () => (moving = true);
    q(root, '#data').value = 'img'; q(root, '#size').value = 's'; q(root, '#pre').value = 'y'; q(root, '#comp').value = 'gpu'; score();
    AI.loop(root, (dt) => {
      let any = false;
      for (let i = 0; i < NAMES.length; i++) { const dd = target[i] - shown[i]; if (Math.abs(dd) > 0.005) { shown[i] += dd * Math.min(1, dt * 2.5); any = true; } else shown[i] = target[i]; }
      if (!any && !moving) return; moving = any;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const lw = Math.min(250, W * 0.45), rowH = (H - 10) / NAMES.length, mx = Math.max(8, ...target) + 0.01, mn = Math.min(0, ...target);
      const ord = shown.map((_, i) => i).sort((a, b) => shown[b] - shown[a]);
      ord.forEach((i, r) => {
        const y = 4 + r * rowH, v = shown[i], top = i === order[0];
        txt(g, NAMES[i], lw - 8, y + rowH / 2 + 4, top ? C.fg : C.muted, 'right', top ? 'bold 12px system-ui' : SANS);
        g.fillStyle = C.bg2; g.fillRect(lw, y + 4, W - lw - 50, rowH - 8);
        const span = W - lw - 50, x0 = lw + (-mn / (mx - mn)) * span, wv = (v / (mx - mn)) * span;
        g.fillStyle = v >= 0 ? (top ? C.green : C.blue) : C.red; g.fillRect(Math.min(x0, x0 + wv), y + 4, Math.abs(wv), rowH - 8);
        txt(g, f2(v, 2), W - 8, y + rowH / 2 + 4, C.muted, 'right', MONO);
      });
    }, { fps: 30 });
  };
})();
