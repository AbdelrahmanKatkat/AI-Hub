/* Generative AI Module 9 labs: multimodal, evaluation and responsible generation.
   gen-mm-contrastive, gen-mm-tokens, gen-mm-consistency, gen-mm-elo, gen-mm-bootstrap, gen-mm-contam, gen-mm-frechet,
   gen-mm-calib, gen-mm-provenance, gen-mm-watermark, gen-mm-budget, gen-mm-guard.
   All simulations are illustrative, seeded (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const f2 = (x, d) => (+x).toFixed(d === undefined ? 4 : d);
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

  /* =====================================================================
     1. Contrastive learning: symmetric InfoNCE on a 3 x 3 matrix
     ===================================================================== */
  AI.labs['gen-mm-contrastive'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Contrastive pre-training: the 3 × 3 similarity matrix and symmetric InfoNCE',
      'illustrative: three image-text pairs; a cursor sweeps the three rows (image to text) then the three columns (text to image)',
      `<canvas id="m" style="height:270px"></canvas><canvas id="c" style="height:190px"></canvas>`,
      `<label>Matrix source <select id="mode"><option value="printed">the printed matrix of the lesson</option><option value="embed">trainable 2-D embeddings</option></select></label>
       ${sl('tau', 'Temperature τ', 0.05, 1, 0.05, 0.5)}
       ${sl('dd', 'Add to every diagonal cosine (printed matrix)', -0.5, 0.5, 0.05, 0)}
       <label><input type="checkbox" id="train"> Train: rotate the six embeddings downhill on the loss (2-D mode)</label>
       <div id="angs">${[0, 1, 2].map((i) => sl('ai' + i, 'image ' + (i + 1) + ' angle (°)', 0, 360, 1, 0)).join('')}${[0, 1, 2].map((i) => sl('at' + i, 'text ' + (i + 1) + ' angle (°)', 0, 360, 1, 0)).join('')}</div>
       ${sl('speed', 'Speed', 0.25, 2, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const o1 = AI.canvas(q(view, '#m')), o2 = AI.canvas(q(view, '#c'));
    const S0 = [[0.9, 0.2, 0.1], [0.3, 0.8, 0.2], [0.1, 0.4, 0.7]];
    const A0 = [0, 50, 110, 25, 40, 90];
    const st = { playing: true };
    let ang, idx, tt, dirty = true;
    const PAL = ['blue', 'orange', 'green'];
    const mode = () => q(ctl, '#mode').value;
    const getS = (a) => {
      if (mode() === 'printed') { const d = +q(ctl, '#dd').value; return S0.map((r, i) => r.map((v, j) => clamp(v + (i === j ? d : 0), -1, 1))); }
      return [0, 1, 2].map((i) => [0, 1, 2].map((j) => Math.cos((a[i] - a[3 + j]) * Math.PI / 180)));
    };
    const losses = (S, tau) => {
      const P = [], Qm = [[], [], []], rl = [], cl = [];
      for (let i = 0; i < 3; i++) { const e = S[i].map((v) => Math.exp(v / tau)), z = e[0] + e[1] + e[2]; P.push(e.map((v) => v / z)); rl.push(-Math.log(e[i] / z)); }
      for (let j = 0; j < 3; j++) { const e = [0, 1, 2].map((i) => Math.exp(S[i][j] / tau)), z = e[0] + e[1] + e[2]; for (let i = 0; i < 3; i++) Qm[i][j] = e[i] / z; cl.push(-Math.log(e[j] / z)); }
      const li = (rl[0] + rl[1] + rl[2]) / 3, lt = (cl[0] + cl[1] + cl[2]) / 3;
      return { P, Q: Qm, rl, cl, li, lt, sym: (li + lt) / 2 };
    };
    const symLoss = (a, tau) => losses(getS(a), tau).sym;
    const syncSliders = () => { for (let i = 0; i < 3; i++) { q(ctl, '#ai' + i).value = ((ang[i] % 360) + 360) % 360; q(ctl, '#at' + i).value = ((ang[3 + i] % 360) + 360) % 360; } };
    const reset = () => { ang = A0.slice(); idx = 0; tt = 0; syncSliders(); dirty = true; };
    const train = (dt) => {
      const tau = +q(ctl, '#tau').value, h = 0.01, lr = 0.5 * (+q(ctl, '#speed').value) * 180 / Math.PI;
      const g = ang.map((v, k) => { const a1 = ang.slice(), a2 = ang.slice(); a1[k] += h; a2[k] -= h; return (symLoss(a1, tau) - symLoss(a2, tau)) / (2 * h); });
      for (let k = 0; k < 6; k++) ang[k] -= lr * g[k] * dt * 0.6;
      syncSliders(); dirty = true;
    };
    const arrow = (g, cx, cy, r, deg, col, label, dashed) => {
      const a = -deg * Math.PI / 180, x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
      g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2; g.setLineDash(dashed ? [5, 4] : []);
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(x, y); g.stroke(); g.setLineDash([]);
      g.beginPath(); g.arc(x, y, 4, 0, 6.3); g.fill();
      txt(g, label, cx + (r + 14) * Math.cos(a), cy + (r + 14) * Math.sin(a) + 4, col, 'bold 11px sans-serif', 'center');
    };
    const draw = () => {
      const c = AI.col(), tau = +q(ctl, '#tau').value, S = getS(ang), L = losses(S, tau);
      const emb = mode() === 'embed';
      q(ctl, '#angs').style.display = emb ? '' : 'none';
      q(ctl, '#dd').parentNode.style.display = emb ? 'none' : '';
      {
        const g = o1.ctx, W = o1.w, H = o1.h; g.clearRect(0, 0, W, H);
        const left = 70, top = 36, cw = Math.min(110, (W - left - 14) / 3), ch = 62;
        const isRow = idx < 3, k = idx % 3;
        for (let j = 0; j < 3; j++) txt(g, 'text ' + (j + 1), left + j * cw + cw / 2, top - 10, c[PAL[j]], 'bold 12px sans-serif', 'center');
        for (let i = 0; i < 3; i++) txt(g, 'image ' + (i + 1), left - 8, top + i * ch + ch / 2 + 4, c[PAL[i]], 'bold 12px sans-serif', 'right');
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          const x = left + j * cw, y = top + i * ch, v = S[i][j];
          g.fillStyle = c.bg2; g.fillRect(x + 2, y + 2, cw - 4, ch - 4);
          g.fillStyle = c.blue; g.globalAlpha = 0.12 + 0.45 * clamp((v + 0.2) / 1.2, 0, 1); g.fillRect(x + 2, y + 2, cw - 4, ch - 4); g.globalAlpha = 1;
          g.strokeStyle = i === j ? c.green : c.border; g.lineWidth = i === j ? 2 : 1; g.strokeRect(x + 2, y + 2, cw - 4, ch - 4); g.lineWidth = 1;
          txt(g, v.toFixed(2), x + cw / 2, y + 26, c.fg, 'bold 15px sans-serif', 'center');
          const inCur = isRow ? i === k : j === k;
          if (inCur) { const p = isRow ? L.P[i][j] : L.Q[i][j]; txt(g, 'p = ' + p.toFixed(3), x + cw / 2, y + 48, c.orange, 'bold 12px sans-serif', 'center'); }
        }
        g.strokeStyle = c.orange; g.lineWidth = 3;
        if (isRow) g.strokeRect(left, top + k * ch, cw * 3, ch); else g.strokeRect(left + k * cw, top, cw, ch * 3);
        g.lineWidth = 1;
        const cur = isRow ? L.rl[k] : L.cl[k];
        txt(g, isRow ? `row ${k + 1}: softmax over texts, target text ${k + 1}` : `column ${k + 1}: softmax over images, target image ${k + 1}`, 8, H - 22, c.muted, '12px sans-serif');
        txt(g, `loss of this line = -ln p(green cell) = ${cur.toFixed(4)}`, 8, H - 6, c.orange, 'bold 12px sans-serif');
      }
      {
        const g = o2.ctx, W = o2.w, H = o2.h; g.clearRect(0, 0, W, H);
        if (!emb) { txt(g, 'Choose "trainable 2-D embeddings" to see the vectors and train them.', W / 2, H / 2, c.muted, '13px sans-serif', 'center'); }
        else {
          const cx = W / 2, cy = H / 2, r = Math.min(H / 2 - 22, 70);
          g.strokeStyle = c.border; g.beginPath(); g.arc(cx, cy, r, 0, 6.3); g.stroke();
          for (let i = 0; i < 3; i++) arrow(g, cx, cy, r, ang[i], c[PAL[i]], 'I' + (i + 1), false);
          for (let i = 0; i < 3; i++) arrow(g, cx, cy, r * 0.8, ang[3 + i], c[PAL[i]], 'T' + (i + 1), true);
          txt(g, 'solid = image embedding, dashed = text embedding; cosine = how aligned', 6, 14, c.muted, '11px sans-serif');
        }
      }
      const bound = Math.log(3) - L.sym;
      q(ctl, '#out').innerHTML = `rows (image to text): ${L.rl.map((v) => v.toFixed(4)).join(', ')} → mean <b>${L.li.toFixed(4)}</b><br>` +
        `columns (text to image): ${L.cl.map((v) => v.toFixed(4)).join(', ')} → mean <b>${L.lt.toFixed(4)}</b><br>` +
        `symmetric loss = (${L.li.toFixed(4)} + ${L.lt.toFixed(4)}) / 2 = <b>${L.sym.toFixed(4)}</b><br>` +
        `ln 3 = ${Math.log(3).toFixed(4)}, so the mutual information bound  ln N − L = <b>${bound.toFixed(4)}</b> nats`;
      q(ctl, '#cap').textContent = emb && q(ctl, '#train').checked ? 'Training: each pair is pulled together and pushed away from the wrong partners, so the loss falls and the lower bound rises toward ln 3.' : 'The cursor turns each line into a 3-way classification: which text (or image) belongs to this one? Lower loss means the green diagonal wins its line.';
      showVals(ctl, { tau: (v) => v.toFixed(2), dd: (v) => v.toFixed(2), speed: (v) => v + '×', ai0: (v) => v, ai1: (v) => v, ai2: (v) => v, at0: (v) => v, at1: (v) => v, at2: (v) => v });
    };
    const adv = () => { idx = (idx + 1) % 6; tt = 0; dirty = true; };
    wire(ctl, st, adv, reset, draw);
    ['tau', 'dd', 'mode'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    q(ctl, '#mode').onchange = () => { dirty = true; };
    [0, 1, 2].forEach((i) => {
      q(ctl, '#ai' + i).oninput = (e) => { ang[i] = +e.target.value; dirty = true; };
      q(ctl, '#at' + i).oninput = (e) => { ang[3 + i] = +e.target.value; dirty = true; };
    });
    o1.onresize = draw; o2.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { tt += dt * (+q(ctl, '#speed').value); if (tt > 3) adv(); }
      if (mode() === 'embed' && q(ctl, '#train').checked && st.playing) train(dt);
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Spatiotemporal tokens: counting and attention patterns
     ===================================================================== */
  AI.labs['gen-mm-tokens'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Video and audio token budget',
      'illustrative: a query token (orange) looks at other tokens; full attention versus factorised space-then-time attention',
      `<canvas style="height:300px"></canvas>`,
      `${sl('sec', 'Video length (seconds)', 1, 8, 1, 4)}
       <label>Frames per second <select id="fps"><option>4</option><option selected>8</option><option>12</option><option>24</option></select></label>
       <label>Frame size (pixels, square) <select id="px"><option>128</option><option selected>256</option><option>512</option></select></label>
       <label>Patch size in the latent grid <select id="pt"><option>1</option><option selected>2</option><option>4</option></select></label>
       <label>Temporal compression <select id="tc"><option>1</option><option>2</option><option selected>4</option><option>8</option></select></label>
       <label>Attention <select id="att"><option value="full">full (every token sees every token)</option><option value="fact" selected>factorised (space phase, then time phase)</option></select></label>
       ${sl('asec', 'Audio length (seconds)', 1, 30, 1, 10)}
       ${sl('cb', 'Audio codebooks per frame', 1, 8, 1, 4)}
       ${sl('speed', 'Speed', 0.25, 2, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let step = 0, tt = 0, dirty = true;
    const calc = () => {
      const sec = +q(ctl, '#sec').value, fps = +q(ctl, '#fps').value, px = +q(ctl, '#px').value, pt = +q(ctl, '#pt').value, tc = +q(ctl, '#tc').value;
      const F = sec * fps, T = Math.ceil(F / tc), side = Math.max(1, Math.floor(px / 8 / pt)), S = side * side, N = T * S;
      const asec = +q(ctl, '#asec').value, cb = +q(ctl, '#cb').value;
      return { sec, fps, px, pt, tc, F, T, side, S, N, full: N * N, fact: T * S * S + S * T * T, aud: 75 * asec * cb, atps: 75 * cb };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, k = calc(); g.clearRect(0, 0, W, H);
      const gd = Math.min(k.side, 6), np = Math.min(k.T, 8), pw = Math.min(110, (W - 20) / np - 8), cs = pw / gd, y0 = 40;
      const total = gd * gd * np, qi = step % total, qf = Math.floor(qi / (gd * gd)), qc = qi % (gd * gd);
      const full = q(ctl, '#att').value === 'full', spacePhase = Math.floor(step / total) % 2 === 0 || true;
      const phaseSpace = Math.floor(step / 4) % 2 === 0;
      txt(g, `latent frames: ${k.T} (drawing ${np}), each a ${k.side} × ${k.side} grid of tokens (drawing ${gd} × ${gd})`, 10, 16, c.muted);
      for (let f = 0; f < np; f++) {
        const x0 = 10 + f * (pw + 8);
        for (let cc = 0; cc < gd * gd; cc++) {
          const x = x0 + (cc % gd) * cs, y = y0 + Math.floor(cc / gd) * cs;
          let att = false;
          if (full) att = true; else att = phaseSpace ? f === qf : cc === qc;
          const isQ = f === qf && cc === qc;
          g.fillStyle = isQ ? c.orange : att ? c.blue : c.bg3; g.globalAlpha = isQ ? 1 : att ? 0.55 : 1;
          g.fillRect(x + 1, y + 1, cs - 2, cs - 2); g.globalAlpha = 1;
        }
        txt(g, 't' + (f + 1), x0 + pw / 2, y0 + gd * cs + 14, c.muted, '11px sans-serif', 'center');
      }
      const by = y0 + gd * cs + 40;
      txt(g, `audio: ${k.atps} tokens per second of sound`, 10, by, c.muted);
      const nb = Math.min(48, k.aud), bw = (W - 20) / 48;
      for (let i = 0; i < 48; i++) { g.fillStyle = i < nb * 48 / Math.max(48, k.aud) * 1 || k.aud >= 48 ? c.teal : c.bg3; g.globalAlpha = 0.8; g.fillRect(10 + i * bw + 1, by + 10, bw - 2, 18); g.globalAlpha = 1; }
      txt(g, 'each bar is a block of audio tokens (not to scale)', 10, by + 44, c.muted, '11px sans-serif');
      q(ctl, '#out').innerHTML = `video: ${k.sec} s × ${k.fps} fps = ${k.F} frames; temporal compression ${k.tc} → ${k.T} latent frames; ${k.px}/8/${k.pt} = ${k.side} tokens per side → S = ${k.S} tokens per frame<br>` +
        `video tokens N = T·S = ${k.T} × ${k.S} = <b>${k.N.toLocaleString()}</b><br>` +
        `full attention pairs N² = <b>${k.full.toLocaleString()}</b>; factorised T·S² + S·T² = <b>${k.fact.toLocaleString()}</b> (${(k.full / k.fact).toFixed(1)}× fewer)<br>` +
        `audio: 75 frames/s × ${k.aud / 75 / (+q(ctl, '#asec').value)} codebooks × ${q(ctl, '#asec').value} s = <b>${k.aud.toLocaleString()}</b> tokens`;
      q(ctl, '#cap').textContent = full ? 'Full attention: the orange query token looks at every token in every frame, so the work grows as N².' : (phaseSpace ? 'Space phase: the query attends to every token in its own frame.' : 'Time phase: the query attends to the same grid position in every frame.');
      showVals(ctl, { sec: (v) => v + ' s', asec: (v) => v + ' s', cb: (v) => v, speed: (v) => v + '×' });
    };
    const adv = () => { step++; tt = 0; dirty = true; };
    wire(ctl, st, adv, () => { step = 0; tt = 0; dirty = true; }, draw);
    ['sec', 'fps', 'px', 'pt', 'tc', 'att', 'asec', 'cb', 'speed'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; q(ctl, '#' + id).onchange = () => { dirty = true; }; });
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { tt += dt * (+q(ctl, '#speed').value); if (tt > 0.7) adv(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Temporal consistency: independent versus coupled frame noise
     ===================================================================== */
  AI.labs['gen-mm-consistency'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Consistency over time: why frames must share information',
      'illustrative: a ball should glide along the dashed path; each frame adds its own error, which can be independent (ρ = 0) or coupled to the last frame',
      `<canvas style="height:340px"></canvas>`,
      `${sl('rho', 'Coupling ρ between neighbouring frames’ errors', 0, 0.98, 0.02, 0)}
       ${sl('sig', 'Error size per frame (fraction of box width)', 0, 0.2, 0.01, 0.08)}
       ${sl('fps', 'Playback (frames per second)', 1, 12, 1, 3)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="seed">New random errors</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const T = 48, st = { playing: true };
    let xi, fr = 0, acc = 0, seed = 7, dirty = true;
    const truth = (t) => 0.5 + 0.3 * Math.sin(2 * Math.PI * t / T);
    const newSeed = () => { const r = AI.rng(seed); xi = []; for (let t = 0; t < T; t++) xi.push(AI.randn(r)); dirty = true; };
    const path = () => {
      const rho = +q(ctl, '#rho').value, s = +q(ctl, '#sig').value, e = []; let prev = 0;
      for (let t = 0; t < T; t++) { prev = t ? rho * prev + Math.sqrt(1 - rho * rho) * xi[t] : xi[t]; e.push(prev); }
      return e.map((v, t) => clamp(truth(t) + s * v, 0.03, 0.97));
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, gp = path(); g.clearRect(0, 0, W, H);
      const L = 40, R = W - 12, pt = 24, pb = H * 0.55, X = (t) => L + t / (T - 1) * (R - L), Y = (v) => pb - v * (pb - pt);
      txt(g, 'ball position over frames', L, 14, c.muted);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, pt); g.lineTo(L, pb); g.lineTo(R, pb); g.stroke();
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); for (let t = 0; t < T; t++) { t ? g.lineTo(X(t), Y(truth(t))) : g.moveTo(X(t), Y(truth(t))); } g.stroke(); g.setLineDash([]);
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); for (let t = 0; t <= fr; t++) { t ? g.lineTo(X(t), Y(gp[t])) : g.moveTo(X(t), Y(gp[t])); } g.stroke(); g.lineWidth = 1;
      g.fillStyle = c.orange; g.beginPath(); g.arc(X(fr), Y(gp[fr]), 4, 0, 6.3); g.fill();
      txt(g, 'frame ' + (fr + 1) + ' of ' + T, R, 14, c.muted, '12px sans-serif', 'right');
      const bx = L, bw = R - L, by = pb + 34, bh = 50;
      g.strokeStyle = c.border; g.strokeRect(bx, by, bw, bh);
      g.strokeStyle = c.fg; g.setLineDash([3, 3]); g.beginPath(); g.arc(bx + truth(fr) * bw, by + bh / 2, 9, 0, 6.3); g.stroke(); g.setLineDash([]);
      g.fillStyle = c.orange; g.beginPath(); g.arc(bx + gp[fr] * bw, by + bh / 2, 9, 0, 6.3); g.fill();
      txt(g, 'the video frame: filled ball = generated, dashed ring = where it should be', L, by + bh + 18, c.muted, '11px sans-serif');
      const stepsG = [], stepsT = []; for (let t = 1; t < T; t++) { stepsG.push(Math.abs(gp[t] - gp[t - 1])); stepsT.push(Math.abs(truth(t) - truth(t - 1))); }
      const m = (a) => a.reduce((s, v) => s + v, 0) / a.length, rmsErr = Math.sqrt(m(gp.map((v, t) => (v - truth(t)) ** 2)));
      q(ctl, '#out').innerHTML = `mean frame-to-frame jump: generated <b>${m(stepsG).toFixed(3)}</b>, true motion ${m(stepsT).toFixed(3)} (box width = 1)<br>root-mean-square distance from the true path: ${rmsErr.toFixed(3)}`;
      const rho = +q(ctl, '#rho').value;
      q(ctl, '#cap').textContent = rho < 0.2 ? 'Independent errors: every frame is wrong in its own direction, so the ball flickers even though each single frame is only slightly off.' : 'Coupled errors: neighbouring frames are wrong in the same way, so the ball moves smoothly (it can still be off the dashed path, but it does not flicker).';
      showVals(ctl, { rho: (v) => v.toFixed(2), sig: (v) => v.toFixed(2), fps: (v) => v });
    };
    const adv = () => { fr = (fr + 1) % T; dirty = true; };
    wire(ctl, st, adv, () => { fr = 0; acc = 0; dirty = true; }, draw);
    q(ctl, '#seed').onclick = () => { seed += 11; newSeed(); draw(); };
    ['rho', 'sig', 'fps'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    newSeed(); o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { acc += dt * (+q(ctl, '#fps').value); while (acc >= 1) { acc -= 1; fr = (fr + 1) % T; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. Elo ratings from printed pairwise outcomes
     ===================================================================== */
  AI.labs['gen-mm-elo'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Elo ratings replayed game by game',
      'illustrative: three models A, B, C, all starting at 1000; each printed outcome moves a rating by K·(1 − expected)',
      `<canvas style="height:230px"></canvas><div id="games" class="chips" style="margin-top:8px"></div>`,
      `${sl('K', 'K factor (step size)', 4, 64, 4, 32)}
       ${sl('rate', 'Games per second', 0.1, 1.5, 0.1, 0.4)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button><button class="btn" id="rev">Reverse game order</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap">Click a game chip to flip its winner and watch every later rating change.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const NM = ['A', 'B', 'C'], G0 = [[0, 1], [0, 2], [1, 2], [2, 0], [1, 0], [1, 2], [0, 2], [2, 1]];
    let G = G0.map((x) => x.slice()), states, info, pos = 0, dirty = true;
    const st = { playing: true };
    const comp = () => {
      const K = +q(ctl, '#K').value; let R = [1000, 1000, 1000]; states = [R.slice()]; info = [];
      G.forEach(([w, l]) => { const E = 1 / (1 + Math.pow(10, (R[l] - R[w]) / 400)), d = K * (1 - E); R = R.slice(); R[w] += d; R[l] -= d; states.push(R.slice()); info.push({ E, d }); });
    };
    const chips = () => {
      const box = q(view, '#games'); box.innerHTML = '';
      G.forEach(([w, l], i) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = `${i + 1}: ${NM[w]} beat ${NM[l]}`; b.dataset.i = i; b.onclick = () => { G[i] = [G[i][1], G[i][0]]; comp(); chips(); dirty = true; }; box.appendChild(b); });
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const i0 = Math.min(G.length, Math.floor(pos)), fr = Math.min(1, pos - i0), i1 = Math.min(G.length, i0 + 1);
      const e = ease(i0 >= G.length ? 0 : fr), R = states[i0].map((v, k) => v + (states[i1][k] - v) * e);
      const lo = 940, hi = 1060, L = 70, Rr = W - 24, X = (v) => L + (v - lo) / (hi - lo) * (Rr - L);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(X(1000), 10); g.lineTo(X(1000), H - 26); g.stroke();
      for (let v = lo; v <= hi; v += 20) txt(g, v, X(v), H - 10, c.muted, '11px sans-serif', 'center');
      const cols = [c.blue, c.orange, c.green];
      for (let k = 0; k < 3; k++) {
        const y = 18 + k * 62; g.fillStyle = cols[k]; g.globalAlpha = 0.75;
        const a = X(1000), b = X(clamp(R[k], lo, hi)); g.fillRect(Math.min(a, b), y, Math.abs(b - a), 40); g.globalAlpha = 1;
        txt(g, 'model ' + NM[k], 8, y + 25, c.fg, 'bold 13px sans-serif'); txt(g, R[k].toFixed(1), b + (b >= a ? 6 : -6), y + 25, c.fg, 'bold 13px sans-serif', b >= a ? 'left' : 'right');
      }
      q(view, '#games').childNodes.forEach((b, i) => { b.style.outline = i === Math.min(G.length - 1, i0) && pos < G.length ? '2px solid ' + c.orange : 'none'; });
      const gi = Math.min(G.length - 1, i0);
      const [w, l] = G[gi], inf = info[gi], before = states[gi];
      q(ctl, '#out').innerHTML = `game ${gi + 1}: ${NM[w]} (${before[w].toFixed(1)}) beat ${NM[l]} (${before[l].toFixed(1)}): expected score of the winner E = 1/(1+10^((${before[l].toFixed(1)}−${before[w].toFixed(1)})/400)) = <b>${inf.E.toFixed(4)}</b>, change = K(1−E) = <b>${inf.d.toFixed(3)}</b><br>final ratings: A ${states[G.length][0].toFixed(1)}, B ${states[G.length][1].toFixed(1)}, C ${states[G.length][2].toFixed(1)} (sum stays 3000)`;
      showVals(ctl, { K: (v) => v, rate: (v) => v + '/s' });
    };
    const adv = () => { pos = Math.min(G.length, Math.floor(pos) + 1); dirty = true; };
    wire(ctl, st, adv, () => { pos = 0; G = G0.map((x) => x.slice()); comp(); chips(); dirty = true; }, draw);
    q(ctl, '#K').oninput = () => { comp(); dirty = true; };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    q(ctl, '#rev').onclick = () => { G.reverse(); comp(); chips(); pos = 0; dirty = true; draw(); };
    comp(); chips(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && pos < G.length) { pos = Math.min(G.length, pos + dt * (+q(ctl, '#rate').value)); dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     5. Bootstrap confidence interval on a small set of judgements
     ===================================================================== */
  AI.labs['gen-mm-bootstrap'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Bootstrap interval for a win rate',
      'illustrative: 20 printed judgements (1 = model A preferred); each resample draws 20 judgements with replacement',
      `<canvas style="height:340px"></canvas>`,
      `<label>Repeat the judgements <select id="rep"><option value="1">1× (n = 20)</option><option value="2">2× (n = 40)</option><option value="4">4× (n = 80)</option></select></label>
       ${sl('rate', 'Resamples per second', 0.5, 20, 0.5, 2)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap">Click a cell in the top row to flip one judgement and restart.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const D0 = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0];
    let D = D0.slice(), data, n, hist, cnt, last, rs, tick = AI.ticker(), dirty = true, cells = [];
    const st = { playing: true };
    const reset = () => { const k = +q(ctl, '#rep').value; data = []; for (let r = 0; r < k; r++) data.push(...D); n = data.length; hist = new Array(n + 1).fill(0); cnt = 0; last = null; rs = AI.rng(21); dirty = true; };
    const add = () => { const pick = []; let w = 0; for (let i = 0; i < n; i++) { const v = data[Math.floor(rs() * n)]; pick.push(v); w += v; } last = pick; hist[w]++; cnt++; dirty = true; };
    const pct = (p) => { let s = 0; for (let k = 0; k <= n; k++) { s += hist[k]; if (s >= p * cnt) return k / n; } return 1; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const L = 14, R = W - 14, cw = (R - L) / n; cells = [];
      txt(g, 'the judgements (A preferred = filled)', L, 14, c.muted);
      for (let i = 0; i < n; i++) { g.fillStyle = data[i] ? c.green : c.bg3; g.fillRect(L + i * cw + 1, 20, cw - 2, 20); }
      const wins = data.reduce((a, b) => a + b, 0);
      txt(g, 'latest resample', L, 60, c.muted);
      for (let i = 0; i < n; i++) { g.fillStyle = last ? (last[i] ? c.orange : c.bg3) : c.bg3; g.fillRect(L + i * cw + 1, 66, cw - 2, 20); }
      const hx0 = 40, hx1 = W - 14, hy0 = 112, hy1 = H - 34;
      txt(g, 'win rates of the resamples', hx0, 104, c.muted);
      const mx = Math.max(1, ...hist), bw = (hx1 - hx0) / (n + 1), X = (r) => hx0 + r * (hx1 - hx0 - bw) + bw / 2;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(hx0, hy1); g.lineTo(hx1, hy1); g.stroke();
      for (let k = 0; k <= n; k++) { const h = hist[k] / mx * (hy1 - hy0); g.fillStyle = c.blue; g.globalAlpha = 0.8; g.fillRect(hx0 + k * bw + 0.5, hy1 - h, Math.max(1, bw - 1), h); g.globalAlpha = 1; }
      for (const v of [0, 0.25, 0.5, 0.75, 1]) txt(g, v.toFixed(2), X(v), hy1 + 14, c.muted, '11px sans-serif', 'center');
      const p = wins / n; g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(p), hy0); g.lineTo(X(p), hy1); g.stroke(); g.setLineDash([]);
      let lo = NaN, hi = NaN;
      if (cnt >= 30) { lo = pct(0.025); hi = pct(0.975); g.strokeStyle = c.red; g.lineWidth = 2; [lo, hi].forEach((v) => { g.beginPath(); g.moveTo(X(v), hy0); g.lineTo(X(v), hy1); g.stroke(); }); g.lineWidth = 1; }
      const se = Math.sqrt(p * (1 - p) / n);
      q(ctl, '#out').innerHTML = `original: ${wins} of ${n} preferred A, win rate <b>${p.toFixed(3)}</b>; formula SE = √(p(1−p)/n) = ${se.toFixed(4)} (normal ± ${(1.96 * se).toFixed(3)})<br>` +
        (cnt >= 30 ? `after ${cnt} resamples: 2.5th percentile <b>${lo.toFixed(3)}</b>, 97.5th percentile <b>${hi.toFixed(3)}</b> (red lines)` : `resamples so far: ${cnt} (percentiles appear after 30)`) +
        `<br>exact answer for 20 judgements at 13 wins: the resample win count is Binomial(20, 0.65), whose 2.5% and 97.5% points are 0.45 and 0.85`;
      showVals(ctl, { rate: (v) => v + '/s' });
    };
    wire(ctl, st, add, reset, draw);
    q(ctl, '#rep').onchange = () => { reset(); };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    cv.onclick = (e) => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; if (y > 18 && y < 44) { const cw = (o.w - 28) / n, i = Math.floor((x - 14) / cw); if (i >= 0 && i < n) { D[i % D.length] ^= 1; reset(); draw(); } } };
    reset(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && cnt < 4000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) add(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     6. Benchmark contamination
     ===================================================================== */
  AI.labs['gen-mm-contam'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'How leaked test questions inflate a benchmark score',
      'illustrative: 100 test items; the scan reveals them one by one. A leaked item may be answered from memory',
      `<canvas style="height:330px"></canvas>`,
      `${sl('c', 'Share of the test set that leaked into training', 0, 0.6, 0.05, 0.2)}
       ${sl('m', 'How often a leaked item is answered correctly (memorised)', 0.5, 1, 0.1, 1)}
       ${sl('b', 'True accuracy on genuinely unseen items', 0.2, 0.9, 0.05, 0.6)}
       ${sl('rate', 'Items revealed per second', 1, 30, 1, 6)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let order, kind, n, tick = AI.ticker(), dirty = true;
    const build = () => {
      const c = +q(ctl, '#c').value, m = +q(ctl, '#m').value, b = +q(ctl, '#b').value, rr = AI.rng(4);
      order = Array.from({ length: 100 }, (_, i) => i); for (let i = 99; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      const Lk = Math.round(100 * c), cl = Math.round(m * Lk), cu = Math.round(b * (100 - Lk)); kind = new Array(100);
      order.forEach((pos, r) => { kind[pos] = r < Lk ? (r < cl ? 'LC' : 'LW') : (r - Lk < cu ? 'UC' : 'UW'); });
      n = 0; dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const s = Math.min((W - 20) / 10, (H - 80) / 10), x0 = (W - 10 * s) / 2, y0 = 8;
      let rep = 0, clean = 0, seenU = 0, seenL = 0, rightL = 0;
      for (let i = 0; i < 100; i++) {
        const x = x0 + (i % 10) * s, y = y0 + Math.floor(i / 10) * s, k = kind[i], shown = i < n;
        if (shown) { if (k === 'LC' || k === 'UC') rep++; if (k[0] === 'U') { seenU++; if (k === 'UC') clean++; } else { seenL++; if (k === 'LC') rightL++; } }
        g.fillStyle = !shown ? c.bg3 : k === 'UC' ? c.green : k === 'UW' ? c.muted : k === 'LC' ? c.red : c.orange; g.globalAlpha = shown ? 0.85 : 0.5;
        g.fillRect(x + 2, y + 2, s - 4, s - 4); g.globalAlpha = 1;
        if (shown && k[0] === 'L') { g.strokeStyle = c.fg; g.lineWidth = 2; g.strokeRect(x + 3, y + 3, s - 6, s - 6); g.lineWidth = 1; }
      }
      const ly = y0 + 10 * s + 16; let lx = 10;
      [['unseen, correct', c.green], ['unseen, wrong', c.muted], ['leaked, correct', c.red], ['leaked, wrong', c.orange]].forEach(([t, col]) => { g.fillStyle = col; g.fillRect(lx, ly - 9, 11, 11); txt(g, t, lx + 15, ly, c.fg, '11px sans-serif'); lx += 18 + g.measureText(t).width + 12; });
      txt(g, 'white outline = the question was in the training data', 10, ly + 18, c.muted, '11px sans-serif');
      const cc = +q(ctl, '#c').value, mm = +q(ctl, '#m').value, bb = +q(ctl, '#b').value;
      const Lk = Math.round(100 * cc), exp = (Math.round(mm * Lk) + Math.round(bb * (100 - Lk))) / 100;
      q(ctl, '#out').innerHTML = `revealed ${n} of 100: reported correct so far <b>${rep}</b>${seenU ? `, of which ${clean} on unseen items (clean accuracy so far ${(clean / seenU).toFixed(2)})` : ''}<br>` +
        `full benchmark: reported score = ${Lk} leaked × ${mm} + ${100 - Lk} unseen × ${bb} = <b>${exp.toFixed(2)}</b>, while the real ability on new questions is ${bb.toFixed(2)}: inflation <b>${(exp - bb).toFixed(2)}</b>`;
      q(ctl, '#cap').textContent = Lk === 0 ? 'No leakage: the benchmark score equals the true accuracy on unseen questions.' : 'The red cells are correct answers that come from memory rather than ability. A reader sees only the total, so the model looks better than it is.';
      showVals(ctl, { c: (v) => Math.round(v * 100) + '%', m: (v) => v, b: (v) => v, rate: (v) => v + '/s' });
    };
    const adv = () => { if (n < 100) { n++; dirty = true; } };
    wire(ctl, st, adv, build, draw);
    ['c', 'm', 'b'].forEach((id) => { q(ctl, '#' + id).oninput = () => { build(); }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    build(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && n < 100) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     7. Frechet distance on 2-D Gaussians
     ===================================================================== */
  AI.labs['gen-mm-frechet'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Fréchet distance between two 2-D clouds',
      'illustrative: blue = real features, orange = generated features; points arrive one pair at a time and the estimate is recomputed from the samples',
      `<canvas style="height:360px"></canvas>`,
      `${sl('mx', 'Generated mean, x', -4, 6, 0.5, 3)}
       ${sl('my', 'Generated mean, y', -4, 6, 0.5, 4)}
       ${sl('sx', 'Generated spread, x', 0.5, 3, 0.25, 2)}
       ${sl('sy', 'Generated spread, y', 0.5, 3, 0.25, 1)}
       ${sl('rho', 'Generated correlation', -0.9, 0.9, 0.1, 0)}
       ${sl('rate', 'Point pairs per second', 1, 30, 1, 5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap">The real cloud is fixed: mean (0, 0), spreads 1 and 2, no correlation.</p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const REAL = { m: [0, 0], s: [1, 2], r: 0 };
    const st = { playing: true };
    let Z, n, tick = AI.ticker(), dirty = true, px, py;
    const par = () => ({ m: [+q(ctl, '#mx').value, +q(ctl, '#my').value], s: [+q(ctl, '#sx').value, +q(ctl, '#sy').value], r: +q(ctl, '#rho').value });
    const cov = (p) => [p.s[0] * p.s[0], p.r * p.s[0] * p.s[1], p.s[1] * p.s[1]];
    const fd = (m1, S1, m2, S2) => {
      const mt = (m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2, tr = S1[0] + S1[2] + S2[0] + S2[2];
      const t12 = S1[0] * S2[0] + 2 * S1[1] * S2[1] + S1[2] * S2[2], d1 = S1[0] * S1[2] - S1[1] ** 2, d2 = S2[0] * S2[2] - S2[1] ** 2;
      const sq = Math.sqrt(Math.max(0, t12 + 2 * Math.sqrt(Math.max(0, d1 * d2))));
      return { mt, tr: tr - 2 * sq, sq, d: mt + tr - 2 * sq };
    };
    const reset = () => { const r = AI.rng(33); Z = []; for (let i = 0; i < 400; i++) Z.push([AI.randn(r), AI.randn(r), AI.randn(r), AI.randn(r)]); n = 0; dirty = true; };
    const samp = (pts) => {
      const k = pts.length, m = [0, 0]; pts.forEach((p) => { m[0] += p[0] / k; m[1] += p[1] / k; });
      let a = 0, b = 0, c2 = 0; pts.forEach((p) => { a += (p[0] - m[0]) ** 2; b += (p[0] - m[0]) * (p[1] - m[1]); c2 += (p[1] - m[1]) ** 2; });
      return { m, S: [a / (k - 1), b / (k - 1), c2 / (k - 1)] };
    };
    const ell = (g, X, Y, m, S, col) => {
      const th = 0.5 * Math.atan2(2 * S[1], S[0] - S[2]), tr = S[0] + S[2], dt = Math.sqrt(Math.max(0, ((S[0] - S[2]) / 2) ** 2 + S[1] ** 2)), l1 = tr / 2 + dt, l2 = Math.max(1e-6, tr / 2 - dt);
      g.strokeStyle = col; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath();
      for (let i = 0; i <= 60; i++) { const a = i / 60 * 6.2832, u = 2 * Math.sqrt(l1) * Math.cos(a), v = 2 * Math.sqrt(l2) * Math.sin(a), x = m[0] + u * Math.cos(th) - v * Math.sin(th), y = m[1] + u * Math.sin(th) + v * Math.cos(th); i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); }
      g.stroke(); g.setLineDash([]); g.lineWidth = 1;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const P = par(), lo = -6, hi = 8, X = (v) => 30 + (v - lo) / (hi - lo) * (W - 44), Y = (v) => H - 24 - (v - lo) / (hi - lo) * (H - 40);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(X(0), Y(lo)); g.lineTo(X(0), Y(hi)); g.moveTo(X(lo), Y(0)); g.lineTo(X(hi), Y(0)); g.stroke();
      const real = [], gen = [];
      for (let i = 0; i < n; i++) {
        const z = Z[i]; real.push([REAL.m[0] + REAL.s[0] * z[0], REAL.m[1] + REAL.s[1] * z[1]]);
        gen.push([P.m[0] + P.s[0] * z[2], P.m[1] + P.m[1] * 0 + P.s[1] * (P.r * z[2] + Math.sqrt(1 - P.r * P.r) * z[3])]);
      }
      g.globalAlpha = 0.7; g.fillStyle = c.blue; real.forEach((p) => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.6, 0, 6.3); g.fill(); });
      g.fillStyle = c.orange; gen.forEach((p) => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.6, 0, 6.3); g.fill(); }); g.globalAlpha = 1;
      const S1 = cov(REAL), S2 = cov(P); ell(g, X, Y, REAL.m, S1, c.blue); ell(g, X, Y, P.m, S2, c.orange);
      const pop = fd(REAL.m, S1, P.m, S2);
      let est = '—';
      if (n >= 3) { const a = samp(real), b = samp(gen); est = fd(a.m, a.S, b.m, b.S).d.toFixed(3); }
      q(ctl, '#out').innerHTML = `population: |μ₁−μ₂|² = <b>${pop.mt.toFixed(3)}</b>; Tr(Σ₁+Σ₂−2(Σ₁Σ₂)^(1/2)) = <b>${pop.tr.toFixed(3)}</b>; Fréchet distance² = <b>${pop.d.toFixed(3)}</b><br>` +
        `estimated from the ${n} point pairs so far: <b>${est}</b> (small samples give noisy estimates)`;
      showVals(ctl, { mx: (v) => v, my: (v) => v, sx: (v) => v, sy: (v) => v, rho: (v) => v.toFixed(1), rate: (v) => v + '/s' });
    };
    const adv = () => { if (n < 400) { n++; dirty = true; } };
    wire(ctl, st, adv, reset, draw);
    ['mx', 'my', 'sx', 'sy', 'rho', 'rate'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    reset(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && n < 400) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     8. Calibration: reliability diagram and ECE
     ===================================================================== */
  AI.labs['gen-mm-calib'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Calibration: confidence versus how often the model is right',
      'illustrative: 400 questions arrive one at a time; each has a confidence and a right/wrong outcome',
      `<canvas style="height:340px"></canvas>`,
      `${sl('k', 'Overconfidence k (1 = calibrated, above 1 = overconfident, below 1 = underconfident)', 0.5, 3, 0.1, 1.8)}
       ${sl('s', 'Difficulty (higher = the model is right less often)', 0.5, 3, 0.25, 1)}
       ${sl('rate', 'Questions per second', 2, 60, 2, 12)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, NQ = 400;
    let U, Wv, n, tick = AI.ticker(), dirty = true;
    const reset = () => { const r = AI.rng(17); U = []; Wv = []; for (let i = 0; i < NQ; i++) { U.push(r()); Wv.push(r()); } n = 0; dirty = true; };
    const stats = () => {
      const k = +q(ctl, '#k').value, s = +q(ctl, '#s').value, bn = Array.from({ length: 5 }, () => ({ n: 0, c: 0, ok: 0 }));
      let ok = 0, cs = 0;
      for (let i = 0; i < n; i++) {
        const p = clamp(0.5 + 0.5 * Math.pow(U[i], s), 0.501, 0.999), y = Wv[i] < p ? 1 : 0, lg = Math.log(p / (1 - p)), cf = 1 / (1 + Math.exp(-k * lg)), b = Math.min(4, Math.floor((cf - 0.5) / 0.1));
        bn[b].n++; bn[b].c += cf; bn[b].ok += y; ok += y; cs += cf;
      }
      let ece = 0; bn.forEach((b) => { if (b.n) ece += b.n / n * Math.abs(b.ok / b.n - b.c / b.n); });
      return { bn, ok, cs, ece };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, S = stats(); g.clearRect(0, 0, W, H);
      const L = 46, R = W - 14, T = 20, B = H - 36, X = (v) => L + (v - 0.5) / 0.5 * (R - L), Y = (v) => B - v * (B - T);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, T); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      for (const v of [0, 0.5, 1]) txt(g, v.toFixed(1), L - 6, Y(v) + 4, c.muted, '11px sans-serif', 'right');
      for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1]) txt(g, v.toFixed(1), X(v), B + 14, c.muted, '11px sans-serif', 'center');
      txt(g, 'confidence the model reports', (L + R) / 2, H - 6, c.muted, '12px sans-serif', 'center'); txt(g, 'accuracy', 6, 14, c.muted, '12px sans-serif');
      g.strokeStyle = c.fg; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(0.5), Y(0.5)); g.lineTo(X(1), Y(1)); g.stroke(); g.setLineDash([]);
      txt(g, 'perfect calibration', X(0.78), Y(0.78) + 18, c.fg, '11px sans-serif');
      S.bn.forEach((b, i) => {
        if (!b.n) return; const x0 = X(0.5 + i * 0.1), x1 = X(0.6 + i * 0.1), acc = b.ok / b.n, cf = b.c / b.n;
        g.fillStyle = c.blue; g.globalAlpha = 0.65; g.fillRect(x0 + 3, Y(acc), x1 - x0 - 6, B - Y(acc)); g.globalAlpha = 1;
        g.strokeStyle = cf > acc ? c.red : c.green; g.lineWidth = 3; g.beginPath(); g.moveTo(X(cf), Y(cf)); g.lineTo(X(cf), Y(acc)); g.stroke(); g.lineWidth = 1;
        txt(g, 'n=' + b.n, (x0 + x1) / 2, Y(acc) - 4, c.fg, '10px sans-serif', 'center');
      });
      q(ctl, '#out').innerHTML = n ? `${n} questions: accuracy <b>${(S.ok / n).toFixed(3)}</b>, mean confidence <b>${(S.cs / n).toFixed(3)}</b>, expected calibration error ECE = Σ (n_b/n)·|acc_b − conf_b| = <b>${S.ece.toFixed(3)}</b>` : 'Press play: questions arrive one at a time.';
      const k = +q(ctl, '#k').value;
      q(ctl, '#cap').textContent = !n ? '' : (Math.abs(k - 1) < 0.15 ? 'Calibrated: bars follow the diagonal, ECE is small. Raise the difficulty and the model is calibrated yet often wrong: calibration says nothing about being right.' : k > 1 ? 'Overconfident: the bars sit below the diagonal (red gaps). It says 90% when it is right far less often.' : 'Underconfident: the bars sit above the diagonal; it is right more often than it claims.');
      showVals(ctl, { k: (v) => v.toFixed(1), s: (v) => v.toFixed(2), rate: (v) => v + '/s' });
    };
    const adv = () => { if (n < NQ) { n++; dirty = true; } };
    wire(ctl, st, adv, reset, draw);
    ['k', 's', 'rate'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    reset(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && n < NQ) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     9. Provenance: signed metadata (toy hash, illustrative only)
     ===================================================================== */
  AI.labs['gen-mm-provenance'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Content provenance: signed metadata and what verification can and cannot say',
      'illustrative: a toy 32-bit hash stands in for a real cryptographic hash and signature; it is not secure',
      `<div id="stage"></div>`,
      `<label>Content (edit it to tamper with it) <input type="text" id="content" value="harbour at sunset, made with Tool X" style="width:100%"></label>
       <div class="row"><button class="btn" id="sign">Re-sign the current content as Tool X</button><button class="btn" id="forge">Sign with a different key</button><button class="btn" id="strip">Strip the metadata</button></div>
       ${sl('speed', 'Speed', 0.25, 2, 0.25, 1)}
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Verify again</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:4em"></div>
       <p class="caption" id="cap"></p>`);
    const stage = q(view, '#stage'), st = { playing: true };
    const h32 = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); };
    const sigOf = (hash, creator, key) => h32(key + '|' + hash + '|' + creator);
    const TRUSTED = 'trusted-key-X';
    let man, step = 0, tt = 0, dirty = true;
    const init = () => { const ct = q(ctl, '#content').value; man = { hash: h32(ct), creator: 'Tool X', sig: sigOf(h32(ct), 'Tool X', TRUSTED), present: true }; step = 0; tt = 0; dirty = true; };
    const verdict = () => {
      const ct = q(ctl, '#content').value, hNow = h32(ct);
      if (!man.present) return { code: 'none', rows: [] };
      const ok1 = hNow === man.hash, ok2 = sigOf(man.hash, man.creator, TRUSTED) === man.sig;
      return { code: !ok1 ? 'changed' : !ok2 ? 'badsig' : 'ok', hNow, ok1, ok2 };
    };
    const draw = () => {
      const v = verdict(), rows = [];
      if (!man.present) rows.push('<div class="callout warn"><b>No credential attached.</b> Nothing can be verified. This is <i>not</i> evidence that the content is fake or real: most content has never carried a credential, and metadata is easily lost when a file is re-saved or shared.</div>');
      else {
        rows.push(`<div style="font-family:monospace;font-size:13px;white-space:normal">manifest → content hash <b>${man.hash}</b>, claim "created by ${man.creator}", signature <b>${man.sig}</b></div>`);
        const lines = [
          ['1 Recompute the hash of the content you hold', `hash(content) = ${v.hNow}`],
          ['2 Compare with the hash in the manifest', v.ok1 ? 'equal: the content is exactly what was signed' : 'NOT equal: the content changed after signing'],
          ['3 Check the signature with the signer’s trusted key', v.ok2 ? 'valid: the claim was signed by the key we trust' : 'INVALID: not signed by the trusted key']
        ];
        lines.forEach((l, i) => {
          const on = step > i, bad = (i === 1 && !v.ok1) || (i === 2 && !v.ok2);
          rows.push(`<div style="padding:6px 10px;margin:4px 0;border-left:4px solid ${on ? (bad ? 'var(--red)' : 'var(--green)') : 'var(--border)'};opacity:${on ? 1 : 0.4}"><b>${l[0]}</b><br><span style="font-size:13px">${on ? l[1] : '…'}</span></div>`);
        });
        if (step >= 3) {
          const msg = v.code === 'ok' ? '<b style="color:var(--green)">Verified:</b> this content is unchanged since Tool X signed it. It does not prove the content is true, only where it came from and that it was not altered since.'
            : v.code === 'changed' ? '<b style="color:var(--red)">Verification failed:</b> the content no longer matches what was signed, so any edit is detected (a harmless re-save counts as an edit unless the edit is itself recorded).'
              : '<b style="color:var(--red)">Verification failed:</b> the signature was made with a different key, so the origin claim cannot be trusted.';
          rows.push(`<div class="callout" style="margin-top:8px">${msg}</div>`);
        }
      }
      stage.innerHTML = rows.join('');
      q(ctl, '#out').textContent = man.present ? `current content hash ${h32(q(ctl, '#content').value)}; manifest hash ${man.hash}` : 'manifest removed';
      q(ctl, '#cap').textContent = 'Provenance is a chain of evidence about origin and edits. A valid signature does not say the content is truthful, and a missing one does not say it is false.';
      showVals(ctl, { speed: (v) => v + '×' });
    };
    const adv = () => { if (man.present && step < 3) { step++; tt = 0; dirty = true; } };
    q(ctl, '#play').onclick = () => { st.playing = !st.playing; q(ctl, '#play').textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; q(ctl, '#play').textContent = '▶ Play'; adv(); draw(); };
    q(ctl, '#reset').onclick = () => { step = 0; tt = 0; dirty = true; };
    q(ctl, '#content').oninput = () => { step = 0; tt = 0; dirty = true; };
    q(ctl, '#sign').onclick = () => { const ct = q(ctl, '#content').value; man = { hash: h32(ct), creator: 'Tool X', sig: sigOf(h32(ct), 'Tool X', TRUSTED), present: true }; step = 0; dirty = true; };
    q(ctl, '#forge').onclick = () => { const ct = q(ctl, '#content').value; man = { hash: h32(ct), creator: 'Tool X', sig: sigOf(h32(ct), 'Tool X', 'someone-else'), present: true }; step = 0; dirty = true; };
    q(ctl, '#strip').onclick = () => { man.present = false; step = 0; dirty = true; };
    q(ctl, '#speed').oninput = () => { dirty = true; };
    init(); draw();
    AI.loop(root, (dt) => { if (st.playing && man.present && step < 3) { tt += dt * (+q(ctl, '#speed').value); if (tt > 1.5) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     10. Statistical watermark detection (green-list z-test) and its limits
     ===================================================================== */
  AI.labs['gen-mm-watermark'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Detecting a statistical text watermark, and where it stops working',
      'illustrative: a token counts as "green" if a secret rule marks it so; a watermarking generator is nudged toward green tokens, ordinary text hits green only by chance',
      `<canvas style="height:320px"></canvas>`,
      `${sl('g', 'Green fraction γ (chance an ordinary token is green)', 0.1, 0.5, 0.05, 0.25)}
       ${sl('pw', 'How strongly the generator prefers green tokens', 0.25, 0.7, 0.05, 0.4)}
       ${sl('rw', 'Share of tokens later replaced by other text (rewriting)', 0, 1, 0.05, 0)}
       ${sl('rate', 'Tokens per second', 2, 60, 2, 12)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true }, NT = 400, ZT = 4;
    let uw, un, ur, uz, n, tick = AI.ticker(), dirty = true;
    const reset = () => { const r = AI.rng(55); uw = []; un = []; ur = []; uz = []; for (let i = 0; i < NT; i++) { uw.push(r()); un.push(r()); ur.push(r()); uz.push(r()); } n = 0; dirty = true; };
    const series = () => {
      const g = +q(ctl, '#g').value, pw = +q(ctl, '#pw').value, rw = +q(ctl, '#rw').value, zw = [], zn = []; let a = 0, b = 0;
      for (let i = 0; i < NT; i++) {
        const sw = ur[i] < rw ? (uz[i] < g) : (uw[i] < pw), sn = un[i] < g;
        a += sw ? 1 : 0; b += sn ? 1 : 0; const t = i + 1, sd = Math.sqrt(t * g * (1 - g));
        zw.push((a - g * t) / sd); zn.push((b - g * t) / sd);
      }
      return { zw, zn, g };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, S = series(); g.clearRect(0, 0, W, H);
      const L = 40, R = W - 12, T = 16, B = H - 30, zmin = -3, zmax = 12, X = (i) => L + i / NT * (R - L), Y = (z) => B - (z - zmin) / (zmax - zmin) * (B - T);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, T); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      for (const z of [0, 4, 8, 12]) { txt(g, z, L - 6, Y(z) + 4, c.muted, '11px sans-serif', 'right'); }
      txt(g, 'tokens read', (L + R) / 2, H - 6, c.muted, '12px sans-serif', 'center'); txt(g, 'z-score', 6, 12, c.muted, '12px sans-serif');
      g.strokeStyle = c.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(L, Y(ZT)); g.lineTo(R, Y(ZT)); g.stroke(); g.setLineDash([]); txt(g, 'detection threshold z = 4', R, Y(ZT) - 4, c.red, '11px sans-serif', 'right');
      const line = (z, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); for (let i = 0; i < n; i++) { const y = Y(clamp(z[i], zmin, zmax)); i ? g.lineTo(X(i + 1), y) : g.moveTo(X(i + 1), y); } g.stroke(); g.lineWidth = 1; };
      line(S.zn, c.blue); line(S.zw, c.orange);
      txt(g, 'orange: generator with watermark (after rewriting)', L + 8, T + 12, c.orange, '11px sans-serif'); txt(g, 'blue: ordinary text', L + 8, T + 26, c.blue, '11px sans-serif');
      const zw = n ? S.zw[n - 1] : 0, zn = n ? S.zn[n - 1] : 0, p = 1 - Phi(ZT);
      q(ctl, '#out').innerHTML = `after ${n} tokens: z = (green − γ·n) / √(n·γ(1−γ)): watermarked stream <b>${zw.toFixed(2)}</b>, ordinary stream <b>${zn.toFixed(2)}</b><br>` +
        `for ordinary text, P(z ≥ 4) ≈ <b>${p.toExponential(2)}</b> per test (a small false-positive rate, not zero). Detected: ${zw >= ZT ? 'watermarked stream yes' : 'watermarked stream not yet'}, ${zn >= ZT ? 'ordinary stream yes (a false alarm)' : 'ordinary stream no'}`;
      const rw = +q(ctl, '#rw').value;
      q(ctl, '#cap').textContent = rw >= 0.5 ? 'Heavy rewriting replaces the marked tokens, so the green excess shrinks and the z-score flattens. Detection is statistical evidence that weakens as text is changed; it is not proof either way.' : 'The watermarked stream drifts upward because it picks green tokens more often than chance; ordinary text wanders around zero.';
      showVals(ctl, { g: (v) => v.toFixed(2), pw: (v) => v.toFixed(2), rw: (v) => Math.round(v * 100) + '%', rate: (v) => v + '/s' });
    };
    const adv = () => { if (n < NT) { n++; dirty = true; } };
    wire(ctl, st, adv, reset, draw);
    ['g', 'pw', 'rw', 'rate'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    reset(); o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && n < NT) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     11. Cost and latency budget
     ===================================================================== */
  AI.labs['gen-mm-budget'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A latency and cost budget for one request',
      'illustrative round numbers, not real prices or measured speeds. The playhead replays one request in simulated seconds',
      `<canvas style="height:260px"></canvas>`,
      `${sl('P', 'Prompt tokens', 100, 8000, 100, 1500)}
       ${sl('O', 'Output tokens', 20, 1000, 20, 300)}
       ${sl('pre', 'Prefill speed (tokens per second)', 1000, 20000, 500, 5000)}
       ${sl('dec', 'Decode speed (tokens per second)', 10, 100, 5, 40)}
       ${sl('qw', 'Queue wait (seconds)', 0, 2, 0.1, 0.1)}
       ${sl('pin', 'Input price ($ per million tokens)', 0.5, 10, 0.5, 2)}
       ${sl('pout', 'Output price ($ per million tokens)', 1, 40, 1, 8)}
       ${sl('cp', 'Prefix tokens that can be cached (shared system prompt)', 0, 3000, 100, 1000)}
       ${sl('h', 'Cache hit rate', 0, 1, 0.1, 0.6)}
       ${sl('bud', 'Latency budget (seconds)', 1, 15, 0.5, 8)}
       ${sl('speed', 'Playback speed', 0.25, 4, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let t = 0, dirty = true;
    const calc = () => {
      const v = (id) => +q(ctl, '#' + id).value, P = v('P'), O = v('O'), pre = v('pre'), dec = v('dec'), qw = v('qw'), pin = v('pin') * 1e-6, po = v('pout') * 1e-6, cp = Math.min(v('cp'), P), h = v('h'), bud = v('bud');
      const pfM = P / pre, pfH = (P - cp) / pre, dd = (O - 1) / dec;
      const cM = P * pin + O * po, cH = cp * pin * 0.1 + (P - cp) * pin + O * po;
      return { P, O, qw, pfM, pfH, dd, ttftM: qw + pfM, ttftH: qw + pfH, latM: qw + pfM + dd, latH: qw + pfH + dd, cM, cH, cE: h * cH + (1 - h) * cM, h, bud, cp };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h, k = calc(); g.clearRect(0, 0, W, H);
      const L = 90, R = W - 14, tmax = Math.max(k.bud, k.latM) * 1.1, X = (s) => L + s / tmax * (R - L);
      const lane = (y, label, qw, pf, dd) => {
        txt(g, label, 8, y + 22, c.fg, 'bold 12px sans-serif');
        const seg = (a, b, col) => { const aa = Math.min(a, t), bb = Math.min(b, t); if (bb > aa) { g.fillStyle = col; g.globalAlpha = 0.8; g.fillRect(X(aa), y, X(bb) - X(aa), 34); g.globalAlpha = 1; } };
        seg(0, qw, c.muted); seg(qw, qw + pf, c.blue); seg(qw + pf, qw + pf + dd, c.green);
        if (t > qw + pf) { const nt = Math.min(k.O, Math.floor((t - qw - pf) * (k.dd > 0 ? (k.O - 1) / k.dd : 0)) + 1); txt(g, nt + ' tokens out', X(Math.min(t, qw + pf + dd)) + 6, y + 22, c.fg, '11px sans-serif'); }
        txt(g, 'total ' + (qw + pf + dd).toFixed(2) + ' s', X(qw + pf + dd) - 2, y + 52, c.muted, '11px sans-serif', 'right');
      };
      lane(30, 'cache miss', k.qw, k.pfM, k.dd); lane(110, 'cache hit', k.qw, k.pfH, k.dd);
      g.strokeStyle = c.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(X(k.bud), 10); g.lineTo(X(k.bud), 175); g.stroke(); g.setLineDash([]); txt(g, 'budget ' + k.bud + ' s', X(k.bud) - 4, 12, c.red, '11px sans-serif', 'right');
      g.strokeStyle = c.fg; g.beginPath(); g.moveTo(X(t), 14); g.lineTo(X(t), 178); g.stroke();
      for (let s = 0; s <= tmax; s += 2) txt(g, s + ' s', X(s), 196, c.muted, '11px sans-serif', 'center');
      let lx = L; [['queue', c.muted], ['prefill (read the prompt)', c.blue], ['decode (write tokens)', c.green]].forEach(([tx, col]) => { g.fillStyle = col; g.fillRect(lx, 222, 11, 11); txt(g, tx, lx + 15, 232, c.fg, '11px sans-serif'); lx += 28 + g.measureText(tx).width + 14; });
      q(ctl, '#out').innerHTML = `time to first token: miss ${k.qw.toFixed(2)} + ${k.P}/${(+q(ctl, '#pre').value)} = <b>${k.ttftM.toFixed(2)} s</b>; hit ${k.qw.toFixed(2)} + ${(k.P - k.cp)}/${(+q(ctl, '#pre').value)} = <b>${k.ttftH.toFixed(2)} s</b><br>` +
        `decode time (O−1)/rate = ${k.dd.toFixed(2)} s; total latency: miss <b>${k.latM.toFixed(2)} s</b>, hit <b>${k.latH.toFixed(2)} s</b> (budget ${k.bud} s: ${k.latM <= k.bud ? 'met' : 'MISSED'} on a miss, ${k.latH <= k.bud ? 'met' : 'MISSED'} on a hit)<br>` +
        `cost per request: miss <b>$${k.cM.toFixed(5)}</b>, hit <b>$${k.cH.toFixed(5)}</b> (cached tokens at 10% of the input price); at hit rate ${k.h.toFixed(1)}: <b>$${k.cE.toFixed(5)}</b><br>` +
        `for 1,000,000 requests a month: <b>$${Math.round(k.cE * 1e6).toLocaleString()}</b> (all misses: $${Math.round(k.cM * 1e6).toLocaleString()})`;
      q(ctl, '#cap').textContent = 'Reading the prompt is fast per token; writing the answer is slow per token, so for long answers the green bar dominates both latency and (at these prices) cost. All numbers here are invented round numbers.';
      showVals(ctl, { P: (v) => v, O: (v) => v, pre: (v) => v, dec: (v) => v, qw: (v) => v.toFixed(1) + ' s', pin: (v) => '$' + v, pout: (v) => '$' + v, cp: (v) => v, h: (v) => Math.round(v * 100) + '%', bud: (v) => v + ' s', speed: (v) => v + '×' });
    };
    const tEnd = () => { const k = calc(); return Math.max(k.latM, k.bud) * 1.1; };
    wire(ctl, st, () => { t = Math.min(tEnd(), t + 0.5); dirty = true; }, () => { t = 0; dirty = true; }, draw);
    ['P', 'O', 'pre', 'dec', 'qw', 'pin', 'pout', 'cp', 'h', 'bud', 'speed'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing) { t += dt * (+q(ctl, '#speed').value); if (t > tEnd() + 1.5) t = 0; dirty = true; } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };

  /* =====================================================================
     12. Guardrail threshold: false positives versus false negatives
     ===================================================================== */
  AI.labs['gen-mm-guard'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A guardrail classifier: moving the threshold',
      'illustrative: 20 labelled requests (10 benign, 10 harmful) with the classifier’s risk score; a request is blocked if score ≥ threshold',
      `<canvas style="height:250px"></canvas>`,
      `${sl('th', 'Block threshold', 0.1, 0.95, 0.05, 0.5)}
       <label>Share of real traffic that is harmful <select id="prev"><option value="0.001">0.1%</option><option value="0.01" selected>1%</option><option value="0.05">5%</option><option value="0.2">20%</option></select></label>
       ${sl('rate', 'Requests checked per second', 0.5, 6, 0.5, 1.5)}
       ${btns}
       <div class="readout" id="out" style="white-space:normal;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const ben = [0.05, 0.10, 0.12, 0.15, 0.22, 0.30, 0.35, 0.42, 0.55, 0.62], har = [0.35, 0.48, 0.58, 0.65, 0.72, 0.78, 0.85, 0.90, 0.93, 0.97];
    const items = ben.map((s) => ({ s, h: 0 })).concat(har.map((s) => ({ s, h: 1 })));
    const rr = AI.rng(8), ord = items.map((_, i) => i); for (let i = ord.length - 1; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
    const st = { playing: true };
    let n = 0, tick = AI.ticker(), dirty = true;
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const th = +q(ctl, '#th').value, L = 40, R = W - 14, X = (s) => L + s * (R - L), yb = 60, yh = 130;
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, 185); g.lineTo(R, 185); g.stroke();
      for (const v of [0, 0.25, 0.5, 0.75, 1]) txt(g, v.toFixed(2), X(v), 202, c.muted, '11px sans-serif', 'center');
      txt(g, 'risk score from the classifier', (L + R) / 2, 222, c.muted, '12px sans-serif', 'center');
      txt(g, 'benign requests', L, yb - 28, c.blue, 'bold 11px sans-serif'); txt(g, 'harmful requests', L, yh - 28, c.red, 'bold 11px sans-serif');
      g.fillStyle = c.red; g.globalAlpha = 0.1; g.fillRect(X(th), 10, R - X(th), 175); g.globalAlpha = 1;
      g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(X(th), 10); g.lineTo(X(th), 185); g.stroke(); g.lineWidth = 1;
      txt(g, 'blocked →', X(th) + 6, 22, c.red, 'bold 11px sans-serif');
      let tp = 0, fp = 0, fn = 0, tn = 0;
      ord.forEach((ix, r) => {
        const it = items[ix], shown = r < n, flag = it.s >= th; if (shown) { if (it.h) { flag ? tp++ : fn++; } else { flag ? fp++ : tn++; } }
        const x = X(it.s), y = it.h ? yh : yb, jit = ((ix * 7) % 5 - 2) * 4;
        g.globalAlpha = shown ? 1 : 0.3; g.fillStyle = it.h ? c.red : c.blue; g.beginPath(); g.arc(x, y + jit, 7, 0, 6.3); g.fill(); g.globalAlpha = 1;
        if (shown) { g.strokeStyle = flag ? c.orange : c.green; g.lineWidth = 2; g.beginPath(); g.arc(x, y + jit, 10, 0, 6.3); g.stroke(); g.lineWidth = 1; const bad = (it.h && !flag) || (!it.h && flag); if (bad) txt(g, '✕', x, y + jit + 4, c.fg, 'bold 11px sans-serif', 'center'); }
      });
      const fpr = fp / 10, tpr = tp / 10, fnr = fn / 10, pr = +q(ctl, '#prev').value, ppv = tpr * pr / (tpr * pr + fpr * (1 - pr));
      const allF = ben.filter((s) => s >= th).length, allN = har.filter((s) => s < th).length;
      q(ctl, '#out').innerHTML = `checked ${n} of 20: harmful blocked ${tp}, harmful missed (false negative) ${fn}, benign blocked (false positive) ${fp}, benign passed ${tn}<br>` +
        `on all 20 at threshold ${th.toFixed(2)}: false positives <b>${allF}</b> of 10 benign (FPR ${allF / 10}), false negatives <b>${allN}</b> of 10 harmful (FNR ${allN / 10})<br>` +
        `if only ${(pr * 100).toString()}% of real traffic is harmful, the share of blocked requests that are truly harmful is TPR·π / (TPR·π + FPR·(1−π)) = <b>${(((10 - allN) / 10) * pr / (((10 - allN) / 10) * pr + (allF / 10) * (1 - pr))).toFixed(3)}</b>`;
      q(ctl, '#cap').textContent = 'Ring colour: orange = blocked, green = passed; ✕ marks a mistake. Raising the threshold removes false positives but lets more harmful requests through; lowering it does the opposite. No threshold removes both here, because the two groups overlap.';
      showVals(ctl, { th: (v) => v.toFixed(2), rate: (v) => v + '/s' });
    };
    const adv = () => { if (n < 20) { n++; dirty = true; } };
    wire(ctl, st, adv, () => { n = 0; dirty = true; }, draw);
    q(ctl, '#th').oninput = () => { n = 0; dirty = true; };
    q(ctl, '#prev').onchange = () => { dirty = true; };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => { if (st.playing && n < 20) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) adv(); } if (dirty) { draw(); dirty = false; } }, { fps: 30 });
  };
})();
