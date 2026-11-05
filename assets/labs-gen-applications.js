/* Generative AI Module 6 labs: prompting, retrieval and agents.
   gen-ap-fewshot, gen-ap-grammar, gen-ap-cosine, gen-ap-chunks, gen-ap-rag, gen-ap-react, gen-ap-compound, gen-ap-context.
   There is no language model in the browser: every lab that stands in for a model uses a small explicit toy rule that is printed on the page.
   All simulations are illustrative, use seeded random numbers (AI.rng), run on dt, start slowly, and have speed / pause / step / reset. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const cl = (v, a, b) => Math.min(b, Math.max(a, v));
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const rrect = (g, x, y, w, h, r) => { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };

  /* ============================== pure helpers (also exported for testing) ============================== */
  const softmax = (a) => { const m = Math.max(...a); const e = a.map((x) => (x === -Infinity ? 0 : Math.exp(x - m))); const s = e.reduce((x, y) => x + y, 0); return e.map((x) => x / s); };
  const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };

  /* --- 1. few-shot toy: logit_l = base_l + alpha * (number of examples labelled l) + beta * [the last example is labelled l] --- */
  const LABELS = ['positive', 'negative', 'neutral'];
  const ICL_INPUTS = [
    { name: '"It arrived on Tuesday."', b: [0.4, 0.2, 0.9] },
    { name: '"Not bad at all."', b: [1.0, 0.3, 0.5] },
    { name: '"Broken on arrival. Awful."', b: [-1.0, 2.0, 0.0] },
  ];
  const iclSeq = (c, mode) => {
    const seq = [];
    if (mode === 'blocks') { for (let l = 0; l < 3; l++) for (let i = 0; i < c[l]; i++) seq.push(l); }
    else if (mode === 'rev') { for (let l = 2; l >= 0; l--) for (let i = 0; i < c[l]; i++) seq.push(l); }
    else { const left = c.slice(); let any = true; while (any) { any = false; for (let l = 0; l < 3; l++) if (left[l] > 0) { seq.push(l); left[l]--; any = true; } } }
    return seq;
  };
  const iclLogits = (b, seq, k, alpha, beta) => { const l = b.slice(); for (let i = 0; i < k; i++) l[seq[i]] += alpha; if (k > 0) l[seq[k - 1]] += beta; return l; };

  /* --- 2. grammar-constrained decoding on a tiny JSON schema: {"n": 7|0 [, "ok": true|false]} --- */
  const V = ['{', '}', '"n"', '"ok"', ':', ',', '7', '0', 'true', 'false', '<eos>'];
  const BASE = [-1, 0.5, -0.5, 0.2, 1.0, 0.1, 0.6, 0.3, 1.2, 0.4, 0.9];
  const GS = [
    { say: 'an opening brace', next: { '{': 1 } },
    { say: 'the key "n"', next: { '"n"': 2 } },
    { say: 'a colon', next: { ':': 3 } },
    { say: 'a digit, 7 or 0', next: { '7': 4, '0': 4 } },
    { say: 'a comma or a closing brace', next: { ',': 5, '}': 9 } },
    { say: 'the key "ok"', next: { '"ok"': 6 } },
    { say: 'a colon', next: { ':': 7 } },
    { say: 'true or false', next: { 'true': 8, 'false': 8 } },
    { say: 'a closing brace', next: { '}': 9 } },
    { say: 'end of text', next: { '<eos>': -1 } },
  ];
  const gramLogits = (s, bonus, extra) => V.map((t, i) => BASE[i] + (GS[s].next[t] !== undefined ? bonus : 0) + (extra && extra.i === i ? extra.v : 0));
  const gramLegal = (s) => V.map((t) => GS[s].next[t] !== undefined);
  const gramValidity = (bonus, T, extra) => {
    const f = (s) => { if (s === -1) return 1; const p = softmax(gramLogits(s, bonus, extra).map((x) => x / T)); let tot = 0; V.forEach((t, i) => { if (GS[s].next[t] !== undefined) tot += p[i] * f(GS[s].next[t]); }); return tot; };
    return f(0);
  };

  /* --- 3. embeddings: cosine on small vectors, and a chunking toy corpus --- */
  const CH = [
    { name: 'A', v: [0.9, 0.2, 0.5] },
    { name: 'B', v: [0.1, 0.9, 0.3] },
    { name: 'C', v: [0.7, 0.1, 0.8] },
  ];
  const D = 8, NT = 8, SEC = 3, NQ = 5;
  const retrBuild = (seed) => {
    const r = AI.rng(seed), cen = [];
    for (let t = 0; t < NT; t++) { const v = Array.from({ length: D }, () => AI.randn(r)); const n = Math.hypot(...v); cen.push(v.map((x) => x / n)); }
    const sent = [];
    for (let t = 0; t < NT; t++) for (let j = 0; j < SEC; j++) sent.push({ t, j, v: cen[t].map((x) => x + 1.2 * AI.randn(r) / Math.sqrt(D)) });
    const qs = [];
    for (let t = 0; t < NT; t++) for (let k = 0; k < NQ; k++) qs.push({ t, c: cen[t], z: Array.from({ length: D }, () => AI.randn(r) / Math.sqrt(D)) });
    return { sent, qs, N: sent.length };
  };
  const qvec = (qq, sigq) => qq.c.map((x, i) => x + sigq * qq.z[i]);
  const retrChunks = (C, c, overlap) => {
    const stride = Math.max(1, c - Math.min(overlap, c - 1)), ch = [];
    for (let s = 0; s < C.N; s += stride) {
      const idx = []; for (let i = s; i < Math.min(s + c, C.N); i++) idx.push(i);
      ch.push({ idx, v: C.sent[0].v.map((_, d) => idx.reduce((a, i) => a + C.sent[i].v[d], 0) / idx.length) });
      if (s + c >= C.N) break;
    }
    return ch;
  };
  const retrTop = (ch, qv, k) => ch.map((h, i) => ({ i, s: cosine(h.v, qv) })).sort((a, b) => b.s - a.s).slice(0, k);
  const retrNeed = (qq) => [qq.t * SEC + 1, qq.t * SEC + 2];
  const retrEval = (C, c, overlap, k, sigq) => {
    const ch = retrChunks(C, c, overlap); let tot = 0, sent = 0;
    for (const qq of C.qs) {
      const top = retrTop(ch, qvec(qq, sigq), k), got = new Set(); top.forEach((x) => ch[x.i].idx.forEach((i) => got.add(i)));
      tot += retrNeed(qq).filter((n) => got.has(n)).length / 2; sent += got.size;
    }
    return { recall: tot / C.qs.length, sent: sent / C.qs.length };
  };

  /* --- 4. retrieval metrics --- */
  const RAG_DOCS = ['shipping times', 'return window: 30 days', 'gift cards', 'how to use the returns portal', 'who pays return shipping', 'warranty terms', 'store locations', 'privacy notice', 'sale items: exceptions', 'contact us'];
  const RETR = [4, 2, 7, 5, 1, 9, 3, 8, 6, 10];
  const RERANK = [2, 5, 9, 4, 1, 7, 3, 8, 6, 10];
  const ragMetrics = (rank, rel, k) => {
    let found = 0, first = 0;
    rank.forEach((d, i) => { if (rel.has(d) && i < k) found++; if (rel.has(d) && !first) first = i + 1; });
    return { found, recall: rel.size ? found / rel.size : 0, precision: found / k, first, mrr: first ? 1 / first : 0 };
  };
  const ragAnswer = (recall, k, found, pFull, pNone, delta) => cl(pNone + (pFull - pNone) * recall - delta * (k - found), 0, 1);

  /* --- 5. agents: deterministic tools, a scripted (non-learned) plan, compounding --- */
  const PRICES = { notebook: 4, pen: 1.5, stapler: 7, ruler: 2.25, folder: 0.8 };
  const calcEval = (s) => {
    let i = 0; const src = String(s).replace(/\s+/g, '');
    const num = () => { const m = /^\d+(\.\d+)?/.exec(src.slice(i)); if (!m) throw new Error('bad number'); i += m[0].length; return +m[0]; };
    const atom = () => { if (src[i] === '(') { i++; const v = expr(); if (src[i] !== ')') throw new Error('missing )'); i++; return v; } return num(); };
    const term = () => { let v = atom(); while (src[i] === '*' || src[i] === '/') { const o = src[i++]; const w = atom(); v = o === '*' ? v * w : v / w; } return v; };
    const expr = () => { let v = term(); while (src[i] === '+' || src[i] === '-') { const o = src[i++]; const w = term(); v = o === '+' ? v + w : v - w; } return v; };
    const v = expr(); if (i !== src.length) throw new Error('trailing text'); return v;
  };
  const tokOf = (s) => Math.ceil(s.length / 4); // rule of thumb: about 4 characters per token (illustrative)
  const reactTrace = (N, M, errStep, retry) => {
    const lines = []; let calls = 0;
    const P0 = 120;
    let ctx = P0, inTok = 0, outTok = 0, price = {};
    const push = (kind, text, callStart) => { lines.push({ kind, text, tok: tokOf(text), call: callStart }); };
    const doCall = (thought, action, obs) => {
      calls++; inTok += ctx; const t1 = tokOf(thought), t2 = tokOf(action), t3 = tokOf(obs); outTok += t1 + t2; ctx += t1 + t2 + t3;
      push('Thought', thought, calls); push('Action', action, calls); push('Observation', obs, calls);
    };
    const steps = [
      { th: 'I need the price of one notebook.', act: 'lookup("notebook")', key: 'notebook', run: () => PRICES.notebook },
      { th: 'Now the price of one pen.', act: 'lookup("pen")', key: 'pen', run: () => PRICES.pen },
    ];
    let failed = false;
    steps.forEach((s, i) => {
      if (failed) return;
      if (errStep === i + 1) {
        doCall(s.th, s.act, 'ERROR: tool timed out');
        if (retry) { doCall('The call failed. I will repeat the same call.', s.act, String(s.run())); price[s.key] = s.run(); } else { price[s.key] = null; }
      } else { doCall(s.th, s.act, String(s.run())); price[s.key] = s.run(); }
    });
    const expr = `${N}*${price.notebook}+${M}*${price.pen}`;
    const act3 = `calc("${expr}")`;
    const th3 = price.notebook === null || price.pen === null ? 'I have what I have. Compute the total.' : 'Total = notebooks times price plus pens times price.';
    let result = null;
    const runCalc = () => { try { return { v: calcEval(expr), e: null }; } catch (e) { return { v: null, e: 'ERROR: cannot evaluate "' + expr + '"' }; } };
    const r3 = runCalc();
    if (errStep === 3) {
      doCall(th3, act3, 'ERROR: tool timed out');
      if (retry) { doCall('The call failed. I will repeat the same call.', act3, String(r3.v)); result = r3.v; }
    } else { doCall(th3, act3, r3.e || String(r3.v)); result = r3.v; }
    const ans = result === null ? 'I could not complete the task.' : 'The total is ' + result + '.';
    calls++; inTok += ctx; const ta = tokOf(ans); outTok += ta; push('Answer', ans, calls);
    return { lines, calls, inTok, outTok, result, expr };
  };
  const survive = (p, n) => Math.pow(p, n);
  const effP = (p, r) => 1 - Math.pow(1 - p, r);
  const tokAt = (k, P0, s) => P0 * k + s * k * (k - 1) / 2;

  /* --- 6. context window --- */
  const CTX_N = 24, CTX_RES = 40, CTX_MEM = 300;
  const ctxBuild = () => {
    const r = AI.rng(21), len = [], vec = [];
    for (let i = 0; i < CTX_N; i++) { len.push(60 + Math.floor(r() * 200)); const v = Array.from({ length: 6 }, () => AI.randn(r)); const n = Math.hypot(...v); vec.push(v.map((x) => x / n)); }
    const z = Array.from({ length: 6 }, () => AI.randn(r) / Math.sqrt(6));
    return { len, vec, z };
  };
  const posRecall = (x) => 0.55 + 0.45 * Math.pow(2 * x - 1, 2); // an ASSUMED U shape, for illustration only
  const ctxState = (C, n, W, strat, S, F, vague) => {
    const avail0 = W - CTX_RES; let sumSummary = 0, reserved = 0;
    if (strat === 'summary') reserved = S; if (strat === 'memory') reserved = CTX_MEM;
    const fit = (avail) => { let t = 0, s = n; while (s > 0 && t + C.len[s - 1] <= avail) { t += C.len[s - 1]; s--; } return s; };
    let s = fit(avail0); if (s > 0 && reserved) s = fit(avail0 - reserved); else reserved = 0;
    const win = []; for (let i = s; i < n; i++) win.push(i);
    const evicted = []; for (let i = 0; i < s; i++) evicted.push(i);
    const evTok = evicted.reduce((a, i) => a + C.len[i], 0), winTok = win.reduce((a, i) => a + C.len[i], 0);
    let retrieved = [];
    const qv = C.vec[F].map((x, i) => x + vague * C.z[i]);
    if (strat === 'memory' && evicted.length) retrieved = evicted.map((i) => ({ i, s: cosine(C.vec[i], qv) })).sort((a, b) => b.s - a.s).slice(0, 2).map((x) => x.i);
    let status, recall = 0, pos = null;
    if (F >= n) status = 'not yet said';
    else if (F >= s) { status = 'in window'; let before = 0; for (let i = s; i < F; i++) before += C.len[i]; pos = (before + C.len[F] / 2) / winTok; recall = posRecall(pos); }
    else if (strat === 'summary') { status = 'summarised'; recall = Math.min(1, S / evTok); }
    else if (strat === 'memory') { const rk = retrieved.indexOf(F); if (rk >= 0) { status = 'retrieved from memory'; pos = 0.1; recall = posRecall(0.1); } else status = 'in memory, not retrieved'; }
    else status = 'truncated away';
    return { s, win, evicted, evTok, winTok, retrieved, status, recall, pos, reserved, qv };
  };

  AI.genAp = { softmax, cosine, iclSeq, iclLogits, ICL_INPUTS, gramLogits, gramLegal, gramValidity, V, GS, CH, retrBuild, retrChunks, retrEval, retrTop, retrNeed, qvec, ragMetrics, ragAnswer, RETR, RERANK, calcEval, reactTrace, survive, effP, tokAt, ctxBuild, ctxState, posRecall, PRICES };

  /* ===================================================================
     1. Few-shot examples shifting a label distribution
     =================================================================== */
  AI.labs['gen-ap-fewshot'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Few-shot examples shifting a label distribution', 'illustrative: a toy scoring rule, not a real language model; examples are added to the prompt one at a time',
      `<canvas style="height:300px"></canvas>`,
      `<label>Test input <select id="inp">${ICL_INPUTS.map((x, i) => `<option value="${i}">${x.name}</option>`).join('')}</select></label>
       ${sl('c0', 'Examples labelled positive', 0, 4, 1, 2)}${sl('c1', 'Examples labelled negative', 0, 4, 1, 1)}${sl('c2', 'Examples labelled neutral', 0, 4, 1, 0)}
       <label>Order of the examples <select id="ord"><option value="blocks">grouped: positive, negative, neutral</option><option value="rev">grouped: neutral, negative, positive</option><option value="alt">alternating</option></select></label>
       ${sl('al', 'Pull per example, α', 0, 2, 0.1, 0.8)}${sl('be', 'Recency bonus for the last example, β', 0, 1.5, 0.1, 0.5)}
       ${sl('rate', 'Examples added per second', 0.1, 2, 0.1, 0.4)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:7.5em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let seq = [], k = 0, acc = 0, disp = [1 / 3, 1 / 3, 1 / 3], dirty = true;
    const params = () => ({ b: ICL_INPUTS[+q(ctl, '#inp').value].b, al: +q(ctl, '#al').value, be: +q(ctl, '#be').value });
    const target = () => { const p = params(); const lg = iclLogits(p.b, seq, k, p.al, p.be); return { lg, pr: softmax(lg) }; };
    const rebuild = () => { seq = iclSeq([+q(ctl, '#c0').value, +q(ctl, '#c1').value, +q(ctl, '#c2').value], q(ctl, '#ord').value); k = 0; acc = 0; dirty = true; };
    const reset = () => { rebuild(); disp = softmax(params().b); };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const t = target(), cols = [c.green, c.red, c.blue];
      txt(g, 'next-label probability, softmax(logits)', 14, 18, c.muted);
      const L = 92, R = W - 54, top = 34, bh = 30, gap = 18;
      for (let l = 0; l < 3; l++) {
        const y = top + l * (bh + gap);
        txt(g, LABELS[l], L - 8, y + bh / 2 + 4, c.fg, '12px sans-serif', 'right');
        g.fillStyle = c.bg3; g.fillRect(L, y, R - L, bh);
        g.fillStyle = cols[l]; g.fillRect(L, y, (R - L) * disp[l], bh);
        txt(g, (disp[l] * 100).toFixed(1) + '%', L + (R - L) * disp[l] + 6, y + bh / 2 + 4, c.fg, 'bold 12px sans-serif');
      }
      const best = disp.indexOf(Math.max(...disp));
      txt(g, 'most likely label: ' + LABELS[best], 14, top + 3 * (bh + gap) + 4, c.fg, 'bold 12px sans-serif');
      // prompt chips
      let y0 = top + 3 * (bh + gap) + 26;
      txt(g, 'the prompt so far (each chip is one worked example):', 14, y0, c.muted); y0 += 14;
      let x = 14, y = y0; const cw = 46, ch = 24;
      for (let i = 0; i < seq.length; i++) {
        if (x + cw > W - 10) { x = 14; y += ch + 6; }
        const on = i < k, last = i === k - 1;
        g.globalAlpha = on ? 1 : 0.22; g.fillStyle = cols[seq[i]]; rrect(g, x, y, cw, ch, 6); g.fill(); g.globalAlpha = 1;
        if (last) { g.strokeStyle = c.fg; g.lineWidth = 2; rrect(g, x, y, cw, ch, 6); g.stroke(); g.lineWidth = 1; }
        txt(g, ['pos', 'neg', 'neu'][seq[i]], x + cw / 2, y + 16, '#0b0f14', 'bold 11px sans-serif', 'center');
        x += cw + 6;
      }
      if (x + 150 > W - 10) { x = 14; y += ch + 6; }
      g.strokeStyle = c.muted; g.setLineDash([4, 3]); rrect(g, x, y, 140, ch, 6); g.stroke(); g.setLineDash([]);
      txt(g, 'input → ?', x + 70, y + 16, c.muted, '11px sans-serif', 'center');
      const p = params();
      q(ctl, '#out').innerHTML = `logits = base + α·count + β·[last]\nbase     ${p.b.map((v) => v.toFixed(2)).join('   ')}\nlogits   ${t.lg.map((v) => v.toFixed(2)).join('   ')}\nprobs    ${t.pr.map((v) => v.toFixed(4)).join('   ')}\n(order: ${LABELS.join(', ')})`;
      q(ctl, '#cap').textContent = k === 0
        ? 'No examples yet: the toy model uses only its built-in preference for this input (the base logits).'
        : `After ${k} of ${seq.length} examples. The newest example was labelled ${LABELS[seq[k - 1]]}: its label gained α, and also the recency bonus β until the next example arrives. Real models also show order effects, but this rule is a stand-in, not a measurement.`;
      showVals(ctl, { c0: (v) => v, c1: (v) => v, c2: (v) => v, al: (v) => v.toFixed(1), be: (v) => v.toFixed(1), rate: (v) => v.toFixed(1) });
    };
    wire(ctl, st, () => { if (k < seq.length) k++; dirty = true; }, () => { reset(); }, draw);
    ['inp', 'c0', 'c1', 'c2', 'ord', 'al', 'be'].forEach((id) => { q(ctl, '#' + id).oninput = () => { rebuild(); }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && k < seq.length) { acc += dt * +q(ctl, '#rate').value; if (acc >= 1) { acc = 0; k++; dirty = true; } }
      const tp = target().pr; let mv = false;
      for (let i = 0; i < 3; i++) { const d = tp[i] - disp[i]; if (Math.abs(d) > 5e-4) { disp[i] += d * (1 - Math.exp(-dt * 3)); mv = true; } else disp[i] = tp[i]; }
      if (mv || dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     2. Grammar-masked decoding
     =================================================================== */
  AI.labs['gen-ap-grammar'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Constrained decoding: mask illegal tokens, renormalise', 'illustrative: a toy next-token table on an 11-token vocabulary; the mask-and-renormalise step is the real mechanism',
      `<canvas style="height:400px"></canvas>`,
      `<label>Decoding <select id="mode"><option value="c">constrained: mask with the grammar</option><option value="u">unconstrained: sample the raw distribution</option></select></label>
       ${sl('bon', 'How strongly the toy model already prefers legal tokens (logit bonus)', 0, 8, 0.5, 4)}
       ${sl('tmp', 'Temperature T', 0.3, 2, 0.1, 1)}
       <label>Edit one token's logit <select id="tok">${V.map((t, i) => `<option value="${i}">${t.replace(/</g, '&lt;')}</option>`).join('')}</select></label>
       ${sl('ex', 'Extra logit added to that token at every step', -4, 4, 0.5, 0)}
       ${sl('spd', 'Speed', 0.25, 4, 0.25, 1)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:7em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const DUR = [2.6, 2.0, 2.0, 1.6];
    let s, phase, t, out, attempt, tries, valid, fail, holdT, pick, sampler, dirty = true;
    const cfg = () => ({ B: +q(ctl, '#bon').value, T: +q(ctl, '#tmp').value, extra: { i: +q(ctl, '#tok').value, v: +q(ctl, '#ex').value }, con: q(ctl, '#mode').value === 'c' });
    const dist = () => { const f = cfg(); const lg = gramLogits(s, f.B, f.extra); return { lg, p: softmax(lg.map((x) => x / f.T)), legal: gramLegal(s) }; };
    const newAttempt = (keepStats) => { s = 0; phase = 0; t = 0; out = []; fail = false; holdT = 0; pick = -1; attempt++; sampler = AI.rng(1000 + attempt * 7); if (!keepStats) { tries = 0; valid = 0; } tries++; dirty = true; };
    const reset = () => { attempt = 0; newAttempt(false); };
    const choose = () => {
      const d = dist(), f = cfg(); let p = d.p;
      if (f.con) { const m = d.lg.map((x, i) => (d.legal[i] ? x / f.T : -Infinity)); p = softmax(m); }
      let r = sampler(), a = 0, ch = p.length - 1; for (let i = 0; i < p.length; i++) { a += p[i]; if (r < a) { ch = i; break; } }
      pick = ch;
    };
    const advance = () => {
      // moves to the next phase (used by both the timer and the Step button)
      const f = cfg();
      if (fail || s === -1) { newAttempt(true); return; }
      if (phase === 0) { phase = f.con ? 1 : 3; t = 0; if (phase === 3) choose(); }
      else if (phase === 1) { phase = 2; t = 0; }
      else if (phase === 2) { phase = 3; t = 0; choose(); }
      else {
        const d = dist(); out.push(pick);
        if (!d.legal[pick]) { fail = true; holdT = 0; phase = 4; t = 0; }
        else { const nx = GS[s].next[V[pick]]; if (nx === -1) { valid++; s = -1; phase = 4; t = 0; } else { s = nx; phase = 0; t = 0; } }
      }
      dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); const f = cfg();
      // output line
      txt(g, 'output so far:', 12, 16, c.muted);
      let x = 12, y = 28; const lg0 = s === -1 ? 9 : s;
      out.forEach((tk, i) => {
        const w = g.measureText(V[tk]).width + 14; if (x + w > W - 8) { x = 12; y += 24; }
        const bad = fail && i === out.length - 1;
        g.fillStyle = bad ? c.red : c.bg3; rrect(g, x, y, w, 20, 5); g.fill(); txt(g, V[tk], x + 7, y + 14, bad ? '#0b0f14' : c.fg, '12px ui-monospace, Consolas, monospace'); x += w + 4;
      });
      if (!out.length) txt(g, '(nothing yet)', 12, y + 14, c.muted);
      const topY = 84;
      if (s !== -1 && !fail) txt(g, `grammar state ${s}: the schema allows only ${GS[s].say}`, 12, topY - 8, c.fg, 'bold 12px sans-serif');
      else if (fail) txt(g, 'INVALID: the sampled token is not allowed here, so the output cannot be parsed. A retry is needed.', 12, topY - 8, c.red, 'bold 12px sans-serif');
      else txt(g, 'valid JSON that matches the schema.', 12, topY - 8, c.green, 'bold 12px sans-serif');
      // bars
      const sd = s === -1 ? 9 : s, d = (() => { const save = s; s = sd; const r = dist(); s = save; return r; })();
      const Zl = d.p.reduce((a, v, i) => a + (d.legal[i] ? v : 0), 0);
      const L = 10, R = W - 10, bw = (R - L) / V.length, bt = topY + 14, bb = H - 120, hmax = bb - bt;
      let ymax = 0.05; d.p.forEach((v, i) => { ymax = Math.max(ymax, v, d.legal[i] ? v / Zl : 0); });
      ymax = Math.min(1, ymax * 1.12);
      const e1 = phase === 1 ? ease(Math.min(1, t / DUR[1])) : phase > 1 || phase === 4 ? 1 : 0;
      const e2 = phase === 2 ? ease(Math.min(1, t / DUR[2])) : phase === 3 ? 1 : 0;
      const masked = f.con && (phase >= 1 && phase <= 3);
      for (let i = 0; i < V.length; i++) {
        const x0 = L + i * bw + 3, w = bw - 6, leg = d.legal[i];
        let hgt = d.p[i];
        if (f.con && !leg) hgt = d.p[i] * (1 - e1);
        if (f.con && leg) hgt = d.p[i] * (1 + (1 / Zl - 1) * e2);
        const hh = (hgt / ymax) * hmax;
        g.fillStyle = leg ? c.blue : c.red; g.globalAlpha = f.con || leg ? 0.85 : 0.5; g.fillRect(x0, bb - hh, w, hh); g.globalAlpha = 1;
        if (f.con && !leg && e1 > 0) { g.fillStyle = c.bg3; g.globalAlpha = 0.85 * e1; g.fillRect(x0, bb - (d.p[i] / ymax) * hmax, w, (d.p[i] / ymax) * hmax - hh); g.globalAlpha = 1; g.strokeStyle = c.red; g.setLineDash([2, 3]); g.strokeRect(x0, bb - (d.p[i] / ymax) * hmax, w, (d.p[i] / ymax) * hmax); g.setLineDash([]); }
        if (phase === 3 && i === pick) { g.strokeStyle = c.yellow; g.lineWidth = 3; g.strokeRect(x0 - 2, bb - hh - 2, w + 4, hh + 4); g.lineWidth = 1; }
        txt(g, (hgt * 100 >= 0.05 ? (hgt * 100).toFixed(1) : '0') + '%', x0 + w / 2, bb - hh - 4, c.fg, '10px sans-serif', 'center');
        txt(g, V[i], x0 + w / 2, bb + 14, leg ? c.fg : c.muted, '10px ui-monospace, Consolas, monospace', 'center');
        txt(g, f.con && !leg && e1 > 0.6 ? '−∞' : d.lg[i].toFixed(1), x0 + w / 2, bb + 28, f.con && !leg && e1 > 0.6 ? c.red : c.muted, '10px sans-serif', 'center');
      }
            g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, bb); g.lineTo(R, bb); g.stroke();
      const nm = ['1. the raw probabilities of the toy model', '2. the grammar mask sets illegal logits to −∞ (probability 0)', '3. renormalise: divide the legal probabilities by their sum', '4. sample one token from this distribution', ''][fail || s === -1 ? 4 : phase];
      txt(g, nm, 12, bb + 52, c.yellow, 'bold 12px sans-serif');
      txt(g, `legal probability mass before the mask: ${(Zl * 100).toFixed(1)}%`, 12, bb + 70, c.muted);
      const vp = gramValidity(f.B, f.T, f.extra);
      q(ctl, '#out').innerHTML = `attempts ${tries}, valid outputs ${valid}\nchance a raw sample is valid at every step: ${(vp * 100).toFixed(2)}%\nexpected attempts until valid: ${(1 / vp).toFixed(2)}\nconstrained decoding: valid chance 100% (every step uses legal mass only)`;
      q(ctl, '#cap').textContent = f.con
        ? 'Constrained: the illegal bars are cut to zero, then the legal bars grow so they sum to 1. Nothing outside the grammar can be sampled, but the model still chooses among the legal tokens.'
        : 'Unconstrained: the raw distribution is sampled as it is. Any illegal token with some probability can be drawn, and one bad token spoils the whole output.';
      showVals(ctl, { bon: (v) => v.toFixed(1), tmp: (v) => v.toFixed(1), ex: (v) => v.toFixed(1), spd: (v) => v + '×' });
    };
    wire(ctl, st, advance, reset, draw);
    ['bon', 'tmp', 'tok', 'ex'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    q(ctl, '#mode').onchange = () => { reset(); draw(); };
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; reset(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) {
        const spd = +q(ctl, '#spd').value; t += dt * spd; dirty = true;
        if (fail || s === -1) { if (t > 2.6) advance(); } else if (t >= DUR[phase]) advance();
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     3a. Cosine similarity on small printed vectors
     =================================================================== */
  AI.labs['gen-ap-cosine'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Cosine similarity, dot product and distance', 'illustrative: three chunks with 3-number "embeddings" that were written by hand; the arithmetic is exact',
      `<canvas style="height:330px"></canvas>`,
      `${sl('q0', 'Query component 1', 0, 1.5, 0.05, 0.8)}${sl('q1', 'Query component 2', 0, 1.5, 0.05, 0.1)}${sl('q2', 'Query component 3', 0, 1.5, 0.05, 0.6)}
       ${sl('sa', 'Stretch chunk A (multiply its vector by)', 0.5, 3, 0.1, 1)}
       ${sl('spd', 'Auto-sweep speed (0 = off, uses the sliders)', 0, 1, 0.05, 0.15)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:9em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const QA = [0.8, 0.1, 0.6], QB = [0.1, 0.8, 0.4];
    let u = 0, dir = 1, dirty = true;
    const setQ = (v) => { ['q0', 'q1', 'q2'].forEach((id, i) => { q(ctl, '#' + id).value = v[i].toFixed(2); }); };
    const calc = () => {
      const qv = ['q0', 'q1', 'q2'].map((id) => +q(ctl, '#' + id).value), sa = +q(ctl, '#sa').value;
      const chs = CH.map((h) => ({ name: h.name, v: h.name === 'A' ? h.v.map((x) => x * sa) : h.v }));
      return { qv, rows: chs.map((h) => ({ name: h.name, v: h.v, cos: cosine(qv, h.v), dot: qv.reduce((a, x, i) => a + x * h.v[i], 0), dist: Math.hypot(...qv.map((x, i) => x - h.v[i])) })) };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); const r = calc();
      const L = 70, R = W - 64, cols = [c.blue, c.green, c.purple];
      const panels = [['cosine similarity (−1 to 1; bigger = more similar direction)', (x) => x.cos, 1, true], ['dot product (depends on length)', (x) => x.dot, Math.max(1.5, ...r.rows.map((x) => x.dot)) * 1.1, false], ['Euclidean distance (smaller = closer)', (x) => x.dist, Math.max(1.5, ...r.rows.map((x) => x.dist)) * 1.1, false]];
      const ph = (H - 20) / 3;
      panels.forEach((pn, pi) => {
        const y0 = 10 + pi * ph; txt(g, pn[0], 12, y0 + 12, c.muted, '11px sans-serif');
        const vals = r.rows.map(pn[1]); const order = vals.map((v, i) => i).sort((a, b) => (pi === 2 ? vals[a] - vals[b] : vals[b] - vals[a]));
        r.rows.forEach((h, i) => {
          const y = y0 + 20 + i * ((ph - 28) / 3), bh = (ph - 28) / 3 - 4, w = (R - L) * (Math.max(0, vals[i]) / pn[2]);
          txt(g, 'chunk ' + h.name, L - 8, y + bh - 3, c.fg, '12px sans-serif', 'right');
          g.fillStyle = cols[i]; g.globalAlpha = order[0] === i ? 1 : 0.55; g.fillRect(L, y, Math.max(1, w), bh); g.globalAlpha = 1;
          txt(g, vals[i].toFixed(3) + (order[0] === i ? '  ← top' : ''), L + w + 6, y + bh - 3, c.fg, '11px sans-serif');
        });
      });
      const f = (v) => '(' + v.map((x) => (+x.toFixed(2)).toString()).join(', ') + ')';
      q(ctl, '#out').innerHTML = `query q = ${f(r.qv)}\n` + r.rows.map((h) => `chunk ${h.name} ${f(h.v)}  cos ${h.cos.toFixed(4)}  dot ${h.dot.toFixed(3)}  dist ${h.dist.toFixed(3)}`).join('\n');
      q(ctl, '#cap').textContent = 'Stretch chunk A: its dot product changes but its cosine does not, because cosine divides out the length. Real embedding models are trained so that nearby directions mean similar text; these three vectors were written by hand.';
      showVals(ctl, { q0: (v) => v.toFixed(2), q1: (v) => v.toFixed(2), q2: (v) => v.toFixed(2), sa: (v) => v.toFixed(1) + '×', spd: (v) => (v === 0 ? 'off' : v.toFixed(2)) });
    };
    wire(ctl, st, () => { u = cl(u + 0.25 * dir, 0, 1); if (u === 0 || u === 1) dir = -dir; setQ(QA.map((x, i) => x + (QB[i] - x) * u)); dirty = true; }, () => { u = 0; dir = 1; setQ(QA); q(ctl, '#sa').value = 1; dirty = true; }, draw);
    ['q0', 'q1', 'q2', 'sa'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      const sp = +q(ctl, '#spd').value;
      if (st.playing && sp > 0) {
        u += dir * dt * sp; if (u >= 1) { u = 1; dir = -1; } if (u <= 0) { u = 0; dir = 1; }
        const e = ease(u); setQ(QA.map((x, i) => x + (QB[i] - x) * e)); dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     3b. Chunk size and recall on a toy corpus
     =================================================================== */
  AI.labs['gen-ap-chunks'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Chunk size changes what retrieval can find', 'illustrative: 24 toy sentences in 8 topics of 3, 8-number embeddings, 40 queries; chunk vector = mean of its sentence vectors',
      `<canvas style="height:520px"></canvas>`,
      `${sl('cs', 'Chunk size (sentences per chunk)', 1, 12, 1, 3)}${sl('ov', 'Overlap (sentences shared with the next chunk)', 0, 11, 1, 0)}
       ${sl('kk', 'Chunks retrieved, k', 1, 3, 1, 1)}${sl('sq', 'Query noise (how vague the question is)', 0.2, 1.6, 0.1, 0.8)}
       ${sl('spd', 'Speed', 0.25, 3, 0.25, 1)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const C = retrBuild(7);
    let qi = 0, t = 0, curve = [], dirty = true;
    const P = () => ({ c: +q(ctl, '#cs').value, ov: +q(ctl, '#ov').value, k: +q(ctl, '#kk').value, sq: +q(ctl, '#sq').value });
    const recompute = () => { const p = P(); curve = []; for (let c = 1; c <= 12; c++) curve.push(retrEval(C, c, Math.min(p.ov, c - 1), p.k, p.sq)); dirty = true; };
    const reset = () => { qi = 0; t = 0; recompute(); };
    const draw = () => {
      const cc = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); const p = P(), cols = [cc.blue, cc.green, cc.purple, cc.orange, cc.yellow, cc.red, cc.pink, cc.teal];
      const qq = C.qs[qi], qv = qvec(qq, p.sq), ov = Math.min(p.ov, p.c - 1), ch = retrChunks(C, p.c, ov), top = retrTop(ch, qv, p.k), need = retrNeed(qq);
      const L = 14, R = W - 14, cw = (R - L) / C.N;
      txt(g, `query ${qi + 1} of ${C.qs.length} (topic ${qq.t + 1}): needs the 2 outlined sentences`, L, 16, cc.muted, '11px sans-serif');
      const sy = 28, sh = 26;
      for (let i = 0; i < C.N; i++) { g.fillStyle = cols[C.sent[i].t]; g.globalAlpha = 0.8; g.fillRect(L + i * cw + 1, sy, cw - 2, sh); g.globalAlpha = 1; if (need.includes(i)) { g.strokeStyle = cc.fg; g.lineWidth = 2; g.strokeRect(L + i * cw + 1, sy, cw - 2, sh); g.lineWidth = 1; } }
      txt(g, 'sentences, coloured by topic', L, sy + sh + 18, cc.muted, '11px sans-serif');
      // chunk bars with cosine
      const grow = ease(Math.min(1, t / 1.2)), by = sy + sh + 36, bb = by + 150, hmax = 130;
      txt(g, 'cosine between the query and each chunk (green = retrieved)', L, by + 2, cc.muted, '11px sans-serif');
      const stride = Math.max(1, p.c - ov); const got = new Set();
      const sims = ch.map((h) => cosine(h.v, qv));
      ch.forEach((h, i) => {
        const x0 = L + h.idx[0] * cw, x1 = L + (h.idx[h.idx.length - 1] + 1) * cw, mid = (x0 + x1) / 2, w = Math.max(5, Math.min(stride * cw * 0.7, 40)), isTop = top.some((x) => x.i === i);
        const hh = Math.max(0, sims[i]) * hmax * grow;
        g.fillStyle = isTop ? cc.green : cc.muted; g.globalAlpha = isTop ? 0.95 : 0.45; g.fillRect(mid - w / 2, bb - hh, w, hh); g.globalAlpha = 1;
        if (isTop && grow > 0.95) { g.strokeStyle = cc.green; g.lineWidth = 3; g.beginPath(); g.moveTo(x0 + 1, sy + sh + 3); g.lineTo(x1 - 1, sy + sh + 3); g.stroke(); g.lineWidth = 1; h.idx.forEach((j) => got.add(j)); }
      });
      g.strokeStyle = cc.border; g.beginPath(); g.moveTo(L, bb); g.lineTo(R, bb); g.stroke();
      const hit = need.filter((n) => got.has(n)).length;
      txt(g, grow > 0.95 ? `retrieved ${got.size} sentences, found ${hit} of the 2 needed` : 'scoring chunks…', L, bb + 16, hit === 2 ? cc.green : cc.orange, 'bold 12px sans-serif');
      // recall curve
      const py0 = bb + 46, py1 = H - 34, px0 = 52, px1 = R - 44;
      txt(g, 'recall over all 40 queries versus chunk size', L, py0 - 8, cc.muted, '11px sans-serif');
      const X = (c) => px0 + ((c - 1) / 11) * (px1 - px0), Y = (v) => py1 - v * (py1 - py0);
      g.strokeStyle = cc.border; g.beginPath(); g.moveTo(px0, py0); g.lineTo(px0, py1); g.lineTo(px1, py1); g.stroke();
      [0, 0.5, 1].forEach((v) => txt(g, v.toFixed(1), px0 - 6, Y(v) + 4, cc.muted, '10px sans-serif', 'right'));
      for (let c = 1; c <= 12; c++) txt(g, c, X(c), py1 + 13, cc.muted, '10px sans-serif', 'center');
      const smax = Math.max(...curve.map((x) => x.sent));
      curve.forEach((x, i) => { const hh = (x.sent / smax) * (py1 - py0) * 0.5; g.fillStyle = cc.muted; g.globalAlpha = 0.25; g.fillRect(X(i + 1) - 6, py1 - hh, 12, hh); g.globalAlpha = 1; });
      g.strokeStyle = cc.blue; g.lineWidth = 2; g.beginPath(); curve.forEach((x, i) => (i ? g.lineTo(X(i + 1), Y(x.recall)) : g.moveTo(X(1), Y(x.recall)))); g.stroke(); g.lineWidth = 1;
      g.fillStyle = cc.yellow; g.beginPath(); g.arc(X(p.c), Y(curve[p.c - 1].recall), 5, 0, 7); g.fill();
      txt(g, 'grey bars: sentences sent to the model (tallest = ' + smax.toFixed(0) + ')', px1, py1 + 27, cc.muted, '10px sans-serif', 'right');
      const cur = curve[p.c - 1];
      q(ctl, '#out').innerHTML = `chunk size ${p.c}, overlap ${ov}, k = ${p.k}\nchunks in the index: ${ch.length}\nrecall of needed sentences: ${cur.recall.toFixed(3)}\nsentences sent per query (average): ${cur.sent.toFixed(1)}\nrecall per sentence sent: ${(cur.recall / cur.sent).toFixed(3)}`;
      q(ctl, '#cap').textContent = p.c < 2 ? 'Chunks of one sentence are precise but a single chunk can never hold both needed sentences.' : p.c === 3 ? 'Size 3 matches how this toy document is built (topics of three sentences), so chunks line up with topics. Real documents have no such tidy structure.' : p.c > 4 ? 'Large chunks mix topics, so the chunk vector is an average that points between topics, and more irrelevant sentences are sent along.' : 'Chunk boundaries that cut across a topic split the needed pair or blur two topics together. Try an overlap.';
      showVals(ctl, { cs: (v) => v, ov: (v) => Math.min(v, p.c - 1) + (v > p.c - 1 ? ' (limited)' : ''), kk: (v) => v, sq: (v) => v.toFixed(1), spd: (v) => v + '×' });
    };
    wire(ctl, st, () => { qi = (qi + 1) % C.qs.length; t = 0; }, reset, draw);
    ['cs', 'ov', 'kk', 'sq'].forEach((id) => { q(ctl, '#' + id).oninput = () => { recompute(); t = 0; }; });
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; recompute(); draw();
    AI.loop(root, (dt) => {
      if (st.playing) { const prev = t; t += dt * +q(ctl, '#spd').value; if (t < 1.2 || prev < 1.2) dirty = true; if (t > 5) { qi = (qi + 1) % C.qs.length; t = 0; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     4. Retrieval metrics versus answer quality (RAG)
     =================================================================== */
  AI.labs['gen-ap-rag'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Retrieve, rerank, then generate: two separate scores', 'illustrative: a fixed toy ranking of 10 chunks; the answer-quality curve is an assumed rule, not a measurement',
      `<canvas style="height:480px"></canvas>`,
      `<label>Ranking <select id="rk"><option value="0">first-stage retriever</option><option value="1">after a reranker</option></select></label>
       ${sl('kk', 'Chunks passed to the generator, k', 1, 10, 1, 4)}
       <div class="row" id="rel" style="gap:4px"></div>
       <p class="caption" style="margin:0">Tick the chunks that really contain the answer (what-if).</p>
       ${sl('pf', 'Assumed chance of a good answer with all evidence present', 0.5, 1, 0.05, 0.9)}${sl('pn', 'Assumed chance with no evidence at all', 0, 0.5, 0.05, 0.1)}${sl('dl', 'Assumed loss per irrelevant chunk in the context', 0, 0.1, 0.01, 0.03)}
       ${sl('spd', 'Speed', 0.25, 3, 0.25, 1)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:9em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const relBox = q(ctl, '#rel');
    relBox.innerHTML = RAG_DOCS.map((d, i) => `<label style="flex-direction:row;gap:3px;align-items:center"><input type="checkbox" data-d="${i + 1}"${[2, 5, 9].includes(i + 1) ? ' checked' : ''}> D${i + 1}</label>`).join('');
    const rel = () => new Set([...relBox.querySelectorAll('input')].filter((x) => x.checked).map((x) => +x.dataset.d));
    let pos = RETR.map((d, i) => i), kdisp = 4, sweep = false, dirty = true, hold = 0;
    const rank = () => (q(ctl, '#rk').value === '1' ? RERANK : RETR);
    const reset = () => { kdisp = +q(ctl, '#kk').value; dirty = true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w; let H = o.h;
      const narrow = W < 540, wantH = narrow ? 740 : 480;
      if (Math.abs(H - wantH) > 2) { cv.style.height = wantH + 'px'; return; }
      g.clearRect(0, 0, W, H);
      const rk = rank(), R0 = rel(), k = +q(ctl, '#kk').value, m = ragMetrics(rk, R0, k);
      const pf = +q(ctl, '#pf').value, pn = +q(ctl, '#pn').value, dl = +q(ctl, '#dl').value;
      const split = narrow ? W - 4 : Math.round(Math.min(W * 0.5, 280)), rowH = 30, y0 = 50;
      txt(g, 'query: "Return window, and who pays shipping?"', 12, 16, c.muted, '11px sans-serif');
      txt(g, 'ranked chunks', 12, y0 - 8, c.muted, '11px sans-serif');
      for (let r = 0; r < 10; r++) {
        const d = rk[r], y = y0 + pos[d - 1] * rowH, inK = r < k, isRel = R0.has(d);
        g.globalAlpha = inK ? 1 : 0.45; g.fillStyle = isRel ? c.green : c.bg3; rrect(g, 12, y, split - 20, rowH - 4, 6); g.fill(); g.globalAlpha = 1;
        g.strokeStyle = inK ? c.yellow : c.border; g.lineWidth = inK ? 2 : 1; rrect(g, 12, y, split - 20, rowH - 4, 6); g.stroke(); g.lineWidth = 1;
        const col = isRel && inK ? '#0b0f14' : c.fg;
        txt(g, '#' + (r + 1), 18, y + 17, col, 'bold 11px sans-serif'); txt(g, 'D' + d + ' ' + RAG_DOCS[d - 1], 44, y + 17, col, '11px sans-serif');
      }
      // pos eases toward rank index (updated in the loop)
      const ky = y0 + k * rowH - 2; g.strokeStyle = c.yellow; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(8, ky); g.lineTo(split, ky); g.stroke(); g.setLineDash([]);
      txt(g, 'context cut at k = ' + k, 12, y0 + 10 * rowH + 14, c.yellow, '11px sans-serif');
      // curves
      const px0 = narrow ? 40 : split + 44, px1 = W - 16, py0 = narrow ? y0 + 10 * rowH + 64 : 66, py1 = py0 + 174;
      if (px1 - px0 > 80) {
        txt(g, 'score versus k', px0, py0 - 22, c.muted, '11px sans-serif');
        const X = (kk) => px0 + ((kk - 1) / 9) * (px1 - px0), Y = (v) => py1 - v * (py1 - py0);
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(px0, py0); g.lineTo(px0, py1); g.lineTo(px1, py1); g.stroke();
        [0, 0.5, 1].forEach((v) => txt(g, v.toFixed(1), px0 - 6, Y(v) + 4, c.muted, '10px sans-serif', 'right'));
        for (let kk = 1; kk <= 10; kk++) txt(g, kk, X(kk), py1 + 13, c.muted, '10px sans-serif', 'center');
        const rc = [], an = []; for (let kk = 1; kk <= 10; kk++) { const mm = ragMetrics(rk, R0, kk); rc.push(mm.recall); an.push(ragAnswer(mm.recall, kk, mm.found, pf, pn, dl)); }
        const line = (arr, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); arr.forEach((v, i) => (i ? g.lineTo(X(i + 1), Y(v)) : g.moveTo(X(1), Y(v)))); g.stroke(); g.lineWidth = 1; g.fillStyle = col; g.beginPath(); g.arc(X(k), Y(arr[k - 1]), 5, 0, 7); g.fill(); };
        line(rc, c.blue); line(an, c.orange);
        txt(g, 'recall at k (retrieval)', px0 + 4, py1 + 32, c.blue, '11px sans-serif'); txt(g, 'assumed answer quality', px0 + 4, py1 + 48, c.orange, '11px sans-serif');
        const t2 = py1 + 74;
        txt(g, 'precision at k: ' + m.precision.toFixed(3), px0, t2, c.fg, '12px sans-serif'); txt(g, 'first relevant rank: ' + (m.first || 'none'), px0, t2 + 16, c.fg, '12px sans-serif');
      }
      const ans = ragAnswer(m.recall, k, m.found, pf, pn, dl);
      q(ctl, '#out').innerHTML = `relevant chunks: ${[...R0].sort((a, b) => a - b).map((d) => 'D' + d).join(', ') || 'none'} (${R0.size})\nin the top ${k}: ${m.found} of ${R0.size}\nrecall@${k} = ${m.found}/${R0.size} = ${m.recall.toFixed(4)}\nprecision@${k} = ${m.found}/${k} = ${m.precision.toFixed(4)}\nMRR = ${m.mrr.toFixed(4)}\nassumed answer quality = ${ans.toFixed(3)}`;
      q(ctl, '#cap').textContent = 'Recall@k only counts how many needed chunks are in the context. It never goes down as k grows. The orange answer curve follows an assumed rule (more evidence helps, extra irrelevant chunks cost a little) and is not a measurement of any model.';
      showVals(ctl, { kk: (v) => v, pf: (v) => v.toFixed(2), pn: (v) => v.toFixed(2), dl: (v) => v.toFixed(2), spd: (v) => v + '×' });
    };
    wire(ctl, st, () => { const k = +q(ctl, '#kk').value; q(ctl, '#kk').value = k % 10 + 1; dirty = true; }, () => { sweep = false; reset(); }, draw);
    ['rk', 'kk', 'pf', 'pn', 'dl'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    relBox.oninput = () => { dirty = true; };
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      // ease row positions toward the current ranking (smooth reorder when the reranker is switched on)
      const rk = rank(); let mv = false;
      rk.forEach((d, r) => { const dd = r - pos[d - 1]; if (Math.abs(dd) > 0.004) { pos[d - 1] += dd * (1 - Math.exp(-dt * 4 * +q(ctl, '#spd').value)); mv = true; } else pos[d - 1] = r; });
      if (st.playing) { hold += dt * +q(ctl, '#spd').value; if (hold > 1.8) { hold = 0; const k = +q(ctl, '#kk').value; q(ctl, '#kk').value = k % 10 + 1; dirty = true; } }
      if (mv || dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     5a. A ReAct-style trace with deterministic toy tools
     =================================================================== */
  AI.labs['gen-ap-react'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A plan, act, observe loop with two real toy tools', 'illustrative: the "agent" is a fixed script (no language model); the two tools, a price lookup and a calculator, really run',
      `<div id="trace" style="font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.6;min-height:330px;padding:6px 4px"></div>`,
      `${sl('nn', 'Notebooks wanted', 1, 9, 1, 3)}${sl('mm', 'Pens wanted', 1, 9, 1, 2)}
       <label>Inject a tool error at step <select id="er"><option value="0">no error</option><option value="1">1 (notebook price)</option><option value="2">2 (pen price)</option><option value="3">3 (calculator)</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="rt"> the agent retries a failed call once</label>
       ${sl('cps', 'Typing speed, characters per second', 5, 120, 5, 30)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8em"></div><p class="caption" id="cap"></p>`);
    const tr = q(view, '#trace'), st = { playing: true };
    let T, li, ci, dirty = true;
    const colOf = { Thought: 'var(--muted)', Action: 'var(--blue)', Observation: 'var(--green)', Answer: 'var(--orange)' };
    const build = () => {
      T = reactTrace(+q(ctl, '#nn').value, +q(ctl, '#mm').value, +q(ctl, '#er').value, q(ctl, '#rt').checked);
      li = 0; ci = 0; dirty = true;
    };
    const paint = () => {
      let h = `<div style="color:var(--muted);margin-bottom:6px">Task: "What do ${q(ctl, '#nn').value} notebooks and ${q(ctl, '#mm').value} pens cost?"  Tools: lookup(item), calc(expression)</div>`;
      for (let i = 0; i <= Math.min(li, T.lines.length - 1); i++) {
        const L = T.lines[i], full = i < li, shown = full ? L.text : L.text.slice(0, Math.floor(ci));
        const err = /^ERROR/.test(L.text);
        h += `<div style="margin:2px 0"><b style="color:${err && L.kind === 'Observation' ? 'var(--red)' : colOf[L.kind]}">${L.kind}${L.kind === 'Answer' ? '' : ' ' + L.call}:</b> <span style="color:${err ? 'var(--red)' : 'var(--fg)'}">${shown.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</span></div>`;
      }
      tr.innerHTML = h;
      const done = li >= T.lines.length;
      let calls = 0, inTok = 0, ctx = 120, outTok = 0;
      const upto = done ? T.lines.length : li;
      for (let i = 0; i < upto; i++) { const L = T.lines[i]; if (L.kind === 'Thought' || L.kind === 'Answer') { inTok += ctx; calls++; } ctx += L.tok; if (L.kind !== 'Observation') outTok += L.tok; }
      q(ctl, '#out').innerHTML = `model calls so far: ${calls} of ${T.calls}
input tokens read so far: ${inTok}
output tokens written: ${outTok}
full run: ${T.calls} calls, ${T.inTok} input tokens, ${T.outTok} output tokens
tool result: ${T.result === null ? 'none' : T.result}`;
      q(ctl, '#cap').textContent = done ? (T.result === null ? 'The failed call was not retried, so the script went on with a missing price and the calculator could not evaluate the expression. Error handling is a design decision, and nothing here guarantees it.' : 'Finished. Every step re-reads everything before it, so the input tokens grow faster than the number of steps.') : 'Each model call reads the whole prompt and the trace so far, then writes one thought and one action. The tool, not the model, produces the observation.';
      showVals(ctl, { nn: (v) => v, mm: (v) => v, cps: (v) => v });
    };
    const step = () => { if (li < T.lines.length) { li++; ci = 0; } dirty = true; };
    wire(ctl, st, step, () => { build(); }, paint);
    ['nn', 'mm', 'er', 'rt'].forEach((id) => { q(ctl, '#' + id).oninput = () => build(); });
    q(ctl, '#rt').onchange = build; q(ctl, '#er').onchange = build;
    q(ctl, '#cps').oninput = () => { dirty = true; };
    build(); paint();
    AI.loop(root, (dt) => {
      if (st.playing && li < T.lines.length) {
        const L = T.lines[li]; ci += dt * +q(ctl, '#cps').value; dirty = true;
        if (ci >= L.text.length + 12) { li++; ci = 0; }
      }
      if (dirty) { paint(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     5b. Error and cost compound over steps
     =================================================================== */
  AI.labs['gen-ap-compound'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Long loops: success shrinks as p^n, cost grows faster than n', 'illustrative: 200 seeded runs; each step succeeds with a probability you choose; token costs follow the printed formula',
      `<canvas style="height:480px"></canvas>`,
      `${sl('p', 'Chance a single step succeeds, p', 0.5, 0.999, 0.001, 0.95)}${sl('n', 'Number of steps, n', 1, 25, 1, 10)}
       ${sl('r', 'Attempts allowed per step (retries)', 1, 3, 1, 1)}${sl('p0', 'Prompt tokens read at every call', 200, 2000, 100, 500)}${sl('gs', 'Tokens added to the context per step', 50, 500, 50, 150)}
       ${sl('spd', 'Steps per second', 0.25, 4, 0.25, 0.75)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:10em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const NA = 200, MAXN = 25, U = []; { const r = AI.rng(31); for (let a = 0; a < NA; a++) { U.push([]); for (let s = 0; s < MAXN; s++) U[a].push(r()); } }
    let front = 0, dirty = true;
    const P = () => ({ p: +q(ctl, '#p').value, n: +q(ctl, '#n').value, r: +q(ctl, '#r').value, p0: +q(ctl, '#p0').value, gs: +q(ctl, '#gs').value });
    const alive = (a, k, pe) => { for (let s = 0; s < k; s++) if (U[a][s] >= pe) return false; return true; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); const p = P(), pe = effP(p.p, p.r), k0 = Math.floor(front), fr = front - k0;
      // dots
      txt(g, `200 runs, step ${Math.min(p.n, front).toFixed(1)}: green correct, grey failed`, 12, 16, c.muted, '11px sans-serif');
      const cols = 40, dx = Math.min(10, (W - 24) / cols), ox = 12, oy = 28;
      for (let a = 0; a < NA; a++) {
        const now = alive(a, Math.min(p.n, k0), pe), nxt = alive(a, Math.min(p.n, k0 + 1), pe);
        const f = now && !nxt ? 1 - ease(Math.min(1, fr)) : now ? 1 : 0;
        g.fillStyle = f > 0 ? c.green : c.muted; g.globalAlpha = f > 0 ? 0.35 + 0.65 * f : 0.3;
        g.beginPath(); g.arc(ox + (a % cols) * dx + dx / 2, oy + Math.floor(a / cols) * dx + dx / 2, dx * 0.38, 0, 7); g.fill(); g.globalAlpha = 1;
      }
      // survival chart
      const t0 = 100, t1 = 100 + 160, L = 40, R = W - 14, X = (k) => L + ((k + 0.5) / (p.n + 1)) * (R - L), Y = (v) => t1 - v * (t1 - t0);
      txt(g, 'still correct after k steps (bars: runs; line: p' + (p.r > 1 ? '′' : '') + '^k)', 12, t0 - 8, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, t0); g.lineTo(L, t1); g.lineTo(R, t1); g.stroke();
      [0, 0.5, 1].forEach((v) => txt(g, v.toFixed(1), L - 5, Y(v) + 4, c.muted, '10px sans-serif', 'right'));
      const bw = Math.max(2, (R - L) / p.n - 3);
      for (let k = 1; k <= p.n; k++) {
        if (k > Math.ceil(front)) break; let cnt = 0, cnt2 = 0; for (let a = 0; a < NA; a++) { if (alive(a, k, pe)) cnt++; if (alive(a, k - 1, pe)) cnt2++; }
        const prog = k <= k0 ? 1 : ease(fr); const v = (cnt2 + (cnt - cnt2) * prog) / NA;
        g.fillStyle = c.green; g.globalAlpha = 0.75; g.fillRect(X(k) - bw / 2, Y(v), bw, t1 - Y(v)); g.globalAlpha = 1;
        if (p.n <= 12) txt(g, k, X(k), t1 + 12, c.muted, '10px sans-serif', 'center');
      }
      g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); for (let k = 0; k <= p.n; k++) (k ? g.lineTo(X(k), Y(Math.pow(pe, k))) : g.moveTo(X(0), Y(1))); g.stroke(); g.lineWidth = 1;
      txt(g, `theory after ${p.n} steps: ${pe.toFixed(4)}^${p.n} = ${Math.pow(pe, p.n).toFixed(4)}`, R, t0 + 12, c.orange, 'bold 12px sans-serif', 'right');
      // cost chart
      const u0 = t1 + 54, u1 = H - 26, tm = tokAt(p.n, p.p0, p.gs), Yc = (v) => u1 - v / tm * (u1 - u0);
      txt(g, 'input tokens after k calls: P·k + s·k(k−1)/2 (dashed: P·k)', 12, u0 - 8, c.muted, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, u0); g.lineTo(L, u1); g.lineTo(R, u1); g.stroke();
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); for (let k = 0; k <= p.n; k++) (k ? g.lineTo(X(k), Yc(tokAt(k, p.p0, p.gs))) : g.moveTo(X(0), Yc(0))); g.stroke(); g.lineWidth = 1;
      g.strokeStyle = c.muted; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(0), Yc(0)); g.lineTo(X(p.n), Yc(p.p0 * p.n)); g.stroke(); g.setLineDash([]);
      const kc = Math.min(p.n, Math.max(0, front)); g.fillStyle = c.yellow; g.beginPath(); g.arc(X(kc), Yc(tokAt(kc, p.p0, p.gs)), 5, 0, 7); g.fill();
      txt(g, tm.toFixed(0) + ' tokens', R, u0 + 12, c.purple, 'bold 12px sans-serif', 'right');
      const calls = (1 - Math.pow(1 - p.p, p.r)) / p.p, total = tm * calls, ps = Math.pow(pe, p.n), nHalf = Math.log(0.5) / Math.log(pe);
      q(ctl, '#out').innerHTML = `per-step success with ${p.r} attempt${p.r > 1 ? 's' : ''}: p′ = 1 − (1 − ${p.p})^${p.r} = ${pe.toFixed(4)}\nrun success after n = ${p.n}: p′^n = ${ps.toFixed(4)}\nsteps until success falls below 50%: ${isFinite(nHalf) ? nHalf.toFixed(1) : '∞'}\ntokens for n = ${p.n} calls: ${tm.toFixed(0)}  (grows like n²; the sum is not n·P)\nexpected calls per step with retries: ${calls.toFixed(3)} → about ${total.toFixed(0)} tokens\ntokens per successful run ≈ ${(total / ps).toFixed(0)}`;
      q(ctl, '#cap').textContent = 'Each step looks fine on its own, yet the product of many chances decays geometrically. Retries buy back some success, and cost more calls. This assumes steps fail independently, which real loops need not do.';
      showVals(ctl, { p: (v) => v.toFixed(3), n: (v) => v, r: (v) => v, p0: (v) => v, gs: (v) => v, spd: (v) => v });
    };
    wire(ctl, st, () => { front = Math.min(P().n, Math.floor(front) + 1); }, () => { front = 0; }, draw);
    ['p', 'n', 'r', 'p0', 'gs'].forEach((id) => { q(ctl, '#' + id).oninput = () => { front = Math.min(front, P().n); dirty = true; }; });
    q(ctl, '#spd').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing && front < P().n) { front = Math.min(P().n, front + dt * +q(ctl, '#spd').value); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* ===================================================================
     6. Context window, truncation, summaries and memory
     =================================================================== */
  AI.labs['gen-ap-context'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'A conversation outgrowing its context window', 'illustrative: 24 turns of seeded length; the recall numbers use assumed rules printed below, not measurements',
      `<canvas style="height:440px"></canvas>`,
      `<label>Strategy when old turns no longer fit <select id="sg"><option value="truncate">truncate: drop the oldest turns</option><option value="summary">summarise: replace them by a short summary</option><option value="memory">memory store: retrieve the best 2 by cosine</option></select></label>
       ${sl('w', 'Context window, tokens', 500, 4000, 100, 1500)}${sl('f', 'The key fact was said in turn', 1, 20, 1, 3)}
       ${sl('sm', 'Summary size, tokens', 50, 400, 25, 150)}${sl('vg', 'Question vagueness (for the memory search)', 0, 2, 0.1, 0.5)}
       ${sl('rate', 'Turns per second', 0.2, 3, 0.1, 0.8)}${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:9em"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    const C = ctxBuild(), TOT = C.len.reduce((a, b) => a + b, 0);
    let n = 1, acc = 0, dirty = true;
    const P = () => ({ sg: q(ctl, '#sg').value, W: +q(ctl, '#w').value, F: +q(ctl, '#f').value - 1, S: +q(ctl, '#sm').value, vg: +q(ctl, '#vg').value });
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H); const p = P(), S = ctxState(C, n, p.W, p.sg, p.S, p.F, p.vg);
      const L = 12, R = W - 12, sc = (R - L) / (Math.max(TOT, p.W) + 300);
      txt(g, `conversation: turn ${n} of ${CTX_N} (width = tokens, star = key fact)`, L, 16, c.muted, '11px sans-serif');
      let x = L; const y1 = 26, bh = 34;
      for (let i = 0; i < CTX_N; i++) {
        const w = C.len[i] * sc; if (i >= n) { g.strokeStyle = c.border; g.strokeRect(x + 0.5, y1, w - 1, bh); x += w; continue; }
        const inWin = i >= S.s;
        g.fillStyle = inWin ? c.blue : c.bg3; g.globalAlpha = inWin ? 0.85 : 1; g.fillRect(x + 0.5, y1, w - 1, bh); g.globalAlpha = 1;
        if (S.retrieved.includes(i)) { g.strokeStyle = c.purple; g.lineWidth = 3; g.strokeRect(x + 1.5, y1 + 1, w - 3, bh - 2); g.lineWidth = 1; }
        if (i === p.F) txt(g, '★', x + w / 2, y1 + bh / 2 + 6, c.yellow, 'bold 18px sans-serif', 'center');
        x += w;
      }
      txt(g, 'grey: outside the window, blue: inside, purple frame: recalled', L, y1 + bh + 14, c.muted, '11px sans-serif');
      // what the model sees
      const y2 = y1 + bh + 40; txt(g, `what the model reads (budget ${p.W}, ${CTX_RES} for the question)`, L, y2, c.muted, '11px sans-serif');
      x = L; const y3 = y2 + 8;
      const blk = (w, col, label, star, frame) => { g.fillStyle = col; g.fillRect(x + 0.5, y3, Math.max(1, w - 1), bh); if (frame) { g.strokeStyle = c.purple; g.lineWidth = 3; g.strokeRect(x + 1.5, y3 + 1, w - 3, bh - 2); g.lineWidth = 1; } if (star) txt(g, '★', x + w / 2, y3 + bh / 2 + 6, c.yellow, 'bold 18px sans-serif', 'center'); if (label && w > 40) txt(g, label, x + w / 2, y3 + bh + 12, c.muted, '10px sans-serif', 'center'); x += w; };
      if (p.sg === 'summary' && S.s > 0) blk(p.S * sc, c.purple, 'summary', false, false);
      if (p.sg === 'memory') S.retrieved.forEach((i) => blk(C.len[i] * sc, c.purple, 'recalled', i === p.F, false));
      S.win.forEach((i) => blk(C.len[i] * sc, c.blue, '', i === p.F, false));
      blk(CTX_RES * sc, c.orange, 'question', false, false);
      // position curve
      const cy0 = y3 + bh + 50, cy1 = H - 34, px0 = 50, px1 = R - 8;
      txt(g, 'ASSUMED recall by position: 0.55 + 0.45(2x−1)² (illustration only)', L, cy0 - 8, c.muted, '11px sans-serif');
      const X = (v) => px0 + v * (px1 - px0), Yv = (v) => cy1 - (v - 0.4) / 0.6 * (cy1 - cy0);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(px0, cy0); g.lineTo(px0, cy1); g.lineTo(px1, cy1); g.stroke();
      [0.5, 0.75, 1].forEach((v) => txt(g, v.toFixed(2), px0 - 5, Yv(v) + 4, c.muted, '10px sans-serif', 'right'));
      txt(g, 'start of context', px0, cy1 + 14, c.muted, '10px sans-serif'); txt(g, 'end', px1, cy1 + 14, c.muted, '10px sans-serif', 'right');
      g.strokeStyle = c.blue; g.lineWidth = 2; g.beginPath(); for (let i = 0; i <= 40; i++) (i ? g.lineTo(X(i / 40), Yv(posRecall(i / 40))) : g.moveTo(X(0), Yv(posRecall(0)))); g.stroke(); g.lineWidth = 1;
      if (S.pos !== null) { g.fillStyle = c.yellow; g.beginPath(); g.arc(X(S.pos), Yv(posRecall(S.pos)), 6, 0, 7); g.fill(); }
      q(ctl, '#out').innerHTML = `turns in the window: ${S.win.length}, tokens ${S.winTok}\nturns left out: ${S.evicted.length} (${S.evTok} tokens)\nkey fact (turn ${p.F + 1}): ${S.status}\nassumed chance the model uses it: ${S.recall.toFixed(3)}` + (p.sg === 'summary' && S.evicted.length ? `\nsummary keeps min(1, ${p.S}/${S.evTok}) = ${Math.min(1, p.S / S.evTok).toFixed(3)} of the left-out material (assumed)` : '') + (p.sg === 'memory' && S.retrieved.length ? `\nmemory returned turns ${S.retrieved.map((i) => i + 1).join(', ')}` : '');
      q(ctl, '#cap').textContent = S.status === 'truncated away' ? 'The fact scrolled out of the window and nothing keeps it. The model cannot use what it is not shown.' : S.status === 'in window' ? 'The fact is still inside the window. Whether a real model uses it can depend on where it sits, and reports on that vary by model and task; the curve is only a drawn assumption.' : S.status === 'summarised' ? 'A summary has a fixed size, so the more material it replaces, the smaller the share of detail it can keep.' : S.status === 'not yet said' ? 'The key fact has not been said yet; wait for the conversation to reach that turn.' : 'The memory search compares the question vector with each stored turn using cosine, as in retrieval. A vague question can miss the right turn.';
      showVals(ctl, { w: (v) => v, f: (v) => v, sm: (v) => v, vg: (v) => v.toFixed(1), rate: (v) => v.toFixed(1) });
    };
    wire(ctl, st, () => { n = Math.min(CTX_N, n + 1); }, () => { n = 1; acc = 0; }, draw);
    ['sg', 'w', 'f', 'sm', 'vg'].forEach((id) => { q(ctl, '#' + id).oninput = () => { dirty = true; }; });
    q(ctl, '#sg').onchange = () => { dirty = true; };
    q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < CTX_N) { acc += dt * +q(ctl, '#rate').value; if (acc >= 1) { acc = 0; n++; dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };
})();
