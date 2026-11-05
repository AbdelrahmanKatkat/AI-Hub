/* Generative AI Module 1 labs: language modelling and tokenisation.
   gen-lm-ngram, gen-lm-perplexity, gen-lm-bpe, gen-lm-encode, gen-lm-embed, gen-lm-neural, gen-lm-decode, gen-lm-beam.
   All simulations are illustrative: tiny models, seeded random numbers (AI.rng), dt-based animation with speed / pause / step / reset.
   The pure algorithms in `core` (counts, BPE, softmax, top-k, top-p) are the same code used to compute every number printed in the lessons. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const btns = `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;
  const txt = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '12px sans-serif'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const wire = (ctl, st, onStep, onReset, redraw) => {
    const pb = q(ctl, '#play');
    st.playing = true;
    pb.onclick = () => { st.playing = !st.playing; pb.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    q(ctl, '#step').onclick = () => { st.playing = false; pb.textContent = '▶ Play'; onStep(); redraw(); };
    q(ctl, '#reset').onclick = () => { onReset(); redraw(); };
  };
  const showVals = (ctl, map) => { for (const id in map) { const e = q(ctl, '#' + id + 'v'); if (e) e.textContent = map[id](+q(ctl, '#' + id).value); } };
  const f3 = (v) => (Math.abs(v) < 0.0005 ? '0' : v.toFixed(3));
  const rrect = (g, x, y, w, h, r) => { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };

  /* ---------- pure algorithms (also run in node to compute the numbers quoted in the lessons) ---------- */
  const core = {};
  // n-gram (bigram) counts. Sentences are padded with <s> in front and </s> at the end.
  core.tokens = (s) => s.toLowerCase().split(/\s+/).filter(Boolean);
  core.bigrams = (sents) => {
    const c = {}, ctx = {}, vocab = new Set(['</s>']);
    sents.forEach((s) => {
      const t = ['<s>', ...core.tokens(s), '</s>'];
      t.forEach((w) => { if (w !== '<s>') vocab.add(w); });
      for (let i = 1; i < t.length; i++) {
        c[t[i - 1]] = c[t[i - 1]] || {}; c[t[i - 1]][t[i]] = (c[t[i - 1]][t[i]] || 0) + 1; ctx[t[i - 1]] = (ctx[t[i - 1]] || 0) + 1;
      }
    });
    return { c, ctx, vocab: [...vocab], V: vocab.size };
  };
  core.pBigram = (m, prev, next, k) => { const cn = (m.c[prev] && m.c[prev][next]) || 0, d = (m.ctx[prev] || 0) + k * m.V; return d === 0 ? 0 : (cn + k) / d; };
  core.unigram = (sents) => {
    const c = {}; let n = 0;
    sents.forEach((s) => { [...core.tokens(s), '</s>'].forEach((w) => { c[w] = (c[w] || 0) + 1; n++; }); });
    return { c, n, V: Object.keys(c).length };
  };
  core.pUnigram = (m, w, k) => ((m.c[w] || 0) + k) / (m.n + k * m.V);

  // byte-pair encoding. A word is a list of symbols; '_' marks the end of a word.
  core.bpeWords = (text) => {
    const f = {}; const order = [];
    core.tokens(text).forEach((w) => { if (!(w in f)) { f[w] = 0; order.push(w); } f[w]++; });
    return order.map((w) => ({ w, f: f[w], sym: [...w, '_'] }));
  };
  core.pairCounts = (words) => {
    const pc = {};
    words.forEach((x) => { for (let i = 0; i < x.sym.length - 1; i++) { const k = x.sym[i] + '\u0001' + x.sym[i + 1]; pc[k] = (pc[k] || 0) + x.f; } });
    return Object.keys(pc).map((k) => { const [a, b] = k.split('\u0001'); return { a, b, n: pc[k] }; })
      .sort((p, q) => q.n - p.n || (p.a + p.b < q.a + q.b ? -1 : 1));
  };
  core.applyMerge = (sym, a, b) => { const out = []; for (let i = 0; i < sym.length; i++) { if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]); } return out; };
  core.bpeTrain = (text, nMerges) => {
    const words = core.bpeWords(text), steps = [];
    for (let m = 0; m < nMerges; m++) {
      const pcs = core.pairCounts(words); if (!pcs.length || pcs[0].n < 1) break;
      const best = pcs[0];
      const before = words.map((x) => x.sym.slice());
      words.forEach((x) => { x.sym = core.applyMerge(x.sym, best.a, best.b); });
      steps.push({ pair: [best.a, best.b], n: best.n, table: pcs.slice(0, 6), before, after: words.map((x) => x.sym.slice()) });
    }
    return { words, steps, merges: steps.map((s) => s.pair) };
  };
  core.bpeEncode = (word, merges) => { let sym = [...word, '_']; merges.forEach(([a, b]) => { sym = core.applyMerge(sym, a, b); }); return sym; };

  // softmax and sampling transforms
  core.softmax = (z, T) => { T = T || 1; const m = Math.max(...z.map((v) => v / T)); const e = z.map((v) => Math.exp(v / T - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
  core.topK = (p, k) => { const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a] || a - b); const keep = new Set(idx.slice(0, k)); const q = p.map((v, i) => (keep.has(i) ? v : 0)); const s = q.reduce((a, b) => a + b, 0); return q.map((v) => v / s); };
  core.topP = (p, P) => { const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a] || a - b); const keep = new Set(); let cum = 0; for (const i of idx) { keep.add(i); cum += p[i]; if (cum >= P - 1e-12) break; } const q = p.map((v, i) => (keep.has(i) ? v : 0)); const s = q.reduce((a, b) => a + b, 0); return q.map((v) => v / s); };
  core.xent = (z, y) => { const p = core.softmax(z); return -Math.log(p[y]); };

  const DEFCORP = 'the cat sat on the mat\nthe cat ate the fish\nthe dog sat on the rug';

  /* =====================================================================
     1. N-gram model: counting pairs, counts become probabilities, add-k
     ===================================================================== */
  AI.labs['gen-lm-ngram'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Counting becomes probability', 'illustrative: a bigram model built by counting word pairs in a tiny text you can edit',
      `<canvas style="height:430px"></canvas>`,
      `<label>Training text (one sentence per line; edit it)<textarea id="corp" rows="4" style="background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-family:ui-monospace,Menlo,Consolas,monospace;width:100%;box-sizing:border-box">${DEFCORP}</textarea></label>
       <label>Context word (the word just seen) <select id="ctx"></select></label>
       ${sl('k', 'Add-k smoothing k', 0, 1, 0.05, 0)}
       ${sl('rate', 'Word pairs counted per second', 0.25, 6, 0.25, 1.5)}
       ${btns}
       <div class="row"><button class="btn" id="gen">Generate a sentence</button></div>
       <div class="readout" id="out" style="white-space:normal;min-height:5em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let sents, pairs, m, pos, dirty = true, tick = AI.ticker(), genSeed = 1, genText = '';
    const parse = () => {
      sents = q(ctl, '#corp').value.split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 12);
      m = core.bigrams(sents); pairs = [];
      sents.forEach((s, si) => { const t = ['<s>', ...core.tokens(s), '</s>']; for (let i = 1; i < t.length; i++) pairs.push({ si, i, prev: t[i - 1], next: t[i] }); });
      const sel = q(ctl, '#ctx'), old = sel.value;
      const opts = ['<s>', ...m.vocab.filter((w) => w !== '</s>')];
      sel.innerHTML = opts.map((w) => `<option value="${w.replace(/</g, '&lt;')}">${w.replace(/</g, '&lt;')}</option>`).join('');
      sel.value = opts.includes(old) ? old : (opts.includes('the') ? 'the' : opts[0]);
      pos = 0; genText = ''; dirty = true;
    };
    const partial = (n) => { const c = {}, tot = {}; for (let i = 0; i < n; i++) { const p = pairs[i]; c[p.prev] = c[p.prev] || {}; c[p.prev][p.next] = (c[p.prev][p.next] || 0) + 1; tot[p.prev] = (tot[p.prev] || 0) + 1; } return { c, tot }; };
    const gen = () => {
      const k = +q(ctl, '#k').value, r = AI.rng(genSeed++ * 7919), out = []; let prev = '<s>';
      for (let n = 0; n < 14; n++) {
        const ps = m.vocab.map((w) => core.pBigram(m, prev, w, k)), s = ps.reduce((a, b) => a + b, 0);
        if (s <= 0) { out.push('[stuck: no counts after "' + prev + '"]'); break; }
        let u = r() * s, j = 0; while (j < ps.length - 1 && u >= ps[j]) { u -= ps[j]; j++; }
        if (m.vocab[j] === '</s>') break; out.push(m.vocab[j]); prev = m.vocab[j];
      }
      genText = out.join(' '); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const ctxw = q(ctl, '#ctx').value, k = +q(ctl, '#k').value, { c: cc, tot } = partial(pos);
      // top: corpus with current pair highlighted
      g.font = '13px ui-monospace,Menlo,Consolas,monospace';
      txt(g, 'training text (each adjacent pair is counted once)', 12, 16, c.muted, '11px sans-serif');
      const rowH = 22, y0 = 26, cur = pairs[Math.min(pos, pairs.length - 1)];
      sents.slice(0, 7).forEach((s, si) => {
        const t = ['<s>', ...core.tokens(s), '</s>']; let x = 12; const y = y0 + si * rowH + 14;
        t.forEach((w, i) => {
          const wd = g.measureText(w).width + 10;
          const isCur = pos < pairs.length && cur.si === si && (cur.i === i || cur.i - 1 === i);
          const done = pairs.findIndex((p) => p.si === si && p.i === i) < pos && i > 0;
          if (isCur) { g.fillStyle = i === cur.i ? c.orange : c.blue; g.globalAlpha = 0.85; rrect(g, x - 2, y - 14, wd, 19, 4); g.fill(); g.globalAlpha = 1; }
          txt(g, w, x + 3, y, isCur ? '#fff' : (done ? c.fg : c.muted), '13px ui-monospace,Menlo,Consolas,monospace');
          x += wd + 3;
        });
      });
      const topEnd = y0 + Math.min(sents.length, 7) * rowH + 10;
      // bottom: bars for the chosen context
      const n = tot[ctxw] || 0, V = m.vocab.length;
      txt(g, `P(next word | previous word = "${ctxw}")   counted so far: ${n} pair${n === 1 ? '' : 's'}, k = ${k}`, 12, topEnd + 14, c.fg, 'bold 12px sans-serif');
      const by = topEnd + 26, avail = H - by - 6, rh = Math.min(22, avail / V), L = 78, R = W - 150;
      m.vocab.forEach((w, i) => {
        const cnt = (cc[ctxw] && cc[ctxw][w]) || 0, d = n + k * V, p = d > 0 ? (cnt + k) / d : 0, y = by + i * rh;
        txt(g, w, L - 8, y + rh * 0.68, c.fg, '12px ui-monospace,Menlo,Consolas,monospace', 'right');
        g.fillStyle = c.border; g.globalAlpha = 0.4; g.fillRect(L, y + 2, R - L, rh - 4); g.globalAlpha = 1;
        g.fillStyle = cnt > 0 ? c.green : c.purple; g.fillRect(L, y + 2, p * (R - L), rh - 4);
        txt(g, d > 0 ? `${cnt + k} / ${n + k * V} = ${f3(p)}` : 'no data yet', R + 8, y + rh * 0.68, c.muted, '11px ui-monospace,Menlo,Consolas,monospace');
      });
      let tp = 0; m.vocab.forEach((w) => { const cnt = (cc[ctxw] && cc[ctxw][w]) || 0, d = n + k * V; if (d > 0) tp += (cnt + k) / d; });
      q(ctl, '#out').innerHTML = `pairs counted: ${pos} / ${pairs.length}   vocabulary V = ${V} (including the end marker)\nsum of the bars = ${f3(tp)}${tp > 0 ? '' : '  (no counts: the model has nothing to say)'}${genText ? '\ngenerated: ' + genText : ''}`;
      q(ctl, '#out').style.whiteSpace = 'pre-wrap';
      let cap = 'Press play: the highlighted pair (orange follows blue) is counted, and the bars for the chosen context change.';
      if (pos >= pairs.length) cap = k === 0 ? 'Counting is done. Words never seen after this context have probability exactly 0. Raise k: every word gets k pseudo-counts, so nothing is impossible, and the seen words give up some probability.' : 'Counting is done. With k > 0 the unseen words (purple) share the probability that k pseudo-counts reserve for them.';
      else if (pos > 0) cap = `Counted ${pairs[pos - 1].prev} → ${pairs[pos - 1].next}. Each count adds 1 to one cell of the table; the bar is that cell divided by the row total (plus smoothing).`;
      q(ctl, '#cap').textContent = cap;
      showVals(ctl, { k: (v) => v, rate: (v) => v });
    };
    const step = () => { if (pos < pairs.length) pos++; dirty = true; };
    wire(ctl, st, step, () => { pos = 0; genText = ''; dirty = true; }, draw);
    q(ctl, '#corp').oninput = () => { parse(); draw(); };
    ['#ctx', '#k'].forEach((id) => { q(ctl, id).oninput = () => { dirty = true; }; });
    q(ctl, '#rate').oninput = () => { dirty = true; };
    q(ctl, '#gen').onclick = () => { if (pos < pairs.length) pos = pairs.length; gen(); draw(); };
    o.onresize = draw; parse(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && pos < pairs.length) { const n = tick(dt, +q(ctl, '#rate').value); if (n) { pos = Math.min(pairs.length, pos + n); dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     2. Perplexity of a sentence token by token
     ===================================================================== */
  AI.labs['gen-lm-perplexity'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Perplexity, one token at a time', 'illustrative: the model is the bigram (or unigram) model from the three training sentences in lesson 1.1, scored on a sentence you type',
      `<canvas style="height:400px"></canvas>`,
      `<label>Sentence to score (words in lower case)<input type="text" id="sent" value="the dog ate the fish"></label>
       <label>Model <select id="mod"><option value="bi">bigram (previous word)</option><option value="uni">unigram (no context)</option></select></label>
       <label>Logarithm base <select id="base"><option value="2">base 2 (bits)</option><option value="e">base e (nats)</option></select></label>
       ${sl('k', 'Add-k smoothing k', 0.05, 1, 0.05, 1)}
       ${sl('rate', 'Tokens scored per second', 0.2, 3, 0.1, 0.6)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const train = DEFCORP.split('\n'), bm = core.bigrams(train), um = core.unigram(train);
    const st = { playing: true };
    let toks, ps, losses, n, dirty = true, tick = AI.ticker(), nChars;
    const setup = () => {
      const words = core.tokens(q(ctl, '#sent').value).slice(0, 10); toks = ['<s>', ...words, '</s>'];
      const k = +q(ctl, '#k').value, uni = q(ctl, '#mod').value === 'uni'; ps = [];
      for (let i = 1; i < toks.length; i++) ps.push(uni ? core.pUnigram(um, toks[i], k) : core.pBigram(bm, toks[i - 1], toks[i], k));
      nChars = words.join(' ').length; n = 0; losses = ps.map((p) => -Math.log2(p)); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const natural = q(ctl, '#base').value === 'e', unit = natural ? 'nats' : 'bits', sc = natural ? Math.LN2 : 1;
      const N = ps.length, L = 50, R = W - 12, T = 44, B = H - 120, cw = (R - L) / N;
      const mx = Math.max(1, ...losses.map((x) => x * sc)) * 1.1;
      txt(g, `loss per token = −log p   (${natural ? 'natural log, nats' : 'base 2, bits'})`, L, 16, c.fg, 'bold 12px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(L, T); g.lineTo(L, B); g.lineTo(R, B); g.stroke();
      txt(g, '0', L - 6, B + 4, c.muted, '11px sans-serif', 'right'); txt(g, AI.fmt(mx, 1), L - 6, T + 8, c.muted, '11px sans-serif', 'right');
      let run = 0;
      for (let i = 0; i < N; i++) {
        const x = L + i * cw, shown = i < n, h = shown ? losses[i] * sc / mx * (B - T) : 0;
        if (shown) run += losses[i] * sc;
        g.fillStyle = i === n - 1 ? c.orange : c.blue; g.globalAlpha = shown ? 0.85 : 0; g.fillRect(x + cw * 0.2, B - h, cw * 0.6, h); g.globalAlpha = 1;
        txt(g, toks[i + 1], x + cw / 2, B + 15, shown ? c.fg : c.muted, '12px ui-monospace,Menlo,Consolas,monospace', 'center');
        if (shown) { txt(g, 'p=' + ps[i].toFixed(3), x + cw / 2, B + 30, c.muted, '10.5px ui-monospace,Menlo,Consolas,monospace', 'center'); txt(g, (losses[i] * sc).toFixed(2), x + cw / 2, B - h - 4, c.fg, '11px sans-serif', 'center'); }
      }
      const mean = n ? run / n : 0;
      if (n) { const y = B - mean / mx * (B - T); g.strokeStyle = c.green; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke(); g.setLineDash([]); g.lineWidth = 1; txt(g, `running mean ${mean.toFixed(3)} ${unit}`, R, y - 5, c.green, 'bold 11px sans-serif', 'right'); }
      const pp = n ? (natural ? Math.exp(mean) : Math.pow(2, mean)) : NaN;
      txt(g, n ? `perplexity = ${natural ? 'e' : '2'}^${mean.toFixed(3)} = ${pp.toFixed(3)}` : 'perplexity: nothing scored yet', L, H - 50, c.fg, 'bold 15px sans-serif');
      txt(g, '= as unsure as a fair die with this many sides.', L, H - 28, c.muted, '12px sans-serif');
      const totBits = losses.slice(0, n).reduce((a, b) => a + b, 0);
      q(ctl, '#out').textContent = n ? `scored ${n} of ${N} tokens (the last is the end marker)\nmean loss ${(totBits / n).toFixed(4)} bits = ${(totBits / n * Math.LN2).toFixed(4)} nats\nperplexity ${Math.pow(2, totBits / n).toFixed(4)}  (the same in either base)\n` + (n === N ? `total ${totBits.toFixed(3)} bits over ${nChars} characters = ${(totBits / nChars).toFixed(3)} bits per character` : '') : 'press play';
      q(ctl, '#cap').textContent = n === 0 ? 'Press play. Each token costs −log of the probability the model gave it: probable tokens cost little, surprising ones cost a lot.' : n < N ? `Scoring "${toks[n]}": the model gave it p = ${ps[n - 1].toFixed(3)}, a loss of ${(losses[n - 1] * sc).toFixed(3)} ${unit}.` : 'Done. Switch the base: the bar heights change, the perplexity does not. Try the unigram model, or an unseen word, and watch the perplexity move.';
      showVals(ctl, { k: (v) => v, rate: (v) => v });
    };
    const step = () => { if (n < ps.length) n++; dirty = true; };
    wire(ctl, st, step, () => { n = 0; dirty = true; }, draw);
    ['#sent', '#mod', '#k'].forEach((id) => { q(ctl, id).oninput = () => { setup(); draw(); }; });
    q(ctl, '#base').oninput = () => { dirty = true; };
    o.onresize = draw; setup(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && n < ps.length) { const k = tick(dt, +q(ctl, '#rate').value); if (k) { n = Math.min(ps.length, n + k); dirty = true; } }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3. Byte-pair encoding: the real merge algorithm, one merge at a time
     ===================================================================== */
  const BPE_DEF = 'low 5\nlower 2\nnewest 6\nwidest 3';
  const bpeText = (s) => s.split('\n').map((l) => l.trim().split(/\s+/)).filter((a) => a[0]).map(([w, n]) => Array(Math.max(1, Math.min(50, parseInt(n, 10) || 1))).fill(w.toLowerCase()).join(' ')).join(' ');
  const symCol = (c, s) => (s === '_' ? c.muted : (s.length > 1 ? c.purple : c.blue));

  AI.labs['gen-lm-bpe'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Byte-pair encoding: merge the most frequent pair', 'illustrative scale, real algorithm: every merge is chosen by counting adjacent pairs; "_" marks the end of a word; ties go to the pair that sorts first',
      `<canvas style="height:450px"></canvas>`,
      `<label>Word list ("word count" per line; edit it)<textarea id="corp" rows="4" style="background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-family:ui-monospace,Menlo,Consolas,monospace;width:100%;box-sizing:border-box">${BPE_DEF}</textarea></label>
       ${sl('M', 'Number of merges to run', 1, 20, 1, 12)}
       ${sl('rate', 'Merges per second', 0.1, 1.5, 0.05, 0.25)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const st = { playing: true };
    let T, base, prog, dirty = true;
    const train = () => {
      const text = bpeText(q(ctl, '#corp').value); T = core.bpeTrain(text, 40);
      base = core.bpeWords(text).map((x) => x.sym); prog = 0; dirty = true;
    };
    const Mx = () => Math.min(+q(ctl, '#M').value, T.steps.length);
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const M = Mx(), s = Math.min(M, Math.floor(prog)), f = prog >= M ? 1 : prog - s, fin = prog >= M;
      const words = core.bpeWords(bpeText(q(ctl, '#corp').value));
      const showAfter = fin || f >= 0.5;
      const syms = fin ? (M ? T.steps[M - 1].after : base) : (showAfter ? T.steps[s].after : (s ? T.steps[s - 1].after : base));
      const step = fin ? null : T.steps[s];
      const split = Math.round(W * 0.58);
      txt(g, 'words (symbol boxes)', 12, 16, c.muted, '11px sans-serif');
      const rowH = 34; g.font = '13px ui-monospace,Menlo,Consolas,monospace';
      words.slice(0, 8).forEach((w, i) => {
        let x = 12; const y = 30 + i * rowH;
        txt(g, '×' + w.f, 12, y + 17, c.muted, '12px sans-serif'); x = 46;
        syms[i].forEach((sy) => {
          const wd = Math.max(22, g.measureText(sy).width + 12);
          const isNew = step && showAfter && sy === step.pair[0] + step.pair[1];
          const isPair = step && !showAfter && f > 0.2;
          g.fillStyle = isNew ? c.orange : c.bg2; g.globalAlpha = isNew ? Math.max(0.4, 1 - (f - 0.5) * 1.2) : 1; rrect(g, x, y, wd, 26, 5); g.fill(); g.globalAlpha = 1;
          g.strokeStyle = symCol(c, sy); g.lineWidth = 1.5; rrect(g, x, y, wd, 26, 5); g.stroke(); g.lineWidth = 1;
          txt(g, sy, x + wd / 2, y + 18, isNew ? '#fff' : c.fg, '13px ui-monospace,Menlo,Consolas,monospace', 'center'); x += wd + 4;
        });
      });
      // right: pair counts
      const rx = split + 14; txt(g, step ? `pair counts before merge ${s + 1}` : 'pair counts', rx, 16, c.muted, '11px sans-serif');
      if (step) {
        const mx = step.table[0].n;
        step.table.forEach((p, i) => {
          const y = 28 + i * 26, win = i === 0 && f > 0.25;
          g.fillStyle = win ? c.orange : c.blue; g.globalAlpha = win ? 0.95 : 0.55; g.fillRect(rx + 66, y, (W - rx - 110) * p.n / mx, 18); g.globalAlpha = 1;
          txt(g, p.a + ' ' + p.b, rx + 60, y + 14, c.fg, '12px ui-monospace,Menlo,Consolas,monospace', 'right'); txt(g, String(p.n), rx + 70 + (W - rx - 110) * p.n / mx, y + 14, c.fg, 'bold 12px sans-serif');
        });
      } else txt(g, `done: ${M} merge${M === 1 ? '' : 's'}`, rx, 40, c.green, 'bold 13px sans-serif');
      // bottom: merges list and vocabulary
      const baseSet = new Set(base.flat()), done = fin ? M : s + (showAfter ? 1 : 0);
      const list = T.merges.slice(0, done).map((p, i) => `${i + 1}. ${p[0]} + ${p[1]} → ${p[0] + p[1]}`);
      const by = 30 + 8 * rowH - 2 > H - 90 ? H - 90 : 30 + Math.min(8, words.length) * rowH + 8;
      txt(g, `merge list so far (${done}):`, 12, by + 8, c.muted, '11px sans-serif');
      const colsN = Math.max(1, Math.floor((W - 24) / 150));
      list.forEach((t, i) => txt(g, t, 12 + (i % colsN) * 150, by + 26 + Math.floor(i / colsN) * 17, i === done - 1 ? c.orange : c.fg, '12px ui-monospace,Menlo,Consolas,monospace'));
      q(ctl, '#out').textContent = `starting symbols (characters + end marker): ${baseSet.size}\nmerges applied: ${done}\nvocabulary size = ${baseSet.size} + ${done} = ${baseSet.size + done}` + (step && f > 0.25 && f < 0.5 ? `\nmost frequent pair: "${step.pair[0]}" + "${step.pair[1]}", ${step.n} occurrences` : '');
      q(ctl, '#cap').textContent = fin ? (M ? 'Finished. Each merge became a new vocabulary item; frequent words end up as one symbol, rare words stay in pieces.' : 'Press play.') : (f < 0.25 ? `Step ${s + 1}: count every adjacent pair of symbols, weighted by word frequency.` : (f < 0.5 ? `The most frequent pair is "${step.pair[0]} ${step.pair[1]}" with ${step.n} occurrences.` : `Merged "${step.pair[0]}" and "${step.pair[1]}" into the new symbol "${step.pair[0] + step.pair[1]}" everywhere it occurs.`));
      showVals(ctl, { M: (v) => v, rate: (v) => v });
    };
    const stepFn = () => { const M = Mx(); prog = prog >= M ? M : Math.floor(prog) + 1; dirty = true; };
    wire(ctl, st, stepFn, () => { prog = 0; dirty = true; }, draw);
    q(ctl, '#corp').oninput = () => { train(); draw(); };
    q(ctl, '#M').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; train(); draw();
    AI.loop(root, (dt) => {
      const M = Mx();
      if (st.playing && prog < M) { prog = Math.min(M, prog + dt * +q(ctl, '#rate').value); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     3b. Encoding text with the learned merges (+ byte fallback, vocabulary size trade-off)
     ===================================================================== */
  AI.labs['gen-lm-encode'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Encoding text with learned merges', 'illustrative: merges learned from the word list of the previous lab (low, lower, newest, widest); base symbols are the 26 lowercase letters, anything else falls back to UTF-8 bytes',
      `<canvas style="height:420px"></canvas>`,
      `<label>Text to encode<input type="text" id="txt" value="the lowest newest wider café"></label>
       ${sl('use', 'Merges applied (the first m of 12)', 0, 12, 1, 12)}
       ${sl('rate', 'Merges per second', 0.2, 3, 0.1, 0.7)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:6em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv);
    const text0 = bpeText(BPE_DEF), T = core.bpeTrain(text0, 12), alphabet = new Set([...'abcdefghijklmnopqrstuvwxyz', '_']);
    const st = { playing: true }; let prog = 0, dirty = true;
    const enc = (word, m) => {
      let sym = []; for (const ch of word) { if (alphabet.has(ch)) sym.push(ch); else Array.from(new TextEncoder().encode(ch)).forEach((b) => sym.push('<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>')); }
      sym.push('_'); T.merges.slice(0, m).forEach(([a, b]) => { sym = core.applyMerge(sym, a, b); }); return sym;
    };
    const words = () => core.tokens(q(ctl, '#txt').value).slice(0, 8);
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const use = +q(ctl, '#use').value, m = Math.floor(Math.min(prog, use)), f = Math.min(prog, use) - m, ws = words();
      g.font = '13px ui-monospace,Menlo,Consolas,monospace';
      txt(g, `after ${m} merge${m === 1 ? '' : 's'}   blue: letter, purple: merged, red: byte`, 12, 16, c.muted, '11px sans-serif');
      let total = 0; const rowH = 32;
      ws.forEach((w, i) => {
        const y = 26 + i * rowH; let x = 12; const e = enc(w, m); total += e.length;
        const last = m > 0 && prog < use ? T.merges[m - 1] : null;
        e.forEach((sy) => {
          const wd = Math.max(22, g.measureText(sy).width + 12), isRaw = /^<0x/.test(sy), fresh = last && sy === last[0] + last[1] && f < 0.6;
          g.fillStyle = fresh ? c.orange : c.bg2; rrect(g, x, y, wd, 24, 5); g.fill();
          g.strokeStyle = isRaw ? c.red : symCol(c, sy); g.lineWidth = 1.5; rrect(g, x, y, wd, 24, 5); g.stroke(); g.lineWidth = 1;
          txt(g, sy, x + wd / 2, y + 17, fresh ? '#fff' : c.fg, '13px ui-monospace,Menlo,Consolas,monospace', 'center'); x += wd + 4;
        });
      });
      // chart: tokens vs merges
      const cy = 26 + 8 * rowH + 18, L = 46, R = W - 16, B = H - 26, Tp = cy + 12;
      const counts = []; for (let k = 0; k <= 12; k++) counts.push(ws.reduce((s, w) => s + enc(w, k).length, 0));
      const mxc = Math.max(1, ...counts);
      txt(g, 'tokens needed for this text, by number of merges', L, cy + 2, c.muted, '11px sans-serif');
      const bw = (R - L) / 13;
      counts.forEach((v, k) => { const h = v / mxc * (B - Tp); g.fillStyle = k === m ? c.orange : c.blue; g.globalAlpha = k <= m ? 0.9 : 0.35; g.fillRect(L + k * bw + 3, B - h, bw - 6, h); g.globalAlpha = 1; txt(g, String(v), L + k * bw + bw / 2, B - h - 3, c.fg, '10.5px sans-serif', 'center'); txt(g, String(k), L + k * bw + bw / 2, B + 13, c.muted, '10.5px sans-serif', 'center'); });
      txt(g, 'merges', L - 8, B + 13, c.muted, '10.5px sans-serif', 'right');
      const chars = ws.reduce((s, w) => s + [...w].length + 1, 0);
      q(ctl, '#out').textContent = `words: ${ws.length}   characters incl. end markers: ${chars}\ntokens now: ${total}   (${(total / Math.max(1, ws.length)).toFixed(2)} per word)\nvocabulary now: 26 letters + end mark + ${m} merges = ${alphabet.size + m}\ntokens with 0 merges: ${counts[0]}, with all 12: ${counts[12]}`;
      q(ctl, '#cap').textContent = prog >= use ? 'Merges are applied in the order they were learned, never re-counted on the new text. A character that is not one of the 26 letters ("é") is spelled with its UTF-8 bytes, so any text can be encoded.' : `Applying merge ${m + 1}: ${T.merges[m][0]} + ${T.merges[m][1]} → ${T.merges[m][0] + T.merges[m][1]} wherever the two are adjacent.`;
      showVals(ctl, { use: (v) => v, rate: (v) => v });
    };
    wire(ctl, st, () => { const u = +q(ctl, '#use').value; prog = prog >= u ? u : Math.floor(prog) + 1; dirty = true; }, () => { prog = 0; dirty = true; }, draw);
    q(ctl, '#txt').oninput = () => { dirty = true; }; q(ctl, '#use').oninput = () => { prog = Math.min(prog, +q(ctl, '#use').value); dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      const u = +q(ctl, '#use').value;
      if (st.playing && prog < u) { prog = Math.min(u, prog + dt * +q(ctl, '#rate').value); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     4. Embeddings: a table lookup, similarity, tied output scores
     ===================================================================== */
  AI.labs['gen-lm-embed'] = function (root) {
    const names = ['cat', 'kitten', 'dog', 'puppy', 'car', 'truck'];
    const E0 = [[0.9, 0.6], [0.8, 0.9], [0.9, 0.3], [0.8, 0.7], [-0.7, 0.1], [-0.8, -0.5]];
    const { view, ctl } = AI.labShell(root, 'The embedding table is a lookup', 'illustrative: six words, two numbers each, set by hand (a trained model learns them). Drag a vector tip on the plot to edit the table',
      `<canvas style="height:540px;touch-action:none;cursor:default"></canvas>`,
      `<label>Selected word (the one-hot input) <select id="sel">${names.map((w, i) => `<option value="${i}">${w}</option>`).join('')}</select></label>
       ${sl('hold', 'Seconds per word in the auto walk', 1, 6, 0.5, 3)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let E = E0.map((r) => r.slice()), sel = 0, slide = 0, clock = 0, dirty = true, drag = -1;
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1], nr = (a) => Math.sqrt(dot(a, a)), cosv = (a, b) => dot(a, b) / (nr(a) * nr(b) || 1);
    const geo = () => { const W = o.w, H = o.h, top = 200, cx = W / 2, cyy = top + (H - top) / 2 + 6, sc = Math.min(W / 2 - 14, (H - top) / 2 - 22) / 1.5; return { W, H, top, cx, cyy, sc }; };
    const draw = () => {
      const c = AI.col(), g = o.ctx, { W, H, top, cx, cyy, sc } = geo(); g.clearRect(0, 0, W, H);
      // left: one-hot and the matrix E
      txt(g, 'one-hot x', 12, 14, c.muted, '11px sans-serif'); txt(g, 'table E (6 rows × 2 columns)', 80, 14, c.muted, '11px sans-serif');
      const rh = 24, y0 = 20, px = W; g.font = '13px ui-monospace,Menlo,Consolas,monospace';
      const yy = y0 + slide * rh;
      g.fillStyle = c.orange; g.globalAlpha = 0.22; g.fillRect(8, yy, 236, rh - 3); g.globalAlpha = 1;
      names.forEach((w, i) => {
        const y = y0 + i * rh, on = i === sel;
        txt(g, on ? '1' : '0', 30, y + 17, on ? c.orange : c.muted, on ? 'bold 14px ui-monospace,Menlo,Consolas,monospace' : '13px ui-monospace,Menlo,Consolas,monospace', 'center');
        txt(g, w, 80, y + 17, on ? c.fg : c.muted, '13px ui-monospace,Menlo,Consolas,monospace');
        txt(g, E[i][0].toFixed(2), 190, y + 17, on ? c.fg : c.muted, '13px ui-monospace,Menlo,Consolas,monospace', 'right'); txt(g, E[i][1].toFixed(2), 236, y + 17, on ? c.fg : c.muted, '13px ui-monospace,Menlo,Consolas,monospace', 'right');
      });
      txt(g, `x · E  =  row ${sel + 1}  =  [${E[sel][0].toFixed(2)}, ${E[sel][1].toFixed(2)}]`, 12, y0 + 6 * rh + 14, c.orange, 'bold 13px ui-monospace,Menlo,Consolas,monospace');
      // right: plot
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(6, cyy); g.lineTo(W - 6, cyy); g.moveTo(cx, top); g.lineTo(cx, H - 6); g.stroke();
      txt(g, 'the same rows as arrows in the 2-D embedding space (drag a tip)', 8, top - 2, c.muted, '11px sans-serif');
      names.forEach((w, i) => {
        const x = cx + E[i][0] * sc, y = cyy - E[i][1] * sc, on = i === sel;
        g.strokeStyle = on ? c.orange : c.blue; g.lineWidth = on ? 3 : 1.5; g.globalAlpha = on ? 1 : 0.8;
        g.beginPath(); g.moveTo(cx, cyy); g.lineTo(x, y); g.stroke(); g.lineWidth = 1; g.globalAlpha = 1;
        g.fillStyle = on ? c.orange : c.blue; g.beginPath(); g.arc(x, y, on ? 6 : 4.5, 0, 7); g.fill();
        txt(g, w, x + 8, y - 6, on ? c.orange : c.fg, '12px sans-serif');
      });
      const out = [];
      const sims = names.map((w, i) => ({ w, i, cos: cosv(E[sel], E[i]), dot: dot(E[sel], E[i]) })).filter((s) => s.i !== sel).sort((a, b) => b.cos - a.cos);
      out.push(`cosine similarity with "${names[sel]}" (|v| = ${nr(E[sel]).toFixed(3)}):`);
      sims.forEach((s) => out.push(`  ${s.w.padEnd(7)} cos ${s.cos.toFixed(3).padStart(6)}   dot ${s.dot.toFixed(3).padStart(6)}`));
      const sc2 = E.map((e) => dot(e, E[sel])), p = core.softmax(sc2);
      out.push('tied output scores h·Eᵀ with h = this row, then softmax:');
      out.push('  ' + names.map((w, i) => `${w} ${p[i].toFixed(3)}`).join('  '));
      q(ctl, '#out').textContent = out.join('\n');
      q(ctl, '#cap').textContent = 'Multiplying a one-hot vector by E just picks one row: nothing is computed, it is a lookup. Words with similar roles end up with similar rows (small angle, cosine near 1). Drag a tip to change a row and watch the cosines move.';
      showVals(ctl, { hold: (v) => v });
    };
    const go = (i) => { sel = i; q(ctl, '#sel').value = String(i); dirty = true; };
    wire(ctl, st, () => go((sel + 1) % 6), () => { E = E0.map((r) => r.slice()); sel = 0; slide = 0; clock = 0; q(ctl, '#sel').value = '0'; dirty = true; }, draw);
    q(ctl, '#sel').oninput = () => { sel = +q(ctl, '#sel').value; st.playing = false; q(ctl, '#play').textContent = '▶ Play'; dirty = true; };
    q(ctl, '#hold').oninput = () => { dirty = true; };
    const pt = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.onpointerdown = (e) => { const { cx, cyy, sc } = geo(), [x, y] = pt(e); let best = -1, bd = 18; E.forEach((v, i) => { const d = Math.hypot(cx + v[0] * sc - x, cyy - v[1] * sc - y); if (d < bd) { bd = d; best = i; } }); if (best >= 0) { drag = best; cv.setPointerCapture(e.pointerId); go(best); st.playing = false; q(ctl, '#play').textContent = '▶ Play'; } };
    cv.onpointermove = (e) => { const { cx, cyy, sc } = geo(), [x, y] = pt(e); if (drag >= 0) { E[drag] = [Math.max(-1.5, Math.min(1.5, (x - cx) / sc)), Math.max(-1.5, Math.min(1.5, (cyy - y) / sc))]; dirty = true; } };
    cv.onpointerup = () => { drag = -1; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing) { clock += dt; if (clock >= +q(ctl, '#hold').value) { clock = 0; go((sel + 1) % 6); } }
      if (Math.abs(slide - sel) > 0.002) { slide += (sel - slide) * Math.min(1, dt * 6); if (Math.abs(slide - sel) < 0.004) slide = sel; dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     5. A neural bigram language model trained with softmax cross-entropy
     ===================================================================== */
  AI.labs['gen-lm-neural'] = function (root) {
    const V = ['the', 'cat', 'dog', 'sat', 'ran', '.'], D = 3;
    const data = [], sentsN = ['the cat sat .', 'the cat ran .', 'the dog sat .', 'the cat sat .'];
    sentsN.forEach((s) => { const t = s.split(' ').map((w) => V.indexOf(w)); for (let i = 0; i < t.length - 1; i++) data.push([t[i], t[i + 1]]); });
    const emp = V.map(() => V.map(() => 0)); data.forEach(([a, b]) => { emp[a][b]++; });
    const empP = emp.map((r) => { const s = r.reduce((x, y) => x + y, 0) || 1; return r.map((v) => v / s); });
    const floor = -data.reduce((s, [a, b]) => s + Math.log(empP[a][b]), 0) / data.length;
    const { view, ctl } = AI.labShell(root, 'Training a tiny neural language model', 'illustrative: 6-word vocabulary, 3-number embeddings, 12 training pairs from "the cat sat .", "the cat ran .", "the dog sat .", "the cat sat ."; plain SGD, one pair per step',
      `<canvas style="height:420px"></canvas>`,
      `<label>Show next-word probabilities after <select id="ctx">${V.map((w, i) => `<option value="${i}">${w}</option>`).join('')}</select></label>
       ${sl('lr', 'Learning rate', 0.05, 2, 0.05, 0.5)}
       ${sl('rate', 'Training steps per second', 0.5, 20, 0.5, 3)}
       <label style="flex-direction:row;align-items:center;gap:8px"><input type="checkbox" id="tie"> tie input and output embeddings (one table)</label>
       ${btns}
       <div class="row"><button class="btn" id="chk">Check gradient numerically</button></div>
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:9em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let E, U, steps, hist, cur, dirty = true, tick = AI.ticker(), note = '';
    const init = () => { const r = AI.rng(7); E = V.map(() => Array.from({ length: D }, () => 0.3 * AI.randn(r))); U = V.map(() => Array.from({ length: D }, () => 0.3 * AI.randn(r))); steps = 0; hist = []; cur = null; note = ''; hist.push(lossAll()); dirty = true; };
    const outW = () => (q(ctl, '#tie').checked ? E : U);
    const logits = (a) => outW().map((u) => u.reduce((s, v, j) => s + v * E[a][j], 0));
    const lossAll = () => data.reduce((s, [a, b]) => s - Math.log(core.softmax(logits(a))[b]), 0) / data.length;
    const sgd = () => {
      const [a, b] = data[steps % data.length], lr = +q(ctl, '#lr').value, tied = q(ctl, '#tie').checked, W = outW();
      const x = E[a].slice(), z = logits(a), p = core.softmax(z), dz = p.map((v, i) => v - (i === b ? 1 : 0));
      const dx = x.map((_, j) => W.reduce((s, u, i) => s + u[j] * dz[i], 0));   // dL/dx = Uᵀ (p − y)
      for (let i = 0; i < V.length; i++) for (let j = 0; j < D; j++) W[i][j] -= lr * dz[i] * x[j];   // dL/dU = (p − y) xᵀ
      for (let j = 0; j < D; j++) E[a][j] -= lr * dx[j];
      cur = { a, b, z, p, dz, loss: -Math.log(p[b]) }; steps++; hist.push(lossAll()); if (hist.length > 600) hist.shift(); dirty = true;
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const ctxi = +q(ctl, '#ctx').value, p = core.softmax(logits(ctxi)), half = Math.round(W * 0.5);
      txt(g, `P(next word | "${V[ctxi]}")`, 12, 14, c.fg, 'bold 12px sans-serif'); txt(g, 'bar: model   green mark: counted in data', 12, 28, c.muted, '10.5px sans-serif');
      const rh = 36, y0 = 40, L = 46, R = half - 50;
      V.forEach((w, i) => {
        const y = y0 + i * rh; txt(g, w, L - 6, y + 22, c.fg, '13px ui-monospace,Menlo,Consolas,monospace', 'right');
        g.fillStyle = c.border; g.globalAlpha = 0.4; g.fillRect(L, y + 6, R - L, 22); g.globalAlpha = 1;
        g.fillStyle = (cur && cur.a === ctxi && cur.b === i) ? c.orange : c.blue; g.fillRect(L, y + 6, p[i] * (R - L), 22);
        g.strokeStyle = c.green; g.lineWidth = 3; g.beginPath(); g.moveTo(L + empP[ctxi][i] * (R - L), y + 3); g.lineTo(L + empP[ctxi][i] * (R - L), y + 31); g.stroke(); g.lineWidth = 1;
        txt(g, p[i].toFixed(3), R + 6, y + 22, c.muted, '11px ui-monospace,Menlo,Consolas,monospace');
      });
      const emptyRow = empP[ctxi].every((v) => v === 0); if (emptyRow) txt(g, 'this word never appears as a context in the data', L, y0 + 6 * rh + 8, c.muted, '11px sans-serif');
      // loss curve
      const cx0 = half + 20, cx1 = W - 14, ct = 40, cb = H - 36;
      txt(g, 'average loss (nats)', cx0, 14, c.fg, 'bold 12px sans-serif'); txt(g, 'over the 12 pairs', cx0, 28, c.muted, '10.5px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(cx0, ct); g.lineTo(cx0, cb); g.lineTo(cx1, cb); g.stroke();
      const mx = Math.max(2, hist[0] * 1.1), Y = (v) => cb - Math.min(1, v / mx) * (cb - ct), first = steps - hist.length + 1, span = Math.max(40, steps);
      txt(g, '0', cx0 - 5, cb + 4, c.muted, '11px sans-serif', 'right'); txt(g, mx.toFixed(1), cx0 - 5, ct + 8, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(cx0, Y(floor)); g.lineTo(cx1, Y(floor)); g.stroke(); g.setLineDash([]);
      txt(g, `floor ${floor.toFixed(3)}`, cx1, Y(floor) - 5, c.muted, '11px sans-serif', 'right');
      g.strokeStyle = c.purple; g.lineWidth = 2; g.beginPath(); hist.forEach((v, i) => { const x = cx0 + (first + i) / span * (cx1 - cx0); i ? g.lineTo(x, Y(v)) : g.moveTo(x, Y(v)); }); g.stroke(); g.lineWidth = 1;
      txt(g, `step ${steps}`, cx1, cb + 18, c.muted, '11px sans-serif', 'right');
      const tied = q(ctl, '#tie').checked;
      q(ctl, '#out').textContent = (cur ? `last example: "${V[cur.a]}" then "${V[cur.b]}"\nlogits z = [${cur.z.map(f3).join(', ')}]\nprobs  p = [${cur.p.map(f3).join(', ')}]\nloss −ln p = ${cur.loss.toFixed(4)}\np − onehot = [${cur.dz.map(f3).join(', ')}]` : 'no step yet: weights are small random numbers, so every word is about 1/6 likely (loss ≈ ln 6 = 1.792)') + `\nparameters: ${tied ? V.length * D + ' (one table)' : 2 * V.length * D + ' (E and U)'}` + (note ? '\n' + note : '');
      q(ctl, '#cap').textContent = steps === 0 ? 'Press play: each step takes one (word, next word) pair, computes probabilities, and nudges the weights along −(p − one-hot).' : 'The correct word\'s bar grows and the others shrink. The model can only reach the frequencies in the data (green lines), never beyond: that is the loss floor.';
      showVals(ctl, { lr: (v) => v, rate: (v) => v });
    };
    const check = () => {
      const [a, b] = data[steps % data.length], z = logits(a), p = core.softmax(z), an = p.map((v, i) => v - (i === b ? 1 : 0)), eps = 1e-5;
      const nu = z.map((_, i) => { const zp = z.slice(), zm = z.slice(); zp[i] += eps; zm[i] -= eps; return (core.xent(zp, b) - core.xent(zm, b)) / (2 * eps); });
      note = `gradient check on next pair "${V[a]}" then "${V[b]}":\nanalytic p − y = [${an.map((v) => v.toFixed(5)).join(', ')}]\nnumeric (central difference) = [${nu.map((v) => v.toFixed(5)).join(', ')}]\nlargest difference ${Math.max(...an.map((v, i) => Math.abs(v - nu[i]))).toExponential(1)}`; dirty = true;
    };
    wire(ctl, st, sgd, init, draw);
    ['#ctx', '#lr', '#rate'].forEach((id) => { q(ctl, id).oninput = () => { dirty = true; }; });
    q(ctl, '#tie').onchange = () => { init(); draw(); }; q(ctl, '#chk').onclick = () => { check(); draw(); };
    o.onresize = draw; init(); draw();
    AI.loop(root, (dt) => {
      if (st.playing && steps < 4000) { const n = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < n; i++) sgd(); }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     6. Decoding: temperature, top-k, top-p, and sampling
     ===================================================================== */
  AI.labs['gen-lm-decode'] = function (root) {
    const names = ['sunny', 'cloudy', 'rainy', 'windy', 'cold', 'purple'], Z0 = [2.0, 1.5, 1.0, 0.2, -0.3, -2.0];
    const { view, ctl } = AI.labShell(root, 'From logits to a chosen word', 'illustrative: six candidate words after "Today the weather is"; edit the logits and the three dials; the order is temperature, then top-k, then top-p, then renormalise',
      `<canvas style="height:560px"></canvas>`,
      `${names.map((w, i) => sl('z' + i, 'logit of ' + w, -4, 4, 0.1, Z0[i])).join('')}
       ${sl('T', 'Temperature T', 0.1, 3, 0.05, 1)}
       ${sl('K', 'Top-k (6 = off)', 1, 6, 1, 6)}
       ${sl('P', 'Top-p (1 = off)', 0.1, 1, 0.05, 1)}
       ${sl('rate', 'Words sampled per second', 0.25, 10, 0.25, 1)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:7em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let counts, n, rs, last = -1, dirty = true, tick = AI.ticker();
    const Zs = () => names.map((_, i) => +q(ctl, '#z' + i).value);
    const pipeline = () => {
      const z = Zs(), T = +q(ctl, '#T').value, K = +q(ctl, '#K').value, P = +q(ctl, '#P').value;
      const p0 = core.softmax(z, T), p1 = core.topK(p0, K), p2 = core.topP(p1, P); return { z, T, K, P, p0, p1, p2 };
    };
    const reset = () => { counts = names.map(() => 0); n = 0; rs = AI.rng(23); last = -1; dirty = true; };
    const draw1 = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pl = pipeline(), L = 62, R = W - 70, secH = (H - 10) / 4;
      const section = (title, p, k, col, extra) => {
        const y0 = 8 + k * secH, rh = (secH - 26) / 6; txt(g, title, 8, y0 + 12, c.fg, 'bold 12px sans-serif');
        p.forEach((v, i) => {
          const y = y0 + 20 + i * rh; txt(g, names[i], L - 6, y + rh * 0.72, v > 0 ? c.fg : c.muted, '11.5px sans-serif', 'right');
          g.fillStyle = col; g.globalAlpha = v > 0 ? 0.9 : 0.15; g.fillRect(L, y + 1.5, Math.max(1, v * (R - L)), rh - 3); g.globalAlpha = 1;
          txt(g, v > 0 ? v.toFixed(4) : 'removed', R + 6, y + rh * 0.72, v > 0 ? c.fg : c.red, '11px ui-monospace,Menlo,Consolas,monospace');
        });
      };
      section(`① softmax(z / T), T = ${pl.T}`, pl.p0, 0, c.blue);
      section(`② top-k = ${pl.K}: keep the ${pl.K} most probable, renormalised`, pl.p1, 1, c.purple);
      section(`③ top-p = ${pl.P}: smallest set with sum ≥ ${pl.P}, renormalised`, pl.p2, 2, c.green);
      // sampled histogram
      const y0 = 8 + 3 * secH, rh = (secH - 26) / 6;
      txt(g, `④ ${n} sampled word${n === 1 ? '' : 's'} (bars) against the distribution ③ (marks)`, 8, y0 + 12, c.fg, 'bold 12px sans-serif');
      names.forEach((w, i) => {
        const y = y0 + 20 + i * rh, fr = n ? counts[i] / n : 0;
        txt(g, w, L - 6, y + rh * 0.72, c.fg, '11.5px sans-serif', 'right');
        g.fillStyle = i === last ? c.orange : c.teal || c.blue; g.globalAlpha = 0.9; g.fillRect(L, y + 1.5, Math.max(0, fr * (R - L)), rh - 3); g.globalAlpha = 1;
        g.strokeStyle = c.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(L + pl.p2[i] * (R - L), y); g.lineTo(L + pl.p2[i] * (R - L), y + rh); g.stroke(); g.lineWidth = 1;
        txt(g, n ? `${counts[i]}  (${fr.toFixed(3)})` : '', R + 6, y + rh * 0.72, c.muted, '11px ui-monospace,Menlo,Consolas,monospace');
      });
      const H2 = (p) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
      const arg = pl.p0.indexOf(Math.max(...pl.p0));
      q(ctl, '#out').textContent = `greedy choice (argmax): ${names[arg]}\nentropy before cuts ${H2(pl.p0).toFixed(3)} bits, after cuts ${H2(pl.p2).toFixed(3)} bits\nwords still possible after the cuts: ${pl.p2.filter((v) => v > 0).length} of 6\nmass kept by top-p before renormalising: ${pl.p1.filter((v, i) => pl.p2[i] > 0).reduce((s, v) => s + v, 0).toFixed(4)}`;
      q(ctl, '#cap').textContent = pl.T < 0.2 ? 'Very low temperature: the top word takes nearly everything, so sampling behaves like greedy decoding.' : (pl.T > 2 ? 'High temperature flattens the distribution: unlikely words (even "purple") are chosen often. Truncation (top-k, top-p) cuts that tail off.' : 'Lower T sharpens panel 1, higher T flattens it. Top-k and top-p remove the tail and divide the survivors by their sum, so panel 3 always adds to 1.');
      showVals(ctl, Object.assign({ T: (v) => v, K: (v) => (v === 6 ? 'off' : v), P: (v) => (v === 1 ? 'off' : v), rate: (v) => v }, ...names.map((_, i) => ({ ['z' + i]: (v) => v.toFixed(1) }))));
    };
    const sample = () => { const p = pipeline().p2; let u = rs(), j = 0; while (j < p.length - 1 && u >= p[j]) { u -= p[j]; j++; } counts[j]++; n++; last = j; dirty = true; };
    wire(ctl, st, sample, reset, draw1);
    ['T', 'K', 'P', 'rate', ...names.map((_, i) => 'z' + i)].forEach((id) => { q(ctl, '#' + id).oninput = () => { if (id !== 'rate') { reset(); } dirty = true; }; });
    o.onresize = draw1; reset(); draw1();
    AI.loop(root, (dt) => {
      if (st.playing && n < 5000) { const k = tick(dt, +q(ctl, '#rate').value); for (let i = 0; i < k; i++) sample(); }
      if (dirty) { draw1(); dirty = false; }
    }, { fps: 30 });
  };

  /* =====================================================================
     7. Greedy versus beam search on a two-step tree
     ===================================================================== */
  AI.labs['gen-lm-beam'] = function (root) {
    const A = ['a', 'b', 'c'], B = ['x', 'y', 'z'], P1 = [0.5, 0.4, 0.1];
    const { view, ctl } = AI.labShell(root, 'Greedy versus beam search', 'illustrative: a two-word sentence, three choices per word; the numbers are made up so the effect is visible',
      `<canvas style="height:430px"></canvas>`,
      `${sl('w', 'Beam width (1 = greedy)', 1, 3, 1, 2)}
       ${sl('sx', 'P(second = x | first = b)', 0.05, 0.95, 0.05, 0.9)}
       ${sl('rate', 'Animation speed (phases per second)', 0.1, 1, 0.05, 0.3)}
       ${btns}
       <div class="readout" id="out" style="white-space:pre-wrap;min-height:8em"></div>
       <p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), st = { playing: true };
    let prog = 0, dirty = true;
    const model = () => { const sx = +q(ctl, '#sx').value, r = (1 - sx) / 2; return [[0.4, 0.3, 0.3], [sx, r, r], [0.34, 0.33, 0.33]]; };
    const search = () => {
      const w = +q(ctl, '#w').value, P2 = model();
      const lvl1 = P1.map((p, i) => ({ i, p })).sort((u, v) => v.p - u.p || u.i - v.i), kept1 = lvl1.slice(0, w).map((x) => x.i);
      const all = []; kept1.forEach((i) => B.forEach((_, j) => all.push({ i, j, p: P1[i] * P2[i][j] })));
      all.sort((u, v) => v.p - u.p || u.i - v.i || u.j - v.j);
      const full = []; P1.forEach((p, i) => B.forEach((_, j) => full.push({ i, j, p: p * P2[i][j] }))); full.sort((u, v) => v.p - u.p || u.i - v.i || u.j - v.j);
      return { w, P2, kept1, all, best: all[0], full, top: full[0] };
    };
    const draw = () => {
      const c = AI.col(), g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const S = search(), ph = Math.min(3, prog), e1 = Math.min(1, ph), e2 = Math.max(0, Math.min(1, ph - 1)), e3 = Math.max(0, Math.min(1, ph - 2));
      const rx = 50, x1 = W * 0.36, x2 = W * 0.78, y1 = (i) => 70 + i * 100;
      const y2 = (i, j) => y1(i) + (j - 1) * 28;
      txt(g, 'first word', x1, 16, c.muted, '11px sans-serif', 'center'); txt(g, 'second word', x2, 16, c.muted, '11px sans-serif', 'center'); txt(g, 'start', rx, 16, c.muted, '11px sans-serif', 'center');
      // level 1
      P1.forEach((p, i) => {
        const kept = S.kept1.includes(i), a = ease(e1);
        g.strokeStyle = kept ? c.blue : c.muted; g.globalAlpha = (0.2 + 0.8 * a) * (e3 > 0 && !kept ? 0.3 : 1); g.lineWidth = 1 + 6 * p;
        g.beginPath(); g.moveTo(rx, 200); g.lineTo(rx + (x1 - rx) * a, 200 + (y1(i) - 200) * a); g.stroke(); g.lineWidth = 1;
        if (a > 0.95) { g.globalAlpha = 1; txt(g, `p = ${p}`, (rx + x1) / 2 - 20, 200 + (y1(i) - 200) / 2 - 6, c.muted, '11px sans-serif', 'center'); }
        g.globalAlpha = 1;
      });
      P1.forEach((p, i) => { const kept = S.kept1.includes(i); g.fillStyle = e2 > 0 && !kept ? c.border : c.bg2; g.strokeStyle = kept || e2 === 0 ? c.blue : c.muted; rrect(g, x1 - 18, y1(i) - 14, 36, 28, 6); g.fill(); g.stroke(); txt(g, A[i], x1, y1(i) + 5, c.fg, 'bold 14px sans-serif', 'center'); });
      txt(g, 'start', rx, 204, c.fg, '12px sans-serif', 'center');
      // level 2 (only for kept prefixes)
      S.kept1.forEach((i) => B.forEach((_, j) => {
        const a = ease(e2), p2 = S.P2[i][j], tot = P1[i] * p2, isBest = e3 > 0.5 && S.best.i === i && S.best.j === j, inAll = S.all.slice(0, S.w).some((x) => x.i === i && x.j === j);
        g.strokeStyle = isBest ? c.orange : c.blue; g.globalAlpha = (0.2 + 0.8 * a) * (e3 > 0 && !inAll ? 0.3 : 1); g.lineWidth = isBest ? 3 : 1 + 5 * p2;
        g.beginPath(); g.moveTo(x1 + 18, y1(i)); g.lineTo(x1 + 18 + (x2 - 18 - x1 - 18) * a, y1(i) + (y2(i, j) - y1(i)) * a); g.stroke(); g.lineWidth = 1;
        if (a > 0.95) { g.globalAlpha = 1; g.fillStyle = isBest ? c.orange : (e3 > 0 && !inAll ? c.border : c.bg2); g.strokeStyle = isBest ? c.orange : c.blue; rrect(g, x2 - 18, y2(i, j) - 11, 36, 22, 5); g.fill(); g.stroke(); txt(g, B[j], x2, y2(i, j) + 4, isBest ? '#fff' : c.fg, '13px sans-serif', 'center'); txt(g, `${A[i]}${B[j]}: ${tot.toFixed(3)}`, x2 + 24, y2(i, j) + 4, inAll && e3 > 0 ? c.fg : c.muted, '11px ui-monospace,Menlo,Consolas,monospace'); }
        g.globalAlpha = 1;
      }));
      const bestS = A[S.best.i] + B[S.best.j], topS = A[S.top.i] + B[S.top.j];
      q(ctl, '#out').textContent = `beam width ${S.w}: kept first words {${S.kept1.map((i) => A[i]).join(', ')}}\nsearch returns "${bestS}" with probability ${S.best.p.toFixed(4)}\nbest sentence of all 9: "${topS}" with ${S.top.p.toFixed(4)}\n` + (S.best.p >= S.top.p - 1e-12 ? 'the search found the best sentence' : 'the search MISSED the best sentence') + `\n(P(second|first) rows: a: 0.4/0.3/0.3, b: ${S.P2[1][0].toFixed(2)}/${S.P2[1][1].toFixed(3)}/${S.P2[1][2].toFixed(3)}, c: 0.34/0.33/0.33)`;
      q(ctl, '#cap').textContent = ph < 1 ? 'Phase 1: score every first word by its probability.' : ph < 2 ? `Phase 2: keep only the best ${S.w} first word${S.w > 1 ? 's' : ''} and extend each with every second word. Words that were dropped are never explored.` : (prog < 3 ? 'Phase 3: multiply the probabilities along each path and keep the best.' : 'Done. Greedy (width 1) commits to the locally best first word "a"; a wider beam also tried "b", whose continuation "x" is very likely. Lower P(x | b) below 0.5 and greedy becomes right again.');
      showVals(ctl, { w: (v) => v, sx: (v) => v.toFixed(2), rate: (v) => v });
    };
    wire(ctl, st, () => { prog = Math.min(3, Math.floor(prog + 1e-9) + 1); dirty = true; }, () => { prog = 0; dirty = true; }, draw);
    q(ctl, '#w').oninput = () => { prog = Math.min(prog, 2); dirty = true; }; q(ctl, '#sx').oninput = () => { dirty = true; }; q(ctl, '#rate').oninput = () => { dirty = true; };
    o.onresize = draw; draw();
    AI.loop(root, (dt) => {
      if (st.playing && prog < 3) { prog = Math.min(3, prog + dt * +q(ctl, '#rate').value * 3 / 3); dirty = true; }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
  };

  if (typeof module !== 'undefined') module.exports = { core };
})();
