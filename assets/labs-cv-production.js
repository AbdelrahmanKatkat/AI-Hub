/* Computer-vision production labs (Module 9): augment, finetune, pipesim, quant, prune, distill, drift, fgsm, memcalc.
 * Requires assets/lib.js, assets/arch.js and assets/labs-cv.js (AI.cv scenes).
 *
 *   <div data-lab="augment"></div>          augmentation playground: random samples generated one at a time
 *   <div data-lab="finetune"></div>         ResNet-50 freeze toggles, trainable parameters, layer-wise LR, LoRA, loss comparison
 *   <div data-lab="pipesim"></div>          end-to-end pipeline simulator (queues, batching, utilisation, Little's law)
 *   <div data-lab="quant"></div>            float weights snapping to integer levels
 *   <div data-lab="prune"></div>            magnitude / structured pruning of a weight matrix
 *   <div data-lab="distill"></div>          teacher/student softmax with temperature
 *   <div data-lab="drift"></div>            training vs live distribution, PSI/KS detector
 *   <div data-lab="fgsm"></div>             adversarial perturbation on a tiny linear classifier
 *   <div data-lab="memcalc" data-mode="infer|train"></div>   memory = weights + activations + workspace / optimizer
 * All animations are dt-based, slow by default, with a speed slider and pause/step.
 */
(function () {
  const clamp = AI.clamp, lerp = AI.lerp, smooth = AI.smooth;
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const transport = (extra) => `<div class="row"><button class="btn small primary" id="play">Pause</button><button class="btn small" id="step">Step</button><button class="btn small" id="reset">Restart</button>${extra || ''}</div>${sl('sp', 'Speed', 0.25, 3, 0.05, 1)}`;
  const wire = (root, fmts) => root.querySelectorAll('input[type=range]').forEach((inp) => {
    const v = root.querySelector('#' + inp.id + 'v'), f = fmts && fmts[inp.id];
    const u = () => { if (v) v.textContent = f ? f(+inp.value) : AI.fmt(+inp.value, 3); };
    inp.addEventListener('input', u); u();
  });
  const big = (n) => (Math.round(n)).toLocaleString('en-US');
  const mb = (b) => (b / 1e6 >= 1000 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB');
  const fmtN = (v) => (v >= 1e9 ? (v / 1e9).toFixed(2) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' K' : String(Math.round(v)));
  const ease = (dt, rate) => 1 - Math.exp(-dt * rate);
  const Phi = (z) => { // standard normal CDF (Abramowitz-Stegun 7.1.26)
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  };
  const text = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = al || 'left'; g.fillText(s, x, y); };

  /* ====================================================================== augment */
  AI.labs.augment = function (root) {
    const CV = AI.cv;
    const { view, ctl } = AI.labShell(root, 'Augmentation playground', 'edit the ranges; one random augmented sample is generated at a time, slowly, and added to the batch',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<label>Scene <select id="scene">${Object.entries(CV.SCENES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></label>
       ${sl('pf', 'Horizontal flip probability', 0, 1, 0.05, 0.5)}
       ${sl('rot', 'Rotation: ± degrees', 0, 45, 1, 15)}
       ${sl('crop', 'Random crop: min side (% of image)', 40, 100, 5, 70)}
       ${sl('bri', 'Brightness: ± %', 0, 60, 5, 30)}
       ${sl('noi', 'Gaussian noise σ (% of 255)', 0, 25, 1, 8)}
       <label>Dataset size N (images) <input type="number" id="n" value="1000" min="1" step="100"></label>
       ${transport('<button class="btn small" id="clear">Clear batch</button>')}`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), R = 96;
    wire(root, { pf: (v) => Math.round(v * 100) + '%', crop: (v) => v + '%', bri: (v) => v + '%', noi: (v) => v, rot: (v) => '±' + v + '°', sp: (v) => v.toFixed(2) + '×' });
    const mk = (w, h) => { const k = document.createElement('canvas'); k.width = w; k.height = h; return k; };
    let src = null;
    const loadScene = () => {
      const d = CV.scene(q('#scene').value); src = mk(120, 120);
      const g = src.getContext('2d'), im = g.createImageData(120, 120); im.data.set(d); g.putImageData(im, 0, 0);
    };
    loadScene();
    const work = mk(R, R), wg = work.getContext('2d', { willReadFrequently: true });
    const val = (id) => +q('#' + id).value;
    const newSample = () => {
      const c0 = val('crop') / 100, rn = Math.random;
      const cs = lerp(c0, 1, rn()), half = cs / 2;
      const noise = new Float32Array(R * R); for (let i = 0; i < noise.length; i++) noise[i] = AI.randn(rn);
      return { flip: rn() < val('pf'), ang: (rn() * 2 - 1) * val('rot'), c: cs, cx: lerp(half, 1 - half, rn()), cy: lerp(half, 1 - half, rn()), b: (rn() * 2 - 1) * val('bri') / 100, sig: val('noi') / 100, noise, T: 0, snap: null };
    };
    const render = (p, tg, tp) => { // geometry progress tg, photometric progress tp -> work canvas
      wg.fillStyle = '#000'; wg.fillRect(0, 0, R, R); wg.save(); wg.translate(R / 2, R / 2);
      wg.rotate(tg * p.ang * Math.PI / 180); wg.scale(p.flip ? lerp(1, -1, tg) : 1, 1);
      const ct = lerp(1, p.c, tg), cx = lerp(0.5, p.cx, tg), cy = lerp(0.5, p.cy, tg), sw = ct * 120;
      wg.imageSmoothingEnabled = true; wg.drawImage(src, clamp(cx * 120 - sw / 2, 0, 120 - sw), clamp(cy * 120 - sw / 2, 0, 120 - sw), sw, sw, -R / 2, -R / 2, R, R);
      wg.restore();
      if (tp > 0) {
        const im = wg.getImageData(0, 0, R, R), d = im.data, f = 1 + p.b * tp, s = p.sig * 255 * tp;
        for (let i = 0, k = 0; i < R * R; i++, k += 4) { const nz = p.noise[i] * s; d[k] = clamp(d[k] * f + nz, 0, 255); d[k + 1] = clamp(d[k + 1] * f + nz, 0, 255); d[k + 2] = clamp(d[k + 2] * f + nz, 0, 255); }
        wg.putImageData(im, 0, 0);
      }
    };
    let cur = newSample(), samples = [], next = 0, playing = true, stepTo = -1, key = '';
    const PH = [1.2, 2.6, 3.8, 5.0];
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; const nx = PH.find((t) => t > cur.T + 0.01); stepTo = nx === undefined ? PH[3] : nx; };
    q('#reset').onclick = () => { cur = newSample(); };
    q('#clear').onclick = () => { samples = []; next = 0; };
    q('#scene').onchange = () => { loadScene(); samples = []; next = 0; cur = newSample(); };
    const stats = () => {
      const n = Math.max(1, Math.floor(val('n') || 1)), smin = Math.ceil(val('crop') / 100 * 224); let win = 0;
      for (let s = smin; s <= 224; s++) win += (225 - s) * (225 - s);
      const flipF = val('pf') > 0 ? 2 : 1, rotF = 2 * val('rot') + 1, per = win * flipF * rotF;
      q('#out').innerHTML = `<span class="m">crop windows at 224 px (sides ${smin}…224)</span> <b>${big(win)}</b>\n<span class="m">× flip ${flipF} × whole-degree rotations ${rotF}</span> = <b class="o">${big(per)}</b> geometric variants of ONE image\n<span class="m">brightness and noise are continuous: infinitely many more</span>\n<span class="m">dataset of N = ${big(n)} images → up to</span> <b class="g">${big(n * per)}</b> <span class="m">different training inputs (strongly correlated, not independent)</span>`;
    };
    root.addEventListener('input', stats); stats();
    AI.loop(root, (dt) => {
      if (playing) cur.T += dt * val('sp'); else if (stepTo >= 0) { cur.T = Math.min(stepTo, cur.T + dt * 1.2); if (cur.T >= stepTo) stepTo = -1; }
      const C = AI.col(), g = c.ctx, T = cur.T;
      if (T >= PH[3]) { const k = mk(R, R); render(cur, 1, 1); k.getContext('2d').drawImage(work, 0, 0); samples[next % 64] = k; next++; cur = newSample(); return; }
      const tg = smooth((T - PH[0]) / (PH[1] - PH[0])), tp = smooth((T - PH[1]) / (PH[2] - PH[1])), fly = smooth((T - PH[2]) / (PH[3] - PH[2]));
      render(cur, tg, tp);
      g.clearRect(0, 0, c.w, c.h);
      const S = clamp(c.w * 0.36, 150, 250), sx = 14, sy = 34, th = 60, gx = sx + S + 46;
      const cols = Math.max(2, Math.floor((c.w - gx - 8) / (th + 7))), rows = Math.max(1, Math.floor((c.h - 38) / (th + 7))), cap = cols * rows;
      text(g, 'stage: the sample being built', sx, 20, C.muted, '12px sans-serif');
      text(g, `augmented batch (${Math.min(samples.length, cap)} of ${cap} slots)`, gx, 20, C.muted, '12px sans-serif');
      g.imageSmoothingEnabled = false;
      const slot = (i) => ({ x: gx + (i % cols) * (th + 7), y: 34 + Math.floor(i / cols) * (th + 7) });
      const nowSlot = slot(next % cap);
      for (let i = 0; i < Math.min(cap, samples.length); i++) { const s = samples[i], p = slot(i); if (!s || (i === next % cap && fly > 0)) { /* being replaced */ } if (s) { g.drawImage(s, p.x, p.y, th, th); g.strokeStyle = C.border; g.strokeRect(p.x + 0.5, p.y + 0.5, th - 1, th - 1); } }
      for (let i = samples.length; i < cap; i++) { const p = slot(i); g.strokeStyle = C.border; g.setLineDash([3, 3]); g.strokeRect(p.x + 0.5, p.y + 0.5, th - 1, th - 1); g.setLineDash([]); }
      const px = lerp(sx, nowSlot.x, fly), py = lerp(sy, nowSlot.y, fly), ps = lerp(S, th, fly);
      g.drawImage(work, px, py, ps, ps); g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(px, py, ps, ps); g.lineWidth = 1;
      const ph = T < PH[0] ? 0 : T < PH[1] ? 1 : T < PH[2] ? 2 : 3;
      const caps = ['1  original image from the dataset', `2  geometry: flip ${cur.flip ? 'YES' : 'no'}, rotate ${cur.ang >= 0 ? '+' : ''}${cur.ang.toFixed(1)}°, crop ${(cur.c * 100).toFixed(0)}% of the side`, `3  photometry: brightness ×${(1 + cur.b).toFixed(2)}, noise σ = ${(cur.sig * 255).toFixed(0)} grey levels`, '4  the finished sample joins the batch the network will see this step'];
      q('#cap').textContent = caps[ph] + '. Each sample uses fresh random draws — the label (the object class) never changes.';
    }, { fps: 30 });
  };

  /* ====================================================================== finetune */
  AI.labs.finetune = function (root) {
    const B = AIArch.build(AIArch.PRESETS.resnet50.spec).blocks, sum = (a, z) => B.slice(a, z + 1).reduce((s, x) => s + x.params, 0);
    const G = [{ n: 'stem', p: sum(1, 1), H: 112, C: 64 }, { n: 'layer1', p: sum(3, 5), H: 56, C: 256 }, { n: 'layer2', p: sum(6, 9), H: 28, C: 512 }, { n: 'layer3', p: sum(10, 15), H: 14, C: 1024 }, { n: 'layer4', p: sum(16, 18), H: 7, C: 2048 }];
    const BACK = G.reduce((s, x) => s + x.p, 0);
    const { view, ctl } = AI.labShell(root, 'Fine-tuning ResNet-50', 'click a block to freeze/unfreeze it; the gradient only flows back as far as the first trainable block',
      '<canvas id="a" style="height:230px;cursor:pointer"></canvas><div class="caption" id="cap"></div><canvas id="b" style="height:190px;margin-top:6px"></canvas><div class="caption">Validation loss vs epoch. Curve shapes come from a simple illustrative model (not a measurement); the parameter counts and learning rates are exact.</div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<div class="row" id="pre"></div>
       <label>New head classes K <select id="k"><option>2</option><option selected>10</option><option>100</option><option>1000</option></select></label>
       ${sl('lr', 'Head learning rate (10^x)', -4, -1, 0.25, -2)}
       ${sl('gam', 'Layer-wise decay γ per block', 0.1, 1, 0.05, 0.5)}
       ${sl('ds', 'Your dataset size N (images)', 2, 5, 0.1, 3.3)}
       ${sl('shift', 'Domain shift from ImageNet', 0, 1, 0.05, 0.5)}
       ${transport()}
       <div class="readout" id="lora"></div>
       ${sl('d', 'LoRA: matrix size d (d×d)', 256, 4096, 256, 2048)}
       ${sl('r', 'LoRA rank r', 1, 64, 1, 8)}`);
    const q = (s) => AI.$(s, root), ca = AI.canvas(q('#a')), cb = AI.canvas(q('#b'));
    wire(root, { lr: (v) => (10 ** v).toExponential(1), ds: (v) => big(10 ** v), gam: (v) => v.toFixed(2), sp: (v) => v.toFixed(2) + '×', shift: (v) => v.toFixed(2) });
    let frozen = [true, true, true, true, true], playing = true, T = 0, tl = 0;
    const PRE = { 'head only': [1, 1, 1, 1, 1], 'last stage': [1, 1, 1, 1, 0], 'last two stages': [1, 1, 1, 0, 0], 'everything': [0, 0, 0, 0, 0] };
    Object.entries(PRE).forEach(([k, v]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => { frozen = v.map(Boolean); tl = 0; }; q('#pre').appendChild(b); });
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; T += 0.5; };
    q('#reset').onclick = () => { T = 0; tl = 0; };
    const bx = (i, w) => 18 + i * ((w - 36) / 6);
    q('#a').addEventListener('click', (e) => { const r = q('#a').getBoundingClientRect(), x = e.clientX - r.left, bw = (ca.w - 36) / 6, i = Math.floor((x - 18) / bw); if (i >= 0 && i < 5) { frozen[i] = !frozen[i]; tl = 0; } });
    const head = () => 2048 * +q('#k').value + +q('#k').value;
    const model = (rhoP, Pm, N, pre, shift) => { // illustrative validation-loss model
      const bias = 0.55 * shift * Math.pow(1 - rhoP, 1.5), gap = (pre ? 0.5 : 1) * 0.9 * (1 - Math.exp(-Pm * 25 / N));
      return 0.08 + bias + gap;
    };
    const curve = (floor, k, L0, t) => floor + (L0 - floor) * Math.exp(-k * t);
    const lrs = () => { const base = 10 ** +q('#lr').value, g = +q('#gam').value; return G.map((_, i) => base * Math.pow(g, 5 - i)); };
    const upd = () => {
      const H = head(), tr = G.reduce((s, x, i) => s + (frozen[i] ? 0 : x.p), 0) + H, tot = BACK + H, L = lrs();
      q('#out').innerHTML = `<span class="m">block      params      state      learning rate</span>\n` + G.map((x, i) => `${x.n.padEnd(10)} ${fmtN(x.p).padStart(8)}    ${frozen[i] ? '<span class="m">frozen </span>' : '<b class="o">train  </b>'}    ${frozen[i] ? '<span class="m">0</span>' : AI.fmt(L[i], 3)}`).join('\n') + `\nnew head   ${fmtN(H).padStart(8)}    <b class="o">train  </b>    ${AI.fmt(10 ** +q('#lr').value, 3)}\n<span class="m">trainable</span> <b class="o">${big(tr)}</b> <span class="m">of ${big(tot)} =</span> <b>${(tr / tot * 100).toFixed(2)}%</b>  <span class="m">| Adam states for them: ${mb(tr * 8)}  (vs ${mb(tot * 8)} if all trained)</span>`;
      const d = +q('#d').value, r = +q('#r').value;
      q('#lora').innerHTML = `<span class="m">full d×d matrix</span> <b>${big(d * d)}</b>\n<span class="m">LoRA 2·d·r</span> <b class="o">${big(2 * d * r)}</b> <span class="m">(${(2 * d * r / (d * d) * 100).toFixed(2)}% = 2r/d)</span>`;
    };
    root.addEventListener('input', upd); root.addEventListener('click', () => setTimeout(upd, 0)); upd();
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; if (playing) { T += dt * sp; tl += dt * sp; }
      const C = AI.col(), g = ca.ctx, w = ca.w, h = ca.h; g.clearRect(0, 0, w, h);
      const nB = 7, bw = (w - 36) / nB, cy = h * 0.5, first = frozen.indexOf(false), firstT = first < 0 ? 5 : first, cyc = 9, ph = T % cyc;
      const items = G.map((x, i) => ({ n: x.n, H: x.H, C: x.C, p: x.p, tr: !frozen[i] })).concat([{ n: 'new head', H: 1, C: 1, p: head(), tr: true }]);
      items.forEach((it, i) => {
        const ww0 = bw * 0.62, x = 18 + i * (w - 36 - ww0) / 5, hh = it.n === 'new head' ? 30 : 24 + 13 * Math.log2(it.H), ww = bw * 0.62;
        it.x = x; it.w = ww; it.hh = hh;
        g.fillStyle = it.tr ? C.orange : C.blue; g.globalAlpha = it.tr ? 0.9 : 0.35; g.fillRect(x, cy - hh / 2, ww, hh); g.globalAlpha = 1;
        g.strokeStyle = it.tr ? C.orange : C.border; g.lineWidth = it.tr ? 2 : 1; g.strokeRect(x, cy - hh / 2, ww, hh); g.lineWidth = 1;
        text(g, it.n, x + ww / 2, h - 44, C.fg, '600 11px sans-serif', 'center');
        text(g, fmtN(it.p), x + ww / 2, h - 30, C.orange, '11px ui-monospace,monospace', 'center');
        text(g, it.tr ? 'TRAIN' : 'FROZEN', x + ww / 2, h - 16, it.tr ? C.orange : C.muted, '10px sans-serif', 'center');
      });
      g.strokeStyle = C.border; for (let i = 0; i < 5; i++) { g.beginPath(); g.moveTo(items[i].x + items[i].w, cy); g.lineTo(items[i + 1].x, cy); g.stroke(); }
      let capTxt;
      const fwd = clamp(ph / 3, 0, 1), bwd = clamp((ph - 4) / 3.5, 0, 1);
      const xAt = (u) => { const i = Math.min(4, Math.floor(u)), f = u - i; return lerp(items[i].x + items[i].w / 2, items[i + 1].x + items[i + 1].w / 2, f); };
      if (ph < 3) { capTxt = '1  forward pass: the image flows through every block, frozen or not (frozen blocks are used, just not updated).'; g.fillStyle = C.fg; g.beginPath(); g.arc(xAt(fwd * 5), cy, 6, 0, 7); g.fill(); }
      else if (ph < 4) capTxt = '2  the loss is measured at the new head.';
      else if (ph < 7.5) {
        capTxt = first < 0 ? '3  backward pass: gradient reaches only the new head; every backbone block is frozen, so backprop stops right there.' : `3  backward pass: gradients flow back and stop at ${G[firstT].n}, the first trainable block; blocks before it need no gradient at all.`;
        const stopU = first < 0 ? 5 : firstT; const u = 5 - bwd * (5 - stopU); g.fillStyle = C.orange; g.beginPath(); g.arc(xAt(u), cy - 20, 6, 0, 7); g.fill();
        if (first > 0) text(g, 'no gradient here', items[0].x, cy - 52, C.muted, '11px sans-serif');
      } else capTxt = '4  update: only the orange blocks change, each with its own learning rate.';
      q('#cap').textContent = capTxt;
      // loss comparison
      const g2 = cb.ctx, w2 = cb.w, h2 = cb.h; g2.clearRect(0, 0, w2, h2);
      const N = 10 ** +q('#ds').value, shift = +q('#shift').value, H = head(), trB = G.reduce((s, x, i) => s + (frozen[i] ? 0 : x.p), 0);
      const confs = [
        { n: 'head only', col: C.blue, f: model(0, H / 1e6, N, true, shift), k: 0.5, L0: 2.3 },
        { n: 'your selection', col: C.orange, f: model(trB / BACK, (trB + H) / 1e6, N, true, shift), k: 0.32 - 0.12 * (trB / BACK), L0: 2.3 },
        { n: 'from scratch', col: C.red, f: model(1, (BACK + H) / 1e6, N, false, 0), k: 0.1, L0: 2.6 },
      ];
      const v = AI.view(cb, 0, 30, 0, 2.8, { l: 40, r: 12, t: 10, b: 24 }); AI.axes(g2, v);
      text(g2, 'epoch', v.pad.l + v.pw / 2, h2 - 2, C.muted, '11px sans-serif', 'center');
      const te = Math.min(30, (tl * 3) % 40);
      confs.forEach((cf, i) => {
        g2.strokeStyle = cf.col; g2.lineWidth = 2.5; g2.beginPath();
        for (let e = 0; e <= te; e += 0.25) { const y = curve(cf.f, cf.k, cf.L0, e); e === 0 ? g2.moveTo(v.X(e), v.Y(y)) : g2.lineTo(v.X(e), v.Y(y)); }
        g2.stroke(); g2.lineWidth = 1;
        text(g2, `${cf.n}: ${curve(cf.f, cf.k, cf.L0, te).toFixed(2)}`, v.pad.l + 8, v.pad.t + 14 + i * 14, cf.col, '11px ui-monospace,monospace');
      });
    }, { fps: 30 });
  };

  /* ====================================================================== pipesim */
  AI.labs.pipesim = function (root) {
    const DEF = [['decode', 0, 4, 1, 1, 0], ['resize', 0, 1, 1, 1, 0], ['normalise', 0, 0.5, 1, 1, 0], ['inference', 6, 2, 1, 1, 10], ['NMS', 0, 1, 1, 1, 0], ['store', 0, 1, 1, 1, 0]];
    const S = DEF.map((d) => ({ name: d[0], fixed: d[1], per: d[2], workers: d[3], B: d[4], wait: d[5], q: [], srv: [], util: 0 }));
    const { view, ctl } = AI.labShell(root, 'End-to-end pipeline simulator', 'frames stream through stages; edit any latency, batch size or worker count and watch the bottleneck',
      `<canvas style="height:300px"></canvas><div class="caption" id="cap"></div>
       <table class="tbl" style="font-size:12px;margin-top:8px"><tr><th>stage</th><th>fixed ms</th><th>ms / item</th><th>workers</th><th>max batch</th><th>max wait ms</th><th>capacity /s</th></tr>
       ${S.map((s, i) => `<tr><td><b>${s.name}</b></td>${['fixed', 'per', 'workers', 'B', 'wait'].map((f) => `<td><input type="number" data-i="${i}" data-f="${f}" value="${s[f]}" min="${f === 'workers' || f === 'B' ? 1 : 0}" step="${f === 'workers' || f === 'B' ? 1 : 0.5}" style="width:64px;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:5px;padding:2px 4px"></td>`).join('')}<td id="cap${i}" class="mono"></td></tr>`).join('')}</table>`,
      `${sl('fps', 'Camera rate (frames/s)', 10, 400, 5, 110)}
       <div class="row" id="pre"></div>
       ${transport()}
       <div class="readout" id="out"></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { fps: (v) => v + ' /s', sp: (v) => (10 * v).toFixed(1) + ' sim-ms per second' });
    const QLIM = 40;
    let t = 0, nextArr = 0, frames = [], nid = 0, dropped = 0, done = 0, lat = [], comp = [], inSys = 0, playing = true, simAcc = 0, lastKey = '', capT = 0;
    S.forEach((s) => { s.srv = [{ batch: null, rem: 0 }]; });
    const setP = (arr) => { arr.forEach((a, i) => { Object.assign(S[i], { fixed: a[0], per: a[1], workers: a[2], B: a[3], wait: a[4] }); }); AI.$$('input[data-f]', root).forEach((inp) => (inp.value = S[+inp.dataset.i][inp.dataset.f])); reset(); };
    [['baseline (B=1)', [[0, 4, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0.5, 1, 1, 0], [6, 2, 1, 1, 10], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]],
      ['batch 8, wait 10 ms', [[0, 4, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0.5, 1, 1, 0], [6, 2, 1, 8, 10], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]],
      ['slow decode (12 ms)', [[0, 12, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0.5, 1, 1, 0], [6, 2, 1, 1, 10], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]],
      ['3 decode workers', [[0, 12, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0.5, 1, 1, 0], [6, 2, 1, 1, 10], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]]].forEach(([n, a]) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = n; b.onclick = () => setP(a); q('#pre').appendChild(b); });
    root.addEventListener('input', (e) => { const i = e.target.dataset && e.target.dataset.i; if (i === undefined) return; const f = e.target.dataset.f, v = +e.target.value; if (v >= 0) S[+i][f] = (f === 'workers' || f === 'B') ? Math.max(1, Math.round(v)) : v; while (S[+i].srv.length < S[+i].workers) S[+i].srv.push({ batch: null, rem: 0 }); capUpd(); });
    const capOf = (s) => s.workers * s.B / (s.fixed + s.per * s.B) * 1000;
    const capUpd = () => S.forEach((s, i) => { q('#cap' + i).textContent = AI.fmt(capOf(s), 4); });
    const reset = () => { t = 0; nextArr = 0; frames = []; dropped = 0; done = 0; lat = []; comp = []; inSys = 0; S.forEach((s) => { s.q = []; s.srv = Array.from({ length: Math.max(1, s.workers) }, () => ({ batch: null, rem: 0 })); s.util = 0; }); };
    capUpd();
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; for (let i = 0; i < 8; i++) step(0.25); };
    q('#reset').onclick = reset;
    const finish = (f) => { const L = t - f.t0; lat.push(L); if (lat.length > 40) lat.shift(); comp.push(t); done++; inSys--; f.dead = true; f.fade = 1; };
    function step(h) {
      t += h;
      while (nextArr <= t) {
        const f = { id: ++nid, t0: nextArr, x: 20, y: 150, enq: nextArr, dead: false };
        if (S[0].q.length >= QLIM) { dropped++; } else { frames.push(f); S[0].q.push(f); inSys++; }
        nextArr += 1000 / +q('#fps').value;
      }
      for (let i = S.length - 1; i >= 0; i--) {
        const s = S[i];
        s.srv.forEach((sv) => { if (sv.batch) { sv.rem -= h; if (sv.rem <= 0) { const b = sv.batch; sv.batch = null; if (i < S.length - 1) b.forEach((f) => { f.enq = t; S[i + 1].q.push(f); }); else b.forEach(finish); } } });
        for (let k = 0; k < Math.min(s.workers, s.srv.length); k++) {
          const sv = s.srv[k];
          if (!sv.batch && s.q.length && (s.q.length >= s.B || t - s.q[0].enq >= s.wait)) { const n = Math.min(s.B, s.q.length); sv.batch = s.q.splice(0, n); sv.rem = s.fixed + s.per * n; }
        }
        const busy = s.srv.filter((v, k) => v.batch && k < s.workers).length / Math.max(1, s.workers);
        s.util += (busy - s.util) * (1 - Math.exp(-h / 500));
      }
      while (comp.length && comp[0] < t - 1000) comp.shift();
    }
    let acc = 0;
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value;
      if (playing) { acc += dt * 40 * sp; let n = Math.min(60, Math.floor(acc)); acc -= n; while (n-- > 0) step(0.25); }
      const C = AI.col(), g = c.ctx, W = c.w, Hh = c.h; g.clearRect(0, 0, W, Hh);
      const n = S.length, x0 = 70, slotW = (W - x0 - 14) / n, bw = slotW * 0.46, bh = 84, by = 90;
      const caps = S.map(capOf), minCap = Math.min(...caps), bn = caps.indexOf(minCap), lam = +q('#fps').value;
      // camera
      g.fillStyle = C.bg2; g.strokeStyle = C.border; g.beginPath(); g.roundRect(8, by + 20, 44, 44, 8); g.fill(); g.stroke(); text(g, 'camera', 30, by + 46, C.fg, '10px sans-serif', 'center');
      text(g, `t = ${t.toFixed(1)} ms`, W - 12, 16, C.muted, '12px ui-monospace,monospace', 'right');
      const pos = S.map((s, i) => ({ x: x0 + i * slotW + slotW * 0.5, y: by }));
      S.forEach((s, i) => {
        const bx = pos[i].x, hot = i === bn && lam > minCap * 0.8;
        g.fillStyle = C.bg2; g.strokeStyle = hot ? C.red : C.border; g.lineWidth = hot ? 3 : 1; g.beginPath(); g.roundRect(bx, by, bw, bh, 8); g.fill(); g.stroke(); g.lineWidth = 1;
        text(g, s.name, bx + bw / 2, by - 8, hot ? C.red : C.fg, '600 12px sans-serif', 'center');
        g.fillStyle = C.border; g.fillRect(bx, by + bh + 8, bw, 6); g.fillStyle = s.util > 0.9 ? C.red : s.util > 0.6 ? C.yellow : C.green; g.fillRect(bx, by + bh + 8, bw * clamp(s.util, 0, 1), 6);
        text(g, `util ${(s.util * 100).toFixed(0)}%`, bx + bw / 2, by + bh + 28, C.muted, '10.5px ui-monospace,monospace', 'center');
        text(g, `queue ${s.q.length}`, bx - 4, by + bh + 44, s.q.length > 8 ? C.red : C.muted, '10.5px ui-monospace,monospace');
        if (hot) text(g, 'BOTTLENECK', bx + bw / 2, by + bh + 60, C.red, '600 10px sans-serif', 'center');
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(i ? pos[i - 1].x + bw : 52, by + bh / 2); g.lineTo(bx - 2, by + bh / 2); g.stroke();
      });
      // targets for frames
      S.forEach((s, i) => {
        s.q.forEach((f, j) => { const col = j % 4, row = Math.floor(j / 4); f.tx = pos[i].x - 9 - col * 12; f.ty = by + 10 + Math.min(row, 6) * 12; });
        let k = 0; s.srv.forEach((sv) => { if (sv.batch) sv.batch.forEach((f) => { f.tx = pos[i].x + 14 + (k % 5) * 12; f.ty = by + 18 + Math.floor(k / 5) * 12; k++; }); });
      });
      const e = ease(dt, 4 + 2 * sp);
      frames.forEach((f) => { if (f.dead) { f.tx = W - 8; f.ty = by + bh / 2; f.fade -= dt * 0.6; } f.x += ((f.tx ?? f.x) - f.x) * e; f.y += ((f.ty ?? f.y) - f.y) * e; });
      frames = frames.filter((f) => !f.dead || f.fade > 0);
      const hues = [C.blue, C.green, C.orange, C.purple, C.teal, C.pink];
      frames.forEach((f) => { g.globalAlpha = f.dead ? Math.max(0, f.fade) : 1; g.fillStyle = hues[f.id % 6]; g.beginPath(); g.roundRect(f.x - 5, f.y - 5, 10, 10, 3); g.fill(); });
      g.globalAlpha = 1;
      // metrics
      const X = comp.length / (Math.min(t, 1000) / 1000 || 1), W_ = lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : 0, p95 = lat.length ? [...lat].sort((a, b) => a - b)[Math.floor(lat.length * 0.95)] : 0;
      const sat = lam > minCap;
      const key = [Math.round(t / 20), dropped, bn, lam].join();
      if (key !== lastKey) {
        lastKey = key;
        q('#out').innerHTML = `<span class="m">arrival λ</span> <b>${lam}</b> /s\n<span class="m">throughput X</span> <b class="g">${X.toFixed(0)}</b> /s\n<span class="m">mean latency W</span> <b class="o">${W_.toFixed(1)}</b> ms (p95 ${p95.toFixed(1)})\n<span class="m">frames in system L</span> <b>${inSys}</b>\n<span class="m">Little: X·W</span> <b>${(X * W_ / 1000).toFixed(1)}</b>\n<span class="m">pipeline max</span> <b class="${sat ? 'r' : 'b'}">${minCap.toFixed(0)}</b> /s (${S[bn].name})\n<span class="m">dropped</span> <b class="${dropped ? 'r' : ''}">${dropped}</b>`;
        q('#cap').innerHTML = sat ? `<b>Overloaded.</b> The camera sends ${lam} frames/s but <b>${S[bn].name}</b> can only finish ${minCap.toFixed(0)}/s, so its queue grows, latency climbs and frames are dropped (queue limit ${QLIM}). Speed up or parallelise that stage, or lower the rate.` : `Stable: every stage can keep up. Throughput equals the camera rate; latency is the sum of service times plus waiting. The stage with the lowest capacity (<b>${S[bn].name}</b>, ${minCap.toFixed(0)}/s) sets the ceiling.`;
      }
    }, { fps: 30 });
  };

  /* ====================================================================== quant */
  AI.labs.quant = function (root) {
    const rng = AI.rng(11), W0 = []; for (let i = 0; i < 400; i++) W0.push(clamp(AI.randn(rng) * 0.4, -1.4, 1.4));
    const { view, ctl } = AI.labShell(root, 'Quantization: snapping floats to integer levels', 'each float weight slides to the nearest of 2^bits levels; the gap is the quantization error',
      '<canvas style="height:400px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('bits', 'Bit width', 2, 8, 1, 4)}
       <label>Scheme <select id="mode"><option value="asym">asymmetric (scale + zero point)</option><option value="sym">symmetric (zero point 0)</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="out1"> add one outlier weight (3.2)</label>
       ${transport()}`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { bits: (v) => v + ' bit (' + 2 ** v + ' levels)', sp: (v) => v.toFixed(2) + '×' });
    let T = 0, playing = true, shownCap = '';
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; T += 0.45; };
    q('#reset').onclick = () => { T = 0; };
    const Q = (bits, mode, W) => {
      let mn = Infinity, mx = -Infinity, am = 0; W.forEach((w) => { mn = Math.min(mn, w); mx = Math.max(mx, w); am = Math.max(am, Math.abs(w)); });
      const n = 2 ** bits; let s, z, lo, hi;
      if (mode === 'asym') { s = (mx - mn) / (n - 1); z = Math.round(-mn / s); lo = 0; hi = n - 1; } else { hi = 2 ** (bits - 1) - 1; lo = -hi; s = am / hi; z = 0; }
      const qf = (w) => clamp(Math.round(w / s) + z, lo, hi);
      return { s, z, lo, hi, mn, mx, q: qf, dq: (w) => (qf(w) - z) * s };
    };
    const stat = (bits, mode, W) => { const Qz = Q(bits, mode, W); let se = 0, sa = 0, m = 0, v = 0; W.forEach((w) => { m += w; }); m /= W.length; W.forEach((w) => { const e = Qz.dq(w) - w; se += e * e; sa += Math.abs(e); v += (w - m) ** 2; }); return { Q: Qz, mse: se / W.length, mae: sa / W.length, nsr: se / v }; };
    AI.loop(root, (dt) => {
      if (playing) T += dt * +q('#sp').value;
      const out = q('#out1').checked, W = out ? W0.map((w, i) => (i === 2 ? 3.2 : w)) : W0, bits = +q('#bits').value, mode = q('#mode').value;
      const st = stat(bits, mode, W), Qz = st.Q, C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const lo = -1.6, hi = out ? 3.5 : 1.6, v = AI.view(c, lo, hi, 0, 1, { l: 24, r: 24, t: 8, b: 8 }), ny = 120, X = v.X;
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(v.pad.l, ny); g.lineTo(w - 24, ny); g.stroke();
      // levels
      const levels = []; for (let k = Qz.lo; k <= Qz.hi; k++) levels.push((k - Qz.z) * Qz.s);
      g.strokeStyle = C.teal; levels.forEach((lv, i) => { g.globalAlpha = levels.length > 64 ? 0.5 : 1; g.beginPath(); g.moveTo(X(lv) + 0.5, ny - 9); g.lineTo(X(lv) + 0.5, ny + 9); g.stroke(); }); g.globalAlpha = 1;
      if (levels.length <= 32) levels.forEach((lv, i) => { if (i % (levels.length > 16 ? 2 : 1) === 0) text(g, String(Qz.lo + i), X(lv), ny + 24, C.teal, '10px ui-monospace,monospace', 'center'); });
      for (let k = Math.ceil(lo); k <= hi; k++) text(g, String(k), X(k), ny - 16 - 60, C.muted, '10px ui-monospace,monospace', 'center');
      text(g, 'float value  (teal ticks = the integer levels the hardware can store; labels = integer q)', 24, 18, C.muted, '11px sans-serif');
      // weights
      let active = -1; const dis = 16, step = 0.45;
      for (let i = 0; i < dis; i++) {
        const e = smooth((T - i * step - 0.8) / 1.1); if (e > 0 && e < 1 && active < 0) active = i;
        const wv = W[i], dq = Qz.dq(wv), xv = lerp(wv, dq, e), yy = ny - 28 - (i % 4) * 14;
        g.strokeStyle = C.border; g.beginPath(); g.moveTo(X(xv), yy); g.lineTo(X(xv), ny); g.stroke();
        g.fillStyle = e >= 1 ? C.green : C.blue; g.beginPath(); g.arc(X(xv), yy, i === active ? 6 : 4.5, 0, 7); g.fill();
        if (e > 0.5) { g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(X(wv), ny + 40); g.lineTo(X(dq), ny + 40); g.stroke(); g.lineWidth = 1; }
      }
      if (T > dis * step + 4.5) T = 0;
      let capT = 'Press play: the 16 sample weights snap one by one to the nearest level.';
      if (active >= 0) { const wv = W[active], qq = Qz.q(wv); capT = `weight ${active + 1}: w = ${wv.toFixed(4)}  →  w/s + z = ${(wv / Qz.s + Qz.z).toFixed(2)}  →  q = ${qq}  →  ŵ = (q − z)·s = ${Qz.dq(wv).toFixed(4)}   (error ${(Qz.dq(wv) - wv >= 0 ? '+' : '') + (Qz.dq(wv) - wv).toFixed(4)}, bound s/2 = ${(Qz.s / 2).toFixed(4)})`; }
      if (capT !== shownCap) { shownCap = capT; q('#cap').textContent = capT; }
      // histogram of errors
      const px = 24, py = 190, pw = Math.min(w * 0.46, 330), ph = h - py - 28, bins = new Array(10).fill(0);
      W.forEach((x) => { const er = (Qz.dq(x) - x) / Qz.s; bins[clamp(Math.floor((er + 0.5) * 10), 0, 9)]++; });
      const mxb = Math.max(...bins, 1);
      text(g, 'error histogram over 400 weights (units of s)', px, py - 8, C.muted, '11px sans-serif');
      bins.forEach((b, i) => { g.fillStyle = C.orange; g.fillRect(px + i * pw / 10 + 2, py + ph - b / mxb * ph, pw / 10 - 4, b / mxb * ph); });
      g.strokeStyle = C.border; g.strokeRect(px, py, pw, ph); text(g, '−0.5', px, py + ph + 14, C.muted, '10px monospace'); text(g, '+0.5', px + pw, py + ph + 14, C.muted, '10px monospace', 'right'); text(g, '0', px + pw / 2, py + ph + 14, C.muted, '10px monospace', 'center');
      // trend
      const tx = px + pw + 40, tw = w - tx - 24; if (tw > 120) {
        text(g, 'illustrative top-1 accuracy vs bit width (ResNet-50, baseline 76.1%)', tx, py - 8, C.muted, '11px sans-serif');
        const vv = AI.view({ w: tw + 40, h: ph + 24 }, 1.5, 8.5, 0, 80, { l: 40, r: 0, t: 0, b: 24 });
        g.save(); g.translate(tx - 40, py); g.strokeStyle = C.border; g.strokeRect(40, 0, tw, ph);
        const pts = []; for (let b = 2; b <= 8; b++) { const s2 = stat(b, mode, W); pts.push([b, Math.max(0, 76.1 - Math.min(76.1, 600 * s2.nsr))]); }
        const Xx = (b) => 40 + (b - 1.5) / 7 * tw, Yy = (a) => ph - a / 80 * ph;
        g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(Xx(p[0]), Yy(p[1])) : g.moveTo(Xx(p[0]), Yy(p[1])))); g.stroke(); g.lineWidth = 1;
        pts.forEach((p) => { g.fillStyle = p[0] === bits ? C.orange : C.purple; g.beginPath(); g.arc(Xx(p[0]), Yy(p[1]), p[0] === bits ? 6 : 3.5, 0, 7); g.fill(); text(g, String(p[0]), Xx(p[0]), ph + 14, C.muted, '10px monospace', 'center'); });
        text(g, '80', 34, 10, C.muted, '10px monospace', 'right'); text(g, '0', 34, ph, C.muted, '10px monospace', 'right');
        g.restore();
      }
      q('#out').innerHTML = `<span class="m">range</span> [${Qz.mn.toFixed(3)}, ${Qz.mx.toFixed(3)}]   <span class="m">scale s</span> <b class="b">${Qz.s.toFixed(5)}</b>   <span class="m">zero point z</span> <b class="b">${Qz.z}</b>   <span class="m">levels</span> ${2 ** bits}\n<span class="m">max error ≤ s/2</span> <b>${(Qz.s / 2).toFixed(5)}</b>   <span class="m">mean |error|</span> <b class="o">${st.mae.toFixed(5)}</b>   <span class="m">noise/signal power</span> <b>${st.nsr.toExponential(2)}</b>   <span class="m">size vs float32</span> <b class="g">${(32 / bits).toFixed(1)}× smaller</b>`;
    }, { fps: 30 });
  };

  /* ====================================================================== prune */
  AI.labs.prune = function (root) {
    const N = 12, rng = AI.rng(5), Wm = [], rs = [];
    for (let i = 0; i < N; i++) { const s = Math.exp(AI.randn(rng) * 0.55) * 0.5; rs.push(s); Wm.push(Array.from({ length: N }, () => AI.randn(rng) * s)); }
    const X = Array.from({ length: 24 }, () => Array.from({ length: N }, () => AI.randn(rng)));
    const flat = []; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) flat.push({ i, j, a: Math.abs(Wm[i][j]) });
    const order = [...flat].sort((a, b) => a.a - b.a), rowN = Wm.map((r) => Math.hypot(...r)), rowOrd = rowN.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const maskOf = (mode, sp) => {
      const m = Wm.map((r) => r.map(() => 1));
      if (mode === 'unstr') order.slice(0, Math.round(sp * N * N)).forEach((o) => (m[o.i][o.j] = 0)); else rowOrd.slice(0, Math.round(sp * N)).forEach((i) => m[i].fill(0));
      return m;
    };
    const errOf = (m) => { let num = 0, den = 0; X.forEach((x) => { for (let i = 0; i < N; i++) { let y = 0, yp = 0; for (let j = 0; j < N; j++) { y += Wm[i][j] * x[j]; yp += Wm[i][j] * m[i][j] * x[j]; } num += (y - yp) ** 2; den += y * y; } }); return Math.sqrt(num / den); };
    const curves = { unstr: [], str: [] }; for (let p = 0; p <= 95; p += 5) { curves.unstr.push([p, errOf(maskOf('unstr', p / 100))]); curves.str.push([p, errOf(maskOf('str', p / 100))]); }
    const { view, ctl } = AI.labShell(root, 'Pruning a weight matrix', 'remove the smallest weights (or whole neurons) and measure how much the layer output changes',
      '<canvas style="height:340px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('spar', 'Sparsity to remove', 0, 95, 1, 40)}
       <label>Method <select id="mode"><option value="unstr">unstructured: smallest |w| anywhere</option><option value="str">structured: whole rows (neurons) by L2 norm</option></select></label>
       ${transport('<button class="btn small" id="grad">Prune gradually</button>')}`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { spar: (v) => v + '%', sp: (v) => v.toFixed(2) + '×' });
    const al = Array.from({ length: N * N }, () => 1); let grad = false, playing = true, lastKey = '', rows = 0;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; q('#spar').value = Math.min(95, +q('#spar').value + 5); q('#sparv').textContent = q('#spar').value + '%'; };
    q('#reset').onclick = () => { grad = false; q('#spar').value = 0; q('#sparv').textContent = '0%'; };
    q('#grad').onclick = () => { grad = true; q('#spar').value = 0; q('#sparv').textContent = '0%'; };
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value; let s = +q('#spar').value;
      if (grad && playing) { s = Math.min(95, s + dt * 4 * sp); q('#spar').value = s; q('#sparv').textContent = Math.round(s) + '%'; if (s >= 95) grad = false; }
      const mode = q('#mode').value, m = maskOf(mode, s / 100), C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      const cs = Math.min(24, (h - 40) / N, w * 0.4 / N), mx = 16, my = 28; let kept = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const k = i * N + j; al[k] += ((m[i][j] ? 1 : 0.06) - al[k]) * ease(dt, 5); kept += m[i][j]; }
      g.clearRect(0, 0, w, h);
      text(g, `${N}×${N} weight matrix (blue +, orange −; brightness = |w|)`, mx, 16, C.muted, '12px sans-serif');
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const k = i * N + j, v = Wm[i][j], a = clamp(Math.abs(v) / 1.2, 0.08, 1); g.globalAlpha = al[k]; g.fillStyle = v >= 0 ? C.blue : C.orange; g.globalAlpha = al[k] * (0.25 + 0.75 * a); g.fillRect(mx + j * cs, my + i * cs, cs - 1, cs - 1); }
      g.globalAlpha = 1;
      // sorted magnitudes
      const rx = mx + N * cs + 40, rw = w - rx - 16; if (rw > 140) {
        const ph1 = 110, th = order[Math.min(N * N - 1, Math.round(s / 100 * N * N))].a;
        text(g, 'all weights sorted by |w|; line = threshold', rx, 16, C.muted, '12px sans-serif');
        order.forEach((o, k) => { g.fillStyle = (mode === 'unstr' ? k < Math.round(s / 100 * N * N) : !m[o.i][o.j]) ? C.red : C.green; const bh = o.a / 2 * ph1; g.fillRect(rx + k / (N * N) * rw, my + ph1 - bh, Math.max(1, rw / (N * N) - 0.5), bh); });
        if (mode === 'unstr') { g.strokeStyle = C.fg; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(rx, my + ph1 - th / 2 * ph1); g.lineTo(rx + rw, my + ph1 - th / 2 * ph1); g.stroke(); g.setLineDash([]); }
        const ey = my + ph1 + 50, eh = h - ey - 26; text(g, 'relative output error vs sparsity', rx, ey - 8, C.muted, '12px sans-serif');
        g.strokeStyle = C.border; g.strokeRect(rx, ey, rw, eh);
        [['unstr', C.blue], ['str', C.orange]].forEach(([k, col]) => { g.strokeStyle = col; g.lineWidth = k === mode ? 3 : 1.2; g.beginPath(); curves[k].forEach((p, i) => { const x = rx + p[0] / 95 * rw, y = ey + eh - Math.min(1, p[1]) * eh; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); });
        g.lineWidth = 1; const er = errOf(m); g.fillStyle = C.fg; g.beginPath(); g.arc(rx + s / 95 * rw, ey + eh - Math.min(1, er) * eh, 5, 0, 7); g.fill();
        text(g, 'unstructured', rx + 6, ey + 14, C.blue, '11px sans-serif'); text(g, 'structured (rows)', rx + 6, ey + 28, C.orange, '11px sans-serif');
        text(g, '0%', rx, ey + eh + 14, C.muted, '10px monospace'); text(g, '95%', rx + rw, ey + eh + 14, C.muted, '10px monospace', 'right');
        const key = [Math.round(s), mode].join();
        if (key !== lastKey) { lastKey = key; const keptN = mode === 'unstr' ? N * N - Math.round(s / 100 * N * N) : (N - Math.round(s / 100 * N)) * N;
          q('#out').innerHTML = `<span class="m">kept weights</span> <b>${keptN}</b> of ${N * N}  <span class="m">(sparsity ${(100 - keptN / (N * N) * 100).toFixed(1)}%)</span>   <span class="m">threshold |w| ≤</span> <b>${th.toFixed(3)}</b>   <span class="m">output error</span> <b class="o">${(er * 100).toFixed(1)}%</b>\n<span class="m">${mode === 'unstr' ? 'irregular holes: the matrix is still 12×12, so a dense GPU kernel gains nothing unless it supports sparsity (e.g. 2:4 patterns)' : 'whole rows gone: this is a genuinely smaller 12×' + (N - Math.round(s / 100 * N)) + ' dense matrix and runs faster on any hardware'}</span>`;
          q('#cap').textContent = 'Slide the sparsity or press "Prune gradually": the smallest weights fade first; the dot on the right plot shows what that costs.'; }
      }
    }, { fps: 30 });
  };

  /* ====================================================================== distill */
  AI.labs.distill = function (root) {
    const NM = ['cat', 'dog', 'fox'];
    const { view, ctl } = AI.labShell(root, 'Knowledge distillation', 'a high temperature reveals the teacher\'s "dark knowledge"; the student is trained to match it',
      '<canvas style="height:300px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('T', 'Temperature T', 1, 10, 0.5, 4)}
       ${sl('al', 'Weight α on the soft (teacher) loss', 0, 1, 0.05, 0.5)}
       ${NM.map((n, i) => sl('t' + i, 'Teacher logit: ' + n, -6, 8, 0.5, [5, 2, -1][i])).join('')}
       ${NM.map((n, i) => sl('s' + i, 'Student logit: ' + n, -6, 8, 0.05, [3, 2.5, -0.5][i])).join('')}
       <div class="row"><button class="btn small primary" id="play">Train student</button><button class="btn small" id="step">Step</button><button class="btn small" id="reset">Reset student</button></div>${sl('sp', 'Speed', 0.25, 3, 0.05, 1)}`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { T: (v) => v.toFixed(1), al: (v) => v.toFixed(2), sp: (v) => v.toFixed(2) + '×' });
    const sm = (z, T) => { const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / T)), S = e.reduce((a, b) => a + b); return e.map((v) => v / S); };
    const zt = () => NM.map((_, i) => +q('#t' + i).value), zs = () => NM.map((_, i) => +q('#s' + i).value);
    let training = false, disp = { pt: [0, 0, 0], ps: [0, 0, 0] }, tick = AI.ticker(), lastKey = '', iters = 0;
    const doStep = () => {
      const T = +q('#T').value, a = +q('#al').value, s = zs(), pT = sm(zt(), T), pS = sm(s, T), p1 = sm(s, 1);
      NM.forEach((_, i) => { const gr = a * T * (pS[i] - pT[i]) + (1 - a) * (p1[i] - (i === 0 ? 1 : 0)); q('#s' + i).value = clamp(s[i] - 0.3 * gr, -6, 8); q('#s' + i + 'v').textContent = AI.fmt(+q('#s' + i).value, 3); });
      iters++;
    };
    q('#play').onclick = () => { training = !training; q('#play').textContent = training ? 'Pause training' : 'Train student'; };
    q('#step').onclick = () => { training = false; q('#play').textContent = 'Train student'; doStep(); };
    q('#reset').onclick = () => { [3, 2.5, -0.5].forEach((v, i) => { q('#s' + i).value = v; q('#s' + i + 'v').textContent = v; }); training = false; iters = 0; q('#play').textContent = 'Train student'; };
    AI.loop(root, (dt) => {
      if (training) { let n = tick(dt, 4 * +q('#sp').value); while (n-- > 0) doStep(); }
      const T = +q('#T').value, a = +q('#al').value, zT = zt(), zS = zs(), pT = sm(zT, T), pS = sm(zS, T), t1 = sm(zT, 1);
      const e = ease(dt, 6); let moving = false;
      for (let i = 0; i < 3; i++) { const dT = pT[i] - disp.pt[i], dS = pS[i] - disp.ps[i]; if (Math.abs(dT) > 1e-4 || Math.abs(dS) > 1e-4) moving = true; disp.pt[i] += dT * e; disp.ps[i] += dS * e; }
      const kl = pT.reduce((s, p, i) => s + p * Math.log(p / pS[i]), 0), ce = -Math.log(sm(zS, 1)[0]), tot = a * T * T * kl + (1 - a) * ce;
      const key = [pT.join(), pS.join(), a, T].join();
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      if (!moving && key === lastKey) return; lastKey = key; g.clearRect(0, 0, w, h);
      const px = 50, pw = w - 70, ph = h - 70, py = 22, gw = pw / 3;
      g.strokeStyle = C.border; g.strokeRect(px, py, pw, ph);
      [0, 0.5, 1].forEach((v) => { const y = py + ph - v * ph; g.beginPath(); g.moveTo(px, y); g.lineTo(px + pw, y); g.stroke(); text(g, String(v), px - 6, y + 4, C.muted, '10px monospace', 'right'); });
      NM.forEach((n, i) => {
        const x = px + i * gw + gw * 0.16, bw = gw * 0.28;
        g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.strokeRect(x + 0.5, py + ph - t1[i] * ph, bw - 1, t1[i] * ph); g.setLineDash([]);
        g.fillStyle = C.teal; g.fillRect(x, py + ph - disp.pt[i] * ph, bw, disp.pt[i] * ph);
        g.fillStyle = C.orange; g.fillRect(x + bw + 6, py + ph - disp.ps[i] * ph, bw, disp.ps[i] * ph);
        text(g, pT[i].toFixed(3), x + bw / 2, py + ph - disp.pt[i] * ph - 5, C.teal, '11px monospace', 'center'); text(g, pS[i].toFixed(3), x + bw * 1.5 + 6, py + ph - disp.ps[i] * ph - 5, C.orange, '11px monospace', 'center');
        text(g, n, x + bw + 3, py + ph + 18, C.fg, '600 12px sans-serif', 'center');
      });
      text(g, 'teal = teacher softmax(z/T)   orange = student softmax(z/T)   dashed = teacher at T=1', px, h - 8, C.muted, '11px sans-serif');
      q('#out').innerHTML = `<span class="m">teacher T=1</span> ${t1.map((v) => v.toFixed(3)).join(' ')}   <span class="m">teacher T=${T}</span> ${pT.map((v) => v.toFixed(3)).join(' ')}\n<span class="m">soft loss T²·KL(teacher‖student)</span> <b class="b">${(T * T * kl).toFixed(4)}</b>   <span class="m">hard CE (label = cat)</span> <b class="o">${ce.toFixed(4)}</b>   <span class="m">total</span> <b class="g">${tot.toFixed(4)}</b>   <span class="m">steps</span> ${iters}`;
      q('#cap').textContent = T < 2 ? 'At T=1 the teacher is almost one-hot: little to learn beyond the label. Raise T.' : 'At higher T the teacher shows that a dog is more cat-like than a fox: that similarity structure is what the student absorbs.';
    }, { fps: 30 });
  };

  /* ====================================================================== drift */
  AI.labs.drift = function (root) {
    const MU = 110, SD = 25, EDGE = [-1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816];
    const { view, ctl } = AI.labShell(root, 'Drift detection', 'the live camera feature slowly leaves the training distribution; the PSI detector notices before anyone has labels',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('rate', 'Brightness drift (grey levels per day)', 0, 3, 0.1, 1.2)}
       ${sl('grow', 'Spread growth (% per day)', 0, 3, 0.1, 0)}
       ${sl('n', 'Window: images per day', 50, 1000, 50, 300)}
       ${sl('thr', 'Alert threshold on PSI', 0.05, 0.5, 0.05, 0.2)}
       ${transport()}`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { rate: (v) => v.toFixed(1), grow: (v) => v.toFixed(1) + '%', thr: (v) => v.toFixed(2), sp: (v) => v.toFixed(2) + '×' });
    const DAYS = 40; let day = 0, tAcc = 0, playing = true, hist = [], samp = [], lastCap = '', alertDay = -1, sdRng = AI.rng(99);
    const gen = (d) => { const n = +q('#n').value, mu = MU + (+q('#rate').value) * d, sd = SD * (1 + (+q('#grow').value) / 100 * d), r = AI.rng(1000 + d); samp = Array.from({ length: n }, () => mu + sd * AI.randn(r)); };
    const metrics = () => {
      const n = samp.length, cnt = new Array(10).fill(0); samp.forEach((x) => { const z = (x - MU) / SD; let k = 0; while (k < 9 && z > EDGE[k]) k++; cnt[k]++; });
      let psi = 0; cnt.forEach((cn) => { const a = Math.max(cn / n, 1e-4), e = 0.1; psi += (a - e) * Math.log(a / e); });
      const so = [...samp].sort((a, b) => a - b); let ks = 0; so.forEach((x, i) => { const F = Phi((x - MU) / SD); ks = Math.max(ks, Math.abs((i + 1) / n - F), Math.abs(i / n - F)); });
      return { psi, ks };
    };
    const reset = () => { day = 0; tAcc = 0; hist = []; alertDay = -1; gen(0); const m = metrics(); hist.push(m.psi); };
    reset();
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    const adv = () => { if (day >= DAYS) return; day++; gen(day); const m = metrics(); hist.push(m.psi); if (alertDay < 0 && m.psi > +q('#thr').value) alertDay = day; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; adv(); };
    q('#reset').onclick = reset; ['rate', 'grow', 'n'].forEach((id) => q('#' + id).addEventListener('change', reset));
    AI.loop(root, (dt) => {
      if (playing) { tAcc += dt * +q('#sp').value; if (tAcc >= 1.2) { tAcc = 0; if (day < DAYS) adv(); } }
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const m = metrics(), lw = Math.min(w * 0.46, 380), thr = +q('#thr').value;
      // left: histograms
      const v = AI.view({ w: lw, h: h }, 20, 220, 0, 0.034, { l: 36, r: 10, t: 24, b: 30 });
      text(g, `feature: mean image brightness — day ${day}`, 36, 14, C.muted, '12px sans-serif');
      g.strokeStyle = C.border; g.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      const bins = 32, bwid = 200 / bins, cnt = new Array(bins).fill(0); samp.forEach((x) => { const k = Math.floor((x - 20) / bwid); if (k >= 0 && k < bins) cnt[k]++; });
      cnt.forEach((cn, k) => { const dens = cn / samp.length / bwid; g.fillStyle = C.orange; g.globalAlpha = 0.65; g.fillRect(v.X(20 + k * bwid), v.Y(dens), v.pw / bins - 1, v.Y(0) - v.Y(dens)); }); g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); for (let x = 20; x <= 220; x += 2) { const y = Math.exp(-0.5 * ((x - MU) / SD) ** 2) / (SD * Math.sqrt(2 * Math.PI)); x === 20 ? g.moveTo(v.X(x), v.Y(y)) : g.lineTo(v.X(x), v.Y(y)); } g.stroke(); g.lineWidth = 1;
      [20, 70, 120, 170, 220].forEach((x) => text(g, String(x), v.X(x), h - 12, C.muted, '10px monospace', 'center'));
      text(g, 'blue = training distribution   orange = live window', 36, h - 0, C.muted, '10px sans-serif');
      // right: PSI vs day
      const rx = lw + 40, rw = w - rx - 14; if (rw > 120) {
        const ymax = Math.max(0.6, ...hist) * 1.1, ph = h - 54, py = 24;
        text(g, 'PSI per day (no labels needed)', rx, 14, C.muted, '12px sans-serif'); g.strokeStyle = C.border; g.strokeRect(rx, py, rw, ph);
        const Y = (p) => py + ph - Math.min(1, p / ymax) * ph, Xd = (d) => rx + d / DAYS * rw;
        g.strokeStyle = C.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(rx, Y(thr)); g.lineTo(rx + rw, Y(thr)); g.stroke(); g.setLineDash([]); text(g, `threshold ${thr.toFixed(2)}`, rx + rw - 4, Y(thr) - 4, C.red, '10px monospace', 'right');
        g.strokeStyle = C.purple; g.lineWidth = 2.5; g.beginPath(); hist.forEach((p, d) => (d ? g.lineTo(Xd(d), Y(p)) : g.moveTo(Xd(d), Y(p)))); g.stroke(); g.lineWidth = 1;
        if (alertDay >= 0) { g.strokeStyle = C.red; g.beginPath(); g.moveTo(Xd(alertDay), py); g.lineTo(Xd(alertDay), py + ph); g.stroke(); text(g, 'ALERT day ' + alertDay, Xd(alertDay) + 4, py + 14, C.red, '600 11px sans-serif'); }
        text(g, '0', rx, py + ph + 14, C.muted, '10px monospace'); text(g, String(DAYS) + ' days', rx + rw, py + ph + 14, C.muted, '10px monospace', 'right');
      }
      const cp = alertDay >= 0 ? `Day ${alertDay}: PSI crossed ${thr.toFixed(2)} — raise an alert, inspect the images, consider retraining. Labels would only arrive weeks later.` : `Day ${day}: PSI ${m.psi.toFixed(3)} is below the threshold; the model still sees what it was trained on.`;
      if (cp !== lastCap) { lastCap = cp; q('#cap').textContent = cp; }
      q('#out').innerHTML = `<span class="m">live mean</span> <b>${(samp.reduce((a, b) => a + b, 0) / samp.length).toFixed(1)}</b> <span class="m">(train ${MU})</span>   <span class="m">PSI</span> <b class="${m.psi > thr ? 'r' : 'g'}">${m.psi.toFixed(3)}</b>   <span class="m">KS</span> <b class="b">${m.ks.toFixed(3)}</b> <span class="m">(≈ 1.36/√n = ${(1.36 / Math.sqrt(samp.length)).toFixed(3)} is the 5% noise level)</span>`;
    }, { fps: 20 });
  };

  /* ====================================================================== fgsm */
  AI.labs.fgsm = function (root) {
    const { view, ctl } = AI.labShell(root, 'FGSM on a tiny linear classifier', 'nudge every feature by ε in the worst direction; watch the decision flip with a perturbation too small to notice',
      '<canvas style="height:330px;touch-action:none;cursor:crosshair"></canvas><div class="caption" id="cap">Drag the point. Score s = w·x + b; class = sign(s).</div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('w1', 'w₁', -3, 3, 0.1, 2)}${sl('w2', 'w₂', -3, 3, 0.1, -1.5)}${sl('b', 'bias b', -2, 2, 0.1, -0.2)}
       ${sl('eps', 'Perturbation budget ε (L∞)', 0, 0.5, 0.01, 0.2)}
       ${transport()}
       ${sl('dd', 'Image pixels d = 10^x', 1, 5.2, 0.1, 5.2)}${sl('mm', 'typical |wᵢ| = 10^x', -4, 0, 0.1, -2)}`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv);
    wire(root, { eps: (v) => v.toFixed(2), dd: (v) => big(10 ** v), mm: (v) => (10 ** v).toExponential(1), sp: (v) => v.toFixed(2) + '×' });
    let x = [0.55, 0.25], drag = false, epsA = 0, playing = true, hold = 0;
    q('#play').onclick = () => { playing = !playing; q('#play').textContent = playing ? 'Pause' : 'Play'; };
    q('#step').onclick = () => { playing = false; q('#play').textContent = 'Play'; epsA = Math.min(+q('#eps').max, epsA + 0.01); };
    q('#reset').onclick = () => { epsA = 0; hold = 0; };
    const geom = () => { const S = Math.min(c.h - 30, c.w * 0.52); return { S, x0: 34, y0: 12 }; };
    const toX = (e) => { const r = cv.getBoundingClientRect(), G = geom(); return [clamp((e.clientX - r.left - G.x0) / G.S, 0, 1), clamp(1 - (e.clientY - r.top - G.y0) / G.S, 0, 1)]; };
    cv.addEventListener('pointerdown', (e) => { drag = true; cv.setPointerCapture(e.pointerId); x = toX(e); epsA = 0; });
    cv.addEventListener('pointermove', (e) => { if (drag) x = toX(e); });
    cv.addEventListener('pointerup', () => (drag = false));
    AI.loop(root, (dt) => {
      const w = [+q('#w1').value, +q('#w2').value], b = +q('#b').value, eps = +q('#eps').value, sp = +q('#sp').value;
      if (playing && !drag) { if (epsA < eps) epsA = Math.min(eps, epsA + dt * 0.05 * sp); else { hold += dt; if (hold > 2 / sp) { hold = 0; epsA = 0; } } } else if (!playing && !drag && epsA > eps) epsA = eps;
      const s = w[0] * x[0] + w[1] * x[1] + b, y = s >= 0 ? 1 : -1, l1 = Math.abs(w[0]) + Math.abs(w[1]), xa = [x[0] - y * epsA * Math.sign(w[0]), x[1] - y * epsA * Math.sign(w[1])];
      const sa = w[0] * xa[0] + w[1] * xa[1] + b, flipped = Math.sign(sa) !== y && sa !== 0, epsStar = l1 ? Math.abs(s) / l1 : Infinity;
      const C = AI.col(), g = c.ctx, Gm = geom(), S = Gm.S, X = (u) => Gm.x0 + u * S, Y = (u) => Gm.y0 + (1 - u) * S; g.clearRect(0, 0, c.w, c.h);
      const cells = 36; for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) { const u = (i + 0.5) / cells, v = (j + 0.5) / cells, sc = w[0] * u + w[1] * v + b; g.fillStyle = sc >= 0 ? C.green : C.red; g.globalAlpha = 0.12 + 0.08 * Math.min(1, Math.abs(sc)); g.fillRect(X(u), Y(v + 1 / cells), S / cells + 1, S / cells + 1); } g.globalAlpha = 1;
      g.strokeStyle = C.border; g.strokeRect(Gm.x0, Gm.y0, S, S);
      // boundary w.x+b=0
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); let started = false;
      for (let i = 0; i <= 100; i++) { const u = i / 100; let v = Math.abs(w[1]) > 1e-6 ? -(w[0] * u + b) / w[1] : NaN; if (Number.isFinite(v) && v >= 0 && v <= 1) { started ? g.lineTo(X(u), Y(v)) : g.moveTo(X(u), Y(v)); started = true; } } g.stroke(); g.lineWidth = 1;
      text(g, 'class +', X(0.02), Y(0.96), C.green, '600 11px sans-serif'); text(g, 'class −', X(0.98), Y(0.04), C.red, '600 11px sans-serif', 'right');
      // L-inf box
      g.strokeStyle = C.orange; g.setLineDash([4, 3]); g.strokeRect(X(x[0] - epsA), Y(x[1] + epsA), 2 * epsA * S, 2 * epsA * S); g.setLineDash([]);
      AI.arrow(g, X(x[0]), Y(x[1]), X(xa[0]), Y(xa[1]), flipped ? C.red : C.orange, 2.5);
      g.fillStyle = C.blue; g.beginPath(); g.arc(X(x[0]), Y(x[1]), 6, 0, 7); g.fill(); g.strokeStyle = C.fg; g.stroke();
      g.fillStyle = flipped ? C.red : C.orange; g.beginPath(); g.arc(X(xa[0]), Y(xa[1]), 6, 0, 7); g.fill();
      text(g, 'x', X(x[0]) + 9, Y(x[1]) - 7, C.fg, '600 12px sans-serif'); text(g, "x'", X(xa[0]) + 9, Y(xa[1]) + 14, C.fg, '600 12px sans-serif');
      // p vs eps plot
      const rx = Gm.x0 + S + 44, rw = c.w - rx - 14; if (rw > 120) {
        const py = 24, ph = 150; text(g, 'P(true class) as ε grows', rx, 16, C.muted, '12px sans-serif'); g.strokeStyle = C.border; g.strokeRect(rx, py, rw, ph);
        const Xe = (e) => rx + e / 0.5 * rw, Yp = (p) => py + ph - p * ph, sg = (z) => 1 / (1 + Math.exp(-z));
        g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(rx, Yp(0.5)); g.lineTo(rx + rw, Yp(0.5)); g.stroke(); g.setLineDash([]);
        g.strokeStyle = C.purple; g.lineWidth = 2.5; g.beginPath(); for (let i = 0; i <= 100; i++) { const e = i / 200, p = sg(y * s - e * l1); i ? g.lineTo(Xe(e), Yp(p)) : g.moveTo(Xe(e), Yp(p)); } g.stroke(); g.lineWidth = 1;
        if (epsStar <= 0.5) { g.strokeStyle = C.red; g.beginPath(); g.moveTo(Xe(epsStar), py); g.lineTo(Xe(epsStar), py + ph); g.stroke(); text(g, 'ε* = ' + epsStar.toFixed(3), Xe(epsStar) + 4, py + 14, C.red, '11px monospace'); }
        g.fillStyle = C.orange; g.beginPath(); g.arc(Xe(epsA), Yp(sg(y * s - epsA * l1)), 5, 0, 7); g.fill();
        text(g, '0', rx, py + ph + 14, C.muted, '10px monospace'); text(g, 'ε = 0.5', rx + rw, py + ph + 14, C.muted, '10px monospace', 'right'); text(g, '0.5 = coin flip', rx + 4, Yp(0.5) - 4, C.muted, '10px sans-serif');
      }
      const d = 10 ** +q('#dd').value, m = 10 ** +q('#mm').value;
      q('#out').innerHTML = `<span class="m">score s = w·x+b</span> <b>${s.toFixed(3)}</b>  p(true)=<b>${(1 / (1 + Math.exp(-y * s))).toFixed(3)}</b>   <span class="m">‖w‖₁</span> <b>${l1.toFixed(2)}</b>\n<span class="m">after attack (ε=${epsA.toFixed(3)})</span> s' = s − y·ε·‖w‖₁ = <b class="${flipped ? 'r' : 'g'}">${sa.toFixed(3)}</b>  ${flipped ? '<b class="r">DECISION FLIPPED</b>' : ''}\n<span class="m">smallest flipping ε* = |s| / ‖w‖₁</span> = <b class="o">${epsStar.toFixed(4)}</b>\n<span class="m">same attack in d = ${big(d)} dims with ε = 1/255:</span> score shift = ε·d·mean|w| = <b class="o">${(d * m / 255).toFixed(3)}</b>`;
    }, { fps: 30 });
  };

  /* ====================================================================== memcalc */
  AI.labs.memcalc = function (root) {
    const mode = root.dataset.mode === 'train' ? 'train' : 'infer', MODELS = ['resnet50', 'resnet18', 'mobilenetv2', 'vgg16', 'vit-b16'], BS = [1, 2, 4, 8, 16, 32, 64, 128, 256], GPU = { 'Jetson 8 GB': 8e9, 'T4 16 GB': 16e9, 'RTX 3090 24 GB': 24e9, 'A100 40 GB': 40e9, 'A100 80 GB': 80e9 };
    const inferCtl = `<label>Precision <select id="prec"><option value="4">float32 (4 B)</option><option value="2">float16 (2 B)</option><option value="1">int8 (1 B)</option></select></label>${sl('ws', 'Workspace (MB, cuDNN / TensorRT scratch)', 0, 1024, 32, 256)}`;
    const trainCtl = `<label>Optimizer <select id="opt"><option value="4">SGD + momentum (4 B/param)</option><option value="8" selected>Adam (8 B/param)</option></select></label><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="amp"> mixed precision (fp16 + fp32 master)</label>${sl('kk', 'Saved tensors per block output (k)', 2, 8, 1, 4)}`;
    const { view, ctl } = AI.labShell(root, mode === 'train' ? 'Training memory calculator' : 'Inference memory calculator', mode === 'train' ? 'weights + gradients + optimizer state + saved activations' : 'weights + activations + workspace',
      '<canvas style="height:150px"></canvas><div class="readout" id="out" style="margin-top:8px"></div><div class="caption" id="cap"></div>',
      `<label>Model <select id="mod">${MODELS.map((m) => `<option value="${m}" ${m === 'resnet50' ? 'selected' : ''}>${AIArch.PRESETS[m].title}</option>`).join('')}</select></label>
       ${sl('bi', 'Batch size', 0, BS.length - 1, 1, 5)}
       ${mode === 'train' ? trainCtl : inferCtl}
       <label>Your GPU <select id="gpu">${Object.keys(GPU).map((k, i) => `<option ${i === 1 ? 'selected' : ''}>${k}</option>`).join('')}</select></label>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    wire(root, { bi: (v) => BS[v], ws: (v) => v + ' MB', kk: (v) => v });
    const cache = {}; const info = (m) => cache[m] || (cache[m] = AIArch.build(AIArch.PRESETS[m].spec).totals);
    let disp = null, lastKey = '';
    const compute = () => {
      const t = info(q('#mod').value), B = BS[+q('#bi').value], P = t.params, segs = [];
      if (mode === 'infer') {
        const by = +q('#prec').value, ws = +q('#ws').value * 1e6;
        segs.push(['weights', P * by, 'blue'], ['activations (2 live buffers of the largest tensor)', 2 * t.peak * B * by, 'green'], ['workspace', ws, 'orange']);
      } else {
        const amp = q('#amp').checked, ob = +q('#opt').value, k = +q('#kk').value;
        segs.push(['weights', P * 4 + (amp ? P * 2 : 0), 'blue'], ['gradients', P * (amp ? 2 : 4), 'purple'], ['optimizer state', P * ob, 'orange'], ['saved activations', k * t.act * B * (amp ? 2 : 4), 'green']);
      }
      return { segs, tot: segs.reduce((a, s) => a + s[1], 0), t, B };
    };
    AI.loop(root, (dt) => {
      const r = compute(), gp = GPU[q('#gpu').value], C = AI.col(), g = c.ctx, w = c.w, h = c.h;
      if (!disp || disp.length !== r.segs.length) disp = r.segs.map(() => 0);
      let moving = false; r.segs.forEach((s, i) => { const d = s[1] - disp[i]; if (Math.abs(d) > 1e5) moving = true; disp[i] += d * ease(dt, 5); });
      const key = [r.tot, gp].join(); if (!moving && key === lastKey) return; lastKey = key;
      g.clearRect(0, 0, w, h);
      const px = 14, pw = w - 28, mx = Math.max(r.tot, gp) * 1.08, X = (b) => px + b / mx * pw; let acc = 0;
      r.segs.forEach((s, i) => { g.fillStyle = C[s[2]]; g.fillRect(X(acc), 40, X(acc + disp[i]) - X(acc), 40); acc += disp[i]; });
      g.strokeStyle = C.border; g.strokeRect(px, 40, pw, 40);
      const over = r.tot > gp; g.strokeStyle = over ? C.red : C.fg; g.lineWidth = 3; g.beginPath(); g.moveTo(X(gp), 28); g.lineTo(X(gp), 92); g.stroke(); g.lineWidth = 1;
      text(g, `${q('#gpu').value}${over ? ' — does NOT fit' : ' — fits'}`, X(gp) - 4, 22, over ? C.red : C.fg, '600 12px sans-serif', X(gp) > w * 0.6 ? 'right' : 'left');
      text(g, `total ${mb(r.tot)}`, px, 20, C.fg, '600 12px sans-serif');
      r.segs.forEach((s, i) => { g.fillStyle = C[s[2]]; g.fillRect(px + (i % 2) * (w / 2 - 10), 104 + Math.floor(i / 2) * 18, 10, 10); text(g, `${s[0]}: ${mb(s[1])}`, px + 16 + (i % 2) * (w / 2 - 10), 113 + Math.floor(i / 2) * 18, C.fg, '11px sans-serif'); });
      const t = r.t;
      q('#out').innerHTML = `<span class="m">parameters</span> <b>${big(t.params)}</b>   <span class="m">largest tensor</span> <b>${big(t.peak)}</b> values   <span class="m">sum of block outputs</span> <b>${big(t.act)}</b> values   <span class="m">batch</span> <b>${r.B}</b>\n` + r.segs.map((s) => `<span class="m">${s[0]}</span> = <b>${mb(s[1])}</b>`).join('\n') + `\n<span class="m">total</span> <b class="o">${mb(r.tot)}</b>   <span class="m">per image (activations)</span> ${mb(r.segs[r.segs.length - (mode === 'train' ? 1 : 2)][1] / r.B)}`;
      q('#cap').textContent = mode === 'train' ? 'Activations grow with batch size; weights, gradients and optimizer state do not. Mixed precision halves the activations but still needs a fp32 master copy.' : 'Weights are fixed; activations scale with batch size. Lower precision shrinks both.';
    }, { fps: 30 });
  };
})();
