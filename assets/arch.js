/* Architecture engine: parse a tiny layer DSL, track tensor shapes, parameters, MACs and receptive fields.
 * Pure logic (no DOM) so it can be tested in node. Used by the `archviz` lab (assets/labs-cv.js).
 *
 * DSL (one layer per line, '#' comments, 'xN' repeats, '@name' labels a block):
 *   input C H W
 *   conv OUT kK sS pP gG [bias]      p defaults to K//2; bias is automatic unless a bn follows
 *   dwconv kK sS                     depthwise conv (keeps channels)
 *   bn | relu | relu6 | gelu | dropout       attached to the previous block as tags (bn adds 2C params)
 *   maxpool kK sS pP | avgpool kK sS pP | gap | flatten | fc OUT
 *   res OUT [sS]                     ResNet basic block (2x 3x3 + skip)
 *   bottleneck OUT [midM] [sS]       ResNet bottleneck (1x1, 3x3, 1x1 + skip); mid defaults to OUT/4
 *   dwsep OUT [sS]                   MobileNet-v1 depthwise-separable block
 *   invres OUT [tT] [sS]             MobileNet-v2 inverted residual (expansion T)
 *   dense L G                        DenseNet block of L layers with growth rate G
 *   up OUT                           transposed conv 2x2 stride 2 (doubles H, W)
 *   concat @name | add @name         merge with an earlier labelled block (skip connection)
 *   patch P D | cls | pos | norm     ViT: patch embedding, class token, position embedding, LayerNorm
 *   tblock HEADS MLP [xL]            L identical transformer encoder blocks, grouped in one box
 *   seqfc OUT                        classifier on the class token
 */
(function (root) {
  const A = {};
  const ACT = ['relu', 'relu6', 'gelu', 'dropout', 'silu'];

  A.parse = function (text) {
    const layers = [], errors = [];
    text.split('\n').forEach((raw, i) => {
      const line = raw.replace(/#.*/, '').trim();
      if (!line) return;
      const tk = line.split(/\s+/), L = { type: tk[0].toLowerCase(), pos: [], line: i + 1, raw: line };
      for (const t of tk.slice(1)) {
        let m;
        if ((m = /^@(\w+)$/.exec(t))) L.label = m[1];
        else if ((m = /^x(\d+)$/.exec(t))) L.rep = +m[1];
        else if ((m = /^(mid|k|s|p|g|t)(\d+)$/.exec(t))) L[m[1]] = +m[2];
        else if (t === 'bias') L.bias = true;
        else if (/^\d+$/.test(t)) L.pos.push(+t);
        else errors.push(`line ${i + 1}: unknown token "${t}"`);
      }
      layers.push(L);
    });
    return { layers, errors };
  };

  const imgSize = (st) => (st.k === 'img' ? st.C * st.H * st.W : st.k === 'seq' ? st.N * st.D : st.D);
  const shapeStr = (st) => (st.k === 'img' ? `${st.C}×${st.H}×${st.W}` : st.k === 'seq' ? `${st.N}×${st.D}` : `${st.D}`);
  A.shapeStr = shapeStr;

  // primitive ops: return {st, params, macs, name}
  function conv(st, o, name) {
    if (st.k !== 'img') throw new Error(`${name || 'conv'} needs an image-shaped input`);
    const { out, k, s = 1, p = Math.floor(k / 2), g = 1 } = o;
    const H = Math.floor((st.H + 2 * p - k) / s) + 1, W = Math.floor((st.W + 2 * p - k) / s) + 1;
    if (H < 1 || W < 1) throw new Error(`${name || 'conv'}: the feature map would shrink below 1 pixel`);
    const w = (st.C / g) * k * k * out, params = w + (o.bias ? out : 0);
    return { st: { k: 'img', C: out, H, W, rf: st.rf + (k - 1) * st.j, j: st.j * s }, params, macs: w * H * W, name: name || `conv ${k}×${k}${s > 1 ? ' /' + s : ''}${g > 1 ? ' g' + g : ''}` };
  }
  const bnOp = (st) => ({ st, params: 2 * st.C, macs: 0, name: 'batch norm' });

  function applyBlock(L, st, ctx) {
    const sub = [];
    const push = (r) => { sub.push({ name: r.name, out: shapeStr(r.st), params: r.params, macs: r.macs, rf: r.st.k === 'img' ? r.st.rf : null }); return r.st; };
    let out = st, skip = null, tags = [];
    const need = (n) => { if (L.pos[0] === undefined && n) throw new Error(`${L.type} needs a number`); };
    switch (L.type) {
      case 'input': { const [C, H, W] = L.pos; out = { k: 'img', C, H, W, rf: 1, j: 1 }; tags = ['input']; sub.push({ name: 'input', out: shapeStr(out), params: 0, macs: 0, rf: 1 }); break; }
      case 'conv': {
        const k = L.k || 3, noBias = L.next === 'bn' && !L.bias;
        const r = conv(st, { out: L.pos[0], k, s: L.s, p: L.p, g: L.g, bias: !noBias }); out = push(r); break;
      }
      case 'dwconv': { const r = conv(st, { out: st.C, k: L.k || 3, s: L.s, p: L.p, g: st.C, bias: false }, `depthwise ${L.k || 3}×${L.k || 3}`); out = push(r); break; }
      case 'maxpool': case 'avgpool': {
        const k = L.k || 2, s = L.s || k, p = L.p || 0;
        const H = Math.floor((st.H + 2 * p - k) / s) + 1, W = Math.floor((st.W + 2 * p - k) / s) + 1;
        out = { k: 'img', C: st.C, H, W, rf: st.rf + (k - 1) * st.j, j: st.j * s };
        push({ st: out, params: 0, macs: 0, name: `${L.type === 'maxpool' ? 'max' : 'avg'} pool ${k}×${k} /${s}` }); break;
      }
      case 'gap': out = { k: 'vec', D: st.C, rf: st.rf, j: st.j }; if (st.k !== 'img') throw new Error('gap needs an image'); push({ st: out, params: 0, macs: 0, name: 'global average pool' }); break;
      case 'flatten': out = { k: 'vec', D: imgSize(st), rf: st.rf, j: st.j }; push({ st: out, params: 0, macs: 0, name: 'flatten' }); break;
      case 'fc': { const inD = imgSize(st), o = L.pos[0]; out = { k: 'vec', D: o, rf: st.rf }; push({ st: out, params: inD * o + o, macs: inD * o, name: `fully connected ${inD}→${o}` }); break; }
      case 'res': {
        const o = L.pos[0], s = L.s || 1;
        let x = push(conv(st, { out: o, k: 3, s, bias: false }, `conv 3×3${s > 1 ? ' /' + s : ''}`)); push(bnOp(x));
        x = push(conv(x, { out: o, k: 3, bias: false }, 'conv 3×3')); push(bnOp(x));
        if (s !== 1 || st.C !== o) { const r = conv(st, { out: o, k: 1, s, p: 0, bias: false }, `skip: 1×1 projection${s > 1 ? ' /' + s : ''}`); sub.push({ name: r.name, out: shapeStr(r.st), params: r.params + 2 * o, macs: r.macs, rf: null }); }
        out = { ...x, rf: x.rf, j: x.j }; skip = 'add'; break;
      }
      case 'bottleneck': {
        const o = L.pos[0], mid = L.mid || Math.round(o / 4), s = L.s || 1;
        let x = push(conv(st, { out: mid, k: 1, p: 0, bias: false }, 'conv 1×1 reduce')); push(bnOp(x));
        x = push(conv(x, { out: mid, k: 3, s, bias: false }, `conv 3×3${s > 1 ? ' /' + s : ''}`)); push(bnOp(x));
        x = push(conv(x, { out: o, k: 1, p: 0, bias: false }, 'conv 1×1 expand')); push(bnOp(x));
        if (s !== 1 || st.C !== o) { const r = conv(st, { out: o, k: 1, s, p: 0, bias: false }, `skip: 1×1 projection${s > 1 ? ' /' + s : ''}`); sub.push({ name: r.name, out: shapeStr(r.st), params: r.params + 2 * o, macs: r.macs, rf: null }); }
        out = x; skip = 'add'; break;
      }
      case 'dwsep': {
        const o = L.pos[0], s = L.s || 1;
        let x = push(conv(st, { out: st.C, k: 3, s, g: st.C, bias: false }, `depthwise 3×3${s > 1 ? ' /' + s : ''}`)); push(bnOp(x));
        x = push(conv(x, { out: o, k: 1, p: 0, bias: false }, 'pointwise 1×1')); push(bnOp(x)); out = x; break;
      }
      case 'invres': {
        const o = L.pos[0], t = L.t || 6, s = L.s || 1, e = st.C * t;
        let x = st;
        if (t !== 1) { x = push(conv(x, { out: e, k: 1, p: 0, bias: false }, `expand 1×1 (×${t})`)); push(bnOp(x)); }
        x = push(conv(x, { out: e, k: 3, s, g: e, bias: false }, `depthwise 3×3${s > 1 ? ' /' + s : ''}`)); push(bnOp(x));
        x = push(conv(x, { out: o, k: 1, p: 0, bias: false }, 'project 1×1 (linear)')); push(bnOp(x));
        out = x; if (s === 1 && st.C === o) skip = 'add'; break;
      }
      case 'dense': {
        const [n, g] = L.pos; let C = st.C, x = st;
        for (let i = 0; i < n; i++) {
          const b = { ...x, C }; push({ st: b, params: 2 * C, macs: 0, name: `layer ${i + 1}: bn` });
          const r1 = conv(b, { out: 4 * g, k: 1, p: 0, bias: false }, `layer ${i + 1}: 1×1 → ${4 * g}`); sub.push({ name: r1.name, out: shapeStr(r1.st), params: r1.params + 8 * g, macs: r1.macs, rf: null });
          const r2 = conv(r1.st, { out: g, k: 3, bias: false }, `layer ${i + 1}: 3×3 → ${g}`); sub.push({ name: r2.name, out: shapeStr(r2.st), params: r2.params, macs: r2.macs, rf: r2.st.rf });
          C += g; x = { ...r2.st, C };
        }
        out = x; skip = 'concat'; break;
      }
      case 'up': { const o = L.pos[0]; if (st.k !== 'img') throw new Error('up needs an image'); out = { k: 'img', C: o, H: st.H * 2, W: st.W * 2, rf: st.rf, j: st.j / 2 }; push({ st: out, params: st.C * o * 4 + o, macs: st.C * o * 4 * st.H * st.W, name: `up-conv 2×2 ×2 → ${o}` }); break; }
      case 'concat': case 'add': {
        const ref = ctx.labels[L.label]; if (!ref) throw new Error(`${L.type}: unknown label @${L.label}`);
        const r = ref.out; if (r.H !== st.H || r.W !== st.W) throw new Error(`${L.type} @${L.label}: spatial sizes differ (${shapeStr(r)} vs ${shapeStr(st)})`);
        if (L.type === 'add' && r.C !== st.C) throw new Error(`add @${L.label}: channel counts differ`);
        out = { ...st, C: L.type === 'concat' ? st.C + r.C : st.C }; skip = { from: ref.index, kind: L.type };
        push({ st: out, params: 0, macs: 0, name: `${L.type} with @${L.label} (${shapeStr(r)})` }); break;
      }
      case 'patch': {
        const [P, D] = L.pos; if (st.k !== 'img') throw new Error('patch needs an image');
        const r = conv(st, { out: D, k: P, s: P, p: 0, bias: true }, `patchify ${P}×${P} + linear → ${D}`);
        out = { k: 'seq', N: r.st.H * r.st.W, D, rf: P, j: P }; push({ st: out, params: r.params, macs: r.macs, name: r.name + ` (${out.N} tokens)` }); break;
      }
      case 'cls': out = { ...st, N: st.N + 1 }; push({ st: out, params: st.D, macs: 0, name: 'class token' }); break;
      case 'pos': out = st; push({ st: out, params: st.N * st.D, macs: 0, name: 'position embedding' }); break;
      case 'norm': out = st; push({ st: out, params: 2 * (st.D || st.C), macs: 0, name: 'layer norm' }); break;
      case 'tblock': {
        if (st.k !== 'seq') throw new Error('tblock needs tokens (use patch first)');
        const [heads, mlp] = L.pos, D = st.D, N = st.N, n = L.rep || 1;
        const attn = 4 * D * D + 4 * D, ff = 2 * D * mlp + mlp + D, ln = 4 * D;
        const mAttn = N * 4 * D * D + 2 * N * N * D, mFf = N * 2 * D * mlp;
        sub.push({ name: `multi-head attention (${heads} heads, d=${D / heads} each) ×${n}`, out: shapeStr(st), params: n * attn, macs: n * mAttn, rf: null });
        sub.push({ name: `MLP ${D}→${mlp}→${D} ×${n}`, out: shapeStr(st), params: n * ff, macs: n * mFf, rf: null });
        sub.push({ name: `2 layer norms ×${n}`, out: shapeStr(st), params: n * ln, macs: 0, rf: null });
        out = st; skip = 'add'; break;
      }
      case 'seqfc': { const o = L.pos[0]; out = { k: 'vec', D: o }; push({ st: out, params: st.D * o + o, macs: st.D * o, name: `class-token head ${st.D}→${o}` }); break; }
      default: throw new Error(`unknown layer "${L.type}"`);
    }
    return { out, sub, skip, tags };
  }

  /** build(text, {resolution}) -> {blocks, totals, errors} */
  A.build = function (text, opts = {}) {
    const { layers, errors } = A.parse(text);
    const blocks = [], ctx = { labels: {} };
    let st = null;
    // expand xN repeats (except tblock, which groups), and annotate "next layer" for the bias heuristic
    const flat = [];
    layers.forEach((L) => { const n = L.type === 'tblock' ? 1 : L.rep || 1; for (let i = 0; i < n; i++) flat.push({ ...L, rep: L.type === 'tblock' ? L.rep : undefined, label: i === n - 1 ? L.label : undefined }); });
    flat.forEach((L, i) => { L.next = flat[i + 1] && flat[i + 1].type; });
    for (const L of flat) {
      try {
        if (ACT.includes(L.type) || L.type === 'bn') {
          const b = blocks[blocks.length - 1]; if (!b) continue;
          b.tags.push(L.type === 'bn' ? 'BN' : L.type.toUpperCase());
          if (L.type === 'bn') { const p = 2 * (b.out.C || 0); b.params += p; b.sub.push({ name: 'batch norm', out: b.outStr, params: p, macs: 0, rf: null }); }
          continue;
        }
        if (L.type === 'input' && opts.resolution) { L.pos = [L.pos[0], opts.resolution, opts.resolution]; }
        if (!st && L.type !== 'input') throw new Error('start with: input C H W');
        const r = applyBlock(L, st, ctx);
        const params = r.sub.reduce((a, x) => a + x.params, 0), macs = r.sub.reduce((a, x) => a + x.macs, 0);
        const b = { type: L.type, label: L.label, line: L.line, name: displayName(L), in: st, out: r.out, outStr: shapeStr(r.out), sub: r.sub, params, macs, tags: r.tags.slice(), skip: r.skip, rep: L.rep, index: blocks.length, act: imgSize(r.out) * (L.rep || 1) };
        if (L.label && L.type !== 'concat' && L.type !== 'add') ctx.labels[L.label] = b;
        blocks.push(b); st = r.out;
      } catch (e) { errors.push(`line ${L.line}: ${e.message}`); break; }
    }
    const totals = { params: 0, macs: 0, act: 0, peak: 0, rf: st && st.rf };
    blocks.forEach((b) => { totals.params += b.params; totals.macs += b.macs; totals.act += b.act; totals.peak = Math.max(totals.peak, b.act); });
    return { blocks, totals, errors, final: st };
  };

  function displayName(L) {
    const p = L.pos;
    switch (L.type) {
      case 'input': return 'input';
      case 'conv': return `conv ${L.k || 3}×${L.k || 3}${L.s > 1 ? ' /' + L.s : ''}`;
      case 'dwconv': return 'depthwise';
      case 'maxpool': return 'max pool'; case 'avgpool': return 'avg pool'; case 'gap': return 'global pool';
      case 'flatten': return 'flatten'; case 'fc': return `FC ${p[0]}`;
      case 'res': return `res block${L.s > 1 ? ' /' + L.s : ''}`;
      case 'bottleneck': return `bottleneck${L.s > 1 ? ' /' + L.s : ''}`;
      case 'dwsep': return `dw-sep${L.s > 1 ? ' /' + L.s : ''}`;
      case 'invres': return `inv. res ×${L.t || 6}${L.s > 1 ? ' /' + L.s : ''}`;
      case 'dense': return `dense ${p[0]}×g${p[1]}`;
      case 'up': return 'up-conv'; case 'concat': return 'concat'; case 'add': return 'add';
      case 'patch': return `patch ${p[0]}`; case 'cls': return 'cls token'; case 'pos': return 'pos emb'; case 'norm': return 'layer norm';
      case 'tblock': return `transformer${L.rep ? ' ×' + L.rep : ''}`; case 'seqfc': return 'head';
      default: return L.type;
    }
  }

  A.PRESETS = {
    'small-cnn': { title: 'Small CNN (CIFAR-style)', spec: `input 3 32 32
conv 32 k3
bn
relu
maxpool k2
conv 64 k3
bn
relu
maxpool k2
conv 128 k3
bn
relu
gap
fc 10` },
    lenet5: { title: 'LeNet-5 (1998)', spec: `input 1 32 32
conv 6 k5 p0
relu
avgpool k2
conv 16 k5 p0
relu
avgpool k2
flatten
fc 120
relu
fc 84
relu
fc 10` },
    alexnet: { title: 'AlexNet (2012)', spec: `input 3 224 224
conv 64 k11 s4 p2
relu
maxpool k3 s2
conv 192 k5 p2
relu
maxpool k3 s2
conv 384 k3
relu
conv 256 k3
relu
conv 256 k3
relu
maxpool k3 s2
flatten
fc 4096
relu
fc 4096
relu
fc 1000` },
    vgg16: { title: 'VGG-16 (2014)', spec: `input 3 224 224
conv 64 k3
relu
conv 64 k3
relu
maxpool k2
conv 128 k3
relu
conv 128 k3
relu
maxpool k2
conv 256 k3 x3
maxpool k2
conv 512 k3 x3
maxpool k2
conv 512 k3 x3
maxpool k2
flatten
fc 4096
relu
fc 4096
relu
fc 1000` },
    resnet18: { title: 'ResNet-18 (2015)', spec: `input 3 224 224
conv 64 k7 s2 p3
bn
relu
maxpool k3 s2 p1
res 64 x2
res 128 s2
res 128
res 256 s2
res 256
res 512 s2
res 512
gap
fc 1000` },
    resnet50: { title: 'ResNet-50 (2015)', spec: `input 3 224 224
conv 64 k7 s2 p3
bn
relu
maxpool k3 s2 p1
bottleneck 256 mid64 x3
bottleneck 512 mid128 s2
bottleneck 512 mid128 x3
bottleneck 1024 mid256 s2
bottleneck 1024 mid256 x5
bottleneck 2048 mid512 s2
bottleneck 2048 mid512 x2
gap
fc 1000` },
    mobilenetv1: { title: 'MobileNet-v1 (2017)', spec: `input 3 224 224
conv 32 k3 s2
bn
relu
dwsep 64
dwsep 128 s2
dwsep 128
dwsep 256 s2
dwsep 256
dwsep 512 s2
dwsep 512 x5
dwsep 1024 s2
dwsep 1024
gap
fc 1000` },
    mobilenetv2: { title: 'MobileNet-v2 (2018)', spec: `input 3 224 224
conv 32 k3 s2
bn
relu6
invres 16 t1
invres 24 t6 s2
invres 24 t6
invres 32 t6 s2
invres 32 t6 x2
invres 64 t6 s2
invres 64 t6 x3
invres 96 t6 x3
invres 160 t6 s2
invres 160 t6 x2
invres 320 t6
conv 1280 k1
bn
relu6
gap
fc 1000` },
    densenet121: { title: 'DenseNet-121 (2017)', spec: `input 3 224 224
conv 64 k7 s2 p3
bn
relu
maxpool k3 s2 p1
dense 6 32
bn
conv 128 k1
avgpool k2
dense 12 32
bn
conv 256 k1
avgpool k2
dense 24 32
bn
conv 512 k1
avgpool k2
dense 16 32
bn
gap
fc 1000` },
    'vit-b16': { title: 'ViT-B/16 (2020)', spec: `input 3 224 224
patch 16 768
cls
pos
tblock 12 3072 x12
norm
seqfc 1000` },
    'unet-small': { title: 'U-Net (2015, small)', spec: `input 1 128 128
conv 64 k3
bn
relu
conv 64 k3 @e1
bn
relu
maxpool k2
conv 128 k3
bn
relu
conv 128 k3 @e2
bn
relu
maxpool k2
conv 256 k3
bn
relu
conv 256 k3 @e3
bn
relu
maxpool k2
conv 512 k3
bn
relu
conv 512 k3
bn
relu
up 256
concat @e3
conv 256 k3
bn
relu
up 128
concat @e2
conv 128 k3
bn
relu
up 64
concat @e1
conv 64 k3
bn
relu
conv 2 k1` },
  };

  root.AIArch = A;
  if (typeof module !== 'undefined') module.exports = A;
})(typeof window !== 'undefined' ? window : globalThis);
