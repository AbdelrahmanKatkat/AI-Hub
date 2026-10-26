/* Module 4 (CNN architectures) labs.
 *   blockflow     <div data-lab="blockflow" data-blocks="basic,bottleneck,..." data-block="basic" data-C="64" data-H="56" data-cfg="Cout=128;s=2">
 *   gradflow      plain vs residual gradient flow (toy random-Jacobian model)
 *   convcost      standard vs grouped vs depthwise-separable vs inverted-residual cost
 *   relumanifold  why the inverted residual projection is linear
 *   scaling       compound scaling explorer (EfficientNet)
 *   modelstrip    <div data-lab="modelstrip" data-model="googlenet|convnext-t|efficientnet-b0|<AIArch preset>">
 *   archcompare   sorted bars of params / MACs / activations / depth / published top-1
 *   archlab       capstone: edit an architecture under a parameter / MAC budget
 * The pure part (CVA) also runs under node for verification.
 */
(function (root) {
  'use strict';
  const CVA = {};
  const fmtN = (v) => (v >= 1e9 ? (v / 1e9).toFixed(2) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' K' : String(Math.round(v)));
  const com = (v) => Math.round(v).toLocaleString('en-US');
  const sh = (o) => (o.H === 1 && o.C ? `${o.C}` : `${o.C}×${o.H}×${o.H}`);
  CVA.fmtN = fmtN;

  /* ---------------- block graphs ---------------- */
  function G() {
    const g = { nodes: [], edges: [] };
    g.node = (id, label, o) => { const n = Object.assign({ id, label, kind: 'conv', params: 0, macs: 0, col: 0, row: 0, tags: '', inn: null, out: null, formula: '', note: '' }, o); g.nodes.push(n); return n; };
    g.edge = (a, b, o) => g.edges.push(Object.assign({ a, b }, o || {}));
    g.by = (id) => g.nodes.find((n) => n.id === id);
    return g;
  }
  // convolution node: cin -> cout, kernel k, groups gr, stride s, input spatial H
  function cnode(g, id, label, o) {
    const { cin, cout, k, gr = 1, s = 1, H, bn = true, bias = false } = o, Ho = Math.floor(H / s), w = (cin / gr) * k * k * cout;
    const formula = `${gr > 1 ? '(' + cin + '/' + gr + ')' : cin}·${k}²·${cout} = ${com(w)}` + (bn ? ` + BN ${2 * cout}` : '') + (bias ? ` + bias ${cout}` : '');
    return g.node(id, label, Object.assign({ params: w + (bn ? 2 * cout : 0) + (bias ? cout : 0), macs: w * Ho * Ho, inn: { C: cin, H }, out: { C: cout, H: Ho }, formula, kind: o.kind || (gr === cin && gr > 1 ? 'dw' : 'conv') }, { col: o.col, row: o.row || 0, tags: o.tags || '' }));
  }
  const io = (g, id, label, C, H, col, row) => g.node(id, label, { kind: 'io', out: { C, H }, inn: { C, H }, col, row: row || 0 });
  const snap = (v, d) => { v = Math.round(v); while (d > 1 && v % d) v++; return v; };

  const B = {};
  B.plain = { title: 'Plain 2-conv block (no skip)', def: { Cout: 64 }, ctl: [['Cout', 'C_out', 8, 512, 8]], note: 'Two stacked 3×3 convolutions. Nothing carries the input past them.', build(c) {
    const g = G(), { C, H, Cout } = c; io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'a', 'conv 3×3', { cin: C, cout: Cout, k: 3, H, col: 1, tags: 'BN · ReLU' }); cnode(g, 'b', 'conv 3×3', { cin: Cout, cout: Cout, k: 3, H, col: 2, tags: 'BN · ReLU' });
    io(g, 'y', 'output', Cout, H, 3); g.edge('x', 'a'); g.edge('a', 'b'); g.edge('b', 'y'); return g; } };
  function skipPath(g, c, Cout, s, Hout, lastCol) {
    const { C, H } = c;
    if (s !== 1 || C !== Cout) { cnode(g, 'p', 'skip 1×1 conv', { cin: C, cout: Cout, k: 1, s, H, col: 1, row: -1.2, tags: 'BN (projection)' }); g.edge('x', 'p'); g.edge('p', 'add'); }
    else g.edge('x', 'add', { lane: -1.2, label: 'identity (no cost)', dash: true });
  }
  B.basic = { title: 'ResNet basic block', def: { Cout: 64, s: 1 }, ctl: [['Cout', 'C_out', 8, 512, 8], ['s', 'stride', 1, 2, 1]], note: 'y = F(x) + x with F = two 3×3 convs. If shapes differ the skip uses a 1×1 projection.', build(c) {
    const g = G(), { C, H, Cout, s } = c, Ho = Math.floor(H / s); io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'a', 'conv 3×3', { cin: C, cout: Cout, k: 3, s, H, col: 1, tags: 'BN · ReLU' }); cnode(g, 'b', 'conv 3×3', { cin: Cout, cout: Cout, k: 3, H: Ho, col: 2, tags: 'BN' });
    g.node('add', 'add', { kind: 'merge', op: '+', col: 3, inn: { C: Cout, H: Ho }, out: { C: Cout, H: Ho }, tags: 'ReLU' }); io(g, 'y', 'output', Cout, Ho, 4);
    g.edge('x', 'a'); g.edge('a', 'b'); g.edge('b', 'add'); g.edge('add', 'y'); skipPath(g, c, Cout, s, Ho); return g; } };
  B.bottleneck = { title: 'ResNet bottleneck block', def: { Cout: 256, mid: 64, s: 1 }, ctl: [['Cout', 'C_out', 16, 2048, 16], ['mid', 'mid channels', 8, 512, 8], ['s', 'stride', 1, 2, 1]], note: '1×1 squeezes channels, 3×3 does the spatial work on few channels, 1×1 expands again; skip adds the input.', build(c) {
    const g = G(), { C, H, Cout, mid, s } = c, Ho = Math.floor(H / s); io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'a', 'conv 1×1 reduce', { cin: C, cout: mid, k: 1, H, col: 1, tags: 'BN · ReLU' }); cnode(g, 'b', 'conv 3×3', { cin: mid, cout: mid, k: 3, s, H, col: 2, tags: 'BN · ReLU' }); cnode(g, 'c', 'conv 1×1 expand', { cin: mid, cout: Cout, k: 1, H: Ho, col: 3, tags: 'BN' });
    g.node('add', 'add', { kind: 'merge', op: '+', col: 4, inn: { C: Cout, H: Ho }, out: { C: Cout, H: Ho }, tags: 'ReLU' }); io(g, 'y', 'output', Cout, Ho, 5);
    g.edge('x', 'a'); g.edge('a', 'b'); g.edge('b', 'c'); g.edge('c', 'add'); g.edge('add', 'y'); skipPath(g, c, Cout, s, Ho); return g; } };
  B.resnext = { title: 'ResNeXt block (grouped 3×3)', def: { Cout: 256, mid: 128, gr: 32 }, ctl: [['Cout', 'C_out', 16, 1024, 16], ['mid', 'mid channels', 32, 512, 32], ['gr', 'groups', 1, 64, 1]], note: 'A bottleneck whose 3×3 is split into groups: each group sees only mid/groups channels.', build(c) {
    const g = G(), { C, H, Cout } = c, gr = Math.max(1, c.gr), mid = snap(c.mid, gr); io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'a', 'conv 1×1 reduce', { cin: C, cout: mid, k: 1, H, col: 1, tags: 'BN · ReLU' }); cnode(g, 'b', `grouped 3×3 (g=${gr})`, { cin: mid, cout: mid, k: 3, gr, H, col: 2, tags: 'BN · ReLU', kind: 'conv' }); cnode(g, 'c', 'conv 1×1 expand', { cin: mid, cout: Cout, k: 1, H, col: 3, tags: 'BN' });
    g.node('add', 'add', { kind: 'merge', op: '+', col: 4, inn: { C: Cout, H }, out: { C: Cout, H } }); io(g, 'y', 'output', Cout, H, 5);
    g.edge('x', 'a'); g.edge('a', 'b'); g.edge('b', 'c'); g.edge('c', 'add'); g.edge('add', 'y'); skipPath(g, Object.assign({}, c), Cout, 1, H); return g; } };
  B.inception = { title: 'Inception module (GoogLeNet 3a)', def: { b1: 64, r3: 96, b3: 128, r5: 16, b5: 32, pp: 32, naive: 0 }, ctl: [['b1', '1×1 branch', 8, 256, 8], ['r3', '3×3 reduce', 8, 256, 8], ['b3', '3×3 branch', 8, 256, 8], ['r5', '5×5 reduce', 4, 64, 4], ['b5', '5×5 branch', 8, 128, 8], ['pp', 'pool proj', 8, 128, 8], ['naive', 'naive (no 1×1 reduce): 0/1', 0, 1, 1]], note: 'Four parallel branches see the same input at different scales; their outputs are concatenated along channels. The 1×1 reductions keep the cost down.', build(c) {
    const g = G(), { C, H, b1, r3, b3, r5, b5, pp, naive } = c; io(g, 'x', 'input x', C, H, 0);
    const o = { H, tags: 'ReLU', bn: false, bias: true };
    cnode(g, 'c1', 'conv 1×1', Object.assign({ cin: C, cout: b1, k: 1, col: 1, row: -1.5 }, o));
    if (naive) {
      cnode(g, 'c3', 'conv 3×3', Object.assign({ cin: C, cout: b3, k: 3, col: 1, row: -0.5 }, o)); cnode(g, 'c5', 'conv 5×5', Object.assign({ cin: C, cout: b5, k: 5, col: 1, row: 0.5 }, o));
      g.node('mp', 'max pool 3×3', { kind: 'pool', col: 1, row: 1.5, inn: { C, H }, out: { C, H } });
      g.node('cat', 'concat', { kind: 'merge', op: '‖', col: 2, inn: { C: b1 + b3 + b5 + C, H }, out: { C: b1 + b3 + b5 + C, H } });
      ['c1', 'c3', 'c5', 'mp'].forEach((n) => g.edge('x', n)); ['c1', 'c3', 'c5', 'mp'].forEach((n) => g.edge(n, 'cat'));
    } else {
      cnode(g, 'rd3', '1×1 reduce', Object.assign({ cin: C, cout: r3, k: 1, col: 1, row: -0.5 }, o)); cnode(g, 'c3', 'conv 3×3', Object.assign({ cin: r3, cout: b3, k: 3, col: 2, row: -0.5 }, o));
      cnode(g, 'rd5', '1×1 reduce', Object.assign({ cin: C, cout: r5, k: 1, col: 1, row: 0.5 }, o)); cnode(g, 'c5', 'conv 5×5', Object.assign({ cin: r5, cout: b5, k: 5, col: 2, row: 0.5 }, o));
      g.node('mp', 'max pool 3×3', { kind: 'pool', col: 1, row: 1.5, inn: { C, H }, out: { C, H } }); cnode(g, 'pp', 'conv 1×1 proj', Object.assign({ cin: C, cout: pp, k: 1, col: 2, row: 1.5 }, o));
      g.node('cat', 'concat', { kind: 'merge', op: '‖', col: 3, inn: { C: b1 + b3 + b5 + pp, H }, out: { C: b1 + b3 + b5 + pp, H } });
      g.edge('x', 'c1'); g.edge('x', 'rd3'); g.edge('x', 'rd5'); g.edge('x', 'mp'); g.edge('rd3', 'c3'); g.edge('rd5', 'c5'); g.edge('mp', 'pp'); g.edge('c1', 'cat'); g.edge('c3', 'cat'); g.edge('c5', 'cat'); g.edge('pp', 'cat');
    }
    io(g, 'y', 'output', g.by('cat').out.C, H, g.by('cat').col + 1); g.edge('cat', 'y'); return g; } };
  B.dense = { title: 'DenseNet dense block', def: { L: 4, gk: 32 }, ctl: [['L', 'layers L', 2, 6, 1], ['gk', 'growth rate k', 4, 48, 4]], note: 'Every layer receives the concatenation of all earlier outputs and adds k new channels.', build(c) {
    const g = G(), { C, H, L, gk } = c; io(g, 'x', 'x₀ (input)', C, H, 0);
    for (let i = 1; i <= L; i++) {
      const cin = C + gk * (i - 1), w1 = cin * 4 * gk, w2 = 4 * gk * 9 * gk;
      g.node('h' + i, `H${i}: BN-ReLU-1×1(${4 * gk}) BN-ReLU-3×3(${gk})`.replace(/(.{0})/, ''), { kind: 'conv', col: i, params: 2 * cin + w1 + 8 * gk + w2, macs: (w1 + w2) * H * H, inn: { C: cin, H }, out: { C: gk, H }, label: `H${i}`, tags: `in ${cin} ch → +${gk}`, formula: `BN ${2 * cin} + 1×1: ${cin}·${4 * gk} = ${com(w1)} + BN ${8 * gk} + 3×3: ${4 * gk}·9·${gk} = ${com(w2)}` });
    }
    const Co = C + gk * L; g.node('cat', 'concat all', { kind: 'merge', op: '‖', col: L + 1, inn: { C: Co, H }, out: { C: Co, H } }); io(g, 'y', 'output', Co, H, L + 2);
    for (let i = 1; i <= L; i++) { g.edge(i === 1 ? 'x' : 'h' + (i - 1), 'h' + i); for (let j = 0; j < i - 1; j++) g.edge(j === 0 ? 'x' : 'h' + j, 'h' + i, { arc: 26 + 20 * (i - j), nolabel: true }); }
    g.edge('h' + L, 'cat'); g.edge('x', 'cat', { arc: 26 + 20 * (L + 1), nolabel: true }); for (let j = 1; j < L; j++) g.edge('h' + j, 'cat', { arc: 20 + 18 * (L + 1 - j), nolabel: true });
    g.edge('cat', 'y'); return g; } };
  B.dwsep = { title: 'MobileNet-v1 depthwise-separable block', def: { Cout: 128, s: 1 }, ctl: [['Cout', 'C_out', 8, 1024, 8], ['s', 'stride', 1, 2, 1]], note: 'Depthwise 3×3 filters each channel alone; a pointwise 1×1 then mixes channels.', build(c) {
    const g = G(), { C, H, Cout, s } = c; io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'a', 'depthwise 3×3', { cin: C, cout: C, k: 3, gr: C, s, H, col: 1, tags: 'BN · ReLU' }); cnode(g, 'b', 'pointwise 1×1', { cin: C, cout: Cout, k: 1, H: Math.floor(H / s), col: 2, tags: 'BN · ReLU' });
    io(g, 'y', 'output', Cout, Math.floor(H / s), 3); g.edge('x', 'a'); g.edge('a', 'b'); g.edge('b', 'y'); return g; } };
  B.invres = { title: 'MobileNet-v2 inverted residual', def: { Cout: 64, t: 6, s: 1 }, ctl: [['Cout', 'C_out', 8, 512, 8], ['t', 'expansion t', 1, 8, 1], ['s', 'stride', 1, 2, 1]], note: 'Narrow → wide → narrow: expand 1×1, depthwise 3×3 on the wide tensor, linear 1×1 projection (no ReLU).', build(c) {
    const g = G(), { C, H, Cout, t, s } = c, e = C * t, Ho = Math.floor(H / s); io(g, 'x', 'input x', C, H, 0); let col = 1, prev = 'x';
    if (t !== 1) { cnode(g, 'a', `expand 1×1 (×${t})`, { cin: C, cout: e, k: 1, H, col, tags: 'BN · ReLU6' }); g.edge(prev, 'a'); prev = 'a'; col++; }
    cnode(g, 'b', 'depthwise 3×3', { cin: e, cout: e, k: 3, gr: e, s, H, col, tags: 'BN · ReLU6' }); g.edge(prev, 'b'); col++;
    cnode(g, 'c', 'project 1×1 (linear)', { cin: e, cout: Cout, k: 1, H: Ho, col, tags: 'BN, no ReLU' }); g.edge('b', 'c'); col++;
    if (s === 1 && C === Cout) { g.node('add', 'add', { kind: 'merge', op: '+', col, inn: { C: Cout, H }, out: { C: Cout, H } }); g.edge('c', 'add'); g.edge('x', 'add', { lane: -1.2, label: 'identity', dash: true }); io(g, 'y', 'output', Cout, Ho, col + 1); g.edge('add', 'y'); }
    else { io(g, 'y', 'output', Cout, Ho, col); g.edge('c', 'y'); } return g; } };
  B.se = { title: 'Squeeze-and-Excitation block', def: { r: 16 }, ctl: [['r', 'reduction r', 2, 32, 2]], note: 'Squeeze: global pool to one number per channel. Excite: a tiny MLP + sigmoid gives a gate per channel. Scale: multiply the feature map by its gates.', build(c) {
    const g = G(), { C, H } = c, r = c.r, m = Math.max(1, Math.floor(C / r)); io(g, 'x', 'feature map U', C, H, 0);
    g.node('gap', 'global avg pool', { kind: 'pool', col: 1, row: 1.3, inn: { C, H }, out: { C, H: 1 }, note: 'squeeze' });
    g.node('f1', `FC ${C}→${m}`, { kind: 'fc', col: 2, row: 1.3, params: C * m + m, macs: C * m, inn: { C, H: 1 }, out: { C: m, H: 1 }, tags: 'ReLU', formula: `${C}·${m} + ${m} = ${com(C * m + m)}` });
    g.node('f2', `FC ${m}→${C}`, { kind: 'fc', col: 3, row: 1.3, params: m * C + C, macs: m * C, inn: { C: m, H: 1 }, out: { C, H: 1 }, formula: `${m}·${C} + ${C} = ${com(m * C + C)}` });
    g.node('sg', 'sigmoid', { kind: 'norm', col: 4, row: 1.3, inn: { C, H: 1 }, out: { C, H: 1 } });
    g.node('mul', 'scale', { kind: 'merge', op: '×', col: 5, inn: { C, H }, out: { C, H } }); io(g, 'y', 'recalibrated', C, H, 6);
    g.edge('x', 'gap'); g.edge('gap', 'f1'); g.edge('f1', 'f2'); g.edge('f2', 'sg'); g.edge('sg', 'mul'); g.edge('x', 'mul', { lane: -1.2, label: 'U (unchanged)', dash: true }); g.edge('mul', 'y'); return g; } };
  B.convnext = { title: 'ConvNeXt block', def: {}, ctl: [], note: 'Depthwise 7×7 (spatial mixing) → LayerNorm → 1×1 up ×4 → GELU → 1×1 down → layer-scale → add. An inverted bottleneck with a large kernel.', build(c) {
    const g = G(), { C, H } = c; io(g, 'x', 'input x', C, H, 0);
    cnode(g, 'dw', 'depthwise 7×7', { cin: C, cout: C, k: 7, gr: C, H, col: 1, bn: false, bias: true });
    g.node('ln', 'LayerNorm', { kind: 'norm', col: 2, params: 2 * C, inn: { C, H }, out: { C, H }, formula: `2·${C}` });
    cnode(g, 'p1', 'pointwise 1×1 ×4', { cin: C, cout: 4 * C, k: 1, H, col: 3, bn: false, bias: true, tags: 'GELU' }); cnode(g, 'p2', 'pointwise 1×1 ÷4', { cin: 4 * C, cout: C, k: 1, H, col: 4, bn: false, bias: true });
    g.node('ls', 'layer scale γ', { kind: 'norm', col: 5, params: C, inn: { C, H }, out: { C, H }, formula: `${C}` });
    g.node('add', 'add', { kind: 'merge', op: '+', col: 6, inn: { C, H }, out: { C, H } }); io(g, 'y', 'output', C, H, 7);
    ['x>dw', 'dw>ln', 'ln>p1', 'p1>p2', 'p2>ls', 'ls>add', 'add>y'].forEach((s) => g.edge(...s.split('>'))); g.edge('x', 'add', { lane: -1.2, label: 'identity', dash: true }); return g; } };
  B.shuffle = { title: 'ShuffleNet unit', def: { gr: 3 }, ctl: [['gr', 'groups', 1, 8, 1]], note: 'Grouped 1×1 convs are cheap but isolate channel groups; a channel shuffle mixes the groups before the depthwise 3×3.', build(c) {
    const g = G(), { C, H } = c, gr = Math.max(1, c.gr), Cs = snap(C, gr * 4), mid = Cs / 4; io(g, 'x', 'input x', Cs, H, 0);
    cnode(g, 'a', `grouped 1×1 (g=${gr})`, { cin: Cs, cout: mid, k: 1, gr, H, col: 1, tags: 'BN · ReLU', kind: 'conv' });
    g.node('sf', 'channel shuffle', { kind: 'pool', col: 2, inn: { C: mid, H }, out: { C: mid, H }, note: 'free: only reorders channels' });
    cnode(g, 'b', 'depthwise 3×3', { cin: mid, cout: mid, k: 3, gr: mid, H, col: 3, tags: 'BN' }); cnode(g, 'c', `grouped 1×1 (g=${gr})`, { cin: mid, cout: Cs, k: 1, gr, H, col: 4, tags: 'BN', kind: 'conv' });
    g.node('add', 'add', { kind: 'merge', op: '+', col: 5, inn: { C: Cs, H }, out: { C: Cs, H }, tags: 'ReLU' }); io(g, 'y', 'output', Cs, H, 6);
    ['x>a', 'a>sf', 'sf>b', 'b>c', 'c>add', 'add>y'].forEach((s) => g.edge(...s.split('>'))); g.edge('x', 'add', { lane: -1.2, label: 'identity', dash: true }); return g; } };
  CVA.BLOCKS = B;
  CVA.blockStats = (g) => ({ params: g.nodes.reduce((a, n) => a + n.params, 0), macs: g.nodes.reduce((a, n) => a + n.macs, 0) });

  /* ---------------- custom models (same block format as AIArch.build) ---------------- */
  const wdepth = (name) => /^(conv|depthwise|pointwise|expand|project|fully|FC|layer \d+: (1×1|3×3)|stem|patchify|1×1|3×3|5×5)/.test(name) && !/skip/.test(name);
  function finish(blocks) {
    const t = { params: 0, macs: 0, act: 0, peak: 0, depth: 0 };
    blocks.forEach((b) => { t.params += b.params; t.macs += b.macs; const a = (b.out.C || b.out.D) * ((b.out.H || 1) ** 2) * (b.rep || 1); b.act = a; t.act += a; t.peak = Math.max(t.peak, a); t.depth += b.depth || 0; });
    return { blocks, totals: t, errors: [] };
  }
  function blk(name, out, sub, o) { const p = sub.reduce((a, x) => a + x.params, 0), m = sub.reduce((a, x) => a + x.macs, 0); return Object.assign({ name, out, outStr: out.H ? sh(out) : String(out.C), params: p, macs: m, sub, tags: [], type: 'conv' }, o || {}); }
  const sb = (name, out, params, macs, rf) => ({ name, out: typeof out === 'string' ? out : sh(out), params, macs, rf: null });
  CVA.googlenet = function () {
    const bl = []; let rf = 1, j = 1; const cvr = (cin, cout, k, H, s) => { const w = cin * k * k * cout; rf += (k - 1) * j; j *= s || 1; return sb(`conv ${k}×${k}${s > 1 ? ' /' + s : ''}`, { C: cout, H }, w + cout, w * H * H, rf); };
    bl.push(blk('input', { C: 3, H: 224 }, [sb('input', { C: 3, H: 224 }, 0, 0, 1)], { type: 'input' }));
    bl.push(blk('conv 7×7 /2', { C: 64, H: 112 }, [cvr(3, 64, 7, 112, 2)], { depth: 1 }));
    rf += 2 * j; j *= 2; bl.push(blk('max pool /2', { C: 64, H: 56 }, [sb('max pool 3×3 /2', { C: 64, H: 56 }, 0, 0, rf)], { type: 'maxpool' }));
    bl.push(blk('conv 1×1', { C: 64, H: 56 }, [cvr(64, 64, 1, 56)], { depth: 1 })); bl.push(blk('conv 3×3', { C: 192, H: 56 }, [cvr(64, 192, 3, 56)], { depth: 1 }));
    rf += 2 * j; j *= 2; bl.push(blk('max pool /2', { C: 192, H: 28 }, [sb('max pool 3×3 /2', { C: 192, H: 28 }, 0, 0, rf)], { type: 'maxpool' }));
    const mods = [['3a', 28, 192, 64, 96, 128, 16, 32, 32], ['3b', 28, 256, 128, 128, 192, 32, 96, 64], ['pool', 14], ['4a', 14, 480, 192, 96, 208, 16, 48, 64], ['4b', 14, 512, 160, 112, 224, 24, 64, 64], ['4c', 14, 512, 128, 128, 256, 24, 64, 64], ['4d', 14, 512, 112, 144, 288, 32, 64, 64], ['4e', 14, 528, 256, 160, 320, 32, 128, 128], ['pool', 7], ['5a', 7, 832, 256, 160, 320, 32, 128, 128], ['5b', 7, 832, 384, 192, 384, 48, 128, 128]];
    let lastC = 192;
    mods.forEach(([n, H, i, a, b, c, d, e, f]) => {
      if (n === 'pool') { rf += 2 * j; j *= 2; bl.push(blk('max pool /2', { C: lastC, H }, [sb('max pool 3×3 /2', { C: lastC, H }, 0, 0, rf)], { type: 'maxpool' })); return; }
      const cv = (ci, co, k) => { const w = ci * k * k * co; return [w + co, w * H * H]; }, rows = [['1×1 branch', i, a, 1], ['3×3 reduce 1×1', i, b, 1], ['3×3 branch', b, c, 3], ['5×5 reduce 1×1', i, d, 1], ['5×5 branch', d, e, 5], ['pool proj 1×1', i, f, 1]];
      const sub = rows.map(([nm, ci, co, k]) => { const [p, m] = cv(ci, co, k); return sb(nm + ` (${ci}→${co})`, { C: co, H }, p, m, k === 1 ? null : rf + (k - 1) * j); }); sub.push(sb('concat 4 branches', { C: a + c + e + f, H }, 0, 0));
      bl.push(blk('inception ' + n, { C: a + c + e + f, H }, sub, { depth: 2, type: 'inception' })); lastC = a + c + e + f; rf += 4 * j;
    });
    bl.push(blk('global pool', { C: 1024, H: 1 }, [sb('global average pool', { C: 1024, H: 1 }, 0, 0, rf)], { type: 'gap' })); bl.push(blk('FC 1000', { C: 1000, H: 1 }, [sb('fully connected 1024→1000', '1000', 1024001 - 1 + 1000 - 1000 + 0, 1024000)], { type: 'fc', depth: 1 }));
    bl[bl.length - 1].sub[0].params = 1024 * 1000 + 1000; bl[bl.length - 1].params = 1024 * 1000 + 1000; const r = finish(bl); return r;
  };
  CVA.convnext = function () {
    const bl = [], dims = [96, 192, 384, 768], deps = [3, 3, 9, 3]; let H = 56, rf = 4, j = 4;
    bl.push(blk('input', { C: 3, H: 224 }, [sb('input', { C: 3, H: 224 }, 0, 0, 1)], { type: 'input' }));
    bl.push(blk('stem 4×4 /4 + LN', { C: 96, H: 56 }, [sb('patchify conv 4×4 /4 (3→96)', { C: 96, H: 56 }, 3 * 16 * 96 + 96, 3 * 16 * 96 * 56 * 56, 4), sb('LayerNorm', { C: 96, H: 56 }, 192, 0)], { depth: 1 }));
    dims.forEach((C, s) => {
      if (s > 0) { const ci = dims[s - 1]; H /= 2; rf += j; j *= 2; bl.push(blk(`downsample 2×2 /2`, { C, H }, [sb('LayerNorm', { C: ci, H: H * 2 }, 2 * ci, 0), sb(`conv 2×2 /2 (${ci}→${C})`, { C, H }, ci * 4 * C + C, ci * 4 * C * H * H, rf)], { depth: 1 })); }
      const n = deps[s], pb = 49 * C + C + 2 * C + C * 4 * C + 4 * C + 4 * C * C + C + C, mb = (49 * C + 4 * C * C + 4 * C * C) * H * H; rf += 6 * j * n;
      bl.push(blk(`stage ${s + 1}: ${n}× ConvNeXt block`, { C, H }, [sb(`dw 7×7 ×${n}`, { C, H }, n * (49 * C + C), n * 49 * C * H * H, rf), sb(`LayerNorm + 1×1 ${C}→${4 * C} + GELU ×${n}`, { C: 4 * C, H }, n * (2 * C + C * 4 * C + 4 * C), n * C * 4 * C * H * H), sb(`1×1 ${4 * C}→${C} + layer-scale ×${n}`, { C, H }, n * (4 * C * C + C + C), n * 4 * C * C * H * H)], { depth: 3 * n, type: 'convnext', rep: 1 }));
      const last = bl[bl.length - 1]; last.params = n * pb; last.macs = n * mb;
    });
    bl.push(blk('global pool + LN', { C: 768, H: 1 }, [sb('global average pool', { C: 768, H: 1 }, 0, 0, rf), sb('LayerNorm', { C: 768, H: 1 }, 1536, 0)], { type: 'gap' }));
    bl.push(blk('FC 1000', { C: 1000, H: 1 }, [sb('fully connected 768→1000', '1000', 768 * 1000 + 1000, 768 * 1000)], { type: 'fc', depth: 1 }));
    const r = finish(bl); return r;
  };
  CVA.effnet = function () {
    const bl = [], cfg = [[1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5], [6, 80, 3, 2, 3], [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]]; let H = 112, cin = 32, rf = 3, j = 2;
    bl.push(blk('input', { C: 3, H: 224 }, [sb('input', { C: 3, H: 224 }, 0, 0, 1)], { type: 'input' }));
    bl.push(blk('stem conv 3×3 /2', { C: 32, H: 112 }, [sb('conv 3×3 /2 + BN', { C: 32, H: 112 }, 3 * 9 * 32 + 64, 3 * 9 * 32 * 112 * 112, 3)], { depth: 1 }));
    cfg.forEach(([t, c, n, s, k], si) => {
      let p = 0, m = 0; const Hin = H;
      for (let i = 0; i < n; i++) { const st = i ? 1 : s, e = cin * t, Ho = st === 2 ? H / 2 : H, sq = Math.max(1, Math.floor(cin / 4)); if (t !== 1) { p += cin * e + 2 * e; m += cin * e * H * H; } p += e * k * k + 2 * e; m += e * k * k * Ho * Ho; p += e * sq + sq + sq * e + e; m += e * sq + sq * e; p += e * c + 2 * c; m += e * c * Ho * Ho; if (st === 2) { rf += (k - 1) * j; j *= 2; } rf += (k - 1) * j; H = Ho; cin = c; }
      bl.push(blk(`stage ${si + 1}: ${n}× MBConv${t} k${k}`, { C: c, H }, [sb(`MBConv ×${n} (expand ${t}, dw ${k}×${k}, SE, project)`, { C: c, H }, p, m, rf)], { depth: (t === 1 ? 2 : 3) * n, type: 'invres', params: p, macs: m }));
    });
    bl.push(blk('head conv 1×1', { C: 1280, H: 7 }, [sb('conv 1×1 320→1280 + BN', { C: 1280, H: 7 }, 320 * 1280 + 2560, 320 * 1280 * 49, rf)], { depth: 1 }));
    bl.push(blk('global pool', { C: 1280, H: 1 }, [sb('global average pool', { C: 1280, H: 1 }, 0, 0, rf)], { type: 'gap' }));
    bl.push(blk('FC 1000', { C: 1000, H: 1 }, [sb('fully connected 1280→1000', '1000', 1281000, 1280000)], { type: 'fc', depth: 1 }));
    const r = finish(bl); return r;
  };
  // published ImageNet top-1 (single crop, torchvision / original papers; training recipes differ)
  CVA.TOP1 = { lenet5: null, alexnet: 56.5, vgg16: 71.6, googlenet: 69.8, resnet18: 69.8, resnet50: 76.1, densenet121: 74.4, mobilenetv1: 70.6, mobilenetv2: 71.9, 'efficientnet-b0': 77.1, 'convnext-t': 82.1 };
  CVA.TITLES = { googlenet: 'GoogLeNet (2014)', 'efficientnet-b0': 'EfficientNet-B0 (2019)', 'convnext-t': 'ConvNeXt-T (2022)' };
  CVA.model = function (name) {
    if (name === 'googlenet') return CVA.googlenet(); if (name === 'convnext-t') return CVA.convnext(); if (name === 'efficientnet-b0') return CVA.effnet();
    const A = root.AIArch, r = A.build(A.PRESETS[name].spec); r.totals.depth = depthOf(r); return r;
  };
  function depthOf(r) { let d = 0; r.blocks.forEach((b) => { if (b.type === 'dense') d += 2 * b.sub.filter((s) => /1×1/.test(s.name)).length; else b.sub.forEach((s) => { if (/^(conv|depthwise|pointwise|expand|project|fully)/.test(s.name)) d++; }); }); return d; }
  CVA.depthOf = depthOf;
  CVA.FAMILY = [['B0', 5.3, 0.39, 77.1, 1.0, 1.0, 224], ['B1', 7.8, 0.70, 79.1, 1.0, 1.1, 240], ['B2', 9.2, 1.0, 80.1, 1.1, 1.2, 260], ['B3', 12, 1.8, 81.6, 1.2, 1.4, 300], ['B4', 19, 4.2, 82.9, 1.4, 1.8, 380], ['B5', 30, 9.9, 83.6, 1.6, 2.2, 456], ['B6', 43, 19, 84.0, 1.8, 2.6, 528], ['B7', 66, 37, 84.3, 2.0, 3.1, 600]];

  if (typeof module !== 'undefined') module.exports = CVA;
  root.CVA = CVA;
  if (typeof window === 'undefined') return;

  /* =================== UI helpers =================== */
  const $ = (s, r) => AI.$(s, r);
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const KCOL = { conv: 'blue', dw: 'teal', pool: 'red', fc: 'orange', norm: 'purple', merge: 'green', io: 'muted' };
  const rrect = (g, x, y, w, h, r) => { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };
  const playBtns = '<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="prev">◀ Step</button><button class="btn" id="next">Step ▶</button></div>';
  function playCtl(root, st) { const q = (s) => $(s, root); q('#play').onclick = () => { st.playing = !st.playing; q('#play').textContent = st.playing ? '⏸ Pause' : '▶ Play'; }; const stop = () => { st.playing = false; q('#play').textContent = '▶ Play'; }; return { stop, prev: (f) => (q('#prev').onclick = () => { stop(); f(-1); }), next: (f) => (q('#next').onclick = () => { stop(); f(1); }) }; }

  /* =================== blockflow =================== */
  AI.labs.blockflow = function (root) {
    const d = root.dataset, keys = (d.blocks || d.block || 'basic').split(',').map((s) => s.trim());
    let key = d.block || keys[0], cfg = {}, G0 = null, st = { playing: true }, T = 0, sel = '', lastRows = '';
    const { view, ctl } = AI.labShell(root, d.title || 'Block diagram', 'data flows through the block; tensor shape on every edge',
      '<div class="arch-scroll"><canvas style="height:330px"></canvas></div><div class="caption" id="cap" style="min-height:40px"></div><div id="tbl" style="overflow-x:auto"></div>',
      `<label>Block <select id="blk">${keys.map((k) => `<option value="${k}" ${k === key ? 'selected' : ''}>${B[k].title}</option>`).join('')}</select></label>
       <div id="dyn" style="display:flex;flex-direction:column;gap:8px"></div>
       ${sl('sp', 'Speed <small>(stages per second)</small>', 0.1, 1.5, 0.05, 0.4)}${playBtns}<div class="readout" id="sum"></div>`);
    const q = (s) => $(s, root), cv = $('canvas', view), c = AI.canvas(cv), pc = playCtl(root, st);
    const over = {}; (d.cfg || '').split(';').filter(Boolean).forEach((p) => { const [k, v] = p.split('='); over[k.trim()] = +v; });
    let base = { C: +(d.c || d.C || 64), H: +(d.h || d.H || 56) };
    const sliders = () => {
      const bd = B[key], vals = Object.assign({}, bd.def, over); over.__ = 0;
      const defs = [['C', 'C_in', 4, 1024, 4], ['H', 'H = W', 7, 112, 7]].concat(bd.ctl);
      q('#dyn').innerHTML = defs.map(([k, l, mn, mx, sp]) => sl('s_' + k, l, mn, mx, sp, k === 'C' || k === 'H' ? base[k] : (vals[k] !== undefined ? vals[k] : bd.def[k]))).join('') + `<div class="caption">${bd.note}</div>`;
      defs.forEach(([k]) => { const e = q('#s_' + k); e.oninput = rebuild; });
      rebuild();
    };
    const rebuild = () => {
      const bd = B[key]; cfg = {}; ['C', 'H'].concat(bd.ctl.map((x) => x[0])).forEach((k) => { const e = q('#s_' + k); cfg[k] = +e.value; q('#s_' + k + 'v').textContent = cfg[k]; });
      base = { C: cfg.C, H: cfg.H }; G0 = bd.build(cfg); T = 0; lastRows = '';
      const stt = CVA.blockStats(G0), out = G0.by('y').out;
      q('#sum').innerHTML = `<span class="m">whole block</span>\nin  <b>${sh({ C: cfg.C, H: cfg.H })}</b>\nout <b>${sh(out)}</b>\nparams <b class="o">${com(stt.params)}</b>\nMACs   <b class="b">${com(stt.macs)}</b>`;
      layout(); tbl();
    };
    let geo = {}, maxCol = 1, W = 0, Hh = 0;
    const CS = 184, RS = 72;
    const layout = () => {
      maxCol = Math.max(...G0.nodes.map((n) => n.col)); const rows = G0.nodes.map((n) => n.row).concat(G0.edges.filter((e) => e.lane !== undefined).map((e) => e.lane)); const rmin = Math.min(0, ...rows), rmax = Math.max(0, ...rows);
      const arcs = Math.max(0, ...G0.edges.map((e) => e.arc || 0));
      W = 60 + (maxCol + 1) * CS; Hh = Math.max(250, 50 + (rmax - rmin) * RS + 90 + arcs); cv.style.height = Hh + 'px'; cv.style.width = Math.max(AI.$('.arch-scroll', root).clientWidth - 16, W) + 'px';
      const y0 = 40 + arcs + (-rmin) * RS + 20;
      geo = {}; G0.nodes.forEach((n) => { const w = n.kind === 'io' ? 92 : n.kind === 'merge' ? 34 : 112, h = n.kind === 'merge' ? 34 : 54; geo[n.id] = { x: 70 + n.col * CS, y: y0 + n.row * RS, w, h, n }; });
      geo._y0 = y0;
    };
    const path = (e) => {
      const A = geo[e.a], Bn = geo[e.b], pts = [];
      if (e.arc) { const x0 = A.x, y0 = A.y - A.h / 2, x1 = Bn.x, y1 = Bn.y - Bn.h / 2, cy = Math.min(y0, y1) - e.arc * 1.6; for (let i = 0; i <= 24; i++) { const t = i / 24, u = 1 - t; pts.push([u * u * x0 + 2 * u * t * ((x0 + x1) / 2) + t * t * x1, u * u * y0 + 2 * u * t * cy + t * t * y1]); } }
      else if (e.lane !== undefined) { const ly = geo._y0 + e.lane * RS, xs = A.x + A.w / 2; pts.push([xs, A.y], [xs + 14, A.y], [xs + 14, ly], [Bn.x, ly], [Bn.x, Bn.y - Bn.h / 2]); }
      else { const x0 = A.x + A.w / 2, y0 = A.y, x1 = Bn.x - Bn.w / 2, y1 = Bn.y; if (Math.abs(y0 - y1) < 2) pts.push([x0, y0], [x1, y1]); else for (let i = 0; i <= 20; i++) { const t = i / 20, s = AI.smooth(t); pts.push([AI.lerp(x0, x1, t), AI.lerp(y0, y1, s)]); } }
      let L = 0; const cum = [0]; for (let i = 1; i < pts.length; i++) { L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); cum.push(L); }
      return { pts, cum, L };
    };
    const at = (p, f) => { const t = f * p.L; let i = 1; while (i < p.cum.length - 1 && p.cum[i] < t) i++; const s = (t - p.cum[i - 1]) / Math.max(1e-6, p.cum[i] - p.cum[i - 1]); return [AI.lerp(p.pts[i - 1][0], p.pts[i][0], s), AI.lerp(p.pts[i - 1][1], p.pts[i][1], s)]; };
    const tbl = () => {
      q('#tbl').innerHTML = `<table class="tbl" style="font-size:12px"><tr><th>layer</th><th>in → out</th><th>how the parameters are counted</th><th>params</th><th>MACs</th></tr>${G0.nodes.filter((n) => n.params || n.macs || n.kind === 'conv' || n.kind === 'fc').map((n) => `<tr data-id="${n.id}"><td>${n.label}${n.tags ? ' <small style="color:var(--muted)">' + n.tags + '</small>' : ''}</td><td class="mono">${n.inn ? sh(n.inn) : ''} → ${sh(n.out)}</td><td class="mono">${n.formula}</td><td>${com(n.params)}</td><td>${fmtN(n.macs)}</td></tr>`).join('')}</table>`;
    };
    q('#blk').onchange = () => { key = q('#blk').value; sliders(); };
    pc.prev((s) => (T = Math.max(0, Math.floor(T - 0.01) + s < 0 ? 0 : Math.floor(T - 0.01) + s))); pc.next((s) => (T = Math.min(maxCol + 1, Math.floor(T) + 1)));
    cv.addEventListener('click', (ev) => { const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top; sel = ''; for (const id in geo) { const g = geo[id]; if (id[0] !== '_' && Math.abs(x - g.x) < g.w / 2 && Math.abs(y - g.y) < g.h / 2) sel = id; } });
    sliders();
    new ResizeObserver(() => G0 && layout()).observe(AI.$('.arch-scroll', root));
    AI.loop(root, (dt) => {
      if (!G0) return; const tmax = maxCol + 1.6;
      if (st.playing) { T += dt * +q('#sp').value; if (T > tmax + 0.8) T = 0; }
      q('#spv').textContent = (+q('#sp').value).toFixed(2); const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      // edges
      G0.edges.forEach((e) => {
        const p = path(e), A = G0.by(e.a), Bn = G0.by(e.b), dep = A.col + 0.5, arr = Bn.col, f = AI.clamp((T - dep) / Math.max(0.3, arr - dep), 0, 1);
        g.strokeStyle = C.border; g.lineWidth = 2; g.setLineDash(e.dash ? [6, 5] : []); g.beginPath(); p.pts.forEach((q2, i) => (i ? g.lineTo(q2[0], q2[1]) : g.moveTo(q2[0], q2[1]))); g.stroke();
        if (f > 0) { g.strokeStyle = C.blue; g.lineWidth = 3; g.beginPath(); const n = Math.max(2, Math.ceil(f * p.pts.length)); for (let i = 0; i < n; i++) i ? g.lineTo(p.pts[i][0], p.pts[i][1]) : g.moveTo(p.pts[0][0], p.pts[0][1]); g.stroke(); }
        g.setLineDash([]);
        if (f > 0 && f < 1) { const [x, y] = at(p, f); g.shadowColor = C.fg; g.shadowBlur = 12; g.fillStyle = C.fg; g.beginPath(); g.arc(x, y, 5.5, 0, 7); g.fill(); g.shadowBlur = 0; }
        if (!e.nolabel || e.last) { const [x, y] = at(p, e.lane !== undefined ? 0.5 : 0.5); g.fillStyle = C.muted; g.font = '10.5px ui-monospace,Menlo,monospace'; g.textAlign = 'center'; const lb = (e.label ? e.label + ' · ' : '') + sh(A.out); g.fillText(lb, x, y - 6); }
      });
      // nodes
      const act = [];
      G0.nodes.forEach((n) => {
        const G2 = geo[n.id], on = T >= n.col && T < n.col + 0.5, reached = T >= n.col; if (on) act.push(n);
        const col = C[KCOL[n.kind] || 'blue']; g.globalAlpha = reached ? 1 : 0.4;
        if (on) { g.shadowColor = col; g.shadowBlur = 18; }
        if (n.kind === 'merge') { g.fillStyle = col; g.beginPath(); g.arc(G2.x, G2.y, 17, 0, 7); g.fill(); g.shadowBlur = 0; g.fillStyle = '#fff'; g.font = '700 18px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(n.op, G2.x, G2.y + 1); g.textBaseline = 'alphabetic'; g.fillStyle = C.muted; g.font = '10.5px sans-serif'; g.fillText(n.label + (n.tags ? ' · ' + n.tags : ''), G2.x, G2.y + 32); g.fillText(sh(n.out), G2.x, G2.y + 44); }
        else {
          rrect(g, G2.x - G2.w / 2, G2.y - G2.h / 2, G2.w, G2.h, 8); g.fillStyle = col; g.globalAlpha *= on ? 0.95 : 0.28; g.fill(); g.globalAlpha = reached ? 1 : 0.4; g.shadowBlur = 0; g.lineWidth = on || sel === n.id ? 2.5 : 1; g.strokeStyle = on || sel === n.id ? C.fg : col; g.stroke();
          g.fillStyle = C.fg; g.textAlign = 'center'; g.font = '600 11.5px sans-serif'; g.fillText(n.label, G2.x, G2.y - 8); g.font = '10.5px ui-monospace,Menlo,monospace'; g.fillText(sh(n.out), G2.x, G2.y + 5); if (n.params) { g.fillStyle = C.orange; g.fillText(fmtN(n.params) + ' params', G2.x, G2.y + 18); } else if (n.tags) { g.fillStyle = C.muted; g.fillText(n.tags, G2.x, G2.y + 18); }
          if (n.params && n.tags) { g.fillStyle = C.muted; g.font = '9.5px sans-serif'; g.fillText(n.tags, G2.x, G2.y + G2.h / 2 + 11); }
        }
        g.globalAlpha = 1;
      });
      const cur = sel ? [G0.by(sel)] : act;
      q('#cap').innerHTML = cur.length ? cur.map((n) => `<b>${n.label}</b>: ${n.inn && n.kind !== 'io' ? sh(n.inn) + ' → ' : ''}${sh(n.out)}${n.kind === 'merge' ? (n.op === '+' ? ' — element-wise addition of the two incoming tensors (same shape).' : n.op === '‖' ? ' — channels of all incoming tensors are stacked: ' + G0.edges.filter((e) => e.b === n.id).map((e) => G0.by(e.a).out.C).join(' + ') + ' = ' + n.out.C + '.' : ' — each channel of the feature map is multiplied by its gate in (0,1).') : n.macs ? ` — ${com(n.params)} params, ${fmtN(n.macs)} MACs.` : n.note ? ' — ' + n.note : ''}`).join('<br>') : (T < 0.05 ? 'The input tensor enters the block.' : 'Data travels along the edges; the white dot is a tensor in flight.');
      q('#tbl').querySelectorAll('tr[data-id]').forEach((r) => { const on = cur.some((n) => n.id === r.dataset.id); r.style.background = on ? 'var(--bg2)' : ''; });
    }, { fps: 30 });
  };

  /* =================== gradflow =================== */
  AI.labs.gradflow = function (root) {
    const { view, ctl } = AI.labShell(root, d0(root, 'Gradient flow: plain vs residual'), 'a backward pass in a toy random-Jacobian network — click a residual block to cut its skip',
      '<canvas style="height:400px;cursor:pointer"></canvas><div class="caption" id="cap" style="min-height:40px"></div>',
      `${sl('L', 'Depth L (blocks)', 6, 56, 1, 24)}${sl('j', 'Branch gain j <small>(how much one layer scales a gradient)</small>', 0.3, 1.3, 0.05, 0.7)}${sl('s', 'Residual branch scale s <small>(zero-init trick)</small>', 0, 1, 0.05, 0.5)}${sl('sp', 'Speed <small>(blocks per second)</small>', 1, 12, 0.5, 4)}
       ${playBtns}<div class="row"><button class="btn" id="re">Restore all skips</button></div><div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), st = { playing: true }, pc = playCtl(root, st);
    let L = 24, cut = new Set(), wave = 0, cur = { p: [], r: [] }, key = '';
    const u = AI.rng(7), noise = [...Array(60)].map(() => u());
    const calc = () => {
      L = +q('#L').value; const j = +q('#j').value, s = +q('#s').value; const p = [1], r = [1];
      for (let k = 1; k <= L; k++) { const jl = j * (0.8 + 0.4 * noise[k - 1]); p.push(p[k - 1] * jl); r.push(r[k - 1] * (cut.has(L - k) ? jl : Math.sqrt(1 + (s * jl) ** 2))); }
      cur = { p, r }; // index k = number of blocks the gradient has travelled back through
    };
    const redraw = () => { q('#Lv').textContent = q('#L').value; q('#jv').textContent = q('#j').value; q('#sv').textContent = q('#s').value; q('#spv').textContent = q('#sp').value; cut = new Set([...cut].filter((i) => i < +q('#L').value)); calc(); key = ''; };
    ['#L', '#j', '#s'].forEach((s) => q(s).addEventListener('input', () => { redraw(); }));
    q('#re').onclick = () => { cut.clear(); redraw(); };
    pc.prev(() => (wave = Math.max(0, Math.floor(wave) - 1))); pc.next(() => (wave = Math.min(L, Math.floor(wave) + 1)));
    const geom = () => { const m = 46, w = (c.w - m - 14) / L; return { m, w }; };
    c.cv.addEventListener('click', (e) => { const r = c.cv.getBoundingClientRect(), { m, w } = geom(), i = Math.floor((e.clientX - r.left - m) / w); if (i >= 0 && i < L) { cut.has(i) ? cut.delete(i) : cut.add(i); calc(); } });
    redraw();
    AI.loop(root, (dt) => {
      if (st.playing) { wave += dt * +q('#sp').value; if (wave > L + 3) wave = 0; }
      const C = AI.col(), g = c.ctx, { m, w } = geom(); g.clearRect(0, 0, c.w, c.h);
      const lo = -8, hi = 4, rowH = 150, ys = [34, 34 + rowH + 40];
      [['plain network (no skips)', cur.p, 0], ['residual network  y = F(x) + x  (click a block to cut its skip)', cur.r, 1]].forEach(([title, arr, ri]) => {
        const y0 = ys[ri]; g.fillStyle = C.fg; g.font = '600 12px sans-serif'; g.textAlign = 'left'; g.fillText(title, m, y0 - 12);
        const Y = (v) => y0 + 44 + (1 - (AI.clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo)) * (rowH - 52);
        g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '10px ui-monospace,monospace'; g.textAlign = 'right';
        for (let e = lo; e <= hi; e += 4) { g.beginPath(); g.moveTo(m, Y(10 ** e)); g.lineTo(c.w - 10, Y(10 ** e)); g.stroke(); g.fillText('1e' + e, m - 4, Y(10 ** e) + 3); }
        g.strokeStyle = C.muted; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(m, Y(1)); g.lineTo(c.w - 10, Y(1)); g.stroke(); g.setLineDash([]);
        for (let i = 0; i < L; i++) { // block i (0 = first, near input); gradient arriving at its input has travelled L-i blocks back
          const x = m + i * w, k = L - i, v = arr[k], reached = wave >= k - 0.0 ? 1 : 0, frac = AI.clamp(wave - (k - 1), 0, 1);
          const isCut = ri === 1 && cut.has(i);
          g.fillStyle = ri === 1 && !isCut ? C.green : C.muted; g.globalAlpha = 0.9; g.fillRect(x + 1, y0 + 6, Math.max(2, w - 2), 14); g.globalAlpha = 1;
          if (ri === 1) { g.strokeStyle = isCut ? C.red : C.green; g.lineWidth = isCut ? 2 : 1.5; g.setLineDash(isCut ? [3, 3] : []); g.beginPath(); g.moveTo(x + 1, y0 + 6); g.bezierCurveTo(x + 1, y0 - 6, x + w - 1, y0 - 6, x + w - 1, y0 + 6); g.stroke(); g.setLineDash([]); g.lineWidth = 1; }
          const vv = Math.pow(10, AI.lerp(0, Math.log10(Math.max(v, 1e-12)), AI.smooth(frac))), top = Y(vv), base = Y(10 ** lo);
          if (frac > 0) { g.fillStyle = v < 1e-3 ? C.red : v > 1e3 ? C.orange : C.blue; g.fillRect(x + 2, top, Math.max(1.5, w - 4), base - top); }
        }
        // wave front marker
        const xf = m + (L - wave) * w; if (wave > 0 && wave < L) { g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); g.moveTo(xf, y0 + 2); g.lineTo(xf, y0 + rowH); g.stroke(); g.lineWidth = 1; }
      });
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('input', m, c.h - 4); g.textAlign = 'right'; g.fillText('loss → gradient starts here (=1)', c.w - 10, c.h - 4);
      const k = Math.min(L, Math.floor(wave)), pk = Math.floor(wave) >= L ? cur.p[L] : cur.p[L], rk = cur.r[L];
      const kk = `${L}|${cut.size}|${q('#j').value}|${q('#s').value}`; if (kk !== key) { key = kk; q('#out').innerHTML = `gradient reaching the first block\nplain      <b class="r">${cur.p[L].toExponential(2)}</b>\nresidual   <b class="g">${rk.toExponential(2)}</b>\nskips cut: ${cut.size}`; }
      q('#cap').innerHTML = `Reading the chart: bar height = size of ∂L/∂x at the input of each block (log scale, loss gradient = 1). The yellow line is the backward wave moving from the loss toward the input. In the plain network every layer multiplies the gradient by a factor ${AI.fmt(+q('#j').value, 2)}-ish, so it shrinks like j<sup>L</sup>. With the skip, ∂L/∂x = ∂L/∂y·(1 + ∂F/∂x) keeps the "1": the gradient has a direct highway. Cut a skip and watch the highway break there.`;
    }, { fps: 30 });
  };
  function d0(root, t) { return root.dataset.title || t; }

  /* =================== convcost =================== */
  AI.labs.convcost = function (root) {
    const { view, ctl } = AI.labShell(root, d0(root, 'Convolution cost calculator'), 'standard vs grouped vs depthwise-separable vs inverted residual',
      '<canvas style="height:300px"></canvas><div class="caption" id="cap"></div>',
      `${sl('ci', 'C_in', 8, 512, 8, 128)}${sl('co', 'C_out', 8, 512, 8, 128)}${sl('k', 'kernel k', 1, 7, 2, 3)}${sl('H', 'H = W', 7, 112, 7, 56)}${sl('g', 'groups g (grouped conv)', 1, 64, 1, 4)}${sl('t', 'expansion t (inverted residual)', 1, 8, 1, 6)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="lg"> log scale</label><div class="row"><button class="btn primary" id="sweep">▶ Sweep C_out 8 → 512</button></div><div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)); let shown = [1, 1, 1, 1], sweep = 0;
    const comp = () => {
      const ci = +q('#ci').value, co = +q('#co').value, k = +q('#k').value, H = +q('#H').value, t = +q('#t').value; let gg = +q('#g').value; while ((ci % gg || co % gg) && gg > 1) gg--;
      const std = k * k * ci * co, grp = std / gg, dw = k * k * ci, pw = ci * co, e = ci * t, inv = (t === 1 ? 0 : ci * e) + k * k * e + e * co;
      return { ci, co, k, H, t, gg, rows: [['standard k×k', std], [`grouped (g=${gg})`, grp], ['depthwise-separable', dw + pw], [`inverted residual (t=${t})`, inv]], std, dw, pw };
    };
    AI.loop(root, (dt) => {
      if (sweep) { const v = +q('#co').value + dt * 40; if (v >= 512) { sweep = 0; q('#sweep').textContent = '▶ Sweep C_out 8 → 512'; } else q('#co').value = v; }
      ['ci', 'co', 'k', 'H', 'g', 't'].forEach((i) => (q('#' + i + 'v').textContent = q('#' + i).value));
      const r = comp(), C = AI.col(), g = c.ctx, log = q('#lg').checked; g.clearRect(0, 0, c.w, c.h);
      const cols = ['blue', 'purple', 'teal', 'orange'], mx = Math.max(...r.rows.map((x) => x[1])), bw = c.w - 40;
      r.rows.forEach(([n, v], i) => {
        const tgt = log ? Math.log10(v + 1) / Math.log10(mx + 1) : v / r.std; shown[i] += (tgt - shown[i]) * Math.min(1, dt * 5);
        const y = 30 + i * 62; g.fillStyle = C.fg; g.font = '600 12px sans-serif'; g.textAlign = 'left'; g.fillText(n, 14, y - 6); g.font = '11px ui-monospace,monospace'; g.fillStyle = C.muted; g.fillText(com(v) + ' weights · ' + fmtN(v * r.H * r.H) + ' MACs · ' + (v / r.std * 100).toFixed(1) + '% of standard', 14 + 200, y - 6);
        const wv = Math.max(2, Math.min(1.3, shown[i]) * (log ? bw : bw / 1.3)); g.fillStyle = C[cols[i]]; g.fillRect(14, y, wv, 24);
        
      });
      const ratio = r.rows[2][1] / r.std; q('#out').innerHTML = `ratio dw-sep / standard\nmeasured  <b class="o">${ratio.toFixed(4)}</b>\n1/C_out + 1/k² = ${(1 / r.co).toFixed(4)} + ${(1 / (r.k * r.k)).toFixed(4)}\n          = <b class="g">${(1 / r.co + 1 / (r.k * r.k)).toFixed(4)}</b>  (≈ ${(1 / ratio).toFixed(1)}× cheaper)\nstandard MACs ${fmtN(r.std * r.H * r.H)}`;
      q('#cap').textContent = log ? 'Log scale: bars show orders of magnitude.' : 'Bars are relative to the standard convolution (100%). Depthwise + pointwise: the k² and the C_out terms separate, so the cost becomes a sum instead of a product.';
    }, { fps: 30 });
    q('#sweep').onclick = () => { sweep = 1; q('#co').value = 8; q('#sweep').textContent = '… sweeping'; };
  };

  /* =================== relumanifold =================== */
  AI.labs.relumanifold = function (root) {
    const { view, ctl } = AI.labShell(root, d0(root, 'Why the projection is linear: ReLU destroys low-dimensional information'), 'embed a 2-D curve in N dimensions, apply ReLU, try to recover it',
      '<canvas style="height:300px"></canvas><div class="caption" id="cap"></div>',
      `${sl('n', 'Expansion dimension N', 2, 60, 1, 3)}${sl('sp', 'Speed', 0.1, 1.5, 0.05, 0.35)}<div class="row"><button class="btn" id="seed">New random embedding</button></div>${playBtns}<div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)), st = { playing: true }, pc = playCtl(root, st);
    let T = 0, seed = 3, data = null;
    const build = () => {
      const N = +q('#n').value, r = AI.rng(seed), M = [...Array(N)].map(() => [AI.randn(r), AI.randn(r)]), bias = M.map(() => 0);
      const pts = [...Array(120)].map((_, i) => { const t = i / 119, a = 1.2 + t * 4.2, rad = 0.35 + 0.65 * t; return [rad * Math.cos(a) * 1.6, rad * Math.sin(a) * 1.6]; });
      let a = 0, b = 0, cc = 0; M.forEach((m) => { a += m[0] * m[0]; b += m[0] * m[1]; cc += m[1] * m[1]; }); const det = a * cc - b * b;
      const rec = pts.map((p) => { const y = M.map((m) => Math.max(0, m[0] * p[0] + m[1] * p[1])); let s0 = 0, s1 = 0; y.forEach((v, i) => { s0 += M[i][0] * v; s1 += M[i][1] * v; }); return [(cc * s0 - b * s1) / det, (-b * s0 + a * s1) / det]; });
      const lin = pts.map((p) => { const y = M.map((m) => m[0] * p[0] + m[1] * p[1]); let s0 = 0, s1 = 0; y.forEach((v, i) => { s0 += M[i][0] * v; s1 += M[i][1] * v; }); return [(cc * s0 - b * s1) / det, (-b * s0 + a * s1) / det]; });
      let err = 0, dead = 0, tot = 0; pts.forEach((p, i) => { err += (rec[i][0] - p[0]) ** 2 + (rec[i][1] - p[1]) ** 2; M.forEach((m) => { tot++; if (m[0] * p[0] + m[1] * p[1] <= 0) dead++; }); });
      data = { pts, rec, lin, N, M, rmse: Math.sqrt(err / pts.length), dead: dead / tot }; q('#out').innerHTML = `N = <b>${N}</b>\nrecovery error (RMS) <b class="${data.rmse < 0.1 ? 'g' : 'r'}">${data.rmse.toFixed(3)}</b>\nfraction of ReLU units that are 0: ${(100 * data.dead).toFixed(0)}%`;
    };
    q('#n').oninput = () => { q('#nv').textContent = q('#n').value; build(); }; q('#nv').textContent = 3; q('#seed').onclick = () => { seed++; build(); }; build();
    pc.prev(() => (T = Math.max(0, Math.floor(T - 0.01)))); pc.next(() => (T = Math.min(3, Math.floor(T) + 1)));
    AI.loop(root, (dt) => {
      if (st.playing) { T += dt * +q('#sp').value; if (T > 3.6) T = 0; } q('#spv').textContent = (+q('#sp').value).toFixed(2);
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); const pw = c.w / 3, sc = Math.min(pw, c.h) / 6.5, ph = Math.floor(T);
      const panel = (i, pts, title, hot) => { const cx = pw * i + pw / 2, cy = c.h / 2 + 8; g.strokeStyle = hot ? C.blue : C.border; g.lineWidth = hot ? 2 : 1; g.strokeRect(pw * i + 6, 8, pw - 12, c.h - 16); g.lineWidth = 1; g.fillStyle = hot ? C.fg : C.muted; g.font = '600 11.5px sans-serif'; g.textAlign = 'center'; g.fillText(title, cx, 26); pts.forEach((p, k) => { g.fillStyle = AI.mix('#2f81f7', '#f0883e', k / 119); g.beginPath(); g.arc(cx + p[0] * sc, cy - p[1] * sc, 2.6, 0, 7); g.fill(); }); };
      panel(0, data.pts, '① original x (2-D)', ph === 0); panel(1, data.pts.map((p) => [p[0] * 0 + 0, 0]).map((_, i) => [data.M[0] ? Math.max(0, data.M[0][0] * data.pts[i][0] + data.M[0][1] * data.pts[i][1]) - 1 : 0, data.M[1] ? Math.max(0, data.M[1][0] * data.pts[i][0] + data.M[1][1] * data.pts[i][1]) - 1 : 0]), `② ReLU(Tx), N=${data.N}`, ph === 1); panel(2, data.rec, '③ recovered', ph === 2);
      const f = AI.smooth(T - 1.2); if (T > 0.3 && T < 3.3) { const a = AI.clamp((T - 0.3) / 3, 0, 1); g.fillStyle = C.fg; g.beginPath(); g.arc(pw * a * 3 - 0 > c.w ? c.w : pw * (a * 3) + 0, 18, 4, 0, 7); g.fill(); }
      q('#cap').textContent = ['① We start with a 2-D curve (colour = position along it).', `② A random 1×1-style expansion T (${data.N} units) followed by ReLU zeroes every negative unit — about ${(100 * data.dead).toFixed(0)}% of them here.`, `③ Project back with the best linear map. With N=${data.N} the curve ${data.rmse < 0.1 ? 'survives' : 'is distorted: information was destroyed'} (error ${data.rmse.toFixed(2)}). Try N = 2, then 30.`, 'Lesson: ReLU is safe in a wide space, lossy in a narrow one — so MobileNetV2 projects back to the narrow space linearly.'][Math.min(3, ph)];
    }, { fps: 30 });
  };

  /* =================== scaling =================== */
  AI.labs.scaling = function (root) {
    const { view, ctl } = AI.labShell(root, d0(root, 'Compound scaling explorer'), 'one knob φ scales depth, width and resolution together',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap"></div>',
      `${sl('phi', 'φ (compound coefficient)', 0, 7, 0.1, 1)}${sl('al', 'α (depth base)', 1, 2, 0.05, 1.2)}${sl('be', 'β (width base)', 1, 1.6, 0.05, 1.1)}${sl('ga', 'γ (resolution base)', 1, 1.6, 0.05, 1.15)}
       <div>Strategy at the same MAC budget<div class="row" id="mode" style="margin-top:4px"></div></div><div class="readout" id="out"></div>`);
    const q = (s) => $(s, root), c = AI.canvas($('canvas', view)); let mode = 'compound', anim = { d: 1, w: 1, r: 1 };
    ['compound', 'depth only', 'width only', 'resolution only'].forEach((m) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = m; b.onclick = () => { mode = m; }; q('#mode').appendChild(b); });
    const F = CVA.FAMILY, interp = (macs) => { const x = Math.log(macs); if (x <= Math.log(F[0][2])) return F[0][3]; for (let i = 1; i < F.length; i++) if (x <= Math.log(F[i][2])) { const a = Math.log(F[i - 1][2]), b = Math.log(F[i][2]); return AI.lerp(F[i - 1][3], F[i][3], (x - a) / (b - a)); } return F[F.length - 1][3]; };
    AI.loop(root, (dt) => {
      ['phi', 'al', 'be', 'ga'].forEach((i) => (q('#' + i + 'v').textContent = q('#' + i).value));
      const phi = +q('#phi').value, al = +q('#al').value, be = +q('#be').value, ga = +q('#ga').value, prod = al * be * be * ga * ga, M = prod ** phi;
      let d = al ** phi, w = be ** phi, r = ga ** phi; if (mode === 'depth only') { d = M; w = 1; r = 1; } else if (mode === 'width only') { d = 1; w = Math.sqrt(M); r = 1; } else if (mode === 'resolution only') { d = 1; w = 1; r = Math.sqrt(M); }
      const k = Math.min(1, dt * 4); anim.d += (d - anim.d) * k; anim.w += (w - anim.w) * k; anim.r += (r - anim.r) * k;
      const macs = 0.39 * d * w * w * r * r, par = 5.29 * d * w * w, res = Math.round(224 * r);
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      // schematic left
      const lw = c.w * 0.48, n = Math.max(1, Math.round(6 * anim.d)), side = 40 + 38 * Math.min(anim.r, 3.2), th = 4 + 5 * Math.min(anim.w, 4);
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText(`depth × ${d.toFixed(2)} · width × ${w.toFixed(2)} · resolution × ${r.toFixed(2)} (${res}px)`, 12, 20);
      const show = Math.min(n, 22), stepx = Math.min(16, (lw - side) / Math.max(1, show));
      for (let i = 0; i < show; i++) { const x = 20 + i * stepx, y = 150 - side / 2; g.fillStyle = C.blue; g.globalAlpha = 0.55; g.fillRect(x, y, th, side); g.globalAlpha = 1; g.strokeStyle = C.fg; g.strokeRect(x, y, th, side); }
      g.fillStyle = C.muted; g.fillText(`${n} layer groups drawn (base 6)${n > show ? ' — capped' : ''}`, 12, 300);
      // chart right
      const v = AI.view(c, 0, 1, 0, 1), px = lw + 30, pw = c.w - px - 14, py = 30, ph = c.h - 80, lx = (m) => px + (Math.log10(m) - Math.log10(0.3)) / (Math.log10(60) - Math.log10(0.3)) * pw, ly = (a) => py + (1 - (a - 74) / (86 - 74)) * ph;
      g.strokeStyle = C.border; g.fillStyle = C.muted; g.font = '10px sans-serif'; g.textAlign = 'right'; for (let a = 74; a <= 86; a += 4) { g.beginPath(); g.moveTo(px, ly(a)); g.lineTo(px + pw, ly(a)); g.stroke(); g.fillText(a + '%', px - 3, ly(a) + 3); } g.textAlign = 'center'; [0.3, 1, 3, 10, 30].forEach((m) => g.fillText(m + 'G', lx(m), py + ph + 14)); g.fillText('MACs (log) →   top-1 ↑  (published B0–B7)', px + pw / 2, py + ph + 30);
      g.strokeStyle = C.muted; g.beginPath(); F.forEach((f, i) => (i ? g.lineTo(lx(f[2]), ly(f[3])) : g.moveTo(lx(f[2]), ly(f[3])))); g.stroke(); F.forEach((f) => { g.fillStyle = C.muted; g.beginPath(); g.arc(lx(f[2]), ly(f[3]), 4, 0, 7); g.fill(); g.fillStyle = C.fg; g.textAlign = 'left'; g.fillText(f[0], lx(f[2]) + 5, ly(f[3]) + 12); });
      const mm = AI.clamp(macs, 0.3, 60); g.shadowColor = C.orange; g.shadowBlur = 12; g.fillStyle = C.orange; g.beginPath(); g.arc(lx(mm), ly(interp(mm)), 7, 0, 7); g.fill(); g.shadowBlur = 0;
      q('#out').innerHTML = `α·β²·γ² = <b class="${Math.abs(prod - 2) < 0.15 ? 'g' : 'o'}">${prod.toFixed(3)}</b>  <span class="m">(target ≈ 2)</span>\nMACs × ${(d * w * w * r * r).toFixed(2)}  →  <b class="b">${macs.toFixed(2)} G</b>\nparams ≈ d·w² × 5.29M = <b class="o">${par.toFixed(1)} M</b>\ninput ${res}×${res}\nestimated top-1 ≈ ${mode === 'compound' ? interp(mm).toFixed(1) + '% (interpolated from published B0–B7)' : 'n/a — single-axis scaling saturates sooner'}`;
      q('#cap').textContent = 'The orange dot is your scaled network (cost model: params ∝ d·w², MACs ∝ d·w²·r²; accuracy interpolated from the published EfficientNet family, not trained here). Switch the strategy to spend the same MACs on one axis only.';
    }, { fps: 30 });
  };

  /* =================== strip renderer (modelstrip + archlab) =================== */
  const TCOL = { input: 'muted', conv: 'blue', maxpool: 'red', avgpool: 'red', gap: 'red', fc: 'orange', flatten: 'muted', res: 'purple', bottleneck: 'purple', dwsep: 'teal', invres: 'teal', dense: 'pink', inception: 'green', convnext: 'pink', up: 'green', concat: 'green', add: 'green', dwconv: 'teal' };
  function Strip(root, cv, opts) {
    const c = AI.canvas(cv), S = { blocks: [], pos: 0, playing: true, sel: -1, speed: 0.4, onSel: null }; const STEP = 92, X0 = 52;
    S.set = (blocks, tot) => { S.blocks = blocks; S.tot = tot || { params: 1, macs: 1 }; S.pos = Math.min(S.pos, Math.max(0, blocks.length - 1)); cv.style.width = Math.max(AI.$('.arch-scroll', root).clientWidth - 16, X0 * 2 + blocks.length * STEP) + 'px'; };
    cv.addEventListener('click', (e) => { const r = cv.getBoundingClientRect(), i = AI.clamp(Math.round((e.clientX - r.left - X0) / STEP), 0, S.blocks.length - 1); S.playing = false; S.pos = i; S.sel = i; if (S.onSel) S.onSel(i, true); });
    S.draw = (dt) => {
      const B2 = S.blocks, n = B2.length; if (!n) return; if (S.playing) { S.pos += dt * S.speed; if (S.pos >= n - 0.01) S.pos = 0; }
      const i = Math.min(n - 1, Math.floor(S.pos)), f = S.pos - i, cur = S.playing ? i : (S.sel >= 0 ? S.sel : i); if (S.onSel && S.playing) S.onSel(cur, false);
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); const cy = c.h * 0.4;
      const geo = B2.map((b, k) => { const s = b.out, H = s.H || 1, Cc = s.C || s.D || 1; return { x: X0 + k * STEP, h: Math.min(c.h * 0.5, 10 + 13 * Math.log2(Math.max(H, 1))), t: Math.min(60, 6 + 4.5 * Math.log2(Math.max(Cc, 1))) }; });
      g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(geo[0].x, cy); g.lineTo(geo[n - 1].x, cy); g.stroke();
      B2.forEach((b, k) => {
        const G2 = geo[k], col = C[TCOL[b.type] || 'blue'], hot = k === cur, x = G2.x - G2.t / 2, y = cy - G2.h / 2; if (hot) { g.shadowColor = col; g.shadowBlur = 16; }
        g.fillStyle = col; g.globalAlpha = hot ? 0.95 : 0.6; g.fillRect(x, y, G2.t, G2.h); g.globalAlpha = hot ? 0.8 : 0.4; g.beginPath(); g.moveTo(x, y); g.lineTo(x + 8, y - 8); g.lineTo(x + G2.t + 8, y - 8); g.lineTo(x + G2.t, y); g.closePath(); g.fill(); g.beginPath(); g.moveTo(x + G2.t, y); g.lineTo(x + G2.t + 8, y - 8); g.lineTo(x + G2.t + 8, y + G2.h - 8); g.lineTo(x + G2.t, y + G2.h); g.closePath(); g.fill(); g.globalAlpha = 1; g.shadowBlur = 0; g.strokeStyle = hot ? C.fg : col; g.strokeRect(x, y, G2.t, G2.h);
        g.textAlign = 'center'; g.fillStyle = hot ? C.fg : C.muted; g.font = `${hot ? '600 ' : ''}10.5px sans-serif`; const ly = c.h * 0.4 + c.h * 0.3; const nm = b.name.length > 15 ? b.name.slice(0, 14) + '…' : b.name; g.fillText(nm, G2.x, ly); g.font = '10px ui-monospace,monospace'; g.fillText(b.outStr, G2.x, ly + 13); if (b.params) { g.fillStyle = C.orange; g.fillText(fmtN(b.params), G2.x, ly + 26); }
        if (opts && opts.cost) { const bh = 40, base = c.h - 4, pf = b.params / S.tot.params, mf = b.macs / S.tot.macs; g.fillStyle = C.orange; g.fillRect(G2.x - 14, base - bh * Math.min(1, pf * 3), 12, bh * Math.min(1, pf * 3)); g.fillStyle = C.blue; g.fillRect(G2.x + 2, base - bh * Math.min(1, mf * 3), 12, bh * Math.min(1, mf * 3)); }
      });
      const x0 = geo[i].x, x1 = geo[Math.min(n - 1, i + 1)].x, mv = f < 0.55 ? 0 : AI.smooth((f - 0.55) / 0.45); if (S.playing) { g.shadowColor = C.fg; g.shadowBlur = 14; g.fillStyle = C.fg; g.beginPath(); g.arc(AI.lerp(x0, x1, mv), cy, 5.5, 0, 7); g.fill(); g.shadowBlur = 0; }
      if (opts && opts.cost) { g.fillStyle = C.muted; g.font = '10px sans-serif'; g.textAlign = 'left'; g.fillText('bars under each block: orange = share of params, blue = share of MACs (×3 scale)', 6, c.h - 46); }
    };
    return S;
  }
  const detailHTML = (b, i) => `<h4 style="margin:6px 0 4px">${i + 1}. ${b.name} <small style="color:var(--muted)">${(b.tags || []).join(' · ')}</small></h4><div class="mono" style="font-size:12px">${b.in ? AIArch.shapeStr(b.in) + ' → ' : ''}<b>${b.outStr}</b> &nbsp; params <b>${fmtN(b.params)}</b> &nbsp; MACs <b>${fmtN(b.macs)}</b></div><table class="tbl" style="font-size:12px"><tr><th>layer</th><th>output</th><th>params</th><th>MACs</th><th>RF</th></tr>${b.sub.map((s) => `<tr><td>${s.name}</td><td class="mono">${s.out}</td><td>${fmtN(s.params)}</td><td>${fmtN(s.macs)}</td><td>${s.rf ?? ''}</td></tr>`).join('')}</table>`;

  AI.labs.modelstrip = function (root) {
    const d = root.dataset, name = d.model || 'googlenet'; const m = CVA.model(name), t = m.totals;
    root.classList.add('lab');
    root.innerHTML = `<div class="lab-head"><i class="live-dot"></i><b>${d.title || CVA.TITLES[name] || AIArch.PRESETS[name].title}</b><span>click a block for its sub-layer table</span></div>
      <div class="arch-bar"><label>Speed<input type="range" id="sp" min="0.1" max="1.5" step="0.05" value="0.4"></label><button class="btn small primary" id="play">⏸ Pause</button><button class="btn small" id="prev">◀ Prev</button><button class="btn small" id="next">Next ▶</button></div>
      <div class="arch-scroll"><canvas style="height:260px"></canvas></div><div class="caption arch-cap" id="cap"></div>
      <div class="arch-bottom"><div id="detail"></div><div class="readout" id="sum"></div></div>`;
    const q = (s) => $(s, root), S = Strip(root, $('canvas', root), { cost: true }); S.set(m.blocks, t);
    q('#sum').innerHTML = `<span class="m">whole network (224×224 input)</span>\nparameters <b class="o">${fmtN(t.params)}</b>\nMACs       <b class="b">${fmtN(t.macs)}</b>\nactivations <b class="g">${fmtN(t.act)}</b>\nweighted layers ${t.depth}\nfinal receptive field ${t.rf || '—'} px`;
    let shown = -1; S.onSel = (i) => { if (i !== shown) { shown = i; q('#detail').innerHTML = detailHTML(S.blocks[i], i); } };
    q('#play').onclick = () => { S.playing = !S.playing; q('#play').textContent = S.playing ? '⏸ Pause' : '▶ Play'; };
    const go = (dd) => { S.playing = false; q('#play').textContent = '▶ Play'; S.sel = S.pos = AI.clamp(Math.round(S.pos) + dd, 0, S.blocks.length - 1); shown = -1; };
    q('#prev').onclick = () => go(-1); q('#next').onclick = () => go(1);
    AI.loop(root, (dt) => { S.speed = +q('#sp').value; S.draw(dt); }, { fps: 30 });
  };

  /* =================== archcompare =================== */
  AI.labs.archcompare = function (root) {
    const ALL = ['lenet5', 'alexnet', 'vgg16', 'googlenet', 'resnet18', 'resnet50', 'densenet121', 'mobilenetv1', 'mobilenetv2', 'efficientnet-b0', 'convnext-t'];
    const sel = new Set((root.dataset.models || 'alexnet,vgg16,googlenet,resnet50,densenet121,mobilenetv2,efficientnet-b0,convnext-t').split(',')); const cache = {};
    const get = (k) => cache[k] || (cache[k] = CVA.model(k)); const name = (k) => CVA.TITLES[k] || AIArch.PRESETS[k].title;
    const METRICS = { params: ['Parameters', (m) => m.totals.params, fmtN], macs: ['MACs', (m) => m.totals.macs, fmtN], act: ['Activations (sum of block outputs)', (m) => m.totals.act, fmtN], depth: ['Depth (conv + FC layers)', (m) => m.totals.depth, (v) => String(v)], top1: ['Published top-1 (%)', (m, k) => CVA.TOP1[k], (v) => v.toFixed(1)] };
    const { view, ctl } = AI.labShell(root, root.dataset.title || 'Architecture comparison', 'pick models, pick a metric — bars re-sort and glide',
      '<canvas style="height:330px"></canvas><div class="caption" id="cap"></div><canvas style="height:260px"></canvas><div class="caption">Scatter: published top-1 against MACs (log). Bigger is not always better: the best designs sit up and to the left.</div>',
      `<div>Metric<div class="row" id="met" style="margin-top:4px"></div></div><div>Models<div class="row" id="mods" style="margin-top:4px"></div></div><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="lg"> log scale</label>`);
    const q = (s) => $(s, root), cs = view.querySelectorAll('canvas'), c = AI.canvas(cs[0]), c2 = AI.canvas(cs[1]); let metric = 'params'; const pos = {}, wid = {};
    Object.keys(METRICS).forEach((k) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = METRICS[k][0].split(' (')[0]; b.onclick = () => (metric = k); q('#met').appendChild(b); });
    ALL.forEach((k) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = name(k).replace(/ \(.*/, ''); b.onclick = () => { sel.has(k) ? sel.delete(k) : sel.add(k); }; q('#mods').appendChild(b); b.dataset.k = k; });
    AI.loop(root, (dt) => {
      q('#mods').querySelectorAll('button').forEach((b) => (b.style.borderColor = sel.has(b.dataset.k) ? 'var(--blue)' : '')); q('#met').querySelectorAll('button').forEach((b, i) => (b.style.borderColor = Object.keys(METRICS)[i] === metric ? 'var(--blue)' : ''));
      const [mt, fn, ff] = METRICS[metric], items = [...sel].map((k) => ({ k, v: fn(get(k), k) })).filter((x) => x.v !== null && x.v !== undefined).sort((a, b) => b.v - a.v), log = q('#lg').checked;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); const mx = Math.max(1, ...items.map((x) => x.v)), rowH = Math.min(30, (c.h - 20) / Math.max(1, items.length)), lw = 150, bw = c.w - lw - 120;
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText(mt + (log ? ' — log scale' : ''), 8, 14);
      items.forEach((it, i) => { const ty = 24 + i * rowH; pos[it.k] = pos[it.k] === undefined ? ty : pos[it.k] + (ty - pos[it.k]) * Math.min(1, dt * 6); const tw = Math.max(2, (log ? Math.log10(it.v + 1) / Math.log10(mx + 1) : it.v / mx) * bw); wid[it.k] = wid[it.k] === undefined ? 0 : wid[it.k] + (tw - wid[it.k]) * Math.min(1, dt * 5); const y = pos[it.k];
        g.fillStyle = C.fg; g.textAlign = 'right'; g.font = '12px sans-serif'; g.fillText(name(it.k).replace(/ \(.*/, ''), lw - 8, y + rowH * 0.62); g.fillStyle = C[['blue', 'purple', 'teal', 'orange', 'green', 'pink'][ALL.indexOf(it.k) % 6]]; g.fillRect(lw, y + 3, wid[it.k], rowH - 7); g.fillStyle = C.fg; g.textAlign = 'left'; g.font = '11.5px ui-monospace,monospace'; g.fillText(ff(it.v), lw + wid[it.k] + 6, y + rowH * 0.62); });
      q('#cap').textContent = metric === 'act' ? 'Activation memory grows with resolution² and is what limits batch size in training; depthwise nets have small weights but large activations.' : metric === 'top1' ? 'Single-crop ImageNet-1k top-1 from the original papers / torchvision; training recipes differ, so treat differences of 1–2 points with care.' : '';
      const g2 = c2.ctx; g2.clearRect(0, 0, c2.w, c2.h); const px = 44, py = 12, pw = c2.w - px - 20, ph = c2.h - py - 34, lx = (m) => px + (Math.log10(m) - Math.log10(0.2)) / (Math.log10(20) - Math.log10(0.2)) * pw, ly = (a) => py + (1 - (a - 50) / (86 - 50)) * ph;
      g2.strokeStyle = C.border; g2.fillStyle = C.muted; g2.font = '10px sans-serif'; g2.textAlign = 'right'; for (let a = 50; a <= 85; a += 10) { g2.beginPath(); g2.moveTo(px, ly(a)); g2.lineTo(px + pw, ly(a)); g2.stroke(); g2.fillText(a + '%', px - 4, ly(a) + 3); } g2.textAlign = 'center'; [0.2, 1, 5, 20].forEach((m) => g2.fillText(m + ' G', lx(m), py + ph + 14)); g2.fillText('MACs (log)', px + pw / 2, py + ph + 28);
      [...sel].forEach((k) => { const a = CVA.TOP1[k]; if (!a) return; const m = get(k).totals.macs / 1e9, x = lx(AI.clamp(m, 0.2, 20)), y = ly(a); g2.fillStyle = C[['blue', 'purple', 'teal', 'orange', 'green', 'pink'][ALL.indexOf(k) % 6]]; g2.beginPath(); g2.arc(x, y, 6, 0, 7); g2.fill(); g2.fillStyle = C.fg; g2.textAlign = 'left'; g2.font = '10.5px sans-serif'; g2.fillText(name(k).replace(/ \(.*/, ''), x + 8, y + 3); });
    }, { fps: 30 });
  };

  /* =================== archlab (capstone) =================== */
  AI.labs.archlab = function (root) {
    const TASKS = { phone: ['Phone app: ≤ 4 M params, ≤ 0.35 GMACs, 1000 classes, input 224', 4, 0.35, 224], camera: ['Edge camera: ≤ 2 M params, ≤ 0.15 GMACs, input 160', 2, 0.15, 160], server: ['Server: ≤ 30 M params, ≤ 5 GMACs, input 224', 30, 5, 224], mcu: ['Microcontroller: ≤ 0.25 M params, ≤ 0.03 GMACs, input 96', 0.25, 0.03, 96] };
    const START = { 'resnet18': 'resnet18', 'mobilenetv1': 'mobilenetv1', 'mobilenetv2': 'mobilenetv2', 'vgg16': 'vgg16', 'small-cnn': 'small-cnn' };
    root.classList.add('lab');
    root.innerHTML = `<div class="lab-head"><i class="live-dot"></i><b>${root.dataset.title || 'Design under a budget'}</b><span>edit the layers; the budget meters answer instantly</span></div>
      <div class="arch-bar"><label>Task <select id="task">${Object.entries(TASKS).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join('')}</select></label><label>Start from <select id="start"><option value="">—</option>${Object.keys(START).map((k) => `<option>${k}</option>`).join('')}</select></label><label>Speed<input type="range" id="sp" min="0.1" max="1.5" step="0.05" value="0.4"></label><button class="btn small primary" id="play">⏸ Pause</button></div>
      <div class="arch-scroll"><canvas style="height:250px"></canvas></div><div class="caption arch-cap" id="cap"></div>
      <div class="arch-bottom"><div><div id="meters"></div><div id="checks" style="margin-top:8px"></div><div id="hint" class="caption"></div><div id="detail"></div></div><div><label style="display:block;font-size:12px;color:var(--muted)">Architecture (one layer per line — layers: conv dwconv dwsep invres res bottleneck dense maxpool avgpool gap fc …)<textarea id="spec" rows="16" spellcheck="false"></textarea></label><div class="err" id="err"></div></div></div>`;
    const q = (s) => $(s, root), S = Strip(root, $('canvas', root), { cost: true }); let anim = { p: 0, m: 0 }, R = null, task = 'phone';
    const setTask = () => { task = q('#task').value; recalc(); };
    const recalc = () => {
      let spec = q('#spec').value; const tk = TASKS[task]; spec = spec.replace(/^input\s+(\d+)\s+\d+\s+\d+/m, (_, c) => `input ${c} ${tk[3]} ${tk[3]}`); const r = AIArch.build(spec); q('#err').textContent = r.errors.join('\n'); if (!r.blocks.length) return; r.totals.depth = CVA.depthOf(r); R = r; S.set(r.blocks, r.totals);
      const t = r.totals, okP = t.params <= tk[1] * 1e6, okM = t.macs <= tk[2] * 1e9, okR = t.rf && t.rf >= tk[3] * 0.75, last = r.final, okC = last && last.k === 'vec' && last.D === 1000, ok = okP && okM && okR && okC && !r.errors.length;
      q('#checks').innerHTML = [[okP, `parameters ${fmtN(t.params)} ≤ ${tk[1]} M`], [okM, `compute ${fmtN(t.macs)}MACs ≤ ${tk[2]} G`], [okR, `receptive field ${t.rf} px ≥ 75% of the ${tk[3]} px input`], [okC, 'ends in a 1000-way classifier (… gap, fc 1000)']].map(([o, s]) => `<div style="color:${o ? 'var(--green)' : 'var(--red)'}">${o ? '✔' : '✘'} ${s}</div>`).join('') + `<div style="margin-top:6px;font-weight:700;font-size:15px;color:${ok ? 'var(--green)' : 'var(--orange)'}">${ok ? '★ BUDGET MET — now try to shrink it further' : 'budget not met yet'}</div>`;
      const top = r.blocks.slice().sort((a, b) => b.params - a.params)[0], topM = r.blocks.slice().sort((a, b) => b.macs - a.macs)[0];
      q('#hint').innerHTML = `Biggest parameter spender: <b>${top.name}</b> (${(100 * top.params / t.params).toFixed(0)}%). Biggest compute spender: <b>${topM.name}</b> (${(100 * topM.macs / t.macs).toFixed(0)}%).` + (!okP || !okM ? ' Ideas: swap <code>conv</code>/<code>res</code> for <code>dwsep</code> or <code>invres</code>, replace <code>flatten</code> + big <code>fc</code> by <code>gap</code>, downsample earlier, narrow the channels.' : '');
      target = { p: t.params / (tk[1] * 1e6), m: t.macs / (tk[2] * 1e9) };
    };
    let target = { p: 0, m: 0 };
    q('#spec').value = AIArch.PRESETS.resnet18.spec; let tm; q('#spec').addEventListener('input', () => { clearTimeout(tm); tm = setTimeout(recalc, 250); });
    q('#task').onchange = setTask; q('#start').onchange = () => { if (q('#start').value) { q('#spec').value = AIArch.PRESETS[q('#start').value].spec; recalc(); } };
    q('#play').onclick = () => { S.playing = !S.playing; q('#play').textContent = S.playing ? '⏸ Pause' : '▶ Play'; }; let shown = -1; S.onSel = (i) => { if (i !== shown) { shown = i; q('#detail').innerHTML = detailHTML(S.blocks[i], i); } };
    recalc();
    AI.loop(root, (dt) => {
      S.speed = +q('#sp').value; S.draw(dt); anim.p += (target.p - anim.p) * Math.min(1, dt * 5); anim.m += (target.m - anim.m) * Math.min(1, dt * 5); const C = AI.col();
      const bar = (lab, v, real) => { const w = Math.min(100, v * 50); return `<div style="font-size:12px;color:var(--muted)">${lab} <b style="color:var(--fg)">${real}</b> <span>(${(v * 100).toFixed(0)}% of budget)</span></div><div style="position:relative;height:16px;background:var(--bg2);border-radius:4px;margin:2px 0 8px"><div style="height:100%;width:${w}%;background:${v <= 1 ? C.green : C.red};border-radius:4px"></div><div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:2px;background:${C.fg}"></div></div>`; };
      if (R) q('#meters').innerHTML = bar('Parameters', anim.p, fmtN(R.totals.params)) + bar('MACs', anim.m, fmtN(R.totals.macs)); q('#cap').textContent = 'White line = your budget. Bars fill green while you are under budget and turn red when you exceed it.';
    }, { fps: 30 });
  };
})(typeof window !== 'undefined' ? window : globalThis);
