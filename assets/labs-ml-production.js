/* ML production labs (Module 9): sysarch, featstore, expreg, serving, rollout, driftmon, governance, capstone.
 * Requires assets/lib.js. All numbers shown by the labs are ILLUSTRATIVE (round example values you can edit), not benchmarks.
 *
 *   <div data-lab="sysarch" data-arch="realtime|batch|streaming|edge"></div>   animated reference architectures, bottleneck + cost, failure injection
 *   <div data-lab="featstore"></div>   offline/online feature store, naive vs point-in-time join on a timeline
 *   <div data-lab="expreg"></div>      experiment table, gates and model registry stages
 *   <div data-lab="serving"></div>     request stream, cache, batching window, latency histogram
 *   <div data-lab="rollout" ></div>    canary / shadow / A-B with automatic rollback
 *   <div data-lab="driftmon"></div>    drifting features, PSI/KS, alarm, retraining trigger
 *   <div data-lab="governance"></div>  model card form, risk tier and release gate
 *   <div data-lab="capstone"></div>    end-to-end project simulator with scorecard
 * All animations are dt-based, slow by default, with speed slider and pause/step.
 */
(function () {
  const clamp = AI.clamp, lerp = AI.lerp, smooth = AI.smooth;
  const sl = (id, label, min, max, step, val) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const transport = (extra) => `<div class="row"><button class="btn small primary" id="play">Pause</button><button class="btn small" id="step">Step</button><button class="btn small" id="reset">Restart</button>${extra || ''}</div>${sl('sp', 'Speed', 0.25, 3, 0.05, 1)}`;
  const wire = (root, fmts) => root.querySelectorAll('input[type=range]').forEach((inp) => {
    const v = root.querySelector('#' + inp.id + 'v'), f = fmts && fmts[inp.id];
    const u = () => { if (v) v.textContent = f ? f(+inp.value) : AI.fmt(+inp.value, 3); };
    inp._u = u; inp.addEventListener('input', u); u();
  });
  const refresh = (root) => root.querySelectorAll('input[type=range]').forEach((i) => i._u && i._u());
  const fmtN = (v) => (v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + 'k' : v >= 10 ? String(Math.round(v)) : v.toFixed(1));
  const text = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const rr = (g, x, y, w, h, r) => { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };
  const Phi = (z) => {
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  };
  const PhiInv = (p) => { let lo = -8, hi = 8; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (Phi(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  const setPlay = (root, playing) => { root.querySelector('#play').textContent = playing ? 'Pause' : 'Play'; };
  const money = (v) => (v >= 100 ? '$' + v.toFixed(0) : v >= 1 ? '$' + v.toFixed(2) : v >= 0.01 ? '$' + v.toFixed(3) : '$' + v.toFixed(4));

  /* ====================================================================== sysarch */
  const N = (id, label, x, y, lat, cap, rep, cost, o) => Object.assign({ id, label, x, y, lat, cap, rep, cost }, o || {});
  const F = (path, frac, kind, lab, o) => Object.assign({ path, frac, kind, lab }, o || {});
  const ARCHS = {
    realtime: {
      name: 'Real-time API', unit: 'requests/s', lam: [1, 2000, 150], hasHit: true,
      note: 'Online serving: the user waits. Latency percentiles, cache hit rate and replicas decide capacity and cost.',
      nodes: [N('client', 'Client', 0.07, 0.26, 0, 1e6, 1, 0, { ext: 1 }), N('gw', 'API gateway', 0.25, 0.26, 3, 800, 2, 0.05), N('cache', 'Cache', 0.45, 0.07, 1, 5000, 1, 0.1),
        N('fs', 'Feature store (online)', 0.45, 0.48, 6, 600, 2, 0.15), N('model', 'Model server', 0.67, 0.26, 25, 40, 4, 0.4), N('mon', 'Monitor', 0.89, 0.26, 2, 3000, 1, 0.05),
        N('lake', 'Data lake', 0.07, 0.86, 0, 1e6, 1, 0.1, { off: 1 }), N('fsOff', 'Feature store (offline)', 0.27, 0.86, 0, 1e6, 1, 0.1, { off: 1 }),
        N('train', 'Training job', 0.5, 0.86, 0, 1e6, 1, 0.3, { off: 1 }), N('reg', 'Model registry', 0.72, 0.86, 0, 1e6, 1, 0.02, { off: 1 })],
      build: (h, dn) => {
        const L = [];
        if (!dn.cache) { L.push(F(['client', 'gw', 'cache', 'gw', 'client'], h, 'req', 'hit')); L.push(F(['client', 'gw', 'cache', 'fs', 'model', 'gw', 'client'], 1 - h, 'req', 'miss')); }
        else L.push(F(['client', 'gw', 'fs', 'model', 'gw', 'client'], 1, 'req', 'miss'));
        L.push(F(['model', 'mon'], 1, 'data', 'log', { async: 1 }));
        L.push(F(['lake', 'fsOff', 'train', 'reg', 'model'], 1, 'model', 'new model', { slow: 1 }));
        L.push(F(['fsOff', 'fs'], 1, 'data', 'features', { slow: 1 }));
        L.push(F(['mon', 'train'], 1, 'data', 'retrain', { slow: 1 }));
        return L;
      },
      story: [['hit', '① A request enters through the API gateway (blue packet).'], ['hit', '② Cache HIT: the answer returns in a few milliseconds; the model server is never touched.'], ['miss', '③ Cache MISS: fetch the user\'s latest features from the ONLINE feature store, then run the model.'],
        ['log', '④ Every prediction is logged to the monitor (teal) — drift statistics are computed from these logs.'], ['features', '⑤ Offline → online: a materialisation job copies the latest feature values so serving sees what training saw.'],
        ['new model', '⑥ Offline path: data lake → offline features → training job → registry → model server (purple = new model version).'], ['retrain', '⑦ The monitor can trigger a retraining job when drift or performance rules fire.']],
    },
    batch: {
      name: 'Batch scoring', unit: 'records/s', lam: [100, 50000, 6000],
      note: 'Scheduled scoring: nobody waits on one record. Throughput and run time matter; predictions are as old as the schedule interval.',
      nodes: [N('lake', 'Data lake', 0.08, 0.26, 1, 60000, 1, 0.1), N('fsOff', 'Feature store (offline)', 0.29, 0.26, 1, 30000, 1, 0.12), N('scorer', 'Batch scoring job', 0.5, 0.26, 2, 2000, 4, 0.4), N('out', 'Prediction table', 0.71, 0.26, 1, 40000, 1, 0.1),
        N('app', 'App / CRM reads', 0.91, 0.26, 1, 20000, 1, 0.05), N('sched', 'Scheduler', 0.29, 0.8, 0, 1e6, 1, 0.01, { off: 1 }), N('reg', 'Model registry', 0.5, 0.8, 0, 1e6, 1, 0.02, { off: 1 }), N('mon', 'Monitor', 0.8, 0.8, 1, 40000, 1, 0.05)],
      build: () => [F(['lake', 'fsOff', 'scorer', 'out', 'app'], 1, 'data', 'rows'), F(['out', 'mon'], 1, 'data', 'stats', { async: 1 }), F(['sched', 'scorer'], 1, 'req', 'run nightly', { slow: 1 }), F(['reg', 'scorer'], 1, 'model', 'model v7', { slow: 1 })],
      story: [['run nightly', '① The scheduler starts the job (say 02:00).'], ['model v7', '② The job loads the Production model version from the registry.'], ['rows', '③ Rows stream from the lake through offline features into the scorer, many at a time.'], ['rows', '④ Predictions are written to a table; apps simply read them (always up to 24 h old).'], ['stats', '⑤ Prediction statistics go to the monitor for drift checks.']],
    },
    streaming: {
      name: 'Streaming', unit: 'events/s', lam: [10, 3000, 150],
      note: 'Events flow through a durable queue. When consumers fall behind the queue grows (backlog) instead of dropping requests immediately.',
      nodes: [N('src', 'Event source', 0.07, 0.26, 0, 1e6, 1, 0, { ext: 1 }), N('broker', 'Queue (Kafka)', 0.26, 0.26, 2, 3000, 2, 0.2, { queue: 1 }), N('stream', 'Stream processor', 0.46, 0.26, 15, 120, 3, 0.25),
        N('fs', 'Feature store (online)', 0.46, 0.62, 5, 800, 2, 0.15), N('model', 'Model server', 0.68, 0.26, 20, 60, 3, 0.4), N('sink', 'Alerts / output topic', 0.89, 0.26, 2, 5000, 1, 0.05),
        N('lake', 'Data lake (archive)', 0.26, 0.88, 0, 1e6, 1, 0.1, { off: 1 }), N('train', 'Training job', 0.62, 0.88, 0, 1e6, 1, 0.3, { off: 1 }), N('mon', 'Monitor', 0.89, 0.62, 2, 5000, 1, 0.05)],
      build: () => [F(['src', 'broker', 'stream', 'fs', 'model', 'sink'], 1, 'req', 'event'), F(['broker', 'lake'], 1, 'data', 'archive', { async: 1 }), F(['sink', 'mon'], 1, 'data', 'metrics', { async: 1 }), F(['lake', 'train', 'model'], 1, 'model', 'new model', { slow: 1 }), F(['mon', 'train'], 1, 'data', 'retrain', { slow: 1 })],
      story: [['event', '① Events (clicks, sensor readings) are published to the queue.'], ['event', '② The stream processor consumes them and computes windowed features (e.g. clicks in the last 10 min).'], ['event', '③ It enriches with the online feature store, then calls the model.'], ['event', '④ Predictions go to an output topic (alerts, downstream services).'], ['archive', '⑤ Raw events are archived to the lake for training.'], ['new model', '⑥ New models are trained from the archive and pushed to the model server.']],
    },
    edge: {
      name: 'Edge device', unit: 'inferences/s', lam: [1, 200, 20],
      note: 'The model runs on the device: no network round trip, works offline, private. Uploads and model updates are asynchronous.',
      nodes: [N('dev', 'Device / sensor', 0.08, 0.26, 0, 1e6, 1, 0, { ext: 1 }), N('local', 'On-device model', 0.3, 0.26, 12, 80, 1, 0.02), N('reg', 'Registry + OTA update', 0.55, 0.26, 0, 1e6, 1, 0.02, { off: 1 }), N('train', 'Training job', 0.8, 0.26, 0, 1e6, 1, 0.3, { off: 1 }),
        N('buf', 'Local buffer', 0.3, 0.8, 1, 500, 1, 0, {}), N('sync', 'Upload when online', 0.55, 0.8, 80, 30, 1, 0.01), N('cloud', 'Cloud ingest / lake', 0.8, 0.8, 5, 2000, 1, 0.3)],
      build: () => [F(['dev', 'local', 'dev'], 1, 'req', 'frame'), F(['local', 'buf', 'sync', 'cloud'], 0.1, 'data', 'hard cases', { async: 1 }), F(['cloud', 'train', 'reg', 'local'], 1, 'model', 'OTA update', { slow: 1 })],
      story: [['frame', '① The device captures an input and the on-device model predicts: no network needed.'], ['hard cases', '② Only uncertain / interesting samples (~10%) are buffered locally.'], ['hard cases', '③ When a connection exists the buffer uploads to the cloud lake (mark "Upload" down to see inference continue).'], ['OTA update', '④ Cloud retraining produces a new model; the registry pushes it over the air.']],
    },
  };

  AI.labs.sysarch = function (root) {
    const { view, ctl } = AI.labShell(root, 'ML system architecture', 'illustrative: blue = requests, teal = data, purple = model updates · click a component to edit it, then inject failures',
      '<canvas style="height:410px;cursor:pointer"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<label>Reference architecture<select id="arch"><option value="realtime">Real-time API</option><option value="batch">Batch scoring</option><option value="streaming">Streaming</option><option value="edge">Edge device</option></select></label>
       ${sl('lam', 'Offered load', 0, 100, 1, 50)}
       <div id="hitw">${sl('hit', 'Cache hit rate', 0, 0.95, 0.05, 0.6)}</div>
       <label>Selected component<select id="comp"></select></label>
       ${sl('lat', 'Latency (ms)', 0, 300, 1, 10)}${sl('cap', 'Capacity per replica (/s)', 1, 5000, 1, 100)}${sl('rep', 'Replicas', 1, 20, 1, 1)}${sl('cost', 'Cost per replica ($/h)', 0, 2, 0.01, 0.1)}
       <div class="row"><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="down"> failed (down)</label><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="slow"> slow ×3</label></div>
       ${transport()}`);
    const q = (s) => root.querySelector(s), cv = AI.$('canvas', view), c = AI.canvas(cv);
    let A, S, selId, playing = true, story = 0, storyT = 0, spawnAcc = 0, slowAcc = 0, pk = [], cnt = { ok: 0, drop: 0 }, bl = {}, lastCap = '', M = null, geo = {};
    const lamOf = (v) => { const [a, b] = A.lam; return a * Math.pow(b / a, v / 100); };
    const setV = (id, v) => { q('#' + id).value = v; };
    const initArch = (key) => {
      A = ARCHS[key]; S = {}; bl = {}; pk = []; cnt = { ok: 0, drop: 0 };
      A.nodes.forEach((n) => { S[n.id] = { lat: n.lat, cap: n.cap, rep: n.rep, cost: n.cost, down: false, slow: false }; bl[n.id] = 0; });
      q('#comp').innerHTML = A.nodes.filter((n) => !n.ext).map((n) => `<option value="${n.id}">${n.label}</option>`).join('');
      selId = A.nodes.find((n) => !n.ext).id; if (key === 'realtime') selId = 'model';
      q('#comp').value = selId;
      const [a, b, d] = A.lam; setV('lam', Math.round(100 * Math.log(d / a) / Math.log(b / a)));
      q('#hitw').style.display = A.hasHit ? '' : 'none';
      story = 0; storyT = 0; loadSel(); refresh(root); geoUpdate(); model();
    };
    const loadSel = () => { const s = S[selId]; setV('lat', s.lat); q('#cap').max = Math.max(5000, s.cap * 1.5); setV('cap', s.cap); setV('rep', s.rep); setV('cost', s.cost); q('#down').checked = s.down; q('#slow').checked = s.slow; const ext = A.nodes.find((n) => n.id === selId).off; ['lat', 'cap'].forEach((i) => (q('#' + i).disabled = !!ext)); };
    const geoUpdate = () => {
      const w = c.w, h = c.h, bw = Math.min(138, w * 0.18), bh = 40, padx = bw / 2 + 8;
      geo = { bw, bh }; A.nodes.forEach((n) => { n.px = padx + n.x * (w - 2 * padx - 0) * 1 + 0; n.py = 26 + n.y * (h - 78); });
      // normalise x so that max x maps to the right padding
      const mx = Math.max(...A.nodes.map((n) => n.x)), mn = Math.min(...A.nodes.map((n) => n.x));
      A.nodes.forEach((n) => (n.px = padx + ((n.x - mn) / (mx - mn)) * (w - 2 * padx)));
    };
    c.onresize = () => { if (A) geoUpdate(); };
    const node = (id) => A.nodes.find((n) => n.id === id);
    const down = () => { const d = {}; for (const id in S) d[id] = S[id].down; return d; };
    const flows = () => A.build(+q('#hit').value, down());
    /* fluid model */
    function model() {
      const lam = lamOf(+q('#lam').value), L = flows(), visit = {};
      L.filter((f) => !f.slow).forEach((f) => new Set(f.async ? f.path.slice(1) : f.path).forEach((id) => { if (!node(id).ext) visit[id] = (visit[id] || 0) + f.frac; }));
      const rho = {}, capTot = {}; let bott = null, bR = -1, maxQ = Infinity;
      for (const id in visit) { const s = S[id]; capTot[id] = s.down ? 0 : s.cap * s.rep; rho[id] = capTot[id] ? lam * visit[id] / capTot[id] : Infinity; if (!node(id).off && rho[id] > bR) { bR = rho[id]; bott = id; } if (capTot[id] && !node(id).off) maxQ = Math.min(maxQ, capTot[id] / visit[id]); }
      let okF = 0, totF = 0, latSum = 0, latW = 0;
      L.filter((f) => !f.slow && !f.async).forEach((f) => {
        totF += f.frac; let succ = 1, lat = 0;
        const u = new Set(f.path); u.forEach((id) => { if (node(id).ext) return; succ *= S[id].down ? 0 : Math.min(1, 1 / rho[id]); });
        f.path.forEach((id) => { if (node(id).ext) return; const s = S[id]; lat += s.lat * (s.slow ? 3 : 1) / (1 - Math.min(rho[id], 0.95)); });
        okF += f.frac * succ; if (succ > 0) { latSum += f.frac * succ * lat; latW += f.frac * succ; }
      });
      let costH = 0; A.nodes.forEach((n) => (costH += S[n.id].cost * S[n.id].rep));
      const mean = latW ? latSum / latW : NaN;
      M = { lam, L, visit, rho, capTot, bott, bR, maxQ, succ: totF ? okF / totF : 1, mean, costH, cost1k: costH / (lam * 3.6) };
      return M;
    }
    const nodeRho = (id) => (M && M.rho[id] !== undefined ? M.rho[id] : 0);
    const pickFlow = (slow) => {
      const L = M.L.filter((f) => !!f.slow === slow); if (!L.length) return null;
      const w = L.map((f) => f.frac * (f.async ? 0.35 : 1)); const tot = w.reduce((a, b) => a + b, 0); if (tot <= 0) return null;
      let r = Math.random() * tot; for (let i = 0; i < L.length; i++) { r -= w[i]; if (r <= 0) return L[i]; } return L[L.length - 1];
    };
    const spawn = (f) => { if (!f || pk.length > 80) return; pk.push({ f, i: 0, t: 0, st: 'move', dw: 0, fade: 0, dead: false, rnd: Math.random() }); };
    const off = (a, b) => { const dx = b.px - a.px, dy = b.py - a.py, l = Math.hypot(dx, dy) || 1; return [-dy / l * 3.5, dx / l * 3.5]; };
    const hex = (k) => { const C = AI.col(); return k === 'req' ? C.blue : k === 'data' ? C.teal : C.purple; };
    const adv = () => { story = (story + 1) % A.story.length; storyT = 0; const f = M.L.find((x) => x.lab === A.story[story][0]); if (f) spawn(f); };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); adv(); };
    q('#reset').onclick = () => { pk = []; cnt = { ok: 0, drop: 0 }; story = 0; storyT = 0; for (const k in bl) bl[k] = 0; };
    q('#arch').onchange = () => initArch(q('#arch').value);
    q('#comp').onchange = () => { selId = q('#comp').value; loadSel(); refresh(root); };
    const upd = () => { const s = S[selId]; if (!q('#lat').disabled) { s.lat = +q('#lat').value; s.cap = +q('#cap').value; } s.rep = +q('#rep').value; s.cost = +q('#cost').value; s.down = q('#down').checked; s.slow = q('#slow').checked; model(); };
    ['lat', 'cap', 'rep', 'cost', 'down', 'slow', 'lam', 'hit'].forEach((id) => q('#' + id).addEventListener('input', upd));
    q('#down').addEventListener('change', upd); q('#slow').addEventListener('change', upd);
    cv.addEventListener('click', (e) => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; const n = A.nodes.find((n) => !n.ext && Math.abs(x - n.px) < geo.bw / 2 && Math.abs(y - n.py) < geo.bh / 2); if (n) { selId = n.id; q('#comp').value = n.id; loadSel(); refresh(root); } });
    wire(root, { lam: (v) => (A ? fmtN(lamOf(v)) + ' ' + A.unit : ''), hit: (v) => Math.round(v * 100) + '%', lat: (v) => v + ' ms', cap: (v) => v + '/s', rep: (v) => v, cost: (v) => '$' + (+v).toFixed(2), sp: (v) => v.toFixed(2) + '×' });
    initArch(root.dataset.arch && ARCHS[root.dataset.arch] ? root.dataset.arch : 'realtime'); q('#arch').value = root.dataset.arch && ARCHS[root.dataset.arch] ? root.dataset.arch : 'realtime';
    let lastOut = '';
    AI.loop(root, (dt) => {
      const sp = +q('#sp').value, d = playing ? dt * sp : 0;
      if (!M) return;
      if (playing) {
        const lam = M.lam, rate = 1.2 + 3.2 * clamp(Math.log10(lam) / 3.7, 0, 1);
        spawnAcc += d * rate; while (spawnAcc >= 1) { spawnAcc -= 1; spawn(pickFlow(false)); }
        slowAcc += d * 0.22; while (slowAcc >= 1) { slowAcc -= 1; spawn(pickFlow(true)); }
        storyT += d; if (storyT > 5.5) adv();
        for (const id in bl) { const r = M.rho[id]; const dd = S[id].down ? 1 : (r === undefined ? -1 : (r > 1 ? (1 - 1 / r) : -0.5 * (1 - r))); bl[id] = clamp(bl[id] + dd * d * 0.35, 0, 1); }
        for (const p of pk) {
          if (p.dead) { p.fade += d / 0.9; continue; }
          const path = p.f.path;
          if (p.st === 'dwell') { p.dw -= d; if (p.dw <= 0) { if (p.i >= path.length - 1) { p.dead = true; p.fade = 0.01; p.ok = true; if (!p.f.slow) cnt.ok++; } else p.st = 'move'; } continue; }
          p.t += d / 1.5;
          if (p.t >= 1) {
            p.t = 0; p.i++; const id = path[p.i], n = node(id), s = S[id];
            if (!n.ext && !p.f.slow) { const r = nodeRho(id); if (s.down || (r > 1 && p.rnd > 1 / r)) { p.dead = true; p.fade = 0.01; p.bad = true; cnt.drop++; continue; } }
            else if (!n.ext && p.f.slow && s.down) { p.dead = true; p.fade = 0.01; p.bad = true; continue; }
            p.st = 'dwell'; p.dw = n.ext ? 0.05 : 0.25 + 0.6 * Math.min(1, s.lat * (s.slow ? 3 : 1) / 80) + bl[id] * 1.2; p.rnd = Math.random();
          }
        }
        pk = pk.filter((p) => !(p.dead && p.fade >= 1));
      }
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      // edges
      const edges = new Map(); M.L.forEach((f) => { for (let i = 0; i < f.path.length - 1; i++) { const k = f.path[i] + '>' + f.path[i + 1]; if (!edges.has(k)) edges.set(k, { a: f.path[i], b: f.path[i + 1], kind: f.kind, slow: f.slow }); } });
      const sf = M.L.find((f) => f.lab === A.story[story][0]); const hl = new Set(); if (sf) for (let i = 0; i < sf.path.length - 1; i++) hl.add(sf.path[i] + '>' + sf.path[i + 1]);
      edges.forEach((e, k) => {
        const a = node(e.a), b = node(e.b), o = off(a, b), on = hl.has(k);
        g.strokeStyle = on ? hex(e.kind) : C.border; g.lineWidth = on ? 2.5 : 1.2; g.setLineDash(e.slow ? [5, 4] : []);
        g.beginPath(); g.moveTo(a.px + o[0], a.py + o[1]); g.lineTo(b.px + o[0], b.py + o[1]); g.stroke(); g.setLineDash([]);
        const mx = lerp(a.px, b.px, 0.62) + o[0], my = lerp(a.py, b.py, 0.62) + o[1], ux = (b.px - a.px), uy = (b.py - a.py), ul = Math.hypot(ux, uy) || 1;
        AI.arrow(g, mx - ux / ul * 6, my - uy / ul * 6, mx + ux / ul * 4, my + uy / ul * 4, on ? hex(e.kind) : C.muted, 1);
      });
      // nodes
      const bw = geo.bw, bh = geo.bh;
      A.nodes.forEach((n) => {
        const s = S[n.id], r = nodeRho(n.id), isSel = n.id === selId;
        const heat = s.down ? C.red : (n.ext || n.off || M.visit[n.id] === undefined) ? C.bg3 : (r < 0.7 ? AI.mix(C.bg3, C.green, 0.45) : r < 1 ? AI.mix(C.bg3, C.orange, 0.6) : AI.mix(C.bg3, C.red, 0.7));
        g.fillStyle = heat; rr(g, n.px - bw / 2, n.py - bh / 2, bw, bh, 8); g.fill();
        g.strokeStyle = isSel ? C.blue : (n.id === M.bott && M.bR > 0.01 ? C.orange : C.border); g.lineWidth = isSel ? 2.5 : 1.3; rr(g, n.px - bw / 2, n.py - bh / 2, bw, bh, 8); g.stroke();
        const fs = bw < 100 ? 10 : 11;
        text(g, n.label, n.px, n.py - 3, C.fg, '600 ' + fs + 'px sans-serif', 'center');
        const sub = n.ext ? 'source' : n.off ? 'offline job' : `${Math.round(s.lat * (s.slow ? 3 : 1))}ms · ${fmtN(s.cap * s.rep)}/s${M.visit[n.id] !== undefined ? ' · ρ' + (r > 9 ? '>9' : r.toFixed(2)) : ''}`;
        text(g, sub, n.px, n.py + 11, C.muted, (bw < 100 ? 8 : 9) + 'px monospace', 'center');
        if (s.down) text(g, '✕ DOWN', n.px, n.py - bh / 2 - 4, C.red, '700 10px sans-serif', 'center');
        else if (s.slow) text(g, 'slow ×3', n.px, n.py - bh / 2 - 4, C.orange, '700 10px sans-serif', 'center');
        if (!n.ext && !n.off) { g.fillStyle = C.border; g.fillRect(n.px - bw / 2, n.py + bh / 2 + 3, bw, 4); g.fillStyle = bl[n.id] > 0.8 ? C.red : C.orange; g.fillRect(n.px - bw / 2, n.py + bh / 2 + 3, bw * bl[n.id], 4); }
      });
      // packets
      pk.forEach((p) => {
        const path = p.f.path, a = node(path[Math.min(p.i, path.length - 1)]);
        let x, y;
        if (p.st === 'dwell' || p.i >= path.length - 1 && p.dead) { x = a.px; y = a.py - 14 + (p.rnd - 0.5) * 8; if (p.i > 0) { const pa = node(path[p.i - 1]), o = off(pa, a); x += o[0]; } }
        else { const b = node(path[p.i + 1]), o = off(a, b), t = smooth(p.t); x = lerp(a.px, b.px, t) + o[0]; y = lerp(a.py, b.py, t) + o[1]; }
        const col = p.bad ? C.red : hex(p.f.kind), al = p.dead ? 1 - p.fade : 1;
        g.globalAlpha = clamp(al, 0, 1); g.fillStyle = col; g.beginPath(); g.arc(x, y, p.bad ? 5 + p.fade * 6 : 5, 0, 7); g.fill();
        if (!p.dead) text(g, p.f.lab, x + 8, y - 6, col, '9px sans-serif');
        else if (p.bad) text(g, 'dropped', x + 8, y - 6, C.red, '9px sans-serif');
        g.globalAlpha = 1;
      });
      text(g, 'bar under a component = queue backlog', 8, h - 6, C.muted, '10px sans-serif');
      const cp = A.story[story][1] + (cnt.drop > 0 ? '   (' + cnt.drop + ' packets dropped)' : '');
      if (cp !== lastCap) { lastCap = cp; q('#cap').textContent = cp; }
      const m = M, bn = m.bott ? node(m.bott).label : '—', p95 = m.mean * 3.0, p99 = m.mean * 4.6;
      const o = `<span class="m">load</span> <b>${fmtN(m.lam)} ${A.unit}</b>   <span class="m">success</span> <b class="${m.succ > 0.99 ? 'g' : m.succ > 0.9 ? 'o' : 'r'}">${(m.succ * 100).toFixed(1)}%</b>   <span class="m">mean latency</span> <b>${isFinite(m.mean) ? m.mean.toFixed(0) + ' ms' : '—'}</b>   <span class="m">≈p95 / p99</span> <b>${isFinite(m.mean) ? p95.toFixed(0) + ' / ' + p99.toFixed(0) + ' ms' : '—'}</b>\n<span class="m">(p95 ≈ 3×mean, p99 ≈ 4.6×mean: exponential-tail approximation)</span>\n<span class="m">max sustainable</span> <b>${isFinite(m.maxQ) ? fmtN(m.maxQ) : '—'} ${A.unit}</b>   <span class="m">bottleneck</span> <b class="${m.bR > 1 ? 'r' : m.bR > 0.7 ? 'o' : 'g'}">${bn} (ρ ${m.bR > 9 ? '>9' : m.bR.toFixed(2)})</b>\n<span class="m">fleet cost</span> <b>${money(m.costH)}/h</b>   <span class="m">cost per 1k predictions</span> <b>${money(m.cost1k)}</b> <span class="m">(illustrative prices)</span>\n<span class="m">${A.note}</span>`;
      if (o !== lastOut) { lastOut = o; q('#out').innerHTML = o; }
    }, { fps: 30 });
  };

  /* ====================================================================== featstore */
  AI.labs.featstore = function (root) {
    const { view, ctl } = AI.labShell(root, 'Feature store: offline history, online latest, point-in-time join', 'illustrative: the clock moves along time; at each label time a training row is built from the feature store',
      '<canvas style="height:470px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<label>Join used to build training rows<select id="mode"><option value="naive">Naive join: latest value overall (WRONG)</option><option value="pit">Point-in-time join: latest value with day ≤ t</option></select></label>
       ${sl('v1', 'Feature value set on day 3', 0, 9, 1, 1)}${sl('v2', 'Feature value set on day 9', 0, 9, 1, 2)}${sl('v3', 'Feature value set on day 15', 0, 9, 1, 5)}${sl('v4', 'Feature value set on day 22', 0, 9, 1, 6)}
       ${sl('l1', 'Training example 1: prediction day', 1, 30, 1, 10)}${sl('l2', 'Training example 2: prediction day', 1, 30, 1, 18)}${sl('l3', 'Training example 3: prediction day', 1, 30, 1, 26)}
       ${sl('ttl', 'TTL: value older than … days counts as missing', 1, 30, 1, 30)}
       ${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { ttl: (v) => (v >= 30 ? 'off' : v + ' d'), sp: (v) => v.toFixed(2) + '×' });
    const DAYS = [3, 9, 15, 22], TMAX = 30;
    let clock = 0, playing = true, lastCap = '', lastOut = '', hold = 0;
    const vals = () => DAYS.map((d, i) => ({ d, v: +q('#v' + (i + 1)).value }));
    const labels = () => [1, 2, 3].map((i) => +q('#l' + i).value);
    const join = (t, ev, mode) => {
      if (mode === 'naive') { const e = ev[ev.length - 1]; return { e, leak: e.d > t, miss: false }; }
      const past = ev.filter((e) => e.d <= t); if (!past.length) return { e: null, leak: false, miss: true, why: 'no record yet' };
      const e = past[past.length - 1], ttl = +q('#ttl').value;
      if (ttl < 30 && t - e.d > ttl) return { e, leak: false, miss: true, why: 'stale (TTL)' };
      return { e, leak: false, miss: false };
    };
    const jump = (to) => { clock = to; };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); const marks = [...DAYS, ...labels()].sort((a, b) => a - b).filter((m) => m > clock + 0.01); jump(marks.length ? marks[0] : 0); };
    q('#reset').onclick = () => { clock = 0; hold = 0; };
    AI.loop(root, (dt) => {
      if (playing) { if (hold > 0) { hold -= dt; if (hold <= 0) { hold = 0; clock = 0; } } else { clock += dt * 1.5 * +q('#sp').value; if (clock >= TMAX) { clock = TMAX; hold = 3; } } }
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const ev = vals(), L = labels(), mode = q('#mode').value, narrow = w < 560;
      const tl = { l: 44, r: 16, t: 26, b: 0 }, th = 170, v = AI.view({ w, h: th + 40 }, 0, TMAX, 0, 9.5, { l: tl.l, r: tl.r, t: 40, b: 22 });
      text(g, 'feature "support tickets (30 d)" for one customer — and the days when we must make a prediction', 8, 12, C.muted, '12px sans-serif');
      g.strokeStyle = C.border; g.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      for (let d = 0; d <= TMAX; d += 5) { g.beginPath(); g.moveTo(v.X(d), v.pad.t); g.lineTo(v.X(d), v.pad.t + v.ph); g.stroke(); text(g, 'day ' + d, v.X(d), v.pad.t + v.ph + 14, C.muted, '9px monospace', 'center'); }
      // step line
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); let px = v.X(0), py = v.Y(0); g.moveTo(px, py);
      ev.forEach((e) => { g.lineTo(v.X(e.d), py); py = v.Y(e.v); g.lineTo(v.X(e.d), py); }); g.lineTo(v.X(TMAX), py); g.stroke(); g.lineWidth = 1;
      ev.forEach((e, i) => { const on = clock >= e.d; g.fillStyle = on ? C.blue : C.bg3; g.strokeStyle = C.blue; g.beginPath(); g.arc(v.X(e.d), v.Y(e.v), 6, 0, 7); g.fill(); g.stroke(); text(g, 'v=' + e.v, v.X(e.d) + 8, v.Y(e.v) - 8, C.blue, '10px monospace'); });
      // labels
      const rows = L.map((t) => ({ t, r: join(t, ev, mode) }));
      rows.forEach((o, i) => {
        const x = v.X(o.t); g.strokeStyle = C.orange; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(x, v.pad.t); g.lineTo(x, v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
        text(g, 'E' + (i + 1), x, v.pad.t + 11, C.orange, '700 11px sans-serif', 'center');
        if (clock >= o.t && o.r.e) { // connector
          const p = smooth((clock - o.t) / 1.2), ex = v.X(o.r.e.d), ey = v.Y(o.r.e.v), col = o.r.leak ? C.red : C.green;
          g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); g.moveTo(x, v.Y(0) - 4); g.quadraticCurveTo((x + ex) / 2, v.Y(0) - 30 - Math.abs(ex - x) * 0.1, lerp(x, ex, p), lerp(v.Y(0) - 4, ey, p)); g.stroke(); g.lineWidth = 1;
          if (p > 0.95 && o.r.leak) text(g, 'LEAK: value from the future', ex, ey - 22, C.red, '700 11px sans-serif', 'center');
        }
      });
      // playhead
      g.strokeStyle = C.fg; g.lineWidth = 1.5; g.beginPath(); g.moveTo(v.X(clock), v.pad.t - 4); g.lineTo(v.X(clock), v.pad.t + v.ph); g.stroke(); g.lineWidth = 1;
      text(g, 'now = day ' + clock.toFixed(1), clamp(v.X(clock), 50, w - 50), v.pad.t - 10, C.fg, '600 10px monospace', 'center');
      // lower panels
      const y0 = th + 78, colW = (w - 24) / 3, ph = h - y0 - 6;
      const box = (i, title) => { const x = 8 + i * (colW + 4); g.strokeStyle = C.border; rr(g, x, y0, colW - 4, ph, 8); g.stroke(); text(g, title, x + 8, y0 + 15, C.muted, '600 11px sans-serif'); return x; };
      let x = box(0, 'OFFLINE store (history, appended)'); const done = ev.filter((e) => clock >= e.d);
      done.forEach((e, i) => { const rowY = y0 + 34 + i * 18; text(g, `day ${e.d}   tickets=${e.v}`, x + 8, rowY, C.fg, '11px monospace'); });
      if (!done.length) text(g, '(empty)', x + 8, y0 + 34, C.muted, '11px monospace');
      x = box(1, 'ONLINE store (latest only, key lookup)'); const last = done[done.length - 1];
      if (last) { const fl = clamp(1 - (clock - last.d) / 1.5, 0, 1); g.fillStyle = AI.mix(C.bg2, C.green, 0.15 + 0.5 * fl); rr(g, x + 6, y0 + 24, colW - 16, 40, 6); g.fill(); text(g, 'cust_42:', x + 12, y0 + 40, C.muted, '11px monospace'); text(g, `tickets = ${last.v}`, x + 12, y0 + 56, C.fg, '600 12px monospace'); text(g, `as of day ${last.d}`, x + colW - 20, y0 + 56, C.muted, '10px monospace', 'right'); }
      else text(g, '(no value yet)', x + 8, y0 + 34, C.muted, '11px monospace');
      text(g, 'overwritten at every update', x + 8, y0 + ph - 8, C.muted, '10px sans-serif');
      x = box(2, `TRAINING rows (${mode === 'naive' ? 'naive' : 'point-in-time'})`);
      let leaks = 0, built = 0;
      rows.forEach((o, i) => {
        if (clock >= o.t) { built++; if (o.r.leak) leaks++; const rowY = y0 + 34 + i * 20; const val = o.r.miss ? 'missing' : o.r.e.v, tag = o.r.leak ? 'LEAK' : o.r.miss ? o.r.why : 'ok'; text(g, `E${i + 1} t=${o.t}: x=${val} (day ${o.r.e ? o.r.e.d : '–'})`, x + 8, rowY, C.fg, '10px monospace'); text(g, tag, x + colW - 14, rowY + 11, o.r.leak ? C.red : o.r.miss ? C.orange : C.green, '700 10px monospace', 'right'); }
      });
      if (!built) text(g, '(waiting for first label time)', x + 8, y0 + 34, C.muted, '11px monospace');
      const nx = ev.find((e) => e.d > clock), nl = L.filter((t) => t > clock).sort((a, b) => a - b)[0];
      const cp = clock < 0.2 ? 'The clock starts at day 0. Features change over time; we will build one training row at each prediction day E1–E3.' : (nl !== undefined && (!nx || nl <= nx.d) ? `Next: building the training row for prediction day ${nl}. ${mode === 'naive' ? 'Naive join will take the LATEST value in the table, whenever it was written.' : 'Point-in-time join will take the latest value recorded on or before that day.'}` : nx ? `Next feature update on day ${nx.d}: the online store will overwrite its single value, the offline store appends a row.` : 'End of timeline.');
      if (cp !== lastCap) { lastCap = cp; q('#cap').textContent = cp; }
      const all = rows.map((o) => o.r), nl2 = all.filter((r) => r.leak).length, nm = all.filter((r) => r.miss).length;
      const o = `<span class="m">mode</span> <b>${mode === 'naive' ? 'naive' : 'point-in-time'}</b>   <span class="m">rows built so far</span> <b>${built}/3</b>   <span class="m">leaking rows (all 3)</span> <b class="${nl2 ? 'r' : 'g'}">${nl2}</b>   <span class="m">missing/stale</span> <b class="${nm ? 'o' : 'g'}">${nm}</b>\n<span class="m">Online value now (day ${clock.toFixed(1)}): ${last ? last.v : 'none'}. A correct offline row for the same moment must equal it: offline/online consistency.</span>`;
      if (o !== lastOut) { lastOut = o; q('#out').innerHTML = o; }
    }, { fps: 30 });
  };

  /* ====================================================================== expreg */
  AI.labs.expreg = function (root) {
    if (!document.getElementById('expreg-css')) { const st = document.createElement('style'); st.id = 'expreg-css'; st.textContent = '@keyframes erpop{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}.er-new{animation:erpop 1.1s ease-out}.er-tbl{width:100%;border-collapse:collapse;font-size:11.5px;font-family:ui-monospace,Menlo,Consolas,monospace}.er-tbl th,.er-tbl td{border-bottom:1px solid var(--border);padding:3px 6px;text-align:right;white-space:nowrap}.er-tbl th:first-child,.er-tbl td:first-child{text-align:left}.er-tbl tr.sel td{background:var(--bg3)}.er-tbl tr.run{cursor:pointer}.er-lane{display:flex;gap:6px;margin-top:8px}.er-lane>div{flex:1;border:1px solid var(--border);border-radius:8px;padding:5px 6px;min-height:54px;font-size:11px}.er-lane b{display:block;color:var(--muted);font-weight:600;margin-bottom:3px}.er-chip{display:inline-block;border:1px solid var(--border);border-radius:12px;padding:0 7px;margin:1px;font-family:ui-monospace,monospace;font-size:11px;background:var(--bg3)}.er-bar{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:6px}.er-bar i{display:block;height:100%;background:var(--green)}'; document.head.appendChild(st); }
    const { view, ctl } = AI.labShell(root, 'Experiment tracker & model registry', 'illustrative runs: metrics come from a made-up generator with seed noise · click a run, then promote it through the stages',
      '<div id="trainbar" class="caption" style="margin:0 0 4px"></div><div class="er-bar"><i id="bar" style="width:0"></i></div><div style="overflow-x:auto;margin-top:6px"><table class="er-tbl" id="tbl"></table></div><canvas style="height:120px;margin-top:8px"></canvas><div class="er-lane" id="lane"></div><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('gauc', 'Gate: validation AUC ≥', 0.8, 0.88, 0.002, 0.85)}${sl('glat', 'Gate: p95 latency ≤ (ms)', 4, 16, 0.5, 9)}${sl('gfair', 'Gate: fairness gap ≤', 0.02, 0.1, 0.005, 0.05)}
       <div id="gates" class="readout" style="white-space:normal"></div>
       <div class="row"><button class="btn small primary" id="promote">Promote ▶</button><button class="btn small" id="rollback">Roll back</button><button class="btn small" id="seeds">Check 5 seeds</button></div>
       ${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { gauc: (v) => v.toFixed(3), glat: (v) => v + ' ms', gfair: (v) => v.toFixed(3), sp: (v) => v.toFixed(2) + '×' });
    const LRS = [0.3, 0.1, 0.05, 0.02], DEPTH = [3, 4, 6, 8, 5, 7], MAXR = 14;
    const aucFn = (lr, depth, seed) => { const r = AI.rng(seed * 7919 + depth * 31 + Math.round(lr * 1000)); const n = AI.randn(r); return 0.82 + 0.055 * (1 - Math.exp(-(depth - 1) / 3)) - 0.02 * Math.abs(Math.log10(lr / 0.05)) + 0.006 * n; };
    let runs = [], sel = -1, progress = 0, playing = true, prod = -1, stag = -1, arch = [], msg = '', seq = 0, flash = -1, seedOut = '', lastKey = '';
    const mk = (k) => { const r = AI.rng(100 + k * 13), lr = LRS[Math.floor(r() * 4)], depth = DEPTH[Math.floor(r() * 6)], seed = k % 5; return { id: 'r-' + (101 + k), lr, depth, seed, auc: aucFn(lr, depth, seed + k), p95: 2 + 1.1 * depth + 1.5 * r(), gap: 0.02 + 0.004 * depth + 0.03 * r(), tests: r() > 0.12, stage: 'None' }; };
    const gates = (r) => [['AUC', r.auc >= +q('#gauc').value, r.auc.toFixed(3) + ' ≥ ' + (+q('#gauc').value).toFixed(3)], ['latency', r.p95 <= +q('#glat').value, r.p95.toFixed(1) + ' ≤ ' + q('#glat').value + ' ms'], ['fairness', r.gap <= +q('#gfair').value, r.gap.toFixed(3) + ' ≤ ' + (+q('#gfair').value).toFixed(3)], ['tests', r.tests, r.tests ? 'pass' : 'FAIL']];
    const reset = () => { runs = []; sel = -1; progress = 0; prod = -1; stag = -1; arch = []; msg = ''; seq = 0; seedOut = ''; lastKey = ''; };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); progress = 1; };
    q('#reset').onclick = reset;
    q('#promote').onclick = () => {
      if (sel < 0) { msg = 'Select a run in the table first.'; lastKey = ''; return; }
      const r = runs[sel], gg = gates(r), bad = gg.filter((x) => !x[1]).map((x) => x[0]);
      if (bad.length) { msg = `BLOCKED: ${r.id} fails gate(s): ${bad.join(', ')}. Promotion is refused automatically.`; lastKey = ''; return; }
      if (r.stage === 'None') { r.stage = 'Staging'; if (stag >= 0 && stag !== sel) runs[stag].stage = 'None'; stag = sel; msg = `${r.id} promoted to Staging (all gates passed). Shadow/canary checks would run here.`; }
      else if (r.stage === 'Staging') { if (prod >= 0) { runs[prod].stage = 'Archived'; arch.push(prod); } r.stage = 'Production'; prod = sel; stag = -1; msg = `${r.id} is now Production. Previous model archived and kept for rollback.`; }
      else msg = `${r.id} is already ${r.stage}.`;
      flash = sel; lastKey = '';
    };
    q('#rollback').onclick = () => { if (!arch.length) { msg = 'Nothing to roll back to: no archived Production model.'; lastKey = ''; return; } const back = arch.pop(); if (prod >= 0) runs[prod].stage = 'Archived'; runs[back].stage = 'Production'; msg = `Rolled back: ${runs[back].id} is Production again (${runs[prod] ? runs[prod].id : '—'} archived).`; prod = back; flash = back; lastKey = ''; };
    q('#seeds').onclick = () => { if (sel < 0) { seedOut = 'Select a run first.'; lastKey = ''; return; } const r = runs[sel], a = [0, 1, 2, 3, 4].map((s) => aucFn(r.lr, r.depth, 50 + s)); const m = a.reduce((x, y) => x + y) / 5, sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / 4); seedOut = `Same config as ${r.id} with 5 seeds: ${a.map((x) => x.toFixed(3)).join(', ')} → mean ${m.toFixed(4)}, s ${sd.toFixed(4)}, SE ${(sd / Math.sqrt(5)).toFixed(4)}. Differences smaller than s are noise.`; lastKey = ''; };
    ['gauc', 'glat', 'gfair'].forEach((id) => q('#' + id).addEventListener('input', () => (lastKey = '')));
    const tbl = q('#tbl'), lane = q('#lane');
    tbl.addEventListener('click', (e) => { const tr = e.target.closest('tr[data-i]'); if (tr) { sel = +tr.dataset.i; lastKey = ''; } });
    const render = () => {
      const best = runs.reduce((b, r, i) => (b < 0 || r.auc > runs[b].auc ? i : b), -1);
      tbl.innerHTML = '<tr><th>run</th><th>lr</th><th>depth</th><th>seed</th><th>data</th><th>val AUC</th><th>p95 ms</th><th>fair gap</th><th>tests</th><th>stage</th></tr>' + runs.map((r, i) => `<tr class="run ${i === sel ? 'sel' : ''} ${i === runs.length - 1 && seq === runs.length && progress < 0.25 ? 'er-new' : ''}" data-i="${i}"><td>${r.id}${i === best ? ' ★' : ''}</td><td>${r.lr}</td><td>${r.depth}</td><td>${r.seed}</td><td>d7</td><td>${r.auc.toFixed(3)}</td><td>${r.p95.toFixed(1)}</td><td>${r.gap.toFixed(3)}</td><td>${r.tests ? '✓' : '✗'}</td><td>${r.stage}</td></tr>`).join('');
      const chips = (st) => runs.map((r, i) => (r.stage === st ? `<span class="er-chip ${i === flash ? 'er-new' : ''}">${r.id}</span>` : '')).join('');
      lane.innerHTML = ['None', 'Staging', 'Production', 'Archived'].map((s) => `<div><b>${s}</b>${chips(s)}</div>`).join('');
      const gd = sel >= 0 ? gates(runs[sel]) : [];
      q('#gates').innerHTML = sel < 0 ? 'Click a run to see its gate checks.' : `<b>${runs[sel].id}</b> gates:<br>` + gd.map((x) => `<span style="color:var(${x[1] ? '--green' : '--red'})">${x[1] ? '✔' : '✘'} ${x[0]}: ${x[2]}</span>`).join('<br>');
      q('#cap').textContent = msg || 'Runs are launched one after another; each row records parameters, seed, data version and metrics. A best-looking run is not automatically promotable: it must pass every gate.';
      q('#out').innerHTML = seedOut ? `<span class="m">${seedOut}</span>` : `<span class="m">Illustrative generator: AUC = f(depth, lr) + seed noise (σ≈0.006). ★ = highest AUC so far. All numbers are made up for teaching.</span>`;
    };
    AI.loop(root, (dt) => {
      let ch = false;
      if (playing && seq < MAXR) { progress += dt * +q('#sp').value / 2.4; if (progress >= 1) { runs.push(mk(seq)); seq++; progress = 0; ch = true; flash = -1; } }
      q('#bar').style.width = (seq >= MAXR ? 100 : progress * 100) + '%';
      q('#trainbar').textContent = seq >= MAXR ? `Search finished: ${MAXR} runs recorded.` : `Training run r-${101 + seq} … ${Math.round(progress * 100)}% (lr ${LRS[Math.floor(AI.rng(100 + seq * 13)() * 4)]}…)`;
      const key = runs.length + '|' + sel + '|' + msg + '|' + q('#gauc').value + q('#glat').value + q('#gfair').value + '|' + seedOut;
      if (ch || key !== lastKey) { render(); lastKey = key; }
      // scatter
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const v = AI.view(c, 0, MAXR + 0.5, 0.78, 0.9, { l: 40, r: 10, t: 14, b: 20 });
      g.strokeStyle = C.border; g.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      text(g, 'val AUC per run', 8, 11, C.muted, '10px sans-serif'); [0.8, 0.84, 0.88].forEach((y) => text(g, y.toFixed(2), v.pad.l - 4, v.Y(y) + 3, C.muted, '9px monospace', 'right'));
      const gy = +q('#gauc').value; g.strokeStyle = C.red; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(v.pad.l, v.Y(gy)); g.lineTo(v.pad.l + v.pw, v.Y(gy)); g.stroke(); g.setLineDash([]); text(g, 'gate', v.pad.l + v.pw - 2, v.Y(gy) - 3, C.red, '9px monospace', 'right');
      let bst = 0; g.strokeStyle = C.green; g.beginPath(); runs.forEach((r, i) => { bst = Math.max(bst, r.auc); i ? g.lineTo(v.X(i + 1), v.Y(bst)) : g.moveTo(v.X(1), v.Y(bst)); }); g.stroke();
      runs.forEach((r, i) => { const x = v.X(i + 1), y = v.Y(r.auc); g.fillStyle = r.stage === 'Production' ? C.green : i === sel ? C.orange : C.blue; g.beginPath(); g.arc(x, y, i === sel ? 6 : 4, 0, 7); g.fill(); });
    }, { fps: 20 });
  };

  /* ====================================================================== serving */
  AI.labs.serving = function (root) {
    const { view, ctl } = AI.labShell(root, 'Serving: cache, batching window and latency', 'illustrative discrete-event simulation (1 s on screen = 100 ms of simulated time at speed 1×) · dots are coloured by latency',
      '<canvas style="height:430px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `${sl('lam', 'Request rate (per second)', 20, 800, 10, 250)}${sl('hit', 'Cache hit rate', 0, 95, 5, 40)}${sl('mb', 'Max batch size', 1, 64, 1, 8)}${sl('win', 'Batching window (ms)', 0, 50, 1, 10)}
       ${sl('t0', 'Fixed cost per batch t₀ (ms)', 1, 20, 1, 8)}${sl('t1', 'Cost per item t₁ (ms)', 0.2, 4, 0.1, 1.5)}${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { lam: (v) => v + '/s', hit: (v) => v + '%', mb: (v) => v, win: (v) => v + ' ms', t0: (v) => v + ' ms', t1: (v) => (+v).toFixed(1) + ' ms', sp: (v) => v.toFixed(2) + '×' });
    const HIT_MS = 2, BINS = 30, BW = 5;
    let t, nextArr, queue, busyUntil, serving, vis, lat, comp, playing = true, rng, lastCap = '', lastOut = '', q50 = 0, q95 = 0, q99 = 0, hAcc = 0, thr = 0, hist = new Array(BINS).fill(0), id = 0;
    const reset = () => { t = 0; nextArr = 0; queue = []; busyUntil = 0; serving = null; vis = []; lat = []; comp = []; rng = AI.rng(12345); thr = 0; hist.fill(0); q50 = q95 = q99 = 0; };
    reset();
    const done = (r, ms) => { const l = ms - r.t0; lat.push(l); if (lat.length > 600) lat.shift(); comp.push(ms); r.lat = l; r.out = performance.now(); r.st = 'out'; };
    const stepSim = (dtMs) => {
      const lam = +q('#lam').value, h = +q('#hit').value / 100, B = +q('#mb').value, W = +q('#win').value, t0 = +q('#t0').value, t1 = +q('#t1').value, end = t + dtMs;
      while (t < end) {
        t += 0.5;
        while (nextArr <= t) {
          const r = { id: id++, t0: nextArr, st: 'in', born: performance.now(), hit: rng() < h };
          nextArr += -Math.log(1 - rng()) / lam * 1000;
          if (r.hit) { r.st = 'hit'; r.doneAt = r.t0 + HIT_MS; } else queue.push(r);
          vis.push(r);
        }
        vis.forEach((r) => { if (r.st === 'hit' && t >= r.doneAt) done(r, r.doneAt); });
        if (serving && t >= busyUntil) { serving.forEach((r) => done(r, busyUntil)); serving = null; }
        if (!serving && queue.length && (queue.length >= B || t - queue[0].t0 >= W)) { const b = Math.min(B, queue.length); serving = queue.splice(0, b); serving.forEach((r) => (r.st = 'srv')); busyUntil = t + t0 + t1 * b; serving.start = t; }
      }
      while (comp.length && comp[0] < t - 1000) comp.shift();
      if (vis.length > 400) vis = vis.filter((r) => r.st !== 'out' || performance.now() - r.out < 900);
    };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); stepSim(10); };
    q('#reset').onclick = reset;
    ['lam', 'hit', 'mb', 'win', 't0', 't1'].forEach((i) => q('#' + i).addEventListener('change', () => { }));
    AI.loop(root, (dt) => {
      if (playing) { stepSim(dt * 100 * +q('#sp').value); }
      const now = performance.now(), C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const B = +q('#mb').value, W = +q('#win').value, lam = +q('#lam').value, hr = +q('#hit').value / 100, t0 = +q('#t0').value, t1 = +q('#t1').value;
      const latCol = (l) => AI.mix(C.green, C.red, clamp(l / 120, 0, 1));
      // ---- pipeline row
      const rowY = 70, xs = { src: 40, cache: w * 0.22, batch: w * 0.46, srv: w * 0.72, out: w - 36 };
      text(g, 'requests', xs.src, 14, C.muted, '11px sans-serif', 'center'); text(g, 'cache', xs.cache, 14, C.muted, '11px sans-serif', 'center'); text(g, `batch box (≤${B} items or ${W} ms)`, xs.batch, 14, C.muted, '11px sans-serif', 'center'); text(g, 'model server', xs.srv, 14, C.muted, '11px sans-serif', 'center'); text(g, 'answers', xs.out, 14, C.muted, '11px sans-serif', 'center');
      g.strokeStyle = C.border; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(xs.src, rowY); g.lineTo(xs.out, rowY); g.stroke();
      g.fillStyle = AI.mix(C.bg3, C.blue, 0.25); rr(g, xs.cache - 30, rowY - 30, 60, 60, 8); g.fill(); g.strokeStyle = C.blue; rr(g, xs.cache - 30, rowY - 30, 60, 60, 8); g.stroke(); text(g, Math.round(hr * 100) + '% hit', xs.cache, rowY + 4, C.fg, '600 11px sans-serif', 'center');
      // batch box with slots
      const bx = xs.batch - 62, by = rowY - 34; g.strokeStyle = C.orange; g.lineWidth = 1.5; rr(g, bx, by, 124, 68, 8); g.stroke(); g.lineWidth = 1;
      const wait = queue.length ? clamp((t - queue[0].t0) / Math.max(W, 1), 0, 1) : 0;
      g.fillStyle = C.orange; g.globalAlpha = 0.35; g.fillRect(bx + 4, by + 62, 116 * wait, 3); g.globalAlpha = 1; text(g, 'window timer', bx + 4, by + 76, C.muted, '9px sans-serif');
      queue.slice(0, 64).forEach((r, i) => { const col = i % 16, rw = Math.floor(i / 16); g.fillStyle = latCol(t - r.t0); g.fillRect(bx + 6 + col * 7, by + 6 + rw * 9, 6, 7); });
      if (queue.length > 64) text(g, '+' + (queue.length - 64), bx + 118, by + 56, C.red, '10px monospace', 'right');
      text(g, 'queue ' + queue.length, bx + 62, by - 5, queue.length > 3 * B ? C.red : C.muted, '10px monospace', 'center');
      // server
      const sx = xs.srv - 46, sy = rowY - 34; g.fillStyle = serving ? AI.mix(C.bg3, C.purple, 0.45) : C.bg3; rr(g, sx, sy, 92, 68, 8); g.fill(); g.strokeStyle = C.purple; rr(g, sx, sy, 92, 68, 8); g.stroke();
      if (serving) { const p = clamp((t - serving.start) / (busyUntil - serving.start), 0, 1); g.fillStyle = C.purple; g.fillRect(sx + 6, sy + 56, 80 * p, 5); text(g, 'batch of ' + serving.length, sx + 46, sy + 24, C.fg, '600 11px sans-serif', 'center'); text(g, ((busyUntil - serving.start)).toFixed(1) + ' ms', sx + 46, sy + 42, C.muted, '10px monospace', 'center'); } else text(g, 'idle', sx + 46, sy + 38, C.muted, '11px sans-serif', 'center');
      // dots
      vis.forEach((r) => {
        let x, y; const age = (now - r.born) / 1000;
        if (r.st === 'out') { const p = (now - r.out) / 900; if (p > 1) return; x = lerp(r.hit ? xs.cache + 30 : xs.srv + 46, xs.out, smooth(p)); y = r.hit ? rowY - 46 + (r.id % 5) * 2 : rowY + 46 + (r.id % 5) * 2; g.globalAlpha = 1 - p * 0.5; g.fillStyle = latCol(r.lat); g.beginPath(); g.arc(x, y, 3.2, 0, 7); g.fill(); g.globalAlpha = 1; return; }
        if (r.st === 'in' || r.st === 'hit') { const p = clamp(age / 0.6, 0, 1); x = lerp(xs.src, xs.cache - 30, smooth(p)); y = rowY + ((r.id % 7) - 3) * 3; if (r.st === 'hit' && p >= 1) { x = xs.cache; y = rowY - 40; } g.fillStyle = r.hit ? C.blue : C.muted; g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); return; }
        if (r.st === 'srv') { const i = serving ? serving.indexOf(r) : 0; x = sx + 10 + (i % 10) * 8; y = sy + 8 + Math.floor(i / 10) * 8; g.fillStyle = latCol(t - r.t0); g.fillRect(x, y, 6, 6); }
      });
      vis = vis.filter((r) => !(r.st === 'out' && now - r.out > 900));
      // ---- histogram
      hAcc += dt; if (hAcc > 0.25) { hAcc = 0; hist.fill(0); const s = [...lat].sort((a, b) => a - b); lat.forEach((l) => hist[Math.min(BINS - 1, Math.floor(l / BW))]++); q50 = AI.quantile(s, 0.5) || 0; q95 = AI.quantile(s, 0.95) || 0; q99 = AI.quantile(s, 0.99) || 0; thr = comp.length; }
      const hy = 150, v = AI.view({ w, h: h }, 0, BINS * BW, 0, Math.max(10, ...hist) * 1.15, { l: 40, r: 14, t: hy + 10, b: 26 });
      text(g, 'latency histogram of the last ' + lat.length + ' answers (ms; last bin = overflow)', 8, hy + 2, C.muted, '11px sans-serif');
      g.strokeStyle = C.border; g.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      hist.forEach((n, i) => { g.fillStyle = latCol(i * BW + 2); g.globalAlpha = 0.8; g.fillRect(v.X(i * BW) + 1, v.Y(n), v.pw / BINS - 2, v.Y(0) - v.Y(n)); g.globalAlpha = 1; });
      [[q50, 'p50', C.blue], [q95, 'p95', C.orange], [q99, 'p99', C.red]].forEach(([x, n, col]) => { const px = v.X(Math.min(x, BINS * BW)); g.strokeStyle = col; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(px, v.pad.t); g.lineTo(px, v.pad.t + v.ph); g.stroke(); g.setLineDash([]); text(g, n + ' ' + x.toFixed(0), px + 3, v.pad.t + 11 + (n === 'p95' ? 12 : n === 'p99' ? 24 : 0), col, '10px monospace'); });
      [0, 30, 60, 90, 120, 150].forEach((x) => text(g, String(x), v.X(x), v.pad.t + v.ph + 14, C.muted, '9px monospace', 'center'));
      // ---- readouts
      const tb = t0 + t1 * B, capS = B / tb * 1000, miss = lam * (1 - hr), rho = miss / capS, bs = Math.min(B, Math.max(1, miss * Math.max(W, 0) / 1000 + 1));
      const cp = serving ? `A batch of ${serving.length} is on the server: it takes t₀ + t₁·b = ${t0} + ${t1}·${serving.length} = ${(busyUntil - serving.start).toFixed(1)} ms; everyone in it shares that cost.` : queue.length ? `${queue.length} cache misses wait in the batch box until it holds ${B} items or the oldest has waited ${W} ms.` : 'No misses waiting. Cache hits (blue) skip the model and answer in ~2 ms.';
      if (cp !== lastCap) { lastCap = cp; q('#cap').textContent = cp; }
      const o = `<span class="m">server capacity with full batches</span> <b>${capS.toFixed(0)}/s</b> <span class="m">(= ${B}/(${t0}+${t1}·${B}) ms)</span>   <span class="m">miss load</span> <b>${miss.toFixed(0)}/s</b>   <span class="m">utilisation</span> <b class="${rho > 1 ? 'r' : rho > 0.8 ? 'o' : 'g'}">${(rho * 100).toFixed(0)}%</b>\n<span class="m">measured</span> p50 <b>${q50.toFixed(0)}</b> · p95 <b>${q95.toFixed(0)}</b> · p99 <b>${q99.toFixed(0)} ms</b>   <span class="m">throughput (last 1 s sim)</span> <b>${thr}/s</b>   <span class="m">mean latency if no queueing: hit ${HIT_MS} ms, miss ≈ window + t₀ + t₁b</span>\n<span class="m">${rho > 1 ? 'Overloaded: the queue grows without bound — raise the batch size, the cache hit rate or add replicas.' : 'Stable: all offered misses can be served.'}</span>`;
      if (o !== lastOut) { lastOut = o; q('#out').innerHTML = o; }
    }, { fps: 30 });
  };

  /* ====================================================================== rollout */
  AI.labs.rollout = function (root) {
    const { view, ctl } = AI.labShell(root, 'Rollout: canary, shadow and A/B', 'illustrative: error counts are random draws from the rates you set · automatic rollback rule: z-test on errors or latency regression',
      '<canvas style="height:440px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<label>Strategy<select id="mode"><option value="canary">Canary ramp 1 → 5 → 25 → 50 → 100%</option><option value="shadow">Shadow (copy of traffic, output discarded)</option><option value="ab">A/B test 50 / 50 (conversion)</option></select></label>
       ${sl('pb', 'Baseline error rate', 0.1, 3, 0.1, 0.5)}${sl('pc', 'Candidate error rate', 0.1, 3, 0.1, 1.5)}${sl('lr', 'Candidate latency vs baseline', 0.7, 1.8, 0.05, 1)}
       ${sl('zt', 'Rollback if z-score ≥', 1.64, 3.3, 0.02, 2.58)}${sl('minn', 'Minimum canary requests before judging', 50, 1000, 50, 300)}
       ${sl('cb', 'A/B: baseline conversion', 5, 20, 0.5, 10)}${sl('lift', 'A/B: true lift (percentage points)', -1, 1.5, 0.1, 0.5)}
       ${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { pb: (v) => (+v).toFixed(1) + '%', pc: (v) => (+v).toFixed(1) + '%', lr: (v) => (+v).toFixed(2) + '×', zt: (v) => (+v).toFixed(2), minn: (v) => v, cb: (v) => v + '%', lift: (v) => (v > 0 ? '+' : '') + (+v).toFixed(1) + ' pp', sp: (v) => v.toFixed(2) + '×' });
    const STAGES = [0.01, 0.05, 0.25, 0.5, 1], STAGE_MIN = 4, TRAFFIC = 2000;
    let S, playing = true, lastCap = '', lastOut = '', tick = 0, rng = AI.rng(777), dots = [];
    const reset = () => { rng = AI.rng(1 + Math.floor(Math.random() * 1e6)); S = { min: 0, nb: 0, nc: 0, eb: 0, ec: 0, cb: 0, cc: 0, hist: [], outcome: null, first: null, latB: 80, latC: 80, acc: 0 }; dots = []; };
    reset();
    const binom = (n, p) => { if (n <= 0) return 0; if (n < 40) { let k = 0; for (let i = 0; i < n; i++) if (rng() < p) k++; return k; } const m = n * p, s = Math.sqrt(n * p * (1 - p)); return Math.max(0, Math.min(n, Math.round(m + s * AI.randn(rng)))); };
    const zOf = (xc, nc, xb, nb) => { if (nc < 1 || nb < 1) return 0; const pp = (xc + xb) / (nc + nb), se = Math.sqrt(pp * (1 - pp) * (1 / nc + 1 / nb)); return se > 0 ? (xc / nc - xb / nb) / se : 0; };
    const share = () => { const m = q('#mode').value; if (m === 'ab') return 0.5; if (m === 'shadow') return 1; const k = Math.min(4, Math.floor(S.min / STAGE_MIN)); return S.outcome === 'rollback' ? 0 : STAGES[k]; };
    const mps = () => (q('#mode').value === 'ab' ? 4 : 1);
    const stepSim = (dMin) => {
      if (S.outcome) return;
      const mode = q('#mode').value, pb = +q('#pb').value / 100, pc = +q('#pc').value / 100, lr = +q('#lr').value;
      S.acc += TRAFFIC * dMin; const n = Math.floor(S.acc); S.acc -= n; S.min += dMin;
      const sh = share(), nc = mode === 'ab' ? binom(n, 0.5) : mode === 'shadow' ? n : binom(n, sh), nb = mode === 'shadow' ? n : n - nc;
      if (mode === 'ab') { const p0 = +q('#cb').value / 100, p1 = p0 + (+q('#lift').value) / 100; S.cb += binom(nb, p0); S.cc += binom(nc, p1); S.nb += nb; S.nc += nc; }
      else { S.eb += binom(nb, pb); S.ec += binom(nc, pc); S.nb += nb; S.nc += nc; S.latB = 80 * (1 + 0.04 * AI.randn(rng) / Math.sqrt(Math.max(1, S.nb / 100))); S.latC = 80 * lr * (1 + 0.04 * AI.randn(rng) / Math.sqrt(Math.max(1, S.nc / 100))); }
      const z = mode === 'ab' ? zOf(S.cc, S.nc, S.cb, S.nb) : zOf(S.ec, S.nc, S.eb, S.nb);
      S.hist.push({ m: S.min, z, pb: S.nb ? (mode === 'ab' ? S.cb : S.eb) / S.nb : 0, pc: S.nc ? (mode === 'ab' ? S.cc : S.ec) / S.nc : 0 });
      const zt = +q('#zt').value, minn = +q('#minn').value;
      if (mode === 'ab') { if (!S.first && Math.abs(z) >= 1.96) S.first = { m: S.min, n: S.nc, z }; const p0 = +q('#cb').value / 100, d = Math.abs(+q('#lift').value) / 100; const need = d > 0 ? 2 * 7.84 * (p0 + d / 2) * (1 - p0 - d / 2) / (d * d) : Infinity; if (S.nc >= need) S.outcome = Math.abs(z) >= 1.96 ? (z > 0 ? 'win' : 'loss') : 'flat'; }
      else {
        const slow = S.latC > 1.25 * S.latB && S.nc >= minn;
        if (S.nc >= minn && (z >= zt || slow)) { S.outcome = mode === 'shadow' ? 'reject' : 'rollback'; S.why = z >= zt ? `error-rate z = ${z.toFixed(2)} ≥ ${zt.toFixed(2)}` : `p95 latency ${S.latC.toFixed(0)} ms > 1.25 × ${S.latB.toFixed(0)} ms`; S.at = S.min; }
        else if (mode === 'canary' && S.min >= STAGE_MIN * 5) S.outcome = 'promoted';
        else if (mode === 'shadow' && S.min >= 20) S.outcome = 'accept';
      }
    };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); stepSim(0.5); };
    q('#reset').onclick = reset;
    ['mode', 'pb', 'pc', 'lr', 'zt', 'minn', 'cb', 'lift'].forEach((i) => q('#' + i).addEventListener('change', reset));
    AI.loop(root, (dt) => {
      if (playing && !S.outcome) { tick += dt; const dMin = dt * mps() * +q('#sp').value; stepSim(dMin); }
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const mode = q('#mode').value, sh = share(), nowM = S.min;
      // ---- stage bar
      text(g, mode === 'canary' ? 'Canary ramp (each stage lasts ' + STAGE_MIN + ' simulated minutes · 1 s on screen = 1 min)' : mode === 'shadow' ? 'Shadow: the candidate receives a copy of 100% of traffic, users never see its answers' : 'A/B test: fixed 50/50 split (1 s on screen = 4 min)', 8, 14, C.muted, '12px sans-serif');
      const bx = 8, bw = w - 16, by = 24, bh = 22;
      if (mode === 'canary') STAGES.forEach((s, i) => { const x0 = bx + bw * i / 5, ww = bw / 5 - 4, p = clamp((nowM - i * STAGE_MIN) / STAGE_MIN, 0, 1); g.strokeStyle = C.border; rr(g, x0, by, ww, bh, 5); g.stroke(); g.fillStyle = S.outcome === 'rollback' ? C.red : C.green; g.globalAlpha = 0.5; g.fillRect(x0 + 1, by + 1, (ww - 2) * p, bh - 2); g.globalAlpha = 1; text(g, Math.round(s * 100) + '%', x0 + ww / 2, by + 15, C.fg, '600 11px sans-serif', 'center'); });
      else { g.strokeStyle = C.border; rr(g, bx, by, bw, bh, 5); g.stroke(); const nreq = mode === 'ab' ? 57700 : 0; if (mode === 'ab') { g.fillStyle = C.blue; g.globalAlpha = 0.4; g.fillRect(bx + 1, by + 1, (bw - 2) * clamp(S.nc / nreq, 0, 1), bh - 2); g.globalAlpha = 1; text(g, `candidate users ${S.nc.toLocaleString('en-US')} of ~57,700 planned (for +0.5 pp, 80% power)`, bx + bw / 2, by + 15, C.fg, '11px sans-serif', 'center'); } else text(g, 'minute ' + nowM.toFixed(1) + ' of 20', bx + bw / 2, by + 15, C.fg, '11px sans-serif', 'center'); }
      // ---- router diagram
      const ry = 105, rx = w * 0.18, bxn = w * 0.62, cy1 = ry - 26, cy2 = ry + 30;
      g.fillStyle = C.bg3; rr(g, 14, ry - 20, 90, 40, 8); g.fill(); g.strokeStyle = C.border; rr(g, 14, ry - 20, 90, 40, 8); g.stroke(); text(g, 'traffic', 59, ry - 2, C.fg, '600 11px sans-serif', 'center'); text(g, TRAFFIC + ' req/min', 59, ry + 12, C.muted, '10px monospace', 'center');
      const box = (y, label, col, sub) => { g.fillStyle = AI.mix(C.bg3, col, 0.25); rr(g, bxn, y - 20, 130, 40, 8); g.fill(); g.strokeStyle = col; rr(g, bxn, y - 20, 130, 40, 8); g.stroke(); text(g, label, bxn + 65, y - 3, C.fg, '600 11px sans-serif', 'center'); text(g, sub, bxn + 65, y + 12, C.muted, '10px monospace', 'center'); };
      box(cy1, 'Baseline v6', C.blue, S.nb.toLocaleString('en-US') + ' req'); box(cy2, 'Candidate v7', C.orange, S.nc.toLocaleString('en-US') + ' req' + (mode === 'shadow' ? ' (shadow)' : ''));
      const wB = mode === 'shadow' ? 1 : 1 - sh, wC = sh; g.lineWidth = 1 + 5 * wB; g.strokeStyle = C.blue; g.beginPath(); g.moveTo(104, ry); g.lineTo(bxn, cy1); g.stroke(); g.lineWidth = 1 + 5 * wC; g.strokeStyle = S.outcome === 'rollback' ? C.red : C.orange; g.setLineDash(mode === 'shadow' ? [5, 4] : []); g.beginPath(); g.moveTo(104, ry); g.lineTo(bxn, cy2); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      if (playing && !S.outcome && Math.random() < dt * 9 * +q('#sp').value) dots.push({ t: 0, c: rng() < sh ? 1 : 0 });
      dots.forEach((d) => { d.t += dt * 0.5 * +q('#sp').value; }); dots = dots.filter((d) => d.t < 1);
      dots.forEach((d) => { const ty = d.c ? cy2 : cy1; g.fillStyle = d.c ? C.orange : C.blue; g.beginPath(); g.arc(lerp(104, bxn, d.t), lerp(ry, ty, d.t), 3.2, 0, 7); g.fill(); });
      const sub = mode === 'shadow' ? 'users get baseline answers only' : sh > 0 ? `${(sh * 100).toFixed(0)}% of users → candidate` : 'no traffic to candidate';
      text(g, sub, rx + 20, ry + 50, C.muted, '11px sans-serif'); if (S.outcome === 'rollback') text(g, 'ROLLED BACK', 14, ry + 52, C.red, '700 13px sans-serif');
      if (S.outcome === 'promoted') text(g, 'PROMOTED to 100%', 14, ry + 52, C.green, '700 13px sans-serif');
      // metric tiles on right
      const tx = bxn + 150, tw = w - tx - 10; if (tw > 100) { const m = (S.nb ? (mode === 'ab' ? S.cb : S.eb) / S.nb : 0) * 100, mc = (S.nc ? (mode === 'ab' ? S.cc : S.ec) / S.nc : 0) * 100; text(g, mode === 'ab' ? 'conversion' : 'error rate', tx, cy1 - 8, C.muted, '10px sans-serif'); text(g, m.toFixed(2) + '%', tx, cy1 + 10, C.blue, '600 14px monospace'); text(g, mc.toFixed(2) + '%', tx, cy2 + 10, C.orange, '600 14px monospace'); }
      // ---- charts
      const cy0 = 190, ch = h - cy0 - 6, half = (w - 30) / 2;
      const mmax = mode === 'canary' ? STAGE_MIN * 5 : mode === 'shadow' ? 20 : Math.max(60, nowM * 1.1);
      const vz = AI.view({ w: half + 8, h: h }, 0, mmax, -1, 4.5, { l: 34, r: 4, t: cy0 + 14, b: 22 });
      text(g, 'z-score of candidate vs baseline', 8, cy0 + 8, C.muted, '11px sans-serif'); g.strokeStyle = C.border; g.strokeRect(vz.pad.l, vz.pad.t, vz.pw, vz.ph);
      const zt = mode === 'ab' ? 1.96 : +q('#zt').value; [[zt, C.red, mode === 'ab' ? 'significant 1.96' : 'rollback ' + zt.toFixed(2)], [0, C.border, '']].forEach(([zv, col, lab]) => { g.strokeStyle = col; g.setLineDash(zv ? [5, 4] : []); g.beginPath(); g.moveTo(vz.pad.l, vz.Y(zv)); g.lineTo(vz.pad.l + vz.pw, vz.Y(zv)); g.stroke(); g.setLineDash([]); if (lab) text(g, lab, vz.pad.l + vz.pw - 2, vz.Y(zv) - 3, col, '9px monospace', 'right'); });
      g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); S.hist.forEach((p, i) => { const x = vz.X(p.m), y = vz.Y(clamp(p.z, -1, 4.5)); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.lineWidth = 1;
      [0, 1, 2, 3, 4].forEach((zv) => text(g, String(zv), vz.pad.l - 4, vz.Y(zv) + 3, C.muted, '9px monospace', 'right')); text(g, 'minutes →', vz.pad.l + vz.pw, vz.pad.t + vz.ph + 14, C.muted, '9px sans-serif', 'right');
      const x2 = half + 20, ymaxE = mode === 'ab' ? 0.14 : Math.max(0.02, +q('#pb').value / 100 * 3, +q('#pc').value / 100 * 1.6), yminE = mode === 'ab' ? 0.06 : 0;
      const ve = AI.view({ w: w - x2 + 20 + x2 - 20, h: h }, 0, mmax, yminE, ymaxE, { l: x2 + 36, r: 8, t: cy0 + 14, b: 22 });
      text(g, mode === 'ab' ? 'cumulative conversion by arm' : 'cumulative error rate by arm', x2, cy0 + 8, C.muted, '11px sans-serif'); g.strokeStyle = C.border; g.strokeRect(ve.pad.l, ve.pad.t, ve.pw, ve.ph);
      [['pb', C.blue], ['pc', C.orange]].forEach(([k, col]) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); let st = false; S.hist.forEach((p) => { if (!(p.m > 0.9)) return; const x = ve.X(p.m), y = ve.Y(clamp(p[k], yminE, ymaxE)); st ? g.lineTo(x, y) : (g.moveTo(x, y), (st = true)); }); g.stroke(); g.lineWidth = 1; });
      [yminE, (yminE + ymaxE) / 2, ymaxE].forEach((yv) => text(g, (yv * 100).toFixed(1) + '%', ve.pad.l - 4, ve.Y(yv) + 3, C.muted, '9px monospace', 'right'));
      // ---- text
      const last = S.hist[S.hist.length - 1], z = last ? last.z : 0, zt2 = +q('#zt').value;
      let cp;
      if (S.outcome === 'rollback') cp = `ROLLBACK at minute ${S.at.toFixed(1)}: ${S.why}. Traffic returned to the baseline; only ${S.nc.toLocaleString('en-US')} requests ever reached the candidate.`;
      else if (S.outcome === 'promoted') cp = 'The candidate survived every stage with no rule firing: promote to 100% (and keep monitoring).';
      else if (S.outcome === 'reject') cp = `Shadow comparison rejected the candidate (${S.why}) without any user ever seeing it.`;
      else if (S.outcome === 'accept') cp = 'Shadow run finished: no regression detected; the candidate may proceed to a canary.';
      else if (S.outcome) cp = S.outcome === 'win' ? `A/B finished: candidate converts better (z = ${z.toFixed(2)}).` : S.outcome === 'loss' ? `A/B finished: candidate converts worse (z = ${z.toFixed(2)}).` : 'A/B finished at the planned sample size: no significant difference.';
      else cp = mode === 'canary' ? `Stage ${Math.min(5, Math.floor(nowM / STAGE_MIN) + 1)} of 5: ${(sh * 100).toFixed(0)}% of traffic to the candidate. The rule needs ≥ ${q('#minn').value} candidate requests and z ≥ ${zt2.toFixed(2)} (or a >25% p95 latency regression).` : mode === 'shadow' ? 'Shadow: both models score every request; only the baseline answer is returned. Compare errors, latency and disagreement.' : 'A/B: random 50/50 assignment. Fix the sample size in advance; stopping the first time |z| > 1.96 ("peeking") inflates false positives.';
      if (cp !== lastCap) { lastCap = cp; q('#caption') ; q('#cap').textContent = cp; }
      const o = mode === 'ab' ? `<span class="m">baseline</span> ${S.nb.toLocaleString('en-US')} users, <b>${S.nb ? (S.cb / S.nb * 100).toFixed(2) : '0'}%</b>   <span class="m">candidate</span> ${S.nc.toLocaleString('en-US')} users, <b>${S.nc ? (S.cc / S.nc * 100).toFixed(2) : '0'}%</b>   <span class="m">z</span> <b class="${Math.abs(z) >= 1.96 ? 'g' : 'o'}">${z.toFixed(2)}</b>\n<span class="m">${S.first ? `|z| first exceeded 1.96 at ${S.first.n.toLocaleString('en-US')} candidate users (minute ${S.first.m.toFixed(0)}) — early crossings can be false alarms.` : 'Not significant yet. Planned n per arm = 2(1.96+0.84)²·p̄(1−p̄)/δ² (80% power).'}</span>`
        : `<span class="m">baseline</span> ${S.nb.toLocaleString('en-US')} req, ${S.eb} errors (<b>${S.nb ? (S.eb / S.nb * 100).toFixed(2) : '0'}%</b>)   <span class="m">candidate</span> ${S.nc.toLocaleString('en-US')} req, ${S.ec} errors (<b>${S.nc ? (S.ec / S.nc * 100).toFixed(2) : '0'}%</b>)   <span class="m">z</span> <b class="${z >= zt2 ? 'r' : 'g'}">${z.toFixed(2)}</b>\n<span class="m">p95 latency</span> baseline <b>${S.latB.toFixed(0)} ms</b>, candidate <b>${S.latC.toFixed(0)} ms</b>   <span class="m">users exposed to candidate:</span> <b>${mode === 'shadow' ? 0 : S.nc.toLocaleString('en-US')}</b>`;
      if (o !== lastOut) { lastOut = o; q('#out').innerHTML = o; }
    }, { fps: 30 });
  };

  /* ====================================================================== driftmon */
  AI.labs.driftmon = function (root) {
    const { view, ctl } = AI.labShell(root, 'Drift monitor & retraining trigger', 'illustrative: each tick is one monitoring window · PSI/KS are computed exactly from the sampled windows · AUC is a made-up function of drift',
      '<canvas style="height:470px"></canvas><div class="caption" id="cap"></div><div class="readout" id="out" style="margin-top:8px"></div>',
      `<label>Feature that drifts<select id="ft"><option value="0">avg order value (importance 0.5)</option><option value="1">sessions per week (0.3)</option><option value="2">days since signup (0.2)</option></select></label>
       <label>Drift type<select id="ty"><option value="grad">Gradual mean shift</option><option value="jump">Sudden jump at window 10</option><option value="var">Variance growth</option><option value="none">No drift (noise only)</option></select></label>
       ${sl('amp', 'Drift size (σ)', 0, 3, 0.1, 1.5)}${sl('n', 'Window size (rows per tick)', 100, 2000, 100, 500)}${sl('thr', 'PSI alert threshold', 0.05, 0.5, 0.01, 0.25)}${sl('k', 'Consecutive windows above threshold', 1, 5, 1, 3)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="auto" checked> automatic retraining when the rule fires</label>
       <div class="row"><button class="btn small" id="retrain">Retrain now</button></div>
       ${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { amp: (v) => (+v).toFixed(1) + 'σ', n: (v) => v, thr: (v) => (+v).toFixed(2), k: (v) => v, sp: (v) => v.toFixed(2) + '×' });
    const FE = [{ n: 'order value', mu: 60, sd: 15, w: 0.5 }, { n: 'sessions/wk', mu: 5, sd: 2, w: 0.3 }, { n: 'days since signup', mu: 200, sd: 60, w: 0.2 }];
    const EDGE = [-1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816], NW = 60, LAG = 6;
    const COLS = ['blue', 'orange', 'teal'];
    let w, ref, hist, trueAuc, obsAuc, markers, run, over, alarm, playing = true, tAcc = 0, samp, rng, lastCap = '', lastOut = '', phase = -1, pt = 0, retrains;
    const shift = (i, win) => { const f = +q('#ft').value, ty = q('#ty').value, A = +q('#amp').value; if (i !== f || ty === 'none') return { m: 0, s: 1 }; if (ty === 'grad') return { m: A * Math.min(1, win / 30), s: 1 }; if (ty === 'jump') return { m: win >= 10 ? A : 0, s: 1 }; return { m: 0, s: 1 + 0.6 * A * Math.min(1, win / 30) }; };
    const live = (i, win) => { const f = FE[i], sh = shift(i, win); return { mu: f.mu + sh.m * f.sd, sd: f.sd * sh.s }; };
    const reset = () => { w = 0; ref = FE.map((f) => ({ mu: f.mu, sd: f.sd })); hist = [[], [], []]; trueAuc = []; obsAuc = []; markers = []; run = 0; over = 0; alarm = false; samp = null; rng = AI.rng(4242); phase = -1; pt = 0; retrains = 0; tick(); };
    const metrics = () => {
      const n = +q('#n').value, out = [], cur = FE.map((f, i) => live(i, w)); samp = [];
      for (let i = 0; i < 3; i++) {
        const xs = Array.from({ length: n }, () => cur[i].mu + cur[i].sd * AI.randn(rng)); samp.push(xs);
        const cnt = new Array(10).fill(0); xs.forEach((x) => { const z = (x - ref[i].mu) / ref[i].sd; let k = 0; while (k < 9 && z > EDGE[k]) k++; cnt[k]++; });
        let psi = 0; cnt.forEach((cn) => { const a = Math.max(cn / n, 1e-4); psi += (a - 0.1) * Math.log(a / 0.1); });
        const so = [...xs].sort((a, b) => a - b); let ks = 0; so.forEach((x, j) => { const F = Phi((x - ref[i].mu) / ref[i].sd); ks = Math.max(ks, Math.abs((j + 1) / n - F), Math.abs(j / n - F)); });
        out.push({ psi, ks });
      }
      return out;
    };
    function tick() {
      const m = metrics(); m.forEach((o, i) => hist[i].push(o));
      let sev = 0; for (let i = 0; i < 3; i++) { const cur = live(i, w), s = Math.min(1, (Math.abs(cur.mu - ref[i].mu) / ref[i].sd + Math.abs(cur.sd / ref[i].sd - 1) * 1.5) / 2); sev += FE[i].w * s; }
      const ta = 0.88 - 0.14 * sev + 0.003 * AI.randn(rng); trueAuc.push(ta); obsAuc.push(w >= LAG ? trueAuc[w - LAG] : null);
      const thr = +q('#thr').value, k = +q('#k').value, mx = Math.max(...m.map((o) => o.psi));
      over = mx > thr ? over + 1 : 0;
      const perf = obsAuc[w] !== null && obsAuc[w] < 0.85;
      if (phase < 0 && (over >= k || perf)) { if (!alarm) markers.push({ w, t: 'alarm' }); alarm = true; if (q('#auto').checked) startRetrain(); }
      if (over === 0 && !perf && phase < 0) alarm = false;
      w++;
    }
    const startRetrain = () => { if (phase >= 0) return; phase = 0; pt = 0; markers.push({ w, t: 'retrain' }); };
    const finishRetrain = () => { ref = FE.map((f, i) => { const l = live(i, w); return { mu: l.mu, sd: l.sd }; }); phase = -1; over = 0; alarm = false; retrains++; markers.push({ w, t: 'deploy' }); };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); if (w < NW) tick(); };
    q('#reset').onclick = reset; q('#retrain').onclick = startRetrain;
    ['ft', 'ty', 'amp', 'n'].forEach((i) => q('#' + i).addEventListener('change', reset));
    reset();
    const PH = ['collect recent data', 'train candidate', 'validate on holdout', 'deploy through registry gates'];
    AI.loop(root, (dt) => {
      if (playing) {
        if (phase >= 0) { pt += dt * +q('#sp').value / 1.1; if (pt >= 1) { pt = 0; phase++; if (phase >= 4) finishRetrain(); } }
        tAcc += dt * +q('#sp').value; if (tAcc >= 1.1) { tAcc = 0; if (w < NW) tick(); else reset(); }
      }
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const thr = +q('#thr').value, f = +q('#ft').value, cur = Math.max(0, w - 1), lw = Math.min(W * 0.42, 380);
      // histogram of drifting feature, live window vs reference
      const xs = samp ? samp[f] : [], rf = ref[f], lo = rf.mu - 4 * rf.sd, hi = rf.mu + 4 * rf.sd + (shift(f, w).m > 0 ? shift(f, w).m * FE[f].sd : 0), bins = 30, bwid = (hi - lo) / bins, cnt = new Array(bins).fill(0);
      xs.forEach((x) => { const k = Math.floor((x - lo) / bwid); if (k >= 0 && k < bins) cnt[k]++; });
      const ymaxH = 1.25 / (rf.sd * Math.sqrt(2 * Math.PI)) * 1.0; const v = AI.view({ w: lw, h: 200 }, lo, hi, 0, ymaxH, { l: 10, r: 8, t: 26, b: 22 });
      text(g, `live window ${Math.max(0, w - 1)} vs training reference — ${FE[f].n}`, 10, 14, C.muted, '11px sans-serif'); g.strokeStyle = C.border; g.strokeRect(v.pad.l, v.pad.t, v.pw, v.ph);
      cnt.forEach((n, k) => { const dens = n / Math.max(1, xs.length) / bwid; g.fillStyle = C.orange; g.globalAlpha = 0.6; g.fillRect(v.X(lo + k * bwid), v.Y(dens), v.pw / bins - 1, v.Y(0) - v.Y(dens)); }); g.globalAlpha = 1;
      g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); for (let x = lo; x <= hi; x += bwid / 3) { const y = Math.exp(-0.5 * ((x - rf.mu) / rf.sd) ** 2) / (rf.sd * Math.sqrt(2 * Math.PI)); x === lo ? g.moveTo(v.X(x), v.Y(y)) : g.lineTo(v.X(x), v.Y(y)); } g.stroke(); g.lineWidth = 1;
      text(g, 'blue = reference (training) · orange = live', 10, 214, C.muted, '10px sans-serif');
      // PSI chart
      const px = lw + 24, pw = W - px - 10, py = 26, ph = 170, ymax = Math.max(0.6, thr * 2.2, ...hist.map((a) => Math.max(0, ...a.map((o) => o.psi)))) * 1.05;
      text(g, 'PSI per feature per window (10 reference-quantile bins)', px, 14, C.muted, '11px sans-serif'); g.strokeStyle = C.border; g.strokeRect(px, py, pw, ph);
      const X = (i) => px + (i / NW) * pw, Y = (p) => py + ph - Math.min(1, p / ymax) * ph;
      g.strokeStyle = C.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(px, Y(thr)); g.lineTo(px + pw, Y(thr)); g.stroke(); g.setLineDash([]); text(g, 'threshold ' + thr.toFixed(2), px + pw - 3, Y(thr) - 3, C.red, '10px monospace', 'right');
      for (let i = 0; i < 3; i++) { g.strokeStyle = C[COLS[i]]; g.lineWidth = i === f ? 2.5 : 1.3; g.beginPath(); hist[i].forEach((o, j) => (j ? g.lineTo(X(j), Y(o.psi)) : g.moveTo(X(j), Y(o.psi)))); g.stroke(); text(g, FE[i].n, px + 6, py + 12 + i * 12, C[COLS[i]], '10px sans-serif'); } g.lineWidth = 1;
      markers.forEach((m) => { const x = X(m.w); g.strokeStyle = m.t === 'alarm' ? C.red : m.t === 'retrain' ? C.purple : C.green; g.beginPath(); g.moveTo(x, py); g.lineTo(x, py + ph); g.stroke(); text(g, m.t === 'alarm' ? 'ALARM' : m.t === 'retrain' ? 'retrain' : 'deployed', x + 3, py + ph - 4 - (m.t === 'deploy' ? 12 : m.t === 'retrain' ? 0 : 24), g.strokeStyle, '9px monospace'); });
      // AUC chart
      const ay = 262, ah = 110, ax = 40, aw = W - ax - 12, Ya = (a) => ay + ah - ((a - 0.72) / (0.9 - 0.72)) * ah;
      text(g, 'model quality (AUC, illustrative): faint = true, solid = what you can measure once labels arrive (' + LAG + ' windows late)', 10, ay - 8, C.muted, '11px sans-serif'); g.strokeStyle = C.border; g.strokeRect(ax, ay, aw, ah);
      [0.75, 0.8, 0.85, 0.9].forEach((a) => text(g, a.toFixed(2), ax - 4, Ya(a) + 3, C.muted, '9px monospace', 'right'));
      g.strokeStyle = C.red; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(ax, Ya(0.85)); g.lineTo(ax + aw, Ya(0.85)); g.stroke(); g.setLineDash([]); text(g, 'performance rule: AUC < 0.85', ax + aw - 3, Ya(0.85) - 3, C.red, '9px monospace', 'right');
      g.globalAlpha = 0.4; g.strokeStyle = C.fg; g.beginPath(); trueAuc.forEach((a, j) => (j ? g.lineTo(ax + j / NW * aw, Ya(a)) : g.moveTo(ax, Ya(a)))); g.stroke(); g.globalAlpha = 1;
      g.strokeStyle = C.purple; g.lineWidth = 2; g.beginPath(); let st = false; obsAuc.forEach((a, j) => { if (a === null) return; const x = ax + j / NW * aw; st ? g.lineTo(x, Ya(a)) : (g.moveTo(x, Ya(a)), (st = true)); }); g.stroke(); g.lineWidth = 1;
      // retrain pipeline
      const ry = 396, rw = (W - 20) / 4;
      PH.forEach((p, i) => { const x = 10 + i * rw, act = phase === i, dn = phase > i; g.fillStyle = act ? AI.mix(C.bg3, C.purple, 0.5) : dn ? AI.mix(C.bg3, C.green, 0.35) : C.bg3; rr(g, x, ry, rw - 6, 34, 7); g.fill(); g.strokeStyle = act ? C.purple : C.border; rr(g, x, ry, rw - 6, 34, 7); g.stroke(); if (act) { g.fillStyle = C.purple; g.fillRect(x + 4, ry + 28, (rw - 14) * pt, 3); } text(g, (i + 1) + ' ' + p, x + (rw - 6) / 2, ry + 18, C.fg, '10px sans-serif', 'center'); });
      text(g, phase < 0 ? 'retraining pipeline idle' + (alarm ? ' — alarm raised: waiting for you (automatic retraining is off)' : '') : 'retraining job running — the new model must still pass the registry gates', 10, ry + 52, phase < 0 ? C.muted : C.purple, '11px sans-serif');
      const lastP = hist.map((a) => a[a.length - 1] || { psi: 0, ks: 0 }), k = +q('#k').value;
      const cp = phase >= 0 ? `Retraining: step ${Math.min(4, phase + 1)} of 4 (${PH[Math.min(3, phase)]}).` : alarm ? `ALARM: PSI stayed above ${thr.toFixed(2)} for ${over} window(s) (rule needs ${k}) — or the measured AUC fell below 0.85.` : over > 0 ? `PSI above threshold for ${over}/${k} windows — not yet an alarm (one noisy window must not trigger a retrain).` : 'All features stable: PSI below the threshold.';
      if (cp !== lastCap) { lastCap = cp; q('#cap').textContent = cp; }
      const o = lastP.map((p, i) => `<span class="m">${FE[i].n}</span> PSI <b class="${p.psi > thr ? 'r' : p.psi > thr / 2.5 ? 'o' : 'g'}">${p.psi.toFixed(3)}</b> KS <b>${p.ks.toFixed(3)}</b>`).join('   ') + `\n<span class="m">window ${w}/${NW}   n = ${q('#n').value} → KS 5% critical value ≈ ${(1.36 * Math.sqrt(2 / +q('#n').value)).toFixed(3)} (KS alone fires on tiny shifts when n is large; PSI measures size)   retrains so far: ${retrains}</span>`;
      if (o !== lastOut) { lastOut = o; q('#out').innerHTML = o; }
    }, { fps: 30 });
  };

  /* ====================================================================== governance */
  AI.labs.governance = function (root) {
    if (!document.getElementById('gov-css')) { const st = document.createElement('style'); st.id = 'gov-css'; st.textContent = '.gv-tier{border-radius:8px;padding:8px 12px;font-weight:600;transition:background .6s,color .6s}.gv-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.gv-grid label{display:flex;flex-direction:column;font-size:11px;color:var(--muted);gap:2px}.gv-grid input,.gv-grid textarea{background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--fg);padding:4px 6px;font:12px sans-serif;resize:vertical}.gv-grid .wide{grid-column:1/3}.gv-list{margin:8px 0 0;padding:0;list-style:none;font-size:12px}.gv-list li{padding:2px 0;transition:color .5s}.gv-bar{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin-top:6px}.gv-bar i{display:block;height:100%;background:var(--green)}@media(max-width:600px){.gv-grid{grid-template-columns:1fr}.gv-grid .wide{grid-column:1}}'; document.head.appendChild(st); }
    const { view, ctl } = AI.labShell(root, 'Governance: model card, risk tier & release gate', 'ILLUSTRATIVE tiering inspired by risk-based regulation (not legal advice) · fill the card, tick the controls, watch the gate',
      `<div id="tier" class="gv-tier"></div><div class="gv-bar"><i id="comp" style="width:0"></i></div><div class="caption" id="cap" style="margin-top:4px"></div>
       <canvas style="height:96px;margin-top:6px"></canvas>
       <div class="gv-grid" id="form"></div><ul class="gv-list" id="list"></ul><div class="readout" id="out" style="margin-top:8px;white-space:pre-wrap"></div>`,
      `<label>Use case (preset)<select id="uc"><option value="rec">Movie recommender</option><option value="spam">Spam filter</option><option value="chat">Customer-support chatbot</option><option value="credit" selected>Loan approval</option><option value="hire">CV screening for hiring</option><option value="med">Medical triage</option><option value="score">Social scoring of citizens</option></select></label>
       <b style="font-size:12px">System properties</b>
       ${['pd:Uses personal data', 'legal:Decisions with legal or similarly significant effect', 'safety:Safety / health impact', 'vuln:Affects vulnerable groups', 'human:Interacts with people / generates content', 'banned:Social scoring or real-time biometric ID'].map((s) => { const [k, l] = s.split(':'); return `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="p_${k}"> ${l}</label>`; }).join('')}
       <b style="font-size:12px">Controls in place</b>
       ${['oversight:Human oversight and appeal', 'logging:Logging / audit trail', 'inform:Users told they face an AI', 'robust:Robustness / security tested', 'dpia:Privacy impact assessment'].map((s) => { const [k, l] = s.split(':'); return `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="c_${k}"> ${l}</label>`; }).join('')}
       ${sl('tA', 'True-positive rate, group A', 0.4, 1, 0.01, 0.82)}${sl('tB', 'True-positive rate, group B', 0.4, 1, 0.01, 0.7)}
       <div class="row"><button class="btn small" id="fill">Fill example card</button><button class="btn small" id="clear">Clear</button></div>`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    wire(root, { tA: (v) => (+v).toFixed(2), tB: (v) => (+v).toFixed(2) });
    const FIELDS = [['name', 'Model name & version', 0], ['owner', 'Owner / contact', 0], ['use', 'Intended use', 1], ['oos', 'Out-of-scope uses', 1], ['data', 'Training data (source, period, gaps)', 1], ['eval', 'Evaluation metrics (overall)', 1], ['sub', 'Metrics per subgroup', 1], ['lim', 'Limitations & risks', 1], ['eth', 'Ethical considerations', 1], ['mon', 'Monitoring & retraining plan', 1]];
    q('#form').innerHTML = FIELDS.map(([k, l, big]) => `<label class="${big ? 'wide' : ''}">${l}${big ? `<textarea id="f_${k}" rows="2"></textarea>` : `<input type="text" id="f_${k}">`}</label>`).join('');
    const EX = { name: 'loan-risk-gbm v7', owner: 'credit-ml@example.org', use: 'Rank consumer loan applications for manual review.', oos: 'Not for final automated rejection; not for business loans.', data: 'Applications 2022-2025, 1.2M rows; thin-file applicants under-represented.', eval: 'AUC 0.88, precision 0.49 / recall 0.43 at threshold 0.3.', sub: 'TPR 0.82 (group A) vs 0.70 (group B); selection rates 0.45 vs 0.30.', lim: 'Accuracy drops for first-time borrowers; sensitive to income reporting changes.', eth: 'Proxy features (postcode) audited; applicants can request human review.', mon: 'PSI weekly on top-10 features; retrain when PSI>0.25 for 3 weeks or AUC<0.85.' };
    const PRE = { rec: { }, spam: { }, chat: { human: 1 }, credit: { pd: 1, legal: 1 }, hire: { pd: 1, legal: 1, vuln: 1 }, med: { pd: 1, safety: 1, legal: 1, vuln: 1 }, score: { pd: 1, legal: 1, banned: 1 } };
    const setPre = () => { const p = PRE[q('#uc').value]; ['pd', 'legal', 'safety', 'vuln', 'human', 'banned'].forEach((k) => (q('#p_' + k).checked = !!p[k])); };
    q('#uc').onchange = setPre; setPre();
    q('#fill').onclick = () => FIELDS.forEach(([k]) => (q('#f_' + k).value = EX[k])); q('#clear').onclick = () => FIELDS.forEach(([k]) => (q('#f_' + k).value = ''));
    const pr = (k) => q('#p_' + k).checked, cc = (k) => q('#c_' + k).checked;
    let compShown = 0, passPos = 0, lastKey = '', ready = false, tierCol = 'green';
    const evalState = () => {
      const filled = FIELDS.filter(([k]) => q('#f_' + k).value.trim().length >= 8).length, comp = filled / FIELDS.length;
      const gap = +q('#tA').value - +q('#tB').value;
      const tier = pr('banned') ? 'Unacceptable' : (pr('legal') || pr('safety')) ? 'High' : pr('human') ? 'Limited' : 'Minimal';
      const items = [['Model card ≥ 80% complete', comp >= 0.8, 'every tier']];
      if (tier === 'Limited' || tier === 'High') items.push(['Users informed they interact with AI', cc('inform'), 'transparency']);
      if (tier === 'High') {
        items.push(['Subgroup TPR gap ≤ 0.05 (now ' + Math.abs(gap).toFixed(2) + ')', Math.abs(gap) <= 0.05, 'bias audit']);
        items.push(['Human oversight and appeal route', cc('oversight'), 'oversight']);
        items.push(['Logging / audit trail', cc('logging'), 'traceability']);
        items.push(['Robustness / security tested', cc('robust'), 'robustness']);
        items.push(['Monitoring plan written in the card', q('#f_mon').value.trim().length >= 8, 'post-market monitoring']);
        if (pr('pd')) items.push(['Privacy impact assessment done', cc('dpia'), 'data protection']);
      }
      const okAll = items.every((i) => i[1]) && tier !== 'Unacceptable';
      return { comp, gap, tier, items, okAll, filled };
    };
    const TC = { Minimal: 'green', Limited: 'yellow', High: 'orange', Unacceptable: 'red' };
    AI.loop(root, (dt) => {
      const s = evalState(), C = AI.col(), key = JSON.stringify([s.tier, s.items.map((i) => i[1]), s.filled, s.gap.toFixed(2), FIELDS.map((f) => q('#f_' + f[0]).value)]);
      compShown += (s.comp - compShown) * (1 - Math.exp(-dt * 3)); q('#comp').style.width = (compShown * 100).toFixed(1) + '%';
      if (key !== lastKey) {
        lastKey = key; const col = TC[s.tier]; const t = q('#tier'); t.style.background = `color-mix(in srgb, var(--${col}) 22%, var(--bg2))`; t.style.border = `1px solid var(--${col})`;
        t.textContent = `Risk tier: ${s.tier}${s.tier === 'Unacceptable' ? ' — prohibited practice; no release path' : s.tier === 'High' ? ' — strict requirements before release' : s.tier === 'Limited' ? ' — transparency duties' : ' — voluntary good practice'}`;
        q('#list').innerHTML = s.items.map((i) => `<li style="color:var(${i[1] ? '--green' : '--red'})">${i[1] ? '☑' : '☐'} ${i[0]} <span style="color:var(--muted)">· ${i[2]}</span></li>`).join('');
        q('#cap').textContent = `Model card completeness ${s.filled}/${FIELDS.length} fields. ${s.okAll ? 'All required checks pass: the release gate opens.' : s.tier === 'Unacceptable' ? 'Blocked by policy.' : 'Release gate closed: ' + s.items.filter((i) => !i[1]).length + ' requirement(s) missing.'}`;
        const card = FIELDS.map(([k, l]) => `${l}: ${q('#f_' + k).value.trim() || '— (missing)'}`).join('\n');
        q('#out').textContent = 'MODEL CARD PREVIEW\n' + card;
      }
      passPos += ((s.okAll ? 1 : 0.55) - passPos) * (1 - Math.exp(-dt * 1.6));
      const g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const y = H / 2, x0 = 20, x1 = W - 20, gx = x0 + (x1 - x0) * 0.62;
      g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
      g.fillStyle = s.okAll ? C.green : C.red; g.fillRect(gx - 3, y - 26, 6, 52); text(g, s.okAll ? 'RELEASE GATE: OPEN' : 'RELEASE GATE: BLOCKED', gx, y - 30, s.okAll ? C.green : C.red, '700 11px sans-serif', 'center');
      const cx = lerp(x0, x1 - 30, passPos), pulse = s.okAll ? 0 : Math.sin(performance.now() / 300) * 0.5;
      g.fillStyle = C.bg3; g.strokeStyle = C.blue; rr(g, cx - 14 + pulse, y - 17, 28, 34, 4); g.fill(); g.stroke(); text(g, 'card', cx + pulse, y + 4, C.fg, '9px sans-serif', 'center'); text(g, 'production', x1, y + 22, C.muted, '10px sans-serif', 'right'); text(g, 'draft', x0, y + 22, C.muted, '10px sans-serif');
      g.lineWidth = 1;
    }, { fps: 30 });
  };

  /* ====================================================================== capstone */
  AI.labs.capstone = function (root) {
    if (!document.getElementById('cap-css')) { const st = document.createElement('style'); st.id = 'cap-css'; st.textContent = '.cp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin-top:8px}.cp-card{border:1px solid var(--border);border-radius:8px;padding:8px 10px;background:var(--bg);font-size:12px;transition:opacity .6s,border-color .6s}.cp-card h5{margin:0 0 4px;font-size:12px;color:var(--muted)}.cp-card .big{font-size:20px;font-weight:700;font-family:ui-monospace,monospace}.cp-card.off{opacity:.25}'; document.head.appendChild(st); }
    const { view, ctl } = AI.labShell(root, 'End-to-end project simulator', 'ILLUSTRATIVE: every number comes from transparent made-up formulas (binormal score model, round prices) — the interactions are the lesson',
      '<canvas style="height:150px"></canvas><div class="caption" id="cap"></div><div class="cp-grid" id="cards"></div><div class="readout" id="out" style="margin-top:8px;white-space:pre-wrap"></div>',
      `${sl('n', 'Dataset size (rows)', 3, 6, 0.1, 5)}
       <label>Preprocessing<select id="pre"><option value="0">None (raw columns)</option><option value="1" selected>Impute + scale (one shared pipeline)</option><option value="2">+ engineered features</option></select></label>
       <label>Model<select id="mod"><option value="0">Logistic regression</option><option value="1">Random forest (300 trees)</option><option value="2" selected>Gradient boosting</option><option value="3">Small neural network</option></select></label>
       ${sl('thr', 'Decision threshold on probability', 0.05, 0.95, 0.01, 0.3)}
       <label>Deployment mode<select id="dep"><option value="0">Batch (nightly)</option><option value="1" selected>Real-time API</option><option value="2">Edge device</option></select></label>
       <label>Monitoring<select id="mon"><option value="0">None</option><option value="1">Basic alerts</option><option value="2" selected>Full drift + auto retrain</option></select></label>
       ${transport()}`);
    const q = (s) => root.querySelector(s), c = AI.canvas(AI.$('canvas', view));
    const nOf = (v) => Math.round(Math.pow(10, v));
    wire(root, { n: (v) => nOf(+v).toLocaleString('en-US'), thr: (v) => (+v).toFixed(2), sp: (v) => v.toFixed(2) + '×' });
    const STAGES = ['Data', 'Preprocess', 'Train', 'Evaluate', 'Deploy', 'Monitor'], MN = ['logistic regression', 'random forest', 'gradient boosting', 'small neural net'];
    const CAP = [0.8, 0.86, 0.88, 0.87], HUN = [0.2, 0.6, 0.6, 1.5], INF = [0.2, 8, 3, 6], SIZE = [0.01, 80, 6, 12], TRC = [0.5, 6, 4, 15];
    const PRE = [[-0.06, -0.015, -0.015, -0.07], [0, 0, 0, 0], [0.025, 0.01, 0.01, 0.01]];
    const PI = 0.1, CFN = 50, CFP = 5;
    let st = 0, ph = 0, playing = true, lastKey = '', R = null, pos = 0;
    const compute = () => {
      const n = nOf(+q('#n').value), pre = +q('#pre').value, m = +q('#mod').value, thr = +q('#thr').value, dep = +q('#dep').value, mon = +q('#mon').value;
      const auc = clamp(CAP[m] - HUN[m] * 0.06 / Math.sqrt(n / 1000) + PRE[pre][m], 0.55, 0.97), d = Math.SQRT2 * PhiInv(auc);
      const lg = Math.log(thr / (1 - thr)), xc = (lg + d * d / 2 - Math.log(PI / (1 - PI))) / d, tpr = 1 - Phi(xc - d), fpr = 1 - Phi(xc), prec = PI * tpr / (PI * tpr + (1 - PI) * fpr || 1), f1 = 2 * prec * tpr / (prec + tpr || 1);
      const bizCost = 1000 * (PI * (1 - tpr) * CFN + (1 - PI) * fpr * CFP);
      const perDay = 200000; let p95, lat = '', fail = null, infraDay, repl = 0;
      if (dep === 1) { p95 = (3 + 6 + INF[m] + 2) * 1.6; lat = p95.toFixed(1) + ' ms p95 (budget 30)'; repl = Math.max(2, Math.ceil(20 * (INF[m] + 9) / 1000 / 0.6)); infraDay = repl * 0.08 * 24; if (p95 > 30) fail = 'latency budget 30 ms exceeded'; }
      else if (dep === 0) { const hrs = perDay * INF[m] / 1000 / 4 / 3600; lat = `nightly run ≈ ${(hrs * 60).toFixed(1)} min on 4 workers`; infraDay = 4 * 0.10 * Math.max(hrs, 0.05) + 0.05; }
      else { p95 = INF[m] * 6 + 4; lat = `${p95.toFixed(0)} ms on device, model ${SIZE[m]} MB (limit 50)`; infraDay = 0.0004 * perDay / 1000 * 1000 / 1000 + 0.1; if (SIZE[m] > 50) fail = `model ${SIZE[m]} MB does not fit the 50 MB device limit`; if (p95 > 50) fail = 'on-device latency over 50 ms'; }
      const cost1k = infraDay / (perDay / 1000), train = TRC[m] * n / 1e6;
      const mit = [0, 0.5, 0.9][mon], auc6 = auc - 0.06 * (1 - mit);
      const skew = [2, 1, 2][pre], drift = [3, 1.5, 0.5][mon], dpl = [0.5, 1, 1.5][dep], cx = [0.3, 1, 0.8, 1][m], risk = skew + drift + dpl + cx + (fail ? 3 : 0), level = risk >= 6 ? 'HIGH' : risk >= 4 ? 'MEDIUM' : 'LOW';
      return { n, pre, m, thr, dep, mon, auc, d, tpr, fpr, prec, f1, bizCost, lat, fail, cost1k, infraDay, train, auc6, risk, level, repl, skew, drift, dpl, cx };
    };
    const reset = () => { R = compute(); st = 0; ph = 0; pos = 0; lastKey = ''; };
    q('#play').onclick = () => { playing = !playing; setPlay(root, playing); };
    q('#step').onclick = () => { playing = false; setPlay(root, false); st = Math.min(6, st + 1); ph = 0; lastKey = ''; };
    q('#reset').onclick = reset;
    ['n', 'pre', 'mod', 'thr', 'dep', 'mon'].forEach((i) => q('#' + i).addEventListener('change', reset));
    ['n', 'thr'].forEach((i) => q('#' + i).addEventListener('input', () => { R = compute(); lastKey = ''; }));
    R = compute();
    const stageText = (i) => {
      const r = R; return [`${r.n.toLocaleString('en-US')} labelled rows, split by time`, ['raw columns, no imputation or scaling', 'impute + scale, fitted on train only, one shared pipeline', 'shared pipeline + engineered features'][r.pre], `fit ${MN[r.m]} (≈ $${r.train.toFixed(2)} compute)`, `AUC ${r.auc.toFixed(3)}, threshold ${r.thr.toFixed(2)} → precision ${r.prec.toFixed(2)}, recall ${r.tpr.toFixed(2)}`, r.fail ? 'FAILED: ' + r.fail : ['batch job scheduled nightly', `real-time API, ${r.repl} replicas`, 'ship to device over the air'][r.dep], ['no monitoring', 'basic alerts on inputs', 'drift + performance rules + auto retrain'][r.mon]][i];
    };
    AI.loop(root, (dt) => {
      if (playing && st < 6) { ph += dt * +q('#sp').value / 2.2; if (ph >= 1) { ph = 0; st++; lastKey = ''; } }
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h; g.clearRect(0, 0, W, H);
      const bw = (W - 20) / 6, by = 40, bh = 52;
      g.strokeStyle = C.border; g.lineWidth = 2; g.beginPath(); g.moveTo(10, by + bh / 2); g.lineTo(W - 10, by + bh / 2); g.stroke(); g.lineWidth = 1;
      STAGES.forEach((s, i) => {
        const x = 10 + i * bw, act = st === i, dn = st > i, bad = i === 4 && R.fail && st >= 4;
        g.fillStyle = bad ? AI.mix(C.bg3, C.red, 0.55) : act ? AI.mix(C.bg3, C.blue, 0.5) : dn ? AI.mix(C.bg3, C.green, 0.35) : C.bg3; rr(g, x + 3, by, bw - 8, bh, 8); g.fill();
        g.strokeStyle = bad ? C.red : act ? C.blue : C.border; rr(g, x + 3, by, bw - 8, bh, 8); g.stroke();
        text(g, (i + 1) + ' ' + s, x + bw / 2 - 1, by + 20, C.fg, '600 12px sans-serif', 'center');
        if (act) { g.fillStyle = C.blue; g.fillRect(x + 8, by + bh - 9, (bw - 18) * ph, 4); }
        if (dn) text(g, '✓', x + bw / 2, by + 40, bad ? C.red : C.green, '700 14px sans-serif', 'center');
      });
      const target = st >= 6 ? 5 + 1 : st + ph; pos += (target - pos) * (1 - Math.exp(-dt * 4));
      const px = 10 + clamp(pos, 0, 6) * bw - bw / 2; if (st < 6) { g.fillStyle = C.orange; g.beginPath(); g.arc(clamp(px, 14, W - 14), by - 10, 6, 0, 7); g.fill(); text(g, 'data / model', clamp(px, 40, W - 40), by - 20, C.orange, '10px sans-serif', 'center'); }
      const cap = st >= 6 ? 'Pipeline complete — change any choice to rerun it.' : `${STAGES[st]}: ${stageText(st)}`; text(g, cap, 10, 16, C.fg, '600 12px sans-serif');
      text(g, 'Illustrative values only', W - 10, 16, C.muted, '10px sans-serif', 'right');
      const key = [st, JSON.stringify(R)].join('|');
      if (key !== lastKey) {
        lastKey = key; const r = R, vis = [st >= 4, st >= 4, st >= 5, st >= 5, st >= 6];
        q('#cap').textContent = 'Scorecard fills in as the pipeline reaches each stage (accuracy after Evaluate, latency and cost after Deploy, risk after Monitor).';
        const card = (i, t, big, body, col) => `<div class="cp-card ${vis[i] ? '' : 'off'}"><h5>${t}</h5><div class="big" style="color:var(${col})">${vis[i] ? big : '…'}</div>${vis[i] ? body : ''}</div>`;
        q('#cards').innerHTML = card(0, 'Accuracy (AUC)', r.auc.toFixed(3), `F1 ${r.f1.toFixed(2)} · precision ${r.prec.toFixed(2)} · recall ${r.tpr.toFixed(2)}<br>FPR ${r.fpr.toFixed(3)}`, '--blue')
          + card(1, 'Business cost / 1k decisions', '$' + r.bizCost.toFixed(0), `missed return $${CFN}, false alarm $${CFP}, prevalence ${PI * 100}%`, '--orange')
          + card(2, 'Latency / run time', r.fail ? 'FAIL' : 'OK', r.fail ? r.fail : r.lat, r.fail ? '--red' : '--green')
          + card(3, 'Infra cost', '$' + (r.cost1k < 0.01 ? r.cost1k.toFixed(4) : r.cost1k.toFixed(3)) + '/1k', `$${r.infraDay.toFixed(2)}/day at 200k predictions/day<br>training ≈ $${r.train.toFixed(2)} per run`, '--purple')
          + card(4, 'Risk', r.level, `AUC after 6 months of drift ≈ ${r.auc6.toFixed(3)}<br>skew ${r.skew} · drift ${r.drift} · deploy ${r.dpl} · complexity ${r.cx}`, r.level === 'LOW' ? '--green' : r.level === 'MEDIUM' ? '--orange' : '--red');
        q('#out').textContent = `Model: AUC = ${CAP[r.m]} (ceiling) − data penalty − preprocessing effect → ${r.auc.toFixed(3)}; binormal separation d′ = √2·Φ⁻¹(AUC) = ${r.d.toFixed(3)}.\nThreshold ${r.thr.toFixed(2)} → TPR ${r.tpr.toFixed(3)}, FPR ${r.fpr.toFixed(3)}. Lower the threshold to catch more returns when a miss costs 10× a false alarm.\nAll prices, latencies and accuracies are round example numbers, not measurements.`;
      }
    }, { fps: 30 });
  };
})();
