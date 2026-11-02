/* Robotics · Module 4 (Perception for robots) labs. Synthetic data only; every simulation is illustrative.
 * perc-cloud     <div data-lab="perc-cloud"></div>     rotate a point cloud, voxel-downsample it
 * perc-icp       <div data-lab="perc-icp"></div>       2-D ICP: match, fit by SVD, move, repeat
 * perc-iou       <div data-lab="perc-iou"></div>       box overlap: BEV and 3-D IoU
 * perc-latency   <div data-lab="perc-latency"></div>   sensor rate versus detector latency
 * perc-pose      <div data-lab="perc-pose"></div>      6-DoF pose from keypoints (Levenberg–Marquardt)
 * perc-grasp     <div data-lab="perc-grasp"></div>     antipodal grasp check with a friction cone
 * perc-servo     <div data-lab="perc-servo"></div>     image-based visual servoing (illustrative)
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const DEG = Math.PI / 180;
  const fx = (v, d = 2) => (Math.abs(v) < 5e-10 ? 0 : v).toFixed(d);

  /* ---------- shared control helpers ---------- */
  const sliderRow = (ctl, name, min, max, step, value, fmt) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = fmt(+inp.value));
    upd();
    ctl.appendChild(l);
    return { el: inp, get: () => +inp.value, set: (x) => { inp.value = x; upd(); }, onInput: (fn) => (inp.oninput = () => { upd(); fn(); }) };
  };
  const speedRow = (ctl) => sliderRow(ctl, 'Speed', 0.25, 2, 0.05, 1, (x) => x.toFixed(2) + '×');
  const buttons = (ctl, labels) => {
    const d = document.createElement('div');
    d.className = 'row';
    d.innerHTML = labels.map((t) => `<button class="btn small">${t}</button>`).join('');
    ctl.appendChild(d);
    return [...d.querySelectorAll('button')];
  };
  const readoutEl = (ctl) => {
    const d = document.createElement('div');
    d.className = 'readout';
    d.style.fontFamily = MONO;
    d.style.fontSize = '12px';
    ctl.appendChild(d);
    return d;
  };
  const captionEl = (view) => AI.$('.caption', view);
  const dotAt = (ctx, x, y, r, color) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); };
  const lineAt = (ctx, x0, y0, x1, y1, color, w = 1) => { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
  /* small linear solver (Gaussian elimination with partial pivoting), A is n×n, b length n */
  const solve = (A, b) => {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const k = M[r][c] / M[c][c];
        for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
      }
    }
    return M.map((r, i) => r[n] / r[i]);
  };

  /* ---------- 1 · point cloud: rotate and voxel-downsample ---------- */
  AI.labs['perc-cloud'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Point cloud: rotate, then voxel-downsample',
      'a 0.40 × 0.30 × 0.20 m box surface, 900 points (illustrative scan)',
      '<canvas style="height:320px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const TILT = 0.35;
    const HALF = [0.2, 0.15, 0.1];
    const rng = AI.rng(11);
    const P = [];
    for (let i = 0; i < 900; i++) {
      const q = [0, 0, 0].map((_, k) => (rng() * 2 - 1) * HALF[k]);
      const ax = Math.floor(rng() * 3);
      q[ax] = (rng() < 0.5 ? -1 : 1) * HALF[ax];
      P.push(q);
    }
    const voxelize = (v) => {
      const m = new Map();
      for (const p of P) {
        const key = p.map((x) => Math.floor(x / v)).join(',');
        let g = m.get(key);
        if (!g) m.set(key, (g = { s: [0, 0, 0], n: 0 }));
        g.s[0] += p[0]; g.s[1] += p[1]; g.s[2] += p[2]; g.n++;
      }
      return [...m.values()].map((g) => g.s.map((s) => s / g.n));
    };
    const [btn] = buttons(ctl, ['⏸ Pause']);
    const speed = speedRow(ctl);
    const vox = sliderRow(ctl, 'Voxel size', 0, 0.1, 0.005, 0.05, (x) => (x === 0 ? 'off' : (x * 100).toFixed(1) + ' cm'));
    const out = readoutEl(ctl);
    let yaw = 0, playing = true, cache = null, cacheV = -1, lastKey = '';
    const proj = (p) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const X = p[0] * cy + p[1] * sy;
      const D = -p[0] * sy + p[1] * cy;
      const ct = Math.cos(TILT), st = Math.sin(TILT);
      return [X, p[2] * ct - D * st, D];
    };
    const sync = () => (btn.textContent = playing ? '⏸ Pause' : '▶ Play');
    btn.onclick = () => { playing = !playing; sync(); };
    sync();
    const refresh = () => {
      const v = vox.get();
      if (v !== cacheV) { cacheV = v; cache = v > 0 ? voxelize(v) : null; }
      out.innerHTML = `raw points: <b>${P.length}</b><br>` +
        (cache ? `voxel size: <b>${(cacheV * 100).toFixed(1)} cm</b> → centroids: <b>${cache.length}</b> (${fx(100 * (1 - cache.length / P.length), 0)} % fewer)` : 'voxel filter off: every raw point is kept');
      cap.textContent = cache
        ? 'Each occupied voxel is replaced by the mean of its points (orange). Points inside one voxel are merged, so the cloud shrinks while its shape stays.'
        : 'The raw cloud rotates about the vertical axis at 0.3 rad/s (slow on purpose). Drag the voxel size to merge points into centroids.';
    };
    vox.onInput(() => { lastKey = ''; refresh(); });
    const draw = () => {
      const C = AI.col();
      const ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const sc = (Math.min(c.w, c.h) * 0.42) / 0.25, ox = c.w / 2, oy = c.h / 2;
      const put = (q) => [ox + q[0] * sc, oy - q[1] * sc, q[2]];
      ctx.globalAlpha = cache ? 0.25 : 0.7;
      P.map(proj).forEach((q) => { const [x, y] = put(q); dotAt(ctx, x, y, 1.6, C.blue); });
      ctx.globalAlpha = 1;
      if (cache) cache.map(proj).forEach((q) => { const [x, y] = put(q); dotAt(ctx, x, y, 3.4, C.orange); });
      ctx.fillStyle = C.muted;
      ctx.font = '12px ' + MONO;
      ctx.fillText(`yaw ${fx((((yaw / DEG) % 360) + 360) % 360, 0)}°`, 10, c.h - 10);
    };
    c.onresize = () => { lastKey = ''; };
    refresh();
    AI.loop(root, (dt) => {
      if (playing) yaw += dt * speed.get() * 0.3;
      const key = yaw.toFixed(3) + '|' + cacheV + '|' + c.w;
      if (key !== lastKey) { lastKey = key; draw(); }
    });
  };

  /* ---------- 2 · 2-D ICP ---------- */
  AI.labs['perc-icp'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'ICP: iterative closest point (2-D, illustrative)',
      'orange = map (target), blue = scan moved by the current pose estimate; units are metres',
      '<canvas style="height:300px"></canvas><canvas style="height:90px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), cvs = view.querySelectorAll('canvas');
    const cv = AI.canvas(cvs[0]), plot = AI.canvas(cvs[1]);
    const N = 60;
    const target = [];
    for (let i = 0; i < N; i++) {
      const a = (2 * Math.PI * i) / N, r = 1 + 0.25 * Math.cos(3 * a) + 0.1 * Math.sin(2 * a);
      target.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    const rng = AI.rng(3);
    const scan = [];
    for (let i = 0; i < N; i++) {
      const a = (2 * Math.PI * (i + 0.5)) / N + 0.02, r = 1 + 0.25 * Math.cos(3 * a) + 0.1 * Math.sin(2 * a);
      scan.push([r * Math.cos(a) + 0.01 * (rng() - 0.5), r * Math.sin(a) + 0.01 * (rng() - 0.5)]);
    }
    const apply = (R, t, p) => [R[0] * p[0] + R[1] * p[1] + t[0], R[2] * p[0] + R[3] * p[1] + t[1]];
    const rotM = (a) => [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)];
    const mul = (A, B) => [A[0] * B[0] + A[1] * B[1], A[0] * B[2] + A[1] * B[3], A[2] * B[0] + A[3] * B[1], A[2] * B[2] + A[3] * B[3]];

    const btns = buttons(ctl, ['⏸ Pause', 'Reset']);
    const speed = speedRow(ctl);
    const th0 = sliderRow(ctl, 'Initial heading error', 0, 160, 1, 20, (x) => x.toFixed(0) + '°');
    const out = readoutEl(ctl);

    let Rt, tv, cur, from, to, stage, timer, iter, hist, pairs, playing, stageLen, done, lastKey, pendR, pendT;
    const reset = () => {
      const a0 = th0.get() * DEG;
      Rt = rotM(a0);
      tv = [0.2, -0.15];
      pendR = Rt; pendT = tv;
      cur = scan.map((p) => apply(Rt, tv, p));
      from = cur; to = cur; stage = 0; timer = 0; iter = 0; hist = []; pairs = null; done = false;
      playing = true; lastKey = ''; sync();
      hist.push(rms(cur));
    };
    const rms = (P) => {
      let s = 0;
      for (const p of P) { let b = Infinity; for (const q of target) b = Math.min(b, (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2); s += b; }
      return Math.sqrt(s / P.length);
    };
    const nearest = (P) => P.map((p) => { let b = 0, bd = Infinity; target.forEach((q, j) => { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; if (d < bd) { bd = d; b = j; } }); return b; });
    /* one ICP update: match, then closed-form 2-D rigid fit (Kabsch in 2-D) */
    const matchAndFit = () => {
      const idx = nearest(cur);
      const Q = idx.map((j) => target[j]);
      const mp = [0, 0], mq = [0, 0];
      cur.forEach((p, i) => { mp[0] += p[0] / N; mp[1] += p[1] / N; mq[0] += Q[i][0] / N; mq[1] += Q[i][1] / N; });
      let dot = 0, cr = 0;
      cur.forEach((p, i) => {
        const a = [p[0] - mp[0], p[1] - mp[1]], b = [Q[i][0] - mq[0], Q[i][1] - mq[1]];
        dot += a[0] * b[0] + a[1] * b[1];
        cr += a[0] * b[1] - a[1] * b[0];
      });
      const dth = Math.atan2(cr, dot), Rd = rotM(dth);
      const Rp = [Rd[0] * mp[0] + Rd[1] * mp[1], Rd[2] * mp[0] + Rd[3] * mp[1]];
      const dt2 = [mq[0] - Rp[0], mq[1] - Rp[1]];
      pairs = { Q, idx };
      return { Rd, dt2 };
    };
    const sync = () => (btns[0].textContent = playing ? '⏸ Pause' : '▶ Play');
    btns[0].onclick = () => { if (!done) playing = !playing; sync(); };
    btns[1].onclick = () => reset();
    th0.onInput(() => reset());
    reset();
    stageLen = [1.4, 1.6];

    const draw = (P) => {
      const C = AI.col(), ctx = cv.ctx;
      ctx.clearRect(0, 0, cv.w, cv.h);
      const sc = Math.min(cv.w, cv.h) / 3.4, ox = cv.w / 2, oy = cv.h / 2;
      const X = (p) => ox + p[0] * sc, Y = (p) => oy - p[1] * sc;
      target.forEach((q) => dotAt(ctx, X(q), Y(q), 2.6, C.orange));
      if (pairs && stage === 0 && !done) {
        P.forEach((p, i) => lineAt(ctx, X(p), Y(p), X(pairs.Q[i]), Y(pairs.Q[i]), C.muted, 0.6));
      }
      P.forEach((p) => dotAt(ctx, X(p), Y(p), 2.4, C.blue));
      // plot of RMS per iteration
      const pc = plot.ctx;
      pc.clearRect(0, 0, plot.w, plot.h);
      const mx = Math.max(0.05, ...hist), W = plot.w - 40, H = plot.h - 18;
      pc.fillStyle = C.muted; pc.font = '11px ' + MONO;
      pc.fillText('RMS distance to map (m) per iteration', 6, 12);
      pc.fillText(mx.toFixed(2), 2, 30);
      pc.fillText('0', 26, plot.h - 4);
      pc.strokeStyle = C.border; pc.beginPath(); pc.moveTo(34, 18); pc.lineTo(34, plot.h - 6); pc.lineTo(34 + W, plot.h - 6); pc.stroke();
      pc.strokeStyle = C.green; pc.lineWidth = 2; pc.beginPath();
      hist.forEach((v, k) => { const x = 34 + (W * k) / 30, y = plot.h - 6 - (H - 14) * (v / mx); k ? pc.lineTo(x, y) : pc.moveTo(x, y); });
      pc.stroke(); pc.lineWidth = 1;
    };

    AI.loop(root, (dt) => {
      if (playing && !done) {
        timer += dt * speed.get();
        if (stage === 0 && timer >= stageLen[0]) {
          timer = 0;
          const u = matchAndFit();
          to = cur.map((p) => apply(u.Rd, u.dt2, p));
          from = cur; stage = 1;
          // the pose composes as: new rotation = Rd·R, new translation = Rd·t + dt
          pendR = mul(u.Rd, Rt);
          pendT = [u.Rd[0] * tv[0] + u.Rd[1] * tv[1] + u.dt2[0], u.Rd[2] * tv[0] + u.Rd[3] * tv[1] + u.dt2[1]];
        } else if (stage === 1 && timer >= stageLen[1]) {
          timer = 0;
          cur = to; Rt = pendR; tv = pendT; stage = 0; iter++;
          hist.push(rms(cur));
          if (iter >= 30 || (hist.length > 2 && Math.abs(hist[hist.length - 1] - hist[hist.length - 2]) < 1e-7)) { done = true; playing = false; sync(); }
        }
      }
      let P;
      if (stage === 1 && !done) {
        const u = Math.min(1, timer / stageLen[1]), e = AI.smooth(u);
        P = from.map((p, i) => [AI.lerp(p[0], to[i][0], e), AI.lerp(p[1], to[i][1], e)]);
      } else P = cur;
      const rotDeg = Math.atan2(Rt[2], Rt[0]) / DEG;
      const key = (P.length ? P[0][0].toFixed(4) + P[0][1].toFixed(4) : '') + '|' + cv.w + '|' + done + stage + iter;
      if (key === lastKey && !playing) return;
      lastKey = key;
      draw(P);
      out.innerHTML = `iteration: <b>${iter}</b> / 30 · RMS: <b>${rms(stage === 1 ? to : cur).toFixed(4)} m</b><br>` +
        `heading estimate: <b>${fx(rotDeg, 1)}°</b> (truth 0°) · translation: <b>(${fx(tv[0], 3)}, ${fx(tv[1], 3)}) m</b>`;
      cap.textContent = done
        ? (rms(cur) < 0.02 ? 'Converged: the scan lies on the map. Try 20° or 30°: ICP can stop in a wrong local minimum (RMS about 0.06 m) even though the fit looks plausible.' : 'Stopped at a local minimum: the scan is close to the map but not on it. A poor initial guess caused this; the fit cannot leave it.')
        : (stage === 0 ? '① Match: each blue point takes its nearest orange point (grey lines). Several scan points may share one map point.' : '② Fit and move: the best rigid rotation and translation (closed form, SVD in 3-D) move the scan. Then repeat.');
    });
  };

  /* ---------- 3 · box overlap: BEV and 3-D IoU ---------- */
  AI.labs['perc-iou'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Box overlap: IoU in bird’s-eye view and in 3-D',
      'box A is fixed; box B moves in the plane (metres); the verdict uses the threshold you set',
      '<canvas style="height:280px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const A = { x0: 0, x1: 4, y0: 0, y1: 2, z0: 0, z1: 1.5 };
    const bx = sliderRow(ctl, 'Box B centre x', -1, 5, 0.05, 2, (x) => x.toFixed(2) + ' m');
    const by = sliderRow(ctl, 'Box B centre y', -1, 3.5, 0.05, 1, (x) => x.toFixed(2) + ' m');
    const bw = sliderRow(ctl, 'Box B length (x)', 1, 5, 0.1, 4, (x) => x.toFixed(1) + ' m');
    const bh = sliderRow(ctl, 'Box B width (y)', 1, 4, 0.1, 2, (x) => x.toFixed(1) + ' m');
    const bz = sliderRow(ctl, 'Box B height (z top)', 0.5, 3, 0.05, 2, (x) => x.toFixed(2) + ' m');
    const th = sliderRow(ctl, 'Match threshold', 0.1, 0.9, 0.05, 0.5, (x) => x.toFixed(2));
    const btns = buttons(ctl, ['▶ Drift B', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    let drift = false, phase = 0, lastKey = '';
    const boxes = () => {
      const cx = bx.get() + (drift ? 1.2 * Math.sin(phase) : 0);
      const B = { x0: cx - bw.get() / 2, x1: cx + bw.get() / 2, y0: by.get() - bh.get() / 2, y1: by.get() + bh.get() / 2, z0: 0.5, z1: bz.get() };
      return B;
    };
    const ov = (p, q) => Math.max(0, Math.min(p.x1, q.x1) - Math.max(p.x0, q.x0));
    const ovy = (p, q) => Math.max(0, Math.min(p.y1, q.y1) - Math.max(p.y0, q.y0));
    const ovz = (p, q) => Math.max(0, Math.min(p.z1, q.z1) - Math.max(p.z0, q.z0));
    const compute = (B) => {
      const aA = (A.x1 - A.x0) * (A.y1 - A.y0), aB = (B.x1 - B.x0) * (B.y1 - B.y0);
      const iB = ov(A, B) * ovy(A, B);
      const iou = iB / (aA + aB - iB);
      const vA = aA * (A.z1 - A.z0), vB = aB * (B.z1 - B.z0);
      const v3 = ov(A, B) * ovy(A, B) * ovz(A, B);
      return { aA, aB, iB, iou, vA, vB, v3, iou3: v3 / (vA + vB - v3) };
    };
    const sync = () => (btns[0].textContent = drift ? '⏸ Hold B' : '▶ Drift B');
    btns[0].onclick = () => { drift = !drift; sync(); };
    btns[1].onclick = () => { bx.set(2); by.set(1); bw.set(4); bh.set(2); bz.set(2); th.set(0.5); drift = false; sync(); };
    sync();
    const draw = (B, r) => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const xmin = -1.5, xmax = 6.5, ymin = -1.5, ymax = 3.5;
      const sc = Math.min(c.w / (xmax - xmin), c.h / (ymax - ymin));
      const ox = (c.w - sc * (xmax - xmin)) / 2, oy = c.h - (c.h - sc * (ymax - ymin)) / 2;
      const X = (x) => ox + (x - xmin) * sc, Y = (y) => oy - (y - ymin) * sc;
      const rect = (b, col, fill) => {
        ctx.fillStyle = fill; ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.fillRect(X(b.x0), Y(b.y1), X(b.x1) - X(b.x0), Y(b.y0) - Y(b.y1));
        ctx.strokeRect(X(b.x0), Y(b.y1), X(b.x1) - X(b.x0), Y(b.y0) - Y(b.y1));
      };
      rect(A, C.blue, 'rgba(47,129,247,0.18)');
      rect(B, C.orange, 'rgba(219,109,40,0.16)');
      const I = { x0: Math.max(A.x0, B.x0), x1: Math.min(A.x1, B.x1), y0: Math.max(A.y0, B.y0), y1: Math.min(A.y1, B.y1) };
      if (I.x1 > I.x0 && I.y1 > I.y0) rect(I, C.green, 'rgba(63,185,80,0.45)');
      ctx.fillStyle = C.muted; ctx.font = '11px ' + MONO;
      ctx.fillText('x (m) →', c.w - 70, c.h - 8);
      ctx.fillText('y (m)', 8, 14);
      ctx.fillStyle = C.blue; ctx.fillText('A', X(0.1), Y(1.9));
      ctx.fillStyle = C.orange; ctx.fillText('B', X(B.x1 - 0.25), Y(B.y1 - 0.15));
    };
    AI.loop(root, (dt) => {
      if (drift) phase += dt * speed.get() * 0.5;
      const B = boxes();
      const r = compute(B);
      const key = [B.x0, B.y0, B.x1, B.y1, B.z1, th.get()].map((v) => v.toFixed(3)).join('|') + c.w;
      if (key === lastKey) return;
      lastKey = key;
      draw(B, r);
      const pass = r.iou >= th.get();
      out.innerHTML = `BEV: intersection <b>${fx(r.iB, 2)} m²</b>, union <b>${fx(r.aA + r.aB - r.iB, 2)} m²</b>, IoU = <b>${fx(r.iou, 3)}</b><br>` +
        `3-D: intersection <b>${fx(r.v3, 2)} m³</b>, union <b>${fx(r.vA + r.vB - r.v3, 2)} m³</b>, IoU<sub>3D</sub> = <b>${fx(r.iou3, 3)}</b><br>` +
        `verdict at threshold ${th.get().toFixed(2)}: <b style="color:${pass ? AI.col().green : AI.col().red}">${pass ? 'match' : 'no match'}</b>`;
      cap.textContent = 'IoU = intersection ÷ union. Area is in m² in the top view; volume in m³ includes height, where B spans 0.5 m to its top. Drift B to watch the score fall as the boxes separate.';
    });
  };

  /* ---------- 4 · sensor rate versus detector latency ---------- */
  AI.labs['perc-latency'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Sensor rate versus detector latency (illustrative)',
      'camera frames arrive at a fixed rate; a detector takes a fixed time per frame; the timeline shows what happens',
      '<canvas style="height:260px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const fs = sliderRow(ctl, 'Sensor frame rate', 5, 30, 1, 10, (x) => x.toFixed(0) + ' Hz');
    const lat = sliderRow(ctl, 'Detector latency per frame', 20, 300, 5, 130, (x) => x.toFixed(0) + ' ms');
    const pol = document.createElement('label');
    pol.innerHTML = '<span>Frame policy</span><select><option value="latest">latest frame only</option><option value="queue">queue every frame</option></select>';
    ctl.appendChild(pol);
    const sel = pol.querySelector('select');
    const btn = buttons(ctl, ['⏸ Pause'])[0];
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const TMAX = 60, WIN = 6;
    let clock = 0, playing = true, ev = null, lastKey = '';
    const build = () => {
      const f = fs.get(), L = lat.get() / 1000, outs = [];
      if (sel.value === 'latest') {
        let free = 0, used = -1;
        while (free < TMAX) {
          let t = free, k = Math.floor(t * f + 1e-9);
          if (k <= used) { k = used + 1; t = k / f; }
          const end = t + L;
          outs.push({ cap: k / f, start: t, end });
          used = k; free = end;
        }
      } else {
        let free = 0;
        for (let k = 0; k / f < TMAX; k++) {
          const t = k / f, start = Math.max(t, free), end = start + L;
          outs.push({ cap: t, start, end });
          free = end;
        }
      }
      ev = { outs, f, L, policy: sel.value };
    };
    const stats = () => {
      const a = 20, b = 50;
      const inWin = ev.outs.filter((o) => o.end >= a && o.end < b);
      const rate = inWin.length / (b - a);
      const age = inWin.length ? inWin.reduce((s, o) => s + (o.end - o.cap), 0) / inWin.length : NaN;
      return { rate, age };
    };
    const sync = () => (btn.textContent = playing ? '⏸ Pause' : '▶ Play');
    btn.onclick = () => { playing = !playing; sync(); };
    sync();
    [fs, lat].forEach((s) => s.onInput(() => { build(); lastKey = ''; }));
    sel.onchange = () => { build(); lastKey = ''; };
    build();
    const draw = (now) => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const x0 = 110, x1 = c.w - 14, t0 = now - WIN, sx = (t) => x0 + ((t - t0) / WIN) * (x1 - x0);
      const rows = [['sensor', 40], ['detector', 110], ['output', 180]];
      ctx.font = '12px ' + MONO; ctx.fillStyle = C.muted;
      rows.forEach(([n, y]) => { ctx.fillText(n, 10, y + 4); lineAt(ctx, x0, y, x1, y, C.border, 1); });
      // sensor ticks
      const period = 1 / ev.f;
      for (let t = Math.ceil(t0 / period) * period; t <= now; t += period) {
        if (t < 0) continue;
        lineAt(ctx, sx(t), 34, sx(t), 46, C.blue, 2);
      }
      // detector bars
      ev.outs.forEach((o) => {
        if (o.end < t0 || o.start > now) return;
        const a = Math.max(o.start, t0), b = Math.min(o.end, now);
        ctx.fillStyle = o.end <= now ? 'rgba(47,129,247,0.55)' : 'rgba(47,129,247,0.25)';
        ctx.fillRect(sx(a), 102, Math.max(1, sx(b) - sx(a)), 16);
      });
      // outputs
      let last = null;
      ev.outs.forEach((o) => {
        if (o.end > now) return;
        last = o;
        if (o.end >= t0) dotAt(ctx, sx(o.end), 180, 4, C.orange);
      });
      if (last) {
        ctx.fillStyle = C.orange; ctx.fillText(`newest result age ${(1000 * (last.end - last.cap)).toFixed(0)} ms`, x0, 214);
      }
      ctx.fillStyle = C.muted;
      ctx.fillText(`window ${t0.toFixed(1)}–${now.toFixed(1)} s`, x0, 238);
    };
    AI.loop(root, (dt) => {
      if (playing) clock += dt * speed.get() * 0.8;
      if (clock > 40) clock -= 40;
      const key = clock.toFixed(2) + '|' + c.w + '|' + ev.f + ev.L + ev.policy;
      if (key === lastKey && !playing) return;
      lastKey = key;
      draw(clock);
      const s = stats();
      const period = 1000 / ev.f;
      out.innerHTML = `sensor period: <b>${period.toFixed(0)} ms</b> · detector time: <b>${(ev.L * 1000).toFixed(0)} ms</b><br>` +
        `results per second (steady): <b>${s.rate.toFixed(2)} Hz</b> · mean age of a result: <b>${(s.age * 1000).toFixed(0)} ms</b>`;
      cap.textContent = ev.policy === 'latest'
        ? 'Latest frame only: the detector takes the newest frame whenever it is free. Frames are skipped, so no queue grows, but each result is already old by its own latency.'
        : (ev.L > 1 / ev.f ? 'Queue every frame: the detector is slower than the sensor, so the backlog grows and results get older and older. The age in the readout keeps rising.' : 'Queue every frame: the detector keeps up, so each result is only its own latency old.');
    });
  };

  /* ---------- 5 · 6-DoF pose from keypoints (Levenberg–Marquardt) ---------- */
  AI.labs['perc-pose'] = function (root) {
    const { view, ctl } = AI.labShell(root, '6-DoF pose from keypoints (illustrative pinhole camera)',
      'orange = observed keypoints (true pose hidden); blue × = projection of the model pose; press Solve',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const F = 600, CX = 320, CY = 240, W = 640, H = 480;
    const KP = [[-0.05, -0.04, 0], [0.05, -0.04, 0], [0.05, 0.04, 0], [-0.05, 0.04, 0], [0, 0, 0.04], [0.03, 0, -0.03]];
    const Rm = (yaw, pitch, roll) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
      return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
      ];
    };
    const project = (p, X) => {
      const R = Rm(p[0], p[1], p[2]);
      const c0 = R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + p[3];
      const c1 = R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + p[4];
      const c2 = R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + p[5];
      return [F * c0 / c2 + CX, F * c1 / c2 + CY];
    };
    const TRUE = [30 * DEG, -10 * DEG, 5 * DEG, 0.1, -0.05, 2.0];
    const rng = AI.rng(5);
    const obs = KP.map((X) => {
      const q = project(TRUE, X);
      return [q[0] + 0.8 * (rng() * 2 - 1), q[1] + 0.8 * (rng() * 2 - 1)];
    });
    const resid = (p) => {
      const r = [];
      KP.forEach((X, i) => { const q = project(p, X); r.push(q[0] - obs[i][0], q[1] - obs[i][1]); });
      return r;
    };
    const cost = (r) => r.reduce((s, v) => s + v * v, 0);
    const rmsPx = (r) => Math.sqrt(cost(r) / KP.length);

    const sl = {
      yaw: sliderRow(ctl, 'Model yaw (rot. about z)', -180, 180, 0.5, 0, (x) => x.toFixed(1) + '°'),
      pitch: sliderRow(ctl, 'Model pitch (rot. about y)', -90, 90, 0.5, 0, (x) => x.toFixed(1) + '°'),
      roll: sliderRow(ctl, 'Model roll (rot. about x)', -90, 90, 0.5, 0, (x) => x.toFixed(1) + '°'),
      tx: sliderRow(ctl, 'Model t_x', -0.5, 0.5, 0.01, 0, (x) => x.toFixed(2) + ' m'),
      ty: sliderRow(ctl, 'Model t_y', -0.5, 0.5, 0.01, 0, (x) => x.toFixed(2) + ' m'),
      tz: sliderRow(ctl, 'Model t_z (depth)', 0.5, 4, 0.01, 2, (x) => x.toFixed(2) + ' m'),
    };
    const btns = buttons(ctl, ['Solve (LM)', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    /* the solver keeps exact numbers; sliders only display them (they snap to their step) */
    let P = [0, 0, 0, 0, 0, 2];
    const getP = () => P;
    const readSliders = () => [sl.yaw.get() * DEG, sl.pitch.get() * DEG, sl.roll.get() * DEG, sl.tx.get(), sl.ty.get(), sl.tz.get()];
    const setP = (p) => {
      P = p.slice();
      sl.yaw.set((p[0] / DEG).toFixed(1)); sl.pitch.set((p[1] / DEG).toFixed(1)); sl.roll.set((p[2] / DEG).toFixed(1));
      sl.tx.set(p[3].toFixed(3)); sl.ty.set(p[4].toFixed(3)); sl.tz.set(p[5].toFixed(3));
    };
    let solving = false, timer = 0, lam = 1e-2, lastKey = '', iters = 0;
    const jac = (p) => {
      const h = 1e-6, r0 = resid(p), J = [];
      for (let j = 0; j < 6; j++) {
        const pp = p.slice(), pm = p.slice();
        pp[j] += h; pm[j] -= h;
        const rp = resid(pp), rm = resid(pm);
        J.push(r0.map((_, i) => (rp[i] - rm[i]) / (2 * h)));
      }
      return { J, r0 };
    };
    const lmStep = () => {
      const p = getP();
      const { J, r0 } = jac(p);
      const A = [], g = [];
      for (let a = 0; a < 6; a++) {
        A.push([]);
        for (let b = 0; b < 6; b++) A[a].push(J[a].reduce((s, _, i) => s + J[a][i] * J[b][i], 0));
        g.push(-J[a].reduce((s, _, i) => s + J[a][i] * r0[i], 0));
      }
      // Levenberg–Marquardt damping on the diagonal
      const Ad = A.map((row, a) => row.map((v, b) => (a === b ? v + lam * (v + 1e-9) : v)));
      const dp = solve(Ad, g);
      const trial = p.map((v, i) => v + dp[i]);
      if (cost(resid(trial)) < cost(r0)) { setP(trial); lam = Math.max(lam / 3, 1e-7); return true; }
      lam = Math.min(lam * 4, 1e7);
      return false;
    };
    const sync = () => (btns[0].textContent = solving ? '⏸ Stop' : 'Solve (LM)');
    btns[0].onclick = () => { solving = !solving; iters = 0; timer = 0; sync(); };
    btns[1].onclick = () => { setP([0, 0, 0, 0, 0, 2]); lam = 1e-2; iters = 0; solving = false; sync(); lastKey = ''; };
    Object.values(sl).forEach((s) => s.onInput(() => { P = readSliders(); solving = false; sync(); lastKey = ''; }));
    sync();
    const draw = (p) => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const s = Math.min(c.w / W, c.h / H), ox = (c.w - W * s) / 2, oy = (c.h - H * s) / 2;
      ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, W * s, H * s);
      const pts = KP.map((X) => project(p, X));
      pts.forEach((q, i) => lineAt(ctx, ox + obs[i][0] * s, oy + obs[i][1] * s, ox + q[0] * s, oy + q[1] * s, C.muted, 1));
      obs.forEach((q) => dotAt(ctx, ox + q[0] * s, oy + q[1] * s, 5, C.orange));
      ctx.strokeStyle = C.blue; ctx.lineWidth = 2;
      pts.forEach((q) => {
        const x = ox + q[0] * s, y = oy + q[1] * s;
        ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
      });
      ctx.lineWidth = 1;
    };
    AI.loop(root, (dt) => {
      if (solving) {
        timer += dt * speed.get();
        if (timer > 0.8) {
          timer = 0;
          lmStep();
          iters++;
          if (iters >= 40 || rmsPx(resid(getP())) < 0.05) { solving = false; sync(); }
        }
      }
      const p = getP();
      const key = p.map((v) => v.toFixed(4)).join('|') + '|' + c.w + '|' + solving;
      if (key === lastKey && !solving) return;
      lastKey = key;
      draw(p);
      const r = resid(p);
      out.innerHTML = `reprojection RMS: <b>${rmsPx(r).toFixed(2)} px</b> over ${KP.length} keypoints · LM iterations: <b>${iters}</b><br>` +
        `model: yaw ${(p[0] / DEG).toFixed(1)}°, pitch ${(p[1] / DEG).toFixed(1)}°, roll ${(p[2] / DEG).toFixed(1)}° · t = (${p[3].toFixed(2)}, ${p[4].toFixed(2)}, ${p[5].toFixed(2)}) m`;
      cap.textContent = solving
        ? 'Each step linearises the reprojection error, solves a 6 × 6 system for the pose change, and keeps it only if the error falls. Watch the blue crosses slide onto the orange points.'
        : 'Six numbers (three rotation angles, three translations) define the pose. Each keypoint gives two pixel equations. Press Solve, or move the sliders and watch the error change.';
    });
  };

  /* ---------- 6 · antipodal grasp with friction cone ---------- */
  AI.labs['perc-grasp'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Antipodal grasp check (2-D cross-section, illustrative)',
      'a box of 80 × 40 mm; the two jaws close along the grasp line; green = both contacts inside their friction cones',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const HA = 0.04, HB = 0.02;
    const th = sliderRow(ctl, 'Grasp angle θ', 0, 180, 1, 20, (x) => x.toFixed(0) + '°');
    const ph = sliderRow(ctl, 'Object angle φ', 0, 90, 1, 20, (x) => x.toFixed(0) + '°');
    const mu = sliderRow(ctl, 'Friction coefficient μ', 0.1, 1.0, 0.05, 0.5, (x) => x.toFixed(2));
    const jm = sliderRow(ctl, 'Jaw maximum opening', 40, 120, 1, 90, (x) => x.toFixed(0) + ' mm');
    const [btn, resetBtn] = buttons(ctl, ['▶ Sweep θ', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    let sweep = false, lastKey = '', sweepDeg = 20;
    const contact = (theta, phi, sg) => {
      const dx = Math.cos(theta), dy = Math.sin(theta), cp = Math.cos(phi), sp = Math.sin(phi);
      const ux = sg * (cp * dx + sp * dy), uy = sg * (-sp * dx + cp * dy);
      const tx = Math.abs(ux) > 1e-12 ? HA / Math.abs(ux) : Infinity;
      const ty = Math.abs(uy) > 1e-12 ? HB / Math.abs(uy) : Infinity;
      let t, nl;
      if (tx < ty) { t = tx; nl = [Math.sign(ux), 0]; } else { t = ty; nl = [0, Math.sign(uy)]; }
      const n = [cp * nl[0] - sp * nl[1], sp * nl[0] + cp * nl[1]];
      return { t, n, p: [sg * t * dx, sg * t * dy] };
    };
    const evalGrasp = () => {
      const theta = th.get() * DEG, phi = ph.get() * DEG, alpha = Math.atan(mu.get());
      const dx = Math.cos(theta), dy = Math.sin(theta);
      const c1 = contact(theta, phi, 1), c2 = contact(theta, phi, -1);
      const width = (c1.t + c2.t) * 1000;
      const angA = Math.acos(Math.max(-1, Math.min(1, dx * c1.n[0] + dy * c1.n[1]))) / DEG;
      const angB = Math.acos(Math.max(-1, Math.min(1, -dx * c2.n[0] - dy * c2.n[1]))) / DEG;
      const okCone = angA <= alpha / DEG + 1e-9 && angB <= alpha / DEG + 1e-9;
      const okWidth = width <= jm.get();
      return { theta, phi, alpha, c1, c2, width, angA, angB, okCone, okWidth, ok: okCone && okWidth, dx, dy };
    };
    btn.onclick = () => { sweep = !sweep; sweepDeg = th.get(); btn.textContent = sweep ? '⏸ Stop sweep' : '▶ Sweep θ'; };
    resetBtn.onclick = () => { th.set(20); ph.set(20); mu.set(0.5); jm.set(90); sweep = false; sweepDeg = 20; btn.textContent = '▶ Sweep θ'; lastKey = ''; };
    const draw = (g) => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const sc = Math.min(c.w, c.h) / 0.13, ox = c.w / 2, oy = c.h / 2;
      const X = (x) => ox + x * sc, Y = (y) => oy - y * sc;
      // object polygon
      const cp = Math.cos(g.phi), sp = Math.sin(g.phi);
      const corners = [[HA, HB], [-HA, HB], [-HA, -HB], [HA, -HB]].map(([x, y]) => [cp * x - sp * y, sp * x + cp * y]);
      ctx.beginPath(); corners.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.closePath(); ctx.fillStyle = 'rgba(139,148,158,0.25)'; ctx.fill(); ctx.strokeStyle = C.muted; ctx.lineWidth = 2; ctx.stroke();
      const col = g.ok ? C.green : C.red;
      // grasp line
      lineAt(ctx, X(g.c2.p[0]), Y(g.c2.p[1]), X(g.c1.p[0]), Y(g.c1.p[1]), col, 2);
      // friction cones and normals at both contacts
      [[g.c1, 1], [g.c2, -1]].forEach(([cc]) => {
        const inward = [-cc.n[0], -cc.n[1]];
        const base = Math.atan2(inward[1], inward[0]);
        [-g.alpha, g.alpha].forEach((da) => {
          const a = base + da, L = 0.018;
          lineAt(ctx, X(cc.p[0]), Y(cc.p[1]), X(cc.p[0] + L * Math.cos(a)), Y(cc.p[1] + L * Math.sin(a)), C.yellow, 1.5);
        });
        AI.arrow(ctx, X(cc.p[0]), Y(cc.p[1]), X(cc.p[0] + 0.02 * cc.n[0]), Y(cc.p[1] + 0.02 * cc.n[1]), C.purple, 2);
        dotAt(ctx, X(cc.p[0]), Y(cc.p[1]), 4, col);
      });
      ctx.fillStyle = C.muted; ctx.font = '11px ' + MONO;
      ctx.fillText('purple = contact normal · yellow = friction cone edges (inward)', 8, c.h - 8);
    };
    AI.loop(root, (dt) => {
      if (sweep) {
        sweepDeg += dt * speed.get() * 15; // 15 deg/s at speed 1
        if (sweepDeg > 180) sweepDeg = 0;
        th.set(Math.round(sweepDeg));
      }
      const key = [th.get(), ph.get(), mu.get(), jm.get(), c.w].join('|');
      if (key === lastKey && !sweep) return;
      lastKey = key;
      const g = evalGrasp();
      draw(g);
      out.innerHTML = `jaw width: <b>${g.width.toFixed(1)} mm</b> (limit ${jm.get().toFixed(0)} mm)<br>` +
        `contact angles to normal: <b>${g.angA.toFixed(1)}°</b> and <b>${g.angB.toFixed(1)}°</b> · cone half-angle α = atan μ = <b>${(g.alpha / DEG).toFixed(1)}°</b><br>` +
        `verdict: <b style="color:${g.ok ? AI.col().green : AI.col().red}">${g.ok ? 'antipodal and inside both cones' : (!g.okWidth ? 'too wide for the jaws' : 'slips: a contact is outside its cone')}</b>`;
      cap.textContent = 'Grasp = centre, angle θ, and jaw width. Each contact force must lie inside its friction cone around the inward normal. Sweep θ to find the grasp angles that work.';
    });
  };

  /* ---------- 7 · image-based visual servoing (illustrative) ---------- */
  AI.labs['perc-servo'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Image-based visual servoing (illustrative)',
      'four image points must reach the orange square; the camera velocity comes from the image Jacobian and the error',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = captionEl(view), c = AI.canvas(AI.$('canvas', view));
    const ZTRUE = 0.5;
    const DES = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]];
    const lam = sliderRow(ctl, 'Gain λ', 0.1, 3, 0.05, 1, (x) => x.toFixed(2) + ' 1/s');
    const dz = sliderRow(ctl, 'Depth estimate error', -40, 40, 1, 0, (x) => (x >= 0 ? '+' : '') + x.toFixed(0) + ' % of 0.5 m');
    const roll0 = sliderRow(ctl, 'Initial roll error', -30, 30, 1, 15, (x) => x.toFixed(0) + '°');
    const [btn, resetBtn] = buttons(ctl, ['⏸ Pause', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    let F, trail, playing = true, sim = 0, lastKey = '', vel = [0, 0, 0];
    const reset = () => {
      const a = roll0.get() * DEG, ca = Math.cos(a), sa = Math.sin(a);
      F = DES.map(([x, y]) => [ca * x - sa * y + 0.15, sa * x + ca * y - 0.1]);
      trail = F.map((p) => [p]);
      sim = 0; playing = true; vel = [0, 0, 0]; lastKey = ''; sync();
    };
    const sync = () => (btn.textContent = playing ? '⏸ Pause' : '▶ Play');
    btn.onclick = () => { playing = !playing; sync(); };
    resetBtn.onclick = reset;
    [lam, dz, roll0].forEach((s) => s.onInput(() => reset()));
    reset();
    /* interaction matrix rows for one point: [ẋ; ẏ] = L v, v = (vx, vy, ωz) */
    const Lrows = (x, y, Z) => [[-1 / Z, 0, y], [0, -1 / Z, -x]];
    const solve3 = (A, b) => solve(A, b);
    const controlV = () => {
      const Zh = ZTRUE * (1 + dz.get() / 100);
      const L = [], e = [];
      F.forEach(([x, y], i) => {
        const rows = Lrows(x, y, Zh);
        L.push(rows[0], rows[1]);
        e.push(x - DES[i][0], y - DES[i][1]);
      });
      // v = -λ (LᵀL)⁻¹ Lᵀ e
      const LtL = [0, 1, 2].map((a) => [0, 1, 2].map((b) => L.reduce((s, r) => s + r[a] * r[b], 0)));
      const Lte = [0, 1, 2].map((a) => L.reduce((s, r, i) => s + r[a] * e[i], 0));
      const v = solve3(LtL, Lte);
      return v.map((x) => -lam.get() * x);
    };
    const errNorm = () => Math.sqrt(F.reduce((s, p, i) => s + (p[0] - DES[i][0]) ** 2 + (p[1] - DES[i][1]) ** 2, 0));
    const draw = () => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const L = 0.6, s = Math.min(c.w, c.h) / (2 * L), ox = c.w / 2, oy = c.h / 2;
      const X = (x) => ox + x * s, Y = (y) => oy - y * s;
      ctx.strokeStyle = C.border; ctx.strokeRect(X(-L), Y(L), 2 * L * s, 2 * L * s);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2; ctx.beginPath();
      DES.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.closePath(); ctx.stroke();
      trail.forEach((tr) => { ctx.strokeStyle = C.muted; ctx.globalAlpha = 0.5; ctx.beginPath(); tr.forEach(([x, y], k) => (k ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke(); });
      ctx.globalAlpha = 1;
      ctx.strokeStyle = C.blue; ctx.lineWidth = 2; ctx.beginPath();
      F.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.closePath(); ctx.stroke(); ctx.lineWidth = 1;
      F.forEach(([x, y]) => dotAt(ctx, X(x), Y(y), 4, C.blue));
      ctx.fillStyle = C.muted; ctx.font = '11px ' + MONO;
      ctx.fillText('image plane (normalised units)', 8, c.h - 8);
    };
    AI.loop(root, (dt) => {
      if (playing && errNorm() > 2e-3) {
        const h = 0.05;
        let budget = dt * speed.get();
        while (budget > 1e-9) {
          const step = Math.min(h, budget); budget -= step;
          const v = controlV();
          vel = v;
          // true feature motion uses the true depth
          F = F.map(([x, y]) => [x + step * (-v[0] / ZTRUE + y * v[2]), y + step * (-v[1] / ZTRUE - x * v[2])]);
          sim += step;
        }
        F.forEach((p, i) => { trail[i].push(p); if (trail[i].length > 120) trail[i].shift(); });
      } else if (playing) { playing = false; sync(); }
      const en = errNorm();
      const key = (sim * 100).toFixed(0) + '|' + c.w + '|' + playing;
      if (key === lastKey && !playing) return;
      lastKey = key;
      draw();
      const v = vel;
      out.innerHTML = `image error ‖e‖: <b>${en.toFixed(4)}</b> (normalised) · simulated time: <b>${sim.toFixed(1)} s</b><br>` +
        `camera velocity: v<sub>x</sub> = <b>${(v[0] || 0).toFixed(3)} m/s</b>, v<sub>y</sub> = <b>${(v[1] || 0).toFixed(3)} m/s</b>, ω<sub>z</sub> = <b>${(v[2] || 0).toFixed(3)} rad/s</b>`;
      cap.textContent = en < 2e-3
        ? 'Converged: the blue square sits on the orange target. The controller used an estimated depth; try a depth error of ±40 % to see how much it matters.'
        : 'Control loop: ① measure the four image points ② error e = current − target ③ v = −λ L⁺ e (L is the image Jacobian at the estimated depth) ④ move the camera and repeat.';
    });
  };
})();
