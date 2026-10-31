/* GeoAI Module 9 (geo-production.html) labs: splitter, lazygraph, publish, drift, privacy, appsel.
 * All costs, throughputs, labour rates and scores are illustrative assumptions; the lesson text states them.
 * Animations use dt (AI.loop), run in compressed or slow time, have speed/pause/step/restart and redraw only on change. */
(function () {
  const sl = (id, label, min, max, step, v) => `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${v}"></label>`;
  const transport = (extra) => `<div class="row"><button class="btn small primary" id="play">Pause</button><button class="btn small" id="step">Step</button><button class="btn small" id="reset">Restart</button>${extra || ''}</div>${sl('sp', 'Speed (×)', 0.25, 3, 0.05, 1)}`;
  const dp = (step) => { const s = String(step); return s.indexOf('.') >= 0 ? s.length - s.indexOf('.') - 1 : 0; };
  const wire = (ctl) => {
    const upd = () => ctl.querySelectorAll('input[type=range]').forEach((i) => { const o = ctl.querySelector('#' + i.id + 'v'); if (o) o.textContent = (+i.value).toFixed(dp(i.step)); });
    ctl.querySelectorAll('input[type=range]').forEach((i) => i.addEventListener('input', upd));
    upd();
  };
  const val = (ctl, id) => +ctl.querySelector('#' + id).value;
  const speed = (ctl) => val(ctl, 'sp');
  const txt = (g, s, x, y, col, font, align) => { g.fillStyle = col; g.font = font || '11px sans-serif'; g.textAlign = align || 'left'; g.fillText(s, x, y); g.textAlign = 'left'; };
  const rr = (g, x, y, w, h, r) => { r = Math.max(0, Math.min(r, w / 2, h / 2)); g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); };
  const fmtTime = (s) => String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(Math.floor(s % 60)).padStart(2, '0');
  // play / step / restart wiring shared by the labs
  const bind = (ctl, st, onRestart) => {
    const b = (id) => ctl.querySelector('#' + id);
    b('play').onclick = () => { st.play = !st.play; b('play').textContent = st.play ? 'Pause' : 'Play'; };
    b('step').onclick = () => { st.step = true; st.play = false; b('play').textContent = 'Play'; };
    b('reset').onclick = () => onRestart();
  };

  /* ================================================================
   * 1. National job splitter
   * ================================================================ */
  AI.labs.splitter = function (root) {
    const { view, ctl } = AI.labShell(root, 'National job splitter: tiles, workers and cost',
      'illustrative · time is compressed: at speed 1× one real second is one simulated minute · grey cells lie outside the border and are skipped',
      '<canvas style="height:340px"></canvas>', '');
    ctl.innerHTML = sl('px', 'Tile size (px)', 128, 512, 128, 256) + sl('ov', 'Overlap (px)', 0, 64, 2, 26)
      + sl('wk', 'Workers', 1, 40, 1, 20) + sl('rate', 'Tiles per second per worker', 0.25, 4, 0.25, 1)
      + sl('price', 'Price per worker-hour ($)', 0.1, 3, 0.05, 0.9) + sl('area', 'Country area (km²)', 50000, 600000, 10000, 300000)
      + transport() + '<div class="readout" id="ro"></div>';
    wire(ctl);
    const ro = ctl.querySelector('#ro');
    const c = AI.canvas(AI.$('canvas', view));
    const GW = 24, GH = 16;
    const inside = (u, v) => { const x = (u - 0.5) / 0.46, y = (v - 0.5) / 0.4; return x * x + y * y < 1 + 0.25 * Math.sin(7 * u + 1) * Math.cos(5 * v); };
    const grid = [];
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) grid.push({ i, j, out: !inside((i + 0.5) / GW, (j + 0.5) / GH), s: 'todo', p: 0 });
    const cells = grid.filter((q) => !q.out);
    const nc = cells.length;
    const st = { play: true, step: false };
    let simT = 0, next = 0, lanes = [], lastPk = '', lastDraw = '';
    const restart = () => {
      simT = 0; next = 0;
      cells.forEach((q) => { q.s = 'todo'; q.p = 0; });
      lanes = Array.from({ length: val(ctl, 'wk') }, () => ({ cell: -1, rem: 0 }));
    };
    const model = () => {
      const px = val(ctl, 'px'), ov = val(ctl, 'ov'), wk = val(ctl, 'wk'), rate = val(ctl, 'rate'), price = val(ctl, 'price'), area = val(ctl, 'area');
      const N = Math.ceil(area / (((px - ov) * 0.01) ** 2));   // a tile owns (P-O) pixels of 10 m
      const W = N / nc;                                         // real tiles represented by one drawn cell
      const cellSec = W / rate;                                 // seconds for one worker to finish one cell
      const finish = Math.ceil(nc / wk) * cellSec;              // equal cells, greedy queue: rounds × cell time
      return { N, W, wk, price, area, cellSec, finish, ideal: N / rate };
    };
    const advance = (dt, m) => {
      lanes.forEach((L, l) => {
        let left = dt;
        while (left > 1e-9) {
          if (L.cell < 0) {
            if (next >= cells.length) break;
            L.cell = next++; L.rem = m.cellSec;
            const q0 = cells[L.cell]; q0.s = 'run'; q0.p = 0;
          }
          const use = Math.min(left, L.rem);
          L.rem -= use; left -= use;
          const q1 = cells[L.cell]; q1.p = 1 - L.rem / m.cellSec;
          if (L.rem <= 1e-9) { q1.s = 'done'; q1.p = 1; L.cell = -1; }
        }
      });
    };
    AI.loop(root, (dt) => {
      const pk = ['px', 'ov', 'wk', 'rate', 'price', 'area'].map((id) => val(ctl, id)).join('|');
      if (pk !== lastPk) { lastPk = pk; restart(); }
      const m = model();
      let moved = false;
      if (st.step) {
        st.step = false;
        const inc = Math.min(m.cellSec / 4, Math.max(0, m.finish - simT));
        if (inc > 0) { simT += inc; advance(inc, m); moved = true; }
      } else if (st.play && simT < m.finish) {
        const inc = Math.min(dt * 60 * speed(ctl), m.finish - simT);
        simT += inc; advance(inc, m); moved = true;
      }
      const dkey = Math.round(simT * 2) + '|' + pk + '|' + c.w + 'x' + c.h + '|' + document.documentElement.dataset.theme;
      if (!moved && dkey === lastDraw) return;
      lastDraw = dkey;

      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const top = 12, mapW = W * 0.56, mapH = H - 80;
      const cs = Math.max(4, Math.min((mapW - 12) / GW, mapH / GH));
      grid.forEach((q) => {
        const x = 10 + q.i * cs, y = top + q.j * cs, s = cs - 1;
        if (q.out) { g.fillStyle = C.bg2; g.fillRect(x, y, s, s); return; }
        g.fillStyle = C.bg3; g.fillRect(x, y, s, s);
        if (q.s === 'done') { g.fillStyle = C.green; g.fillRect(x, y, s, s); }
        else if (q.s === 'run') { g.fillStyle = C.orange; g.fillRect(x, y + s * (1 - q.p), s, s * q.p); }
      });
      txt(g, 'country tiled into cells', 10, top + GH * cs + 14, C.muted, '10px sans-serif');
      const lx = W * 0.6, lw = W - lx - 12, lh = Math.max(9, Math.min(24, (mapH - 16) / Math.max(1, lanes.length)));
      txt(g, 'workers pull from one queue', lx, top + 10, C.muted, '10px sans-serif');
      lanes.forEach((L, l) => {
        const y = top + 18 + l * lh;
        const bx = lx + 40, bw = lw - 40;
        if (lh >= 12) txt(g, 'w' + (l + 1), lx, y + lh * 0.7, C.muted, '10px monospace');
        g.fillStyle = C.bg3; rr(g, bx, y + 2, bw, lh - 4, 3); g.fill();
        if (L.cell >= 0) {
          g.fillStyle = C.orange; rr(g, bx, y + 2, Math.max(2, bw * cells[L.cell].p), lh - 4, 3); g.fill();
        } else if (simT >= m.finish - 1e-6) {
          txt(g, 'done', bx + 4, y + lh * 0.7, C.green, '10px monospace');
        } else {
          txt(g, 'idle (billed)', bx + 4, y + lh * 0.7, C.muted, '10px monospace');
        }
      });
      const idle = next >= cells.length && simT < m.finish - 1e-6;
      const phase = simT <= 0 ? '① Cut the country into tiles. Grey cells are outside the border and cost nothing.'
        : simT < m.finish - 1e-6 ? '② Each worker takes the next cell, reads only its own window of the COG, runs the model and writes its output.'
        : '③ All cells are done. Merge the tiles into one mosaic and publish. The bill stops when the last worker stops.';
      txt(g, phase, 10, H - 40, C.fg, '12px sans-serif');
      txt(g, idle ? '④ Some workers now wait for the last round: idle, but still billed (no autoscaling).' : '', 10, H - 20, C.orange, '11px sans-serif');

      const done = cells.filter((q) => q.s === 'done').length;
      const part = cells.reduce((a, q) => a + (q.s === 'run' ? q.p : 0), 0);
      const doneTiles = Math.min(m.N, (done + part) * m.W);
      const costNow = m.wk * Math.min(simT, m.finish) / 3600 * m.price;
      const costTotal = m.wk * m.finish / 3600 * m.price;
      const costIdeal = m.ideal / 3600 * m.price;
      ro.innerHTML = [
        `tiles        ${m.N.toLocaleString('en-US')}`,
        `cells        ${nc} drawn, ${m.W.toFixed(0)} tiles each`,
        `tiles done   ${Math.round(doneTiles).toLocaleString('en-US')}`,
        `time         ${fmtTime(simT)} of ${fmtTime(m.finish)}`,
        `cost now     $${costNow.toFixed(2)}`,
        `cost actual  $${costTotal.toFixed(2)}  (idle lanes billed)`,
        `cost ideal   $${costIdeal.toFixed(2)}  (no idle lanes)`,
        `per km²      $${(costTotal / m.area).toFixed(6)}`,
      ].join('\n');
    }, { fps: 30 });
    bind(ctl, st, restart);
    ctl.querySelector('#play').textContent = 'Pause';
    restart();
  };

  /* ================================================================
   * 2. Lazy graph: read -> mask -> NDVI -> reduce, chunk by chunk
   * ================================================================ */
  AI.labs.lazygraph = function (root) {
    const { view, ctl } = AI.labShell(root, 'Lazy graph: read → mask → NDVI → reduce, chunk by chunk',
      'illustrative 2×6 raster in three chunks (columns 1–2, 3–4, 5–6) · the graph is only a recipe until you ask for the result',
      '<canvas style="height:330px"></canvas>', '');
    ctl.innerHTML = sl('thr', 'Cloud mask: masked where red is above', 0.2, 0.35, 0.01, 0.27) + transport()
      + '<div class="row"><button class="btn small" id="lazy">Lazy</button><button class="btn small" id="eager">Eager</button><button class="btn small primary" id="go">Compute result</button></div>'
      + '<div class="readout" id="ro"></div>';
    wire(ctl);
    const ro = ctl.querySelector('#ro');
    const c = AI.canvas(AI.$('canvas', view));
    const RED = [[0.10, 0.12, 0.30, 0.08, 0.09, 0.25], [0.11, 0.10, 0.28, 0.07, 0.10, 0.22]];
    const NIR = [[0.50, 0.46, 0.32, 0.55, 0.48, 0.40], [0.52, 0.49, 0.35, 0.58, 0.47, 0.44]];
    const CH = [[0, 1], [2, 3], [4, 5]];
    const STAGES = ['read', 'mask', 'ndvi', 'partial'];
    const NAME = { read: 'read', mask: 'mask', ndvi: 'NDVI', partial: 'reduce' };
    const PREV = { mask: 'read', ndvi: 'mask', partial: 'ndvi' };
    const st = { play: false, step: false, mode: 'lazy' };
    let O = {}, tasks = [], ti = 0, tt = 0, held = new Set(), peak = 0, lastPk = '', lastDraw = '';
    const thr = () => val(ctl, 'thr');
    const pix = (k) => { const [a, b] = CH[k]; return [[0, a], [0, b], [1, a], [1, b]]; };
    const plan = () => {
      const t = [];
      if (st.mode === 'lazy') { for (let k = 0; k < 3; k++) STAGES.forEach((s) => t.push([s, k])); }
      else STAGES.forEach((s) => { for (let k = 0; k < 3; k++) t.push([s, k]); });
      t.push(['combine', -1]);
      return t;
    };
    const restart = () => {
      O = { read: {}, mask: {}, ndvi: {}, partial: {} };
      tasks = plan(); ti = 0; tt = 0; held = new Set(); peak = 0;
      st.play = false; ctl.querySelector('#play').textContent = 'Play';
    };
    const compute = (stage, k) => {
      if (stage === 'read') return pix(k).map(([i, j]) => ({ i, j, r: RED[i][j], n: NIR[i][j] }));
      if (stage === 'mask') return O.read[k].map((p) => ({ ...p, bad: p.r > thr() }));
      if (stage === 'ndvi') return O.mask[k].map((p) => ({ ...p, v: p.bad ? NaN : (p.n - p.r) / (p.n + p.r) }));
      let s = 0, n = 0;
      O.ndvi[k].forEach((p) => { if (!Number.isNaN(p.v)) { s += p.v; n++; } });
      return { s, n };
    };
    const finishTask = () => {
      const [stage, k] = tasks[ti];
      if (stage === 'combine') held.clear();
      else {
        O[stage][k] = compute(stage, k);
        held.add(stage + k);
        if (st.mode === 'lazy' && PREV[stage]) held.delete(PREV[stage] + k);   // lazy frees a chunk once the next step has used it
        peak = Math.max(peak, held.size);
      }
      ti++; tt = 0;
    };
    const status = (stage, k) => {
      if (ti < tasks.length && tasks[ti][0] === stage && tasks[ti][1] === k) return 'run';
      if (O[stage] && O[stage][k] !== undefined) return held.has(stage + k) ? 'held' : 'freed';
      return 'todo';
    };
    const summary = (stage, k) => {
      const v = O[stage][k];
      if (v === undefined) return '';
      if (stage === 'read') return '4 px read';
      if (stage === 'mask') return 'bad ' + v.filter((p) => p.bad).length;
      if (stage === 'ndvi') { const ok = v.filter((p) => !Number.isNaN(p.v)); return 'valid ' + ok.length + '/4'; }
      return 'S ' + v.s.toFixed(4) + '  n ' + v.n;
    };
    AI.loop(root, (dt) => {
      const pk = thr() + '|' + st.mode;
      if (pk !== lastPk) { lastPk = pk; restart(); }
      if (st.step) { st.step = false; if (ti < tasks.length) finishTask(); }
      else if (st.play && ti < tasks.length) { tt += dt * speed(ctl) / 1.1; if (tt >= 1) finishTask(); }
      if (ti >= tasks.length && st.play) { st.play = false; ctl.querySelector('#play').textContent = 'Play'; }
      const dkey = ti + '|' + tt.toFixed(2) + '|' + pk + '|' + held.size + '|' + c.w + 'x' + c.h + '|' + document.documentElement.dataset.theme;
      if (dkey === lastDraw && !st.play) return;
      lastDraw = dkey;

      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const colW = (W - 190) / 4, top = 26;
      STAGES.forEach((stage, s) => {
        const x = 12 + s * colW;
        txt(g, NAME[stage], x + (colW - 24) / 2, 16, C.fg, '600 12px sans-serif', 'center');
        if (s > 0) { g.strokeStyle = C.muted; g.beginPath(); g.moveTo(x - 6, top + 60); g.lineTo(x + 2, top + 60); g.stroke(); }
        for (let k = 0; k < 3; k++) {
          const y = top + 14 + k * 74, w = colW - 26, h = 62, stt = status(stage, k);
          const fill = stt === 'run' ? C.orange : stt === 'held' ? AI.mix(C.bg3, C.blue, 0.55) : stt === 'freed' ? C.bg2 : C.bg3;
          g.fillStyle = fill; rr(g, x, y, w, h, 6); g.fill();
          g.strokeStyle = stt === 'run' ? C.orange : C.border; g.lineWidth = stt === 'freed' ? 1 : 1.3;
          if (stt === 'freed') g.setLineDash([4, 3]);
          rr(g, x, y, w, h, 6); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
          txt(g, 'chunk ' + 'ABC'[k], x + 6, y + 14, stt === 'freed' ? C.muted : C.fg, '10px monospace');
          txt(g, stt === 'todo' ? 'not started' : stt === 'run' ? 'running…' : stt === 'freed' ? 'freed' : summary(stage, k), x + 6, y + 36, stt === 'run' ? C.bg : C.muted, '10px monospace');
          if (stt === 'held' || stt === 'freed') txt(g, stt, x + 6, y + 52, stt === 'held' ? C.blue : C.muted, '10px sans-serif');
        }
      });
      const rx = 12 + 4 * colW, rw = W - rx - 12;
      g.fillStyle = C.bg2; rr(g, rx, top + 14, rw, 3 * 74 - 12, 8); g.fill();
      txt(g, 'result', rx + 10, top + 34, C.fg, '600 12px sans-serif');
      let ans = 'not computed', wrong = '';
      if (ti >= tasks.length) {
        let S = 0, N = 0, mm = 0;
        for (let k = 0; k < 3; k++) { S += O.partial[k].s; N += O.partial[k].n; mm += O.partial[k].s / O.partial[k].n; }
        ans = (S / N).toFixed(5);
        wrong = 'mean of chunk means ' + (mm / 3).toFixed(5) + ' (wrong)';
      }
      txt(g, 'mean NDVI', rx + 10, top + 58, C.muted, '10px sans-serif');
      txt(g, ans, rx + 10, top + 80, ti >= tasks.length ? C.green : C.muted, '600 16px monospace');
      if (wrong) txt(g, wrong, rx + 10, top + 100, C.red, '10px monospace');
      txt(g, st.mode === 'lazy' ? 'lazy: each chunk runs read → mask → NDVI → reduce before the next chunk starts.' : 'eager: every chunk finishes one step before any chunk starts the next, so all of them sit in memory.', 10, H - 22, C.fg, '11px sans-serif');
      txt(g, ti === 0 ? 'Nothing has run. Press Compute result: only then does the graph execute.' : '', 10, H - 6, C.muted, '11px sans-serif');

      ro.innerHTML = [
        `mode                 ${st.mode}`,
        `tasks run            ${Math.min(ti, tasks.length)} of ${tasks.length}`,
        `chunk outputs held   ${held.size}   (peak ${peak})`,
        `answer               ${ans}`,
      ].join('\n');
    }, { fps: 30 });
    bind(ctl, st, () => restart());
    ctl.querySelector('#play').textContent = 'Play';
    ctl.querySelector('#go').onclick = () => { if (ti >= tasks.length) restart(); st.play = true; ctl.querySelector('#play').textContent = 'Pause'; };
    ctl.querySelector('#lazy').onclick = () => { st.mode = 'lazy'; restart(); };
    ctl.querySelector('#eager').onclick = () => { st.mode = 'eager'; restart(); };
    restart();
  };

  /* ================================================================
   * 3. Publishing pipeline: COG, checksums, STAC Item, validation, collection, release
   * ================================================================ */
  AI.labs.publish = function (root) {
    const { view, ctl } = AI.labShell(root, 'Publishing a product: COG, STAC items and versions',
      'illustrative · each stage waits for its checks; switch one off to see the pipeline block · one stage takes about 1.6 s at speed 1×',
      '<canvas style="height:300px"></canvas>', '');
    const CHECKS = [['tile', 'Internal tiling (512 px blocks)'], ['ov', 'Overviews ×2 to ×32'], ['crs', 'CRS and geotransform stored'],
      ['nd', 'No-data value declared (255)'], ['sum', 'SHA-256 checksums listed'], ['val', 'STAC Item validates'],
      ['lic', 'Licence and DOI on collection'], ['ver', 'Version bump matches change log']];
    const STAGE_DEF = [
      ['Write COG', ['tile', 'ov', 'crs', 'nd']],
      ['Checksums', ['sum']],
      ['STAC Item', ['crs']],
      ['Validate', ['val']],
      ['Collection', ['lic']],
      ['Release', ['ver']],
    ];
    ctl.innerHTML = '<div class="row" style="flex-wrap:wrap">' + CHECKS.map(([id, label]) => `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="c_${id}" checked> ${label}</label>`).join('') + '</div>'
      + '<label>Version bump <select id="bump"><option value="patch">patch: fix some tiles</option><option value="minor">minor: new year or better model, same classes</option><option value="major">major: new classes or schema</option></select></label>'
      + transport('<button class="btn small primary" id="pub">Publish</button>')
      + '<div class="readout" id="ro"></div>';
    wire(ctl);
    const ro = ctl.querySelector('#ro');
    const c = AI.canvas(AI.$('canvas', view));
    const on = (id) => ctl.querySelector('#c_' + id).checked;
    const st = { play: false, step: false, stage: -1, t: 0, blocked: -1 };
    let vers = [], lastDraw = '';
    const initVers = () => [[1, 0, 0], [1, 1, 0], [1, 2, 0]];
    const vs = (v) => 'v' + v.join('.');
    const bumped = () => {
      const [M, m, p] = vers[vers.length - 1];
      const t = ctl.querySelector('#bump').value;
      return t === 'major' ? [M + 1, 0, 0] : t === 'minor' ? [M, m + 1, 0] : [M, m, p + 1];
    };
    const restart = () => { vers = initVers(); st.stage = -1; st.t = 0; st.blocked = -1; st.play = false; st.step = false; ctl.querySelector('#play').textContent = 'Play'; };
    ctl.querySelector('#pub').onclick = () => {
      restart(); st.stage = 0; st.play = true; ctl.querySelector('#play').textContent = 'Pause';
    };
    AI.loop(root, (dt) => {
      const go = st.play || st.step;
      if (go && st.stage >= 0 && st.stage < STAGE_DEF.length && st.blocked < 0) {
        st.t += st.step ? 1 : dt * speed(ctl) / 1.6;
        st.step = false;
        if (st.t >= 1) {
          st.t = 0;
          const missing = STAGE_DEF[st.stage][1].filter((id) => !on(id));
          if (missing.length) { st.blocked = st.stage; st.play = false; ctl.querySelector('#play').textContent = 'Play'; }
          else st.stage++;
          if (st.stage === STAGE_DEF.length && st.blocked < 0) { vers.push(bumped()); st.play = false; ctl.querySelector('#play').textContent = 'Play'; }
        }
      }
      const checkKey = CHECKS.map(([id]) => on(id) ? 1 : 0).join('');
      const dkey = st.stage + '|' + st.blocked + '|' + Math.round(st.t * 50) + '|' + checkKey + '|' + vers.length + '|' + ctl.querySelector('#bump').value + '|' + c.w + 'x' + c.h + '|' + document.documentElement.dataset.theme;
      if (dkey === lastDraw) return;
      lastDraw = dkey;

      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const n = STAGE_DEF.length, gap = 8, bw = (W - 20 - gap * (n - 1)) / n, by = 36, bh = 64;
      const xs = STAGE_DEF.map((_, i) => 10 + i * (bw + gap));
      STAGE_DEF.forEach(([name, req], i) => {
        const x = xs[i];
        let col = C.bg3;
        if (st.blocked === i) col = C.red; else if (st.stage > i) col = C.green; else if (st.stage === i) col = C.orange;
        g.fillStyle = AI.mix(C.bg2, col === C.bg3 ? C.bg2 : col, col === C.bg3 ? 1 : 0.35); rr(g, x, by, bw, bh, 8); g.fill();
        g.strokeStyle = col; g.lineWidth = 1.5; rr(g, x, by, bw, bh, 8); g.stroke(); g.lineWidth = 1;
        txt(g, name, x + bw / 2, by + 22, C.fg, '600 11px sans-serif', 'center');
        const miss = req.filter((id) => !on(id));
        txt(g, miss.length ? 'missing: ' + miss.length : 'checks ok', x + bw / 2, by + 42, miss.length ? C.red : C.muted, '10px sans-serif', 'center');
        txt(g, 'stage ' + (i + 1), x + bw / 2, by + 56, C.muted, '9px sans-serif', 'center');
      });
      const pos = st.stage < 0 ? -1 : st.stage >= n ? n - 1 : st.stage;
      if (pos >= 0) {
        const cx = st.stage >= n ? xs[n - 1] + bw : st.blocked >= 0 ? xs[pos] + bw / 2 : xs[pos] + bw / 2 + (st.stage < n ? (st.t) * (bw + gap) : 0);
        const cy = by + bh + 16;
        g.fillStyle = st.blocked >= 0 ? C.red : C.blue; g.beginPath(); g.arc(Math.min(cx, W - 10), cy, 6, 0, 7); g.fill();
      }
      const vy = by + bh + 44;
      txt(g, 'versions in the collection', 10, vy, C.muted, '10px sans-serif');
      vers.forEach((v, i) => {
        const x = 10 + i * 88, fresh = i === vers.length - 1 && st.stage >= n;
        g.fillStyle = fresh ? C.green : C.bg3; rr(g, x, vy + 8, 80, 22, 6); g.fill();
        txt(g, vs(v), x + 40, vy + 23, fresh ? C.bg : C.fg, '600 11px monospace', 'center');
      });
      const cap = st.blocked >= 0 ? 'Blocked at "' + STAGE_DEF[st.blocked][0] + '": the stage needs ' + STAGE_DEF[st.blocked][1].filter((id) => !on(id)).map((id) => CHECKS.find((x) => x[0] === id)[1]).join('; ') + '.'
        : st.stage < 0 ? 'Press Publish. A stage cannot finish until its checks pass; the version only moves when every stage passes.'
        : st.stage >= n ? 'Published ' + vs(vers[vers.length - 1]) + '. Users find it by its STAC collection; the COG is read in windows.'
        : 'Stage ' + (st.stage + 1) + ' of ' + n + ': ' + STAGE_DEF[st.stage][0] + ' (' + STAGE_DEF[st.stage][1].length + ' check' + (STAGE_DEF[st.stage][1].length > 1 ? 's' : '') + ').';
      txt(g, cap, 10, H - 14, st.blocked >= 0 ? C.red : C.fg, '11px sans-serif');

      const ovs = [35000, 17500, 8750, 4375, 2188];
      ro.innerHTML = [
        `mosaic      70,000 × 55,000 px at 10 m, 2 bands, uint8, no-data 255`,
        `overviews   ×2 ${ovs[0].toLocaleString('en-US')}×27,500 · ×4 ${ovs[1].toLocaleString('en-US')}×13,750 · ×8 ${ovs[2].toLocaleString('en-US')}×6,875 · ×16 4,375×3,438 · ×32 ${ovs[4].toLocaleString('en-US')}×1,719`,
        `items       56,711 STAC Items, one per 256 px tile (ids carry the version)`,
        `current     ${vs(vers[vers.length - 1])}     next ${ctl.querySelector('#bump').value} bump → ${vs(bumped())}`,
      ].join('\n');
    }, { fps: 30 });
    bind(ctl, st, restart);
    restart();
  };

  /* ================================================================
   * 4. Drift monitor: seasonal NDVI distributions, alarm, re-validation
   * ================================================================ */
  AI.labs.drift = function (root) {
    const { view, ctl } = AI.labShell(root, 'Seasonal drift: NDVI distributions, an alarm and a re-validation trigger',
      'illustrative NDVI · reference 2019–2024: mean 0.62, standard deviation 0.08 · one season per step · the alarm compares each season mean with the reference in units of σ',
      '<canvas style="height:280px"></canvas><canvas style="height:150px;margin-top:6px"></canvas>', '');
    ctl.innerHTML = sl('tgt', '2026 season mean NDVI', 0.40, 0.70, 0.01, 0.55) + sl('tau', 'Alarm threshold τ (reference σ)', 0.25, 1.5, 0.05, 0.5)
      + sl('n', 'Re-validation sample size n', 50, 1000, 10, 200) + sl('acc', 'Measured map accuracy a', 0.70, 0.99, 0.01, 0.90)
      + transport() + '<div class="readout" id="ro"></div>';
    wire(ctl);
    const ro = ctl.querySelector('#ro');
    const cs = AI.$$('canvas', view).map((el) => AI.canvas(el));
    const REF = 0.62, SD = 0.08, ACC_REF = 0.90;
    const YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];
    const WIG = [0.004, -0.003, 0.002, -0.004, 0.003, -0.002];
    const seasonMeans = () => { const tgt = val(ctl, 'tgt'); return YEARS.map((y, i) => (i < 6 ? REF + WIG[i] : i === 6 ? REF + 0.5 * (tgt - REF) : tgt)); };
    const st = { play: false, step: false, pos: 0 };
    let muShow = REF, lastDraw = '';
    const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const restart = () => { st.pos = 0; st.play = false; st.step = false; muShow = REF; ctl.querySelector('#play').textContent = 'Play'; };
    AI.loop(root, (dt) => {
      if (st.step) { st.step = false; st.pos = Math.min(7, Math.floor(st.pos) + 1); }
      else if (st.play) { st.pos = Math.min(7, st.pos + dt * speed(ctl) / 2.2); if (st.pos >= 7) { st.play = false; ctl.querySelector('#play').textContent = 'Play'; } }
      const M = seasonMeans(), tau = val(ctl, 'tau'), n = val(ctl, 'n'), a = val(ctl, 'acc'), tgt = val(ctl, 'tgt');
      const ci = Math.min(7, Math.floor(st.pos));
      muShow += (M[ci] - muShow) * Math.min(1, dt * 2.5);
      const dkey = st.pos.toFixed(2) + '|' + muShow.toFixed(4) + '|' + tgt + tau + n + a + '|' + cs[0].w + 'x' + cs[0].h + '|' + document.documentElement.dataset.theme;
      if (dkey === lastDraw && !st.play) return;
      lastDraw = dkey;

      // top: distributions
      const T = cs[0], g = T.ctx, C = AI.col();
      g.clearRect(0, 0, T.w, T.h);
      const v = AI.view(T, 0.2, 0.9, 0, 5.8);
      AI.axes(g, v, { noY: true });
      const curve = (m, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); for (let k = 0; k <= 160; k++) { const x = 0.2 + 0.7 * k / 160; const y = pdf(x, m, SD); k ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.stroke(); g.lineWidth = 1; };
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      curve(REF, C.blue, 2.5);
      curve(muShow, C.orange, 2.5);
      g.restore();
      g.setLineDash([4, 4]); g.strokeStyle = C.blue; g.beginPath(); g.moveTo(v.X(REF), v.pad.t); g.lineTo(v.X(REF), v.pad.t + v.ph); g.stroke();
      g.strokeStyle = C.orange; g.beginPath(); g.moveTo(v.X(muShow), v.pad.t); g.lineTo(v.X(muShow), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      txt(g, 'blue = reference 2019–2024', v.pad.l + 8, v.pad.t + 14, C.blue, '11px sans-serif');
      txt(g, 'orange = season ' + YEARS[ci] + (ci < 7 ? '' : '') + ' (mean ' + muShow.toFixed(3) + ')', v.pad.l + 8, v.pad.t + 30, C.orange, '11px sans-serif');
      txt(g, 'NDVI', v.X(0.9) - 4, v.pad.t + v.ph - 6, C.muted, '10px sans-serif', 'right');

      // bottom: season strip with alarm band
      const B = cs[1], gb = B.ctx;
      gb.clearRect(0, 0, B.w, B.h);
      const vb = AI.view(B, -0.5, 7.5, 0.45, 0.70, { l: 40, r: 12, t: 10, b: 24 });
      gb.fillStyle = AI.mix(C.bg3, C.green, 0.35);
      gb.fillRect(vb.pad.l, vb.Y(REF + tau * SD), vb.pw, vb.Y(REF - tau * SD) - vb.Y(REF + tau * SD));
      gb.strokeStyle = C.blue; gb.setLineDash([4, 4]); gb.beginPath(); gb.moveTo(vb.pad.l, vb.Y(REF)); gb.lineTo(vb.pad.l + vb.pw, vb.Y(REF)); gb.stroke(); gb.setLineDash([]);
      let alarmYear = null;
      gb.font = '10px sans-serif';
      YEARS.forEach((y, i) => {
        if (i > ci) return;
        const x = vb.X(i), yy = vb.Y(M[i]), z = (M[i] - REF) / SD, bad = Math.abs(z) > tau;
        if (bad && alarmYear === null) alarmYear = y;
        gb.fillStyle = bad ? C.red : C.blue; gb.beginPath(); gb.arc(x, yy, 4.5, 0, 7); gb.fill();
        gb.fillStyle = C.muted; gb.textAlign = 'center'; gb.fillText(String(y).slice(2), x, vb.pad.t + vb.ph + 15); gb.textAlign = 'left';
      });
      gb.strokeStyle = C.orange; gb.beginPath(); gb.moveTo(vb.X(st.pos), vb.pad.t); gb.lineTo(vb.X(st.pos), vb.pad.t + vb.ph); gb.stroke();
      txt(gb, 'band = reference ± τσ (green) · dots: season means (red = alarm)', vb.pad.l + 6, vb.pad.t + 12, C.muted, '10px sans-serif');

      // numbers
      const zT = (tgt - REF) / SD;
      const half = 1.96 * Math.sqrt(a * (1 - a) / n), lo = a - half, hi = a + half;
      const inside = ACC_REF >= lo && ACC_REF <= hi;
      const verdict = alarmYear !== null
        ? 'ALARM at ' + alarmYear + ': schedule re-validation within 30 days (draw n stratified reference points).'
        : 'no alarm yet: keep the monthly season check running.';
      ro.innerHTML = [
        `2026 shift            z = (${tgt.toFixed(2)} − 0.62) / 0.08 = ${zT.toFixed(3)} σ   (threshold τ = ${tau.toFixed(2)} σ)`,
        `alarm band            ${(REF - tau * SD).toFixed(3)} to ${(REF + tau * SD).toFixed(3)} NDVI`,
        `accuracy 95% CI       a ± 1.96·√(a(1−a)/n) = ${a.toFixed(2)} ± ${half.toFixed(4)}  →  [${lo.toFixed(3)}, ${hi.toFixed(3)}]`,
        `reference accuracy 0.90 ${inside ? 'is inside the interval: a drop is not yet conclusive' : 'is outside the interval: the change is conclusive'}`,
        verdict,
      ].join('\n');
    }, { fps: 30 });
    bind(ctl, st, restart);
    restart();
  };

  /* ================================================================
   * 5. Location privacy: precision, grouping and identifiable records
   * ================================================================ */
  AI.labs.privacy = function (root) {
    const { view, ctl } = AI.labShell(root, 'Location privacy: precision, grouping and identifiable records',
      'illustrative: eight invented coordinates, not real sites · rounding snaps each record to a grid cell · a record is identifiable when its cell holds fewer than k records',
      '<canvas style="height:340px"></canvas>', '');
    ctl.innerHTML = sl('dp', 'Decimal places kept', 1, 4, 1, 3) + sl('k', 'Group size k', 2, 5, 1, 3) + transport() + '<div class="readout" id="ro"></div>';
    wire(ctl);
    const ro = ctl.querySelector('#ro');
    const c = AI.canvas(AI.$('canvas', view));
    const PTS = [[9.03123, 38.74691], [9.03118, 38.74702], [9.03131, 38.74688], [9.03904, 38.75511], [9.04270, 38.76003], [9.03127, 38.74695], [9.05511, 38.72019], [9.03119, 38.74699]];
    const LAT0 = 9.03, LON0 = 38.74, KX = 109.94, KY = 111.32; // km per degree of longitude (at 9°N) and latitude
    const toKm = (la, lo) => [(lo - LON0) * KX, (la - LAT0) * KY];
    const pts = PTS.map(([la, lo]) => { const p = toKm(la, lo); return { x: p[0], y: p[1], tx: p[0], ty: p[1], ox: p[0], oy: p[1] }; });
    const st = { play: true, step: false };
    let ext = null, lastDraw = '', lastD = -1;
    const restart = () => { pts.forEach((p) => { p.x = p.ox; p.y = p.oy; }); ext = null; };
    AI.loop(root, (dt) => {
      const d = val(ctl, 'dp'), k = val(ctl, 'k');
      if (d !== lastD) { lastD = d; }
      const keys = PTS.map(([la, lo]) => la.toFixed(d) + ',' + lo.toFixed(d));
      const count = {};
      keys.forEach((kk) => { count[kk] = (count[kk] || 0) + 1; });
      PTS.forEach(([la, lo], i) => { const p = toKm(+la.toFixed(d), +lo.toFixed(d)); pts[i].tx = p[0]; pts[i].ty = p[1]; });
      const e = st.play ? Math.min(1, dt * 3) : 1;
      pts.forEach((p) => { p.x += (p.tx - p.x) * e; p.y += (p.ty - p.y) * e; });
      // view: fit all records, original and snapped, with a minimum span; eased
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach((p) => { x0 = Math.min(x0, p.x, p.ox); x1 = Math.max(x1, p.x, p.ox); y0 = Math.min(y0, p.y, p.oy); y1 = Math.max(y1, p.y, p.oy); });
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, span = Math.max(6, x1 - x0 + 2, y1 - y0 + 2);
      const tgt = { x0: cx - span / 2, x1: cx + span / 2, y0: cy - span / 2, y1: cy + span / 2 };
      if (!ext) ext = { ...tgt };
      const ee = Math.min(1, dt * 2);
      ['x0', 'x1', 'y0', 'y1'].forEach((q) => { ext[q] += (tgt[q] - ext[q]) * ee; });
      const dkey = d + '|' + k + '|' + pts.map((p) => p.x.toFixed(2) + p.y.toFixed(2)).join(',') + '|' + c.w + 'x' + c.h + '|' + document.documentElement.dataset.theme;
      if (dkey === lastDraw && !st.play) return;
      lastDraw = dkey;

      const C = AI.col(), g = c.ctx;
      g.clearRect(0, 0, c.w, c.h);
      const v = AI.view(c, ext.x0, ext.x1, ext.y0, ext.y1);
      AI.axes(g, v);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      // occupied grid cells, drawn at the snapped centre, sized by the precision
      const half_y = 0.5 * Math.pow(10, -d) * KY, half_x = 0.5 * Math.pow(10, -d) * KX;
      const done = {};
      PTS.forEach(([la, lo], i) => {
        const kk = keys[i]; if (done[kk]) return; done[kk] = true;
        const p = toKm(+la.toFixed(d), +lo.toFixed(d));
        const bad = count[kk] < k, col = bad ? C.red : C.green;
        g.fillStyle = AI.mix(C.bg, col, 0.18); g.strokeStyle = col; g.setLineDash([4, 3]);
        g.fillRect(v.X(p[0] - half_x), v.Y(p[1] + half_y), v.X(p[0] + half_x) - v.X(p[0] - half_x), v.Y(p[1] - half_y) - v.Y(p[1] + half_y));
        g.strokeRect(v.X(p[0] - half_x), v.Y(p[1] + half_y), v.X(p[0] + half_x) - v.X(p[0] - half_x), v.Y(p[1] - half_y) - v.Y(p[1] + half_y));
        g.setLineDash([]);
        txt(g, 'n=' + count[kk], v.X(p[0]) + 4, v.Y(p[1]) - 4, col, '600 11px monospace');
      });
      g.restore();
      // records: hollow = original position, filled = after snapping
      pts.forEach((p, i) => {
        const kk = keys[i], col = count[kk] < k ? C.red : C.green;
        g.strokeStyle = C.muted; g.beginPath(); g.arc(v.X(p.ox), v.Y(p.oy), 3, 0, 7); g.stroke();
        g.fillStyle = col; g.beginPath(); g.arc(v.X(p.x), v.Y(p.y), 4.5, 0, 7); g.fill();
      });
      const cellsN = Object.keys(count).length;
      const ident = keys.filter((kk) => count[kk] < k).length;
      const sizes = Object.values(count);
      txt(g, 'red = fewer than k records share the cell: someone could be picked out by place and time', 10, c.h - 24, C.red, '11px sans-serif');
      txt(g, 'grey ring = true position · filled dot = position after rounding (the view widens when points move apart)', 10, c.h - 8, C.muted, '10px sans-serif');
      ro.innerHTML = [
        `decimal places ${d}    cell ≈ ${(111320 * Math.pow(10, -d)).toFixed(1)} m north–south × ${(109940 * Math.pow(10, -d)).toFixed(1)} m east–west`,
        `occupied cells ${cellsN}     largest ${Math.max(...sizes)}    smallest ${Math.min(...sizes)}`,
        `identifiable records (cell count < ${k}):  ${ident} of ${PTS.length}`,
        ident === 0 ? 'every occupied cell holds at least k records' : 'rounding alone does not protect isolated records: suppress them or merge them into a coarser cell',
      ].join('\n');
    }, { fps: 30 });
    bind(ctl, st, restart);
    ctl.querySelector('#play').textContent = 'Pause';
    restart();
  };

  /* ================================================================
   * 6. Application selector: which stages decide success
   * ================================================================ */
  AI.labs.appsel = function (root) {
    const { view, ctl } = AI.labShell(root, 'Which pipeline stages matter most for each application?',
      'illustrative judgement, not a measurement · each bar says how much that stage decides whether the product succeeds (0 to 10)',
      '<canvas style="height:280px"></canvas>', '');
    const STAGES = ['Acquire', 'Preprocess', 'Model', 'Inference', 'Post-process', 'Publish & monitor'];
    const CASES = {
      agri: { name: 'Agriculture', w: [6, 7, 6, 5, 4, 9], latency: 'weeks to months', need: 'Crop maps through the season from a cloud-robust composite; field-parcel labels; monitoring every season, because the same field looks different in each phase.' },
      disaster: { name: 'Disaster response', w: [10, 8, 6, 9, 6, 7], latency: 'hours', need: 'Flood or damage extent within hours; SAR (Sentinel-1) when clouds block optical; a pre-event baseline; fast tiled inference; uncertainty flags; a rapid COG release.' },
      urban: { name: 'Urban planning', w: [4, 6, 9, 5, 9, 6], latency: 'months', need: 'Building footprints from high-resolution imagery; instance masks and polygon QA; yearly change; labels from the city; topology checks before release.' },
      climate: { name: 'Climate', w: [7, 10, 5, 4, 4, 9], latency: 'years', need: 'Decades of archive with calibration consistent across sensors; time-series products; long-term drift checks; versioned, citable releases.' },
      cons: { name: 'Conservation', w: [5, 6, 7, 4, 5, 10], latency: 'weekly alerts', need: 'Deforestation or poaching alerts; sensitive locations kept coarse or delayed; licences and community consent; a published method with error rates.' },
    };
    const keys = Object.keys(CASES);
    ctl.innerHTML = '<div class="row">' + keys.map((k, i) => `<button class="btn small${i === 0 ? ' primary' : ''}" data-case="${k}">${CASES[k].name}</button>`).join('') + '</div>'
      + '<div class="callout" id="need"></div>' + '<div class="readout" id="ro"></div>';
    const c = AI.canvas(AI.$('canvas', root.querySelector('.lab-view')));
    const need = ctl.querySelector('#need'), ro = ctl.querySelector('#ro');
    const st = { key: 'agri' };
    let cur = CASES.agri.w.map(() => 0), lastDraw = '';
    const setCase = (k) => {
      st.key = k;
      ctl.querySelectorAll('button[data-case]').forEach((b) => b.classList.toggle('primary', b.dataset.case === k));
      const cs = CASES[k];
      need.innerHTML = '<b>What the pipeline needs.</b> ' + cs.need + ' <i>Typical cadence: ' + cs.latency + '.</i>';
      ro.innerHTML = STAGES.map((s, i) => `${s.padEnd(16)} ${String(cs.w[i]).padStart(2)} / 10`).join('\n');
    };
    ctl.querySelectorAll('button[data-case]').forEach((b) => { b.onclick = () => setCase(b.dataset.case); });
    setCase('agri');
    AI.loop(root, (dt) => {
      const target = CASES[st.key].w;
      let moving = false;
      cur = cur.map((v, i) => { const nv = v + (target[i] - v) * Math.min(1, dt * 4); if (Math.abs(nv - target[i]) > 0.01) moving = true; return Math.abs(nv - target[i]) < 0.01 ? target[i] : nv; });
      const dkey = st.key + '|' + cur.map((v) => v.toFixed(2)).join(',') + '|' + c.w + 'x' + c.h + '|' + document.documentElement.dataset.theme;
      if (!moving && dkey === lastDraw) return;
      lastDraw = dkey;
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h;
      g.clearRect(0, 0, W, H);
      const colors = [C.blue, C.teal, C.purple, C.orange, C.pink, C.green];
      const lx = 10, labW = 130, bx = lx + labW, bw = W - bx - 60, rowH = (H - 40) / STAGES.length;
      txt(g, CASES[st.key].name + ': how much each stage decides success', lx, 14, C.fg, '600 12px sans-serif');
      STAGES.forEach((s, i) => {
        const y = 26 + i * rowH;
        txt(g, s, lx, y + rowH * 0.62, C.fg, '11px sans-serif');
        g.fillStyle = C.bg3; rr(g, bx, y + rowH * 0.2, bw, rowH * 0.5, 5); g.fill();
        g.fillStyle = colors[i]; rr(g, bx, y + rowH * 0.2, Math.max(4, bw * cur[i] / 10), rowH * 0.5, 5); g.fill();
        txt(g, cur[i].toFixed(1) + ' / 10', bx + bw + 8, y + rowH * 0.62, C.muted, '10px monospace');
      });
      txt(g, 'Scores are a teaching judgement: the stage with the highest bar is where the team should spend its effort first.', lx, H - 6, C.muted, '10px sans-serif');
    }, { fps: 30 });
  };
})();
