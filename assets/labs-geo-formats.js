/* GeoAI Module 1 labs (geo-formats.html): tiled raster & overviews, range requests, coordinate order,
 * chunked arrays, resampling, cloud masks. Requires assets/lib.js.
 *   <div data-lab="tiledRaster"></div>   <div data-lab="rangeRequests"></div>   <div data-lab="coordCheck"></div>
 *   <div data-lab="chunkLab"></div>      <div data-lab="resampleLab"></div>     <div data-lab="cloudMask"></div>
 * Byte counts are decimal (1 MB = 1 000 000 B) and uncompressed unless a lab says otherwise. Animations run on dt,
 * with a speed slider, pause and single-step. Illustrative bandwidth and latency numbers are labelled as such.
 */
(function () {
  const fi = (n) => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  const fb = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : b >= 1e3 ? (b / 1e3).toFixed(1) + ' kB' : Math.round(b) + ' B');
  const MONO = '11px ui-monospace,Menlo,Consolas,monospace';
  const NUM = (root, k) => +root.querySelector(`[data-k="${k}"]`).value;
  const STR = (root, k) => root.querySelector(`[data-k="${k}"]`).value;
  const inp = (k, label, v, min, max, step = 1) => `<label>${label}<input type="number" data-k="${k}" value="${v}" min="${min}" max="${max}" step="${step}"></label>`;
  const sel = (k, label, opts, v) => `<label>${label}<select data-k="${k}">${opts.map(([val, txt]) => `<option value="${val}"${String(val) === String(v) ? ' selected' : ''}>${txt}</option>`).join('')}</select></label>`;
  const BTN = (a, txt, extra = '') => `<button class="btn small" data-a="${a}" ${extra}>${txt}</button>`;
  const TX = '<div class="row"><button class="btn small" data-a="play">⏸ Pause</button><button class="btn small" data-a="step">Step ▸</button>'
    + '<label style="flex-direction:row;align-items:center;gap:6px">Speed <input type="range" min="0.25" max="2" step="0.05" value="1" data-a="speed" style="width:120px"></label></div>';
  const CAP = '<div class="geo-cap" style="font-size:13px;line-height:1.5;min-height:44px;margin-top:8px"></div>';
  const RO = '<div class="readout geo-ro" style="white-space:pre-wrap"></div>';

  /* Binds the shared pause / step / speed controls inside `root`. Returns a state object read each frame. */
  function tx(root) {
    const st = { playing: true, speed: 1, step: false };
    const play = root.querySelector('[data-a=play]'), stp = root.querySelector('[data-a=step]'), sp = root.querySelector('[data-a=speed]');
    play.onclick = () => { st.playing = !st.playing; play.textContent = st.playing ? '⏸ Pause' : '▶ Play'; };
    stp.onclick = () => { st.playing = false; play.textContent = '▶ Play'; st.step = true; };
    sp.oninput = () => (st.speed = +sp.value);
    return st;
  }
  /* Delegated listeners: controls (not the transport bar) call onChange; buttons with data-a call onAct. */
  function wire(root, onChange, onAct) {
    const inputs = (e) => { if (!e.target.dataset.a) onChange(e.target); };
    root.addEventListener('input', inputs);
    root.addEventListener('change', inputs);
    root.addEventListener('click', (e) => { const a = e.target.dataset.a; if (a && a !== 'play' && a !== 'step') onAct(a, e.target); });
  }

  /* ================= 1. Tiled raster explorer ================= */
  function tilePlan(p, L) {
    const s = 2 ** L, dim = Math.ceil(p.N / s), tiles = Math.ceil(dim / p.T);
    const r0 = Math.floor(p.row / s), c0 = Math.floor(p.col / s);
    const r1 = Math.min(dim - 1, Math.floor((p.row + p.rows - 1) / s)), c1 = Math.min(dim - 1, Math.floor((p.col + p.cols - 1) / s));
    const tr0 = Math.floor(r0 / p.T), tr1 = Math.floor(r1 / p.T), tc0 = Math.floor(c0 / p.T), tc1 = Math.floor(c1 / p.T);
    const list = [];
    for (let tr = tr0; tr <= tr1; tr++) for (let tc = tc0; tc <= tc1; tc++) list.push([tr, tc]);
    const tileB = p.T * p.T * p.bands * p.bps;
    return { s, dim, tiles, r0, c0, r1, c1, tr0, tr1, tc0, tc1, wr: r1 - r0 + 1, wc: c1 - c0 + 1, list, tileB,
      bytes: list.length * tileB, ideal: (r1 - r0 + 1) * (c1 - c0 + 1) * p.bands * p.bps };
  }

  AI.labs.tiledRaster = function (root) {
    const ctl = `<div class="row">${inp('N', 'Side N (px)', 10000, 512, 100000)}${sel('T', 'Tile T (px)', [[256, '256'], [512, '512'], [1024, '1024']], 512)}</div>
      <div class="row">${inp('bands', 'Bands', 4, 1, 12)}${sel('bps', 'Bytes / sample', [[1, '1 (uint8)'], [2, '2 (uint16)'], [4, '4 (float32)']], 2)}</div>
      <div class="row">${sel('L', 'Overview level shown', [0, 1, 2, 3, 4, 5].map((i) => [i, i ? 'level ' + i + ' (÷' + 2 ** i + ')' : '0 · full resolution']), 0)}</div>
      <div class="row">${inp('row', 'Window row', 1000, 0, 99999)}${inp('col', 'Window col', 1500, 0, 99999)}</div>
      <div class="row">${inp('rows', 'Window rows', 1000, 1, 99999)}${inp('cols', 'Window cols', 1000, 1, 99999)}</div>
      <div class="row">${BTN('req', 'Request window')}${BTN('clear', 'Clear')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'Tiles, overviews & one window',
      'a COG is cut into tiles; a window request fetches whole tiles',
      '<canvas data-c="grid" style="height:300px"></canvas><canvas data-c="pyr" style="height:120px"></canvas>' + CAP, ctl);
    const cg = AI.canvas(view.querySelector('[data-c=grid]')), cp = AI.canvas(view.querySelector('[data-c=pyr]'));
    const cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro');
    const st = tx(root);
    let list = [], fetched = 0, fetching = false, acc = 0, dirty = true;
    const P = () => ({ N: NUM(root, 'N'), T: NUM(root, 'T'), bands: NUM(root, 'bands'), bps: NUM(root, 'bps'), L: NUM(root, 'L'),
      row: NUM(root, 'row'), col: NUM(root, 'col'), rows: NUM(root, 'rows'), cols: NUM(root, 'cols') });
    const levels = (p) => {
      const out = [];
      for (let L = 0; L < 12; L++) { const dim = Math.ceil(p.N / 2 ** L), tiles = Math.ceil(dim / p.T); out.push({ dim, tiles }); if (tiles === 1) break; }
      return out;
    };
    const reset = () => { list = tilePlan(P(), P().L).list; fetched = 0; fetching = false; dirty = true; };
    reset();
    wire(root, reset, (a) => {
      if (a === 'req') { list = tilePlan(P(), P().L).list; fetched = 0; acc = 0; fetching = true; }
      if (a === 'clear') { fetched = 0; fetching = false; }
      dirty = true;
    });
    cg.onresize = cp.onresize = () => (dirty = true);

    const drawGrid = (p, pl) => {
      const g = cg.ctx, C = AI.col();
      g.clearRect(0, 0, cg.w, cg.h);
      const side = Math.min(cg.w - 20, cg.h - 40), k = side / (pl.tiles * p.T), x0 = 10, y0 = 30;
      const idx = new Map(list.map((t, i) => [t[0] + ',' + t[1], i]));
      g.fillStyle = C.bg2; g.strokeStyle = C.border; g.lineWidth = 1;
      g.fillRect(x0, y0, pl.tiles * p.T * k, pl.tiles * p.T * k);
      for (let tr = 0; tr < pl.tiles; tr++) for (let tc = 0; tc < pl.tiles; tc++) {
        const x = x0 + tc * p.T * k, y = y0 + tr * p.T * k, i = idx.get(tr + ',' + tc);
        if (i !== undefined) { g.fillStyle = i < fetched ? AI.col().blue : AI.col().orange; g.globalAlpha = i < fetched ? 0.55 : 0.28; g.fillRect(x, y, p.T * k, p.T * k); g.globalAlpha = 1; }
        g.strokeRect(x + 0.5, y + 0.5, p.T * k, p.T * k);
      }
      g.setLineDash([6, 4]); g.strokeStyle = C.red; g.lineWidth = 2;
      g.strokeRect(x0 + pl.c0 * k, y0 + pl.r0 * k, pl.wc * k, pl.wr * k);
      g.setLineDash([]);
      g.fillStyle = C.muted; g.font = MONO;
      g.fillText(`level ${p.L}: ${fi(pl.dim)} × ${fi(pl.dim)} px · ${pl.tiles} × ${pl.tiles} tiles · orange = touched, blue = read`, x0, 18);
    };
    const drawPyr = (p) => {
      const g = cp.ctx, C = AI.col(), Ls = levels(p);
      g.clearRect(0, 0, cp.w, cp.h);
      const slot = cp.w / Ls.length;
      Ls.forEach((lv, i) => {
        const side = Math.min(slot - 16, cp.h - 44), x = i * slot + 8, y = 6, k = side / lv.tiles;
        g.strokeStyle = i === p.L ? C.blue : C.border; g.lineWidth = i === p.L ? 2 : 1;
        for (let r = 0; r < lv.tiles; r++) for (let q = 0; q < lv.tiles; q++) g.strokeRect(x + q * k, y + r * k, k, k);
        g.fillStyle = i === p.L ? C.blue : C.muted; g.font = MONO; g.textAlign = 'center';
        g.fillText(`L${i} · ${lv.tiles}×${lv.tiles}`, x + side / 2, y + side + 14);
        g.fillText(`${fi(lv.dim)} px`, x + side / 2, y + side + 28);
      });
      g.textAlign = 'left';
    };
    AI.loop(root, (dt) => {
      if (fetching) {
        if (st.step) { fetched++; st.step = false; dirty = true; }
        else if (st.playing) { acc += dt * st.speed; while (acc >= 0.45 && fetched < list.length) { fetched++; acc -= 0.45; dirty = true; } }
        if (fetched >= list.length) { fetching = false; dirty = true; }
      } else st.step = false;
      if (!dirty) return;
      dirty = false;
      const p = P(), pl = tilePlan(p, p.L);
      drawGrid(p, pl); drawPyr(p);
      const fetchedB = fetched * pl.tileB;
      ro.textContent = [
        `Level ${p.L}: ${fi(pl.dim)} × ${fi(pl.dim)} px, tiled ${pl.tiles} × ${pl.tiles} = ${fi(pl.tiles * pl.tiles)} tiles of ${p.T} × ${p.T}`,
        `Window touches ${pl.list.length} tiles (tile rows ${pl.tr0}–${pl.tr1}, tile cols ${pl.tc0}–${pl.tc1})`,
        `One tile = ${p.T} × ${p.T} × ${p.bands} bands × ${p.bps} B = ${fi(pl.tileB)} B`,
        `All touched tiles = ${fi(pl.bytes)} B (${fb(pl.bytes)})`,
        `Exact window pixels = ${pl.wr} × ${pl.wc} × ${p.bands} × ${p.bps} = ${fi(pl.ideal)} B`,
        `Read so far: ${fetched} of ${pl.list.length} tiles = ${fi(fetchedB)} B`,
        `Whole level, every padded tile = ${fi(pl.tiles * pl.tiles * pl.tileB)} B (uncompressed)`,
      ].join('\n');
      if (fetching && fetched < list.length) {
        const [r, q] = list[fetched];
        cap.textContent = `Reading tile (${r}, ${q}). The whole ${p.T} × ${p.T} tile comes back, even if the window needs one corner of it.`;
      } else if (fetched > 0) {
        cap.textContent = `Done. The red box needed ${fi(pl.ideal)} B of pixels, but ${fi(pl.bytes)} B of whole tiles were read. Real files are compressed, so every byte count here is an upper bound.`;
      } else {
        cap.textContent = 'Orange tiles are the ones the red window touches. Press Request window to read them, or change the overview level to see how the pyramid shrinks the work.';
      }
    }, { fps: 30 });
  };

  /* ================= 2. Range requests ================= */
  const HDR = 65536; // header + IFD + tile index, first request (illustrative size)
  AI.labs.rangeRequests = function (root) {
    const ctl = `<div class="row">${inp('N', 'Side N (px)', 10000, 512, 100000)}${sel('T', 'Tile T (px)', [[256, '256'], [512, '512'], [1024, '1024']], 512)}</div>
      <div class="row">${inp('bands', 'Bands', 4, 1, 12)}${sel('bps', 'Bytes / sample', [[1, '1'], [2, '2'], [4, '4']], 2)}</div>
      <div class="row">${inp('row', 'Window row', 1000, 0, 99999)}${inp('col', 'Window col', 1500, 0, 99999)}</div>
      <div class="row">${inp('rows', 'Window rows', 1000, 1, 99999)}${inp('cols', 'Window cols', 1000, 1, 99999)}</div>
      <div class="row">${sel('merge', 'Ranges', [['row', 'one range per tile row'], ['tile', 'one range per tile']], 'row')}</div>
      <div class="row">${inp('lat', 'Latency (ms)', 80, 0, 2000)}${inp('bw', 'Bandwidth (MB/s)', 25, 0.1, 10000)}</div>
      <div class="row">${BTN('req', 'Read window')}${BTN('naive', 'Download whole file')}${BTN('reset', 'Reset')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'Partial reads over HTTP',
      'the client asks for byte ranges; the server returns only those bytes (HTTP 206)',
      '<canvas data-c="net" style="height:330px"></canvas>' + CAP, ctl);
    const cv = AI.canvas(view.querySelector('[data-c=net]')), cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro');
    const st = tx(root);
    let reqs = [], total = 0, nx = 1, done = 0, running = false, t = 0, dirty = true, tb = 1, mode = 'window';
    const P = () => ({ N: NUM(root, 'N'), T: NUM(root, 'T'), bands: NUM(root, 'bands'), bps: NUM(root, 'bps'), row: NUM(root, 'row'),
      col: NUM(root, 'col'), rows: NUM(root, 'rows'), cols: NUM(root, 'cols'), merge: STR(root, 'merge'), lat: NUM(root, 'lat'), bw: NUM(root, 'bw') });
    const build = (naive) => {
      const p = P();
      nx = Math.ceil(p.N / p.T); tb = p.T * p.T * p.bands * p.bps; total = HDR + nx * nx * tb;
      const durOf = (bytes) => p.lat / 1000 + bytes / (p.bw * 1e6);
      if (naive) {
        mode = 'naive';
        const all = []; for (let r = 0; r < nx; r++) for (let q = 0; q < nx; q++) all.push([r, q]);
        return [{ label: 'the whole file, one request', from: 0, to: total - 1, tiles: all, bytes: total, dur: durOf(total) }];
      }
      mode = 'window';
      const r0 = p.row, c0 = p.col, r1 = Math.min(p.N - 1, r0 + p.rows - 1), c1 = Math.min(p.N - 1, c0 + p.cols - 1);
      const tr0 = Math.floor(r0 / p.T), tr1 = Math.floor(r1 / p.T), tc0 = Math.floor(c0 / p.T), tc1 = Math.floor(c1 / p.T);
      const out = [{ label: 'header, IFD and tile index', from: 0, to: HDR - 1, tiles: [], bytes: HDR, dur: durOf(HDR) }];
      for (let tr = tr0; tr <= tr1; tr++) {
        const groups = p.merge === 'row' ? [[tc0, tc1]] : Array.from({ length: tc1 - tc0 + 1 }, (_, i) => [tc0 + i, tc0 + i]);
        for (const [a, b] of groups) {
          const from = HDR + (tr * nx + a) * tb, to = HDR + (tr * nx + b + 1) * tb - 1;
          const tiles = []; for (let q = a; q <= b; q++) tiles.push([tr, q]);
          out.push({ label: a === b ? `tile (${tr}, ${a})` : `tile row ${tr}, tiles ${a}–${b}`, from, to, tiles, bytes: to - from + 1, dur: durOf(to - from + 1) });
        }
      }
      return out;
    };
    const restart = () => { reqs = build(false); done = 0; t = 0; running = false; dirty = true; };
    restart();
    wire(root, restart, (a) => {
      if (a === 'req') { reqs = build(false); done = 0; t = 0; running = true; }
      if (a === 'naive') { reqs = build(true); done = 0; t = 0; running = true; }
      if (a === 'reset') restart();
      dirty = true;
    });
    cv.onresize = () => (dirty = true);

    const simTime = () => reqs.slice(0, done).reduce((s, r) => s + r.dur, 0) + (done < reqs.length ? Math.min(t, reqs[done].dur) : 0);
    const doneBytes = () => reqs.slice(0, done).reduce((s, r) => s + r.bytes, 0);

    AI.loop(root, (dt) => {
      if (running && done < reqs.length) {
        if (st.step) { t = 1e9; st.step = false; }
        else if (st.playing) t += dt * st.speed;
        if (t >= reqs[done].dur) { done++; t = 0; if (done >= reqs.length) running = false; }
        dirty = true;
      } else st.step = false;
      if (!dirty) return;
      dirty = false;
      const g = cv.ctx, C = AI.col(), p = P();
      g.clearRect(0, 0, cv.w, cv.h);
      // client panel
      const cx0 = 8, cw = Math.min(150, cv.w * 0.26), ch = cv.h - 16;
      g.fillStyle = C.bg2; g.strokeStyle = C.border; g.fillRect(cx0, 8, cw, ch); g.strokeRect(cx0 + 0.5, 8.5, cw, ch);
      g.fillStyle = C.fg; g.font = '600 12px ui-monospace,Menlo,Consolas,monospace'; g.fillText('your code', cx0 + 10, 28);
      g.fillStyle = C.muted; g.font = MONO; g.fillText('reader', cx0 + 10, 44);
      // remote file: grid of tiles in file order
      const fx0 = cx0 + cw + 60, side = Math.min(cv.w - fx0 - 12, cv.h - 60), k = side / nx, fy0 = 46;
      g.fillStyle = C.muted; g.font = MONO; g.fillText('remote file (object storage)', fx0, 20);
      g.fillStyle = C.border; g.fillRect(fx0, fy0 - 14, side, 8);
      g.fillStyle = C.muted; g.fillText('header', fx0, fy0 - 18);
      const fill = new Map();
      const cur = done < reqs.length ? reqs[done] : null;
      const frac = cur ? Math.min(1, t / cur.dur) : 0;
      reqs.slice(0, done).forEach((r) => r.tiles.forEach(([a, b]) => fill.set(a + ',' + b, 'done')));
      if (cur) cur.tiles.slice(0, Math.floor(frac * cur.tiles.length)).forEach(([a, b]) => fill.set(a + ',' + b, 'now'));
      for (let r = 0; r < nx; r++) for (let q = 0; q < nx; q++) {
        const f = fill.get(r + ',' + q), x = fx0 + q * k, y = fy0 + r * k;
        g.fillStyle = f === 'done' ? C.green : f === 'now' ? C.blue : C.bg2;
        g.globalAlpha = f ? 0.8 : 1;
        g.fillRect(x, y, k, k); g.globalAlpha = 1;
        if (k > 6) { g.strokeStyle = C.border; g.strokeRect(x + 0.5, y + 0.5, k, k); }
      }
      if (cur && done < reqs.length) {
        // response packet travelling from the file to the client
        const px = fx0 - 20 - frac * (fx0 - cx0 - cw - 26), py = Math.min(cv.h - 24, fy0 + side + 30);
        g.fillStyle = C.blue; g.globalAlpha = 0.9; g.fillRect(px - 6, py - 9, 130, 18); g.globalAlpha = 1;
        g.fillStyle = '#fff'; g.font = MONO; g.fillText(`206 · ${fb(cur.bytes)}`, px, py + 4);
      }
      g.fillStyle = C.muted; g.font = MONO;
      g.fillText(`file ${fb(total)} · ${nx}×${nx} tiles of ${p.T} px, row-major`, fx0, fy0 + side + 16);
      // readout
      const lines = reqs.map((r, i) => {
        const s = i < done ? '✓ done' : i === done && running ? '… in flight' : 'waiting';
        return `#${i + 1}  bytes=${fi(r.from)}–${fi(r.to)}  ${fi(r.bytes)} B  ${s}  (${r.label})`;
      });
      const pct = total ? (100 * doneBytes() / total) : 0;
      ro.textContent = `requests done: ${done} of ${reqs.length} · bytes received: ${fi(doneBytes())} B · simulated time: ${simTime().toFixed(2)} s\n`
        + `share of the file received: ${pct.toFixed(3)} %  (file = ${fi(total)} B)\n` + (lines.length > 8 ? lines.slice(0, 8).join('\n') + `\n… ${lines.length - 8} more` : lines.join('\n'));
      if (mode === 'naive') {
        cap.textContent = running ? `One request for the whole ${fb(total)}. Its time is latency + size / bandwidth. Nothing can be used until it finishes.` : `Naive download: ${fb(total)} read to use ${fi(P().rows * P().cols * P().bands * P().bps)} B of pixels.`;
      } else if (running && cur) {
        cap.textContent = `Request ${done + 1}: ${cur.label}. Asks for bytes ${fi(cur.from)}–${fi(cur.to)}. Each range is a separate round trip, so merging neighbours saves latency.`;
      } else if (done > 0 && done >= reqs.length) {
        cap.textContent = `Done: ${reqs.length} requests, ${fi(doneBytes())} B, about ${(100 * doneBytes() / total).toFixed(2)} % of the file. Switch Ranges to one range per tile and compare the count.`;
      } else {
        cap.textContent = 'Press Read window. Step 1 reads the header and tile index (64 KiB). Then each tile row is one range request, merged into one contiguous byte block.';
      }
    }, { fps: 30 });
  };

  /* ================= 3. Coordinate order checker ================= */
  const BOX = [['France', -5.2, 9.6, 41.3, 51.1], ['Spain', -9.3, 3.3, 36.0, 43.8], ['Kenya', 33.9, 41.9, -4.7, 5.0],
    ['Japan', 129.5, 145.8, 31.0, 45.5], ['Brazil', -73.9, -34.8, -33.7, 5.3]];
  const PRE = { Paris: [2.3522, 48.8566, 'France'], Nairobi: [36.8219, -1.2921, 'Kenya'], Tokyo: [139.6917, 35.6895, 'Japan'] };
  AI.labs.coordCheck = function (root) {
    const ctl = `<div class="row">${inp('lon', 'Lon (x), true place', 2.3522, -360, 360, 0.0001)}${inp('lat', 'Lat (y), true place', 48.8566, -360, 360, 0.0001)}</div>
      <div class="row">${Object.keys(PRE).map((n) => BTN('pre', n, `data-v="${n}"`)).join('')}</div>
      <div class="row">${sel('prod', 'The file stores the pair as', [['ll', '[lon, lat]  GeoJSON order'], ['la', '[lat, lon]']], 'll')}</div>
      <div class="row">${sel('read', 'The reader assumes the pair is', [['ll', '[lon, lat]'], ['la', '[lat, lon]']], 'll')}</div>
      <div class="row">${sel('exp', 'Expected country', BOX.map((b) => [b[0], b[0]]), 'France')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'Which order is the pair in?',
      'a swapped pair does not crash the software; it puts the feature somewhere else',
      '<canvas data-c="map" style="height:300px"></canvas>' + CAP, ctl);
    const cv = AI.canvas(view.querySelector('[data-c=map]')), cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro');
    const st = tx(root);
    const evalc = () => {
      const lon = NUM(root, 'lon'), lat = NUM(root, 'lat'), prod = STR(root, 'prod'), read = STR(root, 'read'), exp = STR(root, 'exp');
      const stored = prod === 'll' ? [lon, lat] : [lat, lon];
      const rlon = read === 'll' ? stored[0] : stored[1], rlat = read === 'll' ? stored[1] : stored[0];
      const valid = Math.abs(rlat) <= 90 && Math.abs(rlon) <= 180;
      const box = valid ? BOX.find((b) => rlon >= b[1] && rlon <= b[2] && rlat >= b[3] && rlat <= b[4]) : null;
      const match = !!box && box[0] === exp;
      return { lon, lat, stored, rlon, rlat, valid, box, exp, match };
    };
    let dirty = true, disp = null, from = null, to = null, tp = 1;
    const target = () => { const e = evalc(); return { x: e.rlon, y: e.rlat }; };
    const retarget = () => {
      const tg = target();
      from = disp ? { ...disp } : { ...tg };
      to = tg; tp = disp ? 0 : 1; dirty = true;
    };
    retarget();
    wire(root, retarget, (a, el) => {
      if (a === 'pre') { const [lo, la, ex] = PRE[el.dataset.v]; root.querySelector('[data-k=lon]').value = lo; root.querySelector('[data-k=lat]').value = la; root.querySelector('[data-k=exp]').value = ex; retarget(); }
    });
    cv.onresize = () => (dirty = true);
    const px = (lon, w) => ((lon + 180) / 360) * w, py = (lat, h) => ((90 - lat) / 180) * h;
    AI.loop(root, (dt) => {
      if (tp < 1) {
        if (st.step) { tp = 1; st.step = false; } else if (st.playing) tp = Math.min(1, tp + dt * st.speed / 1.4);
        dirty = true;
      } else st.step = false;
      if (!dirty) return;
      dirty = false;
      const g = cv.ctx, C = AI.col(), e = evalc(), w = cv.w, h = cv.h, s = AI.smooth(tp);
      g.clearRect(0, 0, w, h);
      g.strokeStyle = C.border; g.lineWidth = 1; g.fillStyle = C.muted; g.font = MONO;
      for (let lo = -180; lo <= 180; lo += 30) { g.beginPath(); g.moveTo(px(lo, w) + 0.5, 0); g.lineTo(px(lo, w) + 0.5, h); g.stroke(); g.fillText(lo + '°', px(lo, w) + 3, h - 4); }
      for (let la = -90; la <= 90; la += 30) { g.beginPath(); g.moveTo(0, py(la, h) + 0.5); g.lineTo(w, py(la, h) + 0.5); g.stroke(); }
      BOX.forEach((b) => {
        const x0 = px(b[1], w), x1 = px(b[2], w), y0 = py(b[4], h), y1 = py(b[3], h);
        g.globalAlpha = 0.16; g.fillStyle = b[0] === e.exp ? C.orange : C.blue; g.fillRect(x0, y0, x1 - x0, y1 - y0); g.globalAlpha = 1;
        g.strokeStyle = b[0] === e.exp ? C.orange : C.border; g.lineWidth = b[0] === e.exp ? 2 : 1; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
        g.fillStyle = C.fg; g.fillText(b[0], x0 + 4, y0 + 13);
      });
      // true place (ring) and where the reader puts the feature (dot)
      const tx0 = px(Math.max(-180, Math.min(180, e.lon)), w), ty0 = py(Math.max(-90, Math.min(90, e.lat)), h);
      g.strokeStyle = C.green; g.lineWidth = 2; g.beginPath(); g.arc(tx0, ty0, 8, 0, 7); g.stroke();
      const cx = AI.lerp(from.x, to.x, s), cy = AI.lerp(from.y, to.y, s);
      disp = { x: cx, y: cy };
      const dx = px(Math.max(-180, Math.min(180, cx)), w), dy = py(Math.max(-90, Math.min(90, cy)), h);
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(tx0, ty0); g.lineTo(dx, dy); g.stroke(); g.setLineDash([]);
      g.fillStyle = !e.valid ? C.red : e.match ? C.green : C.red; g.beginPath(); g.arc(dx, dy, 6, 0, 7); g.fill();
      if (!e.valid) { g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(dx - 10, dy - 10); g.lineTo(dx + 10, dy + 10); g.moveTo(dx + 10, dy - 10); g.lineTo(dx - 10, dy + 10); g.stroke(); }
      ro.textContent = [
        `stored in file:  [${e.stored[0]}, ${e.stored[1]}]`,
        `reader reads:    longitude = ${e.rlon}, latitude = ${e.rlat}`,
        `checks:          |latitude| ≤ 90 ${Math.abs(e.rlat) <= 90 ? '✓' : '✗ INVALID'}   |longitude| ≤ 180 ${Math.abs(e.rlon) <= 180 ? '✓' : '✗'}`,
        `lands in:        ${e.box ? e.box[0] + ' (approximate box)' : e.valid ? 'no listed box (ocean or other land)' : 'nowhere (not a place on Earth)'}`,
        `expected:        ${e.exp} → ${e.match ? 'MATCH' : e.valid ? 'WRONG PLACE' : 'INVALID'}`,
      ].join('\n');
      if (!e.valid) cap.textContent = `The reader treats ${e.rlat} as a latitude, which is above 90°. No point on Earth has that value, so a validator rejects the file. Swap the reader setting to fix it.`;
      else if (!e.match && e.box) cap.textContent = `The feature jumped to ${e.box[0]}, which is wrong: the true place is the green ring. Coordinates were read in the other order.`;
      else if (!e.box) cap.textContent = `The point is in the sea or an unlisted area. Latitude and longitude are probably swapped; the stored order should be [lon, lat] for GeoJSON.`;
      else cap.textContent = `The reader and the producer agree. The point sits inside ${e.box[0]}, the expected country.`;
      if (tp < 1) cap.textContent = 'Moving the point to where the reader puts it. ' + cap.textContent;
    }, { fps: 30 });
  };

  /* ================= 4. Chunked cube and lazy reads ================= */
  AI.labs.chunkLab = function (root) {
    const SHAPES = [['10,256,256', 'A · 10 × 256 × 256'], ['1827,32,32', 'B · 1827 × 32 × 32'], ['30,512,512', 'C · 30 × 512 × 512'], ['183,128,128', 'D · 183 × 128 × 128']];
    const ctl = `<div class="row">${inp('Tn', 'Times T', 1827, 1, 100000)}${inp('Yn', 'Rows Y', 2000, 1, 100000)}${inp('Xn', 'Cols X', 2000, 1, 100000)}</div>
      <div class="row">${sel('shape', 'Chunk shape preset', SHAPES, '10,256,256')}</div>
      <div class="row">${inp('ct', 'chunk t', 10, 1, 100000)}${inp('cy', 'chunk y', 256, 1, 100000)}${inp('cx', 'chunk x', 256, 1, 100000)}</div>
      <div class="row">${inp('t0', 'query t from', 0, 0, 99999)}${inp('t1', 'to', 1826, 0, 99999)}</div>
      <div class="row">${inp('y0', 'y from', 1000, 0, 99999)}${inp('y1', 'to', 1000, 0, 99999)}${inp('x0', 'x from', 1000, 0, 99999)}${inp('x1', 'to', 1000, 0, 99999)}</div>
      <div class="row">${BTN('pix', 'Pixel time series')}${BTN('date', 'One date, all pixels')}${BTN('box', '400 × 400 box, all times')}</div>
      <div class="row">${BTN('req', 'Compute (reads chunks)')}${BTN('reset', 'Reset')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'A cube cut into chunks',
      'a lazy query reads only the chunks it overlaps; each chunk is one object in the store',
      '<canvas data-c="sp" style="height:250px"></canvas><canvas data-c="tm" style="height:70px"></canvas>' + CAP, ctl);
    const cs = AI.canvas(view.querySelector('[data-c=sp]')), ct = AI.canvas(view.querySelector('[data-c=tm]'));
    const cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro');
    const st = tx(root);
    const cl = (v, hi) => Math.min(hi - 1, Math.max(0, Math.round(v)));
    let Pn = null, fetched = 0, acc = 0, running = false, dirty = true, loaded = new Set(), cellTot = [], cellLoad = [], Ds = 1, Dx = 1, Dt = 1;
    const plan = () => {
      const Tn = NUM(root, 'Tn'), Yn = NUM(root, 'Yn'), Xn = NUM(root, 'Xn'), ctn = NUM(root, 'ct'), cy = NUM(root, 'cy'), cx = NUM(root, 'cx');
      const t0 = cl(NUM(root, 't0'), Tn), t1 = Math.max(t0, cl(NUM(root, 't1'), Tn));
      const y0 = cl(NUM(root, 'y0'), Yn), y1 = Math.max(y0, cl(NUM(root, 'y1'), Yn));
      const x0 = cl(NUM(root, 'x0'), Xn), x1 = Math.max(x0, cl(NUM(root, 'x1'), Xn));
      const nT = Math.ceil(Tn / ctn), nY = Math.ceil(Yn / cy), nX = Math.ceil(Xn / cx);
      const list = [];
      for (let yi = Math.floor(y0 / cy); yi <= Math.floor(y1 / cy); yi++) for (let xi = Math.floor(x0 / cx); xi <= Math.floor(x1 / cx); xi++)
        for (let ti = Math.floor(t0 / ctn); ti <= Math.floor(t1 / ctn); ti++) list.push([ti, yi, xi]);
      return { Tn, Yn, Xn, ct: ctn, cy, cx, nT, nY, nX, cb: ctn * cy * cx * 4, t0, t1, y0, y1, x0, x1, list,
        need: (t1 - t0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1) * 4, total: nT * nY * nX };
    };
    const rebuildCells = () => {
      Pn = plan();
      Ds = Math.min(Pn.nY, 12); Dx = Math.min(Pn.nX, 12); Dt = Math.min(Pn.nT, 60);
      cellTot = new Array(Ds * Dx).fill(0); cellLoad = new Array(Ds * Dx).fill(0);
      Pn.list.forEach(([ti, yi, xi]) => cellTot[Math.floor(yi * Ds / Pn.nY) * Dx + Math.floor(xi * Dx / Pn.nX)]++);
      loaded = new Set(); fetched = 0; acc = 0; running = false; dirty = true;
    };
    rebuildCells();
    wire(root, (el) => {
      if (el.dataset.k === 'shape') { const [a, b, d] = el.value.split(',').map(Number); root.querySelector('[data-k=ct]').value = a; root.querySelector('[data-k=cy]').value = b; root.querySelector('[data-k=cx]').value = d; }
      rebuildCells();
    }, (a) => {
      const Tn = NUM(root, 'Tn'), Yn = NUM(root, 'Yn'), Xn = NUM(root, 'Xn'), set = (k, v) => (root.querySelector(`[data-k=${k}]`).value = v);
      if (a === 'pix') { set('t0', 0); set('t1', Tn - 1); set('y0', 1000); set('y1', 1000); set('x0', 1000); set('x1', 1000); rebuildCells(); }
      if (a === 'date') { set('t0', 900); set('t1', 900); set('y0', 0); set('y1', Yn - 1); set('x0', 0); set('x1', Xn - 1); rebuildCells(); }
      if (a === 'box') { set('t0', 0); set('t1', Tn - 1); set('y0', 800); set('y1', 1199); set('x0', 800); set('x1', 1199); rebuildCells(); }
      if (a === 'req') { running = true; fetched = 0; acc = 0; loaded = new Set(); dirty = true; }
      if (a === 'reset') rebuildCells();
    });
    cs.onresize = ct.onresize = () => (dirty = true);
    const per = () => Math.min(0.3, Math.max(0.04, 18 / Math.max(1, Pn.list.length)));
    AI.loop(root, (dt) => {
      if (running && fetched < Pn.list.length) {
        if (st.step) { advance(); st.step = false; }
        else if (st.playing) { acc += dt * st.speed; while (acc >= per() && fetched < Pn.list.length) { advance(); acc -= per(); } }
        if (fetched >= Pn.list.length) running = false;
        dirty = true;
      } else st.step = false;
      if (!dirty) return;
      dirty = false;
      const C = AI.col();
      const g = cs.ctx, W = cs.w, H = cs.h;
      g.clearRect(0, 0, W, H);
      const cw = (W - 8) / Dx, chh = (H - 26) / Ds;
      for (let gy = 0; gy < Ds; gy++) for (let gx = 0; gx < Dx; gx++) {
        const i = gy * Dx + gx, x = 4 + gx * cw, y = 22 + gy * chh;
        g.fillStyle = C.bg2; g.fillRect(x, y, cw - 2, chh - 2);
        if (cellTot[i]) {
          g.globalAlpha = 0.25 + 0.75 * (cellLoad[i] / cellTot[i]);
          g.fillStyle = cellLoad[i] ? C.blue : C.orange; g.fillRect(x, y, cw - 2, chh - 2); g.globalAlpha = 1;
        }
      }
      g.fillStyle = C.muted; g.font = MONO;
      g.fillText(`spatial chunks (y down, x across): ${Pn.nY} × ${Pn.nX}, shown ${Ds} × ${Dx} cells`, 4, 14);
      // time strip for the spatial chunk being read
      const t = ct.ctx, TW = ct.w, TH = ct.h;
      t.clearRect(0, 0, TW, TH);
      const cur = Pn.list[Math.min(fetched, Pn.list.length - 1)] || [0, 0, 0];
      const tcw = (TW - 8) / Dt;
      for (let gt = 0; gt < Dt; gt++) {
        const ti0 = Math.floor(gt * Pn.nT / Dt), ti1 = Math.max(ti0, Math.floor((gt + 1) * Pn.nT / Dt) - 1);
        let inq = false, ld = false;
        for (let ti = ti0; ti <= ti1; ti++) { if (ti >= Math.floor(Pn.t0 / Pn.ct) && ti <= Math.floor(Pn.t1 / Pn.ct)) inq = true; if (loaded.has(ti + ',' + cur[1] + ',' + cur[2])) ld = true; }
        t.fillStyle = C.bg2; t.fillRect(4 + gt * tcw, 22, tcw - 1, TH - 30);
        if (inq) { t.globalAlpha = 0.35; t.fillStyle = ld ? C.green : C.orange; t.fillRect(4 + gt * tcw, 22, tcw - 1, TH - 30); t.globalAlpha = 1; }
        if (ld) { t.fillStyle = C.green; t.fillRect(4 + gt * tcw, 22, tcw - 1, TH - 30); }
      }
      t.fillStyle = C.muted; t.font = MONO;
      t.fillText(`time chunks for spatial chunk (${cur[1]}, ${cur[2]}): ${Pn.nT} along time, shown ${Dt}`, 4, 14);
      const got = fetched;
      const readB = got * Pn.cb;
      ro.textContent = [
        `chunk grid: ${fi(Pn.nT)} × ${Pn.nY} × ${Pn.nX} = ${fi(Pn.total)} chunks, each ${Pn.ct} × ${Pn.cy} × ${Pn.cx} × 4 B = ${fi(Pn.cb)} B`,
        `query needs ${fi(Pn.need)} B of values (${Pn.t1 - Pn.t0 + 1} × ${Pn.y1 - Pn.y0 + 1} × ${Pn.x1 - Pn.x0 + 1} × 4)`,
        `query touches ${fi(Pn.list.length)} chunks = ${fi(Pn.list.length * Pn.cb)} B read`,
        `efficiency (needed / read): ${(100 * Pn.need / (Pn.list.length * Pn.cb)).toPrecision(3)} %`,
        `read so far: ${got} of ${Pn.list.length} chunks = ${fi(readB)} B`,
      ].join('\n');
      if (!running && got === 0) cap.textContent = 'Lazy: building the query reads nothing. Press Compute to fetch the chunks it overlaps, one at a time for clarity; real stores fetch many in parallel.';
      else if (running) cap.textContent = `Reading chunk (t ${cur[0]}, y ${cur[1]}, x ${cur[2]}). Chunks outside the query are never opened.`;
      else cap.textContent = `Done: ${fi(readB)} B read for ${fi(Pn.need)} B of values. Try shape B for the pixel series and shape C for the date query.`;
    }, { fps: 30 });

    function advance() {
      if (fetched >= Pn.list.length) return;
      const [ti, yi, xi] = Pn.list[fetched];
      loaded.add(ti + ',' + yi + ',' + xi);
      cellLoad[Math.floor(yi * Ds / Pn.nY) * Dx + Math.floor(xi * Dx / Pn.nX)]++;
      fetched++;
    }
  };

  /* ================= 5. Resampling on a 3 × 3 patch ================= */
  const KEYS = (x) => { x = Math.abs(x); const a = -0.5; if (x < 1) return (a + 2) * x ** 3 - (a + 3) * x * x + 1; if (x < 2) return a * x ** 3 - 5 * a * x * x + 8 * a * x - 4 * a; return 0; };
  const WTS = (q, method) => {
    const cl = Math.min(2, Math.max(0, q));
    if (method === 'nearest') return [{ i: Math.min(2, Math.max(0, Math.round(cl))), w: 1 }];
    const i0 = Math.floor(cl);
    if (method === 'bilinear') { const i1 = Math.min(2, i0 + 1), f = cl - i0; return i1 === i0 ? [{ i: i0, w: 1 }] : [{ i: i0, w: 1 - f }, { i: i1, w: f }]; }
    const out = []; for (let k = -1; k <= 2; k++) { const i = Math.min(2, Math.max(0, i0 + k)); out.push({ i, w: KEYS(cl - (i0 + k)) }); }
    return out;
  };
  AI.labs.resampleLab = function (root) {
    const SRC = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    const ctl = `<div class="row">${sel('method', 'Method', [['nearest', 'nearest'], ['bilinear', 'bilinear (2×2 neighbours)'], ['cubic', 'cubic (4×4, Keys a = −0.5)']], 'bilinear')}</div>
      <div class="row">${sel('kind', 'Data', [['cont', 'continuous (reflectance)'], ['cat', 'categorical (class labels)']], 'cont')}${inp('M', 'Output side M', 5, 2, 8)}</div>
      <div class="geo-grid3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">${SRC.map((v) => `<input type="number" data-v step="any" value="${v}" style="width:100%">`).join('')}</div>
      <div class="row">${BTN('cls', 'Load class map')}${BTN('reset', 'Reset values')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'Resampling a 3 × 3 patch',
      'output pixel centres are mapped back into the source grid, then the neighbours are blended',
      '<canvas data-c="src" style="height:170px"></canvas><canvas data-c="out" style="height:260px"></canvas>' + CAP, ctl);
    const cS = AI.canvas(view.querySelector('[data-c=src]')), cO = AI.canvas(view.querySelector('[data-c=out]'));
    const cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro');
    const st = tx(root);
    const vals = () => [...root.querySelectorAll('[data-v]')].map((e) => +e.value || 0);
    let cur = 0, acc = 0, dirty = true;
    const setVals = (arr, kind) => { root.querySelectorAll('[data-v]').forEach((e, i) => (e.value = arr[i])); root.querySelector('[data-k=kind]').value = kind; dirty = true; };
    wire(root, () => { dirty = true; }, (a) => {
      if (a === 'cls') setVals([1, 1, 2, 1, 2, 2, 3, 3, 2], 'cat');
      if (a === 'reset') setVals(SRC, 'cont');
    });
    cS.onresize = cO.onresize = () => (dirty = true);
    const ramp = (t) => { const v = Math.round(255 * Math.min(1, Math.max(0, t))); return `rgb(${v},${v},${v})`; };
    const PAL = ['#2f81f7', '#3fb950', '#d29922', '#f85149', '#a371f7', '#39c5cf'];
    AI.loop(root, (dt) => {
      if (st.step) { cur = (cur + 1) % (M() * M()); st.step = false; dirty = true; }
      else if (st.playing) { acc += dt * st.speed; if (acc >= 0.9) { acc = 0; cur = (cur + 1) % (M() * M()); dirty = true; } }
      if (!dirty) return;
      dirty = false;
      const A = vals(), m = M(), method = STR(root, 'method'), cat = STR(root, 'kind') === 'cat';
      const lo = Math.min(...A), hi = Math.max(...A), C = AI.col();
      const pos = (j) => (j + 0.5) * (3 / m) - 0.5;
      const eval2 = (px, py) => {
        const wy = WTS(py, method), wx = WTS(px, method); let s = 0; const taps = [];
        for (const a of wy) for (const b of wx) { s += a.w * b.w * A[a.i * 3 + b.i]; taps.push({ r: a.i, c: b.i, w: a.w * b.w }); }
        return { v: s, taps, wx, wy };
      };
      const r = Math.floor(cur / m), q = cur % m;
      const here = eval2(pos(q), pos(r));
      // source
      const gs = cS.ctx, sw = cS.w, sh = cS.h, cw = sw / 3, chh = sh / 3;
      gs.clearRect(0, 0, sw, sh);
      for (let i = 0; i < 9; i++) {
        const x = (i % 3) * cw, y = Math.floor(i / 3) * chh, v = A[i];
        gs.fillStyle = cat ? PAL[Math.round(v) % PAL.length] : ramp((v - lo) / (hi - lo || 1));
        gs.fillRect(x, y, cw, chh);
        gs.strokeStyle = C.border; gs.strokeRect(x + 0.5, y + 0.5, cw - 1, chh - 1);
        gs.fillStyle = (!cat && (v - lo) / (hi - lo || 1) > 0.55) ? '#000' : '#fff'; gs.font = '600 14px ui-monospace,Menlo,Consolas,monospace'; gs.textAlign = 'center';
        gs.fillText(String(v), x + cw / 2, y + chh / 2 + 5);
      }
      gs.textAlign = 'left';
      here.taps.forEach((t) => {
        gs.strokeStyle = C.red; gs.lineWidth = 3; gs.strokeRect((t.c * cw) + 3, (t.r * chh) + 3, cw - 6, chh - 6);
      });
      gs.lineWidth = 1;
      const sxp = (pos(q) + 0.5) * cw, syp = (pos(r) + 0.5) * chh;
      gs.fillStyle = C.orange; gs.beginPath(); gs.arc(AI.clamp(sxp, 2, sw - 2), AI.clamp(syp, 2, sh - 2), 5, 0, 7); gs.fill();
      // output
      const go = cO.ctx, ow = cO.w, oh = cO.h, oc = Math.min(ow, oh) / m, ox = (ow - oc * m) / 2, oy = 6;
      go.clearRect(0, 0, ow, oh);
      for (let rr = 0; rr < m; rr++) for (let cc = 0; cc < m; cc++) {
        const v = eval2(pos(cc), pos(rr)).v, x = ox + cc * oc, y = oy + rr * oc, isInt = Math.abs(v - Math.round(v)) < 1e-9;
        go.fillStyle = cat ? PAL[Math.round(v) % PAL.length] : ramp((v - lo) / (hi - lo || 1));
        go.fillRect(x, y, oc, oc);
        if (cat && !isInt) { go.strokeStyle = C.red; go.lineWidth = 2; go.beginPath(); go.moveTo(x, y + oc); go.lineTo(x + oc, y); go.stroke(); }
        go.lineWidth = 1;
        const isCur = rr * m + cc === cur;
        go.strokeStyle = isCur ? C.orange : C.border; go.lineWidth = isCur ? 3 : 1;
        go.strokeRect(x + 1.5, y + 1.5, oc - 3, oc - 3); go.lineWidth = 1;
        go.fillStyle = (!cat && (v - lo) / (hi - lo || 1) > 0.55) ? '#000' : '#fff'; go.font = `${Math.max(9, Math.min(14, oc / 3.2))}px ui-monospace,Menlo,Consolas,monospace`; go.textAlign = 'center';
        go.fillText(Number.isInteger(v) ? String(v) : v.toFixed(2), x + oc / 2, y + oc / 2 + 4);
      }
      go.textAlign = 'left';
      const wtxt = (arr) => arr.map((t) => `${t.i}:${(t.w * 100).toFixed(1)}%`).join('  ');
      const isClass = Math.abs(here.v - Math.round(here.v)) < 1e-9;
      ro.textContent = [
        `output pixel (row ${r}, col ${q}) of ${m} × ${m}`,
        `source position x = ${pos(q).toFixed(3)}, y = ${pos(r).toFixed(3)}  (3 × 3 source; edges clamped)`,
        `${method}: x weights ${wtxt(here.wx)}   y weights ${wtxt(here.wy)}`,
        `value = Σ wx·wy·v = ${here.v.toFixed(3)}${cat ? (isClass ? '  (a valid class)' : '  NOT a class: blended labels are invented') : ''}`,
      ].join('\n');
      cap.textContent = method === 'nearest'
        ? 'Nearest copies one source value: no new numbers appear, so it is the right choice for class labels.'
        : `${method} blends ${here.taps.length} neighbours with weights that sum to 1. ${cat ? 'For labels this invents class values between two classes.' : 'For reflectance this gives smooth, plausible values.'}`;
    }, { fps: 30 });
    function M() { return Math.min(8, Math.max(2, NUM(root, 'M') || 5)); }
  };

  /* ================= 6. Cloud mask with a quality band ================= */
  AI.labs.cloudMask = function (root) {
    const ctl = `<div class="row">${sel('seed', 'Scene', [[7, 'scene A'], [21, 'scene B'], [42, 'scene C']], 7)}</div>
      <div class="row"><label style="width:100%">Cloud threshold: masked where probability ≥ <b data-o="thr">50</b> %<input type="range" min="0" max="100" step="1" value="50" data-k="thr" style="width:100%"></label></div>
      <div class="row">${BTN('sweep', 'Sweep threshold 0 → 100 → 0')}</div>${TX}${RO}`;
    const { view, ctl: c } = AI.labShell(root, 'Masking clouds with a quality band',
      'a band of reflectance and a band of cloud probability; the threshold decides what is usable',
      '<canvas data-c="band" style="height:200px"></canvas><div class="row" style="gap:10px;margin-top:10px"><canvas data-c="prob" style="height:200px;flex:1 1 0"></canvas><canvas data-c="mask" style="height:200px;flex:1 1 0"></canvas></div>' + CAP, ctl);
    const cB = AI.canvas(view.querySelector('[data-c=band]')), cP = AI.canvas(view.querySelector('[data-c=prob]')), cM = AI.canvas(view.querySelector('[data-c=mask]'));
    const cap = view.querySelector('.geo-cap'), ro = root.querySelector('.geo-ro'), thrEl = root.querySelector('[data-k=thr]'), thrOut = root.querySelector('[data-o=thr]');
    const st = tx(root);
    const NN = 12;
    let scene = null, seedUsed = -1, sweeping = false, sv = 50, dir = 1, dirty = true;
    const gen = (seed) => {
      const r = AI.rng(seed), blobs = [0, 1, 2].map(() => ({ x: 1.5 + 9 * r(), y: 1.5 + 9 * r(), s: 0.9 + 1.1 * r() }));
      const band = [], prob = [], nd = [];
      for (let y = 0; y < NN; y++) for (let x = 0; x < NN; x++) {
        let p = 0; for (const b of blobs) p = Math.max(p, 100 * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.s * b.s)));
        p = Math.min(100, Math.round(p + Math.floor(r() * 6)));
        const base = 1100 + Math.floor(r() * 150);
        const isND = x === NN - 1 && y >= NN - 3;
        band.push(isND ? 0 : base + Math.round((p / 100) * 1700));
        prob.push(isND ? 0 : p); nd.push(isND);
      }
      return { band, prob, nd };
    };
    const stats = (t) => {
      let cloud = 0, nod = 0, sumU = 0, nU = 0, sumAll = 0, nAll = 0;
      for (let i = 0; i < NN * NN; i++) {
        if (scene.nd[i]) { nod++; continue; }
        sumAll += scene.band[i]; nAll++;
        if (scene.prob[i] >= t) { cloud++; continue; }
        sumU += scene.band[i]; nU++;
      }
      return { cloud, nod, usable: nU, meanU: nU ? sumU / nU : NaN, meanAll: nAll ? sumAll / nAll : NaN, total: NN * NN };
    };
    wire(root, () => { dirty = true; }, (a) => { if (a === 'sweep') { sweeping = !sweeping; sweepBtn(); } });
    const sweepBtn = () => { root.querySelector('[data-a=sweep]').textContent = sweeping ? 'Stop sweep' : 'Sweep threshold 0 → 100 → 0'; };
    thrEl.addEventListener('input', () => { sv = +thrEl.value; });
    cB.onresize = cP.onresize = cM.onresize = () => (dirty = true);
    const heat = (p) => { const t = p / 100; return `rgb(${Math.round(40 + 215 * t)},${Math.round(70 + 60 * (1 - Math.abs(t - 0.5) * 2))},${Math.round(220 - 180 * t)})`; };
    AI.loop(root, (dt) => {
      if (sweeping && st.playing) { sv += dir * 18 * dt * st.speed; if (sv >= 100) { sv = 100; dir = -1; } if (sv <= 0) { sv = 0; dir = 1; } thrEl.value = Math.round(sv); dirty = true; }
      if (st.step) { sv = sv >= 100 ? 0 : Math.min(100, Math.round(sv) + 5); thrEl.value = sv; st.step = false; dirty = true; }
      const seed = +root.querySelector('[data-k=seed]').value;
      if (seed !== seedUsed) { scene = gen(seed); seedUsed = seed; dirty = true; }
      if (!dirty) return;
      dirty = false;
      const t = Math.round(sv), C = AI.col();
      thrEl.value = t; thrOut.textContent = t;
      const s = stats(t);
      const cw = cB.w / NN, ch = cB.h / NN, gb = cB.ctx;
      const valid = scene.band.filter((_, i) => !scene.nd[i]);
      const bl = Math.min(...valid), bh = Math.max(...valid);
      gb.clearRect(0, 0, cB.w, cB.h);
      for (let y = 0; y < NN; y++) for (let x = 0; x < NN; x++) {
        const i = y * NN + x;
        gb.fillStyle = scene.nd[i] ? '#000' : `rgb(${Math.round(255 * (scene.band[i] - bl) / (bh - bl || 1))},${Math.round(255 * (scene.band[i] - bl) / (bh - bl || 1))},${Math.round(255 * (scene.band[i] - bl) / (bh - bl || 1))})`;
        gb.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
      }
      const gp = cP.ctx, pw = cP.w / NN, ph = cP.h / NN;
      gp.clearRect(0, 0, cP.w, cP.h);
      for (let y = 0; y < NN; y++) for (let x = 0; x < NN; x++) { const i = y * NN + x; gp.fillStyle = scene.nd[i] ? '#000' : heat(scene.prob[i]); gp.fillRect(x * pw, y * ph, pw + 0.5, ph + 0.5); }
      const gm = cM.ctx, mw = cM.w / NN, mh = cM.h / NN;
      gm.clearRect(0, 0, cM.w, cM.h);
      for (let y = 0; y < NN; y++) for (let x = 0; x < NN; x++) {
        const i = y * NN + x, px = x * mw, py = y * mh;
        if (scene.nd[i]) { gm.fillStyle = '#000'; gm.fillRect(px, py, mw, mh); continue; }
        if (scene.prob[i] >= t) { gm.fillStyle = C.red; gm.globalAlpha = 0.75; gm.fillRect(px, py, mw, mh); gm.globalAlpha = 1; gm.strokeStyle = C.red; gm.beginPath(); gm.moveTo(px, py + mh); gm.lineTo(px + mw, py); gm.stroke(); continue; }
        const v = Math.round(255 * (scene.band[i] - bl) / (bh - bl || 1));
        gm.fillStyle = `rgb(${v},${v},${v})`; gm.fillRect(px, py, mw, mh);
      }
      const pct = (n) => (100 * n / s.total).toFixed(1);
      ro.textContent = [
        `threshold t = ${t} %   masked: probability ≥ ${t}`,
        `cloudy pixels: ${s.cloud} of ${s.total} = ${pct(s.cloud)} %`,
        `nodata pixels (band = 0): ${s.nod}`,
        `usable pixels: ${s.usable} of ${s.total} = ${pct(s.usable)} %`,
        `mean reflectance, usable pixels only: ${isNaN(s.meanU) ? '–' : s.meanU.toFixed(1)}`,
        `mean reflectance, all valid pixels (no mask): ${isNaN(s.meanAll) ? '–' : s.meanAll.toFixed(1)}`,
      ].join('\n');
      cap.textContent = t >= 90 ? 'Only very confident clouds are masked: thin or medium clouds stay in the data and bias the statistics.'
        : t <= 10 ? 'A low threshold masks almost everything bright, including clear pixels with a little cloud risk. You lose data.'
        : 'Lower threshold: more pixels masked. Higher threshold: fewer masked, more cloud left behind. Choose it on labelled data.';
    }, { fps: 30 });
  };
})();
