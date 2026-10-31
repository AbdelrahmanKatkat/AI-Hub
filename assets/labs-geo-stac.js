/* GeoAI · Module 2 (STAC) labs. Synthetic data only; teaching simplifications are stated in each lab.
 * stac-tree      <div data-lab="stac-tree"></div>      catalogue opens step by step
 * stac-links     <div data-lab="stac-links"></div>     follow rel links through a catalogue
 * stac-item      <div data-lab="stac-item"></div>      Item anatomy: fields, footprint vs bbox
 * stac-ext       <div data-lab="stac-ext"></div>       which fields an extension declares
 * stac-search    <div data-lab="stac-search"></div>    /search by bbox, date and cloud cover
 * stac-harvest   <div data-lab="stac-harvest"></div>   select assets, filter, build a time stack
 * stac-validate  <div data-lab="stac-validate"></div>  edit an Item, watch the checks run
 */
(function () {
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const rrect = (g, x, y, w, h, r) => { g.beginPath(); if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h); };
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const KIND_COL = { Catalog: 'purple', Collection: 'blue', Item: 'green', Asset: 'orange' };

  /* speed slider: returns a getter for the current multiplier */
  const speedRow = (ctl) => {
    const l = document.createElement('label');
    l.innerHTML = '<span>Speed <span class="v"></span></span><input type="range" min="0.25" max="2" step="0.05" value="1">';
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const upd = () => (v.textContent = (+inp.value).toFixed(2) + '×');
    inp.oninput = upd; upd(); ctl.appendChild(l);
    return () => +inp.value;
  };
  const buttons = (ctl, labels) => {
    const d = document.createElement('div'); d.className = 'row';
    d.innerHTML = labels.map((t) => `<button class="btn small">${t}</button>`).join('');
    ctl.appendChild(d);
    return [...d.querySelectorAll('button')];
  };
  const readoutEl = (ctl) => { const d = document.createElement('div'); d.className = 'readout'; ctl.appendChild(d); return d; };

  /* ---------- 1 · catalogue tree, opened step by step ---------- */
  AI.labs['stac-tree'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Browse a STAC catalogue', 'catalog → collection → item → asset, one link at a time',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const NODES = [
      { kind: 'Catalog', name: 'demo-earth-archive', lv: 0, row: 1.5, p: -1,
        json: '{ "type": "Catalog", "id": "demo-earth-archive",\n  "links": [ {"rel": "child", "href": "./demo-sentinel-2-l2a/collection.json"},\n            {"rel": "child", "href": "./demo-landsat-l2/collection.json"} ] }' },
      { kind: 'Collection', name: 'demo-sentinel-2-l2a', lv: 1, row: 0, p: 0,
        json: '{ "type": "Collection", "id": "demo-sentinel-2-l2a",\n  "license": "CC-BY-4.0",\n  "extent": { "spatial": { "bbox": [[-180, -56, 180, 84]] } } }' },
      { kind: 'Collection', name: 'demo-landsat-l2', lv: 1, row: 3, p: 0, json: '(not opened in this walk)' },
      { kind: 'Item', name: 'S2B_DEMO_20260714_T31UFT_L2A', lv: 2, row: 0, p: 1,
        json: '{ "type": "Feature", "id": "S2B_DEMO_20260714_T31UFT_L2A",\n  "bbox": [10.6, 48.2, 11.9, 49.2],\n  "properties": { "datetime": "2026-07-14T10:31:22Z" } }' },
      { kind: 'Item', name: 'S2A_DEMO_20260701_T31UFT_L2A', lv: 2, row: 1, p: 1, json: '(not opened in this walk)' },
      { kind: 'Asset', name: 'B04.tif · role data', lv: 3, row: 0, p: 3,
        json: '"B04": { "href": "./B04.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "roles": ["data"] }' },
      { kind: 'Asset', name: 'TCI.tif · role visual', lv: 3, row: 1, p: 3,
        json: '"visual": { "href": "./TCI.tif", "roles": ["visual"] }' },
    ];
    const STEPS = [0, 1, 3, 5, 6];
    const CAPS = [
      'Step 1. Open the root catalogue. A catalogue has no footprint of its own; it lists two child links.',
      'Step 2. Follow the child link to the Sentinel-2 collection. Its licence and extent apply to every item below it.',
      'Step 3. Open one item from the collection’s item links. Its geometry, bbox and datetime are read here. The second item (dashed) is not opened.',
      'Step 4. Open asset B04. Its role is data, so this is the file a measurement tool downloads.',
      'Step 5. Open the visual asset. Its role is visual: a preview for people, never used for measurement. The Landsat collection (dashed) was never opened, so it cost nothing.',
    ];
    const btns = buttons(ctl, ['⏸ Pause', 'Step ▸', 'Reset']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const TRAVEL = 1.2, STEP_T = 3.2;
    let cur = -1, clock = 0, playing = true, pulse = 0, boxes = [];
    const stepIdx = (i) => STEPS.indexOf(i);
    const sync = () => (btns[0].textContent = playing ? '⏸ Pause' : '▶ Play');
    const setStep = (k) => { cur = k; clock = 0; cap.textContent = CAPS[k]; out.textContent = NODES[STEPS[k]].json; };
    cap.textContent = 'Press play: the browser starts at the root catalogue.';
    out.textContent = 'nothing opened yet';
    btns[0].onclick = () => { if (!playing && cur === STEPS.length - 1) cur = -1; playing = !playing; sync(); };
    btns[1].onclick = () => { playing = false; sync(); if (cur < STEPS.length - 1) setStep(cur + 1); };
    btns[2].onclick = () => { cur = -1; clock = 0; playing = true; sync(); cap.textContent = 'Press play: the browser starts at the root catalogue.'; out.textContent = 'nothing opened yet'; };
    c.cv.addEventListener('click', (e) => {
      const r = c.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const i = boxes.findIndex((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
      if (i < 0 || stepIdx(i) < 0) return;
      playing = false; sync(); setStep(stepIdx(i));
    });

    const arriving = (i) => cur >= 0 && STEPS[cur] === i && clock < TRAVEL && NODES[i].p >= 0;
    const opened = (i) => { const k = stepIdx(i); return k >= 0 && k <= cur && !arriving(i); };
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, bw = Math.min(200, W * 0.27), bh = 44;
      g.clearRect(0, 0, W, H);
      const X = (lv) => 8 + lv * ((W - bw - 16) / 3);
      const Y = (row) => 14 + row * ((H - bh - 28) / 3);
      boxes = NODES.map((n) => ({ x: X(n.lv), y: Y(n.row), w: bw, h: bh }));
      NODES.forEach((n, i) => {
        if (n.p < 0) return;
        const a = boxes[n.p], b = boxes[i], on = opened(i);
        g.strokeStyle = on ? C[KIND_COL[n.kind]] : C.border; g.lineWidth = on ? 2.5 : 1;
        g.setLineDash(on ? [] : [5, 5]);
        g.beginPath(); g.moveTo(a.x + bw, a.y + bh / 2); g.lineTo(b.x, b.y + bh / 2); g.stroke();
      });
      g.setLineDash([]);
      if (cur >= 0 && arriving(STEPS[cur])) {
        const i = STEPS[cur], a = boxes[NODES[i].p], b = boxes[i], k = AI.smooth(clock / TRAVEL);
        g.fillStyle = C[KIND_COL[NODES[i].kind]]; g.shadowColor = g.fillStyle; g.shadowBlur = 12;
        g.beginPath(); g.arc(AI.lerp(a.x + bw, b.x, k), AI.lerp(a.y + bh / 2, b.y + bh / 2, k), 6, 0, 7); g.fill();
        g.shadowBlur = 0;
      }
      NODES.forEach((n, i) => {
        const p = boxes[i], on = opened(i), col = C[KIND_COL[n.kind]], focus = cur >= 0 && STEPS[cur] === i && !arriving(i);
        g.save();
        g.fillStyle = on ? AI.mix(C.bg2, col, 0.28) : C.bg;
        if (focus) { g.shadowColor = col; g.shadowBlur = 10 + 6 * Math.sin(pulse * 2.4); }
        rrect(g, p.x, p.y, p.w, p.h, 8); g.fill();
        g.shadowBlur = 0; g.strokeStyle = on ? col : C.border; g.lineWidth = focus ? 2.5 : 1;
        g.setLineDash(on ? [] : [4, 3]); rrect(g, p.x, p.y, p.w, p.h, 8); g.stroke();
        g.restore();
        g.font = `bold 11px ${MONO}`; g.fillStyle = on ? col : C.muted;
        g.fillText(n.kind, p.x + 10, p.y + 16);
        g.font = `11px ${MONO}`; g.fillStyle = on ? C.fg : C.muted;
        g.fillText(n.name.length > 26 ? n.name.slice(0, 25) + '…' : n.name, p.x + 10, p.y + 33);
      });
    };
    AI.loop(root, (dt) => {
      pulse += dt;
      if (playing) {
        if (cur < 0) setStep(0);
        else {
          clock += dt * speed();
          if (clock >= STEP_T) {
            if (cur < STEPS.length - 1) setStep(cur + 1);
            else { playing = false; sync(); }
          }
        }
      }
      draw();
    }, { fps: 30 });
  };

  /* ---------- 2 · link walker ---------- */
  AI.labs['stac-links'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Follow the links', 'every document lists its neighbours; a client moves by choosing a rel and following an href',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const DOC = {
      root: { kind: 'Catalog', name: 'demo-earth-archive', x: 0.5, y: 0, links: [
        ['child', 'col-s2', './demo-sentinel-2-l2a/collection.json'], ['child', 'col-ls', './demo-landsat-l2/collection.json']] },
      'col-s2': { kind: 'Collection', name: 'demo-sentinel-2-l2a', x: 0.2, y: 0.5, links: [
        ['root', 'root', '../catalog.json'], ['parent', 'root', '../catalog.json'],
        ['item', 'it-s2b', './items/S2B_DEMO_20260714_T31UFT_L2A.json'], ['item', 'it-s2a', './items/S2A_DEMO_20260701_T31UFT_L2A.json']] },
      'col-ls': { kind: 'Collection', name: 'demo-landsat-l2', x: 0.8, y: 0.5, links: [
        ['root', 'root', '../catalog.json'], ['parent', 'root', '../catalog.json'], ['item', 'it-ls1', './items/LC09_DEMO_20260708_L2.json']] },
      'it-s2b': { kind: 'Item', name: 'S2B_DEMO_20260714_T31UFT_L2A', x: 0.02, y: 1, links: [
        ['root', 'root', '../../catalog.json'], ['parent', 'col-s2', '../collection.json'], ['collection', 'col-s2', '../collection.json']] },
      'it-s2a': { kind: 'Item', name: 'S2A_DEMO_20260701_T31UFT_L2A', x: 0.36, y: 1, links: [
        ['root', 'root', '../../catalog.json'], ['parent', 'col-s2', '../collection.json'], ['collection', 'col-s2', '../collection.json']] },
      'it-ls1': { kind: 'Item', name: 'LC09_DEMO_20260708_L2', x: 0.8, y: 1, links: [
        ['root', 'root', '../../catalog.json'], ['parent', 'col-ls', '../collection.json'], ['collection', 'col-ls', '../collection.json']] },
    };
    const REL_TXT = {
      child: 'child: a document one level down. The parent knows only its address, not its contents.',
      item: 'item: one observation in this collection, a GeoJSON feature with its own footprint and time.',
      parent: 'parent: one level up. A crawler can climb out of a deep item without starting again at the root.',
      root: 'root: the top of the tree. Any document reaches it in one link.',
      collection: 'collection: the collection this item belongs to, which holds the licence and the providers.',
      back: 'back: the document you were on before (your own history, not a STAC link).',
    };
    const group = (rel) => (rel === 'child' || rel === 'item' ? rel : 'up');
    const filterBtns = buttons(ctl, ['All', 'child', 'item', 'up']);
    const linkBox = document.createElement('div'); linkBox.className = 'row'; ctl.appendChild(linkBox);
    const navBtns = buttons(ctl, ['◂ Back', 'Reset']);
    const trail = document.createElement('div'); trail.className = 'caption'; ctl.appendChild(trail);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const TRAVEL = 1.2;
    let here = 'root', hist = ['root'], move = null, filt = 'all', pulse = 0;
    const P = (d, W, H, nw, nh) => ({ x: 8 + d.x * (W - nw - 16), y: 8 + d.y * (H - nh - 16) });

    const setOut = () => (out.textContent = JSON.stringify({ type: DOC[here].kind, id: DOC[here].name,
      links: DOC[here].links.map(([rel, , href]) => ({ rel, href })) }, null, 1));
    const renderLinks = () => {
      linkBox.innerHTML = '';
      DOC[here].links.forEach(([rel, to]) => {
        const b = document.createElement('button'); b.className = 'btn small';
        b.textContent = `${rel} → ${DOC[to].name}`;
        b.style.display = filt === 'all' || group(rel) === filt ? '' : 'none';
        b.disabled = !!move;
        b.onclick = () => goTo(to, rel, true);
        linkBox.appendChild(b);
      });
      navBtns[0].disabled = !!move || hist.length < 2;
      trail.textContent = 'history: ' + hist.map((id) => DOC[id].name).join(' › ');
    };
    const goTo = (to, rel, push) => {
      if (move) return;
      move = { from: { x: DOC[here].x, y: DOC[here].y }, to: { x: DOC[to].x, y: DOC[to].y }, t: 0, dst: to, rel, push };
      cap.textContent = `Following rel = ${rel} to ${DOC[to].name}…`;
      renderLinks();
    };
    const arrive = () => {
      const m = move; move = null; here = m.dst;
      if (m.push) hist.push(here);
      cap.textContent = `You are at the ${DOC[here].kind.toLowerCase()} ${DOC[here].name}. ${REL_TXT[m.rel] || ''}`;
      setOut(); renderLinks();
    };
    filterBtns.forEach((b, i) => (b.onclick = () => {
      filt = ['all', 'child', 'item', 'up'][i];
      filterBtns.forEach((x, j) => x.classList.toggle('on', j === i));
      renderLinks();
    }));
    filterBtns[0].classList.add('on');
    navBtns[0].onclick = () => {
      if (move || hist.length < 2) return;
      hist.pop(); const prev = hist[hist.length - 1];
      move = { from: { x: DOC[here].x, y: DOC[here].y }, to: { x: DOC[prev].x, y: DOC[prev].y }, t: 0, dst: prev, rel: 'back', push: false };
      cap.textContent = 'Going back to ' + DOC[prev].name + '…';
      renderLinks();
    };
    navBtns[1].onclick = () => {
      here = 'root'; hist = ['root']; move = null;
      cap.textContent = 'Start at the root catalogue. Choose a link to move.';
      setOut(); renderLinks();
    };
    cap.textContent = 'Start at the root catalogue. Choose a link to move.';
    setOut(); renderLinks();

    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, nw = Math.min(160, W * 0.25), nh = 40;
      g.clearRect(0, 0, W, H);
      const box = {};
      Object.keys(DOC).forEach((id) => (box[id] = P(DOC[id], W, H, nw, nh)));
      const ctr = (p) => ({ x: p.x + nw / 2, y: p.y + nh / 2 });
      Object.keys(DOC).forEach((id) => DOC[id].links.forEach(([rel, to]) => {
        const a = ctr(box[id]), b = ctr(box[to]), outgoing = id === here && !move && (filt === 'all' || group(rel) === filt);
        g.strokeStyle = outgoing ? C.blue : C.border; g.lineWidth = outgoing ? 2.5 : 1;
        g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
        if (outgoing) {
          g.font = `10px ${MONO}`; g.fillStyle = C.blue;
          g.fillText(rel, (a.x + b.x) / 2 + 4, (a.y + b.y) / 2 - 4);
        }
      }));
      Object.keys(DOC).forEach((id) => {
        const p = box[id], col = C[KIND_COL[DOC[id].kind]], isHere = id === here && !move;
        g.save();
        g.fillStyle = isHere ? AI.mix(C.bg2, col, 0.35) : C.bg2;
        g.strokeStyle = isHere ? col : C.border; g.lineWidth = isHere ? 2.5 : 1;
        rrect(g, p.x, p.y, nw, nh, 8); g.fill(); g.stroke();
        g.restore();
        g.font = `bold 10px ${MONO}`; g.fillStyle = col; g.fillText(DOC[id].kind, p.x + 8, p.y + 14);
        g.font = `10px ${MONO}`; g.fillStyle = C.fg;
        const nm = DOC[id].name; g.fillText(nm.length > 24 ? nm.slice(0, 23) + '…' : nm, p.x + 8, p.y + 30);
        if (hist.includes(id)) { g.fillStyle = C.green; g.beginPath(); g.arc(p.x + nw - 8, p.y + 8, 3.5, 0, 7); g.fill(); }
      });
      let mx, my;
      if (move) {
        const k = AI.smooth(move.t / TRAVEL), fx = AI.lerp(move.from.x, move.to.x, k), fy = AI.lerp(move.from.y, move.to.y, k);
        const p = P({ x: fx, y: fy }, W, H, nw, nh); mx = p.x + nw / 2; my = p.y + nh / 2;
      } else { const p = ctr(box[here]); mx = p.x; my = p.y; }
      g.fillStyle = C.blue; g.shadowColor = C.blue; g.shadowBlur = 8 + 4 * Math.sin(pulse * 3);
      g.beginPath(); g.arc(mx + nw / 2 - 12, my - nh / 2 + 4, 6, 0, 7); g.fill(); g.shadowBlur = 0;
      g.font = `9px ${MONO}`; g.fillStyle = C.blue; g.fillText('you', mx + nw / 2 - 20, my - nh / 2 - 4);
    };
    AI.loop(root, (dt) => {
      pulse += dt;
      if (move) { move.t += dt * speed(); if (move.t >= TRAVEL) arrive(); }
      draw();
    }, { fps: 30 });
  };

  /* ---------- 3 · Item anatomy: fields, footprint and bbox ---------- */
  AI.labs['stac-item'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Read an Item, field by field', 'click a field: its meaning appears, and the map shows what it describes',
      '<canvas style="height:260px"></canvas><div class="sj" style="margin-top:10px"></div><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view)), list = AI.$('.sj', view);
    const CX = 11.25, CY = 48.7, HW = 0.65, HH = 0.5; // the footprint is a 1.3 x 1.0 degree rectangle at the start
    const ring = (deg) => {
      const t = (deg * Math.PI) / 180, co = Math.cos(t), si = Math.sin(t);
      const p = [[-HW, -HH], [HW, -HH], [HW, HH], [-HW, HH]].map(([x, y]) => [CX + x * co - y * si, CY + x * si + y * co]);
      return [...p, p[0]];
    };
    const bboxOf = (r) => { const xs = r.map((p) => p[0]), ys = r.map((p) => p[1]); return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; };
    const EXP = {
      type: '"type": "Feature" tells a GeoJSON reader that this record is a map object with a geometry and properties. The STAC-specific parts sit beside it.',
      stac_version: '"stac_version" is the version of the core STAC rules this record follows. Validators choose their rules from it.',
      id: '"id" is a name unique inside the collection. Other records, such as a training manifest, refer to the item by this id.',
      geometry: 'The footprint: a ring of [longitude, latitude] vertices in WGS 84 degrees; the last vertex repeats the first. Drag the tilt slider to rotate the scene on the ground and watch the vertices change.',
      bbox: 'The box around the geometry, [minx, miny, maxx, maxy]. Searches test it first because it is cheap. Edit one number to see when the stored box stops matching the footprint.',
      datetime: '"properties.datetime" is the capture time in UTC (the trailing Z). Time search compares against this value; the footprint does not affect it.',
      cloud: '"eo:cloud_cover" describes the whole scene, not your field. The prefix eo: says it comes from the eo extension (Lesson 2.4).',
      assets: 'Each asset is one file. Its href is relative to the item file. The map shows the footprint these files cover; a download fetches the files themselves.',
    };
    const ROWS = [
      ['type', () => '"type": "Feature"'],
      ['stac_version', () => '"stac_version": "1.0.0"'],
      ['id', () => '"id": "S2B_DEMO_20260714_T31UFT_L2A"'],
      ['geometry', () => `"geometry": ${JSON.stringify({ type: 'Polygon', coordinates: [ringR(shownTarget())] })}`],
      ['bbox', () => `"bbox": ${JSON.stringify(stored)}`],
      ['datetime', () => '"properties.datetime": "2026-07-14T10:31:22Z"'],
      ['cloud', () => '"properties.eo:cloud_cover": 12.4'],
      ['assets', () => '"assets.B04.href": "./B04.tif"'],
    ];
    const rowEls = ROWS.map(([k]) => {
      const d = document.createElement('div'); d.dataset.k = k;
      d.style.cssText = `font-family:${MONO};font-size:12px;padding:4px 8px;border-radius:6px;cursor:pointer;white-space:pre-wrap;word-break:break-all;border:1px solid transparent`;
      list.appendChild(d); return d;
    });
    const ringR = (deg) => ring(deg).map(([x, y]) => [r3(x), r3(y)]);
    const shownTarget = () => target;

    /* controls */
    const l = document.createElement('label');
    l.innerHTML = '<span>Tilt of the scene <span class="v"></span></span><input type="range" min="0" max="45" step="1" value="0">';
    const tiltIn = l.querySelector('input'), tiltV = l.querySelector('.v');
    ctl.appendChild(l);
    const rb = buttons(ctl, ['Recompute bbox from geometry']);
    const inputs = ['west (minx)', 'south (miny)', 'east (maxx)', 'north (maxy)'].map((name) => {
      const lab = document.createElement('label');
      lab.innerHTML = `<span>${name}</span><input type="number" step="0.001">`;
      ctl.appendChild(lab); return lab.querySelector('input');
    });
    const status = document.createElement('div'); status.className = 'caption'; ctl.appendChild(status);
    const ratio = readoutEl(ctl);

    let target = 0, shown = 0, auto = true, stored = [10.6, 48.2, 11.9, 49.2], selected = 'geometry';
    const setInputs = () => inputs.forEach((el, i) => (el.value = stored[i].toFixed(3)));
    const computed = () => bboxOf(ring(target)).map(r3);
    const refresh = () => {
      const comp = computed();
      const same = stored.every((v, i) => Math.abs(v - comp[i]) < 0.0005);
      const contains = stored[0] <= comp[0] && stored[1] <= comp[1] && stored[2] >= comp[2] && stored[3] >= comp[3];
      if (stored[0] > stored[2] || stored[1] > stored[3]) status.textContent = '✗ min and max are swapped: check the order [minx, miny, maxx, maxy].';
      else if (same) status.textContent = '✓ the stored bbox equals the box around the geometry.';
      else if (contains) status.textContent = 'The stored bbox contains the geometry but is larger than needed. It is valid, just not the tightest box.';
      else status.textContent = '✗ the stored bbox does not contain the geometry: a search could miss this scene.';
      status.style.color = same || contains ? '' : 'var(--red)';
      const area = (stored[2] - stored[0]) * (stored[3] - stored[1]), fp = (2 * HW) * (2 * HH);
      ratio.textContent = `footprint area   ${fp.toFixed(3)} deg²\nstored bbox area ${area.toFixed(3)} deg²\nbbox ÷ footprint  ${(area / fp).toFixed(2)}   (degrees², not m²)`;
      rowEls.forEach((el, i) => { el.textContent = ROWS[i][1](); });
    };
    const select = (k) => {
      selected = k;
      rowEls.forEach((el) => { const on = el.dataset.k === k; el.style.background = on ? 'var(--bg3)' : ''; el.style.borderColor = on ? 'var(--blue)' : 'transparent'; });
      cap.textContent = EXP[k];
    };
    rowEls.forEach((el) => (el.onclick = () => select(el.dataset.k)));
    tiltIn.oninput = () => {
      target = +tiltIn.value; tiltV.textContent = target + '°';
      if (auto) { stored = computed(); setInputs(); }
      refresh();
    };
    inputs.forEach((el, i) => (el.oninput = () => { auto = false; stored[i] = +el.value; refresh(); }));
    rb[0].onclick = () => { auto = true; stored = computed(); setInputs(); refresh(); };
    tiltV.textContent = '0°'; setInputs(); refresh(); select('geometry');

    const draw = () => {
      const C = AI.col(), g = c.ctx, v = AI.view(c, 10.0, 12.5, 47.9, 49.5);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      const r = ring(shown), comp = bboxOf(r);
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      // bbox
      const bx = auto ? comp : stored; // follows the geometry until the user edits the numbers
      const hi = selected === 'bbox';
      g.setLineDash([6, 4]); g.strokeStyle = hi ? C.orange : C.muted; g.lineWidth = hi ? 2.5 : 1.5;
      g.strokeRect(v.X(bx[0]), v.Y(bx[3]), v.X(bx[2]) - v.X(bx[0]), v.Y(bx[1]) - v.Y(bx[3]));
      g.setLineDash([]);
      // footprint
      const hg = selected === 'geometry';
      g.beginPath();
      r.forEach(([x, y], i) => (i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y))));
      g.closePath();
      g.fillStyle = AI.mix(C.bg2, C.blue, hg ? 0.55 : 0.35); g.fill();
      g.strokeStyle = C.blue; g.lineWidth = hg ? 3 : 2; g.stroke();
      if (hg) r.slice(0, 4).forEach(([x, y]) => { g.fillStyle = C.blue; g.beginPath(); g.arc(v.X(x), v.Y(y), 3.5, 0, 7); g.fill(); });
      g.restore();
      g.fillStyle = C.muted; g.font = `11px ${MONO}`;
      g.fillText('footprint (geometry)', v.X(10.05), v.Y(49.42));
      g.fillStyle = hi ? C.orange : C.muted; g.fillText('bbox (dashed)', v.X(10.05), v.Y(49.25));
      g.fillStyle = C.muted; g.fillText('drawn in plain lon/lat, so shapes look stretched at this latitude', v.pad.l, c.h - 6);
    };
    let last = '';
    AI.loop(root, (dt) => {
      shown += (target - shown) * Math.min(1, dt * 3);
      if (Math.abs(target - shown) < 0.01) shown = target;
      const key = shown.toFixed(3) + selected + stored.join() + auto;
      if (key === last) return;
      last = key; draw();
    });
    select('geometry');
  };

  /* ---------- 4 · extensions: which fields a reader can use ---------- */
  AI.labs['stac-ext'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Extensions: switch the declarations on and off', 'the same record; what a reader can interpret depends on the schemas it declares',
      '<div class="sj"></div><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), list = AI.$('.sj', view);
    const EXT = {
      eo: 'https://stac-extensions.github.io/eo/v1.1.0/schema.json',
      proj: 'https://stac-extensions.github.io/projection/v1.1.0/schema.json',
      raster: 'https://stac-extensions.github.io/raster/v1.1.0/schema.json',
    };
    const ROWS = [
      { k: 'type', t: '"type": "Feature"', ext: null },
      { k: 'stac_version', t: '"stac_version": "1.0.0"', ext: null },
      { k: 'stac_extensions', ext: null, dyn: true },
      { k: 'id', t: '"id": "S2B_DEMO_20260714_T31UFT_L2A"', ext: null },
      { k: 'geometry', t: '"geometry": {"type": "Polygon", "coordinates": [[[10.6, 48.2], …]]}', ext: null },
      { k: 'bbox', t: '"bbox": [10.6, 48.2, 11.9, 49.2]', ext: null },
      { k: 'platform', t: '"properties.platform": "sentinel-2b"', ext: null },
      { k: 'datetime', t: '"properties.datetime": "2026-07-14T10:31:22Z"', ext: null },
      { k: 'gsd', t: '"properties.gsd": 10', ext: null },
      { k: 'eo:cloud_cover', t: '"properties.eo:cloud_cover": 12.4', ext: 'eo' },
      { k: 'proj:epsg', t: '"properties.proj:epsg": 32632', ext: 'proj' },
      { k: 'proj:shape', t: '"properties.proj:shape": [10980, 10980]', ext: 'proj' },
      { k: 'eo:bands', t: '"assets.B04.eo:bands": [{"name": "B04", "common_name": "red", "center_wavelength": 0.665}]', ext: 'eo' },
      { k: 'raster:bands', t: '"assets.B04.raster:bands": [{"data_type": "uint16", "nodata": 0}]', ext: 'raster' },
    ];
    const NOTE = {
      eo: 'eo declares the schema, so eo:cloud_cover and eo:bands are defined and readable. Without it, they are present but no schema defines them.',
      proj: 'proj declares the projection fields. Without it, proj:epsg and proj:shape are undefined, so nothing says which grid the pixels use.',
      raster: 'raster declares the per-band fields. The record already contains raster:bands, so switching raster on gives its definition.',
    };
    const on = { eo: true, proj: true, raster: false };
    const mk = (name, label) => {
      const l = document.createElement('label'); l.style.flexDirection = 'row'; l.style.alignItems = 'center'; l.style.gap = '8px';
      l.innerHTML = `<input type="checkbox" ${on[name] ? 'checked' : ''}><span style="color:var(--fg)">${label}</span>`;
      ctl.appendChild(l); return l.querySelector('input');
    };
    const cb = { eo: mk('eo', 'eo  (cloud cover, bands)'), proj: mk('proj', 'proj  (EPSG, shape)'), raster: mk('raster', 'raster  (per-band data type)') };
    const status = readoutEl(ctl);
    const els = ROWS.map((r) => {
      const d = document.createElement('div'); d.dataset.k = r.k;
      d.style.cssText = `font-family:${MONO};font-size:12px;padding:4px 8px;border-radius:6px;white-space:pre-wrap;word-break:break-all;display:flex;justify-content:space-between;gap:10px`;
      const txt = document.createElement('span'); const tag = document.createElement('span'); tag.style.cssText = 'font-size:11px;white-space:nowrap;align-self:center';
      d.appendChild(txt); d.appendChild(tag); list.appendChild(d);
      return { r, d, txt, tag, a: 1 };
    });
    const extList = () => Object.keys(EXT).filter((k) => on[k]).map((k) => EXT[k]);
    const render = () => {
      els.forEach((e) => {
        if (e.r.dyn) { e.txt.textContent = '"stac_extensions": ' + JSON.stringify(extList()); e.tag.textContent = 'core: lists declared schemas'; e.tag.style.color = 'var(--muted)'; }
        else {
          e.txt.textContent = e.r.t;
          if (!e.r.ext) { e.tag.textContent = 'core'; e.tag.style.color = 'var(--muted)'; }
          else if (on[e.r.ext]) { e.tag.textContent = 'declared: read'; e.tag.style.color = 'var(--green)'; }
          else { e.tag.textContent = 'undeclared: not interpreted'; e.tag.style.color = 'var(--red)'; }
        }
      });
      const core = ROWS.filter((r) => !r.ext).length;
      const read = ROWS.filter((r) => r.ext && on[r.ext]).length;
      const bad = ROWS.filter((r) => r.ext && !on[r.ext]).map((r) => r.k);
      status.textContent = `readable by a core-only reader: ${core} fields\nextension fields declared and read: ${read}\nstrict validator errors: ${bad.length}${bad.length ? ' (' + bad.join(', ') + ')' : ''}`;
    };
    const toggled = (name) => {
      on[name] = cb[name].checked;
      cap.textContent = NOTE[name] + (on[name] ? '' : ' Turning it off changes the record only by declaration: the values stay.');
      render();
    };
    Object.keys(cb).forEach((k) => (cb[k].onchange = () => toggled(k)));
    cap.textContent = 'Turn a box off to remove a schema declaration. The field values stay in the record.';
    render();
    AI.loop(root, (dt) => {
      els.forEach((e) => {
        const target = !e.r.ext || on[e.r.ext] ? 1 : 0.35;
        e.a += (target - e.a) * Math.min(1, dt * 4);
        if (Math.abs(target - e.a) < 0.003) e.a = target;
        e.d.style.opacity = e.a.toFixed(3);
      });
    });
  };

  /* ---------- 5 · /search: bbox, date window, cloud threshold, pages ---------- */
  AI.labs['stac-search'] = function (root) {
    const D0 = Date.UTC(2025, 0, 1), DAY = 864e5, NDAYS = 670, PAGE = 10;
    const dayIso = (d) => new Date(D0 + d * DAY).toISOString().slice(0, 10);
    const { view, ctl } = AI.labShell(root, 'Search a catalogue', 'synthetic scenes (footprints are simplified to rectangles); three filters are applied in order',
      `<canvas style="height:330px;cursor:crosshair;touch-action:none"></canvas><div class="caption"></div><ol class="sr" style="margin:8px 0 0;padding-left:22px;font-family:${MONO};font-size:12px;line-height:1.6;min-height:9em"></ol><div class="pg" style="margin-top:6px"></div>`, '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view)), list = AI.$('.sr', view), pgBox = AI.$('.pg', view);
    const pgBtns = buttons(pgBox, ['◂ Prev page', 'Next page ▸']);
    const pgLabel = document.createElement('span'); pgLabel.style.cssText = 'font-size:12px;color:var(--muted);margin-left:8px'; pgBox.appendChild(pgLabel);

    /* 300 synthetic scenes: 45% clustered over Europe and Africa so that the default box has results */
    const rnd = AI.rng(7), SCENES = [];
    for (let i = 0; i < 300; i++) {
      const eu = rnd() < 0.45;
      const cx = eu ? -10 + rnd() * 50 : -170 + rnd() * 340;
      const cy = eu ? 30 + rnd() * 35 : -50 + rnd() * 120;
      const day = Math.floor(rnd() * NDAYS), sec = Math.floor(rnd() * 86400);
      SCENES.push({ id: 'SYN_' + String(i + 1).padStart(3, '0'), x0: cx - 0.9, x1: cx + 0.9, y0: cy - 0.7, y1: cy + 0.7,
        t: D0 + day * DAY + sec * 1000, cloud: Math.round(rnd() * 1000) / 10, a: 0, m: 0 });
    }

    let bb = [5, 40, 25, 55], sF = 59, eF = 545, thr = 30, page = 0, playing = false, lastKey = '';
    let sDay = 59, eDay = 545;
    const slider = (name, min, max, step, val) => {
      const l = document.createElement('label');
      l.innerHTML = `<span>${name} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
      ctl.appendChild(l); return { inp: l.querySelector('input'), v: l.querySelector('.v') };
    };
    const sIn = slider('start of window', 0, NDAYS - 1, 1, sDay), eIn = slider('end of window', 0, NDAYS - 1, 1, eDay), cIn = slider('cloud cover below', 0, 100, 1, thr);
    const bIn = ['west', 'south', 'east', 'north'].map((name, i) => {
      const l = document.createElement('label'); l.innerHTML = `<span>${name}</span><input type="number" step="0.5">`;
      ctl.appendChild(l); const inp = l.querySelector('input');
      inp.oninput = () => { const v = +inp.value; if (Number.isFinite(v)) { bb[i] = v; lastKey = ''; } };
      return inp;
    });
    const playBtn = buttons(ctl, ['▶ Play time window'])[0];
    const speed = speedRow(ctl);
    const req = readoutEl(ctl), fun = readoutEl(ctl);
    const sync = () => {
      sIn.inp.value = sDay; eIn.inp.value = eDay; cIn.inp.value = thr;
      sIn.v.textContent = dayIso(sDay); eIn.v.textContent = dayIso(eDay); cIn.v.textContent = thr + '%';
    };
    sIn.inp.oninput = () => { sF = sDay = +sIn.inp.value; if (eDay < sDay) eF = eDay = sDay; lastKey = ''; sync(); };
    eIn.inp.oninput = () => { eF = eDay = +eIn.inp.value; if (sDay > eDay) sF = sDay = eDay; lastKey = ''; sync(); };
    cIn.inp.oninput = () => { thr = +cIn.inp.value; lastKey = ''; sync(); };
    playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '⏸ Pause time window' : '▶ Play time window'; };
    pgBtns[0].onclick = () => { page = Math.max(0, page - 1); lastKey = ''; };
    pgBtns[1].onclick = () => { page += 1; lastKey = ''; };

    const viewOf = () => AI.view(c, -180, 180, -60, 84);
    const evalAll = () => {
      const tS = D0 + sDay * DAY, tE = D0 + (eDay + 1) * DAY - 1000;
      let nb = 0, nd = 0; const hit = [];
      SCENES.forEach((s) => {
        const inB = s.x0 <= bb[2] && bb[0] <= s.x1 && s.y0 <= bb[3] && bb[1] <= s.y1;
        const inT = s.t >= tS && s.t <= tE;
        if (inB) nb++;
        if (inB && inT) nd++;
        s.m = inB && inT && s.cloud < thr ? 1 : 0;
        if (s.m) hit.push(s);
      });
      hit.sort((p, q) => p.t - q.t);
      const pages = Math.max(1, Math.ceil(hit.length / PAGE));
      page = Math.min(page, pages - 1);
      list.innerHTML = hit.slice(page * PAGE, page * PAGE + PAGE)
        .map((s) => `<li>${s.id}   ${dayIso(Math.floor((s.t - D0) / DAY))}   cloud ${s.cloud.toFixed(1)}%</li>`).join('')
        || '<li style="color:var(--muted)">no scene passes all three tests: widen the box, the window or the threshold</li>';
      pgLabel.textContent = `page ${page + 1} of ${pages} · ${hit.length} matched`;
      pgBtns[0].disabled = page === 0; pgBtns[1].disabled = page >= pages - 1;
      req.textContent = JSON.stringify({
        collections: ['demo-sentinel-2-l2a'], bbox: bb.map(r3),
        datetime: `${dayIso(sDay)}T00:00:00Z/${dayIso(eDay)}T23:59:59Z`, limit: PAGE,
        'filter-lang': 'cql2-json', filter: { op: '<', args: [{ property: 'eo:cloud_cover' }, thr] },
      }, null, 1);
      fun.textContent = `1  bbox meets footprint        ${nb}\n2  time inside window        ${nd}\n3  cloud below ${thr}%          ${hit.length}`;
      bIn.forEach((inp, i) => { if (document.activeElement !== inp) inp.value = bb[i].toFixed(1); });
      cap.textContent = hit.length
        ? `${hit.length} scenes pass all three tests. A search returns them ${PAGE} per page; each further page is one more request.`
        : 'Each footprint is tested for three things: does it meet the box, is its time inside the window, is its cloud cover below the threshold?';
    };

    /* drag the box: inside = move, bottom-right corner = resize, outside = jump there */
    let drag = null;
    const toLL = (e) => { const r = c.cv.getBoundingClientRect(), v = viewOf(); return { lx: v.ix(e.clientX - r.left), ly: v.iy(e.clientY - r.top), px: e.clientX - r.left, py: e.clientY - r.top }; };
    c.cv.addEventListener('pointerdown', (e) => {
      const { lx, ly, px, py } = toLL(e), v = viewOf();
      if (Math.abs(px - v.X(bb[2])) < 12 && Math.abs(py - v.Y(bb[1])) < 12) drag = { mode: 'resize' };
      else if (lx >= bb[0] && lx <= bb[2] && ly >= bb[1] && ly <= bb[3]) drag = { mode: 'move', ox: lx - bb[0], oy: ly - bb[1] };
      else {
        const w = bb[2] - bb[0], h = bb[3] - bb[1];
        bb = [lx - w / 2, ly - h / 2, lx + w / 2, ly + h / 2];
        drag = { mode: 'move', ox: w / 2, oy: h / 2 };
      }
      c.cv.setPointerCapture(e.pointerId); lastKey = '';
    });
    c.cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const { lx, ly } = toLL(e);
      if (drag.mode === 'resize') { bb[2] = Math.max(bb[0] + 1, Math.min(180, lx)); bb[1] = Math.min(bb[3] - 1, Math.max(-60, ly)); }
      else {
        const w = bb[2] - bb[0], h = bb[3] - bb[1];
        const x0 = Math.min(180 - w, Math.max(-180, lx - drag.ox)), y0 = Math.min(84 - h, Math.max(-60, ly - drag.oy));
        bb = [x0, y0, x0 + w, y0 + h];
      }
      lastKey = '';
    });
    c.cv.addEventListener('pointerup', () => (drag = null));

    const LAND = [
      [[-168, 66], [-141, 70], [-120, 70], [-95, 72], [-80, 73], [-62, 60], [-67, 45], [-76, 35], [-81, 25], [-97, 26], [-97, 16], [-88, 15], [-80, 8], [-84, 10], [-105, 20], [-117, 32], [-124, 42], [-125, 50], [-140, 60], [-165, 60]],
      [[-80, 10], [-60, 11], [-35, -6], [-40, -22], [-58, -38], [-68, -55], [-75, -45], [-72, -18], [-81, -5]],
      [[-10, 36], [10, 44], [28, 41], [40, 36], [60, 25], [80, 8], [100, 2], [120, 20], [140, 40], [150, 60], [180, 68], [180, 75], [100, 78], [60, 70], [30, 70], [10, 58], [-5, 50], [-10, 44]],
      [[-17, 21], [-5, 36], [10, 37], [32, 31], [43, 12], [51, 11], [40, -15], [32, -29], [18, -35], [12, -17], [9, 4], [-8, 5], [-17, 14]],
      [[114, -22], [130, -12], [142, -11], [153, -27], [146, -39], [131, -32], [115, -34]],
    ];
    const draw = () => {
      const C = AI.col(), g = c.ctx, v = viewOf();
      g.clearRect(0, 0, c.w, c.h);
      g.strokeStyle = C.border; g.lineWidth = 1; g.font = `10px ${MONO}`; g.fillStyle = C.muted;
      for (let x = -180; x <= 180; x += 30) { g.beginPath(); g.moveTo(v.X(x) + 0.5, v.Y(84)); g.lineTo(v.X(x) + 0.5, v.Y(-60)); g.stroke(); g.fillText(x + '°', v.X(x) - 8, v.Y(-60) + 14); }
      for (let y = -60; y <= 80; y += 20) { g.beginPath(); g.moveTo(v.X(-180), v.Y(y) + 0.5); g.lineTo(v.X(180), v.Y(y) + 0.5); g.stroke(); g.fillText(y + '°', v.pad.l - 30, v.Y(y) + 4); }
      LAND.forEach((poly) => {
        g.beginPath(); poly.forEach(([x, y], i) => (i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y))));
        g.closePath(); g.fillStyle = AI.mix(C.bg2, C.border, 0.7); g.fill();
      });
      SCENES.forEach((s) => {
        const x = v.X(s.x0), y = v.Y(s.y1), w = v.X(s.x1) - x, h = v.Y(s.y0) - y;
        g.globalAlpha = 0.2 + 0.7 * s.a; g.fillStyle = AI.mix(C.muted, C.blue, s.a);
        g.fillRect(x, y, w, h);
      });
      g.globalAlpha = 1;
      const x0 = v.X(bb[0]), x1 = v.X(bb[2]), y0 = v.Y(bb[3]), y1 = v.Y(bb[1]);
      g.setLineDash([6, 4]); g.strokeStyle = C.orange; g.lineWidth = 2; g.strokeRect(x0, y0, x1 - x0, y1 - y0); g.setLineDash([]);
      g.fillStyle = C.orange; g.fillRect(x1 - 5, y1 - 5, 10, 10);
      g.fillStyle = C.muted; g.fillText('land is schematic; footprints are synthetic', v.pad.l + 4, c.h - 6);
    };

    AI.loop(root, (dt) => {
      if (playing) {
        const adv = dt * speed() * 6; // 6 days of window per second at speed 1: a year in about a minute
        sF += adv; eF += adv;
        if (eF > NDAYS - 1) { const w = eF - sF; sF = 0; eF = w; }
        sDay = Math.round(sF); eDay = Math.round(eF); sync(); lastKey = '';
      }
      const key = [sDay, eDay, thr, bb.map((v) => v.toFixed(2)).join(), page].join('|');
      if (key !== lastKey) { lastKey = key; evalAll(); }
      SCENES.forEach((s) => (s.a += (s.m - s.a) * Math.min(1, dt * 4)));
      draw();
    }, { fps: 30 });
    sync(); evalAll();
  };

  /* ---------- 6 · harvest: band by common name, cloud filter, time stack ---------- */
  AI.labs['stac-harvest'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'From search results to a time stack', 'select the band, filter by cloud, read one window per scene, stack them in time order',
      '<canvas style="height:300px"></canvas><div class="caption"></div>', '');
    const cap = AI.$('.caption', view), c = AI.canvas(AI.$('canvas', view));
    const ITEMS = [
      { id: 'S2A_DEMO_20260505_T31UFT', date: '2026-05-05', cloud: 6.2, p: 'A', epsg: 32632 },
      { id: 'S2B_DEMO_20260508_T31UFT', date: '2026-05-08', cloud: 35.0, p: 'A', epsg: 32632 },
      { id: 'LC09_DEMO_20260515_L2', date: '2026-05-15', cloud: 14.0, p: 'B', epsg: 32632 },
      { id: 'S2A_DEMO_20260522_T32UMU', date: '2026-05-22', cloud: 3.1, p: 'A', epsg: 32633 },
      { id: 'LC09_DEMO_20260531_L2', date: '2026-05-31', cloud: 48.7, p: 'B', epsg: 32632 },
      { id: 'S2B_DEMO_20260606_T31UFT', date: '2026-06-06', cloud: 20.0, p: 'A', epsg: 32632 },
    ];
    /* the same band has a different asset key per provider; common_name is the shared name */
    const KEY = { A: { red: 'B04', green: 'B03', nir: 'B08' }, B: { red: 'red', green: 'green', nir: 'nir08' } };
    const BANDCOL = { red: 'red', green: 'teal', nir: 'green' };
    const rnd = AI.rng(11);
    const TILE = ITEMS.map(() => {
      let a = Array.from({ length: 256 }, () => rnd());
      for (let k = 0; k < 2; k++) a = a.map((v, i) => { const x = i % 16, y = (i / 16) | 0; const nb = [a[y * 16 + ((x + 1) % 16)], a[y * 16 + ((x + 15) % 16)], a[((y + 1) % 16) * 16 + x], a[((y + 15) % 16) * 16 + x]]; return (v + nb[0] + nb[1] + nb[2] + nb[3]) / 5; });
      const lo = Math.min(...a), hi = Math.max(...a);
      return a.map((v) => (v - lo) / (hi - lo + 1e-9));
    });
    const bandSel = document.createElement('label');
    bandSel.innerHTML = '<span>Band</span><select style="background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:4px"><option value="red">red (common_name)</option><option value="nir">nir</option><option value="green">green</option></select>';
    ctl.appendChild(bandSel);
    const bandIn = bandSel.querySelector('select');
    const cl = document.createElement('label');
    cl.innerHTML = '<span>Keep scenes with cloud cover below <span class="v"></span></span><input type="range" min="0" max="100" step="1" value="20">';
    ctl.appendChild(cl);
    const clIn = cl.querySelector('input'), clV = cl.querySelector('.v');
    const nSel = document.createElement('label');
    nSel.innerHTML = '<span>Bands stored per scene</span><select style="background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:4px"><option value="1">1 (this band only)</option><option value="4">4 (red, green, nir, blue)</option></select>';
    ctl.appendChild(nSel);
    const nIn = nSel.querySelector('select');
    const btns = buttons(ctl, ['⏸ Pause', 'Replay']);
    const speed = speedRow(ctl);
    const out = readoutEl(ctl);
    const N = ITEMS.length, TA = N * 0.8, TB = TA + 0.6, TEST_END = TB + N * 0.6, TC = TEST_END + 0.6;
    let tc = 0, playing = true, phase = -1, kept = [];
    const fmtN = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    const KEPT = () => ITEMS.filter((s) => s.cloud < +clIn.value);
    const sync = () => (btns[0].textContent = playing ? '⏸ Pause' : '▶ Play');
    btns[0].onclick = () => { playing = !playing; sync(); };
    btns[1].onclick = () => { tc = 0; playing = true; phase = -1; sync(); };
    clIn.oninput = () => { clV.textContent = clIn.value + '%'; tc = Math.min(tc, TEST_END); phase = -1; };
    bandIn.onchange = () => { phase = -1; };
    nIn.onchange = () => { phase = -1; };
    clV.textContent = clIn.value + '%';
    sync();

    const CAPS = [
      '① Items arrive from the search, sorted by time. Each thumbnail is a 16×16 illustration of the same 256×256 window.',
      '② Filter on the scene-level eo:cloud_cover: keep it only when it is below the threshold. Failing scenes fade out. The cloud value covers the whole scene, not only your window.',
      '③ Pick the asset by common_name, not by key: provider A calls red "B04", provider B calls it "red". ④ Read only the window and stack the kept windows in time order.',
      '⑤ Write the manifest: which item, which asset, which cloud value and which CRS produced each slice. Without it nobody can reproduce this stack.',
    ];
    const stackPos = (k, W, H, tw) => ({ x: W / 2 - tw / 2 + k * 10, y: H - tw - 22 - k * 10 });
    const draw = () => {
      const C = AI.col(), g = c.ctx, W = c.w, H = c.h, tw = Math.min(56, (W - 20) / N - 10);
      g.clearRect(0, 0, W, H);
      const band = bandIn.value, th = +clIn.value;
      const K = KEPT(); // the stack is built from the kept scenes
      const rowY = 52;
      const rowX = (i) => 10 + i * ((W - 20) / N);
      let nUp = 0;
      ITEMS.forEach((s, i) => {
        if (tc < i * 0.8) return;
        const passes = s.cloud < th, testT = TB + i * 0.6;
        let alpha = 1, bcol = C.border;
        if (tc >= testT) {
          bcol = passes ? C.green : C.red;
          if (!passes) alpha = 1 - 0.75 * AI.smooth((tc - testT) / 0.6);
        }
        const k = K.indexOf(s);
        let x = rowX(i), y = rowY;
        if (passes && tc >= TC + Math.max(0, k) * 1.0 && k >= 0) {
          const p = AI.smooth((tc - TC - k * 1.0) / 0.9), sp = stackPos(k, W, H, tw);
          x = AI.lerp(x, sp.x, p); y = AI.lerp(y, sp.y, p);
          nUp++;
        }
        g.save(); g.globalAlpha = alpha;
        const T = TILE[i];
        for (let yy = 0; yy < 16; yy++) for (let xx = 0; xx < 16; xx++) {
          g.fillStyle = AI.mix(C.bg3, C[BANDCOL[band]] || C.blue, 0.2 + 0.8 * T[yy * 16 + xx]);
          g.fillRect(x + xx * (tw / 16), y + yy * (tw / 16), tw / 16 + 0.5, tw / 16 + 0.5);
        }
        g.strokeStyle = bcol; g.lineWidth = tc >= testT ? 2 : 1; g.strokeRect(x - 1, y - 1, tw + 2, tw + 2);
        g.font = '9px ui-monospace,Menlo,Consolas,monospace'; g.fillStyle = C.muted;
        if (k < 0 || tc < TC) {
          g.fillText(s.date.slice(5), x, y + tw + 11);
          g.fillText(`${s.cloud.toFixed(0)}% cloud`, x, y + tw + 22);
        }
        g.restore();
      });
      g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.fillStyle = C.muted;
      g.fillText(`stack: ${K.length} scene(s), time →`, W / 2 - tw / 2, H - 6);
    };
    AI.loop(root, (dt) => {
      if (playing) {
        tc += dt * speed();
        const end = TC + KEPT().length * 1.0 + 2.0;
        if (tc >= end) { tc = end; playing = false; sync(); }
      }
      const K = KEPT(), n = +nIn.value, band = bandIn.value;
      let ph = 0;
      if (tc >= TC + K.length * 1.0 + 0.6) ph = 3; else if (tc >= TC) ph = 2; else if (tc >= TB) ph = 1;
      if (ph !== phase) { phase = ph; cap.textContent = CAPS[ph]; }
      if (ph === 3) {
        const T = K.length, bytes = T * 256 * 256 * n * 2;
        const crs = [...new Set(K.map((s) => s.epsg))];
        const manifest = K.map((s) => ({ id: s.id, datetime: s.date + 'T10:31:22Z', asset: KEY[s.p][band], common_name: band,
          'eo:cloud_cover': s.cloud, 'proj:epsg': s.epsg, href: `./${s.id}/${KEY[s.p][band]}.tif` }));
        out.textContent = `kept ${T} of ${N} scenes (cloud < ${clIn.value}%)\nstack shape [${T}, 256, 256] per band\nbytes: ${T} × 256 × 256 × ${n} × 2 = ${fmtN(bytes)} (${(bytes / 1048576).toFixed(2)} MiB)` +
          (crs.length > 1 ? `\n✗ CRS differs (${crs.join(', ')}): reproject to one grid before stacking.` : '') +
          '\n' + JSON.stringify(manifest, null, 1);
      } else {
        out.textContent = 'the manifest appears when the stack is complete';
      }
      draw();
    }, { fps: 30 });
  };

  /* ---------- 7 · validate an Item: editable JSON, checks run in sequence ---------- */
  AI.labs['stac-validate'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Check an Item before you publish it', 'edit the JSON on the left: each rule of this lesson is checked in turn',
      `<textarea class="sv" spellcheck="false" style="width:100%;min-height:330px;font-family:${MONO};font-size:12px;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:8px;box-sizing:border-box"></textarea><div class="caption"></div>`, '');
    const ta = AI.$('.sv', view), cap = AI.$('.caption', view);
    ta.value = `{
  "type": "Feature",
  "stac_version": "1.0.0",
  "stac_extensions": ["https://stac-extensions.github.io/eo/v1.1.0/schema.json"],
  "id": "S2B_DEMO_20260714_T31UFT_L2A",
  "geometry": {"type": "Polygon", "coordinates": [[[10.6, 48.2], [11.9, 48.2], [11.9, 49.2], [10.6, 49.2], [10.6, 48.2]]]},
  "bbox": [10.6, 48.2, 11.9, 49.2],
  "properties": {"datetime": "2026-07-14T10:31:22Z", "eo:cloud_cover": 12.4},
  "links": [{"rel": "self", "href": "./S2B_DEMO_20260714_T31UFT_L2A.json", "type": "application/geo+json"}],
  "assets": {"B04": {"href": "./B04.tif", "roles": ["data"]}}
}`;
    const CHECKS = [
      ['json', 'parses as JSON'],
      ['type', 'type is "Feature"'],
      ['version', 'stac_version is a string'],
      ['id', 'id is a non-empty string'],
      ['geometry', 'geometry is null or a closed lon/lat Polygon ring'],
      ['bbox', 'bbox has four numbers when geometry is set'],
      ['contains', 'bbox contains the geometry and is not swapped'],
      ['datetime', 'properties.datetime is UTC (ends in Z)'],
      ['links', 'links is a list of {rel, href}'],
      ['assets', 'assets is a map of objects with href'],
      ['exts', 'stac_extensions is a list of https URLs'],
      ['declared', 'every prefixed property has its extension declared'],
    ];
    const PFX = { eo: '/eo/', proj: '/projection/', raster: '/raster/' };
    const ringOf = (geo) => (geo && geo.type === 'Polygon' && Array.isArray(geo.coordinates) && Array.isArray(geo.coordinates[0]) ? geo.coordinates[0] : null);
    const validate = (text) => {
      const R = {}, set = (k, ok, msg) => (R[k] = { ok, msg });
      let o = null;
      try { o = JSON.parse(text); set('json', true, 'parsed'); } catch (e) { set('json', false, e.message.split('\n')[0]); }
      const g = o && typeof o === 'object' && !Array.isArray(o) ? o : {};
      set('type', g.type === 'Feature', g.type === 'Feature' ? '"Feature"' : `found ${JSON.stringify(g.type)}`);
      set('version', typeof g.stac_version === 'string', typeof g.stac_version === 'string' ? g.stac_version : 'missing');
      set('id', typeof g.id === 'string' && g.id.length > 0, typeof g.id === 'string' && g.id ? g.id : 'missing or empty');
      const geo = g.geometry;
      let geoOk = false, geoMsg = 'missing';
      const ring = ringOf(geo);
      if (geo === null) { geoOk = true; geoMsg = 'null (allowed: the item has no footprint)'; }
      else if (geo && typeof geo === 'object') {
        if (!ring) geoMsg = 'expected type "Polygon" with coordinates[0] as a ring';
        else if (ring.length < 4) geoMsg = 'a ring needs at least 4 positions';
        else if (JSON.stringify(ring[0]) !== JSON.stringify(ring[ring.length - 1])) geoMsg = 'ring not closed: the last position must equal the first';
        else if (ring.some((p) => !(Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90))) geoMsg = 'positions must be longitude and latitude in degrees (|lon| ≤ 180, |lat| ≤ 90)';
        else { geoOk = true; geoMsg = `closed ring, ${ring.length - 1} vertices`; }
      }
      set('geometry', geoOk, geoMsg);
      const bb = g.bbox;
      const bbOk = geo === null || (Array.isArray(bb) && bb.length === 4 && bb.every((v) => typeof v === 'number'));
      set('bbox', bbOk, geo === null ? 'not needed without geometry' : bbOk ? 'four numbers' : 'missing or not four numbers');
      let cOk = false, cMsg = 'not evaluated';
      if (geo === null) { cOk = true; cMsg = 'no geometry: nothing to contain'; }
      else if (ring && Array.isArray(bb) && bb.length === 4) {
        const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
        const comp = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
        if (bb[0] > bb[2] || bb[1] > bb[3]) cMsg = 'min and max are swapped: order is [minx, miny, maxx, maxy]';
        else if (bb[0] <= comp[0] && bb[1] <= comp[1] && bb[2] >= comp[2] && bb[3] >= comp[3]) {
          cOk = true; cMsg = bb.every((v, i) => Math.abs(v - comp[i]) < 1e-6) ? 'equal to the box of the geometry' : 'contains the geometry (looser than needed)';
        } else cMsg = `does not contain the geometry; its box is [${comp.map(r3).join(', ')}]`;
      }
      set('contains', cOk, cMsg);
      const pr = g.properties && typeof g.properties === 'object' ? g.properties : {};
      const dt = pr.datetime;
      const dtOk = (typeof dt === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/.test(dt)) || (typeof pr.start_datetime === 'string' && typeof pr.end_datetime === 'string');
      set('datetime', dtOk, dtOk ? (typeof dt === 'string' ? dt : 'start and end given') : 'need datetime ending in Z, or start_datetime and end_datetime');
      const L = g.links;
      const lOk = Array.isArray(L) && L.every((l) => l && typeof l.rel === 'string' && typeof l.href === 'string');
      set('links', lOk, lOk ? `${L.length} links` : 'each link needs a string rel and href');
      const A = g.assets;
      const aOk = !!A && typeof A === 'object' && !Array.isArray(A) && Object.values(A).every((a) => a && typeof a.href === 'string');
      set('assets', aOk, aOk ? `${Object.keys(A).length} assets` : 'each asset needs a string href');
      const X = g.stac_extensions;
      const xOk = X === undefined || (Array.isArray(X) && X.every((u) => typeof u === 'string' && u.startsWith('https://')));
      set('exts', xOk, X === undefined ? 'absent (fine if no prefixed fields)' : xOk ? `${X.length} declared` : 'must be a list of https URLs');
      const X2 = Array.isArray(X) ? X : [];
      const pref = [...new Set(Object.keys(pr).filter((k) => /^[a-z]+:/.test(k)).map((k) => k.split(':')[0]))];
      const miss = pref.filter((p) => !(PFX[p] && X2.some((u) => u.includes(PFX[p]))));
      set('declared', miss.length === 0, miss.length ? `undeclared prefix: ${miss.join(', ')}` : pref.length ? `${pref.join(', ')} declared` : 'no prefixed fields');
      return CHECKS.map(([id, label]) => ({ id, label, ok: R[id] ? R[id].ok : false, msg: R[id] ? R[id].msg : '' }));
    };

    const rows = document.createElement('div'); ctl.appendChild(rows);
    const btns = buttons(ctl, ['⏸ Pause', 'Step ▸', 'Replay']);
    const speed = speedRow(ctl);
    const sum = readoutEl(ctl);
    const STEP = 0.45;
    let res = [], shownN = 0, acc = 0, playing = true, lastKey = '';
    const sync = () => (btns[0].textContent = playing ? '⏸ Pause' : '▶ Play');
    btns[0].onclick = () => { playing = !playing; sync(); };
    btns[1].onclick = () => { playing = false; sync(); if (shownN < res.length) shownN++; };
    btns[2].onclick = () => { shownN = 0; acc = 0; playing = true; sync(); };
    ta.addEventListener('input', () => { shownN = 0; acc = 0; playing = true; sync(); res = validate(ta.value); });
    res = validate(ta.value); sync();

    AI.loop(root, (dt) => {
      if (playing && shownN < res.length) {
        acc += dt * speed();
        while (acc >= STEP && shownN < res.length) { shownN++; acc -= STEP; }
      }
      const key = shownN + '|' + res.map((r) => (r.ok ? 1 : 0) + r.msg).join('|');
      if (key === lastKey) return;
      lastKey = key;
      rows.innerHTML = res.map((r, i) => {
        if (i >= shownN) return `<div style="font-size:12px;color:var(--muted);opacity:.35;padding:3px 0">… ${r.label}</div>`;
        const col = r.ok ? 'var(--green)' : 'var(--red)';
        return `<div style="font-size:12px;padding:3px 0"><b style="color:${col}">${r.ok ? '✓' : '✗'}</b> ${esc(r.label)}<div style="color:var(--muted);font-family:${MONO};font-size:11px;padding-left:14px">${esc(r.msg)}</div></div>`;
      }).join('');
      if (shownN < res.length) {
        cap.textContent = `Checking: ${res[shownN].label}`;
        sum.textContent = `${shownN} of ${res.length} checked`;
      } else {
        const bad = res.filter((r) => !r.ok);
        cap.textContent = bad.length ? `First problem: ${bad[0].label}. Fix it and the checks run again.` : 'All checks pass for the rules in this lesson.';
        sum.textContent = `${res.length - bad.length} of ${res.length} checks pass\n\nThis is a teaching checker: it covers the rules of Lesson 2.7, not the full JSON Schema. Use a validator for publishing.`;
      }
    }, { fps: 30 });
  };
})();
