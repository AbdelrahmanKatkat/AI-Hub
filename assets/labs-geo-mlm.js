/* GeoAI Module 6 (STAC MLM) labs. Every animation runs on dt (seconds), with speed, pause and step controls.
   Records shown here follow the MLM v1.6.0 field names used in geo-mlm.html. Numbers are computed in code, not typed in. */
(function () {
  const AI = window.AI;
  const ESC = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ---------- shared styles (scoped to .mlm-*) ---------- */
  if (!document.getElementById('mlm-lab-style')) {
    const s = document.createElement('style');
    s.id = 'mlm-lab-style';
    s.textContent = `
.mlm-row{display:flex;gap:8px;flex-wrap:wrap;align-items:stretch}
.mlm-card{border:1px solid var(--border);border-radius:10px;padding:10px 12px;background:var(--bg);flex:1;min-width:190px}
.mlm-card h4{margin:0 0 6px;font-size:13px;font-family:ui-monospace,Menlo,Consolas,monospace}
.mlm-ok{color:var(--green)}.mlm-bad{color:var(--red)}.mlm-wait{color:var(--muted)}.mlm-warn{color:var(--yellow)}
.mlm-pre{font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:8px 10px;overflow:auto;margin:0;white-space:pre}
.mlm-pre .ln{display:block;border-radius:4px;padding:0 4px;transition:background .2s}
.mlm-pre .bad{background:rgba(248,81,73,.18);color:var(--red)}
.mlm-pre .on{background:rgba(63,185,80,.18)}
.mlm-chip{display:inline-block;border:1px solid var(--border);border-radius:14px;padding:3px 10px;margin:2px;cursor:pointer;font-size:12px;background:var(--bg);user-select:none}
.mlm-chip.on{border-color:var(--green);color:var(--green)}
.mlm-chip.sel{outline:2px solid var(--blue)}
.mlm-zone{border:2px dashed var(--border);border-radius:12px;min-height:110px;padding:8px;flex:1;min-width:150px;transition:border-color .2s}
.mlm-zone.over{border-color:var(--blue)}
.mlm-zone h4{margin:0 0 6px;font-size:13px}
.mlm-cap{font-size:13px;line-height:1.5;border-left:3px solid var(--green);padding:6px 10px;background:var(--bg2);border-radius:0 8px 8px 0;margin-top:8px;min-height:44px}
.mlm-tl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.mlm-tl label{font-size:12px;color:var(--muted)}
.mlm-tl input[type=range]{vertical-align:middle}
.mlm-field{display:flex;gap:6px;align-items:center;justify-content:space-between;font-size:12px;margin:4px 0;flex-wrap:wrap}
.mlm-field input[type=text],.mlm-field input[type=number],.mlm-field select{font:12px ui-monospace,Menlo,Consolas,monospace;min-width:150px}
.mlm-err{color:var(--red);font-size:12px;min-height:16px;margin:2px 0}
.mlm-cv{width:100%;height:300px;display:block}
.mlm-tbl{width:100%;border-collapse:collapse;font-size:12px}
.mlm-tbl td,.mlm-tbl th{border-bottom:1px solid var(--border);padding:5px 6px;text-align:left;vertical-align:top}
.mlm-strike{text-decoration:line-through;color:var(--muted)}
`;
    document.head.appendChild(s);
  }

  /* ---------- helpers ---------- */
  // Adds play/pause, step and speed controls to ctl and returns the clock state. dt-based.
  function clock(ctl, dur, o = {}) {
    const extra = o.extra || '';
    ctl.insertAdjacentHTML('afterbegin', `<div class="mlm-tl"><button class="btn small" data-a="play">⏸ Pause</button><button class="btn small" data-a="step">Step ▸</button><button class="btn small" data-a="reset">↺ Restart</button><label>Speed <input type="range" min="0.25" max="2" step="0.05" value="${o.speed || 0.6}"></label>${extra}</div>`);
    const st = { t: o.t0 || 0, dur, loop: o.loop !== false, playing: o.playing !== false, speed: o.speed || 0.6, stepReq: 0, stepLen: o.stepLen || 0.5, lastKey: null };
    const play = ctl.querySelector('[data-a=play]');
    const setPlay = () => (play.textContent = st.playing ? '⏸ Pause' : '▶ Play');
    play.onclick = () => { st.playing = !st.playing; setPlay(); };
    ctl.querySelector('[data-a=step]').onclick = () => { st.playing = false; setPlay(); st.stepReq += 1; };
    ctl.querySelector('[data-a=reset]').onclick = () => { st.t = 0; st.playing = true; setPlay(); st.lastKey = null; };
    ctl.querySelector('input[type=range]').oninput = (e) => (st.speed = +e.target.value);
    st.setPlay = setPlay;
    st.advance = (dt) => {
      if (st.stepReq) { st.t += st.stepLen; st.stepReq = 0; }
      else if (st.playing) st.t += dt * st.speed;
      else return false;
      if (st.loop) st.t = st.t % st.dur;
      else if (st.t > st.dur) { st.t = st.dur; st.playing = false; setPlay(); }
      return true;
    };
    return st;
  }

  function fitCanvas(root) {
    const cv = root.querySelector('canvas');
    return { cv, c: AI.canvas(cv) };
  }

  const rr = (g, x, y, w, h, r, fill, stroke) => {
    g.beginPath(); g.roundRect(x, y, w, h, r);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); }
  };
  const txt = (g, s, x, y, color, font, align) => {
    g.fillStyle = color; g.font = font || '12px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = align || 'left'; g.fillText(s, x, y);
  };
  const isInt = (s) => /^\s*\d+\s*$/.test(String(s));

  /* ---------- the U-Net family used in 6.5, 6.8 and 6.9 (same counting as the Module 6 text) ---------- */
  function unet(C, w, K, p) {
    const L = [];
    const conv = (name, cin, cout, k, h, bias) => L.push({ name, params: k * k * cin * cout + (bias ? cout : 0), bufs: 0, inAct: cin * h * h, outAct: cout * h * h });
    const bn = (name, c, h) => L.push({ name, params: 2 * c, bufs: 2 * c, inAct: c * h * h, outAct: c * h * h });
    const act = (name, c, h) => L.push({ name, params: 0, bufs: 0, inAct: c * h * h, outAct: c * h * h });
    const dconv = (pfx, cin, cout, h) => {
      conv(pfx + '.conv1', cin, cout, 3, h, false); bn(pfx + '.bn1', cout, h); act(pfx + '.relu1', cout, h);
      conv(pfx + '.conv2', cout, cout, 3, h, false); bn(pfx + '.bn2', cout, h); act(pfx + '.relu2', cout, h);
    };
    const convT = (name, cin, cout, hin) => L.push({ name, params: cin * cout * 4 + cout, bufs: 0, inAct: cin * hin * hin, outAct: cout * (2 * hin) * (2 * hin) });
    dconv('enc1', C, w, p); dconv('enc2', w, 2 * w, p / 2); dconv('bott', 2 * w, 4 * w, p / 4);
    convT('up1', 4 * w, 2 * w, p / 4); dconv('dec1', 4 * w, 2 * w, p / 2);
    convT('up2', 2 * w, w, p / 2); dconv('dec2', 2 * w, w, p);
    conv('head', w, K, 1, p, true);
    const P = L.reduce((s, l) => s + l.params, 0);
    const B = L.reduce((s, l) => s + l.bufs, 0);
    const peak = Math.max(...L.map((l) => l.inAct + l.outAct));
    const blocks = {};
    for (const l of L) { const k = l.name.split('.')[0]; blocks[k] = (blocks[k] || 0) + l.params; }
    return { P, B, peak, blocks, layers: L };
  }
  // bytes = weights (params + BN stats, float32) + batch * peak activation (float32)
  const memBytes = (m, batch) => (m.P + m.B) * 4 + batch * m.peak * 4;

  /* ============ 6.1 mlmMatch: catalogue query, field by field ============ */
  AI.labs.mlmMatch = function (root) {
    const M = [
      { name: 'unet-landcover-s2-demo', tasks: ['semantic-segmentation'], bands: ['B02', 'B03', 'B04', 'B08'], patch: 256 },
      { name: 'scene-tagger', tasks: ['classification'], bands: ['B02', 'B03', 'B04', 'B08'], patch: 64 },
      { name: 'hires-building-net', tasks: ['semantic-segmentation'], bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'], patch: 512 },
    ];
    const NEED = ['B02', 'B03', 'B04', 'B08'];
    const { view, ctl } = AI.labShell(root, 'Model catalogue query', 'every record is tested field by field',
      '<div class="mlm-row" data-o="cards"></div><div class="mlm-cap" data-o="cap"></div>',
      `<div class="mlm-field"><span>mlm:tasks =</span><select data-q="task"><option>semantic-segmentation</option><option>classification</option><option>scene-classification</option></select></div>
       <div class="mlm-field"><span>patch =</span><select data-q="patch"><option>256</option><option>64</option><option>512</option></select></div>
       <div class="mlm-field"><span>accept a superset of the bands</span><input type="checkbox" data-q="sup"></div>
       <div class="mlm-field"><span>bands required</span><code>${NEED.join(', ')}</code></div>
       <button class="btn small" data-a="run">Run query</button>`);
    const st = clock(ctl, 9, { t0: 9, loop: false });
    const q = {};
    const readQuery = () => { q.task = root.querySelector('[data-q=task]').value; q.patch = +root.querySelector('[data-q=patch]').value; q.sup = root.querySelector('[data-q=sup]').checked; };
    const bandsOK = (m) => (q.sup ? NEED.every((b) => m.bands.includes(b)) : m.bands.length === NEED.length && NEED.every((b) => m.bands.includes(b)));
    const crit = (m) => [
      { k: 'tasks', ok: m.tasks.includes(q.task), why: `mlm:tasks [${m.tasks}] ${m.tasks.includes(q.task) ? 'contains' : 'does not contain'} ${q.task}` },
      { k: 'bands', ok: bandsOK(m), why: `bands [${m.bands.length}] ${bandsOK(m) ? 'satisfy' : 'fail'} the ${q.sup ? 'superset' : 'exact'} rule` },
      { k: 'patch', ok: m.patch === q.patch, why: `input shape patch ${m.patch} ${m.patch === q.patch ? '=' : '≠'} ${q.patch}` },
    ];
    const cardsEl = root.querySelector('[data-o=cards]'), capEl = root.querySelector('[data-o=cap]');
    cardsEl.innerHTML = M.map((m, i) => `<div class="mlm-card"><h4>${ESC(m.name)}</h4><div data-c="${i}"></div></div>`).join('');
    M.forEach((m, i) => { cardsEl.querySelector(`[data-c="${i}"]`).innerHTML = [0, 1, 2].map((k) => `<div class="mlm-field"><span>${['mlm:tasks', 'input bands', 'input patch'][k]}</span><b data-r="${i}-${k}" class="mlm-wait">…</b></div>`).join(''); });
    const run = () => { readQuery(); st.t = 0; st.playing = true; st.setPlay(); st.lastKey = null; };
    ctl.querySelector('[data-a=run]').onclick = run;
    ctl.querySelectorAll('select,input').forEach((el) => el.addEventListener('change', run));
    readQuery();
    AI.loop(root, (dt) => {
      if (!st.advance(dt) && st.lastKey !== null) return;
      const idx = Math.min(9, Math.floor(st.t / 0.9));
      const key = idx + '|' + q.task + q.patch + q.sup;
      if (key === st.lastKey) return;
      st.lastKey = key;
      const allDone = idx >= 9;
      M.forEach((m, i) => {
        const cs = crit(m);
        cs.forEach((c, k) => {
          const flat = i * 3 + k, el = cardsEl.querySelector(`[data-r="${i}-${k}"]`);
          if (flat < idx) { el.textContent = c.ok ? '✔ pass' : '✖ fail'; el.className = c.ok ? 'mlm-ok' : 'mlm-bad'; }
          else if (flat === idx && !allDone) { el.textContent = '…checking'; el.className = 'mlm-wait'; }
          else { el.textContent = '…'; el.className = 'mlm-wait'; }
        });
      });
      if (!allDone) {
        const flat = idx, i = Math.floor(flat / 3), k = flat % 3;
        capEl.textContent = `Testing ${M[i].name}: ${crit(M[i])[k].why}.`;
      } else {
        const hits = M.filter((m) => crit(m).every((c) => c.ok)).map((m) => m.name);
        capEl.textContent = hits.length ? `Matches: ${hits.join(', ')}. Change the query and run again to see how the answer moves.` : 'No record matches this query. Relax one field (patch, bands or task) and run again.';
      }
    });
  };

  /* ============ 6.2 mlmBuilder: record form, live JSON, validation, task picker ============ */
  const TASKS = ['regression', 'classification', 'scene-classification', 'detection', 'object-detection', 'segmentation', 'semantic-segmentation', 'instance-segmentation', 'panoptic-segmentation', 'similarity-search', 'generative', 'image-captioning', 'super-resolution', 'downscaling'];
  const ACCEL = ['', 'amd64', 'cuda', 'xla', 'amd-rocm', 'intel-ipex-cpu', 'intel-ipex-gpu', 'macos-arm'];
  AI.labs.mlmBuilder = function (root) {
    const EX = { name: 'unet-landcover-s2-demo', arch: 'U-Net', tasks: 'semantic-segmentation', fw: 'pytorch', fwv: '2.4.1', params: '117590', mem: '50804568', batch: '4', accel: 'cuda', pre: 'false', contract: true, model: true };
    const { view, ctl } = AI.labShell(root, 'MLM record builder', 'fill the form; the JSON on the left is the record, and errors are highlighted in it',
      `<pre class="mlm-pre" data-o="pre"></pre><div class="mlm-err" data-o="err"></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-field"><span>mlm:name</span><input type="text" data-f="name"></div>
       <div class="mlm-field"><span>mlm:architecture</span><input type="text" data-f="arch"></div>
       <div class="mlm-field"><span>mlm:tasks (comma list)</span><input type="text" data-f="tasks"></div>
       <div data-o="chips"></div>
       <div class="mlm-field"><span>mlm:framework</span><input type="text" data-f="fw"></div>
       <div class="mlm-field"><span>mlm:framework_version</span><input type="text" data-f="fwv"></div>
       <div class="mlm-field"><span>mlm:total_parameters</span><input type="text" data-f="params"></div>
       <div class="mlm-field"><span>mlm:memory_size (bytes)</span><input type="text" data-f="mem"></div>
       <div class="mlm-field"><span>mlm:batch_size_suggestion</span><input type="text" data-f="batch"></div>
       <div class="mlm-field"><span>mlm:accelerator</span><select data-f="accel">${ACCEL.map((a) => `<option value="${a}">${a || 'null'}</option>`).join('')}</select></div>
       <div class="mlm-field"><span>mlm:pretrained</span><select data-f="pre"><option value="false">false</option><option value="true">true</option></select></div>
       <div class="mlm-field"><span>mlm:input and mlm:output present</span><input type="checkbox" data-f="contract"></div>
       <div class="mlm-field"><span>an asset with role mlm:model</span><input type="checkbox" data-f="model"></div>
       <button class="btn small" data-a="example">Load the worked example</button>`);
    const pre = root.querySelector('[data-o=pre]'), errEl = root.querySelector('[data-o=err]'), cap = root.querySelector('[data-o=cap]');
    const chipsEl = root.querySelector('[data-o=chips]');
    const tasksIn = root.querySelector('[data-f=tasks]');
    const f = {};
    const st = clock(ctl, 4, { loop: true });
    chipsEl.innerHTML = TASKS.map((t) => `<span class="mlm-chip" data-t="${t}">${t}</span>`).join('');
    const tokens = () => tasksIn.value.split(',').map((s) => s.trim()).filter(Boolean);
    const syncChips = () => { const set = new Set(tokens()); chipsEl.querySelectorAll('.mlm-chip').forEach((el) => el.classList.toggle('on', set.has(el.dataset.t))); };
    chipsEl.addEventListener('click', (e) => {
      const t = e.target.dataset.t; if (!t) return;
      const set = tokens(); const i = set.indexOf(t);
      if (i >= 0) set.splice(i, 1); else set.push(t);
      tasksIn.value = set.join(', '); syncChips(); render();
    });
    const read = () => {
      f.name = root.querySelector('[data-f=name]').value.trim();
      f.arch = root.querySelector('[data-f=arch]').value.trim();
      f.tokens = tokens();
      f.fw = root.querySelector('[data-f=fw]').value.trim();
      f.fwv = root.querySelector('[data-f=fwv]').value.trim();
      f.params = root.querySelector('[data-f=params]').value.trim();
      f.mem = root.querySelector('[data-f=mem]').value.trim();
      f.batch = root.querySelector('[data-f=batch]').value.trim();
      f.accel = root.querySelector('[data-f=accel]').value;
      f.pre = root.querySelector('[data-f=pre]').value === 'true';
      f.contract = root.querySelector('[data-f=contract]').checked;
      f.model = root.querySelector('[data-f=model]').checked;
    };
    const setForm = (e) => {
      root.querySelector('[data-f=name]').value = e.name; root.querySelector('[data-f=arch]').value = e.arch;
      tasksIn.value = e.tasks; root.querySelector('[data-f=fw]').value = e.fw; root.querySelector('[data-f=fwv]').value = e.fwv;
      root.querySelector('[data-f=params]').value = e.params; root.querySelector('[data-f=mem]').value = e.mem;
      root.querySelector('[data-f=batch]').value = e.batch; root.querySelector('[data-f=accel]').value = e.accel;
      root.querySelector('[data-f=pre]').value = e.pre; root.querySelector('[data-f=contract]').checked = e.contract;
      root.querySelector('[data-f=model]').checked = e.model; syncChips();
    };
    // Each line: [key, value, error or null]
    const build = () => {
      const L = [];
      const e = (k, v, err) => L.push({ k, v, err });
      e('stac_extensions', ['https://stac-extensions.github.io/mlm/v1.6.0/schema.json'], null);
      e('mlm:name', f.name || '', f.name ? null : 'mlm:name is required (Item-only)');
      e('mlm:architecture', f.arch || '', f.arch ? null : 'mlm:architecture is required');
      const bad = f.tokens.filter((t) => !TASKS.includes(t));
      e('mlm:tasks', f.tokens, !f.tokens.length ? 'mlm:tasks needs at least one value' : bad.length ? `"${bad[0]}" is not an enumerated task` : null);
      if (f.fw) e('mlm:framework', f.fw, null);
      if (f.fwv) e('mlm:framework_version', f.fwv, null);
      if (f.params !== '') e('mlm:total_parameters', isInt(f.params) ? +f.params : f.params, isInt(f.params) ? null : 'must be a whole number of parameters');
      if (f.mem !== '') e('mlm:memory_size', isInt(f.mem) ? +f.mem : f.mem, isInt(f.mem) ? null : 'must be an integer number of bytes');
      if (f.batch !== '') e('mlm:batch_size_suggestion', isInt(f.batch) && +f.batch >= 1 ? +f.batch : f.batch, isInt(f.batch) && +f.batch >= 1 ? null : 'must be an integer of at least 1');
      e('mlm:accelerator', f.accel || null, null);
      e('mlm:pretrained', f.pre, null);
      e('mlm:input', f.contract ? [{ name: '…' }] : null, f.contract ? null : 'mlm:input is required (Lesson 6.3)');
      e('mlm:output', f.contract ? [{ name: '…' }] : null, f.contract ? null : 'mlm:output is required (Lesson 6.3)');
      e('assets', f.model ? { model: { roles: ['mlm:model'], 'mlm:artifact_type': '…' } } : {}, f.model ? null : 'needs an asset whose roles include mlm:model');
      return L;
    };
    const nowKey = () => JSON.stringify(f);
    const render = () => {
      read();
      const L = build();
      const errs = L.filter((x) => x.err);
      let html = '{\n  "type": "Feature",\n  "stac_version": "1.0.0",\n';
      L.forEach((x, i) => {
        const val = JSON.stringify(x.v);
        const line = `  "${x.k}": ${val}${i < L.length - 1 ? ',' : ''}`;
        const cls = x.err ? 'ln bad' : 'ln';
        html += `<span class="${cls}" data-err="${x.err ? 1 : 0}">${ESC(line)}</span>\n`;
      });
      html += '}';
      pre.innerHTML = html;
      errEl.textContent = errs.length ? `${errs.length} problem${errs.length > 1 ? 's' : ''}: ${errs.map((x) => `${x.k}: ${x.err}`).join(' · ')}` : '';
      f.errs = errs;
      st.lastKey = null;
    };
    root.addEventListener('input', render);
    root.addEventListener('change', render);
    root.querySelector('[data-a=example]').onclick = () => { setForm(EX); render(); };
    setForm(EX);
    render();
    AI.loop(root, (dt) => {
      const moved = st.advance(dt);
      const errs = f.errs || [];
      const first = errs[0];
      cap.textContent = first ? `Caption: ${first.k} is wrong: ${first.err}. Fix it, and the line stops pulsing.` : 'All required fields of this form are present and well formed. The record is still a claim until it is checked against the model (Lesson 6.9).';
      if (!moved && st.lastKey !== null) return;
      st.lastKey = 'x';
      const phase = Math.sin(st.t * 3.2) * 0.5 + 0.5;
      pre.querySelectorAll('.ln.bad').forEach((el) => (el.style.opacity = (0.45 + 0.55 * phase).toFixed(3)));
    });
  };

  /* ============ 6.2 mlmPlacement: drag a field to Item, Asset or Collection ============ */
  const PLACE = [
    ['mlm:name', { item: 'yes', asset: 'no', coll: 'yes', why: 'Item-only field, and the one MLM field a Collection may also carry (as a name, not a summary).' }],
    ['mlm:input', { item: 'yes', asset: 'no', coll: 'no', why: 'Item-only: the input contract describes the model, so it lives on the Item and is left out of Collection summaries.' }],
    ['mlm:output', { item: 'yes', asset: 'no', coll: 'no', why: 'Item-only: the output contract, left out of Collection summaries.' }],
    ['mlm:hyperparameters', { item: 'yes', asset: 'no', coll: 'no', why: 'Item-only and free-form, so not summarised.' }],
    ['mlm:tasks', { item: 'yes', asset: 'no', coll: 'yes', why: 'A model-level fact; summarised in a Collection as the set of tasks.' }],
    ['mlm:total_parameters', { item: 'yes', asset: 'no', coll: 'yes', why: 'A model-level number; a Collection can summarise it as a range.' }],
    ['mlm:memory_size', { item: 'yes', asset: 'no', coll: 'yes', why: 'A model-level number of bytes; summarised as a range.' }],
    ['mlm:accelerator', { item: 'yes', asset: 'no', coll: 'yes', why: 'A model-level hardware fact.' }],
    ['mlm:artifact_type', { item: 'no', asset: 'yes', coll: 'no', why: 'Asset field. Required on the asset that has the mlm:model role.' }],
    ['mlm:compile_method', { item: 'no', asset: 'yes', coll: 'no', why: 'Asset field on the model asset: aot, jit or null.' }],
    ['mlm:entrypoint', { item: 'check', asset: 'yes', coll: 'no', why: 'Asset field on the source-code asset; setting it makes the code role mandatory. The MLM repository example places it on the Item, so check which version your validator expects.' }],
    ['mlm:source_code', { item: 'no', asset: 'yes', coll: 'no', why: 'Asset object (the code asset), recommended, with the code role.' }],
    ['mlm:container', { item: 'no', asset: 'yes', coll: 'no', why: 'Asset object for a container with the runtime role.' }],
  ];
  AI.labs.mlmPlacement = function (root) {
    const { view, ctl } = AI.labShell(root, 'Where does this field go?', 'drag a field onto Item, Asset or Collection (or click a field, then a zone)',
      `<div class="mlm-row" data-o="chips"></div><div class="mlm-row" style="margin-top:10px" data-o="zones"></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-tl"><span style="font-size:12px;color:var(--muted)">Verdicts follow the MLM v1.6.0 README: item-only fields, asset fields and the Collection rule.</span></div>`);
    const chipsEl = root.querySelector('[data-o=chips]'), zonesEl = root.querySelector('[data-o=zones]'), cap = root.querySelector('[data-o=cap]');
    const ZN = [['item', 'Item (properties)'], ['asset', 'Asset (in assets{})'], ['coll', 'Collection summaries']];
    const placed = {};
    let sel = null, flash = null;
    chipsEl.innerHTML = PLACE.map(([k]) => `<span class="mlm-chip" draggable="true" data-k="${k}">${k}</span>`).join('');
    zonesEl.innerHTML = ZN.map(([z, label]) => `<div class="mlm-zone" data-z="${z}"><h4>${label}</h4><div data-l="${z}"></div></div>`).join('');
    const verdict = (k, z) => {
      const d = PLACE.find((p) => p[0] === k)[1], v = d[z];
      const word = v === 'yes' ? '✔ allowed' : v === 'no' ? '✖ not allowed' : '⚠ check version';
      return { v, word, why: d.why };
    };
    const drop = (k, z) => {
      const vd = verdict(k, z);
      placed[z] = placed[z] || [];
      if (!placed[z].includes(k)) placed[z].push(k);
      flash = { z, t: 0 };
      cap.innerHTML = `<b>${ESC(k)}</b> in <b>${ESC(ZN.find((x) => x[0] === z)[1])}</b>: ${vd.word}. ${ESC(vd.why)}`;
      draw();
    };
    const draw = () => {
      ZN.forEach(([z]) => {
        const el = zonesEl.querySelector(`[data-l=${z}]`);
        el.innerHTML = (placed[z] || []).map((k) => {
          const vd = verdict(k, z);
          const cls = vd.v === 'yes' ? 'mlm-ok' : vd.v === 'no' ? 'mlm-bad' : 'mlm-warn';
          return `<div class="mlm-field"><code>${ESC(k)}</code><b class="${cls}">${vd.word}</b></div>`;
        }).join('') || '<span class="mlm-wait" style="font-size:12px">drop fields here</span>';
      });
    };
    chipsEl.addEventListener('dragstart', (e) => { if (e.target.dataset.k) e.dataTransfer.setData('text/plain', e.target.dataset.k); });
    chipsEl.addEventListener('click', (e) => {
      const k = e.target.dataset.k; if (!k) return;
      sel = sel === k ? null : k;
      chipsEl.querySelectorAll('.mlm-chip').forEach((el) => el.classList.toggle('sel', el.dataset.k === sel));
      cap.textContent = sel ? `Selected ${sel}. Now click a zone.` : 'Drag a field onto a zone, or select a field and click a zone.';
    });
    zonesEl.querySelectorAll('.mlm-zone').forEach((zEl) => {
      const z = zEl.dataset.z;
      zEl.addEventListener('dragover', (e) => { e.preventDefault(); zEl.classList.add('over'); });
      zEl.addEventListener('dragleave', () => zEl.classList.remove('over'));
      zEl.addEventListener('drop', (e) => { e.preventDefault(); zEl.classList.remove('over'); const k = e.dataTransfer.getData('text/plain'); if (k) drop(k, z); });
      zEl.addEventListener('click', () => { if (sel) drop(sel, z); });
    });
    const st = clock(ctl, 3, { loop: true });
    draw();
    cap.textContent = 'Drag a field onto a zone, or select a field and click a zone. Try mlm:artifact_type on the Item, then mlm:tasks on a Collection.';
    AI.loop(root, (dt) => {
      if (!st.advance(dt) && !flash) return;
      if (flash) {
        flash.t += dt;
        const a = Math.max(0, 1 - flash.t / 1.2);
        zonesEl.querySelectorAll('.mlm-zone').forEach((el) => (el.style.boxShadow = el.dataset.z === flash.z ? `0 0 0 ${(2 + 3 * a).toFixed(1)}px rgba(47,129,247,${(0.5 * a).toFixed(2)})` : ''));
        if (a <= 0) { flash = null; zonesEl.querySelectorAll('.mlm-zone').forEach((el) => (el.style.boxShadow = '')); }
      }
    });
  };

  /* ============ 6.3 mlmInputContract: pick bands by name, scale, clip ============ */
  AI.labs.mlmInputContract = function (root) {
    const STACK = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'];
    const { view, ctl } = AI.labShell(root, 'Band selection and scaling', 'the scene stores ten bands in file order; the record names the bands it needs',
      '<canvas class="mlm-cv"></canvas><div class="mlm-cap" data-o="cap"></div>',
      `<div class="mlm-field"><span>record bands (in order)</span><input type="text" data-f="bands" value="B02, B03, B04, B08" style="min-width:240px"></div>
       <div class="mlm-err" data-o="err"></div>
       <div class="mlm-field"><span>raw DN of the B08 pixel</span><input type="range" min="0" max="12000" step="50" value="12000" data-f="dn"><b data-o="dnv">12000</b></div>
       <pre class="mlm-pre" data-o="scale"></pre>`);
    const { cv, c } = fitCanvas(root);
    const cap = root.querySelector('[data-o=cap]'), errEl = root.querySelector('[data-o=err]'), scaleEl = root.querySelector('[data-o=scale]');
    const st = clock(ctl, 10, { loop: true, speed: 0.5 });
    let want = [];
    const stripes = STACK.map((n, j) => ({ n, j, x: 0, y: 0, tx: 0, ty: 0, init: false }));
    const parse = () => {
      const names = root.querySelector('[data-f=bands]').value.split(',').map((s) => s.trim()).filter(Boolean);
      want = names;
      const missing = names.filter((n) => !STACK.includes(n));
      errEl.textContent = missing.length ? `not in the scene stack: ${missing.join(', ')}` : '';
      root.querySelector('[data-o=dnv]').textContent = root.querySelector('[data-f=dn]').value;
      const dn = +root.querySelector('[data-f=dn]').value;
      const sc = dn * 0.0001, cl = Math.min(1, Math.max(0, sc));
      scaleEl.textContent = `step 1  scale 0.0001:  ${dn} × 0.0001 = ${sc.toFixed(4)}\nstep 2  clip [0, 1]:   ${sc.toFixed(4)} → ${cl.toFixed(4)}`;
      scaleEl.dataset.v = cl;
      st.lastKey = null;
    };
    root.querySelector('[data-f=bands]').addEventListener('input', parse);
    root.querySelector('[data-f=dn]').addEventListener('input', parse);
    parse();
    let barV = null;
    AI.loop(root, (dt) => {
      const W = c.w, H = c.h, g = c.ctx, C = AI.col();
      const x0 = 14, xL = W * 0.06, xR = W * 0.62, rowH = (H - 70) / 10, rowR = (H - 70) / 6;
      const cl = +scaleEl.dataset.v;
      if (barV === null) barV = cl;
      const k = Math.min(1, dt * 3);
      barV += (cl - barV) * k;
      let moving = Math.abs(cl - barV) > 0.002;
      // targets
      stripes.forEach((s) => {
        const sel = want.indexOf(s.n);
        if (!s.init) { s.x = xL; s.y = 40 + s.j * rowH; s.init = true; }
        if (sel >= 0 && sel < 6) { s.tx = xR; s.ty = 40 + sel * rowR; }
        else { s.tx = xL; s.ty = 40 + s.j * rowH; }
        const dx = s.tx - s.x, dy = s.ty - s.y;
        if (Math.abs(dx) > 0.3 || Math.abs(dy) > 0.3) moving = true;
        s.x += dx * k; s.y += dy * k;
      });
      st.advance(dt);
      if (!moving && st.lastKey !== null && !st.playing) return;
      st.lastKey = 'd';
      g.clearRect(0, 0, W, H);
      txt(g, 'scene file: 10 bands, stack order', xL, 26, C.muted);
      txt(g, 'model input: channels 0..3', xR, 26, C.muted);
      stripes.forEach((s) => {
        const sel = want.indexOf(s.n);
        const inR = sel >= 0 && sel < 6;
        const colr = inR ? C.green : C.border;
        const bw = inR ? W * 0.3 : W * 0.5, bh = (inR ? rowR : rowH) - 6;
        rr(g, s.x, s.y, bw, bh, 6, C.bg2, colr);
        txt(g, `${s.n}  pos ${s.j}`, s.x + 8, s.y + bh / 2 + 4, inR ? C.green : C.fg);
      });
      want.forEach((n, i) => { if (!STACK.includes(n) && i < 6) txt(g, `${n}: not in stack`, xR, 40 + i * rowR + rowR / 2, C.red); });
      // value bar
      const by = H - 26, bx = 14, bwid = W - 28;
      rr(g, bx, by, bwid, 12, 4, C.bg2, C.border);
      rr(g, bx, by, bwid * barV / 1.2, 12, 4, C.blue, null);
      g.fillStyle = C.fg; g.fillRect(bx + bwid / 1.2 - 1, by - 4, 2, 20);
      txt(g, 'pixel value (after scale and clip); marker = 1.0', bx, by - 6, C.muted);
      const sel = want.map((n) => STACK.indexOf(n)).filter((i) => i >= 0);
      const first = stripes.find((s) => s.n === want[0]);
      cap.textContent = want.length
        ? `Each band is found by name. ${want[0] ? want[0] + ' sits at file position ' + STACK.indexOf(want[0]) + ' and becomes channel 0.' : ''} The raw B08 value (${root.querySelector('[data-f=dn]').value}) is scaled and then clipped to [0, 1]; the bar shows the result.`
        : 'Type the bands the record names. Each one moves from its file position into its model channel.';
      void sel; void first;
    });
  };

  /* ============ 6.4 mlmRoles: asset roles wired to a runner ============ */
  AI.labs.mlmRoles = function (root) {
    const { view, ctl } = AI.labShell(root, 'Assets, roles and the runner', 'the runner finds each file by its role, never by its name',
      '<canvas class="mlm-cv" style="height:320px"></canvas><div class="mlm-cap" data-o="cap"></div>',
      `<div class="mlm-field"><span>model asset has role mlm:model</span><input type="checkbox" data-s="modelRole" checked></div>
       <div class="mlm-field"><span>model asset has mlm:artifact_type</span><input type="checkbox" data-s="artifact" checked></div>
       <div class="mlm-field"><span>weights asset present (role mlm:weights)</span><input type="checkbox" data-s="weights" checked></div>
       <div class="mlm-field"><span>source-code asset has role code</span><input type="checkbox" data-s="codeRole" checked></div>
       <div class="mlm-field"><span>source-code asset sets mlm:entrypoint</span><input type="checkbox" data-s="entry" checked></div>`);
    const { cv, c } = fitCanvas(root);
    const cap = root.querySelector('[data-o=cap]');
    const S = { modelRole: true, artifact: true, weights: true, codeRole: true, entry: true };
    root.querySelectorAll('[data-s]').forEach((el) => el.addEventListener('change', () => { S[el.dataset.s] = el.checked; st.lastKey = null; }));
    const st = clock(ctl, 10, { loop: true, speed: 0.5 });
    const checks = () => {
      const out = [];
      const hasModel = S.modelRole;
      out.push({ ok: hasModel, lvl: 'err', msg: hasModel ? 'found an asset with mlm:model: model loaded' : 'no asset has mlm:model: the runner cannot find a model' });
      if (hasModel) out.push({ ok: S.artifact, lvl: 'err', msg: S.artifact ? 'mlm:artifact_type present: the file type is known' : 'mlm:artifact_type missing: the runner cannot tell how to load the model' });
      out.push({ ok: !S.entry || S.codeRole, lvl: 'err', msg: !S.entry || S.codeRole ? 'entrypoint and code role agree' : 'mlm:entrypoint is set but no code role: the code role is mandatory' });
      out.push({ ok: S.weights, lvl: 'warn', msg: S.weights ? 'weights asset found and loaded into the architecture' : 'no weights asset: the model runs with random initial weights (a warning)' });
      return out;
    };
    AI.loop(root, (dt) => {
      const W = c.w, H = c.h, g = c.ctx, C = AI.col();
      st.advance(dt);
      const res = checks();
      const runnable = res.every((r) => r.ok || r.lvl === 'warn');
      const key = JSON.stringify(S) + '|' + Math.floor(st.t * 4) % 40 + '|' + (st.playing ? 'p' : 's');
      if (key === st.lastKey) return;
      st.lastKey = key;
      g.clearRect(0, 0, W, H);
      const ix = 10, iy = 10, iw = W * 0.46, ih = H - 20;
      rr(g, ix, iy, iw, ih, 12, null, C.border);
      txt(g, 'STAC Item (model record)', ix + 12, iy + 20, C.muted);
      const assets = [
        { k: 'model', y: iy + 40, roles: S.modelRole ? ['mlm:model'] : [], extra: S.artifact ? 'mlm:artifact_type: torch.save' : (S.modelRole ? 'mlm:artifact_type: (missing)' : ''), col: C.blue, on: true },
        { k: 'weights', y: iy + 40 + 96, roles: S.weights ? ['mlm:weights'] : [], extra: S.weights ? 'state dict, 470 KB' : '(asset absent)', col: C.purple, on: S.weights },
        { k: 'source_code', y: iy + 40 + 192, roles: S.codeRole ? ['mlm:source_code', 'code'] : ['mlm:source_code'], extra: S.entry ? 'mlm:entrypoint: infer.py:predict' : '', col: C.teal, on: true },
      ];
      const rx = W * 0.62, ry = H * 0.2, rw = W * 0.34, rh = H * 0.6;
      rr(g, rx, ry, rw, rh, 12, C.bg2, runnable ? C.green : C.red);
      txt(g, 'reference runner', rx + 12, ry + 22, C.fg);
      res.forEach((r, i) => txt(g, (r.ok ? '✔ ' : (r.lvl === 'warn' ? '⚠ ' : '✖ ')) + r.msg.slice(0, 44), rx + 12, ry + 48 + i * 40, r.ok ? C.green : r.lvl === 'warn' ? C.yellow : C.red, '11px ui-monospace,Menlo,Consolas,monospace'));
      assets.forEach((a) => {
        const h = 80;
        rr(g, ix + 14, a.y, iw - 28, h, 10, C.bg, a.on ? a.col : C.border);
        txt(g, a.k, ix + 26, a.y + 20, a.on ? C.fg : C.muted);
        txt(g, 'roles: [' + (a.roles.join(', ') || 'none') + ']', ix + 26, a.y + 40, a.roles.length ? a.col : C.red, '11px ui-monospace,Menlo,Consolas,monospace');
        txt(g, a.extra, ix + 26, a.y + 60, C.muted, '11px ui-monospace,Menlo,Consolas,monospace');
        // wire to runner
        const wy = a.y + h / 2;
        const linked = a.roles.length > 0 && (a.k !== 'model' || S.modelRole);
        g.beginPath(); g.moveTo(ix + iw - 14, wy); g.lineTo(rx, ry + rh / 2 + (a.k === 'model' ? -40 : a.k === 'weights' ? 0 : 40));
        g.setLineDash(linked ? [] : [4, 6]); g.strokeStyle = linked ? a.col : C.red; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
        if (linked && runnable) {
          for (let p = 0; p < 2; p++) {
            const u = (st.t * 0.22 + p * 0.5 + (a.k === 'weights' ? 0.2 : a.k === 'source_code' ? 0.4 : 0)) % 1;
            const x = AI.lerp(ix + iw - 14, rx, u), y = AI.lerp(wy, ry + rh / 2 + (a.k === 'model' ? -40 : a.k === 'weights' ? 0 : 40), u);
            g.beginPath(); g.arc(x, y, 4, 0, 7); g.fillStyle = a.col; g.fill();
          }
        }
      });
      const firstBad = res.find((r) => !r.ok && r.lvl === 'err') || res.find((r) => !r.ok);
      cap.textContent = firstBad
        ? `Stop: ${firstBad.msg}. Fix the record, not the runner.`
        : 'The runner reads roles, not file names: it loads the asset with mlm:model, the weights and the code it names, and then runs.';
    });
  };

  /* ============ 6.5 mlmMemory: parameters and bytes from the architecture ============ */
  AI.labs.mlmMemory = function (root) {
    const { view, ctl } = AI.labShell(root, 'Parameters and memory of the U-Net', 'change the width, patch, batch, bands and classes: the counts and bytes follow',
      `<canvas class="mlm-cv" style="height:250px"></canvas><div class="mlm-row" style="margin-top:8px" data-o="read"></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-field"><span>base width w</span><select data-s="w"><option>8</option><option selected>16</option><option>32</option></select></div>
       <div class="mlm-field"><span>patch side p (pixels)</span><select data-s="p"><option>128</option><option selected>256</option><option>512</option></select></div>
       <div class="mlm-field"><span>batch b</span><input type="range" min="1" max="16" step="1" value="4" data-s="b"><b data-o="bv">4</b></div>
       <div class="mlm-field"><span>input bands C</span><input type="range" min="1" max="13" step="1" value="4" data-s="C"><b data-o="Cv">4</b></div>
       <div class="mlm-field"><span>classes K</span><input type="range" min="2" max="12" step="1" value="6" data-s="K"><b data-o="Kv">6</b></div>`);
    const { cv, c } = fitCanvas(root);
    const read = root.querySelector('[data-o=read]'), cap = root.querySelector('[data-o=cap]');
    const S = {};
    const st = clock(ctl, 10, { loop: true, speed: 0.5 });
    const cur = { P: 0, A: 0, blocks: {} };
    const state = () => {
      S.w = +root.querySelector('[data-s=w]').value; S.p = +root.querySelector('[data-s=p]').value;
      S.b = +root.querySelector('[data-s=b]').value; S.C = +root.querySelector('[data-s=C]').value; S.K = +root.querySelector('[data-s=K]').value;
      root.querySelector('[data-o=bv]').textContent = S.b; root.querySelector('[data-o=Cv]').textContent = S.C; root.querySelector('[data-o=Kv]').textContent = S.K;
      const m = unet(S.C, S.w, S.K, S.p);
      S.m = m; S.wb = (m.P + m.B) * 4; S.ab = S.b * m.peak * 4; S.total = S.wb + S.ab;
      const match = S.w === 16 && S.p === 256 && S.b === 4 && S.C === 4 && S.K === 6;
      read.innerHTML = `<div class="mlm-card"><h4>mlm:total_parameters</h4><b>${m.P.toLocaleString('en-US')}</b></div>
        <div class="mlm-card"><h4>mlm:memory_size (bytes)</h4><b>${S.total.toLocaleString('en-US')}</b><div class="mlm-wait" style="font-size:12px">${(S.total / 1e6).toFixed(2)} MB · ${(S.total / 1048576).toFixed(2)} MiB</div></div>
        <div class="mlm-card"><h4>record check</h4><b class="${match ? 'mlm-ok' : 'mlm-warn'}">${match ? '✔ equals the record (117 590; 50 804 568)' : 'differs from the record claim'}</b></div>`;
      st.lastKey = null;
    };
    root.querySelectorAll('[data-s]').forEach((el) => el.addEventListener('input', state));
    root.querySelectorAll('[data-s]').forEach((el) => el.addEventListener('change', state));
    state();
    AI.loop(root, (dt) => {
      const W = c.w, H = c.h, g = c.ctx, C = AI.col();
      const k = Math.min(1, dt * 4);
      const names = ['enc1', 'enc2', 'bott', 'up1', 'dec1', 'up2', 'dec2', 'head'];
      let moving = false;
      names.forEach((n) => {
        const target = S.m.blocks[n] || 0;
        if (cur.blocks[n] === undefined) cur.blocks[n] = target;
        cur.blocks[n] += (target - cur.blocks[n]) * k;
        if (Math.abs(target - cur.blocks[n]) > 0.5) moving = true;
      });
      const tw = S.wb, ta = S.ab;
      cur.P += (tw - cur.P) * k; cur.A += (ta - cur.A) * k;
      if (Math.abs(tw - cur.P) > 1 || Math.abs(ta - cur.A) > 1) moving = true;
      st.advance(dt);
      if (!moving && st.lastKey !== null && !st.playing) return;
      st.lastKey = 'd';
      g.clearRect(0, 0, W, H);
      const maxB = Math.max(1, ...names.map((n) => cur.blocks[n]));
      const top = 30, bot = H - 60, bw = (W * 0.62) / names.length;
      names.forEach((n, i) => {
        const h = (cur.blocks[n] / maxB) * (bot - top);
        rr(g, 16 + i * bw + 4, bot - h, bw - 8, h, 4, C.blue, null);
        txt(g, n, 16 + i * bw + bw / 2, bot + 16, C.muted, '10px ui-monospace,Menlo,Consolas,monospace', 'center');
      });
      txt(g, 'trainable parameters per block', 16, 18, C.fg);
      // memory bar
      const mx = W * 0.68, mw = W * 0.28, mh = bot - top, tot = cur.P + cur.A || 1;
      const hw = (cur.P / tot) * mh, ha = (cur.A / tot) * mh;
      rr(g, mx, bot - hw, mw, hw, 4, C.purple, null);
      rr(g, mx, bot - hw - ha, mw, ha, 4, C.orange, null);
      txt(g, 'memory bytes', mx, 18, C.fg);
      txt(g, 'weights', mx + mw + 6, bot - hw / 2 + 4, C.purple);
      txt(g, 'activations', mx + mw + 6, bot - hw - ha / 2 + 4, C.orange);
      cap.textContent = `Weights cost 4 bytes per parameter and do not change with the patch or batch. Activations are ${S.b} × ${S.m.peak.toLocaleString('en-US')} numbers × 4 bytes: they grow with the patch area (p = ${S.p}) and the batch. Try p = 512 and watch the orange part grow.`;
    });
  };

  /* ============ 6.6 mlmTasks: from the question to the task value ============ */
  AI.labs.mlmTasks = function (root) {
    const GROUPS = [
      ['per input', ['scene-classification', 'classification']],
      ['per pixel', ['semantic-segmentation', 'segmentation']],
      ['per object or segment', ['instance-segmentation', 'panoptic-segmentation', 'object-detection', 'detection']],
      ['a number', ['regression']],
      ['similar places', ['similarity-search']],
      ['new image', ['generative', 'image-captioning', 'super-resolution', 'downscaling']],
    ];
    const LEAVES = GROUPS.flatMap(([, ls]) => ls);
    const Q = {
      'a label for a whole patch': ['scene-classification', 'classification'],
      'a class for every pixel': ['semantic-segmentation', 'segmentation'],
      'a separate mask for each object': ['instance-segmentation', 'panoptic-segmentation'],
      'a box around each object': ['object-detection', 'detection'],
      'a number per place': ['regression'],
      'places that look like this one': ['similarity-search'],
      'a new or sharper image': ['super-resolution', 'downscaling', 'generative', 'image-captioning'],
    };
    const MEAN = {
      'scene-classification': 'one label for a whole image patch (flooding yes/no)',
      classification: 'one label for an input (the general word)',
      'semantic-segmentation': 'a class for every pixel; objects of one class merge',
      segmentation: 'the general word for a per-pixel label map',
      'instance-segmentation': 'a separate mask per object (each building its own)',
      'panoptic-segmentation': 'countable objects as instances plus stuff classes (water, grass)',
      'object-detection': 'boxes around named objects, with class and score',
      detection: 'boxes around objects, with class and score',
      regression: 'a number per input (biomass, temperature)',
      'similarity-search': 'an embedding; results are nearby places in embedding space',
      generative: 'a new image produced by the model',
      'image-captioning': 'a text description of an image',
      'super-resolution': 'a sharper image from a coarser one',
      downscaling: 'a coarser field from a finer one (climate)',
    };
    const { view, ctl } = AI.labShell(root, 'Choosing the task value', 'pick the shape of the answer; the matching task values light up',
      '<canvas class="mlm-cv" style="height:340px"></canvas><div class="mlm-cap" data-o="cap"></div>',
      `<div class="mlm-field"><span>the model answers with</span><select data-o="q">${Object.keys(Q).map((k) => `<option>${k}</option>`).join('')}</select></div>
       <div class="mlm-field"><span>mlm:tasks would be</span><code data-o="out"></code></div>`);
    const { cv, c } = fitCanvas(root);
    const qSel = root.querySelector('[data-o=q]'), outEl = root.querySelector('[data-o=out]'), cap = root.querySelector('[data-o=cap]');
    const st = clock(ctl, 6, { loop: true, speed: 0.5 });
    let picked = null;
    cv.addEventListener('click', (e) => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = leafPos.find((p) => Math.hypot(p.x - x, p.y - y) < 22);
      if (hit) { picked = hit.n; st.lastKey = null; }
    });
    qSel.addEventListener('change', () => { picked = null; st.t = 0; st.lastKey = null; });
    let leafPos = [];
    AI.loop(root, (dt) => {
      const W = c.w, H = c.h, g = c.ctx, C = AI.col();
      st.advance(dt);
      const hi = new Set(Q[qSel.value]);
      outEl.textContent = [...hi].join(', ');
      const key = qSel.value + '|' + picked + '|' + Math.floor(st.t * 5);
      if (key === st.lastKey) return;
      st.lastKey = key;
      g.clearRect(0, 0, W, H);
      const rootX = W / 2, rootY = 28;
      const gpos = GROUPS.map(([name, ls], gi) => {
        const xs = ls.map((n) => LEAVES.indexOf(n) / (LEAVES.length - 1) * (W - 60) + 30);
        return { name, ls, x: xs.reduce((a, b) => a + b, 0) / xs.length, y: 92 };
      });
      leafPos = LEAVES.map((n, i) => ({ n, x: 30 + (i / (LEAVES.length - 1)) * (W - 60), y: H - 52 + (i % 2) * 20 }));
      const gOf = (n) => gpos.find((gp) => gp.ls.includes(n));
      // edges
      gpos.forEach((gp) => { g.beginPath(); g.moveTo(rootX, rootY + 10); g.lineTo(gp.x, gp.y - 10); g.strokeStyle = C.border; g.lineWidth = 1; g.stroke(); });
      leafPos.forEach((lp) => {
        const gp = gOf(lp.n);
        const on = hi.has(lp.n) || picked === lp.n;
        g.beginPath(); g.moveTo(gp.x, gp.y + 10); g.lineTo(lp.x, lp.y - 8); g.strokeStyle = on ? C.green : C.border; g.lineWidth = on ? 2 : 1; g.stroke();
      });
      // nodes
      rr(g, rootX - 110, rootY - 12, 220, 24, 8, C.bg2, C.fg);
      txt(g, 'what does the answer look like?', rootX, rootY + 4, C.fg, '12px ui-monospace,Menlo,Consolas,monospace', 'center');
      gpos.forEach((gp) => {
        const any = gp.ls.some((n) => hi.has(n));
        rr(g, gp.x - 80, gp.y - 10, 160, 20, 6, C.bg, any ? C.green : C.border);
        txt(g, gp.name, gp.x, gp.y + 4, any ? C.green : C.muted, '11px ui-monospace,Menlo,Consolas,monospace', 'center');
      });
      // moving pulses from root to each highlighted leaf
      [...hi].forEach((n, idx) => {
        const lp = leafPos.find((p) => p.n === n), gp = gOf(n);
        const u = ((st.t + idx * 0.35) % 2.4) / 2.4;
        const x = u < 0.5 ? AI.lerp(rootX, gp.x, u * 2) : AI.lerp(gp.x, lp.x, (u - 0.5) * 2);
        const y = u < 0.5 ? AI.lerp(rootY + 10, gp.y - 10, u * 2) : AI.lerp(gp.y + 10, lp.y - 8, (u - 0.5) * 2);
        g.beginPath(); g.arc(x, y, 4, 0, 7); g.fillStyle = C.green; g.fill();
      });
      leafPos.forEach((lp) => {
        const on = hi.has(lp.n) || picked === lp.n;
        txt(g, lp.n, lp.x, lp.y + 4, on ? C.green : C.muted, '10px ui-monospace,Menlo,Consolas,monospace', 'center');
      });
      const who = picked || [...hi][0];
      cap.innerHTML = `<b>${ESC(who)}</b>: ${ESC(MEAN[who])}. ${picked ? '' : 'Click a leaf to read its meaning.'}`;
    });
  };

  /* ============ 6.7 mlmCollection: summarise simple fields, leave contracts out ============ */
  AI.labs.mlmCollection = function (root) {
    const ITEMS = [
      { id: 'A', name: 'unet-landcover-s2-demo', tasks: ['semantic-segmentation'], fw: 'pytorch', params: 117590, mem: 50804568, on: true },
      { id: 'B', name: 'unet-tiny (illustrative)', tasks: ['semantic-segmentation'], fw: 'pytorch', params: 61522, mem: 27262976, on: true },
      { id: 'C', name: 'unet-landcover-kenya-v2 (fine-tune of A)', tasks: ['semantic-segmentation'], fw: 'pytorch', params: 117590, mem: 50804568, on: true },
    ];
    const { view, ctl } = AI.labShell(root, 'Collection summary', 'include or exclude items and edit their numbers; the summary is rebuilt from the items',
      `<div class="mlm-row" data-o="items"></div><div class="mlm-row" style="margin-top:10px" data-o="sum"></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-tl"><span style="font-size:12px;color:var(--muted)">Only simple fields are summarised. mlm:input, mlm:output and mlm:hyperparameters are excluded.</span></div>`);
    const itemsEl = root.querySelector('[data-o=items]'), sumEl = root.querySelector('[data-o=sum]'), cap = root.querySelector('[data-o=cap]');
    const st = clock(ctl, 6, { loop: false, t0: 6, speed: 0.6 });
    itemsEl.innerHTML = ITEMS.map((it, i) => `<div class="mlm-card"><h4>${ESC(it.id)}: ${ESC(it.name)}</h4>
        <div class="mlm-field"><span>include in collection</span><input type="checkbox" data-i="${i}" data-k="on" ${it.on ? 'checked' : ''}></div>
        <div class="mlm-field"><span>mlm:total_parameters</span><input type="text" data-i="${i}" data-k="params" value="${it.params}"></div>
        <div class="mlm-field"><span>mlm:memory_size</span><input type="text" data-i="${i}" data-k="mem" value="${it.mem}"></div>
        <div class="mlm-field"><span>mlm:tasks</span><code>${it.tasks.join(', ')}</code></div></div>`).join('');
    const restart = () => { st.t = 0; st.playing = true; st.setPlay(); };
    // edits update the data in place (no re-render, so typing is not interrupted)
    itemsEl.addEventListener('input', (e) => {
      const i = +e.target.dataset.i, k = e.target.dataset.k; if (isNaN(i)) return;
      if ((k === 'params' || k === 'mem') && isInt(e.target.value)) ITEMS[i][k] = +e.target.value;
      restart();
    });
    itemsEl.addEventListener('change', (e) => { const i = +e.target.dataset.i; if (e.target.dataset.k === 'on') { ITEMS[i].on = e.target.checked; restart(); } });
    AI.loop(root, (dt) => {
      st.advance(dt);
      const inc = ITEMS.filter((x) => x.on);
      if (!inc.length) { sumEl.innerHTML = '<span class="mlm-warn">Include at least one item.</span>'; return; }
      const tasks = [...new Set(inc.flatMap((x) => x.tasks))], fws = [...new Set(inc.map((x) => x.fw))];
      const pmin = Math.min(...inc.map((x) => x.params)), pmax = Math.max(...inc.map((x) => x.params));
      const mmin = Math.min(...inc.map((x) => x.mem)), mmax = Math.max(...inc.map((x) => x.mem));
      const steps = [
        ['mlm:tasks (set)', `[${tasks.join(', ')}]`],
        ['mlm:framework (set)', `[${fws.join(', ')}]`],
        ['mlm:total_parameters', `{ minimum: ${pmin.toLocaleString('en-US')}, maximum: ${pmax.toLocaleString('en-US')} }`],
        ['mlm:memory_size (bytes)', `{ minimum: ${mmin.toLocaleString('en-US')}, maximum: ${mmax.toLocaleString('en-US')} }`],
      ];
      const shown = Math.min(steps.length, Math.floor(st.t / 1.2) + 1);
      sumEl.innerHTML = `<div class="mlm-card"><h4>summaries</h4>${steps.slice(0, shown).map(([k, v]) => `<div class="mlm-field"><code>${ESC(k)}</code><b class="mlm-ok">${ESC(v)}</b></div>`).join('')}</div>
        <div class="mlm-card"><h4>not summarised (item-only)</h4><div class="mlm-field"><span class="mlm-strike">mlm:input</span><span class="mlm-wait">stays on each item</span></div><div class="mlm-field"><span class="mlm-strike">mlm:output</span><span class="mlm-wait">stays on each item</span></div><div class="mlm-field"><span class="mlm-strike">mlm:hyperparameters</span><span class="mlm-wait">stays on each item</span></div></div>`;
      cap.textContent = shown < steps.length
        ? `Summarising: ${steps[shown - 1][0]}. Simple values collapse into a set or a range.`
        : 'Done. The collection describes the family at a glance; each item keeps its own contract and numbers.';
    });
  };

  /* ============ 6.7 mlmLineage: derived_from links ============ */
  AI.labs.mlmLineage = function (root) {
    const N = [{ name: 'unet-landcover-s2-demo', parent: -1 }];
    const { view, ctl } = AI.labShell(root, 'Lineage with derived_from', 'each fine-tune links back to its base; click a model to see its link',
      `<canvas class="mlm-cv" style="height:300px"></canvas><pre class="mlm-pre" data-o="link" style="margin-top:8px"></pre><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-field"><span>new fine-tune name</span><input type="text" data-o="nm" value="unet-landcover-sahel-v1"></div>
       <div class="mlm-field"><span>derived from</span><select data-o="par"></select></div>
       <div class="mlm-tl"><button class="btn small" data-a="add">Add fine-tune</button><button class="btn small" data-a="del">Remove last</button></div>`);
    const { cv, c } = fitCanvas(root);
    const linkEl = root.querySelector('[data-o=link]'), cap = root.querySelector('[data-o=cap]'), parSel = root.querySelector('[data-o=par]');
    const st = clock(ctl, 100, { loop: true, speed: 0.5 });
    let sel = 0;
    const seed = () => {
      N.length = 1;
      N.push({ name: 'unet-landcover-kenya-v2', parent: 0 });
      N.push({ name: 'unet-cropland-kenya-v2b', parent: 1 });
      N.push({ name: 'unet-landcover-sahel-v1', parent: 0 });
    };
    seed();
    const refreshSel = () => { parSel.innerHTML = N.map((n, i) => `<option value="${i}">${ESC(n.name)}</option>`).join(''); };
    refreshSel();
    root.querySelector('[data-a=add]').onclick = () => {
      const name = root.querySelector('[data-o=nm]').value.trim() || 'unet-fine-tune';
      N.push({ name, parent: +parSel.value }); sel = N.length - 1; refreshSel(); st.lastKey = null;
    };
    root.querySelector('[data-a=del]').onclick = () => { if (N.length > 1) { N.pop(); sel = Math.min(sel, N.length - 1); refreshSel(); st.lastKey = null; } };
    cv.addEventListener('click', (e) => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = pos.findIndex((p) => Math.abs(p.x - x) < 90 && Math.abs(p.y - y) < 16);
      if (hit >= 0) { sel = hit; st.lastKey = null; }
    });
    let pos = [];
    AI.loop(root, (dt) => {
      const W = c.w, H = c.h, g = c.ctx, C = AI.col();
      st.advance(dt);
      const depth = N.map((n, i) => (n.parent < 0 ? 0 : depth_(i) ));
      function depth_(i) { let d = 0, p = N[i].parent; while (p >= 0) { d++; p = N[p].parent; } return d; }
      const maxD = Math.max(...depth);
      const byLevel = {};
      N.forEach((n, i) => { (byLevel[depth[i]] = byLevel[depth[i]] || []).push(i); });
      pos = N.map((n, i) => {
        const lv = byLevel[depth[i]], k = lv.indexOf(i);
        return { x: ((k + 0.5) / lv.length) * (W - 40) + 20, y: 34 + (depth[i] / Math.max(1, maxD)) * (H - 70) };
      });
      const key = JSON.stringify(N.map((n) => [n.name, n.parent])) + sel + '|' + Math.floor(st.t * 6);
      if (key === st.lastKey) return;
      st.lastKey = key;
      g.clearRect(0, 0, W, H);
      N.forEach((n, i) => {
        if (n.parent < 0) return;
        const a = pos[i], b = pos[n.parent];
        const my = (a.y + b.y) / 2;
        g.beginPath(); g.moveTo(a.x, a.y - 14); g.bezierCurveTo(a.x, my, b.x, my, b.x, b.y + 14);
        g.strokeStyle = i === sel ? C.green : C.border; g.lineWidth = 2; g.setLineDash([6, 6]);
        g.lineDashOffset = -st.t * 20; g.stroke(); g.setLineDash([]);
        AI.arrow(g, b.x, b.y + 14, b.x + (a.x - b.x) * 0.02, b.y + 20, i === sel ? C.green : C.muted, 1.5);
      });
      N.forEach((n, i) => {
        const p = pos[i], on = i === sel;
        const w = Math.min(200, n.name.length * 6.6 + 20);
        rr(g, p.x - w / 2, p.y - 14, w, 28, 8, C.bg2, on ? C.green : (n.parent < 0 ? C.blue : C.border));
        txt(g, n.name, p.x, p.y + 4, on ? C.green : C.fg, '11px ui-monospace,Menlo,Consolas,monospace', 'center');
      });
      const s = N[sel];
      const link = s.parent < 0
        ? '// root model: no derived_from link (it is the base)'
        : JSON.stringify({ rel: 'derived_from', href: `https://example.org/stac/models/${N[s.parent].name}.json`, type: 'application/json', title: `Base model: ${N[s.parent].name}` }, null, 2);
      linkEl.textContent = `${s.name}\n${link}`;
      cap.textContent = s.parent < 0
        ? 'The base model has no derived_from link. Its own record is the root of the lineage.'
        : `${s.name} points to its base with rel = derived_from. Walk the links to reach the root and read its licence and training data there.`;
    });
  };

  /* ============ 6.8 mlmRun: the runner reads the record field by field, tiles, paints ============ */
  AI.labs.mlmRun = function (root) {
    const W = 24, Hh = 16;
    const FIELDS = [
      ['stac_extensions', '["…/mlm/v1.6.0/schema.json"]', 'Check the MLM schema is declared, so these fields are known.'],
      ['mlm:input.bands', '["B02","B03","B04","B08"]', 'Select the bands by name, in this order.'],
      ['value_scaling', '[scale 0.0001, clip 0..1]', 'Apply in order: divide by 10 000, then clip.'],
      ['input.shape', '[-1, 4, P, P]', 'Patch side P (256 in the record; the demo uses a small P). -1 is the batch.'],
      ['resize_type', '"pad"', 'Pad edge patches with no-data. Never resample the label grid.'],
      ['batch_size_suggestion', '4', 'Send patches to the model in groups of 4.'],
      ['post_processing', 'argmax over class axis', 'Pick the highest-scoring class per pixel; invalid pixels become 255.'],
      ['assets: mlm:model', '"unet_landcover_full.pt"', 'Load the file with the mlm:model role, then run.'],
    ];
    const { view, ctl } = AI.labShell(root, 'Running a model from its record', 'the demo scene is 24 × 16 cells; the record is read first, then tiles are cut and painted',
      `<div class="mlm-row"><div style="flex:1;min-width:200px"><pre class="mlm-pre" data-o="rec"></pre></div><div style="flex:1;min-width:200px"><canvas class="mlm-cv" style="height:260px"></canvas></div></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-field"><span>demo patch side P (cells)</span><select data-o="P"><option>6</option><option selected>8</option><option>12</option></select></div>
       <div class="mlm-field"><span>stride</span><select data-o="s"><option value="full">= P (no overlap)</option><option value="half">= P/2 (50 % overlap)</option></select></div>`);
    const { cv, c } = fitCanvas(root);
    const recEl = root.querySelector('[data-o=rec]'), cap = root.querySelector('[data-o=cap]');
    const st = clock(ctl, 1, { loop: false, speed: 0.6 });
    const CL = (x, y) => {
      if ((x - 6) ** 2 + (y - 5) ** 2 <= 6) return 0;
      if (y < 5) return 1;
      if (x > 15 && y > 9 && y < 14) return 4;
      if (x % 8 < 4 && y >= 5 && y < 11) return 3;
      return 2;
    };
    let plan = null;
    const makePlan = () => {
      const P = +root.querySelector('[data-o=P]').value, half = root.querySelector('[data-o=s]').value === 'half';
      const s = half ? Math.max(1, P / 2) : P;
      const nx = Math.ceil((W - P) / s) + 1, ny = Math.ceil((Hh - P) / s) + 1;
      const patches = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) patches.push({ x: i * s, y: j * s });
      const batches = [];
      for (let k = 0; k < patches.length; k += 4) batches.push(patches.slice(k, k + 4));
      plan = { P, s, nx, ny, patches, batches, tRead: 8 * 0.9, tTile: 1.6 };
      st.dur = plan.tRead + plan.batches.length * plan.tTile + 3;
      st.t = 0; st.playing = true; st.setPlay(); st.lastKey = null;
    };
    root.querySelectorAll('select').forEach((el) => el.addEventListener('change', makePlan));
    makePlan();
    recEl.innerHTML = FIELDS.map(([k, v], i) => `<span class="ln" data-i="${i}">  ${ESC(k)}: ${ESC(v)}</span>`).join('\n');
    AI.loop(root, (dt) => {
      const moved = st.advance(dt);
      const g = c.ctx, Cc = AI.col(), w = c.w, h = c.h;
      const t = st.t;
      const readIdx = t < plan.tRead ? Math.floor(t / 0.9) : 8;
      const batchesDone = t < plan.tRead ? 0 : Math.min(plan.batches.length, Math.floor((t - plan.tRead) / plan.tTile) + 1);
      const curBatch = t < plan.tRead ? -1 : Math.min(plan.batches.length - 1, Math.floor((t - plan.tRead) / plan.tTile));
      const key = `${readIdx}|${batchesDone}|${curBatch}|${plan.P}|${plan.s}`;
      if (!moved && key === st.lastKey) return;
      st.lastKey = key;
      recEl.querySelectorAll('.ln').forEach((el, i) => el.classList.toggle('on', i === readIdx));
      g.clearRect(0, 0, w, h);
      const cw = (w - 20) / W, ch = (h - 40) / Hh, ox = 10, oy = 26;
      // painted prediction so far
      for (let y = 0; y < Hh; y++) for (let x = 0; x < W; x++) {
        let painted = false;
        for (let b = 0; b < batchesDone; b++) {
          for (const p of plan.batches[b]) if (x >= p.x && x < p.x + plan.P && y >= p.y && y < p.y + plan.P) { painted = true; break; }
          if (painted) break;
        }
        const k = CL(x, y), palette = [Cc.blue, Cc.green, Cc.teal, Cc.yellow, Cc.muted];
        g.fillStyle = painted ? palette[k] : Cc.bg2;
        g.globalAlpha = painted ? 0.9 : 1;
        g.fillRect(ox + x * cw, oy + y * ch, cw - 0.5, ch - 0.5);
        g.globalAlpha = 1;
      }
      // the current batch outline
      if (curBatch >= 0 && t >= plan.tRead) {
        plan.batches[curBatch].forEach((p) => {
          g.strokeStyle = Cc.orange; g.lineWidth = 2;
          g.strokeRect(ox + p.x * cw, oy + p.y * ch, Math.min(plan.P, W - p.x) * cw, Math.min(plan.P, Hh - p.y) * ch);
        });
      }
      txt(g, t < plan.tRead ? 'reading the record…' : `batch ${Math.max(1, curBatch + 1)} of ${plan.batches.length}`, ox, 16, Cc.fg);
      let msg;
      if (t < plan.tRead) msg = `Reading the record, field ${readIdx + 1} of 8: ${FIELDS[readIdx][2]}`;
      else if (batchesDone < plan.batches.length) msg = `Batch ${curBatch + 1}: ${plan.batches[curBatch].length} patches of ${plan.P}×${plan.P} cells cut from the scene (edge patches padded), run through the model, and the argmax written back to their place (orange outline).`;
      else msg = `All ${plan.patches.length} patches are predicted. Stitched on the original grid, the map has the scene's shape. Overlapping cells keep the last prediction in this demo; a real runner blends them.`;
      cap.textContent = msg;
    });
  };

  /* ============ 6.9 mlmValidate: claims against the artifact ============ */
  AI.labs.mlmValidate = function (root) {
    const { view, ctl } = AI.labShell(root, 'Checking the record against the artifact', 'each claim is measured on the artifact and compared with the tolerance',
      `<div data-o="tbl"></div><div class="mlm-cap" data-o="cap"></div>`,
      `<div class="mlm-field"><span>claim: mlm:total_parameters</span><input type="text" data-c="params" value="117590"></div>
       <div class="mlm-field"><span>claim: mlm:memory_size (bytes)</span><input type="text" data-c="mem" value="50804568"></div>
       <div class="mlm-field"><span>claim: bands in the record</span><input type="text" data-c="bands" value="B02, B03, B04, B08"></div>
       <div class="mlm-field"><span>runtime overhead on the GPU (%)</span><input type="range" min="0" max="30" step="1" value="8" data-c="ovh"><b data-o="ovhv">8</b></div>
       <div class="mlm-field"><span>tolerance for memory (%)</span><input type="range" min="0" max="50" step="1" value="15" data-c="tol"><b data-o="tolv">15</b></div>
       <div class="mlm-field"><span>record lists tasks and all required fields</span><input type="checkbox" data-c="schema" checked></div>`);
    const tbl = root.querySelector('[data-o=tbl]'), cap = root.querySelector('[data-o=cap]');
    const st = clock(ctl, 7, { loop: false, t0: 7, speed: 0.6 });
    const ART = { trainedBands: ['B02', 'B03', 'B04', 'B08'], K: 6, p: 256, C: 4, w: 16, b: 4 };
    const m = unet(ART.C, ART.w, ART.K, ART.p);
    const measuredMem = () => memBytes(m, ART.b) * (1 + +root.querySelector('[data-c=ovh]').value / 100);
    const rows = () => {
      const val = (k) => root.querySelector(`[data-c=${k}]`).value.trim();
      const memC = +val('mem'), mm = measuredMem(), tol = +root.querySelector('[data-c=tol]').value / 100;
      const bandsClaim = val('bands').split(',').map((s) => s.trim()).filter(Boolean);
      const paramsClaim = +val('params');
      return [
        { id: 'schema', label: 'schema: required fields and enumerations', claim: root.querySelector('[data-c=schema]').checked ? 'all present' : 'mlm:tasks missing', meas: 'validator run', ok: root.querySelector('[data-c=schema]').checked, why: 'The schema check reads the record alone; it cannot see the weights.' },
        { id: 'params', label: 'mlm:total_parameters (exact)', claim: paramsClaim.toLocaleString('en-US'), meas: m.P.toLocaleString('en-US'), ok: paramsClaim === m.P, why: 'Counted from the weights with the same convention as the record (trainable and non-trainable tensors).' },
        { id: 'out', label: 'output channels and patch (exact)', claim: `K=6, ${ART.p}×${ART.p}`, meas: `K=${ART.K}, ${ART.p}×${ART.p}`, ok: true, why: 'A dummy input of shape (1,4,256,256) gives an output of (1,6,256,256): the contract holds.' },
        { id: 'mem', label: 'mlm:memory_size (within tolerance)', claim: memC.toLocaleString('en-US'), meas: Math.round(mm).toLocaleString('en-US'), ok: Math.abs(mm - memC) / memC <= tol, why: `Measured peak = estimate × (1 + overhead). Gap ${((mm - memC) / memC * 100).toFixed(1)} % against a ${Math.round(tol * 100)} % tolerance.` },
        { id: 'bands', label: 'band order (exact)', claim: bandsClaim.join(', '), meas: ART.trainedBands.join(', '), ok: bandsClaim.join(',') === ART.trainedBands.join(','), why: 'A scrambled band list gives confident nonsense; this is checked on a sample with known labels in practice.' },
      ];
    };
    const draw = (idx) => {
      const rs = rows();
      tbl.innerHTML = `<table class="mlm-tbl"><tr><th>check</th><th>claim</th><th>artifact</th><th>status</th></tr>${rs.map((r, i) => {
        let s;
        if (i < idx) s = r.ok ? '<b class="mlm-ok">✔ pass</b>' : '<b class="mlm-bad">✖ fail</b>';
        else if (i === idx) s = '<b class="mlm-wait">checking…</b>';
        else s = '<span class="mlm-wait">queued</span>';
        return `<tr><td>${ESC(r.label)}</td><td><code>${ESC(r.claim)}</code></td><td><code>${ESC(i < idx || i === idx ? r.meas : '…')}</code></td><td>${s}</td></tr>`;
      }).join('')}</table>`;
      return rs;
    };
    root.querySelectorAll('[data-c]').forEach((el) => el.addEventListener('input', () => {
      root.querySelector('[data-o=ovhv]').textContent = root.querySelector('[data-c=ovh]').value;
      root.querySelector('[data-o=tolv]').textContent = root.querySelector('[data-c=tol]').value;
      st.t = 0; st.playing = true; st.setPlay(); st.lastKey = null;
    }));
    AI.loop(root, (dt) => {
      st.advance(dt);
      const idx = Math.min(5, Math.floor(st.t / 1.1));
      const key = idx + '|' + Array.from(root.querySelectorAll('[data-c]')).map((el) => el.type === 'checkbox' ? el.checked : el.value).join('|');
      if (key === st.lastKey) return;
      st.lastKey = key;
      const rs = draw(idx);
      if (idx < 5) {
        cap.textContent = `Checking: ${rs[idx].label}. ${rs[idx].why}`;
      } else {
        const bad = rs.filter((r) => !r.ok);
        cap.innerHTML = bad.length
          ? `<b class="mlm-bad">Refuse the record</b> until these are fixed: ${bad.map((r) => ESC(r.label)).join('; ')}. A claim that fails is not a small detail: the planner would buy the wrong hardware or trust a wrong contract.`
          : `<b class="mlm-ok">Trusted for these checks.</b> Accuracy still needs its own report: test region, split and per-class scores (Module 3 and Module 8).`;
      }
    });
  };
})();
