/* Data Science Module 1 (Data wrangling & cleaning) labs.
   ds-wr-filter, ds-wr-join, ds-wr-missing, ds-wr-dupes, ds-wr-range, ds-wr-parse, ds-wr-group.
   Rows arrive one by one on a dt-based clock (slow by default); every lab has speed, play/pause, step and reset.
   All data are tiny illustrative tables printed in the lessons. */
(function () {
  const q = (r, s) => r.querySelector(s);
  const clamp01 = (t) => Math.max(0, Math.min(1, t));
  const ease = (t) => { t = clamp01(t); return t * t * (3 - 2 * t); };
  const lerp = (a, b, t) => a + (b - a) * t;
  const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
  const median = (a) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const f2 = (v) => (Number.isFinite(v) ? String(+v.toFixed(2)) : '–');
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

  const COMMON = `<label>Speed <span class="v" id="spv"></span><input type="range" id="sp" min="0.2" max="3" step="0.1" value="0.8"></label>
    <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">Reset</button></div>`;

  /** Shared clock: st.t counts "rows" (one row per unit). draw(t) is called only when time advances or state changes. */
  function runner(root, ctl, getTotal, draw) {
    const st = { t: 0, play: true, dirty: true };
    const sp = q(ctl, '#sp'), spv = q(ctl, '#spv'), pb = q(ctl, '#play');
    const lbl = () => { pb.textContent = st.play ? '⏸ Pause' : '▶ Play'; };
    sp.oninput = () => { spv.textContent = (+sp.value).toFixed(1) + ' rows/s'; };
    pb.onclick = () => { if (!st.play && st.t >= getTotal()) st.t = 0; st.play = !st.play; lbl(); st.dirty = true; };
    q(ctl, '#step').onclick = () => { st.play = false; lbl(); const T = getTotal(); st.t = st.t >= T ? 0 : Math.min(T, Math.floor(st.t + 1e-9) + 1); st.dirty = true; };
    q(ctl, '#reset').onclick = () => { st.t = 0; st.play = false; lbl(); st.dirty = true; };
    st.restart = () => { st.t = 0; st.play = true; lbl(); st.dirty = true; };
    AI.loop(root, (dt) => {
      if (st.play) {
        const T = getTotal();
        st.t = Math.min(T, st.t + dt * +sp.value);
        if (st.t >= T) { st.play = false; lbl(); }
        draw(st.t); st.dirty = false;
      } else if (st.dirty) { draw(st.t); st.dirty = false; }
    }, { fps: 30 });
    sp.oninput(); lbl();
    return st;
  }

  const text = (g, s, x, y, col, font, al) => { g.fillStyle = col; g.font = font || '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = al || 'left'; g.fillText(s, x, y); };
  const cut = (g, s, w) => { s = String(s); if (g.measureText(s).width <= w) return s; while (s.length > 1 && g.measureText(s + '…').width > w) s = s.slice(0, -1); return s + '…'; };
  const cellText = (g, s, x, y, w, col, al) => {
    g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.fillStyle = col; g.textAlign = al || 'left';
    g.fillText(cut(g, s, w - 4), al === 'right' ? x + w - 4 : x + 3, y);
  };
  const fillA = (g, col, a) => { g.globalAlpha = a; g.fillStyle = col; };
  const box = (g, x, y, w, h, col, a) => { g.save(); g.globalAlpha = a; g.fillStyle = col; g.fillRect(x, y, w, h); g.restore(); };

  /* ======================================================================
     1. Filter, select and sort
     ====================================================================== */
  AI.labs['ds-wr-filter'] = function (root) {
    const COLS = ['id', 'city', 'item', 'qty', 'price', 'total'];
    const NUMC = { id: 1, qty: 1, price: 1, total: 1 };
    const RAW = [[1, 'Cairo', 'Tea', 2, 4.5], [2, 'Giza', 'Coffee', 1, 3], [3, 'Cairo', 'Cake', 3, 5], [4, 'Luxor', 'Tea', 4, 4.5], [5, 'Cairo', 'Coffee', 2, 3],
      [6, 'Giza', 'Cake', 1, 5], [7, 'Cairo', 'Latte', 2, 6], [8, 'Luxor', 'Coffee', 5, 3], [9, 'Cairo', 'Tea', 1, 4.5], [10, 'Giza', 'Latte', 3, 6]];
    const ROWS = RAW.map((r) => ({ id: r[0], city: r[1], item: r[2], qty: r[3], price: r[4], total: r[3] * r[4] }));
    const N = ROWS.length;
    const { view, ctl } = AI.labShell(root, 'Filter, select and sort', 'a boolean mask decides which rows survive; the original table is never changed (illustrative data)',
      `<canvas style="height:470px"></canvas>`,
      `<label>Keep rows where… (join tests with and / or)<input type="text" id="cond" value="city = Cairo and total > 8" spellcheck="false"></label><div class="err" id="err"></div>
       <div class="legend" style="gap:8px">${COLS.map((c) => `<label style="flex-direction:row;gap:4px;align-items:center"><input type="checkbox" class="colchk" value="${c}" checked>${c}</label>`).join('')}</div>
       <div class="row"><select id="sk" style="flex:1"><option value="">no sort</option>${COLS.map((c) => `<option value="${c}">sort by ${c}</option>`).join('')}</select>
       <select id="sd" style="flex:1"><option value="1">ascending</option><option value="-1">descending</option></select></div>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const cond = q(ctl, '#cond'), err = q(ctl, '#err'), out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let test = () => true, condSrc = '', mask = [], slot = [], keptN = 0, st;

    const atom = (s) => {
      const m = /^\s*(\w+)\s*(>=|<=|!=|==|=|>|<)\s*(.+?)\s*$/.exec(s);
      if (!m) throw new Error('Write each test as: column operator value, e.g. qty >= 2');
      const col = m[1];
      if (!COLS.includes(col)) throw new Error('Unknown column "' + col + '". Columns: ' + COLS.join(', '));
      let v = m[3].replace(/^['"]|['"]$/g, '');
      const op = m[2] === '=' ? '==' : m[2];
      if (NUMC[col]) { v = +v; if (!Number.isFinite(v)) throw new Error(col + ' is a number column'); }
      else if (!['==', '!='].includes(op)) throw new Error('Text columns support = and != only');
      else v = v.toLowerCase();
      return (r) => { const x = NUMC[col] ? r[col] : String(r[col]).toLowerCase(); return op === '==' ? x === v : op === '!=' ? x !== v : op === '>' ? x > v : op === '<' ? x < v : op === '>=' ? x >= v : x <= v; };
    };
    const parse = (src) => {
      if (!src.trim()) return () => true;
      const ors = src.split(/\s+or\s+/i).map((p) => p.split(/\s+and\s+/i).map(atom));
      return (r) => ors.some((ands) => ands.every((f) => f(r)));
    };
    const plan = () => {
      mask = ROWS.map((r) => test(r));
      let idx = ROWS.map((_, i) => i).filter((i) => mask[i]);
      const k = q(ctl, '#sk').value, d = +q(ctl, '#sd').value;
      if (k) idx.sort((a, b) => { const x = ROWS[a][k], y = ROWS[b][k]; return (x < y ? -1 : x > y ? 1 : a - b) * d || a - b; });
      slot = new Array(N).fill(-1); idx.forEach((ri, s) => (slot[ri] = s)); keptN = idx.length;
    };
    const sel = () => [...ctl.querySelectorAll('.colchk')].filter((x) => x.checked).map((x) => x.value);
    const change = () => {
      try { test = parse(cond.value); condSrc = cond.value; cond.classList.remove('bad'); err.textContent = ''; }
      catch (e) { cond.classList.add('bad'); err.textContent = e.message; return; }
      plan(); st.restart();
    };
    cond.addEventListener('input', change);
    ctl.querySelectorAll('.colchk,#sk,#sd').forEach((e) => e.addEventListener('change', change));

    const RH = 18, Y0 = 44, YR = 268;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const x0 = 34, tw = W - x0 - 8, cw = tw / COLS.length;
      const colX = (k) => x0 + k * cw;
      const sc = sel(), rcw = Math.min(90, tw / Math.max(1, sc.length)), rX = (k) => x0 + k * rcw;
      text(g, 'original table: ' + N + ' rows, never modified', 8, 16, c.muted, 'bold 12px sans-serif');
      box(g, x0, 24, tw, 16, c.bg3, 1);
      COLS.forEach((cn, k) => cellText(g, cn, colX(k), 36, cw, c.muted, NUMC[cn] ? 'right' : 'left'));
      text(g, 'mask', 4, 36, c.muted, '10px sans-serif');
      for (let i = 0; i < N; i++) {
        const p = t - i, y = Y0 + i * RH;
        const decided = p >= 0.3, rej = decided && !mask[i];
        if (p >= 0 && p < 1) box(g, 0, y - 12, W, RH, c.blue, 0.14);
        g.save(); g.globalAlpha = rej ? lerp(1, 0.35, ease((p - 0.3) / 0.2)) : 1;
        COLS.forEach((cn, k) => cellText(g, ROWS[i][cn], colX(k), y, cw, c.fg, NUMC[cn] ? 'right' : 'left'));
        g.restore();
        if (decided) text(g, mask[i] ? '✓' : '✗', 10, y, mask[i] ? c.green : c.red, 'bold 12px sans-serif');
        g.strokeStyle = c.border; g.beginPath(); g.moveTo(x0, y + 5.5); g.lineTo(x0 + tw, y + 5.5); g.stroke();
      }
      text(g, 'result: select ' + (sc.length ? sc.join(', ') : '(no columns)') + (q(ctl, '#sk').value ? ', sorted by ' + q(ctl, '#sk').value : ''), 8, YR - 32, c.muted, 'bold 12px sans-serif');
      box(g, x0, YR - 24, tw, 16, c.bg3, 1);
      sc.forEach((cn, k) => cellText(g, cn, rX(k), YR - 12, rcw, c.muted, NUMC[cn] ? 'right' : 'left'));
      for (let i = 0; i < N; i++) {
        if (!mask[i]) continue;
        const p = t - i, s = ease((p - 0.4) / 0.6);
        if (s <= 0) continue;
        const yo = Y0 + i * RH, yt = YR + slot[i] * RH, y = lerp(yo, yt, s);
        box(g, x0, y - 12, tw, RH, c.green, 0.12 * (1 - 0.3 * s));
        sc.forEach((cn, k) => cellText(g, ROWS[i][cn], lerp(colX(COLS.indexOf(cn)), rX(k), s), y, s < 1 ? cw : rcw, c.fg, NUMC[cn] ? 'right' : 'left'));
      }
      const done = t >= N - 1e-6, r = Math.min(N - 1, Math.floor(t));
      out.innerHTML = `condition: <span class="b">${esc(condSrc || '(keep all)')}</span>\nrows kept: <span class="g">${done ? keptN : mask.slice(0, Math.floor(t + 0.3)).filter(Boolean).length}</span> of ${N}\nraw table still has <span class="o">${N}</span> rows`;
      cap.textContent = done ? `Done. ${keptN} of ${N} rows passed the mask; the file you started with is unchanged (illustrative).`
        : `Row id ${ROWS[r].id}: mask = ${mask[r] ? 'True, so a copy flies into the result' : 'False, so the row is left behind'}.`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    test = parse(cond.value); condSrc = cond.value;
    plan(); st = runner(root, ctl, () => N, draw);
  };

  /* ======================================================================
     2. Joins
     ====================================================================== */
  AI.labs['ds-wr-join'] = function (root) {
    const AMT = [20, 35, 15, 40, 25, 30, 10, 45];
    const CUST = [['C1', 'Amal'], ['C2', 'Omar'], ['C3', 'Lina'], ['C4', 'Sami'], ['C5', 'Nour']];
    const { view, ctl } = AI.labShell(root, 'A join matching keys', 'each order looks up its key in the customers table (illustrative data)',
      `<canvas style="height:440px"></canvas>`,
      `<label>Order keys (comma separated, up to 8)<input type="text" id="keys" value="C1,C2,C2,C3,C6,C1,C3" spellcheck="false"></label>
       <label>Join type<select id="jt"><option value="inner">inner: only matched rows</option><option value="left">left: keep every order</option><option value="outer">outer: keep everything</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="dup">customers lists C2 twice (a duplicate key)</label>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let keys = [], custs = [], rows = [], extra = [], st;
    const build = () => {
      const ks = q(ctl, '#keys').value.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean).slice(0, 8);
      keys = ks.length ? ks : ['C1'];
      custs = CUST.slice(); if (q(ctl, '#dup').checked) custs.push(['C2', 'Omar*']);
      const type = q(ctl, '#jt').value; rows = []; const used = new Set();
      keys.forEach((k, i) => {
        const m = custs.map((cc, j) => (cc[0] === k ? j : -1)).filter((j) => j >= 0);
        m.forEach((j) => used.add(j));
        if (m.length) m.forEach((j, n) => rows.push({ oi: i, ci: j, ph: i + 0.4 + 0.3 * Math.min(n, 1) }));
        else if (type !== 'inner') rows.push({ oi: i, ci: -1, ph: i + 0.5 });
      });
      extra = [];
      if (type === 'outer') custs.forEach((cc, j) => { if (!used.has(j)) { extra.push(j); rows.push({ oi: -1, ci: j, ph: keys.length + extra.length - 1 + 0.5 }); } });
    };
    const total = () => keys.length + extra.length;
    const change = () => { build(); st.restart(); };
    ctl.querySelectorAll('#keys,#jt,#dup').forEach((e) => e.addEventListener(e.id === 'keys' ? 'input' : 'change', change));

    const RH = 16;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const ox = 8, ow = [34, 40, 40], cx = W - 8 - 120, cwid = [40, 80];
      text(g, 'orders (left)', ox, 14, c.muted, 'bold 12px sans-serif'); text(g, 'customers (right)', cx, 14, c.muted, 'bold 12px sans-serif');
      box(g, ox, 20, 114, 16, c.bg3, 1); ['oid', 'cid', 'amt'].forEach((h, k) => cellText(g, h, ox + ow.slice(0, k).reduce((a, b) => a + b, 0), 32, ow[k], c.muted));
      box(g, cx, 20, 120, 16, c.bg3, 1); ['cid', 'name'].forEach((h, k) => cellText(g, h, cx + cwid.slice(0, k).reduce((a, b) => a + b, 0), 32, cwid[k], c.muted));
      const oy = (i) => 50 + i * RH, cy = (j) => 50 + j * RH;
      const cur = Math.floor(t), pc = t - cur, n = keys.length;
      const hot = new Set();
      if (t < n) rows.filter((r) => r.oi === cur).forEach((r) => r.ci >= 0 && hot.add(r.ci));
      else if (t < total()) hot.add(extra[cur - n]);
      keys.forEach((k, i) => {
        if (t >= i && t < i + 1) box(g, 0, oy(i) - 12, ox + 114, RH, c.blue, 0.16);
        cellText(g, 'o' + (i + 1), ox, oy(i), ow[0], c.fg); cellText(g, k, ox + ow[0], oy(i), ow[1], c.fg); cellText(g, AMT[i], ox + ow[0] + ow[1], oy(i), ow[2], c.fg);
      });
      custs.forEach((cc, j) => {
        if (hot.has(j)) box(g, cx, cy(j) - 12, 120, RH, c.orange, 0.22);
        cellText(g, cc[0], cx, cy(j), cwid[0], c.fg); cellText(g, cc[1], cx + cwid[0], cy(j), cwid[1], c.fg);
      });
      // key lookup lines for the current order
      if (t < n) {
        const i = cur, ms = rows.filter((r) => r.oi === i && r.ci >= 0);
        const a = clamp01((pc - 0.05) / 0.3);
        if (a > 0) {
          const x1 = ox + 114, y1 = oy(i) - 4;
          if (ms.length) ms.forEach((r) => {
            const x2 = cx - 2, y2 = cy(r.ci) - 4, xm = lerp(x1, x2, a), ym = lerp(y1, y2, a);
            g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(x1, y1); g.lineTo(xm, ym); g.stroke(); g.lineWidth = 1;
          });
          else { g.strokeStyle = c.red; g.lineWidth = 2; g.beginPath(); g.moveTo(x1, y1); g.lineTo(lerp(x1, x1 + 60, a), y1); g.stroke(); g.lineWidth = 1; if (a > 0.9) text(g, 'no match for ' + keys[i], x1 + 66, y1 + 3, c.red, '11px sans-serif'); }
        }
      }
      // result table
      const RY = 232, rw = (W - 16) / 4, MAXR = 12;
      text(g, 'result of the join', 8, RY - 22, c.muted, 'bold 12px sans-serif');
      box(g, 8, RY - 14, W - 16, 16, c.bg3, 1); ['oid', 'cid', 'amt', 'name'].forEach((h, k) => cellText(g, h, 8 + k * rw, RY - 2, rw, c.muted));
      let shown = 0;
      rows.forEach((r, s) => {
        const a = ease((t - r.ph) / 0.3); if (a <= 0) return; shown++;
        if (s >= MAXR) return;
        const y = RY + 14 + s * RH, off = (1 - a) * -12;
        g.save(); g.globalAlpha = a;
        box(g, 8, y - 12 + off, W - 16, RH, r.ci < 0 || r.oi < 0 ? c.orange : c.green, 0.12);
        const oRow = r.oi >= 0 ? ['o' + (r.oi + 1), keys[r.oi], AMT[r.oi]] : ['NULL', 'NULL', 'NULL'], nm = r.ci >= 0 ? custs[r.ci][1] : 'NULL';
        oRow.concat([nm]).forEach((v, k) => cellText(g, v, 8 + k * rw, y + off, rw, v === 'NULL' ? c.orange : c.fg));
        g.restore();
      });
      if (rows.length > MAXR && t >= total()) text(g, `… ${rows.length - MAXR} more rows not drawn`, 8, RY + 14 + MAXR * RH, c.muted, '11px sans-serif');
      const done = t >= total() - 1e-6, type = q(ctl, '#jt').value;
      const nInner = rows.filter((r) => r.oi >= 0 && r.ci >= 0).length;
      out.innerHTML = `orders: ${n} rows   customers: ${custs.length} rows\njoin type: <span class="b">${type}</span>\nrows out: <span class="g">${done ? rows.length : shown}</span>${done ? `   (matched pairs: ${nInner})` : ''}`;
      cap.textContent = done
        ? (nInner > n - keys.filter((k) => !custs.some((cc) => cc[0] === k)).length ? 'A repeated key on the right made one order match twice, so the join produced more rows than there are orders.' : 'Each order matched at most one customer: many orders to one customer.') + ' (illustrative)'
        : t < n ? `Order o${cur + 1} looks for key ${keys[cur]} in the customers table.` : `Outer join: customer ${custs[extra[cur - n]][0]} has no orders, so it is added with NULL order columns.`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    build(); st = runner(root, ctl, total, draw);
  };

  /* ======================================================================
     3. Missing data
     ====================================================================== */
  AI.labs['ds-wr-missing'] = function (root) {
    const AGE = [23, 25, 28, 31, 35, 40, 44, 48, 52, 58], INC = [30, 35, 38, 42, 45, 50, 55, 70, 90, 120];
    const PRE = { MCAR: [3, 7, 9], MAR: [1, 2, 3], MNAR: [8, 9, 10] };
    const N = 10, TRUE = mean(INC);
    const { view, ctl } = AI.labShell(root, 'Missing cells lighting up and being filled', 'income in thousands; the true values are hidden from the method but shown here (illustrative data)',
      `<canvas style="height:340px"></canvas>`,
      `<label>Why are cells missing?<select id="mech"><option value="MCAR">MCAR: random rows (3, 7, 9)</option><option value="MAR">MAR: the youngest skip (1, 2, 3)</option><option value="MNAR">MNAR: top earners hide (8, 9, 10)</option><option value="custom">custom: click rows on the canvas</option></select></label>
       <label>Handling<select id="strat"><option value="delete">delete rows with a missing income</option><option value="mean">fill with the mean of the observed</option><option value="median">fill with the median of the observed</option><option value="group">fill with the mean of the same age group (under 40 / 40 and over)</option></select></label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="ind">add an indicator column income_missing</label>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let miss = new Array(N).fill(false), fill = [], st;
    const setMask = (ids) => { miss = INC.map((_, i) => ids.includes(i + 1)); };
    const plan = () => {
      const s = q(ctl, '#strat').value, obs = INC.filter((_, i) => !miss[i]);
      fill = INC.map((v, i) => {
        if (!miss[i]) return v;
        if (s === 'delete') return null;
        if (s === 'mean') return mean(obs);
        if (s === 'median') return median(obs);
        const grp = INC.filter((_, j) => !miss[j] && (AGE[j] < 40) === (AGE[i] < 40));
        return grp.length ? mean(grp) : mean(obs);
      });
    };
    const mech = q(ctl, '#mech');
    mech.onchange = () => { if (mech.value !== 'custom') setMask(PRE[mech.value]); plan(); st.restart(); };
    q(ctl, '#strat').onchange = () => { plan(); st.restart(); };
    q(ctl, '#ind').onchange = () => { st.dirty = true; };
    const RH = 18, Y0 = 44;
    cv.addEventListener('click', (e) => {
      const r = cv.getBoundingClientRect(), i = Math.floor((e.clientY - r.top - (Y0 - 12)) / RH);
      if (i < 0 || i >= N) return;
      miss[i] = !miss[i]; mech.value = 'custom'; plan(); st.restart();
    });
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const ind = q(ctl, '#ind').checked, cols = ind ? ['id', 'age', 'income', 'income_missing'] : ['id', 'age', 'income'];
      const cw = Math.min(110, (W - 16) / cols.length), ws = [cw * 0.6, cw * 0.8, cw, cw * 1.3].slice(0, cols.length);
      const xs = ws.map((_, k) => 8 + ws.slice(0, k).reduce((a, b) => a + b, 0));
      text(g, 'table of 10 people (click a row to hide or show its income)', 8, 16, c.muted, 'bold 12px sans-serif');
      box(g, 8, 24, ws.reduce((a, b) => a + b, 0), 16, c.bg3, 1);
      cols.forEach((h, k) => cellText(g, h, xs[k], 36, ws[k], c.muted, 'right'));
      const known = []; let cnt = 0;
      for (let i = 0; i < N; i++) {
        const p = t - i, y = Y0 + i * RH, strat = q(ctl, '#strat').value;
        if (p >= 0 && p < 1) box(g, 0, y - 12, W, RH, c.blue, 0.1);
        const done = p >= 0.6;
        g.save();
        if (miss[i] && strat === 'delete' && done) g.globalAlpha = 0.3;
        cellText(g, i + 1, xs[0], y, ws[0], c.fg, 'right'); cellText(g, AGE[i], xs[1], y, ws[1], c.fg, 'right');
        if (!miss[i]) cellText(g, INC[i], xs[2], y, ws[2], c.fg, 'right');
        else if (!done) {
          const pulse = p >= 0 ? 0.45 + 0.35 * Math.sin(p * 18) : 0.45;
          box(g, xs[2] + 4, y - 11, ws[2] - 6, 15, c.red, p >= 0 ? pulse : 0.2); cellText(g, '?', xs[2], y, ws[2], c.fg, 'right');
        } else if (strat === 'delete') { cellText(g, 'dropped', xs[2], y, ws[2], c.red, 'right'); g.strokeStyle = c.red; g.beginPath(); g.moveTo(8, y - 4); g.lineTo(xs[2] + ws[2], y - 4); g.stroke(); }
        else { const a = ease((p - 0.6) / 0.3); g.globalAlpha = a; box(g, xs[2] + 4, y - 11, ws[2] - 6, 15, c.orange, 0.25); cellText(g, f2(fill[i]), xs[2], y, ws[2], c.orange, 'right'); }
        g.restore();
        if (ind && p >= 0.3) cellText(g, miss[i] ? 1 : 0, xs[3], y, ws[3], miss[i] ? c.orange : c.muted, 'right');
        if (!miss[i] || (done && fill[i] !== null)) { known.push(!miss[i] ? INC[i] : fill[i]); }
        if (!miss[i] || (done && fill[i] !== null)) cnt++;
      }
      // number line of income estimates
      const ly = 285, lx0 = 24, lx1 = W - 24, X = (v) => lerp(lx0, lx1, (v - 20) / 110);
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(lx0, ly); g.lineTo(lx1, ly); g.stroke();
      [20, 50, 80, 110].forEach((v) => text(g, v, X(v), ly + 14, c.muted, '10px sans-serif', 'center'));
      INC.forEach((v, i) => {
        const y = ly - 12 - (i % 3) * 9; g.beginPath(); g.arc(X(v), y, 3.5, 0, 7);
        if (miss[i]) { g.strokeStyle = c.red; g.stroke(); } else { g.fillStyle = c.blue; g.fill(); }
      });
      const est = mean(known);
      g.strokeStyle = c.green; g.lineWidth = 2; g.beginPath(); g.moveTo(X(TRUE), ly - 48); g.lineTo(X(TRUE), ly + 2); g.stroke(); g.lineWidth = 1;
      text(g, 'true mean ' + f2(TRUE), X(TRUE) + 4, ly - 50, c.green, '11px sans-serif');
      if (Number.isFinite(est)) { g.strokeStyle = c.orange; g.lineWidth = 2; g.beginPath(); g.moveTo(X(est), ly - 36); g.lineTo(X(est), ly + 6); g.stroke(); g.lineWidth = 1; text(g, 'estimate ' + f2(est), X(est) - 4, ly - 38, c.orange, '11px sans-serif', X(est) > X(TRUE) ? 'left' : 'right'); }
      text(g, 'filled dots: seen   hollow red: hidden value', 8, H - 8, c.muted, '10px sans-serif');
      const obs = INC.filter((_, i) => !miss[i]);
      out.innerHTML = `missing: <span class="r">${miss.filter(Boolean).length}</span> of ${N}   rows used so far: ${cnt}\nmean of observed: ${f2(mean(obs))}\ncurrent estimate: <span class="o">${f2(est)}</span>   true mean: <span class="g">${f2(TRUE)}</span>`;
      const done = t >= N - 1e-6, r = Math.min(N - 1, Math.floor(t));
      cap.textContent = done ? `Finished: the estimate is ${f2(est)} against a true ${f2(TRUE)} (error ${f2(est - TRUE)}). The truth is only known because this is a simulation (illustrative).`
        : (miss[r] ? `Row ${r + 1}: income is missing, so the cell lights up and the chosen method fills it or drops the row.` : `Row ${r + 1}: income observed, nothing to do.`);
    };
    o.onresize = () => { if (st) st.dirty = true; };
    setMask(PRE.MCAR); plan(); st = runner(root, ctl, () => N, draw);
  };

  /* ======================================================================
     4a. Duplicates
     ====================================================================== */
  AI.labs['ds-wr-dupes'] = function (root) {
    const D = [['Amal Hassan', 'amal@mail.com'], [' amal hassan', 'AMAL@mail.com'], ['Omar Ali', 'omar@mail.com'], ['Omar Ali', 'omar@mail.com'],
      ['Lina Said', 'lina@mail.com'], ['Lina  Said', 'lina@mail.com'], ['Sami Nabil', 'sami@mail.com'], ['Omar Ali', 'omar@mail.org']];
    const N = D.length;
    const { view, ctl } = AI.labShell(root, 'Finding duplicates by key', 'a row is a duplicate if its key was seen before; the cleaning steps change the key (illustrative data)',
      `<canvas style="height:320px"></canvas>`,
      `<label>Two rows are the same person if these match<select id="by"><option value="both">name and email</option><option value="name">name only</option><option value="email">email only</option></select></label>
       <div class="legend" style="gap:8px">
         <label style="flex-direction:row;gap:4px;align-items:center"><input type="checkbox" id="tr">trim spaces</label>
         <label style="flex-direction:row;gap:4px;align-items:center"><input type="checkbox" id="lo">lowercase</label>
         <label style="flex-direction:row;gap:4px;align-items:center"><input type="checkbox" id="co">collapse repeated spaces</label></div>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let keyOf = [], dupOf = [], st;
    const vis = (s) => s.replace(/ /g, '·');
    const plan = () => {
      const by = q(ctl, '#by').value, tr = q(ctl, '#tr').checked, lo = q(ctl, '#lo').checked, co = q(ctl, '#co').checked;
      const nz = (s) => { if (co) s = s.replace(/ +/g, ' '); if (tr) s = s.trim(); if (lo) s = s.toLowerCase(); return s; };
      const seen = new Map(); keyOf = []; dupOf = [];
      D.forEach((r, i) => {
        const parts = by === 'both' ? [r[0], r[1]] : by === 'name' ? [r[0]] : [r[1]];
        const k = parts.map(nz).join(' | '); keyOf.push(k);
        if (seen.has(k)) dupOf.push(seen.get(k)); else { seen.set(k, i); dupOf.push(-1); }
      });
    };
    ctl.querySelectorAll('#by,#tr,#lo,#co').forEach((e) => e.addEventListener('change', () => { plan(); st.restart(); }));
    const RH = 34;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const ws = [20, 0.3 * W, 0.3 * W, 0]; ws[3] = W - 16 - ws[0] - ws[1] - ws[2];
      const xs = ws.map((_, k) => 8 + ws.slice(0, k).reduce((a, b) => a + b, 0));
      box(g, 8, 8, W - 16, 18, c.bg3, 1);
      ['#', 'name (· = space)', 'email', 'verdict'].forEach((h, k) => cellText(g, h, xs[k], 21, ws[k], c.muted));
      for (let i = 0; i < N; i++) {
        const p = t - i, y = 46 + i * RH;
        if (p >= 0 && p < 1) box(g, 0, y - 15, W, RH - 2, c.blue, 0.13);
        cellText(g, i + 1, xs[0], y, ws[0], c.fg); cellText(g, '"' + vis(D[i][0]) + '"', xs[1], y, ws[1], c.fg); cellText(g, D[i][1], xs[2], y, ws[2], c.fg);
        if (p >= 0.3) cellText(g, 'key: ' + vis(keyOf[i]), xs[1], y + 13, W - xs[1] - 8, c.purple);
        if (p >= 0.6) {
          const a = ease((p - 0.6) / 0.3); g.save(); g.globalAlpha = a;
          if (dupOf[i] < 0) cellText(g, 'new, keep', xs[3], y, ws[3], c.green); else { box(g, 0, y - 15, W, RH - 2, c.orange, 0.12); cellText(g, 'dup of ' + (dupOf[i] + 1), xs[3], y, ws[3], c.orange); }
          g.restore();
        }
      }
      const upto = Math.min(N, Math.floor(t + 0.6)), kept = dupOf.slice(0, upto).filter((d) => d < 0).length;
      const done = t >= N - 1e-6;
      out.innerHTML = `rows in: ${N}\nunique after ${done ? 'all rows' : upto + ' rows'}: <span class="g">${kept}</span>\nduplicates flagged: <span class="o">${upto - kept}</span>`;
      const r = Math.min(N - 1, Math.floor(t));
      cap.textContent = done ? `Done: ${N} rows became ${dupOf.filter((d) => d < 0).length} with this rule. Try "name only": it merges two different Omars (illustrative).`
        : `Row ${r + 1}: build the key, then check whether that key was already seen.`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    plan(); st = runner(root, ctl, () => N, draw);
  };

  /* ======================================================================
     4b. Range checks and outliers
     ====================================================================== */
  AI.labs['ds-wr-range'] = function (root) {
    const { view, ctl } = AI.labShell(root, 'Impossible versus unusual', 'ages arrive one by one; a range check flags impossible values, an IQR fence flags unusual ones (illustrative data)',
      `<canvas style="height:280px"></canvas>`,
      `<label>Ages (comma separated, up to 24)<input type="text" id="vals" value="34,29,41,38,27,45,52,31,36,-3,212,39,98,33" spellcheck="false"></label>
       <div class="row"><label style="flex:1">Valid min<input type="number" id="mn" value="0"></label><label style="flex:1">Valid max<input type="number" id="mx" value="120"></label></div>
       <label>IQR fence multiplier k <span class="v" id="kv"></span><input type="range" id="k" min="0.5" max="3" step="0.1" value="1.5"></label>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let vals = [], st;
    const quant = (s, p) => { const x = p * (s.length - 1), l = Math.floor(x); return s[l] + (x - l) * ((s[l + 1] === undefined ? s[l] : s[l + 1]) - s[l]); };
    const read = () => { const a = q(ctl, '#vals').value.split(',').map((s) => parseFloat(s)).filter(Number.isFinite).slice(0, 24); vals = a.length >= 4 ? a : [34, 29, 41, 38, 27, 45, 52, 31, 36]; };
    const info = () => {
      const s = [...vals].sort((a, b) => a - b), q1 = quant(s, 0.25), q3 = quant(s, 0.75), k = +q(ctl, '#k').value, iqr = q3 - q1;
      return { q1, q3, lo: q1 - k * iqr, hi: q3 + k * iqr, mn: +q(ctl, '#mn').value, mx: +q(ctl, '#mx').value, k };
    };
    const cls = (v, I) => (v < I.mn || v > I.mx ? 'bad' : v < I.lo || v > I.hi ? 'odd' : 'ok');
    ['#vals', '#mn', '#mx'].forEach((s) => q(ctl, s).addEventListener('input', () => { read(); st.restart(); }));
    q(ctl, '#k').addEventListener('input', () => { q(ctl, '#kv').textContent = (+q(ctl, '#k').value).toFixed(1); st.dirty = true; });
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const I = info(), N = vals.length;
      const lo = Math.min(...vals, I.mn, I.lo) - 10, hi = Math.max(...vals, I.mx, I.hi) + 10, X = (v) => lerp(20, W - 20, (v - lo) / (hi - lo)), base = H - 40;
      box(g, X(I.mn), 10, X(I.mx) - X(I.mn), base - 10, c.green, 0.09);
      g.strokeStyle = c.green; g.setLineDash([]); g.beginPath(); g.moveTo(X(I.mn), 10); g.lineTo(X(I.mn), base); g.moveTo(X(I.mx), 10); g.lineTo(X(I.mx), base); g.stroke();
      text(g, 'valid range', X(I.mn) + 4, 22, c.green, '11px sans-serif');
      g.strokeStyle = c.orange; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(X(I.lo), 10); g.lineTo(X(I.lo), base); g.moveTo(X(I.hi), 10); g.lineTo(X(I.hi), base); g.stroke(); g.setLineDash([]);
      text(g, 'IQR fences ' + f2(I.lo) + ' to ' + f2(I.hi), X(I.hi) + 4, 36, c.orange, '11px sans-serif');
      g.strokeStyle = c.border; g.beginPath(); g.moveTo(10, base); g.lineTo(W - 10, base); g.stroke();
      const step = AI.niceStep(hi - lo, 8); for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) text(g, v, X(v), base + 14, c.muted, '10px sans-serif', 'center');
      let nb = 0, no = 0;
      vals.forEach((v, i) => {
        const p = t - i; if (p < 0) return;
        const a = ease(p / 0.8), yT = base - 12 - (i % 4) * 14, y = lerp(18, yT, a), k = cls(v, I);
        if (p >= 0.8) { if (k === 'bad') nb++; else if (k === 'odd') no++; }
        const col = p < 0.8 ? c.blue : k === 'bad' ? c.red : k === 'odd' ? c.orange : c.blue;
        g.fillStyle = col; g.beginPath(); g.arc(X(v), y, 5, 0, 7); g.fill();
        if (p >= 0.8 && k !== 'ok') text(g, v, X(v), yT - 9, col, '11px sans-serif', 'center');
      });
      text(g, 'blue: fine   orange: unusual (check)   red: impossible (fix or drop)', 8, H - 6, c.muted, '10px sans-serif');
      const kept = vals.filter((v) => cls(v, I) === 'ok'), valid = vals.filter((v) => cls(v, I) !== 'bad');
      const done = t >= N - 1e-6;
      out.innerHTML = `n = ${N}   Q1 = ${f2(I.q1)}   Q3 = ${f2(I.q3)}\nimpossible: <span class="r">${done ? vals.filter((v) => cls(v, I) === 'bad').length : nb}</span>   unusual: <span class="o">${done ? vals.filter((v) => cls(v, I) === 'odd').length : no}</span>\nmean of all: ${f2(mean(vals))}   of the range-valid: ${f2(mean(valid))}   of the inside-both: ${f2(mean(kept))}`;
      cap.textContent = done ? 'A value outside the valid range is an error; one outside the IQR fence is only rare, and might be real, so look before deleting (illustrative).'
        : `Value ${vals[Math.min(N - 1, Math.floor(t))]} falls onto the axis and is tested against both rules.`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    read(); q(ctl, '#kv').textContent = '1.5'; st = runner(root, ctl, () => vals.length, draw);
  };

  /* ======================================================================
     5. Parsing types, units and formats
     ====================================================================== */
  AI.labs['ds-wr-parse'] = function (root) {
    const MODES = {
      prices: {
        title: 'price strings to numbers', raw: ['$1,200', '950', '1.2k', 'USD 80', 'n/a', '1,200.50'],
        opts: [['strip', 'remove $, USD, commas and spaces', 1], ['suffix', 'read the suffix k as thousands', 1], ['na', 'treat n/a as missing, not an error', 1]],
        parse: (s, o) => {
          if (o.na && /^n\/?a$/i.test(s)) return { v: 'missing', k: 'miss' };
          let x = s; if (o.strip) x = x.replace(/USD|\$|,|\s/g, '');
          const n = o.suffix && /k$/i.test(x) ? Number(x.slice(0, -1)) * 1000 : Number(x);
          return Number.isFinite(n) ? { v: String(n), n, k: 'ok' } : { v: 'NaN (failed)', k: 'bad' };
        },
        sum: (r) => { const ok = r.filter((x) => x.k === 'ok'); return `parsed ${ok.length} of ${r.length}; mean of parsed = ${ok.length ? f2(mean(ok.map((x) => x.n))) : '–'}`; },
      },
      distances: {
        title: 'distances with mixed units', raw: ['5 km', '3200 m', '2 mi', '800 m', '1.5 km'],
        opts: [['num', 'read the number and drop the unit text', 1], ['conv', 'convert every value to metres', 1]],
        parse: (s, o) => {
          const m = /^([\d.]+)\s*(km|m|mi)$/.exec(s); if (!o.num || !m) return { v: 'NaN (failed)', k: 'bad' };
          let n = +m[1]; if (o.conv) n *= { km: 1000, m: 1, mi: 1609.344 }[m[2]];
          return { v: String(+n.toFixed(3)) + (o.conv ? ' m' : ''), n, k: o.conv ? 'ok' : 'warn' };
        },
        sum: (r) => { const ok = r.filter((x) => x.n !== undefined); return `mean = ${ok.length ? f2(mean(ok.map((x) => x.n))) : '–'}${ok.length && ok[0].k === 'warn' ? ' (units mixed, so this is meaningless)' : ok.length ? ' metres' : ''}`; },
      },
      dates: {
        title: 'ambiguous day/month dates', raw: ['03/04/2024', '12/05/2024', '25/12/2024', '07/08/2024'],
        opts: [], select: ['order', 'day first (DMY)', 'month first (MDY)'],
        parse: (s, o) => {
          const [a, b, y] = s.split('/').map(Number), d = o.order === 'MDY' ? b : a, m = o.order === 'MDY' ? a : b;
          const dim = new Date(Date.UTC(y, m, 0)).getUTCDate();
          if (m < 1 || m > 12 || d < 1 || d > dim) return { v: 'invalid date', k: 'bad' };
          const amb = a <= 12 && b <= 12 && a !== b;
          return { v: `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`, k: amb ? 'warn' : 'ok' };
        },
        sum: (r) => `invalid: ${r.filter((x) => x.k === 'bad').length}; ambiguous (both orders give a real date): ${r.filter((x) => x.k === 'warn').length}`,
      },
      times: {
        title: 'timestamps with time zones', raw: ['2024-03-04 23:30 -05:00', '2024-03-05 01:15 +02:00', '2024-03-05 08:00 +09:00'],
        opts: [['utc', 'convert to UTC before comparing', 1]],
        parse: (s, o, i) => {
          const m = /^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d) ([+-])(\d\d):(\d\d)$/.exec(s);
          const ms = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) - (m[6] === '-' ? -1 : 1) * (+m[7] * 60 + +m[8]) * 60000;
          const iso = new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
          return o.utc ? { v: iso + ' UTC', n: ms, k: 'ok' } : { v: s.slice(0, 16) + ' (local)', n: Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]), k: 'warn' };
        },
        sum: (r) => { const ord = r.map((x, i) => [x.n, i + 1]).sort((a, b) => a[0] - b[0]).map((p) => 'row ' + p[1]); return 'earliest to latest: ' + ord.join(' < '); },
      },
    };
    const { view, ctl } = AI.labShell(root, 'Parsing strings into typed values', 'raw text flows through the parser you switch on; failures and silent mistakes are coloured (illustrative)',
      `<canvas style="height:290px"></canvas>`,
      `<label>Column<select id="mode"><option value="prices">prices: "$1,200", "1.2k", "n/a"</option><option value="distances">distances: km, m, mi</option><option value="dates">dates: 03/04/2024</option><option value="times">timestamps: time zones</option></select></label>
       <div id="opts" style="display:flex;flex-direction:column;gap:6px"></div>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap'), optBox = q(ctl, '#opts');
    let mode = MODES.prices, res = [], st;
    const opts = () => { const r = {}; optBox.querySelectorAll('input[type=checkbox]').forEach((x) => (r[x.dataset.k] = x.checked)); const s = optBox.querySelector('select'); if (s) r.order = s.value; return r; };
    const plan = () => { const op = opts(); res = mode.raw.map((s, i) => mode.parse(s, op, i)); };
    const buildOpts = () => {
      optBox.innerHTML = (mode.opts.map((x) => `<label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" data-k="${x[0]}" ${x[2] ? 'checked' : ''}>${x[1]}</label>`).join(''))
        + (mode.select ? `<select><option value="DMY">${mode.select[1]}</option><option value="MDY">${mode.select[2]}</option></select>` : '');
      optBox.querySelectorAll('input,select').forEach((e) => e.addEventListener('change', () => { plan(); st.restart(); }));
    };
    q(ctl, '#mode').onchange = (e) => { mode = MODES[e.target.value]; buildOpts(); plan(); st.restart(); };
    const colOf = (k) => (k === 'ok' ? c.green : k === 'bad' ? c.red : k === 'warn' ? c.orange : c.purple);
    const RH = 30;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const N = mode.raw.length, w1 = W * 0.46, x2 = 16 + w1 + W * 0.07, w2 = W - x2 - 8;
      text(g, mode.title, 8, 16, c.muted, 'bold 12px sans-serif');
      box(g, 8, 24, W - 16, 18, c.bg3, 1); cellText(g, 'raw text', 14, 37, w1, c.muted); cellText(g, 'parsed value', x2, 37, w2, c.muted);
      for (let i = 0; i < N; i++) {
        const p = t - i, y = 66 + i * RH;
        if (p >= 0 && p < 1) box(g, 0, y - 17, W, RH - 2, c.blue, 0.12);
        cellText(g, '"' + mode.raw[i] + '"', 14, y, w1, c.fg);
        if (p >= 0.25) {
          const a = ease((p - 0.25) / 0.3), ax = lerp(16 + w1 * 0.9, x2 - 6, a);
          g.save(); g.globalAlpha = Math.min(1, a * 1.2);
          g.strokeStyle = c.muted; g.beginPath(); g.moveTo(16 + w1 * 0.9, y - 4); g.lineTo(ax, y - 4); g.stroke();
          if (p >= 0.55) { const b = ease((p - 0.55) / 0.3); g.globalAlpha = b; cellText(g, res[i].v, x2, y, w2, colOf(res[i].k)); }
          g.restore();
        }
      }
      const N2 = mode.raw.length, done = t >= N2 - 1e-6;
      if (done) text(g, mode.sum(res), 10, H - 14, c.fg, 'bold 12px sans-serif');
      text(g, 'green: parsed   orange: parsed but risky   red: failed   purple: missing', 8, H - 34 + 14 * 0, c.muted, '10px sans-serif');
      out.innerHTML = done ? esc(mode.sum(res)) : `parsed so far: ${Math.min(N2, Math.floor(t + 0.55))} of ${N2}`;
      const r = Math.min(N2 - 1, Math.floor(t));
      cap.textContent = done ? 'A silent wrong number (orange) is worse than an error (red): decide each rule on purpose (illustrative).' : `Row ${r + 1}: the parser reads "${mode.raw[r]}" with the rules you ticked.`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    buildOpts(); plan(); st = runner(root, ctl, () => mode.raw.length, draw);
  };

  /* ======================================================================
     6. Group and aggregate
     ====================================================================== */
  AI.labs['ds-wr-group'] = function (root) {
    const G = [['A', 'wk', 10], ['B', 'wk', 20], ['A', 'wk', 12], ['A', 'we', 11], ['C', 'we', 40], ['B', 'we', 22], ['A', 'wk', 13], ['A', 'we', 12], ['B', 'wk', 24], ['A', 'wk', 10]];
    const N = G.length;
    const AGG = { sum: (a) => a.reduce((s, x) => s + x, 0), mean, count: (a) => a.length, max: (a) => Math.max(...a), median };
    const { view, ctl } = AI.labShell(root, 'Split, apply, combine', 'each row is sent to the bucket of its key; then one summary per bucket (illustrative data: sales per shift)',
      `<canvas style="height:330px"></canvas>`,
      `<label>Group by<select id="by"><option value="store">store</option><option value="day">day type (wk / we)</option><option value="both">store and day type (pivot)</option><option value="none">nothing (one group)</option></select></label>
       <label>Apply<select id="ag"><option value="sum">sum</option><option value="mean" selected>mean</option><option value="count">count</option><option value="max">max</option><option value="median">median</option></select></label>
       ${COMMON}<div class="readout" id="out"></div><p class="caption" id="cap"></p>`);
    const cv = q(view, 'canvas'), o = AI.canvas(cv), c = AI.col();
    const out = q(ctl, '#out'), cap = q(ctl, '#cap');
    let keys = [], gOf = [], st;
    const keyFn = () => { const b = q(ctl, '#by').value; return (r) => (b === 'store' ? r[0] : b === 'day' ? r[1] : b === 'both' ? r[0] + '-' + r[1] : 'all'); };
    const plan = () => { const f = keyFn(); gOf = G.map((r) => f(r)); keys = [...new Set(gOf)].sort(); };
    ctl.querySelectorAll('#by,#ag').forEach((e) => e.addEventListener('change', () => { plan(); st.restart(); }));
    const palette = () => [c.blue, c.orange, c.green, c.purple, c.pink, c.teal];
    const RH = 18;
    const draw = (t) => {
      const g = o.ctx, W = o.w, H = o.h; g.clearRect(0, 0, W, H);
      const pal = palette(), sw = 150, bx = sw + 12, bw = (W - bx - 8) / keys.length, ag = q(ctl, '#ag').value;
      text(g, 'rows (store, day, sales)', 8, 14, c.muted, 'bold 11px sans-serif');
      text(g, 'buckets, one per key (bold number = ' + ag + ')', bx, 14, c.muted, 'bold 11px sans-serif');
      keys.forEach((k, j) => { g.strokeStyle = c.border; g.strokeRect(bx + j * bw + 2, 24, bw - 4, H - 30); text(g, k, bx + j * bw + bw / 2, 40, pal[j % pal.length], 'bold 12px sans-serif', 'center'); });
      const counts = keys.map(() => 0), vals = keys.map(() => []);
      for (let i = 0; i < N; i++) {
        const j = keys.indexOf(gOf[i]), p = t - i, y = 44 + i * RH;
        const slotN = vals[j].length, ty = 54 + slotN * 15, tx = bx + j * bw + 6;
        const s = ease((p - 0.2) / 0.8);
        if (p >= 1) { vals[j].push(G[i][2]); }
        else if (p >= 0.2) { /* in flight */ }
        const x = lerp(8, tx, s), yy = lerp(y, ty + 0, s);
        if (p >= 0 && p < 1) box(g, 0, y - 12, sw + 4, RH, c.blue, 0.12);
        const w = Math.min(bw - 12, 60);
        if (s <= 0) { cellText(g, G[i].join('  '), 8, y, sw, c.fg); }
        else {
          const col = pal[j % pal.length];
          if (s < 1) cellText(g, '', 0, 0, 0, c.fg);
          box(g, x, yy - 11, s < 1 ? Math.max(w, 70 - 10 * s) : w, 14, col, 0.28);
          cellText(g, s < 1 ? G[i][0] + ' ' + G[i][1] + ' ' + G[i][2] : String(G[i][2]), x, yy, 80, c.fg);
        }
        counts[j] += 0;
      }
      // summaries
      const done = t >= N - 1e-6;
      const res = keys.map((k, j) => (vals[j].length ? AGG[ag](vals[j]) : NaN));
      keys.forEach((k, j) => { if (vals[j].length) { const cx = bx + j * bw + bw / 2; text(g, f2(res[j]), cx, H - 22, pal[j % pal.length], 'bold 11px sans-serif', 'center'); text(g, 'n = ' + vals[j].length, cx, H - 9, c.muted, '10px sans-serif', 'center'); } });
      const all = G.slice(0, Math.floor(t + 1e-9)).map((r) => r[2]);
      const gm = res.filter(Number.isFinite);
      out.innerHTML = `rows placed: ${Math.min(N, Math.floor(t + 1e-9))} of ${N}\n` + keys.map((k, j) => `${k}: ${vals[j].length ? f2(res[j]) : '–'}  (n=${vals[j].length})`).join('\n')
        + (done ? `\npooled ${ag} over all ${N} rows: <span class="g">${f2(AGG[ag](G.map((r) => r[2])))}</span>${ag === 'mean' ? `\nmean of the group means: <span class="o">${f2(mean(gm))}</span> (not the same!)` : ''}` : '');
      const r = Math.min(N - 1, Math.floor(t));
      cap.textContent = done ? `Done. ${keys.length} group${keys.length > 1 ? 's' : ''}, sizes ${vals.map((v) => v.length).join(', ')}, total ${vals.reduce((s, v) => s + v.length, 0)} rows (illustrative).`
        : `Row ${r + 1} (${G[r].join(', ')}) goes to the bucket "${gOf[r]}".`;
    };
    o.onresize = () => { if (st) st.dirty = true; };
    plan(); st = runner(root, ctl, () => N, draw);
  };
})();
