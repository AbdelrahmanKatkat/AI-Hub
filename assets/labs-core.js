/* Reusable teaching labs configured from HTML data-attributes.
 *
 * fplot       <div data-lab="fplot" data-f="a*x+b; x^2" data-params="a=1:-3..3:0.1; b=0:-3..3:0.1" data-view="-5,5,-5,5" data-roots data-diag data-title="..."></div>
 * calc        <div data-lab="calc" data-params="..." data-rows="label = expr; label = expr"></div>
 * machine     <div data-lab="machine" data-f="2*x+1"></div>
 * sigma       <div data-lab="sigma" data-term="i^2"></div>
 * rn          <div data-lab="rn"></div>
 * unitcircle  <div data-lab="unitcircle"></div>
 * vec2        <div data-lab="vec2" data-show="dot,proj,angle"></div>
 * highdim     <div data-lab="highdim"></div>
 */
(function () {
  const COLORS = ['blue', 'orange', 'green', 'purple', 'pink', 'teal'];
  const parseParams = (s) => (s || '').split(';').map((p) => p.trim()).filter(Boolean).map((p) => {
    const m = /^(\w+)\s*=\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)(?:\s*:\s*([\d.]+))?$/.exec(p);
    if (!m) throw new Error('Bad data-params entry: ' + p);
    return { n: m[1], v: +m[2], min: +m[3], max: +m[4], step: +(m[5] || 0.1) };
  });
  const paramControls = (ctl, params, scope) => {
    params.forEach((p) => {
      scope[p.n] = p.v;
      const l = document.createElement('label');
      l.innerHTML = `<span>${p.n} <span class="v"></span></span><input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.v}">`;
      const inp = l.querySelector('input'), val = l.querySelector('.v');
      const upd = () => { scope[p.n] = +inp.value; val.textContent = AI.fmt(+inp.value, 3); };
      inp.oninput = upd; upd(); ctl.appendChild(l);
    });
  };

  /* ---------- function plotter ---------- */
  AI.labs.fplot = function (root) {
    const d = root.dataset, src = (d.f || 'x').split(';').map((s) => s.trim()).filter(Boolean), params = parseParams(d.params);
    const [x0, x1, y0, y1] = (d.view || '-5,5,-5,5').split(',').map(Number);
    const { view, ctl } = AI.labShell(root, d.title || 'Function plotter', d.sub || 'edit the formulas, drag the sliders', `<canvas style="height:${d.height || 340}px"></canvas>`, '');
    const c = AI.canvas(AI.$('canvas', view)), scope = {}, names = ['x', ...params.map((p) => p.n)];
    const fns = src.map(() => null), err = document.createElement('div'); err.className = 'err';
    src.forEach((s, i) => {
      const l = document.createElement('label'); l.innerHTML = `<span style="color:var(--${COLORS[i % 6]})">${d.names ? d.names.split(';')[i] : 'f' + (i + 1) + '(x)'} =</span><input type="text" spellcheck="false" value="${s.replace(/"/g, '&quot;')}">`;
      const inp = l.querySelector('input'); ctl.appendChild(l);
      AI.exprInput(inp, err, { allowed: names, onChange: (f) => (fns[i] = f) }).run();
    });
    ctl.appendChild(err);
    paramControls(ctl, params, scope);
    const out = document.createElement('div'); out.className = 'readout'; ctl.appendChild(out);
    if (!d.roots && !d.readout) out.remove();
    let lastKey = '';
    AI.loop(root, () => {
      const key = JSON.stringify(scope) + AI.$$('input[type=text]', ctl).map((i) => i.value).join('|') + c.w + 'x' + c.h + document.documentElement.dataset.theme;
      if (key === lastKey) return;
      lastKey = key;
      const C = AI.col(), g = c.ctx, v = AI.view(c, x0, x1, y0, y1);
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      const ev = (i, x) => (fns[i] ? fns[i]({ ...scope, x }) : NaN);
      if (d.diag !== undefined) { g.strokeStyle = C.muted; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(v.X(Math.min(x0, y0)), v.Y(Math.min(x0, y0))); g.lineTo(v.X(Math.max(x1, y1)), v.Y(Math.max(x1, y1))); g.stroke(); g.setLineDash([]); }
      g.save(); g.beginPath(); g.rect(v.pad.l, v.pad.t, v.pw, v.ph); g.clip();
      fns.forEach((f, i) => {
        if (!f) return;
        g.strokeStyle = C[COLORS[i % 6]]; g.lineWidth = 2.5; g.beginPath(); let pen = false, py = 0;
        for (let k = 0; k <= 500; k++) {
          const x = x0 + ((x1 - x0) * k) / 500, y = ev(i, x);
          if (!Number.isFinite(y) || (pen && Math.abs(y - py) > (y1 - y0) * 3)) { pen = false; py = y; continue; }
          pen ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); pen = true; py = y;
        }
        g.stroke();
      });
      let txt = '';
      if (d.roots !== undefined) {
        const h = (x) => ev(0, x) - (fns[1] ? ev(1, x) : 0), pts = [];
        let a = x0, ha = h(a);
        for (let k = 1; k <= 800; k++) {
          const b = x0 + ((x1 - x0) * k) / 800, hb = h(b);
          if (Number.isFinite(ha) && Number.isFinite(hb) && ha * hb <= 0 && ha !== hb) {
            let lo = a, hi = b; for (let j = 0; j < 40; j++) { const m = (lo + hi) / 2; (h(lo) * h(m) <= 0) ? (hi = m) : (lo = m); }
            const r = (lo + hi) / 2; if (!pts.length || Math.abs(pts[pts.length - 1] - r) > 1e-3) pts.push(r);
          }
          a = b; ha = hb;
        }
        g.fillStyle = C.fg;
        pts.forEach((r) => { g.beginPath(); g.arc(v.X(r), v.Y(ev(0, r)), 6, 0, 7); g.fill(); });
        txt = pts.length ? 'where the curves meet (f1 = f2):\n' + pts.map((r) => `  x = ${AI.fmt(r, 4)}   y = ${AI.fmt(ev(0, r), 4)}`).join('\n') : 'the curves do not meet in this window';
      }
      g.restore();
      if (d.readout) txt += (txt ? '\n' : '') + d.readout.split(';').map((s) => { const [lab, e] = s.split('='); try { return `${lab.trim()} = ${AI.fmt(AI.compile(AI.parse(e), [...names])({ ...scope, x: 0 }), 4)}`; } catch (e2) { return ''; } }).join('\n');
      out.textContent = txt;
    });
  };

  /* ---------- calc: sliders + live results ---------- */
  AI.labs.calc = function (root) {
    const d = root.dataset, params = parseParams(d.params), scope = {};
    const { view, ctl } = AI.labShell(root, d.title || 'Try it', d.sub || 'drag the sliders, watch the numbers', '<div class="readout" style="font-size:13px"></div>' + (d.note ? `<div class="caption">${d.note}</div>` : ''), '');
    paramControls(ctl, params, scope);
    const rows = d.rows.split(';').map((r) => { const i = r.indexOf('='); return { l: r.slice(0, i).trim(), f: AI.compile(AI.parse(r.slice(i + 1)), params.map((p) => p.n)) }; });
    const out = AI.$('.readout', view);
    const upd = () => (out.innerHTML = rows.map((r) => `<span class="m">${r.l}</span> = <b>${AI.fmt(r.f(scope), 5)}</b>`).join('\n'));
    ctl.addEventListener('input', upd); upd();
  };

  /* ---------- function machine ---------- */
  AI.labs.machine = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, d.title || 'Function machine', 'numbers go in, the rule transforms them, numbers come out',
      '<canvas style="height:190px"></canvas><div class="readout" id="hist" style="margin-top:10px"></div>',
      `<label>Rule &nbsp;f(x) = <input type="text" id="e" value="${d.f || '2*x+1'}" spellcheck="false"></label><div class="err" id="er"></div>
       <label>Feed your own x <input type="number" id="inx" value="3" step="0.5"></label>
       <div class="row"><button class="btn primary" id="feed">Feed it in</button><label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="auto" checked> auto-feed</label></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let fn = null, items = [], timer = 0, log = [];
    const ei = AI.exprInput(q('#e'), q('#er'), { allowed: ['x'], onChange: (f) => { fn = f; items = []; log = []; } }); ei.run();
    const feed = (x) => { const y = fn({ x }); if (Number.isFinite(y)) items.push({ x, y, t: 0 }); };
    q('#feed').onclick = () => feed(+q('#inx').value);
    AI.loop(root, (dt) => {
      timer -= dt; if (q('#auto').checked && timer <= 0) { feed(Math.round((Math.random() * 8 - 4) * 2) / 2); timer = 3.6; }
      const C = AI.col(), g = c.ctx, w = c.w, y = c.h / 2, bx = w / 2, bw = Math.min(190, w * 0.34);
      g.clearRect(0, 0, w, c.h);
      g.strokeStyle = C.border; g.lineWidth = 6; g.lineCap = 'round'; g.beginPath(); g.moveTo(14, y + 26); g.lineTo(w - 14, y + 26); g.stroke();
      g.fillStyle = C.bg2; g.strokeStyle = C.purple; g.lineWidth = 3; g.beginPath(); g.roundRect(bx - bw / 2, y - 50, bw, 100, 14); g.fill(); g.stroke();
      g.fillStyle = C.fg; g.font = '600 15px ui-monospace,Menlo,monospace'; g.textAlign = 'center'; g.fillText('f(x) = ' + q('#e').value, bx, y + 5);
      for (const it of items) {
        it.t += dt * 0.19;
        const p = it.t, px = 30 + p * (w - 60), passed = px > bx;
        const lift = Math.abs(px - bx) < bw / 2 ? 0 : 0;
        g.fillStyle = passed ? C.orange : C.blue; g.beginPath(); g.arc(px, y + lift, 17, 0, 7); g.fill();
        g.fillStyle = '#fff'; g.font = '600 12px ui-monospace,Menlo,monospace'; g.fillText(AI.fmt(passed ? it.y : it.x, 2), px, y + 4);
        if (passed && !it.logged) { it.logged = true; log.unshift(`f(${AI.fmt(it.x, 2)}) = ${AI.fmt(it.y, 3)}`); log = log.slice(0, 6); }
      }
      items = items.filter((it) => it.t < 1.05);
      q('#hist').innerHTML = log.length ? log.map((l) => `<span class="o">${l}</span>`).join('\n') : '<span class="m">outputs appear here</span>';
    });
  };

  /* ---------- sigma / pi ---------- */
  AI.labs.sigma = function (root) {
    const d = root.dataset;
    const { view, ctl } = AI.labShell(root, 'Σ and Π machine', 'a loop written as a symbol',
      '<canvas style="height:260px"></canvas><div class="readout" id="o" style="margin-top:10px;white-space:pre-wrap"></div>',
      `<label>Operation <select id="op"><option value="S">Σ  add up the terms</option><option value="P">Π  multiply the terms</option></select></label>
       <label>Term t(i) = <input type="text" id="e" value="${d.term || 'i^2'}" spellcheck="false"></label><div class="err" id="er"></div>
       <label>Start i = <input type="number" id="lo" value="1" min="-5" max="10"></label>
       <label>End n <span class="v" id="nv"></span><input type="range" id="n" min="1" max="12" value="5"></label>
       <div class="row"><button class="btn primary" id="rep">▶ Replay</button></div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    let fn = null, p = 0;
    const ei = AI.exprInput(q('#e'), q('#er'), { allowed: ['i', 'n'], onChange: (f) => { fn = f; p = 0; } }); ei.run();
    ['#op', '#lo', '#n'].forEach((s) => q(s).addEventListener('input', () => (p = 0))); q('#rep').onclick = () => (p = 0);
    AI.loop(root, (dt) => {
      const lo = Math.round(+q('#lo').value), n = Math.max(lo, +q('#n').value), prod = q('#op').value === 'P'; q('#nv').textContent = n;
      p = Math.min(n - lo + 1, p + dt * 1.1);
      const terms = [], run = []; let acc = prod ? 1 : 0;
      for (let i = lo; i <= n; i++) { const t = fn({ i, n }); terms.push(t); acc = prod ? acc * t : acc + t; run.push(acc); }
      const C = AI.col(), g = c.ctx, L = terms.length, pad = { l: 40, r: 10, t: 14, b: 24 };
      g.clearRect(0, 0, c.w, c.h);
      const show = Math.floor(p), fin = p >= L;
      const vals = [...terms, ...run].filter(Number.isFinite), hi = Math.max(1, ...vals.map(Math.abs)), lw = Math.min(0, ...vals);
      const v = AI.view(c, lo - 0.7, n + 0.7, lw * 1.1, hi * 1.1, pad); AI.axes(g, v, { noGrid: false });
      const bw = Math.min(30, v.pw / L * 0.6);
      for (let k = 0; k < L; k++) {
        if (k > show) break; const s = k < show ? 1 : AI.smooth(p - show), i = lo + k;
        g.fillStyle = C.blue; g.globalAlpha = 0.85; const h = (terms[k] - 0) * s; g.fillRect(v.X(i) - bw / 2, Math.min(v.Y(0), v.Y(h)), bw, Math.abs(v.Y(h) - v.Y(0)));
        g.globalAlpha = 1;
      }
      g.strokeStyle = C.orange; g.lineWidth = 3; g.beginPath();
      for (let k = 0; k <= Math.min(show, L - 1); k++) { const i = lo + k; k ? g.lineTo(v.X(i), v.Y(run[k])) : g.moveTo(v.X(i), v.Y(run[k])); } g.stroke();
      for (let k = 0; k <= Math.min(show, L - 1); k++) { g.fillStyle = C.orange; g.beginPath(); g.arc(v.X(lo + k), v.Y(run[k]), 4, 0, 7); g.fill(); }
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText(`■ term t(i)   ● running ${prod ? 'product' : 'total'}`, v.pad.l + 6, 12);
      const sym = prod ? ' × ' : ' + ';
      q('#o').innerHTML = `<span class="m">${prod ? 'Π' : 'Σ'} from i=${lo} to ${n} of ${q('#e').value}</span>\n` + terms.slice(0, 12).map((t) => AI.fmt(t, 3)).join(sym) + ` = <b class="o">${AI.fmt(acc, 5)}</b>`;
    });
  };

  /* ---------- R^n explorer ---------- */
  AI.labs.rn = function (root) {
    const { view, ctl } = AI.labShell(root, 'One point, many dimensions', 'the same idea in 1, 2 and 3 coordinates',
      '<canvas style="height:260px"></canvas><div class="readout" id="o" style="margin-top:10px"></div>',
      `<label>x₁ <span class="v" id="v1"></span><input type="range" id="a" min="0" max="255" value="240"></label>
       <label>x₂ <span class="v" id="v2"></span><input type="range" id="b" min="0" max="255" value="140"></label>
       <label>x₃ <span class="v" id="v3"></span><input type="range" id="c" min="0" max="255" value="40"></label>
       <div class="caption">x₁ is a position on a line; (x₁, x₂) is a position on a plane; (x₁, x₂, x₃) here is also a colour — a pixel is a point in ℝ³ (red, green, blue). A 28×28 image is a point in ℝ⁷⁸⁴.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view));
    AI.loop(root, () => {
      const a = +q('#a').value, b = +q('#b').value, cc = +q('#c').value; q('#v1').textContent = a; q('#v2').textContent = b; q('#v3').textContent = cc;
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const sec = w / 3;
      // 1D
      g.strokeStyle = C.muted; g.lineWidth = 2; g.beginPath(); g.moveTo(14, h / 2); g.lineTo(sec - 14, h / 2); g.stroke();
      const x1 = 14 + (a / 255) * (sec - 28); g.fillStyle = C.blue; g.beginPath(); g.arc(x1, h / 2, 8, 0, 7); g.fill();
      g.fillStyle = C.muted; g.font = '12px sans-serif'; g.textAlign = 'center'; g.fillText('ℝ¹: a number line', sec / 2, 22);
      // 2D
      const px = sec + 20, py = 34, ps = Math.min(sec - 40, h - 70);
      g.strokeStyle = C.border; g.strokeRect(px, py, ps, ps); g.strokeStyle = C.muted; g.setLineDash([4, 4]);
      const X = px + (a / 255) * ps, Y = py + ps - (b / 255) * ps; g.beginPath(); g.moveTo(X, py + ps); g.lineTo(X, Y); g.lineTo(px, Y); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.orange; g.beginPath(); g.arc(X, Y, 8, 0, 7); g.fill(); g.fillStyle = C.muted; g.fillText('ℝ²: a plane', sec + sec / 2, 22);
      // 3D as colour
      g.fillStyle = `rgb(${a},${b},${cc})`; g.beginPath(); g.roundRect(2 * sec + 30, 40, sec - 60, h - 90, 14); g.fill();
      g.fillStyle = C.muted; g.fillText('ℝ³: a colour', 2 * sec + sec / 2, 22); g.fillText(`(${a}, ${b}, ${cc})`, 2 * sec + sec / 2, h - 16);
      q('#o').innerHTML = `x = (<b class="b">${a}</b>, <b class="o">${b}</b>, <b class="g">${cc}</b>) ∈ ℝ³\n<span class="m">three numbers = one point = one colour. Each extra feature is one more coordinate.</span>`;
    });
  };

  /* ---------- unit circle ---------- */
  AI.labs.unitcircle = function (root) {
    const { view, ctl } = AI.labShell(root, 'The unit circle makes sine and cosine', 'cosine = horizontal position, sine = vertical position',
      '<canvas style="height:300px"></canvas>',
      `<label>Angle θ <span class="v" id="tv"></span><input type="range" id="t" min="0" max="6.2832" step="0.01" value="0.8"></label>
       <div class="row"><button class="btn primary" id="spin">▶ Spin</button></div>
       <div class="readout" id="o"></div>
       <div class="caption">Radians measure the angle by the arc length travelled on the unit circle; 2π ≈ 6.283 is one full turn.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)); let spin = false;
    q('#spin').onclick = () => { spin = !spin; q('#spin').textContent = spin ? '⏸ Stop' : '▶ Spin'; };
    AI.loop(root, (dt) => {
      if (spin) q('#t').value = ((+q('#t').value + dt * 0.45) % 6.2832).toFixed(3);
      const th = +q('#t').value, cs = Math.cos(th), sn = Math.sin(th); q('#tv').textContent = `${AI.fmt(th, 2)} rad = ${AI.fmt((th * 180) / Math.PI, 0)}°`;
      const C = AI.col(), g = c.ctx, w = c.w, h = c.h; g.clearRect(0, 0, w, h);
      const R = Math.min(h / 2 - 22, w * 0.2), cx = R + 34, cy = h / 2;
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(cx - R - 14, cy); g.lineTo(cx + R + 14, cy); g.moveTo(cx, cy - R - 14); g.lineTo(cx, cy + R + 14); g.stroke();
      g.strokeStyle = C.muted; g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
      g.strokeStyle = C.purple; g.lineWidth = 3; g.beginPath(); g.arc(cx, cy, 24, 0, -th, true); g.stroke();
      const px = cx + R * cs, py = cy - R * sn;
      g.strokeStyle = C.orange; g.lineWidth = 3; g.beginPath(); g.moveTo(cx, cy); g.lineTo(px, cy); g.stroke();
      g.strokeStyle = C.green; g.beginPath(); g.moveTo(px, cy); g.lineTo(px, py); g.stroke();
      g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, cy); g.lineTo(px, py); g.stroke();
      g.fillStyle = C.fg; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill();
      // waves
      const wx = cx + R + 60, ww = w - wx - 14, vv = AI.view(c, 0, 6.2832, -1.3, 1.3, { l: 0, r: 0, t: 14, b: 14 });
      const X = (t) => wx + (t / 6.2832) * ww, Y = (v) => cy - v * R;
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(wx, cy); g.lineTo(wx + ww, cy); g.stroke();
      for (const [f, col] of [[Math.cos, C.orange], [Math.sin, C.green]]) {
        g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath(); for (let k = 0; k <= 200; k++) { const t = (6.2832 * k) / 200; k ? g.lineTo(X(t), Y(f(t))) : g.moveTo(X(t), Y(f(t))); } g.stroke();
        g.fillStyle = col; g.beginPath(); g.arc(X(th), Y(f(th)), 5, 0, 7); g.fill();
      }
      g.setLineDash([4, 4]); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(px, py); g.lineTo(X(th), Y(sn)); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('cos θ (orange)   sin θ (green)', wx, 14);
      q('#o').innerHTML = `<span class="o">cos θ = ${AI.fmt(cs, 4)}</span>\n<span class="g">sin θ = ${AI.fmt(sn, 4)}</span>\ncos²+sin² = ${AI.fmt(cs * cs + sn * sn, 4)}`;
    });
  };

  /* ---------- draggable 2-D vectors ---------- */
  AI.labs.vec2 = function (root) {
    const d = root.dataset, show = (d.show || 'dot,proj,angle').split(',');
    const { view, ctl } = AI.labShell(root, d.title || 'Drag the vectors', 'drag the two arrow tips',
      '<canvas style="height:380px;touch-action:none;cursor:grab"></canvas>', '<div class="readout" id="o" style="font-size:12.5px"></div><div class="caption">Drag the blue or orange arrow tip anywhere.</div>');
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv), R = 5;
    const A = { x: +(d.ax || 3), y: +(d.ay || 1.5) }, B = { x: +(d.bx || 1), y: +(d.by || 3) };
    let drag = null;
    const toW = (e) => { const r = cv.getBoundingClientRect(), s = Math.min(c.w, c.h) / (2 * R); return { x: (e.clientX - r.left - c.w / 2) / s, y: -(e.clientY - r.top - c.h / 2) / s }; };
    cv.addEventListener('pointerdown', (e) => { const p = toW(e), da = Math.hypot(p.x - A.x, p.y - A.y), db = Math.hypot(p.x - B.x, p.y - B.y); drag = da < db ? A : B; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; Object.assign(drag, p); });
    cv.addEventListener('pointermove', (e) => { if (drag) Object.assign(drag, toW(e)); });
    cv.addEventListener('pointerup', () => { drag = null; cv.style.cursor = 'grab'; });
    AI.loop(root, () => {
      const C = AI.col(), g = c.ctx, s = Math.min(c.w, c.h) / (2 * R), ox = c.w / 2, oy = c.h / 2, P = (x, y) => [ox + x * s, oy - y * s];
      g.clearRect(0, 0, c.w, c.h); g.lineWidth = 1; g.strokeStyle = C.border;
      for (let i = -R; i <= R; i++) { g.beginPath(); g.moveTo(...P(i, -R)); g.lineTo(...P(i, R)); g.moveTo(...P(-R, i)); g.lineTo(...P(R, i)); g.stroke(); }
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(...P(-R, 0)); g.lineTo(...P(R, 0)); g.moveTo(...P(0, -R)); g.lineTo(...P(0, R)); g.stroke();
      const dot = A.x * B.x + A.y * B.y, na = Math.hypot(A.x, A.y), nb = Math.hypot(B.x, B.y), cosv = dot / (na * nb || 1), ang = Math.acos(AI.clamp(cosv, -1, 1));
      const k = nb ? dot / (nb * nb) : 0, pr = { x: k * B.x, y: k * B.y };
      if (show.includes('proj')) { g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(...P(A.x, A.y)); g.lineTo(...P(pr.x, pr.y)); g.stroke(); g.setLineDash([]); AI.arrow(g, ...P(0, 0), ...P(pr.x, pr.y), C.green, 4); }
      if (show.includes('dist')) { g.strokeStyle = C.red; g.lineWidth = 2; g.beginPath(); g.moveTo(...P(A.x, A.y)); g.lineTo(...P(B.x, B.y)); g.stroke(); }
      if (show.includes('angle')) { g.strokeStyle = C.purple; g.lineWidth = 2.5; g.beginPath(); const a0 = -Math.atan2(A.y, A.x), a1 = -Math.atan2(B.y, B.x); g.arc(ox, oy, 32, a0, a1, ((a1 - a0 + 4 * Math.PI) % (2 * Math.PI)) > Math.PI); g.stroke(); }
      AI.arrow(g, ...P(0, 0), ...P(A.x, A.y), C.blue, 3); AI.arrow(g, ...P(0, 0), ...P(B.x, B.y), C.orange, 3);
      g.fillStyle = C.blue; g.beginPath(); g.arc(...P(A.x, A.y), 8, 0, 7); g.fill(); g.fillStyle = C.orange; g.beginPath(); g.arc(...P(B.x, B.y), 8, 0, 7); g.fill();
      g.font = '600 13px sans-serif'; g.fillStyle = C.blue; g.fillText('a', ...P(A.x, A.y).map((v, i) => v + (i ? -12 : 12))); g.fillStyle = C.orange; g.fillText('b', ...P(B.x, B.y).map((v, i) => v + (i ? -12 : 12)));
      let t = `a = (${AI.fmt(A.x, 2)}, ${AI.fmt(A.y, 2)})   b = (${AI.fmt(B.x, 2)}, ${AI.fmt(B.y, 2)})\n`;
      if (show.includes('dot')) t += `a·b = ${AI.fmt(A.x, 2)}×${AI.fmt(B.x, 2)} + ${AI.fmt(A.y, 2)}×${AI.fmt(B.y, 2)} = <b class="o">${AI.fmt(dot, 3)}</b>\n`;
      if (show.includes('norm')) t += `|a| = ${AI.fmt(na, 3)}   |b| = ${AI.fmt(nb, 3)}\n`;
      if (show.includes('angle')) t += `cos θ = a·b / (|a||b|) = <b class="b">${AI.fmt(cosv, 3)}</b>\nθ = ${AI.fmt((ang * 180) / Math.PI, 1)}°  <span class="m">${Math.abs(dot) < 0.15 ? '(≈ perpendicular)' : dot > 0 ? '(pointing alike)' : '(pointing apart)'}</span>\n`;
      if (show.includes('proj')) t += `projection of a on b = ${AI.fmt(k, 3)}·b = (${AI.fmt(pr.x, 2)}, ${AI.fmt(pr.y, 2)})  <span class="g">green</span>\n`;
      if (show.includes('dist')) t += `distance |a−b| = <b class="r">${AI.fmt(Math.hypot(A.x - B.x, A.y - B.y), 3)}</b>  <span class="m">(red segment)</span>\n`;
      q('#o').innerHTML = t;
    });
  };

  /* ---------- high-dimensional distances ---------- */
  AI.labs.highdim = function (root) {
    const { view, ctl } = AI.labShell(root, 'Why high dimensions feel strange', '60 random points: how different are their pairwise distances?',
      '<canvas style="height:280px"></canvas>',
      `<label>Dimensions d <span class="v" id="dv"></span><input type="range" id="d" min="0" max="3" step="0.02" value="0.3"></label>
       <div class="readout" id="o"></div>
       <div class="caption">Each distance is divided by the average distance. In 2-D they spread widely: some pairs are near, some far. As d grows they all crowd around 1 — "nearest" and "farthest" neighbours become almost the same.</div>`);
    const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), BINS = 50, r = AI.rng(11);
    let key = '', target = new Array(BINS).fill(0), cur = new Array(BINS).fill(0), info = '';
    const compute = (d) => {
      const P = Array.from({ length: 60 }, () => Array.from({ length: d }, r)), ds = [];
      for (let i = 0; i < 60; i++) for (let j = i + 1; j < 60; j++) { let s = 0; for (let k = 0; k < d; k++) { const e = P[i][k] - P[j][k]; s += e * e; } ds.push(Math.sqrt(s)); }
      const m = ds.reduce((a, b) => a + b, 0) / ds.length, mn = Math.min(...ds), mx = Math.max(...ds), h = new Array(BINS).fill(0);
      ds.forEach((v) => { const b = Math.floor((v / m / 2) * BINS); if (b >= 0 && b < BINS) h[b]++; });
      const hm = Math.max(...h); target = h.map((v) => v / hm); info = `farthest / nearest = <b class="o">${AI.fmt(mx / mn, 3)}</b>`;
    };
    AI.loop(root, (dt) => {
      const d = Math.round(Math.pow(10, +q('#d').value)); q('#dv').textContent = d;
      if (key !== d) { key = d; compute(d); }
      const k = 1 - Math.exp(-dt * 4); for (let i = 0; i < BINS; i++) cur[i] += (target[i] - cur[i]) * k;
      const C = AI.col(), g = c.ctx, v = AI.view(c, 0, 2, 0, 1, { l: 16, r: 16, t: 14, b: 26 }), bw = v.pw / BINS;
      g.clearRect(0, 0, c.w, c.h); AI.axes(g, v, { noY: true, noGrid: true });
      g.fillStyle = C.blue; cur.forEach((h, i) => g.fillRect(v.pad.l + i * bw + 1, v.pad.t + v.ph - h * v.ph, bw - 2, h * v.ph));
      g.strokeStyle = C.orange; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(v.X(1), v.pad.t); g.lineTo(v.X(1), v.pad.t + v.ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('distance ÷ average distance', v.pad.l + 4, 12);
      q('#o').innerHTML = `d = ${d}\n${info}`;
    });
  };

  /* ---------- half-plane: which side of the line? ---------- */
  AI.labs.halfplane = function (root) {
    const { view, ctl } = AI.labShell(root, 'A line splits the plane in two', 'w₁x₁ + w₂x₂ + b = 0 is the boundary; the sign of the score says which side',
      '<canvas style="height:380px;touch-action:none;cursor:crosshair"></canvas>',
      `<label>w₁ <span class="v"></span><input type="range" id="w1" min="-3" max="3" step="0.1" value="1"></label>
       <label>w₂ <span class="v"></span><input type="range" id="w2" min="-3" max="3" step="0.1" value="1"></label>
       <label>b <span class="v"></span><input type="range" id="b" min="-4" max="4" step="0.1" value="-1"></label>
       <div class="readout" id="o"></div><div class="caption">Click or drag in the plane to move the test point.</div>`);
    const q = (s) => AI.$(s, root), cv = AI.$('canvas', view), c = AI.canvas(cv), R = 5; let pt = { x: 2, y: 1.5 }, down = false;
    const set = (e) => { const r = cv.getBoundingClientRect(), s = Math.min(c.w, c.h) / (2 * R); pt = { x: (e.clientX - r.left - c.w / 2) / s, y: -(e.clientY - r.top - c.h / 2) / s }; };
    cv.addEventListener('pointerdown', (e) => { down = true; cv.setPointerCapture(e.pointerId); set(e); }); cv.addEventListener('pointermove', (e) => down && set(e)); cv.addEventListener('pointerup', () => (down = false));
    AI.loop(root, () => {
      ['w1', 'w2', 'b'].forEach((k) => (AI.$('#' + k, root).previousElementSibling.textContent = q('#' + k).value));
      const w1 = +q('#w1').value, w2 = +q('#w2').value, b = +q('#b').value, C = AI.col(), g = c.ctx, s = Math.min(c.w, c.h) / (2 * R), ox = c.w / 2, oy = c.h / 2, P = (x, y) => [ox + x * s, oy - y * s];
      g.clearRect(0, 0, c.w, c.h);
      const im = 14; // shade by sign of the score
      for (let i = 0; i < c.w; i += im) for (let j = 0; j < c.h; j += im) { const x = (i + im / 2 - ox) / s, y = -(j + im / 2 - oy) / s; g.fillStyle = w1 * x + w2 * y + b > 0 ? C.blue : C.orange; g.globalAlpha = 0.16; g.fillRect(i, j, im, im); }
      g.globalAlpha = 1; g.strokeStyle = C.border; g.lineWidth = 1;
      for (let i = -R; i <= R; i++) { g.beginPath(); g.moveTo(...P(i, -R)); g.lineTo(...P(i, R)); g.moveTo(...P(-R, i)); g.lineTo(...P(R, i)); g.stroke(); }
      // boundary
      g.strokeStyle = C.fg; g.lineWidth = 3; g.beginPath();
      if (Math.abs(w2) > 1e-6) { g.moveTo(...P(-R * 2, (-b + w1 * R * 2) / w2)); g.lineTo(...P(R * 2, (-b - w1 * R * 2) / w2)); } else if (Math.abs(w1) > 1e-6) { g.moveTo(...P(-b / w1, -R * 2)); g.lineTo(...P(-b / w1, R * 2)); }
      g.stroke();
      // weight vector (normal) from the origin's nearest boundary point
      const n2 = w1 * w1 + w2 * w2 || 1, fx = (-b * w1) / n2, fy = (-b * w2) / n2;
      AI.arrow(g, ...P(fx, fy), ...P(fx + w1 * 0.8, fy + w2 * 0.8), C.purple, 3);
      const score = w1 * pt.x + w2 * pt.y + b, dist = Math.abs(score) / Math.sqrt(n2);
      g.fillStyle = score > 0 ? C.blue : C.orange; g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.arc(...P(pt.x, pt.y), 9, 0, 7); g.fill(); g.stroke();
      q('#o').innerHTML = `test point x = (${AI.fmt(pt.x, 2)}, ${AI.fmt(pt.y, 2)})\nscore = ${AI.fmt(w1, 1)}×${AI.fmt(pt.x, 2)} + ${AI.fmt(w2, 1)}×${AI.fmt(pt.y, 2)} + ${AI.fmt(b, 1)}\n      = <b class="${score > 0 ? 'b' : 'o'}">${AI.fmt(score, 3)}</b>  → class <b>${score > 0 ? 'blue (+)' : 'orange (−)'}</b>\ndistance to line = |score| / ‖w‖ = ${AI.fmt(dist, 3)}\n<span class="m">purple arrow = w: points toward the blue side</span>`;
    });
  };

  /* ---------- decode an equation, piece by piece ----------
   * <div data-lab="decode"><script type="application/json">{"title":"...","parts":[{"tex":"\frac1N","say":"..."}],"code":"..."}</script></div> */
  AI.labs.decode = function (root) {
    const cfg = JSON.parse(AI.$('script', root).textContent);
    root.classList.add('lab');
    root.innerHTML = `<div class="lab-head"><i class="live-dot"></i><b>${cfg.title || 'Decode the equation'}</b><span>click each piece</span></div>
      <div style="padding:16px"><div class="chips" id="parts" style="gap:10px;font-size:18px"></div>
      <div class="readout" id="say" style="margin-top:14px;white-space:pre-wrap;font-family:inherit;font-size:14px;min-height:3.5em"></div>
      ${cfg.code ? `<div class="caption" style="margin-top:12px">The same thing as code:</div><pre style="margin-top:4px"><code>${cfg.code.replace(/</g, '&lt;')}</code></pre>` : ''}</div>`;
    const box = AI.$('#parts', root), say = AI.$('#say', root);
    cfg.parts.forEach((p, i) => {
      const b = document.createElement('button'); b.className = 'chip'; b.style.fontSize = '18px'; b.style.padding = '4px 12px';
      if (window.katex) window.katex.render(p.tex, b, { throwOnError: false }); else b.textContent = p.tex;
      b.onclick = () => { AI.$$('.chip', box).forEach((x) => (x.style.borderColor = '')); b.style.borderColor = 'var(--blue)'; say.innerHTML = p.say; };
      box.appendChild(b); if (i === 0) b.click();
    });
  };
})();
