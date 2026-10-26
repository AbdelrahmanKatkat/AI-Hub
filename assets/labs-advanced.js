/* Module 8 labs (advanced topics): Markov chains, EM / Beta-Bernoulli, kernel lift, kernel matrix, Gaussian process, VAE / ELBO, diffusion.
   All animations are dt-based, slow by default, with speed sliders and pause/step. */
(function () {
  const $ = (root, s) => AI.$(s, root);
  const alpha = (hex, a) => { const [r, g, b] = AI.rgb(hex); return `rgba(${r},${g},${b},${a})`; };
  const ptr = (cv, e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const pdfN = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI);
  const sl = (id, label, min, max, step, val) =>
    `<label>${label} <span class="v" id="${id}v"></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  function bind(root, id, f, on) {
    const el = $(root, '#' + id), v = $(root, '#' + id + 'v');
    const upd = () => { v.textContent = f ? f(+el.value) : el.value; if (on) on(+el.value); };
    el.addEventListener('input', upd); upd(); return el;
  }
  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    z -= 1; let x = c[0];
    for (let i = 1; i < 9; i++) x += c[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  function chol(A, n) {
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j];
    }
    return L;
  }
  const solveL = (L, b, n) => { const y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; } return y; };
  const solveLT = (L, y, n) => { const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; };
  function jacobiEig(M, n) {
    const A = M.map((r) => Float64Array.from(r));
    for (let sw = 0; sw < 40; sw++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
      }
    }
    return Array.from({ length: n }, (_, i) => A[i][i]).sort((a, b) => b - a);
  }
  const arrowHead = (g, x, y, ang, col, h = 8) => {
    g.fillStyle = col; g.beginPath(); g.moveTo(x, y);
    g.lineTo(x - h * Math.cos(ang - 0.4), y - h * Math.sin(ang - 0.4));
    g.lineTo(x - h * Math.cos(ang + 0.4), y - h * Math.sin(ang + 0.4)); g.closePath(); g.fill();
  };

  /* ====================================================== Markov chain ====================================================== */
  AI.labs.markov = function (root) {
    const pageRank = (d) => [[0, .5, .5, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]].map((r) => r.map((v) => +(d * v + (1 - d) / 4).toFixed(4)));
    const presets = {
      'Weather': { names: ['Sunny', 'Cloudy', 'Rainy'], P: [[.8, .15, .05], [.3, .4, .3], [.2, .4, .4]] },
      'Slow mixing': { names: ['A', 'B', 'C'], P: [[.95, .05, 0], [.05, .9, .05], [0, .05, .95]] },
      'PageRank (d=0.85)': { names: ['Page A', 'Page B', 'Page C', 'Page D'], P: pageRank(0.85) },
      'Cycle (periodic)': { names: ['A', 'B', 'C'], P: [[0, 1, 0], [0, 0, 1], [1, 0, 0]] },
      'Absorbing': { names: ['A', 'B', 'C'], P: [[1, 0, 0], [.3, .4, .3], [0, 0, 1]] },
    };
    const { view, ctl } = AI.labShell(root, 'Markov chain lab', 'a random walker on a graph — the distribution evolves as πₜ₊₁ = πₜ P',
      `<canvas id="g" style="height:330px"></canvas><canvas id="b" style="height:190px"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<div class="chips" id="pre"></div>
       <div>Transition matrix P <span style="font-size:11px;color:var(--muted)">(row = from, column = to; each row is renormalised to sum to 1)</span><div id="mat" style="margin-top:6px"></div></div>
       <div class="row" id="starts"></div>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Step</button><button class="btn" id="reset">↺ Restart</button></div>
       ${sl('spd', 'Speed (steps per second)', 0.25, 4, 0.25, 1)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="walk" checked> show one random walker</label>
       <div class="readout" id="out"></div>`);
    const cg = AI.canvas($(root, '#g')), cb = AI.canvas($(root, '#b'));
    let names, P, n, start = 0, dist, disp, cur, prev, cnt, total, t, timer = 0, playing = true, pi, spd = 1;
    const rng = AI.rng(11), rnd = () => rng();
    bind(root, 'spd', (v) => v + '×', (v) => (spd = v));
    const stat = () => {
      let p = Array.from({ length: n }, (_, i) => (i === start ? 1 : 0)), acc = new Array(n).fill(0);
      const mul = (p) => p.map((_, j) => { let s = 0; for (let i = 0; i < n; i++) s += p[i] * P[i][j]; return s; });
      for (let k = 0; k < 1500; k++) p = mul(p);
      for (let k = 0; k < 2520; k++) { p = mul(p); for (let j = 0; j < n; j++) acc[j] += p[j] / 2520; }
      return acc;
    };
    const normRow = (i) => { let s = 0; for (let j = 0; j < n; j++) { P[i][j] = Math.max(0, P[i][j]); s += P[i][j]; } if (s <= 0) { P[i].fill(0); P[i][i] = 1; } else for (let j = 0; j < n; j++) P[i][j] = +(P[i][j] / s).toFixed(4); };
    const restart = () => {
      dist = Array.from({ length: n }, (_, i) => (i === start ? 1 : 0)); disp = dist.slice();
      cur = prev = start; cnt = new Array(n).fill(0); cnt[start] = 1; total = 1; t = 0; timer = 0; pi = stat();
      AI.$$('#starts .chip', root).forEach((b, i) => b.classList.toggle('on', i === start));
    };
    const buildMatrix = () => {
      const m = $(root, '#mat'); m.innerHTML = '';
      const grid = document.createElement('div'); grid.style.cssText = `display:grid;grid-template-columns:auto repeat(${n},1fr);gap:4px;align-items:center`;
      grid.appendChild(document.createElement('span'));
      names.forEach((nm) => { const s = document.createElement('span'); s.textContent = '→ ' + nm.replace('Page ', ''); s.style.cssText = 'font-size:11px;color:var(--muted)'; grid.appendChild(s); });
      for (let i = 0; i < n; i++) {
        const l = document.createElement('span'); l.textContent = names[i].replace('Page ', ''); l.style.cssText = 'font-size:11px;color:var(--muted)'; grid.appendChild(l);
        for (let j = 0; j < n; j++) {
          const el = document.createElement('input'); el.type = 'number'; el.step = 0.05; el.min = 0; el.max = 1; el.value = P[i][j]; el.dataset.i = i; el.dataset.j = j; el.style.padding = '4px 5px';
          el.addEventListener('change', () => {
            P[i][j] = Math.max(0, Math.min(1, +el.value || 0)); normRow(i);
            AI.$$('input', grid).forEach((x) => { if (+x.dataset.i === i) x.value = P[i][+x.dataset.j]; });
            pi = stat();
          });
          grid.appendChild(el);
        }
      }
      m.appendChild(grid);
      const st = $(root, '#starts'); st.innerHTML = '<span style="color:var(--muted)">start at:</span>';
      names.forEach((nm, i) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = nm; b.onclick = () => { start = i; restart(); }; st.appendChild(b); });
    };
    const load = (key) => {
      const p = presets[key]; names = p.names; P = p.P.map((r) => r.slice()); n = names.length; start = 0;
      AI.$$('#pre .chip', root).forEach((b) => b.classList.toggle('on', b.textContent === key));
      buildMatrix(); restart();
    };
    Object.keys(presets).forEach((k) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => load(k); $(root, '#pre').appendChild(b); });
    $(root, '#play').onclick = () => { playing = !playing; $(root, '#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    $(root, '#reset').onclick = restart;
    const doStep = () => {
      const nd = new Array(n).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) nd[j] += dist[i] * P[i][j];
      dist = nd; prev = cur;
      let u = rnd(), a = 0, nx = n - 1;
      for (let j = 0; j < n; j++) { a += P[cur][j]; if (u < a) { nx = j; break; } }
      cur = nx; cnt[cur]++; total++; t++;
    };
    $(root, '#step').onclick = () => { doStep(); timer = 0; };
    load('Weather');

    AI.loop(root, (dt) => {
      if (playing) { timer += dt * spd; let k = 0; while (timer >= 1 && k++ < 5) { timer -= 1; doStep(); } }
      const ke = 1 - Math.exp(-dt * 6);
      for (let i = 0; i < n; i++) disp[i] += (dist[i] - disp[i]) * ke;
      const C = AI.col(), pal = [C.blue, C.orange, C.green, C.purple, C.pink, C.teal];
      /* ---- graph ---- */
      let g = cg.ctx; g.clearRect(0, 0, cg.w, cg.h);
      const cx = cg.w / 2, cy = cg.h / 2 + 6, R = Math.min(cg.w * 0.33, cg.h * 0.34);
      const pos = names.map((_, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / n; return [cx + R * Math.cos(a), cy + R * Math.sin(a), a]; });
      const rad = (i) => 21 + 18 * Math.sqrt(disp[i]);
      g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const p = P[i][j]; if (p < 0.004) continue;
        g.strokeStyle = alpha(C.fg, 0.2 + 0.6 * p); g.lineWidth = 1 + 5 * p;
        if (i === j) {
          const a = pos[i][2], r0 = rad(i), lx = pos[i][0] + Math.cos(a) * (r0 + 9), ly = pos[i][1] + Math.sin(a) * (r0 + 9);
          g.beginPath(); g.arc(lx, ly, 11, 0, 7); g.stroke();
          g.fillStyle = C.fg; g.fillText(AI.fmt(p, 2), pos[i][0] + Math.cos(a) * (r0 + 30), pos[i][1] + Math.sin(a) * (r0 + 30) + 4);
        } else {
          const [x0, y0] = pos[i], [x1, y1] = pos[j], dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
          const sx = x0 + ux * rad(i), sy = y0 + uy * rad(i), ex = x1 - ux * (rad(j) + 3), ey = y1 - uy * (rad(j) + 3);
          const mx = (sx + ex) / 2 - uy * len * 0.17, my = (sy + ey) / 2 + ux * len * 0.17;
          g.beginPath(); g.moveTo(sx, sy); g.quadraticCurveTo(mx, my, ex, ey); g.stroke();
          arrowHead(g, ex, ey, Math.atan2(ey - my, ex - mx), alpha(C.fg, 0.2 + 0.6 * p), 5 + 4 * p);
          const lx = 0.25 * sx + 0.5 * mx + 0.25 * ex, ly = 0.25 * sy + 0.5 * my + 0.25 * ey;
          g.fillStyle = C.bg; g.globalAlpha = 0.75; g.fillRect(lx - 14, ly - 8, 28, 15); g.globalAlpha = 1;
          g.fillStyle = C.fg; g.fillText(AI.fmt(p, 2), lx, ly + 4);
        }
      }
      for (let i = 0; i < n; i++) {
        const [x, y] = pos[i], r = rad(i);
        g.beginPath(); g.arc(x, y, r, 0, 7); g.fillStyle = alpha(pal[i % 6], 0.25 + 0.6 * disp[i]); g.fill();
        g.lineWidth = 2; g.strokeStyle = pal[i % 6]; g.stroke();
        g.fillStyle = C.fg; g.font = 'bold 11px sans-serif'; g.fillText(names[i].replace('Page ', ''), x, y - 1);
        g.font = '10px ui-monospace,monospace'; g.fillText(AI.fmt(100 * disp[i], 1) + '%', x, y + 11);
      }
      if ($(root, '#walk').checked) {
        const u = AI.smooth(Math.min(1, timer * 1.6)), a = pos[prev], b = pos[cur];
        let wx = AI.lerp(a[0], b[0], u), wy = AI.lerp(a[1], b[1], u);
        if (prev !== cur) { const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy), bump = Math.sin(Math.PI * u) * l * 0.085 * 2; wx += (-dy / l) * bump; wy += (dx / l) * bump; }
        g.beginPath(); g.arc(wx, wy, 6, 0, 7); g.fillStyle = C.yellow; g.fill(); g.lineWidth = 2; g.strokeStyle = C.bg; g.stroke();
      }
      /* ---- bars ---- */
      g = cb.ctx; g.clearRect(0, 0, cb.w, cb.h);
      const pl = 34, pr = 10, pt = 26, pb = 24, bw = (cb.w - pl - pr) / n, bh = cb.h - pt - pb;
      g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'right'; g.fillStyle = C.muted;
      [0, 0.5, 1].forEach((v) => { const y = pt + bh * (1 - v); g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(pl, y); g.lineTo(cb.w - pr, y); g.stroke(); g.fillText(v, pl - 5, y + 4); });
      g.textAlign = 'left';
      g.fillStyle = C.blue; g.fillText('■ πₜ (exact, from the matrix)', pl, 13);
      g.fillStyle = C.orange; g.fillText('■ walker visit frequency', pl + 190, 13);
      g.fillStyle = C.yellow; g.fillText('— stationary π', pl + 360, 13);
      g.textAlign = 'center';
      for (let i = 0; i < n; i++) {
        const x0 = pl + i * bw + bw * 0.12, w = bw * 0.36;
        const h1 = bh * disp[i], h2 = bh * (cnt[i] / total);
        g.fillStyle = alpha(C.blue, 0.8); g.fillRect(x0, pt + bh - h1, w, h1);
        g.fillStyle = alpha(C.orange, 0.8); g.fillRect(x0 + w + 3, pt + bh - h2, w, h2);
        g.strokeStyle = C.yellow; g.lineWidth = 2.5; g.beginPath(); g.moveTo(x0 - 4, pt + bh * (1 - pi[i])); g.lineTo(x0 + 2 * w + 7, pt + bh * (1 - pi[i])); g.stroke();
        g.fillStyle = C.fg; g.fillText(names[i].replace('Page ', ''), pl + i * bw + bw / 2, cb.h - 8);
      }
      /* ---- text ---- */
      let tv = 0, res = 0;
      for (let j = 0; j < n; j++) { tv += Math.abs(dist[j] - pi[j]) / 2; let s = 0; for (let i = 0; i < n; i++) s += pi[i] * P[i][j]; res += Math.abs(s - pi[j]); }
      const f = (a) => '(' + a.map((v) => AI.fmt(v, 3)).join(', ') + ')';
      $(root, '#out').innerHTML = `t = <b>${t}</b>\nπₜ  = ${f(dist)}\nπ   = ${f(pi)}   <span class="m">stationary</span>\ndistance to π (total variation) = <b>${AI.fmt(tv, 4)}</b>\ncheck ‖πP − π‖₁ = ${AI.fmt(res, 6)}`;
      const ph = timer < 0.5 ? 'each state sends its probability mass along its arrows (thicker = more)' : 'the yellow walker takes one random hop chosen by the row of P for its current state';
      $(root, '#cap').textContent = `step ${t}: ${ph}. The blue bars are the exact πₜ; the orange bars are what one walker actually visited; yellow ticks are the stationary distribution (a left eigenvector of P with eigenvalue 1).`;
    }, { fps: 30 });
  };

  /* ====================================================== EM for a 1-D Gaussian mixture ====================================================== */
  AI.labs.em = function (root) {
    const { view, ctl } = AI.labShell(root, 'Gaussian mixture & EM', 'responsibilities colour the points; EM alternates E-step and M-step, slowly',
      `<canvas style="height:340px;touch-action:none"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">Next phase</button></div>
       <div class="row"><button class="chip" id="r1">↺ reset (mild start)</button><button class="chip" id="r2">↺ bad start</button></div>
       ${sl('spd', 'Speed', 0.25, 2, 0.05, 0.6)}
       <div style="font-size:12px;color:var(--muted)">Drag the two triangles on the x-axis to move the component means yourself, then press play and watch EM take over.</div>
       <div class="readout" id="out"></div>`);
    const c = AI.canvas($(root, 'canvas')), cv = $(root, 'canvas');
    const rg = AI.rng(7), rn = () => AI.randn(rg), xs = [], jit = [];
    for (let i = 0; i < 26; i++) xs.push(-2 + 0.8 * rn());
    for (let i = 0; i < 20; i++) xs.push(2.6 + 1.0 * rn());
    const N = xs.length; for (let i = 0; i < N; i++) jit.push(rg());
    let mu, sg, pi, r, rT, muT, sgT, piT, phase = 'E', ph = 0, iter = 0, playing = true, spd = 0.6, drag = -1;
    bind(root, 'spd', (v) => v + '×', (v) => (spd = v));
    const respond = (m, s, p) => xs.map((x) => { const a = p[0] * pdfN(x, m[0], s[0]), b = p[1] * pdfN(x, m[1], s[1]); return a / (a + b + 1e-300); });
    const mstep = (rr) => {
      const N1 = rr.reduce((a, b) => a + b, 0), N2 = N - N1;
      const m1 = xs.reduce((a, x, i) => a + rr[i] * x, 0) / N1, m2 = xs.reduce((a, x, i) => a + (1 - rr[i]) * x, 0) / N2;
      const v1 = xs.reduce((a, x, i) => a + rr[i] * (x - m1) ** 2, 0) / N1, v2 = xs.reduce((a, x, i) => a + (1 - rr[i]) * (x - m2) ** 2, 0) / N2;
      return { m: [m1, m2], s: [Math.max(0.25, Math.sqrt(v1)), Math.max(0.25, Math.sqrt(v2))], p: [N1 / N, N2 / N] };
    };
    const enterE = () => { phase = 'E'; ph = 0; iter++; rT = respond(mu, sg, pi); };
    const enterM = () => { phase = 'M'; ph = 0; const t = mstep(rT); muT = t.m; sgT = t.s; piT = t.p; };
    const reset = (m) => { mu = m.slice(); sg = [1.2, 1.2]; pi = [0.5, 0.5]; r = xs.map(() => 0.5); iter = 0; enterE(); };
    const next = () => { if (phase === 'E') enterM(); else enterE(); };
    $(root, '#play').onclick = () => { playing = !playing; $(root, '#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    $(root, '#step').onclick = next;
    $(root, '#r1').onclick = () => reset([-0.5, 1.0]);
    $(root, '#r2').onclick = () => reset([3.8, 4.8]);
    reset([-0.5, 1.0]);
    const X0 = -6.5, X1 = 7;
    const vw = () => AI.view(c, X0, X1, 0, 0.3, { l: 30, r: 10, t: 14, b: 40 });
    cv.addEventListener('pointerdown', (e) => {
      const v = vw(), [px, py] = ptr(cv, e); let best = -1, bd = 24;
      for (let k = 0; k < 2; k++) { const d = Math.hypot(px - v.X(mu[k]), py - (v.Y(0) + 16)); if (d < bd) { bd = d; best = k; } }
      if (best >= 0) { drag = best; cv.setPointerCapture(e.pointerId); }
    });
    cv.addEventListener('pointermove', (e) => { if (drag < 0) return; const v = vw(); mu[drag] = AI.clamp(v.ix(ptr(cv, e)[0]), X0, X1); enterE(); });
    cv.addEventListener('pointerup', () => (drag = -1));
    AI.loop(root, (dt) => {
      if (playing && drag < 0) { ph += dt * spd; if (ph >= 2.2) next(); }
      const k = 1 - Math.exp(-dt * 2.2 * Math.max(0.3, spd));
      if (phase === 'E') for (let i = 0; i < N; i++) r[i] += (rT[i] - r[i]) * k;
      else for (let j = 0; j < 2; j++) { mu[j] += (muT[j] - mu[j]) * k; sg[j] += (sgT[j] - sg[j]) * k; pi[j] += (piT[j] - pi[j]) * k; }
      const C = AI.col(), g = c.ctx, v = vw(); g.clearRect(0, 0, c.w, c.h);
      AI.axes(g, v, { noY: true, noGrid: true });
      const cols = [C.blue, C.orange];
      for (let kc = 0; kc < 2; kc++) {
        g.beginPath(); for (let i = 0; i <= 200; i++) { const x = X0 + ((X1 - X0) * i) / 200, y = pi[kc] * pdfN(x, mu[kc], sg[kc]); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
        g.lineTo(v.X(X1), v.Y(0)); g.lineTo(v.X(X0), v.Y(0)); g.closePath(); g.fillStyle = alpha(cols[kc], 0.15); g.fill();
        g.beginPath(); for (let i = 0; i <= 200; i++) { const x = X0 + ((X1 - X0) * i) / 200, y = pi[kc] * pdfN(x, mu[kc], sg[kc]); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
        g.strokeStyle = cols[kc]; g.lineWidth = 2; g.stroke();
      }
      g.beginPath(); for (let i = 0; i <= 200; i++) { const x = X0 + ((X1 - X0) * i) / 200, y = pi[0] * pdfN(x, mu[0], sg[0]) + pi[1] * pdfN(x, mu[1], sg[1]); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
      g.strokeStyle = C.fg; g.setLineDash([5, 4]); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
      for (let i = 0; i < N; i++) {
        const y = v.Y(0) - 8 - jit[i] * 56; g.beginPath(); g.arc(v.X(xs[i]), y, 4.5, 0, 7);
        g.fillStyle = AI.mix(C.orange, C.blue, r[i]); g.fill(); g.lineWidth = 1; g.strokeStyle = C.bg; g.stroke();
      }
      for (let kc = 0; kc < 2; kc++) { const x = v.X(mu[kc]), y = v.Y(0) + 16; g.beginPath(); g.moveTo(x, y - 9); g.lineTo(x - 8, y + 5); g.lineTo(x + 8, y + 5); g.closePath(); g.fillStyle = cols[kc]; g.fill(); g.strokeStyle = C.bg; g.stroke(); }
      let ll = 0; for (let i = 0; i < N; i++) ll += Math.log(pi[0] * pdfN(xs[i], mu[0], sg[0]) + pi[1] * pdfN(xs[i], mu[1], sg[1]) + 1e-300);
      $(root, '#out').innerHTML = `EM iteration <b>${iter}</b>   phase: <b>${phase === 'E' ? 'E-step' : 'M-step'}</b>\n<span class="b">comp 1</span>: μ=${AI.fmt(mu[0], 2)} σ=${AI.fmt(sg[0], 2)} π=${AI.fmt(pi[0], 2)}\n<span class="o">comp 2</span>: μ=${AI.fmt(mu[1], 2)} σ=${AI.fmt(sg[1], 2)} π=${AI.fmt(pi[1], 2)}\nlog-likelihood = <b>${AI.fmt(ll, 2)}</b>  <span class="m">(never decreases)</span>`;
      $(root, '#cap').textContent = phase === 'E'
        ? `E-step: with the components held fixed, each point gets a responsibility r = P(component 1 | x) by Bayes' rule. Blue = "probably component 1", orange = "probably component 2", purple = unsure.`
        : `M-step: with the colours (responsibilities) held fixed, each component moves to the weighted mean of its points, and its width and weight are re-estimated from the same weights.`;
    }, { fps: 30 });
  };

  /* ====================================================== Beta-Bernoulli ====================================================== */
  AI.labs.betabern = function (root) {
    const { view, ctl } = AI.labShell(root, 'Beta–Bernoulli: updating beliefs about a coin', 'prior × likelihood → posterior, one flip at a time',
      `<canvas style="height:330px"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<div class="row"><button class="btn primary" id="flip">Flip the coin</button><button class="btn" id="auto">▶ Auto-flip</button><button class="btn" id="reset">↺ Reset</button></div>
       <div class="row"><button class="chip" id="mh">Force heads</button><button class="chip" id="mt">Force tails</button></div>
       ${sl('p', 'True probability of heads (hidden from the learner)', 0.05, 0.95, 0.05, 0.7)}
       ${sl('a', 'Prior α (pseudo-heads)', 1, 10, 1, 2)}
       ${sl('b', 'Prior β (pseudo-tails)', 1, 10, 1, 2)}
       ${sl('rate', 'Auto-flip rate (flips/s)', 0.25, 4, 0.25, 1)}
       <div class="readout" id="out"></div>`);
    const c = AI.canvas($(root, 'canvas'));
    let h = 0, t = 0, hist = [], pT = 0.7, a0 = 2, b0 = 2, A = 2, B = 2, auto = false, rate = 1, timer = 0, ymax = 3, lastFlip = -9;
    bind(root, 'p', (v) => v, (v) => (pT = v)); bind(root, 'a', null, (v) => (a0 = v)); bind(root, 'b', null, (v) => (b0 = v)); bind(root, 'rate', null, (v) => (rate = v));
    const add = (hd) => { if (hd) h++; else t++; hist.push(hd); if (hist.length > 30) hist.shift(); lastFlip = performance.now(); };
    $(root, '#flip').onclick = () => add(Math.random() < pT);
    $(root, '#mh').onclick = () => add(true); $(root, '#mt').onclick = () => add(false);
    $(root, '#reset').onclick = () => { h = t = 0; hist = []; };
    $(root, '#auto').onclick = () => { auto = !auto; $(root, '#auto').textContent = auto ? '⏸ Stop auto-flip' : '▶ Auto-flip'; };
    const beta = (x, a, b) => Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x));
    AI.loop(root, (dt) => {
      if (auto) { timer += dt * rate; if (timer >= 1) { timer -= 1; add(Math.random() < pT); } }
      const k = 1 - Math.exp(-dt * 5), aT = a0 + h, bT = b0 + t;
      A += (aT - A) * k; B += (bT - B) * k;
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const mode = A > 1 && B > 1 ? (A - 1) / (A + B - 2) : 0.5;
      const tgt = Math.max(2, beta(AI.clamp(mode, 0.01, 0.99), Math.max(A, 1.001), Math.max(B, 1.001)) * 1.15);
      ymax += (tgt - ymax) * k;
      const v = AI.view(c, 0, 1, 0, ymax, { l: 36, r: 14, t: 38, b: 42 });
      AI.axes(g, v, { noGrid: false });
      const curve = (a, b, col, w, fill, dash) => {
        g.beginPath(); for (let i = 1; i < 200; i++) { const x = i / 200, y = Math.min(beta(x, a, b), ymax * 1.5); i > 1 ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); }
        g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]);
        if (fill) { g.lineTo(v.X(0.995), v.Y(0)); g.lineTo(v.X(0.005), v.Y(0)); g.closePath(); g.fillStyle = alpha(col, 0.18); g.fill(); }
      };
      curve(a0, b0, C.muted, 1.5, false, [5, 4]);
      curve(A, B, C.blue, 2.5, true);
      const mark = (x, col, lab, dy) => { g.strokeStyle = col; g.lineWidth = 1.5; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(v.X(x), v.Y(0)); g.lineTo(v.X(x), v.Y(ymax)); g.stroke(); g.setLineDash([]); g.fillStyle = col; g.font = '11px sans-serif'; g.textAlign = 'center'; g.fillText(lab, v.X(x), v.pad.t + 11 + dy); };
      mark(pT, C.green, 'true p', 0);
      if (h + t > 0) mark(h / (h + t), C.orange, 'MLE', 13);
      mark(aT / (aT + bT), C.blue, 'post. mean', 26);
      g.textAlign = 'left'; g.font = '11px sans-serif'; g.fillStyle = C.muted; g.fillText('┄ prior', v.pad.l + 6, v.pad.t + 12); g.fillStyle = C.blue; g.fillText('━ posterior', v.pad.l + 6, v.pad.t + 25);
      g.fillStyle = C.muted; g.fillText('θ = probability of heads', c.w - 170, c.h - 4);
      for (let i = 0; i < hist.length; i++) { const x = c.w - 14 - (hist.length - 1 - i) * 13, last = i === hist.length - 1; g.beginPath(); g.arc(x, 14, last ? 5.5 : 4.5, 0, 7); g.fillStyle = hist[i] ? C.green : C.red; g.globalAlpha = last ? 1 : 0.7; g.fill(); g.globalAlpha = 1; }
      const sd = Math.sqrt((aT * bT) / ((aT + bT) ** 2 * (aT + bT + 1)));
      $(root, '#out').innerHTML = `data: <b>${h}</b> heads, <b>${t}</b> tails\nposterior = Beta(${a0}+${h}, ${b0}+${t}) = Beta(<b>${aT}</b>, <b>${bT}</b>)\nposterior mean = ${AI.fmt(aT / (aT + bT), 4)}   sd = ${AI.fmt(sd, 3)}\nMAP (mode) = ${aT > 1 && bT > 1 ? AI.fmt((aT - 1) / (aT + bT - 2), 4) : '–'}   MLE = ${h + t ? AI.fmt(h / (h + t), 4) : '–'}`;
      $(root, '#cap').textContent = `Conjugacy in action: every head adds 1 to α, every tail adds 1 to β — the posterior stays a Beta, so "learning" is just counting. The prior acts like ${a0 + b0} imaginary flips (${a0} heads, ${b0} tails) you saw before the experiment.`;
    }, { fps: 30 });
  };

  /* ====================================================== Kernel lift (feature map to 3-D) ====================================================== */
  AI.labs.kernellift = function (root) {
    const { view, ctl } = AI.labShell(root, 'Feature map: lifting 2-D data into 3-D', 'φ(x, y) = (x, y, x² + y²) — a plane in 3-D is a circle in 2-D',
      `<div style="display:grid;grid-template-columns:1fr 1.25fr;gap:10px"><canvas id="a" style="height:340px"></canvas><canvas id="b" style="height:340px;touch-action:none;cursor:grab"></canvas></div><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<div class="row"><button class="btn primary" id="lift">▶ Lift the data</button><button class="btn" id="back">⟲ Flatten</button></div>
       ${sl('t', 'Lift amount t (z = t·(x²+y²))', 0, 1, 0.01, 0)}
       ${sl('c', 'Plane height c (separator z = t·c)', 0.1, 4.2, 0.05, 1.5)}
       ${sl('spd', 'Animation speed', 0.25, 2, 0.05, 0.6)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="rot" checked> slowly rotate the 3-D view (or drag it)</label>
       <div class="readout" id="out"></div>`);
    const ca = AI.canvas($(root, '#a')), cb = AI.canvas($(root, '#b')), cvb = $(root, '#b');
    const rg = AI.rng(5), pts = [];
    for (let i = 0; i < 70; i++) { const r = 0.85 * Math.sqrt(rg()), a = 2 * Math.PI * rg(); pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), cls: 0 }); }
    for (let i = 0; i < 70; i++) { const r = 1.5 + 0.5 * rg(), a = 2 * Math.PI * rg(); pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), cls: 1 }); }
    let T = 0, cH = 1.5, ang = 0.6, spd = 0.6, mode = 0, drag = false, lx = 0;
    const tEl = bind(root, 't', (v) => AI.fmt(v, 2), (v) => (T = v));
    bind(root, 'c', (v) => AI.fmt(v, 2), (v) => (cH = v)); bind(root, 'spd', (v) => v + '×', (v) => (spd = v));
    $(root, '#lift').onclick = () => (mode = 1); $(root, '#back').onclick = () => (mode = -1);
    tEl.addEventListener('input', () => (mode = 0));
    cvb.addEventListener('pointerdown', (e) => { drag = true; lx = e.clientX; cvb.setPointerCapture(e.pointerId); });
    cvb.addEventListener('pointermove', (e) => { if (drag) { ang += (e.clientX - lx) * 0.01; lx = e.clientX; } });
    cvb.addEventListener('pointerup', () => (drag = false));
    AI.loop(root, (dt) => {
      if (mode) { T = AI.clamp(T + mode * dt * 0.16 * spd, 0, 1); tEl.value = T; $(root, '#tv').textContent = AI.fmt(T, 2); if (T <= 0 || T >= 1) mode = 0; }
      if ($(root, '#rot').checked && !drag) ang += dt * 0.25 * spd;
      const C = AI.col();
      /* 2-D panel */
      let g = ca.ctx; g.clearRect(0, 0, ca.w, ca.h);
      const s2 = Math.min(ca.w, ca.h) / 5.2, ox = ca.w / 2, oy = ca.h / 2;
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(0, oy); g.lineTo(ca.w, oy); g.moveTo(ox, 0); g.lineTo(ox, ca.h); g.stroke();
      if (T > 0.02) { g.beginPath(); g.arc(ox, oy, Math.sqrt(cH) * s2, 0, 7); g.strokeStyle = C.green; g.lineWidth = 2; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]); }
      let ok = 0;
      for (const p of pts) {
        const z = T * (p.x * p.x + p.y * p.y), pred = z > T * cH ? 1 : 0; if (pred === p.cls) ok++;
        g.beginPath(); g.arc(ox + p.x * s2, oy - p.y * s2, 4.5, 0, 7); g.fillStyle = p.cls ? C.orange : C.blue; g.fill();
        if (pred !== p.cls) { g.strokeStyle = C.red; g.lineWidth = 2; g.stroke(); }
      }
      g.fillStyle = C.muted; g.font = '12px sans-serif'; g.fillText('original 2-D space', 8, 16);
      if (T > 0.02) { g.fillStyle = C.green; g.fillText('green circle = the plane z = t·c seen from above', 8, ca.h - 8); }
      /* 3-D panel */
      g = cb.ctx; g.clearRect(0, 0, cb.w, cb.h);
      const ca_ = Math.cos(ang), sa = Math.sin(ang), el = 0.55, ce = Math.cos(el), se = Math.sin(el), s3 = Math.min(cb.w / 6, cb.h / 7.2);
      const proj = (x, y, z) => { const X = x * ca_ - y * sa, Y = x * sa + y * ca_, zz = z - 1.8; return [cb.w / 2 + X * s3, cb.h * 0.54 - (zz * ce + Y * se) * s3, Y * ce - zz * se]; };
      const poly = (arr, fill, stroke) => { g.beginPath(); arr.forEach((q, i) => { const p = proj(...q); i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.closePath(); if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.stroke(); } };
      poly([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], null, C.border);
      const ax = (a, b, col, lab) => { const p = proj(...a), q = proj(...b); AI.arrow(g, p[0], p[1], q[0], q[1], col, 1.5); g.fillStyle = col; g.font = '11px sans-serif'; g.fillText(lab, q[0] + 4, q[1] - 4); };
      ax([-2, -2, 0], [2.4, -2, 0], C.muted, 'x'); ax([-2, -2, 0], [-2, 2.4, 0], C.muted, 'y'); ax([-2, -2, 0], [-2, -2, 4.4], C.muted, 'z');
      const order = pts.map((p) => ({ p, q: proj(p.x, p.y, T * (p.x * p.x + p.y * p.y)) })).sort((u, w) => w.q[2] - u.q[2]);
      for (const { p, q } of order) {
        const f = proj(p.x, p.y, 0);
        if (T > 0.05) { g.strokeStyle = alpha(C.muted, 0.25); g.beginPath(); g.moveTo(f[0], f[1]); g.lineTo(q[0], q[1]); g.stroke(); }
        g.beginPath(); g.arc(q[0], q[1], 4.5, 0, 7); g.fillStyle = p.cls ? C.orange : C.blue; g.fill();
      }
      poly([[-2.1, -2.1, T * cH], [2.1, -2.1, T * cH], [2.1, 2.1, T * cH], [-2.1, 2.1, T * cH]], alpha(C.green, 0.2), C.green);
      g.fillStyle = C.muted; g.font = '12px sans-serif'; g.fillText('lifted 3-D feature space', 8, 16);
      $(root, '#out').innerHTML = `φ(x, y) = (x, y, <b>${AI.fmt(T, 2)}</b>·(x²+y²))
plane z = ${AI.fmt(T * cH, 2)}  ⇔  circle radius ${AI.fmt(Math.sqrt(cH), 2)}
accuracy (above/below plane): <b>${AI.fmt((100 * ok) / pts.length, 1)}%</b>`;
      $(root, '#cap').textContent = T < 0.05 ? 'Flat (t = 0): blue points are surrounded by orange — no straight line in the plane can separate them. Press "Lift the data".' : T < 0.95 ? 'Lifting: each point rises by t·(x²+y²). Points far from the centre rise high, central points stay low.' : 'Fully lifted: the blue points sit low, the orange ring sits high. A flat plane now separates them perfectly — a LINEAR classifier in 3-D, a CIRCLE in 2-D.';
    }, { fps: 30 });
  };

  /* ====================================================== Kernel matrix heatmap ====================================================== */
  AI.labs.kernelmat = function (root) {
    const { view, ctl } = AI.labShell(root, 'The kernel (Gram) matrix', 'K[i][j] = k(xᵢ, xⱼ) for 24 points on a line; a valid kernel gives a positive semi-definite K',
      `<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><canvas id="h" style="height:300px"></canvas><canvas id="p" style="height:300px"></canvas></div><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<label>Kernel <select id="kind"><option value="rbf">Gaussian (RBF)  exp(−d²/2ℓ²)</option><option value="lap">Laplacian  exp(−|d|/ℓ)</option><option value="lin">Linear  x·x′</option><option value="poly">Polynomial  (1 + x·x′)²</option><option value="bad">NOT a kernel:  −|x − x′|</option></select></label>
       ${sl('l', 'Length-scale ℓ', 0.2, 3, 0.05, 1)}
       <div class="readout" id="out"></div>`);
    const ch = AI.canvas($(root, '#h')), cp = AI.canvas($(root, '#p'));
    ch.onresize = cp.onresize = () => (key = '');
    const n = 24, xs = Array.from({ length: n }, (_, i) => -3 + (6 * i) / (n - 1));
    let ell = 1, kind = 'rbf', key = '', eig = [], K = [];
    bind(root, 'l', (v) => AI.fmt(v, 2), (v) => (ell = v));
    $(root, '#kind').onchange = (e) => (kind = e.target.value);
    const kf = (a, b) => { const d = a - b; switch (kind) { case 'rbf': return Math.exp(-d * d / (2 * ell * ell)); case 'lap': return Math.exp(-Math.abs(d) / ell); case 'lin': return a * b; case 'poly': return (1 + a * b) ** 2; default: return -Math.abs(d); } };
    AI.loop(root, () => {
      const k2 = kind + ell; if (k2 === key) return; key = k2;
      K = xs.map((a) => xs.map((b) => kf(a, b))); eig = jacobiEig(K, n);
      const C = AI.col();
      let g = ch.ctx; g.clearRect(0, 0, ch.w, ch.h);
      const pad = 26, cs = Math.min(ch.w - pad - 4, ch.h - pad - 4) / n; let mx = 1e-9; K.forEach((r) => r.forEach((v) => (mx = Math.max(mx, Math.abs(v)))));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = K[i][j] / mx; g.fillStyle = v >= 0 ? AI.mix(C.bg2, C.blue, Math.min(1, v)) : AI.mix(C.bg2, C.orange, Math.min(1, -v)); g.fillRect(pad + j * cs, pad + i * cs, cs + 0.5, cs + 0.5); }
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillText('K  (rows & columns = points x₁…x₂₄, left to right; blue = large positive, orange = negative)', 2, 12);
      g.fillText('x₁', pad, pad - 3); g.save(); g.translate(pad - 5, pad + 14); g.rotate(-Math.PI / 2); g.fillText('x₁', 0, 0); g.restore();
      /* eigenvalue bars */
      g = cp.ctx; g.clearRect(0, 0, cp.w, cp.h);
      const emax = Math.max(1e-9, ...eig.map(Math.abs)), bw = (cp.w - 40) / n, y0 = cp.h * 0.55, hh = cp.h * 0.4;
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(30, y0); g.lineTo(cp.w - 6, y0); g.stroke();
      eig.forEach((e, i) => { const h = (e / emax) * hh; g.fillStyle = e < -1e-8 ? C.red : C.green; g.fillRect(34 + i * bw, h >= 0 ? y0 - h : y0, bw - 2, Math.abs(h)); });
      g.fillStyle = C.muted; g.font = '11px sans-serif'; g.fillText('eigenvalues of K, largest first (green ≥ 0, red < 0)', 6, 14);
      g.fillText('kernel profile k(0, x′)', 6, cp.h - 8);
      const vv = AI.view(cp, -3, 3, -1.2, 1.2, { l: 30, r: 6, t: cp.h * 0.6, b: 22 });
      g.beginPath(); for (let i = 0; i <= 120; i++) { const x = -3 + i * 0.05, y = AI.clamp(kf(0, x) / (kind === 'lin' || kind === 'poly' || kind === 'bad' ? 3 : 1), -1.2, 1.2); i ? g.lineTo(vv.X(x), vv.Y(y)) : g.moveTo(vv.X(x), vv.Y(y)); }
      g.strokeStyle = C.blue; g.lineWidth = 2; g.stroke();
      const mn = eig[n - 1];
      $(root, '#out').innerHTML = `largest eigenvalue = ${AI.fmt(eig[0], 3)}\nsmallest eigenvalue = <b class="${mn < -1e-8 ? 'r' : 'g'}">${AI.fmt(mn, 4)}</b>\ntrace = ${AI.fmt(eig.reduce((a, b) => a + b, 0), 3)}  (= sum of diagonal)\nrank ≈ ${eig.filter((e) => Math.abs(e) > 1e-8 * emax).length} of ${n}`;
      $(root, '#cap').textContent = kind === 'bad' ? 'A function that merely looks like "distance" is not automatically a kernel: this matrix has NEGATIVE eigenvalues, so no feature map φ can reproduce it (and an SVM/GP built on it can break).' : kind === 'rbf' ? 'RBF: nearby points are similar (bright diagonal band). Small ℓ → nearly the identity matrix (everyone is a stranger); large ℓ → nearly all ones (everyone alike, rank collapses). All eigenvalues stay ≥ 0.' : 'All eigenvalues are ≥ 0 (up to rounding error ~1e-15): a valid kernel always gives a positive semi-definite Gram matrix. Linear and polynomial kernels have low rank because their feature spaces are small (1 and 3 dimensions).';
    });
  };

  /* ====================================================== Gaussian-process regression ====================================================== */
  AI.labs.gp = function (root) {
    const { view, ctl } = AI.labShell(root, 'Gaussian-process regression', 'click to add data, drag to move, double-click to delete — the posterior mean and uncertainty react live',
      `<canvas style="height:380px;touch-action:none;cursor:crosshair"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `${sl('l', 'Length-scale ℓ (how fast f may wiggle)', 0.3, 3, 0.05, 1)}
       ${sl('sn', 'Observation noise σₙ', 0.02, 1, 0.01, 0.1)}
       ${sl('sf', 'Signal amplitude σ_f', 0.5, 2, 0.05, 1)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="smp" checked> draw 3 sample functions from the posterior</label>
       <div class="row"><button class="btn" id="new">New samples</button><button class="btn" id="clr">Clear points</button><button class="btn" id="rst">Reset points</button></div>
       <div class="readout" id="out"></div>`);
    const cv = $(root, 'canvas'), c = AI.canvas(cv), G = 80, XL = -5, XR = 5, YL = -3.2, YH = 3.2;
    const gx = Array.from({ length: G }, (_, i) => XL + ((XR - XL) * i) / (G - 1));
    const init = () => [[-3.5, 0.6], [-1.5, -0.8], [0.5, 1.0], [2.5, 0.2]];
    c.onresize = () => (key = '');
    let pts = init(), ell = 1, sn = 0.1, sf = 1, drag = -1, key = '', Z = [0, 1, 2].map(() => gx.map(() => AI.randn())), res = null;
    bind(root, 'l', (v) => AI.fmt(v, 2), (v) => (ell = v)); bind(root, 'sn', (v) => AI.fmt(v, 2), (v) => (sn = v)); bind(root, 'sf', (v) => AI.fmt(v, 2), (v) => (sf = v));
    $(root, '#new').onclick = () => { Z = [0, 1, 2].map(() => gx.map(() => AI.randn())); key = ''; };
    $(root, '#clr').onclick = () => (pts = []); $(root, '#rst').onclick = () => (pts = init());
    const vw = () => AI.view(c, XL, XR, YL, YH, { l: 34, r: 10, t: 10, b: 26 });
    cv.addEventListener('pointerdown', (e) => {
      const v = vw(), [px, py] = ptr(cv, e); let best = -1, bd = 14;
      pts.forEach((p, i) => { const d = Math.hypot(px - v.X(p[0]), py - v.Y(p[1])); if (d < bd) { bd = d; best = i; } });
      if (best < 0 && pts.length < 14 && px > v.pad.l && px < v.pad.l + v.pw) { pts.push([v.ix(px), AI.clamp(v.iy(py), YL, YH)]); best = pts.length - 1; }
      if (best >= 0) { drag = best; cv.setPointerCapture(e.pointerId); }
    });
    cv.addEventListener('pointermove', (e) => { if (drag < 0) return; const v = vw(), [px, py] = ptr(cv, e); pts[drag] = [AI.clamp(v.ix(px), XL, XR), AI.clamp(v.iy(py), YL, YH)]; });
    cv.addEventListener('pointerup', () => (drag = -1));
    cv.addEventListener('dblclick', (e) => { const v = vw(), [px, py] = ptr(cv, e); pts = pts.filter((p) => Math.hypot(px - v.X(p[0]), py - v.Y(p[1])) > 14); });
    const k = (a, b) => sf * sf * Math.exp(-((a - b) ** 2) / (2 * ell * ell));
    function solve() {
      const n = pts.length; let mean = new Float64Array(G), sd = new Float64Array(G), lml = 0, V = [];
      let L = null, al = null;
      if (n) {
        const A = pts.map((p, i) => pts.map((q, j) => k(p[0], q[0]) + (i === j ? sn * sn : 0))); L = chol(A, n);
        const y = Float64Array.from(pts.map((p) => p[1])); al = solveLT(L, solveL(L, y, n), n);
        let yy = 0; for (let i = 0; i < n; i++) { yy += y[i] * al[i]; lml -= Math.log(L[i][i]); }
        lml += -0.5 * yy - 0.5 * n * Math.log(2 * Math.PI);
      }
      for (let g = 0; g < G; g++) {
        if (!n) { V.push(new Float64Array(0)); mean[g] = 0; continue; }
        const ks = Float64Array.from(pts.map((p) => k(gx[g], p[0]))), v = solveL(L, ks, n); V.push(v);
        let m = 0; for (let i = 0; i < n; i++) m += ks[i] * al[i]; mean[g] = m;
      }
      const cov = gx.map((a, i) => { const r = new Float64Array(G); for (let j = 0; j < G; j++) { let s = k(a, gx[j]); for (let m = 0; m < n; m++) s -= V[i][m] * V[j][m]; r[j] = s + (i === j ? 1e-6 : 0); } return r; });
      for (let g = 0; g < G; g++) sd[g] = Math.sqrt(Math.max(0, cov[g][g]));
      const Lc = chol(cov, G), samples = Z.map((z) => { const f = new Float64Array(G); for (let i = 0; i < G; i++) { let s = 0; for (let j = 0; j <= i; j++) s += Lc[i][j] * z[j]; f[i] = mean[i] + s; } return f; });
      return { mean, sd, samples, lml };
    }
    AI.loop(root, () => {
      const kk = JSON.stringify([pts, ell, sn, sf, Z[0][0]]) + $(root, '#smp').checked; if (kk !== key) { key = kk; res = solve(); } else return;
      const C = AI.col(), g = c.ctx, v = vw(); g.clearRect(0, 0, c.w, c.h); AI.axes(g, v);
      g.beginPath(); gx.forEach((x, i) => { const y = v.Y(AI.clamp(res.mean[i] + 2 * res.sd[i], YL, YH)); i ? g.lineTo(v.X(x), y) : g.moveTo(v.X(x), y); });
      for (let i = G - 1; i >= 0; i--) g.lineTo(v.X(gx[i]), v.Y(AI.clamp(res.mean[i] - 2 * res.sd[i], YL, YH)));
      g.closePath(); g.fillStyle = alpha(C.blue, 0.2); g.fill();
      if ($(root, '#smp').checked) res.samples.forEach((f, si) => { g.beginPath(); gx.forEach((x, i) => { const y = v.Y(AI.clamp(f[i], YL, YH)); i ? g.lineTo(v.X(x), y) : g.moveTo(v.X(x), y); }); g.strokeStyle = alpha([C.purple, C.teal, C.pink][si], 0.8); g.lineWidth = 1.3; g.stroke(); });
      g.beginPath(); gx.forEach((x, i) => { const y = v.Y(AI.clamp(res.mean[i], YL, YH)); i ? g.lineTo(v.X(x), y) : g.moveTo(v.X(x), y); }); g.strokeStyle = C.blue; g.lineWidth = 3; g.stroke();
      pts.forEach((p) => { g.beginPath(); g.arc(v.X(p[0]), v.Y(p[1]), 6, 0, 7); g.fillStyle = C.orange; g.fill(); g.lineWidth = 2; g.strokeStyle = C.bg; g.stroke(); });
      g.fillStyle = C.blue; g.font = '11px sans-serif'; g.fillText('━ posterior mean   ▒ ±2σ band (uncertainty about f)', v.pad.l + 6, v.pad.t + 12);
      $(root, '#out').innerHTML = `${pts.length} observation(s)\nlog marginal likelihood = <b>${pts.length ? AI.fmt(res.lml, 2) : '–'}</b>\nmax uncertainty (σ) = ${AI.fmt(Math.max(...res.sd), 2)}`;
      $(root, '#cap').textContent = 'Far from data the band widens back to the prior (±2σ_f); near data it pinches. Small ℓ → wiggly fits and a narrow zone of influence; large ℓ → smooth fits. Larger σₙ lets the mean ignore noisy points instead of passing through them.';
    }, { fps: 30 });
  };

  /* ====================================================== VAE / ELBO ====================================================== */
  AI.labs.vae = function (root) {
    const { view, ctl } = AI.labShell(root, 'VAE for one data point: encoder, sampling, decoder, ELBO', 'encoder q(z|x) = N(μ, σ²) · prior p(z) = N(0, 1) · decoder p(x|z) = N(d·z, s²)',
      `<canvas id="cz" style="height:250px"></canvas><canvas id="cx" style="height:130px"></canvas><canvas id="ce" style="height:190px"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `${sl('x', 'Data point x', -3, 3, 0.1, 1.5)}
       ${sl('mu', 'Encoder mean μ', -3, 3, 0.05, 1)}
       ${sl('sg', 'Encoder std σ', 0.05, 2, 0.05, 0.5)}
       ${sl('d', 'Decoder gain d', 0.2, 2, 0.05, 1)}
       ${sl('s', 'Decoder noise s', 0.2, 1.5, 0.05, 0.5)}
       ${sl('beta', 'β (weight on the KL term)', 0, 4, 0.1, 1)}
       <div class="row"><button class="chip" id="snap">snap q to the true posterior</button><button class="chip" id="pri">q = prior</button></div>
       <div class="row"><button class="btn primary" id="play">⏸ Pause</button><button class="btn" id="step">New ε</button></div>
       ${sl('spd', 'Sampling speed', 0.25, 2, 0.05, 0.6)}
       <div class="readout" id="out"></div>`);
    const cz = AI.canvas($(root, '#cz')), cx = AI.canvas($(root, '#cx')), ce = AI.canvas($(root, '#ce'));
    let X = 1.5, mu = 1, sg = 0.5, d = 1, s = 0.5, beta = 1, spd = 0.6, playing = true, eps = 0.8, ph = 0, phase = 0;
    const set = (id, v) => { const el = $(root, '#' + id); el.value = v; el.dispatchEvent(new Event('input')); };
    bind(root, 'x', null, (v) => (X = v)); bind(root, 'mu', null, (v) => (mu = v)); bind(root, 'sg', null, (v) => (sg = v)); bind(root, 'd', null, (v) => (d = v)); bind(root, 's', null, (v) => (s = v)); bind(root, 'beta', null, (v) => (beta = v)); bind(root, 'spd', (v) => v + '×', (v) => (spd = v));
    const post = () => { const pr = 1 + (d * d) / (s * s); return { m: ((d * X) / (s * s)) / pr, sd: Math.sqrt(1 / pr) }; };
    $(root, '#snap').onclick = () => { const p = post(); set('mu', AI.clamp(+p.m.toFixed(2), -3, 3)); set('sg', AI.clamp(+p.sd.toFixed(2), 0.05, 2)); };
    $(root, '#pri').onclick = () => { set('mu', 0); set('sg', 1); };
    const newEps = () => { eps = AI.randn(); ph = 0; phase = 0; };
    $(root, '#play').onclick = () => { playing = !playing; $(root, '#play').textContent = playing ? '⏸ Pause' : '▶ Play'; };
    $(root, '#step').onclick = newEps;
    const DUR = 1.4;
    AI.loop(root, (dt) => {
      if (playing) { ph += dt * spd; if (ph >= DUR) { ph = 0; phase++; if (phase > 3) newEps(); } }
      const C = AI.col(), z = mu + sg * eps, xh = d * z;
      /* --- latent canvas --- */
      let g = cz.ctx; g.clearRect(0, 0, cz.w, cz.h);
      const P = post(), ymx = Math.max(0.42, 1 / (sg * SQ2PI), 1 / (P.sd * SQ2PI)) * 1.08;
      const v = AI.view(cz, -4, 4, 0, ymx, { l: 20, r: 14, t: 78, b: 22 }), top = { y0: 62, h: 34 };
      AI.axes(g, v, { noY: true, noGrid: true });
      const fillc = (f, col, a, dash) => { g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -4 + i * 0.05, y = f(x); i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)); } g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.stroke(); g.setLineDash([]); if (a) { g.lineTo(v.X(4), v.Y(0)); g.lineTo(v.X(-4), v.Y(0)); g.closePath(); g.fillStyle = alpha(col, a); g.fill(); } };
      fillc((x) => pdfN(x, 0, 1), C.muted, 0.1, [5, 4]);
      fillc((x) => pdfN(x, P.m, P.sd), C.green, 0, [3, 3]);
      fillc((x) => pdfN(x, mu, sg), C.blue, 0.2);
      g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.muted; g.fillText('┄ prior p(z)', 24, 74 + 18);
      g.fillStyle = C.green; g.fillText('┄ true posterior p(z|x)', 100, 74 + 18); g.fillStyle = C.blue; g.fillText('━ encoder q(z|x)', 240, 74 + 18);
      /* eps strip */
      g.strokeStyle = C.border; g.beginPath(); g.moveTo(v.X(-4), top.y0 - 6); g.lineTo(v.X(4), top.y0 - 6); g.stroke();
      g.beginPath(); for (let i = 0; i <= 160; i++) { const x = -4 + i * 0.05, y = top.y0 - 6 - pdfN(x, 0, 1) * 70; i ? g.lineTo(v.X(x), y) : g.moveTo(v.X(x), y); } g.strokeStyle = C.muted; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = C.muted; g.fillText('ε ~ N(0, 1)  (the only randomness; no parameters inside)', 24, 14);
      const eAlpha = phase === 0 ? AI.smooth(ph / 0.7) : 1;
      const ex = v.X(eps), ey = top.y0 - 6, zx = v.X(z), zy = v.Y(0);
      const u = phase === 1 ? AI.smooth(ph / DUR) : phase > 1 ? 1 : 0;
      g.globalAlpha = eAlpha; g.beginPath(); g.arc(AI.lerp(ex, zx, u), AI.lerp(ey, zy - 4, u), 7, 0, 7); g.fillStyle = C.yellow; g.fill(); g.lineWidth = 2; g.strokeStyle = C.bg; g.stroke(); g.globalAlpha = 1;
      if (phase >= 1) { g.strokeStyle = alpha(C.yellow, 0.5); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(ex, ey); g.lineTo(AI.lerp(ex, zx, u), AI.lerp(ey, zy - 4, u)); g.stroke(); g.setLineDash([]); }
      g.fillStyle = C.yellow; g.textAlign = 'center';
      g.fillText(phase === 0 ? `① draw ε = ${AI.fmt(eps, 2)}` : phase === 1 ? `② z = μ + σ·ε = ${AI.fmt(mu, 2)} + ${AI.fmt(sg, 2)}·${AI.fmt(eps, 2)}` : `z = ${AI.fmt(z, 2)}`, AI.clamp(AI.lerp(ex, zx, u), 90, cz.w - 90), AI.lerp(ey - 12, zy - 16, u));
      /* --- decoder canvas --- */
      g = cx.ctx; g.clearRect(0, 0, cx.w, cx.h);
      const w2 = AI.view(cx, -5, 5, 0, 1.4, { l: 20, r: 14, t: 18, b: 22 }); AI.axes(g, w2, { noY: true, noGrid: true });
      const show = phase >= 2;
      if (show) {
        g.beginPath(); for (let i = 0; i <= 200; i++) { const x = -5 + i * 0.05, y = pdfN(x, xh, s) * s * 1.2; i ? g.lineTo(w2.X(x), w2.Y(y)) : g.moveTo(w2.X(x), w2.Y(y)); } g.strokeStyle = C.orange; g.lineWidth = 2; g.stroke();
        g.beginPath(); g.arc(w2.X(xh), w2.Y(0) - 4, 6, 0, 7); g.fillStyle = C.orange; g.fill();
      }
      g.strokeStyle = C.green; g.lineWidth = 3; g.beginPath(); g.moveTo(w2.X(X), w2.Y(0)); g.lineTo(w2.X(X), w2.Y(1.3)); g.stroke();
      g.font = '11px sans-serif'; g.textAlign = 'left'; g.fillStyle = C.green; g.fillText('data x = ' + AI.fmt(X, 2), 24, 12);
      g.fillStyle = C.orange; g.fillText(show ? `③ decoder: x̂ = d·z = ${AI.fmt(xh, 2)}   (error ${AI.fmt(Math.abs(X - xh), 2)}) — curve is p(x|z)` : '③ decoder p(x|z) appears after z is sampled', 120, 12);
      /* --- ELBO bars --- */
      const rec = -0.5 * Math.log(2 * Math.PI * s * s) - ((X - d * mu) ** 2 + d * d * sg * sg) / (2 * s * s);
      const kl = 0.5 * (mu * mu + sg * sg - 1 - Math.log(sg * sg));
      const elbo = rec - beta * kl, lp = -0.5 * Math.log(2 * Math.PI * (d * d + s * s)) - (X * X) / (2 * (d * d + s * s));
      g = ce.ctx; g.clearRect(0, 0, ce.w, ce.h);
      const rows = [['reconstruction  E[log p(x|z)]', rec, C.green], ['− β·KL(q‖p)  (regulariser)', -beta * kl, C.orange], [beta === 1 ? 'ELBO = sum' : 'β-ELBO = sum', elbo, C.blue], ['log p(x)  (evidence, the target)', lp, C.purple]];
      const mxv = Math.max(4, ...rows.map((r) => Math.abs(r[1]))), L0 = 230, wR = ce.w - L0 - 70, rh = (ce.h - 20) / 4;
      g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'left';
      rows.forEach((r, i) => {
        const y = 10 + i * rh, wd = (Math.min(Math.abs(r[1]), mxv) / mxv) * wR;
        g.fillStyle = C.fg; g.fillText(r[0], 6, y + rh * 0.55); g.fillStyle = alpha(r[2], 0.8); g.fillRect(L0 + wR - wd, y + 4, wd, rh - 10);
        g.fillStyle = C.fg; g.fillText(AI.fmt(r[1], 3), L0 + wR + 6, y + rh * 0.55);
      });
      g.strokeStyle = C.muted; g.beginPath(); g.moveTo(L0 + wR, 6); g.lineTo(L0 + wR, ce.h - 8); g.stroke();
      g.fillStyle = C.muted; g.fillText('0', L0 + wR - 4, ce.h - 0); g.fillText('← more negative = worse', L0, ce.h - 0);
      const gap = lp - elbo;
      $(root, '#out').innerHTML = `z = μ+σε = ${AI.fmt(z, 3)}   x̂ = ${AI.fmt(xh, 3)}\nreconstruction = ${AI.fmt(rec, 3)}\nKL(q‖p) = ½(μ²+σ²−1−ln σ²) = <b>${AI.fmt(kl, 3)}</b>\n${beta === 1 ? 'ELBO' : 'β-ELBO'} = ${AI.fmt(elbo, 3)}    log p(x) = ${AI.fmt(lp, 3)}\n${beta === 1 ? `gap = log p(x) − ELBO = <b>${AI.fmt(gap, 4)}</b> = KL(q ‖ p(z|x))` : '<span class="m">(gap formula holds exactly only for β = 1)</span>'}\ntrue posterior: N(${AI.fmt(P.m, 3)}, ${AI.fmt(P.sd * P.sd, 3)})`;
      $(root, '#cap').textContent = beta === 1 ? 'Tune μ and σ until the blue encoder sits on the green true posterior: the ELBO touches log p(x) and the gap (a KL divergence, always ≥ 0) closes.' : 'With β ≠ 1 we trade reconstruction against how close q must stay to the prior: β > 1 squeezes q toward N(0,1) (smoother, blurrier latent space), β < 1 lets q specialise (sharper, but a messier latent space).';
    }, { fps: 30 });
  };

  /* ====================================================== Diffusion ====================================================== */
  AI.labs.diffusion = function (root) {
    const { view, ctl } = AI.labShell(root, 'Diffusion: noising a shape, then denoising it with the score', 'x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε   ·   reverse: follow the score ∇log p_t, plus a little noise',
      `<canvas style="height:420px"></canvas><div class="readout sub" id="cap" style="white-space:normal"></div>`,
      `<div class="chips" id="shp"></div>
       <div class="row"><button class="btn primary" id="fw">▶ Noise it</button><button class="btn primary" id="rv">▶ Generate (denoise)</button><button class="btn" id="stop">⏸ Pause</button></div>
       ${sl('t', 'Timestep t (drag to scrub the forward process)', 0, 200, 1, 0)}
       ${sl('spd', 'Speed (steps per second)', 2, 60, 1, 14)}
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="arr" checked> show the score field ∇log pₜ (arrows)</label>
       <label style="flex-direction:row;gap:6px;align-items:center"><input type="checkbox" id="inj" checked> inject fresh noise z each reverse step (DDPM)</label>
       <div class="readout" id="out"></div>`);
    const cv = $(root, 'canvas'), c = AI.canvas(cv), N = 300, T = 200;
    const beta = new Float64Array(T + 1), alpha_ = new Float64Array(T + 1), abar = new Float64Array(T + 1); abar[0] = 1;
    for (let i = 1; i <= T; i++) { beta[i] = 0.0005 + (0.0995 * (i - 1)) / (T - 1); alpha_[i] = 1 - beta[i]; abar[i] = abar[i - 1] * alpha_[i]; }
    const X0x = new Float64Array(N), X0y = new Float64Array(N), lab = new Uint8Array(N), Ex = new Float64Array(N), Ey = new Float64Array(N), Yx = new Float64Array(N), Yy = new Float64Array(N), d2 = new Float64Array(N);
    const rg = AI.rng(3), rn = () => AI.randn(rg);
    for (let i = 0; i < N; i++) { Ex[i] = rn(); Ey[i] = rn(); }
    const shapes = {
      Moons: (i) => { const half = i < N / 2, a = Math.PI * (half ? i : i - N / 2) / (N / 2 - 1); return half ? [Math.cos(a), Math.sin(a), 0] : [1 - Math.cos(a), 0.5 - Math.sin(a), 1]; },
      Spiral: (i) => { const a = 0.4 + (i / N) * 3.2 * Math.PI, r = a * 0.32; return [r * Math.cos(a), r * Math.sin(a), i < N / 2 ? 0 : 1]; },
      Ring: (i) => { if (i < N * 0.75) { const a = (2 * Math.PI * i) / (N * 0.75); return [1.5 * Math.cos(a), 1.5 * Math.sin(a), 0]; } const a = (2 * Math.PI * (i - N * 0.75)) / (N * 0.25) * 2.7; return [0.35 * Math.cos(a) * ((i - N * 0.75) / (N * 0.25)), 0.35 * Math.sin(a) * ((i - N * 0.75) / (N * 0.25)), 1]; },
    };
    let t = 0, mode = 'fwd', playing = false, spd = 14, tc = T, shape = 'Moons';
    const tick = AI.ticker(), jr = AI.rng(9);
    const load = (name) => {
      shape = name;
      let mx = 0, my = 0, vv = 0; const tmp = [];
      for (let i = 0; i < N; i++) { const p = shapes[name](i); const jx = (jr() - 0.5) * 0.06, jy = (jr() - 0.5) * 0.06; tmp.push([p[0] + jx, p[1] + jy, p[2]]); mx += tmp[i][0] / N; my += tmp[i][1] / N; }
      for (const p of tmp) vv += ((p[0] - mx) ** 2 + (p[1] - my) ** 2) / (2 * N);
      const sd = Math.sqrt(vv);
      tmp.forEach((p, i) => { X0x[i] = (p[0] - mx) / sd; X0y[i] = (p[1] - my) / sd; lab[i] = p[2]; });
      mode = 'fwd'; playing = false; t = 0; $(root, '#t').value = 0;
      AI.$$('#shp .chip', root).forEach((b) => b.classList.toggle('on', b.textContent === name));
    };
    Object.keys(shapes).forEach((k) => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = k; b.onclick = () => load(k); $(root, '#shp').appendChild(b); });
    bind(root, 'spd', (v) => v + ' steps/s', (v) => (spd = v));
    const tEl = bind(root, 't', (v) => 't = ' + v, (v) => { mode = 'fwd'; playing = false; t = v; });
    $(root, '#fw').onclick = () => { mode = 'fwd'; if (t >= T - 0.5) t = 0; playing = true; };
    $(root, '#rv').onclick = () => { if (mode !== 'rev' || tc <= 0) { mode = 'rev'; tc = T; for (let i = 0; i < N; i++) { Yx[i] = rn(); Yy[i] = rn(); } t = T; } playing = true; };
    $(root, '#stop').onclick = () => (playing = false);
    load('Moons');
    const abAt = (tt) => { const i = Math.min(T - 1, Math.floor(tt)), f = tt - i; return AI.lerp(abar[i], abar[i + 1], f); };
    function score(x, y, ab, out) {
      const sa = Math.sqrt(ab), v = Math.max(1 - ab, 2e-4); let best = 1e18;
      for (let i = 0; i < N; i++) { const dx = x - sa * X0x[i], dy = y - sa * X0y[i]; const q = dx * dx + dy * dy; d2[i] = q; if (q < best) best = q; }
      let sw = 0, sx = 0, sy = 0;
      for (let i = 0; i < N; i++) { const w = Math.exp(-(d2[i] - best) / (2 * v)); sw += w; sx += w * (sa * X0x[i] - x); sy += w * (sa * X0y[i] - y); }
      out[0] = sx / (sw * v); out[1] = sy / (sw * v);
    }
    const sc = [0, 0];
    AI.loop(root, (dt) => {
      if (playing) {
        if (mode === 'fwd') { t = Math.min(T, t + dt * spd); if (t >= T) playing = false; }
        else {
          const n = tick(dt, spd), inj = $(root, '#inj').checked;
          for (let s = 0; s < n && tc > 0; s++) {
            const ab = abar[tc], a = alpha_[tc], b = beta[tc];
            for (let i = 0; i < N; i++) {
              score(Yx[i], Yy[i], ab, sc);
              const nz = inj && tc > 1 ? Math.sqrt(b) : 0;
              Yx[i] = (Yx[i] + b * sc[0]) / Math.sqrt(a) + nz * AI.randn(); Yy[i] = (Yy[i] + b * sc[1]) / Math.sqrt(a) + nz * AI.randn();
            }
            tc--;
          }
          t = tc; if (tc <= 0) playing = false;
        }
        tEl.value = Math.round(t); $(root, '#tv').textContent = 't = ' + Math.round(t);
      }
      const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h);
      const sc2 = Math.min(c.w / 7, c.h / 5.4), ox = c.w / 2, oy = c.h / 2, P = (x, y) => [ox + x * sc2, oy - y * sc2];
      const ab = mode === 'fwd' ? abAt(t) : abar[Math.max(0, tc)];
      g.strokeStyle = C.border; g.lineWidth = 1; g.beginPath(); g.moveTo(0, oy); g.lineTo(c.w, oy); g.moveTo(ox, 0); g.lineTo(ox, c.h); g.stroke();
      if ($(root, '#arr').checked && !(mode === 'rev' && tc <= 0)) {
        const gx = Math.floor(c.w / 52), gy = Math.floor(c.h / 52); g.lineWidth = 1.5;
        for (let i = 0; i < gx; i++) for (let j = 0; j < gy; j++) {
          const px = (i + 0.5) * (c.w / gx), py = (j + 0.5) * (c.h / gy), x = (px - ox) / sc2, y = (oy - py) / sc2;
          score(x, y, Math.max(ab, 1e-6), sc); const m = Math.hypot(sc[0], sc[1]); if (m < 1e-6) continue;
          const len = 24 * Math.tanh(m / 2.5), ux = sc[0] / m, uy = -sc[1] / m, col = alpha(C.purple, 0.35 + 0.5 * Math.tanh(m / 3));
          AI.arrow(g, px - (ux * len) / 2, py - (uy * len) / 2, px + (ux * len) / 2, py + (uy * len) / 2, col, 1.5);
        }
      }
      const sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
      for (let i = 0; i < N; i++) {
        let x, y, col;
        if (mode === 'fwd') { x = sa * X0x[i] + sn * Ex[i]; y = sa * X0y[i] + sn * Ey[i]; col = lab[i] ? C.orange : C.blue; }
        else { x = Yx[i]; y = Yy[i]; col = C.teal; }
        const p = P(x, y); g.beginPath(); g.arc(p[0], p[1], 3, 0, 7); g.fillStyle = col; g.globalAlpha = 0.9; g.fill();
      }
      g.globalAlpha = 1;
      const ti = Math.round(mode === 'rev' ? tc : t), bt = beta[Math.max(1, ti)];
      $(root, '#out').innerHTML = `mode: <b>${mode === 'fwd' ? 'forward (closed form)' : 'reverse (sampling)'}</b>
t = <b>${ti}</b> / ${T}   βₜ = ${AI.fmt(bt, 4)}
ᾱₜ = ${AI.fmt(ab, 4)}
signal √ᾱₜ = <b class="g">${AI.fmt(sa, 3)}</b>
noise √(1−ᾱₜ) = <b class="o">${AI.fmt(sn, 3)}</b>`;
      $(root, '#cap').textContent = mode === 'fwd'
        ? `Forward process: one closed-form formula jumps straight to any t. Early on the shape is still visible (signal weight ${AI.fmt(sa, 2)}); by t = ${T} it is pure N(0, I) noise. Purple arrows are the score ∇log pₜ(x): where the noised data is denser. Press "Generate" to run the process backwards.`
        : tc > 0 ? `Reverse step ${T - tc + 1}: ① the model (here the exact score of the noised data) says which way is "more data-like" → ② move x ← (x + βₜ·s)/√αₜ → ③ add a little fresh noise √βₜ·z. Repeat from t = ${T} down to 1. (An exact score of a finite training set reproduces the training points; a trained network generalises between them.)` : 'Done: pure noise has been turned into a sample on the data shape.';
    }, { fps: 30 });
  };
})();
