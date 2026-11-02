/* Robotics · Module 8 (Foundation models for robots) labs. Teaching simulations: every lab that stands in for a
 * learned model or a real robot is labelled "illustrative" in its subtitle or caption.
 * robo-episodes   <div data-lab="robo-episodes"></div>   teleoperated episodes grouped by task, storage formula
 * robo-tokens     <div data-lab="robo-tokens"></div>     continuous action rounded to bins, grouped into chunks
 * robo-vla        <div data-lab="robo-vla"></div>        image + instruction -> action tokens -> arm (stand-in model)
 * robo-diffusion  <div data-lab="robo-diffusion"></div>  annealed denoising of a two-mode action (illustrative)
 * robo-planner    <div data-lab="robo-planner"></div>    LLM proposal x affordance, normalised over feasible skills
 * robo-eval       <div data-lab="robo-eval"></div>       success rates with Wilson intervals, trial by trial
 */
(function () {
  const MONO = 'ui-monospace,Menlo,Consolas,monospace';
  const smooth = (t) => AI.smooth(t);

  /* ---------- control helpers ---------- */
  const slider = (ctl, label, min, max, step, val, fmt, onChange) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label} <span class="v"></span></span><input type="range" min="${min}" max="${max}" step="${step}" value="${val}">`;
    const inp = l.querySelector('input'), v = l.querySelector('.v');
    const show = () => (v.textContent = fmt ? fmt(+inp.value) : String(+inp.value));
    inp.addEventListener('input', () => { show(); if (onChange) onChange(+inp.value); });
    show();
    ctl.appendChild(l);
    return { get: () => +inp.value, set: (x) => { inp.value = x; show(); }, el: inp, show };
  };
  const speedSlider = (ctl) => slider(ctl, 'Speed', 0.25, 2, 0.05, 1, (v) => v.toFixed(2) + '×');
  const btnRow = (ctl, items) => {
    const d = document.createElement('div');
    d.className = 'row';
    d.innerHTML = items.map((it) => `<button class="btn small">${it[0]}</button>`).join('');
    ctl.appendChild(d);
    const bs = [...d.querySelectorAll('button')];
    bs.forEach((b, i) => b.addEventListener('click', () => items[i][1](b)));
    return bs;
  };
  const readoutEl = (ctl) => {
    const d = document.createElement('div');
    d.className = 'readout';
    ctl.appendChild(d);
    return d;
  };
  const textRow = (ctl, label, val, onChange) => {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span><input type="text">`;
    const inp = l.querySelector('input');
    inp.value = val;
    const err = document.createElement('div');
    err.className = 'err';
    ctl.appendChild(l);
    ctl.appendChild(err);
    inp.addEventListener('input', () => onChange(inp.value, err));
    return inp;
  };
  const mkLab = (root, title, sub, extraView) => {
    const { view, ctl } = AI.labShell(root, title, sub,
      `<canvas style="height:300px"></canvas><div class="caption"></div>${extraView || ''}`, '');
    return { view, ctl, cap: AI.$('.caption', view), c: AI.canvas(AI.$('canvas', view)) };
  };
  const f3 = (v) => AI.fmt(v, 3);
  const f4 = (v) => (Number.isFinite(v) ? v.toFixed(4) : '–');

  /* ---------- 1 · robo-episodes ---------- */
  AI.labs['robo-episodes'] = function (root) {
    const { ctl, cap, c } = mkLab(root, 'Record a robot dataset',
      'illustrative: a 2-D table stands in for a real arm; the storage numbers are exact arithmetic');
    const TASKS = [
      { name: 'put cup on plate', goal: [0.72, 0.30], col: 'blue' },
      { name: 'open drawer', goal: [0.20, 0.25], col: 'orange' },
      { name: 'stack two blocks', goal: [0.55, 0.72], col: 'green' },
    ];
    const START = [0.5, 0.9];
    const R = AI.rng(7);
    const EPS = [];
    for (let i = 0; i < 60; i++) {
      const k = i % 3, T = TASKS[k];
      EPS.push({
        k,
        s: [START[0] + (R() - 0.5) * 0.12, START[1] + (R() - 0.5) * 0.08],
        g: [T.goal[0] + (R() - 0.5) * 0.08, T.goal[1] + (R() - 0.5) * 0.08],
        bend: (R() - 0.5) * 0.14,
      });
    }
    const pathAt = (e, p) => {
      const q = smooth(p);
      const x = e.s[0] + (e.g[0] - e.s[0]) * q, y = e.s[1] + (e.g[1] - e.s[1]) * q;
      const dx = e.g[0] - e.s[0], dy = e.g[1] - e.s[1], L = Math.hypot(dx, dy) || 1;
      const b = e.bend * Math.sin(Math.PI * p);
      return [x - (dy / L) * b, y + (dx / L) * b];
    };
    let nEp = 30, secs = 20, hz = 10, cams = 2, px = 224;
    let cur = 0, p = 0, playing = true, dirty = true;
    const shown = () => Math.min(nEp, EPS.length);
    const reset = () => { cur = 0; p = 0; playing = true; dirty = true; playBtn.textContent = 'Pause'; };

    slider(ctl, 'Episodes in the dataset', 5, 200, 5, 30, null, (v) => { nEp = v; reset(); });
    slider(ctl, 'Episode length (s)', 5, 60, 1, 20, (v) => v + ' s', (v) => { secs = v; dirty = true; });
    slider(ctl, 'Recording rate (Hz)', 2, 50, 1, 10, (v) => v + ' Hz', (v) => { hz = v; dirty = true; });
    slider(ctl, 'Cameras', 1, 3, 1, 2, null, (v) => { cams = v; dirty = true; });
    slider(ctl, 'Image side (px)', 64, 448, 16, 224, (v) => v + ' px', (v) => { px = v; dirty = true; });
    const spd = speedSlider(ctl);
    const [playBtn, stepBtn, resetBtn] = btnRow(ctl, [
      ['Pause', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; }],
      ['Step: next episode', () => { playing = false; playBtn.textContent = 'Play'; if (cur < shown()) { cur++; p = 0; dirty = true; } }],
      ['Reset', () => reset()],
    ]);
    const ro = readoutEl(ctl);

    AI.loop(root, (dt) => {
      if (playing && cur < shown()) {
        p += (dt * spd.get()) / 3;
        if (p >= 1) { p = 0; cur++; }
        dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const draw = () => {
      const C = AI.col(), ctx = c.ctx, W = c.w, H = c.h;
      ctx.clearRect(0, 0, W, H);
      const s = Math.min(H - 24, W * 0.58), ox = 14, oy = 12;
      const P = (u, v) => [ox + u * s, oy + (1 - v) * s];
      ctx.fillStyle = C.bg2; ctx.fillRect(ox, oy, s, s);
      ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, s, s);
      TASKS.forEach((T) => {
        const [gx, gy] = P(T.goal[0], T.goal[1]);
        ctx.beginPath(); ctx.arc(gx, gy, 9, 0, Math.PI * 2); ctx.fillStyle = C[T.col]; ctx.globalAlpha = 0.35; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = C[T.col]; ctx.stroke();
      });
      const [sx, sy] = P(START[0], START[1]);
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fillStyle = C.fg; ctx.fill();
      const drawTrail = (e, upto, alpha) => {
        ctx.strokeStyle = C[TASKS[e.k].col]; ctx.lineWidth = 2; ctx.globalAlpha = alpha;
        ctx.beginPath();
        for (let i = 0; i <= 40; i++) {
          const q = (upto * i) / 40, [u, v] = pathAt(e, q), [x, y] = P(u, v);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      };
      const n = shown();
      for (let i = 0; i < Math.min(cur, n); i++) drawTrail(EPS[i], 1, 0.45);
      if (cur < n) {
        drawTrail(EPS[cur], p, 1);
        const [u, v] = pathAt(EPS[cur], p), [x, y] = P(u, v);
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = C.fg; ctx.fill();
      }
      // per-task counts
      const x0 = ox + s + 24, bw = Math.max(40, W - x0 - 14);
      ctx.font = `12px ${MONO}`; ctx.textAlign = 'left';
      const done = Math.min(cur, n);
      TASKS.forEach((T, k) => {
        const cnt = EPS.slice(0, done).filter((e) => e.k === k).length;
        const y = oy + 30 + k * 58;
        ctx.fillStyle = C.muted; ctx.fillText(T.name, x0, y);
        ctx.fillStyle = C.border; ctx.fillRect(x0, y + 8, bw, 14);
        ctx.fillStyle = C[T.col]; ctx.fillRect(x0, y + 8, bw * Math.min(1, cnt / Math.max(1, Math.ceil(n / 3))), 14);
        ctx.fillStyle = C.fg; ctx.fillText(cnt + ' recorded', x0, y + 38);
      });
      ctx.fillStyle = C.muted;
      ctx.fillText(cur < n ? `recording episode ${cur + 1} of ${n}` : `all ${n} shown episodes recorded`, x0, oy + 30 + 3 * 58 - 8);

      const frames = nEp * secs * hz;
      const perFrame = cams * px * px * 3;
      const total = frames * perFrame;
      ro.innerHTML =
        `frames       = ${nEp} episodes × ${secs} s × ${hz} Hz = <span class="g">${frames.toLocaleString('en-US')}</span>\n` +
        `bytes/frame  = ${cams} cam × ${px}×${px}×3 = <span class="b">${perFrame.toLocaleString('en-US')} B</span>\n` +
        `total images = <span class="o">${f3(total / 1e9)} GB</span> (${f3(total / 2 ** 30)} GiB), before compression\n` +
        `recorded now = ${Math.min(cur, n)} of ${nEp} episodes; the table shows the first ${n}`;
      cap.textContent = cur < n
        ? `Episode ${cur + 1}: task "${TASKS[EPS[cur].k].name}". The trail is the end-effector path, drawn slowly; each dot is one recorded step.`
        : 'Every shown episode is recorded. Change the sliders to see how storage scales.';
    };
  };

  /* ---------- 2 · robo-tokens ---------- */
  AI.labs['robo-tokens'] = function (root) {
    const { view, ctl, cap, c } = mkLab(root, 'Actions as tokens and chunks',
      'one action component, sampled at 10 Hz and rounded to bins; illustrative curve, real arithmetic');
    const RATE = 10, N = 41, LO = -1, HI = 1;
    let B = 16, K = 4, fn = (s) => 0.8 * Math.sin(2 * s.t), shown = 0, playing = true, dirty = true;
    let xs = [];
    const tokOf = (x) => Math.min(B - 1, Math.max(0, Math.floor((x - LO) / ((HI - LO) / B))));
    const decOf = (k) => LO + (k + 0.5) * ((HI - LO) / B);
    const sample = () => {
      xs = [];
      for (let i = 0; i < N; i++) {
        let v; try { v = fn({ t: i / RATE }); } catch (e) { v = 0; }
        xs.push(AI.clamp(Number.isFinite(v) ? v : 0, LO, HI));
      }
    };

    const inp = textRow(ctl, 'Curve f(t), t in seconds (0 to 4)', '0.8*sin(2*t)', (val, err) => {
      try {
        const fnNew = AI.compile(AI.parse(val), ['t']);
        fn = fnNew; sample(); shown = 0; playing = true; dirty = true; err.textContent = '';
        inp.classList.remove('bad');
      } catch (e) { inp.classList.add('bad'); err.textContent = e.message; }
    });
    sample();
    const sB = slider(ctl, 'Bins B', 2, 256, 1, 16, null, (v) => { B = v; dirty = true; });
    const sK = slider(ctl, 'Chunk length K (actions per call)', 1, 8, 1, 4, null, (v) => { K = v; dirty = true; });
    const spd = speedSlider(ctl);
    const tick = AI.ticker();
    const [playBtn] = btnRow(ctl, [
      ['Pause', (b) => { playing = !playing; b.textContent = playing ? 'Pause' : 'Play'; }],
      ['Step: one sample', (b) => { playing = false; b.textContent = 'Play'; if (shown < N) { shown++; dirty = true; } }],
      ['Reset', () => { shown = 0; playing = true; dirty = true; }],
    ]);
    const ro = readoutEl(ctl);

    AI.loop(root, (dt) => {
      if (playing && shown < N) { shown = Math.min(N, shown + tick(dt, 2.5 * spd.get())); dirty = true; }
      if (shown >= N && playing) playing = false;
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const draw = () => {
      const C = AI.col(), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      const v = AI.view(c, 0, 4.1, -1.25, 1.25, { l: 44, r: 12, t: 12, b: 34 });
      AI.axes(ctx, v);
      // chunk shading: one band per model call (K samples each)
      const calls = Math.ceil(N / K);
      ctx.font = '11px ' + MONO;
      for (let j = 0; j < calls; j++) {
        const i0 = j * K, i1 = Math.min(N, (j + 1) * K) - 1;
        if (shown <= i0) break;
        const xa = v.X(i0 / RATE), xb = v.X(i1 / RATE) + v.pw / (N - 1) / 2;
        ctx.fillStyle = j % 2 ? C.bg3 : C.bg2;
        ctx.globalAlpha = 0.6;
        ctx.fillRect(xa, v.pad.t, xb - xa, v.ph);
        ctx.globalAlpha = 1;
        ctx.fillStyle = C.muted;
        ctx.fillText('call ' + (j + 1), xa + 3, v.pad.t + 12);
      }
      // continuous curve (faint)
      ctx.strokeStyle = C.muted; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const t = (4 * i) / 200;
        let y; try { y = AI.clamp(fn({ t }), LO, HI); } catch (e) { y = 0; }
        const X = v.X(t), Y = v.Y(y);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
      // decoded staircase and samples
      const w = (HI - LO) / B;
      const tokW = v.pw / (N - 1);
      for (let i = 0; i < shown; i++) {
        const t = i / RATE, x = xs[i], k = tokOf(x), xh = decOf(k);
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(v.X(t) - tokW / 2, v.Y(xh)); ctx.lineTo(v.X(t) + tokW / 2, v.Y(xh)); ctx.stroke();
        ctx.beginPath(); ctx.arc(v.X(t), v.Y(x), 3.5, 0, Math.PI * 2); ctx.fillStyle = C.orange; ctx.fill();
        if (tokW >= 16 || i % 2 === 0) {
          ctx.fillStyle = C.muted; ctx.font = '9px ' + MONO; ctx.textAlign = 'center';
          ctx.fillText(String(k), v.X(t), v.pad.t + v.ph - 4);
        }
      }
      ctx.textAlign = 'left';
      // readout
      let maxErr = 0;
      for (let i = 0; i < N; i++) maxErr = Math.max(maxErr, Math.abs(decOf(tokOf(xs[i])) - xs[i]));
      const calls2 = RATE / K, dur = K / RATE;
      ro.innerHTML =
        `bins B = <span class="b">${B}</span>, width w = (1 − (−1))/B = <span class="b">${f4(w)}</span> (action units)\n` +
        `largest error over all samples = <span class="o">${f4(maxErr)}</span>  (bound w/2 = ${f4(w / 2)})\n` +
        `chunk K = ${K}: calls per second = 10 Hz / ${K} = <span class="g">${f3(calls2)} /s</span>\n` +
        `each chunk covers ${K} × 0.1 s = <span class="g">${f3(dur)} s</span> of motion\n` +
        `samples tokenised: ${shown} of ${N}; calls made: ${Math.ceil(shown / K)}`;
      const i = Math.max(0, shown - 1);
      cap.textContent = shown > 0
        ? `Sample ${i} at t = ${f3(i / RATE)} s: value ${f4(xs[i])} → bin ${tokOf(xs[i])} → decoded ${f4(decOf(tokOf(xs[i])))}. Chunk shading shows which samples one model call would produce.`
        : 'Press play: each sample is rounded to a bin, and its index is the token.';
    };
  };

  /* ---------- 3 · robo-vla ---------- */
  AI.labs['robo-vla'] = function (root) {
    const { ctl, cap, c } = mkLab(root, 'Image and instruction to action tokens',
      'illustrative: a hand-written stand-in for a trained VLA, to show the data flow; not a real model');
    const ORIG = { red: [0.28, 0.55], blue: [0.52, 0.38], green: [0.70, 0.62] };
    const NAME = { red: 'red cup', blue: 'blue block', green: 'green block' };
    const SINK = [0.84, 0.24], EE0 = [0.12, 0.86];
    const DIMS = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'grip'];
    const B = 256, LO = -1, HI = 1, W_ = (HI - LO) / B;
    const tokenOf = (x) => Math.min(B - 1, Math.max(0, Math.floor((AI.clamp(x, LO, HI) - LO) / W_)));
    const detok = (k) => LO + (k + 0.5) * W_;
    let instr = 'put the red cup in the sink', spd = null, playing = true, dirty = true;
    let OBJ, S, tokCur = null, decCur = null, actCur = null;
    const parse = () => {
      const s = instr.toLowerCase();
      const found = ['red', 'blue', 'green'].find((k) => s.includes(k)) || null;
      return { found, sink: s.includes('sink'), words: s.split(/[^a-z]+/).filter(Boolean) };
    };
    const clampA = (x) => AI.clamp(x, -1, 1);
    const plan = () => {
      const ee = S.ee, tg = S.tgt;
      const dx = tg[0] - ee[0], dy = tg[1] - ee[1];
      actCur = [clampA(dx), clampA(dy), 0, 0, 0, 0, S.carry ? 1 : -1];
      tokCur = actCur.map(tokenOf);
      decCur = tokCur.map(detok);
      S.from = ee.slice();
      S.to = [ee[0] + 0.5 * decCur[0], ee[1] + 0.5 * decCur[1]];
    };
    const go = (ph) => {
      S.ph = ph; S.t = 0; S.sub = 'plan';
      if (ph === 'call') { S.calls = 0; plan(); }
      dirty = true;
    };
    const resetS = () => {
      OBJ = JSON.parse(JSON.stringify(ORIG));
      S = { ph: 'img', t: 0, sub: 'plan', ee: EE0.slice(), from: null, to: null, held: null, carry: false, calls: 0, tgt: null, tgtName: null, sinkReq: false, grip: 0, msg: '' };
      tokCur = null; decCur = null; actCur = null; dirty = true;
    };
    resetS();

    const inp = textRow(ctl, 'Instruction (try the words red, blue, green, sink)', instr, (val, err) => {
      instr = val; resetS(); err.textContent = '';
      if (!parse().found) err.textContent = 'This stand-in knows only red, blue, green and sink.';
    });
    spd = speedSlider(ctl);
    const [playBtn] = btnRow(ctl, [
      ['Pause', (b) => { playing = !playing; b.textContent = playing ? 'Pause' : 'Play'; }],
      ['Restart', () => { resetS(); playing = true; playBtn.textContent = 'Pause'; }],
    ]);
    const ro = readoutEl(ctl);

    const step = (dt) => {
      S.t += dt * spd.get();
      switch (S.ph) {
        case 'img': if (S.t >= 2.2) go('txt'); break;
        case 'txt': if (S.t >= 2.2) go('model'); break;
        case 'model':
          if (S.t >= 2.2) {
            const P = parse();
            if (!P.found) { S.msg = 'no known object in the instruction; the stand-in policy stops'; go('done'); break; }
            S.tgtName = P.found; S.tgt = OBJ[P.found].slice(); S.sinkReq = P.sink; go('call');
          }
          break;
        case 'call':
          if (S.sub === 'plan') { if (S.t >= 1.2) { S.sub = 'move'; S.t = 0; } }
          else {
            const u = smooth(S.t / 1.2);
            S.ee = [AI.lerp(S.from[0], S.to[0], u), AI.lerp(S.from[1], S.to[1], u)];
            if (S.t >= 1.2) {
              S.ee = S.to.slice(); S.calls++;
              const dist = Math.hypot(S.tgt[0] - S.ee[0], S.tgt[1] - S.ee[1]);
              if (dist < 0.02) go(S.carry ? 'release' : 'grip');
              else if (S.calls >= 14) { S.msg = 'stopped after 14 calls without reaching the target'; go('done'); }
              else { plan(); S.sub = 'plan'; S.t = 0; }
            }
          }
          break;
        case 'grip':
          S.grip = smooth(S.t / 1.6);
          if (S.t >= 1.6) {
            if (S.sinkReq && !S.carry) { S.held = S.tgtName; S.carry = true; S.tgt = SINK.slice(); go('call'); }
            else { S.msg = `grasped the ${NAME[S.tgtName]}`; go('done'); }
          }
          break;
        case 'release':
          S.grip = 1 - smooth(S.t / 1.6);
          if (S.t >= 1.6) { OBJ[S.held] = SINK.slice(); S.held = null; S.carry = false; S.msg = `released the ${NAME[S.tgtName]} over the sink`; go('done'); }
          break;
        case 'done': if (S.t >= 2.5) resetS(); break;
      }
      dirty = true;
    };

    AI.loop(root, (dt) => {
      if (playing) step(dt);
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const draw = () => {
      const C = AI.col(), ctx = c.ctx, W = c.w, H = c.h;
      ctx.clearRect(0, 0, W, H);
      const s = Math.min(H - 16, W * 0.56), ox = 10, oy = 8;
      const P = (u, v) => [ox + u * s, oy + (1 - v) * s];
      ctx.fillStyle = C.bg2; ctx.fillRect(ox, oy, s, s);
      ctx.strokeStyle = C.border; ctx.strokeRect(ox + 0.5, oy + 0.5, s, s);
      // sink
      const [sx0, sy0] = P(SINK[0] - 0.1, SINK[1] + 0.07), [sx1, sy1] = P(SINK[0] + 0.1, SINK[1] - 0.07);
      ctx.fillStyle = C.bg3; ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
      ctx.strokeStyle = C.muted; ctx.strokeRect(sx0 + 0.5, sy0 + 0.5, sx1 - sx0, sy1 - sy0);
      ctx.fillStyle = C.muted; ctx.font = '11px ' + MONO; ctx.fillText('sink', sx0 + 4, sy0 - 4);
      // objects
      Object.keys(OBJ).forEach((k) => {
        if (S.held === k) return;
        const [x, y] = P(OBJ[k][0], OBJ[k][1]);
        ctx.fillStyle = C[k === 'red' ? 'red' : k === 'blue' ? 'blue' : 'green'];
        ctx.fillRect(x - 9, y - 9, 18, 18);
        ctx.fillStyle = C.muted; ctx.fillText(NAME[k], x - 14, y + 22);
      });
      // patch grid during stage 1
      if (S.ph === 'img') {
        ctx.strokeStyle = C.yellow; ctx.globalAlpha = 0.25 + 0.2 * Math.sin(S.t * 3);
        for (let i = 1; i < 14; i++) {
          ctx.beginPath(); ctx.moveTo(ox + (i / 14) * s, oy); ctx.lineTo(ox + (i / 14) * s, oy + s); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(ox, oy + (i / 14) * s); ctx.lineTo(ox + s, oy + (i / 14) * s); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      // end-effector and held object
      const [ex, ey] = P(S.ee[0], S.ee[1]);
      if (S.held) { ctx.fillStyle = C[S.held]; ctx.fillRect(ex - 8, ey - 8, 16, 16); }
      const gap = 10 - 6 * (S.grip || 0);
      ctx.strokeStyle = C.fg; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ex - gap, ey - 14); ctx.lineTo(ex - gap, ey + 2); ctx.moveTo(ex + gap, ey - 14); ctx.lineTo(ex + gap, ey + 2); ctx.stroke();
      ctx.lineWidth = 1;
      // instruction chips
      const x0 = ox + s + 22, cw = Math.max(160, W - x0 - 10);
      ctx.font = '12px ' + MONO; ctx.fillStyle = C.muted;
      ctx.fillText('instruction words → tokens', x0, oy + 14);
      const P2 = parse();
      let cx = x0, cy = oy + 24;
      P2.words.slice(0, 12).forEach((w) => {
        const tw = ctx.measureText(w).width + 12;
        if (cx + tw > x0 + cw) { cx = x0; cy += 22; }
        ctx.fillStyle = C.bg3; ctx.fillRect(cx, cy, tw, 18);
        ctx.fillStyle = C.fg; ctx.fillText(w, cx + 6, cy + 13);
        cx += tw + 5;
      });
      // action token table
      const ty = cy + 42;
      ctx.fillStyle = C.muted; ctx.fillText('action: 7 tokens (256 bins on [−1, 1])', x0, ty - 8);
      ctx.fillText('component      token   decoded', x0, ty + 8);
      DIMS.forEach((d, i) => {
        const y = ty + 26 + i * 20;
        ctx.fillStyle = C.fg;
        ctx.fillText(d, x0, y);
        if (tokCur) {
          ctx.fillStyle = C.orange; ctx.fillText(String(tokCur[i]), x0 + 110, y);
          ctx.fillStyle = C.blue; ctx.fillText(f4(decCur[i]), x0 + 170, y);
        } else { ctx.fillStyle = C.muted; ctx.fillText('–', x0 + 110, y); }
      });
      // readout
      const stage = { img: '1 image', txt: '2 text', model: '3 model', call: '4-5 tokens, move', grip: '6 grip', release: '6 release', done: 'done' }[S.ph];
      ro.innerHTML =
        `target   = <span class="b">${S.tgtName ? NAME[S.tgtName] : '–'}</span>${S.sinkReq ? ' then sink' : ''}\n` +
        `calls    = ${S.calls || 0} (each call: 7 tokens, then the arm moves 50% of the commanded step)\n` +
        `bin size = 2/256 = <span class="o">${f4(W_)}</span> per component\n` +
        `stage    = ${stage}`;
      const txt = {
        img: '① Image to patch tokens: the photo is cut into 14 × 14 = 196 patches, one token each (grid).',
        txt: `② Instruction to text tokens: "${instr}" becomes ${P2.words.length} words, each mapped to a token.`,
        model: `③ Model: one pass reads all image and text tokens and writes the next action. Stand-in rule: move toward the target.`,
        call: `④ Call ${S.calls + 1}: action tokens [${tokCur ? tokCur.slice(0, 2).join(', ') : ''}, …] decode to dx, dy = ${decCur ? f4(decCur[0]) + ', ' + f4(decCur[1]) : '–'}; ⑤ the arm moves.`,
        grip: `⑥ Close the gripper on the ${NAME[S.tgtName] || 'object'}.`,
        release: 'Open the gripper: the object is released over the sink.',
        done: S.msg ? `Done: ${S.msg}. The scene restarts after 2.5 s.` : 'Done.',
      }[S.ph];
      cap.textContent = txt;

    };
  };

  /* ---------- 4 · robo-diffusion ---------- */
  AI.labs['robo-diffusion'] = function (root) {
    const { ctl, cap, c } = mkLab(root, 'Denoising a two-mode action sequence',
      'illustrative: annealed Langevin sampling from a known two-mode target stands in for a learned denoiser');
    const H = 16, NSTEP = 60, SMIN = 0.05;
    const bump = Array.from({ length: H }, (_, i) => Math.sin((Math.PI * i) / (H - 1)));
    let A = 0.8, SMAX = 1.5, seed = 1, tally = { L: 0, R: 0 }, runs = 0;
    let x = [], k = 0, rnd = Math.random, snaps = [], hold = 0, phase = 'run', dirty = true, showMean = true, playing = true;
    const sig = (kk) => SMAX * Math.pow(SMIN / SMAX, kk / (NSTEP - 1));
    const newRun = () => {
      rnd = AI.rng(seed);
      x = Array.from({ length: H }, () => SMAX * AI.randn(rnd));
      k = 0; snaps = []; phase = 'run'; dirty = true;
    };
    const stepLang = () => {
      const s = sig(k), eta = 0.5 * s * s;
      const mp = bump.map((b) => A * b), mm = mp.map((v) => -v);
      let d1 = 0, d2 = 0;
      for (let i = 0; i < H; i++) { d1 += (x[i] - mp[i]) ** 2; d2 += (x[i] - mm[i]) ** 2; }
      const l1 = -d1 / (2 * s * s), l2 = -d2 / (2 * s * s), mx = Math.max(l1, l2);
      const e1 = Math.exp(l1 - mx), e2 = Math.exp(l2 - mx), w1 = e1 / (e1 + e2), w2 = 1 - w1;
      for (let i = 0; i < H; i++) {
        const score = (w1 * (mp[i] - x[i]) + w2 * (mm[i] - x[i])) / (s * s);
        x[i] += eta * score + Math.sqrt(2 * eta) * AI.randn(rnd);
      }
      k++;
      if (k % 5 === 0) { snaps.push(x.slice()); if (snaps.length > 10) snaps.shift(); }
    };
    slider(ctl, 'Mode amplitude A (m)', 0.1, 1.0, 0.05, 0.8, (v) => v.toFixed(2) + ' m', (v) => { A = v; newRun(); });
    slider(ctl, 'Start noise σ_max (m)', 0.5, 2.5, 0.1, 1.5, (v) => v.toFixed(1) + ' m', (v) => { SMAX = v; newRun(); });
    const spd = speedSlider(ctl);
    btnRow(ctl, [
      ['Pause', (b) => { playing = !playing; b.textContent = playing ? 'Pause' : 'Play'; }],
      ['Mean baseline: on', (b) => { showMean = !showMean; b.textContent = 'Mean baseline: ' + (showMean ? 'on' : 'off'); dirty = true; }],
      ['New sample', () => { seed++; newRun(); }],
    ]);
    const ro = readoutEl(ctl);
    const tick = AI.ticker();
    newRun();

    AI.loop(root, (dt) => {
      if (playing) {
        if (phase === 'run') {
          const n = tick(dt, 4 * spd.get());
          for (let j = 0; j < n && k < NSTEP; j++) stepLang();
          if (k >= NSTEP) {
            const mean = x.reduce((a, b) => a + b, 0) / H;
            const side = mean >= 0 ? 'R' : 'L';
            tally[side]++; runs++; phase = 'hold'; hold = 0;
          }
          dirty = true;
        } else {
          hold += dt * spd.get();
          if (hold >= 2.5) { seed++; newRun(); }
        }
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const draw = () => {
      const C = AI.col(), ctx = c.ctx, W = c.w, H_ = c.h;
      ctx.clearRect(0, 0, W, H_);
      const v = AI.view(c, 0, H - 1, -1.3, 1.3, { l: 48, r: 14, t: 12, b: 30 });
      AI.axes(ctx, v);
      const line = (arr, col, dash, lw, alpha) => {
        ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.globalAlpha = alpha;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        arr.forEach((y, i) => { const X = v.X(i), Y = v.Y(y); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
        ctx.stroke(); ctx.restore();
      };
      const mp = bump.map((b) => A * b);
      line(mp, C.green, [6, 5], 1.5, 0.9);
      line(mp.map((y) => -y), C.orange, [6, 5], 1.5, 0.9);
      if (showMean) line(new Array(H).fill(0), C.pink || C.purple, [2, 4], 1.5, 0.9);
      snaps.forEach((sn, j) => line(sn, C.muted, null, 1, 0.12 + 0.05 * j));
      line(x, C.blue, null, 2.5, 1);
      ctx.fillStyle = C.blue;
      x.forEach((y, i) => { ctx.beginPath(); ctx.arc(v.X(i), v.Y(y), 2.5, 0, Math.PI * 2); ctx.fill(); });
      ctx.fillStyle = C.muted; ctx.font = '11px ' + MONO;
      ctx.fillText('time step τ (0 to 15)', v.pad.l + v.pw / 2 - 60, v.pad.t + v.ph + 26);
      ctx.save(); ctx.translate(12, v.pad.t + v.ph / 2 + 30); ctx.rotate(-Math.PI / 2); ctx.fillText('offset (m)', 0, 0); ctx.restore();
      const s = sig(Math.min(k, NSTEP - 1));
      const mean = x.reduce((a, b) => a + b, 0) / H;
      ro.innerHTML =
        `iteration k = ${k} of ${NSTEP}; noise level σ_k = <span class="o">${f3(s)} m</span>\n` +
        `sample mean over τ = <span class="b">${f4(mean)} m</span>  →  ${k < NSTEP ? 'still moving' : mean >= 0 ? '<span class="g">committed to RIGHT</span>' : '<span class="g">committed to LEFT</span>'}\n` +
        `tally so far: left ${tally.L}, right ${tally.R} (runs ${runs})\n` +
        `mean baseline (least squares) = 0.0000 m at every τ: no demonstrator went there`;
      cap.textContent = k < NSTEP
        ? `Step ${k}: the sample is pulled toward the nearer demonstrated mode while the noise level falls. Dashed green is the right mode, dashed orange the left.`
        : 'The sample has settled on one of the two modes. A new run starts after 2.5 s. Both modes appear over many runs; their mean never does.';
    };
  };

  /* ---------- 5 · robo-planner ---------- */
  AI.labs['robo-planner'] = function (root) {
    const { ctl, cap, c } = mkLab(root, 'Language proposal times affordance',
      'illustrative kitchen: LLM scores are editable inputs, affordances follow the robot state');
    const SKILLS = [
      { id: 'pick', name: 'pick sponge', llm: 0.45, pre: (s) => s.at === 'table' && s.held === null && !s.wiped && s.rinsed === false },
      { id: 'wipe', name: 'wipe table', llm: 0.35, pre: (s) => s.at === 'table' && s.held === 'sponge' && !s.wiped },
      { id: 'sink', name: 'go to sink', llm: 0.10, pre: (s) => s.at === 'table' && !(s.wiped && s.rinsed) },
      { id: 'rinse', name: 'rinse sponge', llm: 0.08, pre: (s) => s.at === 'sink' && s.held === 'sponge' && !s.rinsed },
      { id: 'home', name: 'go to table', llm: 0.02, pre: (s) => s.at === 'sink' },
    ];
    let reach = 0.9;
    const freshState = () => ({ at: 'table', held: null, wiped: false, rinsed: false });
    let st = freshState(), phase = 'score', t = 0, pick = null, steps = [], playing = true, dirty = true, from = 0, to = 0, ro_x = 0, done = false;
    const affOf = (sk, s) => (sk.pre(s) ? (sk.id === 'pick' ? reach : 0.9) : 0);
    const scores = (s) => {
      const rows = SKILLS.map((sk) => { const a = affOf(sk, s); return { sk, llm: sk.llm, aff: a, prod: sk.llm * a }; });
      const Z = rows.reduce((z, r) => z + r.prod, 0);
      rows.forEach((r) => (r.norm = Z > 0 ? r.prod / Z : 0));
      return { rows, Z };
    };
    const goalReached = (s) => s.wiped && s.rinsed;
    const restart = () => { st = freshState(); phase = 'score'; t = 0; pick = null; steps = []; done = false; ro_x = st.at === 'sink' ? 1 : 0; dirty = true; };
    const sliders = {};
    SKILLS.forEach((sk) => {
      sliders[sk.id] = slider(ctl, `LLM score: ${sk.name}`, 0, 1, 0.01, sk.llm, (v) => v.toFixed(2), (v) => { sk.llm = v; restart(); });
    });
    slider(ctl, 'Affordance: reach the sponge', 0, 1, 0.01, reach, (v) => v.toFixed(2), (v) => { reach = v; restart(); });
    const spd = speedSlider(ctl);
    const [playBtn] = btnRow(ctl, [
      ['Pause', (b) => { playing = !playing; b.textContent = playing ? 'Pause' : 'Play'; }],
      ['Step: one plan', () => {
        playing = false; playBtn.textContent = 'Play';
        if (done) return;
        if (phase === 'exec') finishExec();
        else { executeChoice(); if (!done) finishExec(); }
        dirty = true;
      }],
      ['Restart', () => { restart(); playing = true; playBtn.textContent = 'Pause'; }],
    ]);
    const ro = readoutEl(ctl);

    const executeChoice = () => {
      const { rows } = scores(st);
      const feas = rows.filter((r) => r.prod > 0);
      if (!feas.length) { done = true; pick = null; return; }
      const best = feas.reduce((a, b) => (b.norm > a.norm ? b : a));
      pick = best;
      from = ro_x; to = best.sk.id === 'sink' ? 1 : best.sk.id === 'home' ? 0 : ro_x;
      steps.push(best.sk.name);
    };
    const applyEffect = () => {
      const id = pick.sk.id;
      if (id === 'pick') st.held = 'sponge';
      if (id === 'wipe') st.wiped = true;
      if (id === 'sink') st.at = 'sink';
      if (id === 'rinse') st.rinsed = true;
      if (id === 'home') st.at = 'table';
    };
    const finishExec = () => {
      applyEffect(); ro_x = to;
      if (goalReached(st)) done = true;
      phase = 'score'; t = 0;
    };

    AI.loop(root, (dt) => {
      if (playing && !done) {
        t += dt * spd.get();
        if (phase === 'score' && t >= 1.2) { executeChoice(); if (!done) { phase = 'exec'; t = 0; } }
        else if (phase === 'exec') {
          ro_x = AI.lerp(from, to, smooth(Math.min(1, t / 1.8)));
          if (t >= 1.8) finishExec();
        }
        dirty = true;
      }
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const draw = () => {
      const C = AI.col(), ctx = c.ctx, W = c.w, H = c.h;
      ctx.clearRect(0, 0, W, H);
      const s = scores(st);
      const kw = Math.min(W * 0.42, 340), ox = 12, oy = 12, kh = H - 24;
      ctx.fillStyle = C.bg2; ctx.fillRect(ox, oy, kw, kh);
      ctx.strokeStyle = C.border; ctx.strokeRect(ox + 0.5, oy + 0.5, kw, kh);
      const xT = ox + 0.25 * kw, xS = ox + 0.8 * kw, yC = oy + kh * 0.55;
      ctx.fillStyle = C.bg3; ctx.fillRect(xT - 46, yC - 34, 92, 68); ctx.fillRect(xS - 46, yC - 34, 92, 68);
      ctx.fillStyle = C.muted; ctx.font = '12px ' + MONO; ctx.textAlign = 'center';
      ctx.fillText('table', xT, yC - 40); ctx.fillText('sink', xS, yC - 40);
      const rx = AI.lerp(xT, xS, ro_x);
      if (st.held === null && !st.wiped) { ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(xT - 22, yC + 18, 7, 0, Math.PI * 2); ctx.fill(); }
      if (st.wiped === false && st.held === 'sponge') { /* sponge is in the robot's hand */ }
      if (st.wiped) { ctx.fillStyle = C.teal || C.blue; ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.ellipse(xT + 14, yC + 26, 30, 10, 0, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1; }
      else { ctx.fillStyle = C.teal || C.blue; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.ellipse(xT + 14, yC + 26, 18, 7, 0, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1; }
      ctx.fillStyle = C.fg; ctx.beginPath(); ctx.arc(rx, yC - 52, 11, 0, Math.PI * 2); ctx.fill();
      if (st.held === 'sponge') { ctx.fillStyle = C.green; ctx.fillRect(rx + 10, yC - 54, 12, 8); }
      ctx.textAlign = 'left';
      // bars
      const bx = ox + kw + 18, bw = Math.max(80, W - bx - 14);
      ctx.font = '12px ' + MONO;
      s.rows.forEach((r, i) => {
        const y = oy + 22 + i * 52;
        const feas = r.aff > 0;
        ctx.fillStyle = feas ? C.fg : C.muted;
        ctx.fillText(r.sk.name, bx, y);
        const bars = [['LLM', r.llm, C.blue], ['aff', r.aff, C.green], ['norm', r.norm, C.orange]];
        bars.forEach(([lab, val, col], j) => {
          const yy = y + 8 + j * 12;
          ctx.fillStyle = C.border; ctx.fillRect(bx + 44, yy, bw - 110, 8);
          ctx.fillStyle = col; ctx.fillRect(bx + 44, yy, (bw - 110) * AI.clamp(val, 0, 1), 8);
          ctx.fillStyle = C.muted; ctx.fillText(lab, bx, yy + 8);
          ctx.fillStyle = C.fg; ctx.fillText(val.toFixed(3), bx + bw - 58, yy + 8);
        });
      });
      const names = steps.length ? steps.map((n, i) => `${i + 1}. ${n}`).join('  ') : 'none yet';
      ro.innerHTML =
        `state: at ${st.at}, holding ${st.held || 'nothing'}, wiped ${st.wiped}, rinsed ${st.rinsed}\n` +
        `feasible mass Z = Σ LLM × aff = <span class="b">${f3(s.Z)}</span>\n` +
        `executed: ${names}\n` +
        (done ? `<span class="g">${goalReached(st) ? 'goal reached: table wiped and sponge rinsed' : 'stopped: no feasible skill (defined fallback)'}</span>` : 'planning…');
      const bestNow = s.rows.filter((r) => r.prod > 0).sort((a, b) => b.norm - a.norm)[0];
      cap.textContent = done
        ? (goalReached(st) ? 'The plan finished. Every step was chosen from the feasible skills.' : 'No skill is feasible from this state, so the planner stops and asks for help.')
        : phase === 'score'
          ? `Scoring: each feasible skill gets LLM × affordance, normalised. Best now: "${bestNow ? bestNow.sk.name : '–'}" with share ${bestNow ? f3(bestNow.norm) : '–'}.`
          : `Executing "${pick ? pick.sk.name : ''}". After it finishes, the state changes and the scores are recomputed.`;
    };
  };

  /* ---------- 6 · robo-eval ---------- */
  AI.labs['robo-eval'] = function (root) {
    const { ctl, cap, c } = mkLab(root, 'Success rates with uncertainty',
      'illustrative: trials are spread evenly for display, the real order of outcomes is not known');
    const Z = 1.96;
    const wilson = (k, n) => {
      const p = k / n, d = 1 + (Z * Z) / n;
      const centre = (p + (Z * Z) / (2 * n)) / d;
      const half = (Z * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n))) / d;
      return [centre - half, centre + half];
    };
    const wald = (k, n) => { const p = k / n, h = Z * Math.sqrt((p * (1 - p)) / n); return [p - h, p + h]; };
    let P = { A: { k: 37, n: 50 }, B: { k: 30, n: 50 } };
    let tt = 0, playing = true, dirty = true;
    const outcome = (pol, i) => { const { k, n } = pol; return Math.floor(((i + 1) * k) / n) > Math.floor((i * k) / n); };
    const succSoFar = (pol, m) => Math.floor((m * pol.k) / pol.n);
    const rebuild = () => { tt = 0; playing = true; dirty = true; };
    const kA = slider(ctl, 'Policy A: successes', 0, 50, 1, 37, null, (v) => { P.A.k = v; rebuild(); });
    slider(ctl, 'Policy A: trials', 1, 200, 1, 50, null, (v) => {
      P.A.n = v; P.A.k = Math.min(P.A.k, v); kA.el.max = v; kA.set(P.A.k); rebuild();
    });
    const kB = slider(ctl, 'Policy B: successes', 0, 50, 1, 30, null, (v) => { P.B.k = v; rebuild(); });
    slider(ctl, 'Policy B: trials', 1, 200, 1, 50, null, (v) => {
      P.B.n = v; P.B.k = Math.min(P.B.k, v); kB.el.max = v; kB.set(P.B.k); rebuild();
    });
    kA.el.max = 50; kB.el.max = 50;
    const spd = speedSlider(ctl);
    const [playBtn] = btnRow(ctl, [
      ['Pause', (b) => { playing = !playing; b.textContent = playing ? 'Pause' : 'Play'; }],
      ['Reset', () => { rebuild(); playBtn.textContent = 'Pause'; }],
    ]);
    const ro = readoutEl(ctl);
    const tick = AI.ticker();

    AI.loop(root, (dt) => {
      const total = Math.max(P.A.n, P.B.n);
      if (playing && tt < total) { tt = Math.min(total, tt + tick(dt, 3 * spd.get())); dirty = true; }
      if (tt >= total && playing) playing = false;
      if (dirty) { draw(); dirty = false; }
    }, { fps: 30 });
    c.onresize = () => { dirty = true; };

    const row = (ctx, C, y, pol, label, col, shownN) => {
      const W = c.w, x0 = 14, w = W - 28;
      ctx.fillStyle = C.muted; ctx.font = '12px ' + MONO;
      ctx.fillText(`${label}: ${pol.k} successes in ${pol.n} trials`, x0, y - 6);
      const cols = Math.min(pol.n, 50), cell = Math.min(14, w / cols);
      for (let i = 0; i < pol.n; i++) {
        const cx = x0 + (i % cols) * cell, cy = y + Math.floor(i / cols) * (cell + 2);
        ctx.fillStyle = i < shownN ? (outcome(pol, i) ? C.green : C.red) : C.border;
        ctx.fillRect(cx, cy, cell - 2, cell - 2);
      }
      const rows = Math.ceil(pol.n / cols);
      const yb = y + rows * (cell + 2) + 22;
      // interval axis 0..1
      const ax = (p) => x0 + p * w;
      ctx.strokeStyle = C.border; ctx.beginPath(); ctx.moveTo(x0, yb); ctx.lineTo(x0 + w, yb); ctx.stroke();
      const m = Math.max(1, shownN);
      if (shownN > 0) {
        const [lo, hi] = wilson(succSoFar(pol, shownN), shownN);
        ctx.fillStyle = col; ctx.globalAlpha = 0.25; ctx.fillRect(ax(lo), yb - 6, ax(hi) - ax(lo), 12); ctx.globalAlpha = 1;
      }
      if (shownN >= pol.n) {
        const [lo, hi] = wilson(pol.k, pol.n);
        ctx.fillStyle = col; ctx.fillRect(ax(lo), yb - 4, ax(hi) - ax(lo), 8);
        ctx.fillStyle = C.fg; ctx.fillRect(ax(pol.k / pol.n) - 1, yb - 10, 2, 20);
      }
      ctx.fillStyle = C.muted; ctx.fillText('0', x0, yb + 16); ctx.fillText('1', x0 + w - 6, yb + 16);
      return yb;
    };
    const draw = () => {
      const C = AI.col(), ctx = c.ctx, W = c.w, H = c.h;
      ctx.clearRect(0, 0, W, H);
      const shown = (n) => Math.min(n, tt);
      const yA = row(ctx, C, 20, P.A, 'A', C.blue, shown(P.A.n));
      const yB = row(ctx, C, yA + 56, P.B, 'B', C.orange, shown(P.B.n));
      const [loA, hiA] = wilson(P.A.k, P.A.n), [loB, hiB] = wilson(P.B.k, P.B.n);
      const [wlA, whA] = wald(P.A.k, P.A.n);
      const overlap = loA <= hiB && loB <= hiA;
      const done = tt >= Math.max(P.A.n, P.B.n);
      ro.innerHTML =
        `A: rate ${f3(P.A.k / P.A.n)}  Wilson 95% [${f4(loA)}, ${f4(hiA)}]  (Wald [${f4(wlA)}, ${f4(whA)}])\n` +
        `B: rate ${f3(P.B.k / P.B.n)}  Wilson 95% [${f4(loB)}, ${f4(hiB)}]\n` +
        (done
          ? (overlap ? `<span class="o">intervals overlap: not yet separated; more trials needed</span>` : `<span class="g">intervals do not overlap (a rough check, not a formal test)</span>`)
          : `trials shown: ${Math.floor(tt)} of ${Math.max(P.A.n, P.B.n)}; running A ${succSoFar(P.A, shown(P.A.n))}/${shown(P.A.n)}`);
      cap.textContent = done
        ? 'All trials shown. Bars are the Wilson intervals; the tick is the observed rate. Overlap is not proof of equality.'
        : 'Each cell is one trial: green success, red failure. The shaded band is the Wilson interval after the trials so far.';
    };
  };
})();
