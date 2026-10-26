/* Animated pipeline: data packets travel through stages; click a stage to pin its details. */
AI.labs.pipeline = function (root) {
  const S = window.PIPELINES[root.dataset.id].stages, N = S.length;
  root.classList.add('pipe');
  root.innerHTML = `
    <div class="pipe-tools">
      <button class="btn small" data-a="play">⏸ Pause</button>
      <label>Speed <input type="range" min="0.3" max="3" step="0.1" value="1"></label>
      <span>Click a stage to pin its details.</span>
    </div>
    <div class="pipe-scroll"><div class="pipe-track" style="--n:${N}">
      <canvas></canvas>
      <div class="pipe-stages">${S.map((s, i) => `
        <button class="stage" data-i="${i}" style="--c:${s.color}">
          <b>${i + 1}. ${s.name}</b>
          <ul>${s.items.map((t) => `<li>${t}</li>`).join('')}</ul>
          <code>→ ${s.out}</code>
        </button>`).join('')}</div>
    </div></div>
    <div class="pipe-detail"><div class="d-desc"></div><div class="d-comp"></div><div class="d-stor"></div></div>`;

  const track = root.querySelector('.pipe-track'), cv = root.querySelector('canvas');
  const c = AI.canvas(cv), stages = [...root.querySelectorAll('.stage')];
  const btn = root.querySelector('[data-a=play]'), speedEl = root.querySelector('input');
  let playing = true, speed = 1, pinned = -1, shown = -1, packets = [], spawn = 0, cx = [], activeKey = '';

  const measure = () => (cx = stages.map((s) => s.offsetLeft + s.offsetWidth / 2));
  c.onresize = measure;
  new ResizeObserver(measure).observe(track);

  const show = (i) => {
    if (i === shown || i < 0) return;
    shown = i;
    root.querySelector('.d-desc').innerHTML = `<h4>${S[i].name}</h4>${S[i].desc}`;
    root.querySelector('.d-comp').innerHTML = `<h4>Computation</h4>${S[i].compute}`;
    root.querySelector('.d-stor').innerHTML = `<h4>Storage</h4>${S[i].storage}`;
  };
  show(0);

  btn.onclick = () => { playing = !playing; btn.textContent = playing ? '⏸ Pause' : '▶ Play'; };
  speedEl.oninput = () => (speed = +speedEl.value);
  stages.forEach((el) => (el.onclick = () => {
    const i = +el.dataset.i;
    pinned = pinned === i ? -1 : i;
    stages.forEach((s, j) => s.classList.toggle('pinned', j === pinned));
    shown = -1;
    show(pinned >= 0 ? pinned : i);
  }));

  AI.loop(root, (dt) => {
    if (playing) {
      spawn -= dt * speed;
      if (spawn <= 0) { packets.push({ t: 0 }); spawn = 1.7; }
      packets.forEach((p) => (p.t += dt * speed * 0.6));
      packets = packets.filter((p) => p.t < N + 0.2);
    }
    const C = AI.col(), g = c.ctx, y = 42;
    g.clearRect(0, 0, c.w, c.h);
    if (!cx.length) measure();
    // lane
    g.strokeStyle = C.border; g.lineWidth = 2;
    g.beginPath(); g.moveTo(cx[0] - 40, y); g.lineTo(cx[N - 1] + 60, y); g.stroke();
    cx.forEach((x, i) => {
      g.fillStyle = S[i].color; g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill();
      g.strokeStyle = S[i].color; g.globalAlpha = 0.35; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x, c.h); g.stroke(); g.globalAlpha = 1;
    });
    const act = new Set();
    for (const p of packets) {
      const i = Math.min(N - 1, Math.floor(p.t)), f = p.t - i, mv = f < 0.55 ? 0 : AI.smooth((f - 0.55) / 0.45);
      const last = p.t >= N - 1 + 0.55;
      const x0 = cx[i], x1 = i < N - 1 ? cx[i + 1] : cx[i] + 60;
      const x = AI.lerp(x0, x1, p.t >= N ? 1 : mv);
      const colA = S[i].color, colB = S[Math.min(N - 1, i + 1)].color;
      const col = AI.mix(colA, colB, mv);
      if (f < 0.55 && p.t < N) act.add(i);
      const alpha = p.t > N - 1 + 0.55 ? Math.max(0, 1 - (p.t - (N - 1 + 0.55)) / (0.65)) : 1;
      g.globalAlpha = last ? alpha : 1;
      g.shadowColor = col; g.shadowBlur = 12; g.fillStyle = col;
      const pulse = f < 0.55 ? 1 + 0.25 * Math.sin(f * 30) : 1, s = 11 * pulse;
      // shape morphs: dwelling at a stage squares the packet into a rounded tensor-like block
      g.beginPath(); g.roundRect(x - s / 2, y - s / 2, s, s, f < 0.55 ? 3 : s / 2); g.fill();
      g.shadowBlur = 0;
      g.fillStyle = C.muted; g.font = '11px ui-monospace,Menlo,Consolas,monospace'; g.textAlign = 'center';
      g.fillText(S[f < 0.55 ? i : Math.min(N - 1, i)].out, x, y - 14);
      g.globalAlpha = 1;
    }
    const key = [...act].join();
    if (key !== activeKey) {
      activeKey = key;
      stages.forEach((s, j) => s.classList.toggle('active', act.has(j)));
      if (pinned < 0 && act.size) show([...act].pop());
    }
  });
};
