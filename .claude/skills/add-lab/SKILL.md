---
name: add-lab
description: Build a new interactive animated lab for the AI Hub (editable equation or parameters, slow captioned animation, no lag) in the right assets/labs-*.js file. Use when a lesson needs a visual that existing labs (fplot, calc, vec2, matrix, descent, …) do not provide.
---

# Add an interactive lab

Labs are functions registered as `AI.labs.name = function (root) {…}` and mounted with `<div data-lab="name"></div>` (config via data-attributes).
Put them in the labs file for the module (`assets/labs-<module>.js`); load it before `shell.js`. First check `assets/labs-core.js`,
`labs-math.js`, `labs-ml.js` — reuse before writing.

## Design principles (what makes it intuitive)
1. **One idea per lab.** The learner changes one thing and sees one effect.
2. **Editable.** Expose a formula (`AI.exprInput`), numbers (sliders/inputs) or draggable objects — not just a play button.
3. **Show the mechanism, not only the result.** Break processes into labelled phases with captions
   (e.g. "① predict → ② measure error → ③ gradient → ④ update"), trails of past steps, highlighted quantities, live numeric readout
   that matches the picture (`.readout`).
4. **Slow and smooth.** Defaults must be watchable: steps/second via `AI.ticker()`, ≥1 s per phase, easing toward targets
   (`x += (target-x)*(1-Math.exp(-dt*k))`), a speed slider, Pause and Step. Never advance state "per frame".
5. **Light on the CPU.** Skip redraw when a state key is unchanged; precompute heatmaps/images and throttle them; `{fps:30}` for heavy ones;
   avoid per-frame allocation; rely on `AI.loop` pausing off-screen.
6. **Theme-safe.** Colours from `AI.col()`; canvases via `AI.canvas(cv)` (handles DPR/resize).
7. **Fail softly.** Invalid expression → red message under the input, previous valid state keeps running.

## Skeleton
```js
AI.labs.myLab = function (root) {
  const { view, ctl } = AI.labShell(root, 'Title', 'one-line instruction',
    '<canvas style="height:320px"></canvas>',                // left: visual
    '<label>Formula <input type="text" id="e" value="x^2"></label><div class="err" id="er"></div>' + // right: controls
    '<label>Speed <input type="range" id="sp" min="1" max="20" value="4"></label><div class="readout" id="o"></div>');
  const q = (s) => AI.$(s, root), c = AI.canvas(AI.$('canvas', view)), tick = AI.ticker();
  let fn = null;
  const ei = AI.exprInput(q('#e'), q('#er'), { allowed: ['x'], onChange: (f) => { fn = f; /* reset state */ } }); ei.run();
  AI.loop(root, (dt) => {
    const n = tick(dt, +q('#sp').value); for (let i = 0; i < n; i++) { /* advance simulation one step */ }
    const C = AI.col(), g = c.ctx; g.clearRect(0, 0, c.w, c.h); /* draw with AI.view/AI.axes/AI.arrow */
    q('#o').innerHTML = '…';
  }, { fps: 40 });
};
```
Helpers: `AI.parse/compile/autodiff/exprInput`, `AI.view/axes/arrow`, `AI.fmt/clamp/lerp/smooth/mix/rgb`, `AI.stats/quantile`, `AI.rng/randn`.

## In the lesson
Place a `<div class="try"><b>Try this</b><ol>…</ol></div>` before the lab telling the learner exactly what to do and what to notice.
Then verify with skill `verify-site`.
