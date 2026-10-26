# Authoring guide — deep lessons

The reference implementation is **`math-foundations.html`** (Module 0). Read it first: copy its page skeleton, its lesson
structure and its tone. The curriculum spec (what each lesson must cover) is **`curriculum/mathematics.md`**.

## The standard (non-negotiable)
Each lesson is a `<section class="lesson" id="LESSON-ID" data-lesson>` using the id from the curriculum table, containing:

1. `lesson-head` with level badge (`1.1`, `1.2`…), `<h2>`, and `<button class="btn small mark"></button>`.
2. **The question** (`<div class="question"><b>The question</b>…`) — a concrete problem that makes the idea necessary.
3. **Intuition** (`<h3 class="sec">Intuition</h3>`) — analogy / mental picture in plain words. *No formula before the idea.*
4. **Build it up** — step by step. At least one `<div class="worked"><b>Worked example</b>…` with **real numbers computed by hand**.
5. **The formal definition** — equation(s) in `<div class="eq">…</div>` plus a `<table class="sym">` explaining *every symbol*.
6. **Lab(s)** — preceded by `<div class="try"><b>Try this</b><ol>…</ol></div>` with concrete things to do and what they should notice.
7. `<div class="trap"><b>Common traps</b>…`
8. `<div class="aihook"><b>Where it shows up in AI</b>…` — concrete algorithms/models, not vague statements.
9. `<h3 class="sec">Check yourself</h3>` with 3 × `<details class="check"><summary>…</summary><div class="ans">…</div></details>`.
10. `<div class="takeaways"><h4>Takeaways</h4><ul>…</ul></div>`.

Depth: write as a patient teacher, not as notes. Each lesson should be ~700–1500 words of prose plus equations, examples and
labs. Build from the simplest case to full understanding; define terms on first use; give the "why", not only the "what".
Never skip a step that a beginner would stumble on. Use analogies, then make them precise.

**Mathematical correctness is mandatory.** State conditions (e.g. "if A is invertible"). Double-check every number in worked
examples and every "Check yourself" answer by computing it (use node/python in the shell). No hand-waving that is false.

## Page skeleton
Copy the `<head>`, header/aside/main/aside structure and footer from `math-foundations.html`. Set
`<body data-topic="math" data-module="N">`. Load scripts in this order (all `defer`): KaTeX, `assets/topics.js`,
`assets/curriculum-math.js`, `assets/lib.js`, `assets/labs-core.js`, `assets/labs-math.js` (if you use matrix/derivative/descent/
backprop/sampling), your own `assets/labs-<module>.js` (if any), then `assets/shell.js` **last**. The shell builds the header, sidebar
(module + lesson tree), outline, progress buttons and prev/next module navigation automatically.
Add a hero (see Module 0) and finish with a short "Module complete" callout pointing to the next module.

## Equations
Use KaTeX. Display: `<div class="eq">\frac{a}{b}</div>`. Inline: `<span class="tex">x^2</span>`. The element's *text* is the TeX:
write `&amp;` for `&` (matrices) and `&lt;`/`&gt;` for `<`/`>` inside TeX. Inside JS strings/JSON double the backslashes.

## Available CSS blocks
`.question .worked .aihook .try .trap .callout .callout.warn .takeaways details.check table.sym table.tbl pre>code .eq .tex
.chips .chip .grid .card .lab`. Do not edit `assets/style.css`; if you truly need a tiny extra style, use an inline `style=""`.

## Ready-made labs (mount with `<div data-lab="NAME" …></div>`)
Configured by attributes — see the header comment of `assets/labs-core.js`:
- `fplot` — function plotter with editable formulas and parameter sliders: `data-f="a*x+b; x^2" data-params="a=1:-3..3:0.1; b=0:-3..3:0.1" data-view="xmin,xmax,ymin,ymax" data-names="f;g" data-title data-roots data-diag data-readout="label = expr; …" data-height`.
- `calc` — sliders + live computed rows: `data-params`, `data-rows="label = expr; …"`, `data-note`.
- `machine`, `sigma`, `rn`, `unitcircle`, `vec2` (draggable vectors; `data-show="dot,proj,angle,norm,dist"`), `halfplane`, `highdim`, `decode`.
- From `assets/labs-math.js`: `matrix` (2×2 transform grid, determinant, eigenvectors), `derivative`, `descent` (SGD/momentum/Adam race on editable f(x,y)), `backprop` (computational graph forward/backward), `sampling` (editable sampler, averaging n draws, CLT), `pipeline`.
Expressions use the safe parser in `assets/lib.js` (`+ - * / ^`, implicit multiplication, `sin cos tan exp log sqrt abs tanh sigmoid relu atan max min`, constants `pi e`).

**Write new labs** whenever a lesson needs a visual that does not exist. Put them in **your own file** `assets/labs-<module>.js` as
`AI.labs.myLab = function (root) { … }` (use `AI.labShell(root, title, subtitle, viewHTML, controlsHTML)` for the standard layout).
Helpers in `assets/lib.js` (read the file): `AI.parse/compile/autodiff/exprInput`, `AI.canvas`, `AI.loop(el, fn, {fps})`, `AI.ticker`,
`AI.view/axes/arrow`, `AI.fmt/clamp/lerp/smooth/mix/rgb/col`, `AI.stats/quantile`, `AI.rng/randn`. See existing labs for patterns.

## Animation rules (learners complained animations were too fast and laggy)
- Everything animates in **real time units** (`dt`), never "per frame". Use `AI.ticker()` for step-based processes (rates in steps/second).
- **Slow by default**: a new idea should take seconds to unfold. Typical: 3–8 steps/s for optimisation, ≥1 s per phase for multi-stage
  processes, ≤ 0.5 rad/s rotations. Give a **speed slider** and a **pause/step** button.
- Make motion **meaningful**: show captions of what is happening ("① predict → ② measure error → ③ gradient → ④ update"), trails,
  labels, highlighted quantities, and smooth interpolation (ease toward targets) rather than jumps.
- Performance: only draw when needed (compare a state key and skip if unchanged), cap heavy labs with `{fps:30}`, keep per-frame work small,
  precompute images/heatmaps and throttle them. Never allocate large arrays per frame.
- Pause automatically when off-screen (`AI.loop` already does). Colours via `AI.col()` so light and dark themes both work.

## Do / don't
- **Create only your own files** (`math-<module>.html`, `assets/labs-<module>.js`). Do **not** edit shared files
  (`assets/lib.js`, `labs-core.js`, `labs-math.js`, `topics.js`, `shell.js`, `style.css`, other modules) — other authors work in parallel.
- Do not git add/commit/push. The maintainer does that.

## Verify before finishing
1. `node --check assets/labs-<module>.js`.
2. Load the page headless and check for console errors, from the repo root, e.g.:
   `"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --enable-logging=stderr --v=0 --virtual-time-budget=9000 --window-size=1500,6000 --screenshot="<scratchpad>/x.png" "file:///E:/On%20Going%20Projects/Knowledge%20Growth/AI-Hub/<page>.html" 2>&1 | grep CONSOLE`
   (no output = no errors) and look at the screenshot with the Read tool: labs must render, equations must typeset (no raw `\frac`), layout must not overflow.
3. Check every lesson id in your page matches `curriculum/mathematics.md` exactly and every lesson has all required parts.
4. Re-compute worked examples and answers numerically.
