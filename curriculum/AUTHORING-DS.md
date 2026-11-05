# Authoring addendum — Data Science

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/data-science.md`. Style references: `math-optimization.html` and `math-information.html` (deep lessons, exact markup),
`ml-foundations.html` (ML tone), hub `data-science.html`, and `assets/labs-information.js` (a small, clean lab file).

## What makes Data Science lessons different
- **Start from a decision.** Every lesson opens with a person who must decide something and a table that could inform it
  (a shop's weekly sales, a clinic's waiting times, an app's sign-up test). Name the decision, then the data.
- **Tiny printed tables.** Worked examples use a table of 5–10 rows that is *printed in the lesson*. Every mean, interval, p-value,
  join count and metric is computed in node/python first and quoted exactly. Show the arithmetic, not only the answer.
- **Uncertainty is always stated.** Give n, the interval or standard error, and the assumption it rests on. Say "evidence for" not "proves".
- **Samples arrive over time in the labs.** Prefer labs where rows, samples, users or days arrive one by one at a slow, steady pace, so
  estimates visibly settle and intervals shrink. Seeded random numbers (`AI.rng(seed)`) so a run can be repeated.
- **Simulations are labelled.** A lab that stands in for a real study is labelled *illustrative* in its caption.
- **Traps are the point.** Leakage, peeking, Simpson's paradox, survivorship bias, p-hacking and misleading axes each get a concrete,
  numeric example, not just a warning.
- **Pipeline view.** Each lesson says where it sits in the six stages of the data science loop (ask and collect, clean and prepare,
  explore, test and model, communicate, deploy and monitor).
- **Link maths and ML instead of re-teaching:** `math-probability.html`, `math-statistics.html`, `math-linear-algebra.html`,
  `math-optimization.html`, `math-information.html`, `ml-foundations.html`, `ml-linear.html`, `ml-evaluation.html` — check each target
  exists with `ls` and use only real lesson ids. Other DS modules link as `ds-<name>.html#lesson-id`.

## Page conventions (copy `math-information.html` exactly for structure)
- File `ds-<name>.html` named in the curriculum module heading; `<body data-topic="ds" data-module="N">`; hero like the other modules.
- Scripts (all `defer`, in this order): KaTeX, `assets/topics.js`, `assets/curriculum-ds.js`, `assets/lib.js`, `assets/labs-core.js`
  (only if you use fplot/calc/vec2), your own `assets/labs-ds-<name>.js`, then `assets/shell.js` **last**.
- Lesson markup: `<section class="lesson" id="LESSON-ID" data-lesson>`; head `<div class="lesson-head"><span class="level" style="color:var(--blue)">N.k</span><h2>Title</h2><button class="btn small mark"></button></div>`;
  then `<div class="prose">` containing, in order: `div.question`, `h3.sec` Intuition, `h3.sec` Build it up with `div.worked` blocks,
  `div.eq` (**raw LaTeX with no `\[ \]` delimiters**), `div.sym` (every symbol explained), the lab `<div data-lab="NAME"></div>`,
  `div.try` (ol, 3 steps), `div.trap`, `div.aihook`, `h3.sec` Check yourself with three `<details class="check"><summary>…</summary><div class="ans">…</div></details>`,
  and `div.takeaways`. Inline maths is `<span class="tex">…raw LaTeX…</span>`. Escape `<` as `&lt;`, `&` as `&amp;`.
- Lesson ids must equal the ids in `curriculum/data-science.md` exactly; no extra `section.lesson` elements. A short intro callout, if
  any, must not be a `section.lesson`.
- End with a "Module complete" callout linking to the next module page (check that it exists, or it is created by another author).
- Link only to pages that exist (`ls`). Colours only via `AI.col()` or CSS variables.

## Labs
- Write labs in **your own file** `assets/labs-ds-<name>.js`, defined as `AI.labs['lab-name'] = function (root) { … }` (lab names must be
  unique across the whole site: prefix them with `ds-`). Mount with `<div data-lab="ds-…"></div>`.
- Use `AI.labShell(root, title, sub, viewHTML, ctlHTML)`, `AI.canvas(canvasEl)` (set `o.onresize = draw`), `AI.loop(root, fn, {fps:30})`,
  `AI.ticker`, `AI.exprInput`, `AI.col()`, `AI.fmt`, `AI.rng`, `AI.randn`. Draw only when state changes.
- Animation rules (AUTHORING.md): dt-based, slow by default, speed slider, pause/step, reset, captions saying what is happening,
  redraw after resize. Controls the learner edits live (sample size, threshold, bin width, effect size, α, a formula via `AI.exprInput`).
- Ideas by module: **wrangling** (rows flowing through filter/join, a join multiplying rows, missing cells filling),
  **EDA** (histogram bins, box plot, Anscombe scatter, Simpson's paradox), **statistics** (sampling distribution forming, CLT with a skewed
  population, confidence-interval coverage over 100 samples, p-value, power curve, bootstrap resampling), **visualisation** (axis
  truncation, encoding accuracy), **experiments** (A/B test accruing users, peeking inflating false positives, multiple comparisons,
  confounder, difference-in-differences), **features** (encoding, scaling, a sliding window, leakage), **modelling** (k-fold rotation,
  learning curve, threshold and confusion matrix), **time series** (decomposition, exponential smoothing, rolling-origin backtest, fan chart),
  **production** (drift, fairness metrics, k-anonymity).

## Files you may create / not touch
Only `ds-<name>.html` and `assets/labs-ds-<name>.js`. Do not edit shared files (`lib.js`, `labs-core.js`, `labs-pipeline.js`, `shell.js`,
`topics.js`, `style.css`, `data-science.html`, curriculum files, other modules). Report shared-file bugs in your final message.
No git. Use a private scratch subfolder (named after your module) inside the scratchpad, never the shared scratchpad root.
**Write the page file early and save often** (rate limits can interrupt agents). Write long files with the Write tool, not shell heredocs.

## Verify
As in AUTHORING.md: `node --check` on your labs file; headless Chrome console check; screenshot review; recompute every worked example;
confirm every lesson id equals the spec; confirm every `data-lab` name is defined in your labs file; confirm every link target exists;
check there are no raw `\frac` or `\[` left (KaTeX renders `.eq` and `.tex` from raw LaTeX).
