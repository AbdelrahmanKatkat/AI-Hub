# Authoring addendum — Machine Learning

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/machine-learning.md`. Style/depth references: `math-foundations.html` (deep lessons) and `cv-architectures.html` / `cv-cnn.html`
(architecture-style lessons with shape tables), plus the hub `machine-learning.html`.

## What makes ML lessons different
- **Watch the model learn.** Labs show the *process*: a boundary bending as training proceeds, a tree splitting step by step (the next best split
  highlighted with its impurity gain), k-means assign/update phases, boosting adding one tree at a time and the residuals shrinking, k-NN neighbours
  lighting up, thresholds sweeping a score distribution. Captions say what is happening at each phase; speed slider, pause, step, reset.
- **Hand-worked on a tiny dataset.** Every algorithm gets a 4–8-row dataset computed by hand in the text (e.g. Gini of each candidate split;
  one k-means iteration; one boosting round with residuals; a confusion matrix → precision/recall) — compute in code first and quote exact numbers.
- **Pipeline view.** Algorithm lessons state: preprocessing needed, what is *stored* (parameters/trees/neighbours), training cost (big-O + practical),
  inference cost, memory, and **when to use / when not to** (strengths, weaknesses table).
- **Architecture:** model architectures (trees, ensembles, MLPs with shape/parameter tables — you can use `archviz`, e.g. `input 1 1 20 / fc 64 / relu / fc 1`)
  and, in the production module, **system architecture** diagrams (animated components with data/requests flowing through queues, caches, stores, monitors).
- **Honesty:** simulations that stand in for real training must be labelled "illustrative". Quote only numbers you computed.
- Link maths instead of re-teaching it: e.g. `math-calculus.html#partials-gradient`, `math-probability.html#conditional-bayes`,
  `math-statistics.html#estimation-mle`, `math-optimization.html#gradient-descent`, `math-linear-algebra.html#svd-pca`, `math-neural.html#backprop`,
  `math-information.html#cross-entropy-kl`. Other ML modules link as `ml-<name>.html#lesson-id` (names in the curriculum headings). Vision details: `cv-cnn.html`.

## Page conventions
- File `ml-<name>.html` (from the curriculum module headings); `<body data-topic="ml" data-module="N">`; hero like Module 0 of math; end with a "Module complete" callout.
- Scripts (all `defer`, this order): KaTeX, `assets/topics.js`, `assets/lib.js`, `assets/labs-core.js`, `assets/mllib.js`, `assets/labs-pipeline.js` (if used),
  `assets/labs-ml.js`, (`assets/arch.js` + `assets/labs-cv.js` only if you use `archviz`), your own `assets/labs-ml-<name>.js`, then `assets/shell.js` **last**.
  The shell loads the curriculum data and builds sidebar/outline/progress/prev-next itself — do **not** include curriculum scripts.
- Lesson ids must equal the ids in `curriculum/machine-learning.md`.

## Tooling you can use
- `assets/mllib.js` (read it): `ML.data(kind,n,seed)` (moons circles blobs xor spiral linear checker), `ML.reg(kind,n,seed)`, `ML.knn`, `ML.tree.fit / bestSplit / impurity`,
  `ML.kmeans` (step-wise), `ML.logistic` (step-wise), `ML.linreg`, `ML.pca2d`, and browser helpers `ML.boundary` (probability heat-map), `ML.points`, `ML.toPx`, `ML.COLORS`.
  Add your own algorithms (random forest, boosting, SVM, naive Bayes, …) in **your own** labs file — don't edit mllib.js.
- Existing labs (mount with `data-lab`): `preprocess`, `regression`, `nn` (neural playground), `threshold` (ROC/precision-recall), `pipeline` (`data-id="ml"`),
  from labs-core: `fplot`, `calc`, `vec2`, `halfplane`, `decode`; from labs-math (load `assets/labs-math.js`): `descent`, `matrix`, `derivative`, `backprop`, `sampling`;
  `archviz` (architecture visualiser; see `curriculum/AUTHORING-CV.md` for the layer DSL).
- Helpers in `assets/lib.js`: `AI.labShell, AI.canvas, AI.loop(root, fn, {fps}), AI.ticker, AI.exprInput, AI.view/axes/arrow, AI.fmt/clamp/lerp/smooth/mix, AI.col, AI.stats/quantile, AI.rng/randn`.
- Write new labs in `assets/labs-ml-<name>.js` as `AI.labs.myLab = function (root) {…}`; follow the animation rules in AUTHORING.md (slow, dt-based, captions, redraw only when needed,
  `{fps:30}` for heavy, redraw after canvas resize — the `AI.canvas` resize clears the canvas).

## Files you may create / not touch
Only `ml-<name>.html` and `assets/labs-ml-<name>.js`. Do not edit shared files (`lib.js, mllib.js, labs-ml.js, labs-core.js, labs-math.js, labs-cv.js, arch.js, shell.js, topics.js,
style.css`, curriculum files, other modules). Report shared-file bugs in your final message. No git. **Write the page file early and save often** (rate limits can interrupt agents).

## Verify
As in AUTHORING.md: `node --check`, headless Chrome console check, screenshot review; recompute worked examples; confirm every lesson id; links you add must point to real pages/anchors
(math-*/cv-*/ml-* files exist; for a not-yet-written ml-* module link to the module page without an anchor).
