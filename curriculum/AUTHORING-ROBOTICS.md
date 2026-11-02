# Authoring addendum — Robotics

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/robotics.md`. Style references: `math-foundations.html` (deep lessons), `cv-architectures.html` (architecture blueprints),
`geo-stac.html` (a module with real JSON and labs), hub `robotics.html`.

## What makes Robotics lessons different
- **Picture the robot first.** A planar 2-link arm, a differential-drive base, a 2-D grid of free and blocked cells, a one-dimensional
  filter. Then the numbers. Every lesson says where it sits in the sense–think–act loop and the control rate it needs.
- **Hand-worked on a tiny example.** Two-link forward kinematics at two joint angles; one Kalman update with numbers; a 4-cell A* search;
  one PID step; a 3-state Markov decision process. Compute everything in code first (node or python) and quote exact numbers.
- **Units always.** Radians or degrees (say which), metres, seconds, N·m, Hz. Never leave a unit implicit.
- **Pipeline view.** Each lesson ends with where it sits in the six stages (sense, estimate, plan, control, learn, deploy & monitor), its
  loop rate, and what it stores.
- **Safety honesty.** Say when a method needs a safety layer. Quote standards by number only when sure (ISO 10218 industrial, ISO 13482
  personal care). Do not invent certification claims.
- **Real systems.** Name real robots, datasets and papers only when sure (e.g. RT-2 and OpenVLA for vision-language-action; Open X-Embodiment
  for a shared dataset; ROS 2 for middleware; PPO for policy optimisation). If unsure, leave the name out.
- **Simulation is labelled.** Any lab that stands in for real hardware is labelled "illustrative" in its caption.
- Link maths instead of re-teaching it: `math-linear-algebra.html`, `math-calculus.html#partials-gradient`, `math-probability.html`,
  `math-optimization.html#gradient-descent`, `math-neural.html#backprop`, `math-information.html#cross-entropy-kl`, `ml-neural.html`.
  Other robotics modules link as `robo-<name>.html#lesson-id`.

## Page conventions
- File `robo-<name>.html` (from the curriculum module headings); `<body data-topic="robotics" data-module="N">`; hero like other modules;
  end with a "Module complete" callout to the next module.
- Scripts (all `defer`, this order): KaTeX, `assets/topics.js`, `assets/lib.js`, `assets/labs-core.js` (only if you use fplot/calc/vec2),
  `assets/labs-pipeline.js` (if you use the pipeline lab), your own `assets/labs-robo-<name>.js`, then `assets/shell.js` **last**.
  The shell loads the curriculum data (`assets/curriculum-robotics.js`) itself; do not include it.
- Lesson ids must equal the ids in `curriculum/robotics.md`.
- Link only to pages that exist. Robotics pages: `robo-*.html`, `robotics.html`; maths pages `math-*.html`; ML `ml-*.html`; CV `cv-*.html`;
  GeoAI `geo-*.html`. Check with `ls` before you finish.

## Labs
- Write new labs in **your own file** `assets/labs-robo-<name>.js` as `AI.labs.myLab = function (root) {…}`. Use `AI.labShell`, `AI.canvas`,
  `AI.loop(root, fn, {fps})`, `AI.ticker`, `AI.exprInput`, `AI.view/axes/arrow`, `AI.col`, `AI.fmt`, `AI.rng`.
- Ideas (use what fits your module): a **planar arm** (drag the joint sliders, the end-effector and its trail move; two elbow solutions shown);
  a **frame tree** (transforms composed step by step); a **1-D Kalman** (measurement and prediction bars, gain slider); a **particle cloud**
  (resampling step by step); an **occupancy grid** (ray casting filling cells); **A\* and RRT** (frontier and tree growing slowly, path
  highlighted); a **PID step response** (P, I, D sliders, overshoot shown); a **mass-spring-damper** (impedance sweep); a **policy rollout**
  (a point robot following a learned or hand-written policy, reward bar filling); a **safety watchdog** (a timer that trips and a motor that stops).
- Animation rules as in AUTHORING.md: slow by default, dt-based, captions, speed slider, pause/step, redraw only when state changes,
  `{fps:30}` for heavy labs, redraw after canvas resize.
- Equations that a learner should edit go through `AI.exprInput`, with safe parsing.

## Files you may create / not touch
Only `robo-<name>.html` and `assets/labs-robo-<name>.js`. Do not edit shared files (`lib.js`, `labs-core.js`, `labs-pipeline.js`, `shell.js`,
`topics.js`, `style.css`, `robotics.html`, curriculum files, other modules). Report shared-file bugs in your final message. No git.
**Write the page file early and save often** (rate limits can interrupt agents).

## Verify
As in AUTHORING.md: `node --check` on your labs file; headless Chrome console check; screenshot review; recompute every worked example;
confirm every lesson id; confirm every link target exists; confirm any JSON or code snippet parses.
