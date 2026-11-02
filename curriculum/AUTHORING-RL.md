# Authoring addendum — Reinforcement Learning

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/reinforcement-learning.md`. Style references: `math-foundations.html` (deep lessons), `ml-neural.html` (value and policy
networks), `robo-learning.html` (a module with illustrative labs), hub `reinforcement-learning.html`.

## What makes RL lessons different
- **Show the agent acting.** Every lesson starts with an agent in an environment: a bandit's arms, a gridworld, a chain of states, a
  game or a robot. The reward is often delayed, so draw where it arrives and how the estimate moves back through time.
- **Hand-worked on a tiny problem.** Use a 2–4 state chain, a 3×3 gridworld, a 2-armed bandit or a 3-step episode. Compute every value,
  return, TD error, importance weight and gradient in code first and quote exact numbers.
- **Discounting and horizons always stated.** Give γ, λ, α, ε and the number of episodes or sweeps every time a number appears.
- **Convergence claims are exact or hedged.** Say "converges under stated conditions" and name the condition (tabular, sufficient
  exploration, step-size conditions). Do not say an algorithm "always works".
- **Deep RL is honest about instability.** Where a method is fragile (deadly triad, sensitivity to seeds), say so and show it in a lab
  where possible.
- **Simulations are labelled.** A lab that stands in for a large experiment is labelled "illustrative" in its caption.
- **Pipeline view.** Each algorithm lesson ends with where it sits in the six stages (interact, store, estimate, improve, evaluate,
  deploy & monitor), what it stores (table, network, buffer) and its cost per update.
- Link maths instead of re-teaching it: `math-probability.html` (expectation), `math-calculus.html#partials-gradient`,
  `math-optimization.html#gradient-descent`, `math-neural.html#backprop`, `math-information.html#cross-entropy-kl`. Other RL modules link
  as `rl-<name>.html#lesson-id`. Robot control and policies: `robo-learning.html`, `robo-control.html`.

## Page conventions
- File `rl-<name>.html` (from the curriculum module headings); `<body data-topic="rl" data-module="N">`; hero like the other module pages;
  end with a "Module complete" callout to the next module.
- Scripts (all `defer`, in this order): KaTeX, `assets/topics.js`, `assets/lib.js`, `assets/labs-core.js` (only if you use fplot/calc/vec2),
  `assets/labs-pipeline.js` (if you use the pipeline lab), your own `assets/labs-rl-<name>.js`, then `assets/shell.js` **last**.
  The shell loads the curriculum data (`assets/curriculum-rl.js`) itself; do not include it.
- Lesson ids must equal the ids in `curriculum/reinforcement-learning.md`.
- Link only to pages that exist. RL pages: `rl-*.html`, `reinforcement-learning.html`; maths `math-*.html`; ML `ml-*.html`; CV `cv-*.html`;
  GeoAI `geo-*.html`; Robotics `robo-*.html`. Check with `ls` before you finish.

## Labs
- Write new labs in **your own file** `assets/labs-rl-<name>.js` as `AI.labs.myLab = function (root) {…}`. Use `AI.labShell`, `AI.canvas`,
  `AI.loop(root, fn, {fps})`, `AI.ticker`, `AI.exprInput`, `AI.view/axes/arrow`, `AI.col`, `AI.fmt`, `AI.rng`.
- Ideas (use what fits your module): **bandit arena** (arms pulled one at a time, estimate bars converging, ε or UCB slider); **gridworld
  value table** (sweeps fill the table, greedy arrows update); **policy iteration** (evaluation and improvement alternating, arrows
  settling); **episode walker** (returns walked back to the start, visit counts filling); **TD error ripple** (δ moving back through a
  chain); **SARSA vs Q-learning on a cliff** (two paths side by side); **replay buffer** (sampling shuffled batches); **target network**
  (the frozen copy stepping over time); **softmax policy bars** (probabilities shifting under REINFORCE); **PPO clip** (ratio and clip band);
  **Dyna-Q / MCTS tree** (simulated updates and tree growth); **learning curve with seeds** (mean and confidence band as seeds are added).
  Keep illustrative numbers labelled.
- Animation rules as in AUTHORING.md: slow by default, dt-based, captions, speed slider, pause/step, reset, redraw only when state changes,
  `{fps:30}` for heavy labs, redraw after canvas resize.
- Equations a learner should edit (update rule, discount, step size) go through `AI.exprInput`, with safe parsing.

## Files you may create / not touch
Only `rl-<name>.html` and `assets/labs-rl-<name>.js`. Do not edit shared files (`lib.js`, `labs-core.js`, `labs-pipeline.js`, `shell.js`,
`topics.js`, `style.css`, `reinforcement-learning.html`, curriculum files, other modules). Report shared-file bugs in your final message.
No git. **Write the page file early and save often** (rate limits can interrupt agents).

## Verify
As in AUTHORING.md: `node --check` on your labs file; headless Chrome console check; screenshot review; recompute every worked example;
confirm every lesson id; confirm every link target exists; confirm any code snippet parses.
