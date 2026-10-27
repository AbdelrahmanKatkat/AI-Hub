# AI Learning Hub — project guide for Claude

A static, GitHub-Pages-hosted learning site that teaches AI from zero to expert, one topic per page-set, with
**editable equations and animated labs**. No build step: plain HTML/CSS/JS. Owner: Abdelrahman Katkat.
Live: https://abdelrahmankatkat.github.io/AI-Hub/ · repo: AbdelrahmanKatkat/AI-Hub

## Topic order (sidebar, home grid) — do not reorder without being asked
Mathematics → Machine Learning → Data Science → Computer Vision → NLP → Voice → Generative AI → Governing AI →
Reinforcement Learning → Domains (GeoAI, Robotics, Healthcare & Bio, Climate & Earth, Finance, Other — each a sub-topic).
Live now: Mathematics (9 modules / 58 lessons), Machine Learning (10 modules / 63 lessons), Computer Vision (10 modules / 70 lessons). Others: `status: 'soon'`.

## Layout
```
index.html                      home (pipeline + topic grid)
mathematics.html                curriculum hub for the Math topic (map + module cards)
math-<module>.html              one page per math module (lessons are <section class="lesson" id="lesson-id">)
machine-learning.html           ML hub; modules are ml-<name>.html (same pattern as cv-*.html / math-*.html)
curriculum/<topic>.md           SOURCE OF TRUTH for what each lesson covers (tables: id | title | must understand | AI link)
curriculum/AUTHORING.md         the lesson standard, lab API, animation rules, verification steps  (READ before writing content)
tools/gen-curriculum.js         curriculum/<topic>.md -> assets/curriculum-<slug>.js for math, cv, ml (run `node tools/gen-curriculum.js` after editing an md)
assets/topics.js                topic registry (order, status, children) + PIPELINES (animated stage definitions)
assets/curriculum-math.js       GENERATED — never edit by hand
assets/lib.js                   expression parser + autodiff, canvas/plot helpers, AI.loop/AI.ticker
assets/shell.js                 header, sidebar tree, outline, search, progress (localStorage), theme, prev/next module nav
assets/labs-*.js                interactive labs, mounted by <div data-lab="name" …>
assets/style.css                GitHub-dark/light theme + lesson building blocks
.github/workflows/pages.yml     deploys every push to main (copies *.html + assets). No manual deploy step.
.claude/skills/                 project skills (see below)
```

## Hard rules
1. **Curriculum first.** Content is written *from* `curriculum/<topic>.md`. To change scope, edit the md, regenerate, then write lessons.
2. **Every lesson meets the deep-lesson standard** in `curriculum/AUTHORING.md`: question → intuition → worked example → formal
   definition (every symbol explained) → lab with "try this" → traps → AI connection → 3 self-check questions → takeaways.
   Explanations must be intuitive and patient, building from the simplest case to full understanding — not notes.
3. **Mathematically exact.** Compute every worked example and quiz answer numerically (node/python) before writing it.
4. **Animations: slow, real-time, meaningful.** Use `dt`/`AI.ticker` (never per-frame stepping), slow defaults, speed slider,
   pause/step, captions that say what is happening, smooth easing, redraw only when state changed, `{fps:30}` for heavy labs.
5. **Edit equations live**: labs should let the learner change a formula/number and see the animation respond (use `AI.exprInput`).
6. Colours only via `AI.col()` / CSS variables so dark and light themes both work.
7. Pages are flat files at repo root (no sub-folders) so relative links and the Pages workflow stay simple.
8. Shared files (`lib.js`, `labs-core.js`, `shell.js`, `style.css`, `topics.js`) are edited deliberately and by one author at a time;
   parallel authors create only their own `math-<module>.html` + `assets/labs-<module>.js`.

## Git
- Commit and push **as the user** (git config is already set). **Do not add Claude/Co-Authored-By/"Generated with" lines** to commits
  or PRs, and never add Claude as a contributor — the user explicitly asked for this.
- Do not commit `AI-Hub.jpg` (original design mockup) unless asked.
- Pushing to `main` redeploys the site automatically via GitHub Actions.

## Verify before declaring done
`node --check assets/*.js`, then load pages headless and read console + screenshot (see skill `verify-site`).
A page is not done until labs render, equations typeset (no raw `\frac`), and there are no console errors.

## Project skills (in .claude/skills)
- `plan-curriculum` — design the zero→hero curriculum for a new topic and generate its data file.
- `write-deep-lesson` — author a module/lesson to the deep standard (the process used for Mathematics).
- `add-lab` — build a new interactive, slow, intuitive animated lab.
- `add-topic` — register/launch a new topic (registry, hub page, modules, sidebar, workflow).
- `verify-site` — headless Chrome check + screenshot review.

## Where things stand / next
- **Mathematics**: hub + modules 0-4 and 6-8 written to the deep standard (`math-*.html`). Module 5 (information theory, `math-information.html`)
  is a *placeholder* — write it (entropy, cross-entropy-kl, mutual-info) per `curriculum/mathematics.md`.
- **Computer Vision**: live. Hub `computer-vision.html` + modules 0-9 (`cv-*.html`, 70 lessons). Shared tooling: `assets/arch.js` (architecture
  engine + DSL, verified against published parameter/MAC counts) and `assets/labs-cv.js` (`pixels`, `convolve`, `archviz`). Spec in
  `curriculum/computer-vision.md`; authoring addendum `curriculum/AUTHORING-CV.md`.
- Known review items: agents verified pages in headless Chrome only (no real-time smoothness check, no hand-clicked interactions);
  arch.js reports ResNet-50 final receptive field 427 px vs the commonly quoted 483 (check definition); DeepLab/SAM archviz specs are
  approximations (stated in the lessons).
- **Machine Learning**: live. Hub `machine-learning.html` + modules 0-9 (`ml-*.html`, 63 lessons). Shared toolkit `assets/mllib.js` (datasets, k-NN, CART,
  k-means, logistic, PCA, boundary renderer) + existing labs in `assets/labs-ml.js`. Spec `curriculum/machine-learning.md`; addendum `curriculum/AUTHORING-ML.md`.
  Many lab simulations are labelled *illustrative*; worked numbers were computed in code.
- Next topics in order: Data Science, NLP, Voice, Generative AI, Governing AI, Reinforcement Learning, Domains (GeoAI, Robotics, ...).
- Agent runs can hit the session rate limit: when that happens, resume the agents (SendMessage) — their files are saved on disk.
