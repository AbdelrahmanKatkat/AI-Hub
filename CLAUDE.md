# AI Learning Hub — project guide for Claude

A static, GitHub-Pages-hosted learning site that teaches AI from zero to expert, one topic per page-set, with
**editable equations and animated labs**. No build step: plain HTML/CSS/JS. Owner: Abdelrahman Katkat.
Live: https://abdelrahmankatkat.github.io/AI-Hub/ · repo: AbdelrahmanKatkat/AI-Hub

## Topic order (sidebar, home grid) — do not reorder without being asked
Mathematics → Machine Learning → Data Science → Computer Vision → NLP → Voice → Generative AI → Governing AI →
Reinforcement Learning → Domains (GeoAI, Robotics, Healthcare & Bio, Climate & Earth, Finance, Other — each a sub-topic).
Live now: Mathematics (9 modules / 58 lessons), Machine Learning (10 modules / 63 lessons), Computer Vision (10 modules / 70 lessons), GeoAI (10 modules / 63 lessons, a sub-topic of Domains, includes STAC and STAC MLM), Robotics (10 modules / 59 lessons, a sub-topic of Domains), Reinforcement Learning (10 modules / 56 lessons), Data Science (10 modules / 59 lessons), Generative AI (10 modules / 61 lessons). Others: `status: 'soon'`.

## Layout
```
index.html                      home (pipeline + topic grid)
mathematics.html                curriculum hub for the Math topic (map + module cards)
math-<module>.html              one page per math module (lessons are <section class="lesson" id="lesson-id">)
machine-learning.html           ML hub; modules are ml-<name>.html (same pattern as cv-*.html / math-*.html)
curriculum/<topic>.md           SOURCE OF TRUTH for what each lesson covers (tables: id | title | must understand | AI link)
curriculum/AUTHORING.md         the lesson standard, lab API, animation rules, verification steps  (READ before writing content)
tools/gen-curriculum.js         curriculum/<topic>.md -> assets/curriculum-<slug>.js for math, cv, ml, geoai, robotics, rl, ds, genai (run `node tools/gen-curriculum.js` after editing an md)
tools/make-frames.js            writes the panel frames and the neon theme CSS into assets/style.css (generated block; run after editing it)
tools/shot.js                   real-time screenshot of a page (THEME=neon node tools/shot.js ...)
tools/smoke.js                  interaction smoke test: clicks every lab control on every page, reports console errors and phone-width overflow (needs puppeteer-core; usage in the file header)
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
6. Colours only via `AI.col()` / CSS variables so the dark, light and neon themes all work (theme = `data-theme` on `<html>`; the header button cycles dark → light → neon and the choice is stored in localStorage). Call `AI.col()` inside each draw: it is reset on the `themechange` event.
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
- **Mathematics**: hub + modules 0-8 written to the deep standard (`math-*.html`). Module 5 (information theory, `math-information.html`) covers
  entropy, cross-entropy and KL, and mutual information, with three labs in `assets/labs-information.js`.
- **Computer Vision**: live. Hub `computer-vision.html` + modules 0-9 (`cv-*.html`, 70 lessons). Shared tooling: `assets/arch.js` (architecture
  engine + DSL, verified against published parameter/MAC counts) and `assets/labs-cv.js` (`pixels`, `convolve`, `archviz`). Spec in
  `curriculum/computer-vision.md`; authoring addendum `curriculum/AUTHORING-CV.md`.
- Known review items: agents verified pages in headless Chrome only (no real-time smoothness check, no hand-clicked interactions; `tools/smoke.js`
  now clicks every control but cannot judge how an animation looks). arch.js reports ResNet-50 final receptive field 427 px because it puts the
  stride on the 3x3 conv (torchvision v1.5 layout); the original paper's layout gives 483 (explained in `cv-cnn.html`). DeepLab/SAM archviz specs
  are approximations (stated in the lessons).
- Themes (dark, light, neon; `data-theme` on `<html>`, the header button cycles them) and panel frames: the frame CSS at the end of `assets/style.css` between `@@FRAMES-START` and `@@FRAMES-END` is GENERATED by `tools/make-frames.js` (edit the generator, then run `node tools/make-frames.js`). Frames are nine-slice SVG border images: chamfered (cut) corners, a diagonal gradient outline (yellow, cyan, magenta in neon) and a baked-in glow; they are used by the sidebar, outline, hero, every `section.lesson`, labs, pipelines, cards and the small blocks (question, worked, trap, ...). Pipeline `.stage` cards use a coloured pseudo-element frame instead (they carry their own `--c`). The design follows `Neon AI Learning Hub Dashboard.png`. The neon background `assets/neon-city.jpg` is the picture supplied with the design (converted from PNG to a 330 KB JPEG) and is loaded only by the neon theme; replace the JPG to change it. Neon also loads the Poppins heading font, shows the topic icons as line icons (masks keyed by `data-t` on the sidebar links) and highlights one word of the page title (`.hl`). Keep glow filters off animated elements (labs); frame images cost nothing at runtime. Design mockups (`AI-Hub.jpg`, the neon dashboard PNG) are not committed (they are in `.git/info/exclude`).
- Notes feature: `assets/shell.js` adds a Notes button to every page (notes saved in the browser, quotes from selected text, Markdown export).
  Phone layout (<= 600px) rules are at the end of `assets/style.css`. `.readout` is `white-space: pre`, so long captions in a readout need an inline `white-space:normal`.
- STAC: every Item example in the GeoAI pages was validated against the official STAC 1.1.0 / 1.0.0, projection, eo, classification and MLM v1.6.0
  schemas (MLM v1.6.0 rejects `mlm:entrypoint` in Item properties; named `mlm:input` bands need the `eo` or `raster` extension or STAC 1.1 `bands`).
- **Machine Learning**: live. Hub `machine-learning.html` + modules 0-9 (`ml-*.html`, 63 lessons). Shared toolkit `assets/mllib.js` (datasets, k-NN, CART,
  k-means, logistic, PCA, boundary renderer) + existing labs in `assets/labs-ml.js`. Spec `curriculum/machine-learning.md`; addendum `curriculum/AUTHORING-ML.md`.
  Many lab simulations are labelled *illustrative*; worked numbers were computed in code.
- **GeoAI**: live (sub-topic of Domains). Hub `geoai.html` + modules `geo-*.html`. Spec `curriculum/geoai.md`; addendum `curriculum/AUTHORING-GEO.md` (verified STAC MLM v1.6.0 fields). Labs in `assets/labs-geo-<module>.js`.
- **Robotics**: live (sub-topic of Domains). Hub `robotics.html` + modules `robo-*.html`, 59 lessons. Spec `curriculum/robotics.md`; addendum `curriculum/AUTHORING-ROBOTICS.md`. Labs in `assets/labs-robo-<module>.js`. Most lab simulations are labelled *illustrative*; no hardware was tested.
- **Reinforcement Learning**: live. Hub `reinforcement-learning.html` + modules `rl-*.html`, 56 lessons. Spec `curriculum/reinforcement-learning.md`; addendum `curriculum/AUTHORING-RL.md`. Labs in `assets/labs-rl-<module>.js`. Most lab simulations are labelled *illustrative*; deep-RL labs are small plain-JS models, not real training runs.
- **Data Science**: live. Hub `data-science.html` + modules `ds-*.html`, 59 lessons. Spec `curriculum/data-science.md`; addendum `curriculum/AUTHORING-DS.md`. Labs in `assets/labs-ds-<module>.js` (names prefixed `ds-`). Most lab simulations are labelled *illustrative*; worked numbers were computed in code.
- **Generative AI**: live. Hub `generative-ai.html` + modules `gen-*.html`, 61 lessons (language modelling, Transformer, LLM training, alignment, efficient inference, retrieval and agents, VAEs/GANs/flows, diffusion and flow matching, multimodal, evaluation and responsible generation). Spec `curriculum/generative-ai.md`; addendum `curriculum/AUTHORING-GENAI.md`. Labs in `assets/labs-gen-<module>.js` (names prefixed `gen-`). All labs are small toy models labelled *illustrative*; no claim is made about any specific commercial model, and statements about published methods outside the formulas are hedged. Real-time verification tools: `tools/shot.js` (screenshot + DOM facts), `tools/smoke.js`, `tools/check-links.js` (headless `--virtual-time-budget` barely advances animations).
- Authoring hazard: never give a lab control `id="side"`, `id="hdr"` or `id="outline"`: the shell's CSS for those ids applies to it.
- Next topics in order: NLP, Voice, Governing AI, then the other Domains (Healthcare & Bio, Climate & Earth, Finance, Other).
- Agent runs can hit the session rate limit: when that happens, resume the agents (SendMessage) — their files are saved on disk.
