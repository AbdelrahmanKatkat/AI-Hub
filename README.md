# AI Learning Hub

Interactive, animated lessons that take you from the basics to expert-level AI. Static site — no build step.

**Live topics (first draft):** Machine Learning, Mathematics for AI.
Planned: Computer Vision, Generative AI, NLP, Voice AI, Data Science, Governing AI, Reinforcement Learning, Domains (GeoAI, Robotics…).

## Run locally
```
python -m http.server 8000   # then open http://localhost:8000
```

## Deploy on GitHub Pages
Deployment is automatic: `.github/workflows/pages.yml` publishes on every push to `main`.

## Structure
- `index.html`, `machine-learning.html`, `mathematics.html` — one page per topic
- `assets/topics.js` — topic registry (sidebar, search, home cards) and pipeline stage definitions
- `assets/lib.js` — expression parser + autodiff, canvas/plot helpers
- `assets/labs-*.js` — interactive labs, mounted by `<div data-lab="name">`
- `assets/shell.js` — header, sidebar, outline, search, progress, theme

## Adding a topic
1. Add an entry to `HUB.topics` (set `status: 'live'`, `href`, `sections`) and a pipeline in `PIPELINES`.
2. Copy an existing topic page, set `<body data-topic="id">`, write sections as `<section class="lesson" id="...">`.
3. Add labs in `assets/labs-<topic>.js` as `AI.labs.name = function (root) {…}`.
