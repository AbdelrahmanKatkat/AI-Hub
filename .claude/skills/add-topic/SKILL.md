---
name: add-topic
description: Launch a new AI Hub topic (Data Science, Computer Vision, NLP, Voice, Generative AI, Governing AI, Reinforcement Learning, a Domain sub-topic, or deepening Machine Learning) end-to-end - curriculum, registry, hub page, module pages, labs, deploy.
---

# Add or upgrade a topic

Order of work (the same path Mathematics followed):

1. **Plan** — skill `plan-curriculum` → `curriculum/<topic>.md` + generated data file. Confirm scope with the user if it is ambiguous.
2. **Registry** — edit `assets/topics.js`: set the topic's `status: 'live'`, `href`, `tags`, `next`, a `PIPELINES.<id>` entry
   (6 animated stages: what happens, `compute`, `storage`) and keep the global topic order (see CLAUDE.md). Domain sub-topics live in
   `children` of the Domains entry; promote a child to its own topic entry/page when it gets content.
3. **Hub page** `<topic>.html` — copy `mathematics.html`: hero, how-to-learn, map (path bar + pipeline lab), module cards.
4. **Module pages** `<topic>-<module>.html` with `<body data-topic="<id>" data-module="N">`; wire the shell if the topic is not `math`
   (generalise the `MATH_MODULES` handling in `assets/shell.js` to read `<ID>_MODULES` per topic — keep behaviour identical for math).
5. **Author lessons** — skill `write-deep-lesson` (parallelise per module). Labs via skill `add-lab`; reuse the shared pipeline lab.
6. **Domain-specific pipeline** — every topic page walks the same stages: preprocessing → computation → inference → post-processing →
   storage. Make the animated pipeline concrete for the topic (e.g. CV: resize/normalise → conv layers → feature maps → NMS → store).
7. **Verify** — skill `verify-site`; check sidebar order, progress counts, search results, prev/next navigation, mobile width.
8. **Ship** — commit as the user (no Claude attribution), push to `main`; the Pages workflow deploys automatically.

## Checklist
- [ ] topic order unchanged except for the new topic going live
- [ ] hub shows module cards and % progress; sidebar tree expands for the current module
- [ ] search finds lessons (keywords come from the curriculum table)
- [ ] README/CLAUDE.md "where things stand" updated
