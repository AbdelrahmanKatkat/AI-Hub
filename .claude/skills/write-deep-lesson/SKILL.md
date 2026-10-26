---
name: write-deep-lesson
description: Author one lesson or a whole module of the AI Hub to the deep, intuition-first standard (question, intuition, worked example, formal definition, labs, traps, AI link, self-check). Use whenever writing or upgrading lesson content such as math-*.html or topic pages.
---

# Write a deep lesson

Read first: `curriculum/AUTHORING.md` (standard + APIs) and the reference page `math-foundations.html`. The lesson ids and scope come from
`curriculum/<topic>.md`.

## Process for each lesson
1. **Find the right first example.** What concrete problem makes the idea *necessary*? Write it as "The question".
2. **Intuition before symbols.** Analogy or picture in plain words (balance scale, wheel with a lamp, vending machine). Then make the
   analogy precise. A beginner must be able to follow every sentence.
3. **Build up in small steps**, simplest case first (1-D → 2-D → n-D; one example → many). Never skip a step a learner could trip on.
4. **Hand-worked example with real numbers.** Compute it in the shell first (node/python) — numbers and quiz answers must be right.
5. **Formal definition** in KaTeX plus a `table.sym` that explains *every symbol* in words. State conditions (e.g. invertible, a≠0).
6. **Labs.** Prefer an existing lab (`fplot`, `calc`, `vec2`, `matrix`, …; see AUTHORING.md); otherwise use skill `add-lab`. Precede each with
   a "Try this" list of concrete actions and what the learner should notice.
7. **Common traps** (real misconceptions), **Where it shows up in AI** (named models/algorithms), **3 self-check questions** with
   answers in `<details>`, **Takeaways**.
8. Length: ~700–1500 words of prose per lesson plus equations/examples/labs. Depth over brevity; patient teacher tone, not lecture notes.

## HTML conventions
- `<section class="lesson" id="LESSON-ID" data-lesson>` with `lesson-head` (level badge `M.N`, `<h2>`, `<button class="btn small mark">`).
- Prose in `<div class="prose">`; labs sit *between* prose blocks (labs are full width).
- Equations: `<div class="eq">TeX</div>`, inline `<span class="tex">TeX</span>`; escape `&`→`&amp;`, `<`→`&lt;`.
- Body: `<body data-topic="math" data-module="N">`; scripts deferred, `shell.js` last.

## Parallel authoring
For a big topic, give each module to its own agent. Brief = point at AUTHORING.md + curriculum md + reference page + lab ideas for that
module; restrict each agent to its own `math-<module>.html` and `assets/labs-<module>.js`; tell them not to touch shared files or run git.
Afterwards review every module yourself (skill `verify-site`), re-check maths, then commit.

## Self-review checklist
- [ ] all required parts present in every lesson, ids match the curriculum
- [ ] each worked example/answer recomputed
- [ ] every symbol explained at first use; no unexplained jump
- [ ] ≥1 lab per lesson that the learner can *edit*; animations slow + captioned
- [ ] no raw TeX visible, no console errors, readable on mobile width
