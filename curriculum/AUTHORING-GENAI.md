# Authoring addendum — Generative AI

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/generative-ai.md`. Style references: `ds-statistics.html` and `math-information.html` (finished deep modules with exact
markup), `assets/labs-ds-statistics.js` and `assets/labs-information.js` (clean lab files), hub `generative-ai.html`.

## What makes Generative AI lessons different
- **Everything is a probability over a small vocabulary.** Teach on a toy vocabulary of 4–8 tokens, a 2–4 token sequence, a 2-D or 1-D
  distribution. Every logit, softmax, attention weight, KL value, merge count, loss and sampled probability is computed in node/python
  first and quoted to a stated precision. Show the arithmetic, not only the answer.
- **Honest about scale.** The labs are toy models. Never claim anything about a specific commercial model (size, data, benchmark scores,
  cost, release date): such facts go out of date or cannot be verified. Name published *methods* (attention, BPE, LoRA, RLHF, DPO, DDPM,
  DDIM, classifier-free guidance, flow matching) and give their formulas. Cite a paper only when you are sure of it; otherwise omit the citation.
  Where a statement depends on the field's current practice, hedge it ("commonly", "many systems").
- **Exact about what is and is not claimed.** Say precisely what a method guarantees: speculative decoding preserves the target distribution;
  DPO is derived from a KL-regularised objective; scaling laws are empirical fits and extrapolate imperfectly; guidance trades diversity for fidelity.
- **Pipeline view.** Each lesson says where it sits in the six stages (collect and tokenise, pretrain, align, adapt and compress, generate,
  evaluate and guard), what it stores, and its cost (compute and memory) in terms of formulas the learner can evaluate.
- **Simulations are labelled** *illustrative* in the caption.
- Link maths and neighbours instead of re-teaching: `math-probability.html`, `math-information.html` (cross-entropy, KL, mutual information),
  `math-neural.html`, `math-optimization.html`, `math-linear-algebra.html`, `ml-neural.html`, `ml-architectures.html`, `cv-transformers.html`,
  `cv-generative.html`, `ds-statistics.html`. Check each target and its `#lesson-id` exists (`ls`, then grep the id). Other Generative AI modules
  link as `gen-<name>.html#lesson-id`.
- **Safety, copyright and misuse are discussed even-handedly**, without sensational claims, and no instruction that would help someone cause harm.

## Page conventions (copy `ds-statistics.html` exactly for structure)
- File `gen-<name>.html` named in the curriculum module heading; `<body data-topic="genai" data-module="N">`; hero like the other modules.
- Scripts (all `defer`, in this order): KaTeX, `assets/topics.js`, `assets/curriculum-genai.js`, `assets/lib.js`, `assets/labs-core.js`
  (only if you use fplot/calc/vec2), your own `assets/labs-gen-<name>.js`, then `assets/shell.js` **last**.
- Lesson markup: `<section class="lesson" id="LESSON-ID" data-lesson>`; head `<div class="lesson-head"><span class="level" style="color:var(--purple)">N.k</span><h2>Title</h2><button class="btn small mark"></button></div>`;
  then `<div class="prose">` containing, in order: `div.question`, `h3.sec` Intuition, `h3.sec` Build it up with `div.worked` blocks,
  `div.eq` (**raw LaTeX with no `\[ \]` delimiters**), `div.sym` (every symbol explained), the lab `<div data-lab="NAME"></div>`,
  `div.try` (ol, 3 steps), `div.trap`, `div.aihook`, `h3.sec` Check yourself with three `<details class="check"><summary>…</summary><div class="ans">…</div></details>`,
  and `div.takeaways`. Inline maths is `<span class="tex">…raw LaTeX…</span>`. **Escape `<` as `&lt;` and `&` as `&amp;` everywhere, including
  inside LaTeX** (a raw `<` in a formula is parsed as an HTML tag and swallows the lab that follows).
- Lesson ids must equal the ids in `curriculum/generative-ai.md` exactly; no extra `section.lesson` elements. A short intro callout must
  not be a `section.lesson`. End with a "Module complete" callout linking to the next module page.
- Link only to pages that exist (`ls`). Colours only via `AI.col()` (call it inside each draw, because it is reset on theme change) or CSS variables.

## Labs
- Write labs in **your own file** `assets/labs-gen-<name>.js`, defined as `AI.labs['lab-name'] = function (root) { … }`. Lab names must be unique
  across the whole site: prefix them with `gen-` plus a module code. Mount with `<div data-lab="gen-…"></div>`.
- Use `AI.labShell(root, title, sub, viewHTML, ctlHTML)`, `AI.canvas(canvasEl)` (set `o.onresize = draw`), `AI.loop(root, fn, {fps:30})`,
  `AI.ticker`, `AI.exprInput`, `AI.col()`, `AI.fmt`, `AI.rng`, `AI.randn`. Draw only when state changes.
- **`.readout` is `white-space: pre`** in the shared CSS, so long captions run off the side: set `style="white-space:normal"` on readouts that hold sentences.
- Animation rules (AUTHORING.md): dt-based, slow by default, speed slider, pause/step, reset, captions that say what is happening,
  redraw after resize. Seeded random numbers (`AI.rng(seed)`) so a run can be repeated. Controls the learner edits live (temperature, top-p,
  a logit, a rank, a guidance scale, a token, a noise level, a formula via `AI.exprInput`).
- Ideas by module: **language** (BPE merges one by one, count table, a softmax with temperature/top-k/top-p, perplexity on a sentence),
  **transformer** (editable Q and K giving attention weights, heads, rotary rotation of a 2-D vector, a block's residual stream, causal mask, KV cache growing),
  **training** (power-law loss curve, 6ND compute calculator, memory-per-GPU stack, sharding), **alignment** (Bradley–Terry pairs, a policy and KL leash,
  DPO margin, group-relative advantages), **efficiency** (low-rank update, quantisation grid and error, prefill/decode timeline, speculative acceptance, MoE router),
  **applications** (few-shot shifting a distribution, grammar-masked logits, cosine retrieval, a ReAct trace), **latent** (VAE ELBO terms, discriminator and generator, a flow warping a density),
  **diffusion** (forward noising of 1-D/2-D points, a denoising arrow field, DDPM vs DDIM paths, guidance sharpening), **multimodal/production** (contrastive matrix, evaluation confidence, a guardrail pipeline).

## Verification (use real time, not headless virtual time)
- `node --check` on your labs file; confirm lesson ids equal the spec; every `data-lab` is defined in your file; all link targets and `#anchors` exist.
- Headless Chrome with `--virtual-time-budget` barely advances `requestAnimationFrame`, so labs look frozen in such screenshots. Use the real-time tools instead
  (they use puppeteer-core, already installed at `C:\Users\iabde\AppData\Local\Temp\smoke`):
  - `node tools/shot.js C:\Users\iabde\AppData\Local\Temp\smoke <repo-dir> <page.html> <anchor-id-or-empty> <out.png> [width] [height]` — loads the page in real time, scrolls to the anchor,
    prints text length, formula count, lab canvas count and console errors, and saves a screenshot. Read the screenshot (open the PNG).
  - `node tools/smoke.js C:\Users\iabde\AppData\Local\Temp\smoke <repo-dir> "^your-page\.html$" <out.json>` — clicks every button, slider and select in every lab and reports console errors and phone-width (390 px) overflow.
  - `node tools/check-links.js` from the repo root — every link and anchor in every page.
- No raw `\frac` or `\[` may remain as visible text (KaTeX keeps a hidden copy of the source in an annotation node; that is fine).
- Check the page at 390 px width: nothing may be wider than the screen (the shared CSS handles tables and long formulas).

## Files you may create / not touch
Only `gen-<name>.html` and `assets/labs-gen-<name>.js`. Do not edit shared files (`lib.js`, `labs-core.js`, `labs-pipeline.js`, `shell.js`,
`topics.js`, `style.css`, `generative-ai.html`, curriculum files, other modules). Report shared-file bugs in your final message.
No git. Use a private scratch subfolder (named after your module) inside the scratchpad, never the shared scratchpad root.
**Write the page file early and save often** (rate limits and dropped connections can interrupt agents). Write long files with the Write tool, not shell heredocs.
