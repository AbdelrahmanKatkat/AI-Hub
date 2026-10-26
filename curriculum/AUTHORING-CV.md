# Authoring addendum — Computer Vision

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). This file adds what is specific to
the Computer Vision topic. Spec of every lesson: `curriculum/computer-vision.md`. Reference page for style/structure: `math-foundations.html`
(deep lessons) and `computer-vision.html` (CV hub, shows the shared labs).

## Emphasis: animation and architecture
- **Animation:** the learner must *see data move* — pixels through kernels, tensors changing shape, boxes filtered, patches becoming tokens.
- **Architecture:** every model is taught as a blueprint: blocks, the tensor shape at each stage, parameters, MACs/FLOPs, receptive field,
  and *why* each design choice exists (the problem the previous design had). Numbers must be exact and match published models
  (ResNet-50 25.56 M params / 4.09 GMACs, VGG-16 138.36 M, MobileNet-v2 3.50 M, ViT-B/16 86.57 M …). Compute by hand/node before writing.
- Architecture lessons end with `<h3 class="sec">Draw it yourself</h3>`: block diagram + shape table + parameter count of the main blocks,
  and a worked count (e.g. "conv 3×3, 64→128: 64·9·128 = 73,728 weights").

## Page conventions
- File `cv-<name>.html` (names in the curriculum table headings); `<body data-topic="cv" data-module="N">`.
- Scripts (all `defer`, in this order): KaTeX, `assets/topics.js`, `assets/lib.js`, `assets/arch.js`, `assets/labs-pipeline.js`,
  `assets/labs-core.js` (only if you use fplot/calc/vec2…), `assets/labs-cv.js`, your own `assets/labs-cv-<name>.js` (if any), then `assets/shell.js` last.
  The shell loads the curriculum data and builds sidebar/outline/progress/prev-next itself. Do **not** include curriculum scripts.
- Hero like Module 0 of math; end with a "Module complete" callout pointing to the next module.
- Lesson ids must equal the ids in `curriculum/computer-vision.md`.

## Shared labs you can mount (read `assets/labs-cv.js` and `assets/arch.js` headers)
- `<div data-lab="pixels"></div>` — zoom into pixels: draggable window, magnified numbers, resolution + bit-depth sliders, gray/RGB/channel views, animation button.
- `<div data-lab="convolve" data-kernel="sobel x"></div>` — animated sliding 3×3 kernel with editable weights, presets, padding, stride, step-by-step arithmetic.
- `<div data-lab="archviz" …></div>` — **architecture visualiser**: boxes are tensors (height ∝ image size, thickness ∝ channels), a pulse flows through,
  skip arcs, click a block for its sub-layer table (shapes, params, MACs, receptive field). Attributes: `data-preset="resnet18"`,
  `data-presets="a,b,c"` (model selector), `data-edit` (editable architecture text), `data-resolution` (input-size slider), `data-title`, `data-height`.
  Presets: small-cnn, lenet5, alexnet, vgg16, resnet18, resnet50, mobilenetv1, mobilenetv2, densenet121, vit-b16, unet-small.
  **Inline custom architecture:** `<div data-lab="archviz" data-title="…"><script type="text/plain">input 3 32 32\nconv 16 k3\nbn\nrelu\n…</script></div>`.
  Layer DSL (see header of `assets/arch.js`): `input C H W`, `conv OUT kK sS pP gG`, `dwconv`, `bn relu relu6 gelu`, `maxpool avgpool gap flatten fc OUT`,
  `res OUT [sS]`, `bottleneck OUT midM [sS]`, `dwsep OUT [sS]`, `invres OUT tT [sS]`, `dense L G`, `up OUT`, `concat @label`, `add @label`,
  `patch P D`, `cls`, `pos`, `norm`, `tblock HEADS MLP xL`, `seqfc OUT`; suffix `xN` repeats a layer; `@name` labels a block for skips.
  Use it for every architecture lesson (show the model, then let the learner modify it). You may add new presets in *your own* labs file via `AIArch.PRESETS.myname = {title, spec}`.
- `AIArch.build(specText, {resolution})` returns blocks/totals in JS (use it in your own labs for live shape/param calculators).
- `AI.cv.scene(kind)` / `AI.cv.resample(kind, n)` / `AI.cv.gray(img)` give procedural test images (`shapes, house, digit, edges, gradient, checker`) as pixel arrays.
- From `labs-core.js`: `fplot`, `calc`, `vec2`, `decode`, etc. `data-lab="pipeline" data-id="cv"` for the vision pipeline.

## Build new labs in your own file `assets/labs-cv-<name>.js`
Register `AI.labs.myLab = function (root) {…}` using `AI.labShell`, `AI.canvas`, `AI.loop(root, fn, {fps})`, `AI.ticker`, `AI.exprInput`, `AI.view/axes/arrow`, the
`AI.cv.*` scene helpers and the animation rules (slow by default, dt-based, captions, speed slider, pause/step, redraw only when needed).
Every lesson needs at least one **animated, editable** lab (a kernel, a size, a stride, a layer list, a threshold, a prompt…).

## Files you may create / not touch
Create only `cv-<name>.html` and `assets/labs-cv-<name>.js`. Do not edit any shared file (`lib.js, arch.js, labs-cv.js, labs-core.js, labs-math.js, shell.js,
topics.js, style.css`, curriculum files, other modules). If you find a bug in a shared file, report it in your final message instead of fixing it. No git.

## Verify
As in AUTHORING.md (node --check, headless Chrome console check, screenshot review). Additionally check every architecture preset/spec you show builds
without errors (`node -e "const A=require('./assets/arch.js'); …"`) and that numbers quoted in the text match `AIArch.build` totals and the published values.
