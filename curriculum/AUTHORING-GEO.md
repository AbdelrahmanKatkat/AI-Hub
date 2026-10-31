# Authoring addendum — GeoAI

Read `curriculum/AUTHORING.md` first (lesson standard, lab API, animation rules, verification). Spec of every lesson:
`curriculum/geoai.md`. Style references: `math-foundations.html` (deep lessons), `ml-data.html` (data-preparation lessons with a
pipeline view), `cv-architectures.html` (architecture blueprints), hub `geoai.html`.

## What makes GeoAI lessons different
- **Maps first.** Start from a map or a picture of ground cells, then the number. Projections, resolutions and dates are always stated.
- **Hand-worked on a tiny raster or catalogue.** E.g. NDVI on a 2×2 pixel patch; a 4×4 raster tiled into 2×2 chips with overlap; a
  spatial split of a 6×6 grid vs a random split (count the leaking neighbours); a confusion matrix from 30 sample points with
  area weights. Compute everything in code first and quote exact numbers.
- **STAC lessons show real JSON**, valid against the current specs (STAC core: Catalog, Collection, Item, Asset; STAC API `/search`).
  Only use field names you are sure of. If unsure, leave the field out and say so.
- **STAC MLM lessons** describe models with the MLM extension (current schema v1.6.0, prefix `mlm:`). Verified facts to use:
  - Core fields: `mlm:name`, `mlm:architecture`, `mlm:tasks`, `mlm:framework`, `mlm:framework_version`, `mlm:memory_size`,
    `mlm:total_parameters`, `mlm:pretrained`, `mlm:pretrained_source`, `mlm:batch_size_suggestion`, `mlm:accelerator`,
    `mlm:accelerator_constrained`, `mlm:accelerator_summary`, `mlm:accelerator_count`, `mlm:input`, `mlm:output`, `mlm:hyperparameters`.
  - Item-only fields (cannot sit on Assets): `mlm:name`, `mlm:input`, `mlm:output`, `mlm:hyperparameters`. Apart from `mlm:name`, these are also left out of Collection summaries.
  - Asset fields: `mlm:model`, `mlm:source_code`, `mlm:container`, `mlm:training`, `mlm:inference`, `mlm:artifact_type`, `mlm:compile_method`, `mlm:entrypoint`.
  - Asset roles: `mlm:model` is required for the main model asset; `mlm:source_code` (with role `code`) is recommended; others include
    `mlm:weights`, `mlm:checkpoint`, `mlm:inference-runtime`, `mlm:training-runtime`, `mlm:compiled`; plain `model`, `weights`, `runtime`, `code` also occur.
  - Task values: `regression`, `classification`, `scene-classification`, `detection`, `object-detection`, `segmentation`,
    `semantic-segmentation`, `instance-segmentation`, `panoptic-segmentation`, `similarity-search`, `generative`, `image-captioning`,
    `super-resolution`, `downscaling`.
  - Delegation: MLM reuses `eo`, `raster`, `datacube`, `classification`, `processing`, `file`, `label`, `item-assets`. Link type
    `derived_from` points from a fine-tuned model to its base model. An MLM item should not list the MLM extension as its only extension.
  - When in doubt, read the spec: https://github.com/stac-extensions/mlm (and the schema at stac-extensions.github.io/mlm/v1.6.0/schema.json).
- **Be honest about projections, units and sensors.** State CRS, resolution and date. Quote real sensor facts only when you are sure
  (e.g. Sentinel-2 bands 10 m for B2/B3/B4/B8; Landsat 30 m; Sentinel-1 is radar and sees through cloud).
- **Pipeline view.** Each lesson ends its build-up with where the step sits in the six-stage pipeline (acquire, preprocess, model,
  inference, post-process, publish & monitor) and what it stores.

## Page conventions
- File `geo-<name>.html` (from the curriculum module headings); `<body data-topic="geoai" data-module="N">`; hero like other modules; end with a "Module complete" callout to the next module.
- Scripts (all `defer`, in this order): KaTeX, `assets/topics.js`, `assets/lib.js`, `assets/labs-core.js` (only if you use fplot/calc/vec2), `assets/labs-pipeline.js` (if you use the pipeline lab),
  your own `assets/labs-geo-<name>.js`, then `assets/shell.js` **last**. The shell loads the curriculum data (`assets/curriculum-geoai.js`) itself; do not include it.
- Lesson ids must equal the ids in `curriculum/geoai.md`.
- Other pages: link as `ml-*.html`, `cv-*.html`, `math-*.html`, `geo-*.html`. Link only to pages that exist (check with `ls` before you finish).

## Labs
- Write new labs in **your own file** `assets/labs-geo-<name>.js` as `AI.labs.myLab = function (root) {…}`. Use `AI.labShell`, `AI.canvas`,
  `AI.loop(root, fn, {fps})`, `AI.ticker`, `AI.exprInput`, `AI.view/axes/arrow`, `AI.col`, `AI.fmt`, `AI.rng`.
- Ideas (use what fits your module): a **STAC catalogue browser** (catalogue → collection → item → asset tree that opens step by step; a search
  box that filters footprints on a map by bbox, date and cloud cover; the search JSON updating); **raster band explorer** (stacked bands,
  editable band maths, NDVI/NDWI pixel by pixel); **tiling & stitching** (a scene cut into overlapping tiles and stitched back, seams shown);
  **spatial vs random split** (grid of blocks with train/validation colours; leakage count of neighbours); **U-Net shape walk** (encoder
  shrinking, decoder growing, skips); **sliding-window inference** (a map painted tile by tile with blended overlaps); **accuracy
  assessment** (random sample points checked against reference, confusion matrix filling, area-weighted estimates);
  **MLM record flow** (a model's metadata record read field by field, the runner setting up inference from it); **catalogue harvest** (footprints
  drawn, time series stacked). Keep illustrative numbers labelled.
- Animation rules as in AUTHORING.md: slow by default, dt-based, captions, speed slider, pause/step, redraw only when state changes,
  `{fps:30}` for heavy labs, redraw after canvas resize.

## Files you may create / not touch
Only `geo-<name>.html` and `assets/labs-geo-<name>.js`. Do not edit shared files (`lib.js`, `labs-core.js`, `labs-pipeline.js`, `shell.js`,
`topics.js`, `style.css`, `geoai.html`, curriculum files, other modules). Report shared-file bugs in your final message. No git.
**Write the page file early and save often** (rate limits can interrupt agents).

## Verify
As in AUTHORING.md: `node --check` on your labs file, headless Chrome console check, screenshot review; recompute every worked example;
confirm every lesson id; confirm every link target exists; confirm JSON examples are syntactically valid (parse them).
