# GeoAI — Curriculum (zero → hero)

Goal: a learner who knows what a map is ends able to take a geospatial question from **finding data** (STAC catalogues and search),
through **preparing rasters and labels without spatial leakage**, to **training and evaluating** deep-learning models on Earth
observation imagery, and to **describing, sharing and running** those models with the **STAC MLM** extension so that other people and
machines can discover, verify and apply them.

**Emphases:** (1) **Data as a catalogue** — STAC is taught as a live, browsable structure (catalogue → collection → item → asset) with
search by place, time and property. (2) **Animation** — rasters are shown as grids of bands, tiles are cut and stitched, spectral
indices are computed pixel by pixel, sliding-window inference paints a map, and metadata flows from a model's STAC MLM item to a
run-time pipeline. (3) **Architecture** — the full path from cloud archive to model to published result, with shapes, tile sizes,
bands, parameters and compute at every step.

Order is strict. Ten modules, 63 lessons. Module pages are `geo-<name>.html`; hub `geoai.html`.

Mathematics and ML prerequisites are linked, not re-taught: `math-linear-algebra.html`, `math-probability.html`,
`ml-evaluation.html` (metrics), `ml-neural.html` (training), `cv-architectures.html` (U-Net and backbones), `cv-processing.html` (filters).

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete geospatial problem that makes the idea necessary.
2. **Intuition** – plain words, a map or picture first.
3. **Build it up** – step by step with a hand-worked example on a tiny raster, table or catalogue (real numbers).
4. **Formal definition** – equations, metadata schema fragments (JSON) where relevant, and a symbol table.
5. **Animated lab** – the learner changes a parameter, a band, a search, a tile size or a field and sees the result move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real datasets, providers, models and workflows.
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.

---
## Module 0 · Geospatial foundations  (`geo-foundations.html`)
*Animation focus: a map projected onto a sphere and flattened; a pixel grid laid over ground cells; bands stacking into a cube.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| maps-coordinates | Coordinates, datums & the Earth as a shape | latitude/longitude; geographic vs projected coordinates; datums (WGS84); why distances need a projection | every GeoAI project |
| crs-projections | Coordinate reference systems & projections | CRS definitions (EPSG codes); equal-area, conformal, equidistant; reprojection distortion; choosing a CRS for a study area | data alignment bugs |
| vector-vs-raster | Vector and raster data | points, lines, polygons vs grids of cells; the geotransform (origin, pixel size); when each is used | labels vs imagery |
| raster-bands-resolution | Bands, resolution & the pixel footprint | spatial, spectral, temporal and radiometric resolution; multispectral vs hyperspectral; bit depth; ground sample distance | sensor choice |
| earth-observation-sensors | Earth-observation sensors | optical (Sentinel-2, Landsat), radar (SAR, Sentinel-1), elevation (DEM/LiDAR); revisit time; cloud cover; what each sees | data selection |

## Module 1 · Geospatial data formats & cloud-native access  (`geo-formats.html`)
*Animation focus: a large GeoTIFF split into internal tiles and overviews; HTTP range requests fetching only the needed bytes.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| geotiff-cog | GeoTIFF & Cloud-Optimised GeoTIFF | tags and georeferencing inside a TIFF; internal tiling; overviews (pyramids); why COG enables partial reads over HTTP | nearly all raster archives |
| vector-formats | Vector formats: GeoJSON, GeoPackage, Shapefile | geometry types; attributes; coordinate order (lon, lat); validity of polygons; file size and limits | labels, boundaries |
| array-formats-zarr | Multidimensional arrays: NetCDF & Zarr | time dimension; chunking; lazy loading; datacube thinking | climate and time series |
| range-requests | Partial reads & the cloud archive | HTTP byte ranges; reading one window of one band; cost and latency of full downloads | scalable pipelines |
| reproject-resample | Reprojection & resampling | nearest, bilinear, cubic and majority resampling; when each is correct (labels vs imagery); alignment of stacks | every multi-source project |
| masks-nodata | No-data, quality masks & cloud masking | nodata values; quality bands (e.g. cloud and shadow probability); applying masks correctly | clean training data |

## Module 2 · Finding data with STAC  (`geo-stac.html`)
*Animation focus: a catalogue tree opening into collections, items and assets; a search box filtering a map of footprints by place and time.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| stac-what | What STAC is and why it exists | a common language for spatiotemporal assets; discovery without downloading; the four entities | all modern EO archives |
| stac-catalog-collection | Catalogues and collections | catalog JSON links; collections with extents, licence and providers; hierarchical browsing; `rel` link types | navigating providers |
| stac-item-asset | Items and assets | an Item as a GeoJSON feature with bbox, geometry, datetime and properties; assets (hrefs, roles, media types); reading an Item in JSON | the atomic data unit |
| stac-extensions | STAC extensions (eo, proj, raster, …) | why extensions exist; `stac_extensions` list; common fields: `eo:bands`, `proj:epsg`, `raster:bands`; extending without breaking tools | interoperable metadata |
| stac-api-search | The STAC API & search | `/search` with bbox, datetime interval, collections, `query` and `filter` (CQL2); pagination; practical search recipes | programmatic data access |
| stac-cloud-harvest | From search results to a dataset | selecting assets by band; cloud-cover filtering; building a time series stack; keeping provenance | preparing training data |
| stac-validation | Validating and publishing STAC | JSON schemas; validators; required vs optional fields; creating a static catalogue for your own data | sharing datasets |

## Module 3 · Preparing geospatial data for machine learning  (`geo-preparation.html`)
*Animation focus: a scene cut into overlapping tiles and stitched back; labels rasterised onto the image grid; a spatial split as blocks of land.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| labels-rasterization | Labels: rasterising polygons | mapping vector labels onto the pixel grid; class maps; label noise and provenance | segmentation datasets |
| tiling-chipping | Tiling, chipping & overlap | tile size vs context vs memory; stride and overlap; discarding empty tiles; edge handling | every EO deep-learning pipeline |
| band-normalization | Band selection & normalisation | percentiles; per-band statistics computed on training data only; scaling reflectance (e.g. ÷10000); no-data handling | input consistency |
| spatial-splits | Spatial train/validation/test splits | spatial autocorrelation; why random pixel or tile splits leak; block and region splits; buffering | honest evaluation |
| augmentation-geo | Geometric augmentation | rotations and flips that preserve geography; band-aware colour augmentation; what breaks (north-up assumption) | robustness |
| dataset-manifests | Dataset cards & manifests | recording CRS, resolution, band order, dates, sources and licences per dataset; reproducible manifests | reproducibility |

## Module 4 · Spectral indices & classical GeoAI  (`geo-classical.html`)
*Animation focus: NDVI computed pixel by pixel from red and near-infrared; per-pixel classifiers colouring a land-cover map; zonal statistics aggregating over polygons.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| spectral-signatures | Spectral signatures | how vegetation, water, soil and urban surfaces reflect light differently; band combinations | why indices work |
| vegetation-indices | Vegetation & water indices | NDVI = (NIR−Red)/(NIR+Red), worked on a pixel; EVI, NDWI, NBR; thresholds and their seasonality | crop and forest monitoring |
| pixel-classifiers | Pixel-based classification | feature vectors per pixel (bands + indices + texture); random forests and gradient boosting for land cover; accuracy | land-cover maps |
| object-based | Object-based image analysis | segmenting into objects first (SLIC, watershed); object features; when objects beat pixels | high-resolution imagery |
| zonal-stats | Zonal statistics & aggregation | aggregating raster values over polygons (mean, mode, histogram); area-weighted summaries; scaling to many polygons | agriculture, insurance, policy |
| change-detection-classical | Change detection without deep learning | image differencing; post-classification comparison; change vs noise; thresholds | deforestation, floods |

## Module 5 · Deep learning for Earth observation  (`geo-deep.html`)
*Animation focus: a U-Net encoder shrinking a satellite tile and a decoder painting a mask back; sliding-window inference building a full-scene map.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| scene-classification | Scene & patch classification | CNN on chips; pretrained backbones on multispectral input (adapting the first conv); class imbalance | land-use datasets |
| segmentation-unet | Semantic segmentation for land cover | U-Net on tiles; per-pixel cross-entropy and Dice; shape table for a 256×256×12 input | land cover, buildings, roads |
| object-detection-eo | Object detection in satellite imagery | small objects; tiling vs resolution; ships, aircraft, vehicles; oriented boxes | defence, ports, mining |
| instance-buildings | Instance segmentation of buildings | footprint polygons; instance masks; vectorising predictions; polygon simplification | cadastre, disaster response |
| change-detection-deep | Deep change detection | siamese networks; before/after pairs; temporal consistency | damage assessment |
| sar-deep | Deep learning on SAR | speckle; log-scaled backscatter; cloud-independent mapping; fusing optical and radar | all-weather monitoring |
| inference-at-scale | Sliding-window inference & stitching | overlap blending; seam artefacts; memory limits; writing predictions as a COG | producing maps |

## Module 6 · STAC MLM: describing machine-learning models  (`geo-mlm.html`)
*Animation focus: a model asset and its metadata record flowing into a registry, then a runner reading the record to set up inference.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| mlm-purpose | Why describe models with metadata | discovering models like datasets; running a model without reading its code; reproducibility and trust | model catalogues |
| mlm-item-structure | The MLM item | an Item carrying `mlm:name`, `mlm:architecture`, `mlm:tasks`, `mlm:framework` and `mlm:framework_version`; `stac_extensions` listing the MLM schema; a worked item | publishing a model |
| mlm-input-output | Describing inputs & outputs | `mlm:input` (bands, patch shape, normalisation, resize), `mlm:output` (classes, shapes, post-processing); why the pipeline needs them | correct inference |
| mlm-assets-roles | Model assets & roles | `mlm:model` (required), `mlm:weights`, `mlm:source_code` (`code` role), runtimes and containers; artifact type; compile method and entrypoint | packaging models |
| mlm-hyperparameters-resources | Hyper-parameters, size & hardware | `mlm:total_parameters`, `mlm:memory_size`, `mlm:batch_size_suggestion`, `mlm:accelerator`, accelerator count and constraints; pretraining source | planning inference cost |
| mlm-tasks-taxonomy | Tasks taxonomy | classification, segmentation (semantic, instance, panoptic), detection, regression, similarity search, super-resolution; choosing the right task value | searchable models |
| mlm-collections-links | Collections, summaries & derived models | collection-level summaries (only simple fields); `derived_from` links from fine-tuned models to base models; provenance chains | model families |
| mlm-run-from-metadata | Running inference from the record | reading the MLM item to build a pipeline: fetch assets, normalise as stated, tile as stated, run, post-process as stated; a reference runner | automated inference |
| mlm-validation | Validating and trusting a model record | schema validation; checking claims (parameter counts, input shapes) against the artifact; reporting accuracy honestly | model governance |

## Module 7 · Foundation models & geospatial embeddings  (`geo-foundation.html`)
*Animation focus: masked patches in a satellite time series reconstructed; embeddings of tiles clustering on a sphere.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| pretraining-eo | Self-supervised pretraining on Earth imagery | masked image modelling; using unlabelled archives; multi-band and multi-temporal inputs | geospatial foundation models |
| temporal-models | Time series of imagery | temporal stacks; phenology; temporal attention; gap filling | crop type, change |
| embeddings-similarity | Embeddings & similarity search | tile embeddings; cosine similarity; nearest-neighbour search over a catalogue; finding look-alike places | search by example |
| finetuning-geo | Fine-tuning foundation models | freezing encoders; adapters; few-shot labels; learning-rate per layer | low-label projects |
| model-zoo-stac | Finding pretrained models via STAC MLM | searching a catalogue for a model by task, input bands and resolution; checking compatibility | choosing a model |

## Module 8 · Evaluation, accuracy & ground truth  (`geo-evaluation.html`)
*Animation focus: an accuracy assessment sampling random points, checking them against reference labels, filling a confusion matrix.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| sampling-accuracy | Sample-based accuracy assessment | stratified random sampling of map points; confusion matrix; overall, user's and producer's accuracy | land-cover products |
| area-adjusted | Area-adjusted accuracy | why raw counts mislead when classes differ in area; estimators with inclusion probabilities | official statistics |
| object-metrics | Object-level metrics | IoU per object, F1 per class, panoptic-style quality; matching predictions to reference polygons | building and ship detection |
| ground-truth-collection | Collecting ground truth | field surveys, crowdsourcing, VHR reference, label agreement; uncertainty in labels | label quality |
| error-maps | Error analysis & visual checks | error maps; spatial clustering of errors; error by region, season and sensor | debugging models |
| uncertainty-geo | Uncertainty & confidence in maps | calibrated probabilities per pixel; entropy maps; flagging uncertain areas for review | decision support |

## Module 9 · Deploying GeoAI responsibly at scale  (`geo-production.html`)
*Animation focus: a national-scale job splitting into tiles across workers, results merging into a map product, cost counters running.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| scale-processing | Processing at scale | tiling a country; parallel workers; cloud-native reads; cost per km²; checkpointing | national mapping |
| openeo-dask | Workflow tools: openEO, Dask & serverless | lazy evaluation; chunked arrays; running a workflow where the data lives | production pipelines |
| product-publishing | Publishing products as COG & STAC | output COGs; a STAC collection for the products; versioning; citation | open products |
| monitoring-drift-geo | Monitoring GeoAI systems | seasonal and sensor drift; new regions; re-validation schedules | operations |
| privacy-ethics-geo | Privacy, ethics & responsible use | precision of location data; sensitive areas; dual-use risks; community consent; open data licences | responsible mapping |
| geoai-applications | Applications & case studies | agriculture, disaster response, urban planning, climate, conservation: what each needs from the pipeline | choosing projects |

---
## Content standards (apply to every lesson)
- Intuition with a map or picture first; every symbol explained; hand-worked numbers on a tiny raster, table or catalogue.
- STAC lessons show real JSON fragments that are valid against the current specs; MLM lessons cite the field names from the MLM extension (v1.6.0 at the time of writing) and say so.
- At least one **animated, editable lab** per lesson.
- Be correct about projections, units and sensors; state assumptions (CRS, resolution, date).
- Animations: slow, dt-based, captioned, pausable (see curriculum/AUTHORING.md).
