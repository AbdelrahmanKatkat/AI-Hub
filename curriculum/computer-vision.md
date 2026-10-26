# Computer Vision — Curriculum (zero → hero)

Goal: a learner who knows only the idea of "a picture" ends able to explain and *draw from memory* how every major vision
architecture works (CNNs, ResNet, MobileNet, ViT, U-Net, YOLO/DETR, diffusion), to compute shapes/parameters/FLOPs by hand,
to design and train a vision model, and to deploy it as a full pipeline (preprocessing → inference → post-processing → storage).

**Two emphases:** (1) **Animation** — every concept is shown as data *moving*: pixels flowing through kernels, tensors changing shape
block by block, boxes being filtered, patches turning into tokens. (2) **Architecture** — every model family is taught as a blueprint:
blocks, tensor shapes at each stage, parameters, FLOPs, receptive field and *why* each design choice exists.

Order is strict: each lesson uses only earlier ideas (Mathematics module prerequisites: linear algebra basics, derivatives, probability).
Ten modules, 70 lessons. Module pages are `cv-<name>.html`; the hub is `computer-vision.html`.

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete visual problem that makes the idea necessary.
2. **Intuition** – plain words, analogy, mental picture. No formula first.
3. **Build it up** – step by step with a hand-worked numeric example (real pixel values / shapes / counts).
4. **Formal definition** – equations + a symbol table; architecture lessons also give a **shape table** (input → output per layer).
5. **Animated lab** – learner changes a kernel/size/stride/layer/threshold and sees data move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – named models and products.
8. **Check yourself** – 3 questions, answers hidden.
9. **Takeaways**.
Architecture lessons additionally end with a **"Draw it yourself" blueprint**: block diagram + shapes + parameter count.

## Shared animated labs (built once, reused by every module — see assets/labs-cv.js)
`pixels` (zoom into a pixel grid with numbers), `convolve` (animated sliding kernel, editable weights, step-by-step output build),
`archviz` (architecture animation: tensor blocks with shapes, a data pulse flowing through layers, click a layer for formula/params/FLOPs),
`archbuilder` (edit a layer list → shapes, parameters, FLOPs, receptive field update live, animated forward pass).

---
## Module 0 · Images as numbers  (`cv-images.html`)
*Animation focus: zooming from a photo down to the numbers; colour channels splitting and recombining.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| what-is-an-image | What is an image? Pixels & grids | grid of samples; resolution; sampling and quantization; bit depth; image as a function I(x,y) | every CV model input |
| channels-color | Channels & colour spaces | grayscale vs RGB; luminance weights; HSV/YUV/Lab; why colour spaces exist | colour jitter augmentation, ISP, video codecs |
| tensors-layout | Images as tensors | shape H×W×C vs C×H×W; batch dimension N; dtypes; value ranges; memory size of a batch | PyTorch NCHW, TensorFlow NHWC, GPU memory |
| histograms | Intensity histograms & statistics | histogram, mean/std, contrast; per-channel statistics; dataset mean/std | normalization, exposure analysis |
| cameras-formats | Capture & storage | sensor and Bayer mosaic; noise; gamma; lossy vs lossless; JPEG pipeline (blocks, DCT, quantization) | data quality, compression artefacts |

## Module 1 · Classical image processing  (`cv-processing.html`)
*Animation focus: kernels sliding over pixels; transforms morphing an image step by step.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| point-ops | Point operations | brightness, contrast, gamma, thresholding; histogram equalization; look-up tables | preprocessing, augmentation |
| convolution-filters | Convolution & filtering | sliding window; correlation vs convolution; box/Gaussian blur; sharpen; padding; separable kernels | the operation inside CNNs |
| edges | Gradients & edges | image gradient; Sobel; gradient magnitude/orientation; Canny (non-max suppression, hysteresis) | early CNN filters, lane detection |
| morphology | Morphology & components | erosion, dilation, opening, closing; structuring elements; connected components | mask clean-up, post-processing |
| resizing-interp | Resizing & interpolation | resampling; nearest/bilinear/bicubic; aliasing; Gaussian pyramids | input resizing, FPN, super-resolution |
| geometric-transforms | Geometric transforms | translation, rotation, scale, affine, homography; inverse warping | augmentation, document scanning, stitching |
| frequency-domain | The frequency view | Fourier transform of images; low/high-pass; why blur removes detail; link to JPEG | compression, texture bias, anti-aliasing |

## Module 2 · Features & geometry  (`cv-features.html`)
*Animation focus: keypoints being found and matched; RANSAC iterations; rays projecting 3-D points to pixels.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| corners-keypoints | Corners & keypoints | what makes a point distinctive; Harris corner response; scale-space idea | SLAM, tracking |
| descriptors-matching | Descriptors & matching | SIFT/ORB idea; descriptor distance; nearest neighbours; Lowe ratio test | image retrieval, AR |
| ransac-homography | RANSAC & homography | robust fitting with outliers; RANSAC loop; panorama stitching | stitching, 3-D reconstruction |
| camera-model | The pinhole camera | projection equations; intrinsics K; extrinsics R,t; lens distortion | calibration, AR, robotics |
| stereo-depth | Stereo & depth | epipolar geometry; disparity; depth = f·B/d | depth sensing, self-driving |
| optical-flow | Motion & optical flow | brightness constancy; Lucas–Kanade; dense flow | video models, stabilisation |

## Module 3 · Convolutional networks from first principles  (`cv-cnn.html`)
*Animation focus: a learned kernel sliding across channels building feature maps; shapes changing layer by layer.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| why-not-mlp | Why not just an MLP? | parameter explosion on images; locality; translation equivariance; weight sharing | motivation for all CNNs |
| conv-layer | The convolutional layer | learned filters; multi-channel input; C_in→C_out; bias; parameter count formula | every CNN |
| stride-padding | Stride, padding, dilation | output-size formula; same/valid; downsampling; dilation | architecture shape arithmetic |
| pooling-downsampling | Pooling & downsampling | max/avg pooling; global average pooling; strided conv; invariance | classifier heads |
| receptive-field | Receptive fields | how RF grows with depth/stride/kernels; effective RF; why depth matters | architecture design |
| activations-norm | Activations & normalisation | ReLU family; BatchNorm/LayerNorm/GroupNorm; dropout; where they go in a block | stable training |
| building-a-cnn | Building a CNN end to end | LeNet-style network; shape table; forward and backward; training loop; augmentation | first working classifier |
| visualizing-cnns | Looking inside a CNN | learned filters; activation maps; saliency; Grad-CAM | debugging, explainability |

## Module 4 · CNN architectures — the blueprints  (`cv-architectures.html`)
*Animation focus: the same input image flowing through each architecture, tensor blocks changing shape, skip connections lighting up.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| arch-design-space | How to read an architecture | stems, stages, blocks, heads; width/depth/resolution; params, FLOPs, activation memory accounting | reading any paper |
| lenet-alexnet-vgg | LeNet → AlexNet → VGG | lineage; ReLU+dropout+GPUs; why stacked 3×3 beat big kernels; VGG shape table | historical baseline, feature extractors |
| inception | Inception & multi-branch blocks | parallel branches; 1×1 bottleneck; parameter savings | GoogLeNet, multi-scale design |
| resnet | ResNet & residual connections | degradation problem; y=F(x)+x; gradient highway; basic vs bottleneck block; ResNet-18/50 shapes; pre-activation | the default backbone everywhere |
| densenet-dual-paths | DenseNet & feature reuse | concatenation vs addition; growth rate; transition layers | medical imaging, compact models |
| efficient-convs | Efficient convolutions | depthwise + pointwise; MobileNet; inverted residual; squeeze-and-excitation; ShuffleNet | mobile/edge vision |
| scaling-efficientnet | Scaling networks | compound scaling (depth, width, resolution); NAS idea; RegNet | EfficientNet, model families |
| convnext-modern | Modernising the ConvNet | ConvNeXt recipe: patchify stem, large kernels, LayerNorm, GELU, inverted bottleneck | CNNs competing with transformers |
| architecture-lab | Capstone: design under a budget | choose blocks to meet params/FLOPs/latency targets; compare designs side by side | model selection |

## Module 5 · Vision Transformers & modern backbones  (`cv-transformers.html`)
*Animation focus: an image being cut into patches, patches becoming tokens, attention weights drawn between patches.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| patch-embedding | From pixels to tokens | patchify; linear projection; token count; sequence length vs resolution | ViT, MAE |
| vit | The Vision Transformer | encoder blocks; class token; positional embeddings; MLP + MSA; shape table; params | ViT-B/16 etc. |
| attention-maps | What attention looks at | self-attention over patches; attention rollout; global vs local | interpretability |
| hierarchical-vit | Hierarchical transformers (Swin) | windowed attention; shifted windows; patch merging; pyramid features | detection/segmentation backbones |
| cnn-vs-vit | CNNs vs ViTs | inductive bias; data hunger; DeiT; hybrid models; compute scaling | choosing a backbone |
| self-supervised-vision | Learning without labels | contrastive (SimCLR/MoCo); masked autoencoders; DINO self-distillation | foundation models |
| clip-multimodal | Vision–language models | CLIP dual encoders; contrastive loss; zero-shot classification; image embeddings | search, VLMs, diffusion conditioning |

## Module 6 · Object detection  (`cv-detection.html`)
*Animation focus: boxes proposed, scored, suppressed; a grid of cells each predicting a box.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| boxes-iou | Boxes & IoU | box formats; intersection over union; GIoU/CIoU | all detectors, metrics |
| sliding-windows-rcnn | From sliding windows to Faster R-CNN | sliding window cost; R-CNN, Fast R-CNN, RPN; RoI pooling/align; two-stage pipeline | accurate detectors |
| anchors-heads | Anchors & detection heads | anchor boxes; offsets encoding; classification + regression heads; loss (focal loss) | RetinaNet, SSD |
| yolo-ssd | Single-shot detectors & FPN | grid cells; multi-scale heads; feature pyramids; YOLO family evolution; shape table | real-time detection |
| nms | Non-maximum suppression | greedy NMS algorithm; threshold effects; soft-NMS; class-wise NMS | post-processing of every detector |
| detr | DETR: detection as set prediction | object queries; Hungarian matching; bipartite loss; no NMS | transformer detectors |
| detection-metrics | Measuring detection | precision/recall curves; AP; mAP@[.5:.95]; confusion cases | benchmarking |

## Module 7 · Segmentation & dense prediction  (`cv-segmentation.html`)
*Animation focus: encoder shrinking the image, decoder growing it back, skip connections carrying detail across.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| semantic-fcn | Semantic segmentation & FCN | per-pixel classification; fully convolutional nets; coarse-to-fine | scene parsing |
| upsampling-decoders | Upsampling | nearest/bilinear + conv; transposed convolution & checkerboard artefacts; pixel shuffle | decoders, GANs, SR |
| unet | U-Net | encoder–decoder; skip connections by concatenation; shape table; why it works with little data | medical imaging, diffusion U-Nets |
| atrous-context | Dilated convolutions & context | atrous convolution; ASPP; DeepLab; context vs resolution | semantic segmentation |
| instance-panoptic | Instance & panoptic segmentation | Mask R-CNN mask head; panoptic quality; things vs stuff | autonomous driving |
| sam-foundation | Promptable segmentation (SAM) | image encoder + prompt encoder + mask decoder; promptable design; ambiguity | annotation, editing tools |
| depth-pose | Depth, keypoints & pose | monocular depth; heatmap keypoints; pose estimation | AR, robotics, sports |

## Module 8 · Generative & video vision  (`cv-generative.html`)
*Animation focus: noise turning into an image; generator and discriminator playing their game; frames flowing through time.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| autoencoders | Autoencoders | bottleneck; reconstruction loss; denoising and variational autoencoders | compression, latent spaces |
| gans | GANs | generator vs discriminator; minimax loss; training dynamics; mode collapse | image synthesis, style transfer |
| diffusion-images | Diffusion for images | forward noising; U-Net noise predictor; sampling; latent diffusion; classifier-free guidance | Stable Diffusion style models |
| image-translation-sr | Image-to-image & super-resolution | pix2pix, CycleGAN idea; SR networks; perceptual loss | restoration, enhancement |
| nerf-3d | 3-D from images | NeRF: rays, volume rendering; Gaussian splatting idea | 3-D capture, VR |
| video-models | Video models | 3-D convolutions; two-stream; temporal attention; video transformers; clips and frame sampling | action recognition |
| tracking | Tracking | tracking-by-detection; Kalman filter; data association; SORT/DeepSORT | surveillance, sports analytics |

## Module 9 · Training, deployment & the full pipeline  (`cv-production.html`)
*Animation focus: the whole pipeline end to end — camera → preprocessing → model → post-processing → storage — with real shapes and timings.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| data-labeling-augmentation | Data, labelling & augmentation | dataset design; label types; augmentation policies; class imbalance; label noise | every project |
| transfer-finetuning | Transfer learning & fine-tuning | pretrained backbones; freezing; learning-rate per layer; LoRA/adapters | most real projects |
| cv-losses-training | Losses & training recipes | losses per task; optimisers and schedules; mixed precision; batch size effects | training |
| cv-pipeline | The end-to-end vision pipeline | preprocessing → inference → post-processing → storage; batching; latency budget; shapes at every step | production systems |
| model-compression | Making models small & fast | quantization; pruning; distillation; speed/accuracy trade-off | edge deployment |
| deployment-edge | Deployment & hardware | ONNX/TensorRT; GPU vs CPU vs NPU; throughput vs latency; memory; serving | cloud and device |
| monitoring-ethics | Monitoring, bias & failure modes | drift; dataset bias; privacy; adversarial examples; safe deployment | responsible AI |

---
## Content standards (apply to every lesson)
- Intuition first; every symbol explained; at least one hand-worked numeric example (pixels, shapes, parameter counts).
- Architecture lessons include a **shape table**, **parameter count**, **FLOPs**, and **receptive field** where relevant, all verified numerically.
- At least one **animated, editable lab** per lesson; architecture lessons use `archviz`/`archbuilder` so the learner *sees* tensors flow.
- Correct and honest: cite the architecture's real published shapes (e.g. ResNet-50: 25.6 M parameters, ≈4.1 GFLOPs at 224²), state conditions.
- Animations: slow, dt-based, captioned, pausable (see curriculum/AUTHORING.md).
