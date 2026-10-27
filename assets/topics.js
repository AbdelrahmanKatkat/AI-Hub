/* Topic registry + pipeline definitions. Add a topic here and it appears in the sidebar, search and home page. */
window.HUB = {
  repo: 'https://github.com/AbdelrahmanKatkat/AI-Hub',
  topics: [
    {
      id: 'math', title: 'Mathematics for AI', icon: '∑', color: '#39c5cf', status: 'live', href: 'mathematics.html', curriculum: 'assets/curriculum-math.js', next: 'ml',
      desc: 'Linear algebra, calculus, optimization, backpropagation and probability — animated.',
      tags: ['Matrices', 'Derivatives', 'Optimizers', 'Backprop', 'Probability'],
      // sections/modules are generated from curriculum/mathematics.md (see assets/curriculum-math.js)
    },
    {
      id: 'ml', title: 'Machine Learning', icon: '🧠', color: '#3fb950', status: 'live', href: 'machine-learning.html', curriculum: 'assets/curriculum-ml.js', next: 'ds',
      desc: 'Zero to hero: data preparation, classical algorithms, ensembles, neural networks, evaluation and production systems — every algorithm animated.',
      tags: ['Data preparation', 'Trees & boosting', 'Neural networks', 'Evaluation', 'MLOps'],
      // lessons are generated from curriculum/machine-learning.md (assets/curriculum-ml.js)
    },
    { id: 'ds', title: 'Data Science', icon: '📊', color: '#388bfd', status: 'soon', desc: 'Exploration, statistics, experimentation and storytelling with data.' },
    {
      id: 'cv', title: 'Computer Vision', icon: '👁️', color: '#2f81f7', status: 'live', href: 'computer-vision.html', curriculum: 'assets/curriculum-cv.js', next: 'nlp',
      desc: 'Teach machines to see: from pixels and filters to CNNs, transformers, detection, segmentation and deployment — with animated architectures.',
      tags: ['Convolutions', 'ResNet', 'Vision Transformers', 'Detection', 'Segmentation'],
    },
    { id: 'nlp', title: 'NLP', icon: '💬', color: '#db6d28', status: 'soon', desc: 'Tokens, embeddings, attention and language understanding.' },
    { id: 'voice', title: 'Voice', icon: '🎙️', color: '#db61a2', status: 'soon', desc: 'Speech recognition, synthesis and real-time voice pipelines.' },
    { id: 'genai', title: 'Generative AI', icon: '✨', color: '#a371f7', status: 'soon', desc: 'Transformers, diffusion, and how models create text, images and code.' },
    { id: 'gov', title: 'Governing AI', icon: '⚖️', color: '#d29922', status: 'soon', desc: 'How AI is kept safe, fair and accountable: risk, ethics, regulation, auditing and responsible deployment.' },
    { id: 'rl', title: 'Reinforcement Learning', icon: '🎮', color: '#f85149', status: 'soon', desc: 'Agents, rewards, policies and learning by interaction.' },
    { id: 'domains', title: 'Domains', icon: '🌍', color: '#56d364', status: 'soon', children: [
        { title: 'GeoAI', icon: '🛰️' }, { title: 'Robotics', icon: '🤖' }, { title: 'Healthcare & Bio', icon: '🧬' }, { title: 'Climate & Earth', icon: '🌦️' }, { title: 'Finance', icon: '💹' }, { title: 'Other domains', icon: '➕' },
      ], desc: 'Applied AI, one page per field: GeoAI, Robotics, Healthcare, Climate, Finance and more.' },
  ],
};

/* Each page's pipeline: stages animate left→right. Edit text here, not in the lab code. */
window.PIPELINES = {
  cv: {
    stages: [
      { name: 'Capture', color: '#2f81f7', items: ['Camera / file', 'Decode', 'Frames'], out: 'H×W×3 uint8', compute: 'Sensor + ISP; CPU decode', storage: 'Object store, video segments', desc: 'Light becomes a grid of numbers. Resolution, bit depth and compression decide how much information survives.' },
      { name: 'Preprocessing', color: '#39c5cf', items: ['Resize / crop', 'Normalise', 'HWC → CHW'], out: '1×3×224×224 float', compute: 'CPU or GPU kernels', storage: 'Cached tensors, TFRecords', desc: 'Resize to the network input, scale values (mean/std) and arrange the tensor layout. Must match training exactly.' },
      { name: 'Backbone', color: '#a371f7', items: ['Conv / attention blocks', 'Downsample', 'Feature maps'], out: 'C×H/32×W/32 features', compute: 'GPU/NPU: conv + matmul', storage: 'Weights (≈100 MB), activations in VRAM', desc: 'Stacked blocks turn pixels into features: edges, textures, parts, objects — shrinking space and growing channels.' },
      { name: 'Head', color: '#db6d28', items: ['Classify / detect / segment', 'Decode outputs'], out: 'logits · boxes · masks', compute: 'GPU, light', storage: 'Task weights', desc: 'A small task-specific head converts features into class scores, box offsets or per-pixel labels.' },
      { name: 'Post-processing', color: '#db61a2', items: ['Softmax / threshold', 'NMS', 'Resize masks'], out: 'final detections', compute: 'CPU, cheap', storage: 'Config, class names', desc: 'Raw numbers become decisions: suppress duplicate boxes, apply confidence thresholds, map coordinates back to the original image.' },
      { name: 'Storage & monitoring', color: '#3fb950', items: ['Log results', 'Sample hard cases', 'Drift'], out: 'labelled data', compute: 'Stream / batch', storage: 'Warehouse, review queue', desc: 'Store predictions with images, watch confidence and drift, and send hard cases back for labelling.' },
    ],
  },
  hub: {
    stages: [
      { name: 'Data', color: '#2f81f7', items: ['Collect', 'Label'], out: 'raw rows', compute: 'I/O bound, CPU', storage: 'Object store / data lake', desc: 'Everything starts as messy real-world data: files, sensors, logs, images.' },
      { name: 'Preprocessing', color: '#39c5cf', items: ['Clean', 'Scale', 'Encode'], out: 'tensor X', compute: 'CPU, vectorised', storage: 'Parquet / feature store', desc: 'Turn raw data into clean numeric tensors the model can digest.' },
      { name: 'Computation', color: '#a371f7', items: ['Forward', 'Loss', 'Backward'], out: 'weights θ', compute: 'GPU / TPU', storage: 'Checkpoints, metrics DB', desc: 'Training: repeatedly adjust parameters to reduce the loss.' },
      { name: 'Inference', color: '#db6d28', items: ['Load model', 'Predict'], out: 'scores ŷ', compute: 'CPU / GPU / edge', storage: 'Model registry, cache', desc: 'Apply the trained model to new, unseen inputs.' },
      { name: 'Post-processing', color: '#db61a2', items: ['Threshold', 'Rank', 'Explain'], out: 'decisions', compute: 'CPU, cheap', storage: 'Business rules', desc: 'Convert raw scores into decisions people and systems can act on.' },
      { name: 'Storage', color: '#3fb950', items: ['Log', 'Monitor', 'Feedback'], out: 'new data', compute: 'Stream / batch jobs', storage: 'Warehouse, logs', desc: 'Predictions and outcomes are stored, monitored and fed back as new training data.' },
    ],
  },
  ml: {
    stages: [
      { name: 'Data collection', color: '#2f81f7', items: ['Sources & APIs', 'Labels', 'Versioning'], out: 'raw rows', compute: 'I/O bound; CPU', storage: 'Object store (S3/GCS), data lake, DVC', desc: 'Gather examples (x, y). Quality and coverage here cap the quality of everything after.' },
      { name: 'Preprocessing', color: '#39c5cf', items: ['Clean & impute', 'Encode', 'Scale'], out: 'tensor X [N×d]', compute: 'CPU, vectorised (NumPy/Spark)', storage: 'Parquet files, feature store', desc: 'Split first, then fit transforms on the training split only (to avoid leakage) and apply them everywhere.' },
      { name: 'Training', color: '#a371f7', items: ['Forward pass', 'Loss', 'Backprop + update'], out: 'weights θ', compute: 'GPU/TPU, batched matrix math', storage: 'Checkpoints, experiment tracker', desc: 'Minimise a loss with gradient-based updates: θ ← θ − η∇L.' },
      { name: 'Inference', color: '#db6d28', items: ['Load model', 'Batch / stream', 'Forward only'], out: 'scores ŷ', compute: 'CPU, GPU or edge; latency-critical', storage: 'Model registry, warm cache', desc: 'Only the forward pass runs. Cheaper than training, but latency and cost per request matter.' },
      { name: 'Post-processing', color: '#db61a2', items: ['Threshold', 'Calibrate', 'Rank / explain'], out: 'decisions', compute: 'CPU, negligible', storage: 'Config & business rules', desc: 'Scores become decisions. The threshold trades precision against recall.' },
      { name: 'Storage & monitoring', color: '#3fb950', items: ['Log I/O', 'Drift checks', 'Feedback'], out: 'new labelled data', compute: 'Stream / batch jobs', storage: 'Warehouse, logs, metrics store', desc: 'Log inputs and outcomes, watch for drift, and loop fresh data back to step one.' },
    ],
  },
  math: {
    stages: [
      { name: 'Data → numbers', color: '#2f81f7', items: ['Vectors', 'Matrices', 'Tensors'], out: 'x ∈ ℝᵈ', compute: 'Linear algebra', storage: 'Arrays in memory', desc: 'Every input becomes a vector. A batch is a matrix; images and sequences are tensors.' },
      { name: 'Model f(x; θ)', color: '#a371f7', items: ['Wx + b', 'Nonlinearity', 'Compose'], out: 'ŷ = f(x)', compute: 'Matrix multiplication (GPU)', storage: 'Parameters θ', desc: 'Layers are matrix transforms followed by nonlinear functions, composed together.' },
      { name: 'Loss', color: '#db6d28', items: ['Distance', 'Likelihood', 'Cross-entropy'], out: 'L(θ) ∈ ℝ', compute: 'Probability & statistics', storage: 'Scalar per batch', desc: 'One number says how wrong the model is. Minimising it equals maximising likelihood.' },
      { name: 'Gradient', color: '#db61a2', items: ['Derivatives', 'Chain rule', 'Backprop'], out: '∇θ L', compute: 'Calculus / autodiff', storage: 'Activations kept for backward', desc: 'The gradient points uphill. Backpropagation computes it efficiently with the chain rule.' },
      { name: 'Update', color: '#39c5cf', items: ['SGD', 'Momentum', 'Adam'], out: 'θ ← θ − η·g', compute: 'Optimization', storage: 'Optimizer state (m, v)', desc: 'Step downhill. Optimizers differ in how they use gradient history to move faster and steadier.' },
      { name: 'Evaluate', color: '#3fb950', items: ['Held-out data', 'Metrics', 'Confidence'], out: 'score ± error', compute: 'Statistics', storage: 'Metrics logs', desc: 'Measure on unseen data and quantify uncertainty before trusting the result.' },
    ],
  },
};
