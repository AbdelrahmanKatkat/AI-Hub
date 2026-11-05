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
    {
      id: 'ds', title: 'Data Science', icon: '📊', color: '#388bfd', status: 'live', href: 'data-science.html', curriculum: 'assets/curriculum-ds.js', next: 'cv',
      desc: 'From a messy table to a trusted decision: wrangling, exploration, statistics, experiments, features, forecasting and responsible communication.',
      tags: ['Wrangling', 'EDA', 'Statistics', 'A/B tests', 'Forecasting'],
      // lessons are generated from curriculum/data-science.md (assets/curriculum-ds.js)
    },
    {
      id: 'cv', title: 'Computer Vision', icon: '👁️', color: '#2f81f7', status: 'live', href: 'computer-vision.html', curriculum: 'assets/curriculum-cv.js', next: 'nlp',
      desc: 'Teach machines to see: from pixels and filters to CNNs, transformers, detection, segmentation and deployment — with animated architectures.',
      tags: ['Convolutions', 'ResNet', 'Vision Transformers', 'Detection', 'Segmentation'],
    },
    { id: 'nlp', title: 'NLP', icon: '💬', color: '#db6d28', status: 'soon', desc: 'Tokens, embeddings, attention and language understanding.' },
    { id: 'voice', title: 'Voice', icon: '🎙️', color: '#db61a2', status: 'soon', desc: 'Speech recognition, synthesis and real-time voice pipelines.' },
    {
      id: 'genai', title: 'Generative AI', icon: '✨', color: '#a371f7', status: 'live', href: 'generative-ai.html', curriculum: 'assets/curriculum-genai.js', next: 'gov',
      desc: 'How models create text, images and more: language modelling, the Transformer, training and alignment, efficient serving, retrieval and agents, diffusion and flow matching.',
      tags: ['Transformers', 'LLM training', 'RLHF & DPO', 'RAG & agents', 'Diffusion'],
      // lessons are generated from curriculum/generative-ai.md (assets/curriculum-genai.js)
    },
    { id: 'gov', title: 'Governing AI', icon: '⚖️', color: '#d29922', status: 'soon', desc: 'How AI is kept safe, fair and accountable: risk, ethics, regulation, auditing and responsible deployment.' },
    {
      id: 'rl', title: 'Reinforcement Learning', icon: '🎮', color: '#f85149', status: 'live', href: 'reinforcement-learning.html', curriculum: 'assets/curriculum-rl.js', next: 'domains',
      desc: 'Zero to hero: bandits, dynamic programming, Monte Carlo and temporal-difference learning, deep Q-learning, policy gradients and PPO, model-based planning, offline RL and RLHF.',
      tags: ['Bandits', 'Q-learning', 'Policy gradient', 'PPO', 'Model-based RL', 'RLHF'],
    },
    { id: 'domains', title: 'Domains', icon: '🌍', color: '#56d364', status: 'soon', children: [
        { title: 'Healthcare & Bio', icon: '🧬' }, { title: 'Climate & Earth', icon: '🌦️' }, { title: 'Finance', icon: '💹' }, { title: 'Other domains', icon: '➕' },
      ], desc: 'Applied AI, one page per field: GeoAI, Robotics, Healthcare, Climate, Finance and more.' },
    {
      id: 'geoai', title: 'GeoAI', icon: '🛰️', color: '#56d364', status: 'live', href: 'geoai.html', curriculum: 'assets/curriculum-geoai.js', sub: 'Domains', next: 'domains',
      desc: 'AI on Earth observation: find data with STAC, prepare rasters without leakage, train segmentation and detection models, and describe them with the STAC MLM extension.',
      tags: ['STAC', 'STAC MLM', 'Satellite imagery', 'Segmentation', 'Cloud-native formats'],
    },
    {
      id: 'robotics', title: 'Robotics', icon: '🤖', color: '#f0883e', status: 'live', href: 'robotics.html', curriculum: 'assets/curriculum-robotics.js', sub: 'Domains', next: 'domains',
      desc: 'Robots that sense, estimate, plan, control and learn: frames and kinematics, filters and SLAM, motion planning, PID and MPC, imitation and reinforcement learning, vision-language-action models and safe deployment.',
      tags: ['Kinematics', 'Kalman filter', 'SLAM', 'Motion planning', 'Control', 'Imitation learning', 'VLA'],
    },
  ],
};

/* Each page's pipeline: stages animate left→right. Edit text here, not in the lab code. */
window.PIPELINES = {
  rl: {
    stages: [
      { name: 'Interact', color: '#2f81f7', items: ['Observe state', 'Choose action', 'Receive reward'], out: 'transition (s, a, r, s′)', compute: 'Environment step; CPU or GPU batch simulation', storage: 'Episode buffer', desc: 'The agent acts in the environment at each step, observes the next state and receives a reward that may arrive long after the action.' },
      { name: 'Store', color: '#39c5cf', items: ['Replay buffer', 'Trajectories', 'Logged data'], out: 'mini-batches of transitions', compute: 'Memory and sampling', storage: 'Replay buffer, dataset', desc: 'Transitions are kept, either for one pass (on-policy) or for many reuses (replay), so that updates are based on a stream of experience.' },
      { name: 'Estimate', color: '#a371f7', items: ['TD error', 'Returns', 'Advantage'], out: 'value and advantage estimates', compute: 'GPU forward passes on value networks', storage: 'Value table or network weights', desc: 'The agent turns rewards into estimates of how good states and actions are, using bootstrapped targets or Monte Carlo returns.' },
      { name: 'Improve', color: '#db6d28', items: ['Greedy step', 'Gradient step', 'Target update'], out: 'updated policy', compute: 'Gradient steps (GPU) or table updates', storage: 'Policy parameters, target network', desc: 'The policy changes toward actions with higher estimated value, either greedily or by a gradient step, with safeguards that limit how far it moves.' },
      { name: 'Evaluate', color: '#db61a2', items: ['Learning curve', 'Seeds & CI', 'Held-out tasks'], out: 'return with confidence band', compute: 'Batch evaluation runs', storage: 'Experiment logs', desc: 'Returns are measured over several seeds and held-out environments, so that a single lucky run is not reported as progress.' },
      { name: 'Deploy & monitor', color: '#3fb950', items: ['Safe rollout', 'Reward monitor', 'Rollback'], out: 'policy in production', compute: 'Serving and monitoring', storage: 'Model registry, telemetry', desc: 'The policy is rolled out gradually with reward and safety monitors, and rolled back when the live reward or constraint checks fail.' },
    ],
  },
  robotics: {
    stages: [
      { name: 'Sense', color: '#2f81f7', items: ['Encoders & IMU', 'Camera & LiDAR', 'Timestamps'], out: 'synchronised sensor frames', compute: 'Real-time drivers, microcontroller and CPU', storage: 'Rosbag / log files', desc: 'Joint encoders, inertial and range sensors are read at fixed rates and time-stamped so that every later step uses the same moment.' },
      { name: 'Estimate', color: '#39c5cf', items: ['Filter / SLAM', 'Point registration', 'Pose'], out: 'state & map estimate', compute: 'CPU, small matrix maths at control rate', storage: 'Map, pose graph', desc: 'A filter fuses noisy sensors into one belief about the robot pose and the surroundings, with an uncertainty for every estimate.' },
      { name: 'Plan', color: '#a371f7', items: ['Collision check', 'Search / sampling', 'Smooth path'], out: 'collision-free trajectory', compute: 'CPU planning, seconds or less', storage: 'Planned trajectory', desc: 'A planner finds a path through free space in configuration or task space, then time-scales it into a smooth trajectory within speed and acceleration limits.' },
      { name: 'Control', color: '#db6d28', items: ['Tracking error', 'PID / MPC / policy', 'Torque limits'], out: 'motor commands', compute: 'Hard real-time loop, 100 Hz to 1 kHz', storage: 'Controller gains, logs', desc: 'A controller compares the measured state to the setpoint and commands torques or velocities, bounded by the actuators and the safety limits.' },
      { name: 'Learn', color: '#db61a2', items: ['Demonstrations', 'Policy network', 'Simulation'], out: 'action chunk', compute: 'GPU training, edge inference', storage: 'Datasets, policy weights', desc: 'Learned policies map observations to actions, trained from demonstrations or simulated experience and then run on the robot.' },
      { name: 'Deploy & monitor', color: '#3fb950', items: ['Safety monitor', 'Telemetry', 'Fleet updates'], out: 'running fleet', compute: 'Onboard watchdogs, cloud analytics', storage: 'Telemetry store, model registry', desc: 'Safety monitors stop the robot when something is wrong; telemetry and logs feed incident review, calibration and updates.' },
    ],
  },
  geoai: {
    stages: [
      { name: 'Acquire', color: '#2f81f7', items: ['STAC search', 'Select assets', 'Read windows'], out: 'bands × H × W', compute: 'I/O: range reads from cloud archive', storage: 'STAC catalogue (metadata), COG archive', desc: 'Find imagery by place, time and cloud cover with STAC, then read only the windows and bands you need from cloud-optimised files.' },
      { name: 'Preprocessing', color: '#39c5cf', items: ['Reproject', 'Mask clouds', 'Normalise', 'Tile'], out: 'tiles 256×256×C', compute: 'CPU/GPU array ops', storage: 'Chip store, dataset manifest', desc: 'Align everything to one CRS and resolution, mask clouds and no-data, normalise with training statistics and cut overlapping tiles.' },
      { name: 'Model', color: '#a371f7', items: ['Backbone', 'Segmentation head', 'Loss'], out: 'per-pixel logits', compute: 'GPU training and inference', storage: 'Weights + STAC MLM record', desc: 'A CNN or transformer maps each tile to a class per pixel or to objects; the model is described by its STAC MLM record.' },
      { name: 'Inference', color: '#db6d28', items: ['Sliding window', 'Blend overlaps'], out: 'scene-scale probability map', compute: 'GPU batches over the scene', storage: 'Probability COG', desc: 'Predict tile by tile with overlap, blend the borders so there are no seams, and write a probability raster.' },
      { name: 'Post-processing', color: '#db61a2', items: ['Threshold', 'Vectorise', 'Filter small areas'], out: 'polygons & class map', compute: 'CPU, vector geometry', storage: 'GeoPackage / GeoJSON', desc: 'Turn probabilities into classes, convert them to polygons, remove slivers and check topology before the product leaves the pipeline.' },
      { name: 'Publish & monitor', color: '#3fb950', items: ['Product COG', 'STAC collection', 'Accuracy checks'], out: 'published product', compute: 'Batch jobs', storage: 'Open data archive, STAC catalogue', desc: 'Publish the map as a versioned product with its own STAC record, and re-check accuracy as seasons and sensors change.' },
    ],
  },
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
  ds: {
    stages: [
      { name: 'Ask & collect', color: '#2f81f7', items: ['Define the question', 'Find data sources', 'Sample or extract'], out: 'raw dataset', compute: 'I/O: queries, exports, APIs', storage: 'Data warehouse, files, versioned extracts', desc: 'A decision defines the question, and the question defines which data to collect and how it was sampled.' },
      { name: 'Clean & prepare', color: '#39c5cf', items: ['Types & units', 'Missing values', 'Joins & duplicates'], out: 'tidy table', compute: 'CPU, dataframes', storage: 'Cleaned tables with a log of every change', desc: 'Raw data is rarely usable: parse types, fix errors, handle missing cells and join sources, checking row counts at every step.' },
      { name: 'Explore', color: '#a371f7', items: ['Summaries', 'Plots', 'Relationships'], out: 'what the data says', compute: 'CPU, interactive notebooks', storage: 'Notebooks and figures', desc: 'Summaries and plots reveal shape, outliers and relationships, and decide what is worth testing or modelling.' },
      { name: 'Test & model', color: '#db6d28', items: ['Hypothesis tests', 'Experiments', 'Predictive models'], out: 'estimates with uncertainty', compute: 'CPU, sometimes GPU', storage: 'Experiment logs, model files', desc: 'Statistics and experiments say how sure we can be; models predict. Every result carries its uncertainty.' },
      { name: 'Communicate', color: '#db61a2', items: ['Honest charts', 'Dashboards', 'Written report'], out: 'decision-ready story', compute: 'BI tools, notebooks', storage: 'Reports, dashboards', desc: 'A finding only matters if the audience understands it: one clear chart, a headline, and the limits stated.' },
      { name: 'Deploy & monitor', color: '#3fb950', items: ['Reproducible pipeline', 'Drift checks', 'Fairness checks'], out: 'trusted, repeatable results', compute: 'Scheduled jobs, monitoring', storage: 'Feature store, metrics store', desc: 'Pipelines run again on new data; monitoring catches drift and unfair outcomes before decisions go wrong.' },
    ],
  },
  genai: {
    stages: [
      { name: 'Collect & tokenise', color: '#2f81f7', items: ['Crawl & filter', 'Deduplicate', 'Tokenise or encode'], out: 'token sequences', compute: 'CPU clusters, heavy I/O', storage: 'Sharded token datasets', desc: 'Raw text, images or audio are filtered, deduplicated and turned into the tokens, patches or latents the model reads.' },
      { name: 'Pretrain', color: '#39c5cf', items: ['Next-token or denoising loss', 'AdamW', 'Parallel training'], out: 'base model weights', compute: 'Large GPU or TPU clusters', storage: 'Checkpoints: weights and optimiser state', desc: 'The model learns to predict the next token, or to remove noise, over a huge dataset. This is the most expensive stage.' },
      { name: 'Align', color: '#a371f7', items: ['Instruction tuning', 'Preference tuning (RLHF, DPO)', 'Safety training'], out: 'assistant model', compute: 'GPU, far fewer tokens than pretraining', storage: 'Tuned weights, preference datasets', desc: 'The base model is taught to follow instructions, prefer helpful and safe answers, and refuse harmful requests.' },
      { name: 'Adapt & compress', color: '#db6d28', items: ['LoRA adapters', 'Quantisation', 'Distillation'], out: 'smaller, cheaper model', compute: 'One GPU up to a small cluster', storage: 'Adapters, quantised weights', desc: 'The model is specialised cheaply and shrunk so that it fits the memory and latency budget of the product.' },
      { name: 'Generate', color: '#db61a2', items: ['Prompt and retrieved context', 'KV-cached decoding or denoising steps', 'Sampling'], out: 'text, image or audio', compute: 'GPU inference; decoding is memory-bound', storage: 'KV cache, vector index', desc: 'The prompt and any retrieved context go in; tokens are sampled one at a time, or an image is denoised over several steps.' },
      { name: 'Evaluate & guard', color: '#3fb950', items: ['Evals and human preference', 'Guardrails', 'Monitoring'], out: 'trusted, monitored outputs', compute: 'Evaluation jobs, filters', storage: 'Eval sets, logs', desc: 'Outputs are tested, filtered and monitored for errors, hallucination and misuse, and the results feed the next training round.' },
    ],
  },
};
