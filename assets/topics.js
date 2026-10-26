/* Topic registry + pipeline definitions. Add a topic here and it appears in the sidebar, search and home page. */
window.HUB = {
  repo: 'https://github.com/', // TODO: set to the real repository URL
  topics: [
    { id: 'cv', title: 'Computer Vision', icon: '👁️', color: '#2f81f7', status: 'soon', desc: 'Teach machines to see: pixels, convolutions, detection, segmentation.' },
    { id: 'genai', title: 'Generative AI', icon: '✨', color: '#a371f7', status: 'soon', desc: 'Transformers, diffusion, and how models create text, images and code.' },
    { id: 'nlp', title: 'NLP', icon: '💬', color: '#db6d28', status: 'soon', desc: 'Tokens, embeddings, attention and language understanding.' },
    { id: 'voice', title: 'Voice AI Engineer', icon: '🎙️', color: '#db61a2', status: 'soon', desc: 'Speech recognition, synthesis and real-time voice pipelines.' },
    {
      id: 'ml', title: 'Machine Learning', icon: '🧠', color: '#3fb950', status: 'live', href: 'machine-learning.html', next: 'math',
      desc: 'From data and preprocessing to training, neural networks, evaluation and production.',
      tags: ['Preprocessing', 'Regression', 'Gradient descent', 'Neural networks', 'Evaluation'],
      sections: [
        { id: 'intro', t: 'What is ML & the pipeline', k: 'pipeline supervised unsupervised' },
        { id: 'data', t: 'Data & preprocessing', k: 'normalize standardize scaling log transform outliers leakage' },
        { id: 'training', t: 'Models & training', k: 'linear regression mse loss gradient descent learning rate' },
        { id: 'neural', t: 'Neural networks', k: 'mlp activation backpropagation decision boundary playground' },
        { id: 'evaluation', t: 'Evaluation & post-processing', k: 'threshold precision recall roc f1 confusion matrix' },
        { id: 'production', t: 'Computation & storage', k: 'gpu inference quantization feature store registry drift monitoring' },
      ],
    },
    { id: 'ds', title: 'Data Science', icon: '📊', color: '#388bfd', status: 'soon', desc: 'Exploration, statistics, experimentation and storytelling with data.' },
    {
      id: 'math', title: 'Mathematics for AI', icon: '∑', color: '#39c5cf', status: 'live', href: 'mathematics.html', next: 'ml',
      desc: 'Linear algebra, calculus, optimization, backpropagation and probability — animated.',
      tags: ['Matrices', 'Derivatives', 'Optimizers', 'Backprop', 'Probability'],
      sections: [
        { id: 'intro', t: 'Why math & the pipeline', k: 'vectors tensors loss gradient' },
        { id: 'linear-algebra', t: 'Linear algebra', k: 'matrix vector transform determinant eigenvalue eigenvector' },
        { id: 'calculus', t: 'Derivatives & gradients', k: 'derivative slope tangent secant limit autodiff' },
        { id: 'optimization', t: 'Optimization', k: 'gradient descent momentum adam saddle local minimum learning rate' },
        { id: 'backprop', t: 'Chain rule & backprop', k: 'computational graph chain rule reverse mode' },
        { id: 'probability', t: 'Probability & statistics', k: 'distribution sampling central limit theorem gaussian histogram' },
      ],
    },
    { id: 'gov', title: 'Governing AI', icon: '⚖️', color: '#d29922', status: 'soon', desc: 'Safety, ethics, regulation and responsible deployment.' },
    { id: 'rl', title: 'Reinforcement Learning', icon: '🎮', color: '#f85149', status: 'soon', desc: 'Agents, rewards, policies and learning by interaction.' },
    { id: 'domains', title: 'Domains (GeoAI, Robotics…)', icon: '🌍', color: '#56d364', status: 'soon', desc: 'AI applied to geospatial data, robotics and other fields.' },
  ],
};

/* Each page's pipeline: stages animate left→right. Edit text here, not in the lab code. */
window.PIPELINES = {
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
