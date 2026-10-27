# Machine Learning — Curriculum (zero → hero)

Goal: a learner who has never trained a model ends able to take a real problem from raw data to a monitored production system:
choose and justify an algorithm, prepare the data without leakage, train and tune it, evaluate it honestly, understand *why* it works
(the maths behind it, derived and animated), and design the system around it (pipeline, compute, storage, serving, monitoring).

**Emphases:** (1) **The pipeline** — every algorithm is placed in the real flow *preprocessing → computation (training) → inference →
post-processing → storage/monitoring* with its compute and storage costs. (2) **Animation** — learners watch models *learn*: boundaries
bending, trees splitting, clusters moving, boosting adding trees, loss curves falling, drift appearing. (3) **Architecture** — both model
architectures (trees, ensembles, neural nets with shape/parameter tables) and **system architecture** (feature store, registry, serving topologies).

Order is strict: each lesson only uses earlier ideas; maths is *linked*, not re-taught (Mathematics for AI modules are the reference:
linear algebra, calculus, probability, statistics, optimization). Ten modules, 63 lessons. Module pages are `ml-<name>.html`; hub `machine-learning.html`.

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete practical problem that makes the idea necessary.
2. **Intuition** – plain words, analogy, picture. No formula first.
3. **Build it up** – step by step with a hand-worked numeric example on a tiny dataset (real numbers).
4. **Formal definition** – the algorithm/equations with a symbol table; a **pseudo-code box** for algorithms; for models, **what is stored** (parameters) and **what is computed** (training vs inference cost).
5. **Animated lab** – the learner changes data / hyper-parameters / an equation and watches the model learn; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real systems and when to choose it (and when not to).
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.
Algorithm lessons add a **"When to use / strengths / weaknesses"** table and the **pipeline view**: preprocessing it needs, training complexity, inference cost, what is stored.

## Shared tooling (see assets/mllib.js and assets/labs-ml.js)
`ML.data(kind, n, seed)` datasets (moons, circles, blobs, xor, spiral, linear/sine regression), `ML.boundary(...)` decision-boundary heat-map renderer,
reference implementations (k-NN, decision tree, k-means, logistic regression, linear regression, PCA-2D) and existing labs `preprocess`, `regression`, `nn`, `threshold`, `pipeline`.

---
## Module 0 · What machine learning is  (`ml-foundations.html`)
*Animation focus: data flowing through the six-stage pipeline; a model improving as data grows.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| what-is-ml | What is machine learning? | rules vs learning from examples; a model as a function f(x;θ); learning = choosing θ; when ML is (not) the right tool | every ML product |
| types-of-learning | Types of learning & tasks | supervised, unsupervised, self-supervised, reinforcement; classification, regression, clustering, ranking, generation | choosing a problem framing |
| ml-pipeline | The ML pipeline end to end | data → preprocessing → training → inference → post-processing → storage & monitoring; compute and storage at each stage; offline vs online | all production systems |
| data-and-features | Data, features & labels | tabular data as matrix X and vector y; feature types; labels and their cost; dataset bias; data quality | every project |
| generalization-basics | Generalisation: train, validation, test | why we split; overfitting and underfitting; learning curves; the test set is used once | model evaluation |

## Module 1 · Data preparation & feature engineering  (`ml-data.html`)
*Animation focus: distributions morphing under transforms; imputation filling gaps; leakage inflating scores.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| eda | Exploring data | distributions, summary statistics, correlations, missing-value patterns, outliers; plots that answer questions | first hour of any project |
| cleaning-missing | Cleaning & missing values | types of missingness (MCAR/MAR/MNAR); imputation (mean, median, model-based); outliers; duplicates; sanity checks | real-world messy data |
| encoding | Encoding categorical & other features | one-hot, ordinal, target/frequency encoding, hashing; dates, text counts; high-cardinality problems | tabular ML |
| scaling-transform | Scaling & transformations | standardisation, min-max, robust scaling, log/power transforms; which models need them; fit on train only | distance- and gradient-based models |
| feature-engineering | Feature engineering & selection | interactions, polynomial and binned features, domain features; filter/wrapper/embedded selection; leakage via features | Kaggle-style wins |
| imbalance-sampling | Imbalanced data | accuracy paradox; class weights; over/under-sampling; SMOTE idea; threshold moving; proper metrics | fraud, medical, defects |
| data-splitting-pipelines | Splitting strategies & pipelines | stratified, group and time-based splits; fit-transform-apply discipline; sklearn-style pipelines; proving no leakage | reproducible workflows |

## Module 2 · Linear models  (`ml-linear.html`)
*Animation focus: a line/plane fitting data step by step; a sigmoid boundary rotating; regularisation shrinking weights.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| linear-regression | Linear regression | model ŷ=w·x+b; squared loss; normal equation and gradient descent derived; interpretation; assumptions; R² | baseline for everything |
| regularization | Polynomials, overfitting & regularisation | polynomial features; variance explosion; ridge (L2), lasso (L1), elastic net; choosing λ; sparsity | high-dimensional data |
| logistic-regression | Logistic regression | log-odds; sigmoid; likelihood and cross-entropy derived; decision boundary; coefficients as odds ratios | classification baseline |
| softmax-multiclass | Multi-class classification | softmax regression; one-vs-rest vs multinomial; cross-entropy gradient p−y | the output layer of neural nets |
| linear-optimization | Training linear models in practice | batch/stochastic/mini-batch gradient descent; learning rate; effect of feature scaling; convergence checks; solver choice | scalable training |

## Module 3 · Classical algorithms  (`ml-classical.html`)
*Animation focus: neighbours voting; tree splitting the plane; margins and support vectors; centroids moving; data projected to fewer dimensions.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| knn | k-nearest neighbours | lazy learning; distance metrics; choice of k and bias–variance; scaling; curse of dimensionality; inference cost O(n) | recommendation, retrieval |
| naive-bayes | Naive Bayes | Bayes rule classifier; conditional independence; Gaussian/multinomial variants; Laplace smoothing; text classification | spam filters |
| decision-trees | Decision trees | recursive partitioning; Gini/entropy/variance impurity and information gain by hand; stopping and pruning; regression trees; interpretability | base of ensembles |
| svm | Support vector machines | margin maximisation; support vectors; hinge loss and soft margin; kernels (RBF/poly); scaling to data size | small/medium data, text |
| clustering | Clustering | k-means (Lloyd's algorithm, init, choosing k); hierarchical; DBSCAN; Gaussian mixtures and EM; silhouette | segmentation |
| dimensionality-reduction | Dimensionality reduction | PCA in practice (variance explained, scree); t-SNE/UMAP intuition and pitfalls; when to reduce | visualisation, compression |
| anomaly-detection | Anomaly detection | statistical, distance, isolation forest, one-class SVM, reconstruction error; evaluating without labels | fraud, monitoring |

## Module 4 · Ensembles & boosting  (`ml-ensembles.html`)
*Animation focus: many weak models voting; trees added one at a time correcting residuals.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| bagging | Why ensembles work & bagging | variance reduction by averaging (derived); bootstrap sampling; out-of-bag error | foundation of forests |
| random-forest | Random forests | bagging + feature subsampling; decorrelation; OOB; feature importance; hyper-parameters | strong tabular default |
| gradient-boosting | Gradient boosting | fitting residuals; functional gradient descent; shrinkage; loss choices; AdaBoost link | the tabular state of the art |
| xgboost-lightgbm | XGBoost, LightGBM & CatBoost | second-order objective, regularised trees, histogram splits, leaf-wise growth, native categoricals; tuning guide | Kaggle, production scoring |
| stacking-blending | Stacking & blending | meta-learners; out-of-fold predictions; diversity; when ensembling stops helping; inference cost | competitions |

## Module 5 · Evaluation & model selection  (`ml-evaluation.html`)
*Animation focus: thresholds sliding across score distributions; folds rotating; learning curves diagnosing a model.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| metrics-classification | Classification metrics | confusion matrix; precision, recall, F1, MCC; ROC-AUC vs PR-AUC; threshold choice by cost | every classifier |
| metrics-regression | Regression & ranking metrics | MAE, MSE/RMSE, R², MAPE; residual analysis; ranking metrics (NDCG, MAP) | forecasting, search |
| cross-validation | Cross-validation | k-fold, stratified, group, time-series CV; nested CV; variance of estimates | honest model comparison |
| hyperparameter-tuning | Hyper-parameter tuning | grid vs random vs Bayesian search; early stopping; budgets; over-tuning the validation set | AutoML |
| learning-curves-diagnosis | Diagnosing models | learning and validation curves; high bias vs high variance remedies; error analysis; data vs model fixes | day-to-day debugging |
| calibration-uncertainty | Calibration & uncertainty | reliability diagrams; Platt/isotonic; Brier score; prediction intervals; conformal idea | risk-sensitive decisions |
| interpretability-fairness | Interpretability & fairness | permutation importance, PDP, SHAP idea; fairness metrics and trade-offs; documenting limits | regulated domains |

## Module 6 · Neural networks  (`ml-neural.html`)
*Animation focus: data flowing through layers with shapes; gradients flowing backwards; decision regions forming.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| perceptron-mlp | From perceptron to MLP | neuron; layers; why depth; universal approximation; parameter count and shape table | all deep learning |
| activations | Activation functions | sigmoid, tanh, ReLU, Leaky/GELU/SiLU; saturation and dead units; derivatives | choosing activations |
| backprop-training | Training: loss, backprop & batches | forward/backward on a worked MLP; mini-batches; epochs; training loop pseudocode | every framework |
| init-normalization | Initialisation & normalisation | Xavier/He init; batch/layer norm; why training stabilises | deep nets |
| regularization-dl | Regularisation for deep nets | dropout, weight decay, early stopping, augmentation, label smoothing; the generalisation gap | preventing overfit |
| optimizers-schedules | Optimisers & schedules in practice | SGD+momentum, Adam/AdamW; learning-rate finder, warmup, cosine; batch-size effects | training recipes |
| mlp-tabular | MLP architectures for tabular data | embeddings for categoricals; residual MLPs; width/depth trade-offs; when trees still win | tabular deep learning |
| debugging-nn | Debugging neural networks | overfit one batch; loss curves; exploding/vanishing signals; common bugs checklist | any training job |

## Module 7 · Deep-learning architectures at a glance  (`ml-architectures.html`)
*Animation focus: the same data flowing through different architectures; hidden states unrolling over time; attention weights.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| cnn-overview | Convolutional networks | local filters and weight sharing; feature maps; pooling; why they suit images (full treatment in the Computer Vision topic) | vision, audio |
| rnn-lstm | Recurrent networks, LSTM & GRU | hidden state; unrolling through time; vanishing gradients; gates with a worked cell step | sequences, time series |
| attention-transformers | Attention & transformers | queries/keys/values; self-attention worked by hand; the transformer block; parameters and cost O(n²) | LLMs, modern vision |
| embeddings-autoencoders | Embeddings & autoencoders | learned representations; embedding tables; autoencoders and bottlenecks; similarity search | recommenders, retrieval |
| gnn-intro | Graph neural networks | message passing; aggregation; node/graph tasks; when data is a graph | molecules, social networks |
| architecture-choice | Choosing an architecture | inductive biases by data type; model size vs data size; compute budget; decision guide | project design |

## Module 8 · Beyond supervised learning & special problems  (`ml-advanced.html`)
*Animation focus: labels propagating; users and items in a shared space; forecasts with widening uncertainty.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| semi-self-supervised | Semi- & self-supervised learning | using unlabeled data; pseudo-labels; consistency; contrastive learning | scarce labels |
| transfer-learning | Transfer learning & fine-tuning | pretrained models; feature extraction vs fine-tuning; domain shift; adapters | most deep-learning projects |
| recommender-systems | Recommender systems | collaborative filtering; matrix factorisation (derived); implicit feedback; cold start; ranking & evaluation | e-commerce, media |
| time-series | Time-series forecasting | trend/seasonality; lag features; stationarity; ARIMA intuition; ML forecasting; time-aware validation | demand, sensors |
| bayesian-ml | Bayesian machine learning | priors and posteriors on parameters; Bayesian linear regression; uncertainty; Gaussian processes intuition | small data, risk |
| causal-ab | Causal thinking & experiments | correlation vs causation; confounding; randomised experiments and A/B tests; uplift idea | measuring product impact |

## Module 9 · ML in production: system architecture & MLOps  (`ml-production.html`)
*Animation focus: requests and data flowing through system components with queues, caches, stores and monitors; drift appearing and alarms firing.*
| id | Lesson | Must understand | Where it appears |
|---|---|---|---|
| ml-system-design | Designing an ML system | requirements and metrics (business, model, system); batch vs real-time; reference architectures; cost model | every deployment |
| feature-stores-data | Data, features & storage | data lake/warehouse; feature store (offline/online); data versioning; training-serving skew | platform teams |
| experiments-registry | Experiments, reproducibility & registry | experiment tracking; seeds and environments; model registry and stages; CI/CD for models | MLOps |
| serving-inference | Serving & inference | batch, online, streaming, edge; latency vs throughput; batching and caching; canary, shadow, A/B rollout; scaling | APIs |
| monitoring-drift | Monitoring & drift | data drift, concept drift, performance and system metrics; statistical drift tests (PSI, KS); retraining triggers | operations |
| ml-governance | Responsible & governed ML | bias audits, privacy, security, model cards, documentation, regulation basics (link to Governing AI) | regulated products |
| end-to-end-project | Capstone: an end-to-end project | choose data, build pipeline, train, evaluate, deploy, monitor — an animated whole-system simulation with editable stages | portfolio project |

---
## Content standards (apply to every lesson)
- Intuition first; every symbol explained; at least one hand-worked numeric example on a tiny dataset; algorithm pseudo-code where relevant.
- Every algorithm lesson states: what it *learns* (stored parameters), training cost, inference cost, preprocessing needs, and when to prefer/avoid it.
- At least one **animated, editable lab** per lesson. Show the *learning process*, not only the final result.
- Correct and honest: derive formulas, verify numbers in code, state assumptions; label illustrative simulations as such.
- Animations: slow, dt-based, captioned, pausable (see curriculum/AUTHORING.md).
