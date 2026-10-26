# Mathematics for AI — Curriculum (zero → hero)

Goal: a learner who starts with school arithmetic ends able to read a modern ML/DL paper, derive backpropagation and
the loss functions by hand, reason about why training succeeds or fails, and understand the maths behind
transformers, VAEs and diffusion models.

Order is strict: every lesson only uses ideas from earlier lessons. Nine modules, 58 lessons.

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete problem that makes the idea necessary.
2. **Intuition** – plain words, analogy, a mental picture. No symbols yet.
3. **Build it up** – step-by-step, with a hand-worked numeric example.
4. **Formal definition** – the equation, with *every symbol explained*.
5. **Lab** – an interactive/animated widget with "try this" instructions.
6. **Common traps** – misconceptions and mistakes.
7. **Where it shows up in AI** – specific, not vague.
8. **Check yourself** – 3 questions, answers hidden in `<details>`.
9. **Takeaways**.

---
## Module 0 · Foundations  (`math-foundations.html`)
*Prerequisite: arithmetic. Builds the language everything else is written in.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| numbers-sets | Numbers, sets & ℝⁿ | naturals→integers→rationals→reals; sets, ∈, ⊂, intervals; tuples; ℝⁿ as "a list of n numbers" | a data point is a point in ℝⁿ; a dataset is a set/matrix |
| functions | Functions | input→output rule; domain/range; graph; composition f∘g; inverse; variables vs parameters f(x;θ) | a model is a function; layers are composed functions |
| algebra | Algebra & equations | equations as balance; solving linear/quadratic; inequalities; 2×2 systems; rearranging formulas | solving for weights; closed-form regression |
| sums-products | Σ and Π notation | index, bounds, expanding, rules (linearity), averages, Π as repeated multiplication | loss = (1/N)Σ…, likelihood = Π… |
| exp-log | Exponents & logarithms | power rules; eˣ; ln as inverse; log turns × into +; log scales; growth/decay | log-likelihood, softmax, sigmoid, numerical underflow |
| trig | Angles, sine & cosine | radians; unit circle; sin/cos as coordinates; waves: amplitude, frequency, phase | rotation matrices, positional encodings, Fourier |
| coords-geometry | Coordinates & geometry | distance formula; lines/planes/hyperplanes; circles/spheres; high-dimensional strangeness | decision boundaries, k-NN, embeddings |
| notation | Reading mathematical notation | Greek letters, subscripts, hats, bold, ∀ ∃, argmin/argmax; translating formula ↔ code | reading papers |

## Module 1 · Linear algebra  (`math-linear-algebra.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| vectors | Vectors | arrows vs lists; add, scale; linear combination; span | embeddings, feature vectors |
| dot-product | Dot product & angle | Σ aᵢbᵢ; geometric meaning; projection; sign ↔ angle; orthogonality | cosine similarity, attention scores, neuron pre-activation |
| norms-distance | Norms & distances | L1, L2, L∞; unit vectors; distance; similarity vs distance | regularization, k-NN, normalization |
| matrices | Matrices | table of numbers; rows/cols; transpose; shapes; data matrix X | batches, weight matrices |
| matmul | Matrix multiplication | three views (rows·cols, combo of columns, composition); shapes; non-commutativity; cost | every dense layer; GPU workload |
| linear-maps | Matrices as transformations | linear map; columns = where basis lands; rotate/scale/shear; composition | layers reshape space |
| det-inverse | Determinant & inverse | area/volume scaling; singular; inverse undoes; when it exists | normal equation; why collinear features break things |
| linear-systems | Linear systems & rank | Ax=b; solutions 0/1/∞; rank; row space/null space; independence | under/over-determined problems, redundancy |
| eigen | Eigenvalues & eigenvectors | Av=λv; finding them; diagonalization; powers of A; symmetric matrices | PCA, stability of dynamics, Hessian curvature |
| svd-pca | SVD & PCA | any matrix = rotate·stretch·rotate; singular values; low-rank approx; PCA as variance max | compression, LoRA, embeddings |
| projections-ls | Projections & least squares | orthogonal projection; normal equation; geometric view of regression | linear regression |

## Module 2 · Calculus  (`math-calculus.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| limits-slope | Rate of change & limits | average→instantaneous rate; limit idea; derivative as slope; differentiability | slope of the loss |
| derivative-rules | Derivative rules | power, sum, product, quotient; derivatives of eˣ, ln, sin, σ, tanh, ReLU | activation derivatives |
| chain-rule | The chain rule | composition; "multiply the local rates"; worked chains | the heart of backprop |
| partials-gradient | Partial derivatives & gradient | one variable at a time; gradient vector; steepest ascent; directional derivative | ∇L |
| jacobian-hessian | Jacobian & Hessian | vector→vector derivative; second derivatives; curvature | layer derivatives; Newton; sharp/flat minima |
| taylor | Linear & quadratic approximation | Taylor series; why gradient descent works locally; step size from curvature | learning-rate limits |
| integrals | Integrals | area under curve; accumulation; integral as continuous Σ; fundamental theorem | probability densities, expectations |

## Module 3 · Probability  (`math-probability.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| prob-basics | Probability & counting | sample space, events, axioms, counting, independence | datasets as samples |
| conditional-bayes | Conditional probability & Bayes | P(A\|B); chain rule of probability; Bayes' theorem; base-rate fallacy | classifiers, spam filters, posteriors |
| random-variables | Random variables & expectation | discrete/continuous; PMF/PDF/CDF; E[X], Var; linearity of expectation | loss = expectation |
| distributions | Key distributions | Bernoulli, Binomial, Categorical, Uniform, Gaussian, Exponential, Poisson | output layers, noise models |
| joint-covariance | Joint, marginal, covariance | joint/marginal/conditional; independence; covariance & correlation; multivariate Gaussian | PCA, Gaussian processes, whitening |
| lln-clt | Law of large numbers & CLT | sample mean converges; bell curve emerges; when it fails | why SGD works, error bars |

## Module 4 · Statistics  (`math-statistics.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| estimation-mle | Estimation & maximum likelihood | parameters vs estimates; likelihood; log-likelihood; MLE derivations | MSE = Gaussian MLE; cross-entropy = Bernoulli/categorical MLE |
| map-bayes | Priors, MAP & regularization | posterior ∝ likelihood×prior; MAP; L2 = Gaussian prior; L1 = Laplace prior | weight decay |
| bias-variance | Bias–variance & generalization | train vs test error; underfit/overfit; decomposition; double descent (intro) | model selection |
| intervals-tests | Uncertainty, intervals & tests | standard error; confidence intervals; hypothesis tests; p-values and their misuse | comparing models honestly |
| resampling | Resampling & evaluation | train/val/test, cross-validation, bootstrap, leakage | trustworthy evaluation |

## Module 5 · Information theory  (`math-information.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| entropy | Surprise & entropy | information = −log p; entropy as average surprise; bits/nats | compression, uncertainty |
| cross-entropy-kl | Cross-entropy & KL divergence | H(p,q); KL ≥ 0; asymmetry; relation to MLE | classification loss, distillation, RLHF |
| mutual-info | Mutual information | dependence; I(X;Y); information bottleneck idea | representation learning, contrastive learning |

## Module 6 · Optimization  (`math-optimization.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| landscapes | Loss landscapes & convexity | minima, maxima, saddles; convex sets/functions; local vs global | why deep nets still train |
| gradient-descent | Gradient descent | update rule; step size; convergence on quadratics; condition number | the training loop |
| sgd | Stochastic & mini-batch GD | noisy gradient estimates; batch size; variance | large-scale training |
| momentum-adam | Momentum, RMSProp & Adam | exponential moving averages; per-parameter scaling; bias correction | default optimizers |
| schedules | Learning-rate schedules & conditioning | warmup, decay, cosine; ill-conditioning; normalization helps | LLM training recipes |
| constraints-reg | Constraints, Lagrange & regularization | constrained optimization; Lagrange multipliers; L1/L2 as constraints; sparsity | regularization, SVM |

## Module 7 · The mathematics of neural networks  (`math-neural.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| neuron | The neuron & universal approximation | affine + nonlinearity; why nonlinearity; stacking; universal approximation intuition | MLP |
| softmax-stability | Softmax, log-sum-exp & numerical stability | softmax derivation; temperature; log-sum-exp trick; float precision | classification heads, attention |
| losses | Loss functions from first principles | MSE, MAE, Huber, BCE, categorical CE, hinge — each derived from an assumption | choosing a loss |
| backprop | Backpropagation | computational graph; forward/backward; derivation for an MLP; autodiff | training |
| matrix-calculus | Matrix calculus for layers | gradient of Wx+b; batch form; shapes rule | writing/understanding layer backward passes |
| gradient-flow | Vanishing/exploding gradients & initialization | products of Jacobians; Xavier/He init; residuals; normalization | deep nets, ResNets |
| attention-math | Attention & transformers | dot-product attention; scaling by √d; softmax weights; multi-head; positional encoding | LLMs |

## Module 8 · Advanced topics for modern AI  (`math-advanced.html`)
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| markov | Markov chains | transition matrices; stationary distribution; eigenvector view | RL, MCMC, language models |
| graphical-bayes | Bayesian inference & graphical models | latent variables; marginalization; posterior inference; conjugacy | generative modeling |
| kernels | Kernels & similarity | feature maps; kernel trick; Gaussian kernel; Gaussian-process intuition | SVM, GP, attention as kernel |
| vae-elbo | Variational inference & VAEs | intractable posterior; ELBO; reparameterization | VAEs |
| diffusion | Diffusion & score matching | noising process; denoising as learning the score; reverse process | image/video generation |

---
## Content standards (apply to every lesson)
- Start from intuition, never from a bare formula. Every new symbol is explained in words the first time.
- At least one **hand-worked numeric example** with real numbers.
- At least one **interactive lab** that lets the learner change an equation or number and see animation.
- Be mathematically correct: state conditions (e.g. "if the matrix is invertible"), no hand-waving that is false.
- End with "Where it shows up in AI" naming concrete algorithms.
