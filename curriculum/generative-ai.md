# Generative AI — Curriculum (zero → hero)

Goal: a learner who knows basic probability and what a neural network is ends able to explain, from first principles, how a model
**generates** new text, images and other data: what it is trained to do, why the **Transformer** works, how a **language model** is
pretrained, aligned and served, how **retrieval and agents** are built on top, how **diffusion** and **flow** models create images,
and how generators are **evaluated, secured and operated** responsibly.

**Emphases:** (1) **One idea per lesson, built on a tiny example** — a vocabulary of 4–6 tokens, a 3-token attention head, a 1-D
diffusion chain — so every number can be checked by hand. (2) **Animation** — probabilities reshape as temperature changes, attention
weights light up between tokens, a KV cache grows token by token, noise is added and removed, a sampler walks a path. (3) **Honesty about
scale** — the labs are small toy models; real systems differ in size, data and engineering, and any statement about a specific
commercial model is avoided because it goes out of date. Formulas and published methods are named; claims about benchmarks are not made.

Order is strict. Ten modules, 61 lessons. Module pages are `gen-<name>.html`; hub `generative-ai.html`.

Maths and ML prerequisites are linked, not re-taught: `math-probability.html` (distributions, expectation, Bayes),
`math-information.html` (cross-entropy, KL, mutual information), `math-neural.html` (backpropagation), `math-optimization.html`
(gradient descent, Adam), `math-linear-algebra.html` (matrix products), `ml-neural.html` (networks, softmax), `ml-architectures.html`
(architectures at a glance), `cv-transformers.html` (the Vision Transformer), `cv-generative.html` (autoencoders, GANs and diffusion for
images), `ds-statistics.html` (sampling, intervals). Computer Vision covers image-specific generation; this topic gives the general theory.

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete generation problem that makes the idea necessary.
2. **Intuition** – plain words and a picture first.
3. **Build it up** – step by step with a hand-worked example on a tiny vocabulary, a 3-token sequence, or a 1-D/2-D toy distribution (real numbers).
4. **Formal definition** – equations and a symbol table.
5. **Animated lab** – the learner changes a temperature, a rank, a guidance scale, a noise level or a prompt token and sees the result move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real systems described by method name, never by a claimed benchmark.
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.

---
## Module 0 · Foundations of generative modelling  (`gen-foundations.html`)
*Animation focus: sampling from a small table of probabilities, a data cloud and a model cloud overlapping as parameters move, the chain rule unrolling one token at a time.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| generative-vs-discriminative | Generative versus discriminative models | p(y given x) versus p(x); sampling new data; conditional generation p(x given c); the link through Bayes | every generative system |
| maximum-likelihood-sampling | Likelihood and sampling | likelihood of a dataset; maximum likelihood as minimising KL from data to model; sampling by inverse transform on a small table | the training objective of language models and flows |
| chain-rule-factorisation | The chain rule: generating one piece at a time | p(x) as a product of conditionals; autoregressive factorisation; ordering; sampling cost grows with length | language models, audio models |
| families-of-generators | The families of generators | autoregressive, latent-variable (VAE), adversarial (GAN), flow, diffusion and score models; trade-offs in likelihood, quality, speed and coverage | choosing a method |
| representing-data | Tokens, patches and latents | discrete versus continuous data; vocabulary size; sequence length; compression into a smaller space | text, image and audio tokens |

## Module 1 · Language modelling & tokenisation  (`gen-language.html`)
*Animation focus: counts becoming probabilities, BPE merging the most frequent pair step by step, a softmax over a vocabulary reshaping as temperature and top-p move.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| ngram-models | N-gram language models | counting; conditional probability from counts; sparsity; add-k smoothing | baselines, spelling and keyboard models |
| perplexity | Perplexity | cross-entropy per token; perplexity as exp of the average loss; bits per character; comparing models on the same text | evaluating language models |
| tokenisation | Tokenisation: words, characters and subwords | the byte-pair merge algorithm by hand; byte-level fallback; vocabulary size trade-offs; tokens and cost | every language model |
| embeddings-lm | Embeddings | an embedding table as a lookup; one-hot times a matrix; similarity of vectors; tying input and output embeddings | every neural language model |
| neural-language-model | A neural language model | context to hidden state to logits over the vocabulary; softmax; cross-entropy and its gradient p minus one-hot | the output head of every LLM |
| decoding-sampling | Decoding and sampling | greedy, beam search, temperature, top-k, top-p; repetition; diversity versus quality | chat models, code completion |

## Module 2 · Attention & the Transformer  (`gen-transformer.html`)
*Animation focus: query–key weights lighting up between tokens, heads attending to different positions, a residual stream passing through blocks, a KV cache growing as tokens are generated.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| attention-lookup | Attention as a soft lookup | query, key and value; a weighted average of values; why a fixed-size summary was a bottleneck | all transformers |
| scaled-dot-product | Scaled dot-product attention | softmax(QK^T / sqrt(d_k)) V; why the scale; a 3-token worked example with real numbers | self-attention |
| multi-head | Multi-head attention | several heads in parallel; projection matrices W_Q, W_K, W_V, W_O; head dimension; parameter count | every transformer layer |
| position-information | Position information | attention ignores order; sinusoidal, learned and rotary (RoPE) positions; relative versus absolute | long-context models |
| transformer-block | The transformer block | residual stream; layer normalisation (pre-norm); the MLP and its expansion; where the parameters live | GPT-style models |
| causal-decoder | The causal decoder | the causal mask; decoder-only architecture; teacher forcing; parallel training versus sequential generation | GPT-style language models |
| cost-and-kv-cache | Cost and the KV cache | quadratic attention in time and memory; parameter-count formula; the KV cache and its memory per token | long context, serving |

## Module 3 · Training large language models  (`gen-training.html`)
*Animation focus: loss falling with tokens seen, a power-law line on a log–log plot, memory bars stacking up per GPU, a sharded state dividing across devices.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| pretraining-objective | The pretraining objective | next-token prediction loss; packing sequences; what the model is forced to learn from this loss alone | pretraining |
| data-for-pretraining | Data for pretraining | sources; filtering; deduplication; mixtures; quality versus quantity; contamination; licensing and consent | corpora |
| scaling-laws | Scaling laws | loss as a power law in parameters, data and compute; the compute estimate 6ND; compute-optimal trade-offs; limits of extrapolation | planning training runs |
| optimisation-at-scale | Optimisation at scale | AdamW; warm-up and cosine decay; gradient clipping; batch size; mixed precision with bfloat16 | LLM training recipes |
| parallelism-memory | Parallelism and memory | memory for parameters, gradients, optimiser state and activations; data, tensor and pipeline parallelism; sharding | multi-GPU training |
| training-diagnostics | Operating a training run | reading loss curves; spikes and divergence; evaluation during training; checkpoints and restarts | training operations |

## Module 4 · Alignment & post-training  (`gen-alignment.html`)
*Animation focus: a policy distribution shifting toward preferred answers while a KL leash holds it near the reference, a reward model scoring pairs, group-relative advantages computed for sampled answers.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| instruction-tuning | Instruction tuning | supervised fine-tuning on prompt–response pairs; chat templates; loss masking on the response; data diversity | assistant models |
| preference-data | Preference data and reward models | pairwise comparisons; the Bradley–Terry model; the reward-model loss; annotator noise | RLHF |
| rlhf | Reinforcement learning from human feedback | the reward plus KL penalty to a reference policy; the policy-gradient and PPO outline; reward hacking | aligned chat models |
| dpo | Direct preference optimisation | deriving the DPO loss from the KL-regularised objective; the implicit reward; comparison with RLHF | open-model tuning |
| verifiable-rewards | Verifiable rewards and reasoning | rewards from checkers for maths and code; group-relative baselines; long reasoning traces; limits and risks | reasoning models |
| ai-feedback-red-teaming | AI feedback and red teaming | constitution-style AI feedback; red teaming; refusals and over-refusal trade-offs; measuring safety behaviour | safety training |

## Module 5 · Efficient adaptation & inference  (`gen-efficiency.html`)
*Animation focus: a low-rank update added to a frozen matrix, weights snapping to a coarse grid, prefill then decode steps on a timeline, a draft model proposing tokens that a target model accepts or rejects.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| lora | Low-rank adaptation (LoRA) | the update delta W = B A; rank r; trainable parameter count; merging into the base weights; where to apply it | fine-tuning on one GPU |
| quantisation | Quantisation | uniform quantisation with scale and zero point; rounding error; int8 and int4; per-channel and group scales; memory savings | on-device and low-cost serving |
| serving-batching | Serving: prefill, decode and batching | the compute-bound prefill versus the memory-bound decode; batching; continuous batching; paged KV cache; latency versus throughput | inference servers |
| speculative-decoding | Speculative decoding | a draft model proposes tokens; the target model verifies in parallel; the acceptance rule that preserves the distribution; expected speed-up | faster decoding |
| distillation | Distillation | teacher–student training with KL on soft targets; temperature; distilling from generated data | small and fast models |
| mixture-of-experts | Mixture of experts | a router picks the top-k experts per token; total versus active parameters; load balancing | large sparse models |

## Module 6 · Prompting, retrieval & agents  (`gen-applications.html`)
*Animation focus: few-shot examples shifting an output distribution, a grammar mask removing illegal tokens, a query vector finding nearest chunks, a plan–act–observe loop running step by step.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| prompting-icl | Prompting and in-context learning | zero- and few-shot prompts; in-context learning; chain of thought; prompt sensitivity; what prompting cannot fix | every application |
| structured-outputs | Structured outputs | JSON schemas; constrained decoding by masking logits with a grammar; validation and retries | tool integrations |
| retrieval-embeddings | Retrieval with embeddings | sentence embeddings; cosine similarity; vector indexes and approximate nearest neighbours; chunking | search and RAG |
| rag-pipeline | Retrieval-augmented generation | retrieve, rerank, generate; grounding and citation; failure modes; evaluating retrieval and answers separately | enterprise assistants |
| tool-use-agents | Tools and agents | function calling; the ReAct loop of reason, act and observe; error handling; the cost of long loops | coding and task assistants |
| context-memory | Context and memory | context-window limits; truncation, summarisation and memory stores; position effects on recall | long conversations |

## Module 7 · Latent-variable & adversarial generators  (`gen-latent.html`)
*Animation focus: a 1-D encoder and decoder with the ELBO terms trading off, a latent space sampled and decoded, a discriminator boundary chasing a generator, a density warped by an invertible map.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| vae-elbo | Variational autoencoders and the ELBO | a latent-variable model; the ELBO as reconstruction minus KL; the reparameterisation trick; a 1-D worked example | VAEs, latent spaces |
| vae-pitfalls | VAE pitfalls and discrete codes | posterior collapse; blurry samples; the beta-VAE; VQ-VAE codebooks and image tokenisers | image and audio tokenisers |
| gan-objective | The GAN objective | the minimax game; the optimal discriminator; the JS divergence view; the non-saturating loss | GANs |
| gan-training | Training GANs | instability; mode collapse; the Wasserstein idea; evaluation difficulties; why diffusion models overtook them | image synthesis history |
| flows-change-of-variables | Normalising flows | the change-of-variables formula; the Jacobian determinant; coupling layers; exact likelihood | density estimation, link to flow matching |

## Module 8 · Diffusion & flow matching  (`gen-diffusion.html`)
*Animation focus: points in 1-D and 2-D dissolving into noise along a schedule and being pulled back, a denoising arrow field, sampler steps on a trajectory, a guidance slider sharpening a conditional distribution.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| forward-noising | The forward noising process | q(x_t given x_0) in closed form; noise schedules; signal-to-noise ratio; a 1-D worked example | training any diffusion model |
| noise-prediction-loss | Learning to predict the noise | the epsilon-prediction target; the simple loss; its link to the score; a hand-worked training step | diffusion training |
| ddpm-sampling | Sampling with DDPM | the reverse update; ancestral sampling; number of steps; the role of added noise | DDPM and descendants |
| fast-samplers | Fast samplers | DDIM and the deterministic ODE view; fewer steps; solver trade-offs; distilling to few-step models | fast image generation |
| score-sde-view | The score and SDE view | the score as the gradient of the log density; Langevin dynamics; the SDE and ODE unification (conceptual) | theory behind samplers |
| guidance | Guidance | conditional models; the classifier-free guidance formula; the scale trade-off between fidelity and diversity | text-to-image models |
| latent-diffusion | Latent diffusion | diffusing in an autoencoder's latent space; text conditioning with cross-attention; U-Net versus transformer backbones | Stable-Diffusion-style systems |
| flow-matching | Flow matching | straight-line paths between noise and data; regressing a velocity; rectified flows; comparison with diffusion | modern image and video generators |

## Module 9 · Multimodal, evaluation & responsible generation  (`gen-multimodal.html`)
*Animation focus: image and text tokens interleaved in one sequence, a contrastive similarity matrix sharpening, an evaluation score with its confidence band, an alert firing in a generation pipeline.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| multimodal-models | Multimodal models | image and audio tokens fed to a language model; adapters; CLIP-style contrastive pretraining and its link to mutual information; vision–language assistants | multimodal assistants |
| video-audio-generation | Video and audio generation | latent diffusion with a time axis; spatiotemporal tokens; audio codecs and token language models; consistency over time (conceptual) | video, speech and music generation |
| evaluating-generators | Evaluating generators | perplexity and benchmarks; contamination; human preference and Elo ratings; LLM-as-judge biases; FID and CLIP score for images | model selection |
| hallucination-factuality | Hallucination and factuality | why models produce plausible falsehoods; calibration; grounding and verification; measuring factuality | trustworthy assistants |
| misuse-provenance | Misuse, provenance and consent | jailbreaks and prompt injection; synthetic media; watermarking and content provenance; copyright and consent debates, stated even-handedly | policy and safety |
| production-genai | Running generative systems in production | cost and latency budgets; caching; guardrails; monitoring; evaluation harnesses; versioning models and prompts | deployed products |
