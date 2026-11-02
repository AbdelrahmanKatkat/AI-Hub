# Reinforcement Learning — Curriculum (zero → hero)

Goal: a learner who knows basic probability and calculus ends able to frame a sequential decision problem as a Markov decision process,
choose and implement the right solution method (bandits, dynamic programming, Monte Carlo, temporal-difference learning, function
approximation, policy gradients, model-based planning), understand why deep RL is unstable and how DQN, actor-critic and PPO fix it, and
reason about reward design, evaluation and the use of RL in games, robots and language-model alignment.

**Emphases:** (1) **Trial, error and credit** — each lesson shows an agent acting, a reward arriving late, and the value estimate moving
backwards through time. (2) **Animation** — value tables fill in sweep by sweep, bandit arms are pulled and their estimates converge,
TD errors ripple back along trajectories, policies sharpen as their probabilities update, and learning curves grow with seeds and
confidence bands. (3) **Architecture** — the agent–environment loop, replay buffers, target networks, actor–critic pairs and planning
trees, each with its data flow and update schedule.

Order is strict. Ten modules, 56 lessons. Module pages are `rl-<name>.html`; hub `reinforcement-learning.html`.

Mathematics and ML prerequisites are linked, not re-taught: `math-probability.html` (expectation, sampling), `math-calculus.html`
(gradients), `math-optimization.html` (gradient descent), `math-neural.html` (backprop), `math-information.html` (KL divergence),
`ml-neural.html` (value and policy networks), `ml-evaluation.html` (confidence intervals), `robo-learning.html` (robot policies and
sim-to-real).

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete decision problem (a slot machine, a gridworld, a game, a robot arm) that makes the idea necessary.
2. **Intuition** – plain words first, with a picture of the agent and the environment.
3. **Build it up** – step by step with a hand-worked example on a 2–4 state chain, a 3×3 gridworld, a 2-armed bandit or a short trajectory (real numbers).
4. **Formal definition** – equations and a symbol table.
5. **Animated lab** – the learner changes a parameter (ε, α, γ, λ, a reward, a step size) and sees values, policies or curves move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real environments, games, robots and systems that use the method.
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.

---
## Module 0 · Foundations of sequential decisions  (`rl-foundations.html`)
*Animation focus: the agent–environment loop passing actions and rewards; a return accumulating with discounting; a gridworld whose states light up as the agent walks.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| what-is-rl | What reinforcement learning is | agent, environment, state, action, reward; trial and error versus supervised learning; the interaction loop | games, robotics, LLM alignment |
| rewards-returns | Rewards, returns & discounting | reward signal; return G_t = R_{t+1} + γR_{t+2} + …; the discount factor γ and its horizon | every RL objective |
| mdp | Markov decision processes | states, actions, transition probabilities, rewards; the Markov property; a gridworld MDP worked by hand | the formal model of RL |
| policies-values | Policies, state values & action values | stochastic policy π(a∣s); V^π(s) and Q^π(s,a); a first look at the Bellman relation | all RL algorithms |
| exploration-intro | Exploration versus exploitation | why acting greedily can miss better options; the cost of exploring | all RL |

## Module 1 · Bandits & exploration  (`rl-bandits.html`)
*Animation focus: arms pulled one at a time with their estimates converging; ε-greedy choices shown as coloured tallies; UCB confidence bars shrinking; a drifting non-stationary target tracked by a constant step size.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| bandit-problem | The k-armed bandit | repeated choice with noisy rewards; the action-value q*(a); regret intuition | A/B testing, recommendations |
| sample-average | Sample averages & incremental updates | Q_n = Q_{n−1} + (1/n)(R_n − Q_{n−1}); why the mean converges | value estimation |
| epsilon-greedy | Epsilon-greedy exploration | greedy versus random choice with probability ε; the trade-off between speed and final performance | simple exploration baseline |
| ucb | Upper confidence bound | optimism under uncertainty; the bonus c√(ln t / N(a)); a worked example | principled exploration |
| gradient-bandit | Gradient bandits & softmax preferences | action preferences H(a); softmax probabilities; the preference update | policy-based learning on bandits |
| nonstationary-bandits | Non-stationary problems & constant step size | why recent rewards should count more; exponential recency weighting; α | tracking drifting users |

## Module 2 · Dynamic programming  (`rl-dynamic-programming.html`)
*Animation focus: a value table sweeping state by state; the greedy arrow field updating after each improvement; policy iteration alternating evaluation and improvement until the arrows stop changing.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| bellman-equations | Bellman expectation & optimality equations | recursive definition of V^π and Q^π; the Bellman optimality equation for V*; a worked two-state example | the foundation of value learning |
| policy-evaluation | Iterative policy evaluation | sweeping the Bellman expectation backup until convergence; stopping tolerance | evaluating any policy |
| policy-improvement | Policy improvement & the greedy policy | the policy improvement theorem; acting greedily with respect to Q^π | improvement step of control |
| policy-iteration | Policy iteration | alternating evaluation and improvement; convergence to an optimal policy in finite MDPs | model-based planning |
| value-iteration | Value iteration | the Bellman optimality backup; truncated policy evaluation; complexity per sweep | planning with a known model |
| generalized-policy-iteration | Generalised policy iteration | evaluation and improvement as competing pressures that converge together | the pattern behind all control methods |

## Module 3 · Monte Carlo methods  (`rl-monte-carlo.html`)
*Animation focus: episodes sampled and their returns walked back to the start; visit counts filling a value table; importance-sampling weights multiplying along a trajectory and then clipped.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| mc-prediction | Monte Carlo prediction | learning from complete episodes; first-visit and every-visit averages; no model needed | evaluating a policy from experience |
| mc-control | Monte Carlo control with exploring starts | action values from episodes; greedy improvement; why exploring starts are unrealistic | model-free control |
| on-policy-mc | On-policy Monte Carlo with ε-soft policies | ε-soft policies keep exploring; the ε-greedy improvement step | practical model-free control |
| importance-sampling | Importance sampling | reweighting returns from a behaviour policy to a target policy; ordinary versus weighted estimators; variance | learning from logged data |
| off-policy-mc | Off-policy Monte Carlo control | learning the greedy policy while following a different exploratory one; the stopping rule when weights become zero | offline evaluation |
| mc-tradeoffs | Monte Carlo strengths & limits | unbiased but high-variance returns; requirement of finite episodes; when MC is the right choice | baseline for all later methods |

## Module 4 · Temporal-difference learning  (`rl-td.html`)
*Animation focus: a TD error appearing at one step and propagating backwards as values update; SARSA and Q-learning paths on the same gridworld diverging near a cliff; eligibility traces decaying along a trajectory.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| td-prediction | TD(0) prediction & bootstrapping | the TD error δ = R + γV(S′) − V(S); learning before the episode ends; bootstrapping's bias and variance | value estimation at scale |
| sarsa | SARSA: on-policy TD control | learning Q from the next action actually taken; the update rule; the cliff-walking example | on-policy control |
| q-learning | Q-learning: off-policy TD control | the max over next actions; learning the optimal policy while exploring; the tabular convergence idea | classic control, DQN ancestor |
| expected-sarsa | Expected SARSA | averaging over the policy at the next state; lower variance than SARSA | stable tabular control |
| n-step-td | n-step returns | blending one-step and Monte Carlo targets; the n-step return and its error | bias–variance trade-off |
| td-lambda | TD(λ) & eligibility traces | λ-returns, eligibility traces, the forward and backward views; λ = 0 and λ = 1 | fast credit assignment |

## Module 5 · Function approximation & deep Q-learning  (`rl-approximation.html`)
*Animation focus: a value surface fitted by a linear model with features lighting up; the deadly triad as a loop that diverges; a replay buffer sampling a shuffled batch; a target network updating in steps.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| linear-approximation | Linear function approximation | v̂(s,w) = wᵀx(s); the semi-gradient update; the feature vector as the design choice | large state spaces |
| features-tiles | Feature construction & tile coding | coarse coding, tile coding, radial basis features; generalisation versus resolution | hand-engineered features |
| deadly-triad | The deadly triad | function approximation, bootstrapping and off-policy learning together can diverge; a small divergent example | why deep RL is fragile |
| experience-replay | Experience replay | storing transitions, sampling mini-batches, breaking correlation between consecutive samples | DQN |
| target-networks | Target networks & DQN | a frozen copy of the Q-network for targets; the loss and the training loop; Atari-style inputs | deep Q-learning (Atari) |
| double-dqn | Overestimation & Double DQN | the max operator's positive bias; decoupling selection from evaluation; double Q-learning | Double DQN, Rainbow-style agents |

## Module 6 · Policy-gradient & actor–critic methods  (`rl-policy-gradient.html`)
*Animation focus: a softmax policy's bars shifting toward rewarded actions; a REINFORCE update showing the sampled return pushing a probability up or down; an actor and critic side by side; PPO's clipped ratio capping a step.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| policy-parameterisation | Parameterising policies | softmax policies over discrete actions; Gaussian policies over continuous actions; why a policy can be stochastic | continuous control |
| policy-gradient-theorem | The policy gradient theorem | ∇J(θ) ∝ E[∇ log π(a∣s) Q(s,a)]; the score-function trick; a worked one-step example | all policy-gradient methods |
| reinforce | REINFORCE | Monte Carlo policy gradient; high variance; the update rule in code form | simplest policy learning |
| baselines-advantage | Baselines & the advantage function | subtracting a baseline without bias; A(s,a) = Q(s,a) − V(s); why the advantage reduces variance | variance reduction |
| actor-critic | Actor–critic methods | an actor (policy) and a critic (value) trained together; bootstrapped advantage estimates | A2C, A3C |
| ppo | Proximal policy optimisation | the probability ratio r(θ); the clipped surrogate objective; why it limits policy change | RLHF, robot locomotion |

## Module 7 · Model-based reinforcement learning  (`rl-model-based.html`)
*Animation focus: Dyna-Q replaying simulated transitions from a learned model between real steps; a Monte Carlo tree growing through selection, expansion, simulation and backup; model error compounding over a rollout.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| models-planning | Learned models & Dyna-Q | a transition and reward model; planning by simulated updates; Dyna-Q worked example | sample-efficient learning |
| mcts | Monte Carlo tree search | selection, expansion, simulation and backpropagation; the UCT rule; an anytime search | AlphaGo and AlphaZero |
| model-error | Model error & compounding rollouts | one-step prediction error grows over a horizon; short rollouts as a remedy | model-based RL limits |
| model-predictive-rl | Model-predictive control with learned models | replanning at each step with a learned model; the link to robotics MPC | robot control, games |

## Module 8 · Advanced topics  (`rl-advanced.html`)
*Animation focus: a count-based novelty bonus filling unvisited states; options switching a high-level policy; two learners in a shared environment drifting out of sync; a dataset of fixed transitions with out-of-distribution actions flagged.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| intrinsic-motivation | Exploration with intrinsic rewards | count-based bonuses; prediction error as curiosity; the risk of rewarding noise | hard-exploration games |
| hierarchical-rl | Hierarchical RL & options | options as temporally extended actions; a high-level and low-level policy; the semi-MDP view | skill learning |
| multi-agent | Multi-agent reinforcement learning | independent learners and non-stationarity; cooperative versus competitive returns; a Nash-equilibrium intuition | team games, traffic |
| offline-rl | Offline reinforcement learning | learning from a fixed dataset; distribution shift and out-of-distribution actions; conservative value estimates | learning from logs |
| inverse-rl | Imitation and inverse reinforcement learning | recovering a reward from demonstrations; the ambiguity of recovered rewards; a link to behaviour cloning | learning from experts |

## Module 9 · Applied RL, alignment & evaluation  (`rl-applied.html`)
*Animation focus: a reward model scoring two responses from a preference pair; a KL penalty pulling a fine-tuned policy back toward its reference; a learning curve with seeds and a confidence band; a reward-hacking example where the proxy rises and the true objective falls.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| rlhf | Reinforcement learning from human feedback | preference data; reward models trained with a pairwise loss; KL-regularised policy optimisation | aligning language models |
| reward-hacking | Reward hacking & specification gaming | proxy rewards diverging from true goals; Goodhart's law; examples from games and robots | safety, evaluation |
| self-play | Self-play & AlphaZero-style training | learning by playing against copies of itself; policy and value networks with search; the training loop | board games, strategy games |
| evaluation-reproducibility | Evaluating RL fairly | seeds and variance; learning curves with confidence intervals; sample-efficiency reporting | reproducible research |
| rl-systems | Building RL systems in production | simulators, logging, offline evaluation before deployment, and safe rollout | recommender and control systems |
| rl-governance | RL in the real world: risk and oversight | reward and action constraints, human oversight, audit trails | governing AI systems |
