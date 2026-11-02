# Robotics — Curriculum (zero → hero)

Goal: a learner who knows basic algebra and what a sensor is ends able to take a robot from **sensing** (what the robot knows about itself and
the world), through **estimating** (where am I, what is around me), **planning and controlling** (how to move without hitting anything),
to **learning** (policies learned from demonstrations, simulation and reward) and **deploying safely** (standards, monitoring, operations).

**Emphases:** (1) **The sense–think–act loop** — every lesson shows where it sits in the loop and how fast the loop must run. (2) **Animation** —
frames and transforms move, arms are drawn from joint angles, filters update belief clouds, planners grow trees, controllers track setpoints,
and policies roll out. (3) **Architecture** — the robot software stack from sensor drivers to actuators, and the data flow of learned policies.

Order is strict. Ten modules, 59 lessons. Module pages are `robo-<name>.html`; hub `robotics.html`.

Maths and ML prerequisites are linked, not re-taught: `math-linear-algebra.html` (matrices, rotations), `math-calculus.html` (derivatives,
Jacobians), `math-probability.html` (Bayes, Gaussians), `math-optimization.html` (gradient descent), `math-neural.html` (backprop),
`math-information.html` (KL), `ml-neural.html` (policies as networks), `cv-cnn.html` (image features), `cv-detection.html` (detectors),
`geo-foundations.html` (coordinate reference systems, shared with GeoAI).

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete robot problem that makes the idea necessary.
2. **Intuition** – plain words, a picture of the robot or the scene first.
3. **Build it up** – step by step with a hand-worked example on a 1–2 link arm, a 2-D grid, a 1-D filter or a tiny table (real numbers).
4. **Formal definition** – equations and a symbol table.
5. **Animated lab** – the learner changes a parameter, a gain, a joint angle, a noise level or a reward and sees the result move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real robots, datasets, frameworks and products.
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.

---
## Module 0 · Robot foundations  (`robo-foundations.html`)
*Animation focus: a planar two-link arm drawn from joint angles; coordinate frames attached to the links; the sense–think–act loop running at a fixed rate.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| what-is-a-robot | What a robot is: sense, think, act | the sense–think–act loop; manipulators vs mobile robots; joints (revolute, prismatic); degrees of freedom | every embodied AI system |
| frames-transforms | Coordinate frames & homogeneous transforms | rotation matrices; 4×4 transforms; composing and inverting frames; pose of an object in a gripper frame | pose estimation, grasping |
| configuration-space | Joint space & configuration space | configuration vector q; the set of reachable configurations; obstacles mapped into configuration space | motion planning |
| workspace-dof | Workspace, degrees of freedom & redundancy | DOF count; reachable workspace; redundant arms; singular poses (preview) | arm design, teleoperation |
| robot-stack | The robot software stack | sensor drivers → estimation → planning → control → actuators; real-time versus best-effort; ROS 2 nodes and topics | every deployed robot |

## Module 1 · Sensors & actuators  (`robo-sensors.html`)
*Animation focus: an encoder disc turning and producing pulses; an IMU integrating a spinning rate into an angle with drift; a stereo pair measuring disparity.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| encoders-motors | Encoders, motors & gear ratios | incremental vs absolute encoders; torque, speed and gear ratio; counts per revolution | joint sensing |
| imu | Inertial sensing: accelerometer and gyroscope | integrating angular rate; bias and drift; why an IMU alone cannot give position for long | balance, dead reckoning |
| cameras-depth | Cameras, stereo and depth sensors | pinhole model; disparity; depth = f·B/d; structured light and time-of-flight depth | grasping, navigation |
| lidar | LiDAR & range sensing | time of flight; the point cloud a scanner makes; beam pattern; range noise | mapping, autonomous driving |
| sensor-noise | Noise, bias & calibration | Gaussian noise; systematic versus random error; intrinsic and extrinsic calibration; why fusion helps | every estimator |
| force-tactile | Force, torque & tactile sensing | wrist force/torque sensors; contact detection; compliance as a sensing problem | assembly, safe contact |

## Module 2 · Kinematics & dynamics  (`robo-kinematics.html`)
*Animation focus: forward kinematics chaining links; inverse kinematics solving with two elbow solutions; the Jacobian as a stretched grid; a pendulum obeying the dynamics equation.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| forward-kinematics | Forward kinematics | chain of link transforms; a two-link planar arm worked by hand; Denavit–Hartenberg parameters | pose from joint readings |
| inverse-kinematics | Inverse kinematics | solving joint angles for a target; multiple and missing solutions; closed-form versus numerical solvers | reaching a target |
| jacobian | The Jacobian & velocity mapping | J(q) maps joint velocity to end-effector velocity; force mapping by the transpose; singularities | teleoperation, control |
| manipulator-dynamics | Manipulator dynamics | M(q)q̈ + C(q,q̇)q̇ + g(q) = τ; inertia, Coriolis and gravity terms; a one-link worked example | simulation, model-based control |
| mobile-kinematics | Mobile robot kinematics & odometry | differential drive and unicycle model; wheel odometry; non-holonomic constraints | warehouse robots, rovers |
| simulation-physics | Simulating robots | numerical integration of dynamics; contact and friction models; the fidelity versus speed trade-off | training in simulation |

## Module 3 · State estimation & mapping  (`robo-estimation.html`)
*Animation focus: a belief spreading and sharpening as measurements arrive; a Kalman estimate chasing a noisy track; a particle cloud resampling around the true pose; a pose graph snapping into shape at loop closure.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| bayes-filter | The Bayes filter | belief over state; prediction and update steps; recursive estimation | localisation |
| kalman-filter | The Kalman filter | prediction and update equations; Kalman gain; a one-dimensional worked example | sensor fusion, tracking |
| extended-kalman | The extended Kalman filter | linearisation with Jacobians; where linearisation breaks down | IMU and GPS fusion |
| particle-filter | Particle filters (Monte Carlo localisation) | particles, weights and resampling; multimodal beliefs; particle count versus accuracy | global localisation |
| occupancy-mapping | Occupancy grids & mapping | log-odds updates; ray casting through cells; mapping when the poses are known | navigation maps |
| slam | SLAM & pose graphs | joint estimation of map and trajectory; loop closure; graph optimisation | autonomous driving, drones |

## Module 4 · Perception for robots  (`robo-perception.html`)
*Animation focus: a point cloud rotated and downsampled; ICP iterations pulling one cloud onto another; a 6-DoF pose drawn as a frame on an object; a grasp rectangle sliding over an object.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| point-clouds | Point clouds & 3D geometry | point sets; surface normals; voxel downsampling; transforming a cloud by a pose | scene understanding |
| registration-icp | Registration & ICP | rigid alignment by SVD; iterative closest point; local minima and good initialisation | localisation, mapping |
| object-detection-3d | 2-D and 3-D object detection for robots | reusing 2-D detectors; 3-D boxes; instance masks; latency budget per frame | pick-and-place |
| pose-estimation | 6-DoF object pose estimation | rotation representations; PnP from keypoints; pose uncertainty | grasping |
| grasp-detection | Grasp detection | a grasp as pose, gripper width and quality score; antipodal grasps; learned grasp networks | bin picking |
| visual-servoing | Visual servoing | image Jacobian; closing the loop on image error; eye-in-hand versus eye-to-hand | camera-guided motion |

## Module 5 · Planning & motion  (`robo-planning.html`)
*Animation focus: A* expanding a frontier on a grid; RRT growing a tree around obstacles; a spline trajectory with velocity and acceleration plotted under it; an optimiser bending a path away from obstacles.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| configuration-planning | Planning as search in configuration space | obstacles and collision checking; discretised configuration grids; what a valid path is | motion planning |
| graph-search | Graph search: Dijkstra and A* | path cost; admissible heuristics; why A* is optimal; a grid worked example | route planning |
| sampling-planners | Sampling-based planning: PRM and RRT | random sampling; connectivity; probabilistic completeness; RRT* and asymptotic optimality | arm motion planning |
| trajectory-generation | Trajectory generation & time scaling | polynomial and spline trajectories; velocity and acceleration limits; time scaling | smooth motion |
| trajectory-optimisation | Trajectory optimisation | cost terms for smoothness and clearance; gradient-based optimisation (CHOMP, TrajOpt style) | humanoid and arm motion |
| task-planning | Task and mission planning | symbolic states and actions; STRIPS-style preconditions and effects; sequencing skills | multi-step manipulation |

## Module 6 · Control  (`robo-control.html`)
*Animation focus: a step response with the setpoint and output; PID terms stacked; a mass-spring-damper responding to a push; an MPC horizon rolling forward along a predicted trajectory.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| feedback-basics | Feedback control & stability intuition | error, setpoint, plant; the step response; why feedback fixes disturbances and why it can oscillate | every robot loop |
| pid | PID control | proportional, integral and derivative terms; tuning by hand; integral windup and anti-windup | motor speed, joint position |
| state-space-control | State-space models & LQR | state, input, A and B matrices; optimal gain from a quadratic cost | balancing, quadrotors |
| lyapunov-stability | Stability & Lyapunov functions | a Lyapunov function as an energy; proving convergence for a simple system | safe control |
| mpc | Model predictive control | rolling horizon; constraints; the prediction model; the computation budget | autonomous driving, legged robots |
| force-impedance | Force and impedance control | mass-spring-damper behaviour; compliance; contact tasks | assembly, safe human contact |

## Module 7 · Learning for robots  (`robo-learning.html`)
*Animation focus: a policy rollout in a simulated scene; DAgger adding expert labels on the learner's own states; a PPO clipped objective; a domain-randomised sim with many worlds.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| behaviour-cloning | Imitation learning & behaviour cloning | demonstrations as supervised data; compounding errors | teaching by demonstration |
| dagger | Interactive imitation (DAgger) | distribution shift; querying the expert on the learner's own states | robust cloning |
| rl-basics | Reinforcement learning for control | Markov decision processes; reward, return, policy and value; exploration versus exploitation | locomotion, manipulation |
| policy-gradient | Policy gradients & PPO | REINFORCE; advantage estimates; the clipped PPO objective | legged locomotion |
| sim-to-real | Simulation training & sim-to-real transfer | domain randomisation; the reality gap; system identification | learned locomotion, dexterous hands |
| reward-design | Reward design & safety constraints | reward shaping; reward hacking; constrained RL | avoiding unsafe behaviour |

## Module 8 · Foundation models for robots  (`robo-foundation.html`)
*Animation focus: a demonstration dataset grouping by task; continuous actions discretised into bins; an image and an instruction going into a model and an action chunk coming out; a diffusion policy denoising an action sequence.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| robot-data | Robot datasets & data collection | teleoperation; scale and diversity; episodes, observations and action labels | training corpora |
| action-tokenisation | Actions as tokens & action chunking | discretising continuous actions; action chunking and its latency benefit | VLA models |
| vision-language-action | Vision-language-action models | image plus instruction in, action out; RT-2 and OpenVLA as examples | language-conditioned robots |
| diffusion-policy | Diffusion policies for action | denoising action sequences; representing multimodal behaviour | learned manipulation |
| llm-planners | Language models as planners | decomposing a task into skills; grounding plans in what the robot can do | task-level planning |
| embodied-evaluation | Evaluating learned robot policies | success rates with confidence intervals; simulation versus real benchmarks; held-out scenes | benchmarking |

## Module 9 · Safety, deployment & operations  (`robo-production.html`)
*Animation focus: a hazard chain from cause to harm; a watchdog timer that trips and stops a motor; a joint-limit filter clipping a learned action; a fleet dashboard of telemetry.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| safety-standards | Safety standards & risk assessment | hazard analysis; industrial (ISO 10218) and personal-care (ISO 13482) standards; protective stops | legal and physical safety |
| runtime-monitoring | Runtime monitoring & fault handling | watchdogs; sensor fault detection; graceful degradation | field reliability |
| safe-control-filters | Safety filters & constraint enforcement | joint and speed limits; control barrier functions (intuition) | safe learned policies |
| hardware-in-loop | Testing: simulation, hardware-in-the-loop & logs | the test pyramid; log replay; regression tests for policies | release process |
| edge-compute | Onboard compute & latency budgets | real-time control loops; GPU versus microcontroller; network latency | deployment |
| fleet-operations | Fleet operations & maintenance | telemetry; over-the-air updates; wear and calibration drift; incident review | running robots at scale |
