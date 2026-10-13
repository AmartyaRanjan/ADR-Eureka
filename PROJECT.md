# Physics-Informed Adaptive Reward Generation for Sim-to-Real Dexterous Hand Control (v2)

Revised 2026-07-18. Supersedes the v1 description (which mislabeled the platform as a "16-joint robotic arm" — DrEureka's 16-DOF platform is the LEAP Hand, an anthropomorphic dexterous hand).

## 1. Overview

An LLM-driven reward generation and policy training pipeline for dexterous in-hand manipulation, extending Eureka / DrEureka in three directions:

1. **Adaptive domain randomization** — DR parameters are re-tuned during the reward-iteration loop based on policy performance, replacing DrEureka's static one-shot DR (which DrEureka's own limitations section flags as future work).
2. **Physics-informed reward diagnosis** — a PINN trained with a physics-residual loss (manipulator + object equations of motion, soft-contact relaxation) tells the LLM *whether a failure is physical* (dynamics/contact mismatch) or a reward-logic error.
3. **Learned failure-mode summarizer** (optional rung) — a lightweight MLP replacing Eureka's hand-crafted reward-reflection statistics with learned failure descriptors.

## 2. Platforms — two hand morphologies, one system

| | Shadow Hand (Eureka) | LEAP Hand (DrEureka) |
|---|---|---|
| DOF | 24 joints, 20 actuated | 16 joints, 16 motors |
| Tasks | Eureka's Isaac Gym + Bidexterous suite (multi-task) | In-hand cube rotation (DrEureka env) |
| Deployment | Simulation only | Simulation + physical hand (~$3K build) |
| Role | Reward-generation + PINN-diagnosis ablations, injection/placebo studies (task diversity; tasks with contact-free phases give the physics residual a fair test) | Adaptive-DR pillar + end-to-end zero-shot real-world transfer (the claim adaptive DR exists for) |

Cross-platform replication of each component's contribution is itself a claim: it answers the "single-task, does it generalize?" objection. The PINN's physics-residual loss is instantiated per platform from each hand's URDF-derived dynamics (mechanical duplication, not new research).

Note: the LEAP policy observes only 16 joint angles + GRU-encoded history (no cube pose), per DrEureka. The PINN reads privileged simulator state (cube pose/velocity) — legitimate, since it only feeds the LLM between training iterations and never runs on the real robot. Real-world validation of the physics-consistency signal itself would require cube tracking and is out of scope; stated explicitly as a limitation.

## 3. Architecture

### 3.1 Reward generation loop (Eureka, unmodified)
Environment source + task description → LLM generates K candidate rewards (evolutionary search) → parallel PPO training in Isaac Gym → reward reflection + task fitness → next iteration. Fitness functions are inherited unchanged from the published benchmarks (e.g. consecutive-success count for reorientation; rotations per 20 s for LEAP cube rotation). **The fitness is never augmented with physics-consistency terms.**

### 3.2 Adaptive DR
DR ranges re-evaluated every N reward iterations, widened when the policy is robust and narrowed when it fails to learn (ADR pattern, coupled to the same feedback channel driving reward edits). Must address DrEureka's finding that intra-training adaptation was unnecessary *for their tasks*: the hypothesis is that adaptation matters precisely where feasible DR ranges are narrow (in-hand manipulation), unlike locomotion.

### 3.3 PINN with routing protocol (anti-shunting design)
- MLP; inputs: DR parameter state + robot/object state. Loss: supervised state-prediction error + differentiable physics residual (Lagrangian EOM; soft-contact relaxation for contact terms).
- **Contact gating**: residual computed everywhere, but the diagnostic signal weights low-contact segments; contact-phase residuals reported separately (soft-contact relaxation makes in-contact residuals partly artifact).
- **Two-stage routing** — the rule that keeps the PINN load-bearing:
  1. LLM first classifies the failure from reflection + PINN signal: `reward-logic` vs `physics-mismatch` (or both, with attribution).
  2. Reward-logic diagnoses may only trigger reward code edits; physics diagnoses may only trigger DR adjustments. The LLM is prohibited (prompt contract + code check) from writing reward terms referencing DR parameters, physics residuals, or dynamics-mismatch avoidance.
  This prevents the LLM from absorbing physics errors into reward space, which would zero out the PINN's measurable contribution.
- Start frozen (pre-trained), move to online retraining only after validation in isolation.

### 3.4 Learned failure-mode summarizer (optional rung)
An MLP consuming full-rate sim telemetry (joint torques, tracking error, contact events, DR values, at the simulator's native control rate across parallel envs), **trained to predict episode outcome plus failure-phase localization** (hindsight labels; success labels are free from the fitness function). Intermediate representations decode into per-checkpoint failure descriptors (success rate, per-finger attribution, trajectory-phase of failure) feeding the reflection channel. Hypothesis under test: learned descriptors carry actionable signal that hand-crafted component statistics miss. (The v1 "120 fps real-time" framing is retired — the contribution is learned compression, not frequency.)

### 3.5 Policy optimization
PPO, published hyperparameters per benchmark, unmodified.

## 4. Ablation & validation design

Rung ladder (run on both platforms in sim; real-world eval on LEAP only):

| Rung | Configuration | Isolates |
|---|---|---|
| 1 | Eureka/DrEureka reproduction (static DR) | Baseline control |
| 2 | + adaptive DR | Adaptive DR alone |
| 3 | + PINN routing | Physics diagnosis, marginal |
| 4 (optional) | + learned summarizer | Learned vs hand-crafted reflection |

Mandatory controls:

- **Injection calibration (run FIRST, before building rung 3).** Into a working rung-1 setup, separately inject (a) known reward bugs (sign flip, wrong scale, missing term) and (b) known dynamics perturbations (mass ×1.5, friction ×0.3, added damping). Report the PINN residual's discrimination AUC between (a) and (b). If it fails on injected ground truth, fix the PINN before proceeding — rung 3 is otherwise uninterpretable. Standalone contribution: no prior work validates physics-failure attribution this way.
- **Placebo control at rung 3**: identical-budget arms with (i) real PINN signal, (ii) shuffled signal, (iii) no signal. (i) ≈ (ii) falsifies the contribution.
- **Routing audit**: log diagnosis → edit-type per iteration; confusion matrix against injection ground truth.
- **2×2 placebo at rung 4** (real/shuffled PINN × real/shuffled summarizer) to separate the two channels.
- **Reward hygiene**: per-component reward dicts (Eureka convention) + check that no component dominates the total by >~10× at convergence.
- **Transfer eval**: LEAP — zero-shot on the physical hand (DrEureka protocol, rotations per 20 s). Shadow Hand — transfer to a held-out "pseudo-real" sim config (perturbations sampled once, hidden from the whole loop, outside all DR ranges), honestly labeled sim-to-OOD-sim.

## 5. Risks

- **Contact dynamics in the physics residual** — primary technical risk, maximally exposed on LEAP cube rotation (continuous contact). Mitigations: contact gating, Shadow Hand tasks with free-flight phases as the fair test, injection study as early kill-switch.
- **Compounding adaptivity / moving-target overfit** — freeze DR + PINN state before final transfer; held-out real rollouts on LEAP.
- **Credit assignment** — handled by the rung ladder + placebo arms + routing audit.
- **Scope** — two platforms double sim experiments (compute is not a constraint: L4/H100 access) and add the LEAP hardware build; rung 4 is explicitly cuttable.

## 6. Novelty vs prior work

Closest works in the literature collection (`project I/`): PIN-WM (physics-informed world model, "physics-aware digital cousins" DR — study closely as the principled-DR competitor), PI-DDPG ("PINN (2).pdf", PINN + manipulator + reward shaping), REvolve (LLM reward evolution), SOPDNA. None combines LLM reward generation + adaptive DR + physics-failure diagnosis; none validates physics-attribution via injection ground truth. Missing from the collection and required reading: OpenAI ADR (Akkaya et al., Rubik's cube) for adaptive-DR threshold mechanics.
