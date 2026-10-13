# Literature Review

All 17 papers from the project corpus (`project I/`), each with its
contribution, the specific limitation that matters for this project, and what
we do about it.

**How to read the "Our fix" column.** Not every paper is a competitor. Some
are direct baselines we extend, some are methods we borrow machinery from, and
some are surveys used for positioning. Where a paper is background rather than
something to "fix," the column says what we take from it instead — inventing a
fix for a survey would be dishonest padding.

Relevance is marked: **[BASELINE]** = directly extended · **[METHOD]** =
machinery borrowed · **[COMPETITOR]** = closest alternative approach ·
**[CONTEXT]** = background / positioning.

---

## A. Core baselines — the systems this project extends

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **Eureka: Human-Level Reward Design via Coding LLMs** — Ma et al., ICLR 2024 **[BASELINE]** | LLM writes executable reward code from unmodified environment source; evolutionary search over K candidates; **reward reflection** via per-component statistics at training checkpoints; ground-truth fitness `F` ranks candidates. Beats human experts on 83% of 29 tasks. | Reflection reports **what** each reward component did, never **why** the policy failed. An inverted-sign reward bug and a simulator contact mismatch produce near-identical feedback, and the LLM has exactly one tool — rewrite the reward — which is the wrong action for the second case. Sim-only; no DR; no transfer mechanism. | Inherit the entire loop **unmodified** (deliberately — any change weakens the ablation) and add a physics-residual diagnosis channel that distinguishes the two failure causes *before* any edit is issued. |
| **DrEureka: Language Model Guided Sim-To-Real Transfer** — Ma et al., RSS 2024 **[BASELINE]** | Adds LLM-generated domain randomization. **RAPP** sweeps each physics parameter against an initial policy to find feasible bounds; the LLM then selects ranges within them. Safety expressed as a natural-language instruction rather than a bolted-on penalty term. Zero-shot sim-to-real on Go1 and LEAP Hand. | DR ranges are **fixed once generated**, calibrated against `π_initial` — but range feasibility is a property of the *current* policy: a late robust policy tolerates far wider randomization. The authors name static DR and the absence of a policy-selection mechanism as future work. | **Adaptive DR** (Contribution 1): re-evaluate ranges every N iterations from live policy performance, driven by the same feedback channel that edits rewards, closing the loop DrEureka leaves one-shot. |

---

## B. LLM-driven reward generation

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **REvolve: Reward Evolution with LLMs using Human Feedback** — Hazra et al., 2025 **[COMPETITOR]** | Genuine evolutionary operators (mutation, crossover, island-model migration) instead of Eureka's greedy iterate-and-discard. Human pairwise rollout comparisons → Elo fitness, plus qualitative NL feedback. Rewards stay interpretable as code. | Requires **humans in the loop** for fitness, which does not scale and cannot run unattended. No domain randomization, no sim-to-real, no physics signal. | Keep the automated benchmark fitness (no human labelling). Our feedback enrichment is a *physics* signal rather than human preference — orthogonal to REvolve, and combinable in principle. |
| **Generative AI in Robotic Manipulation: A Survey** — Zhang et al., 2025 **[CONTEXT]** | Three-layer taxonomy of generative models in manipulation: foundation layer (data + **reward generation**), intermediate (language/code/visual/state), policy layer (grasp + trajectory). | Survey. Notes that LLM reward generation exists, but identifies no mechanism for **verifying** generated rewards or **attributing** their failures. | Positions our work inside its "reward generation" layer; our injection-validation methodology directly addresses the verification gap it identifies. |
| **Reward Engineering and Shaping in RL: A Comprehensive Overview** — Ibrahim et al., IEEE Access 2024 **[CONTEXT]** | PRISMA review of 55 papers. Taxonomy of reward shaping: potential-based shaping, policy-gradient reward design (PGRD, LIRPG), robustness methods; explicit treatment of **reward hacking**. | Survey, not a method. Predates LLM reward generation and offers no failure-attribution mechanism. | Used to position the reward-side contributions. Its reward-hacking analysis directly motivates our rule that **fitness `F` is never augmented with physics terms** — otherwise candidate selection optimizes for apparent physical tidiness rather than task success. |

---

## C. Physics-informed learning — where the PINN machinery comes from

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **Physics-Informed Neural Networks** — Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 2019 **[METHOD]** | The foundational PINN. Encode a PDE residual via automatic differentiation as a loss term; composite objective `MSE_u + MSE_f` over data and collocation points. | Generic PDE solving — not robot dynamics, no contact, no RL. The original composite loss is **unweighted**, which does not suit a multi-term robotics objective. | Adopt the residual-as-loss principle, instantiated on manipulator equations of motion with a contact term. λ weights are made **explicit tunable config** rather than assumed equal. |
| **Physics-Informed Reward Shaped RL Control of a Robot Manipulator (PI-DDPG)** — Fareh et al., Ain Shams Eng. J. 2025 **[METHOD]** | Physics-informed RL on a 4-DOF manipulator: inverse-dynamics-generated replay samples, **Lyapunov-based reward shaping** `R' = R + λ·R_lyap`, adaptive exploration noise. 20-30% tracking improvement, ~3× faster convergence. | Dynamics model has **no friction or contact term at all** — pure rigid-body free-space motion. Reward is fixed and hand-designed. Only 4 DOF. λ is named but never given a value. Physics is used to *shape*, never to *diagnose*. | Borrow the reward-shaping functional form; extend the dynamics to include contact; and critically, use the physics signal as an **attribution** signal routed to DR rather than folded into the reward (see PIRL row for why that distinction matters). |
| **SOPDNA: Self-Orchestrating Physics-Discovering Neural Architecture** — Singla et al., Sci. Reports 2026 **[METHOD]** | PINN + robotic process automation + digital twins. Composite loss `L = λ_d·L_data + λ_p·L_phys + λ_c·L_ctrl`; **learns latent friction/disturbance parameters (δ)** via symbolic regression; automated retraining on anomaly detection. | Not RL — no reward, no policy optimization. Friction enters as a **continuous lumped term** inside continuous dynamics; there is no contact/no-contact switching or impact modelling. λ values are never numerically specified. | Adopt the composite loss **shape**, and the latent-parameter idea — which tells us *which* physical parameter is mismatched, exactly what the adaptive-DR controller needs. Add **contact-phase gating** for the discontinuity SOPDNA does not model. |
| **Zero-Shot Wireless Indoor Navigation through Physics-Informed RL (PIRL)** — Yin et al., ICRA 2024 **[CONTEXT]** | Physics-informed **reward shaping** using domain physics (link-state monotonicity, angle-of-arrival following, SNR increase); physics laws persist across environments, enabling zero-shot generalization without fine-tuning. | Physics is electromagnetic propagation, not manipulator dynamics; navigation, not manipulation. Physics enters as a **reward bonus** — precisely the pattern our routing rule exists to prevent. | Instructive counter-example. We route physics to **diagnosis and DR, never into the reward**, so the reward cannot absorb simulator artifacts that will not transfer to hardware. |
| **PIN-WM: Physics-Informed World Models for Non-Prehensile Manipulation** — Li et al., 2025 **[COMPETITOR]** | Identifies 3D rigid-body dynamics (mass, inertia, friction, restitution) end-to-end from vision using differentiable physics (LCP solver) + 2D Gaussian splatting. **"Physics-aware digital cousins"** randomize *around identified* parameters rather than over a wide blind prior. Real2Sim2Real without fine-tuning. | Non-prehensile pushing and flipping — **not multi-finger in-hand manipulation**. No LLM reward generation. Its DR is principled but still essentially **one-shot** around identified means. | The closest competitor to our DR contribution, and cited as such. We differ in that our ranges **adapt during training** from policy performance, and we add reward generation plus failure attribution — neither of which PIN-WM has any notion of. |

---

## D. Reward shaping for dexterous hands

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **Reward Shaping in RL for Robotic Hand Manipulation** — Deng, Dong & Liu, Neurocomputing 2025 **[CONTEXT]** | Custom 8-DOF D-Hand hardware; combines Lyapunov-based and variance-aware reward terms; uses domain randomization for transfer; achieves sim-to-real valve rotation after ~2 h of simulated training. | Reward is **hand-designed** — the exact bottleneck Eureka removes. DR is standard and non-adaptive. Single task, no failure diagnosis. | Automates the reward design they perform manually, and makes their fixed DR adaptive. Useful as evidence that Lyapunov-style shaping works on real dexterous hardware. |
| **Reward Shaping via Expectation Maximization** — Deng, Liu & Dong, Neurocomputing 2024 **[CONTEXT]** | EM-based search over hyper-parameters blending Lyapunov and variance-aware reward terms (RBF-fitted reward matrix); demonstrated on dexterous valve-turning. | Optimizes the **weights of a fixed reward structure** — it cannot invent new reward terms. No sim-to-real, no physics, no DR. | Sharpens the contrast: our search operates over reward **code/structure**, not just scalar weights, which is the qualitative difference LLM generation buys. |

---

## E. Dexterous manipulation & simulation infrastructure

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **DexDeform: Dexterous Deformable Manipulation with Human Demos and Differentiable Physics** — Li et al., ICLR 2023 **[CONTEXT]** | Multi-finger Shadow Hand manipulating deformable objects; ~10 human teleop demos per task, learned skill abstractions, differentiable-physics trajectory optimization; 6-task suite; beats PPO/BC/DAPG baselines. | **Simulation only** — no sim-to-real, no DR. Requires human demonstrations. Deformable-object focus rather than rigid in-hand reorientation. | No demonstrations required in our pipeline (the reward is LLM-generated). Cited as background for differentiable physics + dexterous manipulation, and as evidence that skill structure matters in contact-rich regimes. |
| **GRIP: General Robotic Incremental Potential Contact Simulation Dataset** — Ma et al., IROS 2025 **[CONTEXT]** | GPU-parallelized Incremental Potential Contact simulation (up to 48× speedup, intersection/inversion-free); 1,200 objects, 100K grasp poses; soft UMI grippers **and rigid LEAP Hands**; deformable-rigid coupled grasping. | A **dataset and simulator for grasp synthesis**, not a policy-learning or reward-design method. Domain randomization appears only as data augmentation. | Not used as training data — our pipeline generates data online. Relevant as evidence that high-fidelity LEAP Hand contact simulation is an active problem, and as an object source if the work is later extended beyond single-object cube reorientation. |

---

## F. Sim-to-real surveys — used for positioning

| Title | Their contribution | Issue | Our fix |
|---|---|---|---|
| **A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models** — Da et al., 2025 **[CONTEXT]** | Organizes sim-to-real by MDP element (observation, action, transition, reward); includes an explicit branch for **foundation-model / LLM-based reward design**. | Survey. Its own taxonomy confirms that no existing work unifies adaptive DR with physics-based failure attribution. | The umbrella positioning reference — its taxonomy is precisely where we locate our contribution, and its coverage supports the novelty claim. |
| **A Survey on Sim-to-Real Transfer Methods for Robotic Manipulation** — Pitkevich & Makarov, SISY 2024 **[CONTEXT]** | Manipulation-specific survey: domain randomization, domain adaptation, imitation learning, meta-learning, knowledge distillation, system identification, NeRF/3D-GS representations. | Survey. Covers **standard** DR, not adaptive DR; no PINNs; no LLM-in-the-loop. | Background for the manipulation sim-to-real setting; corroborates that adaptive DR coupled to reward generation is unaddressed in this literature. |
| **Sim2real for RL Driven Next Generation Networks** — Li et al., 2022 **[CONTEXT]** | Position paper on the sim2real gap for RL in 5G/O-RAN networks; emphasises high-fidelity digital twins for training and validation. | **Telecom domain, not robotics.** No manipulation, no manipulator dynamics, no DR specifics. | Peripheral. Retained only as general sim2real framing; not load-bearing for any claim in this project. |

---

## What the corpus establishes

Reading across all 17 papers, three things are true:

1. **Every individual ingredient exists.** LLM reward generation (Eureka,
   REvolve), automated DR (DrEureka, PIN-WM), and physics-informed networks
   for manipulator dynamics (PI-DDPG, SOPDNA) are all established.
2. **No paper combines them.** Nothing in the corpus couples LLM reward
   generation with adaptive domain randomization *and* physics-based failure
   attribution.
3. **No paper validates a failure-attribution signal against ground truth.**
   This is the largest open methodological gap, and the one our injection
   calibration study addresses directly.

A caveat worth stating rather than hiding: **neither PI-DDPG nor SOPDNA — the
two closest physics-informed manipulator works — models discontinuous
contact.** Since in-hand manipulation is contact-dominated, the physics
residual's reliability in exactly the regime that matters most is an open
question, not a solved one. This is why contact-free and contact-phase
residuals are reported separately, and why the injection AUC gate runs before
the full router is built.
