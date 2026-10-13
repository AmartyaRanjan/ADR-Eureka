# 02 — Infrastructure and environment setup

## 1. Environment of record

| Component | Version | Source |
|---|---|---|
| GPU | 1× NVIDIA L4 (23 GB), driver 595.84 (shared with other users) | — |
| OS / Python | Ubuntu 26.04, Python 3.11 (`../env_isaaclab`) | — |
| Isaac Sim | 5.1.0.0 (pip) | <https://developer.nvidia.com/isaac/sim> |
| Isaac Lab | 2.3.0 @ `3c6e67bb5` (2025-10-28) | <https://github.com/isaac-sim/IsaacLab> |
| rl_games | 1.6.1 | <https://github.com/Denys88/rl_games> |
| torch | 2.7.0+cu128 | — |
| IsaacLabEureka | `716657b`, locally patched for Gemini | <https://github.com/NVlabs/IsaacLabEureka> |
| LEAP_Hand_Isaac_Lab | `c576e6a` (2025-09-04) + one-line 2.3 compatibility patch | <https://github.com/leap-hand/LEAP_Hand_Isaac_Lab> |
| RAPP method | — | <https://github.com/eureka-research/DrEureka> |

## 2. Infrastructure problems and fixes (all were prerequisites for any run)

| # | Problem | Why it happened | Fix | Footprint |
|---|---|---|---|---|
| 1 | `import isaaclab` failed | The venv's editable installs pointed at `/home/.../projects/I`, but the project had been moved to `/data/home/.../projects/I` | Symlink old path → new path | host filesystem (reversible) |
| 2 | URDF→USD importer missing `libxml2.so.2`, `libicu*.so.73` | Absent on Ubuntu 26.04 | `.libshim/` with symlinks to the user's miniconda copies, used only by `scripts/convert_asset.sh`; the converted USD is cached so train/eval need no shim | `pi_adr` only |
| 3 | LEAP `pip install -e` failed | `setup.py` reads `keywords`, missing from its `extension.toml` | Not installed; `PYTHONPATH=source/LEAP_Isaaclab` instead | none |
| 4 | LEAP `train.py` import error | imports `dump_pickle`, removed from `isaaclab.utils.io` in Isaac Lab 2.3 | Use `pi_adr/scripts/train.py`, derived from Isaac Lab 2.3's own `train.py` plus task registration | `pi_adr` only |
| 5 | LEAP `randomize_actuator_gains` event failed | `EventCfg` lacked `operation`, now required | Added `"operation": "abs"` (the 2.1 default the config was written against) | 1 line in `leap_hand_env_cfg.py` |
| 6 | Many later failures looked like crashes | Shared-GPU memory pressure; upstream early stop (see doc 11) | Queue logic: wait for resources, per-attempt logs | scripts |

**Why these were recorded rather than quietly fixed:** each changes what "the stock configuration" means. A reader reproducing the work must apply exactly these fixes.

## 3. The 2-DOF benchmark — built from scratch

No 2-DOF task existed in any repo on the machine, so it was written from the Stage-1 report's constants as an Isaac Lab `DirectRLEnv` (`source/pi_adr/tasks/two_dof_reach/`, asset `assets/two_dof_arm.urdf`).

| Property | Value | Why |
|---|---|---|
| Links / masses | 0.5 m, 0.5 m / 1.0 kg, 0.5 kg | From the report |
| Joints | 2 revolute about Z, **continuous** (see doc 04, defect 3) | ±π hard limits capped every controller at ≈ 69 % |
| Actuation | torque τ = 20·clip(a,−1,1) Nm; |q̇| ≤ 5 rad/s; joint damping 0.05 | From the report |
| Observation | 8-D `[q1,q2,q̇1,q̇2,x_t,y_t,x_t−x_ee,y_t−y_ee]` (angles wrapped to [−π,π]) | From the report |
| Rates | 120 Hz physics, 30 Hz control (decimation 4), 10 s episodes = 300 steps | From the report |
| Targets | r ~ U(0.20, 0.85) m, θ ~ U(0, 2π); success radius 5 cm | From the report |
| Rewards | R_vanilla = −d + 5·I(d<0.05) − 0.005‖a‖²; R_Eureka = e^(−10d) + 100·I(d<0.05) − 0.01‖q̇‖² − 0.1 | Report eqs. 7 and 8 |
| PPO | MLP 8→256→128→64 ELU, lr 1e-3 adaptive, γ 0.99, λ 0.95, clip 0.2, entropy 0.005, KL 0.01, 8 epochs, 4 minibatches, horizon 24 | Report hyperparameters |
| Parallelism | 4096 envs (Session 1); 2048 envs (Session 2 onward, shared GPU) | — |

The horizontal-plane arm means gravity produces no joint torque (g(q) = 0), so the dynamics are M(q)q̈ + C(q,q̇) + F(q̇) = α⊙τ_cmd, with α the per-joint actuator effectiveness.

**Domain randomization** (`DomainRandCfg`) draws per-environment: Coulomb friction, joint-damping scale, actuator effectiveness α, and link-mass scale at each reset. A design rule discovered the hard way (doc 04): nominal values must equal the URDF's, and **DR must be verified by reading values back from the simulator**, not assumed.

## 4. Validation of the environment (`scripts/validate_env.py`, 10/10 passed)

| Check | What it tests | Result |
|---|---|---|
| A | Rates: 30 Hz control, 120 Hz physics, 300 steps/episode | `step_dt = 0.033333 s`, 300 steps |
| B | Simulated end effector equals analytic forward kinematics | max error **0.000 mm** |
| C | Zero torque ⇒ no drift (planar, g = 0) | |q| = |q̇| = 0 after 1 s |
| D | `a = +1` ⇒ +20 Nm on joint 1, positive rotation | applied 20.000 Nm |
| E | Joint velocity within ±5 rad/s under saturated torque | peak 5.101 rad/s (limit engages) |
| F | Targets inside the 1.0 m reach | r ∈ [0.202, 0.849] m |
| G | Observation layout | `(N,8)`, all fields match |
| H | Vanilla and Eureka rewards equal report eqs. 7 and 8 | max deviation 0.00 |
| I | DR actually reaches the simulator and differs per environment | read-back friction std 0.054 |
| J | τ = 20·a·α in the simulator | max deviation 0.0000 Nm (against the closed-form coupled dynamics) |

Check I is deliberate protection against the roadmap's "silent randomizer" failure mode: a DR that is configured but never applied.
Two further validators: `scripts/validate_dynamics.py` (closed-form dynamics vs PhysX: **2.27 %** mean relative error on q̈, **0.036 Nm** on recovered torque) and `scripts/pd_baseline.py` (the precision ceiling, doc 05).

**Three of these checks were themselves wrong at first** (D and J assumed an idealised torque law ignoring drive damping, Coulomb friction, and the fact that per-joint effectiveness stays coupled through M(q), so q̇ is *not* componentwise proportional to α). They were corrected against the closed-form coupled dynamics — an instance of "validate the validator".

## 5. Orchestration infrastructure

- `scripts/train.py` — Isaac Lab-derived wrapper with Hydra overrides (e.g. `env.domain_rand.joint_friction_range=[0.0,0.25]`, `agent.params.config.entropy_coef=0.02`, `env.enable_adr=false`). Verified by inspecting the dumped `params/env.yaml`, not just by exit code 0.
- `scripts/run_adaptive_dr.py` — outer-loop controller orchestrator (chunks of 100 PPO iterations, per-dimension evaluation between chunks). **Implementation finding:** rl_games `--checkpoint` resume restores `epoch_num` and treats `max_epochs` as an absolute target (`a2c_common.py:639`), so `--max_iterations` must be *cumulative*. The first version used a per-chunk value and would have silently trained zero iterations from the second chunk on; a 2-outer-step, 3-iteration smoke test caught it before any real run.
- `scripts/leap_queue.sh` — sequential resumable LEAP queue with flock, GPU/RAM gating, resume-from-checkpoint (doc 11).
- Cron (`@reboot` and every 15 min) restarts the LEAP queue after a reboot/crash; the lock prevents overlap.

## 6. Throughput and wall clock (for planning, not as results)

Vanilla 5000 iterations ≈ 79 min, Eureka 500 ≈ 8 min, Eureka 1500 ≈ 24 min on one L4 (Session 1, partly concurrent so not a clean efficiency metric). 800-iteration runs at 2048 envs ≈ 10–15 min each; an adaptive-DR run (8 outer steps with per-dimension evals) ≈ 30 min; one LEAP 500-iteration run ≈ 55–65 min at ≈ 20 k frames/s.
