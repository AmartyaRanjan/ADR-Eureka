# 01 — Project background and goals

## 1. Where the project started

Two documents were supplied at the start:

1. **A Stage-1 progress report** on a 2-DOF planar arm in Isaac Lab, trained with PPO under two rewards: a human-designed ("vanilla") reward and a reward discovered by the Eureka LLM search. Its central claim: the Eureka reward reaches the goal region (≈ 5 cm) quickly but then stops improving, so fine precision collapses (≈ 98 % at 5 cm but ≈ 18–19 % at 2 cm and ≈ 4–5 % at 1 cm), while the human reward reached 98.36 % at 2 cm and 79.74 % at 1 cm.
2. **A Stage-2 roadmap, "Physics-Informed Adaptive Domain Randomization"**, proposing: Eureka → RAPP → PINN → adaptive DR, tested through an experimental matrix of eight rungs (A–H).

The standing instructions were: implement and run the roadmap; **cite repository URLs throughout**; **keep evaluating with identical metrics so results remain directly comparable**; **maintain a proper log**. In a later clarification the scope was set to: test on the 2-DOF goal-marker task for fast iteration, *and in parallel* on the LEAP Hand on the task it is normally trained for, so the configuration could be taken straight from the upstream project.

## 2. Why this problem matters (the scientific motivation)

- LLM-designed rewards (Eureka) optimise what a *success check* rewards. A large success bonus makes a coarse threshold dominate the return, removing any economic reason to refine from centimetres to millimetres. Whether this is a reward-design problem or an environment/physics limit is not obvious from the report alone.
- Domain randomization (DR) is the standard route to robust sim-to-real policies, but fixed randomization ranges are either too narrow (no robustness) or too wide (training collapses). DrEureka's RAPP finds the physics region a policy tolerates; **adaptive DR** widens ranges as the policy improves (as in OpenAI's Rubik's-cube ADR).
- The roadmap's contribution hypothesis: *a physics-informed observer could tell the adaptive controller where the policy is least certain about the physics, making the curriculum more efficient than blind or success-only widening.*

## 3. The ablation matrix (rungs)

| Rung | Reward | DR | RAPP envelope | PINN | Question it isolates |
|---|---|---|---|---|---|
| A | human | none | — | — | baseline, human reward |
| B | Eureka | none | — | — | baseline, discovered reward; A vs B = reward effect |
| C | human | static (fixed in-distribution ranges) | — | — | does static DR matter for the human reward? |
| D | Eureka | static, RAPP-bounded in-distribution box | yes | — | static DR for the discovered reward; B vs D = DR effect |
| E | Eureka | adaptive, wide arbitrary bounds | **no** | — | does RAPP's ceiling itself matter? |
| F (F4) | Eureka | adaptive, precision-gated | yes | — | adaptive DR vs static (D vs F4) |
| G | Eureka | adaptive + observer-scaled step | yes | frozen observer | does the physics observer help the controller? |
| Gp | Eureka | as G but multipliers shuffled across dimensions | yes | placebo | does G's behaviour come from the *information* or just from the changed schedule? |

(The roadmap also lists a rung H; it was out of scope.)

**Why a placebo (Gp) was added:** a PINN-informed curriculum that scores better than F4 could be better for reasons unrelated to physics information (e.g., simply a different step-size schedule). The shuffled control uses *identical* multiplier values but assigns them to the wrong dimensions, so any gain that survives shuffling is not about information.

## 4. The two tracks

| Track | Platform | Task | Role |
|---|---|---|---|
| A | 2-DOF planar arm (built from scratch here) | Reach a randomly placed goal marker within 10 s | Cheap, fast (≈ 10–30 min/run) testbed for RAPP → PINN → adaptive DR; supports many seeds |
| B | LEAP Hand (16 DoF) | `Isaac-Reorient-Cube-Leap` — in-hand cube reorientation, the task upstream ships trained | Real platform, stock upstream configuration, so settings can be copied from the published project |

Track A has an analytic ceiling (IK + PD controller), closed-form dynamics, and fast turnaround; Track B is the realistic but slow case (≈ 1 h per 500-iteration run).

## 5. Constraints that shaped every decision

- **Shared GPU.** One NVIDIA L4 (23 GB) shared with other users' jobs (vLLM, web services, image generation). Rule adopted: at most two concurrent training runs, usually one; experiments scaled to 2048 envs where possible; queues wait for free memory instead of failing.
- **Reproducibility over speed.** Metrics frozen at version 1.0.0; every protocol change logged; single-seed results never presented as findings (a rule that was initially broken and then enforced after the multi-seed corrections).
- **Secrets.** A Gemini API key was pasted into chat during setup; it was stored only in a gitignored, chmod 600 `.env` and never committed or transmitted. The user was advised to rotate it.

## 6. What "done" was defined as

For Track A: the full A–G matrix at n ≥ 5 seeds with three evaluation distributions, an honest statistical comparison, and a documented explanation of every unexpected number. For Track B: a no-ADR baseline, native ADR, and a warm-started/RAPP-style variant at ≥ 3 seeds under an identical frozen-difficulty evaluation. Track A is complete apart from one re-run; Track B is in progress (`11_leap_hand_track_b.md`).
