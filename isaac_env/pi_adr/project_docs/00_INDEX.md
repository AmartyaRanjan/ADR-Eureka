# Project documentation index — Physics-Informed Adaptive Domain Randomization

**Written for:** the project owner and any co-author or reviewer who needs to understand exactly what was done, why, what was found, and how much each finding can be trusted.
**State of record:** 13 October 2026. The 2-DOF track is complete except one re-run in progress (G/placebo seeds 3–6, see `10_ablation_matrix_A_to_G.md`). The LEAP Hand track is **complete** at 3 seeds (see `11_leap_hand_track_b.md`).
**Companion records (older, chronological):** `../EXPERIMENT_LOG.md` (append-only, every run), `../PROJECT_STATUS_AND_ROADMAP.md`, `../RESULTS_2026-09-16.md`, `../CODE_AND_MATH_COMPILATION.md`. These MDs consolidate and explain them; where they disagree, **`EXPERIMENT_LOG.md` entries dated 11–13 Oct 2026 and these documents are authoritative**, because the older text predates several corrections.

---

**New to the project? Read `20_START_HERE_project_explained_from_scratch.md` first** (a plain-language guided tour that assumes only basic RL knowledge).

## 1. The project in one paragraph

Stage 1 (a prior report) claimed that a reward discovered by an LLM (Eureka) reaches a goal quickly but loses fine precision, while a human-designed reward keeps millimetre precision. Stage 2 (this work) proposed a pipeline — **Eureka reward → RAPP physics envelope → PINN physics observer → adaptive domain randomization (DR)** — and an ablation matrix of eight configurations (rungs A–G). We built an independent 2-DOF reaching benchmark in Isaac Lab, validated it against analytic physics, ran the matrix with 5 seeds per rung, built a physics observer and a PINN-informed adaptive DR controller, and began a parallel LEAP Hand study. The honest outcome so far is a **carefully controlled set of mostly negative results**: no reward or DR variant reliably beat the simple baselines on tail/precision metrics; Stage 1's headline reward comparison does not hold at matched budget; the PINN-informed controller did not beat a shuffled-information placebo. The durable contributions are the validated benchmark, the evaluation/statistics protocol, the defect discoveries, the RAPP sensitivity map, the PINN identifiability analysis, and the discovery that LEAP already ships adaptive DR.

## 2. Reading order

| # | File | Answers |
|---|---|---|
| 01 | `01_project_background_and_goals.md` | What was asked, what the rungs A–G are, why two tracks |
| 02 | `02_infrastructure_and_environment_setup.md` | Hardware/software, install fixes, how the 2-DOF env was built and validated |
| 03 | `03_evaluation_protocol_and_statistics.md` | The frozen metrics, eval protocol, and how we separate real effects from anomalies |
| 04 | `04_defects_found_and_fixed.md` | Every silent error found (including late ones) and how each was detected |
| 05 | `05_stage1_baselines_and_reproduction.md` | Vanilla vs Eureka, the IK+PD ceiling, DR robustness of the baselines |
| 06 | `06_rapp_feasibility_sweep.md` | The RAPP physics envelope |
| 07 | `07_vanilla_exploration_collapse.md` | Why the human-reward baseline underperformed Stage 1, V1/V2 probes, n=5 recheck |
| 08 | `08_pinn_physics_observer.md` | Observer design, four excitation-protocol iterations, identifiability findings |
| 09 | `09_adaptive_dr_development_F_and_E.md` | Controller design, F1→F4, rung E, multi-seed corrections |
| 10 | `10_ablation_matrix_A_to_G.md` | The final 5-seed matrix, rung G, placebo, contaminated retest and its repair |
| 11 | `11_leap_hand_track_b.md` | LEAP: native ADR discovery, the unattended queue, failures, final 3-seed numbers |
| 12 | `12_validity_and_claims_ledger.md` | Claim-by-claim: what was claimed, how it was tested, whether it survived |
| 13 | `13_open_questions_and_further_plan.md` | What is unproven, the research-level plan, publication assessment |
| 14 | `14_reproduction_and_file_map.md` | Commands, file map, run naming, compute notes |
| 15–19 | `15_plan_for_a_positive_result.md`, `16…19_*_preregistration.md` | The plan and the four pre-registrations (E1, E2, E4 step-0, E5) |
| 20 | `20_START_HERE_project_explained_from_scratch.md` | Beginner guide to the whole project |
| 21 | `21_E1_to_E5_results.md` | Results of the pre-registered experiments (E1 positive; E2/E4/E5 nulls) |
| 15 | `15_plan_for_a_positive_result.md` | Proposed experiments (E1–E5) to earn a genuine "success" for the paper: reasoning, pre-declared criteria, costs, schedule, outcomes |

## 3. Headline findings and how far to trust them

Confidence legend: **High** = replicated across ≥5 seeds and/or checked against an independent oracle; **Medium** = ≥3 seeds, directional; **Low** = single seed or post-hoc.

| # | Finding | Key numbers | Confidence |
|---|---|---|---|
| 1 | The environment permits millimetre precision; the ~30 mm floor of learned policies is a reward/optimiser property | IK+PD: 100% at 5 cm … 1 mm; learned: median ≈ 25–35 mm | High (oracle) |
| 2 | Eureka-type reward is coarse-success dominated: ~99% at 5 cm but ~27% at 2 cm and ~8% at 1 cm | B (n=5): 98.98 ± 0.76 / 26.87 ± 6.26 / 7.92 ± 2.07 % | High |
| 3 | Eureka does **not** beat the human reward at matched 800-iteration budget | A vs B nominal mean error 37.75 vs 37.69 mm (p=0.98); 2 cm 28.99 vs 26.87 % (p=0.56) | High (n=5) |
| 4 | Static RAPP-bounded DR lowers 2 cm success relative to no-DR for the Eureka reward when evaluated under physics randomization | 2 cm DRtrain 40.8 % (B) vs 21.4 % (D), p=0.016; DRshift 43.2 vs 19.8 %, p=0.016 | Medium (n=5, near the minimum attainable p) |
| 5 | Adaptive DR without RAPP's envelope (E) sacrifices coarse success at nominal physics | 5 cm: 80.4 ± 12.7 % vs D 97.5 ± 0.8 %, p=0.008 (complete separation, all 5 seeds) | High |
| 6 | No adaptive-DR variant (F4, E) beats static DR on mean/tail error | F4−D mean err +6.96 mm (p=0.37); E−D −10.2 mm DRtrain (p=0.15) | Medium (n=5; underpowered for small effects) |
| 7 | PINN observer: accuracy good inside its training range, unreliable outside; uncertainty weakly tracks error; mass/effectiveness degeneracy partly broken by sequential single-joint steady-state excitation | interp acc RMSE 1.95 rad/s²; extrap 7.73; rank corr +0.27 / −0.16 | Medium |
| 8 | PINN-informed adaptive DR (G) is not demonstrably better than a shuffled-information placebo | 10 clean seeds per arm, DRshift worst-decile (pre-declared primary) 221 vs 392 mm, p = 0.109; nominal/DRtrain p ≥ 0.48 | Medium (direction favours G in 8/10 seeds but not significant) |
| 9 | LEAP Hand already ships native adaptive DR; the stock "baseline" was not a no-DR baseline | num_adr_increases 0 → 8 of 25 over 500 iters | High (read from TensorBoard) |
| 10 | LEAP matrix (no ADR / native ADR / warm-started ADR), 3 seeds | nominal 8.42 / 8.68 / 8.00 rotations per 20 s (p ≥ 0.70); full DR 0.56 / 0.75 / 1.53 (p ≥ 0.20): no condition effect, seed variance dominates, all policies collapse at full DR | High that *no large effect exists at this budget*; budget too short to test ADR |
| 11 | **E1 (pre-registered): long-run exploration collapse** — with the default entropy coefficient the human-reward policy degrades with longer training; entropy 0.02 prevents it | 5000 iterations: mean error 61.8 ± 22.4 vs 30.4 ± 3.1 mm, worst-decile 346 vs 66 mm, p = 0.0079 (both); 2 cm declines in 5/5 seeds; DRshift also better | **High** (5 seeds, pre-registered, mechanism measured) |
| 12 | Pre-registered nulls: Eureka learns no faster than the human reward (E5, p = 0.84); LEAP quality does not collapse before quantity (E4 step-0, 0/3 conditions); LEAP robustness curve does not separate conditions (E2) | doc 21 | Medium |
| 13 | Exploratory: warm-started LEAP ADR (L2) drops the cube 5–10× less at ADR ≥ 10 (all seeds) | doc 21 | **Low** (3 seeds, post hoc) |

## 4. Claims we retracted (details in `12_validity_and_claims_ledger.md`)

1. "Eureka beats the human reward at every threshold" — single seed, mismatched budgets. **Retracted.**
2. "F4 (precision-gated adaptive DR) is the cleanest adaptive-DR win" — seed 42 only; n=3 and n=5 show a tie. **Retracted.**
3. "Rung E is the largest effect, ~2× precision under DR" — seed-42 value was the second-best of three; n=5 shows only the nominal-physics cost is robust. **Retracted.**
4. "G's DRshift advantage over placebo (p=0.024, 5 seeds)" — did not replicate; the 10-seed version was later found contaminated (below). **Not supported.**
5. "LEAP_500 is a no-DR baseline" — it ran with native ADR active. **Relabelled.**
6. Several of my own diagnoses were wrong at first (dual-torque PINN fix; LEAP "host-RAM OOM" explanation) and were corrected in the log.

## 5. Data-integrity events (all disclosed)

| Date | Event | Effect | Resolution |
|---|---|---|---|
| 15 Sep | Eval recorded post-reset distance | 0.39 % vs true 93 % | Fixed (`horizon−2`) |
| 15 Sep | Damping override 0 vs 0.05; ±π joint trap; mass write skipped at 1.0; PINN label noise | See doc 04 | Fixed, verified |
| 12–14 Oct | Concurrent G/placebo runs shared a timestamp-named run directory (seeds 3–6); later one rerun was truncated at 600/800 iterations by a transient evaluation failure | Cross-arm checkpoint mixing; seed 4 arms evaluated the identical file | Quarantined; cleanly rerun with sequential arms (audit passed); 10 clean seeds per arm |
| 12–13 Oct | LEAP queue "failures" | First: real GPU OOM from other users; later: upstream early stop (`score_to_win`) misread as failures | Queue rewritten; early stop disabled for matched budget |

## 6. Glossary

- **Rung A–G:** A human reward, no DR · B Eureka reward, no DR · C human + static DR · D Eureka + RAPP-bounded static DR · E Eureka + adaptive DR *without* RAPP bounds · F4 Eureka + RAPP + precision-gated adaptive DR · G = F4 + frozen-PINN-informed step scaling · Gp = G with shuffled multipliers (placebo).
- **Evals:** *nominal* (no randomization), *DRtrain* (friction [0,0.25], damping×[0.5,4], effectiveness [0.5,1.0], mass×[1.0,1.5]), *DRshift* (friction [0.25,0.5], damping×[4,8], effectiveness [0.3,0.5], mass×[1.5,2.5]) — DRshift is strictly harder and unseen.
- **Precision curve:** fraction of episodes ending within 5 cm / 2 cm / 1 cm / 5 mm / 2 mm / 1 mm of the goal. **Worst-decile error:** mean terminal error of the worst 10 % of episodes (tail metric).
- **RAPP:** Reward-Aware Physics Prior (DrEureka): a one-parameter-at-a-time feasibility sweep giving the physics region where a nominal policy still succeeds.
- **PINN observer (ξ):** network estimating ξ = [m1, m2, f1, f2, b1, b2, α1, α2] (masses, Coulomb friction, viscous damping, actuator effectiveness) from a window of transitions.
- **ADR:** automatic domain randomization — widen randomization ranges when the policy succeeds.

## 7. Repositories and sources cited throughout

Isaac Lab <https://github.com/isaac-sim/IsaacLab> · Isaac Sim <https://developer.nvidia.com/isaac/sim> · rl_games <https://github.com/Denys88/rl_games> · IsaacLabEureka <https://github.com/NVlabs/IsaacLabEureka> · Eureka <https://github.com/eureka-research/Eureka> · DrEureka (RAPP) <https://github.com/eureka-research/DrEureka> · LEAP_Hand_Isaac_Lab <https://github.com/leap-hand/LEAP_Hand_Isaac_Lab> · LEAP_Hand_Sim <https://github.com/leap-hand/LEAP_Hand_Sim> · LEAP_Hand_API <https://github.com/leap-hand/LEAP_Hand_API> · IsaacLabExtensionTemplate <https://github.com/isaac-sim/IsaacLabExtensionTemplate> · BayesSim arXiv:1906.01728 · Understanding Domain Randomization arXiv:2110.03239 · OpenAI ADR (Akkaya et al. 2019) arXiv:1910.07113 · LEAP Hand (Shaw et al., RSS 2023). Two citations in the roadmap (a 2026 IEEE Access physics-informed randomization paper; Phys2Real, ICRA 2026) could not be verified from this machine and remain unchecked.
