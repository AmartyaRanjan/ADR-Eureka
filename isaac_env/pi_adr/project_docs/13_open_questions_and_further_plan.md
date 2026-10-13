# 13 — Open questions, further plan, and publication assessment

## 1. What is still open (ranked by value per unit of effort)

| # | Question | Why it matters | What would settle it | Cost |
|---|---|---|---|---|
| 1 | **Finish the clean G/placebo rerun (seeds 3–6)** | Repairs the contaminated retest; fixes power | `chain15` queued behind LEAP (sequential arms) | ≈ 4 h GPU |
| 2 | **Finish LEAP at 3 seeds** (L0/L1/L2) | Only real-platform evidence; needed for any robotics venue | `leap_queue.sh` (running) | ≈ 6–7 h more |
| 3 | **Does the entropy collapse explain vanilla's long-run failure?** | Decides how much of Stage 1's discrepancy is an artifact | Vanilla at 5000 iterations × 5 seeds, entropy 0.005 vs 0.02 | ≈ 7 h (5 × 79 min × 2 / lanes) |
| 4 | **Why does static DR lower 2 cm precision vs no-DR (B vs D)?** | Mechanism candidate: no-DR policies are forced slower by harsh evaluation physics | Record mean |q̇|/effort per seed; evaluate B/D under an intermediate DR sweep; slow-down control (cap speed) | ≈ 2 h |
| 5 | **Is the observer uncertainty usable out of range?** | G's null is a weak test (damping-dominated, anti-correlated extrapolation) | Better calibration (full covariance, deep-ensemble recalibration, conformal), an observer retrained across seeds | days |
| 6 | **A genuine LEAP RAPP sweep and RAPP-initialised ADR** | The roadmap's contribution on the real platform | One-parameter sweeps on a trained LEAP policy; initialise ADR from the envelope | 1–2 days |
| 7 | **Longer budgets** (LEAP ≥ 1500–5000 iterations; 2-DOF > 800) | 500/800 iterations may be too short for ADR to matter | Re-run the matrix subset | days |
| 8 | **Does a precision-aware ADR criterion matter on LEAP?** | LEAP's rule is coarse-threshold like F1/F2 | Add a quality term to `adr_criteria` | 1 day |
| 9 | **Hardware / sim-to-sim transfer** | The only path to a main-track robotics claim | LEAP_Hand_API, or a second simulator | weeks |

## 2. Experimental improvements to carry into any new run

1. Unique per-run output directories (include seed and arm in `--experiment_name`) — never timestamp-only.
2. Pre-declare the primary metric and sample size; use ≥ 10 seeds for any claim about tail metrics (std on worst-decile is 40–150 % of the mean).
3. Always include a placebo or ablation arm when claiming that *information* helps.
4. Log mean joint speed/effort per seed for every evaluation, so "slower = more precise" can be tested instead of inferred.
5. Treat identical results across arms as a bug.

## 3. Publication assessment (judgment, not computed probabilities)

**What the evidence can support.** A methodology-and-negative-results paper: a validated benchmark and evaluation protocol; multi-seed re-evaluation showing that the headline reward comparison does not survive matched budgets; a RAPP sensitivity map; an identifiability analysis of a physics observer; evidence that a PINN-informed ADR step scaling is not distinguishable from a placebo; and LEAP's native ADR as an unexpectedly hidden confounder in "baseline" runs.

**Venues considered**

| Venue | Fit | Rough odds with current evidence | Notes |
|---|---|---|---|
| ICRA / IROS / CoRL main track | Poor | < 5 % | Needs a working new method or hardware results |
| Robotics/sim-to-real workshops (non-archival) | Good | ≈ 40–60 % | Best near-term target; feedback shapes a longer paper |
| RA-L (rolling; presentable at ICRA/IROS) | Moderate | ≈ 10–20 % now; 20–30 % with LEAP at ≥ 3 seeds and a clear robotics result | Check current policy |
| *Simulation Modelling Practice and Theory* (Elsevier, journal; scope per aggregators: systems simulation and modelling, validation, sensitivity analysis, experimental design) | **Best fit** | ≈ 25–40 % to acceptance after revision | The validation protocol, RAPP sensitivity map and identifiability analysis are core simulation methodology; expect major revisions |
| *Robotics and Autonomous Systems* (Elsevier, journal) | Weaker fit now | ≈ 10–20 % now; 20–30 % if LEAP gives a clear positive | Wants robotics contribution (working method or strong experiments) |

(Odds are rough judgments; the official aims-and-scope pages could not be fetched directly and information came from aggregator sites. A paper cannot be submitted to two journals simultaneously; two papers would need genuinely different contributions.)

**What would raise the odds**
1. A single sharp claim — e.g. *"reported gains of LLM-designed rewards and adaptive DR in precision reaching are largely explained by exploration collapse, unequal budgets and seed variance; here is a validation protocol that exposes this"* — with the entropy long-run experiment (#3 above) as its centrepiece.
2. LEAP at ≥ 3 seeds and, ideally, ≥ 1 extra task.
3. Release of code, logs, quarantine record and the data-integrity audit; this is unusually transparent and reviewers value it.
4. Framing for a simulation-methodology journal: lead with validation, sensitivity (RAPP) and identifiability, with ablation results as case studies.

## 4. Suggested paper outline (for the simulation-methodology framing)

1. Introduction: precision gaps in learned reaching; DR and ADR; the need for validated evaluation.
2. Simulation model and validation (docs 02, 04): analytic FK, closed-form dynamics, IK+PD oracle, defect catalogue.
3. Evaluation protocol and statistics (doc 03).
4. Sensitivity analysis: RAPP envelope (doc 06).
5. Identifiability of physical parameters: PINN observer (doc 08).
6. Case study: reward and DR ablations, A–G (docs 05, 07, 09, 10), with placebo and replication.
7. Platform case study: LEAP native ADR (doc 11).
8. Threats to validity and the integrity audit (doc 12, defect D7).
9. Conclusion and open problems.

## 5. Housekeeping before any submission

- Update `CODE_AND_MATH_COMPILATION.md` and the review slides (`Review2_PhysicsInformed_AdaptiveDR.pptx`) with the contamination correction and the finished LEAP numbers (the slide script is `../create_review2_pptx.py`; the temporary pptx virtualenv under `/tmp` was cleaned and must be recreated).
- Re-verify the two unchecked citations in the roadmap.
- Rotate the Gemini API key that was pasted into chat during setup.
- Archive: results JSONs, `index.jsonl`, logs, checkpoints needed for the headline numbers, and the quarantine directory.
