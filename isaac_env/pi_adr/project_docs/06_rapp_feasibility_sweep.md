# 06 — RAPP: the reward-aware physics prior

**What RAPP is.** From DrEureka (<https://github.com/eureka-research/DrEureka>): given a policy trained at nominal physics, sweep one physical parameter at a time and record where the policy still works. The union of the feasible ranges is the *physics prior*, C_RAPP, which bounds how much randomization it is sensible to train under.

**Why we did it first.** The roadmap's matrix (rungs D–G) all depend on C_RAPP as the outer bound for DR, and rung E exists specifically to test whether that bound matters. It also tells us how much headroom adaptive DR could possibly exploit.

## 1. Protocol (`scripts/run_rapp.py`)

- Policy: nominal Eureka policy (`rapp_eureka500`, trained 500 iterations, seed 42).
- One parameter varied at a time (friction, damping scale, actuator effectiveness, link-mass scale), all others nominal — so a change in success is attributable to a single cause.
- Each candidate value evaluated on **1024 environments**; *feasible* iff S(π, ξ) = P(terminal error < 5 cm) ≥ **0.90**.
- Output stored in `results/rapp_eureka500_envelope.json` and used verbatim by the adaptive controller.

## 2. Result: the envelope

**C_RAPP = friction [0, 4.0] × damping_scale [0.25, 256] × effectiveness [0.3, 1.0] × mass_scale [1.0, 4.0]**

Selected rows of the physics–performance map:

| Parameter | value | S(π,ξ) | mean error (mm) | P(d < 1 cm) | feasible? |
|---|---|---|---|---|---|
| friction | 0.0 (nominal) | 0.974 | 45.0 | 3.61 % | yes |
| friction | 4.0 | 0.946 | 59.3 | 7.13 % | yes |
| friction | 8.0 | 0.843 | 94.6 | — | **no** |
| damping ×1 | nominal | 0.969 | 49.3 | 4.59 % | yes |
| damping ×128 | | 0.921 | 82.8 | **16.21 %** | yes |
| effectiveness | 0.3 | 0.915 | 53.4 | **16.02 %** | yes (edge) |
| effectiveness | 1.0 (nominal) | 0.972 | 47.1 | 8.11 % | yes |
| mass ×0.75 | | 0.786 | 57.9 | — | **no** |
| mass ×1.0 (nominal) | | 0.975 | 46.8 | — | yes |
| mass ×4.0 | | 0.919 | 42.0 | — | yes |

## 3. What the numbers mean

1. **The feasible envelope is very wide.** The policy tolerates friction up to 4 (vs nominal 0), damping up to 256× nominal, actuators at 30 % strength, and links 4× heavier, with nearly flat coarse success across most of it (0.92–0.975). The roadmap anticipated this exact case: *if RAPP already covers a region where performance is flat, adaptive DR has little left to exploit.* This turned out to be prophetic — adaptive DR did not beat static DR (doc 09/10).
2. **Fine precision *improves* in the "hard" corners** (P(d<1 cm) rises from ≈ 4–8 % to ≈ 16 % at damping ×128 or effectiveness 0.3). This is the same phenomenon seen in the DR robustness table (doc 05): physics that slows the arm forces slower, more careful arrival that the reward never required. It is evidence that the 5 cm *feasibility* criterion and *precision* are different things — a recurring theme.
3. **The binding constraint is asymmetric in mass.** The policy fails for *lighter* links (≤ 0.75×) but tolerates heavier ones. Lower inertia gives more acceleration per unit torque, so a policy tuned to nominal inertia overshoots. This direction information was used in the controller design: mass widens only upward; effectiveness only downward (below 0.3 the policy fails); friction and damping upward.
4. **The 5 cm criterion is coarse.** Using P(d<5 cm) ≥ 0.9 as "feasible" is exactly the coarse threshold the project showed can stay satisfied while precision degrades; this later explained why a controller gated on 5 cm widened through precision-degraded regions (doc 09, F1/F2).

## 4. Why we believe the sweep is right (and where it is thin)

- **Verification of the sweep machinery.** Defect D5 (mass write skipped at scale 1.0, doc 04) corrupted the first mass sweep (nominal read 0.816, true 0.975). The fixed sweep reproduces 0.975 at nominal — a consistency check that the sweep's baseline equals independent evaluation.
- **Read-back of DR values** (validate_env check I) confirms each candidate really changed the simulator.
- **Thin spot:** the envelope was measured for *one* policy (one seed). It was not repeated across seeds, so its exact edges (e.g. friction between 4 and 8, effectiveness near 0.3) carry seed uncertainty. Its qualitative shape (wide, asymmetric in mass) is corroborated indirectly: adaptive rung E, with no ceiling, pushed mass to ≈ 8× and effectiveness below 0.3 and lost nominal coarse success — consistent with the envelope marking a real boundary of usefulness for the nominal policy.

## 5. How RAPP was used downstream

| Use | Where | Outcome |
|---|---|---|
| Bound for adaptive widening (`c_rapp` in `AdaptiveDRController`) | F, F4, G | Controllers never exceeded C_RAPP |
| Direction of widening per dimension | controller design | friction/damping/mass ↑, effectiveness ↓ |
| Starting box for static DR "in-distribution" (friction [0, 0.25], damping ×[0.5, 4], effectiveness [0.5, 1], mass ×[1, 1.5]) | D, C, DRtrain eval | A conservative interior of C_RAPP |
| Ablation: replace C_RAPP with arbitrary wide bounds (friction [0,50], damping ×[0.1,1000], effectiveness [0.01,2], mass ×[0.1,20]) | rung E | Without the ceiling, mass widened to ≈ 8× and nominal 5 cm success fell to 80.4 ± 12.7 % (vs 97.5 % for D) |

## 6. Honest assessment

RAPP did what it is supposed to do — it **bounds the nominal-performance cost of aggressive randomization** (rung E) — but the evidence that RAPP-constrained *adaptive* DR improves on RAPP-bounded *static* DR is absent at n = 5. Whether RAPP initialisation helps on a task where the nominal policy is *not* flat across a wide envelope (e.g. LEAP) is the open question that the LEAP track was designed to test; a true LEAP RAPP sweep has **not** been run (doc 11, doc 13).
