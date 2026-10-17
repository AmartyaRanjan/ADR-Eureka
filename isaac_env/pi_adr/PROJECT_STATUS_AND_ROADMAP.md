# Project Status and Research Roadmap

> **UPDATE 11 Oct 2026 (evening) — A–G matrix complete at n=5; supersedes §4, §4b and any "F/E beats D" reading.**
> Full tables + permutation tests: `results/MATRIX_A_to_G.md`; authoritative narrative: `EXPERIMENT_LOG.md` (11 Oct afternoon entry).
> - A, B, C indistinguishable on mean error / tail in all evals (p≥0.18). Static DR (D) lowers 2 cm success vs B under DRtrain/DRshift (p=0.016).
> - E (no RAPP) loses nominal 5 cm success vs D (80.4 vs 97.5%, p=0.008); its tail/mean advantage is NOT significant. F4 does not beat D.
> - G (PINN-scaled widening) has the lowest mean/worst-decile error and spread in every eval but G−F4 is not significant and G has the lowest 2 cm/1 cm success. The shuffled-multiplier placebo (n=5) matches G on nominal/DRtrain (p>=0.74); the DRshift signal (p=0.024 at n=5) was not significant in the final clean 10-seed retest (worst decile p=0.109; seeds 3–6 of the first retest were contaminated and were cleanly rerun — see EXPERIMENT_LOG 14 Oct). Rung G: no demonstrated PINN benefit. Observer uncertainty was dominated by damping scale and is unreliable outside its training range.
> - Still untested: LEAP hand (deferred), longer budgets, placebo at n=5 (done: see log 12 Oct), a reliable-extrapolation observer.


**As of:** 10 October 2026 (Session 2 — see `EXPERIMENT_LOG.md` for the full chronological record;
this document is the standing summary + forward plan, updated at the end of each session)

**Read this first if you are picking the project up cold.** It answers: what exists, what is
measured, what is trusted, what is open, and exactly what to do next, in priority order.

---

## 1. The arc so far, in one paragraph

A Stage-1 technical report described a 2-DOF Isaac Lab reaching benchmark and an Eureka-vs-human
reward comparison, with no corresponding code on this machine. Session 1 (15–16 Sep 2026) rebuilt
that benchmark from its equations, found and fixed five defects that would otherwise have
invalidated every result (most importantly: hard ±π joint limits that cap any controller near 69%
success), reproduced the Eureka-reward precision collapse closely, built RAPP, and built a
physics-informed observer (PINN) — whose own validation produced a negative result (its ensemble
uncertainty is not reliably informative). Session 2 (10 Oct 2026) used that negative result to
scope the next step honestly: built and ran adaptive domain randomization *without* the PINN
(rung F), compared it end-to-end against a static-DR baseline (rung D), diagnosed the PINN
uncertainty negative result down to a confirmed root cause (a mass/actuator-effectiveness
covariance degeneracy the diagonal-only estimate was hiding), and — rather than reporting a clean
win — found a mixed, honestly-reported result (F's nominal-physics performance dropped sharply,
likely from an uncontrolled training-budget mismatch, not from adaptive DR being unhelpful in
principle) that directly motivates a specific, cheap next experiment rather than a vague "try
harder." Every number produced is still single-seed; §5 lays out, in priority order, everything
that would need to happen before any of today's comparisons, or rung G, or a sim-to-real claim, is
defensible.

---

> **11 Oct correction summary:** after multi-seeding (n=3), none of the three apparent findings from
> 10 Oct holds as a clear effect — F4 > D (no), E > D on 2cm precision (directional only), Eureka >
> vanilla at matched budget (no; vanilla is as good and far more stable). Treat §4/§4b/§5.1 text
> above the correction markers as history; `EXPERIMENT_LOG.md` 11 Oct entry is authoritative.

## 2. What is built and trusted right now

| Component | Status | Evidence |
|---|---|---|
| 2-DOF Isaac Lab environment | Built, validated | `validate_env.py` 10/10, `validate_dynamics.py` passed |
| Vanilla + Eureka rewards | Implemented, checked byte-for-byte against report equations | `validate_env.py` check H |
| Per-env domain randomization | Implemented, verified non-silent | `validate_env.py` checks I/J |
| RAPP (DrEureka protocol) | Implemented, envelope measured | `results/rapp_eureka500_envelope.json` |
| Physics-performance map | Full metric vector (`M(xi)`), not just 1cm | `physics_performance_map_full` in today's eval JSONs |
| PINN physics observer | Implemented, trained, validated | `results/pinn_observer.json` |
| PINN uncertainty root cause | **Diagnosed this session** — mass/effectiveness covariance degeneracy confirmed | `results/pinn_covariance_diagnosis.json` |
| IK+PD scripted reference | Implemented | `results/PD_reference.json` — 100% success at every threshold |
| Adaptive DR controller (rung F, no PINN) | Implemented and run end-to-end this session | `source/pi_adr/dr/adaptive.py`, `scripts/run_adaptive_dr.py` |
| LEAP Hand (track B) — native-ADR run (NOT a no-DR baseline, see §5.6) | Trained, evaluated | `results/LEAP_500.json` — 7.52 rotations/20s, native ADR at 8/25 increments |
| Adaptive DR vs static DR (rung F vs D) | **Complete — single seed, mixed/preliminary result, see §4** | `results/adaptive_dr/`, `results/D_*`, `results/F_*` |
| Rung G (PINN-informed adaptive DR) | Built and run at n=5 (12 Oct); benefit over shuffled placebo not demonstrated — see top update block and EXPERIMENT_LOG | — |
| Full A–G ablation matrix, multi-seed | Not started | — |
| Sim-to-real / hardware | Not started, correctly gated behind the above | — |

## 3. What is NOT trusted / open questions carried forward

1. **Vanilla (human) reward does not reproduce its Stage-1 precision numbers on our rebuild**
   (13.3% vs reported 98.4% @2cm). Two measured, not-yet-resolved candidate causes: action-σ
   collapse (1.0→0.024) and a weak precision incentive in the reward itself. Not yet isolated which
   dominates. See §5.3.
2. **PINN ensemble uncertainty is only weakly informative** (rank corr. +0.11). **Root cause now
   confirmed** (10 Oct, same-day diagnostic, no new training needed — see `EXPERIMENT_LOG.md`):
   the diagonal-only covariance estimate hides a strong joint mass/actuator-effectiveness
   degeneracy. Cross-member correlations `corr(m1,a1)=+0.535`, `corr(m2,a2)=+0.466`,
   `corr(a1,a2)=-0.568`; the dominant uncertainty eigenvector is 92% aligned with `m1` and mixes in
   `a1`/`a2`. The fix is now a specific, scoped next step (§5.4), not an open-ended investigation.
3. **RAPP's envelope is wide and nearly flat on 3 of 4 parameters** (friction, damping,
   effectiveness once inside their feasible range). **Partially confirmed by today's D-vs-F run**:
   the adaptive controller's own trajectory (§4) widened mass and effectiveness fastest and
   furthest and found friction's real feasibility edge by stalling near it — consistent with RAPP's
   prediction of where the gradient actually is. What is NOT yet confirmed is whether exploiting
   that gradient via adaptive DR produces a net performance benefit — today's result was mixed/
   negative on that specific question, for reasons tied to curriculum rate rather than to this
   prediction being wrong (see §5.1).
4. **Only one seed per condition exists anywhere in this project.** Every number in this document
   and in `EXPERIMENT_LOG.md` is a single-seed point estimate. None of the comparisons (Eureka vs
   vanilla, D vs F, nominal vs DR) have been tested for statistical significance. This is the
   single biggest scientific-rigor gap right now — see §5.2.

---

## 4. Session 2 experiment: rung F (adaptive DR) vs rung D (static DR)

**Design** (full rationale in `EXPERIMENT_LOG.md`, "2026-10-10" entry, and in
`source/pi_adr/dr/adaptive.py`'s docstring):

- Both: Eureka reward, seed 42, 2048 train envs, 800 total PPO iterations, evaluated with the
  Session-1-frozen metric suite under three physics distributions (nominal, in-distribution,
  shifted/unseen) — exactly the protocol used for Session 1's `B_eureka_1500_DR*` results, so these
  rows sit in the same table.
- **D**: DR range fixed for the whole run at the in-distribution range
  (friction[0,0.25], damping[0.5,4], effectiveness[0.5,1], mass[1,1.5]).
- **F**: same starting range, but a controller (`AdaptiveDRController`) widens each dimension
  toward its RAPP-measured bound whenever the policy clears 90% success on the current range, and
  narrows back after 2 stalled evaluations — the generic ADR-style (Akkaya et al. 2019)
  boundary-widening rule, carrying no PINN signal.

**Status at time of writing:** training launched and in progress (D at iteration ~108/800 when
this paragraph was last updated; F runs after D completes, sequentially, to keep the shared
machine's GPU footprint low). The orchestrator (`scripts/run_adaptive_dr.py`) was smoke-tested
end-to-end (2 outer steps, trivial scale) before the real run, and a real implementation bug was
caught in that smoke test and fixed: rl_games restores the cumulative epoch counter on
`--checkpoint` resume, so `--max_iterations` must be passed cumulatively across chunks, not as a
fixed per-chunk value (documented in `EXPERIMENT_LOG.md`).

### Results — complete (full detail and the controller's step-by-step trajectory in `EXPERIMENT_LOG.md`)

| Run | physics | <5cm | <2cm | <1cm | mean err | settling |
|---|---|---|---|---|---|---|
| D (static DR) | nominal | 97.78% | 32.50% | 8.76% | 30.5 mm | 1.17 s |
| D (static DR) | in-distribution | 99.73% | 21.80% | 6.23% | 28.7 mm | 0.92 s |
| D (static DR) | shifted (unseen) | 96.39% | 21.07% | 4.96% | 57.0 mm | 1.80 s |
| F (adaptive DR) | nominal | **72.88%** | 20.14% | 5.42% | 38.3 mm | **7.18 s** |
| F (adaptive DR) | in-distribution | 94.19% | **25.37%** | **8.15%** | 30.7 mm | 2.23 s |
| F (adaptive DR) | shifted (unseen) | 96.44% | 21.66% | 5.22% | 54.1 mm | 1.63 s |

### Interpretation checklist (completed)

- [x] **Did F reach a wider final range than D's fixed range, inside C_RAPP?** Yes, substantially:
      friction to 60% of its C_RAPP span, damping to 82%, effectiveness to 97%, mass to 92% (D
      stayed at its fixed starting range the whole run: 6%/1.4%/0%/17% of the same spans). The
      controller clearly did something, and something matching RAPP's own asymmetry — mass and
      effectiveness (RAPP's sharpest-gradient dimensions) widened fastest.
- [x] **Does F's shifted-DR performance beat D's, at matched or better nominal performance?**
      No — shifted-DR performance is a statistical wash (differences within plausible single-seed
      noise on all six metrics), and nominal performance is **substantially worse for F**
      (72.88% vs 97.78% @5cm, 7.18s vs 1.17s settling — large enough to trust without more seeds).
- [x] **Which dimension(s) widened fastest / stalled?** Mass and effectiveness widened every single
      outer step without stalling; friction stalled twice early (outer 0–1) then widened for three
      steps before stalling again in the last two steps as success dipped to 0.79/0.75 — i.e. the
      controller correctly found and respected its own feasibility edge on friction, consistent
      with RAPP.
- [ ] **Is the effect size larger than single-seed noise?** The nominal-performance gap is (too
      large and too mechanistically explicable to be noise). The shifted-DR and in-distribution
      comparisons are NOT yet distinguishable from noise — §5.1 (multi-seed) is required before
      either is treated as a real effect.

### What this result actually shows, and the concrete gap it surfaces

Rung F is built, runs correctly end-to-end (hydra-override DR injection, checkpoint-cumulative
resume, per-dimension evaluation, curriculum update — all verified working), and its widening
decisions are interpretable and match the RAPP envelope's own structure. **It does not, in this
single run, demonstrate a net advantage over static DR** — and the clearest single number (nominal
performance) points at a specific, fixable experimental-design flaw rather than a dead end: D
trained on its fixed, comparatively narrow range for the *entire* 800-iteration budget, while F's
training distribution kept growing, so by the end F had spent a shrinking fraction of its fixed
total budget anywhere near the easy/nominal point it is still evaluated against. **D and F were not
actually training under a comparable total "difficulty-weighted" budget despite an identical
iteration count.**

**New, concrete follow-up this directly motivates (add to §5, next experiment to run before any
other D/F conclusion is drawn):**
- Re-run rung F with either (a) a slower widening rate (`step_frac` well below 0.25, e.g. 0.08–0.10)
  so the final distribution stays narrower given the same 800-iteration budget, or (b) a longer
  total budget (e.g. 2000+ iterations, matching Session 1's Eureka@1500-scale runs) so the
  policy has time to consolidate at each successive range before the next widening, or (c) both.
  Re-measure nominal performance specifically as the acceptance test — if it recovers toward D's
  97.78% while shifted-DR performance is retained or improved, that is the first real evidence
  adaptive DR is paying for itself; if nominal stays depressed even with slower widening, that
  points at a deeper issue (e.g. the success-rate signal needs to include nominal/easy-mode
  explicitly, not just the current training range).

---

## 4b. Rung E (adaptive DR, no RAPP) — n=3 result (updated 11 Oct; supersedes the single-seed reading)

Same controller as F4 with `--no_rapp`. Seeds 42/123/7 now complete. Full tables in
`EXPERIMENT_LOG.md` ("2026-10-11 — multi-seed results land").

| | nominal 5cm | in-dist 2cm | in-dist mean err | shifted 2cm | shifted mean err |
|---|---|---|---|---|---|
| D | 97.44 ± 0.47 | 24.10 ± 9.26 | 37.8 ± 9.4 mm | 21.98 ± 8.27 | 58.2 ± 20.2 mm |
| E | **82.06 ± 14.86** | 38.21 ± 15.98 | **27.1 ± 2.4 mm** | 32.12 ± 17.17 | **32.8 ± 3.9 mm** |

- **Robust:** E sacrifices nominal performance (below D's minimum seed in 3/3 seeds).
- **Not established:** 2cm/1cm precision gain under DR (higher mean, but std 16–17 points; t ≈ 1.3).
- **Suggestive:** lower and much tighter *mean (tail) error* under DR in 3/3 seeds — E's benefit looks
  like fewer bad episodes, not better typical precision.
- The 10 Oct "largest effect this session" reading came from seed 42, which was E's second-best seed.

## 5. Detailed, itemized further plan

Ordered by priority. Each item states what, why, how, and its dependency on other items — written
so that whoever picks this up next (including a future instance of this same assistant) can start
immediately without re-deriving the reasoning.

### 5.1 — Priority 1, step 1 DONE / step 2 now the actual priority: curriculum signal, not just rate

**Done this session (F2):** re-ran rung F with `step_frac: 0.25 -> 0.10`, same 800-iteration budget.
Result (full detail and all nine numbers in `EXPERIMENT_LOG.md`, "F2" entry): the budget-mismatch
hypothesis was confirmed **directionally but not cleanly** — nominal 5cm success recovered from
72.9% to 90.9% (D: 97.8%), but F2 became the *worst* of the three runs (D, F1, F2) at 2cm precision
in every single physics condition tested (nominal, in-distribution, and shifted alike).

**New finding, more fundamental than a rate problem:** the controller's widen/hold gate is
`S_d = P(d_T<5cm)` — the same coarse threshold whose insensitivity to fine precision is this
project's original, central finding (Session 1's Eureka-vs-vanilla result). A slower schedule does
not avoid that trap; it makes the controller linger *longer* inside physics bands that clear 5cm but
have already degraded at 2cm, rather than moving through them quickly. **Widening rate was a real
but secondary problem; gating on a coarse threshold is the primary one.**

**What:** Re-run rung F gating the widen/hold decision on `P(d_T<2cm)` (or a weighted combination of
coarse and fine success) instead of `P(d_T<5cm)` alone.
**Why:** Directly motivated by the F2 result above — not a guess. If 2cm precision recovers across
all three physics conditions without reintroducing the nominal-performance collapse, that is the
first genuinely clean adaptive-DR win in this project.

**Done this session (F4), and it worked — the best result so far.** `--gate_threshold` and
`--success_target` wired through `run_adaptive_dr.py` -> `eval_dr_dims.py` (small change; the
controller itself needed no modification, only which threshold feeds it). First attempt (F3,
`success_target` left at 0.90) was mis-specified — no policy in this project reaches 90% at 2cm, so
the range froze; caught after 3 outer steps and relaunched as F4 with `success_target=0.30`
(matched to the actually-observed 2cm success range). **Result: F4 beats both F1 and F2 on 2cm
precision in every physics condition tested**, and beats static D outright in 2 of 3 conditions
(in-distribution 2cm: 29.4% vs D's 21.8%; shifted 5cm/2cm/mean-error: all better than D). The one
place D still wins (nominal mean error) is driven by a heavy tail of occasional bad episodes, not a
worse typical case (median error is essentially tied). Full nine-row comparison table and mechanism
discussion in `EXPERIMENT_LOG.md`'s "F4" entry — do not skip it, the *why* (gating on a
precision-aware signal keeps the controller out of the wide-but-imprecise regimes F1/F2 spent much
of their budget in) matters as much as the numbers.

**Still single-seed — this is the result that makes §5.2 (multi-seed) most urgent,** not a reason to
skip it. This is the first result in the project with a clean, mechanistically-explained story in
adaptive DR's favor; it deserves to be confirmed, not just liked.

**UPDATE — DONE, and it did NOT hold up.** n=3 multi-seed re-run (seeds 42, 123, 7) of both D and
F4: **D and F4 are statistically indistinguishable on every precision threshold in every physics
condition.** At 1cm, D's mean is actually slightly *higher* than F4's in all three conditions
(though within overlapping spread). The seed-42 gap that drove the "F4 beats D" claim
(e.g. 29.42% vs 21.80% @2cm in-distribution) is smaller than the spread either rung shows across its
*own* three seeds (D alone: 16.21–34.30% @2cm in-distribution). Full table and raw per-seed values
in `EXPERIMENT_LOG.md`'s "multi-seed D vs F4" entry — **read it before citing F4 anywhere.**

**This is the single most important correction from this session.** Three adaptive-DR controller
variants were built and tested (F1, F2, F4); none has been shown, with proper multi-seed evaluation,
to outperform simple static DR on this benchmark. This does not rule out a smaller real effect that
n=3 is underpowered to detect, and does not settle rung E (no RAPP) or rung G (PINN-informed, once
fixed) — both still open — but it means **no claim that "adaptive DR helps" should leave this
project without an n≥5 re-check**, and `F4` should not be cited as a finished positive result.

### 5.2 — Priority 2: Multi-seed re-run of the core comparisons — D vs F4 DONE, see above

**What:** Re-run (a) Eureka vs vanilla nominal, (b) rung D vs rung F, each across >= 3 seeds.

**Status: (b) done this session (n=3, see the correction above — result was negative, not
confirmatory). (a) Eureka vs vanilla nominal still not multi-seeded** — Session 1's headline
reward-comparison finding (Eureka beats vanilla on this rebuild) carries the same single-seed
caveat as everything else and has not yet been checked. Given the magnitude of that gap (Eureka
99.05% vs vanilla 93.04% @5cm, 7.91% vs 2.22% @1cm — much larger than the D-vs-F4 gaps that turned
out to be noise), it is more likely to be a real effect, but "more likely" is not "confirmed," and
this should be the next multi-seed check run, ahead of a 4th adaptive-DR controller variant.
**Why:** Every result in this project to date is single-seed. The D-vs-F effect size (§4), the
Eureka-vs-vanilla gap (Session 1), and the DR-improves-precision finding (Session 1) could each be
seed noise rather than a real effect, and nobody can currently tell. This is a correctness gap, not
a nice-to-have.
**How:** `train_and_eval.sh` and `run_adaptive_dr.py` both already take `--seed`; wrap each in a
loop over 3 seeds, aggregate with mean ± std (not just mean) in `summarize_results.py`. Budget:
~3x today's compute for the D-vs-F pair alone (~2 hours of shared-GPU time at today's 2048-env
scale) plus ~1 hour for Eureka-vs-vanilla at 4096 envs.
**Blocks:** Any publication-quality claim about D vs F, Eureka vs vanilla, or DR-improves-precision.

### 5.3 — DONE this session: vanilla-reward sigma collapse, isolated to one cause

**Resolved.** Two independent variants (V1: raise `entropy_coef` 0.005→0.02, reward unchanged; V2:
weaken `action_penalty_coef` 0.005→0.001, entropy unchanged), both vanilla reward, 1500 iters, 2048
envs, seed 42, full detail in `EXPERIMENT_LOG.md`:

| | <5cm | <2cm | <1cm | entropy @100% |
|---|---|---|---|---|
| Session-1 baseline (5000 iters, entropy_coef=0.005) | 93.04% | 13.28% | 2.22% | -4.66 |
| V1 (entropy_coef=0.02) | 97.83% | **17.36%** | **3.91%** | -1.23 |
| V2 (action_penalty_coef=0.001) | 99.24% | 12.23% | 1.51% | -4.12 |

V1 beat the baseline on every metric despite 3.3x fewer iterations and half the envs — not a
compute effect, specifically an exploration effect. V2 left the collapse intact (entropy barely
moved) and did not improve precision — if anything slightly worse. **The entropy coefficient, not
the reward's action-penalty weighting, is the confirmed dominant cause.**

**What remains open (smaller, well-defined now, not urgent):** V1 still reaches only 17.36%/3.91%
@2cm/1cm, nowhere near the Stage-1 report's claimed 98.36%/79.74%. Whether `entropy_coef=0.02` at
the full 5000-iteration budget closes most of the remaining gap, or whether something else entirely
is needed, is an open but low-priority follow-up — it does not block anything else in this plan and
can be picked up opportunistically (§5.8-style, cheap to run whenever there is spare GPU time:
one `train_and_eval.sh`-style run at 5000 iterations with the entropy override).

### 5.4 — Priority 4: Calibrate PINN uncertainty, then build rung G

**What:** Before attempting rung G (PINN-informed adaptive DR), fix or replace the uncertainty
estimate so that `U_PINN` is actually informative, or make a reasoned decision to drop it from the
priority score (`alpha_u = 0`) and document why.
**Why:** §3.2 — the diagonal-only ensemble covariance is the leading suspect for why uncertainty
doesn't track error (mass and actuator effectiveness are degenerate given a single torque regime
and cannot be told apart by a diagonal estimate). Building G without addressing this would waste
the compute budget on a result nobody could trust.
**Status: diagnostic done; first proposed fix tried and REFUTED; corrected fix identified, not yet
tried.** Full detail in `EXPERIMENT_LOG.md` ("PINN covariance diagnosis" and "dual-torque-regime
fix" entries). Sequence so far:

  1. **Diagnostic** (done): `corr(m1,a1)=+0.535`, `corr(m2,a2)=+0.466`, dominant eigenvector 92%
     aligned with `m1` mixing in `a1`/`a2` — the diagonal-only uncertainty estimate hides a real
     mass/actuator-effectiveness degeneracy.
  2. **First fix attempt — varying torque amplitude within each environment's window** (done,
     refuted): hypothesized that different torque magnitudes would shift the relative size of
     Coriolis/friction terms versus `alpha*tau/M` enough to separate `m` from `alpha`. Implemented
     (`collect_pinn_data.py --torque_scale` now accepts multiple values, sampled per hold-cycle),
     re-collected all three splits, retrained, re-diagnosed. **Result: the degeneracy got worse, not
     better** (`corr(m1,a1)` +0.535→+0.652, `corr(m2,a2)` +0.466→+0.861, total ensemble variance
     3.3x larger, several new large correlations appeared). **The physics reasoning behind this fix
     was incomplete**: in `forward_dynamics`, the inertial term is `alpha*tau/M`, which scales
     *linearly* with `tau` for any fixed `(alpha,M)` — so the *ratio* `alpha/M` that actually
     determines observed acceleration is unaffected by which torque amplitude was used to probe it.
     Varying torque amplitude adds more data along the same degenerate direction, not an independent
     direction. This is a genuine, confirmed negative result for this specific fix, not an
     implementation bug (collection logs confirm the fix ran as intended).
  3. **Corrected fix, reasoned from the equations of motion** (not yet tried): the only terms that
     distinguish `m` from `alpha` are velocity-dependent (`C(q,q̇)`, which depends on `m2` but not
     `alpha`; `F(q̇)`, which depends on neither). These terms matter most relative to the inertial
     term either at high velocity or, most cleanly, **at steady state**: holding torque constant
     long enough for `q̈ -> 0` collapses `inverse_dynamics` to `alpha*tau = C(q,q̇_terminal) +
     F(q̇_terminal)` — mass drops out entirely, leaving an equation that constrains
     friction/damping/effectiveness without mass contamination. The current protocol's hold window
     (`--hold 6` steps ≈ 0.2s) never approaches terminal velocity at this arm's inertia/time
     constant. **Concrete next step:** add a long-hold excitation mode to `collect_pinn_data.py`
     (hold one torque value for enough physics steps — estimate the needed duration analytically
     from `dynamics.py`'s `M`/`F` rather than guessing — to approach steady state) interleaved with
     the existing short-hold transient excitation, so each environment's window contains both. Then
     re-run the same diagnostic as the acceptance test.
  4. If step 3 doesn't fully resolve it, fall back to reporting only the identifiable combination
     (e.g. an `alpha_i / sqrt(m_i)`-type ratio, derivable analytically from the mass-matrix
     structure) to the adaptive-DR controller instead of the raw 8-D vector — a safe fallback, not
     the preferred fix, since it reduces what the controller can act on.
  5. **Then** build rung G: same `AdaptiveDRController` structure as rung F, with the priority
     score's uncertainty term reinstated and weighted according to the (now-calibrated) result.

**Research-process note worth preserving:** step 2 is a clean example of a plausible-sounding
physics argument that turned out to be wrong when actually tested — logged here specifically so the
same (incomplete) reasoning isn't repeated. The corrected direction in step 3 was derived by working
through what the *equations* actually require to separate the two parameters, rather than reasoning
qualitatively about "changing the regime" — a useful lesson for how to propose the *next* fix if
step 3 also doesn't fully work.
**Blocks:** Any claim that this project's PINN contribution (as opposed to generic adaptive DR)
adds value. This is the single most important remaining piece for the project's original novelty
claim ("RAPP-initialized physics-informed adaptive domain randomization").

### 5.5 — Priority 5: Full experimental matrix (A–G), properly ablated

**What:** Run the roadmap's full Table 7 matrix (A: human baseline, B: Eureka baseline, C: human+DR,
D: Eureka+static DR, E: Eureka+generic adaptive DR no RAPP, F: Eureka+RAPP+adaptive DR [done this
session], G: +PINN) with the injection-calibration and placebo controls the roadmap specifies
(§25 "Debugging and Verification" in the original roadmap document, and the placebo-arm protocol
from the Stage-1 PROJECT.md's rung ladder: real-signal vs shuffled-signal vs no-signal arms for G).
**Why:** A and B exist; C/D/E/F/G do not all exist yet in a controlled, matched form (D/F exist as
of this session but not C or E). Placebo controls are what would let this project claim the PINN
signal (once calibrated, §5.3) is *load-bearing* rather than incidental.
**How:** Build on `run_adaptive_dr.py`'s structure; E is F with the controller fed random/uniform
widening instead of measured `S_d` (a one-line change — swap the observation for noise, keep
everything else identical, which is itself the cleanest placebo arm). C is a straightforward
addition (vanilla reward + static DR, same protocol as D).
**Blocks:** The project's publication readiness. Depends on §5.1 (multi-seed) being done first or
concurrently — an ablation matrix with single-seed arms is not a defensible result.
**Scale warning:** this is the first item in this plan that genuinely needs a dedicated compute
budget (7 variants x 3+ seeds x 800+ iterations) rather than an afternoon on a shared machine —
plan for this explicitly with whoever controls GPU allocation before starting it.

### 5.6 — Priority 6: Track B (LEAP Hand) — REVISED, major correction this session

**This section's original plan ("port `AdaptiveDRController` to LEAP from scratch") is superseded.**
Close reading of `LEAP_Hand_Isaac_Lab`'s own source (`leap_hand_env_cfg.py`, `utils/adr.py`,
`reorientation_env.py`) while the GPU was occupied by other work this session found: **the upstream
repo already ships a working, OpenAI-ADR-style adaptive domain randomization system, on by default**
(`cfg.enable_adr = True`). It widens every DR parameter in `adr_cfg_dict` by one linear increment
(of 25 total) whenever a measured performance criterion — `adr_criteria = (consecutive_successes /
z_rotation_steps) / episode_length_seconds`, i.e. rotations per second — clears a threshold
(`min_rot_adr_coeff = 0.15`).

**Verified directly against this project's own `LEAP_500` result** (TensorBoard log, not assumed):
`num_adr_increases` rose 0→8 of 25 over training, `adr_criteria` peaked at 0.315. **`results/LEAP_500.json`
is therefore NOT a no-DR / rung-1 baseline — it was measured under actively-widening native ADR from
the start.** Full detail in `EXPERIMENT_LOG.md`'s "major correction: Track B already has native
adaptive DR" entry; `RESULTS_2026-09-16.md`'s Track B section needs a correcting note added.

**Revised plan — three concrete items, no longer "port the mechanism":**
1. **Run a true no-ADR baseline** (`env.enable_adr=false`, confirmed this session to be a valid
   hydra override reaching `LeapHandEnvCfg` — same mechanism used for every 2-DOF override today;
   verification hit only the already-known, unrelated `batch_size % minibatch_size` rl_games
   assertion from using too few envs for a smoke test, not a new bug). This is the one run actually
   required before any LEAP comparison means anything, and it is cheap (one more training run at
   LEAP's normal 4096–8192-env scale).
2. **RAPP-initialize LEAP's native ADR**, rather than letting it widen from the hand-set starting
   ranges in `adr_cfg_dict`/`adr_custom_cfg_dict`. This is the actual non-redundant application of
   this project's real idea (RAPP-derived, task-feasible starting envelope) to the real platform —
   LEAP's `EventCfg` terms (`robot_joint_stiffness_and_damping`, `object_physics_material`,
   `object_scale_mass`, etc.) are the natural sweep targets, using the same one-parameter-at-a-time
   protocol as `run_rapp.py`, adapted to LEAP's own success metric instead of Cartesian terminal
   error.
3. **Compare LEAP's native linear-threshold widening rule against this session's precision-gated
   rule** (F1→F2→F4 progression) as an *alternative update rule* on the same platform — LEAP's rule
   is structurally close to F1/F2 (coarse-threshold-gated, shown this session to have a precision
   blind spot on the 2-DOF testbed); whether LEAP's task has an analogous coarse/fine distinction
   (e.g. rotating fast but sloppily vs. precisely) is an open empirical question worth checking
   directly rather than assuming the 2-DOF finding transfers.

**Blocks:** Nothing upstream; this is naturally the last item before hardware, per the roadmap's
own staging (§30 "Sim-to-Real Strategy": nominal sim -> training DR -> unseen sim -> boundary tests
-> system ID -> hardware transfer, in that order, no skipping). **Status: scoped and ready to
execute, not yet run** — this session's GPU time went to the 2-DOF multi-seed/PINN queue instead.

### 5.7 — DONE this session: physics-performance map extended to the full metric vector

`physics_binned_metrics` added to `source/pi_adr/evaluation/metrics.py` (alongside, not replacing,
the original `physics_binned_success`) and wired into `eval.py` as `physics_performance_map_full`
in every `--dr` evaluation's output JSON from today onward. Reports the full
`M(xi) = [P_5cm, P_2cm, P_1cm, E_d, E_qdot, E_tau, T_settle]` vector per bin, using quantities the
`RolloutAccumulator` already computed (new `per_env_mean_velocity` / `per_env_mean_effort` /
`per_env_settling_time` properties added to expose them per-environment rather than only as
run-wide scalars). Not yet re-run against Session-1's existing DR evaluation JSONs retroactively —
doing so would need the original rollouts, which weren't saved at per-env granularity, only their
reductions, so this only applies going forward from today's runs (D/F's `*_DRtrain`/`*_DRshift`
results above already include it).

### 5.8 — Housekeeping (do opportunistically, not urgently)

- Delete or clearly mark `results/superseded/` entries' relationship to current numbers (already
  separated from the main table by Session 1, just needs a one-line README in that subdirectory).
- `.libshim/` (the library shims for URDF conversion) should be documented as a one-time, Ubuntu-26.04
  -specific workaround in the top-level `README.md`'s "Known issues" if a future session re-images
  this machine or moves to a different OS — currently only mentioned in passing.
- `pyproject.toml` has no pinned dependency versions; low risk while riding on the already-built
  `env_isaaclab` venv, but worth a `pip freeze` snapshot (`requirements-lock.txt`) before any
  environment rebuild is attempted, since Isaac Lab's own dependency pins are notoriously strict.

---

## 6. Research-integrity notes (read before citing any number from this project)

1. **No number in this project should be treated as final until it has survived §5.1 (multi-seed).**
   Treat every single-seed result (which is currently all of them) as a *pilot* finding: directionally
   suggestive, not a claim.
2. **The Stage-1 report's own numbers remain unverified against any artifact on this machine.**
   Session 1 established this explicitly; it has not changed. Where this project's numbers disagree
   with Stage 1 (vanilla precision, most notably), the disagreement is reported, not resolved in
   either direction — do not assume this project's rebuild is more correct than the original report
   without the missing original artifacts to check against.
3. **Every negative result in this project (PINN uncertainty, vanilla reproduction failure) is kept
   in the log, not quietly dropped.** This is a deliberate choice recorded here so it is not
   accidentally reversed by a future editing pass that only wants to show positive results.
