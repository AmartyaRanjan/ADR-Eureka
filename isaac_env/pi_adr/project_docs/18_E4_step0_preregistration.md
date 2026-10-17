# 18 — Pre-registration: E4 step-0 diagnostic — does LEAP policy quality degrade before rotation count?

**Written before any diagnostic evaluation was run.** Analysis-only on the 9 existing checkpoints (no training). It is the pre-declared "step 0" of E4 (doc 15), triggered by the E2 decision rule (doc 17: ≥ 2 of 3 pairings in the hypothesised direction — formally met, though with trivial margins and p ≥ 0.8).

## Motivation
LEAP's ADR widens when *rotations per second ≥ 0.15* (quantity only). On the 2-DOF track the analogous coarse gate hid precision loss (doc 09). The E4 idea (add a quality term to the ADR criterion) is only worth building if, on LEAP, **quality degrades at lower difficulty than rotation count does**. The E2 curve shows rotation count is flat from ADR 10 to 15 and collapses at 20; whether falls rise earlier is unmeasured.

## Measurement (fixed)
`scripts/eval_leap_quality.py` = the frozen evaluation protocol of `eval_leap.py` (1024 envs, 600 control steps, eval seed 1234, ADR frozen) plus, per policy and ADR level: **falls per hand per 20 s** (episodes ended by the cube leaving reach or flipping, `reset_terminated`), **time-outs**, and **mean seconds between falls**. ADR ∈ {0, 5, 10, 15, 20, 25} × 9 checkpoints = 54 evaluations. Rotations/20 s are recomputed in the same run and must reproduce the existing values to within 1 % (consistency check; failure = the new script is not trusted).

## Pre-declared criterion — "a quality-before-quantity gap exists"
For each condition (L0, L1, L2), using seed means: let *q-level* be the lowest ADR at which rotations/20 s ≥ 80 % of the ADR-0 value is **still true** (the quantity gate has not yet fired), and compare falls/20 s there to ADR-0. The gap exists for a condition if, at some ADR ≤ q-level that has rotations ≥ 80 % of ADR-0, **falls per 20 s ≥ 2× their ADR-0 value** (and at least +0.5 falls/20 s absolute, to avoid ratios on near-zero). The diagnostic **supports building E4** only if the gap exists in **at least 2 of 3 conditions** *and* is present in at least 2 of 3 seeds within each of those conditions (per-seed evaluation of the same rule).

## Decision
- Supports E4 → write E4's design and pre-registration, then implement `adr_criteria` with a quality constraint (drop rate threshold calibrated on L1 at ADR 0) — that costs ≈ 35–40 GPU hours and is run only after E1.
- Does not support E4 → **do not build E4**; report on LEAP: *"the ADR quantity gate is aligned with the quality collapse at this budget (no quality-before-quantity gap), so the 2-DOF coarse-gate failure mode does not transfer here."* That is a useful negative result for the paper.

## Rules
All 54 evaluations complete before interpretation; transient simulator start-up errors are retried; results reported either way.
PREREG_TIME: 2026-10-14 12:21:11 UTC
