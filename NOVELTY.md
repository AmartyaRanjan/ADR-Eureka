# What This Project Adds to Eureka and DrEureka

A detailed account of the lineage, the gap, the contributions, and the process.

Companion documents: `PROJECT.md` (system design), `dexreward-system/docs/`
(measured engineering findings), `dexreward-system/llm/README.md` and
`patches/README.md` (implementation specifics).

---

## 1. The lineage, stated precisely

Getting the novelty right requires being exact about what the prior work
already does, because a vague description of the baseline makes almost
anything look novel.

### 1.1 Eureka (Ma et al., ICLR 2024)

Eureka automates reward *code* authorship:

1. The **unmodified environment source code** plus a natural-language task
   description go to a coding LLM as context.
2. The LLM writes an executable reward function. Critically, it must return a
   **dictionary of named components**, not a single scalar —
   `(total_reward, {"pos_reward": ..., "rot_reward": ...})`.
3. K candidates are sampled per iteration and each is trained with PPO in
   massively parallel GPU simulation.
4. A **fitness function `F`** — human-defined, task-specific, held separate
   from the generated reward — ranks the candidates. For in-hand
   reorientation this is consecutive successful reorientations.
5. **Reward reflection**: the per-component statistics are tracked at
   intermediate training checkpoints and summarized as *text* back to the
   LLM. This is what makes targeted editing possible — the LLM can see that
   `align_bonus` never fired, or that `action_reward` dominates everything
   else.
6. The best reward is mutated in-context and the loop repeats.

The insight that makes Eureka work is item 2 + item 5 together: **decomposing
the reward into named components turns an opaque scalar score into a
diagnosable signal.** Without the component dict, reflection has nothing to
report but "it scored badly."

### 1.2 DrEureka (Ma et al., RSS 2024)

DrEureka extends this to sim-to-real by automating domain randomization:

1. A **safety instruction** is appended to the task description in natural
   language rather than as a hand-added penalty term. The observation was
   that bolted-on safety penalties tend to dominate and produce degenerate
   over-conservative policies; letting an instruction-tuned LLM balance the
   terms itself works better.
2. **RAPP (Reward-Aware Physics Prior)**: an initial policy is rolled out
   under swept values of each randomization parameter (friction, mass,
   damping, …), one at a time. Values where the policy still satisfies a
   success criterion define that parameter's feasible bounds. This makes the
   DR range *dependent on the policy that actually exists*, rather than being
   a blind guess.
3. The LLM is shown the RAPP bounds and asked to choose a subset of
   parameters and a range within those bounds.
4. **The chosen ranges are then fixed for the remainder of training.**

DrEureka's own limitations section names two things as future work: that DR
parameters are static once generated, and that there is no mechanism for
predicting which of several candidate policies will transfer best.

---

## 2. The gap

### 2.1 Reward reflection reports *what*, never *why*

Eureka's reflection channel is genuinely informative about reward-component
behaviour. What it cannot express is the *cause* of failure. Consider two
completely different failures:

- **A reward-logic error.** The LLM wrote `rot_reward` with an inverted sign,
  so the policy is rewarded for rotating *away* from the goal.
- **A physics mismatch.** The reward is correct, but the simulator's contact
  model lets the cube stick to the fingertips in a way no real cube does, so
  the policy learns a strategy that only works in simulation.

In the reflection text, **these look nearly identical**: low fitness, some
components flat, others saturating. The LLM receives the same kind of
evidence in both cases and has only one tool — rewrite the reward.

For case A that is correct. For case B it is actively harmful, and that
harm is invisible until deployment.

### 2.2 Static DR is calibrated against a policy that no longer exists

RAPP's bounds are derived from `π_initial` — the policy *before* the real
training run. But DR range feasibility is a property of the *current* policy:
a robust late-stage policy can tolerate much wider randomization than the
early one, and a policy that is failing to learn at all may need ranges
narrowed before it can get traction. Fixing the ranges at iteration zero
means the randomization is mis-calibrated for most of training — too easy
late, potentially too hard early.

### 2.3 There is no channel through which "the simulator is wrong" can be said

This is the deepest gap. In both systems, every piece of feedback flows into
one action: *edit the reward*. There is no representation for "the reward is
fine, the physics is the problem," and no pathway that such a diagnosis could
act through even if it existed.

---

## 3. Contribution 1 — Adaptive domain randomization

Re-evaluate DR ranges every N reward iterations, driven by the same
performance feedback that drives reward edits: widen when the policy is
robust, narrow when it cannot learn.

This follows the pattern established by OpenAI's ADR work, but with a
difference that matters: in ADR, reward and curriculum are separate concerns
with a hand-designed reward. Here, **the DR controller and the reward author
share one feedback channel**, so the randomization schedule and the reward
function co-adapt.

Honest framing: DrEureka reports having tested intra-training DR adaptation
and found it unnecessary *for their tasks* (quadruped locomotion, which
converges readily). The hypothesis here is that adaptation matters precisely
where the feasible DR range is narrow and the task is contact-dominated —
in-hand manipulation — and that claim has to be demonstrated, not assumed.

**Implementation note.** `leap-hand/LEAP_Hand_Isaac_Lab` already ships a
working ADR module (`utils/adr.py`, `class LeapHandADR`) that progressively
widens `EventTerm` ranges on a fixed schedule. That is the *mechanism*. Our
contribution is wrapping it with LLM-driven range **selection** at each
decision point, closing the loop that ADR leaves open-loop and that DrEureka
leaves one-shot.

---

## 4. Contribution 2 — Physics-informed failure diagnosis

This is the core of the project, and the part where PINNs earn their place.

### 4.1 What the physics residual actually measures

A physics-informed neural network is trained on simulated hand+object state
transitions with a loss that includes a **differentiable physics-residual
term** derived from the manipulator's equations of motion:

```
R_phys = M(q)·q̈ + C(q,q̇)·q̇ + G(q) + F_contact − τ
```

where `M` is the mass/inertia matrix, `C` the Coriolis/centrifugal terms, `G`
gravity, `F_contact` the contact forces, and `τ` the applied joint torques.
For an idealized rigid-body system obeying these dynamics exactly, the
residual is zero.

The residual is therefore a measure of **how far the observed simulated
behaviour departs from consistent rigid-body physics** — i.e. how much of
what the policy is experiencing is simulation artifact rather than physics
that would survive contact with a real robot.

Our loss follows SOPDNA's composite shape:

```
L_total = λ_d·L_data + λ_p·L_phys + λ_c·L_ctrl
```

(supervised state-prediction error, physics residual, control-smoothness
regularization). Implemented in `dexreward-system/pinn/losses.py`.

### 4.2 Why a *learned* physics model rather than a physics checker

One could imagine simply computing the residual analytically at each step.
The PINN formulation buys three things:

1. **Evaluation at arbitrary states**, including counterfactual ones the
   policy did not visit — useful for asking "would this region be
   physically consistent?"
2. **A continuous, differentiable signal** rather than a binary flag, which
   can be aggregated, thresholded, and gated.
3. **Latent parameter discovery.** Following SOPDNA's treatment, unmodelled
   effects — variable friction, actuator degradation — can be represented as
   trainable latent parameters (`δ`) recovered by the residual loss itself,
   rather than assumed known. This is the path from "detect mismatch" toward
   "identify *which* physical parameter is mismatched," which is exactly the
   information the DR controller needs.

### 4.3 The hard part, stated honestly: contact

Both of the closest prior works handle only smooth dynamics:

- **PI-DDPG** (Fareh et al., 2025) uses a rigid-body Lagrangian model with
  *no friction or contact term at all*, on a 4-DOF arm doing free-space
  trajectory tracking.
- **SOPDNA** (Singla et al., 2026) includes a lumped disturbance/friction
  term `F_d` and even learns it — but as a *continuous* quantity inside a
  continuous dynamics equation. No contact/no-contact switching, no impact
  events.

In-hand manipulation is the opposite regime: **intermittent, multi-point,
discontinuous contact is the entire task**. A soft/relaxed contact
formulation is required to keep the residual differentiable, and that
relaxation means in-contact residuals are partly artifact of the relaxation
itself rather than evidence about the simulator.

This is the project's primary technical risk and it is treated as such:

- The residual is computed everywhere but the **diagnostic** signal is
  **contact-phase gated** — weighted toward low-contact segments where the
  residual is trustworthy.
- Contact-free and contact-phase residuals are **reported separately, never
  aggregated into one number** (`physics_loss()` in `pinn/losses.py` returns
  both).
- Shadow Hand reorientation, which has genuine free-flight phases between
  regrasps, is the easier validation case; LEAP z-axis rotation, which is
  contact-saturated throughout, is the harder and more representative one.

---

## 5. The routing rule — the insight that makes the PINN measurable

This is subtle, and it is the part most likely to be missed by a reader who
skims the architecture diagram.

### 5.1 The failure mode: reward shunting

Suppose the PINN signal is simply appended to the reflection text alongside
the reward-component statistics, and the LLM is left to use it however it
likes. Being very good at editing rewards, the LLM will do the natural thing:
**fix the symptom in reward space.** Faced with "physics is inconsistent in
region X," it writes a penalty term steering the policy away from region X.

Observe what happens next:

- Fitness improves. The policy avoids the badly-modelled region.
- The ablation reports that adding the PINN improved performance.
- **But the physics error was never addressed — it was absorbed into the
  reward function.**
- That reward encodes an avoidance of a *simulator artifact*, and the real
  robot has no such artifact. The learned avoidance is dead weight or worse
  on hardware.

So the naive integration produces a result that looks like success, hides the
actual cause, and *damages* the sim-to-real transfer the project exists to
improve.

### 5.2 The rule

Diagnosis and action are separated, and the separation is enforced in code
rather than requested in a prompt:

**Stage 1 — Classify.** The LLM receives reward reflection *and* the physics
signal and must first emit a diagnosis: `reward-logic error` or
`physics/dynamics mismatch`.

**Stage 2 — Route.** The diagnosis determines which edit pathway is even
available:

| Diagnosis | Permitted action | Forbidden |
|---|---|---|
| reward-logic error | Edit the reward code | Touching DR ranges |
| physics mismatch | Update DR ranges | Editing the reward code |

The LLM is additionally prohibited from writing reward terms that reference
DR parameters, physics residuals, or dynamics-mismatch avoidance.

Implemented in `dexreward-system/router/diagnosis_router.py`. A dual
(`BOTH`) diagnosis is **deliberately rejected** rather than allowed to touch
both pathways — forcing a single diagnosis per iteration is what keeps the
routing audit interpretable as a confusion matrix against injected ground
truth.

### 5.3 Why this is a contribution and not just plumbing

The routing rule does three things at once:

1. **Makes the PINN's contribution measurable.** If physics diagnoses could
   also trigger reward edits, no ablation could separate "the physics signal
   helped" from "the extra text helped the LLM write better rewards."
2. **Prevents the transfer-harming failure mode** described above.
3. **Forces the physics signal to be load-bearing.** Because a physics
   diagnosis is the *only* thing that can move DR ranges, the signal cannot
   be quietly ignored — it is a routing decision, not optional flavour text.

---

## 6. Contribution 3 — Validating that the diagnosis is real

Arguably the most transferable contribution, and the one with the least prior
art: **nobody in this literature validates a failure-attribution signal
against ground truth.**

The claim "high physics residual ⇒ the failure is a dynamics mismatch" is an
inference, not a fact. A badly-shaped reward can drive the policy into
contact-rich, poorly-modelled states, producing a high residual from what is
purely a reward bug. Without a test, the router could be confidently routing
on a confounded signal.

### 6.1 Injection calibration — a hard gate before any rung-3 claim

Take a working baseline and separately inject:

- **Known reward bugs** — sign flips, wrong scales, missing terms.
- **Known dynamics perturbations** — mass ×1.5, friction ×0.3, added damping.

Then measure whether the residual **discriminates** the two, reported as an
AUC — separately for contact-free and contact-phase segments.

Implemented in `dexreward-system/validation/injection_harness.py`, including
the decision gate as executable logic rather than a judgement call made under
deadline pressure:

| Outcome | Action |
|---|---|
| Contact-free AUC strong, contact-phase weak | **Expected.** Proceed with contact-gated routing as designed. |
| Contact-free AUC also weak | **Stop.** Do not build the full router. Fall back to a simpler heuristic for one more calibration round, then decide whether Contribution 2 is reportable as a negative result. |

A crucial methodological detail: **the injected bug taxonomy must be derived
from failures actually observed during baseline reproduction**, not invented.
A taxonomy of imagined bugs measures how well the PINN detects bugs no real
LLM produces. `validation/injection_taxonomy.md` is deliberately left empty
until real Phase-1 failures exist.

### 6.2 Placebo arms

Even a validated signal might not be *used*. Three arms at identical compute
budget:

1. Real PINN signal
2. **Shuffled** signal — same format, uninformative content
3. No signal

If (1) ≈ (2), the LLM is ignoring the channel or the channel is noise, and
the contribution is unsupported regardless of how good the AUC was. This is
the single strongest defence the project can offer a skeptical reviewer.

### 6.3 Routing audit

Every iteration's diagnosis → edit-type decision is logged. Against the
injection study's ground truth this yields a confusion matrix for the
*whole loop*, not just the PINN in isolation: does a physics injection
actually result in a DR edit downstream?

---

## 7. The process, end to end

```
  task spec + safety instruction + environment source code
                          │
                          ▼
              ┌───────────────────────┐
              │  LLM: generate K      │◄──────── reward-logic diagnosis only
              │  candidate rewards    │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Parallel PPO         │◄──────── DR ranges φ
              │  (Isaac Lab)          │
              └───────────┬───────────┘
                          │ telemetry: joint states, torques,
                          │ contact events, object pose, DR params
              ┌───────────┴───────────┐
              ▼                       ▼
    ┌──────────────────┐   ┌────────────────────┐
    │ PINN             │   │ Reward reflection  │
    │ physics residual │   │ + fitness F        │
    │ (contact-gated)  │   │ (per-component)    │
    └────────┬─────────┘   └─────────┬──────────┘
             └──────────┬────────────┘
                        ▼
             ┌────────────────────────┐
             │  Diagnosis router      │
             │  classify → route      │
             └───┬────────────────┬───┘
     reward-logic│                │physics mismatch
                 │                ▼
                 │      ┌────────────────────┐
                 │      │ Adaptive DR        │
                 │      │ widen / narrow φ   │
                 │      └────────────────────┘
                 └──────► (back to LLM reward edit)
```

Per iteration:

1. **Generate.** K reward candidates from the LLM, given task spec, safety
   instruction, environment source, and the previous round's routed feedback.
2. **Train.** Each candidate trained with PPO under the current DR
   distribution φ.
3. **Score.** The task fitness `F` — inherited unchanged from the published
   benchmark, and **never augmented with physics terms** so that candidate
   selection optimizes for task success rather than apparent physical
   tidiness.
4. **Observe.** Full-rate telemetry feeds two channels: the PINN (physics
   consistency, split by contact phase) and reward reflection (per-component
   statistics, fitness).
5. **Diagnose.** The router classifies the dominant failure mode.
6. **Route.** Reward-logic → reward edit. Physics → DR range update. Never
   both.
7. **Repeat**, with DR re-evaluated every N iterations.

---

## 8. Novelty, honestly delimited

| Component | Eureka | DrEureka | This project |
|---|---|---|---|
| LLM reward generation | ✅ | ✅ inherited | ✅ **inherited unmodified** (deliberately, for ablation validity) |
| Reward reflection | ✅ | ✅ | ✅ inherited |
| Fitness function | ✅ | ✅ | ✅ inherited unchanged |
| Domain randomization | — | ✅ static, one-shot | ✅ **adaptive, closed-loop** |
| Physics-based failure attribution | — | — | ✅ **new** |
| Diagnosis→action routing separation | — | — | ✅ **new** |
| Ground-truth validation of the diagnosis | — | — | ✅ **new** (no prior art found) |
| Cross-morphology replication | — | — | ✅ Shadow Hand + LEAP Hand |

**What is explicitly *not* claimed as novel:** the reward-generation loop,
evolutionary search, reward reflection, the fitness metrics, PPO, or PINNs as
a technique. All are inherited, and inheriting them unmodified is what makes
the ablation ladder interpretable.

### Relation to the closest prior work

- **PIN-WM** (Li et al., 2025) — physics-informed world model with
  "physics-aware digital cousins" (randomize *around identified* parameters).
  The closest thing to principled DR in the literature, and a genuine
  competitor to our Contribution 1. But: non-prehensile pushing, no LLM
  rewards, no failure attribution.
- **PI-DDPG** (Fareh et al., 2025) — PINN-adjacent RL on a manipulator with
  Lyapunov reward shaping. Fixed hand-designed reward, 4 DOF, free space, no
  sim-to-real DR.
- **SOPDNA** (Singla et al., 2026) — the composite physics loss we borrow the
  shape of. Not RL, no reward, no LLM.
- **REvolve** (Hazra et al., 2025) — LLM reward evolution with true genetic
  operators and human-feedback fitness. Needs humans in the loop; no DR, no
  physics signal.

**No prior work combines LLM reward generation, adaptive DR, and physics-based
failure attribution; and none validates an attribution signal against injected
ground truth.**

---

## 9. Known risks, not hidden

1. **Contact discontinuity may defeat the residual.** The central technical
   risk (§4.3). Mitigated by contact gating and separate reporting, and
   gated by the injection study — which is deliberately run *before* the full
   router is built, so this can fail cheaply.
2. **The λ weights have no literature default.** Both source papers leave the
   loss weighting as an untuned free hyperparameter. Ours are explicit
   tunable config, not invented constants.
3. **Compounding adaptivity.** Each adaptive subsystem is another way for the
   policy to overfit a moving target. Mitigated by freezing DR and PINN state
   before final transfer evaluation.
4. **Hardware ceiling on candidate count.** Measured: this machine sustains
   **2 concurrent workers** against Eureka's published **16** — an 8× gap
   that tuning cannot close (`docs/concurrency_limits.md`). Whichever
   compromise is chosen becomes a stated deviation from the reproduced
   baseline, not a silent one.
5. **Single task per platform.** Isaac Lab ships ~4-5 hand task families, not
   Bi-DexHands' 20. Breadth is future work and must be stated as a
   limitation rather than quietly narrowed.
6. **Transfer claims differ per platform.** LEAP has real hardware and a real
   zero-shot claim; Shadow Hand is sim-to-OOD-sim against a held-out
   perturbed configuration. These are different evidentiary strengths and
   must never be conflated in a writeup.
