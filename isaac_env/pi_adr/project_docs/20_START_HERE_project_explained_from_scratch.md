# START HERE — The whole project explained from scratch

**Who this is for:** someone who knows what reinforcement learning (RL) is, but knows nothing about this project, the tools, or the jargon (Eureka, RAPP, PINN, ADR, domain randomization, …).
**What you will get:** the story of the project in plain language — what problem we were trying to solve, every idea involved explained from zero, every experiment we ran and *why*, what each number means, what we found, which of our own claims turned out to be wrong, and where things stand right now.
**How to read it:** top to bottom. Each part uses only ideas explained earlier. The other documents in this folder (`00_INDEX.md` … `19_…`) are the detailed technical records; this one is the guided tour. Where useful it points to them.

> **Honest one-sentence summary of the project:** we tried to show that a pipeline of smart components (an AI-written reward, a physics-aware randomization strategy, and a physics-learning neural network) makes robot-learning better — and after very careful testing we found that, on our tests, the components did **not** reliably beat much simpler baselines; the lasting value is a validated testbed, a rigorous way of testing, several diagnosed pitfalls, and a couple of genuine findings (and, if the last experiments hold up, one clear explanatory result).

---

## Part 0 — Contents

1. RL refresher, in just the parts this project needs
2. The big picture: what problem are we attacking?
3. The ideas, one at a time (reward design, Eureka, the "precision problem", simulation and the sim-to-real gap, domain randomization, adaptive DR, RAPP, physics observers/PINNs, uncertainty)
4. The two testbeds (a toy robot arm and a robot hand)
5. How we measure things (metrics, seeds, statistics, placebo controls)
6. The story of the project, step by step
7. What we found, in plain words
8. How we checked that findings were real (and the ones that were not)
9. What went wrong along the way (bugs and lessons)
10. Where the project stands today and what happens next
11. Glossary
12. Where everything lives on disk

---

## Part 1 — RL refresher (only what you need here)

You already know the basic loop: an **agent** (a neural network called the **policy**) looks at an **observation**, picks an **action**, the **environment** responds with a new observation and a **reward** number, and training adjusts the policy to collect more total reward.

Four details matter a lot in this project:

1. **The reward function is chosen by a human (or, here, an AI).** The agent will optimise exactly what the reward says, not what you meant. This is the root of the "precision problem" below.
2. **PPO (Proximal Policy Optimization)** is the training algorithm we use everywhere. It collects experience, then improves the policy in small, safe steps ("clipped" updates so one bad update cannot wreck it). A "training iteration" or "epoch" in this project means one round of: collect a batch of experience from many parallel robots → update the policy.
3. **Exploration vs. exploitation and the entropy bonus.** The policy outputs a *distribution* over actions (a mean and a standard deviation, σ). Early on σ is large (lots of random trying). As training proceeds σ shrinks (the agent becomes confident). **Entropy** measures how spread-out the distribution is. PPO adds a small bonus (`entropy_coef`) for staying spread out so the agent does not stop exploring too early. If σ collapses too early the agent can get stuck in a mediocre habit — we call this **exploration collapse**, and it matters a lot later.
4. **Many parallel simulated robots.** Instead of one robot, GPU simulators run thousands of copies at once (here 2048–4096). This makes training fast and makes "one training run" a single number out of a noisy process: **running the same training twice with a different random seed gives different results.** Hold that thought — it is the most important fact in this whole project.

---

## Part 2 — The big picture: what problem are we attacking?

### 2.1 Teaching robots in simulation, then using them in the real world
Training robots in the real world is slow and risky (things break). So we train in a physics **simulator** and later transfer the policy to the real robot ("**sim-to-real**"). The catch: the simulator is never perfectly like reality — friction, motor strength, mass, delays are all slightly off. A policy that is brilliant in the simulator can fail on the real robot. This mismatch is the **sim-to-real gap**.

### 2.2 Two popular tricks, and a research idea combining them
- **Trick 1 — let an AI write the reward (Eureka).** Designing a good reward by hand is hard and time-consuming. A recent approach called **Eureka** (NVIDIA research) uses a large language model (LLM) to write the reward function's code, tries it out by training, looks at the outcome, and revises it — repeating until the reward produces good behaviour.
- **Trick 2 — randomise the simulator (domain randomization).** Instead of one fixed simulated world, train the robot across many slightly different worlds (different friction, different motor strength, …) so it learns to cope with variation, hopefully including reality. A follow-up approach, **DrEureka**, also uses an LLM to help choose how much to randomise; its component for finding the safe range of randomisation is called **RAPP**.
- **The research idea of this project (the "Stage-2 roadmap"):** combine them and add a smart ingredient. Pipeline: **Eureka reward → RAPP physics envelope → a neural network that figures out the physics (a "PINN observer") → adaptive domain randomization that uses the observer's knowledge** to choose how to vary the world during training. The hope: this pipeline would beat simpler alternatives, i.e. give a more robust robot.

### 2.3 What we actually did
We were given two documents: a **Stage-1 report** (earlier work on a simple robot arm, claiming the Eureka reward reaches the goal quickly but is imprecise, while a human-made reward is precise) and the **Stage-2 roadmap** (the pipeline above plus a list of experiments called an *ablation matrix*, rungs A–H). Our job: **build the testbed, reproduce Stage 1's claims, and run the Stage-2 experiments**, citing sources, keeping evaluation identical across experiments, and logging everything.

An **ablation** means: remove or swap one ingredient at a time and see what changes. It is how you find out which ingredient is actually responsible for any improvement.

---

## Part 3 — The ideas, one at a time

### 3.1 Reward design and "the precision problem"
In the main toy task, a robot arm must touch a goal point. Two rewards were compared:

- **Vanilla (human) reward:** `reward = −distance + 5 (if within 5 cm) − 0.005·(effort)²`. Closer is better, a bonus for "success", a small penalty for wasting effort.
- **Eureka reward:** `reward = e^(−10·distance) + 100 (if within 5 cm) − 0.01·(speed)² − 0.1`. A smooth "getting closer" term, but a **huge** +100 bonus once within 5 cm.

**Why this matters:** once the arm is within 5 cm, the +100 bonus is already collected. The extra reward for improving from 3 cm to 3 mm is tiny by comparison. So the agent has almost **no incentive to get more precise than 5 cm**. The Stage-1 report called this the *coarse-success vs. fine-precision* problem: the policy succeeds at the coarse 5 cm threshold almost every time (~99 %) but only reaches 2 cm in roughly a quarter of episodes and 1 cm in under a tenth.

This single phenomenon recurs throughout the whole project (it even shows up inside the controller we built), so remember it: **an objective that looks satisfied at a coarse threshold can hide failure at the precision that matters.**

### 3.2 Eureka, a little more concretely
Eureka's loop: (1) give the LLM a description of the task and the simulator code; (2) it writes several candidate reward functions; (3) each candidate is used to train a policy; (4) the results (and statistics about reward components) are fed back to the LLM, which writes improved candidates; (5) repeat and keep the best. In our project **we did not run this search ourselves** — the Eureka reward used for the arm was the one given in the Stage-1 report. (That matters later: the *cost* of the search was not included in any comparison we made.)

### 3.3 Simulation, the physics engine, and Isaac Lab
We use **Isaac Sim** (NVIDIA's physics simulator, PhysX under the hood) and **Isaac Lab** (a robot-learning framework on top of it). They let us run thousands of robot copies on one GPU. Training uses **rl_games** (a PPO implementation). Everything ran on one NVIDIA L4 GPU that was **shared with other people's jobs** — which shaped how we worked (small runs, waiting for free memory) and caused several operational problems (Part 9).

### 3.4 Domain randomization (DR)
Each parallel robot copy can have **different physics**. We randomise four things on the arm:
- **friction** (how much the joints resist motion),
- **damping** (a kind of "thick syrup" resistance proportional to speed),
- **actuator effectiveness** (how strong the motors really are compared to nominal; 1.0 = nominal, 0.5 = half strength),
- **link mass** (how heavy the arm segments are).

A policy trained across variety should cope with surprises.

**Static DR:** choose ranges once and keep them fixed. **Too narrow** → not robust. **Too wide** → the task becomes too hard and training fails.

### 3.5 Adaptive DR (ADR)
Start with a narrow range and **widen it as the policy gets good** (and hold or shrink if it struggles). This is a curriculum on the difficulty of the world; OpenAI famously used it to train a robot hand to solve a Rubik's cube. The challenge is deciding *when* and *how fast* to widen.

### 3.6 RAPP — finding the safe region
RAPP ("Reward-Aware Physics Prior", from DrEureka) answers: *"How different can the physics be before my current (nominal) policy stops working?"* You vary **one** parameter at a time (others normal), run the policy in each setting, and mark a value "feasible" if it still succeeds often enough (we used: reaches within 5 cm ≥ 90 % of the time). The union of feasible ranges is the **RAPP envelope**; it tells you the outer limits of sensible randomisation. In our project it served as the *ceiling* for adaptive DR.

### 3.7 The physics observer and the PINN
The roadmap's key new idea: a neural network that **infers the robot's physical parameters from watching it move**. Imagine watching a short video of the arm and estimating "the second link is heavier than usual and the motor is weak." Formally the network sees a short window of (joint positions, velocities, commanded torques, accelerations) and outputs the physical parameter vector ξ = [mass₁, mass₂, friction₁, friction₂, damping₁, damping₂, effectiveness₁, effectiveness₂].

It is called a **PINN (physics-informed neural network)** because it is trained not only to match the true parameters but also to obey the **equations of motion** of the arm (a physics formula relating torque, acceleration and the parameters). The hope: knowing the physics lets an adaptive controller decide *which* aspects of the world are worth practising more.

### 3.8 Uncertainty: ensembles and why it matters
We train **5 copies** of the observer (an **ensemble**) with different random starts. Where the 5 disagree, the network is "unsure". That disagreement is a proxy for **uncertainty**. The roadmap proposed using it to steer the curriculum: practise more where the observer is unsure. For that to work, the uncertainty must actually **track how wrong the observer is** — and we measured that it mostly does not (Part 7).

### 3.9 Identifiability — why some parameters can't be separated
A subtle but important concept. Suppose a robot link accelerates a certain amount when pushed. A **heavier link with a stronger motor** can produce exactly the same acceleration as a **lighter link with a weaker motor**. From that motion alone you cannot tell which is true — the two are **not identifiable** separately; only their ratio is. Our observer kept confusing mass and motor effectiveness for exactly this reason, and a large part of the project (Part 6, step 6) was about designing experiments that break the tie.

---

## Part 4 — The two testbeds

### 4.1 Testbed A: a 2-DOF planar arm (built from scratch)
- **What it is:** two links connected by two joints, moving in a flat plane. A random target point appears; the arm must reach it within a 10-second episode.
- **Details:** links 0.5 m each; motors push with up to 20 N·m; control at 30 Hz, physics at 120 Hz; 300 control steps per episode; the policy sees an 8-number observation (joint angles, joint speeds, target position, and the vector from fingertip to target).
- **Why a toy?** Each training run takes ~10–30 minutes, so we can run **many seeds**, and the physics is simple enough to write in closed form (needed for the PINN) and to compare against a perfect scripted controller.
- **We had to build it ourselves** from the Stage-1 report because no such task existed in any software we had.

### 4.2 Testbed B: the LEAP Hand (the realistic one)
- **What it is:** a 16-joint robot hand. The task `Isaac-Reorient-Cube-Leap` is to **rotate a cube in the hand** to match target orientations, as many times as possible in 20 seconds. Score = **rotations per 20 seconds per hand**.
- **Why:** it is a real, published platform and task, so settings can be copied from the published project; it is much harder and slower (≈1 hour per 500-iteration run).
- **A big surprise:** the LEAP code **already contains an adaptive-DR system** (switched on by default). Our first "baseline" run was therefore *not* a "no-DR baseline" at all (Part 6, step 11).

---

## Part 5 — How we measure things

### 5.1 The frozen metrics
To compare experiments fairly, evaluation must be **identical every time**. We froze a measurement protocol ("metrics version 1.0.0"):
- run the trained policy on **4096** fresh robot copies, deterministic (no random action noise);
- use the **same random seed (1234)** so every policy faces the same goals and physics draws;
- record the **final distance to the goal** at 9.93 s of the 10-s episode;
- report the **precision curve**: what fraction of episodes end within 5 cm, 2 cm, 1 cm, 5 mm, 2 mm, 1 mm;
- also **mean error**, **median**, and the **worst-decile error** (the average error of the worst 10 % of episodes — a way of measuring "how bad are the rare disasters?"), plus speed, effort and settling time.

**Reading the numbers:** "2 cm success = 27 %" means 27 out of 100 test episodes ended within 2 cm. "Mean error 37 mm" means the arm typically finished ~3.7 cm away. A perfect controller gets 100 % at every threshold down to 1 mm (we checked, Part 6 step 3).

### 5.2 Three physics conditions for evaluation
- **nominal** — the standard, un-randomised physics (the "easy" test);
- **DRtrain** — randomised within the moderate range used in training (friction 0–0.25, damping ×0.5–4, effectiveness 0.5–1.0, mass ×1–1.5);
- **DRshift** — a **harder, unseen** range (friction 0.25–0.5, damping ×4–8, effectiveness 0.3–0.5, mass ×1.5–2.5). Tests generalisation.

### 5.3 Seeds and why one run proves nothing
Because training is random, the **same experiment run with different seeds gives different numbers**. In our data, the 2 cm success rate for the same setup commonly varied by 5–17 percentage points between seeds. So a single run showing "method A beat method B by 8 points" can easily be luck. **Rule adopted mid-project: no claim from a single seed.** We use five seeds (42, 123, 7, 1, 2) for the main comparisons.

### 5.4 Is a difference real? Permutation tests in one paragraph
Say method X's five results are {30, 28, 33, 29, 31} and method Y's are {25, 27, 24, 26, 28}. Are they really different? A **permutation test** shuffles the ten numbers into two random groups of five many times and asks: "how often does shuffling alone produce a gap at least as big as what we saw?" That fraction is the **p-value**. Small p (like 0.008) → the gap is unlikely to be luck. Large p (like 0.6) → consistent with luck. With only 5 vs 5 results, the smallest possible p is 0.0079 (when every X value beats every Y value). Important fine print: if you run **many** comparisons, some will look "significant" by chance (we did 150; about 7–8 false alarms are expected at p<0.05), so we treat a lone p≈0.03 as a hint, not a finding.

### 5.5 Controls: the "placebo" idea
To test whether *information* helps, you must test against the same procedure with the information **scrambled**. Our test for the PINN-informed controller ("rung G") included a **placebo** that used the identical numbers but assigned them to the wrong parameters. If the real version can't beat the scrambled one, the information isn't what's helping.

---

## Part 6 — The story of the project, step by step

*(Dates are 2026. "Doc N" refers to a file in this folder for deeper detail.)*

### Step 1 — Set up the machine and software (15 Sep)
Before any experiment, several installation problems had to be solved: the Python environment's paths were broken after the project folder was moved (fixed with a symlink); a 3D-asset converter needed libraries missing from the operating system (worked around with a small shim); the LEAP code had incompatibilities with the newest Isaac Lab (a missing import and a missing parameter). Each fix was documented because someone reproducing the work must apply them too. (Doc 02.)

### Step 2 — Build and verify the arm environment (15 Sep)
We wrote the 2-DOF arm environment from the report's numbers, then **checked it against known truths** instead of trusting it: the simulated fingertip position matches the analytic geometry formula to 0.000 mm; zero torque causes zero drift; commanded torque is applied exactly; the rewards equal the report's equations; randomisation really changes per-robot physics (read back from the simulator, not assumed). Ten checks, ten passes — after we corrected three of the *checks themselves* that had assumed an over-simple physics. (Doc 02.)

### Step 3 — Establish baselines and the "perfect controller" test (15–16 Sep)
We trained PPO with each reward and evaluated. Then a crucial control: a **scripted controller** (inverse kinematics + a simple PD controller — an "optimal" reference, no learning) driven through the same interface. It reached **100 % success at every threshold down to 1 mm**. **What that proves:** the arm, motors and control rate are fully capable of millimetre precision, so the ≈30 mm floor of learned policies is a property of the *reward and training*, not a hardware limit. (Doc 05.)

### Step 4 — Discover the bugs that would have ruined everything (15 Sep)
Careful checking uncovered silent defects (details in Part 9 and Doc 04): the evaluation measured distance *after* the environment had auto-reset (a policy scored 0.39 % when it was truly ~93 %); damping was silently set to 0 instead of 0.05; joint limits at ±π trapped the arm and capped *any* controller at ~69 %; mass randomisation skipped a write at scale 1.0. All fixed and verified. After fixing, results were re-run; the earlier set is kept only as a record of what exposed the bugs.

### Step 5 — Reproduce Stage 1's claims (15–16 Sep)
The Eureka-reward signature **reproduced**: ~97–99 % success at 5 cm but only ~26 % at 2 cm and ~4–8 % at 1 cm. But the claim that the **human reward reaches 98 % at 2 cm and 80 % at 1 cm did not reproduce** — ours reached 13 % and 2 %. So either Stage 1 had an unreported detail or something in training was off. (Doc 05.)

### Step 6 — RAPP: how much can the physics change? (16 Sep)
We ran the one-parameter-at-a-time sweep on the nominal Eureka policy. **Result:** the policy tolerated a very wide range: friction up to 4, damping up to 256× nominal, motors at 30 % strength, links up to 4× heavier. Only lighter links (≤0.75× nominal) broke it (lighter links accelerate more per unit torque, so a policy tuned for heavier ones overshoots). **Meaning:** the safe region is huge and mostly flat, so a controller that widens randomisation inside it may have little room to add value — which later turned out to be the case. A curiosity: in the harder corners the arm moved slower and was actually *more* precise (physics forced a gentleness the reward never asked for). (Doc 06.)

### Step 7 — Why did the human reward fail? Exploration collapse (10 Oct)
Looking at training logs, the action σ had collapsed from 1.0 to ~0.02 and the policy got *worse* the longer it trained. Two single-variable probes: (V1) raise the entropy bonus 0.005→0.02; (V2) weaken the effort penalty. Only V1 helped (beating the old baseline on every metric with 3× fewer iterations). We later **re-tested at a matched 800-iteration budget with 5 seeds** and found *no significant* effect of entropy at that short budget — consistent with collapse needing a long training run to hurt. That led to the **E1 experiment** (Step 17), which tests the long run directly. (Doc 07.)

### Step 8 — Build the physics observer — and repeatedly fix it (15 Sep – 11 Oct)
This was a long saga with a clear arc:
1. **Wrong labels.** The first observer was poor. Instead of tuning the network, we checked the data: plugging the *true* parameters into the physics formula gave a *worse* error than the network. The training data itself violated physics (accelerations were computed over a long control step instead of a single physics step; saturated-speed samples were included). After fixing the data, the network trained properly.
2. **Weak uncertainty.** The ensemble's disagreement barely tracked its real error (rank correlation ≈ +0.1).
3. **Diagnosis = identifiability.** Looking at *how* the 5 members disagreed showed they confused mass with motor effectiveness (correlations of +0.5 etc.) — exactly the heavy-link/strong-motor ambiguity.
4. **Attempt 1 (refuted):** "use two torque amplitudes." We tested it; the confusion got **worse**. Re-deriving the physics showed why: the relevant term scales linearly with torque, so amplitude gives more points on the *same* ambiguous line. The mistake in the reasoning was documented.
5. **Attempt 2:** hold a constant torque until the arm reaches steady speed, where the mass term drops out of the equation. Fixed joint 1's confusion; joint 2 still confused (the joints influence each other).
6. **Attempt 3:** drive the joints one at a time ("sequential single-joint holds"). Reduced the confusion further and improved in-range accuracy — but **out-of-range predictions stayed unreliable** and uncertainty stayed weakly informative (rank correlation +0.27 in range, −0.16 out of range).
**Takeaway:** a working observer for in-range physics, a genuine identifiability insight, but not a trustworthy uncertainty signal. (Doc 08.)

### Step 9 — Build adaptive DR and watch three "wins" evaporate (10–11 Oct)
We built a controller that, every 100 training iterations, measures how well the policy does under each parameter's current range and widens (or holds/narrows) it.
- **F1** (widen quickly, gate on the coarse 5 cm success): worse at nominal physics — it widened so fast the policy spent little time on easy physics.
- **F2** (widen slowly): better at nominal, but **worst at fine precision** — because gating on 5 cm success repeats the coarse-vs-fine trap *inside the controller*.
- **F3** (gate on 2 cm) was mis-configured: its success target (90 %) was unreachable at the 2 cm threshold (no policy ever got past ~42 %), so it never widened. Caught after three steps; fixed as **F4** (target 30 %).
- **F4 on one seed looked like the cleanest win.** Then more seeds arrived: the supposed gap sat entirely inside the natural seed-to-seed spread. **Retracted.**
- **Rung E** (adaptive DR *without* the RAPP ceiling, letting mass widen to 8×): one seed showed double the precision under randomisation — but it was the second-best of three seeds. **Retracted** — except the robust part: removing the ceiling **reliably costs success at nominal physics** (5 of 5 seeds, complete separation, p = 0.008).
- **Eureka vs human reward** (single-seed claim) also **retracted** at matched budget.
This is the central methodological lesson of the project: three exciting single-seed results, three failures to replicate. (Doc 09.)

### Step 10 — The full A–G matrix, five seeds (12–13 Oct)
We ran every configuration under identical protocol and five seeds:

| Rung | Reward | Randomisation | RAPP | PINN |
|---|---|---|---|---|
| A | human | none | – | – |
| B | Eureka | none | – | – |
| C | human | static | – | – |
| D | Eureka | static, RAPP-bounded | yes | – |
| E | Eureka | adaptive, no ceiling | no | – |
| F4 | Eureka | adaptive (precision-gated) | yes | – |
| G | Eureka | adaptive + observer-informed steps | yes | frozen observer |
| Gp | same as G but information scrambled (placebo) | | | |

Outcomes (details and tables in Doc 10): A≈B≈C on mean and tail error (no reward effect; no benefit of static DR); static DR for the Eureka reward actually **lowered 2 cm success** when tested under randomisation (e.g. 40.8 %→21.4 %, p = 0.016); adaptive DR (F4, E) never beat static DR; removing the RAPP ceiling hurt nominal success; G had the best-looking tail error and least seed variance **but the placebo matched it** on nominal and in-distribution tests.

### Step 11 — LEAP's hidden ADR and the LEAP matrix (10–13 Oct)
Reading the LEAP code, we found it already ships an OpenAI-style ADR (widen the randomisation whenever the hand achieves ≥0.15 rotations/second; 25 increments available). Checking the first run's logs confirmed it widened 8 of 25 steps. So our "baseline" was an ADR run. We then ran three conditions × 3 seeds, 500 iterations each: **L0** (ADR off), **L1** (native ADR as shipped), **L2** (ADR started at difficulty 8/25 — a stand-in for "start where ADR got to"; *not* RAPP, because we never ran a LEAP RAPP sweep). All were evaluated with the difficulty **frozen** so every policy faces identical conditions.
**Results:** nominal 8.4 / 8.7 / 8.0 rotations per 20 s (L0/L1/L2) — **indistinguishable**; individual seeds within one condition varied by 4–5 rotations, more than any difference between conditions; at the hardest setting (all 25 increments) every policy collapsed to ≈1 rotation. **Meaning:** at this short budget and seed count, LEAP can't tell the conditions apart, and it can't test whether ADR helps. (Doc 11.)

### Step 12 — The PINN-informed controller vs. its placebo (12–14 Oct)
This was the pre-declared, careful test of the roadmap's main claim. A first 5-seed look showed a hint (p = 0.024) — which we treated with suspicion and **pre-declared a retest** with a fixed primary metric (worst-decile error under the hard unseen physics). The retest, run with ten seeds per arm, first gave p = 0.281 — then we noticed something odd: two runs had **identical** numbers. Investigation found a data-integrity bug (next step). After repair and a clean rerun the final answer was **p = 0.109 — not significant.** G is lower than its placebo in 8 of 10 seeds but not by enough to rule out luck, and there's no gain at all on the easier tests. **Verdict: no demonstrated benefit of the physics information.** Note the test was a weak one because the observer's uncertainty was dominated by one parameter and unreliable out of range. (Doc 10.)

### Step 13 — The contaminated retest and a truncated run (13–14 Oct)
Two data-integrity problems were found *because* we investigated a suspiciously identical result:
- **Shared folders:** two training runs launched at the same moment wrote to a folder named by the current second, so some checkpoints were mixed between arms. Auditing every run showed exactly which seeds were affected; they were quarantined and rerun sequentially.
- **A silently truncated run:** one run stopped at 600 of 800 iterations after a random simulator start-up error, but the pipeline carried on and evaluated the half-trained policy. Found when a result file was missing; quarantined and rerun.
Both were logged openly and the affected numbers were removed rather than quietly kept. (Doc 04.)

### Step 14 — Running long jobs on a shared machine (12–14 Oct)
Several time-consuming misdiagnoses happened while running the LEAP queue unattended: out-of-memory crashes from other users (the queue must *wait*, not fail), a training run that "failed" only because the upstream config stops training once the reward passes a threshold (we first blamed memory — wrong, and retracted), and a lock file held by a stray process. Lessons are in Part 9.

### Step 15 — The write-up and the plan for a "positive result" (13–14 Oct)
We produced 15+ documentation files, an experiment log, and a plan (Doc 15) for experiments that could give the paper a genuine positive finding, each with the success criterion **written down before any data** (Docs 16–19):
- **E1:** the long-run exploration-collapse experiment (5000 iterations, entropy 0.005 vs 0.02, 5 seeds each).
- **E2:** the LEAP "robustness curve" at intermediate difficulty on existing policies.
- **E4 step-0:** do LEAP policies start dropping the cube before their rotation count falls?
- **E5:** does the Eureka reward *learn faster* even though it ends in the same place?

### Step 16 — E2 result (14 Oct)
Across difficulty levels the three LEAP conditions share one curve: flat to difficulty 5, ~35 % lower by 10, flat to 15, a cliff at 20, floor at 25. The pre-registered measure (area under the curve) did not separate them (p ≥ 0.8). One un-pre-registered hint: at difficulty 20 the warm-started policy was above the native-ADR one in all three seeds — only a hint, since with 3 seeds the smallest possible p is 0.10.

### Step 17 — E1: the first clear positive result (14 Oct) 
All ten runs finished (two seeds had to be resumed after an unexplained process kill; a sensitivity check excluding them gives the same answer). **Pre-registered success criterion: met.** After 5000 training iterations on the human reward:
- with the **default entropy bonus** the arm ends at a typical **61.8 mm** error with a worst-10 % error of **346 mm**;
- with the **larger entropy bonus (0.02)** it ends at **30.4 mm** and **66 mm**;
- every one of the five larger-bonus runs beat every one of the five default runs on both measures (the smallest p-value possible with five runs each, 0.0079).
It also **degrades with training** — at the default setting, 2 cm success fell in all five seeds between iteration 800 and 5000 — and the measured **action spread at the end of training collapsed to e^−3.5 ≈ 0.03** for the default runs but stayed large for the 0.02 runs. That is the exploration-collapse mechanism, measured directly. The advantage even carries over to the hard unseen physics. **What it explains:** why Stage 1's long human-reward run looked so bad, and why the earlier 800-iteration test saw no entropy effect (the collapse has not happened yet that early). **What it does not show:** anything about the Eureka reward, RAPP, adaptive DR or the PINN — only the human reward on the toy arm. (Doc 21.)

### Step 18 — The other three pre-registered experiments (14 Oct)
- **E2 (LEAP robustness curve):** no separation between the three LEAP conditions; a common curve with a cliff at difficulty 20.
- **E4 step-0 (does LEAP's quality collapse before its rotation count?):** no — falls and rotations change together, so the planned precision-aware ADR method was **not built** (as the rule decided in advance).
- **E5 (does Eureka *learn faster*?):** no — the area under the learning curve is the same (40.8 vs 41.4 mm, p = 0.84). So the reward comparison is a null on speed as well as on final quality.
- **A new hint, not a finding:** the warm-started LEAP policy (L2) drops the cube 5–10 times less often than the others at every difficulty from 10 up, in all three seeds. This was noticed after the fact, not pre-registered, and three seeds cannot reach significance, so it needs a confirmatory run.
- **A newly found silent bug:** a few evaluations had "succeeded" without the simulator actually running (arm never moved; one reported a 73-metre error). A full audit found four such files, all repaired; every evaluation script now checks that the arm really moved. (Doc 04, D7c.)

---

## Part 7 — What we found, in plain words

**Findings we are confident about**
1. **The testbed is trustworthy.** It matches analytic physics, and a perfect scripted controller reaches 1 mm — so differences between learned policies are about the learning, not the robot.
2. **The Eureka-style reward has a precision problem.** ~99 % at 5 cm but ~27 % at 2 cm, in every seed. (Reason: the large success bonus.)
3. **At the same training budget, the human reward and the Eureka reward end up equally good** (mean error 37.75 vs 37.69 mm). Stage 1's claim that Eureka is better is not supported; its claim that the human reward is far more precise could not be reproduced.
4. **Removing the RAPP ceiling from adaptive DR costs success at nominal physics** (5 of 5 seeds).
5. **A physics observer can be built and is accurate in range, but mass and motor strength are hard to separate from motion alone**; breaking the tie needs specially designed excitation (steady-state, one joint at a time).
6. **LEAP already includes adaptive DR**, so "stock = no-DR baseline" is wrong.

7. **The human reward's long-run failure is exploration collapse (E1).** Default entropy → performance *degrades* with longer training (mean error 38 → 62 mm; tail 149 → 346 mm); entropy 0.02 → stays good (30 mm; tail 66 mm); 5 of 5 seeds; measured mechanism.
8. **The Eureka reward does not learn faster than the human reward** (E5, p = 0.84), as well as not ending better.
9. **LEAP's quantity-based ADR gate is aligned with the quality collapse at this budget** (E4 step-0: 0 of 3 conditions show quality degrading first).

**Findings that are suggestive but unproven**
- Static DR may *lower* fine precision for the Eureka reward when tested under randomisation (p = 0.016, 5 seeds; close to the smallest p possible).
- The PINN-informed controller might help on hard unseen physics (8 of 10 seeds in its favour, p = 0.109).
- Warm-started LEAP ADR (L2) seems to make the hand drop the cube far less under hard physics (all seeds, but only 3 seeds and noticed after the fact).

**Things we claimed and then withdrew:** Eureka beats the human reward; F4 beats static DR; rung E doubles precision; "entropy fixes vanilla's precision" at short budgets; the G-vs-placebo p = 0.024 and p = 0.281; "LEAP_500 is a no-DR baseline"; two of my own early diagnoses (a flawed PINN fix and a wrong memory-error theory).

**The honest bottom line:** none of the roadmap's *proposed improvements* (RAPP-bounded adaptive DR, PINN-informed DR) has been shown to beat simpler baselines on our tests. The project's solid value is the rigorous testbed and protocol, the diagnosed pitfalls, and the clarity about what does *not* work at these budgets.

---

## Part 8 — How we tested that findings were real

| Safeguard | Plain-language idea | Example from the project |
|---|---|---|
| **Multiple seeds** | Run it again with different luck | F4's single-seed win vanished once D's own seeds spanned 16–34 % |
| **Matched budget** | Compare at equal effort | Eureka vs human compared at 800 iterations, not 1500 vs 5000 |
| **Independent oracle** | Check against something you know is right | The scripted controller's 100 %, the analytic geometry formula |
| **Read back, don't assume** | Ask the simulator what it actually did | Randomised friction values read from the simulator |
| **Placebo control** | Scramble the "information" and see if the benefit remains | G versus shuffled G |
| **Pre-declared primary metric** | Decide what counts as success before looking | DRshift worst-decile error fixed before the retest data |
| **Retest of surprises** | A result that surprises you must repeat | G's p = 0.024 did not repeat |
| **Look at raw per-seed values** | Means hide outliers | E's "double" was the 2nd best of 3 seeds |
| **Red-flag audits** | Treat "too identical" as a bug | Identical numbers revealed the shared-folder contamination |
| **Test your own hypotheses to destruction** | Try to refute yourself | The dual-torque PINN fix was tested and failed |
| **Pre-registration (E1–E5)** | Write the success rule down first | Docs 16–19 timestamped before any run |
| **Log corrections, never erase** | Keep history honest | The experiment log is append-only; retractions are shown, not hidden |

---

## Part 9 — What went wrong (and what it taught us)

**Environment/measurement bugs (found by checking against known truths)**
- Evaluation measured *after* auto-reset → 0.39 % instead of ~93 %. *Lesson: if a metric disagrees badly with training progress, suspect the metric.*
- Damping silently 0 → every randomised-vs-nominal comparison would have carried a hidden offset. *Lesson: read configuration back from the simulator.*
- ±π joint limits trapped the arm, capping any controller at ~69 %. *Lesson: test with an optimal reference controller.*
- PINN training data violated physics. *Lesson: check the data against an oracle before tuning the model.*

**Process/ops failures**
- **Shared-folder collision** and **silent truncation** (Part 6 step 13). *Lesson: unique output folders per run; verify each run reached its target length; treat identical results across arms as a bug.*
- **Misreading failures:** treating GPU-memory shortage as job failure; blaming memory when the real cause was an early-stop setting. *Lesson: keep a separate log per attempt; resource shortage should mean "wait", not "fail".*
- **Unexplained process kills (07:43 and ≈13:16 on 14 Oct)** and a crontab that was found with every restart entry commented out. *Lesson: unattended jobs on a shared machine need supervision — and a person who changes the machine should tell the owner of the jobs.*
- **Reasoning errors on my part,** each logged: the dual-torque PINN fix rested on wrong physics; the LEAP "RAM kill" theory was wrong; I repeatedly matched my own shell with `pkill -f`.

**Overclaiming** — the biggest recurring risk. Three exciting single-seed results had to be retracted. The rule that fixed it: *no claim until replicated; decide the success rule first.*

---

## Part 10 — Where things stand now, and what's next

**Finished**
- The 2-DOF testbed, validation, baselines, RAPP map, PINN observer analysis, the full A–G five-seed matrix, the G-versus-placebo test (clean, ten seeds per arm), the LEAP three-condition matrix (three seeds each), the LEAP robustness curve (E2). All documented in Docs 00–15.

**State of the experiments (14 Oct, ≈19:00 UTC):** all queued experiments are **finished** and nothing is running. (The jobs had been stopped once, at ≈13:16, by an unknown actor who also disabled the restart schedule; on the owner's instruction they were restarted and then ran to completion.) Results: E1 positive; E2, E4 step-0 and E5 are clean nulls (Doc 21).

**What the paper can now say**
- **Lead result (explanatory success):** *"The failure of the human-designed reward in long training is a measurable exploration-collapse artifact that an entropy bonus prevents — with a dose-response, a mechanism, and carry-over to hard physics."* This rebuts the reading of Stage 1 in which the human reward is inherently imprecise and Eureka is better.
- **Supporting:** the validated testbed and evaluation protocol; the audit trail (retractions, contamination, invalid-evaluation repair); the RAPP sensitivity map; the PINN identifiability analysis; the pre-registered nulls (reward-comparison on quality and speed, adaptive DR vs static, PINN-informed vs placebo, LEAP).
- **Rough, unverified odds:** workshop ≈ 65–75 %, simulation-methodology journal ≈ 45–55 %, robotics main track still low (one toy task, no hardware, no positive method).
- **Best single next experiment:** a confirmatory, pre-registered test of the warm-start-reduces-drops hint on LEAP (≈ 10 GPU-hours), and E1's claim on a second task.

**Decisions waiting on the owner:** (1) the target venue; (2) whether to run the confirmatory LEAP warm-start experiment and/or repeat E1 on a second task; (3) who stopped the jobs at 13:16 and disabled the schedule (unresolved); (4) rotating a Gemini API key that was pasted into chat earlier (it is stored only in a protected local file and was never committed).

---

## Part 11 — Glossary

- **Ablation:** removing/swapping one ingredient to see its effect.
- **ADR (automatic/adaptive domain randomization):** widen the randomisation as the policy improves.
- **Actuator effectiveness:** how strong a motor really is relative to nominal.
- **Coarse vs fine success:** success at a loose threshold (5 cm) vs a strict one (1 cm).
- **DR / domain randomization:** training across varied physics.
- **DRtrain / DRshift / nominal:** the three evaluation physics conditions.
- **Entropy / entropy bonus:** a measure of how spread-out the policy's actions are / a reward for staying spread out.
- **Ensemble:** several copies of a model; their disagreement signals uncertainty.
- **Eureka:** an LLM-driven method that writes reward functions.
- **Exploration collapse:** the action spread shrinking so early the agent stops trying new things.
- **Frozen protocol:** an evaluation procedure that is never changed, so results stay comparable.
- **Identifiability:** whether parameters can be told apart from the available data.
- **IK + PD controller:** a classic non-learning controller used as a perfect reference.
- **Isaac Sim / Isaac Lab:** the physics simulator and robot-learning framework used.
- **LEAP Hand:** the 16-joint robot hand testbed.
- **Nominal:** the standard physics with no randomisation.
- **p-value:** how often luck alone would produce a gap this large.
- **Placebo control:** the same procedure with the information scrambled.
- **PPO:** the RL algorithm used.
- **PINN / physics observer:** a network that estimates physical parameters from motion and obeys physics equations.
- **Pre-registration:** writing the success rule down before running.
- **RAPP:** a one-parameter-at-a-time sweep finding the physics region where a policy still works.
- **Rung (A–G):** one configuration in the ablation matrix.
- **Seed:** the random-number starting point of a run; different seeds give different results.
- **Sim-to-real gap:** the mismatch between a simulator and reality.
- **Static DR:** randomisation ranges fixed during training.
- **Worst-decile error:** the average error of the worst 10 % of episodes.
- **Vanilla / human reward:** the hand-designed reward.

---

## Part 12 — Where everything lives

Everything is under `/data/home/amartyaranjan888/projects/I/isaac_env/pi_adr/`:

| What | Where |
|---|---|
| **This guide and all documentation** | `project_docs/` (start with this file, then `00_INDEX.md`) |
| Chronological log of every run and correction | `EXPERIMENT_LOG.md` |
| Environment, evaluation, controller and observer code | `source/pi_adr/` |
| Scripts for training, evaluation, analysis, queues | `scripts/` |
| One result file per evaluation, and matrix tables | `results/` (and `results/MATRIX_A_to_G.md`, `MATRIX_LEAP.md`, `E2_analysis.md`) |
| Logs of every run | `logs/` |
| Observer datasets and trained observers | `data/` |
| Quarantined (contaminated/truncated) results kept for audit | `results/_quarantine_*` |
| Pre-registrations (success rules written before the runs) | `project_docs/16…19_*` |

**Suggested reading path for a newcomer:** this file → `00_INDEX.md` (headline findings and glossary) → `12_validity_and_claims_ledger.md` (every claim and its fate) → `10_ablation_matrix_A_to_G.md` (the main results) → `15_plan_for_a_positive_result.md` and `16…19` (what is being tested now).
