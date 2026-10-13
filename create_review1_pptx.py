# create_review1_pptx.py — Review 1 deck, grounded in what is actually built & measured
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

NAVY   = RGBColor(0x0B, 0x25, 0x45)
TEAL   = RGBColor(0x13, 0x7C, 0x8A)
ACCENT = RGBColor(0xF2, 0x8C, 0x28)
GREEN  = RGBColor(0x1E, 0x7A, 0x4B)
LIGHT  = RGBColor(0xEE, 0xF2, 0xF5)
DARK   = RGBColor(0x1A, 0x1A, 0x1A)
GREY   = RGBColor(0x5A, 0x62, 0x6B)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
CODEBG = RGBColor(0x10, 0x16, 0x1D)
CODETX = RGBColor(0xD8, 0xE1, 0xE8)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def new_slide():
    N[0] += 1
    return prs.slides.add_slide(BLANK)


def footer(slide):
    tb = slide.shapes.add_textbox(Inches(12.6), Inches(7.1), Inches(0.7), Inches(0.35))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = str(N[0]); r.font.size = Pt(10); r.font.color.rgb = GREY


def header(slide, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background(); bar.shadow.inherit = False
    tf = bar.text_frame; tf.margin_left = Inches(0.4); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(23); r.font.bold = True; r.font.color.rgb = WHITE
    acc = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.0), prs.slide_width, Inches(0.06))
    acc.fill.solid(); acc.fill.fore_color.rgb = ACCENT
    acc.line.fill.background(); acc.shadow.inherit = False
    footer(slide)


def bullets(slide, items, left, top, width, height, size=13):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        text, level, bold = it if isinstance(it, tuple) else (it, 0, False)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = ("    " * level) + ("• " if level == 0 else "- ") + text
        r.font.size = Pt(size - level * 2); r.font.color.rgb = DARK; r.font.bold = bold
    return tb


def codebox(slide, text, left, top, width, height, size=10):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sh.fill.solid(); sh.fill.fore_color.rgb = CODEBG
    sh.line.color.rgb = TEAL; sh.line.width = Pt(1); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1); tf.margin_bottom = Inches(0.1)
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        r = p.add_run(); r.text = line
        r.font.name = "Consolas"; r.font.size = Pt(size); r.font.color.rgb = CODETX
    return sh


def table(slide, rows, left, top, width, col_widths, size=11):
    gf = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, Inches(0.36 * len(rows)))
    t = gf.table
    for i, w in enumerate(col_widths):
        t.columns[i].width = w
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = t.cell(ri, ci); c.text = str(val)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = Pt(6); c.margin_right = Pt(4)
            c.margin_top = Pt(2); c.margin_bottom = Pt(2)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    if ri == 0:
                        r.font.bold = True; r.font.color.rgb = WHITE
                    else:
                        r.font.color.rgb = DARK
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if ri == 0 else (LIGHT if ri % 2 == 0 else WHITE)
    return t


def box(slide, text, left, top, width, h=Inches(0.66), fill=NAVY, size=12):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE
    return sh


def arrow(slide, left, top, w, h, kind=MSO_SHAPE.DOWN_ARROW, color=ACCENT):
    a = slide.shapes.add_shape(kind, left, top, w, h)
    a.fill.solid(); a.fill.fore_color.rgb = color
    a.line.fill.background(); a.shadow.inherit = False


# ---------------- 1 TITLE ----------------
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.0), Inches(2.2), Inches(0.08))
acc.fill.solid(); acc.fill.fore_color.rgb = ACCENT; acc.line.fill.background(); acc.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(2.2))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "Physics-Informed Adaptive Reward Generation\nfor Sim-to-Real Dexterous Hand Control"
r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = WHITE
p = tf.add_paragraph(); p.space_before = Pt(14)
r = p.add_run()
r.text = "Eureka  •  Adaptive Domain Randomization  •  PINN Failure Diagnosis  •  PPO on NVIDIA Isaac Lab"
r.font.size = Pt(16); r.font.color.rgb = ACCENT
p = tf.add_paragraph(); p.space_before = Pt(24)
r = p.add_run()
r.text = "Final Year Project  |  Review 1  |  Team: [Name / Roll], [Name], [Name]  |  Guide: [Name]  |  2026"
r.font.size = Pt(13); r.font.color.rgb = LIGHT

# ---------------- 2 ABSTRACT & PROBLEM ----------------
s = new_slide()
header(s, "Abstract & Problem Definition")
bullets(s, [
    ("Reward engineering is the bottleneck in RL for dexterous manipulation.", 0, True),
    ("Eureka (ICLR'24) automates it: an LLM writes reward code from environment source, refined by evolutionary search + per-component reward reflection.", 1, False),
    ("DrEureka (RSS'24) adds LLM-generated domain randomization for zero-shot sim-to-real.", 1, False),
    ("Gap 1 - Reflection reports WHAT, never WHY.", 0, True),
    ("An inverted-sign reward bug and a contact-model mismatch produce near-identical feedback. The LLM has one tool - rewrite the reward - which is wrong for the second case, and the harm stays invisible until deployment.", 1, False),
    ("Gap 2 - Static DR is calibrated against a policy that no longer exists.", 0, True),
    ("DrEureka fixes ranges from an initial policy; range feasibility is a property of the CURRENT policy.", 1, False),
    ("Gap 3 - No channel through which 'the simulator is wrong' can be expressed.", 0, True),
], Inches(0.45), Inches(1.25), Inches(7.5), Inches(5.8), size=12)
codebox(s, """
PLATFORMS (both, cross-morphology)
----------------------------------
Sim   : NVIDIA Isaac Lab 2.3.0
        (Isaac Sim 5.1.0, Python 3.11)
RL    : PPO via rl_games
        - matches Eureka/DrEureka
          for ablation validity

Task A: Isaac-Repose-Cube-Shadow
        Shadow Hand, 24 joints / 20 act.
Task B: Isaac-Reorient-Cube-Leap
        LEAP Hand, 16 DoF, + real robot

LLM   : Gemini 3.5 Flash
""", Inches(8.25), Inches(1.35), Inches(4.6), Inches(4.4), size=11)

# ---------------- 3 OBJECTIVES ----------------
s = new_slide()
header(s, "Objectives & Expected Outcomes")
bullets(s, [
    ("O1 - Reproduce the Eureka/DrEureka baseline unmodified (rung 1).", 0, True),
    ("Inherited deliberately: reward loop, reflection, fitness metric, PPO. Unmodified inheritance is what makes the ablation interpretable.", 1, False),
    ("O2 - Adaptive domain randomization (Contribution 1).", 0, True),
    ("Re-evaluate DR ranges every N iterations from policy performance, closing DrEureka's one-shot loop.", 1, False),
    ("O3 - PINN physics-residual failure diagnosis + routing (Contribution 2).", 0, True),
    ("Separate 'fix the reward' from 'fix the physics assumptions'; enforce the separation in code.", 1, False),
    ("O4 - Ground-truth validation of the diagnosis (Contribution 3).", 0, True),
    ("Injection calibration + placebo arms. No prior work validates an attribution signal this way.", 1, False),
], Inches(0.45), Inches(1.25), Inches(7.4), Inches(4.4), size=12)
table(s, [
    ["Outcome", "How it is judged"],
    ["Rung-1 fidelity", "Matches published Eureka behaviour"],
    ["Adaptive DR gain", "Transfer vs static DR, held-out config"],
    ["Diagnosis validity", "AUC vs injected ground truth"],
    ["Signal is load-bearing", "Real vs shuffled vs no signal"],
    ["Real transfer (LEAP)", "Zero-shot on physical hand"],
], Inches(8.1), Inches(1.35), Inches(4.8), [Inches(2.0), Inches(2.8)], size=11)

# ---------------- 4 SCOPE & FEASIBILITY ----------------
s = new_slide()
header(s, "Scope & Feasibility")
table(s, [
    ["Component", "IN scope", "OUT of scope"],
    ["Simulator", "Isaac Lab 2.3.0 / Isaac Sim 5.1.0", "Isaac Gym (deprecated, archived Apr 2026)"],
    ["Robots", "Shadow Hand (sim), LEAP Hand (sim+real)", "Arms, quadrupeds, locomotion"],
    ["Observations", "Proprioceptive state", "Vision (no RT cores on datacenter GPUs)"],
    ["RL algorithm", "PPO (rl_games)", "SAC/HER - would break baseline parity"],
    ["PINN role", "Failure diagnosis via physics residual", "Contact-solver modification"],
    ["LLM", "Gemini 3.5 Flash (API)", "Fine-tuning"],
], Inches(0.45), Inches(1.25), Inches(7.4), [Inches(1.5), Inches(3.0), Inches(2.9)], size=10)
codebox(s, """
COMPUTE - MEASURED, NOT ESTIMATED
---------------------------------
GPU        : 1x NVIDIA L4, 23 GB
Driver     : 595.71.05
Throughput : 1.70 s / epoch @ 8192 envs
             (~77k FPS steady state)
Startup    : ~53 s per process launch
Concurrency: 2 workers MAX
             (3 exhausts system RAM)
Budget     : ~376 GPU-hours for the
             full 2-platform ladder
""", Inches(8.1), Inches(1.35), Inches(4.8), Inches(3.3), size=11)
bullets(s, [
    "All upstream code open-source; Phase 0 already installed, patched and validated end-to-end (see Progress slide).",
], Inches(8.1), Inches(4.85), Inches(4.8), Inches(1.2), size=11)

# ---------------- 5 ARCHITECTURE ----------------
s = new_slide()
header(s, "System Architecture - Two Loops, One Router")
box(s, "Task spec + safety instruction + environment source code", Inches(3.6), Inches(1.25), Inches(6.2), fill=GREY, size=12)
arrow(s, Inches(6.5), Inches(1.95), Inches(0.35), Inches(0.24))
box(s, "LLM (Gemini): generate K candidate reward functions", Inches(3.6), Inches(2.24), Inches(6.2), fill=ACCENT)
arrow(s, Inches(6.5), Inches(2.94), Inches(0.35), Inches(0.24))
box(s, "Parallel PPO training  -  Isaac Lab (Shadow Hand / LEAP Hand)", Inches(3.6), Inches(3.23), Inches(6.2), fill=NAVY)
arrow(s, Inches(6.5), Inches(3.93), Inches(0.35), Inches(0.24))
box(s, "PINN physics residual\n(contact-phase gated)", Inches(2.3), Inches(4.22), Inches(3.1), h=Inches(0.72), fill=TEAL, size=11)
box(s, "Reward reflection\n+ task fitness F", Inches(8.0), Inches(4.22), Inches(3.1), h=Inches(0.72), fill=TEAL, size=11)
arrow(s, Inches(6.5), Inches(5.06), Inches(0.35), Inches(0.24))
box(s, "DIAGNOSIS ROUTER  -  classify: reward-logic error  vs  physics mismatch", Inches(3.0), Inches(5.35), Inches(7.4), fill=ACCENT, size=12)
box(s, "reward-logic  ->  edit reward code", Inches(1.3), Inches(6.35), Inches(4.6), h=Inches(0.6), fill=NAVY, size=11)
box(s, "physics  ->  update DR ranges", Inches(7.4), Inches(6.35), Inches(4.6), h=Inches(0.6), fill=TEAL, size=11)
arrow(s, Inches(3.4), Inches(6.05), Inches(0.32), Inches(0.24), MSO_SHAPE.LEFT_ARROW)
arrow(s, Inches(9.6), Inches(6.05), Inches(0.32), Inches(0.24), MSO_SHAPE.RIGHT_ARROW)
tb = s.shapes.add_textbox(Inches(0.5), Inches(7.02), Inches(12.0), Inches(0.4))
r = tb.text_frame.paragraphs[0].add_run()
r.text = "Enforced in code, not requested in a prompt: each diagnosis can trigger only its own pathway."
r.font.size = Pt(11); r.font.italic = True; r.font.color.rgb = GREY

# ---------------- 6 ALGORITHM 1 ----------------
s = new_slide()
header(s, "Algorithm 1 - Eureka Reward Synthesis  (inherited unmodified)")
codebox(s, """
Input: task description T, env source E, LLM M, iterations G, population K
1:  R_0 <- M.generate(T, E, K)              # K reward programs, zero-shot
2:  for g in 0..G-1:
3:      for r in R_g:
4:          pi_r  <- PPO_train(E, r)         # Isaac Lab, parallel envs
5:          s(r)  <- F(pi_r)                 # task fitness - NEVER the LLM reward
6:          ref(r)<- per_component_stats(r)  # reward reflection
7:      R_{g+1} <- M.mutate(T, best(R_g), ref)
8:  return argmax_r s(r)

KEY DESIGN POINT (Eureka's actual insight):
  reward MUST return a dict of NAMED components, not a scalar
  -> turns an opaque score into a diagnosable signal
""", Inches(0.45), Inches(1.25), Inches(7.6), Inches(4.3), size=11)
bullets(s, [
    ("Inherited without modification:", 0, True),
    "Evolutionary search structure",
    "Reward reflection channel",
    "Task fitness F, unchanged from benchmark",
    "PPO + hyperparameters",
    ("Why not modify it?", 0, True),
    "Every change to the baseline weakens the ablation. Rung 1 must be a faithful reproduction or nothing above it is interpretable.",
], Inches(8.3), Inches(1.25), Inches(4.6), Inches(5.0), size=12)

# ---------------- 7 ALGORITHM 2 : ADAPTIVE DR ----------------
s = new_slide()
header(s, "Algorithm 2 - Adaptive Domain Randomization  (Contribution 1)")
codebox(s, """
DrEureka (baseline)          |  OURS (adaptive)
-----------------------------|---------------------------------
RAPP sweep on pi_initial     |  RAPP sweep -> initial ranges phi_0
LLM picks ranges within it   |  LLM picks ranges within it
ranges FIXED for training    |  every N iterations:
                             |    perf <- fitness + reflection
                             |    if robust : widen  phi
                             |    if failing: narrow phi
                             |    (LLM-selected, within feasibility)

Why it matters: range feasibility is a property of the CURRENT
policy. A late robust policy tolerates far wider randomization
than the initial one RAPP was calibrated against.
""", Inches(0.45), Inches(1.25), Inches(7.8), Inches(4.2), size=10)
bullets(s, [
    ("Implementation basis:", 0, True),
    "LEAP_Hand_Isaac_Lab already ships a working ADR module (LeapHandADR) that widens EventTerm ranges on a fixed schedule.",
    "That is the mechanism. Our contribution is LLM-driven range SELECTION at each decision point - closing the loop ADR leaves open and DrEureka leaves one-shot.",
    ("Honest caveat:", 0, True),
    "DrEureka tested intra-training adaptation and found it unnecessary for locomotion. We hypothesise it matters where feasible ranges are narrow and contact dominates - and must demonstrate it, not assume it.",
], Inches(8.5), Inches(1.25), Inches(4.4), Inches(5.2), size=11)

# ---------------- 8 PINN ----------------
s = new_slide()
header(s, "PINN - Physics-Residual Failure Diagnosis  (Contribution 2)")
codebox(s, """
Equations of motion (Lagrangian form):
   R_phys = M(q)*q_ddot + C(q,q_dot)*q_dot + G(q) + F_contact - tau

For ideal rigid-body dynamics, R_phys = 0. The residual therefore
measures how far observed SIMULATED behaviour departs from
consistent physics - i.e. how much is simulator artifact.

Loss (SOPDNA composite shape):
   L_total = lam_d * L_data + lam_p * L_phys + lam_c * L_ctrl

   lam_* have NO literature default - both source papers leave
   them untuned. Ours are explicit tunable config, not constants.
""", Inches(0.45), Inches(1.25), Inches(7.8), Inches(3.5), size=10)
bullets(s, [
    ("Why learned, not an analytic check:", 0, True),
    "Evaluable at arbitrary / counterfactual states",
    "Continuous differentiable signal -> can be gated",
    "Latent parameter discovery: identifies WHICH physical parameter is mismatched -> exactly what the DR controller needs",
    ("THE HARD PART - contact:", 0, True),
    "Prior work (PI-DDPG, SOPDNA) handles only smooth dynamics. In-hand manipulation is discontinuous multi-point contact.",
    "Mitigation: contact-phase gating; contact-free and contact-phase residuals reported SEPARATELY, never aggregated.",
], Inches(8.5), Inches(1.25), Inches(4.4), Inches(5.4), size=11)

# ---------------- 9 ROUTING RULE ----------------
s = new_slide()
header(s, "The Routing Rule - Why Separation Is Necessary")
box(s, "FAILURE MODE:  reward shunting", Inches(0.45), Inches(1.3), Inches(5.9), h=Inches(0.5), fill=RGBColor(0x8B,0x1A,0x1A), size=13)
bullets(s, [
    "Give the LLM a physics signal with no constraint, and it does the natural thing: fixes the symptom in reward space - a penalty steering the policy away from the badly-modelled region.",
    ("Result:", 0, True),
    "Fitness improves. Ablation reports 'PINN helped'.",
    "But the physics error was never addressed - it was ABSORBED into the reward.",
    "That reward encodes avoidance of a simulator artifact the real robot does not have.",
    ("So the naive integration looks like success, hides the cause, and DAMAGES the sim-to-real transfer the project exists to improve.", 0, True),
], Inches(0.45), Inches(1.95), Inches(5.9), Inches(4.6), size=11)
box(s, "THE RULE:  diagnose, then route", Inches(6.9), Inches(1.3), Inches(5.9), h=Inches(0.5), fill=GREEN, size=13)
table(s, [
    ["Diagnosis", "Permitted", "Forbidden"],
    ["reward-logic error", "Edit reward code", "Touching DR ranges"],
    ["physics mismatch", "Update DR ranges", "Editing reward code"],
], Inches(6.9), Inches(2.0), Inches(5.9), [Inches(1.9), Inches(2.0), Inches(2.0)], size=11)
bullets(s, [
    ("Three things at once:", 0, True),
    "Makes the PINN's contribution MEASURABLE - otherwise 'physics helped' cannot be separated from 'extra text helped'.",
    "Prevents the transfer-harming failure above.",
    "Forces the signal to be load-bearing - a physics diagnosis is the ONLY thing that can move DR ranges.",
    "A dual diagnosis is rejected by design, keeping the routing audit interpretable as a confusion matrix.",
], Inches(6.9), Inches(3.5), Inches(5.9), Inches(3.2), size=11)

# ---------------- 10 VALIDATION ----------------
s = new_slide()
header(s, "Validating the Diagnosis  (Contribution 3)")
bullets(s, [
    ("The claim 'high residual => physics failure' is an INFERENCE, not a fact.", 0, True),
    "A bad reward can drive the policy into contact-rich, poorly-modelled states, producing a high residual from a pure reward bug. Without a test, the router routes on a confounded signal.",
    ("1. Injection calibration - a hard gate before any rung-3 claim", 0, True),
    "Seed KNOWN reward bugs (sign flip, wrong scale, missing term) vs KNOWN dynamics shifts (mass x1.5, friction x0.3, added damping); measure discrimination AUC.",
    "Taxonomy drawn from failures ACTUALLY OBSERVED in rung 1 - not invented, or it measures detection of bugs no real LLM produces.",
    ("2. Placebo arms", 0, True),
    "Real signal vs SHUFFLED signal vs no signal, identical budgets. If real ~= shuffled, the contribution is unsupported however good the AUC.",
    ("3. Routing audit", 0, True),
    "Diagnosis -> edit-type logged every iteration; confusion matrix for the whole loop against injected ground truth.",
], Inches(0.45), Inches(1.25), Inches(7.6), Inches(5.6), size=11)
table(s, [
    ["AUC outcome", "Decision"],
    ["contact-free strong,\ncontact-phase weak", "EXPECTED. Proceed with\ncontact-gated routing."],
    ["contact-free ALSO weak", "STOP. Do not build the\nfull router. Fall back,\nrecalibrate, or report a\nnegative result."],
], Inches(8.3), Inches(1.35), Inches(4.6), [Inches(2.0), Inches(2.6)], size=10)
bullets(s, [
    "Encoded as executable logic, not a judgement call made under deadline pressure.",
], Inches(8.3), Inches(4.5), Inches(4.6), Inches(1.0), size=10)

# ---------------- 11 DATA ----------------
s = new_slide()
header(s, "Data & Benchmarks")
bullets(s, [
    ("There is NO fixed offline dataset. All training data is generated ONLINE by GPU-parallel simulation.", 0, True),
    "This is inherent to the method: Eureka-style reward search requires fresh rollouts under each candidate reward, so a pre-collected corpus cannot serve it.",
], Inches(0.45), Inches(1.25), Inches(12.4), Inches(1.1), size=12)
table(s, [
    ["Corpus", "Source", "Role"],
    ["Benchmark environments", "Isaac Lab: Isaac-Repose-Cube-Shadow-Direct-v0; LEAP_Hand_Isaac_Lab: Isaac-Reorient-Cube-Leap", "The two evaluation platforms"],
    ["Rollout telemetry", "Generated per step: joint pos/vel/torque, tracking error, contact events, active DR values", "Reward reflection + summarizer"],
    ["PINN corpus", "Privileged sim transitions: hand state + object 6-DoF pose/twist + DR params", "Self-supervised - physics residual IS the label"],
    ["Injection corpus", "Runs with deliberately seeded reward bugs vs seeded dynamics shifts", "GROUND TRUTH for the AUC gate"],
    ["Real-world eval", "Physical LEAP Hand rollouts, zero-shot", "Sim-to-real evidence"],
], Inches(0.45), Inches(2.5), Inches(12.4), [Inches(2.2), Inches(6.3), Inches(3.9)], size=10)
bullets(s, [
    "Lineage note: the Shadow Hand task descends from Eureka's Bi-DexHands suite; the LEAP task from DrEureka. Object sets such as YCB are relevant only if the work is later extended beyond single-object cube reorientation.",
], Inches(0.45), Inches(5.6), Inches(12.4), Inches(1.0), size=11)

# ---------------- 12 PROGRESS ----------------
s = new_slide()
header(s, "Progress to Date - Phase 0 Complete & Validated")
table(s, [
    ["Milestone", "Status", "Evidence"],
    ["Isaac Lab 2.3.0 + Isaac Sim 5.1.0 installed", "DONE", "Python 3.11, torch 2.7.0+cu128"],
    ["Headless GPU compatibility retired as a risk", "DONE", "PPO run, exit 0, checkpoint saved"],
    ["Shadow Hand task validated", "DONE", "20 epochs @ 8192 envs, 87 s"],
    ["Gemini integration (replacing OpenAI)", "DONE", "Provider layer + model resolver"],
    ["Both hand tasks registered in Eureka", "DONE", "Shadow + LEAP in TASKS_CFG"],
    ["FULL EUREKA LOOP END-TO-END", "DONE", "2 cand. x 2 iters, exit 0, 761 s"],
    ["Concurrency ceiling measured", "DONE", "2 workers max (3 exhausts RAM)"],
], Inches(0.45), Inches(1.25), Inches(7.6), [Inches(3.7), Inches(1.1), Inches(2.8)], size=10)
codebox(s, """
TWO SILENT-FAILURE BUGS FOUND & FIXED
-------------------------------------
1. Gemini rejects n>1 with HTTP 400
   "Multiple candidates is not enabled"
   -> would CRASH on the first LLM call
   -> fix: detect + fan out to N calls,
      result cached per process

2. Kit structured-log race with parallel
   workers: RingBuffer corruption killed
   one worker SILENTLY
   -> 0 errors single-process, 4 with 2
   -> fix: stagger worker startup

Both would have degraded the search
without raising any error.
""", Inches(8.3), Inches(1.35), Inches(4.6), Inches(4.5), size=10)
bullets(s, [
    "Gemini-authored reward code already includes quaternion double-cover handling, temperature-scaled shaping, and multiplicative task reward.",
], Inches(0.45), Inches(4.6), Inches(7.6), Inches(1.2), size=11)

# ---------------- 13 ABLATION LADDER ----------------
s = new_slide()
header(s, "Evaluation Plan - Staged Ablation Ladder")
table(s, [
    ["Rung", "Configuration", "Isolates", "Primary metric"],
    ["1", "Eureka/DrEureka reproduction, static DR", "Baseline control", "Task fitness F"],
    ["2", "+ Adaptive DR", "Adaptive DR alone", "Transfer to held-out config"],
    ["3", "+ PINN diagnosis & routing", "Physics diagnosis, marginal", "AUC + placebo delta"],
    ["4 (optional)", "+ Learned failure summarizer", "Learned vs hand-crafted reflection", "2x2 placebo"],
], Inches(0.45), Inches(1.25), Inches(12.4), [Inches(1.3), Inches(4.2), Inches(3.4), Inches(3.5)], size=11)
bullets(s, [
    ("One component changes per rung. Everything not under test - prompting, population size, PPO hyperparameters, fitness - is held identical.", 0, True),
    ("Transfer evidence differs by platform and must not be conflated:", 0, True),
    "LEAP Hand - zero-shot on the PHYSICAL robot (DrEureka protocol).",
    "Shadow Hand - held-out perturbed sim config, never seen in training or DR. This is sim-to-OOD-sim, and will be labelled as such.",
    ("No results are claimed yet. Review 1 reports the validated pipeline and the measured cost model; rung numbers follow.", 0, True),
], Inches(0.45), Inches(3.5), Inches(12.4), Inches(3.4), size=12)

# ---------------- 14 RISKS ----------------
s = new_slide()
header(s, "Risk Analysis & Mitigation")
table(s, [
    ["Risk", "Prob.", "Impact", "Mitigation / current status"],
    ["Contact discontinuity defeats the physics residual", "High", "High", "Contact-phase gating; injection AUC gate runs BEFORE the router is built, so it fails cheaply"],
    ["Candidate pool capped at 2 vs Eureka's 16", "Certain", "High", "MEASURED limit. Options: fewer candidates + more iterations, fewer envs/worker, sequential, or larger GPU. Stated as a deviation, not hidden"],
    ["lam weights have no literature default", "Certain", "Med", "Explicit tunable config; both source papers leave them untuned"],
    ["Compounding adaptivity overfits a moving target", "Med", "Med", "Freeze DR + PINN state before final transfer eval"],
    ["Single task per platform (Isaac Lab has ~4-5 hand tasks)", "Certain", "Med", "Stated as a limitation; breadth is future work"],
], Inches(0.45), Inches(1.25), Inches(12.4), [Inches(3.4), Inches(1.0), Inches(1.0), Inches(7.0)], size=10)

# ---------------- 15 CO MAPPING ----------------
s = new_slide()
header(s, "Evaluation Rubric - CO Mapping")
table(s, [
    ["No.", "Parameter", "Marks", "CO1", "CO2", "CO3", "CO4"],
    ["1", "Project title & technical relevance", "1", "-", "Y", "-", "Y"],
    ["2", "Abstract & problem definition", "1", "-", "Y", "-", "Y"],
    ["3", "Objectives & expected outcome", "1", "-", "Y", "-", "Y"],
    ["4", "Scope & feasibility", "1", "Y", "-", "-", "Y"],
    ["5", "Preliminary work plan", "1", "Y", "-", "Y", "Y"],
], Inches(0.45), Inches(1.4), Inches(12.4),
   [Inches(0.8), Inches(5.6), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)], size=12)

# ---------------- 16 REFERENCES ----------------
s = new_slide()
header(s, "Key References")
bullets(s, [
    "Y. J. Ma et al., \"Eureka: Human-Level Reward Design via Coding Large Language Models,\" ICLR 2024.",
    "Y. J. Ma et al., \"DrEureka: Language Model Guided Sim-To-Real Transfer,\" RSS 2024.",
    "J. Schulman et al., \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
    "M. Raissi, P. Perdikaris, G. E. Karniadakis, \"Physics-Informed Neural Networks,\" J. Comput. Phys., vol. 378, 2019.",
    "I. Akkaya et al., \"Solving Rubik's Cube with a Robot Hand,\" arXiv:1910.07113, 2019.  (Automatic Domain Randomization)",
    "W. Li et al., \"PIN-WM: Physics-Informed World Models for Non-Prehensile Manipulation,\" 2025.",
    "R. Fareh et al., \"Physics-informed reward shaped RL control of a robot manipulator,\" Ain Shams Eng. J., 2025.",
    "K. Shaw, A. Agarwal, D. Pathak, \"LEAP Hand: Low-Cost, Efficient, Anthropomorphic Hand for Robot Learning,\" RSS 2023.",
    "NVIDIA, Isaac Lab 2.3.0 / Isaac Sim 5.1.0 documentation, 2026.",
    "Y. Chen et al., \"Bi-DexHands: Bimanual Dexterous Manipulation with RL,\" IEEE TPAMI, 2023.",
], Inches(0.45), Inches(1.3), Inches(12.4), Inches(5.6), size=12)

# ---------------- 17 CLOSING ----------------
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(1), Inches(2.1), Inches(11.3), Inches(1.5))
r = tb.text_frame.paragraphs[0].add_run(); r.text = "Thank You  -  Questions?"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
codebox(s, """
github.com/eureka-research/Eureka
github.com/eureka-research/DrEureka
github.com/NVlabs/IsaacLabEureka
github.com/isaac-sim/IsaacLab
github.com/leap-hand/LEAP_Hand_Isaac_Lab
""", Inches(4.1), Inches(3.9), Inches(5.1), Inches(1.9), size=11)

prs.save("Review1_PhysicsInformed_AdaptiveReward.pptx")
print("Saved Review1_PhysicsInformed_AdaptiveReward.pptx with", len(prs.slides._sldIdLst), "slides")
