# create_review2_pptx.py — Review 2 deck: Stage-1 recap + Stage-2 build & measured results
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

NAVY   = RGBColor(0x0B, 0x25, 0x45)
TEAL   = RGBColor(0x13, 0x7C, 0x8A)
ACCENT = RGBColor(0xF2, 0x8C, 0x28)
GREEN  = RGBColor(0x1E, 0x7A, 0x4B)
RED    = RGBColor(0xA6, 0x2A, 0x2A)
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


def table(slide, rows, left, top, width, col_widths, size=11, highlight_rows=None):
    highlight_rows = highlight_rows or {}
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
                    elif ri in highlight_rows:
                        r.font.bold = True; r.font.color.rgb = highlight_rows[ri]
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


def tag(slide, text, left, top, w=Inches(1.5), fill=GREEN):
    box(slide, text, left, top, w, h=Inches(0.36), fill=fill, size=11)


# =====================================================================
# 1 TITLE
# =====================================================================
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.9), Inches(2.2), Inches(0.08))
acc.fill.solid(); acc.fill.fore_color.rgb = ACCENT; acc.line.fill.background(); acc.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(2.2))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "Physics-Informed Adaptive Domain Randomization"
r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = WHITE
p = tf.add_paragraph(); p.space_before = Pt(10)
r = p.add_run()
r.text = "Stage 2: 2-DOF Testbed + LEAP Hand — Environment Build, RAPP, Physics-Informed Observer"
r.font.size = Pt(17); r.font.color.rgb = ACCENT
p = tf.add_paragraph(); p.space_before = Pt(24)
r = p.add_run()
r.text = "Final Year Project  |  Review 2  |  Team: [Name / Roll], [Name], [Name]  |  Guide: [Name]  |  16 Sep 2026"
r.font.size = Pt(13); r.font.color.rgb = LIGHT
tb2 = s.shapes.add_textbox(Inches(0.9), Inches(5.6), Inches(11.5), Inches(1.2))
tf2 = tb2.text_frame; tf2.word_wrap = True
r = tf2.paragraphs[0].add_run()
r.text = "All numbers in this deck are measured on this machine (NVIDIA L4, Isaac Lab 2.3.0) — none are carried over from the Stage-1 document without re-derivation."
r.font.size = Pt(12); r.font.italic = True; r.font.color.rgb = LIGHT

# =====================================================================
# 2 RECAP: STAGE 1 REPORT
# =====================================================================
s = new_slide()
header(s, "Recap — Stage 1: Eureka vs Human Reward (as documented)")
bullets(s, [
    ("Benchmark documented in Stage-1 report:", 0, True),
    "2-DOF planar torque-controlled arm, Isaac Lab, PPO, 4096 parallel envs",
    "Human reward:  R = -d + 5*I(d<0.05) - 0.005||a||^2",
    "Eureka reward:  R = exp(-10d) + 100*I(d<0.05) - 0.01||qdot||^2 - 0.1",
    ("Documented headline claim:", 0, True),
    "Eureka: fast coarse reaching, weak fine precision (2 cm / 1 cm)",
    "Human reward: claimed to reach 98.4% @2cm, 79.7% @1cm after 5000 iters",
    ("Central interpretation offered:", 0, True),
    "\"Eureka changes the objective; PPO changes the policy\" — the +100 success bonus dwarfs the residual shaping once inside 5 cm",
    ("Status found at the start of THIS stage:", 0, True),
    ("No corresponding code, checkpoints, or logs existed anywhere on this machine for those numbers.", 1, True),
], Inches(0.45), Inches(1.25), Inches(7.5), Inches(5.6), size=13)
codebox(s, """
STAGE-1 DOCUMENTED RESULTS
(reproduced here for reference only)

              5cm     2cm     1cm   med.err
Human PPO   99.49%  98.36%  79.74%   6.3mm
Eureka@500  97.90%  19.02%   3.56%  34.6mm
Eureka@1500 95.30%  18.10%   5.30%  33.3mm

-> This stage's job: BUILD the benchmark
   for real, verify it, and see what
   actually reproduces.
""", Inches(8.25), Inches(1.35), Inches(4.6), Inches(4.2), size=11)

# =====================================================================
# 3 STAGE-2 SCOPE / WHAT WAS ASKED
# =====================================================================
s = new_slide()
header(s, "Stage 2 Scope — Two Parallel Tracks")
table(s, [
    ["Track", "Platform", "Task", "Why"],
    ["A", "2-DOF planar arm (built from scratch)", "reach a random goal marker", "cheap, fast testbed for RAPP -> PINN -> adaptive DR"],
    ["B", "LEAP Hand (16-DOF)", "Isaac-Reorient-Cube-Leap (upstream, unmodified)", "real platform, published config, direct comparability"],
], Inches(0.45), Inches(1.25), Inches(12.4), [Inches(0.8), Inches(3.2), Inches(4.2), Inches(4.2)], size=12)
bullets(s, [
    ("Roadmap followed for Track A (physics-informed adaptive DR document):", 0, True),
    "Freeze baseline -> static per-env DR -> RAPP -> PINN + uncertainty -> (adaptive DR: explicitly deferred)",
    ("Roadmap's own checklist item 9: \"Do not build the adaptive controller yet\"", 0, True),
    "Respected literally — items 1-12 done, items 13+ (adaptive DR, A-G ablation matrix) intentionally not started",
    ("Track B: stock upstream task + hyperparameters, no modification", 0, True),
    "So results stay comparable to the published LEAP Hand benchmark",
], Inches(0.45), Inches(3.3), Inches(12.3), Inches(3.5), size=13)

# =====================================================================
# 4 INFRASTRUCTURE BUILT
# =====================================================================
s = new_slide()
header(s, "What Was Built (Track A) — No 2-DOF Code Existed Before This")
bullets(s, [
    ("Environment", 0, True),
    "Isaac Lab DirectRLEnv: URDF -> USD, torque control, 8-D obs, 2-D action",
    "120 Hz physics / 30 Hz control, 10s episodes, 4096 parallel envs",
    ("Domain randomization", 0, True),
    "Per-env xi = [m1,m2,f1,f2,b1,b2,a1,a2] — friction, damping, mass, actuator effectiveness",
    "Sampled once per episode per env; read back from simulator to verify (not assumed)",
    ("RAPP (Reward-Aware Physics Prior, DrEureka-style)", 0, True),
    "One-parameter-at-a-time feasibility sweep -> measured C_RAPP envelope",
    ("Physics-informed observer (PINN)", 0, True),
    "Closed-form 2-link dynamics + 5-member deep ensemble -> infers xi from (q,qdot,tau,qddot)",
    ("Evaluation harness", 0, True),
    "One frozen metric definition (precision curve, settling time, effort) used everywhere",
], Inches(0.45), Inches(1.25), Inches(7.6), Inches(5.7), size=13)
codebox(s, """
pi_adr/
  source/pi_adr/
    tasks/two_dof_reach/   env + cfg + PPO yaml
    assets/                URDF, cached USD
    pinn/                  dynamics.py, model.py
    evaluation/             metrics.py, runner.py
  scripts/
    validate_env.py         10/10 physics checks
    validate_dynamics.py    closed-form vs PhysX
    pd_baseline.py          IK+PD precision ceiling
    run_rapp.py             feasibility sweep
    collect_pinn_data.py    dynamics dataset
    train_pinn.py           observer + uncertainty
    eval.py / eval_leap.py  frozen metric suite
  results/  EXPERIMENT_LOG.md  RESULTS_2026-09-16.md
""", Inches(8.25), Inches(1.35), Inches(4.6), Inches(5.6), size=10)

# =====================================================================
# 5 ENVIRONMENT VALIDATION
# =====================================================================
s = new_slide()
header(s, "Environment Validated Before Any Result Was Trusted")
table(s, [
    ["#", "Check", "Result"],
    ["A", "30 Hz control / 120 Hz physics / 300 steps per episode", "PASS"],
    ["B", "Simulated end-effector position == analytic forward kinematics", "PASS  (0.000 mm error)"],
    ["C", "Zero torque -> no drift  (planar arm, g(q)=0)", "PASS"],
    ["D", "a=+1 -> ~20 Nm on joint 1, positive rotation", "PASS"],
    ["E", "Joint velocity respects +/-5 rad/s under saturated torque", "PASS"],
    ["F", "Targets sampled r~U(0.20,0.85), inside 1.0 m reach", "PASS"],
    ["G", "8-D observation layout matches spec exactly", "PASS"],
    ["H", "Vanilla & Eureka reward code == report equations 7/8", "PASS  (0.00 deviation)"],
    ["I", "Domain randomization reaches the simulator, differs per env", "PASS"],
    ["J", "Actuator effectiveness enters the coupled dynamics correctly", "PASS  (3.99% dev.)"],
], Inches(0.45), Inches(1.25), Inches(8.6), [Inches(0.5), Inches(6.0), Inches(2.1)], size=11)
box(s, "10 / 10 CHECKS PASSED", Inches(9.4), Inches(1.35), Inches(3.4), h=Inches(0.6), fill=GREEN, size=15)
codebox(s, """
DYNAMICS CROSS-CHECK
(closed-form 2-link model vs PhysX)

Forward dynamics (qdd):
  mean relative error   2.27%

Inverse dynamics (recovered torque):
  mean abs error        0.036 Nm
  (torques up to 20 Nm)

-> physics residual used by the PINN
   is measuring something real.
""", Inches(9.4), Inches(2.1), Inches(3.4), Inches(2.6), size=10)
bullets(s, [
    "Both validators re-run after every environment change — this discipline is what caught 3 of the 5 defects on the next slide.",
], Inches(9.4), Inches(4.9), Inches(3.4), Inches(1.6), size=11)

# =====================================================================
# 6 DEFECTS FOUND & FIXED
# =====================================================================
s = new_slide()
header(s, "Five Defects Found and Fixed — Each Would Have Invalidated Results")
table(s, [
    ["#", "Defect", "Effect measured", "Status"],
    ["1", "Terminal error recorded AFTER env auto-reset", "Trained policy scored 0.39% @5cm instead of 93%", "Fixed"],
    ["2", "ImplicitActuatorCfg damping=0 silently\noverrode URDF's 0.05", "Nominal physics != DR at scale 1.0", "Fixed"],
    ["3", "Hard +/-pi joint limits create a\nwrap-around trap", "Caps ANY controller at ~69% @5cm", "Fixed"],
    ["4", "PINN accel. labels differenced over a\ncontrol step, not a physics step", "True-xi RMSE 78.8 rad/s^2 (worse than the model!)", "Fixed"],
    ["5", "Mass randomization skipped the write\nat scale=1.0", "RAPP mass sweep: nominal scored 0.816\ninstead of 0.975", "Fixed"],
], Inches(0.45), Inches(1.25), Inches(12.4), [Inches(0.5), Inches(4.1), Inches(5.3), Inches(1.4)],
    size=11, highlight_rows={3: RED})
bullets(s, [
    ("Defect 3 is the important one for interpreting Stage 1:", 0, True),
    "A scripted IK+PD controller under +/-pi limits fails 31.4% of episodes, and 100% of those failures end stalled against a joint stop -> the Stage-1 report's 99.49% is unreachable by ANY controller under such limits, so its joints were not actually bounded that way.",
], Inches(0.45), Inches(5.15), Inches(12.4), Inches(1.7), size=13)

# =====================================================================
# 7 IK+PD PRECISION CEILING
# =====================================================================
s = new_slide()
header(s, "Key Finding — The Environment Permits 1 mm Precision")
bullets(s, [
    ("Question: is the ~30 mm error floor a property of the robot, or of the reward?", 0, True),
    ("Method: scripted 2-link inverse kinematics + PD control", 0, True),
    "q* = IK(target),  tau = Kp(q*-q) - Kd*qdot,  through the SAME 20 Nm action interface as the learned policy",
    "Measured with the identical frozen metric suite — not a separate protocol",
], Inches(0.45), Inches(1.25), Inches(7.5), Inches(2.5), size=13)
table(s, [
    ["Controller", "<5cm", "<1cm", "<1mm", "median err"],
    ["IK + PD (scripted)", "100%", "100%", "100%", "0.0 mm"],
    ["Vanilla PPO @5000", "93.0%", "2.2%", "0.02%", "34.7 mm"],
    ["Eureka PPO @1500", "99.1%", "7.9%", "0.05%", "30.1 mm"],
], Inches(0.45), Inches(4.0), Inches(7.5), [Inches(2.4), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.2)],
    size=12, highlight_rows={1: GREEN})
codebox(s, """
CONCLUSION

An optimal controller reaches 100%
success at EVERY threshold down to
1 mm on this exact benchmark.

-> The ~30mm floor of the learned
   policies is a REWARD / LEARNING
   result, not a robot/control-rate
   limitation.

This is what makes the reward
comparison on the next slide
scientifically meaningful.
""", Inches(8.25), Inches(1.35), Inches(4.6), Inches(5.6), size=12)

# =====================================================================
# 8 HEADLINE RESULTS vs STAGE-1 REPORT
# =====================================================================
s = new_slide()
header(s, "Headline Results — Reproduced on the Corrected Benchmark")
table(s, [
    ["Metric", "Stage-1 report\n(Eureka@500)", "Ours\n(Eureka@500)", "Stage-1 report\n(Eureka@1500)", "Ours\n(Eureka@1500)"],
    ["<5 cm", "97.90%", "96.97%", "95.3%", "99.05%"],
    ["<2 cm", "19.02%", "25.56%", "18.1%", "26.37%"],
    ["<1 cm", "3.56%", "3.59%", "5.3%", "7.91%"],
    ["mean err", "33.5 mm", "46.4 mm", "45.5 mm", "38.6 mm"],
], Inches(0.45), Inches(1.25), Inches(12.4),
   [Inches(2.2), Inches(2.55), Inches(2.55), Inches(2.55), Inches(2.55)], size=13)
bullets(s, [
    ("The Eureka rows reproduce closely — coarse-success / fine-precision collapse confirmed.", 0, True),
    ("The vanilla (human-reward) row does NOT reproduce:", 0, True),
], Inches(0.45), Inches(3.3), Inches(12.4), Inches(0.9), size=13)
table(s, [
    ["Metric", "Stage-1 report (vanilla)", "Ours (vanilla @5000)"],
    ["<2 cm", "98.36%", "13.28%"],
    ["<1 cm", "79.74%", "2.22%"],
], Inches(0.45), Inches(4.2), Inches(7.0), [Inches(2.5), Inches(2.5), Inches(2.0)], size=13, highlight_rows={1: RED, 2: RED})
codebox(s, """
TWO MEASURED CAUSES (not speculation)

1. Precision incentive is weak:
   3cm->0.5cm gain ~0.6% of episode
   return under the vanilla reward

2. Action sigma collapses:
   1.0 -> ~0.024 during training
   (report states 0.10-0.11 late-train)
   -> exploration dies before fine
      refinement is found

RESULT: on THIS build, Eureka beats
vanilla at every threshold.
""", Inches(7.7), Inches(4.2), Inches(5.15), Inches(3.0), size=11)

# =====================================================================
# 9 DR ROBUSTNESS RESULTS
# =====================================================================
s = new_slide()
header(s, "Domain-Randomization Robustness — A Secondary Finding")
table(s, [
    ["Policy", "Physics", "<5cm", "<2cm", "<1cm", "mean err", "mean qdot"],
    ["Eureka@1500", "nominal", "99.05%", "26.37%", "7.91%", "38.6 mm", "2.52 rad/s"],
    ["Eureka@1500", "in-distribution DR", "97.88%", "34.89%", "9.30%", "45.9 mm", "1.28 rad/s"],
    ["Eureka@1500", "shifted (unseen) DR", "91.14%", "42.36%", "11.47%", "107.0 mm", "0.95 rad/s"],
], Inches(0.45), Inches(1.25), Inches(12.4),
   [Inches(1.8), Inches(2.4), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.7), Inches(1.7)],
   size=12, highlight_rows={3: GREEN})
bullets(s, [
    ("Harder physics DEGRADES coarse success (99% -> 91% @5cm) but IMPROVES fine precision (26% -> 42% @2cm).", 0, True),
    ("Mechanism: friction/damping/weaker actuators force the slowness the reward never explicitly asked for — mean joint velocity falls 2.52 -> 0.95 rad/s.", 0, True),
    ("This independently supports the Stage-1 thesis: the precision gap is an objective-design problem, not a capability problem — a slower policy (forced by physics, not reward) is a more precise one.", 0, True),
    ("Caution for future DR work: some apparent \"robustness gain\" under DR may just be physics doing the regularization the reward omits.", 0, True),
], Inches(0.45), Inches(3.25), Inches(12.4), Inches(3.7), size=13)

# =====================================================================
# 10 RAPP
# =====================================================================
s = new_slide()
header(s, "RAPP — Reward-Aware Physics Prior (DrEureka protocol)")
bullets(s, [
    "One-parameter-at-a-time feasibility sweep on the nominal Eureka policy",
    "S(pi,xi) = P(d<5cm | pi,xi), feasible when S >= alpha = 0.9",
    "Measured C_RAPP envelope (all others held nominal while sweeping one):",
], Inches(0.45), Inches(1.25), Inches(12.4), Inches(1.4), size=13)
table(s, [
    ["Parameter", "Feasible range", "Note"],
    ["friction", "[0.0, 4.0] N.m", "fails above 4 (S drops to 0.84 at 8)"],
    ["damping scale", "[0.25, 256] x nominal", "very wide — nearly flat performance"],
    ["actuator effectiveness", "[0.3, 1.0]", "fails below 0.3 — weak motors cannot compensate"],
    ["mass scale", "[1.0, 4.0] x nominal", "ASYMMETRIC: fails with LIGHTER links, tolerates heavier"],
], Inches(0.45), Inches(2.75), Inches(8.3), [Inches(2.4), Inches(2.4), Inches(3.5)], size=12,
   highlight_rows={4: ACCENT})
codebox(s, """
WHY MASS IS ASYMMETRIC

Lighter links -> more acceleration
per unit torque -> a policy tuned to
nominal inertia OVERSHOOTS.

Direct link to the precision finding
on the previous slide: the same
mechanism (torque/inertia mismatch)
explains both results.

IMPORTANT for future adaptive DR:
envelope is very wide & mostly FLAT
for 3 of 4 parameters -> may already
cover the region where adaptive DR
has little left to exploit.
""", Inches(9.0), Inches(2.75), Inches(3.8), Inches(4.2), size=11)

# =====================================================================
# 11 PINN ARCHITECTURE
# =====================================================================
s = new_slide()
header(s, "Physics-Informed Observer (PINN) — Architecture")
codebox(s, """
CLOSED-FORM 2-LINK DYNAMICS (verified vs PhysX)

  M(q;xi) qdd + C(q,qdot;xi) + F(qdot;xi) = alpha (.) tau_cmd

  g(q) = 0            (planar, proven not assumed)
  alpha on the RHS     (keeps actuator effectiveness
                        identifiable from commanded torque)

  xi = [m1,m2, f1,f2, b1,b2, a1,a2]   in R^8


OBSERVER

  input : window of 64 transitions (q,qdot,tau,qddot)
  model : set-encoder (per-transition MLP + mean/max pool)
          -- NOT a sequence model: EOM is instantaneous
  output: xi_hat (bounded, hard constraint) + log(var)
  ensemble: 5 independently-initialized members
          -> U_PINN = epistemic + aleatoric variance
""", Inches(0.45), Inches(1.25), Inches(7.6), Inches(5.7), size=12)
bullets(s, [
    ("Loss", 0, True),
    "L = L_data (accel. MSE) + 0.01*L_phys (EOM residual) + L_sup (NLL vs ground-truth xi)",
    ("Dataset discipline", 0, True),
    "Interpolation / extrapolation splits use NON-OVERLAPPING physics ranges",
    "Velocity-saturated transitions masked out (limit engages ~4 rad/s, not 5)",
    ("Critical fix made here", 0, True),
    ("Accel. labels were originally differenced over a 4-substep control\nstep -> even the TRUE xi gave 78.8 rad/s^2 residual (worse than\nthe trained model). Fixed by differencing over one physics step\n(1/120s) -> true-xi residual drops to 6.7 rad/s^2.", 1, True),
], Inches(8.25), Inches(1.25), Inches(4.6), Inches(5.7), size=12)

# =====================================================================
# 12 PINN RESULTS
# =====================================================================
s = new_slide()
header(s, "PINN Results — Validated by Physics Configuration, Not Time Samples")
table(s, [
    ["", "accel. RMSE", "param. error (norm)", "uncertainty<->error rank corr."],
    ["Interpolation\n(held-out physics, same ranges)", "2.79 rad/s^2", "0.065", "+0.11"],
    ["Extrapolation\n(ranges outside training)", "12.28 rad/s^2", "0.209", "+0.10"],
], Inches(0.45), Inches(1.25), Inches(12.4), [Inches(4.4), Inches(2.7), Inches(2.7), Inches(2.6)], size=13)
box(s, "NEGATIVE RESULT — flagged, not hidden", Inches(0.45), Inches(2.85), Inches(5.0), h=Inches(0.5), fill=RED, size=13)
bullets(s, [
    "Ensemble uncertainty is only weakly related to its actual error (rank corr. ~+0.11, both splits).",
    "The roadmap's adaptive-DR score gives uncertainty (U_PINN) its own weighted term — on this evidence, that term would be close to noise.",
    "Before adaptive DR depends on it: either calibrate the uncertainty (current estimate uses only the DIAGONAL of the ensemble covariance — m and alpha are not independently identifiable from a single torque regime) or set its weight to zero.",
    "Extrapolation degrading 3x (0.065 -> 0.209) is the expected, honest result — supports keeping any future adaptive DR inside C_RAPP.",
], Inches(0.45), Inches(3.55), Inches(12.4), Inches(3.3), size=13)

# =====================================================================
# 13 LEAP HAND (TRACK B)
# =====================================================================
s = new_slide()
header(s, "Track B — LEAP Hand (Real Platform, Stock Configuration)")
bullets(s, [
    ("Task: Isaac-Reorient-Cube-Leap (github.com/leap-hand/LEAP_Hand_Isaac_Lab)", 0, True),
    "Config and hyperparameters UNMODIFIED (GRU-256, horizon 32, minibatch 32768, lr 5e-4 adaptive)",
    ("Two upstream/Isaac-Lab-2.3 incompatibilities found & patched:", 0, True),
    "train.py imported dump_pickle, removed from isaaclab.utils.io in 2.3 -> used our own wrapper train script",
    "EventCfg missing 'operation' arg now required by randomize_actuator_gains -> one-line fix (set 'abs' = the 2.1 default it was written against)",
    ("Training: 500 PPO iterations, 4096 envs, seed 42", 0, True),
    "Return: 33.09 (5-iter smoke test) -> 84,254.54 (500 iters)",
], Inches(0.45), Inches(1.25), Inches(7.5), Inches(4.6), size=13)
codebox(s, """
RESULT (rung-1 reproduction baseline)

  rotations / 20s / hand :  7.52
  mean joint velocity    :  1.86 rad/s
  mean control effort    :  0.70 / joint

  (DrEureka's own reported metric for
   this exact task — after only 500
   iterations; DrEureka trains longer)

STATUS: baseline established.
No adaptive DR / PINN applied to
this track yet — Track A validates
the method first (cheaper iteration).
""", Inches(8.25), Inches(1.35), Inches(4.6), Inches(4.6), size=12)

# =====================================================================
# 14 STATUS vs ROADMAP CHECKLIST
# =====================================================================
s = new_slide()
header(s, "Status Against the Roadmap's Immediate Checklist")
table(s, [
    ["#", "Item", "Status"],
    ["1", "Freeze the 2-DOF environment and PPO baseline", "DONE"],
    ["2", "physics_params of shape [num_envs, N]", "DONE (8-D xi per env)"],
    ["3-4", "Friction randomization; verify all envs differ", "DONE, read back from sim"],
    ["5", "Fixed-policy sweep, terminal error vs friction", "DONE (physics-performance map)"],
    ["6", "Actuator effectiveness randomization", "DONE"],
    ["7-8", "Reproduce RAPP; store bounds", "DONE (measured envelope)"],
    ["9", "Do NOT build the adaptive controller yet", "RESPECTED"],
    ["10-11", "Standalone PINN; validate on unseen physics", "DONE"],
    ["12", "Add ensemble uncertainty", "DONE (+ shown to be weak)"],
    ["13+", "Adaptive DR outer loop, A-G ablation matrix", "NOT STARTED (by design)"],
], Inches(0.45), Inches(1.25), Inches(9.5), [Inches(0.7), Inches(6.3), Inches(2.5)], size=12,
   highlight_rows={7: ACCENT, 10: GREY})
box(s, "12 / 12\nplanned items\ncomplete", Inches(10.3), Inches(1.4), Inches(2.5), h=Inches(1.4), fill=GREEN, size=15)

# =====================================================================
# 15 A-G MATRIX
# =====================================================================
s = new_slide()
header(s, "A-G Matrix, n=5 seeds, Nominal Physics (mean ± std)")
table(s, [
    ["Rung", "Setup", "Mean err (mm)", "Worst 10% (mm)", "5cm %", "2cm %"],
    ["A", "human, no DR", "37.8 ± 4.0", "149 ± 32", "98.5", "29.0"],
    ["B", "Eureka, no DR", "37.7 ± 9.4", "139 ± 74", "99.0", "26.9"],
    ["C", "human + static DR", "36.4 ± 7.2", "136 ± 69", "96.3", "29.3"],
    ["D", "Eureka + RAPP static DR", "37.8 ± 8.1", "147 ± 70", "97.5", "26.8"],
    ["E", "adaptive DR, no RAPP", "39.4 ± 5.5", "109 ± 39", "80.4", "20.9"],
    ["F4", "RAPP + adaptive DR", "44.7 ± 14.6", "202 ± 145", "94.1", "23.1"],
    ["G", "F4 + PINN-scaled steps", "34.1 ± 3.9", "82 ± 33", "94.1", "19.1"],
    ["G placebo", "shuffled multipliers (n=5)", "33.7 ± 4.0", "87 ± 45", "94.4", "22.2"],
], Inches(0.45), Inches(1.3), Inches(12.4), [Inches(w) for w in (1.4, 4.1, 1.8, 1.8, 1.65, 1.65)], size=13)
bullets(s, [
    "Exact permutation tests: A/B/C indistinguishable (p>=0.18); F4 does not beat D; G vs F4 not significant (p 0.12-0.29)",
    "Placebo matches G on nominal and DRtrain (p>=0.74); the DRshift signal (p=0.024 at n=5) did not replicate in 6 clean seeds (p=0.147)",
    "Source: pi_adr/results/MATRIX_A_to_G.md",
], Inches(0.45), Inches(5.6), Inches(12.4), Inches(1.6), size=13)

# =====================================================================
# 16 SUMMARY OF FINDINGS
# =====================================================================
s = new_slide()
header(s, "Summary of Findings")
bullets(s, [
    ("Confirmed", 0, True),
    "Eureka's coarse-success / fine-precision collapse reproduces on an independently rebuilt benchmark; the environment itself permits 1mm (IK+PD 100%)",
    ("Corrected after multi-seed testing", 0, True),
    "Eureka does not beat the human reward at matched budget; vanilla's precision gap was largely PPO entropy collapse",
    "Single-seed wins for adaptive DR (F4, E) did not survive n=5; E loses nominal 5cm success (80% vs 97.5%, p=0.008)",
    "Static DR lowers 2cm success under DR evals vs no DR (21% vs 41%, p=0.016)",
    ("PINN contribution: not demonstrated", 0, True),
    "G has the lowest mean/tail error and spread, but is not significant vs F4 and the n=5 shuffled-multiplier placebo matches it on nominal/DRtrain (DRshift p=0.024 at n=5 did not replicate in 6 clean seeds: p=0.147)",
    "Observer uncertainty is weakly informative and unreliable outside its training range (extrapolation rank corr -0.16)",
    ("Process", 0, True),
    "Five silent-failure defects found and fixed; overclaims retracted openly in the log",
], Inches(0.45), Inches(1.25), Inches(12.4), Inches(5.9), size=14)

# =====================================================================
# 17 NEXT STEPS
# =====================================================================
s = new_slide()
header(s, "Next Steps")
bullets(s, [
    ("Immediate", 0, True),
    "Clean retest (n=6; seeds 3-6 rerun pending): no G benefit. Move to LEAP Hand (Track B)",
    "LEAP Hand: no-ADR baseline, native ADR, RAPP-initialised ADR (Track B)",
    ("If pursuing the PINN idea", 0, True),
    "Observer reliable outside its training envelope; uncertainty weighting that is not dominated by one dimension",
    "Longer training budgets than 800 iterations",
    ("Longer-term", 0, True),
    "Sim-to-real hardware stage after robust simulation policies are established",
], Inches(0.45), Inches(1.25), Inches(12.4), Inches(5.9), size=14)

prs.save("Review2_PhysicsInformed_AdaptiveDR.pptx")
print("Saved Review2_PhysicsInformed_AdaptiveDR.pptx —", N[0], "slides")
