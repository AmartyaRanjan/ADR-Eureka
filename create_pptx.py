# create_pptx.py  —  Review 1 PPT: Eureka + DrEureka + PINN
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

NAVY   = RGBColor(0x0B, 0x25, 0x45)
TEAL   = RGBColor(0x13, 0x7C, 0x8A)
ACCENT = RGBColor(0xF2, 0x8C, 0x28)
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
SLIDE_NO = [0]

def new_slide():
    SLIDE_NO[0] += 1
    s = prs.slides.add_slide(BLANK)
    return s

def footer(slide):
    tb = slide.shapes.add_textbox(Inches(12.6), Inches(7.1), Inches(0.7), Inches(0.35))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = str(SLIDE_NO[0]); r.font.size = Pt(10); r.font.color.rgb = GREY

def header(slide, title):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background(); bar.shadow.inherit = False
    tf = bar.text_frame; tf.margin_left = Inches(0.4); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = WHITE
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
    gf = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, Inches(0.4 * len(rows)))
    t = gf.table
    for i, w in enumerate(col_widths):
        t.columns[i].width = w
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = t.cell(ri, ci); c.text = str(val)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = Pt(6); c.margin_right = Pt(4)
            c.margin_top = Pt(3); c.margin_bottom = Pt(3)
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

def flow(slide, text, top, width=Inches(4.4), fill=NAVY, left=Inches(4.6), h=Inches(0.7)):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = WHITE
    return sh

def arrow(slide, left, top, w, h, kind=MSO_SHAPE.DOWN_ARROW):
    a = slide.shapes.add_shape(kind, left, top, w, h)
    a.fill.solid(); a.fill.fore_color.rgb = ACCENT
    a.line.fill.background(); a.shadow.inherit = False

# ============ SLIDE 1 : TITLE ============
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.0), Inches(2.2), Inches(0.08))
acc.fill.solid(); acc.fill.fore_color.rgb = ACCENT; acc.line.fill.background(); acc.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(2.0))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "Physics-Informed Reward & Dynamics Learning\nfor Dexterous Manipulation"
r.font.size = Pt(36); r.font.bold = True; r.font.color.rgb = WHITE
p = tf.add_paragraph(); p.space_before = Pt(14)
r = p.add_run(); r.text = "Eureka  •  DrEureka (RAPP)  •  PINN  •  HER + SAC  on  NVIDIA Isaac Gym"
r.font.size = Pt(18); r.font.color.rgb = ACCENT
p = tf.add_paragraph(); p.space_before = Pt(26)
r = p.add_run(); r.text = "Final Year Project  |  Review 1  |  Team: [Name1 / Roll], [Name2], [Name3]  |  Guide: [Name]  |  Aug 2026"
r.font.size = Pt(13); r.font.color.rgb = LIGHT

# ============ SLIDE 2 : ABSTRACT & PROBLEM ============
s = new_slide()
header(s, "Abstract & Problem Definition")
bullets(s, [
    ("Reward-engineering bottleneck:", 0, True),
    ("Hand-crafted rewards per task; do not port across morphologies (suite: 10 robots, 29 tasks)", 1, False),
    ("Sim-to-real gap:", 0, True),
    ("Contact-rich tasks fail on friction / mass / latency mismatch at deployment", 1, False),
    ("Sparse multi-goal rewards -> sample-inefficient RL; naive DR ranges -> untrainable policies", 0, False),
    ("Physics-agnostic dynamics models waste data; physics must enter as inductive bias (PINN)", 0, False),
    ("Thesis: couple LLM reward synthesis (Eureka) + reward-aware DR (DrEureka/RAPP) with a LuGre-consistent PINN friction model for data-efficient, transferable dexterous policies", 0, False),
], Inches(0.5), Inches(1.3), Inches(6.6), Inches(5.6))
codebox(s, """
TARGET TASKS & BENCHMARKS
-------------------------
Bench : Isaac Gym (NeurIPS'21) + Dexterity
Task1 : ShadowHand pen-spin
Task2 : ShadowHand cube (Rubik) rotation
Task3 : Allegro grasp-and-throw
Task4 : Bimanual hand-over (ARCTIC setup)
Metric: success %, sim-real gap, samples
""", Inches(7.4), Inches(1.4), Inches(5.4), Inches(3.2))

# ============ SLIDE 3 : OBJECTIVES & OUTCOMES ============
s = new_slide()
header(s, "Objectives & Expected Outcomes")
bullets(s, [
    ("O1 - LLM reward synthesis (Eureka): GPT-4/CodeLlama reward-code + evolutionary search", 0, False),
    ("O2 - Physics-consistent DR (DrEureka/RAPP): feasible friction/mass/motor/latency ranges", 0, False),
    ("O3 - PINN friction & contact model: LuGre-consistent loss on human-hand trajectories", 0, False),
    ("O4 - Sparse-reward RL core: HER + SAC, ablate DHER / MOEHER", 0, False),
], Inches(0.5), Inches(1.3), Inches(6.6), Inches(3.0))
table(s, [
    ["Metric", "Target"],
    ["Reward vs human experts", ">= 83% tasks matched/beaten"],
    ["Sim-to-real gap", "< 8% (DrEureka: 12%)"],
    ["PINN force pred. error", "< 5%; data use -50..70%"],
    ["Grasping success", ">= 90%"],
    ["Train budget", "<= 60 GPU-hrs / task"],
], Inches(7.3), Inches(1.4), Inches(5.5), [Inches(2.4), Inches(3.1)])

# ============ SLIDE 4 : SCOPE & FEASIBILITY ============
s = new_slide()
header(s, "Scope & Feasibility")
table(s, [
    ["Component", "IN scope", "OUT of scope"],
    ["Simulation", "Isaac Gym, MuJoCo", "Gazebo, Webots"],
    ["Robots", "ShadowHand, Allegro, Panda", "Quadrupeds, UAVs"],
    ["Tasks", "Manipulation, grasping", "Locomotion, SLAM"],
    ["PINN focus", "Friction, contact dynamics", "Vision, perception"],
    ["LLM", "GPT-4 API, CodeLlama-34B", "Custom fine-tunes"],
], Inches(0.5), Inches(1.3), Inches(7.0), [Inches(1.6), Inches(2.7), Inches(2.7)])
codebox(s, """
COMPUTE BUDGET
--------------
Eureka/DrEureka : 4x A100-40GB, ~2-3 wk / task family
PINN            : 1x RTX 3090/4090, 3-5 days
RL (HER+SAC)    : Isaac Gym, 4096 parallel envs
Fallback        : cloud credits; CodeLlama offline
""", Inches(7.8), Inches(1.4), Inches(5.0), Inches(2.4))
bullets(s, [
    "Feasible: all code open-source (eureka-research/eureka, /dreureka); datasets public; sim-first, real robot only wk 10+",
], Inches(7.8), Inches(4.0), Inches(5.0), Inches(1.5), size=11)

# ============ SLIDE 5 : WORK PLAN ============
s = new_slide()
header(s, "Preliminary Work Plan (16 Weeks)")
table(s, [
    ["Phase", "Weeks", "Deliverable", "Success criterion"],
    ["Setup & literature", "1-2", "Isaac Gym + repos integrated", "Envs render, baselines run"],
    ["Baseline RL", "3-4", "HER+SAC pick-and-place", ">= 60% success"],
    ["Eureka integration", "5-7", "LLM reward pipeline (GPT-4)", "+15% over baseline, 3 tasks"],
    ["DrEureka + RAPP", "8-10", "DR configs; sim-real ShadowHand", "Zero-shot on 2 tasks"],
    ["PINN friction", "11-13", "LuGre-PINN on ARCTIC", "< 5% test error"],
    ["Integration/ablation", "14-15", "Full pipeline, 10-task eval", ">= 85% avg success"],
    ["Report & demo", "16", "Thesis, video, defense", "-"],
], Inches(0.5), Inches(1.3), Inches(12.3), [Inches(2.4), Inches(1.2), Inches(4.4), Inches(4.3)], size=11)
table(s, [
    ["Member", "Ownership"],
    ["M1", "Eureka/DrEureka pipeline, LLM, RAPP"],
    ["M2", "PINN: friction model, data, physics loss"],
    ["M3", "Sim setup, benchmarks, sim-to-real deploy"],
], Inches(0.5), Inches(5.3), Inches(7.0), [Inches(1.4), Inches(5.6)], size=11)

# ============ SLIDE 6 : ARCHITECTURE ============
s = new_slide()
header(s, "System Architecture")
tops = [1.25, 2.23, 3.21, 4.19, 5.17, 6.15]
labels = [
    "Task spec (NL) + physics params",
    "EUREKA: LLM reward synthesis + evolutionary search",
    "RL core: SAC + HER in Isaac Gym (4096 envs)",
    "DrEUREKA / RAPP: reward-aware DR ranges",
    "PINN: LuGre-consistent friction & contact model",
    "Sim-to-Real: ShadowHand / Allegro zero-shot",
]
fills = [TEAL, NAVY, TEAL, NAVY, TEAL, NAVY]
for i, (t, f) in enumerate(zip(labels, fills)):
    flow(s, t, Inches(tops[i]), fill=f)
    if i < 5:
        arrow(s, Inches(6.62), Inches(tops[i] + 0.72), Inches(0.35), Inches(0.26))
flow(s, "GPT-4 / CodeLlama-34B\nreward code + feedback", Inches(1.35), width=Inches(3.0), h=Inches(0.7), fill=ACCENT, left=Inches(0.5))
arrow(s, Inches(3.55), Inches(1.45), Inches(0.5), Inches(0.28), MSO_SHAPE.RIGHT_ARROW)
flow(s, "ARCTIC 2.1M frames\nDexterCap / DexGraspNet traj.", Inches(5.27), width=Inches(3.0), h=Inches(0.7), fill=ACCENT, left=Inches(0.5))
arrow(s, Inches(3.55), Inches(5.37), Inches(0.5), Inches(0.28), MSO_SHAPE.RIGHT_ARROW)

# ============ SLIDE 7 : ALGO 1 EUREKA ============
s = new_slide()
header(s, "Algorithm 1 - Eureka Reward Synthesis")
codebox(s, """
Algorithm 1: EUREKA (Ma et al., 2023)
Input: task NL T, env E, LLM M, gens G=4, pop N=8
1:  R_0 <- M.sample_code(T, N)            # zero-shot reward programs
2:  for g in 0..G-1:
3:    for r in R_g:  pi_r <- PPO_train(E, r)     # Isaac Gym, GPU
4:    s(r) <- EVAL(pi_r, seeds=3)                # return curves
5:    R_{g+1} <- M.evolve(T, top-k(R_g), s)      # in-context reflection
6:  return argmax_r s(r)
""", Inches(0.5), Inches(1.3), Inches(7.2), Inches(3.4), size=11)
bullets(s, [
    ("Proof-of-work knobs:", 0, True),
    "Population N=8, generations G=4, top-k=2",
    "In-context feedback = training return curves",
    "Eval: 3 seeds/task, 20 Eureka tasks benchmark",
    "Baseline claim: beats human rewards on 83% tasks",
], Inches(8.0), Inches(1.3), Inches(4.8), Inches(3.5), size=12)

# ============ SLIDE 8 : ALGO 2 RAPP ============
s = new_slide()
header(s, "Algorithm 2 - DrEureka / RAPP Domain Randomization")
codebox(s, """
Algorithm 2: RAPP (DrEureka, 2024)
Input: reward r*, env E, params P = {mu, m, motor, lat}
1: for p in P:                          # reward-aware feasibility sweep
2:   for v in grid(p):  pi_v <- short_train(E(p=v), r*)
3:   F_p <- { v : success(pi_v) >= 0.8 }
4:   [l_p, u_p] <- bounds(F_p)          # RAPP prior
5: DR <- M_llm( prompt(r*, {[l_p, u_p]}) )   # joint config selection
6: rank configs by sim success + real alignment
Default ranges: mu in [0.3, 1.2], m +-20%,
                motor in [0.8, 1.2], lat in [0, 50] ms
""", Inches(0.5), Inches(1.3), Inches(7.6), Inches(3.6), size=11)
bullets(s, [
    ("Why RAPP > manual DR / ADR:", 0, True),
    "Ranges conditioned on the *learned reward*, not task-agnostic",
    "Short-horizon sweeps keep compute bounded",
    "LLM composes joint config from per-param priors",
], Inches(8.4), Inches(1.3), Inches(4.4), Inches(3.0), size=12)

# ============ SLIDE 9 : PINN ============
s = new_slide()
header(s, "PINN - Physics-Informed Friction & Contact Model")
codebox(s, """
Dynamics : m*q_ddot + C(q,q_dot)*q_dot + g(q) + f_phi(q,q_dot) = tau
LuGre    : f = s0*z + s1*z_dot + s2*q_dot
           z_dot = q_dot - s0*|q_dot|*z / g(q_dot)
Loss     : L = L_data + lam * L_phys
L_data   : ||q_pred - q_gt||^2        (ARCTIC / DexterCap traj.)
L_phys   : ||m*q_ddot + C*q_dot + g + f_phi - tau||^2
Net      : MLP 8x256 tanh | AdamW lr 1e-3 cosine | lam: 0.1 -> 1
""", Inches(0.5), Inches(1.3), Inches(7.6), Inches(3.2), size=11)
bullets(s, [
    ("Training data:", 0, True),
    "ARCTIC: 2.1M frames, 3D hand + object pose",
    "DexterCap: Rubik's / multi-joint sequences",
    "DexGraspNet: 1.32M ShadowHand grasps",
    ("Output use:", 0, True),
    "f_phi injected into Isaac Gym contact solver",
    "Semantic tokens of (s0,s1,s2) fed to LLM prompts",
], Inches(8.4), Inches(1.3), Inches(4.4), Inches(4.5), size=12)

# ============ SLIDE 10 : RL CORE ============
s = new_slide()
header(s, "RL Core - HER + SAC")
codebox(s, """
HER (Andrychowicz 2017):
  (s,a,r,s',g) -> relabel g' = goal(s'), k=8 future states
  r' = -||s' - g'||_2 ;  store original + relabeled
SAC objective:
  max_pi  E[ Q(s,a) ] + alpha * H(pi(.|s))   (auto-alpha)
""", Inches(0.5), Inches(1.3), Inches(6.6), Inches(2.2), size=11)
table(s, [
    ["Hyperparam", "Value"],
    ["Replay buffer", "1e6"],
    ["Batch", "256"],
    ["LR actor/critic", "3e-4"],
    ["Gamma", "0.99"],
    ["Parallel envs", "4096"],
    ["Horizon", "500"],
], Inches(7.6), Inches(1.3), Inches(5.0), [Inches(2.4), Inches(2.6)])

# ============ SLIDE 11 : DATASETS ============
s = new_slide()
header(s, "Datasets & Benchmarks")
table(s, [
    ["Dataset", "Type", "Scale", "Role in project"],
    ["Isaac Gym Envs", "Sim tasks", "10 robots / 29 tasks", "RL benchmark suite"],
    ["ARCTIC (CVPR'23)", "Bimanual mocap", "2.1M frames", "PINN trajectory data"],
    ["DexterCap", "Dexterous + Rubik's", "1.4K sequences", "Cube-rotation task"],
    ["DexGraspNet", "Grasps", "1.32M, 5355 objs", "Grasp priors"],
    ["HRDexDB", "Human-robot paired", "100 objects", "Cross-embodiment eval"],
], Inches(0.5), Inches(1.3), Inches(12.3), [Inches(2.6), Inches(2.4), Inches(2.8), Inches(4.5)])

# ============ SLIDE 12 : EXPECTED RESULTS ============
s = new_slide()
header(s, "Expected Results & Ablations")
table(s, [
    ["Metric", "HER", "Eureka", "DrEureka", "Ours (target)"],
    ["Pen-spin success", "45%", "78%", "82%", "88%"],
    ["Cube rotation", "38%", "71%", "76%", "84%"],
    ["Grasping success", "62%", "81%", "85%", "90%"],
    ["Sim-real gap", "35%", "18%", "12%", "< 8%"],
    ["GPU-hrs / task", "168", "72", "68", "60"],
    ["Data efficiency", "1.0x", "1.5x", "1.6x", "2.3x (PINN)"],
], Inches(0.5), Inches(1.3), Inches(12.3), [Inches(2.8), Inches(2.0), Inches(2.2), Inches(2.4), Inches(2.9)])
bullets(s, [
    "Ablations: reward source (human vs Eureka) | DR (manual vs ADR vs RAPP) | PINN on/off | HER vs DHER vs MOEHER",
], Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.0), size=12)

# ============ SLIDE 13 : RISKS ============
s = new_slide()
header(s, "Risk Analysis & Mitigation")
table(s, [
    ["Risk", "Prob.", "Impact", "Mitigation"],
    ["GPU shortage", "Med", "High", "Cloud credits; batch/env reduction"],
    ["PINN instability", "High", "Med", "Curriculum lambda; progressive training"],
    ["Sim-real failure", "Med", "High", "ADR fallback on top of RAPP"],
    ["LLM API cost", "Low", "Med", "Response cache; CodeLlama-34B offline"],
    ["Dataset access", "Low", "Low", "All datasets public (5 sources)"],
], Inches(0.5), Inches(1.3), Inches(12.3), [Inches(2.8), Inches(1.3), Inches(1.4), Inches(6.8)])

# ============ SLIDE 14 : CO MAPPING ============
s = new_slide()
header(s, "Evaluation Rubric - CO Mapping")
table(s, [
    ["No.", "Parameter", "Marks", "CO1", "CO2", "CO3", "CO4"],
    ["1", "Project title & technical relevance", "1", "-", "Y", "-", "Y"],
    ["2", "Abstract & problem definition", "1", "-", "Y", "-", "Y"],
    ["3", "Objectives & expected outcome", "1", "-", "Y", "-", "Y"],
    ["4", "Scope & feasibility", "1", "Y", "-", "-", "Y"],
    ["5", "Preliminary work plan", "1", "Y", "-", "Y", "Y"],
], Inches(0.5), Inches(1.3), Inches(12.3), [Inches(0.8), Inches(5.5), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)])

# ============ SLIDE 15 : REFERENCES ============
s = new_slide()
header(s, "Key References")
bullets(s, [
    "Ma et al., Eureka: Human-Level Reward Design via Coding LLMs, 2023 (arXiv:2310.12931)",
    "DrEureka: Language Model Guided Sim-To-Real Transfer, 2024 (arXiv:2406.01967)",
    "Makoviychuk et al., Isaac Gym: High Performance GPU-Based Physics Simulation, NeurIPS 2021",
    "Andrychowicz et al., Hindsight Experience Replay, NeurIPS 2017",
    "Raissi et al., Physics-Informed Neural Networks, JCP 2019",
    "PINN for Friction Dynamics / LuGre-PINN, 2024-25",
    "ARCTIC: Dataset for Dexterous Bimanual Hand-Object Manipulation, CVPR 2023",
    "DexterCap (PKU-MOCCA) & DexGraspNet (Galbot)",
], Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.5), size=13)

# ============ SLIDE 16 : CLOSING ============
s = new_slide()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(1), Inches(2.2), Inches(11.3), Inches(1.5))
r = tb.text_frame.paragraphs[0].add_run(); r.text = "Thank You  -  Questions?"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
codebox(s, """
github.com/eureka-research/eureka
github.com/eureka-research/dreureka
github.com/isaac-sim/IsaacGymEnvs
arctic.is.tue.mpg.de
""", Inches(4.4), Inches(4.0), Inches(4.5), Inches(1.8), size=11)

prs.save("Review1_Eureka_PINN.pptx")
print("Saved Review1_Eureka_PINN.pptx with", len(prs.slides._sldIdLst), "slides")
