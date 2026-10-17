# pi_adr — Physics-Informed Adaptive Domain Randomization

Working code for the second-stage research plan (`2DOF_Physics_Informed_Adaptive_DR_Research_Roadmap`).

**New to the project? Start with `project_docs/20_START_HERE_project_explained_from_scratch.md`.**

**Full written documentation of the whole project (why/what/findings/validation, 15 files): `project_docs/00_INDEX.md`.**

**Start here if picking this project up:** `PROJECT_STATUS_AND_ROADMAP.md` — current state, what's
trusted, what's open, and a prioritized, itemized plan for what to do next. `EXPERIMENT_LOG.md` is
the full chronological record everything in the status doc is drawn from.

Two benchmarks are driven in parallel from one metric definition:

| Track | Platform | Task | Why |
|---|---|---|---|
| **A** | 2-DOF planar arm (built here) | reach a randomly placed goal marker | cheap, fast testbed for RAPP → PINN → adaptive DR |
| **B** | LEAP Hand (16-DOF) | `Isaac-Reorient-Cube-Leap`, the task it ships trained for | real platform, stock upstream configuration |

Everything runs on the existing install at `../` (Isaac Lab 2.3.0 + Isaac Sim 5.1.0, venv `../env_isaaclab`, NVIDIA L4).

## Repositories and sources

Code in this directory builds directly on, or takes configuration verbatim from:

| What | URL | Local path | Used for |
|---|---|---|---|
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | `../IsaacLab` | simulator, `DirectRLEnv`, URDF→USD converter, the `train.py` this repo's wrapper is derived from |
| Isaac Sim | https://developer.nvidia.com/isaac/sim | venv pip pkgs | PhysX / USD runtime |
| rl_games | https://github.com/Denys88/rl_games | venv pip pkg (1.6.1) | PPO implementation for both tracks |
| IsaacLabEureka | https://github.com/NVlabs/IsaacLabEureka | `../IsaacLabEureka` | LLM reward search (locally patched for Gemini) |
| Eureka | https://github.com/eureka-research/Eureka | — | original reward-generation method |
| DrEureka | https://github.com/eureka-research/DrEureka | — | RAPP (Reward-Aware Physics Prior), the DR baseline we extend |
| LEAP_Hand_Isaac_Lab | https://github.com/leap-hand/LEAP_Hand_Isaac_Lab | `../LEAP_Hand_Isaac_Lab` | track-B task, config and pretrained checkpoint |
| LEAP_Hand_Sim | https://github.com/leap-hand/LEAP_Hand_Sim | (Isaac Gym version) | predecessor of the above |
| LEAP_Hand_API | https://github.com/leap-hand/LEAP_Hand_API | — | hardware SDK for eventual sim2real |
| IsaacLabExtensionTemplate | https://github.com/isaac-sim/IsaacLabExtensionTemplate | — | out-of-tree extension layout followed here |

Literature referenced by the roadmap: BayesSim (arXiv:1906.01728), Understanding Domain
Randomization for Sim-to-Real Transfer (arXiv:2110.03239), LEAP Hand (Shaw et al., RSS 2023).
The roadmap also cites a 2026 IEEE Access physics-informed dynamics-randomization paper and
Phys2Real (ICRA 2026); **these were not verifiable from this machine and are carried as
unchecked citations.**

## Track A: the 2-DOF reacher

Built from scratch (no such task existed in any repo here). Physical constants come from the
stage-1 report, and `scripts/validate_env.py` asserts the implementation matches its equations.

- links 0.5 m / 0.5 m, masses 1.0 kg / 0.5 kg, 2 revolute joints about Z
- torque control `tau = 20 * clip(a, -1, 1)`, velocity limit 5 rad/s, joint damping 0.05
- 8-D observation `[q1,q2,qd1,qd2,tx,ty,tx-ex,ty-ey]`, 2-D action
- 120 Hz physics / 30 Hz control (decimation 4), 10 s episodes = 300 control steps
- targets `r ~ U(0.20,0.85)`, `theta ~ U(0,2pi)`; success radius 5 cm
- the arm lies in the XY plane with joints about Z, so gravity exerts no joint torque
  (`g(q) = 0`), matching the report's dynamics

Two rewards, identical in every other respect:

```
R_vanilla = -d + 5*I(d<0.05) - 0.005*||a||^2          (human-designed)
R_eureka  = exp(-10d) + 100*I(d<0.05) - 0.01*||qd||^2 - 0.1   (LLM-discovered)
```

### Commands

```bash
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
PY=$ROOT/env_isaaclab/bin/python
cd $ROOT/pi_adr

# one-off: URDF -> USD (needs the library shim, see Known issues)
./scripts/convert_asset.sh --force

# verify the physics before trusting any number
$PY -u scripts/validate_env.py --headless

# train
$PY -u scripts/train.py --task PIADR-TwoDofReach-Vanilla-v0 --headless \
    --num_envs 4096 --max_iterations 5000 --seed 42
$PY -u scripts/train.py --task PIADR-TwoDofReach-Eureka-v0  --headless \
    --num_envs 4096 --max_iterations 500  --seed 42

# evaluate (frozen metric suite -> results/<run_name>.json + results/index.jsonl)
$PY -u scripts/eval.py --task PIADR-TwoDofReach-Vanilla-v0 --headless \
    --checkpoint logs/rl_games/two_dof_reach/<run>/nn/two_dof_reach.pth \
    --run_name A_vanilla_nominal
```

## Track B: LEAP Hand

Stock upstream task and hyperparameters — nothing tuned here, so results stay comparable with
the published configuration.

```bash
cd $ROOT/LEAP_Hand_Isaac_Lab
PYTHONPATH=$ROOT/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab \
  $PY -u $ROOT/pi_adr/scripts/train.py --task Isaac-Reorient-Cube-Leap \
  --headless --num_envs 1024 --max_iterations 5 --seed 42
```

## Metrics (frozen — `source/pi_adr/evaluation/metrics.py`)

Changing any definition means bumping `METRICS_VERSION` and re-running everything, because the
whole point is cross-run comparability:

- precision curve `P(d < eps)` for eps in {5 cm, 2 cm, 1 cm, 5 mm, 2 mm, 1 mm}
- mean / median / p90 / worst-decile terminal error
- mean and max joint velocity (rad/s), mean control effort (`||a||^2`), mean |a|
- settling time (s), overshoot (m)
- physics–performance map: `P(d<1cm | physics bin)` when DR is on

## Known issues / environment notes

1. **Moved install.** The venv's editable installs point at `/home/.../projects/I`, which no
   longer exists; the project now lives at `/data/home/.../projects/I`. A symlink
   `/home/amartyaranjan888/projects/I -> /data/home/amartyaranjan888/projects/I` restores them.
   Removing that symlink breaks every `import isaaclab`.
2. **URDF importer.** Isaac Sim 5.1's URDF importer needs `libxml2.so.2` and `libicu*.so.73`,
   absent on Ubuntu 26.04. `scripts/convert_asset.sh` shims them from the user's miniconda via
   `.libshim/`. Only asset conversion needs this — the cached USD makes training and evaluation
   shim-free.
3. **LEAP repo vs Isaac Lab 2.3.** Upstream targets Isaac Lab 2.1. Two incompatibilities:
   its `scripts/rl_games/train.py` imports `dump_pickle`, removed from `isaaclab.utils.io`
   (worked around by using `pi_adr/scripts/train.py` instead), and its `EventCfg` omits the
   `operation` param that `randomize_actuator_gains` now requires (one-line patch applied to
   `leap_hand_env_cfg.py`, set to `"abs"` = the old default).
4. **LEAP `pip install -e` fails** — its `setup.py` reads a `keywords` key absent from its
   `extension.toml`. Avoided entirely by putting `source/LEAP_Isaaclab` on `PYTHONPATH`.
5. Secrets live in `.env` (gitignored, chmod 600).
