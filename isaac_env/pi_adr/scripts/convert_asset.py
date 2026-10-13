"""Convert the 2-DOF arm URDF to USD (run once; the USD is then cached)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="reconvert even if the USD exists")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pi_adr.tasks.two_dof_reach.two_dof_reach_env_cfg import build_arm_usd  # noqa: E402

usd_path = build_arm_usd(force=args_cli.force)
print(f"[pi_adr] USD written to: {usd_path}")

simulation_app.close()
