#!/usr/bin/env bash
# URDF->USD conversion needs libs Ubuntu 26.04 lacks; shim them in just for this step.
set -euo pipefail
ROOT=/data/home/amartyaranjan888/projects/I/isaac_env
export LD_LIBRARY_PATH="$ROOT/pi_adr/.libshim:/data/home/amartyaranjan888/projects/I/isaac_env/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.asset_converter-5.0.17+107.3.1.lx64.r.cp311.u353/asset_converter_native_bindings/libs:${LD_LIBRARY_PATH:-}"
exec "$ROOT/env_isaaclab/bin/python" "$ROOT/pi_adr/scripts/convert_asset.py" "$@"
