"""Exit 0 iff an evaluation JSON is physically plausible. A failed simulator start-up (PhysX direct-GPU API not initialised) can
silently write a result with zero joint motion; file existence alone is not proof the evaluation ran."""
import json, sys
d = json.load(open(sys.argv[1]))
ok = d.get("mean_joint_velocity_rad_s", 1.0) > 0.05 and d.get("mean_terminal_error_m", 0.0) < 1.5
sys.exit(0 if ok else 1)
