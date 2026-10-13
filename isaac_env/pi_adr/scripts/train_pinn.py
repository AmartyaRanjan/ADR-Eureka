"""Train and validate the physics observer (PINN) on the dynamics dataset.

Pure torch — no simulator, so this does not need the Isaac Sim app.

Reports what the roadmap asks for: acceleration prediction error and recovered
parameter error, on held-out *physics configurations*, with interpolation and
extrapolation kept separate. Also reports whether the ensemble's uncertainty is
informative, since U_PINN only earns its place in the adaptive-DR priority
score if it correlates with actual error.

    env_isaaclab/bin/python pi_adr/scripts/train_pinn.py --epochs 300
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from pi_adr.pinn.dynamics import XI_NAMES, forward_dynamics
from pi_adr.pinn.model import XI_BOUNDS, ObserverEnsemble, normalize_xi, observer_losses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


TAG = ""


def load_split(name: str, device: torch.device):
    path = os.path.join(DATA_DIR, f"pinn_{name}{TAG}.pt")
    blob = torch.load(path, map_location="cpu")
    return blob["window"].to(device), blob["valid"].to(device), blob["xi"].to(device)


def sample_windows(window, valid, xi, batch: int, win_len: int, generator=None):
    """Draw a batch of environments and, within each, a set of usable transitions."""
    n = window.shape[0]
    idx = torch.randint(0, n, (batch,), device=window.device, generator=generator)
    w, v, x = window[idx], valid[idx], xi[idx]

    # Sample transition indices with probability proportional to validity so a
    # window never contains velocity-limited (physically unexplainable) samples.
    weights = v.float() + 1e-6
    picks = torch.multinomial(weights, win_len, replacement=True)
    w = torch.gather(w, 1, picks.unsqueeze(-1).expand(-1, -1, w.shape[-1]))
    return w, x


@torch.no_grad()
def evaluate(model, window, valid, xi, win_len: int, device) -> dict:
    model.eval()
    w, x = sample_windows(window, valid, xi, batch=window.shape[0], win_len=win_len)
    xi_hat, total_var, epistemic = model(w)

    bounds = XI_BOUNDS.to(device)
    param_err = (normalize_xi(xi_hat, bounds) - normalize_xi(x, bounds)).abs()
    abs_err = (xi_hat - x).abs()

    q, qd, tau, qdd = w.split(2, dim=-1)
    xi_b = xi_hat.unsqueeze(1).expand(-1, w.shape[1], -1)
    qdd_pred = forward_dynamics(q, qd, tau, xi_b)
    acc_rmse = (qdd_pred - qdd).pow(2).mean().sqrt()

    # Does predicted uncertainty track actual error? Spearman-style rank check.
    unc = total_var.mean(dim=-1)
    err = param_err.mean(dim=-1)
    rank_corr = torch.corrcoef(torch.stack([unc.argsort().argsort().float(), err.argsort().argsort().float()]))[0, 1]

    model.train()
    return {
        "acc_rmse_rad_s2": acc_rmse.item(),
        "param_err_normalised_mean": param_err.mean().item(),
        "param_err_abs": {n: abs_err[:, i].mean().item() for i, n in enumerate(XI_NAMES)},
        "uncertainty_error_rank_corr": rank_corr.item(),
        "mean_epistemic_var": epistemic.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--win_len", type=int, default=64)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_data", type=float, default=1.0)
    parser.add_argument("--w_phys", type=float, default=0.01)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tag", type=str, default="", help="load pinn_<split>_<tag>.pt, save pinn_observer_<tag>.pt")
    args = parser.parse_args()
    global TAG
    TAG = f"_{args.tag}" if args.tag else ""

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_w, train_v, train_x = load_split("train", device)
    val_w, val_v, val_x = load_split("val", device)
    extrap_w, extrap_v, extrap_x = load_split("extrap", device)
    print(f"train envs={train_w.shape[0]}  val envs={val_w.shape[0]}  extrap envs={extrap_w.shape[0]}")

    model = ObserverEnsemble(num_members=args.members).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bounds = XI_BOUNDS.to(device)

    for epoch in range(1, args.epochs + 1):
        # Each ensemble member sees its own bootstrap-style batch, otherwise the
        # members converge to the same function and the spread stops being a
        # meaningful uncertainty signal.
        total = 0.0
        opt.zero_grad()
        for member in model.members:
            wm, xm = sample_windows(train_w, train_v, train_x, args.batch, args.win_len)
            xi_hat, log_var = member(wm)
            losses = observer_losses(xi_hat, log_var, xm, wm, bounds)
            loss = args.w_data * losses["data"] + args.w_phys * losses["phys"] + args.w_sup * losses["sup"]
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            m = evaluate(model, val_w, val_v, val_x, args.win_len, device)
            print(
                f"epoch {epoch:4d}  loss={total / args.members:9.4f}  "
                f"val acc_rmse={m['acc_rmse_rad_s2']:7.3f}  "
                f"val param_err={m['param_err_normalised_mean']:.4f}"
            )

    val_metrics = evaluate(model, val_w, val_v, val_x, args.win_len, device)
    extrap_metrics = evaluate(model, extrap_w, extrap_v, extrap_x, args.win_len, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "data", f"pinn_observer{TAG}.pt"))
    out = {
        "config": vars(args),
        "validation_interpolation": val_metrics,
        "validation_extrapolation": extrap_metrics,
    }
    with open(os.path.join(RESULTS_DIR, f"pinn_observer{TAG}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n================ PINN OBSERVER ================")
    for label, m in (("interpolation", val_metrics), ("extrapolation", extrap_metrics)):
        print(f"  [{label}]")
        print(f"    acceleration RMSE      {m['acc_rmse_rad_s2']:.4f} rad/s^2")
        print(f"    parameter error (norm) {m['param_err_normalised_mean']:.4f}")
        print(f"    uncertainty/error rank corr {m['uncertainty_error_rank_corr']:+.3f}")
        print("    per-parameter abs error: " + "  ".join(
            f"{k}={v:.4f}" for k, v in m["param_err_abs"].items()))
    print("==============================================\n")


if __name__ == "__main__":
    main()
