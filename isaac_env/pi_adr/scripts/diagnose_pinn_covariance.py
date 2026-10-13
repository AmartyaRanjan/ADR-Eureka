"""Diagnose the PINN uncertainty negative result: is the weak U_PINN<->error
correlation (results/pinn_observer.json: rank corr +0.11) caused by the
ensemble's DIAGONAL-ONLY covariance estimate hiding a real correlation between
parameters that are not independently identifiable from a single torque regime?

Hypothesis (PROJECT_STATUS_AND_ROADMAP.md, Priority 3, step 1): at a given
torque, mass (m) and actuator effectiveness (alpha) trade off — a heavier link
driven by a stronger-than-believed actuator can produce the same acceleration
as a lighter link with a weaker actuator. If so, the 5 ensemble members should
disagree on m and alpha in a correlated (likely anti-correlated) way even
though each member's own OUTPUT is bounded and looks reasonable in isolation.

This is pure analysis of already-saved artifacts (data/pinn_observer.pt,
data/pinn_val.pt) — no simulator, no new training. Runs on CPU/GPU via plain
torch in seconds.

    env_isaaclab/bin/python pi_adr/scripts/diagnose_pinn_covariance.py
"""

from __future__ import annotations

import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

import sys  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "source"))
from pi_adr.pinn.dynamics import XI_NAMES  # noqa: E402
from pi_adr.pinn.model import ObserverEnsemble  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='')
    tag = ('_' + ap.parse_args().tag) if ap.parse_args().tag else ''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ObserverEnsemble(num_members=5).to(device)
    state = torch.load(os.path.join(DATA_DIR, f"pinn_observer{tag}.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    blob = torch.load(os.path.join(DATA_DIR, f"pinn_val{tag}.pt"), map_location=device)
    window, valid = blob["window"], blob["valid"]

    win_len = 64
    n = window.shape[0]
    weights = valid.float() + 1e-6
    picks = torch.multinomial(weights, win_len, replacement=True)
    w = torch.gather(window, 1, picks.unsqueeze(-1).expand(-1, -1, window.shape[-1]))

    with torch.no_grad():
        member_preds = torch.stack([m(w)[0] for m in model.members])  # (K, N, 8)

    K, N, D = member_preds.shape
    mean = member_preds.mean(dim=0)
    centered = member_preds - mean  # (K, N, 8)

    # Full (off-diagonal) empirical cross-covariance per environment, averaged
    # over environments -- this is what the diagonal-only estimate in
    # ObserverEnsemble.forward() currently discards.
    cov = torch.einsum("kni,knj->nij", centered, centered) / (K - 1)  # (N, 8, 8)
    mean_cov = cov.mean(dim=0)  # (8, 8)
    std = mean_cov.diagonal().clamp_min(1e-12).sqrt()
    corr = mean_cov / (std.unsqueeze(0) * std.unsqueeze(1))

    print("\n=== Mean cross-member correlation matrix (xi_hat), off-diagonal = the thing the")
    print("    diagonal-only uncertainty estimate in ObserverEnsemble.forward() discards ===\n")
    header = "        " + "".join(f"{n:>8}" for n in XI_NAMES)
    print(header)
    for i, name in enumerate(XI_NAMES):
        row = "".join(f"{corr[i, j].item():8.3f}" for j in range(D))
        print(f"{name:<8}{row}")

    pairs_of_interest = [("m1", "a1"), ("m2", "a2"), ("m1", "m2"), ("a1", "a2")]
    print("\n=== Pairs of interest (identifiability degeneracy hypothesis) ===")
    results = {}
    for p, q in pairs_of_interest:
        i, j = XI_NAMES.index(p), XI_NAMES.index(q)
        c = corr[i, j].item()
        results[f"{p}_{q}"] = c
        flag = "  <-- notable (|r|>0.3)" if abs(c) > 0.3 else ""
        print(f"  corr({p}, {q}) = {c:+.3f}{flag}")

    diag_var = mean_cov.diagonal()
    total_var_if_diag_only = diag_var.sum().item()
    eigvals = torch.linalg.eigvalsh(mean_cov)
    total_var_full = eigvals.clamp_min(0).sum().item()
    print(f"\nsum(diagonal variances)          = {total_var_if_diag_only:.6f}")
    print(f"sum(eigenvalues, full covariance) = {total_var_full:.6f}  (equal by trace invariance;")
    print("  the point is not total variance but how it is DISTRIBUTED across directions —")
    print("  see the dominant eigenvector below.)")

    top_eigval, top_eigvec = torch.linalg.eigh(mean_cov)
    dom_idx = torch.argmax(top_eigval)
    dom_vec = top_eigvec[:, dom_idx]
    print(f"\nDominant uncertainty direction (eigenvalue {top_eigval[dom_idx].item():.6f}):")
    for name, coef in sorted(zip(XI_NAMES, dom_vec.tolist()), key=lambda t: -abs(t[1])):
        print(f"  {name:<6} {coef:+.3f}")

    out = {
        "correlations_of_interest": results,
        "full_correlation_matrix": {XI_NAMES[i]: {XI_NAMES[j]: corr[i, j].item() for j in range(D)} for i in range(D)},
        "dominant_eigenvalue": top_eigval[dom_idx].item(),
        "dominant_eigenvector": dict(zip(XI_NAMES, dom_vec.tolist())),
        "sum_diagonal_variance": total_var_if_diag_only,
    }
    path = os.path.join(RESULTS_DIR, f"pinn_covariance_diagnosis{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
