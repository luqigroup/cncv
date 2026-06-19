"""Evaluate trained CNCV on the Helmholtz FWI joint dataset.

Adapts eval_cncv_darcy_kl.py for the Helmholtz data:
  - Uses key "u_kl" (alias for "z") from the npz file.
  - Computes per-component and per-observation VRF.
  - Saves a 2-panel figure: per-component VRF + per-observation VRF.

Usage:
  python scripts/eval_helmholtz_cncv.py --gpu 0 \
      --cncv_checkpoint data/helmholtz/checkpoints_cncv/cncv_best.pth \
      --data_path data/helmholtz/helmholtz_K10_N131072_grid32.npz
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils.download import ensure_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/helmholtz/checkpoints_cncv_131k/cncv_best.pth")
    parser.add_argument("--data_path", type=str,
                        default="data/helmholtz/helmholtz_K10_N131072_grid32.npz")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_train", type=int, default=120000)
    parser.add_argument("--num_val", type=int, default=11072)
    parser.add_argument("--output_dir", type=str, default="figs/helmholtz")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading CNCV: {args.cncv_checkpoint}")
    cncv_ckpt = torch.load(args.cncv_checkpoint, map_location=device, weights_only=False)
    d_z = cncv_ckpt["d_z"]
    d_y = cncv_ckpt["d_y"]
    y_mean = cncv_ckpt["y_mean"].cpu()
    y_std = cncv_ckpt["y_std"].cpu()
    n_ensemble = cncv_ckpt["n_ensemble"]
    cncv_hidden = cncv_ckpt["cncv_hidden"]
    cncv_depth = cncv_ckpt["cncv_depth"]
    cncv_mlp_layers = cncv_ckpt["cncv_mlp_layers"]
    print(f"  CNCV: epoch {cncv_ckpt['epoch']}, L={n_ensemble}, depth={cncv_depth}, "
          f"hidden={cncv_hidden}, mlp={cncv_mlp_layers}")

    ensemble = create_shared_ensemble(
        n_in=d_z, n_cond=d_y, n_hidden=cncv_hidden,
        n_ensemble=n_ensemble, depth=cncv_depth,
        n_mlp_layers=cncv_mlp_layers, seed=12,
    )
    ensemble = SharedEnsembleCVWrapper(ensemble).to(device)
    ensemble.load_state_dict(cncv_ckpt["ensemble_state_dict"])
    ensemble.eval()

    cnf_checkpoint = cncv_ckpt["cnf_checkpoint"]
    print(f"\nLoading CNF: {cnf_checkpoint}")
    cnf_ckpt = torch.load(cnf_checkpoint, map_location=device, weights_only=False)
    cnf = ConditionalNF(
        n_in=cnf_ckpt["d_z"], n_cond=cnf_ckpt["d_y"],
        n_hidden=cnf_ckpt["cnf_hidden"],
        n_flow_layers=cnf_ckpt["cnf_flow_layers"],
        depth=cnf_ckpt.get("cnf_depth", None),
        n_mlp_layers=cnf_ckpt["cnf_mlp_layers"],
    ).to(device)
    cnf.load_state_dict(cnf_ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    print(f"\nLoading data: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path, allow_pickle=True)
    z_key = "u_kl" if "u_kl" in data.files else "z"
    z_all = torch.tensor(data[z_key], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    eigenvalues = data["eigenvalues"]

    test_start = args.num_train + args.num_val
    n_available = z_all.shape[0] - test_start
    if n_available < args.n_test_obs:
        test_start = z_all.shape[0] - args.n_test_obs
    z_test = z_all[test_start:test_start + args.n_test_obs]
    y_test = y_all[test_start:test_start + args.n_test_obs]
    y_test_n = (y_test - y_mean) / y_std
    print(f"  Test: {args.n_test_obs} obs from indices {test_start}..")

    all_vrfs = []
    all_comp_vrfs = []
    all_corrs = []

    for i in tqdm(range(args.n_test_obs), desc="Test obs"):
        y_obs = y_test_n[i].to(device)
        with torch.no_grad():
            z_samples = cnf.sample(y_obs.unsqueeze(0), args.num_samples)
        y_rep = y_obs.unsqueeze(0).expand(args.num_samples, -1)
        h = z_samples
        score = cnf.score(z_samples, y_rep)
        with torch.no_grad():
            g_combined, _ = ensemble(z_samples, y_rep, score)
        g = g_combined.T

        comp_vrfs, comp_corrs = [], []
        for k in range(d_z):
            var_h = h[:, k].var().item()
            var_hg = (h[:, k] - g[:, k]).var().item()
            vrf_k = var_hg / max(var_h, 1e-12)
            corr_k = torch.corrcoef(torch.stack([h[:, k], g[:, k]]))[0, 1].item()
            comp_vrfs.append(vrf_k)
            comp_corrs.append(corr_k)

        all_vrfs.append(np.mean(comp_vrfs))
        all_comp_vrfs.append(comp_vrfs)
        all_corrs.append(np.mean(comp_corrs))
        if i < 5:
            print(f"  Obs {i}: VRF={all_vrfs[-1]:.4f}, corr={all_corrs[-1]:.4f}")

    all_vrfs = np.array(all_vrfs)
    all_comp_vrfs = np.array(all_comp_vrfs)
    all_corrs = np.array(all_corrs)

    print(f"\n=== Helmholtz CNCV results ===")
    print(f"  Per-obs VRF:    {all_vrfs.mean():.4f} +/- {all_vrfs.std():.4f}")
    print(f"  Per-comp VRF:   median={np.median(all_comp_vrfs.mean(0)):.4f}, "
          f"mean={all_comp_vrfs.mean():.4f}")
    print(f"  Mean correlation: {all_corrs.mean():.4f}")

    # Save
    np.savez(os.path.join(args.output_dir, "helmholtz_eval_results.npz"),
             all_vrfs=all_vrfs, all_comp_vrfs=all_comp_vrfs,
             all_corrs=all_corrs, eigenvalues=eigenvalues)

    # Figure
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "figure.dpi": 150, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    ax = axes[0]
    comp_mean = all_comp_vrfs.mean(0)
    sort_idx = np.argsort(eigenvalues)[::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, d_z))
    ax.bar(range(d_z), comp_mean[sort_idx], color=colors, width=1.0,
           edgecolor="none", alpha=0.85)
    ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.6)
    ax.set_xlabel("KL component (sorted by eigenvalue)")
    ax.set_ylabel("Per-component VRF")
    ax.set_title(f"(a) Per-component VRF (Helmholtz, d={d_z})")
    ax2 = ax.twinx()
    ax2.plot(range(d_z), eigenvalues[sort_idx], color="C1", lw=1.2, alpha=0.7)
    ax2.set_ylabel("Eigenvalue", color="C1")

    ax = axes[1]
    ax.bar(range(args.n_test_obs), all_vrfs, color="#4C72B0", alpha=0.85)
    ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(all_vrfs.mean(), color="green", ls="-", lw=1.5, alpha=0.7,
               label=f"Mean={all_vrfs.mean():.3f}")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Mean VRF across components")
    ax.set_title(f"(b) Per-observation VRF")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(args.output_dir, "helmholtz_vrf.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
