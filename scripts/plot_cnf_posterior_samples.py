"""Generate CNF posterior samples figure for Appendix E.

5 rows (test observations) x 5 columns:
  Col 1: pressure p† with sensor locations
  Col 2: true log-permeability u†
  Cols 3-5: three independent CNF posterior samples

Uses the same contour_field / setup_style as plot_darcy_kl_results.py
so that visual properties (colorbars, fonts, contour lines) match
the main paper and gallery figures.

Usage:
    python scripts/plot_cnf_posterior_samples.py --gpu 0
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from cncv.models.conditional_nf import ConditionalNF
from cncv.utils.darcy import DarcyKLPrior, DarcyForwardModel
from cncv.utils.download import ensure_dataset

# Import shared plotting utilities from the main Darcy results script
_spec = importlib.util.spec_from_file_location(
    "plot_darcy_kl_results",
    os.path.join(os.path.dirname(__file__), "plot_darcy_kl_results.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
setup_style = _mod.setup_style
contour_field = _mod.contour_field
FIELD_CMAP = _mod.FIELD_CMAP


def load_cnf(cncv_ckpt_path, device):
    """Load just the CNF from the CNCV checkpoint."""
    ckpt = torch.load(cncv_ckpt_path, map_location=device, weights_only=False)
    y_mean = ckpt["y_mean"].cpu()
    y_std = ckpt["y_std"].cpu()

    cnf_ckpt = torch.load(ckpt["cnf_checkpoint"], map_location=device,
                          weights_only=False)
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

    print(f"CNF: epoch {cnf_ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params")
    return cnf, y_mean, y_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v4/cncv_epoch_320.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--vis_res", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="figs/darcy_kl")
    parser.add_argument("--obs_indices", type=int, nargs="+",
                        default=[0, 4, 8, 12, 16],
                        help="Test obs indices to plot (5 rows)")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CNF
    cnf, y_mean, y_std = load_cnf(args.cncv_checkpoint, device)

    # Load data
    print(f"Loading: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    z_all = torch.tensor(data["u_kl"], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    sensor_x = data["sensor_x"]
    sensor_y = data["sensor_y"]
    grid_size = int(data["grid_size"])

    # Test set
    test_start = z_all.shape[0] - args.n_test_obs
    z_test = z_all[test_start:]
    y_test = y_all[test_start:]
    y_test_n = (y_test - y_mean) / y_std

    # KL priors
    prior_vis = DarcyKLPrior(K=10, grid_size=args.vis_res,
                             alpha=0.0, s=1.1, sigma=1.0)
    prior_nat = DarcyKLPrior(K=10, grid_size=grid_size,
                             alpha=0.0, s=1.1, sigma=1.0)

    # Forward model for PDE solve
    print("Initializing PDE solver...")
    fwd = DarcyForwardModel(prior_nat, n_sensors=33, time_steps=5000)

    n_rows = len(args.obs_indices)
    n_samples = 3  # CNF posterior samples per obs
    n_cols = 2 + n_samples

    setup_style()

    # Match gallery figure sizing: 7.0 wide, 1.8 per row
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.0, n_rows * 1.8))
    fig.subplots_adjust(wspace=0.35, hspace=0.35,
                        left=0.04, right=0.97,
                        top=0.93, bottom=0.05)

    for row, obs_idx in enumerate(args.obs_indices):
        print(f"\nRow {row+1}/{n_rows}: test obs {obs_idx}")

        # True field and pressure
        z_true = z_test[obs_idx].numpy()
        u_true_nat = prior_nat.reconstruct(z_true[None])[0]
        u_true_vis = prior_vis.reconstruct(z_true[None])[0]

        print("  Solving forward PDE...")
        p_true = fwd.solve_pressure(u_true_nat)
        zoom_factor = args.vis_res / grid_size
        p_true_vis = zoom(p_true, zoom_factor, order=3)

        # Generate CNF samples
        y_obs = y_test_n[obs_idx].to(device)
        with torch.no_grad():
            z_samples = cnf.sample(y_obs.unsqueeze(0), n_samples).cpu().numpy()
        sample_fields = prior_vis.reconstruct(z_samples)  # (3, R, R)

        # Shared color range for permeability (truth + samples)
        all_u = np.concatenate([u_true_vis.ravel()] +
                               [sf.ravel() for sf in sample_fields])
        vmin_u, vmax_u = np.percentile(all_u, [1, 99])

        is_top = (row == 0)
        is_bottom = (row == n_rows - 1)

        # Col 0: pressure with sensors
        ax = axes[row, 0]
        title = r"Pressure $p^\dagger$" if is_top else None
        contour_field(ax, p_true_vis.T, title, cmap="GnBu_r")
        ax.scatter(sensor_x, sensor_y, s=18, facecolors="none",
                   edgecolors="k", linewidths=0.6, zorder=5)
        ax.set_ylabel(r"$s_2$", fontsize=8)
        if is_bottom:
            ax.set_xlabel(r"$s_1$", fontsize=8)

        # Col 1: true log-permeability
        ax = axes[row, 1]
        title = r"True $u^\dagger$" if is_top else None
        contour_field(ax, u_true_vis.T, title, vmin=vmin_u, vmax=vmax_u)
        ax.set_yticklabels([])
        if is_bottom:
            ax.set_xlabel(r"$s_1$", fontsize=8)

        # Cols 2-4: CNF posterior samples
        for s in range(n_samples):
            ax = axes[row, 2 + s]
            title = f"Sample {s+1}" if is_top else None
            contour_field(ax, sample_fields[s].T, title,
                          vmin=vmin_u, vmax=vmax_u)
            ax.set_yticklabels([])
            if is_bottom:
                ax.set_xlabel(r"$s_1$", fontsize=8)

    out_path = os.path.join(args.output_dir, "darcy_cnf_samples.pdf")
    for suffix in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir,
                                 f"darcy_cnf_samples.{suffix}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
