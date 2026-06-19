"""Generate Darcy posterior gallery — one figure per observation.

Produces 20 figures (obs_00.pdf .. obs_19.pdf) so the user can pick
which to include in the paper.

Usage:
    python scripts/generate_darcy_gallery.py --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils.darcy import DarcyKLPrior, DarcyForwardModel
from cncv.utils.download import ensure_dataset


# ── Style ───────────────────────────────────────────────────────────
FIELD_CMAP = "YlGnBu_r"
STD_CMAP = "inferno"
N_CONTOUR = 15
CONTOUR_LW = 0.3
CONTOUR_CLR = "k"
CONTOUR_ALPHA = 0.35


def setup_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


def add_cbar(mappable, ax, pad=0.04, width="4%", **kw):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    return plt.colorbar(mappable, cax=cax, **kw)


def contour_field(ax, field, title, extent=(0, 1, 0, 1),
                  cmap=FIELD_CMAP, vmin=None, vmax=None,
                  n_levels=N_CONTOUR, cbar=True):
    ny, nx = field.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(
        vmin if vmin is not None else field.min(),
        vmax if vmax is not None else field.max(),
        n_levels + 1)
    cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap, extend="both")
    ax.contour(X, Y, field, levels=levels, colors=CONTOUR_CLR,
               linewidths=CONTOUR_LW, alpha=CONTOUR_ALPHA)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    if cbar:
        cb = add_cbar(cf, ax, format="%.2f")
        return cf, cb
    return cf, None


def load_models(cncv_ckpt_path, device):
    ckpt = torch.load(cncv_ckpt_path, map_location=device, weights_only=False)
    d_z, d_y = ckpt["d_z"], ckpt["d_y"]
    y_mean = ckpt["y_mean"].cpu()
    y_std = ckpt["y_std"].cpu()

    ensemble = create_shared_ensemble(
        n_in=d_z, n_cond=d_y,
        n_hidden=ckpt["cncv_hidden"],
        n_ensemble=ckpt["n_ensemble"],
        depth=ckpt["cncv_depth"],
        n_mlp_layers=ckpt["cncv_mlp_layers"],
        seed=12)
    ensemble = SharedEnsembleCVWrapper(ensemble).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

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

    return cnf, ensemble, y_mean, y_std, d_z, d_y


def generate_samples(cnf, ensemble, y_obs, n, device, chunk=2000):
    zs, gs = [], []
    done = 0
    while done < n:
        bs = min(chunk, n - done)
        with torch.no_grad():
            z = cnf.sample(y_obs.unsqueeze(0), bs)
        yr = y_obs.unsqueeze(0).expand(bs, -1)
        s = cnf.score(z, yr)
        with torch.no_grad():
            g, _ = ensemble(z, yr, s)
        zs.append(z.cpu()); gs.append(g.T.cpu())
        done += bs
    return torch.cat(zs, 0)[:n], torch.cat(gs, 0)[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v2b/cncv_best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--n_obs", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--vis_res", type=int, default=128)
    parser.add_argument("--output_dir", type=str,
                        default="figs/darcy_gallery_v2b")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    cnf, ensemble, y_mean, y_std, d_z, d_y = load_models(
        args.cncv_checkpoint, device)

    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    z_all = torch.tensor(data["u_kl"], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    sensor_x = data["sensor_x"]
    sensor_y = data["sensor_y"]
    grid_size = int(data["grid_size"])

    test_start = max(0, z_all.shape[0] - args.n_obs)
    z_test = z_all[test_start:]
    y_test = y_all[test_start:]
    y_test_n = (y_test - y_mean) / y_std
    n_obs = z_test.shape[0]

    prior_vis = DarcyKLPrior(K=10, grid_size=args.vis_res,
                                alpha=0.0, s=1.1, sigma=1.0)
    prior_nat = DarcyKLPrior(K=10, grid_size=grid_size,
                                alpha=0.0, s=1.1, sigma=1.0)
    fwd = DarcyForwardModel(prior_nat, n_sensors=33, time_steps=5000)

    setup_style()
    zoom_factor = args.vis_res / grid_size

    # Build list of (label, z_true, y_obs_normalized) tuples
    obs_list = []

    # Add Beskos truth field if available
    if "u_kl_truth" in data:
        z_beskos = data["u_kl_truth"]
        sigma_y = float(data["sigma_y"])
        y_truth_clean = data["y_truth_clean"]
        rng = np.random.default_rng(42)
        y_truth_obs = y_truth_clean + sigma_y * rng.standard_normal(
            y_truth_clean.shape).astype(np.float32)
        y_beskos_n = ((torch.tensor(y_truth_obs) - y_mean) / y_std).to(device)
        obs_list.append(("beskos_truth", z_beskos, y_beskos_n))
        print("Added Beskos truth field as first observation")

    # Add test observations
    for i in range(n_obs):
        obs_list.append((f"obs_{i:02d}", z_test[i].numpy(),
                         y_test_n[i].to(device)))

    for idx, (label, z_true, y_obs_n) in enumerate(obs_list):
        print(f"\n=== {label} ({idx+1}/{len(obs_list)}) ===")

        # Reconstruct truth
        u_true_vis = prior_vis.reconstruct(z_true[None])[0]

        # Pressure field
        u_true_nat = prior_nat.reconstruct(z_true[None])[0]
        p_true = fwd.solve_pressure(u_true_nat)
        p_true_vis = zoom(p_true, zoom_factor, order=3)

        # Generate posterior samples
        z_post, g_post = generate_samples(cnf, ensemble, y_obs_n,
                                          args.n_samples, device)
        cv_mean_kl = (z_post - g_post).mean(0).numpy()
        cv_mean_vis = prior_vis.reconstruct(cv_mean_kl[None])[0]

        # Posterior std
        z_np = z_post.numpy()
        fields_sum = np.zeros((args.vis_res, args.vis_res), dtype=np.float64)
        fields_sq = np.zeros_like(fields_sum)
        for ci in range(0, args.n_samples, 500):
            chunk = z_np[ci:ci + 500]
            fields = prior_vis.reconstruct(chunk)
            fields_sum += fields.sum(0)
            fields_sq += (fields ** 2).sum(0)
        post_mean = fields_sum / args.n_samples
        post_std = np.sqrt(fields_sq / args.n_samples - post_mean ** 2)

        # VRF
        vrfs = []
        for k in range(d_z):
            vh = z_post[:, k].var().item()
            vhg = (z_post[:, k] - g_post[:, k]).var().item()
            vrfs.append(vhg / max(vh, 1e-12))
        vrf_mean = np.mean(vrfs)

        # Plot 4-panel figure
        fig = plt.figure(figsize=(7.2, 1.8))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 4, figure=fig, wspace=0.55,
                      left=0.04, right=0.97, top=0.88, bottom=0.08)
        axes = [fig.add_subplot(gs[j]) for j in range(4)]

        contour_field(axes[0], p_true_vis.T, r"(a) Pressure $p^\dagger$",
                      cmap="GnBu_r")
        axes[0].scatter(sensor_x, sensor_y, s=18, facecolors="none",
                        edgecolors="k", linewidths=0.6, zorder=5)

        contour_field(axes[1], u_true_vis.T, r"(b) True $u^\dagger$")

        contour_field(axes[2], cv_mean_vis.T, r"(c) CNCV mean",
                      vmin=u_true_vis.min(), vmax=u_true_vis.max())

        contour_field(axes[3], post_std.T, r"(d) Posterior std",
                      cmap=STD_CMAP)

        out = os.path.join(args.output_dir, f"{label}.pdf")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  VRF={vrf_mean:.3f}, saved {out}")

    print(f"\nDone! {len(obs_list)} figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
