"""Publication figures for Darcy KL-space CNCV.

Figure 1 (figure*, 5 panels):
  Pressure p† with sensors, true u†, CNCV post. mean, posterior std, |error|

Figure 2 (figure*, 4 panels):
  Estimator std (MC), estimator std (CNCV), std ratio (MC/CNCV), per-obs VRF

Figure 3 (single-column figure):
  Per-component VRF bar chart

Usage:
    python scripts/plot_darcy_kl_results.py --gpu 0 \
        --cncv_checkpoint data/darcy/checkpoints_cncv_kl_v2_beskos/cncv_best.pth
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils.darcy import DarcyKLPrior, DarcyForwardModel, beskos_truth
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
                  n_levels=N_CONTOUR, cbar=True, cbar_kw=None):
    """Filled contour plot with overlaid contour lines (Beskos style)."""
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
        cb = add_cbar(cf, ax, format="%.2f", **(cbar_kw or {}))
        return cf, cb
    return cf, None


def load_models(cncv_ckpt_path, device):
    """Load CNCV ensemble and frozen CNF."""
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

    print(f"CNCV: epoch {ckpt['epoch']}, L={ckpt['n_ensemble']}, "
          f"depth={ckpt['cncv_depth']}, params={ckpt['n_params']:,}")
    print(f"CNF: epoch {cnf_ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params")

    return cnf, ensemble, y_mean, y_std, d_z, d_y


def generate_samples(cnf, ensemble, y_obs, n, device, chunk=2000):
    """Return z (n, d) and g (n, d) on CPU."""
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


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v2_beskos/cncv_best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_vrf_samples", type=int, default=5000)
    parser.add_argument("--n_post_samples", type=int, default=5000,
                        help="Samples for posterior mean/std")
    parser.add_argument("--vis_res", type=int, default=128,
                        help="Resolution for reconstructed fields")
    parser.add_argument("--vis_obs_idx", type=int, default=16)
    parser.add_argument("--use_beskos_truth", action="store_true",
                        help="Use Beskos truth field instead of random test obs for visualization")
    parser.add_argument("--output_dir", type=str, default="figs/darcy_kl_v2")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    cnf, ensemble, y_mean, y_std, d_z, d_y = load_models(
        args.cncv_checkpoint, device)

    # Load data
    print(f"\nLoading: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    z_all = torch.tensor(data["u_kl"], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    eigenvalues = data["eigenvalues"]
    sensor_x = data["sensor_x"]
    sensor_y = data["sensor_y"]
    grid_size = int(data["grid_size"])

    # Test set: use last n_test_obs samples from the dataset
    test_start = max(0, z_all.shape[0] - args.n_test_obs)
    z_test = z_all[test_start:]
    y_test = y_all[test_start:]
    y_test_n = (y_test - y_mean) / y_std
    print(f"Test: {z_test.shape[0]} obs (indices {test_start}..{z_all.shape[0]})")
    # Adjust n_test_obs to actual available
    args.n_test_obs = z_test.shape[0]

    # KL priors at vis and native resolution
    prior_vis = DarcyKLPrior(K=10, grid_size=args.vis_res,
                                alpha=0.0, s=1.1, sigma=1.0)
    prior_nat = DarcyKLPrior(K=10, grid_size=grid_size,
                                alpha=0.0, s=1.1, sigma=1.0)

    # Forward model for PDE solve (native grid)
    print("Initializing PDE solver...")
    fwd = DarcyForwardModel(prior_nat, n_sensors=33, time_steps=5000)

    # Choose visualization field
    if args.use_beskos_truth and "u_kl_truth" in data:
        print("Using Beskos truth field for visualization")
        z_true = data["u_kl_truth"]
        # Generate noisy obs for Beskos truth
        rng = np.random.default_rng(42)
        sigma_y = float(data["sigma_y"])
        y_truth_clean = data["y_truth_clean"]
        y_truth_obs = y_truth_clean + sigma_y * rng.standard_normal(
            y_truth_clean.shape).astype(np.float32)
        y_obs_vis_n = ((torch.tensor(y_truth_obs) - y_mean) / y_std).to(device)
    else:
        vis_idx = min(args.vis_obs_idx, args.n_test_obs - 1)
        z_true = z_test[vis_idx].numpy()
        y_obs_vis_n = y_test_n[vis_idx].to(device)

    u_true_nat = prior_nat.reconstruct(z_true[None])[0]
    u_true_vis = prior_vis.reconstruct(z_true[None])[0]

    print("Solving forward PDE for true permeability...")
    p_true = fwd.solve_pressure(u_true_nat)

    from scipy.ndimage import zoom
    zoom_factor = args.vis_res / grid_size
    p_true_vis = zoom(p_true, zoom_factor, order=3)

    # ================================================================
    # Part 1: Per-component VRF (all test obs)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Per-component VRF ({args.n_test_obs} obs, "
          f"{args.n_vrf_samples} samples each)")
    print(f"{'='*60}")

    all_comp_vrfs = np.zeros((args.n_test_obs, d_z))
    all_mean_vrfs = np.zeros(args.n_test_obs)

    for i in tqdm(range(args.n_test_obs), desc="VRF"):
        y_obs = y_test_n[i].to(device)
        z_s, g_s = generate_samples(cnf, ensemble, y_obs,
                                    args.n_vrf_samples, device)
        for k in range(d_z):
            vh = z_s[:, k].var().item()
            vhg = (z_s[:, k] - g_s[:, k]).var().item()
            all_comp_vrfs[i, k] = vhg / max(vh, 1e-12)
        all_mean_vrfs[i] = all_comp_vrfs[i].mean()

    print(f"Overall VRF: {all_mean_vrfs.mean():.4f} "
          f"+/- {all_mean_vrfs.std():.4f}")

    # ================================================================
    # Part 2: Posterior mean and std
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Posterior mean/std ({args.n_post_samples} samples)")
    print(f"{'='*60}")

    z_post, g_post = generate_samples(cnf, ensemble, y_obs_vis_n,
                                      args.n_post_samples, device)

    mc_mean_kl = z_post.mean(0).numpy()
    cv_mean_kl = (z_post - g_post).mean(0).numpy()

    mc_mean_vis = prior_vis.reconstruct(mc_mean_kl[None])[0]
    cv_mean_vis = prior_vis.reconstruct(cv_mean_kl[None])[0]

    print("Reconstructing posterior samples for pointwise std...")
    z_post_np = z_post.numpy()
    chunk_size = 500
    post_fields_sum = np.zeros((args.vis_res, args.vis_res), dtype=np.float64)
    post_fields_sq_sum = np.zeros_like(post_fields_sum)
    for ci in range(0, args.n_post_samples, chunk_size):
        chunk = z_post_np[ci:ci + chunk_size]
        fields = prior_vis.reconstruct(chunk)
        post_fields_sum += fields.sum(axis=0)
        post_fields_sq_sum += (fields ** 2).sum(axis=0)
    n = args.n_post_samples
    post_mean_vis = post_fields_sum / n
    post_std_vis = np.sqrt(post_fields_sq_sum / n - post_mean_vis ** 2)

    # ================================================================
    # Part 3: Multi-trial estimator variance
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Estimator std ({args.n_trials} trials x "
          f"{args.n_samples} samples)")
    print(f"{'='*60}")

    van_means, cv_means = [], []
    for _ in tqdm(range(args.n_trials), desc="Trials"):
        zt, gt = generate_samples(cnf, ensemble, y_obs_vis_n,
                                  args.n_samples, device)
        van_means.append(zt.mean(0))
        cv_means.append((zt - gt).mean(0))
    van_means = torch.stack(van_means).numpy()
    cv_means = torch.stack(cv_means).numpy()

    van_fields = prior_vis.reconstruct(van_means)
    cv_fields = prior_vis.reconstruct(cv_means)

    std_van = van_fields.std(axis=0)
    std_cv = cv_fields.std(axis=0)
    ratio_vis = std_van / np.clip(std_cv, 1e-12, None)

    van_nat = prior_nat.reconstruct(van_means)
    cv_nat = prior_nat.reconstruct(cv_means)
    std_van_nat = van_nat.std(axis=0)
    std_cv_nat = cv_nat.std(axis=0)
    ratio_nat = std_cv_nat / np.clip(std_van_nat, 1e-12, None)

    print(f"Ambient {grid_size}x{grid_size}: "
          f"std_van={std_van_nat.mean():.4f}, "
          f"std_cv={std_cv_nat.mean():.4f}, "
          f"ratio={ratio_nat.mean():.3f}")

    # ================================================================
    # FIGURE 1: Posterior summary (4 panels)
    # ================================================================
    setup_style()
    ext = (0, 1, 0, 1)

    fig1 = plt.figure(figsize=(7.2, 1.8))
    gs1 = GridSpec(1, 4, figure=fig1, wspace=0.55,
                   left=0.04, right=0.97, top=0.88, bottom=0.08)
    axes1 = [fig1.add_subplot(gs1[j]) for j in range(4)]

    contour_field(axes1[0], p_true_vis.T, r"(a) Pressure $p^\dagger$",
                  cmap="GnBu_r")
    axes1[0].scatter(sensor_x, sensor_y, s=18, facecolors="none",
                     edgecolors="k", linewidths=0.6, zorder=5)
    axes1[0].set_ylabel(r"$s_2$", fontsize=8)
    axes1[0].set_xlabel(r"$s_1$", fontsize=8)

    all_u = np.concatenate([u_true_vis.ravel(), cv_mean_vis.ravel()])
    vmin_u, vmax_u = np.percentile(all_u, [1, 99])

    contour_field(axes1[1], u_true_vis.T, r"(b) True $u^\dagger$",
                  vmin=vmin_u, vmax=vmax_u)
    axes1[1].set_xlabel(r"$s_1$", fontsize=8)
    axes1[1].set_yticklabels([])

    contour_field(axes1[2], cv_mean_vis.T, r"(c) CNCV mean",
                  vmin=vmin_u, vmax=vmax_u)
    axes1[2].set_xlabel(r"$s_1$", fontsize=8)
    axes1[2].set_yticklabels([])

    contour_field(axes1[3], post_std_vis.T, r"(d) Post.\ std",
                  cmap=STD_CMAP, vmin=0)
    axes1[3].set_xlabel(r"$s_1$", fontsize=8)
    axes1[3].set_yticklabels([])

    for suffix in ("pdf", "png"):
        fig1.savefig(os.path.join(args.output_dir,
                                  f"darcy_posterior.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {args.output_dir}/darcy_posterior.pdf")

    # ================================================================
    # FIGURE 2: Estimator std comparison (4 panels)
    # ================================================================
    fig2 = plt.figure(figsize=(7.2, 1.8))
    gs2 = GridSpec(1, 4, figure=fig2, wspace=0.55,
                   left=0.04, right=0.97, top=0.88, bottom=0.08)

    ax_a = fig2.add_subplot(gs2[0])
    ax_b = fig2.add_subplot(gs2[1])
    ax_c = fig2.add_subplot(gs2[2])
    ax_d = fig2.add_subplot(gs2[3])

    vmax_std = max(std_van.max(), std_cv.max())

    contour_field(ax_a, std_van.T,
                  r"(a) Std of $\hat\mu$ (MC)",
                  cmap=STD_CMAP, vmin=0, vmax=vmax_std)
    ax_a.set_ylabel(r"$s_2$", fontsize=8)
    ax_a.set_xlabel(r"$s_1$", fontsize=8)

    contour_field(ax_b, std_cv.T,
                  r"(b) Std of $\hat\mu$ (CNCV)",
                  cmap=STD_CMAP, vmin=0, vmax=vmax_std)
    ax_b.set_xlabel(r"$s_1$", fontsize=8)
    ax_b.set_yticklabels([])

    contour_field(ax_c, ratio_vis.T,
                  r"(c) Std ratio (MC\,/\,CNCV)",
                  cmap="RdYlGn", vmin=1.0, vmax=ratio_vis.max())
    ax_c.set_xlabel(r"$s_1$", fontsize=8)
    ax_c.set_yticklabels([])

    ax_d.bar(range(args.n_test_obs), all_mean_vrfs,
             color="#4C72B0", alpha=0.85, edgecolor="none", width=0.7)
    ax_d.axhline(1.0, color="C3", ls="--", lw=0.8, alpha=0.5)
    ax_d.axhline(all_mean_vrfs.mean(), color="C2", ls="-", lw=1.5,
                 alpha=0.8, label=f"Mean $= {all_mean_vrfs.mean():.2f}$")
    ax_d.fill_between(
        [-1, args.n_test_obs],
        all_mean_vrfs.mean() - all_mean_vrfs.std(),
        all_mean_vrfs.mean() + all_mean_vrfs.std(),
        color="C2", alpha=0.12)
    ax_d.set_xlabel(r"Test obs.\ index", fontsize=8)
    ax_d.set_title(r"(d) Per-obs VRF", fontsize=9, pad=4)
    ax_d.set_xlim(-0.5, args.n_test_obs - 0.5)
    ax_d.set_ylim(0, max(0.35, all_mean_vrfs.max() * 1.3))
    ax_d.set_aspect(args.n_test_obs / max(0.35, all_mean_vrfs.max() * 1.3))
    ax_d.legend(fontsize=6, loc="upper right")
    ax_d.tick_params(labelsize=7)

    for suffix in ("pdf", "png"):
        fig2.savefig(os.path.join(args.output_dir,
                                  f"darcy_estimator_std.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {args.output_dir}/darcy_estimator_std.pdf")

    # ================================================================
    # FIGURE 3: Per-component VRF
    # ================================================================
    fig3, ax_i = plt.subplots(1, 1, figsize=(3.4, 1.6))
    fig3.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.20)

    sort_idx = np.argsort(eigenvalues)[::-1]
    comp_vrf_sorted = all_comp_vrfs.mean(axis=0)[sort_idx]
    eig_sorted = eigenvalues[sort_idx]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, d_z))
    ax_i.bar(range(d_z), comp_vrf_sorted, color=colors,
             width=1.0, edgecolor="none", alpha=0.85)
    ax_i.axhline(1.0, color="C3", ls="--", lw=0.8, alpha=0.5)
    ax_i.set_xlabel("KL component (sorted by decreasing eigenvalue)",
                    fontsize=8)
    ax_i.set_ylabel("VRF", fontsize=8)
    ax_i.set_xlim(-1, d_z)
    ax_i.set_ylim(0, min(1.15, comp_vrf_sorted.max() * 1.15))
    ax_i.tick_params(labelsize=7)

    ax_i2 = ax_i.twinx()
    ax_i2.plot(range(d_z), eig_sorted, color="C1", lw=1.2, alpha=0.7)
    ax_i2.set_ylabel(r"Eigenvalue $\lambda$", color="C1", fontsize=8)
    ax_i2.tick_params(axis="y", colors="C1", labelsize=7)
    ax_i2.spines["right"].set_color("C1")

    for suffix in ("pdf", "png"):
        fig3.savefig(os.path.join(args.output_dir,
                                  f"darcy_component_vrf.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {args.output_dir}/darcy_component_vrf.pdf")

    # ================================================================
    # Paper numbers
    # ================================================================
    comp_vrf_mean = all_comp_vrfs.mean(axis=0)
    print(f"\n{'='*60}")
    print("Numbers for paper:")
    print(f"{'='*60}")
    print(f"  VRF (KL): {all_mean_vrfs.mean():.2f} +/- "
          f"{all_mean_vrfs.std():.2f}")
    print(f"  Reduction: {(1 - all_mean_vrfs.mean()) * 100:.0f}%")
    print(f"  Per-comp range: [{comp_vrf_mean.min():.3f}, "
          f"{comp_vrf_mean.max():.3f}]")
    print(f"  Per-comp median: {np.median(comp_vrf_mean):.3f}")
    print(f"  Ambient std ratio: {ratio_nat.mean():.3f}")
    print(f"  ESS multiplier: {1.0 / all_mean_vrfs.mean():.1f}x")
    print(f"  Std ratio (MC/CNCV) range: [{ratio_vis.min():.2f}, "
          f"{ratio_vis.max():.2f}]")
    print(f"  Std ratio (MC/CNCV) mean: {ratio_vis.mean():.2f}")

    np.savez(os.path.join(args.output_dir, "darcy_paper_results.npz"),
             all_comp_vrfs=all_comp_vrfs, all_mean_vrfs=all_mean_vrfs,
             std_van_nat=std_van_nat, std_cv_nat=std_cv_nat,
             ratio_nat=ratio_nat, u_true_vis=u_true_vis,
             p_true_vis=p_true_vis,
             eigenvalues=eigenvalues, sensor_x=sensor_x, sensor_y=sensor_y,
             post_std_vis=post_std_vis)


if __name__ == "__main__":
    main()
