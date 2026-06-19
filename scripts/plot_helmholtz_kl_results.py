"""Publication figures for Helmholtz FWI KL-space CNCV.

Uses the trained N=131k Helmholtz CNF + CNCV checkpoints and produces the
following figures:

  - helmholtz_posterior.pdf            (4 panels: true v, post mean, post std, ...)
  - helmholtz_estimator_std.pdf        (4 panels: MC std, CNCV std, ratio, per-obs VRF)
  - helmholtz_component_vrf.pdf        (per-KL component VRF + eigenvalue overlay)
  - helmholtz_sample_efficiency.pdf    (VRF + MSE vs N, log-log)
  - helmholtz_kl_scatter.pdf           (3x2 KL scatter: prior + post)
  - helmholtz_loss_curves.pdf          (CNF + CNCV training/val curves)
  - helmholtz_posterior_obs{i}.pdf     (gallery: 4 panels per held-out obs)

Usage:
    python scripts/plot_helmholtz_kl_results.py --gpu 0
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
from scipy.ndimage import zoom

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils.velocity_kl_prior import VelocityKLPrior
from cncv.utils.helmholtz import make_src_rec
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
        "legend.fontsize": 7,
    })


def add_cbar(mappable, ax, pad=0.04, width="4%", **kw):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    return plt.colorbar(mappable, cax=cax, **kw)


def _auto_fmt(vmax_abs):
    """Pick a colorbar format string based on the data magnitude."""
    if vmax_abs == 0:
        return "%.2f"
    if vmax_abs < 1e-2:
        return "%.1e"
    if vmax_abs < 0.1:
        return "%.3f"
    return "%.2f"


def contour_field(ax, field, title, extent=(0, 1, 0, 1),
                  cmap=FIELD_CMAP, vmin=None, vmax=None,
                  n_levels=N_CONTOUR, cbar=True, cbar_kw=None,
                  cbar_fmt=None):
    """Filled contour with overlaid line contours."""
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
        if cbar_fmt is None:
            cbar_fmt = _auto_fmt(max(abs(levels[0]), abs(levels[-1])))
        cb = add_cbar(cf, ax, format=cbar_fmt, **(cbar_kw or {}))
        cb.ax.tick_params(labelsize=6)
        return cf, cb
    return cf, None


def velocity_perturbation(prior, z):
    """Reconstruct velocity perturbation v - v_bg from KL coefficients.

    Args:
        prior: VelocityKLPrior.
        z: KL coefficients, shape (..., d).
    Returns:
        delta_v: velocity perturbation, shape (..., grid_size, grid_size).
    """
    m = prior.reconstruct(z)
    v = prior.to_velocity(m)
    return v - prior.v_background


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

    return cnf, ensemble, y_mean, y_std, d_z, d_y, ckpt, cnf_ckpt


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/helmholtz/checkpoints_cncv_131k/cncv_best.pth")
    parser.add_argument("--cnf_checkpoint_full", type=str,
                        default="data/helmholtz/checkpoints_cnf_131k/cnf_epoch_79.pth",
                        help="Full-training CNF checkpoint for loss curves")
    parser.add_argument("--cncv_checkpoint_full", type=str,
                        default="data/helmholtz/checkpoints_cncv_131k/cncv_epoch_149.pth",
                        help="Full-training CNCV checkpoint for loss curves")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/helmholtz/helmholtz_K10_N131072_grid32.npz")
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_vrf_samples", type=int, default=5000)
    parser.add_argument("--n_post_samples", type=int, default=5000)
    parser.add_argument("--vis_res", type=int, default=128,
                        help="Resolution for reconstructed fields")
    parser.add_argument("--vis_obs_idx", type=int, default=0,
                        help="Test obs idx for main posterior figure")
    parser.add_argument("--gallery_obs", type=int, nargs="+",
                        default=[0, 4, 7, 10, 12, 14, 18],
                        help="Test obs indices for posterior_obs gallery")
    parser.add_argument("--scatter_obs", type=int, nargs="+",
                        default=[0, 4, 12])
    parser.add_argument("--sample_sizes", type=int, nargs="+",
                        default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument("--n_eff_obs", type=int, default=5,
                        help="Test obs to average over in sample-efficiency study")
    parser.add_argument("--n_eff_trials", type=int, default=200,
                        help="Bootstrap trials per sample size")
    parser.add_argument("--num_train", type=int, default=120000)
    parser.add_argument("--num_val", type=int, default=11072)
    parser.add_argument("--output_dir", type=str, default="figs/helmholtz")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────
    cnf, ensemble, y_mean, y_std, d_z, d_y, cncv_ckpt, cnf_ckpt = load_models(
        args.cncv_checkpoint, device)

    # ── Load data ──────────────────────────────────────────────────
    print(f"\nLoading: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path, allow_pickle=True)
    z_key = "u_kl" if "u_kl" in data.files else "z"
    z_all = torch.tensor(data[z_key], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    eigenvalues = data["eigenvalues"]
    grid_size = int(data["grid_size"])
    K = int(data["K"])
    v_bg = float(data["v_background"])
    sigma_m = float(data["sigma_m"])

    # KL prior at native and visualization resolutions.
    prior_nat = VelocityKLPrior(K=K, grid_size=grid_size, v_background=v_bg,
                                sigma_m=sigma_m, alpha=0.0, s=1.1, sigma=1.0)
    prior_vis = VelocityKLPrior(K=K, grid_size=args.vis_res, v_background=v_bg,
                                sigma_m=sigma_m, alpha=0.0, s=1.1, sigma=1.0)

    # Source / receiver positions on native grid (data uses pad=2).
    src_pos, rec_pos = make_src_rec(nx=grid_size, nz=grid_size,
                                    n_src=int(data["n_src"]),
                                    n_rec=int(data["n_rec"]), pad=2)
    # Convert to spatial coords in [0,1]
    src_xy = np.array([(ix / (grid_size - 1), iz / (grid_size - 1))
                       for ix, iz in src_pos])
    rec_xy = np.array([(ix / (grid_size - 1), iz / (grid_size - 1))
                       for ix, iz in rec_pos])

    # Test split
    test_start = args.num_train + args.num_val
    n_available = z_all.shape[0] - test_start
    if n_available < args.n_test_obs:
        test_start = z_all.shape[0] - args.n_test_obs
    z_test = z_all[test_start:test_start + args.n_test_obs]
    y_test = y_all[test_start:test_start + args.n_test_obs]
    y_test_n = (y_test - y_mean) / y_std
    args.n_test_obs = z_test.shape[0]
    print(f"Test: {args.n_test_obs} obs from idx {test_start}")

    # Visualization observation
    vis_idx = min(args.vis_obs_idx, args.n_test_obs - 1)
    z_true_vis = z_test[vis_idx].numpy()
    y_obs_vis_n = y_test_n[vis_idx].to(device)

    setup_style()

    # ================================================================
    # Part 1: Per-component VRF (over all test obs)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Per-component VRF ({args.n_test_obs} obs, {args.n_vrf_samples} samples)")
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

    print(f"Overall VRF: {all_mean_vrfs.mean():.4f} +/- {all_mean_vrfs.std():.4f}")

    # ================================================================
    # Part 2: Posterior mean + std (vis observation)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Posterior mean/std ({args.n_post_samples} samples) for obs {vis_idx}")
    print(f"{'='*60}")

    z_post, g_post = generate_samples(cnf, ensemble, y_obs_vis_n,
                                      args.n_post_samples, device)
    cv_mean_kl = (z_post - g_post).mean(0).numpy()

    # True velocity perturbation
    dv_true_vis = velocity_perturbation(prior_vis, z_true_vis[None])[0]
    dv_true_nat = velocity_perturbation(prior_nat, z_true_vis[None])[0]
    dv_cv_vis = velocity_perturbation(prior_vis, cv_mean_kl[None])[0]

    # Posterior std on velocity perturbation, computed in chunks
    print("Reconstructing posterior fields for pointwise std...")
    z_post_np = z_post.numpy()
    chunk_size = 500
    fields_sum = np.zeros((args.vis_res, args.vis_res), dtype=np.float64)
    fields_sq_sum = np.zeros_like(fields_sum)
    for ci in range(0, args.n_post_samples, chunk_size):
        chunk = z_post_np[ci:ci + chunk_size]
        fields = velocity_perturbation(prior_vis, chunk)  # (B, R, R)
        fields_sum += fields.sum(axis=0)
        fields_sq_sum += (fields ** 2).sum(axis=0)
    n = args.n_post_samples
    post_mean_vis = fields_sum / n
    post_std_vis = np.sqrt(np.clip(fields_sq_sum / n - post_mean_vis ** 2, 0, None))

    # ================================================================
    # Part 3: Multi-trial estimator std
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Estimator std ({args.n_trials} trials x {args.n_samples} samples)")
    print(f"{'='*60}")

    van_means, cv_means = [], []
    for _ in tqdm(range(args.n_trials), desc="Trials"):
        zt, gt = generate_samples(cnf, ensemble, y_obs_vis_n,
                                  args.n_samples, device)
        van_means.append(zt.mean(0))
        cv_means.append((zt - gt).mean(0))
    van_means = torch.stack(van_means).numpy()
    cv_means = torch.stack(cv_means).numpy()

    van_fields = velocity_perturbation(prior_vis, van_means)
    cv_fields = velocity_perturbation(prior_vis, cv_means)

    std_van = van_fields.std(axis=0)
    std_cv = cv_fields.std(axis=0)
    ratio_vis = std_van / np.clip(std_cv, 1e-12, None)

    van_nat = velocity_perturbation(prior_nat, van_means)
    cv_nat = velocity_perturbation(prior_nat, cv_means)
    std_van_nat = van_nat.std(axis=0)
    std_cv_nat = cv_nat.std(axis=0)
    ratio_nat = std_cv_nat / np.clip(std_van_nat, 1e-12, None)

    print(f"Ambient {grid_size}x{grid_size}: "
          f"std_van={std_van_nat.mean():.5f}, "
          f"std_cv={std_cv_nat.mean():.5f}, "
          f"ratio={ratio_nat.mean():.3f}")

    # ================================================================
    # FIGURE A: Posterior summary (4 panels)
    # ================================================================
    ext = (0, 1, 0, 1)

    fig1 = plt.figure(figsize=(7.2, 1.8))
    gs1 = GridSpec(1, 4, figure=fig1, wspace=0.55,
                   left=0.04, right=0.97, top=0.88, bottom=0.08)
    axes1 = [fig1.add_subplot(gs1[j]) for j in range(4)]

    # (a) True velocity perturbation with src/rec markers
    all_dv = np.concatenate([dv_true_vis.ravel(), dv_cv_vis.ravel()])
    vmin_v, vmax_v = np.percentile(all_dv, [1, 99])

    contour_field(axes1[0], dv_true_vis.T,
                  r"(a) True $\delta v^\dagger$ + src/rec",
                  vmin=vmin_v, vmax=vmax_v)
    axes1[0].scatter(src_xy[:, 0], src_xy[:, 1], s=22, marker="*",
                     facecolors="white", edgecolors="k", linewidths=0.6,
                     zorder=5)
    axes1[0].scatter(rec_xy[:, 0], rec_xy[:, 1], s=10,
                     facecolors="none", edgecolors="k", linewidths=0.5,
                     zorder=5)
    axes1[0].set_ylabel(r"$s_2$", fontsize=8)
    axes1[0].set_xlabel(r"$s_1$", fontsize=8)

    contour_field(axes1[1], dv_true_vis.T, r"(b) True $\delta v^\dagger$",
                  vmin=vmin_v, vmax=vmax_v)
    axes1[1].set_xlabel(r"$s_1$", fontsize=8)
    axes1[1].set_yticklabels([])

    contour_field(axes1[2], dv_cv_vis.T, r"(c) CNCV mean",
                  vmin=vmin_v, vmax=vmax_v)
    axes1[2].set_xlabel(r"$s_1$", fontsize=8)
    axes1[2].set_yticklabels([])

    contour_field(axes1[3], post_std_vis.T, r"(d) Post.\ std",
                  cmap=STD_CMAP, vmin=0)
    axes1[3].set_xlabel(r"$s_1$", fontsize=8)
    axes1[3].set_yticklabels([])

    for suffix in ("pdf", "png"):
        fig1.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_posterior.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {args.output_dir}/helmholtz_posterior.pdf")

    # ================================================================
    # FIGURE B: Estimator std (4 panels) — same layout as Darcy
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
    ymax = max(0.35, all_mean_vrfs.max() * 1.3)
    ax_d.set_ylim(0, ymax)
    ax_d.set_aspect(args.n_test_obs / ymax)
    ax_d.legend(fontsize=6, loc="upper right")
    ax_d.tick_params(labelsize=7)

    for suffix in ("pdf", "png"):
        fig2.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_estimator_std.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {args.output_dir}/helmholtz_estimator_std.pdf")

    # ================================================================
    # FIGURE C: Per-component VRF
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
    # If max VRF > 1.5, use log; else linear up to 1.15
    if comp_vrf_sorted.max() > 1.5:
        ax_i.set_yscale("log")
        ax_i.set_ylim(max(1e-3, comp_vrf_sorted.min() * 0.5),
                      comp_vrf_sorted.max() * 1.5)
    else:
        ax_i.set_ylim(0, min(1.15, comp_vrf_sorted.max() * 1.15))
    ax_i.tick_params(labelsize=7)

    ax_i2 = ax_i.twinx()
    ax_i2.plot(range(d_z), eig_sorted, color="C1", lw=1.2, alpha=0.7)
    ax_i2.set_ylabel(r"Eigenvalue $\lambda$", color="C1", fontsize=8)
    ax_i2.tick_params(axis="y", colors="C1", labelsize=7)
    ax_i2.spines["right"].set_color("C1")

    for suffix in ("pdf", "png"):
        fig3.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_component_vrf.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {args.output_dir}/helmholtz_component_vrf.pdf")

    # ================================================================
    # FIGURE D: Sample efficiency (VRF + MSE vs N) — Darcy layout
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Sample efficiency ({args.n_eff_obs} obs, {args.n_eff_trials} trials)")
    print(f"{'='*60}")

    sample_sizes = sorted(args.sample_sizes)
    max_N = max(sample_sizes)
    total_needed = max_N * args.n_eff_trials

    # Reference posterior mean: large pool of CNCV-corrected samples
    # used as ground truth for MSE.
    vrf_per_obs = {N: [] for N in sample_sizes}
    mse_raw_per_obs = {N: [] for N in sample_sizes}
    mse_cv_per_obs = {N: [] for N in sample_sizes}

    for i in tqdm(range(min(args.n_eff_obs, args.n_test_obs)), desc="Eff obs"):
        y_obs = y_test_n[i].to(device)

        # Pool of samples
        zs, gs = [], []
        n_gen = 0
        cs = 4000
        while n_gen < total_needed:
            bs = min(cs, total_needed - n_gen)
            with torch.no_grad():
                zb = cnf.sample(y_obs.unsqueeze(0), bs)
            yr = y_obs.unsqueeze(0).expand(bs, -1)
            sc = cnf.score(zb, yr)
            with torch.no_grad():
                gb, _ = ensemble(zb, yr, sc)
            zs.append(zb.cpu()); gs.append(gb.T.cpu())
            n_gen += bs
        z_pool = torch.cat(zs, 0).numpy()  # (total_needed, d)
        g_pool = torch.cat(gs, 0).numpy()
        # ground truth = CNCV-corrected pool mean (lower variance)
        truth = (z_pool - g_pool).mean(axis=0)  # (d,)

        z_trials = z_pool[:args.n_eff_trials * max_N].reshape(
            args.n_eff_trials, max_N, d_z)
        g_trials = g_pool[:args.n_eff_trials * max_N].reshape(
            args.n_eff_trials, max_N, d_z)

        for N in sample_sizes:
            zN = z_trials[:, :N]
            gN = g_trials[:, :N]
            van_m = zN.mean(axis=1)            # (n_trials, d)
            cv_m = (zN - gN).mean(axis=1)      # (n_trials, d)

            # VRF: mean over components of var(estimator)/var(raw mc)
            var_van = van_m.var(axis=0).mean()
            var_cv = cv_m.var(axis=0).mean()
            vrf = var_cv / max(var_van, 1e-15)
            vrf_per_obs[N].append(vrf)

            mse_raw = ((van_m - truth) ** 2).mean()
            mse_cv = ((cv_m - truth) ** 2).mean()
            mse_raw_per_obs[N].append(mse_raw)
            mse_cv_per_obs[N].append(mse_cv)

    # Aggregate (mean across observations)
    vrf_mean_arr = np.array([np.mean(vrf_per_obs[N]) for N in sample_sizes])
    vrf_std_arr = np.array([np.std(vrf_per_obs[N]) for N in sample_sizes])
    mse_raw_arr = np.array([np.mean(mse_raw_per_obs[N]) for N in sample_sizes])
    mse_cv_arr = np.array([np.mean(mse_cv_per_obs[N]) for N in sample_sizes])
    mse_raw_std = np.array([np.std(mse_raw_per_obs[N]) for N in sample_sizes])
    mse_cv_std = np.array([np.std(mse_cv_per_obs[N]) for N in sample_sizes])

    print("\n  N     VRF (mean+/-std)    MSE_raw       MSE_cv")
    for i, N in enumerate(sample_sizes):
        print(f"  {N:5d}  {vrf_mean_arr[i]:.4f}+/-{vrf_std_arr[i]:.4f}   "
              f"{mse_raw_arr[i]:.3e}  {mse_cv_arr[i]:.3e}")

    # Plot — match Darcy darcy_sample_efficiency.pdf style
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                         "axes.titlesize": 13, "legend.fontsize": 10})
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))

    # (a) VRF vs N with shaded std band
    ax1.plot(sample_sizes, vrf_mean_arr, "k-s", markersize=5, linewidth=1.8,
             label=f"Mean (d={d_z})", zorder=5)
    ax1.fill_between(sample_sizes,
                     vrf_mean_arr - vrf_std_arr,
                     vrf_mean_arr + vrf_std_arr,
                     color="#4C72B0", alpha=0.15, edgecolor="#4C72B0",
                     linewidth=0.5)
    # Draw individual obs as faint lines, like Darcy.
    for i in range(min(args.n_eff_obs, args.n_test_obs)):
        line = [vrf_per_obs[N][i] for N in sample_sizes]
        ax1.plot(sample_sizes, line, "-", color="#4C72B0",
                 linewidth=0.6, alpha=0.4)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Sample size $N$")
    ax1.set_ylabel("VRF")
    ax1.set_title("(a) Variance reduction factor")
    ax1.legend()
    ax1.grid(alpha=0.2, linewidth=0.5)

    # (b) MSE vs N (log-log) — Raw MC dashed gray, CNCV solid black
    ax2.plot(sample_sizes, mse_raw_arr, "--^", color="gray",
             markersize=5, linewidth=1.5, alpha=0.85, label="Raw MC")
    ax2.plot(sample_sizes, mse_cv_arr, "k-s", markersize=5,
             linewidth=1.8, label="CNCV", zorder=5)
    # 1/N reference
    n_arr = np.array(sample_sizes, dtype=float)
    c_ref = mse_raw_arr[0] * n_arr[0]
    ax2.plot(n_arr, c_ref / n_arr, ":", color="gray", linewidth=1.0,
             alpha=0.6, label=r"$\propto 1/N$")
    # CNCV std band
    ax2.fill_between(sample_sizes,
                     np.clip(mse_cv_arr - mse_cv_std, 1e-12, None),
                     mse_cv_arr + mse_cv_std,
                     color="#4C72B0", alpha=0.15, edgecolor="none")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Sample size $N$")
    ax2.set_ylabel("MSE")
    ax2.set_title("(b) Mean squared error")
    ax2.legend()
    ax2.grid(alpha=0.2, linewidth=0.5, which="both")

    plt.tight_layout(w_pad=0.5)
    for suffix in ("pdf", "png"):
        fig4.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_sample_efficiency.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {args.output_dir}/helmholtz_sample_efficiency.pdf")

    # Reset style
    setup_style()

    # ================================================================
    # FIGURE E: KL scatter (3 obs x 2 panels) — Darcy parallel
    # ================================================================
    print(f"\n{'='*60}")
    print(f"KL scatter ({len(args.scatter_obs)} obs)")
    print(f"{'='*60}")

    z_prior = z_all[:1000]

    fig5, axes5 = plt.subplots(3, 2, figsize=(3.4, 4.5))
    fig5.subplots_adjust(hspace=0.45, wspace=0.35, left=0.14, right=0.96,
                         top=0.95, bottom=0.06)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    obs_labels = ["Obs 1", "Obs 2", "Obs 3"]

    for row, (oidx, olabel) in enumerate(zip(args.scatter_obs[:3], obs_labels)):
        if oidx >= args.n_test_obs:
            print(f"  Skip oidx={oidx} (out of range)")
            continue
        z_true = z_test[oidx]
        y_obs_n = y_test_n[oidx].to(device)

        with torch.no_grad():
            z_post_s = cnf.sample(y_obs_n.unsqueeze(0), 1000).cpu()
        z_post_mean = z_post_s.mean(0)

        # Left: prior + post + truth
        ax_l = axes5[row, 0]
        ax_l.scatter(z_prior[:, 0].numpy(), z_prior[:, 1].numpy(),
                     s=1, alpha=0.15, c="#4C72B0", rasterized=True,
                     label="Prior")
        ax_l.scatter(z_post_s[:, 0].numpy(), z_post_s[:, 1].numpy(),
                     s=2, alpha=0.3, c="#DD8452", rasterized=True,
                     label="CNF post.")
        ax_l.scatter(z_true[0].item(), z_true[1].item(),
                     s=25, marker="*", c="red", zorder=5,
                     label=r"$\mathbf{z}^\dagger$")
        ax_l.set_title(f"{panel_labels[2*row]} {olabel}",
                       fontsize=8, pad=3)
        if row == 2:
            ax_l.set_xlabel(r"$z_1$")
        ax_l.set_ylabel(r"$z_2$")
        if row == 0:
            ax_l.legend(fontsize=5.5, loc="upper right",
                        handletextpad=0.3, markerscale=1.5)

        # Right: zoomed posterior
        ax_r = axes5[row, 1]
        ax_r.scatter(z_post_s[:, 0].numpy(), z_post_s[:, 1].numpy(),
                     s=2, alpha=0.3, c="#DD8452", rasterized=True)
        ax_r.scatter(z_true[0].item(), z_true[1].item(),
                     s=25, marker="*", c="red", zorder=5,
                     label=r"$\mathbf{z}^\dagger$")
        ax_r.scatter(z_post_mean[0].item(), z_post_mean[1].item(),
                     s=18, marker="D", c="#55A868", zorder=5,
                     label=r"CNF mean")

        z0_lo = z_post_s[:, 0].quantile(0.01).item()
        z0_hi = z_post_s[:, 0].quantile(0.99).item()
        z1_lo = z_post_s[:, 1].quantile(0.01).item()
        z1_hi = z_post_s[:, 1].quantile(0.99).item()
        m0 = 0.15 * max(z0_hi - z0_lo, 1e-3)
        m1 = 0.15 * max(z1_hi - z1_lo, 1e-3)
        ax_r.set_xlim(z0_lo - m0, z0_hi + m0)
        ax_r.set_ylim(z1_lo - m1, z1_hi + m1)
        ax_r.set_title(f"{panel_labels[2*row+1]} {olabel} (zoom)",
                       fontsize=8, pad=3)
        if row == 2:
            ax_r.set_xlabel(r"$z_1$")
        if row == 0:
            ax_r.legend(fontsize=5.5, loc="upper right",
                        handletextpad=0.3, markerscale=1.5)

    for suffix in ("pdf", "png"):
        fig5.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_kl_scatter.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved: {args.output_dir}/helmholtz_kl_scatter.pdf")

    # ================================================================
    # FIGURE F: Loss curves (CNF + CNCV) — uses full-training ckpts
    # ================================================================
    print(f"\n{'='*60}")
    print("Loss curves")
    print(f"{'='*60}")

    cnf_full = torch.load(args.cnf_checkpoint_full, map_location="cpu",
                          weights_only=False)
    cncv_full = torch.load(args.cncv_checkpoint_full, map_location="cpu",
                           weights_only=False)
    cnf_train = np.array(cnf_full["train_losses"])
    cnf_val = np.array(cnf_full["val_losses"])
    cncv_train = np.array(cncv_full["train_losses"])
    cncv_val = np.array(cncv_full["val_losses"])

    fig6, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.4, 1.6))
    fig6.subplots_adjust(wspace=0.50, left=0.12, right=0.97,
                         top=0.88, bottom=0.22)

    def smooth(x, k=50):
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="valid")

    # (a) CNF
    ax_a.plot(cnf_train, alpha=0.15, lw=0.3, c="#4C72B0")
    cnf_smooth = smooth(cnf_train)
    ax_a.plot(np.arange(49, 49 + len(cnf_smooth)), cnf_smooth,
              lw=1.0, c="#4C72B0", label="Train")
    bpe = max(1, len(cnf_train) / max(1, len(cnf_val)))
    val_iters = np.arange(len(cnf_val)) * bpe
    ax_a.plot(val_iters, cnf_val, lw=1.0, c="#DD8452", label="Val")
    ax_a.set_xlabel("Batch iteration")
    ax_a.set_ylabel("NLL")
    ax_a.set_title("(a) CNF", fontsize=8, pad=3)
    ax_a.legend(fontsize=6, loc="upper right")
    ax_a.tick_params(labelsize=7)

    # (b) CNCV
    ax_b.plot(cncv_train, alpha=0.15, lw=0.3, c="#4C72B0")
    cncv_smooth = smooth(cncv_train)
    ax_b.plot(np.arange(49, 49 + len(cncv_smooth)), cncv_smooth,
              lw=1.0, c="#4C72B0", label="Train")
    bpe_cv = max(1, len(cncv_train) / max(1, len(cncv_val)))
    val_iters_cv = np.arange(len(cncv_val)) * bpe_cv
    ax_b.plot(val_iters_cv, cncv_val, lw=1.0, c="#DD8452", label="Val")
    ax_b.set_xlabel("Batch iteration")
    ax_b.set_ylabel("MSE")
    ax_b.set_title("(b) CNCV", fontsize=8, pad=3)
    ax_b.legend(fontsize=6, loc="upper right")
    ax_b.tick_params(labelsize=7)

    for suffix in ("pdf", "png"):
        fig6.savefig(os.path.join(args.output_dir,
                                  f"helmholtz_loss_curves.{suffix}"),
                     dpi=300, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved: {args.output_dir}/helmholtz_loss_curves.pdf")

    # ================================================================
    # FIGURE G: Posterior obs gallery (4 panels per obs)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Posterior obs gallery ({len(args.gallery_obs)} obs)")
    print(f"{'='*60}")

    for oidx in args.gallery_obs:
        if oidx >= args.n_test_obs:
            print(f"  Skip oidx={oidx} (out of range)")
            continue
        z_true_g = z_test[oidx].numpy()
        y_obs_g = y_test_n[oidx].to(device)

        z_pg, g_pg = generate_samples(cnf, ensemble, y_obs_g,
                                      args.n_post_samples, device)
        cv_kl_g = (z_pg - g_pg).mean(0).numpy()

        dv_true_g = velocity_perturbation(prior_vis, z_true_g[None])[0]
        dv_cv_g = velocity_perturbation(prior_vis, cv_kl_g[None])[0]

        z_pg_np = z_pg.numpy()
        fs = np.zeros((args.vis_res, args.vis_res), dtype=np.float64)
        fsq = np.zeros_like(fs)
        for ci in range(0, args.n_post_samples, 500):
            ck = z_pg_np[ci:ci + 500]
            ff = velocity_perturbation(prior_vis, ck)
            fs += ff.sum(axis=0)
            fsq += (ff ** 2).sum(axis=0)
        post_mean_g = fs / args.n_post_samples
        post_std_g = np.sqrt(np.clip(fsq / args.n_post_samples
                                     - post_mean_g ** 2, 0, None))

        all_dv_g = np.concatenate([dv_true_g.ravel(), dv_cv_g.ravel()])
        vmn, vmx = np.percentile(all_dv_g, [1, 99])

        figg = plt.figure(figsize=(7.2, 1.8))
        gsg = GridSpec(1, 4, figure=figg, wspace=0.55,
                       left=0.04, right=0.97, top=0.88, bottom=0.08)
        axg = [figg.add_subplot(gsg[j]) for j in range(4)]

        contour_field(axg[0], dv_true_g.T,
                      r"(a) True $\delta v^\dagger$ + src/rec",
                      vmin=vmn, vmax=vmx)
        axg[0].scatter(src_xy[:, 0], src_xy[:, 1], s=22, marker="*",
                       facecolors="white", edgecolors="k", linewidths=0.6,
                       zorder=5)
        axg[0].scatter(rec_xy[:, 0], rec_xy[:, 1], s=10,
                       facecolors="none", edgecolors="k", linewidths=0.5,
                       zorder=5)
        axg[0].set_ylabel(r"$s_2$", fontsize=8)
        axg[0].set_xlabel(r"$s_1$", fontsize=8)

        contour_field(axg[1], dv_true_g.T,
                      r"(b) True $\delta v^\dagger$",
                      vmin=vmn, vmax=vmx)
        axg[1].set_xlabel(r"$s_1$", fontsize=8)
        axg[1].set_yticklabels([])

        contour_field(axg[2], dv_cv_g.T, r"(c) CNCV mean",
                      vmin=vmn, vmax=vmx)
        axg[2].set_xlabel(r"$s_1$", fontsize=8)
        axg[2].set_yticklabels([])

        contour_field(axg[3], post_std_g.T, r"(d) Posterior std",
                      cmap=STD_CMAP, vmin=0)
        axg[3].set_xlabel(r"$s_1$", fontsize=8)
        axg[3].set_yticklabels([])

        for suffix in ("pdf", "png"):
            figg.savefig(os.path.join(args.output_dir,
                                      f"helmholtz_posterior_obs{oidx}.{suffix}"),
                         dpi=300, bbox_inches="tight")
        plt.close(figg)
        print(f"  Saved obs {oidx}")

    # ================================================================
    # Save numerical results
    # ================================================================
    out_npz = os.path.join(args.output_dir, "helmholtz_paper_results.npz")
    np.savez(out_npz,
             all_comp_vrfs=all_comp_vrfs,
             all_mean_vrfs=all_mean_vrfs,
             std_van=std_van, std_cv=std_cv, ratio_vis=ratio_vis,
             std_van_nat=std_van_nat, std_cv_nat=std_cv_nat,
             ratio_nat=ratio_nat,
             dv_true_vis=dv_true_vis, dv_cv_vis=dv_cv_vis,
             post_std_vis=post_std_vis, post_mean_vis=post_mean_vis,
             eigenvalues=eigenvalues,
             src_xy=src_xy, rec_xy=rec_xy,
             sample_sizes=np.array(sample_sizes),
             vrf_mean=vrf_mean_arr, vrf_std=vrf_std_arr,
             mse_raw=mse_raw_arr, mse_cv=mse_cv_arr,
             mse_raw_std=mse_raw_std, mse_cv_std=mse_cv_std,
             vis_obs_idx=vis_idx, gallery_obs=np.array(args.gallery_obs))

    # Numbers for paper
    cm = all_comp_vrfs.mean(0)
    print(f"\n{'='*60}")
    print("Numbers for paper:")
    print(f"{'='*60}")
    print(f"  VRF (KL): {all_mean_vrfs.mean():.3f} +/- {all_mean_vrfs.std():.3f}")
    print(f"  Reduction: {(1 - all_mean_vrfs.mean()) * 100:.0f}%")
    print(f"  Per-comp range: [{cm.min():.3f}, {cm.max():.3f}]")
    print(f"  Per-comp median: {np.median(cm):.3f}")
    print(f"  Ambient std ratio: {ratio_nat.mean():.3f}")
    print(f"  ESS multiplier: {1.0 / all_mean_vrfs.mean():.1f}x")
    print(f"  Saved: {out_npz}")


if __name__ == "__main__":
    main()
