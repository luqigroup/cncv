"""Generate sample efficiency figure for Darcy KL-space CNCV (paper Fig. 10).

Mirrors Figure 3 from the Gaussian d=4 experiment but adapted for d=100 Darcy:
  - No analytical posterior: ground truth mean estimated from a large CNF
    sample pool.
  - Average VRF/MSE across all KL components, plus mean curve.
  - Models (frozen CNF + CNCV ensemble) loaded from the CNCV checkpoint, which
    stores the CNF checkpoint path internally. No PDE solve / no Devito needed:
    posterior samples come from the CNF, the control variate from the ensemble.

Produces a 1x2 panel:
  (a) VRF vs N
  (b) MSE vs N  -- both follow the 1/N rate, CNCV curve shifted down

Usage:
    python scripts/generate_darcy_sample_efficiency.py --gpu 0 \
        --cncv_checkpoint data/darcy/checkpoints_cncv_kl_v2b/cncv_best.pth \
        --data_path data/darcy/darcy_K10_N131072_grid40.npz \
        --output_dir figs/darcy_kl_v2b
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils.download import ensure_dataset

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve(path):
    """Resolve a possibly-relative path against the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO, path)


plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
})

SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def load_models(cncv_ckpt_path, device):
    """Load CNCV ensemble and the frozen CNF it was trained against."""
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

    cnf_ckpt = torch.load(_resolve(ckpt["cnf_checkpoint"]), map_location=device,
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
    """Return z (n, d) and g (n, d) on CPU for a single normalized observation."""
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
    parser = argparse.ArgumentParser(
        description="Generate Darcy KL sample efficiency figure (paper Fig. 10)")
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v2b/cncv_best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--num_pool", type=int, default=50000,
                        help="Large sample pool for the ground-truth mean; kept "
                             ">> max(SAMPLE_SIZES) to avoid finite-pool bias")
    parser.add_argument("--n_test_obs", type=int, default=5,
                        help="Number of test observations to average over")
    parser.add_argument("--n_trials", type=int, default=1000,
                        help="Bootstrap trials per sample size")
    parser.add_argument("--output_dir", type=str, default="figs/darcy_kl_v2b")
    parser.add_argument("--plot_only", action="store_true",
                        help="Regenerate plot from saved _data.npz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(args.output_dir, "darcy_sample_efficiency")
    data_file = f"{out_base}_data.npz"

    if args.plot_only:
        print(f"Loading saved data: {data_file}")
        saved = np.load(data_file)
        all_vrf_med = saved["all_vrf_med"]
        all_mse_raw_med = saved["all_mse_raw_med"]
        all_mse_cv_med = saved["all_mse_cv_med"]
        d_z = int(saved["d_z"]) if "d_z" in saved.files else 100
    else:
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load models
        cnf, ensemble, y_mean, y_std, d_z, d_y = load_models(
            _resolve(args.cncv_checkpoint), device)

        # Load data for test observations
        print(f"\nLoading: {args.data_path}")
        ensure_dataset(_resolve(args.data_path))
        data = np.load(_resolve(args.data_path))
        y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
        n_total = y_all.shape[0]

        # Use last n_test_obs as test set; clamp to the data set size.
        args.n_test_obs = min(args.n_test_obs, n_total)
        test_start = max(0, n_total - args.n_test_obs)
        y_test = y_all[test_start:]
        y_test_n = (y_test - y_mean) / y_std

        # Collect bootstrap results across test observations
        # We average VRF and MSE across observations for the final plot
        all_vrf_med = np.zeros((args.n_test_obs, len(SAMPLE_SIZES)))
        all_mse_raw_med = np.zeros((args.n_test_obs, len(SAMPLE_SIZES)))
        all_mse_cv_med = np.zeros((args.n_test_obs, len(SAMPLE_SIZES)))

        rng = np.random.default_rng(123)

        for obs_i in range(args.n_test_obs):
            print(f"\n{'='*60}")
            print(f"Test observation {obs_i+1}/{args.n_test_obs}")
            print(f"{'='*60}")

            y_obs = y_test_n[obs_i].to(device)

            # Draw large pool
            print(f"  Drawing {args.num_pool} posterior samples...")
            z_pool, g_pool = generate_samples(
                cnf, ensemble, y_obs, args.num_pool, device)
            h_np = z_pool.numpy()   # [N_pool, d]
            g_np = g_pool.numpy()   # [N_pool, d]

            # "Ground-truth" mean = empirical mean of the full CNF pool. By
            # Stein's identity g is zero-mean w.r.t. the CNF posterior, so both
            # raw-MC and CNCV are unbiased for this target (the CNF mean); the
            # comparison isolates variance and does not fold in CNF modeling
            # error. num_pool is kept >> max(SAMPLE_SIZES) so the without-
            # replacement subsampling below has negligible finite-pool coupling.
            gt_mean = h_np.mean(axis=0)  # [d]

            # Subsampling trials (without replacement -> sample-efficiency 1/N study)
            print("  Running subsampling trials...")
            for si, N in enumerate(SAMPLE_SIZES):
                vrf_trials = []
                mse_raw_trials = []
                mse_cv_trials = []
                for _ in range(args.n_trials):
                    idx = rng.choice(args.num_pool, size=N, replace=False)
                    h_sub = h_np[idx]      # [N, d]
                    g_sub = g_np[idx]      # [N, d]
                    resid = h_sub - g_sub  # [N, d]

                    # Per-component VRF, then average across d
                    var_h = np.var(h_sub, axis=0, ddof=1)       # [d]
                    var_r = np.var(resid, axis=0, ddof=1)       # [d]
                    vrf_comp = var_r / np.maximum(var_h, 1e-12) # [d]
                    vrf_trials.append(vrf_comp.mean())

                    # MSE vs ground truth (averaged across components)
                    est_raw = h_sub.mean(axis=0)     # [d]
                    est_cv = resid.mean(axis=0)      # [d]
                    mse_raw_trials.append(
                        np.mean((est_raw - gt_mean) ** 2))
                    mse_cv_trials.append(
                        np.mean((est_cv - gt_mean) ** 2))

                all_vrf_med[obs_i, si] = np.median(vrf_trials)
                all_mse_raw_med[obs_i, si] = np.median(mse_raw_trials)
                all_mse_cv_med[obs_i, si] = np.median(mse_cv_trials)
                print(f"    N={N:5d}: VRF={all_vrf_med[obs_i, si]:.4f}  "
                      f"MSE_raw={all_mse_raw_med[obs_i, si]:.2e}  "
                      f"MSE_cv={all_mse_cv_med[obs_i, si]:.2e}")

    # Average across test observations
    vrf_mean = all_vrf_med.mean(axis=0)
    mse_raw_mean = all_mse_raw_med.mean(axis=0)
    mse_cv_mean = all_mse_cv_med.mean(axis=0)

    # Individual obs for shading (min/max envelope)
    vrf_lo = all_vrf_med.min(axis=0)
    vrf_hi = all_vrf_med.max(axis=0)
    mse_raw_lo = all_mse_raw_med.min(axis=0)
    mse_raw_hi = all_mse_raw_med.max(axis=0)
    mse_cv_lo = all_mse_cv_med.min(axis=0)
    mse_cv_hi = all_mse_cv_med.max(axis=0)

    # -- Plot ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))

    # Panel (a): VRF vs N
    for obs_i in range(all_vrf_med.shape[0]):
        ax1.plot(SAMPLE_SIZES, all_vrf_med[obs_i], "-", color="C0",
                 alpha=0.25, linewidth=0.8)
    ax1.plot(SAMPLE_SIZES, vrf_mean, "k-s", markersize=5, linewidth=1.8,
             label=f"Mean (d={d_z})", zorder=5)
    ax1.fill_between(SAMPLE_SIZES, vrf_lo, vrf_hi,
                     color="C0", alpha=0.12)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample size $N$")
    ax1.set_ylabel("VRF")
    ax1.set_title("(a) Variance reduction factor")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2, linewidth=0.5)

    # Panel (b): MSE vs N
    ax2.plot(SAMPLE_SIZES, mse_raw_mean, "k--^", markersize=4, linewidth=1.5,
             alpha=0.7, label="Raw MC")
    ax2.fill_between(SAMPLE_SIZES, mse_raw_lo, mse_raw_hi,
                     color="gray", alpha=0.12)
    ax2.plot(SAMPLE_SIZES, mse_cv_mean, "k-s", markersize=5, linewidth=1.8,
             label="CNCV", zorder=5)
    ax2.fill_between(SAMPLE_SIZES, mse_cv_lo, mse_cv_hi,
                     color="C0", alpha=0.12)

    # Reference 1/N line
    ref_N = np.array(SAMPLE_SIZES, dtype=float)
    ref_y = mse_raw_mean[0] * SAMPLE_SIZES[0] / ref_N
    ax2.plot(SAMPLE_SIZES, ref_y, ":", color="gray", linewidth=0.8,
             alpha=0.5, label="$\\propto 1/N$")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample size $N$")
    ax2.set_ylabel("MSE")
    ax2.set_title("(b) Mean squared error")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout(w_pad=0.5)

    plt.savefig(f"{out_base}.pdf")
    plt.savefig(f"{out_base}.png")
    print(f"\nSaved: {out_base}.pdf")
    plt.close()

    # Save raw data (skip if plot_only)
    if not args.plot_only:
        np.savez(data_file,
                 sample_sizes=SAMPLE_SIZES,
                 d_z=d_z,
                 all_vrf_med=all_vrf_med,
                 all_mse_raw_med=all_mse_raw_med,
                 all_mse_cv_med=all_mse_cv_med,
                 vrf_mean=vrf_mean,
                 mse_raw_mean=mse_raw_mean,
                 mse_cv_mean=mse_cv_mean)
        print(f"Saved: {data_file}")


if __name__ == "__main__":
    main()
