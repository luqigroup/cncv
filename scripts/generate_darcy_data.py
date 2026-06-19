"""Generate training data for Darcy flow.

Uses DarcyKLPrior (modes 1..K) and DarcyForwardModel (Beskos sensors).
Default: 131072 samples (128k), K=10 (d=100), grid 40x40.

Also stores the truth field for later evaluation.

Usage:
    python scripts/generate_darcy_data.py --N 131072 --K 10 --grid_size 40 \
        --sigma_y 0.01 --seed 12 --skip_scores
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Darcy flow training data (Beskos V2)")
    parser.add_argument("--N", type=int, default=131072, help="Number of samples")
    parser.add_argument("--K", type=int, default=10, help="KL cutoff (d = K^2 modes)")
    parser.add_argument("--grid_size", type=int, default=40, help="PDE grid resolution")
    parser.add_argument("--sigma_y", type=float, default=0.01, help="Observation noise std")
    parser.add_argument("--n_sensors", type=int, default=33, help="Number of sensors")
    parser.add_argument("--time_steps", type=int, default=5000, help="PDE solver timesteps")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="data/darcy", help="Output directory")
    parser.add_argument("--skip_scores", action="store_true",
                        help="Skip score computation (faster)")
    args = parser.parse_args()

    from cncv.utils.darcy import (
        DarcyKLPrior, DarcyForwardModel, beskos_truth,
        compute_score_darcy_batch,
    )

    d = args.K ** 2
    print(f"=== Darcy Flow V2 Data Generation (Beskos-correct) ===")
    print(f"  N={args.N}, K={args.K} (d={d}), grid={args.grid_size}x{args.grid_size}")
    print(f"  sigma_y={args.sigma_y}, n_sensors={args.n_sensors}")
    print(f"  Modes: i1, i2 in {{1, ..., {args.K}}} (Beskos)")
    print()

    kl_prior = DarcyKLPrior(K=args.K, grid_size=args.grid_size)
    forward_model = DarcyForwardModel(
        kl_prior,
        n_sensors=args.n_sensors,
        time_steps=args.time_steps,
    )

    print(f"Prior: {d} KL modes (Beskos indexing)")
    print(f"  Eigenvalue range: [{kl_prior.eigenvalues.min():.6f}, "
          f"{kl_prior.eigenvalues.max():.6f}]")
    print(f"  Sensor layout: 1 center + {args.n_sensors - 1} on full-domain circle")
    print()

    # Beskos truth
    u_kl_truth, u_field_truth = beskos_truth(kl_prior)
    print(f"Beskos truth: u_kl range [{u_kl_truth.min():.3f}, {u_kl_truth.max():.3f}]")
    print(f"  u_field range [{u_field_truth.min():.3f}, {u_field_truth.max():.3f}]")

    # Generate truth observations
    print("Computing truth observations...")
    y_truth_clean = forward_model.forward(u_kl_truth)
    print(f"  y_truth range: [{y_truth_clean.min():.6f}, {y_truth_clean.max():.6f}]")

    # Sample KL coefficients
    print(f"\nSampling {args.N} KL coefficients...")
    u_kl = kl_prior.sample(args.N, seed=args.seed)
    print(f"  u_kl shape: {u_kl.shape}, range: [{u_kl.min():.3f}, {u_kl.max():.3f}]")

    # Forward solves
    print(f"\nRunning {args.N} forward solves...")
    t0 = time.time()
    y_clean = forward_model.forward_batch(u_kl, verbose=True)
    solve_time = time.time() - t0
    print(f"  Done in {solve_time:.1f}s ({solve_time/args.N:.3f}s/solve)")

    # Add noise
    rng = np.random.default_rng(args.seed + 1)
    noise = args.sigma_y * rng.standard_normal(y_clean.shape).astype(np.float32)
    y_obs = y_clean + noise

    # Scores
    scores = None
    if not args.skip_scores:
        print(f"\nComputing posterior scores...")
        t0 = time.time()
        scores = compute_score_darcy_batch(
            u_kl, y_obs, forward_model, kl_prior, args.sigma_y, verbose=True
        )
        score_time = time.time() - t0
        print(f"  Done in {score_time:.1f}s")

    # Save
    save_dict = {
        "u_kl": u_kl, "y_obs": y_obs, "y_clean": y_clean,
        "K": args.K, "d": d, "grid_size": args.grid_size,
        "sigma_y": args.sigma_y, "n_sensors": args.n_sensors,
        "sensor_ix": forward_model.sensor_ix,
        "sensor_iy": forward_model.sensor_iy,
        "sensor_x": forward_model.sensor_x,
        "sensor_y": forward_model.sensor_y,
        "eigenvalues": kl_prior.eigenvalues,
        "time_steps": args.time_steps, "seed": args.seed,
        "u_kl_truth": u_kl_truth,
        "u_field_truth": u_field_truth,
        "y_truth_clean": y_truth_clean,
        "version": "v2_beskos",
    }
    if scores is not None:
        save_dict["scores"] = scores

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(
        args.out_dir, f"darcy_K{args.K}_N{args.N}_grid{args.grid_size}.npz"
    )
    np.savez(save_path, **save_dict)
    print(f"\nSaved to {save_path}")
    file_size_mb = os.path.getsize(save_path) / 1e6
    print(f"  File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
