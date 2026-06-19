"""Generate joint (z, y) samples for amortized CNCV on Helmholtz FWI.

Mirrors generate_darcy_data.py, using the Helmholtz forward solver in
cncv.utils.helmholtz. Uses a 32x32 grid for speed.

Output keys (npz):
  z:           (N, 100) KL coefficients ~ N(0, I)
  y_real:      (N, 200) real-part observations (n_src*n_rec=200 at 32x32 default 5x40)
  y_imag:      (N, 200) imaginary-part observations
  y_obs:       (N, 400) concatenated [real, imag] noisy observations
  m_fields:    (N, 32, 32) squared-slowness fields (optional, large)
  eigenvalues: (100,)
  metadata...

Usage:
  python scripts/generate_helmholtz_joint.py --N 5000 --grid_size 32 --output data/helmholtz_joint.npz
"""

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from cncv.utils.helmholtz import HelmholtzSolver, make_src_rec
from cncv.utils.velocity_kl_prior import VelocityKLPrior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--v_background", type=float, default=2.0)
    parser.add_argument("--sigma_m", type=float, default=0.02)
    parser.add_argument("--n_src", type=int, default=5)
    parser.add_argument("--n_rec", type=int, default=40)
    parser.add_argument("--frequency", type=float, default=4.0)  # Hz
    parser.add_argument("--noise_rel", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_fields", action="store_true",
                        help="Save raw m_fields too (large)")
    args = parser.parse_args()

    print(f"=== Helmholtz FWI joint sample generation ===")
    print(f"  N = {args.N}, K = {args.K}, grid = {args.grid_size}x{args.grid_size}")
    print(f"  n_src = {args.n_src}, n_rec = {args.n_rec}, freq = {args.frequency} Hz")

    # --- Setup ---
    rng = np.random.default_rng(args.seed)
    kl_prior = VelocityKLPrior(K=args.K, grid_size=args.grid_size,
                                v_background=args.v_background,
                                sigma_m=args.sigma_m)
    omega = 2.0 * np.pi * args.frequency  # angular frequency
    nx = nz = args.grid_size
    dx = 1.0 / nx  # domain [0,1] for default 1km

    solver = HelmholtzSolver(nx=nx, nz=nz, dx=dx)
    src_pos, rec_pos = make_src_rec(nx, nz,
                                     n_src=args.n_src, n_rec=args.n_rec,
                                     pad=2)
    n_obs_complex = args.n_src * args.n_rec  # complex
    n_obs_real = 2 * n_obs_complex  # [real; imag]
    print(f"  n_obs (real-valued) = {n_obs_real}")

    # --- Sample z and forward ---
    z_all = kl_prior.sample(args.N, seed=args.seed)
    y_clean_real = np.zeros((args.N, n_obs_complex), dtype=np.float32)
    y_clean_imag = np.zeros((args.N, n_obs_complex), dtype=np.float32)
    if args.save_fields:
        m_all = np.zeros((args.N, nx, nz), dtype=np.float32)

    t0 = time.time()
    for i in tqdm(range(args.N), desc="Forward solves"):
        m_field = kl_prior.reconstruct(z_all[i])
        d = solver.forward(m_field, omega, src_pos, rec_pos)  # (n_src, n_rec) complex64
        y_clean_real[i] = d.real.ravel().astype(np.float32)
        y_clean_imag[i] = d.imag.ravel().astype(np.float32)
        if args.save_fields:
            m_all[i] = m_field.astype(np.float32)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.N - i - 1) / rate
            tqdm.write(f"  {i+1}/{args.N}, rate={rate:.2f}/s, ETA={eta/60:.1f}min")

    total_time = time.time() - t0
    print(f"\nForward solves done in {total_time/60:.1f} min "
          f"({args.N / total_time:.2f}/s)")

    # --- Add noise ---
    #   noise_level = noise_rel * max|d_clean|
    #   d_obs = d_clean + noise_level * (eps_re + 1j*eps_im) / sqrt(2)
    # i.e. complex noise has total variance noise_level^2;
    # real-part and imag-part each have std = noise_level/sqrt(2).
    y_clean_complex = y_clean_real + 1j * y_clean_imag
    noise_level = args.noise_rel * np.abs(y_clean_complex).max()
    sigma_per_part = noise_level / np.sqrt(2.0)
    print(f"  noise_level = {noise_level:.4e} (noise_rel * max|y_clean|)")
    print(f"  per-part std = {sigma_per_part:.4e}")

    eps_real = sigma_per_part * rng.standard_normal((args.N, n_obs_complex)).astype(np.float32)
    eps_imag = sigma_per_part * rng.standard_normal((args.N, n_obs_complex)).astype(np.float32)
    y_real = y_clean_real + eps_real
    y_imag = y_clean_imag + eps_imag
    y_obs = np.concatenate([y_real, y_imag], axis=1)  # (N, 2*n_obs_complex)
    sigma_y = sigma_per_part  # for backwards-compat; the per-part std

    # --- Save ---
    save_dict = dict(
        z=z_all,
        y_obs=y_obs,
        y_real=y_real, y_imag=y_imag,
        y_clean_real=y_clean_real, y_clean_imag=y_clean_imag,
        eigenvalues=kl_prior.eigenvalues,
        sigma_y=sigma_y,
        omega=omega, frequency=args.frequency,
        K=args.K, grid_size=args.grid_size,
        v_background=args.v_background, sigma_m=args.sigma_m,
        n_src=args.n_src, n_rec=args.n_rec,
        seed=args.seed,
    )
    if args.save_fields:
        save_dict["m_fields"] = m_all

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **save_dict)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)")
    print(f"  z: {z_all.shape}")
    print(f"  y_obs: {y_obs.shape} (real-valued, [real|imag])")


if __name__ == "__main__":
    main()
