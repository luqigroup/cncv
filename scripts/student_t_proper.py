"""Student-t noise CNCV with PROPER per-observation conditional evaluation.

A joint (x,y)-sample evaluation of the VRF has a high floor
(Var(E[x|y])/Var(x) ~ 0.92 for Gaussian-like settings) that masks
per-observation variance reduction.

This script:
  (1) trains CNCV under Student-t likelihood (analytical Student-t score),
  (2) at test time, draws X samples conditional on a fixed Y_obs via
      Langevin (using the same analytical score), giving per-observation
      conditional samples,
  (3) computes per-component VRF on those conditional samples.

This matches the paper's per-observation conditional setup and lets us
answer "does CNCV reduce variance under non-Gaussian (Student-t) noise"
fairly.

Output: figs/baselines/student_t_proper_d4.{pdf,png,json}
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cncv.models.ensemble import create_shared_hierarchical_ensemble
from cncv.utils.lr_scheduler import CustomLRScheduler


def student_t_score_likelihood(x, y, scale, nu):
    diff = y - x
    return (nu + 1) * diff / (nu * scale ** 2 + diff ** 2)


def gaussian_prior_score(x, mu_prior, Sigma_prior_inv):
    diff = x - mu_prior
    if x.ndim == 1:
        return -Sigma_prior_inv @ diff
    return -(Sigma_prior_inv @ diff.T).T


def sample_student_t(x, scale, nu, device):
    z = torch.randn_like(x)
    chi2 = torch.distributions.Chi2(df=nu).sample(x.shape).to(device)
    t = z / torch.sqrt(chi2 / nu)
    return x + scale * t


def langevin_posterior(y_obs, n_samples, mu_prior, Sigma_prior_inv,
                        scale, nu, device, n_burn=500, n_thin=2,
                        step_size=0.01, x_init=None):
    """Sample x ~ p(x|y_obs) via Langevin dynamics.

    p(x|y) = exp(log lik(y|x) + log prior(x)) / Z
    score = grad log p(x|y) = score_lik + score_prior

    Langevin: x_{t+1} = x_t + (eps^2/2) * score + eps * noise
    """
    n_dim = len(y_obs)
    if x_init is None:
        # Start from likelihood mode (y_obs) with small jitter
        x = y_obs.unsqueeze(0).expand(n_samples, n_dim).clone()
        x = x + 0.1 * torch.randn_like(x)
    else:
        x = x_init.clone()

    x = x.to(device)
    eps = step_size

    samples = []
    n_total = n_burn + n_samples * n_thin
    for step in range(n_total):
        score = (
            student_t_score_likelihood(x, y_obs.unsqueeze(0), scale, nu)
            + gaussian_prior_score(x, mu_prior, Sigma_prior_inv)
        )
        x = x + (eps ** 2 / 2) * score + eps * torch.randn_like(x)
        if step >= n_burn and (step - n_burn) % n_thin == 0:
            samples.append(x.clone())
    return torch.stack(samples, dim=0)[0]  # take last batch (n_samples, n_dim)


def langevin_posterior_simple(y_obs, n_samples, mu_prior, Sigma_prior_inv,
                              scale, nu, device, n_steps=1000, step_size=0.005):
    """Simpler version: parallel Langevin chains, one per requested sample."""
    n_dim = len(y_obs)
    x = y_obs.unsqueeze(0).expand(n_samples, n_dim).clone() + 0.1 * torch.randn(
        n_samples, n_dim, device=device
    )
    x = x.to(device)
    eps = step_size
    for _ in range(n_steps):
        score = (
            student_t_score_likelihood(x, y_obs.unsqueeze(0), scale, nu)
            + gaussian_prior_score(x, mu_prior, Sigma_prior_inv)
        )
        x = x + (eps ** 2 / 2) * score + eps * torch.randn_like(x)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--nu", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--num_train", type=int, default=131072)
    parser.add_argument("--num_test_obs", type=int, default=20)
    parser.add_argument("--num_samples_per_obs", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_final", type=float, default=1e-5)
    parser.add_argument("--n_ensemble", type=int, default=16)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_mlp_layers", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--langevin_steps", type=int, default=500)
    parser.add_argument("--langevin_step_size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="figs/baselines/student_t_proper")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_dim = args.dim
    mu_prior = torch.zeros(n_dim, device=device)
    Sigma_prior = torch.eye(n_dim, device=device)
    Sigma_prior_inv = torch.eye(n_dim, device=device)
    L_prior = torch.eye(n_dim, device=device)

    print(f"\n=== Training CNCV under Student-t(nu={args.nu}, scale={args.scale}) ===")
    z = torch.randn(args.num_train, n_dim, device=device)
    X_train = mu_prior + (L_prior @ z.T).T
    Y_train = sample_student_t(X_train, args.scale, args.nu, device)

    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_dim, args.n_hidden, args.n_ensemble,
        n_layers=args.n_mlp_layers,
        depth=args.depth,
        n_cv=n_dim, seed=args.seed,
    ).to(device)
    print(f"CNCV params: {sum(p.numel() for p in ensemble.parameters()):,}")

    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr)
    n_iters = args.epochs * (args.num_train // args.batchsize)
    sched = CustomLRScheduler(optimizer, args.lr, args.lr_final, n_iters)

    n_batches = args.num_train // args.batchsize
    indices = torch.arange(args.num_train, device=device)
    for ep in range(args.epochs):
        ensemble.train()
        perm = torch.randperm(args.num_train, device=device)
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batchsize:(b + 1) * args.batchsize]
            X_b, Y_b = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            score = (
                student_t_score_likelihood(X_b, Y_b, args.scale, args.nu)
                + gaussian_prior_score(X_b, mu_prior, Sigma_prior_inv)
            )
            g, _ = ensemble(X_b, Y_b, score)
            g = g.T
            loss = (g - X_b).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            optimizer.step()
            sched.step()
            ep_loss += loss.item()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  Epoch {ep}: train_loss = {ep_loss / n_batches:.4f}, lr = {optimizer.param_groups[0]['lr']:.2e}")
    ensemble.eval()

    # ── Per-obs CONDITIONAL evaluation via Langevin ──
    print(f"\n=== Per-obs conditional evaluation ({args.num_test_obs} test obs) ===")
    cncv_vrf_per_obs = []
    poly1_vrf_per_obs = []

    def polynomial_basis_d1(X, score):
        d = X.shape[1]
        return torch.cat(
            [score]
            + [(1.0 + X[:, k:k + 1] * score[:, k:k + 1]) for k in range(d)],
            dim=1,
        )

    def polynomial_cv_lsq(X, score, h_x, ridge=1e-4):
        G = polynomial_basis_d1(X, score)
        n_basis = G.shape[1]
        beta = torch.linalg.solve(
            G.T @ G + ridge * torch.eye(n_basis, device=X.device),
            G.T @ h_x,
        )
        return G @ beta

    for i in range(args.num_test_obs):
        # Draw a test (X_true, Y_obs)
        X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        Y_obs = sample_student_t(
            X_true.unsqueeze(0), args.scale, args.nu, device
        ).squeeze(0)

        # Sample X from p(x|Y_obs) via Langevin
        X_samples = langevin_posterior_simple(
            Y_obs, args.num_samples_per_obs,
            mu_prior, Sigma_prior_inv, args.scale, args.nu,
            device, n_steps=args.langevin_steps,
            step_size=args.langevin_step_size,
        )
        Y_rep = Y_obs.unsqueeze(0).expand(args.num_samples_per_obs, -1)

        # Score at conditional samples
        score = (
            student_t_score_likelihood(X_samples, Y_rep, args.scale, args.nu)
            + gaussian_prior_score(X_samples, mu_prior, Sigma_prior_inv)
        )

        h_x = X_samples
        with torch.no_grad():
            g_cncv, _ = ensemble(X_samples, Y_rep, score)
            g_cncv = g_cncv.T
        g_poly1 = polynomial_cv_lsq(X_samples, score, h_x, ridge=1e-4)

        var_h = h_x.var(dim=0)
        vrf_cncv = (h_x - g_cncv).var(dim=0) / torch.clamp(var_h, min=1e-12)
        vrf_poly1 = (h_x - g_poly1).var(dim=0) / torch.clamp(var_h, min=1e-12)
        cncv_vrf_per_obs.append(vrf_cncv.cpu().numpy())
        poly1_vrf_per_obs.append(vrf_poly1.cpu().numpy())

        if i < 3 or i % 5 == 0:
            print(
                f"  obs {i + 1}/{args.num_test_obs}: "
                f"CNCV VRF = {vrf_cncv.mean().item():.4f}, "
                f"Poly-1 VRF = {vrf_poly1.mean().item():.4f}"
            )

    cncv_arr = np.array(cncv_vrf_per_obs)  # [n_obs, d]
    poly1_arr = np.array(poly1_vrf_per_obs)
    cncv_per_comp = cncv_arr.mean(axis=0)
    poly1_per_comp = poly1_arr.mean(axis=0)
    cncv_per_obs_scalar = cncv_arr.mean(axis=1)
    poly1_per_obs_scalar = poly1_arr.mean(axis=1)

    summary = {
        "nu": args.nu, "scale": args.scale, "dim": n_dim,
        "epochs": args.epochs, "n_hidden": args.n_hidden,
        "depth": args.depth, "n_mlp_layers": args.n_mlp_layers,
        "num_train": args.num_train,
        "langevin_steps": args.langevin_steps,
        "langevin_step_size": args.langevin_step_size,
        "cncv_per_component": cncv_per_comp.tolist(),
        "poly1_per_component": poly1_per_comp.tolist(),
        "cncv_per_obs_mean": float(cncv_per_obs_scalar.mean()),
        "cncv_per_obs_std": float(cncv_per_obs_scalar.std()),
        "poly1_per_obs_mean": float(poly1_per_obs_scalar.mean()),
        "poly1_per_obs_std": float(poly1_per_obs_scalar.std()),
        "frac_below_1_cncv": float((cncv_per_obs_scalar < 1).mean()),
        "frac_below_1_poly1": float((poly1_per_obs_scalar < 1).mean()),
    }

    print("\n=== Results ===")
    print(f"Student-t(nu={args.nu}, scale={args.scale}), per-obs conditional VRF (Langevin posterior):")
    print(f"  CNCV:   mean = {summary['cncv_per_obs_mean']:.4f} ± {summary['cncv_per_obs_std']:.4f},  frac<1 = {summary['frac_below_1_cncv']:.2f}")
    print(f"  Poly-1: mean = {summary['poly1_per_obs_mean']:.4f} ± {summary['poly1_per_obs_std']:.4f},  frac<1 = {summary['frac_below_1_poly1']:.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_dir / 'summary.json'}")

    # Plot
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    methods = ["CNCV\n(amortized)", "Poly-1\n(per-obs LSQ)"]
    x = [0, 1]
    means = [summary["cncv_per_obs_mean"], summary["poly1_per_obs_mean"]]
    stds = [summary["cncv_per_obs_std"], summary["poly1_per_obs_std"]]
    bars = ax.bar(x, means, width=0.5, yerr=stds, color=["C0", "C3"],
                  capsize=4, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(1.0, color="gray", ls="--", lw=1, zorder=2,
               label="VRF $=1$ (no reduction)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.2)
    ax.set_ylabel("Per-observation VRF")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.08, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y", zorder=0)
    fig.tight_layout()
    fig.savefig(out_dir / "student_t_proper.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "student_t_proper.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'student_t_proper.pdf'}")


if __name__ == "__main__":
    main()
