"""Oh-style unconditional neural control variate baseline.

Reproduces the Stein-based neural CV from Bedaque & Oh (2024) and Oh (2025),
both cited in the paper (lattice field theory). Direct contender to CNCV
because it uses the same Stein construction but is **unconditional** —
the network phi(x) is trained per single test observation y, not amortized
across y.

Architecture: simple MLP phi: R^d -> R^d. We do not replicate the lattice-
specific symmetries of the original, as they are not relevant to a fair
comparison on our problems.

Stein CV:
    g(x) = nabla . phi(x) + phi(x) . score(x)

Loss: || h(x) - g(x) ||^2 averaged over MC samples from a single posterior.

Output:
  figs/baselines/oh_unconditional_comparison_d{D}.pdf
  figs/baselines/oh_unconditional_results_d{D}.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cncv import compute_score_posterior_gaussian
from cncv.models import create_shared_hierarchical_ensemble
from run_baselines import polynomial_cv


class OhUnconditionalNCV(nn.Module):
    """phi: R^d -> R^d MLP. Unconditional — does not see observation y.

    Stein CV g(x) = div(phi)(x) + phi(x) . score(x)
    Computed via autograd: trace(jacobian) gives divergence.
    """

    def __init__(self, d: int, n_hidden: int = 128, n_layers: int = 4,
                 activation: str = "tanh"):
        super().__init__()
        self.d = d
        layers = [nn.Linear(d, n_hidden)]
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers.append(act())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(act())
        final = nn.Linear(n_hidden, d)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.phi = nn.Sequential(*layers)

    def forward(self, x):
        return self.phi(x)

    def stein_cv(self, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Component-wise Stein CV: g_j(x) = d phi_j / d x_j + phi_j . score_j.

        Note: the original Bedaque/Oh paper sums over j to get a scalar CV.
        For component-wise comparison with CNCV, we compute per-j Stein:
          g_j = d_phi_j/d_x_j + phi_j(x) * score_j(x)
        which gives a vector g of shape [B, d].
        """
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            phi_x = self.phi(x)  # [B, d]
            # Diagonal of Jacobian: d phi_j / d x_j for each j
            # Use autograd.grad with grad_outputs being one-hot per j
            B, d = phi_x.shape
            diag = torch.zeros_like(phi_x)
            for j in range(d):
                grad_outputs = torch.zeros_like(phi_x)
                grad_outputs[:, j] = 1.0
                gj = torch.autograd.grad(
                    phi_x, x, grad_outputs=grad_outputs,
                    create_graph=self.training, retain_graph=True,
                )[0]  # [B, d]
                diag[:, j] = gj[:, j]
        return diag.detach() + phi_x.detach() * score


def train_oh_per_obs(model, X_samples, score, h_x, n_epochs=200, lr=1e-3,
                     batch_size=512, device="cuda:0", verbose=False):
    """Train Oh's unconditional NCV on samples from a single posterior."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    N = X_samples.shape[0]
    losses = []
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batch = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb, sb, hb = X_samples[idx], score[idx], h_x[idx]
            xb_g = xb.detach().requires_grad_(True)
            phi_x = model.phi(xb_g)
            B, d = phi_x.shape
            diag = torch.zeros_like(phi_x)
            for j in range(d):
                grad_outputs = torch.zeros_like(phi_x)
                grad_outputs[:, j] = 1.0
                gj = torch.autograd.grad(
                    phi_x, xb_g, grad_outputs=grad_outputs,
                    create_graph=True, retain_graph=True,
                )[0]
                diag[:, j] = gj[:, j]
            g = diag + phi_x * sb
            loss = ((hb - g) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        sched.step()
        losses.append(epoch_loss / n_batch)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}: loss={losses[-1]:.4f}")
    model.eval()
    return losses


def find_paper_cncv_checkpoint(dim, qofi="mean"):
    import re
    base = "data/checkpoints"
    pattern = f"gaussian_ensemble_cv_input_dim-{dim}_"
    all_dirs = [d for d in os.listdir(base) if d.startswith(pattern)
                and f"qofi-{qofi}" in d]
    matches = [d for d in all_dirs
               if re.search(r"\.[0-9a-f]{40}$", d)
               and "n_ensemble_members-16" in d
               and "max_epochs-50" in d
               and "n_hidden-64" in d]
    if not matches:
        matches = [d for d in all_dirs if "n_ensemble_members-16" in d]
    if not matches:
        return None
    nonempty = [m for m in matches
                if any(f.startswith("checkpoint_") and f.endswith(".pth")
                       for f in os.listdir(os.path.join(base, m)))]
    if not nonempty:
        return None
    chosen = sorted(nonempty)[-1]
    ckpt_dir = os.path.join(base, chosen)
    ckpt_files = [f for f in os.listdir(ckpt_dir)
                  if f.startswith("checkpoint_") and f.endswith(".pth")]
    latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckpt_dir, latest)


def gaussian_setup(dim, sigma, device):
    mu_prior = torch.zeros(dim, dtype=torch.float32, device=device)
    torch.manual_seed(42)
    diag_values = torch.linspace(1.0, 2.0, dim)
    D = torch.diag(diag_values).to(device=device, dtype=torch.float32)
    if dim > 1:
        A = torch.randn(dim, dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        Sigma_prior = Q @ D @ Q.T
    else:
        Sigma_prior = D
    L_prior = torch.linalg.cholesky(Sigma_prior)
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    I = torch.eye(dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1.0 / sigma ** 2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)
    return mu_prior, Sigma_prior, Sigma_prior_inv, L_prior, Sigma_post, L_post


def vrf(h_x, g):
    var_h = h_x.var(dim=0)
    var_hg = (h_x - g).var(dim=0)
    return (var_hg / var_h.clamp(min=1e-12)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--n_test_obs", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--oh_hidden", type=int, default=128)
    parser.add_argument("--oh_layers", type=int, default=4)
    parser.add_argument("--oh_epochs", type=int, default=200)
    parser.add_argument("--oh_lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str,
                        default="figs/baselines")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, dim={args.dim}")

    mu_prior, Sigma_prior, Sigma_prior_inv, L_prior, Sigma_post, L_post = \
        gaussian_setup(args.dim, args.sigma, device)

    cncv_ckpt = find_paper_cncv_checkpoint(args.dim)
    cncv = None
    if cncv_ckpt:
        ckpt = torch.load(cncv_ckpt, map_location=device, weights_only=False)
        args_ckpt = ckpt["args"]
        cncv = create_shared_hierarchical_ensemble(
            args.dim, args.dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
            n_layers=args_ckpt.n_layers,
            depth=getattr(args_ckpt, "depth", None) or None,
            n_cv=args.dim, seed=12,
        ).to(device)
        cncv.load_state_dict(ckpt["ensemble_state_dict"])
        cncv.eval()
        print(f"Loaded CNCV: {os.path.basename(cncv_ckpt)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = ["cncv", "poly1", "poly2", "oh"]
    vrfs = {m: [] for m in methods}
    times = {m: [] for m in methods}

    for i in tqdm(range(args.n_test_obs), desc=f"d={args.dim}"):
        X_true = mu_prior + L_prior @ torch.randn(args.dim, device=device)
        Y_obs = X_true + args.sigma * torch.randn(args.dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / args.sigma ** 2) * Y_obs)
        noise = torch.randn(args.num_samples, args.dim, device=device)
        X = mu_post + (L_post @ noise.T).T
        Y_rep = Y_obs.repeat(args.num_samples, 1)
        score = compute_score_posterior_gaussian(X, Y_rep, mu_prior, Sigma_prior_inv, args.sigma)
        h = X

        if cncv is not None:
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                g_combined, _ = cncv(X, Y_rep, score)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            times["cncv"].append(t1 - t0)
            vrfs["cncv"].append(vrf(h, g_combined.T))

        # Poly1
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        g_poly1 = polynomial_cv(X, score, h, degree=1)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times["poly1"].append(time.perf_counter() - t0)
        vrfs["poly1"].append(vrf(h, g_poly1))

        # Poly2
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        g_poly2 = polynomial_cv(X, score, h, degree=2)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times["poly2"].append(time.perf_counter() - t0)
        vrfs["poly2"].append(vrf(h, g_poly2))

        # Oh: per-obs train + eval
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        oh = OhUnconditionalNCV(args.dim, n_hidden=args.oh_hidden,
                                  n_layers=args.oh_layers).to(device)
        train_oh_per_obs(oh, X, score, h,
                          n_epochs=args.oh_epochs, lr=args.oh_lr,
                          device=device)
        with torch.no_grad():
            # Recompute g (no grad needed since model is trained)
            x_g = X.detach().requires_grad_(True)
            with torch.enable_grad():
                phi_x = oh.phi(x_g)
                B, d = phi_x.shape
                diag = torch.zeros_like(phi_x)
                for j in range(d):
                    grad_outputs = torch.zeros_like(phi_x)
                    grad_outputs[:, j] = 1.0
                    gj = torch.autograd.grad(
                        phi_x, x_g, grad_outputs=grad_outputs,
                        retain_graph=True,
                    )[0]
                    diag[:, j] = gj[:, j]
            g_oh = (diag + phi_x * score).detach()
        torch.cuda.synchronize() if device.type == "cuda" else None
        times["oh"].append(time.perf_counter() - t0)
        vrfs["oh"].append(vrf(h, g_oh))

    print(f"\n=== Results d={args.dim} ===")
    print(f"{'Method':<8} {'VRF mean':>10} {'VRF std':>10} {'time(s)':>10}")
    for m in methods:
        if not vrfs[m]:
            continue
        print(f"{m:<8} {np.mean(vrfs[m]):>10.4f} {np.std(vrfs[m]):>10.4f} "
              f"{np.mean(times[m]):>10.3f}")

    # Save
    save = {
        "vrfs": {m: vrfs[m] for m in methods if vrfs[m]},
        "times": {m: times[m] for m in methods if times[m]},
        "vrf_mean": {m: float(np.mean(vrfs[m])) for m in methods if vrfs[m]},
        "vrf_std": {m: float(np.std(vrfs[m])) for m in methods if vrfs[m]},
        "time_mean": {m: float(np.mean(times[m])) for m in methods if times[m]},
        "args": vars(args),
    }
    out_json = os.path.join(args.output_dir, f"oh_unconditional_results_d{args.dim}.json")
    with open(out_json, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Figure ──
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "figure.dpi": 150, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    method_labels = {"cncv": "CNCV (ours)", "poly1": "Poly-1", "poly2": "Poly-2", "oh": "Oh NCV (per-obs)"}
    method_colors = {"cncv": "#4C72B0", "poly1": "#DD8452", "poly2": "#55A868", "oh": "#8172B2"}

    valid = [m for m in methods if vrfs[m]]
    means = [np.mean(vrfs[m]) for m in valid]
    stds = [np.std(vrfs[m]) for m in valid]
    axes[0].bar(range(len(valid)), means, yerr=stds, capsize=4,
                color=[method_colors[m] for m in valid], alpha=0.85)
    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.6)
    axes[0].set_xticks(range(len(valid)))
    axes[0].set_xticklabels([method_labels[m] for m in valid], rotation=15)
    axes[0].set_ylabel("VRF (lower better)")
    axes[0].set_yscale("log")
    axes[0].set_title(f"(a) VRF, Gaussian d={args.dim}")
    axes[0].grid(alpha=0.2, axis="y", which="both")

    times_s = [np.mean(times[m]) for m in valid]
    axes[1].bar(range(len(valid)), times_s,
                color=[method_colors[m] for m in valid], alpha=0.85)
    axes[1].set_xticks(range(len(valid)))
    axes[1].set_xticklabels([method_labels[m] for m in valid], rotation=15)
    axes[1].set_ylabel("Per-obs time (s)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Per-obs cost")
    axes[1].grid(alpha=0.2, axis="y", which="both")

    plt.tight_layout()
    out = os.path.join(args.output_dir, f"oh_unconditional_comparison_d{args.dim}.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
