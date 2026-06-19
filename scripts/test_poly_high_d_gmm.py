"""Test polynomial Stein CV on high-d non-Gaussian distributions.

Self-contained script: defines DiagonalGMM inline.
Demonstrates polynomial Stein CV on Gaussian mixture targets at
d ∈ {10, 32, 64} to show where polynomial methods break down outside
the linear-Gaussian regime.

Score is computed via autograd on log_prob.

This is an UNCONDITIONAL test — there is no observation y. We measure VRF
for E[x] and E[x^2] estimation under the GMM target.

Output: figs/baselines/poly_high_d_gmm.pdf, .json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_baselines import polynomial_cv
from cncv.models.shared_cv import create_shared_ensemble

# Self-contained DiagonalGMM (avoids external imports for code release)
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
import torch.nn as nn


class OhUnconditionalNCV(nn.Module):
    """Unconditional Stein neural CV: phi(x) MLP, g = div(phi) + phi.score.

    Inline copy of the architecture used for the Gaussian d=4 comparison.
    Computes per-component Stein CV via O(d) autograd backward passes.
    """
    def __init__(self, d, n_hidden=128, n_layers=4, activation=nn.Tanh):
        super().__init__()
        self.d = d
        layers = [nn.Linear(d, n_hidden), activation()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_hidden, n_hidden), activation()]
        final = nn.Linear(n_hidden, d)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.phi = nn.Sequential(*layers)


def train_oh(model, x, score, h, n_epochs=100, lr=1e-3, batch_size=512):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    N = x.shape[0]
    d = x.shape[1]
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx].detach().requires_grad_(True)
            sb, hb = score[idx], h[idx]
            phi_x = model.phi(xb)
            B, dd = phi_x.shape
            diag = torch.zeros_like(phi_x)
            for j in range(dd):
                go = torch.zeros_like(phi_x); go[:, j] = 1.0
                gj = torch.autograd.grad(phi_x, xb, grad_outputs=go,
                                          create_graph=True, retain_graph=True)[0]
                diag[:, j] = gj[:, j]
            g = diag + phi_x * sb
            loss = ((hb - g) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()


def eval_oh_g(model, x, score):
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        phi_x = model.phi(x)
        B, d = phi_x.shape
        diag = torch.zeros_like(phi_x)
        for j in range(d):
            go = torch.zeros_like(phi_x); go[:, j] = 1.0
            gj = torch.autograd.grad(phi_x, x, grad_outputs=go,
                                      retain_graph=True)[0]
            diag[:, j] = gj[:, j]
    return (diag + phi_x * score).detach()


class DiagonalGMM:
    """Gaussian mixture model with diagonal covariances in d dimensions.

    Means are spaced along the first axis at separation * max(stds).
    """
    def __init__(self, d, variances, weights=None, separation=8.0):
        self.d = d
        n = len(variances)
        stds = torch.tensor(variances, dtype=torch.float32).sqrt()
        spacing = separation * stds.max()
        offsets = torch.arange(n, dtype=torch.float32) * spacing
        offsets = offsets - offsets.mean()
        means = torch.zeros(n, d)
        means[:, 0] = offsets
        if weights is None:
            weights = [1.0 / n] * n
        mix = Categorical(torch.tensor(weights))
        comp = Independent(Normal(means, stds.unsqueeze(-1).expand(-1, d)), 1)
        self.dist = MixtureSameFamily(mix, comp)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, n):
        return self.dist.sample((n,))


def compute_gmm_score(gmm, x):
    """Score: nabla_x log p(x) for the GMM target via autograd."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        logp = gmm.log_prob(x)
        grad = torch.autograd.grad(logp.sum(), x, create_graph=False)[0]
    return grad.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 32, 64])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--variances", type=float, nargs="+", default=[0.5, 2.0])
    parser.add_argument("--separation", type=float, default=8.0)
    parser.add_argument("--oh_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str,
                        default="figs/baselines")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Keep on CPU — torch.distributions GMM lives on CPU; small d makes this fine.
    device = torch.device("cpu")

    results = {}
    for d in args.dims:
        print(f"\n=== d={d} ===")
        results[d] = {"poly1_vrf": [], "poly2_vrf": [], "oh_vrf": [], "cncv_vrf": [],
                      "poly1_time": [], "poly2_time": [], "oh_time": [], "cncv_time": []}

        for seed in range(args.n_repeats):
            torch.manual_seed(seed * 7 + 123)
            gmm = DiagonalGMM(d=d, variances=args.variances,
                              separation=args.separation)
            x = gmm.sample(args.n_samples).to(device)  # [N, d]
            score = compute_gmm_score(gmm, x)
            h = x  # E[x] estimation
            var_h = h.var(dim=0)

            # Polynomial-1
            t0 = time.perf_counter()
            g_poly1 = polynomial_cv(x, score, h, degree=1)
            t1 = time.perf_counter()
            var_hg = (h - g_poly1).var(dim=0)
            vrf1 = (var_hg / var_h.clamp(min=1e-12)).mean().item()
            results[d]["poly1_vrf"].append(vrf1)
            results[d]["poly1_time"].append(t1 - t0)

            # Polynomial-2
            t0 = time.perf_counter()
            try:
                g_poly2 = polynomial_cv(x, score, h, degree=2)
                t1 = time.perf_counter()
                var_hg = (h - g_poly2).var(dim=0)
                vrf2 = (var_hg / var_h.clamp(min=1e-12)).mean().item()
                results[d]["poly2_vrf"].append(vrf2)
                results[d]["poly2_time"].append(t1 - t0)
            except RuntimeError as e:
                print(f"  poly2 failed at d={d}: {e}")
                results[d]["poly2_vrf"].append(float("nan"))
                results[d]["poly2_time"].append(float("nan"))

            # Oh-style unconditional neural CV
            t0 = time.perf_counter()
            oh = OhUnconditionalNCV(d=d, n_hidden=128, n_layers=4).to(device)
            train_oh(oh, x, score, h, n_epochs=args.oh_epochs, lr=1e-3,
                     batch_size=512)
            g_oh = eval_oh_g(oh, x, score)
            t1 = time.perf_counter()
            var_hg = (h - g_oh).var(dim=0)
            vrf_oh = (var_hg / var_h.clamp(min=1e-12)).mean().item()
            results[d]["oh_vrf"].append(vrf_oh)
            results[d]["oh_time"].append(t1 - t0)

            # Our architecture (unconditional hierarchical coupling).
            # Same Stein construction as our amortized CNCV but trained without
            # observation conditioning, so g(x) = diag(phi) + phi(x) . score(x)
            # with phi from a hierarchical coupling tree (exact triangular Jac).
            t0 = time.perf_counter()
            depth = max(1, int(np.log2(d)))
            ours = create_shared_ensemble(
                n_in=d, n_cond=1, n_hidden=128, n_ensemble=16,
                depth=min(3, depth), n_mlp_layers=4, seed=42,
            ).to(device)
            opt = torch.optim.Adam(ours.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.oh_epochs)
            dummy_y = torch.zeros(x.shape[0], 1, device=device)
            for ep in range(args.oh_epochs):
                perm = torch.randperm(x.shape[0])
                for i in range(0, x.shape[0], 512):
                    idx = perm[i:i + 512]
                    g = ours(x[idx], dummy_y[idx], score[idx])
                    loss = ((h[idx] - g) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(ours.parameters(), 5.0)
                    opt.step()
                sched.step()
            ours.eval()
            with torch.no_grad():
                g_ours = ours(x, dummy_y, score)
            t1 = time.perf_counter()
            var_hg = (h - g_ours).var(dim=0)
            vrf_ours = (var_hg / var_h.clamp(min=1e-12)).mean().item()
            results[d]["cncv_vrf"].append(vrf_ours)
            results[d]["cncv_time"].append(t1 - t0)

            print(f"  seed {seed}: poly1={vrf1:.3f}, poly2={results[d]['poly2_vrf'][-1]:.3f}, "
                  f"oh={vrf_oh:.3f} ({results[d]['oh_time'][-1]:.1f}s), "
                  f"ours={vrf_ours:.3f} ({results[d]['cncv_time'][-1]:.1f}s)")

        print(f"  poly1 mean VRF: {np.mean(results[d]['poly1_vrf']):.3f} +/- {np.std(results[d]['poly1_vrf']):.3f}")
        if not np.isnan(results[d]['poly2_vrf']).all():
            print(f"  poly2 mean VRF: {np.nanmean(results[d]['poly2_vrf']):.3f} +/- {np.nanstd(results[d]['poly2_vrf']):.3f}")
        print(f"  oh    mean VRF: {np.mean(results[d]['oh_vrf']):.3f} +/- {np.std(results[d]['oh_vrf']):.3f} "
              f"({np.mean(results[d]['oh_time']):.1f} s/run)")
        print(f"  ours  mean VRF: {np.mean(results[d]['cncv_vrf']):.3f} +/- {np.std(results[d]['cncv_vrf']):.3f} "
              f"({np.mean(results[d]['cncv_time']):.1f} s/run)")

    # Save
    save = {str(d): {k: list(map(float, v)) for k, v in results[d].items()}
            for d in args.dims}
    with open(os.path.join(args.output_dir, "poly_high_d_gmm.json"), "w") as f:
        json.dump(save, f, indent=2)

    # Figure
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "figure.dpi": 150, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    poly1_means = [np.mean(results[d]["poly1_vrf"]) for d in args.dims]
    poly1_stds = [np.std(results[d]["poly1_vrf"]) for d in args.dims]
    poly2_means = [np.nanmean(results[d]["poly2_vrf"]) for d in args.dims]
    poly2_stds = [np.nanstd(results[d]["poly2_vrf"]) for d in args.dims]
    oh_means = [np.mean(results[d]["oh_vrf"]) for d in args.dims]
    oh_stds = [np.std(results[d]["oh_vrf"]) for d in args.dims]
    cncv_means = [np.mean(results[d]["cncv_vrf"]) for d in args.dims]
    cncv_stds = [np.std(results[d]["cncv_vrf"]) for d in args.dims]
    ax.errorbar(args.dims, cncv_means, yerr=cncv_stds, marker="D", lw=2.0,
                capsize=4, color="#4C72B0", label="Ours (hierarchical coupling)")
    ax.errorbar(args.dims, oh_means, yerr=oh_stds, marker="^", lw=1.8,
                capsize=4, color="#8172B2", label="Oh-style neural CV")
    ax.errorbar(args.dims, poly1_means, yerr=poly1_stds, marker="o", lw=1.8,
                capsize=4, color="#DD8452", label="Poly-1 Stein CV")
    ax.errorbar(args.dims, poly2_means, yerr=poly2_stds, marker="s", lw=1.8,
                capsize=4, color="#55A868", label="Poly-2 Stein CV")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.6, label="VRF = 1")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("VRF (mean E[$x$] estimation)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(args.dims)
    ax.set_xticklabels([str(d) for d in args.dims])
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Polynomial Stein CV on GMM ({len(args.variances)}-component)")
    plt.tight_layout()
    out = os.path.join(args.output_dir, "poly_high_d_gmm.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
