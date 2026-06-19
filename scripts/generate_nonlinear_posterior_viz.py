"""Generate the nonlinear posterior-visualization data for paper Fig. 6.

Fig. 6 (`plot_nonlinear_corner.py`) is a corner plot of the nonlinear forward
model F(x) = Ax + sin(x) posterior for a handful of observations. It consumes
`data/nonlinear/multi_posterior_viz.npz`, which previously had no committed
producer (the file was generated in an ad-hoc session). This script regenerates
that npz from the committed pipeline so the figure is reproducible from scratch.

Posterior sampling matches the canonical path used everywhere else for the
nonlinear problem -- the conditional DDPM (`run_nonlinear_forward.py`'s
`evaluate_nonlinear` with `use_ddpm=True`), since pSGLD does not converge for
this posterior. For each of `n_obs` deterministically drawn observations
y = F(x_true) + noise, we draw `num_samples` posterior samples by running the
reverse diffusion conditioned on y.

The npz keys consumed by the corner plot are `post{i}`, `true{i}` for
i in range(n_obs); `yobs{i}`, `X_prior`, and `A` are saved for completeness.

Note: the specific observations are seeded (`--obs_seed`), so the figure is
reproducible but the three posteriors need not be byte-identical to the original
ad-hoc npz; they are representative draws illustrating posterior shape and
amortization, which is what the figure conveys.

Usage:
    python scripts/generate_nonlinear_posterior_viz.py --gpu 0
    python scripts/generate_nonlinear_posterior_viz.py --gpu 0 --overwrite \
        --ddpm_checkpoint data/checkpoints/ddpm_nonlinear_.../checkpoint_999.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cncv.utils.nonlinear import setup_nonlinear_problem
from cncv.models.ddpm_mlp import ConditionalScoreMLPSimple
from cncv.diffusion import NoiseScheduler

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def find_ddpm_checkpoint(dim: int) -> str:
    """Auto-discover the most recent trained nonlinear DDPM checkpoint.

    Mirrors the projorg run-dir convention used elsewhere: pick the latest
    matching run directory that actually holds checkpoint files, then the
    highest-epoch checkpoint within it.
    """
    base = os.path.join(REPO, "data", "checkpoints")
    pattern = f"ddpm_nonlinear_input_dim-{dim}"

    def _has_checkpoints(name: str) -> bool:
        run_dir = os.path.join(base, name)
        return os.path.isdir(run_dir) and any(
            f.startswith("checkpoint_") and f.endswith(".pth")
            for f in os.listdir(run_dir)
        )

    runs = [d for d in os.listdir(base)
            if d.startswith(pattern) and _has_checkpoints(d)]
    if not runs:
        raise FileNotFoundError(
            f"No trained DDPM run found under {base} matching {pattern!r}. "
            f"Train one first: python scripts/ddpm_nonlinear.py"
        )
    run_dir = os.path.join(base, sorted(runs)[-1])
    ckpts = [f for f in os.listdir(run_dir)
             if f.startswith("checkpoint_") and f.endswith(".pth")]
    latest = max(ckpts, key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(run_dir, latest)


def load_ddpm(ddpm_checkpoint: str, dim: int, device):
    """Build and load the conditional DDPM + its noise scheduler.

    Construction mirrors `run_nonlinear_forward.py::evaluate_nonlinear` exactly
    (hidden 512, 5 layers, time-emb 64) so samples match the canonical pipeline.
    """
    ckpt = torch.load(ddpm_checkpoint, map_location=device, weights_only=False)
    model = ConditionalScoreMLPSimple(
        x_dim=dim, y_dim=dim,
        hidden_dim=512, n_layers=5, time_emb_dim=64,
        nt=ckpt.get("nt", 1000),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scheduler = NoiseScheduler(
        nt=ckpt.get("nt", 1000),
        beta_start=0.0001, beta_end=0.02,
        beta_schedule=ckpt.get("beta_schedule", "linear"),
        device=device,
    )
    return model, scheduler


def sample_ddpm_posterior(model, scheduler, y_obs, n, dim, device):
    """Draw `n` posterior samples by reverse diffusion conditioned on `y_obs`."""
    x = torch.randn(n, dim, device=device)
    y_batch = y_obs.unsqueeze(0).repeat(n, 1)
    with torch.no_grad():
        for t in reversed(range(scheduler.nt)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            noise_pred = model(x, y_batch, t_batch)
            x = scheduler.step(noise_pred, t, x)
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Generate nonlinear posterior-viz npz for paper Fig. 6")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--n_obs", type=int, default=3,
                        help="Number of observations to visualize")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Posterior samples per observation")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Observation noise std")
    parser.add_argument("--cond_number", type=float, default=2.0,
                        help="Condition number of the forward-model matrix A")
    parser.add_argument("--ddpm_checkpoint", type=str, default=None,
                        help="DDPM checkpoint; auto-discovered if omitted")
    parser.add_argument("--obs_seed", type=int, default=99,
                        help="Seed fixing which observations are drawn")
    parser.add_argument("--sample_seed", type=int, default=123,
                        help="Base seed for posterior/prior sampling")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default="data/nonlinear/multi_posterior_viz.npz")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output npz if it already exists")
    args = parser.parse_args()

    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Refusing to overwrite existing {out_path} (pass --overwrite). "
              f"This file backs paper Fig. 6.")
        return

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dim = args.dim
    problem = setup_nonlinear_problem(
        dim, sigma=args.sigma, cond_number=args.cond_number, device=device)
    mu_prior = problem["mu_prior"]
    L_prior = problem["L_prior"]
    forward_model = problem["forward_model"]

    ddpm_checkpoint = args.ddpm_checkpoint or find_ddpm_checkpoint(dim)
    print(f"Loading DDPM: {ddpm_checkpoint}")
    model, scheduler = load_ddpm(ddpm_checkpoint, dim, device)

    # Fix the observations first, independent of the sampling RNG, so obs
    # selection does not depend on num_samples.
    torch.manual_seed(args.obs_seed)
    trues, yobs = [], []
    for _ in range(args.n_obs):
        x_true = mu_prior + L_prior @ torch.randn(dim, device=device)
        y_obs = (forward_model.forward(x_true.unsqueeze(0)).squeeze(0)
                 + args.sigma * torch.randn(dim, device=device))
        trues.append(x_true)
        yobs.append(y_obs)

    # Prior reference samples.
    torch.manual_seed(args.sample_seed)
    noise = torch.randn(args.num_samples, dim, device=device)
    x_prior = mu_prior + (L_prior @ noise.T).T

    out = {
        "A": forward_model.A.cpu().numpy(),
        "X_prior": x_prior.cpu().numpy(),
        "n_obs": args.n_obs,
        "dim": dim,
        "sigma": args.sigma,
        "cond_number": args.cond_number,
    }
    for i in range(args.n_obs):
        torch.manual_seed(args.sample_seed + 1 + i)
        post = sample_ddpm_posterior(
            model, scheduler, yobs[i], args.num_samples, dim, device)
        out[f"post{i}"] = post.cpu().numpy()
        out[f"true{i}"] = trues[i].cpu().numpy()
        out[f"yobs{i}"] = yobs[i].cpu().numpy()
        print(f"  obs {i}: posterior mean {post.mean(0).cpu().numpy().round(3)} "
              f"| true {trues[i].cpu().numpy().round(3)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
