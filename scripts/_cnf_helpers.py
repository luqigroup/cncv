"""Shared loading utilities for CNF-based CNCV figure scripts.

Avoids duplicating checkpoint-finding and model-loading boilerplate
across generate_cnf_scatter_correlation.py, generate_cnf_sample_efficiency.py,
and run_cnf_dimension_scaling.py.
"""

import os
import re
import torch
import torch.nn as nn

from cncv.models.conditional_nf import ConditionalNF
from cncv.models import create_shared_hierarchical_ensemble
from cncv.utils.gaussian import compute_exact_posterior


def _find_cnf_checkpoint(dim=4):
    """Find the best CNF checkpoint for a given dimension.

    Looks in data/checkpoints for cnf_gaussian directories matching the dimension.
    Prefers the config with num_train=262144 (large dataset) if available.

    Returns:
        (checkpoint_path, cnf_args_from_dirname)
    """
    ckpt_base = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")
    ckpt_base = os.path.normpath(ckpt_base)
    pattern = f"cnf_gaussian_input_dim-{dim}_"

    all_dirs = [d for d in os.listdir(ckpt_base) if d.startswith(pattern)]
    if not all_dirs:
        raise FileNotFoundError(f"No CNF checkpoint found for dim={dim}")

    # Prefer large-dataset checkpoints
    large = [d for d in all_dirs if "num_train-262144" in d]
    candidates = large if large else all_dirs

    # Prefer largest n_hidden (better model)
    def _extract_hidden(dirname):
        for part in dirname.split("_"):
            if part.startswith("hidden-"):
                try:
                    return int(part.split("-")[1])
                except ValueError:
                    pass
        return 0

    candidates.sort(key=lambda d: (_extract_hidden(d), d))
    exp_name = candidates[-1]
    ckpt_path = os.path.join(ckpt_base, exp_name)

    ckpt_files = [
        f for f in os.listdir(ckpt_path)
        if f.startswith("checkpoint_") and f.endswith(".pth")
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {ckpt_path}")

    latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))

    # Parse config from directory name
    config = {}
    for part in exp_name.split("_"):
        if "-" in part:
            key, _, val = part.partition("-")
            try:
                config[key] = int(val)
            except ValueError:
                try:
                    config[key] = float(val)
                except ValueError:
                    config[key] = val

    return os.path.join(ckpt_path, latest), config


def _find_cnf_cv_checkpoint(dim=4):
    """Find the best CNCV (CNF-score) checkpoint for a given dimension.

    Looks for gaussian_cnf_cv directories matching the dimension.
    """
    ckpt_base = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")
    ckpt_base = os.path.normpath(ckpt_base)
    pattern = f"gaussian_cnf_cv_input_dim-{dim}_"

    all_dirs = [d for d in os.listdir(ckpt_base) if d.startswith(pattern)]
    if not all_dirs:
        raise FileNotFoundError(f"No CNCV-CNF checkpoint found for dim={dim}")

    # Prefer max_epochs-200 if available
    long_train = [d for d in all_dirs if "max_epochs-200" in d]
    candidates = long_train if long_train else all_dirs

    # Prefer most recently modified checkpoint directory
    exp_name = max(candidates, key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    ckpt_path = os.path.join(ckpt_base, exp_name)

    ckpt_files = [
        f for f in os.listdir(ckpt_path)
        if f.startswith("checkpoint_") and f.endswith(".pth")
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {ckpt_path}")

    latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckpt_path, latest)


def load_cnf_and_ensemble(dim=4, device="cuda:0"):
    """Load pre-trained CNF and CNCV ensemble for a given dimension.

    Args:
        dim: Problem dimension (default: 4).
        device: Torch device string.

    Returns:
        cnf: ConditionalNF model (eval, frozen).
        ensemble: SharedEnsembleCVWrapper (eval).
        mu_prior: Prior mean [dim].
        Sigma_prior: Prior covariance [dim, dim].
        sigma: Observation noise (float).
    """
    device = torch.device(device)

    # --- Load CNF ---
    cnf_ckpt_path, cnf_config = _find_cnf_checkpoint(dim)
    print(f"CNF checkpoint: {cnf_ckpt_path}")

    cnf = ConditionalNF(
        n_in=dim,
        n_cond=dim,
        n_hidden=cnf_config.get("hidden", 64),
        n_flow_layers=cnf_config.get("layers", 8),
        depth=cnf_config.get("depth", 2),
        n_mlp_layers=cnf_config.get("layers", 3),
    ).to(device)

    cnf_ckpt = torch.load(cnf_ckpt_path, map_location=device, weights_only=False)

    # The checkpoint stores the actual config used; prefer that
    if "model_state_dict" in cnf_ckpt:
        # Reconstruct with correct params from checkpoint args
        ckpt_args = cnf_ckpt.get("args", None)
        if ckpt_args is not None:
            cnf = ConditionalNF(
                n_in=dim,
                n_cond=dim,
                n_hidden=getattr(ckpt_args, "n_hidden", 64),
                n_flow_layers=getattr(ckpt_args, "n_flow_layers", 8),
                depth=getattr(ckpt_args, "depth", 2),
                n_mlp_layers=getattr(ckpt_args, "n_mlp_layers", 3),
            ).to(device)
        cnf.load_state_dict(cnf_ckpt["model_state_dict"])
    else:
        cnf.load_state_dict(cnf_ckpt)

    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    cnf_epoch = cnf_ckpt.get("epoch", "?")
    cnf_params = sum(p.numel() for p in cnf.parameters())
    print(f"CNF loaded (epoch {cnf_epoch}, {cnf_params:,} params, frozen)")

    # --- Load CNCV ensemble ---
    cv_ckpt_path = _find_cnf_cv_checkpoint(dim)
    print(f"CNCV checkpoint: {cv_ckpt_path}")

    cv_ckpt = torch.load(cv_ckpt_path, map_location=device, weights_only=False)
    train_args = cv_ckpt["args"]

    ensemble = create_shared_hierarchical_ensemble(
        dim, dim, train_args.n_hidden, train_args.n_ensemble_members,
        n_layers=train_args.n_layers,
        depth=getattr(train_args, "depth", None) or None,
        n_cv=dim, seed=12,
    ).to(device)
    ensemble.load_state_dict(cv_ckpt["ensemble_state_dict"])
    ensemble.eval()

    cv_epoch = cv_ckpt.get("epoch", "?")
    cv_params = sum(p.numel() for p in ensemble.parameters())
    print(f"CNCV loaded (epoch {cv_epoch}, {cv_params:,} params)")

    # Problem parameters from CNCV checkpoint
    mu_prior = cv_ckpt["mu_prior"].to(device)
    Sigma_prior = cv_ckpt["Sigma_prior"].to(device)
    sigma = cv_ckpt["sigma"]

    return cnf, ensemble, mu_prior, Sigma_prior, sigma


def setup_problem(mu_prior, Sigma_prior, sigma, dim, device):
    """Generate test observation and compute analytical posterior.

    Uses seed=42 for reproducibility (same as analytical-score figures).

    Args:
        mu_prior: Prior mean [dim].
        Sigma_prior: Prior covariance [dim, dim].
        sigma: Observation noise.
        dim: Problem dimension.
        device: Torch device.

    Returns:
        Y_obs: Test observation [dim].
        mu_post: Analytical posterior mean [dim].
        Sigma_post: Analytical posterior covariance [dim, dim].
    """
    torch.manual_seed(42)
    L_prior = torch.linalg.cholesky(Sigma_prior)
    X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
    Y_obs = X_true + sigma * torch.randn(dim, device=device)
    mu_post, Sigma_post = compute_exact_posterior(Y_obs, mu_prior, Sigma_prior, sigma)
    return Y_obs, mu_post, Sigma_post
