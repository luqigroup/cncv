"""Train CNF + CNCV pipeline for Darcy flow in KL space.

Two-phase pipeline:
  - Score norm monitoring
  - Configurable checkpoint frequency (default every 10 epochs)
  - NaN detection: stops training and saves last good checkpoint
  - Best-model tracking: saves cnf_best.pth / cncv_best.pth

Usage:
    # Phase 1: Train CNF
    python scripts/train_darcy_kl.py --phase cnf --gpu 0 \
        --config configs/darcy_kl_cnf.json

    # Phase 2: Train CNCV
    python scripts/train_darcy_kl.py --phase cncv --gpu 0 \
        --config configs/darcy_kl_cncv.json

    # Both phases
    python scripts/train_darcy_kl.py --phase both --gpu 0 \
        --config configs/darcy_kl_cnf.json
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import create_shared_ensemble, SharedEnsembleCVWrapper
from cncv.utils import CustomLRScheduler
from cncv.utils.download import ensure_dataset


def load_data(data_path, device="cpu"):
    """Load V2 KL-space Darcy data."""
    print(f"Loading data from {data_path}")
    ensure_dataset(data_path)
    d = np.load(data_path)
    z_key = "u_kl" if "u_kl" in d.files else "z"
    z = torch.tensor(d[z_key], dtype=torch.float32)
    y = torch.tensor(d["y_obs"], dtype=torch.float32)
    print(f"  z: {z.shape}, y: {y.shape}")
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    if "version" in d:
        print(f"  version: {d['version']}")
    return z, y


def _save_checkpoint(path, **kwargs):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)
    print(f"  Saved checkpoint: {path}")


# =============================================================================
# Phase 1: Train CNF
# =============================================================================

def train_cnf(args):
    """Train conditional normalizing flow on (z, y) joint samples."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Phase 1: Training CNF (V2 Beskos)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Load data
    z_all, y_all = load_data(args.data_path)
    N = z_all.shape[0]
    d_z = z_all.shape[1]
    d_y = y_all.shape[1]

    # Train/val split
    num_train = min(args.cnf_num_train, N - args.cnf_num_val)
    num_val = args.cnf_num_val
    z_train, y_train = z_all[:num_train], y_all[:num_train]
    z_val, y_val = z_all[num_train:num_train + num_val], y_all[num_train:num_train + num_val]
    print(f"Train: {num_train}, Val: {num_val}")

    # Normalize y
    y_mean = y_train.mean(0)
    y_std = y_train.std(0).clamp(min=1e-6)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    # Create CNF
    cnf_depth = args.cnf_depth if args.cnf_depth > 0 else None
    cnf = ConditionalNF(
        n_in=d_z, n_cond=d_y,
        n_hidden=args.cnf_hidden,
        n_flow_layers=args.cnf_flow_layers,
        depth=cnf_depth,
        n_mlp_layers=args.cnf_mlp_layers,
    ).to(device)

    n_params = sum(p.numel() for p in cnf.parameters() if p.requires_grad)
    print(f"\nCNF: {args.cnf_flow_layers} flow layers, depth={cnf.depth}, "
          f"hidden={args.cnf_hidden}, mlp_layers={args.cnf_mlp_layers}")
    print(f"Total parameters: {n_params:,}")

    # Optimizer + scheduler
    wd = getattr(args, "cnf_weight_decay", 0.0)
    optimizer = torch.optim.AdamW(cnf.parameters(), lr=args.cnf_lr, weight_decay=wd)
    if wd > 0:
        print(f"Using AdamW with weight_decay={wd}")
    num_batches = num_train // args.cnf_batchsize
    max_step = num_batches * args.cnf_epochs
    lr_sched = CustomLRScheduler(optimizer, args.cnf_lr, args.cnf_lr_final, max_step)

    grad_clip = getattr(args, "cnf_grad_clip", 10.0)
    save_freq = getattr(args, "checkpoint_freq", 10)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(z_train, y_train_n)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.cnf_batchsize, shuffle=True,
        pin_memory=True, num_workers=2,
    )
    val_ds = torch.utils.data.TensorDataset(z_val, y_val_n)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.cnf_batchsize, shuffle=False,
        pin_memory=True, num_workers=2,
    )

    print(f"\nTraining: {args.cnf_epochs} epochs, bs={args.cnf_batchsize}, "
          f"{num_batches} batches/epoch")
    print(f"LR: {args.cnf_lr} -> {args.cnf_lr_final}, grad_clip={grad_clip}")
    print(f"Checkpoints every {save_freq} epochs")

    train_losses = []
    val_losses = []
    score_norms = []
    ckpt_dir = args.cnf_ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    last_good_epoch = -1
    t0 = time.time()

    def _make_ckpt_kwargs(epoch):
        return dict(
            model_state_dict=cnf.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            score_norms=score_norms,
            y_mean=y_mean, y_std=y_std,
            d_z=d_z, d_y=d_y, n_params=n_params,
            cnf_hidden=args.cnf_hidden,
            cnf_flow_layers=args.cnf_flow_layers,
            cnf_depth=args.cnf_depth if args.cnf_depth else cnf.depth,
            cnf_mlp_layers=args.cnf_mlp_layers,
        )

    for epoch in tqdm(range(args.cnf_epochs), desc="CNF", unit="ep"):
        # Validation
        cnf.eval()
        val_nll = 0.0
        epoch_score_norms = []
        with torch.no_grad():
            for zb, yb in val_loader:
                zb, yb = zb.to(device), yb.to(device)
                val_nll += -cnf.log_prob(zb, yb).mean().item()

        # Score norm monitoring (on first val batch)
        zb_check, yb_check = next(iter(val_loader))
        zb_check, yb_check = zb_check[:256].to(device), yb_check[:256].to(device)
        s = cnf.score(zb_check, yb_check)
        s_norms = s.norm(dim=-1)
        mean_score_norm = s_norms.mean().item()
        max_score_norm = s_norms.max().item()
        score_norms.append(mean_score_norm)

        val_nll /= len(val_loader)
        val_losses.append(val_nll)

        # NaN detection
        if np.isnan(val_nll):
            tqdm.write(f"  NaN detected at epoch {epoch}! Stopping training.")
            if last_good_epoch >= 0:
                tqdm.write(f"  Last good checkpoint: epoch {last_good_epoch}")
            break

        # Best model tracking
        if val_nll < best_val:
            best_val = val_nll
            _save_checkpoint(
                os.path.join(ckpt_dir, "cnf_best.pth"),
                **_make_ckpt_kwargs(epoch),
            )

        last_good_epoch = epoch

        # Training
        cnf.train()
        epoch_loss = 0.0
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            loss = -cnf.log_prob(zb, yb).mean()

            if torch.isnan(loss):
                tqdm.write(f"  NaN in training loss at epoch {epoch}! Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), grad_clip)
            optimizer.step()
            lr_sched.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())
        else:
            epoch_loss /= num_batches

            # Checkpoint
            if epoch % save_freq == 0 or epoch == args.cnf_epochs - 1:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"cnf_epoch_{epoch}.pth"),
                    **_make_ckpt_kwargs(epoch),
                )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - t0
                eta = elapsed / (epoch + 1) * (args.cnf_epochs - epoch - 1)
                tqdm.write(
                    f"  Epoch {epoch}: train_nll={epoch_loss:.4f}, "
                    f"val_nll={val_nll:.4f}, score_norm={mean_score_norm:.1f}/"
                    f"{max_score_norm:.1f}, lr={lr_sched.compute_lr():.6f}, "
                    f"elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min")
            continue
        # Break outer loop if inner loop broke (NaN)
        break

    total_time = time.time() - t0
    print(f"\nCNF training done in {total_time/60:.1f} min")
    if train_losses:
        print(f"Final: train_nll={train_losses[-1]:.4f}, val_nll={val_losses[-1]:.4f}")
    print(f"Best val NLL: {best_val:.4f}")

    best_ckpt = os.path.join(ckpt_dir, "cnf_best.pth")
    return best_ckpt


# =============================================================================
# Phase 2: Train CNCV
# =============================================================================

def train_cncv(args, cnf_checkpoint=None):
    """Train shared ensemble CV using frozen CNF score."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Phase 2: Training CNCV (V2 Beskos)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    if cnf_checkpoint is None:
        cnf_checkpoint = args.cnf_checkpoint
    print(f"CNF checkpoint: {cnf_checkpoint}")

    # Load CNF
    ckpt = torch.load(cnf_checkpoint, map_location=device, weights_only=False)
    d_z = ckpt["d_z"]
    d_y = ckpt["d_y"]
    y_mean = ckpt["y_mean"].cpu()
    y_std = ckpt["y_std"].cpu()

    cnf = ConditionalNF(
        n_in=d_z, n_cond=d_y,
        n_hidden=ckpt["cnf_hidden"],
        n_flow_layers=ckpt["cnf_flow_layers"],
        depth=ckpt.get("cnf_depth", None),
        n_mlp_layers=ckpt["cnf_mlp_layers"],
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    cnf_params = sum(p.numel() for p in cnf.parameters())
    print(f"CNF loaded: epoch {ckpt['epoch']}, {cnf_params:,} params (frozen)")
    print(f"CNF val_nll at load: {ckpt['val_losses'][-1]:.4f}")

    # Load data
    z_all, y_all = load_data(args.data_path)
    N = z_all.shape[0]
    num_train = min(args.cncv_num_train, N - args.cncv_num_val)
    num_val = args.cncv_num_val

    z_train, y_train = z_all[:num_train], y_all[:num_train]
    z_val, y_val = z_all[num_train:num_train + num_val], y_all[num_train:num_train + num_val]

    # Normalize y
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    print(f"Train: {num_train}, Val: {num_val}")

    # Create ensemble CV
    depth = args.cncv_depth
    if depth is None or depth == 0:
        import math
        depth = max(1, int(math.log2(d_z))) if d_z > 1 else 0
    print(f"\nEnsemble: L={args.cncv_ensemble}, depth={depth}, "
          f"hidden={args.cncv_hidden}, mlp_layers={args.cncv_mlp_layers}")

    ensemble = create_shared_ensemble(
        n_in=d_z, n_cond=d_y, n_hidden=args.cncv_hidden,
        n_ensemble=args.cncv_ensemble, depth=depth,
        n_mlp_layers=args.cncv_mlp_layers, seed=12,
    )
    ensemble = SharedEnsembleCVWrapper(ensemble).to(device)

    n_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"Ensemble parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.cncv_lr)
    num_batches = num_train // args.cncv_batchsize
    max_step = num_batches * args.cncv_epochs
    lr_sched = CustomLRScheduler(optimizer, args.cncv_lr, args.cncv_lr_final, max_step)

    save_freq = getattr(args, "checkpoint_freq", 10)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(z_train, y_train_n)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.cncv_batchsize, shuffle=True,
        pin_memory=True, num_workers=2,
    )
    val_ds = torch.utils.data.TensorDataset(z_val, y_val_n)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.cncv_batchsize, shuffle=False,
        pin_memory=True, num_workers=2,
    )

    print(f"\nTraining: {args.cncv_epochs} epochs, bs={args.cncv_batchsize}, "
          f"{num_batches} batches/epoch")
    print(f"LR: {args.cncv_lr} -> {args.cncv_lr_final}")
    print(f"Checkpoints every {save_freq} epochs")

    train_losses = []
    val_losses = []
    score_norms = []
    ckpt_dir = args.cncv_ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    last_good_epoch = -1
    t0 = time.time()

    def _make_ckpt_kwargs(epoch):
        return dict(
            ensemble_state_dict=ensemble.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            score_norms=score_norms,
            cnf_checkpoint=cnf_checkpoint,
            y_mean=y_mean, y_std=y_std,
            d_z=d_z, d_y=d_y,
            n_ensemble=args.cncv_ensemble,
            cncv_hidden=args.cncv_hidden,
            cncv_depth=depth,
            cncv_mlp_layers=args.cncv_mlp_layers,
            n_params=n_params,
        )

    for epoch in tqdm(range(args.cncv_epochs), desc="CNCV", unit="ep"):
        # Validation
        ensemble.eval()
        val_loss = 0.0
        with torch.no_grad():
            for zb, yb in val_loader:
                zb, yb = zb.to(device), yb.to(device)
                score = cnf.score(zb, yb)
                g_combined, _ = ensemble(zb, yb, score)
                residual = zb.T - g_combined
                val_loss += (residual ** 2).mean().item()

        # Score norm monitoring
        zb_check, yb_check = next(iter(val_loader))
        zb_check, yb_check = zb_check[:256].to(device), yb_check[:256].to(device)
        s = cnf.score(zb_check, yb_check)
        mean_score_norm = s.norm(dim=-1).mean().item()
        score_norms.append(mean_score_norm)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # NaN detection
        if np.isnan(val_loss):
            tqdm.write(f"  NaN detected at epoch {epoch}! Stopping training.")
            if last_good_epoch >= 0:
                tqdm.write(f"  Last good checkpoint: epoch {last_good_epoch}")
            break

        # Best model tracking
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(
                os.path.join(ckpt_dir, "cncv_best.pth"),
                **_make_ckpt_kwargs(epoch),
            )

        last_good_epoch = epoch

        # Training
        ensemble.train()
        epoch_loss = 0.0
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            score = cnf.score(zb, yb)
            g_combined, _ = ensemble(zb, yb, score)
            residual = zb.T - g_combined
            loss = (residual ** 2).mean()

            if torch.isnan(loss):
                tqdm.write(f"  NaN in training loss at epoch {epoch}! Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 10.0)
            optimizer.step()
            lr_sched.step()

            epoch_loss += loss.item()
            train_losses.append(loss.item())
        else:
            epoch_loss /= num_batches

            # Checkpoint
            if epoch % save_freq == 0 or epoch == args.cncv_epochs - 1:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"cncv_epoch_{epoch}.pth"),
                    **_make_ckpt_kwargs(epoch),
                )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - t0
                eta = elapsed / (epoch + 1) * (args.cncv_epochs - epoch - 1)
                tqdm.write(
                    f"  Epoch {epoch}: train={epoch_loss:.6f}, "
                    f"val={val_loss:.6f}, score_norm={mean_score_norm:.1f}, "
                    f"lr={lr_sched.compute_lr():.6f}, "
                    f"elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min")
            continue
        break

    total_time = time.time() - t0
    print(f"\nCNCV training done in {total_time/60:.1f} min")
    if train_losses:
        print(f"Final: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")
    print(f"Best val loss: {best_val:.6f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Darcy KL-space CNF + CNCV training (Beskos V2)")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["cnf", "cncv", "both"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (overrides CLI args)")

    # CNF args
    parser.add_argument("--cnf_hidden", type=int, default=256)
    parser.add_argument("--cnf_flow_layers", type=int, default=16)
    parser.add_argument("--cnf_depth", type=int, default=6)
    parser.add_argument("--cnf_mlp_layers", type=int, default=3)
    parser.add_argument("--cnf_epochs", type=int, default=200)
    parser.add_argument("--cnf_batchsize", type=int, default=4096)
    parser.add_argument("--cnf_lr", type=float, default=1e-3)
    parser.add_argument("--cnf_lr_final", type=float, default=1e-5)
    parser.add_argument("--cnf_num_train", type=int, default=120000)
    parser.add_argument("--cnf_num_val", type=int, default=11072)
    parser.add_argument("--cnf_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cnf_grad_clip", type=float, default=10.0)
    parser.add_argument("--cnf_ckpt_dir", type=str,
                        default="data/darcy/checkpoints_cnf_kl_v2_beskos")

    # CNCV args
    parser.add_argument("--cnf_checkpoint", type=str, default=None,
                        help="Path to trained CNF checkpoint (for cncv phase)")
    parser.add_argument("--cncv_ensemble", type=int, default=32)
    parser.add_argument("--cncv_hidden", type=int, default=256)
    parser.add_argument("--cncv_depth", type=int, default=3)
    parser.add_argument("--cncv_mlp_layers", type=int, default=5)
    parser.add_argument("--cncv_epochs", type=int, default=400)
    parser.add_argument("--cncv_batchsize", type=int, default=4096)
    parser.add_argument("--cncv_lr", type=float, default=1e-4)
    parser.add_argument("--cncv_lr_final", type=float, default=1e-5)
    parser.add_argument("--cncv_num_train", type=int, default=120000)
    parser.add_argument("--cncv_num_val", type=int, default=11072)
    parser.add_argument("--cncv_ckpt_dir", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v2_beskos")

    # Shared
    parser.add_argument("--checkpoint_freq", type=int, default=10)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Save config
    config_dict = vars(args)
    os.makedirs("data/darcy", exist_ok=True)
    config_path = "data/darcy/darcy_kl_v2_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")

    if args.phase in ("cnf", "both"):
        cnf_ckpt = train_cnf(args)

    if args.phase in ("cncv", "both"):
        if args.phase == "both":
            train_cncv(args, cnf_checkpoint=cnf_ckpt)
        else:
            if args.cnf_checkpoint is None:
                raise ValueError("--cnf_checkpoint required for cncv phase")
            train_cncv(args)


if __name__ == "__main__":
    main()
