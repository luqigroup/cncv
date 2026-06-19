"""Per-dimension hyperparameter sweep for the Gaussian CNCV (honest eval).

For each dimension and HP config, locate an existing matching checkpoint --
validated by loading ``ckpt['args']`` rather than by the (truncated) projorg
directory name -- or train one via gaussian_ensemble_cv.py, then evaluate VRF
over the honest 100-observation protocol
(eval_vrf_100obs.evaluate_checkpoint_100obs). One JSON per (d, config) is
written under results/per_d_hp/; summarize_per_d_hp.py picks the best config per
dimension and renders the comparison figure.

Checkpoint discovery validates depth / layer_type / ensemble size against the
loaded checkpoint, so configs that differ only in depth (c2 vs c3 at d>=8) are
never confused, and stale L=4/L=6 or wrong-recipe runs are never reused.

Configs (analytical score, seed 12, L=16, evaluated over 100 obs x 10k samples):
  c1_baseline : depth 2, h 64,  mlp 3, 50ep,  lr 1e-3->1e-4, N 65536 (paper recipe)
  c2_wide     : depth 2, h 128, mlp 5, 100ep, lr 1e-4->1e-5, N 65536
  c3_deep     : depth min(3,floor(log2 d)), h 128, mlp 5, 100ep, lr 1e-4->1e-5, N 65536
  c4_deepdata : depth min(3,floor(log2 d)), h 128, mlp 5, 100ep, lr 1e-4->1e-5, N 131072

Usage:
    python scripts/run_per_d_hp_sweep.py --dry_run        # show the plan only
    python scripts/run_per_d_hp_sweep.py --gpu 0          # train (as needed) + eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_vrf_100obs import evaluate_checkpoint_100obs

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_BASE = os.path.join(REPO, "data", "checkpoints")


def deep_depth(d: int) -> int:
    """Deep-config depth: min(3, floor(log2 d)); valid at every d."""
    return min(3, max(1, int(math.log2(d))))


def configs_for(d: int):
    dd = deep_depth(d)
    return [
        dict(label="c1_baseline", depth=2,  n_hidden=64,  n_layers=3,
             max_epochs=50,  lr=0.001,  lr_final=0.0001, num_train=65536,  num_val=2048),
        dict(label="c2_wide",     depth=2,  n_hidden=128, n_layers=5,
             max_epochs=100, lr=0.0001, lr_final=1e-05,  num_train=65536,  num_val=4096),
        dict(label="c3_deep",     depth=dd, n_hidden=128, n_layers=5,
             max_epochs=100, lr=0.0001, lr_final=1e-05,  num_train=65536,  num_val=4096),
        dict(label="c4_deepdata", depth=dd, n_hidden=128, n_layers=5,
             max_epochs=100, lr=0.0001, lr_final=1e-05,  num_train=131072, num_val=4096),
    ]


def find_checkpoint(d: int, r: dict):
    """Locate a checkpoint matching recipe r for dimension d, or None.

    Cheap pre-filter on the directory-name tokens that appear before projorg's
    truncation point (input_dim, max_epochs, n_ensemble, n_hidden, n_layers,
    num_train, and the 'qofi-mea' prefix that excludes qofi-var), then validate
    layer_type / depth / ensemble size by loading the checkpoint itself.
    """
    if not os.path.isdir(CKPT_BASE):
        return None
    toks = [
        f"gaussian_ensemble_cv_input_dim-{d}_",
        f"max_epochs-{r['max_epochs']}_",
        "n_ensemble_members-16_",
        f"n_hidden-{r['n_hidden']}_",
        f"n_layers-{r['n_layers']}_",
        f"num_train-{r['num_train']}_",
        "qofi-mea",  # matches qofi-mean and its truncation qofi-mea; excludes qofi-var
    ]
    cands = [dd for dd in os.listdir(CKPT_BASE) if all(t in dd for t in toks)]
    cands.sort(key=lambda dd: os.path.getmtime(os.path.join(CKPT_BASE, dd)), reverse=True)
    for dd in cands:
        run = os.path.join(CKPT_BASE, dd)
        cks = [f for f in os.listdir(run)
               if f.startswith("checkpoint_") and f.endswith(".pth")]
        if not cks:
            continue
        latest = max(cks, key=lambda f: int(f.split("_")[1].split(".")[0]))
        path = os.path.join(run, latest)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        a = ck.get("args")
        if (ck.get("layer_type") == "shared_hierarchical"
                and int(getattr(a, "depth", -1)) == int(r["depth"])
                and int(getattr(a, "n_ensemble_members", -1)) == 16):
            return path
    return None


def train(d: int, r: dict, gpu: int):
    cmd = [
        sys.executable, os.path.join(REPO, "scripts", "gaussian_ensemble_cv.py"),
        "--phase", "train", "--input_dim", str(d), "--qofi", "mean",
        "--layer_type", "shared_hierarchical", "--depth", str(r["depth"]),
        "--n_hidden", str(r["n_hidden"]), "--n_layers", str(r["n_layers"]),
        "--max_epochs", str(r["max_epochs"]), "--lr", str(r["lr"]),
        "--lr_final", str(r["lr_final"]), "--num_train", str(r["num_train"]),
        "--num_val", str(r["num_val"]), "--save_freq", str(r["max_epochs"]),
        "--gpu_id", str(gpu),
    ]
    print(f">>> TRAIN d={d} {r['label']}")
    subprocess.run(cmd, check=True, cwd=REPO)


def main():
    parser = argparse.ArgumentParser(description="Per-d Gaussian CNCV HP sweep")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/per_d_hp")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the train/reuse plan without training or evaluating")
    args = parser.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for d in args.dims:
        seen = set()
        for r in configs_for(d):
            key = (r["depth"], r["n_hidden"], r["n_layers"], r["max_epochs"], r["num_train"])
            dup = key in seen
            seen.add(key)
            ckpt = find_checkpoint(d, r)
            status = "REUSE" if ckpt else "TRAIN"
            print(f"d={d:2d} {r['label']:12s} depth={r['depth']} -> {status}"
                  f"{' (dup-recipe; eval only)' if dup else ''}  {ckpt or ''}")
            if args.dry_run:
                continue
            try:
                if ckpt is None:
                    train(d, r, args.gpu)
                    ckpt = find_checkpoint(d, r)
                if ckpt is None:
                    print(f"!! no checkpoint after training d={d} {r['label']}; skipping")
                    continue
                res = evaluate_checkpoint_100obs(ckpt, qofi="mean", device=f"cuda:{args.gpu}")
                res["label"] = f"d{d}_{r['label']}"
                res["config"] = r
                with open(os.path.join(out_dir, f"d{d}_{r['label']}.json"), "w") as f:
                    json.dump(res, f, indent=2)
                print(f"   VRF(100-obs mean) = {res['vrf_mean']:.4f}  "
                      f"(median over comp {res['vrf_median_over_components']:.4f})")
            except Exception as e:  # isolate per-config failures
                print(f"!! d={d} {r['label']} failed: {e}")

    if not args.dry_run:
        subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "summarize_per_d_hp.py"),
             "--results_dir", out_dir, "--output_dir", "figs/per_d_hp"],
            cwd=REPO, check=False)


if __name__ == "__main__":
    main()
