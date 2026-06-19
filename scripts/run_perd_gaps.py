"""Close the per-d cascade data gaps before replacing Table 1 / Fig 1.

The per-d sweep (run_per_d_hp_sweep.py) covered qofi=mean only and did not
complete d=16 c4_deepdata. To make per-d-tuned the headline consistently we also
need (a) the variance-estimation column under the SAME per-d recipe, and (b) the
d=16 high-data (c4_deepdata) run to close the "more data helps at every d" hole.

This trains those gap configs via gaussian_ensemble_cv.py (skipping any that
already exist, validated by ckpt['args']) and evaluates each over the honest
100-observation protocol (eval_vrf_100obs.evaluate_checkpoint_100obs). Writes one
JSON per (d, qofi) into results/per_d_hp/.

Per-d recipe = the mean-best from the sweep, applied to both QoIs:
  d=2: depth1 / d=4: depth2 / d=8: depth3 -- all h128 mlp5 N131072 100ep (c4_deepdata)
  d=16: depth3 h128 mlp5 N65536 100ep (c3_deep); c4_deepdata (N131072) also
        attempted for mean to test whether more data also wins at d=16.

Usage:
    python scripts/run_perd_gaps.py --dry_run
    python scripts/run_perd_gaps.py --gpu 0
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
    return min(3, max(1, int(math.log2(d))))


def c4(d: int) -> dict:
    """c4_deepdata recipe for dimension d (N=131072)."""
    return dict(depth=deep_depth(d), n_hidden=128, n_layers=5, max_epochs=100,
                lr=1e-4, lr_final=1e-5, num_train=131072, num_val=4096)


def c3_16() -> dict:
    """c3_deep recipe at d=16 (N=65536)."""
    return dict(depth=3, n_hidden=128, n_layers=5, max_epochs=100,
                lr=1e-4, lr_final=1e-5, num_train=65536, num_val=4096)


# Gap configs: (d, qofi, recipe). var for all 4 dims (per-d best recipe);
# plus the missing d=16 mean high-data run.
GAPS = [
    (2,  "var",  c4(2)),
    (4,  "var",  c4(4)),
    (8,  "var",  c4(8)),
    (16, "var",  c3_16()),
    (16, "mean", c4(16)),
]


def find_checkpoint(d: int, qofi: str, r: dict):
    """Locate a checkpoint matching (d, qofi, recipe) by validating ckpt['args']."""
    if not os.path.isdir(CKPT_BASE):
        return None
    toks = [
        f"gaussian_ensemble_cv_input_dim-{d}_",
        f"max_epochs-{r['max_epochs']}_",
        "n_ensemble_members-16_",
        f"n_hidden-{r['n_hidden']}_",
        f"n_layers-{r['n_layers']}_",
        f"num_train-{r['num_train']}_",
        f"qofi-{qofi[:3]}",   # qofi-mea (mean) or qofi-var
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


def train(d: int, qofi: str, r: dict, gpu: int):
    cmd = [
        sys.executable, os.path.join(REPO, "scripts", "gaussian_ensemble_cv.py"),
        "--phase", "train", "--input_dim", str(d), "--qofi", qofi,
        "--layer_type", "shared_hierarchical", "--depth", str(r["depth"]),
        "--n_hidden", str(r["n_hidden"]), "--n_layers", str(r["n_layers"]),
        "--max_epochs", str(r["max_epochs"]), "--lr", str(r["lr"]),
        "--lr_final", str(r["lr_final"]), "--num_train", str(r["num_train"]),
        "--num_val", str(r["num_val"]), "--save_freq", str(r["max_epochs"]),
        "--gpu_id", str(gpu),
    ]
    print(f">>> TRAIN d={d} qofi={qofi}")
    subprocess.run(cmd, check=True, cwd=REPO)


def main():
    p = argparse.ArgumentParser(description="Close per-d cascade gaps (var + d16 c4)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="results/per_d_hp")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for d, qofi, r in GAPS:
        ckpt = find_checkpoint(d, qofi, r)
        status = "REUSE" if ckpt else "TRAIN"
        tag = "c4_deepdata" if r["num_train"] == 131072 else "c3_deep"
        print(f"d={d:2d} qofi={qofi:4s} {tag:11s} depth={r['depth']} -> {status}  {ckpt or ''}")
        if args.dry_run:
            continue
        try:
            if ckpt is None:
                train(d, qofi, r, args.gpu)
                ckpt = find_checkpoint(d, qofi, r)
            if ckpt is None:
                print(f"!! no checkpoint after training d={d} qofi={qofi}; skipping")
                continue
            res = evaluate_checkpoint_100obs(ckpt, qofi=qofi, device=f"cuda:{args.gpu}")
            res["label"] = f"d{d}_{qofi}_{tag}"
            res["recipe"] = r
            out = os.path.join(out_dir, f"d{d}_{qofi}_{tag}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"   VRF(100-obs mean over components) = {res['vrf_mean']:.4f}")
        except Exception as e:
            print(f"!! d={d} qofi={qofi} failed: {e}")


if __name__ == "__main__":
    main()
