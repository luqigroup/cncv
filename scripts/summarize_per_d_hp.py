"""Summarize the per-d HP sweep: best config per dimension + comparison figure.

Reads the per-(dimension, config) VRF JSONs written by run_per_d_hp_sweep.py
(via eval_vrf_100obs), picks the best config per dimension by the honest
100-observation mean VRF, prints a baseline-vs-tuned table, and renders VRF
against dimension. All numbers are loaded from the evaluation JSONs.

Usage:
    python scripts/summarize_per_d_hp.py \
        --results_dir results/per_d_hp --output_dir figs/per_d_hp
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12, "font.family": "serif",
    "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
})


def main():
    parser = argparse.ArgumentParser(description="Summarize the per-d HP sweep")
    parser.add_argument("--results_dir", type=str, default="results/per_d_hp")
    parser.add_argument("--output_dir", type=str, default="figs/per_d_hp")
    args = parser.parse_args()

    recs = {}  # d -> {label: {vrf_mean, vrf_median, recipe}}
    for f in sorted(glob.glob(os.path.join(args.results_dir, "d*_*.json"))):
        r = json.load(open(f))
        d = int(r["n_dim"])
        label = r.get("label", os.path.basename(f)).split("_", 1)[-1]
        recs.setdefault(d, {})[label] = {
            "vrf_mean": float(r["vrf_mean"]),
            "vrf_median": float(r.get("vrf_median_over_components", float("nan"))),
            "recipe": r.get("config", {}),
        }
    if not recs:
        print(f"No result JSONs found in {args.results_dir}")
        return

    dims = sorted(recs)
    baseline, best, best_label = [], [], []
    print(f"\n{'d':>4} {'baseline':>10} {'best':>10} {'gain':>7}  best config")
    print("-" * 60)
    for d in dims:
        cfgs = recs[d]
        b = cfgs.get("c1_baseline", {}).get("vrf_mean", float("nan"))
        bc_label, bc = min(cfgs.items(), key=lambda kv: kv[1]["vrf_mean"])
        baseline.append(b)
        best.append(bc["vrf_mean"])
        best_label.append(bc_label)
        gain = b / bc["vrf_mean"] if bc["vrf_mean"] > 0 else float("inf")
        print(f"{d:>4} {b:>10.4f} {bc['vrf_mean']:>10.4f} {gain:>6.2f}x  {bc_label}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dims, baseline, "o-", color="C0", lw=2, ms=8,
            label="Paper fixed recipe")
    ax.plot(dims, best, "s-", color="C2", lw=2, ms=8,
            label="Best per-$d$ HP")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="VRF $=1$ (no reduction)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("VRF (mean over 100 obs)")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "per_d_hp_tuned.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)

    summary = {"dims": dims, "baseline": baseline, "best": best,
               "best_label": best_label,
               "all": {str(d): recs[d] for d in dims}}
    with open(os.path.join(args.output_dir, "per_d_hp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
