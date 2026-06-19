"""Re-render the Gaussian d=4 head-to-head VRF comparison with PAIRED 95% CIs.

Reads per-observation VRF arrays from
    figs/baselines/oh_unconditional_results_d4.json

For each pair (CNCV vs Oh-NCV, CNCV vs Poly-1, CNCV vs Poly-2) computes the
per-observation paired difference VRF_method1 - VRF_method2, the mean, and a
bootstrap (and t-) 95% paired CI on that difference distribution.

Outputs:
  figs/baselines/oh_unconditional_comparison_d4_paired.pdf  (and .png)
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_PATH = "figs/baselines/oh_unconditional_results_d4.json"
OUT_PDF = "figs/baselines/oh_unconditional_comparison_d4_paired.pdf"


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 20000, alpha: float = 0.05,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def t_ci(diffs: np.ndarray, alpha: float = 0.05):
    from math import sqrt
    # Welch-equivalent paired t (one-sample t on the differences)
    n = diffs.shape[0]
    m = float(diffs.mean())
    s = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = s / sqrt(n) if n > 0 else 0.0
    # Use scipy if available; fallback to z if not
    try:
        from scipy.stats import t as student_t
        crit = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        crit = 1.96
    return m - crit * se, m + crit * se, crit


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)

    methods = ["cncv", "poly1", "poly2", "oh"]
    method_labels = {
        "cncv": "CNCV (ours)",
        "poly1": "Poly-1",
        "poly2": "Poly-2",
        "oh": "Oh NCV (per-obs)",
    }
    method_colors = {
        "cncv": "#4C72B0",
        "poly1": "#DD8452",
        "poly2": "#55A868",
        "oh": "#8172B2",
    }

    vrfs = {m: np.array(d["vrfs"][m], dtype=np.float64) for m in methods}
    n = len(vrfs["cncv"])
    for m in methods:
        assert len(vrfs[m]) == n, f"length mismatch for {m}: {len(vrfs[m])} vs {n}"
    print(f"Per-observation paired arrays confirmed for n={n} test observations.")

    # ---- Paired statistics ----
    pairs = [("cncv", "oh"), ("cncv", "poly1"), ("cncv", "poly2")]
    paired_stats = {}
    print("\nPaired-difference statistics (method1 - method2; negative favors method1):")
    print(f"{'pair':<15} {'mean diff':>12} {'95% boot CI':>30} {'95% t CI':>30} "
          f"{'wins/n':>8} {'excludes 0?':>12}")
    for m1, m2 in pairs:
        diffs = vrfs[m1] - vrfs[m2]
        mean_d = float(diffs.mean())
        lo_b, hi_b = bootstrap_ci(diffs, n_boot=20000, seed=12)
        lo_t, hi_t, t_crit = t_ci(diffs)
        wins = int((diffs < 0).sum())  # method1 better than method2 on that obs
        excludes_zero = (lo_b > 0) or (hi_b < 0)
        paired_stats[f"{m1}_minus_{m2}"] = {
            "diffs": diffs.tolist(),
            "mean_diff": mean_d,
            "boot_ci_low": lo_b,
            "boot_ci_high": hi_b,
            "t_ci_low": lo_t,
            "t_ci_high": hi_t,
            "t_crit": t_crit,
            "wins_method1": wins,
            "n": n,
            "excludes_zero_boot": bool(excludes_zero),
            "excludes_zero_t": bool((lo_t > 0) or (hi_t < 0)),
        }
        print(f"{m1}-{m2:<8} {mean_d:>12.5f} "
              f"[{lo_b:>+9.5f}, {hi_b:>+9.5f}] "
              f"[{lo_t:>+9.5f}, {hi_t:>+9.5f}] "
              f"{wins:>4}/{n:<3} {str(excludes_zero):>12}")

    # ---- Figure ----
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Panel (a): per-method VRF bars with 95% t-CI on each method's mean
    ax = axes[0]
    valid = methods
    means_per_method = []
    cis_per_method = []
    for m in valid:
        arr = vrfs[m]
        mu = float(arr.mean())
        s = float(arr.std(ddof=1)) if n > 1 else 0.0
        se = s / np.sqrt(n)
        try:
            from scipy.stats import t as student_t
            crit = float(student_t.ppf(0.975, df=n - 1))
        except Exception:
            crit = 1.96
        means_per_method.append(mu)
        cis_per_method.append(crit * se)
    ax.bar(
        range(len(valid)), means_per_method,
        yerr=cis_per_method, capsize=4,
        color=[method_colors[m] for m in valid], alpha=0.85,
    )
    ax.axhline(1.0, color="gray", ls="--", alpha=0.6)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([method_labels[m] for m in valid], rotation=15)
    ax.set_ylabel("VRF (lower better)")
    ax.set_yscale("log")
    ax.set_title(f"(a) Per-method VRF, independent 95% CIs (n={n})")
    ax.grid(alpha=0.2, axis="y", which="both")

    # Panel (b): paired differences, prioritizing CNCV vs Oh NCV
    ax = axes[1]
    pair_priority = pairs  # [(cncv,oh),(cncv,poly1),(cncv,poly2)]
    pair_labels = {
        ("cncv", "oh"): "CNCV - Oh",
        ("cncv", "poly1"): "CNCV - Poly1",
        ("cncv", "poly2"): "CNCV - Poly2",
    }
    pair_colors = {
        ("cncv", "oh"): "#8172B2",
        ("cncv", "poly1"): "#DD8452",
        ("cncv", "poly2"): "#55A868",
    }

    x_offsets = np.arange(len(pair_priority)) * 1.5
    for k, (m1, m2) in enumerate(pair_priority):
        st = paired_stats[f"{m1}_minus_{m2}"]
        diffs = np.array(st["diffs"])
        x0 = x_offsets[k]
        # scatter individual observation differences with small horizontal jitter
        rng = np.random.default_rng(0)
        jitter = (rng.random(diffs.shape[0]) - 0.5) * 0.35
        ax.scatter(
            x0 + jitter, diffs,
            s=36, color=pair_colors[(m1, m2)], alpha=0.75,
            edgecolor="white", linewidth=0.6,
            label="per-obs diff" if k == 0 else None,
        )
        # CI bar
        lo, hi = st["boot_ci_low"], st["boot_ci_high"]
        mean_d = st["mean_diff"]
        ax.errorbar(
            [x0 + 0.55], [mean_d],
            yerr=[[mean_d - lo], [hi - mean_d]],
            fmt="D", color="black", ecolor="black",
            capsize=6, markersize=6, markerfacecolor="white",
            markeredgewidth=1.4, elinewidth=1.4,
            label="mean $\\pm$ 95% paired CI" if k == 0 else None,
        )

    ax.axhline(0.0, color="gray", ls="--", alpha=0.7,
               label="zero (tie)")
    ax.set_xticks(x_offsets + 0.25)
    ax.set_xticklabels([pair_labels[p] for p in pair_priority])
    ax.set_ylabel("Paired VRF difference (negative favors CNCV)")
    ax.set_title(f"(b) Paired differences, 95% bootstrap CI (n={n})")
    ax.grid(alpha=0.2, axis="y", which="both")
    ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.85)

    # Annotate the CNCV-vs-Oh CI (both bootstrap and paired t)
    st_oh = paired_stats["cncv_minus_oh"]
    annot = (
        f"CNCV - Oh paired diffs (n={n}):\n"
        f"  mean = {st_oh['mean_diff']:+.4f}; "
        f"CNCV better in {st_oh['wins_method1']}/{n} obs\n"
        f"  bootstrap 95% CI = [{st_oh['boot_ci_low']:+.4f}, "
        f"{st_oh['boot_ci_high']:+.4f}] "
        f"({'excludes' if st_oh['excludes_zero_boot'] else 'INCLUDES'} 0)\n"
        f"  paired-t 95% CI    = [{st_oh['t_ci_low']:+.4f}, "
        f"{st_oh['t_ci_high']:+.4f}] "
        f"({'excludes' if st_oh['excludes_zero_t'] else 'INCLUDES'} 0)"
    )
    ax.text(0.02, 0.02, annot, transform=ax.transAxes,
            fontsize=7.0, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="0.7", alpha=0.95))

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PDF.replace(".pdf", ".png"))
    print(f"\nSaved: {OUT_PDF}")
    print(f"Saved: {OUT_PDF.replace('.pdf', '.png')}")

    # Persist stats next to the figure for downstream consumers
    stats_out = OUT_PDF.replace(".pdf", "_stats.json")
    with open(stats_out, "w") as f:
        json.dump({"n_obs": n, "paired_stats": paired_stats,
                   "vrfs": {m: vrfs[m].tolist() for m in methods}}, f, indent=2)
    print(f"Saved: {stats_out}")


if __name__ == "__main__":
    main()
