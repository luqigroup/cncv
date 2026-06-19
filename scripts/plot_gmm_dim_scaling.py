"""Plot VRF vs dimension on GMM for poly-1, poly-2, Oh-style NCV, and CNCV.

Mirrors the layout/style of the paper's Gaussian dimension-scaling figure
(figs/gaussian/dimension_scaling_mean.pdf), but with four methods on the same
panel and the GMM rows from figs/baselines/poly_high_d_gmm.json.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": False,
})

HALF_COL = 3.25
FULL_COL = 6.75

COLORS = {
    "poly1": "#7f7f7f",
    "poly2": "#9467bd",
    "oh":    "#1f77b4",
    "cncv":  "#d62728",
}
MARKERS = {"poly1": "v", "poly2": "s", "oh": "^", "cncv": "o"}
LABELS = {
    "poly1": "Polynomial-1 (Mira et al., 2013)",
    "poly2": "Polynomial-2 (Mira et al., 2013)",
    "oh":    "Oh-style neural Stein CV (Bedaque-Oh, Oh)",
    "cncv":  "CNCV (ours, hierarchical coupling)",
}


def aggregate(json_path):
    with open(json_path) as f:
        data = json.load(f)
    dims = sorted(int(d) for d in data.keys())
    out = {m: {"mean": [], "std": [], "valid": []} for m in ["poly1", "poly2", "oh", "cncv"]}
    for d in dims:
        row = data[str(d)]
        for m in ["poly1", "poly2", "oh", "cncv"]:
            arr = np.array(row[f"{m}_vrf"], dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                out[m]["mean"].append(np.nan)
                out[m]["std"].append(np.nan)
                out[m]["valid"].append(False)
            else:
                out[m]["mean"].append(float(arr.mean()))
                out[m]["std"].append(float(arr.std()))
                out[m]["valid"].append(True)
    return dims, out


def main():
    src = "figs/baselines/poly_high_d_gmm.json"
    out_dir = "figs/baselines"
    dims, agg = aggregate(src)

    # Two-panel: (a) linear-y, (b) log-y, mirroring paper's mean/var split.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_COL, 2.6))

    ax1.axhline(1.0, color="#8c564b", linestyle="--", alpha=0.6, linewidth=1,
                label="No reduction (VRF$=1$)")
    ax2.axhline(1.0, color="#8c564b", linestyle="--", alpha=0.6, linewidth=1)

    for m in ["poly1", "poly2", "oh", "cncv"]:
        means = np.array(agg[m]["mean"], dtype=float)
        stds = np.array(agg[m]["std"], dtype=float)
        valid = np.array(agg[m]["valid"])
        x = np.array(dims)[valid]
        y = means[valid]
        e = stds[valid]
        if len(x) == 0:
            continue
        ax1.errorbar(x, y, yerr=e, fmt=MARKERS[m] + "-", color=COLORS[m],
                     capsize=3, capthick=1, label=LABELS[m])
        # log-y panel: clip to floor for visualization
        y_log = np.clip(y, 1e-5, None)
        ax2.errorbar(x, y_log, yerr=np.minimum(e, y_log * 0.5),
                     fmt=MARKERS[m] + "-", color=COLORS[m],
                     capsize=3, capthick=1, label=LABELS[m])

    # Annotate Poly-2 failures (overfit at d=32, OOM at d=64).
    for d_idx, d in enumerate(dims):
        if not agg["poly2"]["valid"][d_idx]:
            ax1.annotate("OOM", xy=(d, 0.05), ha="center", fontsize=9,
                         color=COLORS["poly2"], fontweight="bold")
            ax2.annotate("OOM", xy=(d, 1.5e-1), ha="center", fontsize=9,
                         color=COLORS["poly2"], fontweight="bold")
        elif d == 32 and agg["poly2"]["mean"][d_idx] < 1e-3:
            ax1.annotate("overfit", xy=(d, 0.06), ha="center", fontsize=9,
                         color=COLORS["poly2"], fontweight="bold")
            ax2.annotate("overfit",
                         xy=(d, max(agg["poly2"]["mean"][d_idx], 1e-3) * 1.3),
                         ha="center", fontsize=9,
                         color=COLORS["poly2"], fontweight="bold")

    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("VRF")
    ax1.set_xticks(dims)
    ax1.set_ylim(-0.03, 0.55)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=10,
             fontweight="bold", va="top")

    ax2.set_xlabel("Dimension $d$")
    ax2.set_ylabel("VRF (log scale)")
    ax2.set_yscale("log")
    ax2.set_xticks(dims)
    ax2.set_ylim(8e-4, 2.0)
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # Single legend below plots.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.18),
               frameon=True, fancybox=False, edgecolor="black", fontsize=9)

    fig.suptitle("Variance reduction on a 2-component diagonal Gaussian mixture",
                 fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    pdf_path = os.path.join(out_dir, "gmm_dim_scaling.pdf")
    png_path = os.path.join(out_dir, "gmm_dim_scaling.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)

    # Single-panel compact companion.
    fig, ax = plt.subplots(figsize=(FULL_COL * 0.6, 2.6))
    ax.axhline(1.0, color="#8c564b", linestyle="--", alpha=0.6, linewidth=1,
               label="No reduction")
    for m in ["poly1", "poly2", "oh", "cncv"]:
        means = np.array(agg[m]["mean"], dtype=float)
        stds = np.array(agg[m]["std"], dtype=float)
        valid = np.array(agg[m]["valid"])
        x = np.array(dims)[valid]
        y = means[valid]
        e = stds[valid]
        if len(x) == 0:
            continue
        ax.errorbar(x, y, yerr=e, fmt=MARKERS[m] + "-", color=COLORS[m],
                    capsize=3, capthick=1, label=LABELS[m].split(" (")[0])
    for d_idx, d in enumerate(dims):
        if not agg["poly2"]["valid"][d_idx]:
            ax.annotate("OOM", xy=(d, 0.05), ha="center", fontsize=9,
                        color=COLORS["poly2"], fontweight="bold")
        elif d == 32 and agg["poly2"]["mean"][d_idx] < 1e-3:
            ax.annotate("overfit", xy=(d, 0.06), ha="center", fontsize=9,
                        color=COLORS["poly2"], fontweight="bold")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("VRF (lower is better)")
    ax.set_xticks(dims)
    ax.set_ylim(-0.03, 0.55)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="center right",
              fontsize=8)
    plt.tight_layout()
    pdf_path2 = os.path.join(out_dir, "gmm_dim_scaling_compact.pdf")
    plt.savefig(pdf_path2)
    plt.savefig(pdf_path2.replace(".pdf", ".png"))
    print(f"Saved: {pdf_path2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
