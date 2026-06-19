"""Regenerate Table 1 + Fig 1 (Gaussian dimension scaling) with per-d-tuned numbers.

Sources every number from the per-d eval JSONs (results/per_d_hp/, written by the
100-obs evaluator) -- nothing is hand-transcribed. Writes, into the paper
repo's figs/gaussian/:
  - results_table.tex          (Table 1: mean + variance VRF per d, mean +/- std)
  - dimension_scaling.{pdf,png} (Fig 1: VRF vs d for mean+var; correlation vs d)

Per-dimension winners (mean and var use the same per-d recipe):
  d=2,4,8 -> c4_deepdata (N=131072);  d=16 -> c3_deep (N=65536; c4_deepdata did
  not train at d=16). See Appendix E for the recipes.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results", "per_d_hp")
PAPER_FIGS = os.path.join(os.path.dirname(__file__), "..", "figs", "gaussian")
DIMS = [2, 4, 8, 16]
MEAN = {2: "d2_c4_deepdata", 4: "d4_c4_deepdata", 8: "d8_c4_deepdata", 16: "d16_c3_deep"}
VAR = {2: "d2_var_c4_deepdata", 4: "d4_var_c4_deepdata", 8: "d8_var_c4_deepdata",
       16: "d16_var_c3_deep"}


def load(tag):
    d = json.load(open(os.path.join(RES, tag + ".json")))
    return (d["vrf_mean"], d["vrf_std_over_components"],
            d.get("correlation_mean", float("nan")))


def main():
    mean = {d: load(MEAN[d]) for d in DIMS}
    var = {d: load(VAR[d]) for d in DIMS}

    # --- Table 1 ---------------------------------------------------------
    rows = "".join(
        f"${d}$  & ${mean[d][0]:.3f} \\pm {mean[d][1]:.3f}$ & "
        f"${var[d][0]:.3f} \\pm {var[d][1]:.3f}$ \\\\\n" for d in DIMS)
    table = (
        r"""\begin{table}[t]
\centering
\caption{Variance reduction for Gaussian posteriors under per-dimension
hyperparameter tuning (recipes in Appendix~\ref{app:hyperparams}).
VRF $=$ Var$(h{-}g)$/Var$(h)$; lower is better. Values are mean $\pm$ std across
$d$ components, each averaged over 100 observations.}
\label{tab:gaussian_results}
\small
\begin{tabular}{ccc}
\toprule
$d$ & Mean Est.\ VRF & Var.\ Est.\ VRF \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
""")
    os.makedirs(PAPER_FIGS, exist_ok=True)
    with open(os.path.join(PAPER_FIGS, "results_table.tex"), "w") as f:
        f.write(table)
    print("Wrote results_table.tex:")
    print(table)

    # --- Fig 1 -----------------------------------------------------------
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif", "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.2))

    mv = [mean[d][0] for d in DIMS]; ms = [mean[d][1] for d in DIMS]
    vv = [var[d][0] for d in DIMS]; vs = [var[d][1] for d in DIMS]
    ax1.errorbar(DIMS, mv, yerr=ms, fmt="o-", capsize=3, capthick=1,
                 markersize=5, linewidth=1.5, color="C0", label="Mean est.")
    ax1.errorbar(DIMS, vv, yerr=vs, fmt="s--", capsize=3, capthick=1,
                 markersize=5, linewidth=1.5, color="C2", label="Var. est.")
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(DIMS); ax1.set_xticklabels([str(d) for d in DIMS])
    ax1.set_xlabel("Dimension $d$")
    ax1.set_ylabel("VRF (lower is better)")
    ax1.set_ylim(0.004, 1.5)
    ax1.legend(loc="lower right", fontsize=8); ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontweight="bold", va="top")

    cr = [mean[d][2] for d in DIMS]
    ax2.plot(DIMS, cr, "o-", color="C1", markersize=5, linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(DIMS); ax2.set_xticklabels([str(d) for d in DIMS])
    ax2.set_xlabel("Dimension $d$")
    ax2.set_ylabel(r"Correlation $\rho(h, g)$")
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", va="top")

    plt.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER_FIGS, f"dimension_scaling.{ext}"))
    plt.close(fig)
    print(f"Wrote dimension_scaling.{{pdf,png}} to {PAPER_FIGS}")


if __name__ == "__main__":
    main()
