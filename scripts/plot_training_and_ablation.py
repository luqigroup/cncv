#!/usr/bin/env python
"""Combined Figure 5: (a) Training efficiency, (b) Ensemble size ablation."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Publication-quality rcParams
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 13,
    'axes.titlesize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.5, 'lines.markersize': 5,
})

# ---------- Load training efficiency data ----------
eff_dir = Path("data/training_efficiency")

gauss = np.load(eff_dir / "gaussian_dim4_epoch_curve_shared_hierarchical.npz")
g_epochs = gauss["epochs"]
g_mean_eff = gauss["vrf_means"]
g_std_eff = gauss["vrf_stds"]

rosen = np.load(eff_dir / "rosenbrock_epoch_curve_shared_hierarchical.npz")
r_epochs = rosen["epochs"]
r_mean_eff = rosen["vrf_means"]
r_std_eff = rosen["vrf_stds"]

g_samples_k = g_epochs * 65536 / 1000.0
r_samples_k = r_epochs * 65536 / 1000.0

# ---------- Load ablation data ----------
abl_dir = Path("data/ablations")
LS = [1, 2, 4, 8]
NTREES = [1, 2, 4, 8]

groups_g = defaultdict(list)
groups_r = defaultdict(list)
for L in LS:
    for T in NTREES:
        for exp, groups in [("gaussian", groups_g), ("rosenbrock", groups_r)]:
            path = abl_dir / f"heatmap_{exp}_L{L}_ntrees{T}.npz"
            if path.exists():
                d = np.load(path)
                groups[L * T].append(d["all_vrfs"])

def aggregate(groups):
    totals = sorted(groups.keys())
    means = [np.mean(np.concatenate(groups[t])) for t in totals]
    stds = [np.std(np.concatenate(groups[t])) for t in totals]
    return np.array(totals), np.array(means), np.array(stds)

g_totals, g_mean_abl, g_std_abl = aggregate(groups_g)
r_totals, r_mean_abl, r_std_abl = aggregate(groups_r)

# ---------- Figure ----------
fig = plt.figure(figsize=(6.75, 2.8))
gs = GridSpec(1, 2, figure=fig, wspace=0.32)

# ---- Panel (a): Training efficiency ----
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(g_samples_k, g_mean_eff, 'o-', color='tab:blue', label=r'Gaussian ($d\!=\!4$)')
ax_a.fill_between(g_samples_k, g_mean_eff - g_std_eff, g_mean_eff + g_std_eff,
                   color='tab:blue', alpha=0.15)
ax_a.plot(r_samples_k, r_mean_eff, 's-', color='tab:red', label=r'Rosenbrock ($d\!=\!2$)')
ax_a.fill_between(r_samples_k, r_mean_eff - r_std_eff, r_mean_eff + r_std_eff,
                   color='tab:red', alpha=0.15)
ax_a.axhline(1.0, ls='--', lw=0.8, color='grey', zorder=0)
ax_a.set_yscale('log')
ax_a.set_xlabel(r'Training samples seen ($\times 10^3$)')
ax_a.set_ylabel('VRF')
ax_a.legend(loc='upper right')
ax_a.text(-0.14, 1.05, '(a)', transform=ax_a.transAxes, fontweight='bold',
          fontsize=13, va='bottom', ha='left')

# ---- Panel (b): Ensemble size ablation ----
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(g_totals, g_mean_abl, 'o-', color='#1f77b4', label=r'Gaussian $d\!=\!4$')
ax_b.fill_between(g_totals, g_mean_abl - g_std_abl, g_mean_abl + g_std_abl,
                   alpha=0.15, color='#1f77b4')
ax_b.plot(r_totals, r_mean_abl, 's-', color='#d62728', label=r'Rosenbrock $d\!=\!2$')
ax_b.fill_between(r_totals, r_mean_abl - r_std_abl, r_mean_abl + r_std_abl,
                   alpha=0.15, color='#d62728')
ax_b.set_xscale('log', base=2)
ax_b.set_yscale('log')
ax_b.set_xticks(sorted(set(g_totals.tolist() + r_totals.tolist())))
ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_b.set_xlabel(r'Ensemble size $L$')
ax_b.set_ylabel('VRF')
ax_b.axhline(1.0, ls='--', lw=0.8, color='grey', zorder=0)
ax_b.legend(frameon=True, fancybox=False, edgecolor='0.7')
ax_b.text(-0.14, 1.05, '(b)', transform=ax_b.transAxes, fontweight='bold',
          fontsize=13, va='bottom', ha='left')

# ---------- Save ----------
out_dir = Path("figs/experiments")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "training_and_ablation.pdf")
print(f"Saved: {out_dir / 'training_and_ablation.pdf'}")

local = Path("figs/experiments")
local.mkdir(parents=True, exist_ok=True)
fig.savefig(local / "training_and_ablation.pdf")
print(f"Saved: {local / 'training_and_ablation.pdf'}")
plt.close()
