"""Regenerate nonlinear_results figure from hierarchical DDPM data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
data = np.load("data/nonlinear/nonlinear_dim4_shared_hierarchical_ddpm_results.npz")
all_vrfs = data["all_vrfs"]  # (10, 4)
print(f"all_vrfs shape: {all_vrfs.shape}")
print(f"vrf_mean={all_vrfs.mean():.4f}, vrf_std={all_vrfs.std():.4f}")

# Publication-quality settings
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
})

fig = plt.figure(figsize=(6.75, 2.2))
gs = GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Panel (a): Per-component VRF ---
ax_a = fig.add_subplot(gs[0, 0])
n_components = all_vrfs.shape[1]
comp_mean = all_vrfs.mean(axis=0)  # mean across 10 observations
comp_std = all_vrfs.std(axis=0)    # std across 10 observations
comp_labels = [f"$x_{{{i+1}}}$" for i in range(n_components)]
x_comp = np.arange(n_components)

bars_a = ax_a.bar(x_comp, comp_mean, yerr=comp_std, width=0.6,
                  color="steelblue", edgecolor="black", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 0.8})
ax_a.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="VRF = 1")
ax_a.set_xticks(x_comp)
ax_a.set_xticklabels(comp_labels)
ax_a.set_ylabel("VRF")
ax_a.set_title("Per-component VRF")
ax_a.legend(loc="upper right")
ax_a.set_ylim(bottom=0)
ax_a.text(-0.15, 1.05, "(a)", transform=ax_a.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")

# --- Panel (b): Per-observation VRF ---
ax_b = fig.add_subplot(gs[0, 1])
n_obs = all_vrfs.shape[0]
obs_mean = all_vrfs.mean(axis=1)  # mean across 4 components
obs_std = all_vrfs.std(axis=1)    # std across 4 components
x_obs = np.arange(n_obs)

bars_b = ax_b.bar(x_obs, obs_mean, yerr=obs_std, width=0.6,
                  color="coral", edgecolor="black", linewidth=0.5,
                  capsize=3, error_kw={"linewidth": 0.8})
ax_b.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="VRF = 1")
ax_b.set_xticks(x_obs)
ax_b.set_xticklabels([str(i) for i in range(n_obs)])
ax_b.set_xlabel("Observation index")
ax_b.set_ylabel("VRF")
ax_b.set_title("Per-observation VRF")
ax_b.legend(loc="upper right")
ax_b.set_ylim(bottom=0)
ax_b.text(-0.15, 1.05, "(b)", transform=ax_b.transAxes,
          fontsize=9, fontweight="bold", va="bottom", ha="left")

# Save
out_base = "figs/experiments/nonlinear_results"
fig.savefig(out_base + ".pdf")
fig.savefig(out_base + ".png")
print(f"Saved to {out_base}.pdf and .png")
plt.close(fig)
