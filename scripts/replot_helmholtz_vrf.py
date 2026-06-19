"""Re-render the Helmholtz VRF figure from saved results with log-y scaling.

The default eval plot suffers from a few low-frequency components with VRF in
the 5--14 range that dominate the linear-y axis and visually flatten the rest.
Log-y exposes the actual distribution.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "text.usetex": False,
})

NPZ = "figs/helmholtz/helmholtz_eval_results.npz"
OUT_DIR = "figs/helmholtz"

data = np.load(NPZ)
all_vrfs = data["all_vrfs"]                  # (n_test_obs,)
all_comp_vrfs = data["all_comp_vrfs"]        # (n_test_obs, d_z)
all_corrs = data["all_corrs"]                # (n_test_obs,)
eigenvalues = data["eigenvalues"]            # (d_z,)

n_obs, d_z = all_comp_vrfs.shape
comp_mean = all_comp_vrfs.mean(0)
sort_idx = np.argsort(eigenvalues)[::-1]
comp_mean_sorted = comp_mean[sort_idx]
eig_sorted = eigenvalues[sort_idx]

median_pc = float(np.median(comp_mean))
mean_pc = float(comp_mean.mean())
median_po = float(np.median(all_vrfs))
mean_po = float(all_vrfs.mean())
mean_corr = float(all_corrs.mean())

print(f"Per-component VRF:   median={median_pc:.4f}  mean={mean_pc:.4f}")
print(f"Per-observation VRF: median={median_po:.4f}  mean={mean_po:.4f}")
print(f"Mean correlation:    {mean_corr:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

ax = axes[0]
ax.bar(range(d_z), np.clip(comp_mean_sorted, 1e-3, None),
       color="#4C72B0", width=1.0, edgecolor="none", alpha=0.85)
ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.6, label="VRF = 1")
ax.axhline(median_pc, color="green", ls="-", lw=1.4, alpha=0.7,
           label=f"median = {median_pc:.3f}")
ax.set_yscale("log")
ax.set_ylim(1e-3, 30)
ax.set_xlabel("KL component (sorted by eigenvalue)")
ax.set_ylabel("Per-component VRF (log scale)")
ax.set_title(f"(a) Per-component VRF on Helmholtz (d=100)")
ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black",
          fontsize=9)
ax2 = ax.twinx()
ax2.plot(range(d_z), eig_sorted, color="C1", lw=1.2, alpha=0.7)
ax2.set_ylabel("Eigenvalue", color="C1")
ax2.tick_params(axis="y", colors="C1")

ax = axes[1]
clr = ["#d62728" if v > 1.0 else "#4C72B0" for v in all_vrfs]
ax.bar(range(n_obs), all_vrfs, color=clr, alpha=0.85)
ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.6, label="VRF = 1")
ax.axhline(median_po, color="green", ls="-", lw=1.5, alpha=0.7,
           label=f"median = {median_po:.3f}")
ax.set_xlabel("Test observation")
ax.set_ylabel("Mean VRF across components")
ax.set_title(f"(b) Per-observation VRF (mean corr. = {mean_corr:.2f})")
ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black",
          fontsize=9)
ax.set_ylim(0, max(2.6, all_vrfs.max() * 1.05))

plt.tight_layout()
pdf = os.path.join(OUT_DIR, "helmholtz_vrf.pdf")
plt.savefig(pdf)
plt.savefig(pdf.replace(".pdf", ".png"))
print(f"\nSaved: {pdf}")
plt.close(fig)
