"""Corner plot of nonlinear posterior for multiple observations.

Only lower-triangle panels (no diagonal), tight layout to minimize whitespace.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.size': 7, 'font.family': 'serif', 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5, 'lines.linewidth': 0.8,
})

d = np.load('data/nonlinear/multi_posterior_viz.npz')
posts = [d['post0'], d['post1'], d['post2']]
trues = [d['true0'], d['true1'], d['true2']]

colors = ['#1f77b4', '#d62728', '#2ca02c']
dim_labels = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
obs_labels = [r'$\mathbf{x}^{(1)}_{\mathrm{true}}$',
              r'$\mathbf{x}^{(2)}_{\mathrm{true}}$',
              r'$\mathbf{x}^{(3)}_{\mathrm{true}}$']

ndim = 4
# Lower triangle pairs: (row, col) where row > col
pairs = []
for i in range(1, ndim):
    for j in range(i):
        pairs.append((i, j))
# 6 panels: arrange as 3 rows x 2 cols + legend in remaining space
# Row 0: (1,0)
# Row 1: (2,0), (2,1)
# Row 2: (3,0), (3,1), (3,2)
# Use a 3x3 grid, fill lower-left triangle

fig, axes_grid = plt.subplots(3, 3, figsize=(4.5, 4.5))

# Map (i,j) with i>0, j<i to grid positions
# Row in grid = i-1, Col in grid = j
panel_map = {}
for i in range(1, ndim):
    for j in range(i):
        panel_map[(i, j)] = axes_grid[i - 1, j]

# Hide unused axes (upper triangle + extra)
for gi in range(3):
    for gj in range(3):
        used = False
        for (i, j), ax in panel_map.items():
            if ax is axes_grid[gi, gj]:
                used = True
                break
        if not used:
            axes_grid[gi, gj].set_visible(False)


def plot_contours_2d(ax, samples, color, levels_frac=(0.5, 0.9)):
    """Plot KDE contour lines for 2D samples."""
    try:
        kde = gaussian_kde(samples.T)
        pad = 0.5
        xmin, xmax = samples[:, 0].min() - pad, samples[:, 0].max() + pad
        ymin, ymax = samples[:, 1].min() - pad, samples[:, 1].max() + pad
        xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        # Convert probability fractions to density levels
        zz_sorted = np.sort(zz.ravel())[::-1]
        cum = np.cumsum(zz_sorted) / zz_sorted.sum()
        level_vals = []
        for lev in levels_frac:
            idx = np.searchsorted(cum, lev)
            if idx < len(zz_sorted):
                level_vals.append(zz_sorted[idx])
            else:
                level_vals.append(zz_sorted[-1])
        level_vals = sorted(level_vals)

        ax.contour(xx, yy, zz, levels=level_vals, colors=[color],
                   linewidths=0.8, alpha=0.9)
    except Exception:
        ax.scatter(samples[:, 0], samples[:, 1], s=0.3, alpha=0.1,
                   color=color)


for (i, j), ax in panel_map.items():
    for obs_idx, (post, xt, color) in enumerate(zip(posts, trues, colors)):
        samples_2d = post[:, [j, i]]
        plot_contours_2d(ax, samples_2d, color)
        ax.scatter([xt[j]], [xt[i]], marker='*', s=35, color=color,
                   edgecolors='black', linewidths=0.3, zorder=10)

    # Axis labels only on edges
    grid_row, grid_col = i - 1, j
    if grid_row == 2:  # bottom row
        ax.set_xlabel(dim_labels[j])
    else:
        ax.set_xticklabels([])
    if grid_col == 0:  # left column
        ax.set_ylabel(dim_labels[i])
    else:
        ax.set_yticklabels([])

# Legend in upper-right empty space
legend_ax = axes_grid[0, 2]
legend_ax.set_visible(True)
legend_ax.axis('off')
for color, label in zip(colors, obs_labels):
    legend_ax.scatter([], [], marker='*', s=50, color=color,
                      edgecolors='black', linewidths=0.3, label=label)
legend_ax.legend(loc='center', frameon=True, fancybox=False,
                 edgecolor='black', fontsize=7, handletextpad=0.3,
                 borderpad=0.5)

fig.subplots_adjust(hspace=0.12, wspace=0.12)

out_dir = 'data/nonlinear'
paper_dir = 'figs/experiments'
for ext in ['pdf', 'png']:
    fig.savefig(f'{out_dir}/nonlinear_posterior_viz.{ext}')
    fig.savefig(f'{paper_dir}/nonlinear_posterior_viz.{ext}')
    print(f'Saved {ext}')
plt.close()
