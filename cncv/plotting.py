"""Plotting utilities for CNCV evaluation.

This module provides standalone plotting functions for visualizing
control variate performance, variance reduction, and correlation analysis.

All functions are pure (no side effects except file I/O) and accept
explicit data arguments rather than relying on class state.

Color Scheme:
    - Training loss: #4a4a4a (dark gray)
    - Validation loss: #a1a1a1 (light gray)
    - Vanilla MC: #7f7f7f (medium gray)
    - Control Variates: #d62728 (red)
    - Prior samples: #000000 (black)

Configuration:
    All styling is controlled via PLOT_CONFIG dict, which can be
    overridden per-function via the config parameter.

Example:
    >>> from cncv.plotting import plot_results, compute_h_x
    >>> h_x = compute_h_x(X_samples, qofi="mean")
    >>> plot_results(
    ...     plots_dir="./plots",
    ...     train_obj=train_losses,
    ...     val_obj=val_losses,
    ...     sample_sizes=[10, 50, 100],
    ...     vanilla_mse=vanilla_mse_arr,
    ...     cv_mse=cv_mse_arr,
    ...     h_x=h_x,
    ...     g_combined=control_variates,
    ...     all_g=layer_cvs,
    ...     true_mean=ground_truth
    ... )
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any


# Configuration dictionary for all plotting settings
PLOT_CONFIG = {
    # Colors - Publication quality with high contrast
    "color_training": "#1f77b4",  # Blue for training
    "color_validation": "#ff7f0e",  # Orange for validation
    "color_vanilla": "#7f7f7f",  # Gray for vanilla MC
    "color_cv": "#d62728",  # Red for control variates
    "color_prior": "#000000",  # Black for prior
    "color_scatter": "#2ca02c",  # Green for scatter
    "color_reference": "#8c564b",  # Brown for reference lines

    # Figure settings
    "dpi": 300,
    "bbox_inches": "tight",

    # Figure sizes - Optimized for single/double column papers
    "figsize_training": (6, 4),
    "figsize_comparison": (12, 3.5),  # Wider, less tall for subplots
    "figsize_diagonal": (10, 4),
    "figsize_posterior_prior": (5, 5),
    "figsize_scatter_2x2": (8, 6),  # For 4D scatter plots
    "figsize_heatmap": (10, 3.5),  # For high-D heatmaps

    # Font sizes - Publication ready
    "fontsize_title": 14,
    "fontsize_suptitle": 16,
    "fontsize_label": 13,
    "fontsize_tick": 11,
    "fontsize_legend": 11,
    "fontsize_annotation": 10,

    # Plot aesthetics
    "alpha_training": 0.8,
    "alpha_validation": 1.0,
    "alpha_violin": 0.7,
    "alpha_scatter": 0.3,
    "alpha_prior": 0.15,
    "alpha_posterior": 0.4,
    "alpha_reference": 0.6,
    "grid_alpha": 0.3,

    # Marker sizes
    "markersize_default": 8,
    "scatter_size_small": 1.0,
    "scatter_size_medium": 2.0,

    # Line widths
    "linewidth_default": 2.5,
    "linewidth_reference": 2.0,

    # Analysis parameters
    "estimator_dist_sample_size": 100,
    "estimator_dist_n_trials": 200,
    "variance_comp_sample_size": 100,
    "variance_comp_n_trials": 500,
    "heatmap_max_diagonal_labels": 10,
}


def compute_h_x(
    X_samples: torch.Tensor,
    qofi: str,
    mu_post: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute quantity of interest h(x).

    Args:
        X_samples: Posterior samples [num_samples, n_dim]
        qofi: Quantity of interest ("mean" or "var")
        mu_post: Posterior mean [n_dim], required if qofi="var"

    Returns:
        h_x: Quantity of interest [num_samples, n_dim]

    Raises:
        ValueError: If qofi is unknown or mu_post is missing for qofi="var"

    Example:
        >>> X = torch.randn(1000, 5)
        >>> h_x = compute_h_x(X, qofi="mean")
        >>> print(h_x.shape)  # (1000, 5)
    """
    if qofi == "mean":
        return X_samples
    elif qofi == "var":
        if mu_post is None:
            raise ValueError("mu_post must be provided when qofi='var'")
        return (X_samples - mu_post) ** 2
    else:
        raise ValueError(f"Unknown qofi: {qofi}. Expected 'mean' or 'var'.")


def plot_training_loss(
    train_obj: List[float],
    val_obj: List[float],
    save_path: str,
    title_suffix: str = "",
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot training and validation loss curves.

    Args:
        train_obj: Training loss history (list of floats, one per batch)
        val_obj: Validation loss history (list of floats, one per epoch)
        save_path: Full path where to save the plot
        title_suffix: Optional suffix for title (e.g., "(Rosenbrock)")
        config: Optional config override (defaults to PLOT_CONFIG)

    Example:
        >>> plot_training_loss(
        ...     train_obj=[1.2, 0.9, 0.7, 0.5],
        ...     val_obj=[1.0, 0.8, 0.6],
        ...     save_path="./plots/training_loss.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}

    fig, ax = plt.subplots(figsize=cfg["figsize_training"])

    # Training loss (one per batch) - align to epoch scale
    epochs_train = np.linspace(0, len(val_obj), len(train_obj))
    ax.plot(
        epochs_train,
        train_obj,
        color=cfg["color_training"],
        label="Training",
        alpha=cfg["alpha_training"],
        linewidth=cfg["linewidth_default"],
    )

    # Validation loss (one per epoch)
    epochs_val = np.arange(len(val_obj))
    ax.plot(
        epochs_val,
        val_obj,
        color=cfg["color_validation"],
        label="Validation",
        linewidth=cfg["linewidth_default"],
    )

    ax.set_xlabel("Epoch", fontsize=cfg["fontsize_label"])
    ax.set_ylabel("Loss", fontsize=cfg["fontsize_label"])
    ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
    ax.legend(fontsize=cfg["fontsize_legend"], frameon=True, fancybox=False, edgecolor='black')
    ax.grid(alpha=cfg["grid_alpha"], linestyle='--')
    plt.tight_layout()

    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_mse_comparison(
    sample_sizes: List[int],
    vanilla_mse: np.ndarray,
    cv_mse: np.ndarray,
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot MSE comparison between vanilla MC and CV across sample sizes.

    Creates separate plot files per component.

    Args:
        sample_sizes: List of sample sizes (e.g., [10, 20, 50, ...])
        vanilla_mse: Vanilla MC MSE array [n_sample_sizes, n_dim]
        cv_mse: CV MSE array [n_sample_sizes, n_dim]
        save_path: Base path where to save plots (will add _compX.png suffix)
        config: Optional config override

    Example:
        >>> plot_mse_comparison(
        ...     sample_sizes=[10, 50, 100],
        ...     vanilla_mse=np.array([[0.1, 0.2], [0.05, 0.1], [0.02, 0.05]]),
        ...     cv_mse=np.array([[0.05, 0.1], [0.02, 0.05], [0.01, 0.02]]),
        ...     save_path="./plots/mse_comparison.png"
        ... )
        # Creates: mse_comparison_comp0.png, ..._comp1.png, etc.
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = vanilla_mse.shape[1]

    # Extract directory and filename
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)
    name_parts = os.path.splitext(save_name)

    # Create separate file per component
    for i in range(n_dim):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            sample_sizes,
            vanilla_mse[:, i],
            "o-",
            label="Vanilla MC",
            linewidth=cfg["linewidth_default"],
            markersize=cfg["markersize_default"],
            color=cfg["color_vanilla"],
        )
        ax.plot(
            sample_sizes,
            cv_mse[:, i],
            "s-",
            label="CV",
            linewidth=cfg["linewidth_default"],
            markersize=cfg["markersize_default"],
            color=cfg["color_cv"],
        )

        # Reference 1/n line
        ref_line = vanilla_mse[0, i] * (sample_sizes[0] / np.array(sample_sizes))
        ax.plot(
            sample_sizes,
            ref_line,
            "--",
            label="$1/n$",
            linewidth=cfg["linewidth_reference"],
            color=cfg["color_reference"],
            alpha=cfg["alpha_reference"],
        )

        ax.set_xlabel("Sample size", fontsize=cfg["fontsize_label"])
        ax.set_ylabel("MSE", fontsize=cfg["fontsize_label"])
        ax.set_title(f"Component {i}", fontsize=cfg["fontsize_title"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
        ax.legend(fontsize=cfg["fontsize_legend"], frameon=True, loc='upper right')
        ax.grid(alpha=cfg["grid_alpha"], linestyle='--', which='both')

        plt.tight_layout()

        # Save with component index
        component_path = os.path.join(save_dir, f"{name_parts[0]}_comp{i}{name_parts[1]}")
        plt.savefig(component_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
        print(f"Saved {component_path}")
        plt.close()


def plot_vrf_vs_sample_size(
    sample_sizes: List[int],
    vanilla_mse: np.ndarray,
    cv_mse: np.ndarray,
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot Variance Reduction Factor vs sample size.

    Creates separate plot files per component showing sample-size invariance.

    Args:
        sample_sizes: List of sample sizes
        vanilla_mse: Vanilla MC MSE array [n_sample_sizes, n_dim]
        cv_mse: CV MSE array [n_sample_sizes, n_dim]
        save_path: Base path where to save plots (will add _compX.png suffix)
        config: Optional config override

    Example:
        >>> plot_vrf_vs_sample_size(
        ...     sample_sizes=[10, 50, 100],
        ...     vanilla_mse=np.array([[0.1, 0.2], [0.05, 0.1], [0.02, 0.05]]),
        ...     cv_mse=np.array([[0.05, 0.1], [0.02, 0.05], [0.01, 0.02]]),
        ...     save_path="./plots/vrf_vs_sample_size.png"
        ... )
        # Creates: vrf_vs_sample_size_comp0.png, ..._comp1.png, etc.
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = vanilla_mse.shape[1]

    # Compute VRF for each sample size
    vrf = cv_mse / vanilla_mse

    # Extract directory and filename
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)
    name_parts = os.path.splitext(save_name)

    # Create separate file per component
    for i in range(n_dim):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            sample_sizes,
            vrf[:, i],
            "o-",
            linewidth=cfg["linewidth_default"],
            markersize=cfg["markersize_default"],
            color=cfg["color_cv"],
        )

        # Reference line at VRF = 1.0 (no reduction)
        ax.axhline(
            1.0,
            color=cfg["color_reference"],
            linestyle="--",
            linewidth=cfg["linewidth_reference"],
            alpha=cfg["alpha_reference"],
        )

        # Compute mean VRF for annotation
        mean_vrf = vrf[:, i].mean()
        reduction_pct = (1.0 - mean_vrf) * 100

        ax.text(
            0.98, 0.02,
            f"{reduction_pct:.0f}\\% reduction",
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=cfg["fontsize_annotation"],
        )

        ax.set_xlabel("Sample size", fontsize=cfg["fontsize_label"])
        ax.set_ylabel("VRF", fontsize=cfg["fontsize_label"])
        ax.set_title(f"Component {i}", fontsize=cfg["fontsize_title"])
        ax.set_xscale("log")
        ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
        ax.grid(alpha=cfg["grid_alpha"], linestyle='--')
        ax.set_ylim([0, 1.2])

        plt.tight_layout()

        # Save with component index
        component_path = os.path.join(save_dir, f"{name_parts[0]}_comp{i}{name_parts[1]}")
        plt.savefig(component_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
        print(f"Saved {component_path}")
        plt.close()


def plot_estimator_distributions(
    h_x: torch.Tensor,
    g_combined: torch.Tensor,
    true_mean: torch.Tensor,
    save_path: str,
    sample_size: Optional[int] = None,
    n_trials: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot violin plots comparing Vanilla MC vs Ensemble CV distributions.

    Creates 2x2 subplot for 4D.

    Args:
        h_x: Quantity of interest samples [num_samples, n_dim]
        g_combined: Combined control variates [n_dim, num_samples]
        true_mean: Ground truth values [n_dim]
        save_path: Full path where to save the plot
        sample_size: Sample size for subsampling (default from config)
        n_trials: Number of trials (default from config)
        config: Optional config override

    Example:
        >>> h_x = torch.randn(1000, 5)
        >>> g_combined = torch.randn(5, 1000)
        >>> true_mean = torch.zeros(5)
        >>> plot_estimator_distributions(
        ...     h_x, g_combined, true_mean,
        ...     save_path="./plots/estimator_distributions.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    sample_size = sample_size or cfg["estimator_dist_sample_size"]
    n_trials = n_trials or cfg["estimator_dist_n_trials"]
    n_dim = h_x.shape[1]

    # Collect estimates
    vanilla_estimates = np.zeros((n_trials, n_dim))
    cv_estimates = np.zeros((n_trials, n_dim))

    for trial in range(n_trials):
        indices = torch.randperm(h_x.shape[0])[:sample_size]

        # Vanilla MC
        vanilla_estimates[trial] = h_x[indices].mean(dim=0).cpu().numpy()

        # CV estimator
        cv_residual = h_x[indices].T - g_combined[:, indices]
        cv_estimates[trial] = cv_residual.mean(dim=1).cpu().numpy()

    # Create subplot layout
    if n_dim == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
    else:
        ncols = min(n_dim, 4)
        nrows = (n_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_dim == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    # Create violin plots
    for i in range(n_dim):
        parts = axes[i].violinplot(
            [vanilla_estimates[:, i], cv_estimates[:, i]],
            positions=[1, 2],
            showmeans=True,
            showextrema=True,
        )

        # Color the violins
        for j, pc in enumerate(parts["bodies"]):
            if j == 0:
                pc.set_facecolor(cfg["color_vanilla"])
                pc.set_alpha(cfg["alpha_violin"])
            else:
                pc.set_facecolor(cfg["color_cv"])
                pc.set_alpha(cfg["alpha_violin"])

        # Add ground truth line
        truth = true_mean[i].cpu().numpy()
        axes[i].axhline(
            truth,
            color=cfg["color_reference"],
            linestyle="--",
            linewidth=cfg["linewidth_default"],
        )

        # Compute VRF for this component
        var_vanilla = vanilla_estimates[:, i].var()
        var_cv = cv_estimates[:, i].var()
        vrf = var_cv / var_vanilla
        reduction_pct = (1.0 - vrf) * 100

        # Add VRF annotation
        axes[i].text(
            0.5,
            0.98,
            f"{reduction_pct:.0f}\\% reduction",
            transform=axes[i].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=cfg["fontsize_annotation"],
        )

        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(["Vanilla", "CV"], fontsize=cfg["fontsize_tick"])
        axes[i].set_ylabel("Estimate", fontsize=cfg["fontsize_label"])
        axes[i].set_title(f"Component {i}", fontsize=cfg["fontsize_title"])
        axes[i].tick_params(axis='y', labelsize=cfg["fontsize_tick"])
        axes[i].grid(alpha=cfg["grid_alpha"], axis="y", linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_variance_comparison(
    h_x: torch.Tensor,
    g_combined: torch.Tensor,
    true_mean: torch.Tensor,
    save_path: str,
    sample_size: Optional[int] = None,
    n_trials: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot bar chart comparing Var[Vanilla] vs Var[CV].

    Creates 2x2 subplot for 4D.

    Args:
        h_x: Quantity of interest samples [num_samples, n_dim]
        g_combined: Combined control variates [n_dim, num_samples]
        true_mean: Ground truth values [n_dim]
        save_path: Full path where to save the plot
        sample_size: Sample size for subsampling (default from config)
        n_trials: Number of trials (default from config)
        config: Optional config override

    Example:
        >>> h_x = torch.randn(1000, 5)
        >>> g_combined = torch.randn(5, 1000)
        >>> true_mean = torch.zeros(5)
        >>> plot_variance_comparison(
        ...     h_x, g_combined, true_mean,
        ...     save_path="./plots/variance_comparison.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    sample_size = sample_size or cfg["variance_comp_sample_size"]
    n_trials = n_trials or cfg["variance_comp_n_trials"]
    n_dim = h_x.shape[1]

    # Collect estimates
    vanilla_estimates = np.zeros((n_trials, n_dim))
    cv_estimates = np.zeros((n_trials, n_dim))

    for trial in range(n_trials):
        indices = torch.randperm(h_x.shape[0])[:sample_size]

        # Vanilla MC
        vanilla_estimates[trial] = h_x[indices].mean(dim=0).cpu().numpy()

        # CV estimator
        cv_residual = h_x[indices].T - g_combined[:, indices]
        cv_estimates[trial] = cv_residual.mean(dim=1).cpu().numpy()

    # Compute variances
    var_vanilla = vanilla_estimates.var(axis=0)
    var_cv = cv_estimates.var(axis=0)

    # Create subplot layout
    if n_dim == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
    else:
        ncols = min(n_dim, 4)
        nrows = (n_dim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_dim == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    # Create bar charts
    x = np.arange(2)
    width = 0.6

    for i in range(n_dim):
        bars = axes[i].bar(
            x,
            [var_vanilla[i], var_cv[i]],
            width,
            color=[cfg["color_vanilla"], cfg["color_cv"]],
            alpha=cfg["alpha_violin"],
        )

        # Compute VRF
        vrf = var_cv[i] / var_vanilla[i]
        reduction_pct = (1.0 - vrf) * 100

        # Add VRF annotation
        axes[i].text(
            0.5,
            0.95,
            f"{reduction_pct:.0f}\\% reduction",
            transform=axes[i].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=cfg["fontsize_annotation"],
        )

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["Vanilla", "CV"], fontsize=cfg["fontsize_tick"])
        axes[i].set_ylabel("Variance", fontsize=cfg["fontsize_label"])
        axes[i].set_title(f"Component {i}", fontsize=cfg["fontsize_title"])
        axes[i].tick_params(axis='y', labelsize=cfg["fontsize_tick"])
        axes[i].grid(alpha=cfg["grid_alpha"], axis="y", linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_diagonal_correlations(
    h_x: np.ndarray,
    g_combined: torch.Tensor,
    all_g: List[torch.Tensor],
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot diagonal correlations (h_i vs g_i) for all components.

    Args:
        h_x: Samples [num_samples, n_dim] (numpy array)
        g_combined: Combined control variates [n_dim, num_samples]
        all_g: List of individual layer CVs
        save_path: Full path where to save the plot
        config: Optional config override

    Example:
        >>> h_x = np.random.randn(1000, 5)
        >>> g_combined = torch.randn(5, 1000)
        >>> all_g = [torch.randn(5, 1000) for _ in range(3)]
        >>> plot_diagonal_correlations(
        ...     h_x, g_combined, all_g,
        ...     save_path="./plots/diagonal_correlations.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = h_x.shape[1]
    n_layers = len(all_g)

    # Compute diagonal correlations
    diag_corr_layers = np.zeros((n_layers, n_dim))
    for j, g_layer in enumerate(all_g):
        g_layer_np = g_layer.cpu().numpy()
        for i in range(n_dim):
            diag_corr_layers[j, i] = np.corrcoef(
                h_x[:, i], g_layer_np[i, :]
            )[0, 1]

    g_combined_np = g_combined.cpu().numpy()
    diag_corr_combined = np.zeros(n_dim)
    for i in range(n_dim):
        diag_corr_combined[i] = np.corrcoef(h_x[:, i], g_combined_np[i, :])[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=cfg["figsize_diagonal"])

    x = np.arange(n_dim)
    width = 0.8 / (n_layers + 1)

    # Plot individual layers
    for j in range(n_layers):
        ax.bar(
            x + j * width,
            diag_corr_layers[j],
            width,
            label=f"Layer {j + 1}",
            alpha=cfg["alpha_violin"],
        )

    # Plot combined
    ax.bar(
        x + n_layers * width,
        diag_corr_combined,
        width,
        label="Combined",
        alpha=0.9,
        color=cfg["color_cv"],
    )

    ax.set_xlabel("Component index $i$", fontsize=cfg["fontsize_label"])
    ax.set_ylabel("$\\rho(h_i, g_i)$", fontsize=cfg["fontsize_label"])
    ax.set_title(f"Diagonal Correlations ({n_dim}D)", fontsize=cfg["fontsize_title"])
    ax.set_xticks(x + width * n_layers / 2)
    ax.set_xticklabels([f"{i}" for i in range(n_dim)], fontsize=cfg["fontsize_tick"])
    ax.tick_params(axis='y', labelsize=cfg["fontsize_tick"])
    ax.legend(fontsize=cfg["fontsize_legend"], frameon=True, loc='best')
    ax.grid(alpha=cfg["grid_alpha"], axis="y", linestyle='--')
    ax.axhline(0, color=cfg["color_reference"], linewidth=1.0, alpha=cfg["alpha_reference"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_scatter_h_vs_gcv(
    h_x: np.ndarray,
    g_combined: torch.Tensor,
    all_g: List[torch.Tensor],
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot scatter plots or correlation heatmaps: h vs g for each component.

    For low dimensions (<=6), creates separate scatter plot files per component.
    For high dimensions (>6), shows correlation heatmap and calls
    plot_diagonal_correlations automatically.

    Args:
        h_x: Samples [num_samples, n_dim] (numpy array)
        g_combined: Combined control variates [n_dim, num_samples]
        all_g: List of individual layer CVs
        save_path: Base path for saving plots (will add _compX.png suffix)
        config: Optional config override

    Note:
        For high dimensions, also generates diagonal_correlations.png in same directory.

    Example:
        >>> h_x = np.random.randn(1000, 4)
        >>> g_combined = torch.randn(4, 1000)
        >>> all_g = [torch.randn(4, 1000) for _ in range(2)]
        >>> plot_scatter_h_vs_gcv(
        ...     h_x, g_combined, all_g,
        ...     save_path="./plots/scatter_h_vs_gcv.png"
        ... )
        # Creates: scatter_h_vs_gcv_comp0.png, ..._comp1.png, ..._comp2.png, ..._comp3.png
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = h_x.shape[1]
    n_layers = len(all_g)

    # Extract directory and filename
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)
    name_parts = os.path.splitext(save_name)

    if n_dim <= 6:
        # For low dimensions, create separate scatter plot files per component
        for i in range(n_dim):
            fig, ax = plt.subplots(figsize=(5, 5))

            g_combined_np = g_combined[i, :].cpu().numpy()
            ax.scatter(
                h_x[:, i],
                g_combined_np,
                s=cfg["scatter_size_medium"],
                alpha=cfg["alpha_scatter"],
                color=cfg["color_cv"],
                rasterized=True,  # For better PDF rendering
            )

            # Compute correlation
            corr_combined = np.corrcoef(h_x[:, i], g_combined_np)[0, 1]

            # Add reference line y=x
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lims = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
            ax.plot(lims, lims, '--', color=cfg["color_reference"],
                    alpha=cfg["alpha_reference"], linewidth=cfg["linewidth_reference"],
                    label='Perfect correlation')

            ax.set_xlabel(f"$h_{{{i}}}(x)$", fontsize=cfg["fontsize_label"])
            ax.set_ylabel(f"$g_{{{i}}}(x, y)$", fontsize=cfg["fontsize_label"])
            ax.set_title(f"$\\rho$ = {corr_combined:.3f}",
                        fontsize=cfg["fontsize_title"])
            ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
            ax.grid(alpha=cfg["grid_alpha"], linestyle='--')
            ax.legend(fontsize=cfg["fontsize_legend"], loc='lower right', frameon=True)

            plt.tight_layout()

            # Save with component index
            component_path = os.path.join(save_dir, f"{name_parts[0]}_comp{i}{name_parts[1]}")
            plt.savefig(component_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
            print(f"Saved {component_path}")
            plt.close()

    else:
        # For high dimensions, use correlation heatmaps
        # Only show combined ensemble for cleaner visualization
        fig, ax = plt.subplots(1, 1, figsize=cfg["figsize_heatmap"])

        # Compute combined CV correlation matrix only
        g_combined_np = g_combined.cpu().numpy()  # [n_dim, num_samples]
        corr_matrix = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for k in range(n_dim):
                corr_matrix[i, k] = np.corrcoef(
                    h_x[:, i], g_combined_np[k, :]
                )[0, 1]

        # Plot heatmap
        im = ax.imshow(
            corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        ax.set_xlabel("Control variate component $j$", fontsize=cfg["fontsize_label"])
        ax.set_ylabel("Target function component $i$", fontsize=cfg["fontsize_label"])
        ax.set_title(f"Correlation Matrix: $\\rho(h_i, g_j)$ ({n_dim}D)",
                    fontsize=cfg["fontsize_title"])
        ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])

        # Add colorbar with label
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=cfg["fontsize_label"])
        cbar.ax.tick_params(labelsize=cfg["fontsize_tick"])

        # Add diagonal correlation values as text
        max_labels = cfg["heatmap_max_diagonal_labels"]
        for i in range(min(max_labels, n_dim)):
            ax.text(
                i,
                i,
                f"{corr_matrix[i, i]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr_matrix[i, i]) > 0.5 else "black",
                fontsize=cfg["fontsize_annotation"],
                weight='bold',
            )

        plt.tight_layout()

        plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
        print(f"Saved {save_path}")
        plt.close()

        # Also create a summary plot showing diagonal correlations
        diag_save_path = os.path.join(
            os.path.dirname(save_path), "diagonal_correlations.png"
        )
        plot_diagonal_correlations(h_x, g_combined, all_g, diag_save_path, config)


def plot_noise_cancellation(
    h_x: torch.Tensor,
    g_combined: torch.Tensor,
    all_g: List[torch.Tensor],
    save_path: str,
    component_idx: int = 0,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Visualize noise cancellation mechanism in the ensemble.

    Shows how individual layers have high variance but anticorrelate,
    resulting in low variance when averaged.

    Args:
        h_x: Quantity of interest samples [num_samples, n_dim]
        g_combined: Combined control variates [n_dim, num_samples]
        all_g: List of individual layer CVs
        save_path: Full path where to save the plot
        component_idx: Which component to visualize (default: 0)
        config: Optional config override

    Example:
        >>> plot_noise_cancellation(
        ...     h_x, g_combined, all_g,
        ...     save_path="./plots/noise_cancellation.png",
        ...     component_idx=0
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_layers = len(all_g)

    # Extract component for analysis
    h_comp = h_x[:, component_idx].cpu().numpy()
    g_comp_combined = g_combined[component_idx, :].cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Individual layer CVs vs combined
    sample_indices = np.arange(min(200, len(h_comp)))  # Show first 200 samples

    # Plot individual layers
    for j, g_layer in enumerate(all_g):
        g_layer_comp = g_layer[component_idx, sample_indices].cpu().numpy()
        ax1.plot(sample_indices, g_layer_comp, alpha=0.4, linewidth=1.0,
                label=f'Layer {j+1}')

    # Plot combined (average)
    ax1.plot(sample_indices, g_comp_combined[sample_indices],
            color=cfg["color_cv"], linewidth=2.5, label='Ensemble (avg)',
            zorder=10)

    # Plot target function
    ax1.plot(sample_indices, h_comp[sample_indices],
            color='black', linewidth=2.0, linestyle='--',
            label='$h(x)$', alpha=0.7, zorder=11)

    ax1.set_xlabel("Sample index", fontsize=cfg["fontsize_label"])
    ax1.set_ylabel("Value", fontsize=cfg["fontsize_label"])
    ax1.set_title("Noise cancellation mechanism", fontsize=cfg["fontsize_title"])
    ax1.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
    ax1.legend(fontsize=cfg["fontsize_legend"]-2, loc='best', ncol=2)
    ax1.grid(alpha=cfg["grid_alpha"], linestyle='--')

    # Right plot: Correlation matrix between layers
    layer_matrix = np.zeros((n_layers, len(h_comp)))
    for j, g_layer in enumerate(all_g):
        layer_matrix[j, :] = g_layer[component_idx, :].cpu().numpy()

    # Compute correlation matrix
    corr_matrix = np.corrcoef(layer_matrix)

    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xlabel("Layer index", fontsize=cfg["fontsize_label"])
    ax2.set_ylabel("Layer index", fontsize=cfg["fontsize_label"])
    ax2.set_title("Layer cross-correlations", fontsize=cfg["fontsize_title"])
    ax2.tick_params(axis='both', labelsize=cfg["fontsize_tick"])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=cfg["fontsize_label"])
    cbar.ax.tick_params(labelsize=cfg["fontsize_tick"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_effective_sample_size(
    sample_sizes: List[int],
    vanilla_mse: np.ndarray,
    cv_mse: np.ndarray,
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot effective sample size: how many vanilla MC samples = N CV samples.

    This shows the practical benefit in terms of sample efficiency.

    Args:
        sample_sizes: List of sample sizes
        vanilla_mse: Vanilla MC MSE array [n_sample_sizes, n_dim]
        cv_mse: CV MSE array [n_sample_sizes, n_dim]
        save_path: Full path where to save the plot (will append component index)
        config: Optional config override

    Example:
        >>> plot_effective_sample_size(
        ...     sample_sizes=[10, 50, 100],
        ...     vanilla_mse=np.array([[0.1], [0.05], [0.02]]),
        ...     cv_mse=np.array([[0.05], [0.02], [0.01]]),
        ...     save_path="./plots/effective_sample_size.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = vanilla_mse.shape[1]

    # Extract directory and filename
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)
    name_parts = os.path.splitext(save_name)

    for comp in range(n_dim):
        fig, ax = plt.subplots(figsize=(5, 4))

        # For each CV sample size, find equivalent vanilla sample size
        effective_sizes = []
        for i, n_cv in enumerate(sample_sizes):
            mse_cv = cv_mse[i, comp]
            # Find vanilla sample size that achieves same MSE
            # MSE scales as 1/n, so: vanilla_mse[0] / n_eff = mse_cv
            n_eff = vanilla_mse[0, comp] / mse_cv
            effective_sizes.append(n_eff)

        # Plot
        ax.plot(sample_sizes, effective_sizes, 'o-',
               linewidth=cfg["linewidth_default"],
               markersize=cfg["markersize_default"],
               color=cfg["color_cv"],
               label='Effective sample size')

        # Add reference line (no improvement)
        ax.plot(sample_sizes, sample_sizes, '--',
               linewidth=cfg["linewidth_reference"],
               color=cfg["color_reference"],
               alpha=cfg["alpha_reference"],
               label='No improvement')

        # Fill area showing gain
        ax.fill_between(sample_sizes, sample_sizes, effective_sizes,
                       alpha=0.2, color=cfg["color_cv"])

        ax.set_xlabel("CV sample size", fontsize=cfg["fontsize_label"])
        ax.set_ylabel("Equivalent vanilla MC samples", fontsize=cfg["fontsize_label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
        ax.legend(fontsize=cfg["fontsize_legend"], frameon=True, loc='upper left')
        ax.grid(alpha=cfg["grid_alpha"], linestyle='--', which='both')

        # Add annotation showing gain at largest sample size
        gain = effective_sizes[-1] / sample_sizes[-1]
        ax.text(0.98, 0.02, f"{gain:.1f}× sample efficiency",
               transform=ax.transAxes,
               verticalalignment="bottom",
               horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
               fontsize=cfg["fontsize_annotation"])

        plt.tight_layout()

        # Save with component index
        component_path = os.path.join(save_dir, f"{name_parts[0]}_comp{comp}{name_parts[1]}")
        plt.savefig(component_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
        print(f"Saved {component_path}")
        plt.close()


def plot_layer_contributions(
    h_x: torch.Tensor,
    all_g: List[torch.Tensor],
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot how much each ensemble layer contributes to variance reduction.

    Shows correlation and variance for each individual layer.

    Args:
        h_x: Quantity of interest samples [num_samples, n_dim]
        all_g: List of individual layer CVs
        save_path: Full path where to save the plot
        config: Optional config override

    Example:
        >>> plot_layer_contributions(
        ...     h_x, all_g,
        ...     save_path="./plots/layer_contributions.png"
        ... )
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_dim = h_x.shape[1]
    n_layers = len(all_g)

    h_x_np = h_x.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Correlation of each layer with h(x) (averaged over components)
    layer_corrs = np.zeros(n_layers)
    for j, g_layer in enumerate(all_g):
        g_layer_np = g_layer.cpu().numpy()
        corrs = []
        for i in range(n_dim):
            corr = np.corrcoef(h_x_np[:, i], g_layer_np[i, :])[0, 1]
            corrs.append(abs(corr))
        layer_corrs[j] = np.mean(corrs)

    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    ax1.bar(np.arange(n_layers), layer_corrs, color=colors, alpha=0.8)
    ax1.set_xlabel("Layer index", fontsize=cfg["fontsize_label"])
    ax1.set_ylabel("Mean |correlation| with $h(x)$", fontsize=cfg["fontsize_label"])
    ax1.set_title("Layer correlations", fontsize=cfg["fontsize_title"])
    ax1.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
    ax1.grid(alpha=cfg["grid_alpha"], axis='y', linestyle='--')
    ax1.set_ylim([0, 1])

    # Right: Variance of each layer (averaged over components)
    layer_vars = np.zeros(n_layers)
    for j, g_layer in enumerate(all_g):
        g_layer_np = g_layer.cpu().numpy()
        layer_vars[j] = np.mean([g_layer_np[i, :].var() for i in range(n_dim)])

    ax2.bar(np.arange(n_layers), layer_vars, color=colors, alpha=0.8)
    ax2.set_xlabel("Layer index", fontsize=cfg["fontsize_label"])
    ax2.set_ylabel("Mean variance", fontsize=cfg["fontsize_label"])
    ax2.set_title("Layer variances", fontsize=cfg["fontsize_title"])
    ax2.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
    ax2.grid(alpha=cfg["grid_alpha"], axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_posterior_vs_prior(
    X_posterior: torch.Tensor,
    rosenbrock_dist: Any,
    save_path: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Plot posterior samples overlaid on prior samples (Rosenbrock-specific).

    Only works for 2D case. Returns early if n_dim != 2.

    Args:
        X_posterior: Posterior samples [num_samples, n_dim]
        rosenbrock_dist: RosenbrockDistribution instance for sampling prior
        save_path: Full path where to save the plot
        config: Optional config override

    Example:
        >>> X_posterior = torch.randn(1000, 2)
        >>> plot_posterior_vs_prior(
        ...     X_posterior, rosenbrock_dist,
        ...     save_path="./plots/posterior_vs_prior.png"
        ... )
    """
    print("\n=== Generating Posterior vs Prior Plot ===")

    # Only plot for 2D case
    n_dim = X_posterior.shape[1]
    if n_dim != 2:
        print(f"Skipping posterior vs prior plot for {n_dim}D (only for 2D)")
        return

    cfg = {**PLOT_CONFIG, **(config or {})}

    # Generate prior samples (same number as posterior)
    num_samples = X_posterior.shape[0]
    device = X_posterior.device
    X_prior = rosenbrock_dist.sample(num_samples, device=device)

    # Convert to numpy for plotting
    X_prior_np = X_prior.cpu().numpy()
    X_posterior_np = X_posterior.cpu().numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=cfg["figsize_posterior_prior"])
    ax.patch.set_facecolor("white")

    # Plot prior samples (black, low alpha)
    ax.scatter(
        X_prior_np[:, 0],
        X_prior_np[:, 1],
        s=cfg["scatter_size_small"],
        color=cfg["color_prior"],
        alpha=cfg["alpha_prior"],
        label="Prior samples",
    )

    # Plot posterior samples (red, higher alpha)
    ax.scatter(
        X_posterior_np[:, 0],
        X_posterior_np[:, 1],
        s=cfg["scatter_size_medium"],
        color=cfg["color_cv"],
        alpha=cfg["alpha_posterior"],
        label="Posterior samples (pSGLD)",
    )

    ax.set_xlim([-3, 3])
    ax.set_ylim([-2.5, 7])
    ax.set_xlabel("$x_1$", fontsize=cfg["fontsize_label"])
    ax.set_ylabel("$x_2$", fontsize=cfg["fontsize_label"])
    ax.set_title("Posterior vs Prior", fontsize=cfg["fontsize_title"])
    ax.tick_params(axis='both', labelsize=cfg["fontsize_tick"])
    ax.legend(loc="upper right", fontsize=cfg["fontsize_legend"], frameon=True)
    ax.grid(False)

    plt.tight_layout()

    plt.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    print(f"Saved {save_path}")
    plt.close()


def plot_results(
    plots_dir: str,
    train_obj: List[float],
    val_obj: List[float],
    sample_sizes: List[int],
    vanilla_mse: np.ndarray,
    cv_mse: np.ndarray,
    h_x: torch.Tensor,
    g_combined: torch.Tensor,
    all_g: List[torch.Tensor],
    true_mean: torch.Tensor,
    title_suffix: str = "",
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Generate simple, clear evaluation plots focusing on paper claims.

    Generates 6 publication-quality plots showing variance reduction:
    1. training_loss.png - Shows network convergence
    2. scatter_h_vs_gcv.png - Shows correlations (2x2 subplot for 4D)
    3. variance_comparison.png - Shows variance reduction (2x2 subplot)
    4. vrf_vs_sample_size.png - Shows sample-size invariance (2x2 subplot)
    5. mse_comparison.png - Shows 1/n scaling (2x2 subplot)
    6. estimator_distributions.png - Shows tighter distributions (2x2 violin plots)

    Args:
        plots_dir: Directory where to save all plots
        train_obj: Training loss history (list of floats, one per batch)
        val_obj: Validation loss history (list of floats, one per epoch)
        sample_sizes: List of sample sizes (e.g., [10, 20, 50, ...])
        vanilla_mse: Vanilla MC MSE array [n_sample_sizes, n_dim]
        cv_mse: CV MSE array [n_sample_sizes, n_dim]
        h_x: Quantity of interest samples [num_samples, n_dim]
        g_combined: Combined control variates [n_dim, num_samples]
        all_g: List of individual layer CVs
        true_mean: Ground truth values [n_dim]
        title_suffix: Optional suffix for titles (e.g., "(Rosenbrock)")
        config: Optional config override

    Example:
        >>> plot_results(
        ...     plots_dir="./plots",
        ...     train_obj=[1.0, 0.8, 0.6],
        ...     val_obj=[0.9, 0.7],
        ...     sample_sizes=[10, 50, 100],
        ...     vanilla_mse=np.array([[0.1, 0.2], [0.05, 0.1], [0.02, 0.05]]),
        ...     cv_mse=np.array([[0.05, 0.1], [0.02, 0.05], [0.01, 0.02]]),
        ...     h_x=torch.randn(1000, 4),
        ...     g_combined=torch.randn(4, 1000),
        ...     all_g=[torch.randn(4, 1000) for _ in range(3)],
        ...     true_mean=torch.zeros(4)
        ... )
    """
    print(f"\n=== Generating Plots ===")
    print(f"Plot directory: {plots_dir}")

    # 1. Training loss plot
    plot_training_loss(
        train_obj=train_obj,
        val_obj=val_obj,
        save_path=os.path.join(plots_dir, "training_loss.png"),
        title_suffix=title_suffix,
        config=config,
    )

    # 2. MSE comparison plot
    plot_mse_comparison(
        sample_sizes=sample_sizes,
        vanilla_mse=vanilla_mse,
        cv_mse=cv_mse,
        save_path=os.path.join(plots_dir, "mse_comparison.png"),
        config=config,
    )

    # 3. Estimator distributions (violin plots)
    plot_estimator_distributions(
        h_x=h_x,
        g_combined=g_combined,
        true_mean=true_mean,
        save_path=os.path.join(plots_dir, "estimator_distributions.png"),
        config=config,
    )

    # 4. VRF vs sample size
    plot_vrf_vs_sample_size(
        sample_sizes=sample_sizes,
        vanilla_mse=vanilla_mse,
        cv_mse=cv_mse,
        save_path=os.path.join(plots_dir, "vrf_vs_sample_size.png"),
        config=config,
    )

    # 5. Variance comparison (bar charts)
    plot_variance_comparison(
        h_x=h_x,
        g_combined=g_combined,
        true_mean=true_mean,
        save_path=os.path.join(plots_dir, "variance_comparison.png"),
        config=config,
    )

    # 6. Scatter plots (h vs g_cv)
    # Convert h_x to numpy for scatter plot function
    h_x_np = h_x.cpu().numpy()
    n_dim = h_x_np.shape[1]
    if n_dim <= 6:
        save_filename = "scatter_h_vs_gcv.png"
    else:
        save_filename = "correlation_heatmap.png"

    plot_scatter_h_vs_gcv(
        h_x=h_x_np,
        g_combined=g_combined,
        all_g=all_g,
        save_path=os.path.join(plots_dir, save_filename),
        config=config,
    )

    print("\n=== Evaluation Complete ===")
