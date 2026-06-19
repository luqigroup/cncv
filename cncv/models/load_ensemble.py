"""Utilities for loading ensemble checkpoints with validation.

This module provides functions to load ensemble checkpoints while validating
that the split configuration matches between training and testing.
"""

import torch
import os
from typing import Dict, Any, List
from .ensemble import EnsembleDenseCV


def validate_split_config(
    ensemble: EnsembleDenseCV,
    checkpoint_splits: List[Dict[str, Any]],
) -> None:
    """Validate that ensemble split configuration matches checkpoint.

    Args:
        ensemble: The current ensemble model
        checkpoint_splits: Split configuration from checkpoint

    Raises:
        ValueError: If configurations don't match
    """
    # Extract current configuration
    current_splits = [
        {
            "split_idx": layer.split_idx,
            "reverse_split": layer.reverse_split,
        }
        for layer in ensemble.layers
    ]

    # Check ensemble size
    if len(checkpoint_splits) != len(current_splits):
        raise ValueError(
            f"Ensemble size mismatch! Checkpoint has {len(checkpoint_splits)} members, "
            f"but current ensemble has {len(current_splits)} members."
        )

    # Check each layer's configuration
    mismatches = []
    for i, (ckpt_cfg, curr_cfg) in enumerate(zip(checkpoint_splits, current_splits)):
        if ckpt_cfg["split_idx"] != curr_cfg["split_idx"]:
            mismatches.append(
                f"  Layer {i}: split_idx {curr_cfg['split_idx']} != {ckpt_cfg['split_idx']} (checkpoint)"
            )
        if ckpt_cfg["reverse_split"] != curr_cfg["reverse_split"]:
            mismatches.append(
                f"  Layer {i}: reverse_split {curr_cfg['reverse_split']} != {ckpt_cfg['reverse_split']} (checkpoint)"
            )

    if mismatches:
        raise ValueError(
            f"Split configuration mismatch!\n"
            f"The ensemble architecture does not match the checkpoint.\n"
            f"Mismatches found:\n" + "\n".join(mismatches) + "\n\n"
            f"This usually happens if the random seed changed or the ensemble "
            f"creation order changed. Ensure you're using the same seed and "
            f"configuration as during training."
        )


def load_ensemble_checkpoint(
    ensemble: EnsembleDenseCV,
    checkpoint_path: str,
    device: torch.device,
    validate_splits: bool = True,
) -> Dict[str, Any]:
    """Load ensemble checkpoint with optional split configuration validation.

    Args:
        ensemble: The ensemble model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint onto
        validate_splits: Whether to validate split configuration (default: True)

    Returns:
        checkpoint: The loaded checkpoint dictionary

    Raises:
        ValueError: If checkpoint doesn't exist or validation fails
    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")

    print(f"\n=== Loading Checkpoint ===")
    print(f"Path: {checkpoint_path}")

    # Load checkpoint
    if device == torch.device(type="cpu"):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Validate split configurations if requested
    if validate_splits:
        if "split_config" in checkpoint:
            validate_split_config(ensemble, checkpoint["split_config"])
            print("Split configuration validated - matches checkpoint")
        else:
            print("Warning: Checkpoint does not contain split_config (old checkpoint)")
            print("  Assuming splits are correct based on random seed reproducibility")

    # Load ensemble state dict
    ensemble.load_state_dict(checkpoint["ensemble_state_dict"])

    return checkpoint


def get_split_config(ensemble: EnsembleDenseCV) -> List[Dict[str, Any]]:
    """Extract split configuration from ensemble for checkpointing.

    Args:
        ensemble: The ensemble model

    Returns:
        List of dictionaries containing split_idx and reverse_split for each layer
    """
    return [
        {
            "split_idx": layer.split_idx,
            "reverse_split": layer.reverse_split,
        }
        for layer in ensemble.layers
    ]
