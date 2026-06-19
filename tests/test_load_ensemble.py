"""Tests for load_ensemble module."""

import pytest
import torch
import tempfile
import os
from cncv import (
    create_random_split_ensemble,
    load_ensemble_checkpoint,
    validate_split_config,
    get_split_config,
)


class TestGetSplitConfig:
    """Tests for get_split_config function."""

    def test_get_split_config_basic(self):
        """Test extracting split configuration from ensemble."""
        ensemble = create_random_split_ensemble(
            n_in=4,
            n_cond=2,
            n_hidden=8,
            n_ensemble_members=3,
            n_layers=2,
            n_cv=4,
            seed=42,
        )

        split_config = get_split_config(ensemble)

        # Check structure
        assert len(split_config) == 3
        for config in split_config:
            assert "split_idx" in config
            assert "reverse_split" in config
            assert isinstance(config["split_idx"], int)
            assert isinstance(config["reverse_split"], bool)

    def test_get_split_config_matches_layers(self):
        """Test that split config matches actual layer configuration."""
        ensemble = create_random_split_ensemble(
            n_in=6,
            n_cond=2,
            n_hidden=16,
            n_ensemble_members=2,
            seed=123,
        )

        split_config = get_split_config(ensemble)

        for i, (config, layer) in enumerate(zip(split_config, ensemble.layers)):
            assert config["split_idx"] == layer.split_idx
            assert config["reverse_split"] == layer.reverse_split


class TestValidateSplitConfig:
    """Tests for validate_split_config function."""

    def test_validate_split_config_matching(self):
        """Test validation passes with matching configurations."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        split_config = get_split_config(ensemble)

        # Should not raise
        validate_split_config(ensemble, split_config)

    def test_validate_split_config_size_mismatch(self):
        """Test validation fails with different ensemble sizes."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=3, seed=1
        )

        # Create config with wrong size
        wrong_config = [
            {"split_idx": 2, "reverse_split": False},
            {"split_idx": 2, "reverse_split": True},
        ]

        with pytest.raises(ValueError, match="Ensemble size mismatch"):
            validate_split_config(ensemble, wrong_config)

    def test_validate_split_config_idx_mismatch(self):
        """Test validation fails with mismatched split_idx."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        # Get actual config and modify it
        split_config = get_split_config(ensemble)
        split_config[0]["split_idx"] = 999  # Invalid split_idx

        with pytest.raises(ValueError, match="Split configuration mismatch"):
            validate_split_config(ensemble, split_config)

    def test_validate_split_config_reverse_mismatch(self):
        """Test validation fails with mismatched reverse_split."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        # Get actual config and modify it
        split_config = get_split_config(ensemble)
        split_config[0]["reverse_split"] = not split_config[0]["reverse_split"]

        with pytest.raises(ValueError, match="Split configuration mismatch"):
            validate_split_config(ensemble, split_config)


class TestLoadEnsembleCheckpoint:
    """Tests for load_ensemble_checkpoint function."""

    def test_load_ensemble_checkpoint_basic(self):
        """Test basic checkpoint loading with validation."""
        # Create ensemble
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=42
        )

        split_config = get_split_config(ensemble)

        # Create a temporary checkpoint
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pth"
        ) as f:
            checkpoint_path = f.name
            torch.save(
                {
                    "ensemble_state_dict": ensemble.state_dict(),
                    "split_config": split_config,
                    "epoch": 10,
                },
                f,
            )

        try:
            # Create a new ensemble with same configuration
            ensemble_new = create_random_split_ensemble(
                n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=42
            )

            # Load checkpoint
            loaded_checkpoint = load_ensemble_checkpoint(
                ensemble_new, checkpoint_path, torch.device("cpu")
            )

            # Check that checkpoint was loaded
            assert loaded_checkpoint["epoch"] == 10
            assert "ensemble_state_dict" in loaded_checkpoint
            assert "split_config" in loaded_checkpoint

        finally:
            os.unlink(checkpoint_path)

    def test_load_ensemble_checkpoint_missing_file(self):
        """Test error when checkpoint file doesn't exist."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        with pytest.raises(ValueError, match="Checkpoint does not exist"):
            load_ensemble_checkpoint(
                ensemble, "/nonexistent/path.pth", torch.device("cpu")
            )

    def test_load_ensemble_checkpoint_without_split_config(self):
        """Test loading old checkpoint without split_config (should warn)."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=42
        )

        # Create checkpoint without split_config (old format)
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pth"
        ) as f:
            checkpoint_path = f.name
            torch.save(
                {
                    "ensemble_state_dict": ensemble.state_dict(),
                    "epoch": 5,
                },
                f,
            )

        try:
            ensemble_new = create_random_split_ensemble(
                n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=42
            )

            # Should load without error (but print warning)
            loaded_checkpoint = load_ensemble_checkpoint(
                ensemble_new, checkpoint_path, torch.device("cpu")
            )

            assert loaded_checkpoint["epoch"] == 5

        finally:
            os.unlink(checkpoint_path)

    def test_load_ensemble_checkpoint_skip_validation(self):
        """Test loading checkpoint with validation disabled."""
        ensemble = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        # Create checkpoint with WRONG split config
        wrong_config = [
            {"split_idx": 999, "reverse_split": False},
            {"split_idx": 999, "reverse_split": True},
        ]

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pth"
        ) as f:
            checkpoint_path = f.name
            torch.save(
                {
                    "ensemble_state_dict": ensemble.state_dict(),
                    "split_config": wrong_config,
                    "epoch": 3,
                },
                f,
            )

        try:
            ensemble_new = create_random_split_ensemble(
                n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
            )

            # Should load without error when validation is disabled
            loaded_checkpoint = load_ensemble_checkpoint(
                ensemble_new,
                checkpoint_path,
                torch.device("cpu"),
                validate_splits=False,
            )

            assert loaded_checkpoint["epoch"] == 3

        finally:
            os.unlink(checkpoint_path)

    def test_load_ensemble_checkpoint_validation_failure(self):
        """Test that loading fails with mismatched split config."""
        # Create ensemble with seed 1
        ensemble1 = create_random_split_ensemble(
            n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=1
        )

        split_config1 = get_split_config(ensemble1)

        # Create checkpoint
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pth"
        ) as f:
            checkpoint_path = f.name
            torch.save(
                {
                    "ensemble_state_dict": ensemble1.state_dict(),
                    "split_config": split_config1,
                    "epoch": 7,
                },
                f,
            )

        try:
            # Create different ensemble with seed 2 (different splits!)
            ensemble2 = create_random_split_ensemble(
                n_in=4, n_cond=2, n_hidden=8, n_ensemble_members=2, seed=2
            )

            # Should raise error due to split config mismatch
            with pytest.raises(ValueError, match="Split configuration mismatch"):
                load_ensemble_checkpoint(
                    ensemble2, checkpoint_path, torch.device("cpu")
                )

        finally:
            os.unlink(checkpoint_path)
