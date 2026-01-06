# `cncv` - Conditional Neural Control Variates

A Python package implementing conditional neural control variates for variance reduction in Bayesian inference.

## Overview

CNCV uses ensemble of coupling layers to construct control variates that
reduce variance in Monte Carlo estimation of integrals with respect to
posterior distributions in Bayesian inference.

## Installation

For development and running examplesm, clone the repository and install in editable mode:

```bash
# Clone repository
git clone https://github.com/luqigroup/cncv
cd cncv

# Install in editable mode
pip install -e .
```

## Project Structure

```
cncv/
├── cncv/                 # Main package
│   ├── models/          # Coupling layers and ensemble models
│   │   ├── coupling_layer.py
│   │   └── ensemble.py
│   └── utils/           # Distribution utilities and optimizers
│       ├── gaussian.py
│       ├── rosenbrock.py
│       ├── psgld.py
│       └── lr_scheduler.py
├── scripts/             # Training and evaluation scripts
│   ├── gaussian_ensemble_cv.py
│   └── rosenbrock_ensemble_cv.py
├── configs/             # JSON configuration files
├── tests/               # Unit tests
├── data/                # Data storage (created by projorg)
└── plots/               # Visualization outputs (created by projorg)
```

## Quick Start

### Training an ensemble CV model

```bash
# Gaussian example
python scripts/gaussian_ensemble_cv.py --phase train

# Rosenbrock example
python scripts/rosenbrock_ensemble_cv.py --phase train
```

### Evaluating a trained model

```bash
# Test at final epoch (default)
python scripts/gaussian_ensemble_cv.py --phase test

# Test at specific epoch
python scripts/gaussian_ensemble_cv.py --phase test --testing_epoch 49
```

### Running tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_jacobian.py -v
pytest tests/test_coupling_layer.py -v
pytest tests/test_ensemble.py -v
```


## Usage Example

```python
from cncv import (
    create_random_split_ensemble,
    compute_score_posterior_gaussian,
    compute_exact_posterior
)
import torch

# Create ensemble with 2 coupling layers
ensemble = create_random_split_ensemble(
    n_in=2,
    n_cond=2,
    n_hidden=32,
    n_ensemble_members=2,
    n_layers=3,
    n_cv=2
)

# Compute control variates
X = torch.randn(100, 2)  # Samples
C = torch.randn(100, 2)  # Conditioning
score = compute_score_posterior_gaussian(X, C, mu_prior, Sigma_inv, sigma)

g_combined, all_g = ensemble(X, C, score)
```

## Configuration

Experiments are configured via JSON files in `configs/`. Example:

```json
{
    "experiment_name": "gaussian_ensemble_cv",
    "max_epochs": 50,
    "lr": 0.001,
    "lr_final": 0.0001,
    "sigma": 0.3,
    "n_ensemble_members": 2,
    "batchsize": 256,
    "n_hidden": 32,
    "n_layers": 3,
    "num_train": 65536,
    "num_val": 2048,
    "num_samples": 10000
}
```


## Author

Ali Siahkoohi (alisk@ucf.edu)


