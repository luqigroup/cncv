# Conditional Neural Control Variates (CNCV)

Amortized neural control variates for variance reduction in Monte Carlo
estimation of posterior expectations in Bayesian inverse problems.

> **Paper.** *Conditional Neural Control Variates for Bayesian Inverse Problems.*
> Accepted at the Conference on Uncertainty in Artificial Intelligence (UAI), 2026.

## Overview

CNCV combines Stein's identity with an ensemble of hierarchical coupling layers
to learn an observation-conditioned control variate

```
g(x, y) = div phi(x, y) + phi(x, y) . grad log p(x | y)
```

from joint model-data samples `(x_i, y_i) ~ p(x, y)`.
After a single training run, the same network applies to **any** new observation
`y*` at test time without retraining.

The key architectural ingredient is a **hierarchical (HINT-style) binary tree
of affine coupling layers** whose Jacobian diagonal is computed in a single
forward pass, making the divergence term exact and tractable at d=100 -- where
generic neural Stein control variates would require O(d) backward
passes or noisy Hutchinson trace estimation.

## Installation

```bash
git clone https://github.com/luqigroup/cncv
cd cncv
pip install -e .
```

A recent Python (3.10+) with PyTorch is required; core dependencies are listed
in [`requirements.txt`](requirements.txt) and installed by the editable install.
A GPU is recommended for training but not required for evaluating released
checkpoints.

**Optional, only needed to regenerate data from scratch** (the released datasets
are downloaded automatically — see below):

| Experiment | Dependency |
|---|---|
| Darcy flow data generation | Devito-based groundwater solver (set `CNCV_GROUNDWATER_DIR`) |
| Helmholtz FWI data generation | self-contained (`cncv.utils.helmholtz`) — no external solver |

## Datasets

The Darcy and Helmholtz joint training datasets are hosted publicly and
**downloaded automatically on first use** — every script that needs one calls
`cncv.utils.download.ensure_dataset`, which fetches the file if it is missing
and is a no-op afterwards. No manual download or data generation is required.

| Dataset | Size | Direct link |
|---|---|---|
| Darcy KL (d=100, N=131072) | 84 MB | [download](https://www.dropbox.com/scl/fi/71zo8uoyvy3xnsfgjqzvd/darcy_K10_N131072_grid40.npz?rlkey=zlzerd1j0mukltb25jhxh10jm&dl=1) |
| Helmholtz KL (d=100, N=131072) | 428 MB | [download](https://www.dropbox.com/scl/fi/catpkkyqv6rce56uwcge8/helmholtz_K10_N131072_grid32.npz?rlkey=qibi038a3zllw9a8zt1u5ivyr&dl=1) |

The stylized problems (Gaussian, Rosenbrock, nonlinear) generate their data on
the fly and need no download. To regenerate the PDE datasets instead of
downloading them, use `scripts/generate_darcy_data.py` and
`scripts/generate_helmholtz_joint.py`.

## Quick start

```bash
# Run the unit tests
pytest tests/ -v

# Smallest end-to-end example: Gaussian d=4, mean estimation
python scripts/gaussian_ensemble_cv.py --input_dim 4 --qofi mean

# Reproduce a Darcy figure (auto-downloads the dataset on first run)
python scripts/generate_darcy_sample_efficiency.py --gpu 0
```

## Reproducing paper figures

Checkpoints for the non-PDE problems are managed by `projorg` under
`data/checkpoints/` (the directory name encodes the full config); the Darcy and
Helmholtz checkpoints live at fixed paths under `data/darcy/` and
`data/helmholtz/`. Figure scripts write to `figs/<problem>/` by default.

### Main text

| Fig / Table | Description | Script |
|---|---|---|
| Table 1, Fig 1 | Gaussian dimension scaling (VRF + correlation) | `make_fig1_table1_tuned.py` |
| Fig 2 | Sample efficiency, Gaussian d=4 | `generate_sample_efficiency.py` |
| Fig 3 | Amortized CNCV on Rosenbrock | `generate_rosenbrock_figures.py` |
| Fig 4 | Rosenbrock variance estimation | `run_var_estimation.py` |
| Fig 5 | Nonlinear posterior corner plot | `plot_nonlinear_corner.py` |
| Fig 6 | Nonlinear per-component / per-obs VRF | `plot_nonlinear_results.py` |
| Fig 7 | Training efficiency + ensemble-size ablation | `plot_training_and_ablation.py` |
| Fig 8 | Sample efficiency with CNF-learned scores | `generate_cnf_efficiency_combined.py` |
| Fig 9 | Darcy sample efficiency | `generate_darcy_sample_efficiency.py` |
| Fig 10 | Darcy posterior summary | `plot_darcy_kl_results.py` |
| Fig 11 | Darcy estimator-std comparison | `plot_darcy_kl_results.py` |

### Appendix

| Fig | Description | Script |
|---|---|---|
| Fig 12 | Stein identity empirical verification | `verify_stein_identity.py` |
| Fig 13 | Per-KL-component VRF (Darcy) | `plot_darcy_kl_results.py` |
| Fig 14 | CNF KL scatter (Darcy) | `plot_cnf_diagnostics.py` |
| Fig 15 | Per-component score correlation (Darcy) | `plot_cnf_diagnostics.py` |
| Fig 16 | CNF + CNCV training/validation losses | `plot_cnf_diagnostics.py` |
| Fig 17 | CNF posterior-mean bias (Darcy) | `quantify_cnf_bias_darcy.py` |
| Fig 18 | Darcy posterior gallery | `generate_darcy_gallery.py` |
| Fig 19 | Darcy posterior samples | `plot_cnf_posterior_samples.py` |
| Fig 20 | Helmholtz FWI posterior summary | `plot_helmholtz_kl_results.py` |
| Fig 21 | Helmholtz per-component VRF | `plot_helmholtz_kl_results.py` |
| Fig 22 | Helmholtz sample efficiency | `plot_helmholtz_kl_results.py` |
| Fig 23 | Polynomial-CV breakdown on a bimodal mixture | `plot_gmm_dim_scaling.py` |
| Fig 24 | Hutchinson trace ablation, d=4 | `hutchinson_simple.py` |
| Fig 25 | Hutchinson trace ablation, d=100 | `hutchinson_d100.py` |
| Fig 26 | Student-t noise, per-observation VRF | `student_t_proper.py` |
| Fig 27 | VRF versus score-approximation error | `score_perturbation_ablation.py` |

### Pipelines

**Gaussian** (analytical scores, seed 12). Train at each dimension, then plot:

```bash
for d in 2 4 8 16; do
    python scripts/gaussian_ensemble_cv.py --input_dim $d --qofi mean
    python scripts/gaussian_ensemble_cv.py --input_dim $d --qofi var
done
python scripts/make_fig1_table1_tuned.py        # Table 1 + Fig 1
python scripts/generate_sample_efficiency.py    # Fig 2
```

**Rosenbrock** (seed 1):

```bash
python scripts/rosenbrock_ensemble_cv.py --qofi mean
python scripts/generate_rosenbrock_figures.py   # Fig 3
python scripts/run_var_estimation.py --problem rosenbrock   # Fig 4
```

**Nonlinear** (DDPM-based posterior):

```bash
python scripts/ddpm_nonlinear.py
python scripts/run_nonlinear_forward.py --use_ddpm
python scripts/generate_nonlinear_posterior_viz.py   # data for Fig 5
python scripts/plot_nonlinear_corner.py              # Fig 5
python scripts/plot_nonlinear_results.py             # Fig 6
```

**Learned scores** (Table 2, Fig 8). Train a CNF per problem, then CNCV on its
score:

```bash
for p in gaussian rosenbrock nonlinear; do
    python scripts/train_cnf_$p.py    --phase train
    python scripts/${p}_cnf_cv.py     --phase train
done
python scripts/generate_cnf_efficiency_combined.py   # Fig 8
```

**Darcy flow** (KL space, d=100). The dataset auto-downloads; train the CNF
then the CNCV, then plot:

```bash
python scripts/train_darcy_kl.py --phase cnf  --gpu 0
python scripts/train_darcy_kl.py --phase cncv --gpu 0 \
    --cnf_checkpoint data/darcy/checkpoints_cnf_kl_v2b/cnf_best.pth
python scripts/plot_darcy_kl_results.py --gpu 0       # Figs 10, 11, 13
python scripts/plot_cnf_diagnostics.py  --gpu 0       # Figs 14–16
python scripts/generate_darcy_gallery.py --gpu 0      # Fig 18
```

**Helmholtz FWI** (KL space, d=100; appendix). Self-contained — the forward
solver ships in `cncv.utils.helmholtz`:

```bash
# (optional) regenerate instead of auto-downloading:
python scripts/generate_helmholtz_joint.py --N 131072 --grid_size 32 \
    --output data/helmholtz/helmholtz_K10_N131072_grid32.npz
bash scripts/run_helmholtz_pipeline.sh               # train CNF + CNCV
python scripts/plot_helmholtz_kl_results.py --gpu 0  # Figs 20–22
```

## Checkpoints

`verify_stein_identity.py` reads four checkpoints (Gaussian, Rosenbrock,
Nonlinear, Darcy) by path; the CNF checkpoint path is stored inside each CNCV
checkpoint, so only the CNCV path is passed to evaluation scripts.

## Conventions

- **VRF** = `Var(h - g) / Var(h)`. Lower is better; `< 1` means variance
  reduction.
- Training uses **joint** samples from `p(x, y)`. MCMC is
  needed only at test time, to compute reference posterior expectations.
- Default seeds: 12 for Gaussian, 1 for Rosenbrock.

## Tests

```bash
pytest tests/ -v
```

The suite covers coupling-layer Jacobian correctness (finite-difference
verified), conditional normalizing flow parity, ensemble factories, and the
analytical score utilities.

## License

[MIT](LICENSE).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
