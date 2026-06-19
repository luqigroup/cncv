"""Darcy flow inverse problem utilities following Beskos et al. (2017).

Matches Beskos et al. (2017):
  - KL modes: i1, i2 in {1, ..., K} (not {0, ..., K-1})
  - Sensors: 1 center + 32 on full-domain circle (radius ~ (grid_size-1)/2)
  - Truth field: Beskos formula u_i = lambda_i^{1/2} sin(...)
"""

import os
import sys
import numpy as np
import torch

# External Devito-based groundwater solver. Only needed to RE-GENERATE Darcy
# data; loading data or evaluating trained models does not require it. Set the
# CNCV_GROUNDWATER_DIR environment variable to the solver's location to use it.
GROUNDWATER_DIR = os.environ.get("CNCV_GROUNDWATER_DIR")
if GROUNDWATER_DIR:
    sys.path.insert(0, GROUNDWATER_DIR)
try:
    from groundwater.devito_op import GroundwaterEquation
except ImportError:
    # The solver is an optional external dependency; defer the error until a
    # function that actually needs it is called.
    GroundwaterEquation = None


class DarcyKLPrior:
    """KL expansion prior with Beskos-correct mode indexing.

    Modes use i1, i2 in {1, ..., K} (Beskos Eq. 37), giving d = K^2 modes.
    Eigenfunctions: phi_i(x) = 2 * cos(pi*(i+0.5)*x1) * cos(pi*(j+0.5)*x2)
    Eigenvalues:    lambda_i = sigma * (alpha + pi^2*((i+0.5)^2 + (j+0.5)^2))^(-s/2)

    Args:
        K: Cutoff index (i1, i2 in 1..K), giving d = K^2 modes.
        grid_size: Spatial grid resolution for field reconstruction.
        alpha: Regularity parameter (default 0, Beskos).
        s: Decay parameter (default 1.1, Beskos).
        sigma: Scale parameter (default 1.0, Beskos).
    """

    def __init__(self, K=10, grid_size=32, alpha=0.0, s=1.1, sigma=1.0):
        self.K = K
        self.d = K * K
        self.grid_size = grid_size
        self.alpha = alpha
        self.s = s
        self.sigma = sigma

        # Build multi-index pairs (i1, i2) for i1, i2 in 1..K
        i1 = np.arange(1, K + 1)
        i2 = np.arange(1, K + 1)
        i1_grid, i2_grid = np.meshgrid(i1, i2, indexing="ij")
        self.i1 = i1_grid.ravel()  # (d,)
        self.i2 = i2_grid.ravel()  # (d,)

        # Eigenvalues: lambda_i = sigma * (alpha + pi^2*((i1+0.5)^2 + (i2+0.5)^2))^(-s/2)
        arg = alpha + np.pi**2 * ((self.i1 + 0.5) ** 2 + (self.i2 + 0.5) ** 2)
        self.eigenvalues = sigma * arg ** (-s / 2)  # (d,)

        # Precompute eigenfunctions on grid
        x = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size
        x1_grid, x2_grid = np.meshgrid(x, x, indexing="ij")  # (grid_size, grid_size)

        # phi_i(x) = 2 * cos(pi*(i1+0.5)*x1) * cos(pi*(i2+0.5)*x2)
        self.phi = np.zeros((self.d, grid_size, grid_size))
        for k in range(self.d):
            self.phi[k] = 2.0 * (
                np.cos(np.pi * (self.i1[k] + 0.5) * x1_grid)
                * np.cos(np.pi * (self.i2[k] + 0.5) * x2_grid)
            )

    def sample(self, N, seed=None):
        """Sample N KL coefficient vectors u_kl ~ N(0, I)."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((N, self.d)).astype(np.float32)

    def reconstruct(self, u_kl):
        """Reconstruct spatial field: u(x) = sum_i u_kl_i * lambda_i * phi_i(x)."""
        u_kl = np.asarray(u_kl, dtype=np.float32)
        original_shape = u_kl.shape[:-1]
        u_kl_flat = u_kl.reshape(-1, self.d)
        weighted = u_kl_flat * self.eigenvalues[None, :]
        u_field = np.einsum("ni,ijk->njk", weighted, self.phi)
        return u_field.reshape(*original_shape, self.grid_size, self.grid_size)

    def score(self, u_kl):
        """Prior score: grad_u_kl log p(u_kl) = -u_kl."""
        return -np.asarray(u_kl, dtype=np.float32)

    def score_torch(self, u_kl):
        """Prior score as torch tensor."""
        return -u_kl


def create_beskos_sensors(grid_size, n_sensors=33):
    """Create Beskos-style sensor layout: 1 center + (n_sensors-1) on full-domain circle.

    The circle radius is (grid_size - 1) / (2 * grid_size) in [0,1] coordinates,
    which places sensors near the domain boundary.

    Args:
        grid_size: PDE grid resolution.
        n_sensors: Total number of sensors (default 33).

    Returns:
        sensor_ix: (n_sensors,) grid x-indices.
        sensor_iy: (n_sensors,) grid y-indices.
        sensor_x: (n_sensors,) physical x-coordinates in [0,1].
        sensor_y: (n_sensors,) physical y-coordinates in [0,1].
    """
    dx = 1.0 / grid_size

    # Center sensor
    center_x = 0.5
    center_y = 0.5

    # Circle sensors: radius spans nearly the full domain
    radius = (grid_size - 1) / (2.0 * grid_size)
    n_circle = n_sensors - 1
    angles = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
    circle_x = 0.5 + radius * np.cos(angles)
    circle_y = 0.5 + radius * np.sin(angles)

    sensor_x = np.concatenate([[center_x], circle_x])
    sensor_y = np.concatenate([[center_y], circle_y])

    # Convert to grid indices
    sensor_ix = np.clip((sensor_x / dx).astype(int), 0, grid_size - 1)
    sensor_iy = np.clip((sensor_y / dx).astype(int), 0, grid_size - 1)

    return sensor_ix, sensor_iy, sensor_x, sensor_y


class DarcyForwardModel:
    """Forward model for Darcy flow with Beskos sensor placement.

    Uses create_beskos_sensors() for the sensor layout.

    Args:
        kl_prior: DarcyKLPrior instance.
        n_sensors: Number of sensor locations (default 33).
        time_steps: Number of pseudo-timesteps for PDE solver.
    """

    def __init__(self, kl_prior, n_sensors=33, time_steps=5000):
        self.kl_prior = kl_prior
        self.grid_size = kl_prior.grid_size
        self.time_steps = time_steps
        self.n_sensors = n_sensors

        # Initialize Devito solver
        if GroundwaterEquation is None:
            raise RuntimeError(
                "The external Devito groundwater solver is not available. "
                "Install it and/or set the CNCV_GROUNDWATER_DIR environment "
                "variable to point at the 'groundwater' package. It is only "
                "required to (re)generate Darcy data."
            )
        self.eq = GroundwaterEquation(self.grid_size)
        self.f_source = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Beskos sensor layout
        self.sensor_ix, self.sensor_iy, self.sensor_x, self.sensor_y = \
            create_beskos_sensors(self.grid_size, n_sensors)

    def solve_pressure(self, u_field, return_devito=False):
        """Solve PDE for a single log-permeability field."""
        if return_devito:
            p_devito = self.eq.eval_fwd_op(
                self.f_source,
                np.asarray(u_field, dtype=np.float32),
                time_steps=self.time_steps,
                return_array=False,
            )
            p_np = np.array(p_devito.data[1])
            return p_np, p_devito
        else:
            p = self.eq.eval_fwd_op(
                self.f_source,
                np.asarray(u_field, dtype=np.float32),
                time_steps=self.time_steps,
            )
            return p

    def extract_observations(self, p_field):
        """Extract pressure at sensor locations."""
        return p_field[self.sensor_ix, self.sensor_iy].astype(np.float32)

    def forward(self, u_kl):
        """Full forward map: KL coefficients -> observations."""
        u_field = self.kl_prior.reconstruct(u_kl)
        p_field = self.solve_pressure(u_field)
        return self.extract_observations(p_field)

    def forward_batch(self, u_kl_batch, verbose=True):
        """Forward map for a batch of KL coefficients."""
        N = u_kl_batch.shape[0]
        y_batch = np.zeros((N, self.n_sensors), dtype=np.float32)
        u_fields = self.kl_prior.reconstruct(u_kl_batch)

        for i in range(N):
            p_field = self.solve_pressure(u_fields[i])
            y_batch[i] = self.extract_observations(p_field)
            if verbose and (i + 1) % 500 == 0:
                print(f"  Forward solve {i + 1}/{N}")

        return y_batch

    def gradient_wrt_field(self, u_field, y_obs, p_field_np, p_devito, sigma_y):
        """Compute gradient of likelihood w.r.t. log-permeability field via adjoint."""
        y_pred = self.extract_observations(p_field_np)
        residual_sensors = (y_pred - y_obs) / (sigma_y**2)

        residual_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for k in range(self.n_sensors):
            residual_grid[self.sensor_ix[k], self.sensor_iy[k]] += residual_sensors[k]

        grad_field = self.eq.compute_gradient(
            np.asarray(u_field, dtype=np.float32),
            residual_grid,
            p_devito,
        )
        return grad_field


def compute_score_darcy(u_kl, y_obs, forward_model, kl_prior, sigma_y):
    """Compute posterior score for Darcy flow.

    Works with any prior/forward model that has the same interface.
    """
    u_field = kl_prior.reconstruct(u_kl)
    p_field_np, p_devito = forward_model.solve_pressure(u_field, return_devito=True)

    grad_field = forward_model.gradient_wrt_field(
        u_field, y_obs, p_field_np, p_devito, sigma_y
    )

    grad_kl = np.zeros(kl_prior.d, dtype=np.float32)
    for k in range(kl_prior.d):
        grad_kl[k] = kl_prior.eigenvalues[k] * np.sum(grad_field * kl_prior.phi[k])

    score_lik = -grad_kl
    score_prior = -u_kl.astype(np.float32)

    return score_lik + score_prior


def compute_score_darcy_batch(u_kl_batch, y_obs_batch, forward_model, kl_prior,
                                  sigma_y, verbose=True):
    """Batch posterior score computation."""
    N = u_kl_batch.shape[0]
    scores = np.zeros_like(u_kl_batch, dtype=np.float32)

    for i in range(N):
        scores[i] = compute_score_darcy(
            u_kl_batch[i], y_obs_batch[i], forward_model, kl_prior, sigma_y
        )
        if verbose and (i + 1) % 500 == 0:
            print(f"  Score computation {i + 1}/{N}")

    return scores


def beskos_truth(kl_prior):
    """Generate Beskos's fixed truth field.

    u_i = lambda_i^{1/2} * sin((i1 - 1/2)^2 + (i2 - 1/2)^2)
    for each mode (i1, i2) in the prior's mode set.

    Args:
        kl_prior: DarcyKLPrior instance.

    Returns:
        u_kl_truth: (d,) KL coefficients for the truth.
        u_field_truth: (grid_size, grid_size) reconstructed truth field.
    """
    # In this parameterization u_kl ~ N(0, I) and the field is
    # u(x) = sum_i u_kl_i * lambda_i * phi_i(x), so the Beskos coefficient is
    # u_i = lambda_i * u_kl_i. Beskos's truth u_i = lambda_i^{1/2} sin(...),
    # hence u_kl_i = u_i / lambda_i = lambda_i^{-1/2} sin((i1-0.5)^2 + (i2-0.5)^2).
    eigenvalues = kl_prior.eigenvalues
    i1 = kl_prior.i1
    i2 = kl_prior.i2

    arg = (i1 - 0.5) ** 2 + (i2 - 0.5) ** 2
    u_kl_truth = (eigenvalues ** (-0.5)) * np.sin(arg)
    u_kl_truth = u_kl_truth.astype(np.float32)

    u_field_truth = kl_prior.reconstruct(u_kl_truth[None])[0]

    return u_kl_truth, u_field_truth


def setup_darcy_problem(K=10, grid_size=40, sigma_y=0.01, n_sensors=33,
                            time_steps=5000):
    """Set up the Beskos-correct Darcy flow inverse problem."""
    kl_prior = DarcyKLPrior(K=K, grid_size=grid_size)
    forward_model = DarcyForwardModel(
        kl_prior, n_sensors=n_sensors, time_steps=time_steps,
    )

    return {
        "kl_prior": kl_prior,
        "forward_model": forward_model,
        "sigma_y": sigma_y,
        "d": kl_prior.d,
        "n_obs": n_sensors,
    }
