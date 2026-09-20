"""Spectral and mixing-time analysis for HubbardBath Milestone 04.

The central comparison is between the slowest non-stationary decay rate of
the Lindblad generator and the operational time required for a chosen initial
state to remain within a trace-distance tolerance of the Gibbs state.
"""
from __future__ import annotations

import numpy as np

from src.open_systems import evolve_density_matrix, trace_distance

Array = np.ndarray


def liouvillian_spectral_gap(
    generator: Array,
    real_tol: float = 1e-10,
) -> float:
    """Return the smallest strictly positive decay rate -Re(lambda).

    For a relaxing Lindbladian, the stationary eigenvalue is zero and all
    non-stationary modes have non-positive real parts. The gap used here is

        Delta_L = min_{Re(lambda)<0} -Re(lambda).

    The function raises if no decaying mode can be identified.
    """
    eigenvalues = np.linalg.eigvals(generator)
    decay_rates = -eigenvalues.real[eigenvalues.real < -real_tol]

    if decay_rates.size == 0:
        raise ValueError("no decaying Liouvillian mode identified")

    return float(decay_rates.min())


def stationary_mode_count(
    generator: Array,
    tol: float = 1e-8,
) -> int:
    """Count eigenvalues numerically indistinguishable from zero."""
    eigenvalues = np.linalg.eigvals(generator)
    return int(np.sum(np.abs(eigenvalues) < tol))


def persistent_mixing_time(
    times: Array,
    distances: Array,
    epsilon: float = 1e-2,
) -> float:
    """Return first sampled time after which distance stays <= epsilon.

    This is stricter than the first threshold crossing and avoids labeling a
    transient dip as thermal mixing. Returns NaN if the horizon is too short.
    """
    times = np.asarray(times, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if times.ndim != 1 or distances.ndim != 1:
        raise ValueError("times and distances must be one-dimensional")
    if len(times) != len(distances):
        raise ValueError("times and distances must have the same length")
    if len(times) == 0:
        raise ValueError("times and distances cannot be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted")

    suffix_max = np.maximum.accumulate(distances[::-1])[::-1]
    indices = np.flatnonzero(suffix_max <= epsilon)

    if len(indices) == 0:
        return float("nan")

    return float(times[indices[0]])


def thermalisation_metrics(
    generator: Array,
    rho0: Array,
    rho_beta: Array,
    epsilon: float = 1e-2,
    horizon_in_inverse_gaps: float = 12.0,
    num_time_points: int = 241,
) -> dict[str, float]:
    """Compute gap, trajectory, and operational mixing time for one case."""
    if horizon_in_inverse_gaps <= 0:
        raise ValueError("horizon_in_inverse_gaps must be positive")
    if num_time_points < 2:
        raise ValueError("num_time_points must be at least 2")

    gap = liouvillian_spectral_gap(generator)
    inverse_gap = 1.0 / gap
    times = np.linspace(
        0.0,
        horizon_in_inverse_gaps * inverse_gap,
        num_time_points,
    )

    states = evolve_density_matrix(generator, rho0, times)
    distances = np.asarray([
        trace_distance(rho, rho_beta)
        for rho in states
    ])

    mixing_time = persistent_mixing_time(
        times,
        distances,
        epsilon=epsilon,
    )

    return {
        "gap": gap,
        "inverse_gap": inverse_gap,
        "mixing_time": mixing_time,
        "gap_times_mixing_time": (
            gap * mixing_time if np.isfinite(mixing_time) else float("nan")
        ),
        "initial_trace_distance": float(distances[0]),
        "final_trace_distance": float(distances[-1]),
    }


def loglog_power_law_fit(
    x: Array,
    y: Array,
) -> dict[str, float]:
    """Fit y = prefactor * x**exponent in log-log space.

    Returns exponent, prefactor, and R^2 of the log-space linear fit.
    This is a descriptive finite-grid summary, not an asymptotic claim.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("need at least two positive finite data points")

    log_x = np.log(x)
    log_y = np.log(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = exponent * log_x + intercept

    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "exponent": float(exponent),
        "prefactor": float(np.exp(intercept)),
        "r_squared": float(r_squared),
    }
