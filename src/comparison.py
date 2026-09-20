"""Physical-versus-algorithmic thermalisation analysis for Milestone 09.

This module compares two independently defined finite-system difficulty
measures on the same half-filled two-site Fermi-Hubbard target:

Physical route:
    detailed-balance Lindbladian -> spectral gap -> mixing time.

Algorithmic route:
    full four-qubit block-encoding normalization -> bounded thermal
    polynomial -> minimum Chebyshev degree at fixed scalar tolerance.

The comparison is descriptive. A fixed bath rate scale, initial state, finite
system size, and chosen numerical tolerances are part of the definition.
Correlation does not imply that either route causes or universally predicts
the other.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.hubbard import (
    project_to_particle_sector,
    two_site_hubbard_hamiltonian,
)
from src.open_systems import (
    build_thermal_liouvillian,
    doublon_initial_state,
)
from src.qubit_mapping import (
    lcu_normalization,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.spectral import stationary_mode_count, thermalisation_metrics
from src.thermal_polynomial import (
    minimum_degree_for_error,
    sector_gibbs_trace_distance_error,
)


def comparison_case(
    U: float,
    beta: float,
    *,
    physical_epsilon: float = 1e-2,
    algorithmic_epsilon: float = 1e-4,
    rate_scale: float = 0.2,
    horizon_in_inverse_gaps: float = 12.0,
    num_time_points: int = 241,
    max_polynomial_degree: int = 256,
) -> dict[str, float | int | bool]:
    """Compute both difficulty routes for one (U/t, beta t) point.

    The physical route is evaluated in the N=2 sector. The algorithmic
    polynomial is built for the full four-qubit Hamiltonian because that is
    what Milestone 07 block-encodes, but its Gibbs-state validation is then
    conditioned on the same N=2 sector before comparison.
    """
    if U < 0:
        raise ValueError("U must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")

    h_full = two_site_hubbard_hamiltonian(
        t=1.0,
        U=float(U),
        mu=0.0,
    )
    h_half, _ = project_to_particle_sector(
        h_full,
        num_modes=4,
        particle_number=2,
    )

    generator, jumps, rho_beta = build_thermal_liouvillian(
        h_half,
        beta=float(beta),
        rate_scale=rate_scale,
        particle_number=2,
    )
    stationary_modes = stationary_mode_count(generator)

    physical = thermalisation_metrics(
        generator,
        doublon_initial_state(),
        rho_beta,
        epsilon=physical_epsilon,
        horizon_in_inverse_gaps=horizon_in_inverse_gaps,
        num_time_points=num_time_points,
    )

    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(
            t=1.0,
            U=float(U),
            mu=0.0,
        )
    )
    alpha = lcu_normalization(terms)
    polynomial = minimum_degree_for_error(
        beta=float(beta),
        alpha=alpha,
        epsilon=algorithmic_epsilon,
        max_degree=max_polynomial_degree,
    )
    sector_gibbs_error = sector_gibbs_trace_distance_error(
        h_full,
        polynomial,
        num_modes=4,
        particle_number=2,
    )

    comparison_valid = (
        stationary_modes == 1
        and np.isfinite(physical["mixing_time"])
        and sector_gibbs_error < 5e-2
    )

    return {
        "U_over_t": float(U),
        "beta_t": float(beta),
        "physical_epsilon": float(physical_epsilon),
        "algorithmic_scalar_epsilon": float(algorithmic_epsilon),
        "bath_rate_scale": float(rate_scale),
        "number_of_jump_operators": int(len(jumps)),
        "stationary_mode_count": int(stationary_modes),
        "liouvillian_gap": float(physical["gap"]),
        "inverse_liouvillian_gap": float(physical["inverse_gap"]),
        "mixing_time": float(physical["mixing_time"]),
        "gap_times_mixing_time": float(
            physical["gap_times_mixing_time"]
        ),
        "final_physical_trace_distance": float(
            physical["final_trace_distance"]
        ),
        "number_of_pauli_terms": int(len(terms)),
        "lcu_alpha": float(alpha),
        "beta_alpha": float(beta * alpha),
        "thermal_polynomial_degree": int(polynomial.degree),
        "thermal_polynomial_scalar_error": float(
            polynomial.max_scalar_error
        ),
        "half_filled_gibbs_trace_distance": float(
            sector_gibbs_error
        ),
        "comparison_valid": bool(comparison_valid),
    }


def _finite_xy(
    rows: Iterable[dict],
    x_key: str,
    y_key: str,
    *,
    valid_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    for row in rows:
        if valid_only and not bool(row.get("comparison_valid", True)):
            continue
        x = float(row[x_key])
        y = float(row[y_key])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def correlation_summary(
    rows: Iterable[dict],
    x_key: str,
    y_key: str,
    *,
    valid_only: bool = True,
) -> dict[str, float | int | str | None]:
    """Return Pearson and Spearman summaries for two recorded metrics."""
    x, y = _finite_xy(
        rows,
        x_key,
        y_key,
        valid_only=valid_only,
    )

    result: dict[str, float | int | str | None] = {
        "x": x_key,
        "y": y_key,
        "n": int(len(x)),
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }

    if len(x) < 3:
        return result
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return result

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    result.update({
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    })
    return result


def stratified_correlation_summaries(
    rows: list[dict],
    group_key: str,
    x_key: str,
    y_key: str,
) -> list[dict[str, object]]:
    """Compute the same correlation separately within each group value."""
    values = sorted({row[group_key] for row in rows})
    summaries: list[dict[str, object]] = []

    for value in values:
        group_rows = [
            row for row in rows
            if row[group_key] == value
        ]
        summary = correlation_summary(
            group_rows,
            x_key,
            y_key,
            valid_only=True,
        )
        summaries.append({
            "group_key": group_key,
            "group_value": value,
            **summary,
        })

    return summaries


def comparison_summary(
    rows: list[dict],
) -> dict[str, object]:
    """Return the main overall and stratified M9 statistics."""
    valid_rows = [
        row for row in rows
        if bool(row.get("comparison_valid", True))
    ]

    return {
        "total_cases": len(rows),
        "valid_cases": len(valid_rows),
        "mixing_time_vs_degree": correlation_summary(
            rows,
            "mixing_time",
            "thermal_polynomial_degree",
        ),
        "inverse_gap_vs_degree": correlation_summary(
            rows,
            "inverse_liouvillian_gap",
            "thermal_polynomial_degree",
        ),
        "beta_alpha_vs_degree": correlation_summary(
            rows,
            "beta_alpha",
            "thermal_polynomial_degree",
        ),
        "within_beta_mixing_vs_degree":
            stratified_correlation_summaries(
                rows,
                "beta_t",
                "mixing_time",
                "thermal_polynomial_degree",
            ),
        "within_U_mixing_vs_degree":
            stratified_correlation_summaries(
                rows,
                "U_over_t",
                "mixing_time",
                "thermal_polynomial_degree",
            ),
    }
