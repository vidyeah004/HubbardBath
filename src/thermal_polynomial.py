"""Bounded thermal polynomials for HubbardBath Milestone 08.

Milestone 07 block-encodes X = H / alpha with spectrum inside [-1, 1].
A direct thermal amplitude exp(-beta H / 2) can exceed one when H has
negative eigenvalues, which is incompatible with a bounded QSVT polynomial.

We therefore use the globally rescaled function

    f(x) = exp[-(beta * alpha / 2) * (x + 1)],   x in [-1, 1].

For X = H/alpha,

    f(X) = exp[-beta (H + alpha I) / 2]
         = exp[-beta alpha / 2] exp[-beta H / 2].

The extra scalar factor cancels when the squared amplitude operator is
normalized into a Gibbs state, while f is bounded by one on the entire QSVT
domain [-1, 1].

This module builds Chebyshev approximations, checks boundedness, separates even
and odd parity components, applies the polynomial to small dense matrices, and
measures the resulting Gibbs-state error. It does NOT synthesize QSP phases;
that distinction is kept explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate, chebval

from src.hubbard import project_to_particle_sector, sector_indices
from src.open_systems import trace_distance
from src.thermal import gibbs_state

Array = np.ndarray


@dataclass(frozen=True)
class ThermalPolynomial:
    beta: float
    alpha: float
    degree: int
    coefficients: Array
    max_scalar_error: float
    max_absolute_value: float
    safety_scale: float


def bounded_thermal_function(
    x: Array | float,
    beta: float,
    alpha: float,
) -> Array:
    """Return exp[-beta*alpha*(x+1)/2], bounded in [0, 1] on [-1, 1]."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * beta * alpha * (x + 1.0))


def chebyshev_thermal_polynomial(
    beta: float,
    alpha: float,
    degree: int,
    grid_size: int = 4001,
    enforce_unit_bound: bool = True,
) -> ThermalPolynomial:
    """Approximate the bounded thermal function on the full interval [-1,1].

    The polynomial is obtained by Chebyshev interpolation. If numerical
    overshoot makes |p(x)| slightly exceed one on the validation grid, all
    coefficients are rescaled by the same factor so the approximation is
    QSVT-compatible at the grid level. The resulting approximation error is
    reported after that scaling.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if grid_size < 101:
        raise ValueError("grid_size must be at least 101")

    gamma = 0.5 * beta * alpha
    coefficients = np.asarray(
        chebinterpolate(
            lambda points: np.exp(-gamma * (points + 1.0)),
            degree,
        ),
        dtype=float,
    )

    grid = np.linspace(-1.0, 1.0, grid_size)
    target = bounded_thermal_function(grid, beta, alpha)
    values = chebval(grid, coefficients)

    max_abs = float(np.max(np.abs(values)))
    safety_scale = 1.0

    if enforce_unit_bound and max_abs > 1.0:
        safety_scale = max_abs
        coefficients = coefficients / safety_scale
        values = values / safety_scale
        max_abs = float(np.max(np.abs(values)))

    error = float(np.max(np.abs(values - target)))

    return ThermalPolynomial(
        beta=float(beta),
        alpha=float(alpha),
        degree=int(degree),
        coefficients=coefficients,
        max_scalar_error=error,
        max_absolute_value=max_abs,
        safety_scale=float(safety_scale),
    )


def minimum_degree_for_error(
    beta: float,
    alpha: float,
    epsilon: float,
    max_degree: int = 256,
    grid_size: int = 4001,
) -> ThermalPolynomial:
    """Return the lowest-degree bounded Chebyshev approximation meeting epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    for degree in range(max_degree + 1):
        approximation = chebyshev_thermal_polynomial(
            beta=beta,
            alpha=alpha,
            degree=degree,
            grid_size=grid_size,
            enforce_unit_bound=True,
        )
        if approximation.max_scalar_error <= epsilon:
            return approximation

    raise ValueError(
        f"no polynomial up to degree {max_degree} reached epsilon={epsilon}"
    )


def parity_components(
    coefficients: Array,
) -> tuple[Array, Array]:
    """Split Chebyshev coefficients into even- and odd-parity components.

    T_k(-x) = (-1)^k T_k(x), so even Chebyshev indices form an even
    polynomial and odd indices form an odd polynomial.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    even = coefficients.copy()
    odd = coefficients.copy()

    even[1::2] = 0.0
    odd[0::2] = 0.0

    return even, odd


def parity_reconstruction_error(
    coefficients: Array,
    grid_size: int = 4001,
) -> float:
    """Check p(x) = p_even(x) + p_odd(x) on a dense scalar grid."""
    grid = np.linspace(-1.0, 1.0, grid_size)
    even, odd = parity_components(coefficients)
    original = chebval(grid, coefficients)
    reconstructed = chebval(grid, even) + chebval(grid, odd)
    return float(np.max(np.abs(original - reconstructed)))


def chebyshev_matrix_evaluate(
    matrix: Array,
    coefficients: Array,
) -> Array:
    """Evaluate sum_k c_k T_k(matrix) with the Chebyshev recurrence."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("coefficients must be a non-empty vector")

    dimension = matrix.shape[0]
    identity = np.eye(dimension, dtype=complex)

    t0 = identity
    result = coefficients[0] * t0

    if len(coefficients) == 1:
        return result

    t1 = matrix.copy()
    result = result + coefficients[1] * t1

    for k in range(2, len(coefficients)):
        tk = 2.0 * matrix @ t1 - t0
        result = result + coefficients[k] * tk
        t0, t1 = t1, tk

    return result


def exact_shifted_thermal_amplitude(
    hamiltonian: Array,
    beta: float,
    alpha: float,
) -> Array:
    """Return exp[-beta (H + alpha I)/2] by exact diagonalization."""
    energies, vectors = np.linalg.eigh(hamiltonian)
    amplitudes = np.exp(-0.5 * beta * (energies + alpha))
    return (vectors * amplitudes) @ vectors.conj().T


def approximate_thermal_amplitude(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
) -> Array:
    """Apply p(H/alpha) using the stored Chebyshev coefficients."""
    normalized = hamiltonian / approximation.alpha
    return chebyshev_matrix_evaluate(
        normalized,
        approximation.coefficients,
    )


def amplitude_operator_error(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
) -> float:
    """Return ||p(H/alpha) - exp[-beta(H+alpha I)/2]||_2."""
    approximate = approximate_thermal_amplitude(
        hamiltonian,
        approximation,
    )
    exact = exact_shifted_thermal_amplitude(
        hamiltonian,
        approximation.beta,
        approximation.alpha,
    )
    return float(np.linalg.norm(approximate - exact, ord=2))


def gibbs_state_from_amplitude(
    amplitude: Array,
) -> Array:
    """Normalize K K^dagger into a density matrix."""
    positive = amplitude @ amplitude.conj().T
    trace = np.trace(positive)
    if abs(trace) < 1e-15:
        raise ValueError("amplitude produced zero normalization")
    rho = positive / trace
    return 0.5 * (rho + rho.conj().T)


def approximate_gibbs_state(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
) -> Array:
    """Return the normalized Gibbs approximation induced by p(H/alpha)."""
    amplitude = approximate_thermal_amplitude(
        hamiltonian,
        approximation,
    )
    return gibbs_state_from_amplitude(amplitude)


def gibbs_trace_distance_error(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
) -> float:
    """Trace distance between approximate and exact Gibbs states."""
    approximate = approximate_gibbs_state(
        hamiltonian,
        approximation,
    )
    exact = gibbs_state(
        hamiltonian,
        approximation.beta,
    )
    return trace_distance(approximate, exact)


def qsvt_readiness_report(
    approximation: ThermalPolynomial,
    grid_size: int = 4001,
) -> dict[str, float | int | bool]:
    """Summarize boundedness and parity structure relevant to QSVT.

    A single standard QSP/QSVT polynomial has a parity constraint tied to its
    degree. The thermal function is neither purely even nor purely odd, so a
    direct single-sequence implementation is not claimed here. Instead we
    expose the even and odd Chebyshev components that can be synthesized
    separately and combined using a standard ancilla/LCU construction.
    """
    grid = np.linspace(-1.0, 1.0, grid_size)
    even, odd = parity_components(approximation.coefficients)

    even_values = chebval(grid, even)
    odd_values = chebval(grid, odd)
    full_values = even_values + odd_values

    return {
        "degree": approximation.degree,
        "bounded_by_one_on_grid": bool(
            np.max(np.abs(full_values)) <= 1.0 + 1e-12
        ),
        "full_polynomial_max_abs": float(
            np.max(np.abs(full_values))
        ),
        "even_component_max_abs": float(
            np.max(np.abs(even_values))
        ),
        "odd_component_max_abs": float(
            np.max(np.abs(odd_values))
        ),
        "parity_reconstruction_error": parity_reconstruction_error(
            approximation.coefficients,
            grid_size=grid_size,
        ),
        "requires_parity_split_for_standard_qsvt": True,
    }



def sector_conditioned_gibbs_state_from_amplitude(
    amplitude: Array,
    num_modes: int,
    particle_number: int,
) -> Array:
    """Project K K^dagger into a fixed-N sector and renormalize.

    This is the canonical-sector version needed when the polynomial acts on
    the full Fock-space Hamiltonian but the physical comparison is performed
    at fixed particle number.
    """
    positive = amplitude @ amplitude.conj().T
    expected_dimension = 2**num_modes
    if positive.shape != (expected_dimension, expected_dimension):
        raise ValueError("amplitude dimension does not match num_modes")

    indices = sector_indices(num_modes, particle_number)
    block = positive[np.ix_(indices, indices)]
    trace = np.trace(block)

    if abs(trace) < 1e-15:
        raise ValueError("selected particle-number sector has zero weight")

    rho = block / trace
    return 0.5 * (rho + rho.conj().T)


def approximate_sector_gibbs_state(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
    num_modes: int,
    particle_number: int,
) -> Array:
    """Return the polynomial Gibbs approximation conditioned on fixed N."""
    amplitude = approximate_thermal_amplitude(
        hamiltonian,
        approximation,
    )
    return sector_conditioned_gibbs_state_from_amplitude(
        amplitude,
        num_modes=num_modes,
        particle_number=particle_number,
    )


def sector_gibbs_trace_distance_error(
    hamiltonian: Array,
    approximation: ThermalPolynomial,
    num_modes: int,
    particle_number: int,
) -> float:
    """Trace distance from the exact canonical Gibbs state in fixed N."""
    approximate = approximate_sector_gibbs_state(
        hamiltonian,
        approximation,
        num_modes=num_modes,
        particle_number=particle_number,
    )
    h_sector, _ = project_to_particle_sector(
        hamiltonian,
        num_modes=num_modes,
        particle_number=particle_number,
    )
    exact = gibbs_state(
        h_sector,
        approximation.beta,
    )
    return trace_distance(approximate, exact)
