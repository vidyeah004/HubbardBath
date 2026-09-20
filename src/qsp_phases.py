"""Explicit symmetric-QSP phase synthesis for HubbardBath Milestone 10.

This module takes the parity-compatible Chebyshev components constructed in
Milestone 08, synthesizes QSP phase sequences using pyqsp's symmetric Newton
solver, and then independently reconstructs the signal response from the
returned full phase sequence.

Important distinction:
- each even/odd component is implemented by its own valid QSP sequence;
- their sum reconstructs the full thermal polynomial classically;
- a coherent ancilla/LCU combination of the two sequences is not yet compiled
  into a complete fault-tolerant QSVT circuit.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io

import numpy as np
from numpy.polynomial.chebyshev import chebval
from pyqsp import sym_qsp_opt

from src.thermal_polynomial import ThermalPolynomial, parity_components

Array = np.ndarray


@dataclass(frozen=True)
class QSPPhaseSynthesis:
    """Numerical record for one definite-parity QSP component."""

    parity: int
    polynomial_degree: int
    target_chebyshev_coefficients: Array
    parity_reduced_coefficients: Array
    reduced_phases: Array
    full_phases: Array
    solver_residual: float
    solver_iterations: int
    max_response_error: float


def _trim_reduced_coefficients(
    reduced: Array,
    tol: float = 1e-14,
) -> Array:
    """Remove trailing numerically zero highest-order coefficients."""
    reduced = np.asarray(reduced, dtype=float).copy()
    while len(reduced) > 1 and abs(reduced[-1]) <= tol:
        reduced = reduced[:-1]
    return reduced


def _full_coefficients_from_reduced(
    reduced: Array,
    parity: int,
) -> Array:
    """Expand parity-reduced Chebyshev coefficients into a full vector."""
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    reduced = np.asarray(reduced, dtype=float)
    if reduced.ndim != 1 or len(reduced) == 0:
        raise ValueError("reduced coefficients must be a non-empty vector")

    degree = parity + 2 * (len(reduced) - 1)
    full = np.zeros(degree + 1, dtype=float)
    full[parity::2] = reduced
    return full


def qsp_signal_unitary(
    x: float,
) -> Array:
    """Return the Wx-convention scalar QSP signal unitary."""
    x = float(x)
    if x < -1.0 - 1e-12 or x > 1.0 + 1e-12:
        raise ValueError("QSP signal x must lie in [-1, 1]")
    x = min(1.0, max(-1.0, x))
    off_diagonal = 1j * np.sqrt(max(0.0, 1.0 - x * x))
    return np.array(
        [
            [x, off_diagonal],
            [off_diagonal, x],
        ],
        dtype=complex,
    )


def qsp_phase_unitary(
    phi: float,
) -> Array:
    """Return diag(exp(i phi), exp(-i phi))."""
    return np.array(
        [
            [np.exp(1j * phi), 0.0],
            [0.0, np.exp(-1j * phi)],
        ],
        dtype=complex,
    )


def qsp_unitary_from_phases(
    x: float,
    phases: Array,
) -> Array:
    """Evaluate the symmetric-QSP unitary from an explicit phase sequence."""
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or len(phases) == 0:
        raise ValueError("phases must be a non-empty vector")

    unitary = qsp_phase_unitary(phases[0])
    if len(phases) == 1:
        return unitary

    signal = qsp_signal_unitary(x)
    for phi in phases[1:]:
        unitary = unitary @ signal @ qsp_phase_unitary(phi)

    return unitary


def qsp_imaginary_response(
    samples: Array,
    phases: Array,
) -> Array:
    """Return Im(<0|U_QSP(x)|0>) across scalar samples."""
    samples = np.asarray(samples, dtype=float)
    return np.asarray(
        [
            np.imag(qsp_unitary_from_phases(x, phases)[0, 0])
            for x in samples
        ],
        dtype=float,
    )


def synthesize_parity_component(
    coefficients: Array,
    parity: int,
    *,
    crit: float = 1e-12,
    maxiter: int = 100,
    grid_size: int = 2001,
    coefficient_tol: float = 1e-14,
) -> QSPPhaseSynthesis:
    """Synthesize one definite-parity Chebyshev polynomial into QSP phases.

    pyqsp's symmetric-QSP Newton solver expects the non-zero parity subsequence
    [c_parity, c_(parity+2), ...]. Its achieved target appears as the
    imaginary part of the top-left QSP matrix element in the Wx convention.
    """
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    if crit <= 0:
        raise ValueError("crit must be positive")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")

    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("coefficients must be a non-empty vector")

    forbidden = coefficients[1 - parity :: 2]
    if np.any(np.abs(forbidden) > coefficient_tol):
        raise ValueError("polynomial is not of the requested definite parity")

    reduced = _trim_reduced_coefficients(
        coefficients[parity::2],
        tol=coefficient_tol,
    )

    # pyqsp prints its Newton iteration diagnostics; keep library calls quiet
    # in tests/experiments while retaining the returned numerical residual.
    with redirect_stdout(io.StringIO()):
        reduced_phases, residual, iterations, protocol = (
            sym_qsp_opt.newton_solver(
                reduced,
                parity,
                crit=crit,
                maxiter=maxiter,
            )
        )

    full_phases = np.asarray(protocol.full_phases, dtype=float)
    target_full = _full_coefficients_from_reduced(reduced, parity)

    samples = np.linspace(-1.0, 1.0, grid_size)
    target_values = chebval(samples, target_full)
    response_values = qsp_imaginary_response(samples, full_phases)
    max_response_error = float(
        np.max(np.abs(response_values - target_values))
    )

    return QSPPhaseSynthesis(
        parity=int(parity),
        polynomial_degree=int(len(full_phases) - 1),
        target_chebyshev_coefficients=target_full,
        parity_reduced_coefficients=reduced,
        reduced_phases=np.asarray(reduced_phases, dtype=float),
        full_phases=full_phases,
        solver_residual=float(residual),
        solver_iterations=int(iterations),
        max_response_error=max_response_error,
    )


def synthesize_thermal_parity_components(
    approximation: ThermalPolynomial,
    *,
    crit: float = 1e-12,
    maxiter: int = 100,
    grid_size: int = 2001,
) -> tuple[QSPPhaseSynthesis, QSPPhaseSynthesis]:
    """Synthesize the even and odd pieces of one thermal polynomial."""
    even, odd = parity_components(approximation.coefficients)

    even_synthesis = synthesize_parity_component(
        even,
        parity=0,
        crit=crit,
        maxiter=maxiter,
        grid_size=grid_size,
    )
    odd_synthesis = synthesize_parity_component(
        odd,
        parity=1,
        crit=crit,
        maxiter=maxiter,
        grid_size=grid_size,
    )
    return even_synthesis, odd_synthesis


def reconstructed_thermal_response(
    samples: Array,
    even_synthesis: QSPPhaseSynthesis,
    odd_synthesis: QSPPhaseSynthesis,
) -> Array:
    """Add the two independently synthesized parity responses."""
    samples = np.asarray(samples, dtype=float)
    return (
        qsp_imaginary_response(samples, even_synthesis.full_phases)
        + qsp_imaginary_response(samples, odd_synthesis.full_phases)
    )


def thermal_qsp_reconstruction_error(
    approximation: ThermalPolynomial,
    even_synthesis: QSPPhaseSynthesis,
    odd_synthesis: QSPPhaseSynthesis,
    *,
    grid_size: int = 4001,
) -> float:
    """Compare summed QSP responses with the original thermal polynomial."""
    samples = np.linspace(-1.0, 1.0, grid_size)
    target = chebval(samples, approximation.coefficients)
    reconstructed = reconstructed_thermal_response(
        samples,
        even_synthesis,
        odd_synthesis,
    )
    return float(np.max(np.abs(reconstructed - target)))
