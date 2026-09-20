"""Hamiltonian-simulation tools for HubbardBath Milestone 06.

The four-qubit Hubbard Hamiltonian from Milestone 05 is represented as

    H = sum_j alpha_j P_j.

For each Pauli string P_j, P_j^2 = I, so its exponential is exact:

    exp(-i theta P_j) = cos(theta) I - i sin(theta) P_j.

This module compares exact dense time evolution with first-order Lie-Trotter
and second-order symmetric (Strang) product formulas.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.qubit_mapping import pauli_string_matrix, reconstruct_from_pauli_terms

Array = np.ndarray


def exact_propagator(
    terms: dict[str, complex],
    time: float,
) -> Array:
    """Return exp(-i H time) for the dense Pauli Hamiltonian."""
    hamiltonian = reconstruct_from_pauli_terms(terms)
    return expm(-1j * hamiltonian * time)


def pauli_exponential(
    label: str,
    coefficient: complex,
    delta_time: float,
) -> Array:
    """Return exp(-i coefficient * P * delta_time) exactly.

    Hermitian Pauli Hamiltonians require real coefficients.
    """
    coefficient = complex(coefficient)
    if abs(coefficient.imag) > 1e-12:
        raise ValueError("Hermitian Pauli terms require real coefficients")

    pauli = pauli_string_matrix(label)
    identity = np.eye(pauli.shape[0], dtype=complex)
    theta = float(coefficient.real) * float(delta_time)

    return np.cos(theta) * identity - 1j * np.sin(theta) * pauli


def first_order_trotter(
    terms: dict[str, complex],
    time: float,
    steps: int,
) -> Array:
    """First-order Lie-Trotter approximation to exp(-i H time)."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not terms:
        raise ValueError("terms cannot be empty")

    delta_time = float(time) / steps
    num_qubits = len(next(iter(terms)))
    step_unitary = np.eye(2**num_qubits, dtype=complex)

    for label, coefficient in terms.items():
        step_unitary = (
            step_unitary
            @ pauli_exponential(label, coefficient, delta_time)
        )

    return np.linalg.matrix_power(step_unitary, steps)


def second_order_trotter(
    terms: dict[str, complex],
    time: float,
    steps: int,
) -> Array:
    """Second-order symmetric/Strang product-formula approximation."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not terms:
        raise ValueError("terms cannot be empty")

    delta_time = float(time) / steps
    items = list(terms.items())
    num_qubits = len(items[0][0])
    step_unitary = np.eye(2**num_qubits, dtype=complex)

    for label, coefficient in items:
        step_unitary = (
            step_unitary
            @ pauli_exponential(label, coefficient, delta_time / 2.0)
        )

    for label, coefficient in reversed(items):
        step_unitary = (
            step_unitary
            @ pauli_exponential(label, coefficient, delta_time / 2.0)
        )

    return np.linalg.matrix_power(step_unitary, steps)


def operator_norm_error(
    approximate: Array,
    exact: Array,
) -> float:
    """Return ||U_approx - U_exact||_2."""
    if approximate.shape != exact.shape:
        raise ValueError("operators must have the same shape")
    return float(np.linalg.norm(approximate - exact, ord=2))


def computational_basis_state(
    bitstring: str,
) -> Array:
    """Return |bitstring> using the repository's left-to-right qubit order."""
    if not bitstring or any(bit not in "01" for bit in bitstring):
        raise ValueError("bitstring must contain only 0 and 1")
    dimension = 2 ** len(bitstring)
    state = np.zeros(dimension, dtype=complex)
    state[int(bitstring, 2)] = 1.0
    return state


def state_fidelity(
    state_a: Array,
    state_b: Array,
) -> float:
    """Return |<a|b>|^2 for normalized pure states."""
    norm_a = np.linalg.norm(state_a)
    norm_b = np.linalg.norm(state_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("state vectors must be nonzero")

    a = state_a / norm_a
    b = state_b / norm_b
    return float(abs(np.vdot(a, b)) ** 2)


def simulation_metrics(
    terms: dict[str, complex],
    time: float,
    steps: int,
    initial_state: Array | None = None,
) -> dict[str, float]:
    """Compare first/second-order product formulas with exact evolution."""
    exact = exact_propagator(terms, time)
    first = first_order_trotter(terms, time, steps)
    second = second_order_trotter(terms, time, steps)

    metrics = {
        "first_order_operator_error": operator_norm_error(first, exact),
        "second_order_operator_error": operator_norm_error(second, exact),
    }

    if initial_state is not None:
        exact_state = exact @ initial_state
        first_state = first @ initial_state
        second_state = second @ initial_state
        metrics.update({
            "first_order_state_infidelity": (
                1.0 - state_fidelity(exact_state, first_state)
            ),
            "second_order_state_infidelity": (
                1.0 - state_fidelity(exact_state, second_state)
            ),
        })

    return metrics


def convergence_exponent(
    steps: Array,
    errors: Array,
    tail_points: int = 4,
) -> float:
    """Fit error ~ steps^p on the asymptotic tail in log-log space."""
    steps = np.asarray(steps, dtype=float)
    errors = np.asarray(errors, dtype=float)

    mask = np.isfinite(steps) & np.isfinite(errors) & (steps > 0) & (errors > 0)
    steps = steps[mask]
    errors = errors[mask]

    if len(steps) < 2:
        raise ValueError("need at least two positive finite data points")

    tail_points = min(int(tail_points), len(steps))
    slope, _ = np.polyfit(
        np.log(steps[-tail_points:]),
        np.log(errors[-tail_points:]),
        1,
    )
    return float(slope)


def naive_pauli_exponential_count(
    number_of_terms: int,
    steps: int,
    order: int,
) -> int:
    """Simple resource proxy before circuit-level compilation.

    First order applies each term once per step. The symmetric second-order
    implementation applies each term twice per step. Later milestones will
    distinguish this from actual one- and two-qubit gate counts.
    """
    if number_of_terms < 1 or steps < 1:
        raise ValueError("number_of_terms and steps must be positive")
    if order == 1:
        return number_of_terms * steps
    if order == 2:
        return 2 * number_of_terms * steps
    raise ValueError("order must be 1 or 2")
