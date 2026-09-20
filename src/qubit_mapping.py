"""Jordan-Wigner and Pauli-string utilities for HubbardBath Milestone 05.

The existing fermionic code already constructs the Hubbard Hamiltonian through
Jordan-Wigner ladder operators. This module makes the qubit representation
explicit as a weighted sum of Pauli strings so that later milestones can use
Hamiltonian simulation, LCU/block encoding, and QSVT-oriented matrix functions.
"""
from __future__ import annotations

import itertools
import numpy as np

Array = np.ndarray

I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_all(operators: list[Array]) -> Array:
    """Kronecker product of matrices in left-to-right mode order."""
    out = np.array([[1.0]], dtype=complex)
    for operator in operators:
        out = np.kron(out, operator)
    return out


def pauli_string_matrix(label: str) -> Array:
    """Return the matrix represented by a label such as 'XZXI'."""
    if not label:
        raise ValueError("Pauli label cannot be empty")
    try:
        return kron_all([PAULI[symbol] for symbol in label])
    except KeyError as exc:
        raise ValueError(f"invalid Pauli symbol: {exc.args[0]}") from exc


def two_site_hubbard_pauli_terms(
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
) -> dict[str, float]:
    """Analytic Jordan-Wigner Pauli decomposition for the 4-mode model.

    Mode/qubit order:
      q0 = site 0 up
      q1 = site 0 down
      q2 = site 1 up
      q3 = site 1 down

    With n_j = (I - Z_j)/2, the interaction/chemical-potential terms become
    diagonal Z strings. Hopping across non-adjacent fermionic modes produces
    the Jordan-Wigner parity Z string between their endpoints.
    """
    return {
        "IIII": U / 2.0 - 2.0 * mu,
        "ZIII": -U / 4.0 + mu / 2.0,
        "IZII": -U / 4.0 + mu / 2.0,
        "IIZI": -U / 4.0 + mu / 2.0,
        "IIIZ": -U / 4.0 + mu / 2.0,
        "ZZII": U / 4.0,
        "IIZZ": U / 4.0,
        "XZXI": -t / 2.0,
        "YZYI": -t / 2.0,
        "IXZX": -t / 2.0,
        "IYZY": -t / 2.0,
    }


def remove_zero_terms(
    terms: dict[str, complex],
    tol: float = 1e-12,
) -> dict[str, complex]:
    """Drop coefficients whose absolute value is below tolerance."""
    return {
        label: coefficient
        for label, coefficient in terms.items()
        if abs(coefficient) > tol
    }


def reconstruct_from_pauli_terms(
    terms: dict[str, complex],
) -> Array:
    """Construct a dense Hamiltonian from a Pauli dictionary."""
    if not terms:
        raise ValueError("terms cannot be empty")

    lengths = {len(label) for label in terms}
    if len(lengths) != 1:
        raise ValueError("all Pauli labels must have equal length")

    num_qubits = next(iter(lengths))
    dimension = 2**num_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    for label, coefficient in terms.items():
        hamiltonian += coefficient * pauli_string_matrix(label)

    return hamiltonian


def dense_pauli_decomposition(
    operator: Array,
    coefficient_tol: float = 1e-12,
) -> dict[str, complex]:
    """Decompose a small 2^n by 2^n matrix in the Pauli basis.

    Coefficients use the Hilbert-Schmidt identity

        alpha_P = Tr(P^dagger A) / 2^n.

    This exact dense routine is intended only as a validation tool for small n.
    """
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")

    dimension = operator.shape[0]
    num_qubits = int(round(np.log2(dimension)))
    if 2**num_qubits != dimension:
        raise ValueError("operator dimension must be a power of two")

    terms: dict[str, complex] = {}
    for symbols in itertools.product("IXYZ", repeat=num_qubits):
        label = "".join(symbols)
        pauli = pauli_string_matrix(label)
        coefficient = np.trace(pauli.conj().T @ operator) / dimension
        if abs(coefficient) > coefficient_tol:
            if abs(coefficient.imag) < coefficient_tol:
                coefficient = float(coefficient.real)
            terms[label] = coefficient

    return terms


def lcu_normalization(terms: dict[str, complex]) -> float:
    """Return alpha = sum_j |alpha_j| for an LCU Pauli decomposition."""
    return float(sum(abs(coefficient) for coefficient in terms.values()))
