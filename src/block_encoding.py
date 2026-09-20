"""LCU block encoding for HubbardBath Milestone 07.

Given a Pauli decomposition

    H = sum_j alpha_j P_j,

define

    alpha = sum_j |alpha_j|.

PREPARE creates amplitudes sqrt(|alpha_j| / alpha), while SELECT applies
phase(alpha_j) P_j conditioned on ancilla label j. The standard LCU unitary

    U_H = (PREPARE^dagger ⊗ I) SELECT (PREPARE ⊗ I)

then satisfies

    (<0| ⊗ I) U_H (|0> ⊗ I) = H / alpha.

This module constructs the small dense reference unitary explicitly so the
identity can be verified numerically before moving to QSVT.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np

from src.qubit_mapping import (
    lcu_normalization,
    pauli_string_matrix,
    reconstruct_from_pauli_terms,
)

Array = np.ndarray


@dataclass(frozen=True)
class LCUData:
    """Classical data defining the LCU/block-encoding construction."""

    labels: tuple[str, ...]
    coefficients: Array
    probabilities: Array
    phases: Array
    alpha: float
    ancilla_qubits: int
    ancilla_dimension: int
    system_qubits: int
    system_dimension: int


def build_lcu_data(
    terms: dict[str, complex],
) -> LCUData:
    """Return normalized coefficient data for a non-empty Pauli sum."""
    if not terms:
        raise ValueError("terms cannot be empty")

    lengths = {len(label) for label in terms}
    if len(lengths) != 1:
        raise ValueError("all Pauli labels must have the same length")

    labels = tuple(terms.keys())
    coefficients = np.asarray(
        [complex(terms[label]) for label in labels],
        dtype=complex,
    )

    magnitudes = np.abs(coefficients)
    if np.any(magnitudes <= 0):
        raise ValueError("remove zero-coefficient terms before LCU encoding")

    alpha = lcu_normalization(terms)
    probabilities = magnitudes / alpha
    phases = coefficients / magnitudes

    number_of_terms = len(labels)
    ancilla_qubits = 0 if number_of_terms == 1 else ceil(log2(number_of_terms))
    ancilla_dimension = 2**ancilla_qubits

    system_qubits = next(iter(lengths))
    system_dimension = 2**system_qubits

    return LCUData(
        labels=labels,
        coefficients=coefficients,
        probabilities=probabilities,
        phases=phases,
        alpha=alpha,
        ancilla_qubits=ancilla_qubits,
        ancilla_dimension=ancilla_dimension,
        system_qubits=system_qubits,
        system_dimension=system_dimension,
    )


def prepare_target_state(
    data: LCUData,
) -> Array:
    """Return the padded PREPARE target state on the ancilla register."""
    target = np.zeros(data.ancilla_dimension, dtype=complex)
    target[: len(data.labels)] = np.sqrt(data.probabilities)
    return target


def prepare_unitary(
    data: LCUData,
) -> Array:
    """Construct a dense unitary A such that A|0> = PREPARE target.

    Because the target amplitudes sqrt(|alpha_j|/alpha) are real and
    non-negative, a Householder reflection gives a compact exact reference
    construction.
    """
    target = prepare_target_state(data)
    e0 = np.zeros(data.ancilla_dimension, dtype=complex)
    e0[0] = 1.0

    if np.linalg.norm(target - e0) < 1e-14:
        return np.eye(data.ancilla_dimension, dtype=complex)

    vector = e0 - target
    vector /= np.linalg.norm(vector)

    unitary = (
        np.eye(data.ancilla_dimension, dtype=complex)
        - 2.0 * np.outer(vector, vector.conj())
    )

    return unitary


def select_unitary(
    data: LCUData,
) -> Array:
    """Construct SELECT = sum_j |j><j| ⊗ phase_j P_j.

    Unused computational-basis ancilla labels act as identity so the full
    operator remains unitary when the term count is not a power of two.
    """
    total_dimension = data.ancilla_dimension * data.system_dimension
    select = np.zeros((total_dimension, total_dimension), dtype=complex)
    system_identity = np.eye(data.system_dimension, dtype=complex)

    for ancilla_index in range(data.ancilla_dimension):
        start = ancilla_index * data.system_dimension
        stop = start + data.system_dimension

        if ancilla_index < len(data.labels):
            label = data.labels[ancilla_index]
            phase = data.phases[ancilla_index]
            block = phase * pauli_string_matrix(label)
        else:
            block = system_identity

        select[start:stop, start:stop] = block

    return select


def block_encoding_unitary(
    terms: dict[str, complex],
) -> tuple[Array, LCUData]:
    """Construct the dense standard-form LCU block encoding."""
    data = build_lcu_data(terms)
    prepare = prepare_unitary(data)
    select = select_unitary(data)

    system_identity = np.eye(data.system_dimension, dtype=complex)
    prepare_system = np.kron(prepare, system_identity)

    unitary = prepare_system.conj().T @ select @ prepare_system
    return unitary, data


def top_left_system_block(
    unitary: Array,
    system_dimension: int,
) -> Array:
    """Extract (<0|⊗I) U (|0>⊗I) in ancilla-first ordering."""
    return unitary[:system_dimension, :system_dimension]


def block_encoding_error(
    terms: dict[str, complex],
) -> float:
    """Return ||top-left(U_H) - H/alpha||_2."""
    unitary, data = block_encoding_unitary(terms)
    encoded_block = top_left_system_block(unitary, data.system_dimension)
    hamiltonian = reconstruct_from_pauli_terms(terms)
    target = hamiltonian / data.alpha
    return float(np.linalg.norm(encoded_block - target, ord=2))


def unitarity_error(
    unitary: Array,
) -> float:
    """Return ||U^dagger U - I||_2."""
    identity = np.eye(unitary.shape[0], dtype=complex)
    return float(
        np.linalg.norm(unitary.conj().T @ unitary - identity, ord=2)
    )


def projected_action(
    unitary: Array,
    data: LCUData,
    system_state: Array,
) -> tuple[Array, float]:
    """Apply U to |0>_a|psi> and project the ancilla back onto |0>.

    Returns the unnormalized projected system state and the corresponding
    postselection probability. For a correct block encoding,

        projected_state = (H / alpha) |psi>.
    """
    state = np.asarray(system_state, dtype=complex)
    if state.shape != (data.system_dimension,):
        raise ValueError("system_state has the wrong dimension")

    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("system_state must be nonzero")
    state = state / norm

    joint = np.zeros(
        data.ancilla_dimension * data.system_dimension,
        dtype=complex,
    )
    joint[:data.system_dimension] = state

    evolved = unitary @ joint
    projected = evolved[:data.system_dimension]
    probability = float(np.vdot(projected, projected).real)

    return projected, probability
