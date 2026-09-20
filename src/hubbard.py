"""Small-system Fermi-Hubbard utilities used in Milestone 1.

The implementation is intentionally explicit. We construct fermionic
annihilation operators with the Jordan-Wigner representation so that every
sign can be inspected and tested before moving to larger simulations.
"""
from __future__ import annotations

from math import comb
from typing import Iterable

import numpy as np

Array = np.ndarray

I2 = np.eye(2, dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def kron_all(operators: Iterable[Array]) -> Array:
    """Kronecker product of a sequence of matrices."""
    out = np.array([[1.0]], dtype=complex)
    for operator in operators:
        out = np.kron(out, operator)
    return out


def annihilation_operator(num_modes: int, mode: int) -> Array:
    """Return fermionic annihilation operator c_mode via Jordan-Wigner."""
    if num_modes < 1:
        raise ValueError("num_modes must be positive")
    if not 0 <= mode < num_modes:
        raise IndexError("mode out of range")

    operators = [
        Z if j < mode else SIGMA_MINUS if j == mode else I2
        for j in range(num_modes)
    ]
    return kron_all(operators)


def creation_operator(num_modes: int, mode: int) -> Array:
    """Return fermionic creation operator c_mode^dagger."""
    return annihilation_operator(num_modes, mode).conj().T


def number_operator(num_modes: int, mode: int) -> Array:
    """Return number operator n_mode = c_mode^dagger c_mode."""
    c = annihilation_operator(num_modes, mode)
    return c.conj().T @ c


def total_number_operator(num_modes: int) -> Array:
    """Return N = sum_j n_j."""
    dim = 2**num_modes
    total = np.zeros((dim, dim), dtype=complex)
    for mode in range(num_modes):
        total += number_operator(num_modes, mode)
    return total


def anticommutator(a: Array, b: Array) -> Array:
    return a @ b + b @ a


def commutator(a: Array, b: Array) -> Array:
    return a @ b - b @ a


def two_site_hubbard_hamiltonian(
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
) -> Array:
    """Construct the two-site spinful Fermi-Hubbard Hamiltonian.

    Mode order:
      0 = site 0, spin up
      1 = site 0, spin down
      2 = site 1, spin up
      3 = site 1, spin down
    """
    num_modes = 4
    dim = 2**num_modes
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    for left, right in ((0, 2), (1, 3)):
        c_left = annihilation_operator(num_modes, left)
        c_right = annihilation_operator(num_modes, right)
        hamiltonian += -t * (
            c_left.conj().T @ c_right + c_right.conj().T @ c_left
        )

    n0_up = number_operator(num_modes, 0)
    n0_down = number_operator(num_modes, 1)
    n1_up = number_operator(num_modes, 2)
    n1_down = number_operator(num_modes, 3)
    hamiltonian += U * (n0_up @ n0_down + n1_up @ n1_down)
    hamiltonian += -mu * total_number_operator(num_modes)

    return hamiltonian


def basis_particle_number(index: int, num_modes: int) -> int:
    if not 0 <= index < 2**num_modes:
        raise IndexError("basis index out of range")
    return int(index).bit_count()


def sector_indices(num_modes: int, particle_number: int) -> list[int]:
    if not 0 <= particle_number <= num_modes:
        raise ValueError("particle_number must lie between 0 and num_modes")
    indices = [
        i for i in range(2**num_modes)
        if basis_particle_number(i, num_modes) == particle_number
    ]
    assert len(indices) == comb(num_modes, particle_number)
    return indices


def project_to_particle_sector(
    hamiltonian: Array,
    num_modes: int,
    particle_number: int,
) -> tuple[Array, list[int]]:
    expected_dim = 2**num_modes
    if hamiltonian.shape != (expected_dim, expected_dim):
        raise ValueError("hamiltonian shape does not match num_modes")
    indices = sector_indices(num_modes, particle_number)
    return hamiltonian[np.ix_(indices, indices)], indices


def fock_label(index: int, num_modes: int = 4) -> str:
    if not 0 <= index < 2**num_modes:
        raise IndexError("basis index out of range")
    return "|" + format(index, f"0{num_modes}b") + ">"
