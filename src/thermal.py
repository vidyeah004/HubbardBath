"""Finite-temperature utilities and observables for HubbardBath.

Milestone 02 works in a fixed-particle-number sector (canonical ensemble).
The functions here are intentionally dense/exact because the two-site model
is tiny; larger milestones will replace these with sparse methods as needed.
"""
from __future__ import annotations

import numpy as np

from src.hubbard import number_operator, sector_indices

Array = np.ndarray


def gibbs_state(hamiltonian: Array, beta: float) -> Array:
    """Return rho_beta = exp(-beta H) / Z using an eigen-decomposition.

    A ground-energy shift keeps the Boltzmann weights numerically stable and
    cancels exactly in the normalization.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12):
        raise ValueError("hamiltonian must be Hermitian")

    energies, vectors = np.linalg.eigh(hamiltonian)
    shifted = energies - energies.min()
    weights = np.exp(-beta * shifted)
    probabilities = weights / weights.sum()

    rho = (vectors * probabilities) @ vectors.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


def expectation(rho: Array, operator: Array) -> float:
    """Return Tr(rho O), requiring a real result up to numerical noise."""
    if rho.shape != operator.shape:
        raise ValueError("rho and operator must have the same shape")
    value = np.trace(rho @ operator)
    if abs(value.imag) > 1e-10:
        raise ValueError("expectation value has a non-negligible imaginary part")
    return float(value.real)


def von_neumann_entropy(rho: Array) -> float:
    """Return S(rho) = -Tr(rho log rho) in natural-log units."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    eigenvalues = np.clip(eigenvalues.real, 0.0, 1.0)
    nonzero = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(nonzero * np.log(nonzero)))


def project_operator_to_sector(
    operator: Array,
    num_modes: int,
    particle_number: int,
) -> Array:
    """Project an operator onto a fixed-particle-number Fock subspace."""
    expected_dim = 2**num_modes
    if operator.shape != (expected_dim, expected_dim):
        raise ValueError("operator shape does not match num_modes")
    indices = sector_indices(num_modes, particle_number)
    return operator[np.ix_(indices, indices)]


def average_double_occupancy_operator() -> Array:
    """Return (D_0 + D_1)/2 for the two-site spinful Hubbard model."""
    n0_up = number_operator(4, 0)
    n0_down = number_operator(4, 1)
    n1_up = number_operator(4, 2)
    n1_down = number_operator(4, 3)
    return 0.5 * (n0_up @ n0_down + n1_up @ n1_down)


def site_spin_z_operator(site: int) -> Array:
    """Return S^z_i = (n_{i,up} - n_{i,down})/2 for site 0 or 1."""
    if site not in (0, 1):
        raise ValueError("site must be 0 or 1")
    up_mode = 2 * site
    down_mode = up_mode + 1
    return 0.5 * (
        number_operator(4, up_mode) - number_operator(4, down_mode)
    )


def nearest_neighbor_spin_z_correlation_operator() -> Array:
    """Return S^z_0 S^z_1 for the two-site model."""
    return site_spin_z_operator(0) @ site_spin_z_operator(1)


def canonical_thermal_observables(
    hamiltonian_sector: Array,
    beta: float,
    double_occupancy_sector: Array,
    spin_correlation_sector: Array,
) -> dict[str, float]:
    """Evaluate the core Milestone 02 observables in a fixed-N sector."""
    rho = gibbs_state(hamiltonian_sector, beta)
    return {
        "energy": expectation(rho, hamiltonian_sector),
        "entropy": von_neumann_entropy(rho),
        "double_occupancy": expectation(rho, double_occupancy_sector),
        "spin_z_correlation": expectation(rho, spin_correlation_sector),
    }
