"""Open-system dynamics for HubbardBath Milestone 03.

This module builds a finite-dimensional, Davies-inspired Lindblad generator
inside the fixed-particle-number Hubbard sector. Bath couplings preserve total
particle number and transition rates obey thermal detailed balance.

The construction is a controlled small-system model, not a microscopic bath
derivation. Its purpose is to establish and validate thermalisation before
Milestone 04 studies Liouvillian spectra and mixing-time scaling.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from src.hubbard import (
    annihilation_operator,
    number_operator,
    sector_indices,
)
from src.thermal import gibbs_state, project_operator_to_sector

Array = np.ndarray


def local_spin_x_operator(site: int) -> Array:
    """Return c^†_up c_down + c^†_down c_up on one Hubbard site."""
    if site not in (0, 1):
        raise ValueError("site must be 0 or 1")
    up = 2 * site
    down = up + 1
    c_up = annihilation_operator(4, up)
    c_down = annihilation_operator(4, down)
    return c_up.conj().T @ c_down + c_down.conj().T @ c_up


def default_bath_couplings(
    particle_number: int = 2,
) -> list[Array]:
    """Return number-conserving local bath-coupling operators.

    We include four local mode-density operators plus local spin-flip
    operators on each site. Together they couple charge/spin fluctuations
    while remaining inside a fixed total-particle-number sector.
    """
    couplings_full = [number_operator(4, mode) for mode in range(4)]
    couplings_full += [
        local_spin_x_operator(0),
        local_spin_x_operator(1),
    ]
    return [
        project_operator_to_sector(op, 4, particle_number)
        for op in couplings_full
    ]


def detailed_balance_jump_operators(
    hamiltonian: Array,
    beta: float,
    couplings: list[Array],
    rate_scale: float = 0.2,
    matrix_element_tol: float = 1e-12,
    energy_tol: float = 1e-10,
) -> list[Array]:
    """Build energy-basis jump operators satisfying thermal detailed balance.

    For E_high > E_low we use

        gamma_up / gamma_down = exp[-beta (E_high - E_low)].

    The downward rate uses a flat reference bath spectrum multiplied by the
    total squared matrix element of the supplied system-bath couplings.

    Degenerate transitions are assigned equal forward/backward rates.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if rate_scale <= 0:
        raise ValueError("rate_scale must be positive")

    energies, vectors = np.linalg.eigh(hamiltonian)
    dimension = hamiltonian.shape[0]

    for coupling in couplings:
        if coupling.shape != hamiltonian.shape:
            raise ValueError("all coupling operators must match H")

    couplings_energy = [
        vectors.conj().T @ coupling @ vectors
        for coupling in couplings
    ]

    jumps: list[Array] = []

    for low in range(dimension):
        for high in range(low + 1, dimension):
            omega = float(energies[high] - energies[low])
            strength = float(
                sum(
                    abs(coupling[low, high]) ** 2
                    for coupling in couplings_energy
                )
            )

            if strength <= matrix_element_tol:
                continue

            base_rate = rate_scale * strength

            if omega > energy_tol:
                rate_down = base_rate
                rate_up = base_rate * np.exp(-beta * omega)
            else:
                rate_down = base_rate
                rate_up = base_rate

            ket_low = vectors[:, low:low + 1]
            ket_high = vectors[:, high:high + 1]

            jumps.append(
                np.sqrt(rate_down) * (ket_low @ ket_high.conj().T)
            )
            jumps.append(
                np.sqrt(rate_up) * (ket_high @ ket_low.conj().T)
            )

    if not jumps:
        raise ValueError("bath couplings generated no transitions")

    return jumps


def liouvillian(
    hamiltonian: Array,
    jump_operators: list[Array],
) -> Array:
    """Return the dense Liouvillian matrix using column-major vectorization."""
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=complex)

    generator = -1j * (
        np.kron(identity, hamiltonian)
        - np.kron(hamiltonian.T, identity)
    )

    for jump in jump_operators:
        if jump.shape != hamiltonian.shape:
            raise ValueError("jump operator shape does not match H")

        rate_operator = jump.conj().T @ jump
        generator += (
            np.kron(jump.conj(), jump)
            - 0.5 * np.kron(identity, rate_operator)
            - 0.5 * np.kron(rate_operator.T, identity)
        )

    return generator


def density_matrix_derivative(
    generator: Array,
    rho: Array,
) -> Array:
    """Apply a Liouvillian matrix to a density matrix."""
    dimension = rho.shape[0]
    derivative = generator @ rho.reshape(-1, order="F")
    return derivative.reshape((dimension, dimension), order="F")


def evolve_density_matrix(
    generator: Array,
    rho0: Array,
    times: Array,
) -> list[Array]:
    """Evolve rho0 under exp(L t) at uniformly spaced time points."""
    if len(times) < 2:
        raise ValueError("provide at least two time points")
    if not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("times must be uniformly spaced")

    dimension = rho0.shape[0]
    initial_vector = rho0.reshape(-1, order="F")

    vectors = expm_multiply(
        generator,
        initial_vector,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )

    states: list[Array] = []
    for vector in vectors:
        rho = vector.reshape((dimension, dimension), order="F")
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho)
        states.append(rho)

    return states


def trace_distance(rho: Array, sigma: Array) -> float:
    """Return D(rho,sigma)=1/2 ||rho-sigma||_1 for Hermitian states."""
    if rho.shape != sigma.shape:
        raise ValueError("rho and sigma must have the same shape")
    delta = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta))))


def doublon_initial_state(
    particle_number: int = 2,
    site: int = 0,
) -> Array:
    """Return a pure doublon density matrix in the projected sector.

    For half filling, the state has both spins on one site and the other site
    empty. It is deliberately far from equilibrium when U > 0.
    """
    if particle_number != 2:
        raise ValueError("doublon helper currently assumes two particles")
    if site not in (0, 1):
        raise ValueError("site must be 0 or 1")

    # Mode order: (0 up, 0 down, 1 up, 1 down).
    full_basis_index = 12 if site == 0 else 3
    indices = sector_indices(4, particle_number)
    projected_index = indices.index(full_basis_index)

    dimension = len(indices)
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[projected_index, projected_index] = 1.0
    return rho


def build_thermal_liouvillian(
    hamiltonian: Array,
    beta: float,
    rate_scale: float = 0.2,
    particle_number: int = 2,
) -> tuple[Array, list[Array], Array]:
    """Convenience constructor returning (L, jumps, rho_beta)."""
    couplings = default_bath_couplings(particle_number)
    jumps = detailed_balance_jump_operators(
        hamiltonian,
        beta,
        couplings,
        rate_scale=rate_scale,
    )
    generator = liouvillian(hamiltonian, jumps)
    rho_beta = gibbs_state(hamiltonian, beta)
    return generator, jumps, rho_beta
