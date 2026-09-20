"""Open-system dynamics for HubbardBath Milestone 03.

This module builds a finite-dimensional Davies reference Lindblad generator
inside the fixed-particle-number Hubbard sector. Bath couplings preserve total
particle number, transition rates obey thermal detailed balance, and
Bohr-frequency components are constructed from spectral projectors so exact
degeneracies do not introduce arbitrary eigenvector-basis dependence.

The construction remains a controlled small-system reference bath, not a
microscopic material-environment derivation.
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


def energy_eigenspaces(
    hamiltonian: Array,
    energy_tol: float = 1e-10,
) -> list[tuple[float, Array]]:
    """Return basis-invariant spectral projectors of H.

    Eigenvalues that differ by at most energy_tol are grouped into one
    degenerate eigenspace. Each returned projector is therefore invariant
    under arbitrary unitary rotations of the eigenvectors inside that
    degenerate subspace.
    """
    if energy_tol <= 0:
        raise ValueError("energy_tol must be positive")
    if (
        hamiltonian.ndim != 2
        or hamiltonian.shape[0] != hamiltonian.shape[1]
    ):
        raise ValueError("hamiltonian must be square")
    if not np.allclose(
        hamiltonian,
        hamiltonian.conj().T,
        atol=1e-12,
    ):
        raise ValueError("hamiltonian must be Hermitian")

    energies, vectors = np.linalg.eigh(hamiltonian)
    groups: list[list[int]] = []

    for index, energy in enumerate(energies):
        if not groups:
            groups.append([index])
            continue

        reference = float(np.mean(energies[groups[-1]]))
        if abs(float(energy) - reference) <= energy_tol:
            groups[-1].append(index)
        else:
            groups.append([index])

    eigenspaces: list[tuple[float, Array]] = []
    for indices in groups:
        subspace = vectors[:, indices]
        projector = subspace @ subspace.conj().T
        representative_energy = float(np.mean(energies[indices]))
        eigenspaces.append((representative_energy, projector))

    return eigenspaces


def _add_frequency_component(
    grouped: list[tuple[float, Array]],
    omega: float,
    component: Array,
    frequency_tol: float,
) -> None:
    """Accumulate matrix contributions with numerically equal frequencies."""
    for index, (existing_omega, existing_component) in enumerate(grouped):
        if abs(existing_omega - omega) <= frequency_tol:
            grouped[index] = (
                0.5 * (existing_omega + omega),
                existing_component + component,
            )
            return
    grouped.append((float(omega), component.copy()))


def bohr_frequency_components(
    hamiltonian: Array,
    coupling: Array,
    *,
    energy_tol: float = 1e-10,
    frequency_tol: float = 1e-10,
    matrix_element_tol: float = 1e-12,
) -> dict[float, Array]:
    """Return basis-invariant Davies components A(omega).

    The convention is

        [H, A(omega)] = -omega A(omega),  omega >= 0.

    For positive frequency,

        A(omega) = sum_{E_high-E_low=omega}
                   Pi_low A Pi_high,

    so A(omega) lowers system energy by omega. The zero-frequency component is

        A(0) = sum_E Pi_E A Pi_E,

    which retains the complete action inside degenerate energy subspaces.

    Contributions sharing one Bohr frequency are summed before a Lindblad
    jump operator is created. This is the key difference from constructing
    rank-one jumps from arbitrary individual eigenvectors.
    """
    if coupling.shape != hamiltonian.shape:
        raise ValueError("coupling operator must match H")
    if frequency_tol <= 0:
        raise ValueError("frequency_tol must be positive")
    if matrix_element_tol < 0:
        raise ValueError("matrix_element_tol cannot be negative")
    if not np.allclose(coupling, coupling.conj().T, atol=1e-12):
        raise ValueError(
            "Davies reference construction expects Hermitian couplings"
        )

    eigenspaces = energy_eigenspaces(
        hamiltonian,
        energy_tol=energy_tol,
    )
    dimension = hamiltonian.shape[0]

    grouped: list[tuple[float, Array]] = []

    zero_component = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )
    for _, projector in eigenspaces:
        zero_component += projector @ coupling @ projector

    if np.linalg.norm(zero_component, ord="fro") > matrix_element_tol:
        grouped.append((0.0, zero_component))

    for low_index, (energy_low, projector_low) in enumerate(eigenspaces):
        for energy_high, projector_high in eigenspaces[low_index + 1 :]:
            omega = float(energy_high - energy_low)
            if omega <= energy_tol:
                continue

            component = projector_low @ coupling @ projector_high
            if np.linalg.norm(component, ord="fro") <= matrix_element_tol:
                continue

            _add_frequency_component(
                grouped,
                omega,
                component,
                frequency_tol,
            )

    grouped.sort(key=lambda item: item[0])
    return {float(omega): component for omega, component in grouped}


def detailed_balance_jump_operators(
    hamiltonian: Array,
    beta: float,
    couplings: list[Array],
    rate_scale: float = 0.2,
    matrix_element_tol: float = 1e-12,
    energy_tol: float = 1e-10,
    frequency_tol: float = 1e-10,
) -> list[Array]:
    """Build a projector-based Davies reference generator.

    For each Hermitian system coupling A_a, the coupling is decomposed into
    basis-invariant Bohr-frequency components A_a(omega). A flat positive bath
    spectrum is used as the reference model:

        gamma_down(omega) = rate_scale
        gamma_up(omega)   = rate_scale * exp(-beta * omega)

    for omega > 0, so the KMS/detailed-balance ratio is

        gamma_up / gamma_down = exp(-beta * omega).

    Zero-frequency components are included with rate rate_scale.

    This remains a controlled reference bath rather than a microscopic
    material-environment derivation, but unlike the earlier pairwise
    eigenvector construction it is invariant under basis rotations inside
    exactly degenerate energy eigenspaces.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if rate_scale <= 0:
        raise ValueError("rate_scale must be positive")

    for coupling in couplings:
        if coupling.shape != hamiltonian.shape:
            raise ValueError("all coupling operators must match H")

    jumps: list[Array] = []

    for coupling in couplings:
        components = bohr_frequency_components(
            hamiltonian,
            coupling,
            energy_tol=energy_tol,
            frequency_tol=frequency_tol,
            matrix_element_tol=matrix_element_tol,
        )

        for omega, component in components.items():
            if omega <= energy_tol:
                jumps.append(np.sqrt(rate_scale) * component)
                continue

            jumps.append(np.sqrt(rate_scale) * component)
            jumps.append(
                np.sqrt(rate_scale * np.exp(-beta * omega))
                * component.conj().T
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
