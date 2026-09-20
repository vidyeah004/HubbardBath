import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.open_systems import (
    bohr_frequency_components,
    build_thermal_liouvillian,
    default_bath_couplings,
    density_matrix_derivative,
    doublon_initial_state,
    energy_eigenspaces,
    evolve_density_matrix,
    trace_distance,
)


def half_filled_hamiltonian(U=4.0):
    h_full = two_site_hubbard_hamiltonian(t=1.0, U=U)
    h_half, _ = project_to_particle_sector(
        h_full, num_modes=4, particle_number=2
    )
    return h_half


def test_gibbs_state_is_stationary_under_thermal_liouvillian():
    h = half_filled_hamiltonian()
    generator, _, rho_beta = build_thermal_liouvillian(
        h, beta=1.0, rate_scale=0.2
    )
    residual = np.linalg.norm(
        density_matrix_derivative(generator, rho_beta)
    )
    assert residual < 1e-10


def test_liouvillian_has_one_stationary_mode_for_reference_case():
    h = half_filled_hamiltonian()
    generator, _, _ = build_thermal_liouvillian(
        h, beta=1.0, rate_scale=0.2
    )
    eigenvalues = np.linalg.eigvals(generator)
    assert np.sum(np.abs(eigenvalues) < 1e-9) == 1
    assert eigenvalues.real.max() < 1e-9


def test_evolution_preserves_density_matrix_properties():
    h = half_filled_hamiltonian()
    generator, _, _ = build_thermal_liouvillian(
        h, beta=1.0, rate_scale=0.2
    )
    rho0 = doublon_initial_state()
    times = np.linspace(0.0, 20.0, 21)
    states = evolve_density_matrix(generator, rho0, times)

    for rho in states:
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)
        assert np.linalg.eigvalsh(rho).min() >= -1e-10


def test_trace_distance_contracts_toward_gibbs_state():
    h = half_filled_hamiltonian()
    generator, _, rho_beta = build_thermal_liouvillian(
        h, beta=1.0, rate_scale=0.2
    )
    rho0 = doublon_initial_state()
    times = np.linspace(0.0, 120.0, 121)
    states = evolve_density_matrix(generator, rho0, times)

    distances = np.array([
        trace_distance(rho, rho_beta)
        for rho in states
    ])

    assert distances[-1] < 1e-2
    assert np.max(np.diff(distances)) < 1e-9


def test_trace_distance_is_zero_for_identical_states():
    rho = doublon_initial_state()
    assert trace_distance(rho, rho) < 1e-14



def test_energy_eigenspaces_are_complete_orthogonal_projectors():
    h = half_filled_hamiltonian(U=4.0)
    eigenspaces = energy_eigenspaces(h)

    projectors = [projector for _, projector in eigenspaces]
    identity = np.eye(h.shape[0], dtype=complex)

    np.testing.assert_allclose(
        sum(projectors),
        identity,
        atol=1e-12,
    )

    ranks = []
    for i, projector_i in enumerate(projectors):
        np.testing.assert_allclose(
            projector_i @ projector_i,
            projector_i,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            projector_i,
            projector_i.conj().T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            h @ projector_i,
            projector_i @ h,
            atol=1e-12,
        )
        ranks.append(int(round(np.trace(projector_i).real)))

        for j, projector_j in enumerate(projectors):
            if i != j:
                np.testing.assert_allclose(
                    projector_i @ projector_j,
                    np.zeros_like(h),
                    atol=1e-12,
                )

    assert sum(ranks) == h.shape[0]
    assert max(ranks) > 1


def test_bohr_components_reconstruct_each_hermitian_coupling():
    h = half_filled_hamiltonian(U=4.0)
    couplings = default_bath_couplings(particle_number=2)

    for coupling in couplings:
        components = bohr_frequency_components(h, coupling)
        reconstructed = np.zeros_like(coupling, dtype=complex)

        for omega, component in components.items():
            commutator = h @ component - component @ h
            np.testing.assert_allclose(
                commutator,
                -omega * component,
                atol=1e-10,
            )

            if abs(omega) < 1e-10:
                reconstructed += component
            else:
                reconstructed += component + component.conj().T

        np.testing.assert_allclose(
            reconstructed,
            coupling,
            atol=1e-10,
        )


def test_projector_davies_gibbs_stationarity_across_degenerate_grid():
    for U in [0.0, 2.0, 4.0, 8.0]:
        h = half_filled_hamiltonian(U=U)
        for beta in [0.25, 1.0, 4.0]:
            generator, _, rho_beta = build_thermal_liouvillian(
                h,
                beta=beta,
                rate_scale=0.2,
            )
            residual = np.linalg.norm(
                density_matrix_derivative(
                    generator,
                    rho_beta,
                )
            )
            assert residual < 1e-9


def test_projector_davies_has_unique_stationary_mode_on_small_grid():
    for U in [0.0, 4.0, 8.0]:
        h = half_filled_hamiltonian(U=U)
        for beta in [0.5, 2.0]:
            generator, _, _ = build_thermal_liouvillian(
                h,
                beta=beta,
                rate_scale=0.2,
            )
            eigenvalues = np.linalg.eigvals(generator)
            assert np.sum(np.abs(eigenvalues) < 1e-8) == 1
            assert eigenvalues.real.max() < 1e-9
