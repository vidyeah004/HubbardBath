import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.open_systems import (
    build_thermal_liouvillian,
    density_matrix_derivative,
    doublon_initial_state,
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
