import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.thermal import (
    average_double_occupancy_operator,
    canonical_thermal_observables,
    gibbs_state,
    nearest_neighbor_spin_z_correlation_operator,
    project_operator_to_sector,
    von_neumann_entropy,
)


def half_filled_objects(U=4.0):
    h_full = two_site_hubbard_hamiltonian(t=1.0, U=U)
    h_half, _ = project_to_particle_sector(
        h_full, num_modes=4, particle_number=2
    )
    d_half = project_operator_to_sector(
        average_double_occupancy_operator(), 4, 2
    )
    c_half = project_operator_to_sector(
        nearest_neighbor_spin_z_correlation_operator(), 4, 2
    )
    return h_half, d_half, c_half


def test_gibbs_state_is_valid_density_matrix():
    h, _, _ = half_filled_objects()
    rho = gibbs_state(h, beta=1.25)

    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() >= -1e-12


def test_beta_zero_is_maximally_mixed_in_sector():
    h, _, _ = half_filled_objects()
    rho = gibbs_state(h, beta=0.0)
    dim = h.shape[0]
    np.testing.assert_allclose(rho, np.eye(dim) / dim, atol=1e-12)


def test_beta_zero_entropy_is_log_dimension():
    h, _, _ = half_filled_objects()
    rho = gibbs_state(h, beta=0.0)
    np.testing.assert_allclose(
        von_neumann_entropy(rho), np.log(h.shape[0]), atol=1e-12
    )


def test_low_temperature_energy_approaches_ground_state():
    h, d, c = half_filled_objects(U=4.0)
    obs = canonical_thermal_observables(h, 20.0, d, c)
    ground_energy = np.linalg.eigvalsh(h)[0]
    np.testing.assert_allclose(obs["energy"], ground_energy, atol=1e-6)


def test_observables_stay_in_physical_ranges():
    h, d, c = half_filled_objects(U=4.0)
    obs = canonical_thermal_observables(h, 1.0, d, c)

    assert 0.0 <= obs["double_occupancy"] <= 0.5 + 1e-12
    assert -0.25 - 1e-12 <= obs["spin_z_correlation"] <= 0.25 + 1e-12


def test_negative_beta_is_rejected():
    h, _, _ = half_filled_objects()
    try:
        gibbs_state(h, beta=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative beta should raise ValueError")
