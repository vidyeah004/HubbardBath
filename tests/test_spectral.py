import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.open_systems import build_thermal_liouvillian, doublon_initial_state
from src.spectral import (
    liouvillian_spectral_gap,
    loglog_power_law_fit,
    persistent_mixing_time,
    stationary_mode_count,
    thermalisation_metrics,
)


def reference_generator(U=4.0, beta=1.0):
    h_full = two_site_hubbard_hamiltonian(t=1.0, U=U)
    h_half, _ = project_to_particle_sector(h_full, 4, 2)
    generator, _, rho_beta = build_thermal_liouvillian(
        h_half,
        beta=beta,
        rate_scale=0.2,
    )
    return generator, rho_beta


def test_reference_liouvillian_has_positive_gap():
    generator, _ = reference_generator()
    gap = liouvillian_spectral_gap(generator)
    assert gap > 0.0
    assert stationary_mode_count(generator) == 1


def test_persistent_mixing_time_ignores_transient_crossing():
    times = np.arange(6, dtype=float)
    distances = np.array([0.5, 0.2, 0.009, 0.02, 0.008, 0.005])
    assert persistent_mixing_time(times, distances, epsilon=0.01) == 4.0


def test_persistent_mixing_time_returns_nan_if_not_reached():
    times = np.arange(4, dtype=float)
    distances = np.array([0.4, 0.2, 0.08, 0.02])
    value = persistent_mixing_time(times, distances, epsilon=0.01)
    assert np.isnan(value)


def test_reference_case_reaches_mixing_threshold():
    generator, rho_beta = reference_generator()
    metrics = thermalisation_metrics(
        generator,
        doublon_initial_state(),
        rho_beta,
        epsilon=1e-2,
        horizon_in_inverse_gaps=12.0,
        num_time_points=241,
    )
    assert np.isfinite(metrics["mixing_time"])
    assert metrics["final_trace_distance"] < 1e-2
    assert metrics["gap_times_mixing_time"] > 0.0


def test_power_law_fit_recovers_known_exponent():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 3.0 * x**1.5
    fit = loglog_power_law_fit(x, y)
    np.testing.assert_allclose(fit["exponent"], 1.5, atol=1e-12)
    np.testing.assert_allclose(fit["prefactor"], 3.0, atol=1e-12)
    np.testing.assert_allclose(fit["r_squared"], 1.0, atol=1e-12)
