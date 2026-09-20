import numpy as np

from src.hubbard import two_site_hubbard_hamiltonian
from src.qubit_mapping import (
    lcu_normalization,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.thermal_polynomial import (
    amplitude_operator_error,
    bounded_thermal_function,
    chebyshev_thermal_polynomial,
    gibbs_trace_distance_error,
    minimum_degree_for_error,
    parity_components,
    parity_reconstruction_error,
    qsvt_readiness_report,
)


def reference():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0, mu=0.0)
    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )
    alpha = lcu_normalization(terms)
    return h, alpha


def test_bounded_thermal_function_stays_in_unit_interval():
    _, alpha = reference()
    x = np.linspace(-1.0, 1.0, 1001)
    values = bounded_thermal_function(x, beta=1.0, alpha=alpha)

    assert np.min(values) >= 0.0
    assert np.max(values) <= 1.0 + 1e-15
    np.testing.assert_allclose(values[0], 1.0, atol=1e-12)


def test_minimum_degree_meets_reference_scalar_tolerance():
    _, alpha = reference()
    approximation = minimum_degree_for_error(
        beta=1.0,
        alpha=alpha,
        epsilon=1e-4,
        max_degree=64,
    )

    assert approximation.degree <= 16
    assert approximation.max_scalar_error <= 1e-4
    assert approximation.max_absolute_value <= 1.0 + 1e-12


def test_even_odd_parity_split_reconstructs_polynomial():
    _, alpha = reference()
    approximation = chebyshev_thermal_polynomial(
        beta=1.0,
        alpha=alpha,
        degree=10,
    )
    even, odd = parity_components(approximation.coefficients)

    assert np.allclose(even[1::2], 0.0)
    assert np.allclose(odd[0::2], 0.0)
    assert parity_reconstruction_error(
        approximation.coefficients
    ) < 1e-12


def test_matrix_amplitude_error_is_small_when_scalar_error_is_small():
    h, alpha = reference()
    approximation = minimum_degree_for_error(
        beta=1.0,
        alpha=alpha,
        epsilon=1e-4,
        max_degree=64,
    )

    assert amplitude_operator_error(h, approximation) <= 1.1e-4


def test_polynomial_amplitude_recovers_gibbs_state():
    h, alpha = reference()
    approximation = minimum_degree_for_error(
        beta=1.0,
        alpha=alpha,
        epsilon=1e-4,
        max_degree=64,
    )

    assert gibbs_trace_distance_error(h, approximation) < 1e-2


def test_higher_degree_improves_gibbs_state_accuracy():
    h, alpha = reference()
    coarse = chebyshev_thermal_polynomial(
        beta=1.0,
        alpha=alpha,
        degree=4,
    )
    fine = chebyshev_thermal_polynomial(
        beta=1.0,
        alpha=alpha,
        degree=12,
    )

    assert (
        gibbs_trace_distance_error(h, fine)
        < gibbs_trace_distance_error(h, coarse)
    )


def test_qsvt_readiness_report_is_bounded_and_flags_parity_split():
    _, alpha = reference()
    approximation = minimum_degree_for_error(
        beta=1.0,
        alpha=alpha,
        epsilon=1e-4,
        max_degree=64,
    )
    report = qsvt_readiness_report(approximation)

    assert report["bounded_by_one_on_grid"]
    assert report["parity_reconstruction_error"] < 1e-12
    assert report["requires_parity_split_for_standard_qsvt"]
