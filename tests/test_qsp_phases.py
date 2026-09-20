import numpy as np
from numpy.polynomial.chebyshev import chebval

from src.hubbard import two_site_hubbard_hamiltonian
from src.qsp_phases import (
    qsp_imaginary_response,
    qsp_unitary_from_phases,
    synthesize_parity_component,
    synthesize_thermal_parity_components,
    thermal_qsp_reconstruction_error,
)
from src.qubit_mapping import (
    lcu_normalization,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.thermal_polynomial import minimum_degree_for_sector_gibbs_error


def reference_thermal_polynomial():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0, mu=0.0)
    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )
    alpha = lcu_normalization(terms)
    approximation, _ = minimum_degree_for_sector_gibbs_error(
        h,
        beta=1.0,
        alpha=alpha,
        target_trace_distance=1e-2,
        max_degree=64,
    )
    return approximation


def test_explicit_qsp_phase_product_is_unitary():
    phases = np.array([0.2, -0.3, 0.4])
    for x in [-1.0, -0.4, 0.0, 0.7, 1.0]:
        unitary = qsp_unitary_from_phases(x, phases)
        np.testing.assert_allclose(
            unitary.conj().T @ unitary,
            np.eye(2),
            atol=1e-12,
        )


def test_even_chebyshev_component_synthesizes_to_qsp_response():
    coefficients = np.array([0.2, 0.0, -0.1, 0.0, 0.05])
    synthesis = synthesize_parity_component(
        coefficients,
        parity=0,
        crit=1e-12,
    )
    samples = np.linspace(-1.0, 1.0, 401)
    response = qsp_imaginary_response(samples, synthesis.full_phases)
    target = chebval(samples, coefficients)

    assert synthesis.max_response_error < 1e-9
    np.testing.assert_allclose(response, target, atol=1e-9)


def test_reference_thermal_even_odd_phases_reconstruct_polynomial():
    approximation = reference_thermal_polynomial()
    even, odd = synthesize_thermal_parity_components(
        approximation,
        crit=1e-12,
    )

    assert even.parity == 0
    assert odd.parity == 1
    assert even.max_response_error < 1e-8
    assert odd.max_response_error < 1e-8
    assert (
        thermal_qsp_reconstruction_error(
            approximation,
            even,
            odd,
        )
        < 1e-8
    )


def test_phase_count_matches_component_degree_plus_one():
    approximation = reference_thermal_polynomial()
    even, odd = synthesize_thermal_parity_components(approximation)

    assert len(even.full_phases) == even.polynomial_degree + 1
    assert len(odd.full_phases) == odd.polynomial_degree + 1
