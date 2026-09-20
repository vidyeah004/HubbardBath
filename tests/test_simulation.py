import numpy as np

from src.qubit_mapping import (
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.simulation import (
    computational_basis_state,
    convergence_exponent,
    exact_propagator,
    first_order_trotter,
    naive_pauli_exponential_count,
    operator_norm_error,
    pauli_exponential,
    second_order_trotter,
    simulation_metrics,
)


def reference_terms():
    return remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )


def test_pauli_exponential_is_unitary():
    unitary = pauli_exponential("XZXI", -0.5, 0.7)
    identity = np.eye(unitary.shape[0])
    np.testing.assert_allclose(
        unitary.conj().T @ unitary,
        identity,
        atol=1e-12,
    )


def test_product_formulas_are_exact_at_zero_time():
    terms = reference_terms()
    exact = exact_propagator(terms, 0.0)
    first = first_order_trotter(terms, 0.0, steps=3)
    second = second_order_trotter(terms, 0.0, steps=3)

    np.testing.assert_allclose(first, exact, atol=1e-12)
    np.testing.assert_allclose(second, exact, atol=1e-12)


def test_trotter_error_decreases_with_refinement():
    terms = reference_terms()
    exact = exact_propagator(terms, time=1.5)

    first_1 = operator_norm_error(
        first_order_trotter(terms, 1.5, 1),
        exact,
    )
    first_32 = operator_norm_error(
        first_order_trotter(terms, 1.5, 32),
        exact,
    )
    second_1 = operator_norm_error(
        second_order_trotter(terms, 1.5, 1),
        exact,
    )
    second_32 = operator_norm_error(
        second_order_trotter(terms, 1.5, 32),
        exact,
    )

    assert first_32 < first_1
    assert second_32 < second_1
    assert second_32 < first_32


def test_observed_asymptotic_orders_match_product_formula_orders():
    terms = reference_terms()
    exact = exact_propagator(terms, time=1.5)
    steps = np.array([8, 16, 32, 64])

    first_errors = np.array([
        operator_norm_error(
            first_order_trotter(terms, 1.5, int(step)),
            exact,
        )
        for step in steps
    ])
    second_errors = np.array([
        operator_norm_error(
            second_order_trotter(terms, 1.5, int(step)),
            exact,
        )
        for step in steps
    ])

    first_slope = convergence_exponent(steps, first_errors)
    second_slope = convergence_exponent(steps, second_errors)

    assert -1.2 < first_slope < -0.8
    assert -2.2 < second_slope < -1.8


def test_state_infidelity_improves_with_steps():
    terms = reference_terms()
    state = computational_basis_state("1100")

    coarse = simulation_metrics(
        terms,
        time=1.5,
        steps=2,
        initial_state=state,
    )
    fine = simulation_metrics(
        terms,
        time=1.5,
        steps=32,
        initial_state=state,
    )

    assert (
        fine["first_order_state_infidelity"]
        < coarse["first_order_state_infidelity"]
    )
    assert (
        fine["second_order_state_infidelity"]
        < coarse["second_order_state_infidelity"]
    )


def test_naive_resource_count():
    assert naive_pauli_exponential_count(11, 8, order=1) == 88
    assert naive_pauli_exponential_count(11, 8, order=2) == 176
