import numpy as np

from src.block_encoding import (
    block_encoding_error,
    block_encoding_unitary,
    build_lcu_data,
    prepare_target_state,
    prepare_unitary,
    projected_action,
    select_unitary,
    top_left_system_block,
    unitarity_error,
)
from src.qubit_mapping import (
    reconstruct_from_pauli_terms,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.simulation import computational_basis_state


def reference_terms():
    return remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )


def test_prepare_maps_zero_to_lcu_amplitudes():
    data = build_lcu_data(reference_terms())
    prepare = prepare_unitary(data)
    e0 = np.zeros(data.ancilla_dimension, dtype=complex)
    e0[0] = 1.0

    np.testing.assert_allclose(
        prepare @ e0,
        prepare_target_state(data),
        atol=1e-12,
    )
    assert unitarity_error(prepare) < 1e-12


def test_select_is_unitary():
    data = build_lcu_data(reference_terms())
    select = select_unitary(data)
    assert unitarity_error(select) < 1e-12


def test_block_encoding_is_unitary_and_exact():
    terms = reference_terms()
    unitary, data = block_encoding_unitary(terms)

    assert unitarity_error(unitary) < 1e-12
    assert block_encoding_error(terms) < 1e-12

    encoded = top_left_system_block(unitary, data.system_dimension)
    hamiltonian = reconstruct_from_pauli_terms(terms)

    np.testing.assert_allclose(
        encoded,
        hamiltonian / data.alpha,
        atol=1e-12,
    )


def test_reference_lcu_dimensions_and_normalization():
    data = build_lcu_data(reference_terms())

    assert len(data.labels) == 11
    assert data.ancilla_qubits == 4
    assert data.ancilla_dimension == 16
    assert data.system_qubits == 4
    assert data.system_dimension == 16
    np.testing.assert_allclose(data.alpha, 10.0, atol=1e-12)
    np.testing.assert_allclose(data.probabilities.sum(), 1.0, atol=1e-12)


def test_projected_action_matches_h_over_alpha():
    terms = reference_terms()
    unitary, data = block_encoding_unitary(terms)
    state = computational_basis_state("1100")

    projected, probability = projected_action(
        unitary,
        data,
        state,
    )

    hamiltonian = reconstruct_from_pauli_terms(terms)
    expected = (hamiltonian / data.alpha) @ state

    np.testing.assert_allclose(projected, expected, atol=1e-12)
    np.testing.assert_allclose(
        probability,
        np.vdot(expected, expected).real,
        atol=1e-12,
    )


def test_normalized_hamiltonian_has_operator_norm_at_most_one():
    terms = reference_terms()
    data = build_lcu_data(terms)
    hamiltonian = reconstruct_from_pauli_terms(terms)
    normalized_norm = np.linalg.norm(hamiltonian / data.alpha, ord=2)

    assert normalized_norm <= 1.0 + 1e-12
