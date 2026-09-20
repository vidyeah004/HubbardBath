import numpy as np

from src.hubbard import two_site_hubbard_hamiltonian
from src.qubit_mapping import (
    dense_pauli_decomposition,
    lcu_normalization,
    pauli_string_matrix,
    reconstruct_from_pauli_terms,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)


def test_pauli_string_matrix_is_unitary_and_hermitian():
    p = pauli_string_matrix("XZYI")
    identity = np.eye(p.shape[0])
    np.testing.assert_allclose(p.conj().T @ p, identity, atol=1e-12)
    np.testing.assert_allclose(p, p.conj().T, atol=1e-12)


def test_analytic_pauli_mapping_matches_fermionic_hamiltonian():
    for t, U, mu in [
        (1.0, 0.0, 0.0),
        (1.0, 4.0, 0.0),
        (0.7, 3.5, 0.2),
    ]:
        h_fermion = two_site_hubbard_hamiltonian(t=t, U=U, mu=mu)
        terms = remove_zero_terms(
            two_site_hubbard_pauli_terms(t=t, U=U, mu=mu)
        )
        h_qubit = reconstruct_from_pauli_terms(terms)
        np.testing.assert_allclose(h_qubit, h_fermion, atol=1e-12)


def test_dense_decomposition_recovers_expected_terms():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0, mu=0.0)
    numeric = dense_pauli_decomposition(h)
    analytic = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )

    assert numeric.keys() == analytic.keys()
    for label in analytic:
        np.testing.assert_allclose(numeric[label], analytic[label], atol=1e-12)


def test_spectra_are_identical_after_mapping():
    h_fermion = two_site_hubbard_hamiltonian(t=1.0, U=4.0)
    h_qubit = reconstruct_from_pauli_terms(
        remove_zero_terms(two_site_hubbard_pauli_terms(t=1.0, U=4.0))
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(h_qubit),
        np.linalg.eigvalsh(h_fermion),
        atol=1e-12,
    )


def test_lcu_normalization_is_sum_of_absolute_coefficients():
    terms = {"IIII": 2.0, "XZXI": -0.5, "YZYI": -0.5}
    np.testing.assert_allclose(lcu_normalization(terms), 3.0, atol=1e-12)
