import numpy as np

from src.hubbard import (
    annihilation_operator,
    anticommutator,
    commutator,
    project_to_particle_sector,
    sector_indices,
    total_number_operator,
    two_site_hubbard_hamiltonian,
)


def test_canonical_anticommutation_relations():
    num_modes = 4
    dim = 2**num_modes
    zero = np.zeros((dim, dim), dtype=complex)
    identity = np.eye(dim, dtype=complex)

    for i in range(num_modes):
        ci = annihilation_operator(num_modes, i)
        for j in range(num_modes):
            cj = annihilation_operator(num_modes, j)
            expected = identity if i == j else zero
            np.testing.assert_allclose(
                anticommutator(ci, cj.conj().T), expected, atol=1e-12
            )
            np.testing.assert_allclose(
                anticommutator(ci, cj), zero, atol=1e-12
            )


def test_annihilation_operator_is_nilpotent():
    for mode in range(4):
        c = annihilation_operator(4, mode)
        np.testing.assert_allclose(c @ c, 0.0, atol=1e-12)


def test_hubbard_hamiltonian_is_hermitian():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)


def test_particle_number_is_conserved():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0)
    n_total = total_number_operator(4)
    np.testing.assert_allclose(commutator(h, n_total), 0.0, atol=1e-12)


def test_half_filling_sector_has_dimension_six():
    h = two_site_hubbard_hamiltonian(t=1.0, U=4.0)
    h_half, indices = project_to_particle_sector(
        h, num_modes=4, particle_number=2
    )
    assert h_half.shape == (6, 6)
    assert len(indices) == 6
    assert indices == sector_indices(4, 2)
