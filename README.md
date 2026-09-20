# HubbardBath

**Open-system thermalisation and quantum Gibbs-state preparation in finite Fermi–Hubbard systems.**

HubbardBath is a small-system research project at the boundary of quantum many-body physics, open quantum systems, and quantum algorithms. The long-term goal is to compare two routes to the same finite-temperature state of an interacting fermionic system:

1. **Physical thermalisation:** a Fermi–Hubbard system relaxes under a thermal Lindbladian.
2. **Algorithmic preparation:** the same Gibbs state is approached through qubit mappings, Hamiltonian simulation, block-encoding ideas, and polynomial matrix transformations.

The project is intentionally built from first principles before using higher-level quantum libraries. Every numerical step is paired with a validation test so that the final results remain interpretable and defensible.

## Research question

> How do interaction strength and temperature affect thermalisation in a finite Fermi–Hubbard system, and how does the difficulty of physical relaxation compare with the difficulty of algorithmically preparing the same Gibbs state?

## Current milestone — 01: Fermi–Hubbard foundations

The first milestone implements a two-site, spinful Fermi–Hubbard model in the full 16-dimensional Fock space.

Mode ordering:

```text
0 -> site 0, spin up
1 -> site 0, spin down
2 -> site 1, spin up
3 -> site 1, spin down
```

The Hamiltonian is

```text
H = -t sum_sigma (c^†_{0,sigma} c_{1,sigma} + h.c.)
    + U sum_i n_{i,up} n_{i,down}
    - mu sum_{i,sigma} n_{i,sigma}.
```

The fermionic operators are constructed explicitly with the Jordan–Wigner parity string. Milestone 01 verifies the canonical anti-commutation relations, Hermiticity, particle-number conservation, and the six-dimensional half-filled sector.

### Run the checks

```bash
pip install -r requirements.txt
pytest -q
```

### Run the notebook

Open:

```text
notebooks/01_fermi_hubbard_foundations.ipynb
```

It constructs the operators, verifies the algebra, diagonalises the two-site model, restricts to half filling, and plots the finite-system spectrum as the interaction ratio `U/t` is varied.

## Planned research path

```text
Fermi-Hubbard foundations
        ↓
thermal observables and Gibbs states
        ↓
thermal Lindblad dynamics
        ↓
Liouvillian spectrum and mixing time
        ↓
Jordan-Wigner / Pauli representation
        ↓
Hamiltonian simulation
        ↓
block encoding
        ↓
polynomial / QSVT-oriented Gibbs preparation
        ↓
physical vs algorithmic thermalisation
```

The project will remain explicit about finite-size limitations. Small-system numerical evidence will not be presented as an asymptotic many-body result.

## Repository structure

```text
HubbardBath/
├── notebooks/      # derivations and reproducible experiments
├── src/            # reusable numerical code
├── tests/          # algebraic and numerical validation
├── figures/        # publication-quality figures
├── results/        # experiment outputs
└── paper/          # technical report as the project matures
```

## Compute strategy

Early milestones use exact NumPy/SciPy calculations because the two-site system is tiny and exact methods make validation easy. Larger open-system simulations will move to symmetry reduction, sparse operators, Krylov evolution, and GPU acceleration only when the system size makes those methods necessary.
