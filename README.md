# HubbardBath

**Open-system thermalisation and quantum Gibbs-state preparation in finite Fermi–Hubbard systems.**

HubbardBath is a small-system research project at the boundary of quantum many-body physics, open quantum systems, and quantum algorithms. The long-term goal is to compare two routes to the same finite-temperature state of an interacting fermionic system:

1. **Physical thermalisation:** a Fermi–Hubbard system relaxes under a thermal Lindbladian.
2. **Algorithmic preparation:** the same Gibbs state is approached through qubit mappings, Hamiltonian simulation, block-encoding ideas, and polynomial matrix transformations.

The project is intentionally built from first principles before using higher-level quantum libraries. Every numerical step is paired with a validation test so that the final results remain interpretable and defensible.

## Research question

> How do interaction strength and temperature affect thermalisation in a finite Fermi–Hubbard system, and how does the difficulty of physical relaxation compare with the difficulty of algorithmically preparing the same Gibbs state?

## Current milestone — 06: Hamiltonian simulation

Milestones 01–05 established the two-site Fermi–Hubbard model, its finite-temperature and open-system physics, the Liouvillian-gap analysis, and an exact four-qubit Pauli representation. Milestone 06 uses that Pauli Hamiltonian for real-time simulation and benchmarks first- and second-order product formulas against exact evolution.

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

The fermionic operators are constructed explicitly with the Jordan–Wigner parity string. The model is validated through canonical anti-commutation relations, Hermiticity, particle-number conservation, and the six-dimensional half-filled sector. The thermal state is then constructed as `rho_beta = exp(-beta H) / Z` inside that fixed-particle-number sector.

### Run the checks

```bash
pip install -r requirements.txt
pytest -q
```

### Run the notebook

Start with:

```text
notebooks/01_fermi_hubbard_foundations.ipynb
notebooks/02_gibbs_states_and_thermal_observables.ipynb
```

Notebook 01 constructs and validates the fermionic model. Notebook 02 builds exact Gibbs states in the half-filled sector and studies thermal energy, von Neumann entropy, average double occupancy, and nearest-neighbour spin-z correlations across `U/t` and `beta t`.

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


### Milestone 02 reproducible sweep

```bash
python experiments/run_thermal_sweep.py
```

This generates a CSV over the `U/t × beta t` grid and four heatmaps in `figures/`. These equilibrium maps will be the reference state for the upcoming open-system thermalisation milestone.


### Milestone 03 open-system demonstration

```bash
python experiments/run_thermalisation_demo.py
```

The reference experiment starts from a doublon state at `U/t = 4` and `beta t = 1`, constructs a detailed-balance Lindbladian, verifies `||L(rho_beta)|| ≈ 0`, and tracks trace distance plus Hubbard observables as the system relaxes toward equilibrium. See `notes/03_open_system_thermalisation.md` for the modelling assumptions and validation targets.


### Milestone 04 spectral-gap sweep

```bash
python experiments/run_spectral_gap_sweep.py
```

The default exact grid contains 30 two-site cases across `U/t` and `beta t`. For each case it computes the Liouvillian gap, inverse gap, persistent trace-distance mixing time, stationary-mode count, and `Delta_L * t_mix`. It also performs a descriptive log-log fit of mixing time versus inverse gap. This fit is explicitly treated as a finite-grid diagnostic, not an asymptotic many-body scaling law.

See:

```text
notebooks/04_liouvillian_gap_vs_mixing.ipynb
notes/04_liouvillian_gap_and_mixing.md
```


### Milestone 05 Jordan–Wigner mapping

```bash
python experiments/run_jordan_wigner_mapping.py
```

The two-site, four-mode Hubbard Hamiltonian is written as

```text
H = sum_j alpha_j P_j
```

with explicit Jordan–Wigner parity strings. The analytic Pauli coefficients are independently checked using the Hilbert–Schmidt Pauli decomposition, the reconstructed qubit Hamiltonian is compared directly against the fermionic matrix, and the spectra are required to agree to numerical precision.

Milestone 05 also records

```text
alpha = sum_j |alpha_j|
```

which becomes the LCU normalization used later in the block-encoding milestone.

See:

```text
notebooks/05_jordan_wigner_to_qubits.ipynb
notes/05_jordan_wigner_mapping.md
```


### Milestone 06 Hamiltonian simulation

```bash
python experiments/run_hamiltonian_simulation.py
```

The benchmark compares exact `exp(-i H tau)` with first-order Lie–Trotter and second-order symmetric product formulas across increasing Trotter step counts. It records operator-norm error, doublon-state infidelity, fitted convergence exponents, and a simple Pauli-exponential resource proxy.

See:

```text
notebooks/06_hamiltonian_simulation.ipynb
notes/06_hamiltonian_simulation.md
```

The next milestone reuses the same Pauli coefficients to build an explicit LCU/block encoding of `H/alpha`.
