# HubbardBath

**Open-system thermalisation and quantum Gibbs-state preparation in a finite Fermi–Hubbard model.**

HubbardBath is a first-principles research project connecting **open quantum systems**, **many-body physics**, and **quantum algorithms** on one controlled thermal target.

The project asks:

> How do interaction strength and temperature affect physical thermalisation in a finite Fermi–Hubbard system, and how does that difficulty compare with the polynomial complexity of preparing the same Gibbs state algorithmically?

The full study is reproducible in GitHub Actions and is built around a two-site spinful Hubbard model with four fermionic modes.

## Headline result

Across the finite parameter grid, physical and algorithmic thermal difficulty move together, but they are controlled by different structures.

For the open-system route, the mixing time is tightly organized by the Liouvillian gap:

```text
t_mix ≈ 3.751 * (1 / Delta_L)^1.0398
R^2   = 0.9897
```

For the algorithmic route, define `d_G` as the **minimum bounded Chebyshev degree** whose half-filled Gibbs state reaches trace distance `1e-2`.

Across 20 common `U/t x beta t` cases:

```text
t_mix vs d_G
Pearson r    = 0.708
Spearman rho = 0.806

1 / Delta_L vs d_G
Pearson r    = 0.714
Spearman rho = 0.813

beta * alpha vs d_G
Pearson r    = 0.998
Spearman rho = 0.989
```

So the two notions of difficulty show clear trend alignment in this finite model, while the algorithmic degree is much more directly controlled by the thermal steepness parameter `beta * alpha`.

See **[RESULTS.md](RESULTS.md)** for the full interpretation and limitations.

## Physics model

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

The full Fock space has dimension 16. Physical thermalisation is studied in the six-dimensional half-filled `N=2` sector.

## Research pipeline

```text
Fermi-Hubbard Hamiltonian
        |
        v
canonical Gibbs states
        |
        v
detailed-balance Lindblad dynamics
        |
        v
Liouvillian gap + mixing time
        |
        +----------------------------+
        |                            |
        v                            v
Jordan-Wigner mapping          physical difficulty
        |
        v
Pauli Hamiltonian
        |
        v
Hamiltonian simulation
        |
        v
LCU block encoding H/alpha
        |
        v
bounded thermal polynomial
        |
        v
sector-conditioned Gibbs state
        |
        v
algorithmic difficulty d_G
        |
        +------------ compare ------------+
```

## Key validations

### Fermion-to-qubit mapping

The analytic Jordan–Wigner Pauli decomposition is checked independently using Hilbert–Schmidt Pauli coefficients. The dense fermionic and qubit Hamiltonians agree numerically, including their spectra.

### Hamiltonian simulation

For `U/t=4` and `tau t=1.5`, the fitted product-formula convergence exponents are

```text
first order  = -1.013
second order = -2.009
```

consistent with the expected `O(1/r)` and `O(1/r^2)` behavior.

### LCU block encoding

At `U/t=4`, the Hamiltonian has 11 nonzero Pauli terms and

```text
alpha = sum_j |alpha_j| = 10.
```

The dense PREPARE/SELECT construction verifies

```text
(<0| tensor I) U_H (|0> tensor I) = H / alpha
```

with top-left-block operator error approximately `3.5e-16`.

### Thermal polynomial

The block encoding supplies `X = H/alpha` with spectrum in `[-1,1]`. The bounded target is

```text
f(x) = exp[-beta alpha (x + 1)/2].
```

Because

```text
f(H/alpha)
= exp[-beta alpha/2] exp[-beta H/2],
```

the scalar shift cancels after the amplitude operator is squared and normalized into a Gibbs state.

The Chebyshev series is split into even and odd parity components for QSVT-oriented synthesis. **Explicit QSP phase synthesis is not claimed.**

## A methodological result

A fixed uniform scalar approximation error was not sufficient at every cold/high-`U` point.

For example:

```text
U/t=6, beta t=2
degree at scalar error 1e-4: 15
half-filled Gibbs error:     0.379
```

When the algorithmic criterion is changed to the quantity that actually matters,

```text
D(rho_poly, rho_beta) <= 1e-2,
```

the required degree becomes 22.

This is why the final comparison uses the state-targeted degree `d_G`, not just a uniform scalar-function tolerance.

## Reproduce

Install dependencies and run the validation suite:

```bash
pip install -r requirements.txt
python -m pytest -q
```

Run the complete research pipeline:

```bash
python experiments/run_thermal_sweep.py
python experiments/run_thermalisation_demo.py
python experiments/run_spectral_gap_sweep.py
python experiments/run_jordan_wigner_mapping.py
python experiments/run_hamiltonian_simulation.py
python experiments/run_block_encoding.py
python experiments/run_thermal_polynomial_qsvt.py
python experiments/run_physical_algorithmic_comparison.py
```

GitHub Actions also provides a `research-results` workflow that executes the pipeline from a clean environment and uploads generated CSV, JSON, and PNG outputs.

## Repository structure

```text
HubbardBath/
├── experiments/    # reproducible experiment entry points
├── notebooks/      # milestone derivations and exploratory analyses
├── notes/          # modelling assumptions and technical derivations
├── src/            # reusable numerical implementation
├── tests/          # algebraic and numerical validation
├── figures/        # generated research figures
├── results/        # generated/snapshotted results
└── paper/          # report workspace
```

## Scope and limitations

This is a **finite two-site reference study**.

The physical relaxation results depend on the chosen Davies-inspired bath couplings, rate scale, initial state, and mixing threshold. The polynomial degree depends on the Jordan–Wigner/LCU representation, normalization `alpha`, target state error, and approximation strategy.

The dense block encoding is a correctness construction, not an optimized fault-tolerant implementation. Polynomial degree is a query/approximation proxy, not a logical T-count or circuit-depth estimate. Explicit QSP phase synthesis remains future work.

The goal is therefore not to claim a universal relation between Lindbladian mixing and QSVT complexity, but to make the two routes concrete, testable, and directly comparable in one controlled model.
