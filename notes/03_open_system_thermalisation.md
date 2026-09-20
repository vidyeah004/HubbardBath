# Milestone 03 — Open-system thermalisation

## Goal

Move beyond equilibrium thermodynamics and model how the half-filled two-site
Fermi–Hubbard system approaches its Gibbs state when coupled to a thermal
environment.

## Open-system model

We use a finite-dimensional, Davies-inspired Lindblad generator in the
six-dimensional two-electron sector.

The bath couples through local number operators and local spin-flip operators.
These channels preserve total particle number. In the Hamiltonian eigenbasis,
transition rates are chosen to obey

```text
gamma_up / gamma_down = exp(-beta * omega)
```

for a Bohr frequency `omega > 0`.

This guarantees thermal detailed balance for the transition network. The
construction should be viewed as a transparent reference bath for the finite
model, not as a complete microscopic derivation of a specific material
environment.

## Validation targets

Milestone 03 requires:

- Gibbs stationarity: `||L(rho_beta)|| ~ 0`
- trace preservation
- Hermiticity preservation
- positivity up to floating-point tolerance
- non-positive real parts of Liouvillian eigenvalues
- one stationary mode for the reference case
- contraction of trace distance toward the Gibbs state

## Demonstration

The default demonstration uses:

- `U/t = 4`
- `beta t = 1`
- an initial doublon state on site 0
- bath rate scale `0.2`

Run:

```bash
python experiments/run_thermalisation_demo.py
```

Milestone 04 will turn this single validated trajectory into a research sweep
over interaction strength and temperature and will analyze the Liouvillian
spectral gap versus thermal mixing time.
