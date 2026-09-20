# Milestone 04 — Liouvillian spectral gap and thermal mixing

## Research question

How do interaction strength `U/t` and inverse temperature `beta t` change
the relaxation spectrum of the finite Hubbard bath model, and how closely
does the inverse Liouvillian gap track an operational thermal mixing time?

## Quantities

For Liouvillian eigenvalues `lambda_j`, define the finite-system decay gap

```text
Delta_L = min_{Re(lambda_j) < 0} -Re(lambda_j).
```

The operational mixing time uses trace distance to the exact Gibbs state:

```text
D(rho(t), rho_beta) <= epsilon
```

and is defined as the first sampled time after which the trajectory remains
below `epsilon = 1e-2` for the rest of the simulation horizon.

This persistent-threshold definition avoids treating a transient crossing as
mixing.

## Default parameter grid

- `U/t = 0, 1, 2, 4, 6, 8`
- `beta t = 0.25, 0.5, 1, 2, 4`
- 30 exact two-site cases
- initial state: site-0 doublon
- bath rate scale: 0.2
- horizon: 12 inverse-gap times

## Main comparison

The script fits the descriptive finite-grid relation

```text
t_mix = A * (1 / Delta_L)^p
```

in log-log space and reports `p`, `A`, and `R^2`.

A value near `p = 1` would be consistent with inverse-gap control in this
specific finite model and bath construction. It is **not** by itself evidence
of an asymptotic many-body scaling law.

## Reproduce

```bash
python experiments/run_spectral_gap_sweep.py
```

Outputs:

- `results/milestone04_gap_mixing_sweep.csv`
- `results/milestone04_summary.json`
- gap heatmap
- mixing-time heatmap
- mixing-time versus inverse-gap figure

## Interpretation discipline

Milestone 04 deliberately separates:

1. exact observations on this finite grid,
2. descriptive fit statistics,
3. any later physical interpretation.

The two-site model is a controlled test bed, not a thermodynamic-limit
simulation.
