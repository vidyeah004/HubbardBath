# Davies-generator methodology hardening

## Why this change was necessary

The original Milestone 03 reference bath constructed separate rank-one jump
operators from individual eigenvectors returned by numerical diagonalization.

That is acceptable only when the relevant eigenspaces are non-degenerate.

The two-site Hubbard Hamiltonian has exact degeneracies. Inside a degenerate
energy eigenspace, individual eigenvectors are not unique: any unitary rotation
within that subspace is an equally valid eigenbasis. A dissipator defined in
terms of individual vectors can therefore inherit an arbitrary numerical basis
choice.

The corrected construction uses spectral projectors instead.

## Basis-invariant Davies components

Let

```text
H = sum_E E Pi_E
```

where `Pi_E` projects onto the entire eigenspace with energy `E`.

For each Hermitian system-bath coupling `A_a`, define

```text
A_a(omega)
    = sum_{E_high-E_low=omega}
      Pi_low A_a Pi_high
```

for positive Bohr frequency `omega`, and

```text
A_a(0) = sum_E Pi_E A_a Pi_E.
```

The zero-frequency term preserves the complete action inside degenerate
subspaces.

All transitions with the same Bohr frequency are aggregated into one operator
per physical coupling before entering the Lindblad dissipator.

This gives

```text
[H, A_a(omega)] = -omega A_a(omega).
```

For Hermitian couplings,

```text
A_a = A_a(0)
      + sum_{omega>0}
        [A_a(omega) + A_a(omega)^dagger].
```

## Thermal rates

HubbardBath continues to use a deliberately simple flat reference bath
spectrum.

For `omega > 0`:

```text
gamma_down(omega) = kappa
gamma_up(omega)   = kappa exp(-beta omega)
```

so

```text
gamma_up / gamma_down = exp(-beta omega).
```

Zero-frequency components use rate `kappa`.

This satisfies the thermal KMS/detailed-balance ratio required for the Gibbs
state to be stationary.

The bath remains a controlled reference model rather than a microscopic
material-environment derivation.

## What changed numerically

The old construction effectively created one rank-one jump per pair of
individual eigenvectors and combined coupling strengths at the rate level.

The corrected construction:

1. groups degenerate eigenvalues into spectral projectors;
2. keeps interference terms inside each physical coupling;
3. groups all projector transitions sharing one Bohr frequency;
4. applies KMS-related upward and downward rates to those aggregate operators.

Therefore absolute gaps and mixing times can change after this correction.

That is expected and is why all M4 and M9 headline results must be regenerated
before the corrected bath is considered final.

## Validation

The test suite checks:

- projector Hermiticity, idempotence, orthogonality, and completeness;
- existence of nontrivial degenerate eigenspaces;
- the commutator relation for every Bohr component;
- exact reconstruction of each Hermitian bath coupling from its components;
- Gibbs-state stationarity over several `U/t` and `beta t` values;
- uniqueness of the stationary Liouvillian mode on a small parameter grid;
- positivity, Hermiticity, trace preservation, and relaxation tests from the
  earlier open-system milestone.

## Interpretation

This correction strengthens the *methodology* rather than changing the research
question.

The physical-vs-algorithmic comparison remains finite-system and
bath-dependent. Any updated correlations after the rerun should be interpreted
as properties of this projector-based Davies reference bath, not as universal
relations between open-system and quantum-algorithmic complexity.
