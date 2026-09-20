# Milestone 10 — Explicit QSP phase synthesis

## Goal

Turn the bounded Chebyshev thermal polynomial from Milestone 08 into actual
QSP phase sequences and verify the resulting signal response.

The earlier project language deliberately said **QSVT-oriented** because it had
not yet synthesized phases. This milestone closes that gap for the
definite-parity polynomial components.

## Why two QSP sequences?

The bounded thermal polynomial is generally neither even nor odd:

```text
p(x) = p_even(x) + p_odd(x).
```

A standard single QSP sequence has a parity constraint tied to its degree.
Therefore the even and odd Chebyshev components are synthesized separately.

For Chebyshev coefficients

```text
p_even(x) = c0 T0(x) + c2 T2(x) + ...
p_odd(x)  = c1 T1(x) + c3 T3(x) + ...
```

the parity-reduced coefficient arrays are passed to the symmetric-QSP Newton
solver in `pyqsp`.

## Synthesis convention

The implementation uses the symmetric-QSP `Wx` convention. The scalar signal
unitary is

```text
W(x) = [[x, i sqrt(1-x^2)],
        [i sqrt(1-x^2), x]]
```

and each phase unitary is

```text
S(phi) = diag(exp(i phi), exp(-i phi)).
```

For phases `phi_0, ..., phi_d`,

```text
U_QSP(x)
= S(phi_0) W(x) S(phi_1) ... W(x) S(phi_d).
```

In the symmetric-QSP convention used here, the target definite-parity
polynomial appears in

```text
Im <0| U_QSP(x) |0>.
```

## Independent validation

The phase solver is external, but the final response is **not** accepted merely
because the solver reports convergence.

HubbardBath independently reconstructs the 2x2 QSP matrix product directly
from the returned full phases and checks, on a dense `[-1,1]` grid, that

```text
QSP_even(x) ~= p_even(x)
QSP_odd(x)  ~= p_odd(x)
QSP_even(x) + QSP_odd(x) ~= p(x).
```

The generated phase angles are written to

```text
results/milestone10_qsp_phases.csv
```

so the synthesis is reproducible and inspectable.

## What can now be claimed

After this milestone it is accurate to say:

> Explicit symmetric-QSP phase sequences were synthesized for the parity
> components of the bounded thermal polynomial and their signal responses were
> independently reconstructed and numerically validated.

It is **not** yet accurate to say:

> A complete fault-tolerant QSVT Gibbs-preparation circuit was compiled.

The two parity sequences still need a coherent ancilla/LCU combination at the
circuit level, and no logical T-count or error-corrected resource estimate is
claimed.

## Dependency

The phase synthesis uses `pyqsp==0.2.0`, whose symmetric-QSP method implements
the iterative phase-factor evaluation workflow for parity-restricted
Chebyshev polynomials.
