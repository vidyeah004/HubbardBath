# Milestone 08 — Bounded thermal polynomials and QSVT readiness

## Goal

Connect the block encoding from Milestone 07 to a thermal matrix function
without overstating what has been implemented.

The block encoding gives access to

```text
X = H / alpha
```

with spectrum inside `[-1, 1]`.

A naive target

```text
exp(-beta H / 2)
```

is not necessarily bounded by one because the Hubbard Hamiltonian can have
negative eigenvalues. Standard QSVT polynomial transformations require a
bounded target on the encoded interval.

## Bounded thermal amplitude

We therefore define

```text
f(x) = exp[- beta alpha (x + 1) / 2].
```

For every `x in [-1,1]`,

```text
0 < f(x) <= 1.
```

Substituting `X = H/alpha` gives

```text
f(H/alpha)
= exp[-beta (H + alpha I)/2]
= exp[-beta alpha/2] exp[-beta H/2].
```

The prefactor is a scalar. If

```text
K = f(H/alpha),
```

then

```text
K K^dagger
= exp[-beta alpha] exp[-beta H].
```

After dividing by the trace, the scalar cancels and the exact Gibbs state is
recovered.

This is why the shift by `alpha I` is algorithmically useful but does not
change the normalized thermal state.

## Chebyshev approximation

We approximate `f(x)` on the full interval `[-1,1]` using a Chebyshev
polynomial

```text
p_d(x) = sum_{k=0}^d c_k T_k(x).
```

For each `beta`, `alpha`, and target scalar error `epsilon`, the code
finds the smallest degree `d` for which

```text
max_{x in [-1,1]} |p_d(x) - f(x)| <= epsilon
```

on a dense validation grid.

The polynomial is also checked to remain bounded by one on that grid.

## Matrix verification

The polynomial is evaluated directly on the normalized Hamiltonian using the
Chebyshev matrix recurrence:

```text
T_0(X) = I
T_1(X) = X
T_k(X) = 2 X T_(k-1)(X) - T_(k-2)(X).
```

We compare

```text
p_d(H/alpha)
```

with the exact shifted thermal amplitude

```text
exp[-beta(H + alpha I)/2].
```

We then form

```text
rho_poly =
p_d(H/alpha) p_d(H/alpha)^dagger
/
Tr[p_d(H/alpha) p_d(H/alpha)^dagger]
```

and measure its trace distance from the exact Gibbs state.

## QSVT parity issue

A standard single QSP/QSVT polynomial sequence has a parity constraint linked
to its degree. The thermal function above is neither purely even nor purely
odd.

Using

```text
T_k(-x) = (-1)^k T_k(x),
```

we split the Chebyshev series into

```text
p_even(x) = sum over even k
p_odd(x)  = sum over odd k
```

so that

```text
p(x) = p_even(x) + p_odd(x).
```

These parity-compatible components are the correct objects to pass to QSP
phase synthesis and then combine with an ancilla/LCU construction.

## What is and is not implemented

Implemented and validated:

- bounded thermal target on the full block-encoding interval;
- Chebyshev approximation;
- minimum degree versus tolerance;
- even/odd parity decomposition;
- dense matrix polynomial evaluation;
- recovery of the normalized Gibbs state;
- degree sweeps versus `beta` and `alpha(U)`.

Not yet claimed:

- explicit QSP phase angles;
- a compiled fault-tolerant QSVT circuit;
- T-count or logical-depth estimates.

That distinction is intentional. The next step can add genuine phase synthesis
using a verified QSP synthesis routine and validate the resulting signal
polynomial against these reference coefficients.

## Research connection

The physical half of HubbardBath measured a thermal mixing time under a
Lindblad bath. The algorithmic half now has a concrete thermal-polynomial
degree.

This gives us two independently defined difficulty measures:

```text
physical:    t_mix, Delta_L
algorithmic: polynomial degree d(beta, U, epsilon)
```

The final comparison milestone will ask how these vary across the same
`U/t x beta t` parameter space without assuming in advance that they must
correlate.
