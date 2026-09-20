# HubbardBath — Results and Discussion

This document records the reproduced numerical results of the finite-system HubbardBath study. All headline results below were regenerated from clean GitHub Actions runs; the explicit QSP synthesis extension was reproduced on commit `a5f358b6df087dd520f885583244f2589cbf74ea`.

## Research question

How do interaction strength and temperature affect thermalisation in a finite Fermi–Hubbard system, and how does the difficulty of physical relaxation compare with the difficulty of algorithmically preparing the same half-filled Gibbs state?

The comparison deliberately uses two different notions of difficulty:

- **Physical:** Liouvillian spectral gap `Delta_L` and trace-distance mixing time `t_mix`.
- **Algorithmic:** the minimum bounded Chebyshev degree `d_G` required for the polynomial thermal route to reproduce the same `N=2` Gibbs state within trace distance `1e-2`.

The calculation is exact for a two-site, four-mode model. The full Fock space has dimension 16; physical thermalisation is studied in the six-dimensional half-filled `N=2` sector.

## 1. Physical thermalisation is tightly controlled by the Liouvillian gap

Across 30 `U/t x beta t` cases, the fitted finite-grid relation was

```text
t_mix = 3.751 * (1 / Delta_L)^1.0398
R^2   = 0.9897
```

and `Delta_L * t_mix` stayed between 3.3 and 4.6.

This is a strong finite-model result: for the chosen Davies-inspired bath and the doublon initial state, the slowest Liouvillian decay scale almost completely organizes the observed mixing time.

This should **not** be read as a thermodynamic-limit theorem. The gap, mixing time, and prefactor depend on the bath construction, rate scale, initial state, tolerance, and system size.

## 2. The fermionic-to-qubit pipeline passes independent numerical checks

For `U/t = 4`, the Jordan–Wigner Hamiltonian contains 11 nonzero Pauli terms.

The Hamiltonian-simulation benchmark at `tau t = 1.5` recovered the expected product-formula convergence:

```text
first-order fitted exponent   = -1.013
second-order fitted exponent  = -2.009
```

At 64 Trotter steps:

```text
first-order operator error   = 2.957e-2
second-order operator error  = 9.187e-4
second-order state infidelity = 2.972e-7
```

These measurements provide a numerical sanity check that the Pauli representation is behaving as expected before it is used in the LCU/block-encoding construction.

## 3. The dense LCU block encoding is numerically exact

For the same reference Hamiltonian,

```text
alpha = sum_j |alpha_j| = 10
||H / alpha||_2        = 0.8
```

The dense reference PREPARE/SELECT construction produced

```text
unitarity error             = 2.07e-15
top-left block error        = 3.47e-16
projected doublon error     = 5.89e-17
```

so the defining block-encoding identity

```text
(<0| tensor I) U_H (|0> tensor I) = H / alpha
```

is satisfied to floating-point precision.

This is a **correctness construction**, not an optimized fault-tolerant circuit.

## 4. Thermal polynomial degree is dominated by beta * alpha

The bounded thermal target is

```text
f(x) = exp[- beta alpha (x + 1) / 2],   x in [-1,1].
```

At the reference point `U/t = 4`, `beta t = 1`, `alpha = 10`, the minimum degree for uniform scalar error `1e-4` was 9.

However, the full research run exposed an important distinction: a small **uniform scalar-function error** does not always guarantee an accurate **normalized Gibbs state**, especially for colder/high-`U` cases. Normalization can amplify small amplitude errors when the thermal weight becomes strongly concentrated.

For that reason, the final comparison uses the state-targeted degree

```text
d_G = minimum degree such that
D(rho_poly, rho_beta) <= 1e-2
```

in the half-filled sector.

Across the 20 common cases, `d_G` ranged from 2 to 28.

The strongest relationship in the algorithmic data was

```text
beta * alpha vs d_G:
Pearson r  = 0.9982
Spearman rho = 0.9894
```

This is consistent with the structure of the bounded target itself: increasing inverse temperature or LCU normalization steepens the exponential function that the polynomial must approximate.

## 5. Physical and algorithmic difficulty move together, but they are not the same quantity

Using the same 20 `U/t x beta t` cases and the same `1e-2` trace-distance accuracy scale on both sides:

```text
t_mix vs d_G:
Pearson r    = 0.7084
Spearman rho = 0.8056

1 / Delta_L vs d_G:
Pearson r    = 0.7135
Spearman rho = 0.8132
```

So, within this controlled finite model, parameter points that are physically slower to thermalise also tend to require higher-degree thermal polynomials.

But the algorithmic degree is much more directly tied to `beta * alpha` than to the Liouvillian relaxation scale. That distinction matters: the physical route is governed by the dissipative spectrum of the chosen bath, while the polynomial route is governed by the shape and normalization of the encoded thermal matrix function.

The result is therefore **trend alignment, not an equivalence of complexity measures**.

## 6. The interaction trend survives fixed-temperature slices

The overall correlation is not only a trivial consequence of varying temperature. At fixed `beta t`, the Pearson correlation between `t_mix` and `d_G` was:

| beta t | Pearson r |
| ---: | ---: |
| 0.25 | 0.913 |
| 0.50 | 0.953 |
| 1.00 | 0.985 |
| 2.00 | 0.975 |

Each slice contains only five interaction values, so these numbers should be treated descriptively. Still, they show that increasing `U/t` makes both the selected physical bath dynamics and the thermal-polynomial task harder over this grid.

## 7. A useful methodological finding

The original comparison used degree at fixed uniform scalar error `1e-4`. That metric failed to guarantee a uniformly good Gibbs state at the coldest/highest-interaction points.

Examples:

```text
U/t=4, beta t=2: scalar-degree 13 -> Gibbs error 0.183
U/t=6, beta t=2: scalar-degree 15 -> Gibbs error 0.379
U/t=8, beta t=2: scalar-degree 17 -> Gibbs error 0.270
```

Requiring the actual half-filled Gibbs trace distance to be at most `1e-2` increased the corresponding degrees to 16, 22, and 28.

This refinement is part of the result: **the approximation metric matters** when a matrix function is later squared and normalized into a thermal state.

## 8. Explicit QSP phase synthesis closes the polynomial-to-phases gap

For the reference point `U/t = 4`, `beta t = 1`, the state-targeted thermal polynomial reaches half-filled Gibbs trace distance `0.00883` at degree 8.

Because the thermal polynomial has mixed parity, its even and odd Chebyshev components were synthesized separately using symmetric QSP:

```text
even component:
degree                 = 8
full QSP phase count   = 9
solver residual        = 1.17e-16
signal-response error  = 3.89e-16

odd component:
degree                 = 7
full QSP phase count   = 8
solver residual        = 1.33e-16
signal-response error  = 3.33e-16
```

HubbardBath independently reconstructs the `Wx`-convention QSP matrix product from the returned phase angles rather than relying only on the phase solver's internal convergence result.

The sum of the independently reconstructed even and odd QSP responses matches the original thermal polynomial with maximum error

```text
6.66e-16.
```

This supports the stronger statement that the project contains **explicit, numerically validated QSP phase synthesis** for the parity-resolved thermal transformation.

The remaining circuit-level step is to combine the two parity sequences coherently, for example through an ancilla/LCU construction, and then compile that construction into a full QSVT Gibbs-preparation circuit. That step is not claimed here.

## Interpretation limits

The study is intentionally small and explicit.

- It is a two-site finite system, not a thermodynamic-limit calculation.
- The physical mixing scale is specific to the chosen Davies-inspired coupling operators, rate scale `0.2`, doublon initial state, and trace-distance threshold.
- The LCU normalization depends on the chosen Pauli representation.
- `d_G` is a polynomial/query-complexity proxy, not a logical T-count or compiled circuit depth.
- The project synthesizes and independently validates explicit symmetric-QSP phases for the even and odd thermal-polynomial components. It does not claim a fully compiled coherent or fault-tolerant QSVT Gibbs-preparation circuit.
- Correlation coefficients summarize a deterministic parameter grid; their p-values should not be interpreted as population-level statistical evidence.

## Main conclusion

For this finite Fermi–Hubbard reference model, physical thermalisation and algorithmic Gibbs preparation become harder in broadly aligned regions of interaction-temperature space, but for different underlying reasons. The physical route is closely organized by the inverse Liouvillian gap, whereas the algorithmic polynomial degree is overwhelmingly organized by the thermal steepness parameter `beta * alpha`.

That separation is the central result of HubbardBath.
