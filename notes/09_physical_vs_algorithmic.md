# Milestone 09 — Physical vs algorithmic thermalisation

## Central question

Across the same finite Fermi–Hubbard parameter grid, do states that are slower
to thermalise under the chosen open-system dynamics also require higher-degree
thermal polynomials in the block-encoding/QSVT-oriented route?

The project does **not** assume that the answer must be yes.

## Consistency correction before comparison

Milestones 02–04 work in the canonical half-filled sector:

~~~text
N = 2 electrons
Hilbert-space dimension = 6.
~~~

Milestones 05–08 use the full four-qubit / 16-dimensional Fock-space
Hamiltonian because that is the natural Jordan–Wigner block encoding.

The M8 polynomial was therefore originally validated against the full
Fock-space Gibbs normalization. That is a legitimate Gibbs target, but it is
not the same target used by the physical thermalisation analysis.

Milestone 09 fixes this comparison issue without changing the block encoding:

1. build the bounded polynomial on the full H/alpha;
2. evaluate the polynomial thermal amplitude on the full Hamiltonian;
3. form the positive amplitude-squared operator;
4. project that operator into the N=2 sector;
5. renormalize inside the sector;
6. compare with the exact canonical N=2 Gibbs state.

This gives both routes the **same thermal target**.

A future symmetry-aware quantum implementation could encode the fixed-N sector
more directly and potentially improve normalization/resource costs. That
optimization is outside the present small-system reference construction.

## Physical difficulty

For the fixed bath model:

~~~text
physical difficulty metrics:
    Liouvillian gap Delta_L
    inverse gap 1 / Delta_L
    trace-distance mixing time t_mix
~~~

The default operational definition is

~~~text
D(rho(t), rho_beta) <= 1e-2
~~~

persistently over the remaining sampled trajectory.

The initial state is the same site-0 doublon used in earlier milestones.

## Algorithmic difficulty

For the full four-qubit LCU block encoding:

~~~text
alpha = sum_j |alpha_j|
~~~

and the bounded thermal function is

~~~text
f(x) = exp[-beta alpha (x+1)/2].
~~~

At fixed uniform scalar tolerance

~~~text
epsilon_poly = 1e-4,
~~~

the main algorithmic metric is

~~~text
d = minimum Chebyshev degree.
~~~

Degree is a matrix-polynomial/query-complexity proxy. It is **not** presented
as a logical T-count, compiled circuit depth, or full fault-tolerant cost.

## Common parameter grid

~~~text
U/t    = 0, 2, 4, 6, 8
beta t = 0.25, 0.5, 1, 2
~~~

for 20 common cases.

A case is marked valid for the primary comparison only if:

- the reference Lindbladian has one numerical stationary mode;
- the physical trajectory reaches the mixing tolerance within the chosen
  horizon;
- the sector-conditioned polynomial Gibbs approximation has trace-distance
  error below 0.05.

The raw row is retained even if a case fails one of those checks.

## Statistical comparison

The main descriptive relationships are

~~~text
t_mix                 vs polynomial degree
1 / Delta_L           vs polynomial degree
beta * alpha          vs polynomial degree
~~~

Both Pearson and Spearman correlations are reported.

We also calculate correlations **within fixed beta** and **within fixed U**.
This matters because a high overall correlation could simply reflect the fact
that temperature changes both routes at once. Stratification helps distinguish
that shared trend from interaction-dependent variation.

## Interpretation limits

No correlation in this milestone is universal.

The physical side depends on:

- the chosen Davies-inspired bath couplings;
- bath rate scale;
- initial state for the operational mixing time;
- finite two-site system.

The algorithmic side depends on:

- the chosen Jordan–Wigner/LCU representation;
- LCU normalization alpha;
- polynomial approximation strategy;
- selected scalar tolerance;
- absence of explicit QSP phase and gate synthesis.

Therefore the result is a controlled finite-model comparison, not a theorem
relating open-system mixing to QSVT complexity.

## Why this is still useful

The two difficulty measures come from genuinely different mechanisms:

~~~text
physical:
bath-induced dissipative spectrum and state relaxation

algorithmic:
bounded approximation of a thermal matrix function of a block-encoded H
~~~

Putting them on the same U/t x beta t grid reveals whether their trends align,
diverge, or are largely explained by temperature/normalization.

That is the main research output of HubbardBath rather than a predetermined
claim.
