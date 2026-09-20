# Milestone 07 — LCU block encoding

## Goal

Construct an explicit standard-form block encoding of the same four-qubit
Fermi–Hubbard Hamiltonian used throughout HubbardBath.

Milestone 05 gave

```text
H = sum_j alpha_j P_j
```

with Pauli strings `P_j`. Define

```text
alpha = sum_j |alpha_j|.
```

For the reference point `U/t = 4, mu = 0`, there are 11 nonzero Pauli
terms and `alpha = 10`.

## PREPARE

Introduce an ancilla label register and define

```text
PREPARE |0>
    = sum_j sqrt(|alpha_j| / alpha) |j>.
```

Eleven terms require four ancilla qubits because four qubits provide 16
computational-basis labels. The five unused labels simply have zero PREPARE
amplitude.

The dense reference code constructs PREPARE using a Householder reflection.
This is a mathematical reference implementation, not an optimized quantum
circuit synthesis of the state-preparation oracle.

## SELECT

Define

```text
SELECT
    = sum_j |j><j| tensor phase(alpha_j) P_j.
```

The coefficient magnitude lives in PREPARE; the sign/complex phase lives in
SELECT. Unused ancilla labels apply identity to the system register.

## Block encoding

The LCU unitary is

```text
U_H =
(PREPARE^dagger tensor I)
SELECT
(PREPARE tensor I).
```

Its all-zero-ancilla block is

```text
(<0| tensor I) U_H (|0> tensor I)
    = H / alpha.
```

Milestone 07 constructs the full dense reference unitary and verifies this
identity in operator norm.

## Why alpha matters

The encoded matrix is `H/alpha`, not `H`. This normalization ensures the
encoded block has operator norm at most one, which is essential for the
polynomial-transformation viewpoint used by QSVT.

The value of `alpha` therefore enters algorithmic resource estimates; a
larger LCU normalization compresses the Hamiltonian spectrum more strongly
inside the unit interval.

## Postselection interpretation

Starting with

```text
|0>_ancilla |psi>_system
```

and applying `U_H`, projection of the ancilla back onto `|0>` gives the
unnormalized system state

```text
(H / alpha) |psi>.
```

The probability of observing the zero ancilla is

```text
|| H |psi> ||^2 / alpha^2.
```

The experiment verifies this explicitly for the doublon state `|1100>`.

## Important limitation

This milestone builds a **dense reference block encoding**, not an
fault-tolerant gate-optimized PREPARE/SELECT circuit. That distinction will
remain explicit. Its purpose is to validate the mathematics and provide the
correct encoded matrix for the QSVT milestone.

## Next milestone

Milestone 08 will use the normalized spectral variable supplied by the block
encoding and study polynomial approximation of a thermal matrix function such
as

```text
exp(-beta H / 2).
```

That is the bridge from block encoding to QSVT-oriented Gibbs-state
preparation.
