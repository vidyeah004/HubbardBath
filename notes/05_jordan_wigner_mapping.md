# Milestone 05 — Fermions to qubits with Jordan–Wigner

## Goal

Make the qubit Hamiltonian explicit before attempting quantum simulation or
block encoding.

The two-site spinful Hubbard model uses four fermionic modes, which map to four
qubits in the ordering

```text
q0 = site 0 up
q1 = site 0 down
q2 = site 1 up
q3 = site 1 down
```

The Jordan–Wigner ladder operator is

```text
c_j = Z_0 Z_1 ... Z_(j-1) sigma^-_j.
```

## Number operators

Since

```text
n_j = c_j^dagger c_j = (I - Z_j)/2,
```

the on-site interaction

```text
U n_(i,up) n_(i,down)
```

becomes a sum of identity, one-qubit Z, and two-qubit ZZ terms.

## Hopping terms

A hopping term between fermionic modes p < q produces the parity string
between them. For this mode ordering,

```text
-t (c_0^dagger c_2 + h.c.)
    = -(t/2) (X Z X I + Y Z Y I)

-t (c_1^dagger c_3 + h.c.)
    = -(t/2) (I X Z X + I Y Z Y).
```

The middle Z is not decorative: it carries fermionic parity information.

## Full Pauli Hamiltonian

For general `t, U, mu`,

```text
H =
(U/2 - 2 mu) IIII
+ (-U/4 + mu/2) (ZIII + IZII + IIZI + IIIZ)
+ (U/4) (ZZII + IIZZ)
- (t/2) (XZXI + YZYI + IXZX + IYZY).
```

Terms with zero coefficients may be dropped.

## Validation

Milestone 05 checks the mapping three ways:

1. reconstruct the dense qubit Hamiltonian from the analytic Pauli sum;
2. compare it element-by-element with the original fermionic Hamiltonian;
3. independently recover the Pauli coefficients through the
   Hilbert–Schmidt formula

```text
alpha_P = Tr(P^dagger H) / 2^n.
```

The spectra must also agree to numerical precision.

## Why this matters for the next milestones

We now have

```text
H = sum_j alpha_j P_j.
```

That is exactly the representation needed for:

- product-formula / Trotter Hamiltonian simulation,
- LCU constructions,
- block encoding,
- and later polynomial matrix transformations.

We also compute

```text
alpha = sum_j |alpha_j|,
```

the standard LCU normalization that will reappear in the block-encoding
milestone.
