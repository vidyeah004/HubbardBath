# Milestone 06 — Hamiltonian simulation

## Goal

Use the explicit four-qubit Pauli Hamiltonian from Milestone 05 to simulate

```text
U(tau) = exp(-i H tau)
```

and quantify the approximation error of product formulas.

## Exact reference

Because the current four-qubit system is tiny, the reference propagator is
computed by dense matrix exponentiation. This is used only as a validation
oracle; it is not the scalable quantum algorithm.

## Pauli exponentials

For every Pauli string `P`,

```text
P^2 = I
```

so

```text
exp(-i theta P) = cos(theta) I - i sin(theta) P.
```

Each individual Pauli-term exponential is therefore represented exactly in
the numerical model.

## Product formulas

First order:

```text
exp[-i (H1 + ... + Hm) tau]
approximately
[ exp(-i H1 dt) ... exp(-i Hm dt) ]^r
```

with `dt = tau/r`.

Second-order symmetric / Strang:

```text
[ exp(-i H1 dt/2) ... exp(-i Hm dt/2)
  exp(-i Hm dt/2) ... exp(-i H1 dt/2) ]^r.
```

For fixed total evolution time, asymptotic operator error is expected to scale
approximately as

```text
first order:  O(1/r)
second order: O(1/r^2)
```

for this finite decomposition. Milestone 06 measures those exponents rather
than simply assuming them.

## Default benchmark

- `U/t = 4`
- total simulation time `tau t = 1.5`
- Trotter steps `r = 1, 2, 4, 8, 16, 32, 64`
- initial state `|1100>`, the site-0 doublon

Metrics:

- operator norm error `||U_PF - U_exact||_2`
- pure-state infidelity for the doublon initial state
- fitted convergence exponent on the fine-step tail
- naive number of Pauli exponentials

## Resource-count caveat

The Pauli-exponential count is only an algorithm-level proxy. It is **not**
yet a hardware gate count. Circuit compilation of each Pauli exponential,
connectivity costs, cancellations, and fault-tolerant synthesis are later
questions.

## Why this milestone matters

Milestone 05 provided

```text
H = sum_j alpha_j P_j.
```

Milestone 06 shows how that decomposition can generate real-time evolution.
Milestone 07 will use the same coefficients in a different way: an LCU
construction whose top-left block represents `H/alpha`.
