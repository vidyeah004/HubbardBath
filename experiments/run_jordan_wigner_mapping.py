"""Milestone 05: explicit Jordan-Wigner / Pauli decomposition.

Run from the repository root:
    python experiments/run_jordan_wigner_mapping.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.hubbard import two_site_hubbard_hamiltonian
from src.qubit_mapping import (
    dense_pauli_decomposition,
    lcu_normalization,
    reconstruct_from_pauli_terms,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    t = 1.0
    U = 4.0
    mu = 0.0

    h_fermion = two_site_hubbard_hamiltonian(t=t, U=U, mu=mu)
    analytic_terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=t, U=U, mu=mu)
    )
    numeric_terms = dense_pauli_decomposition(h_fermion)
    h_qubit = reconstruct_from_pauli_terms(analytic_terms)

    matrix_error = float(np.linalg.norm(h_qubit - h_fermion, ord=2))
    spectrum_error = float(np.max(np.abs(
        np.linalg.eigvalsh(h_qubit)
        - np.linalg.eigvalsh(h_fermion)
    )))
    alpha = lcu_normalization(analytic_terms)

    csv_path = results_dir / "milestone05_pauli_terms.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pauli_string", "coefficient"])
        for label, coefficient in sorted(analytic_terms.items()):
            writer.writerow([label, coefficient])

    summary = {
        "t": t,
        "U": U,
        "mu": mu,
        "number_of_pauli_terms": len(analytic_terms),
        "matrix_operator_norm_error": matrix_error,
        "maximum_spectrum_error": spectrum_error,
        "analytic_matches_numeric_term_set": (
            analytic_terms.keys() == numeric_terms.keys()
        ),
        "lcu_normalization_alpha": alpha,
    }

    summary_path = results_dir / "milestone05_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Pauli decomposition for U/t = 4")
    print("--------------------------------")
    for label, coefficient in sorted(analytic_terms.items()):
        print(f"{coefficient:+.6f}  {label}")

    print()
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
