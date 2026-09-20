"""Milestone 07: build and verify the LCU block encoding.

Run from the repository root:
    python experiments/run_block_encoding.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.block_encoding import (
    block_encoding_error,
    block_encoding_unitary,
    prepare_target_state,
    projected_action,
    top_left_system_block,
    unitarity_error,
)
from src.qubit_mapping import (
    reconstruct_from_pauli_terms,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.simulation import computational_basis_state


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )

    unitary, data = block_encoding_unitary(terms)
    hamiltonian = reconstruct_from_pauli_terms(terms)
    encoded_block = top_left_system_block(
        unitary,
        data.system_dimension,
    )

    state = computational_basis_state("1100")
    projected, postselection_probability = projected_action(
        unitary,
        data,
        state,
    )
    expected = (hamiltonian / data.alpha) @ state

    prepare_state = prepare_target_state(data)

    coefficient_csv = results_dir / "milestone07_lcu_data.csv"
    with coefficient_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "ancilla_index",
            "pauli_string",
            "coefficient",
            "probability",
            "prepare_amplitude",
            "phase",
        ])
        for index, label in enumerate(data.labels):
            writer.writerow([
                index,
                label,
                data.coefficients[index],
                data.probabilities[index],
                prepare_state[index],
                data.phases[index],
            ])

    summary = {
        "number_of_pauli_terms": len(data.labels),
        "system_qubits": data.system_qubits,
        "ancilla_qubits": data.ancilla_qubits,
        "total_logical_qubits_in_dense_reference": (
            data.system_qubits + data.ancilla_qubits
        ),
        "system_dimension": data.system_dimension,
        "ancilla_dimension": data.ancilla_dimension,
        "full_block_encoding_dimension": unitary.shape[0],
        "lcu_normalization_alpha": data.alpha,
        "normalized_hamiltonian_operator_norm": float(
            np.linalg.norm(hamiltonian / data.alpha, ord=2)
        ),
        "prepare_unitarity_implicit_in_full_unitary": True,
        "block_encoding_unitarity_error": unitarity_error(unitary),
        "top_left_block_operator_error": block_encoding_error(terms),
        "direct_top_left_matrix_error": float(
            np.linalg.norm(
                encoded_block - hamiltonian / data.alpha,
                ord=2,
            )
        ),
        "doublon_postselection_probability": postselection_probability,
        "doublon_projected_action_error": float(
            np.linalg.norm(projected - expected)
        ),
    }

    summary_path = results_dir / "milestone07_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("LCU / block-encoding summary")
    print("----------------------------")
    print(json.dumps(summary, indent=2))
    print()
    print("PREPARE amplitudes:")
    for index, label in enumerate(data.labels):
        print(
            f"|{index:02d}>  {label}  "
            f"sqrt(p)={prepare_state[index].real:.6f}  "
            f"phase={data.phases[index]}"
        )

    print(f"\nWrote {coefficient_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
