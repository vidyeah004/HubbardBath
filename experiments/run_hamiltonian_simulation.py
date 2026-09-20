"""Milestone 06: exact vs product-formula Hamiltonian simulation.

Run from the repository root:
    python experiments/run_hamiltonian_simulation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.qubit_mapping import (
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.simulation import (
    computational_basis_state,
    convergence_exponent,
    naive_pauli_exponential_count,
    simulation_metrics,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    total_time = 1.5
    steps_values = np.array([1, 2, 4, 8, 16, 32, 64])
    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=4.0, mu=0.0)
    )
    initial_state = computational_basis_state("1100")

    rows = []
    for steps in steps_values:
        metrics = simulation_metrics(
            terms,
            time=total_time,
            steps=int(steps),
            initial_state=initial_state,
        )
        rows.append({
            "time": total_time,
            "steps": int(steps),
            "first_order_pauli_exponentials":
                naive_pauli_exponential_count(len(terms), int(steps), 1),
            "second_order_pauli_exponentials":
                naive_pauli_exponential_count(len(terms), int(steps), 2),
            **metrics,
        })

    first_errors = np.array([
        row["first_order_operator_error"] for row in rows
    ])
    second_errors = np.array([
        row["second_order_operator_error"] for row in rows
    ])

    first_exponent = convergence_exponent(
        steps_values,
        first_errors,
        tail_points=4,
    )
    second_exponent = convergence_exponent(
        steps_values,
        second_errors,
        tail_points=4,
    )

    csv_path = results_dir / "milestone06_trotter_convergence.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "U_over_t": 4.0,
        "simulation_time": total_time,
        "number_of_nonzero_pauli_terms": len(terms),
        "initial_state": "1100 (site-0 doublon)",
        "first_order_tail_exponent": first_exponent,
        "second_order_tail_exponent": second_exponent,
        "finest_first_order_operator_error": float(first_errors[-1]),
        "finest_second_order_operator_error": float(second_errors[-1]),
        "finest_first_order_state_infidelity": float(
            rows[-1]["first_order_state_infidelity"]
        ),
        "finest_second_order_state_infidelity": float(
            rows[-1]["second_order_state_infidelity"]
        ),
    }

    summary_path = results_dir / "milestone06_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(7.0, 4.8))
    plt.loglog(
        steps_values,
        first_errors,
        marker="o",
        label=fr"first order, slope={first_exponent:.2f}",
    )
    plt.loglog(
        steps_values,
        second_errors,
        marker="o",
        label=fr"second order, slope={second_exponent:.2f}",
    )
    plt.xlabel("Trotter steps")
    plt.ylabel(r"Operator error $||U_{PF}-U||_2$")
    plt.title(r"Hubbard Hamiltonian simulation at $U/t=4$, $\tau t=1.5$")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone06_trotter_operator_error.png",
        dpi=180,
    )
    plt.close()

    first_infidelity = np.array([
        row["first_order_state_infidelity"] for row in rows
    ])
    second_infidelity = np.array([
        row["second_order_state_infidelity"] for row in rows
    ])

    plt.figure(figsize=(7.0, 4.8))
    plt.loglog(
        steps_values,
        first_infidelity,
        marker="o",
        label="first order",
    )
    plt.loglog(
        steps_values,
        second_infidelity,
        marker="o",
        label="second order",
    )
    plt.xlabel("Trotter steps")
    plt.ylabel("Doublon-state infidelity")
    plt.title("State-specific simulation error")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone06_state_infidelity.png",
        dpi=180,
    )
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
