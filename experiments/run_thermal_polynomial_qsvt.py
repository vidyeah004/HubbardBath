"""Milestone 08: bounded thermal polynomials and QSVT readiness.

Run from the repository root:
    python experiments/run_thermal_polynomial_qsvt.py

This experiment measures how the Chebyshev degree required for a bounded
thermal amplitude depends on beta and on the LCU normalization alpha(U).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hubbard import two_site_hubbard_hamiltonian
from src.qubit_mapping import (
    lcu_normalization,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.thermal_polynomial import (
    amplitude_operator_error,
    gibbs_trace_distance_error,
    minimum_degree_for_error,
    qsvt_readiness_report,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    U_values = [0.0, 2.0, 4.0, 6.0, 8.0]
    beta_values = [0.25, 0.5, 1.0, 2.0]
    epsilon = 1e-4

    rows = []

    for U in U_values:
        h = two_site_hubbard_hamiltonian(
            t=1.0,
            U=U,
            mu=0.0,
        )
        terms = remove_zero_terms(
            two_site_hubbard_pauli_terms(
                t=1.0,
                U=U,
                mu=0.0,
            )
        )
        alpha = lcu_normalization(terms)

        for beta in beta_values:
            approximation = minimum_degree_for_error(
                beta=beta,
                alpha=alpha,
                epsilon=epsilon,
                max_degree=256,
                grid_size=4001,
            )
            readiness = qsvt_readiness_report(approximation)

            rows.append({
                "U_over_t": U,
                "beta_t": beta,
                "alpha": alpha,
                "beta_alpha": beta * alpha,
                "target_scalar_epsilon": epsilon,
                "degree": approximation.degree,
                "scalar_max_error":
                    approximation.max_scalar_error,
                "polynomial_max_abs":
                    approximation.max_absolute_value,
                "amplitude_operator_error":
                    amplitude_operator_error(h, approximation),
                "gibbs_trace_distance":
                    gibbs_trace_distance_error(h, approximation),
                "even_component_max_abs":
                    readiness["even_component_max_abs"],
                "odd_component_max_abs":
                    readiness["odd_component_max_abs"],
                "parity_reconstruction_error":
                    readiness["parity_reconstruction_error"],
            })

    csv_path = results_dir / "milestone08_thermal_polynomial_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    reference_terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(
            t=1.0,
            U=4.0,
            mu=0.0,
        )
    )
    reference_alpha = lcu_normalization(reference_terms)
    tolerance_rows = []

    for beta in [0.25, 0.5, 1.0, 2.0]:
        for target_epsilon in [1e-2, 1e-4, 1e-6]:
            approximation = minimum_degree_for_error(
                beta=beta,
                alpha=reference_alpha,
                epsilon=target_epsilon,
                max_degree=256,
                grid_size=4001,
            )
            tolerance_rows.append({
                "U_over_t": 4.0,
                "beta_t": beta,
                "alpha": reference_alpha,
                "epsilon": target_epsilon,
                "degree": approximation.degree,
                "actual_scalar_error":
                    approximation.max_scalar_error,
            })

    tolerance_csv = (
        results_dir
        / "milestone08_reference_tolerance_sweep.csv"
    )
    with tolerance_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tolerance_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(tolerance_rows)

    beta_alpha = np.array([
        row["beta_alpha"] for row in rows
    ])
    degrees = np.array([
        row["degree"] for row in rows
    ])

    plt.figure(figsize=(7.0, 4.8))
    for U in U_values:
        mask = np.array([
            row["U_over_t"] == U for row in rows
        ])
        plt.plot(
            beta_alpha[mask],
            degrees[mask],
            marker="o",
            label=fr"$U/t={U}$",
        )
    plt.xlabel(r"Thermal difficulty parameter $\beta\alpha$")
    plt.ylabel("Minimum Chebyshev degree")
    plt.title(
        r"Bounded thermal polynomial degree at $\epsilon=10^{-4}$"
    )
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone08_degree_vs_beta_alpha.png",
        dpi=180,
    )
    plt.close()

    degree_grid = np.zeros(
        (len(beta_values), len(U_values)),
        dtype=float,
    )
    gibbs_error_grid = np.zeros_like(degree_grid)

    for row in rows:
        i = beta_values.index(row["beta_t"])
        j = U_values.index(row["U_over_t"])
        degree_grid[i, j] = row["degree"]
        gibbs_error_grid[i, j] = row["gibbs_trace_distance"]

    for data, title, filename in [
        (
            degree_grid,
            r"Minimum thermal-polynomial degree",
            "milestone08_degree_heatmap.png",
        ),
        (
            gibbs_error_grid,
            r"Gibbs-state trace-distance error",
            "milestone08_gibbs_error_heatmap.png",
        ),
    ]:
        plt.figure(figsize=(7.0, 4.8))
        image = plt.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[
                min(U_values),
                max(U_values),
                min(beta_values),
                max(beta_values),
            ],
        )
        plt.xlabel(r"Interaction strength $U/t$")
        plt.ylabel(r"Inverse temperature $\beta t$")
        plt.title(title)
        plt.colorbar(image)
        plt.tight_layout()
        plt.savefig(
            figures_dir / filename,
            dpi=180,
        )
        plt.close()

    reference = minimum_degree_for_error(
        beta=1.0,
        alpha=reference_alpha,
        epsilon=1e-4,
        max_degree=256,
    )
    readiness = qsvt_readiness_report(reference)

    summary = {
        "main_sweep_cases": len(rows),
        "main_scalar_epsilon": epsilon,
        "reference_U_over_t": 4.0,
        "reference_beta_t": 1.0,
        "reference_alpha": reference_alpha,
        "reference_degree": reference.degree,
        "reference_scalar_error":
            reference.max_scalar_error,
        "reference_qsvt_readiness": readiness,
        "phase_synthesis_completed": False,
        "phase_synthesis_note": (
            "Milestone 08 validates a bounded parity-decomposed polynomial "
            "transformation. Explicit QSP phase synthesis is deliberately "
            "not claimed by this experiment."
        ),
    }

    summary_path = results_dir / "milestone08_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {tolerance_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
