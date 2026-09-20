"""Milestone 09: compare physical and algorithmic thermal difficulty.

Run from the repository root:
    python experiments/run_physical_algorithmic_comparison.py

The experiment deliberately uses the same half-filled two-electron target on
both sides. The physical route uses the detailed-balance Lindbladian from
Milestones 03-04. The algorithmic route uses the full four-qubit block
encoding from Milestones 05-08 but conditions its thermal state onto N=2
before measuring Gibbs-state error.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.comparison import comparison_case, comparison_summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    U_values = [0.0, 2.0, 4.0, 6.0, 8.0]
    beta_values = [0.25, 0.5, 1.0, 2.0]

    physical_epsilon = 1e-2
    algorithmic_epsilon = 1e-4
    rate_scale = 0.2

    rows = []
    for U in U_values:
        for beta in beta_values:
            row = comparison_case(
                U=U,
                beta=beta,
                physical_epsilon=physical_epsilon,
                algorithmic_epsilon=algorithmic_epsilon,
                rate_scale=rate_scale,
                horizon_in_inverse_gaps=12.0,
                num_time_points=241,
                max_polynomial_degree=256,
            )
            rows.append(row)
            print(
                f"U/t={U:>4.1f} beta*t={beta:>4.2f} "
                f"tmix={row['mixing_time']:.5g} "
                f"1/gap={row['inverse_liouvillian_gap']:.5g} "
                f"degree={row['thermal_polynomial_degree']:>3d} "
                f"valid={row['comparison_valid']}"
            )

    csv_path = results_dir / "milestone09_physical_vs_algorithmic.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = comparison_summary(rows)
    summary.update({
        "physical_target": (
            "half-filled N=2 Gibbs state under fixed detailed-balance bath"
        ),
        "algorithmic_target": (
            "same half-filled N=2 Gibbs state, conditioned after applying "
            "the full four-qubit thermal polynomial"
        ),
        "physical_mixing_tolerance": physical_epsilon,
        "algorithmic_scalar_tolerance": algorithmic_epsilon,
        "bath_rate_scale": rate_scale,
        "interpretation": (
            "All correlations are descriptive finite-system observations. "
            "The absolute physical time scale depends on the fixed bath rate, "
            "and polynomial degree is not a compiled logical-gate count."
        ),
    })

    summary_path = results_dir / "milestone09_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    valid = [row for row in rows if row["comparison_valid"]]

    plt.figure(figsize=(7.0, 4.8))
    for beta in beta_values:
        group = [row for row in valid if row["beta_t"] == beta]
        if not group:
            continue
        plt.scatter(
            [row["thermal_polynomial_degree"] for row in group],
            [row["mixing_time"] for row in group],
            label=fr"$\beta t={beta}$",
        )
    plt.xlabel("Minimum bounded thermal-polynomial degree")
    plt.ylabel(r"Physical mixing time $t_{\mathrm{mix}}$")
    plt.title("Physical vs algorithmic thermalisation difficulty")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone09_mixing_vs_degree.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    for beta in beta_values:
        group = [row for row in valid if row["beta_t"] == beta]
        if not group:
            continue
        plt.scatter(
            [row["thermal_polynomial_degree"] for row in group],
            [row["inverse_liouvillian_gap"] for row in group],
            label=fr"$\beta t={beta}$",
        )
    plt.xlabel("Minimum bounded thermal-polynomial degree")
    plt.ylabel(r"Inverse Liouvillian gap $1/\Delta_{\mathcal L}$")
    plt.title("Spectral relaxation scale vs polynomial degree")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone09_inverse_gap_vs_degree.png",
        dpi=180,
    )
    plt.close()

    degree_grid = np.full(
        (len(beta_values), len(U_values)),
        np.nan,
    )
    mixing_grid = np.full_like(degree_grid, np.nan)

    for row in rows:
        i = beta_values.index(row["beta_t"])
        j = U_values.index(row["U_over_t"])
        degree_grid[i, j] = row["thermal_polynomial_degree"]
        if row["comparison_valid"]:
            mixing_grid[i, j] = row["mixing_time"]

    for data, title, filename in [
        (
            degree_grid,
            "Algorithmic difficulty: polynomial degree",
            "milestone09_degree_heatmap.png",
        ),
        (
            mixing_grid,
            "Physical difficulty: mixing time",
            "milestone09_mixing_heatmap.png",
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
        plt.savefig(figures_dir / filename, dpi=180)
        plt.close()

    print()
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
