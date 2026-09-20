"""Generate the Milestone 02 finite-temperature parameter sweep.

Run from the repository root:
    python experiments/run_thermal_sweep.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.thermal import (
    average_double_occupancy_operator,
    canonical_thermal_observables,
    nearest_neighbor_spin_z_correlation_operator,
    project_operator_to_sector,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    U_values = np.linspace(0.0, 8.0, 41)
    beta_values = np.linspace(0.0, 5.0, 41)

    d_sector = project_operator_to_sector(
        average_double_occupancy_operator(), 4, 2
    )
    c_sector = project_operator_to_sector(
        nearest_neighbor_spin_z_correlation_operator(), 4, 2
    )

    rows = []
    grids = {
        "energy": np.zeros((len(beta_values), len(U_values))),
        "entropy": np.zeros((len(beta_values), len(U_values))),
        "double_occupancy": np.zeros((len(beta_values), len(U_values))),
        "spin_z_correlation": np.zeros((len(beta_values), len(U_values))),
    }

    for u_idx, U in enumerate(U_values):
        h_full = two_site_hubbard_hamiltonian(t=1.0, U=float(U))
        h_sector, _ = project_to_particle_sector(
            h_full, num_modes=4, particle_number=2
        )

        for b_idx, beta in enumerate(beta_values):
            obs = canonical_thermal_observables(
                h_sector,
                float(beta),
                d_sector,
                c_sector,
            )
            rows.append({"U_over_t": U, "beta_t": beta, **obs})
            for key in grids:
                grids[key][b_idx, u_idx] = obs[key]

    csv_path = results_dir / "milestone02_thermal_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        ("energy", "Thermal energy"),
        ("entropy", "Von Neumann entropy"),
        ("double_occupancy", "Average double occupancy"),
        ("spin_z_correlation", "Nearest-neighbour spin-z correlation"),
    ]

    for key, title in labels:
        plt.figure(figsize=(7.0, 4.8))
        image = plt.imshow(
            grids[key],
            origin="lower",
            aspect="auto",
            extent=[U_values.min(), U_values.max(),
                    beta_values.min(), beta_values.max()],
        )
        plt.xlabel(r"Interaction strength $U/t$")
        plt.ylabel(r"Inverse temperature $\beta t$")
        plt.title(title)
        plt.colorbar(image)
        plt.tight_layout()
        plt.savefig(figures_dir / f"milestone02_{key}.png", dpi=180)
        plt.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote 4 figures to {figures_dir}")


if __name__ == "__main__":
    main()
