"""Milestone 03 demonstration of Hubbard thermalisation.

Run from the repository root:
    python experiments/run_thermalisation_demo.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.open_systems import (
    build_thermal_liouvillian,
    doublon_initial_state,
    evolve_density_matrix,
    trace_distance,
)
from src.thermal import (
    average_double_occupancy_operator,
    expectation,
    nearest_neighbor_spin_z_correlation_operator,
    project_operator_to_sector,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    figures_dir = repo_root / "figures"
    results_dir = repo_root / "results"
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    U = 4.0
    beta = 1.0
    rate_scale = 0.2

    h_full = two_site_hubbard_hamiltonian(t=1.0, U=U)
    h_half, _ = project_to_particle_sector(h_full, 4, 2)

    d_half = project_operator_to_sector(
        average_double_occupancy_operator(), 4, 2
    )
    c_half = project_operator_to_sector(
        nearest_neighbor_spin_z_correlation_operator(), 4, 2
    )

    generator, jumps, rho_beta = build_thermal_liouvillian(
        h_half,
        beta=beta,
        rate_scale=rate_scale,
    )

    rho0 = doublon_initial_state(site=0)
    times = np.linspace(0.0, 160.0, 321)
    states = evolve_density_matrix(generator, rho0, times)

    rows = []
    distances = []

    for time, rho in zip(times, states):
        distance = trace_distance(rho, rho_beta)
        distances.append(distance)
        rows.append({
            "time": float(time),
            "trace_distance_to_gibbs": distance,
            "energy": expectation(rho, h_half),
            "double_occupancy": expectation(rho, d_half),
            "spin_z_correlation": expectation(rho, c_half),
        })

    output_csv = results_dir / "milestone03_thermalisation_demo.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(7.0, 4.6))
    plt.semilogy(times, distances)
    plt.xlabel(r"Time (in units of $1/t$)")
    plt.ylabel("Trace distance to Gibbs state")
    plt.title(r"Thermalisation of a doublon state: $U/t=4$, $\beta t=1$")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone03_trace_distance.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(7.0, 4.6))
    double_occupancy = [row["double_occupancy"] for row in rows]
    spin_correlation = [row["spin_z_correlation"] for row in rows]
    plt.plot(times, double_occupancy, label="double occupancy")
    plt.plot(times, spin_correlation, label=r"$\langle S^z_0 S^z_1\rangle$")
    plt.xlabel(r"Time (in units of $1/t$)")
    plt.ylabel("Observable")
    plt.title("Relaxation of Hubbard observables")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone03_observables.png",
        dpi=180,
    )
    plt.close()

    stationary_residual = np.linalg.norm(
        generator @ rho_beta.reshape(-1, order="F")
    )

    print(f"Number of jump operators: {len(jumps)}")
    print(f"Gibbs stationarity residual: {stationary_residual:.3e}")
    print(f"Initial trace distance: {distances[0]:.6f}")
    print(f"Final trace distance: {distances[-1]:.6f}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
