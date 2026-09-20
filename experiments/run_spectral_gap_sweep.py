"""Milestone 04: Liouvillian gap versus thermal mixing time.

Run from the repository root:
    python experiments/run_spectral_gap_sweep.py

The default grid is intentionally small enough for exact dense calculations.
It tests a finite-system relationship and must not be interpreted as an
asymptotic many-body scaling law.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.hubbard import project_to_particle_sector, two_site_hubbard_hamiltonian
from src.open_systems import build_thermal_liouvillian, doublon_initial_state
from src.spectral import (
    loglog_power_law_fit,
    stationary_mode_count,
    thermalisation_metrics,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    U_values = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
    beta_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    epsilon = 1e-2
    rate_scale = 0.2
    rho0 = doublon_initial_state()

    rows = []

    for U in U_values:
        h_full = two_site_hubbard_hamiltonian(t=1.0, U=U)
        h_half, _ = project_to_particle_sector(h_full, 4, 2)

        for beta in beta_values:
            generator, jumps, rho_beta = build_thermal_liouvillian(
                h_half,
                beta=beta,
                rate_scale=rate_scale,
            )

            metrics = thermalisation_metrics(
                generator,
                rho0,
                rho_beta,
                epsilon=epsilon,
                horizon_in_inverse_gaps=12.0,
                num_time_points=241,
            )

            rows.append({
                "U_over_t": U,
                "beta_t": beta,
                "epsilon": epsilon,
                "number_of_jump_operators": len(jumps),
                "stationary_mode_count": stationary_mode_count(generator),
                **metrics,
            })

    csv_path = results_dir / "milestone04_gap_mixing_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    inverse_gap = np.array([row["inverse_gap"] for row in rows])
    mixing_time = np.array([row["mixing_time"] for row in rows])
    fit = loglog_power_law_fit(inverse_gap, mixing_time)

    summary = {
        "number_of_cases": len(rows),
        "epsilon": epsilon,
        "rate_scale": rate_scale,
        "fit_model": "mixing_time = prefactor * inverse_gap ** exponent",
        **fit,
        "min_gap_times_mixing_time": float(
            np.nanmin([row["gap_times_mixing_time"] for row in rows])
        ),
        "max_gap_times_mixing_time": float(
            np.nanmax([row["gap_times_mixing_time"] for row in rows])
        ),
    }

    summary_path = results_dir / "milestone04_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    order = np.argsort(inverse_gap)
    fitted_y = (
        fit["prefactor"]
        * inverse_gap[order] ** fit["exponent"]
    )

    plt.figure(figsize=(7.0, 4.8))
    for beta in beta_values:
        mask = np.array([row["beta_t"] == beta for row in rows])
        plt.scatter(
            inverse_gap[mask],
            mixing_time[mask],
            label=fr"$\beta t={beta}$",
        )
    plt.plot(
        inverse_gap[order],
        fitted_y,
        linestyle="--",
        label=fr"log-log fit, exponent={fit['exponent']:.3f}",
    )
    plt.xlabel(r"Inverse Liouvillian gap $1/\Delta_{\mathcal L}$")
    plt.ylabel(r"Mixing time $t_{\mathrm{mix}}$")
    plt.title("Finite-system gap versus thermal mixing time")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone04_mixing_vs_inverse_gap.png",
        dpi=180,
    )
    plt.close()

    gap_grid = np.zeros((len(beta_values), len(U_values)))
    mixing_grid = np.zeros_like(gap_grid)

    for row in rows:
        i = beta_values.index(row["beta_t"])
        j = U_values.index(row["U_over_t"])
        gap_grid[i, j] = row["gap"]
        mixing_grid[i, j] = row["mixing_time"]

    for data, title, filename in [
        (
            gap_grid,
            r"Liouvillian spectral gap $\Delta_{\mathcal L}$",
            "milestone04_gap_heatmap.png",
        ),
        (
            mixing_grid,
            r"Mixing time $t_{\mathrm{mix}}$",
            "milestone04_mixing_time_heatmap.png",
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

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
