"""Milestone 10: explicit symmetric-QSP phase synthesis.

Run from the repository root:
    python experiments/run_qsp_phase_synthesis.py

The experiment synthesizes separate QSP sequences for the even and odd
Chebyshev components of the reference thermal polynomial, independently
reconstructs each signal response from the returned full phases, and verifies
that their sum matches the original bounded thermal polynomial.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval

from src.hubbard import two_site_hubbard_hamiltonian
from src.qsp_phases import (
    qsp_imaginary_response,
    reconstructed_thermal_response,
    synthesize_thermal_parity_components,
    thermal_qsp_reconstruction_error,
)
from src.qubit_mapping import (
    lcu_normalization,
    remove_zero_terms,
    two_site_hubbard_pauli_terms,
)
from src.thermal_polynomial import (
    minimum_degree_for_sector_gibbs_error,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = repo_root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    U = 4.0
    beta = 1.0
    target_gibbs_error = 1e-2

    h = two_site_hubbard_hamiltonian(t=1.0, U=U, mu=0.0)
    terms = remove_zero_terms(
        two_site_hubbard_pauli_terms(t=1.0, U=U, mu=0.0)
    )
    alpha = lcu_normalization(terms)

    approximation, achieved_gibbs_error = (
        minimum_degree_for_sector_gibbs_error(
            h,
            beta=beta,
            alpha=alpha,
            target_trace_distance=target_gibbs_error,
            max_degree=64,
        )
    )

    even, odd = synthesize_thermal_parity_components(
        approximation,
        crit=1e-12,
        maxiter=100,
        grid_size=4001,
    )

    reconstruction_error = thermal_qsp_reconstruction_error(
        approximation,
        even,
        odd,
        grid_size=4001,
    )

    phase_csv = results_dir / "milestone10_qsp_phases.csv"
    with phase_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "component",
            "phase_index",
            "phase_radians",
        ])
        for component_name, synthesis in [
            ("even", even),
            ("odd", odd),
        ]:
            for index, phase in enumerate(synthesis.full_phases):
                writer.writerow([
                    component_name,
                    index,
                    float(phase),
                ])

    summary = {
        "reference_U_over_t": U,
        "reference_beta_t": beta,
        "reference_alpha": alpha,
        "gibbs_trace_distance_target": target_gibbs_error,
        "state_targeted_thermal_polynomial_degree":
            approximation.degree,
        "achieved_half_filled_gibbs_trace_distance":
            achieved_gibbs_error,
        "even_component": {
            "degree": even.polynomial_degree,
            "full_phase_count": len(even.full_phases),
            "solver_residual": even.solver_residual,
            "solver_iterations": even.solver_iterations,
            "max_signal_reconstruction_error":
                even.max_response_error,
        },
        "odd_component": {
            "degree": odd.polynomial_degree,
            "full_phase_count": len(odd.full_phases),
            "solver_residual": odd.solver_residual,
            "solver_iterations": odd.solver_iterations,
            "max_signal_reconstruction_error":
                odd.max_response_error,
        },
        "summed_even_odd_qsp_response_error":
            reconstruction_error,
        "explicit_qsp_phase_synthesis_completed": True,
        "full_coherent_qsvt_circuit_completed": False,
        "implementation_note": (
            "Separate symmetric-QSP phase sequences are synthesized and "
            "validated for the even and odd parity components. Their signal "
            "responses reconstruct the full thermal polynomial. A coherent "
            "ancilla/LCU combination of the two QSP sequences is not yet "
            "compiled into a complete fault-tolerant QSVT circuit."
        ),
    }

    summary_path = results_dir / "milestone10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    samples = np.linspace(-1.0, 1.0, 1001)
    target = chebval(samples, approximation.coefficients)
    even_response = qsp_imaginary_response(
        samples,
        even.full_phases,
    )
    odd_response = qsp_imaginary_response(
        samples,
        odd.full_phases,
    )
    total_response = reconstructed_thermal_response(
        samples,
        even,
        odd,
    )

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(samples, target, label="target thermal polynomial")
    plt.plot(
        samples,
        total_response,
        linestyle="--",
        label="sum of synthesized QSP responses",
    )
    plt.xlabel(r"Encoded eigenvalue $x$")
    plt.ylabel("Signal polynomial")
    plt.title("Explicit QSP reconstruction of thermal polynomial")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone10_qsp_reconstruction.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(samples, even_response, label="even QSP response")
    plt.plot(samples, odd_response, label="odd QSP response")
    plt.xlabel(r"Encoded eigenvalue $x$")
    plt.ylabel(r"$\mathrm{Im}\langle 0|U_{QSP}(x)|0\rangle$")
    plt.title("Parity-resolved symmetric-QSP responses")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "milestone10_parity_qsp_responses.png",
        dpi=180,
    )
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Wrote {phase_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
