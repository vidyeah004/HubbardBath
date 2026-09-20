import numpy as np

from src.comparison import (
    comparison_case,
    comparison_summary,
    correlation_summary,
)


def test_correlation_summary_recovers_perfect_monotone_relation():
    rows = [
        {"x": float(i), "y": float(3 * i + 2), "comparison_valid": True}
        for i in range(1, 6)
    ]
    summary = correlation_summary(rows, "x", "y")

    np.testing.assert_allclose(summary["pearson_r"], 1.0, atol=1e-12)
    np.testing.assert_allclose(summary["spearman_rho"], 1.0, atol=1e-12)


def test_invalid_rows_are_excluded_from_correlation():
    rows = [
        {"x": 1.0, "y": 1.0, "comparison_valid": True},
        {"x": 2.0, "y": 2.0, "comparison_valid": True},
        {"x": 3.0, "y": 3.0, "comparison_valid": True},
        {"x": 100.0, "y": -100.0, "comparison_valid": False},
    ]
    summary = correlation_summary(rows, "x", "y")
    assert summary["n"] == 3
    np.testing.assert_allclose(summary["pearson_r"], 1.0, atol=1e-12)


def test_reference_comparison_case_uses_same_half_filled_target():
    row = comparison_case(
        U=4.0,
        beta=1.0,
        num_time_points=121,
        horizon_in_inverse_gaps=12.0,
        max_polynomial_degree=64,
    )

    assert row["stationary_mode_count"] == 1
    assert np.isfinite(row["mixing_time"])
    assert row["thermal_polynomial_degree"] > 0
    assert row["state_targeted_polynomial_degree"] > 0
    assert row["state_targeted_gibbs_trace_distance"] <= 1e-2
    assert row["comparison_valid"]


def test_comparison_summary_contains_both_primary_relationships():
    rows = []
    for i in range(1, 5):
        rows.append({
            "U_over_t": float(i),
            "beta_t": 1.0,
            "mixing_time": float(i),
            "inverse_liouvillian_gap": float(2 * i),
            "thermal_polynomial_degree": int(i + 2),
            "state_targeted_polynomial_degree": int(i + 3),
            "beta_alpha": float(3 * i),
            "comparison_valid": True,
        })

    summary = comparison_summary(rows)

    assert summary["valid_cases"] == 4
    assert summary["mixing_time_vs_state_targeted_degree"]["n"] == 4
    assert summary["inverse_gap_vs_state_targeted_degree"]["n"] == 4
