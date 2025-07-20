"""Unit-Tests für das Modul evaluation_utils."""

from __future__ import annotations

from shapiq_student.evaluation_utils import (
    compare_methods_robust,
    find_best_and_worst_subsets,
)


def test_find_best_and_worst_subsets_dummy():
    """Testet find_best_and_worst_subsets mit einfachen Interaktionen."""
    features = [0, 1, 2]
    interactions = {
        (): 0.0,
        (0,): 1.0,
        (1,): -0.5,
        (2,): 0.2,
        (0, 1): 0.3,
        (1, 2): -0.4,
        (0, 2): 0.1,
        (0, 1, 2): 0.5,
    }

    S_max, v_max, S_min, v_min = find_best_and_worst_subsets(
        features=features,
        interaction_values=interactions,
        subset_size=2,
        method="beam",
        max_order=3,
    )

    assert isinstance(S_max, set)
    assert isinstance(S_min, set)
    assert isinstance(v_max, float)
    assert isinstance(v_min, float)
    assert v_max >= v_min


def test_compare_methods_robust_dummy():
    """Vergleicht beam und brute force auf Dummy-Daten."""
    features = [0, 1, 2]
    interactions = {
        (): 0.0,
        (0,): 1.0,
        (1,): -0.5,
        (2,): 0.2,
        (0, 1): 0.3,
        (1, 2): -0.4,
        (0, 1, 2): 0.6,
    }

    results = compare_methods_robust(
        features=features,
        interactions=interactions,
        subset_size=2,
        max_order=3,
        runs=2,
    )

    for method in ["beam", "brute"]:
        res = results[method]
        assert isinstance(res["best_subset"], set)
        assert isinstance(res["worst_subset"], set)
        assert isinstance(res["best_score"], float)
        assert isinstance(res["worst_score"], float)
        assert isinstance(res["avg_runtime"], float)
        assert res["best_score"] >= res["worst_score"]
