"""Testet den Subset_Finder incl. aller Unterklassen anhand eines Beispiel-Datensatzes."""

from __future__ import annotations

import logging

import pytest

from shapiq_student.evaluation_utils import compare_methods_robust, find_best_and_worst_subsets

logger = logging.getLogger(__name__)


@pytest.fixture
def test_data() -> tuple[list[int], dict[tuple[int, ...], float], int, int]:
    """Erzeugt Beispiel-Daten für Tests."""
    features = [0, 1, 2, 3]
    interactions = {
        (0,): 0.1,
        (1,): -0.2,
        (2,): 0.3,
        (3,): 0.0,
        (0, 1): 0.5,
        (1, 2): -0.1,
        (2, 3): 0.2,
        (0, 2): 0.1,
    }
    return features, interactions, 2, 2


def test_find_best_and_worst_subsets_brute(
    test_data: tuple[list[int], dict[tuple[int, ...], float], int, int],
) -> None:
    """Testet Brute-Force methode auf Korrektheit der Rückgabewerte."""
    features, interactions, subset_size, max_order = test_data

    best, best_score, worst, worst_score = find_best_and_worst_subsets(
        features,
        interactions,
        subset_size,
        method="brute",
        max_order=max_order,
    )

    logger.info("Brute-Force best subset: %s → %.2f", best, best_score)
    logger.info("Brute-Force worst subset: %s → %.2f", worst, worst_score)

    assert isinstance(best, set)
    assert isinstance(worst, set)
    assert len(best) == subset_size
    assert len(worst) == subset_size
    assert isinstance(best_score, float)
    assert isinstance(worst_score, float)
    assert best_score >= worst_score


def test_find_best_and_worst_subsets_beam(
    test_data: tuple[list[int], dict[tuple[int, ...], float], int, int],
) -> None:
    """Testett Beam Search Methode auf Korrektheit der Rückgabewerte."""
    features, interactions, subset_size, max_order = test_data

    best, best_score, worst, worst_score = find_best_and_worst_subsets(
        features,
        interactions,
        subset_size,
        method="beam",
        max_order=max_order,
        beam_start_fraction=1.0,
        beam_decay=0.5,
    )

    logger.info("Beam Search best subset: %s → %.2f", best, best_score)
    logger.info("Beam Search worst subset: %s → %.2f", worst, worst_score)

    assert isinstance(best, set)
    assert isinstance(worst, set)
    assert len(best) == subset_size
    assert len(worst) == subset_size
    assert isinstance(best_score, float)
    assert isinstance(worst_score, float)


NUM_RUNS = 3


def test_compare_methods_robust(
    test_data: tuple[list[int], dict[tuple[int, ...], float], int, int],
) -> None:
    """Vergleicht Performance der Methoden Brute-Force und Beam Search."""
    features, interactions, subset_size, max_order = test_data

    results = compare_methods_robust(
        features,
        interactions,
        subset_size,
        max_order=max_order,
        runs=NUM_RUNS,
    )

    for method in ("brute", "beam"):
        assert method in results
        res = results[method]

        logger.info(
            "%s - best subset: %s → %.2f",
            method.capitalize(),
            res["best_subset"],
            res["best_score"],
        )
        logger.info(
            "%s - worst subset: %s → %.2f",
            method.capitalize(),
            res["worst_subset"],
            res["worst_score"],
        )
        logger.info(
            "%s - avg runtime: %.4fs, std runtime: %.4fs",
            method.capitalize(),
            res["avg_runtime"],
            res["std_runtime"],
        )

        assert isinstance(res["best_subset"], set)
        assert isinstance(res["worst_subset"], set)
        assert len(res["best_subset"]) == subset_size
        assert len(res["worst_subset"]) == subset_size

        assert isinstance(res["best_score"], float)
        assert isinstance(res["worst_score"], float)

        assert isinstance(res["avg_runtime"], float)
        assert res["avg_runtime"] >= 0.0

        assert isinstance(res["std_runtime"], float)
        assert res["std_runtime"] >= 0.0

        all_runtimes = res["all_runtimes"]
        assert isinstance(all_runtimes, list)
        assert len(all_runtimes) == NUM_RUNS
        for rt in all_runtimes:
            assert isinstance(rt, float)
            assert rt >= 0.0
