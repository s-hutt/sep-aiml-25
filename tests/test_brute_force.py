"""Testet die Brute-Force-Suche anhand eines Beispiel-Datensatzes."""

from __future__ import annotations

import logging

from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition


def test_brute_force_example():
    """Testet Brute-Force für eine Koalitionsgröße von l=2 mit Dummy-Interaktionen."""
    features = ["A", "B", "C", "D"]
    interactions = {
        (): 0.0,
        ("A",): 1.0,
        ("B",): -0.5,
        ("C",): 0.2,
        ("D",): -0.1,
        ("A", "B"): 0.3,
        ("B", "C"): -0.4,
        ("A", "C", "D"): 0.6,
    }

    def evaluate_fn(S, e):
        return evaluate_interaction_coalition(S, e, max_order=3)

    S_max, val_max = brute_force_find_extrema(
        features, interactions, evaluate_fn, size=2, mode="max"
    )
    S_min, val_min = brute_force_find_extrema(
        features, interactions, evaluate_fn, size=2, mode="min"
    )

    assert isinstance(S_max, set)
    assert isinstance(val_max, float)
    assert isinstance(S_min, set)
    assert isinstance(val_min, float)

    logging.info("Brute Max-Koalition (l=2): %s → %.2f", S_max, val_max)
    logging.info("Brute Min-Koalition (l=2): %s → %.2f", S_min, val_min)
