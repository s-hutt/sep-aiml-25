"""Testet die Brute-Force-Suche anhand von Dummyvariablen."""

from __future__ import annotations

import logging

from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition


def test_brute_force_dummy_input():
    """Testet brute_force_find_extrema mit dummy-Werten bei l = 2."""
    features = [0, 1, 2]
    interactions = {
        (): 1.0,
        (0,): 2.0,
        (1,): -1.0,
        (2,): 0.5,
        (0, 1): 0.1,
        (1, 2): -0.3,
        (0, 2): 0.2,
        (0, 1, 2): 5.0,
    }

    def evaluate(S, e):
        return evaluate_interaction_coalition(S, e, max_order=2)

    s_max, v_max = brute_force_find_extrema(features, interactions, evaluate, size=2, mode="max")
    s_min, v_min = brute_force_find_extrema(features, interactions, evaluate, size=2, mode="min")

    logging.info("Brute max: %s → %.2f", sorted(s_max), v_max)
    logging.info("Brute min: %s → %.2f", sorted(s_min), v_min)

    assert s_max != s_min or v_max == v_min
    assert isinstance(v_max, float)
