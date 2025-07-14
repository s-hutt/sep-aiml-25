"""Testet den BeamCoalitionFinder anhand von Dummyvariablen."""

from __future__ import annotations

import logging

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.evaluation import evaluate_interaction_coalition


def test_beam_finder_dummy_input():
    """Testet BeamCoalitionFinder mit Dummy-Interaktionen."""
    features = [0, 1, 2]
    interactions = {
        (): 0.0,
        (0,): 1.0,
        (1,): 1.0,
        (2,): -2.0,
        (0, 1): 0.5,
        (1, 2): -0.4,
        (0, 1, 2): 2.0,
    }

    def evaluate(S, e):
        return evaluate_interaction_coalition(S, e, max_order=2)

    finder = BeamCoalitionFinder(
        features=features,
        interactions=interactions,
        evaluate_fn=evaluate,
    )

    s_max, v_max = finder.find_max(size=2)
    s_min, v_min = finder.find_min(size=2)

    logging.info("Beam max: %s → %.2f", sorted(s_max), v_max)
    logging.info("Beam min: %s → %.2f", sorted(s_min), v_min)

    assert isinstance(v_max, float)
    assert isinstance(v_min, float)
