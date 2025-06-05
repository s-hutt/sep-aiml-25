"""Testet den BeamCoalitionFinder anhand eines Beispiel-Datensatzes."""

from __future__ import annotations

import logging

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.evaluation import evaluate_interaction_coalition


def test_beam_finder_example():
    """Testet die Beam-Search für eine Koalitionsgröße von l=2 mit Dummy-Interaktionen."""
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

    beam = BeamCoalitionFinder(
        features=features,
        interactions=interactions,
        evaluate_fn=lambda S, e: evaluate_interaction_coalition(S, e, max_order=3),
        beam_reduction_strategy="linear",
        beam_start_fraction=0.7,
    )

    S_max, val_max = beam.find_max(2)
    S_min, val_min = beam.find_min(2)

    assert isinstance(S_max, set)
    assert isinstance(val_max, float)
    assert isinstance(S_min, set)
    assert isinstance(val_min, float)

    logging.info("Max-Koalition (l=2): %s → %.2f", S_max, val_max)
    logging.info("Min-Koalition (l=2): %s → %.2f", S_min, val_min)
