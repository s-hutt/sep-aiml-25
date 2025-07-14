"""Unit-Tests für subset_finding mit kontrollierten Dummy-Interaktionen.

Getestet werden u.a.:

- Rückgabe von S_min, S_max, v_min, v_max bei vollständigen Interaktionen,
- Verhalten bei festen Koalitionsgrößen (max_size),
- Struktur und Korrektheit der erzeugten InteractionValues,
- Kombination mit Interaktionen bis maximaler Ordnung 3.
"""

from __future__ import annotations

import logging

import numpy as np
from shapiq import InteractionValues

from shapiq_student.subset_finding import subset_finding


def test_subset_finding_with_order3_interactions():
    """Testet subset_finding mit vollständigen Interaktionen bis Ordnung 3 und mit l=2."""
    EXPECTED_SIZE = 2

    interactions = {
        (): 1.0,
        (0,): 2.0,
        (1,): -1.0,
        (2,): 0.5,
        (0, 1): 0.3,
        (0, 2): -0.2,
        (1, 2): 0.1,
        (0, 1, 2): 5.0,
    }

    lookup = {k: i for i, k in enumerate(interactions)}
    values = np.array([interactions[k] for k in lookup])

    iv = InteractionValues(
        values=values,
        index="k-SII",
        max_order=2,
        min_order=0,
        n_players=3,
        interaction_lookup=lookup,
        estimated=True,
        estimation_budget=1024,
        baseline_value=1.0,
    )

    result = subset_finding(iv, max_size=2)

    s_min = result._s_min  # noqa: SLF001
    s_max = result._s_max  # noqa: SLF001
    v_min = result._v_min  # noqa: SLF001
    v_max = result._v_max  # noqa: SLF001

    logging.info("Subset (Order 2) max: %s → %.2f", sorted(s_max), v_max)
    logging.info("Subset (Order 2) min: %s → %.2f", sorted(s_min), v_min)

    assert isinstance(v_min, float)
    assert isinstance(v_max, float)
    assert s_min != s_max or v_min == v_max
    assert len(s_max) == EXPECTED_SIZE
    assert len(s_min) == EXPECTED_SIZE
