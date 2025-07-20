"""Testet Fehlerbehandlung von subset_finding bei ungültigen Eingabewerten."""

from __future__ import annotations

import numpy as np
import pytest
from shapiq import InteractionValues

from shapiq_student.subset_finding import subset_finding


@pytest.fixture
def dummy_iv() -> InteractionValues:
    """Erzeugt ein einfaches InteractionValues-Objekt mit 3 Spielern."""
    return InteractionValues(
        values=np.array([0.0]),
        index="k-SII",
        max_order=1,
        min_order=1,
        n_players=3,
        interaction_lookup={(): 0},
        estimated=False,
        estimation_budget=None,
        baseline_value=0.0,
    )


def test_subset_finding_max_size_negative(dummy_iv):
    """Testet, ob negativer Wert für max_size korrekt abgefangen wird."""
    with pytest.raises(ValueError, match="must be non-negative"):
        subset_finding(dummy_iv, max_size=-1)


def test_subset_finding_max_size_too_large(dummy_iv):
    """Testet, ob max_size > n_players korrekt abgefangen wird."""
    with pytest.raises(ValueError, match=r"max_size cannot exceed number of players."):
        subset_finding(dummy_iv, max_size=10)


def test_subset_finding_max_size_wrong_type(dummy_iv):
    """Testet, ob falscher Typ für max_size zu TypeError führt."""
    with pytest.raises(TypeError, match="must be an integer"):
        subset_finding(dummy_iv, max_size="gt")
