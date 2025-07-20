"""Testet explizit Randfälle der Funktion subset_finding."""

from __future__ import annotations

import numpy as np
import pytest
from shapiq import InteractionValues

from shapiq_student.subset_finding import subset_finding


def test_subset_finding_max_size_zero():
    """Testet den Spezialfall max_size=0: Rückgabe der leeren Koalition mit Basiswert e₀."""
    e0 = 42.0

    iv = InteractionValues(
        values=np.array([e0]),
        index="k-SII",
        max_order=0,
        min_order=0,
        n_players=3,
        interaction_lookup={(): 0},
        estimated=False,
        estimation_budget=None,
        baseline_value=e0,
    )

    result = subset_finding(iv, max_size=0)

    assert isinstance(result, InteractionValues)
    assert result.values.size == 1
    assert result.interaction_lookup == {(): 0}
    assert result.values[0] == e0
    assert result.s_min == result.s_max == set()
    assert result.v_min == result.v_max == e0


def test_subset_finding_smin_equals_smax():
    """Testet den Fall S_min == S_max, z.B. bei l = N."""
    iv = InteractionValues(
        values=np.array([0.0]),
        index="k-SII",
        max_order=1,
        min_order=1,
        n_players=1,
        interaction_lookup={(0,): 0},
        estimated=False,
        estimation_budget=None,
        baseline_value=0.0,
    )

    result = subset_finding(iv, max_size=1)

    assert isinstance(result, InteractionValues)
    assert isinstance(result.s_min, set)
    assert isinstance(result.s_max, set)
    assert result.s_min == result.s_max
    assert result.v_min == result.v_max


def test_subset_finding_smin_not_equals_smax():
    """Testet Fall S_min ≠ S_max bei einfachen Interaktionen."""
    iv = InteractionValues(
        values=np.array([0.5, -0.2, 0.1]),
        index="k-SII",
        max_order=2,
        min_order=1,
        n_players=2,
        interaction_lookup={
            (0,): 0,
            (1,): 1,
            (0, 1): 2,
        },
        estimated=False,
        estimation_budget=None,
        baseline_value=0.0,
    )

    result = subset_finding(iv, max_size=1)

    assert isinstance(result, InteractionValues)
    assert isinstance(result.s_min, set)
    assert isinstance(result.s_max, set)
    assert len(result.s_min) == 1
    assert len(result.s_max) == 1
    assert result.s_min.issubset({0, 1})
    assert result.s_max.issubset({0, 1})
    assert result.s_min != result.s_max
    assert result.v_min <= result.v_max


def test_subset_finding_max_size_greater_than_n_players():
    """Testet, ob bei zu großem max_size ein Fehler geworfen wird."""
    iv = InteractionValues(
        values=np.array([0.1, 0.2]),
        index="k-SII",
        max_order=1,
        min_order=1,
        n_players=1,
        interaction_lookup={(0,): 0},
        estimated=False,
        estimation_budget=None,
        baseline_value=0.0,
    )

    with pytest.raises(ValueError, match=r"max_size cannot exceed number of players."):
        subset_finding(iv, max_size=2)
