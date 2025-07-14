"""Unit-Tests für die Bewertungsfunktion evaluate_interaction_coalition.

Diese Tests prüfen die korrekte Aggregation der Interaktionswerte auf Basis
der gegebenen Koalition und der Ordnung (max_order). Dabei wird u.a. getestet:

- Rückgabe des Basiswerts e₀ für die leere Koalition,
- korrekte Bewertung mit Einzel- und Paarinteraktionen,
- Ignorieren von Hyperkanten oberhalb der max_order-Grenze.
"""

from __future__ import annotations

from shapiq_student.evaluation import evaluate_interaction_coalition


def test_evaluate_empty_coalition_with_e0():
    """Testet, dass die leere Koalition nur e₀ zurückgibt."""
    EXPECTED_VALUE = 42.0
    interactions = {(): 42.0}
    S = set()
    value = evaluate_interaction_coalition(S, interactions, max_order=3)
    assert value == EXPECTED_VALUE


def test_evaluate_with_single_and_pairwise():
    """Testet vollständige Bewertung mit e₀, Einzel- und Paarinteraktionen."""
    interactions = {
        (): 1.0,
        (0,): 2.0,
        (1,): -1.0,
        (0, 1): 0.5,
        (0, 1, 2): 100.0,
    }
    S = {0, 1}
    value = evaluate_interaction_coalition(S, interactions, max_order=2)
    assert value == 1.0 + 2.0 - 1.0 + 0.5  # = 2.5


def test_evaluate_ignores_above_max_order():
    """Testet, dass Hyperkanten > max_order ignoriert werden."""
    interactions = {
        (): 0.0,
        (0,): 1.0,
        (1,): 1.0,
        (0, 1, 2): 999.0,
    }
    S = {0, 1, 2}
    value = evaluate_interaction_coalition(S, interactions, max_order=2)
    assert value == 1.0 + 1.0 + 0.0
