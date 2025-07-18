"""Modul zur Bewertung von Feature-Koalitionen anhand gegebener Interaktionswerte."""

from __future__ import annotations


def evaluate_interaction_coalition(
    S: set[int],
    interaction_values: dict[tuple[int, ...], float],
    max_order: int,
) -> float:
    """Bewertet eine Koalition S basierend auf einer Interaktionsstruktur.

    Alle Teilmengen T ⊆ S mit |T| ≤ max_order werden aufaddiert.

    Args:
        S: Die zu bewertende Feature-Koalition.
        interaction_values: Dictionary mit Interaktionswerten.
        max_order: Maximale Ordnung der Interaktionen, die berücksichtigt werden sollen.

    Returns:
        Der aggregierte Interaktionswert dieser Koalition.
    """
    total = interaction_values.get((), 0.0)

    for T, v in interaction_values.items():
        if 0 < len(T) <= max_order and set(T).issubset(S):
            total += v

    return total
