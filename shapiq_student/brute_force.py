"""Brute-Force-Suche nach optimalen Koalitionen anhand gegebener Interaktionen."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable


def brute_force_find_extrema(
    features: list,
    interactions: dict[tuple, float],
    evaluate_fn: Callable[[set, dict[tuple, float]], float],
    size: int,
    mode: Literal["max", "min"] = "max",
) -> tuple[set, float]:
    """Durchsucht alle Koalitionen der Größe `size` und gibt das Extremum zurück.

    :param features: Liste der verfügbaren Feature-Indizes oder -Namen.
    :param interactions: Dictionary mit Interaktionswerten.
    :param evaluate_fn: Bewertungsfunktion für eine Koalition.
    :param size: Gesuchte Koalitionsgröße.
    :param mode: "max" für Maximum, "min" für Minimum.
    :return: Koalition (als Set) und zugehöriger Wert.
    """
    best_set = None
    best_value = float("-inf") if mode == "max" else float("inf")

    for S in combinations(features, size):
        S_set = set(S)
        value = evaluate_fn(S_set, interactions)

        if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
            best_set = S_set
            best_value = value

    return best_set or set(), best_value
