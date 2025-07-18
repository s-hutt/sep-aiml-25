"""Hilfsfunktionen zum Vergleich verschiedener Methoden der Koalitionssuche (beam vs. brute-force)."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition

# Typalias zur Auswahl der Methode
MethodType = Literal["beam", "brute"]


def find_best_and_worst_subsets(
    features: list[int],
    interaction_values: dict[tuple[int, ...], float],
    subset_size: int,
    method: MethodType = "beam",
    max_order: int = 3,
    beam_start_fraction: float = 1.0,
    beam_decay: float = 0.3,
) -> tuple[set[int], float, set[int], float]:
    """Findet Koalitionen S_max,l und S_min,l mittels heuristischer oder exakter Suche.

    Args:
        features: Liste der Feature-Indizes.
        interaction_values: Dictionary mit Interaktionswerten.
        subset_size: Zielgröße der Koalitionen.
        method: "beam" oder "brute" zur Wahl der Suchstrategie.
        max_order: Maximale Ordnung der Interaktionen, die berücksichtigt werden sollen.
        beam_start_fraction: Startanteil für den Beam Search.
        beam_decay: Reduktionsfaktor für Beamgröße pro Iteration.

    Returns:
        Ein Tuple bestehend aus:
        - bester Koalition (Set),
        - deren Score (float),
        - schlechtester Koalition (Set),
        - deren Score (float).
    """

    def evaluate_fn(S: set[int], interactions: dict[tuple[int, ...], float]) -> float:
        return evaluate_interaction_coalition(S, interactions, max_order)

    if method == "brute":
        best, best_score = brute_force_find_extrema(
            features, interaction_values, evaluate_fn, subset_size, mode="max"
        )
        worst, worst_score = brute_force_find_extrema(
            features, interaction_values, evaluate_fn, subset_size, mode="min"
        )
    elif method == "beam":
        finder = BeamCoalitionFinder(
            features=features,
            interactions=interaction_values,
            evaluate_fn=evaluate_fn,
            beam_start_fraction=beam_start_fraction,
            beam_decay=beam_decay,
        )
        best, best_score = finder.find_max(subset_size)
        worst, worst_score = finder.find_min(subset_size)
    else:
        msg = f"Unbekannte Methode: {method}"
        raise ValueError(msg)

    return best, best_score, worst, worst_score


def compare_methods_robust(
    features: list[int],
    interactions: dict[tuple[int, ...], float],
    subset_size: int,
    max_order: int,
    runs: int = 3,
) -> dict[str, dict[str, Any]]:
    """Vergleicht Beam Search und Brute Force über mehrere Durchläufe.

    Für jede Methode wird über mehrere Runs das jeweils beste und schlechteste
    Subset bestimmt sowie die durchschnittliche Laufzeit berechnet.

    Args:
        features: Liste aller verfügbaren Feature-Indizes.
        interactions: Dictionary mit Interaktionswerten.
        subset_size: Zielgröße der Teilmengen, die betrachtet werden.
        max_order: Maximale Ordnung der Interaktionen.
        runs: Anzahl der Wiederholungen für jede Methode.

    Returns:
        Ein Dictionary mit Ergebnissen für jede Methode, inklusive:
            - best_subset / worst_subset
            - best_score / worst_score
            - alle Laufzeiten (Liste)
            - durchschnittliche und Standardabweichung der Laufzeit
    """
    results: dict[str, dict[str, Any]] = {}
    methods: list[MethodType] = ["brute", "beam"]

    for method in methods:
        runtimes: list[float] = []
        best_subset: set[int] | None = None
        best_score = float("-inf")
        worst_subset: set[int] | None = None
        worst_score = float("inf")

        for _ in range(runs):
            start = time.time()
            best, b_score, worst, w_score = find_best_and_worst_subsets(
                features,
                interactions,
                subset_size,
                method=method,
                max_order=max_order,
            )
            end = time.time()
            runtimes.append(end - start)

            if b_score > best_score:
                best_subset = best
                best_score = b_score

            if w_score < worst_score:
                worst_subset = worst
                worst_score = w_score

        results[method] = {
            "best_subset": best_subset,
            "best_score": best_score,
            "worst_subset": worst_subset,
            "worst_score": worst_score,
            "all_runtimes": runtimes,
            "avg_runtime": float(np.mean(runtimes)),
            "std_runtime": float(np.std(runtimes)),
        }

    return results
