"""Modul zur Teilmengenfindung für die Auswahl relevanter Merkmale basierend auf Interaktionswerten."""

from __future__ import annotations

import numpy as np
from shapiq import InteractionValues

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.evaluation import evaluate_interaction_coalition


def subset_finding(interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Findet S_max,l und S_min,l für eine Koalition der Größe max_size mittels Beam Search."""
    if max_size == 0:
        return InteractionValues(
            values=np.array([]),
            index=interaction_values.index,
            max_order=0,
            min_order=0,
            n_players=interaction_values.n_players,
            interaction_lookup={},
            estimated=True,
            estimation_budget=None,
            baseline_value=interaction_values.baseline_value,
        )

    features = list(range(interaction_values.n_players))
    interactions = interaction_values.dict_values
    max_order = interaction_values.max_order

    def evaluate(S: set[int], e: dict[tuple[int, ...], float]) -> float:
        return evaluate_interaction_coalition(S, e, max_order)

    finder = BeamCoalitionFinder(
        features=features,
        interactions=interactions,
        evaluate_fn=evaluate,
    )

    S_max, _ = finder.find_max(max_size)
    S_min, _ = finder.find_min(max_size)

    selected_items = [
        (frozenset(S_min), evaluate(S_min, interactions)),
        (frozenset(S_max), evaluate(S_max, interactions)),
    ]
    sorted_items = sorted(selected_items, key=lambda x: x[1])

    interaction_lookup = {tuple(sorted(k)): i for i, (k, _) in enumerate(sorted_items)}
    values = np.array([v for _, v in sorted_items])

    return InteractionValues(
        values=values,
        index=interaction_values.index,
        max_order=max_size,
        min_order=max_size,
        n_players=interaction_values.n_players,
        interaction_lookup=interaction_lookup,
        estimated=True,
        estimation_budget=None,
        baseline_value=interaction_values.baseline_value,
    )
