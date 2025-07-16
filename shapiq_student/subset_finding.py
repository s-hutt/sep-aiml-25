"""Heuristischer Koalitionsfinder auf Basis der Beam-Search-Strategie.

Die Funktion `subset_finding(...)` bestimmt approximativ die Koalition der Größe `l` (max_size),
die jeweils den höchsten (`S_max`) und niedrigsten (`S_min`) Interaktionswert gemäß einer
gegebenen Interaktionsstruktur aufweist. Die Rückgabe erfolgt als `InteractionValues`-Objekt.

Besonderheiten:
- Die Suche basiert auf einem heuristischen Beam-Search-Verfahren mit begrenztem Suchpfad.
- Unterstützt beliebige Interaktionswerte bis zu einem maximalen
  Ordnungsterm (z.B. bis Dreifachinteraktionen).
- Spezialfälle wie l=0, N=l, ungültige Werte oder Typfehler werden explizit behandelt.
- In der Rückgabe sind zusätzlich Metadaten `_s_min`, `_s_max`, `_v_min`, `_v_max` hinterlegt,
  um auch außerhalb des reinen Werte-Dictionaries gezielt auf die Extremkoalitionen zugreifen
  zu können.

Voraussetzung:
- Als Eingabe dient ein vollständiges `InteractionValues`-Objekt, wie es z.B. aus der Bibliothek
  `shapiq` erzeugt wird.
"""

from __future__ import annotations

import numpy as np
from shapiq import InteractionValues

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.evaluation import evaluate_interaction_coalition


def subset_finding(interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Findet S_max,l und S_min,l für eine Koalition der Größe max_size mittels Beam Search."""
    if not isinstance(max_size, int):
        msg = "max_size must be an integer."
        raise TypeError(msg)

    if max_size < 0:
        msg = "max_size must be non-negative."
        raise ValueError(msg)

    if max_size > interaction_values.n_players:
        msg = "max_size cannot exceed number of players."
        raise ValueError(msg)

    if max_size == 0:
        e0 = interaction_values.dict_values.get((), 0.0)
        result = InteractionValues(
            values=np.array([e0]),
            index=interaction_values.index,
            max_order=0,
            min_order=0,
            n_players=interaction_values.n_players,
            interaction_lookup={(): 0},
            estimated=True,
            estimation_budget=None,
            baseline_value=interaction_values.baseline_value,
        )
        result._s_min = set()  # noqa: SLF001
        result._s_max = set()  # noqa: SLF001
        result._v_min = e0  # noqa: SLF001
        result._v_max = e0  # noqa: SLF001
        return result

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

    v_min = evaluate(S_min, interactions)
    v_max = evaluate(S_max, interactions)

    if S_min == S_max:
        sorted_key = tuple(sorted(S_min))
        values = np.array([v_min])
        interaction_lookup = {sorted_key: 0}
    else:
        selected_items = [
            (frozenset(S_min), v_min),
            (frozenset(S_max), v_max),
        ]
        sorted_items = sorted(selected_items, key=lambda x: x[1])
        values = np.array([v for _, v in sorted_items])
        interaction_lookup = {tuple(sorted(k)): i for i, (k, _) in enumerate(sorted_items)}

    result = InteractionValues(
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

    result._s_min = S_min  # noqa: SLF001
    result._s_max = S_max  # noqa: SLF001
    result._v_min = v_min  # noqa: SLF001
    result._v_max = v_max  # noqa: SLF001

    return result
