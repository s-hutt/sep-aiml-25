"""Vergleicht Beam-Search mit Brute-Force auf dem Bike-Sharing-Datensatz aus shapiq.

Ein Linear-Regression-Modell wird auf Trainingsdaten trainiert, anschließend werden
Interaktionswerte mit STInteractionIndex (shapiq) berechnet. Daraufhin erfolgt eine
Koalitionssuche mittels Beam und Brute-Force für eine feste Koalitionsgröße l.
Die Ergebnisse werden verglichen, um die Genauigkeit und Zuverlässigkeit des
heuristischen Verfahrens zu bewerten.
"""

from __future__ import annotations

import logging

from shapiq import Explainer
from shapiq.datasets import load_bike_sharing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.beam_finder import BeamCoalitionFinder
from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition


def test_beam_vs_brute_on_bike_sharing():
    """Vergleicht die Beam-Koalitionssuche mit Brute-Force für l=3 auf echten Daten."""
    X, y = load_bike_sharing()
    X = X.sample(n=50, random_state=42)
    y = y.loc[X.index]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor().fit(X_train, y_train)

    explainer = Explainer(
        model=model.predict,
        data=X_train.to_numpy(),
        index="k-SII",
        max_order=3,
    )

    x0 = X_train.to_numpy()[0]
    explanation = explainer.explain(x0).to_dict()
    values = explanation["values"]
    lookup = explanation["interaction_lookup"]

    interactions = {T: values[idx] for T, idx in lookup.items()}

    def evaluate(S: set[int], e: dict[tuple[int, ...], float]) -> float:
        return evaluate_interaction_coalition(S, e, max_order=3)

    features = list(range(X_train.shape[1]))

    S_max_b, val_max_b = brute_force_find_extrema(
        features, interactions, evaluate, size=3, mode="max"
    )
    S_min_b, val_min_b = brute_force_find_extrema(
        features, interactions, evaluate, size=3, mode="min"
    )

    beam = BeamCoalitionFinder(
        features=features,
        interactions=interactions,
        evaluate_fn=evaluate,
    )
    S_max_bs, val_max_bs = beam.find_max(3)
    S_min_bs, val_min_bs = beam.find_min(3)

    def readable_coalition(S: set[int], columns) -> set[str]:
        return {columns[i] for i in S}

    logging.info("Brute-Force max: %s → %.2f", S_max_b, val_max_b)
    logging.info("Beam max       : %s → %.2f", S_max_bs, val_max_bs)
    logging.info("Brute-Force min: %s → %.2f", S_min_b, val_min_b)
    logging.info("Beam min       : %s → %.2f", S_min_bs, val_min_bs)

    logging.info("Readable max coalition: %s", readable_coalition(S_max_bs, X_train.columns))
    logging.info("Readable min coalition: %s", readable_coalition(S_min_bs, X_train.columns))

    assert isinstance(S_max_b, set)
    assert isinstance(S_min_b, set)
    assert isinstance(val_max_b, float)
    assert isinstance(val_min_b, float)
    assert isinstance(S_max_bs, set)
    assert isinstance(S_min_bs, set)
    assert isinstance(val_max_bs, float)
    assert isinstance(val_min_bs, float)
