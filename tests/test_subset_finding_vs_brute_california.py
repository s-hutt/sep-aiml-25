"""Vergleicht Beam Search mit Brute-Force auf dem California-Housing-Datensatz.

Ein RandomForestRegressor-Modell wird auf Trainingsdaten trainiert. Anschließend werden
Interaktionswerte mit dem k-SII-Index (via shapiq) berechnet. Für eine gewählte
Koalitionsgröße l wird die subset_finding-Funktion (Beam Search) gegen Brute-Force
verglichen. Ziel ist es, die Genauigkeit und Laufzeit des heuristischen Verfahrens zu evaluieren.
"""

from __future__ import annotations

import logging

from shapiq import Explainer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition
from shapiq_student.subset_finding import subset_finding


def test_subset_finding_vs_brute_california():
    """Vergleicht subset_finding (Beam) mit Brute-Force auf California Housing."""
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X = X.sample(n=100, random_state=42)
    y = y.loc[X.index]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestRegressor().fit(X_train, y_train)

    explainer = Explainer(
        model=model.predict,
        data=X_train.to_numpy(),
        index="k-SII",
        max_order=3,
    )
    x0 = X_train.to_numpy()[5]
    interaction_values = explainer.explain(x0, budget=512)
    interactions = interaction_values.dict_values
    features = list(range(interaction_values.n_players))

    beam_output = subset_finding(interaction_values=interaction_values, max_size=3)
    beam_sets = list(beam_output.dict_values.keys())

    def evaluate(S: set[int], e: dict[tuple[int, ...], float]) -> float:
        return evaluate_interaction_coalition(S, e, max_order=3)

    S_max_b, val_max_b = brute_force_find_extrema(
        features, interactions, evaluate, size=3, mode="max"
    )
    S_min_b, val_min_b = brute_force_find_extrema(
        features, interactions, evaluate, size=3, mode="min"
    )

    logging.info("Brute-Force max: %s → %.3f", S_max_b, val_max_b)
    logging.info("Brute-Force min: %s → %.3f", S_min_b, val_min_b)
    logging.info("Beam Subsets    : %s", beam_sets)
