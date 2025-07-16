"""Vergleicht Beam Search mit Brute-Force auf dem Bike-Sharing-Datensatz aus shapiq.

Ein RandomForestRegressor-Modell wird auf Trainingsdaten trainiert. Anschließend werden
Interaktionswerte mit dem k-SII-Index (shapiq) berechnet. Daraufhin erfolgt eine
Koalitionssuche mittels Beam Search und Brute-Force für eine feste Koalitionsgröße l.
Die Ergebnisse werden verglichen, um die Genauigkeit und Zuverlässigkeit des
heuristischen Beam-Ansatzes zu bewerten.
"""

from __future__ import annotations

import logging

from shapiq import Explainer
from shapiq.datasets import load_bike_sharing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.evaluation import compare_methods_robust


def test_beam_vs_brute_on_bike_sharing():
    """Vergleicht Beam und Brute auf BikeSharing-Daten mit robustem Mehrfachvergleich."""
    X, y = load_bike_sharing()
    X = X.sample(n=100, random_state=42)
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
    explanation = explainer.explain(x0, budget=512).to_dict()
    values = explanation["values"]
    lookup = explanation["interaction_lookup"]
    interactions = {T: values[i] for T, i in lookup.items()}
    features = list(range(X_train.shape[1]))

    results = compare_methods_robust(
        features=features,
        interactions=interactions,
        subset_size=3,
        max_order=3,
        runs=5,
    )

    for method in ["brute", "beam"]:
        res = results[method]
        logging.info("Methode: %s", method)
        logging.info("  Best Subset : %s → %.2f", res["best_subset"], res["best_score"])
        logging.info("  Worst Subset: %s → %.2f", res["worst_subset"], res["worst_score"])
        logging.info("  Ø Runtime   : %.3f ± %.3f Sekunden", res["avg_runtime"], res["std_runtime"])

    assert results["beam"]["best_score"] <= results["brute"]["best_score"] + 1e-6
