"""Vergleicht Beam Search mit Brute-Force auf einem synthetischen Datensatz mit 1000 Features.

Ein RandomForestRegressor-Modell wird auf Trainingsdaten trainiert. Anschließend werden
Interaktionswerte mit dem k-SII-Index berechnet. Danach erfolgt ein Vergleich der
Koalitionssuche mittels Beam Search und Brute-Force für eine gewählte Koalitionsgröße l.
Es werden Laufzeiten sowie die Interaktionswerte der gefundenen Koalitionen verglichen.
"""

from __future__ import annotations

import logging
import time

from shapiq import Explainer
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition
from shapiq_student.subset_finding import subset_finding

MAX_ORDER = 4


def test_subset_finding_vs_brute_synthetic():
    """Vergleicht subset_finding (Beam) mit Brute-Force auf synthetischen Daten."""
    X, y = make_regression(n_samples=100, n_features=50, noise=0.1, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestRegressor().fit(X_train, y_train)

    explainer = Explainer(
        model=model.predict,
        data=X_train,
        index="k-SII",
        max_order=MAX_ORDER,
    )
    x0 = X_train[0]
    interaction_values = explainer.explain(x0, budget=1024)
    interactions = interaction_values.dict_values
    features = list(range(interaction_values.n_players))

    def evaluate(S: set[int], e: dict[tuple[int, ...], float]) -> float:
        return evaluate_interaction_coalition(S, e, max_order=MAX_ORDER)

    start_brute = time.perf_counter()
    S_max_b, val_max_b = brute_force_find_extrema(
        features, interactions, evaluate, size=MAX_ORDER, mode="max"
    )
    S_min_b, val_min_b = brute_force_find_extrema(
        features, interactions, evaluate, size=MAX_ORDER, mode="min"
    )
    end_brute = time.perf_counter()
    brute_time = end_brute - start_brute

    start_beam = time.perf_counter()
    beam_output = subset_finding(interaction_values=interaction_values, max_size=MAX_ORDER)
    end_beam = time.perf_counter()
    beam_time = end_beam - start_beam

    beam_results = list(beam_output.dict_values.items())
    s_max, val_max = beam_results[1]
    s_min, val_min = beam_results[0]

    logging.info("Brute-Force max: %s → %.3f", set(S_max_b), val_max_b)
    logging.info("Brute-Force min: %s → %.3f", set(S_min_b), val_min_b)
    logging.info("Beam max       : %s → %.3f", set(s_max), val_max)
    logging.info("Beam min       : %s → %.3f", set(s_min), val_min)
    logging.info("Laufzeit Brute Force: %.3f Sekunden", brute_time)
    logging.info("Laufzeit Beam Search: %.3f Sekunden", beam_time)
