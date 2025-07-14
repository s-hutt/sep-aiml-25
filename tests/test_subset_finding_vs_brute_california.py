"""Vergleicht Beam Search mit Brute-Force auf dem California-Housing-Datensatz.

Ein RandomForestRegressor-Modell wird auf Trainingsdaten trainiert. Anschließend werden
Interaktionswerte mit dem k-SII-Index (via shapiq) berechnet. Für eine gewählte
Koalitionsgröße l wird die subset_finding-Funktion (Beam Search) gegen Brute-Force
verglichen. Ziel ist es, die Genauigkeit und Laufzeit des heuristischen Verfahrens zu evaluieren.
"""

from __future__ import annotations

import logging
import time

from shapiq import Explainer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.brute_force import brute_force_find_extrema
from shapiq_student.evaluation import evaluate_interaction_coalition
from shapiq_student.subset_finding import subset_finding

MAX_ORDER = 3


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
        max_order=MAX_ORDER,
    )
    x0 = X_train.to_numpy()[0]
    interaction_values = explainer.explain(x0, budget=512)
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
    s_min = beam_output._s_min  # noqa: SLF001
    s_max = beam_output._s_max  # noqa: SLF001
    val_min = beam_output._v_min  # noqa: SLF001
    val_max = beam_output._v_max  # noqa: SLF001

    logging.info("Brute-Force max: %s → %.3f", set(S_max_b), val_max_b)
    logging.info("Brute-Force min: %s → %.3f", set(S_min_b), val_min_b)

    logging.info("Beam max: %s → %.3f", set(s_max), val_max)
    logging.info("Beam min: %s → %.3f", set(s_min), val_min)

    # Laufzeiten
    logging.info("Laufzeit Brute Force: %.3f Sekunden", brute_time)
    logging.info("Laufzeit Beam Search: %.3f Sekunden", beam_time)
