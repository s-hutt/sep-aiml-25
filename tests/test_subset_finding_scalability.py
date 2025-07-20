"""Testet die Skalierbarkeit von subset_finding (Beam) auf synthetischen Daten mit vielen Features.

Ziel ist es zu prüfen, ob der Beam-Algorithmus auch für große Feature-Anzahlen (z.B. 500 oder 1000)
und größere Koalitionsgrößen (l) noch effizient läuft. Der Brute-Force-Vergleich entfällt.
"""

from __future__ import annotations

import logging
import time

from shapiq import Explainer
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student.subset_finding import subset_finding

MAX_ORDER = 750


def test_subset_finding_scalability():
    """Testet die Laufzeit von subset_finding bei vielen Features."""
    X, y = make_regression(n_samples=100, n_features=1000, noise=0.1, random_state=42)
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

    start = time.perf_counter()
    output = subset_finding(interaction_values=interaction_values, max_size=MAX_ORDER)
    end = time.perf_counter()

    elapsed = end - start
    assert output is not None
    logging.info(
        "Beam Search erfolgreich für N=%d, l=%d, Laufzeit: %.2f Sekunden",
        X.shape[1],
        MAX_ORDER,
        elapsed,
    )
