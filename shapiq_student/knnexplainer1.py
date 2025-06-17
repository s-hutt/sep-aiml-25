"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from shapiq.explainer import Explainer


class KNNExplainer(Explainer):
    """Klasse zur Bestimmung datenpunktbasierter Einflusswerte mithilfe des KNN-Verfahrens."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "KNN-Shapley",
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und gewählter Methode zur Einflussbewertung."""
        super().__init__(model, x_train, y_train)
        self.model = model
        self.X_train = x_train
        self.y_train = y_train
        self.method = method

    def explain_function(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet Shapley-Werte für alle Eingabepunkte in X mithilfe der gewählten KNN-Methode.

        Unterstützt: Standard, Threshold oder Weighted KNN-Shapley.
        """
        x_test = np.atleast_2d(x_test)

        results = []
        for x in x_test:
            if self.method == "standard":
                sv = self.knn_shapley_standard(x)
            elif self.method == "threshold":
                sv = self.knn_shapley_threshold(x)
            elif self.method == "weighted":
                sv = self.knn_shapley_weighted(x)
            else:
                error_msg = "Unbekannte Methode"
                raise ValueError(error_msg)
            results.append(sv)
        return np.array(results)  # NumPy-Array der Shapley-Werte

    def knn_shapley_standard(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet klassische KNN-Shapley-Werte für einen einzelnen Testpunkt.

        Dabei nimmt der Einfluss jedes Nachbarn mit seiner Position im Distanzranking ab (1/(Rang)).
        """
        neighbors, _ = self.get_neighbors(x_test)
        shapley_values = np.zeros(len(self.X_train))

        # sortiert Nachbarn nach Distanz
        for i, idx in enumerate(neighbors):
            contribution = 1.0 / (i + 1)
            shapley_values[idx] += contribution
        return shapley_values

    def knn_shapley_threshold(
        self, x_test: np.ndarray, threshold: float = 0.5
    ) -> InteractionValues:
        """Berechnet Shapley-Werte basierend auf einem festen Distanzschwellwert.

        Nur Trainingspunkte innerhalb dieses Schwellenwerts beeinflussen die Erklärwerte.
        """
        # Dummy-Rückgabe

    def knn_shapley_weighted(self, x_test: np.ndarray, alpha: float = 1.0) -> InteractionValues:
        """Berechnet gewichtet abgeschwächte KNN-Shapley-Werte auf Basis der Distanzen.

        Einflusswerte sinken mit wachsender Distanz gemäß einer exponentiellen Gewichtung.
        """
        # Dummy-Rückgabe

    def get_neighbors(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gibt Indizes und Distanzen der Trainingspunkte zurück, sortiert nach Distanz."""
        x = np.asarray(x_test).reshape(1, -1)
        dists = np.linalg.norm(self.X_train - x, axis=1)  # berechnet euklidische Distanz (L2-Norm)
        neighbors = np.argsort(dists)  # sortierte Indizes nach Distanz
        return neighbors, dists[neighbors]
