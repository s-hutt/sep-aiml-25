"""KNNExplainer-Modul zur Berechnung von KNN-basierten Shapley-Werten."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.explainer import Explainer

if TYPE_CHECKING:
    import numpy as np
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNNExplainer(Explainer):
    """Explainer für k-nearest-neighbor-basierte Shapley-Werte."""

    def __init__(
        self,
        model: KNeighborsClassifier | KNeighborsRegressor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "KNN-Shapley",
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Methode."""
        super().__init__(model, x_train, y_train)
        self.method = method

    def explain_function(self, x_test: np.ndarray) -> InteractionValues:
        """Je nach ausgewählter Methode wird passender Explainer aufgerufen."""
        if self.method == "KNN-Shapley":
            return self.knn_shapley(x_test)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x_test)

        if self.method == "weighted_knn_shapley":
            return self.weighted_knn_shapley(x_test)

        msg = "Method not supported"
        raise ValueError(msg)

    def knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet klassische KNN-Shapley-Werte für x_test."""

    def threshold_knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet threshold-basierte KNN-Shapley-Werte für x_test."""

    def weighted_knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet gewichtet basierte KNN-Shapley-Werte für x_test."""
