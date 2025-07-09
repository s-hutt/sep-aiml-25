"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from shapiq.explainer import Explainer

"""
Papergetreuer KNNExplainer nach:
- Jia et al. (2019) "Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms"
- Wang et al. (2024) "Efficient Data Shapley for Weighted Nearest Neighbor Algorithms"
"""


class KNNExplainer(Explainer):
    """KNNExplainer zur Berechnung von Shapley-Werten für KNN-Modelle."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k: int = 5,
        mode: str = "standard",
        alpha: float = 1.0,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Parameter."""
        super().__init__(model, X_train)

        if mode not in ["standard", "threshold", "weighted"]:
            error_msg = "Mode must be 'standard', 'threshold' or 'weighted'."
            raise ValueError(error_msg)

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.mode = mode
        self.alpha = alpha  # Exponent für Weighted-KNN-Shapley

    def explain_function(self, X_test: np.ndarray, y_test: np.ndarray) -> InteractionValues:
        """Berechnet Shapley-Werte für alle Testpunkte basierend auf dem gewählten Modus."""
        shapley_matrix = []
        for x, y in zip(X_test, y_test, strict=False):
            if self.mode == "standard":
                shapley_matrix.append(self.knn_shapley_standard(x, y))
            elif self.mode == "weighted":
                shapley_matrix.append(self.knn_shapley_weighted(x, y))
        return np.array(shapley_matrix)

    def knn_shapley_standard(self, x_test: np.ndarray, y_test: np.ndarray) -> InteractionValues:
        """Exakte Standard KNN-Shapley Berechnung nach Jia et al. (2019), Theorem 1."""
        n = len(self.X_train)
        shapley_values = np.zeros(n)

        distances = np.linalg.norm(self.X_train - x_test, axis=1)
        sorted_indices: np.ndarray = np.argsort(distances).astype(int)

        idx_last = int(
            sorted_indices[-1]
        )  # Rekursion beginnend ab letztem Punkt in sortierter Liste
        label_last = self.y_train[idx_last]
        shapley_values[idx_last] = int(label_last == y_test) / self.k

        for i in range(n - 2, -1, -1):
            idx_i = sorted_indices[i]
            idx_next = sorted_indices[i + 1]

            label_i = self.y_train[idx_i]
            label_next = self.y_train[idx_next]

            # berechnet Marginalbeitrag
            delta = (int(label_i == y_test) - int(label_next == y_test)) / min(self.k, i + 1)
            shapley_values[idx_i] = shapley_values[idx_next] + delta

        return shapley_values
