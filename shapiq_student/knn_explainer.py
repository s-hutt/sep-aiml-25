"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier

import math

import numpy as np
from shapiq.explainer import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.metrics.pairwise import cosine_similarity

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
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "KNN-Shapley",  # Basisexplainer falls nichts angegeben wird
        k: int = 5,
        tau: float = -0.5,
        alpha: float = 1.0,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Parameter."""
        super().__init__(model, data=x_train)
        self.method = method
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.tau = tau
        self.alpha = alpha

    def explain_function(self, x: np.ndarray, y_test: int | None = None) -> InteractionValues:
        """Je nach ausgewählter Methode wird passender Explainer aufgerufen."""
        if self.method == "KNN-Shapley":
            if y_test is None:
                y_test = self.model.predict(x.reshape(1, -1))[0]
            return self.knn_shapley_standard(x, y_test)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x)  # y_test nicht nötig, macht predict intern

        if self.method == "weighted_knn_shapley":
            return self.weighted_knn_shapley(x)  # evtl später auch optional y_test übergeben

        msg = "Method not supported"
        raise ValueError(msg)

    def knn_shapley_standard(self, x_test: np.ndarray, y_test: np.ndarray) -> InteractionValues:
        """Exakte Standard KNN-Shapley Berechnung nach Jia et al. (2019), Theorem 1."""
        n = len(self.x_train)
        shapley_values = np.zeros(n)

        distances = np.linalg.norm(self.x_train - x_test, axis=1)
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

    def function_a1(self, z_i: int, c_tau: int, c_tau_plus: int, x_test_label: int) -> float:
        """Gibt an, ob z_i bei der Vorhersage hilft oder stört - abhängig vom Label."""
        if self.y_train[z_i] == x_test_label:
            return (1 / c_tau) - (c_tau_plus) / (c_tau * (c_tau - 1))

        return -(c_tau_plus) / (c_tau * (c_tau - 1))

    def function_a2(self, c_tau: int, c: int) -> float:
        """Schätzt, wie oft zᵢ statistisch nötig ist, um in zufälligen Subsets genügend Nachbarn im Radius zu erreichen."""
        # Wie stark verändert sich die Nachbarschaft, wenn wir zᵢ zu random Subsets hinzufügen?
        # Wenn zᵢ oft gebraucht wird, um ein gutes Subset zu bilden ⇒ großer A₂-Wert -> analytischh statt Monte carlo

        a2 = 0.0
        for k in range(c + 1):
            a2 += 1 / (k + 1) - 1 / (k + 1) * (math.comb(c - k, c_tau)) / (math.comb(c + 1, c_tau))

        return a2 - 1

    def correction_term(self, z_i: int, c_tau: int, x_test_label: int) -> float:
        """Liefert sinnvollen Shapleywert , auch wenn Nachbarschaft zu klein ist und gleicht somit bei >2 auch den Shapley-Wert von zᵢ aus, um seinen direkten Einfluss unabhängig von anderen zu berücksichtigen."""
        if c_tau == 0:
            return 0.0

        C = len(np.unique(self.y_train))  # Klassen
        indicator = int(self.y_train[z_i] == x_test_label)
        return (indicator - 1 / C) / c_tau

    def threshold_knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet die analytischen Shapley-Werte für x_test nach Theorem 13 der Threshold-KNN-Methode (Wang et al., 2024)."""
        # analytisch
        # Implementiert auf Basis von Theorem 13 aus: Wang et al. (2024)

        tau = self.tau  # Radius/Threshold
        n = len(self.x_train)
        x_test_label = self.model.predict(x_test.reshape(1, -1))[0]

        shapley_values = np.zeros(n)  # Wird noch später mit echten Shapley Werten befüllt

        cos_similarity = cosine_similarity(self.x_train, x_test.reshape(1, -1)).flatten()
        distances = -cos_similarity
        neighbours = np.where(distances <= tau)[0]
        min_neighbours = 2

        # effizienter C Vektor -> Zuerst CD dann für jeden Trainingspunkt : CD-zi (immer passend 1 abziehen)
        c_all = len(self.x_train)  # alle Punkte mit z_i
        c_tau_all = len(neighbours)  # Alle Nachbarn inklusive z_i
        c_tau_plus_all = np.sum(
            self.y_train[neighbours] == x_test_label
        )  # ctauall, die gleiches Label wie xtest haben

        for z_i in range(n):
            if z_i not in neighbours:
                shapley_values[z_i] = 0  # Außerhalb tau alle Shapleywerte 0

            else:
                c = c_all - 1  # Jetzt ohne z_i
                c_tau = c_tau_all - 1
                if self.y_train[z_i] == x_test_label:
                    c_tau_plus = c_tau_plus_all - 1
                else:
                    c_tau_plus = c_tau_plus_all

                # Bedingung , dass ctau >= 2 und gesamte Formel Theorem 13, genug Nachbarn
                if c_tau >= min_neighbours:
                    shapley = self.function_a1(
                        z_i, c_tau, c_tau_plus, x_test_label
                    ) * self.function_a2(c_tau, c) + self.correction_term(z_i, c_tau, x_test_label)
                else:
                    shapley = self.correction_term(z_i, c_tau, x_test_label)

                shapley_values[z_i] = shapley

        return InteractionValues(
            values=shapley_values,
            index=None,
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )
