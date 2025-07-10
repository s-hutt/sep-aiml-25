"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier

from itertools import combinations_with_replacement
import math

import numpy as np
from shapiq.explainer import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.metrics.pairwise import cosine_similarity

"""
Papergetreuer KNNExplainer nach:
- Jia et al. (2019) "Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms"
- Wang et al. (2024) "Threshold KNN-Shapley: A Linear-Time and Privacy-Friendly Approach to Data Valuation"
- Wang et al. (2024) "Efficient Data Shapley for Weighted Nearest Neighbor Algorithms"
"""


class KNNExplainer(Explainer):
    """KNNExplainer zur Berechnung von Shapley-Werten für KNN-Modelle."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        data: np.ndarray,
        labels: np.ndarray,
        method: str | None = None,  # Basisexplainer falls nichts angegeben wird
        k: int = 5,
        tau: float = -0.5,
        alpha: float = 1.0,
        class_index: int | None = None,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Parameter."""
        super().__init__(model, data=data, labels=labels, max_order=1)

        self.x_train = data
        self.y_train = labels
        self.k = k
        self.tau = tau
        self.alpha = alpha
        self.class_index = class_index

        if method is None:
            if hasattr(model, "radius") and model.radius is not None:
                method = "threshold_knn_shapley"
            elif hasattr(model, "weights") and model.weights == "distance":
                method = "weighted_knn_shapley"
            else:
                method = "KNN-Shapley"

        self.method = method

        if method == "KNN-Shapley":
            self.mode = "normal"
        elif method == "threshold_knn_shapley":
            self.mode = "threshold"
        elif method == "weighted_knn_shapley":
            self.mode = "weighted"
        else:
            msg = f"Unknown method {method}"
            raise ValueError(msg)

    def explain_function(self, x: np.ndarray, y_test: int | None = None) -> InteractionValues:
        """Je nach ausgewählter Methode wird passender Explainer aufgerufen."""
        if y_test is None:
            if self.class_index is not None:
                y_test = self.class_index
            else:
                y_test = self.model.predict(x.reshape(1, -1))[
                    0
                ]  # selber generieren, Nutzer hat nichts angegeben

        if self.method == "KNN-Shapley":
            return self.knn_shapley_standard(x, y_test)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x, y_test)

        if self.method == "weighted_knn_shapley":
            return self.weighted_knn_shapley(
                x, y_test
            )  # evtl später auch optional y_test übergeben

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

        return InteractionValues(
            values=shapley_values,
            index="SV",
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )

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

    def threshold_knn_shapley(self, x_test: np.ndarray, y_test: int) -> InteractionValues:
        """Berechnet die analytischen Shapley-Werte für x_test nach Theorem 13 der Threshold-KNN-Methode (Wang et al., 2024)."""
        # analytisch
        # Implementiert auf Basis von Theorem 13 aus: Wang et al. (2024)

        tau = self.tau  # Radius/Threshold
        n = len(self.x_train)
        shapley_values = np.zeros(n)  # Wird noch später mit echten Shapley Werten befüllt

        cos_similarity = cosine_similarity(self.x_train, x_test.reshape(1, -1)).flatten()
        distances = -cos_similarity
        neighbours = np.where(distances <= tau)[0]
        min_neighbours = 2

        # effizienter C Vektor -> Zuerst CD dann für jeden Trainingspunkt : CD-zi (immer passend 1 abziehen)
        c_all = len(self.x_train)  # alle Punkte mit z_i
        c_tau_all = len(neighbours)  # Alle Nachbarn inklusive z_i
        c_tau_plus_all = np.sum(
            self.y_train[neighbours] == y_test
        )  # ctauall, die gleiches Label wie xtest haben

        for z_i in range(n):
            if z_i not in neighbours:
                shapley_values[z_i] = 0  # Außerhalb tau alle Shapleywerte 0

            else:
                c = c_all - 1  # Jetzt ohne z_i
                c_tau = c_tau_all - 1
                c_tau_plus = c_tau_plus_all - 1 if self.y_train[z_i] == y_test else c_tau_plus_all

                # Bedingung , dass ctau >= 2 und gesamte Formel Theorem 13, genug Nachbarn
                if c_tau >= min_neighbours:
                    shapley = self.function_a1(z_i, c_tau, c_tau_plus, y_test) * self.function_a2(
                        c_tau, c
                    ) + self.correction_term(z_i, c_tau, y_test)
                else:
                    shapley = self.correction_term(z_i, c_tau, y_test)

                shapley_values[z_i] = shapley

        return InteractionValues(
            values=shapley_values,
            index="SV",
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )

    @staticmethod
    def discretize_array(arr: np.ndarray, b: int = 3) -> np.ndarray:
        """Diskretisieren, um ähnliche Stufen zu erhalten."""
        return np.round(arr * (2**b - 1)) / (2**b - 1)

    def _compute_possible_sums(
        self, weight_levels: np.ndarray, k: int
    ) -> tuple[list[float], dict[float, int]]:
        possible_sums = sorted(
            {
                sum(comb)
                for r in range(k)
                for comb in combinations_with_replacement(weight_levels, r)
            }
        )
        return possible_sums, {s: idx for idx, s in enumerate(possible_sums)}

    def _compute_f_i(
        self,
        disc_weight: np.ndarray,
        z_i: int,
        m_star: int,
        k: int,
        possible_sums: list[float],
        sum_to_index: dict[float, int],
        num_sums: int,
    ) -> np.ndarray:
        f_i = np.zeros((m_star, k, num_sums))
        for m in range(m_star):
            if m != z_i:
                s_idx = sum_to_index[disc_weight[m]]
                f_i[m, 0, s_idx] = 1

        for level in range(1, k - 1):
            f_prev = np.sum(f_i[0:level, level - 1, :], axis=0)
            for m in range(m_star):
                if m != z_i:
                    w_m = disc_weight[m]
                    for s in possible_sums:
                        s_prev = s - w_m
                        if s_prev in sum_to_index:
                            f_i[m, level, sum_to_index[s]] = f_prev[sum_to_index[s_prev]]
        return f_i

    def _compute_g_i(
        self,
        disc_weight: np.ndarray,
        z_i: int,
        k: int,
        m_star: int,
        f_i: np.ndarray,
        possible_sums: list[float],
        sum_to_index: dict[float, int],
        y_test: int,
    ) -> np.ndarray:
        g_i = np.zeros(k)
        g_i[0] = 1.0 if disc_weight[z_i] < 0 else 0.0
        for level in range(1, k):
            g_level = 0.0
            for m in range(m_star):
                if m != z_i:
                    lower, upper = (
                        (min(-disc_weight[z_i], 0), max(-disc_weight[z_i], 0))
                        if self.y_train[m] == y_test
                        else (min(0, -disc_weight[z_i]), max(0, -disc_weight[z_i]))
                    )
                    for s in possible_sums:
                        if lower <= s <= upper and s in sum_to_index:
                            g_level += f_i[m, level, sum_to_index[s]]
            g_i[level] = g_level
        return g_i

    def weighted_knn_shapley(
        self: KNNExplainer, x_test: np.ndarray, y_test: int
    ) -> InteractionValues:
        """Berechnet gewichtete  KNN-Shapley-Werte für x_test."""
        k = self.k
        distances = np.linalg.norm(self.x_train - x_test, axis=1)
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        w = np.exp(-self.alpha * sorted_distances**2)
        weight = (2 * (self.y_train[sorted_indices] == y_test) - 1) * w
        disc_weight = self.discretize_array(weight, b=3)

        n = len(self.x_train)
        shapley_values = np.zeros(n)
        m_star = self.m_star if self.m_star is not None else int(math.sqrt(n))
        weight_levels = np.unique(disc_weight)

        possible_sums, sum_to_index = self._compute_possible_sums(weight_levels, k)
        num_sums = len(possible_sums)

        for z_i in range(n):
            xi_label = self.y_train[z_i]
            f_i = self._compute_f_i(
                disc_weight, z_i, m_star, k, possible_sums, sum_to_index, num_sums
            )

            r_0 = np.sum(np.delete(f_i[:, k - 2, :], z_i, axis=0), axis=0)
            r_i_m = 0.0
            for m in range(max(z_i + 1, k + 1), m_star):
                if m != z_i:
                    lower, upper = (
                        (
                            min(-disc_weight[z_i], -disc_weight[m]),
                            max(-disc_weight[z_i], -disc_weight[m]),
                        )
                        if xi_label == y_test
                        else (
                            min(-disc_weight[m], -disc_weight[z_i]),
                            max(-disc_weight[m], -disc_weight[z_i]),
                        )
                    )
                    for s in possible_sums:
                        if lower <= s <= upper and s in sum_to_index:
                            r_i_m += r_0[sum_to_index[s]]
                    r_0 += f_i[m, k - 2, :]

            g_i = self._compute_g_i(
                disc_weight, z_i, k, m_star, f_i, possible_sums, sum_to_index, y_test
            )

            g_sum = sum(g_i[level] / math.comb(m_star - 1, level) for level in range(k))
            sign_u = 1 if weight[z_i] < 0 else 0
            shapley_values[z_i] = sign_u * ((1 / m_star) * g_sum + r_i_m)

        return InteractionValues(
            values=shapley_values,
            index=None,
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )
