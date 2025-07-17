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
        m_star: int | None = None,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Parameter."""
        super().__init__(model, data=data, labels=labels, max_order=1)

        self.x_train = data
        self.y_train = labels
        self.k = k
        self.tau = tau
        self.alpha = alpha  # Exponent für Weighted-KNN-Shapley
        self.class_index = class_index
        self.m_star = m_star

        if method is None:
            # Automatische Erkennung für KNeighborsClassifier
            if hasattr(model, "weights") and model.weights == "distance":
                method = "weighted_knn_shapley"
            else:
                # Standard KNN-Shapley als Default
                # threshold_knn_shapley muss explizit gewählt werden
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
            return self.standard_knn_shapley(x, y_test)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x, y_test)

        if self.method == "weighted_knn_shapley":
            return self.weighted_knn_shapley(
                x, y_test
            )  # evtl später auch optional y_test übergeben

        msg = "Method not supported"
        raise ValueError(msg)

    def standard_knn_shapley(self, x_test: np.ndarray, y_test: np.ndarray) -> InteractionValues:
        """Exakte Standard KNN-Shapley Berechnung nach Jia et al. (2019), Theorem 1 - Formel (7)."""
        n = len(self.x_train)
        shapley_values = np.zeros(n)

        # Distanzen berechnen und Indizes sortieren (aufsteigend nach Distanz)
        distances = np.linalg.norm(self.x_train - x_test, axis=1)
        sorted_indices: np.ndarray = np.argsort(distances).astype(int)

        # Initialisierung: letzter Punkt in sortierter Liste (i = N), entspricht alpha_N
        idx_last = int(sorted_indices[-1])
        label_last = self.y_train[idx_last]
        shapley_values[idx_last] = (
            int(label_last == y_test) / n
        )  # gemäß Formel (6): s_{alpha_N} = 1/N * 1[y = y_test]

        # Rekursion rückwärts gemäß Formel (7)
        for i in range(n - 2, -1, -1):  # i ∈ {N-2, ..., 0}
            idx_i = sorted_indices[i]
            idx_next = sorted_indices[i + 1]

            label_i = self.y_train[idx_i]
            label_next = self.y_train[idx_next]

            i_pos = i + 1  # Theorem 1 ist 1-basiert, Python 0-basiert ⇒ i+1 ∈ {1, ..., N-1}

            # Formel (7): s_{alpha_i} = s_{alpha_{i+1}} + (1[y_i = y_test] - 1[y_{i+1} = y_test]) / K * min(K, i+1)/ (i+1)
            delta = (
                (int(label_i == y_test) - int(label_next == y_test))
                / self.k
                * (min(self.k, i_pos) / i_pos)
            )
            shapley_values[idx_i] = shapley_values[idx_next] + delta

        return InteractionValues(
            values=shapley_values,
            index="SV",
            max_order=1,
            n_players=n,
            min_order=1,
            baseline_value=0.0,
        )

    def function_a1(self, z_i: int, c_tau: int, c_tau_plus: int, y_test: int) -> float:
        """Gibt an, ob z_i bei der Vorhersage hilft oder stört - abhängig vom Label."""
        if self.y_train[z_i] == y_test:
            return (1 / c_tau) - (c_tau_plus) / (c_tau * (c_tau - 1))

        return -(c_tau_plus) / (c_tau * (c_tau - 1))

    @staticmethod
    def function_a2(c_tau: int, c: int) -> float:
        """Schätzt, wie oft zᵢ statistisch nötig ist, um in zufälligen Subsets genügend Nachbarn im Radius zu erreichen."""
        # Wie stark verändert sich die Nachbarschaft, wenn wir zᵢ zu random Subsets hinzufügen?
        # Wenn zᵢ oft gebraucht wird, um ein gutes Subset zu bilden ⇒ großer A₂-Wert -> analytisch statt Monte carlo

        a2 = 0.0
        for k in range(c + 1):
            a2 += 1 / (k + 1) - 1 / (k + 1) * (math.comb(c - k, c_tau)) / (math.comb(c + 1, c_tau))

        return a2 - 1

    def correction_term(self, z_i: int, c_tau: int, y_test: int) -> float:
        """Liefert sinnvollen Shapleywert , auch wenn Nachbarschaft zu klein ist und gleicht somit bei >2 auch den Shapley-Wert von zᵢ aus, um seinen direkten Einfluss unabhängig von anderen zu berücksichtigen."""
        if c_tau == 0:
            return 0.0

        C = len(np.unique(self.y_train))  # Klassen
        indicator = int(self.y_train[z_i] == y_test)
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
        c_tau_plus_all = np.sum(self.y_train[neighbours] == y_test)

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
                    shapley = self.correction_term(z_i, c_tau, y_test)  # Zu wenige Nachbarn

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
    def discretize_array(arr: np.ndarray, b: int = 3) -> np.ndarray:  # b=3 wie im Paper
        """Diskretisieren, um ähnliche Stufen zu erhalten."""
        return np.round(arr * (2**b - 1)) / (2**b - 1)  # fragen welche spezifische formel

    def compute_f_i(
        self,
        disc_weight: np.ndarray,
        z_i: int,
        m_star: int,
        k: int,
        w_k: list[float],
        s_to_index: dict[float, int],
    ) -> np.ndarray:
        """."""
        f = np.zeros((m_star, k - 1, len(w_k)))  # F=0 initsialisieren
        # for m in range(m_star):
        for m in range(m_star):
            if m == z_i:
                continue

            w_m = disc_weight[m]
            for s_index, s in enumerate(w_k):
                if s == w_m:  # Fragen
                    f[m, 0, s_index] = 1  # 0 für ell=1, Basisfall
        # Rekursion
        for ell in range(2, k):
            for m in range(ell, m_star):
                if m == z_i:
                    continue  # m ohne i
                w_m = disc_weight[m]
                for s_idx, s in enumerate(w_k):
                    s_prev = s - w_m
                    if s_prev in s_to_index:
                        idx_prev = s_to_index[s_prev]
                        f[m, ell - 1, s_idx] = np.sum(f[:m, ell - 2, idx_prev])

        return f

    def compute_g_i(
        self,
        disc_weight: np.ndarray,
        z_i: int,
        k: int,
        m_star: int,
        f_i: np.ndarray,
        w_k: list[float],
        s_to_index: dict[float, int],
        y_val: int,
    ) -> np.ndarray:
        """."""
        g_i = np.zeros(k)

        if disc_weight[z_i] < 0:
            g_i[0] = 1.0

        for ell in range(1, k):
            total = 0.0
            for m in range(m_star):
                if m == z_i:
                    continue  # i nicht mitzählen
                w_i = disc_weight[z_i]

                # relevante s-Werte je nach Labelgleichheit
                if self.y_train[m] == y_val:  # kippt ergebniss wenn innerhalb grenzen
                    lower = min(-w_i, 0)  # statt if schleife schneller gelöst
                    upper = max(-w_i, 0)
                else:
                    lower = min(0, -w_i)
                    upper = max(0, -w_i)

                # Summe über alle s in [lower, upper]
                for s in w_k:
                    if lower <= s <= upper and s in s_to_index:
                        s_idx = s_to_index[s]
                        total += f_i[m, ell - 1, s_idx]

            g_i[ell] = total

        return g_i

    def compute_r_i(  # nicht optimierte version, theorem 8
        self,
        f_i: np.ndarray,
        disc_weight: np.ndarray,
        w_k: list[float],
        s_to_index: dict[float, int],
        z_i: int,
        k: int,
        m_star: int,
    ) -> np.ndarray:
        """."""
        r_i = np.zeros(m_star)

        for m in range(max(z_i + 1, k), m_star):  # punkte nach z_i
            r_val = 0.0
            w_i = disc_weight[z_i]
            w_m = disc_weight[m]

            lower = min(-w_i, -w_m)
            upper = max(-w_i, -w_m)
            relevant_s = [s for s in w_k if lower <= s <= upper]

            for t in range(m):
                if t == z_i:
                    continue

                for s in relevant_s:
                    if s in s_to_index:
                        idx_s = s_to_index[s]
                        r_val += f_i[t, k - 1 - 1, idx_s]  # -1 weil Index bei 0 beginnt

            r_i[m] = r_val
        return r_i

    def weighted_knn_shapley(self, x_test: np.ndarray, y_test: int) -> InteractionValues:
        """Berechnet gewichtete  KNN-Shapley-Werte für x_test."""
        k = self.k
        distances = np.linalg.norm(self.x_train - x_test, axis=1)

        w = np.exp(-self.alpha * distances**2)  # RBF-Kernel-Gewicht mit Skalierungsfaktor
        weight = (2 * (self.y_train == y_test) - 1) * w
        sorted_indices = np.argsort(distances)

        disc_weight = self.discretize_array(weight, b=3)

        n = len(self.x_train)
        shapley_values = np.zeros(n)
        m_star = self.m_star if self.m_star is not None else int(math.sqrt(n))
        mstar_set = sorted_indices[:m_star]
        weight_levels = np.unique(disc_weight)  # Gewichtsstufen nach diskretisieren

        # Berechne  W(k) für s <- W(k)
        w_k = sorted(
            {
                round(sum(comb), 6)
                for ell in range(1, k)  # Subsets der Länge 1 bis k - 1
                for comb in combinations_with_replacement(weight_levels, ell)
            }
        )
        # Mappe jede Summe auf einen eindeutigen Index (damit wir später in F damit arbeiten können)
        s_to_index = {s: idx for idx, s in enumerate(w_k)}

        for j, z_i in enumerate(mstar_set):
            f_i = self.compute_f_i(disc_weight, j, m_star, k, w_k, s_to_index)

            r_i = self.compute_r_i(f_i, disc_weight, w_k, s_to_index, j, k, m_star)

            g_i = self.compute_g_i(disc_weight, j, k, m_star, f_i, w_k, s_to_index, y_test)

            sign_u = 1 if weight[z_i] > 0 else -1 if weight[z_i] < 0 else 0
            phi = 0.0
            for ell in range(k):
                phi += g_i[ell] / (n * math.comb(n - 1, ell))
            for m in range(max(j + 1, k + 1), m_star):
                denom = m * math.comb(m - 1, k)
                if denom != 0:
                    phi += r_i[m] / denom

            shapley_values[z_i] = sign_u * phi

        return InteractionValues(
            values=shapley_values,
            index="SV",
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )
