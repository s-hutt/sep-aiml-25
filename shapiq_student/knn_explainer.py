"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier

from collections import defaultdict
import math
from math import comb

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
        M_star: int | None = None,
        bits: int = 3,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Parameter."""
        super().__init__(model, data=data, labels=labels, max_order=1)

        self.x_train = data
        self.y_train = labels
        self.k = k
        self.tau = tau
        self.alpha = alpha  # Exponent für Weighted-KNN-Shapley
        self.class_index = class_index
        self.M_star = M_star
        self.bits = bits

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
            return self.standard_knn_shapley(x, y_test)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x, y_test)  # y_test nicht nötig, macht predict intern

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

    def compute_discretized_weights(self, dists: np.ndarray, W: int) -> np.ndarray:
        """Berechnet diskretisierte Gewichte basierend auf normierten Distanzen."""
        max_d, min_d = dists[-1], dists[0]
        norm_weights = (max_d - dists) / (max_d - min_d + 1e-8)
        bins = np.linspace(0, 1, W)
        return bins[np.digitize(norm_weights, bins) - 1]

    # F initialisieren & berechnen (Theorem 17 - rekursive Zählung der gewichteten Subsets)
    # ruff: noqa: C901
    def compute_f_table(
        self, i: int, M_star: int, K: int, signed_weights: np.ndarray, W_vals: np.ndarray, N: int
    ) -> dict:
        """Berechnet die F-Tabelle gemäß Theorem 17 aus Wang et al. (2024).

        Die Funktion zählt rekursiv die Anzahl gewichteter Subsets mit Gewichtssummen s,
        unter Ausschluss des aktuellen Punktes i. Für ell ≥ K wird zusätzlich der Faktor
        binomial(N-m, ell-K) berücksichtigt (siehe Theorem 17).
        """
        F = defaultdict(int)
        for m in range(M_star):
            if m == i:
                continue
            wm = signed_weights[m]
            F[(m, 1, round(wm, 6))] = 1

        for ell in range(2, K):
            for m in range(M_star):
                if m == i:
                    continue
                wm = signed_weights[m]
                for s in W_vals:
                    total = 0
                    for t in range(m):
                        s_prev = round(s - wm, 6)
                        total += F.get((t, ell - 1, s_prev), 0)
                    F[(m, ell, s)] = total

        # Für ell ≥ K trägt nur m > i bei (Theorem 17: F[m, ell, s] = 0 für m < i)
        for ell in range(K, N):
            for m in range(i + 1, M_star):
                for s in W_vals:
                    total = 0
                    for t in range(m):
                        total += F.get((t, K - 1, s), 0)
                    F[(m, ell, s)] = total * comb(N - m, ell - K)

        return F

    # R berechnen gemäß Definition 10 (basierend auf Struktur aus Theorem 8 + Bedingung aus Theorem 2)
    def compute_r_table(
        self,
        i: int,
        M_star: int,
        K: int,
        signed_weights: np.ndarray,
        y_train: np.ndarray,
        y_test: int,
        F: dict,
        W_vals: np.ndarray,
    ) -> dict[int, int]:
        """Berechnet die R-Tabelle gemäß Definition 10 aus Wang et al. (2024).

        Die Funktion akkumuliert für jedes m > i alle Subsets mit K gewichteten Nachbarn,
        deren gewichtete Summe in einen bestimmten Bereich fällt. Dabei wird nach Labelgleichheit
        differenziert. Die Zählung basiert auf F und berücksichtigt die Fallunterscheidung in Theorem 8.
        """
        R = defaultdict(int)
        R_accum = defaultdict(int)
        for s in W_vals:
            R_accum[s] = sum(F.get((t, K - 1, s), 0) for t in range(min(i, M_star)))

        for m in range(max(i + 1, K + 1), M_star + 1):
            wi = signed_weights[i]
            R_val = 0
            for s in W_vals:
                if (y_train[i] == y_test and -wi <= s < 0) or (
                    y_train[i] != y_test and 0 <= s < -wi
                ):
                    R_val += R_accum[s]
            R[m] = R_val
            for s in W_vals:
                R_accum[s] += F.get((m, K - 1, s), 0)
        return R

    # G tilde berechnen (Definition 10 - über F, gemäß Theorem 6)
    def compute_g_tilde(
        self,
        i: int,
        M_star: int,
        K: int,
        signed_weights: np.ndarray,
        y_train: np.ndarray,
        y_test: int,
        F: dict,
        W_vals: np.ndarray,
    ) -> dict[int, int]:
        """Berechnet die G̃-Tabelle gemäß Definition 10 und Theorem 6 aus Wang et al. (2024).

        Für jede l < K wird über alle Trainingspunkte m ≠ i die Anzahl gültiger gewichteter Subsets summiert,
        deren Gewichtssumme innerhalb eines zulässigen Intervalls liegt. Dabei wird über F gezählt
        und abhängig vom Label von zᵢ differenziert (positiv oder negativ gewichtete Beiträge).
        """
        G_tilde = defaultdict(int)
        for ell in range(K):
            G_val = 0
            for m in range(M_star):
                if m == i:
                    continue
                wi = signed_weights[i]
                for s_raw in W_vals:
                    s = round(s_raw, 6)
                    if (y_train[i] == y_test and -wi <= s < 0) or (
                        y_train[i] != y_test and 0 <= s < -wi
                    ):
                        G_val += F.get((m, ell, s), 0)
            G_tilde[ell] = G_val
        return G_tilde

    # SV berechnen gemäß Definition 10 (strukturähnlich zu Theorem 8, aber auf M* begrenzt)
    def weighted_knn_shapley(self, x: np.ndarray, y_test: int) -> InteractionValues:
        """Berechnet approximative Shapley-Werte gemäß Definition 10 (Wang et al., 2024).

        Die Methode verwendet gewichtete Nachbarn mit diskretisierten Distanzen und zählt
        über rekursive Tabellen F, R und G̃ den Beitrag jedes Trainingspunkts zum Vorhersageergebnis.
        M* legt fest, wie viele Punkte in der Approximation berücksichtigt werden (standardmäßig √N).
        Die Rückgabe erfolgt als InteractionValues-kompatibler Vektor für die Integration in shapiq.
        """
        x_train = self.x_train
        y_train = self.y_train
        K = self.k
        N = len(x_train)
        M_star = (
            self.M_star if hasattr(self, "M_star") and self.M_star is not None else int(np.sqrt(N))
        )

        bits = self.bits if hasattr(self, "bits") else 3
        W = 2**bits
        W_vals = np.round(np.linspace(0, 1, W), 6)

        dists = np.linalg.norm(x_train - x, axis=1)
        sorted_idx = np.argsort(dists)
        dists = dists[sorted_idx]

        weights = self.compute_discretized_weights(dists, W)
        signed_weights = np.where(y_train == y_test, weights, -weights)

        shapley_values = np.zeros(N)

        for i in range(N):
            F = self.compute_f_table(i, M_star, K, signed_weights, W_vals, N)
            R = self.compute_r_table(i, M_star, K, signed_weights, y_train, y_test, F, W_vals)
            G_tilde = self.compute_g_tilde(i, M_star, K, signed_weights, y_train, y_test, F, W_vals)

            sign = 1 if weights[i] > 0 else -1 if weights[i] < 0 else 0
            phi = 0.0
            for ell in range(K):
                phi += G_tilde[ell] / (N * comb(N - 1, ell))
            for m in range(max(i + 1, K + 1), M_star + 1):
                phi += R[m] / (m * comb(m - 1, K))
            shapley_values[i] = sign * phi

        return InteractionValues(
            values=shapley_values,
            index="SV",
            max_order=1,
            n_players=len(self.x_train),
            min_order=1,
            baseline_value=0.0,
        )
