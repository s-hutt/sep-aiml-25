"""KNNExplainer-Modul zur Berechnung von KNN-basierten Shapley-Werten."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapiq.explainer import Explainer
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainer(Explainer):
    """Explainer für k-nearest-neighbor-basierte Shapley-Werte."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "KNN-Shapley",  # Basisexplainer falls nichts angegeben wird
        k: int = 5,
        t: int = 3,
        samples: int = 100,
        alpha: float = 1.0,
    ) -> None:
        """Initialisiert den KNNExplainer mit Modell, Trainingsdaten und Methode."""
        super().__init__(model, data=x_train)
        self.method = method
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.t = t
        self.samples = samples
        self.alpha = alpha

    def explain_function(self, x: np.ndarray) -> InteractionValues:
        """Je nach ausgewählter Methode wird passender Explainer aufgerufen."""
        if self.method == "KNN-Shapley":
            return self.knn_shapley(x)

        if self.method == "threshold_knn_shapley":
            return self.threshold_knn_shapley(x)

        if self.method == "weighted_knn_shapley":
            return self.weighted_knn_shapley(x)

        msg = "Method not supported"
        raise ValueError(msg)

    def knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet klassische KNN-Shapley-Werte für x_test."""

    def euclidian_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet euklidische Distanz zwischen zwei Trainingspunkten."""
        return np.linalg.norm(a - b)

    def vt_s(self, subset: list[int], x_test: np.ndarray, k: int, t: int) -> int:
        """Bewertungsfunktion v_t(S):.

        v_t(S) = 1, wenn unter den k nächsten Nachbarn in subset
        mindestens t das gleiche Label wie x_test haben.
        v_t(S) = 0, sonst.
        """
        # Indizes -> Tatsächlichen Werte und Labels
        subset_x = self.x_train[subset]

        # Distanzen berechnen um in Teilmenge knn zu finden
        distances = [self.euclidian_distance(x, x_test) for x in subset_x]
        sorted_subset = np.argsort(distances)

        subset_knn = subset if len(subset) <= k else [subset[i] for i in sorted_subset[:k]]
        # Alle Trainingspunkte in Teilmenge sind k nearest neighbor (Indizes)

        subset_knn_labels = self.y_train[subset_knn]

        x_test_label = self.model.predict(x_test.reshape(1, -1))[
            0
        ]  # Matrix mit einer Zeile und beliebig vielen Spalten

        # Labels der k nearest neighbor mit x_test Label vergelichen
        subset_knn_t = [label for label in subset_knn_labels if label == x_test_label]

        vt_S = 1 if len(subset_knn_t) >= t else 0

        return vt_S

    def threshold_knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet threshold-basierte KNN-Shapley-Werte für x_test."""
        # Ziel:
        # Für den gegebenen Testpunkt x_test sollen die Shapley-Werte aller Trainingspunkte berechnet werden.
        # Der Shapley-Wert eines Trainingspunkts x_i beschreibt seinen durchschnittlichen Beitrag zur Klassifikation von x_test.

        # Vorgehen:
        # Für jeden Trainingspunkt x_i:
        #   - Ziehe 'samples'-mal eine zufällige Teilmenge S aus den Trainingspunkten (ohne x_i)
        #   - Für jede Teilmenge S:
        #       - Berechnet v_t(S): Bewertungsfunktion, die 1 zurückgibt, wenn mindestens t von k nächsten Nachbarn von x_test in S dasselbe Label wie x_test haben (sonst 0)
        #       - Berechnet v_t(S U {x_i}): Gleich wie oben, aber x_i ist zusätzlich Teil der Menge
        #       - Berechnet den marginalen Beitrag: v_t(S U {x_i}) - v_t(S)
        #   - Der Shapley-Wert von x_i ist der Mittelwert aller marginalen Beiträge über die gezogenen Stichproben

        # Das Ergebnis ist ein Vektor mit einem Shapley-Wert pro Trainingspunkt, der angibt, wie sehr dieser Punkt zur Entscheidung für x_test beigetragen hat.

        k = self.k
        t = self.t
        samples = (
            self.samples
        )  # Wie oft soll ShapleyBerechnung wiederholt werden für verschiedene Teilmengen

        n = len(self.x_train)
        shapley_values = np.zeros(n)  # Wird noch später mit echten Shapley Werten befüllt

        # Sampling zufälliger Teilmengen
        for x_i in range(n):  # Jeden Index/Trainingspunkt durchgehen
            marginal_contributions = []
            for _ in range(samples):  # Für die einzelnen Teilmengen gilt:
                set_without_xi = [
                    j for j in range(n) if j != x_i
                ]  # Teilmenge der Trainingsdaten bilden ohne Punkt den wir gerade untersuchen

                rng = np.random.default_rng()
                subset_size = rng.integers(
                    1, len(set_without_xi) + 1
                )  # Länge der Teilmenge zufällig bestimmen( 1 <= random < set_without_xi +1)

                subset = list(rng.choice(set_without_xi, size=subset_size, replace=False))

                vt_S = self.vt_s(subset, x_test, k, t)

                # v_t(S U {x_i})
                subset_with_xi = [*subset, x_i]
                vt_Si = self.vt_s(subset_with_xi, x_test, k, t)

                marginal_contributions.append(vt_Si - vt_S)  # Marginaler Beitrag

            shapley_values[x_i] = np.mean(marginal_contributions)

        return InteractionValues(values=shapley_values)

    def vw_s(self, subset: list[int], x_test: np.ndarray, alpha: float) -> float:
        """Berechnet die gewichtete Summe der Einflüsse der Subset-Punkte auf x_test basierend auf deren Distanz."""
        subset_x = self.x_train[subset]
        distances = [self.euclidian_distance(x, x_test) for x in subset_x]

        # v(S) = sum(exp(-alpha * ||x_test - x_i||^2)) -> Nahe Punkte bekommen mehr Einfluss, entfernte fast keinen
        weight = [np.exp(-alpha * d**2) for d in distances]

        return sum(weight)

    def weighted_knn_shapley(self, x_test: np.ndarray) -> InteractionValues:
        """Berechnet gewichtete  KNN-Shapley-Werte für x_test."""
        # Statt strenge Trennung mit k -> Indirekte leichte Trennung (k = Trainingspunkte mit höherem Gewicht , Im paper : which implicitly emphasizes closer neighbors, without requiring a hard number of neighbors k

        alpha = self.alpha  # Größeres a : Nur die nächsten Nachbarn werden berücksichtigt , sonst :weitere haben Einfluß
        # a bestimmt, wie schnell der Einfluss abfällt.

        samples = self.samples

        n = len(self.x_train)
        shapley_values = np.zeros(n)  # Wird noch später mit echten Shapley Werten befüllt

        # Sampling zufälliger Teilmengen
        for x_i in range(n):  # Jeden Index/Trainingspunkt durchgehen
            marginal_contributions = []
            for _ in range(samples):
                set_without_xi = [
                    j for j in range(n) if j != x_i
                ]  # Teilmenge der Trainingsdaten bilden ohne Punkt den wir gerade untersuchen

                rng = np.random.default_rng()
                subset_size = rng.integers(
                    1, len(set_without_xi) + 1
                )  # Länge der Teilmenge zufällig bestimmen( 1 <= random < set_without_xi +1)

                subset = list(rng.choice(set_without_xi, size=subset_size, replace=False))

                vw_S = self.vw_s(subset, x_test, alpha)

                # v_t(S U {x_i})
                subset_with_xi = [*subset, x_i]
                vw_Si = self.vw_s(subset_with_xi, x_test, alpha)

                marginal_contributions.append(vw_Si - vw_S)  # Marginaler Beitrag

            shapley_values[x_i] = np.mean(marginal_contributions)

        return InteractionValues(values=shapley_values)
