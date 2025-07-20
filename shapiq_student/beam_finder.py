"""Modul zur Implementierung der Beam Search Koalitionsfindung."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_HYPEREDGE_MIN_ORDER = 3


class BeamCoalitionFinder:
    """Heuristischer Algorithmus zur Suche optimaler Feature-Koalitionen mittels Beam Search.

    Die Koalitionssuche erfolgt mit adaptiver Beam-Breitenreduktion: Beginnend mit einer
    Breite von p * n (Standard: 100% der Features) wird diese pro Iteration exponentiell
    reduziert. Die Stärke des Abfalls kann über den Parameter beam_decay ∈ (0, 1] gesteuert
    werden. Ein kleinerer Wert führt zu stärkerer Reduktion (Standard: 0.7).

    Zusätzlich erfolgt eine zweiphasige Bewertung: Solange die Beam-Breite größer als ein
    konfigurierbarer Schwellenwert (exact_threshold) ist, wird eine schnelle heuristische
    Bewertung verwendet. Diese basiert auf einer vorab durchgeführten Verteilung
    der Hyperkanten auf Einzelspieler. Sobald die Breite ≤ exact_threshold fällt, erfolgt eine
    exakte Bewertung aller Kandidaten.

    Die finale Rückgabe ist stets eine exakt bewertete Koalition mit maximalem/minimalem Wert.
    """

    def __init__(
        self,
        features: list[int],
        interactions: dict[tuple[int, ...], float],
        evaluate_fn: Callable[[set[int], dict[tuple[int, ...], float]], float],
        beam_start_fraction: float = 1.0,
        beam_decay: float = 0.7,
        exact_threshold: int = 5,
    ) -> None:
        """Initialisiert den BeamCoalitionFinder.

        :param features: Liste aller verfügbaren Feature-Indizes.
        :param interactions: Dictionary mit Interaktionswerten.
        :param evaluate_fn: Bewertungsfunktion, die einer Koalition einen Wert zuweist.
        :param beam_start_fraction: Startfaktor p ∈ (0, 1], bestimmt die initiale Beam-Breite als p * n.
            Bei großen Featuremengen kann p < 1 gewählt werden, um die Startbreite zu begrenzen.
            Standard ist 1.0 (alle Features zu Beginn berücksichtigt).
        :param beam_decay: Abfallfaktor r ∈ (0, 1], steuert die Stärke der exponentiellen Reduktion
            der Beam-Breite über die Iterationen. Kleinere Werte führen zu schnellerem Abfall.
            Standard ist 0.7.
        :param exact_threshold: Schwellwert für die Beam-Breite, ab dem die Bewertung von
            heuristisch auf exakt wechselt. Der finale Kandidat wird stets exakt bewertet.
        """
        self.features = features
        self.interactions = interactions
        self.evaluate_fn = evaluate_fn
        self.start_fraction = min(1.0, max(beam_start_fraction, 0.01))
        self.beam_decay = min(1.0, max(beam_decay, 0.01))
        self.exact_threshold = exact_threshold
        self.n_features = len(features)
        self._adjusted_weights: dict[int, float] | None = None

    def _beam_width_at(self, iteration: int) -> int:
        """Berechnet die Beam-Breite für eine gegebene Iteration innerhalb des Beam Search.

        Die initiale Beam-Breite wird als p * n festgelegt (p = beam_start_fraction).
        Ab der zweiten Iteration wird die Breite exponentiell reduziert gemäß:

            w_t = w0 * (r ** (t - 1))

        wobei w0 = Startbreite, r = beam_decay und t = aktuelle Iteration.
        Eine Mindestbreite von 2 wird immer eingehalten.

        :param iteration: Aktuelle Iteration im Beam Search (beginnend bei 1).
        :return: Ganze Zahl ≥ 2 als Beam-Breite für die aktuelle Iteration.
        """
        n = len(self.features)
        w0 = round(n * self.start_fraction)
        w0 = min(n, max(2, w0))

        if iteration == 1:
            return w0

        r = self.beam_decay
        w = w0 * (r ** (iteration - 1))

        return max(2, round(w))

    def _prepare_adjusted_weights(self) -> dict[int, float]:
        """Berechnet angepasste Einzelgewichte durch Vorabverteilung von Hyperkanten.

        Alle Interaktionen der Ordnung > 2 (Hyperkanten) werden gleichmäßig auf die enthaltenen
        Features verteilt. Die resultierenden Gewichte werden zu den bestehenden Einzelwerten
        addiert. Das Ergebnis ist ein Dictionary mit angepassten Einzelwerten für jedes Feature.

        :return: Dictionary mit Feature-Index → angepasster Einzelwert (ε_i^adjusted).
        """
        adjusted = {i: self.interactions.get((i,), 0.0) for i in self.features}
        for T, v in self.interactions.items():
            if len(T) >= _HYPEREDGE_MIN_ORDER:
                for i in T:
                    adjusted[i] += v / len(T)
        return adjusted

    def _evaluate_heuristic(self, S: set[int], adjusted_weights: dict[int, float]) -> float:
        """Bewertet eine Koalition heuristisch.

        Die Bewertung erfolgt mit dem Bias-Wert, den vorverteilten Einzelgewichten (ε_i^adjusted)
        sowie den vollständigen Kanteninteraktionen (|T| = 2), sofern beide Knoten in S enthalten
        sind. Hyperkanten werden nicht direkt einbezogen, sondern nur über die Verteilung.

        :param S: Die zu bewertende Koalition.
        :param adjusted_weights: Vorverteilte Einzelgewichte.
        :return: Heuristisch approximierter Wert der Koalition.
        """
        total = self.interactions.get((), 0.0)
        for i in S:
            total += adjusted_weights.get(i, 0.0)
        for i in S:
            for j in S:
                if i < j and (i, j) in self.interactions:
                    total += self.interactions[(i, j)]
        return total

    def find_max(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den höchsten Bewertungswert erzielt.

        Solange die Beam-Breite oberhalb des Schwellwerts liegt, wird eine schnelle heuristische
        Bewertung durchgeführt. Bei geringer Beam-Breite erfolgt eine exakte Bewertung.
        Der finale Rückgabewert ist immer exakt berechnet.
        """
        beam = [frozenset({f}) for f in self.features]
        beam_scores = [self.evaluate_fn(set(b), self.interactions) for b in beam]

        for depth in range(1, size):
            beam_width = self._beam_width_at(depth)

            candidates: dict[frozenset[int], float] = {}

            for S in beam:
                for f in self.features:
                    if f not in S:
                        new_S = frozenset(S | {f})
                        if new_S not in candidates:
                            candidates[new_S] = 0.0

            if beam_width <= self.exact_threshold:
                final_scores = {S: self.evaluate_fn(set(S), self.interactions) for S in candidates}
            else:
                weights = self._adjusted_weights
                if weights is None:
                    weights = self._prepare_adjusted_weights()
                    self._adjusted_weights = weights

                final_scores = {S: self._evaluate_heuristic(set(S), weights) for S in candidates}

            sorted_pairs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            beam = [S for S, _ in sorted_pairs[:beam_width]]
            beam_scores = [score for _, score in sorted_pairs[:beam_width]]

        best_idx = beam_scores.index(max(beam_scores))
        best_set = set(beam[best_idx])
        exact_score = self.evaluate_fn(best_set, self.interactions)

        return best_set, exact_score

    def find_min(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den niedrigsten Bewertungswert erzielt.

        Solange die Beam-Breite oberhalb des Schwellwerts liegt, wird eine schnelle heuristische
        Bewertung nach Tipp B durchgeführt. Bei geringer Beam-Breite erfolgt eine exakte Bewertung.
        Der finale Rückgabewert ist immer exakt berechnet.
        """
        beam = [frozenset({f}) for f in self.features]
        beam_scores = [self.evaluate_fn(set(b), self.interactions) for b in beam]

        for depth in range(1, size):
            beam_width = self._beam_width_at(depth)
            candidates: dict[frozenset[int], float] = {}

            for S in beam:
                for f in self.features:
                    if f not in S:
                        new_S = frozenset(S | {f})
                        if new_S not in candidates:
                            candidates[new_S] = 0.0

            if beam_width <= self.exact_threshold:
                final_scores = {S: self.evaluate_fn(set(S), self.interactions) for S in candidates}
            else:
                weights = self._adjusted_weights
                if weights is None:
                    weights = self._prepare_adjusted_weights()
                    self._adjusted_weights = weights

                final_scores = {S: self._evaluate_heuristic(set(S), weights) for S in candidates}

            sorted_pairs = sorted(final_scores.items(), key=lambda x: x[1])
            beam = [S for S, _ in sorted_pairs[:beam_width]]
            beam_scores = [score for _, score in sorted_pairs[:beam_width]]

        best_idx = beam_scores.index(min(beam_scores))
        best_set = set(beam[best_idx])
        exact_score = self.evaluate_fn(best_set, self.interactions)

        return best_set, exact_score
