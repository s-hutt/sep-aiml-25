"""Modul zur Implementierung der Beam Search Koalitionsfindung."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class BeamCoalitionFinder:
    """Heuristischer Algorithmus zur Suche optimaler Feature-Koalitionen mittels Beam Search.

    Die Koalitionssuche erfolgt mit adaptiver Beam-Breitenreduktion: Beginnend mit einer
    Breite von p * n (Standard: 100% der Features) wird diese pro Iteration exponentiell
    reduziert. Die Stärke des Abfalls kann über den Parameter beam_decay ∈ (0, 1] gesteuert
    werden. Ein kleinerer Wert führt zu stärkerer Reduktion (Standard: 0.3).

    Die initiale Breite kann über beam_start_fraction ebenfalls angepasst werden - z. B. bei
    sehr großen Featuremengen zur Reduktion des Anfangssuchraums.
    """

    def __init__(
        self,
        features: list[int],
        interactions: dict[tuple[int, ...], float],
        evaluate_fn: Callable[[set[int], dict[tuple[int, ...], float]], float],
        beam_start_fraction: float = 1.0,
        beam_decay: float = 0.3,
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
            Standard ist 0.3.
        """
        self.features = features
        self.interactions = interactions
        self.evaluate_fn = evaluate_fn
        self.start_fraction = min(1.0, max(beam_start_fraction, 0.01))
        self.beam_decay = min(1.0, max(beam_decay, 0.01))

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

    def find_max(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den höchsten Bewertungswert erzielt.

        Die Beam-Breite wird pro Iteration dynamisch reduziert.
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
                            score = self.evaluate_fn(set(new_S), self.interactions)
                            candidates[new_S] = score

            sorted_pairs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            beam = [S for S, _ in sorted_pairs[:beam_width]]
            beam_scores = [score for _, score in sorted_pairs[:beam_width]]

        best_idx = beam_scores.index(max(beam_scores))
        return set(beam[best_idx]), beam_scores[best_idx]

    def find_min(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den niedrigsten Bewertungswert erzielt.

        Die Beam-Breite wird pro Iteration dynamisch reduziert.
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
                            score = self.evaluate_fn(set(new_S), self.interactions)
                            candidates[new_S] = score

            sorted_pairs = sorted(candidates.items(), key=lambda x: x[1])
            beam = [S for S, _ in sorted_pairs[:beam_width]]
            beam_scores = [score for _, score in sorted_pairs[:beam_width]]

        best_idx = beam_scores.index(min(beam_scores))
        return set(beam[best_idx]), beam_scores[best_idx]
