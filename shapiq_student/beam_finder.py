"""Modul zur Implementierung der Beam Search Koalitionsfindung."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class BeamCoalitionFinder:
    """Heuristischer Algorithmus zur Suche optimaler Feature-Koalitionen mittels Beam Search.

    Die Klasse erlaubt die Suche nach Koalitionen fester Größe, die den höchsten (find_max)
    oder niedrigsten (find_min) Interaktionswert gemäß einer Bewertungsfunktion erzielen.

    Die Beam-Breite wird standardmäßig iterativ reduziert, um in frühen Suchphasen breit zu
    explorieren und in späteren Schritten effizient zu fokussieren. Die Reduktion kann linear,
    logarithmisch oder exponentiell erfolgen. Alternativ kann eine feste Breite angegeben werden.

    Optional erlaubt ein Startfaktor p ∈ (0, 1] die Skalierung der initialen Beam-Breite
    (Standard: 70 % von n).
    """

    def __init__(
        self,
        features: list[int],
        interactions: dict[tuple[int, ...], float],
        evaluate_fn: Callable[[set[int], dict[tuple[int, ...], float]], float],
        beam_reduction_strategy: str = "linear",
        beam_start_fraction: float = 0.7,
    ) -> None:
        """Initialisiert den BeamCoalitionFinder.

        :param features: Liste aller verfügbaren Feature-Indizes.
        :param interactions: Dictionary mit Interaktionswerten.
        :param evaluate_fn: Bewertungsfunktion, die einer Koalition einen Wert zuweist.
        :param beam_reduction_strategy: Strategie zur iterativen Reduktion der Beam-Breite
            bei wachsender Koalitionstiefe. Optionen:
                - "linear" (Standard): gleichmäßiger Abfall
                - "log": langsamer, logarithmischer Abfall
                - "exp": schneller, exponentieller Abfall
        :param beam_start_fraction: Startfaktor p ∈ (0, 1], bestimmt initiale Breite als p * n.
            Standard ist 0.7 (70 % der Feature-Anzahl).
        """
        self.features = features
        self.interactions = interactions
        self.evaluate_fn = evaluate_fn
        self.reduction_strategy = beam_reduction_strategy
        self.start_fraction = min(1.0, max(beam_start_fraction, 0.01))

    def _beam_width_at(self, iteration: int, total_steps: int) -> int:
        """Berechnet die Beam-Breite für eine gegebene Iteration innerhalb des Beam Search.

        Falls beam_width gesetzt ist, wird dieser Wert zurückgegeben. Ansonsten wird
        die Beam-Breite adaptiv reduziert, basierend auf der gewählten Strategie
        und dem initialen Startfaktor p.

        :param iteration: Aktuelle Iteration im Beam Search, beginnend bei 1.
        :param total_steps: Anzahl der Iterationen insgesamt.
        :return: Ganze Zahl ≥ 2 als Beam-Breite für die aktuelle Tiefe.
        """
        n = len(self.features)
        w0 = round(n * self.start_fraction)
        w0 = min(n, max(2, w0))

        t = iteration
        T = max(1, total_steps)

        if self.reduction_strategy == "log":
            import math

            w = w0 / math.log2(t + 2)
        elif self.reduction_strategy == "exp":
            r = 0.7
            w = w0 * (r ** (t - 1))
        else:  # linear (default)
            slope = (w0 - 2) / T
            w = w0 - slope * (t - 1)

        return max(2, round(w))

    def find_max(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den höchsten Bewertungswert erzielt.

        Die Beam-Breite wird pro Iteration dynamisch reduziert.
        """
        beam = [frozenset({f}) for f in self.features]
        beam_scores = [self.evaluate_fn(set(b), self.interactions) for b in beam]

        for depth in range(1, size):
            beam_width = self._beam_width_at(depth, size - 1)

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
            beam_width = self._beam_width_at(depth, size - 1)

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
