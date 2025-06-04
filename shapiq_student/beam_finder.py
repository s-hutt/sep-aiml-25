"""Modul zur Implementierung der Beam Search Koalitionsfindung."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class BeamCoalitionFinder:
    """Klasse zur Suche nach optimalen Koalitionen mit dem Beam Search Algorithmus.

    Ermöglicht die Suche nach Koalitionen fester Größe, die den höchsten oder
    niedrigsten Interaktionswert liefern. Die Beam-Breite kann dabei dynamisch
    gewählt werden. Wird für die Beam-Breite kein Wert angegeben, so wird eine
    Heuristik als Default verwendet.
    """

    def __init__(
        self,
        features: list[int],
        interactions: dict[tuple[int, ...], float],
        evaluate_fn: Callable[[set[int], dict[tuple[int, ...], float]], float],
        beam_width: int | None = None,
    ) -> None:
        """Initialisiert den BeamCoalitionFinder.

        :param features: Liste aller verfügbaren Feature-Indizes.
        :param interactions: Dictionary mit Interaktionswerten.
        :param evaluate_fn: Bewertungsfunktion für eine Koalition.
        :param beam_width: Optional feste Beam-Breite. Falls None, wird eine Heuristik verwendet.
        """
        self.features = features
        self.interactions = interactions
        self.evaluate_fn = evaluate_fn
        self.default_width = beam_width

    def _get_beam_width(self, size: int) -> int:
        """Ermittelt eine sinnvolle Beam-Breite abhängig von der Koalitionsgröße l.

        Bei größeren Koalitionen wird die Breite reduziert, um Effizienz zu wahren.
        """
        if self.default_width is not None:
            return self.default_width
        return max(2, round(0.5 * len(self.features) / size))

    def find_max(self, size: int) -> tuple[set[int], float]:
        """Finde die Koalition der Größe `size`, die den höchsten Bewertungswert erzielt."""
        beam_width = self._get_beam_width(size)
        beam = [frozenset({f}) for f in self.features]
        beam_scores = [self.evaluate_fn(set(b), self.interactions) for b in beam]

        for _ in range(1, size):
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
        """Finde die Koalition der Größe `size`, die den niedrigsten Bewertungswert erzielt."""
        beam_width = self._get_beam_width(size)
        beam = [frozenset({f}) for f in self.features]
        beam_scores = [self.evaluate_fn(set(b), self.interactions) for b in beam]

        for _ in range(1, size):
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
