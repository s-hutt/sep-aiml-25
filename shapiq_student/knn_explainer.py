"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from shapiq.explainer import Explainer


class KNNExplainer(Explainer):
    """Klasse zur Bestimmung datenpunktbasierter Einflusswerte mithilfe des KNN-Verfahrens."""
