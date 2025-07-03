"""KNNExplainer-Klasse für das shapiq-Paket."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from shapiq.explainer import Explainer


class KNNExplainer(Explainer):
    """Klasse zur Bestimmung datenpunktbasierter Einflusswerte mithilfe des KNN-Verfahrens."""
    pass