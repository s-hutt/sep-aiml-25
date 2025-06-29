"""Example test case for the Explainer class."""

from __future__ import annotations

import numpy as np
from shapiq.games.imputer.base import Imputer

from shapiq_student import GaussianImputer

from .utils import get_random_coalitions

def test_gaussian_imputer_init(knn_basic, data_test, x_explain):
    """Test init of GaussianImputer."""
    assert issubclass(GaussianImputer, Imputer), (
        "GaussianCopulaImputer should be a subclass of Imputer."
    )

def test_knn_has_shapiq_predict_function(knn_basic):
    """Check if the KNN model has _shapiq_predict_function attribute."""
    assert hasattr(knn_basic, "_shapiq_predict_function"), (
        "KNN model is missing the _shapiq_predict_function attribute."
    )
