"""Example test case for the Explainer class."""

from __future__ import annotations

from shapiq import Explainer

from shapiq_student.knnexplainer1 import KNNExplainer


def test_is_explainer_class() -> None:
    """Test if KNNExplainer is a subclass of shapiq's Explainer."""
    assert issubclass(KNNExplainer, Explainer), "KNNExplainer should be a subclass of Explainer."
