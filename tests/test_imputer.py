"""Example test copied from grading_test."""

from __future__ import annotations

import numpy as np
import pytest
from shapiq.games.imputer.base import Imputer

from shapiq_student import GaussianCopulaImputer, GaussianImputer

from .utils import get_random_coalitions


def dummy_model(x: np.ndarray) -> np.ndarray:
    """A dummy model that returns the sum of the features.

    Note:
        This callable is just here that we satisfy the Imputer's model parameter and tha we can
        check if the Imputer can be called with coalitions and returns a vector of "predictions".

    Args:
        x: Input data as a 2D numpy array with shape (n_samples, n_features).

    Returns:
        A 1D numpy array with the sum of the features for each sample.
    """
    return np.sum(x, axis=1)


class TestImputers:
    """Tests for calling the GaussianImputer and GaussianCopulaImputer with coalitions."""

    @staticmethod
    def get_coalitions(data_test) -> np.ndarray:
        """Generate random coalitions for testing."""
        X, _ = data_test
        n_features = X.shape[1]
        # get random coalitions (intentionally without specifying random seed set)
        coalitions = get_random_coalitions(n_features=n_features, n_coalitions=10)
        assert coalitions.shape == (10, n_features), "Coalitions should have the correct shape."
        return coalitions

    def test_gaussian_copula_imputer_init(self, data_test, x_explain):
        """Test init of GaussianCopulaImputer."""
        assert issubclass(GaussianCopulaImputer, Imputer), (
            "GaussianCopulaImputer should be a subclass of Imputer."
        )

        x_test, _ = data_test
        n_features = x_test.shape[1]
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test)
        assert isinstance(imputer, GaussianCopulaImputer)
        assert isinstance(imputer, Imputer)
        assert imputer.x is None, "x should be None initially."
        assert imputer.n_players == n_features

        # test with x set to x_explain
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test, x=x_explain)
        assert isinstance(imputer, GaussianCopulaImputer)
        assert isinstance(imputer, Imputer)
        assert np.array_equal(imputer.x, x_explain), "x should be set to x_explain."

    def test_gaussian_imputer_init(self, data_test, x_explain):
        """Test init of GaussianImputer."""
        assert issubclass(GaussianImputer, Imputer), (
            "GaussianCopulaImputer should be a subclass of Imputer."
        )

        x_test, _ = data_test
        n_features = x_test.shape[1]
        imputer = GaussianImputer(model=dummy_model, data=x_test)
        assert isinstance(imputer, GaussianImputer)
        assert isinstance(imputer, Imputer)
        assert imputer.x is None, "x should be None initially."
        assert imputer.n_players == n_features

        # test with x set to x_explain
        imputer = GaussianImputer(model=dummy_model, data=x_test, x=x_explain)
        assert isinstance(imputer, GaussianImputer)
        assert isinstance(imputer, Imputer)
        assert np.array_equal(imputer.x, x_explain), "x should be set to x_explain."

    def test_gaussian_can_be_called(self, data_test, x_explain):
        """Test if GaussianImputer can be called."""
        coalitions = self.get_coalitions(data_test)

        # check the GaussianImputer can be called with coalitions
        x_test, _ = data_test
        imputer = GaussianImputer(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        assert isinstance(output, np.ndarray)
        assert output.shape == (10,), "Output should be a vector of predictions."

    def test_gaussian_copula_can_be_called(self, data_test, x_explain):
        """Test if GaussianCopulaImputer can be called."""
        coalitions = self.get_coalitions(data_test)

        # check the GaussianCopulaImputer can be called with coalitions
        x_test, _ = data_test
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        assert isinstance(output, np.ndarray)
        assert output.shape == (10,), "Output should be a vector of predictions."

    def test_categorical_features(self, data_test, x_explain):
        """Test if CategoricalFeatures are correctly handled."""
        # check the GaussianCopulaImputer can be called with coalitions
        x_test, _ = data_test
        categorical_features = [0]

        with pytest.raises(ValueError, match="does not support categorical features"):
            GaussianCopulaImputer(
                model=dummy_model, data=x_test, categorical_features=categorical_features
            )

        with pytest.raises(ValueError, match="does not support categorical features"):
            GaussianImputer(
                model=dummy_model, data=x_test, categorical_features=categorical_features
            )

    def test_gaussian_transform_separate_invalid_n_y(self):
        """Test if exception are correctly handled."""
        rng = np.random.default_rng()  # create a Generator instance

        data = rng.random((5, 3))  # replace np.random.rand(5, 3)
        dummy = GaussianCopulaImputer(
            model=lambda x: x,
            data=data,
        )

        yx = rng.random((5, 3))

        with pytest.raises(ValueError, match="n_y should be less than length of yx"):
            dummy.gaussian_transform_separate(yx=yx, n_y=5)  # n_y == len(yx)

    def test_quantile_type7_empty_array(self):
        """Test if exception are correctly handled."""
        dummy = GaussianCopulaImputer(model=lambda x: x, data=np.ones((5, 2)))
        with pytest.raises(ValueError, match="Cannot compute quantile with empty array."):
            dummy.quantile_type7(np.array([]), probs=np.array([0.1, 0.5]))

    def test_quantile_type7_single_element(self):
        """Test if exception are correctly handled."""
        PI_VALUE = 3.14

        dummy = GaussianCopulaImputer(model=lambda x: x, data=np.ones((5, 2)))
        result = dummy.quantile_type7(np.array([3.14]), probs=np.array([0.25, 0.75]))
        assert np.all(result == PI_VALUE)
