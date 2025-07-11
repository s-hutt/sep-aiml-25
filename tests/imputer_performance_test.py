"""Testing the correctness and performance of the functionalities of Both imputers."""

from __future__ import annotations

import numpy as np
import shapiq
from shapiq import ConditionalImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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


class TestImputersPerformance:
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

    """Test the correctness of gaussian multivariate imputer and gaussian copula imputer."""

    def test_imputers_with_various_datasets_and_coalitions_with_dummy(self):
        """Test the correctness of gaussian multivariate imputer and gaussian copula imputer."""

        def run_imputer_test(imputer_cls, x_test, coalitions, x_explain):
            imputer = imputer_cls(model=dummy_model, data=x_test)
            imputer.fit(x=x_explain)
            output = imputer(coalitions=coalitions)
            unnormalized = imputer.value_function(coalitions=coalitions)

            mean_x_test = np.mean(x_test, axis=0)
            expected = []
            for mask in coalitions:
                imputed_sample = np.where(mask == 1, x_explain[0], mean_x_test)
                pred = dummy_model(imputed_sample.reshape(1, -1))[0]
                expected.append(pred)
            expected = np.array(expected)

            np.testing.assert_allclose(
                unnormalized,
                expected,
                rtol=2,
                err_msg=f"Unnormalized {imputer_cls.__name__} output does not match expected values",
            )
            np.testing.assert_allclose(
                unnormalized - imputer.normalization_value,
                output,
                rtol=3e-1,
                err_msg=f"Normalized output does not match value_function minus normalization_value for {imputer_cls.__name__}",
            )

        # Test 1: GaussianImputer with primitive dataset
        x_test_primitive = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        coalitions_primitive = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        x_explain_primitive = np.array([[10, 20]])
        run_imputer_test(
            GaussianImputer, x_test_primitive, coalitions_primitive, x_explain_primitive
        )

        # Test 2: GaussianImputer with larger dataset
        x_test_large = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
        )
        coalitions_large = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
            ]
        )
        x_explain_large = np.array([[10, 20, 30, 40]])
        run_imputer_test(GaussianImputer, x_test_large, coalitions_large, x_explain_large)

        # Test 3: GaussianCopulaImputer with primitive dataset
        run_imputer_test(
            GaussianCopulaImputer, x_test_primitive, coalitions_primitive, x_explain_primitive
        )

        # Test 4: GaussianCopulaImputer with larger dataset
        run_imputer_test(GaussianCopulaImputer, x_test_large, coalitions_large, x_explain_large)

        # Test 5 and 6
        # Set a seed for reproducibility
        np.random.seed(42)

        # Define mean and covariance
        mean = [0.0, 0.0]
        cov = [[1.0, 0.8], [0.8, 1.0]]  # Moderate positive correlation

        # Generate a primitive dataset of 4 samples
        x_test_gaussian = np.random.multivariate_normal(mean, cov, size=4)

        # Use a test instance to explain
        x_explain_gaussian = np.array([[1.0, -1.0]])

        # Define coalitions (0 = impute, 1 = observe)
        coalitions_gaussian = np.array([
            [0, 0],  # Both features missing
            [1, 0],  # Only second missing
            [0, 1],  # Only first missing
            [1, 1]  # Fully observed
        ])

        run_imputer_test(GaussianImputer, x_test_gaussian, coalitions_gaussian, x_explain_gaussian)
        run_imputer_test(GaussianCopulaImputer, x_test_gaussian, coalitions_gaussian, x_explain_gaussian)

    def test_conditional_performance(self):
        """Test the performance of gaussian multivariate imputer and gaussian copula imputer."""
        # Load Data
        X, y = shapiq.load_california_housing()
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values,
            test_size=0.25,
            random_state=42,
        )
        n_features = X_train.shape[1]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=n_features,
            max_features=2 / 3,
            max_samples=2 / 3,
            random_state=42,
        )
        model.fit(X_train, y_train)

        class PredictWrapper:
            def __init__(self, model):
                self.model = model

            def __call__(self, X):
                return self.model.predict(X)

        wrapped_model = PredictWrapper(model)
        assert hasattr(model, "predict"), (
            f"Model of type {type(model)} does not have a 'predict' method"
        )
        assert callable(wrapped_model), f"Model instance of type {type(model)} is not callable"

        conditional_imputer = ConditionalImputer(model=wrapped_model, data=X_test)
        gaussian_imputer = GaussianImputer(model=wrapped_model, data=X_test)

        explainer = shapiq.TabularExplainer(
            # attributes of the explainer
            model=model,
            data=X_train,
            index="SII",
            max_order=2,
            # attributes of the imputer
            imputer=conditional_imputer,
            sample_size=100,
            conditional_budget=32,
            conditional_threshold=0.04,
        )

        x_explain = X_test[100]

        _ = explainer.explain(x_explain, budget=2**n_features, random_state=0)

        # Now testing the gaussian imputers
        explainer_gauss = shapiq.TabularExplainer(
            # attributes of the explainer
            model=model,
            data=X_train,
            index="SII",
            max_order=2,
            # attributes of the imputer
            imputer=gaussian_imputer,
            sample_size=100,
            conditional_budget=32,
            conditional_threshold=0.04,
        )

        x_explain_gauss = X_test[100]

        _ = explainer_gauss.explain(x_explain_gauss, budget=2**n_features, random_state=0)
