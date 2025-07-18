"""Testing the correctness and performance of the functionalities of Both imputers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
import shapiq
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

    def _run_imputer_test_gaussian(self, imputer_cls, x_test, coalitions, x_explain):
        """Testing the gaussian imputer with the expected output."""
        imputer = imputer_cls(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        unnormalized = imputer.value_function(coalitions=coalitions)

        expected = []
        for mask in coalitions:
            observed = mask == 1
            missing = mask == 0

            mu = np.mean(x_test, axis=0)
            sigma = np.cov(x_test.T)

            if np.all(observed):  # no missing
                imputed_sample = x_explain[0]
            else:
                # Partition mean and cov
                mu_o = mu[observed]
                mu_m = mu[missing]
                Sigma_oo = sigma[np.ix_(observed, observed)]
                Sigma_mo = sigma[np.ix_(missing, observed)]

                x_o = x_explain[0][observed]
                conditional_mean = mu_m + Sigma_mo @ np.linalg.pinv(Sigma_oo) @ (x_o - mu_o)

                imputed_sample = np.empty_like(x_explain[0])
                imputed_sample[observed] = x_o
                imputed_sample[missing] = conditional_mean

            pred = dummy_model(imputed_sample.reshape(1, -1))[0]
            expected.append(pred)

        np.testing.assert_allclose(
            unnormalized,
            expected,
            rtol=0.4,
            atol=0.05,
            err_msg=f"Unnormalized {imputer_cls.__name__} output does not match expected values",
        )
        np.testing.assert_allclose(
            unnormalized - imputer.normalization_value,
            output,
            rtol=0.4,
            atol=0.05,
            err_msg=f"Normalized output does not match value_function minus normalization_value for {imputer_cls.__name__}",
        )

    def _run_imputer_test_gaussian_copula(self, imputer_cls, x_test, coalitions, x_explain):
        """Testing the gaussian copula imputer with the expected output."""
        imputer = imputer_cls(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        unnormalized = imputer.value_function(coalitions=coalitions)

        n_samples, n_features = x_test.shape

        # Empirical CDF for each feature (for marginal transform)
        def empirical_cdf(x_col, values):
            # For each value, fraction of samples <= that value
            return np.array([(x_col <= v).mean() for v in values])

        # Inverse empirical CDF by interpolation
        def empirical_icdf(x_col, probs):
            sorted_vals = np.sort(x_col)
            cdf_vals = np.linspace(0, 1, len(sorted_vals))
            return np.interp(probs, cdf_vals, sorted_vals)

        # Transform all training data features to Gaussian latent space
        x_train_gauss = np.zeros_like(x_test, dtype=float)
        for j in range(n_features):
            # Compute empirical CDF values for training data
            ecdf_vals = empirical_cdf(x_test[:, j], x_test[:, j])
            # Convert to Gaussian quantiles
            x_train_gauss[:, j] = norm.ppf(ecdf_vals.clip(1e-6, 1 - 1e-6))

        # Compute mean and covariance in Gaussian space
        mu_gauss = np.mean(x_train_gauss, axis=0)
        sigma_gauss = np.cov(x_train_gauss.T)

        # Transform x_explain into Gaussian space
        x_explain_gauss = np.zeros(n_features)
        for j in range(n_features):
            p = (x_test[:, j] <= x_explain[0, j]).mean()
            p = np.clip(p, 1e-6, 1 - 1e-6)  # avoid extremes
            x_explain_gauss[j] = norm.ppf(p)

        expected = []

        for mask in coalitions:
            observed = mask == 1
            missing = mask == 0

            if np.all(observed):  # no missing
                imputed_sample = x_explain[0]
            else:
                # Partition mean and cov in Gaussian space
                mu_o = mu_gauss[observed]
                mu_m = mu_gauss[missing]
                Sigma_oo = sigma_gauss[np.ix_(observed, observed)]
                Sigma_mo = sigma_gauss[np.ix_(missing, observed)]

                x_o = x_explain_gauss[observed]

                # Conditional mean in Gaussian latent space
                conditional_mean_gauss = mu_m + Sigma_mo @ np.linalg.pinv(Sigma_oo) @ (x_o - mu_o)

                # Transform conditional mean back to original space using inverse empirical CDF
                imputed_sample = np.empty(n_features)
                imputed_sample[observed] = x_explain[0][observed]

                for idx, feat_idx in enumerate(np.where(missing)[0]):
                    # Compute empirical CDF probs in training data for this feature
                    probs = norm.cdf(conditional_mean_gauss[idx])
                    imputed_sample[feat_idx] = empirical_icdf(x_test[:, feat_idx], probs)

            pred = dummy_model(imputed_sample.reshape(1, -1))[0]
            expected.append(pred)

        np.testing.assert_allclose(
            unnormalized,
            expected,
            rtol=0.4,
            atol=0.05,
            err_msg=f"Unnormalized {imputer_cls.__name__} output does not match expected values",
        )
        np.testing.assert_allclose(
            unnormalized - imputer.normalization_value,
            output,
            rtol=0.4,
            atol=0.05,
            err_msg=f"Normalized output does not match value_function minus normalization_value for {imputer_cls.__name__}",
        )

    def test_imputers_with_primitive_datasets_with_dummy(self):
        """Testing the gassian imputer with primitive dataset."""
        x_test_primitive = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        coalitions_primitive = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        x_explain_primitive = np.array([[10, 20]])
        self._run_imputer_test_gaussian(
            GaussianImputer, x_test_primitive, coalitions_primitive, x_explain_primitive
        )

    def test_gaussian_imputer_with_large_dependent_dataset(self):
        """Testing the gassian imputer with larger dataset, but still linearly dependent."""
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
        self._run_imputer_test_gaussian(
            GaussianImputer, x_test_large, coalitions_large, x_explain_large
        )

    def test_gaussian_and_copula_imputers_with_large_dataset(self):
        """Testing two Imputers with large normal distributed dataset."""
        rng = np.random.default_rng(seed=42)

        mean = [0.0, 0.0]
        cov = [[1.0, 0.8], [0.8, 1.0]]

        x_test_gaussian = rng.multivariate_normal(mean, cov, size=2000)
        x_explain_gaussian = np.array([[1.0, -1.0]])

        # Define coalitions (0 = impute, 1 = observe)
        coalitions_gaussian = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        self._run_imputer_test_gaussian(
            GaussianImputer, x_test_gaussian, coalitions_gaussian, x_explain_gaussian
        )
        self._run_imputer_test_gaussian_copula(
            GaussianCopulaImputer, x_test_gaussian, coalitions_gaussian, x_explain_gaussian
        )

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

        gaussian_imputer = GaussianImputer(model=wrapped_model, data=X_test)
        gaussian_copula_imputer = GaussianCopulaImputer(model=wrapped_model, data=X_test)

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

        # Now testing the Copula imputers
        explainer_gauss_copula = shapiq.TabularExplainer(
            # attributes of the explainer
            model=model,
            data=X_train,
            index="SII",
            max_order=2,
            # attributes of the imputer
            imputer=gaussian_copula_imputer,
            sample_size=100,
            conditional_budget=32,
            conditional_threshold=0.04,
        )

        x_explain_gauss_copula = X_test[100]

        _ = explainer_gauss_copula.explain(
            x_explain_gauss_copula, budget=2**n_features, random_state=0
        )
