import warnings

import numpy as np
import pandas as pd

from typing import Literal

from shapiq.approximator.sampling import CoalitionSampler
from shapiq.utils.modules import check_import_module
from shapiq.games.imputer.base import Imputer

class GaussianImputer(Imputer):
    def __init__(
            self,
            model,
            data: np.ndarray,
            x: np.ndarray | None = None,
            sample_size: int = 10,
            conditional_budget: int = 128,
            conditional_threshold: float = 0.05,
            normalize: bool = True,
            categorical_features: list[int] | None = None,
            method= Literal["gaussConditional", "gaussCopula"],
            random_state: int | None = None,
    ) -> None:
        super().__init__(model, data, x, sample_size, categorical_features, random_state)
        if method not in {"gaussianConditional", "gaussCopula"}:
            raise ValueError("Currently only 'gaussConditional' and 'gaussCopula' imputers are implemented.")
        self.method = method
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold
        self.categorical_features = categorical_features

        # set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussConditional":
            self.init_background_gaussian_conditional(data)
        elif method == "gaussCopula":
            self.init_background_gauss_copula(data)

    def init_background_gaussian_conditional(self, data: np.ndarray) -> "GaussianImputer":

        if self.categorical_features:
            raise ValueError(
                f"Gaussian imputer does not support categorical features. "
                f"Found categorical feature indices: {self.categorical_features}"
            )

        # Compute the mean vector (mu) and covariance matrix (cov_mat)
        self._mu = np.mean(data, axis=0)
        cov_mat = np.cov(data, rowvar=False)

        # Ensure the covariance matrix is positive definite (like R's nearPD)
        min_eigenvalue = 1e-6
        eigvals = np.linalg.eigvalsh(cov_mat)
        if np.any(eigvals <= min_eigenvalue):
            # Regularize the covariance matrix slightly (diagonal loading)
            cov_mat += np.eye(cov_mat.shape[0]) * (min_eigenvalue - np.min(eigvals) + 1e-6)

        self._cov_mat = cov_mat

        return self

    def init_background_gauss_copula(self, data: np.ndarray) -> "GaussianImputer":
        pass

    def prepare_data_gaussian(
            x_explain: np.ndarray,
            index_features: list[int],
            S: np.ndarray,
            mu: np.ndarray,
            cov_mat: np.ndarray,
            feature_names: list[str],
            n_explain: int,
            n_features: int,
            n_MC_samples: int,
            causal_sampling: bool = False,
            causal_first_step: bool = False,
    ) -> pd.DataFrame:
        """
        Prepares Gaussian-imputed data for SHAP value estimation.

        Args:
            x_explain: Array of shape (n_explain, n_features) to explain.
            index_features: Indices of features to use from the coalition matrix S.
            S: Coalition matrix, shape (n_coalitions, n_features).
            mu: Mean vector of the feature distribution.
            cov_mat: Covariance matrix of the features.
            feature_names: List of feature names.
            n_explain: Number of explanation points.
            n_features: Total number of features.
            n_MC_samples: Number of Monte Carlo samples to draw.
            causal_sampling: Whether causal SHAP is used.
            causal_first_step: Whether this is the first causal step.

        Returns:
            pd.DataFrame containing imputed data.
        """

        n_coalitions_now = len(index_features)

        # Determine which generator function to use (stubbed for now)
        if causal_sampling:
            prepare_gauss = prepare_data_gaussian_cpp if causal_first_step else prepare_data_gaussian_cpp_caus
            reshape_prepare_gauss_output = causal_first_step
            n_MC_samples_updated = n_MC_samples if causal_first_step else n_explain
        else:
            prepare_gauss = prepare_data_gaussian_cpp
            reshape_prepare_gauss_output = True
            n_MC_samples_updated = n_MC_samples

        # Step 1: Generate standard normal MC samples
        MC_samples_mat = np.random.randn(n_MC_samples_updated, n_features)

        # Step 2: Convert to conditional Gaussian samples
        dt = prepare_gauss(
            MC_samples_mat=MC_samples_mat,
            x_explain_mat=x_explain,
            S=S[index_features, :],
            mu=mu,
            cov_mat=cov_mat,
        )

        # Step 3: Reshape (if required)
        if reshape_prepare_gauss_output:
            dt = dt.reshape((n_coalitions_now * n_explain * n_MC_samples, n_features))

        # Step 4: Create DataFrame
        df = pd.DataFrame(dt, columns=feature_names)

        # Step 5: Add metadata columns
        df["id_coalition"] = np.repeat(index_features, n_MC_samples * n_explain)
        df["id"] = np.tile(np.repeat(np.arange(n_explain), n_MC_samples), n_coalitions_now)
        df["w"] = 1.0 / n_MC_samples

        # Reorder columns
        df = df[["id_coalition", "id"] + feature_names]

        return df

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        # TODO: perhaps should be self.conditional_data instead of self.data
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction