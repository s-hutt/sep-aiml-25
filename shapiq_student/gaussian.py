"""Implementation of the Gaussian conditional imputer using multivariate normal distributions.

This module defines the GaussianImputer class, which performs conditional imputation
based on the multivariate Gaussian assumption. It is used in feature attribution methods
such as SHAP, where imputing missing features conditionally is required for estimating
model values in the presence of partial feature subsets (coalitions). (For gaussian Copula approach see copula.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.linalg import cholesky, pinv
from shapiq.games.imputer.conditional_imputer import ConditionalImputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class GaussianImputer(ConditionalImputer):
    """Conditional imputer based on multivariate Gaussian distributions.

    This imputer assumes a multivariate normal distribution over the features and
    uses conditional Gaussian sampling to fill in missing features, depending on
    which features are present (coalition sets).

    Args:
        model: The predictive model to explain.
        data (np.ndarray): Background data used to estimate the Gaussian distribution.
        x (np.ndarray | None): Instances to explain (optional at init).
        sample_size (int): Number of Monte Carlo samples per coalition. Default is 10.
        conditional_budget (int): Not used directly here, reserved for compatibility.
        conditional_threshold (float): Not used directly here, reserved for compatibility.
        normalize (bool): Whether to normalize predictions by the empty prediction.
        categorical_features (list[int] | None): Indices of categorical features. Not supported.
        method (Literal["gaussConditional"]): Method identifier for this imputer.
        random_state (int | None): Optional random seed for reproducibility.
    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        sample_size: int = 10,
        conditional_budget: int = 128,
        conditional_threshold: float = 0.05,
        normalize: bool = True,
        categorical_features: list[int] | None = None,
        method: Literal["gaussConditional"] = "gaussConditional",
        random_state: int | None = None,
    ) -> None:
        """Initialize the GaussianImputer.

        Args:
            model: The predictive model to explain. Expected to have a `predict` method.
            data (np.ndarray): Background data used to estimate the Gaussian distribution.
            x (np.ndarray | None, optional): Data instances to explain. Defaults to None.
            sample_size (int, optional): Number of Monte Carlo samples per coalition. Default is 10.
            conditional_budget (int, optional): Budget parameter, reserved for compatibility. Default is 128.
            conditional_threshold (float, optional): Threshold parameter, reserved for compatibility. Default is 0.05.
            normalize (bool, optional): Whether to normalize predictions by the empty prediction. Default is True.
            categorical_features (list[int] | None, optional): Indices of categorical features. Not supported by this imputer. Default is None.
            method (Literal["gaussConditional"], optional): Method identifier for this imputer. Must be "gaussConditional". Default is "gaussConditional".
            random_state (int | None, optional): Random seed for reproducibility. Default is None.

        Raises:
            ValueError: If the method specified is not "gaussConditional".
            ValueError: If categorical features are provided (not supported).
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        if method not in {"gaussConditional"}:
            msg = "This contructor is for gaussianConditional imputers only."
            raise ValueError(msg)

        self.method = method
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold

        # set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussConditional":
            self.init_background(data)

    def init_background(self, data: np.ndarray) -> GaussianImputer:
        """Initializes the background Gaussian distribution using the input data.

        Computes the empirical mean and (regularized) covariance matrix from the background data.

        Args:
            data (np.ndarray): Background dataset (n_samples, n_features).

        Returns:
            Self (GaussianImputer): The fitted imputer instance.
        """
        if self._cat_features:
            msg = (
                "Gaussian imputer does not support categorical features. "
                f"Found categorical feature indices: {self._cat_features}"
            )
            raise ValueError(msg)

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

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the model predictions for each coalition via conditional Gaussian sampling.

        Args:
            coalitions (np.ndarray): Boolean array (n_subsets, n_features),
                                     where True means the feature is present.

        Returns:
            np.ndarray: Predicted values for each coalition (n_subsets,).
        """
        n_coalitions, n_features = coalitions.shape

        mu = self._mu
        cov = self._cov_mat
        n_samples = self.sample_size  # or any desired MC sample size

        # Standard normal samples
        rng = np.random.default_rng()
        MC_samples = rng.standard_normal((n_samples, n_features))

        x_explain = self._x  # shape (1, n_features)

        # Run conditional sampling
        imputed_data = self._prepare_data_gaussian_py(
            MC_samples_mat=MC_samples,
            x_explain_mat=x_explain,
            S=coalitions.astype(float),  # shape (n_coalitions, n_features)
            mu=mu,
            cov_mat=cov,
        )  # shape: (n_samples, n_coalitions, n_features)

        # Flatten for prediction
        flat_input = imputed_data.reshape(-1, n_features)
        predictions = self.predict(flat_input)

        # Reshape and average predictions per coalition
        predictions = predictions.reshape(n_samples, n_coalitions)
        avg_predictions = predictions.mean(axis=0)

        # Handle empty coalitions (all features False)
        empty_coalitions = ~np.any(coalitions, axis=1)
        avg_predictions[empty_coalitions] = self.empty_prediction

        return avg_predictions

    def _prepare_data_gaussian_py(
        self,
        MC_samples_mat: np.ndarray,
        x_explain_mat: np.ndarray,
        S: np.ndarray,
        mu: np.ndarray,
        cov_mat: np.ndarray,
    ) -> np.ndarray:
        """Performs conditional Gaussian imputation for all coalitions.

        Args:
            MC_samples_mat (np.ndarray): Monte Carlo samples of shape (n_samples, n_features).
            x_explain_mat (np.ndarray): Input instances to explain (n_explain, n_features).
            S (np.ndarray): Coalition indicator matrix (n_coalitions, n_features).
            mu (np.ndarray): Mean vector of the background distribution.
            cov_mat (np.ndarray): Covariance matrix of the background distribution.

        Returns:
            np.ndarray: Imputed data of shape (n_samples, n_explain * n_coalitions, n_features).
        """
        n_explain, n_features = x_explain_mat.shape
        n_MC_samples = MC_samples_mat.shape[0]
        n_coalitions = S.shape[0]

        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            S_now = S[S_ind]
            THRESHOLD_PRESENT = 0.5
            S_idx = np.where(S_now > THRESHOLD_PRESENT)[0]
            Sbar_idx = np.where(S_now < THRESHOLD_PRESENT)[0]

            x_S_star = x_explain_mat[:, S_idx]
            mu_S = mu[S_idx]
            mu_Sbar = mu[Sbar_idx]

            cov_SS = cov_mat[np.ix_(S_idx, S_idx)]
            cov_SSbar = cov_mat[np.ix_(S_idx, Sbar_idx)]
            cov_SbarS = cov_mat[np.ix_(Sbar_idx, S_idx)]
            cov_SbarSbar = cov_mat[np.ix_(Sbar_idx, Sbar_idx)]

            cov_SbarS_cov_SS_inv = cov_SbarS @ pinv(cov_SS)
            cond_cov_Sbar_given_S = cov_SbarSbar - cov_SbarS_cov_SS_inv @ cov_SSbar

            # Ensure symmetry
            cond_cov_Sbar_given_S = (cond_cov_Sbar_given_S + cond_cov_Sbar_given_S.T) / 2

            chol_cov = cholesky(cond_cov_Sbar_given_S)
            MC_samples_now = MC_samples_mat[:, Sbar_idx] @ chol_cov

            x_Sbar_mean = (cov_SbarS_cov_SS_inv @ (x_S_star - mu_S).T).T + mu_Sbar

            for i in range(n_explain):
                aux = np.zeros((n_MC_samples, n_features))
                aux[:, S_idx] = np.tile(x_S_star[i], (n_MC_samples, 1))
                aux[:, Sbar_idx] = MC_samples_now + x_Sbar_mean[i]
                result_cube[:, S_ind * n_explain + i, :] = aux

        return result_cube

    def calc_empty_prediction(self) -> float:
        """Estimates the model prediction when all features are missing.

        This is done by evaluating the model over the background data
        and averaging the predictions.

        Returns:
            float: The average prediction over the background data.
        """
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction
