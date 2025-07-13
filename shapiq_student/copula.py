"""Implementation of the Gaussian copula imputer.

This module implements a gaussian conditional imputer using the Gaussian copula transformation.
It enables imputing missing features in a feature coalition by modeling dependencies
using a Gaussian copula approach, and generating conditional samples for SHAP or other feature
attribution methods. (For conventional gaussian approach see gaussian.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import norm, rankdata
from shapiq.games.imputer.conditional_imputer import ConditionalImputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class GaussianCopulaImputer(ConditionalImputer):
    """Gaussian copula-based conditional imputer for missing feature imputation.

    This class implements conditional sampling based on a Gaussian copula transformation.
    It is used for imputing missing features in a way that respects the dependence structure
    of the observed data. This is particularly useful in model explanation tasks such as
    computing SHAP values.

    Args:
        model: Predictive model to be explained.
        data (np.ndarray): Background data for modeling the joint distribution.
        x (np.ndarray | None): Data points to explain. If None, will be inferred.
        sample_size (int): Number of Monte Carlo samples per coalition.
        normalize (bool): Whether to normalize using the empty prediction.
        categorical_features (list[int] | None): Indices of categorical features (unsupported).
        method (Literal["gaussCopula"]): Must be "gaussCopula".
        random_state (int | None): Random seed for reproducibility.

    Raises:
        ValueError: If an unsupported method is passed or if categorical features are provided.
    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        sample_size: int = 10,
        normalize: bool = True,
        categorical_features: list[int] | None = None,
        method: Literal["gaussCopula"] = "gaussCopula",
        random_state: int | None = None,
    ) -> None:
        """Initialize the GaussianImputer.

        This class performs conditional imputation based on a multivariate Gaussian
        distribution fitted to the background data. It is primarily used for imputing
        missing features (as defined by coalitions) during model explanation tasks.

        Parameters
        ----------
        model : callable
            The predictive model to explain. Must support `.predict()` with 2D input.
        data : np.ndarray
            Background dataset used to estimate the feature distribution (n_samples, n_features).
        x : np.ndarray or None, optional
            The instance(s) to explain. If not provided during initialization, can be set later.
        sample_size : int, default=10
            Number of Monte Carlo samples to generate per coalition.
        conditional_budget : int, default=128
            Reserved for future use or budget-aware strategies (not directly used here).
        conditional_threshold : float, default=0.05
            Reserved for future use, such as feature pruning based on contribution (not used here).
        normalize : bool, default=True
            Whether to normalize output values by the empty prediction value.
        categorical_features : list of int or None, optional
            Indices of categorical features. Gaussian imputation does not support categoricals.
        method : {'gaussConditional'}, default='gaussConditional'
            The imputation method name. Only 'gaussConditional' is supported in this class.
        random_state : int or None, optional
            Random seed for reproducibility.

        Raises:
        ------
        ValueError
            If any categorical features are provided or if the method is not 'gaussConditional'.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        if method not in {"gaussCopula"}:
            msg = "This constructor is for 'gaussCopula' imputers only."
            raise ValueError(msg)

        self.method = method

        # Set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussCopula":
            # Initialize background distribution via Gaussian copula transform
            self.init_background(data)

    def init_background(self, data: np.ndarray) -> GaussianCopulaImputer:
        """Initializes the background distribution using a Gaussian copula.

        Transforms the background data into Gaussian space using a rank-based transform.
        This transformation is used to compute the joint Gaussian distribution from
        which conditional samples can be drawn.

        Args:
            data (np.ndarray): Input data matrix of shape (n_samples, n_features).

        Returns:
            GaussianCopulaImputer: The instance itself.

        Raises:
            ValueError: If categorical features are provided.
        """
        if self._cat_features:
            msg = (
                "Gaussian Copula imputer does not support categorical features. "
                f"Found categorical feature indices: {self._cat_features}"
            )
            raise ValueError(msg)

        # Gaussianize training data
        data_gauss = np.apply_along_axis(self.gaussian_transform, axis=0, arr=data)

        self._copula_mu = np.zeros(data.shape[1])
        self._copula_cov = np.cov(data_gauss, rowvar=False)
        self._train_data = data.copy()

        return self

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Evaluates the value function for a set of feature coalitions.

        For each coalition, imputes missing features using conditional sampling under
        a Gaussian copula model, evaluates the model on imputed data, and averages
        the predictions.

        Args:
            coalitions (np.ndarray[bool]): Array of shape (n_coalitions, n_features)
                indicating which features are present (True) or missing (False) in each coalition.

        Returns:
            np.ndarray[float]: Array of shape (n_coalitions,) with averaged model predictions.
        """
        # Store x transformed the same way
        x_combined = np.vstack([self._x, self.data])
        x_gauss = np.apply_along_axis(
            self.gaussian_transform_separate, axis=0, arr=x_combined, n_y=1
        )
        n_y = self._x.shape[0]  # number of instances to explain
        self._x_gauss = x_gauss[:n_y]  # all instances to explain

        n_coalitions, n_features = coalitions.shape

        n_samples = self.sample_size  # or any desired MC sample size

        # Standard normal samples
        rng = np.random.default_rng(self.random_state)
        MC_samples = rng.standard_normal((n_samples, n_features))

        # Run conditional sampling using the Gaussian copula approach
        imputed_data = self._prepare_data_copula_py(
            MC_samples_mat=MC_samples,
            x_explain_gauss=self._x_gauss,  # already Gaussianized
            x_explain_original=self._x,  # original for back-transform
            x_train_mat=self._train_data,  # needed for copula rank transforms
            S=coalitions.astype(float),
            mu=self._copula_mu,
            cov_mat=self._copula_cov,
        )

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

    def _prepare_data_copula_py(
        self,
        MC_samples_mat: np.ndarray,
        x_explain_original: np.ndarray,
        x_explain_gauss: np.ndarray,
        x_train_mat: np.ndarray,
        S: np.ndarray,
        mu: np.ndarray,
        cov_mat: np.ndarray,
    ) -> np.ndarray:
        """Performs conditional sampling using a Gaussian copula for each coalition.

        Samples missing features conditional on observed ones by leveraging the
        Gaussian copula transformation and conditional multivariate normal distributions.

        Args:
            MC_samples_mat (np.ndarray): Standard normal samples (n_samples, n_features).
            x_explain_original (np.ndarray): Original values of the data points to explain.
            x_explain_gauss (np.ndarray): Gaussianized values of the data points to explain.
            x_train_mat (np.ndarray): Original training data for inverse transformation.
            S (np.ndarray): Coalition matrix indicating known features.
            mu (np.ndarray): Mean of the Gaussian copula.
            cov_mat (np.ndarray): Covariance matrix of the Gaussian copula.

        Returns:
            np.ndarray: Imputed data of shape (n_samples, n_coalitions * n_points, n_features).
        """
        n_explain, n_features = x_explain_gauss.shape
        n_MC_samples = MC_samples_mat.shape[0]
        n_coalitions = S.shape[0]

        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            S_now = S[S_ind]
            THRESHOLD_PRESENT = 0.5
            S_idx = np.where(S_now > THRESHOLD_PRESENT)[0]
            Sbar_idx = np.where(S_now < THRESHOLD_PRESENT)[0]

            x_S_star = x_explain_original[:, S_idx]
            x_S_star_gauss = x_explain_gauss[:, S_idx]

            mu_S = mu[S_idx]
            mu_Sbar = mu[Sbar_idx]

            cov_SS = cov_mat[np.ix_(S_idx, S_idx)]
            cov_SSbar = cov_mat[np.ix_(S_idx, Sbar_idx)]
            cov_SbarS = cov_mat[np.ix_(Sbar_idx, S_idx)]
            cov_SbarSbar = cov_mat[np.ix_(Sbar_idx, Sbar_idx)]

            cov_SbarS_cov_SS_inv = cov_SbarS @ np.linalg.pinv(cov_SS)
            cond_cov_Sbar_given_S = cov_SbarSbar - cov_SbarS_cov_SS_inv @ cov_SSbar
            cond_cov_Sbar_given_S = (cond_cov_Sbar_given_S + cond_cov_Sbar_given_S.T) / 2

            # Add regularization to avoid singular matrix
            epsilon = 1e-8
            cond_cov_Sbar_given_S += np.eye(cond_cov_Sbar_given_S.shape[0]) * epsilon

            # Add jitter to make covariance matrix positive definite
            eps = 1e-3  # smoother conditioning
            chol_cov = np.linalg.cholesky(cond_cov_Sbar_given_S + eps * np.eye(len(Sbar_idx)))

            MC_samples_now = MC_samples_mat[:, Sbar_idx] @ chol_cov

            x_Sbar_gaussian_mean = (cov_SbarS_cov_SS_inv @ (x_S_star_gauss - mu_S).T).T + mu_Sbar

            for i in range(n_explain):
                aux = np.zeros((n_MC_samples, n_features))
                aux[:, S_idx] = np.tile(x_S_star[i], (n_MC_samples, 1))
                x_sbar_i = MC_samples_now + x_Sbar_gaussian_mean[i]
                x_sbar_i_transformed = self.inv_gaussian_transform(
                    x_sbar_i, x_train_mat[:, Sbar_idx]
                )
                aux[:, Sbar_idx] = x_sbar_i_transformed
                result_cube[:, S_ind * n_explain + i, :] = aux

        return result_cube

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction

    def quantile_type7(self, x: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Computes quantiles using R's type-7 interpolation method.

        This method is used to back-transform Gaussian samples to the original scale.

        Args:
            x (np.ndarray): Sample from the original distribution.
            probs (np.ndarray): Probabilities in [0, 1] to compute quantiles for.

        Returns:
            np.ndarray: Quantile values corresponding to the given probabilities.

        Raises:
            ValueError: If input array is empty.
        """
        n = len(x)
        if n == 0:
            error_msg = "Cannot compute quantile with empty array."
            raise ValueError(error_msg)
        if n == 1:
            return np.full_like(probs, x[0])
        x_sorted = np.sort(x)
        index = 1 + (n - 1) * probs
        lo = np.floor(index).astype(int) - 1
        hi = np.ceil(index).astype(int) - 1
        h = index - np.floor(index)

        qs = x_sorted[lo] * (1 - h) + x_sorted[np.minimum(hi, n - 1)] * h
        return qs

    def inv_gaussian_transform(self, z: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """Inverse Gaussian copula transformation to original feature space.

        Converts Gaussianized data back to the original space using empirical quantiles.

        Args:
            z (np.ndarray): Gaussian samples to transform.
            x_train (np.ndarray): Training data for empirical distribution.

        Returns:
            np.ndarray: Transformed data in the original feature space.
        """
        u = norm.cdf(z)
        transformed = np.empty_like(z)
        for j in range(z.shape[1]):
            transformed[:, j] = self.quantile_type7(x_train[:, j], u[:, j])
        return transformed

    def gaussian_transform(self, x: np.ndarray) -> np.ndarray:
        """Applies Gaussian copula transformation to a feature vector.

        Converts a numeric feature to standard normal marginals using
        rank-based transformation.

        Args:
            x (np.ndarray): Input 1D array (feature values).

        Returns:
            np.ndarray: Transformed array with standard normal marginals.
        """
        ranks = rankdata(x, method="average")  # rank(x)
        u = ranks / (len(x) + 1)  # rank / (n + 1)
        z = norm.ppf(u)  # qnorm(u)
        return z

    def gaussian_transform_separate(self, yx: np.ndarray, n_y: int) -> np.ndarray:
        """Transforms new data to standard normal space using rank information.

        Used to convert a new data point to Gaussian copula space, while referencing
        the rank distribution from a larger sample.

        Args:
            yx (np.ndarray): Combined array of new values and reference sample.
            n_y (int): Number of elements in yx that belong to the new data.

        Returns:
            np.ndarray: Transformed values for the new data portion.

        Raises:
            ValueError: If n_y is not less than the total array length.
        """
        if n_y >= len(yx):
            error_msg = "n_y should be less than length of yx"
            raise ValueError(error_msg)

        ind = np.arange(n_y)
        x = yx[n_y:]

        ranks_yx = rankdata(yx, method="average")
        tmp = ranks_yx[ind]

        rank_tmp = rankdata(tmp, method="average")
        tmp = tmp - rank_tmp + 0.5

        u_y = tmp / (len(x) + 1)
        z_y = norm.ppf(u_y)
        return z_y
