import warnings

import numpy as np
from numpy.linalg import inv, cholesky

from typing import Literal
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
            method: Literal["gaussConditional"] = "gaussConditional",
            random_state: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state
        )
        if method not in {"gaussianConditional", "gaussCopula"}:
            raise ValueError("Currently only 'gaussConditional' and 'gaussCopula' imputers are implemented.")
        self.method = method
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold

        # set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussConditional":
            self.init_background(data)

    def init_background(self, data: np.ndarray) -> "GaussianImputer":

        if self._cat_features:
            raise ValueError(
                f"Gaussian imputer does not support categorical features. "
                f"Found categorical feature indices: {self._cat_features}"
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

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """
        Computes the value function using multivariate Gaussian conditional sampling.

    Args:
        coalitions: Boolean array (n_subsets, n_features), True for present features.

    Returns:
        np.ndarray of shape (n_subsets,), model predictions per coalition.
    """
        n_coalitions, n_features = coalitions.shape

        mu = self._mu
        cov = self._cov_mat
        n_samples = self.sample_size  # or any desired MC sample size

        # Standard normal samples
        rng = np.random.default_rng()
        MC_samples = rng.standard_normal((n_samples, n_features))

        # Expand input x for batch
        x_explain = self._x[np.newaxis, :]  # shape (1, n_features)

        # Run conditional sampling
        imputed_data = self._prepare_data_gaussian_py(
            MC_samples_mat=MC_samples,
            x_explain_mat=x_explain,
            S=coalitions.astype(float),  # shape (n_coalitions, n_features)
            mu=mu,
            cov_mat=cov
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

    def _prepare_data_gaussian_py(self, MC_samples_mat, x_explain_mat, S, mu, cov_mat):
        n_explain, n_features = x_explain_mat.shape
        n_MC_samples = MC_samples_mat.shape[0]
        n_coalitions = S.shape[0]

        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            S_now = S[S_ind]
            S_idx = np.where(S_now > 0.5)[0]
            Sbar_idx = np.where(S_now < 0.5)[0]

            x_S_star = x_explain_mat[:, S_idx]
            mu_S = mu[S_idx]
            mu_Sbar = mu[Sbar_idx]

            cov_SS = cov_mat[np.ix_(S_idx, S_idx)]
            cov_SSbar = cov_mat[np.ix_(S_idx, Sbar_idx)]
            cov_SbarS = cov_mat[np.ix_(Sbar_idx, S_idx)]
            cov_SbarSbar = cov_mat[np.ix_(Sbar_idx, Sbar_idx)]

            cov_SbarS_cov_SS_inv = cov_SbarS @ inv(cov_SS)
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
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction.
        """
        # TODO: perhaps should be self.conditional_data instead of self.data
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction