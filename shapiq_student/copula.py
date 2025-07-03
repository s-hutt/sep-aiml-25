import numpy as np

from typing import Literal
from scipy.stats import norm, rankdata

from shapiq.games.imputer.base import Imputer

class GaussianCopulaImputer(Imputer):
    def __init__(
        self,
        model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        sample_size: int = 10,
        normalize: bool = True,
        categorical_features: list[int] | None = None,
        method: Literal["gaussCopula"] = "gaussCopula",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        if method not in {"gaussCopula"}:
            raise ValueError("This constructor is for 'gaussCopula' imputers only.")
        self.method = method

        # Check that no categorical features are included
        if self._cat_features:
            raise ValueError(
                f"Gaussian Copula imputer does not support categorical features. "
                f"Found categorical feature indices: {self._cat_features}"
            )

        # Set empty value and normalization
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

        # Initialize background distribution via Gaussian copula transform
        self.init_background_gauss_copula(data)

    def init_background_gauss_copula(self, data: np.ndarray) -> "GaussianCopulaImputer":
        """
        Initializes the background distribution for Copula-based imputation.
        Transforms data to Gaussian space via rank-based transform.
        """
        if self._cat_features:
            raise ValueError(
                f"Gaussian Copula imputer does not support categorical features. "
                f"Found categorical feature indices: {self._cat_features}"
            )

        # Gaussianize training data
        data_gauss = np.apply_along_axis(self.gaussian_transform, axis=0, arr=data)

        self._copula_mu = np.zeros(data.shape[1])
        self._copula_cov = np.cov(data_gauss, rowvar=False)
        self._train_data = data.copy()

        # Store x transformed the same way
        x_combined = np.vstack([self._x[np.newaxis, :], data])
        x_gauss = np.apply_along_axis(self.gaussian_transform_separate, axis=0, arr=x_combined, n_y=1)
        self._x_gauss = x_gauss[0]  # Only the first row (explained instance)

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

        mu = self._copula_mu
        cov = self._copula_cov
        n_samples = self.sample_size  # or any desired MC sample size

        # Standard normal samples
        rng = np.random.default_rng(self.random_state)
        MC_samples = rng.standard_normal((n_samples, n_features))

        # Expand input x for batch
        x_explain = self._x # shape (1, n_features)

        # Run conditional sampling using the Gaussian copula approach
        imputed_data = self._prepare_data_copula_py(
            MC_samples_mat=MC_samples,
            x_explain_gauss=self._x_gauss[np.newaxis, :],  # already Gaussianized
            x_explain_original=self._x[np.newaxis, :],  # original for back-transform
            x_train_mat=self._train_data,  # needed for copula rank transforms
            S=coalitions.astype(float),
            mu=self._copula_mu,
            cov_mat=self._copula_cov
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
            cov_mat: np.ndarray
    ) -> np.ndarray:
        n_explain, n_features = x_explain_gauss.shape
        n_MC_samples = MC_samples_mat.shape[0]
        n_coalitions = S.shape[0]

        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            S_now = S[S_ind]
            S_idx = np.where(S_now > 0.5)[0]
            Sbar_idx = np.where(S_now < 0.5)[0]

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

            # Add jitter to make covariance matrix positive definite
            eps = 1e-6  # small value, tune as needed
            chol_cov = np.linalg.cholesky(cond_cov_Sbar_given_S + eps * np.eye(len(Sbar_idx)))

            MC_samples_now = MC_samples_mat[:, Sbar_idx] @ chol_cov

            x_Sbar_gaussian_mean = (cov_SbarS_cov_SS_inv @ (x_S_star_gauss - mu_S).T).T + mu_Sbar

            for i in range(n_explain):
                aux = np.zeros((n_MC_samples, n_features))
                aux[:, S_idx] = np.tile(x_S_star[i], (n_MC_samples, 1))
                x_sbar_i = MC_samples_now + x_Sbar_gaussian_mean[i]
                x_sbar_i_transformed = self.inv_gaussian_transform(x_sbar_i, x_train_mat[:, Sbar_idx])
                aux[:, Sbar_idx] = x_sbar_i_transformed
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

    def quantile_type7(self, x: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Replicates R's type 7 quantile interpolation."""
        n = len(x)
        if n == 0:
            raise ValueError("Cannot compute quantile with empty array.")
        elif n == 1:
            return np.full_like(probs, x[0])
        x_sorted = np.sort(x)
        index = 1 + (n - 1) * probs
        lo = np.floor(index).astype(int) - 1
        hi = np.ceil(index).astype(int) - 1
        h = index - np.floor(index)

        qs = x_sorted[lo] * (1 - h) + x_sorted[np.minimum(hi, n - 1)] * h
        return qs

    def inv_gaussian_transform(self, z: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        u = norm.cdf(z)
        transformed = np.empty_like(z)
        for j in range(z.shape[1]):
            transformed[:, j] = self.quantile_type7(x_train[:, j], u[:, j])
        return transformed

    def gaussian_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms a sample to a standardized normal distribution.
        Equivalent to the R version using rank-based transform.

        Args:
            x: Numeric vector (1D array).

        Returns:
            Transformed vector with standard normal marginals.
        """
        ranks = rankdata(x, method='average')  # rank(x)
        u = ranks / (len(x) + 1)  # rank / (n + 1)
        z = norm.ppf(u)  # qnorm(u)
        return z

    def gaussian_transform_separate(self, yx: np.ndarray, n_y: int) -> np.ndarray:
        """
        Transforms new data to standardized normal (dimension 1) based on other data transformations.

        Args:
            yx: Numeric vector where first n_y items belong to Gaussian data,
                and the rest belong to original data.
            n_y: Number of elements in yx belonging to Gaussian data.

        Returns:
            Vector of back-transformed Gaussian data.
        """
        if n_y >= len(yx):
            raise ValueError("n_y should be less than length of yx")

        ind = np.arange(n_y)
        x = yx[n_y:]  # original data part

        # tmp = rank(yx)[ind]
        ranks_yx = rankdata(yx, method='average')
        tmp = ranks_yx[ind]

        # tmp = tmp - rank(tmp) + 0.5
        rank_tmp = rankdata(tmp, method='average')
        tmp = tmp - rank_tmp + 0.5

        u_y = tmp / (len(x) + 1)
        z_y = norm.ppf(u_y)
        return z_y