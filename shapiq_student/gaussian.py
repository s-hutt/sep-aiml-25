"""Implementierung des Gaussian Conditional Imputers mit multivariaten Normalverteilungen.

Dieses Modul definiert die Klasse GaussianImputer, die conditional Imputation
auf Basis der multivariaten Gaussian-Annahme durchführt. Sie wird in Feature Attribution Methoden
wie SHAP verwendet, wo das conditional Imputieren fehlender features
zur Schätzung von Shapley Werten bei Teilmengen von features (coalitions) nötig ist.
(Für den Gaussian Copula Ansatz siehe copula.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.linalg import cholesky, pinv
from shapiq.games.imputer.conditional_imputer import ConditionalImputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class GaussianImputer(ConditionalImputer):
    """Conditional Imputer basierend auf multivariaten Gaussian-Verteilungen.

    Dieser Imputer nimmt eine multivariate Normalverteilung über die features an und
    nutzt conditional Gaussian Sampling, um fehlende features abhängig davon
    aufzufüllen, welche features vorhanden sind (coalition Sets).

    Args:
        model: Das zu erklärende Predictive Modell.
        data (np.ndarray): Hintergrunddaten zur Schätzung der Gaussian-Verteilung.
        x (np.ndarray | None): Zu erklärende Instanzen (optional bei Init).
        sample_size (int): Anzahl der Monte-Carlo Samples pro coalition. Default ist 10.
        conditional_budget (int): Wird hier nicht direkt verwendet, dient der Kompatibilität.
        conditional_threshold (float): Wird hier nicht direkt verwendet, dient der Kompatibilität.
        normalize (bool): Ob die Prediction anhand der leeren Prediction normalisiert werden.
        categorical_features (list[int] | None): Indizes kategorialer features. Nicht unterstützt.
        method (Literal["gaussConditional"]): Methodenkennung für diesen Imputer.
        random_state (int | None): Optionaler Random Seed zur Reproduzierbarkeit.
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
        """Initialisiert den GaussianImputer.

        Args:
            model: Das zu erklärende Predictive Modell. Erwartet eine `predict`-Methode.
            data (np.ndarray): Hintergrunddaten zur Schätzung der Gaussian-Verteilung.
            x (np.ndarray | None, optional): Zu erklärende Dateninstanzen. Defaults ist None.
            sample_size (int, optional): Anzahl der Monte-Carlo Samples pro coalition. Defaults ist 10.
            conditional_budget (int, optional): Budget-Parameter, dient der Kompatibilität. Defaults ist 128.
            conditional_threshold (float, optional): Schwellenwert-Parameter, dient der Kompatibilität. Defaults ist 0.05.
            normalize (bool, optional): Ob die Prediction anhand der leeren Prediction normalisiert werden soll. Defaults ist True.
            categorical_features (list[int] | None, optional): Indizes kategorialer features. Nicht unterstützt. Defaults ist None.
            method (Literal["gaussConditional"], optional): Methodenkennung für diesen Imputer. Muss "gaussConditional" sein. Defaults ist "gaussConditional".
            random_state (int | None, optional): Random Seed zur Reproduzierbarkeit. Defaults ist None.

        Raises:
            ValueError: Wenn die Methode nicht "gaussConditional" ist.
            ValueError: Wenn kategoriale features übergeben werden (nicht unterstützt).
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
            msg = "Dieser Konstructor ist nur für gaussianConditional Imputer ausschließlich."
            raise ValueError(msg)

        self.method = method
        self.conditional_budget = conditional_budget
        self.conditional_threshold = conditional_threshold

        # Leeren Wert und Normalisierung setzen
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussConditional":
            self.init_background(data)

    def init_background(self, data: np.ndarray) -> GaussianImputer:
        """Initialisiert die Hintergrund-Gaussian-Verteilung mit den Eingabedaten.

        Berechnet den empirischen Mittelwert und die (regularisierte) Kovarianzmatrix
        aus den Hintergrunddaten.

        Args:
            data (np.ndarray): Hintergrunddaten (n_samples, n_features).

        Returns:
            Self (GaussianImputer): Die gefittete Imputer-Instanz.
        """
        if self._cat_features:
            msg = (
                "Gaussian Imputer unterstützt keine kategorialen features."
                f"Gefundene Indizes kategorialer features: {self._cat_features}"
            )
            raise ValueError(msg)

        # Berechne den Mittelwert-Vektor (mu) und die Kovarianzmatrix (cov_mat)
        self._mu = np.mean(data, axis=0)
        cov_mat = np.cov(data, rowvar=False)

        # Sicherstellen, dass die Kovarianzmatrix positiv definit ist (ähnlich wie R's nearPD)
        min_eigenvalue = 1e-6
        eigvals = np.linalg.eigvalsh(cov_mat)
        if np.any(eigvals <= min_eigenvalue):
            # Kovarianzmatrix leicht regularisieren (Diagonal loading)
            cov_mat += np.eye(cov_mat.shape[0]) * (min_eigenvalue - np.min(eigvals) + 1e-6)

        self._cov_mat = cov_mat

        return self

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Berechnet die Modell Predictions für jede coalition mittels conditional Gaussian Sampling.

        Args:
            coalitions (np.ndarray): Boolesches Array (n_subsets, n_features),
                                     wobei True bedeutet, dass das feature vorhanden ist.

        Returns:
            np.ndarray: Predicted Werte für jede coalition (n_subsets,).
        """
        n_coalitions, n_features = coalitions.shape

        mu = self._mu
        cov = self._cov_mat
        n_samples = self.sample_size  # Oder beliebige gewünschte MC-Samplegröße

        # Standard normal samples
        rng = np.random.default_rng()
        MC_samples = rng.standard_normal((n_samples, n_features))

        x_explain = self._x  # shape (1, n_features)

        # conditional sampling durchführen
        imputed_data = self._prepare_data_gaussian_py(
            MC_samples_mat=MC_samples,
            x_explain_mat=x_explain,
            S=coalitions.astype(float),  # shape (n_coalitions, n_features)
            mu=mu,
            cov_mat=cov,
        )  # shape: (n_samples, n_coalitions, n_features)

        # Flatten für Prediction
        flat_input = imputed_data.reshape(-1, n_features)
        predictions = self.predict(flat_input)

        # Reshape und average predictions per coalition
        predictions = predictions.reshape(n_samples, n_coalitions)
        avg_predictions = predictions.mean(axis=0)

        # Leere coalitions behandeln (alle features False)
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
        """Führt conditional Gaussian Imputation für alle coalitions durch.

        Args:
            MC_samples_mat (np.ndarray): Monte-Carlo-Samples der Shape (n_samples, n_features).
            x_explain_mat (np.ndarray): Eingabedaten zum Erklären (n_explain, n_features).
            S (np.ndarray): Indikator-Matrix für coalitions (n_coalitions, n_features).
            mu (np.ndarray): Mittelwertvektor der Hintergrundverteilung.
            cov_mat (np.ndarray): Kovarianzmatrix der Hintergrundverteilung.

        Returns:
            np.ndarray: Imputierte Daten der Shape (n_samples, n_explain * n_coalitions, n_features).
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

            # Symmetrie sicherstellen
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
        """Schätzt die Modellvorhersage, wenn alle features fehlen.

        Dies geschieht durch Auswertung des Modells über die Hintergrunddaten
        und Mittelung der Prediction.

        Returns:
            float: Die durchschnittliche Prediction über die Hintergrunddaten.
        """
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction
