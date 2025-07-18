"""Implementierung des Gaussian Copula Imputers.

Dieses Modul implementiert einen gaußschen konditionalen Imputer unter Verwendung der Gaussian Copula Transformation.
Es ermöglicht das Imputieren fehlender features in einer feature coalition durch Modellierung von Abhängigkeiten
mit einem Gaussian Copula Ansatz und das Erzeugen conditional samples für SHAP oder andere feature
attribution methods. (Für den konventionellen gaußschen Ansatz siehe gaussian.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import norm, rankdata
from shapiq.games.imputer.conditional_imputer import ConditionalImputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class GaussianCopulaImputer(ConditionalImputer):
    """Gaussian Copula-basierter konditionaler Imputer zur fehlender features.

    Diese Klasse implementiert conditional Sampling basierend auf einer Gaussian Copula Transformation.
    Sie wird verwendet, um fehlende features so zu imputieren, dass die dependence struktur
    der beobachteten Daten berücksichtigt wird. Dies ist besonders nützlich bei Modell-Erklärungsaufgaben wie
    der Berechnung von SHAP-Werten.

    Args:
        model: Zu erklärendes Predictive modell.
        data (np.ndarray): Hintergrunddaten zur Modellierung der multivariaten Verteilung.
        x (np.ndarray | None): Zu erklärende Datenpunkte. Falls None, werden sie inferiert.
        sample_size (int): Anzahl der Monte-Carlo-samples pro coalition.
        normalize (bool): Ob mit der leeren Vorhersage normalisiert werden soll.
        categorical_features (list[int] | None): Indizes kategorialer features (nicht unterstützt).
        method (Literal["gaussCopula"]): Muss "gaussCopula" sein.
        random_state (int | None): Random seed für Reproduzierbarkeit.

    Raises:
        ValueError: Wenn eine nicht unterstützte method übergeben wird oder kategoriale features vorhanden sind.
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
        """Initialisierung des GaussianImputers.

        Diese Klasse führt conditional Imputation basierend auf einer multivariaten
        Gaussian-Verteilung durch, die auf den Hintergrunddaten gefittet wird. Sie wird primär verwendet,
        um fehlende features (wie durch coalitions definiert) bei Modell-Erklärungsaufgaben zu imputieren.

        Parameter
        ----------
        model : callable
            Das zu erklärende predictive model. Muss `.predict()` mit 2D-Input unterstützen.
        data : np.ndarray
            Hintergrunddaten zur Schätzung der feature-Verteilung (n_samples, n_features).
        x : np.ndarray oder None, optional
            Die zu erklärenden Instanzen. Falls nicht bei der Initialisierung angegeben, kann später gesetzt werden.
        sample_size : int, default=10
            Anzahl der Monte-Carlo-samples, die pro coalition generiert werden.
        conditional_budget : int, default=128
            Reserviert für zukünftige Nutzung oder budget-basierte Strategien (aktuell nicht verwendet).
        conditional_threshold : float, default=0.05
            Reserviert für zukünftige Nutzung, z.B. feature pruning basierend auf Contribution (aktuell nicht verwendet).
        normalize : bool, default=True
            Ob die Ausgaben mit der leeren Vorhersage normalisiert werden sollen.
        categorical_features : list[int] oder None, optional
            Indizes kategorialer features. Gaussian-Imputation unterstützt keine Kategorischen.
        method : {'gaussConditional'}, default='gaussConditional'
            Name der Imputationsmethode. Nur 'gaussConditional' wird in dieser Klasse unterstützt.
        random_state : int oder None, optional
            Random seed zur Reproduzierbarkeit.

        Raises:
        -------
        ValueError
            Wenn kategoriale features angegeben werden oder die Methode ungleich 'gaussConditional' ist.
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
            msg = "Dieser Konstruktor ist ausschließlich für 'gaussCopula'-Imputer."
            raise ValueError(msg)

        self.method = method

        # Setze leeren Wert und Normalisierung
        self.empty_prediction: float = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction
        if method == "gaussCopula":
            # Initialisiere Hintergrundverteilung über Gaussian Copula Transformation
            self.init_background(data)

    def init_background(self, data: np.ndarray) -> GaussianCopulaImputer:
        """Initialisiert die Hintergrundverteilung mithilfe einer Gaussian Copula.

        Die Hintergrunddaten werden per rank-basierter Transformation in den Gaussian-Raum überführt.
        Diese Transformation dient dazu, die gemeinsame Gaussian-Verteilung zu bestimmen,
        aus der anschließend konditionale Samples gezogen werden können.

        Args:
            data (np.ndarray): Input-Datenmatrix der Form (n_samples, n_features).

        Returns:
            GaussianCopulaImputer: Die Instanz selbst.

        Raises:
            ValueError: Wenn kategoriale features angegeben werden.
        """
        if self._cat_features:
            msg = (
                "Gaussian Copula Imputer unterstützt keine kategorialen features."
                f"Gefundene Indizes kategorialer features: {self._cat_features}"
            )
            raise ValueError(msg)

        # Training data gaussianisieren
        data_gauss = np.apply_along_axis(self.gaussian_transform, axis=0, arr=data)

        self._copula_mu = np.zeros(data.shape[1])
        self._copula_cov = np.cov(data_gauss, rowvar=False)
        self._train_data = data.copy()

        return self

    def value_function(self, coalitions: np.ndarray[bool]) -> np.ndarray[float]:
        """Evaluiert die value function für eine Menge von feature coalitions.

        Für jede coalition werden fehlende features durch conditional sampling unter
        einem Gaussian Copula Modell imputiert, das Modell auf den imputierten Daten evaluiert
        und die averages Prediction.

        Args:
            coalitions (np.ndarray[bool]): Array of shape (n_coalitions, n_features),
                das angibt, welche features in jeder coalition vorhanden (True) oder fehlend (False) sind.

        Returns:
            np.ndarray[float]: Array of shape (n_coalitions,) mit averaged predictions.
        """
        # Speichere x, transformiert auf dieselbe Weise
        x_combined = np.vstack([self._x, self.data])
        x_gauss = np.apply_along_axis(
            self.gaussian_transform_separate, axis=0, arr=x_combined, n_y=1
        )
        n_y = self._x.shape[0]  # Anzahl der zu erklärenden Instanzen
        self._x_gauss = x_gauss[:n_y]  # Alle zu erklärenden Instanzen

        n_coalitions, n_features = coalitions.shape

        n_samples = self.sample_size  # Oder beliebige gewünschte MC-Samplegröße

        # Standard normal samples
        rng = np.random.default_rng(self.random_state)
        MC_samples = rng.standard_normal((n_samples, n_features))

        # Führe conditional sampling mit dem Gaussian Copula Ansatz durch
        imputed_data = self._prepare_data_copula_py(
            MC_samples_mat=MC_samples,
            x_explain_gauss=self._x_gauss,  # Bereits gaussianized
            x_explain_original=self._x,  # Original für Back-Transform
            x_train_mat=self._train_data,  # Wird für Copula Rank Transforms benötigt
            S=coalitions.astype(float),
            mu=self._copula_mu,
            cov_mat=self._copula_cov,
        )

        # Flatten für die Prediction
        flat_input = imputed_data.reshape(-1, n_features)
        predictions = self.predict(flat_input)

        # Reshape und average predictions per coalition
        predictions = predictions.reshape(n_samples, n_coalitions)
        avg_predictions = predictions.mean(axis=0)

        # Leere coalitions behandeln (alle features False)
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
        """Führt conditional sampling mit einer Gaussian Copula für jede coalition durch.

        Fehltende features werden konditional auf beobachtete mittels
        Gaussian Copula Transformation und konditionalen multivariaten Normalverteilungen
        gesampelt.

        Args:
            MC_samples_mat (np.ndarray): Standard-normalverteilte Samples (n_samples, n_features).
            x_explain_original (np.ndarray): Originalwerte der zu erklärenden Datenpunkte.
            x_explain_gauss (np.ndarray): Gaussianisierte Werte der zu erklärenden Datenpunkte.
            x_train_mat (np.ndarray): Original-Trainingsdaten für die inverse Transformation.
            S (np.ndarray): Coalition-Matrix, die bekannte features angibt.
            mu (np.ndarray): Mittelwert der Gaussian Copula.
            cov_mat (np.ndarray): Kovarianzmatrix der Gaussian Copula.

        Returns:
            np.ndarray: Imputierte Daten von shape (n_samples, n_coalitions * n_points, n_features).
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

            # Füge Regularisierung hinzu, um singuläre Matrix zu vermeiden
            epsilon = 1e-8
            cond_cov_Sbar_given_S += np.eye(cond_cov_Sbar_given_S.shape[0]) * epsilon

            # Füge Jitter hinzu, um die Kovarianzmatrix positiv definit zu machen
            eps = 1e-3
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
        """Führt das Modell auf leeren Datenpunkten (alle features fehlen) aus, um die leere Prediction zu erhalten.

        Returns:
            Die leere Prediction.
        """
        empty_predictions = self.predict(self.data)
        empty_prediction = float(np.mean(empty_predictions))
        return empty_prediction

    def quantile_type7(self, x: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Berechnet Quantile mit R's Type-7 Interpolationsmethode.

        Diese Methode wird verwendet, um Gaussian samples zurück auf die Originalskala zu transformieren.

        Args:
            x (np.ndarray): Sample aus der Originalverteilung.
            probs (np.ndarray): Wahrscheinlichkeiten in [0, 1], für die Quantile berechnet werden.

        Returns:
            np.ndarray: Quantilwerte entsprechend den angegebenen Wahrscheinlichkeiten.

        Raises:
            ValueError: Falls das Input-Array leer ist.
        """
        n = len(x)
        if n == 0:
            error_msg = "Quantil kann mit leerem Array nicht berechnet werden."
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
        """Inverse Gaussian Copula Transformation zurück in den originalen feature space.

        Wandelt gaussianisierte Daten mittels empirischer Quantile zurück in den Originalraum.

        Args:
            z (np.ndarray): Gaussian samples zum Transformieren.
            x_train (np.ndarray): Trainingsdaten für die empirische Verteilung.

        Returns:
            np.ndarray: Transformierte Daten im originalen feature space.
        """
        u = norm.cdf(z)
        transformed = np.empty_like(z)
        for j in range(z.shape[1]):
            transformed[:, j] = self.quantile_type7(x_train[:, j], u[:, j])
        return transformed

    def gaussian_transform(self, x: np.ndarray) -> np.ndarray:
        """Wendet die Gaussian Copula Transformation auf einen feature Vektor an.

        Wandelt ein numerisches feature mittels rank-basierter Transformation
        in standard normal verteilte Marginals um.

        Args:
            x (np.ndarray): Input 1D Array (feature Werte).

        Returns:
            np.ndarray: Transformiertes Array mit standard normal verteilten Marginals.
        """
        ranks = rankdata(x, method="average")
        u = ranks / (len(x) + 1)
        z = norm.ppf(u)
        return z

    def gaussian_transform_separate(self, yx: np.ndarray, n_y: int) -> np.ndarray:
        """Transformiert neue Daten in den standard normal Raum mittels Ranginformationen.

        Wird verwendet, um einen neuen Datenpunkt in den Gaussian Copula Raum zu überführen,
        wobei die Rangverteilung einer größeren sampling als Referenz dient.

        Args:
            yx (np.ndarray): Kombiniertes Array aus neuen Werten und Referenzsampling.
            n_y (int): Anzahl der Elemente in yx, die zu den neuen Daten gehören.

        Returns:
            np.ndarray: Transformierte Werte für den neuen Datenanteil.

        Raises:
            ValueError: Falls n_y nicht kleiner als die Gesamtlänge des Arrays ist.
        """
        if n_y >= len(yx):
            error_msg = "n_y sollte kleiner als die Länge von yx sein"
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
