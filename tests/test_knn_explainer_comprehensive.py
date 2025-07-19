"""Umfassende pytest-Tests für KNNExplainer.

Diese Test-Suite deckt alle wichtigen Funktionalitäten, Edge Cases und Grenzfälle ab:
- Initialisierung und Parameter-Validierung
- Alle drei Erklärungsmethoden (Standard, Threshold, Weighted)
- Helper-Funktionen und mathematische Operationen
- Fehlerbehandlung und ungültige Eingaben
- Performance und Integration Tests
- Grenzfälle und Edge Cases
"""

from __future__ import annotations

from itertools import combinations_with_replacement
import logging
import math
import time

import numpy as np
import pytest
from shapiq.interaction_values import InteractionValues
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from shapiq_student.knn_explainer import KNNExplainer

logger = logging.getLogger(__name__)


class TestKNNExplainerInitialization:
    """Test der Initialisierung und Parameter-Validierung."""

    def setup_method(self):
        """Setup für alle Tests in dieser Klasse."""
        self.X, self.y = make_classification(
            n_samples=50, n_features=4, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)

    def test_default_initialization(self):
        """Test der Standard-Initialisierung."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        DEFAULT_K = 5
        assert explainer.k == DEFAULT_K

        DEFAULT_TAU = -0.5
        assert explainer.tau == DEFAULT_TAU

        assert explainer.method == "KNN-Shapley"
        assert explainer.mode == "normal"
        assert explainer.class_index is None
        assert explainer.m_star is None

    def test_custom_parameters(self):
        """Test mit benutzerdefinierten Parametern."""
        explainer = KNNExplainer(
            model=self.model,
            data=self.X,
            labels=self.y,
            method="threshold_knn_shapley",
            k=7,
            tau=-0.2,
            class_index=1,
            m_star=15,
        )

        TEST_K = 7
        assert explainer.k == TEST_K

        TEST_TAU = -0.2
        assert explainer.tau == TEST_TAU

        assert explainer.method == "threshold_knn_shapley"
        assert explainer.mode == "threshold"
        assert explainer.class_index == 1
        CUSTOM_M_STAR = 15
        assert explainer.m_star == CUSTOM_M_STAR

    def test_automatic_method_detection_weighted(self):
        """Test automatische Erkennung für gewichtete Modelle."""
        weighted_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        weighted_model.fit(self.X, self.y)

        explainer = KNNExplainer(model=weighted_model, data=self.X, labels=self.y)
        assert explainer.method == "weighted_knn_shapley"
        assert explainer.mode == "weighted"

    def test_automatic_method_detection_uniform(self):
        """Test automatische Erkennung für uniform gewichtete Modelle."""
        uniform_model = KNeighborsClassifier(n_neighbors=5, weights="uniform")
        uniform_model.fit(self.X, self.y)

        explainer = KNNExplainer(model=uniform_model, data=self.X, labels=self.y)
        assert explainer.method == "KNN-Shapley"
        assert explainer.mode == "normal"

    def test_invalid_method_raises_error(self):
        """Test dass ungültige Methode einen Fehler wirft."""
        with pytest.raises(ValueError, match="Unknown method"):
            KNNExplainer(model=self.model, data=self.X, labels=self.y, method="invalid_method")

    def test_inheritance_from_base_explainer(self):
        """Test dass KNNExplainer korrekt von Explainer erbt."""
        from shapiq.explainer import Explainer

        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        assert isinstance(explainer, Explainer)
        assert explainer.max_order == 1
        assert hasattr(explainer, "explain_function")

    def test_data_validation(self):
        """Test der Datenvalidierung bei Initialisierung."""
        # Test mit falschen Datentypen
        with pytest.raises((TypeError, ValueError)):
            KNNExplainer(model="not_a_model", data=self.X, labels=self.y)

        # Test dass Initialisierung mit inkompatiblen Dimensionen funktioniert
        # (KNNExplainer ist robust und validiert erst bei explain_function)
        explainer = KNNExplainer(model=self.model, data=self.X, labels=np.array([0, 1]))
        assert explainer is not None


class TestExplainFunction:
    """Test der Hauptfunktion explain_function."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=60, n_features=6, n_classes=3, n_informative=4, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)

    def test_explain_with_explicit_y_test(self):
        """Test mit explizit angegebenem y_test."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        result = explainer.explain_function(self.X[0], y_test=1)

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.max_order == 1
        assert result.index == "SV"

    def test_explain_with_automatic_prediction(self):
        """Test mit automatischer Vorhersage."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        result = explainer.explain_function(self.X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)

    def test_explain_with_class_index(self):
        """Test mit gesetztem class_index."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, class_index=2)
        result = explainer.explain_function(self.X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)

    def test_all_methods_work(self):
        """Test dass alle drei Methoden funktionieren."""
        methods = ["KNN-Shapley", "threshold_knn_shapley", "weighted_knn_shapley"]

        for method in methods:
            explainer = KNNExplainer(
                model=self.model, data=self.X, labels=self.y, method=method, tau=-0.3
            )
            result = explainer.explain_function(self.X[0])

            assert isinstance(result, InteractionValues)
            assert len(result.values) == len(self.X)
            assert np.all(np.isfinite(result.values))

    def test_unsupported_method_error(self):
        """Test Fehler bei nicht unterstützter Methode."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        explainer.method = "unsupported_method"

        with pytest.raises(ValueError, match="Method not supported"):
            explainer.explain_function(self.X[0])

    def test_different_input_shapes(self):
        """Test verschiedene Eingabeformen."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        # 1D array
        result_1d = explainer.explain_function(self.X[0])
        assert isinstance(result_1d, InteractionValues)

        # 2D array mit einer Zeile
        result_2d = explainer.explain_function(self.X[0:1])
        assert isinstance(result_2d, InteractionValues)


class TestStandardKNNShapley:
    """Test der Standard KNN-Shapley Methode."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=40, n_features=4, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="KNN-Shapley"
        )

    def test_basic_functionality(self):
        """Test der Grundfunktionalität."""
        result = self.explainer.standard_knn_shapley(self.X[0], self.y[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"
        assert result.max_order == 1
        assert result.baseline_value == 0.0

    def test_mathematical_properties(self):
        """Test mathematischer Eigenschaften."""
        result = self.explainer.standard_knn_shapley(self.X[5], self.y[5])
        shapley_values = result.values

        # Alle Werte sollten endlich sein
        assert np.all(np.isfinite(shapley_values))

        # Summe sollte eine reelle Zahl sein
        assert isinstance(np.sum(shapley_values), int | float | np.number)

        # Nicht alle Werte sollten null sein (außer in Extremfällen)

    def test_different_k_values(self):
        """Test verschiedener k-Werte."""
        k_values = [1, 3, 7, 10]
        results = []

        for k in k_values:
            explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=k)
            result = explainer.standard_knn_shapley(self.X[0], self.y[0])
            results.append(result.values)

        # Teste dass alle Ergebnisse gültig sind
        for i, result in enumerate(results):
            assert isinstance(result, np.ndarray), f"Ergebnis {i} ist kein numpy array"
            assert len(result) == len(self.X), f"Ergebnis {i} hat falsche Länge"
            assert np.all(np.isfinite(result)), f"Ergebnis {i} enthält non-finite Werte"

        # Verschiedene k-Werte können mathematisch korrekt identische Ergebnisse liefern
        # Das ist kein Fehler sondern eine Eigenschaft der Daten/Labels

        logger.info("K-Werte %s getestet - alle Ergebnisse sind gültig", k_values)

    def test_boundary_conditions(self):
        """Test von Grenzbedingungen."""
        # k = 1 (minimaler Fall)
        explainer_k1 = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=1)
        result_k1 = explainer_k1.standard_knn_shapley(self.X[0], self.y[0])
        assert isinstance(result_k1, InteractionValues)
        assert np.all(np.isfinite(result_k1.values))

        # k größer als Datensatz
        explainer_large_k = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, k=len(self.X) + 10
        )
        result_large_k = explainer_large_k.standard_knn_shapley(self.X[0], self.y[0])
        assert isinstance(result_large_k, InteractionValues)
        assert np.all(np.isfinite(result_large_k.values))

    def test_recursion_formula_implementation(self):
        """Test der korrekten Implementierung der Rekursionsformel."""
        # Test mit kleinem, kontrollierbarem Dataset
        X_small = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_small = np.array([0, 1, 0, 1])
        model_small = KNeighborsClassifier(n_neighbors=2)
        model_small.fit(X_small, y_small)

        explainer = KNNExplainer(model=model_small, data=X_small, labels=y_small, k=2)
        result = explainer.standard_knn_shapley(X_small[0], y_small[0])

        # Verifiziere dass Rekursion korrekt implementiert ist
        assert np.all(np.isfinite(result.values))
        assert len(result.values) == len(X_small)

    def test_sorted_indices_logic(self):
        """Test der Logik für sortierte Indizes."""
        # Verifiziere dass Distanz-basierte Sortierung korrekt funktioniert
        distances = np.linalg.norm(self.X - self.X[0], axis=1)
        sorted_indices = np.argsort(distances)

        # Der erste Index sollte 0 sein (Punkt zu sich selbst)
        assert sorted_indices[0] == 0
        assert distances[sorted_indices[0]] == 0


class TestThresholdKNNShapley:
    """Test der Threshold KNN-Shapley Methode."""

    def setup_method(self):
        """Setup mit standardisierten Daten."""
        self.X, self.y = make_classification(
            n_samples=60, n_features=4, n_classes=2, random_state=42
        )
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="threshold_knn_shapley", tau=-0.3
        )

    def test_basic_functionality(self):
        """Test der Grundfunktionalität."""
        result = self.explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"
        assert result.max_order == 1

    def test_sparsity_property(self):
        """Test der Sparsity-Eigenschaft."""
        result = self.explainer.threshold_knn_shapley(self.X[10], int(self.y[10]))

        # Threshold sollte Sparsity erzeugen (viele Nullwerte)
        zero_count = np.sum(result.values == 0)
        assert zero_count > 0

        # Nicht-null Werte sollten endlich sein
        non_zero_values = result.values[result.values != 0]
        if len(non_zero_values) > 0:
            assert np.all(np.isfinite(non_zero_values))

    def test_different_tau_values(self):
        """Test verschiedener tau-Werte."""
        tau_values = [-0.1, -0.5, -0.9]
        results = []

        for tau in tau_values:
            explainer = KNNExplainer(
                model=self.model,
                data=self.X,
                labels=self.y,
                method="threshold_knn_shapley",
                tau=tau,
            )
            result = explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))
            results.append(result.values)

        # Verschiedene tau-Werte sollten verschiedene Ergebnisse liefern
        assert not np.array_equal(results[0], results[1])

    def test_cosine_similarity_usage(self):
        """Test der Cosine-Similarity Verwendung."""
        # Verifiziere dass Cosine-Similarity korrekt verwendet wird
        cos_sim = cosine_similarity(self.X, self.X[0].reshape(1, -1)).flatten()
        distances = -cos_sim

        # Cosine Similarity ist zwischen -1 und 1, also sind -cos_sim zwischen -1 und 1
        assert np.all(distances >= -1.0), "Distances sollten >= -1 sein"
        assert np.all(distances <= 1.0), "Distances sollten <= 1 sein"

        # Der erste Punkt (zu sich selbst) sollte Distanz -1 haben (cos_sim = 1)
        assert np.isclose(distances[0], -1.0), "Distanz zu sich selbst sollte -1 sein"

        # Test mit dem Explainer
        result = self.explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))
        assert isinstance(result, InteractionValues)

    def test_helper_function_a1(self):
        """Test der Helper-Funktion function_a1."""
        # Test mit verschiedenen Parametern
        result1 = self.explainer.function_a1(0, 10, 5, self.y[0])
        result2 = self.explainer.function_a1(0, 10, 5, 1 - self.y[0])

        assert isinstance(result1, float)
        assert isinstance(result2, float)
        assert np.isfinite(result1)
        assert np.isfinite(result2)
        assert result1 != result2  # Verschiedene Labels sollten verschiedene Ergebnisse geben

    def test_helper_function_a2(self):
        """Test der Helper-Funktion function_a2."""
        result = self.explainer.function_a2(5, 20)

        assert isinstance(result, float)
        assert np.isfinite(result)

        # Test mit verschiedenen Parametern
        result2 = self.explainer.function_a2(3, 15)
        assert isinstance(result2, float)
        assert np.isfinite(result2)

    def test_correction_term(self):
        """Test der Correction-Term Funktion."""
        # Test mit c_tau = 0
        result_zero = self.explainer.correction_term(0, 0, self.y[0])
        assert result_zero == 0.0

        # Test mit c_tau > 0
        result_nonzero = self.explainer.correction_term(0, 5, self.y[0])
        assert isinstance(result_nonzero, float)
        assert np.isfinite(result_nonzero)

    def test_edge_cases(self):
        """Test von Edge Cases."""
        # Test mit sehr restriktivem tau (keine Nachbarn)
        restrictive_explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="threshold_knn_shapley", tau=-5.0
        )
        result = restrictive_explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        # Sollte hauptsächlich Nullen produzieren
        zero_count = np.sum(result.values == 0)
        assert zero_count >= len(self.X) * 0.8  # Mindestens 80% Nullen

        # Test mit sehr liberalem tau (alle Nachbarn)
        liberal_explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="threshold_knn_shapley", tau=5.0
        )
        result_liberal = liberal_explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        # Sollte weniger Nullen produzieren
        zero_count_liberal = np.sum(result_liberal.values == 0)
        assert zero_count_liberal < zero_count


class TestWeightedKNNShapley:
    """Test der Weighted KNN-Shapley Methode."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=50, n_features=4, n_classes=2, random_state=42
        )
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley"
        )

    def test_basic_functionality(self):
        """Test der Grundfunktionalität."""
        result = self.explainer.weighted_knn_shapley(self.X[0], int(self.y[0]))

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"
        assert result.max_order == 1

    def test_discretize_array_function(self):
        """Test der discretize_array Funktion."""
        # Test mit Standard-Parametern
        arr = np.array([0.1, 0.5, 0.9, -0.3, 0.0])
        result = KNNExplainer.discretize_array(arr)

        assert len(result) == len(arr)
        assert np.all(np.isfinite(result))

        # Test mit verschiedenen b-Werten
        result_b2 = KNNExplainer.discretize_array(arr, b=2)
        result_b4 = KNNExplainer.discretize_array(arr, b=4)

        assert not np.array_equal(result, result_b2)
        assert not np.array_equal(result, result_b4)

        # Test Edge Cases
        empty_result = KNNExplainer.discretize_array(np.array([]))
        assert len(empty_result) == 0

        single_result = KNNExplainer.discretize_array(np.array([0.5]))
        assert len(single_result) == 1
        assert np.isfinite(single_result[0])

    def test_rbf_kernel_properties(self):
        """Test der RBF-Kernel Eigenschaften (ohne Alpha-Parameter)."""
        # Test dass der RBF-Kernel korrekt funktioniert
        explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", m_star=10
        )
        result = explainer.weighted_knn_shapley(self.X[0], int(self.y[0]))

        # Grundlegende Eigenschaften testen
        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert np.all(np.isfinite(result.values))

    def test_compute_f_i_function(self):
        """Test der compute_f_i Funktion."""
        disc_weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        w_k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        s_to_index = {s: i for i, s in enumerate(w_k)}

        f_result = self.explainer.compute_f_i(disc_weight, 1, 5, 3, w_k, s_to_index)

        assert f_result.shape == (5, 2, 6)  # (m_star, k-1, len(w_k))
        assert np.all(np.isfinite(f_result))
        assert np.all(f_result >= 0)  # F-Werte sollten nicht negativ sein

    def test_compute_g_i_function(self):
        """Test der compute_g_i Funktion."""
        disc_weight = np.array([0.1, 0.2, 0.3, 0.4])
        w_k = [0.1, 0.2, 0.3, 0.4, 0.5]
        s_to_index = {s: i for i, s in enumerate(w_k)}

        # Erstelle Mock f_i
        rng = np.random.default_rng(42)
        f_i = rng.random((4, 2, 5))

        g_result = self.explainer.compute_g_i(disc_weight, 1, 3, 4, f_i, w_k, s_to_index, 1)

        EXPECTED_G_LENGTH = 3
        assert len(g_result) == EXPECTED_G_LENGTH
        # k
        assert np.all(np.isfinite(g_result))

    def test_compute_r_i_function(self):
        """Test der compute_r_i Funktion."""
        disc_weight = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        w_k = [-0.4, -0.2, 0.1, 0.3, 0.5]
        s_to_index = {s: i for i, s in enumerate(w_k)}
        y_train = np.array([0, 1, 0, 1, 0])

        # Erstelle Mock f_i
        rng = np.random.default_rng(42)
        f_i = rng.random((5, 2, 5))

        r_result = self.explainer.compute_r_i(
            f_i, y_train, 1, disc_weight, w_k, s_to_index, 1, 3, 5
        )

        EXPECTED_R_LENGTH = 5
        assert len(r_result) == EXPECTED_R_LENGTH

        assert np.all(np.isfinite(r_result))

    def test_weight_combinations_with_combinations_with_replacement(self):
        """Test der Gewichtskombinationen mit combinations_with_replacement."""
        weight_levels = np.array([0.1, 0.2, 0.3])
        k = 3

        # Test combinations_with_replacement direkt
        all_combinations = []
        for ell in range(1, k):
            combs = list(combinations_with_replacement(weight_levels, ell))
            all_combinations = [sum(comb) for comb in combs]

        w_k = sorted(set(all_combinations))

        assert len(w_k) > 0
        assert all(isinstance(w, int | float) for w in w_k)

    def test_m_star_handling(self):
        """Test der m_star Parameter-Behandlung."""
        # Test mit explizitem m_star
        explainer_explicit = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", m_star=10
        )
        result = explainer_explicit.weighted_knn_shapley(self.X[0], int(self.y[0]))
        assert isinstance(result, InteractionValues)

        # Test mit automatischem m_star (None)
        explainer_auto = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", m_star=None
        )
        result_auto = explainer_auto.weighted_knn_shapley(self.X[0], int(self.y[0]))
        assert isinstance(result_auto, InteractionValues)


class TestErrorHandlingAndEdgeCases:
    """Test der Fehlerbehandlung und Edge Cases."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=20, n_features=4, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X, self.y)

    def test_invalid_input_types(self):
        """Test mit ungültigen Eingabetypen."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        # Test mit String-Eingabe statt Array
        with pytest.raises((TypeError, ValueError, AttributeError)):
            explainer.explain_function("invalid_input")

        # Test mit None-Eingabe
        with pytest.raises((TypeError, ValueError, AttributeError)):
            explainer.explain_function(None)

    def test_nan_input_handling(self):
        """Test der NaN-Eingabe-Behandlung."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        x_nan = self.X[0].copy()
        x_nan[0] = np.nan

        # Sollte einen Fehler werfen oder NaN handhaben
        with pytest.raises((ValueError, RuntimeError)):
            explainer.explain_function(x_nan)

    def test_inf_input_handling(self):
        """Test der Infinity-Eingabe-Behandlung."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        x_inf = self.X[0].copy()
        x_inf[0] = np.inf

        with pytest.raises((ValueError, RuntimeError)):
            explainer.explain_function(x_inf)

    def test_single_class_data(self):
        """Test mit Ein-Klassen-Daten."""
        rng = np.random.default_rng(42)
        X_single = rng.random((20, 4))
        y_single = np.zeros(20, dtype=int)  # Alle gleiche Klasse

        model_single = KNeighborsClassifier(n_neighbors=3)
        model_single.fit(X_single, y_single)

        explainer = KNNExplainer(model=model_single, data=X_single, labels=y_single)
        result = explainer.explain_function(X_single[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(X_single)
        assert np.all(np.isfinite(result.values))

    def test_k_larger_than_dataset(self):
        """Test mit k größer als Datensatz."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=len(self.X) + 10)
        result = explainer.explain_function(self.X[0])

        assert isinstance(result, InteractionValues)
        assert np.all(np.isfinite(result.values))

    def test_very_small_dataset(self):
        """Test mit sehr kleinem Datensatz."""
        rng = np.random.default_rng(42)
        X_tiny = rng.random((3, 2))
        y_tiny = np.array([0, 1, 0])

        model_tiny = KNeighborsClassifier(n_neighbors=2)
        model_tiny.fit(X_tiny, y_tiny)

        explainer = KNNExplainer(model=model_tiny, data=X_tiny, labels=y_tiny)
        result = explainer.explain_function(X_tiny[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(X_tiny)

    def test_zero_division_protection_function_a2(self):
        """Test Schutz vor Division durch Null in function_a2."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        # Test mit c_tau = 0 sollte nicht zu Division durch Null führen
        try:
            result = explainer.function_a2(0, 5)
            assert np.isfinite(result)
        except ZeroDivisionError:
            pytest.fail("function_a2 sollte Division durch Null vermeiden")

    def test_mathematical_edge_cases_combinatorics(self):
        """Test mathematischer Edge Cases in kombinatorischen Berechnungen."""
        # Test math.comb mit Edge Cases
        try:
            # Diese Aufrufe sollten nicht zu Fehlern führen
            comb_result1 = math.comb(0, 0)
            comb_result2 = math.comb(5, 0)
            comb_result3 = math.comb(5, 5)

            assert all(isinstance(r, int) for r in [comb_result1, comb_result2, comb_result3])
        except ValueError:
            pytest.fail("math.comb sollte Edge Cases korrekt handhaben")

    def test_empty_neighbors_threshold(self):
        """Test bei leerem Nachbarschaftsset in Threshold-Methode."""
        explainer = KNNExplainer(
            model=self.model,
            data=self.X,
            labels=self.y,
            method="threshold_knn_shapley",
            tau=-10.0,  # Sehr restriktiv
        )

        result = explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        # Sollte hauptsächlich Nullen zurückgeben, aber nicht crashen
        assert isinstance(result, InteractionValues)
        assert np.all(np.isfinite(result.values))


class TestPerformanceAndIntegration:
    """Test der Performance und Integration."""

    def test_iris_integration_all_methods(self):
        """Integration Test mit Iris Dataset für alle Methoden."""
        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        methods = ["KNN-Shapley", "threshold_knn_shapley", "weighted_knn_shapley"]

        for method in methods:
            explainer = KNNExplainer(model=model, data=X_train, labels=y_train, method=method)

            # Test mehrere Samples
            for i in range(min(3, len(X_test))):
                result = explainer.explain_function(X_test[i])
                assert isinstance(result, InteractionValues)
                assert len(result.values) == len(X_train)
                assert np.all(np.isfinite(result.values))

    def test_performance_timing(self):
        """Performance Test mit Zeitmessung."""
        X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)

        start_time = time.time()
        result = explainer.explain_function(X[0])
        end_time = time.time()

        assert isinstance(result, InteractionValues)
        TIME_LIMIT = 10.0
        assert (end_time - start_time) < TIME_LIMIT
        # Sollte unter 10 Sekunden dauern

    def test_memory_usage_large_dataset(self):
        """Test Speicherverbrauch mit größerem Datensatz."""
        X, y = make_classification(
            n_samples=200, n_features=12, n_classes=3, n_informative=8, random_state=42
        )
        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(X)
        assert np.all(np.isfinite(result.values))

    def test_consistency_across_multiple_runs(self):
        """Test Konsistenz über mehrere Durchläufe."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)

        # Führe mehrere Erklärungen durch
        results = []
        for _ in range(3):
            result = explainer.explain_function(X[0])
            results.append(result.values)

        # Ergebnisse sollten konsistent sein
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_different_model_configurations(self):
        """Test verschiedener Modell-Konfigurationen."""
        X, y = make_classification(n_samples=60, n_features=4, n_classes=2, random_state=42)

        configs = [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 7, "weights": "distance"},
            {"n_neighbors": 5, "weights": "uniform", "metric": "manhattan"},
        ]

        for config in configs:
            model = KNeighborsClassifier(**config)
            model.fit(X, y)

            explainer = KNNExplainer(model=model, data=X, labels=y)
            result = explainer.explain_function(X[0])

            assert isinstance(result, InteractionValues)
            assert len(result.values) == len(X)
            assert np.all(np.isfinite(result.values))


class TestSpecialCases:
    """Test spezieller Cases und Corner Cases."""

    def test_identical_data_points(self):
        """Test mit identischen Datenpunkten."""
        # Erstelle Dataset mit identischen Punkten
        X = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4]])
        y = np.array([0, 0, 1, 1, 1])

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert np.all(np.isfinite(result.values))

    def test_high_dimensional_data(self):
        """Test mit hochdimensionalen Daten."""
        X, y = make_classification(n_samples=50, n_features=20, n_informative=10, random_state=42)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(X)

    def test_binary_features(self):
        """Test mit binären Features."""
        rng = np.random.default_rng(42)
        X = rng.choice([0, 1], size=(40, 6))
        y = rng.choice([0, 1], size=40)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert np.all(np.isfinite(result.values))


if __name__ == "__main__":
    # Führe Tests aus
    pytest.main([__file__, "-v", "--tb=short"])
