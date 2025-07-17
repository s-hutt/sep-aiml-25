"""Ausgewogene Tests für KNNExplainer - wichtigste Funktionen mit guter Abdeckung.

Target: ~92% Code Coverage mit vernünftigem Umfang.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapiq.interaction_values import InteractionValues
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from shapiq_student.knn_explainer import KNNExplainer


class TestKNNExplainerInitialization:
    """Test KNNExplainer Initialisierung und Parameter."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=50, n_features=4, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)

    def test_default_parameters(self):
        """Test Standard-Parameter."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)

        default_k = 5
        assert explainer.k == default_k

        default_tau = -0.5
        assert explainer.tau == default_tau

        default_alpha = 1.0
        assert explainer.alpha == default_alpha

        assert explainer.method == "KNN-Shapley"
        assert explainer.mode == "normal"

    def test_custom_parameters(self):
        """Test benutzerdefinierte Parameter."""
        explainer = KNNExplainer(
            model=self.model,
            data=self.X,
            labels=self.y,
            method="threshold_knn_shapley",
            k=7,
            tau=-0.2,
            alpha=2.0,
            class_index=1,
            m_star=15,
        )

        EXPECTED_K = 7
        assert explainer.k == EXPECTED_K

        EXPECTED_TAU = -0.2
        assert explainer.tau == EXPECTED_TAU

        EXPECTED_ALPHA = 2.0
        assert explainer.alpha == EXPECTED_ALPHA

        assert explainer.method == "threshold_knn_shapley"
        assert explainer.mode == "threshold"
        assert explainer.class_index == 1

        EXPECTED_M_STAR = 15
        assert explainer.m_star == EXPECTED_M_STAR

    def test_automatic_method_detection(self):
        """Test automatische Methodenerkennung."""
        # Weighted model sollte weighted_knn_shapley verwenden
        weighted_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        weighted_model.fit(self.X, self.y)
        explainer = KNNExplainer(model=weighted_model, data=self.X, labels=self.y)
        assert explainer.method == "weighted_knn_shapley"

        # Standard model sollte KNN-Shapley verwenden
        standard_model = KNeighborsClassifier(n_neighbors=5, weights="uniform")
        standard_model.fit(self.X, self.y)
        explainer = KNNExplainer(model=standard_model, data=self.X, labels=self.y)
        assert explainer.method == "KNN-Shapley"

    def test_invalid_method_error(self):
        """Test Fehler bei ungültiger Methode."""
        with pytest.raises(ValueError, match="Unknown method"):
            KNNExplainer(model=self.model, data=self.X, labels=self.y, method="invalid")


class TestExplainFunction:
    """Test Hauptfunktion explain_function."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=80, n_features=8, n_informative=4, n_redundant=2, n_classes=3, random_state=42
        )
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)

    def test_with_y_test_provided(self):
        """Test mit explizitem y_test."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        result = explainer.explain_function(self.X[0], y_test=1)

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.max_order == 1

    def test_auto_prediction(self):
        """Test automatische Vorhersage."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        result = explainer.explain_function(self.X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)

    def test_with_class_index(self):
        """Test mit gesetztem class_index."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, class_index=2)
        result = explainer.explain_function(self.X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)

    def test_all_methods(self):
        """Test alle drei Methoden."""
        methods = ["KNN-Shapley", "threshold_knn_shapley", "weighted_knn_shapley"]

        for method in methods:
            explainer = KNNExplainer(
                model=self.model, data=self.X, labels=self.y, method=method, tau=-0.3, alpha=1.0
            )
            result = explainer.explain_function(self.X[0])

            assert isinstance(result, InteractionValues)
            assert len(result.values) == len(self.X)
            assert np.all(np.isfinite(result.values))

    def test_unsupported_method_error(self):
        """Test Fehler bei nicht unterstützter Methode."""
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y)
        explainer.method = "unsupported"  # Manually set

        with pytest.raises(ValueError, match="Method not supported"):
            explainer.explain_function(self.X[0])


class TestStandardKNNShapley:
    """Test Standard KNN-Shapley."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=40, n_features=6, n_informative=3, n_redundant=1, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="KNN-Shapley"
        )

    def test_basic_functionality(self):
        """Test Grundfunktionalität."""
        result = self.explainer.standard_knn_shapley(self.X[0], self.y[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"
        assert result.baseline_value == 0.0

    def test_mathematical_properties(self):
        """Test mathematische Eigenschaften."""
        result = self.explainer.standard_knn_shapley(self.X[5], self.y[5])
        shapley_values = result.values

        assert np.all(np.isfinite(shapley_values))
        assert isinstance(np.sum(shapley_values), int | float)

    def test_different_k_values(self):
        """Test verschiedene k-Werte."""
        k_values = [1, 3, 7]
        results = []

        for k in k_values:
            explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=k)
            result = explainer.standard_knn_shapley(self.X[0], self.y[0])
            results.append(result.values)

        # Ergebnisse sollten für verschiedene k unterschiedlich sein
        for i in range(len(results) - 1):
            assert not np.array_equal(results[i], results[i + 1])

    def test_boundary_conditions(self):
        """Test Grenzbedingungen für Standard KNN-Shapley."""
        # Test mit k=1 (minimaler Fall)
        explainer_k1 = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=1)
        result_k1 = explainer_k1.standard_knn_shapley(self.X[0], self.y[0])
        assert isinstance(result_k1, InteractionValues)

        # Test mit k größer als Datensatz
        explainer_large_k = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, k=len(self.X) + 10
        )
        result_large_k = explainer_large_k.standard_knn_shapley(self.X[0], self.y[0])
        assert isinstance(result_large_k, InteractionValues)

        # Test Rekursionslogik: Letzter Punkt sollte spezifischen Wert haben
        result = self.explainer.standard_knn_shapley(self.X[0], self.y[0])
        # Shapley-Werte sollten endlich und nicht alle null sein
        assert np.any(result.values != 0)
        assert np.all(np.isfinite(result.values))


class TestThresholdKNNShapley:
    """Test Threshold KNN-Shapley."""

    def setup_method(self):
        """Setup mit Iris Dataset."""
        iris = load_iris()
        self.X, self.y = iris.data, iris.target

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="threshold_knn_shapley", tau=-0.3
        )

    def test_basic_functionality(self):
        """Test Grundfunktionalität."""
        result = self.explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"

    def test_sparsity_property(self):
        """Test Sparsity-Eigenschaft."""
        result = self.explainer.threshold_knn_shapley(self.X[50], int(self.y[50]))

        # Threshold sollte Sparsity erzeugen
        zero_count = np.sum(result.values == 0)
        assert zero_count > 0

        # Nicht-null Werte sollten endlich sein
        non_zero = result.values[result.values != 0]
        if len(non_zero) > 0:
            assert np.all(np.isfinite(non_zero))

    def test_different_tau_values(self):
        """Test verschiedene tau-Werte."""
        tau_values = [-0.1, -0.5, -0.9]
        active_counts = []

        for tau in tau_values:
            explainer = KNNExplainer(
                model=self.model,
                data=self.X,
                labels=self.y,
                method="threshold_knn_shapley",
                tau=tau,
            )
            result = explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))
            active_counts.append(np.sum(result.values != 0))

        # Sollte Variation in aktiven Nachbarn geben
        assert len(set(active_counts)) > 1

    def test_helper_functions(self):
        """Test Helper-Funktionen."""
        # Test function_a1
        result1 = self.explainer.function_a1(0, 10, 5, self.y[0])
        result2 = self.explainer.function_a1(0, 10, 5, 1 if self.y[0] == 0 else 0)
        assert isinstance(result1, float)
        assert np.isfinite(result1)

        assert isinstance(result2, float)
        assert np.isfinite(result2)

        assert result1 != result2

        # Test function_a2
        result = self.explainer.function_a2(5, 20)
        assert isinstance(result, float)
        assert np.isfinite(result)

        # Test correction_term
        result_zero = self.explainer.correction_term(0, 0, self.y[0])
        result_nonzero = self.explainer.correction_term(0, 5, self.y[0])
        assert result_zero == 0.0
        assert isinstance(result_nonzero, float)
        assert np.isfinite(result_nonzero)

    def test_threshold_edge_cases(self):
        """Test zusätzliche Edge Cases für Threshold KNN-Shapley."""
        # Test mit sehr restriktivem tau (keine Nachbarn)
        restrictive_explainer = KNNExplainer(
            model=self.model,
            data=self.X,
            labels=self.y,
            method="threshold_knn_shapley",
            tau=-2.0,  # Sehr restriktiv
        )
        result = restrictive_explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        # Sollte viele Nullen produzieren
        zero_count = np.sum(result.values == 0)
        assert zero_count >= len(self.X) * 0.8  # Mindestens 80% Nullen

        # Test mit sehr liberalem tau (alle Nachbarn)
        liberal_explainer = KNNExplainer(
            model=self.model,
            data=self.X,
            labels=self.y,
            method="threshold_knn_shapley",
            tau=2.0,  # Sehr liberal
        )
        result_liberal = liberal_explainer.threshold_knn_shapley(self.X[0], int(self.y[0]))

        # Sollte weniger Nullen produzieren
        zero_count_liberal = np.sum(result_liberal.values == 0)
        assert zero_count_liberal < zero_count  # Weniger Nullen als bei restriktivem tau


class TestWeightedKNNShapley:
    """Test Weighted KNN-Shapley."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=60, n_features=4, n_classes=2, random_state=42
        )
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.model.fit(self.X, self.y)
        self.explainer = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", alpha=1.0
        )

    def test_basic_functionality(self):
        """Test Grundfunktionalität."""
        result = self.explainer.weighted_knn_shapley(self.X[0], int(self.y[0]))

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(self.X)
        assert result.index == "SV"

    def test_discretize_array(self):
        """Test Diskretisierungs-Funktion."""
        arr = np.array([0.1, 0.5, 0.9, -0.3])

        # Test Standard b=3
        result = KNNExplainer.discretize_array(arr)
        assert len(result) == len(arr)
        assert np.all(np.isfinite(result))

        # Test verschiedene b-Werte
        result_b2 = KNNExplainer.discretize_array(arr, b=2)
        result_b4 = KNNExplainer.discretize_array(arr, b=4)
        assert not np.array_equal(result, result_b2)
        assert not np.array_equal(result, result_b4)

        # Test Edge Cases
        empty = KNNExplainer.discretize_array(np.array([]))
        assert len(empty) == 0

        single = KNNExplainer.discretize_array(np.array([0.5]))
        assert len(single) == 1
        assert np.isfinite(single[0])

    def test_different_alpha_values(self):
        """Test verschiedene Alpha-Werte."""
        # Verwende extremere Alpha-Werte für deutlichere Unterschiede
        alpha_values = [0.1, 1.0, 10.0]
        results = []

        for alpha in alpha_values:
            explainer = KNNExplainer(
                model=self.model,
                data=self.X,
                labels=self.y,
                method="weighted_knn_shapley",
                alpha=alpha,
                m_star=10,
            )
            result = explainer.weighted_knn_shapley(self.X[0], int(self.y[0]))
            results.append(result.values)

        # Teste nur zwischen extremen Werten (erster und letzter)
        assert not np.array_equal(results[0], results[-1]), (
            "Alpha-Werte sollten unterschiedliche Ergebnisse produzieren"
        )

        # Teste dass nicht alle Werte null sind
        assert np.any(results[0] != 0) or np.any(results[-1] != 0), (
            "Mindestens ein Ergebnis sollte nicht-null Werte haben"
        )

    def test_helper_functions(self):
        """Test wichtige Helper-Funktionen."""
        # Setup für Helper-Tests
        disc_weight = np.array([0.1, 0.2, 0.3, 0.4])
        w_k = [0.1, 0.2, 0.3, 0.4, 0.5]
        s_to_index = {s: i for i, s in enumerate(w_k)}

        # Test compute_f_i
        f_result = self.explainer.compute_f_i(disc_weight, 1, 4, 3, w_k, s_to_index)
        assert f_result.shape == (4, 2, 5)
        assert np.all(np.isfinite(f_result))

        # Test compute_g_i
        rng = np.random.default_rng()
        f_i = rng.random((4, 2, 5))

        g_result = self.explainer.compute_g_i(disc_weight, 1, 3, 4, f_i, w_k, s_to_index, 1)
        EXPECTED_LEN_G = 3
        assert len(g_result) == EXPECTED_LEN_G

        assert np.all(np.isfinite(g_result))

    def test_edge_cases_weighted(self):
        """Test Edge Cases für weighted KNN-Shapley."""
        # Test mit sehr kleinem m_star
        explainer_small = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", m_star=3
        )
        result = explainer_small.weighted_knn_shapley(self.X[0], int(self.y[0]))
        assert isinstance(result, InteractionValues)

        # Test mit negativen Gewichten
        explainer_neg = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="weighted_knn_shapley", alpha=-0.5
        )
        result_neg = explainer_neg.weighted_knn_shapley(self.X[0], int(self.y[0]))
        assert isinstance(result_neg, InteractionValues)

        # Test compute_r_i direkt
        disc_weight = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        w_k = [-0.4, -0.2, 0.1, 0.3, 0.5]
        s_to_index = {s: i for i, s in enumerate(w_k)}
        rng = np.random.default_rng()
        f_i = rng.random((5, 2, 5))

        r_result = self.explainer.compute_r_i(f_i, disc_weight, w_k, s_to_index, 1, 3, 5)
        EXPECTED_LEN_R = 5
        assert len(r_result) == EXPECTED_LEN_R

        assert np.all(np.isfinite(r_result))

    def test_weight_combinations(self):
        """Test verschiedene Gewichtskombinationen."""
        # Test mit verschiedenen weight levels
        test_arrays = [
            np.array([0.0, 0.5, 1.0]),  # Standard Fall
            np.array([-1.0, 0.0, 1.0]),  # Mit negativen Werten
            np.array([0.1, 0.1, 0.1]),  # Gleiche Werte
            np.array([0.9, 0.8, 0.7, 0.6]),  # Absteigende Werte
        ]

        for test_array in test_arrays:
            disc_result = KNNExplainer.discretize_array(test_array, b=2)
            assert len(disc_result) == len(test_array)
            assert np.all(np.isfinite(disc_result))


class TestErrorHandlingAndEdgeCases:
    """Test Fehlerbehandlung und Edge Cases."""

    def setup_method(self):
        """Setup für Tests."""
        self.X, self.y = make_classification(
            n_samples=20, n_features=4, n_classes=2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X, self.y)

    def test_invalid_k_value(self):
        """Test mit ungültigem k-Wert."""
        # k größer als Trainingsdaten
        explainer = KNNExplainer(model=self.model, data=self.X, labels=self.y, k=30)
        result = explainer.explain_function(self.X[0])  # Sollte trotzdem funktionieren
        assert isinstance(result, InteractionValues)

    def test_single_class_data(self):
        """Test mit Ein-Klassen-Daten."""
        rng = np.random.default_rng()
        X = rng.random((20, 4))

        y = np.zeros(20, dtype=int)  # Alle gleiche Klasse

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert len(result.values) == len(X)

    def test_nan_input_handling(self):
        """Test NaN-Eingabe-Behandlung."""
        X, y = make_classification(n_samples=30, n_features=4, n_classes=2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)

        x_nan = X[0].copy()
        x_nan[0] = np.nan

        with pytest.raises((ValueError, RuntimeError)):
            explainer.explain_function(x_nan)

    def test_method_selection_edge_cases(self):
        """Test Edge Cases bei Methodenselektion."""
        # Test explizite threshold Methode (manuell spezifiziert)
        explainer_threshold = KNNExplainer(
            model=self.model, data=self.X, labels=self.y, method="threshold_knn_shapley"
        )
        assert explainer_threshold.method == "threshold_knn_shapley"
        assert explainer_threshold.mode == "threshold"

        # Test mit sehr kleinem Dataset
        rng = np.random.default_rng()
        X_tiny = rng.random((3, 2))

        y_tiny = np.array([0, 1, 0])
        model_tiny = KNeighborsClassifier(n_neighbors=2)
        model_tiny.fit(X_tiny, y_tiny)

        explainer_tiny = KNNExplainer(model=model_tiny, data=X_tiny, labels=y_tiny)
        result_tiny = explainer_tiny.explain_function(X_tiny[0])
        assert isinstance(result_tiny, InteractionValues)
        assert len(result_tiny.values) == len(X_tiny)

    def test_zero_division_protection(self):
        """Test Schutz vor Division durch Null."""
        X, y = make_classification(
            n_samples=10, n_features=4, n_informative=2, n_redundant=0, n_classes=2, random_state=42
        )
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X, y)

        # Test mit k größer als verfügbare Daten
        explainer = KNNExplainer(model=model, data=X, labels=y, k=15)
        result = explainer.explain_function(X[0])

        assert isinstance(result, InteractionValues)
        assert np.all(np.isfinite(result.values))


class TestIntegrationAndPerformance:
    """Integration Tests und Performance."""

    def test_iris_integration(self):
        """Integration Test mit Iris Dataset."""
        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Test alle Methoden
        methods = ["KNN-Shapley", "threshold_knn_shapley", "weighted_knn_shapley"]

        for method in methods:
            explainer = KNNExplainer(model=model, data=X_train, labels=y_train, method=method)

            # Test mehrere Samples
            for i in range(3):
                result = explainer.explain_function(X_test[i])
                assert isinstance(result, InteractionValues)
                assert len(result.values) == len(X_train)
                assert np.all(np.isfinite(result.values))

    def test_performance_small_dataset(self):
        """Performance Test kleines Dataset."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)

        explainer = KNNExplainer(model=model, data=X, labels=y)

        import time

        start = time.time()
        result = explainer.explain_function(X[0])
        duration = time.time() - start

        assert isinstance(result, InteractionValues)
        MAX_ALLOWED_DURATION = 5.0
        assert duration < MAX_ALLOWED_DURATION  # Sollte unter 5 Sekunden dauern

    def test_comprehensive_integration(self):
        """Umfassender Integration Test mit verschiedenen Szenarien."""
        # Test verschiedene Modell-Konfigurationen
        configs = [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 7, "weights": "distance"},
            {"n_neighbors": 5, "weights": "uniform"},
        ]

        X, y = make_classification(n_samples=40, n_features=5, n_classes=2, random_state=42)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for config in configs:
            model = KNeighborsClassifier(**config)
            model.fit(X, y)

            # Test automatische Methodenerkennung
            explainer = KNNExplainer(model=model, data=X, labels=y)
            result = explainer.explain_function(X[0])

            assert isinstance(result, InteractionValues)
            assert len(result.values) == len(X)
            assert np.all(np.isfinite(result.values))

            # Verifikation dass richtige Methode gewählt wurde
            if config["weights"] == "distance":
                assert explainer.method == "weighted_knn_shapley"
            else:
                assert explainer.method == "KNN-Shapley"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
