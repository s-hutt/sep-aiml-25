import numpy as np
import pytest
from scipy.stats import norm

from shapiq_student.copula import GaussianCopulaImputer

def make_synthetic_data(n=50, d=5, random_state=42):
    rng = np.random.default_rng(random_state)
    # Multivariate normal data
    mean = np.zeros(d)
    cov = np.eye(d)
    return rng.multivariate_normal(mean, cov, size=n)

def dummy_model_predict(X):
    # Dummy predict: sum of features
    return X.sum(axis=1)

@pytest.fixture
def imputer():
    data = make_synthetic_data()
    imp = GaussianCopulaImputer(
        model=type("DummyModel", (), {"predict": dummy_model_predict})(),
        data=data,
        x=data[0],
        sample_size=5,
        categorical_features=None,
        random_state=123,
    )
    return imp

def test_init_rejects_categorical():
    data = make_synthetic_data()
    with pytest.raises(ValueError, match="does not support categorical features"):
        GaussianCopulaImputer(
            model=None,
            data=data,
            x=data[0],
            categorical_features=[0],
        )

def test_gaussian_transform_inverse_roundtrip(imputer):
    x_col = imputer._train_data[:, 0]
    z = imputer.gaussian_transform(x_col)
    x_back = imputer.inv_gaussian_transform(z[:, np.newaxis], imputer._train_data[:, 0][:, np.newaxis])
    # Back transform should approximately recover original sorted values
    np.testing.assert_allclose(np.sort(x_col), np.sort(x_back[:, 0]), rtol=1e-1, atol=1e-1)

def test_quantile_type7_matches_numpy_quantile(imputer):
    x = np.linspace(0, 10, 100)
    probs = np.linspace(0, 1, 11)
    q_custom = imputer.quantile_type7(x, probs)
    q_np = np.quantile(x, probs, method='linear')  # NumPy default interpolation is type 7
    np.testing.assert_allclose(q_custom, q_np)

def test_gaussian_transform_separate_behavior(imputer):
    # Compose yx vector: 1 Gaussianized, rest original
    y = imputer._x_gauss
    x = imputer._x
    yx = np.concatenate([y, x])
    z = imputer.gaussian_transform_separate(yx, n_y=1)
    assert z.shape == (1,)
    # Output should be finite
    assert np.all(np.isfinite(z))

def test_value_function_output_shape_and_range(imputer):
    n_features = imputer._train_data.shape[1]
    coalitions = np.array([
        [True]*n_features,
        [False]*n_features,
        [True] + [False]*(n_features-1),
        [False, True] + [False]*(n_features-2)
    ])
    vals = imputer.value_function(coalitions)
    assert vals.shape == (coalitions.shape[0],)
    # Since dummy model sums features, values should be floats
    assert np.all(np.isfinite(vals))

def test_empty_coalition_returns_empty_prediction(imputer):
    n_features = imputer._train_data.shape[1]
    coalitions = np.zeros((3, n_features), dtype=bool)
    vals = imputer.value_function(coalitions)
    # All should equal empty prediction
    assert np.allclose(vals, imputer.empty_prediction)

def test_prepare_data_copula_shape_and_consistency(imputer):
    n_features = imputer._train_data.shape[1]
    MC_samples = np.random.normal(size=(3, n_features))
    coalitions = np.array([[True] * n_features, [False] * n_features])
    result = imputer._prepare_data_copula_py(
        MC_samples_mat=MC_samples,
        x_explain_original=imputer._x[np.newaxis, :],
        x_explain_gauss=imputer._x_gauss[np.newaxis, :],
        x_train_mat=imputer._train_data,
        S=coalitions.astype(float),
        mu=imputer._copula_mu,
        cov_mat=imputer._copula_cov
    )
    # Shape: (MC_samples, n_explain * n_coalitions, n_features)
    assert result.shape == (3, 1 * 2, n_features)
    # Values finite
    assert np.all(np.isfinite(result))

def test_calc_empty_prediction_matches_predict_mean(imputer):
    pred_mean = imputer.calc_empty_prediction()
    preds = imputer.predict(imputer.data)
    assert np.isclose(pred_mean, np.mean(preds))

def test_error_on_invalid_method():
    data = make_synthetic_data()
    with pytest.raises(ValueError, match="for 'gaussCopula' imputers only"):
        GaussianCopulaImputer(
            model=None,
            data=data,
            method="invalidMethod"
        )

def test_error_on_gaussian_transform_separate_invalid_ny(imputer):
    yx = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        imputer.gaussian_transform_separate(yx, n_y=2)
