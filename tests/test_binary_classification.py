import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from pyannote.metrics.binary_classification import Calibration
from pyannote.metrics.binary_classification import precision_recall_curve


def test_precision_recall_curve_perfect_scores():
    y_true = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    precision, recall, _, auc = precision_recall_curve(y_true, scores)
    assert auc == pytest.approx(1.0)


def test_precision_recall_curve_auc():
    # The two nonzero-width trapezoids have areas 1/2 and 7/24.
    y_true = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    precision, recall, _, auc = precision_recall_curve(y_true, scores)
    assert auc == pytest.approx(19 / 24)


def test_precision_recall_curve_distances():
    y_true = np.array([False, False, True, True])
    distances = np.array([0.9, 0.8, 0.2, 0.1])
    _, _, thresholds, auc = precision_recall_curve(y_true, distances, distances=True)
    assert auc == pytest.approx(1.0)


@pytest.mark.parametrize("equal_priors", [False, True])
@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration(equal_priors, method):
    rng = np.random.RandomState(0)
    y_true = rng.rand(200) < 0.3
    scores = rng.randn(200) + 2 * y_true
    calibration = Calibration(equal_priors=equal_priors, method=method).fit(scores, y_true)
    test_scores = np.linspace(-3.0, 5.0, 101)
    probabilities = calibration.transform(test_scores)
    assert probabilities.shape == test_scores.shape
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    # probabilities increase with scores
    assert np.all(np.diff(probabilities) >= -1e-12)


def test_calibration_isotonic_matches_isotonic_regression():
    rng = np.random.RandomState(1)
    y_true = rng.rand(200) < 0.3
    scores = rng.randn(200) + 2 * y_true
    calibration = Calibration(method="isotonic").fit(scores, y_true)
    expected = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        scores, y_true
    )
    test_scores = np.linspace(-3.0, 5.0, 101)
    np.testing.assert_allclose(
        calibration.transform(test_scores), expected.predict(test_scores), atol=1e-8
    )
