import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from sklearn.metrics.tests.test_ranking import make_prediction
from sklearn.utils.validation import check_consistent_length

from mcc_f1 import mcc_f1_curve


def test_mcc_f1_curve():
    # Test MCC and F1 values for all points of the curve
    y_true, _, probas_pred = make_prediction(binary=True)
    mcc, f1, thres = mcc_f1_curve(y_true, probas_pred)
    check_consistent_length(mcc, f1, thres)

    expected_mcc, expected_f1 = _mcc_f1_calc(y_true, probas_pred, thres)
    assert_array_almost_equal(f1, expected_f1)
    assert_array_almost_equal(mcc, expected_mcc)


def _mcc_f1_calc(y_true, probas_pred, thresholds):
    # Alternative calculation of (unit-normalized) MCC and F1 scores
    pp = probas_pred
    ts = thresholds
    tps = np.array([np.logical_and(pp >= t, y_true == 1).sum() for t in ts])
    fps = np.array([np.logical_and(pp >= t, y_true == 0).sum() for t in ts])
    tns = np.array([np.logical_and(pp < t, y_true == 0).sum() for t in ts])
    fns = np.array([np.logical_and(pp < t, y_true == 1).sum() for t in ts])

    with np.errstate(divide='ignore', invalid='ignore'):
        f1s = 2*tps / (2*tps + fps + fns)

        d = np.sqrt((tps+fps)*(tps+fns)*(tns+fps)*(tns+fns))
        d = np.array([1 if di == 0 else di for di in d])
        mccs = (tps*tns - fps*fns) / d
        mccs = (mccs + 1) / 2   # Unit-normalize MCC

    return mccs, f1s


def test_mcc_f1_curve_errors():
    # Contains non-binary labels
    with pytest.raises(ValueError):
        mcc_f1_curve([0, 1, 2], [[0.0], [1.0], [1.0]])


def test_mcc_f1_curve_threshold_range():
    # Check a large enough range of thresholds was used
    y_true, _, probas_pred = make_prediction(binary=True)
    _, _, thres = mcc_f1_curve(y_true, probas_pred)
    score_range = probas_pred.max() - probas_pred.min()
    thres_range = thres.max() - thres.min()
    assert thres_range >= score_range


def test_mcc_f1_curve_toydata():
    with np.errstate(all="raise"):
        y_true = [0, 1]
        y_score = [0, 1]
        mcc, f1, _ = mcc_f1_curve(y_true, y_score)
        assert_array_almost_equal(mcc, [1, .5])
        assert_array_almost_equal(f1, [1, 2/3])

        y_true = [0, 1]
        y_score = [1, 0]
        mcc, f1, _ = mcc_f1_curve(y_true, y_score)
        assert_array_almost_equal(mcc, [0, .5])
        assert_array_almost_equal(f1, [0, 2/3])

        y_true = [1, 0]
        y_score = [1, 1]
        mcc, f1, _ = mcc_f1_curve(y_true, y_score)
        assert_array_almost_equal(mcc, [.5])
        assert_array_almost_equal(f1, [2/3])

        y_true = [1, 0]
        y_score = [1, 0]
        mcc, f1, _ = mcc_f1_curve(y_true, y_score)
        assert_array_almost_equal(mcc, [1, .5])
        assert_array_almost_equal(f1, [1, 2/3])

        y_true = [1, 0]
        y_score = [0.5, 0.5]
        mcc, f1, _ = mcc_f1_curve(y_true, y_score)
        assert_array_almost_equal(mcc, [.5])
        assert_array_almost_equal(f1, [2/3])

        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with pytest.raises(ValueError):
            mcc_f1_curve(y_true, y_score)

        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with pytest.raises(ValueError):
            mcc_f1_curve(y_true, y_score)
