import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.conftest import pyplot

from mcc_f1 import mcc_f1_curve
from mcc_f1 import MCCF1CurveDisplay
from mcc_f1 import plot_mcc_f1_curve

# TODO: Remove when https://github.com/numpy/numpy/issues/14397 is resolved
pytestmark = pytest.mark.filterwarnings(
    "ignore:In future, it will be an error for 'np.bool_':DeprecationWarning:"
    "matplotlib.*")


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]


def test_plot_mcc_f1_curve_error_non_binary(pyplot, data):
    X, y = data
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    msg = "DecisionTreeClassifier should be a binary classifier"
    with pytest.raises(ValueError, match=msg):
        plot_mcc_f1_curve(clf, X, y)


@pytest.mark.parametrize(
    "response_method, msg",
    [("predict_proba", "response method predict_proba is not defined in "
                       "MyClassifier"),
     ("decision_function", "response method decision_function is not defined "
                           "in MyClassifier"),
     ("auto", "response method decision_function or predict_proba is not "
              "defined in MyClassifier"),
     ("bad_method", "response_method must be 'predict_proba', "
                    "'decision_function' or 'auto'")])
def test_plot_mcc_f1_curve_error_no_response(pyplot, data_binary,
                                             response_method, msg):
    X, y = data_binary

    class MyClassifier(ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(ValueError, match=msg):
        plot_mcc_f1_curve(clf, X, y, response_method=response_method)


@pytest.mark.parametrize("response_method",
                         ["predict_proba", "decision_function"])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("with_strings", [True, False])
def test_plot_mcc_f1_curve(pyplot, response_method, data_binary,
                           with_sample_weight, with_strings):
    X, y = data_binary

    pos_label = None
    if with_strings:
        y = np.array(["c", "b"])[y]
        pos_label = "c"

    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=(X.shape[0]))
    else:
        sample_weight = None

    lr = LogisticRegression()
    lr.fit(X, y)

    viz = plot_mcc_f1_curve(lr, X, y, alpha=0.8, sample_weight=sample_weight)

    y_pred = getattr(lr, response_method)(X)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    mcc, f1, _ = mcc_f1_curve(y, y_pred, sample_weight=sample_weight,
                              pos_label=pos_label)

    # assert_allclose(viz.mcc_f1, mcc_f1_metric(mcc, f1)) # @TODO: uncomment when metric implemented
    assert_allclose(viz.mcc, mcc)
    assert_allclose(viz.f1, f1)

    assert viz.estimator_name == "LogisticRegression"

    # cannot fail thanks to pyplot fixture
    import matplotlib as mpl  # noqal
    assert isinstance(viz.line_, mpl.lines.Line2D)
    assert viz.line_.get_alpha() == 0.8
    assert isinstance(viz.ax_, mpl.axes.Axes)
    assert isinstance(viz.figure_, mpl.figure.Figure)

    # @TODO: uncomment when metric is implemented
    # expected_label = "LogisticRegression (MCC-F1 = {:0.2f})".format(viz.mcc_f1)
    # assert viz.line_.get_label() == expected_label

    expected_pos_label = 1 if pos_label is None else pos_label
    expected_ylabel = f"MCC (Positive label: " \
                      f"{expected_pos_label})"
    expected_xlabel = f"F1-Score (Positive label: " \
                      f"{expected_pos_label})"

    assert viz.ax_.get_ylabel() == expected_ylabel
    assert viz.ax_.get_xlabel() == expected_xlabel


@pytest.mark.parametrize(
    "clf", [LogisticRegression(),
            make_pipeline(StandardScaler(), LogisticRegression()),
            make_pipeline(make_column_transformer((StandardScaler(), [0, 1])),
                          LogisticRegression())])
def test_mcc_f1_curve_not_fitted_errors(pyplot, data_binary, clf):
    X, y = data_binary
    with pytest.raises(NotFittedError):
        plot_mcc_f1_curve(clf, X, y)
    clf.fit(X, y)
    disp = plot_mcc_f1_curve(clf, X, y)
    assert clf.__class__.__name__ in disp.line_.get_label()
    assert disp.estimator_name == clf.__class__.__name__


def test_plot_mcc_f1_curve_estimator_name_multiple_calls(pyplot, data_binary):
    # non-regression test checking that the `name` used when calling
    # `plot_roc_curve` is used as well when calling `disp.plot()`
    X, y = data_binary
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    disp = plot_mcc_f1_curve(clf, X, y, name=clf_name)
    assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    assert clf_name in disp.line_.get_label()


@pytest.mark.parametrize(
    "mcc_f1, estimator_name, expected_label",
    [
        (0.9, None, "MCC-F1 = 0.90"),
        (None, "my_est", "my_est"),
        (0.8, "my_est2", "my_est2 (MCC-F1 = 0.80)")
    ]
)
def test_default_labels(pyplot, mcc_f1, estimator_name,
                        expected_label):
    mcc = np.array([0.5, 0.75, 1])
    f1 = np.array([0, 0.5, 1])
    disp = MCCF1CurveDisplay(mcc=mcc, f1=f1, mcc_f1=mcc_f1,
                             estimator_name=estimator_name).plot()
    assert disp.line_.get_label() == expected_label

### @TODO: implement when metric implemented
# @pytest.mark.parametrize(
#     "response_method", ["predict_proba", "decision_function"]
# )
# def test_plot_mcc_f1_curve_pos_label(pyplot, response_method):
#     # check that we can provide the positive label and display the proper
#     # statistics
#     X, y = load_breast_cancer(return_X_y=True)
#     # create an highly imbalanced
#     idx_positive = np.flatnonzero(y == 1)
#     idx_negative = np.flatnonzero(y == 0)
#     idx_selected = np.hstack([idx_negative, idx_positive[:25]])
#     X, y = X[idx_selected], y[idx_selected]
#     X, y = shuffle(X, y, random_state=42)
#     # only use 2 features to make the problem even harder
#     X = X[:, :2]
#     y = np.array(
#         ["cancer" if c == 1 else "not cancer" for c in y], dtype=object
#     )
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, random_state=0,
#     )

#     classifier = LogisticRegression()
#     classifier.fit(X_train, y_train)

#     # sanity check to be sure the positive class is classes_[0] and that we
#     # are betrayed by the class imbalance
#     assert classifier.classes_.tolist() == ["cancer", "not cancer"]

#     disp = plot_mcc_f1_curve(
#         classifier, X_test, y_test, pos_label="cancer",
#         response_method=response_method
#     )

#     roc_auc_limit = 0.95679

#     assert disp.roc_auc == pytest.approx(roc_auc_limit)
#     assert np.trapz(disp.tpr, disp.fpr) == pytest.approx(roc_auc_limit)

#     disp = plot_mcc_f1_curve(
#         classifier, X_test, y_test,
#         response_method=response_method,
#     )

#     assert disp.roc_auc == pytest.approx(roc_auc_limit)
#     assert np.trapz(disp.tpr, disp.fpr) == pytest.approx(roc_auc_limit)
