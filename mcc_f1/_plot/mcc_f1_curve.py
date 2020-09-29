try:
    from sklearn.metrics._plot.base import _get_response
except ImportError:
    # Function was not present in sklearn.metrics._plot.base before c3f2516
    from .base import _get_response

from sklearn.utils import check_matplotlib_support

# from .. import mcc_f1_metric, mcc_f1_curve
from .. import mcc_f1_curve


class MCCF1CurveDisplay:
    """MCC-F1 Curve visualization.

    It is recommend to use :func:`~mcc_f1.plot_mcc_f1_curve` to create a
    visualizer. All parameters are stored as attributes.

    Read more in scikit-learn's :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    f1 : ndarray
        F1-Score.

    mcc : ndarray
        Matthews Correlation Coefficient.

    mcc_f1 : float, default=None
        MCC-F1 metric. If None, the mcc_f1 score is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : str or int, default=None
        The class considered as the positive class when computing the metrics.
        By default, `estimators.classes_[1]` is considered as the positive
        class.

        .. versionadded:: 0.24

    Attributes
    ----------
    line_ : matplotlib Artist
        MCC-F1 Curve.

    ax_ : matplotlib Axes
        Axes with MCC-F1 Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> from mcc_f1 import mcc_f1_curve, mcc_f1_metric, MCCF1CurveDisplay
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> f1, mcc, thresholds = mcc_f1_curve(y, pred)
    >>> mcc_f1 = mcc_f1_metric(f1, mcc)
    >>> display = MCCF1CurveDisplay(f1=f1, mcc=mcc, mcc_f1=mcc_f1,\
                                    estimator_name='example estimator')
    >>> display.plot()  # doctest: +SKIP
    >>> plt.show()      # doctest: +SKIP
    """

    def __init__(self, *, f1, mcc,
                 mcc_f1=None, estimator_name=None, pos_label=None):
        self.estimator_name = estimator_name
        self.f1 = f1
        self.mcc = mcc
        self.mcc_f1 = mcc_f1
        self.pos_label = pos_label

    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support('MCCF1CurveDisplay.plot')

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.mcc_f1 is not None and name is not None:
            line_kwargs["label"] = f"{name} (MCC-F1 = {self.mcc_f1:0.2f})"
        elif self.mcc_f1 is not None:
            line_kwargs["label"] = f"MCC-F1 = {self.mcc_f1:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.figure import figaspect

        if ax is None:
            fig, ax = plt.subplots(figsize=figaspect(1.))

        self.line_, = ax.plot(self.f1, self.mcc, **line_kwargs)
        info_pos_label = (f" (Positive label: {self.pos_label})"
                          if self.pos_label is not None else "")

        xlabel = "F1-Score" + info_pos_label
        ylabel = "MCC" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0,1), ylim=(0,1))

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def plot_mcc_f1_curve(estimator, X, y, *, sample_weight=None,
                      response_method="auto", name=None, ax=None,
                      pos_label=None, **kwargs):
    """Plot MCC-F1 curve.

    Extra keyword arguments will be passed to matplotlib's `plot`.

    Read more in scikit-learn's :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    response_method : {'predict_proba', 'decision_function', 'auto'} \
    default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.

    name : str, default=None
        Name of MCC-F1 Curve for labeling. If `None`, use the name of the
        estimator.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.

    pos_label : str or int, default=None
        The class considered as the positive class when computing the metrics.
        By default, `estimators.classes_[1]` is considered as the positive
        class.

        .. versionadded:: 0.24

    Returns
    -------
    display : :class:`~sklearn.metrics.MCCF1CurveDisplay`
        Object that stores computed values.

    See Also
    --------
    mcc_f1_metric : Compute the MCC-F1 metric

    mcc_f1_curve : Compute the MCC-F1 curve

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from sklearn import datasets, metrics, model_selection, svm
    >>> from mcc_f1 import plot_mcc_f1_curve
    >>> X, y = datasets.make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
    ...     X, y, random_state=0)
    >>> clf = svm.SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> plot_mcc_f1_curve(clf, X_test, y_test)  # doctest: +SKIP
    >>> plt.show()                              # doctest: +SKIP
    """
    check_matplotlib_support('plot_mcc_f1_curve')

    y_pred, pos_label = _get_response(
        X, estimator, response_method, pos_label=pos_label)

    mcc, f1, _ = mcc_f1_curve(y, y_pred, pos_label=pos_label,
                              sample_weight=sample_weight)
    # mcc_f1 = mcc_f1_metric(f1, mcc)
    mcc_f1 = None

    name = estimator.__class__.__name__ if name is None else name

    viz = MCCF1CurveDisplay(
        f1=f1,
        mcc=mcc,
        mcc_f1=mcc_f1,
        estimator_name=name,
        pos_label=pos_label
    )

    return viz.plot(ax=ax, name=name, **kwargs)
