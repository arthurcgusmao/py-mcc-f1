import numpy as np

from sklearn.metrics._ranking import _binary_clf_curve


def mcc_f1_curve(y_true, y_score, *, pos_label=None, sample_weight=None,
                 unit_normalize_mcc=True):
    """Compute the MCC-F1 curve

    The MCC-F1 curve combines the Matthews correlation coefficient and the
    F1-Score to clearly differentiate good and bad *binary* classifiers,
    especially with imbalanced ground truths.

    It has been recently proposed as a better alternative for the receiver
    operating characteristic (ROC) and the precision-recall (PR) curves.
    [1]

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function.

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    unit_normalize_mcc : bool, default=True
        Whether to unit-normalize the MCC values, as in the original paper.

    Returns
    -------
    mcc : array, shape = [n_thresholds]
        MCC values such that element i is the MCC of predictions with
        score >= thresholds[i].

    f1 : array, shape = [n_thresholds]
        F1-Score values such that element i is the F1-Score of predictions with
        score >= thresholds[i].

    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        MCC and F1.

    Notes
    -----
    Differently from the original MCC-F1 curve proposal, this implementation
    returns the correct limiting unit-normalized MCC value of 0.5 (or 0 for the
    non-unit-normalized) when its denominator is zero (MCC = 0/0), by
    arbitrarily setting the denominator to 1 in such cases, as suggested in
    [2].

    References
    ----------
    .. [1] `Chang Cao and Davide Chicco and Michael M. Hoffman. (2020)
            The MCC-F1 curve: a performance evaluation technique for binary
            classification.
            <https://arxiv.org/pdf/2006.11278>`_
    .. [2] `Wikipedia entry for the Matthews correlation coefficient
            <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    ps = tps + fps              # Array of total positive predictions
    p = tps[-1]                 # No of positives in ground truth
    n = fps[-1]                 # No of negatives in ground truth

    if p == 0:
        raise ValueError("No positive samples in y_true, "
                         "MCC and F1 are undefined.")
    if n == 0:
        raise ValueError("No negative samples in y_true, "
                         "MCC is undefined.")

    # Compute MCC
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = np.sqrt(p*n*ps*(p + n - ps))
        denominator[denominator == 0] = 1.
        mccs = (n*tps - p*fps) / denominator
    if unit_normalize_mcc:
        mccs = (mccs + 1) / 2   # Unit-normalize MCC values

    # Compute F1
    f1s = 2*tps / (ps + p)

    return mccs, f1s, thresholds
