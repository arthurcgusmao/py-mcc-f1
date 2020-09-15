# MCC-F1 Python package

Recently, the MCC-F1 curve has been proposed as an alternative, better way of assessing the performance of score-based binary classifiers [1].

This Python package implements a function to compute the MCC-F1 curve, namely `mcc_f1_curve`, similarly to the `precision_recall_curve` and `roc_curve` functions of [scikit-learn](https://github.com/scikit-learn).


## Installation

```console
pip install py-mcc-f1
```

## Usage

```python
from mcc_f1 import mcc_f1_curve

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data and train model
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
clf = LogisticRegression().fit(X_train, y_train)

# Get predictions and MCC-F1 curve
y_score = clf.predict_proba(X_test)[:,1]
mcc, f1, thresholds = mcc_f1_curve(y_test, y_score)

# Plot MCC-F1 curve
plt.figure(figsize=(6,6))
plt.plot(f1, mcc)
plt.xlim(0,1)
plt.ylim(0,1)
```

Please refer to the function's docstring for further comments and details.


## Future enhancements

- [ ] Function to plot the MCC-F1 curve, (e.g., `plot_precision_recall_curve`), similar to `sklearn/metrics/_plot/precision_recall_curve.py` and `sklearn/metrics/_plot/roc_curve.py`;
- [ ] Function to compute the MCC-F1 metric, as defined in section 2.2 of the original paper.


## Contributing

If you would like to contribute to this package, please follow the [common community guidelines](https://github.com/MarcDiethelm/contributing).

Please, also keep in mind that the main goal of this project is to be of similar implementation and quality as scikit-learn. Pull requests should pass the existing unit-tests, and add new ones when necessary.

To run the tests:

```console
make test
```

## License

This package is distributed under the [MIT license](./LICENSE).

## References

1. [[2006.11278] The MCC-F1 curve: a performance evaluation technique for binary classification](https://arxiv.org/abs/2006.11278)
