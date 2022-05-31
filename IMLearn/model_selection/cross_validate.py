from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = []
    for i in range(cv):
        folds.append(np.arange(len(X))[int(X.shape[0] * i / cv):int(X.shape[0] * (i + 1) / cv)])
    losses_train = []
    losses_validation = []
    for i in range(cv):
        index = np.setdiff1d(np.arange(len(X)), folds[i])
        fitted = estimator.fit(X[index], y[index])
        losses_train.append(scoring(y[index], fitted.predict(X[index])))
        losses_validation.append(scoring(y[folds[i]], fitted.predict(X[folds[i]])))
    return np.array(losses_train).mean(), np.array(losses_validation).mean()
