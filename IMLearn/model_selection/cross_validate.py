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
        folds.append((X[int(X.shape[0]*i/cv):int(X.shape[0]*(i+1)/cv)], y[int(y.shape[0]*i/cv):int(y.shape[0]*(i+1)/cv)]))
    losses_train = []
    losses_validation = []
    for i in range(cv):
        if len(X.shape) == 2:
            comb_x = np.empty((int(((cv-1)/cv)*X.shape[0]),X.shape[1]))
        else:
            comb_x = np.empty((int(((cv-1)/cv)*X.shape[0])))
        comb_y = np.empty((int(((cv - 1) / cv) * y.shape[0])))
        for j in range(cv):
            if i > j:
                start_index = int(j*comb_x.shape[0]/(cv-1))
                # end_index = int((j+1)*comb_x.shape[0]/(cv-1))
                end_index = start_index + len(folds[j][0])
                comb_x[start_index:end_index] = folds[j][0]
                comb_y[start_index:end_index] = folds[j][1]
            if i < j:
                start_index = int((j-1) * comb_x.shape[0] / (cv - 1))
                # end_index = int(j * comb_x.shape[0] / (cv - 1))
                end_index = start_index + len(folds[j][0])
                comb_x[start_index:end_index] = folds[j][0]
                comb_y[start_index:end_index] = folds[j][1]
        fitted = estimator.fit(comb_x, comb_y)
        losses_train.append(scoring(comb_y, fitted.predict(comb_x)))
        losses_validation.append(scoring(folds[i][1], fitted.predict(folds[i][0])))
    return np.array(losses_train).mean(), np.array(losses_validation).mean()
