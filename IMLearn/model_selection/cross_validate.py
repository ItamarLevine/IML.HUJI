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
    main_x = X
    main_y = y
    for i in range(cv):
        fold_x, fold_y, main_x, main_y = split_train_test(main_x, main_y, 1 / (cv - i))
        folds.append((fold_x, fold_y))
    losses_train = []
    losses_validation = []
    for i in range(cv):
        comb_x = np.empty((((cv-1)/cv)*X.shape[0],X.shape[1]))
        comb_y = np.empty((((cv - 1) / cv) * y.shape[0], y.shape[1]))
        for j in range(cv):
            if i > j:
                comb_x[j*comb_x.shape[0]/(cv-1):(j+1)*comb_x.shape[0]/(cv-1) + 1] = folds[j][0]
                comb_y[j*comb_x.shape[0]/(cv-1):(j+1)*comb_x.shape[0]/(cv-1) + 1] = folds[j][1]
            if i < j:
                comb_x[(j-1)*comb_x.shape[0]/(cv-1):j*comb_x.shape[0]/(cv-1) + 1] = folds[j][0]
                comb_y[(j-1)*comb_x.shape[0]/(cv-1):j*comb_x.shape[0]/(cv-1) + 1] = folds[j][1]
        losses_train.append(scoring(folds[i][1], estimator.fit(comb_x, comb_y).predict(comb_x)))
        losses_validation.append(scoring(folds[i][1], estimator.fit(comb_x, comb_y).predict(folds[i][0])))
    return np.array(losses_train).mean(), np.array(losses_validation).mean()
