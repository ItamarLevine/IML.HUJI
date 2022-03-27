from __future__ import annotations
from typing import NoReturn
import numpy.linalg
from IMLearn.base import BaseEstimator
from IMLearn.metrics import loss_functions
import numpy as np
from numpy.linalg import pinv


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if not self.include_intercept_:
            X = np.hstack((np.ones(len(X)).reshape((-1,1)),X))
        u, sigma,v = numpy.linalg.svd(X)
        sigma_cross = np.zeros(X.shape)
        sigma_cross[:len(sigma),:len(sigma)] = np.diag(1/sigma)
        x_cross = v.T @ sigma_cross.T @ u.T
        self.coefs_ = x_cross @ y
        print(self.coefs_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.include_intercept_:
            X = np.hstack((np.ones(len(X)).reshape((-1,1)),X))
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return loss_functions.mean_square_error(y, self._predict(X))


# lr = LinearRegression(False)
# met = np.array([[2,1],[3,4],[2,2]])
# lr._fit(met, np.array([6,11,7]))
# vec_to_pre = np.arange(12).reshape((-1,3))
# print(lr._predict(met))
#print(lr._loss(met, np.arange(6)))