from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    f_x = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    eps = np.random.normal(0,noise,n_samples)
    noisy_model = f_x + eps
    train_x, train_y, test_x, test_y = split_train_test(x, noisy_model, 2/3)
    train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
    plt.Figure()
    plt.title(f'f(x) with noise={noise} and {n_samples} samples')
    plt.ylabel('f(x)')
    plt.xlabel("x")
    plt.scatter(x, f_x, label="noiseless data",s=5)
    plt.scatter(train_x,train_y, label="train data",s=5)
    plt.scatter(test_x,test_y, label="test data",s=5)
    plt.legend(loc='best')
    plt.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    plt.Figure()
    plt.title(f"Polynomial fitting's cross validation with noise={noise} and {n_samples} samples")
    plt.ylabel('loss')
    plt.xlabel("Polynomial fitting's rank")
    train_losses = []
    validation_losses = []
    for k in range(11):
        losses = cross_validate(PolynomialFitting(k),train_x,train_y,mean_square_error)
        train_losses.append(losses[0])
        validation_losses.append((losses[1]))
    plt.plot(np.arange(11), train_losses, label="train loss")
    plt.plot(np.arange(11), validation_losses, label="validation loss")
    plt.legend(loc='best')
    plt.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_losses)
    best_poly = PolynomialFitting(int(k_star))
    best_poly.fit(train_x,train_y)
    print(f"best polynomial's rank is {k_star}")
    print(f"CV's validation error for rank {k_star} is: {validation_losses[k_star]}")
    print(f"regular Polynomial fitting model with rank {k_star}'s test error is: {mean_square_error(test_y,best_poly.predict(test_x))}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, test_x, train_y, test_y = X[:n_samples], X[n_samples:], y[:n_samples], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range = np.linspace(0,1, n_evaluations)
    train_losses = []
    validation_losses = []
    plt.Figure()
    for lam in lambda_range:
        losses = cross_validate(RidgeRegression(lam),train_x,train_y,mean_square_error)
        train_losses.append(losses[0])
        validation_losses.append((losses[1]))
    plt.plot(lambda_range, train_losses, label="Ridge train loss")
    plt.plot(lambda_range, validation_losses, label="Ridge validation loss")
    plt.legend(loc='best')
    plt.show()
    ridge_lam_star = lambda_range[np.argmin(validation_losses)]

    train_losses = []
    validation_losses = []
    plt.Figure()
    for lam in lambda_range:
        losses = cross_validate(Lasso(lam),train_x,train_y,mean_square_error)
        train_losses.append(losses[0])
        validation_losses.append((losses[1]))
    plt.plot(lambda_range, train_losses, label="Lasso train loss")
    plt.plot(lambda_range, validation_losses, label="Lasso validation loss")
    plt.legend(loc='best')
    plt.show()
    lasso_lam_star = lambda_range[np.argmin(validation_losses)]
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    print(f"best lambda for Ridge regression is {ridge_lam_star}")
    print(f"best lambda for Lasso regression is {lasso_lam_star}")
    lr = LinearRegression().fit(X,y)
    lasso = Lasso(lasso_lam_star).fit(X,y)
    ridge = RidgeRegression(float(ridge_lam_star)).fit(X,y)
    print(f"MSE for Linear regression is {mean_square_error(y,lr.predict(X))}")
    print(f"MSE for Ridge regression with lambda {ridge_lam_star} is {mean_square_error(y,ridge.predict(X))}")
    print(f"MSE for Lasso regression lambda {lasso_lam_star} is {mean_square_error(y,lasso.predict(X))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()