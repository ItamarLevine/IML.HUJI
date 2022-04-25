import numpy as np
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import matplotlib.pyplot as plt
from IMLearn.metrics import loss_functions
from math import atan2


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        loss_callback = lambda perc, x, y: losses.append(perc._loss(x, y))
        perceptron = Perceptron(callback=loss_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        graph = plt.figure()
        plt.title(f"perceptron's loss as function of number of iteration\n in {n} space")
        plt.ylabel('loss')
        plt.xlabel("iterations")
        plt.plot(np.arange(len(losses)),losses)
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray, fig):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    fig.scatter(mu[0], mu[1], marker='x',color='black',s=20)
    return fig.scatter(x=mu[0] + xs, y=mu[1] + ys,s=3)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig.suptitle(f"compare LDA and GNB prediction on dataset: {f}")
        fig.supxlabel("X axis")
        fig.supylabel("Y axis")
        fig.set_figwidth(12)
        for i,model,name in ((0,lda, "LDA"),(1,gnb,"GNB")):
            pred = model.predict(X)
            acc = loss_functions.accuracy(y, pred)
            ax[i].title.set_text(f"{name} accuracy is {acc}")
            # Add traces for data-points setting symbols and colors
            for c in np.unique(y):
                for d in np.unique(y):
                    if c == 0:
                        shape = "."
                    elif c == 1:
                        shape = "^"
                    else:
                        shape = "s"
                    if d == 0:
                        color = "blue"
                    elif d == 1:
                        color = "red"
                    else:
                        color = "green"
                    ind_true = np.where(y == c)[0]
                    ind_pred = np.where(pred == d)[0]
                    ind = np.intersect1d(ind_pred,ind_true)
                    samples = X[ind]
                    ax[i].scatter(samples[:,0], samples[:,1], marker=shape, color=color)
                # Add `X` dots specifying fitted Gaussians' means
                # Add ellipses depicting the covariances of the fitted Gaussians
                if model == lda:
                    cov = model.cov_
                else:
                    cov = np.linalg.inv(np.diag(model.vars_[c].reshape(model.vars_.shape[1], )))
                get_ellipse(model.mu_[c], cov, ax[i])
        # fig.legend(loc='lower left')
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
