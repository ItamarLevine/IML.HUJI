import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ad = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    graph = plt.figure()
    plt.title('AdaBoost loss in function of learners')
    plt.ylabel('loss')
    plt.xlabel("number of learners")
    train_loss = []
    test_loss = []
    for i in range(1, n_learners):
        train_loss.append(ad.partial_loss(train_X, train_y, i))
        test_loss.append(ad.partial_loss(test_X, test_y, i))
    plt.plot(np.arange(n_learners - 1) + 1, train_loss, label="train data")
    plt.plot(np.arange(n_learners - 1) + 1, test_loss, label="test data")
    plt.legend(loc='right')
    plt.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("decision boundary of AdaBoost with different number of learners")
    cm = ListedColormap(['#FF0000','#0000FF'])
    min_t, min_error = np.inf, np.inf
    for i, t in enumerate(T):
        d_s(lambda x: ad.partial_predict(x, t), lims[0], lims[1], ax[i // 2][i % 2])
        ax[i // 2][i % 2].scatter(test_X[:, 0], test_X[:, 1], c=test_y, s=2, cmap=cm)
        ax[i // 2][i % 2].title.set_text(f"{t} fitted learners")
        ax[i // 2][i % 2].axis('off')
        loss = ad.partial_loss(test_X, test_y, t)
        if loss < min_error:
            min_error = loss
            min_t = t
    plt.show()

    # Question 3: Decision surface of best performing ensemble
    graph = plt.figure()
    plt.title(f"best error got from {min_t} learners, and it got loss: {min_error}")
    d_s(lambda x: ad.partial_predict(x, min_t), lims[0], lims[1], plt)
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, s=2, cmap=cm)
    plt.show()
    # Question 4: Decision surface with weighted samples
    graph = plt.figure()
    plt.title(f"decision surface where labels with weights")
    d = (ad.D_ / np.max(ad.D_)) * 5
    d_s(lambda x: ad.partial_predict(x, 250), lims[0], lims[1], plt)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, s=d, cmap=cm)
    plt.show()


def d_s(predict, xrange, yrange, plot, density=120):
    cm_b = ListedColormap(['#FFAAAA','#AAAAFF'])
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])
    plot.pcolormesh(xx, yy, pred.reshape(xx.shape), cmap=cm_b, shading='auto')


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
