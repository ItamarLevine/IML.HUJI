from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    a = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()
    ug.fit(a)

    # Question 2 - Empirically showing sample mean is consistent
    graph = plt.figure()
    plt.title('Accurate of expectation')
    plt.ylabel('bias from expectation')
    plt.xlabel("sample's size")
    for i in range(100):
        vec = np.random.normal(10, 1, (i + 1) * 10)
        ug.fit(vec)
        plt.plot((i + 1) * 10, np.abs(10 - ug.mu_), 'o', color='black')
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    graph = plt.figure()
    plt.title('Gaussian PDF')
    plt.ylabel('PDF')
    plt.xlabel("sample's value")
    ug.fit(a)
    pdf = ug.pdf(a)
    plt.plot(a, pdf, 'o', color='black')
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    a = np.random.multivariate_normal([0, 0, 4, 0], cov_matrix, 1000)
    mg = MultivariateGaussian()
    mg.fit(a)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
