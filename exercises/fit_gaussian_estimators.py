from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import datetime


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    a = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()
    ug.fit(a)
    print(ug.mu_, ug.var_)

    # Question 2 - Empirically showing sample mean is consistent
    graph = plt.figure()
    plt.title('Accurate of expectation')
    plt.ylabel('bias from expectation')
    plt.xlabel("sample's size")
    for i in range(10,1010,10):
        ug.fit(a[:i])
        plt.plot(i, np.abs(10 - ug.mu_), 'o', color='black')
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

    # quiz
    x = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,-4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # ug = UnivariateGaussian()
    # ug.fit(x)
    print(ug.log_likelihood(1,1,x))
    print(ug.log_likelihood(10, 1, x))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    a = np.random.multivariate_normal([0, 0, 4, 0], cov_matrix, 1000)
    mg = MultivariateGaussian()
    mg.fit(a)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    stime = datetime.datetime.now()
    f1 = np.linspace(-10, 10, 200)
    # create all mu model
    all_pairs = np.array(np.meshgrid(f1, f1)).T.reshape(-1, 2)
    all_mu = np.zeros((all_pairs.shape[0], 4))
    all_mu[:, 0] = all_pairs[:, 0]
    all_mu[:, 2] = all_pairs[:, 1]
    # calculate likelihood to all model
    all_likelihood = np.apply_along_axis(mg.log_likelihood, 1, all_mu, cov_matrix, a).reshape((200,200))
    # display heatmap
    fig, ax = plt.subplots()
    c = ax.pcolormesh(f1, f1, all_likelihood, cmap='RdBu', vmin=np.min(all_likelihood), vmax=np.max(all_likelihood),shading='auto')
    ax.axis([np.min(f1), np.max(f1), np.min(f1), np.max(f1)])
    fig.colorbar(c, ax=ax)
    plt.show()
    etime = datetime.datetime.now()
    print(f"question 5 took {etime-stime} seconds")

    # Question 6 - Maximum likelihood
    all_pairs = all_pairs.reshape((200,200,2))
    max_model = all_pairs[all_likelihood == all_likelihood.max()][0]
    print(max_model)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()