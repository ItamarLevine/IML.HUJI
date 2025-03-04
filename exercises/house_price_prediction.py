from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import loss_functions
from typing import NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    # erase negative prices
    full_data = full_data.drop(np.where(full_data["price"].values <= 0)[0])
    # feature i dropped: "lat","long", "id", "date",waterfront
    features = full_data[["bedrooms","bathrooms","sqft_living","sqft_lot","floors","view"
        ,"condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode"
        ,"sqft_living15","sqft_lot15"]]
    features = pd.concat([features,pd.get_dummies(full_data[["waterfront"]])],axis=1)
    labels = full_data["price"]
    return features, labels


def clean_big_deviation(X, k):
    for feature in X.columns.values:
        bad_ind = np.where(np.abs(X[feature] - X[feature].mean()) > k * X[feature].std())[0]
        good_ind = np.setdiff1d(np.arange(len(X)), bad_ind)
        X[feature].values[bad_ind.astype(int)] = X[feature].values[good_ind.astype(int)].mean()
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    cov = np.cov(np.hstack((X,np.array(y).reshape((-1,1)))), rowvar=False)
    variance = np.diagonal(cov)
    pearson_correlation = cov[-1] / ((variance[-1] * variance) ** 0.5)
    pearson_correlation = pearson_correlation[:-1]
    features = ['grade','condition']
    for feature in features:
        index = np.where(X.columns.values == feature)[0]
        graph = plt.figure()
        plt.title(f"the pearson's correlation for {feature} is {pearson_correlation[index]}\n")
        plt.xlabel("feature's data")
        plt.ylabel("price")
        plt.plot(X[feature].values,y.values, 'o',color='blue')
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, response = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, response)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(df, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    graph = plt.figure()
    plt.title('Average loss as function of training size')
    plt.ylabel('average loss')
    plt.xlabel("sample's size")
    std_vec = np.zeros(91)
    loss_vec_vec = np.zeros(91)
    for i in range(10,101):
        # index = int(np.ceil(len(train_x) * i / 100))
        loss_vec = np.zeros(10)
        for j in range(10):
            p_train_x, p_train_y, _, __ = split_train_test(train_x, train_y, train_proportion=i/100)
            lr.fit(p_train_x, p_train_y)
            loss_vec[j] = lr.loss(test_x, test_y)
        loss = loss_vec.mean()
        std_loss = loss_vec.std()
        std_vec[i-10] = std_loss
        loss_vec_vec[i-10] = loss
        print(loss)
    plt.plot((np.arange(91)+10)/100, loss_vec_vec, color='blue')
    plt.fill_between((np.arange(91)+10)/100, loss_vec_vec - 2 * std_vec, loss_vec_vec + 2 * std_vec, alpha=0.2)
    plt.show()

