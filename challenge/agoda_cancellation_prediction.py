from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    # change nan to 0
    full_data = full_data.fillna(0)
    # change dates to numbers
    index_date = np.flatnonzero(np.core.defchararray.find(full_data.columns.values.astype(str), "date") != -1)
    vec_date_to_numeric = np.vectorize(date_to_numeric)
    mat = vec_date_to_numeric(full_data.values[:, index_date.astype(int)])
    for i, feature in enumerate(full_data.columns.values[index_date]):
        full_data = full_data.drop(feature, axis=1)
        full_data.insert(int(index_date[i]), feature, mat[:, i], True)
    # choose the relevant feature, dropped: h_booking_id, h_customer_id 'hotel_area_code', 'hotel_brand_code',
    #                                       'hotel_chain_code', 'hotel_city_code', original_payment_currency
    #                                       'hotel_country_code', 'accommadation_type_name', 'customer_nationality',
    #                                        'guest_nationality_country_name',  'origin_country_code', 'language',
    #                                        'original_payment_method', , 'request_airport'
    features = full_data[['booking_datetime', 'checkin_date', 'checkout_date',
                          'hotel_id', 'hotel_live_date', 'hotel_star_rating',
                          'guest_is_not_the_customer',
                          'no_of_adults', 'no_of_children',
                          'no_of_extra_bed', 'no_of_room',
                          'original_selling_amount', 'is_user_logged_in',
                          'cancellation_policy_code', 'is_first_booking', 'request_nonesmoke',
                          'request_latecheckin', 'request_highfloor', 'request_largebed',
                          'request_twinbeds', 'request_earlycheckin']]
    features = pd.concat([features, pd.get_dummies(full_data[['accommadation_type_name']])], axis=1)
    features = pd.concat([features, pd.get_dummies(full_data[['original_payment_method']])], axis=1)
    features = pd.concat([features, pd.get_dummies(full_data[['guest_nationality_country_name']])], axis=1)
    features = pd.concat([features, pd.get_dummies(full_data[['charge_option']])], axis=1)
    features = pd.concat([features, pd.get_dummies(full_data[['original_payment_type']])], axis=1)
    #labels = np.zeros(len(full_data["cancellation_datetime"]))
    #labels[np.where(full_data["cancellation_datetime"].values != 0)[0]] = 1
    labels = full_data["cancellation_datetime"]
    features = parse_cancellation_policy(features)
    return features, labels


def date_to_numeric(date):
    if date == 0:
        return 0
    date = date.split()[0]
    return int("".join(date.split("-")))


def parse_cancellation_policy(dataframe):
    vec_parse_cancellation = np.vectorize(parse_one_cancellation_policy)
    mat = vec_parse_cancellation(dataframe["cancellation_policy_code"])
    split_cancellation_policy = np.vectorize(cancellation_policy_index, excluded=[1])
    dataframe = dataframe.assign(cpc_d1=split_cancellation_policy(mat, 0))
    dataframe = dataframe.assign(cpc_p1=split_cancellation_policy(mat, 1))
    dataframe = dataframe.assign(cpc_n1=split_cancellation_policy(mat, 2))
    dataframe = dataframe.assign(cpc_d2=split_cancellation_policy(mat, 3))
    dataframe = dataframe.assign(cpc_p2=split_cancellation_policy(mat, 4))
    dataframe = dataframe.assign(cpc_n2=split_cancellation_policy(mat, 5))
    dataframe = dataframe.assign(cpc_no_show_p=split_cancellation_policy(mat, 6))
    dataframe = dataframe.assign(cpc_no_show_n=split_cancellation_policy(mat, 7))
    dataframe = dataframe.drop("cancellation_policy_code", axis=1)
    return dataframe


def cancellation_policy_index(cancellation_a, i):
    return cancellation_a[i]


def parse_one_cancellation_policy(cancellation):
    if cancellation == "UNKNOWN":
        return np.zeros(8)
    cancellation_split = cancellation.split("_")
    parsed = np.zeros(8).astype(pd.Series)
    for i, phase in enumerate(cancellation_split):
        if "D" in phase:
            parsed[0 + i] = int(phase.split("D")[0])
        else:
            if "P" in phase:
                parsed[6] = int(phase.split("P")[0])
            elif "N" in phase:
                parsed[7] = int(phase.split("N")[0])
            continue
        if "P" in phase:
            parsed[1 + i] = int(phase.split("D")[1].split("P")[0])
        elif "N" in phase:
            parsed[2 + i] = int(phase.split("D")[1].split("N")[0])
    return parsed.astype(pd.Series)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, pd.Series(cancellation_labels))

    # linear regression
    linear_regression = LinearRegression()
    linear_regression.fit(train_X, train_y)
    y_ = linear_regression.predict(test_X)
    threshold = 10000000
    y_[np.where(y_ <= threshold)[0]] = 0
    y_[np.where(y_ > threshold)[0]] = 1
    thresh_y = np.zeros(len(test_y))
    thresh_y[np.where(test_y.values != 0)[0]] = 1
    loss = np.abs(y_ - thresh_y).mean()
    print(f"linear regression's loss is {loss}")

    # knn
    # knn = KNeighborsClassifier(15)
    # knn.fit(train_X, train_y)
    # y1_ = knn.predict(test_X)
    # y1_[np.where(y1_ != 0)[0]] = 1
    # loss = np.abs(y1_ - thresh_y).mean()
    # print(f"knn's loss is {loss}")


    # soft svm
    # svm = LogisticRegression()
    # svm.fit(train_X, train_y)
    # y_ = svm.predict(test_X)
    # threshold = 10000000
    # y_[np.where(y_ <= threshold)[0]] = 0
    # y_[np.where(y_ > threshold)[0]] = 1
    # thresh_y = np.zeros(len(test_y))
    # thresh_y[np.where(test_y.values != 0)[0]] = 1
    # loss = np.abs(y_ - thresh_y).mean()
    # print(f"svm's loss is {loss}")


    # y = y_ + y1_
    # y[np.where(y != 2)[0]] = 0
    # y = y/2
    # loss = np.abs(y - thresh_y).mean()
    # print(f"the combine loss is {loss}")
    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
