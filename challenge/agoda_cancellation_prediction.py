from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
import pandas as pd
import sklearn


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
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    print(full_data.columns.values)
    # change dates to numbers
    index_date = np.flatnonzero(np.core.defchararray.find(full_data.columns.values.astype(str), "date") != -1)
    vec_date_to_numeric = np.vectorize(date_to_numeric)
    full_data.values[:, index_date.astype(int)] = vec_date_to_numeric(full_data.values[:, index_date.astype(int)])
    # choose the relevant feature
    features = full_data[['h_booking_id', 'booking_datetime', 'checkin_date', 'checkout_date',
                          'hotel_id', 'hotel_country_code', 'hotel_live_date', 'hotel_star_rating',
                          'accommadation_type_name', 'charge_option', 'h_customer_id',
                          'customer_nationality', 'guest_is_not_the_customer',
                          'guest_nationality_country_name', 'no_of_adults', 'no_of_children',
                          'no_of_extra_bed', 'no_of_room', 'origin_country_code', 'language',
                          'original_selling_amount', 'original_payment_method',
                          'original_payment_type', 'original_payment_currency', 'is_user_logged_in',
                          'cancellation_policy_code', 'is_first_booking', 'request_nonesmoke',
                          'request_latecheckin', 'request_highfloor', 'request_largebed',
                          'request_twinbeds', 'request_airport', 'request_earlycheckin',
                          'hotel_area_code', 'hotel_brand_code', 'hotel_chain_code', 'hotel_city_code']]
    labels = full_data["cancellation_datetime"]

    return features, labels


def date_to_numeric(date):
    date = date.split()[0]
    return int("".join(date.split("-")))


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
    # train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # logistic_regression = sklearn.linear_model.LogisticRegression()

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
