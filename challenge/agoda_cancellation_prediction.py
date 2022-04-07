from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import json
import requests
from sklearn.preprocessing import OneHotEncoder

BASE_DATE = pd.to_datetime('1/1/2016')


def load_data(filename_train: str, filename_test):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename_train: str
        Path to house prices dataset
    filename_test: str
        Path to test set dataset

    Returns
    -------
    Tuple of dateFrame of the train data, ndarray of the labels and dateFrame of the test data
    """
    full_data_train = pd.read_csv(filename_train).drop_duplicates()
    full_data_train = full_data_train.assign(is_train=np.ones(full_data_train.values.shape[0]))
    full_data_test = pd.read_csv(filename_test).drop_duplicates()
    full_data_test = full_data_test.assign(is_train=np.zeros(full_data_test.values.shape[0]))
    full_data = pd.concat([full_data_train, full_data_test], ignore_index=True)
    for d in ["booking_datetime","checkin_date","checkout_date","hotel_live_date","cancellation_datetime"]:
        full_data[d] = pd.to_datetime(full_data[d])
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
    #                                        'original_payment_method', , 'request_airport' 'hotel_id'
    features = full_data[['booking_datetime', 'checkin_date', 'checkout_date',
                          'hotel_live_date', 'hotel_star_rating',
                          'guest_is_not_the_customer',
                          'no_of_adults', 'no_of_children',
                          'no_of_extra_bed', 'no_of_room',
                          'original_selling_amount', 'is_user_logged_in',
                          'cancellation_policy_code', 'is_first_booking', 'request_nonesmoke',
                          'request_latecheckin', 'request_highfloor', 'request_largebed',
                          'request_twinbeds', 'request_earlycheckin',"is_train"]]
    conversion_rates = requests.get('https://v6.exchangerate-api.com/v6/b7516dbaf2d4a78e08d4c8cf/latest/USD').json()[
        "conversion_rates"]
    to_usd = full_data["original_payment_currency"].apply(lambda x: conversion_rates[x])
    features["original_selling_amount"] = features["original_selling_amount"] * to_usd
    dummies = ['accommadation_type_name','original_payment_method','guest_nationality_country_name','charge_option','original_payment_type']
    ohe = OneHotEncoder(handle_unknown='ignore')
    features = pd.concat([features, pd.DataFrame(ohe.fit_transform(full_data[dummies]).toarray(),index=full_data.index,dtype=int)], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['accommadation_type_name']])], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['origin_country_code']])], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['original_payment_method']])], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['charge_option']])], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['original_payment_type']])], axis=1)
    # features = pd.concat([features, pd.get_dummies(full_data[['hotel_brand_code']])], axis=1)

    features = parse_cancellation_policy(features)
    features_train = features[features['is_train'] == 1]
    features_test = features[features['is_train'] == 0]
    features_test = features_test.drop("is_train", axis=1)
    features_train = features_train.drop("is_train", axis=1)
    labels = np.zeros(full_data_train.values.shape[0])
    labels[np.where(full_data['cancellation_datetime'] != 0)[0]] = 1
    return features_train, labels, features_test


def date_to_numeric(date):
    if date == 0:
        return 0
    return (date - BASE_DATE).days


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

    """
    Important Note: we pre-processed the train and the test data combine, so load data function must get two 
    filenames
    """
    # Load data
    df, cancellation_labels, test_data = load_data("../datasets/agoda_cancellation_train.csv","../datasets/test_set_week_2.csv")
    # train_X, train_y, test_X, test_y = split_train_test(df, pd.Series(cancellation_labels), 0.75)
    #
    # thresh_y = np.zeros(len(test_y))
    # thresh_y[np.where(test_y.values != 0)[0]] = 1
    # loss = np.abs(np.zeros(len(test_y)) - thresh_y).mean()
    # print(f"default loss is {loss}")

    # linear regression
    # linear_regression = LinearRegression()
    # linear_regression.fit(train_X, train_y)
    # y_ = linear_regression.predict(test_X)
    # threshold2 = 0.5
    # y_[np.where(y_ <= threshold2)[0]] = 0
    # y_[np.where(y_ > threshold2)[0]] = 1
    # loss = np.abs(y_ - thresh_y).mean()
    # print(f"linear regression's loss is {loss}")
    # recall = np.sum(test_y * y_)
    # print(f"linear regression's TP is {recall} out of {np.sum(test_y)}")

    # logistic regression
    # LR = LogisticRegression()
    # LR.fit(train_X, train_y)
    # y1_ = LR.predict_proba(test_X)[:, 1]
    # cutoff = 0.4
    # y1_[np.where(y1_ <= cutoff)] = 0
    # y1_[np.where(y1_ > cutoff)] = 1
    # loss = np.abs(y1_ - thresh_y).mean()
    # print(f"logistic regression's loss is {loss}")
    # recall = np.sum(test_y * y1_)
    # print(f"logistic regression's TP is {recall} out of {np.sum(test_y)}")

    #knn
    # knn = KNeighborsClassifier()
    # knn.fit(train_X, train_y)
    # y2_ = knn.predict_proba(test_X)[:, 1]
    # cutoff = 0.6
    # y2_[np.where(y2_ <= cutoff)] = 0
    # y2_[np.where(y2_ > cutoff)] = 1
    # loss = np.abs(y2_ - thresh_y).mean()
    # print(f"knn's loss is {loss}")
    # recall = np.sum(test_y * y2_)
    # print(f"knn's TP is {recall} out of {np.sum(test_y)}")

    # or_y = y1_ + y_ + y2_
    # or_y[or_y >= 2] = 1
    # loss = np.abs(or_y - thresh_y).mean()
    # print(f"combine or's loss is {loss}")
    # recall = np.sum(test_y * or_y)
    # print(f"combine or's TP is {recall} out of {np.sum(test_y)}")
    #
    # and_y = y1_ + y_ + y2_
    # and_y[and_y == 1] = 0
    # and_y[and_y == 2] = 0
    # and_y[and_y == 3] = 1
    # loss = np.abs(and_y - thresh_y).mean()
    # print(f"combine and's loss is {loss}")
    # recall = np.sum(test_y * and_y)
    # print(f"combine and's TP is {recall} out of {np.sum(test_y)}")
    #
    # most_y = y1_ + y_ + y2_
    # most_y[and_y == 1] = 0
    # most_y[and_y >= 2] = 1
    # loss = np.abs(most_y - thresh_y).mean()
    # print(f"combine most's loss is {loss}")
    # recall = np.sum(test_y * most_y)
    # print(f"combine most's TP is {recall} out of {np.sum(test_y)}")

    #Fit model over data
    estimator = LinearRegression().fit(df, cancellation_labels)

    # Store model predictions over test set
    y_ = estimator.predict(test_data)
    threshold2 = 0.5
    y_[np.where(y_ <= threshold2)[0]] = 0
    y_[np.where(y_ > threshold2)[0]] = 1
    print(len(np.where(y_==1)[0]))
    pd.DataFrame(y_, columns=["predicted_values"]).to_csv("313434235_311119895_315421768.csv", index=False)
