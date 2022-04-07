import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=True).dropna().drop_duplicates()
    # feature i dropped: "lat","long", "id", "date",waterfront
    full_data = full_data.drop(np.where(full_data["Temp"].values < -10)[0])
    features = full_data[["Country", "City", "Date", "Year", "Month", "Day", "Temp"]]
    vec_convert_date = np.vectorize(convert_string_date_to_num)
    features = features.assign(day_of_year=vec_convert_date(features["Date"]))
    return features


def convert_string_date_to_num(string_date):
    dt = datetime.strptime(string_date, '%Y-%m-%d')
    return dt.timetuple().tm_yday


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    sub_df = df.loc[df['Country'] == "Israel"]
    graph = plt.figure()
    plt.title('Temp as function of day of year')
    plt.ylabel('Temp')
    plt.xlabel("day of year")
    for year in np.unique(sub_df["Year"]):
        ind = np.where(sub_df["Year"] == year)[0].astype(int)
        temp = sub_df["Temp"].values[ind]
        days = sub_df["day_of_year"].values[ind]
        plt.scatter(days, temp, label=str(year),s=2)
    plt.legend(loc='right')
    plt.show()

    by_month = sub_df.groupby('Month').agg('std')
    graph = plt.figure()
    plt.title('std of temp as function of month')
    plt.ylabel('Temp')
    plt.xlabel("month")
    plt.bar(np.arange(12)+1,by_month["Temp"])
    plt.show()
    # Question 3 - Exploring differences between countries
    graph = plt.figure()
    plt.title('std of temp as function of month')
    plt.ylabel('Temp')
    plt.xlabel("month")
    for county in np.unique(df["Country"]):
        sub_df = df.loc[df['Country'] == county]
        by_country = sub_df.groupby("Month").Temp.agg(['mean', 'std'])
        plt.plot(np.arange(12)+1,by_country.values[:,0], label=county)
        plt.fill_between(np.arange(12)+1, by_country.values[:,0] - by_country.values[:,1], by_country.values[:,0] + by_country.values[:,1], alpha=0.2)
    plt.legend(loc='upper left')
    plt.show()
    # Question 4 - Fitting model for different values of `k`
    sub_df = df.loc[df['Country'] == "Israel"]
    day_of_year = sub_df["day_of_year"]
    train_x, train_y, test_x, test_y = split_train_test(day_of_year, sub_df["Temp"])
    graph = plt.figure()
    plt.title("loss as function of degree")
    plt.ylabel('loss')
    plt.xlabel("degree")
    for k in range(1,11):
        pf = PolynomialFitting(k)
        pf.fit(np.array(train_x), np.array(train_y))
        loss = pf.loss(np.array(test_x),np.array(test_y))
        print(loss)
        if k > 2:
            plt.scatter(k, loss, color='blue')
    plt.show()
    # Question 5 - Evaluating fitted model on different countries
    plt.title("loss as function of country")
    plt.ylabel('loss')
    plt.xlabel("Country")
    pf.fit(np.array(day_of_year),np.array(sub_df["Temp"]))
    for county in np.unique(df["Country"]):
        if county != "Israel":
            sub_df = df.loc[df['Country'] == county]
            loss = pf.loss(np.array(sub_df["day_of_year"]), np.array(sub_df["Temp"]))
            plt.bar(county, loss)
    plt.show()