import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS

def get_data(dataset):
    # Load the training and testing data
    train_data = pd.read_csv(f'{dataset}-train.csv').drop(columns='V1')
    test_data = pd.read_csv(f'{dataset}-test.csv').drop(columns='V1')

    # Fill missing values with the mean of the it's column.
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    # remove trend and seasonality
    train_data = preprocess_time_series(train_data)
    test_data = preprocess_time_series(test_data)

    # Drop rows with NaN values (they might appear after detrending)
    train_data = train_data.dropna(how='all')
    test_data = test_data.dropna(how='all')

    return train_data, test_data

def preprocess_time_series(data):
    # Let's remove trend and Seasonality.
    feature_names = data.columns.tolist()
    for feature in feature_names:
        # Removing Trend with OLS
        least_squares = OLS(data[feature].values, list(range(data.shape[0])))
        result = least_squares.fit()
        fit = pd.Series(result.predict(list(range(data.shape[0]))), index=data.index)
        data_ols_detrended = data[feature] - fit
        # Removing Seasonality by Differencing Over Linear Regression Transformed Time-Series
        data_detrended_diff = data_ols_detrended - data_ols_detrended.shift()
        # Saving
        data[feature] = data_detrended_diff
    # Drop rows with NaN values
    data = data.dropna(how='all')

    return data
