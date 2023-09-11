import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # Load the training and testing data
    train_data = pd.read_csv('Hourly-test.csv').drop(columns='V1').dropna(how='any')
    test_data = pd.read_csv('Hourly-train.csv').drop(columns='V1').dropna(how='any')

    # Rename the columns using the 'rename' method and a dictionary
    # new_column_names = [str(i) for i in range(1, 49)]
    # train_data = train_data.rename(columns=dict(zip(df.columns, new_column_names)))
    # test_data = test_data.rename(columns=dict(zip(test_data.columns, new_column_names)))

    # Split the train_data into training and validation sets
    train_ratio = 0.8
    validation_ratio = 0.2
    X_train, X_val = train_test_split(train_data, test_size=validation_ratio, random_state=42)

    # Extract the time column and the multi-dimensional data
    # X_train = train_data.iloc[:, 1:].values
    # X_test = test_data.iloc[:, 1:].values
    # X_train_t = np.arange(len(train_data.iloc[:, 0]))
    # X_test_t = np.arange(len(test_data.iloc[:, 0]))

    # Normalize the data
    scaler_tr = MinMaxScaler()
    scaler_ts = MinMaxScaler()
    X_train = scaler_tr.fit_transform(X_train)
    X_val = scaler_tr.transform(X_val)
    X_test = scaler_ts.fit_transform(test_data)

    # Convert the NumPy arrays back to Pandas DataFrames
    train_df = pd.DataFrame(X_train, columns=train_data.columns)
    val_df = pd.DataFrame(X_val, columns=train_data.columns)
    test_df = pd.DataFrame(X_test, columns=test_data.columns)

    return train_df, val_df, test_df
