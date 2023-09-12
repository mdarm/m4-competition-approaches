from baseline import Baseline
from window_generator import WindowGenerator
from data_reader import get_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from itertools import cycle


# Suppress the specific TensorFlow warning
warnings.filterwarnings("ignore", message=".*AVX_VNNI FMA.*")


def main():

    """
    print(tf.config.list_physical_devices('GPU'))
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu_device, True)

    with tf.device('/GPU:0'):
        # Create your TensorFlow model here
        model = tf.keras.Sequential([...])
    """

    # fetch datasets (trend and seasonality are filtered)
    dataset = 'Daily'
    train_data, test_data = get_data(dataset)

    # Normalize data
    scaler_tr = MinMaxScaler()
    scaler_ts = MinMaxScaler()
    X_train = scaler_tr.fit_transform(train_data)
    X_test = scaler_ts.fit_transform(test_data)
    # Convert the NumPy arrays back to Pandas DataFrames
    X_train = pd.DataFrame(X_train, columns=train_data.columns)
    X_test = pd.DataFrame(X_test, columns=test_data.columns)

    # Add a column with one-hot encodings of days,
    # assuming that the first record starts from Monday.
    encodings = [
        [1, 0, 0, 0, 0, 0 ,0], # monday
        [0, 1, 0, 0, 0, 0, 0], # tuesday
        [0, 0, 1 ,0, 0, 0, 0], # wednesday
        [0, 0 ,0, 1 ,0 ,0, 0], # thursday
        [0, 0, 0, 0 ,1, 0, 0], # friday
        [0, 0, 0, 0, 0, 1, 0], # saturday
        [0, 0 ,0, 0, 0, 0, 1]  # sunday
    ]
    encodings_cycle = cycle(encodings)
    new_columns = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']
    # training set:
    # add new columns
    for column_name in reversed(new_columns):
        X_train.insert(0, column_name, 0)
    # replace the new columns values with the encodings
    for index, row in X_train.iterrows():
        encoding = next(encodings_cycle)
        X_train.loc[index, X_train.columns[:7]] = encoding
    # test set:
    # add new columns
    encodings_cycle = cycle(encodings)
    for column_name in reversed(new_columns):
        X_test.insert(0, column_name, 0)
    # replace the new columns values with the encodings
    for index, row in X_test.iterrows():
        encoding = next(encodings_cycle)
        X_test.loc[index, X_test.columns[:7]] = encoding

    # Split the train_data into training and validation sets
    train_ratio = 0.8
    validation_ratio = 0.2
    X_train, X_val = train_test_split(X_train,
                                      test_size=validation_ratio,
                                      random_state=42,
                                      shuffle=False)

    # Convert the NumPy arrays back to Pandas DataFrames
    train_df = pd.DataFrame(X_train, columns=X_train.columns)
    val_df = pd.DataFrame(X_val, columns=X_train.columns)
    test_df = pd.DataFrame(X_test, columns=X_test.columns)

    # etc.
    MAX_EPOCHS = 20
    target_labels = ['V450']
    val_performance = {}
    performance = {}

    # Multi-step window
    print("Multi-step model")
    CONV_WIDTH = 30
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=target_labels)
    # Multi-layer model
    #with tf.device('/GPU:0'):
    multi_step_dense = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1),
        tf.keras.layers.Reshape([1, -1]),
    ])
    for example_inputs, example_labels in conv_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
    history = compile_and_fit(multi_step_dense, conv_window, MAX_EPOCHS)
    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
    conv_window.plot(model=multi_step_dense, plot_col=target_labels[0])

    """
    The main down-side of this approach is that the resulting model can only be executed
    on input windows of exactly this shape.
    """


def compile_and_fit(model, window, max_epochs, patience=2, ):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


if __name__ == '__main__':
    main()

