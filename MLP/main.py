from baseline import Baseline
from window_generator import WindowGenerator
from data_reader import get_data

from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings

# Suppress the specific TensorFlow warning
warnings.filterwarnings("ignore", message=".*AVX_VNNI FMA.*")

def main():

    # fetch datasets
    train_df, val_df, test_df = get_data()

    # etc.
    target_label = 'V49'
    val_performance = {}
    performance = {}

    # Single-step window
    print("Single-step model")
    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=[target_label])
    print(single_step_window)
    # Single layer model
    MAX_EPOCHS = 20
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(dense, single_step_window, MAX_EPOCHS)
    val_performance['Dense'] = dense.evaluate(single_step_window.val)
    single_step_window.plot(dense)

    print("\n--------------------\n")

    # Multi-step window
    print("Multi-step model")
    CONV_WIDTH = 3
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['V49'])
    print(conv_window)
    # Multi-layer model
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    print('Input shape:', conv_window.example[0].shape)
    print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
    history = compile_and_fit(multi_step_dense, conv_window, MAX_EPOCHS)
    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
    conv_window.plot(multi_step_dense)


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

