#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Evaluation metrics, taken from the competition Github repo
https://github.com/M4Competition/M4-methods
The factor of 2 in SMAPE has been changed to 200 to be consistent with the R code.
"""


import numpy as np

# TAKEN FROM M4 COMPETITION DEFINITION 
# define smape metric 
def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(200.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()   # BT changed 2 to 200


# define mase metric 
def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    # naive forecasts (each previous value becomes the prediction for the next )
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    # 1/n * Sum (y_true - y_hat)
    # ???  why insample[freq : ]
    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    # final formula of mase
    # mean / masep
    return np.mean(abs(y_test - y_hat_test)) / masep
