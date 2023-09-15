## Overview
The notebook contains implementations of the following purely statistical methods:

- Naive 1
- Naive S
- Naive 2
- Arima

## Method description

- Naive 1: simply picks the last value of the training set, replicates it, forming a horizontal line.
- Naive 2: picks the last $S$ values of the training set and replicates them in the same order.
- Naive 2: similar approach to Naive 1; however, data are first checked for any seasonality, then adjusted based on their multiplicative seasonal component.
- Arima(p, d, q): finds best fit for (p, d, q) by exploring successive differences (through ADF tests) and autocorrelation coefficients.

## Error metrics

- Mean absolute percentage error (MAPE)
- Mean absolute error (MAE)
- Mean percentage error (MPE)
- Root means square error (RMSE)

## Dependencies

- numpy
- pandas
- matplotlib
- statsmodels
