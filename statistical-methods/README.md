## Overview
The notebook contains implementations of the following purely statistical methods:

- Naive 1
- Naive S
- Naive 2
- Arima

## Dataset

Dataset of choice was the yearly time-series and therefore it has been automated as such. For other datasets, eg hourly, some minor fiddling might be required.

## Method description

- Naive 1: simply picks the last value of the training set, replicates it, forming a horizontal line.
- Naive S: picks the last $S$ values of the training set and replicates them in the same order.
- Naive 2: similar approach to Naive 1; however, data are first checked for any seasonality through autocorrelation, then adjusted given their multiplicative seasonal component.
- Arima(p, d, q): finds best fit for (p, d, q):
  - p: AR order is picked based on the strongest partial autocorrelation components
  - d: Differences order is picked based on successive ADF tests; experimentation stops when p-value drops below 5%
  - q: MA order is picked based on the strongest autocorrelation components
  - the model assumes no seasonality

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
