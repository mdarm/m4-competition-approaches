## Overview
The notebook contains implementations of the following purely statistical methods:

- Naive 1
- Naive S
- Naive 2
- Arima

## Method description

Naive 1: simply picks the last value of the training set, replicates it, forming a horizontal line.
Naive 2: picks the last $S$ values of the training set and replicates it

- 3d Holt-Winters implementation (that is, multiple series processing in one step)
- multi-series-model with shared trend LSTM
- additive time-series-reconstruction
- autoregressive learner.
- blender module to merge predictions from multiple series
- quantile loss to get prediction intervals.
- compare performance on benchmark dataset


## Replicating results

To replicate the results of this project:

1. Clone this repository to your local machine.
   ```
   git clone git@github.com:mdarm/m4-competition-approaches.git
   ```
2. Navigate to the project directory.
   ```
   cd m4-competion-approaches/es-rnn/src 
   ```
3. Simply run the `main.py` script.
   ```
   python main.py
   ```

## Dependencies

- torch
- numpy
- pandas
- matplotlib
