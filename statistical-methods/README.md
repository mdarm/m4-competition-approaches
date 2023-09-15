## Overview
The notebook contains implementations of the following purely statistical methods:

- Naive 1
- Naive S
- Naive 2
- Arima

## Differences between Slawek's version and this one

The original intention was to reproduce the complete ES-RNN algorithm, and its prediction results, on the entirety of the M4-competition dataset. I still have a way to go. What I have not yet implemented are the following

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
