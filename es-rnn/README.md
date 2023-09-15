## Overview
This is a work-in-progress of implementing ES-RNN by Slawek Smyl, winner of the M4 competition.

This repo contains:
- [x] Holt-Winters implementation.
- [x] GRU residuals learner.
- [x] Multiplicative time-series-reconstruction.


## Differences between Slawek's version and this one

The original intention was to reproduce the complete ES-RNN algorithm, and its prediction results, on the entirety of the M4-competition dataset. I still have a long way to go. What I have not yet implemented, and I would certainly like to, are the following

- 3d Holt-Winters implementation (that is, multiple series processing in one step)
- multi-series-model with shared trend LSTM
- additive time-series-reconstruction
- autoregressive learner
- blender module to merge predictions from multiple series
- quantile loss to get prediction intervals
- Slawek's loss function that optimises two losses: quantile loss + regularization
$$
L_q(y, \hat{y}) = q(y - \hat{y} )_{+} + (1- q) ( \hat{y} - y)_{+}
$$
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


## References

1. [A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207019301153)
2. [The M4 Competition: Results, findings, conclusion and way forward](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward)
3. [M4 Competition Data](https://github.com/M4Competition/M4-methods/tree/master/Dataset)
4. [Dilated Recurrent Neural Networks](https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf)
5. [Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition](https://arxiv.org/abs/1701.03360)
6. [A Dual-Stage Attention-Based recurrent neural network for time series prediction](https://arxiv.org/abs/1704.02971)
