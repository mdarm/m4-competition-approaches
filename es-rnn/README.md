# ES-RNN

# Overview
This is a work-in-progress of implementing ES-RNN by Slawek Smyl, winner of the M4 competition.

## REFERENCES
1. [A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207019301153)
2. [The M4 Competition: Results, findings, conclusion and way forward](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward)
3. [M4 Competition Data](https://github.com/M4Competition/M4-methods/tree/master/Dataset)
4. [Dilated Recurrent Neural Networks](https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf)
5. [Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition](https://arxiv.org/abs/1701.03360)
6. [A Dual-Stage Attention-Based recurrent neural network for time series prediction](https://arxiv.org/abs/1704.02971)

## Progress: 

- [x] Implement single-series-model based on multiplicative seasonality (Holt-Winters)
- [x] Implement toy dataset and data functions
- [x] Test and debug ssm on toy dataset
- [ ] Implement multiplicative season and trend
- [ ] Implement multi-series-model with shared trend LSTM
- [ ] Implement msm output model 
- [ ] Compare performance on benchmark dataset
