# M4 Competition Approaches

In this repository, we present a detailed project that delves into the implementation and comparison of various methodologies from the [M4 Forecasting Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128). Additionally, we have authored an [essay](time2vec-essay/time2vec.pdf) evaluating the publication [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321)

The project was carried out as a key part of the curriculum for the 'DI503 - Time Series Analysis and Applications' course, taught at the National and Kapodistrian University of Athens (NKUA) during the Spring term of 2023. 

## Objective

The aim of this project was to develop a deep understanding of different forecasting models and strategies, as well as their practical applications, by drawing on results from the M4 Competition. Our comparison framework aims to provide clarity on the strengths and weaknesses of each approach, ultimately serving as a useful resource for future reference.


## Project Structure

```bash
$PROJECT_ROOT
¦
+-- CNN-Forecasting 
¦   # Forecasting using Convolutional Neural Networks
¦
+-- MLP
¦   # Forecasting using Multi-Layer Perceptrons
¦
+-- es-rnn 
¦   # Forecasting using Exponential Smoothing and Recurrent Neural Networks
¦
+-- report
¦   # Comprehensive report detailing implementaions, results, and conclusions
¦
+-- statistical-methods 
¦   # Forecasting using traditional statistical  methods
¦   # (Naïve 1,2 & S and ARIMA)
¦
+-- time2vec-essay
    # Detailed essay on "Time2Vec: Learning a Vector Representation of Time" concept
```

## Getting Started

### Algorithms

Before diving into the individual methodologies, ensure you have the necessary dependencies installed. Each implementation directory ([CNN-Forecasting](CNN-Forecasting), [MLP](MLP), [es-rnn](es-rnn), [statistical-methods](statistical-methods)) contains its own `README.md` file; detailing specific requirements and dependencies. It is recommended to check each directory's README before running the respective models.

### Dataset

A high-level overview of the dataset is shown in the following table; while the dataset itself can be downloaded directly from [here](https://github.com/M4Competition/M4-methods/tree/master/Dataset). 

| Time interval between successive observations | Micro | Industry | Macro | Finance | Demographic | Other | Total |
|------------------------------------------------|-------|----------|-------|---------|-------------|-------|-------|
| Yearly                                         | 6,538 | 3,716    | 3,903 | 6,519   | 1,088       | 1,236 | 23,000|
| Quarterly                                      | 6,020 | 4,637    | 5,315 | 5,305   | 1,858       | 865   | 24,000|
| Monthly                                        | 10,975| 10,017   | 10,016| 10,987  | 5,728       | 277   | 48,000|
| Weekly                                         | 112   | 6        | 41    | 164     | 24          | 12    | 359   |
| Daily                                          | 1,476 | 422      | 127   | 1,559   | 10          | 633   | 4,227 |
| Hourly                                         | 0     | 0        | 0     | 0       | 0           | 414   | 414   |
| Total                                          | 25,121| 18,798   | 19,402| 24,534  | 8,708       | 3,437 | 100,000|


### Assessment
For a detailed assessment of methodology and criteria, you can refer to our [report](report/report.pdf).


## Built With

* [Python](https://www.python.org) - The *data science* language
 * [Jupyter](https://jupyter.org/) - An interactive development environment for notebooks
* [PyTorch](https://www.pytorch.org/) - A dynamic framework for computation
* [Keras](https://keras.io/) - Not PyTorch ;)


## Authors

* **Christou Nektarios** - [nekxt](https://github.com/nekxt)
* **Darmanis Michael** - [mdarm](https://github.com/mdarm)
* **Efstathios Kotsis** - [staks1](https://github.com/staks1)
* **Vasilis Venieris** - [vasilisvenieris](https://github.com/vasilisvenieris)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
