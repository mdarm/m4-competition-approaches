
# Convolutional Models for M4 time series forecasting

## About the Project
This project is based on the M4 Time Series Forecasting Competition 'btrotta' entry. In particular I trained and run her convolutional neural networks using her instructions in her repo : (https://github.com/btrotta/m4).


## Built With

The code and models were built using :
* python
* Keras
* Tensorflow


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites
The models were tested with :
```
python==3.8.10
tensorboard==2.13.0
tensorboard-data-server==0.7.1
tensorflow==2.13.0
tensorflow-estimator==2.13.0
tensorflow-io-gcs-filesystem==0.32.0
scikit-learn==1.2.2
scipy==1.10.1
sklearn==0.0.post1
pandas==2.0.3
numpy==1.24.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
```
In a clean python virtual environment you can just run `pip install <The above list in a txt file>` to install them altogether.

Regarding Tensorflow in particular the models were implemented using V1 compatibility mode.

## Usage
To download the data you can visit (https://github.com/Mcompetitions/M4-methods/tree/master/Dataset/Test) and (https://github.com/Mcompetitions/M4-methods/tree/master/Dataset/Train) to download the train and test data. Next Move them into a directory called `data`.
1. Run `shuffle.py` to shuffle the series in each frequency and get the resulting shuffled `.csv`  to use for training.
2. To build, train the models and get the metric results run `main.py`. This wil generate a log file with the metric results, as well as train the models.
3. To make predictions in the test dataset and get the `.csv` with the predictions for each frequency run `predict.py`.The predictions will be saved in the `predictions` directory with a different directory for each frequency.
4. If you want to make the visualizations you must download the directory tree `original_trained_models/results` in your current directory since the scripts read the original author's reproduced results from this path. Alternatively create the same directory tree and copy the content inside. Then you can make visualizations of the original models and the modified models by running `compare_predictions.py` which will read the first series from each frequency from the `*fc*.csv` results and visualize the predictions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Experimentation/Improving the Models
I experimented with the architecture of the convolutional models created and their parameters and i tried to improve the models for each frequency . Some functions used are taken from the original (https://github.com/btrotta/m4) and others were added. In particular besided the new models, functions to observe the training loss and visualize the final predictions are implemented, whereas the final predictions are located in the 'predictions' directory and are split according to frequency.

## Models
If you want to take a look at the models or make modifications and update them see `build_models.py` where each model for each frequency is created. Additionally you can change model parameters like series_length, yearly_count, filter_count, units_count, epochs, bs from the `train_models.py` in the `utils` module, in the line where each model is constructed.
