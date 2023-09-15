import numpy as np 
import pandas as pd
import random
from torch.utils.data import Dataset


def process_and_split_data(train_file, test_file, label):
    """
    Process hourly data: Load datasets, remove trailing NaNs, interpolate remaining missing values, 
    and split into train and test lists.

    Parameters:
    - train_file (str): Path to the training dataset.
    - test_file (str): Path to the test dataset.
    - label (str): The label of the series to be processed.

    Returns:
    - tuple: Two lists representing the cleaned and interpolated training data and test data for the given label.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    series_train = train_df[train_df['V1'] == label].iloc[0, 1:].values
    
    # Remove trailing NaNs
    series_train_no_trailing_nans = series_train[~pd.isnull(series_train)]
    
    # Convert the series to a DataFrame for interpolation
    series_train_df = pd.DataFrame(series_train_no_trailing_nans).astype(float)
    
    # Interpolate remaining NaNs in training series
    train_series_interpolated = series_train_df.interpolate(method='linear', axis=0).values.flatten().tolist()

    test_series = test_df[test_df['V1'] == label].iloc[0, 1:].values.tolist()

    return train_series_interpolated, test_series


class SequenceLabelingDataset(Dataset):
    
    def __init__(self, input_data, max_size=100, sequence_labeling=True, seasonality=24, out_preds=48):
        self.data = input_data
        self.max_size = max_size
        self.sequence_labeling = sequence_labeling
        self.seasonality = seasonality
        self.out_preds = out_preds
        
    def __len__(self):
        return int(10000)
    
    def __getitem__(self, index):
        data_i = self.data
        
        # Randomly shift the inputs to create more data
        if len(data_i) > self.max_size:
            max_rand_int = len(data_i) - self.max_size
            # Take a random start integer
            start_int = random.randint(0, max_rand_int)
            data_i = data_i[start_int:(start_int + self.max_size)]
        else:
            start_int = 0

        inp = np.array(data_i[:-self.out_preds])
        
        if self.sequence_labeling:
            # In case of sequence labeling, shift the input by the range to output
            out = np.array(data_i[self.out_preds:])
        else:
            # In case of sequence classification, return only the last n elements needed in the forecast
            out = np.array(data_i[-self.out_preds:])
        
        # Calculate how much to shift the season
        shift_steps = start_int % self.seasonality
        
        return inp, out, shift_steps
