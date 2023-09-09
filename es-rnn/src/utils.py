import numpy as np 
import pandas as pd
import random
from torch.utils.data import Dataset


class SequenceLabelingDataset(Dataset):
    
    def __init__(self, input_data, max_size=100, sequence_labeling=True, seasonality=12, out_preds=12):
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
