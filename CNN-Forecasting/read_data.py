#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:35:49 2023

@author: st_ko

Based on the script "utilities.py" from btrotta
Please check her repo : 
https://github.com/btrotta/m4/blob/master/utilities.py
"""


import numpy as np
import pandas as pd
import os 


# testing only with hourly_series


def read_data(series,series_length,horizon,cycle_length):
    
    # read series 
    # test only with hourly data 
    #series = pd.read_csv( os.path.join('data','Hourly_train_shuffle.csv'), header=0, index_col=0)
    
    # keep the series name correspondence with row index 
    # in a dictionary 
    #series_headers = dict( [ (i,series.iloc[i].values[0]) for i in range(len(series)) ] )
    
    # array to hold series + empty columns for predictions 
    all_data = np.zeros([len(series),series_length + horizon])
    
    #hourly_data = pd.read_csv("/home/st_ko/Desktop/DSIT_SPRING_SEMESTER/TIME_SERIES/M4-methods-master/211 - btrotta/MyVersion/New_Implementation/data/Hourly_train_shuffle.csv")
    for i in range(len(series)):
        #read row 
        # skip the first column -> which says each series_name
        # should keep the corresponding series original number though in a dictionary 
        
        
        # read row 
        row = series.iloc[i].values
        
        # keep only the columns from this row that have values (filter out empty columns)
        row = row[~pd.isna(row)]
        
        # total columns missing are 
        # predictions(horizon) + missing values ??
        # series_length is the original row length  ??
        # each time we read the chosen "series_length" number of columns from each series
        # to train the current model 
        # what values do we fill here ??
        if (len(row) < series_length + horizon):
            num_missing = series_length + horizon - len(row)
            # what values do we fill here ?
            all_data[i, -len(row):] = row[-(series_length + horizon):]
            
            # if more missing than period 
            # what values do we fill here ?? 
            #
            if num_missing > cycle_length:
                all_data[i, :num_missing] = np.mean(row[:-horizon])  # mean of non-test values
            else:
                all_data[i, :num_missing] = all_data[i, cycle_length:(cycle_length + num_missing)]  # copy from same period in cycle
        else:
            # ?? 
            all_data[i, :] = row[-(series_length + horizon):]   
    # maybe also return the dictionary 
    return all_data
    



 # should understand normalize and denormalize 
def normalise(arr):
    mean_arr = np.mean(arr, 1)
    series_length = arr.shape[1]
    std_arr = np.std(arr, axis=1)
    std_arr = np.where(std_arr == 0, mean_arr, std_arr)
    std_arr = np.where(std_arr == 0, np.ones(mean_arr.shape), std_arr)
    norm_arr = (arr - np.repeat(mean_arr[:, np.newaxis], series_length, 1)) \
               / np.repeat(std_arr[:, np.newaxis], series_length, 1)
    return norm_arr, mean_arr, std_arr


def denormalise(norm_arr, mean_arr, std_arr):
    series_length = norm_arr.shape[1]
    denorm_arr = norm_arr * np.repeat(std_arr[:, np.newaxis], repeats=series_length, axis=1) \
                 + np.repeat(mean_arr[:, np.newaxis], repeats=series_length, axis=1)
    return denorm_arr



def read_extended_data(series,series_length, min_length, horizon, cycle_length, augment=True, test_mode=True):
    """Get data from series dataframe, padding series that are too short."""
    ext_arr = []
    weights = []
    
    # test with HOURLY train for now 
    #series = pd.read_csv( os.path.join('data','Hourly_train_shuffle.csv'), header=0, index_col=0)
    
    ## ?? what do we do here ?? ##
    first_row = int(len(series) * 0.2) if test_mode else 0
    
    # read each series 
    # keep only values != nan 
    for i in range(first_row, len(series)):
        row = series.iloc[i].values
        row = row[np.logical_not(np.isnan(row))]
        
        #################################################################### 
        ######### in this case we add one row per row(time series) #########
        ####################################################################
        if len(row) <= series_length + horizon:
            if series_length == min_length:
                num_missing = series_length + horizon - len(row)
                row_to_add = np.zeros([series_length + horizon])
                
                # imputate values 
                # ?? how does this work ?
                # fill with the available values the row_to_add 
                row_to_add[-len(row):] = row[-(series_length + horizon):]
                
                
                # fill with values 
                if num_missing > cycle_length:
                    # ?? fill missing positions with mean of known (available) values 
                    row_to_add[:num_missing] = np.mean(row[:-horizon])  # mean of non-test values
                else:
                    # copy  number=num_missing values in this period to fill num_missing 
                    # ?? 
                    row_to_add[:num_missing] = row[cycle_length - num_missing:cycle_length]  # copy from same period in cycle
                
                # add this row 
                ext_arr.append(row_to_add)
                weights.append(1)
        else:
            ############ we append multiple rows per row #######
            ############ only for this case ####################
            
            
            # augment data for series with length < minimum length 
            # add one period of values 
            num_to_add = cycle_length if augment else 1
            
            
            # add the minimum between cycle_length , length - length - horizon 
            # add min(period,row-series_length-horizon)
            num_extra = min(num_to_add, len(row) - series_length - horizon)
            
            # 
            for j in range(num_extra):
                # append series_length + horizon values 
                # only for first j = 0 
                if j == 0:
                    ext_arr.append(row[-(series_length + horizon):])
                    
                # for j > 0 add 
                else:
                    # we add  series_length + horizon values going
                    # progressively backwards 
                    # start from length - (series_length + horizon) and go backwards
                    # getting the same number of values but one column back each time 
                    ext_arr.append(row[-(series_length + horizon + j):-j])
                    
                # ??? what weights are those 
                # if j==0 this produces division by zero error ?? (exception)
                # maybe add try catch on this 
                weights.append(1 / num_extra)
            
            # next line only for testing 
            #ext_arr.append(row[-(series_length + horizon ):])
            #pass
             
                
                
    # convert list of lists into array      
    # add the new rows to the ext_arr     
    train_ext = np.stack(ext_arr, 0)
    print(train_ext.shape)
    
    # x_values for train = all_available values without those trying to predict 
    train_x_ext = train_ext[:, :-horizon]
    
    
    ######### SOS : HOW DO WE FILL THOSE horizon values ? (they are ground truth values ?? )
    # y_values for train = all horizon predicted values we use for training 
    train_y_ext = train_ext[:, -horizon:]
    
    # create array of weights 
    # where are the weights used ?? 
    weights = np.array(weights)
    return train_x_ext, train_y_ext, weights


######## 
def blend_predictions(prediction_dict, num_values):
    """Given the original series and dictionary of predictions indexed by the training length,
    produce a blended prediction and store it in prediction[-1]."""
    
    # make predictions for multiple training lengths 
    # for each training_length (example : hourly : [848])
    # we extend the prediction dictionary by adding as last array 
    # the smallest training_length predictions array 
    training_lengths = list(prediction_dict.keys())
    
    # copy the first series_length prediction in the final position
    prediction_dict[-1] = np.copy(prediction_dict[min(training_lengths)])
    
    
    
    ###################### for all training lengths #######################
    for series_length in training_lengths:
        
        
        
        train_length_ok = np.logical_or(series_length == min(training_lengths), num_values >= series_length)
        
        # for all models
        # we get as aggregate the mean for each horizon(dt) for each time series
        # of the values of all models 
        # combined prediction of all training_lengths 
        comb_prediction = np.mean(np.stack([prediction_dict[s] for s in training_lengths if s <= series_length], 2),
                                  2)
        
        # store the combined prediction in final position 
        prediction_dict[-1] = np.where(np.repeat(train_length_ok[:, np.newaxis], prediction_dict[-1].shape[1], 1),
                                       comb_prediction, prediction_dict[-1])


# read extended data ! 
# main here is for testing only 
if __name__ == '__main__':
    series = pd.read_csv( os.path.join('data','Hourly_train_shuffle.csv'), header=0, index_col=0)
    all_data = read_data(series,840,48,7*24)
    extended_data = read_extended_data(series,840, 840,48, 7*24,True, False)
    

    
    
