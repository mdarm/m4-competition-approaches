#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 20:49:22 2023

@author: st_ko
"""
import glob
import keras as ks
import pandas as pd
import numpy as np
import metrics
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
from scipy import stats
#from utilities import *
#from model_definitions import *
from build_models import *
from read_data import *
import matplotlib.pyplot as plt 


# true values have the "V" column index but the predicted have the "F" so we rename the true values 
# to be able to plot corresponding values more easily  
def mapName(df,freq_name):
    if(freq_name == 'Y'):   
        fat = 'Y'
    elif(freq_name == 'Q'):
        fat = 'Q'
    elif(freq_name == 'M'):
        fat = 'M'
    elif(freq_name == 'W'):
        fat = 'W'
    elif(freq_name == 'D'):
        fat = 'D'
    else:
        fat = 'H'
        
    column_mapping = {f'V{i}': f'V{i-1}' for i in range(1, len(df.columns) + 2)}
    # Rename the DataFrame columns using the mappings
    df.rename(columns=column_mapping, inplace=True)
    column_mapping2 = {f'V{i}': f'F{i}' for i in range(1, len(df.columns) + 1)}
    df.rename(columns=column_mapping2, inplace=True)
    # replace row index to be the same as with our model's predictions 
    custom_index =  [ f'{fat}{i}' for i in range(1,len(df)+1 )]
    df.set_index(pd.Index(custom_index), inplace=True)
    
    return df



def plotCompare(R):
    # read the original model's predictions for series 1 
    # TODO: customize with glob to just read the csv file without the name (only fc pattern)
    y_orig = pd.read_csv(os.path.join('original_trained_models/results', 'Submission_fc_230825_1339.csv'), header=0, index_col=0)
    
    if(R == 'D'):
        label = "Daily Predictions" 
        horizon = 14
        y_orig_freq = y_orig.loc[R+'1':R+'4227',:].loc[R+'1',:'F'+str(horizon)]
        df = pd.read_csv(os.path.join('data', 'Daily-train.csv'), header=0, index_col=0)
        
        # y_path 
        y_path = glob.glob(os.path.join('predictions','Daily','Submission_fc_*Daily.csv'))
        y_path = "".join(y_path)
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Daily-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
    elif(R=='H'):
        label = "Hourly Predictions" 
        horizon = 48 
        y_orig_freq = y_orig.loc[R+'1':R+'414',:].loc[R+'1',:'F'+str(horizon)]
        df = pd.read_csv(os.path.join('data', 'Hourly-train.csv'), header=0, index_col=0)
        
        # y_path 
        y_path = glob.glob(os.path.join('predictions','Hourly','Submission_fc_*Hourly.csv'))
        y_path = "".join(y_path)
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Hourly-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
    elif(R=='W'):
        label = "Weekly Predictions" 
        horizon = 13
        df = pd.read_csv(os.path.join('data', 'Weekly-train.csv'), header=0, index_col=0)
        y_orig_freq = y_orig.loc[R+'1':R+'359',:].loc[R+'1',:'F'+str(horizon)]
        y_path = glob.glob(os.path.join('predictions','Weekly','Submission_fc_*Weekly.csv'))
        y_path = "".join(y_path)
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Weekly-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
        
    elif(R=='M'):
        label = "Monthly Predictions" 
        horizon = 18
        y_orig_freq = y_orig.loc[R+'1':R+'48000',:].loc[R+'1',:'F'+str(horizon)]
        df = pd.read_csv(os.path.join('data', 'Monthly-train.csv'), header=0, index_col=0)
        
        y_path = glob.glob(os.path.join('predictions','Monthly','Submission_fc_*Monthly.csv'))
        y_path = "".join(y_path)        
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Monthly-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
        
    elif(R=='Q'):
        label = "Quarterly Predictions" 
        horizon = 8
        df = pd.read_csv(os.path.join('data', 'Quarterly-train.csv'), header=0, index_col=0)
        y_orig_freq = y_orig.loc[R+'1':R+'24000',:].loc[R+'1',:'F'+str(horizon)]
        
        y_path = glob.glob(os.path.join('predictions','Quarterly','Submission_fc_*Quarterly.csv'))
        y_path = "".join(y_path)        
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Quarterly-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
        
    else :
        label = "Yearly Predictions" 
        horizon = 6
        df = pd.read_csv(os.path.join('data', 'Yearly-train.csv'), header=0, index_col=0)
        y_orig_freq = y_orig.loc[R+'1':R+'23000',:].loc[R+'1',:'F'+str(horizon)]
        
        y_path = glob.glob(os.path.join('predictions','Yearly','Submission_fc_*Yearly.csv'))
        y_path = "".join(y_path)
        
        y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
        y_freq_true = pd.read_csv(os.path.join('data', 'Yearly-test.csv'), header=0).iloc[:,1:]
        y_freq_true= mapName(y_freq_true,R)
        y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
    
    # READ ONLY THE LAST 100 VALUES FROM HISTORY 
    x_history = df.loc[R +'1',:].dropna()
    x_history = x_history.iloc[-100:]
    
    
    # create new figure 
    plt.figure()
    for i in range(len(x_history)):
        plt.annotate(x_history.index[i], (i, x_history.values[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 45)
    
    
    # the final data for plotting 
    model_final_1 = pd.concat([x_history,y_model_1])
    model_final_2 = pd.concat([x_history,y_orig_freq])
    model_final_3 = pd.concat([x_history,y_freq_true])
    
    
    
    
    # PLOT PARAMETERS 
    plt.title(label)
    plt.xticks(rotation = 90)
    plt.xlabel("Time Steps : History + Horizon",weight = 'bold',size = 12)
    plt.ylabel("Time Series Observations",weight= 'bold',size = 12)
    
    # history values before horizon 
    plt.plot(x_history, 'o-',color='blue',label = 'History ')
    # prediction of our model 
    plt.plot(model_final_1.iloc[-horizon:], color='green', label='Our Model Predictions')
    # prediction of the author's models 
    plt.plot(model_final_2.iloc[-horizon:], color='red', label="Author's Model Predictions")
    #plot the true values 
    plt.plot(model_final_3[-horizon:],color='skyblue', label='True Horizon Values')
    
    # add legend
    plt.legend()
    
    
    

# plot every frequency for visualization #     
plotCompare('D')
plotCompare('H')
plotCompare('W')
plotCompare('M')    
plotCompare('Q')
plotCompare('Y')