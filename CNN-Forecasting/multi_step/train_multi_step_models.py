#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:51:51 2023

@author: st_ko
"""
import datetime as dt
import pandas as pd
import os 
from read_data import *
import tensorflow as tf
import keras as ks
import numpy as np
from metrics import *
import matplotlib.pyplot as plt
from multi_step_model import *
from read_data import *






def train_multi_step_models(models):
    
    # create log file to write results 
    log_file = 'log_{}.txt'.format(dt.datetime.now().strftime('%y%m%d_%H%M'))
    
    for test_mode in [True, False]:
      for model in models : 
          
            # read each corresponding series 
            series = pd.read_csv(os.path.join('data', '{}_train_shuffle.csv'.format(model.freq_name)), header=0, index_col=0)
        
    
            # creat prediction dictionary 
            # which will hold the arrays of predictions for each series_length
            prediction = dict()
            
            
            
            # count of available_values 
            # not all series have the same number of available values 
            num_values = series.notnull().sum(axis=1)
            
            
            
            # get training_lengths 
            training_lengths = model.training_lengths
            print(training_lengths)
            
            
            # train model for different training_lengths
            # for now test only with period = 840 series values 
            for series_length in training_lengths:
                
                
                # read data 
                all_data = read_data(series,series_length, model.horizon, model.cycle_length)
                
                # read extended data 
                # train_x , train_y , weights 
                train_x_ext, train_y_ext, weights = read_extended_data(series,series_length, min(training_lengths),
                                                                      model.horizon, model.cycle_length, model.augment, test_mode)
                
                # normalize
                train_x_ext, train_mean_ext, train_std_ext = normalise(train_x_ext)
        
                
                train_length_ok = np.logical_or(series_length == min(training_lengths),
                                                num_values.values - model.horizon >= series_length)
                
                
    
                if test_mode:
                    
                    
                    # initialise prediction array # 
                    # prediction[series_length ] ?? why 
                    # array holding all predictions for all series 
                    prediction[series_length] = np.zeros([len(all_data), model.horizon])
                    
                    # 
                    curr_prediction = prediction[series_length]
                    # ?? 
                    curr_prediction[:] = np.nan
                    
                    
         
                ############## train for the whole horizon ################
                
                # for all the horizon and not only one value # sos 
                train_y = train_y_ext[:, :]
                
                ######## I NEED TO FIX THIS #########################
                # MSE IS HUGE 
                train_y = (train_y - np.reshape(train_mean_ext,(-1,1)) )/ np.reshape(train_std_ext,(-1,1))
                max_y = 5 * np.std(train_y)
                train_y[train_y > max_y] = max_y
                train_y[train_y < -max_y] = -max_y
                
               
                ks.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                np.random.seed(0)
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                tf.compat.v1.set_random_seed(0)
                tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                
                ############################################################
                ############ CREATE MODEL FOR TRAINING/PREDICTION ##########
                ############################################################
                
                # est is constructor taken from the function build_m_Model
                model_est = model.model_constructor
                
                # build different model based on the frequency 
                if(model.freq_name=='Daily'):
                    hmodel,epochs,batch_size = model_est(series_length,250,1000,model.horizon)
                elif(model.freq_name=='Hourly'):
                    hmodel,epochs,batch_size = model_est(series_length,168,1000,model.horizon)
                elif(model.freq_name == 'Weekly'):
                    hmodel,epochs,batch_size = model_est(series_length,50,1000,model.horizon)
                elif(model.freq_name == 'Monthly'):
                    hmodel,epochs,batch_size = model_est(series_length,250,1000,model.horizon)
                elif(model.freq_name == 'Quarterly'):
                    hmodel,epochs,batch_size = model_est(series_length,250,1000,model.horizon)
                else :
                    hmodel,epochs,batch_size = model_est(series_length,250,1000,model.horizon)
                    
                # training 
                
                history = hmodel.fit(train_x_ext, train_y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                        callbacks=[ks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)]
                                   )
                
                
                if test_mode :
                    train_x = all_data[:,:series_length]
                    train_x, train_mean, train_std = normalise(train_x)
                    #curr_prediction[:, :] =  hmodel.predict(train_x)
                    
                    
                    curr_prediction[:, :] = np.where(np.reshape(train_length_ok,(-1,1)), hmodel.predict(train_x),
                                                                curr_prediction[:, :])
                                                        
                    
                else:
                    
                    if not os.path.exists('./multi_step/trained_models'):
                        os.makedirs('./multi_step/trained_models')
                    model_file = os.path.join('./multi_step/trained_models',
                                              '{}_length_{}.h5'.format(model.freq_name, series_length
                                                                                 ))
                    # save model 
                    hmodel.save(model_file)
                    #####################################################
                    
                    
                # for test mode only # 
                if test_mode:
                    prediction[series_length] = denormalise(prediction[series_length], train_mean, train_std)    
                        
            #####################################################################            
            ############ AFTER ALL TRAINING IS DONE #############################
            #### USE THE COMBINATION OF DIFFERENT MODELS FOR EACH HORIZON #######
            #####################################################################
            
            if test_mode :
                
                # make final predictions and fill prediction dictionary with results 
                blend_predictions(prediction,num_values.values-model.horizon)
                
                # write results 
                with open(log_file, 'a') as f:
                    # write accuracy 
                    f.write('\n\n{} ACCURACY'.format(model.freq_name.upper()))
                    
            
                ################ for all training lengths ##################     
                for series_length in training_lengths:
                    
                    # accepted training lengths 
                    train_length_ok = np.logical_or(series_length == min(training_lengths),
                                                    num_values.values - model.horizon >= series_length)
                    
                    # log files # 
                    with open(log_file, 'a') as f:
                        f.write('\nAccuracy for series with training_length >= {}'.format(series_length))
                        for train_length in training_lengths + [-1]:
                            
                            if train_length > series_length:
                                continue
                            
                            curr_prediction = prediction[train_length]
                            
                            # initialize arrays for metrics
                            mase_arr = np.zeros([len(all_data)])
                            smape_arr = np.zeros(len(all_data))
                            
                           
                            for i in range(len(all_data)):
                                
                                # actual series
                                row = series.iloc[i].values
                                row = row[np.logical_not(np.isnan(row))]
                                
                                # calculate mase matric FOR CURRENT SERIES 
                                # mase for this frequency and horizon and current prediction (for this time series, for all forecasts)
                                # values_before_horizon,values_of_horizon,predictions for horizon (forecasts) , frequency
                                mase_arr[i] = mase(row[:-model.horizon], row[-model.horizon:], curr_prediction[i, :], model.freq)
                                
                                # calculate smape metric FOR CURRENT SERIES 
                                # difference between actual values of forecast horizon and horizon forecasts 
                                smape_arr[i] = smape(row[-model.horizon:], curr_prediction[i, :])    
                            
                            
                           
                            last_test_row = int(np.round(len(all_data) * 0.2, 0))
                            test_period = np.arange(len(all_data)) < last_test_row
                            bool_ind = train_length_ok & test_period
                         
                            
                            # write results in log file 
                            with open(log_file, 'a') as f:
                                f.write('\nMASE {}: {}'.format(train_length,
                                                           np.mean(mase_arr[bool_ind][np.logical_not(np.isinf(mase_arr[bool_ind]))])))
                                f.write('\nSMAPE {}: {}'.format(train_length,
                                                            np.mean(smape_arr[bool_ind][np.logical_not(np.isinf(smape_arr[bool_ind]))])))
                                

                                
if __name__ == "__main__":
    
    
    
    #models = build_m_Model()
    #train_multi_step_models(models)
    pass