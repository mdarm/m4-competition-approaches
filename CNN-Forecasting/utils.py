#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:51:43 2023

@author: st_ko
"""

#from quarterly_model import build_Model
import datetime as dt
import pandas as pd
import os 
from read_data import *
import tensorflow as tf
import keras as ks
import numpy as np
from metrics import *
import matplotlib.pyplot as plt
from build_models import *
from read_data import *




# functions to plot the predictions for each horizon_step (each time step prediction)  and each model 
def plotAccuracy(history,horizon_step,series_length,freq_name):
    # create the plots directories 
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig=plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy : ' + ' '+freq_name +' '+ str(series_length) +' ' + str(horizon_step) )
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if not (os.path.exists(os.path.join('plots','Accuracy',freq_name ,str(series_length)) ) ):
        os.makedirs(os.path.join('plots','Accuracy',freq_name ,str(series_length)))
    plt.savefig(os.path.join('plots','Accuracy',freq_name ,str(series_length)) +'/'+ str(horizon_step)+ '.png')
    plt.close(fig)
    #plt.show()
    
    

def plotLoss(history,horizon_step,series_length,freq_name):
    # create the plots directories 
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig=plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss : ' + ' '+freq_name +' '+ str(series_length) +' ' + str(horizon_step) )
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if not (os.path.exists(os.path.join('plots','Loss',freq_name ,str(series_length)) ) ):
        os.makedirs(os.path.join('plots','Loss',freq_name ,str(series_length)))
    plt.savefig(os.path.join('plots','Loss',freq_name ,str(series_length)) +'/'+ str(horizon_step)+ '.png')
    plt.close(fig)
    #plt.show()


# create a learning_rate scheduler in case we add the option for learning_rate schedule 
def scheduler(epoch, lr):
  if epoch < 200:
    return lr
  else:
    return lr * tf.math.exp(-0.4)





# build all the models and then feed them into this function to do the training 
# This function is based on btrotta train_models.py 
# https://github.com/btrotta/m4/blob/master/train_models.py
#  train function 
def train_models(models):
    
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
        
                
                # enough values for training ?? 
                ## ?? check that all rows (time series) have enough values 
                ## this returns a vector of (414,) with bool values 
                train_length_ok = np.logical_or(series_length == min(training_lengths),
                                                num_values.values - model.horizon >= series_length)
                
                
        
                
                ######################################################
                ### we use a dictionary for each training length !!!
                ### and store the model predictions there ############
                ######################################################
                
                
                #################################################################################
                ############ initialize array of predictions for current training length ########
                #################################################################################
                # make all the predictions for this training_length 
                # predict for all time_series in the set 
                
                if test_mode:
                    
                    
                    # initialise prediction array # 
                    # prediction[series_length ] ?? why 
                    # array holding all predictions for all series 
                    prediction[series_length] = np.zeros([len(all_data), model.horizon])
                    
                    # 
                    curr_prediction = prediction[series_length]
                    curr_prediction[:] = np.nan
                ###################################
                
                #################################################################################
                ##### MAKE ALL PREDICTIONS FOR THE WHOLE HORIZON USING THE MODEL CREATED  #######
                ##### FOR CURRENT TRAINING LENGTH ###############################################
                #################################################################################
                
                for horizon_step in range(model.horizon):
                    
                    # supervised training of the model 
                    train_y = train_y_ext[:,horizon_step]
                    # standardize
                    train_y = (train_y - train_mean_ext) / train_std_ext
                    
                    # drop outliers 
                    
                    # exclude samples which are not within 5 * std # 
                    max_y = 5 * np.std(train_y)
                    
                    train_y[train_y > max_y] = max_y
                    train_y[train_y < -max_y] = -max_y
                    
                    ############# BEGIN TENSORFLOW SETUP ############
                    # clear session and reset default graph, as suggested here, to speed up training
                    # https://stackoverflow.com/questions/45796167/training-of-keras-model-gets-slower-after-each-repetition
                    ks.backend.clear_session()
                    #tf.reset_default_graph()
                    tf.compat.v1.reset_default_graph()
                    
                    # for reproducibility 
                    np.random.seed(0)
                    
                    # start tensorflow session 
                    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                    tf.compat.v1.set_random_seed(0)
                    tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                    
                    ############################################################
                    ############ CREATE MODEL FOR TRAINING/PREDICTION ##########
                    ############################################################
                    
                    # est is constructor taken from the function build_models()
                    model_est = model.model_constructor
                    
                    # build different model based on the frequency 
                    if(model.freq_name=='Daily'):
                        hmodel,epochs,batch_size = model_est(series_length,7,3,20,250,1000)
                    elif(model.freq_name=='Hourly'):
                        hmodel,epochs,batch_size = model_est(series_length,168,3,20,250,1000)
                    elif(model.freq_name == 'Weekly'):
                        hmodel,epochs,batch_size = model_est(series_length,52,4,52,250,1000)
                    elif(model.freq_name == 'Monthly'):
                        hmodel,epochs,batch_size = model_est(series_length,12,6,50,250,1000)
                    elif(model.freq_name == 'Quarterly'):
                        hmodel,epochs,batch_size = model_est(series_length,4,4,50,250,1000)
                    else :
                        # 'Yearly'
                        hmodel,epochs,batch_size = model_est(series_length,2,4,20,250,1000)
                        
                    
                    #######################################
                    ########### BEGIN TRAINING ############
                    #######################################
                    
                    # initialize shceduler in case we use it 
                    #sched = tf.keras.callbacks.LearningRateScheduler(scheduler)
                    history = hmodel.fit(train_x_ext, train_y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                            callbacks=[ks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)]
                                       )
                    
                    
                    
                    # save the plots for the accurracy and loss
                    # plot only loss function graph for each horizon step and each model/frequency 
                    # plotAccuracy(history,horizon_step,series_length,model.freq_name)
                    plotLoss(history,horizon_step,series_length,model.freq_name)
                    
                    
                    
                    ####### for test mode ######
                    if test_mode :
                        # use as train all the initial data from the series (no extension)
                        # we keep series_length number of values from all the series 
                        train_x = all_data[:,:series_length]
                        train_x, train_mean, train_std = normalise(train_x)
                        
                        ###########################################################################
                        ############# MAKE PREDICTIONS FOR THIS Dt (THIS TIME MOMENT) #############
                        ###########################################################################
                        # array of predictions for all series 
                        # if train_length_ok for series_i = True , then predict using the model
                        # else if train_length_ok for series_i == False , then take current_prediction
                        # and set current_prediction of this as the answer 
                        # is train_length_ok always TRUE ?? 
                        # What do we do in case of False 
                        # Do we keep the curr_prediction NAN values ?? 
                        curr_prediction[:, horizon_step] = np.where(train_length_ok, hmodel.predict(train_x).flatten(),
                                                                    curr_prediction[:, horizon_step])
                    
                    else:
                        ####### create folder and save current model ########
                        # save model #
                        if not os.path.exists('trained_models'):
                            os.mkdir('trained_models')
                        model_file = os.path.join('trained_models',
                                                  '{}_length_{}_step_{}.h5'.format(model.freq_name, series_length,
                                                                                     horizon_step))
                        # save model 
                        hmodel.save(model_file)
                        #####################################################
                    
                    
                # for test mode only # 
                if test_mode:
                    # denormalise
                    # make prediction for this training length and destandardize it to take the actual prediction #
                    # and save it into the array 
                    # get original predictions for all series 
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
                        # we have added "-1" index as the final additional
                        # index for the combination of models for each horizon 
                        # f.e for hourly we have [840,...,-1]
                        for train_length in training_lengths + [-1]:
                            
                            if train_length > series_length:
                                continue
                            
                            # current prediction for all horizons for this
                            # current training_length
                            curr_prediction = prediction[train_length]
                            
                            # initialize arrays for metrics
                            mase_arr = np.zeros([len(all_data)])
                            smape_arr = np.zeros(len(all_data))
                            
                            #################################
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
                            
                            
                            #############################
                            last_test_row = int(np.round(len(all_data) * 0.2, 0))
                            test_period = np.arange(len(all_data)) < last_test_row
                            bool_ind = train_length_ok & test_period
                            ###################################################
                            
                            # write results in log file 
                            with open(log_file, 'a') as f:
                                f.write('\nMASE {}: {}'.format(train_length,
                                                           np.mean(mase_arr[bool_ind][np.logical_not(np.isinf(mase_arr[bool_ind]))])))
                                f.write('\nSMAPE {}: {}'.format(train_length,
                                                            np.mean(smape_arr[bool_ind][np.logical_not(np.isinf(smape_arr[bool_ind]))])))