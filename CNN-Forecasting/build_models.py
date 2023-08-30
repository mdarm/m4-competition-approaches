#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:44:53 2023

@author: st_ko
"""




import keras as ks
import pandas as pd
import numpy as np
#import metrics
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os



# file to create all the models simulaneously
# define model for hourly predictions #
def hourly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    
    ########### layers gets feature_size = series_length ##########
    input = ks.layers.Input((series_length,))
    
    # create the vector( N,1)
    weekly_input = ks.layers.Reshape((series_length, 1))(input)
    
    ############ average pooling ####################
    # why do we use 168 for pool_size ? maybe test other values 
    # firs pool 168 = 24 x 7 --> so we actually pool weeks [week1,week2,week3....weekN]
    weekly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(weekly_input)
    
    # linear dense layers 
    # first flatten, then pass through linear layers 
    weekly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(weekly_avg))
    weekly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(weekly_hidden1)
    
    
    # one output 
    # the output of all weeks together --> an aggregate 
    #############################################################################
    weekly_output = ks.layers.Dense(units=1, activation='linear')(weekly_hidden2)
    #############################################################################
    
    
    # ---------- differences -------------- #
    
    # upsampling 
    weekly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(weekly_avg)




    ######################### calculate weekly_difference : input - weekly_avg_up ##########
    # hourly - weekly_avg 
    daily_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(weekly_avg_up)  ])
    
    
    ############################## we created the difference between : (hour-corresponding_avg_week_data)
    
    
    
    
    # reshape : what kind of reshape do we do here ???
    
    # change reshape here --> we can try other combinations instead of //7
    daily_diff_input = ks.layers.Reshape((series_length // 7, 7, 1))(daily_diff)
    
    
    # convolutional layers #
    # try more filters 
    # this gives (1,series_length//7,1,3) tensor 
    # test also with more filters 
    daily_diff_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, 7), strides=(1, 7),padding='valid')(daily_diff_input)
    
    
    
    
    # linear dense layers 
    # simple 20 outputs 
    daily_diff_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(daily_diff_conv))
    daily_diff_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(daily_diff_hidden1)
    
    # output 1 
    #####################################################################################
    daily_diff_output = ks.layers.Dense(units=1, activation='linear')(daily_diff_hidden2)
    #####################################################################################
    
    ################# average poolong ################ 
    
    # we pool days now : [day1,day2....dayN]
    daily_avg = ks.layers.AveragePooling1D(pool_size=24, strides=24, padding='valid')(weekly_input)
    
    daily_avg_up = ks.layers.UpSampling1D(size=24)(daily_avg)
    
    
    # define hourly difference : input - daily_avg_up 
    #(daily_avg - hourly )
    # SO THEN from every hour we subtract a
    hourly_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(daily_avg_up)])
    hourly_diff_input = ks.layers.Reshape((series_length // 24, 24, 1))(hourly_diff)
    
    
    # convolutional layers #
    hourly_diff_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, 24), strides=(1, 24),padding='valid')(hourly_diff_input)
    
    # dense linear layers 
    hourly_diff_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(hourly_diff_conv))
    hourly_diff_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(hourly_diff_hidden1)
    
    ##########################################################################################
    hourly_diff_output = ks.layers.Dense(units=1, activation='linear')(hourly_diff_hidden2)
    #########################################################################################
    
    
    
    # add together weekly
    # the monthly should be added in a smaller percentage 
    output = ks.layers.Add()([weekly_output, daily_diff_output, hourly_diff_output])


    ############### Build model #################
    est = ks.Model(inputs=input, outputs=output)
    # use adam optimizer
    # use lr = 0.01 
    # use loss = mse
    # use epochs = 250 
    # bath size = 1000
    
    # change also the learning_rate --> 0.0001  (default=0.001)
    
    ###################### DEFINE MODEL PARAMETERS ###################################
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
    epochs = epochs  #60 # 40 # default 250 
    batch_size = bs
    return est, epochs, batch_size
    
    
    

def quarterly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    
    
    input = ks.layers.Input((series_length,))
    quarterly_input = ks.layers.Reshape((series_length, 1))(input)

    # instead of taking the average of year let's take the average of each pair of 2 quarters
    quarterly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(quarterly_input)


    # naive prediction will be taken into account
    # WE WILL TEST BY ADDING THE NAIVE PREDICTIONS AS WELL (IN A LESSER EXTENT)
    # AND ALSO WITHOUT THE NAIVE
    naive_1 = tf.roll(input, axis=0, shift=1)
    # feed naive into a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
    


    quarterly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(quarterly_avg))
    quarterly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(quarterly_hidden1)
    quarterly_output = ks.layers.Dense(units=1, activation='linear')(quarterly_hidden2)


    # ?? size = 4 ? or 2
    quarterly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(quarterly_avg)


    periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(quarterly_avg_up)])

    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)





    # change convolutional filters
    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                     padding='valid')(periodic_input)

    # change units
    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)

    # TODO : TEST WITH AND WITHOUT THE NAIVE ADDED
    '''0.2 * naive_output''' 
    #output = ks.layers.Add()([quarterly_output, periodic_output,naive_output ])
    
    # i also combine the naive solution and feed all the outputs into an average aggregate 
    output = tf.keras.layers.Average()([quarterly_output, periodic_output,naive_output ])


    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse','accuracy'])
    epochs = epochs
    batch_size = bs
    return est, epochs, batch_size




def weekly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):

    # if we make the assumption of full year (with february + 1 day)
    if series_length == 52:
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,1))

    
        # average of input for one year 
        # maybe pool every 2 weeks and not from all the 52 weeks 
        yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=1, padding='valid')(input)
        # reshape into (1,1)
        yearly_avg2 = ks.layers.Reshape((1,1))(yearly_avg)
        
        
        
        # usampling # 
        yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)
        
        
        periodic_diff = ks.layers.Subtract()([input, yearly_avg_up])
        
        
        periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


        periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),padding='valid')(periodic_input)
        
        periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
        periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
        periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)


        # pass the naive predictions through a fcn
        naive_1 = tf.roll(input, axis=0, shift=1)
        naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
        naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
        naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
        
        # final output 
        output = ks.layers.Add()([periodic_output, naive_output])
        
    
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
        epochs = epochs
        batch_size = bs


    else:

        #
        input = ks.layers.Input((series_length,))
        yearly_input = ks.layers.Reshape((series_length, 1))(input)

        # calculate the naive solution
        naive_1 = tf.roll(input, axis=0, shift=1)

        # yearly average
        yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(yearly_input)
        yearly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_avg))
        yearly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(yearly_hidden1)
        yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)


        # usampling # 
        yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)


        periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(yearly_avg_up)])


        periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


        periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                         padding='valid')(periodic_input)
        
        periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
        periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
        periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)
        
        
        
        # pass the naive predictions through a fcn
        naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
        naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
        naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
        
        output = ks.layers.Add()([yearly_output, periodic_output])
        #output = tf.keras.layers.Average()([yearly_output, periodic_output, naive_output ])
        
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
        epochs = 250
        batch_size = 1000
    return est, epochs, batch_size





def daily_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((series_length,1))(input)
    
    
    # naive model 
    naive_1 = tf.roll(input, axis=0, shift=1)
    # feed naive into a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
    
    
    # weekly avg # 
    weekly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(weekly_input)
    weekly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(weekly_avg))
    weekly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(weekly_hidden1)
    weekly_output = ks.layers.Dense(units=1, activation='linear')(weekly_hidden2)
    # average upsampling 
    weekly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(weekly_avg)
    
    
    
    # subtract from the naive solution the average and not from the original series 
    periodic_diff = ks.layers.Subtract()([weekly_input,weekly_avg_up])
    
    
    # tensor and convolution 
    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)
    
    # convolutional 
    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),padding='valid')(periodic_input)
    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)
    
    
    
    # output 
    #output = ks.layers.Add()([weekly_output, periodic_output,naive_output])
    output = tf.keras.layers.Average()([weekly_output, periodic_output,naive_output])
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
    
    epochs = 250
    batch_size = 1000
    return est, epochs, batch_size




def monthly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):


    input = ks.layers.Input((series_length,))
    yearly_input = ks.layers.Reshape((series_length, 1))(input)


    # pool average of 1 year
    # 12 months
    yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(yearly_input)

    # 50 units fcn
    yearly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_avg))
    yearly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(yearly_hidden1)

    # output 1 unit fcn
    yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)




    # from each month subtract the average of the corresponding year
    yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)
    # upsample and create the differences
    periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(yearly_avg_up)])


    # create tensor for cnn
    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                     padding='valid')(periodic_input)


    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)


    # we are going to use each k month
    # and use the value for the k+1 month as an additional term to be added in the output
    # (naive method term)
    # substitute each (k+1) month with k month
    # first shift each month value towards bigger index (each value moves one time step forward)
    # THE LAST MONTH WILL GO CYCLICALLY AND REPLACE THE 1ST MONTH !!! THIS IS NOT A GOOD IDEA
    # SINCE WE LOSE THE 1ST MONTH
    # MAYBE WE SHOULD COPY THE 1ST MONTH VALUE (WHICH WAS SHIFTED TO THE 3ND VALUE)
    # BACK TO THE 1ST VALUE AS WELL
    naive= tf.roll(input,shift=1,axis=0)
    # pass the naive predictions through a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)

    # output 1 unit fcn
    #yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)


    # i also add the naive output in a lesser extent
    #output = ks.layers.Add()([yearly_output, periodic_output,0.4 * naive_output])
    output = tf.keras.layers.Average()([yearly_output, periodic_output,naive_output ])


    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
    epochs = 250
    batch_size = 1000
    return est, epochs, batch_size



# define the yearly model
def yearly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    # different yearly_count depending on series length 
    # for more years try to take longer intervals of past years 
    '''
    if(series_length == 10) :
        yearly_count = 2
    elif(series_length == 20):
        yearly_count = 2
    else :
        yearly_count = 2
    '''

    # read the input shape
    input = ks.layers.Input((series_length,))

    # reshape into N x 1
    yearly_input = ks.layers.Reshape((series_length, 1))(input)

    # average pooling for k years
    # 2 years period
    yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='same')(yearly_input)

    # define yearly diff
    yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)

    # if the aggregates are more we pass them into a fcn to get the correct dimensions
    # or we just drop the out of series_length values
    #yearly_avg_up = yearly_avg_up[:,:series_length,:]




    # also will modify the units for next training
    # maybe add more units
    yearly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_input))
    yearly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(yearly_hidden1)
    yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)





    # define yearly diff
    yearly_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(yearly_avg_up)  ])


    # reshape yearly_diff (check if it is correct)
    yearly_diff_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(yearly_diff)


    # add convolutional filter
    # maybe run with 10 filters or 5 filters
    yearly_diff_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                       padding='valid')(yearly_diff_input)

    # hidden linear
    yearly_diff_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_diff_conv))


    # output 1
    #####################################################################################
    yearly_diff_output = ks.layers.Dense(units=1, activation='linear')(yearly_diff_hidden1)


    # add yearly output with convolutional produced output to produce final output
    output = ks.layers.Add()([yearly_output, yearly_diff_output])


    est = ks.Model(inputs=input, outputs=output)
    # change learning rate to 0.0001
    est.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse','accuracy'])
    epochs =  epochs
    batch_size = bs
    return est, epochs, batch_size




# build the models
def build_Model():
    Model = namedtuple('Model', ['freq_name', 'horizon', 'freq', 'model_constructor', 'training_lengths',
                                 'cycle_length', 'augment'])
    daily = Model('Daily', 14, 1, daily_model, [98,364], 7, True)
    monthly = Model('Monthly', 18, 12, monthly_model, [48, 120, 240], 12, False)
    yearly = Model('Yearly', 6, 1, yearly_model, [10, 20, 30], 1, False)
    quarterly = Model('Quarterly', 8, 4, quarterly_model, [20, 48, 100], 4, False)
    weekly = Model('Weekly', 13, 1, weekly_model, [52, 520, 1040], 52, True)
    hourly = Model('Hourly', 48, 24, hourly_model, [672], 7*24, True)
    return [daily,monthly,quarterly,weekly,hourly,yearly]
