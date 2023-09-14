#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:41:09 2023

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


def quarterly_model(series_length,epochs,bs,horizon):
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)

    # 4 quarters
    if(series_length == 4):
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv3))

    elif(series_length == 8):
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
            conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
            output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))
    else:
            # 12,24,36
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            # stack on top convolutional of 3 successive days after conv2
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            # addd conv, 2 successive pairs with stride = 1 on top
            # or maybe add kernel 1,3 instead again
            conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
            conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
            output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))


    # concatenate both outputs
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    # weighted combination of both the outputs
    output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size
    
    
# 


# new weekly model 
def weekly_model(series_length,epochs,bs,horizon):

    # if we make the assumption of full year (with february + 1 day)
    if series_length == 13:
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


        # ------ Try only with a convolutional model -----------#
        
        # both 2 last weeks influence 
        # we get (None,1,2,32)  // how much each week influences the result
        # convolutional on  groups of 4 weeks (in each group we take the last week as well)
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
        
    
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        
        
        
        # stack on top convolutional of 3 successive weeks after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        
        
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        

        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        
        
        
        # now we have 3 units 
        # we will flatten them and feed to a fully connected network
        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))
        
        
        # concatenate both outputs 
        # we get shape (None,2)
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
        
    elif (series_length == 26) :
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)

    
        # ------ Try only with a convolutional model -----------#
        
        # groups of 4 weeks --> first output 
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        conv1b = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(conv1)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1b))
        
        # second convolutional model 
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv5)
        conv7 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv6)
        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv7))
        
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        
        output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
        
    elif (series_length == 52) :
        

        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


        # ------ Try only with a convolutional model -----------#
        
        # groups of 4 weeks --> first output 
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        conv1b = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(conv1)
        conv1c = ks.layers.Conv2D(filters = 32,kernel_size=(1,3),strides = (1,1), padding = 'valid',use_bias=True,)(conv1b)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1c))
        
        
        # second convolutional model 
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv5)
        conv7 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv6)
        conv8 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv7)
        #conv9 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv8)
        
        
        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv8))
        
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        
        output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)    
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    epochs = 250
    batch_size = 1000    
 
        
    return est, epochs, batch_size




# new convolutional daily model 
# 
def daily_model(series_length,epochs,bs,horizon):
    
    
    
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)
    
    
    
    # ------ Try only with a convolutional model -----------#
    
    # both 2 last weeks influence 
    # we get (None,1,2,32)  // how much each week influences the result
    conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,7),strides = (1,7), padding = 'valid',use_bias=True,)(weekly_input)
    fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
    
    
    #adding pairs of 2 successive days history 
    conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
    
    # stack on top convolutional of 3 successive days after conv2
    conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
    
    
    # addd conv, 2 successive pairs with stride = 1 on top 
    conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
    
    
    # add conv , 3 successive pairs with stride = 1 on top
    conv5 = ks.layers.Conv2D(filters = 128, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
    
    # now we have 3 units 
    # we will flatten them and feed to a fully connected network
    output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))
    
    
    # concatenate both outputs 
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
    # weighted combination of both the outputs 
    #comb_output = Weighted_add(1)(comb)
    
    output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
    
    
    #output = ks.layers.Add()([weekly_output, periodic_output,naive_output])
    #output = tf.keras.layers.Average()([weekly_output, periodic_output,y_linear_reg])
    #output = tf.keras.layers.Add()([weekly_output , naive_output])
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size



# 
def monthly_model(series_length,epochs,bs,horizon):

    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


    # how much each month influences the result
    if(series_length == 6):

        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        # stack on top convolutional of 3 successive days after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv4))


    elif(series_length == 8):
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,8),strides = (1,8), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv4))

    else:
        # 12,24,36
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        # stack on top convolutional of 3 successive days after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        # *** maybe add again the same layers as in 6 months case here

        output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))


    # concatenate both outputs
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    # weighted combination of both the outputs
    output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size



# new quarterly model
def yearly_model(series_length,epochs,bs,horizon):
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)
        # 4 quarters
        if(series_length == 2):
                fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(weekly_input))
    
                #adding pairs of 2 years
                # maybe we dont need this since it is still the same fcn from above -- > conv 2->1 or fcn 2 -> 1 ? they are the same
                # conv1 and fc_1 here are the same
                
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,2),strides = (1,1), padding = 'valid',use_bias=True)(weekly_input)
                output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
    
        elif(series_length == 3):
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,3),strides = (1,3), padding = 'valid',use_bias=True)(weekly_input)
                fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
    
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                conv3 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv3))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
                
        elif(series_length == 6):
                # using half a year weighting
                # maybe use strides 3 instead of 6 , here
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
                output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
        elif(series_length == 8):
                # weighting of groups of 4 years ! maybe use 8 years grouping instead
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                # maybe change the stride here 
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
                output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv5))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
        else:
                # 12 years
                # weighting of groups of 4 years ! maybe use 8 years grouping instead
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv1))
    
                conv2 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
                fc_2 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv2))
    
                #adding pairs of 2 successive months history
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv4 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,2),padding = 'valid' ,use_bias = True)(conv4)
                conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,2),padding = 'valid' ,use_bias = True)(conv5)
                output1 = ks.layers.Dense(horizon)(ks.layers.Flatten()(conv6))
                comb = tf.keras.layers.Concatenate()([fc_1,fc_2,output1])
    
        # weighted combination of both the outputs
        
        output = tf.keras.layers.Dense(horizon, input_shape=(None, comb.shape[-1]))(comb)
        
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse'])
        
        epochs = epochs
        batch_size =  bs
        return est, epochs, batch_size




# build the models
def build_m_Model():
    Model = namedtuple('Model', ['freq_name', 'horizon', 'freq', 'model_constructor', 'training_lengths',
                                 'cycle_length', 'augment'])
    
    
    daily = Model('Daily', 14, 1, daily_model, [7,14,21,28,56,84,364], 7, True)
    weekly = Model('Weekly', 13, 1, weekly_model, [13,26,52], 52, True)
    monthly = Model('Monthly', 18, 12, monthly_model, [6, 8, 12,24,36,120,240], 12, False)
    yearly = Model('Yearly', 6, 1, yearly_model, [2, 3, 6, 8, 12,18], 1, False)
    quarterly = Model('Quarterly', 8, 4, quarterly_model, [4, 8, 12], 4, False)
    
    return [daily,weekly,monthly,quarterly,yearly]
    