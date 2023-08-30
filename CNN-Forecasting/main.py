#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:32:47 2023

@author: st_ko
"""



# main script to do the training 
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
from utils import *


# create all the models 
models = build_Model()

# train the models 
train_models(models)



# Then just run predict.py and it will load the models 
# do the predictions, and write the results to "predictions" directory.