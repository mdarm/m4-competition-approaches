#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:40:42 2023

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
from multi_step_model import *
from utils import *
from train_multi_step_models import *


# create all the models 
models = build_m_Model()

# train the models 
train_multi_step_models(models)



# Then just run predict_multi_step_models.py and it will load the models 
# do the predictions, and write the results to "multi_Step/predictions" directory.
