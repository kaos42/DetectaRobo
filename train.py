
# coding: utf-8

import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Read in the training dataframe
dat = pd.read_csv('data/clean/train.csv')

# separate out target
labels = dat['Robocall'].values
del dat['Robocall']

# Convert necessary columns to char
clist = ['Day','DoW','Quarter','TimeOfDay']
for c in clist:
    dat[c] = dat[c].astype(str)

# Convert to numeric matrix for xgboost
train = pd.get_dummies(dat)
train = train.values
dtrain = xgb.DMatrix(train, label=labels)

# Training with xgboost, cross validation
param = {'bst:max_depth':9, 'bst:eta':0.01, 'silent':1, 'objective':'binary:logistic', 'silent':1, 'bst:min_child_weight':15, 'bst:gamma':0.1 }
num_round = 3000
watchlist = [ (dtrain,'train') ]
bst = xgb.train(param, dtrain, num_round, watchlist)
# save model
bst.save_model('xgb1.model')
