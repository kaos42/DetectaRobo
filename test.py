# coding: utf-8

import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Read in the training dataframe
dat = pd.read_csv('data/clean/test.csv')

# get rid of Robocall col with NAs
del dat['Robocall']

# Convert necessary columns to char
clist = ['Day','DoW','Quarter','TimeOfDay']
for c in clist:
    dat[c] = dat[c].astype(str)

# Convert to numeric matrix for xgboost
test = pd.get_dummies(dat)
test = test.values
dtest = xgb.DMatrix(test)

# predictions
# bst = pickle.load( open('xgb1.model', 'rb'))
bst = xgb.Booster()
bst.load_model('xgb1.model')
preds = bst.predict(dtest)
np.savetxt('predicts.csv', preds, delimiter=',')
