import pandas as pd

# read the predicted probabilities
probs = pd.read_csv('predicts.csv', header=None)
probs.columns = ['prob']
probs['pred'] = (probs['prob'] > 0.66)*1
# classify as robocall if prob > 0.66
test = pd.read_csv('data/raw/ftc_testing_set_Corrected.csv')
test['Robocall'] = probs['pred']
test.to_csv('final.csv', index = False)
