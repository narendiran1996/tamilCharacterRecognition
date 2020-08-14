#!/usr/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
np.random.seed(seed=41)

def Scale(x):
    return (x-np.min(x))/(np.max(x) -  np.min(x))

datasetLoc = '../../../Datasets/transfusion.csv'
DataSet = np.genfromtxt(datasetLoc, delimiter=',',skip_header=1)
X= DataSet[:,1:4]
Y = DataSet[:,4]
Y=Y.reshape(-1,1)
percent = 85
split = int((percent/100.0) * X.shape[0])
rand_indx = np.random.choice(range(X.shape[0]), X.shape[0],\
 replace=False)

Xtrain,Ytrain = X[rand_indx[:split],:],Y[rand_indx[:split],:]
Xtest,Ytest = X[rand_indx[split:],:],Y[rand_indx[split:],:]

RForest=RandomForestClassifier(n_estimators=200,max_depth=4,max_features='log2')
fitted_RForest =RForest.fit(Xtrain,Ytrain.ravel())

print('Training accuracy : ',\
fitted_RForest.score(Xtrain,Ytrain)*100.0)
print('Testing accuracy : ',\
fitted_RForest.score(Xtest,Ytest)*100.0)

print(fitted_RForest.feature_importances_)
