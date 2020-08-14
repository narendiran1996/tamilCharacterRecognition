#!/usr/bin/env python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

def Scale(x):
    return (x-np.min(x))/(np.max(x) -  np.min(x))
datasetLoc = 'Datasets/transfusion.csv'
DataSet = np.genfromtxt(datasetLoc, delimiter=',',skip_header=1)
X= DataSet[:,1:4]
Y = DataSet[:,4]
Y=Y.reshape(-1,1)
percent = 85
split = int((percent/100.0) * X.shape[0])
rand_indx = np.random.choice(range(X.shape[0]), X.shape[0], replace=False)
Xtrain,Ytrain = X[rand_indx[:split],:],Y[rand_indx[:split],:]
Xtest,Ytest = X[rand_indx[split:],:],Y[rand_indx[split:],:]

Ada_C = AdaBoostClassifier(n_estimators=250)
fitted_Ada_C =Ada_C.fit(Xtrain,Ytrain.ravel())

print('Training accuracy : ',fitted_Ada_C.score(Xtrain,Ytrain)*100.0)
print('Testing accuracy : ',fitted_Ada_C.score(Xtest,Ytest)*100.0)
