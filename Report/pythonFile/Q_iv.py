#!/usr/bin/env python
import numpy as np
from sklearn import tree

np.random.seed(seed=4)

dataLoc = '../../../Datasets/transfusion.csv'
DataSet = np.genfromtxt(dataLoc , delimiter=',',skip_header=1)
X= DataSet[:,1:4]
Y = DataSet[:,4]
Y=Y.reshape(-1,1)
percent = 85
split = int((percent/100.0) * X.shape[0])
rand_indx = np.random.choice(range(X.shape[0]), X.shape[0], \
replace=False)
Xtrain,Ytrain = X[rand_indx[:split],:],Y[rand_indx[:split],:]
Xtest,Ytest = X[rand_indx[split:],:],Y[rand_indx[split:],:]

DST = tree.DecisionTreeClassifier(criterion='entropy')
fitted_DST =DST.fit(Xtrain,Ytrain)


print('Training accuracy : ',fitted_DST.score(Xtrain,Ytrain)*100.0)
print('Testing accuracy : ',fitted_DST.score(Xtest,Ytest)*100.0)



print(DST.feature_importances_)
