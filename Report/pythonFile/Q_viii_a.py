#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

datasetLoc = '../../../Datasets/exam_vs_adm.csv'
DataSet = np.genfromtxt(datasetLoc, delimiter=',')
X,Y=DataSet[:,:-1],DataSet[:,-1].reshape(-1,1)
xpos=X[(Y==1)[:,0]]
xneg=X[(Y==0)[:,0]]

plt.plot(xpos[:,0],xpos[:,1],'g*')
plt.plot(xneg[:,0],xneg[:,1],'b.')
plt.show()

X=np.hstack((np.ones((X.shape[0],1)),X))
Y=Y.ravel()

clf = LinearSVC(penalty='l2',random_state=0, tol=1e-5,dual=False)
clf.fit(X, Y)
print(clf.coef_)

print('accuracy : ',100.0*np.sum(clf.predict(X) == Y)/Y.shape[0])
