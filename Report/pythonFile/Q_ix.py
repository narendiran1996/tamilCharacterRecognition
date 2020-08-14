#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datasetLoc = '../../../Datasets/kcl2.csv'
DataSet = np.genfromtxt(datasetLoc, delimiter=',')
ataSet[:,0:2]
Y=DataSet[:,1]
X=X.reshape(-1,2)
plt.plot(X[:,0],X[:,1],'r*')
plt.show()
kmeans = KMeans(n_clusters=35, random_state=0).fit(X)
ans = kmeans.cluster_centers_
plt.plot(X[:,0],X[:,1],'r.')
plt.plot(ans[:,0],ans[:,1],'bo')
plt.show()
