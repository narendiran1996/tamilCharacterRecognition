#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def Scale(x):
    return (x-np.min(x))/(np.max(x) -  np.min(x))
def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))
def hyp(X_,theta):
    R=np.dot(X_,theta)
    return sigmoid(R)
def cost(X_,Y_,theta):
    m=Y_.shape[0]
    p1 = np.multiply(Y_,np.log(hyp(X_,theta)))
    p2 = np.multiply(1-Y_,np.log(1- hyp(X_,theta)))
    return (-1.0/m) * np.sum( p1 + p2 )
def update_theta(X_,Y_,theta,alpha):
    m=Y_.shape[0]
    return((alpha/m)*np.dot(X_.T,(hyp(X_,theta) - Y_)))
def gradient_descent(X_,Y_,alpha = 0.01, num_iter = 100):
    theta = np.zeros((X_.shape[1],1))
    cost_ =[]
    for i in range(num_iter):                
        theta = theta - update_theta(X_,Y_,theta,alpha)
        cost_.append(cost(X_,Y_,theta))
    return theta,cost_

DataSet = np.genfromtxt('../../../Datasets/exam_vs_adm.csv', \
delimiter=',')
X= DataSet[:,:2]
Y = DataSet[:,2]
X=Scale(X)
Y=Y.reshape(-1,1)
X = np.hstack((np.ones((X.shape[0],1)),X))

fig,a =  plt.subplots(4,2,figsize=(15,15))
alphas = np.asarray([0.1,1.5,2.5,5,10,15,20,50]).reshape(4,2)

for i in range(4):
    for j in range(2):
        theta_final, cost_ = gradient_descent(X,Y,alphas[i,j],100)
        a[i][j].plot(cost_)
        a[i][j].set_title('alpha = '+str(alphas[i,j]))
        a[i][j].xlabel = 'No. of iterations'
plt.show()
