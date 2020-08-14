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
    return (-1.0/m) * np.sum(p1 + p2)
def update_theta(X_,Y_,theta,alpha):
    m=Y_.shape[0]
    return((alpha/m)*np.dot(X_.T,(hyp(X_,theta) - Y_)))
def gradient_descent(X_,Y_,alpha = 0.01, num_iter = 100): 
    theta = np.zeros((X_.shape[1],1))
    print('Inital Cost : ',cost(X_,Y_,theta))
    cost_ =[]
    for i in range(num_iter):                
        theta = theta - update_theta(X_,Y_,theta,alpha)
        cost_.append(cost(X_,Y_,theta))
    print('Final Cost : ',cost(X_,Y_,theta))
    return theta,cost_
def accuracy(X_,Y_,theta):
    m=Y_.shape[0]
    return (100.0 * np.sum((hyp(X_,theta)>0.5) == Y_)) / m

DataSet = np.genfromtxt('../../../Datasets/exam_vs_adm.csv', \
delimiter=',')
X= Scale(DataSet[:,:2])
Y = DataSet[:,2].reshape(-1,1)
xpos=X[(Y==1)[:,0]]
xneg=X[(Y==0)[:,0]]
plt.plot(xpos[:,0],xpos[:,1],'g*')
plt.plot(xneg[:,0],xneg[:,1],'b.')
plt.show()
X = np.hstack((np.ones((X.shape[0],1)),X))
theta_final, cost_ = gradient_descent(X,Y,0.01,8500)
print('Final Coefficient and intercept: ',\
theta_final[1:],theta_final[0][0])
print('Accuracy: ',accuracy(X,Y,theta_final),'%')
plt.plot(cost_)
plt.show()
