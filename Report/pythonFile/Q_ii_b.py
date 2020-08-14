#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def Scale(x):
    return (x-np.min(x))/(np.max(x) -  np.min(x))
def hyp(X_,theta):
    R=np.dot(X_,theta)
    return R
def cost(X_,Y_,theta):
    m=Y_.shape[0]
    return (1/(2.0*m))*np.sum((hyp(X_,theta) - Y_)**2)
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

DataSet = np.genfromtxt('../../../Datasets/fish_length.csv', \
delimiter=',')
X= DataSet[:,1:3]
Y = DataSet[:,3]
Y= Scale(Y)
Y=Y.reshape(-1,1)
X = Scale(X)
X = np.hstack((np.ones((X.shape[0],1)),X))
plt.plot(X[:,1],Y,'g*')
plt.show()
plt.plot(X[:,2],Y,'r.')
plt.show()

theta_final, cost_ = gradient_descent(X,Y,0.5,100000)
print('Final Coefficient and intercept: ',\
theta_final[1:],theta_final[0][0])
plt.plot(cost_)
plt.show()

lr = LinearRegression()
lr.fit(X[:,1:], Y)
print('Final Coefficient and intercept: ',lr.coef_,lr.intercept_)
