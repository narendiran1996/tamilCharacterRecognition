#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Dfile = '../../../Datasets/millcost.csv'
dataset = np.genfromtxt(Dfile, delimiter = ',')
X = dataset[:,1].reshape(-1,1)
Y = dataset[:,2].reshape(-1,1)

plt.plot(X,Y,'g*')
plt.xlabel('Production (Thousands of dozens of pairs)')
plt.ylabel('Cost($1000s)')

lr = LinearRegression().fit(X,Y)
print('Coefficient and constant: ', lr.coef_[0], lr.intercept_)
Ynew = lr.coef_[0] * X + lr.intercept_
plt.plot(X,Y,'g*')
plt.plot(X,Ynew,'r')
plt.xlabel('Production (Thousands of dozens of pairs)')
plt.ylabel('Cost($1000s)')

print('Mean squared Error: ', mean_squared_error(Y,Ynew))
