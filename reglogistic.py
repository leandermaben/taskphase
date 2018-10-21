import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
def sigmoid(z):
  return 1/(1+np.exp(-z))
def cost(theta,X,Y,lmda):
    m = len(Y)
    temp1 = np.multiply(y,np.log(sigmoid(np.dot(X,theta))))
    temp2 = np.multiply(1-y,np.log(1-sigmoid(np.dot(X,theta))))
    return -np.sum(temp1+temp2)/ m + np.sum(theta[1:]**2)*lmda/2*m
def gradient(theta, X, y, lmda):
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / 5000 + theta * lmda / 5000
    temp[0] = temp[0] - theta[0] * lmda / 5000
    return temp
df = pd.read_csv('ex2data2.txt', header = None)
X = df.iloc[:,:-1]
Y = df.iloc[:,2]
test = y == 1
passed = plt.scatter(X[test][0].values, X[test][1].values)
failed = plt.scatter(X[~test][0].values, X[~test][1].values)
plt.xlabel('Test1')
plt.ylabel('Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()
output = opt.fmin_tnc(f = cost, x0 = theta.flatten(), fprime = gradient, 
                         args = (X, y.flatten(), lmda))
theta = output[0]

