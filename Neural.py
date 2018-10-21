from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(theta,X,y,lmda):
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    theta1 = theta[0:401*26-1].reshape(401,26) 
    temp1 = np.multiply(y,np.log(sigmoid(np.dot(X,theta))))
    temp2 = np.multiply(1-y,np.log(1-sigmoid(np.dot(X,theta))))
    return np.sum(temp1+temp2)/ (-5000) + np.sum(theta[1:]**2)*lmda/2*5000
def gradient(theta, X, y, lmda):
    temp = sigmoid(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / 5000 + theta * lmda / 5000
    temp[0] = temp[0] - theta[0] * lmda / 5000
    return temp
data = loadmat('ex3data1.mat')

X = data['X']
y = data['y']
_, arr = plt.subplots(10,10,figsize=(15,15))
for i in range(10):
    for j in range(10):
       arr[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       arr[i,j].axis('off')    
plt.show() 

