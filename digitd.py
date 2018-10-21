from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(theta,X,y,lmda):
    temp1 = np.multiply(y,np.log(sigmoid(np.dot(X,theta))))
    temp2 = np.multiply(1-y,np.log(1-sigmoid(np.dot(X,theta))))
    return -np.sum(temp1+temp2)/ 5000 + np.sum(theta[1:]**2)*lmda/2*5000
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
one=np.ones((5000,1))
X = np.hstack((one,X))
lmda = 0.1
theta = np.zeros((10,401)) 
for i in range(10):
    dig = i if i else 10
    theta[i] = opt.fmin_cg(f = cost, x0 = theta[i],  fprime = gradient, args = (X, (y == dig).flatten(), lmda),   maxiter = 50)
final = np.dot(X,theta.T)
print(theta)
print(final)
for i in range(5000):
    res = np.argmax(final[i])
    print(res)

