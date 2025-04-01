import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")
data.head()
# print(data)

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
X_train = data_train[0]
Y_train = data_train[1:n]

def init_params():
  W1 = np.random.ranom(10, 784) - 0.5
  b1 = np.random.random(10, 1) - 0.5
  W2 = np.random.ranom(10, 10) - 0.5
  b2 = np.random.ranom(10, 1) - 0.5
  return W1, b1, W2, b2

def ReLU(Z):
  return np.maximum(0,Z)

def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, x):
  Z1 = W1.dot(x) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(A1)
  return Z1, A1, Z2, A2

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def deriv_ReLU(Z):
  return Z > 0

def back_prop(Z1, A1, Z2, A2, w2, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2, 2)
  dZ1 = w2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1, 2)
  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  