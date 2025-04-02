import numpy as np
import pandas as pd

# reads mnist.csv (0-9 number handwritten dataset) 
data = pd.read_csv("mnist.csv")

# converts data into numpy array
# m, n: rows and columns from the dataset 
# m=60000 -> number of training examples
# n= 785 -> number of pixels (28x28pixels + 1 label)
data = np.array(data)
m, n = data.shape
# shuffles data so it´s not in order 0-9
np.random.shuffle(data)

# transpose first 1000 rows from data (rows turn to columns) each column turns to training example
data_dev = data[0:1000].T
Y_dev = data_dev[0]
# takes all features values from index 1 for all 1000 images [1:785]
# normalizes pixel value to a scale between 0-1
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# training data. Transposes everything from 1000-60000 (rows turn to columns)
# Y_train holds labels corresponding to training images
data_train = data[1000:m].T
Y_train = data_train[0]
# takes all features values from index 1 for all 1000 images [1:785]
# normalizes pixel value to a scale between 0-1
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

Y_train

# returns initialized weights and biasses (W1, W2), (b1, b2)
# 10 - number of neurons in the hidden layer
# W1 -> 784 - number of input features (28pixels x 28pixel)
# W2 -> 10 - number of classes (digits 0-9)
# weights and biasses are values between 0 and 1. Subtraction by .5 turns them into values between -.5 and .5
def init_params():
  W1 = np.random.rand(10, 784) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2

# Rectified Linear Unit activation function
# if z > 0, keep z as is. If z < 0, replace it with 0 
def ReLU(Z):
  return np.maximum(Z, 0)

# converts scores into probabilities that sum up to 1
# np.exp makes all values positive and amplifies large ones
# sum(np.exp) ensures probabilities sum up to 1
def softmax(Z):
  A = np.exp(Z) / sum(np.exp(Z))
  return A

# forward propagation
# Z1 - Compute first hidden layer pre-activation, multiply weight by input and bias
# A1 - ReLU activation function
# Z2 - Computes second hidden layer of pre-activation, multiply weights by A1 and add bias
# A2 - Converrts raw scores into probabilities
def forward_prop(W1, b1, W2, b2, x):
  Z1 = W1.dot(x) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def deriv_ReLU(Z):
  return Z > 0

# creates 0 matrix of shape (examples, classes) [2,5,9] = [[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1]]
# transposes matrix to (class, examples) [[0,0,0],[0,0,0],[1,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,0],[0,0,0],[0,0,1]]
def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

# backward propagation
# finds how much each weight W1, W2 and bias b1, b2 contribute to the error
# propagates the error backward
# dZ2 -> error at the output layer (error in final layer)
# dW2, bd2 -> tells us how to adjust W2 and b2 to reduce error
# dZ1 -> error at the hidden layer
# dW1, db1 -> tells us how to adjust W1 and b1 to reduce error
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
  # converts labels to one_hot encoding
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha  * dW1
  b1 = b1 - alpha  * db1
  W2 = W2 - alpha  * dW2
  b2 = b2 - alpha  * db2
  return W1, b1, W2, b2

# returns the predicted class by finding index of the highest probability
def get_predictions(A2):
  return np.argmax(A2, 0)

# turns predictions into percentage 0-1 0=0%, 1=100%
def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size

# initialize weights and biasses
# iterates through a number of times to reduce loss and improve predictions
# calls forward_prop to calculate activations
# calls back_prop to compute gradients
# updates parameters through update_params
# returns final weights and biasses 
def gradient_descent(X, Y, iterations, alpha):
  W1, b1, W2, b2 = init_params()
  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    if i % 50 == 0:
      print("Iteration: ", i)
      print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
  return W1, b1, W2, b2

# main.py calls for train.py. We only need weights, biasses and trained data
# when main.py is executed, only this part of train.py runs
if __name__ == "__main__":
  W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 2000, 0.10)
  #saves final weights and biasses into trained.npz
  np.savez("trained.npz", W1=W1, b1=b1, W2=W2, b2=b2)
