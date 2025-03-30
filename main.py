# perceptron. This neural network has no hidden layers. Only inputs, neuron and output
import numpy as np

# sigmoid function. Normalizes values to 0 to 1 
# https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# input
training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3,1)) - 1
print(f"Random starting synaptic weights:\n{synaptic_weights}")

# can change the range values to get a better aproximation. Value will never be 1 or 0 due to nature of sigmoid function
for iteration in range(50000):
  input_layer =  training_inputs
  #output
  outputs = sigmoid(np.dot(input_layer,	synaptic_weights))

  error = training_outputs - outputs
  adjustments = error * sigmoid_derivative(outputs)
  synaptic_weights += np.dot(input_layer.T, adjustments)

print(f"Synaptic weights after training:\n{synaptic_weights}")
print(f"Outputs after training:\n{outputs}")