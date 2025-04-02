import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from trained import W1, b1, W2, b2
from train import forward_prop, get_predictions, X_train, Y_train

def make_predictions(X):
  _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
  predictions = get_predictions(A2)
  return predictions

def test_prediction(index):
  current_image = X_train[:, index, None]
  prediction = make_predictions(current_image)
  label = Y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)

  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation="nearest")
  plt.show()

test_prediction(0)
test_prediction(1)
test_prediction(2)
test_prediction(3)