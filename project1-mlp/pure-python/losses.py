'''
This module defines loss functions that can be used to train deep learning models.
'''

import math

# Mean Squared Error Loss: MSE = (1/n) * sum((y_true - y_pred)^2)
class MSELoss:
    def forward(self, y_true, y_pred):
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

    def backward(self, y_true, y_pred):
        n = len(y_true)
        return [2 * (p - t) / n for t, p in zip(y_true, y_pred)]
    
# Cross-Entropy Loss: CE = -sum(true * log(predicted)) for one-hot encoded targets and softmax outputs -- Assume y_true is one-hot encoded and y_pred is the output of a softmax
class CrossEntropyLoss:
    def forward(self, y_true, y_pred):
        return -sum(t * math.log(p) for t, p in zip(y_true, y_pred)) / len(y_true)

    def backward(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)