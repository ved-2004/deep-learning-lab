import numpy as np


class SGD:
    def __init__(self, model, learning_rate = 0.01):
        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.weight_grads
                layer.biases -= self.learning_rate * layer.bias_grads
    

class Momentum:
    def __init__(self, model, learning_rate = 0.01, momentum = 0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocity_weights = [np.zeros_like(layer.weights) for layer in self.model.layers if hasattr(layer, 'weights')]
        self.velocity_biases = [np.zeros_like(layer.biases) for layer in self.model.layers if hasattr(layer, 'biases')]

    def step(self):
        idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                # update velocities
                self.velocities_weights[idx] = self.momentum * self.velocity_weights[idx] + (1 - self.momentum) * layer.weight_grads
                self.velocity_biases[idx] = self.momentum * self.velocity_biases[idx] + (1 - self.momentum) * layer.bias_grads

                # update parameters
                layer.weights -= self.learning_rate * self.velocity_weights[idx]
                layer.biases -= self.learning_rate * self.velocity_biases[idx]  

                idx += 1


class Adagrad: 
    def __init__(self, model, learning_rate = 0.01, epsilon = 1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.cache_weights = [np.zeros_like(layer.weights) for layer in self.model.layers if hasattr(layer, 'weights')]
        self.cache_biases = [np.zeros_like(layer.biases) for layer in self.model.layers if hasattr(layer, 'biases')]

    def step(self):
        idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                # update cache
                self.cache_weights[idx] += layer.weight_grads ** 2
                self.cache_biases[idx] += layer.bias_grads ** 2

                # update parameters
                layer.weights -= self.learning_rate * layer.weight_grads / (np.sqrt(self.cache_weights[idx]) + self.epsilon)
                layer.biases -= self.learning_rate * layer.bias_grads / (np.sqrt(self.cache_biases[idx]) + self.epsilon)

                idx += 1


class RMSProp:
    def __init__(self, model, learning_rate = 0.01, beta = 0.9, epsilon = 1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

        self.cache_weights = [np.zeros_like(layer.weights) for layer in self.model.layers if hasattr(layer, 'weights')]
        self.cache_biases = [np.zeros_like(layer.biases) for layer in self.model.layers if hasattr(layer, 'biases')]

    def step(self):
        idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                # update cache
                self.cache_weights[idx] = self.beta * self.cache_weights[idx]  + (1 - self.beta) * layer.weight_grads ** 2
                self.cache_biases[idx] = self.beta * self.cache_biases[idx] + (1 - self.beta) * layer.weight_grads ** 2

                # update parameters
                layer.weights -= self.learning_rate * layer.weight_grads / (np.sqrt(self.cache_weights[idx]) + self.epsilon)
                layer.biases -= self.learning_rate * layer.bias_grads / (np.sqrt(self.cache_biases[idx]) + self.epsilon)

                idx += 1


class Adam:
    def __init__(self, model, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.first_moment_weights = [np.zeros_like(layer.weights) for layer in self.model.layers if hasattr(layer, 'weights')]
        self.first_moment_biases = [np.zeros_like(layer.biases) for layer in self.model.layers if hasattr(layer, 'biases')]

        self.second_moment_weights = [np.zeros_like(layer.weights) for layer in self.model.layers if hasattr(layer, 'weights')]
        self.second_moment_biases = [np.zeros_like(layer.biases) for layer in self.model.layers if hasattr(layer, 'biases')]    

        self.t = 0

    def step(self):
        self.t += 1
        idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):

                # update first moment
                self.first_moment_weights[idx] = self.beta1 * self.first_moment_weights[idx] + (1 - self.beta1) * layer.weight_grads
                self.first_moment_biases[idx] = self.beta1 * self.first_moment_biases[idx] + (1 - self.beta1) * layer.bias_grads

                # update second moment
                self.second_moment_weights[idx] = self.beta2 * self.second_moment_weights[idx] + (1 - self.beta2) * layer.weight_grads ** 2
                self.second_moment_biases[idx] = self.beta2 * self.second_moment_biases[idx] + (1 - self.beta2) * layer.bias_grads ** 2

                # update weights along with bias correction
                m_hat = self.first_moment_weights[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.second_moment_weights[idx] / (1 - self.beta2 ** self.t)
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # update biases along with bias correction
                m_hat = self.first_moment_biases[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.second_moment_biases[idx] / (1 - self.beta2 ** self.t)
                layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
