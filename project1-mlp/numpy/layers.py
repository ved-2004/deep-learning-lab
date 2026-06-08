import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, 0.1, (input_size, output_size))
        self.biases = np.zeros(output_size)
        # grads, for backpropagation
        self.weight_grads = None
        self.bias_grads = None
        # cache, filled during forward pass, used during backpropagation
        self.x = None

    def forward(self, inputs):
        self.x = inputs
        output = np.dot(self.x, self.weights) + self.biases
        # other ways to do the above operation:
        # output = np.matmul(self.x, self.weights) + self.biases
        # output = self.x @ self.weights + self.biases
        # output = self.x.dot(self.weights) + self.biases
        return output
    
    def backward(self, output_grads):
        self.weight_grads = self.x @ output_grads
        self.bias_grads = np.sum(output_grads, axis=0)
        input_grads = output_grads @ self.weights.T
        return input_grads
    