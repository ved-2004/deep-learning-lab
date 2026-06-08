import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = 1 / (1 + np.exp(-x))
        return self.x
    
    def backward(self, output_grad):
        return output_grad * self.x * (1 - self.x)
    

class Tanh: 
    def __init__(self):
        pass

    def forward(self, x):
        self.x = np.tanh(x)
        return self.x
    
    def backward(self, output_grad):
        return output_grad * (1 - self.x ** 2)
    

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, output_grad):
        return output_grad * (self.x > 0)
    

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, output_grad):
        return output_grad * np.where(self.x > 0, 1, self.alpha)
    

class GeLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, output_grad):
        tanh_term = np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * self.x ** 3))
        return output_grad * (0.5 * (1 + tanh_term) + 0.5 * self.x * (1 - tanh_term ** 2) * (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x ** 2)))