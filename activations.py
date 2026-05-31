import math

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = [1 / (1 + math.exp(-val)) for val in x]
        return self.x

    def backward(self, output_grad):
        return [grad * val * (1 - val) for grad, val in zip(output_grad, self.x)]


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = [math.tanh(val) for val in x]
        return self.x
    
    def backward(self, output_grad):
        return [grad * (1 - tanh_val ** 2) for grad, tanh_val in zip(output_grad, self.x)]


class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return [max(0, val) for val in x]

    def backward(self, output_grad):
        return [grad if val > 0 else 0 for grad, val in zip(output_grad, self.x)]


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return [val if val > 0 else self.alpha * val for val in x]

    def backward(self, output_grad):
        return [grad if val > 0 else self.alpha * grad for grad, val in zip(output_grad, self.x)]
    
class GeLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return [0.5 * val * (1 + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val ** 3))) for val in x]

    def backward(self, output_grad):
        return [grad * (0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val ** 3))) + 0.5 * val * (1 - math.tanh(math.sqrt(2 / math.pi) * (val + 0.044715 * val ** 3)) ** 2) * (math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * val ** 2))) for grad, val in zip(output_grad, self.x)]
    