"""
Defining various MLP classes, which are neural networks with varying hidden layers.
"""

class FCLayer:
    """
    A fully connected layer, which is a layer of neurons where each neuron is connected to every neuron in the previous layer.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[0.0 for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [0.0 for _ in range(output_size)]
        # grads, for backpropagation
        self.weight_grads = None
        self.bias_grads = None
        # cache, filled during forward pass, used during backpropagation
        self.x = None

    def forward(self, inputs):
        """
        Forward pass through the layer.
        """
        self.x = inputs
        output = [0.0 for _ in range(self.output_size)]
        for i in range(self.output_size):
            for j in range(self.input_size):
                output[i] += self.x[j] * self.weights[j][i]
            output[i] += self.biases[i]
        return output
    
    def backward(self, output_grads):
        """
        Backward pass through the layer, calculating gradients for weights and biases.
        """
        self.weight_grads = [[0.0 for _ in range(self.output_size)] for _ in range(self.input_size)]
        self.bias_grads = [0.0 for _ in range(self.output_size)]
        input_grads = [0.0 for _ in range(self.input_size)]
        
        for i in range(self.output_size):
            self.bias_grads[i] = output_grads[i]
            for j in range(self.input_size):
                self.weight_grads[j][i] = self.x[j] * output_grads[i]
                input_grads[j] += self.weights[j][i] * output_grads[i]
        
        return input_grads