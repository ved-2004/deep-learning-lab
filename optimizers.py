
class SGD:
    def __init__(self, model,learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue

            # update weights
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    layer.weights[i][j] -= self.learning_rate * layer.weight_grads[i][j]

            # update biases
            for i in range(layer.output_size):
                layer.biases[i] -= self.learning_rate * layer.bias_grads[i]

            