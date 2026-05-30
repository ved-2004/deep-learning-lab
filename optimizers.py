'''
This file contains the implementation of various optimizers. 
'''

# The SGD optimizer updates the weights and biases of the model based on the gradients computed during backpropagation. The learning rate determines how much to adjust the weights and biases in response to the computed gradients.
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

            
# Momentum optimizer is an extension of SGD that adds a momentum term to the weight updates. This helps accelerate convergence and can help escape local minima.
class Momentum:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize velocity for each layer
        self.velocities = {}
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue
            weight_velocity = [[0.0 for _ in range(layer.output_size)] for _ in range(layer.input_size)]
            bias_velocity = [0.0 for _ in range(layer.output_size)]
            self.velocities[id(layer)] = (weight_velocity, bias_velocity)


    def step(self):
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue

            weight_velocity, bias_velocity = self.velocities[id(layer)]

            # update velocity
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    weight_velocity[i][j] = self.momentum * weight_velocity[i][j] + (1 - self.momentum) * layer.weight_grads[i][j]

            for i in range(layer.output_size):
                bias_velocity[i] = self.momentum * bias_velocity[i] + (1 - self.momentum) * layer.bias_grads[i]
            
            # update weights
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    layer.weights[i][j] -= self.learning_rate * weight_velocity[i][j]

            # update bias 
            for i in range(layer.output_size):
                layer.biases[i] -= self.learning_rate * bias_velocity[i]

# Adagrad optimizer is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter based on the historical gradients for that parameter. This allows for larger updates for infrequent parameters and smaller updates for frequent parameters.
class Adagrad:
    def __init__(self, model, learning_rate=0.01, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Initialize cache for each layer
        self.caches = {}
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                weight_cache = [[0.0 for _ in range(layer.output_size)] for _ in range(layer.input_size)]
                bias_cache = [0.0 for _ in range(layer.output_size)]
                self.caches[id(layer)] = (weight_cache, bias_cache)

    def step(self):
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue
        
            weight_cache, bias_cache = self.caches[id(layer)]

            # update cache
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    weight_cache[i][j] = weight_cache[i][j] + (layer.weight_grads[i][j] ** 2)
            
            for i in range(layer.output_size):
                bias_cache[i] = bias_cache[i] + (layer.bias_grads[i] ** 2)

            # update weights
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    layer.weights[i][j] -= self.learning_rate * (layer.weight_grads[i][j] / (weight_cache[i][j] ** 0.5 + self.epsilon))

            # update bias
            for i in range(layer.output_size):
                layer.biases[i] -= self.learning_rate * (layer.bias_grads[i] / (bias_cache[i] ** 0.5 + self.epsilon))

# RMSProp optimizer is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter based on the average of recent squared gradients for that parameter. This helps to stabilize the training process and can lead to faster convergence.
class RMSProp:
    def __init__(self, model, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

        # Initialize cache for each layer
        self.caches = {}
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                weight_cache = [[0.0 for _ in range(layer.output_size)] for _ in range(layer.input_size)]
                bias_cache = [0.0 for _ in range(layer.output_size)]
                self.caches[id(layer)] = (weight_cache, bias_cache)

    def step(self):
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue
        
            weight_cache, bias_cache = self.caches[id(layer)]

            # update cache
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    weight_cache[i][j] = self.beta * weight_cache[i][j] + (1 - self.beta) * (layer.weight_grads[i][j] ** 2)
            
            for i in range(layer.output_size):
                bias_cache[i] = self.beta * bias_cache[i] + (1 - self.beta) * (layer.bias_grads[i] ** 2)

            # update weights
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    layer.weights[i][j] -= self.learning_rate * (layer.weight_grads[i][j] / (weight_cache[i][j] ** 0.5 + self.epsilon))

            # update bias
            for i in range(layer.output_size):
                layer.biases[i] -= self.learning_rate * (layer.bias_grads[i] / (bias_cache[i] ** 0.5 + self.epsilon))

# Adam optimizer is an adaptive learning rate optimization algorithm that combines the benefits of both Momentum and RMSProp optimizers. It computes adaptive learning rates for each parameter based on the first and second moments of the gradients.
class Adam:
    def __init__(self, model, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step

        # Initialize first moment for each layer
        self.first_moments = {}
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue
            weight_first_moment = [[0.0 for _ in range(layer.output_size)] for _ in range(layer.input_size)]
            bias_first_moment = [0.0 for _ in range(layer.output_size)]
            self.first_moments[id(layer)] = (weight_first_moment, bias_first_moment)

        # Initialize second moment for each layer
        self.second_moments = {}
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue
            weight_second_moment = [[0.0 for _ in range(layer.output_size)] for _ in range(layer.input_size)]
            bias_second_moment = [0.0 for _ in range(layer.output_size)]
            self.second_moments[id(layer)] = (weight_second_moment, bias_second_moment)

    
    def step(self):
        self.t += 1  # increment time step
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue

            weight_first_moment, bias_first_moment = self.first_moments[id(layer)]
            weight_second_moment, bias_second_moment = self.second_moments[id(layer)]

            # update first moment -- mean of gradients
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    weight_first_moment[i][j] = self.beta1 * weight_first_moment[i][j] + (1 - self.beta1) * layer.weight_grads[i][j]

            for i in range(layer.output_size):
                bias_first_moment[i] = self.beta1 * bias_first_moment[i] + (1 - self.beta1) * layer.bias_grads[i]

            # update second moment -- mean of squared gradients
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    weight_second_moment[i][j] = self.beta2 * weight_second_moment[i][j] + (1 - self.beta2) * (layer.weight_grads[i][j] ** 2)

            for i in range(layer.output_size):
                bias_second_moment[i] = self.beta2 * bias_second_moment[i] + (1 - self.beta2) * (layer.bias_grads[i] ** 2)

            # update weights along with bias correction
            for i in range(layer.input_size):
                for j in range(layer.output_size):
                    m_hat = weight_first_moment[i][j] / (1 - self.beta1 ** self.t)  # bias-corrected first moment
                    v_hat = weight_second_moment[i][j] / (1 - self.beta2 ** self.t)  # bias-corrected second moment
                    layer.weights[i][j] -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
            
            # update bias along with bias correction
            for i in range(layer.output_size):
                m_hat = bias_first_moment[i] / (1 - self.beta1 ** self.t)  # bias-corrected first moment
                v_hat = bias_second_moment[i] / (1 - self.beta2 ** self.t)  # bias-corrected second moment  
                layer.biases[i] -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
