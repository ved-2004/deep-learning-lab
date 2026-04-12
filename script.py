from layers import FCLayer
from activations import ReLU
from mlp import MLP

# Build a tiny network: 3 -> 4 -> 2
model = MLP([
    FCLayer(3, 4),
    ReLU(),
    FCLayer(4, 2),
])

# Forward pass
x = [1.0, 2.0, 3.0]
output = model.forward(x)
print("Output:", output)

# Fake a simple loss: pretend target is [1, 0]
# dL/d_output for MSE = 2*(output - target) / n
target = [1.0, 0.0]
loss_grad = [2 * (o - t) / len(target) for o, t in zip(output, target)]
print("Loss grad:", loss_grad)

# Backward pass
model.backward(loss_grad)

# Check gradients got filled
for i, layer in enumerate(model.layers):
    if isinstance(layer, FCLayer):
        print(f"\nFCLayer {i}:")
        print("  weight_grads:", layer.weight_grads)
        print("  bias_grads:", layer.bias_grads)