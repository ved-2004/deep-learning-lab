from losses import CrossEntropyLoss
import numpy as np

loss_fn = CrossEntropyLoss()

# batch of 3 samples, 4 classes
logits = np.array([[2.0, 1.0, 0.5, 0.1],
                   [0.5, 2.5, 0.3, 0.1],
                   [0.1, 0.2, 3.0, 0.5]])

# one-hot targets
y_true = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

loss = loss_fn.forward(y_true, logits)
grad = loss_fn.backward(y_true, logits)

print("Loss:", loss)           # should be a small positive number
print("Grad shape:", grad.shape)  # should be (3, 4)
print("Grad:", grad)           # rows should sum to ~0