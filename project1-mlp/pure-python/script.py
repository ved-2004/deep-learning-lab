import random
from sklearn.datasets import make_moons

from layers import FCLayer
from activations import ReLU, Sigmoid
from mlp import MLP
from losses import MSELoss
from optimizers import SGD


# ─── 1. Get data ───────────────────────────────────────────────
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
# X is a numpy array of shape (300, 2), y is shape (300,) with values 0 or 1
# Convert to plain Python lists since our framework uses lists
X = X.tolist()                                  # list of 300 [x1, x2] pairs
y = [[float(label)] for label in y]             # list of 300 [0.0] or [1.0]


# ─── 2. Build model ────────────────────────────────────────────
# 2 inputs → 8 hidden → 8 hidden → 1 output (probability-ish)
model = MLP([
    FCLayer(2, 8),
    ReLU(),
    FCLayer(8, 8),
    ReLU(),
    FCLayer(8, 1),
    Sigmoid(),       # squash output to (0, 1) for binary classification
])


# ─── 3. Loss + optimizer ───────────────────────────────────────
loss_fn = MSELoss()
optimizer = SGD(model, learning_rate=0.1)


# ─── 4. Training loop ──────────────────────────────────────────
n_epochs = 100
n_samples = len(X)

for epoch in range(n_epochs):
    # Shuffle indices each epoch — important so the model doesn't memorize order
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    epoch_loss = 0.0
    correct = 0
    
    for idx in indices:
        x_sample = X[idx]           # [x1, x2]
        y_sample = y[idx]           # [0.0] or [1.0]
        
        # Forward
        pred = model.forward(x_sample)
        
        # Loss
        loss = loss_fn.forward(y_sample, pred)
        epoch_loss += loss
        
        # Track accuracy: prediction > 0.5 means class 1
        predicted_class = 1 if pred[0] > 0.5 else 0
        if predicted_class == int(y_sample[0]):
            correct += 1
        
        # Backward
        loss_grad = loss_fn.backward(y_sample, pred)
        model.backward(loss_grad)
        
        # Update parameters
        optimizer.step()
    
    avg_loss = epoch_loss / n_samples
    accuracy = correct / n_samples
    
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d} | loss: {avg_loss:.4f} | acc: {accuracy:.3f}")