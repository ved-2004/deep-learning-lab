import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from layers import FCLayer
from activations import ReLU
from mlp import MLP
from losses import CrossEntropyLoss
from optimizers import Adam

# ─── 1. Load MNIST ───────────────────────────────────────────
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# normalize to [0, 1]
X = X / 255.0

# one-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 2. Build model ──────────────────────────────────────────
model = MLP([
    FCLayer(784, 128),
    ReLU(),
    FCLayer(128, 64),
    ReLU(),
    FCLayer(64, 10),   # 10 classes, raw logits — softmax inside loss
])

# ─── 3. Loss + optimizer ─────────────────────────────────────
loss_fn = CrossEntropyLoss()
optimizer = Adam(model, learning_rate=0.001)

# ─── 4. Training loop ────────────────────────────────────────
n_epochs = 20
batch_size = 64
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

for epoch in range(n_epochs):
    # shuffle each epoch
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0.0

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size

        X_batch = X_shuffled[start:end]   # (64, 784)
        y_batch = y_shuffled[start:end]   # (64, 10)

        # forward
        logits = model.forward(X_batch)

        # loss
        loss = loss_fn.forward(y_batch, logits)
        epoch_loss += loss

        # backward
        grad = loss_fn.backward(y_batch, logits)
        model.backward(grad)

        # update
        optimizer.step()

    avg_loss = epoch_loss / n_batches

    # ─── accuracy on test set ────────────────────────────────
    test_logits = model.forward(X_test)
    shifted = test_logits - np.max(test_logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(preds == true_labels)

    print(f"Epoch {epoch+1:2d}/{n_epochs} | loss: {avg_loss:.4f} | test acc: {accuracy:.3f}")