# MLP From Scratch

Building deep learning from the ground up — pure Python first, then NumPy, then PyTorch/JAX.

---

## Progress

### Phase 1 — Pure Python
- [x] Fully connected layer (`FCLayer`) — forward, backward, gradient caching
- [x] Activations — Sigmoid, Tanh, ReLU, LeakyReLU
- [x] MLP container — chains layers, reverses for backprop
- [x] SGD optimizer

### Phase 2 — NumPy
- [x] Loss functions — MSELoss, CrossEntropyLoss
- [ ] Rewrite layers, activations, MLP with NumPy
- [ ] Weight initialization — Xavier / He
- [ ] Training loop on a toy dataset (moons / spirals)
- [ ] Momentum + Adam

### Phase 3 — PyTorch / JAX
- [ ] Port to PyTorch (autograd, `nn.Module`)
- [ ] Port to JAX (`jit`, `grad`)

---

## Topics

| Concept | Covered |
|---|---|
| Forward pass (dot product, bias) | Yes |
| Backpropagation (chain rule) | Yes |
| Activation functions + derivatives | Yes |
| Gradient descent (SGD) | Yes |
| Loss functions (MSE, cross-entropy) | In progress |
| Weight initialization | No |
| Optimizers (momentum, Adam) | No |
| Regularization (L2, dropout) | No |
| Batching / mini-batch SGD | No |
| Softmax | No |

---

## Running

```bash
python script.py
```
