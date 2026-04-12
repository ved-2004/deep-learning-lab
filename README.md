# MLP From Scratch

Learning multi-layer perceptrons by building every piece by hand — no NumPy, no autograd, no shortcuts.

## Why

**Vibe coding is fast, but intuition is load-bearing.**

1. I vibe code a lot. That's fine until it isn't — debugging something subtle, writing something genuinely new, or catching AI slop requires the mental model that only comes from having built it yourself.
2. Translating math directly into code (without a library absorbing the complexity) is the most honest way to learn what's actually happening.
3. Both of those reasons compound: the deeper the intuition, the better the debugging, the faster you spot when generated code is wrong.

The goal isn't to replace PyTorch. It's to understand what PyTorch is doing well enough that using it feels grounded, not magical.

---

## What's Built

### `layers.py` — Fully Connected Layer

`FCLayer(input_size, output_size)`

- Forward: `output[i] = sum(x[j] * W[j][i]) + b[i]` — the raw dot product, written out as nested loops
- Backward: computes `weight_grads`, `bias_grads`, and `input_grads` from the upstream gradient
- Caches `x` during the forward pass so the backward pass can use it

### `activations.py` — Activation Functions

Each activation is a class with `forward` / `backward` methods, same interface as a layer so they can be stacked directly in an MLP.

| Class | Forward | Backward (derivative) |
|---|---|---|
| `Sigmoid` | `1 / (1 + e^-x)` | `σ(x) · (1 - σ(x))` |
| `Tanh` | `tanh(x)` | `1 - tanh(x)²` |
| `ReLU` | `max(0, x)` | `1 if x > 0 else 0` |
| `LeakyReLU` | `x if x > 0 else αx` | `1 if x > 0 else α` |

Uses only the standard library (`math`).

### `mlp.py` — MLP

`MLP(layers)` takes a list of layers and activations.

- `forward(x)` — pipes input through each layer in order
- `backward(output_grad)` — walks the list in reverse, passing gradients back through each layer

### `script.py` — Wiring It Together

End-to-end demo: build a `3 → 4 → 2` network, run a forward pass, compute an MSE loss gradient by hand, run the backward pass, and print the resulting weight/bias gradients.

```python
model = MLP([
    FCLayer(3, 4),
    ReLU(),
    FCLayer(4, 2),
])
```

---

## What's Next

- [ ] Weight initialization (Xavier / He) — zeros don't train
- [ ] `optimizers.py` — SGD, then momentum, then Adam
- [ ] Loss functions — MSE and cross-entropy as explicit classes with their own backward
- [ ] A real training loop on a toy dataset (spirals or moons)
- [ ] L2 regularization / dropout

---

## Running It

```bash
python script.py
```

No dependencies beyond the standard library for the core code. `requirements.txt` lists numpy, matplotlib, and scikit-learn for future dataset and visualization work.
