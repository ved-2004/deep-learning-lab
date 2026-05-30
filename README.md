# deep-learning-lab

Building deep learning from scratch — no PyTorch, no autograd, no magic. Pure Python first, then NumPy, then we go further.

This repo follows a structured roadmap from MLP internals all the way to Transformer. Each project is a mini-paper: clean code, a write-up, and experiments with real observations.

---

## Project 1 — MLP from scratch

A multi-layer perceptron implemented from first principles. Every forward pass, every backward pass, every gradient update is written by hand.

### File layout

```
layers.py        — FCLayer: the core building block
activations.py   — ReLU, LeakyReLU, Sigmoid, Tanh
mlp.py           — MLP: chains layers together
losses.py        — MSELoss, CrossEntropyLoss
optimizers.py    — SGD, Momentum, Adam
script.py        — training loop on sklearn make_moons
```

---

### The building block — FCLayer

A fully connected layer connects every input neuron to every output neuron. It has two learnable parameters: a weight matrix `W` and a bias vector `b`.

```
weights[j][i]  →  a 2D list, shape [input_size][output_size]
                  weights[j][i] = weight from input j to output i

biases[i]      →  a 1D list, shape [output_size]
                  one bias per output neuron
```

**Forward pass:**

For each output neuron `i`, sum up contributions from every input neuron `j`:

```
output[i] = Σ( x[j] * weights[j][i] ) + biases[i]
```

The input `x` is cached as `self.x` — backward needs it.

**Backward pass:**

Three gradients get computed from `output_grads` (the gradient flowing in from the next layer):

```
weight_grads[j][i] = x[j] * output_grads[i]     ← how much each weight contributed to the loss
bias_grads[i]      = output_grads[i]              ← bias gradient is just the incoming gradient
input_grads[j]     = Σ( weights[j][i] * output_grads[i] )  ← passed back to the previous layer
```

`weight_grads` and `bias_grads` stay on the layer. `input_grads` gets returned so the previous layer can do its own backward.

---

### Activation functions

Each activation is its own class with `forward` and `backward`. No weights — just a transform and its derivative.

| Activation | Forward | Backward | Caches |
|------------|---------|----------|--------|
| ReLU | `max(0, x)` | `grad if x > 0 else 0` | input x |
| LeakyReLU | `x if x > 0 else α·x` | `grad if x > 0 else α·grad` | input x |
| Sigmoid | `1 / (1 + e^(-x))` | `grad · σ · (1 - σ)` | output σ(x) |
| Tanh | `tanh(x)` | `grad · (1 - tanh²)` | output tanh(x) |

ReLU and LeakyReLU cache the **input** — they need to check the sign of the original value. Sigmoid and Tanh cache the **output** — their derivatives are expressed in terms of the output directly. This distinction matters.

---

### How the MLP chains them

```python
model = MLP([
    FCLayer(2, 8),
    ReLU(),
    FCLayer(8, 8),
    ReLU(),
    FCLayer(8, 1),
    Sigmoid(),
])
```

Forward: loops through layers in order, passing output of each as input to the next.

Backward: loops in reverse, passing the gradient from each layer to the one before it.

---

### One full training step

```
1. output = model.forward(x)
         x → FCLayer → activation → FCLayer → activation → FCLayer → ŷ

2. loss = loss_fn.forward(y_true, output)
         L = (1/n) · Σ(ŷᵢ − yᵢ)²

3. loss_grad = loss_fn.backward(y_true, output)
         ∂L/∂ŷᵢ = (2/n) · (ŷᵢ − yᵢ)

4. model.backward(loss_grad)
         Each layer computes its own weight/bias grads
         and passes input_grads to the layer before it

5. optimizer.step()
         For SGD: W ← W − η · weight_grads
                  b ← b − η · bias_grads
```

The key insight: **parameters (weights, biases) get updated by the optimizer. Gradients (weight_grads, bias_grads) are computed by backward. The cache (self.x) is set during forward and read during backward.** These are three separate responsibilities.

---

### Data structures at a glance

```
FCLayer
├── params (set at init, updated by optimizer each step)
│   ├── self.weights       list[list[float]]  shape [in][out]
│   └── self.biases        list[float]        shape [out]
│
├── cache (set during forward, read during backward)
│   └── self.x             list[float]        the input that arrived
│
└── grads (set during backward, read by optimizer)
    ├── self.weight_grads  list[list[float]]  shape [in][out]
    └── self.bias_grads    list[float]        shape [out]

ReLU / LeakyReLU
└── self.x                 list[float]        pre-activation input values

Sigmoid / Tanh
└── self.x                 list[float]        post-activation output values (reused in backward)
```

---

### Optimizers

All optimizers loop through `model.layers`, skip layers without `weights`, and update `weights` and `biases` using the stored `weight_grads` and `bias_grads`.

**SGD:**
```
W ← W − η · dW
b ← b − η · db
```

**SGD + Momentum:**
```
v ← β · v + (1 − β) · dW
W ← W − η · v
```
Momentum needs a velocity buffer `v` (same shape as W) initialized to zero and persisted across steps.

**Adam:**
```
m ← β₁ · m + (1 − β₁) · dW          ← first moment (mean)
v ← β₂ · v + (1 − β₂) · dW²         ← second moment (variance)
m̂ = m / (1 − β₁ᵗ)                   ← bias correction
v̂ = v / (1 − β₂ᵗ)
W ← W − η · m̂ / (√v̂ + ε)
```
Adam needs two buffers per parameter (`m` and `v`) and a timestep counter `t`.

---

### Results on make_moons

2-layer network, 300 samples, 100 epochs, lr=0.1, SGD:

```
Epoch   0 | loss: 0.2520 | acc: 0.450
Epoch  10 | loss: 0.0939 | acc: 0.867
Epoch  30 | loss: 0.0507 | acc: 0.933
Epoch  80 | loss: 0.0277 | acc: 0.967
Epoch  99 | loss: 0.0353 | acc: 0.960
```

96% accuracy on noisy moons. All in pure Python, no NumPy, no autograd.

---

### What's next

- NumPy rewrite: same interface, matrix ops replace loops, adds batching
- Optimizer comparison: run SGD vs momentum vs Adam, log curves, write analysis
- Decision boundary visualization
- L2 regularization, dropout
- Move to Project 2: CNNs

---

## Roadmap

| Phase | Projects | Status |
|-------|----------|--------|
| 0 | Setup | ✓ |
| 1 | MLP, CNN, Autoencoders/VAE/GAN, Transfer learning, GNN | MLP in progress |
| 2 | RNN, LSTM, BiLSTM, GRU, Seq2Seq | — |
| 3 | Transformers, GPT from scratch, PEFT/LoRA | — |
| 4 | LLM eval harness, MT-Bench, safety evals, eval dashboard | — |

Goal: build a framework capable of training a 12M parameter transformer on custom data, with custom CUDA kernels, WebGPU fallback, and a TypeScript API — in the spirit of [this](https://mni-ml.github.io/demos/transformer/).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install scikit-learn matplotlib
python script.py
```

No deep learning libraries. That's the point.