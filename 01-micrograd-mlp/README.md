# 01 – Micrograd MLP

~Inspired by micrograd

In `engine.py` are the fundamentals for each Value of the Network (autograd and backpropagation).
In `nn.py` lives the definition of a Neuron, Layer and MLP together with the training loop.

## Basic usage

Adapt the inputs in `__main__` in the var x and redefine the target Values in var target.
If needed change the layout of the Network via the mlp var.

Run from the repo root: `uv run 01-micrograd-mlp/nn.py`
