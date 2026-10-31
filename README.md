# Basic Neuronal Network

A playground for practicing different neural network implementations.
Each exercise lives in its own numbered folder and can be run on its own.

## Exercises

| Folder | Description |
| --- | --- |
| [`01-micrograd-mlp`](01-micrograd-mlp/) | Scalar autograd engine and MLP, inspired by micrograd |

## Usage

All exercises share the dependencies in the root `pyproject.toml`.
Add new ones with `uv add <package>`.

Run an exercise from the repo root, e.g.:

`uv run 01-micrograd-mlp/nn.py`

## Adding a new exercise

1. Create a new folder with the next number, e.g. `02-numpy-mlp/`
2. Add a short `README.md` inside it
3. Add it to the table above
