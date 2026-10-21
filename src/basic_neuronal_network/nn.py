import random

from engine import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs):
        out = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        return out

    def parameters(self):
        return self.weights + [self.bias]


if __name__ == "__main__":
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]

    out = n(x)
    print(out.data)
    print([p.data for p in n.parameters()])
