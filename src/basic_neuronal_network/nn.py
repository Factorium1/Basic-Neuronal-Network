import random

from engine import Value


class Neuron:
    def __init__(self, neuron_input_amount):
        self.weights = [
            Value(random.uniform(-1, 1)) for _ in range(neuron_input_amount)
        ]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs):
        out = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, neuron_input_amount, neuron_amount):
        self.neurons = [Neuron(neuron_input_amount) for _ in range(neuron_amount)]

    def __call__(self, inputs):
        return [n(inputs) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


if __name__ == "__main__":
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]

    out = n(x)
    print(out.data)
    print([p.data for p in n.parameters()])

    layer = Layer(3, 4)
    outputs = layer(x)

    print([out.data for out in outputs])
    print(len(layer.parameters()))
