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


class MLP:
    def __init__(self, layer_output_amount):
        self.layers = [
            Layer(layer_output_amount[i], layer_output_amount[i + 1])
            for i in range(len(layer_output_amount) - 1)
        ]

    def __call__(self, inputs):
        out = []
        for layer in self.layers:
            out = layer(inputs)
            inputs = [o.tanh() for o in out]

        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def reset_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def update(self, learning_rate):
        for p in self.parameters():
            p.data = p.data - learning_rate * p.grad


if __name__ == "__main__":
    x = [
        [Value(2.0), Value(3.0), Value(-1.0)],
        [Value(-2.0), Value(13.0), Value(2.0)],
        [Value(-4.0), Value(-20.0), Value(5.0)],
    ]

    target = [Value(1.0), Value(-1.0), Value(15)]

    mlp = MLP([3, 4, 4, 1])

    prediction = [mlp(v)[0] for v in x]

    loss = sum(
        [(tar - pred) * (tar - pred) for tar, pred in zip(target, prediction)],
        Value(0.0),
    )

    mlp.reset_grad()

    loss.backward()

    print("Loss vorher:", loss.data)

    for i in range(16000):
        prediction = [mlp(v)[0] for v in x]
        print(f"Prediction {i}:", [p.data for p in prediction])
        loss = sum(
            [(tar - pred) * (tar - pred) for tar, pred in zip(target, prediction)],
            Value(0.0),
        )

        mlp.reset_grad()
        loss.backward()
        mlp.update(0.0001)

        print("Loss:", loss.data)

        if round(loss.data, 2) == 0:
            break
