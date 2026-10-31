import math


class Value:
    def __init__(self, data, _prev=(), _operation=(), _backprop=lambda: None) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._operation = _operation  # Just relevant for debugging not for prod
        self._backprop = _backprop

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        def _backprop():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out = Value(self.data + other.data, (self, other), "+", _backprop)
        return out

    def __radd__(self, other):
        return self + Value(other)

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        def _backprop():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad

        out = Value(self.data - other.data, (self, other), "-", _backprop)
        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        def _backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out = Value(self.data * other.data, (self, other), "*", _backprop)
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):

        def _backprop():
            self.grad += (1 - out.data**2) * out.grad

        out = Value(math.tanh(self.data), (self,), "tanh", _backprop)
        return out

    def build_topological_order(self):
        topological_order = []
        visited = set()

        def build(curr):
            if curr in visited:
                return

            visited.add(curr)

            for prev in curr._prev:
                build(prev)

            topological_order.append(curr)

        build(self)
        return topological_order

    def backward(self):
        self.grad = 1.0
        order = self.build_topological_order()

        for x in reversed(order):
            x._backprop()


if __name__ == "__main__":
    a = Value(0.5)
    b = a.tanh()
    b.backward()

    print(b.data)
    print(a.grad)
