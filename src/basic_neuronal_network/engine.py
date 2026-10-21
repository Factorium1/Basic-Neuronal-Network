class Value:
    def __init__(self, data, _prev=(), _operation=(), _backprop=lambda: None) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._operation = _operation
        self._backprop = _backprop

    def __add__(self, other):

        def _backprop():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out = Value(self.data + other.data, (self, other), "+", _backprop)
        return out

    def __sub__(self, other):

        def _backprop():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad

        out = Value(self.data - other.data, (self, other), "-", _backprop)
        return out

    def __mul__(self, other):

        def _backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out = Value(self.data * other.data, (self, other), "*", _backprop)
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
    a = Value(2)
    b = Value(3)
    c = a * b
    d = c + a

    d.backward()

    print(a.grad)  # 4.0
    print(b.grad)  # 2.0
    print(c.grad)  # 1.0
    print(d.grad)  # 1.0
