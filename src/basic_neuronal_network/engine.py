class Value:
    def __init__(self, data, _prev=(), _operation=(), _backprop=()) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._operation = _operation
        self.backprop = _backprop

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


if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = a + b
    d = Value(7)
    e = d * c

    e.grad = 1.0
    e.backprop()
    print([v.data for v in e.build_topological_order()])
