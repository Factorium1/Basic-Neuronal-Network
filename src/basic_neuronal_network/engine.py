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


if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = a + b
    d = Value(7)
    e = d * a
    # print(c.data)
    # print([x.data for x in c._prev])
    e.grad = 1.0
    e.backprop()
    # print(e.grad)
    # print(a.grad)
    print(b.grad)
    print(c.grad)
