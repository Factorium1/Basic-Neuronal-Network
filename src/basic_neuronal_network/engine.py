class Value:
    def __init__(self, data, _prev=(), _operation=()) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = _prev
        self._operation = _operation

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out


if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = a - b
    print(c.data)
    print([x.data for x in c._prev])
