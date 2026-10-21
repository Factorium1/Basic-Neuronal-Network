class Value:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = 0.0

    def __add__(self, other):
        out = Value(self.data + other.data)
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data)
        return out

    def __mul__(self, other):
        out = Value(self.data - other.data)
        return out


if __name__ == "__main__":
    a = Value(2)
    b = Value(3)
    c = a - b
    print(c.data)
