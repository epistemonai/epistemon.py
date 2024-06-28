from epistemon.engine import Value

class Tensor:
    def __init__(self, arr):
        if not all(isinstance(x, (float,Value)) for x in arr): raise ValueError("Tensor can only initialize with either floats or Value")
        self.arr = [x if isinstance(x, Value) else Value(x) for x in arr]

    def __repr__(self):
        return f"Tensor={self.arr}"

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        if (len(self.arr) != len(other.arr)): raise ValueError("Tensors do not match!") 
        return Tensor([a*b for a, b in zip(self.arr, other.arr)])

    def backward(self):
        for i in self.arr: i.backward()

    def sum(self) -> Value:
        sum = Value(0.0)
        for x in self.arr: sum += x
        return sum