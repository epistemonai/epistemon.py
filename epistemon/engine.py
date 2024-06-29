import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data      = data
        self.grad      = 0.0
        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op
        self.label     = label

    def __repr__(self):
        return f"{self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        t   = (math.exp(2.0*self.data)-1.0)/(math.exp(2.0*self.data)+1.0)
        out = Value(t, (self,), "tanh")
        def backward():
            self.grad += (1.0 - tanh(out.data)**2) * out.grad

        out._backward = backward
        return out

    def sigmoid(self):
        ex = math.exp(-self.data)
        out = Value(1.0/(1.0 + ex), (self,), "sigmoid")
        def backward():
            self.grad += ex/(1.0 + ex)**2 * out.grad
        
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def backward():
            self.grad += out.data * out.grad
        
        out._backward = backward
        return out

    def backward(self):
        self.grad = 1.0
        topo      = []
        visited   = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()