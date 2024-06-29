from epistemon.engine import Value
from epistemon.math import flatten_array, array_shape
import random
import numpy as np

def tensor_numpy(nparray) -> 'Tensor':
    tensor = Tensor([1])
    tensor.ndim = len(nparray.shape)
    tensor.shape = nparray.shape
    tensor.arr = nparray.reshape(-1).tolist()
    return tensor

def tensor_zeros(shape: list) -> 'Tensor':
    return Tensor([[0 for _ in range(shape[1])] for _ in range(shape[0])])

def tensor_random(shape: list) -> 'Tensor':
    if len(shape) == 2:
        return Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])])
    elif len(shape) == 1:
        return Tensor([random.random() for _ in range(shape[0])])

def vectors_match(A, B):
    if (A.shape[0] != B.shape[0]) and (A.ndim != 1 and B.ndim != 1): raise ValueError("Tensors do not match!")

class Tensor:
    def __init__(self, arr):
        self.shape = array_shape(arr)[::-1]
        self.ndim  = len(self.shape)
        arr = flatten_array(arr)
        if not all(isinstance(x, (int,float,Value)) for x in arr): raise ValueError("Tensor can only initialize with either floats, ints or Value")
        self.arr = [x if isinstance(x, Value) else Value(x) for x in arr]

    def _print(self, dim: int, index: int, grad: bool):
        if (index != 0): 
            print("            ", end='')
            for i in range(dim): print(" ", end='')
        print("[", end='')

        for i in range(self.shape[dim]):
            if (dim+1 >= self.ndim):
                val = self.arr[self._pindex].data if grad == False else self.arr[self._pindex].grad
                fmt = f"{val:.2f}, " if i+1 < self.shape[dim] else str(fmt)
                print(f"{fmt}", end='')
                self._pindex += 1
            else: 
                self._print(dim+1, i, grad)
            
        if   (dim-1 >= 0) and (index+1 == self.shape[dim-1]): print("]", end='')
        elif (dim == 1) and (self.ndim > 2): print("]", end='\n\n')
        else: print("]", end='\n')

    def __getitem__(self, index: tuple) -> Value:
        if isinstance(index, tuple): return self.arr[index[1]*self.shape[0] + index[0]]
        else: raise TypeError("Invalid index type. Must be int or tuple of ints.")

    def __setitem__(self, index: tuple, value: Value):
        if isinstance(index, tuple): self.arr[index[1]*self.shape[0] + index[0]] = value
        else: raise TypeError("Invalid index type. Must be int or tuple of ints.")

    def data(self):
        self._pindex=0
        print("Tensor<data>", end='')
        self.shape = self.shape[::-1] # lol
        self._print(0,0, grad=False)
        print("")
        self.shape = self.shape[::-1] # lol :(

    def __add__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            if (self.shape != other.shape): raise ValueError("Tensors do not match!") 
            return Tensor([a+b for a, b in zip(self.arr, other.arr)])
        else: 
            return Tensor([a+other for a in self.arr])


    def __sub__(self, other: 'Tensor') -> 'Tensor':
        if (self.shape != other.shape): raise ValueError("Tensors do not match!") 
        return Tensor([a-b for a, b in zip(self.arr, other.arr)])

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        if (self.shape != other.shape): raise ValueError("Tensors do not match!") 
        return Tensor([a*b for a, b in zip(self.arr, other.arr)])

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if other.ndim <= 1: raise ValueError("size mismatch")
        if (self.shape[0] != other.shape[1]): raise ValueError("Tensors do not match!")
        
        l = self.shape[1] if self.ndim > 1 else 1
        n = other.shape[0]
        m = self.shape[0]
        result = tensor_zeros([l,n])

        for i in range(l):
            for j in range(n):
                for k in range(m):
                    result[(j,i)] += self[(k,i)] * other[(j,k)]

        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        return Tensor([a**other for a in self.arr])
        return out

    def backward(self):
        for i in self.arr: i.backward()

    def grad(self):
        self._pindex=0
        print("Tensor<grad>", end='')
        self.shape = self.shape[::-1] # lol
        self._print(0,0, grad=True)
        print("")
        self.shape = self.shape[::-1] # lol :(


    def sum(self) -> Value:
        sum = Value(0.0)
        for x in self.arr: sum += x
        return sum

    def mean(self) -> Value:
        return self.sum()/len(self.arr)

    def abs(self) -> 'Tensor':
        return Tensor([math.abs(a) for a in self.arr])