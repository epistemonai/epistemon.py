from epistemon import Tensor, Value
from epistemon.tensor import tensor_random
import random

class Activation(object):
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

class Sigmoid(Activation):
    def __call__(self, x: Tensor) -> Tensor: return Tensor([a.sigmoid() for a in x.arr])

class Linear(Activation):
    def __call__(self, x: Tensor) -> Tensor: return x

class SoftMax(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        den = Tensor([a.exp() for a in x.arr]).sum()
        out = Tensor([a.exp()/den for a in x.arr])
        return out

class Layer(object):
    def __init__(self, nin: int, neurons: int) -> None:
        self.nin     = nin
        self.neurons = neurons

    def forward(self):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, nin: int, neurons: int, activ: Activation) -> None:
        super().__init__(nin, neurons)
        self.activ   = activ
        self.bias    = Value(random.random())
        self.weights = tensor_random([self.nin, self.neurons])

    def forward(self, x: Tensor) -> Tensor:
        print(f"matmul: {x.shape} @ {self.weights.shape}")
        out = (x @ self.weights) + self.bias
        return self.activ(out)

class Loss(object):
    def __init__(self): pass
    def forward(self, P: Tensor, Y: Tensor) -> Value: raise NotImplementedError()


class MSE(Loss):
    def forward(self, P: Tensor, Y: Tensor) -> Value:
        vectors_match(P, Y)
        return ((P - Y)**2).mean()

class Model(object):
    def __init__(self, layers: list[Layer], loss: Loss, seed: float = 1):
        self.layers = layers
        self.loss   = loss
        self.seed   = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: Tensor) -> Tensor:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def train_batch(self, x_batch, y_batch) -> float:
        pass

class Optimizer(object):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self):
        pass


class Trainer(object):
    def __init__(self, model: Model, optim: Optimizer):
        self.model = model
        setattr(self.optim, 'model', self.model)

    def fit(self, X_train: Tensor, y_train:Tensor,
        X_test: Tensor, y_test: Tensor, epochs: int) -> None:
        pass