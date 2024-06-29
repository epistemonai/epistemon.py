from epistemon.tensor import tensor_random, tensor_numpy
from epistemon import Tensor
from epistemon.nn import Dense, Model, SoftMax, Sigmoid, MSE
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[:-2] + (-1,))
X_test = X_test.reshape(X_test.shape[:-2] + (-1,))

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = tensor_numpy(np.transpose(X_train))
y_train = tensor_numpy(y_train)
X_test  = tensor_numpy(X_test)
y_test  = tensor_numpy(y_test)

mnist = Model(
    layers=[
        Dense(nin=28*28, neurons=128, activ=Sigmoid()),
        Dense(nin=128, neurons=64, activ=Sigmoid()),
        Dense(nin=64, neurons=10, activ=SoftMax()),
    ],
    loss = MSE())

out = mnist.forward(X_train)
print(out.shape)

"""
optimizer = SGD(lr=0.01)
trainer = Trainer(mnist, optimizer)

trainer.fit(X_train, y_train, X_test, y_test, epochs=50)

print(x.shape)

dense0 = Dense(nin=10, neurons=10, activ=SoftMax())

out = dense0.forward(x)
out.backward()

print(dense0.weights.data())

"""