from epistemon import Tensor
from epistemon.math import MSE

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

c = MSE(a, b)
print(c)
c.backward()
b.data()