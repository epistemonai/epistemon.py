from epistemon import Tensor
from epistemon import Value

a = Tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]])

b = Tensor(
    [[11, 12, 13],
     [14, 15, 16],
     [17, 18, 19]])

c = a @ b
c.data()
c.backward()

a.grad()
b.grad()